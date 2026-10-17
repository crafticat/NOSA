"""Track B, milestone 8 (INTERFACE_BC8 I18, frozen 2026-10-17): a BOUNDED, REUSABLE producer workspace. Stdlib and torch
only; this module never imports dispatcher.py. SOURCE AND PORTABLE COMPONENT READINESS ONLY.

What it is. ONE ProducerWorkspace per executor. For each producer call (a prefill, a payload) the executor acquires ONE
storage and writes the call's outputs into views of it, in place of fresh tensors. A storage class is (kind, numel):
the call kind and the number of elements; dtype and device are the workspace's own.

The bound (mode "reuse"; fix round MINOR-1). The classes are DECLARED at construction: classes = {(kind, numel):
count}. The workspace allocates ALL declared storages at construction and never allocates more. The total bound is
fixed before the first call: bound()["storages"] = the sum of the counts, bound()["bytes"] = the sum of count x numel
x the dtype's element size; storage_count() and storage_bytes() are equal to them for the whole life of the workspace.
An acquire of a class that is not declared (another prompt length, for example) is refused by name BEFORE any change
(CapacityRefusal "producer workspace: class <kind>/<numel> is not declared (declared <kind>/<numel>, ...) (I18)"), so
the number of distinct classes can never grow the total. There is no fresh fallback.

The overwrite rule (I16 / I18; the executor.py I16 note). acquire hands out a storage only when it is FREE (never
handed out) or REUSABLE. REUSABLE means:
  (a) the store operation that read the storage since its last hand-out has a DELIVERED handback (on_handback), the
      handback is issued, and its reuse event is complete. The test is the predicate of TensorStore.source_reusable
      (issued, reuse_ev not None, ev_done(reuse_ev)) on the delivered event. A record that detach_completed released
      (its event set to None) was complete when it was released.
  (b) no store operation that reads the storage is without a delivered handback: TensorStore.source_ops(sid), READ-ONLY
      and from the store's own records, is (0, 0) (no OPEN hold names the storage, and no recorded handback that names
      it is still untaken). This sees an operation that did not end yet, and an operation whose handback the lead did
      not forward yet: the lead forwards after a COMPLETE admission batch and after a COMPLETE layer call, so the
      workspace tolerates that timing without a hook.
ONE store operation per hand-out (the contract of the caller; fix round MINOR-3). A SECOND delivered handback in the
same generation makes the storage DEAD (stats["second_op"]; fail closed). The rule cannot see an operation that has
not started: a caller that gives one generation to a later operation breaks the contract. The workspace detects it at
that operation's delivery ONLY when no acquire handed the storage out again before that operation started. The
detection can be ONE operation late: when an acquire happens in the window (the first operation delivered and
complete, the late operation not started yet), the storage starts a new generation; the old generation's late
operation is then the FIRST delivery of the new generation, and only the NEXT operation's delivery makes the storage
DEAD. By then an overwrite may already have happened (the store's byte detectors, check_ring_entry and
check_host_block, would catch it). WorkspaceConsumer cannot reach this path: each of its producer calls gives its
storage to exactly one store operation.
A storage that was handed out but that no store operation read is NOT reusable (fail closed; "unread"). A delivered
handback with issued False (a failed or partially issued operation) makes its storage DEAD: never handed out again (a
leak by design, never an early overwrite; stats["failed"]). The workspace keeps no second hold ledger, no scheduler and
no wait.

The refusal. When no storage of a declared class is FREE or REUSABLE, acquire raises CapacityRefusal "producer
workspace: no storage of class <kind>/<numel> may be overwritten (capacity <c>: <p> pending, <u> undelivered, <r>
unread, <d> dead) (I18)" BEFORE any change: no hand-out, no write, no event change; only stats["refusals"] counts it.
It never waits.

Sizing on the CpuBackend (fix round MINOR-2). An admission's reuse event is its join gate (tensor_store
close_admission), and in a dispatcher run nothing forces the join stream before the final synchronize. So inside a run
a prefill storage is NOT reused, and its class needs one storage per admission of the run. A fixture sizes the
classes from a "fresh" baseline of the same trace (the test fixtures' ws_capacity reads peak_live) or forces the gates.

The mode "fresh" (the MATCHED BASELINE only; NOT bounded; it takes no classes). Every acquire allocates a NEW storage
of any class and nothing is reused. The same tracking runs: at each acquire, the storages that became reusable are
dropped (spent), and peak_live[class] records the largest number of storages that were not reusable, the new one
included. On the same trace, that number is the smallest count that a "reuse" run needs for that class (the set of
storages that are not reusable is the same in both modes at the same point of the trace).

I10 (the executor's reset contract; executor.py). events is a PLAIN dict {(kind, numel, index, op): the delivered reuse
event, or None once released}. The events are OPERATIONAL: a later acquire tests them. holders() lists every entry that
is not None. detach_completed() refuses with no change (OwnershipViolation "producer workspace: detach refused: <n>
pending reuse events: [...] (I18)") while one is pending; else it sets each to None (released) and returns the count.
An executor lists them under its own I10 contract (owner class "executor:workspace") and may set an entry to None
through its own location rule: the effect is the same. The events of a storage leave the dict when the storage is
handed out again (they were complete) or becomes DEAD (no decision reads them).

Instrumentation (not operational): stats, peak_live, and each storage's generation (its hand-out count).

Not claimed: CUDA stream order (the cpupack_core CpuBackend model only); NOSA numerics, HBM fit or throughput.
Inherited limit: an exception inside the lead's forwarding loop (after take_handbacks, before on_handback) loses the
rest of that take; the run aborts there, and the workspace cannot see those operations afterwards.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .backend import ev_done
from .errors import CapacityRefusal, ConfigError, KStepError, OwnershipViolation
from .tensor_store import Handback, storage_id

MODES = ("reuse", "fresh")
REUSE, FRESH = MODES
FREE, OUT, DEAD, SPENT = "free", "out", "dead", "spent"   # storage states (SPENT: fresh mode only, dropped)
REASONS = ("pending", "undelivered", "unread", "dead")     # why a storage that is not FREE is not reusable


class _Storage:
    """One storage of a class: the 1-D tensor t (the whole storage), its id sid (tensor_store.storage_id), its bytes,
    its state, its generation (the number of hand-outs) and ops (the op ids of the delivered handbacks of the current
    generation: at most one, else the storage is DEAD)."""
    __slots__ = ("kind", "numel", "idx", "t", "sid", "nbytes", "state", "gen", "ops")

    def __init__(self, kind: str, numel: int, idx: int, t: torch.Tensor):
        self.kind, self.numel, self.idx, self.t = kind, numel, idx, t
        self.sid = storage_id(t)
        self.nbytes = t.untyped_storage().nbytes()
        self.state, self.gen, self.ops = FREE, 0, []


def _class_name(key: Tuple[str, int]) -> str:
    return "%s/%d" % key


class ProducerWorkspace:
    """I18: the bounded reusable producer workspace (module docstring).
      ProducerWorkspace(classes, dtype, device="cpu", mode="reuse")
                       mode "reuse": classes = {(kind, numel >= 1): count >= 1}, every storage allocated now; mode
                       "fresh": classes None (not bounded). ConfigError for a malformed classes, dtype or mode.
      attach_store(store)
                       the TensorStore whose source_ops the rule reads (OwnershipViolation for a second, other store).
      acquire(kind, numel) -> (tensor, (kind, numel, index, generation))
                       the 1-D storage (dtype, numel) that the caller may overwrite now; CapacityRefusal for a class
                       that is not declared and when no storage of the class may be overwritten (module docstring);
                       KStepError before attach_store and for a size < 1.
      on_handback(hb)  a delivered tensor_store.Handback: per source storage of this workspace, record the op and its
                       reuse event (issued, the first of the generation), or make the storage DEAD (not issued, or a
                       second operation of the generation). Returns the storages it named. A source that is not a
                       storage of this workspace is counted (foreign) and ignored.
      reason(kind, numel, index)
                       READ-ONLY: None (FREE or reusable now) or one of REASONS.
      holders(), detach_completed()
                       I10 (module docstring).
      bound(), storage_count(), storage_bytes(), report()
                       READ-ONLY accounting."""

    def __init__(self, classes: Optional[Dict[Tuple[str, int], int]], dtype: torch.dtype, device: Any = "cpu",
                 mode: str = REUSE):
        if mode not in MODES:
            raise ConfigError("producer workspace: mode %r is not one of %s (I18)" % (mode, MODES))
        if not isinstance(dtype, torch.dtype):
            raise ConfigError("producer workspace: dtype %r is not a torch dtype (I18)" % (dtype,))
        if mode == FRESH:
            if classes is not None:
                raise ConfigError("producer workspace: mode 'fresh' is not bounded and takes no classes, not %r "
                                  "(I18)" % (classes,))
        else:
            bad = (not isinstance(classes, dict) or not classes
                   or any(not (isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str)
                               and isinstance(k[1], int) and not isinstance(k[1], bool) and k[1] >= 1)
                          or isinstance(v, bool) or not isinstance(v, int) or v < 1
                          for k, v in classes.items()))
            if bad:
                raise ConfigError("producer workspace: classes must map each (kind, numel >= 1) to a count >= 1, not "
                                  "%r (I18)" % (classes,))
        self.declared: Dict[Tuple[str, int], int] = dict(classes or {})
        self.dtype, self.device, self.mode = dtype, torch.device(device), mode
        self.store: Any = None
        self.classes: Dict[Tuple[str, int], Dict[int, _Storage]] = {}
        self.by_sid: Dict[Tuple, _Storage] = {}
        self.events: Dict[Tuple, Any] = {}             # I10: OPERATIONAL (module docstring); a PLAIN dict
        self.peak_live: Dict[Tuple[str, int], int] = {}   # fresh mode: instrumentation (the measured need)
        self._next: Dict[Tuple[str, int], int] = {}       # fresh mode: the next index of a class
        self.stats = dict(allocations=0, handouts=0, reuses=0, refusals=0, undeclared=0, delivered=0, failed=0,
                          second_op=0, foreign=0, late_dead=0, retired=0, spent=0)
        for key, count in self.declared.items():       # mode "reuse": the whole bound, now
            self.classes[key] = {}
            for i in range(count):
                self._add(key, i)

    # -- wiring
    def attach_store(self, store: Any) -> None:
        if self.store is not None and self.store is not store:
            raise OwnershipViolation("producer workspace: attach_store of a second store (I18)")
        if not callable(getattr(store, "source_ops", None)):
            raise ConfigError("producer workspace: the store %s has no source_ops (I18)" % (type(store).__name__,))
        self.store = store

    # -- helpers
    @staticmethod
    def _key(s: _Storage, op: int) -> Tuple[str, int, int, int]:
        return (s.kind, s.numel, s.idx, op)

    @staticmethod
    def _name(s: _Storage) -> str:
        return "%s/%d[%d]" % (s.kind, s.numel, s.idx)

    def _add(self, key: Tuple[str, int], idx: int) -> _Storage:
        t = torch.zeros((key[1],), dtype=self.dtype, device=self.device)
        s = _Storage(key[0], key[1], idx, t)
        self.classes[key][idx] = s
        self.by_sid[s.sid] = s
        self.stats["allocations"] += 1
        return s

    def _why(self, s: _Storage) -> Optional[str]:
        """READ-ONLY: None when s may be overwritten now (FREE, or the rule (a) and (b) hold), else the reason."""
        if s.state == FREE:
            return None
        if s.state == DEAD:
            return "dead"
        n_open, untaken = self.store.source_ops(s.sid)
        if n_open or untaken:
            return "undelivered"                       # (b): an operation without a delivered handback
        if not s.ops:
            return "unread"                            # fail closed: no store operation read it
        for op in s.ops:
            ev = self.events.get(self._key(s, op))
            if ev is not None and not ev_done(ev):
                return "pending"                       # (a): the reuse event is not complete
        return None

    def _retire(self, s: _Storage) -> None:
        """The events of the current generation leave the dict (they were complete, or the storage is DEAD)."""
        for op in s.ops:
            self.events.pop(self._key(s, op), None)
        self.stats["retired"] += len(s.ops)
        s.ops = []

    def _dead(self, s: _Storage, counter: str) -> None:
        self._retire(s)
        s.state = DEAD
        self.stats[counter] += 1

    def _hand_out(self, s: _Storage) -> Tuple[torch.Tensor, Tuple[str, int, int, int]]:
        if s.state == OUT:
            self._retire(s)
            self.stats["reuses"] += 1
        s.state = OUT
        s.gen += 1
        self.stats["handouts"] += 1
        return s.t, (s.kind, s.numel, s.idx, s.gen)

    # -- the bounded acquisition
    def acquire(self, kind: str, numel: int) -> Tuple[torch.Tensor, Tuple[str, int, int, int]]:
        if self.store is None:
            raise KStepError("producer workspace: acquire of kind %r before attach_store (I18)" % (kind,))
        if isinstance(numel, bool) or not isinstance(numel, int) or numel < 1:
            raise KStepError("producer workspace: class %s needs a size >= 1, not %r (I18)" % (kind, numel))
        key = (kind, numel)
        if self.mode == FRESH:
            return self._acquire_fresh(key)
        pool = self.classes.get(key)
        if pool is None:                               # MINOR-1: a class that is not declared, BEFORE any change
            self.stats["undeclared"] += 1
            raise CapacityRefusal("producer workspace: class %s/%s is not declared (declared %s) (I18)"
                                  % (kind, numel, ", ".join(sorted(_class_name(k) for k in self.declared))))
        why = dict.fromkeys(REASONS, 0)
        for s in pool.values():                        # index order: the lowest FREE or reusable storage
            r = self._why(s)
            if r is None:
                return self._hand_out(s)
            why[r] += 1
        self.stats["refusals"] += 1                    # BEFORE any change: nothing handed out, nothing written
        raise CapacityRefusal("producer workspace: no storage of class %s/%d may be overwritten (capacity %d: %d "
                              "pending, %d undelivered, %d unread, %d dead) (I18)"
                              % (kind, numel, len(pool), why["pending"], why["undelivered"], why["unread"],
                                 why["dead"]))

    def _acquire_fresh(self, key: Tuple[str, int]) -> Tuple[torch.Tensor, Tuple[str, int, int, int]]:
        """Mode "fresh": the reusable storages are dropped (spent; nothing reuses them), then ONE new storage; the
        storages left (not reusable) plus the new one give peak_live."""
        pool = self.classes.setdefault(key, {})
        for idx in [i for i, s in pool.items() if self._why(s) is None]:
            s = pool.pop(idx)
            self._retire(s)
            s.state = SPENT
            self.by_sid.pop(s.sid, None)
            s.t = None
            self.stats["spent"] += 1
        idx = self._next.get(key, 0)
        self._next[key] = idx + 1
        s = self._add(key, idx)
        self.peak_live[key] = max(self.peak_live.get(key, 0), len(pool))
        return self._hand_out(s)

    # -- delivery
    def on_handback(self, hb: Any) -> int:
        if not isinstance(hb, Handback):
            raise KStepError("producer workspace: on_handback of a %s, not a Handback (I18)" % (type(hb).__name__,))
        n = 0
        for sid in hb.sources:
            s = self.by_sid.get(sid)
            if s is None:
                self.stats["foreign"] += 1             # not a storage of this workspace
                continue
            if s.state == DEAD:
                self.stats["late_dead"] += 1           # another operation of a DEAD generation: stays DEAD
                continue
            if s.state != OUT:
                raise OwnershipViolation("producer workspace: handback op %s names storage %s, which is %s, not handed "
                                         "out (I18)" % (hb.op, self._name(s), s.state))
            n += 1
            if not (hb.issued is True and hb.reuse_ev is not None):
                self._dead(s, "failed")                # a failed or partially issued operation: never reused
                continue
            if s.ops:
                self._dead(s, "second_op")             # MINOR-3: one store operation per hand-out, else fail closed
                continue
            s.ops.append(hb.op)
            self.events[self._key(s, hb.op)] = hb.reuse_ev
            self.stats["delivered"] += 1
        return n

    # -- read-only queries
    def _get(self, kind: str, numel: int, idx: int) -> _Storage:
        s = self.classes.get((kind, numel), {}).get(idx)
        if s is None:
            raise KStepError("producer workspace: no storage %s/%s[%s] (I18)" % (kind, numel, idx))
        return s

    def reason(self, kind: str, numel: int, idx: int) -> Optional[str]:
        return self._why(self._get(kind, numel, idx))

    def tensor(self, kind: str, numel: int, idx: int) -> torch.Tensor:
        return self._get(kind, numel, idx).t

    def generation(self, kind: str, numel: int, idx: int) -> int:
        return self._get(kind, numel, idx).gen

    def state(self, kind: str, numel: int, idx: int) -> str:
        return self._get(kind, numel, idx).state

    def bound(self) -> Dict[str, int]:
        """READ-ONLY, mode "reuse": the total bound fixed at construction: storages (the sum of the declared counts)
        and bytes (the sum of count x numel x the dtype's element size). Mode "fresh": not bounded (both None)."""
        if self.mode == FRESH:
            return dict(storages=None, bytes=None)
        el = torch.empty((), dtype=self.dtype).element_size()
        return dict(storages=sum(self.declared.values()),
                    bytes=sum(c * k[1] * el for k, c in self.declared.items()))

    def storage_count(self) -> int:
        return sum(len(pool) for pool in self.classes.values())

    def storage_bytes(self) -> int:
        return sum(s.nbytes for pool in self.classes.values() for s in pool.values())

    def report(self) -> Dict[str, Any]:
        """READ-ONLY: mode, declared classes, bound, per class (as "kind/numel") the storages and how many are free,
        reusable, pending, undelivered, unread or dead, and the largest generation; the totals, stats, peak_live and
        the number of listed events."""
        cls: Dict[str, Dict[str, int]] = {}
        for key, pool in self.classes.items():
            c = dict(storages=len(pool), free=0, reusable=0, max_gen=0, **dict.fromkeys(REASONS, 0))
            for s in pool.values():
                r = self._why(s)
                if s.state == FREE:
                    c["free"] += 1
                elif r is None:
                    c["reusable"] += 1
                else:
                    c[r] += 1
                c["max_gen"] = max(c["max_gen"], s.gen)
            cls[_class_name(key)] = c
        return dict(mode=self.mode, declared={_class_name(k): v for k, v in self.declared.items()}, bound=self.bound(),
                    classes=cls, storages=self.storage_count(), bytes=self.storage_bytes(), stats=dict(self.stats),
                    peak_live={_class_name(k): v for k, v in self.peak_live.items()}, events=len(self.holders()))

    # -- I10
    def holders(self) -> List[Tuple[Tuple, Any]]:
        """I10, READ-ONLY: every (key, reuse event) of the events dict that is not None, in insertion order."""
        return [(key, ev) for key, ev in self.events.items() if ev is not None]

    def detach_completed(self) -> int:
        """I10, MUTATING: OwnershipViolation "producer workspace: detach refused: <n> pending reuse events: [...]
        (I18)" with NO change while a listed event is pending; else every listed event is set to None (released: the
        delivered handback stays, without an event) and the count is returned."""
        held = self.holders()
        pend = [key for key, ev in held if not ev_done(ev)]
        if pend:
            raise OwnershipViolation("producer workspace: detach refused: %d pending reuse events: %s (I18)"
                                     % (len(pend), pend[:6]))
        for key, _ in held:
            self.events[key] = None
        return len(held)
