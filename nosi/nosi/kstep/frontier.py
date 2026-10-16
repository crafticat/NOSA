"""Track C of K-step M2: JOINED reader frontiers and DURABLE finished-cohort records. Opt-in: Config.reader_frontier =
"joined". The default "single" never imports this module, so it loads, runs and reports exactly as before.

Sources (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md):
  3.1    the B1 restriction to ONE ordered "compute" stream. Its last paragraph: reads on several independent streams
         need a JOINED completion frontier per position, never "the event assigned most recently".
  11.3   the ordered-stream gates of a ring entry (the AND of the compute gate and the D2H gate).
  6 m1   rel_layers is pruned when every layer of a key is released; a late _queue_release / _protect then sees no
         record. 12.7: no asynchronous decision callback before that obligation is resolved.
  B / C interface (lead, 2026-10-16): I1 (the completion handle) and I3 (request and cohort identity).

SCOPE. This prepares multi-stream execution. It makes no real CUDA run and no broad scheduling rewrite. It changes no
schedule rule, guard, wait or decision. It makes no general-liveness claim. The CPU evidence is the deferred CpuBackend
model only.

1. COMPLETION HANDLE (I1). A handle is the native object that be.event_rec(stream) returns: a CpuBackend _Ev
   (benchmarks/Efficiency/cpupack_core.py:354-358, polled through .done) or a CudaBackend torch.cuda.Event (:467-470,
   polled through .query()). backend.ev_done polls either one and never waits. join() makes ONE new native event from
   several handles: for each input that is not None and not complete, be.stream_wait(JOIN_STREAM, e); then
   be.event_rec(JOIN_STREAM). One input (live or complete) comes back unchanged. So the one-stream case installs exactly
   the event that it installs today. No list, tuple or other Python object stands for a set of events: check_native
   refuses it by name. No join calls host_wait or synchronize.

2. THE JOIN STREAM. It is a DEDICATED stream "join". It carries only the waits and the records of the joins.
   Why not a working stream:
   - CpuBackend.stream_wait (:401-402) appends a wait to the named stream. _run (:413-424) executes a stream in order,
     and a wait runs the stream of the waited event up to that event (:426-429). CudaBackend.stream_wait (:480-481) is
     wait_event on that stream. So a wait on a WORKING stream ("compute", "scatter", "d2h") holds EVERY LATER operation
     of that stream until the input completes. On "compute", all later compute work would wait for a reader on another
     stream. On the CpuBackend, a forced later compute event would also run the other reader stream (an effect of the
     same kind as D3 in contract 11.4). That false dependency is too wide.
   The false dependency that "join" adds (stated, not removed):
   - "join" is in order. So a join record completes only after EVERY EARLIER join record on "join". A slow input of an
     earlier, unrelated join delays a later join whose own inputs are complete. This is conservative: it can delay a
     FREE, a release or a retirement. It never makes one early.
   - On the CpuBackend, a host wait on a later join runs "join" from its head. So it also completes the inputs of every
     earlier join (it can complete more work than the object needs).
   The CPU model:
   - The CpuBackend runs nothing until a host wait, a downstream stream wait or synchronize forces it (:367-370).
     ev_done never runs a stream. So a join is not complete at a poll until something runs "join", also when all its
     inputs are complete. The existing forcing paths of the dispatcher run it: releasing_gates, the key reader waits of a
     verifier-only pass, a transport's T3 wait on reader_ev, and finish (synchronize runs every stream, :431-438). A GPU
     executes the record without help. So a multi-stream CPU run can complete later than a one-stream run. No schedule
     parity is claimed for multi-stream runs.
   - The backend must own the stream (check_join_stream): the CpuBackend takes its names at construction (:372-376),
     the CudaBackend takes a dict of streams (:445-450).

3. READER FRONTIER (FrontierTable). Per object, the table keeps the LATEST reader event of EACH stream that read the
   object. The objects: a slot generation (its arena binding shares the slot's handle: a binding lives exactly as long as
   one slot generation, ownership.py take / poll_releases), a (cohort key, layer), and a ring entry generation.
   A stream is in order, so its latest record covers every earlier read on it (the B1 argument of contract 3.1, applied
   per stream). The installed handle is the join of these per-stream events.
   Requirements:
   - The notes of one stream arrive in record order (the dispatcher installs a call's event right after the call,
     contract 3.1).
   - Each note names the stream that recorded the event. A CpuBackend _Ev names its stream (.stream.name), so a wrong
     name is refused. A torch.cuda.Event does not, so on CUDA the caller's name is trusted (stated, not checked).
   A complete component adds nothing and is dropped. prune() drops an object whose components are all complete; it
   never waits. A small cache returns the same join for the same set of live inputs. That is exact: a join waits exactly
   its inputs.

4. FINISHED COHORTS (FinishedCohorts; m1, I3). The identity (uid, inc, key) of a decided cohort stays recorded from its
   decision until BOTH:
   - its request incarnation (uid, inc) ended, and
   - every hold source reports nothing outstanding for (uid, inc): the reader frontiers of that incarnation (built in),
     and the holds that the lead adds with add_source (track B's I2 holds).
   While the record exists, also after the dispatcher pruned rel_layers:
   - a late protection of the key is refused (R1.5);
   - a late release-queue request is a counted no-op;
   - a second release is a named double release (R1.4);
   - a duplicate decision is a counted no-op.
   A reused uid gets a new inc and never matches an old record. A call that names an incarnation that is not live is
   refused (I3).
   Milestone 2 (addendum I5, I6; "# HOOK-C2" in dispatcher.py):
   - Hold providers (CompletionOwnership._bind, at the first admission): "reader_frontier", "transfers" (in-flight
     transfers of the uid whose transport does not report them complete), "ring_gates" (ring entries owned by the uid
     with an incomplete ready / compute / D2H gate, and its admission buffer A_r with an incomplete gate), "d2h" (the
     request's incomplete hist_ev events and its D2H snapshot records with an incomplete event) and, only when the
     dispatcher holds a TensorStore as Dispatcher.store (track B's opt-in mode; detected by duck typing, tensor_store
     is never imported here), "tensor_store" = store.holds.outstanding_count. Records clear only when EVERY provider
     reports 0. Every provider is non-blocking (ev_done / transport done only).
   - Decision entry: Dispatcher._apply, _finish_draft and _end_request first ask CompletionOwnership, BEFORE any
     mutation of accepted drafts, progress, publications, tokens or results. A duplicate decision of a finished key, and
     a draft completion of an old epoch, are counted no-ops. A decision, draft completion or end that names an ended or
     cleared incarnation, a Request object that is not the dispatcher's, or a cohort that is not the DONE head of the
     current epoch is refused by name.
   - No uid-reuse promise beyond this: Dispatcher.submit refuses a uid that it knows, so a uid has one incarnation
     per dispatcher today; the providers attribute work by uid (conservative if a uid ever had two incarnations).

5. ACCOUNTING. The entries are counted as the dispatcher counts its release-queue and reader-event entries (one
   Geometry.unit()["release_entry"] each, ownership.py:875): per frontier object one record plus one per component, one
   per cached join, one per finished-cohort record and one per live incarnation. report() gives the entries, the bytes
   and their peaks. A peak is taken after every operation that can add an entry (a note, a start, a decision), so it is
   the maximum over those operations, not over the steps inside one operation. They are NOT added to the K6 sample: the
   test-side Auditor recounts K6 exactly (retroinfer-eval tests/kstep_toy.py Auditor._k6), so that integration is the
   lead's decision.

6. HOOKS. dispatcher.py calls this module only through CompletionOwnership, at the lines marked "# HOOK-C" and
   "# HOOK-C2".

7. EVENT LIFETIME (addendum I4). FrontierTable keys its objects' components and its join cache by the IDENTITY of
   native event objects (comp[ident][stream] is ev; cache key frozenset(id(e))). The cpupack_core CudaBackend takes its
   events from a pool, and reset() (:452-453) makes later event_rec calls re-record pooled objects. So these identities
   are valid only between two resets. The rule: CompletionOwnership.before_reset(d) must run before any backend reset.
   It refuses (OwnershipViolation "reader frontier: reset refused (I4): ...") while any reader component, installed
   handle or cached join (output or input) is incomplete, or any hold provider of a live incarnation reports a count.
   Otherwise it clears the identity-keyed state (clear_caches). It never waits, forces or drops a reference to make a
   reset possible. backend.py and cpupack_core are unchanged: the caller of reset must call before_reset first.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .backend import ev_done, record_join
from .errors import ConfigError, KStepError, OwnershipViolation
from .ownership import FREE

JOIN_STREAM = "join"
FRONTIER_MODES = ("single", "joined")


# ------------------------------------------------------------------------------------------------- I1: the handle
def check_join_stream(be, name: str = JOIN_STREAM) -> None:
    """The backend must own the stream `name` for the joins (CpuBackend names, cpupack_core.py:372-376; CudaBackend
    streams, :445-450). ConfigError otherwise: a join never falls back to a working stream."""
    s = getattr(be, "s", None)
    if not isinstance(s, dict) or name not in s:
        raise ConfigError("reader frontier: the backend has no stream %r for joins (it carries only the waits and the "
                          "records of the joins; add it to the CpuBackend names or the CudaBackend streams)" % (name,))


def native_event_types(be) -> Tuple[type, ...]:
    """The native handle type of backend `be`: torch.cuda.Event for the cpupack_core CudaBackend (:441-493), the _Ev class
    of the backend's own module for the CpuBackend (:354-358). ConfigError for any other backend (no silent fallback)."""
    if type(be).__name__ == "CudaBackend":
        return (torch.cuda.Event,)
    cls = getattr(sys.modules.get(type(be).__module__), "_Ev", None)
    if isinstance(cls, type):
        return (cls,)
    raise ConfigError("reader frontier: backend %s has no known native event type (the cpupack_core CpuBackend _Ev or "
                      "the CudaBackend torch.cuda.Event)" % (type(be).__name__,))


def check_native(ev, types: Tuple[type, ...], what: str) -> None:
    """TypeError unless `ev` is EXACTLY a native handle type (a subclass or a wrapper is refused too)."""
    if type(ev) not in types:
        raise TypeError("reader frontier: %s is a %s, not a native backend event (%s); a Python object that stands for "
                        "several events is never a completion handle (I1)"
                        % (what, type(ev).__name__, ", ".join(t.__name__ for t in types)))


def recorded_stream(ev) -> Optional[str]:
    """The name of the stream that recorded `ev` when the event shows it (CpuBackend _Ev.stream is the _Stream; a marker
    has None). None for a torch.cuda.Event, which does not show its stream."""
    return getattr(getattr(ev, "stream", None), "name", None)


def join(be, evs: Iterable[Any], stream: str = JOIN_STREAM, types: Optional[Tuple[type, ...]] = None,
         stats: Optional[Dict[str, int]] = None, cache: Optional[Dict] = None):
    """I1: ONE native handle that completes only after every input.
    - None inputs and repeated objects are skipped. Every other input must be native (check_native).
    - No input: None (None counts as complete). One input: that input, unchanged (live or complete).
    - Two or more inputs, at most one live: the live one unchanged; none live: the LAST input, unchanged (complete).
    - Two or more live inputs: be.stream_wait(stream, e) for each live input, in input order, then be.event_rec(stream).
      With a cache, the same set of live inputs returns the join recorded for it earlier (exact: it waits exactly them).
    No host wait and no synchronize. The callers pass the newest input last."""
    types = types if types is not None else native_event_types(be)
    cand: List[Any] = []
    for e in evs:
        if e is None or any(e is x for x in cand):
            continue
        check_native(e, types, "a join input")
        cand.append(e)
    if len(cand) <= 1:
        return cand[0] if cand else None
    live = [e for e in cand if not ev_done(e)]
    if len(live) <= 1:
        return live[0] if live else cand[-1]
    k = frozenset(id(e) for e in live)          # the cache holds the inputs, so these ids are not reused meanwhile
    if cache is not None and k in cache:
        if stats is not None:
            stats["join_cache_hits"] = stats.get("join_cache_hits", 0) + 1
        return cache[k][0]
    out = record_join(be, stream, live)          # lead integration: the one shared primitive (backend.py)
    check_native(out, types, "a join output")
    if cache is not None:
        cache[k] = (out, tuple(live))
    if stats is not None:
        stats["joins"] = stats.get("joins", 0) + 1
        stats["join_waits"] = stats.get("join_waits", 0) + len(live)
    return out


# ------------------------------------------------------------------------------------------------- reader frontier
class FrontierTable:
    """Section 3 of the module docstring. comp[ident] = {stream: the latest reader event of that stream};
    handle[ident] = the native handle installed for the object; owner[ident] = the request incarnation (uid, inc) whose
    reads these are (I3). stats counts notes, duplicate notes (the same event noted again for the same stream), joins,
    their waits, cache hits and pruned objects.
    LIFETIME (I4): comp, handle and cache hold native event OBJECTS and compare them by identity. They are valid only
    between two backend resets: drained() / clear_caches() below, called by CompletionOwnership.before_reset."""

    def __init__(self, be, join_stream: str = JOIN_STREAM, entry_bytes: int = 16):
        check_join_stream(be, join_stream)
        self.be, self.join_stream = be, join_stream
        self.types = native_event_types(be)
        self.entry_bytes = int(entry_bytes)
        self.comp: Dict[Any, Dict[str, Any]] = {}
        self.handle: Dict[Any, Any] = {}
        self.owner: Dict[Any, Any] = {}
        self.cache: Dict[frozenset, Tuple[Any, Tuple]] = {}
        self.stats: Dict[str, int] = dict(notes=0, duplicate_notes=0, joins=0, join_waits=0, join_cache_hits=0, pruned=0)
        self.n_entries = 0
        self.peak_entries = 0

    def note(self, ident, stream: str, ev, owner=None):
        """A read of object `ident` on stream `stream` completes at the native event `ev`. Returns the handle to install:
        the join of the latest event of every stream that read the object and is not complete yet."""
        if type(ev) not in self.types:                          # the message text only on a failure
            check_native(ev, self.types, "the reader event of %r" % (ident,))
        got = recorded_stream(ev)
        if got is not None and got != stream:
            raise OwnershipViolation("reader frontier: the reader event of %r was recorded on stream %r but is noted as "
                                     "stream %r (the per-stream latest rule needs the recording stream)"
                                     % (ident, got, stream))
        comp = self.comp.get(ident)
        if comp is None:
            comp = self.comp[ident] = {}
            self.owner[ident] = owner
            self.n_entries += 1
        elif self.owner[ident] != owner:
            raise OwnershipViolation("reader frontier: %r is read for request incarnation %r, but its frontier belongs to "
                                     "%r (I3)" % (ident, owner, self.owner[ident]))
        if comp.get(stream) is ev:
            self.stats["duplicate_notes"] += 1                  # the same completion again: the same handle, no join
            return self.handle[ident]
        if stream not in comp:
            self.n_entries += 1
        comp[stream] = ev                                       # an ordered stream: its newest record covers the rest
        for s in [s for s, e in comp.items() if s != stream and ev_done(e)]:
            del comp[s]                                         # a complete component adds nothing
            self.n_entries -= 1
        n0 = len(self.cache)
        h = join(self.be, [e for s, e in comp.items() if s != stream] + [ev], self.join_stream, self.types, self.stats,
                 self.cache)
        self.n_entries += len(self.cache) - n0
        self.handle[ident] = h
        self.stats["notes"] += 1
        self.peak_entries = max(self.peak_entries, self.n_entries)
        return h

    def outstanding(self, owner) -> List[Any]:
        """The components that are not complete yet, of every object that belongs to `owner` (non-blocking)."""
        return [e for i, c in self.comp.items() if self.owner[i] == owner for e in c.values() if not ev_done(e)]

    def prune(self) -> int:
        """Drops the complete components, then every object without a component, then every cached join whose output
        or inputs are all complete. Non-blocking (ev_done only). Returns the number of objects dropped."""
        gone = 0
        for i in list(self.comp):
            c = self.comp[i]
            for s in [s for s, e in c.items() if ev_done(e)]:
                del c[s]
                self.n_entries -= 1
            if not c:
                del self.comp[i], self.handle[i], self.owner[i]
                self.n_entries -= 1
                gone += 1
        for k in [k for k, (out, ins) in self.cache.items() if ev_done(out) or all(ev_done(e) for e in ins)]:
            del self.cache[k]
            self.n_entries -= 1
        self.stats["pruned"] += gone
        return gone

    def entries(self) -> int:
        return self.n_entries

    def pending(self) -> Dict[str, int]:
        """I4, non-blocking: the incomplete reader components, the incomplete installed handles and the cached joins whose
        output or any input is incomplete."""
        return dict(components=sum(1 for c in self.comp.values() for e in c.values() if not ev_done(e)),
                    handles=sum(1 for h in self.handle.values() if not ev_done(h)),
                    joins=sum(1 for out, ins in self.cache.values()
                              if not ev_done(out) or not all(ev_done(e) for e in ins)))

    def drained(self) -> bool:
        """I4: no component, installed handle or cached join of the table is incomplete."""
        return not any(self.pending().values())

    def clear_caches(self) -> None:
        """I4: drops every identity-keyed object, component, handle and cached join. Only after drained()."""
        if not self.drained():
            raise OwnershipViolation("reader frontier: clear_caches with incomplete events %s (I4)" % (self.pending(),))
        self.comp.clear()
        self.handle.clear()
        self.owner.clear()
        self.cache.clear()
        self.n_entries = 0
        self.stats["cleared"] = self.stats.get("cleared", 0) + 1

    def report(self) -> Dict[str, Any]:
        return dict(join_stream=self.join_stream, objects=len(self.comp), cached_joins=len(self.cache),
                    entries=self.n_entries, bytes=self.n_entries * self.entry_bytes, peak_entries=self.peak_entries,
                    peak_bytes=self.peak_entries * self.entry_bytes, stats=dict(self.stats))


# ------------------------------------------------------------------------------------------------- finished cohorts
class FinishedCohorts:
    """Section 4 of the module docstring. live[(uid, inc)] = "active" | "ended"; rec[(uid, inc, key)] = the decision
    serial; sources = [(name, fn(uid, inc) -> number of outstanding holds)]. Every check runs before any change."""
    ACTIVE, ENDED = "active", "ended"

    def __init__(self, entry_bytes: int = 16):
        self.entry_bytes = int(entry_bytes)
        self.live: Dict[Tuple[Any, Any], str] = {}
        self.rec: Dict[Tuple[Any, Any, Tuple], int] = {}
        self.sources: List[Tuple[str, Callable[[Any, Any], int]]] = []
        self.stats: Dict[str, int] = dict(started=0, ended=0, finishes=0, duplicate_finish=0, late_queue_ignored=0,
                                          late_protect_refused=0, late_release_refused=0, records_cleared=0,
                                          incarnations_cleared=0, duplicate_decisions_ignored=0,
                                          stale_drafts_ignored=0, stale_callbacks_refused=0)
        self.peak_entries = 0

    def add_source(self, name: str, fn: Callable[[Any, Any], int]) -> None:
        """A hold source: fn(uid, inc) returns how many holds of that incarnation are outstanding (it must not block)."""
        if any(n == name for n, _ in self.sources):
            raise ConfigError("finished cohorts: hold source %r is registered twice" % (name,))
        self.sources.append((name, fn))

    def entries(self) -> int:
        return len(self.rec) + len(self.live)

    def _peak(self) -> None:
        self.peak_entries = max(self.peak_entries, self.entries())

    def state(self, uid, inc, what: Callable[[], str]) -> str:
        """The state of the incarnation; KStepError (I3) when it is not live. `what` makes the message text, only on a
        failure (the hooks call this on every protection)."""
        st = self.live.get((uid, inc))
        if st is None:
            raise KStepError("I3: %s names request incarnation (%s, %s), which is not live (never started, or ended and "
                             "cleared)" % (what(), uid, inc))
        return st

    @staticmethod
    def _own(uid, key) -> None:
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == uid):
            raise KStepError("I3: cohort key %s does not belong to request %s" % (key, uid))

    def start(self, uid, inc) -> None:
        if (uid, inc) in self.live:
            raise KStepError("I3: request incarnation (%s, %s) started twice" % (uid, inc))
        other = [i for (u, i), st in self.live.items() if u == uid and st == self.ACTIVE]
        if other:
            raise KStepError("I3: uid %s is active as incarnation %s; incarnation %s cannot start" % (uid, other[0], inc))
        self.live[(uid, inc)] = self.ACTIVE
        self.stats["started"] += 1
        self._peak()

    def check_finish(self, uid, inc, key) -> bool:
        """The checks of a decision, before any change: the key belongs to uid and the incarnation is live. Returns True
        for a DUPLICATE decision (counted; the caller changes nothing). A first decision after the end is refused."""
        self._own(uid, key)
        st = self.state(uid, inc, lambda: "the decision of cohort %s" % (key,))
        if (uid, inc, key) in self.rec:
            self.stats["duplicate_finish"] += 1
            return True
        if st != self.ACTIVE:
            raise KStepError("I3: decision of cohort %s of request incarnation (%s, %s) after its end" % (key, uid, inc))
        return False

    def check_decision(self, uid, inc, key) -> bool:
        """Milestone 2 (I6), the ENTRY of a decision (Dispatcher._apply), before any mutation: the key belongs to uid; the
        incarnation must be live (else I3 "... which is not live ...") and ACTIVE (else I3 "decision ... after its end",
        also for a finished key); a finished key returns True (a duplicate decision: counted, the caller changes
        nothing). Unlike check_finish, the incarnation state is checked BEFORE the duplicate test."""
        self._own(uid, key)
        try:
            st = self.state(uid, inc, lambda: "the decision of cohort %s" % (key,))
            if st != self.ACTIVE:
                raise KStepError("I3: decision of cohort %s of request incarnation (%s, %s) after its end"
                                 % (key, uid, inc))
        except KStepError:
            self.stats["stale_callbacks_refused"] += 1
            raise
        if (uid, inc, key) in self.rec:
            self.stats["duplicate_decisions_ignored"] += 1
            return True
        return False

    def finish(self, uid, inc, key) -> bool:
        """Records the decision of `key`. Returns False for a duplicate decision (nothing changes)."""
        if self.check_finish(uid, inc, key):
            return False
        self.rec[(uid, inc, key)] = self.stats["finishes"]
        self.stats["finishes"] += 1
        self._peak()
        return True

    def finished(self, uid, inc, key) -> bool:
        return (uid, inc, key) in self.rec

    def refuse_protect(self, uid, inc, key, l: int, block: int) -> None:
        """A protection of `block` at layer l for `key`: refused for a finished key (R1.5) and for an incarnation that
        ended or is not live (I3)."""
        self._own(uid, key)
        st = self.state(uid, inc, lambda: "the protection of block %d at layer %d for cohort %s" % (block, l, key))
        if (uid, inc, key) in self.rec:
            self.stats["late_protect_refused"] += 1
            raise KStepError("R1.5: late protection of block %d at layer %d for cohort %s after the cohort finished "
                             "(durable finished-cohort record, incarnation %s)" % (block, l, key, inc))
        if st != self.ACTIVE:
            raise KStepError("I3: protection of block %d at layer %d for cohort %s of request incarnation (%s, %s) after "
                             "its end" % (block, l, key, uid, inc))

    def late_queue(self, uid, inc, key, l: int) -> bool:
        """A release-queue request for `key` at layer l that finds no rel_layers record: True (a counted no-op) when the
        key finished (every layer was queued at the decision). The incarnation must be live."""
        self._own(uid, key)
        self.state(uid, inc, lambda: "the release queue of cohort %s at layer %d" % (key, l))
        if (uid, inc, key) in self.rec:
            self.stats["late_queue_ignored"] += 1
            return True
        return False

    def refuse_release(self, uid, inc, key, l: int) -> None:
        """A release of `key` at layer l that finds no rel_layers record: a finished key was released before (R1.4)."""
        if (uid, inc, key) in self.rec:
            self.stats["late_release_refused"] += 1
            raise OwnershipViolation("R1.4: double release of cohort %s at layer %d (durable finished-cohort record, "
                                     "incarnation %s; its layer records were pruned)" % (key, l, inc))

    def end(self, uid, inc) -> int:
        """The request incarnation ended. Its records stay until every hold source reports nothing (poll)."""
        if self.state(uid, inc, lambda: "the end of a request") != self.ACTIVE:
            raise KStepError("I3: request incarnation (%s, %s) ended twice" % (uid, inc))
        self.live[(uid, inc)] = self.ENDED
        self.stats["ended"] += 1
        return self.poll()

    def outstanding(self, uid, inc) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, fn in self.sources:                  # lead integration: a COUNT >= 0, never a bool (a drained-
            n = fn(uid, inc)                           # predicate passed by mistake would invert the clearing rule)
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise TypeError("finished cohorts: hold source %r returned %r, not a count >= 0" % (name, n))
            out[name] = n
        return out

    def poll(self) -> int:
        """Clears every ENDED incarnation whose hold sources all report 0 (records and incarnation). Non-blocking.
        Returns the number of records cleared."""
        n = 0
        for uid, inc in [k for k, st in self.live.items() if st == self.ENDED]:
            if any(self.outstanding(uid, inc).values()):
                continue
            mine = [k for k in self.rec if k[0] == uid and k[1] == inc]
            for k in mine:
                del self.rec[k]
            del self.live[(uid, inc)]
            n += len(mine)
            self.stats["incarnations_cleared"] += 1
        self.stats["records_cleared"] += n
        return n

    def report(self) -> Dict[str, Any]:
        return dict(records=len(self.rec), live=len(self.live), entries=self.entries(),
                    bytes=self.entries() * self.entry_bytes, peak_entries=self.peak_entries,
                    peak_bytes=self.peak_entries * self.entry_bytes, sources=[n for n, _ in self.sources],
                    stats=dict(self.stats))


# ------------------------------------------------------------------------------------------------- dispatcher side
class CompletionOwnership:
    """The dispatcher's object in "joined" mode (Dispatcher.frontier): one FrontierTable, one FinishedCohorts whose
    built-in hold source is the table's outstanding reader events of the incarnation, and the HOOK-C entry points. A
    callback names its incarnation through the dispatcher's Request (d.reqs[uid].inc): the dispatcher refuses a second
    submit of a uid today (Dispatcher.submit), so a later asynchronous caller must pass (uid, inc) to FinishedCohorts
    itself.

    Scope, milestone 2 (addendum I5; supersedes the lead's milestone-1 note that only the reader frontiers were
    consulted): _bind registers the providers "transfers", "ring_gates", "d2h" and, when Dispatcher.store has
    holds.outstanding_count, "tensor_store", at the first admission (the dispatcher's own state is complete then).
    A record clears only when EVERY provider reports 0 for its incarnation. Mutation catchers run in "single" mode: in
    "joined" mode the frontier's R1.5 check still fires under no_late_protection_guard."""

    ACCEPTED, CANCELLED, DONE = "accepted", "cancelled", "done"    # dispatcher.py's cohort states (no import cycle)

    def __init__(self, be, join_stream: str = JOIN_STREAM, entry_bytes: int = 16):
        self.table = FrontierTable(be, join_stream, entry_bytes)
        self.cohorts = FinishedCohorts(entry_bytes)
        self.cohorts.add_source("reader_frontier", lambda uid, inc: len(self.table.outstanding((uid, inc))))
        self.peak_entries = 0
        self.bound = False
        self.resets = 0

    # ---------------------------------------------------------------------------------- I5: the hold providers
    def _bind(self, d) -> None:
        """Registers the dispatcher's hold providers once (I5). "tensor_store" only when d.store has
        holds.outstanding_count (track B's opt-in store; duck-typed, tensor_store.py is not imported)."""
        if self.bound:
            return
        self.bound = True
        add = self.cohorts.add_source
        add("transfers", lambda uid, inc: self.transfers_pending(d, uid, inc))
        add("ring_gates", lambda uid, inc: self.ring_gates_pending(d, uid, inc))
        add("d2h", lambda uid, inc: self.d2h_pending(d, uid, inc))
        fn = getattr(getattr(getattr(d, "store", None), "holds", None), "outstanding_count", None)
        if callable(fn):
            add("tensor_store", fn)

    @staticmethod
    def transfers_pending(d, uid, inc) -> int:
        """In-flight transfers of uid that the transport does not report complete (tp.done: non-blocking). A complete
        one is only waiting for the next host poll, which admits or drops it (T3); the arrival path reads no record."""
        return sum(1 for tr in d.inflight.values() if tr.uid == uid and not d.tp.done(tr))

    @staticmethod
    def ring_gates_pending(d, uid, inc) -> int:
        """Ring entries owned by uid (any non-FREE state) with an incomplete ready, compute or D2H gate
        (RingPool.gates), plus uid's admission buffer A_r with an incomplete ready, compute or D2H event."""
        n = 0
        for ring in d.rings:
            for e in range(ring.Q):
                ow = ring.owner[e]
                if ring.state[e] != FREE and ow is not None and ow[0] == uid and ring.gates(e):
                    n += 1
        for A in d.adm:
            if A["uid"] == uid and not all(ev_done(A[k]) for k in ("ready", "cev", "dev")):
                n += 1
        return n

    @staticmethod
    def d2h_pending(d, uid, inc) -> int:
        """The request's incomplete hist_ev events (when d.reqs[uid] is incarnation inc), plus uid's checked-mode D2H
        snapshot records whose hist_ev or snapshot event is incomplete."""
        n = 0
        r = d.reqs.get(uid)
        if r is not None and getattr(r, "inc", None) == inc:
            n += sum(1 for ev in r.hist_ev.values() if not ev_done(ev))
        n += sum(1 for rec in d.d2h_checks
                 if rec["uid"] == uid and not (ev_done(rec["hist_ev"]) and ev_done(rec["snap_ev"])))
        return n

    # ---------------------------------------------------------------------------------- I6: the decision entry
    def _live_request(self, d, r, what: Callable[[], str]) -> None:
        """r must be the dispatcher's Request object of its uid (a stale object of an old incarnation is refused)."""
        if d.reqs.get(r.uid) is not r:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise KStepError("I3: %s names a Request object of uid %s (incarnation %s) that is not the dispatcher's"
                             % (what(), r.uid, getattr(r, "inc", None)))

    def decision_entry(self, d, r, c) -> bool:
        """Dispatcher._apply's entry, BEFORE any mutation. True: a duplicate decision of a finished key (a counted no-op).
        Refused (raises): an ended or cleared incarnation (I3), a stale Request object (I3), or a cohort that is not the
        DONE head of the request's current epoch (OwnershipViolation "decision entry: ...")."""
        self._live_request(d, r, lambda: "the decision of cohort %s" % (c.key,))
        if self.cohorts.check_decision(r.uid, r.inc, c.key):
            return True
        head = next((cc for cc in r.cohorts if cc.state not in (self.ACCEPTED, self.CANCELLED)), None)
        if c is not head or c.state != self.DONE or c.epoch != r.epoch:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise OwnershipViolation("decision entry: cohort %s of request incarnation (%s, %s) is not the DONE head of "
                                     "epoch %d (state %s, head %s)" % (c.key, r.uid, r.inc, r.epoch, c.state,
                                                                       None if head is None else head.key))
        return False

    def draft_entry(self, d, r, dr) -> bool:
        """Dispatcher._finish_draft's entry, BEFORE any mutation. True: a draft completion of an old epoch (a counted
        no-op; _end_of_step skips those already). Refused (raises, I3): a stale Request object, or an incarnation that
        ended or is not live."""
        self._live_request(d, r, lambda: "the draft completion s%d" % (dr.s,))
        try:
            st = self.cohorts.state(r.uid, r.inc, lambda: "the draft completion s%d (epoch %d)" % (dr.s, dr.epoch))
            if st != FinishedCohorts.ACTIVE:
                raise KStepError("I3: draft completion s%d (epoch %d) of request incarnation (%s, %s) after its end"
                                 % (dr.s, dr.epoch, r.uid, r.inc))
        except KStepError:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise
        if dr.epoch != r.epoch:
            self.cohorts.stats["stale_drafts_ignored"] += 1
            return True
        return False

    def end_entry(self, d, r) -> None:
        """Dispatcher._end_request's entry, BEFORE any mutation: refused (I3) for a stale Request object, an incarnation
        that is not live, or a second end."""
        self._live_request(d, r, lambda: "the end of a request")
        try:
            if self.cohorts.state(r.uid, r.inc, lambda: "the end of a request") != FinishedCohorts.ACTIVE:
                raise KStepError("I3: request incarnation (%s, %s) ended twice" % (r.uid, r.inc))
        except KStepError:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise

    # ---------------------------------------------------------------------------------- I4: event lifetime
    def reset_pending(self) -> Dict[str, Any]:
        """Non-blocking: the table's incomplete components, handles and cached joins, and per live incarnation its
        NONZERO hold providers."""
        out: Dict[str, Any] = dict(self.table.pending())
        holds = {}
        for uid, inc in list(self.cohorts.live):
            o = {k: v for k, v in self.cohorts.outstanding(uid, inc).items() if v}
            if o:
                holds[(uid, inc)] = o
        out["holds"] = holds
        return out

    def drained(self) -> bool:
        p = self.reset_pending()
        return not (p["components"] or p["handles"] or p["joins"] or p["holds"])

    def clear_caches(self) -> None:
        self.table.clear_caches()

    def before_reset(self, d=None) -> None:
        """I4: call before any backend reset. Refuses (OwnershipViolation) while anything is outstanding; otherwise
        clears the identity-keyed caches. Never waits, forces or drops a reference."""
        p = self.reset_pending()
        if p["components"] or p["handles"] or p["joins"] or p["holds"]:
            raise OwnershipViolation("reader frontier: reset refused (I4): %d incomplete reader components, %d incomplete "
                                     "installed handles, %d incomplete cached joins, outstanding holds %s"
                                     % (p["components"], p["handles"], p["joins"], p["holds"]))
        self.clear_caches()
        self.resets += 1

    @staticmethod
    def incarnation(d, uid) -> Tuple[Any, Any]:
        r = d.reqs.get(uid)
        return uid, (None if r is None else getattr(r, "inc", None))

    def entries(self) -> int:
        return self.table.entries() + self.cohorts.entries()

    def _peak(self) -> None:
        self.peak_entries = max(self.peak_entries, self.entries())

    def install(self, d, l: int, read_slots, ev, stream: str) -> None:
        """Dispatcher._install_reader in joined mode: the same objects in the same order with the same first-reader
        rule; each object receives the joined handle of its per-stream latest readers instead of `ev`. In arena mode the
        position of a slot receives the slot's handle (Arena.note_reader)."""
        t = self.table
        for l_, b_, h_, idx in read_slots:
            s = d.pools[l_].stream(b_, h_).slots[idx]
            h = t.note(("slot", l_, b_, h_, idx, s.gen), stream, ev, self.incarnation(d, s.owner))
            self._peak()
            if s.first_reader is None:
                s.first_reader = h
            s.last_reader = h
            if d.arena_mode:
                d.arenas[l_].note_reader(s.phys, s.pgen, h)
        for key in d._read_keys:
            kr = d.key_readers.setdefault((key, l), [None, None])
            h = t.note(("key", key, l), stream, ev, self.incarnation(d, key[0]))
            self._peak()
            if kr[0] is None:
                kr[0] = h
            kr[1] = h
        for ll, e, g in d._ring_reads:
            ow = d.rings[ll].owner[e]
            h = t.note(("ring", ll, e, g), stream, ev, self.incarnation(d, None if ow is None else ow[0]))
            self._peak()
            d.rings[ll].note_compute(e, g, h)

    def start_request(self, d, r) -> None:
        self._bind(d)                                           # milestone 2: the hold providers (I5), once
        self.cohorts.start(r.uid, r.inc)
        self._peak()

    def end_request(self, d, r) -> None:
        self.cohorts.end(r.uid, r.inc)

    def begin_finish(self, d, key) -> bool:
        uid, inc = self.incarnation(d, key[0])
        return self.cohorts.check_finish(uid, inc, key)

    def record_finish(self, d, key) -> None:
        uid, inc = self.incarnation(d, key[0])
        self.cohorts.finish(uid, inc, key)
        self._peak()

    def late_queue(self, d, key, l: int) -> bool:
        uid, inc = self.incarnation(d, key[0])
        return self.cohorts.late_queue(uid, inc, key, l)

    def refuse_late_protect(self, d, key, l: int, block: int) -> None:
        uid, inc = self.incarnation(d, key[0])
        self.cohorts.refuse_protect(uid, inc, key, l, block)

    def refuse_late_release(self, d, key, l: int) -> None:
        uid, inc = self.incarnation(d, key[0])
        self.cohorts.refuse_release(uid, inc, key, l)

    def poll(self, d) -> None:
        self.table.prune()
        self.cohorts.poll()

    def report(self) -> Dict[str, Any]:
        u = self.table.entry_bytes
        return dict(mode="joined", entries=self.entries(), bytes=self.entries() * u, peak_entries=self.peak_entries,
                    peak_bytes=self.peak_entries * u, frontier=self.table.report(), finished=self.cohorts.report())
