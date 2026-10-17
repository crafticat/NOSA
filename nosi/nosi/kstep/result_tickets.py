"""Track C of K-step M2, milestone 6 (INTERFACE_BC6 I14): an OPT-IN deferred-result ticket component.

SCOPE. Source preparation and portable contract coverage only. It is not live asynchronous scheduling and makes no
CUDA, model, sharing or performance claim. The serialized default (Config.result_tickets = 0) does not use this
module. With the lead's adapter (HOOK-L8, dispatcher.py: Config.result_tickets > 0, DispatcherTicketView), the
dispatcher's consume call keeps its host wait and step() stays synchronous. A ticket only moves the identity check,
the resolution, the exact comparison, the per-row application and the release of one physical-consumption call
behind a PASSIVE poll, so that a later milestone can defer the wait. The CPU evidence is the deferred CpuBackend.

1. ROW IDENTITY. RowKey(uid, inc, epoch, cohort, pos, kind, layer, call) is an immutable snapshot of one consumed
   row, taken by the caller when it enqueues the call. A ticket keeps ONLY: its id, call and layer, the RowKeys in call
   order, the hold keys of the call's gather backings, the executor's handle (owned by the ticket from enqueue until
   its release) and the call's completion event. It NEVER keeps Request.cur, Cohort.hidden, Cohort.ws or any other
   mutable dispatcher object. Whether a row is still live is asked of the view (live(key), read-only) when the ticket
   is applied, never cached.
2. CAPACITY. At most `capacity` tickets are outstanding (capacity is an int >= 1; ConfigError otherwise). enqueue
   first checks its input (KStepError for a malformed call: no key, a key that is not a RowKey, a duplicate key, a key
   of another call or layer, no completion event, holds that are not a sequence), then the capacity: with `capacity`
   tickets outstanding it raises CapacityRefusal BEFORE any change of the tickets (only the counter "refused"
   counts it): the handle and the holds are NOT taken, so the tickets hold nothing of the refused call (the caller
   still owns them).
3. POLL. poll() is PASSIVE: it never waits, never synchronizes, never forces an event; it asks view.done(cev) only.
   A ticket whose event is not complete is left exactly as it was: no check, resolve, reference, apply, layer
   advance or release. Completion alone releases nothing. A COMPLETE ticket is finished, in this order:
   view.check(handle) (the identity of every gathered item, first; it raises, for example StaleReader);
   view.resolve(handle) and view.reference(handle); the EXACT comparison of every row (each head, then the output;
   ConsumptionMismatch); then, for each row in call order, view.live(key): True -> view.apply(key, i, out) exactly
   once; False -> the row is discarded (never published); then view.release(holds, handle) (the gather backings and
   the handle are released only now: at application or safe discard). The comparison covers every row, also a row
   that is discarded: the gathered bytes are checked whether or not they are published.
4. ORDER. A ticket names the request incarnations (uid, inc) of its rows. A ticket is finished only after every
   EARLIER outstanding ticket that names one of the same incarnations: a later complete ticket waits behind an
   earlier pending one (it is counted as a pending visit).
5. FAULTS. An exception while a complete ticket is finished propagates from poll(). The ticket is then BROKEN: it stays
   outstanding (so holds_request stays True, a request end and a quiescent reset stay refused) and no later poll
   touches it again (no second check, no second apply, no second release). A fault BEFORE view.release (in check,
   resolve, reference, the comparison, live or apply) leaves every hold held and the handle kept. A fault INSIDE
   view.release leaves what the view did until the fault: some holds may already be released and the handle may be
   partly cleared; the ticket keeps its hold keys and its handle reference, but they are not released again. Rows
   applied before the fault stay applied: this is NOT a transactional rollback.
   A finished ticket gets the state "finished" before it is removed (it is no longer reachable from the queue). The
   row-count text of a resolve / reference length mismatch ("result tickets: ticket <id> (call <c> layer <l>)
   resolved ...") differs from the serialized consumer's text ("C1.2: call <c> resolved ..."); both are KStepError.
6. VIEW PROTOCOL (the lead's DispatcherTicketView; a component test may use its own view on the REAL CpuBackend):
   done(ev) -> bool (never waits); check(handle); resolve(handle) -> (outs, heads); reference(handle) -> (outs, heads);
   live(key) -> bool (READ-ONLY); apply(key, i, out); release(holds, handle).
7. READ-ONLY QUERIES. pending() (the outstanding count and the first outstanding ticket: id, call, layer, keys, state),
   outstanding(), holds_request(uid, inc). stats: enqueued, pending_polls (ticket visits by poll() that left the
   ticket outstanding: pending, behind an earlier ticket, or broken), finished, applied, discarded, refused.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .errors import CapacityRefusal, ConfigError, ConsumptionMismatch, KStepError

RowKey = namedtuple("RowKey", "uid inc epoch cohort pos kind layer call")
VIEW_METHODS = ("done", "check", "resolve", "reference", "live", "apply", "release")
STATS = ("enqueued", "pending_polls", "finished", "applied", "discarded", "refused")
PENDING, BROKEN, FINISHED = "pending", "broken", "finished"


class _Ticket:
    """One outstanding call (section 1): immutable identity and the owned handle, holds and event."""
    __slots__ = ("tid", "call", "layer", "keys", "handle", "cev", "holds", "owners", "state")

    def __init__(self, tid: int, call, layer, keys: Tuple[RowKey, ...], handle, cev, holds: Tuple):
        self.tid, self.call, self.layer, self.keys = tid, call, layer, keys
        self.handle, self.cev, self.holds = handle, cev, holds
        self.owners: FrozenSet[Tuple[Any, Any]] = frozenset((k.uid, k.inc) for k in keys)
        self.state = PENDING


class ResultTickets:
    """The module docstring: bounded, passive, apply-once result tickets of physical-consumption calls."""

    def __init__(self, capacity: int, view):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ConfigError("result tickets: capacity must be an int >= 1, not %r (I14)" % (capacity,))
        miss = [m for m in VIEW_METHODS if not callable(getattr(view, m, None))]
        if miss:
            raise ConfigError("result tickets: the view %s lacks %s (I14)" % (type(view).__name__, miss))
        self.capacity = capacity
        self.view = view
        self.q: List[_Ticket] = []
        self.next_id = 1
        self.stats: Dict[str, int] = {k: 0 for k in STATS}

    # -------------------------------------------------------------------------------------------------------- enqueue
    def enqueue(self, call, layer, keys, handle, cev, holds) -> int:
        """Section 2: checks the input, then the capacity, then takes ownership of handle and holds. Returns the id."""
        where = "result tickets: call %s layer %s" % (call, layer)
        try:
            keys = tuple(keys)
        except TypeError:
            raise KStepError("%s: the keys are not a sequence (%s) (I14)" % (where, type(keys).__name__)) from None
        if not keys:
            raise KStepError("%s: no row key (key count 0) (I14)" % where)
        bad = [k for k in keys if type(k) is not RowKey]
        if bad:
            raise KStepError("%s: a key that is not a RowKey: %r (I14)" % (where, bad[0]))
        seen = set()
        for k in keys:
            if k in seen:
                raise KStepError("%s: duplicate row key %r (I14)" % (where, k))
            seen.add(k)
        other = [k for k in keys if k.call != call or k.layer != layer]
        if other:
            raise KStepError("%s: a row key of another call or layer: %r (I14)" % (where, other[0]))
        if cev is None:
            raise KStepError("%s: no completion event (I14)" % where)
        try:
            holds = tuple(holds)
        except TypeError:
            raise KStepError("%s: the holds are not a sequence (%s) (I14)" % (where, type(holds).__name__)) from None
        if len(self.q) >= self.capacity:
            self.stats["refused"] += 1
            raise CapacityRefusal("%s: refused before enqueue: %d of %d result tickets are outstanding (nothing of "
                                  "this call is held by the tickets) (I14)" % (where, len(self.q), self.capacity))
        t = _Ticket(self.next_id, call, layer, keys, handle, cev, holds)
        self.next_id += 1
        self.q.append(t)
        self.stats["enqueued"] += 1
        return t.tid

    # ----------------------------------------------------------------------------------------------------------- poll
    def poll(self) -> int:
        """Sections 3-4: PASSIVE. Finishes every complete ticket that no earlier outstanding ticket of the same
        incarnation blocks, in enqueue order. Returns the number of tickets finished by this poll."""
        finished = 0
        blocked: set = set()
        for t in list(self.q):
            if t.state == BROKEN or (t.owners & blocked) or not self.view.done(t.cev):
                self.stats["pending_polls"] += 1
                blocked |= t.owners
                continue
            self._finish(t)
            finished += 1
        return finished

    def _finish(self, t: _Ticket) -> None:
        """Section 3 for one complete ticket; section 5 on a fault (the ticket stays, BROKEN)."""
        v = self.view
        t.state = BROKEN                       # until every step passed: a fault leaves it broken, never retried
        v.check(t.handle)
        outs, heads = v.resolve(t.handle)
        ref_outs, ref_heads = v.reference(t.handle)
        n = len(t.keys)
        if not (len(outs) == len(heads) == len(ref_outs) == len(ref_heads) == n):
            raise KStepError("result tickets: ticket %d (call %s layer %s) resolved %d outputs / %d head lists and "
                             "%d / %d reference ones for %d rows (I14)"
                             % (t.tid, t.call, t.layer, len(outs), len(heads), len(ref_outs), len(ref_heads), n))
        for i, k in enumerate(t.keys):
            if len(heads[i]) != len(ref_heads[i]):
                raise ConsumptionMismatch("C1 consumption mismatch: call %s row (%s, %s, %s): %d consumed heads != %d "
                                          "reference heads" % (t.call, k.uid, k.pos, k.kind, len(heads[i]),
                                                               len(ref_heads[i])))
            for h, (a, b) in enumerate(zip(heads[i], ref_heads[i])):
                if a != b:
                    raise ConsumptionMismatch("C1 consumption mismatch: call %s row (%s, %s, %s) head %d: consumed %r "
                                              "!= reference %r" % (t.call, k.uid, k.pos, k.kind, h, a, b))
            if outs[i] != ref_outs[i]:
                raise ConsumptionMismatch("C1 consumption mismatch: call %s row (%s, %s, %s) output: consumed %r != "
                                          "reference %r" % (t.call, k.uid, k.pos, k.kind, outs[i], ref_outs[i]))
        for i, k in enumerate(t.keys):         # call order; each row exactly once
            if v.live(k):
                v.apply(k, i, outs[i])
                self.stats["applied"] += 1
            else:
                self.stats["discarded"] += 1
        v.release(t.holds, t.handle)           # a fault inside it: section 5 (partly released; BROKEN, never again)
        t.state = FINISHED
        self.q.remove(t)
        t.handle = None                        # the ticket's reference is dropped with the ticket
        self.stats["finished"] += 1

    # ----------------------------------------------------------------------------------------------- read-only queries
    def pending(self) -> Dict[str, Any]:
        """READ-ONLY: the outstanding count and the first outstanding ticket (id, call, layer, keys, state)."""
        t = self.q[0] if self.q else None
        first: Optional[Dict[str, Any]] = None if t is None else dict(id=t.tid, call=t.call, layer=t.layer,
                                                                      keys=t.keys, state=t.state)
        return dict(outstanding=len(self.q), first=first)

    def outstanding(self) -> int:
        return len(self.q)

    def holds_request(self, uid, inc) -> bool:
        """READ-ONLY: True while an outstanding ticket (pending or broken) names the incarnation (uid, inc)."""
        return any((uid, inc) in t.owners for t in self.q)
