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
   still owns them). Since milestone 7 the capacity counts outstanding tickets AND open reservations (outstanding()),
   and so does <n> in enqueue's refusal text "... refused before enqueue: <n> of <cap> result tickets are outstanding
   (...)": <n> is outstanding(). The text itself is the milestone-6 one (RT5 pins it); with no open reservation <n> is
   the ticket count, as in milestone 6.
2a. RESERVATION BEFORE ALLOCATION (milestone 7, INTERFACE_BC7 I15). reserve(call, layer) checks the capacity BEFORE
   the caller allocates or holds anything: outstanding tickets + OPEN reservations >= capacity -> CapacityRefusal
   "result tickets: call <c> layer <l>: refused before reservation: <n> of <cap> result tickets are outstanding (<t>
   tickets, <r> open reservations; nothing is allocated or held for this call) (I15)"; nothing changes but
   stats["refused"], and no view method is called. A reservation names no row and no request. commit(rid, keys,
   handle, cev, holds) runs enqueue's input checks first (KStepError before any change: the reservation stays OPEN),
   then the reservation becomes the ticket (its id is the reservation's id; no second capacity check). abort(rid) drops
   an OPEN reservation (the caller's allocation failed or was refused before any read was issued); no view call.
   commit and abort raise KStepError for an unknown rid ("... unknown reservation (I15)") and for one that is no
   longer open ("... the reservation is not open (it was committed or aborted) (I15)"). abort is NOT the cleanup of an
   issued read: a committed ticket finishes only through poll (live -> apply, stale -> discard) or stays BROKEN.
   enqueue is: the input checks, then reserve, then commit (so its order and its texts stay the milestone-6 ones).
   Tickets keep their COMMIT order (section 4 "earlier" means committed earlier).
3. POLL. poll() is PASSIVE: it never waits, never synchronizes, never forces an event; it asks view.done(cev) only.
   A ticket whose event is not complete is left exactly as it was: no check, resolve, reference, apply, layer
   advance or release. Completion alone releases nothing. A COMPLETE ticket is finished, in this order:
   view.check(handle) (the identity of every gathered item, first; it raises, for example StaleReader);
   view.resolve(handle) and view.reference(handle); the EXACT comparison of every row (each head, then the output;
   ConsumptionMismatch); then, for each row in call order, view.live(key): True -> view.apply(key, i, out) exactly
   once; False -> the row is discarded (never published); then view.release(holds, handle) (the gather backings and
   the handle are released only now: at application or safe discard). The comparison covers every row, also a row
   that is discarded: the gathered bytes are checked whether or not they are published. (I19: a row of a CANCELLED
   incarnation is discarded before view.live is asked; section 8.)
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
7. READ-ONLY QUERIES. pending() (the outstanding count, the first outstanding ticket: id, call, layer, keys, state,
   and, only while a reservation is open, "reservations": [dict(id=, call=, layer=)]); outstanding() (tickets AND open
   reservations); is_outstanding(tid) (True while the TICKET tid is pending or broken; False once finished, for an
   unknown id and for an open reservation's id); holds_request(uid, inc) (tickets only: a reservation names no
   request). stats: enqueued (tickets created: every commit counts one, also commit through enqueue), pending_polls
   (ticket visits by poll() that left the ticket outstanding: pending, behind an earlier ticket, or broken), finished,
   applied, discarded, refused (by reserve or enqueue).
8. CANCELLATION (milestone 8, INTERFACE_BC8 I19). cancel(uid, inc) is an EXPLICIT cancellation of a request
   incarnation whose physical read may still be pending. It returns n, the number of OUTSTANDING tickets (pending or
   broken) that name the incarnation (an open reservation names no request). With n > 0 it records ONE cancellation
   mark; with n == 0 it records nothing. Refused (KStepError, no change): a malformed uid / inc ("... malformed (the uid
   must not be None and the inc must be an int >= 1) (I19)" or "... (the uid is not hashable) (I19)"), and a second
   cancel while the mark is open ("... its cancellation mark is already open (<n> outstanding tickets name it) (I19)").
   cancel calls NO view method, queries no event, releases no hold and drops no handle: every ticket keeps its holds
   and its handle (the CHARGED backing) until it finishes through poll, after its event completed. Nothing is forced.
   At the finish of a ticket (section 3), check, resolve, reference and the exact comparison still cover EVERY row,
   also a cancelled one. Then, per row in call order, a row of a MARKED incarnation is discarded WITHOUT calling
   view.live (stats "cancelled_discarded"; it is never published, also when view.live would say True); every other row
   is decided by view.live as before. Then view.release as before. When the LAST outstanding ticket that names a marked
   incarnation finishes, the mark is dropped (stats "cancels_drained"). A BROKEN ticket stays outstanding, so a mark
   that it names never drains (holds_request stays True; the lead's request end and the reset stay refused).
   Stats: "cancels" (marks recorded), "cancelled_discarded" and "cancels_drained" exist from the FIRST recorded mark on;
   while nothing was ever cancelled, the stats, pending() and every text are the milestone-6 / 7 ones. "discarded"
   keeps its meaning (rows that view.live found not live); for a finished ticket, applied + discarded +
   cancelled_discarded = its row count. READ-ONLY: cancel_pending(uid, inc); pending()["cancels"] = [dict(uid=, inc=,
   tickets=<outstanding tickets that name it now>)], only while a mark is open.
   THE BOUND: a mark exists only for an incarnation that at least one OUTSTANDING ticket names, and at most one per
   incarnation (a second cancel is refused). So the open marks are at most the distinct incarnations named by the
   outstanding tickets: at most capacity x R, with R the largest row count of an outstanding ticket. With the lead's
   one global pending call (I17), at most the rows of that one call.
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
        self.res: Dict[int, Tuple[Any, Any]] = {}          # I15: the OPEN reservations, rid -> (call, layer), in order
        self.marks: Dict[Tuple[Any, Any], int] = {}        # I19: OPEN marks, (uid, inc) -> the outstanding tickets
                                                           # that named it AT CANCEL TIME (a record only; pending()
                                                           # and the drain count the outstanding tickets again)
        self.next_id = 1                                   # one id space: a reservation's id becomes its ticket's id
        self.stats: Dict[str, int] = {k: 0 for k in STATS}

    # ---------------------------------------------------------------------------------------- I15: reserve / commit
    def reserve(self, call, layer) -> int:
        """I15 (section 2a): checks the capacity BEFORE the caller allocates or holds anything: outstanding tickets +
        open reservations >= capacity -> CapacityRefusal (nothing changes but stats["refused"]; no view call). The
        reservation names no row. Returns its id (rid)."""
        return self._reserve(call, layer, "reservation")

    def commit(self, rid, keys, handle, cev, holds) -> int:
        """I15: the reservation rid becomes a ticket: first enqueue's input checks (KStepError before any change; the
        reservation stays OPEN), then the ticket takes ownership of handle and holds (no second capacity check).
        KStepError for an unknown rid or one that is no longer open (committed or aborted). Returns the ticket id (=
        rid)."""
        call, layer = self._open(rid, "commit")
        keys, holds = self._checked(call, layer, keys, cev, holds)
        return self._commit(rid, keys, handle, cev, holds)

    def abort(self, rid) -> None:
        """I15: drops an OPEN reservation (the caller's allocation failed or was refused BEFORE any read was issued). No
        view call. KStepError for an unknown rid or one that is no longer open. abort is NOT the cleanup of an issued
        read: a committed ticket finishes only through poll (live -> apply, stale -> discard) or stays BROKEN."""
        self._open(rid, "abort")
        del self.res[rid]

    # ------------------------------------------------------------------------------------------- I19: cancellation
    CANCEL_STATS = ("cancels", "cancelled_discarded", "cancels_drained")

    def cancel(self, uid, inc) -> int:
        """I19 (section 8): an EXPLICIT cancellation of the request incarnation (uid, inc). Returns n, the number of
        OUTSTANDING tickets (pending or broken) that name it; a reservation names no request, so it does not count.
        n > 0: ONE cancellation mark is recorded (stats["cancels"]). n == 0: nothing is recorded (there is nothing to
        discard). Refused (KStepError, no change): a malformed uid / inc, and a second cancel while the mark is open.
        It calls NO view method, queries no event, releases no hold and drops no handle: every ticket keeps its holds
        and its handle (the CHARGED backing) until it finishes through poll."""
        if uid is None or isinstance(inc, bool) or not isinstance(inc, int) or inc < 1:
            raise KStepError("result tickets: cancel of request incarnation (%r, %r): malformed (the uid must not be "
                             "None and the inc must be an int >= 1) (I19)" % (uid, inc))
        try:
            hash(uid)
        except TypeError:
            raise KStepError("result tickets: cancel of request incarnation (%r, %r): malformed (the uid is not "
                             "hashable) (I19)" % (uid, inc)) from None
        if (uid, inc) in self.marks:
            raise KStepError("result tickets: cancel of request incarnation (%r, %r): its cancellation mark is already "
                             "open (%d outstanding tickets name it) (I19)"
                             % (uid, inc, sum(1 for t in self.q if (uid, inc) in t.owners)))
        n = sum(1 for t in self.q if (uid, inc) in t.owners)
        if n:
            for k in self.CANCEL_STATS:        # the cancel counters exist from the first recorded mark on, so the
                self.stats.setdefault(k, 0)    # milestone-6 / 7 stats shape is unchanged while nothing is cancelled
            self.marks[(uid, inc)] = n
            self.stats["cancels"] += 1
        return n

    def cancel_pending(self, uid, inc) -> bool:
        """I19, READ-ONLY: True while the cancellation mark of (uid, inc) is open."""
        try:
            return (uid, inc) in self.marks
        except TypeError:
            return False

    def enqueue(self, call, layer, keys, handle, cev, holds) -> int:
        """Section 2 (milestone 6): enqueue's input checks, then reserve, then commit. The order and the texts are the
        milestone-6 ones: a malformed call is refused (KStepError) before the capacity is checked, and a refused
        capacity has the text "... refused before enqueue: <n> of <cap> result tickets are outstanding ..." (I14), where
        <n> is outstanding(): the tickets AND the open reservations (section 2). Returns the ticket id."""
        keys, holds = self._checked(call, layer, keys, cev, holds)
        rid = self._reserve(call, layer, "enqueue")
        return self._commit(rid, keys, handle, cev, holds)

    def _reserve(self, call, layer, what: str) -> int:
        n = self.outstanding()
        if n >= self.capacity:
            self.stats["refused"] += 1
            if what == "enqueue":
                raise CapacityRefusal("result tickets: call %s layer %s: refused before enqueue: %d of %d result "
                                      "tickets are outstanding (nothing of this call is held by the tickets) (I14)"
                                      % (call, layer, n, self.capacity))
            raise CapacityRefusal("result tickets: call %s layer %s: refused before reservation: %d of %d result "
                                  "tickets are outstanding (%d tickets, %d open reservations; nothing is allocated or "
                                  "held for this call) (I15)" % (call, layer, n, self.capacity, len(self.q),
                                                                 len(self.res)))
        rid = self.next_id
        self.next_id += 1
        self.res[rid] = (call, layer)
        return rid

    def _open(self, rid, what: str) -> Tuple[Any, Any]:
        """The (call, layer) of the OPEN reservation rid, else KStepError (no change)."""
        if isinstance(rid, bool) or not isinstance(rid, int) or not 1 <= rid < self.next_id:
            raise KStepError("result tickets: %s of reservation %r: unknown reservation (I15)" % (what, rid))
        if rid not in self.res:
            raise KStepError("result tickets: %s of reservation %d: the reservation is not open (it was committed or "
                             "aborted) (I15)" % (what, rid))
        return self.res[rid]

    def _commit(self, rid: int, keys, handle, cev, holds) -> int:
        call, layer = self.res.pop(rid)
        t = _Ticket(rid, call, layer, keys, handle, cev, holds)
        self.q.append(t)
        self.stats["enqueued"] += 1
        return t.tid

    def _checked(self, call, layer, keys, cev, holds) -> Tuple[Tuple[RowKey, ...], Tuple]:
        """enqueue's input checks (section 2): KStepError before any change. Returns (keys, holds) as tuples."""
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
        return keys, holds

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
            if (k.uid, k.inc) in self.marks:   # I19: a cancelled incarnation: discarded WITHOUT asking view.live
                self.stats["cancelled_discarded"] += 1
            elif v.live(k):
                v.apply(k, i, outs[i])
                self.stats["applied"] += 1
            else:
                self.stats["discarded"] += 1
        v.release(t.holds, t.handle)           # a fault inside it: section 5 (partly released; BROKEN, never again)
        t.state = FINISHED
        self.q.remove(t)
        t.handle = None                        # the ticket's reference is dropped with the ticket
        self.stats["finished"] += 1
        for inc_ in sorted(t.owners & set(self.marks), key=repr):   # I19: the LAST ticket of a mark drains it
            if not any(inc_ in u.owners for u in self.q):
                del self.marks[inc_]
                self.stats["cancels_drained"] += 1

    # ----------------------------------------------------------------------------------------------- read-only queries
    def pending(self) -> Dict[str, Any]:
        """READ-ONLY: the outstanding count (tickets and open reservations), the first outstanding ticket (id, call,
        layer, keys, state; None without a ticket) and, ONLY while a reservation is open, "reservations": a list of
        dict(id=<rid>, call=<call>, layer=<layer>), one per open reservation in reservation order (so the milestone-6
        shape is unchanged without reservations); ONLY while a cancellation mark is open (I19), "cancels": a list of
        dict(uid=<uid>, inc=<inc>, tickets=<outstanding tickets that name it now>), in cancel order."""
        t = self.q[0] if self.q else None
        first: Optional[Dict[str, Any]] = None if t is None else dict(id=t.tid, call=t.call, layer=t.layer,
                                                                      keys=t.keys, state=t.state)
        out: Dict[str, Any] = dict(outstanding=self.outstanding(), first=first)
        if self.res:
            out["reservations"] = [dict(id=rid, call=c, layer=l) for rid, (c, l) in self.res.items()]
        if self.marks:                         # I19: only while a mark is open (else the milestone-7 shape)
            out["cancels"] = [dict(uid=u, inc=i, tickets=sum(1 for t in self.q if (u, i) in t.owners))
                              for (u, i) in self.marks]
        return out

    def outstanding(self) -> int:
        """READ-ONLY: outstanding tickets (pending or broken) AND open reservations (I15: the lead's
        quiesce_state()["tickets"] uses it, so an open reservation blocks a reset)."""
        return len(self.q) + len(self.res)

    def is_outstanding(self, tid) -> bool:
        """READ-ONLY (no change, no view call): True while the TICKET tid is outstanding (pending or broken); False once
        it finished, for an unknown id, and for the id of an open reservation (it is not a ticket until commit). Only an
        int is an id (a bool is not: True == 1 must not name ticket 1)."""
        if isinstance(tid, bool) or not isinstance(tid, int):
            return False
        return any(t.tid == tid for t in self.q)

    def holds_request(self, uid, inc) -> bool:
        """READ-ONLY: True while an outstanding ticket (pending or broken) names the incarnation (uid, inc). An open
        reservation names no request, so it does not count here (I15)."""
        return any((uid, inc) in t.owners for t in self.q)
