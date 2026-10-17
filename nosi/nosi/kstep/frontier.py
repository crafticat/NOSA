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
   - Hold providers (CompletionOwnership._bind, at the first admission, before the request enters Dispatcher.reqs):
     "reader_frontier", "transfers" (in-flight transfers of the uid whose transport does not report them complete),
     "ring_gates" (ring entries owned by the uid with an incomplete ready / compute / D2H gate, and its admission buffer
     A_r with an incomplete gate), "d2h" (the request's DISTINCT incomplete D2H events: hist_ev and snapshot events,
     each counted once) and, when Dispatcher.store is not None (track B's opt-in mode), the provider that
     store.hold_source() returns ("tensor_store"). tensor_store.py is never imported here. Fail closed: a store without
     hold_source, Config.store other than "none" without a store, or a name that is already registered is a named
     ConfigError, raised before anything is registered. Records clear only when EVERY provider reports 0. Every
     provider is non-blocking (ev_done / transport done only).
   - Decision entry: Dispatcher._apply, _finish_draft and _end_request first ask CompletionOwnership, BEFORE any
     mutation of accepted drafts, progress, publications, tokens or results. Counted no-ops: a decision of a finished
     key (duplicate_decisions_ignored, or cancelled_decisions_ignored for a cancelled cohort) and a draft completion of
     an old epoch (stale_drafts_ignored). Refused by name: a decision, draft completion or end that names an ended or
     cleared incarnation or a Request object that is not the dispatcher's; a decision of a cohort that is not the DONE
     head of the current epoch; a draft completion in the current epoch whose draft is not RUNNING or not the next one.
   - No uid-reuse promise beyond this: Dispatcher.submit refuses a uid that it knows, so a uid has one incarnation per
     dispatcher today. Under uid reuse the provider attribution would not be exact (CompletionOwnership docstring).
   Milestone 8 (INTERFACE_BC8 I19): the incarnation state CANCELLING, between ACTIVE and ENDED.
   - FinishedCohorts.cancel(uid, inc): ACTIVE -> CANCELLING, refused (I3) before any change for an incarnation that is
     not live, that ended, or that is already cancelling. CompletionOwnership.cancel_entry(d, r) is the dispatcher's
     entry (also refuses a stale Request object). The counter "cancelled" exists from the first cancellation on.
   - In CANCELLING: a decision RECORD (check_finish / finish: the cancellation teardown records each cancelled cohort
     key, as the rejection path's _finished_key does), late_queue and refuse_release work as in ACTIVE; a protection
     (refuse_protect), a decision ENTRY (check_decision, decision_entry) and a draft completion entry (draft_entry) are
     REFUSED by name ("... while it is cancelling (I19)"); end (and end_entry) accept ACTIVE or CANCELLING (-> ENDED;
     a second end stays refused). A uid that is cancelling cannot start another incarnation.
   - The records stay until every hold provider reports 0 for that incarnation (the existing rule; poll clears ENDED
     incarnations only, so a CANCELLING one is never cleared before its end).
   - A NEW incarnation that reuses the cancelled request's batch slot has a NEW uid (submit refuses a known uid). Every
     provider attributes by uid ("transfers": Transfer.uid; "ring_gates": RingPool.owner[e][0] and A_r["uid"];
     "d2h": d.reqs[uid].hist_ev when its inc matches, and the D2H records' "uid"; "reader_frontier": the table's
     owner incarnation; "tensor_store", when Dispatcher.store exists: track B's source holds, exact by their (uid,
     inc) owner), so the counts of the CANCELLING / ENDED incarnation never include the new incarnation's work, and it
     clears exactly when ITS OWN providers drain. (The controls CC6-CC8 exercise d2h, ring_gates with A_r, and
     transfers; tensor_store is not exercised there.)

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
   are valid only between two resets. The rule: CompletionOwnership.before_reset() must run before any backend reset.
   It refuses (OwnershipViolation "reader frontier: reset refused (I4): ...") while any reader component, installed
   handle or cached join (output or input) in the table is incomplete, or any hold provider of a live incarnation
   reports a count. Otherwise it clears the identity-keyed state (clear_caches). It never waits, forces or drops a
   reference to make a reset possible. backend.py and cpupack_core are unchanged: the caller of reset must call
   before_reset first.
   before_reset is NECESSARY, NOT SUFFICIENT. The table forgets an object when its components are complete (prune), but
   the handle installed for it can stay referenced outside the table (slot / position last_reader, key_readers, a ring
   compute gate, Transfer.reader_ev): for example a join output whose inputs are all complete but whose record has not
   run yet (the CpuBackend runs "join" only when forced). A reset path must also check those holders (the pools' and
   the dispatcher's own references).

8. QUIESCENT RESET (milestone 3, addendum I8; milestone 4, addendum I10; milestone 5, addendum I12).
   QuiescentReset(disp, backend_reset) is that reset path. It uses only the lead's Dispatcher hooks (HOOK-L3:
   quiesce_state, event_holders, detach_completed_events; HOOK-L4 puts the executor's reset contract into the last
   two; HOOK-L7: session, session_reason, begin_reset, end_reset, fail_reset, host_poll, and finish), the executor
   module's reset_contract_of / reset_methods_missing, this module's reset_pending / before_reset, and track B's store
   API (drained_all, clear_caches).
   THE SESSION (I12). Dispatcher.session is "open" (initial), "resetting" or "failed". Outside "open", the lead's
   submit, step (so run), finish and host_poll refuse before any change. QuiescentReset is the only caller of
   begin_reset ("open" -> "resetting"), end_reset ("resetting" -> "open") and fail_reset ("resetting" -> "failed",
   LATCHED: nothing returns a failed session to "open"). session_end() is the explicit session-end sequence:
   Dispatcher.finish (drain every stream and validate), then ONE explicit host poll (Dispatcher.host_poll), then
   reset(). There is no waiting loop, no automatic retry and no speculative mid-cohort reset: an item that is still
   outstanding after the one poll is a refusal (the session stays "open"), and the caller decides what comes next.
   THE DRAIN (I12.1). finish and host_poll are the named DRAIN phases "finish" and "host_poll" of session_end. They
   can change state and then raise: finish synchronizes every stream and then validates (_validate, the D2H checks);
   host_poll removes an arrived transfer from inflight before its admission readback. So each one runs through the
   same _phase as the mutating phases, with every lookup inside it, but with the session "open" (begin_reset is NOT
   called before them: the lead's guards refuse finish and host_poll outside "open"). Any exception latches the
   session from "open" (begin_reset, then fail_reset(<phase>, "<type>: <text>")) and raises OwnershipViolation
   "quiescent reset failed in phase <phase>: <type>: <text> (the session is latched failed: ...)" from it; a drain
   phase that returns but leaves the session other than "open" is a fault too. A successful drain keeps the session
   "open" until reset()'s read-only check; a refusal there leaves it "open". session_end refuses a session that is
   not "open" (for example "resetting") before any drain, with no change.
   reset():
   (0) a FAILED session refuses at once: OwnershipViolation "quiescent reset refused: the session is latched failed
       (<reason>)", with no hook, store or backend call.
   (1) the FULL check, with no mutation (pending()): every quiesce_state() value 0 (an OPEN dict: every key, whatever
       its name); every event_holders() event complete (ev_done); the frontier table (incomplete components, installed
       handles, cached joins) and every registered hold provider of every live incarnation at 0; store.drained_all()
       when Dispatcher.store exists; the executor declares the I10 contract. Anything outstanding: OwnershipViolation
       "quiescent reset refused: <items>" that names EVERY outstanding owner class, one item each, in a fixed order:
       "state:<key>=<n>" (quiesce_state keys, sorted), "holder:<owner class>=<pending references>" (sorted; a declared
       executor's classes "executor:..." too), "frontier:components|handles|joins=<n>", "provider:<name>(<uid>,<inc>)=
       <n>" (incarnations and names sorted), "store:not_drained=1", "tickets:reservations=<n>" (I15),
       "tickets:cancels=<n>" (I19), "executor:no_reset_contract=1". Each class is
       named once with its count. Nothing is changed and the session stays "open"; so does an exception that the
       read-only check itself raises (for example a HOOK-L4 refusal inside event_holders()). The check never polls: a
       store record that completed but was not polled yet is refused, and so are registered validators, D2H checks
       and admission buffers (the lead's quiesce_state counts them; their completion runs byte checks that may raise).
       The caller drains them first: session_end() (Dispatcher.finish, then one host poll). An outstanding deferred-
       result ticket (I14, result_tickets.py; Config.result_tickets > 0) is refused through the lead's quiesce_state
       key "tickets" (state:tickets=<n>); its gather backings are also counted (state:consume_ledger_keyed=<n>).
       An OPEN reservation of the tickets (I15: reserved, not yet committed or aborted) is counted in "tickets" too and
       is also named by its own item "tickets:reservations=<n>" (read-only, from Dispatcher.tickets.pending()).
       An OPEN cancellation mark of the tickets (I19) implies an outstanding ticket (so the reset is refused already);
       it is also named by its own item "tickets:cancels=<n>" (from pending()["cancels"]).
   (2) begin_reset(): the session becomes "resetting".
   (3) The MUTATING phases, in this order, each by its name:
       "detach"        detach_completed_events() (HOOK-L4: a declared executor's detach_completed() first; the hook
                       VERIFIES it before any dispatcher change: its count must equal the references it listed, and
                       nothing may be left listed; then the passive host poll, which also runs this module's poll; its
                       post-condition; then every remaining reference to a complete event is set to None);
       "post_check"    event_holders() is empty, else OwnershipViolation "quiescent reset: the detach left <n> event
                       references: [<first 6 (class, location)>]; backend_reset did not run (I10)";
       "before_reset"  this module's before_reset() (the identity caches; it checks again);
       "clear_caches"  store.clear_caches() (when Dispatcher.store exists);
       "backend_reset" backend_reset() (the point where pooled events may be re-recorded).
       reset() runs PHASES in order, one method each (_run_<phase>), and each method looks its target up when it runs
       (a missing frontier.before_reset or store.clear_caches is a fault of that phase). ANY exception in a phase
       calls fail_reset(phase, "<type>: <text>") and raises OwnershipViolation "quiescent reset failed in phase
       <phase>: <type>: <text> (the session is latched failed: no resume, no admission, no reset retry)" from it.
       <text> is str() of the exception, else repr(), else "<unprintable <type>>". The session is then "failed": no
       later submit, step, finish, host_poll, detach_completed_events or reset runs, and the backend is not used again
       through them. (An exception that is not an Exception, such as KeyboardInterrupt, also latches the session, and
       is raised again unchanged.) After each phase that returns, the session must still be "resetting"; else the
       phase is a fault with the text "OwnershipViolation: the phase left the session <state> (expected
       'resetting')". Latching follows the state that the phase left: "resetting" -> fail_reset; "open" ->
       begin_reset, then fail_reset; "failed" -> nothing is called (already latched; its reason is kept); any other
       state, or a hook that raises while latching -> the session is NOT latched, and the bracket says so instead:
       "(the session could NOT be latched failed: [<hook error>; ]the session is <state>)". If begin_reset itself
       leaves the session other than "resetting", reset() refuses before any mutating phase.
       A phase failure is NOT a no-change refusal and NOT a rollback: the phases before it, and the failing phase up
       to its fault, have already changed state. In "detach" the executor's own detach may already have changed the
       executor's state when the hook's verification refuses it (the dispatcher is unchanged at that point); after a
       later fault, the dispatcher's complete references are already set to None. Only the coordinator's counts are
       unchanged, and no later phase runs.
       The post-check is a SECOND guard, against a dispatcher hook defect (the hook already verifies the executor's
       detach).
   (4) end_reset() (the session is "open" again), then the counts.
   The crucial case: the table can be drained (its components completed and pruned) while a JOINED output that it
   installed stays referenced outside it (a slot's last reader) and incomplete. Only the holder enumeration sees it.
   THE EXECUTOR (I10). An executor can keep OPERATIONAL events between calls (an event it will wait on, hand on or test
   later, for example the last write of a private output buffer). The dispatcher cannot see them. So a reset needs the
   executor's declaration: reset_contract = "I10" IN ITS OWN CLASS BODY (executor.reset_contract_of reads
   type(ex).__dict__; neither inheritance nor attribute forwarding declares it), with reset_holders() (read-only: every
   operational reference, owner classes "executor:...") and detach_completed() (refuses with no mutation while one is
   pending; else sets them to None). An executor without the declaration is refused by the item
   "executor:no_reset_contract=1", even when nothing else is outstanding. Construction refuses a declared executor
   whose classes do not define reset_holders and detach_completed (executor.reset_methods_missing: the classes of
   type(ex).__mro__ only; __getattr__ forwarding and instance attributes do not count) with ConfigError "quiescent
   reset: the executor <cls> declares reset_contract 'I10' without the class methods [...] (I10)". The hook calls the
   two methods through the class (type(ex).reset_holders(ex)), so an instance attribute cannot shadow them. The hook's
   own I10 refusals are of two kinds. Those raised inside event_holders() (an owner class that does not start with
   "executor:", missing class methods, the TypeError of a wrapper over an undeclared toy) come out of pending()
   unchanged, before any change; the session stays "open". A detach that fails the hook's verification is refused
   AFTER the executor's own detach ran: the dispatcher is unchanged, but the executor's own state may already be
   changed (no transactional rollback), and the coordinator latches the session "failed" in phase "detach".
   Instrumentation history that an executor keeps (logs, counters, eager copies that no later decision reads) is not
   listed and not detached; its docstring names it.
   BOUNDARY: the coordinator relies on the declared list. An operational event that reset_holders() leaves out is not
   seen (the omitted-provider hazard); every executor class is audited on its own. No claim for an arbitrary backend or
   an undeclared executor.
   The no-reset path is unchanged: no coordinator exists unless a caller constructs one, and without one the session
   stays "open". No live-reset scheduling, no uid-reuse redesign (Dispatcher.submit still refuses a uid that it knows).
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
    """Section 4 of the module docstring. live[(uid, inc)] = "active" | "cancelling" (I19) | "ended"; rec[(uid, inc,
    key)] = the decision serial; sources = [(name, fn(uid, inc) -> number of outstanding holds)]. Every check runs
    before any change."""
    ACTIVE, CANCELLING, ENDED = "active", "cancelling", "ended"

    def __init__(self, entry_bytes: int = 16):
        self.entry_bytes = int(entry_bytes)
        self.live: Dict[Tuple[Any, Any], str] = {}
        self.rec: Dict[Tuple[Any, Any, Tuple], int] = {}
        self.sources: List[Tuple[str, Callable[[Any, Any], int]]] = []
        self.stats: Dict[str, int] = dict(started=0, ended=0, finishes=0, duplicate_finish=0, late_queue_ignored=0,
                                          late_protect_refused=0, late_release_refused=0, records_cleared=0,
                                          incarnations_cleared=0, duplicate_decisions_ignored=0,
                                          cancelled_decisions_ignored=0, stale_drafts_ignored=0,
                                          duplicate_drafts_refused=0, stale_callbacks_refused=0)
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
        other = [i for (u, i), st in self.live.items() if u == uid and st == self.CANCELLING]
        if other:                                              # I19: a cancelling incarnation still owns the uid
            raise KStepError("I3: uid %s is cancelling as incarnation %s; incarnation %s cannot start (I19)"
                             % (uid, other[0], inc))
        self.live[(uid, inc)] = self.ACTIVE
        self.stats["started"] += 1
        self._peak()

    def check_finish(self, uid, inc, key) -> bool:
        """The checks of a decision, before any change: the key belongs to uid and the incarnation is live. Returns True
        for a DUPLICATE decision (counted; the caller changes nothing). A first decision after the end is refused. I19:
        a CANCELLING incarnation accepts it (the cancellation teardown records each cancelled cohort key, as the
        rejection path's _finished_key does)."""
        self._own(uid, key)
        st = self.state(uid, inc, lambda: "the decision of cohort %s" % (key,))
        if (uid, inc, key) in self.rec:
            self.stats["duplicate_finish"] += 1
            return True
        if st == self.ENDED:
            raise KStepError("I3: decision of cohort %s of request incarnation (%s, %s) after its end" % (key, uid, inc))
        return False

    def check_decision(self, uid, inc, key) -> bool:
        """Milestone 2 (I6), the ENTRY of a decision (Dispatcher._apply), before any mutation: the key belongs to uid; the
        incarnation must be live (else I3 "... which is not live ...") and ACTIVE (else I3 "decision ... after its end",
        also for a finished key); a finished key returns True (the caller changes nothing and counts it by its kind).
        Unlike check_finish, the incarnation state is checked BEFORE the finished-key test. I19: a CANCELLING
        incarnation is refused by name ("I3: decision of cohort <key> of request incarnation (<uid>, <inc>) while it is
        cancelling (I19)")."""
        self._own(uid, key)
        try:
            st = self.state(uid, inc, lambda: "the decision of cohort %s" % (key,))
            if st == self.CANCELLING:
                raise KStepError("I3: decision of cohort %s of request incarnation (%s, %s) while it is cancelling "
                                 "(I19)" % (key, uid, inc))
            if st != self.ACTIVE:
                raise KStepError("I3: decision of cohort %s of request incarnation (%s, %s) after its end"
                                 % (key, uid, inc))
        except KStepError:
            self.stats["stale_callbacks_refused"] += 1
            raise
        return (uid, inc, key) in self.rec

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
        ended or is not live (I3). I19: a CANCELLING incarnation is refused by name ("I3: protection of block <b> at
        layer <l> for cohort <key> of request incarnation (<uid>, <inc>) while it is cancelling (I19)"), after the
        finished-key test."""
        self._own(uid, key)
        st = self.state(uid, inc, lambda: "the protection of block %d at layer %d for cohort %s" % (block, l, key))
        if (uid, inc, key) in self.rec:
            self.stats["late_protect_refused"] += 1
            raise KStepError("R1.5: late protection of block %d at layer %d for cohort %s after the cohort finished "
                             "(durable finished-cohort record, incarnation %s)" % (block, l, key, inc))
        if st == self.CANCELLING:
            raise KStepError("I3: protection of block %d at layer %d for cohort %s of request incarnation (%s, %s) "
                             "while it is cancelling (I19)" % (block, l, key, uid, inc))
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
        """The request incarnation ended (from ACTIVE, or from CANCELLING: I19). Its records stay until every hold
        source reports nothing (poll)."""
        if self.state(uid, inc, lambda: "the end of a request") == self.ENDED:
            raise KStepError("I3: request incarnation (%s, %s) ended twice" % (uid, inc))
        self.live[(uid, inc)] = self.ENDED
        self.stats["ended"] += 1
        return self.poll()

    def cancel(self, uid, inc) -> None:
        """I19: ACTIVE -> CANCELLING, refused (KStepError, I3) BEFORE any change for an incarnation that is not live
        (the state() text, "the cancellation of a request"), that ended ("I3: cancellation of request incarnation
        (<uid>, <inc>) after its end (I19)") or that is already cancelling ("I3: request incarnation (<uid>, <inc>) is
        cancelled twice (I19)"). The counter stats["cancelled"] exists from the first cancellation on."""
        st = self.state(uid, inc, lambda: "the cancellation of a request")
        if st == self.CANCELLING:
            raise KStepError("I3: request incarnation (%s, %s) is cancelled twice (I19)" % (uid, inc))
        if st != self.ACTIVE:
            raise KStepError("I3: cancellation of request incarnation (%s, %s) after its end (I19)" % (uid, inc))
        self.live[(uid, inc)] = self.CANCELLING
        self.stats["cancelled"] = self.stats.get("cancelled", 0) + 1

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
        cands = [k for k, st in self.live.items() if st == self.ENDED]
        clear = [k for k in cands if not any(self.outstanding(*k).values())]   # every provider of every candidate
        for uid, inc in clear:                                  # first, so a provider error leaves nothing half-cleared
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

    Scope (lead integration of tracks B and C, after the independent review): the built-in hold source is the reader
    frontiers ONLY. In-flight transfers (d.inflight), ring ready / D2H gates, hist_ev and Track B's source holds are not
    consulted yet, so a record may clear while such work of the incarnation is still pending. That is fail-closed: after
    clearing, a late protect, queue or decision raises the I3 "not live" error; only the finer diagnosis is lost. Track
    B's holds attach with add_source("tensor_store", store.holds.outstanding_count) when the dispatcher uses the store.
    Mutation catchers run in "single" mode: in "joined" mode the frontier's R1.5 check still fires under
    no_late_protection_guard.

    Milestone 2 (addendum I5; HOOK-C2; this extends the note above): _bind registers the providers "transfers",
    "ring_gates", "d2h" and, when Dispatcher.store is not None, the provider that store.hold_source() returns
    ("tensor_store"), at the first admission and BEFORE the request enters Dispatcher.reqs. A record clears only when
    EVERY registered provider reports 0 for its incarnation. Fail closed: a store without hold_source, a Config.store
    other than "none" without a store, or a provider name that is already registered is a named ConfigError.
    Attribution: "transfers", "ring_gates" and the D2H snapshot records match by uid; hist_ev is read only from
    d.reqs[uid] when its inc matches. Dispatcher.submit refuses a known uid, so a uid has one incarnation per dispatcher
    today. If a uid ever had two incarnations, the attribution would not be exact: the uid-matched providers would count
    the work of both, and the hist_ev of the older one would no longer be visible."""

    ACCEPTED, CANCELLED, DONE = "accepted", "cancelled", "done"    # dispatcher.py's cohort states (no import cycle)
    RUNNING = "running"                                           # dispatcher.py's draft state of a draft in flight

    def __init__(self, be, join_stream: str = JOIN_STREAM, entry_bytes: int = 16):
        self.table = FrontierTable(be, join_stream, entry_bytes)
        self.cohorts = FinishedCohorts(entry_bytes)
        self.cohorts.add_source("reader_frontier", lambda uid, inc: len(self.table.outstanding((uid, inc))))
        self.peak_entries = 0
        self.bound = False
        self.resets = 0

    # ---------------------------------------------------------------------------------- I5: the hold providers
    def _bind(self, d) -> None:
        """Registers the dispatcher's hold providers once (I5; lead decision D1). Every check runs BEFORE the first
        registration, and `bound` is set only after the last one, so a refusal registers nothing:
          - Dispatcher.store not None: its hold_source() must exist and return ("tensor_store", a callable);
          - Dispatcher.store None: Config.store (when the field exists) must be "none";
          - no provider name may be registered already.
        Each failure is a named ConfigError (fail closed). tensor_store.py is never imported here."""
        if self.bound:
            return
        plan = [("transfers", lambda uid, inc: self.transfers_pending(d, uid, inc)),
                ("ring_gates", lambda uid, inc: self.ring_gates_pending(d, uid, inc)),
                ("d2h", lambda uid, inc: self.d2h_pending(d, uid, inc))]
        store = getattr(d, "store", None)
        if store is not None:
            hs = getattr(store, "hold_source", None)
            if not callable(hs):
                raise ConfigError("reader frontier: Dispatcher.store has no hold_source(); the 'tensor_store' hold "
                                  "provider cannot be registered (I5; fail closed)")
            src = hs()
            if not (isinstance(src, tuple) and len(src) == 2 and src[0] == "tensor_store" and callable(src[1])):
                raise ConfigError("reader frontier: Dispatcher.store.hold_source() returned %r, not ('tensor_store', a "
                                  "callable) (I5; fail closed)" % (src,))
            plan.append(src)
        elif getattr(d.cfg, "store", "none") != "none":
            raise ConfigError("reader frontier: Config.store is %r but the dispatcher holds no store; the "
                              "'tensor_store' hold provider cannot be registered (I5; fail closed)" % (d.cfg.store,))
        have = {n for n, _ in self.cohorts.sources}
        clash = [n for n, _ in plan if n in have]
        if clash:
            raise ConfigError("reader frontier: hold provider(s) %s already registered before the dispatcher's "
                              "registration (I5)" % (clash,))
        for name, fn in plan:
            self.cohorts.add_source(name, fn)
        self.bound = True

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
        """The DISTINCT incomplete D2H events of the request, each counted once: its hist_ev events (read from
        d.reqs[uid] only when that Request is incarnation inc) and the hist_ev and snapshot events of uid's checked-mode
        D2H records (a record's hist_ev is the same object as the request's, so it is not counted twice)."""
        evs: Dict[int, Any] = {}
        r = d.reqs.get(uid)
        if r is not None and getattr(r, "inc", None) == inc:
            for ev in r.hist_ev.values():
                evs[id(ev)] = ev
        for rec in d.d2h_checks:
            if rec["uid"] == uid:
                evs[id(rec["hist_ev"])] = rec["hist_ev"]
                evs[id(rec["snap_ev"])] = rec["snap_ev"]
        return sum(1 for ev in evs.values() if not ev_done(ev))

    # ---------------------------------------------------------------------------------- I6: the decision entry
    def _live_request(self, d, r, what: Callable[[], str]) -> None:
        """r must be the dispatcher's Request object of its uid (a stale object of an old incarnation is refused)."""
        if d.reqs.get(r.uid) is not r:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise KStepError("I3: %s names a Request object of uid %s (incarnation %s) that is not the dispatcher's"
                             % (what(), r.uid, getattr(r, "inc", None)))

    def decision_entry(self, d, r, c) -> bool:
        """Dispatcher._apply's entry, BEFORE any mutation. True: a decision of a finished key, a counted no-op: counted
        as cancelled_decisions_ignored when the cohort was cancelled, else as duplicate_decisions_ignored. Refused
        (raises): an ended or cleared incarnation (I3), a stale Request object (I3), or a cohort that is not the DONE
        head of the request's current epoch (OwnershipViolation "decision entry: ...")."""
        self._live_request(d, r, lambda: "the decision of cohort %s" % (c.key,))
        if self.cohorts.check_decision(r.uid, r.inc, c.key):
            self.cohorts.stats["cancelled_decisions_ignored" if c.state == self.CANCELLED
                               else "duplicate_decisions_ignored"] += 1
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
        no-op, stale_drafts_ignored; _end_of_step skips those already). Refused (raises, I3): a stale Request object, or
        an incarnation that ended or is not live. Refused (raises OwnershipViolation "draft entry: ...", counted as
        duplicate_drafts_refused): a completion in the CURRENT epoch whose draft is not RUNNING (a duplicate delivery of
        a completed draft) or is not the next draft (dr.s != len(r.drafts) + 1)."""
        self._live_request(d, r, lambda: "the draft completion s%d" % (dr.s,))
        try:
            st = self.cohorts.state(r.uid, r.inc, lambda: "the draft completion s%d (epoch %d)" % (dr.s, dr.epoch))
            if st == FinishedCohorts.CANCELLING:                         # I19: refused by name
                raise KStepError("I3: draft completion s%d (epoch %d) of request incarnation (%s, %s) while it is "
                                 "cancelling (I19)" % (dr.s, dr.epoch, r.uid, r.inc))
            if st != FinishedCohorts.ACTIVE:
                raise KStepError("I3: draft completion s%d (epoch %d) of request incarnation (%s, %s) after its end"
                                 % (dr.s, dr.epoch, r.uid, r.inc))
        except KStepError:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise
        if dr.epoch != r.epoch:
            self.cohorts.stats["stale_drafts_ignored"] += 1
            return True
        if dr.state != self.RUNNING or dr.s != len(r.drafts) + 1:
            self.cohorts.stats["duplicate_drafts_refused"] += 1
            raise OwnershipViolation("draft entry: draft s%d (epoch %d, state %s) of request incarnation (%s, %s) is not "
                                     "the RUNNING next draft s%d" % (dr.s, dr.epoch, dr.state, r.uid, r.inc,
                                                                     len(r.drafts) + 1))
        return False

    def end_entry(self, d, r) -> None:
        """Dispatcher._end_request's entry, BEFORE any mutation: refused (I3) for a stale Request object, an incarnation
        that is not live, or a second end. I19: an ACTIVE or a CANCELLING incarnation may end."""
        self._live_request(d, r, lambda: "the end of a request")
        try:
            if self.cohorts.state(r.uid, r.inc, lambda: "the end of a request") == FinishedCohorts.ENDED:
                raise KStepError("I3: request incarnation (%s, %s) ended twice" % (r.uid, r.inc))
        except KStepError:
            self.cohorts.stats["stale_callbacks_refused"] += 1
            raise

    def cancel_entry(self, d, r) -> None:
        """I19, the dispatcher's cancellation entry (Dispatcher.cancel), BEFORE any mutation: refused (KStepError, I3;
        counted as stale_callbacks_refused) for a stale Request object, an incarnation that is not live, an ended one
        and a second cancel (FinishedCohorts.cancel's texts); then FinishedCohorts.cancel (ACTIVE -> CANCELLING)."""
        self._live_request(d, r, lambda: "the cancellation of a request")
        try:
            self.cohorts.cancel(r.uid, r.inc)
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

    def before_reset(self) -> None:
        """I4: call before any backend reset. Refuses (OwnershipViolation) while anything is outstanding; otherwise
        clears the identity-keyed caches. Never waits, forces or drops a reference. NECESSARY, NOT SUFFICIENT: see
        section 7 of the module docstring (handles installed outside the table)."""
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


# -------------------------------------------------------------------------------------------- milestone 3-4: I8, I10
class QuiescentReset:
    """Section 8 of the module docstring (I8, I10, I12): the ONE reset path that may let pooled events be re-recorded.
    disp: the Dispatcher (it must have the HOOK-L3 methods quiesce_state, event_holders and detach_completed_events,
    and the HOOK-L7 session API); backend_reset: the callable that resets the backend (for example CudaBackend.reset);
    it runs last. pending() is READ-ONLY. reset() refuses a latched failed session at once; refuses with no change
    while anything is outstanding (the session stays "open"); else begin_reset, the named mutating phases (detach, the
    executor's references first; post_check; before_reset; clear_caches; backend_reset), end_reset. A failure in a
    phase latches the session "failed". session_end(): finish, ONE host poll, reset(). Counts: resets (passed
    resets), detached (references the hook set to None, the executor's included)."""

    HOOKS = ("quiesce_state", "event_holders", "detach_completed_events")
    SESSION_ATTRS = ("session", "session_reason")                                      # HOOK-L7 (I12): attributes
    SESSION_API = SESSION_ATTRS + ("begin_reset", "end_reset", "fail_reset", "host_poll", "finish")   # and methods
    TABLE_PARTS = ("components", "handles", "joins")
    DRAIN_PHASES = ("finish", "host_poll")                                              # I12.1: session_end's drain
    PHASES = ("detach", "post_check", "before_reset", "clear_caches", "backend_reset")  # I12 (3): reset() runs these
    OPEN, RESETTING, FAILED = "open", "resetting", "failed"                              # in this order (_run_<phase>)
    LATCHED = "the session is latched failed: no resume, no admission, no reset retry"

    def __init__(self, disp, backend_reset: Callable[[], Any]):
        # I10 (HOOK-L4): the readers of an executor's declaration and of its contract methods. Imported here, so that
        # the milestone 1-2 parts of this module do not need them.
        from .executor import RESET_CONTRACT, reset_contract_of, reset_methods_missing
        missing = [h for h in self.HOOKS if not callable(getattr(disp, h, None))]
        session = [h for h in self.SESSION_API
                   if not (hasattr(disp, h) if h in self.SESSION_ATTRS else callable(getattr(disp, h, None)))]
        faults, tags = [], set()
        if missing:
            faults.append("the dispatcher lacks the HOOK-L3 methods %s" % (missing,))
            tags.add("I8")
        if session:
            faults.append("the dispatcher lacks the HOOK-L7 session API %s" % (session,))
            tags.add("I12")
        if not callable(backend_reset):
            faults.append("backend_reset is not callable (%s)" % (type(backend_reset).__name__,))
            tags.add("I8")
        ex = getattr(disp, "ex", None)
        if reset_contract_of(ex) == RESET_CONTRACT:
            bad = reset_methods_missing(ex)                     # the classes of type(ex).__mro__ only (no forwarding)
            if bad:
                faults.append("the executor %s declares reset_contract %r without the class methods %s"
                              % (type(ex).__name__, RESET_CONTRACT, bad))
                tags.add("I10")
        if faults:                                              # names only the actual fault(s); tags in a fixed order
            raise ConfigError("quiescent reset: %s (%s)" % ("; ".join(faults),
                                                            ", ".join(t for t in ("I8", "I10", "I12") if t in tags)))
        self.d = disp
        self.backend_reset = backend_reset
        self.contract_of, self.contract = reset_contract_of, RESET_CONTRACT
        self.resets = 0
        self.detached = 0

    def pending(self) -> Dict[str, Any]:
        """READ-ONLY (no poll, no wait, no change): every outstanding item, by owner class. Returns state (the nonzero
        quiesce_state counts), holders (owner class -> incomplete references), first (owner class -> the location of its
        first incomplete reference, for diagnosis), frontier (the nonzero table parts), providers ({(uid, inc): {name:
        count}}, nonzero only), store (True when a store exists and is not drained), reservations (the open
        reservations of Dispatcher.tickets, I15; 0 without tickets), contract (the executor's own declaration,
        executor.reset_contract_of; None when undeclared) and items (the refusal text items, in the fixed order of
        section 8; empty when the reset may proceed). A declared executor's operational events come through
        event_holders() (HOOK-L4), so its owner classes are holder items like any other."""
        d = self.d
        contract = self.contract_of(getattr(d, "ex", None))           # I10: the own class body only
        state: Dict[str, int] = {}
        for k, v in d.quiesce_state().items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise TypeError("quiescent reset: quiesce_state()[%r] is %r, not a count >= 0" % (k, v))
            if v:
                state[k] = v
        holders: Dict[str, int] = {}
        first: Dict[str, Any] = {}
        for cls, loc, ev in d.event_holders():
            if not ev_done(ev):
                holders[cls] = holders.get(cls, 0) + 1
                first.setdefault(cls, loc)
        fo = getattr(d, "frontier", None)
        table: Dict[str, int] = {}
        providers: Dict[Tuple[Any, Any], Dict[str, int]] = {}
        if fo is not None:
            p = fo.reset_pending()
            table = {k: p[k] for k in self.TABLE_PARTS if p[k]}
            providers = p["holds"]
        store = getattr(d, "store", None)
        store_out = store is not None and not store.drained_all()
        tk = getattr(d, "tickets", None)                               # I15: the result tickets' OPEN reservations
        tk_pending = getattr(tk, "pending", None)                      # (read-only; their count is also part of the
        tkp = tk_pending() if callable(tk_pending) else {}             # lead's "tickets"); I19: the OPEN
        reservations = len(tkp.get("reservations", ()))                # cancellation marks (an open mark implies
        cancels = len(tkp.get("cancels", ()))                          # an outstanding ticket: named, not new)
        items = (["state:%s=%d" % (k, state[k]) for k in sorted(state)]
                 + ["holder:%s=%d" % (c, holders[c]) for c in sorted(holders)]
                 + ["frontier:%s=%d" % (k, table[k]) for k in self.TABLE_PARTS if k in table]
                 + ["provider:%s(%s,%s)=%d" % (name, inc_[0], inc_[1], n)
                    for inc_ in sorted(providers, key=repr) for name, n in sorted(providers[inc_].items())]
                 + (["store:not_drained=1"] if store_out else [])
                 + (["tickets:reservations=%d" % reservations] if reservations else [])
                 + (["tickets:cancels=%d" % cancels] if cancels else [])
                 + (["executor:no_reset_contract=1"] if contract != self.contract else []))
        return dict(state=state, holders=holders, first=first, frontier=table, providers=providers, store=store_out,
                    reservations=reservations, cancels=cancels, contract=contract, items=items)

    def _refuse_if_failed(self) -> None:
        """I12 (0): a latched failed session refuses at once (reads only Dispatcher.session / session_reason: no hook,
        store or backend call)."""
        if self.d.session == self.FAILED:
            raise OwnershipViolation("quiescent reset refused: the session is latched failed (%s)"
                                     % (self.d.session_reason,))

    @staticmethod
    def _text(e: BaseException) -> str:
        """"<type>: <text>" of an exception. str(e) can raise (any BaseException, also KeyboardInterrupt from a
        __str__); then repr(e), which can raise too; then the fixed placeholder "<unprintable <type>>". A BaseException
        raised by str() or repr() here is NOT propagated: the failure path goes on with the placeholder."""
        try:
            msg = str(e)
        except BaseException:
            try:
                msg = repr(e)
            except BaseException:
                msg = "<unprintable %s>" % type(e).__name__
        return "%s: %s" % (type(e).__name__, msg)

    def _latch(self, phase: str, text: str) -> Optional[str]:
        """Latches the session "failed" for `phase` through the lead's hooks, as far as its state allows:
          "resetting"  fail_reset(phase, text) (a mutating phase of reset(): the normal case);
          "open"       begin_reset, then fail_reset(phase, text) (a drain phase of session_end: the normal case; or a
                       mutating phase that reopened the session);
          "failed"     already latched: nothing is called and its reason is kept;
          other        nothing is called.
        Returns None when the session is "failed" afterwards, else why it is not (an Exception that a hook raised here
        is caught and named; the session's state is named).
        NOT COVERED: a BaseException that is not an Exception (for example KeyboardInterrupt) raised by begin_reset or
        fail_reset THEMSELVES while they latch. It is not caught here: it propagates unchanged out of the failing
        phase (with the phase's own exception as its __context__), no later phase runs, and the session can stay
        unlatched: "open" (begin_reset did not finish; then the lead's guards still allow submit, step, finish and
        host_poll) or "resetting" (fail_reset did not finish; then those guards refuse, and so does begin_reset, so
        reset() and session_end() refuse too). A BaseException from str() / repr() of the phase's exception is
        covered (_text)."""
        d = self.d
        try:
            if d.session == self.OPEN:
                d.begin_reset()
            if d.session == self.RESETTING:
                d.fail_reset(phase, text)
        except Exception as e:                         # a BaseException that is not an Exception propagates (above)
            return "%s; the session is %r" % (self._text(e), d.session)
        return None if d.session == self.FAILED else "the session is %r" % (d.session,)

    def _failure(self, phase: str, text: str, why: Optional[str]) -> str:
        tail = self.LATCHED if why is None else "the session could NOT be latched failed: %s" % why
        return "quiescent reset failed in phase %s: %s (%s)" % (phase, text, tail)

    def _phase(self, phase: str, fn: Callable[[], Any], expect: str = "resetting") -> Any:
        """One named phase: a mutating phase of reset() (I12 (3); the session is "resetting") or a drain phase of
        session_end() (I12.1; expect "open"). fn does every attribute lookup itself, so a missing attribute is a fault
        of the phase. Any exception latches the session (_latch(phase, "<type>: <text>"): from "resetting" by
        fail_reset; from "open", in a drain phase, by begin_reset and then fail_reset) and is raised again as the named
        OwnershipViolation (from it); a BaseException that is not an Exception (for example KeyboardInterrupt) latches
        the session too and is raised again unchanged. A phase that returns but leaves the session other than `expect`
        is a fault too: the session is latched as far as its state allows (_latch), and OwnershipViolation "quiescent
        reset failed in phase <phase>: OwnershipViolation: the phase left the session <state> (expected <expect>)
        (...)" is raised. The text in brackets says whether the session is latched."""
        try:
            out = fn()
        except BaseException as e:
            text = self._text(e)
            why = self._latch(phase, text)
            if not isinstance(e, Exception):
                raise
            raise OwnershipViolation(self._failure(phase, text, why)) from e
        if self.d.session != expect:
            text = "OwnershipViolation: the phase left the session %r (expected %r)" % (self.d.session, expect)
            raise OwnershipViolation(self._failure(phase, text, self._latch(phase, text)))
        return out

    # I12.1: the drain phases of session_end, one method each, in the order of DRAIN_PHASES (each looks its target up
    # when it runs). Both start and must end with the session "open" (the lead's guards refuse finish and host_poll
    # outside "open", so begin_reset is NOT called before them).
    def _run_finish(self) -> None:
        self.d.finish()

    def _run_host_poll(self) -> None:
        self.d.host_poll()

    # I12 (3): the mutating phases, one method each, in the order of PHASES. Each one looks its target up when it runs.
    def _run_detach(self) -> int:
        return self.d.detach_completed_events()

    def _run_post_check(self) -> None:
        left = [(cls, loc) for cls, loc, _ in self.d.event_holders()]
        if left:                                                       # I10: complete references detach BEFORE reuse
            raise OwnershipViolation("quiescent reset: the detach left %d event references: %s; backend_reset did not "
                                     "run (I10)" % (len(left), left[:6]))

    def _run_before_reset(self) -> None:
        fo = getattr(self.d, "frontier", None)
        if fo is not None:
            fo.before_reset()

    def _run_clear_caches(self) -> None:
        store = getattr(self.d, "store", None)
        if store is not None:
            store.clear_caches()

    def _run_backend_reset(self) -> None:
        self.backend_reset()

    def reset(self) -> None:
        """I12 (section 8): (0) a latched failed session refuses at once, with no hook, store or backend call; (1)
        pending(), READ-ONLY: a refusal names every outstanding owner class, each once with its count, and changes
        nothing (the session stays "open"; so does an exception that the check itself raises); (2) begin_reset; (3) the
        named mutating phases detach (the executor's own detach first, verified by the hook, HOOK-L4), post_check (no
        event_holders() entry left), before_reset (the frontier's identity caches), clear_caches (the store's),
        backend_reset, run from PHASES (_phase: every lookup inside the phase; the session must still be "resetting"
        after each one); (4) end_reset and the counts. A failure in a phase latches the session "failed" (_latch, as
        far as the session's state allows; section 8) and raises "quiescent reset failed in phase <phase>: ...". If
        begin_reset leaves the session other than "resetting", reset() refuses before any phase. A phase failure is
        NOT a no-change refusal and NOT a rollback:
        the earlier phases (and the failing one up to its fault) already changed state; a detach that fails the hook's
        verification may already have changed the executor's own state (the dispatcher is unchanged then); after a
        later fault the complete references are already set to None. Only the coordinator's counts stay unchanged."""
        self._refuse_if_failed()                                       # (0)
        p = self.pending()                                             # (1) READ-ONLY
        if p["items"]:
            raise OwnershipViolation("quiescent reset refused: %s" % "; ".join(p["items"]))
        d = self.d
        d.begin_reset()                                                # (2) "open" -> "resetting"
        if d.session != self.RESETTING:                                # a hook defect; no mutating phase has run
            raise OwnershipViolation("quiescent reset refused: begin_reset left the session %r (expected %r); no "
                                     "mutating phase ran" % (d.session, self.RESETTING))
        out = {phase: self._phase(phase, getattr(self, "_run_" + phase)) for phase in self.PHASES}   # (3), in order
        d.end_reset()                                                  # (4) "resetting" -> "open", then the counts
        self.resets += 1
        self.detached += out["detach"]

    def session_end(self) -> None:
        """I12 / I12.1: the explicit session-end sequence. (0) A latched failed session refuses at once, as reset()
        does; a session that is not "open" (for example "resetting") refuses with no change. (1) The named DRAIN phases
        "finish" (Dispatcher.finish: drain every stream, validate every read, the D2H checks) and "host_poll" (ONE
        explicit host poll), in that order, through _phase with the session "open": finish and host_poll can change
        state and then raise (a validation after the synchronize; an arrival removed from inflight before its
        readback), so any exception latches the session "failed" from "open" (begin_reset, then fail_reset(<phase>,
        "<type>: <text>")) and raises the named OwnershipViolation from it; a drain phase that returns but leaves the
        session other than "open" is a fault too. (2) reset(): its read-only check; a refusal there leaves the session
        "open". No waiting loop and no retry: whatever is still outstanding after the one poll is reset()'s refusal."""
        self._refuse_if_failed()
        if self.d.session != self.OPEN:
            raise OwnershipViolation("quiescent reset refused: session_end needs the session 'open', not %r (%s)"
                                     % (self.d.session, self.d.session_reason))
        for phase in self.DRAIN_PHASES:
            self._phase(phase, getattr(self, "_run_" + phase), self.OPEN)
        self.reset()
