"""Shared CPU LOCKSTEP FIXTURES of the residency service (docs/R2_STEP7_FETCH_RETENTION.md note 12m; C's B2 review,
docs/R2_OWN2_B2_REVIEW_C.md sections 2 and 3).

The transitions that the lockstep schedules (F6, F12b, F20b, F21b, F22) never reach, as short SCRIPTS that any
service with the residency interface can run: the vendored v1 (under the S-C schedule), S-C dense, S-C sparse (B1),
and B's B2. This module needs only torch. The caller gives the worlds, the snapshot and the Row class.

A script is a list of OPERATIONS. `run` applies each operation to every world in lockstep. After each operation it
compares, across the worlds: the exception raised (by its class name), the value returned, and the snapshot. The
first difference ends the run; its operation index is the first field of the difference.

THE OPERATIONS (every world gets the same one):
  ("begin", b, uid[, hist_blocks])       begin_request
  ("end", uid)                           end_request
  ("tick", rows, reader)                 ONE TICK: poll; every layer resolves the same rows with the same selection
                                         and gets the reader event (read_done); end_tick. Its VALUE is the page
                                         tables: per layer, row and head, the slots listed (num_pages of them).
                                         rows: [(b, kind, key, blocks of head 0, blocks of head 1[, tail])];
                                         tail: [(block, slot or "TRASH")]; pos = 1000 + op index; hist_bound = HB.
                                         reader: an event NAME, pending until a "done" operation; or None (done).
                                         An unnamed reader is named "t<op index>".
  ("poll",)                              poll
  ("land",)                              force every transfer still in the transport, then poll (the world's own
                                         `land()` if it has one)
  ("done", name)                         the named reader event completes
  ("decided", key) / ("cancel", key)     decided / cancel
  ("cancel_epoch", uid, epoch)           cancel_epoch (its return value is compared)
  ("history", uid, block, name or None)  history_written with a named (pending) or a done event
  ("watch", name, value) / ("flag", name, value)
                                         watch_flag with a new 0-dim bool flag / set that flag in place (S-C only)
  ("raise", exc_name, op)                op must raise exc_name in every world. A TERMINAL raise (its index is in
                                         fixture.terminal) is the last operation: the services may differ after it
                                         by design (v1 refuses inside resolve, S-C at end_tick before any change).
                                         A non-terminal raise happens before any change in every world.

EACH FIXTURE has its capacities (C, X, R_ver), its worlds, a PRECONDITION pre(result, first world) (the transition
happens, counted), and its PLANTS: a defect in one world that the lockstep must catch at a predicted operation, by a
predicted field ("raise"; "value": a tick's page tables; or a field of the vendored snapshot: state, counters,
readers, forward, journal, gates, mirror, trash). The preconditions read the first world's snapshots and service: put a vendored world first.

REVIEW KINDS (docs/R2_OWN2_B2_REVIEW_C.md section 3, "MISSING KINDS"): Plant.kind "1".."8" names them; the others
are the vendored tests' own controls (F3, F12e, F12f, F13, F14b, F15, F16, T3) and one B1 control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

K1, K2 = ("u0", 1, 1), ("u0", 1, 2)          # two cohort keys of request u0, epoch 1
J1, J2 = ("u1", 1, 1), ("u1", 1, 2)          # two cohort keys of request u1, epoch 1
HB = 30                                       # every row's hist_bound (S-C needs hist_bound <= nblk)


class Ev:
    """A reader / write-back event of the CPU worlds (a done flag, as the vendored ev_done reads it)."""

    def __init__(self, done: bool = False):
        self.done = done


def sel_tensor(H: int, heads: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
    """(H, R, k) int64, -1 padded: heads[i][h] = the blocks of row i at head h (heads past the given ones: none)."""
    k = max([len(bl) for r in heads for bl in r] + [1])
    t = torch.full((H, len(heads), k), -1, dtype=torch.int64)
    for i, r in enumerate(heads):
        for h in range(min(H, len(r))):
            if r[h]:
                t[h, i, :len(r[h])] = torch.tensor(list(r[h]), dtype=torch.int64)
    return t


@dataclass
class Plant:
    name: str                 # the test id; the caller maps it to the defective world
    kind: str                 # "1".."8" (the review's missing kinds) or the vendored control it repeats
    world: str                # the world that carries the defect
    at: int                   # the operation where the lockstep reports the FIRST difference
    field: str                # "raise", "value", or the first differing field of the vendored snapshot
    defect: str               # the defect, in words


@dataclass
class Fixture:
    name: str
    C: int
    X: int
    R_ver: int
    ops: List[Tuple]
    pre: Callable[["Result", Any], bool]
    doc: str
    plants: Tuple[Plant, ...] = ()
    worlds: Tuple[str, ...] = ("v1", "sc", "sparse")
    trash: int = 0
    terminal: Tuple[int, ...] = ()


@dataclass
class Result:
    diffs: List[Tuple] = field(default_factory=list)            # (op index, "raise" | "value" | "snapshot", detail)
    raised: List[Optional[str]] = field(default_factory=list)   # per op: the exception name of the first world
    values: List[Any] = field(default_factory=list)             # per op: the value of the first world
    snaps: List[Any] = field(default_factory=list)              # per op: the snapshot of the first world
    tags: Dict[int, str] = field(default_factory=dict)          # id(event) -> its name (for the snapshots)


def _land(w) -> None:
    own = getattr(w, "land", None)
    if callable(own):
        own()
        return
    for tr in list(w.svc.inflight.values()):
        if not tr.delivered and tr in w.tp.pending:
            w.tp.force(tr)
    w.svc.poll()


def _apply(op: Tuple, w, row_cls, ctx: Dict, tags: Dict[int, str], i: int):
    svc = w.svc
    kind = op[0]
    if kind == "begin":
        return svc.begin_request(op[1], op[2], *op[3:])
    if kind == "end":
        return svc.end_request(op[1])
    if kind == "poll":
        return svc.poll()
    if kind == "land":
        return _land(w)
    if kind == "done":
        ctx["evs"][op[1]].done = True
        return None
    if kind == "decided":
        return svc.decided(op[1])
    if kind == "cancel":
        return svc.cancel(op[1])
    if kind == "cancel_epoch":
        return svc.cancel_epoch(op[1], op[2])
    if kind == "history":
        ev = Ev(done=op[3] is None)
        ctx["keep"].append(ev)
        if op[3] is not None:
            ctx["evs"][op[3]] = ev
        return svc.history_written(op[1], op[2], ev)
    if kind == "watch":
        flag = torch.tensor(bool(op[2]))
        ctx["flags"][op[1]] = flag
        return svc.watch_flag(op[1], flag)
    if kind == "flag":
        ctx["flags"][op[1]].fill_(bool(op[2]))
        return None
    if kind == "tick":
        spec, reader = op[1], op[2]
        name = reader if reader is not None else "t%d" % i
        ev = Ev(done=reader is None)
        ctx["evs"][name] = ev
        ctx["keep"].append(ev)
        tags[id(ev)] = name
        rows = []
        for r in spec:
            tail = [(blk, svc.trash_slot if s == "TRASH" else s) for (blk, s) in (r[5] if len(r) > 5 else [])]
            rows.append(row_cls(r[0], r[1], r[2], 1000 + i, HB, tail))
        sel = sel_tensor(svc.H, [(r[3], r[4]) for r in spec])
        svc.poll()
        pages = []
        for l in range(svc.L):
            p = svc.resolve(l, rows, sel)
            svc.read_done(l, p, ev)
            pages.append([[p.page_table[j, h, :int(p.num_pages[j, h])].tolist() for h in range(svc.H)]
                          for j in range(len(rows))])
        svc.end_tick()
        return pages
    raise ValueError("unknown operation %r" % (kind,))


def run(fx: Fixture, worlds: Dict[str, Any], snapshot: Callable[[Any, Dict[int, str]], Any], row_cls) -> Result:
    """Apply every operation of fx to every world (in the dict's order; the first is the reference) in lockstep.
    After each operation: the raised exception's class name must be the same in every world (and the expected one
    for a "raise" operation, None otherwise); the returned values must have the same repr; the snapshots must be
    equal. A snapshot that is a dict names its first differing field. The first difference ends the run."""
    res = Result()
    names = list(worlds)
    ctx = {n: dict(evs={}, flags={}, keep=[]) for n in names}
    for i, op in enumerate(fx.ops):
        want, inner = (op[1], op[2]) if op[0] == "raise" else (None, op)
        got, vals = {}, {}
        for n in names:
            try:
                vals[n] = _apply(inner, worlds[n], row_cls, ctx[n], res.tags, i)
                got[n] = None
            except Exception as e:                       # noqa: BLE001 - compared by class name across the worlds
                got[n] = type(e).__name__
                vals[n] = str(e)[:300]
        res.raised.append(got[names[0]])
        res.values.append(vals[names[0]] if got[names[0]] is None else None)
        if len(set(got.values())) > 1 or got[names[0]] != want:
            res.diffs.append((i, "raise", dict(want=want, got=got, msg={n: vals[n] for n in names if got[n]})))
            break
        if want is not None and i in fx.terminal:
            res.snaps.append(None)
            break
        if want is None and len({repr(v) for v in vals.values()}) > 1:
            res.diffs.append((i, "value", vals))
            break
        snaps = {n: snapshot(worlds[n], res.tags) for n in names}
        res.snaps.append(snaps[names[0]])
        ref = snaps[names[0]]
        for n in names[1:]:
            if snaps[n] != ref:
                fld = next((f for f in ref if snaps[n].get(f) != ref[f]), "?") if isinstance(ref, dict) else "?"
                res.diffs.append((i, "snapshot", dict(world=n, field=fld)))
                break
        if res.diffs:
            break
    return res


# =====================================================================================================================
# precondition helpers (the first world's snapshots and service)
# =====================================================================================================================
def jdelta(res: Result, i: int, kind: str) -> int:
    """The transitions of journal kind `kind` that operation i made (from the snapshots' journal counts)."""
    now = dict(res.snaps[i]["journal"]).get(kind, 0)
    before = dict(res.snaps[i - 1]["journal"]).get(kind, 0) if i > 0 else 0
    return now - before


def block_log(w, l: int, b: int, h: int, block: int) -> List[str]:
    """The transition kinds of one block in one stream's vendored log (Stream.log), in order."""
    return [e[0] for e in w.svc.pools[l].stream(b, h).log if e[1] == block]


# =====================================================================================================================
# the fixtures
# =====================================================================================================================
def _fixtures() -> Dict[str, Fixture]:
    fx: Dict[str, Fixture] = {}

    # (a) F13 / F14b plus a land: end_request, the next request, drop_unowned --------------------------------------
    fx["end_request"] = Fixture(
        "end_request", C=4, X=4, R_ver=1,
        doc="u0 drafts block 5 (lands: LRU) and block 6 (stays in flight; op 4 asks for it again: the dedup merges, "
            "no take). u0 ends (op 5): its LRU entry is released ('end'), the transfer stays in flight. u1 begins "
            "on the same batch slot; its verifier selects block 6 (op 7): u0's transfer is not u1's, so u1 takes its "
            "own urgent slot (the freed slot 0), never merges or waits (F13). The final land (op 8) drops u0's "
            "arrival (drop_unowned: RELEASING with no reader, FREE in the same poll).",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [5], [])], None),
             ("land",),
             ("tick", [(0, "d", K1, [6], [])], None),
             ("tick", [(0, "d", K1, [6], [])], None),
             ("end", "u0"),
             ("begin", 0, "u1"),
             ("tick", [(0, "v", J1, [6], [])], None),
             ("land",)],
        pre=lambda r, w: (jdelta(r, 4, "take") == 0 and jdelta(r, 5, "release:end") >= 1
                          and jdelta(r, 8, "drop_unowned") >= 1 and w.svc.metrics.k43["new"] >= 1
                          and w.svc.metrics.k43["in_flight"] == 0),
        plants=(
            Plant("dup_take_device", "1", "sc", 4, "raise",
                  "the device's draft decision does not see the block in flight (S-B no_dedup): a second take of "
                  "block 6; the host's take check refuses it"),
            Plant("dup_take_host", "1", "v1", 4, "state",
                  "the host's in-flight lookup finds nothing (NoDedup): a second transfer of block 6"),
            Plant("owner_not_mirrored", "F14b", "sc", 5, "mirror",
                  "the owner change is not sent with the mirror delta: the device still sees u0's transfer as the "
                  "owner's"),
            Plant("owner_blind", "F13", "v1", 7, "value",
                  "the in-flight lookup ignores the owner: u1's verifier waits for u0's transfer and reads its slot "
                  "(page 1, not its own urgent slot 0)"),
            Plant("no_uid_check", "T3", "sc", 8, "raise",
                  "an arrival is admitted whatever request owns the slot: u0's block enters u1's LRU, then u1's own "
                  "arrival of the block is refused (already resident)"),
        ))

    # (b) F3 and F11: cancel, a refused cancelled key, late arrivals, the late-protection guard, cancel_epoch -------
    fx["cancel"] = Fixture(
        "cancel", C=8, X=2, R_ver=1,
        doc="K1 drafts three blocks (in flight) and is cancelled: its transfers keep their slots; K2 (same epoch) "
            "still resolves (F11's single cancel); a K1 row is refused before any change (op 4). The late arrivals "
            "enter the LRU (arrivals after a cancel). K2's verify read queues its (key, l) releases; a later K2 row is "
            "refused by the late-protection guard (op 7). cancel_epoch cancels the epoch's remaining key K2 (op 8).",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [5, 6], [7])], None),
             ("cancel", K1),
             ("tick", [(0, "d", K2, [10, 11], [10])], None),
             ("raise", "OwnershipViolation", ("tick", [(0, "d", K1, [5], [])], None)),
             ("land",),
             ("tick", [(0, "v", K2, [10], [])], None),
             ("raise", "OwnershipViolation", ("tick", [(0, "d", K2, [11], [])], None)),
             ("cancel_epoch", "u0", 1)],
        pre=lambda r, w: (r.raised[4] == "OwnershipViolation" and r.raised[7] == "OwnershipViolation"
                          and r.values[8] == [K2] and w.svc.report()["arrivals_after_cancel"] >= 1),
        plants=(
            Plant("free_on_cancel", "F3", "sc", 2, "state",
                  "the cancel frees the in-flight destinations (K3X free_on_cancel)"),
            Plant("cancel_refusal_off", "3", "v1", 4, "raise",
                  "a row of a cancelled key is accepted: the cancelled-key check AND the late-protection guard are "
                  "off (either one alone still refuses it: cancel() queues the key's release at every layer)"),
            Plant("late_guard_off", "3", "sc", 7, "raise",
                  "a row whose (key, l) release is queued is accepted (K3X no_late_protection_guard)"),
        ))

    # (c) a 2-tick C-2 script: two draft rows of one stream in one layer (I8's O1 rounds) ---------------------------
    fx["o1_two_rows"] = Fixture(
        "o1_two_rows", C=2, X=2, R_ver=0,
        doc="Tick 1: K1 drafts blocks 0-3 (2 local slots, 2 borrowed); they land: LRU [2, 3], PO {0, 1} (protected). "
            "Tick 2 has TWO draft rows of K1 in one stream: row 1 reads block 0 (PO), row 2 reads block 2 (LRU); both "
            "are READY before the recency (O1). Per row (K3X O1): row 1 acquires 0 and its evict_to_c moves 2 to PO; "
            "row 2 acquires 2 back and its evict_to_c moves 3 to PO. Final LRU [0, 2], PO {1, 3}. ONE batch eviction "
            "after both rows reaches the SAME state (acquire 0, touch 2, evict 3): only the transitions differ.",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [0, 1, 2, 3], [])], None),
             ("land",),
             ("tick", [(0, "d", K1, [0], []), (0, "d", K1, [2], [])], None)],
        pre=lambda r, w: (block_log(w, 0, 0, 0, 2)[-2:] == ["evict_po", "acquire"]
                          and list(w.svc.pools[0].stream(0, 0).lru) == [0, 2]
                          and sorted(w.svc.pools[0].stream(0, 0).po) == [1, 3]),
        plants=(
            Plant("batch_recency", "8", "v1", 3, "journal",
                  "the layer's draft recency evicts once per stream after ALL rows (one batch eviction): the state is "
                  "equal, the journal is not (acquire 1 + touch 1 + evict_po 1 instead of acquire 2 + evict_po 2)"),
        ))

    # (d) void ticks, T4, the trash slot, non-monotone readers -------------------------------------------------------
    fx["void_overflow"] = Fixture(
        "void_overflow", C=1, X=1, R_ver=1, terminal=(1,),
        doc="F12f: a verify row needs 3 slots where K5 grants 2 and Policy E has no candidate: CapacityRefusal (v1 in "
            "resolve after two takes, S-C at end_tick before any change): a void tick, terminal.",
        ops=[("begin", 0, "u0"),
             ("raise", "CapacityRefusal", ("tick", [(0, "v", K1, [0, 1, 2], [])], None))],
        pre=lambda r, w: r.raised[1] == "CapacityRefusal",
        plants=(
            Plant("no_overflow_flag", "F12f", "sc", 1, "raise",
                  "the device does not count the overflow: the hole reaches the host (OwnershipViolation)"),
        ))
    fx["void_flag"] = Fixture(
        "void_flag", C=4, X=4, R_ver=1, worlds=("sc", "sparse"),
        doc="F16: a watched device flag set at a tick voids THAT tick (CapacityRefusal at its end_tick, before any "
            "change: the state after it is the state before it); the flag cleared, the same rows apply at the next "
            "tick. S-C only (v1 has no device flags). B1: the void tick's pins are recorded again at the next tick.",
        ops=[("begin", 0, "u0"),
             ("watch", "twin_overflow", False),
             ("tick", [(0, "d", K1, [5], [])], None),
             ("flag", "twin_overflow", True),
             ("raise", "CapacityRefusal", ("tick", [(0, "d", K1, [6], [])], None)),
             ("flag", "twin_overflow", False),
             ("tick", [(0, "d", K1, [6], [])], None)],
        pre=lambda r, w: (r.raised[4] == "CapacityRefusal" and r.snaps[4]["state"] == r.snaps[3]["state"]
                          and r.snaps[4]["counters"] == r.snaps[3]["counters"]
                          and any(tr.block == 6 for tr in w.svc.inflight.values())),
        plants=(
            Plant("flag_ignored", "F16", "sc", 4, "raise", "the watched flags are never read"),
            Plant("void_keeps_pins", "B1", "sparse", 6, "state",
                  "a void tick keeps the device pin table: the next tick's pin of block 6 is not recorded again, so "
                  "the host never protects it"),
        ))
    fx["t4"] = Fixture(
        "t4", C=4, X=4, R_ver=1, terminal=(3,),
        doc="T4 (F11's history events): u0's prefill wrote blocks 0-3 (hist_blocks 4). Block 5 has a registered host "
            "write-back: its take carries the event (the transport waits on it). Block 6 has none: its take is "
            "refused (OwnershipViolation; v1 in resolve, S-C at end_tick), terminal.",
        ops=[("begin", 0, "u0", 4),
             ("history", "u0", 5, None),
             ("tick", [(0, "d", K1, [5], [])], None),
             ("raise", "OwnershipViolation", ("tick", [(0, "d", K1, [6], [])], None))],
        pre=lambda r, w: r.raised[3] == "OwnershipViolation" and any(f[3] == 5 and f[7] for f in r.snaps[2]["forward"]),
        plants=(
            Plant("no_t4_refusal", "2", "v1", 3, "raise",
                  "a take of a block past the prefill without a registered host write-back is not refused"),
        ))
    fx["trash"] = Fixture(
        "trash", C=4, X=4, R_ver=1, trash=1, terminal=(4,),
        doc="F15: with trash_slots=1, draft takes (bias fills), arrivals, a verify urgent gather and the polls never "
            "write the trash slot (its sentinel stays: snapshot field trash); a tail naming it is refused (ValueError; "
            "v1 in _finish_plan after its changes, S-C in _tick_rows before them), terminal. Op 4 uses K2: K1's "
            "release is queued (the late-protection guard would refuse K1 first).",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [1, 2], [3])], None),
             ("land",),
             ("tick", [(0, "v", K1, [1, 4], [3])], None),
             ("raise", "ValueError", ("tick", [(0, "d", K2, [1], [], [(10, "TRASH")])], None))],
        pre=lambda r, w: (r.raised[4] == "ValueError" and all(s["trash"] is True for s in r.snaps if s is not None)
                          and w.svc.metrics.k43["new"] >= 1),
        plants=(
            Plant("trash_written", "F15", "v1", 1, "trash", "a take's bias fill also writes the trash slot"),
        ))
    fx["nonmono"] = Fixture(
        "nonmono", C=2, X=4, R_ver=2,
        doc="F12e, NON-MONOTONE readers: u0's verifier reads slot 0 with a PENDING reader e1 (op 3); u0 ends (slots 0 "
            "and 1 RELEASING). u1's verifier misses block 5 with a reader that is done while e1 still pends (op 6): "
            "slot 1 is FREE (no reader), slot 0 is not: the urgent take uses slot 1. e1 completes; the next poll "
            "frees slot 0; u1's draft takes slot 0 again (op 9) and the slot keeps e1 as its last reader, forwarded "
            "to the transfer (T3). u1 ends, u2 begins, the transfer lands and is dropped (op 12): the drop resets the "
            "slot's last reader. A tick-watermark reader model (B2) must refuse this harness, not diverge silently.",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [0, 1], [])], None),
             ("land",),
             ("tick", [(0, "v", K1, [0], [])], "e1"),
             ("end", "u0"),
             ("begin", 0, "u1"),
             ("tick", [(0, "v", J1, [5], [])], None),
             ("done", "e1"),
             ("poll",),
             ("tick", [(0, "d", J2, [6], [])], None),
             ("end", "u1"),
             ("begin", 0, "u2"),
             ("land",)],
        pre=lambda r, w: (r.snaps[6]["gates"] >= 1 and (0, 0, 0, 0, "e1", "e1") in r.snaps[6]["readers"]
                          and (0, 0, 0, 1, "t6", "t6") in r.snaps[6]["readers"]
                          and any(f[3] == 6 and f[6] == "e1" for f in r.snaps[9]["forward"])
                          and jdelta(r, 12, "drop_unowned") >= 1),
        plants=(
            Plant("no_reader_wait", "F12e", "v1", 6, "value",
                  "RELEASING -> FREE without the last reader (K3X no_reader_wait): the urgent take reuses slot 0 "
                  "(page 0, not 1)"),
            Plant("lost_forward", "7", "v1", 9, "readers",
                  "a take drops the slot's last reader: the transfer is not ordered after e1 (the T3 forward lost)"),
            Plant("drop_keeps_reader", "6", "v1", 12, "readers",
                  "a dropped arrival (drop_unowned) keeps the slot's last reader"),
        ))
    fx["vres_drop"] = Fixture(
        "vres_drop", C=4, X=4, R_ver=1,
        doc="THE POLL ORDER (arrivals, then the queued drops, then the releases): a verifier's urgent block (landed in "
            "place) is admitted VRES at the next poll and that poll's drops release it (release:unprotect); the slot "
            "keeps the verify read as its last reader. Drops BEFORE arrivals would drop it unprotected "
            "(drop_vres_unprotected, last reader reset): the same slot states, other transitions and readers.",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "v", K1, [5], [])], None),
             ("poll",)],
        pre=lambda r, w: (jdelta(r, 2, "admit_vres") >= 1 and jdelta(r, 2, "release:unprotect") >= 1
                          and dict(r.snaps[2]["journal"]).get("drop_vres_unprotected", 0) == 0),
        plants=(
            Plant("poll_order_swapped", "5", "v1", 2, "readers", "poll runs the queued drops before the arrivals"),
        ))
    fx["key_alias"] = Fixture(
        "key_alias", C=4, X=4, R_ver=1,
        doc="Two live keys of one request pin the same block: K1's draft takes block 5, K2's draft merges into the "
            "transfer and pins it too (prot = {K1, K2}). B1's device pin table names keys by a key id: two keys that "
            "share an id alias (B2's key-slot reuse is the same defect class).",
        ops=[("begin", 0, "u0"),
             ("tick", [(0, "d", K1, [5], [])], None),
             ("tick", [(0, "d", K2, [5], [])], None)],
        pre=lambda r, w: set(w.svc.pools[0].stream(0, 0).prot.get(5, ())) == {K1, K2},
        plants=(
            Plant("key_alias", "4", "sparse", 2, "state",
                  "one key id per REQUEST in B1's pin table: K2's pin of block 5 looks recorded and is never applied"),
        ))
    return fx


FIXTURES: Dict[str, Fixture] = _fixtures()
REVIEW_KINDS = ("1", "2", "3", "4", "5", "6", "7", "8")
