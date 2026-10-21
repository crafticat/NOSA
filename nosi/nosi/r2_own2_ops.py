"""R2 step 7, piece 5, (B2): the vectorized operations of the ARRAY-BACKED ownership state (docs/R2_OWN2_B2_DESIGN.md
rev 2). SOURCE ONLY: B2-T (benchmarks/Efficiency/r2_own2_b2t.py) times them; OwnResidency.apply_tick will compose them.

THE STATE (Table): per slot (L, B, H, NPOS = C + X) the columns of SLOT_COLS; the PINS as bit planes of C's 32-bit
words (pin_w (2 KEYS, L, B, H, ceil(NBLK / 32)) int64: plane k = key slot k for verify, plane KEYS + k for draft; bit i
of word w = block 32 w + i, the layout of sparse.py _pack_bits); per request owner, key readers, the release queue,
cancelled key slots, the T4 write-back ids; xfer (the 4.8 transfer counts).

DEVICE-AGNOSTIC: a Table lives on one device (CPU or CUDA); every op creates its tensors on the table's device. The
report counts are returned as 0-dim tensors (no host read); the host reads that remain are the ones a shape or an early
exit needs (nonzero / a boolean mask / unique / int(max) / bool(any)): on CUDA each is a synchronization (B2-T reports
them by component).

THE READERS are EVENT IDS (rev 2 section 0.1 (1)): ReaderWindow maps an id to its event object and E's mark; done(e) is
exact for completion in any order. -1 = no reader (complete), as ev_done(None). Its host mirror is uploaded once per
change.

EVERY OPERATION is torch ops over whole columns or over the records, with no Python loop per entry. The loops that
remain are per ROUND (the recency rounds, the pin rounds: at most K + 1) or per live event (the window refresh).
Writes go through an Undo log, so a failed post-write predicate restores the state (a void tick).

The vendored rules each function implements are cited with file:line (res.py = r2_residency.py, own.py =
kstep/ownership.py; rev 2 section 3.1).
"""
from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from .kstep.errors import CapacityRefusal
from .r2_residency_sb import rank_in_group
from .r2_residency_sc import C_FREE, C_INFLIGHT, C_LRU, C_PO, C_RELEASING, C_VRES
from .r2_residency_sparse import (DUP, EVICT, EVICT_INFL, FIRST, GRANT, HOLE, INFL, PIN, READY,  # noqa: F401
                                   REFUSED, _first_in_group)

DEM_D, DEM_V = 1, 2          # demand bits: "draft", "verify" (Transfer.demand)
ISS_D, ISS_V = 1, 2          # issuer (Transfer.kind)
CLS_AD, CLS_V = 1, 2         # slot class "ad" / "v" (Slot.cls; res.py:480)
BIG = 1 << 62
HALF = 1 << 61               # Policy E: in-flight candidates sort after every LRU one (sc.py HALF_STAMP)
WBITS = 32                   # C's packed words carry 32 bits each (sparse.py _pack_bits)


class Geo(NamedTuple):
    L: int
    B: int
    H: int
    C: int
    X: int
    R_ver: int
    NBLK: int
    KEYS: int

    @property
    def NPOS(self) -> int:
        return self.C + self.X

    @property
    def S(self) -> int:
        return self.L * self.B * self.H

    @property
    def WB(self) -> int:
        return (self.NBLK + WBITS - 1) // WBITS

    @property
    def WP(self) -> int:
        return (self.NPOS + WBITS - 1) // WBITS


# (name, dtype, the value of an unused slot)
SLOT_COLS = (("code", torch.int8, C_FREE), ("block", torch.int32, -1), ("gen", torch.int32, 0),
             ("uid", torch.int32, -1), ("cls", torch.int8, 0), ("stamp", torch.int64, 0),
             ("rd_first", torch.int64, -1), ("rd_last", torch.int64, -1), ("seq", torch.int64, -1),
             ("issue_tick", torch.int32, -1), ("demand", torch.int8, 0), ("issuer", torch.int8, 0),
             ("req_key", torch.int8, -1), ("landed", torch.bool, False), ("hist_gate", torch.int64, -1),
             ("rd_n", torch.int32, 0), ("rv_n", torch.int32, 0))
ARRAYS = ("pin_w", "owner", "key_rd", "rel_q", "kcancel", "hist_from", "hist_id", "xfer")


class Table:
    """B2's state (module docstring), on one device."""

    def __init__(self, geo: Geo, device="cpu"):
        if geo.KEYS < 1 or geo.KEYS > 7:
            raise ValueError("KEYS must be 1..7, got %d" % geo.KEYS)
        self.geo = geo
        self.dev = torch.device(device)
        d = self.dev
        sh = (geo.L, geo.B, geo.H, geo.NPOS)
        self.c: Dict[str, torch.Tensor] = {n: torch.full(sh, v, dtype=t, device=d) for (n, t, v) in SLOT_COLS}
        self.pin_w = torch.zeros((2 * geo.KEYS, geo.L, geo.B, geo.H, geo.WB), dtype=torch.int64, device=d)
        self.owner = torch.full((geo.B,), -1, dtype=torch.int32, device=d)
        self.key_rd = torch.full((geo.L, geo.B, geo.KEYS), -1, dtype=torch.int64, device=d)
        self.rel_q = torch.zeros((geo.L, geo.B, geo.KEYS), dtype=torch.bool, device=d)
        self.kcancel = torch.zeros((geo.B, geo.KEYS), dtype=torch.bool, device=d)
        self.hist_from = torch.full((geo.B,), 1 << 40, dtype=torch.int64, device=d)   # no T4 check until begin_request
        self.hist_id = torch.full((geo.B, geo.NBLK), -1, dtype=torch.int64, device=d)
        self.xfer = torch.zeros((geo.B, geo.L, geo.H, geo.NBLK), dtype=torch.int32, device=d)
        self.clock = 1

    def tensors(self) -> Dict[str, torch.Tensor]:
        out = dict(self.c)
        for a in ARRAYS:
            out[a] = getattr(self, a)
        return out

    def copy_from(self, other: "Table") -> None:
        mine, theirs = self.tensors(), other.tensors()
        for n, t in mine.items():
            t.copy_(theirs[n])
        self.clock = other.clock

    def clone(self) -> "Table":
        t = Table(self.geo, self.dev)
        t.copy_from(self)
        return t

    def to(self, device) -> "Table":
        t = Table(self.geo, device)
        t.copy_from(self)
        return t

    def diff(self, other: "Table") -> List[str]:
        mine, theirs = self.tensors(), other.tensors()
        out = [n for n in mine if not torch.equal(mine[n], theirs[n].to(mine[n].device))]
        if self.clock != other.clock:
            out.append("clock")
        return out

    def flat(self, name: str) -> torch.Tensor:
        return self.c[name].view(-1)


class Undo:
    """The write log of one apply: put() saves the old values, rollback() restores them in reverse (a void tick)."""

    def __init__(self):
        self.log: List[Tuple] = []

    def put(self, t: torch.Tensor, idx: torch.Tensor, val) -> None:
        """t[idx] = val on a flat (contiguous) view t; idx must not repeat with different values."""
        self.log.append((t, idx, t[idx].clone()))
        t[idx] = val.to(t.dtype) if torch.is_tensor(val) else val

    def call(self, restore: Callable[[], None]) -> None:
        self.log.append((None, None, restore))

    def rollback(self) -> None:
        for (t, idx, old) in reversed(self.log):
            if t is None:
                old()
            else:
                t[idx] = old
        self.log = []


def _put(undo: Optional[Undo], t: torch.Tensor, idx: torch.Tensor, val) -> None:
    if undo is not None:
        undo.put(t, idx, val)
    else:
        t[idx] = val.to(t.dtype) if torch.is_tensor(val) else val


def unpack_words(words: torch.Tensor, n: int) -> torch.Tensor:
    """(..., W) int64 32-bit words -> (..., n) bool on the words' device (sparse.py unpack_bits is host-only)."""
    sh = torch.arange(WBITS, dtype=torch.int64, device=words.device)
    bits = torch.bitwise_and(torch.bitwise_right_shift(words.unsqueeze(-1), sh), 1) != 0
    return bits.reshape(*words.shape[:-1], -1)[..., :n]


# ---------------------------------------------------------------------------------------------------- readers
class ReaderWindow:
    """Reader event ids (rev 2 section 0.1 (1)). register() gives every read_done call (res.py:743-761) the next id;
    refresh() asks each LIVE event object (ev_done, kstep/backend.py:27-34), so completion in any order is exact; ids
    below lo are complete. mark: E's mark of the id (tick x L + layer), for issue_desc's reader_marks. The host mirror
    (win_c, mark_c) is the truth; the device copy is uploaded once per change (_sync)."""

    def __init__(self, cap: int = 1 << 14, device="cpu"):
        self.cap = int(cap)
        self.lo = 0
        self.next = 0
        self.dev = torch.device(device)
        self.win_c = torch.zeros(self.cap, dtype=torch.bool)
        self.mark_c = torch.full((self.cap,), -1, dtype=torch.int64)
        self.events: Dict[int, object] = {}
        if self.dev.type == "cpu":
            self.win, self.mark = self.win_c, self.mark_c
        else:
            self.win, self.mark = self.win_c.to(self.dev), self.mark_c.to(self.dev)
        self._dirty = False

    def _sync(self) -> None:
        if self._dirty and self.dev.type != "cpu":
            self.win.copy_(self.win_c)
            self.mark.copy_(self.mark_c)
        self._dirty = False

    def register(self, ev, mark: int) -> int:
        e = self.next
        if e - self.lo >= self.cap:
            raise CapacityRefusal("B2 reader window: %d live reader events >= cap %d" % (e - self.lo, self.cap))
        i = e % self.cap
        self.win_c[i] = False
        self.mark_c[i] = int(mark)
        self.events[e] = ev
        self.next += 1
        self._dirty = True
        return e

    def refresh(self, ev_done: Callable[[object], bool]) -> int:
        """Ask every live event; returns the number of live events after the refresh."""
        for e in range(self.lo, self.next):
            i = e % self.cap
            if not bool(self.win_c[i]) and ev_done(self.events.get(e)):
                self.win_c[i] = True
                self._dirty = True
        while self.lo < self.next and bool(self.win_c[self.lo % self.cap]):
            self.events.pop(self.lo, None)
            self.lo += 1
        self._sync()
        return self.next - self.lo

    def wait(self, e: int, host_wait: Callable[[object], None], ev_done: Callable[[object], bool]) -> None:
        """The host wait of ONE reader id (Policy E's pending reader, res.py:410-415): that event only."""
        if e < self.lo or e >= self.next:
            return
        ev = self.events.get(e)
        host_wait(ev)
        if ev_done(ev):
            self.win_c[e % self.cap] = True
            self._dirty = True
        self._sync()

    def done_of(self, ids: torch.Tensor) -> torch.Tensor:
        self._sync()
        i = ids.remainder(self.cap)
        return (ids < self.lo) | ((ids < self.next) & self.win[i])

    def mark_of(self, ids: torch.Tensor) -> torch.Tensor:
        """E's mark of each id, -1 when it is complete (nothing to wait for)."""
        return torch.where(self.done_of(ids), torch.full_like(ids, -1), self.mark[ids.remainder(self.cap)])


# ---------------------------------------------------------------------------------------------------- helpers
def any_pin_words(T: Table) -> torch.Tensor:
    """(S, WB) words: a block is pinned when ANY key plane holds its bit (Stream.prot non-empty)."""
    pw = T.pin_w
    acc = pw[0].clone()
    for p in range(1, pw.shape[0]):
        acc |= pw[p]
    return acc.view(T.geo.S, T.geo.WB)


def bit_test(words: torch.Tensor, s: torch.Tensor, blk: torch.Tensor) -> torch.Tensor:
    """words (S, W) 32-bit words; s, blk int64 of one shape: bit blk of stream s (blk >= 0)."""
    w = words[s, torch.div(blk, WBITS, rounding_mode="floor")]
    return torch.bitwise_and(torch.bitwise_right_shift(w, blk.remainder(WBITS)), 1) != 0


def stream_b(T: Table, s: torch.Tensor) -> torch.Tensor:
    g = T.geo
    return torch.div(s, g.H, rounding_mode="floor").remainder(g.B)


def owner_of_slots(T: Table) -> torch.Tensor:
    """(L, B, H, NPOS) bool: the slot belongs to its batch slot's CURRENT owner (S-C's cur; res.py:509-517)."""
    g = T.geo
    return T.c["uid"] == T.owner.view(1, g.B, 1, 1)


def slot_of(T: Table, with_counts: bool = False):
    """(S, NBLK) int32: the slot of each block that is RESIDENT (LRU / PO / VRES) or IN FLIGHT FOR THE CURRENT OWNER
    (rev 2 I1: res.py:453, own.py:266-267), else -1. with_counts: also (S, NBLK) int32 counts (I1: at most 1)."""
    g = T.geo
    d = T.dev
    code = T.c["code"]
    res = (code == C_LRU) | (code == C_PO) | (code == C_VRES)
    m = res | ((code == C_INFLIGHT) & owner_of_slots(T))
    blk = T.c["block"].view(g.S, g.NPOS).to(torch.int64).clamp(min=0)
    base = torch.arange(g.S, dtype=torch.int64, device=d).view(g.S, 1) * g.NBLK
    dummy = g.S * g.NBLK
    flat = torch.where(m.view(g.S, g.NPOS), base + blk, torch.full_like(blk, dummy)).view(-1)
    so = torch.full((dummy + 1,), -1, dtype=torch.int32, device=d)
    so[flat] = torch.arange(g.NPOS, dtype=torch.int32, device=d).repeat(g.S)
    if not with_counts:
        return so[:dummy].view(g.S, g.NBLK)
    cnt = torch.zeros((dummy + 1,), dtype=torch.int32, device=d).index_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int32))
    return so[:dummy].view(g.S, g.NBLK), cnt[:dummy].view(g.S, g.NBLK)


# ---------------------------------------------------------------------------------------------------- poll
def evict_over_c(T: Table, streams: torch.Tensor, anyw: torch.Tensor, undo: Optional[Undo] = None,
                 mut: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """K1 (own.py:280-303): while a stream's LRU holds more than C, the least recent entry leaves: a pinned one becomes
    PO (it keeps its slot, K3), an unpinned one is released (readers kept). Only the given streams; only those over C
    are sorted (a top-k of their few victims). mut: "evict_po_unpinned" (NEGATIVE CONTROL) makes every victim PO.
    Host reads: bool(any) (the early exit) and int(max) (the top-k width)."""
    g = T.geo
    zero = torch.zeros((), dtype=torch.int64, device=T.dev)
    out = dict(evict_po=zero, evict_rel=zero)
    if streams.numel() == 0:
        return out
    code2 = T.c["code"].view(g.S, g.NPOS)
    rows = code2[streams]
    nl = (rows == C_LRU).sum(1)
    over = nl > g.C
    if not bool(over.any()):
        return out
    so = streams[over]
    ex = nl[over] - g.C
    st = T.c["stamp"].view(g.S, g.NPOS)[so]
    key = torch.where(code2[so] == C_LRU, st, torch.full_like(st, BIG))
    m = int(ex.max())
    _, slots = torch.topk(key, m, dim=1, largest=False)
    take = torch.arange(m, device=T.dev).view(1, m) < ex.view(-1, 1)
    vs = slots[take]
    vss = so.view(-1, 1).expand(-1, m)[take]
    fv = vss * g.NPOS + vs
    blk = T.flat("block")[fv].to(torch.int64)
    pinned = bit_test(anyw, vss, blk)
    if mut and mut.get("evict_po_unpinned"):
        out["forced_po"] = (~pinned).sum()                      # the victims the control makes PO wrongly
        pinned = torch.ones_like(pinned)
    _put(undo, T.flat("code"), fv, torch.where(pinned, torch.full_like(blk, C_PO), torch.full_like(blk, C_RELEASING)))
    out["evict_po"] = pinned.sum()
    out["evict_rel"] = (~pinned).sum()
    return out


def force_seq(T: Table, tick: int, landed_upto: int, lag: Optional[int]) -> int:
    """The arrival rule (res.py:265-282, note 12k): the largest seq of a transfer DUE at this tick (issue_tick + lag
    <= tick) that has not landed; the caller forces the transport up to it before poll_arrivals. -1: none."""
    if lag is None:
        return -1
    code = T.flat("code")
    infl = (code == C_INFLIGHT) & ~T.flat("landed")
    seq = T.flat("seq")
    due = infl & (T.flat("issue_tick").to(torch.int64) + lag <= tick) & ~(seq <= landed_upto)
    return int(seq[due].max()) if bool(due.any()) else -1


def poll_arrivals(T: Table, tick: int, landed_upto: int, lag: Optional[int], undo: Optional[Undo] = None,
                  mut: Optional[Dict] = None) -> Dict:
    """K3X _poll's arrival half (res.py:245-264) with E's job watermark: a slot landed IN PLACE by the main stream (an
    urgent gather, `landed`) is admitted at once (res.py:258-259 skips the arrival rule for it); a transport transfer
    (seq <= landed_upto) is admitted under the arrival rule (lag: at issue_tick + lag exactly, res.py:265-282; landed
    earlier = HELD; due and not landed = refused here, the caller forces it first, force_seq). The admission branches
    (own.py:322-361):
      the slot's uid is not the stream's owner -> RELEASING, rd_last := -1 (drop_unowned; rd_first kept);
      else rd_first := -1, and: draft in the demand -> LRU at MRU (stamps in issue order), then one K1 eviction per
      admission (bulk: the least recent leave, evict_over_c); else pinned -> VRES; else RELEASING, rd_last := -1
      (drop_vres_unprotected). seq := -1.
    mut: "keep_rd_last_on_drop" (NEGATIVE CONTROL: the two resets are skipped)."""
    g = T.geo
    d = T.dev
    code = T.flat("code")
    infl = code == C_INFLIGHT
    seq = T.flat("seq")
    landed = T.flat("landed")
    tport = infl & ~landed & (seq <= landed_upto)                     # landed by the transport
    zero = torch.zeros((), dtype=torch.int64, device=d)
    out = dict(admitted=0, admit_lru=zero, admit_vres=zero, drop_unowned=zero, drop_vres_unprotected=zero, held=zero)
    if lag is not None:
        due = infl & ~landed & (T.flat("issue_tick").to(torch.int64) + lag <= tick)
        out["held"] = (tport & ~due).sum()
        if bool((due & ~tport).any()):
            raise CapacityRefusal("B2 poll: a transfer due at tick %d has not landed (force_seq first)" % tick)
        tport = tport & due
    arrived = (infl & landed) | tport
    f = arrived.nonzero().squeeze(1)
    if f.numel() == 0:
        return out
    s = torch.div(f, g.NPOS, rounding_mode="floor")
    b = stream_b(T, s)
    dropped = T.flat("uid")[f] != T.owner[b]
    to_lru = ~dropped & (torch.bitwise_and(T.flat("demand")[f], DEM_D) != 0)
    blk = T.flat("block")[f].to(torch.int64)
    pinned = bit_test(any_pin_words(T), s, blk)
    to_vres = ~dropped & ~to_lru & pinned
    to_rel = ~to_lru & ~to_vres
    newc = torch.where(to_lru, torch.full_like(blk, C_LRU), torch.where(to_vres, torch.full_like(blk, C_VRES),
                                                                          torch.full_like(blk, C_RELEASING)))
    old_stamp = T.flat("stamp")[f]
    _put(undo, code, f, newc)
    _put(undo, seq, f, -1)
    _put(undo, landed, f, False)
    _put(undo, T.flat("rd_first"), f[~dropped], -1)
    if not (mut and mut.get("keep_rd_last_on_drop")):
        _put(undo, T.flat("rd_last"), f[to_rel], -1)
    fl = f[to_lru]
    if fl.numel():
        order = torch.argsort(old_stamp[to_lru], stable=True)          # per stream the issue order (take stamps)
        _put(undo, T.flat("stamp"), fl[order], torch.arange(fl.numel(), dtype=torch.int64, device=d) + T.clock)
        T.clock += int(fl.numel())
        out.update(evict_over_c(T, torch.unique(s[to_lru]), any_pin_words(T), undo))
    out["admitted"] = int(f.numel())
    out["admit_lru"] = to_lru.sum()
    out["admit_vres"] = to_vres.sum()
    out["drop_unowned"] = dropped.sum()
    out["drop_vres_unprotected"] = (to_rel & ~dropped).sum()
    return out


def drop_due_keys(T: Table, win: ReaderWindow, undo: Optional[Undo] = None) -> Dict:
    """The queued reference releases whose (key, l) reader completed (res.py:283-297 middle part, _drop :298-305,
    Stream.unprotect own.py:213-237): both kind planes of the key slot are cleared at (l, b); PO / VRES slots of those
    streams with no pin bit left -> RELEASING (readers kept)."""
    due = T.rel_q & win.done_of(T.key_rd)
    out = dict(keys_dropped=0, unprotect_rel=0)
    if not bool(due.any()):
        return out
    l_i, b_i, k_i = due.nonzero(as_tuple=True)
    _put(undo, T.rel_q.view(-1), (l_i * T.geo.B + b_i) * T.geo.KEYS + k_i, False)
    return drop_keys(T, l_i, b_i, k_i, undo)


def drop_keys(T: Table, l_i: torch.Tensor, b_i: torch.Tensor, k_i: torch.Tensor, undo: Optional[Undo] = None) -> Dict:
    """Drop the references of key slots k_i at (l_i, b_i) (Stream.unprotect own.py:213-237, _drop res.py:298-305): both
    kind planes cleared; the PO / VRES slots of those streams with no pin bit left -> RELEASING (readers kept).
    Distinct (l, b, k) triples. Returns the counts (unprotect_rel: the releases, journal "release:unprotect")."""
    g = T.geo
    d = T.dev
    out = dict(keys_dropped=int(l_i.numel()), unprotect_rel=0)
    if l_i.numel() == 0:
        return out
    pw = T.pin_w
    for planes in (k_i, k_i + g.KEYS):
        old = pw[planes, l_i, b_i].clone()
        if undo is not None:
            undo.call(lambda p=planes, li=l_i, bi=b_i, o=old: pw.__setitem__((p, li, bi), o))
        pw[planes, l_i, b_i] = 0
    lb = torch.unique(l_i * g.B + b_i)
    streams = (lb.view(-1, 1) * g.H + torch.arange(g.H, device=d).view(1, -1)).view(-1)
    rows = T.c["code"].view(g.S, g.NPOS)[streams]
    pv = (rows == C_PO) | (rows == C_VRES)
    blk = T.c["block"].view(g.S, g.NPOS)[streams].to(torch.int64).clamp(min=0)
    pinned = bit_test(any_pin_words(T), streams.view(-1, 1).expand_as(blk), blk)
    rel = pv & ~pinned
    f = (streams.view(-1, 1) * g.NPOS + torch.arange(g.NPOS, device=d).view(1, -1))[rel]
    _put(undo, T.flat("code"), f, C_RELEASING)
    out["unprotect_rel"] = int(f.numel())
    return out


def end_owner(T: Table, b: int, undo: Optional[Undo] = None) -> int:
    """Request end (Stream.end_owner own.py:363-384 at every layer and head of batch slot b): its LRU / PO / VRES slots
    -> RELEASING (readers kept; journal "release:end"), every pin of b cleared, the owner -1 (its transfers in flight
    stay in flight and are dropped at their arrival, T3). Returns the releases."""
    g = T.geo
    code = T.c["code"][:, b]                                           # (L, H, NPOS)
    m = (code == C_LRU) | (code == C_PO) | (code == C_VRES)
    lo = torch.arange(g.L, device=T.dev).view(-1, 1, 1)
    ho = torch.arange(g.H, device=T.dev).view(1, -1, 1)
    so = torch.arange(g.NPOS, device=T.dev).view(1, 1, -1)
    f = ((((lo * g.B + b) * g.H + ho) * g.NPOS) + so).expand(g.L, g.H, g.NPOS)[m]
    _put(undo, T.flat("code"), f, C_RELEASING)
    if undo is not None:
        old = T.pin_w[:, :, b].clone()
        undo.call(lambda o=old: T.pin_w[:, :, b].copy_(o))
    T.pin_w[:, :, b] = 0
    T.owner[b] = -1
    return int(f.numel())


def poll_releases(T: Table, win: ReaderWindow, undo: Optional[Undo] = None) -> int:
    """K4 (own.py:435-452): RELEASING -> FREE once the slot's last reader completed; block, owner, class cleared;
    gen and readers kept."""
    code = T.flat("code")
    rel = (code == C_RELEASING).nonzero().squeeze(1)
    fr = rel[win.done_of(T.flat("rd_last")[rel])]
    _put(undo, code, fr, C_FREE)
    _put(undo, T.flat("block"), fr, -1)
    _put(undo, T.flat("uid"), fr, -1)
    _put(undo, T.flat("cls"), fr, 0)
    return int(fr.numel())


# ---------------------------------------------------------------------------------------------------- apply: parse
class Rows(NamedTuple):
    """The tick's rows (the same for every layer): batch slot and key slot of each verify / draft row; k entries per
    (row, head). The tensors live on the table's device."""
    vb: torch.Tensor
    vk: torch.Tensor
    db: torch.Tensor
    dk: torch.Tensor
    k: int


def parse(T: Table, layers: List[Dict], rows: Rows) -> Dict[str, Dict[str, torch.Tensor]]:
    """C's per-layer records (note 12l: vrec (9, n) = VREC fields, drec (5, n) = DREC fields) -> two flat record sets
    with the layer, the row j = o // (H k), the head (o // k) % H, the batch slot, the key slot and the stream s."""
    g = T.geo
    d = T.dev
    out = {}
    for part, name, nf, rb, rk in (("V", "vrec", 9, rows.vb, rows.vk), ("D", "drec", 5, rows.db, rows.dk)):
        recs = [x[name] for x in layers if x.get(name) is not None]
        ls = [x["l"] for x in layers if x.get(name) is not None]
        if not recs:
            out[part] = None
            continue
        R = torch.cat(recs, 1)
        n = torch.tensor([r.shape[1] for r in recs], dtype=torch.int64, device=d)
        l = torch.repeat_interleave(torch.tensor(ls, dtype=torch.int64, device=d), n, output_size=int(R.shape[1]))
        o = R[0]
        j = torch.div(o, g.H * rows.k, rounding_mode="floor")
        h = torch.div(o, rows.k, rounding_mode="floor").remainder(g.H)
        b = rb[j]
        dd = dict(l=l, j=j, h=h, b=b, key=rk[j], s=(l * g.B + b) * g.H + h, o=o, blk=R[1], flags=R[2], slot=R[3])
        if part == "V":
            dd.update(infl_slot=R[4], infl_gen=R[5], rslot=R[6], take_slot=R[7], evict_blk=R[8])
        else:
            dd.update(take_slot=R[4])
        out[part] = dd
    return out


def has(flags: torch.Tensor, bit: int) -> torch.Tensor:
    return torch.bitwise_and(flags, bit) != 0


# ---------------------------------------------------------------------------------------------------- apply: checks
def row_checks(T: Table, rows: Rows) -> Dict[str, torch.Tensor]:
    """I11 (a row of a cancelled key, res.py:534-535) and I12 (a row protecting at layer l after its (key, l) release
    was queued, res.py:538-539). Returns the failing (row) masks."""
    out = {}
    for part, b, k in (("v", rows.vb, rows.vk), ("d", rows.db, rows.dk)):
        out["cancelled_" + part] = T.kcancel[b, k]
        out["late_" + part] = T.rel_q[:, b, k].any(0)
    return out


def k5_events(P: Dict, T: Table) -> Dict[str, torch.Tensor]:
    """The tick's K5 events in K3X order per layer (verify part, then draft part; inside each the entry order o):
    FREE-slot grants (kind 0), Policy E takes (kind 1), refusals (kind 2)."""
    parts = []
    V, D = P["V"], P["D"]
    if V is not None:
        m = has(V["flags"], GRANT) | has(V["flags"], EVICT)
        kind = torch.where(has(V["flags"], EVICT), torch.ones_like(V["o"]), torch.zeros_like(V["o"]))
        parts.append((V, m, kind, 0))
    if D is not None:
        m = has(D["flags"], GRANT) | has(D["flags"], REFUSED)
        kind = torch.where(has(D["flags"], REFUSED), torch.full_like(D["o"], 2), torch.zeros_like(D["o"]))
        parts.append((D, m, kind, 1))
    cols = {n: [] for n in ("l", "s", "b", "h", "blk", "take_slot", "key", "kind", "part", "o", "src")}
    for (R, m, kind, part) in parts:
        idx = m.nonzero().squeeze(1)
        for n in ("l", "s", "b", "h", "blk", "take_slot", "key", "o"):
            cols[n].append(R[n][idx])
        cols["kind"].append(kind[idx])
        cols["part"].append(torch.full_like(idx, part))
        cols["src"].append(idx)
    ev = {n: torch.cat(v) if v else torch.zeros(0, dtype=torch.int64, device=T.dev) for n, v in cols.items()}
    if ev["o"].numel():
        order = torch.argsort((ev["l"] * 2 + ev["part"]) * (1 << 40) + ev["o"], stable=True)
        ev = {n: v[order] for n, v in ev.items()}
    return ev


def k5_check(T: Table, ev: Dict[str, torch.Tensor], mut: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """I4 (LayerPool.can_take own.py:419-427, borrowed :409-413, first_free :429-433, _take res.py:456-465,
    _free_slot_for_verifier :420-437) with rev 2's proof: usage_at = usage0 + the stream's earlier FREE-slot grants,
    borrowed_at = borrowed0 + the layer's earlier borrowing grants. A grant needs can_take and a FREE slot; a Policy E
    take needs the verifier NOT to have one; a refusal needs the draft not to have one. I5's first_free: the q-th grant
    of a stream takes the stream's q-th FREE slot of the tick start (the FREE set only shrinks inside a tick).
    mut: "inclusive_prefix" (NEGATIVE CONTROL: the take's own grant counted)."""
    g = T.geo
    code2 = T.c["code"].view(g.S, g.NPOS)
    free2 = code2 == C_FREE
    usage0 = g.NPOS - free2.sum(1)
    borrowed0 = (usage0 - g.C).clamp(min=0).view(g.L, -1).sum(1)
    grant = ev["kind"] == 0
    pf = rank_in_group(ev["s"], grant) if grant.numel() else grant.to(torch.int64)
    if mut and mut.get("inclusive_prefix"):
        pf = pf + grant.to(torch.int64)
    ua = usage0[ev["s"]] + pf
    fa = g.NPOS - ua
    bor = grant & (ua >= g.C)
    pb = rank_in_group(ev["l"], bor) if bor.numel() else bor.to(torch.int64)
    fe = g.X - (borrowed0[ev["l"]] + pb)
    isv = ev["part"] == 0
    can = (ua < g.C) | torch.where(isv, fe > 0, fe > g.R_ver)
    has_slot = can & (fa > 0)
    ok = torch.where(grant, has_slot, torch.where(ev["kind"] == 1, isv & ~has_slot, ~isv & ~has_slot))
    fr = torch.cumsum(free2.to(torch.int16), 1) - 1
    ts = ev["take_slot"]
    inr = (ts >= 0) & (ts < g.NPOS)
    f = ev["s"] * g.NPOS + ts.clamp(min=0, max=g.NPOS - 1)
    ff = ~grant | (inr & free2.view(-1)[f] & (fr.view(-1)[f].to(torch.int64) == pf))
    return dict(k5=~ok, first_free=~ff, slot_range=~inr & (ev["kind"] != 2))


def take_checks(T: Table, ev: Dict[str, torch.Tensor], so: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The takes (kind 0 / 1): the block is neither resident nor in flight for the current owner (I1 dedup, res.py:453),
    no two takes of one (stream, block) in the tick. Every miss, a refusal included (kind 2): T4 (I10, res.py:197-201;
    _take calls _hist_ev_for BEFORE K5 decides, res.py:449): a block >= hist_from needs a registered write-back."""
    tk = ev["kind"] != 2
    blk = ev["blk"].clamp(min=0)
    owned = (so.view(-1)[ev["s"] * T.geo.NBLK + blk] >= 0) & tk
    key = torch.where(tk, ev["s"] * T.geo.NBLK + blk,
                      torch.full_like(blk, BIG) + torch.arange(blk.numel(), device=T.dev))
    twice = ~_first_in_group(key) & tk if key.numel() else tk
    t4 = (ev["blk"] >= T.hist_from[ev["b"]]) & (T.hist_id[ev["b"], blk] < 0)
    return dict(dedup=owned, take_twice=twice, t4=t4)


def ready_check(T: Table, D: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """A READY draft record names the stream's resident slot of its block (note 12l; res.py:523-525), owned by the
    current owner."""
    g = T.geo
    r = has(D["flags"], READY)
    sl = D["slot"]
    inr = (sl >= 0) & (sl < g.NPOS)
    f = D["s"] * g.NPOS + sl.clamp(min=0, max=g.NPOS - 1)
    c = T.flat("code")[f]
    ok = inr & ((c == C_LRU) | (c == C_PO) | (c == C_VRES)) & (T.flat("block")[f].to(torch.int64) == D["blk"]) \
        & (T.flat("uid")[f] == T.owner[D["b"]])
    return dict(ready=r & ~ok)


def merge_check(T: Table, P: Dict, so: torch.Tensor, ev: Dict[str, torch.Tensor],
                check: bool = True) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """The in-flight merges (res.py:605-611, :672-674): a verify INFL record names the current owner's INFLIGHT slot
    and generation; a DUP record names the slot an earlier record of this tick took for the same (stream, block); a
    draft INFL record's slot is the current owner's in-flight slot (B2's slot_of, or a verify take of this tick).
    Returns (failing masks, the verify merge slots (flat), the draft merge slots (flat)). check False (the UNCHECKED
    mode): the merge targets only (the write needs them), no masks."""
    g = T.geo
    d = T.dev
    out = {}
    V, D = P["V"], P["D"]
    tk = ev["kind"] != 2
    tkey = (ev["s"] * g.NBLK + ev["blk"].clamp(min=0))[tk]
    tslot = ev["take_slot"][tk]
    srt, perm = torch.sort(tkey)
    tslot = tslot[perm]

    def this_tick(key):
        if srt.numel() == 0:
            return torch.full_like(key, -1)
        i = torch.searchsorted(srt, key).clamp(max=srt.numel() - 1)
        return torch.where(srt[i] == key, tslot[i], torch.full_like(key, -1))

    fv = torch.zeros(0, dtype=torch.int64, device=d)
    if V is not None:
        mi = has(V["flags"], INFL)
        sl = V["infl_slot"].clamp(min=0, max=g.NPOS - 1)
        f = V["s"] * g.NPOS + sl
        md = has(V["flags"], DUP)
        if check:
            ok = (V["infl_slot"] >= 0) & (T.flat("code")[f] == C_INFLIGHT) \
                & (T.flat("block")[f].to(torch.int64) == V["blk"]) \
                & (T.flat("gen")[f].to(torch.int64) == V["infl_gen"]) & (T.flat("uid")[f] == T.owner[V["b"]])
            want = this_tick(V["s"] * g.NBLK + V["blk"].clamp(min=0))
            okd = (want >= 0) & (want == V["rslot"])
            out["v_merge"] = mi & ~ok
            out["v_dup"] = md & ~okd
        fv = torch.cat([f[mi], (V["s"] * g.NPOS + V["rslot"].clamp(min=0))[md]])
    fd = torch.zeros(0, dtype=torch.int64, device=d)
    if D is not None:
        mi = has(D["flags"], INFL)
        key = D["s"] * g.NBLK + D["blk"].clamp(min=0)
        sl = so.view(-1)[key].to(torch.int64)
        sl2 = torch.where(sl >= 0, sl, this_tick(key))
        f = D["s"] * g.NPOS + sl2.clamp(min=0)
        if check:
            ok = (sl2 >= 0) & ((sl >= 0) & (T.flat("code")[f] == C_INFLIGHT) | (sl < 0))
            out["d_merge"] = mi & ~ok
        fd = f[mi]
    return out, fv, fd


def policy_e_check(T: Table, ev: Dict[str, torch.Tensor], P: Dict, sel_w: torch.Tensor, anyw: torch.Tensor,
                   inflight_kind: bool = True, mut: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """I7 (res.py:320-362, _evict_now :399-418, _evict_inflight :363-388): the r-th Policy E take of a stream (entry
    order) evicts the stream's r-th candidate: first the LRU entries unpinned at the layer start and selected by no row
    of the layer (sel_w, the device's bitmap: common-mode), least recent first; then (inflight_kind) the CURRENT
    owner's unpinned, unselected, not-landed DRAFT transfers in flight, oldest take first. Only the Policy E streams
    are gathered and top-k'd. mut: "newest_first" (NEGATIVE CONTROL)."""
    g = T.geo
    pe = ev["kind"] == 1
    out = dict(policy_e=torch.zeros_like(pe))
    if not bool(pe.any()):
        return out
    s = ev["s"][pe]
    ps = torch.unique(s)
    code = T.c["code"].view(g.S, g.NPOS)[ps]
    stamp = T.c["stamp"].view(g.S, g.NPOS)[ps]
    blk = T.c["block"].view(g.S, g.NPOS)[ps].to(torch.int64)
    uid = T.c["uid"].view(g.S, g.NPOS)[ps]
    srep = ps.view(-1, 1).expand_as(blk)
    bc = blk.clamp(min=0)
    pinned = bit_test(anyw, srep, bc)
    selected = bit_test(sel_w.reshape(g.S, g.WB), srep, bc)
    cl = (code == C_LRU) & ~pinned & ~selected
    own = T.owner[stream_b(T, ps)].view(-1, 1)
    cf = (code == C_INFLIGHT) & (uid == own) & (T.c["issuer"].view(g.S, g.NPOS)[ps] == ISS_D) \
        & ~T.c["landed"].view(g.S, g.NPOS)[ps] & ~pinned & ~selected
    if not inflight_kind:
        cf = torch.zeros_like(cf)
    big = torch.full_like(stamp, BIG)
    key = torch.where(cl, stamp, torch.where(cf, stamp + HALF, big))
    if mut and mut.get("newest_first"):
        key = torch.where(cl, HALF - 1 - stamp, key)
    r = rank_in_group(s, torch.ones_like(s, dtype=torch.bool))
    m = int(r.max()) + 1
    vals, cand = torch.topk(key, min(m, g.NPOS), dim=1, largest=False)
    row = torch.searchsorted(ps, s)
    rr = r.clamp(max=g.NPOS - 1)
    exp = cand[row, rr]
    val = vals[row, rr]
    V = P["V"]
    if V is not None:
        infl_flag = has(V["flags"][ev["src"][pe]], EVICT_INFL)
        evb = V["evict_blk"][ev["src"][pe]]
    else:
        infl_flag = torch.zeros_like(s, dtype=torch.bool)
        evb = torch.full_like(s, -1)
    ok = (r < g.NPOS) & (val < BIG) & (exp == ev["take_slot"][pe]) & (blk[row, exp] == evb) \
        & (infl_flag == (val >= HALF))
    bad = torch.zeros_like(pe)
    bad[pe] = ~ok
    out["policy_e"] = bad
    return out


# ---------------------------------------------------------------------------------------------------- apply: writes
def pins_or(T: Table, rowsel: torch.Tensor, rows_b: torch.Tensor, planes: torch.Tensor,
            undo: Optional[Undo] = None, mut: Optional[Dict] = None) -> int:
    """K2 (every selected block is protected by the row's key, res.py:597 VERIFY, :668 DRAFT): rowsel (L, R, H, WB)
    words OR'ed into plane planes[r] at batch slot rows_b[r]. Rows that share (plane, b) (a cohort's verify rows,
    core.py:146-150) go in ROUNDS, so each round writes distinct (plane, b). Returns the rounds.
    mut: "sum_not_or" (NEGATIVE CONTROL: words added, not OR'ed)."""
    g = T.geo
    if rows_b.numel() == 0:
        return 0
    grp = planes * g.B + rows_b
    rnd = rank_in_group(grp, torch.ones_like(grp, dtype=torch.bool))
    pw = T.pin_w
    n = int(rnd.max()) + 1
    for r in range(n):
        idx = (rnd == r).nonzero().squeeze(1)
        p, b = planes[idx], rows_b[idx]
        old = pw[p, :, b].clone()                                    # (n_r, L, H, WB)
        new = rowsel[:, idx].permute(1, 0, 2, 3)
        if undo is not None:
            undo.call(lambda p=p, b=b, o=old: pw.__setitem__((p, slice(None), b), o))
        pw[p, :, b] = (old + new) if (mut and mut.get("sum_not_or")) else torch.bitwise_or(old, new)
    return n


def write_takes(T: Table, ev: Dict[str, torch.Tensor], tick: int, job_seq: int, win: ReaderWindow,
                undo: Optional[Undo] = None, mut: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """FREE -> INFLIGHT for every take of the tick in K3X order (own.py:238-269, res.py:438-489): gen + 1, rd_first :=
    -1, rd_last KEPT (the T3 forward, res.py:478), uid := owner, class "v" / "ad", stamp := the take clock (issue
    order), the transfer's columns, hist_gate := the block's write-back id (T4), xfer + 1 (refetch = a second transfer
    of (request, l, h, block), res.py:863-868). A Policy E take first frees its victim (mid-tick FREE, res.py:399-418,
    :363-388): a pending last reader is counted as a host wait. Returns the issue list (E's desc rows) and flags.
    mut: "gen_plus_two" / "drop_rd_last" (NEGATIVE CONTROLS)."""
    g = T.geo
    d = T.dev
    tk = ev["kind"] != 2
    idx = tk.nonzero().squeeze(1)
    E_ = {n: v[idx] for n, v in ev.items()}
    f = E_["s"] * g.NPOS + E_["take_slot"]
    n = int(f.numel())
    isv = E_["part"] == 0
    pe = E_["kind"] == 1
    rd_old = T.flat("rd_last")[f].clone()
    c_old = T.flat("code")[f]
    force_waits = (pe & (c_old == C_LRU) & ~win.done_of(rd_old)).sum()     # _evict_now's host wait
    superseded = (pe & (c_old == C_INFLIGHT)).sum()                         # _evict_inflight: the device waits
    gen_new = T.flat("gen")[f].to(torch.int64) + (2 if (mut and mut.get("gen_plus_two")) else 1)
    _put(undo, T.flat("code"), f, C_INFLIGHT)
    _put(undo, T.flat("block"), f, E_["blk"])
    _put(undo, T.flat("gen"), f, gen_new)
    _put(undo, T.flat("uid"), f, T.owner[E_["b"]])
    _put(undo, T.flat("cls"), f, torch.where(isv, torch.full_like(f, CLS_V), torch.full_like(f, CLS_AD)))
    _put(undo, T.flat("stamp"), f, torch.arange(n, dtype=torch.int64, device=d) + T.clock)
    T.clock += n
    _put(undo, T.flat("rd_first"), f, -1)
    if mut and mut.get("drop_rd_last"):
        _put(undo, T.flat("rd_last"), f, -1)
    _put(undo, T.flat("seq"), f, job_seq)
    _put(undo, T.flat("issue_tick"), f, tick)
    _put(undo, T.flat("demand"), f, torch.where(isv, torch.full_like(f, DEM_V), torch.full_like(f, DEM_D)))
    _put(undo, T.flat("issuer"), f, torch.where(isv, torch.full_like(f, ISS_V), torch.full_like(f, ISS_D)))
    _put(undo, T.flat("req_key"), f, E_["key"])
    _put(undo, T.flat("landed"), f, isv)                           # the urgent gather lands in place (main stream)
    blk = E_["blk"].clamp(min=0)
    _put(undo, T.flat("hist_gate"), f, T.hist_id[E_["b"], blk])
    _put(undo, T.flat("rd_n"), f, 0)
    _put(undo, T.flat("rv_n"), f, 0)
    xi = ((E_["b"] * g.L + E_["l"]) * g.H + E_["h"]) * g.NBLK + blk
    xn = T.xfer.view(-1)[xi] + 1
    _put(undo, T.xfer.view(-1), xi, xn)
    return dict(f=f, l=E_["l"], b=E_["b"], h=E_["h"], blk=E_["blk"], slot=E_["take_slot"], gen=gen_new, isv=isv,
                pe=pe, rd_old=rd_old, refetch=xn > 1, force_waits=force_waits, superseded=superseded)


def write_merges(T: Table, fv: torch.Tensor, fd: torch.Tensor, undo: Optional[Undo] = None) -> None:
    """The demand merges (res.py:605-611 VERIFY, :672-674 DRAFT): the slot's transfer gets the bit."""
    for f, bit in ((fv, DEM_V), (fd, DEM_D)):
        if f.numel():
            f = torch.unique(f)
            _put(undo, T.flat("demand"), f, torch.bitwise_or(T.flat("demand")[f], bit))


def recency(T: Table, D: Dict[str, torch.Tensor], rows: Rows, undo: Optional[Undo] = None,
            mut: Optional[Dict] = None) -> Dict:
    """K1 / O1 (_draft_recency res.py:686-706): per draft row IN ORDER, its READY records in ascending block order (the
    record order): an LRU slot is touched (to MRU), a PO / VRES slot is acquired (-> LRU at MRU, own.py:305-321), any
    other state is a violation; then evict_to_c for that row's streams (own.py:297-303). Rows of different requests are
    independent, so ROUND r = the r-th draft row of each request (rev 2 point (2)); R2's planner gives one round
    (core.py:157-164). Must run AFTER the tick's pins (evict_to_c tests the protection then). The counts (touch,
    acquire, evict_po, evict_rel) equal the vendored journal's (B2 reports them; rev 2 0.1 (2)).
    mut: "reverse_stamps" (NEGATIVE CONTROL); "one_batch" (all rows' touches, then one eviction: a REPORT variant. With
    the pins fixed during the recency and every read block pinned by phase A (res.py:668), it reaches the SAME end
    state: the final LRU is the C largest final stamps either way, and the rounds' extra victims are re-acquired.)"""
    g = T.geo
    d = T.dev
    zero = torch.zeros((), dtype=torch.int64, device=d)
    out = dict(rounds=0, touch=zero, acquire=zero, bad=zero, evict_po=zero, evict_rel=zero)
    if D is None or rows.db.numel() == 0:
        return out
    occ = rank_in_group(rows.db, torch.ones_like(rows.db, dtype=torch.bool))
    r_of = occ[D["j"]]
    ready = has(D["flags"], READY)
    n = int(occ.max()) + 1
    one = bool(mut and mut.get("one_batch"))
    plan = [(None, None)] if one else [(r, r) for r in range(n)]
    for (r0, _) in plan:
        m = ready if one else (ready & (r_of == r0))
        f = (D["s"][m] * g.NPOS + D["slot"][m].clamp(min=0))
        c0 = T.flat("code")[f]
        acq = (c0 == C_PO) | (c0 == C_VRES)
        out["bad"] = out["bad"] + (~acq & (c0 != C_LRU)).sum()
        _put(undo, T.flat("code"), f[acq], C_LRU)
        sv = torch.arange(f.numel(), dtype=torch.int64, device=d) + T.clock
        if mut and mut.get("reverse_stamps"):
            sv = sv.flip(0)                                         # NEGATIVE CONTROL: O1's ascending order lost
        _put(undo, T.flat("stamp"), f, sv)
        T.clock += int(f.numel())
        out["touch"] = out["touch"] + (c0 == C_LRU).sum()
        out["acquire"] = out["acquire"] + acq.sum()
        sel_rows = rows.db if one else rows.db[occ == r0]
        streams = ((torch.arange(g.L, device=d).view(-1, 1, 1) * g.B + sel_rows.view(1, -1, 1)) * g.H
                   + torch.arange(g.H, device=d).view(1, 1, -1)).reshape(-1)
        if one:
            streams = torch.unique(streams)
        e = evict_over_c(T, streams, any_pin_words(T), undo, mut)
        out["evict_po"] = out["evict_po"] + e["evict_po"]
        out["evict_rel"] = out["evict_rel"] + e["evict_rel"]
        if "forced_po" in e:
            out["forced_po"] = out.get("forced_po", 0) + e["forced_po"]
        if mut and mut.get("reverse_stamps") and bool(m.any()):   # (tests only: the control's precondition)
            out["max_stream_reads"] = max(out.get("max_stream_reads", 0), int(torch.bincount(D["s"][m]).max()))
        out["rounds"] += 1
    return out


# ---------------------------------------------------------------------------------------------------- post checks
POST_CHECKS = ("I1", "I2", "I3", "I9", "K5sum")


def post_checks(T: Table, gen_before: Optional[torch.Tensor], n_takes: int,
                which=POST_CHECKS) -> Dict[str, torch.Tensor]:
    """POST-WRITE predicates (rev 2 section 3.3): I1 (resident + current-owner INFLIGHT at most 1 per block), I2 (LRU <=
    C), I3 (PO / VRES pinned), I9 (gen changes only at the takes, by +1; needs gen_before), the K5 sum (borrowed <= X per
    layer). Each is a 0-dim bool tensor, True = the predicate FAILS (the caller reads them together: one host read).
    which: the subset to compute (the poll checks I1 and I2 only)."""
    g = T.geo
    code2 = T.c["code"].view(g.S, g.NPOS)
    out = {}
    if "I1" in which:
        _, cnt = slot_of(T, with_counts=True)
        out["I1"] = (cnt > 1).any()
    if "I2" in which:
        out["I2"] = ((code2 == C_LRU).sum(1) > g.C).any()
    if "I3" in which:
        pv = ((code2 == C_PO) | (code2 == C_VRES)).view(-1).nonzero().squeeze(1)
        s = torch.div(pv, g.NPOS, rounding_mode="floor")
        out["I3"] = (~bit_test(any_pin_words(T), s, T.flat("block")[pv].to(torch.int64).clamp(min=0))).any()
    if "I9" in which and gen_before is not None:
        dg = T.flat("gen").to(torch.int64) - gen_before.view(-1).to(torch.int64)
        out["I9"] = ((dg != 0) & (dg != 1)).any() | ((dg != 0).sum() != n_takes)
    if "K5sum" in which:
        usage = g.NPOS - (code2 == C_FREE).sum(1)
        out["K5sum"] = ((usage - g.C).clamp(min=0).view(g.L, -1).sum(1) > g.X).any()
    return out


# ---------------------------------------------------------------------------------------------------- readers, metrics
def read_done(T: Table, vread: torch.Tensor, D: Optional[Dict[str, torch.Tensor]], rows: Rows,
              e_l: torch.Tensor) -> Dict:
    """The layers' reader events (res.py:743-761): every slot a row read gets rd_first (if none) and rd_last := the
    layer's event id e_l[l]; the verify reads come from vread (L, B, H, WP) words, the draft reads from the READY
    records; the rows' keys get key_rd; the verify keys' (key, l) are queued (rel_q) behind the reader. The 4.8
    lifetime counts rv_n / rd_n count the reads (once per layer for the verify reads, as B1)."""
    g = T.geo
    vb = unpack_words(vread, g.NPOS).reshape(g.L, g.B, g.H, g.NPOS)
    e = e_l.view(g.L, 1, 1, 1)
    rf, rl = T.c["rd_first"], T.c["rd_last"]
    rf.copy_(torch.where(vb & (rf < 0), e, rf))
    rl.copy_(torch.where(vb, e, rl))
    T.c["rv_n"].add_(vb.to(torch.int32))
    nd = 0
    if D is not None:
        m = has(D["flags"], READY)
        f = D["s"][m] * g.NPOS + D["slot"][m]
        ed = e_l[D["l"][m]]
        rff = T.flat("rd_first")
        rff[f] = torch.where(rff[f] < 0, ed, rff[f])
        T.flat("rd_last")[f] = ed
        T.flat("rd_n").index_add_(0, f, torch.ones_like(f, dtype=torch.int32))
        nd = int(f.numel())
    for b, k in ((rows.vb, rows.vk), (rows.db, rows.dk)):
        if b.numel():
            T.key_rd[:, b, k] = e_l.view(-1, 1)
    if rows.vb.numel():
        T.rel_q[:, rows.vb, rows.vk] = True
    return dict(verify_reads=vb.sum(), draft_reads=nd)


def metrics(P: Dict, vlook: Optional[torch.Tensor], takes: Dict[str, torch.Tensor], rows: Rows, geo: Geo,
            n_vkeys: int) -> Dict:
    """The compared metrics as array reductions (test.py:607-611): k43 ready / in_flight / new / layer_calls, k46 (the
    device's ready classes + fetched_first / refetch), masked, draft_pairs, refused, verifier_evictions. 0-dim tensors
    (one host read by the caller). n_vkeys: the tick's distinct verify keys (host-known, from the rows)."""
    V, D = P["V"], P["D"]
    out = {}
    if vlook is not None:
        s = vlook.reshape(-1, 4).sum(0)
        out.update(k43_ready=s[0], k46_lru=s[1], k46_po=s[2], k46_vres=s[3])
    if V is not None:
        first = has(V["flags"], FIRST)
        out["k43_in_flight"] = (first & has(V["flags"], INFL)).sum()
        out["k43_new"] = (first & (has(V["flags"], GRANT) | has(V["flags"], EVICT))).sum()
        out["verifier_evictions"] = has(V["flags"], EVICT).sum()
        out["verifier_inflight_evictions"] = has(V["flags"], EVICT_INFL).sum()
        out["k43_layer_calls"] = n_vkeys * geo.L
    if takes["f"].numel():
        out["k46_fetched_refetch"] = (takes["isv"] & takes["refetch"]).sum()
        out["k46_fetched_first"] = (takes["isv"] & ~takes["refetch"]).sum()
    if D is not None:
        nr = ~has(D["flags"], READY)
        pair = (D["l"] * rows.db.numel() + D["j"]) * geo.H + D["h"]
        npairs = geo.L * rows.db.numel() * geo.H
        out["masked"] = (torch.bincount(pair[nr], minlength=npairs) > 0).sum()
        out["draft_pairs"] = npairs
        out["refused"] = has(D["flags"], REFUSED).sum()
    return out


def issue_desc_args(T: Table, takes: Dict[str, torch.Tensor], win: ReaderWindow) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """E's tp.issue_desc(desc, reader_marks, hist_mark) for the DRAFT takes (Q-B3): desc (6, n) CONTIGUOUS int64 (l, b,
    h, block, slot, gen) (lessons L28); reader_marks (L,) the newest mark of the destination slots' kept last readers
    per layer (-1: none pending); hist_mark the newest write-back mark of the job's (b, block) (-1: none; E's call takes
    an int: one host read)."""
    dm = ~takes["isv"]
    l, b = takes["l"][dm], takes["b"][dm]
    desc = torch.stack([l, b, takes["h"][dm], takes["blk"][dm], takes["slot"][dm], takes["gen"][dm]]).contiguous()
    marks = win.mark_of(takes["rd_old"][dm])
    rm = torch.full((T.geo.L,), -1, dtype=torch.int64, device=T.dev)
    if l.numel():
        rm.scatter_reduce_(0, l, marks, reduce="amax", include_self=True)
        hm = win.mark_of(T.hist_id[b, takes["blk"][dm].clamp(min=0)])
        return desc, rm, int(hm.max())
    return desc, rm, -1
