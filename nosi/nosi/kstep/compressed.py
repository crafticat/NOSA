"""A6: the request-local compressed-key chain of one (layer, KV head), with a journal.

NOSA keeps per layer a no-compress key buffer that is SHIFTED at a compression event and a compressed-key table that
GROWS by append (NOSI cache_engine.py:966-995 at de61f9a; batch-wide there). The chain here is the request-local form:
  buf   the keys fed since the last compression boundary (len < stride)
  cks   one compressed key per completed window of `stride` consecutive positions (append-only)
  count positions fed
append(key) adds a key; when buf reaches `stride` keys a COMPRESSION EVENT appends reduce(buf) to cks and clears buf.
snapshot()/restore() put the chain back across any number of events (truncate cks, restore buf, count), the per-request
form of the batch-wide NOSI LayerJournal take/restore (nosi/nosi/spec_loop.py:692-763; scalars + cloned shifted buffer +
the reference of the appended table). The reduction is the model's (NOSA: a mean of keys; the M1 toy: an exact sum).

C1.1 (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 10, 12.3 A2 and
12.3.1 V1-V5): `kernel` (default = stride, today's chain exactly: the same code path). With kernel > stride the windows
OVERLAP: window w holds the positions [w * stride, w * stride + kernel); an event fires when count >= kernel and
(count - kernel) % stride == 0 and reduces the LAST `kernel` elements; buf then keeps the last kernel - stride elements
(the next window's start), so it holds at most kernel elements and copy / snapshot / restore carry the overlap. An
element is whatever the dispatcher appends (ABI v4: the pair (k, c) of one producer). row_prefixes is V3's verifier
loop: each row selects over the chain as it stands at its own position, before its own element is appended.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

from .errors import KStepError


class CompressedChain:
    __slots__ = ("stride", "kernel", "reduce", "cks", "buf", "count", "events")

    def __init__(self, stride: int, reduce: Callable[[List[Any]], Any], kernel: int = None):
        if stride < 1:
            raise ValueError("stride >= 1")
        self.stride, self.reduce = int(stride), reduce
        self.kernel = self.stride if kernel is None else int(kernel)
        if self.kernel < self.stride:
            raise ValueError("C1 chain geometry: kernel %d < stride %d" % (self.kernel, self.stride))
        self.cks: List[Any] = []
        self.buf: List[Any] = []
        self.count = 0
        self.events = 0

    def append(self, key) -> bool:
        """Feed one key; True when this append fired a compression event."""
        self.buf.append(key)
        self.count += 1
        if self.kernel == self.stride:                    # today's chain (c4ec309), unchanged
            if len(self.buf) == self.stride:
                self.cks.append(self.reduce(list(self.buf)))
                self.buf = []
                self.events += 1
                return True
            return False
        if self.count >= self.kernel and (self.count - self.kernel) % self.stride == 0:   # C1.1 overlapping window
            self.cks.append(self.reduce(list(self.buf[-self.kernel:])))
            self.buf = self.buf[len(self.buf) - (self.kernel - self.stride):]
            self.events += 1
            return True
        return False

    def snapshot(self) -> Tuple[int, Tuple[Any, ...], int]:
        return (len(self.cks), tuple(self.buf), self.count)

    def restore(self, snap) -> None:
        n_cks, buf, count = snap
        if n_cks > len(self.cks):
            raise ValueError("restore: snapshot has %d compressed keys, the chain only %d" % (n_cks, len(self.cks)))
        del self.cks[n_cks:]
        self.buf = list(buf)
        self.count = int(count)

    def copy(self) -> "CompressedChain":
        c = CompressedChain(self.stride, self.reduce, self.kernel)
        c.cks = list(self.cks)
        c.buf = list(self.buf)
        c.count = self.count
        return c

    def windows_complete(self) -> int:
        return len(self.cks)


def row_prefixes(chain: CompressedChain, positions: List[int], elems: List[Any], who: str, where: str) -> List[List[Any]]:
    """C1.1 V3 (ABI v4 verifier rows): for each row in position order, the chain must hold exactly that row's position
    (else KStepError "C1 row prefix: <who> row <pos> <where> chain holds <n>"); the row's visible windows are the chain's
    windows at that point (a new list per row); then the row's element is appended. Returns the per-row window lists."""
    out = []
    for pos, x in zip(positions, elems):
        if chain.count != pos:
            raise KStepError("C1 row prefix: %s row %d %s chain holds %d" % (who, pos, where, chain.count))
        out.append(list(chain.cks))
        chain.append(x)
    return out
