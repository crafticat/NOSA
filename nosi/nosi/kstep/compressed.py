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
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple


class CompressedChain:
    __slots__ = ("stride", "reduce", "cks", "buf", "count", "events")

    def __init__(self, stride: int, reduce: Callable[[List[Any]], Any]):
        if stride < 1:
            raise ValueError("stride >= 1")
        self.stride, self.reduce = int(stride), reduce
        self.cks: List[Any] = []
        self.buf: List[Any] = []
        self.count = 0
        self.events = 0

    def append(self, key) -> bool:
        """Feed one key; True when this append fired a compression event."""
        self.buf.append(key)
        self.count += 1
        if len(self.buf) == self.stride:
            self.cks.append(self.reduce(list(self.buf)))
            self.buf = []
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
        c = CompressedChain(self.stride, self.reduce)
        c.cks = list(self.cks)
        c.buf = list(self.buf)
        c.count = self.count
        return c

    def windows_complete(self) -> int:
        return len(self.cks)
