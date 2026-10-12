"""The model-executor interface of the joint layer call (spec D1, D5). M1 binds it to the test-side toy model
(retroinfer-eval tests/kstep_toy.py); M2 binds it to the NOSA-8B kernels.

A layer is two phases, as in NOSA (QKV + compressed-key scoring / top-k, then sparse attention + O / FFN):
  joint_pre(l, items)     items = [(hidden, pos)]; returns [(q, k, v)], each a list over KV heads. Phase 1 of every row.
  select(l, h, q_h, cks, n_cand)
                          NOSA's own selection of one (row, head): a sorted list of block ids < n_cand, computed from
                          the row's own query and the compressed keys `cks` (one per window of `stride` positions,
                          window w covers [w*stride, (w+1)*stride)). Draft rows pass the DRAFT chain, verifier rows the
                          EXACT chain (D5); the dispatcher never hands a draft selection to a verifier.
  joint_post(l, rows)     rows = [Row]; returns the output hidden state per row. Row.kv[h] is the list of (k, v) the row
                          attends at head h, in increasing position order: the selected blocks it may read and its causal
                          recent region (tail block up to and including its own position).
  dist(hidden)            the next-token distribution (a list of exact fractions over the vocabulary).
  embed(token)            the layer-0 input of a fed token.
  compress(keys)          the reduction of one window of keys into a compressed key.
  prefill(tokens)         exact KV of feeding tokens[0 .. n-1]: kv[pos][l][h] = (k, v) (start of decoding, outside M1).
Attributes: L, H, D (per-head KV width), V, bs (block size), stride (compression stride, divides bs), max_sel (largest
number of blocks one (row, head) selects).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

DRAFT, VERIFY = "draft", "verify"


@dataclass
class Row:
    """One row of a joint layer call, as handed to the executor (D1)."""
    uid: Any
    b: int
    pos: int
    kind: str                         # DRAFT | VERIFY
    cohort: Optional[Tuple]           # (uid, epoch, m): the cohort of d_s for a draft row, the cohort for a verifier row
    sel: List[List[int]]              # selected blocks per head
    extent: int                       # causal extent: the row reads positions <= extent
    readable: List[List[int]]         # the selected HISTORICAL blocks per head it may read (resident, arrival observed)
    hidden: Any = None
    q: Any = None
    kv: List[List[Tuple]] = field(default_factory=list)


REQUIRED = ("joint_pre", "select", "joint_post", "dist", "embed", "compress", "prefill")
ATTRS = ("L", "H", "D", "V", "bs", "stride", "max_sel")


def check_executor(ex) -> None:
    miss = [m for m in REQUIRED if not callable(getattr(ex, m, None))] + [a for a in ATTRS if not hasattr(ex, a)]
    if miss:
        raise TypeError("executor %r lacks %s" % (type(ex).__name__, miss))
    if ex.bs % ex.stride:
        raise ValueError("the compression stride %d must divide the block size %d" % (ex.stride, ex.bs))
