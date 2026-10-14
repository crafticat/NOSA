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

M2 ARENA PROTOCOL (Config.physical = "arena"; retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-
contract.md revision 2, section 3.1), versioned by the attribute reads_arena = True (arena mode refuses an executor
without it). Row.kv stays the eager input of the row (read through the address map at call time, as above). In addition:
  attach_arena(be, ak, av, pgen, register)
                          once, at dispatcher construction: the backend; per layer the arena rows ak[l], av[l]
                          ((P_l * bs, D): row r of position p is row p * bs + r) and the generation mirror pgen[l] ((P_l,)
                          int64, written at every bind); register(l, row, h, block, phys, pgen, ws, exp, pg) records ONE
                          deferred read with the dispatcher (ws the deferred copy, exp an eager clone taken at call time,
                          pg a one-element copy of pgen[l][phys] made at the read's execution, or None).
  joint_post(l, rows) -> (outs, read_ev)
                          Row.addr[h] = [(block, phys, pgen)] in increasing BLOCK order (the declared logical order) for
                          the historical blocks the row reads at head h. The executor enqueues every arena read on the ONE
                          ordered stream "compute" and records read_ev there AFTER its last read of the call (None when it
                          read nothing). The dispatcher installs read_ev as the last reader of every slot and position read
                          IMMEDIATELY after joint_post returns; nothing frees a position in between. Reads on several
                          independent streams would need a joined completion frontier: not part of this protocol.
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
    addr: Optional[List[List[Tuple[int, int, int]]]] = None   # M2 arena: per head [(block, phys, pgen)], block order


REQUIRED = ("joint_pre", "select", "joint_post", "dist", "embed", "compress", "prefill")
ATTRS = ("L", "H", "D", "V", "bs", "stride", "max_sel")
ARENA_REQUIRED = ("attach_arena",)


def check_executor(ex) -> None:
    miss = [m for m in REQUIRED if not callable(getattr(ex, m, None))] + [a for a in ATTRS if not hasattr(ex, a)]
    if getattr(ex, "reads_arena", False):
        miss += [m for m in ARENA_REQUIRED if not callable(getattr(ex, m, None))]
    if miss:
        raise TypeError("executor %r lacks %s" % (type(ex).__name__, miss))
    if ex.bs % ex.stride:
        raise ValueError("the compression stride %d must divide the block size %d" % (ex.stride, ex.bs))
