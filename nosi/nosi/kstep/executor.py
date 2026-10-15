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

M2 RING PROTOCOL, interface v3 (Config.recent = "ring"; contract revisions 4-5, section 11.3), versioned by the attribute
reads_ring = True (requires reads_arena; ring mode refuses an executor without it). Checkpoint 1 (2026-10-14) added this
interface (Row.ring, the attach_ring requirement of check_executor); checkpoint 2 is the dispatcher's use of it.
In addition to the arena protocol:
  attach_ring(be, rk, rv, rgen, register_ring)
                          once, at dispatcher construction: per layer the ring rows rk[l], rv[l] ((Q_l, D): row e = entry
                          e) and the generation mirror rgen[l] ((Q_l,) int64, written at every bind);
                          register_ring(l, row, h, pos, entry, rgen, kind, which, ws, pg) records ONE deferred ring read
                          (which "k" or "v"; ws the deferred copy of the entry's row; pg a one-element copy of
                          rgen[l][entry] made at the read's execution).
  Row.ring[h] = [(pos, entry, rgen, kind)] in position order for every NON-SLOT position the row attends at head h (tail
                          block, selected non-historical blocks, retained exact, own pre, predecessor prov, own draft).
  joint_post(l, rows) -> (outs, read_ev)
                          read_ev is recorded on "compute" after the LAST read of ANY arena position OR ring entry, and
                          is None only when the call read nothing. The executor reads the K row and the V row of every
                          Row.ring entry (registering each). Row.kv stays the eager input (the semantic reference).

C1.1 ABI v4 (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 10, 12.3 A0-A9
and 12.3.1; opt-in): declared by the attribute abi = 4 together with chain_kernel. Without it everything above is
unchanged (v3). With it (needs reads_arena; the dispatcher needs physical "arena" and recent "ring" or "eager_matched"):
  joint_pre(l, items)     returns [(q, k, v, c)]: c the per-KV-head bias of the row's own position (NOSA cis).
  prefill(tokens)         kv[pos][l][h] = (k, v, c).
  compress(elems)         elems are (k, c) pairs of one window; returns the pair (reduced k, reduced c).
  select(l, h, q_h, cks, n_cand)
                          cks are the (ck, cc) windows VISIBLE to the row (12.3.1 V1, V3: complete windows of positions
                          before the row's own); the per-row extent is len(cks); window w holds the positions
                          [w * stride, w * stride + chain_kernel); no chain-length assertion (the references check it).
  Row.kv[h]               (k, v, c) per attended position, in increasing position order (C1.1 still consumes Row.kv).
  attach_bias(be, ac, rc) once, at dispatcher construction: per layer the arena bias rows ac[l] ((P_l * bs,): row
                          p * bs + r) and, ring mode only, the ring bias rows rc[l] ((Q_l,): row e); rc is None
                          otherwise. joint_post registers ONE more deferred arena read per (row, head, block) of
                          Row.addr, the bias rows (a 1-D copy, identity copy as for K / V), and ONE more ring read per
                          Row.ring entry with which = "c" (a 1-element copy of rc[l][e:e + 1]).
  Geometry (12.3.1 V5, check_chain_geometry): stride >= 1, stride divides bs and chain_kernel, stride <= chain_kernel
  <= bs; NOSA: (bs, chain_kernel, stride) = (64, 32, 16) only.
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
    ring: Optional[List[List[Tuple[int, int, int, str]]]] = None   # M2 B2 v3: per head [(pos, entry, rgen, kind)]


REQUIRED = ("joint_pre", "select", "joint_post", "dist", "embed", "compress", "prefill")
ATTRS = ("L", "H", "D", "V", "bs", "stride", "max_sel")
ARENA_REQUIRED = ("attach_arena",)
RING_REQUIRED = ("attach_ring",)
ABI4_REQUIRED = ("attach_bias",)                 # C1.1 ABI v4 (docstring)
ABI4_ATTRS = ("chain_kernel",)


def abi_of(ex) -> int:
    """The executor's declared ABI: 4 with abi = 4 (C1.1), else 3 (every earlier interface)."""
    abi = getattr(ex, "abi", 3)
    if abi not in (3, 4):
        raise ValueError("executor %r declares abi %r (3 or 4)" % (type(ex).__name__, abi))
    return int(abi)


def check_chain_geometry(bs: int, kernel: int, stride: int) -> None:
    """C1.1 V5: stride >= 1, stride divides bs and kernel, and stride <= kernel <= bs (ValueError otherwise)."""
    if stride < 1 or bs % stride or kernel % stride or not stride <= kernel <= bs:
        raise ValueError("C1 chain geometry (bs %d, kernel %d, stride %d): need stride >= 1, stride dividing bs and "
                         "kernel, and stride <= kernel <= bs" % (bs, kernel, stride))


def check_executor(ex) -> None:
    miss = [m for m in REQUIRED if not callable(getattr(ex, m, None))] + [a for a in ATTRS if not hasattr(ex, a)]
    if getattr(ex, "reads_arena", False):
        miss += [m for m in ARENA_REQUIRED if not callable(getattr(ex, m, None))]
    if getattr(ex, "reads_ring", False):
        miss += [m for m in RING_REQUIRED if not callable(getattr(ex, m, None))]
    abi4 = abi_of(ex) == 4
    if abi4:
        miss += [m for m in ABI4_REQUIRED if not callable(getattr(ex, m, None))] + [
            a for a in ABI4_ATTRS if not hasattr(ex, a)]
    if miss:
        raise TypeError("executor %r lacks %s" % (type(ex).__name__, miss))
    if ex.bs % ex.stride:
        raise ValueError("the compression stride %d must divide the block size %d" % (ex.stride, ex.bs))
    if abi4:
        check_chain_geometry(ex.bs, ex.chain_kernel, ex.stride)
