"""C1.3a (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md 12.10, revisions 12-13):
the PORTABLE parts of the serialized NOSA layer-0 numerical reference-gather harness. Pure torch and this package:
nothing here imports CUDA, triton or the engine. The engine constants used here are repeated with their source lines at
NOSI 17cf3cc (nl = nosi/nosi/nosa_llama.py, mpf = nosi/nosi/max_pooling_fused.py).

Every tensor operation is issued in program order on the current stream of the tensors' device. The caller (a later,
separately approved GPU gate; NOT part of C1.3a) sets the one stream, records the read-completion event and host-waits
it before check_identity. On the CPU every operation completes in program order.

  n_w, m_p, forced_ids, row_layout   12.10.3 notation and the 12.10.4 slot layout
  select_blocks, complete_blocks     12.10.3 P5: the host selection with the deterministic tie rule (value desc, id asc)
  C13Ledger, Backings                a PARTIAL owned-tensor-backing ledger (BiasOpLedger with the c13 kinds; 12.10.5)
  C13Stores                          the layer-0 stores in the A4 layout, the registered address maps, the producer
                                     and the block promotion (12.10.5 steps 1-2)
  build_descriptor                   Row.addr / Row.ring per KV head from the bind tables (step 3: the capture)
  gather_candidate                   the candidate gather into one W = 64 x 64 cache (step 5); `substitute` is the
                                     harness hook of the matched controls (12.10.7) and is None in a normal call
  twin_r_cache                       Twin R from the logical state only (12.10.4)
  check_store_readback, check_identity, compare_cache   step 4 (CP2), step 7 and CP4

No NOSA numerics, kernel call, model load, CUDA lifetime, memory fit or throughput claim follows from this module."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .errors import CacheMismatch, ConfigError, KStepError, OwnershipViolation, StaleReader
from .ownership import BiasOpLedger

BS = 64                   # block_size (nl:956; asserted at nl:960)
SLOTS = 64                # attended slots per row = topk_blocks (nl:952; asserted at nl:961); W = SLOTS * BS rows
TOPK = 64                 # the final top-k of the CIS ranking (topk_blocks)
KERNEL, STRIDE = 32, 16   # pooling_block_size (nl:958), pooling_stride (nl:957): window w covers [16 w, 16 w + 32)
QK_SELECT = 33            # nl:45
INIT_BLOCKS = 1           # init_blocks (nl:954)
LOCAL_BLOCKS = 16         # window_blocks (nl:955); nosa_pooling passes LOCAL_BLOCKS + 1 (mpf:105) to mpf:61
MASK_BIAS = -3.0e4        # avail_policy.py:113 (finite on purpose); stored as bf16 it is -29952
C13_KINDS = ("c13_arena", "c13_ring", "c13_chain", "c13_staging", "c13_output")
P_MULT, P_ADD = 37, 11    # phys(j, h) = (37 (H j + h) + 11) mod P, P = H NB + 2 (12.10.5)
Q_MULT, Q_ADD = 29, 5     # entry(q, h) = (29 (H (q mod bs) + h) + 5) mod Q, Q = H bs + 2


# ------------------------------------------------------------------------------------------------------------ geometry
def n_w(p: int, kernel: int = KERNEL, stride: int = STRIDE) -> int:
    """12.3.1 V1: the number of windows visible to the row at p (every position of a visible window is < p)."""
    return 0 if p < kernel else (p - kernel) // stride + 1


def m_p(p: int, bs: int = BS) -> int:
    """The row-local pooled width: the complete blocks 0 .. p // bs - 1 and the own block p // bs (id M_p - 1)."""
    return p // bs + 1


def forced_ids(M: int, init_blocks: int = INIT_BLOCKS, local_blocks: int = LOCAL_BLOCKS) -> List[int]:
    """The ids that nosa_pooling scores +inf: m < init_blocks or M - m <= local_blocks + 1 (mpf:61-63, mpf:105)."""
    return [m for m in range(M) if m < init_blocks or M - m <= local_blocks + 1]


def row_layout(p: int, n_c: int, bs: int = BS, slots: int = SLOTS) -> Tuple[int, int, int]:
    """12.10.4: (tail slot, tail length, extent) of the row at p with n_c selected complete blocks per KV head. The tail
    slot is n_c; the tail holds the positions bs (p // bs) .. p (the current row included); the extent (cache_seqlens)
    is bs n_c + p % bs + 1. n_c <= slots - 1, so the extent is at most slots * bs."""
    if not 0 <= n_c <= min(slots - 1, p // bs):
        raise KStepError("C1.3 layout: row %d with %d complete blocks does not fit %d slots" % (p, n_c, slots))
    tail = p % bs + 1
    return n_c, tail, bs * n_c + tail


# ----------------------------------------------------------------------------------------------------------- selection
def select_blocks(M: int, pq: Optional[Sequence[Any]] = None, pc: Optional[Sequence[Any]] = None, head: int = 0,
                  qk_select: int = QK_SELECT, topk: int = TOPK) -> Tuple[List[int], Tuple[bool, bool]]:
    """12.10.3 P5 for one KV head. M <= topk: every id (no score is read). Else the QK-forced ids are the first
    `qk_select` ids in the order (PQ desc, id asc); PC' = PC with +inf at those ids; the result is the first `topk` ids
    in the order (PC' desc, id asc), returned ascending. Python's sort with the key (-value, id) gives each order (+inf
    first, -inf last). A NaN raises ValueError (head and id named); a result without the own id M - 1 raises KStepError.
    Also returns, as instrumentation, whether the values at the QK cut and at the CIS cut tie."""
    if M <= topk:
        return list(range(M)), (False, False)
    vecs = []
    for name, vec in (("PQ", pq), ("PC", pc)):
        if vec is None or len(vec) != M:
            raise ValueError("C1.3 select: head %d %s needs %d values" % (head, name, M))
        vals = [float(x) for x in vec]
        for i, x in enumerate(vals):
            if math.isnan(x):
                raise ValueError("C1.3 select: head %d %s id %d is NaN" % (head, name, i))
        vecs.append(vals)
    q, c = vecs
    oq = sorted(range(M), key=lambda i: (-q[i], i))
    forced = set(oq[:qk_select])
    c2 = [math.inf if i in forced else c[i] for i in range(M)]
    oc = sorted(range(M), key=lambda i: (-c2[i], i))
    sel = sorted(oc[:topk])
    if M - 1 not in sel:
        raise KStepError("C1.3 select: head %d own id %d is not selected" % (head, M - 1))
    ties = (M > qk_select and q[oq[qk_select - 1]] == q[oq[qk_select]], c2[oc[topk - 1]] == c2[oc[topk]])
    return sel, ties


def complete_blocks(sel: Sequence[int], M: int) -> List[int]:
    """The selected complete blocks: the selection without the own id M - 1, ascending."""
    return [j for j in sorted(sel) if j != M - 1]


# -------------------------------------------------------------------------------------------------------------- ledger
class C13Ledger:
    """A PARTIAL owned-tensor-backing ledger (12.10.5, revision 13). BiasOpLedger(kinds=C13_KINDS) charges, from
    allocation to release, numel x element_size of each FULL OWNED CONTIGUOUS backing that the harness allocates (a
    view, a storage offset or a partial storage is refused), keyed by (owner prefix, name). It is NOT the total
    operational or GPU memory: the model weights, the RoPE table, the P1 temporaries, PQ / PC, metadata (descriptors,
    bind tables, cache_seqlens, index tensors) and extension / library workspace are excluded (contract 12.10.5).
    recount() counts the unique storages that the OWNERS report alive, apart from the ledger's own records. A release is
    proven by the owners dropping their references (the tests check that the tensors are collected), not by the ledger
    dropping its reference alone. No memory-fit or throughput claim."""

    def __init__(self) -> None:
        self.led = BiasOpLedger(kinds=C13_KINDS)

    def own(self, kind: str, key: Any, t: torch.Tensor) -> int:
        nb = t.numel() * t.element_size()
        if (t.numel() == 0 or not t.is_contiguous() or t.storage_offset() != 0
                or t.untyped_storage().nbytes() != nb):
            raise OwnershipViolation("C1.3 ledger: %s %r is not a full owned contiguous backing" % (kind, key))
        return self.led.hold(kind, t, key=key)

    def drop(self, kind: str, key: Any) -> None:
        self.led.release_key(kind, key)

    @property
    def held(self) -> int:
        return self.led.held

    @property
    def peak(self) -> int:
        return self.led.peak

    def by_kind(self) -> Dict[str, int]:
        out = {k: 0 for k in C13_KINDS}
        for kind, _, nb in self.led.outstanding():
            out[kind] += nb
        return out

    @staticmethod
    def recount(owners: Sequence[Any]) -> int:
        """The bytes of the unique storages of the tensors that the owners (objects with owned()) hold now."""
        seen: Dict[int, int] = {}
        for ow in owners:
            for _, _, t in ow.owned():
                s = t.untyped_storage()
                seen.setdefault(s.data_ptr(), s.nbytes())
        return sum(seen.values())

    def check(self, owners: Sequence[Any], where: str) -> int:
        n = self.recount(owners)
        if n != self.led.held:
            raise OwnershipViolation("C1.3 ledger recount at %s: the owners hold %d B, the ledger %d B"
                                     % (where, n, self.led.held))
        return n


class Backings:
    """The tensors that one harness object owns; each is charged in the ledger (if any) from add() to release()."""

    def __init__(self, ledger: Optional[C13Ledger], prefix: Any) -> None:
        self.ledger, self.prefix = ledger, prefix
        self.t: Dict[str, Tuple[str, torch.Tensor]] = {}

    def add(self, kind: str, name: str, t: torch.Tensor) -> torch.Tensor:
        if name in self.t:
            raise OwnershipViolation("C1.3 backings %r: %s is already owned" % (self.prefix, name))
        if self.ledger is not None:
            self.ledger.own(kind, (self.prefix, name), t)
        self.t[name] = (kind, t)
        return t

    def owned(self) -> List[Tuple[str, Any, torch.Tensor]]:
        return [(kind, (self.prefix, n), t) for n, (kind, t) in self.t.items()]

    def release(self) -> None:
        for n in list(self.t):
            kind = self.t.pop(n)[0]
            if self.ledger is not None:
                self.ledger.drop(kind, (self.prefix, n))


# -------------------------------------------------------------------------------------------------------------- stores
class C13Stores:
    """12.10.5: the harness's layer-0 stores in the A4 layout, for one case. Arena K / V (P bs, D), bias (P bs,) and
    pgen_t (P,) int64; ring K / V (Q, D), bias (Q,) and rgen_t (Q,) int64; P = H NB + 2 (NB = the complete blocks at the
    case's last position), Q = H bs + 2. Every store row starts as NaN (poison); the generations start at 0.
      produce(q, k, v, c)   step 1: the ring entries entry(q, h) get the bytes first, then the generation (host table
                            and rgen_t); an entry must be FREE.
      promote(j)            step 2: per head, block j's bs ring rows copy to arena position phys(j, h), then the
                            generation; then the entries return to FREE. The caller runs it after the last reader of
                            those entries.
      alloc_chain(n)        the chain tensors CK (n, H, D) and CC (H, 1, n) (12.10.3 P3), charged as c13_chain.
      release()             drops every backing (ledger and object references)."""

    def __init__(self, NB: int, H: int = 2, D: int = 128, bs: int = BS, dtype: torch.dtype = torch.bfloat16,
                 device: Any = "cpu", ledger: Optional[C13Ledger] = None, case: str = "?") -> None:
        self.NB, self.H, self.D, self.bs, self.dtype, self.case = NB, H, D, bs, dtype, case
        self.P, self.Q = H * NB + 2, H * bs + 2
        if math.gcd(P_MULT, self.P) != 1 or math.gcd(Q_MULT, self.Q) != 1:
            raise ConfigError("C1.3 stores: the address maps need gcd(%d, P %d) = gcd(%d, Q %d) = 1"
                              % (P_MULT, self.P, Q_MULT, self.Q))
        self.b = Backings(ledger, (case, "stores"))
        nan = float("nan")
        poison = lambda *shape: torch.full(shape, nan, dtype=dtype, device=device)  # noqa: E731
        gens = lambda n: torch.zeros((n,), dtype=torch.int64, device=device)       # noqa: E731
        self.arena_k = self.b.add("c13_arena", "arena_k", poison(self.P * bs, D))
        self.arena_v = self.b.add("c13_arena", "arena_v", poison(self.P * bs, D))
        self.arena_c = self.b.add("c13_arena", "arena_c", poison(self.P * bs))
        self.pgen_t = self.b.add("c13_arena", "pgen_t", gens(self.P))
        self.ring_k = self.b.add("c13_ring", "ring_k", poison(self.Q, D))
        self.ring_v = self.b.add("c13_ring", "ring_v", poison(self.Q, D))
        self.ring_c = self.b.add("c13_ring", "ring_c", poison(self.Q))
        self.rgen_t = self.b.add("c13_ring", "rgen_t", gens(self.Q))
        self.ck: Optional[torch.Tensor] = None
        self.cc: Optional[torch.Tensor] = None
        self.pgen = [0] * self.P
        self.rgen = [0] * self.Q
        self.arena_owner: List[Optional[Tuple[int, int]]] = [None] * self.P
        self.ring_owner: List[Optional[Tuple[int, int]]] = [None] * self.Q
        self.arena_bind: Dict[Tuple[int, int], Tuple[int, int]] = {}     # (block, head) -> (phys, pgen)
        self.ring_bind: Dict[Tuple[int, int], Tuple[int, int]] = {}      # (position, head) -> (entry, rgen)

    def owned(self) -> List[Tuple[str, Any, torch.Tensor]]:
        return self.b.owned()

    def phys_of(self, j: int, h: int) -> int:
        if not (0 <= j < self.NB and 0 <= h < self.H):
            raise ConfigError("C1.3 stores: block %d head %d outside NB %d / H %d" % (j, h, self.NB, self.H))
        return (P_MULT * (self.H * j + h) + P_ADD) % self.P

    def entry_of(self, q: int, h: int) -> int:
        return (Q_MULT * (self.H * (q % self.bs) + h) + Q_ADD) % self.Q

    def alloc_chain(self, n: int) -> None:
        dev = self.arena_k.device
        self.ck = self.b.add("c13_chain", "ck", torch.full((n, self.H, self.D), float("nan"), dtype=self.dtype,
                                                           device=dev))
        self.cc = self.b.add("c13_chain", "cc", torch.full((self.H, 1, n), float("nan"), dtype=self.dtype, device=dev))

    def produce(self, q: int, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> None:
        """Step 1: k, v (H, D) and c (H,) of position q into the ring entries entry(q, h)."""
        for t in (k, v, c):
            if t.dtype != self.dtype:
                raise ConfigError("C1.3 stores: position %d payload dtype %s, stores %s" % (q, t.dtype, self.dtype))
        for h in range(self.H):
            e = self.entry_of(q, h)
            if self.ring_owner[e] is not None or (q, h) in self.ring_bind:
                raise OwnershipViolation("C1.3 ring: entry %d for (position %d, head %d) is bound to %s"
                                         % (e, q, h, self.ring_owner[e]))
            self.ring_k[e].copy_(k[h])
            self.ring_v[e].copy_(v[h])
            self.ring_c[e].copy_(c[h])
            self.rgen[e] += 1
            self.rgen_t[e].fill_(self.rgen[e])
            self.ring_owner[e] = (q, h)
            self.ring_bind[(q, h)] = (e, self.rgen[e])

    def promote(self, j: int) -> None:
        """Step 2 for block j (complete): ring -> arena per head, then the generation, then the entries are FREE."""
        bs = self.bs
        for h in range(self.H):
            ph = self.phys_of(j, h)
            if self.arena_owner[ph] is not None:
                raise OwnershipViolation("C1.3 arena: position %d for (block %d, head %d) is bound to %s"
                                         % (ph, j, h, self.arena_owner[ph]))
            ents = []
            for q in range(j * bs, (j + 1) * bs):
                b = self.ring_bind.get((q, h))
                if b is None:
                    raise OwnershipViolation("C1.3 promote: block %d head %d position %d is not in the ring"
                                             % (j, h, q))
                ents.append(b[0])
            idx = torch.tensor(ents, dtype=torch.int64, device=self.ring_k.device)
            rows = slice(ph * bs, (ph + 1) * bs)
            self.arena_k[rows].copy_(self.ring_k.index_select(0, idx))
            self.arena_v[rows].copy_(self.ring_v.index_select(0, idx))
            self.arena_c[rows].copy_(self.ring_c.index_select(0, idx))
            self.pgen[ph] += 1
            self.pgen_t[ph].fill_(self.pgen[ph])
            self.arena_owner[ph] = (j, h)
            self.arena_bind[(j, h)] = (ph, self.pgen[ph])
            for q in range(j * bs, (j + 1) * bs):
                e = self.ring_bind.pop((q, h))[0]
                self.ring_owner[e] = None

    def release(self) -> None:
        self.b.release()
        self.arena_k = self.arena_v = self.arena_c = self.pgen_t = None
        self.ring_k = self.ring_v = self.ring_c = self.rgen_t = None
        self.ck = self.cc = None


# ---------------------------------------------------------------------------------------------------------- descriptor
@dataclass
class AddrItem:
    block: int
    phys: Optional[int]           # None for an unreadable (masked) block
    pgen: Optional[int]
    readable: bool


@dataclass
class RingItem:
    pos: int
    entry: int
    rgen: int


@dataclass
class RowDesc:
    case: str
    p: int
    n_c: int
    tail_len: int
    extent: int
    addr: List[List[AddrItem]]    # per KV head, increasing block id
    ring: List[List[RingItem]]    # per KV head, increasing position (the own block up to p)


def _check_selection(p: int, H: int, complete_sel: Sequence[Sequence[int]], masked: Set[Tuple[int, int]], bs: int,
                     who: str) -> int:
    own = p // bs
    if len(complete_sel) != H or len({len(s) for s in complete_sel}) != 1:
        raise KStepError("C1.3 %s: row %d needs one selection per head, of equal length" % (who, p))
    for h in range(H):
        sel = list(complete_sel[h])
        if sel != sorted(set(sel)) or any(not 0 <= j < own for j in sel):
            raise KStepError("C1.3 %s: row %d head %d selection %s is not increasing complete blocks below %d"
                             % (who, p, h, sel, own))
    for (h, j) in masked:
        if not (0 <= h < H and j in complete_sel[h]):
            raise KStepError("C1.3 %s: row %d masked (head %d, block %d) is not selected" % (who, p, h, j))
    return len(complete_sel[0])


def build_descriptor(stores: C13Stores, case: str, p: int, complete_sel: Sequence[Sequence[int]],
                     masked: Any = ()) -> RowDesc:
    """Step 3 (the capture): per KV head, Row.addr = the selected complete blocks with their bound (phys, pgen) (an
    unreadable block is masked: no address) and Row.ring = the own block's positions bs (p // bs) .. p with their bound
    (entry, rgen). An unbound readable block or ring position raises KStepError."""
    H, bs = stores.H, stores.bs
    msk = set(masked)
    n_c = _check_selection(p, H, complete_sel, msk, bs, "descriptor")
    _, tail, extent = row_layout(p, n_c, bs)
    own = p // bs
    addr: List[List[AddrItem]] = []
    ring: List[List[RingItem]] = []
    for h in range(H):
        items = []
        for j in complete_sel[h]:
            if (h, j) in msk:
                items.append(AddrItem(j, None, None, False))
                continue
            b = stores.arena_bind.get((j, h))
            if b is None:
                raise KStepError("C1.3 descriptor: case %s row %d block %d head %d is not bound in the arena"
                                 % (case, p, j, h))
            items.append(AddrItem(j, b[0], b[1], True))
        rits = []
        for q in range(own * bs, p + 1):
            b = stores.ring_bind.get((q, h))
            if b is None:
                raise KStepError("C1.3 descriptor: case %s row %d position %d head %d is not bound in the ring"
                                 % (case, p, q, h))
            rits.append(RingItem(q, b[0], b[1]))
        addr.append(items)
        ring.append(rits)
    return RowDesc(case, p, n_c, tail, extent, addr, ring)


# ---------------------------------------------------------------------------------------------------------- candidate
class Staging:
    """The candidate's cache of one row (12.10.4): k, v (1, SLOTS bs, H, D) and c (1, SLOTS bs, H), zero-filled, and
    the int64 generation copies `gens` (one per read), all charged as c13_staging; cs = cache_seqlens (int32 (1,);
    metadata, not charged); `reads` (one record per generation copy: where, head, loc, expected, label, idx); `ops` (the
    issue order: ("data", g) after a read's data copies, ("gen", g) after its generation copy, ("mask", h, i));
    `counts` (instrumentation)."""

    def __init__(self, ledger: Optional[C13Ledger], case: str, p: int) -> None:
        self.case, self.p = case, p
        self.b = Backings(ledger, (case, p, "staging"))
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.gens: Optional[torch.Tensor] = None
        self.cs: Optional[torch.Tensor] = None
        self.reads: List[Dict[str, Any]] = []
        self.ops: List[Tuple] = []
        self.counts = dict(arena_items=0, ring_items=0, masked_items=0, kv_rows=0, bias_rows=0, padding_rows=0)

    def owned(self) -> List[Tuple[str, Any, torch.Tensor]]:
        return self.b.owned()

    def cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k, self.v, self.c, self.cs

    def release(self) -> None:
        self.b.release()
        self.k = self.v = self.c = self.gens = self.cs = None


def gather_candidate(stores: C13Stores, desc: RowDesc, ledger: Optional[C13Ledger] = None, slots: int = SLOTS,
                     mask_bias: float = MASK_BIAS,
                     substitute: Optional[Dict[Tuple[str, int, int], Tuple[int, int]]] = None) -> Staging:
    """Step 5, from the descriptor and the stores only. Slot i of head h holds Row.addr item i (bs exact byte copies of
    K, V and bias from arena position phys); a masked item keeps K / V at 0, gets mask_bias on its bs bias rows and
    reads no store; slot n_c holds Row.ring item r at row n_c bs + r (one copy each of K, V, bias from the entry). After
    each read's data copies, its generation (pgen_t[phys] or rgen_t[entry]) is copied into gens. substitute (the matched
    controls' harness hook, 12.10.7; None otherwise):
      ("addr", h, i): (phys, pgen)   Row.addr item i of head h reads K, V and bias at phys; its generation copy is
                                     checked against pgen (the substitute's captured generation);
      ("bias", h, r): (entry, rgen)  Row.ring item r of head h takes ONLY its bias from entry (K, V kept); one more
                                     generation copy, checked against rgen.
    A substitute that matches no item raises KStepError."""
    H, D, bs = stores.H, stores.D, stores.bs
    dev, dt = stores.arena_k.device, stores.dtype
    sub = dict(substitute or {})
    for key in sub:
        if key[0] not in ("addr", "bias"):
            raise KStepError("C1.3 gather: unknown substitute %r" % (key,))
    for h in range(H):
        if len(desc.addr[h]) != desc.n_c or len(desc.ring[h]) != desc.tail_len:
            raise KStepError("C1.3 gather: case %s row %d head %d descriptor does not match n_c %d / tail %d"
                             % (desc.case, desc.p, h, desc.n_c, desc.tail_len))
    n_reads = (sum(1 for h in range(H) for a in desc.addr[h] if a.readable) + sum(len(r) for r in desc.ring)
               + sum(1 for key in sub if key[0] == "bias"))
    R = slots * bs
    st = Staging(ledger, desc.case, desc.p)
    st.k = st.b.add("c13_staging", "k", torch.zeros((1, R, H, D), dtype=dt, device=dev))
    st.v = st.b.add("c13_staging", "v", torch.zeros((1, R, H, D), dtype=dt, device=dev))
    st.c = st.b.add("c13_staging", "c", torch.zeros((1, R, H), dtype=dt, device=dev))
    st.gens = st.b.add("c13_staging", "gens", torch.zeros((n_reads,), dtype=torch.int64, device=dev))
    used: Set[Tuple[str, int, int]] = set()
    g = 0
    for h in range(H):
        for i, a in enumerate(desc.addr[h]):
            rows = slice(i * bs, (i + 1) * bs)
            if not a.readable:
                st.c[0, rows, h].fill_(mask_bias)
                st.ops.append(("mask", h, i))
                st.counts["masked_items"] += 1
                continue
            ph, pg = a.phys, a.pgen
            if ("addr", h, i) in sub:
                ph, pg = sub[("addr", h, i)]
                used.add(("addr", h, i))
            src = slice(ph * bs, (ph + 1) * bs)
            st.k[0, rows, h].copy_(stores.arena_k[src])
            st.v[0, rows, h].copy_(stores.arena_v[src])
            st.c[0, rows, h].copy_(stores.arena_c[src])
            st.ops.append(("data", g))
            st.gens[g].copy_(stores.pgen_t[ph])
            st.ops.append(("gen", g))
            st.reads.append(dict(where="arena", head=h, loc=ph, expected=pg, label="block %d" % a.block, idx=g))
            g += 1
            st.counts["arena_items"] += 1
            st.counts["kv_rows"] += bs
            st.counts["bias_rows"] += bs
        base = desc.n_c * bs
        for r, it in enumerate(desc.ring[h]):
            row = base + r
            be, brg = it.entry, None
            if ("bias", h, r) in sub:
                be, brg = sub[("bias", h, r)]
                used.add(("bias", h, r))
            st.k[0, row, h].copy_(stores.ring_k[it.entry])
            st.v[0, row, h].copy_(stores.ring_v[it.entry])
            st.c[0, row, h].copy_(stores.ring_c[be])
            ids = [g] if brg is None else [g, g + 1]
            for x in ids:
                st.ops.append(("data", x))
            st.gens[g].copy_(stores.rgen_t[it.entry])
            st.ops.append(("gen", g))
            st.reads.append(dict(where="ring", head=h, loc=it.entry, expected=it.rgen, label="position %d" % it.pos,
                                 idx=g))
            if brg is not None:
                st.gens[g + 1].copy_(stores.rgen_t[be])
                st.ops.append(("gen", g + 1))
                st.reads.append(dict(where="ring", head=h, loc=be, expected=brg, label="bias of position %d" % it.pos,
                                     idx=g + 1))
            g += len(ids)
            st.counts["ring_items"] += 1
            st.counts["kv_rows"] += 1
            st.counts["bias_rows"] += 1
    if used != set(sub):
        raise KStepError("C1.3 gather: substitute %s matches no item" % sorted(set(sub) - used))
    if g != n_reads:
        raise KStepError("C1.3 gather: %d generation copies, %d expected" % (g, n_reads))
    st.counts["padding_rows"] = (R - desc.extent) * H
    st.cs = torch.tensor([desc.extent], dtype=torch.int32, device=dev)
    return st


# -------------------------------------------------------------------------------------------------------------- Twin R
def twin_r_cache(LS: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], p: int, complete_sel: Sequence[Sequence[int]],
                 masked: Any = (), slots: int = SLOTS, bs: int = BS,
                 mask_bias: float = MASK_BIAS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Twin R (12.10.4): the same layout written from the logical state LS = (K_L (N, H, D), V_L (N, H, D), C_L (N, H)),
    the selection and the mask set ONLY; it never reads the stores, a descriptor or a candidate tensor. Returns (k, v,
    c, cs); reference instrumentation, not charged."""
    K_L, V_L, C_L = LS
    H, D = int(K_L.shape[1]), int(K_L.shape[2])
    msk = set(masked)
    n_c = _check_selection(p, H, complete_sel, msk, bs, "Twin R")
    _, tail, extent = row_layout(p, n_c, bs, slots)
    own = p // bs
    R = slots * bs
    k = torch.zeros((1, R, H, D), dtype=K_L.dtype, device=K_L.device)
    v = torch.zeros((1, R, H, D), dtype=V_L.dtype, device=V_L.device)
    c = torch.zeros((1, R, H), dtype=C_L.dtype, device=C_L.device)
    for h in range(H):
        for i, j in enumerate(complete_sel[h]):
            rows, src = slice(i * bs, (i + 1) * bs), slice(j * bs, (j + 1) * bs)
            if (h, j) in msk:
                c[0, rows, h].fill_(mask_bias)
                continue
            k[0, rows, h].copy_(K_L[src, h])
            v[0, rows, h].copy_(V_L[src, h])
            c[0, rows, h].copy_(C_L[src, h])
        t0 = n_c * bs
        k[0, t0:t0 + tail, h].copy_(K_L[own * bs:p + 1, h])
        v[0, t0:t0 + tail, h].copy_(V_L[own * bs:p + 1, h])
        c[0, t0:t0 + tail, h].copy_(C_L[own * bs:p + 1, h])
    cs = torch.tensor([extent], dtype=torch.int32, device=K_L.device)
    return k, v, c, cs


# -------------------------------------------------------------------------------------------------------------- checks
def check_store_readback(stores: C13Stores, desc: RowDesc, LS: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> int:
    """Step 4 (CP2; instrumentation): each readable descriptor item's store bytes equal LS at (block or position, head);
    the first difference raises KStepError "C1.3 store mismatch: ...". Returns the items checked."""
    K_L, V_L, C_L = LS
    bs, n = stores.bs, 0
    for h in range(stores.H):
        for i, a in enumerate(desc.addr[h]):
            if not a.readable:
                continue
            src, ref = slice(a.phys * bs, (a.phys + 1) * bs), slice(a.block * bs, (a.block + 1) * bs)
            for name, got, want in (("K", stores.arena_k[src], K_L[ref, h]), ("V", stores.arena_v[src], V_L[ref, h]),
                                    ("bias", stores.arena_c[src], C_L[ref, h])):
                if not torch.equal(got, want):
                    raise KStepError("C1.3 store mismatch: case %s row %d head %d item %d (block %d, arena position %d)"
                                     " tensor %s" % (desc.case, desc.p, h, i, a.block, a.phys, name))
            n += 1
        for r, it in enumerate(desc.ring[h]):
            for name, got, want in (("K", stores.ring_k[it.entry], K_L[it.pos, h]),
                                    ("V", stores.ring_v[it.entry], V_L[it.pos, h]),
                                    ("bias", stores.ring_c[it.entry], C_L[it.pos, h])):
                if not torch.equal(got, want):
                    raise KStepError("C1.3 store mismatch: case %s row %d head %d item %d (position %d, ring entry %d)"
                                     " tensor %s" % (desc.case, desc.p, h, desc.n_c + r, it.pos, it.entry, name))
            n += 1
    return n


def check_identity(st: Staging) -> int:
    """Step 7, after the read-completion gate and BEFORE any gathered value is compared or published: every copied
    generation equals the expected one, else StaleReader "stale consumption: ...". Returns the reads checked."""
    obs = st.gens.cpu().tolist()
    for rec in st.reads:
        o = int(obs[rec["idx"]])
        if o != int(rec["expected"]):
            what = ("ring entry %d rgen" if rec["where"] == "ring" else "arena position %d pgen") % rec["loc"]
            raise StaleReader("stale consumption: case %s row %d head %d %s expected %d, read %d (%s)"
                              % (st.case, st.p, rec["head"], what, int(rec["expected"]), o, rec["label"]))
    return len(st.reads)


def compare_cache(cand: Sequence[torch.Tensor], twin: Sequence[torch.Tensor], case: str, p: int, bs: int = BS) -> None:
    """CP4: the candidate's (k, v, c, cs) against Twin R's, tensor by tensor in the order K, V, bias, cache_seqlens
    (torch.equal; a NaN never equals). The first difference (head ascending, then cache row ascending) raises
    CacheMismatch "C1.3 cache mismatch: case X row p head h slot i row r tensor T" with a `detail` dict."""
    for name, a, b in zip(("K", "V", "bias"), cand[:3], twin[:3]):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise CacheMismatch("C1.3 cache mismatch: case %s row %d tensor %s shape / dtype %s %s vs %s %s"
                                % (case, p, name, tuple(a.shape), a.dtype, tuple(b.shape), b.dtype))
        if torch.equal(a, b):
            continue
        ne = a.ne(b)
        if ne.dim() == 4:
            ne = ne.any(dim=-1)
        hr = ne[0].t().nonzero()
        if hr.numel() == 0:
            raise CacheMismatch("C1.3 cache mismatch: case %s row %d tensor %s (no element located)" % (case, p, name))
        h, r = int(hr[0][0]), int(hr[0][1])
        e = CacheMismatch("C1.3 cache mismatch: case %s row %d head %d slot %d row %d tensor %s"
                          % (case, p, h, r // bs, r % bs, name))
        e.detail = dict(case=case, row=p, head=h, slot=r // bs, slot_row=r % bs, tensor=name)
        raise e
    a, b = cand[3].cpu(), twin[3].cpu()
    if a.dtype != b.dtype or not torch.equal(a, b):
        raise CacheMismatch("C1.3 cache mismatch: case %s row %d tensor cache_seqlens (%s %s vs %s %s)"
                            % (case, p, a.tolist(), a.dtype, b.tolist(), b.dtype))
