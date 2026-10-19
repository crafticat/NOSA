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
  HOOK-L6 (INTERFACE_BC5 I11; Config.store_gen "consumer" only): the pgen[l] / rgen[l] that attach_arena / attach_ring
                          receive are then the store's CONSUMER mirrors, not the pools' host mirrors (a later attach
                          by TensorStore.attach_gen_mirrors replaces the earlier one). The store writes them only by
                          backend copies, ordered before each binding's arrival (arena: the S0 issue's event) or ready
                          event (ring). So an executor reads them only through its deferred copies (on its stream,
                          after the data), never by a host read at call time; the generation it EXPECTS is the one in
                          Row.addr / Row.ring (the host bind identity).
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

C1.2 SERIALIZED PHYSICAL CONSUMPTION (contract revision 11, 12.9.2; dispatcher Config.consume = "physical"; needs ABI v4,
reads_ring and the attribute consumes_physical = True). The dispatcher calls, per joint call with rows:
  joint_post_consume(l, rows) -> (handle, read_ev)
                          the call's validator reads exactly as joint_post (same registrations, same order); then per
                          (row, head) ONE gather on "compute" of the row's attended positions in the declared logical
                          order (increasing position: Row.addr blocks as bs rows each, merged with Row.ring entries):
                          K (n, D), V (n, D) and bias (n,) from the arena / ring stores, and one identity copy per block
                          (pgen_t) or entry (rgen_t), each copied AFTER its data; read_ev on "compute" after the call's
                          LAST read (the completion event). It also computes, at call time, the eager reference from
                          Row.kv (instrumentation). The handle is a dict the dispatcher owns from then on.
  gather_tensors(handle)  [((row index, head, part), tensor)]: every gather backing (part "k", "v", "c", "id").
  consumed_identities(handle)
                          after completion: one record per gathered block / entry: where ("arena" / "ring"), layer,
                          addr, expected (the generation captured at call time, independently of the copy), observed
                          (the copied value), uid, pos, kind, head.
  resolve(handle) -> (outs, heads)
                          after completion: per row the output and per head the head result, computed ONLY from the
                          gathered K / V / bias and the row's own phase-1 q and hidden (never from Row.kv).
  reference(handle) -> (outs, heads)
                          the eager reference of the same rows, same order, same arithmetic.
check_consumed_identity(records) is the dispatcher's identity check of those records (StaleReader "stale consumption").

I10 RESET CONTRACT (INTERFACE_BC4 I10, 2026-10-17; frontier.QuiescentReset; dispatcher HOOK-L4). An executor that
retains operational events between calls must let a quiescent reset see and detach them before pooled events are
re-recorded.
It DECLARES the contract IN ITS OWN CLASS BODY (reset_contract = "I10"; reset_contract_of reads type(ex).__dict__ only,
so neither inheritance nor attribute forwarding, such as a wrapper's __getattr__, can declare it for a class that was
not audited) and provides:
  reset_holders() -> [(owner_class, location, event)]
                          READ-ONLY: every OPERATIONAL event reference it keeps between calls (one it will wait on, hand
                          on or test later); owner_class starts with "executor:"; None is not listed. Instrumentation
                          history (logs, counters, eager reference copies, records no later decision reads) is not
                          listed; the class docstring names it.
  detach_completed() -> int
                          MUTATING: refuses (OwnershipViolation "executor detach refused: <n> pending events: [...]")
                          with NO mutation while a listed event is not complete; else sets every listed reference to
                          None and returns the count.
Dispatcher.event_holders appends reset_holders() of a declared executor; Dispatcher.detach_completed_events calls
detach_completed() first after its pre-check (both through the class). An executor without the declaration is refused
by the coordinator. The dispatcher VERIFIES a detach (its count and nothing left) before any dispatcher change; it does
not undo the executor's own changes: a detach that fails its verification may already have changed the executor's own
state (no transactional rollback; INTERFACE_BC5 I12, where such a failure latches the session "failed").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import StaleReader

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
CONSUME_REQUIRED = ("joint_post_consume", "gather_tensors", "consumed_identities", "resolve", "reference")   # C1.2
# I16 / I17 (INTERFACE_BC7, lead, 2026-10-17), OPTIONAL: an executor that reuses its producer workspaces declares
# accepts_handback = True in its OWN class and defines on_handback(hb). With Config.store_handback the dispatcher
# forwards every tensor_store.Handback in op order (after each admission and after each layer call). The executor may
# overwrite a source storage only when store.source_reusable(hb) is True for EVERY handback that names that storage,
# and never while an operation that reads it has no handback yet (the store gives permission per operation; it does not
# merge them). A handback covers the producer's SOURCE only: the ring / arena / gather backings stay protected by
# their own lifetime rules. Without the declaration nothing is forwarded (the handbacks are taken and counted only).


def values_differ(a: Any, b: Any) -> bool:
    """Milestone 10 (INTERFACE_K3X section 3, lead): the consumption comparison of one consumed value against its
    reference. Python values (the toys) compare with != exactly as before; when either is a torch tensor (a real
    model's output), they are equal only as torch.equal tensors of the same dtype and shape (bitwise; no tolerance)."""
    import torch
    if torch.is_tensor(a) or torch.is_tensor(b):
        return not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                    and torch.equal(a, b))
    return a != b


def check_consumed_identity(records: List[Dict[str, Any]]) -> int:
    """C1.2 (12.9.2 step 4), BEFORE any gathered value is used: every gathered block / entry's copied generation
    (observed) equals the generation captured at call time (expected); else StaleReader. Returns the records checked."""
    for rec in records:
        if int(rec["observed"]) != int(rec["expected"]):
            what = ("ring entry %d (layer %d) rgen" if rec["where"] == "ring" else "position %d (layer %d) pgen") % (
                rec["addr"], rec["layer"])
            raise StaleReader("stale consumption: %s expected %d, read %d (row (%s, %d, %s) head %d)"
                              % (what, int(rec["expected"]), int(rec["observed"]), rec["uid"], rec["pos"], rec["kind"],
                                 rec["head"]))
    return len(records)


RESET_CONTRACT = "I10"                           # I10 (docstring): the declared reset contract


def reset_contract_of(ex) -> Optional[str]:
    """I10: the reset contract declared in the executor's OWN class body (type(ex).__dict__), else None. Inherited or
    forwarded attributes do not count: every executor class is audited and declared on its own."""
    return type(ex).__dict__.get("reset_contract")


def reset_methods_missing(ex) -> List[str]:
    """I10: the contract methods (reset_holders, detach_completed) that the executor's classes do not provide as
    callables. As in Python's own attribute lookup, the FIRST class of type(ex).__mro__ that defines the name in its
    __dict__ decides (a subclass that shadows a base method with a non-callable is missing it). A method that only an
    attribute-forwarding __getattr__ (or an instance attribute) would supply does not count. The one caller, the
    dispatcher (HOOK-L4), calls the methods THROUGH THE CLASS (type(ex).reset_holders(ex), type(ex).detach_completed(ex);
    INTERFACE_BC5 I12, 2026-10-17), so an instance attribute of the same name is never called; the coordinator calls
    them only through those dispatcher hooks."""
    out = []
    for m in ("reset_holders", "detach_completed"):
        owner = next((k for k in type(ex).__mro__ if m in vars(k)), None)
        if owner is None or not callable(vars(owner)[m]):
            out.append(m)
    return out


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
    if getattr(ex, "consumes_physical", False):                       # C1.2 (docstring)
        miss += [m for m in CONSUME_REQUIRED if not callable(getattr(ex, m, None))]
    if miss:
        raise TypeError("executor %r lacks %s" % (type(ex).__name__, miss))
    if ex.bs % ex.stride:
        raise ValueError("the compression stride %d must divide the block size %d" % (ex.stride, ex.bs))
    if abi4:
        check_chain_geometry(ex.bs, ex.chain_kernel, ex.stride)
