"""Milestone 10, Track B (INTERFACE_K3X section 2): NosaExecutor, the tensor-native C1.2 consumer of the store's mode
for ANY model of section 1 (the real NOSA-8B model, the tiny NOSA-shaped model, or a test stub). SOURCE ONLY until the
lead's runs; it never imports tests/ and never imports dispatcher.py.

Template. The protocol layers of the research toy executors (tests/kstep_toy.py ArenaToy -> RingArenaToy ->
ConsumingRingArenaToy) and Track B's store hooks (tests/kstep_tensor_fixtures.py TensorPayloadConsumer), moved here
WITHOUT the integer codes: the eager items ARE the model's tensors. The protocol is executor.py's (M2 ARENA PROTOCOL, M2
RING PROTOCOL v3, C1.1 ABI v4, C1.2 SERIALIZED PHYSICAL CONSUMPTION, I10) and tensor_store.py's DISPATCHER ADAPTER
(tensor_payload, fresh_outputs, kv_dtype / bias_dtype, store_device, prefill_tensors, payload, values; I9
producer_ready).

Two backend modes (milestone-10 review fixes). The "host" mode is the cpupack_core CpuBackend (its class name; also
behind a test wrapper that keeps it in its `be` attribute: tensor_store.is_host_backend), whose streams are DEFERRED
queues of host copies: torch ops run eagerly, so only backend copies (memcpy) are stream-ordered. Every other backend
(the CudaBackend, or any CpuBackend-shaped backend of another class) is a "device" mode: torch ops run on the current
stream, and the backend's streams are independent (non-blocking) streams. Revision 43 (M1): in the device mode a
consuming or validating call starts with "compute" waiting an event of the current stream (the Row.kv items that its
validator expectations stack were cloned there by the dispatcher).

  NosaExecutor(model, check_identity=True, ready_stream=None, gather_events=False, check_divergence=None,
               gather=None, debug_cap=100000)
      The constructor of the lead's builders (kstep_toy.build passes the model as `toy`; store_kmod calls
      executor_cls(ex.toy, check_identity=ex.check_identity)). Unknown attributes are forwarded to the model (as
      ArenaToy.__getattr__ does); "model" and dunder names are never forwarded. check_divergence None: on in the host
      mode, off in the device mode (the dispatcher's own C1.2 comparison runs anyway). gather None: "copies" in the host
      mode, "batched" in the device mode (below).
  attributes      abi = 4, reads_arena, reads_ring, consumes_physical, tensor_payload, fresh_outputs = True (every
                  prefill / payload tensor is a fresh storage that it never touches again), producer_ready = True (I9:
                  prefill_tensors and a producer payload return (k, v, c, ready)), reset_contract = "I10"; from the
                  model: L, H, D, V, bs, stride, max_sel, chain_kernel; kv_dtype = bias_dtype = model.dtype,
                  store_device = model.device.
  forwarded       joint_pre, select, compress, embed, dist, prefill: the model's own methods, unchanged.
  attach_arena / attach_bias / attach_ring
                  the protocol's store attachments. The generation mirrors are the pools' host mirrors, or the store's
                  consumer mirrors under HOOK-L6 (Config.store_gen "consumer"). In the DEVICE mode the identity copies
                  must read stream-ordered device mirrors: a call refuses (ConfigError, before any read of the call)
                  when a generation mirror it reads is a host (CPU) tensor and check_identity is on, because a copy from
                  a host mirror runs at call time, so the identity check would be vacuous. The dispatcher attaches the
                  pools' host mirrors first and re-attaches the consumer mirrors later (HOOK-L6), so the refusal is
                  at the first call, not at an attach.
  joint_post(l, rows) -> (outs, read_ev)
                  the validator reads (below), outputs from Row.kv (the non-consuming arena / ring protocol).
  validator reads exactly the registrations and order of RingArenaToy._validator_reads. The EXPECTED bytes of an
                  arena read are the row's eager Row.kv items of that block (as the ring validator's expectation is the
                  eager value), never a clone of the store taken inside the stream context.
  joint_post_consume(l, rows) -> (handle, read_ev)
                  exactly the validator reads of joint_post, then per (row, head) ONE gather on "compute" of K (n, D),
                  V (n, D), bias (n,) in the declared logical order (Row.addr blocks as bs rows each, merged with
                  Row.ring entries, increasing position), with ONE identity copy per block (pgen) or entry (rgen) AFTER
                  its data; read_ev on "compute" after the call's LAST read. gather "copies": per block / entry backend
                  memcpys (the host mode: deferred, as the toy); gather "batched": per (row, head) index_select /
                  index_copy_ kernels from precomputed row indices on "compute" (data first, then the identity rows),
                  the same bytes and order. On the host backend the batched gather runs EAGERLY (torch ops are not
                  deferred there): it is correct only when no producer copy of the stores it reads is still queued; the
                  default never chooses it there (tests use it on complete stores to compare the two). The handle is a
                  dict that the dispatcher owns; this executor keeps no reference to it.
  gather_tensors / consumed_identities
                  as the protocol (every gather backing; one identity record per block / entry, expected = the
                  Row.addr / Row.ring generation captured at call time, observed = the copied value; ONE host transfer
                  of each gather's identity tensor).
  resolve(handle) -> (outs, heads)
                  per row, in call order: heads[h] = model.head_out(l, q_h, K, V, C) on the GATHERED tensors only, then
                  out = model.mix(l, x, heads). It never reads Row.kv for a result. CUDA gather backings are marked
                  record_stream(current stream) before use (the dispatcher drops them after the resolution).
  reference(handle) -> (outs, heads)
                  the same ops in the same order from Row.kv (the eager tensors, stacked in Row.kv order).
  check_gathered(handle)
                  READ-ONLY detector for a caller (the dispatcher does not call it today): after the caller's wait on
                  read_ev and its identity check, the first gathered K / V / bias row that differs (torch.equal) from
                  Row.kv raises CacheMismatch "C1.2 gathered cache mismatch: call <c> layer <l> row (<uid>, <pos>,
                  <kind>) head <h> position <p> tensor <K | V | bias>". Returns the gathers checked.
  prefill_tensors(kv) -> (k, v, c, ready)   per layer (n, H, D), (n, H, D) and (n, H): FRESH tensors of the model's
                  tensors kv[pos][l][h] in kv_dtype on store_device (ONE stack per layer and tensor kind);
  payload(kvv, at=None)                     the item's (k, v, c) as FRESH tensors (D,), (D,), (); a producer (at
                  given) gets (k, v, c, ready), an expectation (at None) (k, v, c);
  values(k, v, c)                           the eager items [(k_i, v_i, c_i)] of rows read from the stores: views of the
                  caller's tensors (the dispatcher hands clones of the store bytes), never Python lists.
  producer ordering (I9; milestone-10 review MAJOR-1)
                  host mode: the outputs are eager host tensors, complete when returned: ready None, or, with a
                  ready_stream, an event recorded on that (deferred) stream after them (as before).
                  device mode, every producer call: ev0 = be.event_rec(None) on the CURRENT stream (after the model's
                  writes of the call's inputs); with a ready_stream: be.stream_wait(ready_stream, ev0), then the copies
                  into fresh outputs inside be.stream(ready_stream) (backend memcpys), then ready = be.event_rec(
                  ready_stream); without one: the copies on the current stream, then ready = be.event_rec(None). So
                  ready is NEVER None in the device mode, and the store's copies (which wait ready) are ordered after
                  the model's writes and the copies. A prefill's per-layer stack runs inside that stream context too.

Instrumentation (kept in self.debug; it NEVER decides; nothing reads it for a decision; every list is bounded by
debug_cap through _log, dropped counts the rest; NO backend event is kept after the call: booleans and times only):
  first_divergence / divergences
                  per call, the FIRST place where resolve and reference differ: dict(call, layer, uid, pos, kind,
                  head, op, index), op in "gather_k" | "gather_v" | "gather_c" (a gathered row differs from Row.kv;
                  index = the position), "head_out" (a head), "mix" (the row output; head None). The comparison runs
                  when both results of a call exist (whichever of resolve / reference comes second), with torch.equal.
  identities      every consumed identity record (expected / observed generations), call, at consumed_identities.
  gathers         per gather: call, layer, uid, pos, kind, head, n positions, bytes, and, set by resolve: consumed True,
                  events (gather_events True: an event was recorded on "compute" right before the gather's first copy
                  and right after its last), ready_done / read_done (those events' ev_done at resolve) and gather_ns
                  (the time between them when both completed: CUDA elapsed_time, or the CpuBackend completion times);
                  the events stay in the dispatcher-owned handle only.
  calls           per call: call, layer, rows, gathers, bytes, pack_ns (building the gather plan and the backings),
                  gather_ns (enqueueing the copies), attn_ns / mix_ns (resolve's head_out / mix), ref_attn_ns /
                  ref_mix_ns (reference's): host wall time from time.perf_counter_ns (on the GPU without a synchronize
                  this is launch time, not kernel time).
  gathered_bytes_peak
                  the largest bytes of one call's gather backings (K, V, bias, identity copies).

I10 (reset contract; declared in THIS class body). Audit: it keeps the model (weights and the model's own state; the
section-1 model API holds no backend event), check_identity, the backend `be`, the attached stores and generation
mirrors (ak, av, ac, pgen, rk, rv, rc, rgen), the register callbacks, disp (the harness's dispatcher handle), the
counters and the debug record (instrumentation history WITHOUT event references). Each call's handle and read_ev go to
the dispatcher and are not kept; ready events are returned, not kept. So it keeps NO operational event between calls:
reset_holders() == [], detach_completed() == 0.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .backend import ev_done
from .errors import CacheMismatch, ConfigError
from .tensor_store import is_host_backend

DIVERGE_OPS = ("gather_k", "gather_v", "gather_c", "head_out", "mix")   # the debug record's op names, in check order
GATHER_MODES = ("copies", "batched")
HOST_BACKEND = "CpuBackend"                    # the class name of the deferred host backend (is_host_backend)


class NosaExecutor:
    """Milestone 10 (INTERFACE_K3X section 2): the tensor-native C1.2 consumer over a section-1 model (module
    docstring)."""
    abi = 4
    reads_arena = True
    reads_ring = True
    consumes_physical = True
    tensor_payload = True
    fresh_outputs = True
    producer_ready = True
    reset_contract = "I10"

    def __init__(self, model: Any, check_identity: bool = True, ready_stream: Optional[str] = None,
                 gather_events: bool = False, check_divergence: Optional[bool] = None, gather: Optional[str] = None,
                 debug_cap: int = 100000):
        if gather is not None and gather not in GATHER_MODES:
            raise ConfigError("NosaExecutor: gather %r is not one of %s" % (gather, GATHER_MODES))
        self.model = model
        self.check_identity = bool(check_identity)
        self.L, self.H, self.D, self.V = model.L, model.H, model.D, model.V
        self.bs, self.stride, self.max_sel = model.bs, model.stride, model.max_sel
        self.chain_kernel = model.chain_kernel
        self.kv_dtype = self.bias_dtype = model.dtype
        self.store_device = model.device
        self.ready_stream = ready_stream
        self.gather_events = bool(gather_events)
        self.check_divergence = check_divergence
        self.gather = gather
        self.debug_cap = int(debug_cap)
        self.be = self.ak = self.av = self.pgen = self.register = None
        self.ac = self.rc = None
        self.rk = self.rv = self.rgen = self.register_ring = None
        self.disp = None                        # the dispatcher (set by the harness): call ids for the debug record
        self.reads = self.ring_reads = 0        # validator reads (arena blocks, ring entries)
        self.gathers = self.consume_calls = self.consumed_rows = 0
        self.prefills = self.payloads = 0
        self.debug: Dict[str, Any] = dict(first_divergence=None, divergences=[], identities=[], gathers=[], calls=[],
                                          gathered_bytes_peak=0, dropped=0)

    # -- forwarding (the lead's builders; ArenaToy.__getattr__)
    def __getattr__(self, name: str):
        if name == "model" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.model, name)

    def joint_pre(self, l, items):
        return self.model.joint_pre(l, items)

    def select(self, l, h, q_h, cks, n_cand):
        return self.model.select(l, h, q_h, cks, n_cand)

    def compress(self, elems):
        return self.model.compress(elems)

    def embed(self, token):
        return self.model.embed(token)

    def dist(self, x):
        return self.model.dist(x)

    def probs(self, x):
        """S3 (contract 14.7; Config.dist "tensor"): the model's float32 probability tensor, whose Fraction list is
        dist(x) (an explicit forward; __getattr__ would also reach it)."""
        return self.model.probs(x)

    def prefill(self, tokens):
        return self.model.prefill(tokens)

    # -- I10
    def reset_holders(self) -> List[Tuple[str, Tuple, Any]]:
        """I10, READ-ONLY: no operational event is kept between calls (the audit in the module docstring)."""
        return []

    def detach_completed(self) -> int:
        """I10: nothing is listed, so nothing is detached."""
        return 0

    # -- attachments (the protocol)
    def attach_arena(self, be, ak, av, pgen, register) -> None:
        self.be, self.ak, self.av, self.pgen, self.register = be, ak, av, pgen, register

    def attach_bias(self, be, ac, rc) -> None:
        self.ac, self.rc = ac, rc

    def attach_ring(self, be, rk, rv, rgen, register_ring) -> None:
        self.rk, self.rv, self.rgen, self.register_ring = rk, rv, rgen, register_ring

    # -- modes and helpers
    def host_mode(self) -> bool:
        """True for the deferred host backend (tensor_store.is_host_backend: the cpupack_core CpuBackend, also behind a
        test wrapper that keeps it in `be`) or before any attach (module docstring)."""
        return self.be is None or is_host_backend(self.be)

    def _enter_compute(self) -> None:
        """M1 (revision 43): in the device mode, at the entry of a consuming / validating call, "compute" waits an
        event recorded on the CURRENT stream, so its reads (and the stacks of the Row.kv items, which the dispatcher
        cloned on the current stream) come after everything the current stream did before the call."""
        if self.host_mode():
            return
        ev = self.be.event_rec(None)
        self.be.stream_wait("compute", ev)

    def _divergence_on(self) -> bool:
        return self.host_mode() if self.check_divergence is None else bool(self.check_divergence)

    def _gather_mode(self) -> str:
        if self.gather is not None:
            return self.gather
        return "copies" if self.host_mode() else "batched"

    def _call_id(self) -> Any:
        return getattr(self.disp, "call_id", None)

    def _arenas(self, l: int):
        return (self.ak[l], self.av[l]) + ((self.ac[l],) if self.ac is not None else ())

    def _log(self, name: str, rec: Dict[str, Any]) -> None:
        lst = self.debug[name]
        if len(lst) < self.debug_cap:
            lst.append(rec)
        else:
            self.debug["dropped"] += 1

    def _check_mirrors(self, l: int, rows) -> None:
        """MINOR-2: in the device mode, a call whose identity copies would read a HOST generation mirror is refused
        (ConfigError) before any read of the call (module docstring, attach_arena). A backend that declares
        orders_host_copies = True (its copies of host tensors are ordered on its named streams: a test backend over the
        deferred host backend) accepts host mirrors."""
        if self.host_mode() or not self.check_identity or getattr(self.be, "orders_host_copies", False) is True:
            return
        bad = []
        if any(row.addr for row in rows) and self.pgen[l].device.type == "cpu":
            bad.append("pgen")
        if any(row.ring for row in rows) and self.rgen is not None and self.rgen[l].device.type == "cpu":
            bad.append("rgen")
        if bad:
            raise ConfigError("NosaExecutor: on backend %s the generation mirrors %s of layer %d are host tensors: "
                              "their identity copies would read the call-time value, not the stream-ordered one; use "
                              "Config.store_gen 'consumer' (I11 HOOK-L6: device mirrors)"
                              % (type(self.be).__name__, bad, l))

    @staticmethod
    def _stack_row(items: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row.kv[h] items (k, v, c) in their order -> K (n, D), V (n, D), C (n,)."""
        return (torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items]),
                torch.stack([it[2].reshape(()) for it in items]))

    def _plan(self, row, h: int) -> List[Tuple[int, str, int, int, Any]]:
        """The declared logical order of one (row, head): Row.addr blocks (first position blk * bs) merged with the
        Row.ring entries, by increasing position."""
        bs = self.bs
        items = [(blk * bs, "arena", p, g, blk) for blk, p, g in (row.addr[h] if row.addr else ())]
        items += [(pos, "ring", e, g, kind) for pos, e, g, kind in (row.ring[h] if row.ring else ())]
        items.sort(key=lambda x: x[0])
        return items

    def _block_offsets(self, row, h: int) -> Dict[int, int]:
        """The index in Row.kv[h] of the first item of each Row.addr block (the plan's order is Row.kv's order)."""
        out, off = {}, 0
        for p0, where, a, g, tag in self._plan(row, h):
            if where == "arena":
                out[tag] = off
                off += self.bs
            else:
                off += 1
        return out

    # -- the validator reads (RingArenaToy._validator_reads: the same registrations and order)
    def _validator_reads(self, l: int, rows) -> Tuple[int, int]:
        bs, n, nr = self.bs, 0, 0
        for row in rows:
            for h, ads in enumerate(row.addr or ()):
                offs = self._block_offsets(row, h) if ads else {}
                for blk, p, g in ads:
                    items = row.kv[h][offs[blk]:offs[blk] + bs]             # MINOR-2: the eager expectation
                    exps = self._stack_row(items)
                    for j, arena in enumerate(self._arenas(l)):
                        src = arena[p * bs:(p + 1) * bs]
                        exp = exps[j].to(dtype=src.dtype, device=src.device)
                        ws = torch.empty_like(src)
                        self.be.memcpy(ws, src, "compute")                  # the deferred read (a kernel read)
                        pg = None
                        if self.check_identity:
                            pg = torch.empty(1, dtype=torch.int64, device=self.pgen[l].device)
                            self.be.memcpy(pg, self.pgen[l][p:p + 1], "compute")   # the map entry at execution
                        self.register(l, row, h, blk, p, g, ws, exp, pg)
                    n += 1
            for h, items in enumerate(row.ring or ()):
                for pos, e, g, kind in items:
                    for which, tab in (("k", self.rk[l]), ("v", self.rv[l])):
                        ws = torch.empty_like(tab[e])
                        self.be.memcpy(ws, tab[e], "compute")               # the deferred ring read
                        pg = torch.empty(1, dtype=torch.int64, device=self.rgen[l].device)
                        self.be.memcpy(pg, self.rgen[l][e:e + 1], "compute")      # its generation at execution
                        self.register_ring(l, row, h, pos, e, g, kind, which, ws, pg)
                    if self.rc is not None:                                 # ABI v4: the entry's bias (1 element)
                        ws = torch.empty(1, dtype=self.rc[l].dtype, device=self.rc[l].device)
                        self.be.memcpy(ws, self.rc[l][e:e + 1], "compute")
                        pg = torch.empty(1, dtype=torch.int64, device=self.rgen[l].device)
                        self.be.memcpy(pg, self.rgen[l][e:e + 1], "compute")
                        self.register_ring(l, row, h, pos, e, g, kind, "c", ws, pg)
                    nr += 1
        return n, nr

    def joint_post(self, l, rows):
        """The non-consuming protocol: the validator reads, outputs from Row.kv (the reference ops)."""
        self._check_mirrors(l, rows)
        self._enter_compute()                                           # M1 (revision 43)
        outs = []
        for row in rows:
            heads = [self.model.head_out(l, row.q[h], *self._stack_row(row.kv[h])) for h in range(self.H)]
            outs.append(self.model.mix(l, row.hidden, heads))
        with self.be.stream("compute"):
            n, nr = self._validator_reads(l, rows)
            read_ev = self.be.event_rec("compute") if (n or nr) else None
        self.reads += n
        self.ring_reads += nr
        return outs, read_ev

    # -- C1.2 consumption
    def _gather(self, l: int, i: int, row, h: int, call: Any, mode: str) -> Dict[str, Any]:
        bs, D, be = self.bs, self.D, self.be
        t0 = time.perf_counter_ns()
        items = self._plan(row, h)
        n = sum(bs if it[1] == "arena" else 1 for it in items)
        K = torch.empty((n, D), dtype=self.ak[l].dtype, device=self.ak[l].device)
        V = torch.empty((n, D), dtype=self.av[l].dtype, device=self.av[l].device)
        C = torch.empty((n,), dtype=self.ac[l].dtype, device=self.ac[l].device)
        ID = torch.empty((len(items),), dtype=torch.int64, device=self.pgen[l].device)
        idents, positions = [], []
        for t, (p0, where, a, g, tag) in enumerate(items):
            positions.extend(range(p0, p0 + bs) if where == "arena" else (p0,))
            idents.append(dict(where=where, layer=l, addr=a, expected=int(g), slot=t, head=h))
        t1 = time.perf_counter_ns()
        ev_a = be.event_rec("compute") if self.gather_events else None
        if mode == "batched":
            self._gather_batched(l, items, K, V, C, ID)
        else:
            k = 0
            for t, (p0, where, a, g, tag) in enumerate(items):
                if where == "arena":
                    r0 = a * bs
                    be.memcpy(K[k:k + bs], self.ak[l][r0:r0 + bs], "compute")
                    be.memcpy(V[k:k + bs], self.av[l][r0:r0 + bs], "compute")
                    be.memcpy(C[k:k + bs], self.ac[l][r0:r0 + bs], "compute")
                    be.memcpy(ID[t:t + 1], self.pgen[l][a:a + 1], "compute")   # the generation AFTER its data
                    k += bs
                else:
                    be.memcpy(K[k], self.rk[l][a], "compute")
                    be.memcpy(V[k], self.rv[l][a], "compute")
                    be.memcpy(C[k:k + 1], self.rc[l][a:a + 1], "compute")
                    be.memcpy(ID[t:t + 1], self.rgen[l][a:a + 1], "compute")   # the generation AFTER its data
                    k += 1
        ev_b = be.event_rec("compute") if self.gather_events else None
        t2 = time.perf_counter_ns()
        nbytes = sum(x.untyped_storage().nbytes() for x in (K, V, C, ID))
        log = dict(call=call, layer=l, uid=row.uid, pos=row.pos, kind=row.kind, head=h, n=n, bytes=nbytes,
                   consumed=False, events=self.gather_events, ready_done=None, read_done=None, gather_ns=None)
        self._log("gathers", log)
        return dict(i=i, h=h, K=K, V=V, C=C, ID=ID, idents=idents, positions=positions, uid=row.uid, pos=row.pos,
                    kind=row.kind, bytes=nbytes, log=log, ev_a=ev_a, ev_b=ev_b, pack_ns=t1 - t0, gather_ns=t2 - t1)

    def _gather_batched(self, l: int, items, K, V, C, ID) -> None:
        """MINOR-3: the gather of one (row, head) as index kernels on the caller's "compute" stream context: the arena
        rows and the ring rows each by ONE index_select from precomputed row indices, placed by ONE index_copy_ into
        the declared order; then the identity rows (pgen / rgen) the same way, AFTER the data. Eager on the host
        backend (module docstring)."""
        bs = self.bs
        dev = K.device
        dst_a, src_a, dst_r, src_r, slot_a, gen_a, slot_r, gen_r = [], [], [], [], [], [], [], []
        k = 0
        for t, (p0, where, a, g, tag) in enumerate(items):
            if where == "arena":
                dst_a.extend(range(k, k + bs))
                src_a.extend(range(a * bs, (a + 1) * bs))
                slot_a.append(t)
                gen_a.append(a)
                k += bs
            else:
                dst_r.append(k)
                src_r.append(a)
                slot_r.append(t)
                gen_r.append(a)
                k += 1
        if src_a:
            si = torch.tensor(src_a, dtype=torch.int64, device=dev)
            di = torch.tensor(dst_a, dtype=torch.int64, device=dev)
            K.index_copy_(0, di, self.ak[l].index_select(0, si))
            V.index_copy_(0, di, self.av[l].index_select(0, si))
            C.index_copy_(0, di, self.ac[l].index_select(0, si))
        if src_r:
            si = torch.tensor(src_r, dtype=torch.int64, device=self.rk[l].device)
            di = torch.tensor(dst_r, dtype=torch.int64, device=dev)
            K.index_copy_(0, di, self.rk[l].index_select(0, si).to(dev))
            V.index_copy_(0, di, self.rv[l].index_select(0, si).to(dev))
            C.index_copy_(0, di, self.rc[l].index_select(0, si.to(self.rc[l].device)).to(dev))
        if gen_a:                                                       # the identity rows AFTER the data
            ID.index_copy_(0, torch.tensor(slot_a, dtype=torch.int64, device=ID.device),
                           self.pgen[l].index_select(0, torch.tensor(gen_a, dtype=torch.int64,
                                                                     device=self.pgen[l].device)).to(ID.device))
        if gen_r:
            ID.index_copy_(0, torch.tensor(slot_r, dtype=torch.int64, device=ID.device),
                           self.rgen[l].index_select(0, torch.tensor(gen_r, dtype=torch.int64,
                                                                     device=self.rgen[l].device)).to(ID.device))

    def joint_post_consume(self, l, rows):
        self._check_mirrors(l, rows)
        self._enter_compute()                                           # M1 (revision 43)
        call = self._call_id()
        mode = self._gather_mode()
        gathers = []
        with self.be.stream("compute"):
            n, nr = self._validator_reads(l, rows)                      # exactly joint_post's validator reads
            for i, row in enumerate(rows):
                for h in range(self.H):
                    gathers.append(self._gather(l, i, row, h, call, mode))
            read_ev = self.be.event_rec("compute") if (n or nr) else None   # after the call's LAST read
        self.reads += n
        self.ring_reads += nr
        self.gathers += len(gathers)
        self.consume_calls += 1
        self.consumed_rows += len(rows)
        nbytes = sum(gt["bytes"] for gt in gathers)
        self.debug["gathered_bytes_peak"] = max(self.debug["gathered_bytes_peak"], nbytes)
        rec = dict(call=call, layer=l, rows=len(rows), gathers=len(gathers), bytes=nbytes, mode=mode,
                   pack_ns=sum(gt["pack_ns"] for gt in gathers), gather_ns=sum(gt["gather_ns"] for gt in gathers),
                   attn_ns=None, mix_ns=None, ref_attn_ns=None, ref_mix_ns=None)
        self._log("calls", rec)
        return dict(l=l, call=call, rows=list(rows), gathers=gathers, rec=rec), read_ev

    def gather_tensors(self, handle):
        return [((gt["i"], gt["h"], part), gt[name]) for gt in handle["gathers"]
                for part, name in (("k", "K"), ("v", "V"), ("c", "C"), ("id", "ID"))]

    def consumed_identities(self, handle):
        out = []
        for gt in handle["gathers"]:
            obs = gt["ID"].tolist()                                     # ONE host transfer per gather
            for rec in gt["idents"]:
                out.append(dict(rec, observed=int(obs[rec["slot"]]), uid=gt["uid"], pos=gt["pos"], kind=gt["kind"]))
        for rec in out:
            self._log("identities", dict(rec, call=handle.get("call")))
        return out

    @staticmethod
    def _elapsed_ns(a: Any, b: Any) -> Optional[int]:
        """The time between two completed events: CUDA elapsed_time (ms), or the CpuBackend completion times."""
        if a is None or b is None or not (ev_done(a) and ev_done(b)):
            return None
        et = getattr(a, "elapsed_time", None)
        if callable(et):
            try:
                return int(et(b) * 1e6)
            except Exception:
                return None
        ta, tb = getattr(a, "t_ns", None), getattr(b, "t_ns", None)
        return None if ta is None or tb is None else int(tb - ta)

    def resolve(self, handle):
        l, rows = handle["l"], handle["rows"]
        per = {(gt["i"], gt["h"]): gt for gt in handle["gathers"]}
        cur = torch.cuda.current_stream() if any(gt["K"].is_cuda for gt in handle["gathers"]) else None
        outs, heads = [], []
        attn = mix = 0
        for i, row in enumerate(rows):                                  # ONLY the gathered rows (never Row.kv)
            hh = []
            for h in range(self.H):
                gt = per[(i, h)]
                if cur is not None:                                     # NIT-5: the backings are used here
                    for name in ("K", "V", "C", "ID"):
                        if gt[name].is_cuda:
                            gt[name].record_stream(cur)
                t0 = time.perf_counter_ns()
                hh.append(self.model.head_out(l, row.q[h], gt["K"], gt["V"], gt["C"]))
                attn += time.perf_counter_ns() - t0
                log = gt["log"]
                log["consumed"] = True
                if gt["ev_a"] is not None:                              # NIT-4: booleans and times, no event kept
                    log["ready_done"], log["read_done"] = bool(ev_done(gt["ev_a"])), bool(ev_done(gt["ev_b"]))
                    log["gather_ns"] = self._elapsed_ns(gt["ev_a"], gt["ev_b"])
            t0 = time.perf_counter_ns()
            y = self.model.mix(l, row.hidden, hh)
            mix += time.perf_counter_ns() - t0
            heads.append(hh)
            outs.append(y)
        handle["rec"]["attn_ns"], handle["rec"]["mix_ns"] = attn, mix
        handle["resolved"] = (outs, heads)
        self._compare(handle)
        return outs, heads

    def reference(self, handle):
        if "referenced" in handle:
            outs, heads = handle["referenced"]
            return outs, heads
        l, rows = handle["l"], handle["rows"]
        outs, heads = [], []
        attn = mix = 0
        for row in rows:                                                # the eager tensors (Row.kv), same ops
            hh = []
            for h in range(self.H):
                K, Vv, C = self._stack_row(row.kv[h])
                t0 = time.perf_counter_ns()
                hh.append(self.model.head_out(l, row.q[h], K, Vv, C))
                attn += time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            y = self.model.mix(l, row.hidden, hh)
            mix += time.perf_counter_ns() - t0
            heads.append(hh)
            outs.append(y)
        handle["rec"]["ref_attn_ns"], handle["rec"]["ref_mix_ns"] = attn, mix
        handle["referenced"] = (outs, heads)
        self._compare(handle)
        return outs, heads

    # -- instrumentation: the first divergence of a call (never decides)
    def _compare(self, handle) -> None:
        if not self._divergence_on() or "resolved" not in handle or "referenced" not in handle \
                or handle.get("compared"):
            return
        handle["compared"] = True
        l, rows = handle["l"], handle["rows"]
        (outs, heads), (routs, rheads) = handle["resolved"], handle["referenced"]
        per = {(gt["i"], gt["h"]): gt for gt in handle["gathers"]}
        found = None
        for i, row in enumerate(rows):
            for h in range(self.H):
                gt = per[(i, h)]
                K, Vv, C = self._stack_row(row.kv[h])
                for op, a, b in (("gather_k", gt["K"], K), ("gather_v", gt["V"], Vv), ("gather_c", gt["C"], C)):
                    if a.shape != b.shape:
                        found = (i, h, op, None)
                        break
                    if not torch.equal(a, b):
                        bad = (a != b).reshape(a.shape[0], -1).any(dim=1).nonzero()
                        j = int(bad[0]) if bad.numel() else 0
                        found = (i, h, op, gt["positions"][j] if j < len(gt["positions"]) else None)
                        break
                if found is None and not torch.equal(heads[i][h], rheads[i][h]):
                    found = (i, h, "head_out", None)
                if found is not None:
                    break
            if found is None and not torch.equal(outs[i], routs[i]):
                found = (i, None, "mix", None)
            if found is not None:
                break
        if found is None:
            return
        i, h, op, idx = found
        row = rows[i]
        rec = dict(call=handle.get("call"), layer=l, uid=row.uid, pos=row.pos, kind=row.kind, head=h, op=op,
                   index=idx)
        self._log("divergences", rec)
        if self.debug["first_divergence"] is None:
            self.debug["first_divergence"] = rec

    # -- a read-only detector for a caller (never called by this executor itself)
    def check_gathered(self, handle) -> int:
        """READ-ONLY: after the caller's wait on the call's read_ev (and its identity check), every gathered K / V /
        bias row equals Row.kv (torch.equal), else CacheMismatch naming the first difference (module docstring)."""
        n = 0
        for gt in handle["gathers"]:
            row = handle["rows"][gt["i"]]
            K, Vv, C = self._stack_row(row.kv[gt["h"]])
            for name, a, b in (("K", gt["K"], K), ("V", gt["V"], Vv), ("bias", gt["C"], C)):
                if a.shape != b.shape or not torch.equal(a, b):
                    if a.shape != b.shape:
                        pos = None
                    else:
                        bad = (a != b).reshape(a.shape[0], -1).any(dim=1).nonzero()
                        pos = gt["positions"][int(bad[0])]
                    raise CacheMismatch("C1.2 gathered cache mismatch: call %s layer %d row (%s, %d, %s) head %d "
                                        "position %s tensor %s" % (handle.get("call"), handle["l"], row.uid, row.pos,
                                                                   row.kind, gt["h"], pos, name))
            n += 1
        return n

    # -- the store hooks (tensor_store.py DISPATCHER ADAPTER; I9 with the producer ordering of the module docstring)
    def _device_copy(self, src: torch.Tensor, dtype: torch.dtype, stream: Optional[str]) -> torch.Tensor:
        out = torch.empty(src.shape, dtype=dtype, device=self.store_device)
        self.be.memcpy(out, src, stream)
        return out

    def _produce(self, build) -> Tuple[List[torch.Tensor], Any]:
        """One producer call. build(stream) returns [(source tensor, dtype)]; it runs on the producer stream context
        (device mode). Host mode: fresh eager tensors, ready None (or an event recorded on the deferred ready_stream
        after them, when one is configured). Device mode: ev0 on the current stream; with a
        ready_stream, a wait on ev0 there, the backend copies into fresh outputs there, ready recorded there; without
        one, the copies and ready on the current stream (module docstring, producer ordering)."""
        be, rs = self.be, self.ready_stream
        if self.host_mode():
            outs = [src.detach().to(device=self.store_device, dtype=dt).clone() for src, dt in build(None)]
            return outs, (be.event_rec(rs) if (rs is not None and be is not None) else None)
        ev0 = be.event_rec(None)                                       # after the model's writes (current stream)
        if rs is None:
            outs = [self._device_copy(src, dt, None) for src, dt in build(None)]
            return outs, be.event_rec(None)
        be.stream_wait(rs, ev0)
        with be.stream(rs):
            outs = [self._device_copy(src, dt, rs) for src, dt in build(rs)]
            ready = be.event_rec(rs)
        return outs, ready

    def prefill_tensors(self, kv):
        """Per layer (n, H, D) K, V and (n, H) bias: FRESH tensors of kv[pos][l][h] = (k, v, c) in kv_dtype on
        store_device, from ONE stack per layer and kind; returns (k, v, c, ready) (I9; module docstring)."""
        n, H, L, D = len(kv), self.H, self.L, self.D

        def build(stream):
            srcs = []
            for part, dt, shape in ((0, self.kv_dtype, (n, H, D)), (1, self.kv_dtype, (n, H, D)),
                                    (2, self.bias_dtype, (n, H))):
                for l in range(L):
                    flat = [kv[p][l][h][part].reshape(-1) if part < 2 else kv[p][l][h][part].reshape(())
                            for p in range(n) for h in range(H)]
                    srcs.append((torch.stack(flat).reshape(shape), dt))
            return srcs
        outs, ready = self._produce(build)
        self.prefills += 1
        return outs[:L], outs[L:2 * L], outs[2 * L:], ready

    def payload(self, kvv, at=None):
        """The item's (k, v, c) as FRESH tensors (D,), (D,), (); a producer (at given) gets (k, v, c, ready) (I9), an
        expectation (at None) gets (k, v, c) (fresh eager clones: the dispatcher's expectations, no ordering)."""
        k, v, c = kvv[0], kvv[1], kvv[2]
        self.payloads += 1
        if at is None:
            return (self._fresh(k, self.kv_dtype), self._fresh(v, self.kv_dtype),
                    self._fresh(c, self.bias_dtype).reshape(()))
        outs, ready = self._produce(lambda stream: [(k, self.kv_dtype), (v, self.kv_dtype),
                                                    (c.reshape(()), self.bias_dtype)])
        return outs[0], outs[1], outs[2], ready

    def _fresh(self, t: Any, dtype: torch.dtype) -> torch.Tensor:
        return t.detach().to(device=self.store_device, dtype=dtype).clone()

    def values(self, k, v, c):
        """The eager items of rows read from the stores: [(k[i], v[i], c[i])] views of the caller's tensors (the
        dispatcher's _row_kv hands clones of the store bytes)."""
        return list(zip(k.unbind(0), v.unbind(0), c.unbind(0)))
