"""Milestone 10, Track B (INTERFACE_K3X section 2): NosaExecutor, the tensor-native C1.2 consumer of the store's mode
for ANY model of section 1 (the real NOSA-8B model, the tiny NOSA-shaped model, or a test stub). SOURCE ONLY until the
lead's runs; it never imports tests/ and never imports dispatcher.py.

Template. The protocol layers of the research toy executors (tests/kstep_toy.py ArenaToy -> RingArenaToy ->
ConsumingRingArenaToy) and Track B's store hooks (tests/kstep_tensor_fixtures.py TensorPayloadConsumer), moved here
WITHOUT the integer codes: the eager items ARE the model's tensors. The protocol is executor.py's (M2 ARENA PROTOCOL, M2
RING PROTOCOL v3, C1.1 ABI v4, C1.2 SERIALIZED PHYSICAL CONSUMPTION, I10) and tensor_store.py's DISPATCHER ADAPTER
(tensor_payload, fresh_outputs, kv_dtype / bias_dtype, store_device, prefill_tensors, payload, values; I9
producer_ready).

  NosaExecutor(model, check_identity=True, ready_stream=None, gather_events=False, check_divergence=True,
               debug_cap=100000)
      The constructor of the lead's builders (kstep_toy.build passes the model as `toy`; store_kmod calls
      executor_cls(ex.toy, check_identity=ex.check_identity)). Unknown attributes are forwarded to the model (as
      ArenaToy.__getattr__ does); "model" and dunder names are never forwarded.
  attributes      abi = 4, reads_arena, reads_ring, consumes_physical, tensor_payload, fresh_outputs = True (every
                  prefill / payload tensor is a fresh storage that it never touches again), producer_ready = True (I9:
                  prefill_tensors and a producer payload return (k, v, c, ready)), reset_contract = "I10"; from the
                  model: L, H, D, V, bs, stride, max_sel, chain_kernel; kv_dtype = bias_dtype = model.dtype,
                  store_device = model.device.
  forwarded       joint_pre, select, compress, embed, dist, prefill: the model's own methods, unchanged.
  attach_arena / attach_bias / attach_ring
                  the protocol's store attachments (the generation mirrors are the store's consumer mirrors under
                  HOOK-L6 when configured; this executor reads them only through deferred copies, never on the host).
  joint_post(l, rows) -> (outs, read_ev)
                  the validator reads (below), outputs from Row.kv (the non-consuming arena / ring protocol).
  joint_post_consume(l, rows) -> (handle, read_ev)
                  exactly the validator reads of joint_post (same registrations, same order), then per (row, head)
                  ONE gather on "compute" of K (n, D), V (n, D), bias (n,) in the declared logical order (Row.addr
                  blocks as bs rows each, merged with Row.ring entries, increasing position), with ONE identity copy
                  per block (pgen) or entry (rgen) AFTER its data; read_ev on "compute" after the call's LAST read. The
                  handle is a dict that the dispatcher owns; this executor keeps no reference to it.
  gather_tensors / consumed_identities
                  as the protocol (every gather backing; one identity record per block / entry, expected = the
                  Row.addr / Row.ring generation captured at call time, observed = the copied value).
  resolve(handle) -> (outs, heads)
                  per row, in call order: heads[h] = model.head_out(l, q_h, K, V, C) on the GATHERED tensors only, then
                  out = model.mix(l, x, heads). It never reads Row.kv for a result.
  reference(handle) -> (outs, heads)
                  the same ops in the same order from Row.kv (the eager tensors, stacked in Row.kv order).
  check_gathered(handle)
                  READ-ONLY detector for a caller (the dispatcher does not call it today): after the caller's wait on
                  read_ev and its identity check, the first gathered K / V / bias row that differs (torch.equal) from
                  Row.kv raises CacheMismatch "C1.2 gathered cache mismatch: call <c> layer <l> row (<uid>, <pos>,
                  <kind>) head <h> position <p> tensor <K | V | bias>". Returns the gathers checked.
  prefill_tensors(kv) -> (k, v, c, ready)   per layer (n, H, D), (n, H, D) and (n, H): FRESH stacks of the model's
                  tensors kv[pos][l][h] in kv_dtype on store_device;
  payload(kvv, at=None)                     the item's (k, v, c) as FRESH tensors (D,), (D,), (); a producer (at
                  given) gets (k, v, c, ready), an expectation (at None) (k, v, c);
  values(k, v, c)                           the eager items [(k_i, v_i, c_i)] of rows read from the stores: views of the
                  caller's tensors (the dispatcher hands clones of the store bytes), never Python lists.
  ready (I9)      None when no ready_stream is configured: the outputs are complete tensors when they are returned (the
                  CpuBackend, or a caller that synchronizes). With ready_stream (a stream name of the backend, for
                  example on the CudaBackend), the stacks / copies run inside be.stream(ready_stream) and ONE event is
                  recorded there after them (the ready event of the call).

Instrumentation (kept in self.debug; it NEVER decides; nothing reads it for a decision):
  first_divergence / divergences
                  per call, the FIRST place where resolve and reference differ: dict(call, layer, uid, pos, kind,
                  head, op, index), op in "gather_k" | "gather_v" | "gather_c" (a gathered row differs from Row.kv;
                  index = the position), "head_out" (a head), "mix" (the row output; head None). The comparison runs
                  when both results of a call exist (whichever of resolve / reference comes second), with torch.equal
                  (check_divergence; on the GPU it synchronizes, so it is optional).
  identities      every consumed identity record (expected / observed generations), call, at consumed_identities.
  gathers         per gather: call, layer, uid, pos, kind, head, n positions, bytes, ready_ev (an event recorded on
                  "compute" right BEFORE the gather's first copy: every earlier producer copy on "compute" precedes
                  it) and read_ev (right AFTER its last copy) when gather_events is True (else None and the call's
                  read_ev), and consumed (True once resolve used it; the dispatcher releases the backing after the
                  call's resolution: its consume ledger holds it until then). These events are instrumentation history
                  (never waited on, never listed by reset_holders).
  calls           per call: call, layer, rows, gathers, bytes, pack_ns (building the gather plan and the backings),
                  gather_ns (enqueueing the copies), attn_ns / mix_ns (resolve's head_out / mix), ref_attn_ns /
                  ref_mix_ns (reference's): host wall time from time.perf_counter_ns (on the GPU without a synchronize
                  this is launch time, not kernel time).
  gathered_bytes_peak
                  the largest bytes of one call's gather backings (K, V, bias, identity copies).
  identities and gathers keep at most debug_cap records each (dropped counts the rest).

I10 (reset contract; declared in THIS class body). Audit: it keeps the model (weights and the model's own state; the
section-1 model API holds no backend event), check_identity, the backend `be`, the attached stores and generation
mirrors (ak, av, ac, pgen, rk, rv, rc, rgen), the register callbacks, disp (the harness's dispatcher handle) and the
counters and the debug record (instrumentation history; its gather events are never waited on or read for a
decision). Each call's handle and read_ev go to the dispatcher and are not kept; ready events are returned, not
kept. So it keeps NO operational event between calls: reset_holders() == [], detach_completed() == 0.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .errors import CacheMismatch

DIVERGE_OPS = ("gather_k", "gather_v", "gather_c", "head_out", "mix")   # the debug record's op names, in check order


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
                 gather_events: bool = False, check_divergence: bool = True, debug_cap: int = 100000):
        self.model = model
        self.check_identity = bool(check_identity)
        self.L, self.H, self.D, self.V = model.L, model.H, model.D, model.V
        self.bs, self.stride, self.max_sel = model.bs, model.stride, model.max_sel
        self.chain_kernel = model.chain_kernel
        self.kv_dtype = self.bias_dtype = model.dtype
        self.store_device = model.device
        self.ready_stream = ready_stream
        self.gather_events = bool(gather_events)
        self.check_divergence = bool(check_divergence)
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

    # -- helpers
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

    def _ready(self) -> Any:
        """I9: the ready event of a producer call, recorded on ready_stream after the call's writes (None without a
        ready_stream: the outputs are complete when they are returned)."""
        if self.ready_stream is None or self.be is None:
            return None
        return self.be.event_rec(self.ready_stream)

    def _producer_stream(self):
        """The context of a producer call's writes: be.stream(ready_stream) with a ready_stream, else nothing."""
        if self.ready_stream is None or self.be is None:
            return contextlib.nullcontext()
        return self.be.stream(self.ready_stream)

    def _fresh(self, t: Any, dtype: torch.dtype) -> torch.Tensor:
        return t.detach().to(device=self.store_device, dtype=dtype).clone()

    @staticmethod
    def _stack_row(items: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row.kv[h] items (k, v, c) in their order -> K (n, D), V (n, D), C (n,)."""
        return (torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items]),
                torch.stack([it[2].reshape(()) for it in items]))

    # -- the validator reads (RingArenaToy._validator_reads, unchanged in order and registrations)
    def _validator_reads(self, l: int, rows) -> Tuple[int, int]:
        bs, n, nr = self.bs, 0, 0
        for row in rows:
            for h, ads in enumerate(row.addr or ()):
                for blk, p, g in ads:
                    for arena in self._arenas(l):
                        src = arena[p * bs:(p + 1) * bs]
                        exp = src.clone()                                   # the eager clone at call time
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
    def _plan(self, row, h: int) -> List[Tuple[int, str, int, int, Any]]:
        """The declared logical order of one (row, head): Row.addr blocks (first position blk * bs) merged with the
        Row.ring entries, by increasing position."""
        bs = self.bs
        items = [(blk * bs, "arena", p, g, blk) for blk, p, g in (row.addr[h] if row.addr else ())]
        items += [(pos, "ring", e, g, kind) for pos, e, g, kind in (row.ring[h] if row.ring else ())]
        items.sort(key=lambda x: x[0])
        return items

    def _gather(self, l: int, i: int, row, h: int, call: Any) -> Dict[str, Any]:
        bs, D, be = self.bs, self.D, self.be
        t0 = time.perf_counter_ns()
        items = self._plan(row, h)
        n = sum(bs if it[1] == "arena" else 1 for it in items)
        K = torch.empty((n, D), dtype=self.ak[l].dtype, device=self.ak[l].device)
        V = torch.empty((n, D), dtype=self.av[l].dtype, device=self.av[l].device)
        C = torch.empty((n,), dtype=self.ac[l].dtype, device=self.ac[l].device)
        ID = torch.empty((len(items),), dtype=torch.int64, device=self.pgen[l].device)
        t1 = time.perf_counter_ns()
        ready_ev = be.event_rec("compute") if self.gather_events else None
        idents, positions = [], []
        k = 0
        for t, (p0, where, a, g, tag) in enumerate(items):
            if where == "arena":
                r0 = a * bs
                be.memcpy(K[k:k + bs], self.ak[l][r0:r0 + bs], "compute")
                be.memcpy(V[k:k + bs], self.av[l][r0:r0 + bs], "compute")
                be.memcpy(C[k:k + bs], self.ac[l][r0:r0 + bs], "compute")
                be.memcpy(ID[t:t + 1], self.pgen[l][a:a + 1], "compute")   # the generation AFTER its data
                positions.extend(range(p0, p0 + bs))
                k += bs
            else:
                be.memcpy(K[k], self.rk[l][a], "compute")
                be.memcpy(V[k], self.rv[l][a], "compute")
                be.memcpy(C[k:k + 1], self.rc[l][a:a + 1], "compute")
                be.memcpy(ID[t:t + 1], self.rgen[l][a:a + 1], "compute")   # the generation AFTER its data
                positions.append(p0)
                k += 1
            idents.append(dict(where=where, layer=l, addr=a, expected=int(g), slot=t, head=h))
        read_ev = be.event_rec("compute") if self.gather_events else None
        t2 = time.perf_counter_ns()
        nbytes = sum(x.untyped_storage().nbytes() for x in (K, V, C, ID))
        log = dict(call=call, layer=l, uid=row.uid, pos=row.pos, kind=row.kind, head=h, n=n, bytes=nbytes,
                   ready_ev=ready_ev, read_ev=read_ev, consumed=False)
        self._log("gathers", log)
        return dict(i=i, h=h, K=K, V=V, C=C, ID=ID, idents=idents, positions=positions, uid=row.uid, pos=row.pos,
                    kind=row.kind, bytes=nbytes, log=log, pack_ns=t1 - t0, gather_ns=t2 - t1)

    def joint_post_consume(self, l, rows):
        call = self._call_id()
        gathers = []
        with self.be.stream("compute"):
            n, nr = self._validator_reads(l, rows)                      # exactly joint_post's validator reads
            for i, row in enumerate(rows):
                for h in range(self.H):
                    gathers.append(self._gather(l, i, row, h, call))
            read_ev = self.be.event_rec("compute") if (n or nr) else None   # after the call's LAST read
        if not self.gather_events:
            for gt in gathers:
                gt["log"]["read_ev"] = read_ev
        self.reads += n
        self.ring_reads += nr
        self.gathers += len(gathers)
        self.consume_calls += 1
        self.consumed_rows += len(rows)
        nbytes = sum(gt["bytes"] for gt in gathers)
        self.debug["gathered_bytes_peak"] = max(self.debug["gathered_bytes_peak"], nbytes)
        rec = dict(call=call, layer=l, rows=len(rows), gathers=len(gathers), bytes=nbytes,
                   pack_ns=sum(gt["pack_ns"] for gt in gathers), gather_ns=sum(gt["gather_ns"] for gt in gathers),
                   attn_ns=None, mix_ns=None, ref_attn_ns=None, ref_mix_ns=None)
        self.debug["calls"].append(rec)
        return dict(l=l, call=call, rows=list(rows), gathers=gathers, rec=rec), read_ev

    def gather_tensors(self, handle):
        return [((gt["i"], gt["h"], part), gt[name]) for gt in handle["gathers"]
                for part, name in (("k", "K"), ("v", "V"), ("c", "C"), ("id", "ID"))]

    def consumed_identities(self, handle):
        out = [dict(rec, observed=int(gt["ID"][rec["slot"]]), uid=gt["uid"], pos=gt["pos"], kind=gt["kind"])
               for gt in handle["gathers"] for rec in gt["idents"]]
        for rec in out:
            self._log("identities", dict(rec, call=handle.get("call")))
        return out

    def resolve(self, handle):
        l, rows = handle["l"], handle["rows"]
        per = {(gt["i"], gt["h"]): gt for gt in handle["gathers"]}
        outs, heads = [], []
        attn = mix = 0
        for i, row in enumerate(rows):                                  # ONLY the gathered rows (never Row.kv)
            hh = []
            for h in range(self.H):
                gt = per[(i, h)]
                t0 = time.perf_counter_ns()
                hh.append(self.model.head_out(l, row.q[h], gt["K"], gt["V"], gt["C"]))
                attn += time.perf_counter_ns() - t0
                gt["log"]["consumed"] = True
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
        if not self.check_divergence or "resolved" not in handle or "referenced" not in handle \
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
        self.debug["divergences"].append(rec)
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

    # -- the store hooks (tensor_store.py DISPATCHER ADAPTER; I9)
    def prefill_tensors(self, kv):
        """Per layer (n, H, D) K, V and (n, H) bias: FRESH stacks of kv[pos][l][h] = (k, v, c) in kv_dtype on
        store_device; returns (k, v, c, ready) (I9; ready None without a ready_stream)."""
        n, H, L = len(kv), self.H, self.L
        ks, vs, cs = [], [], []
        with self._producer_stream():
            for l in range(L):
                ks.append(torch.stack([torch.stack([kv[p][l][h][0] for h in range(H)]) for p in range(n)])
                          .to(device=self.store_device, dtype=self.kv_dtype).clone())
                vs.append(torch.stack([torch.stack([kv[p][l][h][1] for h in range(H)]) for p in range(n)])
                          .to(device=self.store_device, dtype=self.kv_dtype).clone())
                cs.append(torch.stack([torch.stack([kv[p][l][h][2].reshape(()) for h in range(H)]) for p in range(n)])
                          .to(device=self.store_device, dtype=self.bias_dtype).clone())
            ready = self._ready()
        self.prefills += 1
        return ks, vs, cs, ready

    def payload(self, kvv, at=None):
        """The item's (k, v, c) as FRESH tensors (D,), (D,), (); a producer (at given) gets (k, v, c, ready) (I9), an
        expectation (at None) gets (k, v, c)."""
        k, v, c = kvv[0], kvv[1], kvv[2]
        self.payloads += 1
        if at is None:
            return (self._fresh(k, self.kv_dtype), self._fresh(v, self.kv_dtype),
                    self._fresh(c, self.bias_dtype).reshape(()))
        with self._producer_stream():
            out = (self._fresh(k, self.kv_dtype), self._fresh(v, self.kv_dtype),
                   self._fresh(c, self.bias_dtype).reshape(()))
            ready = self._ready()
        return out + (ready,)

    def values(self, k, v, c):
        """The eager items of rows read from the stores: [(k[i], v[i], c[i])] views of the caller's tensors (the
        dispatcher's _row_kv hands clones of the store bytes)."""
        return list(zip(k.unbind(0), v.unbind(0), c.unbind(0)))
