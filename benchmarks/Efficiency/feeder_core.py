"""FEEDER DIAGNOSTIC core (benchmark-only; authorized 2026-10-04 by the user via Codex as a narrow follow-up to the d9348e5
scheduling attribution, retroinfer-eval docs/evidence/cpupack_confirm_2179735/sched_attribution/; prior-art row
docs/superpowers/specs/2026-09-02-prior-art-table.md '2026-10-04 feeder diagnostic'). Pure torch + threading and
device-agnostic: the SAME code runs on the GPU (cpupack_core.CudaBackend, driver feeder_diag.py) and on CPU tensors with
LockedCpuBackend (cpupack_core's DEFERRED stream model made thread-safe) in retroinfer-eval tests/test_feeder_diag.py.

LABEL: finite-burst transport / placement-ready in SCRATCH only; NOT sustained live drafting or committed throughput.

ARMS (identical plans, descriptors, chunk lists, chunk order, bytes, staging and landing rings, transport cores):
  S0  the current serial optimized feeder (run_s0): ONE coordinator thread per layer waits for the layer's list, builds the
      descriptor live, then cpupack_core.Pipe.layer packs chunk k and submits its H2D and scatter before packing chunk k+1
      (the 2179735 transport() order, cpupack_transport.py Runner.transport). Pipe.layer is called with host_stamps only.
  S1  the same work split after packing (run_split): the coordinator thread is the CPU PRODUCER (list wait, descriptor,
      pack into a GRANTED staging slot, publish to a bounded ready FIFO); a SOLE CUDA SUBMITTER thread pops, submits the H2D
      and the scatter and publishes the H2D completion handle. Barrier: the producer builds layer l+1's descriptor only
      after the submitter ACKNOWLEDGED the submission of layer l's final chunk (an acknowledgement, not a device drain).
  S2  S1 plus at most ONE next-layer descriptor ahead: after the producer published layer l's final chunk, and after layer
      l+1's list D2H of THIS repetition completed, it may build layer l+1's descriptor before the acknowledgement; chunk
      planning and PACKING of layer l+1 still wait for it. Only descriptor construction crosses the S1 barrier. The plan of a
      layer comes from the repetition's own list (no future-query oracle).

OWNERSHIP (explicit slot + generation; SlotTable):
  staging slot s: FREE -grant (gen + 1)-> PACKING -> READY (in the FIFO) -> SUBMITTING (popped) -> PUBLISHED (the H2D end
  event of (s, gen) is published) -> FREE only after the PRODUCER waited on that PUBLISHED handle and it reports complete
  (H2D COMPLETE = reusable). A queue pop or an event publication alone never frees storage (negative controls free_on_pop,
  free_on_publish). Landing slot s (= the staging slot index; round-robin grants make the chunk -> slot map identical in
  every arm): the copy stream waits for the scatter that last read it (no_landing_wait); the scatter waits for its own H2D
  (no_h2d_wait). The submitter owns the event pool, the streams and every submission: OwnedBackend refuses a mutating call
  from any other thread; the producer receives only wait_handle / is_done for PUBLISHED handles. A FIFO item carries
  (slot, gen); the submitter refuses a stale generation (StaleGeneration).
FAILURE: the first exception in either thread is recorded in Cancel, every condition wait wakes and raises Cancelled, both
threads return, and the caller re-raises FeederError after both returned (no thread is left submitting); the caller then
drains the device and resets the tables before the next repetition.
STAMPS: absolute host-monotonic perf_counter_ns instants (never subtracted from a device clock), thread_time_ns per thread
separately, device events per chunk from the submitter's pool (pk at pop, h2d0, h2d1, sc0, sc1, sub after the submission;
S0's Pipe.layer records the same six). Light instrumentation keeps only h2d1 and sc1 (the lifetime handles) and no host
stamps. Logs are preallocated per repetition.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

import cpupack_core as K

LABEL = "finite-burst transport / placement-ready in SCRATCH only; NOT sustained live drafting or committed throughput"
OCCUPANCY_LABEL = "CUDA-event occupancy of the copy stream (NOT a physical bus counter)"
ARMS = ("S0", "S1", "S2")
FREE, PACKING, READY, SUBMITTING, PUBLISHED = "FREE", "PACKING", "READY", "SUBMITTING", "PUBLISHED"
STATES = (FREE, PACKING, READY, SUBMITTING, PUBLISHED)

CHUNK_COLS = ("k", "i", "l", "c", "g0", "g1", "n", "slot", "gen", "wire", "useful",
              "fw0", "fw1", "p0", "p1", "pcpu", "qf0", "pub", "dpub",
              "iw0", "pop", "dpop", "sub0", "cp0", "cp1", "hp", "sc0h", "sc1h", "scpu", "last")
EV_COLS = ("pk", "h2d0", "h2d1", "sc0", "sc1", "sub")
LAYER_COLS = ("i", "l", "n", "nch", "lw0", "lw1", "d0", "d1", "dcpu", "bw0", "bw1", "ahead", "pub_last", "ack")
_P_COLS = ("k", "i", "l", "c", "g0", "g1", "n", "slot", "gen", "wire", "useful", "fw0", "fw1", "p0", "p1", "pcpu", "qf0", "pub", "dpub", "last")
_S_COLS = ("iw0", "pop", "dpop", "sub0", "cp0", "cp1", "hp", "sc0h", "sc1h", "scpu")


def now() -> int:
    return time.perf_counter_ns()


# -------------------------------------------------------------------------------------------------------- faults
@dataclass
class FeederFaults:
    """Negative controls (each must be DETECTED) and the delays that make a missing wait visible. Measurement asserts any()
    is False: no fault and no delay ever runs in a timed repetition."""
    free_on_pop: bool = False                # the submitter frees the staging slot when it pops the item
    free_on_publish: bool = False            # the slot is freed when its H2D handle is published (before H2D completion)
    no_landing_wait: bool = False            # the H2D into a landing slot does not wait for that slot's previous scatter
    no_h2d_wait: bool = False                # the scatter does not wait for its own H2D
    stale_generation: Optional[Tuple[int, int]] = None   # (layer position, chunk): the item carries gen - 1
    s2_pack_cross: bool = False              # S2: packing layer l+1 does not wait for the acknowledgement of layer l
    desc_before_d2h: bool = False            # the descriptor is built without waiting for the layer's list D2H
    skip_scatter_chunk: Optional[int] = None
    poison_stage: bool = False               # the first staged K row of chunk 0 of layer position 0 is overwritten
    producer_raise_at: Optional[Tuple[int, int]] = None
    submitter_raise_at: Optional[Tuple[int, int]] = None
    delay_copy_ms: float = 0.0               # device sleep on the copy stream before chunk 0 of each layer
    delay_scatter_ms: float = 0.0            # device sleep on the scatter stream before chunk 0 of each layer
    delay_ack_s: float = 0.0                 # host sleep in the submitter before acknowledging a layer (barrier controls)

    def any(self) -> bool:
        return any((self.free_on_pop, self.free_on_publish, self.no_landing_wait, self.no_h2d_wait, self.stale_generation is not None,
                    self.s2_pack_cross, self.desc_before_d2h, self.skip_scatter_chunk is not None, self.poison_stage,
                    self.producer_raise_at is not None, self.submitter_raise_at is not None, self.delay_copy_ms > 0,
                    self.delay_scatter_ms > 0, self.delay_ack_s > 0))

    def pipe_faults(self) -> K.Faults:
        """The S0 (Pipe.layer) equivalents: freeing a slot early (pop / publish) = Pipe's missing staging-slot wait."""
        return K.Faults(skip_scatter_chunk=self.skip_scatter_chunk, poison_stage=self.poison_stage, no_landing_wait=self.no_landing_wait,
                        no_h2d_wait=self.no_h2d_wait, no_slot_wait=(self.free_on_pop or self.free_on_publish),
                        delay_copy_ms=self.delay_copy_ms, delay_scatter_ms=self.delay_scatter_ms)


class FeederError(RuntimeError):
    """A repetition failed in the producer or the submitter (the origin's traceback is in .origin_tb)."""

    def __init__(self, where, exc, tb):
        super().__init__("feeder %s failed: %s: %s" % (where, type(exc).__name__, exc))
        self.where, self.origin, self.origin_tb = where, exc, tb


class Cancelled(RuntimeError):
    pass


class StaleGeneration(RuntimeError):
    pass


class OwnershipViolation(RuntimeError):
    pass


class FeederInjected(RuntimeError):
    """The exception controls' injected failure."""


class Cancel:
    """The shared failure record of one repetition. fail() wakes every watched condition."""

    def __init__(self):
        self.lock = threading.Lock()
        self.flag = threading.Event()
        self.cvs: List[threading.Condition] = []
        self.reset()

    def reset(self):
        with self.lock:
            self.err, self.where, self.tb = None, None, None
        self.flag.clear()

    def watch(self, cv):
        self.cvs.append(cv)

    def fail(self, where, exc, tb=None):
        with self.lock:
            if self.err is None:
                self.err, self.where, self.tb = exc, where, tb or traceback.format_exc()
        self.flag.set()
        for cv in self.cvs:
            with cv:
                cv.notify_all()

    def check(self):
        if self.flag.is_set():
            raise Cancelled("cancelled: the %s failed (%r)" % (self.where, self.err))


# ------------------------------------------------------------------------------------------------------- backends
class LockedCpuBackend(K.CpuBackend):
    """cpupack_core.CpuBackend (DEFERRED: an enqueued op runs only when a wait forces it = the worst case a GPU can produce
    for a missing lifetime wait) with every operation under one lock, so two threads can share it. The current stream of
    stream() is thread-local."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lock = threading.RLock()
        self._tl = threading.local()

    @property
    def cur(self):
        return getattr(self._tl, "cur", None)

    @cur.setter
    def cur(self, v):
        if not hasattr(self, "_tl"):
            self._tl = threading.local()
        self._tl.cur = v

    def reset(self):
        pass

    def event_rec(self, name=None):
        with self.lock:
            return super().event_rec(name)

    def marker(self):
        with self.lock:
            return super().marker()

    def host_wait(self, ev):
        with self.lock:
            return super().host_wait(ev)

    def stream_wait(self, name, ev):
        with self.lock:
            return super().stream_wait(name, ev)

    def memcpy(self, dst, src, name=None):
        with self.lock:
            return super().memcpy(dst, src, name)

    def index_copy(self, dst2d, idx, src, name=None):
        with self.lock:
            return super().index_copy(dst2d, idx, src, name)

    def synchronize(self, order=None):
        with self.lock:
            return super().synchronize(order)


def is_done(ev) -> bool:
    """A handle's completion WITHOUT waiting (CUDA: Event.query(); deferred CPU events: .done)."""
    q = getattr(ev, "query", None)
    if callable(q):
        return bool(q())
    return bool(getattr(ev, "done", False))


class OwnedBackend:
    """The submitter's backend: every mutating call (events from the pool, stream waits, copies, scatters, markers, sleeps,
    the pool cursor) is refused from any thread but the bound owner. host_wait (waiting on a PUBLISHED handle) is allowed from
    any thread."""

    MUTATING = ("event_rec", "marker", "memcpy", "index_copy", "stream_wait", "sleep", "reset")

    def __init__(self, be):
        self.be = be
        self.owner = None
        self.owner_name = None

    def bind(self):
        self.owner = threading.get_ident()
        self.owner_name = threading.current_thread().name

    def _own(self, what):
        if self.owner is None or threading.get_ident() != self.owner:
            raise OwnershipViolation("%s from thread %r: the submitter %r owns the events / streams / submissions"
                                     % (what, threading.current_thread().name, self.owner_name))

    def __getattr__(self, name):
        attr = getattr(self.be, name)
        if name in OwnedBackend.MUTATING:
            def guarded(*a, **kw):
                self._own(name)
                return attr(*a, **kw)
            return guarded
        return attr

    @contextmanager
    def stream(self, name):
        self._own("stream")
        with self.be.stream(name):
            yield

    def host_wait(self, ev):
        return self.be.host_wait(ev)


# ------------------------------------------------------------------------------------------------ the slot table
class SlotTable:
    """Staging-slot ownership (module docstring). Transitions are appended to .trans as (slot, gen, from, to, who)."""

    def __init__(self, ring: int, cancel: Cancel):
        self.ring = int(ring)
        self.cv = threading.Condition()
        self.cancel = cancel
        cancel.watch(self.cv)
        self.reset()

    def reset(self):
        with self.cv:
            self.state = [FREE] * self.ring
            self.gen = [0] * self.ring
            self.handle = [None] * self.ring
            self.hgen = [-1] * self.ring
            self.trans: List[Tuple] = []

    def _move(self, s, g, a, b, who):
        if self.state[s] != a:
            raise OwnershipViolation("slot %d gen %d: %s -> %s by %s but the slot is %s" % (s, g, a, b, who, self.state[s]))
        self.state[s] = b
        self.trans.append((s, g, a, b, who))
        self.cv.notify_all()

    def _wait(self, pred):
        while not pred():
            self.cancel.check()
            self.cv.wait(0.05)

    def acquire(self, s: int, wait_handle: Callable, done: Callable = is_done) -> int:
        """PRODUCER: the slot's previous H2D completed (wait on its PUBLISHED handle), then grant it (gen + 1)."""
        with self.cv:
            self._wait(lambda: self.state[s] in (FREE, PUBLISHED))
            st, h, g = self.state[s], self.handle[s], self.gen[s]
        if st == PUBLISHED:
            wait_handle(h)                                               # outside the lock: the device event of (s, g)
            with self.cv:
                if self.state[s] != PUBLISHED or self.gen[s] != g or self.handle[s] is not h:
                    raise OwnershipViolation("slot %d changed while its producer waited on gen %d's handle (%s gen %d)"
                                             % (s, g, self.state[s], self.gen[s]))
                if not done(h):
                    raise OwnershipViolation("slot %d gen %d: the published H2D handle is not complete after the wait" % (s, g))
                self._move(s, g, PUBLISHED, FREE, "producer:h2d_complete")
                self.handle[s] = None
        with self.cv:
            if self.state[s] != FREE:
                raise OwnershipViolation("slot %d: grant of a %s slot" % (s, self.state[s]))
            self.gen[s] += 1
            self._move(s, self.gen[s], FREE, PACKING, "producer:grant")
            return self.gen[s]

    def ready(self, s, g):
        with self.cv:
            if self.gen[s] != g:
                raise StaleGeneration("ready: slot %d gen %d but the table holds gen %d" % (s, g, self.gen[s]))
            self._move(s, g, PACKING, READY, "producer:ready")

    def take(self, s, g, faults: FeederFaults):
        """SUBMITTER: the popped item's (slot, gen) must be the slot's current READY generation."""
        with self.cv:
            if self.gen[s] != g or self.state[s] != READY:
                raise StaleGeneration("take: item (slot %d, gen %d) but the slot is %s gen %d" % (s, g, self.state[s], self.gen[s]))
            self._move(s, g, READY, SUBMITTING, "submitter:take")
            if faults.free_on_pop:                                       # NEGATIVE CONTROL: storage freed by the pop
                self._move(s, g, SUBMITTING, FREE, "FAULT:free_on_pop")

    def publish(self, s, g, h, faults: FeederFaults):
        """SUBMITTER: the H2D end event of (s, g) is the slot's completion handle."""
        with self.cv:
            if faults.free_on_pop:
                return                                                   # the fault already freed (and maybe regranted) it
            if self.gen[s] != g:
                raise StaleGeneration("publish: slot %d gen %d but the table holds gen %d" % (s, g, self.gen[s]))
            self.handle[s], self.hgen[s] = h, g
            self._move(s, g, SUBMITTING, PUBLISHED, "submitter:publish")
            if faults.free_on_publish:                                   # NEGATIVE CONTROL: storage freed by the publication
                self.handle[s] = None
                self._move(s, g, PUBLISHED, FREE, "FAULT:free_on_publish")

    def backlog(self) -> Dict:
        with self.cv:
            return dict(states=list(self.state), gens=list(self.gen), held=sum(1 for x in self.state if x in (PACKING, READY, SUBMITTING)))


class ReadyFifo:
    """The bounded ready FIFO (cap items that hold a staging slot; layer-end and stop markers are not counted)."""

    def __init__(self, cap: int, cancel: Cancel):
        self.cap = int(cap)
        self.cv = threading.Condition()
        self.cancel = cancel
        cancel.watch(self.cv)
        self.reset()

    def reset(self):
        with self.cv:
            self.q = deque()
            self.n = 0
            self.max_depth = 0

    def put(self, item, counted=True):
        """-> (wait start ns, publish ns, depth after)."""
        with self.cv:
            t0 = now()
            while counted and self.n >= self.cap:
                self.cancel.check()
                self.cv.wait(0.05)
            self.q.append((item, counted))
            self.n += int(counted)
            t1 = now()
            d = len(self.q)
            self.max_depth = max(self.max_depth, self.n)
            self.cv.notify_all()
            return t0, t1, d

    def get(self):
        """-> (item, wait start ns, pop ns, depth before)."""
        with self.cv:
            t0 = now()
            while not self.q:
                self.cancel.check()
                self.cv.wait(0.05)
            d = len(self.q)
            item, counted = self.q.popleft()
            self.n -= int(counted)
            t1 = now()
            self.cv.notify_all()
            return item, t0, t1, d

    def __len__(self):
        with self.cv:
            return len(self.q)


# ---------------------------------------------------------------------------------------------------- logs
class Log:
    """Preallocated columns (lists) of n rows; two threads may write DIFFERENT columns of the same row."""

    def __init__(self, cols: Sequence[str], n: int):
        self.cols = tuple(cols)
        self.n = int(n)
        self.c = {k: [None] * self.n for k in self.cols}

    def rows(self, m: Optional[int] = None) -> List[List]:
        m = self.n if m is None else m
        return [[self.c[k][i] for k in self.cols] for i in range(m)]


# --------------------------------------------------------------------------------------------------- the spec
@dataclass
class LayerIn:
    l: int
    list_ev: object                    # the list-D2H completion handle of THIS repetition (recorded by the bracket)
    plan_row: torch.Tensor             # the host row [1 + HB + HB*M] (int32), read only after list_ev completed
    src_k: torch.Tensor                # physical host K / V of the layer (src_layout)
    src_v: torch.Tensor


@dataclass
class RepSpec:
    arm: str
    layers: List[LayerIn]
    dst_k: torch.Tensor                # physical scratch K / V (dst_layout)
    dst_v: torch.Tensor
    H: int
    B: int
    M: int
    s_src: int
    s_dst: int
    src_layout: str = "hm"
    dst_layout: str = "orig"
    packer: str = "group"
    placer: str = "row"
    cap: int = 256
    full: bool = True
    faults: FeederFaults = field(default_factory=FeederFaults)
    max_chunks: int = 4096
    R: int = K.R_DEF
    D: int = K.D_DEF

    @property
    def HB(self):
        return self.H * self.B

    def desc(self, lay: LayerIn) -> K.Desc:
        return K.build_desc(lay.plan_row[1 + self.HB:].view(self.H, self.B, self.M), src_layout=self.src_layout, dst_layout=self.dst_layout,
                            s_src=self.s_src, s_dst=self.s_dst, packer=self.packer, placer=self.placer, R=self.R)


def wire_bytes(n: int, cap: int, placer: str, R: int = K.R_DEF, D: int = K.D_DEF) -> int:
    """Bytes of ONE chunk's H2D (index rows + K + V payload; stage_views' [lo, hi))."""
    return (n * R if placer == "row" else n) * K.IDX_BYTES + K.useful_bytes(n, R, D)


# ------------------------------------------------------------------------------------------------------- S0
def run_s0(spec: RepSpec, be, pipe: K.Pipe, wait_list: Callable) -> Dict:
    """S0 on the CALLING thread (the coordinator): per layer in order, wait for its list, build the descriptor live, then
    Pipe.layer (pack chunk k -> H2D -> scatter, then chunk k+1). The cpupack_transport.Runner.transport order without its two
    per-layer aux markers (no other arm records them); host stamps when full."""
    f = spec.faults
    full = spec.full
    out = []
    for i, lay in enumerate(spec.layers):
        r = dict(i=i, l=lay.l)
        t = now()
        wait_list(lay.list_ev)
        if full:
            r["lw0"], r["lw1"] = t, now()
        t, tc = now(), time.thread_time_ns()
        d = spec.desc(lay)
        if full:
            r["d0"], r["d1"], r["dcpu"] = t, now(), time.thread_time_ns() - tc
        r["n"] = d.n
        sk, dk = K.views_for(d, lay.src_k, spec.dst_k, spec.R)
        sv, dv = K.views_for(d, lay.src_v, spec.dst_v, spec.R)
        r["chunks"] = pipe.layer(d, sk, sv, dk, dv, mode="full", faults=f.pipe_faults(), lite=not full, host_stamps=full)
        out.append(r)
    return dict(arm="S0", layers=out)


def collect_s0(spec: RepSpec, res: Dict, cap: int) -> Dict:
    """S0's records in the common layout (after the repetition; never inside the timed region)."""
    lay_rows, ch, ev = [], [], []
    k = 0
    for r in res["layers"]:
        chunks = r.get("chunks") or []
        last = chunks[-1] if chunks else None
        lay_rows.append([r["i"], r["l"], r.get("n"), len(chunks), r.get("lw0"), r.get("lw1"), r.get("d0"), r.get("d1"), r.get("dcpu"), None, None, 0,
                         None, (last.get("sc1h") if last else r.get("d1"))])
        for c, x in enumerate(chunks):
            n = x["g1"] - x["g0"]
            ch.append([k, r["i"], r["l"], c, x["g0"], x["g1"], n, x["slot"], None, x["bytes"], K.useful_bytes(n, spec.R, spec.D),
                       x.get("fw0"), x.get("fw1"), x.get("p0"), x.get("p1"), x.get("pack_cpu_ns"), None, None, None,
                       None, None, None, x.get("sub0"), x.get("cp0"), x.get("cp1"), None, x.get("sc0h"), x.get("sc1h"), None, int(c == len(chunks) - 1)])
            ev.append([x.get(e) for e in EV_COLS])
            k += 1
    return dict(arm="S0", full=spec.full, placer=spec.placer, cap=spec.cap, lay_cols=LAYER_COLS, lay=lay_rows, ch_cols=CHUNK_COLS, ch=ch,
                ev_cols=EV_COLS, ev=ev, n_chunks=k, trans=None, backlog=None, max_ahead=0)


# --------------------------------------------------------------------------------------------------- S1 / S2
class _Item:
    __slots__ = ("kind", "k", "i", "l", "c", "s", "g", "n", "lo", "hi", "last", "dk", "dv")

    def __init__(self, kind, **kw):
        self.kind = kind
        for a in _Item.__slots__[1:]:
            setattr(self, a, kw.get(a))


class SplitState:
    """The shared state of the split arms, created once (the tables are reset per repetition)."""

    def __init__(self, ring: int, fifo_cap: Optional[int] = None):
        self.ring = int(ring)
        self.cancel = Cancel()
        self.slots = SlotTable(self.ring, self.cancel)
        self.fifo = ReadyFifo(fifo_cap or self.ring, self.cancel)
        self.ack_cv = threading.Condition()
        self.cancel.watch(self.ack_cv)
        self.acks: List[Optional[int]] = []

    def reset(self, n_layers: int):
        self.cancel.reset()
        self.slots.reset()
        self.fifo.reset()
        with self.ack_cv:
            self.acks = [None] * n_layers


class SplitRun:
    """One S1 / S2 repetition: produce() runs on the coordinator thread, submit() on the submitter thread."""

    def __init__(self, spec: RepSpec, st: SplitState, be: OwnedBackend, stage: torch.Tensor, land: torch.Tensor, wait_list: Callable,
                 wait_handle: Callable, done: Callable = is_done):
        if spec.arm not in ("S1", "S2"):
            raise ValueError(spec.arm)
        self.spec, self.st, self.be, self.stage, self.land = spec, st, be, stage, land
        self.wait_list, self.wait_handle, self.done = wait_list, wait_handle, done
        self.full = spec.full
        nl, mc = len(spec.layers), int(spec.max_chunks)
        self.ch = Log(_P_COLS + _S_COLS, mc)
        self.ev = Log(EV_COLS, mc)
        self.lay = Log(LAYER_COLS, nl)
        self.k = 0
        self.prod_done = threading.Event()
        self.sub_done = threading.Event()
        self.land_sc = [None] * st.ring
        self.dst2d = None
        st.reset(nl)
        self.max_ahead = 0

    # ---------------------------------------------------------------------------- the producer (coordinator thread)
    def _barrier(self, i_prev):
        st = self.st
        with st.ack_cv:
            while st.acks[i_prev] is None:
                st.cancel.check()
                st.ack_cv.wait(0.05)

    def _desc(self, i, lay, ahead):
        sp, full, L = self.spec, self.full, self.lay.c
        skip = sp.faults.desc_before_d2h and (ahead or sp.arm != "S2")     # the S2 control targets the LOOKAHEAD descriptor
        if not skip:
            t = now()
            self.wait_list(lay.list_ev)                                  # THIS repetition's list D2H of the layer completed
            if full:
                L["lw0"][i], L["lw1"][i] = t, now()
        t, tc = now(), time.thread_time_ns()
        d = sp.desc(lay)
        if full:
            L["d0"][i], L["d1"][i], L["dcpu"][i] = t, now(), time.thread_time_ns() - tc
        L["ahead"][i] = int(ahead)
        L["i"][i], L["l"][i], L["n"][i] = i, lay.l, d.n
        return d

    def produce(self, co=None):
        sp, st, f, full = self.spec, self.st, self.spec.faults, self.full
        C, L = self.ch.c, self.lay.c
        R, D, cap, ring = sp.R, sp.D, sp.cap, st.ring
        ahead = None                                                     # S2: the ONE descriptor built ahead (layer pos, desc)
        try:
            for i, lay in enumerate(sp.layers):
                if sp.arm == "S1" and i > 0:
                    t = now()
                    self._barrier(i - 1)                                 # S1: the descriptor waits for layer i-1's final submission
                    if full:
                        L["bw0"][i], L["bw1"][i] = t, now()
                if ahead is not None and ahead[0] == i:
                    d = ahead[1]
                    ahead = None
                else:
                    d = self._desc(i, lay, False)
                if sp.arm == "S2" and i > 0 and not f.s2_pack_cross:
                    t = now()
                    self._barrier(i - 1)                                 # S2: chunk planning + PACKING still wait for the same ack
                    if full:
                        L["bw0"][i], L["bw1"][i] = t, now()
                chunks = K.chunk_requests(d.req_groups.tolist(), cap)    # after the barrier in S1 AND S2: only the DESCRIPTOR crosses
                L["nch"][i] = len(chunks)
                sk, _ = K.views_for(d, lay.src_k, sp.dst_k, R)
                sv, _ = K.views_for(d, lay.src_v, sp.dst_v, R)
                spg, dpg = d.src_per_group, d.dst_per_group
                for c, (g0, g1) in enumerate(chunks):
                    if f.producer_raise_at == (i, c):
                        raise FeederInjected("producer exception control at layer position %d chunk %d" % (i, c))
                    k = self.k
                    if k >= sp.max_chunks:
                        raise RuntimeError("more than max_chunks=%d chunks in one repetition" % sp.max_chunks)
                    n, s = g1 - g0, k % ring
                    t = now()
                    g = st.slots.acquire(s, self.wait_handle, self.done)   # free-slot wait: the slot's H2D COMPLETE
                    t1 = now()
                    kst, vst, ist, lo, hi = K.stage_views(self.stage[s], n, d.packer, d.placer, cap, torch.bfloat16, R, D)
                    tp, tc = now(), time.thread_time_ns()
                    torch.index_select(sk, 0, d.src_idx[g0 * spg:g1 * spg], out=kst)
                    torch.index_select(sv, 0, d.src_idx[g0 * spg:g1 * spg], out=vst)
                    if f.poison_stage and i == 0 and c == 0:
                        kst.view(torch.int16).view(-1)[:D].fill_(K.POISON_I16 - 1)
                    ist.copy_(d.dst_idx[g0 * dpg:g1 * dpg])
                    tp1, tc1 = now(), time.thread_time_ns()
                    st.slots.ready(s, g)
                    last = c == len(chunks) - 1
                    it = _Item("chunk", k=k, i=i, l=lay.l, c=c, s=s, g=(g - 1 if f.stale_generation == (i, c) else g), n=n, lo=lo, hi=hi, last=last)
                    q0, q1, depth = st.fifo.put(it)
                    for col, v in (("k", k), ("i", i), ("l", lay.l), ("c", c), ("g0", g0), ("g1", g1), ("n", n), ("slot", s), ("gen", g),
                                   ("wire", hi - lo), ("useful", K.useful_bytes(n, R, D)), ("last", int(last))):
                        C[col][k] = v
                    if full:
                        C["fw0"][k], C["fw1"][k], C["p0"][k], C["p1"][k], C["pcpu"][k] = t, t1, tp, tp1, tc1 - tc
                        C["qf0"][k], C["pub"][k], C["dpub"][k] = q0, q1, depth
                    self.k = k + 1
                if not chunks:
                    st.fifo.put(_Item("layer_end", i=i, l=lay.l), counted=False)
                if full:
                    L["pub_last"][i] = now()
                if sp.arm == "S2" and i + 1 < len(sp.layers):
                    nxt = sp.layers[i + 1]                               # at most ONE descriptor ahead: after layer i's final chunk
                    ahead = (i + 1, self._desc(i + 1, nxt, True))       # was published and after layer i+1's own list D2H
                    self.max_ahead = max(self.max_ahead, 1)
            st.fifo.put(_Item("stop"), counted=False)
        except BaseException as e:
            st.cancel.fail("producer", e, traceback.format_exc())
            raise
        finally:
            self.prod_done.set()
        return self.k

    # ----------------------------------------------------------------------------- the submitter (submitter thread)
    def submit(self, worker=None):
        sp, st, f, full, be = self.spec, self.st, self.spec.faults, self.full, self.be
        C, E, L = self.ch.c, self.ev.c, self.lay.c
        R, D, cap = sp.R, sp.D, sp.cap
        try:
            be.bind()
            be.reset()                                                   # the owner's pool cursor (events read after the last rep)
            dk2 = K.rows2d(sp.dst_k) if sp.placer == "row" else K.groups2d(sp.dst_k, R)
            dv2 = K.rows2d(sp.dst_v) if sp.placer == "row" else K.groups2d(sp.dst_v, R)
            while True:
                tc = time.thread_time_ns()
                it, w0, w1, depth = st.fifo.get()
                if it.kind == "stop":
                    break
                if it.kind == "layer_end":
                    self._ack(it.i)
                    continue
                k, s, n = it.k, it.s, it.n
                if f.submitter_raise_at == (it.i, it.c):
                    raise FeederInjected("submitter exception control at layer position %d chunk %d" % (it.i, it.c))
                st.slots.take(s, it.g, f)                                # stale generation -> StaleGeneration
                E["pk"][k] = be.marker() if full else None
                t_sub0 = now()
                with be.stream("copy"):
                    if self.land_sc[s] is not None and not f.no_landing_wait:
                        be.stream_wait("copy", self.land_sc[s])          # the scatter that last read landing slot s is done
                    if f.delay_copy_ms > 0 and it.c == 0:
                        be.sleep(f.delay_copy_ms, "copy")
                    E["h2d0"][k] = be.event_rec("copy") if full else None
                    t_cp0 = now()
                    be.memcpy(self.land[s][it.lo:it.hi], self.stage[s][it.lo:it.hi], "copy")
                    t_cp1 = now()
                    h2d1 = be.event_rec("copy")
                E["h2d1"][k] = h2d1
                st.slots.publish(s, it.g, h2d1, f)                       # the H2D completion handle of (s, gen)
                t_hp = now()
                kd, vd, idd, _, _ = K.stage_views(self.land[s], n, sp.placer, sp.placer, cap, torch.bfloat16, R, D)
                t_sc0 = now()
                with be.stream("scatter"):
                    if not f.no_h2d_wait:
                        be.stream_wait("scatter", h2d1)                  # the scatter reads landing slot s only after its H2D
                    if f.delay_scatter_ms > 0 and it.c == 0:
                        be.sleep(f.delay_scatter_ms, "scatter")
                    E["sc0"][k] = be.event_rec("scatter") if full else None
                    if f.skip_scatter_chunk != it.c:
                        be.index_copy(dk2, idd, kd, "scatter")
                        be.index_copy(dv2, idd, vd, "scatter")
                    sc1 = be.event_rec("scatter")
                E["sc1"][k] = sc1
                self.land_sc[s] = sc1
                t_sc1 = now()
                E["sub"][k] = be.marker() if full else None
                if full:
                    C["iw0"][k], C["pop"][k], C["dpop"][k] = w0, w1, depth
                    C["sub0"][k], C["cp0"][k], C["cp1"][k], C["hp"][k], C["sc0h"][k], C["sc1h"][k] = t_sub0, t_cp0, t_cp1, t_hp, t_sc0, t_sc1
                    C["scpu"][k] = time.thread_time_ns() - tc
                if it.last:
                    if f.delay_ack_s > 0:
                        time.sleep(f.delay_ack_s)                        # CONTROLS ONLY: makes a missing barrier visible
                    self._ack(it.i)
        except BaseException as e:
            st.cancel.fail("submitter", e, traceback.format_exc())
            raise
        finally:
            self.sub_done.set()

    def _ack(self, i):
        st = self.st
        with st.ack_cv:
            st.acks[i] = now()
            st.ack_cv.notify_all()

    # ---------------------------------------------------------------------------------------------- collect
    def collect(self) -> Dict:
        """The records in the common layout (after both threads returned)."""
        L = self.lay.c
        for i in range(len(self.spec.layers)):
            L["ack"][i] = self.st.acks[i] if i < len(self.st.acks) else None
        return dict(arm=self.spec.arm, full=self.full, placer=self.spec.placer, cap=self.spec.cap, lay_cols=LAYER_COLS, lay=self.lay.rows(), ch_cols=CHUNK_COLS,
                    ch=[[self.ch.c[c][k] for c in CHUNK_COLS] for k in range(self.k)], ev_cols=EV_COLS,
                    ev=[[self.ev.c[e][k] for e in EV_COLS] for k in range(self.k)], n_chunks=self.k,
                    trans=list(self.st.slots.trans), backlog=dict(self.st.slots.backlog(), fifo=len(self.st.fifo), fifo_max=self.st.fifo.max_depth),
                    max_ahead=self.max_ahead)


def wait_split(run: SplitRun, timeout_s: float = 600.0):
    """Wait for BOTH threads of a split repetition; then re-raise the first failure (FeederError)."""
    t_end = time.time() + timeout_s
    for ev in (run.prod_done, run.sub_done):
        while not ev.wait(0.05):
            if time.time() > t_end:
                run.st.cancel.fail("caller", TimeoutError("split repetition exceeded %.0f s" % timeout_s))
                ev.wait(30)
                break
    c = run.st.cancel
    if c.err is not None:
        raise FeederError(c.where, c.err, c.tb)


# ------------------------------------------------------------------------------------------------- workers
class Job:
    def __init__(self, fn):
        self.fn, self.done, self.error, self.result, self.exc = fn, threading.Event(), None, None, None


class SubmitterThread(threading.Thread):
    """The persistent sole CUDA submitter: its mask is set once (inside the 8 transport cores); every job runs under
    inference mode (thread-local)."""

    def __init__(self, cpus: Sequence[int], name: str = "fd-submitter"):
        super().__init__(daemon=True, name=name)
        self.q = queue.Queue()
        self.cpus = list(cpus)
        self.tid = None
        self.mask = None
        self.ready = threading.Event()

    def run(self):
        self.tid = threading.get_native_id()
        if self.cpus:
            try:
                os.sched_setaffinity(0, set(self.cpus))
            except OSError:
                pass
        self.mask = sorted(os.sched_getaffinity(0))
        self.ready.set()
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                with torch.inference_mode():
                    job.result = job.fn(self)
            except BaseException as e:
                job.error = traceback.format_exc()
                job.exc = e
            finally:
                job.done.set()

    def submit(self, fn) -> Job:
        j = Job(fn)
        self.q.put(j)
        return j

    def call(self, fn):
        j = self.submit(fn)
        j.done.wait()
        if j.error:
            raise RuntimeError("submitter: " + j.error)
        return j.result

    def set_cpus(self, cpus):
        """Re-pin the submitter (its own thread does it)."""
        def f(w):
            os.sched_setaffinity(0, set(cpus))
            w.cpus = list(cpus)
            w.mask = sorted(os.sched_getaffinity(0))
            return w.mask
        return self.call(f)


# ------------------------------------------------------------------------------------------------- audits
def _rows(rec, which):
    cols = rec["%s_cols" % which]
    return [dict(zip(cols, r)) for r in rec[which]]


def audit_rep(rec: Dict, expect_useful: Optional[int] = None, expect_chunks: Optional[Sequence[Tuple[int, int, int]]] = None) -> List[str]:
    """Host-clock ordering, barrier, ownership and conservation audit of ONE repetition's records (empty = clean). Every
    comparison is between two stamps of the SAME clock (host perf_counter_ns)."""
    bad = []
    arm, full = rec["arm"], rec.get("full", True)
    lays = _rows(rec, "lay")
    chs = _rows(rec, "ch")
    if expect_chunks is not None:
        got = [(x["l"], x["g0"], x["g1"]) for x in chs]
        if got != [tuple(e) for e in expect_chunks]:
            bad.append("chunk list differs from the plan's (%d vs %d chunks)" % (len(got), len(expect_chunks)))
    if expect_useful is not None:
        u = sum(int(x["useful"]) for x in chs)
        if u != int(expect_useful):
            bad.append("useful bytes %d != plan %d (byte conservation)" % (u, expect_useful))
    for x in chs:
        if x["wire"] != wire_bytes(x["n"], rec.get("cap", 0), rec.get("placer", "row")):
            bad.append("chunk %s wire bytes %s != %d (index rows + K + V of %d groups)" % (x["k"], x["wire"], wire_bytes(x["n"], 0, rec.get("placer", "row")), x["n"]))
            break
    if rec.get("backlog"):
        b = rec["backlog"]
        if b.get("held") or b.get("fifo"):
            bad.append("final backlog: %d slots held, %d FIFO items" % (b.get("held", 0), b.get("fifo", 0)))
    if not full:
        return bad
    ack = {x["i"]: x["ack"] for x in lays}
    first_p0 = {}
    for x in chs:
        first_p0.setdefault(x["i"], x["p0"])
        seq = [x.get(k) for k in ("sub0", "cp0", "cp1", "sc0h", "sc1h")]
        if any(v is None for v in seq) or seq != sorted(seq):
            bad.append("chunk %s: submission stamps out of order sub0 <= cp0 <= cp1 (copy returns) <= sc0h (scatter) <= sc1h: %s" % (x["k"], seq))
        if arm != "S0":
            sq = [x.get(k) for k in ("p1", "pub", "pop", "sub0")]
            if any(v is None for v in sq) or sq != sorted(sq):
                bad.append("chunk %s: pack end <= publish <= pop <= submission start violated: %s" % (x["k"], sq))
            if x.get("hp") is None or not (x["cp1"] <= x["hp"] <= x["sc0h"]):
                bad.append("chunk %s: handle publication not between copy return and scatter submission" % x["k"])
    for x in lays:
        i = x["i"]
        if x["d0"] is None and x["n"] is None:
            continue
        if x["lw1"] is None or x["d0"] is None or x["d0"] < x["lw1"]:
            bad.append("layer position %s: descriptor built before its list wait completed" % i)
        if i == 0 or arm == "S0":
            continue
        a_prev = ack.get(i - 1)
        if a_prev is None:
            bad.append("layer position %d: no acknowledgement of layer position %d" % (i, i - 1))
            continue
        if arm == "S1" and x["d0"] < a_prev:
            bad.append("S1 barrier: layer position %d descriptor started before layer %d's final-submission acknowledgement" % (i, i - 1))
        if arm == "S2":
            p0 = first_p0.get(i)
            if p0 is not None and p0 < a_prev:
                bad.append("S2 barrier: layer position %d packing started before layer %d's final-submission acknowledgement" % (i, i - 1))
            prev = next((y for y in lays if y["i"] == i - 1), None)
            if prev is not None and prev.get("pub_last") is not None and x["d0"] < prev["pub_last"]:
                bad.append("S2 lookahead: layer position %d descriptor started before layer %d's final chunk was published" % (i, i - 1))
    if arm != "S0" and rec.get("max_ahead", 0) > 1:
        bad.append("S2 lookahead > 1 descriptor")
    if rec.get("trans") is not None:
        last_gen = {}
        for s, g, a, b, who in rec["trans"]:
            if b == PACKING:
                if g <= last_gen.get(s, 0):
                    bad.append("slot %d regranted at gen %d <= %d" % (s, g, last_gen.get(s, 0)))
                last_gen[s] = g
            if who.startswith("FAULT"):
                bad.append("ownership fault transition %s on slot %d gen %d" % (who, s, g))
    return bad


def chunk_list(req_by_layer: Sequence[Sequence[int]], layers: Sequence[int], cap: int) -> List[Tuple[int, int, int]]:
    """The plan's (layer, g0, g1) chunk list in submission order (identical for every arm)."""
    out = []
    for l in layers:
        for g0, g1 in K.chunk_requests(req_by_layer[l], cap):
            out.append((l, g0, g1))
    return out


def chunk_digest(chunks: Sequence[Tuple[int, int, int]], cap: int, placer: str) -> str:
    h = hashlib.sha256()
    for l, g0, g1 in chunks:
        h.update(("%d:%d:%d:%d;" % (l, g0, g1, wire_bytes(g1 - g0, cap, placer))).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- CPU placement
def audit_affinity(threads: Dict[str, Sequence[int]], team: Sequence[int], forbidden: Sequence[int] = ()) -> List[str]:
    """Every transport thread (producer, its OpenMP helpers, the submitter) is confined to the 8 team cores: no ninth core,
    never the decode launch core."""
    team, bad = set(team), []
    used = set()
    for name, cpus in threads.items():
        cs = set(int(c) for c in (cpus or []))
        if not cs:
            bad.append("%s: unknown affinity" % name)
            continue
        if not cs <= team:
            bad.append("%s runs on %s outside the transport cores %s" % (name, sorted(cs - team), sorted(team)))
        if cs & set(forbidden):
            bad.append("%s may run on the forbidden core(s) %s (decode launch core)" % (name, sorted(cs & set(forbidden))))
        used |= cs
    if len(used | team) > len(team):
        bad.append("transport threads span %d cores > the %d transport cores" % (len(used | team), len(team)))
    return bad


def tid_mask(tid: int) -> List[int]:
    try:
        return sorted(os.sched_getaffinity(tid))
    except OSError:
        return []


def sched_read(tid: int) -> Optional[Dict]:
    """/proc/self/task/<tid>/schedstat (on-CPU ns, RUNQUEUE-WAIT ns, timeslices) + context switches + the last CPU."""
    base = "/proc/self/task/%d" % tid
    try:
        with open(base + "/schedstat") as f:
            run, wait, slices = (int(x) for x in f.read().split()[:3])
    except (OSError, ValueError):
        return None
    out = dict(run_ns=run, wait_ns=wait, slices=slices)
    try:
        with open(base + "/status") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches"):
                    out["vcsw"] = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches"):
                    out["nvcsw"] = int(line.split()[1])
        with open(base + "/stat") as f:
            st = f.read()
        out["last_cpu"] = int(st[st.rindex(")") + 2:].split()[36])
    except (OSError, ValueError, IndexError):
        pass
    return out


def sched_snapshot(tids: Dict[str, int]) -> Dict[str, Optional[Dict]]:
    return {name: sched_read(t) for name, t in tids.items() if t}


def sched_delta(a: Dict, b: Dict) -> Dict[str, Dict]:
    out = {}
    for name, x in b.items():
        y = a.get(name)
        if not x or not y:
            continue
        out[name] = {k: (x[k] - y[k] if k != "last_cpu" else x[k]) for k in x if k in y}
    return out


# ------------------------------------------------------------------------------------------------- the schedule
FULL_ORDERS = ("012", "021", "102", "120", "201", "210") * 2
LIGHT_ORDERS = ("012", "120", "201")


def block_schedule(plan_steps: Sequence[int], first_step: int, n_full: int = 12, n_light: int = 3, warm_per_arm: int = 3,
                   full_orders: Sequence[str] = FULL_ORDERS, light_orders: Sequence[str] = LIGHT_ORDERS) -> List[Dict]:
    """The FROZEN gated-step schedule: one gated step for CORRECT + the warm-ups (3 per arm, order 012 repeated), then one
    per full block (order full_orders[j], plan step plan_steps[j % len]), then one per light block (light_orders[j], plan
    step plan_steps[j % len]). Fixed before the run; no adaptive selection."""
    if n_full > len(full_orders) or n_light > len(light_orders):
        raise ValueError("more blocks than registered orders")
    ps = list(plan_steps)
    out = [dict(gated_step=first_step, kind="correct_warmup", block=None, order="012" * warm_per_arm, plan_step=ps[0])]
    for j in range(n_full):
        out.append(dict(gated_step=first_step + 1 + j, kind="full", block=j, order=full_orders[j], plan_step=ps[j % len(ps)]))
    for j in range(n_light):
        out.append(dict(gated_step=first_step + 1 + n_full + j, kind="light", block=j, order=light_orders[j], plan_step=ps[j % len(ps)]))
    return out


def schedule_digest(sched: Sequence[Dict], plan_digests: Dict) -> str:
    return hashlib.sha256(json.dumps(dict(schedule=list(sched), plans=plan_digests), sort_keys=True).encode()).hexdigest()


def arms_of(order: str) -> List[str]:
    return ["S%s" % ch for ch in order]
