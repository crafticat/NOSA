"""SUSTAINED decode + feeder harness (paper 4.9: can CPU packing keep up over a sustained decode run; 4.4: interference of
data movement with decode): the pure, device-agnostic pieces. The SAME code runs on the GPU (driver sustained_feed.py,
cpupack_core.CudaBackend, torch.cuda.Event clock) and on CPU tensors (tests sustained_tests/test_sustained.py,
EagerBackend, HostClock).

LABEL (every output carries it): a NATURAL greedy decode trajectory of NOSA's shipped batched decode (offload=False: the
whole KV on the GPU, so the decode itself moves no KV over the link), re-warmed only where its warm-up-sized buffers
require it (rewarm_need below); beside it a feeder streams SYNTHETIC plans (rate = bytes / step, fragmentation = blocks
per request per step) from a pinned host KV of the cache's geometry into a GPU SCRATCH that the decode never reads. Not
live drafting, not a cache policy, not committed throughput.

UNITS. A group = 64 consecutive tokens of ONE KV head of one request in one layer (cpupack_core): K and V 16 KiB each,
GROUP_USEFUL = 32 KiB useful bytes. A RELEASE = the work issued at decode step t: n_req requests (rotating over the
batch) x frag groups each; request b's groups go to consecutive (layer, head) streams s = start_b(t) .. + frag - 1
(mod L * H), so one request's groups span ceil(frag / H) layers (frag > L * H wraps: a second group in the same stream),
each into the next scratch slot m of its (b, h) (a per-(b, h) rotating counter, mod M). useful bytes per release =
n_req * frag * GROUP_USEFUL. The feeder processes releases FIFO, layer by layer (ascending): the CPU arm runs
cpupack_core build_desc + Pipe.layer (pack -> one H2D per chunk on "copy" -> scatter on "scatter"); the GPU arm runs
the matched gather (the W8 persistent kernel, the same plan tensor, the same destinations) on "gather". A release's
completion = an event after its last layer's last operation.

RELEASE GATE. "gpu" (default): the feeder starts release t only after the GPU reached decode step t (it waits on the
step's start event): the data that step t asks for is known no earlier. "host": as soon as the host launched step t.

DEADLINE. Release t is needed by decode step t + d: deadline instant = the GPU start event of step t + d. lateness =
done - deadline (ms, GPU clock; > 0 late). Releases stop at step N - d (every deadline lies inside the run); the queue
then drains (drain time reported). on_time_fraction is also tabulated for d in DEADLINE_TABLE from the same events.

QUEUE. backlog_host[t] (sampled by the decode thread at the start of step t, never during a graph capture): releases
queued + the one being processed; outstanding_gpu[t] (post hoc, GPU clock): releases whose gate passed before step t
started and whose completion lies after it. Bounded test queue_trend: least-squares slope (releases per 100 steps) and
last-third mean minus middle-third mean over the feeding window; BOUNDED iff slope <= SLOPE_MAX and that growth <=
GROWTH_MAX (both stated in the output).

REWARM (rewarm_need). NOSA sizes its pooling buffers at the warm-up step (out_len = ceil((seq + 1) / 64),
nosa_llama.py decode_inference) and captures its pooling graph on the stage-1 score, whose last dimension is the
compressed length rounded up to 128 (infllmv2 flash_api.cpp mha_varlen_fwd_stage1: seqlen_k_rounded; the python
wrapper returns the unsliced result). A step needs a fresh warm-up when out_len grows (every 64 tokens: the newest block
would be unselectable) or when the rounded width changes (a compressed key is added every 16 tokens; the steady path's
score_buf.copy_ would raise). The decode thread computes both from the live cache before each step (layer 0:
cached_compressed_max_seqlen m, no_compress_k_len n; this step compresses iff n >= KERNEL) and sets has_buffers False.
A warm-up captures CUDA graphs (torch.cuda.graph, global capture mode): the feeder is PAUSED at a safe point (between
layers, no CUDA call in progress) for the whole warm-up step and resumed after it; the pause time is recorded.

V3 (feature/nosi-sustained-v3, the host-side diagnosis of run b64_s960_v2). ARM OPTIONS (parse_arms): a cpu arm's mode
(CPU_MODES: full; LP = pack into the pinned staging, no CUDA call; L = the per-layer Python skeleton only) and any arm's
sleep (a GPU sleep at the start of every step, before the decode: the decode's launches queue up behind it, as in the
finite-burst harness of job 2179735). Per step the decode thread also records its own CPU time (host_cpu_ms, from
time.thread_time_ns) and the GPU time after the sleep (decode_ms). A mode arm that moves no data must leave the scratch
POISON (content_check) and its last staging slot packed (LP) or still POISON (L) (staged_check).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

LABEL = ("SUSTAINED natural greedy decode (NOSA shipped batched decode, offload=False, re-warmed where its buffers "
         "require) beside a feeder of SYNTHETIC plans into a GPU SCRATCH; not live drafting, not committed throughput")
R_TOK, H_DEF, D_DEF, ELEM = 64, 2, 128, 2
GROUP_USEFUL = 2 * R_TOK * D_DEF * ELEM                # K + V bytes of one group: 32768
KERNEL, STRIDE, ROUND = 32, 16, 128                    # pooling_block_size, pooling_stride, flash_api seqlen_k rounding
METHODS = ("alone", "cpu", "gpu")
GATES = ("gpu", "host")
DEADLINE_TABLE = (1, 2, 3, 4, 8)
SLOPE_MAX = 0.5                                        # releases per 100 steps
GROWTH_MAX = 1.0                                       # releases (last-third mean - middle-third mean)
POISON_I16 = 0x7FC1                                    # a bf16 NaN bit pattern; the source pattern never produces it
PATTERN_MOD = 32512                                    # source int16 values in [0, 0x7F00): finite bf16 bit patterns


# ------------------------------------------------------------------------------------------------------------ arms
CPU_MODES = ("full", "L", "LP")      # cpupack_core.MODES used by a cpu arm: full = pack + H2D + scatter; LP = pack only
                                     # (host gather into pinned staging, no CUDA call); L = the Python skeleton only (plan,
                                     # descriptor, chunking, views: no pack, no index write, no CUDA call)
ARM_OPTIONS = ("mode", "sleep")
MAX_SLEEP_MS = 1000.0


@dataclass(frozen=True)
class ArmSpec:
    method: str                      # alone | cpu | gpu
    rate_mb: float = 0.0             # useful MiB per decode step (cpu / gpu)
    frag: int = 0                    # groups per served request per step
    mode: str = "full"               # cpu arms only: CPU_MODES
    sleep_ms: float = 0.0            # a GPU sleep enqueued on the decode stream at the start of every step (0 = none)

    @property
    def name(self) -> str:
        base = "alone" if self.method == "alone" else "%s_r%g_f%d" % (self.method, self.rate_mb, self.frag)
        if self.mode != "full":
            base += "_m" + self.mode
        if self.sleep_ms > 0:
            base += "_s%g" % self.sleep_ms
        return base

    @property
    def moves_data(self) -> bool:
        """True when the arm's work writes the GPU scratch (the content check expects the last writes); False for alone
        and for the cpu modes L / LP (the scratch must stay POISON)."""
        return self.method in ("cpu", "gpu") and self.mode == "full"


def parse_arms(text: str) -> List[ArmSpec]:
    """'alone,cpu:64:32,gpu:64:32' -> ArmSpecs. A transport arm is method:rate_MiB_per_step:frag. Options follow the
    positional parts as key=value: mode=full|L|LP (cpu arms only) and sleep=<ms> (any arm, 0 .. MAX_SLEEP_MS), e.g.
    'cpu:64:32:mode=L', 'alone:sleep=30', 'cpu:64:32:sleep=30'. Without options the names are the v2 names."""
    out = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        parts = item.split(":")
        method, pos, opts = parts[0], [], {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                if k not in ARM_OPTIONS or k in opts:
                    raise ValueError("arm %r: unknown or repeated option %r (options: %s)" % (item, k, ARM_OPTIONS))
                opts[k] = v
            elif opts:
                raise ValueError("arm %r: positional part %r after an option" % (item, p))
            else:
                pos.append(p)
        try:
            sleep = float(opts.get("sleep", "0"))
        except ValueError:
            raise ValueError("arm %r: sleep must be a number of ms" % item)
        if not 0.0 <= sleep <= MAX_SLEEP_MS:
            raise ValueError("arm %r: sleep %r ms outside [0, %g]" % (item, opts.get("sleep"), MAX_SLEEP_MS))
        mode = opts.get("mode", "full")
        if method == "alone" and not pos:
            if "mode" in opts:
                raise ValueError("arm %r: the alone arm takes no mode" % item)
            out.append(ArmSpec("alone", sleep_ms=sleep))
        elif method in ("cpu", "gpu") and len(pos) == 2:
            rate, frag = float(pos[0]), int(pos[1])
            if rate <= 0 or frag < 1:
                raise ValueError("arm %r: rate and frag must be positive" % item)
            if "mode" in opts and method != "cpu":
                raise ValueError("arm %r: mode applies to cpu arms only" % item)
            if mode not in CPU_MODES:
                raise ValueError("arm %r: mode %r not in %s" % (item, mode, CPU_MODES))
            out.append(ArmSpec(method, rate, frag, mode, sleep))
        else:
            raise ValueError("arm %r: expected 'alone' or 'cpu|gpu:<MiB per step>:<frag>', then key=value options" % item)
    if len({a.name for a in out}) != len(out):
        raise ValueError("duplicate arms in %r" % text)
    return out


def served_requests(rate_mb: float, frag: int, B: int) -> int:
    """n_req = rate / (frag * 32 KiB), exact: the rate must be a whole number of requests of frag groups, 1 <= n_req <= B
    (one request is served at most once per step)."""
    per_req = frag * GROUP_USEFUL
    n = rate_mb * (1 << 20) / per_req
    if abs(n - round(n)) > 1e-9:
        raise ValueError("rate %g MiB/step is not a whole number of requests of %d groups (%g)" % (rate_mb, frag, n))
    n = int(round(n))
    if not 1 <= n <= B:
        raise ValueError("rate %g MiB/step with frag %d serves %d requests per step; need 1 .. B = %d" % (rate_mb, frag, n, B))
    return n


# ------------------------------------------------------------------------------------------------------- schedule
@dataclass
class Geometry:
    B: int                           # batch (requests)
    S: int = 16128                   # source tokens per request (the host KV)
    L: int = 32                      # layers of the source (and plan layer ids)
    H: int = H_DEF
    D: int = D_DEF
    M: int = 64                      # scratch slots per (request, head)
    R: int = R_TOK

    @property
    def nblk(self) -> int:
        return self.S // self.R


@dataclass
class ReleaseSpec:
    """One release (decode step t): parallel int64 vectors over its groups, in assignment order."""
    t: int
    b: torch.Tensor
    h: torch.Tensor
    l: torch.Tensor
    m: torch.Tensor
    blk: torch.Tensor
    n_req: int

    @property
    def n_groups(self) -> int:
        return int(self.b.numel())

    @property
    def useful(self) -> int:
        return self.n_groups * GROUP_USEFUL

    def layers(self) -> List[int]:
        return sorted(set(int(x) for x in torch.unique(self.l).tolist()))

    def plan(self, l: int, geom: Geometry) -> torch.Tensor:
        """The (H, B, M) int32 plan of layer l: slot m of (h, b) receives host block blk; -1 = nothing."""
        sel = self.l == l
        p = torch.full((geom.H, geom.B, geom.M), -1, dtype=torch.int32)
        p[self.h[sel], self.b[sel], self.m[sel]] = self.blk[sel].to(torch.int32)
        return p


class Schedule:
    """Deterministic synthetic releases (module docstring UNITS). Built once per arm BEFORE the decode starts (no plan
    work on the decode thread); last_l / last_blk (B, H, M) = the last (layer, block) written into each scratch slot in
    processing order (releases in t order; inside a release every (b, h, m) occurs once), -1 = never written."""

    def __init__(self, geom: Geometry, n_req: int, frag: int, seed: int = 0):
        g = geom
        if not 1 <= n_req <= g.B or frag < 1:
            raise ValueError("n_req %d (B %d), frag %d" % (n_req, g.B, frag))
        per_bh = -(-frag // g.H)
        if per_bh > g.M:
            raise ValueError("frag %d puts %d groups per (request, head) in one release > M = %d slots" % (frag, per_bh, g.M))
        if g.nblk < 1:
            raise ValueError("source holds no whole block")
        self.g, self.n_req, self.frag, self.seed = g, int(n_req), int(frag), int(seed)
        self.ctr = torch.zeros((g.B, g.H), dtype=torch.int64)
        self.last_l = torch.full((g.B, g.H, g.M), -1, dtype=torch.int64)
        self.last_blk = torch.full((g.B, g.H, g.M), -1, dtype=torch.int64)
        self.next_t: Optional[int] = None

    def release(self, t: int) -> ReleaseSpec:
        """The release of step t (call in increasing t: the slot counters advance)."""
        if self.next_t is not None and t <= self.next_t - 1:
            raise ValueError("releases must be generated in increasing t (%d after %d)" % (t, self.next_t - 1))
        self.next_t = t + 1
        g, nreq, frag = self.g, self.n_req, self.frag
        LH = g.L * g.H
        reqs = (t * nreq + torch.arange(nreq, dtype=torch.int64)) % g.B
        start = (t * 7 + reqs * 13 + self.seed) % LH                                  # (nreq,)
        i = torch.arange(frag, dtype=torch.int64)
        s = (start[:, None] + i[None, :]) % LH                                         # (nreq, frag)
        wrap = (start[:, None] + i[None, :]) // LH
        l, h = s // g.H, s % g.H
        b = reqs[:, None].expand(nreq, frag)
        blk = (t * 2654435 + b * 40503 + s * 9973 + wrap * 7919 + self.seed * 104729) % g.nblk
        onehot = torch.nn.functional.one_hot(h, g.H)                                   # (nreq, frag, H)
        rank = (onehot.cumsum(dim=1) - onehot).gather(2, h[..., None]).squeeze(-1)     # earlier same-(b, h) groups
        m = (self.ctr[b, h] + rank) % g.M
        cnt = onehot.sum(dim=1)                                                        # (nreq, H)
        self.ctr[reqs] += cnt
        b, h, l, m, blk = (x.reshape(-1) for x in (b, h, l, m, blk))
        self.last_l[b, h, m] = l
        self.last_blk[b, h, m] = blk
        return ReleaseSpec(t=t, b=b.clone(), h=h.clone(), l=l.clone(), m=m.clone(), blk=blk.clone(), n_req=nreq)


# --------------------------------------------------------------------------------------------------------- rewarm
def round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def warm_state(seq: int, n_nocomp: int, m_comp: int) -> Dict[str, int]:
    """What a warm-up at a step that starts at cache length seq fixes: out_len = ceil((seq + 1) / 64) pooled blocks and
    the stage-1 score width round128(m'), m' = m + (n >= KERNEL) (the step's own compression)."""
    mp = int(m_comp) + (1 if int(n_nocomp) >= KERNEL else 0)
    return dict(seq=int(seq), out_len=-(-(int(seq) + 1) // R_TOK), width=round_up(mp, ROUND), m_next=mp)


def rewarm_need(seq: int, n_nocomp: int, m_comp: int, warm: Optional[Dict[str, int]]) -> Tuple[bool, str]:
    """(re-warm before this step?, reason): 'first' (no warm-up yet), 'out_len' (a new 64-token block), 'width' (the
    rounded score width changes), '' (steady)."""
    if warm is None:
        return True, "first"
    now = warm_state(seq, n_nocomp, m_comp)
    if now["out_len"] != warm["out_len"]:
        return True, "out_len"
    if now["width"] != warm["width"]:
        return True, "width"
    return False, ""


def simulate_cache_lengths(seq0: int, n0: int, m0: int, steps: int) -> List[Tuple[int, int, int]]:
    """(seq, n, m) at the start of each decode step, by the engine's own rules (cache_engine_gpu.py
    update_no_compress_k_decode / update_compress_k_decode): n >= KERNEL compresses (m + 1, n = STRIDE + 1), else n + 1;
    seq + 1 per step. For the tests and for the driver's dry plan of the re-warm steps."""
    out = []
    seq, n, m = int(seq0), int(n0), int(m0)
    for _ in range(int(steps)):
        out.append((seq, n, m))
        if n >= KERNEL:
            m, n = m + 1, STRIDE + 1
        else:
            n += 1
        seq += 1
    return out


# --------------------------------------------------------------------------------------------------- backends (CPU)
class _HostEv:
    __slots__ = ("t_ns", "done")

    def __init__(self):
        self.t_ns, self.done = time.perf_counter_ns(), True

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class EagerBackend:
    """CPU test backend with cpupack_core's backend interface: every operation runs at once on the calling thread, an
    event is complete when recorded (host perf_counter stamp). Thread-safe enough for one feeder thread (no shared state
    but a counter)."""

    def __init__(self, names=("copy", "scatter", "gather")):
        self.names = tuple(names)
        self.n_ops = 0

    @contextmanager
    def stream(self, name):
        yield

    def event_rec(self, name=None):
        return _HostEv()

    def marker(self):
        return _HostEv()

    def host_wait(self, ev):
        return None

    def stream_wait(self, name, ev):
        return None

    def memcpy(self, dst, src, name=None):
        dst.copy_(src)
        self.n_ops += 1

    def index_copy(self, dst2d, idx, src, name=None):
        dst2d.index_copy_(0, idx, src)
        self.n_ops += 1

    def sleep(self, ms, name=None):
        time.sleep(ms / 1000.0)

    def synchronize(self, order=None):
        return None


class HostClock:
    """The decode-step clock of the CPU tests: an event is a host perf_counter stamp; elapsed(a, b) = (b - a) ms."""

    def event(self):
        return _HostEv()

    def record(self, ev):
        ev.t_ns = time.perf_counter_ns()
        return ev

    @staticmethod
    def elapsed(a, b) -> float:
        return (b.t_ns - a.t_ns) / 1e6

    def synchronize(self):
        return None


class CudaClock:
    """The GPU clock: timing events recorded on the CURRENT stream; elapsed = torch's elapsed_time (ms, both complete)."""

    def event(self):
        return torch.cuda.Event(enable_timing=True)

    def record(self, ev):
        ev.record()
        return ev

    @staticmethod
    def elapsed(a, b) -> float:
        return float(a.elapsed_time(b))

    def synchronize(self):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------------------- feeder
class FeederError(RuntimeError):
    def __init__(self, exc: BaseException, tb: str):
        super().__init__("feeder failed: %s: %s" % (type(exc).__name__, exc))
        self.origin, self.origin_tb = exc, tb


@dataclass
class Release:
    """A release in the feeder: its spec, the gate event, and what the feeder records (host ns, the done event)."""
    spec: ReleaseSpec
    deadline_t: int
    gate: Any = None
    rel_ns: int = 0                  # host instant the decode thread enqueued it
    pop_ns: int = 0
    start_ns: int = 0                # after the gate wait
    end_ns: int = 0                  # after the last operation was submitted
    pause_ns: int = 0                # paused inside it
    gate_ns: int = 0                 # host ns spent waiting on the gate
    layers: int = 0
    done: Any = None                 # the completion event (backend event, after the last operation)


class Feeder:
    """ONE feeder thread: pops releases FIFO and runs work(release, feeder) (which calls feeder.safe_point() before each
    layer). pause() returns once the thread is at a safe point (between layers, or idle) and holds it there until
    resume(): the decode thread brackets a warm-up step (CUDA graph capture) with them. The first exception stops the
    thread; the decode thread sees .error at its next step and raises. Completed releases are kept in .done_list in
    processing order; .inflight holds those whose done event the decode thread has not yet seen complete."""

    def __init__(self, work: Callable[[Release, "Feeder"], None], affinity: Optional[Sequence[int]] = None,
                 name: str = "feeder"):
        self.work, self.affinity, self.name = work, (sorted(affinity) if affinity else None), name
        self.cv = threading.Condition()
        self.q: deque = deque()
        self.done_list: List[Release] = []
        self.inflight: deque = deque()
        self.busy = False
        self.pause_req = False
        self.paused = False
        self.stop_req = False
        self.error: Optional[FeederError] = None
        self.thread: Optional[threading.Thread] = None
        self.tid: Optional[int] = None
        self.pauses: List[Tuple[int, int]] = []          # (requested ns, acknowledged ns)
        self.paused_ns_total = 0
        self.observed_affinity: Optional[List[int]] = None

    # -- decode-thread side
    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("feeder already started")
        self.thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()
        with self.cv:
            while self.tid is None and self.error is None:
                self.cv.wait(timeout=1.0)

    def release(self, rel: Release) -> None:
        rel.rel_ns = time.perf_counter_ns()
        with self.cv:
            self.q.append(rel)
            self.cv.notify_all()

    def backlog(self) -> int:
        """Releases queued + the one in progress (sampled by the decode thread)."""
        with self.cv:
            return len(self.q) + (1 if self.busy else 0)

    def pause(self, timeout_s: float = 600.0) -> int:
        """Request a pause and wait until the thread is at a safe point; returns the wait in ns."""
        t0 = time.perf_counter_ns()
        with self.cv:
            self.pause_req = True
            self.cv.notify_all()
            deadline = time.monotonic() + timeout_s
            while not (self.paused or not self.busy or self.error is not None):
                if not self.cv.wait(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError("feeder did not reach a safe point in %.0f s" % timeout_s)
        t1 = time.perf_counter_ns()
        self.pauses.append((t0, t1))
        return t1 - t0

    def resume(self) -> None:
        with self.cv:
            self.pause_req = False
            self.cv.notify_all()

    def stop(self, drain: bool = True, timeout_s: float = 600.0) -> None:
        """drain: process every queued release first; then end the thread and join it."""
        with self.cv:
            self.pause_req = False
            if not drain:
                self.q.clear()
            self.stop_req = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=timeout_s)
            if self.thread.is_alive():
                raise TimeoutError("feeder thread did not end in %.0f s" % timeout_s)

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error

    # -- feeder-thread side
    def safe_point(self) -> None:
        """Called by work() before each layer (no CUDA call in progress): hold here while a pause is requested."""
        if not self.pause_req:
            return
        t0 = time.perf_counter_ns()
        with self.cv:
            self.paused = True
            self.cv.notify_all()
            while self.pause_req and not self.stop_req:
                self.cv.wait()
            self.paused = False
        dt = time.perf_counter_ns() - t0
        self.paused_ns_total += dt
        if self._cur is not None:
            self._cur.pause_ns += dt

    _cur: Optional[Release] = None

    def _run(self) -> None:
        try:
            if self.affinity:
                os.sched_setaffinity(0, self.affinity)
            self.observed_affinity = sorted(os.sched_getaffinity(0))
        except Exception:                                                    # noqa: BLE001 (recorded, not fatal)
            self.observed_affinity = None
        with self.cv:
            self.tid = threading.get_native_id()
            self.cv.notify_all()
        try:
            while True:
                with self.cv:
                    while (not self.q or self.pause_req) and not self.stop_req:
                        if self.pause_req:
                            self.paused = True
                            self.cv.notify_all()
                        self.cv.wait()
                    self.paused = False
                    if not self.q and self.stop_req:
                        return
                    rel = self.q.popleft()
                    self.busy = True
                rel.pop_ns = time.perf_counter_ns()
                self._cur = rel
                if rel.gate is not None:
                    rel.gate.synchronize()                        # the GPU reached step t (release gate "gpu")
                rel.start_ns = time.perf_counter_ns()
                rel.gate_ns = rel.start_ns - rel.pop_ns
                self.work(rel, self)
                rel.end_ns = time.perf_counter_ns()
                self._cur = None
                self.done_list.append(rel)
                self.inflight.append(rel)
                with self.cv:
                    self.busy = False
                    self.cv.notify_all()
        except BaseException as e:                                            # noqa: BLE001 (first error ends the thread)
            with self.cv:
                self.error = FeederError(e, traceback.format_exc())
                self.busy = False
                self.cv.notify_all()


def outstanding_on_gpu(fd: Feeder) -> int:
    """Releases handed to the device whose done event is not complete yet (non-blocking query; the decode thread pops
    completed ones from the front). Never called during a graph capture."""
    while fd.inflight and fd.inflight[0].done is not None and _query(fd.inflight[0].done):
        fd.inflight.popleft()
    return len(fd.inflight)


def _query(ev) -> bool:
    q = getattr(ev, "query", None)
    return bool(q()) if callable(q) else bool(getattr(ev, "done", True))


# ------------------------------------------------------------------------------------------------- transport work
def cpu_work_fn(K, be, pipe, geom: Geometry, src_k: Sequence[torch.Tensor], src_v: Sequence[torch.Tensor],
                scr_k: torch.Tensor, scr_v: torch.Tensor, mode: str = "full"):
    """The CPU-packing arm's work: per layer of the release (ascending), a LIVE descriptor (cpupack_core.build_desc, row
    packer / row placer, orig layouts, no tail slot) and Pipe.layer in `mode` (CPU_MODES). full: pack -> H2D on "copy"
    -> scatter on "scatter", the release's done event on "scatter" after its last chunk. L / LP: the same Python per
    layer without the H2D and the scatter (LP still packs into the pinned staging); their only CUDA call is the done
    marker on the idle aux stream (an upper bound of the host instant). work.last = the last chunk (descriptor, chunk
    record, layer) for staged_check."""
    if mode not in CPU_MODES:
        raise ValueError("mode %r not in %s" % (mode, CPU_MODES))
    S_dst = geom.M * geom.R
    last: Dict[str, Any] = {}

    def work(rel: Release, fd: Feeder) -> None:
        spec = rel.spec
        for l in spec.layers():
            fd.safe_point()
            plan = spec.plan(l, geom)
            desc = K.build_desc(plan, src_layout="orig", dst_layout="orig", s_src=geom.S, s_dst=S_dst, packer="row",
                                placer="row", R=geom.R, tail_slot=None)
            sk, dk = K.views_for(desc, src_k[l % len(src_k)], scr_k, geom.R)
            sv, dv = K.views_for(desc, src_v[l % len(src_v)], scr_v, geom.R)
            recs = pipe.layer(desc, sk, sv, dk, dv, mode=mode, lite=True)
            if recs:
                last.update(desc=desc, rec=recs[-1], l=l)
            rel.layers += 1
        rel.done = be.event_rec("scatter") if mode == "full" else be.marker()
    work.last = last
    work.mode = mode
    return work


def poison_stage(stage_u8: torch.Tensor, cap_groups: int, R: int = R_TOK) -> torch.Tensor:
    """Fill the payload region [P0, end) of every staging slot with POISON_I16 (the index region [0, P0) is the Pipe's,
    zeroed by Pipe.reset). Run before every arm, so a slot that a mode did not pack still holds the poison."""
    p0 = 8 * cap_groups * R                                  # cpupack_core.idx_region(cap, R): int64 indices
    for s in range(stage_u8.shape[0]):
        stage_u8[s, p0:].view(torch.int16).fill_(POISON_I16)
    return stage_u8


def staged_check(K, pipe, last: Dict[str, Any], src_k: Sequence[torch.Tensor], src_v: Sequence[torch.Tensor],
                 R: int = R_TOK) -> Dict[str, Any]:
    """The last chunk's staging slot against its source rows (after a full synchronize): packed = K and V payload equal
    the rows the descriptor names; poisoned = both still the POISON pattern. A cpu arm in full / LP must be packed and
    not poisoned; in L poisoned and not packed (stage_ok)."""
    if not last:
        return dict(chunks_seen=False, packed=False, poisoned=False)
    desc, r, l = last["desc"], last["rec"], last["l"]
    n = int(r["g1"]) - int(r["g0"])
    kst, vst, _, _, _ = K.stage_views(pipe.stage[r["slot"]], n, desc.packer, desc.placer, pipe.cap, pipe.dtype, R, pipe.D)
    sp = desc.src_per_group
    idx = desc.src_idx[int(r["g0"]) * sp:int(r["g1"]) * sp]
    out = {}
    for name, st, src in (("k", kst, src_k), ("v", vst, src_v)):
        s2, _ = K.views_for(desc, src[l % len(src)], src[l % len(src)], R)
        ref = s2.index_select(0, idx)
        out[name + "_packed"] = bool(torch.equal(st.view(torch.int16), ref.view(torch.int16)))
        out[name + "_poisoned"] = bool((st.view(torch.int16) == POISON_I16).all())
    return dict(chunks_seen=True, layer=l, slot=int(r["slot"]), groups=n, packed=out["k_packed"] and out["v_packed"],
                poisoned=out["k_poisoned"] and out["v_poisoned"])


def stage_ok(mode: str, st: Dict[str, Any]) -> bool:
    if not st.get("chunks_seen"):
        return False
    if mode == "L":
        return bool(st["poisoned"] and not st["packed"])
    return bool(st["packed"] and not st["poisoned"])


def content_check(scr_k: torch.Tensor, scr_v: torch.Tensor, src_k: Sequence[torch.Tensor], src_v: Sequence[torch.Tensor],
                  sched: "Schedule", moves_data: bool, R: int = R_TOK) -> Dict[str, Any]:
    """The scratch check of one drained arm: an arm that moves data must hold the schedule's last writes (and POISON
    elsewhere); an arm that moves none (cpu L / LP) must leave EVERY slot POISON (a stray write = a broken mode)."""
    if moves_data:
        ll, lb = sched.last_l, sched.last_blk
    else:
        ll, lb = torch.full_like(sched.last_l, -1), torch.full_like(sched.last_blk, -1)
    k = check_scratch(scr_k, src_k, ll, lb, R)
    v = check_scratch(scr_v, src_v, ll, lb, R)
    ok = all(not c["groups_bad"] and not c["untouched_bad"] for c in (k, v))
    return dict(k=k, v=v, expect=("last writes" if moves_data else "untouched"), ok=ok)


def gpu_work_fn(be, gather, geom: Geometry, src_k: Sequence[torch.Tensor], src_v: Sequence[torch.Tensor],
                scr_k: torch.Tensor, scr_v: torch.Tensor, ids_host: Sequence[torch.Tensor], ids_dev: Sequence[torch.Tensor]):
    """The matched GPU-gather arm's work: per layer (ascending), the SAME plan tensor copied into a ring of pinned host
    / device id buffers and gather(scr, src, ids) for K and V on "gather" (the W8 persistent kernel on the GPU, a torch
    reference on the CPU). A host id buffer is rewritten only after the H2D that last read it completed (host wait on
    the ring slot's event); the device buffer is read in stream order. The release's done event is recorded on
    "gather" after its last kernel."""
    ring = len(ids_host)
    state = dict(k=0, ev=[None] * ring)

    def work(rel: Release, fd: Feeder) -> None:
        spec = rel.spec
        for l in spec.layers():
            fd.safe_point()
            s = state["k"] % ring
            if state["ev"][s] is not None:
                be.host_wait(state["ev"][s])
            ids_host[s].copy_(spec.plan(l, geom))
            with be.stream("gather"):
                be.memcpy(ids_dev[s], ids_host[s], "gather")
                gather(scr_k, src_k[l % len(src_k)], ids_dev[s])
                gather(scr_v, src_v[l % len(src_v)], ids_dev[s])
                state["ev"][s] = be.event_rec("gather")
            state["k"] += 1
            rel.layers += 1
        rel.done = state["ev"][(state["k"] - 1) % ring]
    return work


def gather_reference(dst: torch.Tensor, src: torch.Tensor, ids: torch.Tensor, R: int = R_TOK) -> None:
    """The W8 kernel's copy by logical indexing (CPU tests): dst[b, m*R:(m+1)*R, h] = src[b, blk*R:(blk+1)*R, h] for
    every ids[h, b, m] = blk >= 0."""
    nz = (ids >= 0).nonzero()
    for h, b, m in nz.tolist():
        blk = int(ids[h, b, m])
        dst[b, m * R:(m + 1) * R, h].copy_(src[b, blk * R:(blk + 1) * R, h].to(dst.device))


# --------------------------------------------------------------------------------------------- source and checks
def fill_source(t: torch.Tensor, layer: int, kv: int) -> torch.Tensor:
    """Deterministic, distinct-per-row int16 bit patterns in a (B, S, H, D) bf16 tensor (finite bf16: values below
    0x7F00), written per request (bounded temporaries). value = (row * 131 + col * 7 + layer * 1009 + kv * 17) % MOD,
    row = (b * S + t) * H + h."""
    B, S, H, D = t.shape
    v = t.view(torch.int16).view(B, S * H, D)
    base = (torch.arange(S * H, dtype=torch.int32)[:, None] * 131 + torch.arange(D, dtype=torch.int32)[None, :] * 7)
    base = base % PATTERN_MOD                              # (row - b * S * H) * 131 + col * 7, reduced once
    for b in range(B):
        c = (b * S * H * 131 + layer * 1009 + kv * 17) % PATTERN_MOD
        v[b].copy_(((base + c) % PATTERN_MOD).to(torch.int16))
    return t


def source_value(b: int, t: int, h: int, col: int, S: int, H: int, layer: int, kv: int) -> int:
    """The int16 value fill_source writes at (b, t, h, col) (the tests' oracle)."""
    row = (b * S + t) * H + h
    return (row * 131 + col * 7 + layer * 1009 + kv * 17) % PATTERN_MOD


def poison(t: torch.Tensor) -> torch.Tensor:
    t.view(torch.int16).fill_(POISON_I16)
    return t


def check_scratch(scr: torch.Tensor, src: Sequence[torch.Tensor], last_l: torch.Tensor, last_blk: torch.Tensor,
                  R: int = R_TOK) -> Dict[str, int]:
    """The content check of one scratch tensor (K or V) after a drained arm: every slot (b, h, m) last written by
    (layer l, block blk) equals src[l % len(src)][b, blk*R:(blk+1)*R, h] bit for bit, and every never-written slot is
    still POISON (a stray write is caught). Returns counts and the first mismatching (b, h, m)."""
    B, SM, H, D = scr.shape
    M = SM // R
    out = dict(groups_checked=0, groups_bad=0, untouched_checked=0, untouched_bad=0, first_bad=None)
    sc = scr.view(torch.int16)
    for l in sorted(set(int(x) for x in torch.unique(last_l).tolist()) - {-1}):
        idx = (last_l == l).nonzero()
        if idx.numel() == 0:
            continue
        b, h, m = idx[:, 0], idx[:, 1], idx[:, 2]
        blk = last_blk[b, h, m]
        sv = src[l % len(src)].view(torch.int16)
        r = torch.arange(R, dtype=torch.int64)
        exp = sv[b[:, None], blk[:, None] * R + r[None, :], h[:, None]]               # (n, R, D) on the host
        got = sc[b.to(scr.device)[:, None], (m.to(scr.device) * R)[:, None] + r.to(scr.device)[None, :],
                 h.to(scr.device)[:, None]]
        eq = (got == exp.to(scr.device)).reshape(got.shape[0], -1).all(dim=1).cpu()
        out["groups_checked"] += int(eq.numel())
        nb = int((~eq).sum())
        out["groups_bad"] += nb
        if nb and out["first_bad"] is None:
            j = int((~eq).nonzero()[0])
            out["first_bad"] = [int(b[j]), int(h[j]), int(m[j]), l, int(blk[j])]
    un = (last_l == -1).nonzero()
    if un.numel():
        b, h, m = un[:, 0], un[:, 1], un[:, 2]
        r = torch.arange(R, dtype=torch.int64)
        got = sc[b.to(scr.device)[:, None], (m.to(scr.device) * R)[:, None] + r.to(scr.device)[None, :],
                 h.to(scr.device)[:, None]]
        okp = (got == POISON_I16).reshape(got.shape[0], -1).all(dim=1).cpu()
        out["untouched_checked"] = int(okp.numel())
        out["untouched_bad"] = int((~okp).sum())
    return out


# ------------------------------------------------------------------------------------------------------- metrics
def pctl(xs: Sequence[float], q: float) -> Optional[float]:
    v = sorted(float(x) for x in xs)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    f, c = int(math.floor(k)), int(math.ceil(k))
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def mean(xs: Sequence[float]) -> Optional[float]:
    v = [float(x) for x in xs]
    return sum(v) / len(v) if v else None


def periods(ts: Sequence[float]) -> List[float]:
    """Per-step period (ms): ts[t + 1] - ts[t] (the GPU start of the next step minus this step's); len(ts) - 1 values."""
    return [float(ts[i + 1]) - float(ts[i]) for i in range(len(ts) - 1)]


def block_bootstrap_ci(xs: Sequence[float], ys: Sequence[float], stat: Callable[[List[float], List[float]], float],
                       n_boot: int = 2000, block: int = 16, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of stat(x, y) over PAIRED steps by a moving-block bootstrap (blocks of `block` consecutive steps,
    resampled with replacement, keeping the pairing): steps are autocorrelated, so single steps are not the unit."""
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("paired series of equal, nonzero length")
    rng = random.Random(seed)
    blk = max(1, min(block, n))
    starts = list(range(0, n - blk + 1))
    vals = []
    for _ in range(n_boot):
        ix: List[int] = []
        while len(ix) < n:
            s0 = rng.choice(starts)
            ix.extend(range(s0, s0 + blk))
        ix = ix[:n]
        vals.append(stat([xs[i] for i in ix], [ys[i] for i in ix]))
    return pctl(vals, 100 * alpha / 2), pctl(vals, 100 * (1 - alpha / 2))


def paired_slowdown(alone: Dict[int, float], arm: Dict[int, float], steps: Sequence[int], n_boot: int = 2000,
                    block: int = 16, seed: int = 0) -> Dict[str, Any]:
    """Decode slowdown of an arm against decode-alone, PAIRED by decode step (the same trajectory: same tokens, same
    context length, same re-warm steps). ratio_of_means - 1 in %, its block-bootstrap CI, and the median per-step
    ratio - 1."""
    st = [t for t in steps if t in alone and t in arm]
    if not st:
        return dict(n_steps=0)
    xs, ys = [alone[t] for t in st], [arm[t] for t in st]

    def rom(a, b):
        return (sum(b) / sum(a) - 1.0) * 100.0
    lo, hi = block_bootstrap_ci(xs, ys, rom, n_boot=n_boot, block=block, seed=seed)
    return dict(n_steps=len(st), alone_mean_ms=mean(xs), arm_mean_ms=mean(ys), alone_p50_ms=pctl(xs, 50),
                arm_p50_ms=pctl(ys, 50), slowdown_pct=rom(xs, ys), ci_lo=lo, ci_hi=hi,
                median_ratio_pct=(pctl([b / a for a, b in zip(xs, ys)], 50) - 1.0) * 100.0, block=block, n_boot=n_boot)


def queue_trend(series: Sequence[float]) -> Dict[str, Any]:
    """The bounded test of a per-step queue series (module docstring QUEUE)."""
    n = len(series)
    if n < 6:
        return dict(n=n, verdict="TOO_SHORT")
    x = list(range(n))
    mx, my = sum(x) / n, sum(series) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, series)) / sxx
    third = n // 3
    mid = series[third:2 * third]
    last = series[2 * third:]
    growth = mean(last) - mean(mid)
    ok = slope * 100.0 <= SLOPE_MAX and growth <= GROWTH_MAX
    return dict(n=n, slope_per_100=slope * 100.0, growth=growth, mean_mid=mean(mid), mean_last=mean(last),
                max=max(series), slope_max=SLOPE_MAX, growth_max=GROWTH_MAX, verdict="BOUNDED" if ok else "GROWING")


def delivery(rel_t: Sequence[int], useful: Sequence[int], done_ms: Sequence[float], gate_ms: Sequence[float],
             ts: Sequence[float], d: int, w0: int, w1: int) -> Dict[str, Any]:
    """Delivery metrics of one arm from GPU-clock times (ms, one reference): ts[t] = start of decode step t (len N + 1:
    ts[N] = after the last step); per release i: its step rel_t[i], useful bytes, gate instant gate_ms[i] (= ts[t] for
    the gpu gate) and done_ms[i]. Window = steps [w0, w1) (releases inside it). Returns the rate, on-time fractions
    (count and bytes; for d and DEADLINE_TABLE), lateness quantiles, per-step delivered bytes and the GPU outstanding
    series."""
    N = len(ts) - 1
    lat = []
    for t, dn in zip(rel_t, done_ms):
        lat.append(float(dn) - float(ts[min(t + d, N)]))
    on = [x <= 0.0 for x in lat]
    nb = sum(useful)
    table = {}
    for dd in DEADLINE_TABLE:
        okd = [float(dn) <= float(ts[min(t + dd, N)]) for t, dn in zip(rel_t, done_ms)]
        table[str(dd)] = dict(count=(sum(okd) / len(okd)) if okd else None,
                              bytes=(sum(u for u, o in zip(useful, okd) if o) / nb) if nb else None)
    win_ms = float(ts[w1]) - float(ts[w0]) if w1 > w0 else 0.0
    done_in = sum(u for u, dn in zip(useful, done_ms) if float(dn) <= float(ts[w1]))
    srt = sorted((float(dn), u) for dn, u in zip(done_ms, useful))
    per_step_deliv, outstanding = [], []
    j, cum = 0, 0
    for t in range(w0, w1):
        while j < len(srt) and srt[j][0] <= float(ts[t + 1]):
            cum += srt[j][1]
            j += 1
        per_step_deliv.append(cum)
    for t in range(w0, w1 + 1):
        outstanding.append(sum(1 for g, dn in zip(gate_ms, done_ms) if float(g) <= float(ts[t]) < float(dn)))
    first_gate = min((float(g) for g in gate_ms), default=None)
    last_done = max((float(dn) for dn in done_ms), default=None)
    return dict(
        releases=len(rel_t), useful_bytes=nb, deadline_steps=d,
        on_time_count=(sum(on) / len(on)) if on else None,
        on_time_bytes=(sum(u for u, o in zip(useful, on) if o) / nb) if nb else None,
        on_time_table=table,
        lateness_ms=dict(p50=pctl(lat, 50), p95=pctl(lat, 95), p99=pctl(lat, 99), max=max(lat) if lat else None),
        window_steps=[w0, w1], window_ms=win_ms,
        delivered_in_window_bytes=done_in,
        delivered_GBps=(done_in / (win_ms / 1000.0) / 1e9) if win_ms > 0 else None,
        offered_GBps=(nb / (win_ms / 1000.0) / 1e9) if win_ms > 0 else None,
        span_GBps=(nb / ((last_done - first_gate) / 1000.0) / 1e9) if (last_done and first_gate is not None
                                                                     and last_done > first_gate) else None,
        drain_ms=(last_done - float(ts[w1])) if last_done is not None else None,
        delivered_cum_bytes=per_step_deliv, outstanding_gpu=outstanding)


# ------------------------------------------------------------------------------------------------------ CPU usage
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def parse_stat(text: str) -> Tuple[str, int, int, int]:
    """(comm, utime, stime, processor) from one /proc/<pid>/task/<tid>/stat line (comm may contain spaces and
    parentheses: split after the LAST ')'; fields 14, 15 and 39 of proc(5))."""
    rp = text.rindex(")")
    comm = text[text.index("(") + 1:rp]
    rest = text[rp + 2:].split()
    return comm, int(rest[11]), int(rest[12]), int(rest[36])


def cpu_snapshot(proc: str = "/proc/self/task") -> Dict[int, Tuple[str, int, int, int]]:
    out = {}
    try:
        tids = os.listdir(proc)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(os.path.join(proc, tid, "stat")) as f:
                out[int(tid)] = parse_stat(f.read())
        except (OSError, ValueError):
            continue
    return out


def cores_used(a: Dict[int, Tuple], b: Dict[int, Tuple], wall_s: float, main_tid: Optional[int],
               feeder_tid: Optional[int], clk: int = CLK_TCK) -> Dict[str, Any]:
    """CPU cores used between two snapshots: (delta utime + stime) / CLK_TCK / wall, total and for the decode (main)
    thread, the feeder thread and every other thread (the packer's intra-op workers among them)."""
    if wall_s <= 0:
        return dict(wall_s=wall_s)
    per, tot, main, feed, other = {}, 0.0, 0.0, 0.0, 0.0
    for tid, (comm, u, s, cpu) in b.items():
        u0, s0 = (a[tid][1], a[tid][2]) if tid in a else (0, 0)
        c = ((u - u0) + (s - s0)) / clk / wall_s
        if c <= 0:
            continue
        per[str(tid)] = dict(comm=comm, cores=c, last_cpu=cpu)
        tot += c
        if tid == main_tid:
            main += c
        elif tid == feeder_tid:
            feed += c
        else:
            other += c
    return dict(wall_s=wall_s, total=tot, main=main, feeder=feed, other=other, threads=per)


# ------------------------------------------------------------------------------------------- host / NUMA record
def parse_maps(text: str) -> List[Tuple[int, int]]:
    """(start, end) of every line of /proc/<pid>/maps."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if not f or "-" not in f[0]:
            continue
        a, b = f[0].split("-", 1)
        out.append((int(a, 16), int(b, 16)))
    return out


def parse_numa_maps(text: str) -> Dict[int, Dict[str, Any]]:
    """start -> {policy, nodes {node: pages}, kpage_kB, raw} from /proc/<pid>/numa_maps (proc(5)). A mapping the
    kernel cannot count (VM_PFNMAP / VM_IO, e.g. driver-mapped pinned memory) has no N<node>= fields."""
    out: Dict[int, Dict[str, Any]] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 2:
            continue
        try:
            start = int(f[0], 16)
        except ValueError:
            continue
        nodes, kp = {}, 4
        for tok in f[2:]:
            if tok.startswith("N") and "=" in tok and tok[1:].split("=", 1)[0].isdigit():
                k, v = tok[1:].split("=", 1)
                nodes[int(k)] = int(v)
            elif tok.startswith("kernelpagesize_kB="):
                kp = int(tok.split("=", 1)[1])
        out[start] = dict(policy=f[1], nodes=nodes, kpage_kB=kp, raw=line.strip())
    return out


def bytes_by_node(ranges: Sequence[Tuple[int, int]], maps: Sequence[Tuple[int, int]],
                  numa: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Bytes per NUMA node of the address ranges: every mapping that overlaps a range contributes its counted pages
    per node times the overlapped fraction of the mapping (pro rata: exact when a range covers whole mappings).
    uncounted_bytes = overlap inside mappings without page counts."""
    per: Dict[int, float] = {}
    lines, uncounted, covered = [], 0, 0
    for s, e in maps:
        ov = sum(max(0, min(e, b) - max(s, a)) for a, b in ranges)
        if ov <= 0:
            continue
        covered += ov
        info = numa.get(s)
        if info is None or not info["nodes"]:
            uncounted += ov
            if info is not None:
                lines.append(info["raw"])
            continue
        frac = ov / float(e - s)
        for node, pages in info["nodes"].items():
            per[node] = per.get(node, 0.0) + pages * info["kpage_kB"] * 1024.0 * frac
        lines.append(info["raw"])
    return dict(bytes_by_node={str(k): v for k, v in sorted(per.items())}, covered_bytes=covered,
                uncounted_bytes=uncounted, n_mappings=len(lines), lines=lines[:64])


def parse_node_meminfo(text: str) -> Dict[str, int]:
    """{key: kB} from /sys/devices/system/node/node<N>/meminfo ('Node 0 MemFree:  123 kB')."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 4 and f[0] == "Node" and f[2].endswith(":"):
            try:
                out[f[2][:-1]] = int(f[3])
            except ValueError:
                continue
    return out


def node_memfree(root: str = "/sys/devices/system/node") -> Dict[str, int]:
    """{node: MemFree kB} of every NUMA node (empty when the sysfs tree is absent)."""
    out = {}
    try:
        names = sorted(n for n in os.listdir(root) if n.startswith("node") and n[4:].isdigit())
    except OSError:
        return out
    for n in names:
        try:
            with open(os.path.join(root, n, "meminfo")) as f:
                out[n[4:]] = parse_node_meminfo(f.read()).get("MemFree", -1)
        except OSError:
            continue
    return out


def host_topology(root: str = "/sys/devices/system/node", status: str = "/proc/self/status") -> Dict[str, Any]:
    """This process's allowed CPUs / memory nodes and every node's CPU list."""
    out: Dict[str, Any] = dict(nodes={})
    try:
        with open(status) as f:
            for line in f:
                if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
                    k, v = line.split(":", 1)
                    out[k] = v.strip()
    except OSError:
        pass
    try:
        for n in sorted(x for x in os.listdir(root) if x.startswith("node") and x[4:].isdigit()):
            with open(os.path.join(root, n, "cpulist")) as f:
                out["nodes"][n[4:]] = f.read().strip()
    except OSError:
        pass
    return out


def numa_record(tensors: Sequence[torch.Tensor], proc: str = "/proc/self") -> Dict[str, Any]:
    """Where the host pages of `tensors` live: /proc/self/maps + numa_maps, restricted to the tensors' address ranges
    (bytes_by_node). Best effort: an unreadable file gives an 'error' entry, never an exception."""
    ranges = sorted((int(t.data_ptr()), int(t.data_ptr()) + t.numel() * t.element_size()) for t in tensors)
    try:
        with open(os.path.join(proc, "maps")) as f:
            maps = parse_maps(f.read())
        with open(os.path.join(proc, "numa_maps")) as f:
            numa = parse_numa_maps(f.read())
    except OSError as e:
        return dict(error="%s: %s" % (type(e).__name__, e), ranges=len(ranges))
    out = bytes_by_node(ranges, maps, numa)
    out["range_bytes"] = sum(b - a for a, b in ranges)
    return out


# --------------------------------------------------------------------------------------------------------- output
STEP_FIELDS = ("arm", "t", "rewarm", "rewarm_why", "start_ms", "period_ms", "span_ms", "host_iter_ms", "released_groups",
               "backlog_host", "outstanding_host", "pause_ms", "decode_ms", "host_cpu_ms")
RELEASE_FIELDS = ("arm", "t", "deadline_t", "groups", "useful", "layers", "gate_ms", "done_ms", "lateness_ms",
                  "on_time", "queue_wait_ms", "gate_wait_ms", "service_ms", "pause_ms")


def jsonl_line(fields: Sequence[str], rec: Dict[str, Any]) -> str:
    missing = [k for k in fields if k not in rec]
    if missing:
        raise KeyError("record lacks %s" % missing)
    return json.dumps({k: rec[k] for k in fields}, sort_keys=False) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
