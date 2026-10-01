"""INTERFERENCE CURVE: the bounded SUSTAINED-transfer pump, the train of natural plans and every accounting rule, pure
(CPU-testable; no CUDA). Authorized 2026-10-01 (the user via Codex; retroinfer-eval REPRODUCE.md 'INTERFERENCE-CURVE
EXTENSION AUTHORIZED'). The GPU driver is interference_curve.py; the CPU tests are retroinfer-eval
tests/test_interference_curve.py.

LABELS (every row carries them):
  'transport/placement-ready, NOT live-LRU-ready'   (cpupack_core.LABEL: replay into a SEPARATE scratch with the cache's
                                                      geometry; no last-reader / publication mechanism is measured)
  SUSTAINED_LABEL   'GPU-resident trace replay' of a FINITE TRAIN of natural per-(step, layer) plans through a bounded
                    in-flight queue, beside back-to-back resident decode ticks (not decode-paced)
  SATURATED_LABEL   'resource-contention control, NOT live verifier throughput' (the train runs as fast as the arm can)
  PREBUILT_LABEL    'prebuilt-plan copy microbenchmark, NOT an integrated LRU getter baseline' (every copy row)
  HS_LABEL          'HiSparse kernel, NOSI-layout i256 adapter -- NOT native HiSparse layout' (native HiSparse = job 2175759)
  RESIDENT_LABEL    resident tokens/s = B x 1000 / decode tick ms: RESIDENT, NOT committed speculative throughput

THE TRAIN. Request i = the natural load plan of ONE (trace step, layer): plan rows of the GPU-resident plan store (the
capture), in (step, layer) order over CV_TRAIN_STEPS consecutive trace steps. useful = K + V bytes of its groups
(cpupack_core.useful_bytes); wire = the bytes the arm moves over PCIe for it (== useful for the GPU gathers; + the int64
destination indices for the CPU packer). The same train (train_digest) and the same tick count K serve every arm of a
gated step; the decode-alone and transfer-alone controls run the SAME train / K.

THE PUMP (pump(); runs on the coordinator thread, NEVER on the decode thread). Requests are issued in train order on the
side stream(s):
  * bounded in-flight queue: before issuing request i the pump host-waits for the OLDEST outstanding request while
    `queue` requests are outstanding, so at most `queue` requests are ever issued and not known complete
    (inflight_at_issue < queue is recorded per request);
  * scratch dependencies: request i first stream-waits for the completion event of the previous request that wrote the
    same scratch region (region = i mod n_regions), so a region is never overwritten before its previous transfer
    completed, on any number of side streams (on ONE side stream the wait is implied by stream order and kept explicit);
  * every side stream first waits for the window's gate (recorded on the decode stream before the first tick);
  * per request: issue marker (an idle aux stream: the host instant on the GPU clock), g0 / g1 events around the
    launch, the host wait for a free queue entry (wait_ns) and the host issue time (api_ns).
Nothing here sleeps; the only blocking call is the pump's own host_wait (cudaEventSynchronize, which releases the GIL,
cpupack_core.CudaBackend.host_wait).

ACCOUNTING (all times in ms from the window's gate; a request's busy interval is [g0, g1]):
  window          [t0 of tick 0, t1 of tick K-1] on the decode stream
  overlap_frac    |union of busy intervals inside the window| / |union of busy intervals|  (realized overlap)
  window_gbps     useful bytes of requests COMPLETED inside the window / window time  (the WINDOW delivery rate)
  window_gbps_prorata  the same with each request's bytes pro-rata to its busy time inside the window
  during_gbps     pro-rata bytes inside the window / busy time inside the window  (transfer rate DURING overlap)
  full_ready_ms   gate -> the last request's completion (the FULL completion time of the train)
  alone_gbps      train useful bytes / (last completion - first start) of a transfer-alone train
  tick coverage   per tick, the fraction of [t0, t1] covered by busy intervals; COVERED ticks (>= cover_min) after the
                  first `skip_first` ticks are the 'decode under sustained transfer' samples; decode-alone uses the same
                  tick positions (skip_first excluded)
  pump latency    completion of the request a queue wait waited for -> the issue marker of the next request (the pump
                  thread's wake-up incl. GIL acquisition); side idle = gaps between consecutive busy intervals
  K               choose_K: ceil(margin x the slowest arm's transfer-alone full_ready_ms / decode-alone tick p50) +
                  skip_first, clamped to [k_min, k_max]; chosen ONCE per batch and frozen (no adaptive extra passes)
CAPACITY: pinned_plan (torch's CachingHostAllocator rounds every pinned block to a power of two,
ATen/core/CachingHostAllocator.h:132), classify_exception / classify_stop (GPU vs host / pinned vs other).
CONTROLS: controls_verdict lives in cpupack_transport.py (both tables judge a CONTROLS payload by each control's own
verdict, never by the raw 'fails' count, which counts the expected GATE_FAILs of the detected controls).
"""
from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SUSTAINED_LABEL = ("GPU-resident trace replay: a FINITE TRAIN of natural per-(step, layer) plans through a bounded in-flight "
                   "queue beside back-to-back resident decode ticks (not decode-paced)")
SATURATED_LABEL = "resource-contention control, NOT live verifier throughput"
PREBUILT_LABEL = "prebuilt-plan copy microbenchmark, NOT an integrated LRU getter baseline"
HS_LABEL = "HiSparse kernel, NOSI-layout i256 adapter -- NOT native HiSparse layout"
RESIDENT_LABEL = "RESIDENT tokens/s (B x 1000 / decode tick ms; NOT committed speculative throughput)"
GROUP_USEFUL = 32768                      # K + V bytes of one (head, request, block) group: 2 x 64 x 128 x 2
IDX_BYTES_PER_ROW = 8                     # the CPU packer ships one int64 destination index per 256-B row (cpupack_core)
ROWS_PER_GROUP = 64


# ------------------------------------------------------------------------------------------------------------ the train
@dataclass(frozen=True)
class Req:
    i: int
    step: int
    layer: int
    groups: int
    useful: int
    wire: int
    region: int


def build_train(groups_of: Callable[[int, int], int], steps: Sequence[int], n_layers: int, n_regions: int = 1,
                wire_extra_per_group: int = 0) -> List[Req]:
    """The requests of one train in (step, layer) order. groups_of(step, layer) = the natural plan's loaded groups (both
    heads, all requests, non-tail slots). wire = useful + wire_extra_per_group x groups (0 for the GPU gathers)."""
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    out = []
    for s in steps:
        for l in range(n_layers):
            g = int(groups_of(int(s), l))
            if g < 0:
                raise ValueError("negative group count at step %d layer %d" % (s, l))
            i = len(out)
            out.append(Req(i=i, step=int(s), layer=l, groups=g, useful=g * GROUP_USEFUL, wire=g * (GROUP_USEFUL + wire_extra_per_group),
                           region=i % n_regions))
    return out


def train_digest(train: Sequence[Req]) -> str:
    h = hashlib.sha256()
    for r in train:
        h.update(("%d,%d,%d,%d;" % (r.step, r.layer, r.groups, r.region)).encode())
    return h.hexdigest()[:16]


def train_bytes(train: Sequence[Req]) -> Dict[str, int]:
    return dict(requests=len(train), groups=sum(r.groups for r in train), useful=sum(r.useful for r in train),
                wire=sum(r.wire for r in train))


# ------------------------------------------------------------------------------------------------------------ the pump
def pump(be, train: Sequence[Req], launch: Callable[[Req, str], None], gate, queue: int,
         stream_of: Optional[Callable[[Req], str]] = None, deps: bool = True,
         clock_ns: Callable[[], int] = time.perf_counter_ns) -> List[Dict]:
    """Issue `train` in order (module docstring, THE PUMP). be: a cpupack_core backend (CudaBackend on the GPU, CpuBackend /
    ImmediateBackend in the tests). launch(req, stream_name) enqueues the request's copy on the CURRENT stream (the caller's
    `be.stream(name)` context is active). deps=False removes the scratch-dependency wait (a NEGATIVE CONTROL only)."""
    if queue < 1:
        raise ValueError("queue must be >= 1")
    stream_of = stream_of or (lambda r: "side")
    outstanding = deque()
    last_on_region: Dict[int, Tuple[int, object]] = {}
    gated_streams = set()
    recs = []
    for r in train:
        rec = asdict(r)
        t = clock_ns()
        waited_for = None
        while len(outstanding) >= queue:
            waited_for, ev = outstanding.popleft()
            be.host_wait(ev)
        rec["wait_ns"] = clock_ns() - t
        rec["waited_for"] = waited_for
        rec["inflight_at_issue"] = len(outstanding)
        s = stream_of(r)
        t = clock_ns()
        rec["issue"] = be.marker()
        with be.stream(s):
            if s not in gated_streams:
                be.stream_wait(s, gate)
                gated_streams.add(s)
            prev = last_on_region.get(r.region)
            rec["dep"] = None
            if prev is not None and deps:
                be.stream_wait(s, prev[1])
                rec["dep"] = prev[0]
            rec["g0"] = be.event_rec(s)
            launch(r, s)
            rec["g1"] = be.event_rec(s)
        rec["stream"] = s
        rec["api_ns"] = clock_ns() - t
        outstanding.append((r.i, rec["g1"]))
        last_on_region[r.region] = (r.i, rec["g1"])
        recs.append(rec)
    return recs


class ImmediateBackend:
    """A thread-safe CPU twin of cpupack_core.CudaBackend in which every enqueued operation runs AT ONCE (an idle GPU):
    events complete when recorded (t_ns stamped), waits are no-ops. For the CPU tests of the threaded window (the pump on
    the coordinator thread, ticks on the main thread); the DEFERRED cpupack_core.CpuBackend is the worst-case model for
    the dependency / lifetime negative controls."""

    class Ev:
        __slots__ = ("done", "t_ns", "stream")

        def __init__(self, stream=None):
            self.done, self.t_ns, self.stream = True, time.perf_counter_ns(), stream

    def __init__(self):
        self.lock = threading.RLock()
        self.host_waits = []                                     # (thread name, event) of every host_wait
        self.n_ops = 0
        self.cur = threading.local()

    @contextmanager
    def stream(self, name):
        prev = getattr(self.cur, "name", None)
        self.cur.name = name
        try:
            yield
        finally:
            self.cur.name = prev

    def event_rec(self, name=None):
        return ImmediateBackend.Ev(name)

    def marker(self):
        return ImmediateBackend.Ev(None)

    def host_wait(self, ev):
        with self.lock:
            self.host_waits.append((threading.current_thread().name, ev))

    def stream_wait(self, name, ev):
        pass

    def memcpy(self, dst, src, name=None):
        with self.lock:
            dst.copy_(src)
            self.n_ops += 1

    def index_copy(self, dst2d, idx, src, name=None):
        with self.lock:
            dst2d.index_copy_(0, idx, src)
            self.n_ops += 1

    def sleep(self, ms, name=None):
        pass

    def synchronize(self, order=None):
        pass

    def reset(self):
        pass


def ev_done(ev) -> bool:
    """Non-blocking completion test of a backend event (CUDA: Event.query())."""
    q = getattr(ev, "query", None)
    return bool(q()) if callable(q) else bool(getattr(ev, "done", False))


def ev_ms(ref, ev) -> Optional[float]:
    """ms from `ref` to `ev` (both complete): CUDA elapsed_time, or the CPU twins' t_ns."""
    if ev is None or ref is None:
        return None
    et = getattr(ref, "elapsed_time", None)
    if callable(et):
        return float(et(ev))
    if getattr(ref, "t_ns", None) is None or getattr(ev, "t_ns", None) is None:
        return None
    return (ev.t_ns - ref.t_ns) / 1e6


# ------------------------------------------------------------------------------------------------------- interval sums
def union_len(iv: Iterable[Tuple[float, float]]) -> float:
    tot, cur = 0.0, None
    for s, e in sorted((float(a), float(b)) for a, b in iv):
        if e <= s:
            continue
        if cur is None or s > cur[1]:
            if cur is not None:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot


def intersect_len(iv: Iterable[Tuple[float, float]], lo: float, hi: float) -> float:
    return union_len([(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi])


def idle_gaps(iv: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Gaps of the side stream between consecutive busy intervals (sorted by start; overlapping intervals merge)."""
    gaps = []
    end = None
    for s, e in sorted(iv):
        if end is not None and s > end:
            gaps.append(s - end)
        end = e if end is None else max(end, e)
    return dict(n=len(gaps), total_ms=sum(gaps), max_ms=(max(gaps) if gaps else 0.0))


def pctl(xs: Iterable[float], q: float) -> float:
    """numpy's default linear-interpolation percentile of individual samples; NaN for none."""
    v = sorted(float(x) for x in xs if x is not None and x == x)
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ------------------------------------------------------------------------------------------------------------- metrics
def tick_coverage(ticks: Sequence[Tuple[float, float]], iv: Sequence[Tuple[float, float]]) -> List[float]:
    return [(intersect_len(iv, a, b) / (b - a)) if b > a else 0.0 for a, b in ticks]


def tick_ms(ticks: Sequence[Tuple[float, float]]) -> List[float]:
    return [b - a for a, b in ticks]


def steady_ticks(ticks, skip_first: int = 1, cov: Optional[Sequence[float]] = None, cover_min: float = 0.95) -> List[float]:
    """Decode tick ms after the first `skip_first` ticks; with `cov`, only the COVERED ticks (coverage >= cover_min)."""
    d = tick_ms(ticks)
    return [d[k] for k in range(skip_first, len(d)) if cov is None or cov[k] >= cover_min]


def transfer_metrics(reqs: Sequence[Dict]) -> Dict:
    """Train-level numbers of one window from per-request dicts with g0 / g1 (ms from the gate), useful, wire and the
    pump's issue / waited_for fields."""
    nan = float("nan")
    rq = [q for q in reqs if q.get("g0") is not None and q.get("g1") is not None]
    if not rq:
        return dict(requests=0, useful_bytes=0, wire_bytes=0, span_ms=nan, alone_gbps=nan, full_ready_ms=nan)
    iv = [(q["g0"], q["g1"]) for q in rq]
    first, last = min(q["g0"] for q in rq), max(q["g1"] for q in rq)
    tot = sum(int(q["useful"]) for q in rq)
    span = last - first
    by_i = {q["i"]: q for q in rq}
    lat = [q["issue"] - by_i[q["waited_for"]]["g1"] for q in rq
           if q.get("waited_for") is not None and q.get("issue") is not None and q["waited_for"] in by_i]
    queue_wait = [q["g0"] - q["issue"] for q in rq if q.get("issue") is not None]
    return dict(requests=len(rq), useful_bytes=tot, wire_bytes=sum(int(q.get("wire", q["useful"])) for q in rq), first_start_ms=first,
                full_ready_ms=last, span_ms=span, busy_ms=union_len(iv), alone_gbps=(tot / (span * 1e6) if span > 0 else nan),
                side_idle=idle_gaps(iv), pump_latency_p50_ms=pctl(lat, 50), pump_latency_p95_ms=pctl(lat, 95),
                issue_to_start_p50_ms=pctl(queue_wait, 50), issue_to_start_p95_ms=pctl(queue_wait, 95),
                max_inflight_at_issue=max((int(q["inflight_at_issue"]) for q in rq if q.get("inflight_at_issue") is not None), default=None))


def window_metrics(ticks: Sequence[Tuple[float, float]], reqs: Sequence[Dict], cover_min: float = 0.95, skip_first: int = 1) -> Dict:
    """The overlap numbers of one window (module docstring, ACCOUNTING)."""
    nan = float("nan")
    out = dict(transfer_metrics(reqs))
    if not ticks:
        return out
    w0, w1 = ticks[0][0], ticks[-1][1]
    W = w1 - w0
    rq = [q for q in reqs if q.get("g0") is not None and q.get("g1") is not None]
    iv = [(q["g0"], q["g1"]) for q in rq]
    busy = union_len(iv)
    busy_in = intersect_len(iv, w0, w1)
    done_in = sum(int(q["useful"]) for q in rq if w0 <= q["g1"] <= w1)
    pro = 0.0
    for q in rq:
        d = q["g1"] - q["g0"]
        if d > 0:
            pro += int(q["useful"]) * max(0.0, min(q["g1"], w1) - max(q["g0"], w0)) / d
        elif w0 <= q["g1"] <= w1:
            pro += int(q["useful"])
    cov = tick_coverage(ticks, iv)
    covered = [k for k in range(skip_first, len(ticks)) if cov[k] >= cover_min]
    out.update(window_ms=W, window_start_ms=w0, window_end_ms=w1, busy_in_window_ms=busy_in,
               overlap_frac=(busy_in / busy if busy > 0 else nan), window_bytes_completed=done_in,
               window_gbps=(done_in / (W * 1e6) if W > 0 else nan), window_bytes_prorata=pro,
               window_gbps_prorata=(pro / (W * 1e6) if W > 0 else nan), during_gbps=(pro / (busy_in * 1e6) if busy_in > 0 else nan),
               train_done_in_window=bool(rq) and max(q["g1"] for q in rq) <= w1, tick_coverage=cov, covered_ticks=covered,
               n_covered=len(covered))
    return out


def choose_K(tick_p50_ms: float, alone_full_ms: Dict[str, float], margin: float = 1.5, k_min: int = 4, k_max: int = 96,
             skip_first: int = 1) -> Dict:
    """The registered tick-count rule (module docstring, K)."""
    if not alone_full_ms:
        return dict(K=k_min, slowest=None, need_ms=0.0, capped=False, rule="no transfer arm: k_min")
    slowest = max(alone_full_ms, key=lambda a: alone_full_ms[a])
    need = margin * float(alone_full_ms[slowest])
    raw = int(math.ceil(need / tick_p50_ms)) + skip_first if tick_p50_ms > 0 else k_max
    K = max(k_min, min(k_max, raw))
    return dict(K=K, raw=raw, slowest=slowest, need_ms=need, tick_p50_ms=tick_p50_ms, margin=margin, capped=raw > k_max,
                rule="K = clamp(ceil(margin x slowest transfer-alone full_ready_ms / decode-alone tick p50) + skip_first, k_min, k_max)")


def bootstrap_ci(xs: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 stat: Callable[[Sequence[float]], float] = None) -> Tuple[float, float]:
    """Percentile bootstrap CI of `stat` (default: mean) over the given units (the gated trace steps)."""
    v = [float(x) for x in xs if x is not None and x == x]
    if len(v) < 2:
        return float("nan"), float("nan")
    stat = stat or (lambda a: sum(a) / len(a))
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return bs[int(math.floor(alpha / 2 * n_boot))], bs[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]


def paired_by_step(alone: Dict[int, List[float]], conc: Dict[int, List[float]]) -> Dict:
    """Per trace step: p50 decode-alone and beside-transfer ticks, extra ms and %, then the across-step mean with a
    bootstrap CI over steps (the unit of uncertainty) and the pooled p50 / p95."""
    steps = sorted(s for s in conc if conc[s] and alone.get(s))
    per = []
    for s in steps:
        a, c = pctl(alone[s], 50), pctl(conc[s], 50)
        per.append(dict(step=s, alone_p50=a, conc_p50=c, extra_ms=c - a, extra_pct=100.0 * (c / a - 1.0), n_alone=len(alone[s]),
                        n_conc=len(conc[s])))
    pa = [x for s in steps for x in alone[s]]
    pc = [x for s in steps for x in conc[s]]
    ex = [p["extra_pct"] for p in per]
    lo, hi = bootstrap_ci(ex)
    exm = [p["extra_ms"] for p in per]
    lom, him = bootstrap_ci(exm)
    nan = float("nan")
    return dict(per_step=per, n_steps=len(per), alone_p50=pctl(pa, 50), alone_p95=pctl(pa, 95), conc_p50=pctl(pc, 50), conc_p95=pctl(pc, 95),
                extra_pct_mean=(sum(ex) / len(ex) if ex else nan), extra_pct_ci=(lo, hi),
                extra_ms_mean=(sum(exm) / len(exm) if exm else nan), extra_ms_ci=(lom, him), n_alone=len(pa), n_conc=len(pc))


def resident_tps(B: int, tick_ms_: float) -> float:
    return B * 1000.0 / tick_ms_ if tick_ms_ and tick_ms_ == tick_ms_ and tick_ms_ > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------------ capacity
def pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length() if n > 0 else 0


def pinned_plan(B: int, S_cpu: int, layers: int = 32, H: int = 2, D: int = 128, elem: int = 2,
                job_mem_bytes: Optional[int] = None) -> Dict:
    """The host KV cache's pinned footprint: 2 x layers tensors of B x S_cpu x H x D x elem bytes, each rounded up to a
    power of two by the caching host allocator. fits = reserved < job memory (the analytic refusal of job 2175791 at
    B = 345: 64 x 8 GiB = 549.8 GB > 480 GB)."""
    per = B * S_cpu * H * D * elem
    res = pow2(per)
    tot = 2 * layers * res
    return dict(B=B, S_cpu=S_cpu, per_tensor_bytes=per, per_tensor_reserved=res, host_cache_logical=2 * layers * per,
                host_cache_reserved=tot, job_mem_bytes=job_mem_bytes, fits_job_mem=(None if job_mem_bytes is None else tot < job_mem_bytes))


def largest_pow2_batch(S_cpu: int, H: int = 2, D: int = 128, elem: int = 2, block: int = 1 << 32) -> int:
    """The largest B whose per-tensor host bytes stay <= `block` (no rounding up to the next power of two)."""
    return block // (S_cpu * H * D * elem)


GPU_CLASSES = ("GPU_OOM", "GPU_PREDICTED")
HOST_CLASSES = ("HOST_PINNED", "HOST_MEMORY", "HOST_KILLED")


def classify_exception(exc: BaseException) -> str:
    """The stop class of an exception that ended a capacity attempt (driver side; text markers are the driver's own)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "MemoryError":
        return "HOST_MEMORY"
    if "pinned" in msg or "cudahostalloc" in msg or "pin_memory" in msg or "hostalloc" in msg or "cachinghostallocator" in msg:
        return "HOST_PINNED"
    if name == "OutOfMemoryError" or ("out of memory" in msg and ("cuda" in msg or "gpu" in msg)):
        return "GPU_OOM"
    if name == "MemGate" and "free hbm" in msg:
        return "GPU_PREDICTED"
    if name == "MemGate":
        return "GPU_PREDICTED" if "peak reserved" in msg else "OTHER_MEMGATE"
    return "OTHER"


def classify_stop(rc: int, marker_class: Optional[str] = None, elapsed_s: Optional[float] = None, bound_s: Optional[float] = None) -> str:
    """The sbatch-level stop class of a capacity attempt: the driver's own marker wins; else the exit status (124 = the
    stage bound; 137 = SIGKILL: before the bound it is a kill from outside the driver, i.e. the job's memory limit)."""
    if marker_class:
        return marker_class
    if rc == 0:
        return "OK"
    if rc == 124:
        return "TIMEOUT"
    if rc == 137:
        if elapsed_s is not None and bound_s is not None and elapsed_s < bound_s - 90:
            return "HOST_KILLED"
        return "TIMEOUT"
    return "OTHER"


def stop_side(cls: str) -> str:
    """GPU / HOST for a capacity stop; NONE when the batch ran to its end (OK, or FAILED_CHECKS: it fit, some rows failed)."""
    return "GPU" if cls in GPU_CLASSES else ("HOST" if cls in HOST_CLASSES else ("NONE" if cls in ("OK", "FAILED_CHECKS") else "UNKNOWN"))
