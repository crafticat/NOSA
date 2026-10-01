"""CACHE-SIZE CURVE: the ARRIVAL-PACED regime, the cross-capacity certification and the plot table, pure (CPU-testable; no
CUDA). Authorized 2026-10-01 (the user via Codex, 'THE CACHE-SIZE CURVE'); the GPU driver is cache_curve.py, the CPU tests are
retroinfer-eval tests/test_cache_curve.py.

TWO REGIMES (same plans, bytes and destinations for both methods within one capacity C):
  SATURATION     the sustained window of interference_curve.py (a finite train of CV_TRAIN_STEPS trace steps x 32 layer plans,
                 all released at the gate, pumped as fast as the arm can) beside back-to-back resident decode ticks. Label:
                 curve_core.SATURATED_LABEL = 'resource-contention control, NOT live verifier throughput'.
  ARRIVAL-PACED  PACED_LABEL: release k = the 32 layer plans of trace step it+k, released at the START of decode tick k (the
                 tick's own t0 event, recorded on the decode stream after the restore), for k = 0..K_p-1. The decode thread only
                 records the event and publishes it (ReleaseBox.publish: a lock + notify, never a wait); the transport thread
                 host-waits for the PUBLICATION (the decode thread has enqueued tick k) and makes the GPU wait for the EVENT, so no
                 copy of release k can start before the GPU reaches tick k:
                   w8    the pump (pump_paced): bounded queue as curve_core.pump, + per release one stream wait on its event;
                   cpu8  the coordinator: per release one stream wait on the plan stream + the 32 list D2Hs (charged), then per
                         request host-wait for its list, descriptors built live, pack + H2D + scatter.
                 Offered rate = the step's natural miss bytes per release interval (the decode tick period, restore included).
                 Nothing sleeps; no barrier holds the decode; the release and hand-off costs are recorded per request
                 (release_wait_ns, issue marker, list events, desc ms). BACKLOG (paced_metrics): the bytes released before
                 release j and not complete at release j; queue growth = the least-squares slope of that backlog over j; the
                 window is OVERLOADED when the backlog at the last release exceeds one step of offered bytes; drain = the last
                 completion after the window's end; outstanding = bytes not complete at the window's end.

CROSS-C CERTIFICATION (cross_c_check): pool invisibility (cache_engine.py:37-57, the attended views) means the SAME token /
selection trajectory at every C: the capture logits (VA.sha per natural step) and the golden reference / advance logits
(sha256 per gated step) at each C must equal C63's. A C that differs, or has no C63 to compare with, is NOT certified and its
cells are reported MISSING; C63 plans are never substituted.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import curve_core as CV

PACED_LABEL = ("ARRIVAL-PACED trace replay: one trace step's 32 natural layer plans released per resident decode tick (at the "
               "tick's start event); offered = natural miss bytes per tick period; GPU-resident, NOT live verifier throughput")
CAPACITIES = (63, 73, 81, 96, 113, 128)
REGIMES = ("saturation", "paced")
METHODS = ("cpu8", "w8")


# ------------------------------------------------------------------------------------------------------- releases
class ReleaseClosed(RuntimeError):
    """The decode thread ended (or failed) before publishing the release a transport thread waits for."""


class ReleaseBox:
    """Release events published by the decode thread, consumed in order by ONE transport thread. publish never blocks on the
    consumer (a lock held for an append + notify); wait blocks the consumer until release j is published or the box closes."""

    def __init__(self, clock_ns: Callable[[], int] = time.perf_counter_ns):
        self.cv = threading.Condition()
        self.evs: List = []
        self.host_ns: List[int] = []
        self.closed = False
        self.clock_ns = clock_ns

    def publish(self, ev) -> int:
        with self.cv:
            self.evs.append(ev)
            self.host_ns.append(self.clock_ns())
            self.cv.notify_all()
            return len(self.evs) - 1

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def wait(self, j: int):
        """(event of release j, host ns waited)."""
        t = self.clock_ns()
        with self.cv:
            while len(self.evs) <= j:
                if self.closed:
                    raise ReleaseClosed("release %d never published (%d published, box closed)" % (j, len(self.evs)))
                self.cv.wait()
            return self.evs[j], self.clock_ns() - t


def paced_train(groups_of: Callable[[int, int], int], it: int, k_p: int, n_layers: int, n_regions: int = 1,
                wire_extra_per_group: int = 0) -> List[CV.Req]:
    """The paced train: trace steps it .. it+k_p-1, release j = step it+j (curve_core.build_train order and byte charging)."""
    return CV.build_train(groups_of, list(range(it, it + k_p)), n_layers, n_regions, wire_extra_per_group)


def release_index(r: CV.Req, it: int) -> int:
    return int(r.step) - int(it)


def pump_paced(be, train: Sequence[CV.Req], launch: Callable[[CV.Req, str], None], gate, queue: int, box: ReleaseBox, it: int,
               stream_of: Optional[Callable[[CV.Req], str]] = None, deps: bool = True,
               clock_ns: Callable[[], int] = time.perf_counter_ns) -> List[Dict]:
    """curve_core.pump with releases (module docstring). Requests go in train order (release order); before the first request
    of release j the pump host-waits for release j's PUBLICATION and every stream that issues a request of release j first
    stream-waits for its EVENT."""
    if queue < 1:
        raise ValueError("queue must be >= 1")
    stream_of = stream_of or (lambda r: "side")
    from collections import deque
    outstanding = deque()
    last_on_region: Dict[int, Tuple[int, object]] = {}
    gated_streams, waited = set(), {}
    rel_seen: Dict[int, Tuple[object, int]] = {}
    recs = []
    for r in train:
        j = release_index(r, it)
        rec = dict(i=r.i, step=r.step, layer=r.layer, groups=r.groups, useful=r.useful, wire=r.wire, region=r.region, release=j)
        rec["release_wait_ns"] = 0
        if j not in rel_seen:
            ev, wns = box.wait(j)
            rel_seen[j] = (ev, wns)
            rec["release_wait_ns"] = wns
        ev_j = rel_seen[j][0]
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
            if waited.get(s) != j:
                be.stream_wait(s, ev_j)
                waited[s] = j
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


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else float("nan")


def paced_metrics(rel_ms: Sequence[float], reqs: Sequence[Dict], ticks: Sequence[Tuple[float, float]] = ()) -> Dict:
    """Backlog / queue growth / overload / drain of one paced window (module docstring). rel_ms[j] = release j on the gate
    clock (ms); reqs carry release (index), useful, g1 (ms from the gate); ticks = the window's decode spans."""
    nan = float("nan")
    rq = [q for q in reqs if q.get("g1") is not None and q.get("release") is not None]
    n_rel = len(rel_ms)
    step_bytes = [0] * n_rel
    for q in reqs:
        j = q.get("release")
        if j is not None and 0 <= j < n_rel:
            step_bytes[j] += int(q["useful"])
    backlog, backlog_req = [], []
    for j in range(n_rel):
        t = rel_ms[j]
        out = [q for q in rq if q["release"] < j and q["g1"] > t]
        backlog.append(sum(int(q["useful"]) for q in out))
        backlog_req.append(len(out))
    intervals = [rel_ms[j + 1] - rel_ms[j] for j in range(n_rel - 1)]
    period = CV.pctl(intervals, 50)
    mean_step = (sum(step_bytes) / n_rel) if n_rel else nan
    offered = (mean_step / (period * 1e6)) if period == period and period > 0 else nan
    ready = [q["g1"] - rel_ms[q["release"]] for q in rq if 0 <= q["release"] < n_rel]
    by_layer: Dict[int, List[float]] = {}
    for q in rq:
        if 0 <= q["release"] < n_rel and q.get("layer") is not None:
            by_layer.setdefault(int(q["layer"]), []).append(q["g1"] - rel_ms[q["release"]])
    w1 = ticks[-1][1] if ticks else (max((q["g1"] for q in rq), default=nan))
    w0 = ticks[0][0] if ticks else (rel_ms[0] if n_rel else nan)
    last = max((q["g1"] for q in rq), default=nan)
    outstanding = sum(int(q["useful"]) for q in rq if q["g1"] > w1)
    released_in = sum(step_bytes)
    slope = _slope(list(range(n_rel)), backlog)
    over = bool(n_rel and backlog and mean_step == mean_step and backlog[-1] > mean_step)
    return dict(releases=n_rel, release_ms=list(rel_ms), release_interval_p50_ms=period, step_bytes=step_bytes, mean_step_bytes=mean_step,
                offered_gbps=offered, offered_in_window_gbps=(released_in / ((w1 - w0) * 1e6) if (w1 == w1 and w0 == w0 and w1 > w0) else nan),
                backlog_bytes=backlog, backlog_requests=backlog_req, backlog_slope_bytes_per_release=slope,
                backlog_end_bytes=(backlog[-1] if backlog else 0), overloaded=over, drain_ms=(max(0.0, last - w1) if last == last and w1 == w1 else nan),
                outstanding_bytes_at_window_end=outstanding, ready_p50_ms=CV.pctl(ready, 50), ready_p95_ms=CV.pctl(ready, 95),
                ready_max_ms=(max(ready) if ready else nan), ready_by_layer_p50_ms={l: CV.pctl(v, 50) for l, v in sorted(by_layer.items())},
                all_released=all(0 <= q.get("release", -1) < n_rel for q in reqs) and len(rq) == len(reqs),
                release_wait_ms_total=sum(q.get("release_wait_ns", 0) for q in reqs) / 1e6)


# ------------------------------------------------------------------------------------------------ cross-C check
def cross_c_check(cells: Dict[int, Dict], ref_C: int = 63) -> Dict:
    """cells[C] = dict(capture_sha=[per natural step], ref_sha={step: sha}, adv_sha={step: sha}) (any may be missing).
    Returns {C: dict(certified, why, compared)} -- certified only when every recorded value equals ref_C's on the common steps
    and at least one capture step and one gated step were compared. ref_C itself is certified when it has records."""
    ref = cells.get(ref_C)
    out = {}
    for C, c in sorted(cells.items()):
        why = []
        if ref is None:
            out[C] = dict(certified=False, why=["no C%d record to compare with" % ref_C], compared=dict(capture=0, ref=0, adv=0))
            continue
        a, b = list(c.get("capture_sha") or []), list(ref.get("capture_sha") or [])
        n = min(len(a), len(b))
        cap_bad = [i for i in range(n) if str(a[i]) != str(b[i])]
        if n == 0:
            why.append("no capture steps to compare")
        if cap_bad:
            why.append("capture logits differ at natural steps %s" % cap_bad[:8])
        comp = dict(capture=n)
        for key in ("ref_sha", "adv_sha"):
            x, y = {int(k): v for k, v in (c.get(key) or {}).items()}, {int(k): v for k, v in (ref.get(key) or {}).items()}
            common = sorted(set(x) & set(y))
            bad = [s for s in common if x[s] != y[s]]
            comp[key.split("_")[0]] = len(common)
            if not common:
                why.append("no gated %s to compare" % key)
            if bad:
                why.append("%s differs at gated steps %s" % (key, bad[:8]))
        out[C] = dict(certified=not why, why=why, compared=comp)
    return out


# ------------------------------------------------------------------------------------------------------- plot table
POINT_FIELDS = ("batch", "C", "P", "method", "regime", "status", "certified", "n_steps", "slowdown_pct", "slowdown_ci_lo", "slowdown_ci_hi",
                "extra_ms", "extra_ms_ci_lo", "extra_ms_ci_hi", "decode_alone_p50_ms", "decode_alone_p95_ms", "tick_p50_ms", "tick_p95_ms",
                "useful_gbps", "useful_gbps_ci_lo", "useful_gbps_ci_hi", "window_gbps", "alone_gbps", "offered_gbps", "useful_bytes",
                "wire_bytes", "full_ready_ms", "overlap_frac", "backlog_slope_bytes_per_release", "backlog_end_bytes", "overloaded_windows",
                "drain_ms", "ready_p95_ms", "h2d_per_stream_step", "pool_hits_per_stream_step", "peak_allocated_gb", "peak_reserved_gb",
                "device_used_gb", "pinned_host_gb", "note")


def blank_point(B: int, C: int, method: str, regime: str, status: str, note: str = "") -> Dict:
    nan = float("nan")
    d = {k: nan for k in POINT_FIELDS}
    d.update(batch=int(B), C=int(C), P=int(C) - 63, method=method, regime=regime, status=status, certified=False, n_steps=0, note=note,
             overloaded_windows=0)
    return d


def step_ci(per_step: Sequence[float]) -> Tuple[float, float, float]:
    v = [float(x) for x in per_step if x is not None and x == x]
    if not v:
        return float("nan"), float("nan"), float("nan")
    lo, hi = CV.bootstrap_ci(v)
    return sum(v) / len(v), lo, hi


def fmt(x) -> str:
    if isinstance(x, float):
        return "" if math.isnan(x) else ("%.6g" % x)
    return str(x)
