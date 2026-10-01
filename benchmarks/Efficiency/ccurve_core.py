"""CACHE-SIZE CURVE: the DECODE-PACED (CC1: "arrival-paced") and EXT-PACED regimes, the cross-capacity certification and
the plot table, pure (CPU-testable; no CUDA). Authorized 2026-10-01 (the user via Codex, 'THE CACHE-SIZE CURVE'); the GPU driver is cache_curve.py, the CPU tests are
retroinfer-eval tests/test_cache_curve.py.

TWO REGIMES (same plans, bytes and destinations for both methods within one capacity C):
  SATURATION     the sustained window of interference_curve.py (a finite train of CV_TRAIN_STEPS trace steps x 32 layer plans,
                 all released at the gate, pumped as fast as the arm can) beside back-to-back resident decode ticks. Label:
                 curve_core.SATURATED_LABEL = 'resource-contention control, NOT live verifier throughput'.
  DECODE-PACED   PACED_LABEL: release k = the 32 layer plans of trace step it+k, released at the START of decode tick k (the
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

RELABEL (Codex's CC1 audit, 2026-10-01): the regime above releases at the start of THAT METHOD's own decode ticks, so its
arrival schedule depends on the method (a method that slows its decode stretches its own release interval) and its offered
rate is per method. It is printed 'decode-paced' everywhere (DECODE_PACED_LABEL; the raw payload keys stay 'paced' so the CC1
payloads and CSVs read unchanged). It is NOT a matched external load.

EXT-PACED (the B64 host-pack diagnostic, CC2; EXT_LABEL): a PRERECORDED external release schedule, fixed per (cell, gated
step) BEFORE any arm runs (ext_schedule): R releases, release k = the 32 natural layer plans of trace step it+k, scheduled at
host time anchor + T0 + k*P; P = ONE value per (cell, step) = that step's decode-alone tick p50 (decode_alone_pre), T0 = a fixed
offset; the same P, T0, release count and bytes for every arm (digest). The releases are issued by a dedicated RELEASE THREAD
(ReleaseThread, a Python thread named 'cc-release'; NOT the decode thread, NOT the transport coordinator): at each scheduled
instant it records an event on its OWN release stream and publishes it to the ReleaseBox; it may sleep (only the decode thread
must never sleep or synchronize). Per release it records the scheduled host time, the actual publication host time, the
lateness (= actual - scheduled) and the publication cost; the release event's GPU time (relative to the window's gate) is read
after the window. The decode thread runs back-to-back resident ticks exactly as in saturation (it publishes nothing). The
transport consumes the releases exactly as in decode-paced (plan-stream wait on the release event, the 32 list D2Hs charged,
per-request host wait, live descriptors, the pipe). ext_metrics: the offered rate = scheduled bytes / (R x P) (ONE value,
identical for every arm), ready / full completion per release and per window, backlog, lateness, handoff, coordinator / pack
duty over the tick window (host_duty), decode duty, bytes completed per second.

HOST-PACK (HOSTPACK_LABEL, arm 'cpu8_hostpack'): the cpu8 code path with the pipe mode 'LP' (pack + index write, NO H2D, NO
scatter): fresh list D2H + live descriptors + CPU pack on 8 cores. 'Done' = packed into the pinned staging slot (host-packed
bytes, NOT delivered). A diagnostic, NOT a transport.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import curve_core as CV

DECODE_PACED_LABEL = ("DECODE-PACED trace replay: releases at each method's OWN decode tick starts: method-dependent arrival "
                      "schedule, NOT a matched external load (one trace step's 32 natural layer plans per tick start; offered = "
                      "natural miss bytes per that method's own tick period; GPU-resident, NOT live verifier throughput)")
PACED_LABEL = DECODE_PACED_LABEL                      # the raw payload's replay_label of the 'paced' regime (CC1 key)
EXT_LABEL = ("EXT-PACED trace replay: a PRERECORDED external release schedule fixed per (cell, step) before any arm runs (release "
             "k = trace step it+k's 32 natural layer plans at host time T0 + k x P, P = that step's decode-alone tick p50), issued by a "
             "dedicated Python release thread; identical for every arm; GPU-resident, NOT live verifier throughput")
HOSTPACK_LABEL = ("host-pack/no-H2D-payload control: fresh list D2H + live descriptors + CPU pack on 8 cores; NO H2D payload, "
                  "NO scatter (a diagnostic, NOT a transport)")
HOSTPACK_DONE_LABEL = "host-packed bytes, NOT delivered"
CAPACITIES = (63, 73, 81, 96, 113, 128)
REGIMES = ("saturation", "paced")
DIAG_REGIMES = ("saturation", "ext-paced")
METHODS = ("cpu8", "w8")
REGIME_NAMES = {"saturation": "saturation", "paced": "decode-paced", "ext-paced": "ext-paced"}
SCHEDULES = {"saturation": "train released at the gate",
             "paced": "decode-paced: each method's OWN tick starts (method-dependent; NOT a matched load)",
             "ext-paced": "external prerecorded schedule (identical for every arm)"}
TABLE_CAVEATS = (
    "(a) useful / during GB/s = bytes / the request-busy union within the tick-window hull, which includes the restore and "
    "receipt gaps, while the slowdown uses >= 95%-covered ticks (saturation) or every steady tick (paced); any ratio of the two "
    "is NOT a cost per delivered byte and NOT a steady-state contention efficiency (no such ratio is reported).",
    "(b) every resident tick (alone AND beside) includes the sanctioned decode-state restore (CounterSnapshot): it copies the "
    "snapshot tensors in HBM between ticks (bytes per tick per cell below; HBM traffic ~2x: read + write). Labelled, part of "
    "every tick.",
    "(c) the H1 / H2 label rules not firing is NOT evidence that host contention is absent.",
    "(d) no additive 'GPU share' is inferred by subtracting slowdowns (overlap and service rates change between arms); the "
    "interpreter lock (GIL) is a HYPOTHESIS until causal evidence; CPU and GPU concurrency effects may coexist.")


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
        self.who: List[str] = []                                         # the publishing thread of every release
        self.closed = False
        self.clock_ns = clock_ns

    def publish(self, ev) -> int:
        with self.cv:
            self.evs.append(ev)
            self.host_ns.append(self.clock_ns())
            self.who.append(threading.current_thread().name)
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


# ------------------------------------------------------------------------------------------- the ext-paced schedule
def ext_schedule(bytes_per_release: Sequence[int], steps: Sequence[int], P_ms: float, T0_ms: float, R: int, ticks: Optional[int] = None) -> Dict:
    """The PRERECORDED external schedule of one (cell, gated step) (module docstring, EXT-PACED): release k at anchor + T0 + k*P
    (integer ns), k = 0..R-1, carrying trace step steps[k]'s bytes. ticks = the decode ticks of an ext window (default: enough
    to cover the last release by one more tick: R + 1 + ceil(T0 / P)). The digest covers R, P, T0, the steps and the bytes, so
    two arms' schedules are identical iff their digests are."""
    P_ms, T0_ms, R = float(P_ms), float(T0_ms), int(R)
    if R < 1 or not (P_ms == P_ms and P_ms > 0) or not (T0_ms == T0_ms and T0_ms >= 0):
        raise ValueError("ext schedule needs R >= 1, P > 0, T0 >= 0 (got R=%s P=%s T0=%s)" % (R, P_ms, T0_ms))
    if len(bytes_per_release) != R or len(steps) != R:
        raise ValueError("ext schedule: %d releases but %d byte counts / %d steps" % (R, len(bytes_per_release), len(steps)))
    P_ns, T0_ns = int(round(P_ms * 1e6)), int(round(T0_ms * 1e6))
    offs = [T0_ns + k * P_ns for k in range(R)]
    tot = int(sum(int(b) for b in bytes_per_release))
    K_t = int(ticks) if ticks is not None else R + 1 + int(math.ceil(T0_ns / P_ns))
    key = json.dumps(dict(R=R, P_ns=P_ns, T0_ns=T0_ns, steps=[int(s) for s in steps], bytes=[int(b) for b in bytes_per_release]), sort_keys=True)
    return dict(R=R, P_ms=P_ns / 1e6, P_ns=P_ns, T0_ms=T0_ns / 1e6, T0_ns=T0_ns, steps=[int(s) for s in steps],
                bytes_per_release=[int(b) for b in bytes_per_release], bytes_total=tot, offsets_ns=offs, scheduled_ms=[o / 1e6 for o in offs],
                ticks=K_t, offered_gbps=tot / (R * (P_ns / 1e6) * 1e6), digest=hashlib.sha256(key.encode()).hexdigest()[:16],
                rule="release k at anchor + T0 + k x P; P = the step's decode-alone tick p50; one schedule per (cell, step), every arm")


class ReleaseThread(threading.Thread):
    """THE RELEASE THREAD of one ext-paced window (module docstring). A Python thread (it competes for the interpreter like any
    other); it sleeps until each scheduled instant (Event.wait: releases the GIL), records an event on its OWN release stream
    and publishes it. late_ns = {k: extra ns}: a DELIBERATE delay of release k (negative control of the lateness record only;
    the scheduled time is unchanged)."""

    NAME = "cc-release"

    def __init__(self, box: ReleaseBox, be, anchor_ns: int, offsets_ns: Sequence[int], stream: str = "release",
                 clock_ns: Callable[[], int] = time.perf_counter_ns, late_ns: Optional[Dict[int, int]] = None):
        super().__init__(daemon=True, name=self.NAME)
        self.box, self.be, self.anchor, self.offs, self.sname = box, be, int(anchor_ns), [int(x) for x in offsets_ns], stream
        self.clock_ns, self.late_ns = clock_ns, dict(late_ns or {})
        self.recs: List[Dict] = []
        self.halted = threading.Event()
        self.error: Optional[str] = None

    def halt(self) -> None:
        self.halted.set()

    def run(self) -> None:
        try:
            for k, off in enumerate(self.offs):
                due = self.anchor + off
                target = due + int(self.late_ns.get(k, 0))
                while True:
                    now = self.clock_ns()
                    if now >= target or self.halted.is_set():
                        break
                    self.halted.wait((target - now) / 1e9)
                if self.halted.is_set():
                    break
                t_iss = self.clock_ns()
                with self.be.stream(self.sname):
                    ev = self.be.event_rec(self.sname)
                j = self.box.publish(ev)
                t_pub = self.box.host_ns[j]
                t_end = self.clock_ns()
                self.recs.append(dict(k=k, index=j, ev=ev, scheduled_ns=off, issue_ns=t_iss - self.anchor, publish_ns=t_pub - self.anchor,
                                      lateness_ns=t_pub - due, publish_cost_ns=t_end - t_iss, injected_delay_ns=int(self.late_ns.get(k, 0))))
        except BaseException:
            self.error = traceback.format_exc()[-3000:]


def lateness_stats(lateness_ms: Sequence[float], tol_ms: float) -> Dict:
    v = [float(x) for x in lateness_ms if x is not None and x == x]
    return dict(n=len(v), p50_ms=CV.pctl(v, 50), p95_ms=CV.pctl(v, 95), max_ms=(max(v) if v else float("nan")),
                late=sum(1 for x in v if x > tol_ms), tol_ms=tol_ms)


def schedule_gate(sched: Dict, ref_digest: Optional[str], rel: Sequence[Dict], published: int, who: Sequence[str], thread_error) -> Dict:
    """The ext window's SCHEDULE gate: its schedule is the step's fixed one (digest), every release was issued (R published by
    the release thread), lateness recorded for each, no release-thread error."""
    R = int(sched["R"])
    why = []
    if ref_digest is None or sched.get("digest") != ref_digest:
        why.append("schedule digest %s != the step's %s" % (sched.get("digest"), ref_digest))
    if published != R or len(rel) != R:
        why.append("%d of %d releases issued (%d records)" % (published, R, len(rel)))
    if any(r.get("lateness_ns") is None for r in rel):
        why.append("lateness not recorded")
    bad_who = sorted({w for w in who if w != ReleaseThread.NAME})
    if bad_who:
        why.append("published by %s (not the release thread)" % bad_who)
    if thread_error:
        why.append("release thread error")
    return dict(ok=not why, why=why, digest=sched.get("digest"), R=R, published=published)


# ---------------------------------------------------------------------------------------------- diagnostic metrics
def host_duty(iv: Dict[str, Sequence[Tuple[float, float]]], w0: float, w1: float) -> Dict:
    """The coordinator's own host work inside the tick window [w0, w1] (ms on the gate clock): iv = {'desc': descriptor builds,
    'pack': CPU packs, 'api': the pipe's enqueue calls, 'release': the per-release plan-stream wait + list-D2H issue}. Waits
    (release publication, list arrival, staging-slot reuse) are NOT busy. Intervals come from host durations ending at aux-stream
    markers (an upper bound of the host instant), so they are approximate on the GPU clock."""
    nan = float("nan")
    W = (w1 - w0) if (w0 == w0 and w1 == w1) else nan
    allv = [x for k in ("desc", "pack", "api", "release") for x in iv.get(k, ())]
    out = dict(window_ms=W, busy_total_ms=CV.union_len(allv), pack_total_ms=CV.union_len(iv.get("pack", ())))
    for k in ("desc", "pack", "api", "release"):
        out["%s_total_ms" % k] = CV.union_len(iv.get(k, ()))
    if W == W and W > 0:
        out["coord_duty"] = CV.intersect_len(allv, w0, w1) / W
        out["pack_duty"] = CV.intersect_len(iv.get("pack", ()), w0, w1) / W
    else:
        out["coord_duty"] = out["pack_duty"] = nan
    return out


def handoff_stats(reqs: Sequence[Dict]) -> Dict:
    """Per request: the coordinator's handoff (host wait for its list + live descriptor build) and the list D2H span (GPU,
    consecutive plan-stream events; ext windows only)."""
    ho = [float(q["wait_ms"]) + float(q["desc_ms"]) for q in reqs if q.get("wait_ms") is not None and q.get("desc_ms") is not None]
    d2h = [q["d2h_ms"] for q in reqs if q.get("d2h_ms") is not None]
    ds = [q["desc_ms"] for q in reqs if q.get("desc_ms") is not None]
    wt = [q["wait_ms"] for q in reqs if q.get("wait_ms") is not None]
    return dict(handoff_p50_ms=CV.pctl(ho, 50), handoff_p95_ms=CV.pctl(ho, 95), handoff_total_ms=sum(ho), d2h_p50_ms=CV.pctl(d2h, 50),
                d2h_p95_ms=CV.pctl(d2h, 95), desc_p50_ms=CV.pctl(ds, 50), desc_p95_ms=CV.pctl(ds, 95), wait_p50_ms=CV.pctl(wt, 50),
                wait_p95_ms=CV.pctl(wt, 95), n=len(ho))


def diag_metrics(reqs: Sequence[Dict], ticks: Sequence[Tuple[float, float]], iv: Optional[Dict] = None, rel_ms: Optional[Sequence[float]] = None,
                 rel: Optional[Sequence[Dict]] = None, sched: Optional[Dict] = None, done_label: str = "", late_tol_ms: float = 1.0) -> Dict:
    """The diagnostic numbers of ONE window (saturation: rel_ms None, every request released at the gate = 0 ms; ext-paced: the
    release event times + the release thread's records + the schedule). 'done' = each request's g1 (scattered for cpu8, packed
    into staging for the host-pack control: done_label). No subtraction between arms happens here or anywhere."""
    nan = float("nan")
    rq = [q for q in reqs if q.get("g1") is not None]
    w0, w1 = (ticks[0][0], ticks[-1][1]) if ticks else (nan, nan)
    W = (w1 - w0) if ticks else nan
    if rel_ms is not None:
        n_rel = len(rel_ms)
        rq = [q for q in rq if q.get("release") is not None and 0 <= q["release"] < n_rel]
        t_of = lambda q: rel_ms[q["release"]]
        first = rel_ms[0] if n_rel else nan
    else:
        t_of = lambda q: 0.0
        first = 0.0
    ready = [q["g1"] - t_of(q) for q in rq]
    by: Dict[int, List[Dict]] = {}
    for q in rq:
        by.setdefault(int(q.get("release") or 0) if rel_ms is not None else 0, []).append(q)
    full_rel = [max(q["g1"] for q in v) - t_of(v[0]) for _, v in sorted(by.items())]
    last = max((q["g1"] for q in rq), default=nan)
    done_in = sum(int(q["useful"]) for q in rq if ticks and w0 <= q["g1"] <= w1)
    out = dict(done_label=done_label, requests=len(rq), ready_p50_ms=CV.pctl(ready, 50), ready_p95_ms=CV.pctl(ready, 95),
               full_release_p50_ms=CV.pctl(full_rel, 50), full_release_p95_ms=CV.pctl(full_rel, 95),
               full_release_max_ms=(max(full_rel) if full_rel else nan), full_window_ms=(last - first if last == last and first == first else nan),
               decode_duty=(sum(b - a for a, b in ticks) / W if ticks and W > 0 else nan), done_bytes_in_window=done_in,
               done_gbps_in_window=(done_in / (W * 1e6) if ticks and W > 0 else nan),
               done_gbps_overall=(sum(int(q["useful"]) for q in rq) / ((last - first) * 1e6) if last == last and first == first and last > first else nan))
    out.update(handoff_stats(rq))
    out.update(host_duty(iv or {}, w0, w1))
    if rel_ms is not None:
        pm = paced_metrics(rel_ms, rq, ticks)
        out.update(backlog_end_bytes=pm["backlog_end_bytes"], backlog_slope_bytes_per_release=pm["backlog_slope_bytes_per_release"],
                   overloaded=pm["overloaded"], drain_ms=pm["drain_ms"], outstanding_bytes_at_window_end=pm["outstanding_bytes_at_window_end"],
                   backlog_bytes=pm["backlog_bytes"], offered_actual_gbps=pm["offered_gbps"])
    else:
        out.update(backlog_end_bytes=nan, backlog_slope_bytes_per_release=nan, overloaded=None, drain_ms=nan, outstanding_bytes_at_window_end=nan)
    if rel is not None:
        lat = [r["lateness_ns"] / 1e6 for r in rel if r.get("lateness_ns") is not None]
        ls = lateness_stats(lat, late_tol_ms)
        out.update(lateness_p50_ms=ls["p50_ms"], lateness_p95_ms=ls["p95_ms"], lateness_max_ms=ls["max_ms"], late_releases=ls["late"],
                   late_tol_ms=late_tol_ms, publish_cost_p50_ms=CV.pctl([r["publish_cost_ns"] / 1e6 for r in rel], 50),
                   publish_cost_max_ms=max((r["publish_cost_ns"] / 1e6 for r in rel), default=nan))
    else:
        out.update(lateness_p50_ms=nan, lateness_p95_ms=nan, lateness_max_ms=nan, late_releases=None)
    if sched is not None:
        out.update(offered_gbps=sched["offered_gbps"], P_ms=sched["P_ms"], T0_ms=sched["T0_ms"], schedule_digest=sched["digest"],
                   scheduled_bytes=sched["bytes_total"])
    else:
        out.update(offered_gbps=nan)
    return out


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
                "device_used_gb", "pinned_host_gb", "note", "schedule", "label", "restore_gb_per_tick")
DIAG_FIELDS = ("batch", "C", "method", "regime", "status", "label", "done", "n_steps", "n_windows", "slowdown_pct", "slowdown_ci_lo", "slowdown_ci_hi",
               "offered_gbps", "P_ms", "T0_ms", "ready_p50_ms", "ready_p95_ms", "full_release_p50_ms", "full_release_p95_ms", "full_window_ms",
               "backlog_end_bytes", "backlog_slope_bytes_per_release", "overloaded_windows", "drain_ms", "lateness_p50_ms", "lateness_p95_ms",
               "lateness_max_ms", "late_releases", "handoff_p50_ms", "handoff_p95_ms", "d2h_p50_ms", "d2h_p95_ms", "coord_duty", "pack_duty",
               "decode_duty", "done_gbps_in_window", "schedule_identical", "note")


def method_label(method: str) -> str:
    return HOSTPACK_LABEL if method.endswith("hostpack") else ""


def blank_point(B: int, C: int, method: str, regime: str, status: str, note: str = "") -> Dict:
    nan = float("nan")
    d = {k: nan for k in POINT_FIELDS}
    d.update(batch=int(B), C=int(C), P=int(C) - 63, method=method, regime=regime, status=status, certified=False, n_steps=0, note=note,
             overloaded_windows=0, schedule=SCHEDULES.get(regime, regime), label=method_label(method))
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
