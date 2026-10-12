"""The stream-backend interface of the dispatcher: EXACTLY the shape of benchmarks/Efficiency/cpupack_core.py CpuBackend
(:367-438, the deferred CPU model) and CudaBackend (:441-493, the GPU mapping), so that M2 binds the GPU without a new
interface and cpupack_core.Pipe (:522-628) runs on the same backend object as the dispatcher.

Methods used:  stream(name) (context manager), event_rec(name), host_wait(ev), stream_wait(name, ev),
               memcpy(dst, src, name), index_copy(dst2d, idx, src, name), synchronize(order), marker(), sleep(ms, name).
Streams used:  "copy", "scatter" (the S0 Pipe), "compute" (the joint-call reads), "d2h" (history writes, A5).

Events: CpuBackend events are cpupack_core._Ev objects with a .done flag (set only when the deferred stream reaches the
record); CudaBackend events are torch.cuda.Event objects with .query(). ev_done() is the NON-BLOCKING poll of either
(no host sync). A host_wait forces completion (on the CpuBackend it runs the stream up to the event).
"""
from __future__ import annotations

from typing import Any

REQUIRED = ("stream", "event_rec", "host_wait", "stream_wait", "memcpy", "index_copy", "synchronize")
STREAMS = ("copy", "scatter", "compute", "d2h")


def check_backend(be: Any) -> None:
    missing = [m for m in REQUIRED if not callable(getattr(be, m, None))]
    if missing:
        raise TypeError("backend %r lacks %s (the cpupack_core CpuBackend / CudaBackend shape)" % (type(be).__name__, missing))


def ev_done(ev: Any) -> bool:
    """Non-blocking poll. None counts as complete (no dependency)."""
    if ev is None:
        return True
    q = getattr(ev, "query", None)
    if callable(q):
        return bool(q())
    return bool(getattr(ev, "done"))
