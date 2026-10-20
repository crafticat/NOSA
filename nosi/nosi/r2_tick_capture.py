"""R2 step 7, W4 of docs/R2_TICK_CAPTURE_PLAN.md: the CAPTURED fixed-capacity tick. r2_tick_fixed.FixedTickEngine's
region (_forward_fixed) recorded in one CUDA graph per capacity (G_d (B, 1), G_v (B (1 + v_cap), v_cap); cap_mode
"max": G_v only), the graphs sharing ONE memory pool (their replays never overlap), and replayed at every later tick of
that capacity.

THE ORDER per capacity: its first `warm` ticks run the region EAGERLY (they compile every Triton kernel with the
capture's specialisation and create the G-b PaddedLinear buffers; they are real ticks); the next tick CAPTURES the
region (capture executes nothing) and replays it; every later tick replays. A replay reads the static row table that
the driver copied in before it (FixedTickEngine.upload, outside the graph) and writes the token log through the
table's logslot column, the twins, the stores and the per-layer counters in place; the logits and the argmax are the
graph's static outputs (overwritten by the next replay of that capacity: run_rows returns views, the caller copies
what it keeps).

state_pointers / pointer_diff: the data_ptr of every state tensor (twins, stores, token log, flags, every capacity's
buffers, the PaddedLinear buffers, the graphs' static outputs); the gate requires the same pointers after the first
capture and after the last replay (PTR). graph_factory (tests) replaces torch.cuda.CUDAGraph: f(engine, cb) -> (graph
with .replay(), static logits, static argmax).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from .r2_tick import TickError
from .r2_tick_fixed import FixedTickEngine

TWIN_PTR = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "overflow", "comp_k", "ring", "total_cis", "ccis", "cu",
            "_win")
CB_PTR = ("flat", "cu_q", "cu_k", "keys", "cis", "ctr", "pq", "pc", "ffn")


class CapturedTickEngine(FixedTickEngine):
    """FixedTickEngine whose region is replayed from CUDA graphs (module docstring). Extra arguments: warm (eager ticks
    per capacity before its capture, >= 1), graph_factory (tests)."""

    def __init__(self, *args, warm: int = 1, graph_factory: Optional[Callable] = None, **kw):
        super().__init__(*args, **kw)
        if isinstance(warm, bool) or int(warm) != warm or int(warm) < 1:
            raise TickError("warm must be an int >= 1 (an eager tick per capacity before its capture)")
        self.warm = int(warm)
        self.graph_factory = graph_factory
        self.graphs: Dict[tuple, dict] = {}
        # the graphs' shared memory pool. NOT `self.pool`: TickEngine.pool is the pooling op (r2_tick.py:266) that the
        # region calls (r2_tick_fixed.py _layer_fixed); be8bce7 named this attribute `pool` and overwrote it with None
        # (CPU job 2190947: G1-G5 "NoneType is not callable"). Test G6 pins the pooling op after construction and clone.
        self.graph_pool = None
        self.stats.update(captures=0, replays=0, eager_region=0, capture_ms=0.0)

    def clone(self, ops=None, **kw):
        kw.setdefault("warm", self.warm)
        kw.setdefault("graph_factory", self.graph_factory)
        return super().clone(ops, **kw)

    def forward(self, cb):
        key = (cb.R_cap, cb.T_cap)
        g = self.graphs.get(key)
        if g is None:
            if cb.uses < self.warm:
                self.stats["eager_region"] += 1
                return self._forward_fixed(cb)
            g = self._capture(cb, key)
        g["graph"].replay()
        self.stats["replays"] += 1
        return g["logits"], g["arg"]

    def _capture(self, cb, key) -> dict:
        t0 = time.perf_counter()
        if self.graph_factory is not None:
            graph, logits, arg = self.graph_factory(self, cb)
        else:
            import torch
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            if self.graph_pool is None:
                self.graph_pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                logits, arg = self._forward_fixed(cb)
            torch.cuda.synchronize()
        self.stats["captures"] += 1
        self.stats["capture_ms"] += 1e3 * (time.perf_counter() - t0)
        g = self.graphs[key] = dict(graph=graph, logits=logits, arg=arg)
        return g


def state_pointers(eng) -> Dict[str, int]:
    """{name: data_ptr} of every state tensor of a fixed / captured tick engine (module docstring)."""
    out = {}
    for l, t in enumerate(eng.twins):
        for k in TWIN_PTR:
            out["twin%d.%s" % (l, k)] = getattr(t, k).data_ptr()
        for name, lst in (("store_k", eng.store_k), ("store_v", eng.store_v), ("store_bias", eng.store_bias)):
            out["%s%d" % (name, l)] = lst[l].data_ptr()
    out["tok_log"] = eng.tok_log.data_ptr()
    out["err_dev"] = eng.err_dev.data_ptr()
    for key in sorted(eng.bufs):
        cb = eng.bufs[key]
        for f in CB_PTR:
            out["cap%s.%s" % (key, f)] = getattr(cb, f).data_ptr()
    padded = getattr(eng.ops, "padded", None)
    if padded is not None:
        for k, buf in sorted(padded._buf.items(), key=repr):
            out["padded%s" % (k,)] = buf.data_ptr()
    for key in sorted(getattr(eng, "graphs", {})):
        g = eng.graphs[key]
        out["graph%s.logits" % (key,)] = g["logits"].data_ptr()
        out["graph%s.arg" % (key,)] = g["arg"].data_ptr()
    return out


def pointer_diff(before: Dict[str, int], after: Dict[str, int]):
    """The names whose data_ptr changed, appeared or disappeared."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
