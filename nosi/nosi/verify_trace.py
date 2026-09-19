"""CUDA-event brackets around one Path 1 verify call (retroinfer-eval fork;
the pilot's ``cost`` mode, spec 2026-09-19-multiposition-verify-path1.md
section 4: the measured verify cost the equal-memory frontier prices with a
formula today, scripts/nosi_equal_memory_frontier.py::spec_round).

Off unless a driver binds ``TRACE`` (the transfer_trace.TRACE pattern): every
call site in nosa_llama.py is ``if _vt is not None: ...``, so the gate modes
of the pilot execute the verify text without a single event. Events change
no float: the numerics of the call are the same with and without them.

The four per-layer marks, in the order verify_forward records them:
  score_begin   before the per-position scoring loop (U x stage 1 + the
                captured pooling/top-k graph + the one-token cis/compress-k
                updates: the decode's scoring kernels, once per position)
  fetch_begin   before cache_engine.verify_round_update_kv (the U tail writes,
                the mirror, the union build with its one host sync, the two
                Triton gathers of the round region)
  fetch_end     after it
  attn_end      after verify_step.verify_attention (the two union-wide value
                copies, two host-sync guards, the two varlen calls, the quotient)
Per call: ``score_ms``, ``fetch_ms``, ``attn_ms`` are sums over the layers of
those three windows; ``rest_ms`` = the whole call (start/end events in
verify_inference) minus the three: qkv/wo/gate-up/down GEMMs at B*U tokens,
rope, the norms, the embedding and the lm_head. A host sync inside a window
shows up as GPU idle time inside that window, which is the cost the caller
pays today (verify_step.py names the two guards that sync).

``split_call`` is the pure arithmetic and is CPU-tested
(retroinfer-eval tests/test_nosi_verify_u.py).
"""
from __future__ import annotations

import torch

TRACE = None   # bound by the pilot's cost mode: VerifyTrace(num_layers)

LAYER_MARKS = ("score_begin", "fetch_begin", "fetch_end", "attn_end")


def _ev():
    return torch.cuda.Event(enable_timing=True)


def split_call(total_ms: float, per_layer) -> dict:
    """per_layer: iterable of (score_ms, fetch_ms, attn_ms) per recorded layer.
    Returns the call's decomposition; ``rest_ms`` is what the three windows
    do not cover. Every input must be finite and >= 0 (a negative window
    means the marks were recorded out of order: refuse, do not clamp)."""
    score = fetch = attn = 0.0
    n = 0
    for s, f, a in per_layer:
        for name, v in (("score", s), ("fetch", f), ("attn", a)):
            if not (v == v) or v < 0:
                raise ValueError("split_call: layer %d %s window %r is negative or NaN (marks out of order?)" % (n, name, v))
        score += float(s)
        fetch += float(f)
        attn += float(a)
        n += 1
    if not (total_ms == total_ms) or total_ms < 0:
        raise ValueError("split_call: total %r" % total_ms)
    rest = float(total_ms) - score - fetch - attn
    return dict(total_ms=float(total_ms), score_ms=score, fetch_ms=fetch, attn_ms=attn, rest_ms=rest, layers=n)


class VerifyTrace:
    """One per process. ``begin_call``/``end_call`` from verify_inference,
    ``begin_layer``/``rec`` from verify_forward; ``harvest`` synchronises once
    and returns one dict per call, in call order, with the caller's ``label``."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.calls: list = []
        self.cur = None
        self.layer = None
        self.next_label = None

    def label(self, label):
        """The label the NEXT begin_call carries (e.g. (U, step))."""
        self.next_label = label

    def begin_call(self):
        c = dict(label=self.next_label, start=_ev(), end=_ev(), layers=[None] * self.num_layers)
        self.next_label = None
        c["start"].record()
        self.calls.append(c)
        self.cur = c
        self.layer = None

    def begin_layer(self, layer_idx: int):
        if self.cur is None:
            return
        self.layer = layer_idx
        self.cur["layers"][layer_idx] = {k: _ev() for k in LAYER_MARKS}

    def rec(self, key: str):
        if self.cur is None or self.layer is None:
            return
        self.cur["layers"][self.layer][key].record()

    def end_call(self):
        if self.cur is not None:
            self.cur["end"].record()
        self.cur = None
        self.layer = None

    def abort_call(self):
        """The call raised (RoundOverflow) before end_call: drop it, so harvest
        never reads an event that was created and not recorded."""
        if self.cur is not None and self.calls and self.calls[-1] is self.cur:
            self.calls.pop()
        self.cur = None
        self.layer = None

    def harvest(self) -> list:
        torch.cuda.synchronize()
        rows = []
        for c in self.calls:
            per_layer = []
            for d in c["layers"]:
                if d is None:
                    continue
                per_layer.append((d["score_begin"].elapsed_time(d["fetch_begin"]),
                                  d["fetch_begin"].elapsed_time(d["fetch_end"]),
                                  d["fetch_end"].elapsed_time(d["attn_end"])))
            row = split_call(c["start"].elapsed_time(c["end"]), per_layer)
            row["label"] = c["label"]
            rows.append(row)
        return rows

    def reset(self):
        self.calls = []
        self.cur = None
        self.layer = None
