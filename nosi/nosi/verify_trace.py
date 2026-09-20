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
# The speculative loop's finer marks (2026-09-20, spec_loop_pilot.py): between
# fetch_end and attn_end, verify_step.verify_attention reports the end of the
# two union-wide value copies and of each of the two varlen calls, so the
# attention bracket splits into copies / call 1 / call 2 / quotient. A mark
# that a call site never records is simply absent from that layer's row.
LAYER_MARKS_V2 = ("score_begin", "fetch_begin", "tail_end", "plan_end", "fetch_end",
                  "copies_end", "attn1_end", "attn2_end", "attn_end")
# (name, from, to). Inside the v1 fetch bracket the loop's engine methods record
# tail_end (after the tail writes: torch copies) and plan_end (after the
# union / availability plan: integer torch ops, plus the one host-sync guard
# of a verify round), so gather_ms is the two Triton gathers alone (the
# PCIe-bound part of a verify round; on a draft step it is the side-stream
# launch plus the bias scratch build, the gathers themselves run elsewhere).
WINDOWS_V2 = (("score_ms", "score_begin", "fetch_begin"), ("fetch_ms", "fetch_begin", "fetch_end"),
              ("tail_ms", "fetch_begin", "tail_end"), ("plan_ms", "tail_end", "plan_end"),
              ("gather_ms", "plan_end", "fetch_end"),
              ("copies_ms", "fetch_end", "copies_end"), ("attn1_ms", "copies_end", "attn1_end"),
              ("attn2_ms", "attn1_end", "attn2_end"), ("quot_ms", "attn2_end", "attn_end"),
              ("attn_ms", "fetch_end", "attn_end"))
# WHAT EACH WINDOW CONTAINS (for the kernel-vs-host table of the report):
#   score   stage-1 kernel + the captured pooling/top-k graph (kernels) + the
#           per-token cis / compress-k table updates (small torch ops)
#   tail    write_tail / draft_tail_write: strided copies (memcpy kernels), a
#           host write-back on a fill
#   plan    plan_round / plan_draft: integer torch ops (~40 small kernels) and,
#           in a verify round, ONE host sync (the overflow/invalid read): a
#           GPU idle gap inside this window
#   gather  verify: the two Triton gathers of the late misses (PCIe-bound);
#           draft: the side-stream launch + the bias scratch copy/masked_fill
#   copies  the two union-wide V * exp(cis) / exp(cis) copies (elementwise kernels)
#           preceded by the two host-sync guards of build_varlen_args (idle gaps)
#   attn1   varlen call 1 (kernel)     attn2   varlen call 2 (kernel)
#   quot    the bf16 quotient (elementwise kernel)
#   rest    GEMMs, norms, rope, embedding, lm_head (kernels) and every host gap
#           between them
KERNEL_WINDOWS = ("score_ms", "gather_ms", "attn1_ms", "attn2_ms")
NONKERNEL_WINDOWS = ("tail_ms", "plan_ms", "copies_ms", "quot_ms")
COUNTERS = ("host_sync",)


def _ev():
    return torch.cuda.Event(enable_timing=True)


def split_marks(total_ms: float, per_layer, counters=None) -> dict:
    """per_layer: iterable of dicts mark -> ms offset from the call start (only
    recorded marks present). Every window of WINDOWS_V2 whose two marks are
    present in a layer is summed over the layers; ``rest_ms`` = total minus
    the score / fetch / attn windows (the same partition as split_call, so a
    v1 row and a v2 row are comparable). A negative window = marks out of
    order: refuse, never clamp."""
    if not (total_ms == total_ms) or total_ms < 0:
        raise ValueError("split_marks: total %r" % total_ms)
    out = {name: 0.0 for name, _, _ in WINDOWS_V2}
    have = {name: 0 for name, _, _ in WINDOWS_V2}
    n = 0
    for d in per_layer:
        for name, a, b in WINDOWS_V2:
            if a in d and b in d:
                w = float(d[b]) - float(d[a])
                if not (w == w) or w < 0:
                    raise ValueError("split_marks: layer %d window %s = %r is negative or NaN (marks out of order?)" % (n, name, w))
                out[name] += w
                have[name] += 1
        n += 1
    row = dict(total_ms=float(total_ms), layers=n)
    for name, _, _ in WINDOWS_V2:
        row[name] = out[name] if have[name] else None
    core = sum(out[k] for k in ("score_ms", "fetch_ms", "attn_ms") if have[k])
    row["rest_ms"] = float(total_ms) - core
    for c in COUNTERS:
        row[c] = int((counters or {}).get(c, 0))
    return row


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
        c = dict(label=self.next_label, start=_ev(), end=_ev(), layers=[None] * self.num_layers,
                 counters={k: 0 for k in COUNTERS})
        self.next_label = None
        c["start"].record()
        self.calls.append(c)
        self.cur = c
        self.layer = None

    def begin_layer(self, layer_idx: int):
        if self.cur is None:
            return
        self.layer = layer_idx
        self.cur["layers"][layer_idx] = {}

    def rec(self, key: str):
        """Record mark ``key`` on the current layer (an event, created on
        first use), or count it when it is a COUNTERS name (a host sync)."""
        if self.cur is None:
            return
        if key in COUNTERS:
            self.cur["counters"][key] += 1
            return
        if self.layer is None:
            return
        if key not in LAYER_MARKS_V2:
            raise KeyError("verify_trace: unknown mark %r" % key)
        d = self.cur["layers"][self.layer]
        if key not in d:
            d[key] = _ev()
        d[key].record()

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
        """One row per call: split_marks over the recorded marks (a call whose
        layers recorded only the four v1 marks gets the v1 windows and None
        for the finer ones), plus the counters and the caller's label."""
        torch.cuda.synchronize()
        rows = []
        for c in self.calls:
            per_layer = []
            for d in c["layers"]:
                if d is None:
                    continue
                per_layer.append({k: c["start"].elapsed_time(ev) for k, ev in d.items()})
            row = split_marks(c["start"].elapsed_time(c["end"]), per_layer, c.get("counters"))
            row["label"] = c["label"]
            rows.append(row)
        return rows

    def reset(self):
        self.calls = []
        self.cur = None
        self.layer = None
