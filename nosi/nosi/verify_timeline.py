"""CUDA-TIMELINE instrumentation of one Path 1 verify call (retroinfer-eval
fork; the pilot's ``timeline`` mode, spec 2026-09-19-multiposition-verify-path1.md
section 4, ledger 2026-09-20 'verify brackets -> kernels').

WHY. The ``cost`` mode's CUDA-event brackets (verify_trace.py) say WHERE a
verify call spends its wall (job 2175382, batch 128, U = 2: fetch 117.7 ms,
attention 93.6, score 14.2, rest 21.9) but not WHAT: a bracket's wall is
kernel time plus GPU idle time, and the idle time is host work (the two
host-sync guards of build_varlen_args, the overflow read of the union
builder, pageable copies, ~40 small integer ops per layer). This module
turns the same four marks into ``torch.profiler.record_function`` ranges,
so a kineto trace (CPU + CUDA activities) attributes every kernel, memcpy
and memset to the bracket whose host code launched it, and the analyzer
(retroinfer-eval ``scripts/verify_timeline_report.py``) reports per bracket:
kernel time by family, wall vs kernel-sum (= host gaps), the largest host
ops inside the gaps, and copy-engine/SM overlap across streams.

HOW. ``TimelineTrace`` is ``VerifyTrace`` (the events stay: their brackets
are the cross-check of the timeline's attribution) plus ``BracketRanges``:
at ``begin_call`` a ``rest`` range opens; every mark closes the open range
and opens the next one (``score_begin`` -> ``score``, ``fetch_begin`` ->
``fetch``, ``fetch_end`` -> ``attention``, ``attn_end`` -> ``rest``);
``end_call`` closes the last one. The ranges never overlap and together
cover the call, so per layer they partition it exactly as split_call does:
  rest      norm, qkv GEMM, nosa_linear, rope  (before score_begin)
  score     U x (compress-k/cis updates, stage 1, the pooling/top-k graph)
  fetch     verify_round_update_kv: tail writes, union build (one host
            sync), the two Triton gathers of the round region
  attention verify_attention: two host-sync guards, exp(cis), V * exp(cis),
            the expand-copy, two varlen calls, the quotient
  rest      wo GEMM, residual, norm, gate-up/down GEMMs, silu  (after attn_end)
The caller wraps the whole ``verify_inference`` in
``record_function("verify_call")`` (and a shipped step in ``"decode_call"``),
outside this module, so the ranges nest call > bracket.

The range factory is injected (``open_range(name) -> close()``): production
uses ``torch.profiler.record_function``; the CPU test (retroinfer-eval
tests/test_verify_timeline_report.py) injects a recorder and checks the
open/close sequence without CUDA. No float is touched by any of this.
"""
from __future__ import annotations

import torch

from .verify_trace import VerifyTrace

BRACKETS = ("score", "fetch", "attention", "rest")
CALL_RANGES = ("verify_call", "decode_call")
# the range that OPENS at each verify_trace mark (the previous one closes)
AFTER_MARK = {"score_begin": "score", "fetch_begin": "fetch", "fetch_end": "attention", "attn_end": "rest"}


def record_function_range(name: str):
    """Open a torch.profiler.record_function range NOW; returns its closer."""
    rf = torch.profiler.record_function(name)
    rf.__enter__()
    return lambda: rf.__exit__(None, None, None)


class BracketRanges:
    """The mark sequence -> non-overlapping named ranges (see the module doc)."""

    def __init__(self, open_range=record_function_range):
        self._open_range = open_range
        self._close = None
        self.name = None

    def _switch(self, name: str):
        if self._close is not None:
            self._close()
        self._close = self._open_range(name)
        self.name = name

    def begin(self):
        self._switch("rest")

    def mark(self, key: str):
        if key not in AFTER_MARK:
            raise KeyError("unknown verify_trace mark %r (expected one of %s)" % (key, sorted(AFTER_MARK)))
        self._switch(AFTER_MARK[key])

    def end(self):
        if self._close is not None:
            self._close()
        self._close = None
        self.name = None

    abort = end


class TimelineTrace(VerifyTrace):
    """VerifyTrace (CUDA-event brackets) + record_function bracket ranges."""

    def __init__(self, num_layers: int, open_range=record_function_range):
        super().__init__(num_layers)
        self.ranges = BracketRanges(open_range)

    def begin_call(self):
        super().begin_call()
        self.ranges.begin()

    def rec(self, key: str):
        super().rec(key)
        if self.cur is not None and self.layer is not None:
            self.ranges.mark(key)

    def end_call(self):
        self.ranges.end()
        super().end_call()

    def abort_call(self):
        self.ranges.abort()
        super().abort_call()
