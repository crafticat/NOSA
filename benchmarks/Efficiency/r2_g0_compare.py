"""R2 milestone 2, step 5: the PURE comparator of the G0 gate (benchmarks/Efficiency/r2_g0_gate.py). Torch only; the
CPU tests (tests_cpu/test_r2_g0.py) run it on planted differences.

A VIEW is {name: (tensor, request_dim)} (request_dim None: compared as a whole). compare_views returns, per differing
name, the differing requests (or "shape" / "whole"). Tracker keeps, per table, the first (step, layer, requests) and
the set of (table, request) that ever differed, and the number of comparisons. Two verdicts:
  verdict_equal(tracker)                                  nothing ever differed
  verdict_neg(tracker, step0, requests0)                  the first difference of every table is at step >= step0,
                                                          at least one table differs at exactly step0, and only the
                                                          requests in requests0 ever differ (None: any request)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def compare_views(va: Dict[str, Tuple], vb: Dict[str, Tuple]) -> Dict[str, object]:
    """Names present in both views are compared; a name in one view only is reported as 'missing'."""
    out: Dict[str, object] = {}
    for name in sorted(set(va) | set(vb)):
        if name not in va or name not in vb:
            out[name] = "missing"
            continue
        (a, da), (b, db) = va[name], vb[name]
        if da != db:
            out[name] = "dim"
            continue
        if not torch.is_tensor(a):
            if a != b:
                out[name] = "whole"
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            out[name] = "shape"
            continue
        if da is None:
            if not bool(torch.equal(a, b)):
                out[name] = "whole"
            continue
        bad = [r for r in range(a.shape[da]) if not bool(torch.equal(a.select(da, r), b.select(da, r)))]
        if bad:
            out[name] = bad
    return out


class Tracker:
    def __init__(self):
        self.first: Dict[str, Tuple[int, Optional[int], object]] = {}
        self.ever: Dict[str, set] = {}
        self.compared = 0
        self.steps = 0

    def update(self, step: int, layer: Optional[int], diffs: Dict[str, object]) -> None:
        self.compared += 1
        self.steps = max(self.steps, step + 1)
        for name, req in diffs.items():
            if name not in self.first:
                self.first[name] = (step, layer, req)
            s = self.ever.setdefault(name, set())
            if isinstance(req, list):
                s.update(req)
            else:
                s.add(req)

    def first_step(self) -> Optional[int]:
        return min((v[0] for v in self.first.values()), default=None)

    def summary(self) -> Dict:
        return dict(compared=self.compared, steps=self.steps, first_step=self.first_step(),
                    first={k: dict(step=v[0], layer=v[1], requests=v[2]) for k, v in sorted(self.first.items())},
                    ever={k: sorted(map(str, v)) for k, v in sorted(self.ever.items())})


def verdict_equal(t: Tracker) -> Tuple[bool, str]:
    if not t.first:
        return True, "bitwise equal: %d comparisons over %d steps" % (t.compared, t.steps)
    return False, "first difference at step %s: %s" % (t.first_step(), {k: v for k, v in sorted(t.first.items())})


def verdict_neg(t: Tracker, step0: int, requests0: Optional[Iterable[int]]) -> Tuple[bool, str]:
    fs = t.first_step()
    if fs is None:
        return False, "NO difference found: the comparator missed the planted change"
    if fs != step0:
        return False, "first difference at step %d, expected exactly %d: %s" % (fs, step0, t.first)
    if requests0 is not None:
        allowed = {int(r) for r in requests0}
        stray = {k: sorted(map(str, v - allowed)) for k, v in t.ever.items() if v - allowed}
        if stray:
            return False, "differences outside requests %s: %s" % (sorted(allowed), stray)
    return True, "first difference at step %d as planted (tables %s)" % (fs, sorted(k for k, v in t.first.items() if v[0] == fs))
