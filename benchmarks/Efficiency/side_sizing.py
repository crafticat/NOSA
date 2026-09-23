"""Side-transfer sizing and validity rules of the GPU-sleep-gated overlap bracket, shared by worker_sweep.py
(NOSI's resident decode step) and hisparse_repro.py (the decode-step proxy). Pure python, CPU-tested
(retroinfer-eval tests/test_hisparse_copy_plan.py).

THE DEFECT THIS FIXES (miniature job 2175723, B = 16, 16 CTAs): passes were chosen so the side lasts
1.2 x the step ALONE (18.4 ms). Beside the transfer the step slowed to 39.5 ms (+114%), so the side
(3 passes) ended first and overlap_frac fell to 0.92: the concurrent step was partly measured alone.

RULE NOW
  1. initial_passes: the side lasts >= ALONE_MARGIN (1.2) x the step alone, from a one-pass side-alone run.
  2. size_side: a TRIAL concurrent bracket at that many passes; while the side lasts < CONC_MARGIN (1.15) x
     the CONCURRENT step, raise passes from the measured per-pass concurrent time and try again (at most
     max_trials trials). The last trial's outcome is recorded (covered or not).
  3. validity (unchanged): every concurrent rep must have overlap_frac >= OVERLAP_MIN (0.95) and every
     rep host_lag <= HOST_LAG_MAX_MS (0.1 ms); rows that fail are kept and flagged INVALID.

DURING-STEP ACCOUNTING (review B3, 2026-09-23). Interval coverage alone cannot show that the copy ran DURING the
step: the whole-interval GB/s (bytes / side interval) includes the >= 15% tail the side runs alone after the step
ends, and a per-layer launch queued behind the step still reads overlap 1.0. So every side function records ONE
event on the side stream after each launch unit (one layer: K and V) through LAUNCH_LOG.mark(bytes), and each
concurrent bracket reports, from those events (all timed from the gate; t0 = gate + host_lag):
  during_bytes  bytes of the launch units whose END falls inside [t0, tm] (the step's interval)
  during_gbps   during_bytes / main_ms: a LOWER BOUND of the bandwidth delivered during the step (the unit still in
                flight at tm is not counted)
  inside_frac   during_bytes / all side bytes of the rep
  last_end_ms   end of the last unit relative to t0 (<= main_ms: the whole side finished inside the step)
"""
import math
from typing import Callable, Dict, List, Tuple

ALONE_MARGIN = 1.2
CONC_MARGIN = 1.15
OVERLAP_MIN = 0.95
HOST_LAG_MAX_MS = 0.1


def initial_passes(step_alone_ms: float, one_pass_ms: float, margin: float = ALONE_MARGIN) -> int:
    return max(1, int(math.ceil(margin * step_alone_ms / max(one_pass_ms, 1e-3))))


def covers(side_ms: float, main_ms: float, margin: float = CONC_MARGIN) -> bool:
    return side_ms >= margin * main_ms


def raised_passes(passes: int, side_conc_ms: float, main_conc_ms: float, margin: float = CONC_MARGIN) -> int:
    """Passes for the side to last >= margin x the concurrent step, from a trial at `passes` (side_conc_ms is
    the trial's side interval, main_conc_ms the trial's step). Unchanged when the trial already covers."""
    if covers(side_conc_ms, main_conc_ms, margin):
        return passes
    per_pass = max(side_conc_ms / max(passes, 1), 1e-6)
    return max(passes + 1, int(math.ceil(margin * main_conc_ms / per_pass)))


def size_side(trial: Callable[[int], Tuple[float, float]], step_alone_ms: float, one_pass_ms: float,
              max_trials: int = 3, alone_margin: float = ALONE_MARGIN, conc_margin: float = CONC_MARGIN) -> Dict:
    """trial(passes) runs ONE concurrent bracket and returns (side_ms, main_ms) (side_ms = side interval
    minus host lag). Returns dict(passes, initial_passes, covered, trials=[{passes, side_ms, main_ms}])."""
    passes = initial_passes(step_alone_ms, one_pass_ms, alone_margin)
    out: Dict = dict(initial_passes=passes, trials=[], covered=False)
    trials: List[Dict] = out["trials"]
    for _ in range(max(1, int(max_trials))):
        side_ms, main_ms = trial(passes)
        trials.append(dict(passes=passes, side_ms=float(side_ms), main_ms=float(main_ms)))
        new = raised_passes(passes, side_ms, main_ms, conc_margin)
        if new == passes:
            out["covered"] = True
            break
        passes = new
    out["passes"] = passes
    return out


def overlap_frac(side_ms: float, host_lag_ms: float, main_ms: float) -> float:
    """The share of the step's interval the side covers (the bracket's definition)."""
    return min(side_ms - host_lag_ms, main_ms) / main_ms if main_ms > 0 else float("nan")


def valid(overlaps, host_lags) -> bool:
    ov, hl = list(overlaps), list(host_lags)
    return bool(ov) and min(ov) >= OVERLAP_MIN and (not hl or max(hl) <= HOST_LAG_MAX_MS)


def lag_ok(host_lags) -> bool:
    """The timer rule alone (the burst regime has no coverage requirement: its side may end before the step)."""
    hl = list(host_lags)
    return bool(hl) and max(hl) <= HOST_LAG_MAX_MS


def during_step(end_from_gate_ms, nbytes, host_lag_ms: float, main_ms: float) -> Dict:
    """Pure: launch-unit end times (ms from the gate event) and their bytes -> the during-step record (see the module
    docstring). A unit ending before t0 or after tm is not counted as delivered during the step."""
    ends = [float(e) - float(host_lag_ms) for e in end_from_gate_ms]
    nb = [int(b) for b in nbytes]
    if len(ends) != len(nb):
        raise ValueError("%d end times for %d byte counts" % (len(ends), len(nb)))
    total = sum(nb)
    during = sum(b for e, b in zip(ends, nb) if 0.0 <= e <= main_ms)
    nan = float("nan")
    return dict(n_launches=len(nb), side_bytes=total, during_bytes=during,
                during_gbps=(during / (main_ms * 1e6) if main_ms > 0 else nan),
                inside_frac=(during / total if total else nan),
                first_end_ms=(min(ends) if ends else nan), last_end_ms=(max(ends) if ends else nan))


def _cuda_timing_event():
    import torch
    return torch.cuda.Event(enable_timing=True)


class LaunchLog:
    """Events recorded on the CURRENT stream (the side stream, inside `with torch.cuda.stream(side)`) after each
    launch unit, with the bytes that unit moved. Events come from a pool that grows once and is reused, so a
    bracket allocates nothing after the first rep. reset() at the start of every bracket."""

    def __init__(self, event_factory=None):
        self._factory = event_factory or _cuda_timing_event
        self._pool: List = []
        self._bytes: List[int] = []
        self.n = 0

    def reset(self) -> None:
        self.n = 0
        self._bytes = []

    def mark(self, nbytes: int) -> None:
        if self.n == len(self._pool):
            self._pool.append(self._factory())
        self._pool[self.n].record()
        self._bytes.append(int(nbytes))
        self.n += 1

    def marks(self) -> List[Tuple[object, int]]:
        return list(zip(self._pool[: self.n], self._bytes))


LAUNCH_LOG = LaunchLog()
