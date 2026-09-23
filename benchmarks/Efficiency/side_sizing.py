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
