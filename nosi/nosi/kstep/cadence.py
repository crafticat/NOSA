"""Cadence (spec section 2, C1-C4): per-request, epoch-relative cohorts and the declared lead bound.

Notation (spec section 1). An epoch of request r starts at E_r = n_r with u_0 = x_r. Draft step s (s = 1, 2, ...) feeds
u_{s-1} at absolute position E_r + s - 1 and produces d_s (the token at position E_r + s) and its recorded q_s.

C1  cohort m verifies d_i, i in I_1 = [1, K-1] (m = 1) or I_m = [(m-1)K, mK-1] (m >= 2). It is TRIGGERED when draft
    step s = mK completes. d_{mK} is the deferred newest token and belongs to cohort m+1. The verifier of cohort m feeds
    u_{i-1}, i in I_m (K-1 inputs for m = 1, K after): inputs of consecutive cohorts are contiguous and disjoint.
C2  lead = drafted tokens of the epoch - tokens of the epoch already committed. A request at the bound LAMBDA does not
    draft in that step (a STALL). LAMBDA is required (no default); LAMBDA <= K is refused (draft-K-then-stop).
C3  full acceptance commits |I_m| tokens and NO bonus; a rejection ends the epoch (cancellation in the dispatcher).
C4  at max_new_tokens the request flushes: the drafts after the last triggered cohort form a final (possibly short)
    cohort, then the request stops drafting.

Prior art (docs/superpowers/specs/2026-09-02-prior-art-table.md, 2026-10-12 row a): PEARL @fe7ddef src/engine.py:164-253
is the closest schedule (K = gamma, lead 2K, no bonus while drafts are in flight, :238); it runs on two devices with
B = 1. Nothing here is copied from it (no license file).
"""
from __future__ import annotations

from typing import Optional, Tuple

from .errors import ConfigError


def check(K, LAMBDA) -> Tuple[int, int]:
    """Refuse K < 2 (C1) and a missing or too small LAMBDA (C2). Returns (K, LAMBDA) as ints."""
    if K is None or isinstance(K, bool) or int(K) != K or int(K) < 2:
        raise ConfigError("C1: K >= 2 is required (got K=%r)" % (K,))
    if LAMBDA is None:
        raise ConfigError("C2: the lead bound LAMBDA is a required parameter (no default)")
    if isinstance(LAMBDA, bool) or int(LAMBDA) != LAMBDA:
        raise ConfigError("C2: LAMBDA must be an integer (got %r)" % (LAMBDA,))
    if int(LAMBDA) <= int(K):
        raise ConfigError("C2: LAMBDA=%d <= K=%d is the draft-K-then-stop-to-verify schedule, which the user forbids"
                          % (int(LAMBDA), int(K)))
    return int(K), int(LAMBDA)


def cohort_of(s: int, K: int) -> int:
    """The cohort that verifies d_s (s >= 1): m = floor(s / K) + 1 (s in [1, K-1] -> 1; s in [(m-1)K, mK-1] -> m)."""
    if s < 1:
        raise ValueError("draft step index starts at 1 (got %d)" % s)
    return s // K + 1


def cohort_range(m: int, K: int) -> Tuple[int, int]:
    """Inclusive index range [a, b] of I_m."""
    if m < 1:
        raise ValueError("cohort index starts at 1 (got %d)" % m)
    return (1, K - 1) if m == 1 else ((m - 1) * K, m * K - 1)


def triggered_by(s: int, K: int) -> Optional[int]:
    """The cohort TRIGGERED when draft step s completes (s = mK), else None."""
    return s // K if (s >= K and s % K == 0) else None


def flush_range(n_triggered: int, S: int, K: int) -> Optional[Tuple[int, int]]:
    """C4: the final cohort after S drafted steps of which n_triggered cohorts were triggered: [a, S] with a the next
    cohort's first index; None when every draft is already in a triggered cohort (cannot happen for S >= 1)."""
    m = n_triggered + 1
    a = cohort_range(m, K)[0]
    return (a, S) if a <= S else None


def lead(drafted: int, committed_in_epoch: int) -> int:
    """C2: uncommitted lead of the epoch."""
    return int(drafted) - int(committed_in_epoch)
