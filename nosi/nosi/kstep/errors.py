"""Errors of the K-step core. Every guard raises one of these where the value is produced (fail fast, never a silent
fallback). The names follow benchmarks/Efficiency/feeder_core.py (OwnershipViolation, StaleGeneration)."""


class KStepError(RuntimeError):
    """Base class: a violated M1 rule."""


class ConfigError(KStepError, ValueError):
    """A refused parameter (K < 2, LAMBDA <= K, a missing LAMBDA, an impossible pool)."""


class OwnershipViolation(KStepError):
    """A slot transition that the ownership rules forbid (K3-K5, T3)."""


class StaleGeneration(OwnershipViolation):
    """A transfer or a release names a slot generation that the slot no longer holds (T3)."""


class NoProgress(KStepError):
    """D4: a verifier-only pass could not advance any cohort, observe any arrival or free any slot."""


class ReadCorruption(KStepError):
    """A deferred read of a slot saw other bytes than the read the joint call was computed from (T3, T4, K4)."""


class CapacityRefusal(KStepError):
    """Capacity mode "rev6" (retroinfer-eval PROPOSAL rev 6, G5 and the draft side of P6): no grant, draft step, transfer,
    reader event or release can ever free the capacity that the oldest waiter needs. An explicit refusal, never a hang."""
