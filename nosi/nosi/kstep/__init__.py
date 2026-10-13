"""K-step live speculative KV, milestone 1 (M1): the CPU-provable core.

Spec: retroinfer-eval docs/superpowers/specs/2026-10-12-kstep-live-m1.md (d598e4e). The design is FROZEN by the user;
this package only makes it exact and testable. Pure python + torch: nothing here imports CUDA, triton or the engine
(cache_engine.py / nosa_llama.py are not touched by M1).

Modules (each names the spec clauses it implements):
  cadence      C1-C4   epoch cohorts I_1 = [1, K-1], I_m = [(m-1)K, mK-1], the deferred newest token, the lead bound
  rng          A1      the counter-based generator keyed by (seed, uid, absolute position, epoch, purpose)
  accept       A1, A2  exact p/q acceptance with the recorded q, residual norm(max(0, p - q)), zero-mass rule; GREEDY
  compressed   A6      the journaled compressed-key chain (append, snapshot, restore across a compression boundary)
  ownership    K1-K6   per-layer pool: logical LRU (C per stream; DRAFT-ONLY recency after repair 1 R2), per-(block,
                       cohort) protection, protected-only alias, verifier-resident (VRES) entries, in flight,
                       reader-gated release, X shared extra slots with a verifier reserve; memory accountant
  transfer     T1-T4   the transfer record (uid, block, slot, generation), the S0 adapter over cpupack_core.Pipe and an
                       adversarial transport (seeded delay, reorder, bursts, complete-only-when-forced)
  backend      -       the stream-backend interface (the cpupack_core CpuBackend / CudaBackend shape) and ev_done()
  executor     D1, D5  the model-executor interface that M2's GPU executor implements (M1: the toy model, test side)
  dispatcher   D1-D6, A3-A7, C1-C4, K1-K5, T2-T4: the REAL dispatch (the tests drive this class, never a copy)
  mutations    R6      TEST-ONLY negative-control switches (all off by default): the 16 of R6, 5 of the fix round,
                       16 of repair 1 (R1 per-layer pin release, R2 draft-only LRU, R3 per-cohort join check) and 7 of
                       the capacity mode "rev6"
Capacity modes (dispatcher.Config.capacity): "m1" (default, the M1 rules unchanged) and "rev6" (retroinfer-eval
docs/superpowers/specs/2026-10-12-kstep-m1-progress-policy-PROPOSAL.md rev 6, section 8: position ledger, step credits,
aggregate draft admission, V-only verifier residuals with age-ordered all-or-nothing grants, no verifier eviction).

Labels: no timing, throughput, interference or memory-fit claim (spec section 9). The KV read sharing recorded per call
(D1) is the IDEAL from actual selections (SUM vs UNION of selected blocks), not a measured kernel saving.
"""
from .errors import (CapacityRefusal, ConfigError, KStepError, NoProgress, OwnershipViolation, ReadCorruption,  # noqa: F401
                     StaleGeneration)
