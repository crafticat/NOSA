"""Negative controls: TEST-ONLY switches. Every switch is OFF by default and the dispatcher reads them at the exact code
point the mutation breaks. A test turns one on and must FAIL (retroinfer-eval tests/test_kstep_mutations.py).
The pattern follows benchmarks/Efficiency/cpupack_core.py:497-510 (Faults) and feeder_core.FeederFaults.

R6 lists the 16 mutations of spec section 7 R6. EXTRA lists the negative controls of the M1 fix round (checker findings
F1, F2, F4, F5): each one survived the first M1 suite and is caught by a test of the fix round. EXTRA also lists the 16
negative controls of repair 1 (retroinfer-eval docs/superpowers/specs/2026-10-12-kstep-m1-repair1.md, R1 per-layer pin
release, R2 draft-only logical LRU, R3 per-cohort join check); the test that each one must fail is named in
retroinfer-eval tests/test_kstep_mutations.py (CATCHERS)."""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Mutations:
    # ---------------------------------------------------------------- spec R6
    free_on_cancel: bool = False            # T3: a cancelled requester frees its in-flight destination before landing
    no_reader_wait: bool = False            # T3/K4: a slot is reused without waiting for its previous reader event
    publish_draft_kv: bool = False          # A4: the draft KV is published as committed KV
    publish_beyond_prefix: bool = False     # A4: a rejection publishes every input of the cohort
    batch_min_commit: bool = False          # A3: 1 + min_b n_acc over the requests that accept in one step
    verifier_recomputes_q: bool = False     # A1: the acceptance uses q := p (the verifier's own distribution)
    greedy_in_sampled: bool = False         # A1: sampled mode accepts iff argmax p == d
    residual_no_clamp: bool = False         # A1: residual p - q without max(0, .)
    cohort_order_skip: bool = False         # D3a: cohort m+1 may execute layer l before cohort m
    no_uid_check: bool = False              # T3: an arrival is admitted whatever request owns the batch slot
    release_first_reader: bool = False      # K4: release after the FIRST reader event instead of the last
    drop_alias: bool = False                # K3: a protected entry evicted from the LRU loses its slot (map kept)
    ignore_lead_bound: bool = False         # C2: drafting continues past LAMBDA
    draft_k_then_stop: bool = False         # C2: a request stops drafting after K drafts until its cohort is verified
    no_compressed_rollback: bool = False    # A6: the draft compressed-key chain is not restored at a new epoch
    read_not_ready: bool = False            # D3b/T4: a verifier executes a layer with an in-flight block
    # ---------------------------------------------------------------- fix round (not in R6)
    leak_cancelled_protection: bool = False     # K4/C3 (F1): a cancelled cohort's protections are never queued for release
    no_draft_hit_protection: bool = False       # K2 (F2): a draft row protects its misses only, not its hits
    no_verifier_ready_protection: bool = False  # K2 (F2): a waiting verifier protects in-flight and missing blocks only
    mem_no_releasing: bool = False              # K6 (F4): the memory sample omits RELEASING slots
    no_dchain_rebuild: bool = False             # D6/A4 (F5): the draft chain keeps draft keys of published positions
    # ---------------------------------------------------------------- repair 1, R1 (per-layer historical pin release)
    release_only_at_decision: bool = False      # R1.6 old behaviour: no per-layer queue; release only at the decision
    release_at_execution: bool = False          # R1.6 early release: dropped when the layer executes, no reader gate
    prot_release_no_reader_gate: bool = False   # R1.6 delayed reader: a queued release ignores the (key, l) reader gate
    release_drops_all_keys: bool = False        # R1.6 shared pins: releasing one cohort drops every key of the block
    cleanup_skips_unqueued_layers: bool = False  # R1.6 rollback: decision cleanup queues no layer
    cleanup_double_queues: bool = False         # R1.6 rollback: decision cleanup queues every layer again
    no_late_protection_guard: bool = False      # R1.5: a protection after its (key, l) release was queued is accepted
    # ---------------------------------------------------------------- repair 1, R2 (draft-only logical LRU)
    verifier_touches_lru: bool = False          # R2.2: verifier reads move LRU entries to MRU (the M1 rule)
    verifier_arrival_enters_lru: bool = False   # R2.4: a verifier-only arrival enters the LRU at MRU (the M1 rule)
    drop_demand_merge: bool = False             # R2.3: a request for an in-flight block adds nothing to its demand set
    skip_acquisition: bool = False              # R2.1(c): a draft read of a PO / VRES entry leaves it outside the LRU
    vres_counted_as_lru: bool = False           # R2.4 / K6: the memory sample counts VRES slots as LRU slots
    silent_verifier_eviction: bool = False      # R2.5: a verifier eviction is not recorded per stream
    verifier_evicts_without_pressure: bool = False  # R2.5: the verifier evicts a draft entry before every verifier fetch
    # ---------------------------------------------------------------- repair 1, R3 (D2 per-cohort join)
    delayed_join_even_steps: bool = False       # = recheck2_cqlhP6/mut/OWN_join_even_steps/MUTATION.diff
    delayed_join_wait_other_requests: bool = False  # = recheck2_cqlhP6/mut/OWN_join_wait_other_requests/MUTATION.diff

    def active(self):
        return [f.name for f in fields(self) if getattr(self, f.name)]


NAMES = tuple(f.name for f in fields(Mutations))
EXTRA_FIX_ROUND = ("leak_cancelled_protection", "no_draft_hit_protection", "no_verifier_ready_protection",
                   "mem_no_releasing", "no_dchain_rebuild")
EXTRA_R1 = ("release_only_at_decision", "release_at_execution", "prot_release_no_reader_gate", "release_drops_all_keys",
            "cleanup_skips_unqueued_layers", "cleanup_double_queues", "no_late_protection_guard")
EXTRA_R2 = ("verifier_touches_lru", "verifier_arrival_enters_lru", "drop_demand_merge", "skip_acquisition",
            "vres_counted_as_lru", "silent_verifier_eviction", "verifier_evicts_without_pressure")
EXTRA_R3 = ("delayed_join_even_steps", "delayed_join_wait_other_requests")
EXTRA_REPAIR1 = EXTRA_R1 + EXTRA_R2 + EXTRA_R3
EXTRA = EXTRA_FIX_ROUND + EXTRA_REPAIR1
R6 = tuple(n for n in NAMES if n not in EXTRA)
