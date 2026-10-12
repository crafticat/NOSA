"""R6 negative controls: TEST-ONLY switches. Every switch is OFF by default and the dispatcher reads them at the exact
code point the mutation breaks. A test turns one on and must FAIL (retroinfer-eval tests/test_kstep_mutations.py).
The pattern follows benchmarks/Efficiency/cpupack_core.py:497-510 (Faults) and feeder_core.FeederFaults."""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Mutations:
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

    def active(self):
        return [f.name for f in fields(self) if getattr(self, f.name)]


NAMES = tuple(f.name for f in fields(Mutations))
