"""The K-step live dispatcher (spec sections 2-6): the REAL dispatch that the M1 tests drive.

ONE DECODE STEP t
  start   admit queued requests into free batch slots (prefill, outside M1); cohorts triggered at the end of step t-1
          join now at layer 0 (D2); every active request either starts its next draft step or STALLS with a counted
          reason: 'lead' (lead == LAMBDA, C2) or 'done' (flushed or finished, C4). If no request drafts and a cohort waits,
          the step is a VERIFIER-ONLY pass (D4, counted).
  layers  l = 0 .. L-1: ONE joint call per layer (D1): the draft rows (one per drafting request) and the verifier rows of
          every cohort that may execute layer l now (D3).
  end     (publication happens only here, between calls, D6) acceptance of every cohort that finished layer L-1, in
          cohort order per request (A1-A3); publication of the verifier's exact KV of the accepted prefix (A4); history
          (A5); rollback and cancellation at a rejection (A6, C3); then the draft rows of the step sample d_s ~ q_s and
          record q_s; cohort triggers (C1) and the flush (C4).

ONE JOINT CALL (layer l)
  1 host poll: arrivals of every transport (each transfer's own completion, T2): a demand set with "draft" enters the
    LRU at MRU, a verifier-only one becomes VRES (repair 1 R2.1(b), R2.4); RELEASING slots whose last reader completed
    become FREE (K4); deferred-read validators whose reader event completed; queued protection releases (key, layer)
    whose last reader event of (key, layer) completed (K4 as amended by repair 1 R1).
  2 verifier cohorts at layer l in cohort order: D3a (cohort m-1 of the epoch executed layer l in an EARLIER call, or is
    accepted). Phase 1 once (saved as workspace, counted in memory, not recomputed): exact q/k/v of every row, the
    cohort's exact compressed-key chain, exact selections (D5). Protection of the exact selection (K2). D3b: every
    selected historical block READY (resident, arrival observed); otherwise the cohort waits: missing blocks are
    requested (verifier fetch, K5), in-flight ones are waited for, and the cohort retries at the next call of layer l.
  3 draft rows: phase 1 from the DRAFT hidden state and the draft chain; protection of every selected historical block
    (hits AND misses) for the cohort of d_s (K2); readable = resident; misses requested (draft fetch, K5; a refusal is
    counted and the draft continues masked); a miss already in flight adds "draft" to that transfer's demand (R2.3).
  4 verifier read log (repair 1 R2.2: verifier reads never touch the LRU).
  5 phase 2: the rows' reads (slot reads after the host-observed arrival, T4; recent region from the published state as
    of the call start, D6) -> executor.joint_post. Every slot read is ALSO enqueued as a deferred copy on the "compute"
    stream; the call's reader event follows it. A slot is reused only after that event (K4, T3); a validator compares the
    deferred copy with the bytes the call computed from (ReadCorruption otherwise).
    Then the DRAFT-ONLY recency (repair 1 R2.1), per stream after the reads: the draft row's LRU hits move to MRU (read
    order), then its reads of PO / VRES entries enter the LRU at MRU (draft acquisition, evicting the least recent).
    Each cohort that executed layer l queues the release of its layer-l references (repair 1 R1.1).
  6 fetch issue: verifier requests (cohort order) then draft requests; slot allocation (K5); one transport issue per
    layer (S0: one Pipe.layer); a request for a block already requested in this call or in flight adds its kind to that
    transfer's demand set (R2.3).
  7 D1 records (per stream SUM and UNION of the rows' selected blocks: the IDEAL sharing, not a measurement) and the
    memory sample (K6).

READINGS WHERE THE SPEC LEAVES A CHOICE (the most conservative one; listed in retroinfer-eval
docs/evidence/kstep_live/M1_RECEIPTS.md):
  * D3a "has executed layer l" = in an EARLIER joint call (outputs of a call become visible only after it).
  * K2 verifier protections are held until the cohort's layer-l references are released (repair 1 R1: after the call
    in which it executed layer l, behind the reader gate), not only while it waits.
  * A block that becomes historical (A5) leaves the recent region and is NOT admitted into the LRU: the LRU admits only
    observed fetch arrivals (K1); a later selection fetches it.
  * The draft's compressed-key chain is the committed exact chain extended by the draft keys of the uncommitted
    positions. It is restored at every new epoch (A6) and REBUILT at every full-acceptance publication (_rebuild_dchain,
    fix round F5: D6/A4 "drafts read the exact KV from then on" covers the compressed keys derived from that KV).
  * K5 draft reserve: a draft fetch may borrow an extra slot only while MORE than R_ver extra slots are free.
  * Positions: the prefill publishes positions 0 .. P0-2; the last prompt token is the pending token x (spec section 1);
    the first generated token is d_1 of epoch 0.
ADDITIONS FOR D4 (reported as spec deviations): the verifier eviction (_verifier_evict) and the pin guard (_pins_high,
a C2 'memory' refusal) with the parameter condition C >= (2K + 1) * max_sel (Config.pin_guard). Repair 1 R2.5 keeps the
verifier eviction ONLY as an explicit, counted progress fallback: each use is recorded per stream with the evicted block
(vev_log, trace[uid]["verifier_evictions_of_draft_entries"], report()["verifier_evictions_of_draft_entries"]).
REPAIR 1 (retroinfer-eval docs/superpowers/specs/2026-10-12-kstep-m1-repair1.md): R1 per-layer release of historical
pins (_queue_release, _finished_key, the release step of _poll, _protect), R2 draft-only logical LRU with demand sets
and VRES (_add_demand, the recency step of _call, Stream.admit / Stream.acquire), R3 join_step at PENDING -> RUNNING.
OBSERVER (test hook, fix round): observer(dispatcher, phase, call id, layer) runs after the host poll of a joint call
(phase "poll") and at its end (phase "call"); the M1 tests install a test-side auditor of K2, K4 and K6 there.

CAPACITY MODE "rev6" (Config.capacity; default "m1" keeps every M1 rule above unchanged). The contract of retroinfer-eval
docs/superpowers/specs/2026-10-12-kstep-m1-progress-policy-PROPOSAL.md rev 6, section 8 (Codex approval 2026-10-13 for a
limited CPU implementation; the C >= (2K + 1) * max_sel guard and the pin guard are UNCHANGED and still apply, except
under the opt-in Config.progress_guard = "rev6" of the C81 / K3 note at the end of this docstring):
  L4  construction refuses d_h < W = (K + 1) * S_max and H * d_h > X_d = X - R_ver (S_max = the asserted row bound,
      pin_row_bound or executor.max_sel). d_h >= W is a SUFFICIENT share for this guarantee mode; the C81 / K3 note
      shows only conditional cases that need it (no minimum for every rule). Each active request reserves H * d_h
      shared draft positions per layer, all-or-nothing, in request age order, without overtaking (_rev6_reserve; stall
      reason "admission"); an ended request's reservation returns only when every slot it owned is FREE.
  L2  P(stream) = its non-FREE "ad" slots (an unpinned LRU entry and a RELEASING slot count until FREE).
  SC2 at the step start (before layer 0) a request starts a draft step only if, on every stream of every layer,
      untriggered draft pins + (T - s + 1) * S_max <= d_h ("horizon") and P + S_max <= C + d_h ("capacity"); it then
      holds S_max STEP CREDITS on every (layer, head) at once (Request.credits).
  SC3 only the row's own new "ad" positions take a credit: a draft fetch (_issue), the V -> A / D conversion of a VRES
      read (_draft_recency) or of a draft join to a verify-only transfer (_add_demand, _issue). The unused credits of
      layer l return at the end of the joint call of layer l. A missing credit is an OwnershipViolation (SC4), never a
      refused fetch: fetch_draft_refused stays 0.
  G1-G6 verifier residual fetches are class "v". At layer l the eligible waiting cohorts are taken in age order
      (trigger step, request age): an ungranted cohort pins only "ad" blocks (resident or in flight); the oldest one is
      GRANTED when its new fetches fit the free verifier reserve (R_ver - v_used(l)) and the FREE positions of its streams
      that no credit holds; no younger cohort is granted at that layer in the same call (no overtaking). A granted cohort
      pins its whole exact selection (sharing "v" slots that another grant holds: inheritance without a free) and
      requests its misses. Refused with nothing outstanding at the layer: CapacityRefusal (G5). _verifier_evict is never
      called (G6).
  L9  verifier pins never take a credit or a FREE position; Stream.pin_kind records the provenance (evict_log).
  Verifier-only passes with a rev6 'capacity' / 'admission' stall force every transfer and reader event; with no cohort
      and nothing outstanding the request is refused (CapacityRefusal), never spun.
  H1  recent-to-history handoff (Codex G3 diagnosis 2026-10-13): a cohort's layer-l workspace keeps the history boundary
      of its phase 1 (ws["hist"]). Blocks below it are historical slots (its grant footprint); blocks at or above it are
      read from the recent region, also after a publication makes them historical: _history then keeps their positions
      in r.pub (charged as recent_kv; holders in r.pub_hold, charged as release entries) while a waiting workspace or an
      executed read whose (key, l) reader gate has not released depends on them (h1_reads), and _release_refs drops them
      after the last such reader. So no granted cohort gains a historical need after its grant (G3 holds), and no extra
      fetch, capacity or draft-LRU change follows from a publication.

N3 P1 / P2 (both capacity modes; retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-n3-useful-progress-contract.md,
Codex approval 2026-10-14 for a limited CPU implementation; NoProgress, waits, forcing and every scheduling decision are
UNCHANGED):
  P1  stats["progress"] keeps exactly its old increments, and each one also increments stats["useful"] (a cohort executes
      a layer, a draft step completes, a cohort is decided) or stats["traffic"] (an arriving transfer, dropped ones too;
      a layer pool whose poll freed any slot; the verifier eviction's direct frees add nothing, as before). Observation
      only: progress = useful + traffic (checked with the invariants; progress_report(), report()["progress"]).
  P2  the lost-work budget: vfetch[(cohort key, layer)] holds the (batch slot, head, block) of every SUCCESSFUL new
      verify issue while that obligation is live. In _issue, after the same-call, resident and in-flight shortcuts and
      before any verifier eviction, credit, slot, sequence or transport change, a second verify fetch ATTEMPT of a recorded
      id is counted (verify_refetch_attempts); in checked mode (check_invariants and no Mutations switch) it raises
      OwnershipViolation, so no second transfer exists; in unchecked mode the issue proceeds as before. Allocation
      refusals and demand joins record nothing. Records are dropped at the layer's execution, at the cohort's decision or
      cancellation and at the request end (metadata only: no reference, reader event, slot or release changes), so
      late arrivals cannot recreate them and a later request of the batch slot inherits none. At most one record per
      undecided cohort (its current layer) and at most H * K * S_max ids per record, so at most
      B * (ceil(LAMBDA / K) + 1) * H * K * S_max live ids; charged as release entries (records) and selection ids (ids).

M2 SLICE A / B1 (Config.physical = "arena"; default "per_stream" keeps every rule above unchanged; retroinfer-eval
docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 2, steps S1-S5 only, Codex approval
2026-10-14). The schedule, cadence, guards, retention, recency, rollback, decisions and counters are the logical ones
above; only the physical backing, the address of a read and WHICH event retires a position change.
  A  storage: per layer one arena (P_l * bs, D) of head-agnostic block positions. Config.arena_backing selects P_l:
     "conservative" (default) P_l = B * H * (C + X), the same amount as per stream, addressed through the map;
     "reduced" (S6, contract revision 3 section 10, Codex approval 2026-10-14) P_l = B * H * C + X, the logical bound
     (a SUFFICIENCY argument, not a measured fit; the logical C + X slot records per stream stay). Only the geometry
     differs; every rule, check and counter is the same. ownership.Arena keeps the position records and the generation
     mirror pgen_t; Stream.take binds the
     lowest FREE position, LayerPool.poll_releases unbinds (ownership docstring). _issue: the bind, the take, the
     Transfer (phys, pgen, reader_ev = the previous PHYSICAL occupant's last reader, Arena.forward) and its inflight
     registration are one unit after every refusal and check; the transport sees it only after the unit. _poll checks the
     binding at admission (and, checked mode, the S0 payload readback). Checked mode adds the arena check points and the
     end-of-call scan, and bound + E + G <= P_l after every bind (E: unconsumed credits, G: granted unissued fetches).
  B1 reader lifetime: an executor with reads_arena = True (required) reads the arena itself (executor docstring); Row.kv
     is still read through the binding at call time. ONE ordered common stream "compute" carries every arena read and
     every reader event; the executor's read_ev is installed on every slot and position read and on (key, l)
     IMMEDIATELY after joint_post, before _issue (whose _verifier_evict polls releases); nothing between capturing
     Row.addr and that install moves a slot to FREE (_draft_recency may release to RELEASING). A deferred read is
     validated with the strict identity check first (the generation it copied at execution, StaleReader) and then
     the bytes (ReadCorruption).
  K6 charges the allocated positions by state and the arena metadata at allocated capacity (ownership.MemoryAccountant).

M2 B2 (Config.recent; default "eager" keeps every rule above unchanged; retroinfer-eval
docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revisions 4-6, section 11, stages S7 / S8; Codex
approval 2026-10-14 of the source implementation). Both need physical "arena" (either backing). The eager dicts (r.pub,
r.draft_kv, c.ws pre, c.prov) stay the SEMANTIC reference that Row.kv is read from; the schedule, guards, cadence,
retention, attention, rollback and recency are unchanged, and ring counters go to report()["ring"] only.
  "ring"  recent exact, draft, pre and provisional KV also live in a pooled per-layer ring (ownership.RingPool) with
     per-request maps r.rmap[l][h][kind] -> (entry, rgen). Producers (_produce): the admission's final partial prompt
     block (EXACT), the draft row's phase 1 (DRAFT), the cohort's phase 1 (PRE); PRE -> PROV at the cohort's execution and
     PROV -> EXACT at publication are re-tags of the same entry. _rsync drains (LIVE -> DRAINING) what is no longer
     logically held: draft KV at supersession or rollback, PRE / PROV of decided or cancelled cohorts, EXACT at its
     block's D2H unless a WAITING workspace reads it (rev6 H1; an executed read is covered by the compute gate), and
     everything at request end. FREE only in RingPool.retire (host poll; backpressure). A bind without a FREE entry
     retires, then host-waits the oldest DRAINING entry's events (ring_backpressure_waits, traced), then raises
     CapacityRefusal when none is DRAINING. Checked mode checks LIVE <= R_log per stream and <= B*H*R_log per layer after
     EVERY bind (admission included). Interface v3: Row.ring, read_ev also for ring-only calls, installed as the compute
     gate of every entry read; deferred ring reads are validated by identity (rgen) and then bytes. Admission: one buffer
     A_r per request (the prefill output; charged until its ready, compute and D2H events completed), complete prompt
     blocks go by D2H from A_r, only the final partial block binds entries. _history gathers EXACT entries on "d2h"
     after waiting their ready events and installs hist_ev as their D2H gate.
  "eager_matched"  the parity reference: eager storage, plus marker events at the same producer points kept in the same
     maps, and the same D2H stream waits on them (the dependency edges D1 / D2 of 11.4), the same snapshot and owner
     record; no ring, no backpressure.
  Both (checked mode): an ordered per-publication SNAPSHOT of the destination host rows on "d2h" right after hist_ev,
  validated without a host wait (identity of every gathered ring entry, then the bytes); at the first poll at which a
  publication's hist_ev is complete, whether the live rows still equal it is RECORDED (d2h_seen, a witness only); the
  host-row owner record hown is checked before every pack (11.3 (3)). These are instrumentation, not charged in K6.

C1.1 ABI v4 (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 10, 12.3 A0-A9
and 12.3.1 V1-V7; Codex approval 2026-10-15 of the portable checkpoint C1.1 only). Opt-in by the executor (abi = 4,
executor docstring); every ABI v3 path, store, counter and trace field is unchanged without it. Needs physical "arena" and
recent "ring" or "eager_matched", and a transport with carries_bias. The schedule, guards, cadence, retention, attention,
rollback and recency are unchanged; C1.1 still computes outputs from Row.kv (no physical consumption, no model numerics).
  bias      every KV item is (k, v, c): r.pub, draft KV, the cohort's pre (q, k, v, c) and provisional KV, Row.kv, and
            published. Stores (12.3 A4, int64 on the CPU, POISON-filled): host_c (B, S_host, H), arena_c (P_l * bs,),
            ring_c (Q_l,) (ring), A_r's c (P0 - 1, H). The bias moves INSIDE the same transaction as its K / V: the ring
            producer writes it before ready_ev (after the previous occupancy's D2H wait); _d2h_block copies it on "d2h"
            before hist_ev (after the same ready waits) and snapshots it; the transports carry it (transfer docstring:
            S0's completion frontier is after the bias scatter); the S0 readback, the arena / ring deferred-read
            validators (identity first, then bytes) and the D2H snapshot compare it; hown covers the host rows it lives
            in. The executor must register one bias read per (row, head, block) of Row.addr and one per Row.ring entry.
  chain     Request chains use CompressedChain(stride, reduce, kernel=chain_kernel) with (k, c) elements (V2). Draft rows
            select over r.dchain as before (count == pos); verifier rows use compressed.row_prefixes (V3: each row over
            the chain at its own position, then its element; KStepError "C1 row prefix: ..."). Rollback copies and the
            full-acceptance rebuild carry the overlap buffer (V4); rollback_across_compression uses the v4 event rule.
  records   (labelled instrumentation, not charged in K6) the windows each draft row's select received per (l, h)
            (Draft.cks; trace field "cks" of a draft record, ABI v4 only); c1stat counters; report()["physical"]["bias"]
            (bias_report): the ACTUAL persistent bias store bytes (numel * element_size at allocated capacity: every
            arena / ring row whatever its state, the host rows, the held admission buffers and their peak), apart from
            the LOGICAL K6 units (Geometry.bias_elem = the actual element size; ownership.Geometry.unit).
  ledger    (Codex correction of 2026-10-15) every queued OPERATIONAL bias backing is owned by self.bias_ledger
            (ownership.BiasOpLedger) from its allocation to its release: a ring producer's fresh bias (until its
            ready_ev), A_r's per-layer bias source (until A_r's ready event), an eager_matched D2H source clone (until
            hist_ev), each S0 issue's bias stage and landing (until that issue's frontier event) and each adversarial
            payload's bias rows (until its landing). Release is a PASSIVE check of those existing events at the host
            poll, at each K6 sample and in finish(); no wait or stream advancement is added. The K6 category
            bias_staging charges the held bytes at each sample; the ledger reports its exact allocation-transition
            peak. Diagnostic copies and synchronous eager-reference temporaries are excluded by name (BIAS_EXCLUDED).

C1.2 SERIALIZED PHYSICAL CONSUMPTION (Config.consume = "physical"; default "none" keeps every path above unchanged;
retroinfer-eval contract revision 11, 12.9; Codex approval 2026-10-15 of the portable slice only). Needs ABI v4, physical
"arena", recent "ring" and an executor with consumes_physical = True (executor docstring). At each joint call with rows
(_call): (1) joint_post_consume: the same validator reads as joint_post, then one gather per (row, head) on "compute"
in the declared order; the gather backings are owned by self.consume_ledger at once; (2) the same registration checks
(_check_post_reads), then the reader gates (_install_reader) and the validators (_arm_validators) BEFORE any wait;
(3) a NEW per-call host wait on the call's completion event: serialized, it changes completion order, so no B2
schedule parity is claimed; (4) check_consumed_identity BEFORE any value is used (StaleReader "stale consumption");
(5) resolve (from the gathered bytes and the rows' own q / hidden only) and the exact per-head / output comparison
with the same-order Row.kv reference (ConsumptionMismatch); (6) release of the call's gather backings and of the handle;
(7) the CONSUMED outputs are published (consumed output is what continues). K6 category consume_staging is the sampled
held gather bytes (0 at every sample: each call releases its own); report()["physical"]["consume"] gives the exact
peak, counters and the first consumed item of each provenance kind (instrumentation). No asynchronous decision callback
and no overlap claim.

C81 / K3 (Config.progress_guard = "rev6"; default "pin" keeps every rule above unchanged; retroinfer-eval
docs/superpowers/specs/2026-10-16-kstep-c81-k3-progress.md; a CANDIDATE, unpromoted; portable CPU toy evidence only,
tests/test_kstep_c81.py). The pin guard
(construction C >= (2K + 1) * S_max and the _pins_high 'memory' stall) protects the M1 verifier eviction: it keeps an
unprotected LRU entry where the oldest cohort waits. Capacity mode rev6 never evicts for a verifier (G6): a verifier
takes FREE "v" positions under an all-or-nothing grant (R_ver >= N_max), and a draft step starts only with S_max step
credits within C + d_h (d_h >= W). The rev6 drafting argument (PROPOSAL section 6, P6 iii) bounds P <= C + K * S_max
after the drain against the cap C + d_h, so C cancels: in rev6 the pin guard protects nothing that the rev6 contract
does not already guarantee, for any C >= 1. Mode "rev6" therefore REPLACES it, explicitly: it needs capacity "rev6" and
pin_guard=False, and every rev6 guarantee refusal (d_h >= W, H * d_h <= X_d, R_ver >= N_max, C + X >= K * S_max) still
applies. ENTITLEMENT, not allocation: a stream of a request that holds a draft reservation may hold up to C + d_h
class "ad" positions (P + credits <= C + d_h; capacity_report()["guarantee"]["per_stream_budget"]). The allocated
historical backing does not change: B * H * (C + X) per layer (per_stream, arena conservative) or B * H * C + X (arena
reduced), X = X_d + R_ver; B * H * C + n_res * H * d_h + R_ver equals the reduced backing only when X_d has no unused
remainder. Ring / recent, admission, metadata, staging and model bytes are outside both. d_h >= W is SUFFICIENT. The
note shows that it is needed only conditionally (the rev6 rules themselves, SC2a; the capacity rule alone under late
arrivals); that is not a minimum for every policy-preserving rule. Counters (this mode only; sampled at the end of every
joint call, not maxima over every transition): stats rev6g_pin_guard_would_stall (draft steps the pin guard would have
stalled), rev6g_peak_positions, rev6g_peak_positions_over_C, rev6g_peak_v_used; capacity_report()["guarantee"].
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from . import accept as acc
from . import cadence as cad
from . import rng
from .backend import check_backend, ev_done
from .compressed import CompressedChain, row_prefixes
from .errors import (CapacityRefusal, ConfigError, ConsumptionMismatch, KStepError, NoProgress, OwnershipViolation,
                     ReadCorruption, StaleReader)
from .executor import (DRAFT, RESET_CONTRACT, VERIFY, Row, abi_of, check_consumed_identity, check_executor,
                       reset_contract_of, reset_methods_missing)   # HOOK-L4: the I10 readers (executor.py)
from .mutations import Mutations
from .result_tickets import ResultTickets, RowKey
# HOOK-L3: DRAINING is imported for quiesce_state / detach_completed_events (INTERFACE_BC3 I8)
from .ownership import (COUNTS, DRAINING, FREE, LIVE, LRU, PO, R_DRAFT, R_EXACT, R_PRE, R_PROV, RELEASING, RING_KINDS, VRES,
                        Arena, BiasOpLedger, Geometry, LayerPool, MemoryAccountant, RingPool)
from .transfer import Transfer

POISON = -7777777                         # no toy KV value is negative

PENDING, RUNNING, DONE, ACCEPTED, CANCELLED = "pending", "running", "done", "accepted", "cancelled"
_DISCARDED = object()                          # I17: the output slot of a row that its ticket discarded (stale)
CAPACITY_MODES = ("m1", "rev6")
PHYSICAL_MODES = ("per_stream", "arena")
ARENA_BACKINGS = ("conservative", "reduced")
RECENT_MODES = ("eager", "ring", "eager_matched")
CONSUME_MODES = ("none", "physical")
PROGRESS_GUARDS = ("pin", "rev6")


@dataclass
class Config:
    K: int
    LAMBDA: int
    B: int
    C: int
    X: int
    R_ver: int
    S_host: int
    mode: str = acc.GREEDY
    seed: int = 0
    geom: Optional[Geometry] = None
    noprogress_passes: int = 6
    check_invariants: bool = True
    drain_order: Tuple[str, ...] = ("copy", "scatter", "compute", "d2h")
    pin_guard: bool = True                 # memory refusal of drafting that guarantees head-cohort progress (see below)
    pin_row_bound: Optional[int] = None    # blocks one (row, head) may select; default executor.max_sel
    capacity: str = "m1"                   # "m1" (the M1 rules) | "rev6" (PROPOSAL rev 6 contract; see the docstring)
    d_share: Optional[int] = None          # rev6: per-head draft share d_h; default W = (K + 1) * S_max; < W is refused
    physical: str = "per_stream"           # "per_stream" (M1 backing) | "arena" (M2 slice A / B1; see the docstring)
    arena_backing: str = "conservative"    # arena mode: "conservative" P_l = B*H*(C + X) | "reduced" P_l = B*H*C + X (S6)
    recent: str = "eager"                  # M2 B2: "eager" | "ring" | "eager_matched" (parity reference); see the docstring
    ring_slack: Optional[int] = None       # M2 B2 ring: S_ring >= 1 (default B * H); Q_l = B * H * R_log + S_ring
    consume: str = "none"                  # C1.2: "none" | "physical" (serialized physical consumption; see the docstring)
    progress_guard: str = "pin"            # "pin" (the pin guard, as before) | "rev6" (the rev6 guarantee REPLACES it; C81)
    reader_frontier: str = "single"        # HOOK-C: "single" (one ordered reader stream, as before) | "joined" (frontier.py)
    store: str = "none"                    # HOOK-B2: "none" (the int64 stores, as before) | "tensor" (tensor_store.py)
    store_join: str = "join"               # HOOK-B2: the backend stream of the admission gate (store "tensor" only)
    store_gen: str = "host"                # HOOK-L6 (I11): "host" (the consumer reads the pools' mirrors, as before) |
                                           # "consumer" (the store's own consumer mirrors; store "tensor", arena, S0)
    result_tickets: int = 0                # HOOK-L8 (I14): 0 (the serialized consumer, as before) | > 0: the capacity
                                           # of the opt-in result tickets (consume "physical"; DispatcherTicketView)
    deferred_calls: str = "off"            # I17 (lead): "off" (every call finishes inside _call, as before) | "one"
                                           # (at most ONE pending physical layer call; result_tickets >= 1)
    store_handback: bool = False           # I17 / I16: forward the store's producer-workspace handbacks to an
                                           # executor that declares accepts_handback (store "tensor")


class Draft:
    __slots__ = ("s", "pos", "tok_in", "epoch", "n_at", "m", "sel", "avail", "hidden", "d", "q", "state", "step", "cks",
                 "next_layer")

    def __init__(self, s, pos, tok_in, epoch, n_at, m, step):
        self.s, self.pos, self.tok_in, self.epoch, self.n_at, self.m, self.step = s, pos, tok_in, epoch, n_at, m, step
        self.sel, self.avail = {}, {}
        self.cks = {}                          # C1.1 ABI v4 instrumentation: per layer the windows select received
        self.hidden = None
        self.d, self.q = None, None
        self.state = RUNNING
        self.next_layer = 0                    # I17 (lead): the layer whose output the draft row publishes next


class Cohort:
    def __init__(self, req, m: int, a: int, b: int, step: int):
        self.uid, self.epoch, self.m, self.a, self.b = req.uid, req.epoch, m, a, b
        self.key = (req.uid, req.epoch, m)
        self.E = req.E
        self.positions = [req.E + i - 1 for i in range(a, b + 1)]
        self.tokens = [req.u(i - 1) for i in range(a, b + 1)]
        self.state = PENDING
        self.trigger_step = step
        self.join_step = None                  # repair 1 R3.1: the step of the PENDING -> RUNNING transition
        self.next_layer = 0
        self.hidden: List[Any] = []
        self.ws: Dict[int, Dict] = {}
        self.prov: Dict[int, List[Dict[int, Tuple]]] = {}
        self.chain: Dict[int, List[CompressedChain]] = {}
        self.exec_call: Dict[int, int] = {}
        self.vsel: Dict[int, List] = {}
        self.p: Dict[int, List[Fraction]] = {}
        self.waits = 0
        self.row_layers = 0
        self.decided_step = None
        self.granted = set()                   # rev6 G3: the layers at which this cohort holds a verifier grant
        self.vhist: Dict[int, int] = {}       # rev6 H1: the frozen history boundary of each executed layer

    def __repr__(self):
        return "Cohort(uid=%s e=%d m=%d [%d,%d] %s L%d)" % (self.uid, self.epoch, self.m, self.a, self.b, self.state, self.next_layer)


class Request:
    def __init__(self, uid, b: int, prompt: List[int], N: int, L: int, H: int, stride: int, reduce, kernel: int = None):
        self.uid, self.b, self.prompt, self.N = uid, b, list(prompt), int(N)
        self.P0 = len(prompt)
        self.toks = list(prompt)
        self.n = self.P0 - 1
        self.epoch = 0
        self.E = self.n
        self.drafts: List[Draft] = []
        self.cohorts: List[Cohort] = []
        self.n_trig = 0
        self.drafting_done = False
        self.done = False
        self.pub = [[dict() for _ in range(H)] for _ in range(L)]
        self.draft_kv = [[dict() for _ in range(H)] for _ in range(L)]
        self.comp = [[CompressedChain(stride, reduce, kernel=kernel) for _ in range(H)] for _ in range(L)]
        self.dchain = None
        self.hist_blocks = 0
        self.hist_ev: Dict[Tuple[int, int], Any] = {}
        self.cur: Optional[Draft] = None
        self.credits: Optional[List[List[int]]] = None   # rev6 SC2: step credits [layer][head] of the current draft step
        self.pub_hold: Dict[Tuple[int, int, int], set] = {}  # rev6 H1: (l, h, pos) below the boundary -> holder keys
        self.rmap: Optional[List[List[Dict[str, Dict]]]] = None  # M2 B2: [l][h][kind] -> key -> (entry, rgen) / marker
        self.hist_pub: Dict[Tuple[int, int], int] = {}      # M2 B2: (l, block) -> publication sequence (hown)
        self.inc = 0                                         # HOOK-C: I3 incarnation (given at submit; 0 = none)

    def u(self, i: int) -> int:
        return self.toks[self.E] if i == 0 else self.drafts[i - 1].d

    def lead(self) -> int:
        return cad.lead(len(self.drafts), self.n - self.E)

    def last_pos(self) -> int:
        return self.P0 + self.N - 1

    def generated(self) -> int:
        return len(self.toks) - self.P0


class DispatcherTicketView:
    """HOOK-L8 (INTERFACE_BC6 I14, lead, 2026-10-17): the view protocol of result_tickets.ResultTickets over the
    serialized physical consumer of ONE dispatcher (Config.result_tickets > 0). Every method does what the serialized
    Dispatcher._consume does at the same point, so the run is the serialized run:
      done(ev)          ev_done (never waits);
      check(handle)     the C1.2 identity check of every gathered block / entry (check_consumed_identity, StaleReader),
                        counted in c1stat["consume_identity_checks"] as the serialized path counts it;
      resolve / reference  the executor's (gathered bytes only / the same-order Row.kv reference);
      live(key)         READ-ONLY: the row still exists with the same incarnation and epoch; a verifier row also with
                        the same cohort and layer progress; a draft row with the same draft (m), position and layer
                        progress (Draft.next_layer, I17);
      apply(key, i, out)  row i's consumed output into out[(call, i)], once (a second apply is KStepError); _call
                        publishes from there;
      release(holds, handle)  the call's c1_gather holds (the consume ledger), then handle.clear().
    Nothing here touches self.d.stats. No method keeps a reference to a row, a request, a cohort or a workspace.
    Milestone 7 (I15 / I17): every call reserves its ticket capacity at its start (Dispatcher._call). With tickets on,
    deferred or not, a failure after that reservation and before the commit leaves the reservation OPEN, so a
    QuiescentReset refuses (state:tickets, tickets:reservations) until the root decides; in milestone 6 such a failure
    left no ticket state. This is fail closed, not a cleanup."""

    def __init__(self, d: "Dispatcher"):
        self.d = d
        self.out: Dict[Tuple[Any, int], Any] = {}

    def done(self, ev) -> bool:
        return ev_done(ev)

    def check(self, handle) -> None:
        d = self.d
        d.c1stat["consume_identity_checks"] += check_consumed_identity(d.ex.consumed_identities(handle))

    def resolve(self, handle):
        return self.d.ex.resolve(handle)

    def reference(self, handle):
        return self.d.ex.reference(handle)

    def live(self, key) -> bool:
        """READ-ONLY. The request key.uid is active (not done) with incarnation key.inc; a verifier row: its cohort
        key.cohort is RUNNING with epoch key.epoch and next_layer key.layer (not yet published at that layer); a draft
        row: the request is at epoch key.epoch and its current draft is RUNNING, the same draft (key.cohort ==
        (uid, epoch, m)) at position key.pos, with next_layer key.layer (I17: Draft.next_layer, the per-layer marker
        that _call_publish advances, so a deferred call's draft row is live only at its own layer). I20: never live once
        the request's cancellation is requested (the lead's invalidation; C's ticket mark discards such a row without
        asking, I19)."""
        r = self.d.reqs.get(key.uid)
        if r is None or r.done or r.inc != key.inc or key.uid in self.d.cancel_req:
            return False
        if key.kind == VERIFY:
            return any(c.key == key.cohort and c.epoch == key.epoch and c.state == RUNNING and c.next_layer == key.layer
                       for c in r.cohorts)
        dr = r.cur
        return (r.epoch == key.epoch and dr is not None and dr.state == RUNNING and dr.pos == key.pos
                and key.cohort == (r.uid, r.epoch, dr.m) and dr.next_layer == key.layer)

    def apply(self, key, i: int, out) -> None:
        k = (key.call, i)
        if k in self.out:
            raise KStepError("HOOK-L8 (I14): row %d of call %s (%s) applied twice" % (i, key.call, key))
        self.out[k] = out

    def release(self, holds, handle) -> None:
        lg = self.d.consume_ledger
        for h in holds:
            lg.release_key("c1_gather", h)
        handle.clear()                                                 # the handle's backing references are dropped


class Dispatcher:
    def __init__(self, cfg: Config, executor, be, make_transport: Callable, mut: Optional[Mutations] = None,
                 progress: Optional[Callable] = None, observer: Optional[Callable] = None):
        check_executor(executor)
        check_backend(be)
        self.K, self.LAMBDA = cad.check(cfg.K, cfg.LAMBDA)
        if cfg.mode not in acc.MODES:
            raise ConfigError("mode must be one of %s" % (acc.MODES,))
        self.cfg, self.ex, self.be = cfg, executor, be
        self.mut = mut or Mutations()
        self.progress = progress
        self.observer = observer
        ex = executor
        self.L, self.H, self.D, self.bs, self.stride = ex.L, ex.H, ex.D, ex.bs, ex.stride
        self.rb = int(cfg.pin_row_bound or ex.max_sel)
        if cfg.progress_guard not in PROGRESS_GUARDS:
            raise ConfigError("progress_guard must be one of %s (got %r)" % (PROGRESS_GUARDS, cfg.progress_guard))
        self.rev6_guard = cfg.progress_guard == "rev6"        # C81 / K3 (module docstring): a replacement, not a removal
        if self.rev6_guard and (cfg.capacity != "rev6" or cfg.pin_guard):
            raise ConfigError("progress_guard 'rev6' replaces the pin guard by the rev6 guarantee: it needs capacity 'rev6' "
                              "and pin_guard=False (got capacity %r, pin_guard %r)" % (cfg.capacity, cfg.pin_guard))
        need = self.K * self.rb
        if cfg.C + cfg.X < need:
            raise ConfigError("K5/D3: C + X = %d < K * max_sel = %d: one cohort's exact selection cannot be resident at once"
                              % (cfg.C + cfg.X, need))
        if cfg.pin_guard and cfg.C < (2 * self.K + 1) * self.rb:
            raise ConfigError("D4 progress guarantee: C = %d < (2K + 1) * max_sel = %d (pin_guard=True; see Dispatcher docs)"
                              % (cfg.C, (2 * self.K + 1) * self.rb))
        if cfg.capacity not in CAPACITY_MODES:
            raise ConfigError("capacity must be one of %s (got %r)" % (CAPACITY_MODES, cfg.capacity))
        if cfg.physical not in PHYSICAL_MODES:
            raise ConfigError("physical must be one of %s (got %r)" % (PHYSICAL_MODES, cfg.physical))
        self.arena_mode = cfg.physical == "arena"
        if cfg.arena_backing not in ARENA_BACKINGS:
            raise ConfigError("arena_backing must be one of %s (got %r)" % (ARENA_BACKINGS, cfg.arena_backing))
        if cfg.arena_backing != "conservative" and not self.arena_mode:
            raise ConfigError("arena_backing %r needs physical 'arena' (got %r)" % (cfg.arena_backing, cfg.physical))
        if self.arena_mode and not getattr(executor, "reads_arena", False):
            raise ConfigError("M2 arena mode needs an executor with reads_arena = True (it reads the arena itself and "
                              "returns its read event; executor docstring)")
        if cfg.recent not in RECENT_MODES:
            raise ConfigError("recent must be one of %s (got %r)" % (RECENT_MODES, cfg.recent))
        self.recent = cfg.recent
        self.ring_mode = cfg.recent == "ring"
        if self.recent != "eager" and not self.arena_mode:
            raise ConfigError("M2 B2: recent %r needs physical 'arena' (got %r)" % (cfg.recent, cfg.physical))
        if self.ring_mode and not getattr(executor, "reads_ring", False):
            raise ConfigError("M2 B2 ring mode needs an executor with reads_ring = True (interface v3; executor docstring)")
        if cfg.ring_slack is not None and (not self.ring_mode or int(cfg.ring_slack) < 1):
            raise ConfigError("M2 B2: ring_slack %r needs recent 'ring' and ring_slack >= 1" % (cfg.ring_slack,))
        self.abi4 = abi_of(executor) == 4                     # C1.1 ABI v4 (module docstring); check_executor ran V5
        self.chain_kernel = int(executor.chain_kernel) if self.abi4 else self.stride
        if self.abi4 and (not self.arena_mode or self.recent not in ("ring", "eager_matched")):
            raise ConfigError("C1.1 ABI v4 needs physical 'arena' and recent 'ring' or 'eager_matched' (got %r, %r)"
                              % (cfg.physical, cfg.recent))
        if cfg.consume not in CONSUME_MODES:
            raise ConfigError("consume must be one of %s (got %r)" % (CONSUME_MODES, cfg.consume))
        self.consume = cfg.consume == "physical"               # C1.2 (12.9.2): the serialized physical consumer
        if self.consume and not (self.abi4 and self.ring_mode and getattr(executor, "consumes_physical", False)):
            raise ConfigError("C1.2 consume 'physical' needs an ABI v4 executor with consumes_physical = True, physical "
                              "'arena' and recent 'ring' (got %r, %r)" % (cfg.physical, cfg.recent))
        self.rev6 = cfg.capacity == "rev6"
        self.W = (self.K + 1) * self.rb                       # rev6 L3: positions through the next trigger, per stream
        self.X_d = cfg.X - cfg.R_ver                          # rev6 L1: the shared draft pool per layer
        self.N_max = self.H * self.K * self.rb                # rev6 section 5: one cohort's largest residual footprint
        self.d_share = 0
        if self.rev6:
            d = self.W if cfg.d_share is None else int(cfg.d_share)
            if d < self.W:
                raise ConfigError("rev6 L4: per-head draft share d_h = %d < W = (K + 1) * S_max = %d (guarantee mode)"
                                  % (d, self.W))
            if self.H * d > self.X_d:
                raise ConfigError("rev6 L4: one request needs H * d_h = %d shared draft positions per layer > X_d = X - R_ver"
                                  " = %d (H %d, d_h %d)" % (self.H * d, self.X_d, self.H, d))
            if cfg.R_ver < self.N_max and not self.mut.rev6_admit_small_reserve:
                raise ConfigError("rev6 G5: R_ver = %d < N_max = H * K * S_max = %d (guarantee mode)" % (cfg.R_ver, self.N_max))
            self.d_share = d
        self.age: Dict[Any, int] = {}              # uid -> batch admission order (rev6 request age)
        self.dres: Dict[Any, bool] = {}            # rev6 L4: uid -> reservation held (True: active, False: ended, draining)
        self.dres_b: Dict[Any, int] = {}           # rev6: batch slot of an ended request whose reservation drains
        self.grant_log: List[Tuple] = []           # rev6 G3: (call, layer, uid, cohort key, new fetches, free V after)
        self.evict_log: List[Tuple] = []           # (call, layer, b, h, block, protected, kinds, where): every LRU eviction
        self.late_log: List[Tuple] = []            # (call, layer, b, h, block, displaced (block, protected, kinds) or None)
        self.res_log: List[Tuple] = []             # rev6 L4: (step, uid) of every reservation
        self.h1_reads: Dict[Tuple, Tuple] = {}     # rev6 H1: (key, l) -> (frozen boundary, selection) read, gate pending
        self.h1_log: List[Tuple] = []              # rev6 H1: ("retain", step, uid, l, h, block, holder states)
        self._rev6_stalled = False
        self.pools = [LayerPool(l, cfg.B, self.H, cfg.C, cfg.X, cfg.R_ver) for l in range(self.L)]
        n_pos = self.pools[0].n_pos
        self.n_pos = n_pos
        z = lambda S: torch.full((cfg.B, S, self.H, self.D), POISON, dtype=torch.int64)
        self.host_k = [z(cfg.S_host) for _ in range(self.L)]
        self.host_v = [z(cfg.S_host) for _ in range(self.L)]
        if self.arena_mode:                                    # M2: conservative B*H*(C + X) or reduced B*H*C + X (S6)
            self.P = cfg.B * self.H * n_pos if cfg.arena_backing == "conservative" else cfg.B * self.H * cfg.C + cfg.X
            self.arenas = [Arena(l, self.P, checked=self._checked()) for l in range(self.L)]
            for pool, ar in zip(self.pools, self.arenas):
                pool.attach_arena(ar)
            za = lambda: torch.full((self.P * self.bs, self.D), POISON, dtype=torch.int64)
            self.arena_k = [za() for _ in range(self.L)]
            self.arena_v = [za() for _ in range(self.L)]
            self.slots_k = self.slots_v = None
            executor.attach_arena(be, self.arena_k, self.arena_v, [ar.pgen_t for ar in self.arenas], self._register_read)
        else:
            self.P, self.arenas, self.arena_k, self.arena_v = None, None, None, None
            self.slots_k = [z(n_pos * self.bs) for _ in range(self.L)]
            self.slots_v = [z(n_pos * self.bs) for _ in range(self.L)]
        # M2 B2 (contract 11.2): R_log = (LAMBDA + bs - 1) EXACT + (LAMBDA + 1) DRAFT + (LAMBDA + 1) PRE or PROV per stream
        self.R_log = (self.LAMBDA + self.bs - 1) + 2 * (self.LAMBDA + 1)
        if self.ring_mode:
            self.S_ring = cfg.B * self.H if cfg.ring_slack is None else int(cfg.ring_slack)
            self.Q = cfg.B * self.H * self.R_log + self.S_ring
            self.rings = [RingPool(l, self.Q, checked=self._checked()) for l in range(self.L)]
            zr = lambda: torch.full((self.Q, self.D), POISON, dtype=torch.int64)
            self.ring_k = [zr() for _ in range(self.L)]
            self.ring_v = [zr() for _ in range(self.L)]
            executor.attach_ring(be, self.ring_k, self.ring_v, [rp.rgen_t for rp in self.rings], self._register_ring_read)
        else:
            self.S_ring = self.Q = None
            self.rings, self.ring_k, self.ring_v = [], None, None
        self.c1stat = defaultdict(int)             # C1.1: bias / chain coverage counters (report()["physical"]["bias"])
        self.adm_bias_peak_bytes = 0               # C1.1: the EXACT peak of the held A_r bias bytes (at allocations)
        if self.abi4:                              # C1.1 A4: the bias stores beside their K / V stores (POISON-filled)
            self.bias_dtype = torch.int64
            self.host_c = [torch.full((cfg.B, cfg.S_host, self.H), POISON, dtype=self.bias_dtype) for _ in range(self.L)]
            self.arena_c = [torch.full((self.P * self.bs,), POISON, dtype=self.bias_dtype) for _ in range(self.L)]
            self.ring_c = ([torch.full((self.Q,), POISON, dtype=self.bias_dtype) for _ in range(self.L)]
                           if self.ring_mode else None)
            executor.attach_bias(be, self.arena_c, self.ring_c)
            self.bias_ledger = BiasOpLedger()      # C1.1 A9: the queued operational bias backings (ownership docstring)
        else:
            self.bias_dtype = self.host_c = self.arena_c = self.ring_c = self.bias_ledger = None
        self.consume_ledger = BiasOpLedger(BiasOpLedger.CONSUME_KINDS) if self.consume else None   # C1.2: the gathers
        rtc = cfg.result_tickets                   # HOOK-L8 (INTERFACE_BC6 I14, lead): the opt-in result tickets
        if isinstance(rtc, bool) or not isinstance(rtc, int) or rtc < 0:
            raise ConfigError("HOOK-L8: result_tickets must be an int >= 0 (got %r)" % (rtc,))
        if rtc and not self.consume:
            raise ConfigError("HOOK-L8: result_tickets %d needs consume 'physical' (got %r)" % (rtc, cfg.consume))
        self.tickets = ResultTickets(rtc, DispatcherTicketView(self)) if rtc else None
        if cfg.deferred_calls not in ("off", "one"):             # I17 (lead): the one-outstanding-call boundary
            raise ConfigError("I17: deferred_calls must be 'off' or 'one' (got %r)" % (cfg.deferred_calls,))
        self.deferred = cfg.deferred_calls == "one"
        if self.deferred and self.tickets is None:
            raise ConfigError("I17: deferred_calls 'one' needs consume 'physical' and result_tickets >= 1")
        self.pending_call: Optional[Dict[str, Any]] = None       # I17: the ONE submitted, unfinished layer call
        self.pending_step: Optional[Dict[str, Any]] = None       # I17: the step that waits on it (resumed by step())
        self.dstat = defaultdict(int)                            # I17: deferred counters (report()["deferred"])
        self.cancel_req: Dict[Any, int] = {}       # I20 (lead): uid -> inc of a cancellation requested during a stopped step
        self.cancelled: Dict[Any, Dict[str, Any]] = {}   # I20: uid -> the record of a cancelled incarnation (no results)
        self.cstat = defaultdict(int)              # I20: cancellation counters (report()["cancel"], after a first cancel)
        self._in_step = False                      # I20: cancel() is a driver operation, refused inside step()
        self.consume_first: Dict[str, Tuple] = {}  # C1.2 instrumentation: per provenance kind its first consumed item
        self.rstat = defaultdict(int)              # M2 B2: ring / D2H-check counters (report()["ring"]; never self.stats)
        self.adm: List[Dict] = []                  # M2 B2 ring: the admission buffers A_r that are not FREE
        self.adm_log: List[Dict] = []              # M2 B2: (step, uid, b, P0, LIVE at the admission start) per admission
        self.adm_peak = 0                          # M2 B2 ring: the EXACT peak of the held A_r items (at allocations)
        self.d2h_checks: List[Dict] = []           # M2 B2 checked mode: publications whose snapshot is not validated yet
        self.d2h_seen: List[Dict] = []             # M2 B2: per publication, the live rows at its first complete poll
        self.hown: Dict[Tuple[int, int, int], Tuple] = {}   # M2 B2: (l, b, block) -> (uid, publication) of the host rows
        self.pub_seq = 0
        self.ring_waits: List[Dict] = []           # M2 B2 ring: every backpressure host wait (trace)
        self._ring_reads: List[Tuple[int, int, int]] = []   # the (layer, entry, rgen) read by the current call's rows
        self._ring_exp: Dict[Tuple, Tuple] = {}    # (id(row), head, position) -> the eager (k, v) of the current call
        self._n_ring_val = 0
        self.store = None                                              # HOOK-B2: Config.store (tensor_store.py)
        if cfg.store != "none":                                        # HOOK-B2: the tensor stores replace the int64
            from .tensor_store import attach_dispatcher               # HOOK-B2: stores and re-attach the executor,
            self.store = attach_dispatcher(self, cfg, executor, be)   # HOOK-B2: BEFORE the transport takes them
        self.tp = make_transport(self)
        if self.store is not None:                                     # HOOK-B2: the transport must carry the store's
            self.store.check_transport(self.tp)                        # HOOK-B2: dtype (S0 stages int64 by default)
        geom = cfg.geom
        if self.abi4:
            if not getattr(self.tp, "carries_bias", False):
                raise ConfigError("C1.1 ABI v4 needs a transport that carries the bias rows (carries_bias; transfer.py)")
            el = torch.empty((), dtype=self.bias_dtype).element_size()   # A9: the ACTUAL bias element size
            g0 = geom if geom is not None else Geometry()
            if g0.bias_elem not in (0, el):
                raise ConfigError("C1.1 A9: Geometry.bias_elem %d != the bias dtype's element size %d" % (g0.bias_elem, el))
            geom = replace(g0, bias_elem=el)
        self.acct = MemoryAccountant(geom, getattr(self.tp, "staging_bytes", 0), arena=self.arena_mode,
                                     ring=self.ring_mode, **(dict(consume=True) if self.consume else {}))
        if self.store is not None:                                     # HOOK-B2: + the K6 category source_staging
            self.acct.enable_sources()                                 # HOOK-B2: (the store's held source bytes)
        if cfg.store_gen not in ("host", "consumer"):                  # HOOK-L6 (I11)
            raise ConfigError("store_gen must be 'host' or 'consumer', not %r (I11)" % (cfg.store_gen,))
        if cfg.store_gen == "consumer":                                # HOOK-L6 (I11): the store's consumer mirrors,
            if self.store is None or not self.arena_mode or not getattr(self.tp, "accepts_gen_publisher", False):
                raise ConfigError("store_gen 'consumer' needs store 'tensor', physical 'arena' and the S0 transport "
                                  "(I11; store %r, physical %r, transport %s)"
                                  % (cfg.store, cfg.physical, type(self.tp).__name__))
            self.store.attach_gen_mirrors(self, self.tp)               # HOOK-L6: re-attaches the executor with them and
            self.acct.enable_gen_mirror()                              # installs tp.gen_publisher; K6 gen_mirror
        self.handback_to = None                                        # I17 / I16: the executor that takes handbacks
        if cfg.store_handback:
            if self.store is None:
                raise ConfigError("I17: store_handback needs store 'tensor' (got %r)" % (cfg.store,))
            self.store.enable_handback()                               # I16 (track B): one Handback per producer op
            if vars(type(executor)).get("accepts_handback", False):    # declared in the executor's OWN class body
                if not callable(getattr(type(executor), "on_handback", None)):
                    raise ConfigError("I17: %s declares accepts_handback but has no callable on_handback"
                                      % type(executor).__name__)
                self.handback_to = executor
        self.session = "open"                                          # HOOK-L7 (I12): "open" | "resetting" | "failed"
        self.session_reason: Optional[str] = None                      # HOOK-L7: the failed phase and its error
        self.queue: List[Tuple] = []
        self.slot_req: List[Optional[Request]] = [None] * cfg.B
        self.reqs: Dict[Any, Request] = {}
        self.t = 0
        self.call_id = 0
        self.seq = 0
        self.inflight: Dict[int, Transfer] = {}
        self.validators: List[Tuple] = []
        self.key_readers: Dict[Tuple, List] = {}
        self.pending_release: List[Tuple] = []
        self.stats = defaultdict(int)
        self.stalls: List[Tuple] = []
        self.exec_log: List[Tuple] = []            # (call, layer, uid, epoch, m)
        self.share: List[Tuple] = []               # (call, layer, b, h, rows, sum, union, sum_hist, union_hist)
        self.trace: Dict[Any, Dict] = {}
        self.published: Dict[Tuple, Tuple] = {}    # (uid, pos, l, h) -> (k, v)
        self.pub_log: List[Tuple] = []             # (uid, pos), one entry per publication of a position
        self.results: Dict[Any, List[int]] = {}
        self.call_step: Dict[int, int] = {}        # call id -> decode step
        self.touch_log: List[Tuple] = []           # (call, layer, b, h, block, event): draft-only recency (R2.1)
        self.vread_log: List[Tuple] = []           # (call, layer, b, h, block, (m, pos)): verifier reads (R2.2)
        self.arrival_log: List[int] = []           # transfer seq in the order the host observed the arrivals
        self.admit_log: List[Tuple] = []           # (call, layer, b, h, block, demand, where, issuer kind) (R2.3)
        self.vev_log: List[Tuple] = []             # (call, layer, b, h, block, uid): verifier evictions (R2.5)
        self.rel_layers: Dict[Tuple, Dict[int, bool]] = {}  # key -> {layer: released}: queued layers (R1.4)
        self.cancelled_keys = set()
        self._noprog = 0
        self.vfetch: Dict[Tuple, set] = {}         # N3 P2: (cohort key, layer) -> {(b, h, block)} verify-fetched while live
        self.vfetch_ids = 0                        # N3 P2: live ids (sum of the record sizes)
        self.vfetch_max_ids = 0
        self._vfail = set()
        self._read_keys = set()
        self._n_new_val = 0
        self._n_bias_reads = 0                     # C1.1: arena bias reads the executor registered in the current call
        # HOOK-C: I3 incarnations (assigned at submit) and Config.reader_frontier. frontier.py is imported only in
        # "joined" mode, so "single" loads, runs and reports exactly as before.
        self.inc_seq = 0
        self.inc_at_submit: Dict[Any, int] = {}
        if cfg.reader_frontier not in ("single", "joined"):
            raise ConfigError("reader_frontier must be one of ('single', 'joined') (got %r)" % (cfg.reader_frontier,))
        self.frontier = None
        if cfg.reader_frontier == "joined":
            from .frontier import CompletionOwnership
            self.frontier = CompletionOwnership(be, entry_bytes=self.acct.u["release_entry"])

    # ================================================================== requests
    def submit(self, uid, prompt: List[int], max_new_tokens: int) -> None:
        self._session_guard("submit")                                  # HOOK-L7 (I12): no admission unless "open"
        if uid in self.reqs or any(q[0] == uid for q in self.queue):
            raise ValueError("duplicate request uid %r" % (uid,))
        if len(prompt) < 1:
            raise ValueError("empty prompt")
        if len(prompt) + max_new_tokens > self.cfg.S_host:
            raise ConfigError("prompt %d + max_new_tokens %d > S_host %d" % (len(prompt), max_new_tokens, self.cfg.S_host))
        self.inc_seq += 1                                              # HOOK-C: I3, the incarnation of this submit
        self.inc_at_submit[uid] = self.inc_seq
        self.queue.append((uid, list(prompt), int(max_new_tokens)))

    def active(self) -> List[Request]:
        return [r for r in self.slot_req if r is not None]

    def _admit(self) -> None:
        for b in range(self.cfg.B):
            if self.slot_req[b] is not None or not self.queue:
                continue
            prep = None
            if self.store is not None:                                 # HOOK-L5 (I9.1): the opt-in executor's prefill
                prep = self.store.prepare_admit(self, self.queue[0][0], self.queue[0][1], b)   # result (slot b) is
            uid, prompt, N = self.queue.pop(0)                         # checked BEFORE this admission's mutation (None:
                                                                       # opt-out)
            r = Request(uid, b, prompt, N, self.L, self.H, self.stride, self.ex.compress, self.chain_kernel)
            r.inc = self.inc_at_submit.pop(uid, 0)                     # HOOK-C: I3 (0: not queued through submit)
            if self.frontier is not None:                              # HOOK-C2: the incarnation becomes live, the
                self.frontier.start_request(self, r)                   # providers bind, BEFORE the uid enters reqs
            self.reqs[uid] = r
            self.slot_req[b] = r
            self.age[uid] = len(self.age)
            for l in range(self.L):
                for h in range(self.H):
                    self.pools[l].stream(b, h).owner = uid
            kv = prep.kv if prep is not None else self.ex.prefill(prompt[:-1])   # HOOK-L5: ONE prefill either way
            for pos in range(len(prompt) - 1):
                for l in range(self.L):
                    for h in range(self.H):
                        r.pub[l][h][pos] = kv[pos][l][h]
                        r.comp[l][h].append(self._ck(kv[pos][l][h]))
            r.dchain = [[c.copy() for c in row] for row in r.comp]
            if self.recent == "eager":
                self._history(r)
            else:
                self._admit_recent(r, kv, prep)                        # M2 B2: A_r, its D2H, the partial-block entries
            self.trace[uid] = dict(prompt=list(prompt), N=N, b=b, epochs=[dict(epoch=0, E=r.E, u0=r.toks[r.E], drafts=[], cohorts=[], outcome=None)])
            self.stats["requests_admitted"] += 1
            if r.generated() >= r.N:
                self._end_request(r)

    def _ck(self, kvv):
        """The chain element of one KV item: its key (ABI v3), or the pair (k, c) of the same item (ABI v4, V2)."""
        return (kvv[0], kvv[2]) if self.abi4 else kvv[0]

    def _end_request(self, r: Request, cancelled: bool = False) -> None:
        """The end of a request. I20: cancelled=True (only from _cancel_teardown) records the incarnation in
        self.cancelled (inc, batch slot, step, the committed tokens) instead of a results entry, and counts
        stats["requests_cancelled"]; every other part of the end is the same."""
        if self.tickets is not None and self.tickets.holds_request(r.uid, r.inc):   # HOOK-L8 (I14), before any change
            raise OwnershipViolation("HOOK-L8 (I14): the end of request %s (incarnation %s) is refused: a pending or "
                                     "broken result ticket still names it (%s)" % (r.uid, r.inc, self.tickets.pending()))
        if self.frontier is not None:                                  # HOOK-C2: a second or stale end is refused
            self.frontier.end_entry(self, r)                           # before any mutation (I3, I6)
        for l in range(self.L):
            for h in range(self.H):
                self.pools[l].stream(r.b, h).end_owner()
        self.pending_release = [x for x in self.pending_release if x[0][0] != r.uid]
        self.key_readers = {k: v for k, v in self.key_readers.items() if k[0][0] != r.uid}
        for rk in [k for k in self.vfetch if k[0][0] == r.uid]:          # N3 P2 metadata only; a later request of
            self._vfetch_drop(rk, "end")                                # this batch slot inherits nothing
        self.rel_layers = {k: v for k, v in self.rel_layers.items() if k[0] != r.uid}
        r.done = True
        r.credits = None
        if r.pub_hold:                                                 # H1: no reader of an ended request remains
            self.stats["rev6_h1_end_cleared_positions"] += len(r.pub_hold)
        r.pub_hold.clear()
        self.h1_reads = {k: v for k, v in self.h1_reads.items() if k[0][0] != r.uid}
        if r.uid in self.dres:                                         # rev6 L4: returned once its slots are all FREE
            self.dres[r.uid] = False
            self.dres_b[r.uid] = r.b
        if r.rmap is not None:                                         # M2 B2: every LIVE entry -> DRAINING (owner kept);
            for l in range(self.L):                                    # pending D2H keeps its gate; A_r is unaffected
                for h in range(self.H):
                    for kind in RING_KINDS:
                        for key in list(r.rmap[l][h][kind]):
                            self._rdrop(l, r, h, kind, key, "end")
        if cancelled:                                                  # I20: no results entry for a cancelled request
            self.cancelled[r.uid] = dict(inc=r.inc, b=r.b, t=self.t, committed=list(r.toks[r.P0:]))
            self.trace[r.uid]["cancelled"] = True
        else:
            self.results[r.uid] = r.toks[r.P0:r.P0 + r.N]
        self.trace[r.uid]["final"] = list(r.toks)
        self.slot_req[r.b] = None
        self.stats["requests_cancelled" if cancelled else "requests_done"] += 1
        if self.frontier is not None:                                  # HOOK-C: its finished-cohort records are cleared
            self.frontier.end_request(self, r)                         # once every hold of the incarnation drained (I3)

    # ================================================================== the step
    def run(self, max_steps: int = 200000, on_pending: Optional[Callable] = None) -> Dict[Any, List[int]]:
        """The synchronous driver. I17: step() itself never waits on a pending call; when it returns with one pending,
        run() (the DRIVER, not the call path) advances the backend by ONE host wait on that call's event (counted in
        dstat["driver_waits"]) and steps again. I20: on_pending(dispatcher, info), when given, is called ONCE per
        pending call, after step() returned with it and BEFORE the driver's host wait; info = dict(cid, l, tid, uids:
        the sorted uids of the call's rows). It is a driver hook (for example cancel(uid)); it must not step or wait."""
        prev = None
        while self.queue or self.active() or self.pending_step is not None:
            self.step()
            pc = self.pending_call
            if pc is not None:
                if pc is prev:                                         # waited, polled, and still not finished: a
                    raise KStepError("I17: run(): the pending call %d stays unfinished after the driver's wait on its "
                                     "completed event (a broken or blocked ticket): no retry" % pc["cid"])
                if on_pending is not None:                             # I20: the driver's hook, before the wait
                    on_pending(self, dict(cid=pc["cid"], l=pc["l"], tid=pc["tid"],
                                          uids=sorted({row.uid for row in pc["ctx"]["jrows"]})))
                self.dstat["driver_waits"] += 1
                self.be.host_wait(pc["cev"])
            prev = pc
            if self.t > max_steps:
                raise KStepError("run: more than %d steps" % max_steps)
        self.finish()
        return self.results

    def pending_event(self) -> Any:
        """I17, READ-ONLY: the completion event of the ONE pending layer call, or None (for a test or driver that
        advances the deferred backend itself)."""
        return None if self.pending_call is None else self.pending_call["cev"]

    def cancel(self, uid) -> str:
        """I20 (lead, INTERFACE_BC8): an EXPLICIT cancellation of the ADMITTED request uid. A driver operation: between
        two steps, or while a deferred step is stopped (run()'s on_pending hook). Refused (KStepError) BEFORE any
        change: inside step(); a session that is not "open" (OwnershipViolation, HOOK-L7); an unknown uid; a uid that
        is queued but not admitted; a done (ended or cancelled) request; a second cancel; an incarnation that the
        tickets cannot name (inc < 1, a request not queued through submit); and, with no stopped step, an outstanding
        (broken) ticket that names it (no forced cleanup). Then frontier.cancel_entry (C, I19: ACTIVE -> CANCELLING;
        its own refusals also come before any change) and tickets.cancel(uid, inc) (C, I19: the discard mark).
          - A stopped deferred step (pending_step): the request is only MARKED (cancel_req; DispatcherTicketView.live
            is False for it from now on). NOTHING else changes: the pending call's holds, handle and backings stay
            charged until its event completes. _resume_step tears it down ONCE, after the call's ticket finished and
            BEFORE the call's publication. Returns "deferred".
          - Otherwise the teardown (_cancel_teardown) runs now. Returns "ended".
        The batch slot of a torn-down request is free for the next admission (a new incarnation; submit keeps refusing
        a known uid, so a cancelled uid is never resubmitted)."""
        self._session_guard("cancel")
        if self._in_step:
            raise KStepError("I20: cancel of uid %r refused: inside step() (a driver operation between steps)" % (uid,))
        r = self.reqs.get(uid)
        if r is None:
            if any(q[0] == uid for q in self.queue):
                raise KStepError("I20: cancel of uid %r refused: it is queued, not admitted" % (uid,))
            raise KStepError("I20: cancel of uid %r refused: unknown request" % (uid,))
        if r.done:
            raise KStepError("I20: cancel of uid %r (incarnation %s) refused: the request is done (%s)"
                             % (uid, r.inc, "cancelled" if uid in self.cancelled else "ended"))
        if uid in self.cancel_req:
            raise KStepError("I20: cancel of uid %r (incarnation %s) refused: its cancellation is already requested"
                             % (uid, r.inc))
        stopped = self.pending_step is not None
        if self.tickets is not None:
            if r.inc < 1:
                raise KStepError("I20: cancel of uid %r refused: incarnation %s (not queued through submit) cannot be "
                                 "named by the result tickets" % (uid, r.inc))
            if not stopped and self.tickets.holds_request(r.uid, r.inc):
                raise KStepError("I20: cancel of uid %r (incarnation %s) refused: an outstanding result ticket names it "
                                 "and no step is stopped (a broken ticket: no forced cleanup) (%s)"
                                 % (uid, r.inc, self.tickets.pending()))
        if self.frontier is not None:
            self.frontier.cancel_entry(self, r)                        # C (I19): ACTIVE -> CANCELLING
        n = self.tickets.cancel(r.uid, r.inc) if self.tickets is not None else 0   # C (I19): the discard mark
        self.cstat["requested"] += 1
        self.cstat["marked_tickets"] += n
        if stopped:
            self.cancel_req[uid] = r.inc
            self.cstat["deferred"] += 1
            return "deferred"
        self._cancel_teardown(r)
        return "ended"

    def _cancel_teardown(self, r: Request) -> None:
        """I20: the ONE teardown of a cancelled request, when no outstanding ticket names it any more. Every undecided
        cohort and every open untriggered key of the current epoch is cancelled as the rejection path cancels them (C3:
        the workspace cleared, the decision recorded through _finished_key (accepted while CANCELLING, I19), the
        requester cancelled (T3: an in-flight destination is never freed early), the cohort freed (K6), its
        verify-fetch metadata dropped); every uncommitted draft is recorded "cancelled" (the one in flight
        "cancelled_in_flight"); then the request ends with cancelled=True: its streams end their owner, its ring
        entries drain behind their gates, its batch slot is free, and it has no results entry. It never goes through
        _apply, _finish_draft, _protect or _issue (refused while CANCELLING) and never waits."""
        ep = self.trace[r.uid]["epochs"][-1]
        m_acc = (min(c.m for c in r.cohorts) - 1) if r.cohorts else r.n_trig   # the epoch's accepted cohorts: 1..m_acc
        for cc in list(r.cohorts):
            if cc.state in (PENDING, RUNNING, DONE):
                was = cc.state
                cc.state = CANCELLED
                self.cstat["cohorts"] += 1
                self.cstat["cohort_rows"] += len(cc.positions)
                cc.ws.clear()
                ep["cohorts"].append(self._cohort_record(r, cc, was=was))
                self._finished_key(cc.key)
                self._cancel_requester(cc.key)
                self._free_cohort(cc)
                self._vfetch_drop_cohort(cc.key, "cancelled")
        n_open = cad.cohort_of(len(r.drafts) + 1, self.K)
        for m in range(m_acc + 1, n_open + 1):
            k = (r.uid, r.epoch, m)
            if not any(cc.key == k for cc in r.cohorts):
                self._finished_key(k)
                self._cancel_requester(k)
                self.cstat["untriggered_keys"] += 1
        for dd in r.drafts:
            if cad.cohort_of(dd.s, self.K) > m_acc:
                self._record_draft(r, dd, "cancelled")
                self.cstat["drafts"] += 1
        if r.cur is not None and r.cur.epoch == r.epoch:
            self._record_draft(r, r.cur, "cancelled_in_flight")
            self.cstat["drafts_in_flight"] += 1
        ep["outcome"] = dict(kind="cancellation", n=r.n, t=self.t)
        r.cohorts = []
        r.cur = None
        self._end_request(r, cancelled=True)
        self.cstat["torn_down"] += 1

    def _forward_handbacks(self) -> None:
        """I17 / I16 (lead): every handback the store recorded since the last take goes, in op order, to the executor
        that declares accepts_handback in its own class (on_handback(hb)); without one they are taken and counted."""
        if not self.cfg.store_handback:
            return
        for hb in self.store.take_handbacks():
            self.dstat["handbacks"] += 1
            if self.handback_to is not None:
                self.handback_to.on_handback(hb)
                self.dstat["handbacks_forwarded"] += 1

    def _resume_step(self) -> None:
        """I17: the step that a pending call stopped. ONE passive ticket poll (no wait): if the call's ticket is still
        outstanding, return with nothing changed (dstat["pending_returns"]); a BROKEN ticket (its own state) is
        KStepError, and run() refuses a call that stays unfinished after its event completed, so neither spins; no
        pending call at all (a failure after an earlier publication) is KStepError; else publish the call ONCE (the
        live rows only; a discarded row publishes nothing and advances nothing) and run the step's remaining layers,
        which may stop again at the next pending call; a pending layer is never resubmitted."""
        ps, pc = self.pending_step, self.pending_call
        if pc is None:                                                 # review fix: a failure AFTER the publication
            raise KStepError("I17: the stopped step of t %d has no pending call (its call was published, then a later "
                             "part of the step failed): no retry; the reset stays refused (pending_steps)" % ps["t"])
        self.tickets.poll()                                            # a finishing fault propagates from here
        # Still outstanding: pending, or BROKEN by a finishing fault. The ticket's OWN state decides: an event that
        # completes on its own after the poll (an asynchronous backend) is no fault, the next poll finishes it.
        if self.tickets.is_outstanding(pc["tid"]):
            first = self.tickets.pending().get("first") or {}
            if first.get("id") == pc["tid"] and first.get("state") == "broken":
                raise KStepError("I17: the pending call %d (layer %d) has a BROKEN ticket %d (a finishing fault): no "
                                 "retry, no forced cleanup (the reset and the request end stay refused)"
                                 % (pc["cid"], pc["l"], pc["tid"]))
            self.dstat["pending_returns"] += 1
            return
        self.pending_call = None
        if self.cancel_req:                                            # I20: each requested cancellation ONCE, after
            for uid in list(self.cancel_req):                          # the call's ticket finished (its rows of the
                self._cancel_teardown(self.reqs[uid])                  # cancelled request were discarded and its holds
                del self.cancel_req[uid]                               # released by the ticket) and BEFORE the
            n0 = len(ps["rows"])                                       # publication; the step resumes with the
            ps["rows"] = [(r, dr) for r, dr in ps["rows"] if not r.done]   # remaining live rows only
            self.cstat["rows_dropped_at_resume"] += n0 - len(ps["rows"])
        self._call_publish(pc["l"], pc["ctx"], self._ticket_outs(pc["l"], pc["ctx"], deferred=True))
        for l in range(pc["l"] + 1, self.L):
            if ps["vonly"]:
                self._force_for_layer(l)
            if not self._call(l, ps["rows"]):
                return                                                 # pending again: ps stays, pending_call is set
        self.pending_step = None
        self._step_tail(ps["t"], ps["rows"], ps["vonly"], ps["before"])

    def step(self) -> None:
        self._session_guard("step")                                    # HOOK-L7 (I12): no resume unless "open"
        self._in_step = True                                           # I20: cancel() is refused inside a step
        try:
            self._step()
        finally:
            self._in_step = False

    def _step(self) -> None:
        if self.pending_step is not None:                              # I17: finish the stopped step first
            return self._resume_step()
        t = self.t
        self._admit()
        self._forward_handbacks()                                      # I17 / I16: the admissions' handbacks
        if self.rev6:
            self._rev6_reserve()
        for r in self.active():
            for c in r.cohorts:
                if c.state == PENDING and c.trigger_step < t:
                    if self.mut.delayed_join_even_steps and t % 2 != 0:
                        continue                                        # NEGATIVE CONTROL (R3.3, OWN_join_even_steps)
                    if self.mut.delayed_join_wait_other_requests and any(
                            cc.state == RUNNING for rr in self.active() if rr is not r for cc in rr.cohorts):
                        continue                                        # NEGATIVE CONTROL (R3.3, OWN_join_wait_other_requests)
                    c.state = RUNNING                                   # D2: joins at layer 0 of the next step
                    c.join_step = t                                     # R3.1
                    c.hidden = [self.ex.embed(tok) for tok in c.tokens]
        rows: List[Tuple[Request, Draft]] = []
        self._rev6_stalled = False
        for r in self.active():
            why = self._stall_reason(r)
            if why:
                self.stalls.append((t, r.uid, why, r.lead()))
                self.stats["stall_" + why] += 1
                if why in ("admission", "horizon", "capacity"):
                    self._rev6_stalled = True
                continue
            if self.rev6:
                self._reserve_credits(r)                               # SC2: before layer 0, every (layer, head) at once
            if self.rev6_guard and self._pins_high(r):                 # C81 witness: the pin guard would stall this step
                self.stats["rev6g_pin_guard_would_stall"] += 1
            s = len(r.drafts) + 1
            dr = Draft(s, r.E + s - 1, r.u(s - 1), r.epoch, r.n, cad.cohort_of(s, self.K), t)
            dr.hidden = self.ex.embed(dr.tok_in)
            r.cur = dr
            rows.append((r, dr))
            self.stats["draft_rows_started"] += 1
        vonly = not rows
        if vonly:
            if not any(c.state in (RUNNING, PENDING) for r in self.active() for c in r.cohorts):
                if not self.active():
                    self.t += 1
                    return
                if not (self.rev6 and self._rev6_stalled):
                    raise KStepError("step %d: no draft row and no cohort, but requests are active" % t)
                if not self._rev6_outstanding():
                    raise CapacityRefusal("rev6 step %d: every active request stalls (%s), no cohort runs and no transfer, "
                                          "reader event or release is outstanding" % (t, [x for x in self.stalls if x[0] == t]))
            self.stats["verifier_only_passes"] += 1
            if self.rev6 and self._rev6_stalled:
                self._rev6_force_all()                                 # rev6: free 'capacity' / 'admission' stalls
            for key_l in [x for x in self.pending_release]:              # D4: protections waiting for a reader event
                fr, lr = self.key_readers.get(key_l, (None, None))
                if lr is not None and not ev_done(lr):
                    self.be.host_wait(lr)
                    self.stats["forced_readers"] += 1
        before = self.stats["progress"]
        for l in range(self.L):
            if vonly:
                self._force_for_layer(l)
            if not self._call(l, rows):                                # I17: the call is pending (deferred "one")
                self.pending_step = dict(t=t, rows=rows, vonly=vonly, before=before)
                return
        self._step_tail(t, rows, vonly, before)

    def _step_tail(self, t: int, rows: List[Tuple[Request, Draft]], vonly: bool, before: int) -> None:
        """The end of one step (unchanged order): _end_of_step, the D4 no-progress count, t + 1."""
        self._end_of_step(rows)
        if vonly and self.stats["progress"] == before:
            self._noprog += 1
            if self._noprog >= self.cfg.noprogress_passes:
                raise NoProgress("D4: %d verifier-only passes without progress at step %d (in flight %d, cohorts %s)"
                                 % (self._noprog, t, len(self.inflight),
                                    [c for r in self.active() for c in r.cohorts if c.state in (RUNNING, PENDING)]))
        else:
            self._noprog = 0
        self.t += 1

    def _stall_reason(self, r: Request) -> Optional[str]:
        if r.drafting_done:
            return "done"
        if self.mut.draft_k_then_stop and r.lead() >= self.K:
            return "lead"                                           # NEGATIVE CONTROL: claims the lead bound
        if r.lead() >= self.LAMBDA and not self.mut.ignore_lead_bound:
            return "lead"
        if self.cfg.pin_guard and self._pins_high(r):
            return "memory"
        if self.rev6:
            return self._rev6_stall(r)
        return None

    def _pins_high(self, r: Request) -> bool:
        """C2 memory refusal (PIN GUARD). While the request has a triggered, undecided cohort, it does not start a draft
        step if any of its streams would then hold more than C - K * max_sel protected blocks. Proof of head progress
        (D4) with the verifier eviction of _issue: at the layer where the epoch's oldest undecided cohort waits, no later
        cohort has verifier protections yet (D3a), so protections there are draft protections (<= C - K*max_sel after
        the guard; <= (K+1)*max_sel before the next trigger, which C >= (2K+1)*max_sel keeps below the same bound) plus
        the waiting cohort's own <= K*max_sel blocks: at most C. After forcing in-flight transfers and reader events, a
        stream at usage >= C therefore holds an unprotected LRU entry, which the verifier eviction frees."""
        if not any(c.state in (PENDING, RUNNING, DONE) for c in r.cohorts):
            return False
        lim = self.cfg.C - self.K * self.rb
        for l in range(self.L):
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                if len(st.prot) + self.rb > lim:
                    return True
        return False

    # ================================================================== capacity mode rev6 (PROPOSAL rev 6, section 8)
    def _rev6_reserve(self) -> None:
        """L4 at the step start. An ended request's reservation returns once every slot it owned (any layer, any head of
        its batch slot) is FREE. Then every active request without a reservation that still drafts gets H * d_h shared
        draft positions per layer (the same at every layer), in request age order and without overtaking: when the
        oldest such request does not fit, no younger one is served."""
        for uid in [u for u, live in self.dres.items() if not live]:
            if not self._owns_slots(uid, self.dres_b[uid]):
                del self.dres[uid]
                del self.dres_b[uid]
                self.stats["rev6_reservations_returned"] += 1
        need = self.H * self.d_share
        free = self.X_d - need * len(self.dres)
        for r in sorted(self.active(), key=lambda x: self.age[x.uid]):
            if r.uid in self.dres or r.drafting_done:
                continue
            if need > free:
                break
            self.dres[r.uid] = True
            self.res_log.append((self.t, r.uid))
            free -= need
            self.stats["rev6_reservations"] += 1

    def _owns_slots(self, uid, b: int) -> bool:
        for pool in self.pools:
            for h in range(self.H):
                for s in pool.stream(b, h).slots:
                    if s.state != FREE and s.owner == uid:
                        return True
        return False

    def _positions(self, st) -> int:
        """L2: P of a stream (every non-FREE "ad" slot). NEGATIVE CONTROL rev6_unpinned_lru_counts_free: the rev-5 rule,
        an unpinned LRU entry counted as available."""
        if self.mut.rev6_unpinned_lru_counts_free:
            return st.positions() - st.unpinned_lru()
        return st.positions()

    @staticmethod
    def _untriggered_pins(r: Request, st) -> int:
        """SC2(a): the blocks of the stream that a key of an UNTRIGGERED cohort of r's current epoch references (only draft
        rows reference them: no verifier exists before the trigger)."""
        n = 0
        for keys in st.prot.values():
            if any(isinstance(k, tuple) and len(k) == 3 and k[0] == r.uid and k[1] == r.epoch and k[2] > r.n_trig
                   for k in keys):
                n += 1
        return n

    def _rev6_stall(self, r: Request) -> Optional[str]:
        """SC2 at the step start: 'admission' (no L4 reservation), 'horizon' (SC2a: untriggered pins + (T - s + 1) * S_max
        > d_h on some stream; never with d_h >= W, P6 ii) or 'capacity' (SC2b: P + S_max > C + d_h on some stream). It
        reserves nothing (_reserve_credits does, after this returns None)."""
        if r.uid not in self.dres:
            return "admission"
        s = len(r.drafts) + 1
        T = ((s + self.K - 1) // self.K) * self.K
        cap = self.cfg.C + self.d_share
        horizon = capacity = False
        for l in range(self.L):
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                if self._untriggered_pins(r, st) + (T - s + 1) * self.rb > self.d_share:
                    horizon = True
                if self._positions(st) + self.rb > cap:
                    capacity = True
        if horizon:
            return "horizon"
        if capacity:
            return "capacity"
        return None

    def _reserve_credits(self, r: Request) -> None:
        """SC2: S_max step credits on EVERY (layer, head) at once, before layer 0. NEGATIVE CONTROL
        rev6_partial_reservation: layer 0 only."""
        part = self.mut.rev6_partial_reservation
        r.credits = [[self.rb if (l == 0 or not part) else 0 for _ in range(self.H)] for l in range(self.L)]
        self.stats["rev6_credits_reserved"] += sum(sum(x) for x in r.credits)

    def _take_credit(self, r: Request, l: int, h: int, why: str) -> None:
        """SC3: one credit for one new "ad" position of r's draft row at (l, h). None left: the SC4 contract is broken."""
        cr = r.credits
        if cr is None or cr[l][h] < 1:
            raise OwnershipViolation("rev6 SC4: the draft row of uid %s needs a new A / D position at layer %d head %d (%s) but "
                                     "holds no step credit (credits %s)" % (r.uid, l, h, why, cr))
        cr[l][h] -= 1
        self.stats["rev6_credits_" + why] += 1                         # rev6_credits_fetch / _join / _vres

    def _credits_held(self, b: int, l: int, h: int) -> int:
        r = self.slot_req[b]
        return 0 if r is None or r.credits is None else r.credits[l][h]

    def _rev6_outstanding(self) -> bool:
        return bool(self.inflight) or bool(self.pending_release) or any(
            s.state == RELEASING for pool in self.pools for st in pool.streams.values() for s in st.slots)

    def _rev6_force_all(self) -> None:
        """A verifier-only pass with a rev6 'admission' / 'horizon' / 'capacity' stall: force every transfer in flight, every
        RELEASING slot's reader event and every queued release's reader event (the deferred CPU model completes nothing
        on its own; a GPU completes them without a host wait)."""
        for tr in sorted(self.inflight.values(), key=lambda x: x.seq):
            self.tp.force(tr)
            self.stats["forced_transfers"] += 1
        for pool in self.pools:
            for ev in pool.releasing_gates():
                self.be.host_wait(ev)
                self.stats["forced_readers"] += 1
        for key_l, (fr, lr) in list(self.key_readers.items()):
            if lr is not None and not ev_done(lr):
                self.be.host_wait(lr)
                self.stats["forced_readers"] += 1

    def _is_ad(self, st, j: int) -> bool:
        """G2: may an UNGRANTED verifier pin block j of stream st? Only a resident or in-flight slot of class "ad"."""
        idx = st.resident(j)
        if idx is None:
            idx = st.pending.get(j)
        return idx is not None and st.slots[idx].cls != "v"

    def _rev6_verifiers(self, l: int, eligible, requests: List[Tuple], exec_cohorts: List[Tuple]) -> None:
        """G1-G6 at layer l (replaces the M1 verifier loop of _call). The eligible cohorts in age order (trigger step,
        request age, m). An ungranted cohort is granted when no older eligible cohort was refused at this layer in this
        call (no overtaking), its new residual fetches fit the free verifier reserve, and every stream has that many FREE
        positions that no step credit holds. A granted cohort pins its whole exact selection (a "v" slot of another grant
        is shared, so it moves to this grant without a free when the other releases it, G4a) and requests its misses
        (class "v"); an ungranted cohort pins only "ad" blocks (G2) and requests nothing. After a refusal the next
        verifier-only pass forces every transfer and reader event of the layer; G5 raises CapacityRefusal when nothing
        at the layer is outstanding."""
        pool = self.pools[l]
        v_free = self.cfg.R_ver - pool.v_used()
        refused = None
        order = sorted(eligible, key=lambda rc: (rc[1].trigger_step, self.age[rc[0].uid], rc[1].m))
        for r, c in order:
            need = self._need(r, c.ws[l]["sel"], l, self._vhist(r, c, l))   # H1: the frozen boundary
            n_new = sum(len(miss) for _, _, miss in need)
            if l in c.granted:
                if n_new:
                    raise OwnershipViolation("rev6 G3: granted cohort %s lost %d blocks of its exact selection at layer %d"
                                             % (c.key, n_new, l))
                granted = True
            else:
                fits = n_new <= v_free and all(
                    pool.stream(r.b, h).free_count() - self._credits_held(r.b, l, h) >= len(need[h][2])
                    for h in range(self.H))
                granted = refused is None and fits
                if granted:
                    c.granted.add(l)
                    v_free -= n_new
                    self.stats["rev6_grants"] += 1
                    self.grant_log.append((self.call_id, l, r.uid, c.key, n_new, v_free))
                else:
                    self.stats["rev6_grant_waits"] += 1
                    if refused is None:
                        refused = (c, n_new, v_free, fits)
            for h in range(self.H):
                st = pool.stream(r.b, h)
                for blocks in need[h]:
                    for j in blocks:
                        if granted or self._is_ad(st, j):
                            self._protect(st, l, j, c.key, VERIFY)     # K2 (granted) / G2 ("ad" only, ungranted)
                if granted:
                    for j in need[h][1]:
                        self._add_demand(st, j, VERIFY)               # R2.3
            if granted and all(not miss and not infl for _, infl, miss in need):
                exec_cohorts.append((r, c))
                continue
            c.waits += 1
            self.stats["cohort_waits"] += 1
            if granted:
                for h in range(self.H):
                    for j in need[h][2]:
                        requests.append((VERIFY, r, h, j, c.key))
        if refused is not None:
            self._vfail.add(l)
            self._g5(l, refused)

    def _g5(self, l: int, refused) -> None:
        """G5: the oldest eligible cohort at layer l was refused. Outstanding at the layer: a transfer in flight, a
        RELEASING slot, a queued release, or a granted cohort that has not executed the layer. None: CapacityRefusal."""
        c, n_new, v_free, fits = refused
        pool = self.pools[l]
        busy = (any(tr.layer == l for tr in self.inflight.values())
                or any(s.state == RELEASING for st in pool.streams.values() for s in st.slots)
                or any(ll == l for _, ll in self.pending_release)
                or any(l in cc.granted and cc.next_layer == l and cc.state == RUNNING
                       for rr in self.active() for cc in rr.cohorts))
        if not busy:
            raise CapacityRefusal("rev6 G5: cohort %s at layer %d needs %d new verifier slots, %d of R_ver %d are free "
                                  "(positions fit: %s), and nothing at the layer is outstanding (v_used %d)"
                                  % (c.key, l, n_new, v_free, self.cfg.R_ver, fits, pool.v_used()))

    # ---------------------------------------------------------------- rev6 H1: recent-to-history handoff
    def _vhist(self, r: Request, c: Cohort, l: int) -> int:
        """H1: the history boundary of cohort c's layer-l workspace, frozen at its phase 1 (rev6). A block at or above it is
        read from the recent region (retained by _history while this workspace or its read depends on it); a block below
        it is a historical slot that the grant covers. Mode m1 (and the negative control rev6_unfrozen_boundary) uses the
        current boundary, so a block that becomes historical while the cohort waits is a new historical need there."""
        if self.rev6 and not self.mut.rev6_unfrozen_boundary:
            return c.ws[l]["hist"]
        return r.hist_blocks

    def _h1_holders(self, r: Request, l: int, h: int, j: int) -> set:
        """H1: the cohorts that still read block j of (l, h) from the RECENT region: a waiting workspace at layer l whose
        frozen boundary is <= j and whose selection holds j at head h; or an executed layer-l read of the same kind whose
        (key, l) reader gate has not released yet (h1_reads)."""
        out = set()
        for c in r.cohorts:
            w = c.ws.get(l)
            if w is not None and w["hist"] <= j and any(j in row[h] for row in w["sel"]):
                out.add(c.key)
        for (key, ll), (hist, rows) in self.h1_reads.items():
            if ll == l and key[0] == r.uid and hist <= j and any(j in row[h] for row in rows):
                out.add(key)
        return out

    def _h1_release(self, r: Request, key, l: int) -> None:
        """H1: `key` no longer reads layer l (its (key, l) reader gate released, or its request ended): it leaves the
        holder sets of the retained positions of layer l; a retained position without holders leaves the recent region."""
        for x in [x for x in r.pub_hold if x[0] == l]:
            hold = r.pub_hold[x]
            hold.discard(key)
            if not hold:
                ll, h, pos = x
                del r.pub_hold[x]
                del r.pub[ll][h][pos]
                self.stats["rev6_h1_released_positions"] += 1
                if r.rmap is not None and pos in r.rmap[ll][h][R_EXACT]:   # M2 B2: never expected (a waiting holder
                    self.rstat["ring_h1_late_drains"] += 1             # keeps its key until its own decision)
                    self._rdrop(ll, r, h, R_EXACT, pos, "h1")

    def _key_live(self, key, l: int) -> bool:
        """Does the cohort `key` still hold its layer-l draft references (undecided and not released at layer l)?"""
        if not (isinstance(key, tuple) and len(key) == 3):
            return False
        uid, e, m = key
        r = self.reqs.get(uid)
        if r is None or r.done or e != r.epoch or key in self.cancelled_keys:
            return False
        if (self.rel_layers.get(key) or {}).get(l):
            return False
        if m > r.n_trig:
            return True
        return any(c.key == key and c.state in (PENDING, RUNNING, DONE) for c in r.cohorts)

    def _rev6g_sample(self, l: int) -> None:
        """progress_guard 'rev6' (C81 / K3 note): peaks SAMPLED at the end of the joint call of layer l (not maxima over
        every transition), every mode of the check (instrumentation): the largest P of a stream, how far it exceeded C
        (0: never), the layer's "v" slots."""
        s, pool = self.stats, self.pools[l]
        for st in pool.streams.values():
            p = st.positions()
            s["rev6g_peak_positions"] = max(s["rev6g_peak_positions"], p)
            s["rev6g_peak_positions_over_C"] = max(s["rev6g_peak_positions_over_C"], p - self.cfg.C)
        s["rev6g_peak_v_used"] = max(s["rev6g_peak_v_used"], pool.v_used())

    def _rev6_check(self, l: int) -> None:
        """The rev6 ledger at the end of the joint call of layer l (OwnershipViolation on failure): P + credits <= C + d_h
        on every stream; v_used <= R_ver; the reservations fit X_d; an LRU / PO slot is "ad" and a VRES slot is "v"; no
        draft reference holds a "v" slot; every reference to a "v" slot is a grant of layer l or a queued release."""
        pool = self.pools[l]
        cap = self.cfg.C + self.d_share
        if self.H * self.d_share * len(self.dres) > self.X_d:
            raise OwnershipViolation("rev6 L4: %d reservations of H * d_h = %d > X_d = %d"
                                     % (len(self.dres), self.H * self.d_share, self.X_d))
        if pool.v_used() > self.cfg.R_ver:
            raise OwnershipViolation("rev6 L1: layer %d holds %d verifier slots > R_ver %d" % (l, pool.v_used(), self.cfg.R_ver))
        grants = {cc.key for rr in self.active() for cc in rr.cohorts if l in cc.granted}
        queued = {k for k, ll in self.pending_release if ll == l}
        for (b, h), st in pool.streams.items():
            p, e = st.positions(), self._credits_held(b, l, h)
            if p + e > cap:
                raise OwnershipViolation("rev6 L2: stream (l %d b %d h %d) holds P %d + credits %d > C + d_h = %d"
                                         % (l, b, h, p, e, cap))
            for blk, idx in list(st.lru.items()) + list(st.po.items()):
                if st.slots[idx].cls == "v":
                    raise OwnershipViolation("rev6 L1: LRU / PO block %d of stream (l %d b %d h %d) is class v" % (blk, l, b, h))
            for blk, idx in st.vres.items():
                if st.slots[idx].cls != "v":
                    raise OwnershipViolation("rev6 L1: VRES block %d of stream (l %d b %d h %d) is not class v" % (blk, l, b, h))
            for s in st.slots:
                if s.state in (FREE, RELEASING) or s.cls != "v" or s.owner != st.owner:
                    continue
                if DRAFT in st.kinds_of(s.block):
                    raise OwnershipViolation("rev6 L1: a draft reference holds verifier slot %d (block %d, l %d b %d h %d)"
                                             % (s.idx, s.block, l, b, h))
                bad = [k for k in st.prot.get(s.block, ()) if k not in grants and k not in queued]
                if bad:
                    raise OwnershipViolation("rev6 G2: verifier slot %d (block %d, l %d b %d h %d) is held by %s, which hold no "
                                             "grant at layer %d" % (s.idx, s.block, l, b, h, bad, l))

    # ================================================================== host poll
    def _poll(self) -> None:
        self.tp.poll()
        arrived = sorted((tr for tr in self.inflight.values() if self.tp.done(tr)), key=lambda x: x.seq)
        for tr in arrived:
            del self.inflight[tr.seq]
            st = self.pools[tr.layer].stream(tr.b, tr.h)
            if self.arena_mode:
                self._arena_admission(tr, st)                          # M2 2.2 / 2.5: the binding (and the readback) first
            res = st.admit(tr.slot, tr.uid, tr.block, tr.gen, self.mut, demand=tr.demand)
            if self.arena_mode:
                self.arenas[tr.layer].landed(tr.phys, tr.seq, tr.uid, tr.b, res["dropped"])
            self.arrival_log.append(tr.seq)
            self.stats["progress"] += 1
            self.stats["traffic"] += 1                                  # N3 P1: an arrival (dropped ones too) is traffic
            if tr.requester in self.cancelled_keys:
                self.stats["arrivals_after_cancel"] += 1               # T3: kept (history is immutable) or dropped
            if res["dropped"]:
                self.stats["arrivals_dropped_unowned"] += 1
            else:
                self.stats["arrivals_admitted"] += 1
                self.stats["arrivals_" + res["where"]] += 1            # arrivals_lru / arrivals_vres / arrivals_releasing
                if tr.kind == VERIFY and DRAFT in tr.demand:
                    self.stats["arrivals_verify_joined_by_draft"] += 1  # R2.3: a draft request joined a verifier transfer
                self.admit_log.append((self.call_id, tr.layer, tr.b, tr.h, tr.block, tuple(sorted(tr.demand)), res["where"],
                                       tr.kind))
                ev = res["evicted"]
                if ev is not None:
                    self.stats["evict_protected_only" if ev["protected"] else "evict_released"] += 1
                    self.evict_log.append((self.call_id, tr.layer, tr.b, tr.h, ev["block"], ev["protected"],
                                           tuple(ev.get("kinds", ())), "arrival"))
                if tr.kind == DRAFT and res["where"] == LRU and not self._key_live(tr.requester, tr.layer):
                    # a LATE O2 arrival: its requesting draft cohort no longer holds layer-l references
                    self.stats["late_draft_arrivals"] += 1
                    disp = None if ev is None else (ev["block"], ev["protected"], tuple(ev.get("kinds", ())))
                    if ev is not None and ev["protected"]:
                        self.stats["late_draft_arrivals_displacing_a_pin"] += 1
                    self.late_log.append((self.call_id, tr.layer, tr.b, tr.h, tr.block, disp))
        if self.cancelled_keys:                                         # K6: kept only while a transfer may still name it
            self.cancelled_keys &= {tr.requester for tr in self.inflight.values()}
        for pool in self.pools:
            n = pool.poll_releases(self.mut)
            if n:
                self.stats["slots_freed"] += n
                self.stats["progress"] += 1
                self.stats["traffic"] += 1                              # N3 P1: one per layer pool that freed ANY slot
        if self.ring_mode:                                              # M2 B2: ring retirement beside poll_releases
            self._ring_retire()                                         # (non-blocking; never a logical counter)
        if self.abi4:                                                   # C1.1 A9: passive release of the operational
            self.bias_ledger.retire()                                   # bias backings whose frontier completed
        self._validate()
        if self.recent != "eager":
            self._d2h_poll()                                            # M2 B2 11.3 (1): snapshots, non-blocking
        keep = []
        for key, l in self.pending_release:
            fr, lr = self.key_readers.get((key, l), (None, None))
            gate = fr if self.mut.release_first_reader else lr
            if self.mut.no_reader_wait or self.mut.prot_release_no_reader_gate or ev_done(gate):
                self._release_refs(key, l)
            else:
                keep.append((key, l))
        self.pending_release = keep
        done = [k for k, lay in self.rel_layers.items() if len(lay) == self.L and all(lay.values())]
        for k in done:                                                  # K6: every layer of the key is released
            del self.rel_layers[k]
        if self.frontier is not None:                                   # HOOK-C: prune complete frontiers; clear the
            self.frontier.poll(self)                                    # records of drained incarnations (non-blocking)

    def _release_refs(self, key, l: int) -> None:
        """R1: drop the references keyed `key` at layer l (one key, R1.2); PO / VRES slots that lose their last reference
        go through the reader gate. A second release of the same (key, l) is an ownership violation (R1.4)."""
        lay = self.rel_layers.get(key)
        if lay is None and self.frontier is not None:                  # HOOK-C: after pruning, the durable finished-
            self.frontier.refuse_late_release(self, key, l)            # cohort record names a double release (m1)
        if lay is None or l not in lay:
            raise KStepError("R1.4: release of cohort %s at layer %d that was never queued" % (key, l))
        if lay[l]:
            raise OwnershipViolation("R1.4: double release of cohort %s at layer %d" % (key, l))
        lay[l] = True
        r = self.reqs[key[0]]
        for h in range(self.H):
            st = self.pools[l].stream(r.b, h)
            for idx in st.unprotect(key, self.mut):
                st.release(idx, "unprotect")
        self.key_readers.pop((key, l), None)
        self.stats["releases_per_layer"] += 1
        if self.rev6:                                                  # H1: the (key, l) read is over (reader gate)
            self.h1_reads.pop((key, l), None)
            self._h1_release(r, key, l)

    def _validate(self, all_: bool = False) -> None:
        keep = []
        for ev, ws, exp, info in self.validators:
            if all_ or ev_done(ev):
                pg = info.get("pg")
                if pg is not None and int(pg[0]) != info["pgen"]:          # M2 3.1 (a): identity first, then the bytes
                    raise StaleReader("stale reader: position %d pgen captured %d, at read %d (layer %d b %d h %d slot %s "
                                      "block %d uid %s call %d)" % (info["phys"], info["pgen"], int(pg[0]), info["layer"],
                                                                    info["b"], info["h"], info["slot"], info["block"],
                                                                    info["uid"], info["call"]))
                rg = info.get("rg")
                if rg is not None and int(rg[0]) != info["rgen"]:          # M2 B2 11.3: ring identity first, then bytes
                    raise StaleReader("stale reader: ring entry %d (layer %d) rgen captured %d, at read %d (%s %s of uid %s "
                                      "b %d h %d position %d call %d)" % (info["entry"], info["layer"], info["rgen"],
                                                                         int(rg[0]), info["kind"], info["which"],
                                                                         info["uid"], info["b"], info["h"], info["pos"],
                                                                         info["call"]))
                if not torch.equal(ws, exp):
                    shown = info if pg is None and rg is None else {k: v for k, v in info.items() if k not in ("pg", "rg")}
                    raise ReadCorruption("deferred read of %s saw other bytes than the call computed from" % (shown,))
                if info.get("ring"):
                    self.rstat["ring_reads_validated"] += 1                # M2 B2: a ring counter, not a logical one
                else:
                    self.stats["reads_validated"] += 1
                if info.get("which") == "c":                               # C1.1: a bias read (identity, then bytes)
                    self.c1stat["ring_bias_reads_validated" if info.get("ring") else "arena_bias_reads_validated"] += 1
            else:
                keep.append((ev, ws, exp, info))
        self.validators = keep

    # ---------------------------------------------------------------- M2 slice A / B1 (Config.physical = "arena")
    def _checked(self) -> bool:
        """Checked mode, as for every other invariant check: check_invariants and no Mutations switch active."""
        return bool(self.cfg.check_invariants) and not self.mut.active()

    def _credits_at(self, l: int) -> int:
        """E(l): the unconsumed rev6 draft step credits held at layer l (0 in mode m1)."""
        return sum(sum(r.credits[l]) for r in self.active() if r.credits is not None)

    def _register_read(self, l: int, row, h: int, block: int, phys: int, pgen: int, ws, exp, pg) -> None:
        """The executor's attach_arena callback: ONE deferred arena read of this call (the K or the V rows of one (row,
        head, block)), validated when the call's read event completes (_arm_validators): the identity copy pg (None: no
        identity check) against pgen, then the bytes ws against the eager clone exp."""
        who = self.arenas[l].bound[phys]
        info = dict(layer=l, b=row.b, h=h, slot=None if who is None else who[2], block=block, uid=row.uid,
                    call=self.call_id, phys=phys, pgen=pgen)
        if pg is not None:
            info["pg"] = pg
        if self.abi4:                                                  # C1.1: the bias rows are the 1-D read (12.3 A4)
            info["which"] = "c" if ws.dim() == 1 else "kv"
            if ws.dim() == 1:
                self._n_bias_reads += 1
                self.c1stat["arena_bias_reads_registered"] += 1
        self.validators.append((None, ws, exp, info))
        self._n_new_val += 1

    def _register_ring_read(self, l: int, row, h: int, pos: int, e: int, g: int, kind: str, which: str, ws, pg) -> None:
        """The executor's attach_ring callback (interface v3): ONE deferred ring read of this call (the K or the V row of
        entry e for (row, head, position)), validated when the call's read event completes: the identity copy pg against
        rgen g, then the bytes ws against the EAGER value the row attends (Row.kv's reference, not a ring clone: the
        entry's payload write may not have executed at call time)."""
        exp = self._ring_exp.get((id(row), h, pos))
        if exp is None or pg is None or which not in (("k", "v", "c") if self.abi4 else ("k", "v")):
            raise KStepError("M2 B2: the executor registered a ring read of position %d (head %d, %s, identity copy %s) that "
                             "row (uid %s pos %d) does not attend" % (pos, h, which, pg is not None, row.uid, row.pos))
        if self.store is not None:                                     # HOOK-B2: the expectation in the stores' dtypes
            val = self.store.dispatch_expect_item(self, exp, which)  # HOOK-B2: (the executor's tensors of exp)
            if which == "c":                                           # HOOK-B2: counted as below
                self.c1stat["ring_bias_reads_registered"] += 1         # HOOK-B2
        elif which == "c":                                             # HOOK-B2: "if" before; C1.1: the entry's bias
            val = torch.tensor([exp[2]], dtype=torch.int64)
            self.c1stat["ring_bias_reads_registered"] += 1
        else:
            val = torch.tensor(exp[0] if which == "k" else exp[1], dtype=torch.int64)
        info = dict(ring=True, layer=l, b=row.b, h=h, pos=pos, entry=e, rgen=g, kind=kind, which=which, uid=row.uid,
                    call=self.call_id, rg=pg)
        self.validators.append((None, ws, val, info))
        self._n_new_val += 1
        self._n_ring_val += 1
        self.rstat["ring_reads_registered"] += 1

    def _install_reader(self, l: int, read_slots, ev, stream: Optional[str] = None) -> None:   # HOOK-C: stream
        """The call's reader event becomes the last reader of every slot read (arena mode: and of its position) and the
        (key, l) reader event of every key that read a slot (K4, R1). M2 B2 ring mode: it is also the compute gate of
        every ring entry the call read, also when no historical slot was read."""
        if not read_slots and not self._ring_reads:
            return
        if self.frontier is not None:                                  # HOOK-C: "joined": each object gets the JOINED
            self.frontier.install(self, l, read_slots, ev,             # handle of its per-stream latest readers; the
                                  stream or getattr(self.ex, "read_stream", "compute"))   # stream that recorded ev
            return
        for l_, b_, h_, idx in read_slots:
            s = self.pools[l_].stream(b_, h_).slots[idx]
            if s.first_reader is None:
                s.first_reader = ev
            s.last_reader = ev
            if self.arena_mode:
                self.arenas[l_].note_reader(s.phys, s.pgen, ev)
        for key in self._read_keys:
            kr = self.key_readers.setdefault((key, l), [None, None])
            if kr[0] is None:
                kr[0] = ev
            kr[1] = ev
        for ll, e, g in self._ring_reads:                              # M2 B2 11.3: the compute gate (one ordered stream)
            self.rings[ll].note_compute(e, g, ev)

    def _arm_validators(self, ev) -> None:
        """The deferred-read validators registered in this call are checked once `ev` completed."""
        for i in range(len(self.validators) - self._n_new_val, len(self.validators)):
            _, ws, exp, info = self.validators[i]
            self.validators[i] = (ev, ws, exp, info)

    def _check_post_reads(self, jrows, read_slots, read_ev) -> None:
        """The executor's reads of the call, checked right after it returned (M2 B1 / B2, C1.1): read_ev exists exactly
        when the call read a slot or a ring entry; two (ABI v4: three) registered reads per ring entry read; one arena
        bias read per (row, head, block) read (ABI v4). Shared by joint_post and, in consume mode, joint_post_consume."""
        if (read_ev is None) != (not read_slots and not self._ring_reads) or (self._n_new_val and read_ev is None):
            raise KStepError("M2 B1: the executor returned read event %r for %d slot reads and %d ring reads (%d "
                             "registered reads)" % (read_ev, len(read_slots), len(self._ring_reads), self._n_new_val))
        n_per = 3 if self.abi4 else 2                                  # M2 B2: the K and the V row of every entry read
        if self._n_ring_val != n_per * len(self._ring_reads):          # (C1.1 ABI v4: and its bias row)
            raise KStepError("M2 B2: the executor registered %d ring reads for %d ring entries of the rows"
                             % (self._n_ring_val, len(self._ring_reads)))
        if self.abi4:                                                  # C1.1: one bias read per (row, head, block) read
            n_addr = sum(len(a) for row in jrows for a in (row.addr or ()))
            if self._n_bias_reads != n_addr:
                raise KStepError("C1.1: the executor registered %d arena bias reads for %d (row, head, block) reads"
                                 % (self._n_bias_reads, n_addr))
        if self._ring_reads and not read_slots:
            self.rstat["ring_only_calls"] += 1                         # M2 B2: read_ev with no historical slot read

    # ---------------------------------------------------------------- C1.2 (Config.consume = "physical"; 12.9.2)
    def _consume_post(self, l: int, jrows: List[Row]):
        """C1.2 step 1: the executor's validator reads and per-(row, head) gathers of the call; every gather backing is
        owned by the consume ledger from now on (kind c1_gather, key (call, row index, head, part))."""
        if not jrows:
            return None, None
        handle, cev = self.ex.joint_post_consume(l, jrows)
        for (i, h, part), t in self.ex.gather_tensors(handle):
            self.consume_ledger.hold("c1_gather", t, key=(self.call_id, i, h, part))
        return handle, cev

    def _consume(self, l: int, jrows: List[Row], handle, cev) -> List[Any]:
        """C1.2 steps 3-6, after the reader gates and the validators of the call were installed and armed: the NEW
        per-call host wait on the call's completion event (serialized: it changes completion order); the identity check
        of every gathered block / entry BEFORE any value is used (StaleReader); resolution from the gathered bytes only;
        the exact comparison with the same-order Row.kv reference per row (call order) and head (ascending), then the
        row output (ConsumptionMismatch); then the release of the call's gather backings and of the handle. Returns the
        CONSUMED outputs: they are what the call publishes."""
        if not jrows:
            return []
        cid = self.call_id
        if cev is None:
            raise KStepError("C1.2: call %d consumes %d rows without a completion event" % (cid, len(jrows)))
        self.be.host_wait(cev)
        self.c1stat["consume_identity_checks"] += check_consumed_identity(self.ex.consumed_identities(handle))
        outs, heads = self.ex.resolve(handle)
        ref_outs, ref_heads = self.ex.reference(handle)
        if not (len(outs) == len(heads) == len(ref_outs) == len(ref_heads) == len(jrows)):
            raise KStepError("C1.2: call %d resolved %d / %d rows for %d consumed rows" % (cid, len(outs), len(ref_outs),
                                                                                        len(jrows)))
        for i, row in enumerate(jrows):
            for h in range(self.H):
                if heads[i][h] != ref_heads[i][h]:
                    raise ConsumptionMismatch("C1 consumption mismatch: call %d row (%s, %d, %s) head %d: consumed %r != "
                                              "reference %r" % (cid, row.uid, row.pos, row.kind, h, heads[i][h],
                                                                ref_heads[i][h]))
            if outs[i] != ref_outs[i]:
                raise ConsumptionMismatch("C1 consumption mismatch: call %d row (%s, %d, %s) output: consumed %r != "
                                          "reference %r" % (cid, row.uid, row.pos, row.kind, outs[i], ref_outs[i]))
        self._consume_count(l, jrows)
        for key in [r["key"] for r in self.consume_ledger.recs.values() if r["key"][0] == cid]:
            self.consume_ledger.release_key("c1_gather", key)
        handle.clear()                                                 # the handle's backing references are dropped
        return list(outs)

    def _ticket_commit(self, rid: int, l: int, jrows: List[Row], owners, handle, cev) -> int:
        """HOOK-L8 / I15 (lead): the reservation rid becomes the call's ticket: one RowKey per row (uid, incarnation,
        the row's cohort epoch, its cohort, position, kind, layer, call), the call's c1_gather hold keys, the handle,
        cev. Returns the ticket id. A commit refusal (malformed input) leaves the reservation open (a reset refuses)."""
        cid = self.call_id
        if cev is None:
            raise KStepError("C1.2: call %d consumes %d rows without a completion event" % (cid, len(jrows)))
        keys = [RowKey(row.uid, r.inc, row.cohort[1], row.cohort, row.pos, row.kind, l, cid)
                for (_, r, _, _), row in zip(owners, jrows)]
        holds = [rec["key"] for rec in self.consume_ledger.recs.values() if rec["key"][0] == cid]
        return self.tickets.commit(rid, keys, handle, cev, holds)

    def _consume_ticket(self, l: int, ctx: Dict[str, Any], cev) -> List[Any]:
        """HOOK-L8 (INTERFACE_BC6 I14, lead): C1.2 steps 3-6 of _consume through the committed ticket, serialized, with
        the same result and counters: ONE passive poll while the call's event is still pending (on the deferred backend
        nothing happens: a pending visit); the KEPT host wait on cev; one poll (identity first, resolution, the exact
        comparison, apply once per live row, release); the outputs in call order and the counts (_ticket_outs). Every
        row of a serialized call is live, so a row without an applied output is KStepError."""
        T = self.tickets
        T.poll()                                                       # passive: the call's event is still pending
        self.be.host_wait(cev)                                         # the KEPT serialized wait (C1.2 step 3)
        T.poll()
        return self._ticket_outs(l, ctx, deferred=False)

    def _ticket_outs(self, l: int, ctx: Dict[str, Any], deferred: bool) -> List[Any]:
        """The applied outputs of a finished call in call order, then the C1.2 counts (every consumed row, as the
        serialized consumer counts them). Serialized: every row must have been applied (else KStepError). I17
        (deferred): a row that its ticket did not apply (stale at application: a changed identity) is _DISCARDED."""
        cid, jrows = ctx["cid"], ctx["jrows"]
        out = self.tickets.view.out
        missing = [i for i in range(len(jrows)) if (cid, i) not in out]
        if missing and not deferred:
            raise KStepError("HOOK-L8 (I14): call %d rows %s were not applied by their ticket (%s)"
                             % (cid, missing, self.tickets.pending()))
        outs = [out.pop((cid, i)) if (cid, i) in out else _DISCARDED for i in range(len(jrows))]
        self.dstat["rows_discarded"] += len(missing)
        self._consume_count(l, jrows)
        return outs

    def _consume_count(self, l: int, jrows: List[Row]) -> None:
        """C1.2 counters and first witnesses (instrumentation), from Row.addr / Row.ring of every consumed (row, head) in
        the declared order: historical rows (per block), EXACT / DRAFT / PRE (own position or an earlier row of the same
        cohort) / PROV entries, tail positions (the row's own block), bias items (every gathered row)."""
        s, bs, cid = self.c1stat, self.bs, self.call_id
        s["consume_calls"] += 1
        s["consumed_rows"] += len(jrows)
        for row in jrows:
            tail0 = (row.pos // bs) * bs
            for h in range(self.H):
                items = [(blk * bs, "historical", blk, p, g) for blk, p, g in (row.addr[h] if row.addr else ())]
                items += [(pos, kind, pos, e, g) for pos, e, g, kind in (row.ring[h] if row.ring else ())]
                for _, kind, what, a, g in sorted(items, key=lambda x: x[0]):
                    n = bs if kind == "historical" else 1
                    keys = [kind, "bias"]
                    if kind == R_PRE:
                        keys.append("pre_own" if what == row.pos else "pre_earlier_row")
                    if kind != "historical" and what >= tail0:
                        keys.append("tail")
                    for k in keys:
                        s["consumed_items_" + k] += n
                        if k not in self.consume_first:
                            self.consume_first[k] = (cid, l, row.uid, row.pos, row.kind, h, what, a, g)

    def consume_report(self) -> Dict[str, Any]:
        """C1.2: the consume ledger (held bytes, the EXACT allocation-transition peak, records, allocations, releases,
        outstanding; the K6 category consume_staging is its SAMPLED value, 0 at every sample in this serialized mode),
        the consumption counters and the first consumed item of each provenance kind (call, layer, uid, pos, row kind,
        head, position or block, entry or position, generation). Not a Python-allocator or GPU measurement."""
        lg = self.consume_ledger
        return dict(mode="physical", serialized=True, held_bytes=lg.held, peak_bytes=lg.peak, held_records=len(lg.recs),
                    peak_records=lg.peak_records, allocs=dict(lg.allocs), alloc_bytes=dict(lg.alloc_bytes),
                    releases=dict(lg.releases), outstanding=lg.outstanding(),
                    counters={k: v for k, v in sorted(self.c1stat.items()) if k.startswith("consum")},
                    first=dict(self.consume_first))

    def _arena_admission(self, tr: Transfer, st) -> None:
        """M2 at the observed arrival, before admission: the slot's binding still equals the transfer's (phys, pgen)
        (contract 2.2); checked mode: the transport's payload readback (S0, contract 2.5)."""
        s = st.slots[tr.slot]
        if s.gen == tr.gen and (s.phys, s.pgen) != (tr.phys, tr.pgen):
            raise OwnershipViolation("arena binding: arrival of transfer %d (uid %s, l %d b %d h %d block %d, slot %d gen %d) "
                                     "names position %s pgen %s, but the slot is bound to %s pgen %s"
                                     % (tr.seq, tr.uid, tr.layer, tr.b, tr.h, tr.block, tr.slot, tr.gen, tr.phys, tr.pgen,
                                        s.phys, s.pgen))
        if self._checked():
            rb = getattr(self.tp, "readback", None)
            if rb is not None:
                rb(tr)

    def _arena_reserve_check(self, l: int, rest, seen) -> None:
        """Checked mode, after every bind of _issue (contract 2.3): bound + E + G <= P_l with E the unconsumed credits held
        at layer l and G the granted verifier fetches of this issue that still need a new position (rev6; 0 in m1)."""
        G = 0
        if self.rev6:
            pool, need = self.pools[l], set()
            for kind, r, h, j, key in rest:
                if kind != VERIFY or (r.b, h, j) in seen:
                    continue
                st = pool.stream(r.b, h)
                if st.resident(j) is None and j not in st.pending:
                    need.add((r.b, h, j))
            G = len(need)
        self.arenas[l].check_reserve(self._credits_at(l), G)

    def _arena_counts(self) -> Dict[str, int]:
        """K6 arena metadata at allocated capacity (contract 2.6; ownership.ARENA_COUNTS)."""
        n_slots = sum(pool.B * pool.H * pool.n_pos for pool in self.pools)
        reserve = (sum(self.L * self.H for r in self.active() if r.credits is not None)
                   + sum(len(c.granted) for r in self.active() for c in r.cohorts))
        return dict(slot_records=n_slots, slot_binds=n_slots, pos_records=sum(ar.P for ar in self.arenas),
                    map_entries=sum(int(ar.pgen_t.numel()) for ar in self.arenas), transfer_binds=len(self.inflight),
                    reserve_counters=reserve)

    def arena_report(self) -> Dict[str, Any]:
        """M2: the physical mode and, in arena mode, the backing and P_l; per layer binds, peak_bound (the EXACT maximum
        of the bound positions: the bound rises only at a bind), peak_reserved (the maximum of bound + E + G over its
        SAMPLES only: after every bind and at the end-of-call scans, checked mode; not an exhaustive temporal maximum);
        the logical bound B * H * C + X (a sufficiency bound, not a measured fit); the S0 adapter counters; and
        cpu_tensor_bytes, the actual numel * element_size of the arena_k / arena_v rows and the pgen_t mirrors (CPU
        tensor bytes: not the K6 geometry convention, not a Python-allocator or GPU measurement, not C81 / B287)."""
        out = dict(mode=self.cfg.physical)
        if self.arena_mode:
            out.update(backing=self.cfg.arena_backing, P=self.P,
                       logical_bound=self.cfg.B * self.H * self.cfg.C + self.cfg.X,
                       binds=[ar.binds for ar in self.arenas], peak_bound=[ar.peak_bound for ar in self.arenas],
                       peak_reserved=[ar.peak_reserved for ar in self.arenas], bound=[ar.n_bound for ar in self.arenas],
                       pool_checks=getattr(self.tp, "pool_checks", None), readbacks=getattr(self.tp, "readbacks", None),
                       cpu_tensor_bytes=dict(
                           arena_k=sum(t.numel() * t.element_size() for t in self.arena_k),
                           arena_v=sum(t.numel() * t.element_size() for t in self.arena_v),
                           pgen_t=sum(ar.pgen_t.numel() * ar.pgen_t.element_size() for ar in self.arenas)))
            if self.abi4:
                out["bias"] = self.bias_report()
            if self.consume:
                out["consume"] = self.consume_report()               # C1.2
        return out

    # C1.1 A9 reporting scope: what bias_report does NOT count, by name (diagnostics and synchronous references).
    BIAS_EXCLUDED = ("Transfer.expect bias rows (S0 readback expectation, checked mode)",
                     "D2H expected rows ec and snapshot sc (checked mode)",
                     "deferred-read validator copies ws / exp / val and identity copies pg",
                     "synchronous eager-reference temporaries (_row_kv's arena bias clone; _d2h_block's list-to-tensor "
                     "construction of the expected ec, kept only as the diagnostic expectation)",
                     "views of the persistent stores (A_r bias rows read by a ring producer or the D2H; ring / arena rows)")

    def bias_report(self) -> Dict[str, Any]:
        """C1.1 A9 (12.3; Codex correction of 2026-10-15), separate quantities, none a Python-allocator or GPU measurement
        or a fit claim:
          persistent_bytes  the ACTUAL numel * element_size of every PERSISTENT bias store at its allocated capacity
                            (host rows; arena and ring rows whatever their state; the admission buffers held now and
                            their exact peak at allocations; no reference gather exists in C1.1);
          operational       the queued OPERATIONAL bias backings (ownership.BiasOpLedger): bytes held now, the EXACT
                            peak at allocation transitions, the most records held at once (in total and per kind),
                            allocations, bytes and releases per kind, and what is still held (outstanding). The K6
                            category bias_staging charges the held bytes at each SAMPLE (a sampled peak, reported in
                            report()["memory"], distinct from this exact peak);
          logical_units     the K6 units (Geometry.bias_elem = the actual element size);
          excluded          BIAS_EXCLUDED, named: diagnostic copies and synchronous eager-reference temporaries;
          counters          c1stat and the transports' bias readbacks / groups / landings."""
        el = lambda ts: sum(t.numel() * t.element_size() for t in (ts or ()))
        u = self.acct.u
        tp = self.tp
        lg = self.bias_ledger
        return dict(dtype=str(self.bias_dtype), element_size=torch.empty((), dtype=self.bias_dtype).element_size(),
                    chain_kernel=self.chain_kernel, stride=self.stride,
                    persistent_bytes=dict(host_c=el(self.host_c), arena_c=el(self.arena_c), ring_c=el(self.ring_c),
                                          admission_held=sum(el(A.get("c")) for A in self.adm),
                                          admission_peak=self.adm_bias_peak_bytes, gather=0),
                    operational=dict(held_bytes=lg.held, peak_bytes=lg.peak, held_records=len(lg.recs),
                                     peak_records=lg.peak_records, max_concurrent=dict(lg.max_concurrent),
                                     allocs=dict(lg.allocs), alloc_bytes=dict(lg.alloc_bytes),
                                     releases=dict(lg.releases), outstanding=lg.outstanding()),
                    logical_units=dict(bias_elem=self.acct.geom.bias_elem, pos_bias=u["pos_bias"],
                                       slot_bias=u["slot_bias"], ck_bias=u["ck_bias"]),
                    excluded=list(self.BIAS_EXCLUDED),
                    counters=dict(sorted(self.c1stat.items())),
                    transport=dict(bias_readbacks=getattr(tp, "bias_readbacks", None),
                                   bias_groups=getattr(tp, "bias_groups", None), bias_lands=getattr(tp, "bias_lands", None)))

    # ---------------------------------------------------------------- M2 B2 (Config.recent "ring" / "eager_matched")
    def _produce(self, l: int, r: Request, h: int, kind: str, key, prov: Tuple, k, v, where: str, c=None) -> None:
        """Contract 11.2 / 11.4 D1: ONE producer at its producer point. ring: the bind transaction (_ring_take with
        backpressure, RingPool.bind), then in checked mode the per-bind LIVE bound (_ring_live_check), the map insert,
        and the payload write enqueued on "compute" after a stream wait on the previous occupancy's D2H event (the
        second safeguard; complete by the FREE rule) with ready_ev after it. eager_matched: a marker event on "compute"
        at the same point, kept in the same map (no entry, no payload, no wait). A second producer of a key is refused.
        C1.1 ABI v4: the payload write includes the entry's bias c (ring_c[l][e]) before the same ready_ev."""
        if self.abi4 and self.ring_mode and c is None:
            raise KStepError("C1.1: the %s producer of uid %s (layer %d h %d, %s) has no bias" % (kind, r.uid, l, h, where))
        m = r.rmap[l][h][kind]
        if key in m:
            raise OwnershipViolation("ring binding: duplicate %s key %s of uid %s (layer %d b %d h %d, %s)"
                                     % (kind, key, r.uid, l, r.b, h, where))
        if not self.ring_mode:
            with self.be.stream("compute"):
                m[key] = self.be.event_rec("compute")
            return
        prep = None
        if self.store is not None:                                     # HOOK-L5 (I9.1): the opt-in executor's payload
            prep = self.store.prepare_produce(self, l, r, h, kind, key, k, v, c)   # is obtained and checked BEFORE
        ring = self.rings[l]                                           # the bind (None: opt-out or A_r source)
        e = self._ring_take(l, r, where)
        g, prev_d2h = ring.bind(e, kind, (r.uid, r.b, h), prov, where)
        if self._checked():
            self._ring_live_check(l, r, h, e, kind, key, where)
        m[key] = (e, g)
        if self.store is not None:                                     # HOOK-B2: the payload write through the store
            ring.set_ready(e, g, self.store.dispatch_produce(self, l, r, h, e, prev_d2h, k, v, c,   # HOOK-B2: its
                                                             kind, key, prepared=prep))         # HOOK-B2: hold owns
            return                                                     # HOOK-B2: the sources (no bias entry)
        sk = k if torch.is_tensor(k) else torch.tensor(k, dtype=torch.int64)
        sv = v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.int64)
        sc = None
        if self.abi4:
            sc = c if torch.is_tensor(c) else torch.tensor([c], dtype=self.bias_dtype)
        with self.be.stream("compute"):
            if prev_d2h is not None:
                self.be.stream_wait("compute", prev_d2h)
            self.be.memcpy(self.ring_k[l][e], sk, "compute")
            self.be.memcpy(self.ring_v[l][e], sv, "compute")
            if sc is not None:                                         # C1.1: the bias, inside the same transaction
                self.be.memcpy(self.ring_c[l][e:e + 1], sc, "compute")
            rev = self.be.event_rec("compute")
            ring.set_ready(e, g, rev)
        if sc is not None and not torch.is_tensor(c):                  # C1.1 A9: a fresh source (an A_r view is not;
            self.bias_ledger.hold("ring_producer", sc, ev=rev)         # A_r is charged) is held until ready_ev

    def _ring_take(self, l: int, r: Request, where: str) -> int:
        """Contract 11.4 backpressure, bounded: the lowest FREE entry; else (1) retirement; (2) a host wait on every
        incomplete event of the OLDEST event-held (DRAINING) entry, counted in ring_backpressure_waits and traced, then
        retirement again (each round frees that entry, so at most Q rounds); (3) CapacityRefusal when no entry is
        event-held. Runs only in a producer, before any Row.ring of the call is built."""
        ring = self.rings[l]
        e = ring.lowest_free()
        while e is None:
            self.rstat["ring_freed"] += ring.retire()
            e = ring.lowest_free()
            if e is not None:
                break
            o = ring.oldest_held()
            if o is None:
                raise CapacityRefusal("ring: layer %d full of logically held entries (LIVE %d of Q_l %d; bind for uid %s)"
                                      % (l, ring.n_live, ring.Q, r.uid))
            evs = ring.gates(o)
            self.ring_waits.append(dict(call=self.call_id, step=self.t, layer=l, entry=o, rgen=ring.rgen[o],
                                        kind=ring.kind[o], owner=ring.owner[o], events=len(evs), uid=r.uid, where=where))
            self.rstat["ring_backpressure_waits"] += 1
            for ev in evs:
                self.be.host_wait(ev)
        return e

    def _ring_live_check(self, l: int, r: Request, h: int, e: int, kind: str, key, where: str) -> None:
        """Checked mode, after EVERY bind (admission included; contract 11.4, revision 6 NB7): LIVE <= R_log on the bound
        entry's stream, THEN LIVE <= B * H * R_log on its layer (both active; the stream bound is the earlier one)."""
        ring = self.rings[l]
        n_s = ring.live_by.get((r.b, h), 0)
        if n_s > self.R_log:
            raise OwnershipViolation("ring: LIVE bound: stream (l %d b %d h %d) holds %d LIVE entries > R_log %d after the "
                                     "bind of entry %d (%s uid %s key %s, %s)" % (l, r.b, h, n_s, self.R_log, e, kind, r.uid,
                                                                                  key, where))
        lim = self.cfg.B * self.H * self.R_log
        if ring.n_live > lim:
            raise OwnershipViolation("ring: LIVE bound: layer %d holds %d LIVE entries > B*H*R_log %d after the bind of "
                                     "entry %d (%s uid %s key %s, %s)" % (l, ring.n_live, lim, e, kind, r.uid, key, where))

    def _rdrop(self, l: int, r: Request, h: int, kind: str, key, why: str) -> None:
        """The logical release of one map entry: removed from the map; ring: LIVE -> DRAINING (FREE only at retirement)."""
        v = r.rmap[l][h][kind].pop(key)
        if self.ring_mode:
            self.rings[l].drain(v[0], v[1], why)

    def _rmove(self, l: int, r: Request, h: int, src: str, dst: str, skey, dkey, prov: Tuple) -> None:
        """PRE -> PROV (execution) or PROV -> EXACT (publication): the same entry (ring: RingPool.retag) or marker."""
        m = r.rmap[l][h]
        if skey not in m[src] or dkey in m[dst]:
            raise OwnershipViolation("ring binding: re-tag %s %s -> %s %s of uid %s (layer %d b %d h %d): source mapped %s, "
                                     "destination mapped %s" % (src, skey, dst, dkey, r.uid, l, r.b, h, skey in m[src],
                                                                dkey in m[dst]))
        v = m[src].pop(skey)
        m[dst][dkey] = v
        if self.ring_mode:
            self.rings[l].retag(v[0], v[1], dst, prov)

    def _ring_ws_held(self, r: Request, l: int, h: int, j: int) -> bool:
        """Contract 11.4: does a WAITING workspace of r still read block j of (l, h) from the recent region (rev6 H1, the
        frozen boundary <= j and j selected at head h)? An executed read is covered by the entry's compute gate."""
        if not self.rev6 or self.mut.rev6_unfrozen_boundary:
            return False
        for c in r.cohorts:
            w = c.ws.get(l)
            if w is not None and w["hist"] <= j and any(j in row[h] for row in w["sel"]):
                return True
        return False

    def _rsync(self, r: Request, why: str) -> None:
        """Drop every map entry of r that is no longer logically LIVE (contract 11.4): EXACT below the boundary without a
        waiting workspace ("h1"); DRAFT not in the epoch's draft KV; PRE without its cohort's workspace; PROV without its
        undecided cohort's provisional KV (`why`: exec / superseded / rollback)."""
        live = {c.key: c for c in r.cohorts if c.state in (PENDING, RUNNING, DONE)}
        bnd = r.hist_blocks * self.bs
        for l in range(self.L):
            for h in range(self.H):
                m = r.rmap[l][h]
                for pos in [p for p in m[R_EXACT] if p < bnd and not self._ring_ws_held(r, l, h, p // self.bs)]:
                    self._rdrop(l, r, h, R_EXACT, pos, "h1")
                for pos in [p for p in m[R_DRAFT] if p not in r.draft_kv[l][h]]:
                    self._rdrop(l, r, h, R_DRAFT, pos, why)
                for key in [k for k in m[R_PRE] if k[0] not in live or l not in live[k[0]].ws]:
                    self._rdrop(l, r, h, R_PRE, key, why)
                for key in [k for k in m[R_PROV] if k[0] not in live or l not in live[k[0]].prov
                            or k[1] not in live[k[0]].prov[l][h]]:
                    self._rdrop(l, r, h, R_PROV, key, why)

    def _ring_ref(self, l: int, r: Request, h: int, qp: int, src, kvv) -> Tuple:
        """Row.ring item of one non-slot position: the LIVE entry of its map (kind, key) with the binding rgen, and the
        eager (k, v) the row attends (the validator's reference)."""
        kind, key = src
        v = r.rmap[l][h][kind].get(key)
        ring = self.rings[l]
        if v is None or ring.state[v[0]] != LIVE or ring.rgen[v[0]] != v[1] or ring.kind[v[0]] != kind:
            raise OwnershipViolation("ring read: no LIVE %s entry for key %s of uid %s (layer %d b %d h %d, call %d): map %s, "
                                     "entry %s" % (kind, key, r.uid, l, r.b, h, self.call_id, v,
                                                   None if v is None else (ring.state[v[0]], ring.rgen[v[0]], ring.kind[v[0]])))
        return (qp, v[0], v[1], kind, kvv)

    def _ring_row(self, l: int, row: Row, rg: List[List[Tuple]]) -> None:
        """Interface v3: Row.ring from the row's items; the eager values for the validators; the (layer, entry, rgen)
        reads whose compute gate _install_reader sets; coverage counters (ring report only)."""
        r = self.reqs[row.uid]
        bnd = r.hist_blocks * self.bs
        row.ring = []
        for h, items in enumerate(rg):
            lst = []
            for qp, e, g, kind, kvv in items:
                self._ring_exp[(id(row), h, qp)] = kvv
                self._ring_reads.append((l, e, g))
                lst.append((qp, e, g, kind))
                self.rstat["ring_reads_%s_%s" % (row.kind, kind)] += 1
                if row.kind == VERIFY and kind == R_EXACT and qp < bnd:
                    self.rstat["ring_reads_h1_retained_exact"] += 1
            row.ring.append(lst)
        if row.kind == VERIFY and not any(row.readable):
            self.rstat["ring_all_recent_verifier_rows"] += 1

    def _ring_retire(self) -> None:
        """Host poll: DRAINING -> FREE for entries whose events completed; an admission buffer is FREE once its ready,
        compute and D2H events completed (non-blocking; never a logical counter)."""
        for ring in self.rings:
            self.rstat["ring_freed"] += ring.retire()
        keep = []
        for A in self.adm:
            if all(ev_done(A[k]) for k in ("ready", "cev", "dev")):
                self.rstat["admission_freed"] += 1
            else:
                keep.append(A)
        self.adm = keep
        if self.store is not None:                                     # HOOK-B2: the passive release of the store's
            self.store.poll()                                         # HOOK-B2: source holds (per-stream rule)

    def _ring_scan(self) -> None:
        """Checked mode, the end of every joint call: every RingPool record (RingPool.check); every map entry of an active
        request names a LIVE entry with its rgen, kind and owner (uid, b, h), 1:1; the LIVE entries are exactly the
        mapped ones; LIVE <= R_log per stream and <= B * H * R_log per layer."""
        lim = self.cfg.B * self.H * self.R_log
        for l, ring in enumerate(self.rings):
            ring.check()
            seen: Dict[int, Tuple] = {}
            for r in self.active():
                for h in range(self.H):
                    for kind in RING_KINDS:
                        for key, (e, g) in r.rmap[l][h][kind].items():
                            if e in seen or ring.state[e] != LIVE or ring.rgen[e] != g or ring.kind[e] != kind \
                                    or ring.owner[e] != (r.uid, r.b, h):
                                raise OwnershipViolation(
                                    "ring binding: map %s key %s of uid %s (layer %d h %d) names entry %d rgen %d; the entry "
                                    "is %s %s rgen %d owner %s (also mapped by %s)" % (kind, key, r.uid, l, h, e, g,
                                                                                       ring.state[e], ring.kind[e],
                                                                                       ring.rgen[e], ring.owner[e],
                                                                                       seen.get(e)))
                            seen[e] = (r.uid, h, kind, key)
            if len(seen) != ring.n_live:
                raise OwnershipViolation("ring binding: layer %d has %d LIVE entries and %d map entries"
                                         % (l, ring.n_live, len(seen)))
            worst = max(ring.live_by.values()) if ring.live_by else 0
            if worst > self.R_log or ring.n_live > lim:
                raise OwnershipViolation("ring: LIVE bound: layer %d holds %d LIVE entries (stream maximum %d) > B*H*R_log %d "
                                         "or R_log %d at the end of call %d" % (l, ring.n_live, worst, lim, self.R_log,
                                                                                self.call_id))

    def _admit_recent(self, r: Request, kv, prep=None) -> None:
        """Contract 11.2 admission (revision 5), ring and eager_matched. The executor's ONE prefill output is the request's
        admission buffer A_r (ring: allocated rows, written on "compute"; eager_matched: a marker), then
          1. every COMPLETE prompt block goes by D2H from A_r to the host rows (_history with adm: after a stream wait on
             A_r's ready event; snapshot and owner record);
          2. only the final partial block (at most bs - 1 positions per stream) binds EXACT entries, in (layer, head,
             position) order, each copied from A_r on "compute";
          3. A_r's gates: a compute event after step 2 and a D2H event after step 1, both recorded also when a step is
             empty. ring: A_r is held (charged in K6) until its ready, compute and D2H events completed (_ring_retire).
        The prompt is not truncated or split; the executor interface is unchanged."""
        if self.store is not None:                                     # HOOK-B2: A_r = the executor's prefill tensors
            return self.store.dispatch_admit(self, r, kv, prepared=prep)   # HOOK-B2 / HOOK-L5 (I9.1: prepared)
        bs, L, H, D = self.bs, self.L, self.H, self.D
        n0 = r.n                                                       # P0 - 1 prompt positions
        r.rmap = [[{kind: {} for kind in RING_KINDS} for _ in range(H)] for _ in range(L)]
        ring = self.ring_mode
        rec = dict(step=self.t, uid=r.uid, b=r.b, P0=r.P0, n=n0, items=n0 * H * L,
                   live_start=[rp.n_live for rp in self.rings] if ring else None)
        self.adm_log.append(rec)
        A = dict(uid=r.uid, b=r.b, n=n0, step=self.t, k=None, v=None, ready=None, cev=None, dev=None)
        srcs = []                                                      # C1.1 A9: the fresh bias sources per layer
        with self.be.stream("compute"):                                # 0. the prefill output and its ready event (D1)
            if ring:
                A["k"] = [torch.full((n0, H, D), POISON, dtype=torch.int64) for _ in range(L)]
                A["v"] = [torch.full((n0, H, D), POISON, dtype=torch.int64) for _ in range(L)]
                if self.abi4:                                          # C1.1: A_r's bias rows (P0 - 1, H) per layer
                    A["c"] = [torch.full((n0, H), POISON, dtype=self.bias_dtype) for _ in range(L)]
                for l in range(L):
                    sk = torch.tensor([[kv[pos][l][h][0] for h in range(H)] for pos in range(n0)],
                                      dtype=torch.int64).reshape(n0, H, D)
                    sv = torch.tensor([[kv[pos][l][h][1] for h in range(H)] for pos in range(n0)],
                                      dtype=torch.int64).reshape(n0, H, D)
                    self.be.memcpy(A["k"][l], sk, "compute")
                    self.be.memcpy(A["v"][l], sv, "compute")
                    if self.abi4:
                        scc = torch.tensor([[kv[pos][l][h][2] for h in range(H)] for pos in range(n0)],
                                           dtype=self.bias_dtype).reshape(n0, H)
                        self.be.memcpy(A["c"][l], scc, "compute")
                        srcs.append(scc)
            A["ready"] = self.be.event_rec("compute")
        for scc in srcs:                                               # C1.1 A9: held until A_r's ready event (the
            self.bias_ledger.hold("admission_src", scc, ev=A["ready"]) # copies that read them precede it)
        blocks0 = r.hist_blocks
        self._history(r, adm=A)                                        # 1. complete prompt blocks: D2H from A_r
        self.rstat["admission_blocks"] += r.hist_blocks - blocks0
        r0 = (n0 // bs) * bs                                           # 2. the final partial block: EXACT entries
        where = "admission step %d" % self.t
        for l in range(L):
            for h in range(H):
                for pos in range(r0, n0):
                    src = (A["k"][l][pos, h], A["v"][l][pos, h]) if ring else (None, None)
                    cc = A["c"][l][pos, h] if (ring and self.abi4) else None    # C1.1: from A_r, as K / V
                    self._produce(l, r, h, R_EXACT, pos, (r.epoch, None, pos), src[0], src[1], where, cc)
                    self.rstat["admission_partial_binds"] += 1
        with self.be.stream("compute"):                                # 3. A_r's two gates
            A["cev"] = self.be.event_rec("compute")
        with self.be.stream("d2h"):
            A["dev"] = self.be.event_rec("d2h")
        rec["exact_after"] = [[len(r.rmap[l][h][R_EXACT]) for h in range(H)] for l in range(L)]
        if ring:
            self.adm.append(A)
            self.adm_peak = max(self.adm_peak, sum(a["n"] * H * L for a in self.adm))
            if self.abi4:                                              # C1.1 A9: the held A_r bias bytes, exact peak
                self.adm_bias_peak_bytes = max(self.adm_bias_peak_bytes, sum(
                    t.numel() * t.element_size() for a in self.adm for t in (a.get("c") or ())))

    def _d2h_block(self, r: Request, l: int, j: int, adm: Optional[Dict]) -> None:
        """Contract 11.3: the D2H of block j of layer l (ring / eager_matched). Source: A_r for an admission block (after
        a stream wait on its ready event), else the ring's EXACT entries (after a stream wait on each entry's ready event;
        each entry's rgen copied at execution; hist_ev installed as its D2H gate), else (eager_matched) a fresh tensor of
        the eager values after the same waits on the markers. Then hist_ev; in checked mode the ordered snapshot of the
        destination rows on the same stream before any later "d2h" enqueue, and its record; the owner record hown."""
        if self.store is not None:                                     # HOOK-B2: the D2H through the store
            return self.store.dispatch_d2h_block(self, r, l, j, adm)  # HOOK-B2: (TensorStore.dispatch_d2h_block)
        bs, b, H = self.bs, r.b, self.H
        r0, r1 = j * bs, (j + 1) * bs
        ek = torch.tensor([[r.pub[l][h][pos][0] for h in range(H)] for pos in range(r0, r1)], dtype=torch.int64)
        evv = torch.tensor([[r.pub[l][h][pos][1] for h in range(H)] for pos in range(r0, r1)], dtype=torch.int64)
        ec = None
        if self.abi4:                                                  # C1.1: the eager exact bias of the block (bs, H)
            ec = torch.tensor([[r.pub[l][h][pos][2] for h in range(H)] for pos in range(r0, r1)], dtype=self.bias_dtype)
        self.pub_seq += 1
        pub = self.pub_seq
        src = rg = rg_exp = csrc = None
        ring = self.rings[l] if self.ring_mode else None
        if adm is None and ring is not None:
            src = []
            for pos in range(r0, r1):
                row = []
                for h in range(H):
                    v = r.rmap[l][h][R_EXACT].get(pos)
                    if v is None or ring.state[v[0]] != LIVE or ring.rgen[v[0]] != v[1]:
                        raise OwnershipViolation("ring D2H: no LIVE EXACT entry for position %d of uid %s (layer %d h %d, "
                                                 "block %d): %s" % (pos, r.uid, l, h, j, v))
                    row.append(v)
                src.append(row)
        with self.be.stream("d2h"):
            if adm is not None:
                self.be.stream_wait("d2h", adm["ready"])
                if ring is not None:
                    self.be.memcpy(self.host_k[l][b, r0:r1], adm["k"][l][r0:r1], "d2h")
                    self.be.memcpy(self.host_v[l][b, r0:r1], adm["v"][l][r0:r1], "d2h")
                    if ec is not None:                                 # C1.1: the bias from A_r, before hist_ev
                        self.be.memcpy(self.host_c[l][b, r0:r1], adm["c"][l][r0:r1], "d2h")
                else:
                    self.be.memcpy(self.host_k[l][b, r0:r1], ek.clone(), "d2h")
                    self.be.memcpy(self.host_v[l][b, r0:r1], evv.clone(), "d2h")
                    if ec is not None:
                        csrc = ec.clone()                              # C1.1 A9: an operational source (ledger)
                        self.be.memcpy(self.host_c[l][b, r0:r1], csrc, "d2h")
            elif ring is not None:
                for row in src:
                    for e, g in row:
                        self.be.stream_wait("d2h", ring.ready[e])
                rg = torch.full((bs, H), -1, dtype=torch.int64)
                for i, row in enumerate(src):
                    for h, (e, g) in enumerate(row):
                        self.be.memcpy(self.host_k[l][b, r0 + i, h], self.ring_k[l][e], "d2h")
                        self.be.memcpy(self.host_v[l][b, r0 + i, h], self.ring_v[l][e], "d2h")
                        if ec is not None:                             # C1.1: the entry's bias, gathered with K / V
                            self.be.memcpy(self.host_c[l][b, r0 + i, h:h + 1], self.ring_c[l][e:e + 1], "d2h")
                        self.be.memcpy(rg[i, h:h + 1], ring.rgen_t[e:e + 1], "d2h")
                rg_exp = [[g for e, g in row] for row in src]
            else:
                for pos in range(r0, r1):
                    for h in range(H):
                        self.be.stream_wait("d2h", r.rmap[l][h][R_EXACT][pos])
                self.be.memcpy(self.host_k[l][b, r0:r1], ek.clone(), "d2h")
                self.be.memcpy(self.host_v[l][b, r0:r1], evv.clone(), "d2h")
                if ec is not None:
                    csrc = ec.clone()                                  # C1.1 A9: an operational source (ledger)
                    self.be.memcpy(self.host_c[l][b, r0:r1], csrc, "d2h")
            hist_ev = self.be.event_rec("d2h")
            r.hist_ev[(l, j)] = hist_ev
            if csrc is not None:                                       # held until hist_ev (the copy precedes it)
                self.bias_ledger.hold("d2h_src", csrc, ev=hist_ev)
            snap = None
            if self._checked():
                sk, sv = torch.empty_like(ek), torch.empty_like(evv)
                self.be.memcpy(sk, self.host_k[l][b, r0:r1], "d2h")
                self.be.memcpy(sv, self.host_v[l][b, r0:r1], "d2h")
                sc = None
                if ec is not None:                                     # C1.1: the bias rows in the same snapshot
                    sc = torch.empty_like(ec)
                    self.be.memcpy(sc, self.host_c[l][b, r0:r1], "d2h")
                snap = (sk, sv, self.be.event_rec("d2h"), sc)
        if src is not None:
            for row in src:
                for e, g in row:
                    ring.note_d2h(e, g, hist_ev)
            self.rstat["d2h_ring_publications"] += 1
        self.hown[(l, b, j)] = (r.uid, pub)
        r.hist_pub[(l, j)] = pub
        self.rstat["d2h_publications"] += 1
        if snap is not None:
            self.d2h_checks.append(dict(uid=r.uid, l=l, b=b, j=j, pub=pub, hist_ev=hist_ev, snap_ev=snap[2], sk=snap[0],
                                        sv=snap[1], ek=ek, ev=evv, rg=rg, rg_exp=rg_exp,
                                        src=None if src is None else [[e for e, g in row] for row in src],
                                        call=self.call_id, step=self.t, seen=None))
            if ec is not None:                                         # C1.1: the bias snapshot and its expectation
                self.d2h_checks[-1].update(sc=snap[3], ec=ec)
            self.rstat["d2h_snapshots"] += 1

    def _d2h_poll(self, final: bool = False) -> None:
        """Contract 11.3 (1), checked mode, non-blocking (and once more after the final synchronize): at the first poll at
        which a publication's hist_ev is complete, whether the LIVE host rows still equal its eager exact KV is RECORDED
        (d2h_seen: a later writer may already have overwritten them; a witness only, it decides nothing); the snapshot is
        validated once its own event completed: the identity of every gathered ring entry first, then the bytes."""
        keep = []
        for rec in self.d2h_checks:
            if rec["seen"] is None and (final or ev_done(rec["hist_ev"])):
                r0 = rec["j"] * self.bs
                live_k = self.host_k[rec["l"]][rec["b"], r0:r0 + self.bs]
                live_v = self.host_v[rec["l"]][rec["b"], r0:r0 + self.bs]
                same = torch.equal(live_k, rec["ek"]) and torch.equal(live_v, rec["ev"])
                if rec.get("ec") is not None:                          # C1.1: the witness covers the bias rows too
                    same = same and torch.equal(self.host_c[rec["l"]][rec["b"], r0:r0 + self.bs], rec["ec"])
                rec["seen"] = "finish" if final else "poll"
                req = self.reqs.get(rec["uid"])
                self.d2h_seen.append(dict(uid=rec["uid"], l=rec["l"], b=rec["b"], j=rec["j"], pub=rec["pub"],
                                          call=self.call_id, step=self.t, phase=rec["seen"], live_equal=same,
                                          ended=req is None or req.done, owner=self.hown.get((rec["l"], rec["b"], rec["j"]))))
            if final or ev_done(rec["snap_ev"]):
                self._d2h_check(rec, rec["sk"], rec["sv"], rec.get("sc"))
            else:
                keep.append(rec)
        self.d2h_checks = keep

    def _d2h_check(self, rec: Dict, k, v, c=None) -> None:
        if rec["rg"] is not None:
            for i, row in enumerate(rec["rg_exp"]):
                for h, g in enumerate(row):
                    if int(rec["rg"][i, h]) != g:
                        raise StaleReader("stale reader: D2H uid %s l %d b %d block %d entry %d (head %d position %d) rgen "
                                          "captured %d, at read %d" % (rec["uid"], rec["l"], rec["b"], rec["j"],
                                                                       rec["src"][i][h], h, rec["j"] * self.bs + i, g,
                                                                       int(rec["rg"][i, h])))
        self._d2h_bytes(rec, k, v, c)
        self.rstat["d2h_snapshots_validated"] += 1

    def _d2h_bytes(self, rec: Dict, k, v, c=None) -> None:
        if not (torch.equal(k, rec["ek"]) and torch.equal(v, rec["ev"])):
            raise ReadCorruption("D2H readback mismatch: uid %s l %d b %d block %d pub %d"
                                 % (rec["uid"], rec["l"], rec["b"], rec["j"], rec["pub"]))
        if rec.get("ec") is not None and c is not None:                # C1.1: then the bias rows of the same snapshot
            if not torch.equal(c, rec["ec"]):
                raise ReadCorruption("D2H readback mismatch: uid %s l %d b %d block %d pub %d (bias rows)"
                                     % (rec["uid"], rec["l"], rec["b"], rec["j"], rec["pub"]))
            self.c1stat["d2h_bias_snapshots_validated"] += 1

    def _ring_counts(self) -> Dict[str, int]:
        """K6 ring-mode counts (ownership.RING_COUNTS, contract 11.4)."""
        c = dict(ring_live_exact=0, ring_live_draft=0, ring_live_prov=0, ring_idle=0)
        for ring in self.rings:
            rc = ring.counts()
            c["ring_live_exact"] += rc[R_EXACT]
            c["ring_live_draft"] += rc[R_DRAFT]
            c["ring_live_prov"] += rc[R_PRE] + rc[R_PROV]
            c["ring_idle"] += rc["idle"]
        c["ring_entries"] = sum(ring.Q for ring in self.rings)
        c["ring_map_entries"] = sum(len(r.rmap[l][h][kind]) for r in self.active() if r.rmap is not None
                                    for l in range(self.L) for h in range(self.H) for kind in RING_KINDS)
        c["admission_pos"] = sum(A["n"] * self.H * self.L for A in self.adm)
        c["admission_records"] = len(self.adm)
        return c

    def ring_report(self) -> Dict[str, Any]:
        """M2 B2 report (never a logical counter): the mode; D2H publications, snapshots and their validations, owner
        checks, first-complete-poll observations (live rows overwritten at a poll); ring mode: Q_l, R_log, S_ring, the
        LIVE bounds, per-layer binds and peaks (EXACT: they change only at binds), backpressure waits with their trace,
        ring reads validated, retirements, admissions (A_r) and the EXACT admission_kv peak (it rises only at an
        allocation), coverage counters, and cpu_tensor_bytes (numel * element_size; not a K6 or GPU measurement)."""
        s = self.rstat
        out = dict(mode=self.recent, d2h_publications=s["d2h_publications"], d2h_snapshots=s["d2h_snapshots"],
                   d2h_snapshots_validated=s["d2h_snapshots_validated"], hown_checks=s["hown_checks"],
                   d2h_seen=len(self.d2h_seen),
                   d2h_live_overwritten_at_poll=sum(1 for x in self.d2h_seen if x["phase"] == "poll" and not x["live_equal"]),
                   admissions=len(self.adm_log))
        if self.ring_mode:
            u = self.acct.u
            out.update(Q=self.Q, R_log=self.R_log, S_ring=self.S_ring, live_bound=self.cfg.B * self.H * self.R_log,
                       binds=[rp.binds for rp in self.rings], peak_live=[rp.peak_live for rp in self.rings],
                       peak_stream=[rp.peak_stream for rp in self.rings], live=[rp.n_live for rp in self.rings],
                       draining=[rp.n_draining for rp in self.rings],
                       ring_backpressure_waits=s["ring_backpressure_waits"], waits=[dict(w) for w in self.ring_waits],
                       ring_reads_registered=s["ring_reads_registered"], ring_reads_validated=s["ring_reads_validated"],
                       ring_freed=s["ring_freed"], admission_freed=s["admission_freed"], admission_held=len(self.adm),
                       admission_kv_peak_items=self.adm_peak, admission_kv_peak_bytes=self.adm_peak * u["pos_kv"],
                       coverage={k: v for k, v in sorted(s.items()) if k.startswith(("ring_reads_", "ring_all_", "ring_only",
                                                                                     "admission_", "d2h_ring"))},
                       h1_late_drains=s["ring_h1_late_drains"],
                       cpu_tensor_bytes=dict(ring_k=sum(t.numel() * t.element_size() for t in self.ring_k),
                                             ring_v=sum(t.numel() * t.element_size() for t in self.ring_v),
                                             rgen_t=sum(rp.rgen_t.numel() * rp.rgen_t.element_size()
                                                        for rp in self.rings)))
        return out

    def _force_for_layer(self, l: int) -> None:
        """Verifier-only pass (D4): wait (host) for the in-flight blocks the waiting cohorts at layer l need; after a
        failed verifier allocation at this layer, also for every in-flight transfer and every releasing slot of the
        layer (the deferred CPU model completes nothing on its own)."""
        need = set()
        for r in self.active():
            for c in r.cohorts:
                if c.state == RUNNING and c.next_layer == l and l in c.ws:
                    for h in range(self.H):
                        st = self.pools[l].stream(r.b, h)
                        for sel in c.ws[l]["sel"]:
                            for j in sel[h]:
                                if j < r.hist_blocks and j in st.pending:
                                    need.add((r.b, h, st.pending[j]))
        force_all = l in self._vfail
        for tr in sorted(self.inflight.values(), key=lambda x: x.seq):
            if tr.layer == l and (force_all or (tr.b, tr.h, tr.slot) in need):
                self.tp.force(tr)
                self.stats["forced_transfers"] += 1
        if force_all:
            for ev in self.pools[l].releasing_gates():
                self.be.host_wait(ev)
                self.stats["forced_readers"] += 1
            for key_l, (fr, lr) in list(self.key_readers.items()):
                if key_l[1] == l and lr is not None and not ev_done(lr):
                    self.be.host_wait(lr)
                    self.stats["forced_readers"] += 1
            self._vfail.discard(l)

    # ================================================================== the joint call
    def _d3a(self, r: Request, c: Cohort, l: int) -> bool:
        if self.mut.cohort_order_skip:
            return True
        i = r.cohorts.index(c)
        if i == 0:
            return True
        prev = r.cohorts[i - 1]
        if prev.state == ACCEPTED:
            return True
        return prev.exec_call.get(l, self.call_id) < self.call_id

    def _chain_base(self, r: Request, c: Cohort, l: int) -> List[CompressedChain]:
        i = r.cohorts.index(c)
        if i > 0 and r.cohorts[i - 1].state != ACCEPTED:
            prev = r.cohorts[i - 1]
            if l not in prev.chain:
                raise KStepError("D3a: %r needs the layer-%d chain of %r, which has not executed it" % (c, l, prev))
            return prev.chain[l]
        return r.comp[l]

    def _verifier_pre(self, r: Request, c: Cohort, l: int, pre) -> None:
        """Phase 1 of a cohort at layer l, computed ONCE and kept as workspace (D3: not recomputed on a retry)."""
        if l in c.ws:
            raise KStepError("D3: phase 1 of %r at layer %d would be recomputed" % (c, l))
        self.stats["verifier_pre_rows"] += len(c.positions)
        chain = [x.copy() for x in self._chain_base(r, c, l)]
        if self.abi4:                                                  # C1.1 V3: the row-local prefix, row by row
            vis = [row_prefixes(chain[h], c.positions, [(pre[i][1][h], pre[i][3][h]) for i in range(len(c.positions))],
                                "cohort %s" % (c.key,), "l %d h %d" % (l, h)) for h in range(self.H)]
            sel = [[self.ex.select(l, h, pre[i][0][h], vis[h][i], pos // self.bs) for h in range(self.H)]
                   for i, pos in enumerate(c.positions)]
            self.c1stat["row_prefix_checks"] += len(c.positions) * self.H
        else:
            for i, pos in enumerate(c.positions):
                for h in range(self.H):
                    chain[h].append(pre[i][1][h])
            sel = []
            for i, pos in enumerate(c.positions):
                sel.append([self.ex.select(l, h, pre[i][0][h], chain[h].cks, pos // self.bs) for h in range(self.H)])
        c.ws[l] = dict(pre=pre, sel=sel, chain=chain, hist=r.hist_blocks)   # rev6 H1: the boundary of this workspace
        if self.recent != "eager":                                     # M2 B2: the cohort's PRE KV (producer D1)
            for i, pos in enumerate(c.positions):
                for h in range(self.H):
                    self._produce(l, r, h, R_PRE, (c.key, pos), (c.epoch, c.key, pos), pre[i][1][h], pre[i][2][h],
                                  "call %d step %d" % (self.call_id, self.t), pre[i][3][h] if self.abi4 else None)

    def _need(self, r: Request, sel_rows, l: int, hist: Optional[int] = None):
        """Per head: (ready, inflight, missing) historical blocks of a set of rows' selections."""
        out = []
        for h in range(self.H):
            st = self.pools[l].stream(r.b, h)
            hb = r.hist_blocks if hist is None else hist
            blocks = sorted({j for sel in sel_rows for j in sel[h] if j < hb})
            ready = [j for j in blocks if st.resident(j) is not None]
            infl = [j for j in blocks if st.resident(j) is None and j in st.pending]
            miss = [j for j in blocks if st.resident(j) is None and j not in st.pending]
            out.append((ready, infl, miss))
        return out

    def _call(self, l: int, rows: List[Tuple[Request, Draft]]) -> bool:
        """One joint layer call. Returns True when the call finished (published, _call_publish); I17 (deferred "one"):
        False when the consume call is PENDING (submitted, its ticket committed, no host wait): self.pending_call names
        it and step() publishes it later, once.
        I15 / I17 (review fix): with tickets on, the ticket capacity is reserved FIRST, before ANY state of this call
        changes (call id, poll, progress hook, productions, protections, chains, recency, gathers): a CapacityRefusal
        leaves the dispatcher, the pools, the rings, the store and the ledgers exactly as they were (only the tickets'
        stats["refused"] counts it). A call that ends with no consume row aborts its reservation (no read was issued);
        a failure after the reservation keeps it open (fail closed: the reset refuses; abort only before any issued
        read, I15). The reservation is open during this call's _poll(), its progress hook and its "poll" observer
        call; none of them reads the tickets today (a hook that does sees outstanding() count it)."""
        rid = None
        if self.tickets is not None:
            rid = self.tickets.reserve(self.call_id + 1, l)            # BEFORE any change of this call
        self.call_id += 1
        cid = self.call_id
        self._read_keys = set()
        self._n_new_val = 0
        self._n_bias_reads = 0
        self._ring_reads, self._ring_exp, self._n_ring_val = [], {}, 0
        self.call_step[cid] = self.t
        if self.arena_mode:
            for ar in self.arenas:
                ar.checked = self._checked()                           # M2: the arena check points follow checked mode
            for rp in self.rings:
                rp.checked = self._checked()
        if self.progress is not None:
            self.progress(self, cid)
        self._poll()
        if self.observer is not None:
            self.observer(self, "poll", cid, l)
        requests: List[Tuple] = []
        exec_cohorts: List[Tuple[Request, Cohort]] = []
        eligible = []
        for r in self.active():
            for c in r.cohorts:
                if c.state == RUNNING and c.next_layer == l:
                    if self._d3a(r, c, l):
                        eligible.append((r, c))
                    else:
                        self.stats["cohort_waits_d3a"] += 1             # R3.2: a counted D3a wait
        # phase 1 of the joint call: ONE executor call for every row that needs it (verifier rows without a saved
        # workspace at this layer, then the draft rows)
        items, owners_pre = [], []
        for r, c in eligible:
            if l not in c.ws:
                owners_pre.append(("v", r, c, len(items)))
                items.extend((c.hidden[i], pos) for i, pos in enumerate(c.positions))
        for r, dr in rows:
            if r.cur is dr:
                owners_pre.append(("d", r, dr, len(items)))
                items.append((dr.hidden, dr.pos))
        pre_all = self.ex.joint_pre(l, items) if items else []
        draft_pre_raw = {}
        for kind, r, obj, at in owners_pre:
            if kind == "v":
                self._verifier_pre(r, obj, l, pre_all[at:at + len(obj.positions)])
            else:
                draft_pre_raw[id(obj)] = pre_all[at]
        if self.rev6:
            for r, c in eligible:
                self._row_bound(r, c.ws[l]["sel"], l, "verifier %s" % (c.key,), self._vhist(r, c, l))
            self._rev6_verifiers(l, eligible, requests, exec_cohorts)  # G1-G6 (capacity mode rev6)
        for r, c in ([] if self.rev6 else eligible):
            need = self._need(r, c.ws[l]["sel"], l)
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                for blocks in (need[h][1:] if self.mut.no_verifier_ready_protection else need[h]):
                    for j in blocks:
                        self._protect(st, l, j, c.key, VERIFY)     # K2: the exact selection, while it waits
                for j in need[h][1]:
                    self._add_demand(st, j, VERIFY)                # R2.3: a verifier request for an in-flight block
            ok = all(not miss and not infl for _, infl, miss in need)
            if not ok and self.mut.read_not_ready:
                ok = all(not miss for _, infl, miss in need)       # NEGATIVE CONTROL: reads in-flight slots
            if ok:
                exec_cohorts.append((r, c))
            else:
                c.waits += 1
                self.stats["cohort_waits"] += 1
                for h in range(self.H):
                    for j in need[h][2]:
                        requests.append((VERIFY, r, h, j, c.key))
        # draft rows: phase 1, selection, protection, availability, misses
        draft_pre = {}
        for r, dr in rows:
            if r.cur is not dr:
                continue
            q, k, v = draft_pre_raw[id(dr)][:3]
            cb = draft_pre_raw[id(dr)][3] if self.abi4 else None       # C1.1: the draft row's own bias per head
            for h in range(self.H):
                if r.dchain[l][h].count != dr.pos:
                    raise KStepError("D6: the draft chain of uid %s (l %d h %d) holds %d keys at draft position %d"
                                     % (r.uid, l, h, r.dchain[l][h].count, dr.pos))
            sel = [self.ex.select(l, h, q[h], r.dchain[l][h].cks, dr.pos // self.bs) for h in range(self.H)]
            if self.abi4:                                              # C1.1 instrumentation: the windows select read
                dr.cks[l] = [list(r.dchain[l][h].cks) for h in range(self.H)]
                self.c1stat["draft_cks_records"] += self.H
            key = (r.uid, r.epoch, dr.m)
            if self.rev6:
                self._row_bound(r, [sel], l, "draft s%d" % dr.s)
            avail = []
            for h in range(self.H):
                r.draft_kv[l][h][dr.pos] = (k[h], v[h]) if cb is None else (k[h], v[h], cb[h])
                r.dchain[l][h].append(k[h] if cb is None else (k[h], cb[h]))
                if self.recent != "eager":                             # M2 B2: the draft row's DRAFT KV (producer D1)
                    self._produce(l, r, h, R_DRAFT, dr.pos, (r.epoch, None, dr.pos), k[h], v[h],
                                  "call %d step %d" % (cid, self.t), None if cb is None else cb[h])
                st = self.pools[l].stream(r.b, h)
                a = []
                for j in sel[h]:
                    if j >= r.hist_blocks:
                        continue
                    if not (self.mut.no_draft_hit_protection and st.resident(j) is not None):
                        self._protect(st, l, j, key, DRAFT)            # K2: hits AND misses, for the cohort of d_s
                    if st.resident(j) is not None:
                        a.append(j)
                    elif j not in st.pending:
                        requests.append((DRAFT, r, h, j, key))
                    else:
                        self._add_demand(st, j, DRAFT)                 # R2.3: joins the transfer already in flight
                avail.append(a)
            dr.sel[l], dr.avail[l] = sel, avail
            draft_pre[id(dr)] = (q, k, v) if cb is None else (q, k, v, cb)
        # R2.2: verifier reads never touch the logical LRU; they are logged separately (cohort order, position order)
        for r, c in exec_cohorts:
            hv = self._vhist(r, c, l)
            for i, pos in enumerate(c.positions):
                for h in range(self.H):
                    st = self.pools[l].stream(r.b, h)
                    for j in c.ws[l]["sel"][i][h]:
                        if j < hv:
                            self.vread_log.append((cid, l, r.b, h, j, (c.m, pos)))
                            if self.mut.verifier_touches_lru:
                                st.touch(j)                            # NEGATIVE CONTROL (R2.6): the M1 rule
        # phase 2: reads + executor
        jrows: List[Row] = []
        owners: List[Tuple] = []
        read_slots: List[Tuple[int, int, int, Any]] = []
        for r, c in exec_cohorts:
            pre, sel = c.ws[l]["pre"], c.ws[l]["sel"]
            hv = self._vhist(r, c, l)                                  # rev6 H1: the workspace's own history boundary
            own = {pos: i for i, pos in enumerate(c.positions)}
            for i, pos in enumerate(c.positions):
                kv = []
                addr = [] if self.arena_mode else None                 # M2: per head [(block, phys, pgen)]
                rg = [] if self.ring_mode else None                    # M2 B2: per head [(pos, entry, rgen, kind, kv)]
                for h in range(self.H):
                    def rsrc(qp, c=c):                                 # M2 B2: which map holds the KV the row reads
                        if qp < r.n:
                            return R_EXACT, qp
                        if qp in own:
                            return R_PRE, (c.key, qp)
                        return R_PROV, (self._prov_cohort(r, c, l, qp).key, qp)

                    def recent(qp, h=h, i=i, c=c):
                        if qp < r.n:
                            if qp not in r.pub[l][h]:
                                raise KStepError("rev6 H1: verifier %s reads position %d (layer %d head %d), which left the "
                                                 "recent region before its last reader" % (c.key, qp, l, h))
                            if qp < r.hist_blocks * self.bs:
                                self.stats["rev6_h1_retained_reads"] += 1   # a retained position (now historical)
                            return r.pub[l][h][qp]
                        if qp in own:
                            if own[qp] > i:
                                raise KStepError("causality: row %d reads a later position %d" % (pos, qp))
                            ii = own[qp]
                            if self.abi4:                              # C1.1: the PRE item with its own bias
                                return (pre[ii][1][h], pre[ii][2][h], pre[ii][3][h])
                            return (pre[ii][1][h], pre[ii][2][h])
                        return self._prov(r, c, l, h, qp)
                    ad = [] if addr is not None else None
                    ro = [] if rg is not None else None
                    kv.append(self._row_kv(r, l, h, pos, sel[i][h], None, recent, read_slots, c.key, hv, ad,
                                           rsrc if ro is not None else None, ro))
                    if addr is not None:
                        addr.append(sorted(ad))
                    if rg is not None:
                        rg.append(ro)
                row = Row(r.uid, r.b, pos, VERIFY, c.key, sel[i], pos,
                          [[j for j in sel[i][hh] if j < hv] for hh in range(self.H)],
                          c.hidden[i], pre[i][0], kv, addr)
                jrows.append(row)
                if rg is not None:
                    self._ring_row(l, row, rg)
                owners.append(("v", r, c, i))
        for r, dr in rows:
            if id(dr) not in draft_pre:
                continue
            q = draft_pre[id(dr)][0]
            kv = []
            addr = [] if self.arena_mode else None                     # M2: per head [(block, phys, pgen)]
            rg = [] if self.ring_mode else None                        # M2 B2: per head [(pos, entry, rgen, kind, kv)]
            for h in range(self.H):
                def recent(qp, h=h):
                    if qp < r.n:
                        return r.pub[l][h][qp]
                    return r.draft_kv[l][h][qp]

                def rsrc(qp):                                          # M2 B2: exact below n, the epoch's draft KV above
                    return (R_EXACT, qp) if qp < r.n else (R_DRAFT, qp)
                ad = [] if addr is not None else None
                ro = [] if rg is not None else None
                kv.append(self._row_kv(r, l, h, dr.pos, dr.sel[l][h], dr.avail[l][h], recent, read_slots,
                                       (r.uid, r.epoch, dr.m), None, ad, rsrc if ro is not None else None, ro))
                if addr is not None:
                    addr.append(sorted(ad))
                if rg is not None:
                    rg.append(ro)
            row = Row(r.uid, r.b, dr.pos, DRAFT, (r.uid, r.epoch, dr.m), dr.sel[l], dr.pos, dr.avail[l], dr.hidden, q, kv,
                      addr)
            jrows.append(row)
            if rg is not None:
                self._ring_row(l, row, rg)
            owners.append(("d", r, dr, None))
        if read_slots and not self.arena_mode:                         # F4: the eager executor's host order (per stream)
            with self.be.stream("compute"):
                rev = self.be.event_rec("compute")
            self._install_reader(l, read_slots, rev, "compute")      # HOOK-C (lead): rev was recorded on "compute"
            self._arm_validators(rev)
        self._draft_recency(l, rows, draft_pre, cid)
        ctx = dict(cid=cid, rows=rows, owners=owners, jrows=jrows, exec_cohorts=exec_cohorts, requests=requests)
        if self.consume:                                               # C1.2 (12.9.2): the serialized consumer
            if rid is not None and not jrows:                          # I15: no consume row, no read issued: the
                self.tickets.abort(rid)                                # reservation (taken at the call's start) is
                rid = None                                             # dropped
            # A failure from the reservation to the commit KEEPS the open reservation: _consume_post may already have
            # issued reads (abort is only for an allocation that failed before any read, I15), so the reservation and
            # the call's keyed holds block a reset until the root decides (fail closed; no forced cleanup).
            handle, cev = self._consume_post(l, jrows)                 # 1. gather backings held at once
            self._check_post_reads(jrows, read_slots, cev)             # 2. the same registration checks, then the
            self._install_reader(l, read_slots, cev)                   #    reader gates and the validators, BEFORE
            self._arm_validators(cev)                                  #    any wait (no poll or release in between)
            if self.tickets is None:
                outs = self._consume(l, jrows, handle, cev)            # 3-6. wait, identity, resolve, compare, release
            elif not jrows:
                outs = []
            else:                                                      # HOOK-L8 (I14): one result ticket
                tid = self._ticket_commit(rid, l, jrows, owners, handle, cev)
                if self.deferred:                                      # I17: return PENDING, no host wait
                    self.pending_call = dict(l=l, cid=cid, tid=tid, cev=cev, ctx=ctx)
                    self.dstat["submitted"] += 1
                    self._forward_handbacks()
                    return False
                outs = self._consume_ticket(l, ctx, cev)               # the serialized wait is kept
        elif self.arena_mode:                                          # M2 B1 (contract 3.1): the executor's own event
            outs, read_ev = self.ex.joint_post(l, jrows) if jrows else ([], None)
            self._check_post_reads(jrows, read_slots, read_ev)
            self._install_reader(l, read_slots, read_ev)               # IMMEDIATELY: no poll or release in between
            self._arm_validators(read_ev)
        else:
            outs = self.ex.joint_post(l, jrows) if jrows else []
        return self._call_publish(l, ctx, outs)

    def _call_publish(self, l: int, ctx: Dict[str, Any], outs: List[Any]) -> bool:
        """The publication of one layer call (the former second half of _call, in its order): the rows' outputs, the
        executing cohorts' per-layer state and next_layer, the release queue, the D1 accounting, the call's transfer
        issue, the samples, the invariant checks, the observer. I17: a row whose output is _DISCARDED (its ticket found
        it stale: a changed identity) publishes nothing, and its cohort is not advanced (no replacement is advanced
        by a late output). Returns True (the call is finished)."""
        cid, rows, owners, jrows = ctx["cid"], ctx["rows"], ctx["owners"], ctx["jrows"]
        exec_cohorts, requests = ctx["exec_cohorts"], ctx["requests"]
        if self.cancelled:                                             # I20: a request torn down at this call's resume
            n0 = len(requests)                                         # issues no transfer (only after a cancellation)
            requests = [q for q in requests if not q[1].done]
            self.cstat["transfer_requests_dropped"] += n0 - len(requests)
        dropped = {id(obj) for (kind, r, obj, i), out in zip(owners, outs) if out is _DISCARDED}
        for (kind, r, obj, i), row, out in zip(owners, jrows, outs):
            if out is _DISCARDED:
                continue
            if self.cancelled and r.done:                              # I20 (fail closed): a cancelled request's row
                raise KStepError("I20: call %d would publish a row of the cancelled request %s (incarnation %s, %s "  # is
                                 "row at %d, layer %d): its ticket applied it" % (cid, r.uid, r.inc, row.kind,   # never
                                                                                  row.pos, l))   # published
            if kind == "v":
                c = obj
                c.hidden[i] = out
                c.row_layers += 1
                self.stats["verifier_row_layers"] += 1
            else:
                obj.hidden = out
                obj.next_layer = l + 1                                 # I17: the draft row's layer progress
        for r, c in exec_cohorts:
            if id(c) in dropped:                                       # I17: its rows were discarded: no advance
                self.dstat["cohort_layers_discarded"] += 1
                continue
            pre = c.ws[l]["pre"]
            c.prov[l] = [{pos: (pre[i][1][h], pre[i][2][h]) + ((pre[i][3][h],) if self.abi4 else ())   # C1.1: with c
                          for i, pos in enumerate(c.positions)} for h in range(self.H)]
            if self.recent != "eager":                                 # M2 B2: PRE -> PROV, the same entries (11.2)
                for pos in c.positions:
                    for h in range(self.H):
                        self._rmove(l, r, h, R_PRE, R_PROV, (c.key, pos), (c.key, pos), (c.epoch, c.key, pos))
            c.chain[l] = c.ws[l]["chain"]
            c.vsel[l] = [[list(x) for x in row] for row in c.ws[l]["sel"]]
            c.vhist[l] = c.ws[l]["hist"]
            c.exec_call[l] = cid
            self._vfetch_drop((c.key, l), "exec")                      # N3 P2: the layer-l obligation is met
            if self.rev6 and not self.mut.rev6_release_recent_before_gate:
                self.h1_reads[(c.key, l)] = (c.ws[l]["hist"], [[list(x) for x in row] for row in c.ws[l]["sel"]])  # H1
            del c.ws[l]
            c.next_layer = l + 1
            self.exec_log.append((cid, l, r.uid, c.epoch, c.m))
            self.stats["progress"] += 1
            self.stats["useful"] += 1                                  # N3 P1: a cohort executed a layer
            if self.mut.release_at_execution:                          # NEGATIVE CONTROL (R1.6 early release)
                self.rel_layers.setdefault(c.key, {})[l] = False
                self._release_refs(c.key, l)
            elif not self.mut.release_only_at_decision:                # (the control keeps the M1 decision-only rule)
                self._queue_release(c.key, l)                          # R1.1: behind the (key, l) reader gate
            if self.rev6 and self.mut.rev6_release_recent_before_gate:
                self._h1_release(r, c.key, l)                          # NEGATIVE CONTROL (H1): before the reader gate
            if l == self.L - 1:
                c.p = {c.a + i: self.ex.dist(c.hidden[i]) for i in range(len(c.positions))}
                c.hidden = []                                          # K6: the rows' hidden states are no longer held
                c.state = DONE
        if self.recent != "eager":                                     # M2 B2: a retained EXACT entry whose last WAITING
            for r in list({id(r): r for r, _ in exec_cohorts}.values()):   # holder executed now drains (its read is
                if r.done:                                             # I20: a cancelled request's entries drained at
                    continue                                           # its end
                self._rsync(r, "exec")                                 # already the compute gate, installed above)
        # D1: per stream SUM and UNION of the selected blocks of the executing rows
        per = defaultdict(lambda: [0, set(), 0, set(), 0])
        for row in jrows:
            for h in range(self.H):
                x = per[(row.b, h)]
                x[0] += len(row.sel[h])
                x[1].update(row.sel[h])
                hist = (list(row.readable[h]) if row.kind == VERIFY       # a verifier row: its workspace's boundary
                        else [j for j in row.sel[h] if j < self.reqs[row.uid].hist_blocks])
                x[2] += len(hist)
                x[3].update(hist)
                x[4] += 1
        for (b, h), x in per.items():
            self.share.append((cid, l, b, h, x[4], x[0], len(x[1]), x[2], len(x[3])))
        self._issue(l, requests, cid)
        if self.rev6 and not self.mut.rev6_credit_leak:                # SC3: the unused credits of layer l return now
            for r, dr in rows:
                if r.credits is not None:
                    self.stats["rev6_credits_returned"] += sum(r.credits[l])
                    r.credits[l] = [0] * self.H
        self._sample_memory()
        if self.rev6_guard:
            self._rev6g_sample(l)
        if self.cfg.check_invariants and not self.mut.active():
            for st in self.pools[l].streams.values():
                st.check()
            if self.rev6:
                self._rev6_check(l)
            s = self.stats
            if s["progress"] != s["useful"] + s["traffic"] or self.vfetch_ids != sum(len(v) for v in self.vfetch.values()):
                raise OwnershipViolation("N3 P1/P2: progress %d != useful %d + traffic %d, or %d live verify-fetch ids != the "
                                         "records' %d" % (s["progress"], s["useful"], s["traffic"], self.vfetch_ids,
                                                          sum(len(v) for v in self.vfetch.values())))
            if self.arena_mode:
                for ll, pool in enumerate(self.pools):                 # M2 2.3: the full scan of every layer (G = 0: a
                    pool.check_arena(self._credits_at(ll), 0)          # grant is issued in the call that grants it)
            if self.ring_mode:
                self._ring_scan()                                      # M2 B2: records, maps 1:1, rgen_t, LIVE bounds
        if self.observer is not None:
            self.observer(self, "call", cid, l)
        self._forward_handbacks()                                      # I17 / I16: the call's ring productions
        return True

    def _prov(self, r: Request, c: Cohort, l: int, h: int, qp: int):
        for cc in r.cohorts:
            if cc is c:
                break
            if cc.state in (ACCEPTED, CANCELLED):
                continue
            if qp in cc.positions:
                if l not in cc.prov:
                    raise KStepError("D3a: %r reads position %d of %r at layer %d before that cohort executed it" % (c, qp, cc, l))
                return cc.prov[l][h][qp]
        raise KStepError("verifier %r: position %d (n=%d) has no published, provisional or own KV" % (c, qp, r.n))

    def _prov_cohort(self, r: Request, c: Cohort, l: int, qp: int) -> Cohort:
        """M2 B2: the predecessor cohort whose provisional KV _prov returns for position qp (the same search)."""
        for cc in r.cohorts:
            if cc is c:
                break
            if cc.state in (ACCEPTED, CANCELLED):
                continue
            if qp in cc.positions:
                return cc
        raise KStepError("verifier %r: position %d (n=%d) has no provisional KV" % (c, qp, r.n))

    def _draft_recency(self, l: int, rows, draft_pre, cid: int) -> None:
        """R2.1 (a), (c) under the ORDERING CONTRACT O1 (repair 1c; Codex decision 2026-10-12): the DRAFT-ONLY recency of
        the call at layer l, after the rows' reads were enqueued (a slot that an eviction releases here is FREE only after
        this call's reader event). Per stream and draft row: the locally READY historical blocks the row read (dr.avail),
        in ASCENDING block id, each either moves to MRU (an LRU hit) or enters the LRU at MRU (a protected-only or VRES
        entry: draft acquisition, no transfer, no new slot); only AFTER the whole batch do the least recent entries leave
        while the LRU holds more than C (protected ones become protected-only, others are released through the reader
        gate). The logical result equals the reference batch update then oldest eviction (memory_replay.py prepare_lru on
        the ready snapshot; tests/kstep_toy.py ref_order_memory_replay). A TRUE miss enters only at its observed arrival
        with draft demand (O2, Stream.admit); verifier reads and verifier-only arrivals never change this order."""
        for r, dr in rows:
            if id(dr) not in draft_pre:
                continue
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                for j in sorted(set(dr.avail[l][h])):                  # O1: ascending block id, hits and acquisitions alike
                    if j in st.lru:
                        st.touch(j)
                        self.touch_log.append((cid, l, r.b, h, j, "hit"))
                        continue
                    if j not in st.po and j not in st.vres:
                        raise KStepError("R2.1: the draft row of uid %s read block %d (stream %s), which is no longer "
                                         "resident at the recency update" % (r.uid, j, (l, h, r.b)))
                    if self.mut.skip_acquisition:
                        continue                                       # NEGATIVE CONTROL (R2.6)
                    if self.rev6 and j in st.vres and st.slots[st.vres[j]].cls == "v":
                        self._take_credit(r, l, h, "vres")             # rev6 SC3: V -> A / D conversion, credit first
                        st.slots[st.vres[j]].cls = "ad"
                    res = st.acquire(j, self.mut, evict=False)
                    self.touch_log.append((cid, l, r.b, h, j, "acquire_" + res["src"]))
                    self.stats["draft_acquisitions_" + res["src"]] += 1  # draft_acquisitions_po / _vres
                for ev in st.evict_to_c(self.mut):                     # O1: oldest eviction after the whole batch
                    self.stats["evict_protected_only" if ev["protected"] else "evict_released"] += 1
                    self.evict_log.append((cid, l, r.b, h, ev["block"], ev["protected"], tuple(ev.get("kinds", ())), "batch"))

    def _protect(self, st, l: int, j: int, key, kind: Optional[str] = None) -> None:
        """K2 with the R1.5 guard: no reference keyed `key` is added at layer l after (key, l) was queued for release.
        `kind` ("draft" / "verify") is the reference's provenance (L9)."""
        lay = self.rel_layers.get(key)
        if lay is not None and l in lay and not self.mut.no_late_protection_guard:
            raise KStepError("R1.5: late protection of block %d at layer %d for cohort %s after its layer-%d references "
                             "were queued for release" % (j, l, key, l))
        if self.frontier is not None:                                  # HOOK-C: the durable finished-cohort record
            self.frontier.refuse_late_protect(self, key, l, j)         # also after rel_layers was pruned (m1, I3)
        st.protect(j, key, kind)

    def _row_bound(self, r: Request, rows, l: int, who: str, hist: Optional[int] = None) -> None:
        """Rev6 F-d: every (row, head) selects at most S_max historical blocks (the bound behind N_max and the credits)."""
        hb = r.hist_blocks if hist is None else hist
        for row in rows:
            for h in range(self.H):
                n = len({j for j in row[h] if j < hb})
                if n > self.rb:
                    raise KStepError("rev6 F-d: %s of uid %s selects %d historical blocks at layer %d head %d > S_max %d"
                                     % (who, r.uid, n, l, h, self.rb))

    def _add_demand(self, st, j: int, kind: str) -> None:
        """R2.3: a request for a block in flight for the stream's owner adds its kind to that transfer's demand set (no
        second transfer)."""
        if self.mut.drop_demand_merge:
            return                                                     # NEGATIVE CONTROL (R2.6)
        seq = st.slots[st.pending[j]].seq
        tr = self.inflight.get(seq)
        if tr is None:
            raise KStepError("R2.3: block %d of stream %s is pending but transfer %s is not in flight"
                             % (j, (st.l, st.h, st.b), seq))
        if kind not in tr.demand:
            tr.demand.add(kind)
            self.stats["demand_merged_" + kind] += 1                   # demand_merged_draft / demand_merged_verify
        if self.rev6 and kind == DRAFT and st.slots[st.pending[j]].cls == "v":
            self._take_credit(self.reqs[st.owner], st.l, st.h, "join")  # rev6 SC3: V -> A / D conversion of the join
            st.slots[st.pending[j]].cls = "ad"

    def _queue_release(self, key, l: int) -> None:
        """R1.1 / R1.4: queue the release of every reference keyed `key` at layer l, at most once per (key, l)."""
        if self.frontier is not None and key not in self.rel_layers and self.frontier.late_queue(self, key, l):
            return                                                     # HOOK-C: a finished key after pruning (m1)
        lay = self.rel_layers.setdefault(key, {})
        if l in lay:
            return
        lay[l] = False
        self.pending_release.append((key, l))

    def _row_kv(self, r: Request, l: int, h: int, pos: int, sel: List[int], readable: Optional[List[int]], recent,
                read_slots: List, key, hist: Optional[int] = None, addr: Optional[List] = None, rsrc=None,
                rout: Optional[List] = None) -> List[Tuple]:
        """The (k, v) list one row attends at one head, in increasing position order: the selected blocks it may read,
        then its causal recent region. Slot reads are eager (the joint call computes from them) AND enqueued as deferred
        copies on the "compute" stream (validated later). M2 arena mode: the eager read goes through the binding (the
        rows of position phys), (block, phys, pgen) is appended to `addr`, and the executor enqueues the deferred reads
        itself (contract 3.1). M2 B2 ring mode: for every non-slot position, rsrc(qp) names the map that holds its KV and
        (pos, entry, rgen, kind, eager kv) is appended to `rout` in position order (contract 11.3, Row.ring)."""
        bs = self.bs
        st = self.pools[l].stream(r.b, h)
        out = []
        tail0 = (pos // bs) * bs
        for j in sel:
            if j < (r.hist_blocks if hist is None else hist):        # rev6 H1: a verifier row passes its own boundary
                if readable is not None and j not in readable:
                    continue                                           # draft: masked miss
                idx = st.resident(j)
                if idx is None and self.mut.read_not_ready:
                    idx = st.pending.get(j)
                if idx is None:
                    raise KStepError("row at %d reads block %d that is not resident (stream %s)" % (pos, j, (l, h, r.b)))
                if self.arena_mode:
                    s = st.slots[idx]
                    if s.phys is None:
                        raise OwnershipViolation("arena binding: row at %d reads slot %d of stream (l %d b %d h %d) (block "
                                                 "%d), which names no position" % (pos, idx, l, r.b, h, j))
                    p0 = s.phys * bs
                    ek, evv = self.arena_k[l][p0:p0 + bs].clone(), self.arena_v[l][p0:p0 + bs].clone()
                    if self.abi4:                                      # C1.1: the bias rows of the same position
                        ec = self.arena_c[l][p0:p0 + bs].clone()
                    addr.append((j, s.phys, s.pgen))
                else:
                    vk = self.slots_k[l][r.b, idx * bs:(idx + 1) * bs, h]
                    vv = self.slots_v[l][r.b, idx * bs:(idx + 1) * bs, h]
                    ek, evv = vk.clone(), vv.clone()
                    wk, wv = torch.empty_like(vk), torch.empty_like(vv)
                    with self.be.stream("compute"):
                        self.be.memcpy(wk, vk, "compute")
                        self.be.memcpy(wv, vv, "compute")
                    info = dict(layer=l, b=r.b, h=h, slot=idx, block=j, uid=r.uid, call=self.call_id)
                    self.validators.append((None, wk, ek, info))
                    self.validators.append((None, wv, evv, info))
                    self._n_new_val += 2
                read_slots.append((l, r.b, h, idx))
                self._read_keys.add(key)
                if self.store is not None:                             # HOOK-B2: the eager read of the store's bytes in
                    out.extend((j * bs + rr, kvv) for rr, kvv in enumerate(self.ex.values(ek, evv, ec)))   # HOOK-B2:
                    continue                                           # HOOK-B2: the executor's value domain
                for rr, (kk, vv_) in enumerate(zip(ek.tolist(), evv.tolist())):
                    out.append((j * bs + rr, (tuple(kk), tuple(vv_)) + ((int(ec[rr]),) if self.abi4 else ())))
            else:
                for qp in range(j * bs, (j + 1) * bs):
                    kvv = recent(qp)                                   # (k, v), or (k, v, c) in ABI v4
                    out.append((qp, kvv))
                    if rout is not None:
                        rout.append(self._ring_ref(l, r, h, qp, rsrc(qp), kvv))
        for qp in range(tail0, pos + 1):
            kvv = recent(qp)
            out.append((qp, kvv))
            if rout is not None:
                rout.append(self._ring_ref(l, r, h, qp, rsrc(qp), kvv))
        if rout is not None:
            rout.sort(key=lambda x: x[0])
        out.sort(key=lambda x: x[0])
        return [kvv for _, kvv in out]

    # ================================================================== fetch issue (K5, T3)
    # ---------------------------------------------------------------- N3 P2: the lost-work budget of verifier obligations
    def _refetch_attempt(self, r: Request, l: int, h: int, j: int, key) -> None:
        """N3 P2. Cohort `key` ATTEMPTS a second verify fetch of block j (layer l, batch slot r.b, head h) while its layer-l
        obligation is live. A waiting verifier pins its whole selection (K2), a pinned block keeps its slot (K3, R2.4), and
        _issue skips resident and in-flight blocks, so in a correct run this never happens for any finite transfer or
        reader delay: the obligation lost a block it had fetched (lost work). Called after the same-call, resident and
        in-flight shortcuts and BEFORE any verifier eviction, credit, slot, sequence or transport change.
        Checked mode (check_invariants and no Mutations switch active, as for the other invariant checks): the attempt is
        counted and OwnershipViolation is raised, so no second transfer exists. Unchecked mode: the attempt is counted
        (stats["verify_refetch_attempts"]) and the issue proceeds as before this check existed."""
        self.stats["verify_refetch_attempts"] += 1
        if self.cfg.check_invariants and not self.mut.active():
            raise OwnershipViolation("N3 lost work: cohort %s attempted a second verify fetch of block %d (layer %d, batch "
                                     "slot %d, head %d) while its layer-%d obligation is live; prevented, no transfer was "
                                     "issued" % (key, j, l, r.b, h, l))

    def _vfetch_count(self, delta: int) -> None:
        self.vfetch_ids += delta
        self.vfetch_max_ids = max(self.vfetch_max_ids, self.vfetch_ids)

    def _vfetch_drop(self, rk, why: str) -> None:
        """N3 P2 metadata cleanup of one (cohort key, layer) record (layer execution, decision or cancellation, request
        end). It touches no reference, reader event, slot or release."""
        ids = self.vfetch.pop(rk, None)
        if ids:
            self._vfetch_count(-len(ids))
            self.stats["vfetch_records_dropped_" + why] += 1

    def _vfetch_drop_cohort(self, key, why: str) -> None:
        for l in range(self.L):
            self._vfetch_drop((key, l), why)

    def progress_report(self) -> Dict[str, Any]:
        """N3 P1 (observation only; NoProgress still uses `progress`): useful = cohort-layer executions + completed draft
        steps + cohort decisions; traffic = arrivals (dropped ones too) + layer-pool polls that freed any slot; progress =
        useful + traffic. N3 P2: refetch attempts and the live verify-fetch records (one record per (cohort, layer), one
        id per (batch slot, head, block)) with their peak and the bound B * (ceil(LAMBDA / K) + 1) * H * K * S_max."""
        s = self.stats
        return dict(progress=s.get("progress", 0), useful=s.get("useful", 0), traffic=s.get("traffic", 0),
                    verify_refetch_attempts=s.get("verify_refetch_attempts", 0), vfetch_records=len(self.vfetch),
                    vfetch_ids=self.vfetch_ids, vfetch_max_ids=self.vfetch_max_ids,
                    vfetch_bound=self.cfg.B * (-(-self.LAMBDA // self.K) + 1) * self.H * self.K * self.rb,
                    vfetch_dropped={w: s.get("vfetch_records_dropped_" + w, 0) for w in ("exec", "decided", "end")})

    def _issue(self, l: int, requests: List[Tuple], cid: int) -> None:
        pool = self.pools[l]
        seen = set()
        issued: Dict[Tuple[int, int, int], Transfer] = {}
        batch = []
        order = [x for x in requests if x[0] == VERIFY] + [x for x in requests if x[0] == DRAFT]
        for i_req, (kind, r, h, j, key) in enumerate(order):
            if (r.b, h, j) in seen:
                tr0 = issued.get((r.b, h, j))
                if tr0 is not None and kind not in tr0.demand and not self.mut.drop_demand_merge:
                    tr0.demand.add(kind)                                  # R2.3: joins the transfer issued in this call
                    self.stats["demand_merged_" + kind] += 1
                    st0 = pool.stream(r.b, h)
                    if self.rev6 and kind == DRAFT and st0.slots[tr0.slot].cls == "v":
                        self._take_credit(r, l, h, "join")                # rev6 SC3: V -> A / D conversion of the join
                        st0.slots[tr0.slot].cls = "ad"
                continue
            seen.add((r.b, h, j))
            st = pool.stream(r.b, h)
            if st.resident(j) is not None:
                continue
            if j in st.pending:
                self._add_demand(st, j, kind)                             # R2.3
                continue
            if kind == VERIFY and (r.b, h, j) in self.vfetch.get((key, l), ()):
                self._refetch_attempt(r, l, h, j, key)                    # N3 P2: before any eviction, credit or slot change
            if self.rev6:
                if kind == DRAFT:
                    self._take_credit(r, l, h, "fetch")                   # rev6 SC3 (a VERIFY fetch holds a grant, G3)
            else:
                if kind == VERIFY and (self.mut.verifier_evicts_without_pressure or not pool.can_take(st, kind)):
                    self._verifier_evict(pool, st, cid)                   # R2.5: the counted progress fallback
                if not pool.can_take(st, kind):
                    if kind == DRAFT:
                        self.stats["fetch_draft_refused"] += 1            # unrequested; the draft continues masked
                    else:
                        self.stats["fetch_verify_noslot"] += 1
                        self._vfail.add(l)
                    continue
            if (l, j) not in r.hist_ev:
                raise KStepError("A5: fetch of block %d of uid %s at layer %d, which is not historical" % (j, r.uid, l))
            idx = pool.first_free(st)
            if idx is None:
                raise KStepError("K5: stream %s has usage %d of %d positions but no FREE slot" % ((l, h, r.b), st.usage(), self.n_pos))
            s = st.slots[idx]
            if self.mut.no_reader_wait:
                prev = None
            elif self.mut.release_first_reader:
                prev = s.first_reader
            else:
                prev = s.last_reader
            self.seq += 1
            gen = st.take(idx, r.uid, j, self.seq, cls="v" if kind == VERIFY else "ad")
            if self.arena_mode:                                           # M2 2.2: the bind inside take captured the
                prev = self.arenas[l].forward(s.phys)                     # previous PHYSICAL occupant's last reader
                if self.mut.no_reader_wait:                               # (negative control: no transfer gate)
                    prev = None
            tr = Transfer(self.seq, r.uid, l, r.b, h, j, idx, gen, kind, key, hist_ev=r.hist_ev.get((l, j)),
                          reader_ev=prev, issued_call=cid, phys=s.phys, pgen=s.pgen)
            self.inflight[tr.seq] = tr
            issued[(r.b, h, j)] = tr
            batch.append(tr)
            self.stats["fetch_%s_issued" % kind] += 1
            if self.arena_mode and self._checked():
                self._arena_reserve_check(l, order[i_req + 1:], seen)   # M2 2.3: bound + E + G <= P_l after the bind
            if kind == VERIFY:                                            # N3 P2: only a successful new verify issue
                ids = self.vfetch.setdefault((key, l), set())
                if (r.b, h, j) not in ids:                                # (unchecked mode: a repeat adds no id)
                    ids.add((r.b, h, j))
                    self._vfetch_count(1)
        if batch:
            if self.recent != "eager" and self._checked():            # M2 B2 11.3 (3): the pack lifetime invariant,
                for tr in batch:                                       # checked before the pack reads the host rows
                    want = (tr.uid, self.reqs[tr.uid].hist_pub.get((l, tr.block)))
                    got = self.hown.get((l, tr.b, tr.block))
                    if got != want:
                        raise OwnershipViolation("host rows: pack of transfer seq %d (uid %s, l %d, b %d, block %d) finds "
                                                 "rows owned by %s" % (tr.seq, tr.uid, l, tr.b, tr.block, got))
                    self.rstat["hown_checks"] += 1
            self.tp.issue(l, batch)

    def _verifier_eviction_candidate(self, st) -> Optional[int]:
        for blk in st.lru:                                          # least recent first
            if not st.protected(blk):
                return blk
        return None

    def _verifier_evict(self, pool, st, cid: int) -> None:
        """VERIFIER EVICTION (an M1 addition for D4, reported as a spec deviation). K5 refused a verifier fetch: the
        stream is at C and no extra slot is free. Entries leave the LRU only at an arrival (K1), and an arrival needs a
        destination, so a full LRU with every extra slot held elsewhere could never change again. The least recent
        UNPROTECTED LRU entry leaves the LRU now; its slot is released through the K4 reader gate as usual.
        Repair 1 R2.5: this is a verifier access that changes the draft LRU, so it stays ONLY as an explicit, counted
        progress fallback, recorded per stream with the evicted block (vev_log, the request's trace, report()).
        Capacity mode rev6 never evicts a draft entry for a verifier (G6): the call is refused there."""
        if self.rev6:
            raise KStepError("rev6 G6: a verifier eviction of a draft LRU entry was requested")
        blk = self._verifier_eviction_candidate(st)
        if blk is None:
            return
        idx = st.lru.pop(blk)
        st.release(idx, "verifier_eviction")
        self.stats["verifier_evictions"] += 1
        if not self.mut.silent_verifier_eviction:                       # (the NEGATIVE CONTROL records nothing)
            self.vev_log.append((cid, st.l, st.b, st.h, blk, st.owner))
            self.trace[st.owner].setdefault("verifier_evictions_of_draft_entries", []).append(
                dict(call=cid, layer=st.l, h=st.h, block=blk))
        n = pool.poll_releases(self.mut)
        if n:
            self.stats["slots_freed"] += n

    # ================================================================== end of step: acceptance, publication, drafts
    def _uniform_of(self, r: Request, c: Cohort):
        def f(i, purpose):
            return rng.uniform(self.cfg.seed, r.uid, c.E + i, c.epoch, purpose)
        return f

    def _end_of_step(self, rows: List[Tuple[Request, Draft]]) -> None:
        outs = []
        for r in self.active():
            head = next((c for c in r.cohorts if c.state not in (ACCEPTED, CANCELLED)), None)
            if head is None or head.state != DONE:
                continue
            d = {i: r.drafts[i - 1].d for i in range(head.a, head.b + 1)}
            q = {i: r.drafts[i - 1].q for i in range(head.a, head.b + 1)}
            o = acc.accept_cohort(self.cfg.mode, head.a, head.b, d, q, head.p, self._uniform_of(r, head), self.mut)
            outs.append((r, head, o))
        if self.mut.batch_min_commit and len(outs) > 1:                 # NEGATIVE CONTROL: the old lockstep rule
            cmin = 1 + min(o.n_acc for _, _, o in outs)
            for r, c, o in outs:
                if o.committed > cmin:
                    i = c.a + cmin - 1
                    o.n_acc, o.i_star, o.correction = cmin - 1, i, acc.argmax(c.p[i])
        for r, c, o in outs:
            self._apply(r, c, o)
        for r, dr in rows:
            if r.cur is not dr:
                continue
            r.cur = None
            if dr.epoch != r.epoch or r.done:
                continue
            self._finish_draft(r, dr)
        for r in self.active():
            if r.generated() >= r.N and not any(c.state in (PENDING, RUNNING, DONE) for c in r.cohorts):
                self._end_request(r)

    def _record_draft(self, r: Request, dr: Draft, status: str) -> None:
        ep = self.trace[r.uid]["epochs"][-1]
        ep["drafts"].append(dict(s=dr.s, pos=dr.pos, tok_in=dr.tok_in, n_at=dr.n_at, step=dr.step,
                                 sel={l: [list(x) for x in v] for l, v in dr.sel.items()},
                                 avail={l: [list(x) for x in v] for l, v in dr.avail.items()},
                                 d=dr.d, q=None if dr.q is None else list(dr.q), status=status))
        if self.abi4:                                                  # C1.1 instrumentation (12.3.1): the windows
            ep["drafts"][-1]["cks"] = {l: [list(x) for x in v] for l, v in dr.cks.items()}   # each select received

    def _finish_draft(self, r: Request, dr: Draft) -> None:
        if self.frontier is not None and self.frontier.draft_entry(self, r, dr):
            return                                                     # HOOK-C2: an old-epoch completion: no mutation
        dist = self.ex.dist(dr.hidden)
        dr.hidden = None                                               # K6: only d and q are kept
        u = rng.uniform(self.cfg.seed, r.uid, dr.pos + 1, r.epoch, "draft")
        dr.d, dr.q = acc.draft_token(self.cfg.mode, dist, u)
        dr.state = DONE
        r.drafts.append(dr)
        self.stats["draft_steps"] += 1
        self.stats["progress"] += 1
        self.stats["useful"] += 1                                      # N3 P1: a draft step completed
        self._record_draft(r, dr, "drafted")
        m = cad.triggered_by(dr.s, self.K)
        if m is not None:
            a, b = cad.cohort_range(m, self.K)
            self._trigger(r, m, a, b)
        if dr.pos + 1 >= r.last_pos():
            r.drafting_done = True
            fr = cad.flush_range(r.n_trig, len(r.drafts), self.K)
            if fr is not None:
                self._trigger(r, r.n_trig + 1, fr[0], fr[1])

    def _trigger(self, r: Request, m: int, a: int, b: int) -> None:
        c = Cohort(r, m, a, b, self.t)
        r.cohorts.append(c)
        r.n_trig = m
        self.stats["cohorts_triggered"] += 1

    def _cohort_record(self, r: Request, c: Cohort, o=None, was: Optional[str] = None) -> Dict:
        c.decided_step = self.t
        return dict(m=c.m, a=c.a, b=c.b, E=c.E, positions=list(c.positions), trigger_step=c.trigger_step,
                    join_step=c.join_step, decided_step=self.t, state=c.state, exec_call=dict(c.exec_call), vsel=dict(c.vsel),
                    p={i: list(v) for i, v in c.p.items()}, waits=c.waits, row_layers=c.row_layers,
                    layers_done=c.next_layer, state_at_decision=was or c.state,
                    outcome=None if o is None else dict(n_acc=o.n_acc, i_star=o.i_star, correction=o.correction))

    @staticmethod
    def _free_cohort(c: Cohort) -> None:
        """K6: a decided cohort keeps no workspace, hidden state, provisional KV, chains or distributions."""
        c.ws.clear()
        c.hidden = []
        c.prov = {}
        c.chain = {}
        c.vsel = {}
        c.vhist = {}
        c.p = {}

    def _rebuild_dchain(self, r: Request) -> None:
        """D6 / A4 (fix round F5): after a full-acceptance publication the draft chain is the committed exact chain
        r.comp extended by the draft keys of the still uncommitted positions r.n, r.n + 1, ... in order."""
        for l in range(self.L):
            for h in range(self.H):
                base = r.comp[l][h]
                if base.count != r.n:
                    raise KStepError("A5/D6: the committed chain of uid %s (l %d h %d) holds %d keys, n = %d"
                                     % (r.uid, l, h, base.count, r.n))
                dk = r.draft_kv[l][h]
                pos = sorted(dk)
                if pos != list(range(r.n, r.n + len(pos))):
                    raise KStepError("D6: draft KV of uid %s (l %d h %d) at %s is not the uncommitted run from n = %d"
                                     % (r.uid, l, h, pos, r.n))
                ch = base.copy()
                for qp in pos:
                    ch.append(self._ck(dk[qp]))
                r.dchain[l][h] = ch

    def _finished_key(self, key) -> None:
        """R1.4 decision cleanup (acceptance, cancellation, untriggered key): queue only the layers of `key` that are not
        queued yet; a layer queued after its execution (R1.1) is not queued again."""
        if self.frontier is not None and self.frontier.begin_finish(self, key):
            return                                                     # HOOK-C: a duplicate decision is a no-op (m1)
        if self.mut.cleanup_skips_unqueued_layers:
            return                                                     # NEGATIVE CONTROL (R1.6 rollback)
        for l in range(self.L):
            if self.mut.cleanup_double_queues:                         # NEGATIVE CONTROL (R1.6 rollback)
                self.rel_layers.setdefault(key, {}).setdefault(l, False)
                self.pending_release.append((key, l))
            else:
                self._queue_release(key, l)
        if self.frontier is not None:                                  # HOOK-C: the durable record, after every layer
            self.frontier.record_finish(self, key)                     # of the key was queued (m1, I3)

    def _apply(self, r: Request, c: Cohort, o) -> None:
        if self.frontier is not None and self.frontier.decision_entry(self, r, c):
            return                                                     # HOOK-C2: a duplicate decision: no mutation (I6)
        ep = self.trace[r.uid]["epochs"][-1]
        self.stats["accepted_drafts"] += o.n_acc
        self.stats["zero_mass_accepts"] += o.zero_mass_accepts
        self.stats["progress"] += 1
        self.stats["useful"] += 1                                      # N3 P1: a cohort decision
        if o.full:
            self._publish(r, c, c.b, c.b)
            r.toks.extend(r.drafts[i - 1].d for i in range(c.a, c.b + 1))
            c.state = ACCEPTED
            self.stats["cohorts_accepted_full"] += 1
            ep["cohorts"].append(self._cohort_record(r, c, o))
            self._finished_key(c.key)
            self._drop_superseded(r)
            for i in range(c.a, c.b + 1):                              # K6: a committed draft keeps d only
                dd = r.drafts[i - 1]
                dd.q, dd.sel, dd.avail = None, {}, {}
            if not self.mut.no_dchain_rebuild:
                self._rebuild_dchain(r)
            r.cohorts.remove(c)                                        # an accepted cohort is no longer held (K6)
            self._free_cohort(c)
            self._vfetch_drop_cohort(c.key, "decided")                 # N3 P2 metadata only
            if self.recent != "eager":
                self._rsync(r, "superseded")                           # M2 B2: draft KV of committed positions drains
            return
        i_star = o.i_star
        upto_pub = c.b if self.mut.publish_beyond_prefix else i_star
        self._publish(r, c, upto_pub, i_star)
        r.toks.extend(r.drafts[i - 1].d for i in range(c.a, i_star))
        r.toks.append(o.correction)
        if len(r.toks) - r.P0 > r.N:
            raise KStepError("C4: committed beyond max_new_tokens (uid %s)" % (r.uid,))
        c.state = ACCEPTED
        self.stats["cohorts_accepted_partial"] += 1
        self.stats["corrections"] += 1
        ep["cohorts"].append(self._cohort_record(r, c, o))
        self._finished_key(c.key)
        self._free_cohort(c)
        self._vfetch_drop_cohort(c.key, "decided")                     # N3 P2 metadata only
        # C3: cancel every later cohort (triggered or not) and every draft with index >= i_star
        for cc in r.cohorts:
            if cc.state in (PENDING, RUNNING, DONE) and cc is not c:
                was = cc.state
                cc.state = CANCELLED
                self.stats["cohorts_cancelled"] += 1
                self.stats["verifier_row_layers_cancelled"] += cc.row_layers
                self.stats["cohort_rows_discarded"] += len(cc.positions)
                self.stats["workspace_rows_discarded"] += len(cc.ws) * len(cc.positions)
                cc.ws.clear()
                ep["cohorts"].append(self._cohort_record(r, cc, was=was))
                if not self.mut.leak_cancelled_protection:
                    self._finished_key(cc.key)
                self._cancel_requester(cc.key)
                self._free_cohort(cc)
                self._vfetch_drop_cohort(cc.key, "decided")            # N3 P2 metadata only (cancellation)
        n_open = cad.cohort_of(len(r.drafts) + 1, self.K)
        for m in range(c.m + 1, n_open + 1):
            k = (r.uid, r.epoch, m)
            if not any(cc.key == k for cc in r.cohorts):
                self._finished_key(k)
                self._cancel_requester(k)
                self.stats["untriggered_keys_cancelled"] += 1
        discarded = [dd for dd in r.drafts if dd.s >= i_star]
        self.stats["draft_steps_discarded"] += len(discarded)
        for dd in discarded:
            self._record_draft(r, dd, "cancelled")
        if r.cur is not None and r.cur.epoch == r.epoch:
            self.stats["draft_steps_discarded"] += 1
            self.stats["draft_rows_cancelled_in_flight"] += 1
            self._record_draft(r, r.cur, "cancelled_in_flight")
        # A6: rollback of the draft list, draft KV, draft compressed chain and positions to the committed state
        hi = r.E + len(r.drafts) + (1 if (r.cur is not None and r.cur.epoch == r.epoch) else 0)
        lo = r.n
        if self.abi4:                                                  # C1.1 V4: the v4 window event rule
            kk, ss = self.chain_kernel, self.stride
            crossed = any(qp + 1 >= kk and (qp + 1 - kk) % ss == 0 for qp in range(lo, hi))
        else:
            crossed = any((qp + 1) % self.stride == 0 for qp in range(lo, hi))
        if crossed:
            self.stats["rollback_across_compression"] += 1
        if any((qp + 1) % self.bs == 0 for qp in range(lo, hi)):
            self.stats["rollback_across_fill"] += 1
        self.stats["rollbacks"] += 1
        ep["outcome"] = dict(kind="rejection", m=c.m, i_star=i_star, correction=o.correction, n=r.n)
        r.epoch += 1
        r.E = r.n
        r.drafts = []
        r.cohorts = []
        r.n_trig = 0
        r.cur = None
        for l in range(self.L):
            for h in range(self.H):
                r.draft_kv[l][h].clear()
        if not self.mut.no_compressed_rollback:
            r.dchain = [[x.copy() for x in row] for row in r.comp]
        r.drafting_done = r.generated() >= r.N
        self.trace[r.uid]["epochs"].append(dict(epoch=r.epoch, E=r.E, u0=r.toks[r.E], drafts=[], cohorts=[], outcome=None))
        if self.recent != "eager":
            self._rsync(r, "rollback")                                 # M2 B2: the old epoch's DRAFT, cancelled PRE / PROV

    def _cancel_requester(self, key) -> None:
        """T3: a cancelled requester never frees an in-flight destination (its arrival is kept or dropped later)."""
        self.cancelled_keys.add(key)
        if not self.mut.free_on_cancel:
            return
        for tr in list(self.inflight.values()):                     # NEGATIVE CONTROL: free before landing
            if tr.requester == key:
                st = self.pools[tr.layer].stream(tr.b, tr.h)
                s = st.slots[tr.slot]
                if s.state == "inflight" and s.gen == tr.gen:
                    s.state, s.block, s.owner = "free", -1, None
                    st.pending.pop(tr.block, None)

    def _publish(self, r: Request, c: Cohort, upto_pub: int, upto_commit: int) -> None:
        """A4: publish the verifier's EXACT KV of inputs u_{a-1} .. u_{upto-1} (positions E+a-1 .. E+upto-1)."""
        for i in range(c.a, upto_pub + 1):
            pos = c.E + i - 1
            self.pub_log.append((r.uid, pos))
            d_kv = [[r.draft_kv[l][h].get(pos) for h in range(self.H)] for l in range(self.L)]
            if all(x is not None for row in d_kv for x in row) and any(
                    d_kv[l][h] != c.prov[l][h][pos] for l in range(self.L) for h in range(self.H)):
                self.stats["published_draft_kv_differs"] += 1           # A4: committed input, draft KV != exact KV
            for l in range(self.L):
                for h in range(self.H):
                    if self.mut.publish_draft_kv:
                        kv = r.draft_kv[l][h][pos]                          # NEGATIVE CONTROL
                    else:
                        kv = c.prov[l][h][pos]
                    r.pub[l][h][pos] = kv
                    r.comp[l][h].append(self._ck(kv))
                    self.published[(r.uid, pos, l, h)] = kv
                    if self.recent != "eager":                         # M2 B2: PROV -> EXACT, the same entry (11.2)
                        self._rmove(l, r, h, R_PROV, R_EXACT, (c.key, pos), pos, (r.epoch, None, pos))
        r.n = c.E + upto_commit
        self._history(r)

    def _drop_superseded(self, r: Request) -> None:
        """D6: draft KV of a published position is superseded (drafts read the exact KV from then on)."""
        for l in range(self.L):
            for h in range(self.H):
                for qp in [x for x in r.draft_kv[l][h] if x < r.n]:
                    del r.draft_kv[l][h][qp]

    def _history(self, r: Request, adm: Optional[Dict] = None) -> None:
        """A5: every committed block that became complete is written to the host (D2H on the "d2h" stream, one event per
        layer and block) and becomes fetchable; its positions leave the recent region. M2 B2 (recent "ring" /
        "eager_matched"): _d2h_block writes the block (admission blocks from `adm`, the request's A_r); afterwards a ring
        EXACT entry of the block drains unless a WAITING workspace still reads it (_ring_ws_held)."""
        bs = self.bs
        while (r.hist_blocks + 1) * bs <= r.n:
            j = r.hist_blocks
            for l in range(self.L):
                if self.recent == "eager":
                    kb = torch.tensor([[r.pub[l][h][pos][0] for h in range(self.H)] for pos in range(j * bs, (j + 1) * bs)], dtype=torch.int64)
                    vb = torch.tensor([[r.pub[l][h][pos][1] for h in range(self.H)] for pos in range(j * bs, (j + 1) * bs)], dtype=torch.int64)
                    with self.be.stream("d2h"):
                        self.be.memcpy(self.host_k[l][r.b, j * bs:(j + 1) * bs], kb, "d2h")
                        self.be.memcpy(self.host_v[l][r.b, j * bs:(j + 1) * bs], vb, "d2h")
                        r.hist_ev[(l, j)] = self.be.event_rec("d2h")
                else:
                    self._d2h_block(r, l, j, adm)                      # M2 B2 11.3: ring / matched D2H, snapshot, hown
                if self.recent != "eager" and adm is None:
                    for h in range(self.H):
                        if not self._ring_ws_held(r, l, h, j):
                            for pos in range(j * bs, (j + 1) * bs):
                                self._rdrop(l, r, h, R_EXACT, pos, "history")
                for h in range(self.H):
                    hold = set()
                    if self.rev6 and not self.mut.rev6_release_recent_at_history:
                        hold = self._h1_holders(r, l, h, j)            # H1: a dependent verifier keeps the recent source
                    if hold:
                        for pos in range(j * bs, (j + 1) * bs):
                            r.pub_hold[(l, h, pos)] = set(hold)
                        self.stats["rev6_h1_retained_positions"] += bs
                        self.h1_log.append(("retain", self.t, r.uid, l, h, j, tuple(
                            (k, any(c.key == k and l in c.granted for c in r.cohorts), self._h1_waits_in_flight(r, k, l))
                            for k in sorted(hold))))
                    else:
                        for pos in range(j * bs, (j + 1) * bs):
                            del r.pub[l][h][pos]
            r.hist_blocks += 1
            self.stats["history_blocks"] += 1

    def _h1_waits_in_flight(self, r: Request, key, l: int) -> bool:
        """H1 evidence: does cohort `key` wait at layer l for a historical block of its workspace that is in flight?"""
        for c in r.cohorts:
            w = c.ws.get(l)
            if c.key != key or w is None:
                continue
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                if any(j < w["hist"] and j in st.pending for row in w["sel"] for j in row[h]):
                    return True
        return False

    # ================================================================== memory, finish
    @staticmethod
    def _chain_entries(ch: CompressedChain) -> int:
        return len(ch.cks) + len(ch.buf)

    def _sample_memory(self) -> None:
        """K6: count EVERY held item (no filter that could hide a leak); MemoryAccountant turns counts into bytes."""
        n = defaultdict(int)
        for pool in self.pools:
            for k, v in (pool.position_counts() if self.arena_mode else pool.counts()).items():
                n[k] += v                                              # physical slots (M2: positions) by state, FREE included
            for st in pool.streams.values():
                n["prot_refs"] += sum(len(v) for v in st.prot.values())
        if self.mut.mem_no_releasing:
            n["releasing"] = 0                                         # NEGATIVE CONTROL (fix round F4)
        if self.mut.vres_counted_as_lru:
            n["lru"] += n["vres"]                                      # NEGATIVE CONTROL (repair 1 R2.6)
            n["vres"] = 0
        n["transfers"] = len(self.inflight)
        n["release_entries"] = (len(self.pending_release) + len(self.key_readers)
                                + sum(len(v) for v in self.rel_layers.values()) + len(self.h1_reads))
        n["sel_ids"] += sum(len(x) for _, rows in self.h1_reads.values() for row in rows for x in row)   # H1 records
        n["release_entries"] += len(self.vfetch)                         # N3 P2: one record per (cohort, layer)
        n["sel_ids"] += self.vfetch_ids                                  # N3 P2: one id per (batch slot, head, block)
        for r in self.active():
            drafts = list(r.drafts) + ([r.cur] if r.cur is not None else [])
            n["drafts"] += len(drafts)
            n["release_entries"] += sum(len(v) for v in r.pub_hold.values())   # H1 holder references
            for dr in drafts:
                n["q_vectors"] += dr.q is not None
                n["hidden_rows"] += dr.hidden is not None
                n["sel_ids"] += sum(len(x) for v in dr.sel.values() for x in v) + sum(len(x) for v in dr.avail.values() for x in v)
            for l in range(self.L):
                for h in range(self.H):
                    n["draft_kv_pos"] += len(r.draft_kv[l][h])
                    n["recent_kv_pos"] += len(r.pub[l][h])
                    n["ck_entries"] += self._chain_entries(r.comp[l][h]) + self._chain_entries(r.dchain[l][h])
            for c in r.cohorts:
                n["cohorts"] += 1
                n["workspace_rows"] += len(c.ws) * len(c.positions)
                n["hidden_rows"] += len(c.hidden)
                n["p_vectors"] += len(c.p)
                n["prov_kv_pos"] += sum(len(d) for lst in c.prov.values() for d in lst)
                n["sel_ids"] += sum(len(x) for rows in c.vsel.values() for row in rows for x in row)
                n["ck_entries"] += sum(self._chain_entries(ch) for lst in c.chain.values() for ch in lst)
                n["ck_entries"] += sum(self._chain_entries(ch) for w in c.ws.values() for ch in w["chain"])
        counts = {k: n[k] for k in COUNTS}
        if self.arena_mode:
            counts.update(self._arena_counts())                        # M2 2.6: arena metadata at allocated capacity
        if self.ring_mode:
            counts.update(self._ring_counts())                         # M2 B2 11.4: the ring and every held A_r
        if self.abi4:                                                  # C1.1 A9: the operational bias backings held
            self.bias_ledger.retire()                                  # now (passive release first; a SAMPLED value)
            counts["bias_staging_bytes"] = self.bias_ledger.held
        if self.consume:                                               # C1.2: the gather backings held now (0: each
            counts["consume_staging_bytes"] = self.consume_ledger.held  # call releases its own; a SAMPLED value)
        if self.store is not None:                                     # HOOK-B2: the store owns the A_r bytes, so no
            counts["admission_pos"] = 0                                # HOOK-B2: second charge as admission_kv; its
            self.store.poll()                                         # HOOK-B2: held sources (passive release first;
            counts["source_staging_bytes"] = self.store.holds.held   # HOOK-B2: a SAMPLED value) as source_staging
            if self.acct.gen:                                          # HOOK-L6 (I11): the consumer mirrors' bytes
                counts["gen_mirror_bytes"] = int(self.store.store_bytes()["gen_mirror"])
        self.acct.sample(counts)

    def finish(self, order: Optional[Tuple[str, ...]] = None) -> None:
        """Drain every stream (deferred reads, scatters, history writes) and validate every deferred read. M2 B2: then
        every remaining D2H snapshot; every publication's snapshot must have been validated exactly once."""
        self._session_guard("finish")                                  # HOOK-L7 (I12): no backend use unless "open"
        self.be.synchronize(order or self.cfg.drain_order)
        self._validate(all_=True)
        if self.store is not None:                                     # HOOK-B2: every stream ran: release the store's
            self.store.poll()                                          # HOOK-B2: holds BEFORE the frontier clears (I5)
        if self.frontier is not None:                                  # HOOK-C: every stream ran: prune and clear what
            self.frontier.poll(self)                                   # drained (non-blocking)
        if self.abi4:                                                  # C1.1 A9: after the drain, every event-held
            self.bias_ledger.retire()                                  # backing is released (adversarial payloads of
                                                                       # transfers never landed stay held)
        if self.recent != "eager":
            self._d2h_poll(final=True)
            if self.rstat["d2h_snapshots"] != self.rstat["d2h_snapshots_validated"] or self.d2h_checks:
                raise OwnershipViolation("D2H readback missing: %d snapshots, %d validated, %d pending"
                                         % (self.rstat["d2h_snapshots"], self.rstat["d2h_snapshots_validated"],
                                            len(self.d2h_checks)))

    # ================================================================== HOOK-L3: quiescent-reset hooks (lead; I8)
    # INTERFACE_BC3 I8 (frozen 2026-10-16). Only frontier.QuiescentReset (track C) calls them; nothing in the no-reset
    # path does, so a run without a coordinator is unchanged. The executor's own events are covered ONLY for an executor
    # that declares the I10 contract in its own class (HOOK-L4: its audited reset_holders() list joins event_holders();
    # an undeclared executor contributes nothing here, and the coordinator refuses it). Not covered: the
    # instrumentation logs (Arena / RingPool / Stream .log, the dispatcher's *_log lists), which keep event objects as
    # history and are never read for a decision. The frontier (reset_pending) and the store (drained_all) report their
    # own state.
    # HOOK-L7 (INTERFACE_BC5 I12, lead, 2026-10-17): the session state. "open" (initial): submit, step and finish run.
    # begin_reset() -> "resetting"; end_reset() -> "open"; fail_reset(phase, text) -> "failed", LATCHED (no method
    # returns a failed session to "open"). Outside "open", submit / step / finish / host_poll refuse before any change,
    # and a "failed" session also refuses detach_completed_events, so a failed reset cannot be resumed silently, admit a
    # request or use the backend again through them. Only the coordinator calls these hooks; a run without a
    # coordinator stays "open" and is unchanged.
    def _session_guard(self, op: str) -> None:
        """HOOK-L7 (I12), READ-ONLY: OwnershipViolation "session <state>: <op> refused (<reason>)" unless "open"."""
        if self.session != "open":
            raise OwnershipViolation("session %s: %s refused (%s)" % (self.session, op, self.session_reason))

    def begin_reset(self) -> None:
        """HOOK-L7 (I12): "open" -> "resetting" (the coordinator, after its read-only check passed)."""
        if self.session != "open":
            raise OwnershipViolation("session %s: begin_reset refused (%s)" % (self.session, self.session_reason))
        self.session = "resetting"

    def end_reset(self) -> None:
        """HOOK-L7 (I12): "resetting" -> "open" (the coordinator, after every mutating phase passed)."""
        if self.session != "resetting":
            raise OwnershipViolation("session %s: end_reset refused (%s)" % (self.session, self.session_reason))
        self.session = "open"

    def fail_reset(self, phase: str, text: str) -> None:
        """HOOK-L7 (I12): "resetting" -> "failed", LATCHED. session_reason = "<phase>: <text>" (at most 512
        characters). A failed session stays failed: there is no transition back to "open"."""
        if self.session != "resetting":
            raise OwnershipViolation("session %s: fail_reset refused (%s)" % (self.session, self.session_reason))
        self.session = "failed"
        self.session_reason = ("%s: %s" % (phase, text))[:512]

    def host_poll(self) -> None:
        """HOOK-L7 (I12): exactly one passive host poll (the existing _poll), for the coordinator's explicit drain."""
        self._session_guard("host_poll")
        self._poll()

    def quiesce_state(self) -> Dict[str, int]:
        """HOOK-L3, READ-ONLY: every count that must be 0 before a quiescent reset (an OPEN dict; a caller requires every
        value 0 and names every nonzero key). The I8 counts: queued, active, inflight, validators, pending_release,
        d2h_checks, adm, key_readers. Validators, D2H checks and admission buffers must be ZERO, not only complete: their
        completion runs the byte checks (_validate, _d2h_poll), which may raise, so the caller drains them first
        (Dispatcher.finish, then one host poll, as F21 / BC2). Then detach_completed_events never runs a check that can
        raise after a mutation. Also counted, because no passive poll can drain them: transport_pending, slots_held
        (slots not FREE and not RELEASING), ring_live (entries not FREE and not DRAINING), and the key-held ledger
        records. RELEASING slots, DRAINING ring entries and event-held ledger records are not counted: their gating
        events are listed by event_holders(), and detach_completed_events() drains them passively."""
        def keyed(lg):
            return 0 if lg is None else sum(1 for rec in lg.recs.values() if rec["ev"] is None)
        out = dict(
            queued=len(self.queue),
            active=sum(1 for r in self.slot_req if r is not None),
            inflight=len(self.inflight),
            validators=len(self.validators),
            pending_release=len(self.pending_release),
            d2h_checks=len(self.d2h_checks),
            adm=len(self.adm),
            key_readers=len(self.key_readers),
            transport_pending=len(getattr(self.tp, "pending", None) or ()),
            slots_held=sum(1 for p in self.pools for st in p.streams.values() for s in st.slots
                           if s.state not in (FREE, RELEASING)),
            ring_live=sum(1 for rg in self.rings for e in range(rg.Q) if rg.state[e] not in (FREE, DRAINING)),
            bias_ledger_keyed=keyed(self.bias_ledger),
            consume_ledger_keyed=keyed(self.consume_ledger),
        )
        if self.tickets is not None:                                   # HOOK-L8 (I14): outstanding result tickets
            out["tickets"] = self.tickets.outstanding()                # (pending or broken) block the reset; the key
        if self.deferred:                                              # exists only with tickets on. I17: a
            out["pending_calls"] = int(self.pending_call is not None)  # submitted, unpublished call or a stopped
            out["pending_steps"] = int(self.pending_step is not None)  # step blocks it too (only in deferred mode)
        if self.cancel_req:                                            # I20: a requested, not yet torn-down
            out["cancels_pending"] = len(self.cancel_req)              # cancellation (the key exists only then)
        return out

    def event_holders(self) -> List[Tuple[str, Tuple, Any]]:
        """HOOK-L3, READ-ONLY: one (owner_class, location, event) per event reference that the dispatcher, its pools,
        ledgers, requests and transport hold (None references are not listed). Owner classes: slot_reader, arena_reader,
        arena_captured, ring_gate, admission, bias_ledger, consume_ledger, transfer, validator, key_reader, hist,
        d2h_check, request_map (a non-ring request map holds events directly), transport (the S0 Pipe's slot events).
        The adversarial transport's pending transfers are a subset of inflight (only _poll removes an inflight entry),
        so their reader_ev / hist_ev appear as "transfer"; quiesce_state counts them as transport_pending."""
        out: List[Tuple[str, Tuple, Any]] = []

        def put(cls, loc, ev):
            if ev is not None:
                out.append((cls, loc, ev))
        for l, p in enumerate(self.pools):
            for (b, h), st in p.streams.items():
                for s in st.slots:
                    put("slot_reader", (l, b, h, s.idx, "first"), s.first_reader)
                    put("slot_reader", (l, b, h, s.idx, "last"), s.last_reader)
        for a in self.arenas or ():
            for q in range(a.P):
                put("arena_reader", (a.l, q), a.last_reader[q])
                put("arena_captured", (a.l, q), a.captured[q])
        for rg in self.rings:
            for e in range(rg.Q):
                for name in ("ready", "cev", "dev"):
                    put("ring_gate", (rg.l, e, name), getattr(rg, name)[e])
        for i, A in enumerate(self.adm):
            for name in ("ready", "cev", "dev"):
                put("admission", (i, A["uid"], name), A[name])
        for cls, lg in (("bias_ledger", self.bias_ledger), ("consume_ledger", self.consume_ledger)):
            if lg is not None:
                for serial, rec in lg.recs.items():
                    put(cls, (rec["kind"], serial), rec["ev"])
        for seq, tr in self.inflight.items():
            for name in ("reader_ev", "hist_ev", "ev"):
                put("transfer", (seq, name), getattr(tr, name))
        for i, v in enumerate(self.validators):
            put("validator", (i,), v[0])
        for k, rd in self.key_readers.items():
            for j, ev in enumerate(rd):
                put("key_reader", (k, j), ev)
        for uid, r in self.reqs.items():
            for lj, ev in r.hist_ev.items():
                put("hist", (uid, lj), ev)
            if r.rmap is not None and not self.ring_mode:
                for l in range(self.L):
                    for h in range(self.H):
                        for kind, m in r.rmap[l][h].items():
                            for key, ev in m.items():
                                put("request_map", (uid, l, h, kind, key), ev)
        for i, rec in enumerate(self.d2h_checks):
            put("d2h_check", (i, rec["uid"], "hist_ev"), rec["hist_ev"])
            put("d2h_check", (i, rec["uid"], "snap_ev"), rec["snap_ev"])
        pipe = getattr(self.tp, "pipe", None)
        if pipe is not None:
            for s in range(pipe.ring):
                put("transport", ("slot_h2d", s), pipe.slot_h2d[s])
                put("transport", ("slot_sc", s), pipe.slot_sc[s])
        for cls, loc, ev in self._executor_holders():                 # HOOK-L4 (I10): the executor's own events
            put(cls, loc, ev)
        return out

    def _executor_holders(self) -> List[Tuple[str, Tuple, Any]]:
        """HOOK-L4 (I10), READ-ONLY: the executor's reset_holders() when its OWN class declares the contract (else [];
        the coordinator then refuses it). A declared executor whose classes do not define both contract methods, or an
        owner class that does not start with "executor:", is refused (OwnershipViolation; nothing changes)."""
        if reset_contract_of(self.ex) != RESET_CONTRACT:
            return []
        miss = reset_methods_missing(self.ex)
        if miss:
            raise OwnershipViolation("I10: the executor %s declares reset_contract %r without the class methods %s"
                                     % (type(self.ex).__name__, RESET_CONTRACT, miss))
        out = list(type(self.ex).reset_holders(self.ex))               # I12: through the class (no instance shadow)
        for cls, loc, ev in out:
            if not (isinstance(cls, str) and cls.startswith("executor:")):
                raise OwnershipViolation("I10: the executor %s lists owner class %r (it must start with 'executor:')"
                                         % (type(self.ex).__name__, cls))
        return out

    def detach_completed_events(self) -> int:
        """HOOK-L3, MUTATING (only frontier.QuiescentReset calls it, after its own full check). First the same check, with
        NO mutation on a refusal: every quiesce_state() count 0 and every event_holders() event complete (ev_done).
        HOOK-L4 (I10): an executor that declares the reset contract in its own class detaches its own references FIRST
        (its refusal changes nothing in the dispatcher); its count is part of the result. Then (1) the existing
        passive host poll (_poll) drains what those events gated: RELEASING slots, DRAINING ring
        entries, event-held ledger records (through retire), the store's and the frontier's own polls. No byte check
        runs there: validators, D2H checks and admission buffers are already 0 (quiesce_state), and nothing is in
        flight; (2) the post-condition, which only an internal inconsistency can break; (3) every
        remaining reference to a complete event is set to None (a FREE slot's or position's history readers, a FREE ring
        entry's gates, the requests' history events and map events, the S0 Pipe's slot events). Returns the number of
        references set to None. A later reader of any of these sees None (complete; no wait), never a pooled object that
        a backend reset re-records. HOOK-L7 (I12): a FAILED session refuses at once (no detach, no poll after a failed
        reset); "open" (a direct caller) and "resetting" (the coordinator) proceed as before."""
        if self.session == "failed":
            raise OwnershipViolation("session failed: detach_completed_events refused (%s)" % (self.session_reason,))
        bad = {k: v for k, v in self.quiesce_state().items() if v}
        live = [(c, loc) for c, loc, ev in self.event_holders() if not ev_done(ev)]
        if bad or live:
            raise OwnershipViolation("detach refused: not quiescent %s; %d pending events: %s"
                                     % (bad, len(live), live[:6]))
        n_ex = 0
        if reset_contract_of(self.ex) == RESET_CONTRACT:              # HOOK-L4 (I10): the executor's own detach FIRST,
            listed = len([1 for _, _, ev in self._executor_holders() if ev is not None])   # before any dispatcher
            n_ex = type(self.ex).detach_completed(self.ex)             # mutation; then it is VERIFIED: a count equal
            rest = [(c, loc) for c, loc, ev in self._executor_holders() if ev is not None]   # to what it listed and
            if isinstance(n_ex, bool) or not isinstance(n_ex, int) or n_ex != listed or rest:   # nothing left
                raise OwnershipViolation("I10: the executor %s detach_completed returned %r for %d listed references "
                                         "and left %s (the dispatcher is unchanged; the executor's own state is as "
                                         "its detach left it)" % (type(self.ex).__name__, n_ex, listed, rest[:6]))
        self._poll()
        if self.consume_ledger is not None:                            # _poll retires the bias ledger only
            self.consume_ledger.retire()
        left = dict(
            releasing=sum(1 for p in self.pools for st in p.streams.values() for s in st.slots if s.state != FREE),
            draining=sum(1 for rg in self.rings for e in range(rg.Q) if rg.state[e] != FREE),
            arena_bound=sum(a.n_bound for a in self.arenas or ()),
            adm=len(self.adm), validators=len(self.validators), d2h_checks=len(self.d2h_checks),
            ledger=sum(len(lg.recs) for lg in (self.bias_ledger, self.consume_ledger) if lg is not None))
        if any(left.values()):
            raise OwnershipViolation("detach: the passive drain left %s" % ({k: v for k, v in left.items() if v},))
        n = 0
        for p in self.pools:
            for st in p.streams.values():
                for s in st.slots:
                    n += (s.first_reader is not None) + (s.last_reader is not None)
                    s.first_reader = s.last_reader = None
        for a in self.arenas or ():
            for q in range(a.P):
                n += (a.last_reader[q] is not None) + (a.captured[q] is not None)
                a.last_reader[q] = a.captured[q] = None
        for rg in self.rings:
            for name in ("ready", "cev", "dev"):
                g = getattr(rg, name)
                for e in range(rg.Q):
                    n += g[e] is not None
                    g[e] = None
        for r in self.reqs.values():
            n += sum(1 for ev in r.hist_ev.values() if ev is not None)
            r.hist_ev.clear()
            if r.rmap is not None and not self.ring_mode:
                for l in range(self.L):
                    for h in range(self.H):
                        for m in r.rmap[l][h].values():
                            n += len(m)
                            m.clear()
        pipe = getattr(self.tp, "pipe", None)
        if pipe is not None:
            for s in range(pipe.ring):
                n += (pipe.slot_h2d[s] is not None) + (pipe.slot_sc[s] is not None)
                pipe.slot_h2d[s] = pipe.slot_sc[s] = None
        return n + n_ex                                                # HOOK-L4: with the executor's own count

    def report(self) -> Dict[str, Any]:
        """The run's accounting, labelled by mode (A2: GREEDY results are reported separately from SAMPLED results).
        committed = accepted drafts + corrections (A7); every discarded draft step, cancelled verifier row-layer, retry
        and verifier-only pass is reported as work; no mean hypothetical yield is reported as committed output."""
        s = self.stats
        out = dict(mode=self.cfg.mode.upper(), committed=s["accepted_drafts"] + s["corrections"],
                   accepted_drafts=s["accepted_drafts"], corrections=s["corrections"],
                   work=dict(draft_steps=s["draft_steps"] + s["draft_rows_cancelled_in_flight"],
                             draft_steps_discarded=s["draft_steps_discarded"], verifier_row_layers=s["verifier_row_layers"],
                             verifier_row_layers_cancelled=s["verifier_row_layers_cancelled"], cohort_waits=s["cohort_waits"],
                             verifier_only_passes=s["verifier_only_passes"]),
                   stalls={k[6:]: v for k, v in s.items() if k.startswith("stall_")},
                   stall_accounting=self.stall_accounting(),
                   verifier_evictions_of_draft_entries=self.vev_report(),
                   sharing_IDEAL=self.sharing(), memory=self.acct.report(), capacity=self.capacity_report(),
                   progress=self.progress_report())
        if self.arena_mode:
            out["physical"] = self.arena_report()                      # M2: addresses / allocation differ by construction
        if self.recent != "eager":
            out["ring"] = self.ring_report()                           # M2 B2: ring / D2H-check counters only
        if self.tickets is not None:
            out["tickets"] = dict(self.tickets.stats)                  # HOOK-L8 (I14): a copy, only with tickets on
        if self.deferred or self.cfg.store_handback:
            out["deferred"] = dict(self.dstat)                         # I17: only in deferred / handback mode
        if self.cstat:
            out["cancel"] = dict(self.cstat, cancelled=sorted(self.cancelled, key=repr))   # I20: only after a cancel
        return out

    def capacity_report(self) -> Dict[str, Any]:
        """The capacity mode and (rev6) its contract counters. Memory is charged for the FULL C + X positions of every
        stream (K6 counts FREE slots; in M2 arena mode the ALLOCATED P_l positions: B * H * (C + X) conservative, or
        B * H * C + X reduced, plus the logical C + X slot records), so no fit claim follows from them."""
        s = self.stats
        out = dict(mode=self.cfg.capacity)
        if self.rev6:
            out.update(W=self.W, d_share=self.d_share, X_d=self.X_d, R_ver=self.cfg.R_ver, N_max=self.N_max,
                       reservations=s.get("rev6_reservations", 0), reservations_returned=s.get("rev6_reservations_returned", 0),
                       credits_reserved=s.get("rev6_credits_reserved", 0), credits_returned=s.get("rev6_credits_returned", 0),
                       credits_fetch=s.get("rev6_credits_fetch", 0), credits_join=s.get("rev6_credits_join", 0),
                       credits_vres=s.get("rev6_credits_vres", 0), grants=s.get("rev6_grants", 0), grant_waits=s.get("rev6_grant_waits", 0),
                       stalls={k: s.get("stall_" + k, 0) for k in ("admission", "horizon", "capacity")},
                       fetch_draft_refused=s.get("fetch_draft_refused", 0), verifier_evictions=s.get("verifier_evictions", 0))
        if self.rev6_guard:                    # C81 / K3 note: the "ad" entitlement (not the allocation), sampled peaks
            out["guarantee"] = dict(
                mode="rev6", replaces="pin guard C >= (2K + 1) * S_max = %d" % ((2 * self.K + 1) * self.rb),
                C=self.cfg.C, S_max=self.rb, K=self.K, LAMBDA=self.LAMBDA, W=self.W, d_h=self.d_share,
                per_stream_budget=self.cfg.C + self.d_share, X_d=self.X_d, R_ver=self.cfg.R_ver, N_max=self.N_max,
                reservations_at_once=self.X_d // (self.H * self.d_share),
                peak_positions=s.get("rev6g_peak_positions", 0),
                peak_positions_over_C=s.get("rev6g_peak_positions_over_C", 0),
                peak_v_used=s.get("rev6g_peak_v_used", 0), memory_stalls=s.get("stall_memory", 0),
                pin_guard_would_stall=s.get("rev6g_pin_guard_would_stall", 0))
        return out

    def vev_report(self) -> Dict[str, Any]:
        """Repair 1 R2.5: every verifier eviction of a draft LRU entry (the counted progress fallback), per stream
        ("l<layer> b<batch slot> h<head>") with the evicted blocks in order, and the total."""
        per: Dict[str, List[int]] = {}
        for cid, l, b, h, blk, uid in self.vev_log:
            per.setdefault("l%d b%d h%d" % (l, b, h), []).append(blk)
        return dict(count=len(self.vev_log), per_stream=per)

    def stall_accounting(self) -> Dict[str, Any]:
        """C2 (fix round F3): every (active request, decode step) either starts a draft row or stalls with ONE reason.
        request_steps = draft_rows_started + stalls; each share is a fraction of request_steps; memory_per_lead is the
        ratio of 'memory' (pin guard) stalls to 'lead' stalls."""
        s = self.stats
        st = {k[6:]: v for k, v in s.items() if k.startswith("stall_")}
        started = s["draft_rows_started"]
        tot = started + sum(st.values())
        out = dict(request_steps=tot, draft_rows_started=started, draft_steps=s["draft_steps"],
                   lead=st.get("lead", 0), memory=st.get("memory", 0), done=st.get("done", 0),
                   admission=st.get("admission", 0), horizon=st.get("horizon", 0), capacity=st.get("capacity", 0))
        for k in ("lead", "memory", "done", "admission", "horizon", "capacity"):
            out["share_" + k] = out[k] / tot if tot else 0.0
        out["share_draft"] = started / tot if tot else 0.0
        out["memory_per_lead"] = out["memory"] / out["lead"] if out["lead"] else float("inf") if out["memory"] else 0.0
        return out

    def sharing(self) -> Dict[str, float]:
        """D1 IDEAL sharing from actual selections: sum over calls and streams of UNION / SUM (1.0 = no sharing)."""
        s = sum(x[5] for x in self.share)
        u = sum(x[6] for x in self.share)
        sh = sum(x[7] for x in self.share)
        uh = sum(x[8] for x in self.share)
        return dict(sum=s, union=u, ratio=(u / s if s else 1.0), sum_hist=sh, union_hist=uh, ratio_hist=(uh / sh if sh else 1.0))

