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
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from . import accept as acc
from . import cadence as cad
from . import rng
from .backend import check_backend, ev_done
from .compressed import CompressedChain
from .errors import ConfigError, KStepError, NoProgress, OwnershipViolation, ReadCorruption
from .executor import DRAFT, VERIFY, Row, check_executor
from .mutations import Mutations
from .ownership import COUNTS, Geometry, LayerPool, MemoryAccountant
from .transfer import Transfer

POISON = -7777777                         # no toy KV value is negative

PENDING, RUNNING, DONE, ACCEPTED, CANCELLED = "pending", "running", "done", "accepted", "cancelled"


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


class Draft:
    __slots__ = ("s", "pos", "tok_in", "epoch", "n_at", "m", "sel", "avail", "hidden", "d", "q", "state", "step")

    def __init__(self, s, pos, tok_in, epoch, n_at, m, step):
        self.s, self.pos, self.tok_in, self.epoch, self.n_at, self.m, self.step = s, pos, tok_in, epoch, n_at, m, step
        self.sel, self.avail = {}, {}
        self.hidden = None
        self.d, self.q = None, None
        self.state = RUNNING


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

    def __repr__(self):
        return "Cohort(uid=%s e=%d m=%d [%d,%d] %s L%d)" % (self.uid, self.epoch, self.m, self.a, self.b, self.state, self.next_layer)


class Request:
    def __init__(self, uid, b: int, prompt: List[int], N: int, L: int, H: int, stride: int, reduce):
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
        self.comp = [[CompressedChain(stride, reduce) for _ in range(H)] for _ in range(L)]
        self.dchain = None
        self.hist_blocks = 0
        self.hist_ev: Dict[Tuple[int, int], Any] = {}
        self.cur: Optional[Draft] = None

    def u(self, i: int) -> int:
        return self.toks[self.E] if i == 0 else self.drafts[i - 1].d

    def lead(self) -> int:
        return cad.lead(len(self.drafts), self.n - self.E)

    def last_pos(self) -> int:
        return self.P0 + self.N - 1

    def generated(self) -> int:
        return len(self.toks) - self.P0


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
        need = self.K * self.rb
        if cfg.C + cfg.X < need:
            raise ConfigError("K5/D3: C + X = %d < K * max_sel = %d: one cohort's exact selection cannot be resident at once"
                              % (cfg.C + cfg.X, need))
        if cfg.pin_guard and cfg.C < (2 * self.K + 1) * self.rb:
            raise ConfigError("D4 progress guarantee: C = %d < (2K + 1) * max_sel = %d (pin_guard=True; see Dispatcher docs)"
                              % (cfg.C, (2 * self.K + 1) * self.rb))
        self.pools = [LayerPool(l, cfg.B, self.H, cfg.C, cfg.X, cfg.R_ver) for l in range(self.L)]
        n_pos = self.pools[0].n_pos
        self.n_pos = n_pos
        z = lambda S: torch.full((cfg.B, S, self.H, self.D), POISON, dtype=torch.int64)
        self.host_k = [z(cfg.S_host) for _ in range(self.L)]
        self.host_v = [z(cfg.S_host) for _ in range(self.L)]
        self.slots_k = [z(n_pos * self.bs) for _ in range(self.L)]
        self.slots_v = [z(n_pos * self.bs) for _ in range(self.L)]
        self.tp = make_transport(self)
        self.acct = MemoryAccountant(cfg.geom, getattr(self.tp, "staging_bytes", 0))
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
        self._vfail = set()
        self._read_keys = set()
        self._n_new_val = 0

    # ================================================================== requests
    def submit(self, uid, prompt: List[int], max_new_tokens: int) -> None:
        if uid in self.reqs or any(q[0] == uid for q in self.queue):
            raise ValueError("duplicate request uid %r" % (uid,))
        if len(prompt) < 1:
            raise ValueError("empty prompt")
        if len(prompt) + max_new_tokens > self.cfg.S_host:
            raise ConfigError("prompt %d + max_new_tokens %d > S_host %d" % (len(prompt), max_new_tokens, self.cfg.S_host))
        self.queue.append((uid, list(prompt), int(max_new_tokens)))

    def active(self) -> List[Request]:
        return [r for r in self.slot_req if r is not None]

    def _admit(self) -> None:
        for b in range(self.cfg.B):
            if self.slot_req[b] is not None or not self.queue:
                continue
            uid, prompt, N = self.queue.pop(0)
            r = Request(uid, b, prompt, N, self.L, self.H, self.stride, self.ex.compress)
            self.reqs[uid] = r
            self.slot_req[b] = r
            for l in range(self.L):
                for h in range(self.H):
                    self.pools[l].stream(b, h).owner = uid
            kv = self.ex.prefill(prompt[:-1])
            for pos in range(len(prompt) - 1):
                for l in range(self.L):
                    for h in range(self.H):
                        r.pub[l][h][pos] = kv[pos][l][h]
                        r.comp[l][h].append(kv[pos][l][h][0])
            r.dchain = [[c.copy() for c in row] for row in r.comp]
            self._history(r)
            self.trace[uid] = dict(prompt=list(prompt), N=N, b=b, epochs=[dict(epoch=0, E=r.E, u0=r.toks[r.E], drafts=[], cohorts=[], outcome=None)])
            self.stats["requests_admitted"] += 1
            if r.generated() >= r.N:
                self._end_request(r)

    def _end_request(self, r: Request) -> None:
        for l in range(self.L):
            for h in range(self.H):
                self.pools[l].stream(r.b, h).end_owner()
        self.pending_release = [x for x in self.pending_release if x[0][0] != r.uid]
        self.key_readers = {k: v for k, v in self.key_readers.items() if k[0][0] != r.uid}
        self.rel_layers = {k: v for k, v in self.rel_layers.items() if k[0] != r.uid}
        r.done = True
        self.results[r.uid] = r.toks[r.P0:r.P0 + r.N]
        self.trace[r.uid]["final"] = list(r.toks)
        self.slot_req[r.b] = None
        self.stats["requests_done"] += 1

    # ================================================================== the step
    def run(self, max_steps: int = 200000) -> Dict[Any, List[int]]:
        while self.queue or self.active():
            self.step()
            if self.t > max_steps:
                raise KStepError("run: more than %d steps" % max_steps)
        self.finish()
        return self.results

    def step(self) -> None:
        t = self.t
        self._admit()
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
        for r in self.active():
            why = self._stall_reason(r)
            if why:
                self.stalls.append((t, r.uid, why, r.lead()))
                self.stats["stall_" + why] += 1
                continue
            s = len(r.drafts) + 1
            dr = Draft(s, r.E + s - 1, r.u(s - 1), r.epoch, r.n, cad.cohort_of(s, self.K), t)
            dr.hidden = self.ex.embed(dr.tok_in)
            r.cur = dr
            rows.append((r, dr))
            self.stats["draft_rows_started"] += 1
        vonly = not rows
        if vonly:
            if not any(c.state in (RUNNING, PENDING) for r in self.active() for c in r.cohorts):
                if self.active():
                    raise KStepError("step %d: no draft row and no cohort, but requests are active" % t)
                self.t += 1
                return
            self.stats["verifier_only_passes"] += 1
            for key_l in [x for x in self.pending_release]:              # D4: protections waiting for a reader event
                fr, lr = self.key_readers.get(key_l, (None, None))
                if lr is not None and not ev_done(lr):
                    self.be.host_wait(lr)
                    self.stats["forced_readers"] += 1
        before = self.stats["progress"]
        for l in range(self.L):
            if vonly:
                self._force_for_layer(l)
            self._call(l, rows)
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

    # ================================================================== host poll
    def _poll(self) -> None:
        self.tp.poll()
        arrived = sorted((tr for tr in self.inflight.values() if self.tp.done(tr)), key=lambda x: x.seq)
        for tr in arrived:
            del self.inflight[tr.seq]
            st = self.pools[tr.layer].stream(tr.b, tr.h)
            res = st.admit(tr.slot, tr.uid, tr.block, tr.gen, self.mut, demand=tr.demand)
            self.arrival_log.append(tr.seq)
            self.stats["progress"] += 1
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
        if self.cancelled_keys:                                         # K6: kept only while a transfer may still name it
            self.cancelled_keys &= {tr.requester for tr in self.inflight.values()}
        for pool in self.pools:
            n = pool.poll_releases(self.mut)
            if n:
                self.stats["slots_freed"] += n
                self.stats["progress"] += 1
        self._validate()
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

    def _release_refs(self, key, l: int) -> None:
        """R1: drop the references keyed `key` at layer l (one key, R1.2); PO / VRES slots that lose their last reference
        go through the reader gate. A second release of the same (key, l) is an ownership violation (R1.4)."""
        lay = self.rel_layers.get(key)
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

    def _validate(self, all_: bool = False) -> None:
        keep = []
        for ev, ws, exp, info in self.validators:
            if all_ or ev_done(ev):
                if not torch.equal(ws, exp):
                    raise ReadCorruption("deferred read of %s saw other bytes than the call computed from" % (info,))
                self.stats["reads_validated"] += 1
            else:
                keep.append((ev, ws, exp, info))
        self.validators = keep

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
        for i, pos in enumerate(c.positions):
            for h in range(self.H):
                chain[h].append(pre[i][1][h])
        sel = []
        for i, pos in enumerate(c.positions):
            sel.append([self.ex.select(l, h, pre[i][0][h], chain[h].cks, pos // self.bs) for h in range(self.H)])
        c.ws[l] = dict(pre=pre, sel=sel, chain=chain)

    def _need(self, r: Request, sel_rows, l: int):
        """Per head: (ready, inflight, missing) historical blocks of a set of rows' selections."""
        out = []
        for h in range(self.H):
            st = self.pools[l].stream(r.b, h)
            blocks = sorted({j for sel in sel_rows for j in sel[h] if j < r.hist_blocks})
            ready = [j for j in blocks if st.resident(j) is not None]
            infl = [j for j in blocks if st.resident(j) is None and j in st.pending]
            miss = [j for j in blocks if st.resident(j) is None and j not in st.pending]
            out.append((ready, infl, miss))
        return out

    def _call(self, l: int, rows: List[Tuple[Request, Draft]]) -> None:
        self.call_id += 1
        cid = self.call_id
        self._read_keys = set()
        self._n_new_val = 0
        self.call_step[cid] = self.t
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
        for r, c in eligible:
            need = self._need(r, c.ws[l]["sel"], l)
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                for blocks in (need[h][1:] if self.mut.no_verifier_ready_protection else need[h]):
                    for j in blocks:
                        self._protect(st, l, j, c.key)             # K2: the exact selection, while it waits
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
            q, k, v = draft_pre_raw[id(dr)]
            for h in range(self.H):
                if r.dchain[l][h].count != dr.pos:
                    raise KStepError("D6: the draft chain of uid %s (l %d h %d) holds %d keys at draft position %d"
                                     % (r.uid, l, h, r.dchain[l][h].count, dr.pos))
            sel = [self.ex.select(l, h, q[h], r.dchain[l][h].cks, dr.pos // self.bs) for h in range(self.H)]
            key = (r.uid, r.epoch, dr.m)
            avail = []
            for h in range(self.H):
                r.draft_kv[l][h][dr.pos] = (k[h], v[h])
                r.dchain[l][h].append(k[h])
                st = self.pools[l].stream(r.b, h)
                a = []
                for j in sel[h]:
                    if j >= r.hist_blocks:
                        continue
                    if not (self.mut.no_draft_hit_protection and st.resident(j) is not None):
                        self._protect(st, l, j, key)                   # K2: hits AND misses, for the cohort of d_s
                    if st.resident(j) is not None:
                        a.append(j)
                    elif j not in st.pending:
                        requests.append((DRAFT, r, h, j, key))
                    else:
                        self._add_demand(st, j, DRAFT)                 # R2.3: joins the transfer already in flight
                avail.append(a)
            dr.sel[l], dr.avail[l] = sel, avail
            draft_pre[id(dr)] = (q, k, v)
        # R2.2: verifier reads never touch the logical LRU; they are logged separately (cohort order, position order)
        for r, c in exec_cohorts:
            for i, pos in enumerate(c.positions):
                for h in range(self.H):
                    st = self.pools[l].stream(r.b, h)
                    for j in c.ws[l]["sel"][i][h]:
                        if j < r.hist_blocks:
                            self.vread_log.append((cid, l, r.b, h, j, (c.m, pos)))
                            if self.mut.verifier_touches_lru:
                                st.touch(j)                            # NEGATIVE CONTROL (R2.6): the M1 rule
        # phase 2: reads + executor
        jrows: List[Row] = []
        owners: List[Tuple] = []
        read_slots: List[Tuple[int, int, int, Any]] = []
        for r, c in exec_cohorts:
            pre, sel = c.ws[l]["pre"], c.ws[l]["sel"]
            own = {pos: i for i, pos in enumerate(c.positions)}
            for i, pos in enumerate(c.positions):
                kv = []
                for h in range(self.H):
                    def recent(qp, h=h, i=i):
                        if qp < r.n:
                            return r.pub[l][h][qp]
                        if qp in own:
                            if own[qp] > i:
                                raise KStepError("causality: row %d reads a later position %d" % (pos, qp))
                            ii = own[qp]
                            return (pre[ii][1][h], pre[ii][2][h])
                        return self._prov(r, c, l, h, qp)
                    kv.append(self._row_kv(r, l, h, pos, sel[i][h], None, recent, read_slots, c.key))
                jrows.append(Row(r.uid, r.b, pos, VERIFY, c.key, sel[i], pos,
                                 [[j for j in sel[i][hh] if j < r.hist_blocks] for hh in range(self.H)],
                                 c.hidden[i], pre[i][0], kv))
                owners.append(("v", r, c, i))
        for r, dr in rows:
            if id(dr) not in draft_pre:
                continue
            q, k, v = draft_pre[id(dr)]
            kv = []
            for h in range(self.H):
                def recent(qp, h=h):
                    if qp < r.n:
                        return r.pub[l][h][qp]
                    return r.draft_kv[l][h][qp]
                kv.append(self._row_kv(r, l, h, dr.pos, dr.sel[l][h], dr.avail[l][h], recent, read_slots,
                                       (r.uid, r.epoch, dr.m)))
            jrows.append(Row(r.uid, r.b, dr.pos, DRAFT, (r.uid, r.epoch, dr.m), dr.sel[l], dr.pos, dr.avail[l], dr.hidden, q, kv))
            owners.append(("d", r, dr, None))
        if read_slots:
            with self.be.stream("compute"):
                rev = self.be.event_rec("compute")
            for l_, b_, h_, idx in read_slots:
                s = self.pools[l_].stream(b_, h_).slots[idx]
                if s.first_reader is None:
                    s.first_reader = rev
                s.last_reader = rev
            for key in self._read_keys:
                kr = self.key_readers.setdefault((key, l), [None, None])
                if kr[0] is None:
                    kr[0] = rev
                kr[1] = rev
            for i in range(len(self.validators) - self._n_new_val, len(self.validators)):
                ev, ws, exp, info = self.validators[i]
                self.validators[i] = (rev, ws, exp, info)
        self._draft_recency(l, rows, draft_pre, cid)
        outs = self.ex.joint_post(l, jrows) if jrows else []
        for (kind, r, obj, i), row, out in zip(owners, jrows, outs):
            if kind == "v":
                c = obj
                c.hidden[i] = out
                c.row_layers += 1
                self.stats["verifier_row_layers"] += 1
            else:
                obj.hidden = out
        for r, c in exec_cohorts:
            pre = c.ws[l]["pre"]
            c.prov[l] = [{pos: (pre[i][1][h], pre[i][2][h]) for i, pos in enumerate(c.positions)} for h in range(self.H)]
            c.chain[l] = c.ws[l]["chain"]
            c.vsel[l] = [[list(x) for x in row] for row in c.ws[l]["sel"]]
            c.exec_call[l] = cid
            del c.ws[l]
            c.next_layer = l + 1
            self.exec_log.append((cid, l, r.uid, c.epoch, c.m))
            self.stats["progress"] += 1
            if self.mut.release_at_execution:                          # NEGATIVE CONTROL (R1.6 early release)
                self.rel_layers.setdefault(c.key, {})[l] = False
                self._release_refs(c.key, l)
            elif not self.mut.release_only_at_decision:                # (the control keeps the M1 decision-only rule)
                self._queue_release(c.key, l)                          # R1.1: behind the (key, l) reader gate
            if l == self.L - 1:
                c.p = {c.a + i: self.ex.dist(c.hidden[i]) for i in range(len(c.positions))}
                c.hidden = []                                          # K6: the rows' hidden states are no longer held
                c.state = DONE
        # D1: per stream SUM and UNION of the selected blocks of the executing rows
        per = defaultdict(lambda: [0, set(), 0, set(), 0])
        for row in jrows:
            for h in range(self.H):
                x = per[(row.b, h)]
                x[0] += len(row.sel[h])
                x[1].update(row.sel[h])
                hist = [j for j in row.sel[h] if j < self.reqs[row.uid].hist_blocks]
                x[2] += len(hist)
                x[3].update(hist)
                x[4] += 1
        for (b, h), x in per.items():
            self.share.append((cid, l, b, h, x[4], x[0], len(x[1]), x[2], len(x[3])))
        self._issue(l, requests, cid)
        self._sample_memory()
        if self.cfg.check_invariants and not self.mut.active():
            for st in self.pools[l].streams.values():
                st.check()
        if self.observer is not None:
            self.observer(self, "call", cid, l)

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

    def _draft_recency(self, l: int, rows, draft_pre, cid: int) -> None:
        """R2.1 (a), (c): the DRAFT-ONLY recency of the call at layer l, after the rows' reads were enqueued (a slot that
        an eviction releases here is FREE only after this call's reader event). Per stream, the draft row's reads of LRU
        entries move to MRU in read order, then its reads of resident non-LRU entries (PO, VRES) enter the LRU at MRU
        (draft acquisition), each followed by the eviction rule of an arrival."""
        for r, dr in rows:
            if id(dr) not in draft_pre:
                continue
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                reads = dr.avail[l][h]
                hits = [j for j in reads if j in st.lru]
                acq = [j for j in reads if j not in st.lru]
                for j in hits:
                    st.touch(j)
                    self.touch_log.append((cid, l, r.b, h, j, "hit"))
                for j in acq:
                    if j not in st.po and j not in st.vres:
                        raise KStepError("R2.1: the draft row of uid %s read block %d (stream %s), which is no longer "
                                         "resident at the recency update" % (r.uid, j, (l, h, r.b)))
                    if self.mut.skip_acquisition:
                        continue                                       # NEGATIVE CONTROL (R2.6)
                    res = st.acquire(j, self.mut)
                    self.touch_log.append((cid, l, r.b, h, j, "acquire_" + res["src"]))
                    self.stats["draft_acquisitions_" + res["src"]] += 1  # draft_acquisitions_po / _vres
                    ev = res["evicted"]
                    if ev is not None:
                        self.stats["evict_protected_only" if ev["protected"] else "evict_released"] += 1

    def _protect(self, st, l: int, j: int, key) -> None:
        """K2 with the R1.5 guard: no reference keyed `key` is added at layer l after (key, l) was queued for release."""
        lay = self.rel_layers.get(key)
        if lay is not None and l in lay and not self.mut.no_late_protection_guard:
            raise KStepError("R1.5: late protection of block %d at layer %d for cohort %s after its layer-%d references "
                             "were queued for release" % (j, l, key, l))
        st.protect(j, key)

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

    def _queue_release(self, key, l: int) -> None:
        """R1.1 / R1.4: queue the release of every reference keyed `key` at layer l, at most once per (key, l)."""
        lay = self.rel_layers.setdefault(key, {})
        if l in lay:
            return
        lay[l] = False
        self.pending_release.append((key, l))

    def _row_kv(self, r: Request, l: int, h: int, pos: int, sel: List[int], readable: Optional[List[int]], recent,
                read_slots: List, key) -> List[Tuple]:
        """The (k, v) list one row attends at one head, in increasing position order: the selected blocks it may read,
        then its causal recent region. Slot reads are eager (the joint call computes from them) AND enqueued as deferred
        copies on the "compute" stream (validated later)."""
        bs = self.bs
        st = self.pools[l].stream(r.b, h)
        out = []
        tail0 = (pos // bs) * bs
        for j in sel:
            if j < r.hist_blocks:
                if readable is not None and j not in readable:
                    continue                                           # draft: masked miss
                idx = st.resident(j)
                if idx is None and self.mut.read_not_ready:
                    idx = st.pending.get(j)
                if idx is None:
                    raise KStepError("row at %d reads block %d that is not resident (stream %s)" % (pos, j, (l, h, r.b)))
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
                for rr, (kk, vv_) in enumerate(zip(ek.tolist(), evv.tolist())):
                    out.append((j * bs + rr, tuple(kk), tuple(vv_)))
            else:
                for qp in range(j * bs, (j + 1) * bs):
                    k, v = recent(qp)
                    out.append((qp, k, v))
        for qp in range(tail0, pos + 1):
            k, v = recent(qp)
            out.append((qp, k, v))
        out.sort(key=lambda x: x[0])
        return [(k, v) for _, k, v in out]

    # ================================================================== fetch issue (K5, T3)
    def _issue(self, l: int, requests: List[Tuple], cid: int) -> None:
        pool = self.pools[l]
        seen = set()
        issued: Dict[Tuple[int, int, int], Transfer] = {}
        batch = []
        order = [x for x in requests if x[0] == VERIFY] + [x for x in requests if x[0] == DRAFT]
        for kind, r, h, j, key in order:
            if (r.b, h, j) in seen:
                tr0 = issued.get((r.b, h, j))
                if tr0 is not None and kind not in tr0.demand and not self.mut.drop_demand_merge:
                    tr0.demand.add(kind)                                  # R2.3: joins the transfer issued in this call
                    self.stats["demand_merged_" + kind] += 1
                continue
            seen.add((r.b, h, j))
            st = pool.stream(r.b, h)
            if st.resident(j) is not None:
                continue
            if j in st.pending:
                self._add_demand(st, j, kind)                             # R2.3
                continue
            if kind == VERIFY and (self.mut.verifier_evicts_without_pressure or not pool.can_take(st, kind)):
                self._verifier_evict(pool, st, cid)                       # R2.5: the counted progress fallback
            if not pool.can_take(st, kind):
                if kind == DRAFT:
                    self.stats["fetch_draft_refused"] += 1                # unrequested; the draft continues masked
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
            gen = st.take(idx, r.uid, j, self.seq)
            tr = Transfer(self.seq, r.uid, l, r.b, h, j, idx, gen, kind, key, hist_ev=r.hist_ev.get((l, j)),
                          reader_ev=prev, issued_call=cid)
            self.inflight[tr.seq] = tr
            issued[(r.b, h, j)] = tr
            batch.append(tr)
            self.stats["fetch_%s_issued" % kind] += 1
        if batch:
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
        progress fallback, recorded per stream with the evicted block (vev_log, the request's trace, report())."""
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

    def _finish_draft(self, r: Request, dr: Draft) -> None:
        dist = self.ex.dist(dr.hidden)
        dr.hidden = None                                               # K6: only d and q are kept
        u = rng.uniform(self.cfg.seed, r.uid, dr.pos + 1, r.epoch, "draft")
        dr.d, dr.q = acc.draft_token(self.cfg.mode, dist, u)
        dr.state = DONE
        r.drafts.append(dr)
        self.stats["draft_steps"] += 1
        self.stats["progress"] += 1
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
                    ch.append(dk[qp][0])
                r.dchain[l][h] = ch

    def _finished_key(self, key) -> None:
        """R1.4 decision cleanup (acceptance, cancellation, untriggered key): queue only the layers of `key` that are not
        queued yet; a layer queued after its execution (R1.1) is not queued again."""
        if self.mut.cleanup_skips_unqueued_layers:
            return                                                     # NEGATIVE CONTROL (R1.6 rollback)
        for l in range(self.L):
            if self.mut.cleanup_double_queues:                         # NEGATIVE CONTROL (R1.6 rollback)
                self.rel_layers.setdefault(key, {}).setdefault(l, False)
                self.pending_release.append((key, l))
            else:
                self._queue_release(key, l)

    def _apply(self, r: Request, c: Cohort, o) -> None:
        ep = self.trace[r.uid]["epochs"][-1]
        self.stats["accepted_drafts"] += o.n_acc
        self.stats["zero_mass_accepts"] += o.zero_mass_accepts
        self.stats["progress"] += 1
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
        if any((qp + 1) % self.stride == 0 for qp in range(lo, hi)):
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
                    r.comp[l][h].append(kv[0])
                    self.published[(r.uid, pos, l, h)] = kv
        r.n = c.E + upto_commit
        self._history(r)

    def _drop_superseded(self, r: Request) -> None:
        """D6: draft KV of a published position is superseded (drafts read the exact KV from then on)."""
        for l in range(self.L):
            for h in range(self.H):
                for qp in [x for x in r.draft_kv[l][h] if x < r.n]:
                    del r.draft_kv[l][h][qp]

    def _history(self, r: Request) -> None:
        """A5: every committed block that became complete is written to the host (D2H on the "d2h" stream, one event per
        layer and block) and becomes fetchable; its positions leave the recent region."""
        bs = self.bs
        while (r.hist_blocks + 1) * bs <= r.n:
            j = r.hist_blocks
            for l in range(self.L):
                kb = torch.tensor([[r.pub[l][h][pos][0] for h in range(self.H)] for pos in range(j * bs, (j + 1) * bs)], dtype=torch.int64)
                vb = torch.tensor([[r.pub[l][h][pos][1] for h in range(self.H)] for pos in range(j * bs, (j + 1) * bs)], dtype=torch.int64)
                with self.be.stream("d2h"):
                    self.be.memcpy(self.host_k[l][r.b, j * bs:(j + 1) * bs], kb, "d2h")
                    self.be.memcpy(self.host_v[l][r.b, j * bs:(j + 1) * bs], vb, "d2h")
                    r.hist_ev[(l, j)] = self.be.event_rec("d2h")
                for h in range(self.H):
                    for pos in range(j * bs, (j + 1) * bs):
                        del r.pub[l][h][pos]
            r.hist_blocks += 1
            self.stats["history_blocks"] += 1

    # ================================================================== memory, finish
    @staticmethod
    def _chain_entries(ch: CompressedChain) -> int:
        return len(ch.cks) + len(ch.buf)

    def _sample_memory(self) -> None:
        """K6: count EVERY held item (no filter that could hide a leak); MemoryAccountant turns counts into bytes."""
        n = defaultdict(int)
        for pool in self.pools:
            for k, v in pool.counts().items():
                n[k] += v                                              # physical slots by state, FREE included
            for st in pool.streams.values():
                n["prot_refs"] += sum(len(v) for v in st.prot.values())
        if self.mut.mem_no_releasing:
            n["releasing"] = 0                                         # NEGATIVE CONTROL (fix round F4)
        if self.mut.vres_counted_as_lru:
            n["lru"] += n["vres"]                                      # NEGATIVE CONTROL (repair 1 R2.6)
            n["vres"] = 0
        n["transfers"] = len(self.inflight)
        n["release_entries"] = (len(self.pending_release) + len(self.key_readers)
                                + sum(len(v) for v in self.rel_layers.values()))
        for r in self.active():
            drafts = list(r.drafts) + ([r.cur] if r.cur is not None else [])
            n["drafts"] += len(drafts)
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
        self.acct.sample({k: n[k] for k in COUNTS})

    def finish(self, order: Optional[Tuple[str, ...]] = None) -> None:
        """Drain every stream (deferred reads, scatters, history writes) and validate every deferred read."""
        self.be.synchronize(order or self.cfg.drain_order)
        self._validate(all_=True)

    def report(self) -> Dict[str, Any]:
        """The run's accounting, labelled by mode (A2: GREEDY results are reported separately from SAMPLED results).
        committed = accepted drafts + corrections (A7); every discarded draft step, cancelled verifier row-layer, retry
        and verifier-only pass is reported as work; no mean hypothetical yield is reported as committed output."""
        s = self.stats
        return dict(mode=self.cfg.mode.upper(), committed=s["accepted_drafts"] + s["corrections"],
                    accepted_drafts=s["accepted_drafts"], corrections=s["corrections"],
                    work=dict(draft_steps=s["draft_steps"] + s["draft_rows_cancelled_in_flight"],
                              draft_steps_discarded=s["draft_steps_discarded"], verifier_row_layers=s["verifier_row_layers"],
                              verifier_row_layers_cancelled=s["verifier_row_layers_cancelled"], cohort_waits=s["cohort_waits"],
                              verifier_only_passes=s["verifier_only_passes"]),
                    stalls={k[6:]: v for k, v in s.items() if k.startswith("stall_")},
                    stall_accounting=self.stall_accounting(),
                    verifier_evictions_of_draft_entries=self.vev_report(),
                    sharing_IDEAL=self.sharing(), memory=self.acct.report())

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
                   lead=st.get("lead", 0), memory=st.get("memory", 0), done=st.get("done", 0))
        for k in ("lead", "memory", "done"):
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

