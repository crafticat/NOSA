"""RUNG L5 of Phase 2 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
BACKGROUND PREFETCH OF THE DRAFT MISSES, with the soft pin, for the speculative round runner (SpecRunner) at levels L3
and L4P (paper sections 3.1 and 3.3; the measurement of section 4.3). OPT-IN: only the cell
benchmarks/Efficiency/r2_spec5_cell.py (--spec-prefetch on, arms L3P and L4PP) and the CPU tests build a SpecPrefetch.
No engine module imports this file. With SpecOps(prefetch=None) the round runner runs the code of the tree
nosi-r2-specp a6c9d10, bit for bit (tests_cpu/test_spec_prefetch.py, Gate 0).

NO NEW KERNEL. SpecPrefetch is a subclass of pool_prefetch.PoolPrefetch (Task 1.2, cell pf1). It reuses, unchanged:
the C++ packer and the copy engine (r2_transport_gilfree_v3, worker "cpp", issue_desc on the flat staging lanes),
PoolPrefetch._issue (dedup, lanes, the descriptor job, the delivery probes), PoolPrefetch._commit and _commit_layer
(the arrival wait, the last-reader wait, the dedup against the window and the pool, the LRU slot that is not pinned,
the SHIPPED scatter kernels flash_h2d_from_mask / flash_h2d_from_mask_bias, the map write and the SOFT PIN
_pool_age = PIN_BASE + n). New code is torch on small tables (the tags, the counters) and host bookkeeping.

THE FLOW OF ONE ROUND (the round runner calls the seams; module SpecRunner)
  on_draft(miss_ids)   after each draft step (L3: the K drafts of the round; L4P: the joint pass's draft row and the
                       K - 1 drafts, the prelude's K drafts). The skipped blocks of that step (DraftResident.miss_ids,
                       Task 1.1's miss list, (H, B, topk) per layer) go into a SNAPSHOT: one stack, one cast to int16,
                       one D2H copy into a pinned buffer and one event, on the main stream. LAG 1: the host then waits
                       the snapshot of the PREVIOUS draft step (that step is done on the GPU; the current one runs) and
                       issues its blocks as one descriptor job. The host never waits the newest step.
  on_verify()          OUTSIDE A ROLLBACK GROUP, before the verify (L3: after the drafts' reject_all, before the verify
                       group; L4P: at the round start, after RequestPin.apply, before the joint pass's group). It issues
                       the last snapshot, classifies each job of the epoch (landed: READY; not landed: IN FLIGHT),
                       merges the epoch's jobs into ONE commit per layer and enqueues it on the commit stream. The host
                       waits only the SUBMISSION of the newest job (the C++ worker's counter, the GIL released); the
                       commit stream waits its arrival on the device.
  the verify           the first residency of layer l waits the commit of layer l on the main stream (CUDA events A_l,
                       B_l around the wait: the IN FLIGHT wait). The verify finds a prefetched block as an ORDINARY POOL
                       HIT (SWAP / MOVE_IN of pool_update) and the urgent GPU gather skips it.
  on_release()         OUTSIDE A GROUP, after the round's commit (before RequestPin.release): the tags that remain are
                       counted (unread in their slot / moved by a draft step or an undo) and every prefetch pin that is
                       left gets the slot's age from before the commit back.
  on_round("end")      L4P: issues the last draft's snapshot (its job lands during the GPU's commit phase; the next
                       round's on_verify commits it).
  An EPOCH = the jobs that one on_verify commits (L3: the drafts of the same round; L4P: the draft rows and drafts of
  the previous round, or the prelude).

ROLLBACK AND PREFETCH (the proof; docs/NOTES_spec5.md section 3). THE PREFETCHED POOL SLOTS AND PINS ARE WRITTEN ONLY
OUTSIDE A ROLLBACK GROUP. A group's journal takes the engine part (the maps, the four pool tables) at each tentative
step's decode_fixed, and the slot journal saves every slot that a step overwrites. A commit that lands before the group
begins is therefore part of the state that every journal entry of the group starts from: the undo of any step gives
back a state that holds the prefetched blocks, with their bytes (the saved slots, the inverse moves of the named slots,
or the host fallback: a prefetched block is a COMPLETE block, so its host rows and its total_cis rows are its bytes).
The undo never needs to know that a block came from the prefetch. on_verify and on_release refuse to run inside a
group. The main stream waits the commit of layer l before the first journal copy of layer l (the wait is in the
outermost hook of the body's decode_fixed, so it runs before the rollback's journal hook). The negative control
commit_mid_group (tests only) commits inside the verify group instead: the undo then names a pool slot with the bytes of
another block, and the residency check catches it.

THE SOFT PIN (paper 3.3). PoolPrefetch writes _pool_age = PIN_BASE + n for each committed slot (here n starts at
PF_SEQ0 = 2**40, so a prefetch pin is above every RequestPin age PIN_BASE + [0, H * B * P)). pool_update takes a pinned
slot as an LRU victim last; a pool hit gives the slot a real stamp, so the pin ends at the first read. It is not a hard
pin: when a row has fewer unpinned unclaimed slots than MOVE_OUTs, pool_update evicts a pinned slot; that is a PIN
OVERFLOW and it is counted (evict_overflow). A prefetched block that is evicted while an unpinned unclaimed slot existed
(evict_free) breaks the pin; it is 0 by the soft-pin rule. The negative control no_pin (tests only) gives the slot the
victim's old age: evict_free > 0.

THE TAGS (the accounting; one (H, B, P + 1) int64 table per layer, column P a dummy). tag = the class of the job that
delivered the slot's block (0 READY: landed before the verify started; 1 IN FLIGHT), -1 none. Each verify residency
(the meter's verify bracket; every layer and every round, with or without a commit) counts, per (head, request): the selected slots (sel), the host misses (new: the urgent GPU
gather), the host misses of a block that the prefetch delivered in this epoch (refetch), the pool hits on a tagged slot
of a verify request (read_ready / read_inflight) and of a draft request of a joint pass (read_draft), the MOVE_OUTs of a
tagged slot (evict_overflow / evict_free); a touched slot loses its tag. on_release counts the tags that remain: unread
(the slot still holds the block) or moved. PIN CONSERVATION (exact, per run):
    committed == read_ready + read_inflight + read_draft + evict_overflow + evict_free + unread + moved
and no prefetch pin is left after a release. THE SPLIT of the verify-selected slots (paper 4.3):
    sel == ready_resident + ready_prefetched (read_ready) + in_flight (read_inflight) + new
ready_resident is the remainder; the CPU test recounts every part from the residency itself.

HOST SYNCS (by design; tests_cpu/test_spec_prefetch.py traps them): the event of the PREVIOUS draft step's snapshot
(_issue_snap), the submission of the newest job (_commit), the job watermark (on_verify). The per-call hook and the
accounting have none. report() synchronizes once.

NEGATIVE CONTROLS (tests only; neg): no_pin (no soft pin: the slot keeps the victim's age), commit_mid_group (L3: the
epoch's commit runs inside the verify group, at verify step K - 1).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .kstep.backend import ev_done
from .pool_prefetch import PIN_BASE, POOL_MOVE_IN, POOL_MOVE_OUT, POOL_SWAP, PoolPrefetch, _Job

PF_SEQ0 = 1 << 40                       # the first prefetch pin offset: PIN_BASE + PF_SEQ0 + n
LEVELS = ("L3", "L4P")
NEG_KINDS = (None, "no_pin", "commit_mid_group")
CLASSES = ("ready", "in_flight")        # tag 0 / 1: the job landed before the verify started / it had not landed
# the device counters of the main stream (acct) and of the commit stream (acct_cs): two tensors, so the two streams
# never write one tensor
ACCT = ("sel", "new", "refetch", "read_ready", "read_inflight", "read_draft", "evict_overflow", "evict_free",
        "unread", "moved")
I_SEL, I_NEW, I_REFETCH, I_READ0, I_READ1, I_READ_DRAFT, I_EV_OVF, I_EV_FREE, I_UNREAD, I_MOVED = range(10)
I_DUMMY = len(ACCT)
ACCT_CS = ("committed", "victim_named")
LAG = 1                                 # snapshots kept back: the host waits a draft step only after the next one runs


class SpecPrefetchError(RuntimeError):
    pass


class _EpochJob(_Job):
    """The merged job of one epoch: per layer the items (5, n) [h, b, X, lane, group start] and their class (n,)."""
    __slots__ = ("cls", "n_class")

    def __init__(self, seq: int, items, cls, n_class):
        super().__init__(seq, items)
        self.cls, self.n_class = cls, n_class


class SpecPrefetch(PoolPrefetch):
    """See the module docstring. Build it after ragged_engine.attach (the per-request engine, victim pool on), pass it
    as SpecOps(prefetch=...); the SpecRunner binds it (bind) and unbinds it (detach). close() after the run.
      rs          the RaggedState
      ext         the gilfree_pack v3 extension (not needed with `transport`)
      lanes       staging lanes per (layer, KV head) and EPOCH (the items past them are refused and counted)
      threads, cores, cap_groups, ring   the packer (PoolPrefetch)
      ops, transport, host_wait          tests: the scatter twins, a transport stand-in, the CPU event wait
      neg         a negative control (NEG_KINDS), tests only"""

    def __init__(self, rs, ext=None, *, lanes: int = 128, threads: int = 8, cores=None, cap_groups: int = 256,
                 ring: int = 2, ops=None, transport=None, host_wait=None, probes: bool = True, neg=None):
        if neg not in NEG_KINDS:
            raise SpecPrefetchError("unknown negative control %r (one of %s)" % (neg, NEG_KINDS))
        if not bool(getattr(rs, "pool", False)):
            raise SpecPrefetchError("the prefetch needs the victim pool (NOSI_POOL_BLOCKS > 0)")
        self.rs, self.neg = rs, neg
        self.runner = None
        self._probe_i = 0
        super().__init__(rs.cache, ext, fetch="gpugather", lanes=lanes, threads=threads, cores=cores,
                         cap_groups=cap_groups, ring=ring, bg_mib=0.0, bg_commit="none", ops=ops, transport=transport,
                         host_wait=host_wait, probes=probes)
        # the bias source: the ragged twin's total_cis (ragged_engine.attach frees the shipped one)
        self.total_cis = [tw.total_cis for tw in rs.st.twins]
        self.pin_seq = PF_SEQ0
        dev, L, H, B, P, M = self.dev, self.L, self.H, self.B, self.P, self.topk
        self.NB = int(rs.st.M_alloc)                                   # block ids of a selection are < M_alloc
        i64 = torch.int64
        self.tags = [torch.full((H, B, P + 1), -1, dtype=i64, device=dev) for _ in range(L)]
        self.age0 = [torch.zeros((H, B, P), dtype=i64, device=dev) for _ in range(L)]
        self.blk = [torch.full((H, B, P), -1, dtype=i64, device=dev) for _ in range(L)]
        self.delivered = [torch.zeros((H, B, self.NB + 1), dtype=torch.bool, device=dev) for _ in range(L)]
        self._clm = torch.zeros((H, B, P + 1), dtype=torch.bool, device=dev)
        self._ones = torch.ones((H * B * M,), dtype=i64, device=dev)
        self.acct = torch.zeros((len(ACCT) + 1,), dtype=i64, device=dev)       # + the dummy entry
        self.acct_cs = torch.zeros((len(ACCT_CS),), dtype=i64, device=dev)
        self._sdt = torch.int16 if self.NB < (1 << 15) - 1 else torch.int32
        self._sbuf = None
        if self.cuda:
            self._sbuf = [torch.empty((L, H, B, M), dtype=self._sdt, pin_memory=True) for _ in range(LAG + 1)]
        self.req = torch.zeros((L * H * B * self.NB,), dtype=torch.bool)    # host: requested in this epoch
        self.snaps: List[tuple] = []
        self._snap_n = 0
        self._need = [False] * L                  # the next residency of layer l waits the layer's commit
        self._live = [False] * L                  # layer l has tags of this epoch
        self._calls = [0] * L                     # residencies of layer l in the verify bracket (L4P: the joint pass)
        self._lay = {e._k_gpu.data_ptr(): l for l, e in enumerate(self.engines)}
        self._plant_job = None
        self._inner = None
        self._own = False
        self.level = None
        self.clock = None
        self.records: List[Dict[str, Any]] = []
        self._cur: Optional[Dict[str, Any]] = None
        self.host = dict(issue_s=0.0, verify_s=0.0, release_s=0.0, account_s=0.0, snap_wait_s=0.0,
                         submit_wait_s=0.0)
        self.issue = dict(snaps=0, seen=0, unsafe=0, dup_epoch=0, sent=0, epochs=0, uncommitted=0)

    # ------------------------------------------------------------------------------------------------ the binding
    def attach(self) -> None:
        raise SpecPrefetchError("SpecPrefetch binds to a SpecRunner (bind), not to the shipped decode_update")

    def begin_step(self, capture: bool = False) -> None:
        raise SpecPrefetchError("SpecPrefetch has no begin_step: the round runner calls on_draft / on_verify")

    def prefetch(self, items) -> int:
        raise SpecPrefetchError("SpecPrefetch takes the draft misses through on_draft")

    def reset_stats(self) -> None:
        super().reset_stats()
        self._probe_i = 0

    def bind(self, runner) -> None:
        """Called by SpecRunner.__init__ (after its DraftResident): the outermost hook on the body's decode_fixed."""
        if self.runner is not None:
            raise SpecPrefetchError("bound already")
        if runner.rs is not self.rs:
            raise SpecPrefetchError("the runner's RaggedState is not this prefetch's")
        if runner.ops.level not in LEVELS:
            raise SpecPrefetchError("level %r: the prefetch is wired for %s (on_verify)" % (runner.ops.level, LEVELS))
        if self.neg == "commit_mid_group" and runner.ops.level != "L3":
            raise SpecPrefetchError("commit_mid_group is an L3 control")
        body = self.rs.body
        self._own = "decode_fixed" in body.__dict__
        self._inner = body.decode_fixed
        body.decode_fixed = self._hook
        self.runner, self.level, self.clock = runner, runner.ops.level, runner.clock
        self.attached = True                       # PoolPrefetch.set_mode refuses a change while bound

    def unbind(self) -> None:
        if self.runner is None:
            return
        if self.runner.rb.tentative:
            raise SpecPrefetchError("unbind inside a group")
        body = self.rs.body
        if body.__dict__.get("decode_fixed") != self._hook:
            raise SpecPrefetchError("unbind: another hook wraps decode_fixed after the prefetch hook")
        if self._own:
            body.decode_fixed = self._inner
        else:
            del body.decode_fixed
        self.runner = None
        self.attached = False

    def close(self) -> None:
        """Every issued job runs to its end (the transport is drained), then the transport stops. The jobs of the last
        epoch were never committed: counted (uncommitted)."""
        if self.runner is not None:
            raise SpecPrefetchError("close: unbind first (SpecRunner.detach)")
        self.tp.flush()
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self.issue["uncommitted"] += sum(int(r.shape[1]) for j in self.pending for r in j.items.values())
        self.pending = []
        self.tp.close()

    # ------------------------------------------------------------------------------------------------ the seams
    def _rec(self, name):
        r = dict(round=name, E0=None, E1=None, waits=[], jobs_ready=0, jobs_inflight=0, items_ready=0,
                 items_inflight=0, landed=None, last_seq=None, layers=0, acct=None, acct_cs=None, issue=None, host=None)
        self.records.append(r)
        return r

    def on_round(self, phase) -> None:
        if phase == "begin":
            self._cur = self._rec(int(self.runner.round))
            self._probe_poll()
        elif phase == "end":
            t0 = time.perf_counter()
            while self.snaps:                      # L4P: the last draft's snapshot (its job lands during the commit)
                self._issue_snap(self.snaps.pop(0))
            self._probe_poll()
            self.host["issue_s"] += time.perf_counter() - t0

    def on_draft(self, miss_ids) -> None:
        if self.runner is None:
            raise SpecPrefetchError("on_draft while unbound")
        t0 = time.perf_counter()
        if self._cur is None:                      # the L4P prelude (before round 0)
            self._cur = self._rec("prelude")
        if self.rs.ops.graph and any(g is None for g in self.rs.graphs):
            raise SpecPrefetchError("a pooling graph is not captured yet: the C++ worker's CUDA calls would break a "
                                    "capture in the global mode (the warm step captures every layer first)")
        self._probe_poll()
        self.snaps.append(self._snapshot(miss_ids))
        while len(self.snaps) > LAG:
            self._issue_snap(self.snaps.pop(0))
        self.host["issue_s"] += time.perf_counter() - t0

    def on_verify(self) -> None:
        """Outside a group, before the verify (module docstring)."""
        rn = self.runner
        if rn is None:
            raise SpecPrefetchError("on_verify while unbound")
        if rn.rb.tentative:
            raise SpecPrefetchError("on_verify inside a rollback group: the undo would not know the prefetched slots")
        t0 = time.perf_counter()
        if self._cur is None:
            self._cur = self._rec(int(rn.round))
        rec = self._cur
        rec["E0"] = self._clock_ev()
        while self.snaps:
            self._issue_snap(self.snaps.pop(0))
        rn._sync_release()                         # the rollback's device work of the last group comes first
        if self.pending:
            landed = int(self.tp.landed_upto())
            rec["landed"], rec["last_seq"] = landed, int(self.pending[-1].seq)
            job = self._merge(landed, rec)
            rec["layers"] = len(job.items)
            ev = self._record()                    # the last reader of every pool slot and pool table: the main stream
            self.reader_ev = [ev] * self.L
            if self.neg == "commit_mid_group":     # NEGATIVE CONTROL (tests only): the commit runs inside the group
                self._plant_job = job
            else:
                self._commit(job)
                for l in job.items:
                    self._need[l] = self._live[l] = True
            self.pending = []
            self.issue["epochs"] += 1
        self.lane_next.zero_()                     # the next epoch's lanes (its jobs wait the commit's reader marks)
        self.req.zero_()
        self._calls = [0] * self.L
        self._committed = [j for j in self._committed if not ev_done(j.commit_ev)]
        rec["E1"] = self._clock_ev()
        self.host["verify_s"] += time.perf_counter() - t0

    def on_release(self) -> None:
        """Outside a group, after the round's commit and before RequestPin.release (module docstring)."""
        rn = self.runner
        if rn is None or rn.rb.tentative:
            raise SpecPrefetchError("on_release while unbound or inside a group")
        t0 = time.perf_counter()
        P = self.P
        for l, e in enumerate(self.engines):
            if self._need[l]:                      # no verify residency reached layer l: wait its commit here
                self._wait(None, self.commit_ev[l])
                self.commit_ev[l] = None
                self._need[l] = False
            if not self._live[l]:
                continue
            T = self.tags[l][:, :, :P]
            tg = T >= 0
            here = tg & (e._pool_map == self.blk[l])
            self.acct[I_UNREAD].add_(here.sum())
            self.acct[I_MOVED].add_((tg & ~here).sum())
            a = e._pool_age
            a.copy_(torch.where(a >= PIN_BASE + PF_SEQ0, self.age0[l], a))   # every prefetch pin that is left ends
            T.fill_(-1)
            self.delivered[l].zero_()
            self._live[l] = False
        self.host["release_s"] += time.perf_counter() - t0
        if self._cur is not None:
            self._cur["acct"] = self.acct.clone()
            self._cur["acct_cs"] = self.acct_cs.clone()
            self._cur["issue"] = dict(self.issue)              # cumulative: report() takes the per-round deltas
            self._cur["host"] = dict(self.host)

    # ------------------------------------------------------------------------------------------------ issue side
    def _clock_ev(self):
        e = self.clock.event()
        self.clock.record(e)
        return e

    def _snapshot(self, miss_ids):
        """The draft step's skipped blocks: (L, H, B, topk) on the host after its event; and the complete-block bound
        of each request (the round-start host mirror: every block below pos // 64 is on the host rows)."""
        x = torch.stack(list(miss_ids)).to(self._sdt)
        safe = (self.runner.pos_host.to(torch.int64) // self.bs).clone()
        self.issue["snaps"] += 1
        if not self.cuda:
            return (x, None, safe)
        buf = self._sbuf[self._snap_n % (LAG + 1)]
        self._snap_n += 1
        buf.copy_(x, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return (buf, ev, safe)

    def _issue_snap(self, snap) -> None:
        """One snapshot -> one descriptor job (PoolPrefetch._issue): complete blocks only, each (layer, head, request,
        block) at most once per epoch."""
        buf, ev, safe = snap
        if ev is not None:
            t0 = time.perf_counter()
            ev.synchronize()                       # the PREVIOUS draft step (LAG 1): done on the GPU
            self.host["snap_wait_s"] += time.perf_counter() - t0
        l, h, b, k = (buf >= 0).nonzero(as_tuple=True)
        X = buf[l, h, b, k].to(torch.int64)
        n_seen = int(X.numel())
        ok = X < safe[b]
        n_ok = int(ok.sum())
        l, h, b, X = l[ok], h[ok], b[ok], X[ok]
        key = ((l * self.H + h) * self.B + b) * self.NB + X
        fresh = ~self.req[key]
        l, h, b, X, key = l[fresh], h[fresh], b[fresh], X[fresh], key[fresh]
        self.req[key] = True
        self.issue["seen"] += n_seen
        self.issue["unsafe"] += n_seen - n_ok
        self.issue["dup_epoch"] += n_ok - int(key.numel())
        if key.numel() == 0:
            return
        self.busy = [False] * self.L               # the lanes of an epoch only grow: jobs of one layer never collide
        self.issue["sent"] += self._issue(torch.stack([l, h, b, X], 1), commit=True, hist=False)

    def _merge(self, landed: int, rec) -> _EpochJob:
        """The epoch's jobs as ONE job: per layer the items of every job (the lanes are distinct), sorted by (head,
        request) with their group starts, and each item's class (0: its job landed by `landed`; 1: in flight)."""
        per, cls, ncl = {}, {}, [0, 0]
        for job in self.pending:
            c = 0 if int(job.seq) <= landed else 1
            rec["jobs_ready" if c == 0 else "jobs_inflight"] += 1
        for l in range(self.L):
            parts, cl = [], []
            for job in self.pending:
                r = job.items.get(l)
                if r is None:
                    continue
                c = 0 if int(job.seq) <= landed else 1
                parts.append(r)
                cl.append(torch.full((int(r.shape[1]),), c, dtype=torch.int64))
                ncl[c] += int(r.shape[1])
            if not parts:
                continue
            rows = torch.cat(parts, 1)
            c = torch.cat(cl)
            k = rows[0] * self.B + rows[1]
            o = torch.argsort(k, stable=True)
            rows, c, k = rows[:, o].contiguous(), c[o].contiguous(), k[o]
            _, counts = torch.unique_consecutive(k, return_counts=True)
            rows[4] = (torch.cumsum(counts, 0) - counts).repeat_interleave(counts)
            per[l] = rows.pin_memory() if self.cuda else rows
            cls[l] = c.pin_memory() if self.cuda else c
        rec["items_ready"] += ncl[0]
        rec["items_inflight"] += ncl[1]
        return _EpochJob(max(int(j.seq) for j in self.pending), per, cls, ncl)

    def _commit(self, job) -> None:
        """PoolPrefetch._commit, with the host wait of the SUBMISSION timed first (a landed job returns at once)."""
        t0 = time.perf_counter()
        self.tp.job_ev(job.seq).wait_submitted()
        self.host["submit_wait_s"] += time.perf_counter() - t0
        self.tp.poll()                             # a failed job raises here: the device would not wait it
        super()._commit(job)                       # (gilfree_pack.cpp stream_wait_job), so never commit its staging

    def _commit_layer(self, job, l: int):
        """PoolPrefetch._commit_layer (unchanged), then the tags of the slots it wrote (commit stream). The kept item i
        has the age PIN_BASE + s0 + i (PoolPrefetch writes it), so the new slots and their items are known exactly."""
        e, P, NB = self.engines[l], self.P, self.NB
        if not self.skip_reader_wait:
            self._wait(self._cs, self.reader_ev[l])
        age0 = e._pool_age.clone()
        pm0 = e._pool_map.clone()
        s0 = self.pin_seq
        super()._commit_layer(job, l)
        n = self.pin_seq - s0
        a = e._pool_age
        off = a - (PIN_BASE + s0)
        new = (off >= 0) & (off < n)
        cls = job.cls[l].to(self.dev, non_blocking=True)
        T = self.tags[l][:, :, :P]
        T.copy_(torch.where(new, cls[off.clamp(0, n - 1)], T))
        self.age0[l].copy_(torch.where(new, age0, self.age0[l]))
        self.blk[l].copy_(torch.where(new, e._pool_map, self.blk[l]))
        self.delivered[l].scatter_(2, torch.where(new, e._pool_map, torch.full_like(e._pool_map, NB)), True)
        self.delivered[l][:, :, NB] = False        # column NB took every slot that is not new: it stays False
        self.acct_cs[0].add_(new.sum())
        self.acct_cs[1].add_((new & (pm0 >= 0)).sum())
        if self.neg == "no_pin":                   # NEGATIVE CONTROL (tests only): the victim's old age, no pin
            a.copy_(torch.where(new, age0, a))
        ev = self._record(self._cs)                # after the tags: the main stream reads them after its wait
        self.commit_ev[l] = ev
        return ev

    def _probe_poll(self) -> None:
        """PoolPrefetch._probe_poll from the first open probe on (the list grows by one per job)."""
        pl = self._probe_list
        while self._probe_i < len(pl) and pl[self._probe_i][2] is not None:
            self._probe_i += 1
        for p in pl[self._probe_i:]:
            if p[2] is not None or not self._submitted(p[0]):
                continue
            p[3] = int(self.tp.landed_upto()) >= p[0]
            self.tp.job_ev(p[0]).wait(self._ps_arrive)
            p[2] = self._record(self._ps_arrive, timing=True)

    # ------------------------------------------------------------------------------------------------ the hook
    def _hook(self, eng, *args, **kwargs):
        """The outermost hook of the body's decode_fixed: the layer's commit wait, then (in the verify bracket of an
        epoch) the accounting of the residency."""
        l = self._lay[eng._k_gpu.data_ptr()]
        rn = self.runner
        t0 = time.perf_counter()
        if self._plant_job is not None and l == 0 and rn.meter.on and self._calls[0] == rn.K - 1:
            job, self._plant_job = self._plant_job, None       # NEGATIVE CONTROL commit_mid_group: inside the group
            self._commit(job)
            for k in job.items:
                self._need[k] = self._live[k] = True
        if self._need[l]:
            self._need[l] = False
            A = self._clock_ev()
            ev = self.commit_ev[l]
            if ev is not None:
                self._wait(None, ev)                           # THE IN FLIGHT WAIT of layer l (a device wait)
                self.commit_ev[l] = None
            self._cur["waits"].append((A, self._clock_ev()))
        acct = rn.meter.on                                     # EVERY verify residency (all layers, every round)
        vm, age0, i = None, None, 0
        if acct:
            i = self._calls[l]
            self._calls[l] = i + 1
            if self.level == "L4P":
                rows = rn.draft.rows                           # (R, B): the draft requests of each call of the pass
                if rows is not None:
                    vm = (~rows[i]).view(1, self.B, 1)
            age0 = eng._pool_age.clone()
        self.host["account_s"] += time.perf_counter() - t0
        out = self._inner(eng, *args, **kwargs)
        if acct:
            t1 = time.perf_counter()
            self._account(l, eng, age0, vm)
            self.host["account_s"] += time.perf_counter() - t1
        return out

    def _account(self, l: int, eng, age0, vm) -> None:
        """One verify residency of layer l (module docstring, THE TAGS). vm (1, B, 1): the verify requests (None: all).
        Device ops only."""
        P, NB, H, B, M = self.P, self.NB, self.H, self.B, self.topk
        a, t, lm = eng._pool_action, eng._pool_target, eng._load_mask
        T = self.tags[l]
        tq = torch.where(t >= 0, t, P)
        cls = T.gather(2, tq)
        tagged = cls >= 0
        hitk = (a == POOL_SWAP) | (a == POOL_MOVE_IN)
        mo = a == POOL_MOVE_OUT
        hit = hitk & tagged
        evc = mo & tagged
        clm = self._clm
        clm.zero_()
        clm.scatter_(2, torch.where(hitk, t, P), True)
        unp = ((age0 < PIN_BASE) & ~clm[:, :, :P]).sum(2, keepdim=True)
        ovf = mo.sum(2, keepdim=True) > unp                    # the row had too few unpinned unclaimed slots
        if vm is None:
            code = torch.where(hit, cls + I_READ0, I_DUMMY)
        else:
            code = torch.where(hit & vm, cls + I_READ0, torch.where(hit, I_READ_DRAFT, I_DUMMY))
        code = torch.where(evc, torch.where(ovf, I_EV_OVF, I_EV_FREE), code)
        self.acct.scatter_add_(0, code.reshape(-1), self._ones)
        T.scatter_(2, torch.where(hit | evc, tq, P), -1)        # a touched slot loses its tag (column P stays -1)
        newm = lm >= 0
        if vm is not None:
            newm = newm & vm
        self.acct[I_NEW].add_(newm.sum())
        self.acct[I_REFETCH].add_((self.delivered[l].gather(2, torch.where(newm, lm, NB)) & newm).sum())
        if vm is None:
            self.acct[I_SEL].add_(H * B * M)
        else:
            self.acct[I_SEL].add_(vm.sum() * (H * M))

    # ------------------------------------------------------------------------------------------------ the report
    def pins_live_prefetch(self) -> int:
        return sum(int((e._pool_age >= PIN_BASE + PF_SEQ0).sum()) for e in self.engines)

    def report(self) -> Dict[str, Any]:
        """ONE synchronize. PoolPrefetch.report (counters requested / committed / dup_dropped / commit_overflow, the
        host statistics, the delivery probes, the worker's stats), plus the per-round split and waits, the totals and
        the conservation checks."""
        base = super().report()
        c = base["counters"]
        c["conserved"] = None                      # PoolPrefetch's identity counts hits in its own hook (not used here)
        el = self.clock.elapsed if self.clock is not None else None
        tot = dict(zip(ACCT, (int(x) for x in self.acct[:len(ACCT)].tolist())))
        tcs = dict(zip(ACCT_CS, (int(x) for x in self.acct_cs.tolist())))
        rounds, prev, prev_cs = [], [0] * len(ACCT), [0] * len(ACCT_CS)
        prev_is, prev_h = {k: 0 for k in self.issue}, {k: 0.0 for k in self.host}
        for r in self.records:
            out = dict(round=r["round"], jobs_ready=r["jobs_ready"], jobs_inflight=r["jobs_inflight"],
                       items_ready=r["items_ready"], items_inflight=r["items_inflight"], landed=r["landed"],
                       last_seq=r["last_seq"])
            if r["acct"] is not None:
                cur = [int(x) for x in r["acct"][:len(ACCT)].tolist()]
                cur_cs = [int(x) for x in r["acct_cs"].tolist()]
                d = dict(zip(ACCT, (x - y for x, y in zip(cur, prev))))
                d.update(zip(ACCT_CS, (x - y for x, y in zip(cur_cs, prev_cs))))
                d["ready_resident"] = d["sel"] - d["new"] - d["read_ready"] - d["read_inflight"]
                out["split"] = d
                prev, prev_cs = cur, cur_cs
            if r["issue"] is not None:
                out["issue"] = {k: r["issue"][k] - prev_is[k] for k in r["issue"]}
                out["host_ms"] = {k: round(1e3 * (r["host"][k] - prev_h[k]), 4) for k in r["host"]}
                prev_is, prev_h = r["issue"], r["host"]
            if el is not None:
                w = [el(a, b) for a, b in r["waits"]]
                out["dev_wait_ms"] = round(sum(w), 4) if w else 0.0
                out["dev_wait_max_ms"] = round(max(w), 4) if w else 0.0
                out["stall_ms"] = (round(el(r["E0"], r["E1"]), 4) if r["E0"] is not None and r["E1"] is not None
                                   else None)
            rounds.append(out)
        lost = tot["read_ready"] + tot["read_inflight"] + tot["read_draft"] + tot["evict_overflow"] + \
            tot["evict_free"] + tot["unread"] + tot["moved"]
        live_tags = sum(int((T[:, :, :self.P] >= 0).sum()) for T in self.tags)
        cons = dict(tags_committed=tcs["committed"], tags_ended=lost,
                    tags_equal_committed=tcs["committed"] == c["committed"],
                    requested_identity=c["requested"] == c["committed"] + c["dup_dropped"] + c["commit_overflow"],
                    live_tags=live_tags, pins_live_prefetch=self.pins_live_prefetch())
        cons["conserved"] = (cons["tags_committed"] == lost and cons["tags_equal_committed"]
                             and cons["requested_identity"] and live_tags == 0 and cons["pins_live_prefetch"] == 0)
        iss = dict(self.issue)                     # the jobs of the last epoch (never committed) count before close()
        iss["uncommitted"] += sum(int(r.shape[1]) for j in self.pending for r in j.items.values())
        base.update(spec=dict(level=self.level, neg=self.neg, totals=tot, totals_cs=tcs, rounds=rounds,
                              conservation=cons, host_s={k: round(v, 4) for k, v in self.host.items()},
                              issue=iss, pin_overflow=c["commit_overflow"] + tot["evict_overflow"],
                              ready_resident=tot["sel"] - tot["new"] - tot["read_ready"] - tot["read_inflight"],
                              item_bytes=self.item_bytes, lanes=self.lanes))
        return base
