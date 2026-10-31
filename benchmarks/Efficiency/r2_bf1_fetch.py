"""BF1 (AREA 2 of the RECON7 follow-up, KS/analysis/plan/DESIGN_BGFETCH_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
USEFUL BACKGROUND FETCH AND RETENTION for the best specKV arm. ARM L4PTFDGUDKWCDB = L4PTFDGUDKWCD (rung I4DC: the kept
store + window loads from it, graphs, compiled helpers) + the draft-issued background fetch below. Control arm
L4PTFDGUDKWCDX = the same classes and store size with the fetch OFF. OPT-IN: benchmarks/Efficiency/r2_bf1_cell.py; no nosi
module and no older cell imports this file. NO NEW KERNEL: torch ops (index_select on pinned host rows, non_blocking
copies, index_copy_, index_put_, sort), CUDA streams and events. This file imports torch, numpy and the I4DC compile
module only.

REUSE (one line, the rule of the lead): upstream basis = our nosi/nosi/pool_prefetch.py (Task 1.2, ea05c6c; LMCache's
"pinned staging -> copy engine -> device scatter" pattern, KS/analysis/plan/RECON7_R4_prior_art.md section 5.5);
incompatibility = it commits into the engine's victim POOL with a soft pin (no room at C 73, P 10: WS-E dropped 28.2 %
for no slot, SCORING_wse.md:227-230; and a pool commit changes the drafts' pool hits, so the drafts and the acceptance
would change); minimal adaptation = the same staging / copy / scatter path, the commit target is the KW arm's KEPT STORE
(r2_i4d_keep.py docstring 2: a cache keyed (layer, block) that the drafts never read and that J already looks up for
both NEW paths); the packer = torch.index_select into pinned staging (the gilfree v3 descriptor format is lane-per-
(layer, head) for pool commits: a larger fork); validation = the EQF MUST (tokens, fingerprints, state = L4PTFD's) and
the keep's fresh-fetch byte checks (kcheck) in the miniature.

THE PROBLEM (LEDGER_rev7.md row 6). J loads every NEW host block urgently (zero-copy SM gathers on the compute stream):
20.25 ms (C 128 B 216) / 115.64 ms (C 73 B 318, best arm) per round EXPOSED (RECON7_R3 table 2). X1 (R/recon7_x1r3_2195630_r7s):
a copy-engine copy overlaps the GEMMs and the fork (interference 0.8-3.4 %), but beside the urgent gathers it shares
the link (+12.3 %). So the bytes must be moved EARLY and must REPLACE urgent loads.

THE MECHANISM (per round r, for J(r + 1); g = the host joint-pass index T.kpass after J(r))
  1. SNAPSHOT (compute stream, eager, in the draft residency TentStateFD._resid_lean, which DG replays eagerly between
     the graphs A(l) and B(l)): ids = lm where the row is a DRAFT row, the residency skipped the load and the block is a
     complete host block (lm < qc // 64), else -1; one int32 copy into a device ring entry, one non_blocking D2H into
     its pinned twin, one event. No host read.
  2. HOST WORKER (a thread; "inline" on the CPU tests): wait the event; the items (head, request, block); drop the items
     this epoch already sent and the items that the kept store holds ALIVE (a D2H snapshot of kid / kage after J(r):
     kage >= g L; only this mechanism writes the map between two joint passes and it never evicts such a slot, so the
     drop is exact; a dead key the store holds is not sent either); THE ROOM RULE (store_room: the device commit's own
     eligibility, counted on the snapshot and decremented per sent item): an item past its (head, request)'s eligible
     slots is dropped BEFORE packing (bf1m at X 129 copied 63 % no-room bytes before this rule), so device no_room and
     revived are 0 by construction (a MUST of the cell's BF gate); cap n_cap; rank per (head, request); pack 64 K and
     64 V rows (256 B each) per item with
     torch.index_select into a pinned staging entry.
  3. ISSUE (one side stream, under the lock): wait the snapshot event (so it starts after J(r) on the device), the
     non_blocking H2D copies of the rows and the item table into the device landing entry, then THE COMMIT on the same
     stream: key = layer x 2^32 + block; a key that the map holds -> REVIVE (kage refreshed, no bytes); else a slot of
     (head, request) that is EMPTY or DEAD (kage < g L, i.e. not used in J(r) and not committed this epoch), oldest
     first; the rows by index_copy_ into the arena store, the bias rows from the twin's total_cis on the device; kid =
     key, kage = g L + l, the prefetch tag. No room -> the item's rows go to store slot 0 of its request (CB1: the chain
     image rewrites it at every joint layer before any read) and count as no_room.
  4. BARRIER (J(r + 1) start, before the pass): the epoch CLOSES under the lock (a job not yet issued is abandoned:
     "late"); the compute stream waits the last issued commit. J then runs unchanged: keep_assign finds hits, kwin
     serves row 0's window loads from the store, the urgent gathers fetch only the rest. No prefetch copy runs beside
     J's urgent gathers.
  5. ACCOUNTING (after the pass, eager): a tagged slot that J used refreshed its age (keep_assign: kage >= (g + 1) L)
     and still holds its key -> used (store path); acct=True also counts the kwin (window) uses in the joint residency.
     Then a new epoch: the kid / kage snapshot (non_blocking D2H + event).
GENERATIONS / LIFETIMES. Store and map: written only by the side stream between J(r) and J(r + 1) (the drafts never
read them; the round's commit and reset do not touch them); every side job waits a draft snapshot event recorded after
J(r); J(r + 1) waits the last issued commit. Landing entry: device copies and its commit are in ONE stream order, so the
next copy into an entry follows the previous commit. Pinned staging entry: the worker waits its previous H2D event before
it packs again. Snapshot entry: the main thread reuses it only after the worker released it. Map snapshot: one per epoch
parity. A stale block cannot be read: the items are complete committed blocks (< qc // 64 at the snapshot; qc never
decreases; the block's host rows were written back before the snapshot in stream order) and the slot's key is written
in the same stream step as its bytes.
PACKER (packer="gilfree", the production switch; "torch" = index_select, the CPU tests' default). Reuse: our
nosi/nosi/r2_transport_gilfree_v3.GilfreeTransport (packer v3 benchmarks/Efficiency/gilfree_pack.{py,cpp}: an OpenMP
team packs with the GIL released, the C++ worker thread copies on its copy stream and scatters on its scatter stream;
MEASURED 21-30 GB/s pack at 8-16 threads, memory note project_retroinfer_eval.md line 158; PF1: a 64 MiB job lands in
11.2-11.6 ms), exactly as pool_prefetch.py builds it (worker "cpp", issue_desc, the flat lane store). Incompatibility:
pool_prefetch gives each LAYER its own lanes (L x lanes x 64 KB: 4.3 GB at our n_cap); adaptation: every layer's
destination is the ONE flat landing ring, a take's lane = ring entry x n_cap + its rank in its head. T3: a lane's previous
reader is the commit of the entry's last use (mark_reader of the commit's done event; reader_marks per job); T4: the
worker host-waited the snapshot event (after the write-back in stream order), so hist_mark -1. The commit reads the lane
rows (lane x 64 + r) x H + h and is otherwise the same. Before every graph capture the transport is drained under the cap
lock (its C++ worker makes CUDA calls: pool_prefetch.begin_step(capture=True)'s rule). The SAME byte checks apply.
LATE (late="defer", the production default of the cell; "drop" = abandon at the barrier, bf1m m2's behaviour). THE
THROUGHPUT THE C 73 LEVER NEEDS (MODEL on MEASURED inputs: r7pb NEW 2.93-3.01 GB per joint pass, flagA 0.469; bf1m m2:
used / sent 0.80, gilfree 8 threads 4.65 GB/s native, every late job of the last draft pass): sent per round about
0.469 x 2.97 / 0.80 = 1.74 GB. With "drop" every byte must be packed and copied between the first draft snapshot and
the joint pass's start: the draft phase 176.4 ms + commit / reset 8.5 ms = 185 ms -> 9.4 GB/s sustained pack (2x the
measured 4.65; the link's 25 GB/s is not the limit: 70 ms). With "defer" the job of layer l is needed only when the
joint pass reaches layer l (about l / 32 x 339 ms after its start): the binding point is D1's layer 0, cumulative
about 0.87 GB (all of D0 + one job) by 185 ms = 4.7 GB/s; the last job (1.74 GB by 185 + 329 ms) needs 3.4 GB/s. So
"defer" lowers the need to about the measured rate; 14 threads and 1024-group chunks (less per-chunk overhead than the
256 of m2) are the cheap rate lever. DEFER, the mechanism: at the joint pass's start the epoch is CLOSING, not closed:
a job issued after it lands (H2D on the side stream; gilfree: the side stream waits the transport job) and waits in
ep.deferred[l]; the joint pass's eager residency of layer l (TentStateBF._resid_joint -> on_joint_layer) first waits on
the host for the queued jobs of layers <= l (within late_wait_ms per pass: the host runs ahead of the GPU by the queued
layers' time), then commits EVERY landed job of layers >= l ON THE COMPUTE STREAM (the map is shared by the layers: no other
stream may write it while the pass runs; slots that the pass's keep used are alive, kage >= (g + 1) L > g L, so the
commit never takes them); a job that lands after its layer ran is dropped ("passed"). A deferred job holds its landing
entry until committed (land_busy; torch packer: the next copy into the entry waits the commit's event; gilfree: its
reader mark). The deferred commits' no-room items count apart (no_room_deferred: the pass's keep may take dead slots
that the host counted as room). The epoch closes at the pass's end; leftovers are dropped.
BATCH (bf1b, f2eba43, C 73 B 318, MEASURED: worker busy 249 ms per epoch > the 185 ms window; per ISSUED job 14.3 ms =
C++ submit 3.74 (2.84 chunks x 1.32 ms) + C++ pack 1.02 + host select and glue 1.41 + side-stream issue and commit
4.33 (about 90 launches) + snapshot and other waits; only 17.5 of 64 jobs per epoch issued). HYPOTHESIS (not yet
measured apart): the cost is PER CALL, not per byte (pack 18.9 GB/s native; 1.32 ms per chunk of about 7 CUDA calls from
the worker while the main thread launches; 4.33 ms per commit of about 90 launches). The report's submit_ms_per_chunk
decides it: per call it stays about 1.3 ms at 2048-group chunks; per byte (about 5 GB/s) it grows to about 4 ms. The
lever is fewer calls per epoch: the worker takes every queued job of the epoch up to `batch` (a job that does not fit
the landing entry goes first into the next batch), and makes ONE pack, ONE release and ONE commit per batch. Custom work:
_release calls NativePipe.submit (packer v3, the transport's own _submit path) with all layers of the batch in one job,
not issue_desc (upstream: our r2_transport_gilfree_v3 at the pinned sha; incompatibility: release_desc makes one release,
>= 1 chunk, per layer; adaptation: every layer's destination is the ONE flat landing, so one release holds every layer,
source l' = layer x B + b; validation: test_bf1_gilfree EQF, the byte checks, the plant).
PIPELINE (bf1b b2, c10bddf, C 73 B 318: worker 375 ms per epoch > the 185 ms window; 59 % of the jobs committed in the
joint pass; x0.953. Of it the Python stages snap 83 + select 52 + issue 52 and the C++ stage pack 93 + submit 60 ran in
SERIES). The worker now runs batch k's select and submission (_prep) BEFORE batch k-1's issue (_finish: the submission
wait, the landing, the commit), so the C++ job of k-1 packs and copies (GIL released) while Python selects k. MODEL:
max(187, 153) + the exposed wait = about 200 ms per epoch (lead's calibrated model: gain about +20 ms per round). Order
and lifetimes: one batch in flight; batch k uses entry k % ring != entry k - 1 (ring >= 2), and its T3 mark / staging
wait name batch k - ring, finished before; the jobs of k - 1 leave the flight (pend: the DEFER wait) at its finish; with
nothing queued the worker finishes k - 1 before it blocks. Counters: pipelined (a batch selected while k - 1 was not
finished), overlapped (and k - 1's C++ job not yet submitted at k's select: real overlap), cxx_wait_s.
COMPILE (bf1b, L4PTFDGUDKWCDX-COMPILE: ONE late event, loc2 at joint pass 0 = dpass 3, after_captures 64, 1.395 s). The
arm's first round runs its draft passes (DG captures at dpass 2) BEFORE its first joint pass, and loc2 specializes on
the store X (its Python int cap = X - 1 and lst_buf (H, B, X)); the C arm compiled X 129 only, so at --bf-store 257 the
first joint call recompiled after the captures. Fix: TentStateBF.bf_warm_joint, called by the runner's constructor
(before any capture), runs pre, loc1 and loc2 once on the state's own buffers (a selection of -1; vs.ti and vs.lst_buf
restored): the compile event is before the captures (after_captures 0), the joint pass hits the cache. Reuse: the
section's own call (_cc_locate) argument for argument; validation: the COMPILE MUST (<= 1 event per helper, none late).
CUDA GRAPHS. Nothing here enters a graph: every op is eager, between graph launches. A capture in global mode fails when
another thread calls the CUDA API, so the runner wraps the graph backend's begin / capture: the capture holds the lock
that the worker takes around every CUDA call.
EXACTNESS. BF1 writes only the arena store, kid, kage and its own tags. The fork reads pages in block-id order wherever
they live (verify_rows._fork_lists) and a store slot holds the host block's bytes: J's outputs, the drafts, the plan, the
engine state are L4PTFDGUDKWCD's = L4PTFD's bit for bit (the EQF MUST).
NEGATIVE CONTROLS (tests only; bneg): "wrong_bytes" (the rows of block + 1 packed under the key of block), "no_copy" (the
map written, the rows not copied). Both must fail EQF.
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
import traceback

import numpy as np
import torch

import r2_i4dc_compile as CD

KM = 1 << 32
BIGI = 1 << 62
MODES = ("ce", "off")
WORKERS = ("thread", "inline")
PACKERS = ("torch", "gilfree")          # the host pack: torch.index_select (pinned) | r2_transport_gilfree_v3 (C++ team)
LATES = ("drop", "defer")               # a job not issued at the joint pass's start: abandoned | committed at ITS layer
BNEG = (None, "wrong_bytes", "no_copy")
# device counters: the side stream's (one tensor) and the compute stream's (another tensor): no tensor has two writers
SIDE_KEYS = ("items", "committed", "revived", "no_room", "no_room_deferred", "passed_items")
MAIN_KEYS = ("used_store", "unused", "used_win")


class BFError(RuntimeError):
    pass


class _PlanVS:
    """The one plan field that _cc_why reads (plan.vs): bf_warm_joint asks the section's own fallback rule."""
    __slots__ = ("vs",)

    def __init__(self, vs):
        self.vs = vs


class _NoEvent:
    """A completed event (CPU)."""

    def record(self, stream=None):
        return None

    def synchronize(self):
        return None

    def query(self):
        return True

    def elapsed_time(self, other):
        return 0.0


class _LaneMap:
    """The flat-store geometry that GilfreeTransport's xpool option reads (as nosi/nosi/pool_prefetch._Lanes): M = A
    lanes of 64 rows, B, H; a descriptor's lane is its slot row (row 4)."""

    def __init__(self, M: int, B: int, H: int):
        self.M = self.A = int(M)
        self.B, self.H = int(B), int(H)

    def lanes_of(self, desc):
        return desc[4].clone()


class _Epoch:
    """The jobs between two joint passes (attributes only)."""

    def __init__(self, seq: int, g: int, par: int, ev_map):
        self.seq, self.g, self.par, self.ev_map = int(seq), int(g), int(par), ev_map
        self.closed = False
        self.last = None                                 # the last issued commit's event
        self.have = None                                 # host bitmap (L, H, B, NBLK): sent or held by the store
        self.free = None                                 # host (H, B): the commit's eligible slots left (ROOM)
        self.free0 = None                                # the same at the epoch's start (the report)
        self.jobs_at_close = None
        self.closing = False                             # defer: the joint pass runs; jobs commit at THEIR layer
        self.consumed_upto = -1                          # defer: the joint pass's layers <= this were committed
        self.deferred = {}                               # defer: layer -> [(entry, n, ready event)] landed, waiting
        self.pend = {}                                   # layer -> jobs queued and not finished (thread worker)
        self.jobs = self.issued = self.late = 0
        self.items_seen = self.items_have = self.items_cap = self.items_sent = self.items_room = 0
        self.t_close = None                              # host clock at the close (the late jobs' age)


def rows_of(h, b, blk, T: int, H: int, bs: int):
    """Host row indices (n * bs,) of the items in a (B * T * H, D) view of the host rows (B, T, H, D): item-major, row r of
    block blk of request b, head h = ((b T + blk bs + r) H + h)."""
    r = np.arange(bs, dtype=np.int64)
    pos = (b.astype(np.int64) * T + blk.astype(np.int64) * bs)[:, None] + r[None, :]
    return (pos * H + h.astype(np.int64)[:, None]).reshape(-1)


def group_rank(h, b, B: int):
    """Rank of each item inside its (head, request) group; the items come sorted by (h, b) (np.nonzero's C order)."""
    n = h.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    grp = h.astype(np.int64) * B + b.astype(np.int64)
    start = np.ones(n, dtype=bool)
    start[1:] = grp[1:] != grp[:-1]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    return np.arange(n, dtype=np.int64) - first


def store_room(kid_h, kage_h, g: int, L: int, L_valid: int, nblk: int, block=None):
    """THE ROOM RULE on the host (the device commit's own rule, _commit): (free (H, B): the slots 1 .. X - 1 that are
    EMPTY or DEAD (kage < g L), minus the test-only blocked slots; and (l, h, b, blk): EVERY key the map holds, alive or
    dead, so that a block the store holds is never sent again). Only BF1's commits write the map between two joint
    passes, they take eligible slots only and make them alive, so the device finds exactly free - (items sent before in
    the epoch) eligible slots: an item past its (head, request)'s room is dropped HERE, before any byte is packed."""
    k1, a1 = kid_h[..., 1:], kage_h[..., 1:]
    el = (k1 < 0) | (a1 < int(g) * int(L))
    if block is not None:
        el = el & ~block
    h, b, _ = np.nonzero(k1 >= 0)
    key = k1[k1 >= 0]
    l = key >> 32
    blk = key & (KM - 1)
    ok = (l >= 0) & (l < L_valid) & (blk < nblk)
    return el.sum(-1).astype(np.int64), (l[ok], h[ok], b[ok], blk[ok])


def kept_alive(kid_h, kage_h, g: int, L: int, nblk: int):
    """(l, h, b, blk) arrays of the keys that the map holds in slots used at the last joint pass or later (kage >= g L):
    no BF1 commit evicts them before J(g + 1) visits their layer."""
    k1, a1 = kid_h[..., 1:], kage_h[..., 1:]
    m = (k1 >= 0) & (a1 >= int(g) * int(L))
    h, b, _ = np.nonzero(m)
    key = k1[m]
    l = key >> 32
    blk = key & (KM - 1)
    ok = (l >= 0) & (l < L) & (blk < nblk)
    return l[ok], h[ok], b[ok], blk[ok]


class BgFetch:
    """The mechanism of the module docstring for ONE tentative state T (a keep state: T.kid, T.kage, T.arena)."""

    def __init__(self, T, mode: str = "ce", n_cap: int = 1024, ring: int = 8, worker: str = "thread",
                 acct: bool = False, bneg=None, close_wait_ms: float = 0.0, packer: str = "torch", threads: int = 8,
                 gf=None, late: str = "drop", late_wait_ms: float = 100.0, stall_ms: float = 0.0, batch: int = 8,
                 hold: int = 0):
        if (mode not in MODES or worker not in WORKERS or bneg not in BNEG or packer not in PACKERS
                or late not in LATES):
            raise BFError("mode %r / worker %r / bneg %r / packer %r / late %r: one of %s / %s / %s / %s / %s" % (
                mode, worker, bneg, packer, late, MODES, WORKERS, BNEG, PACKERS, LATES))
        if float(late_wait_ms) < 0 or float(stall_ms) < 0 or int(batch) < 1 or int(hold) < 0:
            raise BFError("late_wait_ms >= 0, stall_ms >= 0, batch >= 1, hold >= 0")
        self.batch = int(batch)                          # jobs per worker batch (one pack / transport job / commit)
        self.hold = int(hold)                            # tests only: before an epoch's first batch the worker waits for
        self._held_ep = None                             # `hold` queued jobs (or the epoch's close): a CERTAIN backlog
        self.packer, self.threads, self.gf = packer, int(threads), dict(gf or {})
        self.late_mode, self.late_wait_ms = late, float(late_wait_ms)
        self.stall_ms = float(stall_ms)                  # tests only: the worker sleeps per job (forces late jobs)
        if getattr(T, "kid", None) is None or not getattr(T, "kwin", False):
            raise BFError("BF1 needs the kept store with window loads (keep=True, kwin=True)")
        if int(n_cap) < 1 or int(ring) < 2 or float(close_wait_ms) < 0:
            raise BFError("n_cap >= 1, ring >= 2, close_wait_ms >= 0")
        self.T, self.mode, self.worker, self.acct, self.bneg = T, mode, worker, bool(acct), bneg
        self.close_wait_ms = float(close_wait_ms)
        self.inflight = 0                                # thread worker: jobs queued and not finished
        rs = T.rs
        self.L, self.H, self.B, self.bs, self.D, self.topk = T.L, T.H, T.B, T.bs, T.D, T.topk
        self.X = int(T.arena.X)
        self.dev = T.dev
        self.cuda = self.dev.type == "cuda"
        self.engs = [lay.cache_engine for lay in rs.cache.layers]
        self.twins = rs.st.twins
        a = T.arena
        self.sk = a.store_k.view(-1, self.D)            # (B X bs H, D): row ((b X + x) bs + r) H + h
        self.sv = a.store_v.view(-1, self.D)
        self.sb = a.store_bias.reshape(-1)
        if not (a.store_k.is_contiguous() and a.store_v.is_contiguous() and a.store_bias.is_contiguous()):
            raise BFError("the arena store views must be contiguous")
        self.n_cap, self.ring = int(n_cap), int(ring)
        self.lanes_e = max(1, self.n_cap // self.H)     # gilfree: lanes per entry (a lane holds one block per head)
        self.nsnap = 4 * self.L
        self.lock = threading.RLock()                   # the epoch state (both threads; never held across a wait)
        self.cap_lock = threading.RLock()               # the worker's CUDA calls vs a graph capture (CUDA GRAPHS)
        self.cv = threading.Condition()
        self.q = queue.Queue()
        self.thread = None
        self.error = None
        self._seq = 0
        self.epoch = None
        self.rep = dict(epochs=0, jobs=0, issued=0, late=0, items_seen=0, items_have=0, items_cap=0, items_sent=0,
                        pack_s=0.0, issue_s=0.0, snap_stall=0, captures_locked=0, barrier_waits=0, empty_jobs=0)
        self.per_epoch = []
        self.h2d_ev = []                                 # (start, end, bytes) of every issued job
        self.bar_ev = []                                 # (epoch seq, start, end) of every barrier wait
        self.side = None
        self.tp = None                                   # the gilfree transport (packer "gilfree", mode "ce" only)
        self.late_ages = []                              # ms from enqueue to the close, per late job
        self.jep = None                                  # defer: the epoch the running joint pass consumes
        self.jwait_left = 0.0                            # defer: the host wait budget left in this joint pass (s)
        kb = getattr(T, "kblock", None)                  # the keep's test-only slot limit, on the host (ROOM)
        self.kblock_h = kb.cpu().numpy() if kb is not None else None
        self._alloc()

    # ---------------------------------------------------------- buffers
    def _ev(self, timing=False):
        return torch.cuda.Event(enable_timing=timing) if self.cuda else _NoEvent()

    def _pinned(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.cuda)

    def _alloc(self):
        H, B, L, X, D, bs, dev = self.H, self.B, self.L, self.X, self.D, self.bs, self.dev
        e0 = self.engs[0]
        self.Tcpu = int(e0._k_cpu.shape[1])
        self.nblk = (self.Tcpu + bs - 1) // bs
        self.kdt = e0._k_cpu.dtype
        self.item_bytes = 2 * bs * D * e0._k_cpu.element_size()      # the PCIe bytes of one (head, block): K + V
        i64 = torch.int64
        self.m1 = torch.tensor(-1, dtype=i64, device=dev)
        self.ar_bs = torch.arange(bs, dtype=i64, device=dev)
        self.ptag = torch.zeros((H, B, X), dtype=torch.bool, device=dev)
        self.pkey = torch.full((H, B, X), -1, dtype=i64, device=dev)
        self.ptagw = torch.zeros((H, B, X), dtype=torch.bool, device=dev)
        self.cnt_side = torch.zeros((len(SIDE_KEYS),), dtype=i64, device=dev)
        self.cnt_main = torch.zeros((len(MAIN_KEYS),), dtype=i64, device=dev)
        if self.mode == "off":
            return
        self.side = torch.cuda.Stream(device=dev) if self.cuda else None
        self.snap_d = torch.full((self.nsnap, H, B, self.topk), -1, dtype=torch.int32, device=dev)
        self.snap_h = self._pinned((self.nsnap, H, B, self.topk), torch.int32)
        self.snap_ev = [self._ev() for _ in range(self.nsnap)]
        self.snap_busy = [False] * self.nsnap
        self.snap_i = 0
        n = self.n_cap * bs
        self.land_it = torch.zeros((self.ring, self.n_cap, 6), dtype=i64, device=dev)   # h, b, layer, block, rk, lane
        self.stage_it = self._pinned((self.ring, self.n_cap, 6), i64)
        self.h2d_done = [None] * self.ring               # the event of the entry's last H2D (the staging lifetime)
        self.land_busy = [False] * self.ring             # defer: the entry holds a job that waits for its layer
        self.land_reader = [None] * self.ring            # torch packer: the entry's last reader on the compute stream
        self.land_i = 0
        self.tp = None
        if self.packer == "torch":                       # landing (ring, n_cap x 64 rows, D): item-major rows
            self.land_k = torch.empty((self.ring, n, D), dtype=self.kdt, device=dev)
            self.land_v = torch.empty((self.ring, n, D), dtype=self.kdt, device=dev)
            self.stage_k = self._pinned((self.ring, n, D), self.kdt)
            self.stage_v = self._pinned((self.ring, n, D), self.kdt)
        else:                                            # GILFREE (module docstring, PACKER): the flat lanes
            ln = self.ring * self.lanes_e * bs           # the same bytes as the torch landing (n_cap x 64 x H / H)
            self.land_k = torch.zeros((ln, H, D), dtype=self.kdt, device=dev)
            self.land_v = torch.zeros((ln, H, D), dtype=self.kdt, device=dev)
            self.lmark = [-1] * self.ring                # the reader mark (the last commit) of each landing entry
            self.mark_next = 0
            self.tp = self._transport()
        self.kid_h = [self._pinned((H, B, X), i64) for _ in range(2)]
        self.kage_h = [self._pinned((H, B, X), i64) for _ in range(2)]
        self.map_ev = [self._ev(), self._ev()]

    def _transport(self):
        """THE GILFREE PACKER: our r2_transport_gilfree_v3.GilfreeTransport (worker "cpp", descriptor issue, the flat
        lane store of pool_prefetch's use: xpool lanes_of = the descriptor's slot row) with EVERY layer's destination =
        the ONE flat landing (ring x n_cap x 64 rows, H, D); a take (l, b, h, block, lane) writes rows (lane x 64 + r)
        x H + h. gf: ext (the gilfree_pack v3 module; default: built from TORCH_EXTENSIONS_DIR), cls (the transport
        class; default nosi.r2_transport_gilfree_v3), cores, cap_groups."""
        import importlib
        g = self.gf
        ext = g.get("ext")
        if ext is None:
            gp = importlib.import_module("gilfree_pack")
            ext = gp.load(gp.build_dir_from_env())
        cls = g.get("cls") or importlib.import_module("nosi.r2_transport_gilfree_v3").GilfreeTransport
        if int(getattr(ext, "DESC", 0)) < 2:
            raise BFError("packer gilfree: packer v3 with DESC >= 2 (the lane row) is needed")
        L = self.L
        return cls(ext, [e._k_cpu for e in self.engs], [e._v_cpu for e in self.engs], [self.land_k] * L,
                   [self.land_v] * L, self.bs, self.D, self.threads, lambda ev: ev.synchronize(),
                   cap_groups=int(g.get("cap_groups", 256)), ring=2, cores=g.get("cores"), worker="cpp",
                   xpool=_LaneMap(self.ring * self.lanes_e, self.B, self.H))

    def bytes_device(self) -> int:
        """The device bytes BF1 allocated (the landing ring, the snapshot ring, the tags)."""
        if self.mode == "off":
            return sum(t.numel() * t.element_size() for t in (self.ptag, self.pkey, self.ptagw))
        return sum(t.numel() * t.element_size() for t in (self.land_k, self.land_v, self.land_it, self.snap_d,
                                                          self.ptag, self.pkey, self.ptagw))

    # ---------------------------------------------------------- the capture lock (thread worker)
    def wrap_backend(self, be) -> None:
        """Instance attributes over the graph backend's begin / end / capture: a capture holds the lock that the worker
        takes around its CUDA calls (module docstring, CUDA GRAPHS)."""
        if self.mode == "off" or (self.worker != "thread" and self.packer != "gilfree") or be is None:
            return
        ob, oe, oc = getattr(be, "begin", None), getattr(be, "end", None), getattr(be, "capture", None)
        bf = self

        if ob is not None and oe is not None:
            def begin():
                bf.cap_lock.acquire()
                bf.rep["captures_locked"] += 1
                try:
                    bf._drain_transport()
                    return ob()
                except BaseException:
                    bf.cap_lock.release()
                    raise

            def end():
                try:
                    return oe()
                finally:
                    bf.cap_lock.release()
            be.begin, be.end = begin, end
        if oc is not None:
            def capture(fn, T_, vs):
                with bf.cap_lock:
                    bf.rep["captures_locked"] += 1
                    bf._drain_transport()
                    return oc(fn, T_, vs)
            be.capture = capture

    def _drain_transport(self) -> None:
        """Before a capture (the cap lock held, so no new job): the gilfree C++ worker finishes its jobs and idles (its
        CUDA calls must not run during a global-mode capture; pool_prefetch.begin_step(capture=True)'s rule)."""
        if self.tp is not None:
            self.tp.flush()

    # ---------------------------------------------------------- epochs (compute thread)
    def open_epoch(self) -> None:
        """A new epoch after a joint pass (or at the start): the kid / kage snapshot of this parity."""
        if self.mode == "off":
            return
        T = self.T
        par = self._seq % 2
        ev = self.map_ev[par]
        self.kid_h[par].copy_(T.kid, non_blocking=self.cuda)
        self.kage_h[par].copy_(T.kage, non_blocking=self.cuda)
        ev.record()
        with self.lock:
            self.epoch = _Epoch(self._seq, int(T.kpass), par, ev)
            self._seq += 1
            self.rep["epochs"] += 1

    def before_joint(self) -> None:
        """J start: close the epoch, the compute stream waits the last issued commit (the barrier)."""
        if self.error is not None:
            raise BFError("BF1 worker failed: %s" % self.error)
        if self.mode == "off" or self.epoch is None:
            return
        waited = 0.0
        if self.worker == "thread" and self.close_wait_ms > 0:      # a bounded host wait for the last jobs
            t0 = time.perf_counter()
            with self.cv:
                while self.inflight > 0 and self.error is None:
                    left = self.close_wait_ms / 1e3 - (time.perf_counter() - t0)
                    if left <= 0:
                        break
                    self.cv.wait(left)
            waited = time.perf_counter() - t0
        with self.lock:
            ep = self.epoch
            if self.late_mode == "defer":                # LATE = DEFER: the jobs not issued yet commit at their layer
                ep.closing, ep.consumed_upto = True, -1
                self.jep, self.jwait_left = ep, self.late_wait_ms / 1e3
            else:
                ep.closed = True
            ep.t_close = time.perf_counter()
            ep.jobs_at_close = ep.jobs
            last = ep.last
            self.rep["close_wait_s"] = self.rep.get("close_wait_s", 0.0) + waited
            self.per_epoch.append(dict(seq=ep.seq, g=ep.g, jobs=ep.jobs, issued=ep.issued, late=ep.late,
                                       inflight_at_close=self.inflight, items_seen=ep.items_seen,
                                       items_have=ep.items_have, items_room=ep.items_room, items_cap=ep.items_cap,
                                       items_sent=ep.items_sent,
                                       free_mean=(round(float(ep.free0.mean()), 2) if ep.free0 is not None else None),
                                       free_min=(int(ep.free0.min()) if ep.free0 is not None else None)))
        if last is not None and self.cuda:
            a, b = self._ev(True), self._ev(True)
            a.record()
            torch.cuda.current_stream().wait_event(last)
            b.record()
            self.bar_ev.append((ep.seq, a, b))
            self.rep["barrier_waits"] += 1

    def after_joint(self) -> None:
        """J end: the used / unused accounting of the epoch's tags, then a new epoch."""
        if self.mode == "off":
            return
        ep = self.jep
        if ep is not None:                               # defer: the epoch ends with its joint pass
            with self.cv:
                with self.lock:
                    for l in sorted(ep.deferred):        # landed after their layers ran: wasted, entries freed
                        for e, n, _, k, _ in ep.deferred[l]:
                            self.land_busy[e] = False
                            for _ in range(k):
                                self._late(ep, ep.jobs, ep.t_close, "passed", 0)
                            self.rep["items_late"] = self.rep.get("items_late", 0) + n
                    ep.deferred.clear()
                    ep.closed = True
                    self.jep = None
                    self.rep["late_wait_s"] = self.rep.get("late_wait_s", 0.0) + (self.late_wait_ms / 1e3
                                                                                  - self.jwait_left)
                self.cv.notify_all()
        T = self.T
        t1 = int(T.kpass) * self.L
        tg = self.ptag[..., 1:]
        used = tg & (T.kid[..., 1:] == self.pkey[..., 1:]) & (T.kage[..., 1:] >= t1)
        anyu = used | (tg & self.ptagw[..., 1:])
        self.cnt_main[0].add_(used.sum())
        self.cnt_main[1].add_((tg & ~anyu).sum())
        self.cnt_main[2].add_((tg & self.ptagw[..., 1:] & ~used).sum())
        self.ptag.zero_()
        self.ptagw.zero_()
        self.open_epoch()

    def on_joint_layer(self, l: int) -> None:
        """LATE = DEFER, the joint pass's layer l (compute thread, eager, BEFORE its residency reads the map): commit
        every landed job of layers >= l on the COMPUTE stream (serial with keep_assign and kwin: the map is shared by the
        layers, so no other stream may write it while the pass runs); while a job of a layer <= l is still queued, wait
        for it on the host within the pass's budget (the host runs ahead of the GPU by the queued layers' time). A job
        that lands after its layer ran is dropped (late "passed")."""
        ep = self.jep
        if ep is None or self.mode == "off":
            return
        with self.cv:
            while True:
                pend = sum(c for x, c in ep.pend.items() if x <= l)     # read BEFORE the consume: a job that
                with self.lock:                                         # finished before it is consumed now
                    self._consume(ep, l)
                self.cv.notify_all()                     # freed entries: a worker waiting for one goes on
                if pend == 0 or self.jwait_left <= 0 or self.error is not None:
                    break
                t0 = time.perf_counter()
                self.cv.wait(self.jwait_left)
                self.jwait_left -= time.perf_counter() - t0
            with self.lock:
                ep.consumed_upto = l
            self.cv.notify_all()

    def _consume(self, ep, l: int) -> None:
        """Under the cv and the lock: EVERY landed deferred job, committed on the compute stream at layer l: layers >= l
        (a later layer's keys enter early, as the side commits did before the pass; slots of the pass's layers < l are
        alive and never taken; committing all frees every landing entry, so a queued job never waits for an entry that
        only a later layer would free), layers < l dropped (their layer ran)."""
        for x in sorted(ep.deferred):
            for e, n, ready, k, layers in ep.deferred.pop(x):
                if max(layers) < l:                      # every layer of the batch ran
                    for _ in range(k):
                        self._late(ep, ep.jobs, ep.t_close, "passed", 0)
                    self.rep["items_late"] = self.rep.get("items_late", 0) + n
                else:                                    # its items of layers < l are dropped inside (passed_items)
                    if self.cuda:
                        torch.cuda.current_stream().wait_event(ready)
                    itd = self.land_it[e][:n]
                    self._commit(itd, n, ep.g, *self._src(e, n, itd), layers, deferred=True, min_layer=l)
                    done = self._ev()
                    done.record()
                    self._entry_read(e, done)
                    self.rep["deferred_committed"] = self.rep.get("deferred_committed", 0) + k
                    ep.issued += k
                    ep.items_sent += n
                    self.rep["issued"] += k
                    self.rep["items_sent"] += n
                self.land_busy[e] = False

    def _src(self, e: int, n: int, itd):
        """The items' K / V rows (n x 64, D) of landing entry e: the torch packer's item-major rows, or (gilfree) the
        rows (lane x 64 + r) x H + h of each item's lane and head."""
        bs, H, D = self.bs, self.H, self.D
        if self.packer == "torch":
            return self.land_k[e][:n * bs], self.land_v[e][:n * bs]
        rsrc = ((itd[:, 5].view(n, 1) * bs + self.ar_bs.view(1, bs)) * H + itd[:, 0].view(n, 1)).reshape(-1)
        return self.land_k.view(-1, D).index_select(0, rsrc), self.land_v.view(-1, D).index_select(0, rsrc)

    def _entry_read(self, e: int, done) -> None:
        """The landing entry e was read (its commit) up to `done` (under the lock): the next copy into e waits it."""
        if self.packer == "torch":
            self.land_reader[e] = done
        else:
            mk = self.mark_next
            self.mark_next += 1
            self.tp.mark_reader(mk, done)
            self.lmark[e] = mk

    def on_window(self, l: int, eng, skip) -> None:
        """acct=True, after the joint residency of layer l: the row-0 loads that a tagged store slot served (kwin)."""
        if self.mode == "off" or not self.acct:
            return
        lm = eng._load_mask
        ld = (lm >= 0) & ~skip
        key = torch.where(ld, lm + int(l) * KM, torch.full_like(lm, -2))
        E = (key.unsqueeze(-1) == self.pkey[:, :, None, 1:]) & self.ptag[:, :, None, 1:]
        self.ptagw[..., 1:] |= E.any(2)

    # ---------------------------------------------------------- the draft residency hook (compute thread)
    def on_draft(self, l: int, eng, skip, drf, qb3) -> None:
        if self.mode == "off" or self.epoch is None:
            return
        if self.error is not None:
            raise BFError("BF1 worker failed: %s" % self.error)
        B = self.B
        lm = eng._load_mask
        cand = skip & (lm >= 0) & (lm < qb3) & drf.view(1, B, 1)
        s = self._snap_slot()
        self.snap_d[s].copy_(torch.where(cand, lm, self.m1))
        self.snap_h[s].copy_(self.snap_d[s], non_blocking=self.cuda)
        self.snap_ev[s].record()
        with self.lock:
            ep = self.epoch
            ep.jobs += 1
            self.rep["jobs"] += 1
        job = (ep, int(l), s, ep.jobs - 1, time.perf_counter())
        if self.worker == "inline":                      # the CPU tests: the job at once, on the compute stream
            self._batch([job])
        else:
            self._start()
            with self.cv:
                self.inflight += 1
                ep.pend[int(l)] = ep.pend.get(int(l), 0) + 1
            self.q.put(job)

    def _snap_slot(self) -> int:
        with self.cv:
            s = self.snap_i
            if self.snap_busy[s]:
                self.rep["snap_stall"] += 1
                while self.snap_busy[s] and self.error is None:
                    self.cv.wait(0.05)
            self.snap_busy[s] = True
            self.snap_i = (s + 1) % self.nsnap
        return s

    def _release_snap(self, s: int) -> None:
        with self.cv:
            self.snap_busy[s] = False
            self.cv.notify_all()

    # ---------------------------------------------------------- the worker
    def _start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, name="bf1-worker", daemon=True)
            self.thread.start()

    def _loop(self):
        """BATCHES (bf1b: the per-job cost, not the pack, bound the worker: 3,666 of 13,408 jobs issued, 14 ms of worker
        time per issued job, 2.84 transport chunks per job at 1.32 ms of submission each): one blocking get, then every
        queued job of the SAME epoch up to `batch` jobs, processed as ONE pack / transport job / commit."""
        with torch.inference_mode():                     # L32: the main thread's inference mode
            carry, have_carry = None, False
            left = []                                    # (job, ids): read, not placed (the entry was full): first next
            prev = None                                  # PIPELINE: (jobs, pending issue) of the previous batch
            while True:
                if left:
                    first = left[0][0]
                elif have_carry:
                    first, carry, have_carry = carry, None, False
                else:
                    got = False
                    if prev is not None:                 # PIPELINE: a queued job is selected before prev's issue
                        try:
                            first, got = self.q.get_nowait(), True
                        except queue.Empty:
                            pass
                    if not got:                          # nothing queued: prev's issue first, then block
                        prev = self._done(prev)
                        first = self.q.get()
                if first is None:
                    self._done(prev)
                    return
                if self.hold > 1 and not left and first[0] is not self._held_ep:   # tests: once per epoch
                    prev = self._done(prev)              # the hold waits for jobs; the previous batch first
                    self._hold(first)
                    self._held_ep = first[0]
                batch = [x[0] for x in left] or [first]
                idss = [x[1] for x in left] + [None] * (len(batch) - len(left))
                left = []
                while len(batch) < self.batch and not have_carry:
                    try:
                        nxt = self.q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None or nxt[0] is not first[0]:  # the stop, or a newer epoch's job: the next batch
                        carry, have_carry = nxt, True
                        break
                    batch.append(nxt)
                    idss.append(None)
                tj = time.perf_counter()
                self.rep["batch_in_max"] = max(self.rep.get("batch_in_max", 0), len(batch))   # jobs offered (backlog)
                pseq = prev[1]["seq"] if (prev is not None and prev[1] is not None) else 0
                p = None
                try:
                    left, p = self._prep(batch, idss, pseq)
                except BaseException as e:               # the compute thread raises it at its next hook
                    self.error = "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:])
                    for job, ids in zip(batch, idss):
                        if ids is None:
                            self._release_snap(job[2])
                    left, p = [], None
                self.rep["worker_busy_s"] = self.rep.get("worker_busy_s", 0.0) + time.perf_counter() - tj
                prev = self._done(prev)                  # PIPELINE: prev's issue after this batch's select + submit
                prev = (batch[:len(batch) - len(left)], p)  # the leftovers stay in flight (pend, inflight)

    def _done(self, prev):
        """PIPELINE: the previous batch's finish (its submission wait, its issue / commit), then its jobs leave the
        flight (inflight, pend: the joint pass's DEFER wait reads them). -> None."""
        if prev is None:
            return None
        done, p = prev
        t = time.perf_counter()
        try:
            if p is not None:
                self._finish(p)
        except BaseException as e:                       # the compute thread raises it at its next hook
            self.error = "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:])
        finally:
            with self.cv:
                self.inflight -= len(done)
                for job in done:
                    job[0].pend[job[1]] -= 1
                self.rep["worker_busy_s"] = self.rep.get("worker_busy_s", 0.0) + time.perf_counter() - t
                self.rep["batches"] = self.rep.get("batches", 0) + 1
                hs = self.rep.setdefault("batch_sizes", {})              # jobs per batch -> batches
                hs[str(len(done))] = hs.get(str(len(done)), 0) + 1
                self.cv.notify_all()
        return None

    def _hold(self, first) -> None:
        """TESTS ONLY (hold > 1): the CPU world makes 4 jobs per epoch (L 2 x 2 draft passes) and a CPU draft layer
        outlasts a short stall, so the worker took each job as it was queued and no batch of several jobs formed
        (4981010: batch_jobs_max 1). Before an epoch's first batch the worker waits until `hold` jobs are queued (with
        `first`), or the epoch closes (the joint pass started: no more jobs of it), or 10 s; the jobs are then a certain
        backlog (the later batches of the epoch find them queued: the PIPELINE test)."""
        t0 = time.perf_counter()
        ep = first[0]
        while (self.q.qsize() + 1 < self.hold and not (ep.closing or ep.closed) and self.error is None
               and time.perf_counter() - t0 < 10.0):
            time.sleep(0.001)
        self.rep["hold_s"] = self.rep.get("hold_s", 0.0) + time.perf_counter() - t0

    def stop(self) -> None:
        if self.thread is not None:
            self.q.put(None)
            self.thread.join()
            self.thread = None

    def _late(self, ep, j: int, t_enq: float, stage: str, n: int = 0) -> None:
        """A job that the epoch's close abandoned (under the lock). THE CAUSE (lead's request): stage "queued" = the
        worker had not started it when the joint pass closed the epoch; "packing" = closed while it was packed /
        submitted. age = close time - its enqueue time (a job of the last draft layers has a small age: the barrier;
        a large age: the worker's throughput); last_pass: it belongs to the last L jobs of the epoch."""
        ep.late += 1
        r = self.rep
        r["late"] += 1
        r["late_" + stage] = r.get("late_" + stage, 0) + 1
        r["items_late"] = r.get("items_late", 0) + n
        age = (ep.t_close - t_enq) * 1e3 if ep.t_close is not None else None
        last = ep.jobs_at_close is not None and j >= ep.jobs_at_close - self.L
        r["late_last_pass"] = r.get("late_last_pass", 0) + int(bool(last))
        self.late_ages.append(age)

    def _batch(self, jobs, idss=None) -> list:
        """One batch at once (the inline worker; the CPU tests): _prep, then _finish. -> the leftovers."""
        left, p = self._prep(jobs, idss, 0)
        if p is not None:
            self._finish(p)
        return left

    def _prep(self, jobs, idss=None, pseq: int = 0):
        """Steps 2-3 of the module docstring for a BATCH of (draft pass, layer) snapshots of ONE epoch, up to the
        transport's submission: per job the items (the store's keys, the epoch's dedup, the ROOM rule, the caps), then
        ONE pack and ONE transport job (gilfree: ONE release of every layer's items, chunked by cap_groups only; the C++
        worker runs it while the Python worker goes on: PIPELINE) or the pinned staging (torch). idss: the jobs'
        snapshots already read (None: read here). pseq: the previous batch's transport job, not finished yet (0: none);
        a batch whose select starts before that job was submitted counts as OVERLAPPED. -> (LEFTOVERS [(job, ids)],
        the pending issue for _finish or None): the jobs after the first one that would not fit in the landing entry
        (n_cap items, n_cap / H lanes per head) are not placed and not counted; the worker runs them first in its next
        batch (a job is dropped by the cap only when it alone does not fit)."""
        H, B, L, bs, D = self.H, self.B, self.L, self.bs, self.D
        ep = jobs[0][0]
        if self.stall_ms > 0:                            # tests only: a slow worker (late jobs on purpose)
            time.sleep(self.stall_ms * len(jobs) / 1e3)
        idss = list(idss) if idss is not None else [None] * len(jobs)
        ts = time.perf_counter()
        for i, (_, l, s, j, t_enq) in enumerate(jobs):
            if idss[i] is not None:
                continue
            with self.cap_lock:                          # a CUDA call: not during a capture
                self.snap_ev[s].synchronize()
            idss[i] = self.snap_h[s].numpy().copy()
            self._release_snap(s)
        self._tadd("snap_s", ts)
        if ep.closed:
            with self.lock:
                for _, l, s, j, t_enq in jobs:
                    self._late(ep, j, t_enq, "queued")
            return [], None
        t0 = time.perf_counter()
        if pseq:                                         # PIPELINE: the previous batch is not finished
            self.rep["pipelined"] = self.rep.get("pipelined", 0) + 1
            if self.tp is not None and int(self.tp.npipe.submitted) < int(pseq):
                self.rep["overlapped"] = self.rep.get("overlapped", 0) + 1   # its C++ job still ran: real overlap
        if ep.have is None:                              # the epoch's first job: the store's keys and its ROOM
            with self.cap_lock:
                ep.ev_map.synchronize()
            ep.have = np.zeros((L, H, B, self.nblk), dtype=bool)
            ep.free, (kl, kh, kb, kk) = store_room(self.kid_h[ep.par].numpy(), self.kage_h[ep.par].numpy(), ep.g, L,
                                                   L, self.nblk, self.kblock_h)
            ep.free0 = ep.free.copy()
            ep.have[kl, kh, kb, kk] = True
        parts, nonempty, left = [], [], []
        hc = np.zeros((H,), dtype=np.int64)              # gilfree: lanes used per head in this batch
        tot = 0
        for q, ((_, l, s, j, t_enq), ids) in enumerate(zip(jobs, idss)):
            h, b, m = np.nonzero(ids >= 0)
            blk = ids[h, b, m].astype(np.int64)
            seen = blk.shape[0]
            inr = blk < self.nblk
            h, b, blk = h[inr], b[inr], blk[inr]
            keep = ~ep.have[l, h, b, blk]
            h, b, blk = h[keep].astype(np.int64), b[keep].astype(np.int64), blk[keep]
            n_have = seen - h.shape[0]
            room = group_rank(h, b, B) < ep.free[h, b]  # ROOM: never pack what the commit cannot place
            n_room = int(h.shape[0] - room.sum())
            h, b, blk = h[room], b[room], blk[room]
            fit = np.arange(h.shape[0]) < self.n_cap - tot                  # the batch's landing entry: n_cap items
            if self.packer == "gilfree":                 # and n_cap / H lanes per head
                fit &= group_rank(h, np.zeros_like(h), 1) + hc[h] < self.lanes_e
            if q > 0 and not bool(fit.all()):            # FULL: this job and the later ones go to the next batch
                left = list(zip(jobs[q:], idss[q:]))
                jobs = jobs[:q]
                self.rep["carried"] = self.rep.get("carried", 0) + len(left)
                break
            n_cap = int((~fit).sum())
            h, b, blk = h[fit], b[fit], blk[fit]
            n = int(h.shape[0])
            ep.have[l, h, b, blk] = True
            np.subtract.at(ep.free, (h, b), 1)
            np.add.at(hc, h, 1)
            tot += n
            with self.lock:
                ep.items_seen += seen
                ep.items_have += n_have
                ep.items_room += n_room
                ep.items_cap += n_cap
                self.rep["items_seen"] += seen
                self.rep["items_have"] += n_have
                self.rep["items_room"] = self.rep.get("items_room", 0) + n_room
                self.rep["items_cap"] += n_cap
                if n == 0:
                    self.rep["empty_jobs"] += 1
            if n:
                parts.append((np.full_like(h, l), h, b, blk))
                nonempty.append((l, j, t_enq))
        self._tadd("select_s", t0)
        if not parts:
            return left, None
        lt, h, b, blk = (np.concatenate([p[i] for p in parts]) for i in range(4))
        o = np.lexsort((b, h))                          # by (head, request), stable
        lt, h, b, blk = lt[o], h[o], b[o], blk[o]
        n = int(h.shape[0])
        l_min, l_max = int(lt.min()), int(lt.max())
        self.rep["batch_layers_max"] = max(self.rep.get("batch_layers_max", 0), int(np.unique(lt).shape[0]))
        self.rep["batch_jobs_max"] = max(self.rep.get("batch_jobs_max", 0), len(nonempty))
        hr = group_rank(h, np.zeros_like(h), 1)          # the rank in its head (items sorted by head)
        rk = group_rank(h, b, B)
        e = self.land_i
        self.land_i = (e + 1) % self.ring
        tw = time.perf_counter()
        with self.cv:                                    # DEFER: the entry still holds a job that waits for its layer
            while self.land_busy[e] and not ep.closed and self.error is None:
                self.cv.notify_all()                     # a joint pass waiting for jobs consumes (frees) landed ones
                self.cv.wait(0.05)
        lane = e * self.lanes_e + hr                     # gilfree: lanes of entry e, one per (head, rank in head)
        if self.h2d_done[e] is not None:                 # the staging entry's previous H2D finished
            with self.cap_lock:
                self.h2d_done[e].synchronize()
        self._tadd("entry_s", tw)
        pblk = np.minimum(blk + 1, self.nblk - 1) if self.bneg == "wrong_bytes" else blk     # NEGATIVE CONTROL
        it = self.stage_it[e][:n]
        it.copy_(torch.from_numpy(np.stack((h, b, lt, blk, rk, lane), axis=1)))
        p = dict(ep=ep, jobs=jobs, nonempty=nonempty, n=n, e=e, it=it, layers=sorted(set(lt.tolist())), l_min=l_min,
                 l_max=l_max, jev=None, seq=0, t0=t0, tc=None)
        if self.packer == "torch":
            sk3 = self.stage_k[e][:n * bs].view(n, bs, D)
            sv3 = self.stage_v[e][:n * bs].view(n, bs, D)
            for x in sorted(set(lt.tolist())):          # per layer: its host rows, into the items' positions
                pos = np.nonzero(lt == x)[0]
                idx = torch.from_numpy(rows_of(h[pos], b[pos], pblk[pos], self.Tcpu, H, bs))
                p_t = torch.from_numpy(pos)
                eng = self.engs[x]
                sk3.index_copy_(0, p_t, torch.index_select(eng._k_cpu.view(-1, D), 0, idx).view(-1, bs, D))
                sv3.index_copy_(0, p_t, torch.index_select(eng._v_cpu.view(-1, D), 0, idx).view(-1, bs, D))
        else:                                            # GILFREE: ONE release of the batch, packed by the C++ team
            with self.cap_lock:
                if ep.closed or (ep.closing and l_max <= ep.consumed_upto):
                    with self.lock:
                        self._abandon(p, "passed" if not ep.closed else "packing")
                    return left, None
                p["tc"] = time.perf_counter()
                p["seq"] = self._release(e, h, b, lt, pblk, lane)
            p["jev"] = self.tp.job_ev(p["seq"])         # PIPELINE: not waited here (_finish waits its submission)
        return left, p

    def _abandon(self, p, stage: str) -> None:
        """Under the lock: every non-empty job of the batch is late."""
        for l_, j_, t_ in p["nonempty"]:
            self._late(p["ep"], j_, t_, stage, 0)
        self.rep["items_late"] = self.rep.get("items_late", 0) + p["n"]

    def _finish(self, p) -> None:
        """The batch's issue after its pack (PIPELINE: after the NEXT batch's select and submission): the wait for the
        C++ job's submission (cxx_wait_s: the exposed part), then the side stream's landing and the commit (or, DEFER,
        the landed entry for the joint pass)."""
        ep, jobs, nonempty, n, e, it = p["ep"], p["jobs"], p["nonempty"], p["n"], p["e"], p["it"]
        l_min, l_max, jev, t0, bs = p["l_min"], p["l_max"], p["jev"], p["t0"], self.bs
        if jev is not None:
            tw = time.perf_counter()
            jev.wait_submitted()                         # the GIL released: the C++ worker packed and enqueued it
            self._tadd("cxx_wait_s", tw)
            self._tadd("cxx_s", p["tc"])
        t1 = time.perf_counter()
        with self.cap_lock, self.lock:                   # the issue: atomic against the epoch's close
            self.rep["pack_s"] += t1 - t0
            if ep.closed or (ep.closing and l_max <= ep.consumed_upto):    # abandoned (DEFER: its layers already ran)
                self._abandon(p, "passed" if not ep.closed else "packing")
                return
            ctx = torch.cuda.stream(self.side) if self.cuda else contextlib.nullcontext()
            with ctx:
                if self.cuda:
                    self.side.wait_event(self.snap_ev[jobs[-1][2]])   # the newest snapshot of the batch
                    if self.land_reader[e] is not None:  # torch packer: a deferred commit read this entry last
                        self.side.wait_event(self.land_reader[e])
                ea, eb = self._ev(True), self._ev(True)
                ea.record()
                if jev is None:
                    self.land_k[e][:n * bs].copy_(self.stage_k[e][:n * bs], non_blocking=self.cuda)
                    self.land_v[e][:n * bs].copy_(self.stage_v[e][:n * bs], non_blocking=self.cuda)
                else:
                    jev.wait(self.side)                  # the landing (CPU: the host waits the job)
                self.land_it[e][:n].copy_(it, non_blocking=self.cuda)
                eb.record()
                self.h2d_done[e] = eb
                if ep.closing:                           # DEFER: landed; the joint pass commits it at its layers
                    ep.deferred.setdefault(l_min, []).append((e, n, eb, len(nonempty), p["layers"]))
                    self.land_busy[e] = True
                    self.rep["deferred"] = self.rep.get("deferred", 0) + len(nonempty)
                    self.rep["issue_s"] += time.perf_counter() - t1
                    if self.cuda:
                        self.h2d_ev.append((ea, eb, n * self.item_bytes))
                    return
                itd = self.land_it[e][:n]
                self._commit(itd, n, ep.g, *self._src(e, n, itd), p["layers"])
                done = self._ev()
                done.record()
            self._entry_read(e, done)
            ep.last = done
            ep.issued += len(nonempty)
            ep.items_sent += n
            self.rep["issued"] += len(nonempty)
            self.rep["items_sent"] += n
            self.rep["issue_s"] += time.perf_counter() - t1
            if self.cuda:
                self.h2d_ev.append((ea, eb, n * self.item_bytes))

    def _tadd(self, key: str, t0: float) -> None:
        """The worker's stage clocks (report: per epoch): snap_s (snapshot waits), select_s (the items on the host),
        entry_s (the landing entry's waits), cxx_s (the C++ release until its submission: with the PIPELINE it spans the
        next batch's select), cxx_wait_s (the part the worker waited for: the exposed C++ time), issue_s (the
        side-stream launches and the commit), pack_s (select to the issue). worker_busy_s excludes the blocking get."""
        self.rep[key] = self.rep.get(key, 0.0) + time.perf_counter() - t0

    def _release(self, e: int, h, b, lt, pblk, lane) -> int:
        """THE BATCH'S ONE RELEASE (bf1b: 2.84 chunks per job at 1.32 ms of submission each; issue_desc makes one
        release per LAYER, at least one chunk each): every item of the batch, all layers, as ONE job of the C++ worker
        (NativePipe.submit, the transport's own _submit path with xpool) into the ONE flat landing that every layer
        shares: source l' = layer * B + b (the transport's sources, one per (layer, request)), destination m' = the lane,
        so the chunks are ceil(n / cap_groups). T3: the landing entry's last commit (its reader mark); no T4 (the host rows
        exist before the draft pass that named them). -> the job seq."""
        tp = self.tp
        tp._raise_pending_error()
        tp._reap()
        keep = []
        rh = tp._mark_handle("reader", self.lmark[e], keep)          # 0: none, or done
        tp._start_worker()
        lk, lv = self.land_k.unsqueeze(0), self.land_v.unsqueeze(0)    # (1, M * 64, H, D): the transport's flat view
        seq = int(tp.npipe.submit([0] * int(h.shape[0]), h.tolist(), (lt * self.B + b).tolist(), lane.tolist(),
                                  pblk.tolist(), lk, lv, [], [], [], [rh] if rh else []))
        if keep:
            tp._keep[seq] = keep
        tp._last_seq = seq
        return seq

    def _commit(self, it, n: int, g: int, src_k, src_v, layers, deferred: bool = False, min_layer: int = -1) -> None:
        """Step 3's commit on the CURRENT stream (the side stream; the CPU: in line; DEFER: the compute stream at the
        joint pass's layer min_layer). it (n, 6) device rows h, b, layer, block, rank in (h, b), lane (a batch holds
        several layers); src_k / src_v (n x 64, D): the items' rows, item-major. deferred: no room counts apart (the
        joint pass's keep may have taken dead slots since the host counted the room; those bytes were moved and are
        wasted); items of layers < min_layer are not committed (their layer ran: "passed")."""
        T, H, B, X, bs = self.T, self.H, self.B, self.X, self.bs
        h, b, lt, blk, rk = it[:, 0], it[:, 1], it[:, 2], it[:, 3], it[:, 4]
        key = blk + lt * KM
        kid1 = T.kid[..., 1:]
        kage1 = T.kage[..., 1:]
        hit = T.kid[h, b, 1:] == key.view(n, 1)                          # (n, X - 1)
        found = hit.any(1)
        fs0 = hit.to(torch.uint8).argmax(1)                              # the found slot - 1
        held = torch.zeros((H, B, X - 1), dtype=torch.int64, device=self.dev)
        held.index_put_((h, b, fs0), found.to(torch.int64), accumulate=True)
        elig = ((kid1 < 0) | (kage1 < int(g) * self.L)) & (held == 0)
        if getattr(T, "kblock", None) is not None:                      # the keep's test-only slot limit (kslots)
            elig = elig & ~T.kblock
        order = torch.sort(torch.where(elig, kage1, torch.full_like(kage1, BIGI)), dim=-1, stable=True).indices
        nel = elig.sum(-1)
        passed = lt < int(min_layer)                                    # DEFER: their layer ran
        found = found & ~passed
        ok = (~found) & (rk < nel[h, b]) & ~passed
        slot = order[h, b, rk.clamp(max=X - 2)] + 1
        dslot = torch.where(ok, slot, torch.zeros_like(slot))           # not ok: store slot 0 (CB1, rewritten by J)
        rows = (((b * X + dslot).view(n, 1) * bs + self.ar_bs.view(1, bs)) * H + h.view(n, 1)).reshape(-1)
        if self.bneg != "no_copy":                                       # NEGATIVE CONTROL: the rows not copied
            self.sk.index_copy_(0, rows, src_k)
            self.sv.index_copy_(0, rows, src_v)
            pos = (blk * bs).view(n, 1) + self.ar_bs.view(1, bs)
            bias = None
            for x in layers:                             # each item's bias rows from ITS layer's twin
                vals = self.twins[x].total_cis[b.view(n, 1), pos, h.view(n, 1)]
                bias = vals if bias is None else torch.where((lt == x).view(n, 1), vals, bias)
            self.sb.index_copy_(0, rows, bias.reshape(-1).to(self.sb.dtype))
        wslot = torch.where(ok, slot, torch.where(found, fs0 + 1, torch.zeros_like(slot)))
        T.kid.index_put_((h, b, wslot), key)
        T.kage.index_put_((h, b, wslot), int(g) * self.L + lt)
        self.ptag.index_put_((h, b, wslot), ok | found)
        self.pkey.index_put_((h, b, wslot), key)
        nr = ((~ok) & (~found) & ~passed).sum()
        z = torch.zeros_like(nr)
        self.cnt_side.add_(torch.stack((torch.full_like(ok.sum(), n), ok.sum(), found.sum(), z if deferred else nr,
                                        nr if deferred else z, passed.sum())))

    # ---------------------------------------------------------- report (after the run, one sync)
    def report(self) -> dict:
        self.stop()
        tps = None
        if self.tp is not None:                          # drains the C++ worker, then its totals
            self.tp.flush()
            st = dict(self.tp.stats)
            self.tp.close()
            tps = dict(st, threads=self.threads, threads_ok=(st.get("threads_min") == self.threads
                                                             and st.get("threads_max") == self.threads),
                       pack_GBps_native=(round(self.rep["items_sent"] * self.item_bytes / st["pack_ns"], 3)
                                         if st.get("pack_ns") else None),
                       submit_ms_per_chunk=(round(st["submit_ns"] / st["chunks"] / 1e6, 4) if st.get("chunks") else None),
                       submit_GBps=(round(self.rep["items_sent"] * self.item_bytes / st["submit_ns"], 3)
                                    if st.get("submit_ns") else None),
                       groups_per_chunk=(round(st["groups"] / st["chunks"], 1) if st.get("chunks") else None))
            self.tp = None
        if self.cuda:
            torch.cuda.synchronize()
        out = dict(self.rep, mode=self.mode, worker=self.worker, n_cap=self.n_cap, ring=self.ring, acct=self.acct,
                   bneg=self.bneg, X=self.X, error=self.error, device_bytes=self.bytes_device(),
                   item_bytes=self.item_bytes, packer=self.packer, transport=tps, late_mode=self.late_mode,
                   late_wait_ms=self.late_wait_ms, lanes_per_entry=self.lanes_e, batch=self.batch)
        out.update({k: int(v) for k, v in zip(SIDE_KEYS, self.cnt_side.tolist())})
        out.update({k: int(v) for k, v in zip(MAIN_KEYS, self.cnt_main.tolist())})
        out["pending"] = int(self.ptag[..., 1:].sum())                   # tags of jobs after the last joint pass
        ib = self.item_bytes
        out["bytes_sent"] = out["items_sent"] * ib
        out["bytes_committed"] = out["committed"] * ib
        out["bytes_used"] = (out["used_store"] + out["used_win"]) * ib
        out["bytes_wasted"] = out["bytes_sent"] - out["bytes_used"]
        h2d = [(a.elapsed_time(b), nb) for a, b, nb in self.h2d_ev]
        ms = sum(x for x, _ in h2d)
        nb = sum(y for _, y in h2d)
        out["h2d_ms_total"] = round(ms, 3)              # gilfree: the side stream's wait for the landing (not the DMA)
        out["h2d_GBps"] = round(nb / ms / 1e6, 3) if (ms > 0 and self.packer == "torch") else None
        out["pack_GBps"] = round(out["bytes_sent"] / out["pack_s"] / 1e9, 3) if out["pack_s"] > 0 else None
        bw = [(s, a.elapsed_time(b)) for s, a, b in self.bar_ev]
        out["barrier_ms"] = [round(x, 3) for _, x in bw]
        out["barrier_ms_median"] = (sorted(x for _, x in bw)[len(bw) // 2] if bw else None)
        out["per_epoch"] = list(self.per_epoch)
        ages = sorted(a for a in self.late_ages if a is not None)
        out["late_age_ms_median"] = round(ages[len(ages) // 2], 3) if ages else None
        out["late_age_ms_max"] = round(ages[-1], 3) if ages else None
        fm = [p["free_mean"] for p in self.per_epoch if p.get("free_mean") is not None]
        fn = [p["free_min"] for p in self.per_epoch if p.get("free_min") is not None]
        out["room_free_mean"] = round(sum(fm) / len(fm), 2) if fm else None   # eligible slots per (head, request)
        out["room_free_min"] = min(fn) if fn else None
        dem = [p["items_seen"] - p["items_have"] for p in self.per_epoch]     # demand after the store / epoch dedup
        out["room_demand_per_hb_mean"] = round(sum(dem) / len(dem) / (self.H * self.B), 2) if dem else None
        ne = max(1, out["epochs"])                       # THE WORKER'S BUDGET per epoch (vs the draft window)
        out["worker_ms_per_epoch"] = {k: round(out.get(k, 0.0) * 1e3 / ne, 3) for k in (
            "worker_busy_s", "snap_s", "select_s", "entry_s", "cxx_s", "cxx_wait_s", "issue_s", "pack_s")}
        out.setdefault("pipelined", 0)
        out.setdefault("overlapped", 0)
        if tps is not None:
            out["worker_ms_per_epoch"]["cxx_submit"] = round(tps.get("submit_ns", 0) / 1e6 / ne, 3)
            out["worker_ms_per_epoch"]["cxx_pack"] = round(tps.get("pack_ns", 0) / 1e6 / ne, 3)
            out["chunks_per_epoch"] = round(tps.get("chunks", 0) / ne, 2)
        return out


# ------------------------------------------------------------------ the arm classes
_CLASSES = {}


def bf_classes(STF):
    """(TentStateBF, TentRunnerBF) on the I4DC keep D arm's classes of the module STF."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF, _make(STF))
    return _CLASSES[key][1]


def _make(STF):
    SpecError = STF.SpecError
    SKD, RKD = CD.dc_classes(STF)["L4PTFDGUDKWCD"]

    class TentStateBF(SKD):
        """L4PTFDGUDKWCD's state with the BF1 hooks: the draft residency (snapshot), the joint pass (barrier, accounting)
        and, with acct, the joint residency (window uses)."""

        bf = None

        def _resid_lean(self, l, eng, twin, ti, inj, act, drf, qb3):
            skip = super()._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)
            if self.bf is not None:
                self.bf.on_draft(l, eng, skip, drf, qb3)
            return skip

        def _resid_joint(self, l, eng, twin, ti, inj, act, drf, qb3, vs):
            if self.bf is not None:                                    # DEFER: the late jobs of this layer first
                self.bf.on_joint_layer(l)
            skip = super()._resid_joint(l, eng, twin, ti, inj, act, drf, qb3, vs)
            if self.bf is not None:
                self.bf.on_window(l, eng, skip)
            return skip

        def run_pass(self, model, plan):
            if plan.R == 1 or self.bf is None:
                return super().run_pass(model, plan)
            self.bf.before_joint()
            out = super().run_pass(model, plan)
            self.bf.after_joint()
            return out

        def bf_warm_joint(self) -> dict:
            """THE STORE-X WARM (module docstring, COMPILE): ONE call of the joint helpers pre, loc1 and loc2 on THIS
            state's own buffers (its store X: vs.lst_buf (H, B, X), vs.cap = X - 1, _cap_t, _sb1), exactly as the keep
            section's _cc_locate calls them (r2_i4cc_compile.py:310-329, fetch False), with a selection of -1 (no
            block) in vs.ti. Before the arm's first capture: a new store X compiles here (after_captures 0), and the
            first joint pass hits the cache. vs.ti and vs.lst_buf are restored byte for byte; the call counts too."""
            vs = self._vs
            why = self._cc_why(_PlanVS(vs))
            R = int(vs.R)
            hp = self._hps.get(R)
            if why is not None or hp is None:
                return dict(ran=False, why=why or "no hoist buffers of R %d" % R)
            H, B, T = self.H, self.B, int(vs.topk)
            rs = self.rs
            eng = rs.cache.layers[0].cache_engine
            n0, calls0 = len(self.cc["events"]), dict(self.cc["calls"])
            ti0, lb0 = vs.ti.clone(), vs.lst_buf.clone()
            t0 = time.perf_counter()
            try:
                vs.ti.fill_(-1)
                sel = vs.ti.view(H, B, R, T)
                ti, inj, sel_c, is_q, is_n = self._cc("pre", 0, sel, hp.qb3, hp.qb4, hp.qb1_4, vs.minus1)
                v = sel_c.view(H, B, R * T)
                ws, wi = torch.sort(eng._block_map, dim=-1)
                ps, pi = torch.sort(eng._pool_map, dim=-1) if rs.pool else (None, None)
                valid, in_win, wslot, in_pool, pslot, keyh = self._cc("loc1", 0, v, ws, wi, ps, pi, hp.fullv, vs.big,
                                                                      int(vs.topk))
                s = torch.sort(keyh, dim=-1).values
                self._cc("loc2", 0, s, v, valid, in_win, wslot, in_pool, pslot, sel, is_q, is_n, self._eb[0],
                         self._sb1, self.cb0_id, vs.store_b.view(1, B, 1, 1), vs.lst_buf, vs.host_x, vs.minus1,
                         vs.big, self._cap_t, int(vs.cap), int(STF.VR.FORK_BIG), False, bool(vs.ops.stats))
            finally:
                vs.ti.copy_(ti0)
                vs.lst_buf.copy_(lb0)
                self.cc["calls"] = calls0
            ev = self.cc["events"][n0:]
            for e in ev:
                e["warm"] = "store X %d" % int(vs.X)
            return dict(ran=True, X=int(vs.X), cap=int(vs.cap), compiled=[e["name"] for e in ev],
                        s=round(time.perf_counter() - t0, 3))

    class TentRunnerBF(RKD):
        """The arms L4PTFDGUDKWCDB (bf mode "ce") and L4PTFDGUDKWCDX (bf mode "off"). bf: dict(mode, n_cap, ring,
        worker, acct, bneg) for BgFetch."""

        def __init__(self, *args, bf=None, **kw):
            o = self._bfo = dict(bf or {})
            super().__init__(*args, **kw)
            self.bf = BgFetch(self.T, **o)
            self.T.bf = self.bf
            self.bf.wrap_backend(getattr(self.T, "backend", None))
            self.bf.open_epoch()
            with torch.inference_mode():                               # the run's mode (_run_paper)
                warm = self.T.bf_warm_joint()                          # COMPILE: before the arm's first capture
            self.meta.update(bf=dict(o, X=self.bf.X, device_bytes=self.bf.bytes_device(), warm=warm))

        def _arm_name(self):
            return "L4PTFDGUDKWCDX" if self._bfo.get("mode", "ce") == "off" else "L4PTFDGUDKWCDB"

        def _make_state(self, rs, attn, tneg, poison):
            """RKD's state, then its class becomes TentStateBF (a subclass that adds the three hooks and no field)."""
            T = super()._make_state(rs, attn, tneg, poison)
            if type(T) is not SKD:
                raise SpecError("BF1: the keep D arm's state class changed (%s)" % type(T).__name__)
            T.__class__ = TentStateBF
            return T

        @torch.inference_mode()
        def read(self) -> dict:
            rep = self.bf.report()
            out = super().read()
            out.update(bf=rep)
            return out

        def detach(self) -> None:
            self.bf.stop()
            super().detach()

    return TentStateBF, TentRunnerBF
