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
BNEG = (None, "wrong_bytes", "no_copy")
# device counters: the side stream's (one tensor) and the compute stream's (another tensor): no tensor has two writers
SIDE_KEYS = ("items", "committed", "revived", "no_room")
MAIN_KEYS = ("used_store", "unused", "used_win")


class BFError(RuntimeError):
    pass


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
                 gf=None):
        if mode not in MODES or worker not in WORKERS or bneg not in BNEG or packer not in PACKERS:
            raise BFError("mode %r / worker %r / bneg %r / packer %r: one of %s / %s / %s / %s" % (
                mode, worker, bneg, packer, MODES, WORKERS, BNEG, PACKERS))
        self.packer, self.threads, self.gf = packer, int(threads), dict(gf or {})
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
        self.land_it = torch.zeros((self.ring, self.n_cap, 5), dtype=i64, device=dev)
        self.stage_it = self._pinned((self.ring, self.n_cap, 5), i64)
        self.h2d_done = [None] * self.ring               # the event of the entry's last H2D (the staging lifetime)
        self.land_i = 0
        self.tp = None
        if self.packer == "torch":                       # landing (ring, n_cap x 64 rows, D): item-major rows
            self.land_k = torch.empty((self.ring, n, D), dtype=self.kdt, device=dev)
            self.land_v = torch.empty((self.ring, n, D), dtype=self.kdt, device=dev)
            self.stage_k = self._pinned((self.ring, n, D), self.kdt)
            self.stage_v = self._pinned((self.ring, n, D), self.kdt)
        else:                                            # GILFREE (module docstring, PACKER): the flat lanes
            self.land_k = torch.zeros((self.ring * n, H, D), dtype=self.kdt, device=dev)
            self.land_v = torch.zeros((self.ring * n, H, D), dtype=self.kdt, device=dev)
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
                   xpool=_LaneMap(self.ring * self.n_cap, self.B, self.H))

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
            self._job(*job)
        else:
            self._start()
            with self.cv:
                self.inflight += 1
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
        with torch.inference_mode():                     # L32: the main thread's inference mode
            while True:
                job = self.q.get()
                if job is None:
                    return
                tj = time.perf_counter()
                try:
                    self._job(*job)
                except BaseException as e:               # the compute thread raises it at its next hook
                    self.error = "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:])
                    self._release_snap(job[2])
                finally:
                    with self.cv:
                        self.inflight -= 1
                        self.rep["worker_busy_s"] = self.rep.get("worker_busy_s", 0.0) + time.perf_counter() - tj
                        self.cv.notify_all()

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

    def _job(self, ep, l: int, s: int, j: int = 0, t_enq: float = 0.0) -> None:
        """Steps 2-3 of the module docstring for one (draft pass, layer) snapshot."""
        H, B, L, bs, D = self.H, self.B, self.L, self.bs, self.D
        with self.cap_lock:                              # a CUDA call: not during a capture
            self.snap_ev[s].synchronize()
        ids = self.snap_h[s].numpy().copy()
        self._release_snap(s)
        if ep.closed:
            with self.lock:
                self._late(ep, j, t_enq, "queued")
            return
        t0 = time.perf_counter()
        if ep.have is None:                              # the epoch's first job: the store's keys and its ROOM
            with self.cap_lock:
                ep.ev_map.synchronize()
            ep.have = np.zeros((L, H, B, self.nblk), dtype=bool)
            ep.free, (kl, kh, kb, kk) = store_room(self.kid_h[ep.par].numpy(), self.kage_h[ep.par].numpy(), ep.g, L,
                                                   L, self.nblk, self.kblock_h)
            ep.free0 = ep.free.copy()
            ep.have[kl, kh, kb, kk] = True
        h, b, m = np.nonzero(ids >= 0)
        blk = ids[h, b, m].astype(np.int64)
        seen = blk.shape[0]
        inr = blk < self.nblk
        h, b, blk = h[inr], b[inr], blk[inr]
        keep = ~ep.have[l, h, b, blk]
        h, b, blk = h[keep].astype(np.int64), b[keep].astype(np.int64), blk[keep]
        n_have = seen - h.shape[0]
        room = group_rank(h, b, B) < ep.free[h, b]      # ROOM: never pack what the commit cannot place
        n_room = int(h.shape[0] - room.sum())
        h, b, blk = h[room], b[room], blk[room]
        n_cap = max(0, h.shape[0] - self.n_cap)
        h, b, blk = h[:self.n_cap], b[:self.n_cap], blk[:self.n_cap]
        n = int(h.shape[0])
        ep.have[l, h, b, blk] = True
        np.subtract.at(ep.free, (h, b), 1)
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
            with self.lock:
                self.rep["empty_jobs"] += 1
            return
        h, b = h.astype(np.int64), b.astype(np.int64)
        rk = group_rank(h, b, B)
        e = self.land_i
        self.land_i = (e + 1) % self.ring
        lane = e * self.n_cap + group_rank(h, np.zeros_like(h), 1)    # gilfree: a lane per (head, rank in head)
        if self.h2d_done[e] is not None:                 # the staging entry's previous H2D finished
            with self.cap_lock:
                self.h2d_done[e].synchronize()
        pblk = np.minimum(blk + 1, self.nblk - 1) if self.bneg == "wrong_bytes" else blk     # NEGATIVE CONTROL
        it = self.stage_it[e][:n]
        it.copy_(torch.from_numpy(np.stack((h, b, blk, rk, lane), axis=1)))
        jev = None
        if self.packer == "torch":
            idx = torch.from_numpy(rows_of(h, b, pblk, self.Tcpu, H, bs))
            eng = self.engs[l]
            torch.index_select(eng._k_cpu.view(-1, D), 0, idx, out=self.stage_k[e][:n * bs])
            torch.index_select(eng._v_cpu.view(-1, D), 0, idx, out=self.stage_v[e][:n * bs])
        else:                                            # GILFREE: ONE descriptor job, packed by the C++ team
            desc = torch.from_numpy(np.stack((np.full_like(h, l), b, h, pblk, lane, np.zeros_like(h)))).contiguous()
            readers = [-1] * L
            readers[l] = self.lmark[e]                   # T3: the landing entry's last commit read it
            with self.cap_lock:
                if ep.closed:
                    with self.lock:
                        self._late(ep, j, t_enq, "packing", n)
                    return
                seq = int(self.tp.issue_desc(desc, reader_marks=readers, hist_mark=-1))   # T4: the snapshot waited
            jev = self.tp.job_ev(seq)
            jev.wait_submitted()                         # the GIL released: the C++ worker packed and enqueued it
        t1 = time.perf_counter()
        with self.cap_lock, self.lock:                   # the issue: atomic against the epoch's close
            self.rep["pack_s"] += t1 - t0
            if ep.closed:                                # closed while packing: abandoned
                self._late(ep, j, t_enq, "packing", n)
                return
            ctx = torch.cuda.stream(self.side) if self.cuda else contextlib.nullcontext()
            with ctx:
                if self.cuda:
                    self.side.wait_event(self.snap_ev[s])
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
                itd = self.land_it[e][:n]
                if jev is None:
                    src_k, src_v = self.land_k[e][:n * bs], self.land_v[e][:n * bs]
                else:                                    # the item's lane rows of its head
                    rsrc = ((itd[:, 4].view(n, 1) * bs + self.ar_bs.view(1, bs)) * H + itd[:, 0].view(n, 1)).reshape(-1)
                    src_k = self.land_k.view(-1, D).index_select(0, rsrc)
                    src_v = self.land_v.view(-1, D).index_select(0, rsrc)
                self._commit(itd, n, l, ep.g, src_k, src_v)
                done = self._ev()
                done.record()
            if jev is not None:
                mk = self.mark_next
                self.mark_next += 1
                self.tp.mark_reader(mk, done)
                self.lmark[e] = mk
            ep.last = done
            ep.issued += 1
            ep.items_sent += n
            self.rep["issued"] += 1
            self.rep["items_sent"] += n
            self.rep["issue_s"] += time.perf_counter() - t1
            if self.cuda:
                self.h2d_ev.append((ea, eb, n * self.item_bytes))

    def _commit(self, it, n: int, l: int, g: int, src_k, src_v) -> None:
        """Step 3's commit on the CURRENT stream (the side stream; the CPU: in line). it (n, 5) device rows h, b, block,
        rank in (h, b), lane; src_k / src_v (n x 64, D): the items' rows, item-major."""
        T, H, B, X, bs = self.T, self.H, self.B, self.X, self.bs
        h, b, blk, rk = it[:, 0], it[:, 1], it[:, 2], it[:, 3]
        key = blk + int(l) * KM
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
        ok = (~found) & (rk < nel[h, b])
        slot = order[h, b, rk.clamp(max=X - 2)] + 1
        dslot = torch.where(ok, slot, torch.zeros_like(slot))           # not ok: store slot 0 (CB1, rewritten by J)
        rows = (((b * X + dslot).view(n, 1) * bs + self.ar_bs.view(1, bs)) * H + h.view(n, 1)).reshape(-1)
        if self.bneg != "no_copy":                                       # NEGATIVE CONTROL: the rows not copied
            self.sk.index_copy_(0, rows, src_k)
            self.sv.index_copy_(0, rows, src_v)
            tc = self.twins[l].total_cis
            pos = (blk * bs).view(n, 1) + self.ar_bs.view(1, bs)
            bias = tc[b.view(n, 1), pos, h.view(n, 1)]
            self.sb.index_copy_(0, rows, bias.reshape(-1).to(self.sb.dtype))
        wslot = torch.where(ok, slot, torch.where(found, fs0 + 1, torch.zeros_like(slot)))
        T.kid.index_put_((h, b, wslot), key)
        T.kage.index_put_((h, b, wslot), torch.full_like(key, int(g) * self.L + int(l)))
        self.ptag.index_put_((h, b, wslot), ok | found)
        self.pkey.index_put_((h, b, wslot), key)
        self.cnt_side.add_(torch.stack((torch.full_like(ok.sum(), n), ok.sum(), found.sum(), ((~ok) & (~found)).sum())))

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
                                         if st.get("pack_ns") else None))
            self.tp = None
        if self.cuda:
            torch.cuda.synchronize()
        out = dict(self.rep, mode=self.mode, worker=self.worker, n_cap=self.n_cap, ring=self.ring, acct=self.acct,
                   bneg=self.bneg, X=self.X, error=self.error, device_bytes=self.bytes_device(),
                   item_bytes=self.item_bytes, packer=self.packer, transport=tps)
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
            self.meta.update(bf=dict(o, X=self.bf.X, device_bytes=self.bf.bytes_device()))

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
