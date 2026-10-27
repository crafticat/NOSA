"""POOL PREFETCH: background fetch of host KV blocks into the victim pool of cached NOSA, and an A/B of the shipped
fetch (Task 1.2 of kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md; paper 3.1 and 3.3; MAP_cached_nosa.md
hooks (b) and (d)). OPT-IN: nothing in this file runs unless a caller builds a PoolPrefetch and calls attach().
cache_engine.py and nosa_llama.py are NOT changed. With the flag off, the engine runs the shipped code (Gate 0).

REUSE (no new kernel)
- Transport: r2_transport_gilfree_v3.py. It is a BYTE COPY of nosi-r2-f3ldx 3769b64 nosi/nosi/r2_transport_gilfree.py
  (a new file name: this tree keeps its older r2_transport_gilfree.py and that file's test). The C++ packer v3 is
  benchmarks/Efficiency/gilfree_pack.{py,cpp}, byte copies of the same commit. We use the C++ worker (worker="cpp")
  and the descriptor issue (issue_desc), as f3t does, on the transport's flat-store option (xpool=_Lanes).
- Scatter: the shipped Triton kernels flash_h2d_from_mask (K) and flash_h2d_from_mask_bias (V and bias). They take any
  source tensor and its strides (flash_h2d_mask.py:65-88). Our source is a GPU staging view with batch stride 0.
- Pool bookkeeping: the shipped pool_update kernel and flash_pool_swap. We only write _pool_map / _pool_age.

THE DATA PATH (both uses)
  host rows (pinned) --C++ worker, OpenMP pack--> pinned staging ring --copy engine, copy stream--> landing ring (GPU)
  --index_copy_, scatter stream--> GPU STAGING (lane j of layer l: rows (j * 64 + r) * H + h) --arrival event-->
  flash_h2d_from_mask / _bias --> pool slots (use 1) or window slots (use 2).
  The bias does not cross PCIe. We gather its rows from the GPU-resident total_cis (the kv_bias of the step) into a
  small bias staging buffer (one lane per item, as K and V), then the shipped bias kernel copies them with V.

USE 1, BACKGROUND PREFETCH INTO POOL SLOTS (prefetch(); begin_step())
- prefetch(items): items are (layer, KV head, request, host block id). We remove duplicates, give each item a staging
  lane, and send ONE descriptor job to the transport. The C++ worker packs and sends it while the GPU decodes.
- begin_step() of the NEXT step COMMITS each submitted job, per layer, on the COMMIT STREAM:
    1. wait the job's arrival (the transport's job event);
    2. wait the LAST READER of the layer's pool slots (rule T3, kstep-cpu-runs/paper-notes/system-note/
       SYSTEM_NOTE.md:333). The only kernels that read pool slots are the layer's pool_update and flash_pool_swap on
       the main stream. They run in ONE launch per layer and step, and the host cannot know which slots they read.
       So every slot of layer l has the same last-reader event: reader_ev[l], recorded on the main stream after the
       layer's shipped update. (A per-slot table would hold this same event for every slot.)
    3. drop each item whose block is already in the window (_block_map) or in the pool (_pool_map);
    4. choose a pool slot per item: the least recently used slot that is NOT pinned (ages ascending; pinned ages are
       above every real age). If the (head, request) row has no unpinned slot left, drop the item and count it
       (commit_overflow). A commit never overwrites a pinned block;
    5. copy the rows from the staging buffer into the chosen pool slots (the shipped scatter kernels);
    6. write _pool_map[h, b, q] = X and _pool_age[h, b, q] = PIN_BASE + a counter (the SOFT PIN);
    7. record commit_ev[l]. The main stream waits it before the layer's next update. The transport's next job for
       layer l waits it too (a reader mark: the commit read the layer's staging lanes).
- The next pool_update then finds X in the pool: an ORDINARY POOL HIT (pool_update_kernel.cu phase 1). The shipped D2D
  swap moves X into the window and the host gather skips it. A hit rewrites the age to a real stamp: the pin ends.
- SOFT PIN. Real stamps are stamp_base + m, with stamp_base growing by 64 per call (cache_engine.py:511), so they stay
  far below PIN_BASE = 2**60. Pinned slots rank last as LRU victims. It is NOT a hard pin: when every unclaimed slot of
  a row is pinned, pool_update still evicts the oldest pinned slot (pool_update_kernel.cu:237-247). We COUNT each such
  eviction (pin_evicted) and each pin that ends in a hit (pin_hits). While the hooks stay attached:
      committed == pin_hits + pin_evicted + pins_live            (no silent drop; report()["conserved"])
- CAPTURE STEPS. The shipped decode captures its pooling graph at each re-warm step. A CUDA graph capture in the
  default "global" mode fails when another thread calls an unsafe CUDA API (for example an event query or an event
  synchronize). The C++ worker makes such calls. So begin_step(capture=True) drains the transport first and sends no
  new job in that step; prefetch() refuses a call in such a step.

USE 2, --fetch cpupack (the paper 4.4 A/B on the baseline itself)
- The step's OWN misses (the shipped _load_mask after pool_update) go through the same path on the CRITICAL PATH, in
  place of the shipped main-stream GPU gather. Per layer: one D2H of _load_mask and a host wait, the descriptor job
  (chunks of `lanes` items per KV head), a host wait for the job's submission, a main-stream wait for its arrival, then
  the shipped scatter kernels with the staging as the source (destination: the window slots, as the shipped gather).
- The engine method for this mode is DERIVED at attach time from the shipped method decode_update_has_kv_bias_pool:
  its source with the two gather lines replaced by ONE call (self._pf_fetch(kv_bias)). The derivation refuses to run
  when the two lines do not occur exactly once. So the two bodies cannot drift apart.
- Use 1 and use 2 are separate arms: cpupack refuses a background load and prefetch().

FLAG OFF (Gate 0). attach() is the only entry. It replaces the instance attribute decode_update of each CacheEngine
(the shipped binding, cache_engine.py:172-173) with a wrapper; detach() puts back the same object. With fetch
"gpugather" the wrapper calls the SHIPPED bound method itself, between the hooks _before and _after.

NEGATIVE CONTROLS (tests only): skip_reader_wait (the commit does not wait reader_ev), count=False (no pin accounting).
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import linecache
import textwrap
import time
import types
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .kstep.backend import ev_done

PIN_BASE = 1 << 60                          # every soft-pin age is >= this; every real stamp is far below it
POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3   # pool_update_kernel.cu:79-82
COUNTERS = ("requested", "committed", "dup_dropped", "commit_overflow", "pin_hits", "pin_evicted")
FETCH = ("gpugather", "cpupack")
BG_COMMIT = ("none", "pool")
SHIPPED_NAME = "decode_update_has_kv_bias_pool"
DERIVED_NAME = "decode_update_pool_cpupack"
FETCH_K = "flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)"
FETCH_V = ("flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask, "
           "self.block_size)")
FETCH_CALL = "self._pf_fetch(kv_bias)"


class _Done:
    """A completed event (CPU mode)."""
    done = True

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _Lanes:
    """The staging geometry in the form that GilfreeTransport's flat-store option reads (its xpool argument): M
    positions of 64 rows, A = M usable lanes, the batch B and the KV heads H. lanes_of(desc) gives the lane of each
    take: our descriptors put the lane in the slot row (row 4)."""

    def __init__(self, M: int, B: int, H: int):
        self.M = self.A = int(M)
        self.B, self.H = int(B), int(H)

    def lanes_of(self, desc: torch.Tensor) -> torch.Tensor:
        return desc[4].clone()


class _Job:
    __slots__ = ("seq", "items", "deferred", "commit_ev")

    def __init__(self, seq: int, items: Dict[int, torch.Tensor]):
        self.seq, self.items, self.deferred, self.commit_ev = int(seq), items, 0, None


def derive_cpupack(ce_module) -> tuple:
    """The --fetch cpupack engine method: the SOURCE of the shipped CacheEngine.decode_update_has_kv_bias_pool with its
    two gather lines (FETCH_K, FETCH_V) replaced by FETCH_CALL, compiled in the cache_engine module's own globals (so
    it calls the same diff, pool_update, flash_pool_swap and trace objects). Returns (function, derived source).
    Refuses when the two lines are not found exactly once, or the def line is not found exactly once."""
    fn = ce_module.CacheEngine.__dict__[SHIPPED_NAME]
    lines = textwrap.dedent(inspect.getsource(fn)).split("\n")
    at = [i for i in range(len(lines) - 1) if lines[i].strip() == FETCH_K and lines[i + 1].strip() == FETCH_V]
    if len(at) != 1:
        raise RuntimeError("pool_prefetch: the shipped gather lines occur %d times in %s (need exactly 1)"
                           % (len(at), SHIPPED_NAME))
    i = at[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i + 2] = [indent + FETCH_CALL + "   # pool_prefetch --fetch cpupack: replaces the two shipped gathers"]
    head = "def %s(" % SHIPPED_NAME
    defs = [k for k, s in enumerate(lines) if s.startswith(head)]
    if len(defs) != 1:
        raise RuntimeError("pool_prefetch: the def line of %s occurs %d times" % (SHIPPED_NAME, len(defs)))
    lines[defs[0]] = lines[defs[0]].replace(head, "def %s(" % DERIVED_NAME, 1)
    src = "\n".join(lines)
    fname = "<pool_prefetch: %s with --fetch cpupack>" % SHIPPED_NAME
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)   # readable tracebacks
    ns: Dict[str, Any] = {}
    exec(compile(src, fname, "exec"), ce_module.__dict__, ns)                  # noqa: S102 (derived, checked source)
    return ns[DERIVED_NAME], src


def default_ops():
    """The shipped scatter kernels (Triton). The CPU tests pass their own twins (flat-address models, L28)."""
    from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
    from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
    return types.SimpleNamespace(h2d=flash_h2d_from_mask, h2d_bias=flash_h2d_from_mask_bias)


class PoolPrefetch:
    """See the module docstring. Build it AFTER the prefill (it reads the host rows, the pool state and total_cis).

    cache      the InfLLMv2Cache (or any object with .layers[l].cache_engine and .layers[l].total_cis)
    ext        the gilfree_pack v3 extension (gilfree_pack.load); not needed when `transport` is given
    fetch      "gpugather" (the shipped gather; default) or "cpupack" (use 2)
    lanes      staging lanes per (layer, KV head): items per KV head per job and layer (and per cpupack chunk)
    threads    the packer's OpenMP team; cores: the C++ worker's CPU affinity (None: inherited)
    bg_mib     background load per step (MiB of K + V), sent at begin_step; bg_commit "none" (staging only: the
               transport cost alone) or "pool" (the whole use-1 path with soft pins); bg_seed: its generator seed
    backlog_cap  begin_step skips the background job while this many jobs have not landed (counted bg_skipped)
    ops        the scatter kernels (h2d, h2d_bias); host_wait: the host wait of an event (CPU mode, tests)
    count      the pin accounting (pin_hits, pin_evicted); False only as a NEGATIVE CONTROL
    """

    def __init__(self, cache, ext=None, *, fetch: str = "gpugather", lanes: int = 128, threads: int = 8,
                 cores: Optional[Sequence[int]] = None, cap_groups: int = 256, ring: int = 2, bg_mib: float = 0.0,
                 bg_commit: str = "none", bg_seed: int = 0, backlog_cap: int = 16, ops=None, transport=None,
                 host_wait: Optional[Callable[[Any], None]] = None, count: bool = True, probes: bool = True):
        self.cache = cache
        self.engines = [layer.cache_engine for layer in cache.layers]
        if not self.engines:
            raise ValueError("pool_prefetch: the cache has no layers")
        e0 = self.engines[0]
        for l, e in enumerate(self.engines):
            if int(getattr(e, "pool_blocks", 0)) <= 0 or getattr(e, "_pool_map", None) is None:
                raise ValueError("pool_prefetch: layer %d has no victim pool (NOSI_POOL_BLOCKS > 0 and a prefill "
                                 "are needed)" % l)
            if (e._k_gpu.shape != e0._k_gpu.shape or e._k_cpu.shape != e0._k_cpu.shape or e.topk != e0.topk
                    or e.block_size != e0.block_size or e._k_gpu.dtype != e0._k_gpu.dtype):
                raise ValueError("pool_prefetch: layer %d differs in geometry from layer 0" % l)
            bound = e.__dict__.get("decode_update")
            if getattr(getattr(bound, "__func__", None), "__name__", None) != SHIPPED_NAME:
                raise ValueError("pool_prefetch: layer %d's decode_update is not the shipped %s (already attached?)"
                                 % (l, SHIPPED_NAME))
        self.L = len(self.engines)
        self.B, _, self.H, self.D = (int(x) for x in e0._k_gpu.shape)
        self.bs, self.topk, self.P = int(e0.block_size), int(e0.topk), int(e0.pool_blocks)
        self.dev = e0._k_gpu.device
        self.cuda = self.dev.type == "cuda"
        self.dtype = e0._k_gpu.dtype
        self.item_bytes = 2 * self.bs * self.D * e0._k_gpu.element_size()     # K + V of one block of one KV head
        self.lanes = int(lanes)
        if self.lanes < 1:
            raise ValueError("pool_prefetch: lanes >= 1")
        self.total_cis = [getattr(layer, "total_cis", None) for layer in cache.layers]
        self.host_wait = host_wait or (lambda ev: ev.synchronize())
        self.ops = ops if ops is not None else default_ops()
        self.count = bool(count)
        self.probes = bool(probes) and self.cuda
        self.skip_reader_wait = False                          # NEGATIVE CONTROL (tests only)
        self.backlog_cap = int(backlog_cap)
        self.fetch = self.bg_mib = self.bg_commit = None
        self.set_mode(fetch=fetch, bg_mib=bg_mib, bg_commit=bg_commit)
        # the staging buffers: lanes x 64 rows per (layer); K and V (H heads per row), the bias (H per row)
        R = self.lanes * self.bs
        self.stage_k = torch.zeros((self.L * R, self.H, self.D), dtype=self.dtype, device=self.dev)
        self.stage_v = torch.zeros_like(self.stage_k)
        self.bstage = torch.zeros((self.L * R, self.H), dtype=e0._kv_bias_gpu.dtype, device=self.dev)
        self.staging_bytes = 2 * self.stage_k.numel() * self.stage_k.element_size() + \
            self.bstage.numel() * self.bstage.element_size()
        if transport is None:
            from .r2_transport_gilfree_v3 import GilfreeTransport
            if ext is None:
                raise ValueError("pool_prefetch: give the gilfree_pack v3 extension (ext) or a transport")
            transport = GilfreeTransport(
                ext, [e._k_cpu for e in self.engines], [e._v_cpu for e in self.engines],
                [self.stage_k[l * R:(l + 1) * R] for l in range(self.L)],
                [self.stage_v[l * R:(l + 1) * R] for l in range(self.L)], self.bs, self.D, int(threads),
                self.host_wait, cap_groups=int(cap_groups), ring=int(ring), cores=cores, worker="cpp",
                xpool=_Lanes(self.lanes, self.B, self.H))
        self.tp = transport
        self.threads, self.cores = int(threads), (list(cores) if cores else None)
        if self.cuda:
            self._cs = torch.cuda.Stream(device=self.dev)              # the commit stream
            self._ps_issue = torch.cuda.Stream(device=self.dev)        # probes: issue time
            self._ps_arrive = torch.cuda.Stream(device=self.dev)       # probes: arrival time
            self._mask_host = [torch.empty((self.H, self.B, self.topk), dtype=torch.int64, pin_memory=True)
                               for _ in range(2)]
        else:
            self._cs = self._ps_issue = self._ps_arrive = None
            self._mask_host = None
        self._ar = torch.arange(self.bs, dtype=torch.int64, device=self.dev)
        self.counters = torch.zeros(len(COUNTERS), dtype=torch.int64, device=self.dev)
        self.reader_ev: List[Any] = [None] * self.L
        self.commit_ev: List[Any] = [None] * self.L
        self.pinned = [False] * self.L
        self._pinb: List[Optional[torch.Tensor]] = [None] * self.L
        self.pin_seq = 0
        self.pending: List[_Job] = []
        self._committed: List[_Job] = []                       # alive until their commit events completed
        self.busy = [False] * self.L
        self.lane_next = torch.zeros((self.L, self.H), dtype=torch.int64)
        self._rmark = [-1] * self.L                             # the newest staging reader mark per layer (T3)
        self._mark_next = 0
        self._hist_mark = -1
        self._hist_next = 0
        self._last_seq = 0
        self._capture = False
        # the blocks every request has complete NOW: the background load reads only these. The synchronize makes every
        # write-back enqueued so far complete, so these host rows need no write-back wait (T4)
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self.nb_safe = min(int(e.seq_length) for e in self.engines) // self.bs
        self._bg_gen = torch.Generator().manual_seed(int(bg_seed))
        self.attached = False
        self._saved: List[Any] = [None] * self.L
        self._derived = None
        self.derived_source = None
        self.reset_stats()

    # ------------------------------------------------------------------------------------------------ configuration
    def set_mode(self, fetch: str = "gpugather", bg_mib: float = 0.0, bg_commit: str = "none") -> None:
        """Changes the arm (only while detached)."""
        if getattr(self, "attached", False):
            raise RuntimeError("pool_prefetch: set_mode while attached")
        if fetch not in FETCH:
            raise ValueError("pool_prefetch: fetch must be one of %s (got %r)" % (FETCH, fetch))
        if bg_commit not in BG_COMMIT:
            raise ValueError("pool_prefetch: bg_commit must be one of %s (got %r)" % (BG_COMMIT, bg_commit))
        if float(bg_mib) < 0:
            raise ValueError("pool_prefetch: bg_mib >= 0")
        if fetch == "cpupack" and float(bg_mib) > 0:
            raise ValueError("pool_prefetch: --fetch cpupack and a background load are separate arms")
        self.fetch, self.bg_mib, self.bg_commit = fetch, float(bg_mib), bg_commit

    def reset_stats(self) -> None:
        """The per-arm host statistics (report() returns them); the device counters stay cumulative."""
        self.stats: Dict[str, Any] = dict(
            steps=0, jobs=0, items_issued=0, bytes_issued=0, dup_items=0, busy_refused=0, lane_refused=0,
            deferred=0, bg_jobs=0, bg_skipped=0, bg_clipped=0, cp_items=0, cp_chunks=0, cp_layers=0,
            cp_sync_s=0.0, cp_issue_s=0.0, cp_pack_wait_s=0.0, cp_scatter_s=0.0)
        self.backlog: List[int] = []
        self._probe_list: List[list] = []
        self._cp_events: List[tuple] = []

    # ------------------------------------------------------------------------------------------------ small helpers
    def _on(self, stream):
        return torch.cuda.stream(stream) if (self.cuda and stream is not None) else contextlib.nullcontext()

    def _record(self, stream=None, timing: bool = False):
        if not self.cuda:
            return _Done()
        ev = torch.cuda.Event(enable_timing=timing)
        ev.record(stream if stream is not None else torch.cuda.current_stream(self.dev))
        return ev

    def _wait(self, stream, ev) -> None:
        """`stream` (None: the current stream) waits `ev` on the device; CPU mode: the host waits it."""
        if ev is None:
            return
        if self.cuda:
            (stream if stream is not None else torch.cuda.current_stream(self.dev)).wait_event(ev)
        elif not ev_done(ev):
            self.host_wait(ev)

    def _submitted(self, seq: int) -> bool:
        return int(self.tp.npipe.submitted) >= int(seq)

    def _stage4(self, st: torch.Tensor, l: int) -> torch.Tensor:
        """The layer's staging rows as the (B, lanes * 64, H, D) SOURCE of the scatter kernels: batch stride 0, so
        load id j of any (b, h) reads rows j * 64 .. j * 64 + 63 of lane h."""
        R = self.lanes * self.bs
        return st[l * R:(l + 1) * R].unsqueeze(0).expand(self.B, -1, -1, -1)

    def _bstage3(self, l: int) -> torch.Tensor:
        R = self.lanes * self.bs
        return self.bstage[l * R:(l + 1) * R].unsqueeze(0).expand(self.B, -1, -1)

    def _fill_bias(self, l: int, src: torch.Tensor, b, h, X, j) -> None:
        """The bias rows of the items, from the GPU-resident total_cis `src` (B, rows, H), into the layer's bias
        staging lanes: bstage[j * 64 + r, h] = src[b, X * 64 + r, h]. No PCIe."""
        R = self.lanes * self.bs
        bst = self.bstage[l * R:(l + 1) * R].view(self.lanes, self.bs, self.H)
        r = self._ar
        bst[j[:, None], r, h[:, None]] = src[b[:, None], X[:, None] * self.bs + r, h[:, None]].to(bst.dtype)

    def _mark_reader(self, l: int, ev) -> None:
        """The staging lanes of layer l were read up to `ev`: the transport's next job for layer l waits it (T3)."""
        mk = self._mark_next
        self._mark_next += 1
        self.tp.mark_reader(mk, ev)
        self._rmark[l] = mk

    # ------------------------------------------------------------------------------------------------ attach / detach
    def attach(self) -> None:
        if self.attached:
            raise RuntimeError("pool_prefetch: already attached")
        if self.fetch == "cpupack" and self._derived is None:
            ce_module = inspect.getmodule(type(self.engines[0]))
            self._derived, self.derived_source = derive_cpupack(ce_module)
        for l, e in enumerate(self.engines):
            shipped = e.__dict__["decode_update"]
            self._saved[l] = shipped
            if self.fetch == "cpupack":
                e._pf_fetch = functools.partial(self._fetch_cpupack, l, e)
                base = types.MethodType(self._derived, e)
            else:
                base = shipped
            e.decode_update = self._wrap(l, e, base)
        self.attached = True
        self.reset_stats()

    def detach(self) -> None:
        """Drains the transport, commits every pending job, waits the device, and puts back the shipped binding."""
        if not self.attached:
            return
        self.tp.flush()
        for job in self.pending:
            self._commit(job)
        self.pending = []
        self.busy = [False] * self.L
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        for l, e in enumerate(self.engines):
            e.decode_update = self._saved[l]
            if "_pf_fetch" in e.__dict__:
                del e._pf_fetch
            self._saved[l] = None
            self.commit_ev[l] = None
        self.attached = False

    def close(self) -> None:
        self.detach()
        self.tp.close()

    def _wrap(self, l: int, e, base):
        def decode_update(key_states, value_states, kv_bias, topk_idx):
            self._before(l, e)
            out = base(key_states, value_states, kv_bias, topk_idx)
            self._after(l, e)
            return out
        decode_update.pool_prefetch = self                     # the marker the tests read
        decode_update.layer = l
        return decode_update

    # ------------------------------------------------------------------------------------------------ per-layer hooks
    def _before(self, l: int, e) -> None:
        if self._probe_list and self._probe_list[-1][2] is None:
            self._probe_poll()
        ev = self.commit_ev[l]
        if ev is not None:                                     # the layer's commit lands before its pool_update reads
            self._wait(None, ev)
            self.commit_ev[l] = None
        if self.count and self.pinned[l]:
            torch.ge(e._pool_age, PIN_BASE, out=self._pinb[l])

    def _after(self, l: int, e) -> None:
        if self.count and self.pinned[l]:
            a = e._pool_action
            was = torch.gather(self._pinb[l], 2, e._pool_target.clamp(min=0)) & (a != POOL_NONE)
            self.counters[4].add_((was & ((a == POOL_SWAP) | (a == POOL_MOVE_IN))).sum())
            self.counters[5].add_((was & (a == POOL_MOVE_OUT)).sum())
        self.reader_ev[l] = self._record()                     # after the layer's pool ops: the slots' last reader
        if int(e._tail_block_len_on_gpu) == 0:                 # this update wrote the tail block back to the host
            mk = self._hist_next
            self._hist_next += 1
            self.tp.mark_hist(mk, self._record())
            self._hist_mark = mk

    # ------------------------------------------------------------------------------------------------ the step
    def begin_step(self, capture: bool = False) -> None:
        """Call before each decode step. Commits the submitted jobs (on the commit stream), samples the backlog and
        sends the background job. capture=True (a re-warm step: the shipped decode captures graphs in it) drains the
        transport first and sends no job in this step."""
        if not self.attached:
            raise RuntimeError("pool_prefetch: begin_step while detached")
        self.tp.poll()                                         # a worker error raises here
        self._capture = bool(capture)
        if self._capture:
            self.tp.flush()
        self._committed = [j for j in self._committed if not ev_done(j.commit_ev)]
        keep = []
        for job in self.pending:
            if self._submitted(job.seq):
                self._commit(job)
            else:
                job.deferred += 1
                self.stats["deferred"] += 1
                keep.append(job)
        self.pending = keep
        self.busy = [False] * self.L
        for job in keep:
            for l in job.items:
                self.busy[l] = True
        if self._last_seq:
            self.backlog.append(self._last_seq - int(self.tp.landed_upto()))
        self.lane_next.zero_()
        if self.bg_mib > 0 and not self._capture:
            self._issue_bg()
        self.stats["steps"] += 1

    def prefetch(self, items: torch.Tensor) -> int:
        """Use 1: items (n, 4) int64 CPU rows (layer, KV head, request, host block id). Every block must be complete
        on the host (block id < seq_length // 64 of its layer). Returns the number of items sent; the commit happens
        at the next begin_step."""
        if not self.attached:
            raise RuntimeError("pool_prefetch: prefetch while detached")
        if self.fetch == "cpupack":
            raise RuntimeError("pool_prefetch: --fetch cpupack refuses prefetch() (separate arms)")
        if self._capture:
            raise RuntimeError("pool_prefetch: prefetch in a capture step (the worker must stay idle)")
        items = torch.as_tensor(items, dtype=torch.int64).reshape(-1, 4)
        if items.shape[0] == 0:
            return 0
        l, h, b, X = items.unbind(1)
        done = torch.tensor([int(e.seq_length) // self.bs for e in self.engines], dtype=torch.int64)
        if bool(((l < 0) | (l >= self.L) | (h < 0) | (h >= self.H) | (b < 0) | (b >= self.B)).any()):
            raise ValueError("pool_prefetch: an item outside (L %d, H %d, B %d)" % (self.L, self.H, self.B))
        if bool(((X < 0) | (X >= done[l])).any()):
            raise ValueError("pool_prefetch: an item's block is not complete on the host")
        return self._issue(items, commit=True, hist=bool((X >= self.nb_safe).any()))

    def _issue(self, items: torch.Tensor, commit: bool, hist: bool) -> int:
        """ONE descriptor job of `items` (n, 4) [l, h, b, X]: duplicates removed, busy layers refused (commit),
        a lane per item (refused past `lanes`), then issue_desc. commit: the job is committed at the next step."""
        n0 = int(items.shape[0])
        L, H, B = self.L, self.H, self.B
        nblk = max(int(e._k_cpu.shape[1]) for e in self.engines) // self.bs + 1
        key = ((items[:, 0] * H + items[:, 1]) * B + items[:, 2]) * nblk + items[:, 3]
        uniq, inv = torch.unique(key, return_inverse=True)
        first = torch.full((uniq.numel(),), n0, dtype=torch.int64).scatter_reduce_(
            0, inv, torch.arange(n0, dtype=torch.int64), "amin")
        items = items[first.sort().values]
        self.stats["dup_items"] += n0 - int(items.shape[0])
        if commit and any(self.busy):
            busy = torch.tensor(self.busy, dtype=torch.bool)[items[:, 0]]
            self.stats["busy_refused"] += int(busy.sum())
            items = items[~busy]
        if items.shape[0] == 0:
            return 0
        g = items[:, 0] * H + items[:, 1]
        order = torch.argsort(g, stable=True)
        items, g = items[order], g[order]
        cnt = torch.bincount(g, minlength=L * H)
        start = torch.cumsum(cnt, 0) - cnt
        lane = self.lane_next.view(-1)[g] + torch.arange(items.shape[0], dtype=torch.int64) - start[g]
        ok = lane < self.lanes
        self.stats["lane_refused"] += int((~ok).sum())
        items, lane, g = items[ok], lane[ok], g[ok]
        n = int(items.shape[0])
        if n == 0:
            return 0
        self.lane_next.view(-1).index_add_(0, g, torch.ones_like(g))
        desc = torch.stack([items[:, 0], items[:, 2], items[:, 1], items[:, 3], lane,
                            torch.zeros_like(lane)]).contiguous()
        seq = int(self.tp.issue_desc(desc, reader_marks=list(self._rmark),
                                     hist_mark=self._hist_mark if hist else -1))
        self._last_seq = seq
        self.stats["jobs"] += 1
        self.stats["items_issued"] += n
        self.stats["bytes_issued"] += n * self.item_bytes
        if self.probes:
            self._probe_list.append([seq, self._record(self._ps_issue, timing=True), None, False])
        if commit:
            per: Dict[int, torch.Tensor] = {}
            for l in torch.unique(items[:, 0]).tolist():
                m = items[:, 0] == l
                it, ln = items[m], lane[m]
                k = it[:, 1] * B + it[:, 2]                    # (h, b) groups, in order (sorted by h already)
                o = torch.argsort(k, stable=True)
                it, ln, k = it[o], ln[o], k[o]
                _, counts = torch.unique_consecutive(k, return_counts=True)
                gs = (torch.cumsum(counts, 0) - counts).repeat_interleave(counts)
                rows = torch.stack([it[:, 1], it[:, 2], it[:, 3], ln, gs]).contiguous()
                per[int(l)] = rows.pin_memory() if self.cuda else rows
            self.pending.append(_Job(seq, per))
            for l in per:
                self.busy[l] = True
        return n

    def _issue_bg(self) -> None:
        """The background load of one step: bg_mib of K + V, the same number of items per (layer, KV head), random
        requests and random prefill blocks (seeded). bg_commit "none": the job ends in the staging buffer."""
        per = int(self.bg_mib * (1 << 20)) // self.item_bytes // (self.L * self.H)
        if per <= 0:
            return
        if per > self.lanes:
            self.stats["bg_clipped"] += 1
            per = self.lanes
        if self._last_seq and self._last_seq - int(self.tp.landed_upto()) >= self.backlog_cap:
            self.stats["bg_skipped"] += 1
            return
        n = self.L * self.H * per
        l = torch.arange(self.L, dtype=torch.int64).repeat_interleave(self.H * per)
        h = torch.arange(self.H, dtype=torch.int64).repeat_interleave(per).repeat(self.L)
        b = torch.randint(self.B, (n,), generator=self._bg_gen, dtype=torch.int64)
        X = torch.randint(self.nb_safe, (n,), generator=self._bg_gen, dtype=torch.int64)
        if self._issue(torch.stack([l, h, b, X], 1), commit=self.bg_commit == "pool", hist=False):
            self.stats["bg_jobs"] += 1

    # ------------------------------------------------------------------------------------------------ the commit
    def _commit(self, job: _Job) -> None:
        with self._on(self._cs):
            self.tp.job_ev(job.seq).wait(self._cs)             # the arrival (CPU mode: the host waits the job)
            ev = None
            for l in sorted(job.items):
                ev = self._commit_layer(job, l)
            job.commit_ev = ev
        self._committed.append(job)

    def _commit_layer(self, job: _Job, l: int):
        e, bs, H, B, P = self.engines[l], self.bs, self.H, self.B, self.P
        if self.total_cis[l] is None:
            raise RuntimeError("pool_prefetch: layer %d has no total_cis (the bias source)" % l)
        if not self.skip_reader_wait:                          # T3: after the last reader of the layer's pool slots
            self._wait(self._cs, self.reader_ev[l])
        d = job.items[l].to(self.dev, non_blocking=True)
        h, b, X, j, gs = d[0], d[1], d[2], d[3], d[4]
        n = int(d.shape[1])
        inw = (e._block_map[h, b] == X[:, None]).any(1)
        inp = (e._pool_map[h, b] == X[:, None]).any(1)
        ok = ~(inw | inp)
        okl = ok.long()
        excl = torch.cumsum(okl, 0) - okl
        rank = excl - excl[gs]                                 # the item's rank among the kept items of its row
        age = e._pool_age
        srt = torch.argsort(age, dim=2)                        # ascending: empty, then real stamps, then pins
        unp = (age < PIN_BASE).sum(2)
        keep = ok & (rank < unp[h, b])
        q = srt[h, b, rank.clamp(0, P - 1)]
        qd = torch.where(keep, q, torch.full_like(q, P))       # P: a dummy column for the dropped items
        ids = torch.full((H, B, P + 1), -1, dtype=torch.int64, device=self.dev)
        ids[h, b, qd] = torch.where(keep, j, torch.full_like(j, -1))
        ids = ids[:, :, :P].contiguous()
        self._fill_bias(l, self.total_cis[l], b, h, X, j)
        r0 = self.topk * bs
        self.ops.h2d(e._k_gpu[:, r0:], self._stage4(self.stage_k, l), ids, bs)
        self.ops.h2d_bias(e._v_gpu[:, r0:], self._stage4(self.stage_v, l), e._kv_bias_gpu[:, r0:],
                          self._bstage3(l), ids, bs)
        pm = torch.cat([e._pool_map, e._pool_map.new_full((H, B, 1), -1)], 2)
        pm[h, b, qd] = X
        e._pool_map.copy_(pm[:, :, :P])
        pa = torch.cat([age, age.new_zeros((H, B, 1))], 2)
        pa[h, b, qd] = PIN_BASE + self.pin_seq + torch.arange(n, dtype=torch.int64, device=self.dev)
        e._pool_age.copy_(pa[:, :, :P])
        self.pin_seq += n
        c = self.counters
        c[0].add_(n)
        c[1].add_(keep.sum())
        c[2].add_((~ok).sum())
        c[3].add_((ok & ~keep).sum())
        if self._pinb[l] is None:
            self._pinb[l] = torch.zeros((H, B, P), dtype=torch.bool, device=self.dev)
        self.pinned[l] = True
        ev = self._record(self._cs)
        self.commit_ev[l] = ev
        self._mark_reader(l, ev)
        return ev

    # ------------------------------------------------------------------------------------------------ use 2: cpupack
    def _fetch_cpupack(self, l: int, e, kv_bias: torch.Tensor) -> None:
        """The step's own misses (the shipped _load_mask after pool_update) through the CPU path, on the critical
        path. Called by the derived engine method in place of the two shipped gathers."""
        H, B, M, bs = self.H, self.B, self.topk, self.bs
        st = self.stats
        t0 = time.perf_counter()
        ev0 = self._record(timing=True) if self.cuda else None
        if self.cuda:
            mh = self._mask_host[st["cp_layers"] % 2]
            mh.copy_(e._load_mask, non_blocking=True)
            self._record().synchronize()                       # the CPU path cannot start before the mask exists
        else:
            mh = e._load_mask.clone()
        t1 = time.perf_counter()
        st["cp_sync_s"] += t1 - t0
        st["cp_layers"] += 1
        hh, bb, mm = (mh >= 0).nonzero(as_tuple=True)
        n = int(hh.numel())
        if n == 0:
            if self.cuda:
                self._cp_events.append((ev0, self._record(timing=True)))
            return
        X = mh[hh, bb, mm]
        o = torch.argsort(hh, stable=True)
        hh, bb, mm, X = hh[o], bb[o], mm[o], X[o]
        cnt = torch.bincount(hh, minlength=H)
        rank = torch.arange(n, dtype=torch.int64) - (torch.cumsum(cnt, 0) - cnt)[hh]
        chunk, lane = rank // self.lanes, rank % self.lanes
        dev = self.dev
        for c in range(int(chunk.max()) + 1):
            s = chunk == c
            hc, bc, mc, Xc, jc = hh[s], bb[s], mm[s], X[s], lane[s]
            t2 = time.perf_counter()
            desc = torch.stack([torch.full_like(hc, l), bc, hc, Xc, jc, torch.zeros_like(jc)]).contiguous()
            seq = int(self.tp.issue_desc(desc, reader_marks=list(self._rmark), hist_mark=-1))
            ids = torch.full((H, B, M), -1, dtype=torch.int64)
            ids[hc, bc, mc] = jc
            it = torch.stack([bc, hc, Xc, jc])
            if self.cuda:
                ids, it = ids.pin_memory(), it.pin_memory()
            ids_d, it_d = ids.to(dev, non_blocking=True), it.to(dev, non_blocking=True)
            self._fill_bias(l, kv_bias, it_d[0], it_d[1], it_d[2], it_d[3])
            t3 = time.perf_counter()
            self.tp.job_ev(seq).wait(torch.cuda.current_stream(dev) if self.cuda else None)
            t4 = time.perf_counter()
            self.ops.h2d(e._k_gpu, self._stage4(self.stage_k, l), ids_d, bs)
            self.ops.h2d_bias(e._v_gpu, self._stage4(self.stage_v, l), e._kv_bias_gpu, self._bstage3(l), ids_d, bs)
            self._mark_reader(l, self._record())
            t5 = time.perf_counter()
            st["cp_issue_s"] += t3 - t2
            st["cp_pack_wait_s"] += t4 - t3
            st["cp_scatter_s"] += t5 - t4
            st["cp_chunks"] += 1
            self._last_seq = seq
            st["jobs"] += 1
        st["cp_items"] += n
        st["items_issued"] += n
        st["bytes_issued"] += n * self.item_bytes
        if self.cuda:
            self._cp_events.append((ev0, self._record(timing=True)))

    # ------------------------------------------------------------------------------------------------ probes, report
    def _probe_poll(self) -> None:
        """Delivery probes: once a job is submitted, the arrival stream waits its event and records a timing event.
        late = the job had landed already when we looked (the sample is then an upper bound)."""
        for p in self._probe_list:
            if p[2] is not None or not self._submitted(p[0]):
                continue
            p[3] = int(self.tp.landed_upto()) >= p[0]
            self.tp.job_ev(p[0]).wait(self._ps_arrive)
            p[2] = self._record(self._ps_arrive, timing=True)

    def pins_live(self) -> int:
        return sum(int((e._pool_age >= PIN_BASE).sum()) for e in self.engines)

    def report(self) -> Dict[str, Any]:
        """Synchronizes. The cumulative counters (+ pins_live and the conservation check), this arm's host statistics,
        the delivery probes (ms, issue to arrival), the cpupack fetch time per layer (ms) and the worker's stats."""
        if self.attached:
            self._probe_poll()
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        c = dict(zip(COUNTERS, (int(x) for x in self.counters.tolist())))
        c["pins_live"] = self.pins_live()
        c["conserved"] = c["committed"] == c["pin_hits"] + c["pin_evicted"] + c["pins_live"]
        dl = [(round(p[1].elapsed_time(p[2]), 4), bool(p[3])) for p in self._probe_list if p[2] is not None]
        cp = [round(a.elapsed_time(b), 4) for a, b in self._cp_events]
        try:
            tps = {k: v for k, v in dict(self.tp.stats).items()}
        except Exception as ex:                                # the instrument never stops the cell
            tps = dict(error="%s: %s" % (type(ex).__name__, ex))
        return dict(counters=c, stats=dict(self.stats), backlog=list(self.backlog), delivery_ms=dl,
                    cp_fetch_ms_per_layer=cp, transport=tps, mode=dict(fetch=self.fetch, bg_mib=self.bg_mib,
                                                                        bg_commit=self.bg_commit),
                    geometry=dict(L=self.L, B=self.B, H=self.H, D=self.D, topk=self.topk, P=self.P,
                                  lanes=self.lanes, threads=self.threads, cores=self.cores,
                                  staging_bytes=self.staging_bytes, item_bytes=self.item_bytes,
                                  nb_safe=self.nb_safe))
