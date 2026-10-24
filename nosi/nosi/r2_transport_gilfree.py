"""R2 step 7, piece 5: the "gilfree" TRANSPORT (decision D5) -- the adapter of agent E's GIL-free CPU packer v2
(nosi-sustained-gilfree2 f2b981a, benchmarks/Efficiency/gilfree_pack.{py,cpp}: module gilfree_pack_v2, NativePipe) behind
K3X's transport interface (kstep/transfer.py:8-10): issue(layer, transfers), poll(), done(tr), force(tr).

ONE issue(layer, transfers) = ONE NativePipe.release of the layer's transfers: the extension packs host rows into its
pinned staging ring with an OpenMP team of set_threads(N) threads, GIL released, copies each chunk to the landing ring
on the COPY stream (copy engine) and scatters it on the SCATTER stream into the destination rows of the layer's store.
Completion: an event recorded on the scatter stream after the release (every transfer of the issue completes on it).

THE GEOMETRY (FG / SG5 run on holder 2190876, 2026-10-21: "S and S_dst must be multiples of R"; E's R contract,
FIX_DESIGN.md at f2b981a). The packer reads source row (b*S + blk*R + r)*H + h of a contiguous TOKEN-MAJOR (B, S, H, D)
source and writes destination row (b*S_dst + m*R + r)*H + h of a (rows, D) view of a token-major store with the same
B, H, D and dtype; it REFUSES S % R != 0 or S_dst % R != 0 (gilfree_pack.cpp order_and_check; R = 64), and its error
does not say which side.
- The production SOURCE side violated it. The host rows are token-major (B, n_host, H, D) (R2Body.allocate,
  r2_decode_body.py), with n_host = S_max + horizon + 64 (r2_decode_path.ragged_widths): 16128 + 119 + 64 = 16311 in
  SG5 p1 (horizon 48 + LAMBDA 7 + 64), 16128 + 160 + 64 = 16352 in FG (horizon 64 + 16 + 80). Neither is a multiple
  of 64. (Not 16128 + 64: n_host holds the horizon too.)
- The DESTINATION side holds: the store is (B, (C + X + 4 [+ 1 trash]) * 64, H, D) (TickEngine), whole 64-row slots.
- FIX: the adapter hands the packer ONE SOURCE PER (layer, request): the zero-copy view host[l][b:b+1, :S_floor],
  S_floor = (S_host // 64) * 64. It is contiguous (its leading dimension has size 1) and holds every complete block of
  the host rows. A block past S_floor is never complete, so it is never history and never fetched; the adapter
  refuses it at release. A transfer (b, h, slot, block) of layer l becomes the group (b' = 0, h, l' = l * B + b,
  m' = b * n_slots + slot, blk = block). With the source's B = 1 the packer sees S_dst = B * rows_per_request, and
  (0 * S_dst + m' * R + r)*H + h = (b * rows_per_request + slot * R + r)*H + h, the transfer's own destination rows.
- CONSTRUCTION CHECKS (each names its side): the host rows and the store are 4-D token-major tensors of the same B, H,
  D and dtype (a head-major (B, H, S, D) host is refused here, not read as S = H); the host holds at least one
  complete block; the store's rows per request are a multiple of 64 (the tail rings and the trash slot are whole
  slots). The release's "layers" stat counts the sources it read ((layer, request) pairs).
- [xpool] (xpool=r2_xpool.XPool, the shared-pool layout, opt-in): the destination is the FLAT store (M * 64, H, D); a
  transfer's group is (b' = 0, h, l' = l * B + b, m' = Transfer.phys, blk), so (m' * R + r) * H + h is lane h of its
  position (S_dst = M * 64; the packer's m' < M check is the lane bound). The duplicate check is per (h, lane).
  [integ] Both workers: _check holds the flat checks; worker="cpp" submits the flat store as a (1, M * 64, H, D) view
  (the same S_dst); issue_desc (logical slots computed in C++) refuses the pool.

THE ORDER RULES of K3X's transfers, kept here:
  T4  hist_ev (the tick's write-back of the block's host rows): the packer reads host rows only after it, so the
      release HOST-WAITS every transfer's hist_ev first (in the feeder thread, never the tick's).
  T3  reader_ev (the destination slot's previous reader): on the GPU the scatter stream waits on it (a device wait);
      on the CPU (tests: synchronous copies) the release host-waits it.
THE TEAM (FIX_DESIGN.md, kstep-cpu-runs/analysis/sustained_v2): the extension must be the OpenMP build (ext.OPENMP 1)
when more than one thread is asked; every release reports the team it used (threads_min / threads_max) and the adapter
REFUSES a release whose team differs from the request (v1's defect: a serial loop, unseen until its stats were read).
The OpenMP team is created by the calling thread and inherits its CPU affinity: with background=True the adapter runs
the releases in ONE feeder thread pinned to `cores` (os.sched_setaffinity) before its first release.
THE FEEDER THREAD'S MODES (SG5 p2 on holder 2190876, 2026-10-21: "Inplace update to inference tensor outside
InferenceMode is not allowed", raised by the packer's index_copy_ into the store in the feeder). torch's inference mode,
grad mode and current CUDA device are THREAD-LOCAL; a new thread starts with the defaults (inference mode off).
- The feeder enters, around each release, the inference mode that the release's in-place writes need: on when the
  constructing thread was in inference mode, or any store, host or ring tensor is an inference tensor, or the attached
  generation board is one (attach_board; the publication writes the board in place). It enters the constructing
  thread's grad mode (off under inference mode) and, once at its start on the GPU, the store's device.
- Every CUDA operation of a release names its stream (the extension's copy and scatter streams; the publication under
  torch.cuda.stream(scatter); scatter.wait_event), so the thread's current stream is never used.
- The OpenMP team's threads copy raw bytes only (pack_range: no tensor operation, no dispatcher state).
- thread_modes=False (NEGATIVE CONTROL, tests only): the feeder keeps the thread's defaults.
- The synchronous mode (background False) runs in the caller's thread with the caller's modes.
THE C++ WORKER (worker="cpp"; Q-C2, 2026-10-21; the Python feeder stays the DEFAULT, worker="python"). It needs packer
v3 (feature/nosi-r2-gilfree-worker benchmarks/Efficiency/gilfree_pack.{py,cpp}: NativePipe.submit). issue() does the
glue on the CALLING thread (the checks, the index lists, the board pairs, the T4 / T3 events' raw cudaEvent_t handles;
in CPU mode it host-waits T4 / T3 itself) and queues ONE job; a std::thread inside NativePipe runs it with no Python and
no GIL: T4 by cudaEventSynchronize, T3 by cudaStreamWaitEvent on the scatter stream, the v2 release, the board
publication (one index_put_ per chunk on the scatter stream after its rows), the job's completion event. The worker
enters the modes above (inference, grad, device) ONCE, when it starts (at the first issue; attach_board must come
before). Every transfer of a job gets tr.ev = _JobEv at issue: query / synchronize, and torch's Stream.wait_event
protocol (wait(stream): the host waits the submission, then the stream waits on the device). submits_without_gil = True
tells S-C (CudaDeviceOps._submitted, C's named change) that its exact wait needs no host wait for the submission.
THE DESCRIPTOR ISSUE (Q-B3; worker="cpp"; issue(layer, transfers) is unchanged for every current gate). issue_desc(
desc, reader_marks, hist_mark) takes a whole tick's takes as ONE (6, n) int64 tensor (l, b, h, block, slot, gen: B2's
take list) and queues ONE job: C++ applies the adapter's geometry and checks, releases layer by layer, and publishes
the board; no Python object exists per transfer. T3 / T4 become marks: the tick registers its per-layer reader events
(mark_reader(tick * L + layer, ev)) and its write-back event (mark_hist(tick, ev)); all are recorded on the main
stream, so they complete in mark order, and the job waits only the newest mark per layer (T3, a device wait before the
layer's scatters) and the newest write-back (T4, a host wait in the worker). The host admits the takes of every job
seq <= landed_upto() (the job watermark: it only grows; a failed job never lands). S-C's exact wait reads the board.
THE WARM-UP (gfw1e on holder 2190876, job record /mnt/beegfs/ojerbi/r2-runs/gfw/gfw_gfw1e_2190876/summary.json, source
372fe55). GFW1's captured protocol (no submission wait, 64 launches behind the spin) in a fresh process: with
CUDA_MODULE_LOADING=EAGER it still stalled (err 1, spin 7255 ms, enqueue 9.2 ms); with lazy loading and, before the
spin, one add_ on the main stream and one float32 index_copy_ + one int64 index_put_ on the scatter stream, it passed
(err 0, spin 143 ms, enqueue 0.89 ms). So the stall comes from the FIRST CALL of those operations behind the spin, not
from module loading. The constructor therefore calls warm_up() ONCE, before any issue (so before the first tick, in the
eager and the captured tick alike): the same operations, on tiny tensors, on the store's device and dtype (index_copy_
of the store's dtype into a (rows, D) tensor with an int64 index, the C++ worker's and v2's scatter; index_put_ of int64
values with an int64 index into a 1-D int64 tensor, the board publication), each on the stream production uses (add_ on
the constructing thread's current stream; index_copy_ / index_put_ on the scatter stream), then a device synchronize. On
the CPU the same operations run on CPU tensors (no stream, no synchronize). warm_up=False (NEGATIVE CONTROL, tests and
gfw1f's control arm only) skips it. warm_calls counts the calls; warmed names what ran (None when skipped). The host
still waits the submission in both tick modes (S-C's _submitted, 9508109): the warm-up does not replace that rule.
"""
from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .kstep.backend import ev_done
from .kstep.transfer import Transfer

STATS = ("groups", "chunks", "layers", "pack_ns", "wait_ns", "submit_ns", "total_ns", "threads_min", "threads_max")


class _Done:
    """The completion of a CPU-mode release (synchronous copies)."""
    done = True

    def query(self) -> bool:
        return True


class _JobEv:
    """The completion of one C++ worker job (worker="cpp"; every transfer of the job holds it from issue on).
    query(): the job ran without error and (CUDA) its completion event, recorded by the worker on the scatter stream, is
    done. synchronize(): waits for both (the GIL released). wait(stream): torch's Stream.wait_event protocol
    (stream.wait_event(ev) calls ev.wait(stream)): the host waits for the job's SUBMISSION (the GIL released), then the
    stream waits for the job's event on the device. S-C's exact wait uses none of these: it spins on the board."""
    __slots__ = ("tp", "seq", "_landed")

    def __init__(self, tp, seq: int):
        self.tp, self.seq, self._landed = tp, int(seq), False

    def query(self) -> bool:
        if not self._landed:                                   # poll_arrivals asks per transfer and per poll
            self._landed = bool(self.tp.npipe.job_done(self.seq))
        return self._landed

    def synchronize(self) -> None:
        self.tp.npipe.wait_job(self.seq)

    def wait_submitted(self) -> None:
        """The host waits (the GIL released) until the job's GPU work is ENQUEUED, not until it landed."""
        self.tp.npipe.wait_submission(self.seq)

    def wait(self, stream=None) -> None:
        if not self.tp.cuda:
            self.tp.npipe.wait_job(self.seq)
            return
        if stream is None:
            stream = torch.cuda.current_stream()
        self.tp.npipe.stream_wait_job(self.seq, [int(stream.stream_id), int(stream.device_index),
                                                 int(stream.device_type)])


WORKERS = ("python", "cpp")


class GilfreeTransport:
    name = "gilfree"
    # [inflight-array] every transfer of ONE issue() completes on ONE event (the Python worker's release event, the C++
    # worker's job event), and the worker sets tr.ev in the list's order: so the last transfer's done() is the issue's
    # (OwnResidency(inflight_array=True) asks one done per issue)
    one_event_per_issue = True
    # [hostfix] --host-profile (OPT-IN; None: no clock is read): a list that gets one record per release of the
    # Python worker (_release, the feeder thread when background): (layer, transfers, python ns, native ns, the clock
    # at its end). python ns = the release's wall time minus the native pack call (npipe.release, which runs the C++
    # team): the feeder's Python part (the waits, _check, the lists, _publish, the event), which holds the GIL.
    prof = None

    def __init__(self, ext, host_k: Sequence[torch.Tensor], host_v: Sequence[torch.Tensor], slots_k, slots_v, bs: int,
                 D: int, threads: int, host_wait: Callable[[Any], None], cap_groups: int = 256, ring: int = 2,
                 cores: Optional[Sequence[int]] = None, background: bool = False, skip_hist_wait: bool = False,
                 thread_modes: bool = True, xpool=None, worker: str = "python", warm_up: bool = True):
        if not 1 <= int(threads) <= 256:
            raise ValueError("threads must be in 1 .. 256 (got %r)" % (threads,))
        if int(threads) > 1 and int(getattr(ext, "OPENMP", 0)) != 1:
            raise RuntimeError("gilfree: a team of %d threads asked from an extension built without OpenMP" % threads)
        if worker not in WORKERS:
            raise ValueError("worker must be one of %s (got %r)" % (WORKERS, worker))
        if worker == "cpp" and not callable(getattr(ext.NativePipe, "submit", None)):
            raise RuntimeError("gilfree: worker='cpp' needs packer v3 (NativePipe.submit; "
                               "feature/nosi-r2-gilfree-worker gilfree_pack.cpp)")
        self.worker = worker
        self.submits_without_gil = worker == "cpp"     # S-C's exact wait: no host wait for the submission (Q-C2)
        self.ext = ext
        self.host_k, self.host_v, self.slots_k, self.slots_v = list(host_k), list(host_v), slots_k, slots_v
        self.bs, self.D, self.threads = int(bs), int(D), int(threads)
        self.host_wait = host_wait
        self.skip_hist_wait = bool(skip_hist_wait)             # NEGATIVE CONTROL (tests only): T4 not honoured
        dev = slots_k[0].device
        self.cuda = dev.type == "cuda"
        elem = self.host_k[0].element_size()
        extra = int(getattr(ext, "SLOT_EXTRA_PER_GROUP", 0))       # packer v3: the board pairs (v2: 0)
        nb = cap_groups * self.bs * 8 + cap_groups * extra + 2 * cap_groups * self.bs * self.D * elem   # [rows | K | V]
        stage = torch.zeros((ring, nb), dtype=torch.uint8, pin_memory=self.cuda)
        land = torch.zeros((ring, nb), dtype=torch.uint8, device=dev)
        if self.cuda:
            self.copy, self.scatter = torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)
            trip = lambda s: [int(s.stream_id), int(s.device_index), int(s.device_type)]   # noqa: E731
            cs, ss = trip(self.copy), trip(self.scatter)
        else:
            self.copy = self.scatter = None
            cs, ss = [], []
        # the geometry (module docstring, E's R contract): token-major sources and store; one source per (layer,
        # request), the complete blocks of the host rows; every refusal names its side
        hk0, sk0 = self.host_k[0], slots_k[0]
        if hk0.dim() != 4 or any(t.shape != hk0.shape or t.dtype != hk0.dtype for t in self.host_k + self.host_v):
            raise ValueError("gilfree SOURCE: the host rows must be 4-D (B, S, H, D) tensors of one shape and dtype "
                             "(got %s)" % (tuple(hk0.shape),))
        self.xpool = xpool
        if xpool is not None:
            # [xpool] the flat store (M * 64, H, D) of r2_xpool: the packer's destination row (0 * S_dst + m' * R + r)
            # * H + h with S_dst = M * 64 (the source's B is 1) is lane h of position m' = Transfer.phys
            if sk0.dim() != 3 or any(t.shape != sk0.shape or t.dtype != sk0.dtype for t in list(slots_k) + list(slots_v)):
                raise ValueError("gilfree DESTINATION (xpool): the store must be flat (M * 64, H, D) tensors of one shape "
                                 "and dtype (got %s)" % (tuple(sk0.shape),))
            self.B, S_host, Hh, Dh = (int(x) for x in hk0.shape)
            rows = int(sk0.shape[0])
            if rows != xpool.M * self.bs or (int(sk0.shape[1]), int(sk0.shape[2])) != (Hh, Dh) or Dh != self.D \
                    or sk0.dtype != hk0.dtype or xpool.B != self.B or xpool.H != Hh:
                raise ValueError("gilfree (xpool): the host rows %s and the flat store %s must share H, D (D %d) and "
                                 "dtype, with M * %d = %d rows and the xpool's B, H"
                                 % (tuple(hk0.shape), tuple(sk0.shape), self.D, self.bs, xpool.M * self.bs))
        elif sk0.dim() != 4 or any(t.shape != sk0.shape or t.dtype != sk0.dtype for t in list(slots_k) + list(slots_v)):
            raise ValueError("gilfree DESTINATION: the store must be 4-D (B, rows, H, D) tensors of one shape and dtype "
                             "(got %s)" % (tuple(sk0.shape),))
        else:
            self.B, S_host, Hh, Dh = (int(x) for x in hk0.shape)
            rows = int(sk0.shape[1])
        if xpool is None and ((int(sk0.shape[0]), int(sk0.shape[2]), int(sk0.shape[3])) != (self.B, Hh, Dh)
                              or Dh != self.D or sk0.dtype != hk0.dtype):
            raise ValueError("gilfree: the host rows %s (SOURCE) and the store %s (DESTINATION) must both be TOKEN-MAJOR "
                             "(B, S, H, D) with the same B, H, D (D %d) and dtype (%s, %s); a head-major (B, H, S, D) "
                             "tensor is refused" % (tuple(hk0.shape), tuple(sk0.shape), self.D, hk0.dtype, sk0.dtype))
        self.S_floor = (S_host // self.bs) * self.bs
        if self.S_floor < self.bs:
            raise ValueError("gilfree SOURCE: the host rows (%d per request) hold no complete block of %d"
                             % (S_host, self.bs))
        if rows % self.bs:
            raise ValueError("gilfree DESTINATION: the store's rows per request (%d) are not a multiple of %d (a tail "
                             "ring or a trash slot that is not a whole slot)" % (rows, self.bs))
        self.n_slots = rows // self.bs
        self.nblk = self.S_floor // self.bs
        src_k = [hk[b:b + 1, :self.S_floor] for hk in self.host_k for b in range(self.B)]
        src_v = [hv[b:b + 1, :self.S_floor] for hv in self.host_v for b in range(self.B)]
        self.npipe = ext.NativePipe(stage, land, int(cap_groups), self.bs, cs, ss)
        self.npipe.set_sources(src_k, src_v)
        self.npipe.set_threads(self.threads)
        self._stats: Dict[str, int] = {k: 0 for k in STATS[:7]}
        self._stats.update(releases=0, threads_min=None, threads_max=None, published=0)   # published: (slot, gen) pairs
        self.staging_bytes = 2 * ring * nb
        self.background = bool(background) or worker == "cpp"   # the C++ worker is always asynchronous
        self._worker_modes: Optional[tuple] = None             # (inference, grad) the C++ worker started with
        self._keep: Dict[int, list] = {}                       # job seq -> the Python events its raw handles name
        self._last_seq = 0
        # Q-B3 (issue_desc): the registered events per kind, mark -> event; every mark below the floor was registered,
        # completed and dropped; the window (in marks) kept beyond the newest
        self._marks: Dict[str, Dict[int, Any]] = dict(reader={}, hist={})
        self._mark_floor: Dict[str, int] = dict(reader=0, hist=0)
        self._mark_window: Dict[str, int] = dict(reader=4 * len(self.slots_k), hist=4)
        self._stores_set = False
        self._last_flag: Optional[threading.Event] = None
        self.board = None                                      # the generation board (attach_board), or None
        self._error: Optional[BaseException] = None
        # the feeder thread's modes (module docstring): thread-local in torch, so recorded here and entered there
        self.thread_inference = bool(torch.is_inference_mode_enabled()) or any(
            t.is_inference() for t in list(slots_k) + list(slots_v) + self.host_k + self.host_v + [stage, land])
        self.thread_grad = False if self.thread_inference else bool(torch.is_grad_enabled())
        self.thread_device = dev if self.cuda else None
        self.thread_modes = bool(thread_modes)                 # NEGATIVE CONTROL (tests only) when False
        self._feeder_seen: Optional[Dict[str, bool]] = None    # the modes the feeder ran with (feeder_seen)
        self._cores = list(cores) if cores else None           # the feeder's (or the C++ worker's) CPU affinity
        self.warm_calls = 0                                    # THE WARM-UP (module docstring): once, before any issue
        self.warmed: Optional[str] = None
        if warm_up:                                            # False: NEGATIVE CONTROL (tests, gfw1f's control arm)
            self.warm_up()
        if self.background and worker == "python":
            self._q: "queue.Queue" = queue.Queue()
            self._thread = threading.Thread(target=self._feeder, name="gilfree-feeder", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------------------------------------ the warm-up
    WARM_OPS = "add_ (current stream); index_copy_ %s + index_put_ int64 (scatter)"

    def warm_up(self) -> None:
        """THE WARM-UP (module docstring; gfw1e's warm arm): runs once each operation that the releases use behind S-C's
        spin, on tiny tensors of the production device and dtypes, each on its production stream, then synchronizes
        the device. The constructor calls it once, before any issue; it refuses a second call or a call after the
        first issue (the warm-up must precede the first submission)."""
        if self.warm_calls:
            raise RuntimeError("gilfree: warm_up runs once (at construction)")
        if self._last_seq or self._stats.get("releases") or self._last_flag is not None or self._worker_modes:
            raise RuntimeError("gilfree: warm_up after the first issue")
        dev, dt, bs, D = self.slots_k[0].device, self.slots_k[0].dtype, self.bs, self.D
        x = torch.zeros((), device=dev)
        x.add_(1.0)                                            # the main stream's kernel (gfw1e: the launches)
        ctx = torch.cuda.stream(self.scatter) if self.cuda else contextlib.nullcontext()
        with ctx:                                              # the scatter stream: the worker's / v2's scatter kernels
            d = torch.zeros(2 * bs, D, dtype=dt, device=dev)
            d.index_copy_(0, torch.arange(bs, dtype=torch.int64, device=dev), torch.ones(bs, D, dtype=dt, device=dev))
            b = torch.zeros(8, dtype=torch.int64, device=dev)
            b.index_put_((torch.tensor([1, 2], dtype=torch.int64, device=dev),),
                         torch.tensor([3, 4], dtype=torch.int64, device=dev))
        if self.cuda:
            torch.cuda.synchronize(dev)                        # loaded and run before the first tick
        self.warm_calls += 1
        self.warmed = self.WARM_OPS % (str(dt).replace("torch.", ""),)

    # ------------------------------------------------------------------------------------------------ the feeder
    def _feeder(self) -> None:
        if self._cores:
            os.sched_setaffinity(0, self._cores)               # the OpenMP team inherits this thread's affinity
        if self.thread_modes and self.thread_device is not None:
            torch.cuda.set_device(self.thread_device)
        while True:
            item = self._q.get()
            if item is None:
                return
            layer, transfers, flag = item
            try:
                if not self.thread_modes:                      # NEGATIVE CONTROL: the new thread's defaults
                    self._feed_one(layer, transfers)
                else:                                          # read per release: attach_board may have changed it
                    with torch.inference_mode(self.thread_inference), torch.set_grad_enabled(self.thread_grad):
                        self._feed_one(layer, transfers)
            except BaseException as e:                         # reported at the next issue / force
                self._error = e
            finally:
                flag.set()

    def _feed_one(self, layer: int, transfers: List[Transfer]) -> None:
        self.feeder_seen = dict(inference=bool(torch.is_inference_mode_enabled()), grad=bool(torch.is_grad_enabled()))
        self._release(layer, transfers)

    def attach_board(self, board) -> None:
        """The exact device wait (residency note 12f): after each release, (slot, generation) of its transfers is
        published into the board (board.t, board.flat) on the stream that landed the rows, AFTER them: the scatter
        stream on the GPU (a pinned copy and an index_put_ enqueued there), at once on the CPU. The publication is an
        in-place write in the feeder thread: an inference-mode board turns the feeder's inference mode on (read at its
        next release; the queue orders this write before that read). worker="cpp": the worker publishes (in C++, the
        same order) and enters its modes once, at its start: attach_board after that start must not change them."""
        if getattr(board.t, "is_inference", lambda: False)():
            if self._worker_modes is not None and self.thread_modes and not self._worker_modes[0]:
                raise RuntimeError("gilfree: attach_board of an inference-mode board after the C++ worker started "
                                   "outside inference mode (its modes are entered once): attach the board first")
            self.thread_inference, self.thread_grad = True, False
        self.board = board
        if self.worker == "cpp":
            self.npipe.set_board(board.t)

    def _publish(self, transfers: List[Transfer]) -> None:
        if self.board is None or not transfers:
            return
        host = torch.tensor([[self.board.flat(tr.layer, tr.b, tr.h, tr.slot) for tr in transfers],
                             [tr.gen for tr in transfers]], dtype=torch.int64)
        if self.cuda:
            with torch.cuda.stream(self.scatter):
                d = host.pin_memory().to(self.board.t.device, non_blocking=True)
                self.board.t.view(-1).index_put_((d[0],), d[1])
        else:
            self.board.t.view(-1).index_put_((host[0],), host[1])
        self.stats["published"] += len(transfers)

    @property
    def stats(self) -> Dict[str, Any]:
        """The release stats (STATS sums, releases, threads_min / threads_max, published); worker="cpp": read from the
        worker's totals (the jobs that ran without error) at each access."""
        if self.worker == "cpp":
            self._sync_worker()
        return self._stats

    def _sync_worker(self) -> None:
        w = [int(x) for x in self.npipe.worker_stats()]
        for k, v in zip(STATS[:7], w[:7]):
            self._stats[k] = v
        self._stats["threads_min"] = None if w[7] < 0 else w[7]
        self._stats["threads_max"] = None if w[8] < 0 else w[8]
        self._stats["published"] = w[9]
        self._stats["releases"] = w[10]

    @property
    def feeder_seen(self) -> Optional[Dict[str, bool]]:
        """The modes the feeder thread ran its last release with (tests, reports): the Python feeder records them per
        release (_feed_one); worker="cpp": the C++ worker thread's OWN state, read in that thread at its start and
        before each job (NativePipe.modes_seen). None before the first release (cpp: before the worker started)."""
        if self.worker == "cpp":
            inf, grad = (int(x) for x in self.npipe.modes_seen())
            return None if inf < 0 else dict(inference=bool(inf), grad=bool(grad))
        return self._feeder_seen

    @feeder_seen.setter
    def feeder_seen(self, value: Optional[Dict[str, bool]]) -> None:
        self._feeder_seen = value

    def threads_ok(self) -> bool:
        """Agent E's rule (gilfree_pack.threads_ok): every chunk so far ran on exactly the requested team."""
        st = self.stats
        return st["threads_min"] == self.threads and st["threads_max"] == self.threads

    def close(self) -> None:
        if self.worker == "cpp":
            if self._worker_modes is not None:
                self.npipe.stop_worker()                       # drains the queued jobs, then joins
            self._sync_worker()
            self._keep.clear()
        elif self.background:
            self._q.put(None)
            self._thread.join()

    def wait_submitted(self, tr: Transfer) -> None:
        """THE EAGER TICK'S LIVENESS RULE (gfw1b on holder 2190876): a main stream may spin on the board (S-C's exact
        wait) only behind GPU work that is already ENQUEUED. A transfer whose job is submitted after the spin can be
        blocked behind it: the main thread's launches behind the spin fill the launch queue (or the streams share a
        hardware queue), the worker's later H2D / scatter cannot run, and the spin ends only at its bound. So, before
        such a spin, the host waits here until the transfer's job is SUBMITTED: the C++ worker's counter (the GIL
        released: no GIL deadlock), the Python feeder's flag; nothing in the synchronous mode. It does not wait for
        the landing (the device spin does that)."""
        if self.worker == "cpp":
            if tr.ev is None:
                raise RuntimeError("gilfree: transfer seq %d was never released" % tr.seq)
            tr.ev.wait_submitted()
        elif self.background and isinstance(tr.payload, threading.Event):
            tr.payload.wait()

    def flush(self) -> None:
        """Waits until every issued release ran (the tests' sync point; a no-op in the synchronous mode)."""
        if self.worker == "cpp":
            if self._last_seq:
                self.npipe.wait_job(self._last_seq)
        elif self.background and self._last_flag is not None:
            self._last_flag.wait()

    # ------------------------------------------------------------------------------------------------ the C++ worker
    def _start_worker(self) -> None:
        if self._worker_modes is not None:
            return
        inf, grad = (self.thread_inference, self.thread_grad) if self.thread_modes else (False, True)  # NEGATIVE CONTROL
        dev = -1
        if self.cuda:
            dev = self.thread_device.index if self.thread_device.index is not None else torch.cuda.current_device()
        self.npipe.start_worker(bool(inf), bool(grad), int(dev), [int(c) for c in (self._cores or [])])
        self._worker_modes = (bool(inf), bool(grad))

    def _reap(self) -> None:
        if self._keep:
            done = int(self.npipe.submitted)
            for seq in [s for s in self._keep if s <= done]:
                del self._keep[seq]

    def _submit(self, layer: int, transfers: List[Transfer]) -> None:
        """worker="cpp": the glue on this thread, ONE job for the C++ worker (module docstring, THE C++ WORKER)."""
        hist, readers, keep = [], [], []
        for tr in transfers:
            for ev, out, skip in ((tr.hist_ev, hist, self.skip_hist_wait), (tr.reader_ev, readers, False)):
                if ev is None or skip or ev_done(ev):
                    continue
                handle = int(getattr(ev, "cuda_event", 0) or 0) if self.cuda else 0
                if handle:
                    out.append(handle)                         # T4: cudaEventSynchronize; T3: cudaStreamWaitEvent
                    keep.append(ev)                            # alive until the worker used its handle
                else:
                    self.host_wait(ev)                         # CPU mode (synchronous copies): waited here
        self._check(layer, transfers)
        bf, bg = [], []
        if self.board is not None:
            bf = [self.board.flat(tr.layer, tr.b, tr.h, tr.slot) for tr in transfers]
            bg = [int(tr.gen) for tr in transfers]
        self._start_worker()
        flat = self.xpool is not None                          # [integ] xpool on the C++ worker: m' = the lane; the
        sk, sv = self.slots_k[layer], self.slots_v[layer]      # flat (M * 64, H, D) store as a (1, M * 64, H, D) view
        if flat:                                               # (submit checks 4-D; release_impl reads view(-1, D), so
            sk, sv = sk.unsqueeze(0), sv.unsqueeze(0)          # S_dst = M * 64 exactly as the Python worker's release)
        seq = self.npipe.submit([0] * len(transfers), [tr.h for tr in transfers],
                                [layer * self.B + tr.b for tr in transfers],
                                [tr.phys if flat else tr.b * self.n_slots + tr.slot for tr in transfers],
                                [tr.block for tr in transfers], sk, sv, bf, bg, hist, readers)
        ev = _JobEv(self, seq)
        for tr in transfers:
            tr.ev = ev
            tr.payload = None
        if keep:
            self._keep[seq] = keep
        self._last_seq = int(seq)

    # ------------------------------------------------------------------------------------------------ Q-B3: descriptors
    def mark_reader(self, mark: int, ev) -> None:
        """Registers the tick's reader event of one layer: mark = tick * L + layer (ResidencyBackend.after_attention's
        event). Marks grow; all reader events are on the main stream, so they complete in mark order."""
        self._mark("reader", mark, ev)

    def mark_hist(self, mark: int, ev) -> None:
        """Registers the tick's write-back event: mark = the tick (its last commit_block event; main stream)."""
        self._mark("hist", mark, ev)

    def _mark(self, kind: str, mark: int, ev) -> None:
        ring = self._marks[kind]
        mark = int(mark)
        if mark < self._mark_floor[kind] or (ring and mark <= max(ring)):
            raise ValueError("gilfree: %s mark %d does not grow (newest %s, floor %d)"
                             % (kind, mark, max(ring) if ring else None, self._mark_floor[kind]))
        ring[mark] = ev
        lim = mark - self._mark_window[kind]
        for mk in sorted(ring):                                # drop the old marks whose events completed, in order
            if mk >= lim or not ev_done(ring[mk]):
                break
            del ring[mk]
            self._mark_floor[kind] = mk + 1

    def _mark_handle(self, kind: str, mark: int, keep: list) -> int:
        """The raw cudaEvent_t to wait for `mark` (0: nothing to wait); CPU mode: host-waits it here."""
        mark = int(mark)
        if mark < 0 or mark < self._mark_floor[kind]:
            return 0                                           # none, or registered, completed and dropped
        ev = self._marks[kind].get(mark)
        if ev is None:
            raise ValueError("gilfree: %s mark %d was never registered (mark_%s)" % (kind, mark, kind))
        if ev_done(ev):
            return 0
        handle = int(getattr(ev, "cuda_event", 0) or 0) if self.cuda else 0
        if handle:
            keep.append(ev)                                    # alive until the worker used its handle
            return handle
        self.host_wait(ev)                                     # CPU mode (synchronous copies): waited here
        return 0

    def issue_desc(self, desc: torch.Tensor, reader_marks=None, hist_mark: int = -1) -> int:
        """Q-B3 (worker="cpp"): a whole tick's takes as ONE job, no Python object per transfer. desc: (6, n) int64 CPU
        rows l, b, h, block, slot, gen (B2's take list); reader_marks: per layer the newest mark (tick * L + layer) of
        its destinations' last readers (-1: none); hist_mark: the newest write-back tick of the job's host rows (-1:
        none). C++ builds the pack plan (the adapter's geometry and checks) and publishes the board; returns the job's
        seq (0: no take). The host admits the takes of every job seq <= landed_upto()."""
        if self.worker != "cpp" or not int(getattr(self.ext, "DESC", 0)):
            raise RuntimeError("gilfree: issue_desc needs worker='cpp' and packer v3 with DESC (submit_desc)")
        if self.xpool is not None:                             # [integ] C++ computes m' = b * slots + slot (logical)
            raise ValueError("gilfree: issue_desc does not translate xpool lanes (the descriptor names logical slots); "
                             "use issue() with Transfer.phys")
        if (not torch.is_tensor(desc) or desc.dtype != torch.int64 or desc.dim() != 2 or desc.shape[0] != 6
                or desc.device.type != "cpu"):
            raise ValueError("gilfree: desc must be a (6, n) int64 CPU tensor (l, b, h, block, slot, gen)")
        self._raise_pending_error()
        self._reap()
        if desc.shape[1] == 0:
            return 0
        keep: list = []
        readers = []
        if reader_marks is not None:
            marks = [int(x) for x in torch.as_tensor(reader_marks).reshape(-1).tolist()]
            if len(marks) != len(self.slots_k):
                raise ValueError("gilfree: reader_marks needs one mark per layer (%d)" % len(self.slots_k))
            readers = [self._mark_handle("reader", mk, keep) for mk in marks]
        hist = 0 if self.skip_hist_wait else self._mark_handle("hist", hist_mark, keep)
        self._start_worker()
        if not self._stores_set:
            self.npipe.set_stores(list(self.slots_k), list(self.slots_v))
            self._stores_set = True
        seq = int(self.npipe.submit_desc(desc.contiguous(), readers, int(hist)))
        if keep:
            self._keep[seq] = keep
        self._last_seq = seq
        return seq

    def landed_upto(self) -> int:
        """Q-B3: the job watermark (worker="cpp"): every job up to this seq ran without error and its rows landed
        (CUDA: its completion event on the scatter stream is done). It only grows; a failed job never lands."""
        if self.worker != "cpp":
            raise RuntimeError("gilfree: landed_upto needs worker='cpp'")
        return int(self.npipe.landed_upto())

    def job_ev(self, seq: int) -> "_JobEv":
        """[b2-desc] The completion object of descriptor job `seq` (worker="cpp"), the one that issue() gives every
        transfer of a job: a caller that keeps one Transfer per take (OwnResidency(issue="desc")) sets tr.ev to it, so
        done / force / wait_submitted / Stream.wait_event work per job as on the object path."""
        if self.worker != "cpp" or int(seq) < 1:
            raise RuntimeError("gilfree: job_ev needs worker='cpp' and a job seq >= 1 (got %r)" % (seq,))
        return _JobEv(self, int(seq))

    # ------------------------------------------------------------------------------------------------ one release
    def _check(self, layer: int, transfers: List[Transfer]) -> None:
        """One destination per transfer, every transfer inside the geometry (both workers). [integ] xpool's flat-store
        checks (feaa7d6's _release) moved here, so the C++ worker's _submit gets them too."""
        dst = set()
        flat = self.xpool is not None                          # [xpool] the destination is the transfer's lane
        for tr in transfers:
            d = (tr.h, tr.phys) if flat else (tr.b, tr.h, tr.slot)
            if d in dst:
                raise ValueError("two transfers into one destination %s (b %d h %d slot %d)" % (d, tr.b, tr.h, tr.slot))
            dst.add(d)
            if flat and (tr.phys is None or not 0 <= tr.phys < self.xpool.A):   # the history arena (review m2)
                raise ValueError("gilfree (xpool): transfer seq %d (b %d h %d slot %d) has lane %r outside the history "
                                 "arena [0, %d)" % (tr.seq, tr.b, tr.h, tr.slot, tr.phys, self.xpool.A))
            if not (0 <= tr.block < self.nblk and (flat or 0 <= tr.slot < self.n_slots) and 0 <= tr.b < self.B):
                raise ValueError("gilfree: transfer seq %d (b %d, block %d, slot %d) outside the geometry (B %d, %d "
                                 "complete host blocks, %d slots)" % (tr.seq, tr.b, tr.block, tr.slot, self.B,
                                                                     self.nblk, self.n_slots))

    def _release(self, layer: int, transfers: List[Transfer]) -> None:
        prof = self.prof                                       # [hostfix] --host-profile (None: off)
        if prof is not None:
            t_start = time.perf_counter_ns()
        if not self.skip_hist_wait:
            for tr in transfers:                               # T4: the host rows exist before the pack reads them
                if tr.hist_ev is not None and not ev_done(tr.hist_ev):
                    self.host_wait(tr.hist_ev)
        for tr in transfers:                                   # T3: the scatter writes a slot after its last reader
            if tr.reader_ev is not None and not ev_done(tr.reader_ev):
                if self.cuda:
                    self.scatter.wait_event(tr.reader_ev)
                else:
                    self.host_wait(tr.reader_ev)
        self._check(layer, transfers)
        flat = self.xpool is not None                          # [xpool] the destination is the transfer's lane
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64)   # noqa: E731
        cols = (i64([0] * len(transfers)), i64([tr.h for tr in transfers]),
                i64([layer * self.B + tr.b for tr in transfers]),
                i64([tr.phys if flat else tr.b * self.n_slots + tr.slot for tr in transfers]),
                i64([tr.block for tr in transfers]), self.slots_k[layer].view(-1, self.D),
                self.slots_v[layer].view(-1, self.D))
        if prof is not None:
            t_nat = time.perf_counter_ns()
        st = self.npipe.release(*cols)
        if prof is not None:
            t_nat = time.perf_counter_ns() - t_nat
        st = [int(x) for x in st]
        if st[1] > 0 and not (st[7] == st[8] == self.threads):
            raise RuntimeError("gilfree: packing teams of %d .. %d threads != requested %d" % (st[7], st[8], self.threads))
        for k, v in zip(STATS[:7], st[:7]):
            self.stats[k] += v
        self.stats["releases"] += 1
        if st[1] > 0:
            lo, hi = self.stats["threads_min"], self.stats["threads_max"]
            self.stats["threads_min"] = st[7] if lo is None else min(lo, st[7])
            self.stats["threads_max"] = st[8] if hi is None else max(hi, st[8])
        self._publish(transfers)                               # after the rows, on the scatter stream (exact wait)
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record(self.scatter)
        else:
            ev = _Done()
        for tr in transfers:
            tr.ev = ev
        if prof is not None:                                   # [hostfix] (list.append: atomic under the GIL)
            t_end = time.perf_counter_ns()
            prof.append((int(layer), len(transfers), t_end - t_start - t_nat, t_nat, t_end))

    # ------------------------------------------------------------------------------------------------ K3X interface
    def _raise_pending_error(self) -> None:
        if self.worker == "cpp":
            msg = self.npipe.take_error()
            if msg:
                raise RuntimeError(msg)
            return
        if self._error is not None:
            e, self._error = self._error, None
            raise e

    def issue(self, layer: int, transfers: List[Transfer]) -> None:
        if not transfers:
            return
        self._raise_pending_error()
        if self.worker == "cpp":
            self._reap()
            self._submit(layer, transfers)
        elif self.background:
            flag = threading.Event()
            for tr in transfers:
                tr.payload = flag                              # force() waits for the feeder to reach this release
            self._last_flag = flag
            self._q.put((layer, list(transfers), flag))
        else:
            self._release(layer, transfers)

    def poll(self) -> None:
        self._raise_pending_error()
        if self.worker == "cpp":
            self._reap()

    def done(self, tr: Transfer) -> bool:
        return tr.ev is not None and ev_done(tr.ev)

    def force(self, tr: Transfer) -> None:
        if self.worker == "cpp":
            if tr.ev is None:
                raise RuntimeError("gilfree: transfer seq %d was never released" % tr.seq)
            tr.ev.synchronize()                                # the submission, then (CUDA) the landing; GIL released
            self._raise_pending_error()
            return
        if self.background and isinstance(tr.payload, threading.Event):
            tr.payload.wait()
        self._raise_pending_error()
        if tr.ev is None:
            raise RuntimeError("gilfree: transfer seq %d was never released" % tr.seq)
        if self.cuda:
            tr.ev.synchronize()
