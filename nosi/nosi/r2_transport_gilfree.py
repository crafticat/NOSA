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
- The feeder enters, around its whole loop, the inference mode that the release's in-place writes need: on when the
  constructing thread was in inference mode or any store, host or ring tensor is an inference tensor. It enters the
  constructing thread's grad mode (off under inference mode) and, on the GPU, the store's device.
- Every CUDA operation of a release names its stream (the extension's copy and scatter streams; the publication under
  torch.cuda.stream(scatter); scatter.wait_event), so the thread's current stream is never used.
- The OpenMP team's threads copy raw bytes only (pack_range: no tensor operation, no dispatcher state).
- thread_modes=False (NEGATIVE CONTROL, tests only): the feeder keeps the thread's defaults.
- The synchronous mode (background False) runs in the caller's thread with the caller's modes.
"""
from __future__ import annotations

import os
import queue
import threading
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


class GilfreeTransport:
    name = "gilfree"

    def __init__(self, ext, host_k: Sequence[torch.Tensor], host_v: Sequence[torch.Tensor], slots_k, slots_v, bs: int,
                 D: int, threads: int, host_wait: Callable[[Any], None], cap_groups: int = 256, ring: int = 2,
                 cores: Optional[Sequence[int]] = None, background: bool = False, skip_hist_wait: bool = False,
                 thread_modes: bool = True):
        if not 1 <= int(threads) <= 256:
            raise ValueError("threads must be in 1 .. 256 (got %r)" % (threads,))
        if int(threads) > 1 and int(getattr(ext, "OPENMP", 0)) != 1:
            raise RuntimeError("gilfree: a team of %d threads asked from an extension built without OpenMP" % threads)
        self.ext = ext
        self.host_k, self.host_v, self.slots_k, self.slots_v = list(host_k), list(host_v), slots_k, slots_v
        self.bs, self.D, self.threads = int(bs), int(D), int(threads)
        self.host_wait = host_wait
        self.skip_hist_wait = bool(skip_hist_wait)             # NEGATIVE CONTROL (tests only): T4 not honoured
        dev = slots_k[0].device
        self.cuda = dev.type == "cuda"
        elem = self.host_k[0].element_size()
        nb = cap_groups * self.bs * 8 + 2 * cap_groups * self.bs * self.D * elem   # [destination rows | K | V]
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
        if sk0.dim() != 4 or any(t.shape != sk0.shape or t.dtype != sk0.dtype for t in list(slots_k) + list(slots_v)):
            raise ValueError("gilfree DESTINATION: the store must be 4-D (B, rows, H, D) tensors of one shape and dtype "
                             "(got %s)" % (tuple(sk0.shape),))
        self.B, S_host, Hh, Dh = (int(x) for x in hk0.shape)
        rows = int(sk0.shape[1])
        if (int(sk0.shape[0]), int(sk0.shape[2]), int(sk0.shape[3])) != (self.B, Hh, Dh) or Dh != self.D \
                or sk0.dtype != hk0.dtype:
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
        self.stats: Dict[str, int] = {k: 0 for k in STATS[:7]}
        self.stats.update(releases=0, threads_min=None, threads_max=None, published=0)   # published: (slot, gen) pairs
        self.staging_bytes = 2 * ring * nb
        self.background = bool(background)
        self.board = None                                      # the generation board (attach_board), or None
        self._error: Optional[BaseException] = None
        # the feeder thread's modes (module docstring): thread-local in torch, so recorded here and entered there
        self.thread_inference = bool(torch.is_inference_mode_enabled()) or any(
            t.is_inference() for t in list(slots_k) + list(slots_v) + self.host_k + self.host_v + [stage, land])
        self.thread_grad = False if self.thread_inference else bool(torch.is_grad_enabled())
        self.thread_device = dev if self.cuda else None
        self.thread_modes = bool(thread_modes)                 # NEGATIVE CONTROL (tests only) when False
        self.feeder_seen: Optional[Dict[str, bool]] = None     # the modes the feeder ran with (tests, reports)
        if self.background:
            self._q: "queue.Queue" = queue.Queue()
            self._cores = list(cores) if cores else None
            self._thread = threading.Thread(target=self._feeder, name="gilfree-feeder", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------------------------------------ the feeder
    def _feeder(self) -> None:
        if self._cores:
            os.sched_setaffinity(0, self._cores)               # the OpenMP team inherits this thread's affinity
        if not self.thread_modes:                              # NEGATIVE CONTROL: the new thread's defaults
            self._feed()
            return
        if self.thread_device is not None:
            torch.cuda.set_device(self.thread_device)
        with torch.inference_mode(self.thread_inference), torch.set_grad_enabled(self.thread_grad):
            self._feed()

    def _feed(self) -> None:
        self.feeder_seen = dict(inference=bool(torch.is_inference_mode_enabled()), grad=bool(torch.is_grad_enabled()))
        while True:
            item = self._q.get()
            if item is None:
                return
            layer, transfers, flag = item
            try:
                self._release(layer, transfers)
            except BaseException as e:                         # reported at the next issue / force
                self._error = e
            finally:
                flag.set()

    def attach_board(self, board) -> None:
        """The exact device wait (residency note 12f): after each release, (slot, generation) of its transfers is
        published into the board (board.t, board.flat) on the stream that landed the rows, AFTER them: the scatter
        stream on the GPU (a pinned copy and an index_put_ enqueued there), at once on the CPU."""
        self.board = board

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

    def threads_ok(self) -> bool:
        """Agent E's rule (gilfree_pack.threads_ok): every chunk so far ran on exactly the requested team."""
        return self.stats["threads_min"] == self.threads and self.stats["threads_max"] == self.threads

    def close(self) -> None:
        if self.background:
            self._q.put(None)
            self._thread.join()

    # ------------------------------------------------------------------------------------------------ one release
    def _release(self, layer: int, transfers: List[Transfer]) -> None:
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
        dst = set()
        for tr in transfers:
            if (tr.b, tr.h, tr.slot) in dst:
                raise ValueError("two transfers into one destination (b %d h %d slot %d)" % (tr.b, tr.h, tr.slot))
            dst.add((tr.b, tr.h, tr.slot))
            if not (0 <= tr.block < self.nblk and 0 <= tr.slot < self.n_slots and 0 <= tr.b < self.B):
                raise ValueError("gilfree: transfer seq %d (b %d, block %d, slot %d) outside the geometry (B %d, %d "
                                 "complete host blocks, %d slots)" % (tr.seq, tr.b, tr.block, tr.slot, self.B,
                                                                     self.nblk, self.n_slots))
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64)   # noqa: E731
        st = self.npipe.release(i64([0] * len(transfers)), i64([tr.h for tr in transfers]),
                                i64([layer * self.B + tr.b for tr in transfers]),
                                i64([tr.b * self.n_slots + tr.slot for tr in transfers]),
                                i64([tr.block for tr in transfers]), self.slots_k[layer].view(-1, self.D),
                                self.slots_v[layer].view(-1, self.D))
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

    # ------------------------------------------------------------------------------------------------ K3X interface
    def _raise_pending_error(self) -> None:
        if self._error is not None:
            e, self._error = self._error, None
            raise e

    def issue(self, layer: int, transfers: List[Transfer]) -> None:
        if not transfers:
            return
        self._raise_pending_error()
        if self.background:
            flag = threading.Event()
            for tr in transfers:
                tr.payload = flag                              # force() waits for the feeder to reach this release
            self._q.put((layer, list(transfers), flag))
        else:
            self._release(layer, transfers)

    def poll(self) -> None:
        self._raise_pending_error()

    def done(self, tr: Transfer) -> bool:
        return tr.ev is not None and ev_done(tr.ev)

    def force(self, tr: Transfer) -> None:
        if self.background and isinstance(tr.payload, threading.Event):
            tr.payload.wait()
        self._raise_pending_error()
        if tr.ev is None:
            raise RuntimeError("gilfree: transfer seq %d was never released" % tr.seq)
        if self.cuda:
            tr.ev.synchronize()
