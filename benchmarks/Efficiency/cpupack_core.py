"""CPU-packing transport: layouts, address formulas, descriptors, chunking, the pack -> bulk H2D -> GPU scatter pipeline,
references, canaries and layout accounting. Pure torch and device-agnostic: the SAME code runs on CPU tensors with the
deferred CpuBackend (retroinfer-eval tests/test_cpupack_transport.py) and on the GPU with CudaBackend (cpupack_transport.py).

LABELS: every result is 'transport/placement-ready, NOT live-LRU-ready' (no last-reader / version / publication mechanism
exists at dc818e4; plans are replayed into a SEPARATE scratch with the cache's geometry) and 'GPU-resident trace replay'.

THE MISS UNIT (verified at dc818e4): a group = 64 consecutive tokens (one block) of ONE independently selected KV head of one
request. NOSA-8B: H = 2 KV heads, D = 128, bf16. K and V each 64 x 128 x 2 B = 16 KiB per group; 32 KiB useful per group.
Plan = NOSI's per-layer load mask (H, B, 64): slot m of (head h, request b) receives host block plan[h, b, m]; -1 = no load;
slot 63 is the tail and is never loaded (cache_engine.py:666 diff_offload; :684-685 the shipped gathers consume it).

LAYOUTS (the PHYSICAL allocation of one per-layer K or V tensor; the logical tensor is always [B, S, H, D]):
  orig  physical [B, S, H, D]  (cache_engine.py:263-264 host, :295 window). One group = 64 rows of 256 B at a 512-B pitch.
  hm    physical [B, H, S, D]  (head-major). One group = ONE contiguous 16 KiB run. The logical tensor is the permuted VIEW
        phys.permute(0, 2, 1, 3) of the SAME storage (no copy). Converting a host cache is an in-place physical reorder of
        the same allocation (convert_inplace), never a permute of the existing bytes.
ROW index (256-B rows of the physical tensor viewed (-1, D)) of row r of group (b, h, block-or-slot j), S = rows per request:
  orig  (b*S + j*R + r)*H + h        (row step between consecutive r: H)
  hm    (b*H + h)*S + j*R + r        (row step: 1)
GROUP index (hm only; the (-1, R*D) view, S % R == 0):  (b*H + h)*(S // R) + j.
Every formula is int64 and is checked against the tensor's own strides (row_index_from_strides) in the CPU tests.

ORDER AND STAGING. Groups are taken in (b, h, m) order (request-major; within a group r = 0..63), so a request's groups are
contiguous and chunks are cut at request boundaries when the request fits. One staging slot of a pipe with capacity cap holds,
for a chunk of n groups, [int64 destination indices][K rows n*R x D][V rows n*R x D] as ONE contiguous byte region -> ONE H2D
copy per chunk. The index block ENDS at the fixed offset P0 = cap*R*8 and the payload STARTS there, so the index bytes of any
chunk always lie in [0, P0), a region that only ever holds indices (zeroed at every reset): even a transfer that reads a
slot too early (the lifetime negative controls) delivers VALID indices and wrong payload, never payload bytes read as indices
(on the GPU an out-of-range index_copy_ index is a device-side assert that would end the process). The row
packer (index_select of 256-B rows, 64 per group) and the group packer (index_select of 16 KiB groups; hm source only)
write IDENTICAL staging bytes; the row placer (index_copy_ of rows) and the group placer (index_copy_ of groups; hm
destination only) write identical destinations. So the packer/placer ALGORITHM change is separable from the LAYOUT change.

LIFETIMES (every reuse wait has a negative control: Faults, and the deferred CpuBackend makes a missing wait visible):
  staging slot s is overwritten by the host only after the H2D that read it completed (host_wait on its H2D-end event);
  landing slot s is overwritten by an H2D only after the scatter that read it completed (copy-stream wait on its scatter
  event); the scatter of a chunk reads its landing slot only after THAT chunk's H2D completed (scatter-stream wait on the
  H2D end event; negative control Faults.no_h2d_wait, visible with a scatter-first drain / a delayed copy stream).
  Poisoned scratch + canary rows detect a skipped scatter; a poisoned staged row detects content corruption.
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

H_DEF, D_DEF, R_DEF, M_DEF = 2, 128, 64, 64
TAIL_SLOT = 63
LAYOUTS = ("orig", "hm")
PACKERS = ("row", "group")
PLACERS = ("row", "group")
POISON_I16 = 0x7FC1                  # a bf16 NaN bit pattern: no finite K/V value equals it
IDX_BYTES = 8                        # int64 destination indices shipped with the chunk
LABEL = "transport/placement-ready, NOT live-LRU-ready"
REPLAY_LABEL = "GPU-resident trace replay (all 32 layer plans ready at the gate; not decode-paced)"
CEILING_LABEL = "DIAGNOSTIC CEILING (not a transport result)"


def group_bytes(R: int = R_DEF, D: int = D_DEF, elem: int = 2) -> int:
    return R * D * elem


def useful_bytes(n_groups: int, R: int = R_DEF, D: int = D_DEF, elem: int = 2) -> int:
    """K + V bytes of n groups (each group: one head, one request, one block)."""
    return 2 * int(n_groups) * group_bytes(R, D, elem)


# --------------------------------------------------------------------------------------------------------- layouts
def phys_shape(layout: str, B: int, S: int, H: int = H_DEF, D: int = D_DEF) -> Tuple[int, int, int, int]:
    if layout == "orig":
        return (B, S, H, D)
    if layout == "hm":
        return (B, H, S, D)
    raise ValueError("layout must be one of %s" % (LAYOUTS,))


def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= int(s)
    return tuple(reversed(st))


def logical_strides(layout: str, B: int, S: int, H: int = H_DEF, D: int = D_DEF) -> Tuple[int, int, int, int]:
    """Strides of the logical [B, S, H, D] view of a contiguous physical tensor of `layout`."""
    if layout == "orig":
        return contiguous_strides((B, S, H, D))
    sb, sh, ss, sd = contiguous_strides((B, H, S, D))
    return (sb, ss, sh, sd)


def alloc_phys(layout: str, B: int, S: int, H: int = H_DEF, D: int = D_DEF, dtype=torch.bfloat16, device="cpu", pin: bool = False):
    t = torch.empty(phys_shape(layout, B, S, H, D), dtype=dtype, device=device)
    return t.pin_memory() if pin else t


def logical(phys: torch.Tensor, layout: str) -> torch.Tensor:
    """The logical [B, S, H, D] view of a CONTIGUOUS physical tensor (no copy; the permute is a view)."""
    if not phys.is_contiguous():
        raise ValueError("the physical tensor must be contiguous")
    if layout == "orig":
        return phys
    if layout == "hm":
        return phys.permute(0, 2, 1, 3)
    raise ValueError(layout)


def phys_from_logical(log: torch.Tensor, layout: str) -> torch.Tensor:
    """The contiguous physical tensor that shares `log`'s storage, from its actual strides (torch.as_strided, never a copy
    and never a reshape). Refuses a view whose strides are not exactly the layout's."""
    B, S, H, D = log.shape
    if tuple(log.stride()) != logical_strides(layout, B, S, H, D):
        raise ValueError("strides %s are not the %s layout's %s" % (tuple(log.stride()), layout, logical_strides(layout, B, S, H, D)))
    ps = phys_shape(layout, B, S, H, D)
    phys = torch.as_strided(log, ps, contiguous_strides(ps), log.storage_offset())
    assert phys.data_ptr() == log.data_ptr() and phys.is_contiguous()
    return phys


def layout_of(log: torch.Tensor) -> str:
    """'orig' or 'hm' from the logical view's strides (raises otherwise)."""
    B, S, H, D = log.shape
    for lay in LAYOUTS:
        if tuple(log.stride()) == logical_strides(lay, B, S, H, D):
            return lay
    raise ValueError("strides %s match no layout" % (tuple(log.stride()),))


def rows2d(phys: torch.Tensor) -> torch.Tensor:
    """(-1, D) rows of a CONTIGUOUS physical tensor (a view; .view never copies and raises on a non-viewable tensor)."""
    if not phys.is_contiguous():
        raise ValueError("rows2d needs the contiguous physical tensor")
    v = phys.view(-1, phys.shape[-1])
    assert v.data_ptr() == phys.data_ptr()
    return v


def groups2d(phys: torch.Tensor, R: int = R_DEF) -> torch.Tensor:
    """(-1, R*D) groups of a CONTIGUOUS head-major physical tensor [B, H, S, D] (S % R == 0)."""
    if not phys.is_contiguous() or phys.dim() != 4:
        raise ValueError("groups2d needs the contiguous 4-D head-major physical tensor")
    if phys.shape[2] % R:
        raise ValueError("S=%d is not a multiple of R=%d" % (phys.shape[2], R))
    v = phys.view(-1, R * phys.shape[-1])
    assert v.data_ptr() == phys.data_ptr()
    return v


def row_base(layout: str, b: torch.Tensor, h: torch.Tensor, j: torch.Tensor, S: int, H: int = H_DEF, R: int = R_DEF):
    """(row index of r = 0, row step per r) of groups (b, h, j) in the physical (-1, D) view; int64."""
    b, h, j = (x.to(torch.int64) for x in (b, h, j))
    if layout == "orig":
        return (b * S + j * R) * H + h, H
    if layout == "hm":
        return (b * H + h) * S + j * R, 1
    raise ValueError(layout)


def row_index(layout: str, b, h, j, r, S: int, H: int = H_DEF, R: int = R_DEF) -> torch.Tensor:
    base, step = row_base(layout, torch.as_tensor(b), torch.as_tensor(h), torch.as_tensor(j), S, H, R)
    return base + step * torch.as_tensor(r, dtype=torch.int64)


def group_index(b, h, j, S: int, H: int = H_DEF, R: int = R_DEF) -> torch.Tensor:
    """Head-major only: the group's index in the (-1, R*D) view."""
    if S % R:
        raise ValueError("S=%d is not a multiple of R=%d" % (S, R))
    b, h, j = (torch.as_tensor(x).to(torch.int64) for x in (b, h, j))
    return (b * H + h) * (S // R) + j


def row_index_from_strides(log: torch.Tensor, b: int, t: int, h: int) -> int:
    """The 256-B row of logical element (b, t, h, 0), from the tensor's OWN strides and storage offset (the oracle the
    address formulas are tested against)."""
    D = log.shape[-1]
    off = b * log.stride(0) + t * log.stride(1) + h * log.stride(2)
    assert off % D == 0
    return off // D


# ----------------------------------------------------------------------------------------------------- descriptors
@dataclass
class Desc:
    n: int                                   # groups
    B: int
    b: torch.Tensor
    h: torch.Tensor
    m: torch.Tensor
    blk: torch.Tensor
    req_groups: torch.Tensor                 # (B,) groups per request, request order
    src_layout: str
    dst_layout: str
    packer: str
    placer: str
    src_idx: torch.Tensor                    # rows (n*R) for the row packer, groups (n) for the group packer
    dst_idx: torch.Tensor                    # rows (n*R) for the row placer, groups (n) for the group placer
    R: int = R_DEF

    @property
    def src_per_group(self) -> int:
        return self.R if self.packer == "row" else 1

    @property
    def dst_per_group(self) -> int:
        return self.R if self.placer == "row" else 1


def plan_groups(plan_hbm: torch.Tensor, tail_slot: Optional[int] = TAIL_SLOT):
    """(b, h, m, blk) int64 of every load of a (H, B, M) plan, in (b, h, m) order. A load into the tail slot raises."""
    if plan_hbm.dim() != 3:
        raise ValueError("plan must be (H, B, M)")
    pbm = plan_hbm.permute(1, 0, 2)                                   # (B, H, M) view: nonzero walks it in (b, h, m) order
    nz = (pbm >= 0).nonzero()
    b, h, m = nz[:, 0], nz[:, 1], nz[:, 2]
    blk = pbm[b, h, m].to(torch.int64)
    if tail_slot is not None and bool((m == tail_slot).any()):
        raise ValueError("a plan loads the tail slot %d" % tail_slot)
    return b, h, m, blk


def build_desc(plan_hbm: torch.Tensor, *, src_layout: str, dst_layout: str, s_src: int, s_dst: int, packer: str = "row",
               placer: str = "row", R: int = R_DEF, tail_slot: Optional[int] = TAIL_SLOT, check: bool = True) -> Desc:
    """CPU descriptor preparation for one layer's plan (the timed stage (2a) when run by the coordinator)."""
    if packer not in PACKERS or placer not in PLACERS:
        raise ValueError("packer/placer")
    if packer == "group" and src_layout != "hm":
        raise ValueError("the group packer needs a head-major source (a group is contiguous only there)")
    if placer == "group" and dst_layout != "hm":
        raise ValueError("the group placer needs a head-major destination")
    H, B, M = plan_hbm.shape
    b, h, m, blk = plan_groups(plan_hbm, tail_slot)
    n = int(b.numel())
    if check and n:
        if int(blk.min()) < 0 or int((blk.max() + 1) * R) > s_src:
            raise ValueError("a host block is past the source (S=%d)" % s_src)
        if int((m.max() + 1) * R) > s_dst:
            raise ValueError("a slot is past the destination (S=%d)" % s_dst)
    ar = torch.arange(R, dtype=torch.int64)
    if packer == "row":
        base, step = row_base(src_layout, b, h, blk, s_src, H, R)
        src = (base[:, None] + step * ar[None, :]).view(-1)
    else:
        src = group_index(b, h, blk, s_src, H, R)
    if placer == "row":
        base, step = row_base(dst_layout, b, h, m, s_dst, H, R)
        dst = (base[:, None] + step * ar[None, :]).view(-1)
    else:
        dst = group_index(b, h, m, s_dst, H, R)
    req = torch.bincount(b, minlength=B) if n else torch.zeros(B, dtype=torch.int64)
    return Desc(n=n, B=B, b=b, h=h, m=m, blk=blk, req_groups=req, src_layout=src_layout, dst_layout=dst_layout, packer=packer,
                placer=placer, src_idx=src, dst_idx=dst, R=R)


def check_unique_dst(desc: Desc) -> None:
    """Duplicate destinations race in the scatter (index_copy_ is nondeterministic on duplicates): refuse them."""
    if desc.n and int(torch.unique(desc.dst_idx).numel()) != int(desc.dst_idx.numel()):
        raise ValueError("duplicate destination rows")


def source_rows_logical(desc: Desc, R: int = R_DEF):
    """(b, t, h) logical coordinates of every source row, (b, h, m, r) order: the head of every row is its group's head."""
    ar = torch.arange(R, dtype=torch.int64)
    b = desc.b[:, None].expand(-1, R).reshape(-1)
    h = desc.h[:, None].expand(-1, R).reshape(-1)
    t = (desc.blk[:, None] * R + ar[None, :]).reshape(-1)
    return b, t, h


def dest_rows_logical(desc: Desc, R: int = R_DEF):
    ar = torch.arange(R, dtype=torch.int64)
    b = desc.b[:, None].expand(-1, R).reshape(-1)
    h = desc.h[:, None].expand(-1, R).reshape(-1)
    t = (desc.m[:, None] * R + ar[None, :]).reshape(-1)
    return b, t, h


# ------------------------------------------------------------------------------------------------------- chunking
def chunk_requests(req_groups: Sequence[int], cap_groups: int) -> List[Tuple[int, int]]:
    """Group ranges [g0, g1) of the chunks of one layer (groups in request order). Whole requests while they fit; a request
    that does not fit in the remaining capacity starts a new chunk; a request larger than the cap is split across
    consecutive chunks. Depends on the plan only, never on timing."""
    if cap_groups < 1:
        raise ValueError("cap_groups")
    out: List[Tuple[int, int]] = []
    g, start = 0, 0
    for c in (int(x) for x in req_groups):
        if c == 0:
            continue
        if g - start > 0 and (g - start) + c > cap_groups:
            out.append((start, g))
            start = g
        while (g + c) - start > cap_groups:                    # split an oversize request
            cut = start + cap_groups
            c -= cut - g
            g = cut
            out.append((start, g))
            start = g
        g += c
    if g > start:
        out.append((start, g))
    return out


def request_last_chunk(req_groups: Sequence[int], chunks: Sequence[Tuple[int, int]]) -> List[int]:
    """Per request: the index of the chunk that holds its LAST group (-1 for a zero-miss request). Chunks complete in order
    on one scatter stream, so that chunk's scatter end is the request's completion."""
    out, cum, ci = [], 0, 0
    for c in (int(x) for x in req_groups):
        if c == 0:
            out.append(-1)
            continue
        cum += c
        while not (chunks[ci][0] <= cum - 1 < chunks[ci][1]):
            ci += 1
        out.append(ci)
    return out


def idx_region(cap_groups: int, R: int = R_DEF) -> int:
    """P0: the index block of a pipe with capacity cap ends here and the payload starts here."""
    return cap_groups * R * IDX_BYTES


def slot_bytes(cap_groups: int, R: int = R_DEF, D: int = D_DEF, elem: int = 2) -> int:
    """Staging/landing slot size: the largest index block (row placer: R per group) + the K + V payload of cap groups."""
    return idx_region(cap_groups, R) + cap_groups * 2 * group_bytes(R, D, elem)


def stage_views(slot_u8: torch.Tensor, n: int, packer: str, placer: str, cap_groups: int, dtype=torch.bfloat16, R: int = R_DEF,
                D: int = D_DEF):
    """(k, v, idx, lo, hi) views of one slot for n groups of a pipe with capacity cap: idx int64 (n*R rows or n groups) in
    [P0 - ni*8, P0), K and V in [P0, P0 + 2*n*16K), shaped for the packer/placer ((n*R, D) rows or (n, R*D) groups -- the
    SAME bytes). The one H2D copies [lo, hi). Views only, no copy."""
    if n > cap_groups:
        raise ValueError("chunk of %d groups > cap %d" % (n, cap_groups))
    elem = torch.empty((), dtype=dtype).element_size()
    p0 = idx_region(cap_groups, R)
    gb = n * group_bytes(R, D, elem)
    shape = (n * R, D) if packer == "row" else (n, R * D)
    k = slot_u8[p0:p0 + gb].view(dtype).view(shape)
    v = slot_u8[p0 + gb:p0 + 2 * gb].view(dtype).view(shape)
    ni = n * R if placer == "row" else n
    idx = slot_u8[p0 - ni * IDX_BYTES:p0].view(torch.int64)
    return k, v, idx, p0 - ni * IDX_BYTES, p0 + 2 * gb


# -------------------------------------------------------------------------------------------------------- backends
class _Ev:
    __slots__ = ("stream", "done", "t_ns")

    def __init__(self, stream=None, done=False):
        self.stream, self.done, self.t_ns = stream, done, (time.perf_counter_ns() if done else None)


class _Stream:
    def __init__(self, name):
        self.name = name
        self.ops = deque()


class CpuBackend:
    """DEFERRED execution model of streams for the CPU tests: an operation enqueued on a stream runs only when something
    forces it (a host wait on a later event of that stream, a stream wait, or synchronize). That is the WORST case a GPU can
    produce for a missing lifetime wait, so a removed wait deterministically corrupts the result (the negative controls)."""

    def __init__(self, names=("copy", "scatter", "side", "plan"), drain=("copy", "scatter", "side", "plan")):
        self.s = {n: _Stream(n) for n in names}
        self.cur: Optional[str] = None
        self.n_ops = 0
        self.drain = tuple(drain)                              # synchronize()'s default stream order (see synchronize)

    @contextmanager
    def stream(self, name):
        prev, self.cur = self.cur, name
        try:
            yield
        finally:
            self.cur = prev

    def _q(self, name):
        return self.s[name if name is not None else self.cur]

    def event_rec(self, name=None):
        st = self._q(name)
        ev = _Ev(st)
        st.ops.append(("rec", ev))
        return ev

    def marker(self):
        return _Ev(None, done=True)

    def host_wait(self, ev):
        self._flush_until(ev)

    def stream_wait(self, name, ev):
        self._q(name).ops.append(("wait", ev))

    def memcpy(self, dst, src, name=None):
        self._q(name).ops.append(("fn", lambda: dst.copy_(src)))

    def index_copy(self, dst2d, idx, src, name=None):
        self._q(name).ops.append(("fn", lambda: dst2d.index_copy_(0, idx, src)))

    def sleep(self, ms, name=None):
        pass                                                   # deferral already models the delay

    def _run(self, st, until=None):
        while st.ops:
            kind, x = st.ops.popleft()
            if kind == "fn":
                x()
                self.n_ops += 1
            elif kind == "rec":
                x.done, x.t_ns = True, time.perf_counter_ns()
                if x is until:
                    return
            elif kind == "wait":
                self._flush_until(x)

    def _flush_until(self, ev):
        if ev is None or ev.done:
            return
        self._run(ev.stream, ev)

    def synchronize(self, order=None):
        """Drain every stream in `order` (default self.drain). Copy first makes a missing LANDING wait visible (the copy
        overwrites a landing slot before its scatter ran); scatter first makes a missing H2D wait of the scatter visible
        (the scatter reads its landing slot before the copy filled it). The tests run both orders."""
        for n in (self.drain if order is None else order):
            self._run(self.s[n])
        for st in self.s.values():
            self._run(st)


class CudaBackend:
    """The GPU mapping: named torch.cuda streams, timing events from a reusable pool, markers on an idle aux stream (an upper
    bound of the host instant, the hisparse_repro.make_bracket technique)."""

    def __init__(self, streams: Dict[str, "torch.cuda.Stream"], aux: "torch.cuda.Stream", pool_size: int = 4096, clock_hz: float = 1.41e9):
        self.s = dict(streams)
        self.aux = aux
        self.pool: List = [torch.cuda.Event(enable_timing=True) for _ in range(pool_size)]
        self.k = 0
        self.clock_hz = clock_hz

    def reset(self):
        self.k = 0

    def _ev(self):
        if self.k == len(self.pool):
            self.pool.append(torch.cuda.Event(enable_timing=True))
        e = self.pool[self.k]
        self.k += 1
        return e

    @contextmanager
    def stream(self, name):
        with torch.cuda.stream(self.s[name]):
            yield

    def event_rec(self, name=None):
        e = self._ev()
        e.record(self.s[name] if name is not None else torch.cuda.current_stream())
        return e

    def marker(self):
        e = self._ev()
        e.record(self.aux)
        return e

    def host_wait(self, ev):
        ev.synchronize()                                       # releases the GIL (torch 2.6 Event.cpp:162-166)

    def stream_wait(self, name, ev):
        (self.s[name] if name is not None else torch.cuda.current_stream()).wait_event(ev)

    def memcpy(self, dst, src, name=None):
        dst.copy_(src, non_blocking=True)                      # inside `with stream(...)`: cudaMemcpyAsync on that stream

    def index_copy(self, dst2d, idx, src, name=None):
        dst2d.index_copy_(0, idx, src)

    def sleep(self, ms, name=None):
        torch.cuda._sleep(int(ms * 1e-3 * self.clock_hz))

    def synchronize(self, order=None):
        torch.cuda.synchronize()


# -------------------------------------------------------------------------------------------------------- pipeline
@dataclass
class Faults:
    """Negative controls (each must be DETECTED by the checks) and delays for the lifetime controls."""
    skip_scatter_chunk: Optional[int] = None       # this chunk of every layer is not scattered (canary must see poison)
    poison_stage: bool = False                      # the first staged K row of chunk 0 is overwritten after packing
    no_slot_wait: bool = False                      # the host repacks a staging slot without waiting for its H2D
    no_landing_wait: bool = False                   # an H2D overwrites a landing slot without waiting for its scatter
    no_h2d_wait: bool = False                       # the scatter does not wait for its OWN chunk's H2D (reads the slot early)
    delay_copy_ms: float = 0.0                      # a GPU sleep on the copy stream before chunk 0 of each layer
    delay_scatter_ms: float = 0.0                   # a GPU sleep on the scatter stream before chunk 0 of each layer

    def any(self) -> bool:
        return any((self.skip_scatter_chunk is not None, self.poison_stage, self.no_slot_wait, self.no_landing_wait, self.no_h2d_wait,
                    self.delay_copy_ms > 0, self.delay_scatter_ms > 0))


MODES = {                       # (pack, dma, scatter, host writes the slot)
    "full": (True, True, True, True),
    "L": (False, False, False, False),
    "LP": (True, False, False, True),
    "LPD": (True, True, False, True),
    "prepacked": (False, True, True, True),    # DIAGNOSTIC CEILING: payload NOT packed (stale staging), indices written
}


class Pipe:
    """The per-chunk pipeline of one transport rep: pack (CPU) -> one H2D per chunk (copy stream) -> scatter (scatter
    stream), over a ring of `ring` staging slots (pinned host) and `ring` landing slots (device). One Pipe per rep
    sequence; reset() after a full synchronize."""

    def __init__(self, be, stage_u8: torch.Tensor, land_u8: torch.Tensor, cap_groups: int, dtype=torch.bfloat16, R: int = R_DEF,
                 D: int = D_DEF):
        if stage_u8.dim() != 2 or land_u8.shape != stage_u8.shape:
            raise ValueError("stage and landing must be (ring, slot_bytes) of one shape")
        self.be, self.stage, self.land = be, stage_u8, land_u8
        self.ring, self.cap = stage_u8.shape[0], int(cap_groups)
        self.dtype, self.R, self.D = dtype, R, D
        if stage_u8.shape[1] < slot_bytes(self.cap, R, D, torch.empty((), dtype=dtype).element_size()):
            raise ValueError("slot too small for cap_groups")
        self.reset()

    def reset(self):
        """After a full synchronize: forget the ring's events and zero every slot's index region [0, P0) in the staging AND
        the landing ring (so a too-early transfer or a too-early scatter in a negative control can only see indices of this
        pipe or zeros, all in range: the landing ring is shared by pipes of different caps, and a smaller cap's payload
        lies inside a larger cap's index region)."""
        self.k = 0
        self.slot_h2d = [None] * self.ring
        self.slot_sc = [None] * self.ring
        p0 = idx_region(self.cap, self.R)
        self.stage[:, :p0].zero_()
        self.land[:, :p0].zero_()

    def layer(self, desc: Desc, src_k2d, src_v2d, dst_k2d, dst_v2d, mode: str = "full", faults: Optional[Faults] = None,
              lite: bool = False, chunks: Optional[List[Tuple[int, int]]] = None, host_stamps: bool = False) -> List[Dict]:
        """Run every chunk of one layer. src_*2d / dst_*2d are the (-1, D) row views or (-1, R*D) group views matching
        desc.packer / desc.placer. Returns one record per chunk (events + host nanoseconds).
        host_stamps (feeder diagnostic, default off = the unchanged record): ABSOLUTE perf_counter_ns instants per chunk,
        fw0/fw1 (staging-slot wait; equal when no wait ran), p0/p1 (pack + index write), sub0 (submission start), cp0/cp1 (the
        H2D call: entry and return, i.e. AFTER the copy call returns and BEFORE the scatter submission), sc0h/sc1h (scatter
        submission entry / return). The work, its order and its waits are the same with and without them."""
        f = faults or Faults()
        pack, dma, scat, host_writes = MODES[mode]
        be, R = self.be, self.R
        sp, dp = desc.src_per_group, desc.dst_per_group
        chunks = chunk_requests(desc.req_groups.tolist(), self.cap) if chunks is None else chunks
        recs = []
        for ci, (g0, g1) in enumerate(chunks):
            n = g1 - g0
            s = self.k % self.ring
            r = dict(g0=g0, g1=g1, slot=s, bp_ns=0, pack_ns=0, pack_cpu_ns=0, api_ns=0, bytes=0)
            if host_stamps:
                r["fw0"] = r["fw1"] = time.perf_counter_ns()
            if host_writes and self.slot_h2d[s] is not None and not f.no_slot_wait:
                t = time.perf_counter_ns()
                be.host_wait(self.slot_h2d[s])                  # the H2D that last read this slot has completed
                r["bp_ns"] = time.perf_counter_ns() - t
                if host_stamps:
                    r["fw0"], r["fw1"] = t, t + r["bp_ns"]
            kst, vst, ist, lo, hi = stage_views(self.stage[s], n, desc.packer, desc.placer, self.cap, self.dtype, R, self.D)
            r["bytes"] = hi - lo
            t, tc = time.perf_counter_ns(), time.thread_time_ns()
            if pack:
                torch.index_select(src_k2d, 0, desc.src_idx[g0 * sp:g1 * sp], out=kst)
                torch.index_select(src_v2d, 0, desc.src_idx[g0 * sp:g1 * sp], out=vst)
                if f.poison_stage and ci == 0:
                    kst.view(torch.int16).view(-1)[:self.D].fill_(POISON_I16 - 1)
            if host_writes:
                ist.copy_(desc.dst_idx[g0 * dp:g1 * dp])
            r["pack_ns"], r["pack_cpu_ns"] = time.perf_counter_ns() - t, time.thread_time_ns() - tc
            if host_stamps:
                r["p0"], r["p1"] = t, t + r["pack_ns"]
            r["pk"] = None if lite else be.marker()
            t = time.perf_counter_ns()
            if host_stamps:
                r["sub0"] = t
            if dma:
                with be.stream("copy"):
                    if scat and self.slot_sc[s] is not None and not f.no_landing_wait:
                        be.stream_wait("copy", self.slot_sc[s])  # the scatter that last read this landing slot is done
                    if f.delay_copy_ms > 0 and ci == 0:
                        be.sleep(f.delay_copy_ms, "copy")
                    r["h2d0"] = None if lite else be.event_rec("copy")
                    if host_stamps:
                        r["cp0"] = time.perf_counter_ns()
                    be.memcpy(self.land[s][lo:hi], self.stage[s][lo:hi], "copy")
                    if host_stamps:
                        r["cp1"] = time.perf_counter_ns()
                    r["h2d1"] = be.event_rec("copy")
                self.slot_h2d[s] = r["h2d1"]
            if scat:
                kd, vd, idd, _, _ = stage_views(self.land[s], n, desc.placer, desc.placer, self.cap, self.dtype, R, self.D)
                if host_stamps:
                    r["sc0h"] = time.perf_counter_ns()
                with be.stream("scatter"):
                    if not f.no_h2d_wait:
                        be.stream_wait("scatter", r["h2d1"])    # the scatter reads the landing slot only after its H2D
                    if f.delay_scatter_ms > 0 and ci == 0:
                        be.sleep(f.delay_scatter_ms, "scatter")
                    r["sc0"] = None if lite else be.event_rec("scatter")
                    if f.skip_scatter_chunk != ci:
                        be.index_copy(dst_k2d, idd, kd, "scatter")
                        be.index_copy(dst_v2d, idd, vd, "scatter")
                    r["sc1"] = be.event_rec("scatter")
                self.slot_sc[s] = r["sc1"]
                if host_stamps:
                    r["sc1h"] = time.perf_counter_ns()
            r["api_ns"] = time.perf_counter_ns() - t
            r["sub"] = None if lite else be.marker()
            self.k += 1
            recs.append(r)
        return recs


def views_for(desc: Desc, src_phys: torch.Tensor, dst_phys: torch.Tensor, R: int = R_DEF):
    """(src_k-like 2-D view, dst 2-D view) of one tensor pair for the desc's packer / placer."""
    s2 = rows2d(src_phys) if desc.packer == "row" else groups2d(src_phys, R)
    d2 = rows2d(dst_phys) if desc.placer == "row" else groups2d(dst_phys, R)
    return s2, d2


# ------------------------------------------------------------------------------------------------ references/checks
def direct_gather_reference(src_log: torch.Tensor, plan_hbm: torch.Tensor, dst_log: torch.Tensor, R: int = R_DEF) -> None:
    """The per-head block copy by LOGICAL indexing, one (h, b, m) at a time (the slow oracle): slot m of (b, h) receives
    host block plan[h, b, m] of (b, h)."""
    H, B, M = plan_hbm.shape
    for h in range(H):
        for b in range(B):
            for m in range(M):
                blk = int(plan_hbm[h, b, m])
                if blk >= 0:
                    dst_log[b, m * R:(m + 1) * R, h] = src_log[b, blk * R:(blk + 1) * R, h]


def reference_rows(src_log: torch.Tensor, desc: Desc) -> torch.Tensor:
    """The expected (n*R, D) rows, (b, h, m, r) order, by advanced LOGICAL indexing (independent of the address formulas)."""
    b, t, h = source_rows_logical(desc, desc.R)
    return src_log[b, t, h]


def compact_dest(dst_log: torch.Tensor, desc: Desc) -> torch.Tensor:
    """The destination rows of desc, (b, h, m, r) order, by advanced LOGICAL indexing."""
    b, t, h = dest_rows_logical(desc, desc.R)
    return dst_log[b, t, h]


def poison_(t: torch.Tensor) -> torch.Tensor:
    t.view(torch.int16).fill_(POISON_I16)
    return t


def nonpoison_rows(phys: torch.Tensor, chunk_rows: int = 1 << 20) -> torch.Tensor:
    """bool per 256-B physical row: True if any element differs from the poison pattern."""
    v = rows2d(phys).view(torch.int16)
    out = torch.empty(v.shape[0], dtype=torch.bool, device=v.device)
    for a in range(0, v.shape[0], chunk_rows):
        out[a:a + chunk_rows] = (v[a:a + chunk_rows] != POISON_I16).any(dim=1)
    return out


def dest_row_mask(descs: Sequence[Desc], n_rows: int, device="cpu", R: int = R_DEF) -> torch.Tensor:
    """bool per physical row of the destination: rows written by any of `descs` (row units of the dst layout)."""
    mask = torch.zeros(n_rows, dtype=torch.bool, device=device)
    for d in descs:
        if d.n == 0:
            continue
        if d.placer == "row":
            rows = d.dst_idx
        else:
            rows = (d.dst_idx[:, None] * R + torch.arange(R, dtype=torch.int64)[None, :]).view(-1)
        mask[rows.to(device)] = True
    return mask


def canary(phys: torch.Tensor, expected: torch.Tensor) -> Dict:
    """Written rows must be exactly the expected rows: extra = written outside (a stray write), missing = never written
    (a skipped scatter)."""
    got = nonpoison_rows(phys)
    extra = int((got & ~expected).sum())
    missing = int((expected & ~got).sum())
    return dict(ok=(extra == 0 and missing == 0), extra_rows=extra, missing_rows=missing, expected_rows=int(expected.sum()))


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def overfetch_report(desc: Desc, D: int = D_DEF, elem: int = 2) -> Dict:
    """Per-head independence: every packed source row belongs to its own group's head and (b, block); packed bytes equal the
    useful bytes (no second head fetched to manufacture contiguity)."""
    b, t, h = source_rows_logical(desc, desc.R)
    distinct = int(torch.unique((b * (1 << 40)) + (t * 4) + h).numel()) if desc.n else 0
    packed_rows = int(desc.src_idx.numel()) * (desc.R if desc.packer == "group" else 1)
    return dict(groups=desc.n, packed_rows_per_tensor=packed_rows, distinct_source_rows=distinct,
                packed_bytes=2 * packed_rows * D * elem, useful_bytes=useful_bytes(desc.n, desc.R, D, elem),
                ok=(packed_rows == desc.n * desc.R == distinct))


# ------------------------------------------------------------------------------------------ in-place layout change
def _reorder_region(reg: torch.Tensor, S: int, H: int, D: int, src_layout: str, verify: bool) -> Tuple[int, int, int]:
    """Reorder ONE request's region (a flat view of S*H*D elements; the same bytes in both layouts) in place from
    src_layout to the other layout. Returns (bad, reorder_ns, verify_ns)."""
    t0 = time.perf_counter_ns()
    if src_layout == "orig":
        A = reg.view(S, H, D).clone()                                       # original logical [S, H, D]
        reg.view(H, S, D).copy_(A.permute(1, 0, 2))
        t1 = time.perf_counter_ns()
        bad = int(not torch.equal(reg.view(H, S, D).permute(1, 0, 2).view(torch.int16), A.view(torch.int16))) if verify else 0
    else:
        A = reg.view(H, S, D).permute(1, 0, 2).clone()                      # original logical [S, H, D] (clone of a permuted
        reg.view(S, H, D).copy_(A)                                          # view: it keeps the permuted strides; the copy_
        t1 = time.perf_counter_ns()                                         # honours them, so the content is still exact)
        bad = int(not torch.equal(reg.view(S, H, D).view(torch.int16), A.view(torch.int16))) if verify else 0
    return bad, t1 - t0, time.perf_counter_ns() - t1


def _dims(phys: torch.Tensor, layout: str) -> Tuple[int, int, int, int]:
    if layout == "orig":
        B, S, H, D = phys.shape
    else:
        B, H, S, D = phys.shape
    return B, S, H, D


def convert_inplace(phys: torch.Tensor, src_layout: str, dst_layout: str, verify: bool = True) -> Tuple[torch.Tensor, Dict]:
    """Reorder a CONTIGUOUS physical tensor between layouts IN ITS OWN ALLOCATION, one request at a time (a request's region
    [b*S*H*D, (b+1)*S*H*D) is the same in both layouts). Extra memory: one request's region (S*H*D elements, pageable).
    With verify, the logical [S, H, D] content of every request is compared bit for bit with the original after the write.
    Returns (the physical tensor of dst_layout on the same storage, info); info splits the time into ms_reorder (clone +
    permuted write) and ms_verify (the bit-exact check); ms = their sum."""
    if src_layout == dst_layout:
        return phys, dict(requests=0, bad=0, ms=0.0, ms_reorder=0.0, ms_verify=0.0, tmp_bytes=0)
    if not phys.is_contiguous():
        raise ValueError("convert_inplace needs the contiguous physical tensor")
    B, S, H, D = _dims(phys, src_layout)
    flat = phys.view(-1)
    per = S * H * D
    bad, t_re, t_ve = 0, 0, 0
    for b in range(B):
        x, a, v = _reorder_region(flat[b * per:(b + 1) * per], S, H, D, src_layout, verify)
        bad, t_re, t_ve = bad + x, t_re + a, t_ve + v
    out = flat.view(phys_shape(dst_layout, B, S, H, D))
    assert out.data_ptr() == phys.data_ptr()
    return out, dict(requests=B, bad=bad, ms=(t_re + t_ve) / 1e6, ms_reorder=t_re / 1e6, ms_verify=t_ve / 1e6,
                     tmp_bytes=per * phys.element_size(), bytes_per_tensor=phys.numel() * phys.element_size())


def probe_convert(phys: torch.Tensor, layout: str, b: int = 0, verify: bool = True) -> Dict:
    """The conversion-time PREDICTOR: request b's region is reordered to the other layout and back, in place, with the
    same per-request code as convert_inplace (so ms_fwd is one request's forward cost incl. verification). The tensor is
    left bit-identical (checked against a clone of the region). Extra memory: two region copies (pageable)."""
    if not phys.is_contiguous():
        raise ValueError("probe_convert needs the contiguous physical tensor")
    B, S, H, D = _dims(phys, layout)
    if not 0 <= b < B:
        raise ValueError("request %d not in [0, %d)" % (b, B))
    per = S * H * D
    reg = phys.view(-1)[b * per:(b + 1) * per]
    keep = reg.clone()
    other = "hm" if layout == "orig" else "orig"
    bad1, r1, v1 = _reorder_region(reg, S, H, D, layout, verify)
    bad2, r2, v2 = _reorder_region(reg, S, H, D, other, verify)
    same = bool(torch.equal(reg.view(torch.int16), keep.view(torch.int16)))
    return dict(b=b, ms_fwd=(r1 + v1) / 1e6, ms_back=(r2 + v2) / 1e6, ms_reorder_fwd=r1 / 1e6, ms_verify_fwd=v1 / 1e6,
                bad=bad1 + bad2, restored=same, region_bytes=per * phys.element_size())


def tail_write_runs(layout: str, B: int, H: int = H_DEF, R: int = R_DEF, D: int = D_DEF, elem: int = 2) -> Dict:
    """The host write of ONE completed tail block (every 64 decode tokens, cache_engine.py:704-708) per layer per tensor:
    orig writes B runs of R*H*D*elem bytes (both heads interleaved, 32 KiB); hm writes B*H runs of R*D*elem (16 KiB)."""
    if layout == "orig":
        runs, run = B, R * H * D * elem
    elif layout == "hm":
        runs, run = B * H, R * D * elem
    else:
        raise ValueError(layout)
    return dict(layout=layout, runs=runs, run_bytes=run, bytes=runs * run)


def tail_write_accounting(layout: str, B: int, layers: int = 32, H: int = H_DEF, R: int = R_DEF, D: int = D_DEF, elem: int = 2,
                          link_gbps: float = 25.0, measured_ms_per_tensor: Optional[float] = None) -> Dict:
    """Per rollover (once every R decode tokens): bytes K+V over all layers, a link-rate LOWER bound, the measured-based
    value when a per-tensor measurement exists, and both amortized per decode token (divided by R)."""
    tw = tail_write_runs(layout, B, H, R, D, elem)
    per_roll = 2 * layers * tw["bytes"]
    bound_ms = per_roll / (link_gbps * 1e6)
    meas = None if measured_ms_per_tensor is None else 2 * layers * float(measured_ms_per_tensor)
    return dict(tw, layers=layers, bytes_per_rollover=per_roll, link_bound_ms_per_rollover=bound_ms,
                link_bound_ms_per_token=bound_ms / R, measured_ms_per_rollover=meas,
                measured_ms_per_token=(None if meas is None else meas / R), runs_per_rollover=2 * layers * tw["runs"])


def conversion_accounting(B: int, S: int, H: int = H_DEF, D: int = D_DEF, elem: int = 2, layers: int = 32,
                          measured_ms_per_tensor: Optional[Sequence[float]] = None) -> Dict:
    """The ONE-TIME host conversion (original -> head-major) of the whole host cache: bytes per tensor, total, the extra
    memory of the in-place method (one request's region, pageable) and the measured time when given."""
    per = B * S * H * D * elem
    tot = 2 * layers * per
    meas = None if not measured_ms_per_tensor else float(sum(measured_ms_per_tensor))
    return dict(bytes_per_tensor=per, tensors=2 * layers, bytes_total=tot, extra_pinned_bytes=0, tmp_pageable_bytes=S * H * D * elem,
                traffic_bytes_estimate=3 * tot, measured_ms_total=meas,
                measured_gbps=(None if not meas else tot / (meas * 1e6)))
