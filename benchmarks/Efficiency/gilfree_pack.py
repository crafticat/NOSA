"""GIL-FREE CPU PACKER v3 (feature/nosi-r2-gilfree-worker): the JIT build of gilfree_pack.cpp, the Python reference of
the bytes it must produce, and the feeder work of the sustained harness's cpu_native arm. v3 = agent E's v2
(feature/nosi-sustained-gilfree2 f2b981a, the pin of SG5 and of the r2 gilfree transport's default) + an OPTIONAL C++
worker thread inside NativePipe (start_worker / submit / job_done / wait_job / stream_wait_job / take_error /
worker_stats / stop_worker) that runs whole releases with no Python and no GIL, and publishes the generation board in
C++ (gilfree_pack.cpp header). v2's entry points keep their meaning.

The extension does one release per call with the GIL released (gilfree_pack.cpp header): the descriptor, the packing
(an explicit OpenMP loop on set_threads(N) threads, built with -fopenmp; the team size actually used is returned) and
the CUDA submission (one H2D per <= cap-group chunk on the copy stream, index_copy_ scatter on the scatter stream,
cpupack_core.Pipe's slot layout and lifetime waits). v3's slot holds SLOT_EXTRA_PER_GROUP (16) more bytes per group
(the board pairs): a caller sizes the slot as cap * R * 8 + cap * SLOT_EXTRA_PER_GROUP + 2 * cap * R * row_bytes.

BUILD: torch.utils.cpp_extension.load(with_cuda=True: the CUDA event / stream headers and the runtime API; no .cu file,
no nvcc call) into <TORCH_EXTENSIONS_DIR>/gilfree_pack_v3, refused when TORCH_EXTENSIONS_DIR is unset, relative or under
HOME. TORCH_CUDA_ARCH_LIST defaults to 8.0 here when unset (no device is queried on a CPU node).

PRIOR ART (retroinfer-eval docs/superpowers/specs/2026-09-02-prior-art-table.md, 2026-10-04 rows; FIX_DESIGN.md):
the pinned staging ring + one bulk H2D per chunk + device placement is cpupack_core's (PQCache / InfiniGen pattern);
the slot layout and the three lifetime waits are reused unchanged. New in v2: the whole release in one GIL-released
native call. New in v3: a native worker thread (a std::thread job queue, as any native I/O worker) so that the
submission never needs the interpreter, and the board publication on the stream that landed the rows.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE = Path(__file__).resolve().with_name("gilfree_pack.cpp")
EXT_NAME = "gilfree_pack_v3"                  # v3: v2 + the optional C++ worker and the C++ board publication
SUMS = ("groups", "chunks", "layers", "pack_ns", "wait_ns", "submit_ns", "total_ns")
STATS = SUMS + ("threads_min", "threads_max")


def check_build_dir(path: str, home: Optional[str] = None) -> str:
    """An absolute build directory outside HOME (refused otherwise)."""
    if not path or not os.path.isabs(path):
        raise ValueError("the extension build directory must be an absolute path (got %r)" % (path,))
    p = os.path.realpath(path)
    h = os.path.realpath(home if home is not None else os.path.expanduser("~"))
    if p == h or p.startswith(h.rstrip(os.sep) + os.sep):
        raise ValueError("the extension build directory %s lies under HOME %s: build under the run's own directory"
                         % (p, h))
    return p


def build_dir_from_env(env: Optional[Dict[str, str]] = None, home: Optional[str] = None) -> str:
    env = os.environ if env is None else env
    d = env.get("TORCH_EXTENSIONS_DIR", "")
    if not d:
        raise ValueError("TORCH_EXTENSIONS_DIR is not set: the extension is built under the run's own directory, never "
                         "under HOME")
    return os.path.join(check_build_dir(d, home), EXT_NAME)


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def load(build_dir: str, verbose: bool = False, openmp: bool = True):
    """Build (or reuse) and import the extension in build_dir (check_build_dir applies). openmp=True: -fopenmp (the
    packing loop runs on set_threads / threads threads). openmp=False builds the module gilfree_pack_v3_noomp, whose
    loop is serial and reports a team of 1: the NEGATIVE CONTROL of the thread-count check (v1's defect)."""
    build_dir = check_build_dir(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    from torch.utils.cpp_extension import load as _load
    return _load(name=EXT_NAME if openmp else EXT_NAME + "_noomp", sources=[str(SOURCE)],
                 extra_cflags=["-O3"] + (["-fopenmp"] if openmp else []),
                 extra_ldflags=(["-fopenmp"] if openmp else []), build_directory=build_dir, with_cuda=True,
                 verbose=verbose)


def requested_threads(arm, pack_threads: int) -> int:
    """The packing team size of a cpu_native arm: its threads=N option, else --pack-threads."""
    return int(getattr(arm, "threads", 0) or pack_threads)


def threads_ok(requested: int, native: Optional[Dict[str, Any]]) -> bool:
    """Every chunk of the arm ran on exactly the requested team (threads_min == threads_max == requested)."""
    return bool(native) and native.get("threads_min") == requested and native.get("threads_max") == requested


def stream_triple(stream) -> List[int]:
    """(stream_id, device_index, device_type) of a torch.cuda.Stream: c10::Stream::unpack3 in the extension."""
    return [int(stream.stream_id), int(stream.device_index), int(stream.device_type)]


def reference_stream(K, spec, geom, src_k: Sequence[torch.Tensor], src_v: Sequence[torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """THE PYTHON PACKER'S BYTES of one release: per layer ascending, cpupack_core.build_desc (row packer / row placer,
    orig layouts, no tail slot; exactly sustained_core.cpu_work_fn's call) and index_select of the source rows, then
    concatenated: (destination rows int64, K rows, V rows). Pipe.layer cuts each layer's sequence into chunks and stages
    [indices | K | V] per chunk; the concatenation over chunks and layers is this stream."""
    S_dst = geom.M * geom.R
    idx, ks, vs = [], [], []
    for l in spec.layers():
        desc = K.build_desc(spec.plan(l, geom), src_layout="orig", dst_layout="orig", s_src=geom.S, s_dst=S_dst,
                            packer="row", placer="row", R=geom.R, tail_slot=None)
        ks.append(K.rows2d(src_k[l % len(src_k)]).index_select(0, desc.src_idx))
        vs.append(K.rows2d(src_v[l % len(src_v)]).index_select(0, desc.src_idx))
        idx.append(desc.dst_idx)
    if not idx:
        e = src_k[0].new_empty((0, geom.D))
        return torch.empty(0, dtype=torch.int64), e, e.clone()
    return torch.cat(idx), torch.cat(ks), torch.cat(vs)


def native_pack(ext, spec, geom, src_k, src_v, repeat: int = 1, hold_gil: bool = False, threads: int = 1):
    """The extension's flat packing of one release (pack_into) on `threads` threads: (destination rows, K rows, V rows,
    [groups, rows, ns, team size used])."""
    n = spec.n_groups * geom.R
    out_i = torch.empty(n, dtype=torch.int64)
    out_k = torch.empty((n, geom.D), dtype=src_k[0].dtype)
    out_v = torch.empty_like(out_k)
    fn = ext.pack_into_holding_gil if hold_gil else ext.pack_into
    st = fn(spec.b, spec.h, spec.l, spec.m, spec.blk, list(src_k), list(src_v), geom.R, geom.M * geom.R, out_i, out_k,
            out_v, int(repeat), int(threads))
    return out_i, out_k, out_v, st


def native_work_fn(npipe, be, geom, scr_k: torch.Tensor, scr_v: torch.Tensor):
    """The cpu_native arm's feeder work: ONE safe point and ONE GIL-released npipe.release per release (the pause for a
    re-warm waits for the release in progress, never a CUDA call in progress), then the done event on "scatter" (the
    stream the extension scatters on). npipe.set_sources and npipe.set_threads must have been called. work.stats sums
    the extension's per-release stats (SUMS) and keeps the smallest / largest OpenMP team over all chunks
    (threads_min / threads_max, None before the first chunk)."""
    dk, dv = scr_k.view(-1, geom.D), scr_v.view(-1, geom.D)
    totals: Dict[str, Any] = {k: 0 for k in SUMS}
    totals.update(releases=0, threads_min=None, threads_max=None)

    def work(rel, fd) -> None:
        fd.safe_point()
        spec = rel.spec
        st = npipe.release(spec.b, spec.h, spec.l, spec.m, spec.blk, dk, dv)
        rel.layers = int(st[2])
        rel.done = be.event_rec("scatter")
        for k, v in zip(SUMS, st):
            totals[k] += int(v)
        if int(st[1]) > 0:
            lo, hi = int(st[7]), int(st[8])
            totals["threads_min"] = lo if totals["threads_min"] is None else min(totals["threads_min"], lo)
            totals["threads_max"] = hi if totals["threads_max"] is None else max(totals["threads_max"], hi)
        totals["releases"] += 1
    work.stats = totals
    work.last = {}
    return work
