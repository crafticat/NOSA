"""GIL-FREE CPU PACKER for the sustained harness (feature/nosi-sustained-gilfree): the JIT build of gilfree_pack.cpp,
the Python reference of the bytes it must produce, and the feeder work of the cpu_native arm.

The extension does one release per call with the GIL released (gilfree_pack.cpp header): the descriptor, the packing
(at::parallel_for, torch's own OpenMP pool) and the CUDA submission (one H2D per <= cap-group chunk on the copy stream,
index_copy_ scatter on the scatter stream, cpupack_core.Pipe's slot layout and lifetime waits). The decode thread then
meets the feeder at the GIL a few times per release instead of about 2000 times (run b64_s960_v2).

BUILD: torch.utils.cpp_extension.load(with_cuda=True: the CUDA event / stream headers; no .cu file, no nvcc call) into
<TORCH_EXTENSIONS_DIR>/gilfree_pack_v1, refused when TORCH_EXTENSIONS_DIR is unset, relative or under HOME (the run's
own JIT directory: sustained_step.sh sets $JITBASE/sustained_<tag>/torch_ext). TORCH_CUDA_ARCH_LIST defaults to 8.0
here when unset (no device is queried on a CPU node).

PRIOR ART (retroinfer-eval docs/superpowers/specs/2026-09-02-prior-art-table.md, 2026-10-04 rows; FIX_DESIGN.md):
the pinned staging ring + one bulk H2D per chunk + device placement is cpupack_core's (PQCache / InfiniGen pattern);
the slot layout and the three lifetime waits are reused unchanged. New here: the whole release in one GIL-released
native call.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE = Path(__file__).resolve().with_name("gilfree_pack.cpp")
EXT_NAME = "gilfree_pack_v1"
STATS = ("groups", "chunks", "layers", "pack_ns", "wait_ns", "submit_ns", "total_ns")


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
        raise ValueError("TORCH_EXTENSIONS_DIR is not set: the cpu_native arm builds its extension under the run's own "
                         "directory (sustained_step.sh sets it), never under HOME")
    return os.path.join(check_build_dir(d, home), EXT_NAME)


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def load(build_dir: str, verbose: bool = False):
    """Build (or reuse) and import the extension in build_dir (check_build_dir applies)."""
    build_dir = check_build_dir(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    from torch.utils.cpp_extension import load as _load
    return _load(name=EXT_NAME, sources=[str(SOURCE)], extra_cflags=["-O3"], build_directory=build_dir,
                 with_cuda=True, verbose=verbose)


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


def native_pack(ext, spec, geom, src_k, src_v, repeat: int = 1, hold_gil: bool = False):
    """The extension's flat packing of one release (pack_into): (destination rows, K rows, V rows, [groups, rows, ns])."""
    n = spec.n_groups * geom.R
    out_i = torch.empty(n, dtype=torch.int64)
    out_k = torch.empty((n, geom.D), dtype=src_k[0].dtype)
    out_v = torch.empty_like(out_k)
    fn = ext.pack_into_holding_gil if hold_gil else ext.pack_into
    st = fn(spec.b, spec.h, spec.l, spec.m, spec.blk, list(src_k), list(src_v), geom.R, geom.M * geom.R, out_i, out_k,
            out_v, int(repeat))
    return out_i, out_k, out_v, st


def native_work_fn(npipe, be, geom, scr_k: torch.Tensor, scr_v: torch.Tensor):
    """The cpu_native arm's feeder work: ONE safe point and ONE GIL-released npipe.release per release (the pause for a
    re-warm waits for the release in progress, never a CUDA call in progress), then the done event on "scatter" (the
    stream the extension scatters on). npipe.set_sources must have been called. work.stats sums the extension's
    per-release stats (STATS)."""
    dk, dv = scr_k.view(-1, geom.D), scr_v.view(-1, geom.D)
    totals: Dict[str, int] = {k: 0 for k in STATS}
    totals["releases"] = 0

    def work(rel, fd) -> None:
        fd.safe_point()
        spec = rel.spec
        st = npipe.release(spec.b, spec.h, spec.l, spec.m, spec.blk, dk, dv)
        rel.layers = int(st[2])
        rel.done = be.event_rec("scatter")
        for k, v in zip(STATS, st):
            totals[k] += int(v)
        totals["releases"] += 1
    work.stats = totals
    work.last = {}
    return work
