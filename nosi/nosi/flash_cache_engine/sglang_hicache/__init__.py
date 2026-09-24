"""SGLang's HiCache KV-cache IO kernels (sglang 87db743) as a torch extension, plus index plans for SGLang's native
layout and for NOSI's miss descriptors.

  transfer_aot.cu     sgl-kernel kvcacheio transfer.cu (AOT), VERBATIM :1-13 + :15-461 (upstream's own torch launcher
                      on the current stream). Written by Strata's first author (PRs #7382 / #8264): the CLOSEST RELEASED
                      IMPLEMENTATION of Strata's GPU-assisted IO kernel. The Strata paper (arXiv 2508.18572, OSDI '26)
                      releases no code; whether its Fig. 5 kernel is byte-identical to this one is UNVERIFIED.
                      Harness arms strata<W>_t<T>_... call ONLY this kernel.
  hicache_jit.cu      jit hicache.cuh hicache_transfer_per_layer, VERBATIM device code (+ the three utils.cuh helpers it
                      uses); the tvm-ffi host struct is replaced by a torch launcher on the current stream. SGLang's DEFAULT
                      load at 87db743 for 512-B elements; NOT by a Strata author. Arms hicachejit<W>_t1024_... call ONLY this.
  aot_binding.cpp     pybind of the AOT entry points (C++17, as upstream) + checks.h argument checks
  jit_binding.cpp     pybind of the JIT launcher (hicache_jit.cu is built with -std=c++20, as upstream's JIT)
  checks.h            every argument check (pinned host memory whose device pointer is its host pointer, strides, dtypes)
  plan.py             pure torch, CPU-testable: grids, walks, CPU twins, the native plan, the NOSI adapter, plan-cost timers.

HiSparse (hisparse_copy/) is a different kernel and is never labelled Strata.
The extension is built on first use (load()), never at import, so plan.py and this package import on a CPU-only
machine. Two extensions (AOT, JIT) because the two upstream builds use different C++ standards. Build directory:
$TORCH_EXTENSIONS_DIR/<NAME> when set (the sbatch uses a private per-job directory).
"""
import os
from pathlib import Path

from .plan import (  # noqa: F401  (re-exported)
    AOT_DEFAULT_QUOTA,
    AOT_DEFAULT_WARPS,
    BENCH_LABEL,
    ITEMS,
    JIT_DEFAULT_QUOTA,
    LAYOUTS,
    PLAN_COST_LABEL,
    NativePlan,
    PlanRefused,
    aot_grid,
    aot_walk,
    build_native,
    default_unroll,
    jit_grid,
    jit_walk,
    native_counts,
    nosi_bytes,
    nosi_items,
    simulate_aot,
    simulate_jit,
    timed_native_build,
    timed_nosi_build,
    validate_indices,
    walk_covers_once,
)

HERE = Path(__file__).resolve().parent
NAME_AOT = "sglang_hicache_aot_87db743"
NAME_JIT = "sglang_hicache_jit_87db743"
UPSTREAM_COMMIT = "87db74302151956e354a05c01ecda47fd24cb882"
JIT_ELEMENT_SIZES = (256, 512)
JIT_BLOCK_QUOTAS = (1, 2, 4, 8, 16)
STRATA_LABEL = ("closest released implementation of Strata's IO kernel: SGLang AOT kvcacheio transfer.cu @87db743 "
                "(Strata first author, PRs #7382/#8264); the paper kernel itself is unavailable")
JIT_LABEL = "SGLang default HiCache JIT kernel hicache_transfer_per_layer @87db743 (not a Strata author; not Strata)"
AOT_CUDA_FLAGS = ["-O3", "-DNDEBUG", "--expt-extended-lambda"]              # torch adds -std=c++17 (upstream AOT: C++17)
JIT_CUDA_FLAGS = ["-O3", "-DNDEBUG", "-std=c++20"]                         # upstream JIT: -std=c++20 -O3 (arch.py :130-136)
_EXT = {}


def _build(name, sources, cuda_flags, verbose):
    from torch.utils.cpp_extension import load as _load
    build_dir = None
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root:
        build_dir = os.path.join(root, name)
        os.makedirs(build_dir, exist_ok=True)
    return _load(name=name, sources=[str(HERE / s) for s in sources], extra_cflags=["-O3"], extra_cuda_cflags=list(cuda_flags),
                 extra_include_paths=[str(HERE)], build_directory=build_dir, verbose=verbose)


def load_aot(verbose: bool = False):
    """JIT-build (once per process) and return the AOT extension (transfer_aot.cu + aot_binding.cpp)."""
    if "aot" not in _EXT:
        ext = _build(NAME_AOT, ("aot_binding.cpp", "transfer_aot.cu"), AOT_CUDA_FLAGS, verbose)
        assert ext.upstream_commit() == UPSTREAM_COMMIT, ext.upstream_commit()
        _EXT["aot"] = ext
    return _EXT["aot"]


def load_jit(verbose: bool = False):
    """JIT-build (once per process) and return the JIT extension (hicache_jit.cu + jit_binding.cpp)."""
    if "jit" not in _EXT:
        ext = _build(NAME_JIT, ("jit_binding.cpp", "hicache_jit.cu"), JIT_CUDA_FLAGS, verbose)
        assert tuple(ext.jit_element_sizes()) == JIT_ELEMENT_SIZES, ext.jit_element_sizes()
        assert tuple(ext.jit_block_quotas()) == JIT_BLOCK_QUOTAS, ext.jit_block_quotas()
        assert ext.upstream_commit() == UPSTREAM_COMMIT, ext.upstream_commit()
        _EXT["jit"] = ext
    return _EXT["jit"]


def load(verbose: bool = False):
    """Both extensions."""
    return load_aot(verbose), load_jit(verbose)


def strata_per_layer(src_k, dst_k, src_v, dst_v, src_idx, dst_idx, item_size: int, block_quota: int = AOT_DEFAULT_QUOTA,
                     num_warps: int = AOT_DEFAULT_WARPS) -> None:
    """AOT transfer_kv_per_layer (layer_first on both sides) on the CURRENT torch stream."""
    load_aot().aot_per_layer(src_k, dst_k, src_v, dst_v, src_idx, dst_idx, int(item_size), int(block_quota), int(num_warps))


def strata_per_layer_pf_lf(src_k_all, dst_k, src_v_all, dst_v, src_idx, dst_idx, layer_id: int, item_size: int, src_layout_dim: int,
                           block_quota: int = AOT_DEFAULT_QUOTA, num_warps: int = AOT_DEFAULT_WARPS) -> None:
    """AOT transfer_kv_per_layer_pf_lf (page_first host, whole buffers) on the CURRENT torch stream."""
    load_aot().aot_per_layer_pf_lf(src_k_all, dst_k, src_v_all, dst_v, src_idx, dst_idx, int(layer_id), int(item_size), int(src_layout_dim),
                               int(block_quota), int(num_warps))


def jit_per_layer(k_dst2d, v_dst2d, idx_dst, k_src2d, v_src2d, idx_src, block_quota: int = JIT_DEFAULT_QUOTA) -> None:
    """JIT hicache_transfer_per_layer on (-1, D) views, on the CURRENT torch stream (SGLang: transfer_hicache_one_layer)."""
    es = int(k_src2d.shape[1]) * k_src2d.element_size()
    load_jit().jit_per_layer(k_dst2d, v_dst2d, idx_dst, k_src2d, v_src2d, idx_src, es, int(block_quota))


def host_register(t) -> None:
    """SGLang's host allocation (pool_host/common.py :124-160, legacy single call): cudaHostRegister(ptr, size, 0) on a
    CPU tensor. The caller keeps `t` alive and calls host_unregister before freeing it."""
    import torch
    rc = int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0))
    if rc != 0:
        raise RuntimeError("cudaHostRegister failed rc=%d" % rc)


def host_unregister(t) -> None:
    import torch
    torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
