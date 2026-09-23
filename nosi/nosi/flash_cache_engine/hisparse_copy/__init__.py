"""SGLang HiSparse's copy-only swap-in kernel (copy_cache_planned_kernel, sglang 87db743) as a torch
extension, plus the adapter that turns NOSI's miss descriptors into HiSparse miss plans.

  hisparse_copy.cu   the kernel, verbatim from hisparse.cuh :837-892 (with transfer_item_warp :140-166
                     and copy_miss_item's generic path :177-190, :212-224); the only change is the
                     launcher (header of that file lists every difference)
  hisparse_copy.cpp  pybind + every argument check
  plan.py            pure torch, CPU-testable: NOSI (H, B, M) load-id tables -> HiSparse plans

The extension is built on first use (load()), never at import, so plan.py and this package import on a
CPU-only machine. Build directory: $TORCH_EXTENSIONS_DIR/<NAME> when TORCH_EXTENSIONS_DIR is set (the
sbatch points it at a private per-job directory), torch's default otherwise. The arch list comes from
TORCH_CUDA_ARCH_LIST (8.0 in the sbatch).
"""
import os
from pathlib import Path

from .plan import (  # noqa: F401  (re-exported)
    KINDS,
    LABEL_KIND,
    Plan,
    PlanRefused,
    build_plan,
    item_bytes,
    item_label,
    kernel_walk,
    plans_equal,
    simulate_planned_copy,
    timed_build,
    validate_plan,
)

HERE = Path(__file__).resolve().parent
NAME = "hisparse_copy_87db743"
UPSTREAM_COMMIT = "87db74302151956e354a05c01ecda47fd24cb882"
BLOCK_SIZES = (256, 1024)
_EXT = None


def load(verbose: bool = False):
    """JIT-build (once per process) and return the extension module."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load as _load
        build_dir = None
        root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if root:
            build_dir = os.path.join(root, NAME)
            os.makedirs(build_dir, exist_ok=True)
        _EXT = _load(
            name=NAME,
            sources=[str(HERE / "hisparse_copy.cpp"), str(HERE / "hisparse_copy.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=build_dir,
            verbose=verbose,
        )
        assert tuple(_EXT.block_sizes()) == BLOCK_SIZES, _EXT.block_sizes()
        assert _EXT.upstream_commit() == UPSTREAM_COMMIT, _EXT.upstream_commit()
    return _EXT


def copy_planned(miss_src_locs, miss_dst_locs, miss_counts, num_real_reqs, host_k, host_v, dev_k, dev_v,
                 num_blocks: int, block_size: int, item_size_bytes: int, skip_io: bool = False) -> None:
    """One launch of copy_cache_planned_kernel<block_size, IsMLA=false, IsDsv4Layout=false, skip_io> with a
    grid of num_blocks on the CURRENT torch stream."""
    load().copy_planned(miss_src_locs, miss_dst_locs, miss_counts, num_real_reqs, host_k, host_v, dev_k, dev_v,
                        int(num_blocks), int(block_size), int(item_size_bytes), bool(skip_io))


def copy_plan(plan: Plan, host_k, host_v, dev_k, dev_v, num_blocks: int, block_size: int, skip_io: bool = False) -> None:
    """copy_planned with the four plan tensors and the item size taken from a Plan."""
    copy_planned(plan.src, plan.dst, plan.counts, plan.num_real, host_k, host_v, dev_k, dev_v,
                 num_blocks, block_size, plan.item_size_bytes, skip_io)
