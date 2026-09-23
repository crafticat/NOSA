// [vendored] SGLang HiSparse COPY-ONLY swap-in kernel (copy_cache_planned_kernel), vendored for the
// [vendored] retroinfer-eval focused gather experiment (matched arm beside NOSI's decode step).
// [vendored]   source   https://github.com/sgl-project/sglang
// [vendored]   commit   87db74302151956e354a05c01ecda47fd24cb882
// [vendored]   file     python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh
// [vendored]   license  Apache-2.0 (Copyright the SGLang team; see the upstream LICENSE)
// [vendored]
// [vendored] The pinned verbatim upstream file is retroinfer-eval
// [vendored] scripts/hisparse_resolver/hisparse_upstream_87db743.cuh (sha256 58b4d685eca4506e37b50b90e3b84a97e60481913fe274bbdb6266b3ae3efa4b);
// [vendored] every ":NNN" in this directory is a line number of THAT file. retroinfer-eval
// [vendored] tests/test_hisparse_copy_plan.py checks the claims below against it.
// [vendored]
// [vendored] VERBATIM (byte for byte once the lines tagged `// [vendored]` are removed):
// [vendored]   :15-16     `namespace sglang {`
// [vendored]   :22-24     the CUDA branch of WARP_SIZE / BallotMask / FULL_WARP_MASK
// [vendored]   :140-166   transfer_item_warp, CUDA path: one warp moves one item with
// [vendored]              ld.global.nc.v2.b64 16-byte loads and st.global.cg.v2.b64 stores, 8-byte tail
// [vendored]   :177-190   copy_miss_item: comment, template head, signature, the static_assert, `if constexpr (IsDsv4Layout) {`
// [vendored]   :212-224   copy_miss_item: the GENERIC path (host and device both linear with stride
// [vendored]              item_size_bytes; K, then V at the same src/dst locs when !IsMLA)
// [vendored]   :837-892   copy_cache_planned_kernel: comment, template head and body (small fixed grid of
// [vendored]              num_blocks x BLOCK_SIZE threads; warps round-robin over the flattened
// [vendored]              (request, miss) plan; one warp per missed item)
// [vendored]   :926       `}  // namespace sglang`
// [vendored]
// [vendored] EVERY TEXTUAL DIFFERENCE FROM UPSTREAM:
// [vendored]   D1  :1-11, :13 the includes (sgl_kernel/{tensor,utils}.h, utils.cuh, deepseek_v4/kvcacheio.cuh,
// [vendored]                dlpack, tvm/ffi, stdexcept, string) are dropped; :12 <stdint.h> is kept (untagged);
// [vendored]                <cuda_runtime.h> is added (tagged)
// [vendored]   D2  :17-21, :25  the `#ifdef USE_ROCM` / `#else` / `#endif` lines and the ROCm (wave64) constants
// [vendored]                are dropped: CUDA only
// [vendored]   D3  :26-32   TOKEN_HIT, HASH_EMPTY, hash_slot dropped (resolver only)
// [vendored]   D4  :34-139, :167  the ROCm transfer_item_warp + transfer_dsv4_item_warp, and the `#else` /
// [vendored]                `#endif` around the CUDA transfer_item_warp, dropped
// [vendored]   D5  :169-175 popc_mask dropped (resolver only)
// [vendored]   D6  :191-211 the body of copy_miss_item's DSv4 page-padded branch (needs
// [vendored]                device::hisparse::transfer_item from deepseek_v4/kvcacheio.cuh) -> one static_assert
// [vendored]                (tagged); only IsDsv4Layout = false is instantiated
// [vendored]   D7  :226-835 transfer_cache_dsv4_mla(_kernel), warp_inclusive_scan, SmemLayout, the fused resolver
// [vendored]                kernel load_cache_to_device_buffer_kernel and its launcher dropped (the resolver is
// [vendored]                vendored separately, retroinfer-eval scripts/hisparse_resolver/)
// [vendored]   D8  :893-924 the tvm-ffi host launcher copy_cache_planned (LaunchKernel, tvm::ffi::TensorView)
// [vendored]                -> the torch launcher below the BEGIN TORCH LAUNCHER marker:
// [vendored]                <<<num_blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>, instantiated for
// [vendored]                BLOCK_SIZE in {256, 1024} x IsMLA in {false, true} x SkipIO in {false, true} with
// [vendored]                IsDsv4Layout = false; plan_stride = miss_src_locs.stride(0) and the equal-row-stride
// [vendored]                check exactly as :907-910. IsMLA = true is SGLang's SHIPPED instantiation (the caller
// [vendored]                copy_cache_planned_mla: K only) and passes nullptr for both V pointers exactly as
// [vendored]                :919/:921 do; IsMLA = false (separate K and V caches, K then V at the same locs) is
// [vendored]                a template instantiation the SGLang caller never makes (NOSI's KV is not MLA)
// [vendored]   D9  this header, and blank lines between the vendored blocks
// [vendored] Argument checks (dtype / device / stride / pinned / alignment) live in hisparse_copy.cpp.
#include <stdint.h>
#include <cuda_runtime.h>  // [vendored] D1

namespace sglang {

constexpr int WARP_SIZE = 32;
using BallotMask = unsigned int;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  // 128-bit bulk transfer via paired 64-bit loads (avoids alignment issues with uint4)
  const int total_pairs = item_size_bytes / 16;  // number of 16-byte chunks
  {
    const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
    uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
    for (int j = lane_id; j < total_pairs; j += WARP_SIZE) {
      uint64_t lo, hi;
      const uint64_t* s = src + j * 2;
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(s) : "memory");
      uint64_t* d = dst + j * 2;
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(lo), "l"(hi) : "memory");
    }
  }

  // Tail: 64-bit for remaining 8-byte chunk (if item_size not multiple of 16)
  const int tail_8B = (item_size_bytes - total_pairs * 16) / 8;
  if (tail_8B > 0 && lane_id < tail_8B) {
    const uint64_t* __restrict__ src8 =
        reinterpret_cast<const uint64_t*>(static_cast<const char*>(src_addr) + total_pairs * 16);
    uint64_t* __restrict__ dst8 = reinterpret_cast<uint64_t*>(static_cast<char*>(dst_addr) + total_pairs * 16);
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src8 + lane_id) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst8 + lane_id), "l"(tmp) : "memory");
  }
}

// Copy one missed item host->device with one warp. Shared by the fused swap-in
// kernel and copy_cache_planned_kernel so the layout dispatch cannot drift.
template <bool IsMLA, bool IsDsv4Layout>
__device__ __forceinline__ void copy_miss_item(
    int32_t lane_id,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int64_t src_loc,
    int64_t dst_loc,
    int64_t item_size_bytes) {
  static_assert(!IsDsv4Layout || IsMLA, "DSv4 page-padded layout is K-only (MLA).");
  if constexpr (IsDsv4Layout) {
    // [vendored] D6: upstream :191-211 (DSv4 page-padded copy via device::hisparse::transfer_item from
    // [vendored] sgl_kernel/deepseek_v4/kvcacheio.cuh) is not vendored; only IsDsv4Layout = false is instantiated.
    static_assert(!IsDsv4Layout, "[vendored] DSv4 page-padded layout is not vendored");  // [vendored] D6
  } else {
    // Generic path: device + host both linear, stride = item_size_bytes.
    const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
    auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
    transfer_item_warp(lane_id, src_k, dst_k, item_size_bytes);

    if constexpr (!IsMLA) {
      const auto src_v = static_cast<const char*>(host_cache_v) + src_loc * item_size_bytes;
      auto dst_v = static_cast<char*>(device_buffer_v) + dst_loc * item_size_bytes;
      transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
    }
  }
}

// Copy-only swap-in for shared-index skip layers: replays the anchor's recorded
// miss plan (no hit detection / LRU; the anchor's slot table stays valid). The
// small fixed grid (num_blocks) keeps the SM footprint low while overlapping
// compute on a side stream. SkipIO is the same probe as in the fused kernel.
template <int BLOCK_SIZE, bool IsMLA, bool IsDsv4Layout, bool SkipIO>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void copy_cache_planned_kernel(
    const int64_t* __restrict__ miss_src_locs,
    const int32_t* __restrict__ miss_dst_locs,
    const int32_t* __restrict__ miss_counts,
    const int32_t* __restrict__ num_real_reqs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int64_t plan_stride,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_global = blockIdx.x * NUM_WARPS + threadIdx.x / WARP_SIZE;
  const int total_warps = gridDim.x * NUM_WARPS;
  const int real = num_real_reqs[0];

  // Warp-sized windows amortize the miss_counts loads; warps then round-robin
  // the flattened (request, miss) space so a large sparse batch spreads over
  // all warps (183us -> 29us at bs=100 with 2 misses/req on H200) while one
  // request's miss burst still uses every warp.
  int start = 0;  // flat index of the current request's first miss
  for (int base = 0; base < real; base += WARP_SIZE) {
    const int r_lane = base + lane_id;
    const int cnt_lane = (r_lane < real) ? miss_counts[r_lane] : 0;
    const int window = (real - base < WARP_SIZE) ? (real - base) : WARP_SIZE;
    for (int j = 0; j < window; ++j) {
      const int cnt = __shfl_sync(FULL_WARP_MASK, cnt_lane, j);
      if (cnt == 0) continue;
      int m0 = (warp_global - start) % total_warps;
      if (m0 < 0) m0 += total_warps;
      const int64_t r = base + j;
      const int64_t* src_row = miss_src_locs + r * plan_stride;
      const int32_t* dst_row = miss_dst_locs + r * plan_stride;
      for (int m = m0; m < cnt; m += total_warps) {
        // Timing probe: the plan is still walked; only the bytes stay put.
        if constexpr (SkipIO) continue;
        copy_miss_item<IsMLA, IsDsv4Layout>(
            lane_id,
            host_cache_k,
            host_cache_v,
            device_buffer_k,
            device_buffer_v,
            src_row[m],
            static_cast<int64_t>(dst_row[m]),
            item_size_bytes);
      }
      start += cnt;
    }
  }
}

}  // namespace sglang

// [vendored] ==== BEGIN TORCH LAUNCHER (D8: replaces upstream :893-924; nothing below this line is upstream code) ====
// Our launcher. It converts torch tensors to the kernel's raw-pointer contract exactly as upstream's
// copy_cache_planned (:894-924) does, and launches on the CURRENT torch stream, so a caller that enters
// `with torch.cuda.stream(side):` gets the side-stream launch that SGLang's coordinator makes
// (hisparse_coordinator.py _run_copy_only_kernel, inside `with device_module.stream(self.prefetch_stream)`).
// Every dtype / device / stride / pinned / alignment check is in hisparse_copy.cpp; this file only launches.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// Keep in sync with kBlockSizes in hisparse_copy.cpp (tests grep both).
#define HISPARSE_COPY_BLOCK_SIZES 256, 1024

namespace {

template <int BLOCK_SIZE, bool IsMLA, bool SkipIO>
void launch_copy_cache_planned(
    const at::Tensor& miss_src_locs,
    const at::Tensor& miss_dst_locs,
    const at::Tensor& miss_counts,
    const at::Tensor& num_real_reqs,
    const at::Tensor& host_k,
    const at::Tensor& host_v,
    const at::Tensor& dev_k,
    const at::Tensor& dev_v,
    int64_t num_blocks,
    int64_t item_size_bytes) {
  static_assert(BLOCK_SIZE % sglang::WARP_SIZE == 0, "BLOCK_SIZE must be a whole number of warps");
  // upstream :907-910
  const int64_t plan_stride = miss_src_locs.stride(0);
  TORCH_CHECK(miss_dst_locs.stride(0) == plan_stride, "copy_cache_planned: miss_src/miss_dst row strides differ");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sglang::copy_cache_planned_kernel<BLOCK_SIZE, IsMLA, /*IsDsv4Layout=*/false, SkipIO>
      <<<static_cast<unsigned int>(num_blocks), BLOCK_SIZE, 0, stream>>>(
          miss_src_locs.data_ptr<int64_t>(),
          miss_dst_locs.data_ptr<int32_t>(),
          miss_counts.data_ptr<int32_t>(),
          num_real_reqs.data_ptr<int32_t>(),
          host_k.data_ptr(),
          IsMLA ? (const void*)nullptr : host_v.data_ptr(),  // upstream :919 (MLA: K only)
          dev_k.data_ptr(),
          IsMLA ? (void*)nullptr : dev_v.data_ptr(),         // upstream :921
          plan_stride,
          item_size_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void hisparse_copy_planned_cuda(
    const at::Tensor& miss_src_locs,
    const at::Tensor& miss_dst_locs,
    const at::Tensor& miss_counts,
    const at::Tensor& num_real_reqs,
    const at::Tensor& host_k,
    const at::Tensor& host_v,
    const at::Tensor& dev_k,
    const at::Tensor& dev_v,
    int64_t num_blocks,
    int64_t block_size,
    int64_t item_size_bytes,
    bool skip_io,
    bool is_mla) {
  const c10::cuda::CUDAGuard guard(miss_src_locs.device());
#define HISPARSE_COPY_ARGS \
  miss_src_locs, miss_dst_locs, miss_counts, num_real_reqs, host_k, host_v, dev_k, dev_v, num_blocks, item_size_bytes
  if (block_size == 256) {
    if (is_mla) {
      if (skip_io) launch_copy_cache_planned<256, true, true>(HISPARSE_COPY_ARGS);
      else launch_copy_cache_planned<256, true, false>(HISPARSE_COPY_ARGS);
    } else {
      if (skip_io) launch_copy_cache_planned<256, false, true>(HISPARSE_COPY_ARGS);
      else launch_copy_cache_planned<256, false, false>(HISPARSE_COPY_ARGS);
    }
  } else if (block_size == 1024) {
    if (is_mla) {
      if (skip_io) launch_copy_cache_planned<1024, true, true>(HISPARSE_COPY_ARGS);
      else launch_copy_cache_planned<1024, true, false>(HISPARSE_COPY_ARGS);
    } else {
      if (skip_io) launch_copy_cache_planned<1024, false, true>(HISPARSE_COPY_ARGS);
      else launch_copy_cache_planned<1024, false, false>(HISPARSE_COPY_ARGS);
    }
  } else {
    TORCH_CHECK(false, "block_size ", block_size, " is not instantiated; supported: 256, 1024");
  }
#undef HISPARSE_COPY_ARGS
}
