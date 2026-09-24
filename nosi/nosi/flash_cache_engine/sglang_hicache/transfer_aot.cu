// [vendored] SGLang HiCache AOT KV-cache IO kernel (sgl-kernel kvcacheio transfer.cu), vendored for the
// [vendored] retroinfer-eval Strata / HiCache native-control and NOSI-adapter experiment.
// [vendored]   LABEL    CLOSEST RELEASED IMPLEMENTATION of Strata's GPU-assisted IO (arXiv 2508.18572, OSDI '26).
// [vendored]            The Strata paper releases no code of its own; this is SGLang's AOT kvcacheio kernel, written by
// [vendored]            Strata's first author (PRs #7382 = sgl-kernel split of #7313, and #8264). No PR, blog or doc calls
// [vendored]            it "Strata"; whether the paper's Fig. 5 kernel is byte-identical to it is UNVERIFIED.
// [vendored]            Harness arms named strata<W>_... are THIS kernel and nothing else. HiSparse is never Strata.
// [vendored]   source   https://github.com/sgl-project/sglang
// [vendored]   commit   87db74302151956e354a05c01ecda47fd24cb882
// [vendored]   file     python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu
// [vendored]   sha256   a2f14a72b1720bd3bf74e49f9579e6940a7d92d4aff0ccf64c57629ea991e342 (1263 lines)
// [vendored]   license  Apache-2.0 (Copyright the SGLang team; see the upstream LICENSE)
// [vendored]
// [vendored] The pinned verbatim upstream file is retroinfer-eval scripts/sglang_hicache/transfer_upstream_87db743.cu;
// [vendored] every ":NNN" in this directory is a line number of THAT file. retroinfer-eval
// [vendored] tests/test_strata_transfer.py checks the claims below against it.
// [vendored]
// [vendored] VERBATIM (byte for byte once the lines tagged `// [vendored]` are removed):
// [vendored]   :1-13      includes, `#if !defined(USE_ROCM) && !defined(USE_MUSA)`, <dlfcn.h>, `#define WARP_SIZE 32`
// [vendored]   :15-461    the rest of the include block (the ROCm branch is removed by the preprocessor on CUDA),
// [vendored]              transfer_item_warp (CUDA :20-33: one warp per item, 8-B ld.global.nc.b64 / st.global.cg.b64,
// [vendored]              lane j copies words j, j+32, ...; ROCm :34-71 and MUSA :72-88 branches preprocessed out),
// [vendored]              the offset functions :90-172, transfer_page_head_kernel_impl :174-258 (named by the launcher's
// [vendored]              discarded constexpr branch, never instantiated here), transfer_kernel_impl :260-308 (static
// [vendored]              contiguous split: warp w copies items [w*ipw, (w+1)*ipw); K then V at the same index),
// [vendored]              transfer_kv_launcher :310-397 (upstream's OWN torch launcher: grid = ceil(N / (ipw * warps))
// [vendored]              <= block_quota, threads = warps x 32, launched on at::cuda::getCurrentCUDAStream()),
// [vendored]              transfer_kv_per_layer :399-428 (layer_first -> layer_first) and
// [vendored]              transfer_kv_per_layer_pf_lf :430-461 (page_first host -> layer_first device)
// [vendored]
// [vendored] EVERY TEXTUAL DIFFERENCE FROM UPSTREAM:
// [vendored]   D1  :14    `#include "pytorch_extension_utils.h"` -> `#include <torch/library.h>`. Upstream's header comes from
// [vendored]              flashinfer bc29697 csrc/pytorch_extension_utils.h (430 lines; fetched through the sgl-kernel build,
// [vendored]              CMakeLists.txt :76-79): it includes <Python.h>, c10 CUDAGuard / CUDAStream, <torch/library.h>
// [vendored]              and the cuda bf16 / fp16 / fp8 / fp4 headers, and defines dispatch / check macros that :20-461
// [vendored]              do not use. :20-461 need only the complete at::Tensor, which <torch/library.h> provides.
// [vendored]   D2  :462-1263 dropped: transfer_kv_per_layer_ph_lf, the all-layer and MLA entry points, and the
// [vendored]              copy-engine paths (transfer_kv_direct, transfer_embedding_ranges_direct, page_first_direct /
// [vendored]              cudaMemcpyBatchAsync). Not called by this experiment.
// [vendored]   D3  this header and the trailer below
// [vendored] NO launcher change: upstream's transfer_kv_launcher is already a torch launcher on the current stream.
// [vendored] The op REGISTRATION differs: upstream registers the two entry points with TORCH_LIBRARY_FRAGMENT
// [vendored] (common_extension.cc :262-268, torch.ops.sgl_kernel.*); aot_binding.cpp exposes the same functions
// [vendored] through pybind with extra argument checks (pinned host memory, device pointer == host pointer).
// [vendored] BUILD FLAGS differ: upstream sgl-kernel builds with C++17, -O3, -DNDEBUG, --expt-relaxed-constexpr,
// [vendored] --expt-extended-lambda (CMakeLists.txt :122-138; the sm90 library adds -use_fast_math, :322); this
// [vendored] extension (sglang_hicache_aot_87db743) builds with torch's nvcc defaults + -O3 -DNDEBUG --expt-extended-lambda,
// [vendored] C++17 (torch default), sm_80.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include <dlfcn.h>
#define WARP_SIZE 32
#include <torch/library.h>  // [vendored] D1: replaces upstream :14 `#include "pytorch_extension_utils.h"` (see the header)
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"  // WARP_SIZE
#endif

#if !defined(USE_ROCM) && !defined(USE_MUSA)
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
  }
}
#elif defined(USE_ROCM)
// ROCm: use 128-bit streaming load/store when 16B-aligned, so fewer CUs are
// needed to saturate the host fabric; falls back to 64-bit otherwise.
typedef uint32_t sgl_u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uintptr_t addr_or = reinterpret_cast<uintptr_t>(src_addr) | reinterpret_cast<uintptr_t>(dst_addr);
  if ((addr_or & 0xF) == 0) {
    const sgl_u32x4* __restrict__ src = static_cast<const sgl_u32x4*>(src_addr);
    sgl_u32x4* __restrict__ dst = static_cast<sgl_u32x4*>(dst_addr);
    const int chunks16 = item_size_bytes / 16;
    for (int j = lane_id; j < chunks16; j += WARP_SIZE) {
      sgl_u32x4 tmp = __builtin_nontemporal_load(src + j);
      __builtin_nontemporal_store(tmp, dst + j);
    }
    // Trailing bytes: item_size_bytes % 8 == 0 is guaranteed by the launcher,
    // so the remainder is at most one 8B word.
    const int done_bytes = chunks16 * 16;
    const int rem8 = static_cast<int>(item_size_bytes - done_bytes) / 8;
    if (rem8) {
      const uint64_t* __restrict__ src8 =
          reinterpret_cast<const uint64_t*>(static_cast<const char*>(src_addr) + done_bytes);
      uint64_t* __restrict__ dst8 = reinterpret_cast<uint64_t*>(static_cast<char*>(dst_addr) + done_bytes);
      for (int j = lane_id; j < rem8; j += WARP_SIZE) {
        uint64_t tmp = __builtin_nontemporal_load(src8 + j);
        __builtin_nontemporal_store(tmp, dst8 + j);
      }
    }
  } else {
    const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
    uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
    const int total_chunks = item_size_bytes / sizeof(uint64_t);
    for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
      uint64_t tmp = __builtin_nontemporal_load(src + j);
      __builtin_nontemporal_store(tmp, dst + j);
    }
  }
}
#else
// MUSA: keep the original scalar nontemporal load/store path; the 128-bit
// ROCm path above relies on a HIP/clang ext_vector_type extension that isn't
// guaranteed to be available/correct under the MUSA compiler.
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
  }
}
#endif

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // page first
  return base + page_id * page_dim + layer_id * item_size_bytes;
}

// get offset from layer base table when layers are not contiguous
template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  // layer first offset func per head
  return base + layer_id * layer_dim + page_id * item_size_bytes + item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_ph(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  // page head layout: [page_num, head_num, page_size, layer_num, head_dim]
  return base + page_id / page_size * page_size * page_dim +  // page_num dimension offset
         page_dim / head_num * head_id * page_size +          // head_num dimension offset
         page_id % page_size * page_dim / head_num +          // page_size dimension offset
         layer_id * item_size_bytes / head_num;               // layer_num dimension offset
}

template <auto SrcOffsetFn, auto DstOffsetFn>
__global__ void transfer_page_head_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl,
    const int64_t page_size,
    const int64_t head_num) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;
  const int64_t head_size_bytes = item_size_bytes / head_num;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      // For page head layout, the cache of each head in the token is discontinuous, need to loop
      for (int64_t head_id = 0; head_id < head_num; ++head_id) {
        const char* src_k_ptr = SrcOffsetFn(
            static_cast<const char*>(src_k),
            src_k_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_k_ptr = DstOffsetFn(
            static_cast<char*>(dst_k),
            dst_k_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_k_ptr, dst_k_ptr, head_size_bytes);

        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v),
            src_v_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v),
            dst_v_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, head_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
__global__ void transfer_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);

      if constexpr (!IsMLA) {
        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA, bool PageHeadLayout = false>
void transfer_kv_launcher(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const at::Tensor& src_k_layers,
    const at::Tensor& dst_k_layers,
    const at::Tensor& src_v_layers,
    const at::Tensor& dst_v_layers,
    int64_t block_quota,
    int64_t num_warps_per_block,
    const int64_t page_size = 16,
    const int64_t head_num = 1) {
  TORCH_CHECK(src_indices.is_cuda(), "Source indices must be a CUDA tensor");
  TORCH_CHECK(dst_indices.is_cuda(), "Destination indices must be a CUDA tensor");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

  auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  const int64_t num_items = src_indices.numel();
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
  dim3 grid_dim(num_blocks, 1, 1);
  const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;

  const void* src_k_ptr = src_k.defined() ? src_k.data_ptr() : nullptr;
  void* dst_k_ptr = dst_k.defined() ? dst_k.data_ptr() : nullptr;
  const void* src_v_ptr = IsMLA || !src_v.defined() ? nullptr : src_v.data_ptr();
  void* dst_v_ptr = IsMLA || !dst_v.defined() ? nullptr : dst_v.data_ptr();
  const uintptr_t* src_k_tbl_ptr = src_k_layers.defined() ? src_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* dst_k_tbl_ptr = dst_k_layers.defined() ? dst_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* src_v_tbl_ptr = IsMLA || !src_v_layers.defined() ? nullptr : src_v_layers.data_ptr<uintptr_t>();
  const uintptr_t* dst_v_tbl_ptr = IsMLA || !dst_v_layers.defined() ? nullptr : dst_v_layers.data_ptr<uintptr_t>();

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  if constexpr (PageHeadLayout) {
    transfer_page_head_kernel_impl<SrcOffsetFn, DstOffsetFn><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
        src_k_ptr,
        dst_k_ptr,
        src_v_ptr,
        dst_v_ptr,
        src_indices.data_ptr<int64_t>(),
        dst_indices.data_ptr<int64_t>(),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_tbl_ptr,
        dst_k_tbl_ptr,
        src_v_tbl_ptr,
        dst_v_tbl_ptr,
        page_size,
        head_num);
  } else {
    transfer_kernel_impl<SrcOffsetFn, DstOffsetFn, IsMLA><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
        src_k_ptr,
        dst_k_ptr,
        src_v_ptr,
        dst_v_ptr,
        src_indices.data_ptr<int64_t>(),
        dst_indices.data_ptr<int64_t>(),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_tbl_ptr,
        dst_k_tbl_ptr,
        src_v_tbl_ptr,
        dst_v_tbl_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_pf_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}
// [vendored] D2: upstream :462-1263 dropped (see the header). End of the vendored file.
