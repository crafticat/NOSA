// [vendored] SGLang HiCache JIT KV-cache IO kernel (hicache_transfer_per_layer), vendored for the retroinfer-eval
// [vendored] Strata / HiCache native-control and NOSI-adapter experiment.
// [vendored]   LABEL    SGLang's DEFAULT HiCache load kernel at this commit for element sizes % 128 == 0 on CUDA
// [vendored]            (pool_host/mha.py :100-107 can_use_jit, :266-300). Written by DarkSharpness (PR #13453, reverted,
// [vendored]            re-landed as #13764, 2025-11-23): NOT a Strata author and NOT the Strata paper's kernel. Harness
// [vendored]            arms named hicachejit<W>_... are THIS kernel; they are never labelled Strata.
// [vendored]   source   https://github.com/sgl-project/sglang
// [vendored]   commit   87db74302151956e354a05c01ecda47fd24cb882
// [vendored]   files    python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh
// [vendored]              sha256 dc34f219c0c18c9111df0383db55e88bd59ed0717c139f4907096202036b7dcc (524 lines)
// [vendored]            python/sglang/kernels/jit/include/sgl_kernel/utils.cuh (the helpers the kernel uses)
// [vendored]              sha256 d358c786170561de62ed94bf01e7837ca6748dc37fdad089f99464b600d8d8f6 (388 lines)
// [vendored]   license  Apache-2.0 (Copyright the SGLang team; see the upstream LICENSE)
// [vendored]
// [vendored] Pinned verbatim upstream files: retroinfer-eval scripts/sglang_hicache/hicache_upstream_87db743.cuh and
// [vendored] utils_upstream_87db743.cuh. "H:NNN" = a line of hicache.cuh, "U:NNN" = a line of utils.cuh.
// [vendored] retroinfer-eval tests/test_strata_transfer.py checks the claims below against them.
// [vendored]
// [vendored] VERBATIM (byte for byte once the lines tagged `// [vendored]` are removed), in this order:
// [vendored]   U:23-53    <concepts> <cstddef> <type_traits>, the CUDA headers (the ROCm #else branch is preprocessed out)
// [vendored]   U:55       `namespace sglang {`
// [vendored]   U:107-110  `namespace device {` and the SGL_DEVICE macro
// [vendored]   U:135-136  kWarpThreads
// [vendored]   U:208-223  pointer::offset (byte-level void-pointer arithmetic; needs C++20 concepts)
// [vendored]   U:232      `}  // namespace device`
// [vendored]   U:388      `}  // namespace sglang`
// [vendored]   H:11-266   <algorithm> <cstdint> <type_traits>, namespace sglang: load_nc / store_nc (CUDA: 16-B
// [vendored]              ld.global.L1::no_allocate.v4.b32 / st.global.L1::no_allocate.v4.b32, no memory clobber), load_vec /
// [vendored]              store_vec (every load of an item before its stores), SGL_HICACHE_KERNEL, HicacheKernelParams,
// [vendored]              hicache_transfer_per_layer (kWarpThreads / kUnroll threads per item, round-robin grid-stride
// [vendored]              over items, K then V) and hicache_transfer_all_layer (not launched here)
// [vendored]   H:521-524  `#undef SGL_HICACHE_KERNEL` and `}  // namespace sglang`
// [vendored]
// [vendored] EVERY TEXTUAL DIFFERENCE FROM UPSTREAM:
// [vendored]   D1  H:1-10   `#pragma once` and the sgl_kernel / dlpack includes dropped; the shim above supplies the three
// [vendored]              helpers the kernel uses (U lines, verbatim). Everything else in utils.cuh (fp types, PDL helpers,
// [vendored]              the SGL_CUDA_ARCH static_assert U:112-128, host::LaunchKernel) is not included.
// [vendored]   D2  H:267-520 the tvm-ffi host struct HiCacheKernel (run_one / run_all / run_one_mla / run_all_mla:
// [vendored]              TensorMatcher checks, host::LaunchKernel on TVMFFIEnvGetStream) -> the torch launcher below the
// [vendored]              BEGIN TORCH LAUNCHER marker. It keeps run_one's (H:281-348) checks as TORCH_CHECKs, its grid
// [vendored]              num_blocks = min(ceil(length / (kBlockSize / (kWarpThreads / kUnroll))), kBlockQuota) (H:333-334),
// [vendored]              its params (H:335-345) and its int32 / int64 index dispatch (H:330, :346), and launches
// [vendored]              <<<num_blocks, kBlockSize, 0, at::cuda::getCurrentCUDAStream()>>> instead of cudaLaunchKernelEx on
// [vendored]              the tvm-ffi environment stream (no launch attributes are set upstream: PDL is off by default).
// [vendored]              SGLang calls it inside `with device_module.stream(self.host_to_device_stream)`
// [vendored]              (mem_cache/l2_transfer.py :85), so the intended stream is the caller's current stream.
// [vendored]              Instantiated for kElementSize in {256, 512}, kUnroll = 4 (ops/kvcache/hicache.py :124-132
// [vendored]              _default_unroll for <= 512 B), kBlockQuota in {1, 2, 4, 8, 16} (upstream JIT-builds one module per
// [vendored]              quota, hicache.py :24-44; default 2 on CUDA, :21), kBlockSize = 1024 (hicache.py :30), kIsMLA = false.
// [vendored]              length == 0 is refused by a TORCH_CHECK (upstream would launch a 0-block grid and fail the launch).
// [vendored]   D3  this header and blank lines between the vendored blocks
// [vendored] BUILD FLAGS: upstream JIT = -DSGL_CUDA_ARCH=800 -std=c++20 -O3 --expt-relaxed-constexpr -gencode sm_80
// [vendored] (jit/utils/arch.py :130-136); this file = torch's nvcc defaults (incl. --expt-relaxed-constexpr) + -std=c++20
// [vendored] -O3 -DNDEBUG, sm_80. SGL_CUDA_ARCH is only read by the dropped static_assert and the unused kMaxVecBytes.
#include <concepts>
#include <cstddef>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef __grid_constant__
#define __grid_constant__
#endif
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
inline constexpr auto cudaSuccess = hipSuccess;
#define cudaStreamPerThread hipStreamPerThread
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif

namespace sglang {

namespace device {

/// \brief Macro: forced-inline device function qualifier.
#define SGL_DEVICE __forceinline__ __device__

/// \brief Number of threads per warp (always 32 on NVIDIA/AMD GPUs).
inline constexpr auto kWarpThreads = 32u;

/// \brief Safe void-pointer arithmetic (byte-level by default).
namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

}  // namespace device

}  // namespace sglang

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device {

namespace details {

template <typename T, uint32_t N>
struct LocalStorage {
  T data[N];
};

template <int kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <int kUnit>
using PackageType = decltype(get_mem_package<kUnit>());

// NVIDIA exposes an explicit "do not allocate in L1" cache hint via PTX. ROCm
// has no equivalent PTX, but non-temporal (streaming) loads/stores express the
// same intent for one-shot HiCache write-back traffic that should not pollute
// the cache. Guard the PTX behind USE_ROCM so the JIT module also compiles with
// hipcc; see python/sglang/kernels/jit/utils/compile.py for the ROCm build flags.
#ifdef USE_ROCM
// Native Clang vector types so a single __builtin_nontemporal_{load,store} maps
// to one vectorized global_{load,store}_dwordx{2,4}. Issuing N independent
// 32-bit nontemporal ops instead leaves merging to the LoadStoreVectorizer,
// which is not guaranteed and may drop the nontemporal hint, throttling HiCache
// bandwidth. uint2/uint4 already carry 8B/16B alignment matching the vector
// types, so the pointer reinterpret_casts stay correctly aligned.
typedef uint32_t native_uint2 __attribute__((ext_vector_type(2)));
typedef uint32_t native_uint4 __attribute__((ext_vector_type(4)));
#endif

SGL_DEVICE uint1 load_nc(const uint1* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
#else
  return uint1{__builtin_nontemporal_load(&src->x)};
#endif
}

SGL_DEVICE uint2 load_nc(const uint2* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
#else
  native_uint2 tmp = __builtin_nontemporal_load(reinterpret_cast<const native_uint2*>(src));
  return __builtin_bit_cast(uint2, tmp);
#endif
}

SGL_DEVICE uint4 load_nc(const uint4* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
#else
  native_uint4 tmp = __builtin_nontemporal_load(reinterpret_cast<const native_uint4*>(src));
  return __builtin_bit_cast(uint4, tmp);
#endif
}

SGL_DEVICE void store_nc(uint1* __restrict__ dst, const uint1& value) {
#ifndef USE_ROCM
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
#else
  __builtin_nontemporal_store(value.x, &dst->x);
#endif
}

SGL_DEVICE void store_nc(uint2* __restrict__ dst, const uint2& value) {
#ifndef USE_ROCM
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
#else
  __builtin_nontemporal_store(__builtin_bit_cast(native_uint2, value), reinterpret_cast<native_uint2*>(dst));
#endif
}

SGL_DEVICE void store_nc(uint4* __restrict__ dst, const uint4& value) {
#ifndef USE_ROCM
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
#else
  __builtin_nontemporal_store(__builtin_bit_cast(native_uint4, value), reinterpret_cast<native_uint4*>(dst));
#endif
}

}  // namespace details

template <int64_t kBytes, uint32_t kNumThreads>
SGL_DEVICE auto load_vec(const void* __restrict__ src) {
  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;
  using Package = details::PackageType<128 / kNumThreads>;
  using Storage = details::LocalStorage<Package, kLoopCount>;

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kNumThreads;
  Storage vec;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    vec.data[i] = details::load_nc(&src_packed[j]);
  }

  return vec;
}

template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
SGL_DEVICE void store_vec(void* __restrict__ dst, const Storage& vec) {
  using Package = std::decay_t<decltype(vec.data[0])>;
  constexpr uint32_t kBytesPerLoop = sizeof(Package) * kNumThreads;
  constexpr uint32_t kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "Invalid Storage configuration");

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kNumThreads;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    details::store_nc(&dst_packed[j], vec.data[i]);
  }
}

}  // namespace device

#define SGL_HICACHE_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  int64_t kv_cache_src_stride;
  int64_t kv_cache_dst_stride;
  uint32_t length;
  uint32_t num_layers = 0;  // only used in all_layer transfer
};

template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, _ // metadata
  ] = params;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
    store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
      store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
    }
  }
}

template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_all_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using src_ptr_t = const void*;
  using dst_ptr_t = void*;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_ptr_dst, v_ptr_dst, indices_dst, // dst
    k_ptr_src, v_ptr_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, num_layers // metadata
  ] = params;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      const auto k_cache_src = static_cast<const src_ptr_t*>(k_ptr_src)[layer];
      const auto k_cache_dst = static_cast<const dst_ptr_t*>(k_ptr_dst)[layer];
      const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
      store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
      if constexpr (!kIsMLA) {
        const auto v_cache_src = static_cast<const src_ptr_t*>(v_ptr_src)[layer];
        const auto v_cache_dst = static_cast<const dst_ptr_t*>(v_ptr_dst)[layer];
        const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
        const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
        const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
        store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
      }
    }
  }
}

#undef SGL_HICACHE_KERNEL

}  // namespace sglang

// [vendored] ==== BEGIN TORCH LAUNCHER (D2: replaces hicache.cuh H:267-520; nothing below this line is upstream code) ====
// Our launcher. It mirrors upstream HiCacheKernel<...>::run_one (H:281-348): the same tensor contract
// (TensorMatcher({-1, D}).with_strides({N, 1}) for the caches, TensorMatcher({L}) int32 / int64 CUDA indices),
// the same element-size check, grid and params, launched on the CURRENT torch stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// Keep in sync with kJitElementSizes / kJitBlockQuotas in jit_binding.cpp (tests grep both).
#define HICACHE_JIT_ELEMENT_SIZES 256, 512
#define HICACHE_JIT_BLOCK_QUOTAS 1, 2, 4, 8, 16

namespace {

constexpr uint32_t kJitUnroll = 4;        // ops/kvcache/hicache.py :124-132 (_default_unroll: element_size <= 512 -> 4)
constexpr uint32_t kJitBlockSize = 1024;  // ops/kvcache/hicache.py :30 (num_threads)

void check_cache_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D (-1, D) (H:298, :304; got ", t.dim(), "-D)");
  TORCH_CHECK(t.stride(1) == 1, name, " must have unit inner stride (H:299, :305 with_strides({N, 1}))");
  TORCH_CHECK(t.is_cuda() || t.is_pinned(), name, " must be CUDA or page-locked host memory (H:301 kDLGPU / kDLGPUHost / kDLCPU "
              "read through UVA; this launcher refuses pageable host memory, which the device cannot address)");
}

template <int64_t kElementSize, uint32_t kBlockQuota>
void run_one_torch(
    const at::Tensor& k_cache_dst,
    const at::Tensor& v_cache_dst,
    const at::Tensor& indices_dst,
    const at::Tensor& k_cache_src,
    const at::Tensor& v_cache_src,
    const at::Tensor& indices_src) {
  // H:298-314: the caches share D and dtype; src K / V share one row stride N, dst K / V one row stride M
  check_cache_2d(k_cache_src, "k_cache_src");
  check_cache_2d(v_cache_src, "v_cache_src");
  check_cache_2d(k_cache_dst, "k_cache_dst");
  check_cache_2d(v_cache_dst, "v_cache_dst");
  const int64_t D = k_cache_src.size(1);
  TORCH_CHECK(v_cache_src.size(1) == D && k_cache_dst.size(1) == D && v_cache_dst.size(1) == D, "head dimension mismatch (H:290 D)");
  TORCH_CHECK(v_cache_src.stride(0) == k_cache_src.stride(0), "src K and V row strides differ (H:291 N)");
  TORCH_CHECK(v_cache_dst.stride(0) == k_cache_dst.stride(0), "dst K and V row strides differ (H:292 M)");
  const auto dtype = k_cache_src.scalar_type();
  TORCH_CHECK(v_cache_src.scalar_type() == dtype && k_cache_dst.scalar_type() == dtype && v_cache_dst.scalar_type() == dtype,
              "cache dtypes differ (H:294 cache_dtype)");
  TORCH_CHECK(indices_src.dim() == 1 && indices_dst.dim() == 1 && indices_src.numel() == indices_dst.numel(),
              "indices must be 1-D of one length (H:310 TensorMatcher({L}))");
  TORCH_CHECK(indices_src.scalar_type() == indices_dst.scalar_type() &&
                  (indices_src.scalar_type() == at::kInt || indices_src.scalar_type() == at::kLong),
              "indices must share int32 or int64 (H:311)");
  TORCH_CHECK(indices_src.is_cuda() && indices_dst.is_cuda() && indices_src.device() == indices_dst.device(),
              "indices must be on one CUDA device (H:312 with_device<kDLGPU>)");
  TORCH_CHECK(indices_src.is_contiguous() && indices_dst.is_contiguous(), "indices must be contiguous (the kernel reads indices[i])");
  const int64_t dtype_size = k_cache_src.element_size();
  const int64_t element_bytes = D * dtype_size;
  TORCH_CHECK(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch. (H:319)");
  const auto length = static_cast<uint32_t>(indices_src.numel());
  TORCH_CHECK(length > 0, "length == 0: upstream would launch a 0-block grid (launch error); refused here (header D2)");
  const auto kv_cache_src_stride = static_cast<int64_t>(k_cache_src.stride(0) * dtype_size);  // H:328
  const auto kv_cache_dst_stride = static_cast<int64_t>(k_cache_dst.stride(0) * dtype_size);  // H:329
  const bool use_int32 = indices_src.scalar_type() == at::kInt;                                 // H:330
  const c10::cuda::CUDAGuard guard(indices_src.device());

  constexpr auto kWorkersPerBlock = kJitBlockSize / (sglang::device::kWarpThreads / kJitUnroll);  // H:333
  const auto num_blocks = std::min((length + kWorkersPerBlock - 1) / kWorkersPerBlock, kBlockQuota);  // H:334
  const auto params = sglang::HicacheKernelParams{  // H:335-345
      .k_cache_dst = k_cache_dst.data_ptr(),
      .v_cache_dst = v_cache_dst.data_ptr(),
      .indices_dst = indices_dst.data_ptr(),
      .k_cache_src = const_cast<void*>(k_cache_src.data_ptr()),
      .v_cache_src = const_cast<void*>(v_cache_src.data_ptr()),
      .indices_src = indices_src.data_ptr(),
      .kv_cache_src_stride = kv_cache_src_stride,
      .kv_cache_dst_stride = kv_cache_dst_stride,
      .length = length,
  };
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (use_int32) {  // H:346
    sglang::hicache_transfer_per_layer<int32_t, kElementSize, kJitUnroll, kBlockQuota, kJitBlockSize>
        <<<num_blocks, kJitBlockSize, 0, stream>>>(params);
  } else {
    sglang::hicache_transfer_per_layer<int64_t, kElementSize, kJitUnroll, kBlockQuota, kJitBlockSize>
        <<<num_blocks, kJitBlockSize, 0, stream>>>(params);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int64_t kElementSize>
void dispatch_quota(int64_t block_quota, const at::Tensor& kd, const at::Tensor& vd, const at::Tensor& id, const at::Tensor& ks,
                    const at::Tensor& vs, const at::Tensor& is) {
  switch (block_quota) {
    case 1: return run_one_torch<kElementSize, 1>(kd, vd, id, ks, vs, is);
    case 2: return run_one_torch<kElementSize, 2>(kd, vd, id, ks, vs, is);
    case 4: return run_one_torch<kElementSize, 4>(kd, vd, id, ks, vs, is);
    case 8: return run_one_torch<kElementSize, 8>(kd, vd, id, ks, vs, is);
    case 16: return run_one_torch<kElementSize, 16>(kd, vd, id, ks, vs, is);
    default: TORCH_CHECK(false, "block_quota ", block_quota, " is not instantiated; supported: 1, 2, 4, 8, 16");
  }
}

}  // namespace

void hicache_jit_transfer_one(
    const at::Tensor& k_cache_dst,
    const at::Tensor& v_cache_dst,
    const at::Tensor& indices_dst,
    const at::Tensor& k_cache_src,
    const at::Tensor& v_cache_src,
    const at::Tensor& indices_src,
    int64_t element_size,
    int64_t block_quota) {
  if (element_size == 256) {
    dispatch_quota<256>(block_quota, k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src, indices_src);
  } else if (element_size == 512) {
    dispatch_quota<512>(block_quota, k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src, indices_src);
  } else {
    TORCH_CHECK(false, "element_size ", element_size, " is not instantiated; supported: 256, 512");
  }
}
