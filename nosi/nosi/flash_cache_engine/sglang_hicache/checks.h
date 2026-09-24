// Argument checks shared by the two pybind wrappers (aot_binding.cpp, jit_binding.cpp) of the vendored SGLang HiCache
// IO kernels (sglang 87db743). Not upstream code.
//
// Both kernels index raw pointers without bounds checks (transfer.cu :283-289 reads src/dst_indices[item] and
// addresses base + index * item_size; hicache.cuh H:204-208 addresses base + index * row stride), so every
// assumption about the tensors is checked before the launch. Index VALUES are checked once per prebuilt plan in
// plan.py (validate_indices), not per launch: reading device indices here would synchronize every launch.
//
// Host memory: SGLang registers a CPU tensor with cudaHostRegister(ptr, size, 0) (pool_host/common.py :124-160) and
// the kernels dereference the HOST pointer on the device. That is valid only when the device pointer of the
// registered range equals the host pointer (cudaDevAttrCanUseHostPointerForRegisteredMem, UVA).
#pragma once
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdint>

namespace sglang_hicache_checks {

inline int64_t nbytes(const at::Tensor& t) { return t.numel() * t.element_size(); }

inline bool aligned(const at::Tensor& t, uintptr_t a) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) % a) == 0; }

inline bool device_ptr_equals_host_ptr(const void* p) {
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeHost && attr.devicePointer == p && attr.hostPointer == p;
}

inline void check_side(const at::Tensor& t, bool host, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  if (host) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor (got ", t.device(), ")");
    TORCH_CHECK(t.is_pinned(), name, " must be page-locked (pinned or cudaHostRegister'ed): the kernel reads it through UVA");
    TORCH_CHECK(device_ptr_equals_host_ptr(t.data_ptr()), name,
                ": the device pointer of this host buffer is not its host pointer; the kernel would read the wrong address");
  } else {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor (got ", t.device(), ")");
  }
}

inline void check_indices(const at::Tensor& s, const at::Tensor& d, bool allow_int32) {
  TORCH_CHECK(s.defined() && d.defined() && s.is_cuda() && d.is_cuda() && s.device() == d.device(),
              "indices must be CUDA tensors on one device");
  TORCH_CHECK(s.dim() == 1 && d.dim() == 1 && s.numel() == d.numel() && s.numel() > 0, "indices must be 1-D, non-empty, of one length");
  TORCH_CHECK(s.is_contiguous() && d.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(s.scalar_type() == d.scalar_type(), "src / dst indices must share a dtype");
  TORCH_CHECK(s.scalar_type() == at::kLong || (allow_int32 && s.scalar_type() == at::kInt),
              allow_int32 ? "indices must be int32 or int64 (hicache.cuh H:311)" : "indices must be int64 (transfer.cu :333-334)");
}

}  // namespace sglang_hicache_checks
