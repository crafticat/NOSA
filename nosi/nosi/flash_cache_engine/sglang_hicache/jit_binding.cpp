// pybind wrapper for the vendored SGLang HiCache JIT kernel (hicache_jit.cu = hicache.cuh @87db743, device code verbatim,
// torch launcher). SGLang's DEFAULT load at 87db743 for 512-B elements; NOT by a Strata author. Harness arms
// hicachejit<W>_... call ONLY this. hicache_jit.cu is compiled with -std=c++20 as upstream's JIT (jit/utils/arch.py
// :130-136); this file is plain C++17 host code.
#include "checks.h"

#include <algorithm>
#include <string>
#include <vector>

// hicache_jit.cu
void hicache_jit_transfer_one(
    const at::Tensor& k_cache_dst,
    const at::Tensor& v_cache_dst,
    const at::Tensor& indices_dst,
    const at::Tensor& k_cache_src,
    const at::Tensor& v_cache_src,
    const at::Tensor& indices_src,
    int64_t element_size,
    int64_t block_quota);

namespace {

using namespace sglang_hicache_checks;

// Keep in sync with HICACHE_JIT_ELEMENT_SIZES / HICACHE_JIT_BLOCK_QUOTAS in hicache_jit.cu (tests grep both).
const std::vector<int64_t> kJitElementSizes = {256, 512};
const std::vector<int64_t> kJitBlockQuotas = {1, 2, 4, 8, 16};
const char* kUpstreamCommit = "87db74302151956e354a05c01ecda47fd24cb882";

// (-1, D) views with unit inner stride and any row stride (hicache.cuh H:298-309).
void jit_per_layer(at::Tensor k_cache_dst, at::Tensor v_cache_dst, at::Tensor indices_dst, at::Tensor k_cache_src,
                   at::Tensor v_cache_src, at::Tensor indices_src, int64_t element_size, int64_t block_quota) {
  const bool host = k_cache_src.device().is_cpu();
  check_side(k_cache_src, host, "k_cache_src");
  check_side(v_cache_src, host, "v_cache_src");
  check_side(k_cache_dst, false, "k_cache_dst");
  check_side(v_cache_dst, false, "v_cache_dst");
  check_indices(indices_src, indices_dst, true);
  TORCH_CHECK(k_cache_dst.device() == indices_src.device() && v_cache_dst.device() == indices_src.device(),
              "dst and indices must share a device");
  TORCH_CHECK(std::find(kJitElementSizes.begin(), kJitElementSizes.end(), element_size) != kJitElementSizes.end(),
              "element_size ", element_size, " is not instantiated (256, 512)");
  TORCH_CHECK(std::find(kJitBlockQuotas.begin(), kJitBlockQuotas.end(), block_quota) != kJitBlockQuotas.end(),
              "block_quota ", block_quota, " is not instantiated (1, 2, 4, 8, 16)");
  for (const at::Tensor* t : {&k_cache_src, &v_cache_src, &k_cache_dst, &v_cache_dst}) {
    TORCH_CHECK(t->dim() == 2 && t->stride(1) == 1, "caches must be 2-D (-1, D) views with unit inner stride (H:298-305)");
    TORCH_CHECK(aligned(*t, 16) && (t->stride(0) * t->element_size()) % 16 == 0,
                "every cache row must be 16-byte aligned (uint4 ld/st of 16 B, hicache.cuh H:79-90, :111-122)");
  }
  hicache_jit_transfer_one(k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src, indices_src, element_size, block_quota);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jit_per_layer", &jit_per_layer,
        "hicache.cuh hicache_transfer_per_layer (87db743, vendored): kUnroll 4, 1024 threads, one instantiation per quota",
        pybind11::arg("k_cache_dst"), pybind11::arg("v_cache_dst"), pybind11::arg("indices_dst"), pybind11::arg("k_cache_src"),
        pybind11::arg("v_cache_src"), pybind11::arg("indices_src"), pybind11::arg("element_size"), pybind11::arg("block_quota") = 2);
  m.def("jit_element_sizes", []() { return kJitElementSizes; }, "kElementSize values the JIT kernel is instantiated for");
  m.def("jit_block_quotas", []() { return kJitBlockQuotas; }, "kBlockQuota values the JIT kernel is instantiated for");
  m.def("upstream_commit", []() { return std::string(kUpstreamCommit); }, "SGLang commit the kernel is vendored from");
}
