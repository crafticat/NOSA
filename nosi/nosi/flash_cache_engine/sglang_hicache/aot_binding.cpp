// pybind wrapper for the vendored SGLang AOT kvcacheio kernel (transfer_aot.cu = sgl-kernel transfer.cu @87db743
// :1-13 + :15-461, verbatim). CLOSEST RELEASED IMPLEMENTATION of Strata's IO kernel (written by Strata's first author;
// the paper releases no code). Harness arms strata<W>_... call ONLY these two functions.
// Upstream registers the same entry points with TORCH_LIBRARY_FRAGMENT (common_extension.cc :262-268); this file adds
// the checks of checks.h before calling them. Built with torch's default C++17, as upstream (CMakeLists.txt :23, :129).
#include "checks.h"

#include <string>

// transfer_aot.cu (upstream :399-428, :430-461)
void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);
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
    int64_t num_warps_per_block);

namespace {

using namespace sglang_hicache_checks;

const char* kUpstreamCommit = "87db74302151956e354a05c01ecda47fd24cb882";

void check_quota(int64_t block_quota, int64_t num_warps_per_block) {
  TORCH_CHECK(block_quota >= 1 && block_quota <= 1024 && num_warps_per_block >= 1 && num_warps_per_block <= 32,
              "block_quota in [1, 1024] and num_warps_per_block in [1, 32] (got ", block_quota, ", ", num_warps_per_block, ")");
}

// layer_first -> layer_first (get_global_offset_lf, :90-100): record = base + index * item_size on BOTH sides, so each
// side must be one contiguous array of item_size records.
void aot_per_layer(at::Tensor src_k, at::Tensor dst_k, at::Tensor src_v, at::Tensor dst_v, at::Tensor src_indices,
                   at::Tensor dst_indices, int64_t item_size, int64_t block_quota, int64_t num_warps_per_block) {
  const bool host = src_k.device().is_cpu();
  check_side(src_k, host, "src_k");
  check_side(src_v, host, "src_v");
  check_side(dst_k, false, "dst_k");
  check_side(dst_v, false, "dst_v");
  check_indices(src_indices, dst_indices, false);
  TORCH_CHECK(dst_k.device() == src_indices.device() && dst_v.device() == src_indices.device(), "dst and indices must share a device");
  TORCH_CHECK(item_size > 0 && item_size % 8 == 0, "item_size must be a positive multiple of 8 (transfer.cu :336)");
  for (const at::Tensor* t : {&src_k, &src_v, &dst_k, &dst_v}) {
    TORCH_CHECK(t->is_contiguous(), "every cache must be contiguous: records are base + index * item_size (transfer.cu :99)");
    TORCH_CHECK(aligned(*t, 8), "every cache must be 8-byte aligned (8-B ld.global.nc.b64, transfer.cu :30)");
    TORCH_CHECK(nbytes(*t) % item_size == 0, "a cache is not a whole number of ", item_size, "-byte records");
  }
  TORCH_CHECK(nbytes(src_k) == nbytes(src_v) && nbytes(dst_k) == nbytes(dst_v), "K and V must have the same byte size (one index serves both)");
  check_quota(block_quota, num_warps_per_block);
  transfer_kv_per_layer(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, block_quota, num_warps_per_block);
}

// page_first host -> layer_first device (get_global_offset_pf, :102-112): src record = base + index * src_layout_dim +
// layer_id * item_size, with src_k / src_v the WHOLE (tokens, layers, H, Dh) buffers (SGLang mha.py :301-312).
void aot_per_layer_pf_lf(at::Tensor src_k, at::Tensor dst_k, at::Tensor src_v, at::Tensor dst_v, at::Tensor src_indices,
                         at::Tensor dst_indices, int64_t layer_id, int64_t item_size, int64_t src_layout_dim, int64_t block_quota,
                         int64_t num_warps_per_block) {
  const bool host = src_k.device().is_cpu();
  check_side(src_k, host, "src_k");
  check_side(src_v, host, "src_v");
  check_side(dst_k, false, "dst_k");
  check_side(dst_v, false, "dst_v");
  check_indices(src_indices, dst_indices, false);
  TORCH_CHECK(dst_k.device() == src_indices.device() && dst_v.device() == src_indices.device(), "dst and indices must share a device");
  for (const at::Tensor* t : {&src_k, &src_v, &dst_k, &dst_v}) {
    TORCH_CHECK(t->is_contiguous() && aligned(*t, 8), "every cache must be contiguous and 8-byte aligned");
  }
  TORCH_CHECK(item_size > 0 && item_size % 8 == 0, "item_size must be a positive multiple of 8 (transfer.cu :336)");
  TORCH_CHECK(src_layout_dim > 0 && src_layout_dim % item_size == 0, "src_layout_dim must be a whole number of items (L x item_size)");
  TORCH_CHECK(layer_id >= 0 && (layer_id + 1) * item_size <= src_layout_dim, "layer_id is outside the page's layers");
  TORCH_CHECK(nbytes(src_k) % src_layout_dim == 0 && nbytes(src_k) == nbytes(src_v), "src must be whole pages of src_layout_dim bytes; K == V size");
  TORCH_CHECK(nbytes(dst_k) % item_size == 0 && nbytes(dst_k) == nbytes(dst_v), "dst must be whole records; K == V size");
  check_quota(block_quota, num_warps_per_block);
  transfer_kv_per_layer_pf_lf(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, layer_id, item_size, src_layout_dim, block_quota,
                              num_warps_per_block);
}

int64_t host_ptr_is_device_ptr(at::Tensor t) {
  TORCH_CHECK(t.device().is_cpu(), "a host tensor is expected");
  return device_ptr_equals_host_ptr(t.data_ptr()) ? 1 : 0;
}

int64_t can_use_host_pointer_for_registered_mem(int64_t device) {
  int v = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrCanUseHostPointerForRegisteredMem, static_cast<int>(device)) == cudaSuccess,
              "cudaDeviceGetAttribute failed");
  return v;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aot_per_layer", &aot_per_layer,
        "sgl-kernel transfer_kv_per_layer (87db743 transfer.cu :399-428, vendored): layer_first -> layer_first, K then V per item",
        pybind11::arg("src_k"), pybind11::arg("dst_k"), pybind11::arg("src_v"), pybind11::arg("dst_v"), pybind11::arg("src_indices"),
        pybind11::arg("dst_indices"), pybind11::arg("item_size"), pybind11::arg("block_quota") = 2, pybind11::arg("num_warps_per_block") = 32);
  m.def("aot_per_layer_pf_lf", &aot_per_layer_pf_lf,
        "sgl-kernel transfer_kv_per_layer_pf_lf (87db743 transfer.cu :430-461, vendored): page_first host -> layer_first device",
        pybind11::arg("src_k"), pybind11::arg("dst_k"), pybind11::arg("src_v"), pybind11::arg("dst_v"), pybind11::arg("src_indices"),
        pybind11::arg("dst_indices"), pybind11::arg("layer_id"), pybind11::arg("item_size"), pybind11::arg("src_layout_dim"),
        pybind11::arg("block_quota") = 2, pybind11::arg("num_warps_per_block") = 32);
  m.def("host_ptr_is_device_ptr", &host_ptr_is_device_ptr, "1 when cudaPointerGetAttributes maps this host buffer to itself");
  m.def("can_use_host_pointer_for_registered_mem", &can_use_host_pointer_for_registered_mem,
        "cudaDevAttrCanUseHostPointerForRegisteredMem of a device");
  m.def("upstream_commit", []() { return std::string(kUpstreamCommit); }, "SGLang commit the kernel is vendored from");
}
