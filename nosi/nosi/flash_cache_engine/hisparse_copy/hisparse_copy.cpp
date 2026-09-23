// pybind wrapper for the vendored SGLang HiSparse copy-only kernel (hisparse_copy.cu,
// copy_cache_planned_kernel of sglang 87db743, hisparse.cuh :837-892).
//
// Every assumption the kernel makes about its arguments is checked HERE, loudly, because the
// kernel indexes raw pointers: src_row[m] / dst_row[m] with unit stride inside a plan row and
// plan_stride between rows (:874-875), host / device records at loc * item_size_bytes (:214-221),
// and 16-byte vector loads / stores (:150, :152). A silent mismatch would copy the wrong bytes.
//
// Tensor contract (the one upstream's wrapper copy_cache_planned_mla builds,
// python/sglang/kernels/ops/kvcache/hisparse.py, and hisparse_coordinator.py:296-307 allocates):
//   miss_src_locs  int64 CUDA [R, P], unit inner stride     host record index of miss m of request r
//   miss_dst_locs  int32 CUDA [R, P], same row stride       device record index of the same miss
//   miss_counts    int32 CUDA [>= R], contiguous            misses of request r (entries >= count unused)
//   num_real_reqs  int32 CUDA [1]                           requests the kernel walks (must be <= R;
//                                                           a device value, checked by plan.validate_plan,
//                                                           not here: reading it would sync every launch)
//   host_k, host_v pinned CPU, contiguous, same dtype and byte size (read through UVA)
//   dev_k, dev_v   CUDA on the plan's device, contiguous, same dtype and byte size as each other
// is_mla = false: K and V are separate caches and both are copied at the same locs (:213-222).
// is_mla = true : SGLang's shipped instantiation (copy_cache_planned_mla): K only. host_v / dev_v are IGNORED
//                 (not checked, not dereferenced: the launcher passes nullptr as upstream :919/:921 do); the
//                 python wrapper passes empty placeholders.
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

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
    bool is_mla);

namespace {

// Keep in sync with HISPARSE_COPY_BLOCK_SIZES in hisparse_copy.cu (tests grep both).
const std::vector<int64_t> kBlockSizes = {256, 1024};
const char* kUpstreamCommit = "87db74302151956e354a05c01ecda47fd24cb882";

int64_t nbytes(const at::Tensor& t) { return t.numel() * t.element_size(); }

bool aligned16(const at::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 0xF) == 0; }

void check_plan_2d(const at::Tensor& t, at::ScalarType st, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor (got ", t.device(), ")");
  TORCH_CHECK(t.scalar_type() == st, name, " must be ", st, " (got ", t.scalar_type(), ")");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D [requests, plan_stride] (got ", t.dim(), "-D)");
  TORCH_CHECK(t.size(1) <= 1 || t.stride(1) == 1, name, " must have unit inner stride: the kernel reads row[m] (:874-875)");
}

void check_cache(const at::Tensor& t, bool host, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  if (host) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor (got ", t.device(), ")");
    TORCH_CHECK(t.is_pinned(), name, " must be PINNED host memory: the kernel dereferences it through UVA (ld.global.nc)");
  } else {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor (got ", t.device(), ")");
  }
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous: records are addressed as base + loc * item_size_bytes (:214-221)");
  TORCH_CHECK(aligned16(t), name, " must be 16-byte aligned (ld.global.nc.v2.b64 / st.global.cg.v2.b64, :150-152)");
}

}  // namespace

void copy_planned(
    at::Tensor miss_src_locs,
    at::Tensor miss_dst_locs,
    at::Tensor miss_counts,
    at::Tensor num_real_reqs,
    at::Tensor host_k,
    at::Tensor host_v,
    at::Tensor dev_k,
    at::Tensor dev_v,
    int64_t num_blocks,
    int64_t block_size,
    int64_t item_size_bytes,
    bool skip_io,
    bool is_mla) {
  check_plan_2d(miss_src_locs, at::kLong, "miss_src_locs");
  check_plan_2d(miss_dst_locs, at::kInt, "miss_dst_locs");
  const int64_t R = miss_src_locs.size(0);
  const int64_t P = miss_src_locs.size(1);
  TORCH_CHECK(miss_dst_locs.size(0) == R && miss_dst_locs.size(1) == P,
              "miss_dst_locs must have the shape of miss_src_locs ", miss_src_locs.sizes(), " (got ", miss_dst_locs.sizes(), ")");
  TORCH_CHECK(miss_dst_locs.stride(0) == miss_src_locs.stride(0), "miss_src/miss_dst row strides differ (upstream :908-910)");
  TORCH_CHECK(R >= 1 && P >= 1, "empty plan [", R, ", ", P, "]");
  TORCH_CHECK(R * miss_src_locs.stride(0) < (int64_t(1) << 31),
              "plan too large for the kernel's int flat index (:863, :871)");

  TORCH_CHECK(miss_counts.defined() && miss_counts.is_cuda() && miss_counts.scalar_type() == at::kInt
                  && miss_counts.dim() == 1 && miss_counts.is_contiguous() && miss_counts.numel() >= R,
              "miss_counts must be a contiguous 1-D int32 CUDA tensor with >= ", R, " entries (got ",
              miss_counts.scalar_type(), " ", miss_counts.sizes(), " on ", miss_counts.device(), ")");
  TORCH_CHECK(num_real_reqs.defined() && num_real_reqs.is_cuda() && num_real_reqs.scalar_type() == at::kInt
                  && num_real_reqs.numel() == 1,
              "num_real_reqs must be an int32 CUDA tensor with one element (got ", num_real_reqs.scalar_type(), " ",
              num_real_reqs.sizes(), " on ", num_real_reqs.device(), ")");
  const auto dev = miss_src_locs.device();
  TORCH_CHECK(miss_dst_locs.device() == dev && miss_counts.device() == dev && num_real_reqs.device() == dev,
              "the four plan tensors must be on one device");

  check_cache(host_k, true, "host_k");
  check_cache(dev_k, false, "dev_k");
  TORCH_CHECK(dev_k.device() == dev, "dev_k must be on the plan's device ", dev);
  TORCH_CHECK(dev_k.scalar_type() == host_k.scalar_type(), "host_k and dev_k must share one dtype");
  if (!is_mla) {
    check_cache(host_v, true, "host_v");
    check_cache(dev_v, false, "dev_v");
    TORCH_CHECK(dev_v.device() == dev, "dev_v must be on the plan's device ", dev);
    TORCH_CHECK(host_k.scalar_type() == host_v.scalar_type() && dev_v.scalar_type() == host_k.scalar_type(),
                "host_k, host_v, dev_k, dev_v must share one dtype");
    TORCH_CHECK(nbytes(host_k) == nbytes(host_v), "host_k and host_v must have the same byte size (K and V share locs, :214-221)");
    TORCH_CHECK(nbytes(dev_k) == nbytes(dev_v), "dev_k and dev_v must have the same byte size (K and V share locs, :214-221)");
  }

  TORCH_CHECK(item_size_bytes > 0 && item_size_bytes % 16 == 0,
              "item_size_bytes must be a positive multiple of 16 (got ", item_size_bytes, "): transfer_item_warp moves "
              "16-byte pairs (:143-153) and only whole 8-byte words in its tail (:157-165), and a record at loc * item_size "
              "must stay 16-byte aligned for ld.global.nc.v2.b64");
  TORCH_CHECK(nbytes(host_k) % item_size_bytes == 0, "host_k (", nbytes(host_k), " B) is not a whole number of ",
              item_size_bytes, "-byte records");
  TORCH_CHECK(nbytes(dev_k) % item_size_bytes == 0, "dev_k (", nbytes(dev_k), " B) is not a whole number of ",
              item_size_bytes, "-byte records");
  TORCH_CHECK(nbytes(dev_k) / item_size_bytes < (int64_t(1) << 31),
              "device records exceed the int32 range of miss_dst_locs");

  TORCH_CHECK(num_blocks >= 1 && num_blocks <= 65535, "num_blocks must be in [1, 65535] (got ", num_blocks, ")");
  TORCH_CHECK(std::find(kBlockSizes.begin(), kBlockSizes.end(), block_size) != kBlockSizes.end(),
              "block_size ", block_size, " is not instantiated; supported: 256, 1024");

  hisparse_copy_planned_cuda(miss_src_locs, miss_dst_locs, miss_counts, num_real_reqs, host_k, host_v, dev_k, dev_v,
                             num_blocks, block_size, item_size_bytes, skip_io, is_mla);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_planned", &copy_planned,
        "SGLang HiSparse copy_cache_planned_kernel (87db743, vendored): one launch copies every planned item of K and V "
        "(is_mla = true: K only, SGLang's shipped copy_cache_planned_mla)",
        pybind11::arg("miss_src_locs"), pybind11::arg("miss_dst_locs"), pybind11::arg("miss_counts"),
        pybind11::arg("num_real_reqs"), pybind11::arg("host_k"), pybind11::arg("host_v"), pybind11::arg("dev_k"),
        pybind11::arg("dev_v"), pybind11::arg("num_blocks"), pybind11::arg("block_size"),
        pybind11::arg("item_size_bytes"), pybind11::arg("skip_io"), pybind11::arg("is_mla") = false);
  m.def("block_sizes", []() { return kBlockSizes; }, "BLOCK_SIZE values the kernel is instantiated for");
  m.def("upstream_commit", []() { return std::string(kUpstreamCommit); }, "SGLang commit the kernel is vendored from");
}
