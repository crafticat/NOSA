// gilfree_pack.cpp -- the GIL-free CPU packer of the sustained harness (feature/nosi-sustained-gilfree).
//
// WHY: in run b64_s960_v2 the Python CPU-packing feeder (cpupack_core.Pipe driven per layer by sustained_core.
// cpu_work_fn) made about 2000 torch calls per release from a thread that shares the GIL with NOSA's eager decode (about
// 1000 launches per step); the decode became host-bound (+55..61%). Here ONE call per release does the descriptor, the
// packing and the CUDA submission in C++ with the GIL released (py::call_guard<py::gil_scoped_release>).
//
// WHAT (bit-for-bit the Python packer's work, reordered only across chunk boundaries):
//   order    the release's groups (b, h, l, m, blk) sorted by (l, b, h, m) = the concatenation of the Python packer's
//            per-layer (b, h, m) order (cpupack_core.plan_groups walks the (B, H, M) plan); rows r = 0 .. R-1 per group
//   rows     source row (b*S + blk*R + r)*H + h of layer l % len(sources) (orig layout [B, S, H, D]); destination row
//            (b*S_dst + m*R + r)*H + h of the scratch (cpupack_core.row_base 'orig', row packer / row placer)
//   chunks   <= cap groups each, in that order (cap 256 = 8 MiB of K + V payload at R 64, D 128, bf16)
//   slot     cpupack_core's layout: int64 destination rows in [P0 - n*R*8, P0), K rows in [P0, P0 + gb), V rows after,
//            P0 = cap*R*8; ONE H2D of [lo, hi) per chunk
//   waits    cpupack_core.Pipe's three lifetimes: the host repacks staging slot s only after the H2D that last read it
//            (host wait on its event); the H2D into landing slot s waits for the scatter that last read it (copy stream
//            waits); the scatter of a chunk waits for that chunk's H2D (scatter stream waits)
//   packing  at::parallel_for over the chunk's rows (torch's own OpenMP pool of the calling thread; no second OpenMP
//            runtime); 2 memcpy of D * elem bytes per row
// CPU MODE (the tests): a CPU landing ring and CPU scratch run the same code path with synchronous copies, no events.
// The *_holding_gil entry point is the NEGATIVE CONTROL of the GIL test (same work, GIL held).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <vector>

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Release {
  const int64_t* b;
  const int64_t* h;
  const int64_t* l;
  const int64_t* m;
  const int64_t* blk;
  int64_t n;
};

Release check_release(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l, const at::Tensor& m,
                      const at::Tensor& blk) {
  for (const at::Tensor* t : {&b, &h, &l, &m, &blk}) {
    TORCH_CHECK(t->device().is_cpu() && t->scalar_type() == at::kLong && t->dim() == 1 && t->is_contiguous(),
                "release vectors must be contiguous 1-D int64 CPU tensors");
    TORCH_CHECK(t->numel() == b.numel(), "release vectors differ in length");
  }
  return Release{b.data_ptr<int64_t>(), h.data_ptr<int64_t>(), l.data_ptr<int64_t>(), m.data_ptr<int64_t>(),
                 blk.data_ptr<int64_t>(), b.numel()};
}

struct Src {
  std::vector<at::Tensor> keep;          // keeps the source tensors alive while their pointers are used
  std::vector<const uint8_t*> k, v;
  int64_t B = 0, S = 0, H = 0, D = 0, row_bytes = 0;
  at::ScalarType dtype = at::kBFloat16;
};

Src check_sources(const std::vector<at::Tensor>& sk, const std::vector<at::Tensor>& sv) {
  TORCH_CHECK(!sk.empty() && sk.size() == sv.size(), "K and V source lists must be non-empty and of equal length");
  const at::Tensor& r = sk[0];
  TORCH_CHECK(r.dim() == 4, "a source is a (B, S, H, D) tensor");
  Src s;
  s.B = r.size(0);
  s.S = r.size(1);
  s.H = r.size(2);
  s.D = r.size(3);
  s.row_bytes = s.D * static_cast<int64_t>(r.element_size());
  s.dtype = r.scalar_type();
  for (size_t i = 0; i < sk.size(); ++i) {
    for (const at::Tensor* t : {&sk[i], &sv[i]}) {
      TORCH_CHECK(t->device().is_cpu() && t->is_contiguous() && t->sizes() == r.sizes() &&
                      t->scalar_type() == r.scalar_type(),
                  "every source must be a contiguous CPU tensor of the first source's shape and dtype");
      s.keep.push_back(*t);
    }
    s.k.push_back(static_cast<const uint8_t*>(sk[i].data_ptr()));
    s.v.push_back(static_cast<const uint8_t*>(sv[i].data_ptr()));
  }
  return s;
}

// The (l, b, h, m) order of the release, after range checks and the unique-destination check (a duplicate (b, h, m)
// inside one chunk would make index_copy_ nondeterministic; the Schedule never produces one).
std::vector<int64_t> order_and_check(const Release& r, const Src& s, int64_t R, int64_t S_dst) {
  TORCH_CHECK(R > 0 && s.S % R == 0 && S_dst % R == 0, "S and S_dst must be multiples of R");
  const int64_t nblk = s.S / R, M = S_dst / R;
  for (int64_t i = 0; i < r.n; ++i) {
    TORCH_CHECK(r.b[i] >= 0 && r.b[i] < s.B && r.h[i] >= 0 && r.h[i] < s.H && r.l[i] >= 0 && r.m[i] >= 0 &&
                    r.m[i] < M && r.blk[i] >= 0 && r.blk[i] < nblk,
                "group ", i, " out of range: b ", r.b[i], " h ", r.h[i], " l ", r.l[i], " m ", r.m[i], " blk ",
                r.blk[i]);
  }
  std::vector<int64_t> o(r.n);
  std::iota(o.begin(), o.end(), 0);
  std::vector<int64_t> d(o);
  std::sort(d.begin(), d.end(), [&](int64_t x, int64_t y) {
    if (r.b[x] != r.b[y]) return r.b[x] < r.b[y];
    if (r.h[x] != r.h[y]) return r.h[x] < r.h[y];
    return r.m[x] < r.m[y];
  });
  for (int64_t i = 1; i < r.n; ++i) {
    TORCH_CHECK(!(r.b[d[i]] == r.b[d[i - 1]] && r.h[d[i]] == r.h[d[i - 1]] && r.m[d[i]] == r.m[d[i - 1]]),
                "duplicate destination (b, h, m) = (", r.b[d[i]], ", ", r.h[d[i]], ", ", r.m[d[i]], ") in one release");
  }
  std::sort(o.begin(), o.end(), [&](int64_t x, int64_t y) {
    if (r.l[x] != r.l[y]) return r.l[x] < r.l[y];
    if (r.b[x] != r.b[y]) return r.b[x] < r.b[y];
    if (r.h[x] != r.h[y]) return r.h[x] < r.h[y];
    return r.m[x] < r.m[y];
  });
  return o;
}

// Pack groups order[g0 .. g1) into idx_out (rows int64), k_out / v_out (rows of row_bytes).
void pack_range(const std::vector<int64_t>& order, int64_t g0, int64_t g1, const Release& r, const Src& s, int64_t R,
                int64_t S_dst, int64_t* idx_out, uint8_t* k_out, uint8_t* v_out) {
  const int64_t rows = (g1 - g0) * R;
  const int64_t L = static_cast<int64_t>(s.k.size());
  const int64_t rb = s.row_bytes;
  at::parallel_for(0, rows, 512, [&](int64_t a, int64_t e) {
    for (int64_t q = a; q < e; ++q) {
      const int64_t gi = order[g0 + q / R];
      const int64_t rr = q % R;
      const int64_t lay = r.l[gi] % L;
      const int64_t srow = (r.b[gi] * s.S + r.blk[gi] * R + rr) * s.H + r.h[gi];
      std::memcpy(k_out + q * rb, s.k[lay] + srow * rb, rb);
      std::memcpy(v_out + q * rb, s.v[lay] + srow * rb, rb);
      idx_out[q] = (r.b[gi] * S_dst + r.m[gi] * R + rr) * s.H + r.h[gi];
    }
  });
}

int64_t byte_size(const at::Tensor& t) { return t.numel() * static_cast<int64_t>(t.element_size()); }

// The whole release packed into flat outputs, `repeat` times (the bitwise test and the GIL test).
std::vector<int64_t> pack_into(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l, const at::Tensor& m,
                               const at::Tensor& blk, const std::vector<at::Tensor>& sk, const std::vector<at::Tensor>& sv,
                               int64_t R, int64_t S_dst, at::Tensor out_idx, at::Tensor out_k, at::Tensor out_v,
                               int64_t repeat) {
  const Release r = check_release(b, h, l, m, blk);
  const Src s = check_sources(sk, sv);
  const std::vector<int64_t> order = order_and_check(r, s, R, S_dst);
  const int64_t rows = r.n * R;
  TORCH_CHECK(out_idx.device().is_cpu() && out_idx.is_contiguous() && out_idx.scalar_type() == at::kLong &&
                  out_idx.numel() == rows,
              "out_idx must be a contiguous int64 CPU tensor of n * R elements");
  for (const at::Tensor* t : {&out_k, &out_v}) {
    TORCH_CHECK(t->device().is_cpu() && t->is_contiguous() && byte_size(*t) == rows * s.row_bytes,
                "out_k / out_v must be contiguous CPU tensors of n * R rows of D elements");
  }
  const int64_t t0 = now_ns();
  for (int64_t i = 0; i < std::max<int64_t>(1, repeat); ++i) {
    pack_range(order, 0, r.n, r, s, R, S_dst, out_idx.data_ptr<int64_t>(), static_cast<uint8_t*>(out_k.data_ptr()),
               static_cast<uint8_t*>(out_v.data_ptr()));
  }
  return {r.n, rows, now_ns() - t0};
}

class NativePipe {
 public:
  NativePipe(at::Tensor stage, at::Tensor land, int64_t cap_groups, int64_t R, std::vector<int64_t> copy_stream,
             std::vector<int64_t> scatter_stream)
      : stage_(std::move(stage)), land_(std::move(land)), cap_(cap_groups), R_(R) {
    TORCH_CHECK(cap_ >= 1 && R_ >= 1, "cap_groups and R must be positive");
    TORCH_CHECK(stage_.dim() == 2 && land_.sizes() == stage_.sizes() && stage_.scalar_type() == at::kByte &&
                    land_.scalar_type() == at::kByte && stage_.is_contiguous() && land_.is_contiguous(),
                "stage and land must be contiguous uint8 (ring, slot_bytes) tensors of one shape");
    TORCH_CHECK(stage_.device().is_cpu(), "the staging ring is host memory");
    ring_ = stage_.size(0);
    slot_bytes_ = stage_.size(1);
    p0_ = cap_ * R_ * 8;
    TORCH_CHECK(ring_ >= 2 && slot_bytes_ % 8 == 0 && slot_bytes_ > p0_, "ring >= 2, slot_bytes a multiple of 8 > P0");
    cuda_ = land_.is_cuda();
    if (cuda_) {
      TORCH_CHECK(stage_.is_pinned(), "a CUDA landing ring needs a pinned staging ring");
      TORCH_CHECK(copy_stream.size() == 3 && scatter_stream.size() == 3,
                  "CUDA mode needs (stream_id, device_index, device_type) of the copy and the scatter stream");
      copy_.emplace(c10::Stream::unpack3(copy_stream[0], static_cast<c10::DeviceIndex>(copy_stream[1]),
                                         static_cast<c10::DeviceType>(copy_stream[2])));
      scatter_.emplace(c10::Stream::unpack3(scatter_stream[0], static_cast<c10::DeviceIndex>(scatter_stream[1]),
                                            static_cast<c10::DeviceType>(scatter_stream[2])));
    } else {
      TORCH_CHECK(land_.device().is_cpu(), "the landing ring is a CUDA tensor or (tests) a CPU tensor");
    }
    reset();
  }

  void set_sources(const std::vector<at::Tensor>& sk, const std::vector<at::Tensor>& sv) {
    src_ = check_sources(sk, sv);
    TORCH_CHECK(slot_bytes_ >= p0_ + 2 * cap_ * R_ * src_.row_bytes, "slot too small for cap_groups of these rows");
    have_src_ = true;
  }

  // After a full synchronize (cpupack_core.Pipe.reset): forget the ring's events, zero both index regions.
  void reset() {
    k_ = 0;
    if (cuda_) {
      ev_h2d_ = std::vector<at::cuda::CUDAEvent>(ring_);
      ev_sc_ = std::vector<at::cuda::CUDAEvent>(ring_);
    }
    stage_.narrow(1, 0, p0_).zero_();
    land_.narrow(1, 0, p0_).zero_();
  }

  std::vector<int64_t> release(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l, const at::Tensor& m,
                               const at::Tensor& blk, at::Tensor scr_k2d, at::Tensor scr_v2d) {
    TORCH_CHECK(have_src_, "set_sources first");
    const Release r = check_release(b, h, l, m, blk);
    TORCH_CHECK(scr_k2d.dim() == 2 && scr_k2d.size(1) == src_.D && scr_v2d.sizes() == scr_k2d.sizes() &&
                    scr_k2d.scalar_type() == src_.dtype && scr_v2d.scalar_type() == src_.dtype,
                "the scratch views must be (rows, D) of the sources' dtype");
    TORCH_CHECK(scr_k2d.device() == land_.device() && scr_v2d.device() == land_.device(),
                "the scratch lives where the landing ring lives");
    const int64_t S_dst = scr_k2d.size(0) / (src_.B * src_.H);
    TORCH_CHECK(S_dst * src_.B * src_.H == scr_k2d.size(0), "scratch rows must be B * S_dst * H");
    const std::vector<int64_t> order = order_and_check(r, src_, R_, S_dst);
    int64_t layers = 0;
    for (int64_t i = 0; i < r.n; ++i) {
      if (i == 0 || r.l[order[i]] != r.l[order[i - 1]]) ++layers;
    }
    const int64_t t_all = now_ns();
    int64_t pack = 0, wait = 0, sub = 0, chunks = 0;
    for (int64_t g0 = 0; g0 < r.n; g0 += cap_) {
      const int64_t g1 = std::min(r.n, g0 + cap_);
      const int64_t rows = (g1 - g0) * R_;
      const int64_t ib = rows * 8, gb = rows * src_.row_bytes, lo = p0_ - ib, len = ib + 2 * gb;
      const int64_t s = k_ % ring_;
      int64_t t = now_ns();
      if (cuda_) ev_h2d_[s].synchronize();          // the H2D that last read staging slot s has completed
      wait += now_ns() - t;
      t = now_ns();
      uint8_t* slot = stage_.data_ptr<uint8_t>() + s * slot_bytes_;
      pack_range(order, g0, g1, r, src_, R_, S_dst, reinterpret_cast<int64_t*>(slot + lo), slot + p0_,
                 slot + p0_ + gb);
      pack += now_ns() - t;
      t = now_ns();
      at::Tensor st = stage_.select(0, s).narrow(0, lo, len);
      at::Tensor lslot = land_.select(0, s);
      at::Tensor ld = lslot.narrow(0, lo, len);
      at::Tensor idx = lslot.narrow(0, lo, ib).view(at::kLong);
      at::Tensor kd = lslot.narrow(0, p0_, gb).view(src_.dtype).view({rows, src_.D});
      at::Tensor vd = lslot.narrow(0, p0_ + gb, gb).view(src_.dtype).view({rows, src_.D});
      if (cuda_) {
        {
          c10::cuda::CUDAStreamGuard g(*copy_);
          ev_sc_[s].block(*copy_);                   // the scatter that last read landing slot s is done
          ld.copy_(st, /*non_blocking=*/true);
          ev_h2d_[s].record(*copy_);
        }
        {
          c10::cuda::CUDAStreamGuard g(*scatter_);
          ev_h2d_[s].block(*scatter_);               // this chunk's H2D is done
          scr_k2d.index_copy_(0, idx, kd);
          scr_v2d.index_copy_(0, idx, vd);
          ev_sc_[s].record(*scatter_);
        }
      } else {
        ld.copy_(st);
        scr_k2d.index_copy_(0, idx, kd);
        scr_v2d.index_copy_(0, idx, vd);
      }
      sub += now_ns() - t;
      ++k_;
      ++chunks;
    }
    return {r.n, chunks, layers, pack, wait, sub, now_ns() - t_all};
  }

  int64_t chunks_issued() const { return k_; }
  bool is_cuda() const { return cuda_; }
  int64_t p0() const { return p0_; }

 private:
  at::Tensor stage_, land_;
  int64_t cap_, R_, ring_ = 0, slot_bytes_ = 0, p0_ = 0, k_ = 0;
  bool cuda_ = false, have_src_ = false;
  Src src_;
  std::optional<c10::cuda::CUDAStream> copy_, scatter_;
  std::vector<at::cuda::CUDAEvent> ev_h2d_, ev_sc_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.doc() = "GIL-free CPU packer of the sustained harness (one call per release, GIL released)";
  mod.def("pack_into", &pack_into, py::call_guard<py::gil_scoped_release>(),
          "pack a whole release into flat outputs (GIL released): returns [groups, rows, ns]");
  mod.def("pack_into_holding_gil", &pack_into,
          "NEGATIVE CONTROL of the GIL test: pack_into with the GIL held for the whole call");
  py::class_<NativePipe>(mod, "NativePipe")
      .def(py::init<at::Tensor, at::Tensor, int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>>(),
           py::arg("stage"), py::arg("land"), py::arg("cap_groups"), py::arg("R"),
           py::arg("copy_stream") = std::vector<int64_t>{}, py::arg("scatter_stream") = std::vector<int64_t>{})
      .def("set_sources", &NativePipe::set_sources)
      .def("reset", &NativePipe::reset, py::call_guard<py::gil_scoped_release>())
      .def("release", &NativePipe::release, py::call_guard<py::gil_scoped_release>(),
           "one release (GIL released): [groups, chunks, layers, pack_ns, wait_ns, submit_ns, total_ns]")
      .def_property_readonly("chunks_issued", &NativePipe::chunks_issued)
      .def_property_readonly("cuda", &NativePipe::is_cuda)
      .def_property_readonly("p0", &NativePipe::p0);
  mod.attr("VERSION") = 1;
}
