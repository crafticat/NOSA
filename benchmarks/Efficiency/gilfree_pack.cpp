// gilfree_pack.cpp v3 -- agent E's GIL-free CPU packer (v2: feature/nosi-sustained-gilfree2 f2b981a) with an OPTIONAL
// C++ WORKER THREAD (feature/nosi-r2-gilfree-worker; Q-C2 of 2026-10-21: the captured tick needs the host not to wait
// before each replay).
//
// WHY v2: in run b64_s960_v2 the Python CPU-packing feeder (cpupack_core.Pipe driven per layer by sustained_core.
// cpu_work_fn) made about 2000 torch calls per release from a thread that shares the GIL with NOSA's eager decode (about
// 1000 launches per step); the decode became host-bound (+55..61%). v2 does ONE call per release (the descriptor, the
// packing and the CUDA submission) in C++ with the GIL released (py::call_guard<py::gil_scoped_release>).
// WHY v3: the release is still DRIVEN from Python (r2_transport_gilfree.py's feeder thread: queue.get, the T3 / T4
// waits, the index vectors, the board publication, the completion event: all with the GIL). S-C's exact device wait
// (r2_residency_sc.py CudaDeviceOps._submitted) therefore host-waits each pack's SUBMISSION before the main stream may
// spin on the board: a feeder that needs the GIL could not finish while the main thread holds it, blocked in a launch
// behind the spinning stream. v3 adds a worker thread INSIDE NativePipe that runs whole releases with no Python and no
// GIL, so a main stream that spins on the board cannot block it:
//   start_worker(inference, grad, device)  the worker enters c10::InferenceMode(inference), the grad mode and (CUDA) the
//            device ONCE, at its start (torch's modes are thread-local; the r2 tick builds its tensors in inference mode)
//   submit(...) -> seq   one release as a JOB (the index lists, the layer's base store tensors, the board pairs, the
//            T4 / T3 raw cudaEvent_t handles; no tensor made in Python dies in the worker, see Job); FIFO; the job's T4 events are host-waited by cudaEventSynchronize IN THE WORKER, its T3
//            events become cudaStreamWaitEvent on the scatter stream, then the v2 release runs; the job's completion
//            event is recorded on the scatter stream after it; the SUBMITTED counter then advances
//   board    set_board(GenBoard.t): each chunk carries the (board index, generation) pairs of its groups in its
//            staging slot (in front of the destination rows: one H2D as before) and, on the scatter stream after the
//            chunk's two index_copy_, ONE index_put_ writes them into the board (S-C's exact wait reads it)
//   job_done / wait_job / stream_wait_job / take_error / worker_stats / stop_worker: the adapter's done / force / the
//            torch Stream.wait_event protocol / its error and stats.
// The v2 entry points (pack_into, release) are unchanged in meaning; release() publishes nothing (the Python feeder
// publishes). v3 slot layout: P0 = cap * (R + 2) * 8 (SLOT_EXTRA_PER_GROUP = 16 bytes of board pairs per group).
//
// WHAT (bit-for-bit the Python packer's work, reordered only across chunk boundaries):
//   order    the release's groups (b, h, l, m, blk) sorted by (l, b, h, m) = the concatenation of the Python packer's
//            per-layer (b, h, m) order (cpupack_core.plan_groups walks the (B, H, M) plan); rows r = 0 .. R-1 per group
//   rows     source row (b*S + blk*R + r)*H + h of layer l % len(sources) (orig layout [B, S, H, D]); destination row
//            (b*S_dst + m*R + r)*H + h of the scratch (cpupack_core.row_base 'orig', row packer / row placer)
//   chunks   <= cap groups each, in that order (cap 256 = 8 MiB of K + V payload at R 64, D 128, bf16)
//   slot     [P0 - pb - n*R*8, P0 - n*R*8) the chunk's board pairs (pb = 16 * groups when it publishes, else 0: its
//            board indices, then its generations), int64 destination rows in [P0 - n*R*8, P0), K rows in [P0, P0 + gb),
//            V rows after; ONE H2D of [lo, hi) per chunk
//   waits    cpupack_core.Pipe's three lifetimes: the host repacks staging slot s only after the H2D that last read it
//            (host wait on its event); the H2D into landing slot s waits for the scatter (and the publication) that last
//            read it (copy stream waits); the scatter of a chunk waits for that chunk's H2D (scatter stream waits)
//   packing  an explicit OpenMP loop over the chunk's rows, `#pragma omp parallel num_threads(nt)` + `omp for
//            schedule(static)`, nt = the pipe's set_threads / pack_into's threads; 2 memcpy of D * elem bytes per row.
//            Built with -fopenmp (gilfree_pack.load): the system libgomp, beside torch's bundled one (two pools); the team
//            is created by the calling thread (the feeder, or the worker) and inherits its CPU affinity. The team size
//            used is returned with every call (threads_min / threads_max); the worker REFUSES a job whose team differs
//            from set_threads (v1's defect: a serial loop, unseen until its stats were read). Built WITHOUT -fopenmp
//            (the tests' negative control) the loop is serial and reports 1.
// CPU MODE (the tests): a CPU landing ring and CPU scratch run the same code path with synchronous copies, no events.
// The *_holding_gil entry point is the NEGATIVE CONTROL of the GIL test (same work, GIL held). hold_gil_until is a TEST
// call: it holds the GIL while it waits for a release to complete (the worker must progress; a Python feeder cannot).
#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kMaxThreads = 256;
constexpr int64_t kPairBytes = 16;            // one group's board pair: its board index and its generation (int64)

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

// Pack groups order[g0 .. g1) into idx_out (rows int64), k_out / v_out (rows of row_bytes) on an OpenMP team of nt
// threads; returns the team size actually used (1 when built without OpenMP).
int pack_range(const std::vector<int64_t>& order, int64_t g0, int64_t g1, const Release& r, const Src& s, int64_t R,
               int64_t S_dst, int64_t* idx_out, uint8_t* k_out, uint8_t* v_out, int nt) {
  const int64_t rows = (g1 - g0) * R;
  const int64_t L = static_cast<int64_t>(s.k.size());
  const int64_t rb = s.row_bytes;
  auto row = [&](int64_t q) {
    const int64_t gi = order[g0 + q / R];
    const int64_t rr = q % R;
    const int64_t lay = r.l[gi] % L;
    const int64_t srow = (r.b[gi] * s.S + r.blk[gi] * R + rr) * s.H + r.h[gi];
    std::memcpy(k_out + q * rb, s.k[lay] + srow * rb, rb);
    std::memcpy(v_out + q * rb, s.v[lay] + srow * rb, rb);
    idx_out[q] = (r.b[gi] * S_dst + r.m[gi] * R + rr) * s.H + r.h[gi];
  };
  int team = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
  {
#pragma omp single
    team = omp_get_num_threads();
#pragma omp for schedule(static)
    for (int64_t q = 0; q < rows; ++q) row(q);
  }
#else
  (void)nt;
  for (int64_t q = 0; q < rows; ++q) row(q);
#endif
  return team;
}

int check_threads(int64_t nt) {
  TORCH_CHECK(nt >= 1 && nt <= kMaxThreads, "threads must be 1 .. ", kMaxThreads, " (got ", nt, ")");
  return static_cast<int>(nt);
}

int64_t byte_size(const at::Tensor& t) { return t.numel() * static_cast<int64_t>(t.element_size()); }

// The whole release packed into flat outputs, `repeat` times, on `threads` threads (the bitwise test, the thread-count
// test and the GIL test): returns [groups, rows, ns, team size used].
std::vector<int64_t> pack_into(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l, const at::Tensor& m,
                               const at::Tensor& blk, const std::vector<at::Tensor>& sk, const std::vector<at::Tensor>& sv,
                               int64_t R, int64_t S_dst, at::Tensor out_idx, at::Tensor out_k, at::Tensor out_v,
                               int64_t repeat, int64_t threads) {
  const int nt = check_threads(threads);
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
  int team = 0;
  for (int64_t i = 0; i < std::max<int64_t>(1, repeat); ++i) {
    team = pack_range(order, 0, r.n, r, s, R, S_dst, out_idx.data_ptr<int64_t>(), static_cast<uint8_t*>(out_k.data_ptr()),
                      static_cast<uint8_t*>(out_v.data_ptr()), nt);
  }
  return {r.n, rows, now_ns() - t0, team};
}

// One job of the worker: a whole release, its board pairs, its order events, its result. NO TENSOR MADE IN PYTHON may
// die in the worker: a tensor whose Python object died while C++ held it owns that object, and its destruction takes
// the GIL. So the index vectors are plain vectors (the worker wraps them with from_blob), and the store is the layer's
// BASE tensor, which Python keeps alive (the worker makes its (rows, D) views itself).
struct Job {
  int64_t seq = 0;
  std::vector<int64_t> b, h, l, m, blk, bflat, bgen;       // the release's groups and board pairs (dropped once it ran)
  at::Tensor store_k, store_v;                              // the layer's store (B, rows, H, D): alive in Python
  std::vector<int64_t> hist, readers;                       // T4 / T3: raw cudaEvent_t handles (CUDA mode)
  std::vector<int64_t> stats;                               // release_impl's stats
  std::string error;                                        // "" = the job ran
  at::cuda::CUDAEvent done_ev;                              // CUDA: recorded on the scatter stream after the release
};

constexpr int kStats = 10;                    // groups, chunks, layers, pack, wait, submit, total, team lo, team hi, published

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
    p0_ = cap_ * R_ * 8 + cap_ * kPairBytes;
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
    totals_.assign(kStats, 0);
    totals_[7] = totals_[8] = -1;
    reset();
  }

  ~NativePipe() { stop_worker(); }

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

  // v2's entry point (the Python feeder): one release in the caller's thread, no board publication.
  std::vector<int64_t> release(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l, const at::Tensor& m,
                               const at::Tensor& blk, at::Tensor scr_k2d, at::Tensor scr_v2d) {
    std::vector<int64_t> st = release_impl(b, h, l, m, blk, scr_k2d, scr_v2d, nullptr, nullptr);
    st.resize(9);
    return st;
  }

  // ------------------------------------------------------------------------------------------------ the worker
  // The generation board (GenBoard.t, alive in Python); the worker publishes into its flat view, made in the worker.
  void set_board(at::Tensor board) {
    TORCH_CHECK(board.scalar_type() == at::kLong && board.is_contiguous() && board.device() == land_.device(),
                "the board is a contiguous int64 tensor (GenBoard.t) where the landing ring lives");
    board_ = std::move(board);
  }

  void set_skip_publish(bool skip) { skip_publish_ = skip; }   // NEGATIVE CONTROL (tests only): no board publication

  // cores: the worker's CPU affinity (empty: inherited); its OpenMP team inherits it, as the Python feeder's does.
  void start_worker(bool inference, bool grad, int64_t device, std::vector<int64_t> cores) {
    std::lock_guard<std::mutex> lk(mu_);
    TORCH_CHECK(!running_, "the worker is already running");
    stop_ = false;
    running_ = true;
    worker_ = std::thread([this, inference, grad, device, cores] { worker_loop(inference, grad, device, cores); });
  }

  // Drains the queued jobs, then joins the worker.
  void stop_worker() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!running_) return;
      stop_ = true;
    }
    cv_work_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
    cv_done_.notify_all();
  }

  int64_t submit(std::vector<int64_t> b, std::vector<int64_t> h, std::vector<int64_t> l, std::vector<int64_t> m,
                 std::vector<int64_t> blk, at::Tensor store_k, at::Tensor store_v, std::vector<int64_t> bflat,
                 std::vector<int64_t> bgen, std::vector<int64_t> hist, std::vector<int64_t> readers) {
    TORCH_CHECK(have_src_, "set_sources first");
    const size_t n = b.size();
    TORCH_CHECK(n > 0 && h.size() == n && l.size() == n && m.size() == n && blk.size() == n,
                "submit: the release's five vectors must be non-empty and of one length");
    TORCH_CHECK((bflat.empty() && bgen.empty()) || (bflat.size() == n && bgen.size() == n),
                "submit: the board pairs: one per group, or none");
    TORCH_CHECK(store_k.dim() == 4 && store_k.size(3) == src_.D && store_v.sizes() == store_k.sizes() &&
                    store_k.is_contiguous() && store_v.is_contiguous(),
                "submit: the store is the layer's contiguous (B, rows, H, D) K and V tensors");
    auto j = std::make_shared<Job>();
    j->b = std::move(b);
    j->h = std::move(h);
    j->l = std::move(l);
    j->m = std::move(m);
    j->blk = std::move(blk);
    j->store_k = std::move(store_k);
    j->store_v = std::move(store_v);
    j->bflat = std::move(bflat);
    j->bgen = std::move(bgen);
    j->hist = std::move(hist);
    j->readers = std::move(readers);
    {
      std::lock_guard<std::mutex> lk(mu_);
      TORCH_CHECK(running_ && !stop_, "start_worker first (and not after stop_worker)");
      j->seq = ++next_seq_;
      jobs_.push_back(j);
      pending_.push_back(j);
    }
    cv_work_.notify_one();
    return j->seq;
  }

  int64_t submitted() const { return submitted_.load(); }
  int64_t completed() const { return completed_.load(); }

  // Non-blocking: job seq ran without error and (CUDA) its completion event is done.
  bool job_done(int64_t seq) {
    std::shared_ptr<Job> j;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (seq < 1 || seq > next_seq_ || submitted_.load() < seq) return false;
      j = jobs_[seq - 1];
    }
    if (!j->error.empty()) return false;
    return !cuda_ || j->done_ev.query();
  }

  // Blocks (bound with the GIL released) until job seq was submitted, then (CUDA) until it landed.
  void wait_job(int64_t seq) {
    std::shared_ptr<Job> j = wait_submitted(seq);
    if (cuda_ && j->error.empty()) j->done_ev.synchronize();
  }

  // torch's Stream.wait_event protocol: waits (GIL released) until job seq was submitted, then `stream` waits on the
  // job's completion event ON THE DEVICE (cudaStreamWaitEvent).
  void stream_wait_job(int64_t seq, std::vector<int64_t> stream) {
    std::shared_ptr<Job> j = wait_submitted(seq);
    if (cuda_ && j->error.empty()) {
      TORCH_CHECK(stream.size() == 3, "stream_wait_job needs (stream_id, device_index, device_type)");
      j->done_ev.block(c10::cuda::CUDAStream(c10::Stream::unpack3(
          stream[0], static_cast<c10::DeviceIndex>(stream[1]), static_cast<c10::DeviceType>(stream[2]))));
    }
  }

  std::string take_error() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string e;
    e.swap(error_);
    return e;
  }

  // Over the jobs that ran WITHOUT error: [groups, chunks, layers, pack_ns, wait_ns, submit_ns, total_ns, threads_min
  // (-1 none), threads_max (-1 none), published, jobs]; then the jobs run (errors included), the worker's inference
  // mode and its grad mode (-1 before it started).
  std::vector<int64_t> worker_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int64_t> out(totals_);
    out.push_back(ok_jobs_);
    out.push_back(jobs_run_);
    out.push_back(worker_inference_);
    out.push_back(worker_grad_);
    return out;
  }

  // The worker thread's OWN modes (c10::InferenceMode::is_enabled, c10::GradMode::is_enabled, read in that thread):
  // at its start, then before each job it runs (the modes the job ran with). [inference, grad]; -1 before the start.
  std::vector<int64_t> modes_seen() {
    std::lock_guard<std::mutex> lk(mu_);
    return {worker_inference_, worker_grad_};
  }

  void set_threads(int64_t n) { nt_ = check_threads(n); }
  int64_t threads() const { return nt_; }
  int64_t chunks_issued() const { return k_; }
  bool is_cuda() const { return cuda_; }
  int64_t p0() const { return p0_; }

 private:
  std::vector<int64_t> release_impl(const at::Tensor& b, const at::Tensor& h, const at::Tensor& l,
                                    const at::Tensor& m, const at::Tensor& blk, at::Tensor scr_k2d, at::Tensor scr_v2d,
                                    const at::Tensor* bflat, const at::Tensor* bgen) {
    TORCH_CHECK(have_src_, "set_sources first");
    const Release r = check_release(b, h, l, m, blk);
    TORCH_CHECK(scr_k2d.dim() == 2 && scr_k2d.size(1) == src_.D && scr_v2d.sizes() == scr_k2d.sizes() &&
                    scr_k2d.scalar_type() == src_.dtype && scr_v2d.scalar_type() == src_.dtype,
                "the scratch views must be (rows, D) of the sources' dtype");
    TORCH_CHECK(scr_k2d.device() == land_.device() && scr_v2d.device() == land_.device(),
                "the scratch lives where the landing ring lives");
    const bool pub = bflat != nullptr && bflat->numel() > 0 && board_.defined() && !skip_publish_;
    const int64_t* bf = nullptr;
    const int64_t* bg = nullptr;
    if (pub) {
      for (const at::Tensor* t : {bflat, bgen}) {
        TORCH_CHECK(t->device().is_cpu() && t->scalar_type() == at::kLong && t->dim() == 1 && t->is_contiguous() &&
                        t->numel() == r.n,
                    "the board pairs must be contiguous 1-D int64 CPU tensors, one per group");
      }
      bf = bflat->data_ptr<int64_t>();
      bg = bgen->data_ptr<int64_t>();
    }
    const int64_t S_dst = scr_k2d.size(0) / (src_.B * src_.H);
    TORCH_CHECK(S_dst * src_.B * src_.H == scr_k2d.size(0), "scratch rows must be B * S_dst * H");
    const std::vector<int64_t> order = order_and_check(r, src_, R_, S_dst);
    int64_t layers = 0;
    for (int64_t i = 0; i < r.n; ++i) {
      if (i == 0 || r.l[order[i]] != r.l[order[i - 1]]) ++layers;
    }
    const int64_t t_all = now_ns();
    int64_t pack = 0, wait = 0, sub = 0, chunks = 0;
    int team_lo = 0, team_hi = 0;
    for (int64_t g0 = 0; g0 < r.n; g0 += cap_) {
      const int64_t g1 = std::min(r.n, g0 + cap_);
      const int64_t ng = g1 - g0;
      const int64_t rows = ng * R_;
      const int64_t pb = pub ? ng * kPairBytes : 0;
      const int64_t ib = rows * 8, gb = rows * src_.row_bytes, lo = p0_ - ib - pb, len = pb + ib + 2 * gb;
      const int64_t s = k_ % ring_;
      int64_t t = now_ns();
      if (cuda_) ev_h2d_[s].synchronize();          // the H2D that last read staging slot s has completed
      wait += now_ns() - t;
      t = now_ns();
      uint8_t* slot = stage_.data_ptr<uint8_t>() + s * slot_bytes_;
      const int team = pack_range(order, g0, g1, r, src_, R_, S_dst, reinterpret_cast<int64_t*>(slot + p0_ - ib),
                                  slot + p0_, slot + p0_ + gb, nt_);
      if (pub) {                                     // the chunk's board pairs: indices, then generations
        int64_t* pp = reinterpret_cast<int64_t*>(slot + lo);
        for (int64_t q = 0; q < ng; ++q) {
          pp[q] = bf[order[g0 + q]];
          pp[ng + q] = bg[order[g0 + q]];
        }
      }
      team_lo = (chunks == 0) ? team : std::min(team_lo, team);
      team_hi = std::max(team_hi, team);
      pack += now_ns() - t;
      t = now_ns();
      at::Tensor st = stage_.select(0, s).narrow(0, lo, len);
      at::Tensor lslot = land_.select(0, s);
      at::Tensor ld = lslot.narrow(0, lo, len);
      at::Tensor idx = lslot.narrow(0, p0_ - ib, ib).view(at::kLong);
      at::Tensor kd = lslot.narrow(0, p0_, gb).view(src_.dtype).view({rows, src_.D});
      at::Tensor vd = lslot.narrow(0, p0_ + gb, gb).view(src_.dtype).view({rows, src_.D});
      at::Tensor pf, pg;
      if (pub) {
        at::Tensor pr = lslot.narrow(0, lo, pb).view(at::kLong);
        pf = pr.narrow(0, 0, ng);
        pg = pr.narrow(0, ng, ng);
      }
      if (cuda_) {
        {
          c10::cuda::CUDAStreamGuard g(*copy_);
          ev_sc_[s].block(*copy_);                   // the scatter (and publication) that last read landing slot s
          ld.copy_(st, /*non_blocking=*/true);
          ev_h2d_[s].record(*copy_);
        }
        {
          c10::cuda::CUDAStreamGuard g(*scatter_);
          ev_h2d_[s].block(*scatter_);               // this chunk's H2D is done
          scr_k2d.index_copy_(0, idx, kd);
          scr_v2d.index_copy_(0, idx, vd);
          if (pub) board_.view({-1}).index_put_({at::indexing::TensorIndex(pf)}, pg);   // AFTER the chunk's rows
          ev_sc_[s].record(*scatter_);
        }
      } else {
        ld.copy_(st);
        scr_k2d.index_copy_(0, idx, kd);
        scr_v2d.index_copy_(0, idx, vd);
        if (pub) board_.view({-1}).index_put_({at::indexing::TensorIndex(pf)}, pg);
      }
      sub += now_ns() - t;
      ++k_;
      ++chunks;
    }
    completed_.fetch_add(1);
    return {r.n, chunks, layers, pack, wait, sub, now_ns() - t_all, team_lo, team_hi, pub ? r.n : 0};
  }

  std::shared_ptr<Job> wait_submitted(int64_t seq) {
    std::unique_lock<std::mutex> lk(mu_);
    TORCH_CHECK(seq >= 1 && seq <= next_seq_, "job ", seq, " was never submitted");
    cv_done_.wait(lk, [&] { return submitted_.load() >= seq || !running_; });
    TORCH_CHECK(submitted_.load() >= seq, "the worker stopped before job ", seq, " ran");
    return jobs_[seq - 1];
  }

  void run_job(Job& j) {
    try {
      if (!fatal_.empty()) TORCH_CHECK(false, fatal_);
      if (cuda_) {
        for (int64_t e : j.hist) {                   // T4: the host rows exist before the pack reads them
          if (e != 0) C10_CUDA_CHECK(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(e)));
        }
        for (int64_t e : j.readers) {                // T3: the scatter writes a slot after its last reader
          if (e != 0) C10_CUDA_CHECK(cudaStreamWaitEvent(scatter_->stream(), reinterpret_cast<cudaEvent_t>(e), 0));
        }
      }
      const auto opt = at::TensorOptions().dtype(at::kLong);
      auto vec = [&](std::vector<int64_t>& v) {      // a non-owning view: no Python object, no GIL at its death
        return v.empty() ? at::empty({0}, opt) : at::from_blob(v.data(), {static_cast<int64_t>(v.size())}, opt);
      };
      at::Tensor tb = vec(j.b), th = vec(j.h), tl = vec(j.l), tm = vec(j.m), tk = vec(j.blk);
      at::Tensor tf = vec(j.bflat), tg = vec(j.bgen);
      j.stats = release_impl(tb, th, tl, tm, tk, j.store_k.view({-1, src_.D}), j.store_v.view({-1, src_.D}), &tf,
                             &tg);
      TORCH_CHECK(j.stats[1] == 0 || (j.stats[7] == nt_ && j.stats[8] == nt_), "gilfree: packing teams of ",
                  j.stats[7], " .. ", j.stats[8], " threads != requested ", nt_);
      if (cuda_) j.done_ev.record(*scatter_);
    } catch (const c10::Error& e) {
      j.error = e.what_without_backtrace();
    } catch (const std::exception& e) {
      j.error = e.what();
    }
    j.store_k = at::Tensor();                         // Python keeps the store alive: no Python call here
    j.store_v = at::Tensor();
    for (std::vector<int64_t>* v : {&j.b, &j.h, &j.l, &j.m, &j.blk, &j.bflat, &j.bgen}) std::vector<int64_t>().swap(*v);
  }

  void worker_loop(bool inference, bool grad, int64_t device, std::vector<int64_t> cores) {
    if (!cores.empty()) {                            // this thread only (pid 0 = the calling thread on Linux)
      cpu_set_t set;
      CPU_ZERO(&set);
      for (int64_t c : cores) CPU_SET(static_cast<int>(c), &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) fatal_ = "gilfree worker: sched_setaffinity failed";
    }
    std::optional<c10::cuda::CUDAGuard> dev;
    try {
      if (cuda_) dev.emplace(static_cast<c10::DeviceIndex>(device));
    } catch (const std::exception& e) {
      fatal_ = std::string("gilfree worker: cannot enter the device: ") + e.what();
    }
    c10::InferenceMode im(inference);                // ONCE, at the worker's start (thread-local modes)
    c10::AutoGradMode gm(grad);
    {
      std::lock_guard<std::mutex> lk(mu_);
      worker_inference_ = c10::InferenceMode::is_enabled() ? 1 : 0;
      worker_grad_ = c10::GradMode::is_enabled() ? 1 : 0;
    }
    for (;;) {
      std::shared_ptr<Job> j;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_work_.wait(lk, [&] { return stop_ || !pending_.empty(); });
        if (pending_.empty()) break;               // stop_ and drained
        j = pending_.front();
        pending_.pop_front();
      }
      {
        const int64_t inf_now = c10::InferenceMode::is_enabled() ? 1 : 0;   // this thread's modes, for this job
        const int64_t grad_now = c10::GradMode::is_enabled() ? 1 : 0;
        std::lock_guard<std::mutex> lk(mu_);
        worker_inference_ = inf_now;
        worker_grad_ = grad_now;
      }
      run_job(*j);
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (!j->error.empty()) {
          if (error_.empty()) error_ = j->error;
        } else {
          for (int i = 0; i < 7; ++i) totals_[i] += j->stats[i];
          if (j->stats[1] > 0) {
            totals_[7] = totals_[7] < 0 ? j->stats[7] : std::min(totals_[7], j->stats[7]);
            totals_[8] = std::max(totals_[8], j->stats[8]);
          }
          totals_[9] += j->stats[9];
          ++ok_jobs_;
        }
        ++jobs_run_;
        submitted_.store(j->seq);
      }
      cv_done_.notify_all();
    }
  }

  at::Tensor stage_, land_, board_;
  int64_t cap_, R_, ring_ = 0, slot_bytes_ = 0, p0_ = 0, k_ = 0;
  int nt_ = 1;
  bool cuda_ = false, have_src_ = false, skip_publish_ = false;
  Src src_;
  std::optional<c10::cuda::CUDAStream> copy_, scatter_;
  std::vector<at::cuda::CUDAEvent> ev_h2d_, ev_sc_;
  // the worker
  std::mutex mu_;
  std::condition_variable cv_work_, cv_done_;
  std::deque<std::shared_ptr<Job>> pending_;
  std::vector<std::shared_ptr<Job>> jobs_;           // jobs_[seq - 1]
  std::thread worker_;
  bool running_ = false, stop_ = false;
  int64_t next_seq_ = 0, jobs_run_ = 0, ok_jobs_ = 0, worker_inference_ = -1, worker_grad_ = -1;
  std::atomic<int64_t> submitted_{0}, completed_{0};
  std::vector<int64_t> totals_;
  std::string error_, fatal_;
};

// TEST call (the GIL test): runs kick() and then waits up to timeout_ms for pipe.completed() >= target WHILE HOLDING
// THE GIL (bound without a GIL release). Returns [reached (0 / 1), completed, ms waited]. A C++ worker completes its
// release meanwhile; a Python feeder cannot (it needs the GIL to dequeue and to call release).
std::vector<int64_t> hold_gil_until(NativePipe& p, py::function kick, int64_t target, int64_t timeout_ms) {
  kick();
  const int64_t t0 = now_ns();
  while (p.completed() < target && now_ns() - t0 < timeout_ms * 1000000) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  const int64_t seen = p.completed();
  return {seen >= target ? 1 : 0, seen, (now_ns() - t0) / 1000000};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.doc() = "GIL-free CPU packer v3: one call per release (GIL released), or an optional C++ worker thread";
  mod.def("pack_into", &pack_into, py::call_guard<py::gil_scoped_release>(),
          "pack a whole release into flat outputs on `threads` threads (GIL released): [groups, rows, ns, team]");
  mod.def("pack_into_holding_gil", &pack_into,
          "NEGATIVE CONTROL of the GIL test: pack_into with the GIL held for the whole call");
  mod.def("hold_gil_until", &hold_gil_until,
          "TEST: kick(), then wait for completed() >= target holding the GIL: [reached, completed, ms]");
  py::class_<NativePipe>(mod, "NativePipe", py::module_local())   // the OpenMP and no-OpenMP builds load side by side
      .def(py::init<at::Tensor, at::Tensor, int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>>(),
           py::arg("stage"), py::arg("land"), py::arg("cap_groups"), py::arg("R"),
           py::arg("copy_stream") = std::vector<int64_t>{}, py::arg("scatter_stream") = std::vector<int64_t>{})
      .def("set_sources", &NativePipe::set_sources)
      .def("reset", &NativePipe::reset, py::call_guard<py::gil_scoped_release>())
      .def("release", &NativePipe::release, py::call_guard<py::gil_scoped_release>(),
           "one release (GIL released): [groups, chunks, layers, pack_ns, wait_ns, submit_ns, total_ns, threads_min, "
           "threads_max]")
      .def("set_threads", &NativePipe::set_threads, "the OpenMP team size of the packing loop (1 .. 256)")
      .def("set_board", &NativePipe::set_board, "the generation board (board.t.view(-1)) the worker publishes into")
      .def("set_skip_publish", &NativePipe::set_skip_publish, "NEGATIVE CONTROL (tests only): no board publication")
      .def("start_worker", &NativePipe::start_worker, py::call_guard<py::gil_scoped_release>(),
           "start the C++ worker: it enters (inference, grad, device) and its CPU affinity (cores) once")
      .def("stop_worker", &NativePipe::stop_worker, py::call_guard<py::gil_scoped_release>(),
           "drain the queued jobs, then join the worker")
      .def("submit", &NativePipe::submit, py::call_guard<py::gil_scoped_release>(),
           "queue one release for the worker: (b, h, l, m, blk lists, the layer's store K and V (B, rows, H, D), "
           "board indices, board generations, T4 event handles, T3 event handles) -> its job seq")
      .def("job_done", &NativePipe::job_done, py::call_guard<py::gil_scoped_release>())
      .def("wait_job", &NativePipe::wait_job, py::call_guard<py::gil_scoped_release>())
      .def("stream_wait_job", &NativePipe::stream_wait_job, py::call_guard<py::gil_scoped_release>())
      .def("take_error", &NativePipe::take_error)
      .def("worker_stats", &NativePipe::worker_stats)
      .def("modes_seen", &NativePipe::modes_seen, "the worker thread's own [inference, grad] modes (-1: not started)")
      .def_property_readonly("submitted", &NativePipe::submitted)
      .def_property_readonly("completed", &NativePipe::completed)
      .def_property_readonly("threads", &NativePipe::threads)
      .def_property_readonly("chunks_issued", &NativePipe::chunks_issued)
      .def_property_readonly("cuda", &NativePipe::is_cuda)
      .def_property_readonly("p0", &NativePipe::p0);
  mod.attr("VERSION") = 3;
  mod.attr("SLOT_EXTRA_PER_GROUP") = kPairBytes;
#ifdef _OPENMP
  mod.attr("OPENMP") = 1;
#else
  mod.attr("OPENMP") = 0;
#endif
}
