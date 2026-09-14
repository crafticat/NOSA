// Victim-pool bookkeeping for NOSI's GPU block cache (retroinfer-eval fork).
//
// WHY. NOSI's GPU cache is exact-fit: `topk` slots for a `topk`-block
// selection (cache_engine.py `prefill_update`, the `_k_gpu` allocation at
// :174-176), so a block that leaves the selection and re-enters it a few
// steps later is re-fetched over PCIe. Replaying the
// measured selection sequence through an LRU shows misses per (layer, KV
// head, request, step) fall from 3.580 at 63 usable slots to 0.708 at 128,
// so ~80% of the fetch is avoidable (REPRODUCE.md experiment 1).
//
// WHAT. This kernel adds P extra block slots per (layer, KV head, request)
// that live at slot indices topk .. topk+P-1 of the SAME cache tensors,
// BEYOND `_cache_lens`, where flash_attn_nosa never reads (it takes no block
// table: it reads rows 0.._cache_lens-1 as one flat sequence, so the attended
// set is exactly slots 0..topk-1 and must stay that way).
//
// This kernel does bookkeeping only. It decides, for each attended slot the
// upstream diff kernel marked "load from host", whether the wanted block is
// already sitting in the pool, and which pool slot the block being displaced
// from that attended slot should be parked in. The bytes are then moved by
// flash_pool_swap.py. A pool hit clears `load_mask` so the existing host
// gather skips that slot, which is the whole saving.
//
// CONTRACT (mirrored exactly by scripts/nosi_pool_reference.py, which is the
// CPU-testable oracle, and by tests/test_nosi_pool_reference.py):
//   block_map   [H,B,M] in   -- read BEFORE cache_engine.py:382's copy_, so
//                               entry s is Y, the block leaving attended slot s
//   load_mask   [H,B,M] io   -- diff's output; entry s is X >= 0 when slot s
//                               must be filled from host block X. Cleared to
//                               -1 on a pool hit.
//   pool_map    [H,B,P] io   -- block id parked in each pool slot, -1 = empty
//   pool_age    [H,B,P] io   -- LRU stamp; empty slot q carries -P+q, so every
//                               empty sorts below every real stamp (>= 0) and
//                               empties are consumed in slot order
//   pool_target [H,B,M] out  -- pool slot q this attended slot exchanges with
//   pool_action [H,B,M] out  -- POOL_NONE / SWAP / MOVE_IN / MOVE_OUT
//   stamp_base               -- clock * M; the caller increments clock once per
//                               decode_update call (one CacheEngine per layer,
//                               so one clock per layer). Stamps are
//                               stamp_base + s, pairwise distinct forever
//                               within one (h,b) pool, so the LRU order is a
//                               STRICT total order and the kernel is a pure
//                               deterministic function of its inputs.
//   tail_slot                -- topk-1, hard-excluded: it holds the in-progress
//                               tokens, is written locally, is never fetched
//                               (cache_engine.py:216-230, the tail assignment
//                               `_tail_block_idx_on_gpu = self.topk - 1` and
//                               the tail copies) and its host copy does not
//                               exist yet, so it must never be pooled.
//
// INVARIANT the algorithm preserves (proved by induction in the reference's
// docstring, asserted by the tests): attended-union-pool holds no duplicate
// block id. Hits are a bijective exchange, misses insert Y and drop the LRU
// victim, and a fetched X was in neither set. That distinctness is what makes
// the claim phases below single-writer -- but it is a property of the CALLER
// (diff must write every slot of load_map, which needs old_map to have no
// repeated entry), so phase 1 claims with an atomicCAS and DEMOTES the loser
// to a miss rather than trusting it. A duplicate then costs one fetch, not a
// corrupted cache.
//
// Modelled line for line on diff_offload_kernel.cu (the O(M) rank-then-select
// at :62-87 is the same pattern).
//
// STREAM ORDERING: diff_offload now launches on PyTorch's current device
// stream, just like the pool and Triton copies. All callers must enqueue the
// producer and its consumers in the same stream, or supply explicit events.
// This removes the old default-stream restriction; it does not make the host
// LRU timestamp or the entire decode loop CUDA-graph replay safe.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

// action codes; keep in sync with scripts/nosi_pool_reference.py and
// flash_cache_engine/flash_pool_swap.py
#define POOL_NONE      0
#define POOL_SWAP      1
#define POOL_MOVE_IN   2
#define POOL_MOVE_OUT  3

// s_hit[] sentinels
#define HIT_WANTS_SLOT (-1)   // active, missed the pool, has a block to park
#define HIT_INACTIVE   (-2)   // not loading, is the tail, or has nothing to park

extern "C" __global__ void pool_update_kernel(
    const int64_t* __restrict__ block_map,   // [H,B,M] in  (pre-copy: Y)
    int64_t* __restrict__ load_mask,         // [H,B,M] io  (X; -1 on a hit)
    int64_t* __restrict__ pool_map,          // [H,B,P] io
    int64_t* __restrict__ pool_age,          // [H,B,P] io
    int64_t* __restrict__ pool_target,       // [H,B,M] out
    int8_t*  __restrict__ pool_action,       // [H,B,M] out
    int H, int B, int M, int P,
    int64_t stamp_base, int tail_slot
){
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int m = threadIdx.x;          // attended slot index

    if (m >= M) return;

    const int64_t base  = (int64_t)(h * B + b) * M;
    const int64_t pbase = (int64_t)(h * B + b) * P;

    // shared layout: int64 first so the int32 arrays stay aligned
    extern __shared__ int64_t smem_i64[];
    int64_t* s_pmap = smem_i64;                 // [P]
    int64_t* s_page = smem_i64 + P;             // [P]
    int*     s_i32  = (int*)(smem_i64 + 2 * P);
    int*     s_claim = s_i32;                   // [P] attended slot owning q, -1 free
    int*     s_pos   = s_i32 + P;               // [P] rank among UNCLAIMED, -1 claimed
    int*     s_bylru = s_i32 + 2 * P;           // [P] inverse of s_pos
    int*     s_hit   = s_i32 + 3 * P;           // [M]

    // ---- phase 0: pull the pool row into shared, publish the defaults -------
    for (int j = m; j < P; j += M) {
        s_pmap[j]  = pool_map[pbase + j];
        s_page[j]  = pool_age[pbase + j];
        s_claim[j] = -1;
    }

    const int64_t X = load_mask[base + m];      // block wanted at this slot, -1 = none
    const int64_t Y = block_map[base + m];      // block leaving this slot, -1 = none
    const bool active = (m != tail_slot) && (X >= 0);

    pool_target[base + m] = -1;
    pool_action[base + m] = POOL_NONE;
    s_hit[m] = HIT_INACTIVE;

    __syncthreads();

    // ---- phase 1: hits. One pool slot must be claimed by at most ONE attended
    // slot. In a well-formed step that is automatic -- pool ids are pairwise
    // distinct (the invariant) and the non-negative entries of load_mask in one
    // (h,b) row are distinct too: diff writes load_map[target] = new_act[m] with
    // a distinct target per m, and new_act is torch.topk's index output
    // (nosa_llama.py:477-479), which has no repeats.
    //
    // BUT that rests on diff writing EVERY slot of load_map. It writes only the
    // old_hit slots and the target slots (diff_offload_kernel.cu:41-95), and
    // _load_mask is a reused buffer (cache_engine.py `_load_mask =
    // torch.empty_like(...)`, :182), so any future duplicate in _block_map
    // would leave a slot holding LAST step's id. If that stale id collided with
    // a fresh one, two threads would take this branch with the same q and
    // flash_pool_swap's disjointness argument would fail -- two programs
    // writing the same pool rows, wrong bytes in the attended window, silent.
    // atomicCAS costs one shared-memory op on ~3.6 of 64 threads per step and
    // turns that class of bug into ONE EXTRA FETCH: the loser is demoted to a
    // miss and takes the ordinary host-gather path.
    int hit_q = -1;
    if (active) {
        for (int j = 0; j < P; ++j) {
            if (s_pmap[j] == X) { hit_q = j; break; }
        }
        if (hit_q >= 0) {
            if (atomicCAS(&s_claim[hit_q], -1, m) == -1) {
                s_hit[m] = hit_q;                        // the claim is ours
            } else {
                hit_q = -1;                              // someone else owns q
                s_hit[m] = (Y >= 0) ? HIT_WANTS_SLOT : HIT_INACTIVE;
            }
        } else if (Y >= 0) {
            s_hit[m] = HIT_WANTS_SLOT;   // needs a victim slot to park Y in
        }
        // active, missed, and Y < 0: nothing to park, so it consumes no pool
        // slot and stays HIT_INACTIVE. The host gather runs as upstream.
    }

    __syncthreads();

    // ---- phase 2: rank among the threads that want a victim slot. Same O(M)
    // register scan as diff_offload_kernel.cu:66-71. Ascending attended-slot
    // index is the tiebreak: every Y displaced in one step has the same
    // recency, so which one is dropped on overflow is a tie, and this is the
    // same arbitrary within-step order the offline replay uses.
    int rank = -1;
    if (s_hit[m] == HIT_WANTS_SLOT) {
        int cnt = 0;
        for (int j = 0; j < m; ++j) {
            if (s_hit[j] == HIT_WANTS_SLOT) cnt++;
        }
        rank = cnt;
    }

    // ---- phase 3a: rank the UNCLAIMED pool slots by (age, slot index).
    // Cooperative: thread m owns slots m, m+M, ... so the cost is P*P/M
    // comparisons per thread in shared memory (16 at P=32, 256 at P=128, 1024
    // at P=256). If that ever shows in a profile, swap 3a/3b for a shared
    // bitonic sort of (age, index); the interface does not change.
    int unclaimed = 0;
    for (int i = 0; i < P; ++i) {
        if (s_claim[i] < 0) unclaimed++;
    }
    for (int j = m; j < P; j += M) {
        if (s_claim[j] >= 0) { s_pos[j] = -1; continue; }
        const int64_t aj = s_page[j];
        int r = 0;
        for (int i = 0; i < P; ++i) {
            if (s_claim[i] >= 0) continue;
            const int64_t ai = s_page[i];
            if (ai < aj || (ai == aj && i < j)) r++;   // ages are distinct; the
        }                                              // index tiebreak is belt
        s_pos[j] = r;                                  // and braces
    }

    __syncthreads();

    // ---- phase 3b: invert. s_pos is a bijection from the unclaimed set onto
    // 0..unclaimed-1, so s_bylru has exactly one writer per index.
    for (int j = m; j < P; j += M) {
        const int pj = s_pos[j];
        if (pj >= 0) s_bylru[pj] = j;
    }

    __syncthreads();

    // ---- phase 3c + 4: claim and commit. Every write below is single-writer:
    // hit slots were claimed in phase 1, victim slots come from s_bylru[rank]
    // and rank is a bijection from the wanting set onto 0..n_want-1, and the
    // two sets are disjoint by construction (s_bylru only lists unclaimed).
    if (active) {
        if (hit_q >= 0) {
            // The block is already in HBM. No PCIe, whatever else happens.
            load_mask[base + m] = -1;
            pool_target[base + m] = hit_q;
            if (Y >= 0) {
                pool_action[base + m] = POOL_SWAP;      // slot s <-> pool q
                pool_map[pbase + hit_q] = Y;
                pool_age[pbase + hit_q] = stamp_base + m;
            } else {
                pool_action[base + m] = POOL_MOVE_IN;   // pool q -> slot s
                pool_map[pbase + hit_q] = -1;           // q becomes empty
                pool_age[pbase + hit_q] = -(int64_t)P + hit_q;
            }
        } else if (rank >= 0) {
            if (rank < unclaimed) {
                const int q = s_bylru[rank];
                pool_target[base + m] = q;
                pool_action[base + m] = POOL_MOVE_OUT;  // slot s -> pool q
                pool_map[pbase + q] = Y;                // evicts the LRU block
                pool_age[pbase + q] = stamp_base + m;
                // load_mask keeps X: the host gather still fills slot s. Only
                // the return half of a full swap is skipped, which the gather
                // would have overwritten anyway.
            }
            // OVERFLOW (rank >= unclaimed): more blocks entered the selection
            // than the pool can hold -- the first decode step after prefill has
            // 63 of them (cache_engine.py:179, `_block_map = torch.full(...,
            // -1)`, though there every Y is -1 too, so nothing actually
            // overflows). No swap, load_mask keeps X, Y is dropped. Upstream
            // behaviour exactly.
        }
    }
}

void pool_update_cuda(
    at::Tensor block_map,    // [H,B,M] in  (pre-copy)
    at::Tensor load_mask,    // [H,B,M] io
    at::Tensor pool_map,     // [H,B,P] io
    at::Tensor pool_age,     // [H,B,P] io
    at::Tensor pool_target,  // [H,B,M] out
    at::Tensor pool_action,  // [H,B,M] out int8
    int64_t stamp_base,
    int64_t tail_slot
){
    const int H = (int)block_map.size(0);
    const int B = (int)block_map.size(1);
    const int M = (int)block_map.size(2);
    const int P = (int)pool_map.size(2);

    dim3 grid(H, B);
    dim3 block(M);

    const size_t smem = sizeof(int64_t) * 2 * (size_t)P
                      + sizeof(int)     * (3 * (size_t)P + (size_t)M);
    // 48 KB is the static per-block limit on every architecture we run
    // (sm_80 included; the opt-in 164 KB needs cudaFuncSetAttribute, which
    // this kernel does not do). Without this check a large P would fail the
    // launch, and an UNCHECKED failed launch would leave _pool_target /
    // _pool_action holding the PREVIOUS step's plan for flash_pool_swap to
    // execute against the current cache.
    TORCH_CHECK(smem <= 48 * 1024,
                "victim pool needs ", smem, " B of shared memory for P=", P,
                " M=", M, ", above the 48 KB static limit; lower NOSI_POOL_BLOCKS");

    c10::cuda::CUDAGuard device_guard(block_map.device());
    const auto stream = at::cuda::getCurrentCUDAStream(block_map.get_device());
    pool_update_kernel<<<grid, block, smem, stream>>>(
        block_map.data_ptr<int64_t>(),
        load_mask.data_ptr<int64_t>(),
        pool_map.data_ptr<int64_t>(),
        pool_age.data_ptr<int64_t>(),
        pool_target.data_ptr<int64_t>(),
        pool_action.data_ptr<int8_t>(),
        H, B, M, P,
        stamp_base, (int)tail_slot
    );
    // A launch that failed must be LOUD. _pool_target and _pool_action are
    // persistent buffers (cache_engine.py:193-194) and flash_pool_swap runs
    // unconditionally straight after, so a silently failed launch would move
    // last step's plan and corrupt the served output with no error at all.
    // This is cudaGetLastError + TORCH_CHECK; it does not synchronize.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
