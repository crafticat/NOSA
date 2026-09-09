// pybind wrapper for the victim-pool bookkeeping kernel (retroinfer-eval fork).
//
// Built as its OWN extension (name "nosi_pool", cache_engine.py module scope)
// rather than added to the existing `load(name="diff_offload", ...)` sources:
// adding sources there would change that extension's build hash and force
// every NOSI_POOL_BLOCKS=0 run to rebuild, which breaks "p=0 executes exactly
// as today" at the build level. With the pool off this file is never compiled.
//
// The TORCH_CHECKs below are deliberately stricter than diff_offload.cpp's
// (which has none): the kernel indexes shared memory from these shapes and a
// silent mismatch would corrupt the served output rather than merely lose
// performance.
#include <torch/extension.h>

void pool_update_cuda(
    at::Tensor block_map,    // [H,B,M] in  (pre-copy)
    at::Tensor load_mask,    // [H,B,M] io
    at::Tensor pool_map,     // [H,B,P] io
    at::Tensor pool_age,     // [H,B,P] io
    at::Tensor pool_target,  // [H,B,M] out
    at::Tensor pool_action,  // [H,B,M] out int8
    int64_t stamp_base,
    int64_t tail_slot
);

static void check_i64(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(t.dim() == 3, name, " must be [H,B,*]");
}

void pool_update(
    torch::Tensor block_map,
    torch::Tensor load_mask,
    torch::Tensor pool_map,
    torch::Tensor pool_age,
    torch::Tensor pool_target,
    torch::Tensor pool_action,
    int64_t stamp_base,
    int64_t tail_slot
){
    check_i64(block_map, "block_map");
    check_i64(load_mask, "load_mask");
    check_i64(pool_map, "pool_map");
    check_i64(pool_age, "pool_age");
    check_i64(pool_target, "pool_target");

    TORCH_CHECK(pool_action.is_cuda() && pool_action.is_contiguous(),
                "pool_action must be a contiguous CUDA tensor");
    TORCH_CHECK(pool_action.scalar_type() == at::kChar, "pool_action must be int8");
    TORCH_CHECK(pool_action.dim() == 3, "pool_action must be [H,B,M]");

    const int64_t H = block_map.size(0);
    const int64_t B = block_map.size(1);
    const int64_t M = block_map.size(2);
    const int64_t P = pool_map.size(2);

    TORCH_CHECK(load_mask.size(0) == H && load_mask.size(1) == B && load_mask.size(2) == M,
                "load_mask must be [H,B,M]");
    TORCH_CHECK(pool_target.size(0) == H && pool_target.size(1) == B && pool_target.size(2) == M,
                "pool_target must be [H,B,M]");
    TORCH_CHECK(pool_action.size(0) == H && pool_action.size(1) == B && pool_action.size(2) == M,
                "pool_action must be [H,B,M]");
    TORCH_CHECK(pool_age.size(0) == H && pool_age.size(1) == B && pool_age.size(2) == P,
                "pool_age must match pool_map [H,B,P]");

    TORCH_CHECK(P >= 1, "pool_update needs at least one pool slot (got P=", P, ")");
    // The kernel asks for sizeof(int64)*2*P + sizeof(int)*(3P+M) = 28P + 4M
    // bytes of DYNAMIC shared memory, which the driver refuses above the 48 KB
    // static limit -- i.e. from about P = 1700. Bound it here, where the
    // message can name the knob, as well as at the launch.
    const int64_t smem = 8 * 2 * P + 4 * (3 * P + M);
    TORCH_CHECK(smem <= 48 * 1024,
                "NOSI_POOL_BLOCKS=", P, " needs ", smem,
                " B of shared memory per block (28P + 4M), above the 48 KB limit");
    TORCH_CHECK(M >= 1 && M <= 1024, "M must be a legal thread-block width (got ", M, ")");
    TORCH_CHECK(tail_slot >= 0 && tail_slot < M, "tail_slot ", tail_slot, " out of range [0,", M, ")");
    // -P+q is the "empty" stamp; real stamps are stamp_base+m >= 0, so every
    // empty pool slot must sort strictly below every real one.
    TORCH_CHECK(stamp_base >= 0, "stamp_base must be non-negative (got ", stamp_base, ")");

    pool_update_cuda(block_map, load_mask, pool_map, pool_age,
                     pool_target, pool_action, stamp_base, tail_slot);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pool_update", &pool_update, "");
}
