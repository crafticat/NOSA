"""Pure-torch twins of the two movers in ``flash_pool_swap.py`` (CPU, no Triton).

WHY. Triton compiles only on a GPU, so neither ``pool_swap_kernel`` (grid
``(B, H, M)``, the shipped mover) nor ``pool_swap_compact_kernel`` (grid
``(B, H, split)``, NOSI_POOL_SWAP_COMPACT) can run on a login node. The claim
the compact kernel makes -- the same rows end up in the same slots, so
``torch.equal`` holds on K, V and bias after the call -- is a claim about
launch geometry and index arithmetic, and that part is torch. This module
mirrors each kernel's geometry and indexing line for line around ONE shared
per-slot body (``move_slot``, the masked loads then the masked stores of one
attended slot / pool slot pair), so the two twins differ exactly where the
two kernels differ and nowhere else. retroinfer-eval's
``tests/test_nosi_pool_swap_compact.py`` proves the twins ``torch.equal`` each
other on random buffers AND agree with the block-id oracle
``scripts/nosi_pool_reference.apply_moves`` on plans from
``nosi_pool_reference.Pool``, so they are twins of the contract and not merely
of each other; ``scripts/nosi_pool_parity.py`` is the GPU gate that ties the
kernels themselves to that oracle.

Both twins mutate the buffers IN PLACE, as the kernels do, and take the same
arguments as the wrappers ``flash_pool_swap`` / ``flash_pool_swap_compact``.
The plan tensors are indexed flat as ``h*B*M + b*M + m`` exactly as the kernels
index them (they take no strides for the plan), so both must be contiguous.
"""
import torch

# action codes; keep in sync with pool_update_kernel.cu and flash_pool_swap.py
POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3


def _shapes(k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action, topk, block_size):
    B, S_GPU, H, D = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert H == H2 and B == B2, (k_gpu.shape, pool_target.shape)
    assert M == topk, f"pool_target must be [H,B,topk]; got M={M}, topk={topk}"
    assert pool_action.shape == pool_target.shape
    assert v_gpu.shape == k_gpu.shape and tuple(kv_bias_buf.shape) == (B, S_GPU, H)
    assert pool_target.is_contiguous() and pool_action.is_contiguous(), \
        "the kernels index the plan as h*B*M + b*M + m with no strides"
    # P is derived from the tensor the kernel indexes (flash_pool_swap.py)
    P = S_GPU // block_size - topk
    assert P >= 1, f"no pool rows in the cache tensor: S_GPU={S_GPU}, topk={topk}"
    return B, S_GPU, H, D, M, P


def _bc(mask, like):
    """A (bs,) row mask broadcast over the trailing dims of ``like``."""
    return mask.view(-1, *([1] * (like.dim() - 1)))


def move_slot(k_gpu, v_gpu, kv_bias_buf, b, h, m, q, act, topk, block_size, S_GPU):
    """The per-slot body both kernels share, for attended slot ``m`` and pool
    slot ``q`` at ``(b, h)``: the truth table, the row masks, then for each of
    K, V and bias BOTH masked loads before EITHER masked store (a SWAP is an
    exchange, not a double write). ``tl.load(..., mask, other=0)`` is
    modelled as 'read the row if the mask holds, else 0', and a masked store
    as 'write the row if the mask holds'. With P derived from S_GPU every row
    is in range and both masks are all-true; they are kept so the twin is the
    kernel and not a simplification of it."""
    read_att = (act == POOL_SWAP) or (act == POOL_MOVE_OUT)    # attended -> pool
    read_pool = (act == POOL_SWAP) or (act == POOL_MOVE_IN)    # pool -> attended

    offset_s = torch.arange(block_size)
    att_row = m * block_size + offset_s
    pool_row = (topk + q) * block_size + offset_s
    mask_att = att_row < S_GPU
    mask_pool = pool_row < S_GPU
    load_att = mask_att & read_att
    load_pool = mask_pool & read_pool
    store_pool = mask_pool & read_att
    store_att = mask_att & read_pool

    for buf in (k_gpu, v_gpu, kv_bias_buf):
        # loads (advanced indexing copies, so a later store cannot alias them);
        # clamp keeps the twin's own indexing legal where the kernel's mask is
        # false, and torch.where then discards exactly those rows
        a = buf[b, att_row.clamp(max=S_GPU - 1), h]
        p = buf[b, pool_row.clamp(max=S_GPU - 1), h]
        a = torch.where(_bc(load_att, a), a, torch.zeros_like(a))
        p = torch.where(_bc(load_pool, p), p, torch.zeros_like(p))
        # stores
        buf[b, pool_row[store_pool], h] = a[store_pool]
        buf[b, att_row[store_att], h] = p[store_att]


def pool_swap_torch(k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action,
                    topk, block_size=64):
    """Twin of ``pool_swap_kernel`` under grid ``(B, H, M)``: one program per
    attended slot, three scalar early returns, then ``move_slot``."""
    B, S_GPU, H, D, M, P = _shapes(k_gpu, v_gpu, kv_bias_buf, pool_target,
                                   pool_action, topk, block_size)
    tgt = pool_target.reshape(-1).tolist()
    act = pool_action.reshape(-1).tolist()
    for pid_b in range(B):
        for pid_h in range(H):
            for pid_m in range(M):
                idx = pid_h * B * M + pid_b * M + pid_m
                a = act[idx]
                if a == POOL_NONE:
                    continue
                q = tgt[idx]
                if q < 0:
                    continue
                if q >= P:
                    continue
                move_slot(k_gpu, v_gpu, kv_bias_buf, pid_b, pid_h, pid_m, q, a,
                          topk, block_size, S_GPU)


def pool_swap_compact_torch(k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action,
                            topk, block_size=64, split=1):
    """Twin of ``pool_swap_compact_kernel`` under grid ``(B, H, split)``:
    program ``(b, h, r)`` reads its whole (h, b) row of the plan once, visits
    slots ``r, r+split, ...`` and applies the three early returns as one
    predicate before ``move_slot``."""
    B, S_GPU, H, D, M, P = _shapes(k_gpu, v_gpu, kv_bias_buf, pool_target,
                                   pool_action, topk, block_size)
    assert isinstance(split, int) and split >= 1, split
    tgt = pool_target.reshape(-1).tolist()
    act = pool_action.reshape(-1).tolist()
    for pid_b in range(B):
        for pid_h in range(H):
            for pid_r in range(split):
                row = pid_h * B * M + pid_b * M
                acts = act[row:row + M]          # the one vector load each
                tgts = tgt[row:row + M]
                for m in range(pid_r, M, split):
                    a = acts[m]
                    q = tgts[m]
                    if (a != POOL_NONE) and (q >= 0) and (q < P):
                        move_slot(k_gpu, v_gpu, kv_bias_buf, pid_b, pid_h, m, q, a,
                                  topk, block_size, S_GPU)
