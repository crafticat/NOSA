"""CPU TWINS of the CUDA / Triton ops that cached NOSA's pooled decode step calls (not a test file; test_pool_prefetch.py
and its Triton-interpreter child load it by path). Each twin states its kernel. No torch extension, no Triton.

  diff_twin           diff_kernel (flash_cache_engine/diff_offload_kernel.cu:10-99); the vectorized form of
                      tests/r2_fake_world.py diff_offload_ref: it writes ONLY the slots the kernel writes
  pool_update_twin    pool_update_kernel (flash_cache_engine/pool_update_kernel.cu:88-256), per (h, b) row, phase by
                      phase; the lowest m wins a contested claim (the kernel's atomicCAS winner is not fixed; in a
                      well-formed step no claim is contested)
  pool_swap_twin      pool_swap_kernel (flash_cache_engine/flash_pool_swap.py:89-176): both sides loaded before either is
                      stored; out-of-range q skipped
  h2d_twin            flash_h2d_from_mask_kernel (flash_h2d_mask.py:10-62) and, with bias_buf / bias,
                      flash_h2d_from_mask_bias_kernel (flash_h2d_mask_bias.py:10-81), on FLAT ADDRESSES (L28): every
                      address is the tensor's storage offset + its own strides, as the kernel computes it from the
                      pointer and .stride() of the tensor it receives (narrowed views, batch stride 0 included). The load
                      ids are read as a CONTIGUOUS (H, B, M) array, as the kernel reads them (it takes no stride for
                      them). A source row past S_CPU is refused (the kernel's masked load is undefined there).
"""
from __future__ import annotations

import torch

POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3
_INACTIVE, _WANTS = -2, -1


def _flat(t: torch.Tensor):
    """The whole storage of t as a 1-D tensor (element units) and t's own storage offset."""
    n = t.untyped_storage().nbytes() // t.element_size()
    return t.as_strided((n,), (1,), 0), t.storage_offset()


def diff_twin(old_map, new_act, new_map, load_map):
    H, Bm, M = old_map.shape
    old_hit = (old_map.unsqueeze(-1) == new_act.unsqueeze(-2)).any(-1)
    new_hit = (new_act.unsqueeze(-1) == old_map.unsqueeze(-2)).any(-1)
    new_map[old_hit] = old_map[old_hit]
    load_map[old_hit] = -1
    miss, free = ~new_hit, ~old_hit
    dummy = torch.full_like(new_act, M)
    by_rank = torch.full((H, Bm, M + 1), -2, dtype=torch.int64)
    by_rank.scatter_(-1, torch.where(miss, torch.cumsum(miss.long(), -1) - 1, dummy),
                     torch.where(miss, new_act, torch.full_like(new_act, -2)))
    val = by_rank.gather(-1, torch.where(free, torch.cumsum(free.long(), -1) - 1, dummy))
    write = free & (val != -2)
    new_map[write] = val[write]
    load_map[write] = val[write]


def pool_update_twin(block_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base, tail_slot):
    H, Bn, M = block_map.shape
    P = pool_map.shape[2]
    stamp_base, tail_slot = int(stamp_base), int(tail_slot)
    for h in range(H):
        for b in range(Bn):
            X = load_mask[h, b].tolist()
            Y = block_map[h, b].tolist()
            pm = pool_map[h, b].tolist()
            pa = pool_age[h, b].tolist()                      # phase 0: the ages BEFORE any write (s_page)
            claim = [-1] * P
            hit_q = [-1] * M
            state = [_INACTIVE] * M
            active = [m != tail_slot and X[m] >= 0 for m in range(M)]
            for m in range(M):                                # phase 1: hits, one claim per pool slot
                if not active[m]:
                    continue
                q = next((j for j in range(P) if pm[j] == X[m]), -1)
                if q >= 0:
                    if claim[q] == -1:
                        claim[q], hit_q[m], state[m] = m, q, q
                    else:
                        state[m] = _WANTS if Y[m] >= 0 else _INACTIVE
                elif Y[m] >= 0:
                    state[m] = _WANTS
            rank, cnt = [-1] * M, 0                           # phase 2: rank among the slots that want a victim
            for m in range(M):
                if state[m] == _WANTS:
                    rank[m], cnt = cnt, cnt + 1
            free = sorted((j for j in range(P) if claim[j] < 0), key=lambda j: (pa[j], j))   # phase 3
            lm, pm2, pa2 = list(X), list(pm), list(pa)
            tgt, act = [-1] * M, [POOL_NONE] * M
            for m in range(M):                                # phase 4: commit
                if not active[m]:
                    continue
                if hit_q[m] >= 0:
                    q = hit_q[m]
                    lm[m], tgt[m] = -1, q
                    if Y[m] >= 0:
                        act[m], pm2[q], pa2[q] = POOL_SWAP, Y[m], stamp_base + m
                    else:
                        act[m], pm2[q], pa2[q] = POOL_MOVE_IN, -1, -P + q
                elif 0 <= rank[m] < len(free):
                    q = free[rank[m]]
                    tgt[m], act[m], pm2[q], pa2[q] = q, POOL_MOVE_OUT, Y[m], stamp_base + m
            load_mask[h, b] = torch.tensor(lm, dtype=load_mask.dtype)
            pool_map[h, b] = torch.tensor(pm2, dtype=pool_map.dtype)
            pool_age[h, b] = torch.tensor(pa2, dtype=pool_age.dtype)
            pool_target[h, b] = torch.tensor(tgt, dtype=pool_target.dtype)
            pool_action[h, b] = torch.tensor(act, dtype=pool_action.dtype)


def pool_swap_twin(k, v, bias, pool_target, pool_action, topk, block_size=64):
    Bk, S_GPU, Hk, Dk = k.shape
    H2, B2, M = pool_target.shape
    assert Hk == H2 and Bk == B2 and M == topk
    P = S_GPU // block_size - topk
    bs = block_size
    for h in range(H2):
        for b in range(B2):
            for m in range(M):
                a = int(pool_action[h, b, m])
                if a == POOL_NONE:
                    continue
                q = int(pool_target[h, b, m])
                if q < 0 or q >= P:
                    continue
                att = slice(m * bs, (m + 1) * bs)
                pool = slice((topk + q) * bs, (topk + q + 1) * bs)
                read_att = a in (POOL_SWAP, POOL_MOVE_OUT)
                read_pool = a in (POOL_SWAP, POOL_MOVE_IN)
                for t in (k, v, bias):
                    ta, tp = t[b, att, h].clone(), t[b, pool, h].clone()
                    if read_att:
                        t[b, pool, h] = ta
                    if read_pool:
                        t[b, att, h] = tp


def h2d_twin(gpu, src, ids, bs=64, bias_buf=None, bias=None, log=None):
    """flash_h2d_from_mask (bias_buf None) or flash_h2d_from_mask_bias. log: a list that gets one entry per call:
    (gpu.data_ptr(), [(h, b, m, block) of every program that copied])."""
    Bg, S_GPU, Hg, Dg = gpu.shape
    S_CPU = src.shape[1]
    H2, B2, M = ids.shape
    assert Hg == H2 and Bg == B2, (gpu.shape, ids.shape)
    idf, ido = _flat(ids)
    hb = torch.arange(H2).view(-1, 1, 1)
    bb = torch.arange(B2).view(1, -1, 1)
    mm = torch.arange(M).view(1, 1, -1)
    blk = idf[ido + hb * B2 * M + bb * M + mm]
    h, b, m = (blk >= 0).nonzero(as_tuple=True)
    if log is not None:
        log.append((gpu.data_ptr(), list(zip(h.tolist(), b.tolist(), m.tolist(), blk[h, b, m].tolist()))))
    if h.numel() == 0:
        return
    k = blk[h, b, m]
    h_, b_, m_, k_ = (x.view(-1, 1, 1) for x in (h, b, m, k))
    r = torch.arange(bs).view(1, -1, 1)
    d = torch.arange(Dg).view(1, 1, -1)
    srow, drow = k_ * bs + r, m_ * bs + r
    if not bool((srow < S_CPU).all()):
        raise ValueError("h2d twin: a source row is past S_CPU (the kernel's masked load is undefined there)")
    keep = (drow < S_GPU)
    if bias_buf is not None:                                  # the bias kernel copies the bias rows first
        bf, bo = _flat(bias)
        tf, to = _flat(bias_buf)
        s0, s1, s2 = bias.stride()
        t0, t1, t2 = bias_buf.stride()
        sa = (bo + b_ * s0 + srow * s1 + h_ * s2)[..., 0]
        da = (to + b_ * t0 + drow * t1 + h_ * t2)[..., 0]
        kb = keep[..., 0]
        tf[da[kb]] = bf[sa[kb]].to(tf.dtype)
    sf, so = _flat(src)
    gf, go = _flat(gpu)
    s0, s1, s2, s3 = src.stride()
    g0, g1, g2, g3 = gpu.stride()
    sa = so + b_ * s0 + srow * s1 + h_ * s2 + d * s3
    da = go + b_ * g0 + drow * g1 + h_ * g2 + d * g3
    kk = keep.expand_as(da)
    gf[da[kk]] = sf[sa[kk]].to(gf.dtype)
