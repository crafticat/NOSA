"""The CPU WORLD of tests_cpu/test_ragged_engine.py (Task 1.3; not a test file). Load it by file path:

    spec = importlib.util.spec_from_file_location("ragged_world", <tests_cpu>/ragged_world.py)

THE SHIPPED ENGINE RUNS FROM ITS OWN SOURCE. The classes CacheEngine / InfLLMv2CacheLayer / InfLLMv2Cache
(nosi/nosi/cache_engine.py), LlamaLayer / Llama and the helpers layer_norm / compress_k (nosi/nosi/nosa_llama.py) and
the stage-1 wrapper infllmv2_attn_stage1_fast (dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py)
are taken from the files by AST, compiled with their ORIGINAL file name and line numbers, and executed in a namespace
where every CUDA op is a torch stand-in (the pattern of tests/test_r2_decode_body.py, which runs the shipped decode
body from its source). The ragged engine (nosi/nosi/ragged_engine.py) and the R2 modules it ports are loaded through a
synthetic package (their relative imports resolve; nosi/__init__.py and its CUDA imports never run) and get THE SAME
stand-ins. A difference between the two paths is therefore a difference of the engine code, not of the fakes.

THE STAND-INS model each kernel's own arithmetic, with FLAT ADDRESSES where the kernel computes them (lesson L28):
  diff_offload        diff_kernel per (h, b) row (copied from tests/test_r2_decode_body.py:71-94)
  pool_update         pool_update_kernel.cu:88-256 per (h, b) row (phases 0-4; the claim in ascending slot order)
  flash_pool_swap     flash_pool_swap.py:89-214: P from the tensor's rows, strides from the tensor, loads before stores
  flash_h2d_from_mask(_bias)  flash_h2d_mask.py:10-88, flash_h2d_mask_bias.py:10-113: source rows masked by S_CPU,
                      the bias addressed with ITS OWN strides (findings F3 (c))
  nosa_pooling        max_pooling_fused.py:7-107: max_seq_len one scalar, M the buffer width, loads at flat addresses
                      (a compressed-cis column past the row reads the next row, findings F2; past the storage: POISON)
  nosa_pooling_per_request    max_pooling_per_request.py:63-181 (the same arithmetic, n_comp / M_b per request)
  varlen_fwd_stage1   the C++ op of both stage-1 wrappers: per request, per KV head, the softmax over its n_b keys
                      (seqused_k, else the cu_seqlens_k difference) summed over the group rows; zero past n_b
  torch.topk          CUDA's single-block order with sorted=False (TensorTopK.cu, sbtopk::gatherTopK): the elements
                      above the k-th value in index order, then the ones equal to it. CPU torch.topk may order them
                      otherwise, and the order reaches diff_offload's slot assignment (an ASSUMPTION about the CUDA
                      kernel; the GPU cell rg1 measures the result)
  flash_attn_nosa_with_kvcache  softmax attention with the additive kv_bias over rows [0, cache_seqlens[b]), SPLIT-KV:
                      the partition comes from the ALLOCATED rows (64-row n-blocks, num_splits splits), as
                      flash_api.cpp:338 / flash_fwd_kernel.h:594 derive it (job 2173299)
  torch.cuda.graph    FakeGraph: the capture records the with-block's statements and a SNAPSHOT of the caller's locals
                      (tensor objects = the pointers a CUDA graph keeps; Python ints frozen); replay() executes them
  rmsnorm, rope, nosa_linear, silu_and_mul: row-wise torch stand-ins (r2_fake_world.py's)
The CPU float32 data make a one-ULP difference visible. Unwritten host rows and window slots are POISONED with NaN.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import time
import types
import typing
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[3])
N = ROOT / "nosi/nosi"
EFF = ROOT / "benchmarks/Efficiency"
CE_PATH = N / "cache_engine.py"
NL_PATH = N / "nosa_llama.py"
IV2_PATH = ROOT / "dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py"
PKG = "raggedcpu"

B_MAX = 8                                   # the largest CPU batch of the suite
HID, HQ, HKV, D, I, V, L = 32, 4, 2, 8, 16, 50, 2
BS, TOPK = 64, 64
P = 8                                       # the CPU victim pool (C = 63 + 8)
SPLITS = 4                                  # NOSI_ATTN_SPLITS of the GPU cells
TAIL0 = (TOPK - 1) * BS                     # 4032
DT = torch.float32
POISON = 3.0e38                             # a finite poison for a pooling load past the storage
NONE_, SWAP, MOVE_IN, MOVE_OUT = 0, 1, 2, 3
HIT_WANTS, HIT_INACTIVE = -1, -2


# ---------------------------------------------------------------------------
# the synthetic package (the R2 modules and the ragged engine)
# ---------------------------------------------------------------------------
def pkg_load(name):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(N)]
        sys.modules[PKG] = pkg
    full = PKG + "." + name
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(N / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


# ---------------------------------------------------------------------------
# torch.cuda.graph and torch.topk stand-ins, the torch proxy
# ---------------------------------------------------------------------------
_AST = {}


def _file_ast(path):
    if path not in _AST:
        with open(path) as f:
            _AST[path] = ast.parse(f.read(), path)
    return _AST[path]


class FakeGraph:
    """torch.cuda.CUDAGraph: the capture keeps the with-block's code, the caller's globals and a SNAPSHOT of its
    locals; replay() runs that code on the snapshot (a rebound name is NOT seen, as a real graph keeps pointers)."""
    captures = 0

    def __init__(self):
        self.code = self.glob = self.loc = None
        self.replays = 0

    def replay(self):
        if self.code is None:
            raise RuntimeError("FakeGraph.replay before a capture")
        exec(self.code, self.glob, dict(self.loc))
        self.replays += 1


class fake_graph:
    """torch.cuda.graph(g): on entry, find the with statement at the caller's line in the caller's file."""

    def __init__(self, cuda_graph, *args, **kwargs):
        self.g = cuda_graph

    def __enter__(self):
        fr = sys._getframe(1)
        fn = fr.f_code.co_filename
        node = next((n for n in ast.walk(_file_ast(fn)) if isinstance(n, ast.With) and n.lineno == fr.f_lineno), None)
        if node is None:
            raise RuntimeError("fake_graph: no with statement at %s:%d" % (fn, fr.f_lineno))
        self.g.code = compile(ast.Module(body=node.body, type_ignores=[]), fn, "exec")
        self.g.glob = fr.f_globals
        self.g.loc = dict(fr.f_locals)
        FakeGraph.captures += 1
        return self.g

    def __exit__(self, *exc):
        return False


def cuda_topk(input, k, dim=-1, largest=True, sorted=True, out=None):
    """torch.topk with CUDA's sorted=False order (module docstring)."""
    if not largest or sorted or dim not in (-1, input.dim() - 1):
        raise AssertionError("cuda_topk models largest=True, sorted=False on the last dim only")
    x = input
    n = x.shape[-1]
    kth = x.sort(dim=-1, descending=True, stable=True).values[..., k - 1:k]
    cls = 2 - 2 * (x > kth).to(torch.int64) - (x == kth).to(torch.int64)      # above: 0, equal: 1, below: 2
    idx = (cls * n + torch.arange(n, device=x.device)).argsort(dim=-1)[..., :k]
    val = x.gather(-1, idx)
    if out is not None:
        out[0].copy_(val)
        out[1].copy_(idx)
        return out
    return val, idx


class _TorchProxy(types.ModuleType):
    def __getattr__(self, name):
        return getattr(torch, name)


NVTX = types.SimpleNamespace(range_push=lambda *a, **k: None, range_pop=lambda *a, **k: None)
TP = _TorchProxy("torch")
TP.topk = cuda_topk
TP.cuda = types.SimpleNamespace(CUDAGraph=FakeGraph, graph=fake_graph, synchronize=lambda *a, **k: None, nvtx=NVTX,
                                is_available=lambda: False)


@contextlib.contextmanager
def no_pin():
    """CacheEngine.prefill_update pins its host rows (cache_engine.py:195-196); a CPU-only torch cannot pin."""
    old = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.pin_memory = old


# ---------------------------------------------------------------------------
# flat addresses
# ---------------------------------------------------------------------------
def _flat(t):
    """(the whole storage as a 1-D view, t's storage offset, t's strides): a kernel's pointer arithmetic."""
    n = t.untyped_storage().nbytes() // t.element_size()
    return t.as_strided((n,), (1,), 0), t.storage_offset(), t.stride()


# ---------------------------------------------------------------------------
# stand-ins: model ops
# ---------------------------------------------------------------------------
def fake_rmsnorm(x, w, eps):
    """flashinfer rmsnorm: one program per row, so one row at a time here."""
    xf = x.float().reshape(-1, x.shape[-1])
    rows = [r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * w for r in xf.split(1)]
    return torch.cat(rows).reshape(x.shape).to(x.dtype)


def fake_silu_and_mul(h, out):
    half = h.shape[-1] // 2
    out.copy_(F.silu(h[..., :half]) * h[..., half:])


def fake_rope(pos, q, k, head_dim, cache, is_neox):
    f = (pos.to(q.dtype) % 7 + 1).view(-1, 1) * 0.1 + 1.0
    q.mul_(f)
    k.mul_(f)


def fake_nosa_linear(qkv, delta_weight, A, q_size, kv_size, kv_heads):
    """nosa_linear (Triton, one program per row): the q / k / v split and cis = A * softplus(v @ delta^T), row-wise."""
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size:q_size + kv_size].contiguous()
    v = qkv[..., q_size + kv_size:].contiguous()
    return q, k, v, A * F.softplus(row_linear(v, delta_weight))


def row_linear(x, w):
    """The CPU EXACT GEMM: one F.linear per row, so a row's float order never depends on the other rows (the GPU's
    EXACT GEMM is r2_spec_core.PaddedLinear; on the CPU the row's position in a padded buffer could pick another
    micro-kernel)."""
    lead, k = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k)
    return torch.cat([F.linear(x2[i:i + 1], w) for i in range(x2.shape[0])]).reshape(*lead, w.shape[0])


row_linear.__name__ = "row_linear"


def fake_stage1_op(q, k, v, out, cu_q, cu_k, seqused_k, leftpad_k, block_table, alibi, max_q, max_k, dropout_p,
                   scale, zero_tensors, causal, wl, wr, softcap, return_softmax, gen):
    """infllm_v2.C varlen_fwd_stage1 for decode rows (module docstring). Returns (p,), p (Hk, B, round128(max_k))."""
    B = cu_q.numel() - 1
    hk = k.shape[1]
    width = (int(max_k) + 127) // 128 * 128
    p = torch.zeros((hk, B, width), dtype=q.dtype)
    for b in range(B):
        r0, r1, k0 = int(cu_q[b]), int(cu_q[b + 1]), int(cu_k[b])
        n = int(seqused_k[b]) if seqused_k is not None else int(cu_k[b + 1]) - k0
        if n <= 0:
            continue
        sc = torch.einsum("ghd,nhd->hgn", q[r0:r1].float(), k[k0:k0 + n].float()) * scale
        p[:, b, :n] = torch.softmax(sc, dim=-1).sum(1).to(q.dtype)
    return (p,)


STAGE1_EXT = types.SimpleNamespace(varlen_fwd_stage1=fake_stage1_op)


def _pool_flat(q_score, cis_score, maxlen, mb, out_q, out_cis, init_blocks, local_p1, per_request, set_size=5,
               padding=1):
    """nosa_pool_kernel (per_request False) / nosa_pool_kernel_per_request (True) with flat loads."""
    H, B = q_score.shape[:2]
    M = out_q.shape[-1]
    m = torch.arange(M).view(1, 1, M, 1)
    i = torch.arange(8).view(1, 1, 1, 8)
    n = m * (set_size - 1) - padding + i
    mask = (n < maxlen.to(torch.int64).view(1, B, 1, 1)) & (n >= 0) & (i < set_size)
    h = torch.arange(H).view(H, 1, 1, 1)
    b = torch.arange(B).view(1, B, 1, 1)
    mm = torch.arange(M).view(1, 1, M)
    mbv = mb.to(torch.int64).view(1, B, 1)
    for src, dst in ((q_score, out_q), (cis_score, out_cis)):
        flat, off0, st = _flat(src)
        off = off0 + h * st[0] + b * st[1] + n * st[2]
        inb = (off >= 0) & (off < flat.numel())
        vals = flat[off.clamp(0, flat.numel() - 1)]
        vals = torch.where(inb, vals, torch.full_like(vals, POISON))
        vals = torch.where(mask.expand_as(vals), vals, torch.full_like(vals, float("-inf")))
        acc = vals.amax(-1)
        o = torch.where((mm < init_blocks) | (mbv - mm <= local_p1), torch.full_like(acc, float("inf")), acc)
        if per_request:
            o = torch.where(mm < mbv, o, torch.full_like(acc, float("-inf")))
        dst.copy_(o)


def fake_nosa_pooling(q_score, cis_score, max_seqlen, q_score_pooled, cis_score_pooled, block_size=32, stride=16,
                      padding=1, set_size=5, init_blocks=1, local_blocks=16):
    H, B = q_score.shape[:2]
    M = q_score_pooled.shape[-1]
    _pool_flat(q_score, cis_score, torch.full((B,), int(max_seqlen)), torch.full((B,), M), q_score_pooled,
               cis_score_pooled, init_blocks, local_blocks + 1, False, set_size, padding)


def fake_pool_per_request(q_score, cis_score, n_comp, M_b, q_score_pooled, cis_score_pooled, block_size=32,
                          stride=16, padding=1, set_size=5, init_blocks=1, local_blocks=16):
    MPR = pkg_load("max_pooling_per_request")
    H, B = q_score.shape[:2]
    MPR.check_lengths(n_comp, B, "n_comp", q_score.device)
    MPR.check_lengths(M_b, B, "M_b", q_score.device)
    _pool_flat(q_score, cis_score, n_comp, M_b, q_score_pooled, cis_score_pooled, init_blocks, local_blocks + 1, True,
               set_size, padding)


KBLOCK_N = 64


def split_partition(rows: int, num_splits: int):
    """(n-blocks, n-blocks per split) of the fake split-KV attention for `rows` allocated rows."""
    nb = -(-int(rows) // KBLOCK_N)
    ns = max(1, int(num_splits))
    return nb, -(-nb // ns)


def fake_attention(q, k_cache, v_cache, bias, cache_seqlens=None, num_splits=0):
    """flash_attn_nosa_with_kvcache (module docstring): per request, the splits of the ALLOCATED rows, each a partial
    softmax in float32, combined by their maxima."""
    Bq, _, Hq, Dq = q.shape
    R, Hk = k_cache.shape[1], k_cache.shape[2]
    _, per = split_partition(R, num_splits)
    out = torch.empty_like(q)
    for b in range(Bq):
        n = int(cache_seqlens[b])
        qq = q[b, 0].view(Hk, Hq // Hk, Dq).float()
        parts = []
        lo = 0
        while lo < n:
            hi = min(lo + per * KBLOCK_N, n)
            sc = torch.einsum("hgd,lhd->hgl", qq, k_cache[b, lo:hi].float()) * Dq ** -0.5 + \
                bias[b, lo:hi].float().t().unsqueeze(1)
            mx = sc.amax(-1, keepdim=True)
            e = torch.exp(sc - mx)
            parts.append((mx, e.sum(-1, keepdim=True), torch.einsum("hgl,lhd->hgd", e, v_cache[b, lo:hi].float())))
            lo = hi
        mx = torch.stack([p_[0] for p_ in parts]).amax(0)
        num = sum(torch.exp(p_[0] - mx) * p_[2] for p_ in parts)
        den = sum(torch.exp(p_[0] - mx) * p_[1] for p_ in parts)
        out[b, 0] = (num / den).reshape(Hq, Dq).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# stand-ins: the engine kernels
# ---------------------------------------------------------------------------
STATS = {}
TAG = ["none"]


def _stat(key, n=1):
    d = STATS.setdefault(TAG[0], {})
    d[key] = d.get(key, 0) + int(n)


def diff_offload_ref(old_map, new_act, new_map, load_map):
    """diff_kernel (diff_offload_kernel.cu:10-99) per (h, b) row; copied from tests/test_r2_decode_body.py:71-94."""
    H, Bm, M = old_map.shape
    s = old_map.sort(-1).values
    assert not bool(((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).any()), \
        "duplicate block id in a map row (diff_kernel needs distinct ids)"
    t = new_act.sort(-1).values
    assert not bool((t[..., 1:] == t[..., :-1]).any()), "duplicate id in the selection"
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
    _stat("loads", int((load_map >= 0).sum()))


def pool_update_ref(block_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base, tail_slot):
    """pool_update_kernel.cu:88-256, one (h, b) row at a time (the atomicCAS claim taken in ascending slot order)."""
    H, B, M = block_map.shape
    Pn = pool_map.shape[2]
    for h in range(H):
        for b in range(B):
            Y = block_map[h, b].tolist()
            X = load_mask[h, b].tolist()
            pmap = pool_map[h, b].tolist()
            page = pool_age[h, b].tolist()
            claim = [-1] * Pn
            target, action, hit = [-1] * M, [NONE_] * M, [HIT_INACTIVE] * M
            hit_q = [-1] * M
            active = [m != tail_slot and X[m] >= 0 for m in range(M)]
            for m in range(M):                                                  # phase 1
                if not active[m]:
                    continue
                q = next((j for j in range(Pn) if pmap[j] == X[m]), -1)
                if q >= 0 and claim[q] == -1:
                    claim[q] = m
                    hit[m] = hit_q[m] = q
                elif Y[m] >= 0:
                    hit[m] = HIT_WANTS
            rank, cnt = [-1] * M, 0                                             # phase 2
            for m in range(M):
                if hit[m] == HIT_WANTS:
                    rank[m], cnt = cnt, cnt + 1
            free = sorted((j for j in range(Pn) if claim[j] < 0), key=lambda j: (page[j], j))     # phase 3
            for m in range(M):                                                  # phase 4
                if not active[m]:
                    continue
                if hit_q[m] >= 0:
                    q = hit_q[m]
                    load_mask[h, b, m] = -1
                    target[m] = q
                    if Y[m] >= 0:
                        action[m], pmap[q], page[q] = SWAP, Y[m], stamp_base + m
                    else:
                        action[m], pmap[q], page[q] = MOVE_IN, -1, -Pn + q
                elif rank[m] >= 0:
                    if rank[m] < len(free):
                        q = free[rank[m]]
                        target[m], action[m], pmap[q], page[q] = q, MOVE_OUT, Y[m], stamp_base + m
                    else:
                        _stat("overflow")
            pool_map[h, b] = torch.tensor(pmap, dtype=pool_map.dtype)
            pool_age[h, b] = torch.tensor(page, dtype=pool_age.dtype)
            pool_target[h, b] = torch.tensor(target, dtype=pool_target.dtype)
            pool_action[h, b] = torch.tensor(action, dtype=pool_action.dtype)
            for a, name in ((SWAP, "swap"), (MOVE_IN, "move_in"), (MOVE_OUT, "move_out")):
                _stat(name, action.count(a))


def pool_swap_ref(k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action, topk, block_size=64):
    """flash_pool_swap (flash_pool_swap.py:89-214): every (b, h, m) with an action, both sides loaded before either is
    stored; flat addresses from the given tensors."""
    B, S_GPU, H, Dd = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert H == H2 and B == B2 and M == topk and pool_action.shape == pool_target.shape
    Pn = S_GPU // block_size - topk
    assert Pn >= 1, "no pool rows in the cache tensor"
    s = torch.arange(block_size)
    dd = torch.arange(Dd)
    hs, bs_, ms = (pool_action != 0).nonzero(as_tuple=True)
    for h, b, m in zip(hs.tolist(), bs_.tolist(), ms.tolist()):
        act, q = int(pool_action[h, b, m]), int(pool_target[h, b, m])
        if q < 0 or q >= Pn:
            continue
        read_att, read_pool = act in (SWAP, MOVE_OUT), act in (SWAP, MOVE_IN)
        att, pool = m * block_size + s, (topk + q) * block_size + s
        for t, with_d in ((k_gpu, True), (v_gpu, True), (kv_bias_buf, False)):
            f, o, st = _flat(t)
            ao = o + b * st[0] + att * st[1] + h * st[2]
            po = o + b * st[0] + pool * st[1] + h * st[2]
            ma, mp = att < S_GPU, pool < S_GPU
            if with_d:
                ao, po = ao[:, None] + dd * st[3], po[:, None] + dd * st[3]
                ma, mp = ma[:, None].expand_as(ao), mp[:, None].expand_as(po)
            av, pv = f[ao].clone(), f[po].clone()
            if read_att:
                f[po[mp]] = av[mp]
            if read_pool:
                f[ao[ma]] = pv[ma]


def _h2d(gpu, cpu, load_ids, block_size, bias_buf=None, bias=None):
    B, S_GPU, H, Dd = gpu.shape
    S_CPU = cpu.shape[1]
    assert tuple(load_ids.shape[:2]) == (H, B)
    h, b, m = (load_ids >= 0).nonzero(as_tuple=True)
    if h.numel() == 0:
        return
    cid = load_ids[h, b, m]
    s = torch.arange(block_size)
    srow, drow = cid[:, None] * block_size + s, m[:, None] * block_size + s
    msrc, mdst = srow < S_CPU, drow < S_GPU
    bb, hh = b[:, None], h[:, None]
    if bias_buf is not None:                                    # flash_h2d_mask_bias.py:43-59 (bias first)
        fb, ob, sb = _flat(bias)
        fbb, obb, sbb = _flat(bias_buf)
        src = ob + bb * sb[0] + srow * sb[1] + hh * sb[2]
        dst = obb + bb * sbb[0] + drow * sbb[1] + hh * sbb[2]
        assert not bool((src >= fb.numel())[msrc].any()), "a bias gather reads past the cis storage"
        vals = torch.where(msrc, fb[src.clamp(0, fb.numel() - 1)], torch.zeros((), dtype=bias_buf.dtype))
        fbb[dst[mdst]] = vals[mdst]
    dd = torch.arange(Dd)
    fs, os_, ss = _flat(cpu)
    fd, od, sd = _flat(gpu)
    src = (os_ + bb * ss[0] + srow * ss[1] + hh * ss[2])[..., None] + dd * ss[3]
    dst = (od + bb * sd[0] + drow * sd[1] + hh * sd[2])[..., None] + dd * sd[3]
    ms3, md3 = msrc[..., None].expand_as(src), mdst[..., None].expand_as(dst)
    assert not bool((src >= fs.numel())[ms3].any()), "a gather reads past the host storage"
    vals = torch.where(ms3, fs[src.clamp(0, fs.numel() - 1)], torch.zeros((), dtype=gpu.dtype))
    fd[dst[md3]] = vals[md3]


def h2d_ref(gpu_data, cpu_data, load_ids, block_size=64):
    _h2d(gpu_data, cpu_data, load_ids, block_size)


def h2d_bias_ref(gpu_data, cpu_data, kv_bias_buf, kv_bias, load_ids, block_size=64):
    _h2d(gpu_data, cpu_data, load_ids, block_size, kv_bias_buf, kv_bias)


# ---------------------------------------------------------------------------
# the shipped source, executed over the stand-ins
# ---------------------------------------------------------------------------
def _exec_nodes(path, names, ns):
    tree = _file_ast(str(path))
    keep, seen = [], set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            keep.append(node)
            seen.add(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            keep.append(node)
            seen.update(t.id for t in node.targets if isinstance(t, ast.Name))
    missing = set(names) - seen
    if missing:
        raise RuntimeError("%s: %s not found" % (path, sorted(missing)))
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


class _DynamicLayer:
    def __init__(self, *a, **k):
        pass


class _DynamicCache:
    def __init__(self, *a, **k):
        pass


IV2_NS = _exec_nodes(IV2_PATH, ["infllmv2_attn_stage1_fast"], dict(torch=TP, nvtx=NVTX, infllm_cuda=STAGE1_EXT))
TRACE_OFF = types.SimpleNamespace(TRACE=None)
AVAIL_OFF = types.SimpleNamespace(SPEC="", POLICY=None)
CE_NS = _exec_nodes(CE_PATH, ["CacheEngine", "InfLLMv2CacheLayer", "InfLLMv2Cache"], dict(
    torch=TP, F=F, os=os, Optional=typing.Optional, List=typing.List, Tuple=typing.Tuple, Union=typing.Union, nvtx=NVTX,
    diff=types.SimpleNamespace(diff_offload=diff_offload_ref), _pool=types.SimpleNamespace(pool_update=pool_update_ref),
    _flash_pool_swap=pool_swap_ref, POOL_BLOCKS=P, _tt=TRACE_OFF, _avail=AVAIL_OFF, flash_h2d_from_mask=h2d_ref,
    flash_h2d_from_mask_bias=h2d_bias_ref, DynamicLayer=_DynamicLayer, DynamicCache=_DynamicCache, CacheLayerMixin=object,
    Cache=object, StaticCache=object))
NL_NS = _exec_nodes(NL_PATH, ["QK_SELECT", "layer_norm", "calc_chunks_with_stride", "compress_k", "LlamaLayer",
                              "Llama"], dict(
    torch=TP, F=F, gc=None, time=time, nvtx=NVTX, lru_cache=lru_cache, tqdm=lambda x, **k: x, os=os,
    rmsnorm=fake_rmsnorm, silu_and_mul=fake_silu_and_mul, apply_rope_with_cos_sin_cache_inplace=fake_rope,
    infllmv2_attn_stage1_fast=IV2_NS["infllmv2_attn_stage1_fast"], nosa_pooling=fake_nosa_pooling,
    nosa_linear=fake_nosa_linear, flash_attn_nosa_with_kvcache=fake_attention, _ATTN_SPLITS=SPLITS, _tt=TRACE_OFF,
    _avail=AVAIL_OFF, InfLLMv2Cache=CE_NS["InfLLMv2Cache"], InfLLMv2CacheNoOffload=None))

FAKE_NL = types.SimpleNamespace(QK_SELECT=NL_NS["QK_SELECT"], _ATTN_SPLITS=SPLITS, layer_norm=NL_NS["layer_norm"],
                                nosa_linear=fake_nosa_linear, apply_rope_with_cos_sin_cache_inplace=fake_rope,
                                flash_attn_nosa_with_kvcache=fake_attention, silu_and_mul=fake_silu_and_mul)

RE = pkg_load("ragged_engine")
RDP = sys.modules[PKG + ".r2_decode_path"]
RB = sys.modules[PKG + ".r2_decode_body"]
TW = sys.modules[PKG + ".r2_positional_twin"]
WB = pkg_load("r2_writeback")
RDP._nl = lambda: FAKE_NL
RE.torch = TP
RE.nosa_pooling_per_request = fake_pool_per_request

KERNELS = dict(diff=diff_offload_ref, gather_k=h2d_ref, gather_v_bias=h2d_bias_ref, pool_update=pool_update_ref,
               pool_swap=pool_swap_ref, writeback=WB.masked_writeback_torch, stage1_ext=STAGE1_EXT)
CFG = types.SimpleNamespace(num_hidden_layers=L, num_key_value_heads=HKV, head_dim=D)


def bc():
    """benchmarks/Efficiency/r2_base_cell.py (its pure parts: rewarm_need, warm_state, ragged_schedule, ...)."""
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_base_cell
    return r2_base_cell


# ---------------------------------------------------------------------------
# the model, the caches, the prompts
# ---------------------------------------------------------------------------
def make_model(seed=0):
    """A tiny NOSA model made of the SHIPPED LlamaLayer / Llama classes (weights set directly)."""
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g) * 0.2
    layers = []
    for idx in range(L):
        lay = NL_NS["LlamaLayer"](idx, topk_blocks=TOPK, topk_q_blocks=16, init_blocks=1, window_blocks=16,
                                  block_size=BS, pooling_stride=16, pooling_block_size=32)
        lay.wqkv, lay.wo = r(HQ * D + 2 * HKV * D, HID), r(HID, HQ * D)
        lay.q_size, lay.kv_size = HQ * D, HKV * D
        lay.gate_up_proj, lay.down_proj = r(2 * I, HID), r(HID, I)
        lay.delta, lay.A = types.SimpleNamespace(weight=r(HKV, HKV * D)), -(r(HKV).abs() + 0.1)
        lay.input_layernorm_weight, lay.input_layernorm_variance_epsilon = 1 + r(HID), 1e-6
        lay.post_attention_layernorm_weight, lay.post_attention_layernorm_variance_epsilon = 1 + r(HID), 1e-6
        lay.num_heads, lay.num_key_value_heads, lay.head_dim = HQ, HKV, D
        layers.append(lay)
    m = NL_NS["Llama"].__new__(NL_NS["Llama"])
    m.device, m.dtype, m.layers, m.num_layers, m.block_size = "cpu", DT, layers, L, BS
    m.num_key_value_heads, m.has_buffers, m.config, m.cos_sin_cache = HKV, False, CFG, None
    m.embed_tokens, m.lm_head, m.norm_weight, m.norm_variance_epsilon = r(V, HID), r(V, HID), 1 + r(HID), 1e-6
    return m


def new_cache(max_gen):
    """An InfLLMv2Cache of the shipped classes, on the CPU in float32, host rows S + max_gen (cf. run_shipped)."""
    c = CE_NS["InfLLMv2Cache"](config=CFG, num_hidden_layers=L, has_kv_bias=True)
    for lay in c.layers:
        e = lay.cache_engine
        e.device, e.dtype, e.max_gen_len = torch.device("cpu"), DT, int(max_gen)
    return c


def prompt_rows(seed, S_list):
    g = torch.Generator().manual_seed(seed)
    return {(b, l): (torch.randn(S, HKV, D, generator=g), torch.randn(S, HKV, D, generator=g),
                     torch.randn(S, HKV, generator=g) * 0.1)
            for b, S in enumerate(S_list) for l in range(L)}


def shipped_prefill(cache, rows, S, B):
    """LlamaLayer.prefill_forward's cache calls (nosa_llama.py:210-234), one request at a time (prefill_inference,
    T = 1, :771-777), with the synthetic rows in place of the model's keys / values / cis."""
    with no_pin():
        for t in range(B):
            for l in range(L):
                k, v, c = rows[(t, l)]
                cu = torch.tensor([0, S], dtype=torch.int32)
                NL_NS["calc_chunks_with_stride"].cache_clear()           # its lru_cache keys on the tensor's id
                ck, cu_comp = NL_NS["compress_k"](k, cu, HKV, D)
                ck = ck.contiguous()
                mcomp = int((cu_comp[1:] - cu_comp[:-1]).max())
                c1 = c.unsqueeze(0)
                ucis = cache.update_uncompressed_cis(c1, l, t, B)
                cache.update_cis(c1.permute(2, 0, 1), l, t, B)
                cache.prefill_update_kv(k.unsqueeze(0), v.unsqueeze(0), ucis, l, t, B)
                cache.update_compress_k_prefill(ck, l, cu_comp, max_seqlen=mcomp, current_batch_pos=t, total_bsz=B)
                cache.update_no_compress_k_prefill(k.unsqueeze(0)[:, mcomp * 16:], l, 32, 16, current_batch_pos=t,
                                                   total_bsz=B)


def _poison_engine(e, b, S):
    nan = float("nan")
    e._k_cpu[b, S:] = nan
    e._v_cpu[b, S:] = nan
    for x in (e._k_gpu, e._v_gpu, e._kv_bias_gpu):
        x[b, :TAIL0] = nan
        x[b, TAIL0 + S % BS:TOPK * BS] = nan


def poison_shipped(cache, S):
    """NaN on every row that the shipped prefill did not write (host rows >= S, window slots 0..62, tail rows past
    S % 64, total_cis rows >= S, the shipped key / cis tails past their lengths). Pool rows stay zero (:256-258)."""
    nan = float("nan")
    for lay in cache.layers:
        e = lay.cache_engine
        for b in range(e._k_gpu.shape[0]):
            _poison_engine(e, b, S)
        lay.total_cis[:, S:] = nan
        lay.no_compress_k_cache[:, lay.no_compress_k_len:] = nan
        lay.tail_cis[:, :, lay.tail_cis_len:] = nan


def poison_built(rs, S_list):
    for lay in rs.cache.layers:
        for b, S in enumerate(S_list):
            _poison_engine(lay.cache_engine, b, S)


def bits(t):
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    return t


ENGINE_TABLES = ("_block_map", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_cache_lens", "_pool_map",
                 "_pool_age")
MAPS = ("_block_map", "_pool_map", "_pool_age")


# ---------------------------------------------------------------------------
# the drivers
# ---------------------------------------------------------------------------
def shipped_step(model, cache, tok, pos, warm):
    """One step of r2_base_cell.run_shipped (:415-426): the re-warm rule, then Llama.decode_inference."""
    BC = bc()
    l0 = cache.layers[0]
    need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen,
                               warm["state"])
    if need:
        warm["state"] = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
        warm["rewarms"].append(why)
        model.has_buffers = False
    cu = torch.arange(0, tok.shape[0] + 1, dtype=torch.int32)
    return model.decode_inference(tok, cu, pos, cache)


def run_equal(B, S, steps, seed=0, neg=None, f4_fix=False, kernels=None, compare_state=True):
    """THE SHIPPED STEP against the ragged engine at EQUAL lengths: two caches from the same synthetic prefill (the
    shipped prefill_update with the pool), the second bridged by ragged_engine.attach; teacher-forced on the
    shipped tokens; ops natural / torch twin / torch body / graph on. Raises AssertionError at the first difference
    (logits per step, then every engine table and the twin against the shipped tables). Returns the evidence."""
    STATS.clear()
    FakeGraph.captures = 0
    model = make_model(seed)
    rows = prompt_rows(seed + 1, [S] * B)
    caches = []
    for _ in range(2):
        c = new_cache(steps + 64)
        shipped_prefill(c, rows, S, B)
        poison_shipped(c, S)
        caches.append(c)
    A, Bc = caches
    ops = RE.RaggedOps(kernels or KERNELS, twin="torch", body="torch", graph=True, f4_fix=f4_fix, neg=neg)
    rs = RE.attach(model, Bc, S, steps, ops)
    g = torch.Generator().manual_seed(seed + 2)
    tok = torch.randint(0, V, (B, 1), generator=g)
    pos = torch.full((B, 1), S, dtype=torch.int32)
    warm = dict(state=None, rewarms=[])
    ev = dict(rewarms=None, p63_steps=[], mb=[], shipped_captures=0, ragged_captures=0)
    for i in range(steps):
        if (S + i) % BS == BS - 1:
            ev["p63_steps"].append(i)
        TAG[0] = "shipped"
        la = shipped_step(model, A, tok, pos, warm)
        TAG[0] = "ragged"
        lb = RE.decode_step(model, rs, tok, pos)
        ev["mb"].append(int(rs.M_b[0]))
        assert bool(torch.isfinite(la).all()), "step %d: the shipped logits are not finite" % i
        assert torch.equal(bits(la), bits(lb)), "step %d: the ragged logits differ from the shipped step" % i
        tok = la[:, -1].argmax(-1, keepdim=True)
        pos = pos + 1
    ev.update(rewarms=list(warm["rewarms"]), stats={k: dict(v) for k, v in STATS.items()}, rs=rs,
              captures=rs.captures, capture_steps=list(rs.capture_steps), all_captures=FakeGraph.captures)
    if compare_state:
        for l in range(L):
            ea, eb = A.layers[l].cache_engine, Bc.layers[l].cache_engine
            for name in ENGINE_TABLES:
                assert torch.equal(bits(getattr(ea, name)), bits(getattr(eb, name))), "layer %d: %s differs" % (l, name)
            assert ea._pool_stamp_base == eb._pool_stamp_base, "layer %d: _pool_stamp_base differs" % l
            la_, tw = A.layers[l], rs.st.twins[l]
            n = int(la_.cached_compressed_max_seqlen)
            assert torch.equal(tw.n_comp, torch.full((B,), n)), "layer %d: n_comp vs the shipped length" % l
            assert torch.equal(bits(tw.comp_k[:, :n]), bits(la_.compress_k_cache_varlen)), "layer %d: comp_k" % l
            p_end = S + steps
            assert torch.equal(bits(tw.total_cis[:, :p_end]), bits(la_.total_cis[:, :p_end])), "layer %d: cis" % l
            m0 = la_.compressed_cis.shape[-1]
            assert torch.equal(bits(tw.ccis[:, :, :m0]), bits(la_.compressed_cis)), "layer %d: ccis prefix" % l
    return ev


def ragged_ops(neg=None, body="torch", kernels=None, twin="bi"):
    """EXACT mode on the CPU: the row-wise GEMM and the fixed-order twin summaries."""
    return RE.RaggedOps(kernels or KERNELS, gemm="exact", linear=row_linear, twin=twin, body=body, graph=True,
                        neg=neg)


def run_solo(S_list, steps, seed=0, neg=None, p_active=0.7, all_active=8, kernels=None, body="torch"):
    """RAGGED AGAINST SOLO in EXACT mode: a batch with a prompt length per request (ragged_engine.build) against one
    B = 1 engine per request (the batch's widths), the same prompts and the same token stream per request. The first
    `all_active` steps every request is active, then each one with probability p_active. Raises AssertionError at the
    first difference (each active request's logits, every step; then its engine and twin rows). Returns evidence."""
    STATS.clear()
    model = make_model(seed)
    rows = prompt_rows(seed + 1, S_list)
    B = len(S_list)
    rs = RE.build(model, new_cache(0), S_list, steps, ragged_ops(neg, body, kernels), lambda b, l: rows[(b, l)],
                  pin=False)
    poison_built(rs, S_list)
    w = rs.meta["widths"]
    solos = []
    for b, S in enumerate(S_list):
        s = RE.build(model, new_cache(0), [S], steps, ragged_ops(), lambda _b, l, b=b: rows[(b, l)], pin=False,
                     widths=w)
        poison_built(s, [S])
        solos.append(s)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, V, (B, steps + 1), generator=g)
    k = torch.zeros(B, dtype=torch.int64)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    one = torch.ones(1, dtype=torch.bool)
    ev = dict(inactive=0, crossings=[0] * B, mb=[], acts=[])
    for i in range(steps):
        act = torch.ones(B, dtype=torch.bool) if i < all_active else torch.rand(B, generator=g) < p_active
        if not bool(act.any()):
            act[i % B] = True
        ev["inactive"] += int((~act).sum())
        ev["acts"].append(act.clone())
        ids = tokens[torch.arange(B), k].view(B, 1)
        p = S_t + k
        pos = p.view(B, 1).to(torch.int32)
        RE.set_activity(rs, act, act)
        TAG[0] = "batch"
        lg = RE.decode_step(model, rs, ids, pos)
        ev["mb"].append(rs.M_b.clone())
        TAG[0] = "solo"
        for b in range(B):
            if not bool(act[b]):
                continue
            RE.set_activity(solos[b], one, one)
            l1 = RE.decode_step(model, solos[b], ids[b:b + 1], pos[b:b + 1])
            assert bool(torch.isfinite(l1).all()), "step %d, request %d: the solo logits are not finite" % (i, b)
            assert torch.equal(bits(lg[b]), bits(l1[0])), \
                "step %d, request %d: the batch logits differ from the solo run" % (i, b)
            if (int(p[b]) + 1) % BS == 0:
                ev["crossings"][b] += 1
        k += act.to(torch.int64)
    for b in range(B):
        for l in range(L):
            e, e1 = rs.cache.layers[l].cache_engine, solos[b].cache.layers[l].cache_engine
            for name in ENGINE_TABLES:
                x, y = getattr(e, name), getattr(e1, name)
                xb, yb = (x[:, b], y[:, 0]) if name in MAPS else (x[b], y[0])
                if name == "_pool_age":                           # one clock per engine: compare the LRU ORDER
                    xb, yb = xb.argsort(-1), yb.argsort(-1)
                assert torch.equal(bits(xb), bits(yb)), "request %d, layer %d: %s differs from the solo run" % (
                    b, l, name)
            t, t1 = rs.st.twins[l], solos[b].st.twins[l]
            for name in ("pos", "n_comp", "n_ccis", "comp_k", "ring", "total_cis"):
                assert torch.equal(bits(getattr(t, name)[b]), bits(getattr(t1, name)[0])), \
                    "request %d, layer %d: twin %s differs from the solo run" % (b, l, name)
            assert torch.equal(bits(t.ccis[:, b]), bits(t1.ccis[:, 0])), "request %d: twin ccis differs" % b
    ev.update(rs=rs, solos=solos, stats={k_: dict(v) for k_, v in STATS.items()})
    return ev


def run_bodies(S_list, steps, seed=0, p_active=0.7, all_active=4):
    """The torch body (DeviceWritebackBody) against the fused body (r2_body_fused.FusedBody), both with the pool:
    two batches of the same prompts, tokens and activity (ops natural, twin torch). Raises AssertionError at the first
    difference (logits every step, then every engine table). Needs triton (the Triton interpreter on the CPU)."""
    STATS.clear()
    model = make_model(seed)
    rows = prompt_rows(seed + 1, S_list)
    B = len(S_list)
    states = []
    for body in ("torch", "fused"):
        ops = RE.RaggedOps(KERNELS, twin="torch", body=body, graph=True)
        s = RE.build(model, new_cache(0), S_list, steps, ops, lambda b, l: rows[(b, l)], pin=False)
        poison_built(s, S_list)
        states.append(s)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, V, (B, steps + 1), generator=g)
    k = torch.zeros(B, dtype=torch.int64)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    ev = dict(inactive=0, crossings=0)
    for i in range(steps):
        act = torch.ones(B, dtype=torch.bool) if i < all_active else torch.rand(B, generator=g) < p_active
        ev["inactive"] += int((~act).sum())
        p = S_t + k
        ev["crossings"] += int((act & ((p + 1) % BS == 0)).sum())
        ids = tokens[torch.arange(B), k].view(B, 1)
        pos = p.view(B, 1).to(torch.int32)
        out = []
        for s in states:
            RE.set_activity(s, act, act)
            out.append(RE.decode_step(model, s, ids, pos))
        assert torch.equal(bits(out[0][act]), bits(out[1][act])), "step %d: the fused body's logits differ" % i
        k += act.to(torch.int64)
    for l in range(L):
        e0, e1 = states[0].cache.layers[l].cache_engine, states[1].cache.layers[l].cache_engine
        for name in ENGINE_TABLES:
            assert torch.equal(bits(getattr(e0, name)), bits(getattr(e1, name))), "layer %d: %s differs" % (l, name)
    errs = [RE.errors(s) for s in states]
    assert errs[0] == errs[1] == (0, [0] * L), errs
    return ev
