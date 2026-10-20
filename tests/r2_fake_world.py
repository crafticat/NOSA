"""The CPU FAKE WORLD of the R2 decode tests (not a test file): the R2 modules loaded through a synthetic package, a
tiny fake NOSA model, torch stand-ins for every CUDA op (the SAME for every path that uses them), and the CPU twin of
build_ragged. Used by tests/test_r2_capture_step.py (the capture-safe step) and tests/test_r2_base.py (the r2base
driver). Load it by file path (tests/ is not a package):

    spec = importlib.util.spec_from_file_location("r2_fake_world", <tests>/r2_fake_world.py)

It patches stage1_seqused, nosa_pooling_per_request and _nl in the synthetic package's r2_decode_path and r2_capture
(module attributes), so both the validated eager path and the capture-safe path run on the stand-ins.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
N = NOSI_REPO / "nosi/nosi"
EFF = NOSI_REPO / "benchmarks/Efficiency"
PKG = "r2capcpu"


def _pkg_load(name):
    """Load nosi/nosi/<name>.py as PKG.<name> (a synthetic package: the relative imports resolve, nosi/__init__.py and
    its CUDA imports are never run)."""
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


RC = _pkg_load("r2_capture")
RDP = sys.modules[PKG + ".r2_decode_path"]
RB = sys.modules[PKG + ".r2_decode_body"]
TW = sys.modules[PKG + ".r2_positional_twin"]
MPR = sys.modules[PKG + ".max_pooling_per_request"]
WB = _pkg_load("r2_writeback")

B, HID, HQ, HKV, D, I, V, L = 4, 32, 4, 2, 8, 16, 50, 2
BS, TOPK = 64, 64
S_LIST = [4100, 4157, 4223, 4160]          # S % 64 = 4, 61, 63, 0
STEPS, W = 80, 2


# ---------------------------------------------------------------------------
# the CPU stand-ins of the CUDA ops
# ---------------------------------------------------------------------------
def fake_layer_norm(hidden_states, eps, w):
    x = hidden_states.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w).to(hidden_states.dtype)


def fake_nosa_linear(qkv, delta_weight, A, q_size, kv_size, kv_heads):
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size:q_size + kv_size].contiguous()
    v = qkv[..., q_size + kv_size:].contiguous()
    return q, k, v, A * F.softplus(F.linear(v, delta_weight))


def fake_rope(pos, q, k, head_dim, cache, is_neox):
    f = (pos.to(q.dtype) % 7 + 1).view(-1, 1) * 0.1 + 1.0
    q.mul_(f)
    k.mul_(f)


def fake_attention(q, k_cache, v_cache, bias, cache_seqlens=None, num_splits=0):
    Bq, _, Hq, Dq = q.shape
    Hk = k_cache.shape[2]
    out = torch.empty_like(q)
    for b in range(Bq):
        n = int(cache_seqlens[b])
        qq = q[b, 0].view(Hk, Hq // Hk, Dq)
        s = torch.einsum("hgd,lhd->hgl", qq, k_cache[b, :n]) / Dq ** 0.5 + bias[b, :n].t().unsqueeze(1)
        out[b, 0] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, dim=-1), v_cache[b, :n]).reshape(Hq, Dq)
    return out


def fake_silu_and_mul(h, out):
    half = h.shape[-1] // 2
    out.copy_(F.silu(h[..., :half]) * h[..., half:])


def fake_stage1(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None, **kw):
    """(Hkv, B, Lk_max): per request the max over its query group of q . k over its padded key rows, 0 past n_b."""
    Bq, Hk = q.shape[0], k.shape[1]
    kk = k.view(Bq, max_seqlen_k, Hk, -1)
    s = torch.einsum("bhgd,blhd->bhgl", q.view(Bq, Hk, -1, q.shape[-1]), kk).amax(dim=2)
    cols = torch.arange(max_seqlen_k).view(1, 1, -1)
    s = torch.where(cols < seqused_k.to(torch.int64).view(-1, 1, 1), s, torch.zeros_like(s))
    return s.permute(1, 0, 2).contiguous()


def fake_pool(q_score, cis_score, n_comp, M_b, q_score_pooled, cis_score_pooled, block_size=32, stride=16,
              padding=1, set_size=5, init_blocks=1, local_blocks=16):
    rq, rc = MPR.pool_reference(q_score, cis_score, n_comp, M_b, q_score_pooled.shape[-1], init_blocks=init_blocks,
                                local_blocks=local_blocks)
    q_score_pooled.copy_(rq)
    cis_score_pooled.copy_(rc)


FAKE_NL = types.SimpleNamespace(QK_SELECT=33, _ATTN_SPLITS=4, layer_norm=fake_layer_norm, nosa_linear=fake_nosa_linear,
                                apply_rope_with_cos_sin_cache_inplace=fake_rope,
                                flash_attn_nosa_with_kvcache=fake_attention, silu_and_mul=fake_silu_and_mul)
for _mod in (RDP, RC):
    _mod.stage1_seqused = fake_stage1
    _mod.nosa_pooling_per_request = fake_pool
RDP._nl = lambda: FAKE_NL


def diff_offload_ref(old_map, new_act, new_map, load_map):
    """diff_kernel (diff_offload_kernel.cu:10-99) per (h, b) row (tests/test_r2_decode_body.py has the same)."""
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


def _gather(gpu, cpu, load_ids, bs, bias_buf=None, bias=None):
    h, b, m = (load_ids >= 0).nonzero(as_tuple=True)
    if h.numel() == 0:
        return
    ids = load_ids[h, b, m]
    r = torch.arange(bs)
    dst, src = m[:, None] * bs + r, ids[:, None] * bs + r
    gpu[b[:, None], dst, h[:, None]] = cpu[b[:, None], src, h[:, None]]
    if bias_buf is not None:
        bias_buf[b[:, None], dst, h[:, None]] = bias[b[:, None], src, h[:, None]]


STUBS = (diff_offload_ref, lambda g, c, ids, bs: _gather(g, c, ids, bs),
         lambda g, c, bb, bias, ids, bs: _gather(g, c, ids, bs, bb, bias))


class CpuAdapter:
    """R2Body on the CPU: DecodeBody.decode (the torch write-back) with the stand-in kernels."""

    def __init__(self):
        self.body = RB.DecodeBody(False)

    def update(self, eng, key, value, kv_bias, topk_idx, pos, active, pos_host, active_host):
        return self.body.decode(eng, key.unsqueeze(1), value.unsqueeze(1), kv_bias, topk_idx, pos, active, pos_host,
                                active_host, *STUBS)


# ---------------------------------------------------------------------------
# the fake model and the CPU twin of build_ragged
# ---------------------------------------------------------------------------
def make_model(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g) * 0.2

    layers = [types.SimpleNamespace(
        num_key_value_heads=HKV, head_dim=D, num_heads=HQ, q_size=HQ * D, kv_size=HKV * D,
        input_layernorm_variance_epsilon=1e-6, input_layernorm_weight=1 + r(HID), wqkv=r(HQ * D + 2 * HKV * D, HID),
        delta=types.SimpleNamespace(weight=r(HKV, HKV * D)), A=-(r(HKV).abs() + 0.1), pooling_stride=16, block_size=BS,
        window_blocks=16, init_blocks=1, topk_blocks=TOPK, wo=r(HID, HQ * D),
        post_attention_layernorm_variance_epsilon=1e-6, post_attention_layernorm_weight=1 + r(HID),
        gate_up_proj=r(2 * I, HID), down_proj=r(HID, I)) for _ in range(L)]
    return types.SimpleNamespace(layers=layers, embed_tokens=r(V, HID), cos_sin_cache=None, norm_weight=1 + r(HID),
                                 norm_variance_epsilon=1e-6, lm_head=r(V, HID))


def build(model, seed=1, horizon=STEPS, S_list=None):
    """build_ragged's tables on the CPU, from synthetic prompt K / V / cis (no model prefill); deterministic.
    Returns (cache, state, first tokens (B, 1) int64)."""
    S_list = list(S_list or S_LIST)
    Bn = len(S_list)
    g = torch.Generator().manual_seed(seed)
    w = RDP.ragged_widths(S_list, horizon)
    adapter = CpuAdapter()
    engines = []
    for _ in range(L):
        e = types.SimpleNamespace(head_num=HKV, head_dim=D, block_size=BS, topk=TOPK, dtype=torch.float32,
                                  device=torch.device("cpu"), pool_blocks=0)
        adapter.body.allocate(e, Bn, w["n_host"], False)
        for k in ("_k_cpu", "_v_cpu", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_new_block_map_buf", "_load_mask"):
            getattr(e, k).zero_()
        engines.append(e)
    twins = [TW.PositionalTwin(Bn, HKV, D, w["lk_max"], w["n_max"]) for _ in range(L)]
    for b, S in enumerate(S_list):
        for l in range(L):
            k = torch.randn((S, HKV, D), generator=g)
            v = torch.randn((S, HKV, D), generator=g)
            c = torch.randn((S, HKV), generator=g) * 0.1
            twins[l].prefill(b, k, c)
            adapter.body.prefill_request(engines[l], b, k, v, twins[l].total_cis[b, :S])
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e) for e in engines])
    st = RDP.R2State.__new__(RDP.R2State)
    st.B, st.H, st.S = Bn, HKV, max(S_list)
    st.lk_max, st.n_max, st.M_alloc = w["lk_max"], w["n_max"], w["M_alloc"]
    st.twins = twins
    st._common(model, Bn, HKV, torch.device("cpu"), "r2", adapter, False, None, torch.tensor(S_list, dtype=torch.int64))
    cache.r2_state = st
    return cache, st, torch.randint(0, V, (Bn, 1), generator=g)
