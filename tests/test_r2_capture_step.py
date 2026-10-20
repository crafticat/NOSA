"""CPU tests of the capture-safe decode step (nosi/nosi/r2_capture.py) and of the capture gate's plumbing
(benchmarks/Efficiency/r2_capture_gate.py). Design: docs/R2_CAPTURE_AUDIT.md, "X4 and the capture gate".

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2-capture python -m pytest -q tests/test_r2_capture_step.py

A TINY FAKE MODEL on the CPU (B 4, 2 layers, 2 KV heads of 8, 4 query heads, 50 tokens; the shipped block 64 and
topk 64, prompts of 4100 .. 4223 tokens): the CUDA ops (rmsnorm, nosa_linear, rope, stage 1, pooling, attention,
silu_and_mul, the diff kernel and the gathers) are torch stand-ins, THE SAME for both paths. So a difference is a
difference of the R2 code: the validated eager r2_decode_path.decode_inference (CE) against r2_capture.
decode_step_fixed driven by prepare_step / finish_step (CF), on every request, every step (the gate's own
record_all, tables_digest, r2_g0_compare.compare_views and verdicts).

S1  CF == CE bitwise on every request at every step and in the final tables, ragged activity, 80 steps (several block
    boundaries); refusal bits 0; no state pointer moves after the first step.
S2  NG1 (cb.freeze_widths at step W, the eager twin of the captured control): the first difference is exactly
    r2_capture_gate.ng1_first_step.
S3  NG2 (the device mask not written after step W): the first difference is exactly ng2_first_step.
S4  a driver that skips finish_step: the first difference is step 1.
S5  state_pointers / pointer_diff: a rebound buffer is named.
S6  the gate's pure parts: parse_args refusals, positions_before, ng1 / ng2 on hand cases, digest.
GPU-only (the gate): the capture itself, the replay, the Triton write-back, the real kernels.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
N = NOSI_REPO / "nosi/nosi"
EFF = NOSI_REPO / "benchmarks/Efficiency"
if not (N / "r2_capture.py").is_file() or not (EFF / "r2_capture_gate.py").is_file():
    pytest.skip("no r2_capture.py / r2_capture_gate.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)

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
sys.path.insert(0, str(EFF))
import r2_capture_gate as G   # noqa: E402
import r2_g0_compare as C     # noqa: E402

B, HID, HQ, HKV, D, I, V, L = 4, 32, 4, 2, 8, 16, 50, 2
BS, TOPK = 64, 64
S_LIST = [4100, 4157, 4223, 4160]          # S % 64 = 4, 61, 63, 0
STEPS, W = 80, 2


# ---------------------------------------------------------------------------
# the CPU stand-ins of the CUDA ops (the same for both paths)
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


def build(model, seed=1, horizon=STEPS):
    """build_ragged's tables on the CPU, from synthetic prompt K / V / cis (no model prefill); deterministic."""
    g = torch.Generator().manual_seed(seed)
    w = RDP.ragged_widths(S_LIST, horizon)
    adapter = CpuAdapter()
    engines = []
    for _ in range(L):
        e = types.SimpleNamespace(head_num=HKV, head_dim=D, block_size=BS, topk=TOPK, dtype=torch.float32,
                                  device=torch.device("cpu"), pool_blocks=0)
        adapter.body.allocate(e, B, w["n_host"], False)
        for k in ("_k_cpu", "_v_cpu", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_new_block_map_buf", "_load_mask"):
            getattr(e, k).zero_()
        engines.append(e)
    twins = [TW.PositionalTwin(B, HKV, D, w["lk_max"], w["n_max"]) for _ in range(L)]
    for b, S in enumerate(S_LIST):
        for l in range(L):
            k = torch.randn((S, HKV, D), generator=g)
            v = torch.randn((S, HKV, D), generator=g)
            c = torch.randn((S, HKV), generator=g) * 0.1
            twins[l].prefill(b, k, c)
            adapter.body.prefill_request(engines[l], b, k, v, twins[l].total_cis[b, :S])
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e) for e in engines])
    st = RDP.R2State.__new__(RDP.R2State)
    st.B, st.H, st.S = B, HKV, max(S_LIST)
    st.lk_max, st.n_max, st.M_alloc = w["lk_max"], w["n_max"], w["M_alloc"]
    st.twins = twins
    st._common(model, B, HKV, torch.device("cpu"), "r2", adapter, False, None, torch.tensor(S_LIST, dtype=torch.int64))
    cache.r2_state = st
    return cache, st, torch.randint(0, V, (B, 1), generator=g)


def activity(seed=6):
    return RG_activity(STEPS, B, 0.7, seed)


def RG_activity(steps, b, p, seed):
    return G.RG.activity(steps, b, p, seed, True)


def run_ce(model, act):
    cache, st, tok = build(model)
    recs = []
    cu = torch.arange(B + 1, dtype=torch.int32)
    for s in range(STEPS):
        st.set_active(act[s])
        pos = st.pos_host.view(B, 1).to(torch.int32)
        lg = RDP.decode_inference(model, tok, cu, pos, cache)
        tok = torch.where(st.active.view(B, 1), lg[:, -1].argmax(-1, keepdim=True), tok)
        recs.append(G.record_all(cache, st.twins, lg, torch.stack(st.topk_log)))
    return dict(steps=recs, tables=G.tables_digest(cache, st.twins))


def run_cf(model, act, ref, mode=None):
    """The fixed step with the driver; mode None | 'frozen' (NG1) | 'stale' (NG2) | 'nofinish'. Returns (tracker,
    errors seen, pointer differences after the first step)."""
    tr = C.Tracker()
    with torch.inference_mode():
        cache, st, tok = build(model)
        cb = RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch)
        errs, ptr0 = [], None
        for s in range(STEPS):
            RC.prepare_step(st, cb, tok, act[s], write_active=not (mode == "stale" and s > W))
            if mode == "frozen" and s == W:
                cb.freeze_widths(st)
            RC.decode_step_fixed(model, st, cb, cache)
            if mode != "nofinish":
                RC.finish_step(st)
            body_err, twin_err = RC.errors(st, cb)
            if body_err or any(twin_err):
                errs.append((s, body_err, twin_err))
            if s == 0:
                ptr0 = RC.state_pointers(st, cb, cache)
            tok = torch.where(st.active.view(B, 1), cb.logits[:, -1].argmax(-1, keepdim=True), tok)
            tr.update(s, None, C.compare_views(ref["steps"][s], G.record_all(cache, st.twins, cb.logits, cb.topk_log)))
        tr.update(STEPS - 1, None, C.compare_views(ref["tables"], G.tables_digest(cache, st.twins)))
        moved = G.pointer_diff(ptr0, RC.state_pointers(st, cb, cache))
    return tr, errs, moved


@pytest.fixture(scope="module")
def world():
    model = make_model()
    act = activity()
    return model, act, run_ce(model, act)


def test_S1_fixed_step_equals_the_validated_eager_step(world):
    model, act, ref = world
    tr, errs, moved = run_cf(model, act, ref)
    ok, why = C.verdict_equal(tr)
    assert ok, why
    assert errs == [] and moved == []
    pb = G.positions_before(S_LIST, act)
    crossed = sum(int(((pb[-1] // 64) > (pb[0] // 64))[b]) for b in range(B))
    assert crossed >= 3, "the run must cross block boundaries (crossed %d)" % crossed
    assert not bool(act.all()) and bool(act[:, 0].all())


def test_S2_ng1_frozen_widths_first_differs_where_predicted(world):
    model, act, ref = world
    s0 = G.ng1_first_step(S_LIST, act, W)
    assert s0 is not None
    tr, _, _ = run_cf(model, act, ref, mode="frozen")
    ok, why = C.verdict_neg(tr, s0, None)
    assert ok, why


def test_S3_ng2_stale_mask_first_differs_where_predicted(world):
    model, act, ref = world
    s0 = G.ng2_first_step(act, W)
    assert s0 is not None
    tr, _, _ = run_cf(model, act, ref, mode="stale")
    ok, why = C.verdict_neg(tr, s0, None)
    assert ok, why


def test_S4_a_driver_without_finish_step_differs_at_step_1(world):
    model, act, ref = world
    tr, _, _ = run_cf(model, act, ref, mode="nofinish")
    ok, why = C.verdict_neg(tr, 1, None)
    assert ok, why


def test_S5_pointer_snapshot_names_a_rebound_buffer():
    model = make_model()
    with torch.inference_mode():
        cache, st, tok = build(model)
        cb = RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch)
        RC.prepare_step(st, cb, tok, torch.ones(B, dtype=torch.bool))
        RC.decode_step_fixed(model, st, cb, cache)
        before = RC.state_pointers(st, cb, cache)
        assert G.pointer_diff(before, RC.state_pointers(st, cb, cache)) == []
        st.pq = st.pq.clone()                                           # a planted rebinding
        cache.layers[1].cache_engine._cache_lens = cache.layers[1].cache_engine._cache_lens.clone()
        assert G.pointer_diff(before, RC.state_pointers(st, cb, cache)) == ["eng1._cache_lens", "st.pq"]
    with pytest.raises(RDP.R2Error):
        RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch, ng="unknown")


def test_S6_gate_pure_parts():
    base = ["--out", "x", "--workload", "w"]
    for bad in (["--arms", "CF,CE"], ["--arms", "CE,ZZ"], ["--deltas", "0,1"], ["--deltas", "1,0,0,0,0,0,0,0"],
                ["--warmup", "0"], ["--warmup", "192"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    a = G.parse_args(base)
    assert (a.batch, a.steps, a.warmup, a.arms) == (8, 192, 2, list(G.ARMS))
    assert G.prompt_lengths(a) == [16128 - d for d in a.deltas]
    act = torch.tensor([[True, True], [True, False], [True, True], [False, True]])
    assert G.positions_before([10, 20], act).tolist() == [[10, 20], [11, 21], [12, 21], [13, 22]]
    act2 = torch.ones((70, 2), dtype=torch.bool)
    assert G.ng1_first_step([63 * 64 + 60, 100 * 64], act2, 2) == 4          # request 0 reaches 64 * 64 at step 4
    assert G.ng2_first_step(act, 1) == 2 and G.ng2_first_step(act2, 2) is None
    x = torch.arange(12, dtype=torch.float32)
    assert G.digest(x) == G.digest(x.clone()) and G.digest(x) != G.digest(x + 1)
    assert G.pointer_diff({"a": 1, "b": 2}, {"a": 1, "c": 3}) == ["b", "c"]
