"""CPU tests of Task 1.6 rung 1 (--graph-step): nosi/nosi/graph_step.py, the --graph-step flag of
benchmarks/Efficiency/r2_base_cell.py, the pure parts of the GPU cell benchmarks/Efficiency/r2_graphstep_cell.py and
its step script. No CUDA, no Triton launch. graph_step.py is loaded through a synthetic package with the GPU libraries
stubbed; every kernel is a CPU FAKE, and the SAME fake serves the shipped step and the graph-safe step. The fakes model
what the step depends on (positions, lengths, LRU stamps, the selection), not the GPU arithmetic: the real equality is
the GPU cell gs1.

GS1 test_flag_off_is_shipped: --graph-step defaults off; r2_base_cell imports graph_step only inside run_graph_step,
    and main reaches run_graph_step only under `if a.graph_step`; run_shipped and the shipped engine files are
    byte-identical to the base commit ded9d41 (skipped outside a git checkout that has it); the summary gains
    meta["graph_step"] only under the flag.
GS2 test_no_host_sync_in_captured_region: the capture lint (r2_capture_lint.lint_function) on the captured region
    (graph_step.CAPTURED_REGION) and on every shipped callee in it: no SYNC (.item / .cpu / .tolist / synchronize /
    bool / float / int of a device value), no SHAPE, no SCALARIDX, no CALLS_EAGER, no BRANCH on a device value; the
    region's ALLOC / HOSTSTATE findings are only the documented ones; the lint's planted positive control.
GS3 test_bucket_table_covers_512_steps: plan_steps from the shipped prefill counters of S 16128 over 512 steps (mb1):
    the re-warm steps equal r2_base_cell.expected_rewarms (the 9 of mb1's P2); the kc / cc / tf steps, positions,
    widths and frozen pooling scalars per shipped bucket; every step >= warmup has a captured key; 95 graphs.
GS4 test_graph_safe_step_equals_shipped: LOCKSTEP, S 6128 and 6136, P 4 and 0, 100 steps (they cross the key and cis
    compressions, two tail fills, two out_len re-warms and the stage-1 width change): the shipped decode
    (nosa_llama.Llama.decode_inference with the harness's re-warm rule; torch.cuda.graph faked by re-running the
    captured with-body) and GraphStep (capture=False) give torch.equal logits at every step and torch.equal state
    (every engine and cache-layer tensor, the compressed keys, the UNSHIFTED pool ages) after every step; after
    export() every host counter equals the shipped one. Preconditions: the pool hit and parked blocks (P 4); FIX 1
    (CPU job gs_fd35db3_a): the two worlds of one seed are bitwise equal before step 0 after
    r2_graphstep_cell.define_undefined (the memory the shipped prefill allocates with torch.empty and does not write
    gets one value; the per-step scratch is skipped only before step 0), and before that fill they differ only in
    tensors that hold such memory.
GS5 test_negative_controls_are_caught: three planted defects, each with its precondition (L34): the LRU stamp frozen
    at capture (no age shift), the raw-key column frozen at capture, a wrong stage-1 length (half the keys). Each must
    produce a logits or state difference of ITS kind, after the step it can first act at (no_shift: after the first
    park; frozen_ncol: from step 1; short_seqused: in the selection). (A stage-1 length frozen at the bucket start is a
    NO-OP plant: the newest compressed keys pool into the forced local window, so only the softmax scale moves; not
    used.)
GS6 test_age_frame_roundtrip: shift_in / shift_out invert each other across steps and keep the age order.
GS7 test_cell_pure_parts: r2_graphstep_cell's argument rules, boundary / steady / special steps, the launch count, the
    arm comparison on planted differences (the state before step 0 too), the re-warm breakdown, the TBD verdicts, the
    probe, and define_undefined writing exactly the unwritten regions.
GS8 test_step_script_cells: r2_graphstep_step.sh (static): its ROOT, cells and python arguments.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_graph_step.py
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
TESTS = EFF / "tests_cpu"
NOSI = ROOT / "nosi/nosi"
FCE = NOSI / "flash_cache_engine"
BASE = "ded9d41"
PKG = "gscpu"
B, HQ, HKV, D, HID, INTER, VOCAB, NLAYERS = 2, 4, 2, 8, 32, 48, 50, 2


def _bc():
    sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


def _lint():
    sys.path.insert(0, str(TESTS))
    import r2_capture_lint as L
    return L


def _cell():
    sys.path.insert(0, str(EFF))
    import r2_graphstep_cell as GC
    return GC


def _git_show(path: str):
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (BASE, path)], capture_output=True)
    return r.stdout if r.returncode == 0 else None


# ------------------------------------------------------------------ CPU fakes (one set for both paths)
class FakeExt:
    """The load() result (diff_offload, pool_update) and the stage-1 extension (varlen_fwd_stage1), on the CPU. Same
    contracts as diff_offload_kernel.cu, pool_update_kernel.cu and the stage-1 op (row b reads its keys from cu_k[b],
    seqused_k[b] of them when given; a softmax over them, summed over the query group)."""

    def __init__(self):
        self.hits = self.parks = self.real_choices = 0

    def diff_offload(self, old, new, new_map, load):
        H, Bn, M = old.shape
        for h in range(H):
            for b in range(Bn):
                o, n = old[h, b], new[h, b]
                oh, nh = torch.isin(o, n), torch.isin(n, o)
                new_map[h, b][oh] = o[oh]
                load[h, b][oh] = -1
                free, want = (~oh).nonzero().flatten(), (~nh).nonzero().flatten()
                k = min(free.numel(), want.numel())
                new_map[h, b][free[:k]] = n[want[:k]]
                load[h, b][free[:k]] = n[want[:k]]

    def pool_update(self, bm, lm, pmap, page, ptgt, pact, stamp, tail):
        H, Bn, M = bm.shape
        P = pmap.shape[2]
        BM, LM, PM, PA = bm.tolist(), lm.tolist(), pmap.tolist(), page.tolist()
        TG = [[[-1] * M for _ in range(Bn)] for _ in range(H)]
        AC = [[[0] * M for _ in range(Bn)] for _ in range(H)]
        for h in range(H):
            for b in range(Bn):
                X, Y, pm, pa = LM[h][b], BM[h][b], PM[h][b], PA[h][b]
                claim, hitq, want = [-1] * P, [-1] * M, [False] * M
                for m in range(M):
                    if m == tail or X[m] < 0:
                        continue
                    q = next((j for j in range(P) if pm[j] == X[m]), -1)
                    if q >= 0 and claim[q] < 0:
                        claim[q], hitq[m] = m, q
                    elif Y[m] >= 0:
                        want[m] = True
                unclaimed = [j for j in range(P) if claim[j] < 0]
                order = sorted(unclaimed, key=lambda j: (pa[j], j))
                npm, npa = list(pm), list(pa)
                rank = 0
                for m in range(M):
                    if m == tail or X[m] < 0:
                        continue
                    q = hitq[m]
                    if q >= 0:
                        self.hits += 1
                        X[m] = -1
                        TG[h][b][m] = q
                        if Y[m] >= 0:
                            AC[h][b][m], npm[q], npa[q] = 1, Y[m], stamp + m
                            self.parks += 1
                        else:
                            AC[h][b][m], npm[q], npa[q] = 2, -1, -P + q
                    elif want[m]:
                        r, rank = rank, rank + 1
                        if r < len(order):
                            q = order[r]
                            TG[h][b][m], AC[h][b][m], npm[q], npa[q] = q, 3, Y[m], stamp + m
                            self.parks += 1
                            if sum(1 for j in unclaimed if pa[j] >= 0) >= 2:
                                self.real_choices += 1
                PM[h][b], PA[h][b] = npm, npa
        lm.copy_(torch.tensor(LM, dtype=lm.dtype))
        pmap.copy_(torch.tensor(PM, dtype=pmap.dtype))
        page.copy_(torch.tensor(PA, dtype=page.dtype))
        ptgt.copy_(torch.tensor(TG, dtype=ptgt.dtype))
        pact.copy_(torch.tensor(AC, dtype=pact.dtype))

    def varlen_fwd_stage1(self, q, k, v, out, cu_q, cu_k, seqused, leftpad, block_table, alibi, max_q, max_k, p_drop,
                          scale, zero, causal, wl, wr, softcap, ret, gen):
        Bn, H = cu_q.numel() - 1, k.shape[1]
        W = (int(max_k) + 127) // 128 * 128
        p = torch.zeros((H, Bn, W), dtype=q.dtype)
        for b in range(Bn):
            q0, q1, k0 = int(cu_q[b]), int(cu_q[b + 1]), int(cu_k[b])
            L = int(seqused[b]) if seqused is not None else int(cu_k[b + 1]) - k0
            s = torch.einsum("ghd,lhd->hgl", q[q0:q1], k[k0:k0 + L]) * scale
            p[:, b, :L] = torch.softmax(s, dim=-1).sum(dim=1)
        return (p, None, None, None)


def f_rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def f_silu_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def f_rope(positions, q, k, head_size, cache, is_neox):
    p = positions.to(q.dtype)[:, None]
    q.add_(p * 1e-3)
    k.add_(p * 2e-3)


def f_nosa_linear(qkv, delta_w, A, q_size, kv_size, kv_heads):
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size:q_size + kv_size].contiguous()
    v = qkv[..., q_size + kv_size:].contiguous()
    cis = (A * F.softplus(v @ delta_w.t())).contiguous()
    return q, k, v, cis


def f_pooling(q_score, cis_score, max_seqlen, q_pooled, cis_pooled, block_size=32, stride=16, padding=1, set_size=5,
              init_blocks=1, local_blocks=16):
    """nosa_pool_kernel (max_pooling_fused.py:7-75): window m reads entries 4m - 1 .. 4m + 3 below max_seqlen; the
    first init_blocks and the last local_blocks + 1 windows are +inf."""
    H, Bn, M = q_pooled.shape
    m = torch.arange(M)
    j = torch.arange(8)
    idx = m[:, None] * (set_size - 1) - padding + j[None, :]
    valid = (idx < max_seqlen) & (idx >= 0) & (j[None, :] < set_size)
    boundary = (m < init_blocks) | (M - m <= local_blocks + 1)
    for src, dst in ((q_score, q_pooled), (cis_score, cis_pooled)):
        g = src[:, :, idx.clamp(0, src.shape[-1] - 1).reshape(-1)].reshape(H, Bn, M, 8)
        acc = g.masked_fill(~valid, float("-inf")).max(dim=-1).values
        dst.copy_(torch.where(boundary, torch.full_like(acc, float("inf")), acc))


def f_attn(q, k_cache, v_cache, kv_bias, cache_seqlens=None, num_splits=0):
    Bn, _, Hq, Dh = q.shape
    G = Hq // k_cache.shape[2]
    out = torch.empty_like(q)
    for b in range(Bn):
        L = int(cache_seqlens[b])
        for h in range(Hq):
            g = h // G
            s = (k_cache[b, :L, g] @ q[b, 0, h]) / math.sqrt(Dh) + kv_bias[b, :L, g]
            out[b, 0, h] = torch.softmax(s, dim=0) @ v_cache[b, :L, g]
    return out


def f_pool_swap(k_gpu, v_gpu, bias, target, action, topk, block_size=64):
    Bn, S_GPU, H, _ = k_gpu.shape
    P = S_GPU // block_size - topk
    act, tgt = action.tolist(), target.tolist()
    for h in range(H):
        for b in range(Bn):
            for m in range(topk):
                a, q = act[h][b][m], tgt[h][b][m]
                if a == 0 or q < 0 or q >= P:
                    continue
                att = slice(m * block_size, (m + 1) * block_size)
                pl = slice((topk + q) * block_size, (topk + q + 1) * block_size)
                for t in (k_gpu, v_gpu, bias):
                    xa, xp = t[b, att, h].clone(), t[b, pl, h].clone()
                    if a in (1, 3):
                        t[b, pl, h] = xa
                    if a in (1, 2):
                        t[b, att, h] = xp


def _h2d(gpu, cpu, load, block_size, bias_buf=None, bias=None):
    Bn, _, H, _ = gpu.shape
    S_CPU = cpu.shape[1]
    ld = load.tolist()
    for h in range(H):
        for b in range(Bn):
            for m, X in enumerate(ld[h][b]):
                if X < 0:
                    continue
                r0 = X * block_size
                n = max(0, min(block_size, S_CPU - r0))
                gpu[b, m * block_size:m * block_size + n, h] = cpu[b, r0:r0 + n, h]
                if bias_buf is not None:
                    bias_buf[b, m * block_size:m * block_size + n, h] = bias[b, r0:r0 + n, h]


def f_h2d(gpu, cpu, load, block_size=64):
    _h2d(gpu, cpu, load, block_size)


def f_h2d_bias(gpu, cpu, bias_buf, bias, load, block_size=64):
    _h2d(gpu, cpu, load, block_size, bias_buf, bias)


_BODY = {}


def _with_body(code):
    """The compiled body of the `with torch.cuda.graph(...)` block of a function's code object."""
    if code not in _BODY:
        tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
                    and "cuda.graph" in ast.unparse(n.items[0].context_expr))
        mod = ast.Module(body=node.body, type_ignores=[])
        ast.fix_missing_locations(mod)
        _BODY[code] = compile(mod, code.co_filename, "exec")
    return _BODY[code]


class FakeGraph:
    """torch.cuda.CUDAGraph on the CPU: replay() re-runs the captured with-body with the locals frozen at capture
    (the frozen scalars and buffer objects a real capture records)."""

    def __init__(self, *a, **k):
        self.body = None

    def replay(self):
        code, glb, loc = self.body
        exec(code, glb, dict(loc))


class FakeGraphCtx:
    def __init__(self, graph, *a, **k):
        self.graph = graph

    def __enter__(self):
        fr = sys._getframe(1)
        self.graph.body = (_with_body(fr.f_code), fr.f_globals, dict(fr.f_locals))
        return self

    def __exit__(self, *exc):
        return False


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _gpu_only(*a, **k):
    raise RuntimeError("a GPU library function was called in a CPU test (stub)")


def _load_package():
    """graph_step.py (and nosa_llama, cache_engine, stage1_seqused through it) as gscpu.*, the GPU libraries stubbed,
    cpp_extension.load returning FakeExt, NOSI_POOL_BLOCKS 4. Cached in sys.modules."""
    if PKG + ".graph_step" in sys.modules:
        return sys.modules[PKG + ".graph_step"]
    pytest.importorskip("transformers")
    import torch.utils.cpp_extension as cpp
    mp = pytest.MonkeyPatch()
    try:
        stubs = {
            "flashinfer": _stub("flashinfer"),
            "flashinfer.norm": _stub("flashinfer.norm", rmsnorm=_gpu_only),
            "flashinfer.activation": _stub("flashinfer.activation", silu_and_mul=_gpu_only),
            "flashinfer.rope": _stub("flashinfer.rope", apply_rope_with_cos_sin_cache_inplace=_gpu_only),
            "infllm_v2": _stub("infllm_v2", infllmv2_attn_stage1=_gpu_only, infllmv2_attn_stage1_fast=_gpu_only,
                               infllmv2_attn_varlen_func=_gpu_only, infllmv2_attn_with_kvcache=_gpu_only,
                               max_pooling_1d=_gpu_only, max_pooling_1d_varlen=_gpu_only),
            "flash_attn": _stub("flash_attn", flash_attn_with_kvcache=_gpu_only),
            "flash_attn_nosa": _stub("flash_attn_nosa", flash_attn_with_kvcache=_gpu_only),
        }
        try:
            import triton                      # noqa: F401  (a real import works on a CPU node; launches never happen)
            import triton.language             # noqa: F401
        except Exception:
            tr, tl = _stub("triton"), _stub("triton.language")
            tr.jit = lambda fn=None, **k: fn if fn is not None else (lambda f: f)
            tr.cdiv = lambda a, b: -(-a // b)
            tr.language = tl
            tl.constexpr = object
            stubs.update({"triton": tr, "triton.language": tl})
        for name, mod in stubs.items():
            mp.setitem(sys.modules, name, mod)
        mp.setattr(cpp, "load", lambda *a, **k: FakeExt())
        mp.setenv("NOSI_POOL_BLOCKS", "4")
        mp.setenv("NOSI_ATTN_SPLITS", "4")
        for v in ("NOSI_AVAIL", "NOSI_TRANSFER_TRACE", "NOSI_R2_DECODE"):
            mp.delenv(v, raising=False)
        if PKG not in sys.modules:
            pkg = types.ModuleType(PKG)
            pkg.__path__ = [str(NOSI)]
            sys.modules[PKG] = pkg
        spec = importlib.util.spec_from_file_location(PKG + ".graph_step", str(NOSI / "graph_step.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PKG + ".graph_step"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[PKG + ".graph_step"]
            raise
    finally:
        mp.undo()
    return mod


@pytest.fixture
def W(monkeypatch):
    """The loaded modules with every kernel replaced by its CPU fake (undone after the test)."""
    GS = _load_package()
    NL, CE = GS.NL, GS.CE
    ext = FakeExt()

    def shipped_stage1(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False):
        return GS.S1.stage1_seqused(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None,
                                    causal=causal, ext=ext)

    for name, fn in (("rmsnorm", f_rmsnorm), ("silu_and_mul", f_silu_and_mul),
                     ("apply_rope_with_cos_sin_cache_inplace", f_rope), ("nosa_linear", f_nosa_linear),
                     ("nosa_pooling", f_pooling), ("flash_attn_nosa_with_kvcache", f_attn),
                     ("infllmv2_attn_stage1_fast", shipped_stage1)):
        monkeypatch.setattr(NL, name, fn)
    monkeypatch.setattr(NL, "nvtx", types.SimpleNamespace(range_push=lambda *a, **k: None,
                                                          range_pop=lambda *a, **k: None))
    monkeypatch.setattr(GS, "_STAGE1_EXT", ext)
    monkeypatch.setattr(CE, "diff", ext)
    monkeypatch.setattr(CE, "_pool", ext)
    monkeypatch.setattr(CE, "_flash_pool_swap", f_pool_swap)
    monkeypatch.setattr(CE, "flash_h2d_from_mask", f_h2d)
    monkeypatch.setattr(CE, "flash_h2d_from_mask_bias", f_h2d_bias)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", FakeGraphCtx)
    return types.SimpleNamespace(GS=GS, NL=NL, CE=CE, ext=ext, mp=monkeypatch)


def _cache_class(CE):
    class FakeCache:
        """InfLLMv2Cache's own delegating methods on a plain object (no transformers DynamicCache constructor)."""

        def __init__(self, layers):
            self.layers = layers
            self._seen_tokens = 0
    for n in ("prefill_update_kv", "decode_update_kv", "update_cis", "update_uncompressed_cis",
              "update_compress_k_prefill", "update_compress_k_decode", "update_no_compress_k_prefill",
              "update_no_compress_k_decode", "get_seq_length"):
        setattr(FakeCache, n, CE.InfLLMv2Cache.__dict__[n])
    return FakeCache


def build_world(W, S, steps, seed):
    """A NOSA-like model (B 2, 4 query heads, 2 KV heads, head 8, 2 layers, vocabulary 50) and its cache after the
    SHIPPED prefill updates (nosa_llama.prefill_forward's order, the cache-layer methods themselves)."""
    NL, CE = W.NL, W.CE
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.5):
        return torch.randn(shape, generator=g) * scale
    with torch.inference_mode():
        model = NL.Llama.__new__(NL.Llama)
        model.embed_tokens, model.lm_head = rnd(VOCAB, HID), rnd(VOCAB, HID)
        model.norm_weight, model.norm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
        model.cos_sin_cache = torch.zeros(1)
        model.has_buffers, model.block_size, model.num_key_value_heads = False, 64, HKV
        model.hidden_size, model.vocab_size, model.dtype, model.device = HID, VOCAB, torch.float32, "cpu"
        model.layers = []
        for i in range(NLAYERS):
            L = NL.LlamaLayer(i, topk_blocks=64, topk_q_blocks=16, init_blocks=1, window_blocks=16, block_size=64,
                              pooling_stride=16, pooling_block_size=32)
            L.wqkv, L.wo = rnd(HQ * D + 2 * HKV * D, HID), rnd(HID, HQ * D)
            L.gate_up_proj, L.down_proj = rnd(2 * INTER, HID), rnd(HID, INTER)
            L.delta, L.A = types.SimpleNamespace(weight=rnd(HKV, HKV * D)), rnd(HKV)
            L.q_size, L.kv_size = HQ * D, HKV * D
            L.input_layernorm_weight, L.input_layernorm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
            L.post_attention_layernorm_weight, L.post_attention_layernorm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
            L.num_heads, L.num_key_value_heads, L.head_dim = HQ, HKV, D
            model.layers.append(L)
        model.num_layers = NLAYERS
        cfg = types.SimpleNamespace(num_hidden_layers=NLAYERS, num_key_value_heads=HKV, head_dim=D)
        layers = [CE.InfLLMv2CacheLayer(cfg, True) for _ in range(NLAYERS)]
        for L in layers:
            L.cache_engine.device = torch.device("cpu")
            L.cache_engine.dtype = torch.float32            # the model dtype here (bf16 on the GPU, both sides)
            L.cache_engine.max_gen_len = steps + 64
        cache = _cache_class(CE)(layers)
        cu = torch.arange(0, B * S + 1, S, dtype=torch.int32)
        for l in range(NLAYERS):
            keys, vals, cis = rnd(B, S, HKV, D), rnd(B, S, HKV, D), rnd(B, S, HKV, scale=0.2)
            ucis = cache.update_uncompressed_cis(cis, l, 0, B)
            cache.update_cis(cis.permute(2, 0, 1), l, 0, B)
            cache.prefill_update_kv(keys, vals, ucis, l, 0, B)
            ck, cuc = NL.compress_k(keys.reshape(B * S, HKV, D), cu, HKV, D)
            m0 = int((cuc[1:] - cuc[:-1]).max())
            cache.update_compress_k_prefill(ck.contiguous(), l, cuc, max_seqlen=m0, current_batch_pos=0, total_bsz=B)
            cache.update_no_compress_k_prefill(keys[:, m0 * 16:], l, 32, 16, current_batch_pos=0, total_bsz=B)
        tok = torch.randint(0, VOCAB, (B, 1), generator=g)
        pos = torch.full((B, 1), S, dtype=torch.int32)
    return model, cache, tok, pos


ENG = ("_block_map", "_new_block_map_buf", "_load_mask", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_cache_lens", "_k_cpu",
       "_v_cpu")
POOLT = ("_pool_map", "_pool_target", "_pool_action")
LAY = ("no_compress_k_cache", "total_cis", "tail_cis", "compressed_cis", "cached_compressed_cu_seqlens")


def diff_state(GS, c_sh, gs, t):
    """The state entries that differ after step t between the shipped cache and the graph-step cache."""
    bad = []
    for l, (a, b) in enumerate(zip(c_sh.layers, gs.layers)):
        ea, eb = a.cache_engine, b.cache_engine
        for n in ENG + (POOLT if ea.pool_blocks > 0 else ()):
            if not torch.equal(getattr(ea, n), getattr(eb, n)):
                bad.append("eng%d.%s" % (l, n))
        if ea.pool_blocks > 0:
            un = GS.shift_out(eb._pool_age, gs.host0["stamp0"] + GS.TOPK * t)
            if not torch.equal(ea._pool_age, un):
                bad.append("eng%d._pool_age" % l)
        for n in LAY:
            if not torch.equal(getattr(a, n), getattr(b, n)):
                bad.append("layer%d.%s" % (l, n))
        m = gs.rows[t]["m"]
        if not torch.equal(a.compress_k_cache_varlen, gs.ck[l][:, :m]):
            bad.append("layer%d.compressed_keys" % l)
    return bad


HOSTC = ("seq_length", "no_compress_k_len", "cached_compressed_max_seqlen", "cis_len", "tail_cis_len")
ENGC = ("seq_length", "_tail_block_len_on_gpu", "_pool_stamp_base", "_tail_block_idx_on_gpu")


def diff_worlds(c_a, c_b, skip=()):
    """The entries that differ between two caches in the SHIPPED layout (before any GraphStep): every engine and
    cache-layer state tensor (whole tensors, bitwise), the compressed keys and the host counters; `skip` = engine
    tensor names left out."""
    bad = []
    for l, (a, b) in enumerate(zip(c_a.layers, c_b.layers)):
        ea, eb = a.cache_engine, b.cache_engine
        for n in ENG + ((POOLT + ("_pool_age",)) if ea.pool_blocks > 0 else ()):
            if n not in skip and not torch.equal(getattr(ea, n), getattr(eb, n)):
                bad.append("eng%d.%s" % (l, n))
        for n in LAY + ("compress_k_cache_varlen",):
            if not torch.equal(getattr(a, n), getattr(b, n)):
                bad.append("layer%d.%s" % (l, n))
        bad += ["layer%d.%s" % (l, n) for n in HOSTC if getattr(a, n) != getattr(b, n)]
        bad += ["eng%d.%s" % (l, n) for n in ENGC if getattr(ea, n) != getattr(eb, n)]
    return bad


def lockstep(W, S, steps, pool, plant=None, seed=0):
    """The shipped decode (r2_base_cell.run_shipped's loop) and GraphStep side by side; the first logits / state
    difference (None if none). PRECONDITION (FIX 1, CPU job gs_fd35db3_a): the two worlds of one seed are bitwise equal
    before step 0, after r2_graphstep_cell.define_undefined gives one value to the memory the shipped prefill allocates
    with torch.empty and does not write; the per-step scratch (SCRATCH) is skipped only here. ROOT-CAUSE CHECK: before
    that fill, the worlds may differ ONLY in tensors that hold such memory (leftover bytes), never elsewhere."""
    GS, BC, GC = W.GS, _bc(), _cell()
    W.mp.setattr(W.CE, "POOL_BLOCKS", pool)
    m_sh, c_sh, tok, pos = build_world(W, S, steps, seed)
    m_tw, c_tw, tok_t, pos_t = build_world(W, S, steps, seed)
    raw = diff_worlds(c_sh, c_tw)
    holders = set(GC.UNDEFINED_ENGINE) | set(GC.UNDEFINED_LAYER) | set(GC.SCRATCH)
    assert all(n.split(".", 1)[1] in holders for n in raw), "worlds differ outside the undefined memory: %s" % raw
    with torch.inference_mode():
        GC.define_undefined(c_sh)
        GC.define_undefined(c_tw)
    pre = diff_worlds(c_sh, c_tw, skip=GC.SCRATCH)
    assert not pre, "PRECONDITION: the two worlds of seed %d differ before step 0: %s" % (seed, pre)
    assert torch.equal(tok, tok_t) and torch.equal(pos, pos_t)
    for n in ("embed_tokens", "lm_head", "norm_weight"):
        assert torch.equal(getattr(m_sh, n), getattr(m_tw, n)), n
    for la, lb in zip(m_sh.layers, m_tw.layers):
        for n in ("wqkv", "wo", "gate_up_proj", "down_proj", "A"):
            assert torch.equal(getattr(la, n), getattr(lb, n)), n
    cu = torch.arange(B + 1, dtype=torch.int32)
    gs = GS.GraphStep(m_tw, c_tw, cu.clone(), steps, warmup=steps, capture=False, pos_dtype=pos.dtype,
                      shift_ages=(plant != "no_shift"))
    if plant == "frozen_ncol":
        with torch.inference_mode():
            gs.sched[:, 0] = gs.sched[0, 0]
    if plant == "short_seqused":
        orig = GS.S1.stage1_seqused
        half = gs.host0["m0"] // 2

        def short(*a, seqused_k=None, **k):
            if seqused_k is not None:
                seqused_k = torch.full_like(seqused_k, half)
            return orig(*a, seqused_k=seqused_k, **k)
        W.mp.setattr(GS.S1, "stage1_seqused", short)
    out = dict(first_logit=None, first_state=None, rewarms=[], gs=gs, c_sh=c_sh, c_tw=c_tw, host0=dict(gs.host0),
               parks_before_first=None, first_park_step=None, raw_before_fill=raw)
    warm = None
    tok_s = tok
    for t in range(steps):
        l0 = c_sh.layers[0]
        need, why = BC.rewarm_need(c_sh.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
        if need:
            warm = BC.warm_state(c_sh.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            out["rewarms"].append((t, why))
            m_sh.has_buffers = False
        lg_s = m_sh.decode_inference(tok_s, cu, pos, c_sh)
        lg_t = gs.step(t, tok_t, pos_t)
        if out["first_park_step"] is None and W.ext.parks > 0:
            out["first_park_step"] = t
        if out["first_logit"] is None and not torch.equal(lg_s, lg_t):
            out["first_logit"] = t
        if out["first_state"] is None:
            bad = diff_state(GS, c_sh, gs, t)
            if bad:
                out["first_state"] = (t, bad)
        if out["parks_before_first"] is None and (out["first_logit"] is not None or out["first_state"] is not None):
            out["parks_before_first"] = W.ext.parks
        if plant is not None and (out["first_logit"] is not None or out["first_state"] is not None):
            break
        tok_s = lg_s[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_t = lg_t[:, -1, :].argmax(dim=-1, keepdim=True)
        pos, pos_t = pos + 1, pos_t + 1
    return out


# ------------------------------------------------------------------ GS1
def test_GS1_flag_off_is_shipped():
    BC, L = _bc(), _lint()
    base = ["--out", "x", "--workload", "w", "--batch", "216", "--arms", "SH", "--pool", "65"]
    assert BC.parse_args(base).graph_step is False
    assert BC.parse_args(base + ["--graph-step"]).graph_step is True
    src = (EFF / "r2_base_cell.py").read_text()
    tree = ast.parse(src)
    top = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [s for s in top if "graph_step" in s], top
    importers = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(
        isinstance(x, ast.ImportFrom) and any(a.name == "graph_step" for a in x.names) for x in ast.walk(n)))
    assert importers == ["run_graph_step"], importers
    main = L.fn_from_source(src, "main")

    def calls(nodes, name):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
                   for n in nodes for c in ast.walk(n))
    gated = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "a.graph_step"]
    assert any(calls(n.body, "run_graph_step") and calls(n.orelse, "run_shipped") for n in gated)
    assert sum(1 for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == "run_graph_step") == 1
    meta_sets = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "meta['graph_step']"]
    assert len(meta_sets) == 1 and any(meta_sets[0] in n.body for n in gated)
    old = _git_show("benchmarks/Efficiency/r2_base_cell.py")
    if old is None:
        pytest.skip("the base commit %s is not reachable from this checkout" % BASE)
    old = old.decode()
    for fn in ("run_shipped", "run_r2", "rewarm_need", "warm_state", "period_stats", "rates", "free"):
        assert ast.dump(L.fn_from_source(src, fn)) == ast.dump(L.fn_from_source(old, fn)), fn
    for p in ("nosi/nosi/nosa_llama.py", "nosi/nosi/cache_engine.py", "nosi/nosi/max_pooling_fused.py",
              "nosi/nosi/nosa_linear.py", "nosi/nosi/stage1_seqused.py", "nosi/nosi/__init__.py",
              "nosi/nosi/flash_cache_engine/pool_update.cpp", "nosi/nosi/flash_cache_engine/pool_update_kernel.cu",
              "nosi/nosi/flash_cache_engine/diff_offload_kernel.cu", "nosi/nosi/flash_cache_engine/flash_pool_swap.py",
              "nosi/nosi/flash_cache_engine/flash_h2d_mask.py", "nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py"):
        assert (ROOT / p).read_bytes() == _git_show(p), p
    for p in ("nosa_llama.py", "cache_engine.py", "__init__.py"):
        assert "graph_step" not in (NOSI / p).read_text(), p


# ------------------------------------------------------------------ GS2
ALLOWED = [("ALLOC", r"^torch\.empty\(hs\.shape"), ("ALLOC", r"^nck\.clone\(\)$"),
           ("HOSTSTATE", r"^eng\._cache_lens\[\.\.\.\] \+= 1$"), ("HOSTSTATE", r"^eng\._block_map\[\.\.\., tail\] \+= 1$"),
           ("HOSTSTATE", r"^Lc\.cached_compressed_cu_seqlens \+= Lc\.cached_compressed_cu_seqlens_adder$")]
CALLEES = (("nosa_llama.py", "layer_norm"), ("nosa_linear.py", "nosa_linear"), ("max_pooling_fused.py", "nosa_pooling"),
           ("stage1_seqused.py", "stage1_seqused"), ("stage1_seqused.py", "check_seqused"), ("stage1_seqused.py", "_ext"),
           ("flash_cache_engine/flash_pool_swap.py", "flash_pool_swap"),
           ("flash_cache_engine/flash_h2d_mask.py", "flash_h2d_from_mask"),
           ("flash_cache_engine/flash_h2d_mask_bias.py", "flash_h2d_from_mask_bias"))
FA_NOSA = ROOT / "dependencies/flash-attention-nosa/flash_attn_nosa/flash_attn_interface.py"


def test_GS2_no_host_sync_in_captured_region():
    L = _lint()
    src = (NOSI / "graph_step.py").read_text()
    tree = ast.parse(src)
    region = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                   and getattr(n.targets[0], "id", None) == "CAPTURED_REGION"))
    assert set(region) == {"GraphStep._body", "GraphStep._layer", "GraphStep._engine"}
    for q in region:
        fn = L.fn_from_source(src, q)
        found = L.lint_function(fn)
        hard = [f for f in found if f[0] in ("SYNC", "SHAPE", "SCALARIDX", "CALLS_EAGER", "BRANCH")]
        assert not hard, (q, hard)
        assert not L.unexplained(found, ALLOWED), (q, L.unexplained(found, ALLOWED))
        text = ast.unparse(fn)
        for tok in (".item(", ".cpu(", ".tolist(", "synchronize", ".numpy(", "print(", "time."):
            assert tok not in text, (q, tok)
    # the region calls only these engine kernels and shipped helpers (a new callee must be added to the lint list)
    body = "".join(ast.unparse(L.fn_from_source(src, q)) for q in region)
    called = set(re.findall(r"\b(NL|CE|S1)\.([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)\(", body))
    assert called == {("NL", "layer_norm"), ("NL", "nosa_linear"), ("NL", "apply_rope_with_cos_sin_cache_inplace"),
                      ("NL", "nosa_pooling"), ("NL", "flash_attn_nosa_with_kvcache"), ("NL", "silu_and_mul"),
                      ("S1", "stage1_seqused"), ("CE", "diff.diff_offload"), ("CE", "_pool.pool_update"),
                      ("CE", "_flash_pool_swap"), ("CE", "flash_h2d_from_mask"), ("CE", "flash_h2d_from_mask_bias")}, called
    for rel, name in CALLEES:
        found = L.lint_function(L.fn_from_source((NOSI / rel).read_text(), name))
        bad = [f for f in found if f[0] in ("SYNC", "SHAPE", "SCALARIDX", "CALLS_EAGER", "BRANCH")]
        assert not bad, (rel, name, bad)
    if FA_NOSA.exists():
        found = L.lint_function(L.fn_from_source(FA_NOSA.read_text(), "flash_attn_with_kvcache"))
        assert not [f for f in found if f[0] in ("SYNC", "SHAPE", "SCALARIDX", "CALLS_EAGER", "BRANCH")], found
    planted = L.lint_function(ast.parse(L.PLANTED).body[0])
    assert {c for c, _, _ in planted} == set(L.CLASSES)


# ------------------------------------------------------------------ GS3
def test_GS3_bucket_table_covers_512_steps(W):
    GS, BC = W.GS, _bc()
    c = GS.prefill_counters(BC.S0)
    assert c == dict(S=16128, n0=16, m0=1007, tail0=0, tl0=16, cis0=16128, stamp0=0)
    rows, end = GS.plan_steps(steps=512, **c)
    assert len(rows) == 512
    exp = BC.expected_rewarms(BC.S0, 512)
    assert [t for t, _ in exp] == [0, 64, 128, 192, 256, 288, 320, 384, 448]          # mb1's P2 (2190876)
    assert [r["t"] for r in rows if r["rewarm"]] == [t for t, _ in exp]
    assert [r["t"] for r in rows if r["kc"]] == [t for t in range(1, 512) if t % 16 == 0]
    assert [r["t"] for r in rows if r["cc"]] == [t for t in range(512) if t % 16 == 15]
    assert [r["t"] for r in rows if r["tf"]] == [t for t in range(512) if t % 64 == 63]
    for r in rows:
        t = r["t"]
        assert r["cis_col"] == r["bias_col"] == 16128 + t
        assert r["tail_pos"] == 63 * 64 + t % 64
        assert (r["n_col"] == 16) if r["kc"] else (16 <= r["n_col"] <= 31)
        assert r["tl_col"] == 16 + t % 16
        assert r["m"] == 1007 + t // 16
        assert r["width"] == GS.round_up(r["m"], 128)
        assert r["wb_row"] == (16128 + t + 1 - 64 if r["tf"] else None)
        assert r["stamp"] == 64 * t
    starts = [t for t, _ in exp] + [512]
    for a, b in zip(starts, starts[1:]):
        seg = rows[a:b]
        assert {r["out_len"] for r in seg} == {-(-(16128 + a + 1) // 64)}
        assert {r["m_frozen"] for r in seg} == {rows[a]["m"]}
        assert {r["width"] for r in seg} == {GS.round_up(rows[a]["m"], 128)}
    assert rows[288]["width"] == 1152 and rows[287]["width"] == 1024
    plan = GS.capture_plan(rows, 2)
    assert len(plan) == len(set(plan)) == 95                       # m-segment 0: plain, cc; 31 others: kc, plain, cc
    assert all(r["key"] in set(plan) for r in rows[2:])
    assert sum(1 for k in plan if k.tf) == 8
    assert end == dict(seq=16640, n=32, m=1038, tail=0, tl=16, cis=16640, stamp=32768)
    with pytest.raises(ValueError):
        GS.plan_steps(S=16128, n0=16, m0=1007, tail0=65, tl0=16, cis0=16128, steps=1)   # the shipped tail assert


# ------------------------------------------------------------------ GS4
@pytest.mark.parametrize("pool", [4, 0])
@pytest.mark.parametrize("S", [6128, 6136])
def test_GS4_graph_safe_step_equals_shipped(W, S, pool):
    GS = W.GS
    steps = 100
    res = lockstep(W, S, steps, pool)
    gs = res["gs"]
    assert res["host0"] == GS.prefill_counters(S)
    rows = gs.rows
    whys = {w for _, w in res["rewarms"]}
    assert any(r["kc"] for r in rows) and any(r["cc"] for r in rows) and sum(r["tf"] for r in rows) == 2
    assert {"first", "out_len", "width"} <= whys, res["rewarms"]
    assert [t for t, _ in res["rewarms"]] == [r["t"] for r in rows if r["rewarm"]]
    if pool:
        assert W.ext.hits > 0 and W.ext.parks > 0, (W.ext.hits, W.ext.parks)
    assert res["first_logit"] is None, "logits differ first at step %s" % res["first_logit"]
    assert res["first_state"] is None, "state differs first at %s" % (res["first_state"],)
    end = gs.export()
    for a, b in zip(res["c_sh"].layers, res["c_tw"].layers):
        for n in ("seq_length", "no_compress_k_len", "cached_compressed_max_seqlen", "cis_len", "tail_cis_len"):
            assert getattr(a, n) == getattr(b, n), n
        for n in ("seq_length", "_tail_block_len_on_gpu", "_pool_stamp_base"):
            assert getattr(a.cache_engine, n) == getattr(b.cache_engine, n), n
        assert torch.equal(a.compress_k_cache_varlen, b.compress_k_cache_varlen)
        if pool:
            assert torch.equal(a.cache_engine._pool_age, b.cache_engine._pool_age)
    assert res["c_sh"]._seen_tokens == res["c_tw"]._seen_tokens
    assert end["seq"] == S + steps


# ------------------------------------------------------------------ GS5
@pytest.mark.parametrize("plant", ["no_shift", "frozen_ncol", "short_seqused"])
def test_GS5_negative_controls_are_caught(W, plant):
    S, steps = 6128, 100
    rows, _ = W.GS.plan_steps(steps=steps, **W.GS.prefill_counters(S))
    if plant == "frozen_ncol":                                    # precondition: the column moves
        assert len({r["n_col"] for r in rows if not r["kc"]}) >= 2
    if plant == "short_seqused":                                  # precondition: the dropped keys (from m0 // 2 on)
        r0 = rows[0]                                              # pool into blocks OUTSIDE the forced local window
        assert r0["m"] // 2 // 4 + 17 < r0["out_len"] - 17, r0    # (the last 17 blocks; 4 keys per block)
    res = lockstep(W, S, steps, 4, plant=plant)     # its precondition: the worlds are equal before step 0 (FIX 1)
    caught = res["first_logit"] is not None or res["first_state"] is not None
    assert caught, "planted %s was not caught in %d steps" % (plant, steps)
    if plant == "no_shift":                                       # precondition: a block was parked before the catch
        assert res["parks_before_first"] and res["parks_before_first"] > 0
        assert res["first_state"] is not None, res                 # the pool is invisible to the logits
        assert any("_pool_age" in n or "_pool_map" in n for n in res["first_state"][1]), res["first_state"]
        # the plant acts only on an age written at an EARLIER step: the first difference follows the first park
        assert res["first_park_step"] is not None and res["first_state"][0] > res["first_park_step"], res
    if plant == "frozen_ncol":
        assert res["first_state"] and any("no_compress_k_cache" in n for n in res["first_state"][1])
        assert res["first_state"][0] >= 1, res["first_state"]      # row 0 of sched is unchanged by the plant
    if plant == "short_seqused":                                  # it acts through the selection only
        sel = ("_block_map", "_new_block_map_buf", "_load_mask")
        assert res["first_logit"] is not None or any(n.split(".", 1)[1] in sel for n in res["first_state"][1]), res


# ------------------------------------------------------------------ GS6
def test_GS6_age_frame_roundtrip(W):
    GS = W.GS
    P = 4
    ages = torch.tensor([[-4, -3, 70, 5], [-1, 130, 64, -2]], dtype=torch.int64)
    first = 128                                                    # the shipped stamp of the first graph step
    a = GS.shift_in(ages, first)
    assert torch.equal(a[ages < 0], ages[ages < 0])                 # empties untouched
    assert torch.equal(torch.argsort(a.flatten()), torch.argsort(ages.flatten()))
    x = a.clone()
    for k in range(5):                                             # k + 1 graph steps, each shifts first
        x = x - GS.TOPK * (x >= 0).long()
        assert bool((x[ages >= 0] >= 0).all())
        assert torch.equal(GS.shift_out(x, first + GS.TOPK * k), ages)
    assert GS.STAMP_K > 64 * 10 ** 6 and P < GS.TOPK


# ------------------------------------------------------------------ GS7
def test_GS7_cell_pure_parts():
    GC = _cell()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = GC.parse_args(base)
    assert (a.steps, a.warmup, a.pool, a.order, a.prof_start) == (512, 2, 65, "sh,gs", 100)
    for bad in (["--prof-start", "2"], ["--prof-start", "509"], ["--order", "gs"], ["--pool", "66"]):
        with pytest.raises(SystemExit):
            GC.parse_args(base + bad)
    ks = GC.boundary_steps(512)
    assert ks == sorted(set(ks)) and {0, 63, 64, 287, 288, 511} <= set(ks)
    assert GC.boundary_steps(320)[-1] == 319 and max(GC.boundary_steps(320)) < 320
    st = GC.steady_steps("eager", 512, 2, 100, [(0, "first"), (64, "out_len")])
    assert 0 not in st and 64 not in st and 99 not in st and 102 not in st and 103 in st and 1 in st
    st = GC.steady_steps("graph", 512, 2, 100, [])
    assert {0, 1, 2, 99, 100, 101, 102}.isdisjoint(st) and 3 in st and len(st) == 512 - 7
    sp = GC.special_steps(16128, 512)
    assert sp["tf"] == list(range(63, 512, 64)) and sp["kc"] == list(range(16, 512, 16))
    ev = [dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel"), dict(ph="X", cat="cuda_driver", name="cuLaunchKernelEx"),
          dict(ph="X", cat="cuda_runtime", name="cudaGraphLaunch"), dict(ph="X", cat="cuda_runtime", name="cudaMemcpyAsync"),
          dict(ph="X", cat="cuda_runtime", name="cudaStreamSynchronize"), dict(ph="X", cat="kernel", name="gemm"),
          dict(ph="i", cat="cuda_runtime", name="cudaLaunchKernel")]
    assert GC.count_launches(ev) == {"cudaLaunchKernel": 1, "cuLaunchKernelEx": 1, "cudaGraphLaunch": 1,
                                     "cudaMemcpyAsync": 1}

    def arm(seed):
        g = torch.Generator().manual_seed(seed)
        return dict(prefill_digest=torch.randint(0, 9, (3, 16), generator=g),
                    digests=torch.randint(0, 9, (5, 3, 16), generator=g), tokens=torch.randint(0, 9, (3, 6), generator=g),
                    rows=torch.randn((2, 2, 7), generator=g), keep_steps=[0, 4], state={"a": 1, "b": 2},
                    pre_state={"a": 5, "c": 6})
    x, y = arm(0), arm(0)
    c = GC.compare_arms(x, y)
    assert c["prefill_equal"] and not c["digest_mismatches"] and not c["token_mismatches"]
    assert not c["rows_unequal_steps"] and not c["state_mismatches"] and c["pre_state_mismatches"] == []
    y["digests"][3, 1, 7] += 1
    y["tokens"][2, 4] += 1
    y["rows"][1, 0, 0] += 1.0
    y["state"]["b"] = 3
    y["pre_state"]["c"] = 7
    y["prefill_digest"][0, 0] += 1
    c = GC.compare_arms(x, y)
    assert c["digest_mismatches"] == 1 and c["first_digest_mismatch"] == [3, 1] and c["steps_with_mismatch"] == [3]
    assert c["token_mismatches"] == 1 and c["first_token_mismatch"] == (2, 3)
    assert c["rows_unequal_steps"] == [4] and c["state_mismatches"] == ["b"] and not c["prefill_equal"]
    assert c["pre_state_mismatches"] == ["c"]
    del x["pre_state"]
    assert GC.compare_arms(x, y)["pre_state_mismatches"] is None
    # FIX 1: define_undefined writes exactly the unwritten regions (a fake engine with the shipped field names)
    e = types.SimpleNamespace(_tail_block_idx_on_gpu=3, _tail_block_len_on_gpu=5, block_size=8, seq_length=10,
                              _k_gpu=torch.zeros(2, 40, 2, 3), _v_gpu=torch.zeros(2, 40, 2, 3), _kv_bias_gpu=torch.zeros(2, 40, 2),
                              _k_cpu=torch.zeros(2, 16, 2, 3), _v_cpu=torch.zeros(2, 16, 2, 3))
    L = types.SimpleNamespace(cache_engine=e, no_compress_k_cache=torch.zeros(2, 32, 2, 3), no_compress_k_len=20,
                              tail_cis=torch.zeros(2, 2, 32), tail_cis_len=17, total_cis=torch.zeros(2, 30, 2), cis_len=10)
    GC.define_undefined(types.SimpleNamespace(layers=[L]), value=9.0)
    for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu):
        assert bool((t[:, :24] == 9).all()) and bool((t[:, 24:29] == 0).all()) and bool((t[:, 29:32] == 9).all())
        assert bool((t[:, 32:] == 0).all())                        # the pool rows (zeroed by the prefill) untouched
    for t in (e._k_cpu, e._v_cpu):
        assert bool((t[:, :10] == 0).all()) and bool((t[:, 10:] == 9).all())
    assert bool((L.no_compress_k_cache[:, :20] == 0).all()) and bool((L.no_compress_k_cache[:, 20:] == 9).all())
    assert bool((L.tail_cis[:, :, :17] == 0).all()) and bool((L.tail_cis[:, :, 17:] == 9).all())
    assert bool((L.total_cis[:, :10] == 0).all()) and bool((L.total_cis[:, 10:] == 9).all())
    assert GC.SCRATCH == ("_new_block_map_buf", "_load_mask") and GC.UNDEFINED_FILL == 4096.0
    br = GC.rewarm_breakdown([100.0, 6000.0], [(1, "out_len")], {1: dict(gc_ms=3000.0, empty_cache_ms=1500.0,
                                                                         sync_ms=500.0, n_gc=32)})
    assert br[0]["other_ms"] == 1000.0 and br[0]["hostcall_share"] == round(5000 / 6000, 4) and br[0]["n_gc"] == 32
    v = {p[0]: p[1] for p in GC.gs_verdicts(dict(steady_ratio=0.99, effective_ratio=None), GC.PRED_GS)}
    assert v["GS-P1"] == "UNREGISTERED" and v["GS-P2"] == "NOT_RUN"
    v = {p[0]: p[1] for p in GC.gs_verdicts(dict(steady_ratio=0.99, capture_s=50.0),
                                            dict(GC.PRED_GS, steady_ratio=(0.95, 1.0), capture_s=(1, 30)))}
    assert v["GS-P1"] == "CONFIRMED" and v["GS-P4"] == "REFUTED"
    assert all(v is None for v in GC.PRED_GS.values())              # TBD: the lead registers the values
    import gc as gcmod
    pr = GC.Probe()
    orig = gcmod.collect
    pr.install()
    try:
        assert gcmod.collect is not orig
        pr.t = 7
        gcmod.collect()
        pr.t = None
        gcmod.collect()
    finally:
        pr.remove()
    assert gcmod.collect is orig and pr.rows[7]["n_gc"] == 1 and set(pr.rows) == {7}


# ------------------------------------------------------------------ GS8
def test_GS8_step_script_cells():
    txt = (EFF / "r2_graphstep_step.sh").read_text()
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-graphstep " in txt
    assert "OUTROOT=/mnt/beegfs/ojerbi/r2-runs/gstep " in txt
    assert "SELF=benchmarks/Efficiency/r2_graphstep_step.sh" in txt
    for cell in ("gs0) BATCH=16 STEPS=320", "gs1) BATCH=216 ", "gs1r) BATCH=216 ORDER=gs,sh", "mb1g) BATCH=216 "):
        assert cell in txt, cell
    assert '--order "$ORDER"' in txt and "--graph-step)" in txt and "NOSI_POOL_BLOCKS=$POOL" in txt
    for f in ("nosi/nosi/graph_step.py", "benchmarks/Efficiency/r2_graphstep_cell.py",
              "benchmarks/Efficiency/r2_base_cell.py", "nosi/nosi/stage1_seqused.py"):
        assert f in txt, f
    old = (EFF / "r2_base_fused_step.sh").read_text()
    assert "nosi-r2-mbase" in old and "nosi-r2-graphstep" not in old        # the copied script is untouched
