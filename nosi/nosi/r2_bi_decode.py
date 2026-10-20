"""R2 step 7, G0-BI: the BATCH-INVARIANT (BI) SERIAL DECODE, a mode of the R2 decode path
(docs/R2_STEP7_SPEC_ROUND.md sections 3.2-3.5; the reference of the step-7 exactness claim).

ENABLE. attach_bi(state, BIOps(...)) on an R2State with body 'r2' (R2State.attach or build_ragged). Then
r2_decode_path.decode_inference (switch NOSI_R2_DECODE=1, the cache carries the state) runs decode_inference below.
Without attach_bi the R2 path is unchanged.

LAW (AGENTS.md law 1). The float order of every op of a row is a function of that row's own state only, never of the
batch size, the row order or the neighbours. ONE DECODE STEP, per layer, the R2 order (r2_decode_path.layer_decode)
with each op replaced or checked:
  1. rmsnorm (flashinfer, one program per row: microgate B1); wqkv through ops.linear, the GEMM mode being a run
     parameter: EXACT = G-b, r2_spec_core.PaddedLinear (cuBLAS at ONE row count m_fix; the decision of note 3.3);
     FAST = natural cuBLAS (the weaker claim); bigemm = G-a by path (comparison only); nosa_linear (one Triton
     program per row: B1); rope (flashinfer, per row: B1).
  2. twin.step with ops.mean = fixed_order_mean for the two summaries (31 elementwise adds in window order, then / 32):
     torch's .mean may choose its reduction strategy from the tensor shape, that is from B.
  3. stage 1 (stage1_seqused, per-request key lengths, one batch entry per row: B2).
  4. pooling (nosa_pooling_per_request: a max, exact in any order), then the DETERMINISTIC selection
     r2_spec_core.select_blocks (stable sort, value descending, ties to the lower block id, the shipped two-step rule:
     QK_SELECT from the pooled q scores, +inf on them in the pooled cis, top 64): B5.
  5. the body update (r2_decode_body.DecodeBody through R2Body): data movement only.
  6. attention: piece 3's bi_paged_decode (loaded by path; v2 7c826c1 = a fixed 8-page split-KV) on the body's
     GPU window. Per (row, KV head) the 64 slots
     of the selection in ASCENDING BLOCK ID (build_page_table on _block_map, slot 63 named by the row's own block
     p // 64; the update renames it after a full block), the last page (slot 63) cut at last_len = p % 64 + 1: the
     row's own extent, so findings F4 is NOT reproduced (the BI reference is a new declared target, note 3.5).
  7. wo, gate_up, down through ops.linear; silu_and_mul (elementwise); residual adds (elementwise).
Final rmsnorm, lm_head through ops.linear, then .float().

subset_state builds a new cache and R2State that hold some requests of an existing body-'r2' state, in a given order,
by copy (twin rows and counters, host K / V rows, the GPU window, its map and lengths): the gate's batch-invariance
arms (a request alone, the batch reversed) start from bitwise the same per-request state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import r2_decode_path as RP
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK, PositionalTwin
from .r2_spec_core import PaddedLinear, select_blocks
from .stage1_seqused import stage1_seqused


def fixed_order_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """The mean along dim as a chain of elementwise float32 adds in index order, then / n, cast to x's dtype. The
    order is a function of n only (no reduction kernel, so nothing depends on the other dims)."""
    xs = x.unbind(dim)
    acc = xs[0].float()
    for t in xs[1:]:
        acc = acc + t.float()
    return (acc / len(xs)).to(x.dtype)


GEMM_MODES = ("exact", "fast", "bigemm")


class BIOps:
    """The BI ops of one run. attn: a module with bi_paged_decode and build_page_table (piece 3); m_fix: G-b's row
    count (>= every batch of the run). gemm, a RUN PARAMETER (lead, 2026-10-20):
      "exact"   G-b, r2_spec_core.PaddedLinear(m_fix) for wqkv, wo, gate_up, down and lm_head (the exact claim);
      "fast"    natural cuBLAS (F.linear): the WEAKER claim, every non-GEMM op batch-invariant, the GEMM rows depend
                on the row count (the gate reports the difference and classifies it by margin);
      "bigemm"  G-a, linear = the bigemm module's bi_linear (loaded by path; measured slower than G-b at the tick's
                row mix, note 3.3; kept for comparison)."""

    def __init__(self, attn, m_fix: int, gemm: str = "exact", linear=None, linear_name: str = None,
                 mean=fixed_order_mean):
        for name in ("bi_paged_decode", "build_page_table"):
            if not callable(getattr(attn, name, None)):
                raise RP.R2Error("BI attention module has no %s" % name)
        if gemm not in GEMM_MODES:
            raise RP.R2Error("gemm must be one of %s, not %r" % (GEMM_MODES, gemm))
        if (gemm == "bigemm") != (linear is not None):
            raise RP.R2Error("gemm 'bigemm' takes a linear, and only it does")
        self.attn = attn
        self.padded = PaddedLinear(m_fix)
        self.m_fix = self.padded.m_fix
        self.mode = gemm
        self.linear = dict(exact=self.padded, fast=F.linear, bigemm=linear)[gemm]
        self.gemm = dict(exact="EXACT: G-b PaddedLinear(m_fix=%d)" % self.m_fix, fast="FAST: natural cuBLAS F.linear",
                         bigemm="G-a %s" % (linear_name or linear))[gemm]
        self.mean = mean

    def describe(self) -> dict:
        return dict(gemm=self.gemm, mode=self.mode, m_fix=self.m_fix, attn=getattr(self.attn, "__file__", repr(self.attn)),
                    mean="fixed_order_mean" if self.mean is fixed_order_mean else repr(self.mean),
                    selection="r2_spec_core.select_blocks (stable, ties to the lower block id)",
                    f4="not reproduced: the tail page is cut at the row's own extent")


def attach_bi(st, ops: BIOps):
    """Turn an R2State (body 'r2', no negative control) into the BI serial decode."""
    if st.body_mode != "r2":
        raise RP.R2Error("the BI mode needs body 'r2' (the per-request body), not %r" % st.body_mode)
    if st.neg is not None:
        raise RP.R2Error("the BI mode takes no negative-control hook")
    if st.B > ops.m_fix:
        raise RP.R2Error("batch %d > m_fix %d" % (st.B, ops.m_fix))
    st.bi = ops
    st.seq_idx = torch.arange(st.B, dtype=torch.int32, device=st.cu_q.device)
    for t in st.twins:
        t.mean_fn = ops.mean
    st.meta.update(bi=ops.describe())
    return st


def page_table(attn, eng, pos_before: torch.Tensor):
    """(page_table (B, H, topk) int32, num_pages (B, H) int32, last_len (B,) int32) after the body update of the step
    that wrote positions pos_before: slot ids in ascending block id, slot 63 named by its block p // 64."""
    ids = eng._block_map.clone()                                          # (H, B, topk)
    B = ids.shape[1]
    ids[..., eng._tail_block_idx_on_gpu] = (pos_before // BLOCK).view(1, B)
    block_ids = ids.permute(1, 0, 2).contiguous()                         # (B, H, topk)
    slots = torch.arange(ids.shape[-1], device=ids.device).expand_as(block_ids).contiguous()
    table, n = attn.build_page_table(block_ids, slots)
    return table, n, (pos_before % BLOCK + 1).to(torch.int32).contiguous()


def layer_decode_bi(st, l: int, layer, hidden, pos_flat, cos_sin_cache, cache):
    NL = RP._nl()
    ops = st.bi
    lin = ops.linear
    B = hidden.shape[0]
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    twin = st.twins[l]
    residual = hidden
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = lin(h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
    q = q.view(B, -1)
    k = k.view(B, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
    q = q.reshape(B, layer.num_heads, D)
    k = k.reshape(B, Hkv, D)
    v = v.reshape(B, Hkv, D)
    pos_before = twin.pos.clone()
    twin.step(k, cis.view(B, Hkv), active=st.active)
    n_used, M_b = RP.pooling_lengths(pos_before, twin.n_comp, st.steps)
    ck = twin.comp_k_flat()
    score = stage1_seqused(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=n_used)
    nosa_pooling_per_request(score, twin.ccis, n_used, M_b, st.pq, st.pc, stride=layer.pooling_stride,
                             block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
    topk_idx = select_blocks(st.pq.float(), st.pc.float(), NL.QK_SELECT, layer.topk_blocks)   # (H, B, 64) int64
    st.topk_log[l] = topk_idx
    RP.body_update(st, cache, l, k, v, twin, topk_idx, pos_before)
    eng = cache.layers[l].cache_engine
    table, n_pages, last = page_table(ops.attn, eng, pos_before)
    attn = ops.attn.bi_paged_decode(q, eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, st.seq_idx, table, n_pages, last)
    attn = attn.view(B, 1, -1)
    hidden = residual + lin(attn, layer.wo)
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = lin(h, layer.gate_up_proj)
    out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
    NL.silu_and_mul(h, out)
    return residual + lin(out, layer.down_proj)


@torch.inference_mode()
def decode_inference(model, input_ids, cu_seqlens, position_ids, cache):
    """The BI twin of r2_decode_path.decode_inference: input_ids (B, 1), position_ids (B, 1). Returns (B, 1, V) float."""
    NL = RP._nl()
    st = cache.r2_state
    if input_ids.shape[1] != 1:
        raise RP.R2Error("the BI decode takes one token per request")
    hidden = F.embedding(input_ids, model.embed_tokens)
    pos_flat = position_ids.flatten()
    for l, layer in enumerate(model.layers):
        hidden = layer_decode_bi(st, l, layer, hidden, pos_flat, model.cos_sin_cache, cache)
    hidden = NL.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    logits = st.bi.linear(hidden, model.lm_head).float()
    st.steps += 1
    st.pos_host += st.active_host.to(torch.int64)
    return logits


TWIN_FIELDS = ("comp_k", "ring", "total_cis", "pos", "n_comp", "n_ccis", "epoch", "prefilled")
ENGINE_ROWS = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_cache_lens")


def subset_state(model, cache, st, rows, pin: bool = True, new_cache=None, adapter=None):
    """A new cache and R2State (body 'r2', the same f4_fix and widths) holding requests `rows` of (cache, st), in that
    order, by copy. Returns (cache, state); the caller attaches the BI ops (attach_bi) if it wants them.
    new_cache / adapter: the empty cache and the body adapter (defaults: an InfLLMv2Cache and R2Body(f4_fix); the CPU
    tests pass stand-ins)."""
    if st.body_mode != "r2":
        raise RP.R2Error("subset_state copies body-'r2' tables only")
    rows = [int(r) for r in rows]
    if not rows or min(rows) < 0 or max(rows) >= st.B or len(set(rows)) != len(rows):
        raise RP.R2Error("rows %s must be distinct requests of a batch of %d" % (rows, st.B))
    k = len(rows)
    dev = st.cu_q.device
    idx = torch.tensor(rows, dtype=torch.int64, device=dev)
    if new_cache is None:
        from .cache_engine import InfLLMv2Cache
        cfg = model.config
        new_cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    new = new_cache
    adapter = adapter if adapter is not None else RP.R2Body(st.f4_fix)
    twins = []
    for l, (src, dst) in enumerate(zip(cache.layers, new.layers)):
        es, ed = src.cache_engine, dst.cache_engine
        adapter.body.allocate(ed, k, int(es._k_cpu.shape[1]), pin)
        for j, r in enumerate(rows):
            ed._k_cpu[j].copy_(es._k_cpu[r])
            ed._v_cpu[j].copy_(es._v_cpu[r])
        for name in ENGINE_ROWS:
            getattr(ed, name).copy_(getattr(es, name)[idx])
        ed._block_map.copy_(es._block_map[:, idx])
        ed._tail_block_idx_on_gpu = es._tail_block_idx_on_gpu
        t = st.twins[l]
        tn = PositionalTwin(k, t.H, t.D, t.lk_max, t.n_max, dtype=t.ring.dtype, device=t.pos.device)
        for name in TWIN_FIELDS:
            getattr(tn, name).copy_(getattr(t, name)[idx])
        tn.ccis.copy_(t.ccis[:, idx])
        twins.append(tn)
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
    s2 = RP.R2State.__new__(RP.R2State)
    s2.B, s2.H, s2.S = k, st.H, st.S
    s2.lk_max, s2.n_max, s2.M_alloc = st.lk_max, st.n_max, st.M_alloc
    s2.twins = twins
    s2._common(model, k, st.H, dev, "r2", adapter, st.f4_fix, None, st.pos_host[rows].clone())
    s2.steps = st.steps
    s2.meta.update(builder="subset_state", rows=rows, source_B=st.B, source_steps=st.steps,
                   source_builder=st.meta.get("builder", "attach"))
    new.r2_state = s2
    return new, s2
