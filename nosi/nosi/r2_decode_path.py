"""R2 milestone 2, step 5 (WIRING): the per-request decode path of NOSA, behind a switch (default = shipped).

SWITCH. nosa_llama.Llama.decode_inference takes this path only when the environment sets NOSI_R2_DECODE=1 (read at
import, nosa_llama.R2_DECODE) AND the cache carries an R2State (cache.r2_state, attached by R2State.attach). Otherwise
the shipped decode runs unchanged.

ONE DECODE STEP, per layer, in the shipped order (nosa_llama.py LlamaLayer.decode_forward, :539-663 at 6e6f80a):
  1. rmsnorm, wqkv, nosa_linear, rope: the shipped ops on the B rows.
  2. the positional tables: twin.step(keys, cis) on the layer's PositionalTwin (r2_positional_twin.py, on the GPU in
     the cache dtype): deferred key compression, raw-key ring, total_cis row, cis compression, pos += 1 (the shipped
     update_no_compress_k_decode, update_compress_k_decode, update_uncompressed_cis, update_cis, per request).
  3. stage 1 with per-request key lengths: stage1_seqused(q, comp_k, comp_k, cu_q, twin.cu, 1, Lk_max,
     seqused_k = twin.seqused_k()) on the padded table (stage1_seqused.py).
  4. pooling with per-request widths: nosa_pooling_per_request(score, twin.ccis, n_comp, M_b, ...) with
     M_b = pos_before // 64 + 1 (twin.grid_width() before the step) into buffers of width M_alloc = the shipped
     out_len; then the shipped graph's ops eagerly: topk(QK_SELECT), the scatter mask, masked_fill(+inf), topk(64).
     The selection of every layer is kept in state.topk_log[l] (the G0 gate compares it).
  5. the offloading update through ONE adapter (body_update):
       body "shipped": InfLLMv2Cache.decode_update_kv with kv_bias = twin.total_cis (the lockstep shipped body: valid
                       under equal advance only; refused otherwise);
       body "r2":      the per-request body (r2_decode_body.py of feature/nosi-r2-m2-engine, NOT YET COMMITTED; its
                       interface is docs/R2_STEP3_DECODE_BODY.md: pos BEFORE the step, active, kv_bias =
                       twin.total_cis, f4_fix explicit). R2Body binds it; until the module exists it raises.
  6. attention (flash_attn_nosa_with_kvcache, num_splits = NOSI_ATTN_SPLITS), wo, FFN: the shipped ops.
Final rmsnorm and lm_head as shipped.

PREFILL. The R2 path uses the SHIPPED prefill (batch_prefill: the host K/V, the slot cache, the engine tables), then
R2State.attach BRIDGES each layer's positional tables into a PositionalTwin by copy (bridge_layer): the compressed
keys [0, n), the live raw keys (positions [S - ncl, S)) into the ring rows q % 64, total_cis rows [0, S), the
compressed cis [0, M), the counters (pos = S, n_comp = n, n_ccis = M), then twin.check_invariants(). Equal S for every
request (the shipped prefill is lockstep).

F2 (docs/R2_ENGINE_FINDINGS.md): the shipped compressed cis never grows (an empty-slice copy_); the twin's grows. In
the 64-step window of the G0 gate the extra twin columns are read by the pooling only inside the forced local window
(+inf), so no selection changes. The G0 gate compares the prefix both hold and reports the twin's extra columns as the
EXPECTED F2 difference. There is no f2_fix switch: the twin is the reference and keeps growing.

NEGATIVE CONTROL HOOK (the G0 gate only): state.neg = dict(step=s0, request=b0, short=k) shortens the stage-1 and
pooling key length of request b0 by k from decode step s0 on (pooling_lengths). Default None.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK, RING, STRIDE, PositionalTwin
from .stage1_seqused import stage1_seqused


def _nl():
    """nosa_llama, imported on first use (it imports CUDA extensions; the CPU tests load this module without it)."""
    from . import nosa_llama
    return nosa_llama


class R2Error(RuntimeError):
    pass


# ------------------------------------------------------------------ the bridge (shipped prefill -> twin)
def bridge_layer(twin: PositionalTwin, lay, S: int) -> None:
    """Copy one InfLLMv2CacheLayer's positional state after the shipped prefill of S tokens per request into twin.
    Refuses a state that the twin's own prefill would not produce (n = S // 16 - 1 compressed keys and compressed cis,
    ncl = 16 + S % 16 live raw keys)."""
    B = twin.B
    n = int(lay.cached_compressed_max_seqlen)
    ck = lay.compress_k_cache_varlen
    ncl = int(lay.no_compress_k_len)
    cc = lay.compressed_cis
    if n != S // STRIDE - 1 or tuple(ck.shape[:2]) != (B, n):
        raise R2Error("bridge: compressed keys %s with n %d; the twin's prefill gives (%d, %d)" % (tuple(ck.shape), n, B, S // STRIDE - 1))
    if ncl != STRIDE + S % STRIDE or ncl > RING:
        raise R2Error("bridge: %d live raw keys; expected %d" % (ncl, STRIDE + S % STRIDE))
    if tuple(cc.shape[:2]) != (twin.H, B) or cc.shape[2] != n:
        raise R2Error("bridge: compressed cis %s; expected (%d, %d, %d)" % (tuple(cc.shape), twin.H, B, n))
    if n > twin.lk_max or S > twin.n_max:
        raise R2Error("bridge: Lk_max %d / N_max %d too small for n %d / S %d" % (twin.lk_max, twin.n_max, n, S))
    twin.comp_k[:, :n].copy_(ck)
    q = torch.arange(S - ncl, S, dtype=torch.int64, device=twin.ring.device)
    twin.ring[:, q % RING] = lay.no_compress_k_cache[:, :ncl].to(twin.ring.dtype)
    twin.total_cis[:, :S].copy_(lay.total_cis[:, :S])
    twin.ccis[:, :, :n].copy_(cc)
    twin.pos.fill_(S)
    twin.n_comp.fill_(n)
    twin.n_ccis.fill_(n)
    twin.prefilled.fill_(True)
    twin.epoch.fill_(1)
    twin.check_invariants()


def pooling_lengths(pos_before: torch.Tensor, n_comp: torch.Tensor, step: int, neg=None):
    """(n_used, M_b) int32 contiguous: the key length of stage 1 and pooling (n_comp, shortened by the negative-control
    hook), the pooled width M_b = pos_before // 64 + 1 (the block of the position being written is the last one)."""
    n_used = n_comp.to(torch.int32).clone()
    if neg is not None and step >= int(neg["step"]):
        n_used[int(neg["request"])] -= int(neg["short"])
    return n_used.contiguous(), (pos_before // BLOCK + 1).to(torch.int32).contiguous()


# ------------------------------------------------------------------ the body adapter
class R2Body:
    """Binds the per-request decode body (r2_decode_body.py, feature/nosi-r2-m2-engine). The module is not committed
    at e1178d8; until it is, construction raises R2Error. The call below follows docs/R2_STEP3_DECODE_BODY.md and is
    the ONE place to adapt to the committed signature."""

    def __init__(self, f4_fix: bool):
        try:
            from . import r2_decode_body as RB                       # noqa: F401  (not committed yet)
        except ImportError as e:
            raise R2Error("body 'r2': nosi/nosi/r2_decode_body.py is not in this tree (%s)" % e)
        self.RB = RB
        self.f4_fix = bool(f4_fix)

    def update(self, eng, key, value, kv_bias, topk_idx, pos, active):
        return self.RB.decode_update(eng, key.unsqueeze(1), value.unsqueeze(1), kv_bias, topk_idx, pos=pos, active=active,
                                     f4_fix=self.f4_fix)


# ------------------------------------------------------------------ the state
class R2State:
    """Per-request decode state of one batch: one PositionalTwin per layer (bridged from the shipped prefill), the
    pooling / top-k buffers (width M_alloc), the body adapter, the step counter, the per-layer selection log."""

    def __init__(self, model, cache, S: int, body: str = "shipped", f4_fix=None, lk_extra: int = 64,
                 n_extra: int = 1024, neg=None):
        if body not in ("shipped", "r2"):
            raise R2Error("body must be 'shipped' or 'r2'")
        if body == "r2" and f4_fix is None:
            raise R2Error("body 'r2' needs an explicit f4_fix (True or False)")
        lay0 = model.layers[0]
        eng0 = cache.layers[0].cache_engine
        B = int(cache.layers[0].total_cis.shape[0])
        H, D = lay0.num_key_value_heads, lay0.head_dim
        dev = eng0._k_gpu.device
        dt = cache.layers[0].total_cis.dtype
        self.B, self.H, self.S = B, H, int(S)
        self.lk_max = S // STRIDE + lk_extra // STRIDE + 4
        self.n_max = S + n_extra
        self.twins = []
        for lay in cache.layers:
            t = PositionalTwin(B, H, D, self.lk_max, self.n_max, dtype=dt, device=dev)
            bridge_layer(t, lay, S)
            self.twins.append(t)
        self.M_alloc = (S + 1 + BLOCK - 1) // BLOCK                   # the shipped warm-up out_len (nosa_llama.py:817-818)
        NL = _nl()
        bdt = lay0.wqkv.dtype
        self.pq = torch.empty((H, B, self.M_alloc), dtype=bdt, device=dev)
        self.pc = torch.empty((H, B, self.M_alloc), dtype=bdt, device=dev)
        self.tv_q = torch.empty((H, B, NL.QK_SELECT), dtype=bdt, device=dev)
        self.ti_q = torch.empty((H, B, NL.QK_SELECT), dtype=torch.int64, device=dev)
        self.tv = torch.empty((H, B, lay0.topk_blocks), dtype=bdt, device=dev)
        self.ti = torch.empty((H, B, lay0.topk_blocks), dtype=torch.int64, device=dev)
        self.mask = torch.empty((H, B, self.M_alloc), dtype=torch.bool, device=dev)
        self.cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)
        self.active = torch.ones(B, dtype=torch.bool, device=dev)
        self.body_mode = body
        self.body = R2Body(f4_fix) if body == "r2" else None
        self.f4_fix = f4_fix
        self.neg = neg
        self.steps = 0
        self.topk_log = [None] * len(model.layers)
        self.meta = dict(S=self.S, B=B, lk_max=self.lk_max, n_max=self.n_max, M_alloc=self.M_alloc, body=body,
                         f4_fix=f4_fix, neg=neg,
                         f2="no f2_fix switch: the twin's compressed cis grows (the reference); the shipped table does "
                            "not (findings F2); the gate compares the prefix both hold and reports the twin's extra "
                            "columns as the expected F2 difference")

    @classmethod
    def attach(cls, model, cache, S: int, **kw) -> "R2State":
        st = cls(model, cache, S, **kw)
        cache.r2_state = st
        return st


def body_update(st: R2State, cache, l: int, key, value, twin: PositionalTwin, topk_idx, pos_before):
    if st.body_mode == "shipped":
        if not bool(st.active.all()):
            raise R2Error("body 'shipped' is the lockstep body: every request must advance")
        return cache.decode_update_kv(key, value, twin.total_cis, l, topk_idx)
    return st.body.update(cache.layers[l].cache_engine, key, value, twin.total_cis, topk_idx, pos_before, st.active)


# ------------------------------------------------------------------ the decode
def layer_decode(st: R2State, l: int, layer, hidden, pos_flat, cos_sin_cache, cache):
    NL = _nl()
    B = hidden.shape[0]
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    twin = st.twins[l]
    residual = hidden
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = F.linear(h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
    q = q.view(B, -1)
    k = k.view(B, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
    q = q.reshape(B, layer.num_heads, D)
    k = k.reshape(B, Hkv, D)
    v = v.reshape(B, Hkv, D)
    pos_before = twin.pos.clone()
    twin.step(k, cis.view(B, Hkv), active=st.active)
    n_used, M_b = pooling_lengths(pos_before, twin.n_comp, st.steps, st.neg)
    ck = twin.comp_k_flat()
    score = stage1_seqused(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=n_used)
    nosa_pooling_per_request(score, twin.ccis, n_used, M_b, st.pq, st.pc, stride=layer.pooling_stride,
                             block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
    torch.topk(st.pq, NL.QK_SELECT, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
    st.mask.zero_()
    st.mask.scatter_(2, st.ti_q, True)
    st.pc.masked_fill_(st.mask, float("inf"))
    torch.topk(st.pc, layer.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))
    topk_idx = st.ti
    st.topk_log[l] = topk_idx.clone()
    key_states, value_states, kv_bias, cache_lens = body_update(st, cache, l, k, v, twin, topk_idx, pos_before)
    attn = NL.flash_attn_nosa_with_kvcache(q.unsqueeze(1), key_states, value_states, kv_bias, cache_seqlens=cache_lens,
                                           num_splits=NL._ATTN_SPLITS)
    attn = attn.view(B, 1, -1)
    hidden = residual + F.linear(attn, layer.wo)
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = F.linear(h, layer.gate_up_proj)
    out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
    NL.silu_and_mul(h, out)
    return residual + F.linear(out, layer.down_proj)


@torch.inference_mode()
def decode_inference(model, input_ids, cu_seqlens, position_ids, cache):
    """The R2 twin of Llama.decode_inference: input_ids (B, 1), position_ids (B, 1). Returns logits (B, 1, V) float."""
    NL = _nl()
    st = cache.r2_state
    if input_ids.shape[1] != 1:
        raise R2Error("the R2 decode takes one token per request")
    hidden = F.embedding(input_ids, model.embed_tokens)
    pos_flat = position_ids.flatten()
    for l, layer in enumerate(model.layers):
        hidden = layer_decode(st, l, layer, hidden, pos_flat, model.cos_sin_cache, cache)
    hidden = NL.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    logits = F.linear(hidden, model.lm_head).float()
    st.steps += 1
    return logits
