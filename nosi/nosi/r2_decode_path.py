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
       body "r2":      the per-request body r2_decode_body.DecodeBody (feature/nosi-r2-m2-engine e048b3d; the FINAL
                       interface of docs/R2_STEP3_DECODE_BODY.md at 4b8a658: pos BEFORE the step, active, their
                       host mirrors, kv_bias = twin.total_cis, f4_fix explicit, the shipped kernels). R2Body binds it.
  6. attention (flash_attn_nosa_with_kvcache, num_splits = NOSI_ATTN_SPLITS), wo, FFN: the shipped ops.
Final rmsnorm and lm_head as shipped.

PREFILL, two builders:
  R2State.attach (step 5, equal S): the SHIPPED batch prefill (batch_prefill: the host K/V, the slot cache, the engine
    tables), then each layer's positional tables are BRIDGED into a PositionalTwin by copy (bridge_layer): the
    compressed keys [0, n), the live raw keys (positions [S - ncl, S)) into the ring rows q % 64, total_cis rows
    [0, S), the compressed cis [0, M), the counters, then twin.check_invariants(). Widths: the shipped out_len.
  build_ragged (step 6, a prompt length per request): per request b, ONE B = 1 shipped prefill into a scratch cache
    (its host K / V rows and its total_cis rows), then twin.prefill(b, keys, cis) (the twin's own compressions with the
    shipped ops) and DecodeBody.prefill_request(eng, b, ...) into batch tables allocated by DecodeBody.allocate. The
    WIDTHS are the R2 path's own, PREALLOCATED for the whole run from the horizon (no captured graph, no re-warm):
    M_alloc = (S_max + horizon) // 64 + 1 pooled blocks, Lk_max = (S_max + horizon) // 16 + 1 compressed rows (the
    stage-1 score width round128(Lk_max) is constant: no 128-column boundary is crossed by any launch), N_max =
    n_host = S_max + horizon + 64 cis / host rows. Every kernel launch has the same shape whatever the neighbours'
    lengths, which is what makes a request's arithmetic independent of its neighbours.

F2 (docs/R2_ENGINE_FINDINGS.md): the shipped compressed cis never grows (an empty-slice copy_); the twin's grows. In
the 64-step window of the G0 gate the extra twin columns are read by the pooling only inside the forced local window
(+inf), so no selection changes. The G0 gate compares the prefix both hold and reports the twin's extra columns as the
EXPECTED F2 difference. There is no f2_fix switch: the twin is the reference and keeps growing.

NEGATIVE CONTROL HOOKS (gates only; state.neg, default None; pooling_lengths):
  dict(step=s0, request=b0, short=k)            (kind "short", the G0 gate) request b0's key length - k from step s0;
  dict(kind="shared", request=src, step=s0)     every request uses request src's key length and pooled width from
                                                step s0 (a shared counter: it leaks the neighbour's length);
  dict(kind="frozen")                           every request keeps the pooled width of the first decode step (a width
                                                frozen at warm-up: wrong from the first block boundary on).

ACTIVITY (step 6): the driver sets state.active (device) and state.active_host (CPU) before a step; an inactive
request's row is computed and discarded (the twin and the body skip it); pos_host advances for active requests only.

F3 (findings): the SHIPPED total_cis has S + 1024 rows, so the shipped decode stops at step 1025 (a silent no-op cis
write, then a RuntimeError in the tail-bias copy). The R2 path never reads it: the twin's total_cis has N_max rows and
the body bounds the write by min(n_host, N_max), both sized from the horizon (build_ragged). What still bounds a run:
the preallocated Lk_max / N_max / M_alloc / n_host (TwinCapacityError / IndexError refuse, nothing is written past
them), the RoPE cache (max_length + 1024 positions, 132096 by default) and the int64-cast gather offsets.
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


def pooling_lengths(pos_before: torch.Tensor, n_comp: torch.Tensor, step: int, neg=None, m0=None):
    """(n_used, M_b) int32 contiguous: the key length of stage 1 and pooling (n_comp) and the pooled width
    M_b = pos_before // 64 + 1 (the block of the position being written is the last one), per request; then the
    negative-control hook (module docstring): "short" (default kind), "shared" (request src's values for every request
    from step s0) or "frozen" (m0, the widths of the first step, for every step; m0 None = this is the first step)."""
    n_used = n_comp.to(torch.int32).clone()
    m_b = (pos_before // BLOCK + 1).to(torch.int32)
    kind = None if neg is None else neg.get("kind", "short")
    if kind == "short" and step >= int(neg["step"]):
        n_used[int(neg["request"])] -= int(neg["short"])
    elif kind == "shared" and step >= int(neg.get("step", 0)):
        src = int(neg["request"])
        n_used = n_used[src].expand_as(n_used).clone()
        m_b = m_b[src].expand_as(m_b).clone()
    elif kind == "frozen" and m0 is not None:
        m_b = m0.to(torch.int32).clone()
    elif kind not in (None, "short", "shared", "frozen"):
        raise R2Error("unknown negative-control kind %r" % kind)
    return n_used.contiguous(), m_b.contiguous()


# ------------------------------------------------------------------ the body adapter
class R2Body:
    """Binds the per-request decode body: r2_decode_body.DecodeBody (feature/nosi-r2-m2-engine e048b3d; the FINAL
    interface of docs/R2_STEP3_DECODE_BODY.md at 4b8a658):
        body.decode(eng, key_states (B, 1, H, D), value_states, kv_bias = twin.total_cis, topk_idx (H, B, 64) int64,
                    pos (B,) int64 device = the twin counter BEFORE twin.step, active (B,) bool device, pos_host,
                    active_host (the same on the CPU), diff_offload, gather_k, gather_v_bias)
          -> (_k_gpu, _v_gpu, _kv_bias_gpu, _cache_lens), the shipped tuple.
    The kernels are the shipped body's own: cache_engine.diff.diff_offload, flash_h2d_from_mask,
    flash_h2d_from_mask_bias (cache_engine.py:363, :381, :382). Without the module, construction raises R2Error."""

    def __init__(self, f4_fix: bool):
        try:
            from . import r2_decode_body as RB
        except ImportError as e:
            raise R2Error("body 'r2': nosi/nosi/r2_decode_body.py is not in this tree (%s)" % e)
        from . import cache_engine as CE
        self.body = RB.DecodeBody(bool(f4_fix))
        self.kernels = (CE.diff.diff_offload, CE.flash_h2d_from_mask, CE.flash_h2d_from_mask_bias)

    def update(self, eng, key, value, kv_bias, topk_idx, pos, active, pos_host, active_host):
        return self.body.decode(eng, key.unsqueeze(1), value.unsqueeze(1), kv_bias, topk_idx, pos, active, pos_host,
                                active_host, *self.kernels)


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
        self._common(model, B, H, dev, body, R2Body(f4_fix) if body == "r2" else None, f4_fix, neg,
                     torch.full((B,), int(S), dtype=torch.int64))

    def _common(self, model, B, H, dev, body, adapter, f4_fix, neg, pos_host):
        """The buffers and fields shared by both builders (widths M_alloc set before)."""
        NL = _nl()
        lay0 = model.layers[0]
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
        self.pos_host = pos_host.clone()                                   # host mirrors of the counter (no sync)
        self.active_host = torch.ones(B, dtype=torch.bool)
        self.body_mode = body
        self.body = adapter
        self.f4_fix = f4_fix
        self.neg = neg
        self.steps = 0
        self.M0 = None                                                     # the widths of the first step ("frozen")
        self.topk_log = [None] * len(model.layers)
        self.meta = dict(S=self.S, B=B, lk_max=self.lk_max, n_max=self.n_max, M_alloc=self.M_alloc, body=body,
                         f4_fix=f4_fix, neg=neg,
                         f2="no f2_fix switch: the twin's compressed cis grows (the reference); the shipped table does "
                            "not (findings F2); the gate compares the prefix both hold and reports the twin's extra "
                            "columns as the expected F2 difference")

    def set_active(self, active_host: torch.Tensor) -> None:
        """The activity of the next step: (B,) bool on the CPU (copied to the device; no sync)."""
        a = torch.as_tensor(active_host, dtype=torch.bool)
        if tuple(a.shape) != (self.B,):
            raise R2Error("active must be (%d,)" % self.B)
        self.active_host.copy_(a)
        self.active.copy_(self.active_host)                              # pageable source: the copy is synchronous

    @classmethod
    def attach(cls, model, cache, S: int, **kw) -> "R2State":
        st = cls(model, cache, S, **kw)
        cache.r2_state = st
        return st


def ragged_widths(S_list, horizon: int):
    """The R2 path's own preallocated widths for prompts S_list and `horizon` decode steps (module docstring)."""
    end = max(int(s) for s in S_list) + int(horizon)
    lk_max = end // STRIDE + 1
    return dict(M_alloc=end // BLOCK + 1, lk_max=lk_max, n_max=end + BLOCK, n_host=end + BLOCK,
                score_width=(lk_max + 127) // 128 * 128)


def build_ragged(model, prompts, horizon: int, f4_fix: bool, neg=None, pin: bool = True):
    """A batch with a prompt length per request (module docstring, PREFILL). prompts: B 1-D token tensors (CPU), each
    >= 32 tokens. Returns (cache, state, first tokens (B, 1) int64 on the device). Body 'r2' only (the lockstep
    shipped body cannot hold different lengths)."""
    import gc
    from .cache_engine import InfLLMv2Cache
    S_list = [int(p.numel()) for p in prompts]
    if not S_list or min(S_list) < 32:
        raise R2Error("every prompt needs >= 32 tokens (got %s)" % S_list)
    B = len(S_list)
    w = ragged_widths(S_list, horizon)
    lay0 = model.layers[0]
    H, D = lay0.num_key_value_heads, lay0.head_dim
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    adapter = R2Body(f4_fix)
    for lay in cache.layers:
        adapter.body.allocate(lay.cache_engine, B, w["n_host"], pin)
    eng0 = cache.layers[0].cache_engine
    dev, dt = eng0._k_gpu.device, eng0._k_gpu.dtype
    twins = [PositionalTwin(B, H, D, w["lk_max"], w["n_max"], dtype=dt, device=dev) for _ in cache.layers]
    first = torch.empty((B, 1), dtype=torch.int64, device=dev)
    for b, ids in enumerate(prompts):
        tmp = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
        logits, _ = model.batch_prefill(ids.view(1, -1).to(dev), tmp)
        first[b, 0] = logits[0, -1].argmax()
        Sb = S_list[b]
        for l, lay in enumerate(tmp.layers):
            e = lay.cache_engine
            k = e._k_cpu[0, :Sb].to(dev)
            v = e._v_cpu[0, :Sb].to(dev)
            twins[l].prefill(b, k, lay.total_cis[0, :Sb])
            adapter.body.prefill_request(cache.layers[l].cache_engine, b, k, v, twins[l].total_cis[b, :Sb])
        del tmp, logits
        gc.collect()
    torch.cuda.synchronize()
    st = R2State.__new__(R2State)
    st.B, st.H, st.S = B, H, max(S_list)
    st.lk_max, st.n_max, st.M_alloc = w["lk_max"], w["n_max"], w["M_alloc"]
    st.twins = twins
    st._common(model, B, H, dev, "r2", adapter, f4_fix, neg, torch.tensor(S_list, dtype=torch.int64))
    st.meta.update(S_list=S_list, horizon=int(horizon), builder="build_ragged",
                   widths=dict(w, rule="preallocated from the horizon: no captured graph, no re-warm"))
    cache.r2_state = st
    return cache, st, first


def body_update(st: R2State, cache, l: int, key, value, twin: PositionalTwin, topk_idx, pos_before):
    if st.body_mode == "shipped":
        if not bool(st.active.all()):
            raise R2Error("body 'shipped' is the lockstep body: every request must advance")
        return cache.decode_update_kv(key, value, twin.total_cis, l, topk_idx)
    return st.body.update(cache.layers[l].cache_engine, key, value, twin.total_cis, topk_idx, pos_before, st.active,
                          st.pos_host, st.active_host)


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
    n_used, M_b = pooling_lengths(pos_before, twin.n_comp, st.steps, st.neg, st.M0)
    if st.M0 is None:
        st.M0 = M_b.clone()                                            # the first step's widths ("frozen" control)
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
    st.pos_host += st.active_host.to(torch.int64)
    return logits
