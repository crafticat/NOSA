"""R2 milestone 2: the PER-REQUEST speculative round on the model (the pilot's NOSI_SPEC_COMMIT=perreq).

The lockstep round (nosa_llama.draft_forward / spec_verify_forward / spec_rollback) scores with the shipped layer tables
(one counter per table for the batch) and the captured pooling graph, and commits c = 1 + min_b n_acc(b). Here every
request has its own counter, so the round needs per-request TABLES as well as the per-request tail and store of
spec_loop.py (section PER-REQUEST ROUNDS):

  TABLES     one PositionalTwin per layer (r2_positional_twin.py), BRIDGED by copy from the shipped layer tables after
             the shipped warm-up step (bridge_layer_at; every request is at the same length there). The engines move
             to per-request counters at the same moment (spec_loop.to_perreq). The shipped layer tables are not read
             again in the cell.
  SCORING    score_position, per layer and position, for every request: twin.step (the shipped four table updates,
             per request), stage 1 with per-request key lengths (stage1_seqused, seqused_k = n_comp), the per-request
             pooling (nosa_pooling_per_request, n_comp and the width M_b = pos // 64 + 1 per request), then the ops of
             the shipped after-pooling graph EAGERLY (topk QK_SELECT, the scatter mask, masked_fill +inf, topk). The
             buffers are allocated once at the bridge for the whole cell (M_alloc covers the horizon), so no re-warm.
             This mirrors steps 2-4 of r2_decode_path.layer_decode (nosi-r2-m2-wiring, uncommitted at 4839445); when
             that module is committed, score_position should become a call into it.
  ENGINE     cache_engine.spec_draft_update_perreq / spec_round_update_perreq / spec_commit_perreq (the tail, the store
             plans, the gathers; a stream whose union overflows is isolated, its request commits nothing: ctx.round_bad).
  COMMIT     spec_rollback: per layer, the tables commit ctr[keep_b, :, b] and the tail is truncated to keep_b
             (spec_loop.commit_layer_perreq). ctr is the tables' counter trajectory recorded during the verify scoring.
  PROBE      ProbeSnapshot / capture / replay: the rollback probe in per-request form (the post-round state equals a
             fresh replay of keep_b tokens per request from the round-start snapshot: the twin's own advance with
             n = keep and, per request, the LOCKSTEP tail_write.write_tail of keep_b tokens on a B = 1 view).

Equivalence with the shipped scoring (by reading; GPU-confirmed only by the identity gate): the pooled widths equal the
shipped re-warmed out_len; compressed keys after the warm-up are read here and masked in the shipped graph (its frozen
max_seq_len), but they fall in blocks of the forced local window (+inf in both: docs/R2_ENGINE_FINDINGS.md, F1 related
limits); torch.topk(sorted=False) over a wider buffer can return the same index set in another order, which places
blocks in other slots and changes the float summation order of attention (a near-tie effect, not a selection change).

CPU: everything here except the forwards runs on CPU (tests pass stand-ins for the two kernels). nosa_llama (flashinfer,
the CUDA extensions) is imported only inside the forwards.
"""
from __future__ import annotations

import types
from typing import List, Optional

import torch
import torch.nn.functional as F

from . import spec_loop as _sl
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK, KERNEL, RING, STRIDE, PositionalTwin
from .stage1_seqused import stage1_seqused
from .verify import tail_write as _tw


class PerReqModelError(RuntimeError):
    pass


def _nl():
    from . import nosa_llama as NL       # GPU modules: imported only by the forwards
    return NL


# ------------------------------------------------------------------ the bridge (shipped layer tables -> twin)
def complete_windows(pos: int) -> int:
    """Windows of 32 at stride 16 that end at or before position pos - 1: the compressed cis a table holds after pos
    tokens (the shipped prefill: S // 16 - 1)."""
    return 0 if pos < KERNEL else (pos - KERNEL) // STRIDE + 1


def bridge_layer_at(twin: PositionalTwin, lay, pos: int) -> None:
    """Copy one shipped layer's positional tables after pos tokens per request (the prefill plus any shipped decode
    steps, lockstep) into ``twin``: the compressed keys [0, n), the live raw keys (positions [16 n, pos), the shipped
    no_compress_k_cache rows [0, ncl) in position order) into the ring rows q % 64, total_cis rows [0, pos), the
    compressed cis [0, M), the counters. Refuses (PerReqModelError, nothing copied) a layer whose state the twin would
    not hold after pos tokens: cis_len != pos, 16 n + ncl != pos, or M != the complete windows of pos (the shipped
    decode drops every compressed cis it computes, findings F2: a step where a cis window completed leaves the
    shipped table one column short). Then twin.check_invariants()."""
    B = twin.B
    n = int(lay.cached_compressed_max_seqlen)
    ck = lay.compress_k_cache_varlen
    ncl = int(lay.no_compress_k_len)
    cc = lay.compressed_cis
    M = int(lay.comp_cis_len)
    if int(lay.cis_len) != pos:
        raise PerReqModelError("bridge: the layer's cis_len %d is not pos %d" % (int(lay.cis_len), pos))
    if tuple(ck.shape[:2]) != (B, n):
        raise PerReqModelError("bridge: compressed keys %s, expected (%d, %d, ...)" % (tuple(ck.shape), B, n))
    if not (0 < ncl <= KERNEL) or STRIDE * n + ncl != pos:
        raise PerReqModelError("bridge: %d live raw keys and %d compressed keys do not make %d positions" % (ncl, n, pos))
    if tuple(cc.shape) != (twin.H, B, M):
        raise PerReqModelError("bridge: compressed cis %s, expected (%d, %d, %d)" % (tuple(cc.shape), twin.H, B, M))
    if M != complete_windows(pos):
        raise PerReqModelError("bridge: the shipped compressed cis has %d columns, %d windows are complete at %d "
                               "(findings F2: the shipped decode dropped a summary)" % (M, complete_windows(pos), pos))
    if n > twin.lk_max or M > twin.lk_max or pos > twin.n_max:
        raise PerReqModelError("bridge: Lk_max %d / N_max %d too small for n %d, M %d, pos %d" % (twin.lk_max, twin.n_max, n, M, pos))
    twin.comp_k[:, :n].copy_(ck)
    q = torch.arange(pos - ncl, pos, dtype=torch.int64, device=twin.ring.device)
    twin.ring[:, q % RING] = lay.no_compress_k_cache[:, :ncl].to(device=twin.ring.device, dtype=twin.ring.dtype)
    twin.total_cis[:, :pos].copy_(lay.total_cis[:, :pos])
    twin.ccis[:, :, :M].copy_(cc)
    twin.pos.fill_(pos)
    twin.n_comp.fill_(n)
    twin.n_ccis.fill_(M)
    twin.prefilled.fill_(True)
    twin.epoch.fill_(1)
    twin.check_invariants()


# ------------------------------------------------------------------ the state of one per-request cell
class PerReqState:
    """The per-request tables and scoring buffers of one batch: one twin per layer, the pooled / top-k buffers of width
    M_alloc, cu_q; per layer the verify's counter trajectory (ctrs) and per-position inputs (round_inputs, kept for the
    probe as probe_inputs); ``view``: the duck-typed cache of spec_loop.PerReqSpecContext (cache_engine, tables)."""

    def __init__(self, twins: List[PositionalTwin], engines, qk_select: int, topk_blocks: int, M_alloc: int,
                 score_dtype, device):
        if not twins or len(twins) != len(engines):
            raise PerReqModelError("one twin per layer engine")
        B, H = twins[0].B, twins[0].H
        self.B, self.H, self.L = B, H, len(twins)
        self.twins = twins
        self.qk_select, self.topk_blocks, self.M_alloc = int(qk_select), int(topk_blocks), int(M_alloc)
        self.pq = torch.empty((H, B, M_alloc), dtype=score_dtype, device=device)
        self.pc = torch.empty((H, B, M_alloc), dtype=score_dtype, device=device)
        self.tv_q = torch.empty((H, B, qk_select), dtype=score_dtype, device=device)
        self.ti_q = torch.empty((H, B, qk_select), dtype=torch.int64, device=device)
        self.tv = torch.empty((H, B, topk_blocks), dtype=score_dtype, device=device)
        self.ti = torch.empty((H, B, topk_blocks), dtype=torch.int64, device=device)
        self.mask = torch.empty((H, B, M_alloc), dtype=torch.bool, device=device)
        self.cu_q = torch.arange(B + 1, dtype=torch.int32, device=device)
        self.ctrs: list = [None] * self.L
        self.round_inputs: list = [None] * self.L
        self.probe_inputs: list = [None] * self.L
        self.view = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e, tables=t) for e, t in zip(engines, twins)])

    @classmethod
    def bridge(cls, model, cache, horizon: int) -> "PerReqState":
        """After the shipped warm-up step: one twin per layer bridged from the shipped tables (bridge_layer_at), sized
        for ``horizon`` more tokens per request (the cell's tokens plus one round of overshoot), and the buffers."""
        NL = _nl()
        lay0 = cache.layers[0]
        pos = int(lay0.cis_len)
        B = int(lay0.total_cis.shape[0])
        H, D = model.layers[0].num_key_value_heads, model.layers[0].head_dim
        dev, dt = lay0.total_cis.device, lay0.total_cis.dtype
        n_max = pos + int(horizon)
        lk_max = n_max // STRIDE + 2
        twins = []
        for lay in cache.layers:
            if int(lay.cis_len) != pos:
                raise PerReqModelError("bridge: the layers are at different lengths")
            t = PositionalTwin(B, H, D, lk_max, n_max, dtype=dt, device=dev)
            bridge_layer_at(t, lay, pos)
            twins.append(t)
        M_alloc = (n_max + BLOCK - 1) // BLOCK + 1
        return cls(twins, [lay.cache_engine for lay in cache.layers], NL.QK_SELECT, model.layers[0].topk_blocks, M_alloc,
                   model.layers[0].wqkv.dtype, dev)


# ------------------------------------------------------------------ scoring
def score_position(st: PerReqState, l: int, q, k, cis, pooling_stride: int, block_size: int, window_blocks: int,
                   init_blocks: int, stage1=None, pool=None) -> torch.Tensor:
    """One position of every request on layer l. q (B, Hq, D), k (B, Hkv, D), cis (B, Hkv). The tables step first
    (twin.step: the shipped update order), then stage 1 over the padded compressed keys with seqused_k = n_comp, the
    per-request pooling with n_comp and M_b = pos_before // 64 + 1, and the after-pooling graph's ops eagerly. Returns
    the selection (H, B, topk) int64 (a clone: the buffers are reused). ``stage1`` / ``pool`` default to the kernels
    (stage1_seqused, nosa_pooling_per_request); CPU tests pass stand-ins with the same signatures."""
    twin = st.twins[l]
    pos_before = twin.pos.clone()
    twin.step(k, cis)
    n_used = twin.n_comp.to(torch.int32).contiguous()
    # M_b <= M_alloc without a check here (a host read per position): twin.step refuses a position >= n_max, and
    # M_alloc = ceil(n_max / 64) + 1 (PerReqState.bridge)
    M_b = (pos_before // BLOCK + 1).to(torch.int32).contiguous()
    ck = twin.comp_k_flat()
    score = (stage1 or stage1_seqused)(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=n_used)
    (pool or nosa_pooling_per_request)(score, twin.ccis, n_used, M_b, st.pq, st.pc, stride=pooling_stride,
                                       block_size=block_size, local_blocks=window_blocks, init_blocks=init_blocks)
    torch.topk(st.pq, st.qk_select, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
    st.mask.zero_()
    st.mask.scatter_(2, st.ti_q, True)
    st.pc.masked_fill_(st.mask, float("inf"))
    torch.topk(st.pc, st.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))
    return st.ti.clone()


# ------------------------------------------------------------------ the forwards (GPU)
def _ffn(NL, layer, hidden):
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = F.linear(h, layer.gate_up_proj)
    out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
    NL.silu_and_mul(h, out)
    return residual + F.linear(out, layer.down_proj)


def _qkv(NL, layer, hidden, position_ids, cos_sin_cache):
    bsz, n, _ = hidden.size()
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = F.linear(h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, layer.num_key_value_heads)
    q = q.view(bsz * n, -1)
    k = k.view(bsz * n, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), q, k, layer.head_dim, cos_sin_cache, True)
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    return (q.reshape(bsz, n, layer.num_heads, D), k.reshape(bsz, n, Hkv, D), v.reshape(bsz, n, Hkv, D),
            cis.reshape(bsz, n, Hkv))


def draft_layer(layer, st: PerReqState, l: int, hidden, position_ids, cos_sin_cache, cache, ctx):
    """LlamaLayer.draft_forward per request: the scoring of score_position, the engine's spec_draft_update_perreq,
    the shipped decode kernel over the whole allocation with the draft's masked bias, wo, the FFN."""
    NL = _nl()
    _vt = NL._vtr.TRACE
    if _vt is not None: _vt.begin_layer(l)
    residual = hidden
    bsz = hidden.shape[0]
    q, k, v, cis = _qkv(NL, layer, hidden, position_ids, cos_sin_cache)
    if _vt is not None: _vt.rec("score_begin")
    sel = score_position(st, l, q[:, 0], k[:, 0], cis[:, 0], layer.pooling_stride, layer.block_size,
                         layer.window_blocks, layer.init_blocks)
    if _vt is not None: _vt.rec("fetch_begin")
    eng = cache.layers[l].cache_engine
    k_all, v_all, bias, lens = eng.spec_draft_update_perreq(k, v, st.twins[l].total_cis, sel, ctx, l)
    if _vt is not None: _vt.rec("fetch_end")
    attn = NL.flash_attn_nosa_with_kvcache(q, k_all, v_all, bias, cache_seqlens=lens, num_splits=NL._ATTN_SPLITS)
    if _vt is not None: _vt.rec("attn_end")
    hidden = residual + F.linear(attn.view(bsz, 1, -1), layer.wo)
    return _ffn(NL, layer, hidden)


def verify_layer(layer, st: PerReqState, l: int, hidden, position_ids, cos_sin_cache, cache, ctx):
    """LlamaLayer.spec_verify_forward per request: per position u the scoring of score_position (the tables' counter
    trajectory ctr[u] = counters after u positions is kept in st.ctrs[l], the per-position key / value / cis in
    st.round_inputs[l]), the engine's spec_round_update_perreq, the Path-1 verify with seqused_k per request, wo, FFN."""
    NL = _nl()
    _vt = NL._vtr.TRACE
    if _vt is not None: _vt.begin_layer(l)
    residual = hidden
    bsz, U, _ = hidden.size()
    q, k, v, cis = _qkv(NL, layer, hidden, position_ids, cos_sin_cache)
    if _vt is not None: _vt.rec("score_begin")
    twin = st.twins[l]
    ctr = [twin.counters()]
    sels = []
    for u in range(U):
        sels.append(score_position(st, l, q[:, u].contiguous(), k[:, u], cis[:, u], layer.pooling_stride,
                                   layer.block_size, layer.window_blocks, layer.init_blocks))
        ctr.append(twin.counters())
    st.ctrs[l] = torch.stack(ctr)
    st.round_inputs[l] = (k, v, cis)
    sel = torch.stack(sels, dim=0)
    if _vt is not None: _vt.rec("fetch_begin")
    eng = cache.layers[l].cache_engine
    k_all, v_all, b_all, rnd = eng.spec_round_update_perreq(k, v, twin.total_cis, sel, ctx, l)
    if _vt is not None: _vt.rec("fetch_end")
    layer._verify_last_round = rnd
    attn = NL.verify_attention(q.reshape(bsz * U, layer.num_heads, layer.head_dim), k_all, v_all, b_all, rnd,
                               mark=(_vt.rec if _vt is not None else None))
    if _vt is not None: _vt.rec("attn_end")
    hidden = residual + F.linear(attn.view(bsz, U, -1), layer.wo)
    return _ffn(NL, layer, hidden)


def _logits(NL, model, hidden):
    hidden = NL.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    return F.linear(hidden, model.lm_head).float()


@torch.inference_mode()
def draft_inference(model, input_ids, cu_seqlens, position_ids, cache, ctx, st: PerReqState):
    """One DRAFT step (B, 1) -> fp32 logits (B, 1, V), every layer per request."""
    NL = _nl()
    if input_ids.shape[1] != 1:
        raise PerReqModelError("one token per request per draft step")
    _vt = NL._vtr.TRACE
    if _vt is not None: _vt.begin_call()
    hidden = F.embedding(input_ids, model.embed_tokens)
    for l in range(model.num_layers):
        hidden = draft_layer(model.layers[l], st, l, hidden, position_ids, model.cos_sin_cache, cache, ctx)
    logits = _logits(NL, model, hidden)
    if _vt is not None: _vt.end_call()
    return logits


@torch.inference_mode()
def spec_verify_inference(model, input_ids, cu_seqlens, position_ids, cache, ctx, st: PerReqState):
    """One VERIFY round (B, U) -> fp32 logits (B, U, V), every layer per request."""
    NL = _nl()
    _vt = NL._vtr.TRACE
    if _vt is not None: _vt.begin_call()
    hidden = F.embedding(input_ids, model.embed_tokens)
    for l in range(model.num_layers):
        hidden = verify_layer(model.layers[l], st, l, hidden, position_ids, model.cos_sin_cache, cache, ctx)
    logits = _logits(NL, model, hidden)
    if _vt is not None: _vt.end_call()
    return logits


@torch.inference_mode()
def spec_rollback(cache, ctx, st: PerReqState, keep: torch.Tensor):
    """After a verify round: every layer commits keep (B,) (cache_engine.spec_commit_perreq: the tables commit
    ctr[keep_b, :, b], the tail is truncated to keep_b). The verify's inputs move to st.probe_inputs (the probe's
    replay). Returns the layer-0 PerReqTruncate."""
    first = None
    mirror_lo = ctx.store.lay.mirror_lo
    for l, lay in enumerate(cache.layers):
        if st.ctrs[l] is None:
            raise PerReqModelError("spec_rollback: layer %d has no verify trajectory" % l)
        tr = lay.cache_engine.spec_commit_perreq(st.twins[l], st.ctrs[l], keep, mirror_lo)
        if first is None:
            first = tr
        st.probe_inputs[l] = st.round_inputs[l]
        st.round_inputs[l] = None
        st.ctrs[l] = None
    return first


# ------------------------------------------------------------------ the rollback probe, per request
TWIN_TENSORS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis")


class ProbeSnapshot:
    """The round-start state that a per-request round changes and its probe replays: per layer the engine's counter,
    block map and cache_lens, the rows of the tail slot and of the two mirror slots, a host window around every
    request's position, and every tensor of the layer's twin. NOT the store and NOT the window / round-region slots:
    the replay does not redo gathers, so the post-round store stays (it matches the post-round slots)."""

    def __init__(self, cache, st: PerReqState, layout, U: int):
        self.cache, self.st, self.lay, self.U = cache, st, layout, int(U)
        self.slots = []
        self.taken = False

    @torch.inference_mode()
    def take(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()          # the pinned host window (state_snapshot._sync_host_window)
        self.slots = []
        for l, lay in enumerate(self.cache.layers):
            eng = lay.cache_engine
            bs = int(eng.block_size)
            pos = eng._pos
            lo = max(0, int(pos.min()) - 2 * bs)
            hi = min(int(eng._k_cpu.shape[1]), int(pos.max()) + self.U + 2 * bs)
            ts, ms = self.lay.tail_slot * bs, self.lay.mirror_lo * bs
            d = dict(_pos=pos.clone(), _block_map=eng._block_map.clone(), _cache_lens=eng._cache_lens.clone(),
                     host=(lo, hi), k_cpu=eng._k_cpu[:, lo:hi].clone(), v_cpu=eng._v_cpu[:, lo:hi].clone())
            for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
                t = getattr(eng, name)
                d["tail" + name] = t[:, ts:ts + bs].clone()
                d["mirror" + name] = t[:, ms:ms + 2 * bs].clone()
            d["twin"] = {n: getattr(self.st.twins[l], n).clone() for n in TWIN_TENSORS}
            self.slots.append(d)
        self.taken = True
        return self

    @torch.inference_mode()
    def restore(self):
        if not self.taken:
            raise RuntimeError("ProbeSnapshot.restore before take")
        for l, lay in enumerate(self.cache.layers):
            eng = lay.cache_engine
            d = self.slots[l]
            bs = int(eng.block_size)
            ts, ms = self.lay.tail_slot * bs, self.lay.mirror_lo * bs
            eng._pos.copy_(d["_pos"])
            eng._block_map.copy_(d["_block_map"])
            eng._cache_lens.copy_(d["_cache_lens"])
            lo, hi = d["host"]
            eng._k_cpu[:, lo:hi].copy_(d["k_cpu"])
            eng._v_cpu[:, lo:hi].copy_(d["v_cpu"])
            for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
                t = getattr(eng, name)
                t[:, ts:ts + bs].copy_(d["tail" + name])
                t[:, ms:ms + 2 * bs].copy_(d["mirror" + name])
            for n in TWIN_TENSORS:
                getattr(self.st.twins[l], n).copy_(d["twin"][n])
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        return self


def capture(cache, st: PerReqState, pos0: torch.Tensor, layout) -> list:
    """What a committed per-request state must be, per layer and request: the engine's counter, tail length, block map
    column, cache_lens, the tail slot's rows below the tail length, the HOST rows of the blocks the round completed and
    kept ([64 (pos0 // 64), 64 (pos // 64)): the only host rows a later gather reads); the twin's counters (pos,
    n_comp, n_ccis) and live rows (compressed keys, compressed cis, total_cis, the live raw keys). Not compared:
    mirrors, rows at or above a counter, the host rows of the live tail block (never read: the tail is served from
    slot topk-1; an undone fill leaves them written, truncate_tail's docstring), the epoch, the store."""
    out = []
    for l, lay in enumerate(cache.layers):
        eng, tw = lay.cache_engine, st.twins[l]
        bs = int(eng.block_size)
        ts = layout.tail_slot * bs
        per = []
        for b in range(int(eng._pos.shape[0])):
            p, p0 = int(eng._pos[b]), int(pos0[b])
            tl = p % bs
            lo, hi = (p0 // bs) * bs, (p // bs) * bs
            n, m = int(tw.n_comp[b]), int(tw.n_ccis[b])
            d = dict(pos=p, tail_len=tl, map=eng._block_map[:, b].clone(), cache_lens=int(eng._cache_lens[b]),
                     k_tail=eng._k_gpu[b, ts:ts + tl].clone(), v_tail=eng._v_gpu[b, ts:ts + tl].clone(),
                     b_tail=eng._kv_bias_gpu[b, ts:ts + tl].clone(),
                     k_host=eng._k_cpu[b, lo:hi].clone(), v_host=eng._v_cpu[b, lo:hi].clone(),
                     t_pos=int(tw.pos[b]), t_n=n, t_m=m, comp_k=tw.comp_k[b, :n].clone(), ccis=tw.ccis[:, b, :m].clone(),
                     total_cis=tw.total_cis[b, :int(tw.pos[b])].clone(), live=tw.live_keys(b).clone())
            per.append(d)
        out.append(per)
    return out


def states_equal(a: list, b: list):
    """(True, "equal") or (False, the first difference)."""
    for l, (x, y) in enumerate(zip(a, b)):
        for r, (u, w) in enumerate(zip(x, y)):
            for k in u:
                p, q = u[k], w[k]
                if torch.is_tensor(p):
                    if p.shape != q.shape or not torch.equal(p, q):
                        return False, "layer %d request %d %s differs" % (l, r, k)
                elif p != q:
                    return False, "layer %d request %d %s %r vs %r" % (l, r, k, p, q)
    return True, "equal"


def _b1_view(eng, b: int, pos_b: int):
    """Request b of a per-request engine as a B = 1 LOCKSTEP engine (views: writes go through): the lockstep
    tail_write.write_tail runs on it."""
    v = types.SimpleNamespace(block_size=eng.block_size, topk=eng.topk, _tail_block_idx_on_gpu=eng._tail_block_idx_on_gpu)
    v._k_gpu, v._v_gpu, v._kv_bias_gpu = eng._k_gpu[b:b + 1], eng._v_gpu[b:b + 1], eng._kv_bias_gpu[b:b + 1]
    v._k_cpu, v._v_cpu = eng._k_cpu[b:b + 1], eng._v_cpu[b:b + 1]
    v._block_map = eng._block_map[:, b:b + 1]
    v._cache_lens = eng._cache_lens[b:b + 1]
    v.seq_length = int(pos_b)
    v._tail_block_len_on_gpu = int(pos_b) % int(eng.block_size)
    return v


@torch.inference_mode()
def replay(cache, st: PerReqState, snap: ProbeSnapshot, keep, layout, bias_rows=None):
    """Restore the round-start snapshot, then commit keep[b] tokens of the round per request WITHOUT the round's code:
    per layer the twin's own advance(keys, cis, n = keep), and per request with keep_b > 0 the LOCKSTEP
    tail_write.write_tail of its keep_b tokens on a B = 1 view (its counters, map column and cache_lens go through the
    views); the engine counter is pos0 + keep. Inputs: st.probe_inputs (the verify's per-position key / value / cis)."""
    snap.restore()
    keep_h = [int(x) for x in torch.as_tensor(keep).tolist()]
    for l, lay in enumerate(cache.layers):
        eng, tw = lay.cache_engine, st.twins[l]
        inp = st.probe_inputs[l]
        if inp is None:
            raise PerReqModelError("replay: layer %d has no probe inputs" % l)
        k, v, cis = inp
        tw.advance(k, cis, torch.tensor(keep_h, dtype=torch.int64, device=tw.pos.device))
        kvb = tw.total_cis if bias_rows is None else bias_rows(tw.total_cis)
        pos0 = [int(x) for x in eng._pos.tolist()]
        for b, kb in enumerate(keep_h):
            if kb > 0:
                _tw.write_tail(_b1_view(eng, b, pos0[b]), k[b:b + 1, :kb], v[b:b + 1, :kb], kvb[b:b + 1], layout.mirror_lo)
        eng._pos.copy_(torch.tensor([p + kb for p, kb in zip(pos0, keep_h)], dtype=eng._pos.dtype, device=eng._pos.device))
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


def probe(cache, st: PerReqState, snap: ProbeSnapshot, keep, U: int, layout, bias_rows=None) -> dict:
    """THE ROLLBACK PROBE in per-request form: the post-round state equals replay(keep); the control replay(keep') with
    ONE request's keep changed by one (+1, or -1 at U) differs; replay(keep) is deterministic. Leaves the engine in the
    replay(keep) state (= the committed state)."""
    keep_h = [int(x) for x in torch.as_tensor(keep).tolist()]
    pos0 = snap.slots[0]["_pos"].clone()
    after = capture(cache, st, pos0, layout)
    replay(cache, st, snap, keep_h, layout, bias_rows)
    same, why = states_equal(after, capture(cache, st, pos0, layout))
    alt = list(keep_h)
    j = 0
    alt[j] = alt[j] + 1 if alt[j] < U else alt[j] - 1
    replay(cache, st, snap, alt, layout, bias_rows)
    diff, why_alt = states_equal(after, capture(cache, st, pos0, layout))
    replay(cache, st, snap, keep_h, layout, bias_rows)
    same2, _ = states_equal(after, capture(cache, st, pos0, layout))
    return dict(keep=keep_h, U=int(U), same=bool(same), detail=why, control_keep=alt, control_differs=not bool(diff),
                control_detail=why_alt, replay_deterministic=bool(same2), passed=bool(same and not diff and same2))
