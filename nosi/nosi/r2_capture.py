"""R2-M2 step 8 (X4): the CAPTURE-SAFE decode step of the R2 path. Audit and rules: docs/R2_CAPTURE_AUDIT.md.

WHAT. decode_step_fixed(model, st, cb, cache) is r2_decode_path.decode_inference (the validated eager path: G0, ragged
main / long) with every per-step allocation of STATE replaced by a preallocated buffer written in place, every host
read of a device value removed, and the host bookkeeping moved out to the driver:

  eager layer_decode (r2_decode_path.py)          here (layer_decode_fixed)
  pos_before = twin.pos.clone()                   cb.pos_before.copy_(twin.pos)
  twin.step(k, cis, active)                       twin.step_fixed(k, cis, active)                    (X1)
  pooling_lengths(...): clones, host hooks        cb.n_used.copy_(n_comp); cb.M_b.copy_(pos_before // 64 + 1)
  st.M0 / negative-control hooks                  not supported (gates only); NG1 "frozen_widths" below
  st.topk_log[l] = topk_idx.clone()               cb.topk_log[l].copy_(st.ti)
  body_update -> R2Body -> DecodeBody.decode      cb.body.decode_fixed (DeviceWritebackBody, Triton write-back)  (X3)
  out = torch.empty(...) (FFN)                    cb.ffn_out (preallocated), silu_and_mul into it
  logits (returned, new tensor)                   cb.logits.copy_(...)
  st.steps += 1; st.pos_host += active            finish_step(st), on the host, after the step / replay

Transient intermediates (rmsnorm, nosa_linear, the stage-1 score, attention, F.linear, torch.where) still come from
the allocator; inside torch.cuda.graph they come from the graph's private pool at fixed addresses (the lead's
decision, 2026-10-20; docs/R2_CAPTURE_AUDIT.md "The allocation rule").

THE DRIVER (host, outside any captured region), per step s:
  prepare_step(st, cb, tok, active_host[s])     the activity (host mirror + the device mask, in place), the RoPE
                                                positions from the host counters, the input tokens (in place)
  decode_step_fixed(...)  or  graph.replay()
  finish_step(st)                               st.steps += 1; st.pos_host += active
  errors(st, cb)                                one sync: the body's and every twin's refusal bits (0 = none)

CAPTURE: capture_step(model, st, cb, cache) records ONE decode_step_fixed in a torch.cuda.CUDAGraph. Before it, at
least one EAGER decode_step_fixed on the same buffers (it compiles every Triton kernel with the capture's
specialisation and fills the body's index cache); capture executes nothing. state_pointers() lists the data_ptr of
every state tensor (twins, engines, R2State buffers, CaptureBuffers, the body's index cache): the gate requires the
same list before the capture and after the last replay.

NEGATIVE CONTROLS (gates only): cb.freeze_widths(st), called right before the capture, makes the traced step use
cb.M_frozen (the widths of the capture step, from the host counters) instead of pos_before // 64 + 1: a host value
frozen at capture. prepare_step(..., write_active=False) leaves the device mask stale (only the host mirror moves).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import r2_decode_body as RB
from . import r2_decode_path as RDP
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK
from .stage1_seqused import stage1_seqused

NEG_KINDS = (None, "frozen_widths")


def default_kernels():
    """The GPU kernels: the body's (diff_offload, flash_h2d_from_mask, flash_h2d_from_mask_bias; cache_engine.py:363,
    :381, :382) and the device write-back (r2_writeback.masked_writeback)."""
    from . import cache_engine as CE
    from . import r2_writeback as WB
    return (CE.diff.diff_offload, CE.flash_h2d_from_mask, CE.flash_h2d_from_mask_bias), WB.masked_writeback


class CaptureBuffers:
    """Every tensor the fixed step reads or writes that the R2State, the engines and the twins do not already hold.
    Allocated once, before any warm-up; never rebound."""

    def __init__(self, model, st, f4_fix: bool, kernels, writeback_kernel, ng=None):
        if ng not in NEG_KINDS:
            raise RDP.R2Error("unknown capture negative control %r" % (ng,))
        if getattr(st, "body_mode", None) != "r2":
            raise RDP.R2Error("the capture-safe step needs body 'r2'")
        lay0 = model.layers[0]
        B, H, L = int(st.B), int(st.H), len(model.layers)
        dev = st.active.device
        self.B, self.H, self.L = B, H, L
        self.ids = torch.zeros((B, 1), dtype=torch.int64, device=dev)
        self.pos_ids = torch.zeros((B, 1), dtype=torch.int32, device=dev)
        self.pos_flat = self.pos_ids.view(-1)
        self.pos_before = torch.zeros(B, dtype=torch.int64, device=dev)
        self.n_used = torch.zeros(B, dtype=torch.int32, device=dev)
        self.M_b = torch.zeros(B, dtype=torch.int32, device=dev)
        self.M_frozen = torch.zeros(B, dtype=torch.int32, device=dev)
        self.topk_log = torch.zeros((L, H, B, lay0.topk_blocks), dtype=torch.int64, device=dev)
        self.ffn_out = torch.zeros((B, 1, lay0.gate_up_proj.shape[0] // 2), dtype=lay0.gate_up_proj.dtype, device=dev)
        self.logits = torch.zeros((B, 1, model.lm_head.shape[0]), dtype=torch.float32, device=dev)
        self.err = torch.zeros((), dtype=torch.int32, device=dev)
        self.body = RB.DeviceWritebackBody(bool(f4_fix), writeback_kernel)
        self.kernels = tuple(kernels)
        self.ng = ng

    def freeze_widths(self, st) -> None:
        """NG1 only, right before the capture: M_frozen <- the widths the host counters give now, and the frozen
        branch is switched on. It is read when the step is traced, so the eager warm-up steps before it stay exact
        and the captured graph keeps the frozen widths for every replay."""
        self.M_frozen.copy_((st.pos_host // BLOCK + 1).to(torch.int32))
        self.ng = "frozen_widths"


def layer_decode_fixed(st, cb, l: int, layer, hidden, cos_sin_cache, cache):
    """r2_decode_path.layer_decode, capture-safe (module docstring table)."""
    NL = RDP._nl()
    B = hidden.shape[0]
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    twin = st.twins[l]
    residual = hidden
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = F.linear(h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
    q = q.view(B, -1)
    k = k.view(B, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(cb.pos_flat, q, k, D, cos_sin_cache, True)
    q = q.reshape(B, layer.num_heads, D)
    k = k.reshape(B, Hkv, D)
    v = v.reshape(B, Hkv, D)
    cb.pos_before.copy_(twin.pos)
    twin.step_fixed(k, cis.view(B, Hkv), st.active)
    cb.n_used.copy_(twin.n_comp)
    if cb.ng == "frozen_widths":
        cb.M_b.copy_(cb.M_frozen)
    else:
        cb.M_b.copy_(cb.pos_before // BLOCK + 1)
    ck = twin.comp_k_flat()
    score = stage1_seqused(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=cb.n_used)
    nosa_pooling_per_request(score, twin.ccis, cb.n_used, cb.M_b, st.pq, st.pc, stride=layer.pooling_stride,
                             block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
    torch.topk(st.pq, NL.QK_SELECT, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
    st.mask.zero_()
    st.mask.scatter_(2, st.ti_q, True)
    st.pc.masked_fill_(st.mask, float("inf"))
    torch.topk(st.pc, layer.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))
    cb.topk_log[l].copy_(st.ti)
    eng = cache.layers[l].cache_engine
    key_states, value_states, kv_bias, cache_lens = cb.body.decode_fixed(
        eng, k.unsqueeze(1), v.unsqueeze(1), twin.total_cis, st.ti, cb.pos_before, st.active, cb.err, *cb.kernels)
    attn = NL.flash_attn_nosa_with_kvcache(q.unsqueeze(1), key_states, value_states, kv_bias, cache_seqlens=cache_lens,
                                           num_splits=NL._ATTN_SPLITS)
    attn = attn.view(B, 1, -1)
    hidden = residual + F.linear(attn, layer.wo)
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = F.linear(h, layer.gate_up_proj)
    NL.silu_and_mul(h, cb.ffn_out)
    return residual + F.linear(cb.ffn_out, layer.down_proj)


def decode_step_fixed(model, st, cb, cache) -> None:
    """r2_decode_path.decode_inference, capture-safe: reads cb.ids / cb.pos_ids / st.active, writes cb.logits and
    every table in place. No host bookkeeping (finish_step does it)."""
    NL = RDP._nl()
    hidden = F.embedding(cb.ids, model.embed_tokens)
    for l, layer in enumerate(model.layers):
        hidden = layer_decode_fixed(st, cb, l, layer, hidden, model.cos_sin_cache, cache)
    hidden = NL.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    cb.logits.copy_(F.linear(hidden, model.lm_head).float())


# ------------------------------------------------------------------ the driver (host, outside any captured region)
def prepare_step(st, cb, tok, active_host, write_active: bool = True) -> None:
    """The inputs of the next step, written IN PLACE: the activity (the host mirror; the device mask unless
    write_active is False, the NG2 control), the RoPE positions from the host counters, the tokens (B, 1) int64."""
    a = torch.as_tensor(active_host, dtype=torch.bool)
    if tuple(a.shape) != (cb.B,):
        raise RDP.R2Error("active must be (%d,)" % cb.B)
    st.active_host.copy_(a)
    if write_active:
        st.active.copy_(st.active_host)
    cb.pos_ids.copy_(st.pos_host.view(-1, 1))
    cb.ids.copy_(tok)


def finish_step(st) -> None:
    """The host bookkeeping of r2_decode_path.decode_inference (:354-355), after the step or the replay."""
    st.steps += 1
    st.pos_host += st.active_host.to(torch.int64)


def errors(st, cb):
    """ONE sync: (the body's refusal bits, [every twin's refusal bits]); all 0 when nothing was refused."""
    codes = torch.stack([cb.err] + [t.err for t in st.twins]).cpu().tolist()
    return int(codes[0]), [int(c) for c in codes[1:]]


def capture_step(model, st, cb, cache):
    """ONE decode_step_fixed recorded in a CUDA graph (nothing executes during the capture). Call it only after at
    least one eager decode_step_fixed on the same buffers."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_step_fixed(model, st, cb, cache)
    return graph


TWIN_STATE = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "cu", "ring", "total_cis", "ccis", "_win",
              "_all_b", "err")
ENGINE_STATE = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_new_block_map_buf", "_load_mask", "_cache_lens",
                "_k_cpu", "_v_cpu")
R2STATE_BUFFERS = ("pq", "pc", "tv_q", "ti_q", "tv", "ti", "mask", "cu_q", "active")
CB_BUFFERS = ("ids", "pos_ids", "pos_before", "n_used", "M_b", "M_frozen", "topk_log", "ffn_out", "logits", "err")


def state_pointers(st, cb, cache):
    """{name: data_ptr} of every STATE tensor the fixed step reads or writes (module docstring, CAPTURE)."""
    out = {}
    for l, t in enumerate(st.twins):
        for k in TWIN_STATE:
            out["twin%d.%s" % (l, k)] = getattr(t, k).data_ptr()
        e = cache.layers[l].cache_engine
        for k in ENGINE_STATE:
            out["eng%d.%s" % (l, k)] = getattr(e, k).data_ptr()
    for k in R2STATE_BUFFERS:
        out["st.%s" % k] = getattr(st, k).data_ptr()
    for k in CB_BUFFERS:
        out["cb.%s" % k] = getattr(cb, k).data_ptr()
    for key, t in sorted(cb.body._idx.items()):
        out["body.index%s" % (key,)] = t.data_ptr()
    return out
