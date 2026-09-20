"""The REFERENCE TWIN of gate G3 and the in-situ term comparator (spec
retroinfer-eval docs/superpowers/specs/2026-09-20-paired-tick.md, G2 / G3).

Twin = the SAME rows kernel with U = 1 (V rows only) over the same canonical
order, through the same per-layer body (tick.layer_body with a V-only row
plan): the paired tick's V logits over N forced steps must be torch.equal to
the twin's at equal explicit splits (G3). The twin vs the SHIPPED decode
(decode_inference) is the separate 4 x tau_ctrl row (G2): the twin reads the
whole allocation (W slots) while the decode reads the 64-slot view, so their
split-KV seams differ (cache_engine.py prefill_update section 3c; job
2173299) -- the known-benign reorder class tau_ctrl measures.

WHY torch.equal IS EXPECTED, and what can break it (the driver names the
term): every V row of the tick is computed by the same kernels as the twin's
row, on the same inputs, with one difference -- the row COUNT: M = 2B in the
GEMMs (wqkv, wo, gate_up, down, lm_head) and 2B rows in the attention call.
  * attention: the split partition is a function of the ALLOCATED seqlen_k
    and num_splits only (flash_fwd_kernel.h:594-597), the per-row reduction
    never reads another row, and the combine is per row -> equal by
    construction at an explicit num_splits (0 would let the heuristic pick a
    different count for 2B rows: refused).
  * GEMMs: cuBLAS may select another kernel for another M (tile shape,
    split-K), which changes the fp32 accumulation order of EVERY row. This is
    the B2 caveat of verify_pilot.py. It is not a bug of the tick; it is the
    term the comparator isolates so the verdict can say so.
  * the bias / tail rule: the twin's V-only bias must equal the V rows of
    the paired bias (row 2b of the tick = row b of the twin), incl. the exact
    extent; checked per layer.
``InSituTwin`` recomputes, inside the paired tick, each term of every layer
at the twin's shape from the SAME inputs the tick used, and records
torch.equal / max |d| per term; ``diagnose`` names the first differing term
in kernel order: bias -> qkv -> attention -> wo/FFN -> final norm -> lm_head.
A free-running twin arm (``twin_step`` per forced step in its own process)
gives the arm-level G3 verdict; the in-situ records say WHY when it fails.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from . import core
from . import tick as _tick


@torch.inference_mode()
def twin_step(model, cache, sc: _tick.Scratch, cfg: _tick.TickConfig, v_tokens: torch.Tensor, position: torch.Tensor,
              trace=None) -> _tick.TickResult:
    """One forced step of the twin: V rows only (B rows), the shipped engine
    update, the rows kernel at U = 1 over the same canonical order."""
    G = _tick._gpu()
    B = sc.B
    if tuple(v_tokens.shape) != (B,) or tuple(position.shape) != (B,):
        raise ValueError("v_tokens / position must be (B=%d,)" % B)
    plan = core.row_plan(sc.req_all, True, False)
    return _tick._forward(G, model, cache, sc, cfg, v_tokens.unsqueeze(1), position.unsqueeze(1), plan, sc.req_all, True, trace=trace)


@torch.inference_mode()
def twin2b_step(model, cache, sc: _tick.Scratch, cfg: _tick.TickConfig, v_tokens: torch.Tensor, position: torch.Tensor,
                trace=None) -> _tick.TickResult:
    """The twin2b DIAGNOSTIC (DESIGN.md blocker 5): the U = 1 twin whose every
    GEMM (wqkv, wo, gate_up, down, lm_head) runs at M = 2B by appending B zero
    rows (tick.linear_padded), so cuBLAS selects the kernel the paired tick's
    GEMMs get. If G3 against the U = 1 twin fails at a GEMM term and the only
    difference was the kernel choice, the tick's V logits are torch.equal to
    this arm's. Everything else (scoring, engine update, the rows kernel at
    B rows) is the twin's."""
    return twin_step(model, cache, sc, cfg._replace(gemm_pad_rows=int(sc.B)), v_tokens, position, trace=trace)


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


class InSituTwin:
    """Term-by-term comparator run INSIDE a paired tick (equiv mode only; it
    adds one B-row GEMM set, one stage-1 + graph replay and one B-row
    attention per layer, outside any timing claim). ``capture_input`` keeps
    every layer's V-row input before the body runs; ``compare_layer`` and
    ``compare_head`` are called by the body / driver with the tick's own
    intermediate tensors. A failure inside the comparator is RECORDED on the
    layer's record (``error``) and never propagates: the arm must finish
    (job 2175550 lost both equiv arms to a strided-view rope argument).

    Terms, in kernel order (each recomputed at M = B from the SAME inputs):
      bias    the twin's V-only bias vs the V rows of the paired bias (mask, extent, row map)
      qkv     prenorm + wqkv GEMM + nosa_linear, PRE-rope q / k and v / cis
      rope    the flashinfer rope on the twin's pre-rope q / k at V's positions vs the tick's post-rope rows
      score   the stage-1 score on the tick's own V q (the table state V scored against)
      sel     the selection sets after the captured pooling / top-k replay on that score
      attn    the rows kernel at B rows on the tick's own V q / bias
      ffn     wo + gate_up + down at M = B on the tick's own V attention output
    """

    def __init__(self):
        self.inputs = {}
        self.records: List[dict] = []

    def capture_input(self, l: int, hidden: torch.Tensor):
        self.inputs[l] = hidden[:, 0:1].clone()          # (B, 1, hid): the V rows' layer input

    def compare_layer(self, G, model, cache, layer, l, *, position_ids, q_pre, k_pre, q, k, v, cis, score_v, sel_v, vis_v, rb, attn,
                      hidden_out, eng, sc, cfg) -> dict:
        rec = dict(layer=l)
        try:
            self._compare_layer(rec, G, model, cache, layer, l, position_ids, q_pre, k_pre, q, k, v, cis, score_v, sel_v, vis_v, rb, attn,
                                hidden_out, eng, sc, cfg)
        except Exception as e:   # never crash the arm: the record says where the comparator stopped
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        self.records.append(rec)
        return rec

    def _compare_layer(self, rec, G, model, cache, layer, l, position_ids, q_pre, k_pre, q, k, v, cis, score_v, sel_v, vis_v, rb, attn,
                       hidden_out, eng, sc, cfg):
        NL = G.NL
        x = self.inputs[l]
        B = x.shape[0]
        Hq, Hk, D = layer.num_heads, layer.num_key_value_heads, layer.head_dim
        U = q.shape[1]
        # term: bias / tail rule
        cbi = torch.arange(B, dtype=torch.int32, device=x.device)
        rb1 = core.paired_bias(eng._kv_bias_gpu, vis_v, cbi, 1, cfg.masked, out=sc.bias_twin, check_values=False)
        rows_v = torch.arange(0, U * B, U, device=x.device)
        rec["bias_equal"] = bool(torch.equal(rb.bias[rows_v], rb1.bias[:B]) and torch.equal(rb.cache_seqlens[rows_v], rb1.cache_seqlens)
                                 and torch.equal(rb.cache_batch_idx[rows_v], rb1.cache_batch_idx))
        rec["bias_maxabs"] = _maxabs(rb.bias[rows_v], rb1.bias[:B])
        # term: qkv (pre-rope) at M = B on the same layer input
        hs = NL.layer_norm(x, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = F.linear(hs, layer.wqkv)
        q1, k1, v1, c1 = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hk)
        q1 = q1.view(B, -1).contiguous()
        k1 = k1.view(B, -1).contiguous()
        v1 = v1.reshape(B, Hk, D)
        c1 = c1.reshape(B, Hk)
        qp_pre = q_pre.view(B, U, -1)[:, 0] if q_pre is not None else None
        kp_pre = k_pre.view(B, U, -1)[:, 0] if k_pre is not None else None
        vp, cp = v[:, 0], cis[:, 0]
        rec.update(v_equal=torch.equal(v1, vp), cis_equal=torch.equal(c1, cp), v_maxabs=_maxabs(v1, vp), cis_maxabs=_maxabs(c1, cp))
        if qp_pre is not None:
            rec.update(q_pre_equal=torch.equal(q1, qp_pre), k_pre_equal=torch.equal(k1, kp_pre), q_pre_maxabs=_maxabs(q1, qp_pre), k_pre_maxabs=_maxabs(k1, kp_pre))
            rec["qkv_equal"] = bool(rec["q_pre_equal"] and rec["k_pre_equal"] and rec["v_equal"] and rec["cis_equal"])
            rec["qkv_maxabs"] = max(rec["q_pre_maxabs"], rec["k_pre_maxabs"], rec["v_maxabs"], rec["cis_maxabs"])
        # term: rope on the twin's pre-rope rows at V's positions (contiguous: rope.cu CHECK_INPUT)
        NL.apply_rope_with_cos_sin_cache_inplace(_tick.rope_positions(position_ids[:, 0]), q1, k1, D, model.cos_sin_cache, True)
        q1 = q1.reshape(B, Hq, D)
        k1 = k1.reshape(B, Hk, D)
        qp, kp = q[:, 0], k[:, 0]
        rec.update(rope_equal=bool(torch.equal(q1, qp) and torch.equal(k1, kp)), rope_maxabs=max(_maxabs(q1, qp), _maxabs(k1, kp)))
        # term: score (stage 1 on the TICK's own V q against the table state V scored against) and the selection sets
        if score_v is not None and sel_v is not None:
            compressed_k, cu_comp, max_seqlen_comp = cache.update_compress_k_decode(None, l)
            ck = compressed_k.contiguous().flatten(0, 1)
            score1 = NL.infllmv2_attn_stage1_fast(qp.contiguous(), ck, ck, cu_seqlens_q=sc.cu_full, cu_seqlens_k=cu_comp,
                                                  max_seqlen_q=1, max_seqlen_k=max_seqlen_comp, causal=False)
            rec["score_equal"] = torch.equal(score1, score_v)
            rec["score_maxabs"] = _maxabs(score1, score_v)
            layer.score_buf.copy_(score1)
            layer.after_pooling_graph.replay()
            sel1 = model.topk_idx_buf.clone()
            a, b = torch.sort(sel1, dim=-1).values, torch.sort(sel_v, dim=-1).values
            rec["sel_equal"] = torch.equal(a, b)
            rec["sel_rows_differ"] = int((a != b).any(-1).sum())
        # term: the rows kernel at B rows on the PAIRED q (isolates the row-count dependence of the kernel)
        a1 = _tick.attend_rows(G, qp.contiguous(), eng._k_gpu, eng._v_gpu, rb1, cfg.num_splits)
        ap = attn[:, 0].reshape(B, Hq, D)
        rec["attn_equal"] = torch.equal(a1, ap)
        rec["attn_maxabs"] = _maxabs(a1, ap)
        # term: wo + FFN at M = B on the PAIRED attention output (isolates the GEMM M-dependence)
        h = x + F.linear(attn[:, 0:1], layer.wo)
        r2 = h
        hs2 = NL.layer_norm(h, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        gu = F.linear(hs2, layer.gate_up_proj)
        dd = gu.shape[-1] // 2
        act = torch.empty(gu.shape[:-1] + (dd,), dtype=gu.dtype, device=gu.device)
        NL.silu_and_mul(gu, act)
        o1 = r2 + F.linear(act, layer.down_proj)
        rec["ffn_equal"] = torch.equal(o1, hidden_out[:, 0:1])
        rec["ffn_maxabs"] = _maxabs(o1, hidden_out[:, 0:1])

    def compare_head(self, G, model, pre: torch.Tensor, post: torch.Tensor, logits: torch.Tensor) -> dict:
        rec = dict(layer="head")
        try:
            NL = G.NL
            n1 = NL.layer_norm(pre[:, 0:1], model.norm_variance_epsilon, model.norm_weight)
            l1 = F.linear(post[:, 0:1], model.lm_head).float()
            rec.update(norm_equal=torch.equal(n1, post[:, 0:1]), norm_maxabs=_maxabs(n1, post[:, 0:1]),
                       lm_head_equal=torch.equal(l1[:, 0], logits[:, 0]), lm_head_maxabs=_maxabs(l1[:, 0], logits[:, 0]))
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        self.records.append(rec)
        return rec


TERMS = (("bias_equal", "bias / tail rule: the twin's V-only bias differs from the V rows of the paired bias (mask, extent or row map)"),
         ("qkv_equal", "qkv (pre-rope): prenorm + wqkv GEMM at M = 2B vs M = B + nosa_linear (cuBLAS kernel choice -> fp32 accumulation order)"),
         ("rope_equal", "rope: the flashinfer rope at V's positions on the same pre-rope rows"),
         ("score_equal", "score: the stage-1 score of the tick's V q against the table state (compressed keys / cis) V scored against"),
         ("sel_equal", "selection: the captured pooling / top-k replay on the same score (selection SETS differ)"),
         ("attn_equal", "rows attention over 2B rows vs B rows on the SAME q / K / V / bias: the kernel's per-row independence or the split seams"),
         ("ffn_equal", "wo / gate_up / down GEMMs at M = 2B vs M = B on the SAME attention output"),
         ("norm_equal", "final rmsnorm at M = 2B vs M = B"),
         ("lm_head_equal", "lm_head GEMM at M = 2B vs M = B on the SAME normed hidden"))
_MAXABS_KEY = {"bias_equal": "bias_maxabs", "qkv_equal": "qkv_maxabs", "rope_equal": "rope_maxabs", "score_equal": "score_maxabs",
               "sel_equal": "sel_rows_differ", "attn_equal": "attn_maxabs", "ffn_equal": "ffn_maxabs", "norm_equal": "norm_maxabs",
               "lm_head_equal": "lm_head_maxabs"}


def _first_departing(rec: dict):
    if rec.get("error"):
        return "error", rec["error"]
    for key, why in TERMS:
        if key in rec and not rec[key]:
            return key, why
    return None, None


def diagnose(records: List[dict]) -> Optional[str]:
    """The first differing term of one tick's in-situ records, in kernel
    order (or the first comparator error), or None when every term is
    torch.equal."""
    for rec in records:
        key, why = _first_departing(rec)
        if key == "error":
            return "layer %s: comparator ERROR %s" % (rec["layer"], why)
        if key is not None:
            return "layer %s: %s (%s = %s)" % (rec["layer"], why, _MAXABS_KEY[key], rec.get(_MAXABS_KEY[key]))
    return None


def _term_summary(rec: dict) -> str:
    parts = []
    for key, _ in TERMS:
        if key in rec:
            mk = _MAXABS_KEY[key]
            parts.append("%s=%s%s" % (key.replace("_equal", ""), "ok" if rec[key] else "DIFF", "" if rec[key] else "(%s %s)" % (mk.split("_")[-1], rec.get(mk))))
    return " ".join(parts)


def diagnose_layers(records: List[dict]) -> List[str]:
    """One line per layer whose record has a departing term or an error:
    'layer l: first departs at TERM | every term with its max |d|'."""
    out = []
    for rec in records:
        key, why = _first_departing(rec)
        if key is None:
            continue
        if key == "error":
            out.append("layer %s: comparator ERROR %s | %s" % (rec["layer"], why, _term_summary(rec)))
        else:
            out.append("layer %s: first departs at %s | %s" % (rec["layer"], key.replace("_equal", ""), _term_summary(rec)))
    return out


def g3_verdict(paired_v: List[torch.Tensor], twin: List[torch.Tensor]) -> dict:
    """paired_v[t], twin[t]: (B, V) fp32 logits of the same forced step ->
    per step torch.equal and max |d|; ``all_equal`` is gate G3."""
    if len(paired_v) != len(twin):
        raise ValueError("paired %d steps vs twin %d steps" % (len(paired_v), len(twin)))
    eq, mx = [], []
    for a, b in zip(paired_v, twin):
        if a.shape != b.shape:
            raise ValueError("logits shapes differ: %s vs %s" % (tuple(a.shape), tuple(b.shape)))
        eq.append(bool(torch.equal(a, b)))
        mx.append(_maxabs(a, b))
    first_bad = next((i for i, e in enumerate(eq) if not e), None)
    return dict(all_equal=all(eq), per_step_equal=eq, per_step_maxabs=mx, first_differing_step=first_bad, steps=len(eq))
