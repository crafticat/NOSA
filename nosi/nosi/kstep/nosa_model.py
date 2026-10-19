"""Milestone 10 (INTERFACE_K3X section 1; Track C): the NOSA-8B MATH MODEL for the kstep dispatcher, a tensor-valued,
BiasToy-compatible model object, and NosaSerial, its serial reference decoder.

The per-row decode operations are those of nosi.nosa_llama LlamaLayer.decode_forward (nl:545-670) at bsz 1, q_len 1,
in the same order: layer_norm (input), F.linear(wqkv), nosa_linear (q, k, v, cis), RoPE (NeoX halves, the model's
cos_sin_cache), [selection], the NOSA attention with the additive cis bias, F.linear(wo), residual, layer_norm (post),
F.linear(gate_up_proj), silu_and_mul, F.linear(down_proj), residual; then the final norm, lm_head and the float32
softmax (Llama.decode_inference, nl:1121-1125). The selection is the C13 gate's P4 / P5 (scripts/kstep_c13_gate.py
CaseRun.select in the research repo): stage-1 scores of the row's query group against the VISIBLE compressed windows,
nosa_pooling into one score pair per block (PQ, PC), then c13.select_blocks / complete_blocks (forced init and local
blocks, the frozen tie rule: value desc, id asc). For M = n_cand + 1 <= max_sel every complete block is selected and
no score is computed (as c13). The window compression is CaseRun.produce's: window w covers positions
[w * stride, w * stride + chain_kernel); ck = the mean of its K rows, cc = the mean of its cis values; it becomes
visible to the row after its last position (c13.n_w).

Two op sets behind ONE small interface (rmsnorm, nosa_linear, rope, stage1, pool, attn, silu_and_mul):
  TorchOps   pure torch (CPU or GPU): the REFERENCE SEMANTICS. Every reduction is computed in float32 and rounded to the
             model dtype at the kernels' rounding points (written out per method).
  KernelOps  the NOSA kernels as nosa_llama uses them (flashinfer rmsnorm / rope / silu_and_mul, the nosa_linear triton
             kernel, infllmv2_attn_stage1_fast, the nosa_pooling triton kernel, flash_attn_nosa_with_kvcache with
             num_splits = nl._ATTN_SPLITS). CUDA only; nosi.nosa_llama is imported in the constructor, never at module
             import (this module imports torch, fractions and c13 only).
F.linear, F.embedding, the residual additions, torch.stack / mean (the compression) and the final float32 softmax are
the same torch calls in both op sets.

EXACTNESS IS WITHIN ONE OP SET: the serial reference (NosaSerial) and the executor (Track B) use the same model object,
hence the same ops in the same order on the same inputs; their results must be bitwise equal (torch.equal). TorchOps and
KernelOps are NOT bitwise equal; the known semantic gaps of the kernels against TorchOps:
  - nosa_linear (nosa_linear.py:83-94): the cis dot product multiplies bf16 by bf16 in triton (a rounded product) before
    the sum, and computes log(1 + exp(act)) * A; TorchOps sums float32 products and uses F.softplus (linear above 20).
  - RoPE / rmsnorm / silu_and_mul: flashinfer computes in float32 and rounds once to the dtype, as TorchOps, but its
    operation order (fma contraction, the rsqrt and mean formulation) is not claimed equal: about 1 ulp in the dtype.
  - stage 1: the kernel's accumulation and output precision are not read here (bf16 inputs; output width >= n_w);
    TorchOps: float32 logits q . ck * D^-0.5, float32 softmax per query head over the windows, float32 sum over the
    group, rounded to the dtype. KernelOps calls stage 1 per (row, KV head) with the head's G query heads and its own
    windows (nosa_llama calls it once for both KV heads of a batch); per-head equality with the 2-head call is not
    claimed.
  - pooling: a max (exact) over the windows [m (S - 1) - P, m (S - 1) - P + S) inside [0, n_w), S = 5, P = 1 (the
    nosa_pooling defaults, mpf:97), with +inf at the forced ids; -inf when no window. Exact in both op sets; the
    kernel's only rounding is the cast of the max into the dtype, which commutes with max. The tie order of the final
    top-k is c13's (value desc, id asc) in BOTH op sets; the engine's torch.topk(sorted=False) has no specified tie
    order (c13 12.10.3: the sets agree whenever the cut values differ).
  - attention: TorchOps follows the decode kernel's arithmetic (flash_fwd_kernel.h:339-342, as
    nosi/nosi/verify/reference_attention.py attend_bias_in_logits with bias_before_scale=True): logits
    (q . k + bias) * D^-0.5 in float32, float32 softmax, P rounded to the dtype for P . V, float32 accumulation, output
    rounded to the dtype. The kernel uses an online (blocked, split by num_splits) softmax with exp2 and its own
    reduction order: not bitwise equal. KernelOps passes a cache of exactly n rows (the attended rows); the split
    partition depends on that allocated length (c13 E8), so the kernel's result depends on n, which is the same for the
    serial reference and the executor.
  - compression: the mean over the window's chain_kernel rows per KV head (torch.stack(...).mean(0)); the C13 gate
    reduces a (1, kernel, H, D) stack over dim 1: the same values, a possibly different reduction order.

Model attributes: L, HQ, H (KV heads), G = HQ // H, D, Dm, V, bs, stride, chain_kernel, max_sel (= NOSA's topk_blocks,
the own block included: a row selects at most max_sel - 1 complete blocks per KV head), qk_select, init_blocks,
local_blocks, dtype, device, abi = 4, ops. kstep_toy compatibility (the lead's build and the toy protocol layers):
reset_contract = "I10" in this class body with reset_holders() -> [] and detach_completed() -> 0 (the model keeps no
event), _pre / _post / post_heads / joint_post / draft_perturb / draft_noise / calls / rows_seen, and head_out also
takes the toy form head_out(l, q_h, items) with items a list of (k, v, c).

No CUDA, triton, flashinfer or engine import at module import. No claim of equality with NOSA's prefill, its batched
decode, its cache engine or its graph-captured selection; no timing or memory claim."""
from __future__ import annotations

import collections
import copy
import functools
import importlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import c13

SET_SIZE, PADDING = 5, 1        # nosa_pooling defaults (mpf:97): windows per pooled block, left padding (step S - 1)


def _no_grad(fn):
    @functools.wraps(fn)
    def run(*a, **kw):
        with torch.no_grad():
            return fn(*a, **kw)
    return run


# ---------------------------------------------------------------- device-built index tensors (review MINOR-2)
# The per-row path builds its small integer tensors ON THE DEVICE (torch.full / torch.arange: a kernel launch, no
# pageable host-to-device copy of a Python list). Each equals the list-built tensor it replaces (dtype and values; the
# CPU test NM13 checks every case against torch.tensor([...])).
def index1(v: int, dtype: torch.dtype, device: Any) -> torch.Tensor:
    """torch.tensor([v], dtype=dtype, device=device), built on the device."""
    return torch.full((1,), int(v), dtype=dtype, device=device)


def cu_pair(n: int, device: Any) -> torch.Tensor:
    """torch.tensor([0, n], dtype=torch.int32, device=device), built on the device (arange(2) * n)."""
    return torch.arange(2, dtype=torch.int32, device=device) * int(n)


def attended_index(sel: Sequence[int], pos: int, bs: int, device: Any) -> torch.Tensor:
    """The attended stored rows of a row at pos, in increasing position order: the bs rows of each selected complete
    block (sel ascending, every id < pos // bs), then the own block's rows before pos. Equals torch.tensor([p for j in
    sel for p in range(j bs, (j + 1) bs)] + list(range((pos // bs) bs, pos)), dtype=torch.long, device=device), built
    on the device: one torch.arange per run of consecutive rows (adjacent blocks, and the own block's rows after the
    block just before it, merge into one run), concatenated."""
    runs: List[List[int]] = []
    for j in sel:
        a, b = int(j) * bs, (int(j) + 1) * bs
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    own0 = (pos // bs) * bs
    if pos > own0:
        if runs and runs[-1][1] == own0:
            runs[-1][1] = pos
        else:
            runs.append([own0, pos])
    if not runs:
        return torch.empty((0,), dtype=torch.long, device=device)
    parts = [torch.arange(a, b, dtype=torch.long, device=device) for a, b in runs]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def argmax_lowest(p: torch.Tensor) -> int:
    """The lowest index among the maxima of a 1-D tensor (accept.argmax's tie rule on the same values); ONE
    device-to-host read of the index."""
    return int(torch.nonzero(p == p.max())[0, 0])


# ================================================================================================== the op sets
class TorchOps:
    """The reference semantics, pure torch (CPU or GPU). Every method is row-local: one row (or one row and KV head)."""
    name = "torch"

    @staticmethod
    def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """flashinfer rmsnorm (nl:48-53 layer_norm): x * rsqrt(mean(x^2) + eps) * w in float32, rounded once."""
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()).to(x.dtype)

    @staticmethod
    def nosa_linear(qkv: torch.Tensor, delta_w: torch.Tensor, A: torch.Tensor, q_size: int, kv_size: int,
                    H: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """nosa_linear.py reference_nosa_linear (:142-161) for ONE row qkv (1, q_size + 2 kv_size): the q / k / v
        splits (copies) and cis = A * softplus(v . delta_h) per KV head h (the full v of the row, as the kernel), the
        dot product and the softplus in float32, cis rounded to the dtype. Returns q (1, q_size), k, v (1, kv_size),
        cis (1, H)."""
        q = qkv[:, :q_size].contiguous()
        k = qkv[:, q_size:q_size + kv_size].contiguous()
        v = qkv[:, q_size + kv_size:q_size + 2 * kv_size].contiguous()
        dt = (v.float().unsqueeze(1) * delta_w.float().unsqueeze(0)).sum(dim=-1)            # (1, H)
        cis = (A.float() * F.softplus(dt)).to(qkv.dtype)
        return q, k, v, cis

    @staticmethod
    def rope(pos: int, q: torch.Tensor, k: torch.Tensor, D: int, cos_sin: torch.Tensor):
        """apply_rope_with_cos_sin_cache_inplace(pos, q, k, D, cos_sin_cache, is_neox=True) (nl:571), NOT in place:
        per head vector x = (x1, x2) halves, (x1 cos - x2 sin, x2 cos + x1 sin) in float32 with the cache row pos
        (cos in its first D / 2 columns, sin in the next D / 2), rounded once. Returns new (q, k)."""
        half = D // 2
        row = cos_sin[pos].float()
        cos, sin = row[:half], row[half:2 * half]

        def rot(x):
            xf = x.float().reshape(-1, D)
            x1, x2 = xf[:, :half], xf[:, half:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype).reshape(x.shape)
        return rot(q), rot(k)

    @staticmethod
    def stage1(q_h: torch.Tensor, ck: torch.Tensor) -> torch.Tensor:
        """infllmv2_attn_stage1_fast for one (row, KV head) (InfLLM-v2 tests/test_stage1.py naive_attention, decode,
        causal False): per query head g, softmax over the windows of q_g . ck_w * D^-0.5 in float32, summed over the G
        query heads in float32, rounded to the dtype. q_h (G, D), ck (n_w, D) -> (n_w,)."""
        D = q_h.shape[-1]
        logits = (q_h.float() @ ck.float().t()) * (D ** -0.5)                               # (G, n_w)
        return torch.softmax(logits, dim=-1).sum(dim=0).to(q_h.dtype)

    @staticmethod
    def pool(score: torch.Tensor, cc: torch.Tensor, M: int, init_blocks: int, local_blocks: int, bs: int,
             stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """nosa_pooling (mpf:7-75, launched with local_blocks + 1 at mpf:105) for one (row, KV head): pooled block m
        reads the windows m (S - 1) - P .. m (S - 1) - P + S - 1 inside [0, n_w) and takes their max (-inf when none);
        ids m < init_blocks or M - m <= local_blocks + 1 are +inf. score, cc (n_w,) -> (PQ, PC) (M,) in the dtype of
        score / cc. Exact (a max; no arithmetic)."""
        nw = int(score.shape[0])
        s, c = score.float().tolist(), cc.float().tolist()
        pq, pc = [], []
        for m in range(M):
            if m < init_blocks or M - m <= local_blocks + 1:
                pq.append(float("inf"))
                pc.append(float("inf"))
                continue
            lo = m * (SET_SIZE - 1) - PADDING
            ws = [n for n in range(lo, lo + SET_SIZE) if 0 <= n < nw]
            pq.append(max(s[n] for n in ws) if ws else float("-inf"))
            pc.append(max(c[n] for n in ws) if ws else float("-inf"))
        return (torch.tensor(pq, dtype=score.dtype, device=score.device),
                torch.tensor(pc, dtype=cc.dtype, device=cc.device))

    @staticmethod
    def attn(q_h: torch.Tensor, K: torch.Tensor, V: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """The decode kernel's arithmetic for one (row, KV head), query head by query head (reference_attention.py
        attend_bias_in_logits, bias_before_scale=True): s = (q_g . K^T + C) * D^-0.5 in float32, float32 softmax, P
        rounded to the dtype, o = P . V accumulated in float32, rounded to the dtype. q_h (G, D), K / V (n, D), C (n,)
        -> (G, D), rows in increasing position order as given."""
        dt = q_h.dtype
        G, D = q_h.shape
        scale = D ** -0.5
        qf, kf, vf, bf = q_h.float(), K.float(), V.float(), C.float()
        out = torch.empty((G, D), dtype=dt, device=q_h.device)
        for g in range(G):
            s = qf[g:g + 1] @ kf.t()                                                         # (1, n) float32
            s = (s + bf.unsqueeze(0)) * scale
            p = torch.softmax(s, dim=-1).to(dt).float()
            out[g] = (p @ vf).to(dt)[0]
        return out

    @staticmethod
    def silu_and_mul(gu: torch.Tensor) -> torch.Tensor:
        """flashinfer silu_and_mul (nl:663-664): silu(a) * b of the halves (a, b) in float32, rounded once."""
        d = gu.shape[-1] // 2
        return (F.silu(gu[..., :d].float()) * gu[..., d:].float()).to(gu.dtype)


class KernelOps:
    """The NOSA kernels exactly as nosa_llama calls them (bsz 1, q_len 1), per row / (row, KV head). CUDA only. nl: the
    imported nosi.nosa_llama module (None: imported here); splits: num_splits of the attention (None: nl._ATTN_SPLITS,
    the NOSI_ATTN_SPLITS environment default of the engine). Constraints of the kernels (not of the math): nosa_linear
    hard-codes D_DIM 256 (kv_size = H D <= 256; 256 q_size / kv_size a power of two); nosa_pooling has the fixed
    S = 5 / P = 1 geometry (bs = 4 stride, chain_kernel = 2 stride: NosaModel checks it)."""
    name = "kernel"

    def __init__(self, nl: Any = None, splits: Optional[int] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("KernelOps needs CUDA (the NOSA kernels); use TorchOps on the CPU")
        self.nl = nl if nl is not None else importlib.import_module("nosi.nosa_llama")
        self.splits = int(self.nl._ATTN_SPLITS if splits is None else splits)
        self._cu_q = {}                                 # device -> the constant cu_seqlens_q [0, 1] (built once)

    def rmsnorm(self, x, w, eps):
        return self.nl.layer_norm(x, eps, w)

    def nosa_linear(self, qkv, delta_w, A, q_size, kv_size, H):
        q, k, v, cis = self.nl.nosa_linear(qkv.reshape(1, 1, -1).contiguous(), delta_w, A, q_size, kv_size, H)
        return q.reshape(1, q_size), k.reshape(1, kv_size), v.reshape(1, kv_size), cis.reshape(1, H)

    def rope(self, pos, q, k, D, cos_sin):
        q2, k2 = q.clone().contiguous(), k.clone().contiguous()
        p = index1(pos, torch.long, q.device)
        self.nl.apply_rope_with_cos_sin_cache_inplace(p, q2, k2, D, cos_sin, True)
        return q2, k2

    def stage1(self, q_h, ck):
        nw = int(ck.shape[0])
        dev = q_h.device
        kk = ck.reshape(nw, 1, -1).contiguous()
        cu_q = self._cu_q.get(dev)
        if cu_q is None:
            cu_q = self._cu_q[dev] = cu_pair(1, dev)
        cu_k = cu_pair(nw, dev)
        score = self.nl.infllmv2_attn_stage1_fast(q_h.reshape(1, q_h.shape[0], -1).contiguous(), kk, kk,
                                                  cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1,
                                                  max_seqlen_k=nw, causal=False)
        return score[0, 0, :nw]

    def pool(self, score, cc, M, init_blocks, local_blocks, bs, stride):
        nw = int(score.shape[0])
        PQ = torch.empty((1, 1, M), dtype=cc.dtype, device=cc.device)
        PC = torch.empty((1, 1, M), dtype=cc.dtype, device=cc.device)
        self.nl.nosa_pooling(score.reshape(1, 1, nw), cc.reshape(1, 1, nw).contiguous(), nw, PQ, PC, stride=stride,
                             block_size=bs, local_blocks=local_blocks, init_blocks=init_blocks)
        return PQ[0, 0], PC[0, 0]

    def attn(self, q_h, K, V, C):
        G, D = q_h.shape
        n = int(K.shape[0])
        cs = index1(n, torch.int32, q_h.device)
        out = self.nl.flash_attn_nosa_with_kvcache(q_h.reshape(1, 1, G, D).contiguous(),
                                                   K.reshape(1, n, 1, D).contiguous(),
                                                   V.reshape(1, n, 1, D).contiguous(),
                                                   C.reshape(1, n, 1).contiguous(), cache_seqlens=cs,
                                                   num_splits=self.splits)
        return out.reshape(G, D)

    def silu_and_mul(self, gu):
        d = gu.shape[-1] // 2
        out = torch.empty(gu.shape[:-1] + (d,), dtype=gu.dtype, device=gu.device)
        self.nl.silu_and_mul(gu.contiguous(), out)
        return out


# ================================================================================================== the weights
@dataclass
class LayerWeights:
    """One decoder layer, BY REFERENCE (nosa_llama LlamaLayer fields; nothing is copied)."""
    wqkv: torch.Tensor
    delta_w: torch.Tensor          # layer.delta.weight (H, kv_size)
    A: torch.Tensor                # layer.A (H,)
    q_size: int
    kv_size: int
    wo: torch.Tensor
    gate_up: torch.Tensor          # layer.gate_up_proj
    down: torch.Tensor             # layer.down_proj
    in_w: torch.Tensor
    in_eps: float
    post_w: torch.Tensor
    post_eps: float


@dataclass
class NosaWeights:
    embed: torch.Tensor            # embed_tokens (V, Dm)
    cos_sin: torch.Tensor          # cos_sin_cache (N, D) float32: cos in [:, :D/2], sin in [:, D/2:]
    norm_w: torch.Tensor
    norm_eps: float
    lm_head: torch.Tensor          # (V, Dm)
    layers: List[LayerWeights]


def _cos_sin(D: int, max_pos: int, theta: float, dtype: torch.dtype, device) -> torch.Tensor:
    """Llama._set_cos_sin_cache (nl:1024-1028) with rotary dim D: cos / sin of t * inv_freq rounded to the dtype, the
    first halves concatenated, float32."""
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32, device=device) / D))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    return torch.cat((cos[:, :D // 2], sin[:, :D // 2]), dim=-1).to(torch.float32)


# ================================================================================================== the model
class NosaModel:
    """The section-1 math model (module docstring). Construct with from_llama (the real NOSA-8B, weights by reference)
    or tiny (a random NOSA-shaped config for the CPU tests)."""
    abi = 4
    reset_contract = "I10"          # I10: this model keeps no backend event between calls (reset_holders lists none)

    def reset_holders(self):
        """I10: the model holds weights, the op set and instrumentation (calls, last_ties, prefill_hidden), no event."""
        return []

    def detach_completed(self):
        """I10: reset_holders lists nothing, so nothing is detached."""
        return 0

    def __init__(self, weights: NosaWeights, *, L: int, HQ: int, H: int, D: int, Dm: int, V: int, bs: int,
                 stride: int, chain_kernel: int, max_sel: int, qk_select: int, init_blocks: int, local_blocks: int,
                 ops: Any, dtype: torch.dtype, device: Any):
        if HQ % H or D % 2:
            raise ValueError("NosaModel: HQ %d must be a multiple of H %d and D %d even" % (HQ, H, D))
        if bs != (SET_SIZE - 1) * stride or chain_kernel != (PADDING + 1) * stride:
            raise ValueError("NosaModel: nosa_pooling's fixed geometry needs bs = %d stride and chain_kernel = %d "
                             "stride (got bs %d, chain_kernel %d, stride %d)"
                             % (SET_SIZE - 1, PADDING + 1, bs, chain_kernel, stride))
        if not init_blocks + local_blocks + 1 <= qk_select <= max_sel:
            raise ValueError("NosaModel: need init_blocks + local_blocks + 1 <= qk_select <= max_sel (got %d + %d + 1, "
                             "%d, %d): the own block must stay selected" % (init_blocks, local_blocks, qk_select,
                                                                           max_sel))
        if len(weights.layers) != L or tuple(weights.cos_sin.shape[1:]) != (D,):
            raise ValueError("NosaModel: %d layer weights for L %d, cos_sin %s for D %d"
                             % (len(weights.layers), L, tuple(weights.cos_sin.shape), D))
        self.w = weights
        self.L, self.HQ, self.H, self.G, self.D, self.Dm, self.V = L, HQ, H, HQ // H, D, Dm, V
        self.bs, self.stride, self.chain_kernel = bs, stride, chain_kernel
        self.max_sel, self.qk_select = max_sel, qk_select
        self.init_blocks, self.local_blocks = init_blocks, local_blocks
        self.ops, self.dtype, self.device = ops, dtype, torch.device(device)
        self.prefill_hidden: Optional[torch.Tensor] = None
        self.last_ties = None                       # instrumentation: c13's (QK cut tie, CIS cut tie) of the last P5
        self.calls = dict(pre=0, post=0)            # instrumentation (the Toy's counters)
        self.rows_seen = None
        self.draft_noise = frozenset()              # R5 rejection injection (the Toy's): draft rows at these positions

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_llama(cls, llama: Any, ops: Any = None, nl: Any = None, qk_select: Optional[int] = None) -> "NosaModel":
        """The real model from an nl.Llama instance (weights BY REFERENCE: no copy). ops None: KernelOps(nl).
        qk_select None (review NIT-2): ops.nl.QK_SELECT (nl:45) when the ops are KernelOps; else nl.QK_SELECT when
        nl is given; else c13.QK_SELECT (the same 33)."""
        if ops is None:
            ops = KernelOps(nl)
        lay0 = llama.layers[0]
        layers = [LayerWeights(wqkv=ly.wqkv, delta_w=ly.delta.weight.detach(), A=ly.A.detach(), q_size=int(ly.q_size),
                               kv_size=int(ly.kv_size), wo=ly.wo, gate_up=ly.gate_up_proj, down=ly.down_proj,
                               in_w=ly.input_layernorm_weight, in_eps=float(ly.input_layernorm_variance_epsilon),
                               post_w=ly.post_attention_layernorm_weight,
                               post_eps=float(ly.post_attention_layernorm_variance_epsilon))
                  for ly in llama.layers]
        w = NosaWeights(embed=llama.embed_tokens, cos_sin=llama.cos_sin_cache, norm_w=llama.norm_weight,
                        norm_eps=float(llama.norm_variance_epsilon), lm_head=llama.lm_head, layers=layers)
        if qk_select is None:
            if isinstance(ops, KernelOps):
                qk_select = int(ops.nl.QK_SELECT)
            elif nl is not None:
                qk_select = int(nl.QK_SELECT)
            else:
                qk_select = c13.QK_SELECT
        return cls(w, L=int(llama.num_layers), HQ=int(llama.num_heads), H=int(llama.num_key_value_heads),
                   D=int(llama.head_dim), Dm=int(llama.hidden_size), V=int(llama.lm_head.shape[0]),
                   bs=int(lay0.block_size), stride=int(lay0.pooling_stride), chain_kernel=int(lay0.pooling_block_size),
                   max_sel=int(lay0.topk_blocks), qk_select=int(qk_select), init_blocks=int(lay0.init_blocks),
                   local_blocks=int(lay0.window_blocks), ops=ops, dtype=llama.dtype, device=llama.device)

    @classmethod
    def tiny(cls, seed: int = 0, ops: Any = None, dtype: torch.dtype = torch.bfloat16, device: Any = "cpu",
             L: int = 2, HQ: int = 4, H: int = 2, D: int = 8, Dm: int = 16, V: int = 32, Fi: int = 32, bs: int = 8,
             chain_kernel: int = 4, stride: int = 2, max_sel: int = 5, qk_select: int = 4, init_blocks: int = 1,
             local_blocks: int = 1, eps: float = 1e-5, theta: float = 10000.0, max_pos: int = 2048) -> "NosaModel":
        """A random NOSA-shaped model (deterministic in seed; weights drawn in float32 on the CPU, then moved and
        rounded to dtype). Defaults: L 2, HQ 4, H 2, D 8, Dm 16, V 32, FFN 32, bs 8, chain_kernel 4, stride 2,
        max_sel 5, qk_select 4, init 1, local 1 (forced ids 0, M - 2, M - 1: selection is non-trivial once
        M = n_cand + 1 > 5, i.e. from position 40). ops None: TorchOps()."""
        g = torch.Generator().manual_seed(int(seed))
        rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
        cast = lambda t: t.to(device=device, dtype=dtype)
        q_size, kv_size = HQ * D, H * D
        layers = []
        for _ in range(L):
            layers.append(LayerWeights(
                wqkv=cast(rn(q_size + 2 * kv_size, Dm) / Dm ** 0.5), delta_w=cast(rn(H, kv_size) / kv_size ** 0.5),
                A=cast(0.5 + torch.rand(H, generator=g)), q_size=q_size, kv_size=kv_size,
                wo=cast(rn(Dm, q_size) / q_size ** 0.5), gate_up=cast(rn(2 * Fi, Dm) / Dm ** 0.5),
                down=cast(rn(Dm, Fi) / Fi ** 0.5), in_w=cast(1.0 + 0.1 * rn(Dm)), in_eps=eps,
                post_w=cast(1.0 + 0.1 * rn(Dm)), post_eps=eps))
        w = NosaWeights(embed=cast(rn(V, Dm)), cos_sin=_cos_sin(D, max_pos, theta, dtype, "cpu").to(device),
                        norm_w=cast(1.0 + 0.1 * rn(Dm)), norm_eps=eps, lm_head=cast(rn(V, Dm) / Dm ** 0.5),
                        layers=layers)
        return cls(w, L=L, HQ=HQ, H=H, D=D, Dm=Dm, V=V, bs=bs, stride=stride, chain_kernel=chain_kernel,
                   max_sel=max_sel, qk_select=qk_select, init_blocks=init_blocks, local_blocks=local_blocks,
                   ops=ops if ops is not None else TorchOps(), dtype=dtype, device=device)

    def with_ops(self, ops: Any) -> "NosaModel":
        """The same weights (shared) with another op set; fresh instrumentation."""
        m = copy.copy(self)
        m.ops, m.prefill_hidden, m.last_ties, m.calls = ops, None, None, dict(pre=0, post=0)
        return m

    # ---------------------------------------------------------------------------------------------- section 1 API
    @_no_grad
    def embed(self, token: int) -> torch.Tensor:
        """F.embedding of one token (nl:1075): (1, Dm)."""
        return F.embedding(index1(token, torch.long, self.device), self.w.embed)

    @_no_grad
    def joint_pre(self, l: int, items: Sequence[Tuple[torch.Tensor, int]]):
        """Per item (x (1, Dm), pos), computed ONE ROW AT A TIME: (q, k, v, c), q a list over KV heads of (G, D) after
        RoPE (query heads h G .. h G + G - 1), k / v lists of (D,), c a list of () (the cis bias). Fresh tensors."""
        self.calls["pre"] += len(items)
        return [self._pre_row(l, x, pos) for x, pos in items]

    def _pre_row(self, l: int, x: torch.Tensor, pos: int):
        lw, ops, H, G, D = self.w.layers[l], self.ops, self.H, self.G, self.D
        h = ops.rmsnorm(x.reshape(1, self.Dm), lw.in_w, lw.in_eps)
        qkv = F.linear(h, lw.wqkv)
        q, k, v, cis = ops.nosa_linear(qkv, lw.delta_w, lw.A, lw.q_size, lw.kv_size, H)
        q, k = ops.rope(int(pos), q, k, D, self.w.cos_sin)
        qh, kh, vh, ch = q.reshape(self.HQ, D), k.reshape(H, D), v.reshape(H, D), cis.reshape(H)
        return ([qh[i * G:(i + 1) * G].clone() for i in range(H)], [kh[i].clone() for i in range(H)],
                [vh[i].clone() for i in range(H)], [ch[i].clone() for i in range(H)])

    @_no_grad
    def block_scores(self, l: int, h: int, q_h: torch.Tensor, cks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                     n_cand: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """P4 for one (row, KV head): stage 1 over the visible windows cks, then the pooling into (PQ, PC), each (M,)
        with M = n_cand + 1 (+inf at the forced ids)."""
        if not cks:
            raise ValueError("NosaModel.block_scores: no visible window (layer %d head %d, n_cand %d)" % (l, h, n_cand))
        CK = torch.stack([ck for ck, _ in cks])
        CC = torch.stack([cc for _, cc in cks])
        score = self.ops.stage1(q_h, CK)
        return self.ops.pool(score, CC, n_cand + 1, self.init_blocks, self.local_blocks, self.bs, self.stride)

    @_no_grad
    def select(self, l: int, h: int, q_h: torch.Tensor, cks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
               n_cand: int) -> List[int]:
        """NOSA's selection of one (row, KV head): sorted complete block ids < n_cand. M = n_cand + 1 <= max_sel: every
        complete block (c13, no score). Else P4 (block_scores) and P5: c13.select_blocks(M, PQ, PC, h, qk_select,
        max_sel) and complete_blocks (forced init / local ids, the tie rule value desc, id asc)."""
        return self._select_many(l, [h], [q_h], [cks], n_cand)[0]

    @_no_grad
    def select_heads(self, l: int, q: Sequence[torch.Tensor],
                     cks: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]], n_cand: int) -> List[List[int]]:
        """select for EVERY KV head of one row (q and cks: lists over the KV heads): equal to [select(l, h, q[h],
        cks[h], n_cand) for h], with ONE device-to-host read of the PQ / PC of all heads (review MINOR-2). A select
        patched on the INSTANCE (a test-local control, NM11) is honoured: then it is called per head instead."""
        if "select" in self.__dict__:
            return [self.select(l, h, q[h], cks[h], n_cand) for h in range(self.H)]
        return self._select_many(l, list(range(self.H)), q, cks, n_cand)

    def _select_many(self, l: int, heads: Sequence[int], qs, ckss, n_cand: int) -> List[List[int]]:
        if n_cand < 0:
            raise ValueError("NosaModel.select: n_cand %d" % n_cand)
        M = n_cand + 1
        if M <= self.max_sel:
            sel, ties = c13.select_blocks(M, topk=self.max_sel)
            self.last_ties = ties
            comp = [int(j) for j in c13.complete_blocks(sel, M)]
            return [list(comp) for _ in heads]
        scores = [self.block_scores(l, h, q_h, cks, n_cand) for h, q_h, cks in zip(heads, qs, ckss)]
        host = torch.stack([t.float() for pair in scores for t in pair]).cpu().tolist()   # ONE read: 2 x heads rows
        out = []
        for i, h in enumerate(heads):
            sel, ties = c13.select_blocks(M, host[2 * i], host[2 * i + 1], head=h, qk_select=self.qk_select,
                                          topk=self.max_sel)
            self.last_ties = ties
            out.append([int(j) for j in c13.complete_blocks(sel, M)])
        return out

    @_no_grad
    def compress(self, elems: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """The window reduction of chain_kernel (k, c) elements of one KV head: (mean of the k (D,), mean of the c
        ())."""
        if len(elems) != self.chain_kernel:
            raise ValueError("NosaModel.compress: %d elements, chain_kernel %d" % (len(elems), self.chain_kernel))
        return torch.stack([e[0] for e in elems]).mean(dim=0), torch.stack([e[1] for e in elems]).mean(dim=0)

    @_no_grad
    def head_out(self, l: int, q_h: torch.Tensor, K: Any, Vv: Optional[torch.Tensor] = None,
                 C: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The attention of one (row, KV head) over its attended positions in increasing position order: K / Vv (n, D),
        C (n,) -> (G, D). Toy form: head_out(l, q_h, items) with items a list of (k, v, c), stacked in order."""
        if Vv is None and C is None:
            items = list(K)
            K = torch.stack([it[0] for it in items])
            Vv = torch.stack([it[1] for it in items])
            C = torch.stack([it[2] for it in items])
        return self.ops.attn(q_h, K, Vv, C)

    @_no_grad
    def mix(self, l: int, x: torch.Tensor, heads: Sequence[torch.Tensor]) -> torch.Tensor:
        """wo, residual, post norm, gate_up, silu_and_mul, down, residual (nl:651-665): the row output (1, Dm) from the
        layer input x (1, Dm) and the head results in KV-head order."""
        lw, ops = self.w.layers[l], self.ops
        x = x.reshape(1, self.Dm)
        attn = torch.cat(list(heads), dim=0).reshape(1, self.HQ * self.D)
        y = x + F.linear(attn, lw.wo)
        z = F.linear(ops.silu_and_mul(F.linear(ops.rmsnorm(y, lw.post_w, lw.post_eps), lw.gate_up)), lw.down)
        return y + z

    @_no_grad
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """(V,) float32 probabilities: the final norm and lm_head with .float() as Llama.decode_inference
        (nl:1119-1123), which RETURNS those float32 logits; the float32 softmax over them is this model's own step
        (the engine has no softmax there)."""
        hn = self.ops.rmsnorm(x.reshape(1, self.Dm), self.w.norm_w, self.w.norm_eps)
        return torch.softmax(F.linear(hn, self.w.lm_head).float(), dim=-1)[0]

    def argmax_token(self, x: torch.Tensor) -> int:
        """The greedy token of x without building the Fractions: the lowest id among the maxima of probs(x), which
        equals accept.argmax(dist(x)) (Fraction(float(p)) is exact and order-preserving; both take the first maximum).
        dist itself is unchanged."""
        return argmax_lowest(self.probs(x))

    def dist(self, x: torch.Tensor) -> List[Fraction]:
        """Each probability of probs(x) as Fraction(float(p)): the exact rational of the float32 value (the total need
        not be 1)."""
        return [Fraction(p) for p in self.probs(x).cpu().tolist()]

    def prefill(self, tokens: Sequence[int]):
        """kv[pos][l][h] = (k, v, c) of feeding tokens[0 .. n - 1] by the SERIAL decode (NosaSerial: one position at a
        time, the same ops and selection as decoding); model.prefill_hidden = the last position's final hidden (None
        for no token)."""
        s = NosaSerial(self)
        for tok in tokens:
            s.step(tok)
        self.prefill_hidden = s.hidden
        return s.kv

    # ---------------------------------------------------------------------------------------------- toy compatibility
    def _pre(self, l: int, x: torch.Tensor, pos: int):
        """kstep_toy ExactState / twin form: the (q, k, v, c) of one row."""
        return self.joint_pre(l, [(x, pos)])[0]

    def post_heads(self, l: int, x: torch.Tensor, q, kv):
        """(head results, output) of one row from kv[h], a list of (k, v, c) per KV head (BiasToy.post_heads)."""
        heads = [self.head_out(l, q[h], kv[h]) for h in range(self.H)]
        return heads, self.mix(l, x, heads)

    def _post(self, l: int, x: torch.Tensor, q, kv) -> torch.Tensor:
        return self.post_heads(l, x, q, kv)[1]

    def draft_perturb(self, l: int, pos: int, x: torch.Tensor) -> torch.Tensor:
        """The Toy's R5 rejection injection: a draft row at a position in draft_noise gets x + 0.5 (l + 1) after every
        layer (the target never does). Unchanged otherwise."""
        if pos not in self.draft_noise:
            return x
        return x + torch.full_like(x, 0.5 * (l + 1))

    def joint_post(self, l: int, rows):
        """The legacy (Row.kv) executor call: per row post_heads from Row.kv, draft_perturb on draft rows."""
        self.calls["post"] += len(rows)
        if self.rows_seen is not None:
            self.rows_seen.extend((self.calls["post"], l, row) for row in rows)
        out = []
        for row in rows:
            y = self._post(l, row.hidden, row.q, row.kv)
            out.append(self.draft_perturb(l, row.pos, y) if row.kind == "draft" else y)
        return out


# ============================================================================================= the serial reference
class _HeadStore:
    """The K / V / bias rows of one (layer, KV head) of NosaSerial, in position order (a growing buffer)."""

    def __init__(self, D: int, dtype: torch.dtype, device: torch.device, cap: int = 64):
        self.K = torch.zeros((cap, D), dtype=dtype, device=device)
        self.V = torch.zeros((cap, D), dtype=dtype, device=device)
        self.C = torch.zeros((cap,), dtype=dtype, device=device)
        self.n = 0

    def append(self, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> None:
        if self.n == self.K.shape[0]:
            cap = 2 * self.n
            for name in ("K", "V", "C"):
                old = getattr(self, name)
                new = torch.zeros((cap,) + tuple(old.shape[1:]), dtype=old.dtype, device=old.device)
                new[:self.n].copy_(old)
                setattr(self, name, new)
        self.K[self.n].copy_(k)
        self.V[self.n].copy_(v)
        self.C[self.n].copy_(c)
        self.n += 1


class NosaSerial:
    """The serial reference decoder of a NosaModel: feed(token) -> dist, one position at a time, all layers. Per layer:
    joint_pre of the one row; per KV head the selection from its OWN visible windows (n_cand = pos // bs), the attended
    rows in increasing position order (the selected complete blocks' bs rows each, the own block's earlier rows, the
    current row), head_out; mix; then the row's (k, v, c) are stored and a window that ends at this position is
    compressed (visible from the next position on). The SAME model calls in the same order as the executor's per-row
    path: the correctness reference. kv[pos][l][h] = (k, v, c); cks[l][h] = [(ck, cc)]; sel_log = [(pos, l, h,
    selection)]; hidden = the last final hidden.
    Memory (review NIT-1): every position's (k, v, c) is kept TWICE, in kv (the fresh joint_pre tensors, for prefill
    and the tests) and in the per-(layer, head) stores (store[l][h], the rows the attention gathers). sel_log grows by
    L x H entries per position and is unbounded by default; sel_log_cap = n keeps only the last n entries (a deque).
    Per layer, the selections of all KV heads come from model.select_heads (one device-to-host read of their scores);
    the attended row index is built on the device (attended_index)."""

    def __init__(self, model: NosaModel, sel_log_cap: Optional[int] = None):
        self.m = model
        self.pos = 0
        self.hidden: Optional[torch.Tensor] = None
        self.kv: List[List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = []
        self.store = [[_HeadStore(model.D, model.dtype, model.device) for _ in range(model.H)] for _ in range(model.L)]
        self.cks: List[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = [[[] for _ in range(model.H)]
                                                                         for _ in range(model.L)]
        self.sel_log = [] if sel_log_cap is None else collections.deque(maxlen=int(sel_log_cap))

    @_no_grad
    def step(self, token: int) -> torch.Tensor:
        """Feed one token at position self.pos; returns the final hidden (1, Dm)."""
        m, pos, bs = self.m, self.pos, self.m.bs
        x = m.embed(token)
        kvp = []
        for l in range(m.L):
            q, k, v, c = m.joint_pre(l, [(x, pos)])[0]
            heads = []
            sels = m.select_heads(l, q, self.cks[l], pos // bs)
            for h in range(m.H):
                sel = sels[h]
                self.sel_log.append((pos, l, h, tuple(sel)))
                idx = attended_index(sel, pos, bs, m.device)
                st = self.store[l][h]
                K = torch.cat([st.K.index_select(0, idx), k[h].unsqueeze(0)])
                V = torch.cat([st.V.index_select(0, idx), v[h].unsqueeze(0)])
                C = torch.cat([st.C.index_select(0, idx), c[h].unsqueeze(0)])
                heads.append(m.head_out(l, q[h], K, V, C))
            x = m.mix(l, x, heads)
            n, kk = pos + 1, m.chain_kernel
            for h in range(m.H):
                st = self.store[l][h]
                st.append(k[h], v[h], c[h])
                if n >= kk and (n - kk) % m.stride == 0:              # the window that ends at pos (c13.n_w)
                    self.cks[l][h].append(m.compress([(st.K[i], st.C[i]) for i in range(n - kk, n)]))
            kvp.append([(k[h], v[h], c[h]) for h in range(m.H)])
        self.kv.append(kvp)
        self.pos += 1
        self.hidden = x
        return x

    def feed(self, token: int) -> List[Fraction]:
        return self.m.dist(self.step(token))
