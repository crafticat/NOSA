"""Pure-torch reference of block-sparse attention over a union store.

Spec (retroinfer-eval ``docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md``):
section 3 (numerics of the two kernels), section 5(a) (this reference and the
exact-membership probe).

One KV head, one request. Inputs: q ``(U, d)``; union K, V ``(W, block_size, d)``;
cis ``(W, block_size)``; a per-query mask ``(U, K)`` of union slot ids (-1 padded,
as union_store.py emits); ``row_limits`` ``(U,)`` exclusive flat-row bounds
(tail_write.causal_row_limits, IV2 mask.h:181-185). A query attends the rows of
its selected slots below its row limit; rows are gathered in ASCENDING flat-row
order (slot-major). The GPU kernels iterate blocks descending; no bit identity
between this reference and either kernel is claimed for general values (spec
section 3). The bit-for-bit part is the membership probe, which is exact under
any summation order (section 5a).

Two conventions, two functions:

``attend_bias_in_logits``: the decode kernel's arithmetic
(flash-attention-nosa ``src/flash_fwd_kernel.h:321-342``): the bf16 bias is
added to the fp32 ``q.k`` accumulator per column, then softmax. In that kernel
the addition (:339-341) happens BEFORE ``softmax_rescale_o(acc_s, acc_o,
params.scale_softmax_log2)`` (:380-381; ``scale_apply_exp2``, softmax.h:142), so
the kernel's effective logit is ``(q.k + bias) * scale``. ``bias_before_scale``
selects that literal arithmetic (True) or a bias in logit units,
``q.k * scale + bias`` (False). The argument has no default on purpose.

``attend_value_scaled``: the prefill's emulation (nosa_llama.py:303-333):
``ucis = exp(cis)`` in the store dtype (:303), ``scaled_v = V * ucis`` (:304),
call 1 with ``scaled_v``, ``fake_v = ucis`` broadcast over d (:316), call 2 with
``fake_v``, ``out = out1 / out2[..., :1]`` (:332-333). Each call: fp32 softmax
without bias, P rounded to the store dtype for P.V, output in the store dtype.
Mathematically ``softmax(s) @ (V e^c) / softmax(s) @ e^c == softmax(s + c) @ V``,
i.e. this equals ``attend_bias_in_logits(..., bias_before_scale=False)`` up to
rounding.

``dtype`` is the store dtype at which the kernels round (bf16 in the engine);
``torch.float32`` makes every rounding point a no-op. ``scale`` is the softmax
scale (the wrappers use ``d ** -0.5`` when given None:
infllmv2_sparse_attention.py:302-303).

No CUDA, no extension import.
"""
from __future__ import annotations

import torch


def attended_rows(mask_row, row_limit: int, block_size: int):
    """Flat rows (ascending) a query with this mask row may read below row_limit."""
    slots = torch.unique(mask_row[mask_row >= 0].to(torch.int64))   # sorted, deduplicated (the kernel keeps bits)
    rows = (slots.unsqueeze(-1) * block_size + torch.arange(block_size, dtype=torch.int64, device=mask_row.device)).reshape(-1)
    return rows[rows < int(row_limit)]


def _check(q, k, v, cis, mask, row_limits):
    if q.dim() != 2 or k.dim() != 3 or v.dim() != 3 or cis.dim() != 2 or mask.dim() != 2:
        raise ValueError("shapes: q %s k %s v %s cis %s mask %s" % (
            tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(cis.shape), tuple(mask.shape)))
    U, d = q.shape
    W, bs, dk = k.shape
    if tuple(v.shape) != (W, bs, d) or dk != d or tuple(cis.shape) != (W, bs):
        raise ValueError("k %s v %s cis %s do not agree with q %s" % (tuple(k.shape), tuple(v.shape), tuple(cis.shape), tuple(q.shape)))
    if mask.shape[0] != U or len(row_limits) != U:
        raise ValueError("mask has %d rows, row_limits %d, q has %d queries" % (mask.shape[0], len(row_limits), U))
    if int(mask.max()) >= W:
        raise ValueError("mask names slot %d but the store has W=%d slots" % (int(mask.max()), W))
    return U, d, W, bs


def attend_bias_in_logits(q, k, v, cis, mask, row_limits, scale: float, dtype: torch.dtype, bias_before_scale: bool):
    U, d, W, bs = _check(q, k, v, cis, mask, row_limits)
    kf = k.to(dtype).float().reshape(W * bs, d)
    vf = v.to(dtype).float().reshape(W * bs, d)
    bf = cis.to(dtype).float().reshape(W * bs)
    qf = q.to(dtype).float()
    out = torch.full((U, d), float("nan"), dtype=torch.float32, device=q.device)
    for u in range(U):
        rows = attended_rows(mask[u], row_limits[u], bs)
        if rows.numel() == 0:
            continue                                        # nothing attended: 0/0, NaN (spec 5a)
        s = qf[u:u + 1] @ kf[rows].t()                      # (1, n) fp32 accumulate
        if bias_before_scale:
            s = (s + bf[rows].unsqueeze(0)) * scale         # flash_fwd_kernel.h:339-341 then :380
        else:
            s = s * scale + bf[rows].unsqueeze(0)           # bias in logit units
        p = torch.softmax(s, dim=-1)                        # fp32
        p = p.to(dtype).float()                             # P rounded for P.V (convert_type, flash_fwd_kernel.h:384)
        o = p @ vf[rows]                                    # (1, d) fp32 accumulate
        out[u] = o.to(dtype).float()[0]                     # kernel output in the store dtype
    return out


def attend_value_scaled(q, k, v, cis, mask, row_limits, scale: float, dtype: torch.dtype):
    U, d, W, bs = _check(q, k, v, cis, mask, row_limits)
    ucis = torch.exp(cis.to(dtype))                                     # nosa_llama.py:303, in the store dtype
    scaled_v = (v.to(dtype) * ucis.unsqueeze(-1)).reshape(W * bs, d)    # :304, rounded to the store dtype
    fake_v = ucis.unsqueeze(-1).expand(W, bs, d).reshape(W * bs, d)     # :316
    kf = k.to(dtype).float().reshape(W * bs, d)
    qf = q.to(dtype).float()
    scaled_vf = scaled_v.float()
    fake_vf = fake_v.float()
    out = torch.full((U, d), float("nan"), dtype=torch.float32, device=q.device)
    for u in range(U):
        rows = attended_rows(mask[u], row_limits[u], bs)
        if rows.numel() == 0:
            continue
        s = (qf[u:u + 1] @ kf[rows].t()) * scale            # fp32, no bias
        p = torch.softmax(s, dim=-1).to(dtype).float()      # P rounded for P.V
        o1 = (p @ scaled_vf[rows]).to(dtype)                # call 1 (:305-318), output in the store dtype
        o2 = (p @ fake_vf[rows]).to(dtype)                  # call 2 (:319-331)
        out[u] = (o1 / o2[:, :1]).float()[0]                # :332-333, division in the store dtype
    return out
