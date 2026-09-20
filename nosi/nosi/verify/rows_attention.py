"""'Rows attention': the shipped flash-attention-nosa DECODE kernel, unchanged,
with ONE batch row per (request, position) over the union store and a per-row
bias that carries BOTH the CIS bias and the row's selection.

Reuse ledger (retroinfer-eval ``docs/superpowers/specs/2026-09-20-reuse-ledger.md``)
section 3, decision (ii-a): candidate table 3.5, registered prediction 3.6.
This is a general 'rows' helper, not a verify-only special case: a fused
verify + speculate tick (one verify row and one draft row per request, each
with its own selection, in ONE attention call) is the same mechanism.

KERNEL FACTS (by reading; ``FAN/`` = ``dependencies/flash-attention-nosa/csrc/flash_attn_nosa``):
  * The bias is indexed by the QUERY batch row ``bidb``:
    ``bias[bidb * kv_bias_batch_stride + bidh * kv_bias_head_stride + col * kv_bias_row_stride]``
    (``FAN/src/flash_fwd_kernel.h:321-342`` and ``:950-970``; strides read off the
    tensor at ``FAN/flash_api.cpp:60-78``). After the ``seqlenq_ngroups_swapped``
    transpose (``flash_api.cpp:347-353``, taken because ``seqlen_q == 1`` and
    ``num_heads > num_heads_k``) ``bidh`` IS the KV head, so with a contiguous
    ``(rows, W*bs, Hkv)`` bias the element read is ``bias[row, col, kv_head]``.
  * K and V are read through ``bidb_cache = cache_batch_idx[bidb]``
    (``flash_fwd_kernel.h:649-660``): several query rows may share one request's
    K/V without any copy.
  * ``actual_seqlen_k = cache_seqlens[bidb]`` per query row
    (``flash_api.cpp:466-474`` sets ``params.cu_seqlens_k`` non-cumulative;
    ``FAN/src/block_info.h:14-23``).
  * The split partition depends on the ALLOCATED ``seqlen_k = W*bs`` and
    ``num_splits`` only (``flash_fwd_kernel.h:594-597``): rows of one call and
    rows of the shipped decode over the same allocation at the same explicit
    ``num_splits`` see the same seams. ``num_splits = 0`` is the heuristic
    (``flash_api.cpp:181, :235``) and depends on the number of rows, so it
    differs between a B-row call and a B*U-row call.
  * The ONE host obstacle: ``FAN/flash_api.cpp:366``
    ``CHECK_SHAPE(bcache, batch_size_c, seqlen_k, num_heads_k)`` with
    ``batch_size_c = kcache.size(0) = B`` (``:339``), while the kernel addresses
    rows ``0 .. B*U-1`` through ``stride(0)``. MICROBENCH HACK (this module):
    allocate the bias as ``(B*U, W*bs, Hkv)`` contiguous and pass the narrowed
    view ``bias[:B]``: its ``size(0) == B`` satisfies the check, its
    ``stride(0) == W*bs*Hkv`` is what the kernel multiplies ``bidb`` by, and
    the storage holds every row (``rows_attention_args`` asserts all three).
    PRODUCTION FIX: relax ``:366`` to ``CHECK_SHAPE(bcache, batch_size, seqlen_k,
    num_heads_k)`` when ``cache_batch_idx_`` is given (one host line, no kernel
    change) and rebuild the ``flash_attn_2_cuda_nosa`` extension inside
    ``nosa-env.sif`` on a compute node, as the resolver extension was
    (REPRODUCE.md:2996). Until then the narrowed view is the only way in.

THE MASK VALUE IS FINITE ON PURPOSE (``MASKED = -3.0e4``, the same value and
reason as ``avail_policy.MASK_BIAS``): the split-KV kernel's non-masking
iterations call ``softmax_rescale_o<Is_first=false, Check_inf=false>``
(``FAN/src/softmax.h``), so a running row max of ``-inf`` over two consecutive
fully-masked blocks gives ``(-inf) - (-inf) = NaN`` in the rescale factor.
A finite ``-3e4`` keeps every partial max finite, and a masked column's weight
``exp2((q.k - 3e4) * scale * log2e - max_scaled)`` underflows to EXACTLY 0 in
fp32 (the argument is below -2000 for any scale >= 0.08), so masked keys
contribute exact zeros to the row sum and to P.V; a split with no visible
column gets combine weight ``exp(lse_split - lse_total) = 0`` exactly
(``flash_fwd_kernel.h:1331-1339``).

WHAT ONE ROW SEES (``build_rows_bias``): row ``r = b*U + u`` carries request b's
cis on every row of every slot named by ``masks[h, r, :]`` (Path-1 masks in
UNION SLOT ids, ``union_store.build_union``), ``MASKED`` on every other row.
TAIL RULE: at position u the tail stream has ``n = tail_len_0 + u + 1`` tokens
(the ``tail_len_0`` old rows plus the round's tokens 0..u); the tail slot
shows ``min(n, bs)`` rows and the next-tail slot ``max(0, n - bs)`` rows, i.e.
tail rows beyond ``tail_len_0 + u`` are masked. In the engine's MIRROR layout
(tail T at ``W-2``, T+1 at ``W-1``, the last slots) this equals the flat causal
bound ``tail_write.causal_row_limits`` that Path 1 runs under (the CPU test
proves it for every ``tail_len_0`` and U). In the PHYSICAL layout (tail at
slot ``topk-1``) the same rule addresses the rows the shipped decode reads --
that is the identity gate: a position-0 row with no new block selects slots
``0..topk-2`` in full plus tail rows ``0..tail_len_0``, exactly the decode's
``cache_seqlens = (topk-1)*bs + tail_len_0 + 1`` rows, same K/V memory, same
partition at equal explicit ``num_splits`` -> ``torch.equal`` expected
(ledger 3.4). ``rows_bias_from_round`` does the mirror -> physical remap for a
round without rollover (slot ``topk-1`` then holds T's rows exactly as the
mirror does, ``tail_write.write_tail``); a rolled round keeps the mirrors.

PRECONDITION the shipped decode never needed: the rows call LOADS every K/V
row below the row's ``cache_seqlens`` (masked slots, tail rows beyond the
tail length), the decode loads only its window. A never-written row of a
``torch.empty`` allocation may hold NaN/Inf, and ``0 * NaN = NaN`` in P.V
(``q.k = NaN`` in S). The allocation must be finite everywhere below the row
extents: zero-fill ``_k_gpu``/``_v_gpu``/``_kv_bias_gpu`` at creation
(cache_engine.py:293-295, ``torch.empty`` -> ``torch.zeros``) before the first
rows call. The CPU twin test shows the hazard is real.

MEMORY: the bias is ``B*U x W*bs x Hkv x 2 B``: 256 x 8192 x 2 x 2 = 8 MB per
layer at batch 128, U = 2, W = 128; one ``torch.where`` builds it.

``reference_rows_attention`` is the pure-torch twin: per row and q head, the
kernel's bias-in-fp32-logits arithmetic (``reference_attention.attend_bias_in_logits``
with ``bias_before_scale=True``: ``(q.k + bias) * scale``), either over the
row's VISIBLE columns gathered in ascending order (``gather=True``: the same
ops on the same rows as the reference -> ``torch.equal`` in the CPU test) or
densely over all columns below ``cache_seqlens`` with the masked value in the
logits (``gather=False``: proves the masked value is an exact zero; equal to
the gathered twin only up to summation order -> ``allclose`` in fp32).

No CUDA, no extension import at module level: the ONE extension import is
``flash_attn_nosa.flash_attn_with_kvcache`` (the entry nosa_llama.py:42 binds as
``flash_attn_nosa_with_kvcache``), inside ``rows_attention`` only. ``num_splits``
is an explicit argument with no default: the caller resolves its knob.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import torch

# finite on purpose; see the module docstring (same value as avail_policy.MASK_BIAS)
MASKED = -3.0e4


class RowsBias(NamedTuple):
    bias: torch.Tensor             # (B*U, W*bs, Hkv) store dtype, contiguous; row b*U + u
    selected: torch.Tensor         # (B*U, W, Hkv) bool: slot named by the row's mask
    visible_rows: torch.Tensor     # (B*U, W, Hkv) int64 in [0, bs]: rows of the slot the row attends
    cache_batch_idx: torch.Tensor  # (B*U,) int32 = arange(B).repeat_interleave(U)
    cache_seqlens: torch.Tensor    # (B*U,) int32 = (1 + highest slot with a visible row) * bs; 0 if none
    U: int
    block_size: int


class RowsArgs(NamedTuple):
    q: torch.Tensor                # (B*U, 1, Hq, D) contiguous
    k_cache: torch.Tensor          # (B, W*bs, Hkv, D): the WHOLE allocation, no copy
    v_cache: torch.Tensor          # (B, W*bs, Hkv, D)
    bias_view: torch.Tensor        # rb.bias[:B]: size(0) == B for flash_api.cpp:366, stride(0) == W*bs*Hkv
    cache_seqlens: torch.Tensor    # (B*U,) int32
    cache_batch_idx: torch.Tensor  # (B*U,) int32
    num_splits: int


def tail_visible_rows(tail_len_0: int, U: int, block_size: int) -> torch.Tensor:
    """``(U, 2)`` int64: rows of the tail block T (column 0) and of T+1 (column 1)
    visible to position u. ``n = tail_len_0 + u + 1``; T shows ``min(n, bs)``,
    T+1 shows ``max(0, n - bs)``."""
    bs = int(block_size)
    tail_len_0 = int(tail_len_0)
    U = int(U)
    if bs < 1 or U < 1:
        raise ValueError("tail_visible_rows: need block_size >= 1 and U >= 1, got %d, %d" % (bs, U))
    if not (0 <= tail_len_0 < bs):
        raise ValueError("tail_len_0=%d must be in [0, block_size=%d)" % (tail_len_0, bs))
    if tail_len_0 + U >= 2 * bs:
        # tail_write.write_tail refuses a second rollover inside one round too
        raise ValueError("tail_len_0 %d + U %d >= 2*block_size %d: more than one rollover per round" % (tail_len_0, U, 2 * bs))
    n = tail_len_0 + torch.arange(U, dtype=torch.int64) + 1
    return torch.stack([n.clamp(max=bs), (n - bs).clamp(min=0)], dim=1)


def remap_slots(masks: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    """A copy of ``masks`` with every id in ``mapping`` replaced; -1 and other ids untouched."""
    out = masks.clone()
    for src, dst in mapping.items():
        out[masks == int(src)] = int(dst)
    return out


def build_rows_bias(union_cis_rows: torch.Tensor, masks: torch.Tensor, tail_len_0: int, U: int, *,
                    tail_slot: int, block_size: int, tail_next_slot: Optional[int] = None,
                    masked_value: float = MASKED) -> RowsBias:
    """Per-row bias over the union store (module docstring, WHAT ONE ROW SEES).

    union_cis_rows: ``(B, W*bs, Hkv)`` the engine's ``_kv_bias_gpu`` (store dtype).
    masks: ``(Hkv, B*U, K)`` int32 union slot ids, -1 padded, row ``b*U + u``
        (``union_store.UnionRound.mask``, possibly ``remap_slots``-ed).
    tail_len_0: ``_tail_block_len_on_gpu`` at round start; U: positions per request.
    tail_slot: the slot whose rows follow the tail rule (``topk-1`` physical, or the
        mirror ``W-2``); tail_next_slot: the slot holding T+1 after a rollover
        (mirror ``W-1``), None when the layout has none -- then a rollover
        inside the round is refused.
    """
    bs = int(block_size)
    U = int(U)
    tail_len_0 = int(tail_len_0)
    if union_cis_rows.dim() != 3:
        raise ValueError("union_cis_rows must be (B, W*bs, Hkv), got %s" % (tuple(union_cis_rows.shape),))
    if not union_cis_rows.is_floating_point():
        raise TypeError("union_cis_rows must be a floating dtype, got %s" % union_cis_rows.dtype)
    B, rows, H = union_cis_rows.shape
    if bs < 1 or rows % bs != 0:
        raise ValueError("union_cis_rows has %d rows, not a multiple of block_size %d" % (rows, bs))
    W = rows // bs
    if masks.dtype != torch.int32 or masks.dim() != 3:
        raise ValueError("masks must be int32 (Hkv, B*U, K), got %s %s" % (masks.dtype, tuple(masks.shape)))
    Hm, BU, K = masks.shape
    if Hm != H or BU != B * U or U < 1 or K < 1:
        raise ValueError("masks %s do not match (Hkv=%d, B*U=%d*%d, K>=1)" % (tuple(masks.shape), H, B, U))
    if masks.device != union_cis_rows.device:
        raise ValueError("masks and union_cis_rows must be on one device")
    if not (0 <= int(tail_slot) < W):
        raise ValueError("tail_slot=%d outside [0, W=%d)" % (tail_slot, W))
    if tail_next_slot is not None:
        if not (0 <= int(tail_next_slot) < W) or int(tail_next_slot) == int(tail_slot):
            raise ValueError("tail_next_slot=%d must be a slot in [0, W=%d) other than tail_slot=%d" % (tail_next_slot, W, tail_slot))
    if bool((masks < -1).any()) or bool((masks >= W).any()):
        raise ValueError("masks name a slot outside [-1, W=%d)" % W)
    if tail_next_slot is None and tail_len_0 + U > bs:
        raise ValueError("tail_len_0 %d + U %d > block_size %d rolls the tail over inside the round, "
                         "and the layout has no tail_next_slot for T+1" % (tail_len_0, U, bs))
    tv = tail_visible_rows(tail_len_0, U, bs).to(union_cis_rows.device)   # (U, 2)
    dev = union_cis_rows.device

    # selected[r, slot, h] <- slot in masks[h, r, :]   (-1 goes to a dump column)
    idx = torch.where(masks >= 0, masks, torch.full_like(masks, W)).to(torch.int64)
    sel_ext = torch.zeros((H, BU, W + 1), dtype=torch.bool, device=dev)
    sel_ext.scatter_(-1, idx, torch.ones_like(idx, dtype=torch.bool))
    selected = sel_ext[..., :W].permute(1, 2, 0).contiguous()               # (B*U, W, Hkv)

    # rows of each slot the row may attend: bs, except the tail slots (tail rule)
    per_slot = torch.full((BU, W), bs, dtype=torch.int64, device=dev)
    per_slot[:, int(tail_slot)] = tv[:, 0].repeat(B)                         # row b*U+u -> u
    if tail_next_slot is not None:
        per_slot[:, int(tail_next_slot)] = tv[:, 1].repeat(B)
    visible_rows = torch.where(selected, per_slot.unsqueeze(-1), torch.zeros_like(per_slot).unsqueeze(-1))

    r = torch.arange(bs, dtype=torch.int64, device=dev)
    visible = r.view(1, 1, bs, 1) < visible_rows.unsqueeze(2)                # (B*U, W, bs, Hkv)
    neg = torch.tensor(float(masked_value), dtype=union_cis_rows.dtype, device=dev)
    bias = torch.where(visible.view(B, U, W, bs, H), union_cis_rows.view(B, 1, W, bs, H), neg)
    bias = bias.view(B * U, W * bs, H)
    if not bias.is_contiguous():
        raise AssertionError("bias must be contiguous: the kernel reads stride(0), stride(-2), stride(-1) off it")

    any_vis = (visible_rows > 0).any(-1)                                       # (B*U, W)
    slot_idx = torch.arange(W, dtype=torch.int64, device=dev).unsqueeze(0).expand(BU, W)
    top = torch.where(any_vis, slot_idx, torch.full_like(slot_idx, -1)).max(-1).values
    cache_seqlens = ((top + 1) * bs).to(torch.int32).contiguous()
    cache_batch_idx = torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(U).contiguous()
    return RowsBias(bias=bias, selected=selected, visible_rows=visible_rows,
                    cache_batch_idx=cache_batch_idx, cache_seqlens=cache_seqlens, U=U, block_size=bs)


def rows_bias_from_round(kv_bias_gpu: torch.Tensor, rnd, *, physical_tail: bool,
                         masked_value: float = MASKED) -> RowsBias:
    """The rows bias of one engine round (duck-typed ``verify_step.VerifyRound``:
    ``union.mask``, ``union.layout.mirror_lo/mirror_hi``, ``tail.tail_len_before``,
    ``tail.rolled_over``, ``U``, ``block_size``, ``tail_slot``).

    physical_tail=True: the tail is read at slot ``topk-1`` (what the shipped
    decode reads; the identity gate) -- only for a round WITHOUT rollover.
    physical_tail=False: the mirrors ``W-2``/``W-1`` as Path 1 names them.
    """
    lay = rnd.union.layout
    masks = rnd.union.mask
    if physical_tail:
        if bool(rnd.tail.rolled_over):
            raise ValueError("physical tail addressing needs a round without rollover: after a rollover slot "
                             "topk-1 holds T+1's rows over T's (cache_engine S == 1 body); use the mirrors")
        masks = remap_slots(masks, {int(lay.mirror_lo): int(rnd.tail_slot)})
        return build_rows_bias(kv_bias_gpu, masks, rnd.tail.tail_len_before, rnd.U,
                               tail_slot=int(rnd.tail_slot), block_size=int(rnd.block_size),
                               tail_next_slot=None, masked_value=masked_value)
    return build_rows_bias(kv_bias_gpu, masks, rnd.tail.tail_len_before, rnd.U,
                           tail_slot=int(lay.mirror_lo), block_size=int(rnd.block_size),
                           tail_next_slot=int(lay.mirror_hi), masked_value=masked_value)


def rows_attention_args(q: torch.Tensor, k_gpu: torch.Tensor, v_gpu: torch.Tensor, rb: RowsBias,
                        num_splits: int) -> RowsArgs:
    """Every shape/stride/dtype fact the kernel relies on, asserted where the
    arguments are produced (module docstring: KERNEL FACTS, the narrowed view)."""
    if k_gpu.dim() != 4:
        raise ValueError("k_gpu must be (B, W*bs, Hkv, D), got %s" % (tuple(k_gpu.shape),))
    B, rows, Hkv, D = k_gpu.shape
    if tuple(v_gpu.shape) != (B, rows, Hkv, D):
        raise ValueError("v_gpu %s != k_gpu %s" % (tuple(v_gpu.shape), tuple(k_gpu.shape)))
    if not (k_gpu.is_contiguous() and v_gpu.is_contiguous()):
        # the kernel takes k.stride(0) as the per-request batch stride (flash_api.cpp:76-78)
        raise ValueError("k_gpu/v_gpu must be the contiguous allocation")
    if q.dim() != 3 or q.shape[2] != D:
        raise ValueError("q must be (B*U, Hq, D=%d), got %s" % (D, tuple(q.shape)))
    BU, Hq, _ = q.shape
    U = int(rb.U)
    if U < 1 or BU != B * U:
        raise ValueError("q has %d rows but B*U = %d*%d" % (BU, B, U))
    if Hq % Hkv != 0:
        raise ValueError("Hq=%d is not a multiple of Hkv=%d" % (Hq, Hkv))
    if D % 8 != 0:
        # the ngroups swap (flash_api.cpp:347) needs head_size_og % 8 == 0; without it bidh is a q head
        raise ValueError("head_dim %d must be a multiple of 8" % D)
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    bias = rb.bias
    if tuple(bias.shape) != (BU, rows, Hkv):
        raise ValueError("rows bias %s != (B*U, W*bs, Hkv) = %s" % (tuple(bias.shape), (BU, rows, Hkv)))
    if not bias.is_contiguous():
        raise ValueError("rows bias must be contiguous")
    if not (q.dtype == k_gpu.dtype == v_gpu.dtype == bias.dtype):
        raise TypeError("q/k/v/bias dtypes differ: %s %s %s %s" % (q.dtype, k_gpu.dtype, v_gpu.dtype, bias.dtype))
    if not (q.device == k_gpu.device == v_gpu.device == bias.device == rb.cache_seqlens.device == rb.cache_batch_idx.device):
        raise ValueError("q/k/v/bias/cache_seqlens/cache_batch_idx are not on one device")
    cbi = rb.cache_batch_idx
    if cbi.dtype != torch.int32 or tuple(cbi.shape) != (BU,) or not cbi.is_contiguous():
        raise ValueError("cache_batch_idx must be int32 contiguous (B*U,), got %s %s" % (cbi.dtype, tuple(cbi.shape)))
    if int(cbi.min()) < 0 or int(cbi.max()) >= B:
        raise ValueError("cache_batch_idx must index requests 0..B-1=%d" % (B - 1))
    csl = rb.cache_seqlens
    if csl.dtype != torch.int32 or tuple(csl.shape) != (BU,) or not csl.is_contiguous():
        raise ValueError("cache_seqlens must be int32 contiguous (B*U,), got %s %s" % (csl.dtype, tuple(csl.shape)))
    if int(csl.min()) < 0 or int(csl.max()) > rows:
        raise ValueError("cache_seqlens must lie in [0, W*bs=%d]" % rows)
    num_splits = int(num_splits)
    if not (0 <= num_splits <= 128):
        raise ValueError("num_splits=%d outside [0, 128] (flash_api.cpp: num_splits > 128 not supported)" % num_splits)

    # THE NARROWED VIEW (module docstring): size(0) == kcache.size(0) for flash_api.cpp:366,
    # the batch stride the kernel multiplies bidb by, and storage for every row it addresses.
    bias_view = bias[:B]
    if bias_view.shape[0] != B or bias_view.stride(0) != rows * Hkv or bias_view.stride(-1) != 1 \
            or bias_view.stride(-2) != Hkv:
        raise AssertionError("bias[:B] strides %s; the kernel needs (W*bs*Hkv, Hkv, 1)" % (bias_view.stride(),))
    if bias_view.data_ptr() != bias.data_ptr() or bias.numel() != BU * rows * Hkv:
        raise AssertionError("bias[:B] must alias a (B*U, W*bs, Hkv) storage: the kernel reads rows up to B*U-1")
    return RowsArgs(q=q.unsqueeze(1), k_cache=k_gpu, v_cache=v_gpu, bias_view=bias_view,
                    cache_seqlens=csl, cache_batch_idx=cbi, num_splits=num_splits)


def rows_attention(q: torch.Tensor, k_gpu: torch.Tensor, v_gpu: torch.Tensor, rb: RowsBias,
                   num_splits: int) -> torch.Tensor:
    """The shipped decode kernel over B*U rows. Returns ``(B*U, Hq, D)`` in the
    store dtype. GPU only; ``num_splits`` explicit (0 = the library heuristic,
    which depends on the ROW count: identity to the decode needs an explicit value)."""
    a = rows_attention_args(q, k_gpu, v_gpu, rb, num_splits)
    # the SHIPPED entry, unchanged: nosa_llama.py:42 binds this same function as flash_attn_nosa_with_kvcache
    from flash_attn_nosa import flash_attn_with_kvcache
    out = flash_attn_with_kvcache(
        a.q, a.k_cache, a.v_cache, a.bias_view,
        cache_seqlens=a.cache_seqlens, cache_batch_idx=a.cache_batch_idx, num_splits=a.num_splits,
    )
    BU, _, Hq, D = a.q.shape
    if tuple(out.shape) != (BU, 1, Hq, D):
        raise AssertionError("kernel output %s != (B*U, 1, Hq, D) = %s" % (tuple(out.shape), (BU, 1, Hq, D)))
    return out.squeeze(1)


def reference_rows_attention(q: torch.Tensor, k_gpu: torch.Tensor, v_gpu: torch.Tensor, rb: RowsBias,
                             scale: float, dtype: torch.dtype, gather: bool, return_p: bool = False):
    """Pure-torch twin of ``rows_attention`` (module docstring, last paragraph).
    q ``(B*U, Hq, D)``; returns ``(B*U, Hq, D)`` fp32 (NaN for a row/head with no
    visible column, as ``reference_attention`` does); with ``return_p`` also the
    dense probabilities ``(B*U, Hq, W*bs)`` fp32 (0 beyond the row's extent)."""
    B, rows, Hkv, D = k_gpu.shape
    BU, Hq, _ = q.shape
    g = Hq // Hkv
    bs = int(rb.block_size)
    out = torch.full((BU, Hq, D), float("nan"), dtype=torch.float32, device=q.device)
    P = torch.zeros((BU, Hq, rows), dtype=torch.float32, device=q.device) if return_p else None
    r_in = torch.arange(bs, dtype=torch.int64, device=q.device)
    for r in range(BU):
        b = int(rb.cache_batch_idx[r])
        n_ext = int(rb.cache_seqlens[r])
        qf = q[r].to(dtype).float()                                     # (Hq, D)
        for h in range(Hkv):
            kf = k_gpu[b, :, h, :].to(dtype).float()                     # (W*bs, D)
            vf = v_gpu[b, :, h, :].to(dtype).float()
            bf = rb.bias[r, :, h].to(dtype).float()                      # (W*bs,)
            if gather:
                vis = (r_in.view(1, bs) < rb.visible_rows[r, :, h].view(-1, 1)).reshape(-1)
                cols = vis.nonzero().squeeze(-1)                         # ascending flat rows
            else:
                cols = torch.arange(n_ext, dtype=torch.int64, device=q.device)
            if cols.numel() == 0:
                continue
            for j in range(h * g, (h + 1) * g):
                s = qf[j:j + 1] @ kf[cols].t()                           # (1, n) fp32 accumulate
                s = (s + bf[cols].unsqueeze(0)) * scale                  # flash_fwd_kernel.h:339-341 then :380
                p = torch.softmax(s, dim=-1)                             # fp32
                if P is not None:
                    P[r, j, cols] = p[0]
                p = p.to(dtype).float()                                  # P rounded for P.V
                o = p @ vf[cols]                                         # (1, D) fp32 accumulate
                out[r, j] = o.to(dtype).float()[0]
    return (out, P) if return_p else out
