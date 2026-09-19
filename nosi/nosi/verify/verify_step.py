"""Path 1 verify call: the InfLLM-v2 prefill varlen block-sparse kernel over
the union store, one query per position, each masked to its own selection.

Spec (retroinfer-eval ``docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md``):
section 1 (the call, its table, constraints C1-C6), section 3 (numerics: the
two-call ``V * exp(cis)`` trick of nosa_llama.py:303-337, reused verbatim).

Two halves. ``build_varlen_args`` is pure torch and runs on CPU (retroinfer-eval
``tests/test_nosi_verify_engine.py``): it turns the engine's WHOLE allocation
and a ``VerifyRound`` into exactly the tensors and scalars the INTERNAL op
takes, every shape/dtype/stride fact asserted where the value is produced.
``varlen_attention`` is the GPU half: it imports
``_wrapped_infllmv2_attn_varlen_forward`` (IV2 infllmv2_sparse_attention.py:141,
``= _infllmv2_attn_varlen_forward`` :51-137) lazily -- the PUBLIC
``infllmv2_attn_varlen_func`` (:399-417) never forwards ``seqused_k``
(``Infllmv2AttnVarlenFunc.forward`` :282-327 drops it) and Path 1 needs it --
and makes the two calls (values scaled by ``exp(cis)``, then ``exp(cis)``
alone as the value) and the quotient in the store dtype, as the prefill does.

WHAT THE KERNEL DOES WITH THESE ARGUMENTS (by reading; line refs into
``dependencies/infllmv2_cuda_impl``):
  q ``(B*U, Hq, D)`` is packed by the wrapper to ``(B*U*g, Hkv, D)``,
  g = Hq/Hkv, and ``cu_seqlens_q``/``max_seqlen_q`` are multiplied by g
  (infllmv2_sparse_attention.py:90-92): one packed token = one 16-row m_block
  = one mask row (kBlockM = 16, csrc/flash_attn/src/kernel_traits.h). The mask
  row of packed token (b, u) is row ``b*U + u`` of ``topk_idx[h]``
  (flash_blockmask.h:26-31 with ``cu_seqlens_q[b]/16 = b*U``); h is the KV head
  (``params.h = Hkv`` after packing). ``topk_idx`` entries are UNION SLOT ids:
  ``topk_to_uint64`` skips -1 (csrc/topk_to_uint64.cuh:39-40) and silently DROPS
  ids ``>= ceil(max_seqlen_k/64)`` (:43-47) -- hence the ``id < W`` guard here.
  k/v are the WHOLE allocation viewed flat ``(B*W*64, Hkv, D)``: ``mha_varlen_fwd``
  reads one ``k_row_stride`` (flash_api.cpp:578-580; block_info.h:34), so a
  narrowed dim-1 view could not be flattened. ``cu_seqlens_k[b] = b*W*64`` is
  request b's first flat row (block_info.h:17, :34; ``is_seqlens_k_cumulative``
  is set at flash_api.cpp:149) and ``seqused_k[b]`` bounds its rows
  (block_info.h:23; checked int32/contiguous/(B,) at flash_api.cpp:637-643).
  The causal rule (mask.h:181-185, ``orig_row_idx = row/16``) lets query u read
  rows ``< seqused_k - U + u + 1``; its ``n_block_max`` is
  ``ceil((seqused_k - U + u + 1)/64)`` (flash_fwd_kernel.h:170-173 divides by
  ``m_block_dim``), so blocks above it are never loaded, and ``n_masking_steps
  = 2`` (:389-391) masks the two highest set blocks, which is exact because
  only the block holding row ``seqused_k - U + u`` is cut. ``causal=True``
  stays on because ``max_seqlen_q = 16U > 1`` (flash_api.cpp:598); the
  ``seqlenq_ngroups_swapped`` path is off (:605, ``num_heads == num_heads_k``
  after packing). ``softmax_scale`` must be given to the internal op (the
  public one defaults it to ``D**-0.5`` at :302-303): ``D**-0.5`` is passed.
  ``max_seqlen_k`` reaches both the packer (:95) and ``num_blocks_n``
  (flash_api.cpp:710) as ONE value (C6): ``W*64``.

The guards that read a device tensor (``bool(...)``) are host syncs: two per
call. Acceptable for the accuracy pilot; a production caller moves them to
the union builder's flags.

No CUDA, no extension import at module level: this module must stay
importable on CPU. The one GPU import is inside ``varlen_attention``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch

# the varlen kernel's n_block_dim (flash_api.cpp:707) and the block size the
# wrapper packs the mask with (infllmv2_sparse_attention.py:95)
KERNEL_BLOCK = 64


class RoundOverflow(RuntimeError):
    """A verify round whose union does not fit the round region, or a selection
    that names a block the position cannot see (union_store: overflow/invalid).
    Spec 2c: the round falls back to sequential verification and the pilot
    counts it. Raised BEFORE any fetch; the tail rows were already written, so
    the caller restores its snapshot."""

    def __init__(self, msg: str, n_new=None, overflow=None, invalid=None):
        super().__init__(msg)
        self.n_new = n_new
        self.overflow = overflow
        self.invalid = invalid


class VerifyRound(NamedTuple):
    """What ``CacheEngine.verify_round_update`` hands to ``verify_attention``."""

    union: Any                  # union_store.UnionRound: mask (Hkv, B*U, K) int32, round_map, n_new, ...
    tail: Any                   # tail_write.TailWriteResult: seqused_k, rollover_at, mirror rows, ...
    seqused_k: torch.Tensor     # (B,) int32 on the engine's device (all requests roll together)
    U: int
    W: int                      # slots in the allocation = _k_gpu.shape[1] // block_size
    block_size: int
    tail_slot: int              # topk - 1; never named by a verify mask (read through its mirror)


class VarlenArgs(NamedTuple):
    """The internal op's arguments (spec section 1 table)."""

    q: torch.Tensor             # (B*U, Hq, D), request-major, position-minor, contiguous
    k: torch.Tensor             # (B*W*bs, Hkv, D): a VIEW of the whole allocation
    v_scaled: torch.Tensor      # (B*W*bs, Hkv, D) = v * exp(kv_bias), store dtype  (call 1)
    v_fake: torch.Tensor        # (B*W*bs, Hkv, D) = exp(kv_bias) broadcast over D   (call 2)
    cu_seqlens_q: torch.Tensor  # (B+1,) int32 = arange * U      (the wrapper multiplies by Hq/Hkv)
    cu_seqlens_k: torch.Tensor  # (B+1,) int32 = arange * W*bs   (flat rows of the allocation)
    seqused_k: torch.Tensor     # (B,) int32
    max_seqlen_q: int           # U
    max_seqlen_k: int           # W*bs
    topk_idx: torch.Tensor      # (Hkv, B*U, K) int32: union slot ids, -1 padded, row b*U + u
    softmax_scale: float        # D ** -0.5
    causal: bool                # True


def build_varlen_args(q, k_gpu, v_gpu, kv_bias_gpu, rnd: VerifyRound) -> VarlenArgs:
    U, W, bs = int(rnd.U), int(rnd.W), int(rnd.block_size)
    if bs != KERNEL_BLOCK:
        raise ValueError("block_size=%d: the varlen kernel's block is %d (flash_api.cpp:707, topk_to_uint64 :95)" % (bs, KERNEL_BLOCK))
    if U < 1:
        raise ValueError("U must be >= 1, got %d" % U)
    if k_gpu.dim() != 4:
        raise ValueError("k_gpu must be (B, W*bs, Hkv, D), got %s" % (tuple(k_gpu.shape),))
    B, rows, Hkv, D = k_gpu.shape
    if rows != W * bs:
        raise ValueError("k_gpu has %d rows but the round says W=%d slots of %d" % (rows, W, bs))
    if tuple(v_gpu.shape) != (B, rows, Hkv, D):
        raise ValueError("v_gpu %s != k_gpu %s" % (tuple(v_gpu.shape), tuple(k_gpu.shape)))
    if tuple(kv_bias_gpu.shape) != (B, rows, Hkv):
        raise ValueError("kv_bias_gpu %s != (B, W*bs, Hkv) = %s" % (tuple(kv_bias_gpu.shape), (B, rows, Hkv)))
    if not (k_gpu.is_contiguous() and v_gpu.is_contiguous() and kv_bias_gpu.is_contiguous()):
        # the flat (total_k, Hkv, D) view needs ONE row stride (flash_api.cpp:578-580)
        raise ValueError("the allocation must be contiguous to be viewed as (B*W*bs, Hkv, D)")
    if q.dim() != 3 or q.shape[0] != B * U or q.shape[2] != D:
        raise ValueError("q must be (B*U=%d, Hq, D=%d), got %s" % (B * U, D, tuple(q.shape)))
    Hq = q.shape[1]
    if Hq % Hkv != 0:
        raise ValueError("Hq=%d is not a multiple of Hkv=%d" % (Hq, Hkv))
    if not q.is_contiguous():
        raise ValueError("q must be contiguous (the wrapper reshapes it into (B*U*g, Hkv, D))")
    if not (q.dtype == k_gpu.dtype == v_gpu.dtype == kv_bias_gpu.dtype):
        raise TypeError("q/k/v/kv_bias dtypes differ: %s %s %s %s" % (q.dtype, k_gpu.dtype, v_gpu.dtype, kv_bias_gpu.dtype))
    if not (q.device == k_gpu.device == v_gpu.device == kv_bias_gpu.device):
        raise ValueError("q/k/v/kv_bias are not on one device")
    mask = rnd.union.mask
    if mask.dtype != torch.int32 or mask.dim() != 3 or tuple(mask.shape[:2]) != (Hkv, B * U):
        raise ValueError("topk_idx must be int32 (Hkv=%d, B*U=%d, K), got %s %s" % (Hkv, B * U, mask.dtype, tuple(mask.shape)))
    if not mask.is_contiguous() or mask.device != k_gpu.device:
        raise ValueError("topk_idx must be contiguous on the allocation's device (topk_to_uint64 reads data_ptr)")
    bad = (mask >= W) | (mask < -1) | (mask == int(rnd.tail_slot))
    if bool(bad.any()):
        raise ValueError("topk_idx names a slot >= W=%d, < -1, or the live tail slot %d (read through its mirror only); "
                         "ids >= ceil(max_seqlen_k/64) are silently dropped by the packer" % (W, int(rnd.tail_slot)))
    seqused_k = rnd.seqused_k
    if seqused_k.dtype != torch.int32 or tuple(seqused_k.shape) != (B,) or not seqused_k.is_contiguous():
        raise ValueError("seqused_k must be int32 contiguous (B=%d,), got %s %s" % (B, seqused_k.dtype, tuple(seqused_k.shape)))
    if seqused_k.device != k_gpu.device:
        raise ValueError("seqused_k must be on the allocation's device")
    want = int(rnd.tail.seqused_k)
    if not (U <= want <= W * bs):
        raise ValueError("seqused_k=%d outside [U=%d, W*bs=%d]" % (want, U, W * bs))
    if bool((seqused_k != want).any()):
        raise ValueError("seqused_k must equal the tail write's %d on every request" % want)

    total_k = B * W * bs
    k = k_gpu.view(total_k, Hkv, D)
    ucis = torch.exp(kv_bias_gpu)                                                          # nosa_llama.py:303, store dtype
    v_scaled = (v_gpu * ucis.unsqueeze(-1)).view(total_k, Hkv, D)                          # :304, rounded per element
    v_fake = ucis.unsqueeze(-1).expand(B, rows, Hkv, D).contiguous().view(total_k, Hkv, D)  # :320 (repeat materialises; so does this)
    dev = k_gpu.device
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev) * U
    cu_k = torch.arange(B + 1, dtype=torch.int32, device=dev) * (W * bs)
    return VarlenArgs(q=q, k=k, v_scaled=v_scaled, v_fake=v_fake, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      seqused_k=seqused_k, max_seqlen_q=U, max_seqlen_k=W * bs, topk_idx=mask,
                      softmax_scale=float(D) ** -0.5, causal=True)


def varlen_attention(a: VarlenArgs) -> torch.Tensor:
    """The two internal-op calls and the quotient (nosa_llama.py:305-337 with
    ``seqused_k``). Returns ``(B*U, Hq, D)`` in the store dtype. GPU only."""
    # the INTERNAL op (infllmv2_sparse_attention.py:141): the public function drops seqused_k
    from infllm_v2.infllmv2_sparse_attention import _wrapped_infllmv2_attn_varlen_forward as _fwd

    def call(v):
        out = _fwd(
            a.q, a.k, v, a.cu_seqlens_q, a.cu_seqlens_k, a.max_seqlen_q, a.max_seqlen_k,
            0.0, a.softmax_scale,
            causal=a.causal, window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, return_softmax=False, block_table=None, leftpad_k=None,
            seqused_k=a.seqused_k, topk_idx=a.topk_idx,
        )
        return out[0]

    out1 = call(a.v_scaled)          # :305-318, values scaled by exp(cis)
    out2 = call(a.v_fake)            # :321-334, exp(cis) as the value: the real denominator
    return out1 / out2[:, :, :1]     # :336-337, in the store dtype


def verify_attention(q, k_gpu, v_gpu, kv_bias_gpu, rnd: VerifyRound) -> torch.Tensor:
    """Path 1 attention for one round: ``(B*U, Hq, D)`` in the store dtype."""
    return varlen_attention(build_varlen_args(q, k_gpu, v_gpu, kv_bias_gpu, rnd))
