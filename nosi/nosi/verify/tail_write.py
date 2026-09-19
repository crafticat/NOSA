"""Tail write for one verify round of U tokens, pure torch.

This is the twin of the S == 1 tail path of
``cache_engine.CacheEngine.decode_update_has_kv_bias`` (cache_engine.py:335-412),
replayed once per token, plus the 2-slot TAIL MIRROR that the verify kernel
needs. Spec (retroinfer-eval
``docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md``): section 1
(requirement R1), section 2(a) (the per-token replay), section 2(b) (the mirror).

Line citations below are into ``nosi/nosi/cache_engine.py`` of this fork.

LAYOUT of one request's rows (dim 1 of ``_k_gpu``/``_v_gpu``/``_kv_bias_gpu``;
one SLOT = ``block_size`` rows; W = allocated slots = ``_k_gpu.shape[1] // block_size``):

    slot 0 .. topk-2      attended window; block ids in ``_block_map[h, b, :topk-1]``
    slot topk-1  (63)     LIVE TAIL, block id T = ``_block_map[h, b, topk-1]``;
                          ``_tail_block_len_on_gpu`` rows are valid
                          (``_tail_block_idx_on_gpu = topk - 1``: :312, :323;
                          write position :349)
    slot topk .. W-3      victim pool (P slots, :202) then ROUND REGION (R slots,
                          spec 2b; see union_store.py)
    slot W-2              TAIL MIRROR of block T  (spec 2b)
    slot W-1              TAIL MIRROR of block T+1, used only after a rollover
                          inside the round (spec 2b)

WHY A MIRROR. The verify kernel's causal mask (IV2 mask.h:181-185) lets query u
read flat rows ``< seqused_k - U + u + 1``, so the U new tokens must be the LAST
U rows of the request's sequence (R1, spec section 1). Slot 63 is last only
inside the decode's 64-slot window; union slots live at 64+. The mirror puts a
copy of the tail at the END of the allocation, and the verify masks name W-2 /
W-1 instead of 63 (union_store.py).

WHAT ONE TOKEN DOES (the S == 1 body, replayed verbatim by ``_write_one_token``
and ``_write_back_full_tail`` below):
    seq_length += 1                                              :345
    _cache_lens[...] += 1                                        :346
    _tail_write_pos = _tail_block_idx_on_gpu * block_size
                      + _tail_block_len_on_gpu                   :349
    _k_gpu / _v_gpu [:, pos:pos+1] <- token                      :350-351
    _kv_bias_gpu[:, pos:pos+1] <- kv_bias[:, seq_length-1:seq_length]  :354
    _tail_block_len_on_gpu += 1                                  :357
    tail_full = _tail_block_len_on_gpu == block_size             :359
    if tail_full:                                                :401
        host rows [seq_length - block_size, seq_length) <- slot 63   :402-405
        _block_map[..., _tail_block_idx_on_gpu] += 1             :407
        _tail_block_len_on_gpu = 0                               :408
        _cache_lens[...] = (topk-1)*block_size + 0               :409
The MIRROR write per token (spec 2b): the same row is also written at
``(W-2)*block_size + tail_len_0 + i`` before the rollover and at
``(W-1)*block_size + j`` after it; at round start the first ``tail_len_0`` rows
of slot 63 are copied to W-2.

``seqused_k`` (spec section 1 table, 2b). Spec 2(b) writes ``W_used = W-1`` if
the round rolled over else ``W-2`` and ``seqused_k = (W_used-1)*64 +
tail_len_after``; read literally that puts the end of the live block inside
W-2 in the rolled case, which contradicts the same paragraph's layout (T at
W-2, T+1 at W-1) and its own boundary case (``tail_len_after == 0`` ->
``max_block_idx = W-1`` excludes W-1, i.e. ``seqused_k = (W-1)*64``). This
module takes ``W_used`` as the INDEX of the last used slot:
``seqused_k = last_used_slot * block_size + tail_len_after`` with
``last_used_slot = W-1`` if the round rolled over (also when it rolled on its
last token: T+1 is then empty and excluded by ``max_block_idx``) else ``W-2``.

The decode's S == 1 body itself is not touched (P = 0 byte-for-byte promise,
cache_engine.py:59-62; retroinfer-eval tests/test_nosi_fork_guards.py). The
engine's ``verify_round_update`` (GPU, later) calls ``write_tail`` in place of
lines :345-357 and :401-409.

``kv_bias`` is the layer's ``total_cis`` table, shape ``(total_bsz, L, H)``
(cache_engine.py:768, returned at :782), indexed exactly as :354 does.

No CUDA, no extension import: this module must stay importable on CPU.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import torch


class TailWriteResult(NamedTuple):
    """What one round wrote and where; everything the union/mask builder needs."""

    U: int
    tail_len_before: int
    tail_len_after: int
    # True if some token of the round filled block T (:401 ran), including a
    # fill on the round's last token
    rolled_over: bool
    # index of the FIRST token that landed in block T+1 (slot W-1); == U when no
    # token did (no rollover, or a rollover on the last token)
    rollover_at: int
    # flat row (dim 1) each token was written to in the live tail slot (63)
    tail_rows: Tuple[int, ...]
    # flat row each token was written to in the mirror (W-2 before, W-1 after)
    mirror_rows: Tuple[int, ...]
    mirror_lo: int  # slot W-2
    mirror_hi: int  # slot W-1
    # W-1 if the round rolled over, else W-2
    last_used_slot: int
    # last_used_slot * block_size + tail_len_after: the per-request
    # ``seqused_k`` of the verify call (see the module docstring)
    seqused_k: int


def causal_row_limits(seqused_k: int, U: int) -> Tuple[int, ...]:
    """Exclusive flat-row bound per query under the kernel's causal mask.

    IV2 mask.h:181-185 with ``orig_row_idx = u``, ``orig_max_seqlen_q = U`` and
    ``seqlen_k = seqused_k``: query u reads rows ``< seqused_k - U + u + 1``.
    """
    if U < 1 or seqused_k < U:
        raise ValueError("causal_row_limits: need 1 <= U <= seqused_k, got U=%d seqused_k=%d" % (U, seqused_k))
    return tuple(seqused_k - U + u + 1 for u in range(U))


def _write_one_token(engine, key_states, value_states, kv_bias):
    """cache_engine.py:345-359, verbatim with ``self`` -> ``engine``; one token ``(B, 1, H, D)``.

    Returns the row written and whether the block is now full.
    """
    engine.seq_length += 1                                                                                   # :345
    engine._cache_lens[...] += 1                                                                             # :346
    _tail_write_pos = engine._tail_block_idx_on_gpu * engine.block_size + engine._tail_block_len_on_gpu      # :349
    engine._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)           # :350
    engine._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)         # :351
    engine._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, engine.seq_length-1:engine.seq_length, :], non_blocking=True)  # :354
    engine._tail_block_len_on_gpu += 1                                                                       # :357
    tail_full = engine._tail_block_len_on_gpu == engine.block_size                                           # :359
    return int(_tail_write_pos), tail_full


def _write_back_full_tail(engine):
    """cache_engine.py:402-409, verbatim with ``self`` -> ``engine``; the body of ``if tail_full:`` (:401)."""
    tail_block_base_pos = engine._tail_block_idx_on_gpu * engine.block_size                                 # :402
    cpu_block_base_pos = engine.seq_length - engine.block_size                                               # :403
    engine._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+engine.block_size, :, :].copy_(engine._k_gpu[:, tail_block_base_pos:tail_block_base_pos+engine.block_size, :, :], non_blocking=True)  # :404
    engine._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+engine.block_size, :, :].copy_(engine._v_gpu[:, tail_block_base_pos:tail_block_base_pos+engine.block_size, :, :], non_blocking=True)  # :405
    engine._block_map[..., engine._tail_block_idx_on_gpu] += 1                                               # :407
    engine._tail_block_len_on_gpu = 0                                                                        # :408
    engine._cache_lens[...] = (engine.topk - 1) * engine.block_size + engine._tail_block_len_on_gpu          # :409


def write_tail(engine, key_states, value_states, kv_bias, mirror_slot: int) -> TailWriteResult:
    """Write U new tokens into the live tail and its mirror; returns positions.

    ``engine`` is duck-typed: a ``CacheEngine`` or any object carrying the
    attributes named in ``_write_one_token`` / ``_write_back_full_tail`` with
    the engine's meaning. They are MUTATED exactly as the S == 1 body mutates
    them.

    key_states, value_states: ``(B, U, H, D)`` (spec 2a: ``k (B,U,H,D)``).
    kv_bias: ``(B, L, H)``, the ``total_cis`` table; row ``seq_length - 1``
        after the increment is the token's row (:354).
    mirror_slot: the slot index of mirror W-2. Passed explicitly and checked
        here (guard where the value is produced): it must be ``W - 2`` and lie
        outside the decode's window.
    """
    k_gpu = engine._k_gpu
    v_gpu = engine._v_gpu
    kv_bias_gpu = engine._kv_bias_gpu
    bs = int(engine.block_size)
    topk = int(engine.topk)
    tail_slot = int(engine._tail_block_idx_on_gpu)

    if key_states.dim() != 4:
        raise ValueError("key_states must be (B, U, H, D), got %s" % (tuple(key_states.shape),))
    B, U, H, D = key_states.shape
    if tuple(value_states.shape) != (B, U, H, D):
        raise ValueError("value_states %s != key_states %s" % (tuple(value_states.shape), (B, U, H, D)))
    if U < 1:
        raise ValueError("write_tail: U must be >= 1, got %d" % U)
    if k_gpu.shape[0] != B or k_gpu.shape[2] != H or k_gpu.shape[3] != D:
        raise ValueError("_k_gpu %s does not match key_states %s" % (tuple(k_gpu.shape), (B, U, H, D)))
    if k_gpu.shape[1] % bs != 0:
        raise ValueError("_k_gpu has %d rows, not a multiple of block_size %d" % (k_gpu.shape[1], bs))
    W = k_gpu.shape[1] // bs
    if tail_slot != topk - 1:
        # cache_engine.py:312, :323: the tail is always the last window slot
        raise ValueError("_tail_block_idx_on_gpu=%d but topk-1=%d" % (tail_slot, topk - 1))
    mirror_lo = int(mirror_slot)
    mirror_hi = mirror_lo + 1
    if mirror_lo != W - 2:
        raise ValueError("mirror_slot=%d must be W-2=%d (the mirror sits at the END of the allocation, spec 2b)" % (mirror_lo, W - 2))
    if mirror_lo < topk:
        raise ValueError("mirror_slot=%d lies inside the decode window (topk=%d)" % (mirror_lo, topk))
    tail_len_0 = int(engine._tail_block_len_on_gpu)
    if not (0 <= tail_len_0 < bs):
        raise ValueError("_tail_block_len_on_gpu=%d must be in [0, block_size=%d)" % (tail_len_0, bs))
    if tail_len_0 + U >= 2 * bs:
        # a second rollover inside one round would need a third mirror slot
        raise ValueError("tail_len %d + U %d >= 2*block_size %d: more than one rollover per round is not supported" % (tail_len_0, U, 2 * bs))
    seq0 = int(engine.seq_length)
    if kv_bias.shape[0] != B or kv_bias.shape[1] < seq0 + U or kv_bias.shape[2] != H:
        raise ValueError("kv_bias %s must be (B=%d, >= seq_length+U=%d, H=%d)" % (tuple(kv_bias.shape), B, seq0 + U, H))
    if engine._k_cpu.shape[1] < seq0 + U:
        raise ValueError("host window has %d rows, need %d" % (engine._k_cpu.shape[1], seq0 + U))

    # Round start (spec 2b): the first tail_len_0 rows of slot 63 go to mirror W-2,
    # so that mirror W-2 is a complete copy of block T after the round.
    _tb = tail_slot * bs
    _mb = mirror_lo * bs
    k_gpu[:, _mb:_mb + tail_len_0].copy_(k_gpu[:, _tb:_tb + tail_len_0], non_blocking=True)
    v_gpu[:, _mb:_mb + tail_len_0].copy_(v_gpu[:, _tb:_tb + tail_len_0], non_blocking=True)
    kv_bias_gpu[:, _mb:_mb + tail_len_0].copy_(kv_bias_gpu[:, _tb:_tb + tail_len_0], non_blocking=True)

    rolled = False
    rollover_at = U
    tail_rows = []
    mirror_rows = []
    for i in range(U):
        tail_len_i = int(engine._tail_block_len_on_gpu)   # 0-based row inside the live block
        pos, tail_full = _write_one_token(engine, key_states[:, i:i+1], value_states[:, i:i+1], kv_bias)
        tail_rows.append(pos)

        # the mirror row of this token (spec 2b)
        if not rolled:
            mrow = mirror_lo * bs + tail_len_i   # block T:   (W-2)*64 + tail_len_0 + i
        else:
            mrow = mirror_hi * bs + tail_len_i   # block T+1: (W-1)*64 + j
            if rollover_at == U:
                rollover_at = i
        engine._k_gpu[:, mrow:mrow+1, :, :].copy_(key_states[:, i:i+1], non_blocking=True)
        engine._v_gpu[:, mrow:mrow+1, :, :].copy_(value_states[:, i:i+1], non_blocking=True)
        engine._kv_bias_gpu[:, mrow:mrow+1, :].copy_(kv_bias[:, engine.seq_length-1:engine.seq_length, :], non_blocking=True)
        mirror_rows.append(int(mrow))

        if tail_full:                                       # :401
            _write_back_full_tail(engine)
            rolled = True

    tail_len_after = int(engine._tail_block_len_on_gpu)
    last_used_slot = mirror_hi if rolled else mirror_lo
    seqused_k = last_used_slot * bs + tail_len_after
    return TailWriteResult(
        U=U,
        tail_len_before=tail_len_0,
        tail_len_after=tail_len_after,
        rolled_over=rolled,
        rollover_at=rollover_at,
        tail_rows=tuple(tail_rows),
        mirror_rows=tuple(mirror_rows),
        mirror_lo=mirror_lo,
        mirror_hi=mirror_hi,
        last_used_slot=last_used_slot,
        seqused_k=seqused_k,
    )
