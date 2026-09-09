"""EXHAUSTIVE snapshot/restore of everything a NOSA decode step mutates.

WHY IT EXISTS. The acceptance pilot must roll a draft forward K tokens from a
VERIFIED prefix, let it accumulate its own approximate KV, and then verify with
a target that has NEVER SEEN A BYTE OF THAT KV. Restoring this snapshot is the
scratch boundary: after a restore, not one approximate byte survives.

WHY A SNAPSHOT AND NOT THREE ENGINES. One of the two design notes for this
pilot asked for engine_T / engine_D / engine_V as three deep copies, so that
"the draft's tensors are never passed to a target forward" is structural. That
design CONTAINS this one: engine_V still has to be re-synchronised from
engine_T at the start of every round, and that synchronisation is exactly the
restore below. The copies also cost ~800 MB of PINNED host memory per engine at
L=16128 B=1 (`_k_cpu` is (B, S+8192, H, D) bf16, cache_engine.py:171-172), and
NOSA's warm-up buffers and its captured pooling graph are on the MODEL, not the
engine (nosa_llama.py:777-793, :448-476), so they are shared whichever design
is used. The snapshot is the same guarantee for a third of the memory, and --
this is the part that matters -- it is PROVABLE rather than asserted: run one
target step from the verified prefix to get L1, restore, run the whole draft
rollout, restore, run the same target step again to get L2, and require
torch.equal(L1, L2) on the fp32 logits. That single gate covers every item in
the list below, including the ones that fire only every 16 or every 64 steps.

THE LIST, read off the decode path rather than remembered.

CacheEngine, one per layer (cache_engine.py:560-567):
  _k_gpu, _v_gpu, _kv_bias_gpu   tail write :326-330, host gathers :344-345
  _block_map                     :340, and :355 when the tail block fills
  _new_block_map_buf, _load_mask fully rewritten by diff each step, kept anyway
  _cache_lens                    :322 and :357
  seq_length                     :321          (python int)
  _tail_block_len_on_gpu         :333, :356    (python int)
  _k_cpu / _v_cpu, a WINDOW      :352-353. THE ONE THAT SILENTLY POISONS A RUN.
                                 It fires only when a rollout crosses a 64-token
                                 boundary, so a short smoke test never sees it.
                                 The window saved is [seq-2*block, seq+2*block),
                                 which strictly contains every row the write-back
                                 can touch while the sequence advances by at most
                                 one block.

InfLLMv2CacheLayer:
  compress_k_cache_varlen        REBOUND by torch.cat every 16 steps (:619), so
                                 the restore rebinds rather than copies when the
                                 shape has changed
  cached_compressed_cu_seqlens   in-place += :620
  cached_compressed_max_seqlen   :621          (python int)
  no_compress_k_cache            in-place :637-638, :642
  no_compress_k_len              :639, :643    (python int)
  compressed_cis                 in-place :711
  comp_cis_len                   :679          (python int)
  tail_cis                       in-place :704, :713
  tail_cis_len                   :705, :714    (python int)
  total_cis                      in-place :732
  cis_len                        :733          (python int)
  seq_length                     :653          (python int)

InfLLMv2Cache:
  _seen_tokens                   :757          (python int)

DELIBERATELY EXCLUDED, and this is not an oversight. The per-step transient
buffers -- pooling_buf_all, max_pooling_buf, topk_*, mask_buf, score_buf,
compressed_cis_buf -- are recomputed from scratch every step
(nosa_llama.py:592-593 copies into them). score_buf in particular is ALIASED
into the captured CUDA graph at nosa_llama.py:433 and must keep its identity;
`assert_transients_intact` checks exactly that after a restore. The driver-side
`position_ids` is the caller's business and the caller restores it.
"""
from __future__ import annotations

import hashlib

import torch

_ENGINE_TENSORS = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map",
                   "_new_block_map_buf", "_load_mask", "_cache_lens")
_ENGINE_SCALARS = ("seq_length", "_tail_block_len_on_gpu", "_tail_block_idx_on_gpu")
_LAYER_TENSORS = ("compress_k_cache_varlen", "cached_compressed_cu_seqlens",
                  "no_compress_k_cache", "compressed_cis", "tail_cis", "total_cis")
_LAYER_SCALARS = ("cached_compressed_max_seqlen", "no_compress_k_len",
                  "comp_cis_len", "tail_cis_len", "cis_len", "seq_length")


def _sync_host_window():
    """THE HOST WINDOW IS THE ONE PART OF THE SNAPSHOT THAT IS NOT STREAM-ORDERED.

    cache_engine.py:352-353 issues the tail write-back as
    `self._k_cpu[...].copy_(self._k_gpu[...], non_blocking=True)` into PINNED
    host memory, which is a genuinely asynchronous device->host copy. take() and
    restore() touch `_k_cpu`/`_v_cpu` with a plain CPU memcpy, which is ordered
    against nothing. A take() that races the last verified step's write-back
    captures pre-write-back bytes and the next restore then REVERTS a legitimate
    verified write; a restore that races a draft's write-back lets approximate
    draft bytes land after the scratch boundary. Both are silent.

    The last synchronising point inside a decode step is the availability hook's
    device->host copy (avail_policy.on_diff), which happens BEFORE that layer's
    gathers and write-back, so layer 31's write-back is still enqueued when
    decode_inference returns. One synchronize per take/restore costs nothing on
    an accuracy pilot and removes the race.
    """
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


def _grab(obj, names, into: dict):
    for n in names:
        if not hasattr(obj, n):
            continue
        v = getattr(obj, n)
        if v is None:
            into[n] = None
        elif torch.is_tensor(v):
            dst = into.get(n)
            if torch.is_tensor(dst) and dst.shape == v.shape and dst.dtype == v.dtype:
                dst.copy_(v)
            else:
                into[n] = v.detach().clone()
        else:
            into[n] = v


def _put(obj, names, frm: dict):
    for n in names:
        if n not in frm:
            continue
        saved = frm[n]
        if saved is None:
            setattr(obj, n, None)
            continue
        if not torch.is_tensor(saved):
            setattr(obj, n, saved)
            continue
        cur = getattr(obj, n, None)
        if (torch.is_tensor(cur) and cur.shape == saved.shape
                and cur.dtype == saved.dtype and cur.device == saved.device):
            cur.copy_(saved)          # keep buffer identity where we can
        else:
            setattr(obj, n, saved.clone())


class CacheSnapshot:
    """Preallocated once per document and reused, so a restore is a fixed set of
    copy_ calls with no allocation on the steady path."""

    def __init__(self, cache, host_margin_blocks: int = 2):
        self.cache = cache
        self.host_margin_blocks = host_margin_blocks
        self.layers: list = []
        self.top: dict = {}
        self.taken = False

    # -- host window -----------------------------------------------------------
    def _host_window(self, eng):
        blk = eng.block_size
        if eng._k_cpu is None:
            return (0, 0)
        n = eng._k_cpu.shape[1]
        m = self.host_margin_blocks * blk
        lo = max(0, eng.seq_length - m)
        hi = min(n, eng.seq_length + m)
        return (lo, hi)

    def take(self):
        cache = self.cache
        _sync_host_window()
        if not self.layers:
            self.layers = [dict(engine={}, layer={}) for _ in cache.layers]
        for i, lay in enumerate(cache.layers):
            slot = self.layers[i]
            eng = lay.cache_engine
            _grab(eng, _ENGINE_TENSORS, slot["engine"])
            _grab(eng, _ENGINE_SCALARS, slot["engine"])
            lo, hi = self._host_window(eng)
            slot["engine"]["_host_lo"] = lo
            slot["engine"]["_host_hi"] = hi
            if hi > lo:
                for name in ("_k_cpu", "_v_cpu"):
                    src = getattr(eng, name)[:, lo:hi]
                    dst = slot["engine"].get(name)
                    if torch.is_tensor(dst) and dst.shape == src.shape:
                        dst.copy_(src)
                    else:
                        slot["engine"][name] = src.detach().clone()
            _grab(lay, _LAYER_TENSORS, slot["layer"])
            _grab(lay, _LAYER_SCALARS, slot["layer"])
        self.top["_seen_tokens"] = cache._seen_tokens
        self.taken = True
        return self

    def restore(self):
        if not self.taken:
            raise RuntimeError("restore() before take(): there is no verified state to go back to")
        cache = self.cache
        for i, lay in enumerate(cache.layers):
            slot = self.layers[i]
            eng = lay.cache_engine
            _put(eng, _ENGINE_TENSORS, slot["engine"])
            _put(eng, _ENGINE_SCALARS, slot["engine"])
            lo, hi = slot["engine"]["_host_lo"], slot["engine"]["_host_hi"]
            if hi > lo:
                for name in ("_k_cpu", "_v_cpu"):
                    getattr(eng, name)[:, lo:hi].copy_(slot["engine"][name])
            _put(lay, _LAYER_TENSORS, slot["layer"])
            _put(lay, _LAYER_SCALARS, slot["layer"])
        cache._seen_tokens = self.top["_seen_tokens"]
        _sync_host_window()
        return self


def transient_ids(model) -> dict:
    """Identity of every buffer that must NOT be restored. score_buf is aliased
    into the captured pooling graph (nosa_llama.py:433, replayed at :597); if a
    restore ever rebinds it the graph writes into a dead tensor and the
    selection is silently wrong."""
    out = {}
    for i, lay in enumerate(getattr(model, "layers", [])):
        for n in ("score_buf", "compressed_cis_buf"):
            v = getattr(lay, n, None)
            if v is not None:
                out[(i, n)] = id(v)
    for n in ("pooling_buf_all", "max_pooling_buf", "max_pooling_buf_cis",
              "topk_buf_val", "topk_buf_indices", "topk_val_buf_q",
              "topk_idx_buf_q", "topk_val_buf", "topk_idx_buf", "mask_buf"):
        v = getattr(model, n, None)
        if v is not None:
            out[(-1, n)] = id(v)
    return out


def assert_transients_intact(model, before: dict):
    now = transient_ids(model)
    bad = [k for k in before if before[k] != now.get(k)]
    if bad:
        raise AssertionError(
            "a restore rebound a buffer that the captured pooling graph aliases: %r" % (bad,))
    if not getattr(model, "has_buffers", False):
        raise AssertionError(
            "model.has_buffers went False: the next decode would re-enter the warm-up "
            "path and RECAPTURE the pooling graph against different buffers "
            "(nosa_llama.py:777, :793), so nothing after it is comparable")


def host_kv_digest(cache, upto: int, frm: int = 0) -> str:
    """sha256 of the host K/V over rows [frm, upto) for every layer.

    THE RANGE MATTERS AND [0, L) IS THE WRONG ONE. During decode the only writer
    of `_k_cpu`/`_v_cpu` is the tail write-back at cache_engine.py:352-353, which
    writes rows [seq_length - 64, seq_length) with seq_length a multiple of 64
    and STRICTLY GREATER than the prompt length L; the host gathers are
    host->device only (flash_h2d_mask.py:44-60). A digest over [0, L) is
    therefore constant by construction: it can never fail, so it gives false
    assurance for exactly the leak it names.

    The window that CAN move is the one the snapshot itself saves and restores,
    [seq - 2*block, seq + 2*block). Digesting [L - block, L + 2*block) covers the
    first rows a write-back could ever reach AND the rows the restore writes
    back, so a restore that mis-sized its window, or a draft write-back that
    escaped it, changes the digest.

    NOTE THE STEP BUDGET. From a block-aligned prompt the write-back fires only
    on the 64th decode step of sequence advance, and budget_or_die refuses any
    cell that reaches step 64, so within this pilot's legal configurations the
    write-back cannot fire at all. What this digest actually proves there is that
    the host window save/restore corrupted nothing -- which is a real property,
    and is stated rather than dressed up as write-back coverage.
    """
    h = hashlib.sha256()
    chunk = 4096                       # rows, so no 800 MB temporary is built
    frm = max(0, int(frm))
    for lay in cache.layers:
        eng = lay.cache_engine
        for name in ("_k_cpu", "_v_cpu"):
            t = getattr(eng, name)
            if t is None:
                continue
            hi_cap = min(int(upto), t.shape[1])
            for lo in range(frm, hi_cap, chunk):
                hi = min(lo + chunk, hi_cap)
                h.update(t[:, lo:hi].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
