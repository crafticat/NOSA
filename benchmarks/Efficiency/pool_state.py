"""VICTIM-POOL STATE for the cache-size interference curve (authorized 2026-10-01, the user via Codex: 'VICTIM-POOL RESTORATION
AND GATING (authorized measurement-harness extension, NOT a production replacement)'). Pure torch and device-agnostic: the same
code runs on the GPU engines and on the fake CPU engines of retroinfer-eval tests/test_cache_curve.py. Nothing here is imported
by the engine; cache_engine.py is NOT edited (the P = 0 promise, cache_engine.py:59-62).

WHY. NOSI_POOL_BLOCKS = P gives usable capacity C = 63 + P 64-token/head groups per (layer, KV head, request): P victim slots at
slot indices topk .. topk+P-1 of the SAME _k_gpu / _v_gpu / _kv_bias_gpu tensors (cache_engine.py:71, :274-297), invisible to
attention (the decode receives views of topk*64 rows, :349-398 and :841). The pool's bookkeeping is four tensors per layer
(cache_engine.py:305-331: _pool_map (H,B,P), _pool_age (H,B,P), _pool_target (H,B,topk), _pool_action (H,B,topk) int8) plus a HOST
int, _pool_stamp_base (:180, :316, incremented by topk on every pooled decode at :814), plus the P*64 pool rows of the three cache
tensors. The light restore of the harness (state_snapshot.CounterSnapshot, state_snapshot.py:248-249 with :74-80) saves NONE of
them, so a resident tick at C > 63 could not be certified before (cache_engine.py:192-198 refuses NOSI_AVAIL with the pool and
verify_alone.py:452-453 refuses POOL_BLOCKS != 0 outright).

WHAT THIS MODULE ADDS (each piece is certified by a gate or a receipt, never assumed):
  * make_pool_snapshots(ss): PoolCounterSnapshot = CounterSnapshot + the four pool tensors (engine_tensors) + the host stamp
    base; post_reference() copies the POST-REFERENCE pool state into the snapshot, exactly as gated() copies the post-step
    _block_map / _new_block_map_buf (cpupack_transport.py gated()). Why post-reference: the flushed reference decode of a gated
    step refills every non-tail slot (miss_control.flush_map), so a block that sat in the pool comes back by MOVE_IN and its pool
    slot is emptied (pool_update_kernel.cu phase 3c+4). Restoring the PRE-reference pool map over the post-reference window would
    name that block twice (window AND pool) = the duplicate-residency state the kernel's invariant forbids. Pool ROWS are never
    snapshotted: a resident tick moves no pool row (every tick's receipt proves 0 pool actions), and the gate digests the rows.
  * reset_pool(e): the post-prefill pool state of prefill_update section 3b (cache_engine.py:311-331), in place: map -1, ages
    -P+q, target -1, action 0, stamp 0, pool rows zero. The restart (PostPrefillSnapshot, which knows nothing of the pool) is
    followed by it; pool_is_fresh() is the proof.
  * repool(e, P): re-initialise ONE engine at its post-prefill state for another pool size while KEEPING the host KV
    (_k_cpu / _v_cpu are allocated at prefill, cache_engine.py:263-264, independent of P). Investigated at file:line:
      - the knob is read once at import (cache_engine.py:71) and only decides (a) whether the nosi_pool extension and
        flash_pool_swap are loaded (:114-123) and (b) the binding of decode_update at construction (:212-215); the slot count is
        self.pool_blocks, read only by prefill_update (:274) and the pooled decode;
      - the GPU tensors are allocated at prefill (:295-297) with _gpu_slots = topk + pool_blocks (:274), the pool state at
        :311-316, the attended views at :390-393;
      - flash_pool_swap derives P from the tensor it is handed (S_GPU // block_size - topk, flash_pool_swap.py) and pool_update
        from _pool_map's last dim (pool_update.cpp), so no P is cached anywhere else.
    So a process imported with NOSI_POOL_BLOCKS = P_max (extension loaded) can serve every P <= P_max: repool re-runs exactly
    the allocation statements of prefill_update for the new P (the window rows [0, topk*64) are copied, including the tail slot
    the prefill wrote; the int32 envelopes of :288-294 are re-asserted), rebinds decode_update exactly as :212-215 would (P = 0
    binds the UPSTREAM decode_update_has_kv_bias, which returns the whole 64-slot tensor, :715) and resets the pool. Certified by
    the restart proof (the P-independent start families equal the post-prefill digest), pool_is_fresh(), and the cross-C check
    (every capture / reference / advance logits sha at C equals C63's).
  * pool_families(e): the gate's pool digest families: 'pool' = sha256 of the four tensors + stamp base + P; 'poolrows' = fp64
    (cpupack_golden) of the P*64 pool rows of K, V and bias. pool_dup(e): the ABSOLUTE consistency check (window ids U pool ids
    pairwise distinct per stream) -- a deterministic bug would pass a golden comparison, never this.
  * natural_accounting(...): the per-C NATURAL plans from the capture archives: H2D loads (the PCIe misses the transport
    replays), pool hits (SWAP / MOVE_IN: device-to-device, NOT PCIe, NOT replayed), parks (MOVE_OUT), evictions (blocks that left
    HBM), D2D block moves, and the pool-aware invariants P1..P5.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

import cpupack_golden as G

POOL_TENSORS = ("_pool_map", "_pool_age", "_pool_target", "_pool_action")
POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3       # pool_update_kernel.cu:84-87, flash_pool_swap.py
POOL_FAMILIES = ("pool", "poolrows")
GROUP_USEFUL = 32768                                                  # K + V of one (head, request, 64-token block), bf16
GROUP_BIAS = 128                                                      # kv_bias of the same: 64 rows x 2 B
TOPK = 64


def pool_blocks(e) -> int:
    return int(getattr(e, "pool_blocks", 0) or 0)


def pool_active(e) -> bool:
    return pool_blocks(e) > 0 and getattr(e, "_pool_map", None) is not None


def capacity_of(P: int, topk: int = TOPK) -> int:
    """Usable capacity C (64-token/head groups per stream) of a pool of P blocks: 63 attended non-tail slots + P."""
    return topk - 1 + int(P)


def pool_of(C: int, topk: int = TOPK) -> int:
    P = int(C) - (topk - 1)
    if P < 0:
        raise ValueError("capacity %d below the 63 attended non-tail slots" % C)
    return P


# ---------------------------------------------------------------------------------------------------------- snapshots
def make_pool_snapshots(ss):
    """(PoolCounterSnapshot,) built on the given state_snapshot module (the engine's own on the GPU; the file-loaded copy in the
    CPU tests), like cpupack_plans.make_int16_trace_class."""

    class PoolCounterSnapshot(ss.CounterSnapshot):
        """CounterSnapshot (state_snapshot.py:215-249) + the pool bookkeeping tensors + the host stamp base (module docstring).
        Engines without a pool (P = 0, the tensors are None or absent) are handled by the base _grab / _put unchanged."""

        engine_tensors = tuple(ss.CounterSnapshot.engine_tensors) + POOL_TENSORS

        @torch.inference_mode()
        def take(self):
            super().take()
            for i, lay in enumerate(self.cache.layers):
                e = lay.cache_engine
                self.layers[i]["engine"]["_pool_stamp_base"] = int(getattr(e, "_pool_stamp_base", 0) or 0)
                self.layers[i]["engine"]["_pool_P"] = pool_blocks(e)
            return self

        @torch.inference_mode()
        def restore(self):
            super().restore()
            for i, lay in enumerate(self.cache.layers):
                e = lay.cache_engine
                slot = self.layers[i]["engine"]
                if slot.get("_pool_P", 0) != pool_blocks(e):
                    raise RuntimeError("PoolCounterSnapshot: layer %d was taken at P=%s, the engine now has P=%d" % (i, slot.get("_pool_P"), pool_blocks(e)))
                if "_pool_stamp_base" in slot:
                    e._pool_stamp_base = slot["_pool_stamp_base"]
            return self

        @torch.inference_mode()
        def post_reference(self):
            """Called by gated() right after the flushed reference decode (cpupack_transport.py gated(), next to the block-map
            copy): the post-reference pool state becomes the state every restore of this step puts back."""
            if getattr(self, "skip_post_reference", False):          # CONTROLS only: the negative control of this very hook
                return
            for i, lay in enumerate(self.cache.layers):
                e = lay.cache_engine
                if not pool_active(e):
                    continue
                slot = self.layers[i]["engine"]
                for n in POOL_TENSORS:
                    slot[n].copy_(getattr(e, n))
                slot["_pool_stamp_base"] = int(e._pool_stamp_base)

    return (PoolCounterSnapshot,)


# ------------------------------------------------------------------------------------------------- the flushed reference
def check_window_pooled(engine):
    """miss_control._window (verify/miss_control.py:145-158) for the POOLED allocation: that function asserts the shipped
    64-slot tensors (B, topk*64, H, D), which a pooled engine never has. Same checks with (topk + P) * 64 rows."""
    m = engine._block_map
    assert m.dim() == 3 and m.dtype == torch.int64, ("_block_map", tuple(m.shape), m.dtype)
    H, B, M = m.shape
    bs = int(engine.block_size)
    tail = int(engine._tail_block_idx_on_gpu)
    assert M == int(engine.topk) and tail == M - 1, (M, engine.topk, tail)
    rows = (M + pool_blocks(engine)) * bs
    for name in ("_k_gpu", "_v_gpu"):
        t = getattr(engine, name)
        assert tuple(t.shape) == (B, rows, H, int(engine.head_dim)), (name, tuple(t.shape), (B, rows, H, engine.head_dim))
    assert tuple(engine._kv_bias_gpu.shape) == (B, rows, H), tuple(engine._kv_bias_gpu.shape)
    assert engine._new_block_map_buf.shape == m.shape and engine._load_mask.shape == m.shape
    if pool_active(engine):
        P = pool_blocks(engine)
        assert tuple(engine._pool_map.shape) == (H, B, P) and tuple(engine._pool_age.shape) == (H, B, P)
        assert tuple(engine._pool_target.shape) == (H, B, M) and tuple(engine._pool_action.shape) == (H, B, M)
    return H, B, M, bs, tail


class PooledMissControl:
    """The miss_control module with flush_map accepting the pooled allocation (gated()'s flushed reference,
    cpupack_transport.py gated(): every non-tail map entry := -1; the pool is NOT flushed: the reference then serves the
    selected blocks it holds by MOVE_IN, which the golden pass does identically). Every other name is the real module's."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, n):
        return getattr(self._real, n)

    def flush_map(self, engine) -> None:
        H, B, M, bs, tail = check_window_pooled(engine)
        engine._block_map[..., :tail] = -1


# ------------------------------------------------------------------------------------------------- reset / repool
@torch.inference_mode()
def reset_pool(e) -> None:
    """prefill_update section 3b (cache_engine.py:311-331) in place: the pool of a freshly prefilled engine."""
    if not pool_active(e):
        if hasattr(e, "_pool_stamp_base"):
            e._pool_stamp_base = 0
        return
    P = pool_blocks(e)
    e._pool_map.fill_(-1)
    e._pool_age.copy_((torch.arange(P, dtype=torch.int64, device=e._pool_age.device) - P).expand_as(e._pool_age))
    e._pool_target.fill_(-1)
    e._pool_action.zero_()
    e._pool_stamp_base = 0
    att = int(e.topk) * int(e.block_size)
    for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
        getattr(e, n)[:, att:].zero_()


@torch.inference_mode()
def pool_is_fresh(e) -> Dict:
    """The proof of reset_pool / repool (and of a fresh prefill): every field equals section 3b's value."""
    P = pool_blocks(e)
    att = int(e.topk) * int(e.block_size)
    rows = int(e._k_gpu.shape[1])
    out = dict(P=P, rows=rows, rows_ok=rows == (int(e.topk) + P) * int(e.block_size), stamp_ok=int(getattr(e, "_pool_stamp_base", 0) or 0) == 0)
    if P == 0:
        out.update(state_ok=all(getattr(e, n, None) is None for n in POOL_TENSORS), rows_zero=True)
    else:
        H, B = e._block_map.shape[:2]
        dev = e._pool_map.device
        age = (torch.arange(P, dtype=torch.int64, device=dev) - P).expand(H, B, P)
        out["state_ok"] = bool(tuple(e._pool_map.shape) == (H, B, P) and bool((e._pool_map == -1).all()) and torch.equal(e._pool_age, age)
                               and bool((e._pool_target == -1).all()) and not bool(e._pool_action.any()))
        out["rows_zero"] = all(not bool(getattr(e, n)[:, att:].ne(0).any()) for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"))
    out["ok"] = bool(out["rows_ok"] and out["stamp_ok"] and out["state_ok"] and out["rows_zero"])
    return out


def _envelopes(B: int, S_cpu: int, slots: int, R: int, HD: int) -> None:
    """cache_engine.py:288-294: the Triton gathers do not cast their program ids, so both sides of the copy stay int32."""
    assert B * S_cpu * HD < 2 ** 31, "host-side int32 gather overflow: B=%d S_cpu=%d -> %d elements" % (B, S_cpu, B * S_cpu * HD)
    assert B * slots * R * HD < 2 ** 31, "GPU-side int32 gather overflow: B=%d topk+p=%d -> %d elements" % (B, slots, B * slots * R * HD)


@torch.inference_mode()
def repool(e, P: int, ext_loaded: bool = True) -> Dict:
    """Re-initialise ONE engine, at its post-prefill state, for pool size P (module docstring). Returns a record."""
    P = int(P)
    if P < 0:
        raise ValueError("P must be >= 0")
    if int(getattr(e, "_verify_slots", 0) or 0) != 0:
        raise RuntimeError("repool: the verify round region is not supported (NOSI_VERIFY_ROUND_SLOTS must be 0)")
    if P > 0 and not ext_loaded:
        raise RuntimeError("repool to P=%d needs the nosi_pool extension: import the engine with NOSI_POOL_BLOCKS > 0" % P)
    topk, R = int(e.topk), int(e.block_size)
    att = topk * R
    B, rows_old = int(e._k_gpu.shape[0]), int(e._k_gpu.shape[1])
    P_old = pool_blocks(e)
    slots = topk + P
    rec = dict(P_old=P_old, P=P, rows_old=rows_old, rows_new=slots * R, realloc=rows_old != slots * R)
    if P > 0:
        HD = int(e._k_gpu.shape[2]) * int(e._k_gpu.shape[3])
        _envelopes(B, int(e._k_cpu.shape[1]), slots, R, HD)
    if rec["realloc"]:
        for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            old = getattr(e, n)
            new = torch.empty((B, slots * R) + tuple(old.shape[2:]), dtype=old.dtype, device=old.device)
            new[:, :att].copy_(old[:, :att])                          # the window, the prefill's tail slot included
            setattr(e, n, new)
            del old
    e.pool_blocks = P
    if P > 0:
        H = int(e._block_map.shape[0])
        dev = e._block_map.device
        e._pool_map = torch.full((H, B, P), -1, dtype=torch.int64, device=dev)
        e._pool_age = (torch.arange(P, dtype=torch.int64, device=dev) - P).expand(H, B, P).contiguous()
        e._pool_target = torch.full((H, B, topk), -1, dtype=torch.int64, device=dev)
        e._pool_action = torch.zeros((H, B, topk), dtype=torch.int8, device=dev)
        e._att_rows = att
        e._k_gpu_att, e._v_gpu_att, e._kv_bias_gpu_att = e._k_gpu[:, :att], e._v_gpu[:, :att], e._kv_bias_gpu[:, :att]
        for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            getattr(e, n)[:, att:].zero_()
        if hasattr(e, "decode_update_has_kv_bias_pool"):
            e.decode_update = e.decode_update_has_kv_bias_pool        # cache_engine.py:212-213
    else:
        for n in POOL_TENSORS:
            setattr(e, n, None)
        e._att_rows = None
        e._k_gpu_att = e._v_gpu_att = e._kv_bias_gpu_att = None
        if hasattr(e, "decode_update_has_kv_bias"):
            e.decode_update = e.decode_update_has_kv_bias              # cache_engine.py:215 (has_kv_bias: NOSA-8B always)
    e._pool_stamp_base = 0
    rec["fresh"] = pool_is_fresh(e)
    return rec


# ------------------------------------------------------------------------------------------------------- digests
def _rows_fp(t: torch.Tensor, lo: int, batch_chunk: int = 8) -> str:
    return G.fp_stream(t[b0:b0 + batch_chunk, lo:] for b0 in range(0, int(t.shape[0]), batch_chunk))


def pool_families(e) -> Dict[str, str]:
    """{'pool': sha256, 'poolrows': fp64} of one engine (module docstring); 'P0' for an engine without a pool."""
    if not pool_active(e):
        return dict(pool="P0", poolrows="P0")
    att = int(e.topk) * int(e.block_size)
    return dict(pool=G.sha_parts([e._pool_map, e._pool_age, e._pool_target, e._pool_action, int(e._pool_stamp_base), pool_blocks(e)]),
                poolrows=",".join(_rows_fp(x, att) for x in (e._k_gpu, e._v_gpu, e._kv_bias_gpu)))


def pool_digest(cache) -> Dict[str, List[str]]:
    per = [pool_families(lay.cache_engine) for lay in cache.layers]
    return {f: [p[f] for p in per] for f in POOL_FAMILIES}


def dup_count(block_map: torch.Tensor, pool_map: Optional[torch.Tensor]) -> int:
    """Pairs of equal non-negative ids inside window U pool, per stream (H, B), summed. 0 = the kernel's invariant
    (pool_update_kernel.cu 'INVARIANT ...: attended-union-pool holds no duplicate block id')."""
    x = block_map if pool_map is None else torch.cat([block_map, pool_map.to(block_map.dtype)], dim=-1)
    s = torch.sort(x, dim=-1).values
    return int(((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).sum())


def pool_dup(cache) -> List[int]:
    out = []
    for lay in cache.layers:
        e = lay.cache_engine
        out.append(dup_count(e._block_map, e._pool_map if pool_active(e) else None))
    return out


def pool_actions_now(engines) -> int:
    pe = [e for e in engines if pool_active(e)]
    return int(sum(int((e._pool_action != POOL_NONE).sum()) for e in pe)) if pe else 0


def actions_receipt(pool_actions: Sequence[torch.Tensor]):
    """Device scalar (no host read): pool actions of the tick just decoded over every pooled layer (0 = no SWAP / MOVE_IN /
    MOVE_OUT = no pool row moved). pool_update rewrites _pool_action for EVERY slot each call (pool_update_kernel.cu phase 0:
    pool_action[base + m] = POOL_NONE), so the tensor holds the tick's own actions."""
    if not pool_actions:
        return None
    return torch.stack([(a != POOL_NONE).sum() for a in pool_actions]).sum()


# ----------------------------------------------------------------------------------------- the natural plans per C
def make_pool_trace_class(T16, engines):
    """The capture's int16 AloneTrace (cpupack_plans.make_int16_trace_class) + a POST-STEP POOL MAP archive: record_mask is
    called by the pooled decode AFTER pool_update and the gathers (cache_engine.py:823; record_pool, the per-slot actions, at
    :815), so the engine's _pool_map is the post-step pool there. Device-to-device, no sync, int16 (ids <= L/64)."""

    class PoolTrace(T16):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.pmap_archive = None

        def record_mask(self, load_mask, block_map):
            super().record_mask(load_mask, block_map)
            if self.cur is None or not self.full or self.layer is None:
                return
            e = engines[self.layer]
            if not pool_active(e):
                return
            pm = e._pool_map
            if self.pmap_archive is None or self.pmap_archive.shape[2:] != tuple(pm.shape):
                self.pmap_archive = torch.full((self.max_steps, self.num_layers) + tuple(pm.shape), -2, dtype=torch.int16, device=pm.device)
            self.pmap_archive[self.cur.index, self.layer].copy_(pm, non_blocking=True)

    return PoolTrace


def unwritten(acts: Optional[torch.Tensor], pool_maps: Optional[torch.Tensor]) -> int:
    """I1 of the pool archives: entries never written (action -1 = record_pool's fill, map -2 = the archive's fill)."""
    n = 0
    if acts is not None:
        n += int((acts < 0).sum())
    if pool_maps is not None:
        n += int((pool_maps == -2).sum())
    return n


def _member(x: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """x (..., m), sets (..., n): bool (..., m) = x[i] >= 0 and x[i] in sets (per leading index)."""
    if sets.shape[-1] == 0:
        return torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    return (x[..., :, None] == sets[..., None, :]).any(-1) & (x >= 0)


def natural_accounting(masks: torch.Tensor, maps: torch.Tensor, acts: Optional[torch.Tensor], pool_maps: Optional[torch.Tensor],
                       tail_id: int, tail_slot: int = TOPK - 1) -> Dict:
    """Per (step, layer) accounting of a NATURAL capture at one capacity (module docstring).

    masks     (S, L, H, B, M) the POST-pool load masks the engine archived (cache_engine.py:823 record_mask): ids >= 0 crossed
              PCIe (the H2D misses the transport replays)
    maps      (S, L, H, B, M) the post-step window maps
    acts      (S, L, H, B, M) the pool actions (cache_engine.py:815 record_pool) or None (C = 63)
    pool_maps (S, L, H, B, P) the post-step pool maps or None
    The pre-step state of step 0 is the post-prefill one: window -1 except the tail slot (= tail_id), pool empty.
    Returns per-(step, layer) int tensors (S, L) and the invariant violation counts (0 = accepted):
      P1 an H2D-loaded id was already in HBM (window or pool) before the step
      P2 a pool-served slot (SWAP / MOVE_IN) holds an id that was NOT in the stream's previous pool
      P3 a slot that neither loaded nor was served changed its id
      P4 window U pool holds a duplicate id after the step
      P5 per stream, |resident after - resident before| != H2D loads (a new id entered HBM without crossing PCIe, or vice versa)
    """
    S, L, H, B, M = masks.shape
    dev = masks.device
    m64 = masks.to(torch.int64)
    w64 = maps.to(torch.int64)
    has_pool = acts is not None and pool_maps is not None and pool_maps.shape[-1] > 0
    P = int(pool_maps.shape[-1]) if has_pool else 0
    a8 = acts.to(torch.int64) if has_pool else None
    p64 = pool_maps.to(torch.int64) if has_pool else None
    win0 = torch.full((L, H, B, M), -1, dtype=torch.int64, device=dev)
    win0[..., tail_slot] = int(tail_id)
    pool0 = torch.full((L, H, B, P), -1, dtype=torch.int64, device=dev)
    z = lambda: torch.zeros((S, L), dtype=torch.int64)
    out = dict(h2d=z(), hits=z(), swaps=z(), move_in=z(), move_out=z(), evicted=z(), entering=z(), d2d_moves=z())
    inv = dict(P1_h2d_was_resident=0, P2_hit_not_from_pool=0, P3_kept_slot_changed=0, P4_duplicate_residency=0, P5_conservation=0)
    for s in range(S):
        prev_w = win0 if s == 0 else w64[s - 1]
        prev_p = pool0 if s == 0 else (p64[s - 1] if has_pool else pool0)
        cur_w, cur_p = w64[s], (p64[s] if has_pool else pool0)
        loaded = m64[s] >= 0
        loaded[..., tail_slot] = False
        if has_pool:
            act = a8[s]
            served = (act == POOL_SWAP) | (act == POOL_MOVE_IN)
            out["swaps"][s] = (act == POOL_SWAP).sum(dim=(1, 2, 3)).cpu()
            out["move_in"][s] = (act == POOL_MOVE_IN).sum(dim=(1, 2, 3)).cpu()
            out["move_out"][s] = (act == POOL_MOVE_OUT).sum(dim=(1, 2, 3)).cpu()
        else:
            served = torch.zeros_like(loaded)
        out["h2d"][s] = loaded.sum(dim=(1, 2, 3)).cpu()
        out["hits"][s] = served.sum(dim=(1, 2, 3)).cpu()
        prev_res = torch.cat([prev_w, prev_p], dim=-1)
        cur_res = torch.cat([cur_w, cur_p], dim=-1)
        inv["P1_h2d_was_resident"] += int((_member(torch.where(loaded, m64[s], torch.full_like(m64[s], -1)), prev_res)).sum())
        if has_pool:
            inv["P2_hit_not_from_pool"] += int((served & ~_member(cur_w, prev_p)).sum())
        keep = ~loaded & ~served
        keep[..., tail_slot] = False                                  # the tail slot is renamed only at a rollover (refused)
        inv["P3_kept_slot_changed"] += int((keep & (cur_w != prev_w)).sum())
        srt = torch.sort(cur_res, dim=-1).values
        inv["P4_duplicate_residency"] += int(((srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)).sum())
        new_in = (_member(cur_res, cur_res) & ~_member(cur_res, prev_res)).sum(-1)            # ids resident now, not before
        gone = (_member(prev_res, prev_res) & ~_member(prev_res, cur_res)).sum(-1)            # ids resident before, not now
        inv["P5_conservation"] += int((new_in != loaded.sum(-1)).sum())
        out["evicted"][s] = gone.sum(dim=(1, 2)).cpu()
    out["entering"] = out["h2d"] + out["hits"]
    out["d2d_moves"] = 2 * out["swaps"] + out["move_in"] + out["move_out"]
    streams = L * H * B
    tot = {k: int(v.sum()) for k, v in out.items()}
    summary = dict(steps=S, layers=L, streams_per_step=streams, P=P, C=capacity_of(P), totals=tot,
                   h2d_per_stream_step=tot["h2d"] / max(1, S * streams), hits_per_stream_step=tot["hits"] / max(1, S * streams),
                   h2d_bytes=tot["h2d"] * GROUP_USEFUL, d2d_bytes=tot["d2d_moves"] * (GROUP_USEFUL + GROUP_BIAS),
                   invariants=inv, accepted=all(v == 0 for v in inv.values()),
                   replayed="the H2D loads only (PCIe misses into attended slots 0..62); pool SWAP / MOVE_IN / MOVE_OUT are "
                            "device-to-device moves inside the cache tensors, accounted here and NOT replayed by either method")
    return dict(per=out, summary=summary)


def steady_summary(acc: Dict, steps: Sequence[int]) -> Dict:
    """The accounting restricted to the given (steady, non-cold) steps: per-step H2D bytes and pool hits, per stream-step."""
    per = acc["per"]
    S, L = per["h2d"].shape
    st = [s for s in steps if 0 <= s < S]
    streams = acc["summary"]["streams_per_step"]
    pick = lambda k: [int(per[k][s].sum()) for s in st]
    h2d, hits, d2d, ev = pick("h2d"), pick("hits"), pick("d2d_moves"), pick("evicted")
    n = max(1, len(st) * streams)
    return dict(steps=st, h2d_groups=h2d, h2d_bytes=[x * GROUP_USEFUL for x in h2d], hits=hits, d2d_moves=d2d, evicted=ev,
                h2d_per_stream_step=sum(h2d) / n, hits_per_stream_step=sum(hits) / n, evicted_per_stream_step=sum(ev) / n)
