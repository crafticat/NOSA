"""MISS-VOLUME CONTROL for the isolated one-position verifier experiment
(retroinfer-eval ledger, REPRODUCE.md 'SCOPE CORRECTION, 2026-09-20 (author):
MEASURE VERIFICATION ALONE FIRST'; driver benchmarks/Efficiency/verify_alone.py).

THE OBJECT UNDER MEASUREMENT IS THE ORDINARY DECODE STEP. Nothing here runs
inside a timed region. These helpers prepare the engine state BETWEEN two
ordinary steps so that the step's own diff + gathers move exactly D blocks per
(layer, KV head, request) stream -- through the actual fetch path -- and so that
a skipped or misdirected gather cannot pass unnoticed:

  record_targets   the per-layer target selection T_l = the engine's post-step
                   block map, i.e. the slot -> block-id table the attention
                   kernel just read (cache_engine.decode_update_has_kv_bias:
                   `_block_map.copy_(_new_block_map_buf)` runs before the
                   gathers and before flash_attn_nosa_with_kvcache; the map is
                   not touched again in the step unless the tail block fills,
                   which record_targets refuses). The trace's map archive
                   (transfer_trace.record_mask) is a copy of the same tensor at
                   the same instant; the engine's own tensor is used so that the
                   control needs no instrument.
  flush_map        every non-tail map entry := -1, the engine's own post-prefill
                   state (cache_engine.prefill_update section 3: the map is
                   built with torch.full(-1) and the first decode step fetches
                   the whole selection).
  prewarm          diff + the two shipped Triton gathers with a target map as
                   the "selection", untimed, then a second diff on (map, target)
                   to prove through the engine's own kernel that the map now
                   equals the target and that no load remains.
  evict            D non-tail slots per stream: their map entries pointed at a
                   real host block OUTSIDE T_l (so diff sees a legitimate old
                   occupant that the selection no longer names, i.e. a slot it
                   must refill), and their K, V and bias rows POISONED with NaN
                   (a gather that skips a row, or fills the wrong row, leaves
                   NaN under the attention kernel and the logits are NaN).
  expected_bytes   D x 64 x B x 32768: D blocks per stream, 64 streams per
                   request (2 KV heads x 32 layers), 32 KiB of K+V per block.

WHY THE LAYOUT MATTERS, AND WHAT monotone_prestep_map IS FOR. NOSA's decode
attention takes no block table: it reads the window's 64 slots as one flat
sequence, so the fp32 reduction order of the softmax depends on WHICH SLOT
holds which block. The selection order is arbitrary (nosa_llama.py: torch.topk
with sorted=False), and diff_kernel assigns the k-th selected block that is not
resident (in SELECTION order) to the k-th slot whose old block left (in SLOT
order). From an arbitrary layout, evicting a set S of slots and refetching can
therefore land the evicted blocks in permuted slots, the attention would read
the same data in another order, and bit-identity against the reference could
fail for a reason that is not a bug. The one layout that is a FIXED POINT of
"evict any S, then diff(., selection)" is the layout diff produces from a map
with no resident block: slot k holds the k-th non-tail block in selection
order (tests/test_nosi_verify_alone.py proves the fixed point on the pure-torch
twin of diff, and shows a counter-example from a non-monotone layout). So the
driver takes the reference step from a FLUSHED map (63 loads per stream; the
model computation is untouched: the selection is computed from GPU-resident
compressed keys, before the map is consulted), records that layout as T_l, and
every arm -- A, B(D), and the within-layout 'shipped' row -- is prepared to end
in exactly T_l. monotone_prestep_map builds, from the natural pre-step map
M_prev and T_l, the pre-step map M* that holds the SAME resident set as M_prev
in the T_l-compatible slots: the step from M* fetches exactly the natural miss
set and ends in T_l. The literal natural step (from M_prev, layout not
controlled) is also timed by the driver; its logits are compared to the
reference as an OBSERVATION of layout invariance, not as a control.

RULES KEPT (AGENTS.md): every op is per-stream integer bookkeeping (isolation:
stream (h, b) reads only row (h, b) of the maps); no float reduction; no host
branch on a device value -- the checks are returned as 0-d device booleans and
the driver reads them once per row, outside the timed region; this module
imports nothing from the engine (the verify package is the CPU-provable core,
tests/test_nosi_verify_core.py): the engine's diff extension and the two Triton
gathers are INJECTED by the driver (verify_alone.engine_kernels), and the CPU
tests inject pure-torch twins (tests/test_nosi_verify_alone.py).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

BLOCK_TOKENS = 64
HEAD_DIM = 128
BLOCK_BYTES_WIRE = 2 * BLOCK_TOKENS * HEAD_DIM * 2       # K + V, bf16: 32768 per loaded (layer, head, request, slot)
KV_HEADS = 2
NUM_LAYERS = 32
STREAMS_PER_REQUEST = KV_HEADS * NUM_LAYERS              # 64
LOCAL_WINDOW_BLOCKS = 16                                 # nosa_pooling local_blocks: ids T-16 .. T are always selected
PATTERNS = ("spread", "scattered", "adjacent", "natural", "layers4")
CONCENTRATE_LAYERS = 4
_NO_ID = 2 ** 62

class PrewarmResult(NamedTuple):
    ok: torch.Tensor          # 0-d bool (device): map == target, no load remains, diff would keep the map
    loaded: torch.Tensor      # 0-d int64 (device): entries the gather filled


class EvictResult(NamedTuple):
    evicted: torch.Tensor     # (H, B, M) bool
    new_map: torch.Tensor     # (H, B, M) int64
    ok: torch.Tensor          # 0-d bool (device)


# ---------------------------------------------------------------------------
# arithmetic (host ints)
# ---------------------------------------------------------------------------
def expected_bytes(D: int, B: int, heads: int = KV_HEADS, layers: int = NUM_LAYERS) -> int:
    """Wire bytes of D blocks per stream at batch B: D x (heads x layers) x B x 32768."""
    if D < 0 or B < 1:
        raise ValueError("expected_bytes: D=%r B=%r" % (D, B))
    return D * heads * layers * B * BLOCK_BYTES_WIRE


def layer_plan(pattern: str, D: int, num_layers: int = NUM_LAYERS, cap: int = BLOCK_TOKENS - 1) -> list:
    """Blocks per stream to evict in each layer for one row of the experiment.

    'layers4' concentrates the SAME total (num_layers x D per stream) into as
    few layers as possible at `cap` = 63 blocks per launch (the 64th slot is
    the tail, never fetched): at D = 8 that is 256 = 4 x 63 + 4, i.e. four
    layers at 63 and a fifth at 4 -- the ledger's "8 x D in 4 layers" is not
    realisable exactly (64 > 63 usable slots), so the bytes are kept equal and
    the fifth launch is stated. The loaded layers are spread evenly over the
    stack. Every other pattern evicts D in every layer.
    """
    if D < 0 or D > cap:
        raise ValueError("layer_plan: D=%d outside 0..%d" % (D, cap))
    if pattern != "layers4":
        return [D] * num_layers
    total = num_layers * D
    n_full, rem = divmod(total, cap)
    n_used = n_full + (1 if rem else 0)
    if n_used > num_layers:
        raise ValueError("layer_plan: %d blocks per stream do not fit %d layers at %d each" % (total, num_layers, cap))
    plan = [0] * num_layers
    if n_used == 0:
        return plan
    where = [int(round(j * num_layers / n_used)) for j in range(n_used)]
    for j, l in enumerate(where):
        plan[l] = cap if j < n_full else rem
    return plan


# ---------------------------------------------------------------------------
# the engine's kernels are INJECTED: `kernels` = (diff_offload, flash_h2d_from_mask,
# flash_h2d_from_mask_bias). The driver passes the fork's own (verify_alone.engine_kernels);
# the CPU tests pass pure-torch twins. This module imports nothing from the engine
# (tests/test_nosi_verify_core.py pins the verify package's purity).
# ---------------------------------------------------------------------------
def _window(engine):
    """(H, B, M, block_size, tail_slot) with the shape assertions of the shipped layout."""
    m = engine._block_map
    assert m.dim() == 3 and m.dtype == torch.int64, ("_block_map", tuple(m.shape), m.dtype)
    H, B, M = m.shape
    bs = engine.block_size
    tail = engine._tail_block_idx_on_gpu
    assert M == engine.topk and tail == M - 1, (M, engine.topk, tail)
    for name in ("_k_gpu", "_v_gpu"):
        t = getattr(engine, name)
        assert tuple(t.shape) == (B, M * bs, H, engine.head_dim), (name, tuple(t.shape), (B, M * bs, H, engine.head_dim))
    assert tuple(engine._kv_bias_gpu.shape) == (B, M * bs, H), tuple(engine._kv_bias_gpu.shape)
    assert engine._new_block_map_buf.shape == m.shape and engine._load_mask.shape == m.shape
    return H, B, M, bs, tail


def _same_table(t, m):
    assert t.shape == m.shape and t.dtype == torch.int64 and t.device == m.device, (tuple(t.shape), tuple(m.shape), t.dtype, t.device, m.device)
    assert t.is_contiguous(), "the kernels index the map as a flat (H, B, M) int64 array"


# ---------------------------------------------------------------------------
# (a) the target selection of a step
# ---------------------------------------------------------------------------
def record_targets(engine, layer_idx: int) -> torch.Tensor:
    """After an ordinary step: the (H, B, M) slot -> block-id table attention
    read, as a clone. Refuses a step in which the tail block filled (the map's
    tail entry was renamed AFTER attention, cache_engine section D)."""
    _window(engine)
    if engine._tail_block_len_on_gpu == 0:
        raise RuntimeError("record_targets(layer %d): the tail block filled in this step, the map no longer names what attention read" % layer_idx)
    return engine._block_map.clone()


def targets_well_formed(T: torch.Tensor, tail_slot: int) -> torch.Tensor:
    """0-d bool: every entry >= 0, pairwise distinct per stream, tail id at the tail slot >= 1."""
    s = T.sort(dim=-1).values
    distinct = (s[..., 1:] > s[..., :-1]).all()
    return distinct & (T >= 0).all() & (T[..., tail_slot] >= 1).all()


def same_selection(T_a: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
    """0-d bool: the two tables name the same SET of blocks in every stream (layout may differ)."""
    return (T_a.sort(dim=-1).values == T_b.sort(dim=-1).values).all()


# ---------------------------------------------------------------------------
# (b) flush and prewarm
# ---------------------------------------------------------------------------
def flush_map(engine) -> None:
    """Every non-tail map entry := -1 (the engine's own post-prefill state)."""
    H, B, M, bs, tail = _window(engine)
    engine._block_map[..., :tail] = -1


def prewarm(engine, layer_idx: int, T_l: torch.Tensor, kv_bias: torch.Tensor, kernels) -> PrewarmResult:
    """Make every block of T_l resident through the engine's own path: diff on
    (current map, T_l), the map copy, the two Triton gathers of the load mask
    (K from the pinned host K, V + bias from the host V and the layer's
    total_cis table, exactly as the decode step gathers). Then a second diff on
    (map, T_l): `ok` is (map == T_l) & (no load remains) & (diff would keep the
    map), all on the device. `loaded` = entries the first gather filled.

    The map ends equal to T_l only from a compatible pre-state (flushed, T_l
    itself, or monotone_prestep_map's M*); from any other map `ok` is False
    and the driver fails the row -- prewarm never rewrites the map by hand."""
    H, B, M, bs, tail = _window(engine)
    _same_table(T_l, engine._block_map)
    assert kv_bias.dim() == 3 and kv_bias.shape[0] == B and kv_bias.shape[2] == H, ("kv_bias (B, S, H)", tuple(kv_bias.shape))
    diff_fn, gather_k, gather_v_bias = kernels
    diff_fn(engine._block_map, T_l, engine._new_block_map_buf, engine._load_mask)
    engine._block_map.copy_(engine._new_block_map_buf)
    gather_k(engine._k_gpu, engine._k_cpu, engine._load_mask, bs)
    gather_v_bias(engine._v_gpu, engine._v_cpu, engine._kv_bias_gpu, kv_bias, engine._load_mask, bs)
    loaded = (engine._load_mask >= 0).sum()
    diff_fn(engine._block_map, T_l, engine._new_block_map_buf, engine._load_mask)
    ok = (engine._block_map == T_l).all() & (engine._load_mask < 0).all() & (engine._new_block_map_buf == T_l).all()
    return PrewarmResult(ok=ok, loaded=loaded)


# ---------------------------------------------------------------------------
# (c) eviction
# ---------------------------------------------------------------------------
def choose_slots(T_l: torch.Tensor, D: int, pattern: str, seed: int, tail_slot: int, prev_map=None, layer_idx: int = 0) -> torch.Tensor:
    """(H, B, M) bool: the slots to evict, never the tail slot.
      spread     D slots at evenly spaced positions floor((k + 1/2) (M-1) / D), the same in every stream
      scattered  D slots drawn per stream from a generator seeded by (seed, layer_idx)
      adjacent   the slots holding block ids T-1 .. T-D (T = the stream's tail id): consecutive
                 host ids inside the forced local window, so present in every stream for D <= 16
      natural    the slots whose block is NOT in prev_map: exactly what the shipped step from
                 prev_map would fetch (D is ignored)
    """
    H, B, M = T_l.shape
    dev = T_l.device
    sel = torch.zeros(H, B, M, dtype=torch.bool, device=dev)
    if pattern == "natural":
        if prev_map is None:
            raise ValueError("choose_slots('natural') needs prev_map")
        _same_table(prev_map, T_l)
        resident = (T_l[..., :, None] == prev_map[..., None, :]).any(-1)
        sel = ~resident
    elif D == 0:
        pass
    elif pattern == "spread":
        if D > M - 1:
            raise ValueError("spread: D=%d > %d non-tail slots" % (D, M - 1))
        pos = ((torch.arange(D, dtype=torch.float64) + 0.5) * (M - 1) / D).floor().long()
        sel[..., pos.to(dev)] = True
    elif pattern == "scattered":
        if D > M - 1:
            raise ValueError("scattered: D=%d > %d non-tail slots" % (D, M - 1))
        g = torch.Generator().manual_seed(int(seed) * 1000003 + int(layer_idx))
        order = torch.rand(H, B, M - 1, generator=g).argsort(dim=-1)[..., :D].to(dev)      # per-stream random slots
        sel[..., :M - 1].scatter_(2, order, torch.ones_like(order, dtype=torch.bool))
    elif pattern == "adjacent":
        if D > LOCAL_WINDOW_BLOCKS:
            raise ValueError("adjacent: D=%d > the %d-block local window that is present in every stream" % (D, LOCAL_WINDOW_BLOCKS))
        tail_id = T_l[..., tail_slot]                                                 # (H, B)
        targets = tail_id[..., None] - torch.arange(1, D + 1, device=dev)             # (H, B, D): ids T-1 .. T-D
        sel = (T_l[..., :, None] == targets[..., None, :]).any(-1)
    elif pattern == "layers4":
        raise ValueError("'layers4' is a driver-level pattern: use layer_plan() and evict 'spread' per layer")
    else:
        raise ValueError("unknown pattern %r (one of %s)" % (pattern, PATTERNS))
    sel[..., tail_slot] = False
    return sel


def replacement_ids(T_l: torch.Tensor, evicted: torch.Tensor, n_host: int) -> torch.Tensor:
    """(H, B, M) int64: T_l with every evicted slot replaced by a distinct
    block id in [0, n_host) that the stream's T_l does not contain (the k-th
    smallest absent id for the k-th evicted slot). n_host = complete blocks on
    the host = the tail id; n_host >= 2M - 1 guarantees enough absent ids."""
    H, B, M = T_l.shape
    dev = T_l.device
    if n_host < 2 * M - 1:
        raise ValueError("replacement_ids: only %d host blocks for a %d-slot window (need >= %d)" % (n_host, M, 2 * M - 1))
    present = torch.zeros(H, B, n_host + 1, dtype=torch.bool, device=dev)
    present.scatter_(2, T_l.clamp(min=0, max=n_host), torch.ones_like(T_l, dtype=torch.bool))
    absent = ~present[..., :n_host]
    ar = torch.arange(n_host, device=dev).expand(H, B, n_host)
    cand = torch.where(absent, ar, torch.full_like(ar, n_host)).sort(dim=2).values[..., :M]
    rank = (evicted.long().cumsum(2) - 1).clamp(min=0)
    repl = cand.gather(2, rank)
    return torch.where(evicted, repl, T_l)


def poison(engine, evicted: torch.Tensor) -> None:
    """NaN into the K, V and bias rows of every evicted (h, b, slot)."""
    H, B, M, bs, tail = _window(engine)
    assert evicted.shape == (H, B, M) and evicted.dtype == torch.bool
    rows = evicted.permute(1, 0, 2).repeat_interleave(bs, dim=2).permute(0, 2, 1).contiguous()   # (B, M*bs, H): row s*bs+i
    nan = float("nan")
    engine._k_gpu.masked_fill_(rows.unsqueeze(-1), nan)
    engine._v_gpu.masked_fill_(rows.unsqueeze(-1), nan)
    engine._kv_bias_gpu.masked_fill_(rows, nan)


def evict(engine, layer_idx: int, T_l: torch.Tensor, D: int, pattern: str, seed: int, prev_map=None) -> EvictResult:
    """From a map equal to T_l: choose the slots, point their map entries at
    real host blocks outside T_l, poison their rows. `ok` (device) = the map
    was T_l before, every stream has the requested count, every replacement
    id is < n_host. Returns the eviction mask (its sum is the load the next
    ordinary step must perform) and the new map."""
    H, B, M, bs, tail = _window(engine)
    _same_table(T_l, engine._block_map)
    was = (engine._block_map == T_l).all()
    sel = choose_slots(T_l, D, pattern, seed, tail, prev_map=prev_map, layer_idx=layer_idx)
    n_host = (engine.seq_length - engine._tail_block_len_on_gpu) // bs
    new_map = replacement_ids(T_l, sel, n_host)
    count_ok = (sel.sum(-1) == D).all() if pattern != "natural" else torch.tensor(True, device=T_l.device)
    ids_ok = ((new_map < n_host) | ~sel).all() & (new_map[..., tail] == T_l[..., tail]).all()
    engine._block_map.copy_(new_map)
    poison(engine, sel)
    return EvictResult(evicted=sel, new_map=new_map, ok=was & count_ok & ids_ok)


# ---------------------------------------------------------------------------
# (d) the within-layout pre-step map
# ---------------------------------------------------------------------------
def monotone_prestep_map(prev_map: torch.Tensor, T_l: torch.Tensor, tail_slot: int):
    """M*: the resident set of prev_map laid out so that the ordinary step
    (diff on (M*, selection)) fetches exactly the natural misses and ends in
    T_l. Slot k keeps T_l[k] when that block is resident in prev_map; the
    other slots take the leaving blocks of prev_map (sorted by id -- their
    content is overwritten by the step's gather). Returns (M*, ok, misses):
    ok = M* is a permutation of prev_map and the tail is a hit; misses = the
    (H, B, M) bool of slots the step must fetch."""
    _same_table(prev_map, T_l)
    hit = (T_l[..., :, None] == prev_map[..., None, :]).any(-1)              # T_l[k] resident?
    leaving = ~(prev_map[..., :, None] == T_l[..., None, :]).any(-1)        # prev_map[s] not selected
    rank = ((~hit).long().cumsum(2) - 1).clamp(min=0)
    fill = torch.where(leaving, prev_map, torch.full_like(prev_map, _NO_ID)).sort(dim=2).values.gather(2, rank)
    Mstar = torch.where(hit, T_l, fill)
    ok = (Mstar.sort(dim=2).values == prev_map.sort(dim=2).values).all() & hit[..., tail_slot].all() \
        & ((~hit).sum(-1) == leaving.sum(-1)).all()
    return Mstar, ok, ~hit
