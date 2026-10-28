"""WS-C of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the K / V / bias ARENA.

WHY. The fork kernel (benchmarks/Efficiency/fork_vllm_unified_attention.py, a byte copy of nosi-r2-fia6 94d5bb0
nosi/nosi/fork_vllm_unified_attention.py) reads K, V and the CIS bias through ONE base pointer each and a table of
physical block ids (fork :574-576 the table load; :598-609 physical_block_idx * stride_v / k_cache_0; :645-653 the
bias). The union
joint pass needs, per layer, the window and the victim pool of that layer's engine AND the union store (the host-only
blocks of the rows). The shipped engine allocates each layer's GPU rows separately (cache_engine.py:222-224), so the
store cannot share their base pointer. Here ONE allocation per tensor kind holds the GPU rows of EVERY layer and the
store, and every layer's engine tensors become CONTIGUOUS views of it: every shipped kernel sees the same shape and
the same strides as before (lesson L28: no strided view into a raw-pointer gather), and one fork call per layer
reaches the window, the pool slots (read in place, no copy) and the store through one table.

LAYOUT (a block = 64 tokens of one request, all KV heads; NS = topk + P engine slots per request and layer):
    block id  l * B * NS + b * NS + s     layer l, request b, engine slot s (0 .. topk - 1 window, topk .. NS - 1 pool)
    block id  L * B * NS + b * X + x      the store: request b, slot x (0: the next block, 1 .. X - 1: listed blocks)
The k / v tensors are (L * B * NS + B * X, 64, H, D), the bias tensor (L * B * NS + B * X, 64, H): exactly the fork's
k / v / cis_bias layouts (fork :860 k.shape = [num_blocks, block_size, num_kv_heads, head_size]; :872 the bias).

MEMORY. The arena adds only the store: B * X * 64 * H * (2 D + 1) * 2 B (bf16). The engine rows move into it; they are
not doubled. Re-homing during the shipped prefill (PrefillHook) holds ONE layer's original rows besides the arena for
the copy (the caching allocator reuses that block for the next layer's allocation).

USE (GPU cell): arena = for_engine(eng0, L, B, X) BEFORE the prefill; `with PrefillHook(cache, arena):` around the
shipped prefill; then ragged_engine.attach and `attach(rs, arena)` (rs.kv_arena, read by verify_rows' fork path).
USE (CPU tests): rehome(cache, arena) on a built or prefilled world, before ragged_engine.attach / build's first step.

Imports only torch. No kernel, no device read.
"""
from __future__ import annotations

import torch

ENGINE_ROWS = ("_k_gpu", "_v_gpu", "_kv_bias_gpu")


class ArenaError(RuntimeError):
    pass


def store_slots(topk: int, full_max: int) -> int:
    """X, the store slots per request that make the union fit ONE fill (C = 1): slot 0 (the next block) plus the bound
    of the listed blocks. The window holds row 0's selection, so a FULL row r >= 1 adds at most topk host-only blocks:
    the bound is (full_max - 1) x topk (full_max: the most full rows per request and pass)."""
    if int(full_max) < 1 or int(topk) < 1:
        raise ArenaError("full_max and topk must be >= 1")
    return 1 + (int(full_max) - 1) * int(topk)


class KVArena:
    """The arena (module docstring). L layers, B requests, NS engine slots per request and layer, H KV heads, head dim
    D, X store slots per request, block size bs."""

    def __init__(self, L: int, B: int, ns: int, H: int, D: int, X: int, dtype, device, bs: int = 64):
        L, B, ns, H, D, X, bs = (int(x) for x in (L, B, ns, H, D, X, bs))
        if min(L, B, ns, H, D, bs) < 1 or X < 2:
            raise ArenaError("L, B, NS, H, D, bs >= 1 and X >= 2 (the next-block slot and one listed slot)")
        self.L, self.B, self.ns, self.H, self.D, self.X, self.bs = L, B, ns, H, D, X, bs
        self.store_base = L * B * ns
        n = self.store_base + B * X
        self.n_blocks = n
        self.k = torch.empty((n, bs, H, D), dtype=dtype, device=device)
        self.v = torch.empty((n, bs, H, D), dtype=dtype, device=device)
        self.bias = torch.empty((n, bs, H), dtype=dtype, device=device)
        sb = self.store_base
        self.store_k = self.k[sb:].view(B, X * bs, H, D)
        self.store_v = self.v[sb:].view(B, X * bs, H, D)
        self.store_bias = self.bias[sb:].view(B, X * bs, H)
        self.store_k.zero_()                     # defined contents; a store slot is read only where a row lists it
        self.store_v.zero_()
        self.store_bias.zero_()

    def layer_base(self, l: int) -> int:
        """The arena block id of (layer l, request 0, engine slot 0)."""
        if not 0 <= int(l) < self.L:
            raise ArenaError("layer %r outside 0 .. %d" % (l, self.L - 1))
        return int(l) * self.B * self.ns

    def layer_views(self, l: int):
        """(k, v, bias) of layer l: (B, NS x bs, H, D) x 2 and (B, NS x bs, H); contiguous views of the arena."""
        a = self.layer_base(l)
        n = self.B * self.ns
        rows = self.ns * self.bs
        return (self.k[a:a + n].view(self.B, rows, self.H, self.D), self.v[a:a + n].view(self.B, rows, self.H, self.D),
                self.bias[a:a + n].view(self.B, rows, self.H))

    def store_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.store_k, self.store_v, self.store_bias))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.bias))

    def describe(self) -> dict:
        return dict(L=self.L, B=self.B, NS=self.ns, H=self.H, D=self.D, X=self.X, bs=self.bs, n_blocks=self.n_blocks,
                    store_base=self.store_base, store_GB=round(self.store_bytes() / 1e9, 4),
                    arena_GB=round(self.nbytes() / 1e9, 4), dtype=str(self.k.dtype))

    def bound(self, eng, l: int) -> bool:
        """The engine's three GPU row tensors ARE layer l's views (pointer, shape and strides); no device read."""
        for name, t in zip(ENGINE_ROWS, self.layer_views(l)):
            x = getattr(eng, name, None)
            if x is None or x.data_ptr() != t.data_ptr() or tuple(x.shape) != tuple(t.shape) or \
                    tuple(x.stride()) != tuple(t.stride()) or x.dtype != t.dtype:
                return False
        return True

    def bind(self, eng, l: int) -> None:
        """Copy the engine's GPU rows of layer l into the arena and make the engine use the arena's views (and the
        attended views of a pooled engine, cache_engine.py:298-301, over the new tensors). Idempotent."""
        if self.bound(eng, l):
            return
        views = self.layer_views(l)
        for name, dst in zip(ENGINE_ROWS, views):
            src = getattr(eng, name)
            if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype or src.device != dst.device:
                raise ArenaError("layer %d %s: %s %s %s does not fit the arena's %s %s %s" % (
                    l, name, tuple(src.shape), src.dtype, src.device, tuple(dst.shape), dst.dtype, dst.device))
        for name, dst in zip(ENGINE_ROWS, views):
            dst.copy_(getattr(eng, name))
            setattr(eng, name, dst)
        if getattr(eng, "_k_gpu_att", None) is not None:
            n = int(eng._att_rows)
            eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att = views[0][:, :n], views[1][:, :n], views[2][:, :n]


def for_engine(eng, L: int, B: int, X: int) -> KVArena:
    """An arena for L layers of engines like `eng` (topk, pool_blocks, head_num, head_dim, dtype, device, block_size)
    and B requests; it can be made before the prefill (no engine tensor is read)."""
    ns = int(eng.topk) + int(getattr(eng, "pool_blocks", 0) or 0)
    return KVArena(L, B, ns, int(eng.head_num), int(eng.head_dim), X, eng.dtype, eng.device, int(eng.block_size))


def rehome(cache, arena: KVArena) -> None:
    """Bind every layer of a cache whose engines already hold their GPU rows (a built or prefilled world)."""
    if len(cache.layers) != arena.L:
        raise ArenaError("the cache has %d layers, the arena %d" % (len(cache.layers), arena.L))
    for l, lay in enumerate(cache.layers):
        arena.bind(lay.cache_engine, l)


def check(cache, arena: KVArena) -> list:
    """The layers whose engine is not bound (empty: every layer uses the arena)."""
    return [l for l, lay in enumerate(cache.layers) if not arena.bound(lay.cache_engine, l)]


class PrefillHook:
    """Re-home during the SHIPPED prefill: each layer's CacheEngine.prefill_update allocates its GPU rows at
    current_batch_pos == 0 (cache_engine.py:192, :222-224); the wrapper binds them to the arena right after that call,
    so the later sub-batches write into the arena. On exit the wrappers are removed and every layer must be bound."""

    def __init__(self, cache, arena: KVArena):
        self.cache, self.arena = cache, arena
        self.bound_at = []

    def _wrap(self, eng, l):
        orig = eng.prefill_update

        def prefill_update(key_states, value_states, kv_bias, current_batch_pos, total_bsz):
            out = orig(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
            if int(current_batch_pos) == 0:
                self.arena.bind(eng, l)
                self.bound_at.append(l)
            return out
        return prefill_update

    def __enter__(self):
        if len(self.cache.layers) != self.arena.L:
            raise ArenaError("the cache has %d layers, the arena %d" % (len(self.cache.layers), self.arena.L))
        for l, lay in enumerate(self.cache.layers):
            eng = lay.cache_engine
            if "prefill_update" in eng.__dict__:
                raise ArenaError("layer %d: prefill_update is already wrapped" % l)
            eng.prefill_update = self._wrap(eng, l)
        return self

    def __exit__(self, *exc):
        for lay in self.cache.layers:
            lay.cache_engine.__dict__.pop("prefill_update", None)
        if exc[0] is None:
            bad = check(self.cache, self.arena)
            if bad:
                raise ArenaError("layers %s are not bound to the arena after the prefill" % bad[:8])
        return False


def attach(rs, arena: KVArena) -> None:
    """rs.kv_arena = arena, after a check that every layer's engine is bound (verify_rows' fork path reads it)."""
    bad = check(rs.cache, arena)
    if bad:
        raise ArenaError("layers %s are not bound to the arena" % bad[:8])
    if int(rs.B) != arena.B:
        raise ArenaError("the state has B %d, the arena %d" % (int(rs.B), arena.B))
    rs.kv_arena = arena
