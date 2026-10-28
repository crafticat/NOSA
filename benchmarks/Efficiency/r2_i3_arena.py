"""Rung I3 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the K / V / bias ARENA of
WS-C (benchmarks/Efficiency/r2_kv_arena.py, unchanged and imported) with ONE more region, the CHAIN IMAGE, and a prefill
hook that re-homes the engines WITHOUT a copy. Docs: docs/NOTES_i3.md (section 2: the seam; section 6: the memory).

WHY A CHAIN IMAGE. The fork (fork_vllm_unified_attention.py, fia6's bytes) shows a row the tokens 0 .. n - 1 of each
block it lists (n = 64, or last_len on the row's LAST listed page: fork :296, :370). The committed block qb = qc // 64
of request b holds its committed rows 0 .. qc % 64 - 1 in the window's TAIL SLOT and its tentative rows qc % 64 .. in
the TENTATIVE BUFFER (spec_tentative.TentState: contract 4, departure D1 removed: a tentative row never enters the tail
slot). No prefix of either store is the row's view. So, per layer and joint pass, spec_tentative_fork.TentStateF
WRITES TWO SCRATCH BLOCKS per request and lists them in place of the blocks qb and qb + 1:
  CB0 (this region)       rows x < qc % 64 from the tail slot (committed), rows x >= qc % 64 from the buffer row x
  CB1 (store slot 0)      the buffer (64 rows): the rows of block qb + 1 (WS-C's "next block" slot, the same role)
A row at p in block qb sees CB0's rows 0 .. n_tail(p) - 1; a row past the edge sees CB0 whole (if its head selects qb)
and CB1's rows 0 .. n_tail(p) - 1: exactly TentState._row's call A + call B (spec_tentative.py:415-441). The images
are scratch: nothing is written back; the committed state changes only at the round's commit (TentState.commit).

LAYOUT (WS-C's, plus the CB0 region at the end; a block = 64 tokens of one request, all KV heads):
    block id  l * B * NS + b * NS + s     the engines (WS-C)
    block id  L * B * NS + b * X + x      the store (WS-C); slot 0 = CB1 in I3
    block id  L * B * NS + B * X + b      CB0 of request b (one per request, shared by the layers: a layer's fork call
                                          reads it before the next layer writes it, one stream)

NoCopyPrefillHook. r2_kv_arena.PrefillHook lets each engine allocate its GPU rows (cache_engine.py:222-224) and COPIES
them into the arena: one layer's rows are live twice while it binds (MEASURED in wc1b: the arm peak of L4XF is SH's +
4.10 GB = store 2.70 + about 1.4; docs/NOTES_i3.md section 6). Here the three torch.empty calls of those rows return the
arena's views instead (the name `torch` in prefill_update's globals is replaced by a proxy only during the call with
current_batch_pos == 0); the engine then writes its prefill rows straight into the arena. No other allocation and no
other call is changed; after the call every layer must be bound (r2_kv_arena.KVArena.bound), else ArenaError.

Imports torch and r2_kv_arena only. No kernel, no device read.
"""
from __future__ import annotations

import torch

import r2_kv_arena as KA

ArenaError = KA.ArenaError
ENGINE_ROWS = KA.ENGINE_ROWS


class ChainArena(KA.KVArena):
    """WS-C's arena (r2_kv_arena.KVArena: the same attributes, views and checks) with the CB0 region (module
    docstring). L layers, B requests, NS engine slots per request and layer, H KV heads, head dim D, X store slots."""

    def __init__(self, L: int, B: int, ns: int, H: int, D: int, X: int, dtype, device, bs: int = 64):
        L, B, ns, H, D, X, bs = (int(x) for x in (L, B, ns, H, D, X, bs))
        if min(L, B, ns, H, D, bs) < 1 or X < 2:
            raise ArenaError("L, B, NS, H, D, bs >= 1 and X >= 2 (CB1 in slot 0 and one listed slot)")
        self.L, self.B, self.ns, self.H, self.D, self.X, self.bs = L, B, ns, H, D, X, bs
        self.store_base = L * B * ns
        self.cb0_base = self.store_base + B * X
        n = self.cb0_base + B
        self.n_blocks = n
        self.k = torch.empty((n, bs, H, D), dtype=dtype, device=device)
        self.v = torch.empty((n, bs, H, D), dtype=dtype, device=device)
        self.bias = torch.empty((n, bs, H), dtype=dtype, device=device)
        sb, cb = self.store_base, self.cb0_base
        self.store_k = self.k[sb:cb].view(B, X * bs, H, D)
        self.store_v = self.v[sb:cb].view(B, X * bs, H, D)
        self.store_bias = self.bias[sb:cb].view(B, X * bs, H)
        self.cb0_k, self.cb0_v, self.cb0_bias = self.k[cb:], self.v[cb:], self.bias[cb:]   # (B, 64, H, D) / (B, 64, H)
        for t in (self.store_k, self.store_v, self.store_bias, self.cb0_k, self.cb0_v, self.cb0_bias):
            t.zero_()                            # defined contents; a scratch row is read only where a row lists it

    def cb1_views(self):
        """(k, v, bias) of store slot 0 of every request: (B, 64, H, D) x 2 and (B, 64, H), strided views (row stride X
        x 64); the CB1 images are written into them with torch ops only (no raw-pointer kernel reads these views)."""
        bs = self.bs
        return self.store_k[:, :bs], self.store_v[:, :bs], self.store_bias[:, :bs]

    def chain_bytes(self) -> int:
        """CB0's bytes (CB1 lives in the store)."""
        return sum(t.numel() * t.element_size() for t in (self.cb0_k, self.cb0_v, self.cb0_bias))

    def describe(self) -> dict:
        d = super().describe()
        d.update(cb0_base=self.cb0_base, cb0_GB=round(self.chain_bytes() / 1e9, 4), cb1="store slot 0", kind="chain")
        return d


def for_engine(eng, L: int, B: int, X: int) -> ChainArena:
    """r2_kv_arena.for_engine with the chain region (it can be made before the prefill: no engine tensor is read)."""
    ns = int(eng.topk) + int(getattr(eng, "pool_blocks", 0) or 0)
    return ChainArena(L, B, ns, int(eng.head_num), int(eng.head_dim), X, eng.dtype, eng.device, int(eng.block_size))


class _ArenaTorch:
    """The proxy of the name `torch` in prefill_update's globals during ONE call (module docstring): torch.empty of the
    engine's three GPU row shapes returns layer l's arena views in the engine's call order (k, v, then bias); every
    other attribute and call is the real one."""

    def __init__(self, real, views, device):
        self._real, self._views, self._dev = real, list(views), device
        self.served = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _same_device(self, dev) -> bool:
        if dev is None:
            return False
        d, a = torch.device(dev), torch.device(self._dev)
        return d.type == a.type and (d.index is None or a.index is None or d.index == a.index)

    def empty(self, *size, **kw):
        shape = tuple(size[0]) if (len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size))) else tuple(size)
        on_dev = self._same_device(kw.get("device"))
        for i, (name, t) in enumerate(self._views):
            if on_dev and shape == tuple(int(x) for x in t.shape) and kw.get("dtype", t.dtype) == t.dtype:
                del self._views[i]
                self.served.append(name)
                return t
        return self._real.empty(*size, **kw)


class NoCopyPrefillHook:
    """Re-home during the SHIPPED prefill WITHOUT a copy (module docstring). Use as r2_kv_arena.PrefillHook:
    `with NoCopyPrefillHook(cache, arena): model.batch_prefill(ids, cache)`. On exit the wrappers are removed and every
    layer must be bound (else ArenaError). served[l]: the names the proxy returned for layer l (ENGINE_ROWS)."""

    def __init__(self, cache, arena: KA.KVArena):
        self.cache, self.arena = cache, arena
        self.bound_at, self.served = [], {}

    def _wrap(self, eng, l):
        orig = eng.prefill_update
        g = type(eng).prefill_update.__globals__

        def prefill_update(key_states, value_states, kv_bias, current_batch_pos, total_bsz):
            if int(current_batch_pos) != 0:
                return orig(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
            real = g["torch"]
            views = list(zip(ENGINE_ROWS, self.arena.layer_views(l)))
            proxy = _ArenaTorch(real, views, self.arena.k.device)
            g["torch"] = proxy
            try:
                out = orig(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
            finally:
                g["torch"] = real
            self.served[l] = list(proxy.served)
            if tuple(proxy.served) != ENGINE_ROWS or not self.arena.bound(eng, l):
                raise ArenaError("layer %d: the engine's GPU rows did not come from the arena (served %s)" % (
                    l, proxy.served))
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
            if "torch" not in type(eng).prefill_update.__globals__:
                raise ArenaError("layer %d: prefill_update has no global name torch" % l)
            eng.prefill_update = self._wrap(eng, l)
        return self

    def __exit__(self, *exc):
        for lay in self.cache.layers:
            lay.cache_engine.__dict__.pop("prefill_update", None)
        if exc[0] is None:
            bad = KA.check(self.cache, self.arena)
            if bad:
                raise ArenaError("layers %s are not bound to the arena after the prefill" % bad[:8])
        return False
