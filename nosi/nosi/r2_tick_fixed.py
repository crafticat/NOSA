"""R2 step 7: the FIXED-CAPACITY TICK (W2) and the SYNC-FREE EVENTS (E1, W3) of docs/R2_TICK_CAPTURE_PLAN.md. The
eager form of the region that W4 captures in a CUDA graph. r2_tick.TickEngine stays the default and the reference
(SG0-SG4); FixedTickEngine is a subclass behind its own constructor, and every result of a LIVE row is bitwise
TickEngine's (the CPU tests C1-C3, benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py).

CAPACITY. plan_tick verifies at most one cohort of at most K rows per request per tick (r2_spec_core.py:142-153), so
a tick has R <= B (1 + v_cap) rows and T <= v_cap rows per bank (v_cap = K; the lead's Q-T1 answer, 2026-10-21:
keep the cap a parameter, refuse a tick above it). Two capacities (the graph family of plan 8.5):
  G_d  (R_cap, T_cap) = (B, 1)                       a tick with R <= B rows and <= 1 row per bank (draft-only ticks)
  G_v  (R_cap, T_cap) = (B (1 + v_cap), v_cap)      every other tick
cap_mode "family" (the default) picks the smallest that fits; "max" always G_v; "exact" (R, T) (no padding: the
padding-invariance test C2). A tick above G_v is REFUSED before any write (TickError).

PADDED ROWS (rows R .. R_cap - 1). A fixed dummy row: request 0, bank 0, position S_list[0], live 0. It READS only
valid memory (bank 0's tables, seqused_k 1, one page: the trash slot, last_len 1) and WRITES only TRASH CELLS:
  the store's TRASH SLOT NB + 4 (the slot after the four tail-ring slots; C's Q-T2 answer, 2026-10-21: the store has
  (NB + 5) x 64 rows, no pool, transport, gather or tail touches that slot); the twins' scratch column T_cap (n counts
  live rows only); the token-log slot LOG - 1 (the driver refuses a tick that would reach it).
Their outputs are UNDEFINED and never compared (the lesson of the capture gate's first run, e5fc694).

THE REGION (_forward_fixed, _layer_fixed). TickEngine.run_rows / _layer with:
  F2  the row table, the per-bank counts, cu_q, cu_k, the stage-1 / pooling / FFN buffers, the twin scratch and the
      per-layer counters are STATIC per capacity (CapBuffers); the driver writes the table with ONE copy
  F3  padded rows to the trash cells; n_used of a padded row = 1
  F4  PositionalTwin.advance_unchecked (C's 4fe5fc9) over T_cap steps with the device counts (no host read); its
      sticky device flag `overflow`
  F6  the token log written through a `logslot` column (no host offset inside the region)
  F7  a padded row's num_pages = 1, page 0 = the trash slot, last_len = 1
  F8  the `check` of the selection's last two blocks as a device flag (err_dev), live rows only
  F9  the drop path without the host branch on any_draft (page_table(..., any_draft=True): equal bits)
It reads no device value on the host, allocates no state, and has no host branch on a device value (the lint C7).
The transient intermediates of its ops come from the allocator (in a graph: the graph's pool; the allocation rule).

THE DRIVER (run_rows, outside the region): TickEngine.row_table (the host checks), the capacity choice and refusals,
C's capacity condition on the host mirrors (pos + n <= N_max, (pos + n) // 16 + 1 <= Lk_max per bank), the padded
table, ONE host-to-device copy, the region, the host bookkeeping. read_tokens is the ONE decision read: the token
slots, err_dev and every twin's `overflow` in one transfer; a set flag raises TickError (every decision since the
last clean read is void: C's Q-T5 answer).

E1 (apply_events, sync-free, eager): TickEngine.apply_events with
  commit_banks            PositionalTwin.commit's checks on the HOST MIRRORS (the snapshot's epochs = the epoch
                          mirror, keep within the snapshot, not ahead, back <= MAX_ROLLBACK), then commit_unchecked
                          per layer (the same device ops as commit, no read)
  rebuild_from_own_rows_unchecked   r2_tick.rebuild_from_own_rows with advance_unchecked
writeback, copy_ring, copy_rows and sync_draft_bank are TickEngine's / r2_spec_core's (device copies, no read).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from . import r2_spec_core as SC
from .r2_positional_twin import BLOCK, MAX_ROLLBACK, STRIDE, PositionalTwin
from .r2_tick import FIELDS, TickEngine, TickError

XFIELDS = FIELDS + ("live", "logslot", "ucol")
N_TRASH = 1                                   # store slots past the four tail-ring slots
CAP_MODES = ("family", "max", "exact")
TWIN_COPY = ("comp_k", "ring", "total_cis", "pos", "n_comp", "n_ccis", "epoch", "prefilled", "ccis")


# =====================================================================================================================
# E1: the twin operations of apply_events without a host read
# =====================================================================================================================
def commit_unchecked(twin: PositionalTwin, ctr: torch.Tensor, keep: torch.Tensor) -> None:
    """PositionalTwin.commit's writes without its checks: every bank takes ctr[keep[bank], :, bank]; epoch + 1.
    keep: (twin.B,) int64 ON THE TWIN'S DEVICE. The caller checks on its host mirrors (FixedTickEngine.commit_banks)."""
    dev = twin.pos.device
    sel = ctr.to(device=dev, dtype=torch.int64).permute(2, 0, 1)[torch.arange(twin.B, device=dev), keep]   # (B, 4)
    twin.pos.copy_(sel[:, 0])
    twin.n_comp.copy_(sel[:, 1])
    twin.n_ccis.copy_(sel[:, 2])
    twin.epoch.add_(1)


def rebuild_from_own_rows_unchecked(twin: PositionalTwin, b: int, B: int, p: int, dpos: int):
    """r2_tick.rebuild_from_own_rows (K3X F5, C's Q5) with advance_unchecked: FIRST the draft bank's own unverified
    rows, positions [p, dpos), THEN sync_draft_bank, then the bank advances by those rows. Returns (keys, cis)."""
    n = int(dpos) - int(p)
    if not 0 <= n <= BLOCK:
        raise TickError("draft bank %d: %d unverified rows (the ring holds 64)" % (B + b, n))
    dev = twin.pos.device
    idx = torch.arange(int(p), int(dpos), dtype=torch.int64, device=dev)
    dk = twin.ring[B + b, idx % BLOCK].clone()
    dc = twin.total_cis[B + b, idx].clone()
    SC.sync_draft_bank(twin, b, B)
    if n:
        keys = torch.zeros((twin.B, n, twin.H, twin.D), dtype=twin.ring.dtype, device=dev)
        cis = torch.zeros((twin.B, n, twin.H), dtype=twin.total_cis.dtype, device=dev)
        keys[B + b] = dk
        cis[B + b] = dc
        cnt = torch.zeros(twin.B, dtype=torch.int64, device=dev)
        cnt[B + b] = n
        twin.advance_unchecked(keys, cis, cnt)
    return dk, dc


def clone_twins(twins: Sequence[PositionalTwin]) -> List[PositionalTwin]:
    out = []
    for t in twins:
        c = PositionalTwin(t.B, t.H, t.D, t.lk_max, t.n_max, dtype=t.ring.dtype, device=t.pos.device)
        for name in TWIN_COPY:
            getattr(c, name).copy_(getattr(t, name))
        out.append(c)
    return out


# =====================================================================================================================
# the static buffers of one capacity
# =====================================================================================================================
class CapBuffers:
    """Every tensor of the region at one capacity (R_cap, T_cap) that is an input written by the driver, an output read
    after the region, or a buffer the region writes in place. Allocated once per capacity, before its first tick."""

    def __init__(self, eng: "FixedTickEngine", R_cap: int, T_cap: int):
        dev, t0, lay0 = eng.dev, eng.twins[0], eng.model.layers[0]
        NF = len(XFIELDS)
        self.R_cap, self.T_cap = int(R_cap), int(T_cap)
        self.flat = torch.zeros(NF * self.R_cap + 2 * eng.B, dtype=torch.int64, device=dev)   # ONE copy per tick
        self.tab = self.flat[:NF * self.R_cap].view(NF, self.R_cap)                            # (XFIELDS, R_cap)
        self.cnt = self.flat[NF * self.R_cap:]                                                  # (2B,) live rows per bank
        self.cu_q = torch.arange(self.R_cap + 1, dtype=torch.int32, device=dev)
        self.cu_k = torch.full((self.R_cap + 1,), 2 * eng.B * t0.lk_max, dtype=torch.int32, device=dev)
        self.keys = torch.zeros((t0.B, self.T_cap + 1, t0.H, t0.D), dtype=t0.ring.dtype, device=dev)
        self.cis = torch.zeros((t0.B, self.T_cap + 1, t0.H), dtype=t0.total_cis.dtype, device=dev)
        self.ctr = torch.zeros((eng.L, self.T_cap + 1, 4, t0.B), dtype=torch.int64, device=dev)
        bdt = lay0.wqkv.dtype
        self.pq = torch.empty((lay0.num_key_value_heads, self.R_cap, eng.M_alloc), dtype=bdt, device=dev)
        self.pc = torch.empty_like(self.pq)
        self.ffn = torch.empty((self.R_cap, 1, lay0.gate_up_proj.shape[0] // 2), dtype=eng.model.embed_tokens.dtype,
                               device=dev)
        self.uses = 0


# =====================================================================================================================
# the engine
# =====================================================================================================================
class FixedTickEngine(TickEngine):
    """TickEngine at fixed capacity (module docstring). The stores have ONE more slot than TickEngine's: slots
    NB .. NB + 3 are the tail rings, NB + 4 is the TRASH slot (from_engine / from_r2 add it). Extra arguments:
    v_cap (verify rows per request per tick; R_max = B (1 + v_cap), T_max = v_cap) and cap_mode."""

    def __init__(self, model, ops, twins, store_k, store_v, store_bias, S_list, M_alloc: int, v_cap: int = 3,
                 cap_mode: str = "family", **kw):
        super().__init__(model, ops, twins, store_k, store_v, store_bias, S_list, M_alloc, **kw)
        self.NB -= N_TRASH                                   # TickEngine counted the trash slot as history
        if self.NB < 1:
            raise TickError("the store needs (NB + 5) x 64 rows with NB >= 1")
        self.trash = self.NB + 4
        if isinstance(v_cap, bool) or int(v_cap) != v_cap or int(v_cap) < 1:
            raise TickError("v_cap must be a positive int (got %r)" % (v_cap,))
        if cap_mode not in CAP_MODES:
            raise TickError("cap_mode must be one of %s (got %r)" % (CAP_MODES, cap_mode))
        self.v_cap = int(v_cap)
        self.R_max, self.T_max = self.B * (1 + self.v_cap), self.v_cap
        self.cap_mode = cap_mode
        self.bufs: Dict[tuple, CapBuffers] = {}
        self.err_dev = torch.zeros((), dtype=torch.bool, device=self.dev)
        # the host mirror of the twins' epochs (one read here, at construction, never in a tick)
        self.epoch_host = [int(x) for x in self.twins[0].epoch.tolist()]
        for t in self.twins[1:]:
            if [int(x) for x in t.epoch.tolist()] != self.epoch_host:
                raise TickError("the layers' twins disagree on their epochs")
        self.ctr_epoch = None
        self.stats.update(padded_rows=0, caps={})
        # W5 seams (docs/R2_TICK_CAPTURE_PLAN.md 12b): pre_region(eng, cb, rows) runs on the host before the region
        # (a captured tick: before the replay; S-C's host part: sync_mirror, the fixed row buffers), post_region(eng,
        # cb, rows) after it (S-C: the reader event, end_tick); flag_watcher: an S-C service that reads err_dev and the
        # twins' overflow flags with its one read per tick (watch_flags_with)
        self.pre_region = None
        self.post_region = None
        self.flag_watcher = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_engine(cls, eng: TickEngine, ops=None, **kw):
        """A FixedTickEngine with a COPY of a fresh TickEngine's state (NB + 4 store slots): the stores get the trash
        slot NB + 4 (zero). The source is not changed. Keyword arguments: the constructor's (v_cap, cap_mode, ...)."""
        if eng.stats["ticks"] or eng.n_tok:
            raise TickError("from_engine() is for a fresh engine")
        rows = (eng.NB + 4) * BLOCK

        def grow(x):
            y = torch.zeros((x.shape[0], rows + N_TRASH * BLOCK) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
            y[:, :rows].copy_(x[:, :rows])
            return y
        args = dict(nl=eng.nl, stage1=eng.stage1, pool=eng.pool, select=eng.select, draft_drop=eng.draft_drop,
                    drop_seed=eng.drop_seed, check=eng.check)
        args.update(kw)
        return cls(eng.model, ops if ops is not None else eng.ops, clone_twins(eng.twins),
                   [grow(x) for x in eng.store_k], [grow(x) for x in eng.store_v], [grow(x) for x in eng.store_bias],
                   eng.S_list, eng.M_alloc, **args)

    @classmethod
    def from_r2(cls, model, cache, st, ops, v_cap: int = 3, cap_mode: str = "family", **kw):
        """TickEngine.from_r2, then from_engine (the trash slot added)."""
        return cls.from_engine(TickEngine.from_r2(model, cache, st, ops, **kw), v_cap=v_cap, cap_mode=cap_mode)

    def clone(self, ops=None, **kw):
        kw.setdefault("v_cap", self.v_cap)
        kw.setdefault("cap_mode", self.cap_mode)
        return super().clone(ops, **kw)

    # ------------------------------------------------------------------ the driver (host)
    def choose_cap(self, R: int, T: int):
        """(R_cap, T_cap) of a tick with R rows and at most T rows per bank; a tick above G_v is refused."""
        if R > self.R_max or T > self.T_max:
            raise TickError("a tick of %d rows with %d rows in one bank exceeds the capacity %d / %d (v_cap %d): did "
                            "plan_tick change?" % (R, T, self.R_max, self.T_max, self.v_cap))
        if self.cap_mode == "max":
            return self.R_max, self.T_max
        if self.cap_mode == "exact":
            return R, max(T, 1)
        if R <= self.B and T <= 1:
            return self.B, 1
        return self.R_max, self.T_max

    def capacity_check(self, count: Sequence[int]) -> None:
        """C's condition for advance_unchecked (4fe5fc9): per bank, pos + n <= N_max and (pos + n) // 16 + 1 <=
        Lk_max on the host mirrors, once per tick."""
        n_max, lk_max = self.twins[0].n_max, self.twins[0].lk_max
        for bank, c in enumerate(count):
            if c:
                end = self.bank_pos[bank] + c
                if end > n_max or end // STRIDE + 1 > lk_max:
                    raise TickError("bank %d would reach position %d: N_max %d / Lk_max %d (a capacity the twins "
                                    "cannot hold)" % (bank, end, n_max, lk_max))

    def pad_row(self, T_cap: int) -> List[int]:
        """The padded rows' column, in XFIELDS order."""
        tr = self.trash
        vals = dict(pos=self.S_list[0], b=0, bank=0, u=0, ring_row=tr * BLOCK, prev_slot=tr, tail_slot=tr, draft=0,
                    lit=0, slot=0, live=0, logslot=self.LOG - 1, ucol=int(T_cap))
        return [vals[f] for f in XFIELDS]

    def padded_table(self, tab: torch.Tensor, count: Sequence[int], R: int, cb: CapBuffers, lo: int) -> torch.Tensor:
        """The host image of cb.flat: the live rows (TickEngine.row_table's columns, live = 1, logslot = lo + k,
        ucol = u), then the padded rows, then the per-bank live counts."""
        NF, R_cap = len(XFIELDS), cb.R_cap
        host = torch.empty(NF * R_cap + 2 * self.B, dtype=torch.int64)
        t = host[:NF * R_cap].view(NF, R_cap)
        nf = len(FIELDS)
        t[:nf, :R] = tab
        t[nf, :R] = 1
        t[nf + 1, :R] = torch.arange(lo, lo + R, dtype=torch.int64)
        t[nf + 2, :R] = tab[FIELDS.index("u")]
        if R_cap > R:
            t[:, R:] = torch.tensor(self.pad_row(cb.T_cap), dtype=torch.int64).view(NF, 1)
        host[NF * R_cap:] = torch.tensor(list(count), dtype=torch.int64)
        return host

    def cap_buffers(self, R_cap: int, T_cap: int) -> CapBuffers:
        key = (int(R_cap), int(T_cap))
        cb = self.bufs.get(key)
        if cb is None:
            cb = self.bufs[key] = CapBuffers(self, R_cap, T_cap)
        return cb

    def upload(self, cb: CapBuffers, host: torch.Tensor) -> None:
        """ONE host-to-device copy of the tick's table (pinned and non-blocking on CUDA: the caching host allocator
        keeps the staging block until the copy is done, also when draft-only ticks run ahead)."""
        if self.dev.type == "cuda":
            cb.flat.copy_(host.pin_memory(), non_blocking=True)
        else:
            cb.flat.copy_(host)

    def forward(self, cb: CapBuffers):
        """The region at cb's capacity (W4 replaces this call by a graph replay)."""
        return self._forward_fixed(cb)

    @torch.inference_mode()
    def run_rows(self, rows: Sequence, tokens: Sequence):
        """TickEngine.run_rows at fixed capacity (module docstring). Returns the LIVE rows' logits / argmax (views of
        the first R rows), their token-log slots, and the capacity used."""
        t_start = time.perf_counter()
        tab, count, T = self.row_table(rows, tokens)
        R = len(rows)
        R_cap, T_cap = self.choose_cap(R, T)
        self.capacity_check(count)
        lo = self.n_tok
        if lo + R >= self.LOG - 1:
            raise TickError("token log full (slot %d is the padded rows' trash)" % (self.LOG - 1))
        cb = self.cap_buffers(R_cap, T_cap)
        self.upload(cb, self.padded_table(tab, count, R, cb, lo))
        pos0 = list(self.bank_pos)
        if self.pre_region is not None:
            self.pre_region(self, cb, rows)
        logits, arg = self.forward(cb)
        if self.post_region is not None:
            self.post_region(self, cb, rows)
        self.n_tok = lo + R
        for bk, c in enumerate(count):
            self.bank_pos[bk] += c
        self.last_ctr = [cb.ctr[l] for l in range(self.L)]
        self.ctr_epoch = list(self.epoch_host)
        self.last_pos0, self.last_count = pos0, list(count)
        cb.uses += 1
        self.stats["ticks"] += 1
        self.stats["rows"] += R
        self.stats["padded_rows"] += R_cap - R
        self.stats["caps"][(R_cap, T_cap)] = self.stats["caps"].get((R_cap, T_cap), 0) + 1
        nd = sum(1 for r in rows if r.kind == "d")
        self.stats["draft_rows"] += nd
        self.stats["verify_rows"] += R - nd
        self.stats["ms_rows"] += 1e3 * (time.perf_counter() - t_start)
        return SimpleNamespace(logits=logits[:R], argmax=arg[:R], slots=list(range(lo, lo + R)), lo=lo, hi=lo + R,
                               cap=(R_cap, T_cap))

    # ------------------------------------------------------------------ THE REGION
    def _forward_fixed(self, cb: CapBuffers):
        """ONE flattened forward over cb's R_cap rows (module docstring, F2-F9). No host read, no state allocation,
        no host branch on a device value."""
        nl, ops, model = self.nl, self.ops, self.model
        d = cb.tab
        pos, rb, bank, u, ring_row = d[0], d[1], d[2], d[3], d[4]
        prev_slot = d[5].to(torch.int32)
        tail_slot = d[6].to(torch.int32)
        is_draft = d[7] != 0
        lit, slot = d[8], d[9]
        live = d[10] != 0
        logslot, ucol = d[11], d[12]
        pos32 = pos.to(torch.int32).contiguous()
        rb32 = rb.to(torch.int32).contiguous()
        M_b = (pos // BLOCK + 1).to(torch.int32).contiguous()
        last = (pos % BLOCK + 1).masked_fill(~live, 1).to(torch.int32).contiguous()
        cb.cu_k[:cb.R_cap].copy_(bank * self.twins[0].lk_max)
        ids = torch.where(lit >= 0, lit, self.tok_log[slot]).view(cb.R_cap, 1)
        hidden = F.embedding(ids, model.embed_tokens)
        for l, layer in enumerate(model.layers):
            hidden = self._layer_fixed(cb, l, layer, hidden, pos, pos32, rb, rb32, bank, u, ucol, ring_row, prev_slot,
                                       tail_slot, is_draft, live, M_b, last)
        hidden = nl.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
        logits = ops.linear(hidden, model.lm_head).float()[:, -1]
        arg = logits.argmax(-1)
        self.tok_log.index_put_((logslot,), arg)
        return logits, arg

    def _layer_fixed(self, cb, l, layer, hidden, pos, pos32, rb, rb32, bank, u, ucol, ring_row, prev_slot, tail_slot,
                     is_draft, live, M_b, last):
        """TickEngine._layer at fixed capacity (module docstring)."""
        nl, lin = self.nl, self.ops.linear
        R, T_cap = cb.R_cap, cb.T_cap
        Hkv, D = layer.num_key_value_heads, layer.head_dim
        twin = self.twins[l]
        residual = hidden
        h = nl.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = lin(h, layer.wqkv)
        q, k, v, cis = nl.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
        q = q.view(R, -1)
        k = k.view(R, -1)
        nl.apply_rope_with_cos_sin_cache_inplace(pos32, q, k, D, self.model.cos_sin_cache, True)
        q = q.reshape(R, layer.num_heads, D)
        k = k.reshape(R, Hkv, D)
        v = v.reshape(R, Hkv, D)
        cis = cis.reshape(R, Hkv)
        # 2. the positional tables: the live rows in their (bank, u) cells, the padded rows in column T_cap (F3, F4)
        cb.keys.zero_()
        cb.cis.zero_()
        cb.keys.index_put_((bank, ucol), k.to(cb.keys.dtype))
        cb.cis.index_put_((bank, ucol), cis.to(cb.cis.dtype))
        ctr = twin.advance_unchecked(cb.keys[:, :T_cap], cb.cis[:, :T_cap], cb.cnt)        # (T_cap + 1, 4, 2B)
        cb.ctr[l].copy_(ctr)
        n_used = ctr[(u + 1).clamp(max=T_cap), 1, bank].masked_fill(~live, 1).to(torch.int32).contiguous()
        # 3. stage 1 with the alias
        ck = twin.comp_k_flat()
        score = self.stage1(q, ck, ck, cb.cu_q, cb.cu_k, 1, twin.lk_max, seqused_k=n_used)
        # 4. pooling per row, the deterministic selection
        ccis = twin.ccis.index_select(1, bank)
        self.pool(score, ccis, n_used, M_b, cb.pq, cb.pc, stride=layer.pooling_stride, block_size=layer.block_size,
                  local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
        sel = self.select(cb.pq.float(), cb.pc.float(), self.nl.QK_SELECT, layer.topk_blocks)
        if self.check:                                                          # F8: a device flag, live rows only
            blk = (pos // BLOCK).view(1, R)
            bad = live.view(1, R) & ((sel[..., -1] != blk) | (sel[..., -2] != blk - 1))
            self.err_dev.logical_or_(bad.any())
        # 5. the rows' own K / V / bias into their ring pages (padded rows: the trash slot)
        self.store_k[l].index_put_((rb, ring_row), k.to(self.dtype))
        self.store_v[l].index_put_((rb, ring_row), v.to(self.dtype))
        self.store_bias[l].index_put_((rb, ring_row), cis.to(self.dtype))
        # 6. the page lists (F9: no host branch) and ONE BI attention call (F7: a padded row reads one trash page)
        table, npg = self.page_table(l, sel, pos, rb, prev_slot, tail_slot, is_draft, True, layer)
        table[:, :, 0].masked_fill_(~live.view(R, 1), self.trash)
        npg = npg.masked_fill(~live.view(R, 1), 1)
        attn = self.ops.attn.bi_paged_decode(q, self.store_k[l], self.store_v[l], self.store_bias[l], rb32, table,
                                             npg, last)
        attn = attn.view(R, 1, -1)
        hidden = residual + lin(attn, layer.wo)
        residual = hidden
        h = nl.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = lin(h, layer.gate_up_proj)
        nl.silu_and_mul(h, cb.ffn)
        return residual + lin(cb.ffn, layer.down_proj)

    # ------------------------------------------------------------------ the decision read
    def read_tokens(self, lo: int, hi: int) -> List[int]:
        """THE decision read: token-log slots [lo, hi), err_dev and every twin's overflow flag in ONE transfer. A set
        flag voids every tick since the last clean read (TickError). With a flag_watcher (S-C, watch_flags_with) the
        flags ride in S-C's own read at every tick, and this read holds the tokens only."""
        self.stats["decision_reads"] += 1
        flags = None
        if self.flag_watcher is None:
            flags = torch.stack([self.err_dev] + [t.overflow for t in self.twins]).to(torch.int64)
        vals = (self.tok_log[lo:hi] if flags is None else torch.cat([self.tok_log[lo:hi], flags])).tolist()
        n = hi - lo
        self._raise_on_flags(vals[n:])
        return vals[:n]

    def watch_flags_with(self, svc) -> None:
        """C's watch_flag (r2_residency_sc.py, 0d3a94a, Q-T5): S-C reads err_dev ("tick.err") and every twin's sticky
        overflow ("twin<l>.overflow") with its ONE device read at every end_tick; a set flag refuses that tick
        (CapacityRefusal). The decision read then reads the tokens only."""
        svc.watch_flag("tick.err", self.err_dev)
        for l, t in enumerate(self.twins):
            svc.watch_flag("twin%d.overflow" % l, t.overflow)
        self.flag_watcher = svc

    def check_flags(self) -> None:
        """The flags alone (one read): for the end of a run and the tests."""
        flags = torch.stack([self.err_dev] + [t.overflow for t in self.twins]).to(torch.int64).tolist()
        self._raise_on_flags(flags)

    def _raise_on_flags(self, flags: Sequence[int]) -> None:
        if any(flags):
            raise TickError("void tick(s) since the last clean read: selection check %s, twin overflow in layers %s"
                            % (bool(flags[0]), [l for l, f in enumerate(flags[1:]) if f]))

    # ------------------------------------------------------------------ E1: the events without a host read
    def commit_banks(self, keep: Dict[int, int]) -> None:
        """Every exact bank b in keep goes back to this tick's counters after keep[b] of its rows; the other banks
        keep this tick's end state. PositionalTwin.commit's checks run on the HOST MIRRORS; then commit_unchecked per
        layer (no device read)."""
        U = int(self.last_ctr[0].shape[0]) - 1
        if self.ctr_epoch != self.epoch_host:
            raise TickError("stale snapshot: this tick's counters were taken before the last commit")
        full = [U] * (2 * self.B)
        for b, k in keep.items():
            if not 0 <= k <= U:
                raise TickError("bank %d: keep %d outside the snapshot [0, %d]" % (b, k, U))
            back = self.bank_pos[b] - (self.last_pos0[b] + k)
            if back < 0:
                raise TickError("bank %d: the snapshot is ahead of the state" % b)
            if back > MAX_ROLLBACK:
                raise TickError("bank %d: rollback over %d positions; the key ring allows %d" % (b, back, MAX_ROLLBACK))
            full[b] = k
        idx = self._to_device(torch.tensor(full, dtype=torch.int64))
        for l in range(self.L):
            commit_unchecked(self.twins[l], self.last_ctr[l], idx)
        self.epoch_host = [e + 1 for e in self.epoch_host]

    @torch.inference_mode()
    def apply_events(self) -> None:
        """TickEngine.apply_events (the same order and host checks) with commit_banks and
        rebuild_from_own_rows_unchecked: no device read."""
        if not self.events:
            return
        rb, pub = {}, {}
        for kind, b, p in self.events:
            (rb if kind == "rb" else pub)[b] = p
            if kind == "rb":
                pub.pop(b, None)
        self.events = []
        B = self.B
        if rb:                                                                   # 1. the exact banks: ONE commit
            keep = {}
            for b, p in rb.items():
                idx = p - self.last_pos0[b]
                if not 0 <= idx <= self.last_count[b] or p < self.committed[b]:
                    raise TickError("rollback of request %d to %d: outside this tick's rows (%d + %d) or below the "
                                    "committed %d" % (b, p, self.last_pos0[b], self.last_count[b], self.committed[b]))
                keep[b] = idx
            self.commit_banks(keep)
            for b, p in rb.items():
                self.bank_pos[b] = p
                self.committed[b] = p
                self.stats["rollbacks"] += 1
        for b, p in pub.items():
            if p != self.bank_pos[b] or p < self.committed[b]:
                raise TickError("publication of request %d to %d: its exact bank is at %d, committed %d"
                                % (b, p, self.bank_pos[b], self.committed[b]))
            self.committed[b] = p
            self.stats["publications"] += 1
        for b in set(rb) | set(pub):                                             # 2. write-back
            self.writeback(b)
        for b in rb:                                                             # 3. the draft banks
            for l in range(self.L):
                SC.sync_draft_bank(self.twins[l], b, B)
                self.copy_ring(l, b, exact_to_draft=True)
            self.epoch_host[B + b] += 1
            self.bank_pos[B + b] = self.bank_pos[b]
            self.synced[b] = self.bank_pos[b]
        for b, p in pub.items():
            dpos = self.bank_pos[B + b]
            if dpos < p:
                raise TickError("request %d: the draft bank (%d) is behind the publication (%d)" % (b, dpos, p))
            for l in range(self.L):
                rebuild_from_own_rows_unchecked(self.twins[l], b, B, p, dpos)
                self.copy_rows(l, b, range(self.synced[b], p))
            self.epoch_host[B + b] += 1
            self.synced[b] = p
