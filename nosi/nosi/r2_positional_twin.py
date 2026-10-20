"""R2, milestone 2, step 2: a pure-torch CPU TWIN of the positional tables of
ONE NOSA layer, PER REQUEST.

WHY. The shipped engine keeps one scalar counter per table for the whole batch
(seq_length, cis_len, tail_cis_len, comp_cis_len, no_compress_k_len,
cached_compressed_max_seqlen, _tail_block_len_on_gpu; cache_engine.py:189,
:345, :357, :678-691, :669, :727, :731, :753, :770, :781), so every request
must advance in lockstep. R2 needs requests that advance by different amounts
and that roll back. The twin replaces each scalar counter with a (B,) counter
and each growing table with a preallocated table indexed by those counters. A
rollback is then a counter reset: no table row is copied.

This file imports only torch. It is the reference that the R2 engine's tables
must match; it is not the engine. Load it by file path (nosi/nosi/__init__.py
imports CUDA modules).

TABLES (B requests, H kv heads, D head dim):
  pos       (B,) int64   tokens held; the next step writes position pos
  n_comp    (B,) int64   compressed keys held
  n_ccis    (B,) int64   compressed cis held
  epoch     (B,) int64   commits (and prefills) so far; guards stale snapshots
  comp_k    (B, Lk_max, H, D)  compressed keys, request b in rows [0, n_comp[b]).
            Flat view (B * Lk_max, H, D) with the CONSTANT int32
            cu = arange(B + 1) * Lk_max and seqused_k = n_comp.
            Shipped: compress_k_cache_varlen, grown by torch.cat
            (cache_engine.py:667), cu += arange(B + 1) (:668).
  ring      (B, 64, H, D)      raw keys, position q at row q % 64.
            Shipped: no_compress_k_cache (B, 32, H, D) with shifts (:676-692).
  total_cis (B, N_max, H)      raw cis, position q at row q.
            Shipped: (B, S + 1024, H) (:768; findings F3).
  ccis      (H, B, Lk_max)     compressed cis, request b in [:, b, :n_ccis[b]].
            Shipped: compressed_cis (H, B, M) (:728), which never grows
            (findings F2). The (H, B, .) layout is the pooling input layout
            (max_pooling_fused.py:9-10).

DERIVED per request (no state):
  tail_row()   = pos % 64        the row of tail slot 63 that the next token
                                 writes (shipped _tail_block_len_on_gpu, :309, :357, :408)
  tail_block() = pos // 64       the block id that tail slot 63 holds
                                 (shipped _block_map[..., 63], :321, :324, :407)
  cache_lens() = 4032 + pos % 64 int32, what attention reads after the step
                                 that wrote pos - 1 (shipped _cache_lens, :327,
                                 :346, :409). It reproduces findings F4 on purpose.
  grid_width() = pos // 64 + 1   the pooling width that a step writing position
                                 pos needs: the shipped warm-up value
                                 (nosa_llama.py:817-818), per request and per step
                                 (the fix of findings F1).
  seqused_k()  = n_comp as int32 (stage-1 per-request key lengths)

ONE STEP. Request b writes position p = pos[b], in the shipped order
(nosa_llama.py:575-585, then :618):
  1. key compression, DEFERRED: if 16 n_comp + 32 <= p, take the 32 keys of
     window n_comp (positions 16 n_comp .. 16 n_comp + 31, all < p) from the
     ring IN POSITION ORDER, shape (k, 32, H, D), and .mean(1)
     (nosa_llama.py:577). Write comp_k[b, n_comp]; n_comp += 1.
     Shipped: cache_engine.py:683 fires when no_compress_k_len >= 32 BEFORE
     the write of the new key.
  2. ring[b, p % 64] = key.
  3. total_cis[b, p] = cis.
  4. cis compression, IMMEDIATE: if 16 n_ccis + 32 <= p + 1, take total_cis
     rows 16 n_ccis .. 16 n_ccis + 31 (all <= p), permute to (H, k, 32) and
     .mean(-1) (cache_engine.py:757). Write ccis[:, b, n_ccis]; n_ccis += 1.
     Shipped: :754 fires when tail_cis_len reaches 32 AFTER the write; the
     twin appends the summary as the commented-out cat at :758 intends.
  5. pos += 1.
The state after a step is the state that the step's kernels read: stage 1
reads comp_k after (1), the pooling reads ccis after (4), attention reads
cache_lens after (5). At most one event of each kind is due per step
(check_invariants).

PREFILL (one request, S >= 32): n_comp = n_ccis = S // 16 - 1 windows
(nosa_llama.py:69; cache_engine.py:725), computed with the shipped ops
(index_select + view + .mean(1), nosa_llama.py:103-108; avg_pool1d,
cache_engine.py:724-725). The ring gets the last min(S, 64) keys, total_cis
rows [0, S) the cis, pos = S.

ROLLBACK. counters() returns (4, B): pos, n_comp, n_ccis, epoch. advance()
returns ctr (T + 1, 4, B) with ctr[u] = the counters after min(u, n[b]) tokens.
commit(ctr, keep) sets request b to ctr[keep[b], :, b] and increments epoch.
WHY IT IS EXACT: a step writes comp_k, ccis and total_cis only at indices >=
the counters it started from, so a reset leaves every row below the counters
as it was. The ring is the one exception: a write at position w overwrites the
row of position w - 64. After any step (and after prefill),
16 n_comp + 32 > pos - 1, so the live keys [16 n_comp, pos) are at most 32
positions. A rollback over m <= 32 positions therefore never overwrites a live
row; commit() refuses m > 32 (MAX_ROLLBACK). epoch makes commit() refuse a
snapshot taken before the last commit: its rows can belong to a branch that a
rollback discarded.

REFUSALS (an exception; the step changes nothing): a full comp_k or ccis
(Lk_max), a full total_cis (N_max), a prompt shorter than 32, a step for a
request without a prefill, a rollback over more than 32 positions, a stale
snapshot. The shipped engine instead writes into an empty slice (findings F2,
F3 (a)). advance() is atomic per step, not per call: if step u raises, steps
0 .. u-1 stay.

NOT COVERED: kernels (stage 1, pooling, top-k, diff_offload, the host gathers,
attention), the GPU slot cache (_k_gpu, _v_gpu, _kv_bias_gpu, _block_map,
_load_mask), the host K/V buffers, CUDA graphs and streams, GPU reduction
order, more than one layer. See docs/R2_ENGINE_FINDINGS.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

KERNEL = 32          # compression window (nosa_llama.py:678; cache_engine.py:672)
STRIDE = 16          # compression stride (nosa_llama.py:677; cache_engine.py:672)
BLOCK = 64           # KV block (nosa_llama.py:680)
TOPK = 64            # attended slots; tail slot TOPK - 1 (nosa_llama.py:681; cache_engine.py:312)
RING = 64            # raw-key ring rows
MAX_ROLLBACK = RING - KERNEL          # 32, see ROLLBACK above
TAIL_BASE = (TOPK - 1) * BLOCK        # 4032


class TwinCapacityError(RuntimeError):
    """A table is full. The step changed nothing."""


class TwinRollbackError(RuntimeError):
    """commit() refused the snapshot. The state did not change."""


class PositionalTwin:
    """Per-request positional tables of one layer (see the module docstring)."""

    def __init__(self, batch: int, heads: int, head_dim: int, lk_max: int, n_max: int,
                 dtype: torch.dtype = torch.float32, device="cpu"):
        if batch < 1 or heads < 1 or head_dim < 1:
            raise ValueError("batch, heads and head_dim must be >= 1")
        if lk_max < 1 or n_max < KERNEL:
            raise ValueError("lk_max must be >= 1 and n_max >= %d" % KERNEL)
        dev = torch.device(device)
        self.B, self.H, self.D = int(batch), int(heads), int(head_dim)
        self.lk_max, self.n_max = int(lk_max), int(n_max)
        self.pos = torch.zeros(self.B, dtype=torch.int64, device=dev)
        self.n_comp = torch.zeros(self.B, dtype=torch.int64, device=dev)
        self.n_ccis = torch.zeros(self.B, dtype=torch.int64, device=dev)
        self.epoch = torch.zeros(self.B, dtype=torch.int64, device=dev)
        self.prefilled = torch.zeros(self.B, dtype=torch.bool, device=dev)
        self.comp_k = torch.zeros(self.B, self.lk_max, self.H, self.D, dtype=dtype, device=dev)
        self.cu = torch.arange(self.B + 1, dtype=torch.int32, device=dev) * self.lk_max
        self.ring = torch.zeros(self.B, RING, self.H, self.D, dtype=dtype, device=dev)
        self.total_cis = torch.zeros(self.B, self.n_max, self.H, dtype=dtype, device=dev)
        self.ccis = torch.zeros(self.H, self.B, self.lk_max, dtype=dtype, device=dev)
        self._win = torch.arange(KERNEL, dtype=torch.int64, device=dev)
        # the decode summaries' mean (step): None = torch's .mean (the shipped op); the BI mode of R2 step 7
        # (r2_bi_decode.attach_bi) sets a fixed-order mean, whose summation order does not depend on B
        self.mean_fn = None
        # step_unchecked (R2 step 7 tick): a STICKY device flag, True once an unchecked step met a full table or a
        # bank without a prefill (that step wrote nothing for that bank); the caller reads it once per tick
        self.overflow = torch.zeros((), dtype=torch.bool, device=dev)

    def _mean(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.mean(dim=dim) if self.mean_fn is None else self.mean_fn(x, dim)

    # ------------------------------------------------------------------ events
    def _key_due(self, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Deferred: window n holds positions 16 n .. 16 n + 31, all before p."""
        return STRIDE * n + KERNEL <= p

    def _cis_due(self, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Immediate: window n ends at position p."""
        return STRIDE * n + KERNEL <= p + 1

    # ----------------------------------------------------------------- prefill
    def prefill(self, b: int, keys: torch.Tensor, cis: torch.Tensor) -> None:
        """Load request b's prompt: keys (S, H, D), cis (S, H), S >= 32."""
        if not 0 <= b < self.B:
            raise IndexError("request %d outside batch %d" % (b, self.B))
        if keys.dim() != 3 or tuple(keys.shape[1:]) != (self.H, self.D):
            raise ValueError("keys must be (S, %d, %d), got %s" % (self.H, self.D, tuple(keys.shape)))
        S = int(keys.shape[0])
        if tuple(cis.shape) != (S, self.H):
            raise ValueError("cis must be (%d, %d), got %s" % (S, self.H, tuple(cis.shape)))
        if S < KERNEL:
            raise ValueError("a prompt of %d tokens has no complete compression window (needs %d)" % (S, KERNEL))
        n = S // STRIDE - 1
        if n > self.lk_max:
            raise TwinCapacityError("prompt of %d tokens needs %d compressed rows, Lk_max is %d" % (S, n, self.lk_max))
        if S > self.n_max:
            raise TwinCapacityError("prompt of %d tokens exceeds N_max %d" % (S, self.n_max))
        dev = self.pos.device
        keys = keys.to(device=dev, dtype=self.comp_k.dtype)
        cis = cis.to(device=dev, dtype=self.total_cis.dtype)
        # compressed keys: compress_k (nosa_llama.py:97-109) for one request
        idx = (STRIDE * torch.arange(n, dtype=torch.int64, device=dev)[:, None] + self._win[None, :]).reshape(-1)
        self.comp_k[b, :n] = keys.index_select(0, idx).view(n, KERNEL, self.H, self.D).mean(dim=1)
        # raw keys: the last min(S, RING) positions
        q = torch.arange(max(0, S - RING), S, dtype=torch.int64, device=dev)
        self.ring[b, q % RING] = keys[q]
        # raw cis
        self.total_cis[b, :S] = cis
        # compressed cis: update_cis prefill (cache_engine.py:722-729) for one request
        full = (S // STRIDE) * STRIDE
        x = cis.t()[:, :full].reshape(self.H, 1, full)
        self.ccis[:, b, :n] = F.avg_pool1d(x, kernel_size=KERNEL, stride=STRIDE).reshape(self.H, n)
        self.pos[b] = S
        self.n_comp[b] = n
        self.n_ccis[b] = n
        self.prefilled[b] = True
        self.epoch[b] += 1

    # -------------------------------------------------------------------- step
    def key_window(self, b: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Raw keys at positions start .. start + 31 of requests b, in position
        order: (k, 32, H, D). The caller guarantees the positions are live."""
        rows = (start[:, None] + self._win[None, :]) % RING
        return self.ring[b[:, None], rows]

    def step(self, keys: torch.Tensor, cis: torch.Tensor, active=None) -> torch.Tensor:
        """Write one token for every active request. keys (B, H, D), cis (B, H),
        active (B,) bool (default: all). Returns the positions written, (B,)
        int64, -1 for an inactive request."""
        if tuple(keys.shape) != (self.B, self.H, self.D):
            raise ValueError("keys must be (%d, %d, %d), got %s" % (self.B, self.H, self.D, tuple(keys.shape)))
        if tuple(cis.shape) != (self.B, self.H):
            raise ValueError("cis must be (%d, %d), got %s" % (self.B, self.H, tuple(cis.shape)))
        dev = self.pos.device
        if active is None:
            active = torch.ones(self.B, dtype=torch.bool, device=dev)
        active = torch.as_tensor(active, device=dev).to(torch.bool)
        if tuple(active.shape) != (self.B,):
            raise ValueError("active must be (%d,), got %s" % (self.B, tuple(active.shape)))
        written = torch.full((self.B,), -1, dtype=torch.int64, device=dev)
        b = active.nonzero().flatten()
        if b.numel() == 0:
            return written
        if not bool(self.prefilled[b].all()):
            raise RuntimeError("step for a request without a prefill")
        p = self.pos[b]
        n_k = self.n_comp[b]
        n_c = self.n_ccis[b]
        due_k = self._key_due(p, n_k)
        due_c = self._cis_due(p, n_c)
        # refusals first, so a refused step changes nothing
        if bool((p >= self.n_max).any()):
            raise TwinCapacityError("total_cis is full (N_max %d)" % self.n_max)
        if bool((n_k[due_k] >= self.lk_max).any()) or bool((n_c[due_c] >= self.lk_max).any()):
            raise TwinCapacityError("a compressed table is full (Lk_max %d)" % self.lk_max)
        keys = keys.to(device=dev, dtype=self.ring.dtype)
        cis = cis.to(device=dev, dtype=self.total_cis.dtype)
        # SHAPE-STABLE SUMMARIES (R2 step 6, the ragged isolation gate): both means below are computed for EVERY
        # request of the batch ((B, 32, H, D) and (H, B, 32)) and only the due requests are written. The reduction
        # kernel, and so the summation order of request b's summary, then never depends on how many requests are due
        # this step: a ragged batch computes request b's summary with the same kernel launch as an equal-advance
        # batch of the same B. The windows of the requests that are not due are finite reads (ring rows, total_cis
        # rows, clamped), and they are discarded. With every request due (equal advance) this is the former call.
        all_b = torch.arange(self.B, device=dev)
        # 1. deferred key compression
        bk = b[due_k]
        if bk.numel():
            nk = n_k[due_k]
            means = self._mean(self.key_window(all_b, STRIDE * self.n_comp), 1)            # (B, H, D)
            self.comp_k[bk, nk] = means[bk]
            self.n_comp[bk] = nk + 1
        # 2. raw key
        self.ring[b, p % RING] = keys[b]
        # 3. raw cis
        self.total_cis[b, p] = cis[b]
        # 4. immediate cis compression
        bc = b[due_c]
        if bc.numel():
            nc = n_c[due_c]
            rows = (STRIDE * self.n_ccis[:, None] + self._win[None, :]).clamp(max=self.n_max - 1)
            win = self.total_cis[all_b[:, None], rows]                                     # (B, 32, H)
            means_c = self._mean(win.permute(2, 0, 1).contiguous(), -1)                    # (H, B)
            self.ccis[:, bc, nc] = means_c[:, bc]
            self.n_ccis[bc] = nc + 1
        # 5.
        self.pos[b] = p + 1
        written[b] = p
        return written

    # ------------------------------------------------- the unchecked step (R2 step 7 tick)
    def step_unchecked(self, keys: torch.Tensor, cis: torch.Tensor, active: torch.Tensor) -> None:
        """step() WITHOUT A HOST READ (the flattened tick, R2 step 7; step() stays the default and the reference).
        keys (B, H, D), cis (B, H), active (B,) bool ON THE TWIN'S DEVICE. Every bank goes through the same masked
        ops (no nonzero, no bool(), no index list): the events of step() for the active banks, a write-back of the
        value read for the others. The summaries are step()'s: the same shape-stable means over ALL B banks through
        the same mean_fn hook, so a bank's tables are bitwise step()'s.
        CAPACITY. The caller guarantees it from the preallocated widths with ONE batched host check per tick on its
        host mirrors (pos + n <= N_max and (pos + n) // 16 + 1 <= Lk_max for every bank). Should a bank still meet a
        full table (or lack a prefill), that bank writes NOTHING in this step (no clamped write lands) and the sticky
        flag self.overflow turns True; the caller reads it once per tick. Never a silent out-of-range write.
        A TRUE FLAG VOIDS THE TICK: the flagged bank did not advance while its rows were computed, so the caller must
        refuse every decision of that tick (raise), never commit it. (Contributed by the verify agent, reviewed by
        track C under conditions (a)-(d) of the Q3 answer, 2026-10-21.)"""
        dev = self.pos.device
        if tuple(keys.shape) != (self.B, self.H, self.D) or tuple(cis.shape) != (self.B, self.H):
            raise ValueError("keys (%d, %d, %d) and cis (%d, %d) expected" % (self.B, self.H, self.D, self.B, self.H))
        if not torch.is_tensor(active) or active.dtype != torch.bool or tuple(active.shape) != (self.B,) \
                or active.device != dev:
            raise ValueError("active must be a (%d,) bool tensor on %s" % (self.B, dev))
        all_b = torch.arange(self.B, device=dev)
        p, n_k, n_c = self.pos.clone(), self.n_comp.clone(), self.n_ccis.clone()
        due_k = active & self._key_due(p, n_k)
        due_c = active & self._cis_due(p, n_c)
        bad = (active & ((p >= self.n_max) | ~self.prefilled)) | (due_k & (n_k >= self.lk_max)) \
            | (due_c & (n_c >= self.lk_max))
        self.overflow |= bad.any()
        a = active & ~bad
        keys = keys.to(device=dev, dtype=self.ring.dtype)
        cis = cis.to(device=dev, dtype=self.total_cis.dtype)
        # 1. deferred key compression: step()'s mean over all banks; the due banks write it
        means = self._mean(self.key_window(all_b, STRIDE * n_k), 1)                    # (B, H, D)
        wk = due_k & a
        ik = n_k.clamp(max=self.lk_max - 1)
        self.comp_k[all_b, ik] = torch.where(wk.view(-1, 1, 1), means, self.comp_k[all_b, ik])
        self.n_comp += wk.to(self.n_comp.dtype)
        # 2. raw key
        r = p % RING
        self.ring[all_b, r] = torch.where(a.view(-1, 1, 1), keys, self.ring[all_b, r])
        # 3. raw cis
        ip = p.clamp(max=self.n_max - 1)
        self.total_cis[all_b, ip] = torch.where(a.view(-1, 1), cis, self.total_cis[all_b, ip])
        # 4. immediate cis compression: step()'s window and mean over all banks; the due banks write it
        rows = (STRIDE * n_c[:, None] + self._win[None, :]).clamp(max=self.n_max - 1)
        win = self.total_cis[all_b[:, None], rows]                                     # (B, 32, H)
        means_c = self._mean(win.permute(2, 0, 1).contiguous(), -1)                    # (H, B)
        wc = due_c & a
        ic = n_c.clamp(max=self.lk_max - 1)
        self.ccis[:, all_b, ic] = torch.where(wc.view(1, -1), means_c, self.ccis[:, all_b, ic])
        self.n_ccis += wc.to(self.n_ccis.dtype)
        # 5.
        self.pos += a.to(self.pos.dtype)

    def advance_unchecked(self, keys: torch.Tensor, cis: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """advance() over step_unchecked: keys (B, T, H, D), cis (B, T, H), n (B,) int64 ON THE DEVICE (T, the host
        bound, is keys.shape[1]). Returns ctr (T + 1, 4, B) on the device, without a host read."""
        T = int(keys.shape[1])
        if keys.dim() != 4 or keys.shape[0] != self.B or tuple(cis.shape) != (self.B, T, self.H):
            raise ValueError("keys (B, T, H, D) and cis (B, T, H) expected")
        if not torch.is_tensor(n) or n.dtype != torch.int64 or tuple(n.shape) != (self.B,) or n.device != self.pos.device:
            raise ValueError("n must be a (%d,) int64 tensor on the twin's device" % self.B)
        ctr = [self.counters()]
        for u in range(T):
            self.step_unchecked(keys[:, u], cis[:, u], n > u)
            ctr.append(self.counters())
        return torch.stack(ctr)

    # ------------------------------------------------------- commit / rollback
    def counters(self) -> torch.Tensor:
        """(4, B) int64: pos, n_comp, n_ccis, epoch (a copy)."""
        return torch.stack((self.pos, self.n_comp, self.n_ccis, self.epoch))

    def advance(self, keys: torch.Tensor, cis: torch.Tensor, n) -> torch.Tensor:
        """Request b writes keys[b, :n[b]], cis[b, :n[b]]. keys (B, T, H, D),
        cis (B, T, H), n (B,) ints in [0, T]. Returns ctr (T + 1, 4, B):
        ctr[u] = the counters after min(u, n[b]) tokens of request b."""
        if keys.dim() != 4 or keys.shape[0] != self.B:
            raise ValueError("keys must be (%d, T, %d, %d), got %s" % (self.B, self.H, self.D, tuple(keys.shape)))
        T = int(keys.shape[1])
        if tuple(cis.shape) != (self.B, T, self.H):
            raise ValueError("cis must be (%d, %d, %d), got %s" % (self.B, T, self.H, tuple(cis.shape)))
        n = torch.as_tensor(n, dtype=torch.int64, device=self.pos.device)
        if tuple(n.shape) != (self.B,) or bool(((n < 0) | (n > T)).any()):
            raise ValueError("n must be (%d,) with entries in [0, %d]" % (self.B, T))
        ctr = [self.counters()]
        for u in range(T):
            self.step(keys[:, u], cis[:, u], active=n > u)
            ctr.append(self.counters())
        return torch.stack(ctr)

    def commit(self, ctr: torch.Tensor, keep) -> None:
        """Set request b to ctr[keep[b], :, b]. ctr comes from advance() (or is a
        stack of counters()) taken after the last commit."""
        dev = self.pos.device
        if ctr.dim() != 3 or tuple(ctr.shape[1:]) != (4, self.B):
            raise ValueError("ctr must be (U + 1, 4, %d), got %s" % (self.B, tuple(ctr.shape)))
        keep = torch.as_tensor(keep, dtype=torch.int64, device=dev)
        if tuple(keep.shape) != (self.B,) or bool(((keep < 0) | (keep >= ctr.shape[0])).any()):
            raise ValueError("keep must be (%d,) with entries in [0, %d)" % (self.B, ctr.shape[0]))
        sel = ctr.to(device=dev, dtype=torch.int64).permute(2, 0, 1)[torch.arange(self.B, device=dev), keep]  # (B, 4)
        if not torch.equal(sel[:, 3], self.epoch):
            raise TwinRollbackError("stale snapshot: taken before the last commit or prefill")
        back = self.pos - sel[:, 0]
        if bool((back < 0).any()):
            raise TwinRollbackError("snapshot is ahead of the state")
        if bool((back > MAX_ROLLBACK).any()):
            raise TwinRollbackError("rollback over %d positions; the %d-row key ring allows %d"
                                    % (int(back.max()), RING, MAX_ROLLBACK))
        self.pos.copy_(sel[:, 0])
        self.n_comp.copy_(sel[:, 1])
        self.n_ccis.copy_(sel[:, 2])
        self.epoch += 1

    # ----------------------------------------------------------------- derived
    def tail_row(self) -> torch.Tensor:
        return self.pos % BLOCK

    def tail_block(self) -> torch.Tensor:
        return self.pos // BLOCK

    def cache_lens(self) -> torch.Tensor:
        return (TAIL_BASE + self.pos % BLOCK).to(torch.int32)

    def grid_width(self) -> torch.Tensor:
        return self.pos // BLOCK + 1

    def seqused_k(self) -> torch.Tensor:
        return self.n_comp.to(torch.int32).contiguous()

    def comp_k_flat(self) -> torch.Tensor:
        return self.comp_k.view(self.B * self.lk_max, self.H, self.D)

    def live_keys(self, b: int) -> torch.Tensor:
        """Raw keys of request b not yet compressed: positions [16 n_comp, pos),
        in order (shipped: no_compress_k_cache[b, :no_compress_k_len])."""
        lo, hi = STRIDE * int(self.n_comp[b]), int(self.pos[b])
        q = torch.arange(lo, hi, dtype=torch.int64, device=self.pos.device)
        return self.ring[b, q % RING]

    def check_invariants(self) -> None:
        """Raise AssertionError if a prefilled request breaks a counter
        invariant. With last = pos - 1, the last position written:
          16 n_comp + 32 >  last       no deferred key event is overdue
          16 n_comp + 16 <= last + 1   every compressed key is complete
          16 n_ccis + 32 >  last + 1   no cis event is overdue
          16 n_ccis + 16 <= last + 1   every compressed cis is complete
          n_ccis >= n_comp             the pooling reads ccis[:, b, :n_comp[b]]
        The first one bounds the live keys to 32 positions (MAX_ROLLBACK)."""
        b = self.prefilled.nonzero().flatten()
        if b.numel() == 0:
            return
        last = self.pos[b] - 1
        n, m = self.n_comp[b], self.n_ccis[b]
        checks = (
            (STRIDE * n + KERNEL > last, "a deferred key compression is overdue"),
            (STRIDE * n + STRIDE <= last + 1, "a compressed key covers an unwritten position"),
            (STRIDE * m + KERNEL > last + 1, "a cis compression is overdue"),
            (STRIDE * m + STRIDE <= last + 1, "a compressed cis covers an unwritten position"),
            (m >= n, "fewer compressed cis than compressed keys"),
        )
        for ok, msg in checks:
            if not bool(ok.all()):
                raise AssertionError(msg)
