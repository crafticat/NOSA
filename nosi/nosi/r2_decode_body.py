"""R2 step 3: the engine decode body with PER-REQUEST state.

The plan, table (1) and the gates are in docs/R2_STEP3_DECODE_BODY.md.

WHAT IT REPLACES. The shipped decode update keeps one tail counter for the
batch (seq_length, _tail_block_len_on_gpu; cache_engine.py:345, :357) and
writes the tail row, writes the full tail block back to the host, renames the
tail slot and resets _cache_lens for every request at the same step
(:349-359, :401-409). Here every request b has its own write position
p_b = pos[b] (the positional twin's counter BEFORE the step) and its own
activity bit:

  tail row       4032 + p_b % 64, written by index_put_ (an inactive request
                 rewrites its own row: torch.where, no change)
  full           active_b and (p_b + 1) % 64 == 0
  write-back     full requests only: host rows [p_b - 63, p_b] <- slot 63
  tail id        _block_map[:, b, 63] = (p_b + 1) // 64 for active b
  _cache_lens    active b: 4032 + (p_b + 1) % 64, which reproduces findings
                 F4 (the shipped engine); with f4_fix: 4032 + p_b % 64 + 1
  selection      the same diff kernel and gathers; an inactive request keeps
                 its map (new_map = old map) and loads nothing (load = -1)

The module works on the attributes of a CacheEngine (cache_engine.py:85):
_k_gpu, _v_gpu, _kv_bias_gpu (B, topk * block_size, ...), _block_map,
_new_block_map_buf, _load_mask (H, B, topk) int64, _cache_lens (B,) int32,
_k_cpu, _v_cpu (B, N_host, H, D) host, _tail_block_idx_on_gpu, block_size,
topk, head_num, head_dim, dtype, device. cache_engine.py itself is not
edited. The kernels are arguments: on the GPU the caller passes
cache_engine.diff.diff_offload, flash_h2d_from_mask and
flash_h2d_from_mask_bias; the CPU tests pass torch stand-ins.

ORDER IN ONE DECODE STEP (the shipped order, nosa_llama.py:584 then :618):
the twin writes the cis row (twin.step), then decode() runs with
kv_bias = twin.total_cis and pos = the counters from BEFORE twin.step.

PRECONDITIONS, not checked inside decode(): for every active request and
every KV head, topk_idx holds the tail id p_b // 64 (else diff frees slot 63
and the gather overwrites the live tail: findings F1; tail_in_selection()
checks it with one host sync). pos_host / active_host are CPU copies of
pos / active with the same values (the host loop of the write-back reads
them; reading the device values would sync).

WRITE-BACK BACKEND. DecodeBody's own backend is host-driven: one
copy_(non_blocking=True) per full request and tensor; it is not
CUDA-graph-capturable. DeviceWritebackBody (end of this module) takes a
device-driven kernel instead (r2_writeback.masked_writeback: Triton, stores
through the pinned host pointer, reads the mask and the positions on the
device). It is injected, so this module still imports only torch.

Imports only torch.
"""
from __future__ import annotations

import torch


class DecodeBody:
    """The per-request decode body. f4_fix is explicit (no default): False
    reproduces the shipped _cache_lens (findings F4), True gives attention the
    whole tail block on the step that fills it."""

    def __init__(self, f4_fix: bool):
        if not isinstance(f4_fix, bool):
            raise TypeError("f4_fix must be a bool, got %r" % (f4_fix,))
        self.f4_fix = f4_fix

    # ------------------------------------------- the pieces a negative control replaces
    def positions(self, pos: torch.Tensor) -> torch.Tensor:
        """Each request's own write position."""
        return pos

    def full_mask(self, eng, p: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Active requests whose write fills the tail block."""
        return active & ((p + 1) % eng.block_size == 0)

    def cache_lens_value(self, eng, q: torch.Tensor) -> torch.Tensor:
        """The attention length after the step that wrote position q (int64)."""
        bs = eng.block_size
        base = (eng.topk - 1) * bs
        if self.f4_fix:
            return base + q % bs + 1
        return base + (q + 1) % bs

    # ------------------------------------------------------------- allocation
    def allocate(self, eng, batch: int, n_host: int, pin: bool) -> None:
        """Allocate the per-request tables of one layer on eng (the shapes of
        cache_engine.py:195-230 with pool_blocks = 0 and n_host host rows)."""
        if getattr(eng, "pool_blocks", 0):
            raise RuntimeError("the R2 decode body has no victim pool; pool_blocks must be 0")
        if batch < 1 or n_host < eng.block_size:
            raise ValueError("batch must be >= 1 and n_host >= block_size")
        if torch.device(eng.device).type == "cuda" and not pin:
            raise ValueError("a CUDA engine needs pinned host rows: the Triton gathers read them directly")
        H, D, bs, topk = eng.head_num, eng.head_dim, eng.block_size, eng.topk
        dt, dev = eng.dtype, torch.device(eng.device)
        rows = topk * bs
        eng._k_cpu = torch.empty((batch, n_host, H, D), dtype=dt, device="cpu", pin_memory=pin)
        eng._v_cpu = torch.empty((batch, n_host, H, D), dtype=dt, device="cpu", pin_memory=pin)
        eng._k_gpu = torch.empty((batch, rows, H, D), dtype=dt, device=dev)
        eng._v_gpu = torch.empty((batch, rows, H, D), dtype=dt, device=dev)
        eng._kv_bias_gpu = torch.empty((batch, rows, H), dtype=dt, device=dev)
        eng._block_map = torch.full((H, batch, topk), -1, dtype=torch.int64, device=dev)
        eng._new_block_map_buf = torch.empty_like(eng._block_map)
        eng._load_mask = torch.empty_like(eng._block_map)
        eng._cache_lens = torch.full((batch,), (topk - 1) * bs, dtype=torch.int32, device=dev)
        eng._tail_block_idx_on_gpu = topk - 1

    # ---------------------------------------------------------------- prefill
    def prefill_request(self, eng, b: int, key_states: torch.Tensor, value_states: torch.Tensor,
                        kv_bias: torch.Tensor) -> None:
        """Place request b's prompt: key_states, value_states (S, H, D),
        kv_bias (S, H) = its cis rows 0..S-1. cache_engine.py:309-331 for one
        request with its own S (the tail rows into slot 63, the map, the
        attention length, the host rows)."""
        B = eng._k_gpu.shape[0]
        if not 0 <= b < B:
            raise IndexError("request %d outside batch %d" % (b, B))
        if key_states.dim() != 3 or tuple(key_states.shape[1:]) != (eng.head_num, eng.head_dim):
            raise ValueError("key_states must be (S, %d, %d), got %s" % (eng.head_num, eng.head_dim, tuple(key_states.shape)))
        S = int(key_states.shape[0])
        if tuple(value_states.shape) != tuple(key_states.shape) or tuple(kv_bias.shape) != (S, eng.head_num):
            raise ValueError("value_states must match key_states and kv_bias must be (S, H)")
        if S < 1 or S > eng._k_cpu.shape[1]:
            raise IndexError("prompt of %d tokens, %d host rows" % (S, eng._k_cpu.shape[1]))
        bs = eng.block_size
        t0 = eng._tail_block_idx_on_gpu * bs
        tail = S % bs                                                            # :309
        if tail > 0:                                                             # :311-319
            eng._k_gpu[b, t0:t0 + tail].copy_(key_states[S - tail:], non_blocking=True)
            eng._v_gpu[b, t0:t0 + tail].copy_(value_states[S - tail:], non_blocking=True)
            eng._kv_bias_gpu[b, t0:t0 + tail].copy_(kv_bias[S - tail:], non_blocking=True)
        eng._block_map[:, b, :] = -1                                             # a fresh request (:227)
        eng._block_map[:, b, eng._tail_block_idx_on_gpu] = S // bs               # :321 / :324
        eng._cache_lens[b] = (eng.topk - 1) * bs + tail                          # :327
        eng._k_cpu[b, :S].copy_(key_states, non_blocking=True)                   # :330
        eng._v_cpu[b, :S].copy_(value_states, non_blocking=True)                 # :331

    # ----------------------------------------------------------------- decode
    def decode(self, eng, key_states, value_states, kv_bias, topk_idx, pos, active, pos_host, active_host,
               diff_offload, gather_k, gather_v_bias):
        """One decode step. key_states, value_states (B, 1, H, D); kv_bias =
        the cis table (B, N, H) with row p_b already written; topk_idx (H, B,
        topk) int64; pos (B,) int64 and active (B,) bool on the device;
        pos_host, active_host the same values on the CPU. Returns what the
        shipped body returns (cache_engine.py:412). Raises before any write if
        an input is malformed or if ANY active request writes out of range
        (the whole batch is refused)."""
        B, S, H, D = key_states.shape
        if S != 1 or H != eng.head_num or D != eng.head_dim:
            raise ValueError("key_states must be (B, 1, %d, %d), got %s" % (eng.head_num, eng.head_dim, tuple(key_states.shape)))
        if tuple(value_states.shape) != (B, 1, H, D):
            raise ValueError("value_states must be %s, got %s" % ((B, 1, H, D), tuple(value_states.shape)))
        if kv_bias.dim() != 3 or kv_bias.shape[0] != B or kv_bias.shape[2] != H:
            raise ValueError("kv_bias must be (%d, N, %d), got %s" % (B, H, tuple(kv_bias.shape)))
        if tuple(topk_idx.shape) != (H, B, eng.topk) or topk_idx.dtype != torch.int64:
            raise ValueError("topk_idx must be (%d, %d, %d) int64" % (H, B, eng.topk))
        for name, t, dtype in (("pos", pos, torch.int64), ("active", active, torch.bool),
                               ("pos_host", pos_host, torch.int64), ("active_host", active_host, torch.bool)):
            if tuple(t.shape) != (B,) or t.dtype != dtype:
                raise ValueError("%s must be (%d,) %s, got %s %s" % (name, B, dtype, tuple(t.shape), t.dtype))
        if pos_host.device.type != "cpu" or active_host.device.type != "cpu":
            raise ValueError("pos_host and active_host must be CPU tensors")
        limit = min(int(eng._k_cpu.shape[1]), int(kv_bias.shape[1]))
        if bool((active_host & ((pos_host < 0) | (pos_host >= limit))).any()):
            raise IndexError("an active request writes outside [0, %d)" % limit)

        bs = eng.block_size
        p = self.positions(pos)
        p_host = self.positions(pos_host)

        # A. the tail row (cache_engine.py:349-354)
        self.write_tail(eng, key_states, value_states, kv_bias, p, active)

        # B. the selection (:363, :377)
        self.select(eng, topk_idx, active, diff_offload)

        # C. the gathers (:381-382)
        gather_k(eng._k_gpu, eng._k_cpu, eng._load_mask, bs)
        gather_v_bias(eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, kv_bias, eng._load_mask, bs)

        # D. write-back, tail id, attention length (:401-409)
        full = self.full_mask(eng, p, active)
        full_host = self.full_mask(eng, p_host, active_host)
        self.writeback(eng, full, full_host, p, p_host)
        tail_col = eng._tail_block_idx_on_gpu
        eng._block_map[..., tail_col] = torch.where(active.view(1, B), ((p + 1) // bs).view(1, B),
                                                    eng._block_map[..., tail_col])
        eng._cache_lens.copy_(torch.where(active, self.cache_lens_value(eng, p),
                                          eng._cache_lens.to(torch.int64)).to(torch.int32))
        return eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._cache_lens

    def write_tail(self, eng, key_states, value_states, kv_bias, p, active) -> None:
        """Row 4032 + p_b % 64 of K, V and bias <- the step's K, V and cis row
        p_b, active requests only (an inactive request rewrites its own row)."""
        B = p.shape[0]
        row = eng._tail_block_idx_on_gpu * eng.block_size + p % eng.block_size
        bi = torch.arange(B, device=p.device)
        a3 = active.view(B, 1, 1)
        eng._k_gpu[bi, row] = torch.where(a3, key_states[:, 0].to(eng._k_gpu.dtype), eng._k_gpu[bi, row])
        eng._v_gpu[bi, row] = torch.where(a3, value_states[:, 0].to(eng._v_gpu.dtype), eng._v_gpu[bi, row])
        prow = p.clamp(min=0, max=int(kv_bias.shape[1]) - 1)
        eng._kv_bias_gpu[bi, row] = torch.where(active.view(B, 1), kv_bias[bi, prow].to(eng._kv_bias_gpu.dtype),
                                                eng._kv_bias_gpu[bi, row])

    def select(self, eng, topk_idx, active, diff_offload) -> None:
        """diff_offload, then an inactive request keeps its map and loads
        nothing, then the map takes the new assignment."""
        B = active.shape[0]
        diff_offload(eng._block_map, topk_idx, eng._new_block_map_buf, eng._load_mask)
        am = active.view(1, B, 1)
        eng._new_block_map_buf.copy_(torch.where(am, eng._new_block_map_buf, eng._block_map))
        eng._load_mask.masked_fill_(~am, -1)
        eng._block_map.copy_(eng._new_block_map_buf, non_blocking=True)

    def writeback(self, eng, full, full_host, p, p_host) -> None:
        """The masked write-back, torch backend: for each full request (host
        mask), host rows [p_b - 63, p_b] <- slot 63 rows [0, 64)."""
        bs = eng.block_size
        t0 = eng._tail_block_idx_on_gpu * bs
        for b in full_host.nonzero().flatten().tolist():
            r0 = int(p_host[b]) - (bs - 1)
            eng._k_cpu[b, r0:r0 + bs].copy_(eng._k_gpu[b, t0:t0 + bs], non_blocking=True)
            eng._v_cpu[b, r0:r0 + bs].copy_(eng._v_gpu[b, t0:t0 + bs], non_blocking=True)

    # ----------------------------------------------------------------- helpers
    def resync(self, eng, pos: torch.Tensor, mask: torch.Tensor) -> None:
        """After a rollback (twin commit): for the masked requests, clear every
        window entry (slots 0 .. 62) that names a block >= pos // 64, then
        re-derive the tail id and the attention length from pos (the next write
        position).

        WHY THE CLEAR. When the drafts completed a block T and the next step
        loaded T into a window slot, a rollback into block T makes T the tail
        again: without the clear the map would name T twice, and diff_offload
        needs distinct ids (cache_engine.py:145-147). A cleared entry (-1) is a
        free slot. Slot 63's rows are NOT restored here: after a rollback that
        crosses a block boundary, rows [0, pos % 64) of slot 63 must be copied
        back from the host rows of block T (B's masked tail copy)."""
        B = pos.shape[0]
        tail_col = eng._tail_block_idx_on_gpu
        tail = (pos // eng.block_size).view(1, B, 1)
        window = eng._block_map[:, :, :tail_col]
        window.masked_fill_(mask.view(1, B, 1) & (window >= tail), -1)
        eng._block_map[..., tail_col] = torch.where(mask.view(1, B), (pos // eng.block_size).view(1, B),
                                                    eng._block_map[..., tail_col])
        eng._cache_lens.copy_(torch.where(mask, self.cache_lens_value(eng, pos - 1),
                                          eng._cache_lens.to(torch.int64)).to(torch.int32))

    def tail_in_selection(self, eng, topk_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """(B,) bool: the tail id pos // 64 is in the selection of every head."""
        tail = (self.positions(pos) // eng.block_size).view(1, -1, 1)
        return (topk_idx == tail).any(-1).all(0)


class DeviceWritebackBody(DecodeBody):
    """DecodeBody with a DEVICE-DRIVEN write-back: the kernel reads the device mask `full` and the device positions
    `p` (both computed on the device inside decode()); the host mirrors are not read by the write-back (decode() still
    validates them on the host before any write). writeback_kernel has the signature of
    r2_writeback.masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size); it is REQUIRED (no
    default), like f4_fix. Everything else is DecodeBody's."""

    def __init__(self, f4_fix: bool, writeback_kernel):
        super().__init__(f4_fix)
        if not callable(writeback_kernel):
            raise TypeError("writeback_kernel must be callable, got %r" % (writeback_kernel,))
        self.writeback_kernel = writeback_kernel

    def writeback(self, eng, full, full_host, p, p_host) -> None:
        self.writeback_kernel(eng._k_gpu, eng._v_gpu, eng._k_cpu, eng._v_cpu, full, p,
                              int(eng._tail_block_idx_on_gpu) * int(eng.block_size), int(eng.block_size))
