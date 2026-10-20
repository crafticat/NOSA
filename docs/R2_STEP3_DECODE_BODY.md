# R2 step 3: the engine decode body with per-request state

Branch `feature/nosi-r2-m2-engine` (from `feature/nosi-r2-m2` 760bdb3).
Status: SOURCE ONLY. Nothing here comes from a GPU run.

Code: `nosi/nosi/r2_decode_body.py` (imports only torch).
CPU gates: `tests/test_r2_decode_body.py`.
Inputs: `nosi/nosi/r2_positional_twin.py` (the per-request counters),
`docs/R2_ENGINE_FINDINGS.md` (F1-F4).

## Decisions

* PARITY TARGET (lead, 2026-10-20): the R2 body REPRODUCES the shipped F4
  behaviour, so the equal-advance gate is bitwise against the shipped decode.
  A switch (`f4_fix`, an explicit constructor argument, no default) gives the
  corrected attention length. Correcting F4 is an ordinary fix for all three
  arms together, later.
* `cache_engine.py` is NOT edited. The original arm runs the shipped body
  byte for byte, and the citations of `docs/R2_ENGINE_FINDINGS.md` stay
  valid. The R2 body is a module that works on the attributes of a
  `CacheEngine` (`_k_gpu`, `_v_gpu`, `_kv_bias_gpu`, `_block_map`,
  `_new_block_map_buf`, `_load_mask`, `_cache_lens`, `_k_cpu`, `_v_cpu`). The
  kernels come in as arguments (`diff.diff_offload`, `flash_h2d_from_mask`,
  `flash_h2d_from_mask_bias` on the GPU; torch stand-ins on the CPU). The R2
  model path (a later step) calls it.
* The body does not own counters. The caller passes `pos` (B,) = the twin's
  counter BEFORE the step (the position each request writes) and `active`
  (B,) bool. The twin advances itself (`twin.step`). In one decode step the
  order is the shipped one: the twin writes the cis row first
  (`nosa_llama.py:584`), then the body runs (`:618`) with
  `kv_bias = twin.total_cis`.

## FINAL interface (code: e048b3d, `nosi/nosi/r2_decode_body.py`)

Load the module by file path or as `from . import r2_decode_body` inside the
package; it imports only torch. Notation: `B` batch, `H` = `eng.head_num`,
`D` = `eng.head_dim`, `bs` = `eng.block_size` (64), `topk` = `eng.topk` (64),
`N` = rows of the cis table, `N_host` = host rows. "device" = `eng.device`.

`body = DecodeBody(f4_fix)`: `f4_fix` is a `bool`, REQUIRED, no default
(anything else raises `TypeError`). `False` = the shipped `_cache_lens` (F4
reproduced; the G0 setting). `True` = the corrected length.

`body.allocate(eng, batch, n_host, pin) -> None`. Needs `eng.head_num`,
`head_dim`, `block_size`, `topk`, `dtype`, `device`, and `pool_blocks`
absent or 0 (else `RuntimeError`). Sets on `eng`: `_k_cpu`, `_v_cpu`
(batch, n_host, H, D) `eng.dtype` on the CPU (pinned when `pin`); `_k_gpu`,
`_v_gpu` (batch, topk*bs, H, D) on the device; `_kv_bias_gpu` (batch,
topk*bs, H); `_block_map` (H, batch, topk) int64 = -1; `_new_block_map_buf`,
`_load_mask` (same shape, uninitialised); `_cache_lens` (batch,) int32 =
4032; `_tail_block_idx_on_gpu = topk - 1`. A caller may also allocate these
itself with the same shapes.

`body.prefill_request(eng, b, key_states, value_states, kv_bias) -> None`.
`b` a Python int; `key_states`, `value_states` (S, H, D); `kv_bias` (S, H) =
`twin.total_cis[b, :S]` after `twin.prefill(b, ...)`; all on the device of
the tables they go to (or castable by `copy_`). Mutates request `b` only:
slot 63 rows `[0, S % 64)` of `_k_gpu`, `_v_gpu`, `_kv_bias_gpu`;
`_block_map[:, b, :] = -1` and `[:, b, 63] = S // 64`;
`_cache_lens[b] = 4032 + S % 64`; host rows `[0, S)` of `_k_cpu`, `_v_cpu`.
Raises `IndexError` / `ValueError` before writing on a bad `b`, shape or
`S > N_host`.

`body.decode(eng, key_states, value_states, kv_bias, topk_idx, pos, active,
pos_host, active_host, diff_offload, gather_k, gather_v_bias)`
-> `(eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._cache_lens)`, the same
tuple as the shipped body (cache_engine.py:412), the full allocations.

| argument | shape / dtype / device | meaning |
|---|---|---|
| `key_states`, `value_states` | (B, 1, H, D), the table dtype, device | the step's K, V (`decode_update_kv` already unsqueezes, cache_engine.py:702) |
| `kv_bias` | (B, N, H), device | `twin.total_cis`; row `pos[b]` already written by `twin.step` |
| `topk_idx` | (H, B, topk) int64, contiguous, device | the selection; must hold `pos[b] // 64` for active b (C's per-request pooling with `M_b = twin.grid_width()` read before `twin.step`) |
| `pos` | (B,) int64, device | write positions = the twin counter BEFORE `twin.step` for this token |
| `active` | (B,) bool, device | which requests write this step |
| `pos_host`, `active_host` | (B,) int64 / bool, CPU | the same values on the host (the write-back loop reads them; no device sync) |
| `diff_offload`, `gather_k`, `gather_v_bias` | callables | on the GPU: `cache_engine.diff.diff_offload`, `cache_engine.flash_h2d_from_mask`, `cache_engine.flash_h2d_from_mask_bias` (called exactly as at cache_engine.py:363, :381, :382) |

Call order in one layer, one step:
`p = twin.pos.clone(); twin.step(k, cis, active); body.decode(eng, k1, v1, twin.total_cis, topk, p, active, p_host, active_host, ...)`.
(The twin is per layer for its compressed tables but `pos` is the same for
every layer; the model keeps one `pos`.)

Mutates: `_k_gpu`, `_v_gpu`, `_kv_bias_gpu` (the tail row of each active
request, the gathered slots), `_block_map`, `_new_block_map_buf`,
`_load_mask`, `_cache_lens` (active requests), `_k_cpu`, `_v_cpu` (host rows
`[p_b - 63, p_b]` of full active requests). An inactive request's persistent
tables do not change (its `_load_mask` row is -1, its `_new_block_map_buf`
row equals its map). Raises before any write: `ValueError` on a shape or
dtype, `IndexError` if an active request has `pos < 0` or
`pos >= min(N_host, N)`.

`body.resync(eng, pos, mask) -> None`: `pos` (B,) int64 device = the twin
counter AFTER a commit, `mask` (B,) bool device. Sets
`_block_map[:, b, 63] = pos // 64` and `_cache_lens[b]` for masked `b`.
Slot 63 rows are not restored (B's masked tail copy).

`body.tail_in_selection(eng, topk_idx, pos) -> (B,) bool` (host sync; tests
and debug only).

## Table (1): shipped line -> per-request form

Request `b`, write position `p_b = pos[b]`, block 64, tail slot 63, base
`4032 = 63 * 64`.

| Shipped (cache_engine.py) | Line | Per-request form |
|---|---|---|
| `seq_length += 1`; bias row `seq_length - 1` | :345, :354 | bias row `p_b` of `total_cis` (gather by `(b, p_b)`) |
| `_cache_lens += 1` | :346 | recomputed at the end (below) |
| `_tail_write_pos = 63*64 + _tail_block_len_on_gpu` | :349 | row `4032 + p_b % 64` |
| tail `copy_` of K, V, bias | :350-354 | `index_put_` at `(b, 4032 + p_b % 64)`; an inactive request rewrites its own row (`torch.where`), so nothing changes |
| `_tail_block_len_on_gpu += 1`; `tail_full` | :357-359 | `full_b = active_b and (p_b + 1) % 64 == 0` |
| `diff_offload`; `_block_map.copy_` | :363, :377 | the same kernel; for an inactive request `new_map = old_map` and `load_mask = -1` |
| the two gathers | :381-382 | the same kernels |
| write-back of slot 63 to host rows `seq_length - 64 ..` | :401-405 | for `full_b` only: host rows `[p_b - 63, p_b]` <- slot 63 rows `[0, 64)` |
| `_block_map[..., 63] += 1` | :407 | `_block_map[:, b, 63] = (p_b + 1) // 64` for active `b` |
| `_tail_block_len_on_gpu = 0` | :408 | none (derived from `pos`) |
| `_cache_lens = 4032` | :409 | active `b`: `4032 + (p_b + 1) % 64` (shipped, F4); with `f4_fix`: `4032 + p_b % 64 + 1` |
| prefill tail placement | :309-327 | per request `b` with its own `S_b`: tail rows `[S_b - S_b % 64, S_b)` into slot 63, `_block_map[:, b, :] = -1`, `[:, b, 63] = S_b // 64`, `_cache_lens[b] = 4032 + S_b % 64` |
| prefill host rows | :330-331 | host rows `[0, S_b)` of request `b` |

`resync(eng, pos)` re-derives `_block_map[..., 63]` and `_cache_lens` from the
counters after a rollback (twin `commit`). A rollback that crosses a block
boundary also needs slot 63 restored from the host copy. That is B's masked
tail-slot copy (`nosi-r2-perreq`, spec_loop.py, "PER-REQUEST ROUNDS").

## Preconditions (not checked inside the body)

* For every active request and every KV head, `topk_idx` contains the tail
  block id `p_b // 64`. If it does not, `diff_kernel` frees slot 63 and the
  gather overwrites the live tail (findings F1). C's per-request pooling
  (`max_pooling_per_request.py`, `nosi-r2-m2-kernels` 4cbd521) forces the last
  17 ids of `[0, M_b)` to `+inf`. With `M_b = twin.grid_width()` read BEFORE
  the step (`p_b // 64 + 1`), the tail id is always in that window.
  `tail_in_selection()` checks this (one host sync; tests and debug only).
* `p_b < total_cis.shape[1]` and `p_b < _k_cpu.shape[1]` for active requests
  (the twin refuses earlier: `TwinCapacityError`).
* Victim pool off, availability hook off, transfer trace off: the R2 body
  has none of these hooks. `allocate()` refuses `pool_blocks > 0`.

## The masked write-back (open GPU question, plan risk)

Implemented: the TORCH backend, host-driven. The host mirrors (`pos_host`,
`active_host`, CPU tensors) give the full set without a device sync. For
each full request it issues one `copy_(non_blocking=True)` per tensor from
slot 63 into the pinned host rows. On average that is `B / 64` copies per
step per layer for K and the same for V. It is correct on the shipped stream
order: the later H2D gathers run on the same stream. It is NOT
CUDA-graph-capturable (a host loop over a host mask).

Open: a TRITON backend, device-driven. One kernel, grid
`(B, H, 64 / R)`, reads `full[b]` and `p[b]` on the device, and writes slot
63 rows to the pinned host buffer through its device pointer (UVA), as
`flash_h2d_from_mask*` already read pinned host memory. It is capturable and
has no host loop. Questions that only a GPU run answers: the PCIe write
efficiency of 64 x H x D x 2-byte rows from a kernel compared with the DMA
engine, the ordering with the H2D gathers on the same stream, and the int64
offsets (`B * N_host * H * D` passes 2^31 at the target sizes). Gate:
bitwise equal host rows to the torch backend on equal and ragged runs, the
same NaN-poison test.

## Gates (CPU; `tests/test_r2_decode_body.py`)

Run inside a CPU job:

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2 python -m pytest -q tests/test_r2_decode_body.py

* G-EQ (equal advance): the R2 body against the SHIPPED
  `decode_update_has_kv_bias`, extracted from `cache_engine.py` by AST and
  executed with `diff.diff_offload` and the two gathers replaced by torch
  stand-ins (the same stand-ins for both). `torch.equal` on every table and
  every host row (`_k_gpu`, `_v_gpu`, `_kv_bias_gpu`, `_block_map`,
  `_new_block_map_buf`, `_load_mask`, `_cache_lens`, `_k_cpu`, `_v_cpu`)
  after prefill and after every step. `S % 64` in {0, 1, 63}; B = 1 and 4;
  150 steps, so every cell crosses at least two block completions.
* G-RAG (ragged): B = 4 with different `S_b` and a random activity mask
  every step, against 4 independent B = 1 engines. `torch.equal` per request
  on every persistent table and host row after every step. The engine agrees
  with the twin (`cache_lens`, tail block).
* G-POISON: host rows `[p - 64, p + 1]` of every request are NaN before one
  step. In the batch, one request is full and active, one is full and
  inactive, and two are not full. After the step: the full active request
  has clean host rows `[p - 63, p]` equal to its tokens, its neighbour rows
  `p - 64` and `p + 1` are still NaN, and every other request is still NaN.
* NEGATIVE CONTROLS: an unmasked write-back (`full = active`) and a
  write-back that ignores activity both fail G-POISON. An unmasked
  write-back fails G-EQ. A shared tail counter (one position for the batch)
  fails G-RAG and passes G-EQ.
* UNITS: an inactive request changes nothing, the F4 switch, the tail-id
  precondition helper, `resync` after a rollback inside a block, the slot
  cache stays coherent (each slot holds the host rows of the block its map
  names), `allocate` refusals, imports only torch.

## Interfaces with B and C

* B (per-request speculative rounds, `nosi-r2-perreq`): the same `pos`
  meaning (the twin counter, the next write position). `commit` resets the
  counters; the body's `resync` re-derives the tail id and `cache_lens`. B
  owns the restore of slot 63 when a rollback crosses a block boundary.
* C (kernels, `nosi-r2-m2-kernels` 4cbd521): the body consumes `topk_idx`
  from the per-request pooling (`M_b = twin.grid_width()` before the step,
  `n_comp` = `twin.seqused_k()` for stage 1). The body does not touch the
  compressed tables.

## Not covered

The kernels themselves (stand-ins on the CPU), the GPU stream order, the
Triton write-back, CUDA-graph capture of the body (the torch backend has a
host loop; `torch.where` allocates temporaries), the victim pool and the
availability hook, attention.
