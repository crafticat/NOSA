"""Move whole blocks between an attended slot and a victim-pool slot, in place,
inside NOSI's GPU cache tensors (retroinfer-eval fork).

The pool lives at slot indices ``topk .. topk+P-1`` of the SAME
``_k_gpu`` / ``_v_gpu`` / ``_kv_bias_gpu`` tensors, i.e. at row offsets
``(topk+q)*block_size``. Those rows sit BEYOND ``_cache_lens``
(``cache_engine.py:231``, and again at :285/:399/:451:
``(topk-1)*block_size + tail_len``, at most ``topk*block_size - 1`` because
``tail_len <= block_size``), and ``flash_attn_nosa_with_kvcache`` takes no block
table -- it reads rows ``0 .. cache_seqlens-1`` as one flat sequence
(``nosa_llama.py:610-616``, ``cache_seqlens=_cache_lens``). So the pool is
invisible to attention and the attended set is still exactly slots
``0..topk-1``. That is the hard constraint
this whole design is built around: THE CACHE IS THE ATTENDED SET.

Written in the style of ``flash_h2d_mask.py`` (grid ``(B, H, M)``, strides
taken from ``.stride()``, a scalar early return on an inactive entry) so the
two can be read side by side.

ACTIONS, decided by ``pool_update_kernel.cu`` and mirrored in
``scripts/nosi_pool_reference.py``:

===========  ==============================================================
NONE (0)     nothing to do (no load at this slot, the tail slot, a pool
             overflow, or a miss with no block worth parking)
SWAP (1)     attended slot s and pool slot q exchange contents. The wanted
             block was IN the pool, so no host fetch happens at all; the
             block leaving slot s lands in q, which is the eviction, free.
MOVE_IN (2)  pool slot q -> attended slot s, one way. Same as SWAP except
             slot s held nothing worth keeping, so q is left empty.
MOVE_OUT (3) attended slot s -> pool slot q, one way. The wanted block was
             NOT in the pool; the host gather will overwrite slot s a moment
             later, so copying the pool victim back into s would be pure
             waste (~65 kB per missed entry, ~0.2 GB per step at 64K x 64).
===========  ==============================================================

DISJOINTNESS (why there is no race across programs in one launch). Program
``(b, h, s)`` touches attended rows ``[s*bs, (s+1)*bs)`` and pool rows
``[(topk+q)*bs, ...)`` at batch ``b``, head ``h``. Distinct ``s`` give
distinct attended rows; distinct ``s`` give distinct ``q`` because the
bookkeeping kernel claims each pool slot with an atomicCAS and demotes the
loser to an ordinary miss, so one slot cannot be handed to two attended slots
even if its inputs are malformed; and the attended and pool regions never
overlap. Distinct ``(b, h)`` index disjoint memory. So every program writes
memory no other program touches.

INDEXING, both envelopes, because the pool TIGHTENS one of them.

* HOST GATHER (``flash_h2d_mask.py`` / ``flash_h2d_mask_bias.py``, UNCHANGED by
  the pool and NOT int64-cast): ``pid_b * stride_b_cpu`` is int32 x int32, so
  the CPU-side tensor bounds it at ``B * (S + max_gen_len) < 2**23``
  (max_gen_len = 8192; REPRODUCE.md: 64K x 112 passes, 64K x 120 aborts).
* GPU SIDE of the same gathers: ``pid_b * stride_b_gpu`` with
  ``stride_b_gpu = (topk + P) * block_size * H * D``. For NOSA-8B
  ``block_size * H * D = 64 * 2 * 128 = 16384`` (config.json: two KV heads,
  head_dim 128; the CacheEngine defaults agree), so the bound is
  ``B * (topk + P) < 2**31 / 16384 = 131072``. THE POOL DIVIDES THAT HEADROOM
  BY ``(topk + P) / topk`` -- 3.0x at P=129 -- and the constant is
  MODEL-SPECIFIC: an 8-KV-head model has ``block_size * H * D = 65536`` and a
  4x tighter bound.
* This kernel is immune to both by construction: ``pid_b``/``pid_h``/``pid_m``
  and the row/column offsets are ``.to(tl.int64)`` before any multiply.

At the cells this job actually runs: 64K x 64 at P=129 gives 4,718,592 of
8,388,608 (1.8x margin) on the host side and 12,352 of 131,072 (10.6x) on the
GPU side; 16K x 128 at P=17 gives 3,145,728 (2.7x) and 10,368 (12.6x). The pool
becomes the binding constraint only when ``S + 8192 < block_size * (topk + P)``,
i.e. below about 4K of context at P=129. ``cache_engine.prefill_update`` asserts
both bounds at allocation time so an out-of-envelope cell aborts loudly instead
of indexing a wrapped address.

COMPACT GEOMETRY (``NOSI_POOL_SWAP_COMPACT=k``, k >= 1; default 0 = the kernel
above, byte for byte). Job 2175353's profile (16K x 128, pool on;
docs/evidence/stage2_profile_2175353) put ``pool_swap_kernel`` at 102 us per
launch = 3.3 ms per decode step, 3.5x the device-to-device byte time of the
~850 rows it moves per launch (~53 MB at ~1.9 TB/s = ~28 us). The grid
``(B, H, M)`` launches 2 x 128 x 64 = 16384 programs per launch at batch 128
and ~3.3 of every 64 do any work: the other ~95% each pay their scheduling and
one dependent global load before the scalar early return, and that floor is
what the profile measured. HiSparse's resolver (one CTA per request looping
over its slots) has no such floor: 22 us per launch of 256 rows
(docs/evidence/hisparse_resolver_2175357/table.md).

``pool_swap_compact_kernel`` below keeps the per-slot body of
``pool_swap_kernel`` VERBATIM (the same pointer arithmetic, the same masked
loads before the same masked stores; tests/test_nosi_pool_swap_compact.py in
retroinfer-eval pins the lines) and changes only the launch geometry: grid
``(B, H, k)``, program ``(b, h, r)`` loads its whole (h, b) row of the plan
with ONE vector load each for action and target and loops over slots
``r, r+k, r+2k, ...``, folding the three scalar early returns into one
predicate (a loop body cannot return). Distinct programs still write disjoint
memory (the DISJOINTNESS argument above, applied to the strided slot
partition) and within a program the slots are disjoint too, so their order is
irrelevant and the effect is bit-identical to the shipped kernel:
``torch.equal`` on K, V and bias after the call. ``pool_swap_twin.py`` is the
torch twin of both geometries; ``scripts/nosi_pool_parity.py`` runs the compact
kernel as a second GPU arm against the same CPU oracle.

Why not a compacted slot list from ``pool_update_kernel.cu``: it could emit a
per-(h, b) count plus index list cheaply (one more O(M) shared-memory rank like
its phase 2, two more output buffers), but that changes the bookkeeping kernel
that also runs in the default arm and the contract the CPU oracle and the
parity harness model, and it buys nothing: the action row is 64 bytes, one
vector load, and selecting element m from it is a register op, whereas a
compacted list costs one more dependent global load per active slot.

REGISTERED PREDICTION (written before any GPU run; the paired arm is the
falsifier): at 16K x 128 with ~27k pool moves per step (~850 per launch,
~3.3 per (b, h) stream, longest stream ~9), k = 1 lands at 30-50 us per
launch -- the ~28 us byte floor plus the longest program's serial chain of
three dependent round-trips per slot (K, V, bias) -- i.e. 1.0-1.6 ms per step
against 3.3. Falsified if the compact arm measures >= 70 us per launch (then
the 102 us was not the early-exit floor) or <= 25 us (then fewer bytes move
than the ledger's 1.7 GB per step counts). k = 4 (1024 programs, longest
chain /4) is predicted to sit on the byte floor, 28-35 us; it costs nothing
but a knob value.
"""
import torch
import triton
import triton.language as tl

# The action codes, shared with pool_update_kernel.cu (which #defines the same
# four) and with transfer_trace.py. Triton refuses to close over a plain module
# global, so each carries a `tl.constexpr` annotation: that makes it readable
# from inside @triton.jit AND leaves it a perfectly ordinary int for Python
# callers, the CPU reference and the parity harness. Job 2173294 died at gate 0
# in three minutes without the annotations ("Cannot access global variable
# POOL_NONE from within @jit'ed function"), which is what that gate is for.
POOL_NONE: tl.constexpr = 0
POOL_SWAP: tl.constexpr = 1
POOL_MOVE_IN: tl.constexpr = 2
POOL_MOVE_OUT: tl.constexpr = 3


@triton.jit
def pool_swap_kernel(
    k_ptr,           # (B, S_GPU, H, D)
    v_ptr,           # (B, S_GPU, H, D)
    bias_ptr,        # (B, S_GPU, H)
    target_ptr,      # (H, B, M) int64: the pool slot q
    action_ptr,      # (H, B, M) int8:  NONE / SWAP / MOVE_IN / MOVE_OUT
    stride_b_k, stride_m_k, stride_h_k, stride_d_k,
    stride_b_v, stride_m_v, stride_h_v, stride_d_v,
    stride_b_bias, stride_m_bias, stride_h_bias,
    S_GPU, B, H, M, TOPK, P,
    block_size: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(2).to(tl.int64)

    idx = pid_h * B * M + pid_b * M + pid_m
    act = tl.load(action_ptr + idx)

    # the overwhelmingly common case: ~3.6 of 64 slots move per (h, b) step
    if act == POOL_NONE:
        return

    q = tl.load(target_ptr + idx).to(tl.int64)

    # Belt and braces, BOTH ENDS of the range. The bookkeeping kernel only ever
    # pairs a non-NONE action with 0 <= q < P, but the row masks below are
    # upper-bound only (the same shape as flash_h2d_mask.py:56-57), so a
    # negative q would slip past them and read out of bounds, and a q >= P
    # would make `mask_pool` all-false -- which does NOT skip the store: on a
    # SWAP `mask_att` is still true, so `tl.load(..., other=0)` would write 64
    # tokens of ZEROED K, V and bias into an ATTENDED slot. Finite, wrong, and
    # with no NaN or OOB to trip on. Return the way the gather returns on an
    # invalid block id (flash_h2d_mask.py:32-33). Two statements rather than
    # one `|`: a scalar early return is the form this file already uses.
    if q < 0:
        return
    if q >= P:
        return

    read_att = (act == POOL_SWAP) | (act == POOL_MOVE_OUT)   # attended -> pool
    read_pool = (act == POOL_SWAP) | (act == POOL_MOVE_IN)   # pool -> attended

    offset_s = tl.arange(0, block_size)[:, None].to(tl.int64)   # (bs,1)
    offset_d = tl.arange(0, D)[None, :].to(tl.int64)            # (1,D)

    att_row = pid_m * block_size + offset_s
    pool_row = (TOPK + q) * block_size + offset_s

    mask_att = att_row < S_GPU
    mask_pool = pool_row < S_GPU

    # ---- K ------------------------------------------------------------------
    k_att = (k_ptr + pid_b * stride_b_k + att_row * stride_m_k
             + pid_h * stride_h_k + offset_d * stride_d_k)
    k_pool = (k_ptr + pid_b * stride_b_k + pool_row * stride_m_k
              + pid_h * stride_h_k + offset_d * stride_d_k)
    # Both sides are loaded BEFORE either is stored, so a SWAP is a real
    # exchange and not a double-write. The pool store follows the pool load of
    # the same address in program order, which the compiler must preserve
    # because the two provably alias.
    ka = tl.load(k_att, mask=mask_att & read_att, other=0)
    kp = tl.load(k_pool, mask=mask_pool & read_pool, other=0)
    tl.store(k_pool, ka, mask=mask_pool & read_att)
    tl.store(k_att, kp, mask=mask_att & read_pool)

    # ---- V ------------------------------------------------------------------
    v_att = (v_ptr + pid_b * stride_b_v + att_row * stride_m_v
             + pid_h * stride_h_v + offset_d * stride_d_v)
    v_pool = (v_ptr + pid_b * stride_b_v + pool_row * stride_m_v
              + pid_h * stride_h_v + offset_d * stride_d_v)
    va = tl.load(v_att, mask=mask_att & read_att, other=0)
    vp = tl.load(v_pool, mask=mask_pool & read_pool, other=0)
    tl.store(v_pool, va, mask=mask_pool & read_att)
    tl.store(v_att, vp, mask=mask_att & read_pool)

    # ---- kv_bias: (B, S_GPU, H), NO head_dim axis, so the row column alone
    # indexes it -- exactly the shape flash_h2d_mask_bias.py:45-56 uses.
    b_att = (bias_ptr + pid_b * stride_b_bias + att_row * stride_m_bias
             + pid_h * stride_h_bias)
    b_pool = (bias_ptr + pid_b * stride_b_bias + pool_row * stride_m_bias
              + pid_h * stride_h_bias)
    ba = tl.load(b_att, mask=mask_att & read_att, other=0)
    bp = tl.load(b_pool, mask=mask_pool & read_pool, other=0)
    tl.store(b_pool, ba, mask=mask_pool & read_att)
    tl.store(b_att, bp, mask=mask_att & read_pool)


def flash_pool_swap(
    k_gpu,          # (B, S_GPU, H, D)
    v_gpu,          # (B, S_GPU, H, D)
    kv_bias_buf,    # (B, S_GPU, H)
    pool_target,    # (H, B, M) int64
    pool_action,    # (H, B, M) int8
    topk: int,
    block_size: int = 64,
):
    """Apply one decode step's pool plan to the cache tensors, in place."""
    B, S_GPU, H, D = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert H == H2 and B == B2, (k_gpu.shape, pool_target.shape)
    assert M == topk, f"pool_target must be [H,B,topk]; got M={M}, topk={topk}"
    assert pool_action.shape == pool_target.shape
    # P is DERIVED from the tensor the kernel will actually index, not passed
    # in, so a mismatch between the pool state's width and the cache tensor's
    # width cannot open the q >= P hole the kernel guards.
    P = S_GPU // block_size - topk
    assert P >= 1, f"no pool rows in the cache tensor: S_GPU={S_GPU}, topk={topk}"

    stride_b_k, stride_m_k, stride_h_k, stride_d_k = k_gpu.stride()
    stride_b_v, stride_m_v, stride_h_v, stride_d_v = v_gpu.stride()
    stride_b_bias, stride_m_bias, stride_h_bias = kv_bias_buf.stride()

    grid = (B, H, M)

    pool_swap_kernel[grid](
        k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action,
        stride_b_k, stride_m_k, stride_h_k, stride_d_k,
        stride_b_v, stride_m_v, stride_h_v, stride_d_v,
        stride_b_bias, stride_m_bias, stride_h_bias,
        S_GPU, B, H, M, topk, P,
        block_size, D,
        num_warps=8,
    )


@triton.jit
def pool_swap_compact_kernel(
    k_ptr,           # (B, S_GPU, H, D)
    v_ptr,           # (B, S_GPU, H, D)
    bias_ptr,        # (B, S_GPU, H)
    target_ptr,      # (H, B, M) int64: the pool slot q
    action_ptr,      # (H, B, M) int8:  NONE / SWAP / MOVE_IN / MOVE_OUT
    stride_b_k, stride_m_k, stride_h_k, stride_d_k,
    stride_b_v, stride_m_v, stride_h_v, stride_d_v,
    stride_b_bias, stride_m_bias, stride_h_bias,
    S_GPU, B, H, M, TOPK, P,
    block_size: tl.constexpr,
    D: tl.constexpr,
    M_PAD: tl.constexpr,     # next power of two >= M (tl.arange needs one)
    SPLIT: tl.constexpr,     # programs per (b, h); program r visits slots r, r+SPLIT, ...
):
    # pool_swap_kernel's per-slot body under a (B, H, SPLIT) grid. See the
    # module docstring, COMPACT GEOMETRY.
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_r = tl.program_id(2)          # int32 on purpose: it is a loop bound below

    # This program's whole (h, b) row of the plan, one vector load each (64 B
    # and 512 B at M = 64), so the slot loop never waits on memory to learn
    # whether a slot is active. Neither array is written by this kernel.
    row = pid_h * B * M + pid_b * M
    offs_m = tl.arange(0, M_PAD)
    in_row = offs_m < M
    acts = tl.load(action_ptr + row + offs_m, mask=in_row, other=POOL_NONE).to(tl.int32)
    tgts = tl.load(target_ptr + row + offs_m, mask=in_row, other=-1).to(tl.int64)

    offset_s = tl.arange(0, block_size)[:, None].to(tl.int64)   # (bs,1)
    offset_d = tl.arange(0, D)[None, :].to(tl.int64)            # (1,D)

    for m in range(pid_r, M, SPLIT):
        # element m of the prefetched rows. Triton has no dynamic indexing into
        # a register tensor; a one-hot select-and-sum is a few register ops.
        act = tl.sum(tl.where(offs_m == m, acts, 0))
        q = tl.sum(tl.where(offs_m == m, tgts, 0))
        # pool_swap_kernel's three scalar early exits (NONE, q < 0, q >= P)
        # folded into one predicate, because a loop body has no early exit.
        # Same set of (b, h, m) moves, same bytes.
        if (act != POOL_NONE) & (q >= 0) & (q < P):
            pid_m = m.to(tl.int64)
            # ---- from here to the end of the loop body: pool_swap_kernel, verbatim
            read_att = (act == POOL_SWAP) | (act == POOL_MOVE_OUT)   # attended -> pool
            read_pool = (act == POOL_SWAP) | (act == POOL_MOVE_IN)   # pool -> attended

            att_row = pid_m * block_size + offset_s
            pool_row = (TOPK + q) * block_size + offset_s

            mask_att = att_row < S_GPU
            mask_pool = pool_row < S_GPU

            # ---- K --------------------------------------------------------
            k_att = (k_ptr + pid_b * stride_b_k + att_row * stride_m_k
                     + pid_h * stride_h_k + offset_d * stride_d_k)
            k_pool = (k_ptr + pid_b * stride_b_k + pool_row * stride_m_k
                      + pid_h * stride_h_k + offset_d * stride_d_k)
            ka = tl.load(k_att, mask=mask_att & read_att, other=0)
            kp = tl.load(k_pool, mask=mask_pool & read_pool, other=0)
            tl.store(k_pool, ka, mask=mask_pool & read_att)
            tl.store(k_att, kp, mask=mask_att & read_pool)

            # ---- V --------------------------------------------------------
            v_att = (v_ptr + pid_b * stride_b_v + att_row * stride_m_v
                     + pid_h * stride_h_v + offset_d * stride_d_v)
            v_pool = (v_ptr + pid_b * stride_b_v + pool_row * stride_m_v
                      + pid_h * stride_h_v + offset_d * stride_d_v)
            va = tl.load(v_att, mask=mask_att & read_att, other=0)
            vp = tl.load(v_pool, mask=mask_pool & read_pool, other=0)
            tl.store(v_pool, va, mask=mask_pool & read_att)
            tl.store(v_att, vp, mask=mask_att & read_pool)

            # ---- kv_bias: (B, S_GPU, H) ------------------------------------
            b_att = (bias_ptr + pid_b * stride_b_bias + att_row * stride_m_bias
                     + pid_h * stride_h_bias)
            b_pool = (bias_ptr + pid_b * stride_b_bias + pool_row * stride_m_bias
                      + pid_h * stride_h_bias)
            ba = tl.load(b_att, mask=mask_att & read_att, other=0)
            bp = tl.load(b_pool, mask=mask_pool & read_pool, other=0)
            tl.store(b_pool, ba, mask=mask_pool & read_att)
            tl.store(b_att, bp, mask=mask_att & read_pool)


def flash_pool_swap_compact(
    k_gpu,          # (B, S_GPU, H, D)
    v_gpu,          # (B, S_GPU, H, D)
    kv_bias_buf,    # (B, S_GPU, H)
    pool_target,    # (H, B, M) int64
    pool_action,    # (H, B, M) int8
    topk: int,
    block_size: int = 64,
    split: int = 1,
):
    """flash_pool_swap under the compact grid (B, H, split): the same moves
    from 2*B*split programs instead of 2*B*M. ``split`` is the value of
    NOSI_POOL_SWAP_COMPACT; 1 is one program per (b, h) looping over all M
    slots, k > 1 gives each (b, h) k programs over interleaved slots."""
    B, S_GPU, H, D = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert H == H2 and B == B2, (k_gpu.shape, pool_target.shape)
    assert M == topk, f"pool_target must be [H,B,topk]; got M={M}, topk={topk}"
    assert pool_action.shape == pool_target.shape
    # the kernel indexes the plan as h*B*M + b*M + m with no strides
    assert pool_target.is_contiguous() and pool_action.is_contiguous()
    assert isinstance(split, int) and split >= 1, split
    # P is DERIVED from the tensor the kernel will actually index, as above.
    P = S_GPU // block_size - topk
    assert P >= 1, f"no pool rows in the cache tensor: S_GPU={S_GPU}, topk={topk}"

    stride_b_k, stride_m_k, stride_h_k, stride_d_k = k_gpu.stride()
    stride_b_v, stride_m_v, stride_h_v, stride_d_v = v_gpu.stride()
    stride_b_bias, stride_m_bias, stride_h_bias = kv_bias_buf.stride()

    grid = (B, H, split)

    pool_swap_compact_kernel[grid](
        k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action,
        stride_b_k, stride_m_k, stride_h_k, stride_d_k,
        stride_b_v, stride_m_v, stride_h_v, stride_d_v,
        stride_b_bias, stride_m_bias, stride_h_bias,
        S_GPU, B, H, M, topk, P,
        block_size, D, triton.next_power_of_2(M), split,
        num_warps=8,
    )
