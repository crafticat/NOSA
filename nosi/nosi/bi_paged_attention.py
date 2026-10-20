"""R2-M2 step 7, piece 3: the BATCH-INVARIANT PAGED decode attention (BI attention) of the spec round.

Design note: docs/R2_STEP7_SPEC_ROUND.md (f1f9f56) sections 3.4 (the BI paged attention), 5.1-5.2 (slot store, tail
pages). CPU tests: benchmarks/Efficiency/tests_cpu/test_bi_paged_attention.py. GPU gate: benchmarks/Efficiency/
biattn_gate.py (holder step benchmarks/Efficiency/biattn_step.sh).

WHAT IT COMPUTES. Every ROW r (a verify row or a draft row of the flattened tick; several rows may belong to one
request) and every KV head h read their own keys through their OWN PAGE LIST: page_table[r, h, 0 : num_pages[r, h]]
names slots of the row's storage entry seq_idx[r]; slot s is rows s*64 .. s*64+63 of k_cache / v_cache / bias_cache
[seq_idx[r], :, h]. Every listed page is read in full except the LAST, which is cut at last_len[r] rows (the row's own
extent: its position % 64 + 1 in its tail page). For each query head j of the KV group h (j // GROUP == h):

    s_t   = q[r, j] . k_t + bias_t                    (bias = the token's CIS, added to the raw dot, as FA2-NOSA does:
                                                       flash_fwd_kernel.h:950-985, then softmax_rescale_o with the
                                                       scale: the logit is (q.k + bias) * softmax_scale)
    out   = sum_t softmax_t(s * softmax_scale) v_t    over the listed rows only

THE LAW (AGENTS.md law 1; note 3.1, 3.4): the float reduction order of a row is a function of its OWN page list,
num_pages, last_len and data only, never of the batch, the row count, the row order or a neighbour:
  * one program per (row, KV head) (grid (R, Hkv)); a program reads only its own row of q, its own page list and the
    rows of the pages it lists. No program reads another row's pages (block-diagonality is structural, as in T10).
  * NO SPLIT-KV: one program walks all of a row's pages; there is no partition whose seams depend on the longest row
    or on the number of programs, and no combine kernel.
  * ONE TILE SIZE for every row (draft and verify alike): a tile is ONE 64-row page (PAGES_PER_TILE = 1), so the
    online-softmax update sequence is exactly the canonical page sequence. NUM_WARPS / NUM_STAGES are module constants;
    there is no autotune and no shape-keyed choice (an autotuner keyed on the batch would make the order a function
    of the batch).
  * the loop runs ceil(P / PAGES_PER_TILE) tiles, P = page_table.shape[2] (the caller's table width). A tile past
    the row's own pages is an EXACT identity (every score -inf: m unchanged, alpha = 2**0 = 1, p = 0, l and acc
    unchanged; a -0.0 accumulator can become +0.0, which torch.equal treats as equal), so P never reaches a row's
    bits. The integer arguments are not specialized (do_not_specialize), so one compiled kernel serves every call.
  * CANONICAL PAGE ORDER: the caller lists a row's pages in ASCENDING BLOCK-ID order (build_page_table). The tail
    page(s) carry the largest ids, so they come last and the last page is the live tail. Residency is content: the
    slot that holds a block never reaches the arithmetic, only the block order does.
  * masked rows (beyond last_len in the last page, beyond num_pages) are never LOADED: masked loads with other = 0,
    so NaN / Inf in any unlisted page or unused row cannot reach the output (no 0 * NaN in P.V).

NUMERICS (the GPU path). q / k / v / bias in bf16 (the engine's dtype); the two dots on the tensor cores with fp32
accumulation (tl.dot, bf16 operands); online softmax in fp32 in base 2 (exp2, scale * log2(e), the FA2 form); P is
cast to v's dtype before the P.V dot (as FA2 casts P to bf16); the output acc / l in fp32, stored in q's dtype. fp32
operands are accepted ONLY for the CPU interpreter (on the GPU an fp32 tl.dot would run as TF32).
bi_paged_decode_reference is the SAME tiled reduction in torch (per row and head, page by page, the same update
formulas): the CPU twin. It is not bitwise equal to the GPU kernel (a CPU matmul is not a tensor-core dot); it is
the executable definition of the order, and its invariance under batching is provable on CPU. Its two NEGATIVE
CONTROL knobs (split_by_batch, tile_by_batch) make the order depend on the batch, as a split-KV kernel and a
batch-keyed tile would; the tests must catch both.

PRIOR ART (cited; no code copied, so no third-party header): the in-house T10 block-diagonal kernel (retroinfer-eval
speckv/kernels/blockdiag_attend.py: one program per (group, row), fixed tile, no autotune, a torch twin with the same
tiled online softmax, the G-Kinv gate); vLLM's unified attention under batch invariance (one tile size for decode and
prefill rows, no split-KV: the design rule, PA-3 of the step-7 note). FA2-NOSA's paged mode is NOT usable: it needs
page_block_size % 256 == 0 and indexes the bias by the logical column, not through the block table (note section 5).

INTERFACE (fixed; the spec round's tick builds these per layer):
    bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                    softmax_scale=None, out=None) -> (R, Hq, D)
      q           (R, Hq, D)        bf16 (fp16; fp32 interpreter only), stride(-1) == 1
      k_cache     (N, S*64, Hkv, D) q's dtype, stride(-1) == 1          N storage entries (a request's slot store,
      v_cache     (N, S*64, Hkv, D)                                      its tail pages included as slots)
      bias_cache  (N, S*64, Hkv)    q's dtype
      seq_idx     (R,)        int32  the storage entry of row r
      page_table  (R, Hkv, P) int32  slot ids, canonical order (build_page_table); entries >= num_pages never read
      num_pages   (R, Hkv)    int32  >= 1 (the tail page is always listed)
      last_len    (R,)        int32  1 .. 64 (last_len_of(position))
      softmax_scale  float, default D ** -0.5
    No host sync (no device value is read on the host); validate_inputs reads values and is for tests / gates only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:                                                   # triton is needed only to launch the kernel; the reference,
    import triton                                      # the page-table builder and the tests import this module
    import triton.language as tl                       # without it
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None

PAGE = 64                    # rows per page = NOSA's 64-token block
PAGES_PER_TILE = 1           # ONE tile size for every row: one page per online-softmax step (the law; not a knob)
NUM_WARPS = 4                # fixed launch constants (no autotune)
NUM_STAGES = 2
LOG2E = 1.4426950408889634


def block_h_for(group: int) -> int:
    """The query-head tile: the next power of two >= the GQA group, at least 16 (tl.dot needs M >= 16). A function
    of the model (Hq / Hkv), never of the batch."""
    b = 16
    while b < group:
        b *= 2
    return b


def last_len_of(position: torch.Tensor) -> torch.Tensor:
    """A row at absolute position p attends rows 0 .. p % 64 of its tail page: last_len = p % 64 + 1 (int32)."""
    return (position.to(torch.int64) % PAGE + 1).to(torch.int32)


def build_page_table(block_ids: torch.Tensor, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """CANONICAL page lists. block_ids (R, Hkv, P) int: the blocks a row attends per KV head (historical blocks and
    the tail block(s)), -1 padded, in ANY order; slots (R, Hkv, P) int: the slot holding each. Returns
    (page_table (R, Hkv, P) int32: the slots in ascending block-id order, -1 after the valid ones;
     num_pages (R, Hkv) int32). Device ops only (a stable sort), no host sync. The order depends on the block ids
    alone, so two residency layouts of the same content give the same page sequence."""
    if block_ids.shape != slots.shape or block_ids.dim() != 3:
        raise ValueError("block_ids %s and slots %s must be the same (R, Hkv, P)" % (tuple(block_ids.shape), tuple(slots.shape)))
    ids = block_ids.to(torch.int64)
    valid = ids >= 0
    big = torch.iinfo(torch.int64).max
    key = torch.where(valid, ids, torch.full_like(ids, big))
    order = torch.sort(key, dim=-1, stable=True).indices
    sl = torch.gather(slots.to(torch.int64), -1, order)
    ok = torch.gather(valid, -1, order)
    table = torch.where(ok, sl, torch.full_like(sl, -1)).to(torch.int32).contiguous()
    return table, valid.sum(-1).to(torch.int32).contiguous()


def validate_inputs(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len) -> None:
    """Value checks (HOST READS: tests and gates only, never the tick): shapes, num_pages in [1, P], last_len in
    [1, 64], seq_idx in [0, N), every listed slot in [0, S), no slot listed twice in a row's head list."""
    R, Hq, D = q.shape
    N, SR, Hkv, _ = k_cache.shape
    S = SR // PAGE
    P = page_table.shape[2]
    np_ = num_pages.to(torch.int64).cpu()
    ll = last_len.to(torch.int64).cpu()
    si = seq_idx.to(torch.int64).cpu()
    pt = page_table.to(torch.int64).cpu()
    if bool(((np_ < 1) | (np_ > P)).any()):
        raise ValueError("num_pages outside [1, P=%d]" % P)
    if bool(((ll < 1) | (ll > PAGE)).any()):
        raise ValueError("last_len outside [1, %d]" % PAGE)
    if bool(((si < 0) | (si >= N)).any()):
        raise ValueError("seq_idx outside [0, N=%d)" % N)
    for r in range(R):
        for h in range(Hkv):
            lst = pt[r, h, :int(np_[r, h])].tolist()
            if any(s < 0 or s >= S for s in lst):
                raise ValueError("row %d head %d lists a slot outside [0, %d): %s" % (r, h, S, lst))
            if len(set(lst)) != len(lst):
                raise ValueError("row %d head %d lists a slot twice: %s" % (r, h, lst))


def _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len):
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.shape != k_cache.shape or bias_cache.dim() != 3:
        raise ValueError("q (R, Hq, D), k/v_cache (N, S*64, Hkv, D), bias_cache (N, S*64, Hkv); got %s %s %s %s" % (
            tuple(q.shape), tuple(k_cache.shape), tuple(v_cache.shape), tuple(bias_cache.shape)))
    R, Hq, D = q.shape
    N, SR, Hkv, D2 = k_cache.shape
    if D2 != D or SR % PAGE != 0 or tuple(bias_cache.shape) != (N, SR, Hkv) or Hq % Hkv != 0:
        raise ValueError("inconsistent shapes: q %s k %s bias %s (rows a multiple of %d, Hq a multiple of Hkv)" % (
            tuple(q.shape), tuple(k_cache.shape), tuple(bias_cache.shape), PAGE))
    if not (q.dtype == k_cache.dtype == v_cache.dtype == bias_cache.dtype):
        raise TypeError("q / k / v / bias dtypes differ: %s %s %s %s" % (q.dtype, k_cache.dtype, v_cache.dtype, bias_cache.dtype))
    if D < 16 or D & (D - 1):
        raise ValueError("head dim %d: a power of two >= 16" % D)
    for t, name in ((q, "q"), (k_cache, "k_cache"), (v_cache, "v_cache")):
        if t.stride(-1) != 1:
            raise ValueError("%s needs a unit last stride" % name)
    if seq_idx.shape != (R,) or last_len.shape != (R,) or num_pages.shape != (R, Hkv) or page_table.dim() != 3 \
            or page_table.shape[:2] != (R, Hkv):
        raise ValueError("seq_idx (R,), last_len (R,), num_pages (R, Hkv), page_table (R, Hkv, P); got %s %s %s %s" % (
            tuple(seq_idx.shape), tuple(last_len.shape), tuple(num_pages.shape), tuple(page_table.shape)))
    for t, name in ((seq_idx, "seq_idx"), (page_table, "page_table"), (num_pages, "num_pages"), (last_len, "last_len")):
        if t.dtype != torch.int32 or not t.is_contiguous():
            raise ValueError("%s must be contiguous int32 (got %s, contiguous=%s)" % (name, t.dtype, t.is_contiguous()))
    devs = {t.device for t in (q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)}
    if len(devs) != 1:
        raise ValueError("all tensors on one device, got %s" % devs)
    return R, Hq, D, N, Hkv


if triton is not None:
    @triton.jit(do_not_specialize=["s_qr", "s_qh", "s_qd", "s_kn", "s_ks", "s_kh", "s_kd", "s_vn", "s_vs", "s_vh",
                                   "s_vd", "s_bn", "s_bs", "s_bh", "s_or", "s_oh", "s_od", "s_ptr", "s_pth", "s_ptp",
                                   "s_npr", "s_nph", "n_tiles"])
    def _bi_paged_decode_kernel(
            Q, K, V, BIAS, OUT, SEQ, PT, NP, LAST,
            s_qr, s_qh, s_qd,
            s_kn, s_ks, s_kh, s_kd,
            s_vn, s_vs, s_vh, s_vd,
            s_bn, s_bs, s_bh,
            s_or, s_oh, s_od,
            s_ptr, s_pth, s_ptp,
            s_npr, s_nph,
            n_tiles, scale_log2,
            GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, PAGE_C: tl.constexpr,
            PPT: tl.constexpr, TILE: tl.constexpr):
        r = tl.program_id(0)
        h = tl.program_id(1)
        n = tl.load(SEQ + r).to(tl.int64)
        npg = tl.load(NP + r * s_npr + h * s_nph)
        last = tl.load(LAST + r)
        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D)
        offs_t = tl.arange(0, TILE)
        h_ok = offs_h < GROUP
        qh = h * GROUP + offs_h
        q = tl.load(Q + r * s_qr + qh[:, None] * s_qh + offs_d[None, :] * s_qd, mask=h_ok[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        k_base = K + n * s_kn + h * s_kh
        v_base = V + n * s_vn + h * s_vh
        b_base = BIAS + n * s_bn + h * s_bh
        pt_base = PT + r * s_ptr + h * s_pth
        for t in range(0, n_tiles):
            pidx = t * PPT + offs_t // PAGE_C                     # the page (list index) of each tile row
            prow = offs_t % PAGE_C
            p_ok = pidx < npg
            slot = tl.load(pt_base + pidx * s_ptp, mask=p_ok, other=0)
            ok = p_ok & ((pidx < npg - 1) | (prow < last))        # the last listed page is cut at the row's extent
            row = slot.to(tl.int64) * PAGE_C + prow
            k = tl.load(k_base + row[:, None] * s_ks + offs_d[None, :] * s_kd, mask=ok[:, None], other=0.0)
            b = tl.load(b_base + row * s_bs, mask=ok, other=0.0).to(tl.float32)
            s = tl.dot(q, tl.trans(k)) + b[None, :]
            s = tl.where(ok[None, :], s * scale_log2, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
            p = tl.where(ok[None, :], tl.exp2(s - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + row[:, None] * s_vs + offs_d[None, :] * s_vd, mask=ok[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        denom = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / denom[:, None]
        tl.store(OUT + r * s_or + qh[:, None] * s_oh + offs_d[None, :] * s_od, o.to(OUT.dtype.element_ty),
                 mask=h_ok[:, None])
else:                                                  # pragma: no cover
    _bi_paged_decode_kernel = None


def bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                    softmax_scale: Optional[float] = None, out: Optional[torch.Tensor] = None,
                    pages_per_tile: int = PAGES_PER_TILE) -> torch.Tensor:
    """The BI paged decode attention (module docstring). One launch, grid (R, Hkv), fixed constants. Shape / dtype
    checks only (no device read). pages_per_tile exists ONLY for the GPU gate's negative control (a batch-dependent
    tile must be caught); every production caller leaves it at PAGES_PER_TILE."""
    if _bi_paged_decode_kernel is None:
        raise RuntimeError("triton is not installed: use bi_paged_decode_reference")
    R, Hq, D, N, Hkv = _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    if pages_per_tile not in (1, 2):
        raise ValueError("pages_per_tile %r (1 = the law; 2 = the negative control)" % (pages_per_tile,))
    group = Hq // Hkv
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("out must be q's shape and dtype with a unit last stride")
    if R == 0:
        return out
    P = page_table.shape[2]
    n_tiles = (P + pages_per_tile - 1) // pages_per_tile
    _bi_paged_decode_kernel[(R, Hkv)](
        q, k_cache, v_cache, bias_cache, out, seq_idx, page_table, num_pages, last_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        bias_cache.stride(0), bias_cache.stride(1), bias_cache.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        page_table.stride(0), page_table.stride(1), page_table.stride(2),
        num_pages.stride(0), num_pages.stride(1),
        n_tiles, scale * LOG2E,
        GROUP=group, BLOCK_H=block_h_for(group), D=D, PAGE_C=PAGE, PPT=pages_per_tile, TILE=PAGE * pages_per_tile,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


def bi_paged_decode_reference(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                              softmax_scale: Optional[float] = None, pages_per_tile: int = PAGES_PER_TILE,
                              split_by_batch: bool = False, tile_by_batch: bool = False) -> torch.Tensor:
    """The CPU TWIN: the kernel's tiled online softmax in torch, per (row, KV head), tile by tile in list order, with
    the kernel's formulas (fp32 scores = q.k + bias, scaled by scale*log2(e); exp2; alpha; l; acc += P (cast to the
    value dtype) . V in fp32; out = acc / l). A row's result is computed from its own q row, page list, num_pages,
    last_len and the rows of its own pages only, so it is batch-invariant by construction. Host loops: tests only.

    NEGATIVE CONTROLS (they break the law on purpose; the tests must catch both):
      split_by_batch  split-KV with num_splits = 1 + (max num_pages over the BATCH) // 4, each split its own online
                      softmax, then the log-sum-exp combine: a row's partition depends on the longest row it is
                      batched with (what a heuristic split-KV kernel does);
      tile_by_batch   pages_per_tile = 1 for a batch of <= 4 rows, else 2: a batch-keyed tile size."""
    R, Hq, D, N, Hkv = _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    group = Hq // Hkv
    scale_log2 = (float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)) * LOG2E
    ppt = (1 if R <= 4 else 2) if tile_by_batch else int(pages_per_tile)
    f32 = torch.float32
    np_l = num_pages.to(torch.int64).cpu().tolist()
    ll_l = last_len.to(torch.int64).cpu().tolist()
    si_l = seq_idx.to(torch.int64).cpu().tolist()
    pt_l = page_table.to(torch.int64).cpu().tolist()
    n_splits = 1 + max(max(row) for row in np_l) // 4 if split_by_batch else 1
    out = torch.empty_like(q)
    for r in range(R):
        n = si_l[r]
        for h in range(Hkv):
            npg, last = np_l[r][h], ll_l[r]
            pages = pt_l[r][h][:npg]
            qf = q[r, h * group:(h + 1) * group].to(f32)                       # (G, D)
            # the list positions of every tile; with a split, contiguous chunks of the list
            chunks = [list(range(npg))]
            if n_splits > 1:
                per = -(-npg // n_splits)
                chunks = [list(range(i, min(i + per, npg))) for i in range(0, npg, per)]
            parts = []
            for ch in chunks:
                m = torch.full((group,), float("-inf"), dtype=f32)
                l = torch.zeros(group, dtype=f32)
                acc = torch.zeros(group, D, dtype=f32)
                for t0 in range(0, len(ch), ppt):
                    tile = ch[t0:t0 + ppt]
                    ks, vs, bs, oks = [], [], [], []
                    for i in tile:
                        s0 = pages[i] * PAGE
                        rows = PAGE if i < npg - 1 else last
                        ok = torch.arange(PAGE) < rows
                        kp = k_cache[n, s0:s0 + PAGE, h].to(f32)
                        vp = v_cache[n, s0:s0 + PAGE, h]
                        bp = bias_cache[n, s0:s0 + PAGE, h].to(f32)
                        ks.append(torch.where(ok.unsqueeze(-1), kp, torch.zeros_like(kp)))
                        vs.append(torch.where(ok.unsqueeze(-1), vp, torch.zeros_like(vp)))
                        bs.append(torch.where(ok, bp, torch.zeros_like(bp)))
                        oks.append(ok)
                    kt, vt, bt, okt = torch.cat(ks), torch.cat(vs), torch.cat(bs), torch.cat(oks)
                    s = qf @ kt.t() + bt.unsqueeze(0)                          # (G, T)
                    s = torch.where(okt.unsqueeze(0), s * scale_log2, torch.full_like(s, float("-inf")))
                    m_new = torch.maximum(m, s.max(dim=1).values)
                    alpha = torch.where(m_new == float("-inf"), torch.ones_like(m_new), torch.exp2(m - m_new))
                    p = torch.where(okt.unsqueeze(0), torch.exp2(s - m_new.unsqueeze(1)), torch.zeros_like(s))
                    l = l * alpha + p.sum(dim=1)
                    acc = acc * alpha.unsqueeze(1) + p.to(vt.dtype).to(f32) @ vt.to(f32)
                    m = m_new
                parts.append((m, l, acc))
            if len(parts) == 1:
                m, l, acc = parts[0]
            else:                                                              # the split-KV combine (negative control)
                mx = torch.stack([p_[0] for p_ in parts]).max(dim=0).values
                l = torch.zeros(group, dtype=f32)
                acc = torch.zeros(group, D, dtype=f32)
                for m_c, l_c, a_c in parts:
                    w = torch.exp2(m_c - mx)
                    l = l + l_c * w
                    acc = acc + a_c * w.unsqueeze(1)
            denom = torch.where(l == 0, torch.ones_like(l), l)
            out[r, h * group:(h + 1) * group] = (acc / denom.unsqueeze(1)).to(q.dtype)
    return out


def attention_oracle(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                     softmax_scale: Optional[float] = None) -> torch.Tensor:
    """The SEMANTICS in float64, one softmax over the concatenated listed rows: softmax((q.k + bias) * scale) . v.
    Agrees with the kernel and the twin to rounding (not bitwise); the tests' and the gate's correctness check."""
    R, Hq, D, N, Hkv = _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    group = Hq // Hkv
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    f64 = torch.float64
    np_l = num_pages.to(torch.int64).cpu().tolist()
    ll_l = last_len.to(torch.int64).cpu().tolist()
    si_l = seq_idx.to(torch.int64).cpu().tolist()
    pt_l = page_table.to(torch.int64).cpu().tolist()
    out = torch.empty(q.shape, dtype=f64, device=q.device)
    for r in range(R):
        n = si_l[r]
        for h in range(Hkv):
            npg = np_l[r][h]
            rows = []
            for i, s in enumerate(pt_l[r][h][:npg]):
                cnt = PAGE if i < npg - 1 else ll_l[r]
                rows.extend(range(s * PAGE, s * PAGE + cnt))
            idx = torch.tensor(rows, dtype=torch.int64, device=q.device)
            k = k_cache[n, idx, h].to(f64)
            v = v_cache[n, idx, h].to(f64)
            b = bias_cache[n, idx, h].to(f64)
            qf = q[r, h * group:(h + 1) * group].to(f64)
            w = torch.softmax((qf @ k.t() + b.unsqueeze(0)) * scale, dim=1)
            out[r, h * group:(h + 1) * group] = w @ v
    return out
