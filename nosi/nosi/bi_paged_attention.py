"""R2-M2 step 7, piece 3: the BATCH-INVARIANT PAGED decode attention (BI attention) of the spec round, version 2
(FAST: a fixed-size split-KV).

Design note: docs/R2_STEP7_SPEC_ROUND.md (f1f9f56) sections 3.4 (the BI paged attention), 5.1-5.2 (slot store, tail
pages). CPU tests: benchmarks/Efficiency/tests_cpu/test_bi_paged_attention.py. GPU gate: benchmarks/Efficiency/
biattn_gate.py (holder step benchmarks/Efficiency/biattn_step.sh).

LEDGER: VERSION 1 WAS CORRECT AND SLOW, AND ITS REGISTERED TIME PREDICTION IS REFUTED. Version 1 (commit c866d7c,
test fix 4c1243f) passed every correctness gate on the A100 (run /mnt/beegfs/ojerbi/r2-runs/biattn/biattn_v1_2188818:
G1-G4 at B 64 and 128). Its time ratio against FA2-NOSA's decode attention (split 4, B rows, 0.205 ms at B 64,
0.393 ms at B 128) was predicted 1.4 / 1.9 / 2.8 (B 64, 1 + U = 1 / 2 / 4) and 1.1 / 1.6 / 2.4 (B 128); MEASURED
8.82 / 13.42 / 21.71 and 8.95 / 13.25 / 22.13 (BI 1.805 / 2.748 / 4.445 ms at B 64): every value outside its band.
The diagnosis (inferred from the measured 147-244 GB/s logical bandwidth and the structure; NOT profiled):
  * one program per (row, KV head) walking all 64 pages: 128 programs at B 64, about one 4-warp CTA per SM;
  * the unit last-dim strides were in do_not_specialize, so Triton could not prove contiguity along D and could not
    vectorize the K / V loads;
  * a vector page-table load fed every tile's addresses, which left little to overlap.
Version 2 changes the schedule, never the law below: a FIXED-SIZE split-KV (PAGES_PER_SPLIT pages per program: 8x
the programs at 64 pages), the unit last-dim stride assumed (checked by the launcher) and not passed, one SCALAR
page-table load per page.

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
  * SPLIT k of a row covers ITS list positions [k * PPS, (k + 1) * PPS), PPS = PAGES_PER_SPLIT, a module constant:
    the partition is a function of the row's own page count and the constant only (not of the longest row, the batch,
    or the number of SMs). One program per (row, KV head, split), grid (R, Hkv, ceil(P / PPS)); a program whose split
    starts past the row's pages returns at once and writes nothing.
  * inside a split: ONE 64-row page per online-softmax step in list order (all GROUP query heads of the KV head in one
    (16, 64) tile: GQA), fp32 (m, l, acc) partials written to a workspace. A page position past the row's last page
    (only in its last split) is an exact identity step (scores -inf: m unchanged, alpha = 2**0 = 1, p = 0).
  * the COMBINE (one program per (row, KV head)) reads the row's own splits in the FIXED order 0, 1, ...:
    M = max over m_k (in that order), then L = sum l_k * 2**(m_k - M) and ACC = sum acc_k * 2**(m_k - M) in that order,
    out = ACC / L. It loops over ceil(P / PPS) slots and masks the ones past the row's split count with exact
    identities (weight 0, other = 0), so the table width P never reaches a row's bits. A row with one split gets
    acc / l exactly (the weight is 2**0 = 1).
  * NUM_WARPS / NUM_STAGES are module constants; no autotune; integer arguments that change with the table width or
    the workspace shape are not specialized; unit last-dim strides are not arguments (Triton may vectorize along D,
    which changes no float op).
  * CANONICAL PAGE ORDER: the caller lists a row's pages in ASCENDING BLOCK-ID order (build_page_table). The tail
    page(s) carry the largest ids, so they come last and the last page is the live tail.
  * masked rows (beyond last_len in the last page, beyond num_pages) are never LOADED: masked loads with other = 0.

NUMERICS (the GPU path). q / k / v / bias in bf16; the two dots on the tensor cores with fp32 accumulation; online
softmax in fp32 in base 2 (exp2, scale * log2(e)); P cast to v's dtype before P.V (as FA2 casts P to bf16); partials
and the combine in fp32; one rounding to q's dtype at the end. fp32 operands are for the CPU interpreter only.
bi_paged_decode_reference is the SAME split structure and update formulas in torch (the CPU twin): not bitwise the
GPU kernel (a CPU matmul is not a tensor-core dot), but the executable definition of the order. Its NEGATIVE CONTROL
knob batch_keyed_split chooses the pages per split from the batch size, as a heuristic split-KV kernel does; the tests
must catch it.

PRIOR ART (cited; no code copied, so no third-party header): the in-house T10 block-diagonal kernel (retroinfer-eval
speckv/kernels/blockdiag_attend.py: fixed tiling, no autotune, a torch twin, the G-Kinv gate); vLLM's unified
attention under batch invariance (one tile size for decode and prefill rows, no batch-keyed split: PA-3 of the step-7
note); flash-decoding's split-KV with a log-sum-exp combine (here with a CONSTANT split size). FA2-NOSA's paged mode
is NOT usable: page_block_size % 256 == 0 and a bias indexed by the logical column (note section 5).

INTERFACE (fixed since c866d7c; workspace is new and optional; pages_per_split is for the negative controls only and
every production caller leaves it at PAGES_PER_SPLIT):
    bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                    softmax_scale=None, out=None, workspace=None, pages_per_split=PAGES_PER_SPLIT) -> (R, Hq, D)
      q           (R, Hq, D)        bf16 (fp16; fp32 interpreter only), stride(-1) == 1
      k_cache     (N, S*64, Hkv, D) q's dtype, stride(-1) == 1          N storage entries (a request's slot store,
      v_cache     (N, S*64, Hkv, D)                                      its tail pages included as slots)
      bias_cache  (N, S*64, Hkv)    q's dtype
      seq_idx     (R,)        int32  the storage entry of row r
      page_table  (R, Hkv, P) int32  slot ids, canonical order (build_page_table); entries >= num_pages never read
      num_pages   (R, Hkv)    int32  >= 1 (the tail page is always listed)
      last_len    (R,)        int32  1 .. 64 (last_len_of(position))
      softmax_scale  float, default D ** -0.5
      workspace   optional (wacc fp32 (R', Hkv, S', BLOCK_H, D), wml fp32 (R', Hkv, S', 2, BLOCK_H)), R' >= R,
                  S' >= ceil(P / PAGES_PER_SPLIT): a fixed buffer (CUDA-graph friendly)
    No host sync; validate_inputs reads values and is for tests / gates only.
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
PAGES_PER_SPLIT = 8          # the FIXED split size (the law: a constant, never a function of the batch)
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


def splits_for(table_width: int, pages_per_split: int = PAGES_PER_SPLIT) -> int:
    """Split slots per (row, head) for a page table of width P: ceil(P / pages_per_split)."""
    return -(-int(table_width) // int(pages_per_split))


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
    @triton.jit(do_not_specialize=["s_ptr", "s_pth", "s_wr", "s_wh", "s_ws", "s_mr", "s_mh", "s_ms"])
    def _bi_split_kernel(
            Q, K, V, BIAS, SEQ, PT, NP, LAST, WACC, WML,
            s_qr, s_qh,
            s_kn, s_ks, s_kh,
            s_vn, s_vs, s_vh,
            s_bn, s_bs, s_bh,
            s_ptr, s_pth,
            s_npr,
            s_wr, s_wh, s_ws, s_wg,
            s_mr, s_mh, s_ms,
            scale_log2,
            GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, PAGE_C: tl.constexpr, PPS: tl.constexpr):
        r = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.program_id(2)
        npg = tl.load(NP + r * s_npr + h)
        p0 = sp * PPS
        if p0 < npg:                                      # a split past the row's pages writes nothing
            n = tl.load(SEQ + r).to(tl.int64)
            last = tl.load(LAST + r)
            offs_h = tl.arange(0, BLOCK_H)
            offs_d = tl.arange(0, D)
            offs_r = tl.arange(0, PAGE_C)
            h_ok = offs_h < GROUP
            q = tl.load(Q + r * s_qr + (h * GROUP + offs_h)[:, None] * s_qh + offs_d[None, :], mask=h_ok[:, None],
                        other=0.0)
            m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
            l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
            acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
            k_base = K + n * s_kn + h * s_kh
            v_base = V + n * s_vn + h * s_vh
            b_base = BIAS + n * s_bn + h * s_bh
            pt_base = PT + r * s_ptr + h * s_pth
            for i in range(0, PPS):
                p = p0 + i
                p_ok = p < npg
                slot = tl.load(pt_base + p, mask=p_ok, other=0).to(tl.int64)       # one scalar per page
                ok = p_ok & ((p < npg - 1) | (offs_r < last))                     # the last page cut at the extent
                rows = slot * PAGE_C + offs_r
                k = tl.load(k_base + rows[:, None] * s_ks + offs_d[None, :], mask=ok[:, None], other=0.0)
                b = tl.load(b_base + rows * s_bs, mask=ok, other=0.0).to(tl.float32)
                sc = tl.dot(q, tl.trans(k)) + b[None, :]
                sc = tl.where(ok[None, :], sc * scale_log2, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(sc, 1))
                alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
                pr = tl.where(ok[None, :], tl.exp2(sc - m_new[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(pr, 1)
                v = tl.load(v_base + rows[:, None] * s_vs + offs_d[None, :], mask=ok[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
                m_i = m_new
            w_base = WACC + r * s_wr + h * s_wh + sp * s_ws
            tl.store(w_base + offs_h[:, None] * s_wg + offs_d[None, :], acc)
            ml_base = WML + r * s_mr + h * s_mh + sp * s_ms
            tl.store(ml_base + offs_h, m_i)
            tl.store(ml_base + BLOCK_H + offs_h, l_i)

    @triton.jit(do_not_specialize=["s_wr", "s_wh", "s_ws", "s_mr", "s_mh", "s_ms", "n_split_slots"])
    def _bi_combine_kernel(
            WACC, WML, NP, OUT,
            s_wr, s_wh, s_ws, s_wg,
            s_mr, s_mh, s_ms,
            s_npr,
            s_or, s_oh,
            n_split_slots,
            GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, PPS: tl.constexpr):
        r = tl.program_id(0)
        h = tl.program_id(1)
        npg = tl.load(NP + r * s_npr + h)
        ns = (npg + PPS - 1) // PPS
        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D)
        h_ok = offs_h < GROUP
        ml_row = WML + r * s_mr + h * s_mh
        w_row = WACC + r * s_wr + h * s_wh
        mx = tl.full([BLOCK_H], float("-inf"), tl.float32)
        for k in range(0, n_split_slots):                 # pass 1: the max, splits in order
            ok = k < ns
            m_k = tl.load(ml_row + k * s_ms + offs_h, mask=ok & (offs_h >= 0), other=float("-inf"))
            mx = tl.maximum(mx, m_k)
        lsum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        for k in range(0, n_split_slots):                 # pass 2: the weighted sums, splits in order
            ok = k < ns
            m_k = tl.load(ml_row + k * s_ms + offs_h, mask=ok & (offs_h >= 0), other=float("-inf"))
            l_k = tl.load(ml_row + k * s_ms + BLOCK_H + offs_h, mask=ok & (offs_h >= 0), other=0.0)
            a_k = tl.load(w_row + k * s_ws + offs_h[:, None] * s_wg + offs_d[None, :],
                          mask=ok & (offs_h[:, None] >= 0) & (offs_d[None, :] >= 0), other=0.0)
            wgt = tl.where(ok & (offs_h >= 0), tl.exp2(m_k - mx), 0.0)
            lsum = lsum + l_k * wgt
            acc = acc + a_k * wgt[:, None]
        o = acc / tl.where(lsum == 0.0, 1.0, lsum)[:, None]
        tl.store(OUT + r * s_or + (h * GROUP + offs_h)[:, None] * s_oh + offs_d[None, :],
                 o.to(OUT.dtype.element_ty), mask=h_ok[:, None])
else:                                                  # pragma: no cover
    _bi_split_kernel = _bi_combine_kernel = None


def bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                    softmax_scale: Optional[float] = None, out: Optional[torch.Tensor] = None,
                    workspace=None, pages_per_split: int = PAGES_PER_SPLIT) -> torch.Tensor:
    """The BI paged decode attention (module docstring): the split launch (R, Hkv, ceil(P / PPS)) and the combine
    launch (R, Hkv), fixed constants. Shape / dtype checks only (no device read). pages_per_split exists ONLY for
    the tests' and the gate's NEGATIVE control (a split size chosen by the batch must be caught); every production
    caller leaves it at PAGES_PER_SPLIT."""
    if _bi_split_kernel is None:
        raise RuntimeError("triton is not installed: use bi_paged_decode_reference")
    R, Hq, D, N, Hkv = _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    pps = int(pages_per_split)
    if pps < 1:
        raise ValueError("pages_per_split %r" % (pages_per_split,))
    group = Hq // Hkv
    bh = block_h_for(group)
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("out must be q's shape and dtype with a unit last stride")
    if R == 0:
        return out
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1 or page_table.stride(-1) != 1 or num_pages.stride(-1) != 1:
        raise ValueError("unit last strides are assumed by the kernels")
    S = splits_for(page_table.shape[2], pps)
    if workspace is None:
        wacc = torch.empty((R, Hkv, S, bh, D), dtype=torch.float32, device=q.device)
        wml = torch.empty((R, Hkv, S, 2, bh), dtype=torch.float32, device=q.device)
    else:
        wacc, wml = workspace
        if wacc.dtype != torch.float32 or wml.dtype != torch.float32 or wacc.dim() != 5 or wml.dim() != 5 \
                or wacc.shape[0] < R or wacc.shape[1] != Hkv or wacc.shape[2] < S or tuple(wacc.shape[3:]) != (bh, D) \
                or wml.shape[0] < R or wml.shape[1] != Hkv or wml.shape[2] < S or tuple(wml.shape[3:]) != (2, bh) \
                or not (wacc.is_contiguous() and wml.is_contiguous()):
            raise ValueError("workspace must be contiguous fp32 wacc (>= R, Hkv, >= S, %d, %d) and wml (>= R, Hkv, "
                             ">= S, 2, %d)" % (bh, D, bh))
    _bi_split_kernel[(R, Hkv, S)](
        q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, wacc, wml,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        bias_cache.stride(0), bias_cache.stride(1), bias_cache.stride(2),
        page_table.stride(0), page_table.stride(1),
        num_pages.stride(0),
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wml.stride(0), wml.stride(1), wml.stride(2),
        scale * LOG2E,
        GROUP=group, BLOCK_H=bh, D=D, PAGE_C=PAGE, PPS=pps, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    _bi_combine_kernel[(R, Hkv)](
        wacc, wml, num_pages, out,
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wml.stride(0), wml.stride(1), wml.stride(2),
        num_pages.stride(0),
        out.stride(0), out.stride(1),
        S,
        GROUP=group, BLOCK_H=bh, D=D, PPS=pps, num_warps=NUM_WARPS)
    return out


def bi_paged_decode_reference(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                              softmax_scale: Optional[float] = None, pages_per_split: int = PAGES_PER_SPLIT,
                              batch_keyed_split: bool = False) -> torch.Tensor:
    """The CPU TWIN: the kernels' order in torch, per (row, KV head): split k = list positions [k*PPS, (k+1)*PPS),
    page by page inside a split (fp32 scores = q.k + bias, scaled by scale*log2(e); exp2; alpha; l; acc += P (cast to
    the value dtype) . V in fp32), then the combine over the row's own splits in order 0, 1, ... (the max first, then
    l and acc weighted by 2**(m_k - M)), out = acc / l. A row's result is computed from its own q row, page list,
    num_pages, last_len and the rows of its own pages only, so it is batch-invariant by construction. Host loops:
    tests only.

    NEGATIVE CONTROL: batch_keyed_split chooses the pages per split from the BATCH SIZE (8 pages for a batch of <= 4
    rows, else 4), as a heuristic split-KV kernel does; a row's partition then depends on its neighbours."""
    R, Hq, D, N, Hkv = _check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    group = Hq // Hkv
    scale_log2 = (float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)) * LOG2E
    pps = (8 if R <= 4 else 4) if batch_keyed_split else int(pages_per_split)
    f32 = torch.float32
    ninf = float("-inf")
    np_l = num_pages.to(torch.int64).cpu().tolist()
    ll_l = last_len.to(torch.int64).cpu().tolist()
    si_l = seq_idx.to(torch.int64).cpu().tolist()
    pt_l = page_table.to(torch.int64).cpu().tolist()
    out = torch.empty_like(q)
    for r in range(R):
        n = si_l[r]
        last = ll_l[r]
        for h in range(Hkv):
            npg = np_l[r][h]
            pages = pt_l[r][h][:npg]
            qf = q[r, h * group:(h + 1) * group].to(f32)                       # (G, D)
            parts = []
            for p0 in range(0, npg, pps):
                m = torch.full((group,), ninf, dtype=f32)
                l = torch.zeros(group, dtype=f32)
                acc = torch.zeros(group, D, dtype=f32)
                for p in range(p0, min(p0 + pps, npg)):
                    s0 = pages[p] * PAGE
                    rows = PAGE if p < npg - 1 else last
                    ok = torch.arange(PAGE) < rows
                    kp = k_cache[n, s0:s0 + PAGE, h].to(f32)
                    vp = v_cache[n, s0:s0 + PAGE, h]
                    bp = bias_cache[n, s0:s0 + PAGE, h].to(f32)
                    kp = torch.where(ok.unsqueeze(-1), kp, torch.zeros_like(kp))
                    vp = torch.where(ok.unsqueeze(-1), vp, torch.zeros_like(vp))
                    bp = torch.where(ok, bp, torch.zeros_like(bp))
                    s = qf @ kp.t() + bp.unsqueeze(0)                          # (G, 64)
                    s = torch.where(ok.unsqueeze(0), s * scale_log2, torch.full_like(s, ninf))
                    m_new = torch.maximum(m, s.max(dim=1).values)
                    alpha = torch.where(m_new == ninf, torch.ones_like(m_new), torch.exp2(m - m_new))
                    pr = torch.where(ok.unsqueeze(0), torch.exp2(s - m_new.unsqueeze(1)), torch.zeros_like(s))
                    l = l * alpha + pr.sum(dim=1)
                    acc = acc * alpha.unsqueeze(1) + pr.to(vp.dtype).to(f32) @ vp.to(f32)
                    m = m_new
                parts.append((m, l, acc))
            mx = torch.full((group,), ninf, dtype=f32)
            for m_k, _, _ in parts:                                            # the combine: max, in order
                mx = torch.maximum(mx, m_k)
            lsum = torch.zeros(group, dtype=f32)
            tot = torch.zeros(group, D, dtype=f32)
            for m_k, l_k, a_k in parts:                                        # then the sums, in order
                wgt = torch.exp2(m_k - mx)
                lsum = lsum + l_k * wgt
                tot = tot + a_k * wgt.unsqueeze(1)
            denom = torch.where(lsum == 0, torch.ones_like(lsum), lsum)
            out[r, h * group:(h + 1) * group] = (tot / denom.unsqueeze(1)).to(q.dtype)
    return out


def attention_oracle(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                     softmax_scale: Optional[float] = None) -> torch.Tensor:
    """The SEMANTICS in float64, one softmax over the concatenated listed rows: softmax((q.k + bias) * scale) . v.
    Agrees with the kernels and the twin to rounding (not bitwise); the tests' and the gate's correctness check."""
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
