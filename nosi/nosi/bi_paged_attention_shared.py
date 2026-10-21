"""R2 shared ready-KV reads (paper row 6; audit kstep-cpu-runs/paper-notes/recon-rows-5-6.md): a variant of piece 3 v2
(nosi/nosi/bi_paged_attention.py, pinned blob 39ed5e18 of 7c826c1, NOT modified) in which the rows of ONE REQUEST (its
draft row and its verify rows: the rows with the same seq_idx) read each selected KV page from HBM ONCE per program
instead of once per row. Design note: docs/R2_SHAREDKV_PLAN.md. CPU tests: benchmarks/Efficiency/tests_cpu/
test_bi_paged_attention_shared.py. SOURCE ONLY, not run on a GPU.

WHAT IS SHARED. Piece 3 runs one program per (row, KV head, split) and loads the pages of the row's own list. Here a
program serves a GROUP (up to NR rows of one storage entry: NR = rows_max, at most ROWS_MAX = 1 + K = 4) and one KV
head. It walks the group's UNION page list once (the distinct (block id, slot) pairs of its rows, in ascending
(block id, slot) order) and loads each 64-row K / V / bias tile once; every row of the group then runs ITS OWN step on
that tile.

THE LAW IS KEPT (bitwise equal to piece 3 per row, by construction; the GPU gate decides the claim):
  * every row keeps piece 3's split partition: split k of a row = ITS list positions [k * PPS, (k + 1) * PPS), the
    same PPS = PAGES_PER_SPLIT, the same fp32 (m, l, acc) partial per (row, head, split) in the same workspace layout,
    and piece 3's own COMBINE kernel (_bi_combine_kernel of the base module, the same launch);
  * a row's step on a page is piece 3's step with BITWISE THE SAME OPERANDS: the shared tile is loaded under the OR of
    the rows' masks, and each row selects it with its own mask (tl.where(ok, tile, 0)), which is exactly piece 3's
    masked load with other = 0. The dot shapes are piece 3's: (BLOCK_H, D) x (D, 64) and (BLOCK_H, 64) x (64, D) per
    row (the rows are NOT stacked into a larger M tile: a larger M changes the dot / reduction layout, so the rows'
    bits would depend on the group); NUM_WARPS, NUM_STAGES and BLOCK_H are piece 3's;
  * a union page that is not in a row's current split is, for that row, piece 3's EXACT IDENTITY STEP (all scores
    -inf: m unchanged, alpha = 1, p = 0, l + 0, acc * 1 + 0 . 0). Piece 3 runs the same step past a row's last page.
    The one known non-identity: an identity step maps an acc entry of exactly -0.0 to +0.0 (the dot of zeros is +0);
    piece 3 would keep -0.0 there until its next step. It changes only the sign of an exact zero of the output, and
    only when a row's acc entry is -0.0 at that moment (an underflow of a negative acc times a tiny alpha). The bitwise
    gate would report it as a sign-of-zero difference.
  * union order: the (block id, slot) pairs ascending. Each row lists its pages in ascending block id (piece 3's
    canonical order), so every row's own list is a subsequence of the union order and a row sees its pages in its own
    order. The block ids come from the caller (page_key); the slots alone do not give the order (ring slots).

TWO SCHEDULES (constexpr WHOLE):
  "split" (default): grid (R, Hkv, S), program (group g, head h, split k) runs, for every row of g, that row's split k
      over the union window that holds the rows' split-k pages. Each row's split chain is computed entirely by one
      program. A page that one row has in split k and another row in split k + 1 is loaded by both programs, so the
      loads are sum_k |union_j split_k(j)|: between the union and the sum (they equal the union when the rows'
      split boundaries fall on the same pages).
  "whole": grid (R, Hkv, 1), program (g, h) walks the WHOLE union once and flushes each row's partial when the row
      crosses its own split boundary (then resets it to the fresh state -inf / 0 / 0). Loads = the union exactly, but
      only R/NR * Hkv programs: 128 at B 64 with NOSA-8B's Hkv = 2 (piece 3 v1 had 128 programs and was 8.8x slower
      than FA2: its ledger). Kept for the GPU A/B of bytes against parallelism.
  Bitwise the result is the same in both (the same per-row chains); only the loads and the program count differ.

GROUPS. Rows with the same seq_idx form a group, in ascending row index, cut into chunks of NR rows (a request with
more rows than NR gets several groups: still correct, less sharing). Built on the device (sort, cumsum, scatter; no
host read). The grid's first dimension is R (an upper bound of the group count, no host read); programs past the last
group find an empty window and return.

ADDRESS COMPUTATION: unchanged from piece 3 (k_base = K + n * s_kn + h * s_kh; rows = slot * PAGE_C + offs_r; the
same strides). The flat-position X-pool addressing of the parallel branch (nosi-r2-xpool, docs/R2_XPOOL_PLAN.md) is a
separate change to piece 3's kernel; porting it here is a follow-up after both are gated.

INTERFACE:
    SharedKV(base, mode="split", rows_max=ROWS_MAX)    base = the loaded piece 3 module (pinned blob)
      .bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, softmax_scale=None,
                       out=None, workspace=None, pages_per_split=PPS, page_key=None, rows_max=None)
        page_key (R, Hkv, P) int32: the block id of each listed page (the tick's selection, compacted like the table).
        Without page_key, or with rows_max == 1 (no request has two rows: nothing to share), it calls piece 3 itself
        (the reference / serial paths call without keys); .stats counts each kind of call, so a gate can assert the
        shared kernel really ran.
      .needs_page_key = True                           the tick passes page_key / rows_max when it sees this
    bi_paged_decode_shared(base, ...)                  the launcher (the same arguments, page_key required)
    build_union(seq_idx, page_table, num_pages, page_key, rows_max, pps, whole)   the device builder
    count_loads=... (tests / gates only): a (2,) int32 tensor; the kernel adds 1 per (program, union page) tile it
        loads and the number of 64-row tile rows it loads (a constexpr: the production kernel has no counter).
PRIOR ART (cited; no code copied): shared-prefix / cascade attention (Hydragen; FlashInfer cascade / MultiLevelCascade:
a shared KV read once for several queries, merged by log-sum-exp states), tree / speculative verify attention
(SpecInfer, Medusa: several query rows of one sequence over one KV), multi-query split-KV decode (FlashDecoding with
q_len > 1); in-house: the 4.5 joint-HBM harness (nosi-joint-hbm 1c4af3e: one union window per request, every row a
query of one FA2-NOSA batch entry, per-row masks; measured attention DRAM 0.363x, NOT batch-invariant). None keeps a
per-row split partition and a per-row reduction order, which the law needs, so the kernel is new; its per-row step is
piece 3's, its combine is piece 3's kernel.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

try:                                                   # triton is needed only to launch the kernel
    import triton
    import triton.language as tl
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None

ROWS_MAX = 4                 # 1 draft row + K = 3 verify rows per request (the R2 cohort)
MODES = ("split", "whole")
_BIG = torch.iinfo(torch.int64).max


def build_union(seq_idx: torch.Tensor, page_table: torch.Tensor, num_pages: torch.Tensor, page_key: torch.Tensor,
                rows_max: int, pps: int, whole: bool = False) -> dict:
    """The groups and their union page lists, device ops only (no host read). Returns int32 contiguous tensors:
      grow   (R, NR)          the rows of group g (ascending row index), -1 padded; groups past the last are all -1
      ut     (R, Hkv, U)      the union's slots in ascending (block id, slot) order, -1 padded; U = NR * P
      ucnt   (R, Hkv)         the union size
      rowpos (R, Hkv, NR, U)  row j's list position of union page u, or -1 when row j does not list it
      win    (R, Hkv, S', 2)  [u0, u1) per program: split mode S' = ceil(P / pps), the union range holding the rows'
                              split-k pages (empty: 0, 0); whole mode S' = 1, (0, ucnt)
      uidx   (R, Hkv, NR, P)  the union index of row j's list position i (tests), -1 past num_pages / absent rows"""
    R, Hkv, P = page_table.shape
    NR = int(rows_max)
    dev = page_table.device
    i64 = torch.int64
    seq = seq_idx.to(i64)
    order = torch.sort(seq, stable=True).indices
    s_sorted = seq[order]
    idx = torch.arange(R, device=dev, dtype=i64)
    newg = torch.ones(R, dtype=torch.bool, device=dev)
    newg[1:] = s_sorted[1:] != s_sorted[:-1]
    start = torch.where(newg, idx, torch.zeros_like(idx)).cummax(0).values
    m = idx - start
    gid = (m % NR == 0).to(i64).cumsum(0) - 1
    grow = torch.full((R, NR), -1, dtype=i64, device=dev)
    grow[gid, m % NR] = order
    pos = torch.arange(P, device=dev, dtype=i64)
    valid = pos.view(1, 1, P) < num_pages.to(i64).unsqueeze(-1)
    comb = torch.where(valid, (page_key.to(i64) << 32) | page_table.to(i64),
                       torch.full(valid.shape, _BIG, dtype=i64, device=dev))
    gv = grow.clamp(min=0)
    ge = comb[gv]                                                                   # (R, NR, Hkv, P)
    ge = torch.where((grow >= 0).view(R, NR, 1, 1), ge, torch.full_like(ge, _BIG))
    U = NR * P
    flat = ge.permute(0, 2, 1, 3).reshape(R, Hkv, U)
    sv, si = torch.sort(flat, dim=-1, stable=True)
    okv = sv != _BIG
    first = okv.clone()
    first[..., 1:] &= sv[..., 1:] != sv[..., :-1]
    uid = first.to(i64).cumsum(-1) - 1                                              # union index of each sorted entry
    ucnt = first.sum(-1)
    dest = torch.where(first, uid, torch.full_like(uid, U))
    ut = torch.full((R, Hkv, U + 1), -1, dtype=i64, device=dev)
    ut.scatter_(-1, dest, torch.where(first, sv & 0xFFFFFFFF, torch.full_like(sv, -1)))
    jj, ii = si // P, si % P
    dest2 = torch.where(okv, uid, torch.full_like(uid, U))
    rowpos = torch.full((R, Hkv, NR * (U + 1)), -1, dtype=i64, device=dev)
    rowpos.scatter_(-1, jj * (U + 1) + dest2, torch.where(okv, ii, torch.full_like(ii, -1)))
    rowpos = rowpos.view(R, Hkv, NR, U + 1)[..., :U]
    uflat = torch.full((R, Hkv, U), -1, dtype=i64, device=dev)
    uflat.scatter_(-1, si, torch.where(okv, uid, torch.full_like(uid, -1)))
    uidx = uflat.view(R, Hkv, NR, P)
    if whole:
        win = torch.stack([torch.zeros_like(ucnt), ucnt], -1).view(R, Hkv, 1, 2)
    else:
        S = -(-P // int(pps))
        pad = S * int(pps) - P
        ux = torch.nn.functional.pad(uidx, (0, pad), value=-1).view(R, Hkv, NR, S, int(pps))
        v = ux >= 0
        lo = torch.where(v, ux, torch.full_like(ux, _BIG)).amin(dim=(2, 4))
        hi = torch.where(v, ux, torch.full_like(ux, -1)).amax(dim=(2, 4)) + 1
        lo = torch.where(hi > 0, lo, torch.zeros_like(lo))
        win = torch.stack([lo, hi], -1)                                             # (R, Hkv, S, 2)
    c = lambda t: t.to(torch.int32).contiguous()                                    # noqa: E731
    return dict(grow=c(grow), ut=c(ut[..., :U]), ucnt=c(ucnt), rowpos=c(rowpos), win=c(win), uidx=c(uidx),
                rows_max=NR, whole=whole is True, pps=int(pps))


def plant_drop_page(union: dict, g: int = 0, h: int = 0, u: Optional[int] = None) -> dict:
    """NEGATIVE CONTROL (tests and the GPU cell): a copy of the union in which NO row processes union page u of group g,
    head h (its rowpos column set to -1: every row takes the identity step there). u None = the group's LAST union page
    (device index, no host read; in the tick that page holds the newest tokens, so its weight is rarely negligible).
    The plant must change some output; whether it did on the run's own inputs is the caller's PRECONDITION (L34)."""
    rp = union["rowpos"].clone()
    if u is None:
        col = (union["ucnt"][g, h].to(torch.int64) - 1).clamp(min=0).view(1)
    else:
        col = torch.tensor([int(u)], dtype=torch.int64, device=rp.device)
    rp[g, h].index_fill_(1, col, -1)
    return dict(union, rowpos=rp.contiguous())


def validate_keys(page_table: torch.Tensor, num_pages: torch.Tensor, page_key: torch.Tensor) -> None:
    """HOST READS (tests / gates only): every row's keys strictly ascending over its listed pages, all >= 0 and < 2**31
    (the union order needs it: each row's list must be a subsequence of the (key, slot) order)."""
    if page_key.shape != page_table.shape or page_key.dtype != torch.int32 or not page_key.is_contiguous():
        raise ValueError("page_key must be contiguous int32 shaped like page_table %s" % (tuple(page_table.shape),))
    pk = page_key.to(torch.int64).cpu()
    npg = num_pages.to(torch.int64).cpu()
    R, Hkv, _ = pk.shape
    for r in range(R):
        for h in range(Hkv):
            ks = pk[r, h, :int(npg[r, h])].tolist()
            if any(k < 0 for k in ks) or any(b <= a for a, b in zip(ks, ks[1:])):
                raise ValueError("row %d head %d: keys not strictly ascending / negative: %s" % (r, h, ks))


if triton is not None:
    @triton.jit
    def _sk_row_init(Q, NP, LAST, GROW, g, j, h, s_gr, s_qr, s_qh, s_npr, offs_h, offs_d,
                     GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr):
        r = tl.load(GROW + g * s_gr + j)
        valid = r >= 0
        rc = tl.maximum(r, 0)
        npg = tl.where(valid, tl.load(NP + rc * s_npr + h), 0)
        last = tl.load(LAST + rc)
        h_ok = offs_h < GROUP
        q = tl.load(Q + rc * s_qr + (h * GROUP + offs_h)[:, None] * s_qh + offs_d[None, :], mask=h_ok[:, None],
                    other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        return r, valid, npg, last, q, m_i, l_i, acc

    @triton.jit
    def _sk_row_ok(rpb, j, s_rpj, npg, last, sp, offs_r, PPS: tl.constexpr, WHOLE: tl.constexpr):
        rp = tl.load(rpb + j * s_rpj)
        if WHOLE:
            inn = rp >= 0
        else:
            inn = (rp >= sp * PPS) & (rp < sp * PPS + PPS)
        ok = inn & ((rp < npg - 1) | (offs_r < last))                     # piece 3: the last page cut at the extent
        return ok, inn, rp

    @triton.jit
    def _sk_row_step(q, k, b, v, ok, m_i, l_i, acc, scale_log2):
        """Piece 3's page step (bi_paged_attention.py 7c826c1 :239-249) on the row's own masked copy of the tile."""
        kz = tl.where(ok[:, None], k, tl.zeros_like(k))
        bz = tl.where(ok, b, tl.zeros_like(b)).to(tl.float32)
        sc = tl.dot(q, tl.trans(kz)) + bz[None, :]
        sc = tl.where(ok[None, :], sc * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
        pr = tl.where(ok[None, :], tl.exp2(sc - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(pr, 1)
        vz = tl.where(ok[:, None], v, tl.zeros_like(v))
        acc = acc * alpha[:, None] + tl.dot(pr.to(vz.dtype), vz)
        return m_new, l_i, acc

    @triton.jit
    def _sk_row_store(WACC, WML, r, h, kk, cond, m_i, l_i, acc, s_wr, s_wh, s_ws, s_wg, s_mr, s_mh, s_ms,
                      offs_h, offs_d, BLOCK_H: tl.constexpr):
        """Piece 3's partial store (:250-254) for split kk of row r, under the scalar cond."""
        rc = tl.maximum(r, 0)
        kc = tl.maximum(kk, 0)
        w_base = WACC + rc * s_wr + h * s_wh + kc * s_ws
        tl.store(w_base + offs_h[:, None] * s_wg + offs_d[None, :], acc,
                 mask=cond & (offs_h[:, None] >= 0) & (offs_d[None, :] >= 0))
        ml_base = WML + rc * s_mr + h * s_mh + kc * s_ms
        tl.store(ml_base + offs_h, m_i, mask=cond & (offs_h >= 0))
        tl.store(ml_base + BLOCK_H + offs_h, l_i, mask=cond & (offs_h >= 0))

    @triton.jit
    def _sk_row_flush(WACC, WML, r, h, rp, inn, m_i, l_i, acc, s_wr, s_wh, s_ws, s_wg, s_mr, s_mh, s_ms,
                      offs_h, offs_d, BLOCK_H: tl.constexpr, PPS: tl.constexpr):
        """WHOLE mode: before the row's first page of split k > 0, store split k - 1 and reset to the fresh state."""
        cond = inn & (rp > 0) & (rp % PPS == 0)
        _sk_row_store(WACC, WML, r, h, rp // PPS - 1, cond, m_i, l_i, acc, s_wr, s_wh, s_ws, s_wg, s_mr, s_mh, s_ms,
                      offs_h, offs_d, BLOCK_H)
        m_i = tl.where(cond, tl.full([BLOCK_H], float("-inf"), tl.float32), m_i)
        l_i = tl.where(cond, tl.zeros_like(l_i), l_i)
        acc = tl.where(cond, tl.zeros_like(acc), acc)
        return m_i, l_i, acc

    @triton.jit(do_not_specialize=["s_gr", "s_utg", "s_uth", "s_rpg", "s_rph", "s_rpj", "s_wng", "s_wnh", "s_wns",
                                   "s_wr", "s_wh", "s_ws", "s_mr", "s_mh", "s_ms"])
    def _bi_shared_kernel(
            Q, K, V, BIAS, SEQ, NP, LAST, GROW, UT, ROWPOS, WIN, WACC, WML, CNT,
            s_qr, s_qh,
            s_kn, s_ks, s_kh,
            s_vn, s_vs, s_vh,
            s_bn, s_bs, s_bh,
            s_npr,
            s_gr, s_utg, s_uth, s_rpg, s_rph, s_rpj, s_wng, s_wnh, s_wns,
            s_wr, s_wh, s_ws, s_wg,
            s_mr, s_mh, s_ms,
            scale_log2,
            GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, PAGE_C: tl.constexpr, PPS: tl.constexpr,
            NR: tl.constexpr, WHOLE: tl.constexpr, COUNT: tl.constexpr):
        g = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.program_id(2)
        wb = WIN + g * s_wng + h * s_wnh + sp * s_wns
        u0 = tl.load(wb)
        u1 = tl.load(wb + 1)
        if u1 > u0:                                       # an empty window (no row of g has split sp) writes nothing
            offs_h = tl.arange(0, BLOCK_H)
            offs_d = tl.arange(0, D)
            offs_r = tl.arange(0, PAGE_C)
            r0, v0, np0, la0, q0, m0, l0, a0 = _sk_row_init(Q, NP, LAST, GROW, g, 0, h, s_gr, s_qr, s_qh, s_npr,
                                                            offs_h, offs_d, GROUP, BLOCK_H, D)
            if NR >= 2:
                r1, v1, np1, la1, q1, m1, l1, a1 = _sk_row_init(Q, NP, LAST, GROW, g, 1, h, s_gr, s_qr, s_qh, s_npr,
                                                                offs_h, offs_d, GROUP, BLOCK_H, D)
            if NR >= 3:
                r2, v2, np2, la2, q2, m2, l2, a2 = _sk_row_init(Q, NP, LAST, GROW, g, 2, h, s_gr, s_qr, s_qh, s_npr,
                                                                offs_h, offs_d, GROUP, BLOCK_H, D)
            if NR >= 4:
                r3, v3, np3, la3, q3, m3, l3, a3 = _sk_row_init(Q, NP, LAST, GROW, g, 3, h, s_gr, s_qr, s_qh, s_npr,
                                                                offs_h, offs_d, GROUP, BLOCK_H, D)
            n = tl.load(SEQ + tl.maximum(r0, 0)).to(tl.int64)     # one storage entry per group (by construction)
            k_base = K + n * s_kn + h * s_kh
            v_base = V + n * s_vn + h * s_vh
            b_base = BIAS + n * s_bn + h * s_bh
            ut_base = UT + g * s_utg + h * s_uth
            rp_base = ROWPOS + g * s_rpg + h * s_rph
            for u in range(u0, u1):
                slot = tl.load(ut_base + u).to(tl.int64)                    # one scalar per union page
                rpb = rp_base + u
                ok0, in0, rp0 = _sk_row_ok(rpb, 0, s_rpj, np0, la0, sp, offs_r, PPS, WHOLE)
                ld = ok0
                if NR >= 2:
                    ok1, in1, rp1 = _sk_row_ok(rpb, 1, s_rpj, np1, la1, sp, offs_r, PPS, WHOLE)
                    ld = ld | ok1
                if NR >= 3:
                    ok2, in2, rp2 = _sk_row_ok(rpb, 2, s_rpj, np2, la2, sp, offs_r, PPS, WHOLE)
                    ld = ld | ok2
                if NR >= 4:
                    ok3, in3, rp3 = _sk_row_ok(rpb, 3, s_rpj, np3, la3, sp, offs_r, PPS, WHOLE)
                    ld = ld | ok3
                rows = slot * PAGE_C + offs_r
                k = tl.load(k_base + rows[:, None] * s_ks + offs_d[None, :], mask=ld[:, None], other=0.0)
                b = tl.load(b_base + rows * s_bs, mask=ld, other=0.0)
                v = tl.load(v_base + rows[:, None] * s_vs + offs_d[None, :], mask=ld[:, None], other=0.0)
                if COUNT:
                    nld = tl.sum(ld.to(tl.int32), 0)
                    tl.atomic_add(CNT, tl.where(nld > 0, 1, 0))
                    tl.atomic_add(CNT + 1, nld)
                if WHOLE:
                    m0, l0, a0 = _sk_row_flush(WACC, WML, r0, h, rp0, in0, m0, l0, a0, s_wr, s_wh, s_ws, s_wg,
                                               s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H, PPS)
                m0, l0, a0 = _sk_row_step(q0, k, b, v, ok0, m0, l0, a0, scale_log2)
                if NR >= 2:
                    if WHOLE:
                        m1, l1, a1 = _sk_row_flush(WACC, WML, r1, h, rp1, in1, m1, l1, a1, s_wr, s_wh, s_ws, s_wg,
                                                   s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H, PPS)
                    m1, l1, a1 = _sk_row_step(q1, k, b, v, ok1, m1, l1, a1, scale_log2)
                if NR >= 3:
                    if WHOLE:
                        m2, l2, a2 = _sk_row_flush(WACC, WML, r2, h, rp2, in2, m2, l2, a2, s_wr, s_wh, s_ws, s_wg,
                                                   s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H, PPS)
                    m2, l2, a2 = _sk_row_step(q2, k, b, v, ok2, m2, l2, a2, scale_log2)
                if NR >= 4:
                    if WHOLE:
                        m3, l3, a3 = _sk_row_flush(WACC, WML, r3, h, rp3, in3, m3, l3, a3, s_wr, s_wh, s_ws, s_wg,
                                                   s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H, PPS)
                    m3, l3, a3 = _sk_row_step(q3, k, b, v, ok3, m3, l3, a3, scale_log2)
            # the last partial of each row: split sp (split mode, if the row has it) or its last split (whole mode)
            if WHOLE:
                _sk_row_store(WACC, WML, r0, h, (np0 - 1) // PPS, v0, m0, l0, a0, s_wr, s_wh, s_ws, s_wg, s_mr, s_mh,
                              s_ms, offs_h, offs_d, BLOCK_H)
            else:
                _sk_row_store(WACC, WML, r0, h, sp, v0 & (sp * PPS < np0), m0, l0, a0, s_wr, s_wh, s_ws, s_wg, s_mr,
                              s_mh, s_ms, offs_h, offs_d, BLOCK_H)
            if NR >= 2:
                if WHOLE:
                    _sk_row_store(WACC, WML, r1, h, (np1 - 1) // PPS, v1, m1, l1, a1, s_wr, s_wh, s_ws, s_wg, s_mr,
                                  s_mh, s_ms, offs_h, offs_d, BLOCK_H)
                else:
                    _sk_row_store(WACC, WML, r1, h, sp, v1 & (sp * PPS < np1), m1, l1, a1, s_wr, s_wh, s_ws, s_wg,
                                  s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H)
            if NR >= 3:
                if WHOLE:
                    _sk_row_store(WACC, WML, r2, h, (np2 - 1) // PPS, v2, m2, l2, a2, s_wr, s_wh, s_ws, s_wg, s_mr,
                                  s_mh, s_ms, offs_h, offs_d, BLOCK_H)
                else:
                    _sk_row_store(WACC, WML, r2, h, sp, v2 & (sp * PPS < np2), m2, l2, a2, s_wr, s_wh, s_ws, s_wg,
                                  s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H)
            if NR >= 4:
                if WHOLE:
                    _sk_row_store(WACC, WML, r3, h, (np3 - 1) // PPS, v3, m3, l3, a3, s_wr, s_wh, s_ws, s_wg, s_mr,
                                  s_mh, s_ms, offs_h, offs_d, BLOCK_H)
                else:
                    _sk_row_store(WACC, WML, r3, h, sp, v3 & (sp * PPS < np3), m3, l3, a3, s_wr, s_wh, s_ws, s_wg,
                                  s_mr, s_mh, s_ms, offs_h, offs_d, BLOCK_H)
else:                                                  # pragma: no cover
    _bi_shared_kernel = None


def bi_paged_decode_shared(base, q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, page_key,
                           rows_max: int = ROWS_MAX, mode: str = "split", softmax_scale: Optional[float] = None,
                           out: Optional[torch.Tensor] = None, workspace=None, pages_per_split: Optional[int] = None,
                           union: Optional[dict] = None, count_loads: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The shared-read launch (module docstring): build_union (device), the shared split kernel, then piece 3's
    combine kernel with piece 3's launch. Shape / dtype checks only (no device read)."""
    if _bi_shared_kernel is None:
        raise RuntimeError("triton is not installed")
    if mode not in MODES:
        raise ValueError("mode %r not in %s" % (mode, MODES))
    R, Hq, D, N, Hkv = base._check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    pps = int(base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split)
    NR = int(rows_max)
    if not 1 <= NR <= ROWS_MAX or pps < 1:
        raise ValueError("rows_max %r (1 .. %d), pages_per_split %r" % (rows_max, ROWS_MAX, pages_per_split))
    if page_key is None or page_key.shape != page_table.shape or page_key.dtype != torch.int32 \
            or not page_key.is_contiguous() or page_key.device != q.device:
        raise ValueError("page_key: contiguous int32 shaped like page_table on q's device")
    group = Hq // Hkv
    bh = base.block_h_for(group)
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("out must be q's shape and dtype with a unit last stride")
    if R == 0:
        return out
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1 or page_table.stride(-1) != 1 or num_pages.stride(-1) != 1:
        raise ValueError("unit last strides are assumed by the kernels")
    whole = mode == "whole"
    if union is None:
        union = build_union(seq_idx, page_table, num_pages, page_key, NR, pps, whole)
    elif union["rows_max"] != NR or union["whole"] != whole or union["pps"] != pps:
        raise ValueError("union built for another rows_max / mode / split size")
    S = base.splits_for(page_table.shape[2], pps)
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
    cnt = count_loads if count_loads is not None else torch.zeros(2, dtype=torch.int32, device=q.device)
    grow, ut, rowpos, win = union["grow"], union["ut"], union["rowpos"], union["win"]
    _bi_shared_kernel[(R, Hkv, win.shape[2])](
        q, k_cache, v_cache, bias_cache, seq_idx, num_pages, last_len, grow, ut, rowpos, win, wacc, wml, cnt,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        bias_cache.stride(0), bias_cache.stride(1), bias_cache.stride(2),
        num_pages.stride(0),
        grow.stride(0), ut.stride(0), ut.stride(1), rowpos.stride(0), rowpos.stride(1), rowpos.stride(2),
        win.stride(0), win.stride(1), win.stride(2),
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wml.stride(0), wml.stride(1), wml.stride(2),
        scale * base.LOG2E,
        GROUP=group, BLOCK_H=bh, D=D, PAGE_C=base.PAGE, PPS=pps, NR=NR, WHOLE=whole, COUNT=count_loads is not None,
        num_warps=base.NUM_WARPS, num_stages=base.NUM_STAGES)
    base._bi_combine_kernel[(R, Hkv)](                 # piece 3's combine, piece 3's launch (bi_paged_attention.py :345)
        wacc, wml, num_pages, out,
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wml.stride(0), wml.stride(1), wml.stride(2),
        num_pages.stride(0),
        out.stride(0), out.stride(1),
        S,
        GROUP=group, BLOCK_H=bh, D=D, PPS=pps, num_warps=base.NUM_WARPS)
    return out


def bi_paged_decode_shared_reference(base, q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                                     page_key, rows_max: int = ROWS_MAX, mode: str = "split",
                                     softmax_scale: Optional[float] = None, pages_per_split: Optional[int] = None,
                                     union: Optional[dict] = None, loads: Optional[list] = None) -> torch.Tensor:
    """The CPU TWIN of the shared kernel: the kernel's schedule in torch, program by program ((group, head, split) or
    (group, head)), union page by union page, EVERY row's step on every union page of the window (a non-member page
    is the identity step, run, not skipped), with piece 3's twin's step formulas (bi_paged_attention.py 7c826c1
    :392-408) and its combine (:410-420). Bitwise equal to piece 3's twin per row is the CPU claim (tests).
    loads: a list [tiles, rows] incremented like the kernel's COUNT counter (a tile = a union page with some row
    active). Host loops: tests only."""
    if mode not in MODES:
        raise ValueError("mode %r not in %s" % (mode, MODES))
    R, Hq, D, N, Hkv = base._check_shapes(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
    pps = int(base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split)
    NR = int(rows_max)
    PAGE = base.PAGE
    whole = mode == "whole"
    un = union if union is not None else build_union(seq_idx, page_table, num_pages, page_key, NR, pps, whole)
    group = Hq // Hkv
    scale_log2 = (float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)) * base.LOG2E
    f32 = torch.float32
    ninf = float("-inf")
    grow, ut, rowpos, win = (un[x].to(torch.int64).cpu().tolist() for x in ("grow", "ut", "rowpos", "win"))
    np_l = num_pages.to(torch.int64).cpu().tolist()
    ll_l = last_len.to(torch.int64).cpu().tolist()
    si_l = seq_idx.to(torch.int64).cpu().tolist()
    parts = {}                                                         # (r, h, k) -> (m, l, acc)
    ar = torch.arange(PAGE)

    def fresh():
        return (torch.full((group,), ninf, dtype=f32), torch.zeros(group, dtype=f32), torch.zeros(group, D, dtype=f32))

    for g in range(R):
        rows = grow[g]
        for h in range(Hkv):
            for sp in range(len(win[g][h])):
                u0, u1 = win[g][h][sp]
                if u1 <= u0:
                    continue
                n = si_l[rows[0]]
                st = [fresh() for _ in range(NR)]
                qf = [q[max(r, 0), h * group:(h + 1) * group].to(f32) for r in rows]
                for u in range(u0, u1):
                    slot = ut[g][h][u]
                    oks, inns, rps = [], [], []
                    for j in range(NR):
                        r, rp = rows[j], rowpos[g][h][j][u]
                        npg = np_l[r][h] if r >= 0 else 0
                        inn = rp >= 0 if whole else (sp * pps <= rp < sp * pps + pps)
                        cut = PAGE if rp < npg - 1 else (ll_l[r] if r >= 0 else 0)
                        oks.append(torch.full((PAGE,), bool(inn)) & (ar < cut))
                        inns.append(inn)
                        rps.append(rp)
                    ld = oks[0].clone()
                    for o in oks[1:]:
                        ld |= o
                    if loads is not None and bool(ld.any()):
                        loads[0] += 1
                        loads[1] += int(ld.sum())
                    s0 = slot * PAGE
                    kt = k_cache[n, s0:s0 + PAGE, h]
                    vt = v_cache[n, s0:s0 + PAGE, h]
                    bt = bias_cache[n, s0:s0 + PAGE, h]
                    for j in range(NR):
                        r, ok = rows[j], oks[j]
                        m, l, acc = st[j]
                        if whole and inns[j] and rps[j] > 0 and rps[j] % pps == 0:
                            parts[(r, h, rps[j] // pps - 1)] = (m, l, acc)
                            m, l, acc = fresh()
                        kp = torch.where(ok.unsqueeze(-1), kt.to(f32), torch.zeros(PAGE, D, dtype=f32))
                        vp = torch.where(ok.unsqueeze(-1), vt, torch.zeros_like(vt))
                        bp = torch.where(ok, bt.to(f32), torch.zeros(PAGE, dtype=f32))
                        s = qf[j] @ kp.t() + bp.unsqueeze(0)
                        s = torch.where(ok.unsqueeze(0), s * scale_log2, torch.full_like(s, ninf))
                        m_new = torch.maximum(m, s.max(dim=1).values)
                        alpha = torch.where(m_new == ninf, torch.ones_like(m_new), torch.exp2(m - m_new))
                        pr = torch.where(ok.unsqueeze(0), torch.exp2(s - m_new.unsqueeze(1)), torch.zeros_like(s))
                        l = l * alpha + pr.sum(dim=1)
                        acc = acc * alpha.unsqueeze(1) + pr.to(vp.dtype).to(f32) @ vp.to(f32)
                        st[j] = (m_new, l, acc)
                for j in range(NR):
                    r = rows[j]
                    if r < 0:
                        continue
                    if whole:
                        parts[(r, h, (np_l[r][h] - 1) // pps)] = st[j]
                    elif sp * pps < np_l[r][h]:
                        parts[(r, h, sp)] = st[j]
    out = torch.empty_like(q)
    for r in range(R):
        for h in range(Hkv):
            ks = range(-(-np_l[r][h] // pps))
            if any((r, h, k) not in parts for k in ks):
                raise RuntimeError("row %d head %d: a split partial is missing (a group did not cover the row)" % (r, h))
            mx = torch.full((group,), ninf, dtype=f32)
            for k in ks:                                                # piece 3's twin's combine: max, in order
                mx = torch.maximum(mx, parts[(r, h, k)][0])
            lsum = torch.zeros(group, dtype=f32)
            tot = torch.zeros(group, D, dtype=f32)
            for k in ks:                                                # then the sums, in order
                m_k, l_k, a_k = parts[(r, h, k)]
                wgt = torch.exp2(m_k - mx)
                lsum = lsum + l_k * wgt
                tot = tot + a_k * wgt.unsqueeze(1)
            denom = torch.where(lsum == 0, torch.ones_like(lsum), lsum)
            out[r, h * group:(h + 1) * group] = (tot / denom.unsqueeze(1)).to(q.dtype)
    return out


class SharedKV:
    """The tick's selector object (ops.attn): piece 3's interface plus page_key / rows_max. Calls without page_key
    (the BI serial reference) or with rows_max == 1 go to piece 3 itself; .stats counts each kind, so a gate can
    assert that the shared kernel ran (a selector that silently falls back would pass every bitwise gate)."""
    needs_page_key = True

    IMPLS = ("kernel", "reference")

    def __init__(self, base, mode: str = "split", rows_max: int = ROWS_MAX, impl: str = "kernel"):
        """impl 'kernel': the Triton kernels (GPU, or the interpreter); 'reference': the torch twins (the CPU tick
        tests): piece 3's bi_paged_decode_reference and bi_paged_decode_shared_reference."""
        if mode not in MODES or impl not in self.IMPLS:
            raise ValueError("mode %r not in %s / impl %r not in %s" % (mode, MODES, impl, self.IMPLS))
        self.base, self.mode, self.rows_max, self.impl = base, mode, int(rows_max), impl
        self.build_page_table = base.build_page_table
        self.PAGE, self.PAGES_PER_SPLIT = base.PAGE, base.PAGES_PER_SPLIT
        self.__file__ = "%s + sharedkv(%s, %s)" % (getattr(base, "__file__", "piece3"), mode, impl)
        self.stats = dict(shared=0, base_no_key=0, base_single_row=0)

    def _base(self, args, softmax_scale, pps, out=None, workspace=None):
        if self.impl == "reference":
            return self.base.bi_paged_decode_reference(*args, softmax_scale=softmax_scale, pages_per_split=pps)
        return self.base.bi_paged_decode(*args, softmax_scale=softmax_scale, out=out, workspace=workspace,
                                         pages_per_split=pps)

    def _shared(self, args, page_key, nr, softmax_scale, pps, out=None, workspace=None, union=None):
        if self.impl == "reference":
            return bi_paged_decode_shared_reference(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                                    softmax_scale=softmax_scale, pages_per_split=pps, union=union)
        return bi_paged_decode_shared(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                      softmax_scale=softmax_scale, out=out, workspace=workspace,
                                      pages_per_split=pps, union=union)

    def bi_paged_decode(self, q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                        softmax_scale=None, out=None, workspace=None, pages_per_split=None, page_key=None,
                        rows_max=None):
        pps = self.base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split
        nr = self.rows_max if rows_max is None else min(int(rows_max), self.rows_max)
        args = (q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
        if page_key is None or nr <= 1:
            self.stats["base_no_key" if page_key is None else "base_single_row"] += 1
            return self._base(args, softmax_scale, pps, out=out, workspace=workspace)
        self.stats["shared"] += 1
        return self._shared(args, page_key, nr, softmax_scale, pps, out=out, workspace=workspace)


class SharedKVProbe(SharedKV):
    """The GPU cell's in-tick A/B (docs/R2_SHAREDKV_PLAN.md): every keyed call runs piece 3 AND the shared kernel on
    the same inputs, RETURNS piece 3's output (so the tick's path and SG1's comparison stay piece 3's), counts on the
    device the rows whose bits differ (int16 / int32 views: a sign-of-zero difference counts), and records CUDA events
    around both launches, alternating which runs first (L2 warm-up bias). No host read inside a call: .report() reads
    everything once at the end."""

    def __init__(self, base, mode: str = "split", rows_max: int = ROWS_MAX, timing: bool = True, plant: bool = False,
                 impl: str = "kernel", count: bool = True, ncu_window=None):
        """[sharedkv2] ncu_window (skip, count) (the S3 cell, benchmarks/Efficiency/r2_sharedkv_ncu.py): among the
        keyed calls with rows_max == self.rows_max, calls skip .. skip + count - 1 are MARKED: one profiler window
        (cudaProfilerStart / Stop; ncu --profile-from-start off profiles nothing else) around the two launches, each
        in its NVTX range "skv|pr|c<j>" (piece 3: split + combine) / "skv|sh|c<j>" (shared + combine), and the call's
        logical page counts (sum of num_pages; the union size) are kept on the device for the byte bounds. The union
        build and the counting launch stay OUTSIDE the window."""
        super().__init__(base, mode, rows_max, impl)
        self.ncu_window = None if ncu_window is None else (int(ncu_window[0]), int(ncu_window[1]))
        self.full_calls = 0                               # keyed calls with rows_max == self.rows_max
        self.marked = []                                  # (j, rows, first, sum pages (device), union pages (device))
        self.timing = bool(timing)
        self.count = bool(count)                          # an extra UNTIMED counting launch per keyed call
        self.tiles = dict(shared=0, per_row=0, rows_shared=0, rows_per_row=0)   # device sums (0 until a count)
        self.plant = bool(plant)                          # the negative control: plant_drop_page on every keyed call
        self.diff_rows = None
        self.rows_cmp = 0
        self.events = []                                  # (rows_max, first, e0, e1, e2) per keyed call
        self.calls = 0

    def bi_paged_decode(self, q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                        softmax_scale=None, out=None, workspace=None, pages_per_split=None, page_key=None,
                        rows_max=None):
        nr = self.rows_max if rows_max is None else min(int(rows_max), self.rows_max)
        if page_key is None or nr <= 1:
            return super().bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                                           softmax_scale=softmax_scale, out=out, workspace=workspace,
                                           pages_per_split=pages_per_split, page_key=page_key, rows_max=rows_max)
        self.stats["shared"] += 1
        pps = self.base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split
        args = (q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if self.timing and q.is_cuda else None
        if ev:
            ev[3].record()
        un = build_union(seq_idx, page_table, num_pages, page_key, nr, pps, self.mode == "whole")
        if ev:
            ev[4].record()                                # the union build, timed on its own
        if self.plant:
            un = plant_drop_page(un)

        mark = self._ncu_mark(nr)                         # [sharedkv2] None, or the marked call's index j
        prof = mark is not None and q.is_cuda             # the CPU tests mark calls without a profiler / NVTX

        def ranged(tag, fn):
            if not prof:
                return fn()
            torch.cuda.nvtx.range_push("skv|%s|c%d" % (tag, mark))
            try:
                return fn()
            finally:
                torch.cuda.nvtx.range_pop()

        def per_row():
            return ranged("pr", lambda: self._base(args, softmax_scale, pps))

        def shared():
            return ranged("sh", lambda: self._shared(args, page_key, nr, softmax_scale, pps, union=un))
        first = "per_row" if self.calls % 2 == 0 else "shared"
        self.calls += 1
        if mark is not None:
            self.marked.append((mark, int(q.shape[0]), first, num_pages.to(torch.int64).sum(),
                                un["ucnt"].to(torch.int64).sum()))
        if prof:
            torch.cuda.cudart().cudaProfilerStart()
        if ev:
            ev[0].record()
        a = per_row() if first == "per_row" else shared()
        if ev:
            ev[1].record()
        b = shared() if first == "per_row" else per_row()
        if ev:
            ev[2].record()
            self.events.append((nr, first, ev[0], ev[1], ev[2], ev[3], ev[4]))
        if prof:
            torch.cuda.cudart().cudaProfilerStop()
        if self.count and self.impl == "kernel":        # untimed: the kernel's own tile counter (the byte model)
            cnt = torch.zeros(2, dtype=torch.int32, device=q.device)
            bi_paged_decode_shared(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                   softmax_scale=softmax_scale, pages_per_split=pps, union=un, count_loads=cnt)
            npg = num_pages.to(torch.int64)
            self.tiles["shared"] = self.tiles["shared"] + cnt[0].to(torch.int64)
            self.tiles["per_row"] = self.tiles["per_row"] + npg.sum()
            self.tiles["rows_shared"] = self.tiles["rows_shared"] + cnt[1].to(torch.int64)
            self.tiles["rows_per_row"] = self.tiles["rows_per_row"] + ((npg - 1) * self.base.PAGE).sum() \
                + (last_len.to(torch.int64) * num_pages.shape[1]).sum()
        ref, sh = (a, b) if first == "per_row" else (b, a)
        iv = torch.int16 if ref.element_size() == 2 else torch.int32
        d = (ref.view(iv) != sh.view(iv)).flatten(1).any(1).sum()
        self.diff_rows = d if self.diff_rows is None else self.diff_rows + d
        self.rows_cmp += int(q.shape[0])
        if out is not None:
            out.copy_(ref)
            return out
        return ref

    def _ncu_mark(self, nr: int):
        """[sharedkv2] the index j of this keyed call if it lies in ncu_window (counted over the calls with
        nr == rows_max), else None."""
        if self.ncu_window is None or nr < self.rows_max:
            return None
        j = self.full_calls
        self.full_calls += 1
        skip, count = self.ncu_window
        return j if skip <= j < skip + count else None

    def ncu_report(self) -> dict:
        """[sharedkv2] the marked calls (ONE host read): per call j its row count, which kernel ran first, the sum of
        num_pages (the per-row reads: (row, head) pages) and the union size (the distinct (key, slot) pages per
        (group, head)); a (row, head) page is 64 rows of K and V (bf16, D 128): 32768 B."""
        out = []
        for (j, R, first, s, u) in self.marked:
            out.append(dict(call=j, rows=R, first=first, sum_pages=int(s), union_pages=int(u)))
        return dict(window=self.ncu_window, full_calls=self.full_calls, marked=out)

    def report(self) -> dict:
        """ONE host read at the end: the differing-row count and the per-call times (ms) of both kernels."""
        if self.events:
            torch.cuda.synchronize()
        t_pr, t_sh, t_b = [], [], []
        for nr, first, e0, e1, e2, b0, b1 in self.events:
            x, y = e0.elapsed_time(e1), e1.elapsed_time(e2)
            pr, sh = (x, y) if first == "per_row" else (y, x)
            t_pr.append((nr, first, pr))
            t_sh.append((nr, first, sh))
            t_b.append((nr, first, b0.elapsed_time(b1)))
        tiles = {k: int(v) for k, v in self.tiles.items()}
        return dict(stats=dict(self.stats), keyed_calls=self.calls, rows_compared=self.rows_cmp, plant=self.plant,
                    diff_rows=int(self.diff_rows) if self.diff_rows is not None else 0,
                    per_row_ms=t_pr, shared_ms=t_sh, build_ms=t_b, tiles=tiles,
                    tile_ratio=tiles["shared"] / tiles["per_row"] if tiles["per_row"] else None,
                    row_ratio=tiles["rows_shared"] / tiles["rows_per_row"] if tiles["rows_per_row"] else None,
                    mode=self.mode)
