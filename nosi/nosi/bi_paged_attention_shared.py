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
    only when a row's acc entry is -0.0 at that moment (an underflow of a negative acc times a tiny alpha). It goes
    BOTH ways (review MINOR-1): the shared kernel runs more identity steps in the middle of a split, and it can run
    FEWER trailing ones than piece 3 (piece 3 runs PPS - n_last steps past a row's last page; the shared kernel only
    the union pages after it inside the window), so piece 3 can turn a -0.0 into +0.0 where the shared kernel keeps
    -0.0. The bitwise gate counts either as a difference; the probe's report splits sign-of-zero-only elements from
    value differences (F1's triage).
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
same strides). [integ] The X pool's flat address mode (docs/R2_XPOOL_PLAN.md section 3) needs NO change here: it is
a convention of the arguments (stride(0) == 0 views, a page table of absolute positions), and the kernel takes its
strides from the tensors, so n * s_kn adds 0 (docs/R2_INTEG_PLAN.md section 2.3; CPU check ST5, not run).

INTERFACE:
    SharedKV(base, mode="split", rows_max=ROWS_MAX)    base = the loaded piece 3 module (pinned blob)
      .bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, softmax_scale=None,
                       out=None, workspace=None, pages_per_split=PPS, page_key=None, rows_max=None,
                       group_token=None)      [sharedkv3] group_token: the tick's per-tick object (the group cache key)
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
UNION_FIELDS = ("grow", "ut", "ucnt", "rowpos", "win", "uidx")   # [sharedkv4] the tensor fields of a union dict
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


# ---------------------------------------------------------------------------------------------------------------------
# [sharedkv3] the union build in TWO parts (docs/R2_SHAREDKV3_UNION.md): the GROUPS once per tick (they depend on
# seq_idx only: build_groups, cached by the selector per TICK TOKEN + seq_idx object + rows_max; [sharedkv4] an
# in-place change of seq_idx under one token is NOT detected: no producer does it), and the per-layer UNION in ONE
# Triton kernel (_union_kernel via build_union_fused) instead of about 50 torch ops. Same output as build_union (int
# tensors): the union index of an entry is the number of DISTINCT (key, slot) values below it, counted over the
# first occurrences (an entry whose value no earlier row of the group lists); ascending order and dedup follow.
# ---------------------------------------------------------------------------------------------------------------------
def build_groups(seq_idx: torch.Tensor, rows_max: int) -> torch.Tensor:
    """grow (R, NR) int32 contiguous: build_union's group step, the same ops in the same order (device, no host read).
    It depends on seq_idx and rows_max only: the same for every layer of a tick."""
    R = seq_idx.shape[0]
    NR = int(rows_max)
    dev = seq_idx.device
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
    return grow.to(torch.int32).contiguous()


def _windows(uidx: torch.Tensor, ucnt: torch.Tensor, pps: int, whole: bool) -> torch.Tensor:
    """build_union's window step on (uidx, ucnt), the same ops (the torch twin of the fused build uses it)."""
    R, Hkv, NR, P = uidx.shape
    if whole:
        return torch.stack([torch.zeros_like(ucnt), ucnt], -1).view(R, Hkv, 1, 2)
    S = -(-P // int(pps))
    pad = S * int(pps) - P
    ux = torch.nn.functional.pad(uidx, (0, pad), value=-1).view(R, Hkv, NR, S, int(pps))
    v = ux >= 0
    lo = torch.where(v, ux, torch.full_like(ux, _BIG)).amin(dim=(2, 4))
    hi = torch.where(v, ux, torch.full_like(ux, -1)).amax(dim=(2, 4)) + 1
    lo = torch.where(hi > 0, lo, torch.zeros_like(lo))
    return torch.stack([lo, hi], -1)


def build_union_fused_reference(seq_idx, page_table, num_pages, page_key, rows_max: int, pps: int, whole: bool = False,
                                groups: Optional[torch.Tensor] = None, dedup: bool = True) -> dict:
    """The torch TWIN of _union_kernel's algorithm (CPU tests, the CPU tick): per (group, head) the entries
    v = key << 32 | slot of its rows; FIRST = listed and no earlier row of the group lists v; uid = the count of FIRST
    entries with a smaller v; ut[uid] = slot (FIRST entries), rowpos[j, uid] = i, uidx = uid. dedup False is the
    NEGATIVE CONTROL (every listed entry counts as first: duplicates inflate ucnt and shift the uids)."""
    NR = int(rows_max)
    R, Hkv, P = page_table.shape
    i64 = torch.int64
    dev = page_table.device
    grow = (groups if groups is not None else build_groups(seq_idx, NR)).to(i64)
    N = NR * P
    U = N
    gv = grow.clamp(min=0)
    live = grow >= 0
    pos = torch.arange(P, device=dev, dtype=i64)
    npg = num_pages.to(i64)[gv]                                                     # (R, NR, Hkv)
    ok = live.view(R, NR, 1, 1) & (pos.view(1, 1, 1, P) < npg.unsqueeze(-1))         # (R, NR, Hkv, P)
    v = (page_key.to(i64)[gv] << 32) | page_table.to(i64)[gv]
    ok = ok.permute(0, 2, 1, 3).reshape(R, Hkv, N)
    v = torch.where(ok, v.permute(0, 2, 1, 3).reshape(R, Hkv, N), torch.full((R, Hkv, N), -1, dtype=i64, device=dev))
    jj = (torch.arange(N, device=dev) // P).view(1, 1, N)
    ii = (torch.arange(N, device=dev) % P).view(1, 1, N)
    if dedup:
        earlier = (v.unsqueeze(-1) == v.unsqueeze(-2)) & ok.unsqueeze(-1) & (jj.view(1, 1, N, 1) < jj.view(1, 1, 1, N))
        first = ok & ~earlier.any(dim=-2)                                            # [e, f]: e earlier than f
    else:
        first = ok.clone()
    uid = ((v.unsqueeze(-2) < v.unsqueeze(-1)) & first.unsqueeze(-2)).to(i64).sum(-1)   # [e, f]: v_f < v_e
    ucnt = first.to(i64).sum(-1)
    ut = torch.full((R, Hkv, U + 1), -1, dtype=i64, device=dev)
    ut.scatter_(-1, torch.where(first, uid, torch.full_like(uid, U)),
                torch.where(first, v & 0xFFFFFFFF, torch.full_like(v, -1)))
    rowpos = torch.full((R, Hkv, NR * (U + 1)), -1, dtype=i64, device=dev)
    rowpos.scatter_(-1, jj.expand(R, Hkv, N) * (U + 1) + torch.where(ok, uid, torch.full_like(uid, U)),
                    torch.where(ok, ii.expand(R, Hkv, N), torch.full_like(uid, -1)))
    rowpos = rowpos.view(R, Hkv, NR, U + 1)[..., :U]
    uidx = torch.where(ok, uid, torch.full_like(uid, -1)).view(R, Hkv, NR, P)
    win = _windows(uidx, ucnt, pps, whole)
    c = lambda t: t.to(torch.int32).contiguous()                                    # noqa: E731
    return dict(grow=c(grow), ut=c(ut[..., :U]), ucnt=c(ucnt), rowpos=c(rowpos), win=c(win), uidx=c(uidx),
                rows_max=NR, whole=whole is True, pps=int(pps))


def build_union_fused(seq_idx, page_table, num_pages, page_key, rows_max: int, pps: int, whole: bool = False,
                      groups: Optional[torch.Tensor] = None, dedup: bool = True, te: int = 64) -> dict:
    """[sharedkv3] build_union's dict from ONE kernel launch per call (plus build_groups unless `groups` is given:
    the selector passes its per-tick cache). Outputs are allocated with torch.empty (no launch) and written whole by
    the kernel. Shape / dtype / contiguity checks only (no device read)."""
    if _union_kernel is None:
        raise RuntimeError("triton is not installed")
    NR = int(rows_max)
    R, Hkv, P = page_table.shape
    if page_key.shape != page_table.shape or page_key.dtype != torch.int32 or page_table.dtype != torch.int32 \
            or not (page_key.is_contiguous() and page_table.is_contiguous() and num_pages.is_contiguous()) \
            or num_pages.dtype != torch.int32:
        raise ValueError("page_table / page_key: contiguous int32 of one shape; num_pages contiguous int32 (L28: the "
                         "kernel reads the keys with the table's strides)")
    grow = groups if groups is not None else build_groups(seq_idx, NR)
    if grow.shape != (R, NR) or grow.dtype != torch.int32 or not grow.is_contiguous():
        raise ValueError("groups must be build_groups(seq_idx, rows_max): (R, NR) contiguous int32")
    N = NR * P
    U = N
    S = 1 if whole else -(-P // int(pps))
    dev = page_table.device
    e32 = dict(dtype=torch.int32, device=dev)
    ut = torch.empty((R, Hkv, U), **e32)
    ucnt = torch.empty((R, Hkv), **e32)
    rowpos = torch.empty((R, Hkv, NR, U), **e32)
    win = torch.empty((R, Hkv, S, 2), **e32)
    uidx = torch.empty((R, Hkv, NR, P), **e32)
    first = torch.empty((R, Hkv, N), **e32)
    if R > 0:
        _union_kernel[(R, Hkv)](
            page_table, num_pages, page_key, grow, ut, ucnt, rowpos, win, uidx, first,
            page_table.stride(0), page_table.stride(1), num_pages.stride(0), grow.stride(0),
            ut.stride(0), ut.stride(1), ucnt.stride(0), rowpos.stride(0), rowpos.stride(1), rowpos.stride(2),
            win.stride(0), win.stride(1), win.stride(2), uidx.stride(0), uidx.stride(1), uidx.stride(2),
            first.stride(0), first.stride(1),
            NR=NR, P=P, N=N, PPS=int(pps), S=S, TE=int(te), WHOLE=whole is True, DEDUP=bool(dedup))
    return dict(grow=grow, ut=ut, ucnt=ucnt, rowpos=rowpos, win=win, uidx=uidx, rows_max=NR, whole=whole is True,
                pps=int(pps))


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


if triton is not None:      # [sharedkv3] the fused union build (a second triton block, after the shared kernel's)
    @triton.jit
    def _union_entries(PT, NP, KEY, GROW, g, h, e, s_ptr, s_pth, s_npr, s_gr, N: tl.constexpr, P: tl.constexpr):
        """Entries e (a vector of flat indices j * P + i) of group g, head h: (v = key << 32 | slot, listed, j, i)."""
        j = e // P
        i = e % P
        inn = e < N
        r = tl.load(GROW + g * s_gr + j, mask=inn, other=-1)
        live = r >= 0
        rc = tl.maximum(r, 0)
        npg = tl.load(NP + rc * s_npr + h, mask=inn & live, other=0)
        ok = inn & live & (i < npg)
        slot = tl.load(PT + rc * s_ptr + h * s_pth + i, mask=ok, other=0).to(tl.int64)
        key = tl.load(KEY + rc * s_ptr + h * s_pth + i, mask=ok, other=0).to(tl.int64)
        v = tl.where(ok, (key << 32) | slot, -1)
        return v, ok, j, i

    @triton.jit
    def _union_kernel(PT, NP, KEY, GROW, UT, UCNT, ROWPOS, WIN, UIDX, FIRST,
                      s_ptr, s_pth, s_npr, s_gr, s_utr, s_uth, s_ucr, s_rpr, s_rph, s_rpj, s_wr, s_wh, s_ws,
                      s_uxr, s_uxh, s_uxj, s_fr, s_fh,
                      NR: tl.constexpr, P: tl.constexpr, N: tl.constexpr, PPS: tl.constexpr, S: tl.constexpr,
                      TE: tl.constexpr, WHOLE: tl.constexpr, DEDUP: tl.constexpr):
        """[sharedkv3] ONE program per (group g, KV head h): build_union's per-layer outputs for (g, h) (the twin is
        build_union_fused_reference). Pass 0 fills ut / rowpos / uidx with -1 and ucnt / win with 0 (the outputs of
        an EMPTY group, final there); [sharedkv4, review MINOR-2] a program whose group has no row (GROW[g, 0] < 0,
        the grid rows past the last group) stops after pass 0. Otherwise pass A marks the FIRST entries (listed, and
        no earlier row of the group lists the same value) into the scratch FIRST; pass B gives each listed entry its
        uid (the count of FIRST entries with a smaller value), writes uidx, ut[uid] (FIRST entries), rowpos[j, uid]
        and ucnt; pass C (split mode) reads uidx back and writes the split windows (min / max uid over the rows' list
        positions of split k, by unrolled compare-selects). Barriers order the passes (global writes of the CTA are
        visible after the barrier); the branch on GROW[g, 0] is uniform over the CTA, so every thread reaches the
        same barriers. PRECONDITION (review MINOR-3): a row lists each (key, slot) at most once (keys strictly
        ascending per row, the tick's selection check); build_union dedups a repeat inside one row, this kernel does
        not. DEDUP False is the NEGATIVE CONTROL."""
        g = tl.program_id(0)
        h = tl.program_id(1)
        ar = tl.arange(0, TE)
        neg = tl.full([TE], -1, tl.int32)
        for u0 in range(0, N, TE):                                   # pass 0: the empty group's outputs
            uu = u0 + ar
            mu = uu < N
            tl.store(UT + g * s_utr + h * s_uth + uu, neg, mask=mu)
            tl.store(UIDX + g * s_uxr + h * s_uxh + (uu // P) * s_uxj + (uu % P), neg, mask=mu)
            for j in tl.static_range(NR):
                tl.store(ROWPOS + g * s_rpr + h * s_rph + j * s_rpj + uu, neg, mask=mu)
        tl.store(UCNT + g * s_ucr + h, g * 0)
        for k in range(0, S):
            tl.store(WIN + g * s_wr + h * s_wh + k * s_ws, g * 0)
            tl.store(WIN + g * s_wr + h * s_wh + k * s_ws + 1, g * 0)
        r0 = tl.load(GROW + g * s_gr)
        if r0 >= 0:                                                  # a live group (uniform over the CTA)
            tl.debug_barrier()
            for f0 in range(0, N, TE):                               # pass A: the first occurrences
                f = f0 + ar
                vf, okf, jf, i_f = _union_entries(PT, NP, KEY, GROW, g, h, f, s_ptr, s_pth, s_npr, s_gr, N, P)
                dup = tl.zeros([TE], dtype=tl.int32)
                if DEDUP:
                    for e0 in range(0, N, TE):
                        e = e0 + ar
                        ve, oke, je, ie = _union_entries(PT, NP, KEY, GROW, g, h, e, s_ptr, s_pth, s_npr, s_gr, N,
                                                         P)
                        hit = (ve[:, None] == vf[None, :]) & oke[:, None] & (je[:, None] < jf[None, :])
                        dup = dup + tl.sum(hit.to(tl.int32), 0)
                first = okf & (dup == 0)
                tl.store(FIRST + g * s_fr + h * s_fh + f, first.to(tl.int32), mask=f < N)
            tl.debug_barrier()
            cnt = tl.zeros([TE], dtype=tl.int32)
            for e0 in range(0, N, TE):                               # pass B: uid, uidx, ut, rowpos, ucnt
                e = e0 + ar
                ve, oke, je, ie = _union_entries(PT, NP, KEY, GROW, g, h, e, s_ptr, s_pth, s_npr, s_gr, N, P)
                uid = tl.zeros([TE], dtype=tl.int32)
                for f0 in range(0, N, TE):
                    f = f0 + ar
                    vf, okf, jf, i_f = _union_entries(PT, NP, KEY, GROW, g, h, f, s_ptr, s_pth, s_npr, s_gr, N, P)
                    fst = tl.load(FIRST + g * s_fr + h * s_fh + f, mask=f < N, other=0)
                    lt = (vf[None, :] < ve[:, None]) & (fst[None, :] != 0)
                    uid = uid + tl.sum(lt.to(tl.int32), 1)
                fe = tl.load(FIRST + g * s_fr + h * s_fh + e, mask=e < N, other=0)
                cnt = cnt + fe
                me = e < N
                tl.store(UIDX + g * s_uxr + h * s_uxh + je * s_uxj + ie, tl.where(oke, uid, -1), mask=me)
                tl.store(UT + g * s_utr + h * s_uth + uid, (ve & 0xFFFFFFFF).to(tl.int32), mask=me & (fe != 0))
                tl.store(ROWPOS + g * s_rpr + h * s_rph + je * s_rpj + uid, ie.to(tl.int32), mask=me & oke)
            total = tl.sum(cnt, 0)
            tl.store(UCNT + g * s_ucr + h, total)
            if WHOLE:
                tl.store(WIN + g * s_wr + h * s_wh, 0)
                tl.store(WIN + g * s_wr + h * s_wh + 1, total)
            else:
                # [sharedkv3 fix, CPU job 2191241] pass C: the split windows from the STORED uidx, as build_union
                # does (pad, view (NR, S, PPS), amin / amax), with NO reduction primitive: unrolled scalar loads and
                # compare-selects. a360d86 accumulated an [TE, SP] masked tl.min / tl.max across the pass-B tile
                # loop and gave other windows under the interpreter (the repro: test_r2_sharedkv4.py, V1-V4).
                tl.debug_barrier()
                for k in range(0, S):
                    lo_k = g * 0 + 2147483647                            # int32 scalars (program_id arithmetic)
                    hi_k = g * 0
                    for j in tl.static_range(NR):
                        for i in tl.static_range(PPS):
                            pos = k * PPS + i
                            x = tl.load(UIDX + g * s_uxr + h * s_uxh + j * s_uxj + pos, mask=pos < P, other=-1)
                            lo_k = tl.where((x >= 0) & (x < lo_k), x, lo_k)
                            hi_k = tl.where(x + 1 > hi_k, x + 1, hi_k)
                    lo_k = tl.where(hi_k > 0, lo_k, 0)
                    tl.store(WIN + g * s_wr + h * s_wh + k * s_ws, lo_k)
                    tl.store(WIN + g * s_wr + h * s_wh + k * s_ws + 1, hi_k)
else:                                                  # pragma: no cover
    _union_kernel = None


_DUMMY_CNT = {}


def _dummy_counter(device) -> torch.Tensor:
    """[review MAJOR-3] the COUNT pointer of an uncounted launch: one persistent (2,) int32 buffer per device (the
    kernel never touches it with COUNT off), so the production launch has no torch.zeros launch of its own."""
    key = str(device)
    if key not in _DUMMY_CNT:
        _DUMMY_CNT[key] = torch.zeros(2, dtype=torch.int32, device=device)
    return _DUMMY_CNT[key]


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
    for t, name in ((k_cache, "k_cache"), (v_cache, "v_cache"), (bias_cache, "bias_cache")):   # [integ3] L28 (m5):
        inner = 1                                     # the storage layouts the kernel's address arithmetic assumes:
        for s in t.shape[1:]:                         # each storage entry dense ((rows, H[, D]) row-major) and
            inner *= int(s)                           # stride(0) 0 (the xpool flat mode) or the entry's size
        dense = [1] * (t.dim() - 1)
        for i in range(t.dim() - 3, -1, -1):          # row-major strides of dims 1 .. n-1
            dense[i] = dense[i + 1] * int(t.shape[i + 2])
        if list(t.stride()[1:]) != dense or t.stride(0) not in (0, inner):
            raise ValueError("%s strides %s: a dense (rows, H[, D]) entry with stride(0) 0 (xpool) or %d expected"
                             % (name, tuple(t.stride()), inner))
    whole = mode == "whole"
    if union is None:
        union = build_union(seq_idx, page_table, num_pages, page_key, NR, pps, whole)
    elif union["rows_max"] != NR or union["whole"] != whole or union["pps"] != pps:
        raise ValueError("union built for another rows_max / mode / split size")
    for f in ("grow", "ut", "rowpos", "win"):          # [sharedkv4, review MINOR-9, L28] the kernel's unit strides
        t = union[f]
        if t.dtype != torch.int32 or not t.is_contiguous() or t.device != q.device:
            raise ValueError("union[%r] must be a contiguous int32 tensor on q's device (the kernel reads it with "
                             "unit last strides)" % f)
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
    cnt = count_loads if count_loads is not None else _dummy_counter(q.device)   # [review MAJOR-3] no extra launch
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
    UNIONS = ("torch", "fused")

    def __init__(self, base, mode: str = "split", rows_max: int = ROWS_MAX, impl: str = "kernel", union: str = "torch"):
        """impl 'kernel': the Triton kernels (GPU, or the interpreter); 'reference': the torch twins (the CPU tick
        tests): piece 3's bi_paged_decode_reference and bi_paged_decode_shared_reference.
        [sharedkv3] union 'torch' (default: build_union, about 50 ops per call) or 'fused': the groups once per tick
        (build_groups, cached per tick token + seq_idx object + rows_max) and the union in ONE kernel per call
        (build_union_fused; with impl 'reference' its torch twin build_union_fused_reference)."""
        if mode not in MODES or impl not in self.IMPLS or union not in self.UNIONS:
            raise ValueError("mode %r not in %s / impl %r not in %s / union %r not in %s"
                             % (mode, MODES, impl, self.IMPLS, union, self.UNIONS))
        self.base, self.mode, self.rows_max, self.impl = base, mode, int(rows_max), impl
        self.union = union
        self._gcache = None                               # (tick token, seq_idx object, rows_max, grow)
        self._token = None                                # the current call's group_token (set per call)
        self.group_builds = 0                             # build_groups calls (tests: once per tick, not per layer)
        self.union_builds = dict(torch=0, fused=0)        # [sharedkv4, review MINOR-5] union builds per kind
        self.build_page_table = base.build_page_table
        self.PAGE, self.PAGES_PER_SPLIT = base.PAGE, base.PAGES_PER_SPLIT
        self.__file__ = "%s + sharedkv(%s, %s)" % (getattr(base, "__file__", "piece3"), mode, impl)
        self.stats = dict(shared=0, base_no_key=0, base_single_row=0)
        self.by_rows_max = {}                             # [review MINOR-3] shared calls per rows_max (host ints)
        self.count_saved = False                          # [integ3] opt-in: pages_saved, a device counter (no read)
        self.pages_saved = None                           # sum over shared calls of (row pages - union pages)

    def _base(self, args, softmax_scale, pps, out=None, workspace=None):
        if self.impl == "reference":
            return self.base.bi_paged_decode_reference(*args, softmax_scale=softmax_scale, pages_per_split=pps)
        return self.base.bi_paged_decode(*args, softmax_scale=softmax_scale, out=out, workspace=workspace,
                                         pages_per_split=pps)

    def _groups(self, seq_idx, nr):
        """[sharedkv3] build_groups once per tick. The cache key is the caller's TICK TOKEN (group_token: the tick
        passes its per-tick context object, the same object to every layer of the tick and a new one each tick),
        plus the seq_idx object and rows_max; the cache holds references to both objects, so neither identity can be
        reused by another object while cached. No token (None): rebuilt on every call (correct, not cached).
        (The first version keyed on the seq_idx version counter; inference tensors have none: CPU job 2191241.)"""
        tok = self._token
        c = self._gcache
        if tok is not None and c is not None and c[0] is tok and c[1] is seq_idx and c[2] == nr:
            return c[3]
        grow = build_groups(seq_idx, nr)
        self.group_builds += 1
        self._gcache = (tok, seq_idx, nr, grow) if tok is not None else None
        return grow

    def _build(self, args, page_key, nr, pps):
        """The union of one call by the selected build ('torch': build_union; 'fused': the cached groups + the fused
        kernel, or its torch twin with impl 'reference')."""
        seq_idx, page_table, num_pages = args[4], args[5], args[6]
        whole = self.mode == "whole"
        self.union_builds[self.union] += 1                # [sharedkv4, review MINOR-5] which build ran (the gate)
        if self.union == "torch":
            return build_union(seq_idx, page_table, num_pages, page_key, nr, pps, whole)
        grow = self._groups(seq_idx, nr)
        if self.impl == "reference":
            return build_union_fused_reference(seq_idx, page_table, num_pages, page_key, nr, pps, whole, groups=grow)
        return build_union_fused(seq_idx, page_table, num_pages, page_key, nr, pps, whole, groups=grow)

    def _shared(self, args, page_key, nr, softmax_scale, pps, out=None, workspace=None, union=None):
        if union is None:                                 # [sharedkv4] always the selector's build (counted)
            union = self._build(args, page_key, nr, pps)
        if self.count_saved:                              # [integ3] the union read each shared page once: the row
            d = args[6].to(torch.int64).sum() - union["ucnt"].to(torch.int64).sum()   # pages it saved (device ops)
            self.pages_saved = d if self.pages_saved is None else self.pages_saved + d
        if self.impl == "reference":
            return bi_paged_decode_shared_reference(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                                    softmax_scale=softmax_scale, pages_per_split=pps, union=union)
        return bi_paged_decode_shared(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                      softmax_scale=softmax_scale, out=out, workspace=workspace,
                                      pages_per_split=pps, union=union)

    def bi_paged_decode(self, q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                        softmax_scale=None, out=None, workspace=None, pages_per_split=None, page_key=None,
                        rows_max=None, group_token=None):
        self._token = group_token                         # [sharedkv3] the tick's per-tick token (or None)
        pps = self.base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split
        nr = self.rows_max if rows_max is None else min(int(rows_max), self.rows_max)
        args = (q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)
        if page_key is None or nr <= 1:
            self.stats["base_no_key" if page_key is None else "base_single_row"] += 1
            return self._base(args, softmax_scale, pps, out=out, workspace=workspace)
        self.stats["shared"] += 1
        self.by_rows_max[nr] = self.by_rows_max.get(nr, 0) + 1
        return self._shared(args, page_key, nr, softmax_scale, pps, out=out, workspace=workspace)


class SharedKVProbe(SharedKV):
    """The GPU cell's in-tick A/B (docs/R2_SHAREDKV_PLAN.md): every keyed call runs piece 3 AND the shared kernel on
    the same inputs, RETURNS piece 3's output (so the tick's path and SG1's comparison stay piece 3's), counts on the
    device the rows whose bits differ (int16 / int32 views: a sign-of-zero difference counts), and records CUDA events
    around both launches, alternating which runs first (L2 warm-up bias). No host read inside a call: .report() reads
    everything once at the end."""

    def __init__(self, base, mode: str = "split", rows_max: int = ROWS_MAX, timing: bool = True, plant: bool = False,
                 impl: str = "kernel", count: bool = True, ncu_window=None, sleep_cycles: int = 2000000,
                 record_tables: int = 4, union: str = "torch"):
        """[sharedkv2] ncu_window (skip, count) (the S3 cell, benchmarks/Efficiency/r2_sharedkv_ncu.py): among the
        keyed calls with rows_max == self.rows_max, calls skip .. skip + count - 1 are MARKED: one profiler window
        (cudaProfilerStart / Stop; ncu --profile-from-start off profiles nothing else) around the two sides, each in
        its NVTX range: "skv|pr|c<j>" (piece 3: split + combine), "skv|bd|c<j>" (the union build: torch ops) and
        "skv|sh|c<j>" (shared + combine); the call's logical page counts (sum of num_pages; the union size) are kept
        on the device for the byte bounds. The counting launch stays OUTSIDE the window.
        sleep_cycles (review MAJOR-3): a torch.cuda._sleep before the timed pair, so the CUDA events time the device
        work and not the Python launch path (0: off). The shared side's span includes the union build (MAJOR-2).
        record_tables (MAJOR-1): the inputs of the first N keyed calls with rows_max == self.rows_max are kept (device
        clones) for F4's offline check against the CPU oracle (report_tables)."""
        super().__init__(base, mode, rows_max, impl, union=union)
        self.sleep_cycles = int(sleep_cycles)
        self.record_tables = int(record_tables)
        self.tables = []                                  # the recorded inputs (MAJOR-1)
        self.mixes = []                                   # per keyed call: (call, nr, mix (3, 5), sum pages, tiles)
        self.diff_zero = 0                                # differing elements whose floats are equal (+-0.0 only)
        self.diff_value = 0                               # differing elements with another value
        self.samples = []                                 # per keyed call: (call, (flag, flat index, bits ref, bits sh))
        self.union_diff = {f: 0 for f in UNION_FIELDS}    # [sharedkv4, MAJOR-1] differing elements fused vs torch
        self.union_checked = 0                            # keyed calls compared
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
                        rows_max=None, group_token=None):
        nr = self.rows_max if rows_max is None else min(int(rows_max), self.rows_max)
        if page_key is None or nr <= 1:
            return super().bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len,
                                           softmax_scale=softmax_scale, out=out, workspace=workspace,
                                           pages_per_split=pages_per_split, page_key=page_key, rows_max=rows_max,
                                           group_token=group_token)
        self._token = group_token                         # [sharedkv3] the tick's per-tick token (or None)
        self.stats["shared"] += 1
        pps = self.base.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split
        args = (q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len)

        whole = self.mode == "whole"
        call = self.calls
        self.calls += 1
        self.by_rows_max[nr] = self.by_rows_max.get(nr, 0) + 1          # host ints (review MINOR-3: F3's check)
        mark = self._ncu_mark(nr)                         # [sharedkv2] None, or the marked call's index j
        prof = mark is not None and q.is_cuda             # the CPU tests mark calls without a profiler / NVTX
        box = {}

        def ranged(tag, fn):
            if not prof:
                return fn()
            torch.cuda.nvtx.range_push("skv|%s|c%d" % (tag, mark))
            try:
                return fn()
            finally:
                torch.cuda.nvtx.range_pop()

        def build():                                      # [sharedkv3] the selected build (torch / fused)
            un = self._build(args, page_key, nr, pps)
            box["raw"] = un                               # [sharedkv4] the unplanted union (MAJOR-1's comparison)
            return plant_drop_page(un) if self.plant else un

        def per_row():
            return ranged("pr", lambda: self._base(args, softmax_scale, pps))

        def shared():                                     # [review MAJOR-2] the union build is part of the shared side
            box["un"] = ranged("bd", build)
            return ranged("sh", lambda: self._shared(args, page_key, nr, softmax_scale, pps, union=box["un"]))
        first = "per_row" if call % 2 == 0 else "shared"
        timed = self.timing and q.is_cuda
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if timed else None
        if timed and self.sleep_cycles > 0:               # [review MAJOR-3] pre-queue the GPU: while it sleeps the host
            torch.cuda._sleep(self.sleep_cycles)          # enqueues both sides, so the events time the device, not the
        if prof:                                          # Python launch path (outside the profiler window)
            torch.cuda.cudart().cudaProfilerStart()
        if ev:
            ev[0].record()
        a = per_row() if first == "per_row" else shared()
        if ev:
            ev[1].record()
        b = shared() if first == "per_row" else per_row()
        if ev:
            ev[2].record()
            self.events.append((nr, first, ev[0], ev[1], ev[2]))
        if prof:
            torch.cuda.cudart().cudaProfilerStop()
        un = box["un"]
        if self.union == "fused":                         # [sharedkv4, review MAJOR-1] untimed, outside the window:
            self._check_union(box["raw"], seq_idx, page_table, num_pages, page_key, nr, pps)   # fused = build_union
        if mark is not None:
            self.marked.append((mark, int(q.shape[0]), first, num_pages.to(torch.int64).sum(),
                                un["ucnt"].to(torch.int64).sum()))
        npg = num_pages.to(torch.int64)
        # [review MAJOR-1] the call's group mix on the device: per group size s = 1 .. ROWS_MAX, the groups, their
        # summed per-row pages (both heads) and their union pages; the expected tile ratio is derived from this mix
        grow = un["grow"].to(torch.int64)
        live = grow >= 0
        gsz = live.sum(1)                                                       # (R,) 0 = no group
        gsum = torch.where(live, npg.sum(1)[grow.clamp(min=0)], torch.zeros_like(grow)).sum(1)
        guni = un["ucnt"].to(torch.int64).sum(1)
        mix = torch.zeros((3, ROWS_MAX + 1), dtype=torch.int64, device=q.device)
        mix[0].index_add_(0, gsz, torch.ones_like(gsz))
        mix[1].index_add_(0, gsz, gsum)
        mix[2].index_add_(0, gsz, guni)
        tiles_call = None
        if self.count and self.impl == "kernel":        # untimed: the kernel's own tile counter (the byte model)
            cnt = torch.zeros(2, dtype=torch.int32, device=q.device)
            bi_paged_decode_shared(self.base, *args, page_key, rows_max=nr, mode=self.mode,
                                   softmax_scale=softmax_scale, pages_per_split=pps, union=un, count_loads=cnt)
            tiles_call = cnt[0].to(torch.int64)
            self.tiles["shared"] = self.tiles["shared"] + tiles_call
            self.tiles["per_row"] = self.tiles["per_row"] + npg.sum()
            self.tiles["rows_shared"] = self.tiles["rows_shared"] + cnt[1].to(torch.int64)
            self.tiles["rows_per_row"] = self.tiles["rows_per_row"] + ((npg - 1) * self.base.PAGE).sum() \
                + (last_len.to(torch.int64) * num_pages.shape[1]).sum()
        self.mixes.append((call, nr, mix, npg.sum(), tiles_call))
        if len(self.tables) < self.record_tables and nr == self.rows_max:      # [review MAJOR-1] F4's own tables
            self.tables.append(dict(call=call, rows_max=nr, mode=self.mode, seq_idx=seq_idx.clone(),
                                    page_table=page_table.clone(), num_pages=num_pages.clone(),
                                    last_len=last_len.clone(), page_key=page_key.clone(), tiles=tiles_call,
                                    union={f: box["raw"][f].clone() for f in UNION_FIELDS}))   # [sharedkv4]
        ref, sh = (a, b) if first == "per_row" else (b, a)
        iv = torch.int16 if ref.element_size() == 2 else torch.int32
        rv, sv = ref.view(iv), sh.view(iv)
        dm = rv != sv                                                           # bits differ
        same_val = ref == sh                                                    # floats equal: -0.0 == +0.0
        d = dm.flatten(1).any(1).sum()
        self.diff_rows = d if self.diff_rows is None else self.diff_rows + d
        # [review MINOR-2] F1's triage on the device: sign-of-zero-only elements vs value elements, and the first
        # differing element of the call (flag, flat index, both bit patterns), read once in report()
        self.diff_zero = self.diff_zero + (dm & same_val).sum()
        self.diff_value = self.diff_value + (dm & ~same_val).sum()
        fi = torch.argmax(dm.flatten().to(torch.int32))
        self.samples.append((call, torch.stack([dm.any().to(torch.int64), fi.to(torch.int64),
                                                rv.flatten()[fi].to(torch.int64), sv.flatten()[fi].to(torch.int64)])))
        self.rows_cmp += int(q.shape[0])
        if out is not None:
            out.copy_(ref)
            return out
        return ref

    def _check_union(self, raw, seq_idx, page_table, num_pages, page_key, nr, pps) -> None:
        """[sharedkv4, review MAJOR-1] the fused union of this call against build_union on the same inputs, on the
        device (no host read): per field the count of differing elements (a shape or dtype difference counts every
        element of the larger tensor); report() reads the counts once. A wrong window (too wide: identity steps,
        invisible to the bitwise gate) is caught here."""
        ref = build_union(seq_idx, page_table, num_pages, page_key, nr, pps, self.mode == "whole")
        for f in UNION_FIELDS:
            a, b = raw[f], ref[f]
            if a.shape == b.shape and a.dtype == b.dtype:
                d = (a != b).to(torch.int64).sum()
            else:
                d = max(a.numel(), b.numel()) + 1
            self.union_diff[f] = self.union_diff[f] + d
        self.union_checked += 1

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

    def report(self, n_samples: int = 32) -> dict:
        """ONE host read at the end: the differing rows and elements (sign-of-zero only vs value) with the first
        n_samples differing calls' first element; the per-call event times (ms; the shared side includes the union
        build; REPORT only, the S3 cell's device kernel time is the primary time ratio); the tiles; the group mix:
        per group size s the groups, their per-row pages and union pages (all keyed calls, and the calls with
        rows_max == self.rows_max), the counter's identity checks (whole: tiles = union pages; split: union <= tiles <=
        sum) and the mix-derived expectation of the tile ratio (expected_tile_ratio with the registered RHO)."""
        if self.events:
            torch.cuda.synchronize()
        t_pr, t_sh = [], []
        for nr, first, e0, e1, e2 in self.events:
            x, y = e0.elapsed_time(e1), e1.elapsed_time(e2)
            pr, sh = (x, y) if first == "per_row" else (y, x)
            t_pr.append((nr, first, pr))
            t_sh.append((nr, first, sh))
        tiles = {k: int(v) for k, v in self.tiles.items()}
        samples = []
        rows = torch.stack([s for _, s in self.samples]).tolist() if self.samples else []
        for (call, _), (flag, fi, ba, bb) in zip(self.samples, rows):
            if flag and len(samples) < n_samples:
                samples.append(dict(call=call, flat_index=fi, bits_per_row=ba, bits_shared=bb))
        mix_all = [[0] * (ROWS_MAX + 1) for _ in range(3)]
        mix_full = [[0] * (ROWS_MAX + 1) for _ in range(3)]
        ident_bad = []
        per_call = []
        for call, nr, mix, s, t in self.mixes:
            m = mix.tolist()
            for i in range(3):
                for k in range(ROWS_MAX + 1):
                    mix_all[i][k] += m[i][k]
                    if nr == self.rows_max:
                        mix_full[i][k] += m[i][k]
            tc = None if t is None else int(t)
            uni, sm = sum(m[2][1:]), int(s)
            if tc is not None and ((self.mode == "whole" and tc != uni) or not uni <= tc <= sm):
                ident_bad.append(dict(call=call, tiles=tc, union=uni, sum=sm))
            if len(per_call) < 4096:
                per_call.append(dict(call=call, rows_max=nr, groups=m[0], sum_pages=m[1], union_pages=m[2], tiles=tc))
        return dict(stats=dict(self.stats), keyed_calls=self.calls, by_rows_max=dict(self.by_rows_max),
                    rows_compared=self.rows_cmp, plant=self.plant,
                    diff_rows=int(self.diff_rows) if self.diff_rows is not None else 0,
                    diff_elements_zero_sign=int(self.diff_zero), diff_elements_value=int(self.diff_value),
                    diff_samples=samples, per_row_ms=t_pr, shared_ms=t_sh, tiles=tiles,
                    tile_ratio=tiles["shared"] / tiles["per_row"] if tiles["per_row"] else None,
                    row_ratio=tiles["rows_shared"] / tiles["rows_per_row"] if tiles["rows_per_row"] else None,
                    mix_all=mix_all, mix_full=mix_full, tile_identity_bad=ident_bad[:16],
                    tile_identity_checked=sum(1 for x in self.mixes if x[4] is not None),
                    expected_tile_ratio=expected_tile_ratio(mix_all, self.mode),
                    union_ratio_by_size=[(mix_all[2][k] / mix_all[1][k]) if mix_all[1][k] else None
                                         for k in range(ROWS_MAX + 1)],
                    per_call=per_call, mode=self.mode, union=self.union,
                    union_builds=dict(self.union_builds), group_builds=self.group_builds,
                    union_checked=self.union_checked,                                # [sharedkv4] MAJOR-1 / MINOR-5
                    union_diff={f: int(v) for f, v in self.union_diff.items()})

    def report_tables(self) -> list:
        """The recorded inputs (MAJOR-1), moved to the host (one read; the gate saves them with torch.save)."""
        def host(v):
            if torch.is_tensor(v):
                return v.cpu()
            if isinstance(v, dict):                       # [sharedkv4] the recorded fused union
                return {k: host(x) for k, x in v.items()}
            return v
        return [{k: host(v) for k, v in t.items()} for t in self.tables]


# [review MAJOR-1] REGISTERED per group size s (docs/R2_SHAREDKV_PLAN.md P2, revised): the expected tile / sum ratio
# of a group of s rows of one request. s = 4 is the 4.5 harness basis (whole 0.282 -> 0.29; split 0.45, an estimate);
# s = 1 reads its own list (1.0); s = 2, 3 interpolate by the union of s rows' selections (U(3 verify) = 69 of 3 x 63).
RHO = {"whole": {1: 1.00, 2: 0.55, 3: 0.40, 4: 0.29}, "split": {1: 1.00, 2: 0.70, 3: 0.55, 4: 0.45}}
RHO_BAND = (0.80, 1.25)          # the measured tile ratio / the mix-derived expectation


def expected_tile_ratio(mix, mode: str):
    """The registered formula: sum_s RHO[mode][s] x (per-row pages of the groups of size s) / sum_s (per-row pages);
    mix = [groups by size, per-row pages by size, union pages by size]. None without pages."""
    den = sum(mix[1][s] for s in range(1, ROWS_MAX + 1))
    if den <= 0:
        return None
    return sum(RHO[mode][s] * mix[1][s] for s in range(1, ROWS_MAX + 1)) / den
