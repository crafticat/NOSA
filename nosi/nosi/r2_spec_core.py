"""R2 milestone 2, step 7: the PORTABLE core of the speculative round on the R2 path (docs/R2_STEP7_SPEC_ROUND.md).
Torch only (no CUDA, no Triton): the CPU tests drive it with stand-ins; the GPU path calls the same functions.

CONTENTS
  cadence            K3X cadence (C1-C4: cohorts with the newest token deferred, the lead bound LAMBDA, the flush),
                     restated; the CPU tests compare it with nosi/nosi/kstep/cadence.py when that tree is present.
  Round              the per-request round state machine and the tick plan: per tick and request at most one DRAFT row
                     and the verify rows of its oldest READY cohort (each position its own row: the flatten); the
                     greedy decision (accept while the verifier argmax equals the draft; the first difference is the
                     correction; no bonus, C3); per-request epochs; one host decision per tick that has a finished
                     cohort. Callbacks report publications and rollbacks to the storage layer.
  flatten_advance    the positional tables of a tick: ONE PositionalTwin with 2B banks (rows [0, B) exact, [B, 2B)
                     draft; SGLang DSv4 per-draft bank precedent); every bank advances by its rows in position order;
                     per-row counters. rollback_bank / sync_draft_bank / rebuild_draft_bank (K3X F5).
  stage1_alias       the stage-1 arguments of a flattened call: cu_seqlens_k repeated per bank, seqused_k per row.
  select_blocks      the DETERMINISTIC top-k pair (stable sort, value descending, index ascending), the shipped graph's
                     two-step rule, the result in ascending block id (canonical page order).
  page_lists         per (row, head) the slots of the selected blocks in ascending block id, the page count, the last
                     page's row count (the row's own extent); tail pages are a ring of two per bank (tail_page).
  paged_attention_reference
                     the BI paged attention in torch, page by page, online softmax in float32: an INDEPENDENT twin of
                     the piece-3 contract (nosi-r2-m2-biattn, bi_paged_attention.bi_paged_decode: logit =
                     (q.k + bias) * scale, one 64-row page per online-softmax step, the last page cut at last_len), in
                     piece 3 v1's order (4c1243f: no split). Piece 3 v2 (7c826c1, the step's pin since 2026-10-20)
                     splits a row's list every 8 pages and combines the splits in a fixed order, so for rows of more
                     than 8 pages it agrees with this twin to rounding only, not bitwise. The CPU tests S7b (v1) and
                     G9 (v2, test_r2_g0bi.py) and the GPU microgate B4 compare piece 3 with it.
  PaddedLinear       option G-b of the M-invariant GEMM: cuBLAS (F.linear) at ONE fixed row count m_fix; the live rows
                     are copied into a persistent (m_fix, K) buffer, the rows past them are never cleared.
  SlotStore         per (request, head) residency by CONTENT (block id -> slot): FREE / REQUESTED / ARRIVED, holders
                     per cohort key (uid, epoch, m), release only after the holders' decision and the reader event,
                     epoch-versioned arrivals, urgent misses (resident / wait / fetch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .r2_positional_twin import BLOCK, MAX_ROLLBACK, PositionalTwin

MASKED = float("-inf")


# =====================================================================================================================
# cadence (K3X cadence.py C1-C4)
# =====================================================================================================================
def check_cadence(K: int, LAMBDA: int) -> Tuple[int, int]:
    if isinstance(K, bool) or int(K) != K or int(K) < 2:
        raise ValueError("C1: K >= 2 is required (got %r)" % (K,))
    if LAMBDA is None or isinstance(LAMBDA, bool) or int(LAMBDA) != LAMBDA:
        raise ValueError("C2: LAMBDA must be an integer (got %r)" % (LAMBDA,))
    if int(LAMBDA) <= int(K):
        raise ValueError("C2: LAMBDA=%d <= K=%d is draft-K-then-stop" % (LAMBDA, K))
    return int(K), int(LAMBDA)


def cohort_range(m: int, K: int) -> Tuple[int, int]:
    if m < 1:
        raise ValueError("cohort index starts at 1")
    return (1, K - 1) if m == 1 else ((m - 1) * K, m * K - 1)


def triggered_by(s: int, K: int) -> Optional[int]:
    return s // K if (s >= K and s % K == 0) else None


# =====================================================================================================================
# the per-request round
# =====================================================================================================================
@dataclass
class Row:
    """One row of a tick. kind 'd' (draft) or 'v' (verify). pos: the position it writes; token: its input
    (u_{i-1}); i: the draft index it produces (d_i for a draft row) or checks (verify row); m: the cohort."""
    b: int
    kind: str
    pos: int
    token: int
    i: int
    epoch: int
    m: int = 0


@dataclass
class Cohort:
    m: int
    a: int
    b: int
    state: str = "TRIGGERED"                       # TRIGGERED -> RUNNING -> DONE
    g: Dict[int, int] = field(default_factory=dict)


@dataclass
class Req:
    uid: int
    E: int                                         # absolute position of u_0 in the current epoch
    u: List[int]                                   # u[0] = the pending token, u[s] = d_s
    epoch: int = 0
    s: int = 0                                     # drafted in the epoch
    committed_in_epoch: int = 0
    n_triggered: int = 0
    cohorts: List[Cohort] = field(default_factory=list)
    out: List[int] = field(default_factory=list)   # the committed output tokens
    max_new: int = 1 << 30
    flushed: bool = False
    done: bool = False


class Round:
    """The K-step round of B requests on the R2 path. plan_tick() returns the rows; finish_tick(outputs) applies the
    drafts, the verifier argmaxes and the decisions. Callbacks: on_publish(b, epoch, upto_pos) after a full
    acceptance (the exact positions < upto_pos are published), on_rollback(b, old_epoch, keep_pos) after a rejection
    (the exact state keeps positions < keep_pos). ready(b, cohort) gates a TRIGGERED cohort (residency, workspace)."""

    def __init__(self, first_tokens: Sequence[int], start_pos: Sequence[int], K: int = 3, LAMBDA: int = 7,
                 max_new: Optional[Sequence[int]] = None, on_publish: Callable = None, on_rollback: Callable = None,
                 ready: Callable = None):
        self.K, self.LAMBDA = check_cadence(K, LAMBDA)
        self.reqs = [Req(uid=b, E=int(p), u=[int(t)], out=[int(t)]) for b, (t, p) in enumerate(zip(first_tokens, start_pos))]
        if max_new is not None:
            for r, n in zip(self.reqs, max_new):
                r.max_new = int(n)
                r.done = len(r.out) >= r.max_new
        self.on_publish = on_publish or (lambda *a: None)
        self.on_rollback = on_rollback or (lambda *a: None)
        self.ready = ready or (lambda b, c: True)
        self.syncs = 0
        self.ticks = 0
        self.stats = dict(draft_rows=0, verify_rows=0, stall_lead=0, accepted=0, rejected=0, cohorts=0)

    # ------------------------------------------------------------------ plan
    def lead(self, r: Req) -> int:
        return r.s - r.committed_in_epoch

    def plan_tick(self) -> List[Row]:
        rows: List[Row] = []
        for r in self.reqs:
            if r.done:
                continue
            want = len(r.out) + self.lead(r)                     # committed + drafted-uncommitted tokens
            if not r.flushed and want < r.max_new:
                if self.lead(r) < self.LAMBDA:
                    rows.append(Row(r.uid, "d", r.E + r.s, r.u[r.s], r.s + 1, r.epoch))
                else:
                    self.stats["stall_lead"] += 1
            for c in r.cohorts:                                  # the oldest undecided cohort, if ready
                if c.state == "TRIGGERED" and self.ready(r.uid, c):
                    c.state = "RUNNING"
                    for i in range(c.a, c.b + 1):
                        rows.append(Row(r.uid, "v", r.E + i - 1, r.u[i - 1], i, r.epoch, c.m))
                if c.state != "DONE":
                    break
        return rows

    # ------------------------------------------------------------------ finish
    def finish_tick(self, rows: Sequence[Row], outputs: Sequence[int]) -> int:
        """outputs[k] = the row's argmax (draft: d_i; verify: g_i). Returns the number of host decision syncs (0/1).
        Refuses a tick with two draft rows or two cohorts of one request (section 2.3)."""
        seen_d, seen_m = set(), {}
        for row in rows:
            if row.kind == "d":
                if row.b in seen_d:
                    raise RuntimeError("tick: request %d has two draft rows" % row.b)
                seen_d.add(row.b)
            elif seen_m.setdefault(row.b, row.m) != row.m:
                raise RuntimeError("tick: request %d has verify rows of two cohorts" % row.b)
        self.ticks += 1
        done_any = False
        for row, tok in zip(rows, outputs):
            r = self.reqs[row.b]
            if row.epoch != r.epoch:                                         # a row of a cancelled epoch
                continue
            if row.kind == "d":
                self.stats["draft_rows"] += 1
                if row.i != r.s + 1:
                    raise RuntimeError("draft row out of order: i=%d s=%d" % (row.i, r.s))
                r.u.append(int(tok))
                r.s += 1
                m = triggered_by(r.s, self.K)
                if m is not None:
                    r.cohorts.append(Cohort(m, *cohort_range(m, self.K)))
                    r.n_triggered = m
                if len(r.out) + self.lead(r) >= r.max_new and not r.flushed:   # C4: the final short cohort
                    a = cohort_range(r.n_triggered + 1, self.K)[0]
                    if a <= r.s:
                        r.cohorts.append(Cohort(r.n_triggered + 1, a, r.s))
                    r.flushed = True
            else:
                self.stats["verify_rows"] += 1
                c = next(c for c in r.cohorts if c.m == row.m)
                c.g[row.i] = int(tok)
                if len(c.g) == c.b - c.a + 1:
                    c.state = "DONE"
                    done_any = True
        if done_any:
            self.syncs += 1
            for r in self.reqs:
                self._decide(r)
        return int(done_any)

    def _decide(self, r: Req) -> None:
        while r.cohorts and r.cohorts[0].state == "DONE":
            c = r.cohorts.pop(0)
            self.stats["cohorts"] += 1
            for i in range(c.a, c.b + 1):
                if c.g[i] == r.u[i]:
                    r.out.append(r.u[i])
                    r.committed_in_epoch += 1
                    self.stats["accepted"] += 1
                    continue
                self.stats["rejected"] += 1
                r.out.append(c.g[i])                                         # the correction
                old = r.epoch
                keep = r.E + i                                               # positions < keep stay exact
                r.epoch += 1
                r.E, r.u, r.s, r.committed_in_epoch, r.n_triggered = keep, [c.g[i]], 0, 0, 0
                r.cohorts, r.flushed = [], False
                self.on_rollback(r.uid, old, keep)
                break
            else:
                self.on_publish(r.uid, r.epoch, r.E + c.b)
            if len(r.out) >= r.max_new:
                r.done = True
                del r.out[r.max_new:]
                r.cohorts = []
                return


# =====================================================================================================================
# positional tables: one 2B twin (exact banks [0, B), draft banks [B, 2B))
# =====================================================================================================================
def flatten_advance(twin: PositionalTwin, bank_rows: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
    """bank_rows: {bank: (keys (n, H, D), cis (n, H))}, the rows of the tick in position order per bank. Advances every
    bank by its n rows (PositionalTwin.advance: step u writes position pos_b + u for the banks with n_b > u).
    Returns (ctr (T + 1, 4, banks), per-row counters {(bank, u): (pos_after, n_comp, n_ccis)})."""
    nb, H, D = twin.B, twin.H, twin.D
    T = max((int(k.shape[0]) for k, _ in bank_rows.values()), default=0)
    dev = twin.pos.device
    keys = torch.zeros((nb, max(T, 1), H, D), dtype=twin.ring.dtype, device=dev)
    cis = torch.zeros((nb, max(T, 1), H), dtype=twin.total_cis.dtype, device=dev)
    n = torch.zeros(nb, dtype=torch.int64)
    for bank, (k, c) in bank_rows.items():
        keys[bank, :k.shape[0]] = k
        cis[bank, :c.shape[0]] = c
        n[bank] = int(k.shape[0])
    ctr = twin.advance(keys[:, :T] if T else keys[:, :0], cis[:, :T] if T else cis[:, :0], n)
    per_row = {}
    for bank, (k, _) in bank_rows.items():
        for u in range(int(k.shape[0])):
            per_row[(bank, u)] = (int(ctr[u + 1, 0, bank]), int(ctr[u + 1, 1, bank]), int(ctr[u + 1, 2, bank]))
    return ctr, per_row


def rollback_bank(twin: PositionalTwin, ctr: torch.Tensor, keep: Dict[int, int]) -> None:
    """Reset banks to ctr[keep[bank]] (counter reset; the rows below stay as written). Banks not named keep the last
    snapshot of ctr (their state after the tick)."""
    full = torch.full((twin.B,), ctr.shape[0] - 1, dtype=torch.int64)
    for bank, k in keep.items():
        full[bank] = int(k)
    twin.commit(ctr, full)


def sync_draft_bank(twin: PositionalTwin, b: int, B: int) -> None:
    """The draft bank B + b takes the exact bank b: its tables (whole rows: every row below the counters is the exact
    one) and its counters. Epoch start, and the first half of rebuild_draft_bank."""
    d = B + b
    twin.comp_k[d].copy_(twin.comp_k[b])
    twin.ring[d].copy_(twin.ring[b])
    twin.total_cis[d].copy_(twin.total_cis[b])
    twin.ccis[:, d].copy_(twin.ccis[:, b])
    for t in (twin.pos, twin.n_comp, twin.n_ccis):
        t[d] = t[b]
    twin.prefilled[d] = twin.prefilled[b]
    twin.epoch[d] += 1


def rebuild_draft_bank(twin: PositionalTwin, b: int, B: int, draft_keys: torch.Tensor, draft_cis: torch.Tensor) -> None:
    """K3X F5 (_rebuild_dchain) on the R2 tables: after a full-acceptance publication the draft bank becomes the exact
    bank, then replays the draft's own keys / cis of its still-uncommitted positions (draft_keys (n, H, D),
    draft_cis (n, H), in position order). Summaries whose windows cover published positions are recomputed from the
    exact keys."""
    sync_draft_bank(twin, b, B)
    n = int(draft_keys.shape[0])
    if n == 0:
        return
    keys = torch.zeros((twin.B, n, twin.H, twin.D), dtype=twin.ring.dtype, device=twin.pos.device)
    cis = torch.zeros((twin.B, n, twin.H), dtype=twin.total_cis.dtype, device=twin.pos.device)
    keys[B + b] = draft_keys
    cis[B + b] = draft_cis
    cnt = torch.zeros(twin.B, dtype=torch.int64)
    cnt[B + b] = n
    twin.advance(keys, cis, cnt)


# =====================================================================================================================
# stage 1, selection, pages
# =====================================================================================================================
def stage1_alias(row_bank: Sequence[int], n_used: Sequence[int], lk_max: int, n_banks: int):
    """cu_seqlens_k (R + 1,) int32: entry r starts at its bank's table, row_bank[r] * lk_max (repeated for the rows of
    one bank; the op takes the length from seqused_k, BlockInfo actual_seqlen_k = seqused_k[r]); the last entry is the
    table end. seqused_k (R,) int32 = each row's own compressed-key count."""
    R = len(row_bank)
    cu = torch.empty(R + 1, dtype=torch.int32)
    for r, bank in enumerate(row_bank):
        if not 0 <= int(bank) < n_banks:
            raise ValueError("row %d: bank %d outside [0, %d)" % (r, bank, n_banks))
        if not 0 <= int(n_used[r]) <= lk_max:
            raise ValueError("row %d: n_used %d outside [0, %d]" % (r, n_used[r], lk_max))
        cu[r] = int(bank) * lk_max
    cu[R] = n_banks * lk_max
    return cu, torch.tensor([int(x) for x in n_used], dtype=torch.int32)


def stable_topk_desc(x: torch.Tensor, k: int) -> torch.Tensor:
    """The indices of the k largest values along the last dim; ties broken by the LOWER index (a stable sort of the
    values in descending order). The set and the order depend only on the row's values."""
    return torch.sort(x, dim=-1, descending=True, stable=True).indices[..., :k]


def select_blocks(pq: torch.Tensor, pc: torch.Tensor, qk_select: int, k: int) -> torch.Tensor:
    """The shipped graph's two-step selection (nosa_llama.py decode warm-up: topk(pooled q, QK_SELECT), the scatter mask,
    masked_fill(+inf) of the pooled cis, topk(pooled cis, topk)) with the deterministic stable top-k. pq, pc (H, R, M).
    Returns (H, R, k) block ids in ASCENDING order (the canonical page order)."""
    iq = stable_topk_desc(pq, qk_select)
    mask = torch.zeros_like(pc, dtype=torch.bool)
    mask.scatter_(2, iq, True)
    pc2 = pc.masked_fill(mask, float("inf"))
    return torch.sort(stable_topk_desc(pc2, k), dim=-1).values


def tail_page(pos: int) -> int:
    """The tail page (0 or 1) of a bank's ring that holds position pos."""
    return (int(pos) // BLOCK) % 2


def page_lists(sel: torch.Tensor, slot_of: Callable[[int, int, int], int], row_bank: Sequence[int],
               row_pos: Sequence[int], tail_slot: Callable[[int, int], int]):
    """sel (H, R, k) ascending block ids (the row's selection; it holds the row's own block pos // 64). slot_of(row,
    head, block) -> the slot of a historical block (< the tail blocks) or raises KeyError (not resident);
    tail_slot(bank, page) -> the slot of a tail page. The tail blocks of a row are pos // 64 (page tail_page(pos)) and,
    when it is still in the ring, the previous block. Returns pages (R, H, k) int32 slots in ascending block id,
    n_pages (R, H) int32, last_len (R,) int32 = pos % 64 + 1 (the row reads its own position)."""
    H, R, k = sel.shape
    pages = torch.empty((R, H, k), dtype=torch.int32)
    n_pages = torch.empty((R, H), dtype=torch.int32)
    for r in range(R):
        bank, pos = int(row_bank[r]), int(row_pos[r])
        T = pos // BLOCK
        for h in range(H):
            ids = sel[h, r].tolist()
            if ids[-1] != T or ids != sorted(ids):
                raise ValueError("row %d head %d: the selection must be ascending and end at its own block %d" % (r, h, T))
            out = []
            for blk in ids:
                if blk == T:
                    out.append(tail_slot(bank, tail_page(pos)))
                elif blk == T - 1 and slot_of_tail_ring(slot_of, r, h, blk) is None:
                    out.append(tail_slot(bank, tail_page(pos - BLOCK)))
                else:
                    out.append(slot_of(r, h, blk))
            pages[r, h] = torch.tensor(out, dtype=torch.int32)
            n_pages[r, h] = len(out)
    last = torch.tensor([int(p) % BLOCK + 1 for p in row_pos], dtype=torch.int32)
    return pages, n_pages, last


def slot_of_tail_ring(slot_of, r, h, blk):
    """The previous block T - 1 is read from the slot store when it is resident there, else from the tail ring."""
    try:
        return slot_of(r, h, blk)
    except KeyError:
        return None


def paged_attention_reference(q, k_store, v_store, b_store, req, pages, n_pages, last_len, scale, page=BLOCK,
                              p_dtype=None):
    """The BI paged attention, torch, in piece 3 v1's order (bi_paged_attention.bi_paged_decode, 4c1243f; v2 7c826c1
    splits every 8 pages: equal to rounding only for longer lists, see the module docstring): for row r and KV head h, its q heads h*G .. h*G+G-1; pages in list order; per page s = (q k^T + bias) * scale (bias
    before the scale, as FA2-NOSA), masked beyond the row's last-page length; online softmax in float32 (base 2: exp2((s - m) log2 e));
    acc += P V with P cast to p_dtype (default the store dtype). q (R, Hq, D); stores (Bq, slots * page, Hk[, D]);
    req (R,); pages (R, Hk, P); n_pages (R, Hk); last_len (R,). Returns (R, Hq, D) in q's dtype."""
    R, Hq, D = q.shape
    Hk = k_store.shape[2]
    G = Hq // Hk
    p_dtype = p_dtype or k_store.dtype
    log2e = 1.4426950408889634
    out = torch.empty_like(q)
    for r in range(R):
        b = int(req[r])
        for h in range(Hk):
            qh = q[r, h * G:(h + 1) * G].float()
            m_i = torch.full((G,), MASKED)
            l_i = torch.zeros(G)
            acc = torch.zeros(G, D)
            n = int(n_pages[r, h])
            for p in range(n):
                slot = int(pages[r, h, p])
                nv = int(last_len[r]) if p == n - 1 else page
                rows = slice(slot * page, slot * page + nv)
                k = k_store[b, rows, h].float()
                v = v_store[b, rows, h]
                s = (qh @ k.t() + b_store[b, rows, h].float()[None, :]) * (scale * log2e)
                m_new = torch.maximum(m_i, s.max(dim=1).values)
                alpha = torch.exp2(m_i - m_new)
                pexp = torch.exp2(s - m_new[:, None])
                l_i = l_i * alpha + pexp.sum(dim=1)
                acc = acc * alpha[:, None] + pexp.to(p_dtype).float() @ v.float()
                m_i = m_new
            out[r, h * G:(h + 1) * G] = (acc / l_i[:, None]).to(q.dtype)
    return out


# =====================================================================================================================
# G-b: the M-invariant GEMM by a fixed padded row count
# =====================================================================================================================
class PaddedLinear:
    """y = x @ w.T through F.linear at ONE fixed row count m_fix (option G-b, docs/R2_STEP7_SPEC_ROUND.md 3.3).

    One GEMM shape per (weight, dtype) means one cuBLAS algorithm, so the float order of a row does not depend on the
    number of live rows. The live rows go to rows 0 .. M-1 of a persistent (m_fix, K) buffer per (K, dtype, device);
    the rows M .. m_fix-1 keep whatever an earlier call left there (no clear: the gate B3 checks that their content
    and the row's index in the buffer do not reach the row's output). M > m_fix is refused (the tick's row count is at
    most B x (1 + U) = m_fix). The cost is the GEMM at m_fix rows for every call (B3's padded column)."""

    def __init__(self, m_fix: int):
        if isinstance(m_fix, bool) or int(m_fix) != m_fix or int(m_fix) < 1:
            raise ValueError("m_fix must be a positive int (got %r)" % (m_fix,))
        self.m_fix = int(m_fix)
        self._buf: Dict[Tuple, torch.Tensor] = {}

    def buffer(self, K: int, dtype, device) -> torch.Tensor:
        key = (int(K), dtype, str(device))
        buf = self._buf.get(key)
        if buf is None:
            buf = torch.zeros((self.m_fix, int(K)), dtype=dtype, device=device)
            self._buf[key] = buf
        return buf

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        if M > self.m_fix:
            raise ValueError("PaddedLinear: %d rows > m_fix %d" % (M, self.m_fix))
        buf = self.buffer(K, x.dtype, x.device)
        buf[:M].copy_(x2)
        y = torch.nn.functional.linear(buf, w)
        return y[:M].reshape(*lead, w.shape[0])


# =====================================================================================================================
# the slot store (residency by content)
# =====================================================================================================================
class SlotStore:
    """Residency of ONE (request, head): block id -> slot, with transfer state and holders.
    FREE -> REQUESTED (a background gather issued with an event and a generation) -> ARRIVED (the event was observed).
    holders: cohort keys (uid, epoch, m) that need the block; a slot is reusable only when it has no holder, no
    transfer in flight and its last reader event is done. An arrival whose generation is not the slot's current one
    (a cancelled or replaced request) is ignored."""

    def __init__(self, n_slots: int):
        self.n = int(n_slots)
        self.block = [-1] * self.n
        self.state = ["FREE"] * self.n
        self.gen = [0] * self.n
        self.holders: List[set] = [set() for _ in range(self.n)]
        self.reader_done: List[Callable[[], bool]] = [lambda: True] * self.n
        self.where: Dict[int, int] = {}
        self.fetches = 0

    def lookup(self, blk: int) -> Optional[int]:
        return self.where.get(int(blk))

    def _free_slot(self) -> int:
        """A FREE slot first; else a resident block that nobody holds and whose last reader is done (it leaves the
        store); else MemoryError (the caller stalls: STALL_MEM)."""
        for s in range(self.n):
            if self.state[s] == "FREE":
                return s
        for s in range(self.n):
            if self.state[s] == "ARRIVED" and not self.holders[s] and self.reader_done[s]():
                return s
        raise MemoryError("slot store full")

    def request(self, blk: int, holder: Tuple) -> Tuple[int, int, bool]:
        """A background request of blk for holder. Returns (slot, generation, issued): issued False when the block is
        already resident or in flight (no second fetch)."""
        s = self.lookup(blk)
        if s is not None:
            self.holders[s].add(holder)
            return s, self.gen[s], False
        s = self._free_slot()
        if self.block[s] >= 0:
            del self.where[self.block[s]]
        self.gen[s] += 1
        self.block[s], self.state[s], self.holders[s] = int(blk), "REQUESTED", {holder}
        self.where[int(blk)] = s
        self.fetches += 1
        return s, self.gen[s], True

    def arrive(self, slot: int, gen: int) -> bool:
        if gen != self.gen[slot] or self.state[slot] != "REQUESTED":
            return False
        self.state[slot] = "ARRIVED"
        return True

    def urgent(self, blk: int) -> str:
        """For a verify row at a layer: 'resident' (read it), 'wait' (in flight: wait for its event), 'fetch' (a miss:
        gather now on the main stream)."""
        s = self.lookup(blk)
        if s is None:
            return "fetch"
        return "resident" if self.state[s] == "ARRIVED" else "wait"

    def read(self, slot: int, reader_done: Callable[[], bool]) -> None:
        self.reader_done[slot] = reader_done

    def release(self, holder: Tuple) -> List[int]:
        """Drop holder everywhere. Returns the slots that became reusable now (no holder, not in flight, reader done);
        their content stays resident (the cache policy may keep it) until a request reuses the slot."""
        out = []
        for s in range(self.n):
            if holder in self.holders[s]:
                self.holders[s].discard(holder)
                if not self.holders[s] and self.state[s] == "ARRIVED" and self.reader_done[s]():
                    out.append(s)
        return out
