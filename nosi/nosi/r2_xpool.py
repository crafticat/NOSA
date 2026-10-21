"""R2 step S6: the SHARED X POOL store layout, OPT-IN (docs/R2_XPOOL_PLAN.md). kstep/ownership.py:47-50 names it ("the
reduced B * H * C + X is step S6, not approved"); the audit paper-notes/recon-row-1b-alloc.md Q4 sizes it. The default
stays decision D4's per-stream layout (r2_residency.py:11-14): a caller that passes no XPool runs today's code.

THE LAYOUT (mode "pool"). Per layer ONE flat store: K and V (M * 64, H, D), bias (M * 64, H), M = B * (C + 4) + X_pool
POSITIONS of 64 rows. Every row holds the H KV heads, so a position has H LANES and lane h holds KV head h only: the
gilfree packer writes destination row (m * 64 + r) * H + h (gilfree_pack.cpp:152), and the attention kernel adds
h * stride(2); both fix the lane to the head. So the pool is shared by the B requests, per head.
  positions [0, A), A = B * C + X_pool: the HISTORY ARENA, one free list of A lanes per (layer, head). A logical slot
      (b, h, s) of the residency service (s < C + X: the service's NPOS, unchanged) is BOUND to a lane when the device
      decides to take it (ResidencySC.resolve, after device_verify / device_draft, BEFORE the urgent gather and the
      draft bias fill) and UNBOUND when it is FREE at the next tick's first resolve (after sync_mirror). Lowest free
      lane first, in (b, s) order per head. Policy E reuses a slot of the same stream: its binding stays.
  positions [A, M): the TAIL RINGS, static: request b's ring page j (0, 1 exact; 2, 3 draft) is position A + 4 b + j.
CAPACITY (K5, kstep/ownership.py:409-427, usage counts RELEASING too :135-136): per layer the used slots number at most
B * H * C + X over all streams, so per head at most B * C + X. X_pool >= X therefore never runs out; the binder still
checks it (err bit 1). Ownership, K1-K5, Policy E and every refusal are unchanged: the service decides on logical
slots exactly as before, and the binding is a pure function of its decisions.
THE MAP pos_of (L, B, H, C + X + 4) int64 on the store's device (-1 = unbound) is the only translation: the page tables
(TickEngine's backends: translate), the urgent gather (gather_entries + h2d_entries), the draft bias fill
(draft_bias_fill_flat), the bias fill at a host take (Residency._fill_bias), the transports (Transfer.phys, set by
assign_phys at end_tick: ONE device read for the tick's deferred transfers) and the engine's tail-ring writes.
THE KERNEL: unchanged. The engine hands piece 3's bi_paged_decode the views attn_view(store) (B, M * 64, H, D) with
stride(0) = 0 ("FLAT ADDRESS MODE"): the kernel's n * s_kn term is 0, so a page-table entry is an absolute position
(docs/R2_XPOOL_PLAN.md, "Kernel interface").

MODE "resident" (tests; the ResidentStore): every history slot s of (b, h) is statically bound to position b * NB + s
(NB = n_hist): the flat addressing without the pool, the same bytes as the per-stream store.

THE ERROR FLAG err (0-dim int64 on the device; ResidencySC watches it: the tick's ONE decision read refuses the tick):
  1 (ERR_FULL)    the arena of a (layer, head) ran out at a bind;
  2 (ERR_ALIAS)   after a bind the layer's binding is not injective (two bound slots on one lane of a head) or the free
                  list disagrees with the bindings (audit, device ops, every bind);
  4 (ERR_UNBOUND) a page table or a gather names an unbound slot.
NEGATIVE CONTROLS (mut, tests only; host reads):
  "alias"              at the first bind where it can, the lane of the bind's first new slot of a head goes to a new
                       slot of ANOTHER request too (once per object); plant_log records both slots.
  "release_inflight"   release_free also unbinds INFLIGHT slots (a transfer may still land in the lane);
  "release_releasing"  release_free also unbinds RELEASING slots (the lane's last reader may still run);
                       early_log records every early unbind and, at a later bind, the request the lane went to.

THE LANE-LEVEL ORDER (review M4 of 44d5e95; docs/R2_XPOOL_PLAN.md 2.2). A transfer waits the LOGICAL slot's last
reader (Transfer.reader_ev), not the lane's previous occupant's (the M2 arena contract, kstep/transfer.py:26). The lane
is safe because it is released only at FREE (the RELEASING -> FREE reader gate) AND because S-C's decision read is
BLOCKING (ResidencySB._d2h: t.to("cpu") after every layer's attention): every reader of the tick has finished on the
device before the next tick's release. ResidencySC refuses the pool when its class says the read is not blocking
(DECISION_READ_BLOCKING); the CPU test XP10 pins both.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

try:                                                   # triton is needed only to launch h2d_entries' kernel on the GPU
    import triton
    import triton.language as tl
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None

BLOCK = 64
N_TAIL = 4
MODES = ("pool", "resident")
XPOOL_MUTS = ("alias", "release_inflight", "release_releasing")
ERR_FULL, ERR_ALIAS, ERR_UNBOUND = 1, 2, 4
POOL_RULES = ("stream", "k5")
# r2_residency_sc's state codes (C_FREE .. C_RELEASING = range(6)); the CPU test XP10 pins them equal
CODE_FREE, CODE_INFLIGHT, CODE_RELEASING = 0, 1, 5


class XPoolError(RuntimeError):
    """A refusal of the shared-pool layout (construction, an unbound transfer)."""


def equal_memory_pool(C: int, X: int, rule: str = "stream", pool_lanes: int = 0, B: int = 1, H: int = 1) -> int:
    """The shipped engine's NOSI_POOL_BLOCKS at the R2 store's memory per (request, head) (cache_engine.py:202: topk 64
    + P victim slots of K, V and bias per (request, head)).
      "stream" (today's rule, r2_tick_sg5.py, r2_tick_sg5_step.sh:141): P = C + X + 4 - 64: the per-stream store's
               bytes (C + X + 4 slots per stream).
      "k5"     (the shared pool; audit Q3): P = C + 4 - 64 + floor(pool_lanes / (B * H)): the base C + 4 per stream plus
               an equal share of the pool's lanes; pool_lanes = the pool's (position, head) lanes per layer (this
               layout: H * X_pool).
    Rounded down; at least 0 (the harness's max(0, .))."""
    if rule == "stream":
        p = int(C) + int(X) + 4 - 64
    elif rule == "k5":
        if int(B) < 1 or int(H) < 1 or int(pool_lanes) < 0:
            raise ValueError("k5 rule: B, H >= 1 and pool_lanes >= 0")
        p = int(C) + 4 - 64 + int(pool_lanes) // (int(B) * int(H))
    else:
        raise ValueError("rule must be one of %s (got %r)" % (POOL_RULES, rule))
    return max(0, p)


def store_bytes_formula(B: int, H: int, D: int, C: int, X_pool: int, elem: int) -> int:
    """Bytes of ONE layer's pool-layout store (K, V, bias): (B * (C + 4) + X_pool) * 64 rows x H x (2 D + 1) x elem."""
    return (int(B) * (int(C) + N_TAIL) + int(X_pool)) * BLOCK * int(H) * (2 * int(D) + 1) * int(elem)


def per_stream_bytes_formula(B: int, H: int, D: int, n_hist: int, elem: int) -> int:
    """Bytes of ONE layer's per-stream store (D4): B * (n_hist + 4) * 64 rows x H x (2 D + 1) x elem."""
    return int(B) * (int(n_hist) + N_TAIL) * BLOCK * int(H) * (2 * int(D) + 1) * int(elem)


class XPool:
    """The layout of one run's store (module docstring). L, B, H: the engine's; n_hist: the logical history slots per
    stream (the service's C + X; resident: NB); mode "pool" needs C and X_pool >= n_hist - C."""

    def __init__(self, L: int, B: int, H: int, n_hist: int, mode: str = "pool", C: Optional[int] = None,
                 X_pool: Optional[int] = None, device="cpu", mut=None):
        if mode not in MODES:
            raise XPoolError("mode must be one of %s (got %r)" % (MODES, mode))
        self.L, self.B, self.H, self.n_hist = int(L), int(B), int(H), int(n_hist)
        if min(self.L, self.B, self.H, self.n_hist) < 1:
            raise XPoolError("L, B, H, n_hist >= 1")
        self.mode = mode
        NS = self.n_hist
        if mode == "pool":
            if C is None or X_pool is None:
                raise XPoolError("mode 'pool' needs C and X_pool")
            self.C, self.X_pool = int(C), int(X_pool)
            X = NS - self.C
            if self.C < 1 or X < 0:
                raise XPoolError("n_hist %d must be C + X with C %d >= 1 and X >= 0" % (NS, self.C))
            if self.X_pool < X:
                raise XPoolError("X_pool %d < X %d: K5 lets the streams of one head borrow up to X extras per layer, "
                                 "so a smaller pool could run out under a take K5 grants" % (self.X_pool, X))
            self.A = self.B * self.C + self.X_pool
        else:
            self.C, self.X_pool = None, 0
            self.A = self.B * NS
        self.M = self.A + N_TAIL * self.B
        self.device = torch.device(device)
        self.mut = dict(mut or {})
        for k in self.mut:
            if k not in XPOOL_MUTS:
                raise XPoolError("mut %r (known: %s)" % (k, XPOOL_MUTS))
        L_, B_, H_ = self.L, self.B, self.H
        pos = torch.full((L_, B_, H_, NS + N_TAIL), -1, dtype=torch.int64)
        tails = self.A + N_TAIL * torch.arange(B_).view(B_, 1, 1) + torch.arange(N_TAIL).view(1, 1, N_TAIL)
        pos[..., NS:] = tails.expand(B_, H_, N_TAIL)
        if mode == "resident":
            pos[..., :NS] = (torch.arange(B_).view(B_, 1, 1) * NS + torch.arange(NS).view(1, 1, NS)).expand(B_, H_, NS)
        self.pos_of = pos.to(self.device)
        self.occ = torch.full((L_, H_, self.A), mode == "resident", dtype=torch.bool, device=self.device)
        self.err = torch.zeros((), dtype=torch.int64, device=self.device)
        self.peak = torch.zeros((L_, H_), dtype=torch.int64, device=self.device)   # REPORT: most lanes bound at once
        self.binds = 0                                                               # bind calls (host count)
        self.plant_log: List[Dict] = []
        self.early_log: List[Dict] = []                                              # the early-release controls

    # ------------------------------------------------------------------------------------------------ geometry
    def tail_pos(self, b: int, j: int) -> int:
        """The static position of request b's tail-ring page j (0, 1 exact; 2, 3 draft)."""
        if not (0 <= int(b) < self.B and 0 <= int(j) < N_TAIL):
            raise XPoolError("tail page (b %d, j %d) outside (B %d, %d pages)" % (b, j, self.B, N_TAIL))
        return self.A + N_TAIL * int(b) + int(j)

    def hist_pos_static(self, b: int, s: int) -> int:
        """Mode "resident": the static position of history slot s of request b (every head)."""
        if self.mode != "resident":
            raise XPoolError("hist_pos_static is the resident mode's (the pool binds at a take)")
        return int(b) * self.n_hist + int(s)

    def new_store(self, D: int, dtype=torch.float32, device=None):
        """One layer's flat store: K, V (M * 64, H, D) and bias (M * 64, H), zeros."""
        dev = self.device if device is None else device
        k = torch.zeros((self.M * BLOCK, self.H, int(D)), dtype=dtype, device=dev)
        return k, torch.zeros_like(k), torch.zeros((self.M * BLOCK, self.H), dtype=dtype, device=dev)

    def check_store(self, k, v, bias) -> None:
        """A layer's flat store has this layout's shape (K, V (M * 64, H, D) contiguous, bias (M * 64, H))."""
        if k.dim() != 3 or tuple(k.shape[:2]) != (self.M * BLOCK, self.H) or v.shape != k.shape \
                or tuple(bias.shape) != (self.M * BLOCK, self.H) or not (k.is_contiguous() and v.is_contiguous()):
            raise XPoolError("flat store K / V (%d, %d, D) contiguous and bias (%d, %d) expected; got %s %s %s"
                             % (self.M * BLOCK, self.H, self.M * BLOCK, self.H, tuple(k.shape), tuple(v.shape),
                                tuple(bias.shape)))

    def attn_view(self, t: torch.Tensor) -> torch.Tensor:
        """FLAT ADDRESS MODE for piece 3's kernel (docs/R2_XPOOL_PLAN.md "Kernel interface"): the flat store as
        (B, M * 64, H[, D]) with stride(0) = 0, so every row's storage entry n addresses the same base and a page-table
        entry is an absolute position. A view: no copy."""
        return t.unsqueeze(0).expand((self.B,) + tuple(t.shape))

    def clone(self) -> "XPool":
        c = XPool.__new__(XPool)
        c.__dict__.update(self.__dict__)
        for name in ("pos_of", "occ", "err", "peak"):
            setattr(c, name, getattr(self, name).clone())
        c.mut = dict(self.mut)
        c.plant_log = list(self.plant_log)
        c.early_log = [dict(e) for e in self.early_log]
        return c

    # ------------------------------------------------------------------------------------------------ binding
    def bind(self, l: int, bb: torch.Tensor, hh: torch.Tensor, slot: torch.Tensor, mask: torch.Tensor) -> None:
        """Bind layer l's taken slots that have no lane yet (device ops, no host read; mode "pool" only). bb, hh, slot,
        mask: one entry per decision candidate (N,); an entry takes slot `slot` of stream (bb, hh) where mask is
        nonzero. Per head the new slots in (b, s) order get the lowest free lanes. Then the layer's audit."""
        if self.mode != "pool":
            return
        self.binds += 1
        B, H, NS, A = self.B, self.H, self.n_hist, self.A
        dev = self.device
        bb, hh, slot = (x.reshape(-1).to(torch.int64) for x in (bb, hh, slot))   # on the device already
        ok = (mask.reshape(-1) != 0) & (slot >= 0) & (slot < NS)
        f = (bb * H + hh) * NS + slot.clamp(min=0, max=NS - 1)
        want = torch.zeros(B * H * NS + 1, dtype=torch.bool, device=dev)
        want[torch.where(ok, f, torch.full_like(f, B * H * NS))] = True
        want = want[:B * H * NS].view(B, H, NS)
        cur = self.pos_of[l, :, :, :NS]                                             # (B, H, NS) view
        needh = (want & (cur < 0)).permute(1, 0, 2).reshape(H, B * NS)              # per head, (b, s) order
        ni = needh.to(torch.int64)
        rank = ni.cumsum(-1) - ni
        occ = self.occ[l]                                                           # (H, A) view
        ar = torch.arange(A, device=dev).view(1, A).expand(H, A)
        free_sorted = torch.sort(torch.where(occ, torch.full_like(ar, A), ar), dim=-1).values
        nfree = (~occ).to(torch.int64).sum(-1, keepdim=True)
        over = needh & (rank >= nfree)
        assign = needh & ~over
        newpos = torch.gather(free_sorted, 1, rank.clamp(max=A - 1))
        if self.mut.get("alias") and not self.plant_log:
            newpos = self._plant_alias(l, assign, newpos)                           # NEGATIVE CONTROL (host reads)
        if self.early_log:
            self._note_rebinds(l, assign, newpos)                                   # NEGATIVE CONTROL (host reads)
        cur_h = cur.permute(1, 0, 2).reshape(H, B * NS)
        self.pos_of[l, :, :, :NS] = torch.where(assign, newpos, cur_h).view(H, B, NS).permute(1, 0, 2)
        mark = torch.zeros((H, A + 1), dtype=torch.bool, device=dev)
        mark.scatter_(1, torch.where(assign, newpos, torch.full_like(newpos, A)), torch.ones_like(assign))
        occ |= mark[:, :A]
        self.err.bitwise_or_(torch.any(over).to(torch.int64) * ERR_FULL)
        self.peak[l] = torch.maximum(self.peak[l], occ.to(torch.int64).sum(-1))
        self.audit(l)

    def _plant_alias(self, l: int, assign: torch.Tensor, newpos: torch.Tensor) -> torch.Tensor:
        """NEGATIVE CONTROL: per head, the first new slot keeps its lane and the first new slot of ANOTHER request gets
        the same lane (its own lane stays free). Recorded in plant_log; no plant where no head has two requests."""
        NS = self.n_hist
        for h in range(self.H):
            js = assign[h].nonzero().view(-1).tolist()
            if len(js) < 2:
                continue
            j0 = js[0]
            j1 = next((j for j in js[1:] if j // NS != j0 // NS), None)
            if j1 is None:
                continue
            out = newpos.clone()
            out[h, j1] = newpos[h, j0]
            self.plant_log.append(dict(l=int(l), h=h, first=(j0 // NS, j0 % NS), second=(j1 // NS, j1 % NS),
                                       pos=int(newpos[h, j0])))
            return out
        return newpos

    def _note_rebinds(self, l: int, assign: torch.Tensor, newpos: torch.Tensor) -> None:
        """NEGATIVE CONTROLS (early release): every early-unbound lane of layer l that this bind gives to a slot records
        the request it went to (the controls' precondition: a lane changed hands after an early unbind)."""
        NS = self.n_hist
        for h in range(self.H):
            js = assign[h].nonzero().view(-1).tolist()
            lanes = newpos[h].tolist()
            for j in js:
                for e in self.early_log:
                    if e["l"] == l and e["h"] == h and e["lane"] == lanes[j] and e.get("to") is None:
                        e["to"] = (j // NS, j % NS)

    def audit(self, l: int) -> None:
        """Device ops: per head of layer l, every lane holds at most one bound slot, and the free list (occ) is exactly
        the set of bound lanes; else err |= ERR_ALIAS."""
        if self.mode != "pool":
            return
        H, NS, A = self.H, self.n_hist, self.A
        p = self.pos_of[l, :, :, :NS].permute(1, 0, 2).reshape(H, -1)
        bound = p >= 0
        cnt = torch.zeros((H, A + 1), dtype=torch.int64, device=self.device)
        cnt.scatter_add_(1, torch.where(bound, p, torch.full_like(p, A)), bound.to(torch.int64))
        c = cnt[:, :A]
        bad = torch.any(c > 1) | torch.any((c > 0) != self.occ[l])
        self.err.bitwise_or_(bad.to(torch.int64) * ERR_ALIAS)

    def release_free(self, code: torch.Tensor) -> None:
        """Unbind every bound slot that is FREE (code (L, B, H, n_hist): the device mirror's state codes after
        sync_mirror; a FREE slot's last reader completed, K4 / T3). Device ops only (the early-release NEGATIVE
        CONTROLS read the host in _early)."""
        if self.mode != "pool":
            return
        L, H, NS, A = self.L, self.H, self.n_hist, self.A
        dev = self.device
        p = self.pos_of[..., :NS]
        free = code == CODE_FREE                                                    # (the mirror: the same device)
        if self.mut.get("release_inflight") or self.mut.get("release_releasing"):
            free = self._early(code, p, free)
        rel = free & (p >= 0)
        li = torch.arange(L, device=dev).view(L, 1, 1, 1)
        hi = torch.arange(H, device=dev).view(1, 1, H, 1)
        idx = (li * H + hi) * (A + 1) + torch.where(rel, p, torch.full_like(p, A))
        clr = torch.zeros(L * H * (A + 1), dtype=torch.bool, device=dev)
        clr.index_put_((idx.reshape(-1),), rel.reshape(-1))     # the dummy lane A takes only False values
        self.occ &= ~clr.view(L, H, A + 1)[..., :A]
        self.pos_of[..., :NS] = torch.where(rel, torch.full_like(p, -1), p)

    def _early(self, code: torch.Tensor, p: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        """NEGATIVE CONTROLS (tests only): the FREE mask plus the INFLIGHT (release_inflight) or RELEASING
        (release_releasing) slots; each bound one is recorded in early_log (layer, request, head, slot, lane, code)."""
        bad = CODE_INFLIGHT if self.mut.get("release_inflight") else CODE_RELEASING
        early = (code == bad) & (p >= 0)
        for (l, b, h, s) in early.nonzero().tolist():
            self.early_log.append(dict(l=l, b=b, h=h, s=s, lane=int(p[l, b, h, s]), code=bad, to=None))
        return free | early

    # ------------------------------------------------------------------------------------------------ translation
    def translate(self, l: int, table: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        """A page table (R, H, P) of LOGICAL slots (history slots < n_hist, tail slots n_hist .. n_hist + 3; -1 =
        none) of the rows' requests seq_idx (R,) -> absolute positions (int32, contiguous; -1 kept). A listed slot with
        no lane sets ERR_UNBOUND (device ops, no host read)."""
        R, H, P = table.shape
        t = table.to(torch.int64)                                                   # the engine's device
        ok = t >= 0
        b = seq_idx.to(torch.int64).view(R, 1, 1).expand(R, H, P)
        h = torch.arange(H, device=t.device).view(1, H, 1).expand(R, H, P)
        p = self.pos_of[l][b, h, t.clamp(min=0, max=self.n_hist + N_TAIL - 1)]
        self.err.bitwise_or_(torch.any(ok & (p < 0)).to(torch.int64) * ERR_UNBOUND)
        return torch.where(ok, p, torch.full_like(p, -1)).to(torch.int32).contiguous()

    def gather_entries(self, l: int, bb: torch.Tensor, hh: torch.Tensor, blk: torch.Tensor, slot: torch.Tensor,
                       taken: torch.Tensor) -> torch.Tensor:
        """The urgent gather's entries (5, N) int64 CONTIGUOUS (lesson L28): b, h, block, position, valid; valid =
        taken with a lane (a taken slot without one sets ERR_UNBOUND and is skipped). Device ops."""
        tk = taken.reshape(-1) != 0
        bb, hh, blk, slot = (x.reshape(-1).to(torch.int64) for x in (bb, hh, blk, slot))   # the decision's device
        pos = self.pos_of[l][bb, hh, slot.clamp(min=0, max=self.n_hist - 1)]
        self.err.bitwise_or_(torch.any(tk & (pos < 0)).to(torch.int64) * ERR_UNBOUND)
        valid = tk & (pos >= 0) & (blk >= 0)
        return torch.stack([bb, hh, blk.clamp(min=0), pos.clamp(min=0), valid.to(torch.int64)]).contiguous()

    def assign_phys(self, transfers: Sequence) -> None:
        """Transfer.phys := the lane of each transfer's (layer, b, h, slot): ONE device-to-host read for the list (the
        tick's deferred draft transfers, at end_tick). A transfer without a lane is refused (the binder's invariant)."""
        trs = list(transfers)
        if not trs:
            return
        W = self.n_hist + N_TAIL
        f = torch.tensor([((tr.layer * self.B + tr.b) * self.H + tr.h) * W + tr.slot for tr in trs], dtype=torch.int64)
        vals = self.pos_of.view(-1)[f.to(self.device)].tolist()
        for tr, p in zip(trs, vals):
            if p < 0:
                raise XPoolError("transfer seq %d (layer %d b %d h %d slot %d) has no bound lane"
                                 % (tr.seq, tr.layer, tr.b, tr.h, tr.slot))
            tr.phys = int(p)

    def pos_host(self, l: int, b: int, h: int, s: int) -> int:
        """One slot's position (a host read: the CPU stand-ins and tests only)."""
        p = int(self.pos_of[l, b, h, s])
        if p < 0:
            raise XPoolError("slot (layer %d b %d h %d s %d) has no bound lane" % (l, b, h, s))
        return p

    def snapshot(self) -> Dict:
        """TESTS: host copies of the map, the free lists and the flag."""
        return dict(pos_of=self.pos_of.to("cpu").clone(), occ=self.occ.to("cpu").clone(), err=int(self.err))

    def report(self) -> Dict:
        """REPORT (one read; after a run): the layout and the most lanes bound at once per (layer, head)."""
        return dict(mode=self.mode, B=self.B, H=self.H, C=self.C, X_pool=self.X_pool, n_hist=self.n_hist, A=self.A,
                    M=self.M, binds=self.binds, peak_lanes=self.peak.tolist(), err=int(self.err),
                    plant_log=list(self.plant_log))


# ====================================================================================================================
# writers
# ====================================================================================================================
def draft_bias_fill_flat(xp: XPool, l: int, store_b_flat, cis_l, ddec, bbd, hhd, bs: int = BLOCK) -> None:
    """r2_residency_sparse.draft_bias_fill on the flat store: the bias rows of layer l's granted draft takes, ONE
    batched write on the device; lane rows pos * bs .. + bs, column h, get the cis rows block * bs .. + bs of (b, h). A
    granted take without a lane sets ERR_UNBOUND and writes nothing. Device ops only."""
    t = ddec.t
    g = t["granted"] != 0
    N = g.shape[0]
    rows = store_b_flat.shape[0]
    H = store_b_flat.shape[1]
    dev = store_b_flat.device
    slot = t["take_slot"].clamp(min=0, max=xp.n_hist - 1)
    pos = xp.pos_of[l][bbd, hhd, slot]
    xp.err.bitwise_or_(torch.any(g & (pos < 0)).to(torch.int64) * ERR_UNBOUND)
    ok = g & (pos >= 0)
    r = torch.arange(bs, device=dev).view(1, bs)
    dst = torch.where(ok.view(N, 1), pos.clamp(min=0).view(N, 1) * bs + r, torch.full_like(r, rows))
    src = t["blk"].clamp(min=0).view(N, 1) * bs + r
    b2, h2 = bbd.view(N, 1).expand(N, bs), hhd.view(N, 1).expand(N, bs)
    buf = torch.zeros((rows + 1, H), dtype=store_b_flat.dtype, device=dev)
    hit = torch.zeros((rows + 1, H), dtype=torch.bool, device=dev)
    buf[dst, h2] = cis_l[b2, src, h2].to(store_b_flat.dtype)
    hit[dst, h2] = True
    store_b_flat[...] = torch.where(hit[:rows], buf[:rows], store_b_flat)


def _check_entries(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent) -> None:
    """The contracts the kernel and its twin assume (each checked, lesson L28): ent (5, N) int64 contiguous; the flat
    store (rows, H, D) contiguous; the sources (B, S, H, D) with a unit last stride; one dtype for K / V both sides."""
    if ent.dim() != 2 or ent.shape[0] != 5 or ent.dtype != torch.int64 or not ent.is_contiguous():
        raise XPoolError("h2d_entries: ent must be a contiguous (5, N) int64 tensor (got %s %s contiguous=%s)"
                         % (tuple(ent.shape), ent.dtype, ent.is_contiguous()))
    for t, name in ((dst_k, "dst_k"), (dst_v, "dst_v")):
        if t.dim() != 3 or not t.is_contiguous():
            raise XPoolError("h2d_entries: %s must be a contiguous flat store (rows, H, D)" % name)
    for t, name in ((src_k, "src_k"), (src_v, "src_v")):
        if t.dim() != 4 or t.stride(-1) != 1:
            raise XPoolError("h2d_entries: %s must be (B, S, H, D) with a unit last stride" % name)
    if src_k.dtype != dst_k.dtype or src_v.dtype != dst_v.dtype or dst_k.shape[1:] != src_k.shape[2:]:
        raise XPoolError("h2d_entries: source %s %s and store %s %s differ in (H, D) or dtype"
                         % (tuple(src_k.shape), src_k.dtype, tuple(dst_k.shape), dst_k.dtype))
    if (dst_b is None) != (src_c is None):
        raise XPoolError("h2d_entries: dst_b and src_c go together")
    if dst_b is not None and (dst_b.dim() != 2 or src_c.dim() != 3):
        raise XPoolError("h2d_entries: dst_b (rows, H) and src_c (B, S, H)")


def entry_extents(dst_k, dst_b, src_k, src_c):
    """The extents an entry must lie in: (source requests, heads, source K / V rows, source cis rows or -1 without
    bias, destination rows)."""
    return (int(src_k.shape[0]), int(dst_k.shape[1]), int(src_k.shape[1]),
            int(src_c.shape[1]) if (dst_b is not None and src_c is not None) else -1, int(dst_k.shape[0]))


def entry_in_bounds(b, h, blk, pos, ok, bs, nb, nh, s_kv, s_c, rows) -> bool:
    """THE rule the kernel and the twin share (review M2 of 44d5e95): an entry is copied iff it is valid and every row
    it names exists: 0 <= b < nb, 0 <= h < nh, 0 <= blk with (blk + 1) bs <= s_kv (and <= s_c with a bias), and
    0 <= pos with (pos + 1) bs <= rows. Otherwise nothing of it is written."""
    if not ok or not (0 <= b < nb and 0 <= h < nh and blk >= 0 and pos >= 0):
        return False
    if (blk + 1) * bs > s_kv or (s_c >= 0 and (blk + 1) * bs > s_c):
        return False
    return (pos + 1) * bs <= rows


def h2d_entries_twin(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent, bs: int = BLOCK) -> int:
    """The CPU TWIN of _h2d_entries_kernel: the SAME flat element offsets from the base pointers (lesson L28): source
    b * s0 + (blk * bs + r) * s1 + h * s2 + d, destination (pos * bs + r) * t0 + h * t1 + d; bias the same without d.
    An entry outside the extents (entry_in_bounds: the kernel's own rule) is skipped whole, as the kernel skips it.
    Returns the entries copied. Host loops: tests only."""
    _check_entries(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent)
    D = dst_k.shape[2]
    r = torch.arange(bs, dtype=torch.int64).view(bs, 1)
    d = torch.arange(D, dtype=torch.int64).view(1, D)
    n = 0
    pairs = [(dst_k, src_k), (dst_v, src_v)]
    ext = entry_extents(dst_k, dst_b, src_k, src_c)
    for (b, h, blk, pos, ok) in ent.to("cpu").t().tolist():
        if not entry_in_bounds(b, h, blk, pos, ok, bs, *ext):
            continue
        for dst, src in pairs:
            so = b * src.stride(0) + (blk * bs + r) * src.stride(1) + h * src.stride(2) + d
            do = (pos * bs + r) * dst.stride(0) + h * dst.stride(1) + d
            sflat = torch.as_strided(src, (src.untyped_storage().nbytes() // src.element_size(),), (1,), 0)
            dflat = torch.as_strided(dst, (dst.untyped_storage().nbytes() // dst.element_size(),), (1,), 0)
            dflat[(do + dst.storage_offset()).reshape(-1)] = sflat[(so + src.storage_offset()).reshape(-1)]
        if dst_b is not None:
            so = b * src_c.stride(0) + (blk * bs + r.view(-1)) * src_c.stride(1) + h * src_c.stride(2)
            do = (pos * bs + r.view(-1)) * dst_b.stride(0) + h * dst_b.stride(1)
            sflat = torch.as_strided(src_c, (src_c.untyped_storage().nbytes() // src_c.element_size(),), (1,), 0)
            dflat = torch.as_strided(dst_b, (dst_b.untyped_storage().nbytes() // dst_b.element_size(),), (1,), 0)
            dflat[do + dst_b.storage_offset()] = sflat[so + src_c.storage_offset()].to(dst_b.dtype)
        n += 1
    return n


if triton is not None:
    @triton.jit
    def _h2d_entries_kernel(DK, DV, DB, SK, SV, SC, ENT, n_ent,
                            s_kb, s_ks, s_kh, s_vb, s_vs, s_vh, s_cb, s_cs, s_ch,
                            t_k0, t_k1, t_v0, t_v1, t_b0, t_b1,
                            nb, nh, s_kv, s_c, rows,
                            HAS_BIAS: tl.constexpr, BS: tl.constexpr, D: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)
        ok = tl.load(ENT + 4 * n_ent + e)
        b = tl.load(ENT + e)
        h = tl.load(ENT + n_ent + e)
        blk = tl.load(ENT + 2 * n_ent + e)
        pos = tl.load(ENT + 3 * n_ent + e)
        go = (ok != 0) & (b >= 0) & (b < nb) & (h >= 0) & (h < nh) & (blk >= 0) & (pos >= 0)   # entry_in_bounds
        go = go & ((blk + 1) * BS <= s_kv) & ((pos + 1) * BS <= rows)
        if HAS_BIAS:
            go = go & ((blk + 1) * BS <= s_c)
        if go:
            r = tl.arange(0, BS).to(tl.int64)
            d = tl.arange(0, D).to(tl.int64)
            src = blk * BS + r
            dst = pos * BS + r
            kv = tl.load(SK + b * s_kb + src[:, None] * s_ks + h * s_kh + d[None, :])
            tl.store(DK + dst[:, None] * t_k0 + h * t_k1 + d[None, :], kv)
            vv = tl.load(SV + b * s_vb + src[:, None] * s_vs + h * s_vh + d[None, :])
            tl.store(DV + dst[:, None] * t_v0 + h * t_v1 + d[None, :], vv)
            if HAS_BIAS:
                c = tl.load(SC + b * s_cb + src * s_cs + h * s_ch)
                tl.store(DB + dst * t_b0 + h * t_b1, c.to(DB.dtype.element_ty))
else:                                                  # pragma: no cover
    _h2d_entries_kernel = None


def h2d_entries(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent, bs: int = BLOCK, kernel: Optional[bool] = None) -> None:
    """Copy each valid entry (b, h, block, position) of ent (5, N) that lies in the extents (entry_in_bounds): the
    host rows block * bs .. + bs of (b, h) (K, V; with dst_b also the cis rows of src_c) into lane h of position
    `position` of the flat store, on the CURRENT stream. CUDA (kernel None): ONE launch of _h2d_entries_kernel, grid
    (N,) (an entry out of bounds returns at once); the sources may be pinned host rows (zero-copy reads, as NOSI's
    flash_h2d_from_mask). CPU: h2d_entries_twin. kernel True: the kernel on any device (the CPU test XP11 runs it under
    TRITON_INTERPRET=1). Not run on the GPU yet: cell x1 (docs/R2_XPOOL_PLAN.md) is its first GPU run."""
    _check_entries(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent)
    if kernel is None:
        kernel = dst_k.device.type == "cuda"
    if not kernel:
        h2d_entries_twin(dst_k, dst_v, dst_b, src_k, src_v, src_c, ent, bs)
        return
    if _h2d_entries_kernel is None:
        raise RuntimeError("triton is not installed")
    N = int(ent.shape[1])
    if N == 0:
        return
    D = int(dst_k.shape[2])
    if D < 1 or D & (D - 1) or int(bs) & (int(bs) - 1):
        raise XPoolError("h2d_entries: D %d and the block %d must be powers of two (tl.arange)" % (D, bs))
    has_b = dst_b is not None
    db, sc = (dst_b, src_c) if has_b else (dst_k, src_k)
    nb, nh, s_kv, s_c, rows = entry_extents(dst_k, dst_b, src_k, src_c)
    _h2d_entries_kernel[(N,)](
        dst_k, dst_v, db, src_k, src_v, sc, ent, N,
        src_k.stride(0), src_k.stride(1), src_k.stride(2), src_v.stride(0), src_v.stride(1), src_v.stride(2),
        sc.stride(0), sc.stride(1), sc.stride(2),
        dst_k.stride(0), dst_k.stride(1), dst_v.stride(0), dst_v.stride(1), db.stride(0), db.stride(1),
        nb, nh, s_kv, s_c, rows,
        HAS_BIAS=has_b, BS=int(bs), D=D)
