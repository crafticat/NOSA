"""I2 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the integration rung I2 = I1
(WS-A: spec_tentative.TentRunner, the tentative buffers on the schedule of contract 4, arm L4PT) + WS-B (verify_rows mode
"union": ONE residency per layer for the union of the joint pass's rows). ARM L4PTU. OPT-IN: benchmarks/Efficiency/
r2_i2_cell.py (--joint-residency union); no engine module and no older cell imports this module. Docs: docs/NOTES_i2.md.
NO NEW KERNEL: the shipped kernels that spec_tentative.py and verify_rows.py (mode union) already call, and torch ops.

THE SEAM (why WS-B's mode cannot be selected through the same ops object). TentRunner never calls verify_rows.joint_pass:
TentState.run_pass runs its OWN layer loop and uses verify_rows only for the batched selection (_select_rows), the GEMM
(_lin) and the per-row lengths (vs.n_used, vs.M_b); every row runs TentState._row with its own residency. So
VerifyOps(mode="union") on the runner's VerifyState alone would change nothing. verify_rows._union_rows cannot be called
either: it writes the tentative rows into the COMMITTED tail slot and store slot 0, writes a filled block back to the
host and moves the tail id (the FAST end state that a journal must undo: departure D1, which WS-A removed). This module
is the other side of the seam: TentStateU._rows replaces the row-by-row loop of a JOINT pass (R > 1) by the union pass
below. It is built from spec_tentative's own step 4 and step 5 (TentState._resid, TentState._write) and from verify_rows'
union helpers (_union_list, _store_fill, _store_call, _presence, _fa_check_u), all unchanged.

ONE JOINT PASS, per layer (R rows per request at their own positions p_r: a group's draft row skips the deferred
position, whose twin step is replayed from the buffer as in L4PT):
  1. hooks (CHECK-ONLY): on_select for every row, as TentState._row.
  2. THE RESIDENCY, ONE per layer and pass: row 0's selection on the COMMITTED window with spec_tentative's step 4
     (TentState._resid: the shipped diff, victim pool, map, LRU clock and gathers; a draft row 0 skips its misses; the
     tentative block is skipped; past the edge the committed tail id is injected for a head without it). KEPT (D1).
  3. every row's K, V and cis row into the TENTATIVE BUFFER (TentState._write, row order; tpos and the D3 counters).
     Never into the committed tail slot.
  4. THE UNION LIST (verify_rows._union_list): per (head, request) every block of a VERIFY row (a full row) and every
     pool hit of a DRAFT row (a resident row), minus the window's blocks and minus every TENTATIVE block (an id above the
     committed tail block qb = qc // 64: its rows are in the buffer, not on the host; the list reads the selection with
     those ids set to -1). THE STORE (verify_rows._store_fill): C fills of X - 1 blocks, pool hits device to device, misses
     from the host rows (the bound C = ceil((R - 1) topk / (X - 1)): nothing is dropped). Store slot 0 is never used.
  5. ATTENTION per entry e = b R + r (flash_attn_nosa over the E = B R entries, cache_batch_idx, the entry's own bias
     row: route 1, VRF-IDX), merged by a running log-sum-exp in float32 (verify_rows' merge):
       call A  the committed window: row r sees a window slot iff its block is in row r's selection; the tail slot
               (block qb's committed rows) iff the row is not past the edge or its head selects qb; the cut is
               TentState._row's (base + min(qc % 64, n_tail(p_r)), base + qc % 64 past the edge: findings F4 kept)
       store   one call per fill (verify_rows._store_call with in_tail all True: slot 0 is never visible): row r sees a
               listed block iff it is in row r's selection; the length ends at the highest slot that the row reads
       call B  the tentative buffer: entry x visible iff qc <= x <= p_r, F4 in p_r's block, and, past the edge, block
               qb's entries only for a head that selects qb (TentState._row's masks and its controls no_causal /
               no_lower)
     UNION RESIDENCY NEVER BECOMES UNION ATTENTION: each row sees its own selection only (the visible count).
  6. nothing else: the committed state changes only at the round's commit (TentState.commit), as in L4PT.
One-row passes (the P / D slots, a J slot whose requests all have one row) run TentState._row: one residency per pass.

WHAT DIFFERS FROM L4PT (both are FAST; both are checked against SH): after the pass the window holds row 0's selection
(L4PT: the last row's); rows 1 .. R - 1 read their other blocks from store slots (another float order of the parts); a
draft row r >= 1 also reads a block that a verify row of its request fetched (WS-B risk 7.12; a draft is not committed
state); gemm is the cell's choice (WS-B production: natural, one GEMM over E rows; L4PT: rowgroup).

CHECK-ONLY (VerifyOps(check=True): the check cells, the miniature, the CPU tests; OFF in the timing cells): the
selections of every layer (vs.chk_sel); the PRESENCE recount (every selected block of a verify row is in the window, in a
served store slot, block qb (tail slot and buffer) or the tentative block (buffer): stat dropped); the VISIBLE COUNT
(active rows: a verify row sees exactly its selection size, a draft row its selection size minus its skipped blocks:
stat vis_bad); VRF-IDX (vs.fa_check: tag A the window call, B the first store call, T the buffer call); the per-layer
record (VerifyOps(record=True), tests). ACCOUNTING (VerifyOps(stats=True)): roles, the window's ready and residual
loads, the store's pool and host blocks, the spill, the fills used, the streamed rows. HOST COUNTS (always, no device
op): TentStateU.calls resid / joint_resid / union_layers; vs.calls (GEMM, attention, gathers, fills, resid) and vs.passes.
PHASE MARKS (TentRunnerU(phase_marks=True), probe rounds only, excluded from the steady statistics): CUDA events at
"resid" (row 0's residency), "write" (the buffer writes), "union" (the list), "fetch" (fill 0), "attn" (call A and the
fill-0 call), "spill" (fills 1 .. C - 1 and their calls), "buffer" (call B), "merge" (the division), "post" (from the
end of the union rows to the next layer's first mark: the dense GEMMs, the next layer's twin and selection).
HOST SYNCS: none in a pass (VRF-IDX excepted: check-only, once).
NEGATIVE CONTROLS (tests only): TentRunnerU(union=False) runs the joint pass row by row (L4PT's code, R residencies per
layer); uneg "list_tentative" (the union list keeps the tentative block: a host fetch of rows that do not exist yet, and
the row sees the block twice); VerifyOps neg "drop_spill", "union_attention", "short_store_len" (WS-B's);
spec_tentative's tneg (no_causal / no_lower act on the union pass's call B too; recompute_deferred, undo_residency,
writeback_tentative, no_replay act on the shared code).
"""
from __future__ import annotations

import torch

from . import spec_tentative as ST
from . import verify_rows as VR
from .r2_positional_twin import BLOCK

UNEG = (None, "list_tentative")
SpecError = ST.SR.SpecError


def _merge(num, den, mx, o_c, l_c):
    """verify_rows' running log-sum-exp merge (float32): a part of length 0 (lse -inf, out 0) has the weight 0."""
    new = torch.maximum(mx, l_c)
    a, b_ = torch.exp(mx - new), torch.exp(l_c - new)
    return num * a + o_c * b_, den * a + b_, new


class TentStateU(ST.TentState):
    """The tentative buffers with the UNION joint pass (module docstring). union False: a joint pass runs row by row
    (L4PT's code; the negative control of the residency count). uneg: a negative control (UNEG), tests only."""

    def __init__(self, rs, attn, tneg=None, poison: bool = False, union: bool = True, uneg=None):
        if uneg not in UNEG:
            raise SpecError("unknown union control %r (one of %s)" % (uneg, UNEG))
        super().__init__(rs, attn, tneg=tneg, poison=poison)
        self.union, self.uneg = bool(union), uneg
        self.calls.update(resid=0, joint_resid=0, union_layers=0)     # host counts
        self._len = {}                                                 # E -> (E,) int32: the buffer call's length
        self._rin = {}                                                 # R -> the store calls' (row_in, in_tail)

    def _resid(self, l, eng, twin, ti, inj, act, drf, qb3):
        self.calls["resid"] += 1                                       # host count: one per residency call
        return super()._resid(l, eng, twin, ti, inj, act, drf, qb3)

    def _rows(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
        """A joint pass (R > 1): the union pass (or, union False, the rows one by one); a one-row pass: TentState's."""
        if plan.R == 1 or plan.vs is None:
            return super()._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        vs = plan.vs
        n0 = self.calls["resid"]
        if l == 0:
            vs.passes += 1
        if self.union:
            out = self._rows_union(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            self.calls["union_layers"] += 1
        else:                                                          # NEGATIVE CONTROL: row by row
            out = super()._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        n = self.calls["resid"] - n0
        self.calls["joint_resid"] += n
        vs.calls["resid"] += n
        return out

    def _len_e(self, E: int):
        if E not in self._len:
            self._len[E] = torch.full((E,), BLOCK, dtype=torch.int32, device=self.dev)
        return self._len[E]

    def _row_inputs(self, R: int):
        """(row_in, in_tail) for verify_rows._store_call: in_tail all True, so store slot 0 is never visible or read."""
        if R not in self._rin:
            self._rin[R] = (torch.zeros((self.B, R), dtype=torch.int64, device=self.dev),
                            torch.ones((self.B, R), dtype=torch.bool, device=self.dev))
        return self._rin[R]

    def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
        """Steps 1-5 of the module docstring for one layer of a joint pass. Returns (B, R, Hq * D)."""
        vs = plan.vs
        ops = vs.ops
        rs, B, R, bs, H, T, X = self.rs, self.B, plan.R, self.bs, self.H, self.topk, vs.X
        E = B * R
        Hq, Dh = q4.shape[2], q4.shape[3]
        chk, st = ops.check, ops.stats
        qc = self.qc
        qb = qc // bs
        qb3 = qb.view(1, B, 1)
        qb4 = qb.view(1, B, 1, 1)
        P = torch.stack(pbs, 1)                                        # (B, R): the rows' positions
        past = (P // bs) > qb.view(B, 1)                               # (B, R): past the committed block's edge
        hasq = (sel == qb4).any(-1)                                    # (H, B, R): the head selects block qb
        if self.hooks is not None and hasattr(self.hooks, "on_select"):    # CHECK-ONLY (WS-G), as TentState._row
            for r in range(R):
                self.hooks.on_select(l, plan.kind, r, pbs[r], plan.verify[r], sel[:, :, r])
        VR._mark(vs, l, "resid")
        # 2. THE RESIDENCY, one per layer and pass: row 0's selection (TentState._row's step 4 for row 0; kept)
        s0 = sel[:, :, 0]
        inj = (s0 > qb3) & ~hasq[:, :, 0:1]
        ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
        skip0 = self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)
        VR._mark(vs, l, "write")
        # 3. every row's K, V and cis row into the tentative buffer (row order)
        for r in range(R):
            self._write(l, twin, r, k4[:, r], v4[:, r], pbs[r], plan, past[:, r])
        VR._mark(vs, l, "union")
        # 4. the union list: every block of a verify row, every pool hit of a draft row; minus the window and the
        #    tentative blocks (ids above qb: their rows are in the buffer)
        full = plan.verify & plan.act                                  # (R, B)
        res = plan.draft & plan.act
        if self.uneg == "list_tentative":                              # NEGATIVE CONTROL: the tentative block listed
            sel_c, m_arg = sel, qb + 1
        else:
            sel_c, m_arg = torch.where(sel > qb4, vs.minus1, sel), qb
        lst, src_dev, src_host = VR._union_list(vs, eng, sel_c, full, res, m_arg)
        sel_sorted = torch.sort(sel_c, dim=-1).values.contiguous()    # (H, B, R, T): the rows' sets
        if ops.record:                                                 # tests: the byte check (wsb_world.union_bytes)
            vs._rec_fills = []
            vs.log.append(dict(kind="union", layer=l, sel=sel.clone(), win=eng._block_map.clone(), lst=lst.clone(),
                               m=qb.clone(), in_tail=~past, p_rows=P.clone(), full=full.clone(), res=res.clone(),
                               pool_map=(eng._pool_map.clone() if self.pool else None), k_gpu=eng._k_gpu.clone(),
                               v_gpu=eng._v_gpu.clone(), bias_gpu=eng._kv_bias_gpu.clone(), k_cpu=eng._k_cpu.clone(),
                               v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone(), fills=vs._rec_fills))
        # 5. call A: the committed window, a bias row per entry (row r's selection), the cut of TentState._row
        win = eng._block_map                                           # (H, B, T): row 0's selection; slot T - 1 = qb
        winR = win.unsqueeze(2).expand(H, B, R, T).contiguous()
        at = torch.searchsorted(sel_sorted, winR).clamp(max=T - 1)
        vis_a = (sel_sorted.gather(-1, at) == winR) & (winR >= 0)      # (H, B, R, T)
        if ops.neg == "union_attention":                               # NEGATIVE CONTROL (WS-B): every window block
            vis_a = winR >= 0
        vis_a[..., T - 1] = hasq | ~past.view(1, B, R)                 # the tail slot: block qb's committed rows
        n_tail = rs.body.cache_lens_value(eng, P) - self.base          # (B, R): rows of p_r's block the shipped step reads
        cq = (qc % bs).view(B, 1)
        cut = (self.base + torch.where(past, cq.expand(B, R), torch.minimum(cq, n_tail))).to(torch.int32).reshape(E)
        rows_a = T * bs
        m_a = vis_a.permute(1, 2, 3, 0).unsqueeze(3).expand(B, R, T, bs, H).reshape(B, R, rows_a, H)
        if self.pool:
            katt, vatt, batt = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
        else:
            katt, vatt, batt = eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu
        bias_a = torch.where(m_a, batt.unsqueeze(1), self.maskb).reshape(E, rows_a, H)
        q_e = q4.reshape(E, 1, Hq, Dh)
        nsplit = rs.nl._ATTN_SPLITS
        VR._mark(vs, l, "fetch")
        VR._store_fill(vs, eng, twin, src_dev, src_host, 0, X - 1)    # fill 0 (the extra store)
        VR._mark(vs, l, "attn")
        o_a, lse_a = self.attn(q_e, katt, vatt, bias_a[:B], cache_seqlens=cut, cache_batch_idx=vs.cbi,
                               num_splits=nsplit, return_softmax_lse=True)
        vs.calls["attn"] += 1
        picks = sorted({0, E // 3, (2 * E) // 3, E - 1})
        if vs.fa_check is not None and l == 0 and "A" not in vs.fa_check:     # CHECK-ONLY: VRF-IDX, the window call
            VR._fa_check_u(vs, "A", q_e, katt, vatt, bias_a, cut, o_a, picks)
        num = o_a.reshape(E, Hq, -1).float()
        mx = lse_a.reshape(E, Hq, 1).float()
        den = torch.ones_like(mx)
        row_in, in_all = self._row_inputs(R)
        vis_cnt = vis_a.sum(-1) if chk else None                       # CHECK-ONLY: (H, B, R) blocks seen
        if st:                                                         # ACCOUNTING
            streamed = (cut.view(B, R).to(torch.int64) * plan.act.t()).sum()   # active rows only
            used = torch.zeros((), dtype=torch.int64, device=self.dev)
            spill_used = torch.zeros((), dtype=torch.int64, device=self.dev)
        for c in range(vs.C):                                          # the store: fill 0, then the spill fills
            lo, hi = c * (X - 1), (c + 1) * (X - 1)
            if c == 1:
                VR._mark(vs, l, "spill")
            if c >= 1:
                VR._store_fill(vs, eng, twin, src_dev, src_host, lo, hi)
            o_c, l_c, lens_c, seen = VR._store_call(vs, c, lst[..., lo:hi], sel_sorted, q_e, row_in, in_all)
            num, den, mx = _merge(num, den, mx, o_c, l_c)
            if st:
                u_c = (lens_c > 0).any().to(torch.int64)
                used = used + u_c
                if c >= 1:
                    spill_used = spill_used + u_c                      # a spill fill (c >= 1) that served an entry
                streamed = streamed + lens_c.to(torch.int64).sum()
            if vis_cnt is not None:
                vis_cnt = vis_cnt + seen
        VR._mark(vs, l, "buffer")
        # call B: the tentative buffer, TentState._row's masks per entry
        tp = self.tpos                                                 # (B, 64)
        tp3 = tp.view(B, 1, bs)
        lo_ = ((tp >= 0) if self.tneg == "no_lower" else (tp >= qc.view(B, 1))).view(B, 1, bs)   # NEGATIVE CONTROLS
        hi_ = (tp3 >= 0) if self.tneg == "no_causal" else (tp3 <= P.view(B, R, 1))              # (tests only)
        tb3 = tp3 // bs
        own = tb3 == (P // bs).view(B, R, 1)
        ok = lo_ & hi_ & (~own | ((tp3 % bs) < n_tail.view(B, R, 1)))                          # (B, R, 64)
        cmt = (tb3 == qb.view(B, 1, 1)) & past.view(B, R, 1)
        vis_b = ok.unsqueeze(-1) & (~cmt.unsqueeze(-1) | hasq.permute(1, 2, 0).unsqueeze(2))    # (B, R, 64, H)
        bias_b = torch.where(vis_b, self.bias[l].unsqueeze(1), self.maskb).reshape(E, bs, H)
        len_e = self._len_e(E)
        o_b, lse_b = self.attn(q_e, self.k[l], self.v[l], bias_b[:B], cache_seqlens=len_e, cache_batch_idx=vs.cbi,
                               num_splits=nsplit, return_softmax_lse=True)
        vs.calls["attn"] += 1
        if vs.fa_check is not None and l == 0 and "T" not in vs.fa_check:     # CHECK-ONLY: VRF-IDX, the buffer call
            VR._fa_check_u(vs, "T", q_e, self.k[l], self.v[l], bias_b, len_e, o_b, picks)
        num, den, mx = _merge(num, den, mx, o_b.reshape(E, Hq, -1).float(), lse_b.reshape(E, Hq, 1).float())
        VR._mark(vs, l, "merge")
        o = num / den
        out = o.to(q4.dtype).reshape(B, R, -1)
        if st:                                                         # ACCOUNTING (device counters)
            s = vs.stat
            s["rows_full"] += full.sum()
            s["rows_res"] += res.sum()
            s["win_sel"] += ((s0 >= 0) & plan.act[0].view(1, B, 1)).sum()     # active requests only
            s["win_host"] += ((eng._load_mask >= 0) & ~skip0).sum()   # row 0's host loads (after the pool)
            s["fills_used"] += used
            s["spill_fills_used"] += spill_used
            s["streamed_rows"] += streamed + len_e.to(torch.int64).sum()
        if chk:                                                        # CHECK-ONLY: presence and the visible count
            own_b = (vis_b & own.unsqueeze(-1)).any(2).permute(2, 0, 1)   # (H, B, R): call B shows the own block
            vis_cnt = vis_cnt + (own_b & past.view(1, B, R))           # a past row's own tentative block (the buffer)
            found = VR._presence(vs, eng, sel, lst, qb)
            miss = ~found & (sel >= 0)
            f4 = full.t().reshape(1, B, R, 1)
            r4 = res.t().reshape(1, B, R, 1)
            d = miss & f4
            s_ = miss & r4
            actv = (f4 | r4)[..., 0]                                   # (1, B, R): the active rows
            n_sel = (sel >= 0).sum(-1)                                 # (H, B, R)
            want = n_sel - s_.sum(-1)
            bad = actv & (vis_cnt != want)
            s = vs.stat
            s["dropped"] += d.sum()
            s["skipped"] += s_.sum()
            s["vis_bad"] += bad.sum()
            s["visible_blocks"] += (vis_cnt * actv).sum()
            sr_ = vs.stat_row
            sr_["dropped"] += d.sum((0, 1, 3))
            sr_["skipped"] += s_.sum((0, 1, 3))
            sr_["vis_bad"] += bad.sum((0, 1))
            sr_["visible"] += (vis_cnt * actv).sum((0, 1))
            sr_["selected"] += (n_sel * actv).sum((0, 1))
            vs.chk_sel[l].copy_(sel)
        if ops.record:
            vs.log[-1].update(q_e=q_e.clone(), out=o.clone(), cut=cut.clone(), bias_a=bias_a.clone())
        VR._mark(vs, l, "post")
        return out


class TentRunnerU(ST.TentRunner):
    """spec_tentative.TentRunner (FAST, the schedule of contract 4) whose JOINT pass uses the union residency (arm L4PTU;
    module docstring). Level L4P only. union False: the joint pass row by row (negative control). union_check /
    union_stats: VerifyOps(check / stats) (CHECK-ONLY / accounting; off in the timing cells). spill: VerifyOps(spill)
    (tests only). vneg: a VerifyOps control of mode union (tests only); uneg: TentStateU's (tests only); vrecord:
    VerifyOps(record) (tests only). phase_marks: CUDA-event marks inside the joint pass of the PROBE rounds (the cell)."""

    def __init__(self, model, rs, ops, clock, t0=None, hooks=None, check_every: int = 1, poison: bool = False,
                 tneg=None, union: bool = True, union_check: bool = False, union_stats: bool = False, spill=None,
                 vneg=None, uneg=None, vrecord: bool = False, phase_marks: bool = False):
        if ops.level != "L4P":
            raise SpecError("L4PTU: level L4P (the union is a joint-pass mode; got %r)" % (ops.level,))
        if uneg not in UNEG:
            raise SpecError("unknown union control %r (one of %s)" % (uneg, UNEG))
        if vneg not in (None,) + VR.UNION_NEG:
            raise SpecError("vneg %r: one of %s" % (vneg, (None,) + VR.UNION_NEG))
        try:                                                           # every refusal before the meter attaches
            VR.VerifyOps(K=int(ops.K), mode="union", extra=ops.extra, neg=vneg, spill=spill, check=union_check,
                         stats=union_stats)
        except VR.VerifyError as e:
            raise SpecError("L4PTU: %s" % e)
        self._u = dict(union=bool(union), check=bool(union_check), stats=bool(union_stats), spill=spill, vneg=vneg,
                       uneg=uneg, record=bool(vrecord))
        self.phase_marks = bool(phase_marks)
        self.marks = []                                                # the probe rounds' joint-pass marks
        super().__init__(model, rs, ops, clock, t0=t0, hooks=hooks, check_every=check_every, poison=poison,
                         tneg=tneg)
        self.meta.update(joint_residency=("union" if union else "serial"),
                         union=dict(X=self.vs.X, fills=self.vs.C, cap=self.vs.cap, check=bool(union_check),
                                    stats=bool(union_stats), spill=spill, vneg=vneg, uneg=uneg, gemm=ops.gemm))

    def _joint_ops(self, R, gemm, base, extra, attn):
        u = self._u
        return VR.VerifyOps(K=R - 1, mode="union", gemm=gemm, linear=base, extra=extra, attn=attn, neg=u["vneg"],
                            record=u["record"], stats=u["stats"], spill=u["spill"], check=u["check"])

    def _make_state(self, rs, attn, tneg, poison):
        u = self._u
        return TentStateU(rs, attn, tneg=tneg, poison=poison, union=u["union"], uneg=u["uneg"])

    def _ps_round(self, P, probe: bool):
        """TentRunner's round; on a probe round (phase_marks) the joint pass records its phase marks."""
        mark = self.phase_marks and probe
        if mark:
            self.vs.marks, self.vs.clock = [], self.clock
        try:
            return super()._ps_round(P, probe)
        finally:
            if mark:
                self.marks.append(self.vs.marks)
                self.vs.marks = None

    @torch.inference_mode()
    def read(self) -> dict:
        """TentRunner.read() and, after it, ONE more host read: the union statistics and host counts (verify_rows
        read_stats)."""
        out = super().read()
        out.update(union=VR.read_stats(self.vs), joint_residency=self.meta["joint_residency"])
        return out

    @torch.inference_mode()
    def run_teacher(self, tokens, G: int, on_pass=None) -> dict:
        """CHECK-ONLY (the check cells, the CPU tests): the union joint pass TEACHER-FORCED. tokens (B, >= 1 + G R): one
        plain step feeds tokens[:, 0] at the twin position (the window's first load, as the runners' warm step), then G
        joint passes of R = K + 1 rows, EVERY row a verify row at qc .. qc + R - 1 with tokens[:, 1 + g R + r]; each pass
        is committed whole (its K / V become committed state, as SH's). on_pass(g, logits (B, R, V)) after each pass.
        Hooks and the check counters are on for every pass. Host reads: none (the caller reads)."""
        B, R = self.B, self.vs.R
        T = self.T
        tok = tokens.to(device=self.dev, dtype=torch.int64)
        if tok.dim() != 2 or tok.shape[0] != B or tok.shape[1] < 1 + G * R:
            raise SpecError("run_teacher: tokens (B, >= 1 + G R) = (%d, >= %d), got %s" % (B, 1 + G * R,
                                                                                           tuple(tok.shape)))
        self.round = -1
        T.checking, T.hooks, T.round = True, self.hooks, -1
        self.t = tok[:, 0].clone()
        self._step(self.t)                                             # the warm step: SH's step at the first position
        T.init_committed()
        ones = torch.ones((R, B), dtype=torch.bool, device=self.dev)
        zeros = torch.zeros((R, B), dtype=torch.bool, device=self.dev)
        m_all = torch.full((B,), R, dtype=torch.int64, device=self.dev)
        z = torch.zeros((B,), dtype=torch.int64, device=self.dev)
        vs = self.vs
        lin = lambda x, w: VR._lin(vs, x, w)                          # noqa: E731
        try:
            for g in range(G):
                self.round = g
                T.round = g
                ids = tok[:, 1 + g * R:1 + (g + 1) * R].contiguous()
                pos = (T.qc.view(B, 1) + self.arR[:R].view(1, R)).to(torch.int32)
                plan = ST._Plan(R=R, kind="joint", ids=ids, pos=pos, act=ones, verify=ones, draft=zeros, replay=None,
                                replay_rows=(), deferred=None, vs=vs, lin=lin, snap=True)
                lg = T.run_pass(self.model, plan)
                T.calls["joint_passes"] += 1
                T.calls["joint_rows"] += R
                T.commit(m_all, plan.ctr[R - 1], self.rs.st.twins[0].pos.clone(), z)
                if on_pass is not None:
                    on_pass(g, lg)
        finally:
            T.checking, T.hooks = False, None
        return dict(passes=G, R=R)
