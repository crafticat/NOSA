"""I3 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the integration rung I3 = I2
(spec_tentative_union.TentRunnerU: WS-A's tentative buffers on the schedule of contract 4 with WS-B's ONE residency per
layer for the union of the joint pass's rows) whose union joint pass uses WS-C's FORK for the attention: ONE call of the
shared-KV fork per layer over the union, each row masked to its own selection, with a spill-free store (C = 1). ARM
L4PTF. OPT-IN: benchmarks/Efficiency/r2_i3_cell.py (--joint-attn fork); no engine module and no older cell imports
this module. Docs: docs/NOTES_i3.md. NO NEW KERNEL: the fork is fia6's unified_attention (the caller passes it, as
verify_rows' fork path takes it; this module imports no kernel module), the shipped kernels that spec_tentative.py
already calls, and torch ops.

THE SEAM (docs/NOTES_i3.md section 2). TentStateU._rows (spec_tentative_union.py:104-120) calls self._rows_union for a
joint pass; TentStateF overrides _rows_union and keeps everything else of I2 (the residency of row 0, the buffer writes,
the host counts) and of I1 (the replay, the commit, the reset, the schedule). Of I2's _rows_union only the attention
changes: call A (the window, spec_tentative_union.py:197-205), the C store calls (:218-233), call B (the buffer,
:236-252) and the log-sum-exp merge become ONE fork call. Built from verify_rows' WS-C helpers, unchanged: _fork_locate
(where each selected block lives), _fork_lists (the row lists), _fork_call (one launch), _fork_probe (the "F" / "RA"
probes); _fork_state (VerifyOps(joint_attn="fork")) makes the store and the fork's tables.

WHERE THE TENTATIVE ROWS GO (the decision; docs/NOTES_i3.md section 2). INTO THE ARENA, not a separate call merged by
the log-sum-exp: the fork's default 2D launch writes the NORMALISED output only (fork :1060-1063: the segment pointers
are the output itself; no log-sum-exp leaves the kernel), so a separate buffer call could not be merged without the 3D
split path or a kernel edit (NO NEW KERNEL). The fork reads a block's tokens 0 .. n - 1 only (n = 64, or last_len on the
row's last listed page), and block qb = qc // 64 is split: its committed rows in the window's TAIL SLOT, its tentative
rows in the BUFFER (D1: a tentative row never enters the tail slot). So per layer and joint pass two SCRATCH blocks per
request are written (benchmarks/Efficiency/r2_i3_arena.py, ChainArena):
  CB0  rows x < qc % 64 from the tail slot (committed), rows x >= qc % 64 from the buffer row x (block qb's chain)
  CB1  (store slot 0, WS-C's "next block" slot) the buffer: the chain's rows of block qb + 1
and every row lists CB0 in place of block qb and CB1 in place of block qb + 1 (its other blocks: the window, a pool
slot read in place, or the store's listed host blocks, as WS-C). A row at p sees CB0's rows 0 .. n_tail(p) - 1 when p
is in block qb; past the edge it sees CB0 whole when its head selects qb, and CB1's rows 0 .. n_tail(p) - 1: TentState.
_row's call A cut (base + min(qc % 64, n_tail)) and call B masks (qc <= x <= p, x % 64 < n_tail in p's block, block qb
past the edge only for a head that selects it), exactly (F4 kept: n_tail 0 at a block's last position). Nothing is
written back: the committed state changes only at the round's commit (TentState.commit), as in L4PT. The window's tail
slot is never listed (CHECK-ONLY stat tail_reads: 0).

ONE JOINT PASS, per layer (R = K + 1 rows per request at their own positions p_r):
  1. hooks (CHECK-ONLY): on_select per row (as TentStateU).
  2. THE RESIDENCY, ONE per layer and pass: row 0's selection (TentState._resid; kept, D1). As TentStateU.
  3. every row's K, V and cis row into the TENTATIVE BUFFER (TentState._write; D3's counters). As TentStateU.
  4. the chain images CB0 and CB1 (torch.where / copy_: 3 + 3 ops).
  5. where each other selected block lives (verify_rows._fork_locate with block qb and qb + 1 removed from the rows'
     selections) and ONE store fill of the listed host blocks (the shipped gathers; store slot 0 = CB1 is never filled:
     its id stays -1). The store holds X - 1 >= (full_max - 1) x 64 listed blocks (the bound; VerifyState refuses less):
     no spill, nothing dropped. full_max = K: in contract 4 a request has at most K verify rows in a pass (group K - 1,
     final <= K, target 1) and they are rows 0 .. v - 1, so row 0's residency already holds a full row's blocks.
  6. the row lists (verify_rows._fork_lists: the PRESENT blocks of each (row, KV head) in ascending block id, their
     arena block ids; last_len = n_tail(p)) and ONE fork call for every row of every request.
CHECK-ONLY (VerifyOps(check=True): the check cells, the miniature, the CPU tests): the selections (vs.chk_sel); PRESENT
(dropped: a verify row's selected block not present); VIS and VB from the tables the KERNEL wrote (each row sees exactly
its own selection as (block id, tokens); the tail slot is never read: tail_reads); CHAIN (chain_bad: CB0's committed
rows = the tail slot's bits, CB0's other rows and CB1 = the buffer's bits, and every buffer row a row sees holds the
chain's position of its block: tpos); "F" / "RA" probes (vs.fa_check, layer 0, once). ACCOUNTING (VerifyOps(stats)):
roles, the window's loads, the listed host blocks, pool blocks read in place, rows past an edge, F4 rows.
HOST COUNTS (always): TentStateU's resid / joint_resid / union_layers; vs.calls (attn: ONE per layer, fills: ONE per
layer). PHASE MARKS (probe rounds): "resid", "write", "chain", "union" (locate), "fetch" (the fill), "attn" (the lists
and the fork call), "check", "post" (to the next layer's first mark). HOST SYNCS: none in a pass (the probes excepted:
check-only, once).
NEGATIVE CONTROLS (tests only): fneg "window_tail" (block qb read from the window's tail slot: the committed rows only,
the tentative rows of qb unseen); spec_tentative's tneg no_causal (every row sees its own block whole: last_len 64),
no_lower (CB0 takes every row from the buffer: no committed rows), recompute_deferred, undo_residency,
writeback_tentative, no_replay (the shared code); VerifyOps neg fork_union_rows, fork_small_store, fork_probe_swap.
"""
from __future__ import annotations

import torch

from . import spec_tentative as ST
from . import spec_tentative_union as STU
from . import verify_rows as VR
from .r2_positional_twin import BLOCK

FNEG = (None, "window_tail")
CHAIN_STATS = ("chain_bad", "chain_rows", "tail_reads", "chain_reads")
SpecError = ST.SR.SpecError


def _bits(t):
    """The bits of a float tensor as integers (NaN-safe equality)."""
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


class TentStateF(STU.TentStateU):
    """The tentative buffers with the union joint pass and ONE fork call per layer (module docstring). fneg: a negative
    control (FNEG), tests only."""

    def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None):
        if fneg not in FNEG:
            raise SpecError("unknown fork control %r (one of %s)" % (fneg, FNEG))
        super().__init__(rs, attn, tneg=tneg, poison=poison, union=True, uneg=None)
        arena = getattr(rs, "kv_arena", None)
        if arena is None or not hasattr(arena, "cb0_base"):
            raise SpecError("L4PTF needs the chain arena on the state (rs.kv_arena: r2_i3_arena.ChainArena)")
        B, bs = self.B, self.bs
        for t, want in ((arena.cb0_k, self.k[0]), (arena.cb0_v, self.v[0]), (arena.cb0_bias, self.bias[0])):
            if tuple(t.shape) != tuple(want.shape) or t.dtype != want.dtype or t.device != want.device:
                raise SpecError("CB0 %s %s does not fit the buffer %s %s" % (tuple(t.shape), t.dtype,
                                                                            tuple(want.shape), want.dtype))
        self.arena, self.fneg = arena, fneg
        self.cb0_id = (arena.cb0_base + torch.arange(B, dtype=torch.int64, device=self.dev)).view(1, B, 1, 1)
        self.cb1 = arena.cb1_views()
        if tuple(self.cb1[0].shape) != (B, bs, self.H, self.D):
            raise SpecError("store slot 0 is not one block per request")
        self.calls.update(chain=0)                                      # host count: chain images written

    # -------------------------------------------------------------- the chain images (step 4)
    def _chain(self, l, eng) -> None:
        """CB0 <- tail slot rows below qc % 64 + the buffer's rows from it; CB1 (store slot 0) <- the buffer."""
        B, bs, t0 = self.B, self.bs, self.base
        a = self.arena
        if self.tneg == "no_lower":                                    # NEGATIVE CONTROL: no committed rows in CB0
            low = torch.zeros((B, bs), dtype=torch.bool, device=self.dev)
        else:
            low = self.ar64.view(1, bs) < (self.qc % bs).view(B, 1)    # (B, 64): the committed rows of block qb
        l4, l3 = low.view(B, bs, 1, 1), low.view(B, bs, 1)
        torch.where(l4, eng._k_gpu[:, t0:t0 + bs], self.k[l], out=a.cb0_k)
        torch.where(l4, eng._v_gpu[:, t0:t0 + bs], self.v[l], out=a.cb0_v)
        torch.where(l3, eng._kv_bias_gpu[:, t0:t0 + bs], self.bias[l], out=a.cb0_bias)
        ck, cv, cb = self.cb1
        ck.copy_(self.k[l])
        cv.copy_(self.v[l])
        cb.copy_(self.bias[l])
        self.calls["chain"] += 1

    # -------------------------------------------------------------- the joint pass (steps 1-6)
    def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
        """Steps 1-6 of the module docstring for one layer of a joint pass. Returns (B, R, Hq * D): a VIEW of the
        persistent vs.fout (consumed by TentState._layer's output GEMM before the next layer's fork call)."""
        vs = plan.vs
        ops = vs.ops
        rs, B, R, bs = self.rs, self.B, plan.R, self.bs
        E = B * R
        Hq, Dh = q4.shape[2], q4.shape[3]
        chk, st = ops.check, ops.stats
        kern = rs.ops.kernels
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
        # 2. THE RESIDENCY, one per layer and pass: row 0's selection (as TentStateU; kept, D1)
        s0 = sel[:, :, 0]
        inj = (s0 > qb3) & ~hasq[:, :, 0:1]
        ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
        skip0 = self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)
        VR._mark(vs, l, "write")
        # 3. every row's K, V and cis row into the tentative buffer (row order)
        for r in range(R):
            self._write(l, twin, r, k4[:, r], v4[:, r], pbs[r], plan, past[:, r])
        VR._mark(vs, l, "chain")
        # 4. the chain images
        self._chain(l, eng)
        VR._mark(vs, l, "union")
        # 5. where the other selected blocks live; ONE store fill (store slot 0 = CB1 keeps its id -1)
        full = plan.verify & plan.act                                  # (R, B)
        res = plan.draft & plan.act
        if self.fneg == "window_tail":                                 # NEGATIVE CONTROL: block qb from the tail slot
            sel_c = torch.where(sel > qb4, vs.minus1, sel)
            is_q = torch.zeros_like(sel, dtype=torch.bool)
        else:
            sel_c = torch.where(sel >= qb4, vs.minus1, sel)
            is_q = sel == qb4
        is_n = sel == qb4 + 1
        loc = VR._fork_locate(vs, eng, sel_c, full, res, qb, l)
        present = loc["present"] | is_q | is_n
        phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), loc["phys"]))
        VR._mark(vs, l, "fetch")
        vs.host_x[..., 1:].copy_(torch.where(loc["lst"] < VR.FORK_BIG, loc["lst"], vs.minus1))
        kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
        kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
        vs.calls["gather_host"] += 2
        vs.calls["fills"] += 1
        VR._mark(vs, l, "attn")
        # 6. the row lists and ONE fork call for every row of every request (each row masked to its own list)
        VR._fork_lists(vs, sel, dict(present=present, phys=phys), eng, P)
        if self.tneg == "no_causal":                                   # NEGATIVE CONTROL: the own block whole
            vs.plast.fill_(bs)
        qf = q4.reshape(E, Hq, Dh).contiguous()
        VR._fork_call(vs, qf, vs.fout, vs.fork_cu, vs.fork_seq, vs.fork_tab, vs.fork_ext, vs.fork_len, vs.lists,
                      vs.fork_qmax)
        vs.calls["attn"] += 1
        out = vs.fout.reshape(B, R, -1)
        if st:                                                         # ACCOUNTING (device counters)
            s = vs.stat
            roles = (full | res).t()                                   # (B, R)
            s["rows_full"] += full.sum()
            s["rows_res"] += res.sum()
            s["win_sel"] += ((s0 >= 0) & plan.act[0].view(1, B, 1)).sum()
            s["win_host"] += ((eng._load_mask >= 0) & ~skip0).sum()
            s["past_edge"] += (past & roles).sum()
            s["f4_rows"] += ((~past) & (P % bs == bs - 1) & roles).sum()
        if ops.record:                                                 # tests: the byte checks (i3_world)
            vs.log.append(dict(kind="fork_t", layer=l, sel=sel.clone(), win=eng._block_map.clone(),
                               lst=loc["lst"].clone(), present=present.clone(), phys=phys.clone(), qb=qb.clone(),
                               qc=qc.clone(), p_rows=P.clone(), past=past.clone(), full=full.clone(), res=res.clone(),
                               pool_map=(eng._pool_map.clone() if self.pool else None), k_gpu=eng._k_gpu.clone(),
                               v_gpu=eng._v_gpu.clone(), bias_gpu=eng._kv_bias_gpu.clone(), k_cpu=eng._k_cpu.clone(),
                               v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone(), tpos=self.tpos.clone(),
                               buf_k=self.k[l].clone(), buf_v=self.v[l].clone(), buf_bias=self.bias[l].clone(),
                               store_k=vs.ext_k.clone(), store_v=vs.ext_v.clone(), store_bias=vs.ext_bias.clone(),
                               cb0_k=self.arena.cb0_k.clone(), cb0_v=self.arena.cb0_v.clone(),
                               cb0_bias=self.arena.cb0_bias.clone(), tab=vs.fork_tab.clone(), ext=vs.fork_ext.clone(),
                               ulen=vs.fork_len.clone(), pkey=vs.pkey.clone(), pslot=vs.pslot.clone(),
                               npg=vs.npg.clone(), plast=vs.plast.clone(), q_e=qf.clone(), out=vs.fout.float().clone(),
                               base=self.arena.layer_base(l), store_base=self.arena.store_base,
                               cb0_base=self.arena.cb0_base, X=vs.X))
        VR._mark(vs, l, "check")
        if chk:                                                        # CHECK-ONLY
            vs.chk_sel[l].copy_(sel)
            miss = ~present & (sel >= 0)
            f4 = full.t().reshape(1, B, R, 1)
            r4 = res.t().reshape(1, B, R, 1)
            d = miss & f4
            s_ = miss & r4
            s = vs.stat
            s["dropped"] += d.sum()
            s["skipped"] += s_.sum()
            vs.stat_row["dropped"] += d.sum((0, 1, 3))
            vs.stat_row["skipped"] += s_.sum((0, 1, 3))
            vis_cnt, bad, want, role, b0, tail = self._vb(vs, eng, sel, present, loc["lst"], full, res, P, qb, l)
            H = sel.shape[0]
            vbad = role & (vis_cnt != want)
            s["vis_bad"] += vbad.sum()
            s["vb_bad"] += bad.sum()
            s["vb_rows"] += role.sum()
            s["block0_bad"] += b0.sum()
            s["tail_reads"] += tail
            s["visible_blocks"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).sum()
            s["fork_loaded"] += vs.fork_len.to(torch.int64).sum()
            vs.stat_row["vis_bad"] += vbad.view(H, B, R).sum((0, 1))
            vs.stat_row["visible"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).view(H, B, R).sum((0, 1))
            vs.stat_row["selected"] += (sel >= 0).sum(-1).sum((0, 1))
            cbad, crows, creads = self._chain_check(l, eng, sel, P, qb, past, full | res)
            s["chain_bad"] += cbad
            s["chain_rows"] += crows
            s["chain_reads"] += creads
        if vs.fa_check is not None and l == 0 and "F" not in vs.fa_check:
            VR._fork_probe(vs, qf)                                     # CHECK-ONLY: "F" and "RA" (one host sync)
        VR._mark(vs, l, "post")
        return out

    # -------------------------------------------------------------- CHECK-ONLY
    def _vb(self, vs, eng, sel, present, lst, full, res, P, qb, l):
        """VB and VIS from the tables that the KERNEL wrote (verify_rows' WS-C check with the chain blocks): every
        union entry (head, sequence, u < len) is mapped back to (request, block id) through THIS layer's maps (window
        slots other than the tail slot, pool), the store (slot 0 = CB1 = block qb + 1, slot 1 + j = the j-th listed
        block) and CB0 (= block qb). Per (head, row) the multiset of (block id, tokens > 0) that the row's byte shows
        must equal the row's own: a full row every selected block, a resident row its present blocks; 64 tokens, its
        own block (p // 64) n_tail(p) (F4: 0, not shown); no entry of another request; no read of the window's tail
        slot. Returns (vis (H, E), bad (H, E), want (H, E), role (H, E), b0 (H, E), tail_reads ())."""
        rs = self.rs
        H, B, R, T = sel.shape
        E, X, ns, topk, bs = vs.E, vs.X, vs.ns, vs.topk, vs.bs
        a = self.arena
        i64 = torch.int64
        dev = sel.device
        tab = vs.fork_tab.to(i64)
        ext = vs.fork_ext.to(i64)
        NS, Wt = tab.shape[1], tab.shape[2]
        live = torch.arange(Wt, dtype=i64, device=dev).view(1, 1, Wt) < vs.fork_len.to(i64).unsqueeze(-1)
        loc_e = tab - a.layer_base(l)
        on_eng = (loc_e >= 0) & (loc_e < B * ns)
        loc_s = tab - a.store_base
        on_st = (loc_s >= 0) & (loc_s < B * X)
        loc_c = tab - a.cb0_base
        on_c = (loc_c >= 0) & (loc_c < B)
        minus = torch.full_like(tab, -1)
        b_of = torch.where(on_eng, loc_e // ns, torch.where(on_st, loc_s // X, torch.where(on_c, loc_c, minus)))
        is_tail = on_eng & ((loc_e % ns) == topk - 1)                  # the committed tail slot: never listed (D1)
        maps = torch.cat((eng._block_map, eng._pool_map), dim=2) if rs.pool else eng._block_map   # (H, B, NS)
        id_eng = maps.reshape(H, B * ns).gather(-1, loc_e.clamp(0, B * ns - 1).reshape(H, NS * Wt)).view(H, NS, Wt)
        sid = torch.cat(((qb + 1).view(1, B, 1).expand(H, B, 1),
                         torch.where(lst < VR.FORK_BIG, lst, vs.minus1)), dim=2)            # (H, B, X)
        id_st = sid.reshape(H, B * X).gather(-1, loc_s.clamp(0, B * X - 1).reshape(H, NS * Wt)).view(H, NS, Wt)
        id_c = qb.gather(0, loc_c.clamp(0, B - 1).reshape(-1)).view(H, NS, Wt)
        id_of = torch.where(on_eng & ~is_tail, id_eng,
                            torch.where(on_st, id_st, torch.where(on_c, id_c, torch.full_like(tab, -2))))
        kid = torch.full((H, E, Wt), VR.FORK_BIG, dtype=i64, device=dev)
        ktok = torch.zeros((H, E, Wt), dtype=i64, device=dev)
        xreq = torch.zeros((H, E), dtype=torch.bool, device=dev)
        tail = torch.zeros((), dtype=i64, device=dev)
        for j, seqs, rows in vs.fork_slots:
            byte = (ext[:, seqs] >> (8 * j)) & 0xFF                     # (H, n_j, Wt)
            vis = live[:, seqs] & (byte > 0)
            kid[:, rows] = torch.where(vis, id_of[:, seqs], vs.big)
            ktok[:, rows] = torch.where(vis, byte, torch.zeros_like(byte))
            xreq[:, rows] = (vis & (b_of[:, seqs] != vs.seq_req[seqs].view(1, -1, 1))).any(-1)
            tail = tail + (vis & is_tail[:, seqs]).sum()
        kid_s, ko = torch.sort(kid, dim=-1)
        ktok_s = ktok.gather(-1, ko)
        vis_cnt = (kid < VR.FORK_BIG).sum(-1)
        cur = (P // bs).view(1, B, R, 1)
        lastv = (rs.body.cache_lens_value(eng, P) - self.base).view(1, B, R, 1).to(i64)
        tok = torch.where(sel == cur, lastv, torch.full_like(sel, bs))
        f4 = full.t().reshape(1, B, R, 1)
        r4 = res.t().reshape(1, B, R, 1)
        want_m = (sel >= 0) & (f4 | (r4 & present)) & (tok > 0)
        eid = torch.full((H, E, Wt), VR.FORK_BIG, dtype=i64, device=dev)
        etok = torch.zeros((H, E, Wt), dtype=i64, device=dev)
        eid[..., :T] = torch.where(want_m, sel, vs.big).reshape(H, E, T)
        etok[..., :T] = torch.where(want_m, tok, torch.zeros_like(tok)).reshape(H, E, T)
        eid_s, eo = torch.sort(eid, dim=-1)
        etok_s = etok.gather(-1, eo)
        role = (full | res).t().reshape(1, E).expand(H, E)
        bad = (((kid_s != eid_s) | (ktok_s != etok_s)).any(-1) | xreq) & role
        want = want_m.sum(-1).reshape(H, E)
        b0 = role & (vis_cnt > 0) & (kid_s[..., 0] != 0)               # G_block0: block 0 is the row's first entry
        return vis_cnt, bad, want, role, b0, tail

    def _chain_check(self, l, eng, sel, P, qb, past, roles):
        """CHECK-ONLY: the chain images against their sources, independently of how they were built: CB0's rows below
        qc % 64 = the tail slot's bits, CB0's other rows = the buffer's bits, CB1 = the buffer's bits (every row of
        every request; a wrong row counts 1); and every buffer row that a row with a role SEES (CB0 rows qc % 64 ..
        n0 - 1, CB1 rows 0 .. n1 - 1; n the tokens the row must see of block qb / qb + 1) holds the chain's position
        of its block (tpos = 64 qb + x, resp. 64 (qb + 1) + x). Returns (bad, rows checked, chain reads)."""
        B, bs, t0 = self.B, self.bs, self.base
        a = self.arena
        i64 = torch.int64
        x = self.ar64.view(1, bs)
        cq = (self.qc % bs).view(B, 1)
        low = x < cq                                                   # (B, 64)

        def same(u, w):                                                # (B, 64): every bit of the row equal
            d = _bits(u) == _bits(w)
            return d.reshape(B, bs, -1).all(-1)
        tk, tv, tb = eng._k_gpu[:, t0:t0 + bs], eng._v_gpu[:, t0:t0 + bs], eng._kv_bias_gpu[:, t0:t0 + bs]
        bk, bv, bb = self.k[l], self.v[l], self.bias[l]
        c0 = torch.where(low, same(a.cb0_k, tk) & same(a.cb0_v, tv) & same(a.cb0_bias, tb),
                         same(a.cb0_k, bk) & same(a.cb0_v, bv) & same(a.cb0_bias, bb))
        ck, cv, cb = self.cb1
        c1 = same(ck, bk) & same(cv, bv) & same(cb, bb)
        bad = (~c0).sum() + (~c1).sum()
        tp = self.tpos.view(1, B, 1, bs)
        x4 = x.view(1, 1, 1, bs)
        qb4 = qb.view(1, B, 1, 1)
        has_q = (sel == qb4).any(-1, keepdim=True)                     # (H, B, R, 1)
        has_n = (sel == qb4 + 1).any(-1, keepdim=True)
        R_ = int(P.shape[1])
        nt = (self.rs.body.cache_lens_value(eng, P) - self.base).view(1, B, R_, 1)
        n0 = torch.where(has_q, torch.where(~past.view(1, B, R_, 1), nt, torch.full_like(nt, bs)), torch.zeros_like(nt))
        n1 = torch.where(has_n, nt, torch.zeros_like(nt))
        role = roles.t().reshape(1, B, R_, 1)
        vis0 = (x4 >= cq.view(1, B, 1, 1)) & (x4 < n0) & role
        vis1 = (x4 < n1) & role
        ok0 = tp == qb4 * bs + x4
        ok1 = tp == (qb4 + 1) * bs + x4
        bad = bad + (vis0 & ~ok0).sum() + (vis1 & ~ok1).sum()
        rows = vis0.sum() + vis1.sum()
        reads = ((has_q | has_n) & role).sum()
        return bad.to(i64), rows.to(i64), reads.to(i64)


class TentRunnerF(STU.TentRunnerU):
    """spec_tentative.TentRunner (FAST, the schedule of contract 4) whose JOINT pass is the union pass with ONE fork
    call per layer (arm L4PTF; module docstring). Level L4P only. fork: the fork's unified_attention (the GPU:
    benchmarks/Efficiency/fork_vllm_unified_attention.py; the CPU: a stand-in). rs.kv_arena: an r2_i3_arena.ChainArena
    whose store holds 1 + (full_max - 1) x 64 slots (full_max: the most verify rows per request and pass; None = K,
    the schedule's; run_teacher needs K + 1). union_check / union_stats: VerifyOps(check / stats) (CHECK-ONLY /
    accounting; off in the timing cells). vneg: a VerifyOps control of the fork (FORK_NEG), fneg: TentStateF's,
    tneg: spec_tentative's (tests only); vrecord: VerifyOps(record) (tests only). phase_marks: CUDA-event marks in the
    joint pass of the PROBE rounds (the cell)."""

    def __init__(self, model, rs, ops, clock, fork, t0=None, hooks=None, check_every: int = 1, poison: bool = False,
                 tneg=None, union_check: bool = False, union_stats: bool = False, full_max=None, vneg=None, fneg=None,
                 vrecord: bool = False, phase_marks: bool = False):
        if ops.level != "L4P":
            raise SpecError("L4PTF: level L4P (got %r)" % (ops.level,))
        if fneg not in FNEG:
            raise SpecError("unknown fork control %r (one of %s)" % (fneg, FNEG))
        if vneg not in (None,) + VR.FORK_NEG:
            raise SpecError("vneg %r: one of %s" % (vneg, (None,) + VR.FORK_NEG))
        if fork is None or not callable(fork):
            raise SpecError("L4PTF takes the fork's unified_attention (fork=...)")
        arena = getattr(rs, "kv_arena", None)
        if arena is None or not hasattr(arena, "cb0_base"):
            raise SpecError("L4PTF needs rs.kv_arena = an r2_i3_arena.ChainArena (r2_kv_arena.attach)")
        K = int(ops.K)
        R = K + 2 if tneg == "recompute_deferred" else K + 1
        fm = K if full_max is None else int(full_max)
        try:                                                           # every refusal before the meter attaches
            VR.VerifyOps(K=R - 1, mode="union", extra=ops.extra, neg=vneg, check=union_check, stats=union_stats,
                         joint_attn="fork", fork=fork, full_max=min(fm, R))
        except VR.VerifyError as e:
            raise SpecError("L4PTF: %s" % e)
        topk = int(rs.cache.layers[0].cache_engine.topk)
        if int(arena.X) - 1 < (min(fm, R) - 1) * topk and vneg != "fork_small_store":
            raise SpecError("L4PTF: the arena's store lists %d blocks per (head, request), below the bound (%d - 1) x "
                            "%d (X >= %d)" % (int(arena.X) - 1, min(fm, R), topk, 1 + (min(fm, R) - 1) * topk))
        self._u = dict(union=True, check=bool(union_check), stats=bool(union_stats), spill=None, vneg=vneg, uneg=None,
                       record=bool(vrecord))
        self._f = dict(fork=fork, full_max=fm, fneg=fneg)
        self.phase_marks = bool(phase_marks)
        self.marks = []                                                # the probe rounds' joint-pass marks
        ST.TentRunner.__init__(self, model, rs, ops, clock, t0=t0, hooks=hooks, check_every=check_every,
                               poison=poison, tneg=tneg)
        z = lambda: torch.zeros((), dtype=torch.int64, device=self.dev)    # noqa: E731
        self.vs.stat.update({k: z() for k in CHAIN_STATS})
        self.meta.update(joint_residency="union", joint_attn="fork",
                         union=dict(X=self.vs.X, fills=self.vs.C, cap=self.vs.cap, bound=self.vs.bound,
                                    full_max=self.vs.meta.get("full_max"), check=bool(union_check),
                                    stats=bool(union_stats), vneg=vneg, fneg=fneg, gemm=ops.gemm,
                                    fork=getattr(fork, "__module__", None), arena=arena.describe()))

    def _joint_ops(self, R, gemm, base, extra, attn):
        u, f = self._u, self._f
        return VR.VerifyOps(K=R - 1, mode="union", gemm=gemm, linear=base, extra=extra, attn=attn, neg=u["vneg"],
                            record=u["record"], stats=u["stats"], check=u["check"], joint_attn="fork",
                            fork=f["fork"], full_max=min(f["full_max"], R))

    def _make_state(self, rs, attn, tneg, poison):
        return TentStateF(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"])

    @torch.inference_mode()
    def run_teacher(self, tokens, G: int, on_pass=None, warm: int = 1) -> dict:
        """CHECK-ONLY (the check cells, the CPU tests): TentRunnerU.run_teacher with `warm` plain steps first (tokens[:,
        0 .. warm - 1] at the twin positions; warm 1 = TentRunnerU's), then G joint passes of R = K + 1 VERIFY rows at
        qc .. qc + R - 1 with tokens[:, warm + g R + r], each committed whole. warm > 1 moves the passes against the
        64-token block edges (WS-C2's --check-phase: rows past an edge in the check). Needs full_max = R (every row a
        verify row: the store bound 1 + (R - 1) x 64). on_pass(g, logits (B, R, V)) after each pass."""
        B, R = self.B, self.vs.R
        T = self.T
        if int(self.vs.meta.get("full_max") or 0) < R:
            raise SpecError("run_teacher: every row is a verify row: build the runner with full_max = %d (the store "
                            "bound 1 + %d x 64)" % (R, R - 1))
        warm = int(warm)
        tok = tokens.to(device=self.dev, dtype=torch.int64)
        if warm < 1 or tok.dim() != 2 or tok.shape[0] != B or tok.shape[1] < warm + G * R:
            raise SpecError("run_teacher: warm >= 1 and tokens (B, >= warm + G R) = (%d, >= %d), got %s" % (
                B, warm + G * R, tuple(tok.shape)))
        self.round = -1
        T.checking, T.hooks, T.round = True, self.hooks, -1
        for i in range(warm):                                          # SH's steps at the first positions
            self.t = tok[:, i].clone()
            self._step(self.t)
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
                ids = tok[:, warm + g * R:warm + (g + 1) * R].contiguous()
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
        return dict(passes=G, R=R, warm=warm)


def teacher_facts(S0: int, warm: int, G: int, R: int, bs: int = BLOCK) -> dict:
    """The data precondition of the check arm (every request starts at S0): per request, the rows of run_teacher's
    passes past a 64-token block edge (a row whose block is not its pass's row-0 block) and the F4 rows (an in-block
    row at p % 64 = 63), and the passes. Pure (no torch)."""
    past = f4 = 0
    for g in range(int(G)):
        p0 = int(S0) + int(warm) + g * int(R)
        for r in range(int(R)):
            p = p0 + r
            if p // bs != p0 // bs:
                past += 1
            elif p % bs == bs - 1:
                f4 += 1
    return dict(passes=int(G), past_edge_rows=past, f4_rows=f4, first=int(S0) + int(warm),
                last=int(S0) + int(warm) + int(G) * int(R) - 1)
