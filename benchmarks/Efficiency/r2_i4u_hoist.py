"""Rung I4U of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), lever L5 of
KS/analysis/plan/LEDGER_model_vs_measured.md R5.5 item 5 (rev 6 R6.6 E3: "the union lists built once per pass where the
layer does not change them"): ARM L4PTFDU = L4PTFD (rung I4A, r2_i4a_lean.py) whose JOINT pass computes its
LAYER-INVARIANT tensors ONCE per pass (at layer 0) and reuses them in all L layers. OPT-IN: r2_i4u_cell.py (arm
L4PTFDU); no nosi module and no older cell imports this file. NO NEW KERNEL and no new kernel launch kind: the same
kernels and torch ops as L4PTFD's joint pass, fewer of them.

WHY A CLASS FACTORY (as r2_i4a_lean.lean_classes): hoist_classes(STF, base) builds the classes on the GIVEN
spec_tentative_fork module and the GIVEN lean pair base = r2_i4a_lean.lean_classes(STF); this file imports torch only.

WHAT DEPENDS ON THE LAYER (kept per layer). The selection sel (H, B, R, T) is a new tensor in every layer (it reads the
layer's own hidden state), and the window map eng._block_map, the pool map eng._pool_map and the K / V / bias rows are
the layer's own. So every op that reads sel, a map or a row stays per layer: hasq, the tail-id injection, the
residency, the buffer writes' gathers / wheres / index_puts, the chain images' copies, sel_c, is_q, is_n, the window /
pool / list searches, the host list (sort, first, cumsum, scatter), present, phys, the row lists (sort, gather, counts)
and the fork call.

WHAT DOES NOT DEPEND ON THE LAYER (hoisted). Inside a joint pass qc does not move (only commit moves it), every layer's
twin steps at the same positions (so pbs[r] = twin.pos before row r is the same tensor value in every layer; r2_i4a_lean
relies on the same fact), and the plan's masks are fixed. CODE references (nosi at acbe433):
  per pass (computed at layer 0 into persistent buffers):
    STF:135-139      qb = qc // 64, P = stack(pbs), past = P // 64 > qb                                     (4 ops)
    STF:167          qb + 1 (is_n's block id; also _fork_locate's m + 1, VR:1117)                           (1 + 1 ops)
    ST:459-460       ~act (the residency's inactive mask; masked_fill_ input)                              (1 op)
    ST:491, :496     p % 64 and p.clamp(...) of every row's buffer write                                   (2 R ops)
    STF:112          the chain mask x < qc % 64                                                            (2 ops)
    STF:159-160      full = verify & act, res = draft & act                                                 (2 ops)
    VR:1130          fullv = full expanded over (H, R x T) (a reshape copy; the CPU counter shows no op for it)
    VR:1209          the row lists' last_len = n_tail(p) of every row (vs.plast): 5 ops, written ONCE per pass
  per runner (static):
    VR:1146          eb = arena.layer_base(l) + slot_b (one tensor per layer), VR:1148 store_b + 1
    VR:1140-1141     torch.full_like(rank, cap), torch.full_like(s, FORK_BIG): 0-dim constants instead
  constant (the same value, False, in every layer):
    VR:1117-1148     is_next = (v == qb + 1): _fork_locate reads sel_c, whose ids >= qb are -1 (STF:165; the
                     window_tail control STF:162: ids > qb), so no id equals qb + 1 and is_next is False everywhere.
                     Its ops (eq, and, the four "& ~is_next" (8 ops), the where and the "| is_next") are dropped; the
                     values that read it are unchanged (x & True = x, x | False = x, where(False, a, b) = b).
                     CHECK-ONLY counter next_seen (check cells): the dropped test, recounted; it must stay 0.
  the residency (ST:470): tent = (lm >= 0) & (lm > qb3) = lm > qb3 (qb3 >= 0), as r2_i4a_lean's _resid_lean.
Removed per layer >= 1: SAVED_PER_LAYER aten compute ops (a TorchDispatchMode counter, tests_cpu/test_i4u.py); layer 0
adds the hoist (SAVED_LAYER0 is its net count, can be negative). Every value is L4PTFD's BIT FOR BIT (tokens, logits,
buffers, tpos, the engine, the twins, the counters, the trace, the fork's lists and tables).

CUDA GRAPHS (I4G, nosi-r2-i4g/benchmarks/Efficiency/r2_i4g_graph.py: one graph per layer of STF:127-244 on static
inputs, the residency eager). The hoisted tensors live in PERSISTENT buffers (written in place with out= / copy_) and
the static tensors are made at the state's construction, so a graph that reads them keeps valid addresses. To combine:
r2_i4g_graph.graph_classes(STF, hoist_classes(STF, lean_classes(STF))). Layer 0's graph then holds the hoist (it reads
the static pbs and plan masks of I4G's _static); layers 1 .. L - 1's graphs read the buffers. This file's joint pass
calls self._resid (I4G's capture stub replaces it), and the hoist's host counts (hoist passes / layers) are new keys: the
I4G delta check (SKIPPED_T) must then list them (hoist_layers 1 per layer, hoist_passes 1 at layer 0), or the counts
move to the graph report. Not built here (no GPU run of the combination).

NEGATIVE CONTROLS (tests only; hneg): "stale_pass" (the hoist runs at the first joint pass only: later passes reuse
its positions, masks and lengths), "row0_index" (layers >= 1 write every row's K / V / cis at row 0's buffer row),
"no_plast" (the row lists' last_len is never written: the fork reads 0 or an old pass's lengths).
HOST SYNCS: none in a pass (as L4PTFD). HOST COUNTS: TentStateFDU.hoist (passes, layers); TentState.calls stays
L4PTFD's.
"""
import torch

HNEG = (None, "stale_pass", "row0_index", "no_plast")
# THE STATED OP COUNTS (aten compute ops seen by a TorchDispatchMode counter, views and empty allocations excluded; the
# timing configuration: union check and stats off), L4PTFD's joint-pass _rows minus L4PTFDU's, per layer: SAVED_PER_LAYER
# in each layer after the first, SAVED_LAYER0 in the first (the hoist is made there). R = 4 rows (K 3), the pool on.
SAVED_PER_LAYER = 42
SAVED_LAYER0 = 10


def saved_per_pass(L: int) -> int:
    """The stated drop of aten compute ops per joint pass of L layers (L4PTFD - L4PTFDU)."""
    return SAVED_PER_LAYER * (int(L) - 1) + SAVED_LAYER0


class _Hp:
    """The hoisted tensors of one joint pass (persistent buffers; attributes only)."""


_CLASSES = {}


def hoist_classes(STF, base):
    """(TentStateFDU, TentRunnerFDU) built on the module STF and the lean pair base = (TentStateFD, TentRunnerFD)."""
    key = (id(STF), id(base[0]), id(base[1]))
    if key not in _CLASSES:
        _CLASSES[key] = (STF, base) + _make(STF, base)
    return _CLASSES[key][2:]


def _make(STF, base):
    SpecError = STF.SpecError
    VR = STF.VR
    SD, RD = base

    class TentStateFDU(SD):
        """L4PTFD's state (TentStateFD) whose JOINT pass reuses the hoisted tensors (module docstring). hneg: a negative
        control (HNEG), tests only. vs: the runner's VerifyState (the static tensors are made here)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, hneg=None, vs=None):
            if hneg not in HNEG:
                raise SpecError("unknown hoist control %r (one of %s)" % (hneg, HNEG))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg)
            if vs is None or not hasattr(vs, "slot_b"):
                raise SpecError("L4PTFDU needs the fork's VerifyState (vs.slot_b, vs.store_b)")
            caps = {int(tw.total_cis.shape[1]) for tw in rs.st.twins}
            if len(caps) != 1:
                raise SpecError("the twins' cis rows differ in length %s: the hoisted clamp needs one" % sorted(caps))
            self.hneg = hneg
            self._vs = vs
            self._cmax = caps.pop() - 1
            dev, i64 = self.dev, torch.int64
            self._eb = [vs.arena.layer_base(l) + vs.slot_b for l in range(self.L)]   # VR:1146, per layer
            self._sb1 = vs.store_b + 1                                         # VR:1148: store slot 1 of each request
            self._cap_t = torch.tensor(int(vs.cap), dtype=i64, device=dev)    # VR:1140 full_like(rank, cap)
            self._hps = {}                                                     # R -> _Hp (persistent buffers)
            self._hps[int(vs.R)] = self._alloc(int(vs.R), rs.st.twins[0].pos.dtype)   # made here, not in a pass
            self._cur = None                                                   # this pass's _Hp
            self._in_union = False
            self.hoist = dict(passes=0, layers=0)                             # host counts
            self.hoist_next = torch.zeros((), dtype=i64, device=dev)          # CHECK-ONLY: is_next seen (must be 0)

        def hoist_report(self) -> dict:
            """ONE host read (after a run): the host counts and the check-only counter."""
            return dict(self.hoist, next_seen=int(self.hoist_next))

        # ---------------------------------------------------------- the hoist (layer 0 of a joint pass)
        def _alloc(self, R: int, pdt):
            B, H, T, bs, dev = self.B, self.H, self.topk, self.bs, self.dev
            i64, b8 = torch.int64, torch.bool
            hp = _Hp()
            hp.R = R
            hp.qb = torch.zeros((B,), dtype=i64, device=dev)
            hp.qb1 = torch.zeros((B,), dtype=i64, device=dev)
            hp.qb3, hp.qb4, hp.qb1_4 = hp.qb.view(1, B, 1), hp.qb.view(1, B, 1, 1), hp.qb1.view(1, B, 1, 1)
            hp.P = torch.zeros((B, R), dtype=pdt, device=dev)
            hp.past = torch.zeros((B, R), dtype=b8, device=dev)
            hp.full = torch.zeros((R, B), dtype=b8, device=dev)
            hp.res = torch.zeros((R, B), dtype=b8, device=dev)
            hp.fullv = torch.zeros((H, B, R * T), dtype=b8, device=dev)
            hp.i = torch.zeros((R, B), dtype=pdt, device=dev)
            hp.prow = torch.zeros((R, B), dtype=pdt, device=dev)
            hp.low = torch.zeros((B, bs), dtype=b8, device=dev)
            hp.low4, hp.low3 = hp.low.view(B, bs, 1, 1), hp.low.view(B, bs, 1)
            hp.nam = torch.zeros((1, B, 1), dtype=b8, device=dev)
            return hp

        def _hoist(self, eng, pbs, plan, vs):
            """The layer-invariant tensors of THIS joint pass (module docstring), each written in place."""
            if vs is not self._vs:
                raise SpecError("L4PTFDU: the pass's VerifyState is not the runner's")
            B, R, bs, H, T = self.B, plan.R, self.bs, self.H, self.topk
            hp = self._hps.get(R)
            if hp is None:
                hp = self._hps[R] = self._alloc(R, pbs[0].dtype)
            qc = self.qc
            torch.floor_divide(qc, bs, out=hp.qb)                              # STF:135
            torch.add(hp.qb, 1, out=hp.qb1)                                    # STF:167 / VR:1117
            torch.stack(pbs, 1, out=hp.P)                                      # STF:138
            torch.gt(torch.floor_divide(hp.P, bs), hp.qb.view(B, 1), out=hp.past)   # STF:139
            torch.bitwise_and(plan.verify, plan.act, out=hp.full)             # STF:159
            torch.bitwise_and(plan.draft, plan.act, out=hp.res)               # STF:160
            hp.fullv.view(H, B, R, T).copy_(hp.full.t().reshape(1, B, R, 1).expand(H, B, R, T))   # VR:1130
            for r in range(R):                                                 # ST:491, :496
                torch.remainder(pbs[r], bs, out=hp.i[r])
                torch.clamp(pbs[r], max=self._cmax, out=hp.prow[r])
            if self.tneg == "no_lower":                                        # NEGATIVE CONTROL (tests only), STF:110
                hp.low.fill_(False)
            else:
                torch.lt(self.ar64.view(1, bs), torch.remainder(qc, bs).view(B, 1), out=hp.low)   # STF:112
            torch.logical_not(plan.act[0].view(1, B, 1), out=hp.nam)         # ST:459-460
            if self.hneg != "no_plast":                                        # VR:1209, once per pass
                vs.plast.copy_((self.rs.body.cache_lens_value(eng, hp.P) - (vs.topk - 1) * vs.bs).reshape(B * R))
            self.hoist["passes"] += 1
            return hp

        # ---------------------------------------------------------- the residency (the joint pass's call)
        def _resid(self, l, eng, twin, ti, inj, act, drf, qb3):
            """In the joint pass: TentStateU._resid's host count + TentState._resid's kernels and order with ~act hoisted
            and tent = lm > qb3 (qb3 >= 0; r2_i4a_lean._resid_lean). Elsewhere: the parent's."""
            hp = self._cur
            if not self._in_union or hp is None:
                return super()._resid(l, eng, twin, ti, inj, act, drf, qb3)
            self.calls["resid"] += 1
            rs, B, bs = self.rs, self.B, self.bs
            kern = rs.ops.kernels
            kern["diff"](eng._block_map, ti, eng._new_block_map_buf, eng._load_mask)
            eng._load_mask.masked_fill_(hp.nam, -1)                           # an inactive row loads nothing
            if self.pool:
                kern["pool_update"](eng._block_map, eng._load_mask, eng._pool_map, eng._pool_age, eng._pool_target,
                                    eng._pool_action, eng._pool_stamp_base, eng._tail_block_idx_on_gpu)
                kern["pool_swap"](eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action,
                                  eng.topk, eng.block_size)
                eng._pool_stamp_base += eng.topk
            eng._new_block_map_buf.copy_(torch.where(act.view(1, B, 1), eng._new_block_map_buf, eng._block_map))
            eng._block_map.copy_(eng._new_block_map_buf)
            lm = eng._load_mask
            tent = lm > qb3
            skip = (lm >= 0) & (drf.view(1, B, 1) | tent)
            ids = torch.where(skip, self.minus1, lm)
            kern["gather_k"](eng._k_gpu, eng._k_cpu, ids, bs)
            kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, ids, bs)
            eng._block_map.masked_fill_(skip, -1)
            self.skipped[l] += (skip & ~tent).sum()
            if self.checking:                                                  # CHECK-ONLY counters
                self.forced[l] += tent.sum()
                self.injected[l] += inj.sum()
            return skip

        # ---------------------------------------------------------- the buffer write of row r, layer >= 1
        def _write_h(self, l, twin, r, k_r, v_r, plan, hp) -> None:
            """TentState._write's buffer write (ST:494-497) with the hoisted row index and cis row (layer >= 1: no
            tpos, no counter)."""
            B = self.B
            act = plan.act[r]
            bi = self.all_b
            i = hp.i[0] if self.hneg == "row0_index" else hp.i[r]            # NEGATIVE CONTROL (tests only)
            kb, vb, bb = self.k[l], self.v[l], self.bias[l]
            a3, a2 = act.view(B, 1, 1), act.view(B, 1)
            kb[bi, i] = torch.where(a3, k_r.to(kb.dtype), kb[bi, i])
            vb[bi, i] = torch.where(a3, v_r.to(vb.dtype), vb[bi, i])
            bb[bi, i] = torch.where(a2, twin.total_cis[bi, hp.prow[r]].to(bb.dtype), bb[bi, i])

        # ---------------------------------------------------------- the chain images with the hoisted mask
        def _chain_h(self, l, eng, hp) -> None:
            """TentStateF._chain (STF:105-121) with the hoisted mask x < qc % 64."""
            bs, t0 = self.bs, self.base
            a = self.arena
            torch.where(hp.low4, eng._k_gpu[:, t0:t0 + bs], self.k[l], out=a.cb0_k)
            torch.where(hp.low4, eng._v_gpu[:, t0:t0 + bs], self.v[l], out=a.cb0_v)
            torch.where(hp.low3, eng._kv_bias_gpu[:, t0:t0 + bs], self.bias[l], out=a.cb0_bias)
            ck, cv, cb = self.cb1
            ck.copy_(self.k[l])
            cv.copy_(self.v[l])
            cb.copy_(self.bias[l])
            self.calls["chain"] += 1

        # ---------------------------------------------------------- the locate without the constant is_next
        def _locate_h(self, vs, eng, sel, full, res, hp, l: int) -> dict:
            """verify_rows._fork_locate (VR:1105-1177) for sel = sel_c (no id >= qb): the same values; is_next (False)
            dropped, fullv / eb / store_b + 1 / the fill constants hoisted (module docstring)."""
            rs = self.rs
            H, B, R, T = sel.shape
            cap, topk = vs.cap, vs.topk
            FORK_BIG = VR.FORK_BIG
            v = sel.reshape(H, B, R * T)
            valid = v >= 0
            if vs.ops.check:                                                   # CHECK-ONLY: the dropped test, recounted
                self.hoist_next += ((v == hp.qb1.view(1, B, 1)) & valid).sum()
            ws, wi = torch.sort(eng._block_map, dim=-1)
            at_w = torch.searchsorted(ws, v).clamp(max=ws.shape[-1] - 1)
            in_win = (ws.gather(-1, at_w) == v) & valid
            wslot = wi.gather(-1, at_w)
            if rs.pool:
                ps, pi = torch.sort(eng._pool_map, dim=-1)
                at_p = torch.searchsorted(ps, v).clamp(max=ps.shape[-1] - 1)
                in_pool = (ps.gather(-1, at_p) == v) & valid & ~in_win
                pslot = pi.gather(-1, at_p) + topk
            else:
                in_pool = torch.zeros_like(valid)
                pslot = torch.zeros_like(v)
            host = valid & hp.fullv & ~in_win & ~in_pool
            s = torch.sort(torch.where(host, v, vs.big), dim=-1).values
            first = torch.ones_like(s, dtype=torch.bool)
            first[..., 1:] = s[..., 1:] != s[..., :-1]
            first &= s < FORK_BIG
            rank = first.to(torch.int64).cumsum(-1) - 1
            keep = first & (rank < cap)
            buf = vs.lst_buf
            buf.fill_(FORK_BIG)
            buf.scatter_(-1, torch.where(keep, rank, self._cap_t), torch.where(keep, s, vs.big))
            lst = buf[..., :cap]
            at_l = torch.searchsorted(lst.contiguous(), v).clamp(max=cap - 1)
            in_lst = (lst.gather(-1, at_l) == v) & valid & ~in_win & ~in_pool
            present = in_win | in_pool | in_lst
            eb = self._eb[l]
            phys = torch.where(in_win, eb + wslot, torch.where(in_pool, eb + pslot, self._sb1 + at_l))
            phys = torch.where(present, phys, vs.minus1)
            if vs.ops.stats:                                                   # accounting (VR:1150-1175, verbatim)
                st = vs.stat
                rolev = (full | res).t().reshape(1, B, R, 1).expand(H, B, R, T).reshape(H, B, R * T)
                n_host = keep.to(torch.int64).sum(-1)
                n_all = first.to(torch.int64).sum(-1)
                sp = torch.sort(torch.where(in_pool & rolev, v, vs.big), dim=-1).values
                fp = torch.ones_like(sp, dtype=torch.bool)
                fp[..., 1:] = sp[..., 1:] != sp[..., :-1]
                fp &= sp < FORK_BIG
                n_pool = fp.to(torch.int64).sum(-1)
                st["extras"] += n_all.sum() + n_pool.sum()
                st["extras_host"] += n_host.sum()
                st["extras_pool"] += n_pool.sum()
                st["extras_max"] = torch.maximum(st["extras_max"], n_host.max())
                st["unserved"] += (n_all - cap).clamp(min=0).sum()
                st["fills_used"] += (n_host.sum() > 0).to(torch.int64)
                st["pool_reads"] += (in_pool & rolev).sum()
                st["store_reads"] += (in_lst & rolev).sum()
                su = torch.sort(torch.where(valid & rolev, v, vs.big), dim=-1).values
                fu = torch.ones_like(su, dtype=torch.bool)
                fu[..., 1:] = su[..., 1:] != su[..., :-1]
                fu &= su < FORK_BIG
                n_u = fu.to(torch.int64).sum(-1)
                st["union_sum"] += n_u.sum()
                st["union_max"] = torch.maximum(st["union_max"], n_u.max())
                vs.stat_n["union"] += H * B
            return dict(present=present.view(H, B, R, T), phys=phys.view(H, B, R, T), lst=lst,
                        in_pool=in_pool.view(H, B, R, T), in_lst=in_lst.view(H, B, R, T))

        def _lists_h(self, vs, sel, present, phys) -> None:
            """verify_rows._fork_lists (VR:1203-1208) without last_len (written once per pass by the hoist)."""
            H = sel.shape[0]
            E = vs.E
            keys, order = torch.sort(torch.where(present, sel, vs.big), dim=-1)
            slots = phys.gather(-1, order)
            npg = present.sum(-1)
            vs.pkey.copy_(keys.permute(1, 2, 0, 3).reshape(E, H, -1))
            vs.pslot.copy_(slots.permute(1, 2, 0, 3).reshape(E, H, -1))
            vs.npg.copy_(npg.permute(1, 2, 0).reshape(E, H))

        # ---------------------------------------------------------- the joint pass, one layer
        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            """TentStateF._rows_union (STF:124-244) with the hoisted tensors: the same values, op for op where a value
            reads the layer. Returns (B, R, Hq * D): a VIEW of the persistent vs.fout."""
            vs = plan.vs
            ops = vs.ops
            rs, B, R, bs = self.rs, self.B, plan.R, self.bs
            E = B * R
            Hq, Dh = q4.shape[2], q4.shape[3]
            chk, st = ops.check, ops.stats
            kern = rs.ops.kernels
            qc = self.qc
            if l == 0 and not (self.hneg == "stale_pass" and self._cur is not None):   # NEGATIVE CONTROL: no renewal
                self._cur = self._hoist(eng, pbs, plan, vs)
            hp = self._cur
            if hp is None or hp.R != R:
                raise SpecError("L4PTFDU: layer %d of a joint pass without its hoist (R %d)" % (l, R))
            self.hoist["layers"] += 1
            qb, qb3, qb4, P, past = hp.qb, hp.qb3, hp.qb4, hp.P, hp.past
            hasq = (sel == qb4).any(-1)                                        # (H, B, R): the head selects block qb
            if self.hooks is not None and hasattr(self.hooks, "on_select"):    # CHECK-ONLY (WS-G)
                for r in range(R):
                    self.hooks.on_select(l, plan.kind, r, pbs[r], plan.verify[r], sel[:, :, r])
            VR._mark(vs, l, "resid")
            # 2. THE RESIDENCY, one per layer and pass: row 0's selection (kept, D1)
            s0 = sel[:, :, 0]
            inj = (s0 > qb3) & ~hasq[:, :, 0:1]
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            self._in_union = True
            try:
                skip0 = self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)
            finally:
                self._in_union = False
            VR._mark(vs, l, "write")
            # 3. every row's K, V and cis row into the tentative buffer (row order); layer 0: tpos and the counters
            for r in range(R):
                if l == 0:
                    self._write(l, twin, r, k4[:, r], v4[:, r], pbs[r], plan, past[:, r])
                else:
                    self._write_h(l, twin, r, k4[:, r], v4[:, r], plan, hp)
            VR._mark(vs, l, "chain")
            # 4. the chain images
            self._chain_h(l, eng, hp)
            VR._mark(vs, l, "union")
            # 5. where the other selected blocks live; ONE store fill (store slot 0 = CB1 keeps its id -1)
            full, res = hp.full, hp.res                                        # (R, B)
            if self.fneg == "window_tail":                                     # NEGATIVE CONTROL: block qb from the tail
                sel_c = torch.where(sel > qb4, vs.minus1, sel)
                is_q = torch.zeros_like(sel, dtype=torch.bool)
            else:
                sel_c = torch.where(sel >= qb4, vs.minus1, sel)
                is_q = sel == qb4
            is_n = sel == hp.qb1_4
            loc = self._locate_h(vs, eng, sel_c, full, res, hp, l)
            present = loc["present"] | is_q | is_n
            phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), loc["phys"]))
            VR._mark(vs, l, "fetch")
            vs.host_x[..., 1:].copy_(torch.where(loc["lst"] < VR.FORK_BIG, loc["lst"], vs.minus1))
            kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
            kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
            vs.calls["gather_host"] += 2
            vs.calls["fills"] += 1
            VR._mark(vs, l, "attn")
            # 6. the row lists (last_len: the hoist's) and ONE fork call for every row of every request
            if ops.neg == "fork_union_rows":                                   # NEGATIVE CONTROL (WS-C): VR's lists
                VR._fork_lists(vs, sel, dict(present=present, phys=phys), eng, P)
            else:
                self._lists_h(vs, sel, present, phys)
            if self.tneg == "no_causal":                                       # NEGATIVE CONTROL: the own block whole
                vs.plast.fill_(bs)
            qf = q4.reshape(E, Hq, Dh).contiguous()
            VR._fork_call(vs, qf, vs.fout, vs.fork_cu, vs.fork_seq, vs.fork_tab, vs.fork_ext, vs.fork_len, vs.lists,
                          vs.fork_qmax)
            vs.calls["attn"] += 1
            out = vs.fout.reshape(B, R, -1)
            if st:                                                             # ACCOUNTING (device counters), STF:187
                s = vs.stat
                roles = (full | res).t()
                s["rows_full"] += full.sum()
                s["rows_res"] += res.sum()
                s["win_sel"] += ((s0 >= 0) & plan.act[0].view(1, B, 1)).sum()
                s["win_host"] += ((eng._load_mask >= 0) & ~skip0).sum()
                s["past_edge"] += (past & roles).sum()
                s["f4_rows"] += ((~past) & (P % bs == bs - 1) & roles).sum()
            if ops.record:                                                     # tests: the byte checks (i3_world)
                vs.log.append(dict(kind="fork_t", layer=l, sel=sel.clone(), win=eng._block_map.clone(),
                                   lst=loc["lst"].clone(), present=present.clone(), phys=phys.clone(), qb=qb.clone(),
                                   qc=qc.clone(), p_rows=P.clone(), past=past.clone(), full=full.clone(),
                                   res=res.clone(), pool_map=(eng._pool_map.clone() if self.pool else None),
                                   k_gpu=eng._k_gpu.clone(), v_gpu=eng._v_gpu.clone(),
                                   bias_gpu=eng._kv_bias_gpu.clone(), k_cpu=eng._k_cpu.clone(),
                                   v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone(), tpos=self.tpos.clone(),
                                   buf_k=self.k[l].clone(), buf_v=self.v[l].clone(), buf_bias=self.bias[l].clone(),
                                   store_k=vs.ext_k.clone(), store_v=vs.ext_v.clone(),
                                   store_bias=vs.ext_bias.clone(), cb0_k=self.arena.cb0_k.clone(),
                                   cb0_v=self.arena.cb0_v.clone(), cb0_bias=self.arena.cb0_bias.clone(),
                                   tab=vs.fork_tab.clone(), ext=vs.fork_ext.clone(), ulen=vs.fork_len.clone(),
                                   pkey=vs.pkey.clone(), pslot=vs.pslot.clone(), npg=vs.npg.clone(),
                                   plast=vs.plast.clone(), q_e=qf.clone(), out=vs.fout.float().clone(),
                                   base=self.arena.layer_base(l), store_base=self.arena.store_base,
                                   cb0_base=self.arena.cb0_base, X=vs.X))
            VR._mark(vs, l, "check")
            if chk:                                                            # CHECK-ONLY, STF:212-240
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
                vs.stat_row["visible"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).view(H, B, R).sum((0,
                                                                                                                  1))
                vs.stat_row["selected"] += (sel >= 0).sum(-1).sum((0, 1))
                cbad, crows, creads = self._chain_check(l, eng, sel, P, qb, past, full | res)
                s["chain_bad"] += cbad
                s["chain_rows"] += crows
                s["chain_reads"] += creads
            if vs.fa_check is not None and l == 0 and "F" not in vs.fa_check:
                VR._fork_probe(vs, qf)                                         # CHECK-ONLY: "F" and "RA"
            VR._mark(vs, l, "post")
            return out

    class TentRunnerFDU(RD):
        """L4PTFD (TentRunnerFD) whose joint pass reuses the hoisted tensors (arm L4PTFDU). hneg: TentStateFDU's
        control (tests only)."""

        def __init__(self, *args, hneg=None, **kw):
            if hneg not in HNEG:                                               # before the meter attaches
                raise SpecError("unknown hoist control %r (one of %s)" % (hneg, HNEG))
            self._hneg = hneg
            super().__init__(*args, **kw)
            self.meta.update(joint_hoist=dict(hneg=hneg, saved_per_layer=SAVED_PER_LAYER, saved_layer0=SAVED_LAYER0))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDU(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                hneg=self._hneg, vs=self.vs)

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(hoist=self.T.hoist_report())
            return out

    return TentStateFDU, TentRunnerFDU
