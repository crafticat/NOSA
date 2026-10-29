"""Rung I4A of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 1 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (R5.5 item 1, the draft phase): ARM L4PTFD = L4PTF (rung I3,
nosi/nosi/spec_tentative_fork.TentRunnerF) with the LEAN ONE-ROW PASS. Docs: docs/NOTES_i4a.md (the op inventory, the
counts, the bands). OPT-IN: benchmarks/Efficiency/r2_i4a_cell.py (arm L4PTFD); no nosi module and no older cell imports
this file. NO NEW KERNEL and no new kernel launch kind: the same kernels and torch ops as L4PTF's one-row pass, fewer
of them.

WHY A CLASS FACTORY (as r2_body_fused.fused_body_class): lean_classes(STF) builds the classes on the GIVEN
spec_tentative_fork module (the GPU cell passes nosi.spec_tentative_fork; the CPU tests pass the CPU world's synthetic
package copy), so this file imports only torch and no nosi module names it.

WHAT CHANGES (one-row passes only: the K - 1 D slots, the prelude's K drafts, the one-row J slots). L4PTF runs every
one-row pass through spec_tentative.TentState._row (I1's tentative row, spec_tentative.py:395-448): per layer about 85
small torch ops beyond the plain step's, most of them functions of (qc, p, tpos) that are THE SAME IN EVERY LAYER of the
pass. TentStateFD._row_lean computes the same values with fewer ops:
  1. HOISTED ONCE PER PASS (at layer 0; the twins of all layers step together, so p = twin.pos is the same in every
     layer; qc does not move inside a pass): qb, the row's block, past, p % 64, the one-hot write mask, prow, n_tail,
     cq, the call-A cut. After layer 0's buffer write (tpos moves only there): call B's layer-independent masks (lo, hi,
     own, F4, cmt). Per layer only the parts that read the layer's selection stay (hasq, the injection, m63, vis).
  2. THE BUFFER WRITE (step 5): ONE torch.where per tensor over the request's 64 buffer rows with the hoisted one-hot
     mask (row p % 64 of an active request), in place of index + where + index_put per tensor. The same bytes.
  3. CALL A'S BIAS: ONE torch.where(mask, MASK_BIAS, live) into the scratch (the mask broadcast over the 64 rows of a
     slot), in place of the expanded mask copy + the full scratch copy + masked_fill_. The same bytes.
  4. THE RESIDENCY (step 4, D1): TentState._resid's ops and order; tent = lm > qb (= (lm >= 0) & (lm > qb): qb >= 0),
     lm >= 0 once. The same engine bytes.
  5. no torch.stack of the single row (a view).
KEPT, REQUIRED (docs/NOTES_i4a.md section 1): the twin step, the selection, the residency with the draft skips and the
tail-id injection (D1, Task 1.1), the buffer write and tpos (D3: the deferred replay reads the draft's row; the joint
pass's chain images read the buffer), call A on the committed window with the skipped slots and m63 masked, call B on
the buffer and the float32 log-sum-exp merge (a row past the edge sees its own block's chain rows, which have no window
slot: one flash call over the window cannot show them without a page table, i.e. the fork; the merge stays I1's
bit for bit), the layer-0 counters (D3, the draft skips, the span).
RESULT: every value of the pass is L4PTF's BIT FOR BIT (tokens, logits, buffers, tpos, the engine, the twins, the
counters, the trace); only the number of torch ops per one-row pass falls. CPU tests: tests_cpu/test_i4a.py.

NEGATIVE CONTROLS (tests only; dneg): "mask_before_write" (call B's masks hoisted BEFORE layer 0's buffer write: the
row does not see its own entry), "live_bias" (call A reads the live window bias: skipped slots and m63 unmasked),
"stale_pass" (the hoisted tensors are not renewed at a pass start: a later pass uses an earlier pass's positions).
HOST SYNCS: none (as L4PTF). HOST COUNTS: TentStateFD.lean (passes, layers); TentState.calls stays L4PTF's.
"""
import torch

DNEG = (None, "mask_before_write", "live_bias", "stale_pass")
# THE STATED OP COUNTS (aten compute ops seen by a TorchDispatchMode counter, views and empty allocations excluded;
# docs/NOTES_i4a.md section 1), L4PTF's one-row pass minus L4PTFD's: SAVED_PER_LAYER in each layer after the first,
# SAVED_LAYER0 in the first layer (the hoisted tensors are made there). Per one-row pass: SAVED_PER_LAYER x (L - 1) +
# SAVED_LAYER0 (L 32: 1,158). The CODE count is 2 higher per layer and 1 higher in layer 0 (I1's reshape copy of the
# expanded mask, spec_tentative.py:420, and its int32 cast of the cut, :417; the counter shows no difference for them).
SAVED_PER_LAYER = 37
SAVED_LAYER0 = 11
CODE_EXTRA_PER_LAYER = 2
CODE_EXTRA_LAYER0 = 1


def saved_per_pass(L: int) -> int:
    """The stated drop of aten compute ops per one-row pass of L layers (L4PTF - L4PTFD)."""
    return SAVED_PER_LAYER * (int(L) - 1) + SAVED_LAYER0


class _Lp:
    """The hoisted tensors of one one-row pass (attributes only)."""


_CLASSES = {}


def lean_classes(STF):
    """(TentStateFD, TentRunnerFD) built on the spec_tentative_fork module STF; one pair per module object."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF,) + _make(STF)
    return _CLASSES[key][1:]


def _make(STF):
    SpecError = STF.SpecError

    class TentStateFD(STF.TentStateF):
        """L4PTF's state (TentStateF) whose ONE-ROW passes run _row_lean (module docstring). dneg: a negative control
        (DNEG), tests only."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None):
            if dneg not in DNEG:
                raise SpecError("unknown lean control %r (one of %s)" % (dneg, DNEG))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg)
            self.dneg = dneg
            self._lp = None
            self.lean = dict(passes=0, layers=0)                       # host counts (TentState.calls stays L4PTF's)

        def run_pass(self, model, plan):
            if plan.R == 1:
                if self.dneg != "stale_pass":                          # NEGATIVE CONTROL: the tensors kept
                    self._lp = None                                    # the hoisted tensors of THIS pass
                self.lean["passes"] += 1
            return super().run_pass(model, plan)

        def _rows(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            if plan.R != 1:                                            # the joint pass: L4PTF's, unchanged
                return super()._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            self.lean["layers"] += 1
            o = self._row_lean(l, eng, twin, sel[:, :, 0], q4[:, 0], k4[:, 0], v4[:, 0], pbs[0], plan)
            return o.unsqueeze(1)                                      # TentState._rows: torch.stack of one row

        # ---------------------------------------------------------- the hoisted tensors
        def _lean_pass(self, eng, twin, p, act):
            """Part 1 (layer 0, before the buffer write): every tensor of the row that does not read tpos or the
            layer's selection."""
            B, bs = self.B, self.bs
            qc = self.qc
            lp = _Lp()
            lp.p = p
            lp.qb = qc // bs
            lp.qb3 = lp.qb.view(1, B, 1)
            lp.bp = p // bs
            lp.past = lp.bp > lp.qb                                    # (B,): the row is past the committed block's edge
            lp.past3 = lp.past.view(1, B, 1)
            lp.i = p % bs
            lp.oh = (self.ar64.view(1, bs) == lp.i.view(B, 1)) & act.view(B, 1)   # (B, 64): the row's buffer row
            lp.oh3, lp.oh4 = lp.oh.view(B, bs, 1), lp.oh.view(B, bs, 1, 1)
            lp.prow = p.clamp(max=int(twin.total_cis.shape[1]) - 1)
            lp.n_tail = self.rs.body.cache_lens_value(eng, p) - self.base   # rows of p's block the shipped step reads
            cq = qc % bs
            lp.cut = (self.base + torch.where(lp.past, cq, torch.minimum(cq, lp.n_tail))).to(torch.int32)
            lp.okv = lp.ncmt = None
            if self.dneg == "mask_before_write":                       # NEGATIVE CONTROL: tpos before the write
                self._lean_mask(lp)
            return lp

        def _lean_mask(self, lp):
            """Part 2 (layer 0, after the buffer write: tpos moves only there): call B's layer-independent masks
            (TentState._row's lo, hi, own, F4, cmt and its controls no_lower / no_causal)."""
            B, bs = self.B, self.bs
            qc, p = self.qc, lp.p
            tp = self.tpos
            lo = (tp >= 0) if self.tneg == "no_lower" else (tp >= qc.view(B, 1))       # NEGATIVE CONTROLS (tests only)
            hi = (tp >= 0) if self.tneg == "no_causal" else (tp <= p.view(B, 1))
            tb = tp // bs
            own = tb == lp.bp.view(B, 1)
            ok = lo & hi & (~own | ((tp % bs) < lp.n_tail.view(B, 1)))
            cmt = (tb == lp.qb.view(B, 1)) & lp.past.view(B, 1)
            lp.okv = ok.view(B, bs, 1)
            lp.ncmt = ~cmt.view(B, bs, 1)

        # ---------------------------------------------------------- one row, one layer
        def _row_lean(self, l, eng, twin, sel, q_r, k_r, v_r, p, plan):
            """Steps 4-6 of spec_tentative's module docstring for the single row of every request (TentState._row's
            values, bit for bit). Returns (B, Hq * D)."""
            rs, B, bs = self.rs, self.B, self.bs
            act, drf = plan.act[0], plan.draft[0]
            lp = self._lp
            if lp is None:
                lp = self._lp = self._lean_pass(eng, twin, p, act)
            qb3 = lp.qb3
            hasq = (sel == qb3).any(-1, keepdim=True)                  # (H, B, 1)
            if self.hooks is not None and hasattr(self.hooks, "on_select"):    # CHECK-ONLY (WS-G)
                self.hooks.on_select(l, plan.kind, 0, p, plan.verify[0], sel)
            nhq = ~hasq
            inj = (sel > qb3) & nhq
            ti = torch.where(inj, qb3.expand_as(sel), sel).contiguous()
            # 4. the residency on the committed window (kept, D1)
            skip = self._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)
            # 5. the row's K, V and cis row into the buffer (never into the committed tail slot)
            self._write_lean(l, twin, k_r, v_r, p, plan, lp)
            if lp.okv is None:
                self._lean_mask(lp)
            # 6. attention: A (the committed window) + B (the buffer), merged by the log-sum-exp
            m63 = nhq & lp.past3
            mslot = torch.cat((skip[:, :, :self.tail], skip[:, :, self.tail:] | m63), dim=2)
            if self.pool:
                live, katt, vatt = eng._kv_bias_gpu_att, eng._k_gpu_att, eng._v_gpu_att
            else:
                live, katt, vatt = eng._kv_bias_gpu, eng._k_gpu, eng._v_gpu
            if self.dneg == "live_bias":                               # NEGATIVE CONTROL: no masked slot
                bias_a = live
            else:
                T, H = self.topk, self.H
                torch.where(mslot.permute(1, 2, 0).unsqueeze(2), self.maskb, live.view(B, T, bs, H),
                            out=self.scratch_att.view(B, T, bs, H))
                bias_a = self.scratch_att
            nsplit = rs.nl._ATTN_SPLITS
            qq = q_r.unsqueeze(1).contiguous()
            o_a, lse_a = self.attn(qq, katt, vatt, bias_a, cache_seqlens=lp.cut, num_splits=nsplit,
                                   return_softmax_lse=True)
            vis = lp.okv & (lp.ncmt | hasq.permute(1, 2, 0))
            bias_b = torch.where(vis, self.bias[l], self.maskb)
            o_b, lse_b = self.attn(qq, self.k[l], self.v[l], bias_b, cache_seqlens=self.len_b, num_splits=nsplit,
                                   return_softmax_lse=True)
            Hq = q_r.shape[1]                                          # TentState._row's merge, op for op
            la = lse_a.reshape(B, Hq, 1).float()
            lb = lse_b.reshape(B, Hq, 1).float()
            mx = torch.maximum(la, lb)
            wa, wb = torch.exp(la - mx), torch.exp(lb - mx)
            o = (o_a.reshape(B, Hq, -1).float() * wa + o_b.reshape(B, Hq, -1).float() * wb) / (wa + wb)
            return o.to(q_r.dtype).reshape(B, -1)

        def _resid_lean(self, l, eng, twin, ti, inj, act, drf, qb3):
            """TentState._resid (with TentStateU's host count): the same kernels, ops and order, the same engine
            bytes; tent = lm > qb (qb >= 0) and lm >= 0 once."""
            self.calls["resid"] += 1
            rs, B, bs = self.rs, self.B, self.bs
            kern = rs.ops.kernels
            kern["diff"](eng._block_map, ti, eng._new_block_map_buf, eng._load_mask)
            am = act.view(1, B, 1)
            eng._load_mask.masked_fill_(~am, -1)                       # an inactive row loads nothing, moves nothing
            if self.pool:
                kern["pool_update"](eng._block_map, eng._load_mask, eng._pool_map, eng._pool_age, eng._pool_target,
                                    eng._pool_action, eng._pool_stamp_base, eng._tail_block_idx_on_gpu)
                kern["pool_swap"](eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action,
                                  eng.topk, eng.block_size)
                eng._pool_stamp_base += eng.topk
            eng._new_block_map_buf.copy_(torch.where(am, eng._new_block_map_buf, eng._block_map))
            eng._block_map.copy_(eng._new_block_map_buf)
            lm = eng._load_mask
            tent = lm > qb3                                            # = (lm >= 0) & (lm > qb3): qb3 >= 0
            skip = (lm >= 0) & (drf.view(1, B, 1) | tent)
            ids = torch.where(skip, self.minus1, lm)
            kern["gather_k"](eng._k_gpu, eng._k_cpu, ids, bs)
            kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, ids, bs)
            eng._block_map.masked_fill_(skip, -1)                      # the next row loads a skipped block again
            self.skipped[l] += (skip & ~tent).sum()
            if self.checking:                                          # CHECK-ONLY counters
                self.forced[l] += tent.sum()
                self.injected[l] += inj.sum()
            return skip

        def _write_lean(self, l, twin, k_r, v_r, p, plan, lp) -> None:
            """TentState._write for the single row: one torch.where per tensor with the hoisted one-hot mask; layer
            0: tpos and the counters (TentState._write's, op for op)."""
            B, bs = self.B, self.bs
            act, drf = plan.act[0], plan.draft[0]
            kb, vb, bb = self.k[l], self.v[l], self.bias[l]
            bi = self.all_b
            torch.where(lp.oh4, k_r.to(kb.dtype).unsqueeze(1), kb, out=kb)
            torch.where(lp.oh4, v_r.to(vb.dtype).unsqueeze(1), vb, out=vb)
            torch.where(lp.oh3, twin.total_cis[bi, lp.prow].to(bb.dtype).unsqueeze(1), bb, out=bb)
            if l == 0:
                qc = self.qc
                i = lp.i
                torch.where(lp.oh, p.view(B, 1), self.tpos, out=self.tpos)
                c = self.c
                c["rows"] += act.sum()
                c["draft_rows"] += (act & drf).sum()
                c["verify_rows"] += plan.verify[0].sum()
                if plan.deferred is not None:                          # the D3 contract: 0 in the tentative mode
                    c["deferred_writes"] += (act & drf & (p == plan.deferred)).sum()
                c["span_max"].copy_(torch.maximum(c["span_max"], torch.where(act, p + 1 - qc,
                                                                             torch.zeros_like(p)).max()))
                if self.checking:
                    c["edge_rows"] += (act & lp.past).sum()
                    c["fill_rows"] += (act & (p % bs == bs - 1)).sum()
                if self.rec is not None:                               # CHECK-ONLY (tests)
                    self.rec.append(dict(round=self.round, kind=plan.kind, r=0, act=act.clone(), draft=drf.clone(),
                                         p=p.clone(), rope=plan.pos[:, 0].to(torch.int64).clone(), k=k_r.clone(),
                                         c=bb[bi, i].clone(), verify=plan.verify[0].clone(), qc=qc.clone()))

    class TentRunnerFD(STF.TentRunnerF):
        """L4PTF (TentRunnerF) with the lean one-row pass (arm L4PTFD). dneg: TentStateFD's control (tests only)."""

        def __init__(self, *args, dneg=None, **kw):
            if dneg not in DNEG:                                       # before the meter attaches
                raise SpecError("unknown lean control %r (one of %s)" % (dneg, DNEG))
            self._dneg = dneg
            super().__init__(*args, **kw)
            self.meta.update(draft_step="lean", lean=dict(dneg=dneg, saved_per_layer=SAVED_PER_LAYER,
                                                          saved_layer0=SAVED_LAYER0))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFD(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg)

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(lean=dict(self.T.lean))
            return out

    return TentStateFD, TentRunnerFD
