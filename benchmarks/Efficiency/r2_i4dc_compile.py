"""Rung I4DC of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the OFF-THE-SHELF
COMPILER trial (torch.compile / Inductor) extended from the joint pass (rung I4CC, benchmarks/Efficiency/
r2_i4cc_compile.py) to the DRAFT (one-row) passes. The arms:
  ARM L4PTFDGUDCD   = L4PTFDGUDC   (I4CC) whose one-row passes run three compiled draft helpers
  ARM L4PTFDGUDKWCD = L4PTFDGUDKWC (I4CC) with the same draft helpers
OPT-IN: benchmarks/Efficiency/r2_i4dc_cell.py; no nosi module and no older cell imports this file. NO NEW KERNEL by hand
and no kernel edit: Inductor generates the fused kernels from OUR Python (rung 1 of the skill reusing-before-building);
the shipped kernels, the residency, the attention calls and the float merge stay as they are. This file imports torch,
time and r2_i4cc_compile only.

ABBREVIATIONS. CC = benchmarks/Efficiency/r2_i4cc_compile.py; LN = benchmarks/Efficiency/r2_i4a_lean.py (the lean one-row
pass, TentStateFD._row_lean LN:145-194, _write_lean LN:226-254); DG = benchmarks/Efficiency/r2_i4h_graph.py (the draft
graphs: per layer A(l) before the eager residency, B(l) after it, DG:13-28).

WHAT IS COMPILED (module-level pure functions, fullgraph=True, dynamic=False, mode default; a layer index is never a
graph constant, so 32 layers give ONE compile per helper; the basis: the I4C draft traces, the docstring of DHELPERS):
  dpre   graph A(l): hasq, nhq, inj, ti (LN:154-159), the residency's inputs and the masks that B(l) reads
  dwrite graph B(l): the row's K, V and cis row into the tentative buffer (LN:231-235) on INTEGER VIEWS, as an index /
         where / index_put of the B rows: row p % 64 of request b <- the row when b is active, else its own bytes.
         LN writes the same bytes with one torch.where over all 64 rows of every request (a full read and write of the
         buffer: 30 / 42 us per tensor and layer at B 216 / 318 in the I4C traces); the index form touches B rows only.
         The SAME BYTES by construction: oh = (ar64 == p % 64) & act selects exactly the row (b, p % 64) of an active b.
  dmask  graph B(l): m63, the slot mask (cat), call A's bias into the scratch (one where: a bit copy, LN:167-179), vis and
         call B's bias (LN:184-185), on integer views. Call B's bias is made before call A here (it reads no output of
         call A: the same values).
WHAT STAYS EAGER OR UNCOMPILED: the layer-0 hoists of the pass (LN _lean_pass, _lean_mask), layer 0's tpos and counters
  (LN:236-250), the residency (DG's split point), both attention calls, the float32 log-sum-exp merge, the twin step, the
  selection, the GEMMs and every shipped kernel.
EXACTNESS (the EQF MUST: = L4PTFD bit for bit). Every compiled op is an integer, boolean or bit-copy op (==, >, any, ~,
  &, |, where, cat, index, index_put, copy_ on integer views). No float arithmetic, no sort, no GEMM, no shipped kernel.
  The bf16 tensors enter as integer views (CC.bits): no NaN canonicalisation through fp32.
COMPOSITION WITH OUR CAPTURE (DG's split graphs): the first one-row pass of a key is DG's WARM pass (eager, on the side
  stream, outside any capture): it compiles the three helpers. The second pass CAPTURES A(l) and B(l): the compiled
  wrappers launch their kernels into the capture (their outputs come from the graph pool, as the eager ops' did). Later
  passes replay. In the cells a C arm (the joint helpers) runs before the D arm in the same process, so the D arm
  compiles only the draft helpers, before its first capture of any kind.
COMPILE EVENTS (as CC's, plus): every event records its kind (joint / draft), the one-row pass index (lean passes) and
  captures_begun (the captures of either kind begun so far in the arm). THE GATE <arm>-COMPILE (compile_gate_cd, MUST):
  CC's joint conditions, no draft fallback, every one-row layer through the helpers (dsections = (warm + capture passes
  + eager) x L of DG's report), every helper called, no graph break, EVERY compile event before the arm's first capture
  (captures_begun 0, none inside a capture), at most one compile per helper, the dtype checks.
FALLBACK (counted in cc["dfallback"]; the gate needs none): a one-row pass with the test record, hooks, a test control
  (tneg, dneg) or unequal bias / scratch dtypes runs LN's eager _row_lean.
NEGATIVE CONTROLS (tests only; cdneg): "lost_dwrite" (the K rows into a copy of the buffer: EQF must fail), "dlate"
  (inside a draft capture the dpre helper runs behind another code object: its first compile happens there; COMPILE must
  fail).
HOST SYNCS: none added (the compile events read a host counter only).
"""
import time

import torch

import r2_i4cc_compile as CC

CDNEG = (None, "lost_dwrite", "dlate")
# The draft helpers. The I4C draft traces (KS/cpu-logs/i4cc_ana_s1.txt, one-row pass of L4PTFDP, layers 1..31; us per
# layer at B 216 / B 318): dpre = eq, any, not, gt, and, where (23.2 / 24.6); dwrite = 3 where + index (75.6 / 101.9);
# dmask = and, or, cat, where + or, and, where (44.2 / 49.8).
DHELPERS = ("dpre", "dwrite", "dmask")
ARMS_D = dict(L4PTFDGUDCD="L4PTFDGUDC", L4PTFDGUDKWCD="L4PTFDGUDKWC")
bits = CC.bits


# ===================================================================================================================
# the compiled draft helpers (pure: no attribute of a state, no host read)
# ===================================================================================================================
def dpre_fn(sel, qb3):
    """LN:154-159: hasq (H, B, 1), nhq, inj (H, B, T), ti (the residency's selection, contiguous)."""
    hasq = (sel == qb3).any(-1, keepdim=True)
    nhq = ~hasq
    inj = (sel > qb3) & nhq
    ti = torch.where(inj, qb3.expand_as(sel), sel).contiguous()
    return hasq, nhq, inj, ti


def dwrite_fn(kb, vb, bb, cis, bi, i, prow, act, k_r, v_r):
    """LN:231-235 on integer views, rows only (module docstring): kb, vb (B, 64, H, D), bb (B, 64, H), cis (B, n, H),
    bi = arange(B), i = p % 64 (B,), prow (B,), act (B,), k_r / v_r (B, H, D). In place."""
    a3, a2 = act.view(-1, 1, 1), act.view(-1, 1)
    kb[bi, i] = torch.where(a3, k_r, kb[bi, i])
    vb[bi, i] = torch.where(a3, v_r, vb[bi, i])
    bb[bi, i] = torch.where(a2, cis[bi, prow], bb[bi, i])


def dmask_fn(skip, nhq, past3, maskb, live, scratch, okv, ncmt, hasq, bias, tail: int):
    """LN:167-179 and 184-185 on integer views: call A's bias into scratch (B, T, 64, H) (in place) and returns call
    B's bias (B, 64, H). skip / nhq / hasq (H, B, .), past3 (1, B, 1), live (B, T, 64, H), okv / ncmt (B, 64, 1)."""
    m63 = nhq & past3
    mslot = torch.cat((skip[:, :, :tail], skip[:, :, tail:] | m63), dim=2)
    scratch.copy_(torch.where(mslot.permute(1, 2, 0).unsqueeze(2), maskb, live))
    vis = okv & (ncmt | hasq.permute(1, 2, 0))
    return torch.where(vis, bias, maskb)


def _dpre_late(sel, qb3):
    """The control "dlate" only: dpre_fn behind another code object (dynamo compiles it at its first call)."""
    return dpre_fn(sel, qb3)


DFNS = dict(dpre=dpre_fn, dwrite=dwrite_fn, dmask=dmask_fn, dpre_late=_dpre_late)
_DCOMPILED = {}


def compiled_d(name: str, backend: str):
    """The compiled draft helper (one per (name, backend) per process, as CC.compiled)."""
    key = (name, backend)
    f = _DCOMPILED.get(key)
    if f is None:
        if backend not in CC.BACKENDS:
            raise ValueError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
        f = _DCOMPILED[key] = torch.compile(DFNS[name], fullgraph=True, dynamic=False, backend=backend)
    return f


_CLASSES = {}


def dc_classes(STF) -> dict:
    """{arm: (state class, runner class)} for ARMS_D, built on the spec_tentative_fork module STF and CC's classes."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF, _make(STF))
    return _CLASSES[key][1]


def _make(STF):
    SpecError = STF.SpecError
    CCL = CC.compile_classes(STF)
    SC, RC = CCL["L4PTFDGUDC"]
    SKC, RKC = CCL["L4PTFDGUDKWC"]

    class _CDMix:
        """The draft helpers' plumbing on a C state (a mixin before CC's state class in the MRO)."""

        def _cd_setup(self, cdneg):
            if cdneg not in CDNEG:
                raise SpecError("unknown draft compile control %r (one of %s)" % (cdneg, CDNEG))
            self.cdneg = cdneg
            self._cd_cap = False                                       # set by the runner's begin / end wrappers
            self.cc.update(dsections=0, dfallback={}, captures_begun=0, first_dcapture=None)
            bb = self.bias[0]
            self._cd_ok = bool(self._cc_ok and self.scratch.dtype == bb.dtype and self.maskb.dtype == bb.dtype)

        def _cd_why(self):
            """None: the one-row pass runs through the compiled helpers; else the reason for LN's eager row."""
            if self.rec is not None:
                return "record"
            if self.hooks is not None:
                return "hooks"
            if self.tneg is not None or self.dneg is not None:
                return "control"
            if not self._cd_ok:
                return "dtype"
            return None

        def _cc(self, name, l, *args):
            """CC's joint call (CC _CCMix._cc) with the event's kind, pass index and captures_begun."""
            n0 = len(self.cc["events"])
            out = super()._cc(name, l, *args)
            for e in self.cc["events"][n0:]:
                e.update(kind="joint", dpass=int(self.lean["passes"]), after_captures=int(self.cc["captures_begun"]))
            return out

        def _cd(self, name, l, *args):
            """One call of the compiled draft helper `name` (module docstring: COMPILE EVENTS)."""
            f = compiled_d(name, self._cc_backend)
            if self.cdneg == "dlate" and self._cd_cap and name == "dpre":     # NEGATIVE CONTROL: a compile in a capture
                f = compiled_d("dpre_late", self._cc_backend)
            from torch._dynamo.utils import counters as C
            g0 = int(C["stats"]["unique_graphs"])
            t0 = time.perf_counter()
            out = self._cc_wrap(name, f, args) if self._cc_wrap is not None else f(*args)
            dt = time.perf_counter() - t0
            g1 = int(C["stats"]["unique_graphs"])
            self.cc["calls"][name] = self.cc["calls"].get(name, 0) + 1
            if g1 != g0:
                self.cc["events"].append(dict(name=name, graphs=g1 - g0, s=round(dt, 3), layer=int(l),
                                              joint_pass=int(self.calls["joint_passes"]),
                                              capture=bool(self._cd_cap or self._cc_cap), kind="draft",
                                              dpass=int(self.lean["passes"]),
                                              after_captures=int(self.cc["captures_begun"])))
                self.cc["compile_s"] += dt
            return out

        # ---------------------------------------------------------- one row, one layer (LN:145-194)
        def _row_lean(self, l, eng, twin, sel, q_r, k_r, v_r, p, plan):
            why = self._cd_why()
            if why is not None:
                self.cc["dfallback"][why] = self.cc["dfallback"].get(why, 0) + 1
                return super()._row_lean(l, eng, twin, sel, q_r, k_r, v_r, p, plan)
            rs, B, bs = self.rs, self.B, self.bs
            act, drf = plan.act[0], plan.draft[0]
            lp = self._lp
            if lp is None:
                lp = self._lp = self._lean_pass(eng, twin, p, act)
            qb3 = lp.qb3
            self.cc["dsections"] += 1
            hasq, nhq, inj, ti = self._cd("dpre", l, sel, qb3)        # graph A(l)
            # 4. the residency on the committed window (kept, D1; DG's split point)
            skip = self._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)
            # 5. the row's K, V and cis row into the buffer (graph B(l) from here)
            self._cd_write(l, twin, k_r, v_r, p, plan, lp)
            if lp.okv is None:
                self._lean_mask(lp)
            # 6. attention: A (the committed window) + B (the buffer), merged by the log-sum-exp
            if self.pool:
                live, katt, vatt = eng._kv_bias_gpu_att, eng._k_gpu_att, eng._v_gpu_att
            else:
                live, katt, vatt = eng._kv_bias_gpu, eng._k_gpu, eng._v_gpu
            T, H = self.topk, self.H
            bb = self.bias[l]
            if live.dtype != bb.dtype:                                 # the integer views need equal dtypes
                raise SpecError("%s: the live bias %s differs from the buffer bias %s" % (type(self).__name__,
                                                                                      live.dtype, bb.dtype))
            if live.untyped_storage().data_ptr() == self.scratch.untyped_storage().data_ptr():
                raise SpecError("%s: the live bias is a view of the scratch (an aliased input of dmask)" %
                                type(self).__name__)                   # a host attribute read, no device sync
            bias_b = self._cd("dmask", l, skip, nhq, lp.past3, bits(self.maskb), bits(live).view(B, T, bs, H),
                              bits(self.scratch_att).view(B, T, bs, H), lp.okv, lp.ncmt, hasq, bits(bb),
                              int(self.tail)).view(bb.dtype)
            nsplit = rs.nl._ATTN_SPLITS
            qq = q_r.unsqueeze(1).contiguous()
            o_a, lse_a = self.attn(qq, katt, vatt, self.scratch_att, cache_seqlens=lp.cut, num_splits=nsplit,
                                   return_softmax_lse=True)
            o_b, lse_b = self.attn(qq, self.k[l], self.v[l], bias_b, cache_seqlens=self.len_b, num_splits=nsplit,
                                   return_softmax_lse=True)
            Hq = q_r.shape[1]                                          # LN's merge (TentState._row's), op for op
            la = lse_a.reshape(B, Hq, 1).float()
            lb = lse_b.reshape(B, Hq, 1).float()
            mx = torch.maximum(la, lb)
            wa, wb = torch.exp(la - mx), torch.exp(lb - mx)
            o = (o_a.reshape(B, Hq, -1).float() * wa + o_b.reshape(B, Hq, -1).float() * wb) / (wa + wb)
            return o.to(q_r.dtype).reshape(B, -1)

        def _cd_write(self, l, twin, k_r, v_r, p, plan, lp) -> None:
            """LN _write_lean: the compiled row write (every layer); layer 0 also tpos and the counters (LN:236-250, op
            for op; the test record is a fallback reason, so it never runs here)."""
            B, bs = self.B, self.bs
            act, drf = plan.act[0], plan.draft[0]
            kb, vb, bb = self.k[l], self.v[l], self.bias[l]
            if k_r.dtype != kb.dtype or v_r.dtype != vb.dtype:          # the integer views need equal dtypes
                raise SpecError("%s: K / V row %s / %s differs from the buffer %s / %s" % (
                    type(self).__name__, k_r.dtype, v_r.dtype, kb.dtype, vb.dtype))
            kx = bits(kb)
            if self.cdneg == "lost_dwrite":                            # NEGATIVE CONTROL (tests only)
                kx = kx.clone()
            self._cd("dwrite", l, kx, bits(vb), bits(bb), bits(twin.total_cis), self.all_b, lp.i, lp.prow,
                     act.contiguous(), bits(k_r.contiguous()), bits(v_r.contiguous()))
            if l == 0:
                qc = self.qc
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

        def cc_report(self) -> dict:
            """CC's compile record + the draft keys."""
            out = super().cc_report()
            c = self.cc
            out.update(dsections=int(c["dsections"]), dfallback=dict(c["dfallback"]),
                       captures_begun=int(c["captures_begun"]), first_dcapture=c["first_dcapture"],
                       cd_ok=bool(self._cd_ok), cdneg=self.cdneg, dhelpers=DHELPERS)
            return out

    # ------------------------------------------------------------------ L4PTFDGUDCD
    class TentStateFDGUDCD(_CDMix, SC):
        """L4PTFDGUDC whose one-row passes run the compiled draft helpers (module docstring)."""

    class TentRunnerFDGUDCD(RC):
        """The arm L4PTFDGUDCD. Keyword arguments: CC's (cc_backend, ccneg) and STK's, cdneg (CDNEG, tests only)."""

        def __init__(self, *args, cdneg=None, **kw):
            if cdneg not in CDNEG:
                raise SpecError("unknown draft compile control %r (one of %s)" % (cdneg, CDNEG))
            self._cdneg = cdneg
            super().__init__(*args, **kw)
            _wrap_dcapture(self.T)
            self.meta["compile"].update(arm=self._arm_name(), cdneg=cdneg, dhelpers=DHELPERS)

        def _arm_name(self):
            return "L4PTFDGUDCD"

        def _make_state(self, rs, attn, tneg, poison):
            T = TentStateFDGUDCD(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                 gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg)
            T._cc_setup(*self._ccopt)
            T._cd_setup(self._cdneg)
            return T

    # ------------------------------------------------------------------ L4PTFDGUDKWCD
    class TentStateFDGUDKCD(_CDMix, SKC):
        """L4PTFDGUDKWC whose one-row passes run the compiled draft helpers (module docstring)."""

    class TentRunnerFDGUDKCD(RKC):
        """The arm L4PTFDGUDKWCD (keep=True, kwin=True; kcheck in check cells). Keyword arguments: CC's, STK's,
        cdneg (tests only)."""

        def __init__(self, *args, cdneg=None, **kw):
            if cdneg not in CDNEG:
                raise SpecError("unknown draft compile control %r (one of %s)" % (cdneg, CDNEG))
            self._cdneg = cdneg
            super().__init__(*args, **kw)
            _wrap_dcapture(self.T)
            self.meta["compile"].update(arm=self._arm_name(), cdneg=cdneg, dhelpers=DHELPERS)

        def _arm_name(self):
            return "L4PTFDGUDKWCD"

        def _make_state(self, rs, attn, tneg, poison):
            T = TentStateFDGUDKCD(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                  gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg,
                                  cneg=self._cneg, **self._k)
            T._cc_setup(*self._ccopt)
            T._cd_setup(self._cdneg)
            return T

    return dict(L4PTFDGUDCD=(TentStateFDGUDCD, TentRunnerFDGUDCD),
                L4PTFDGUDKWCD=(TentStateFDGUDKCD, TentRunnerFDGUDKCD))


def _wrap_dcapture(T) -> None:
    """The draft capture flag and the capture count (module docstring, COMPILE EVENTS): instance attributes over the
    backend's begin() / end() (DG's split capture) and capture() (the joint capture, already CC's wrapper)."""
    be = T.backend
    ob, oe, oc = be.begin, be.end, be.capture

    def begin():
        T.cc["captures_begun"] += 1
        if T.cc["first_dcapture"] is None:
            T.cc["first_dcapture"] = dict(dpass=int(T.lean["passes"]), joint_pass=int(T.calls["joint_passes"]))
        ob()
        T._cd_cap = True

    def end():
        T._cd_cap = False
        return oe()

    def capture(fn, T_, vs):
        T.cc["captures_begun"] += 1
        return oc(fn, T_, vs)
    be.begin, be.end, be.capture = begin, end, capture


def compile_gate_cd(arm: str, rd, L: int, names) -> tuple:
    """<arm>-COMPILE (MUST) of a D arm (module docstring): CC.compile_gate's conditions for the joint section, and for
    the draft: no fallback, dsections = (warm + capture passes + eager) x L of the draft-graph report, a draft capture
    ran; every helper in `names` called; no graph break; EVERY compile event before the arm's first capture of any kind
    (captures_begun 0 at the event; none inside a capture); at most one compile per helper; the dtype checks."""
    c = rd.get("compile") or {}
    g = rd.get("graph") or {}
    dg = rd.get("dgraph") or {}
    ev = c.get("events") or []
    late = [e for e in ev if e.get("capture") or e.get("after_captures", 1) != 0]
    per = {}
    for e in ev:
        per[e["name"]] = per.get(e["name"], 0) + 1
    calls = c.get("calls") or {}
    want_sections = g.get("eager", 0) + g.get("warm", 0) + g.get("captures", 0)
    want_d = (dg.get("warm", 0) + dg.get("capture_passes", 0) + dg.get("eager", 0)) * int(L)
    brk = (c.get("dynamo") or {}).get("graph_breaks")
    ok = (not c.get("fallback") and c.get("sections") == want_sections and want_sections > 0
          and not c.get("dfallback") and c.get("dsections") == want_d and want_d > 0
          and c.get("first_dcapture") is not None
          and all(calls.get(n, 0) > 0 for n in names) and brk == 0 and not late and all(v <= 1 for v in per.values())
          and c.get("dtype_ok") is True and c.get("cd_ok") is True)
    return ("%s-COMPILE" % arm, True, bool(ok),
            "backend %s; joint sections %s (eager + warm + captures %s); draft sections %s ((warm + capture passes + "
            "eager) x L %s); fallback %s / %s; calls %s; compile events %s (late %s; per helper %s); first draft "
            "capture %s; captures begun %s; compile %s s; graph breaks %s; dynamo %s" % (
                c.get("backend"), c.get("sections"), want_sections, c.get("dsections"), want_d, c.get("fallback"),
                c.get("dfallback"), calls,
                [(e["name"], e.get("kind"), e["joint_pass"], e.get("dpass"), e["layer"], e.get("after_captures"),
                  e["s"]) for e in ev], len(late), per, c.get("first_dcapture"), c.get("captures_begun"),
                c.get("compile_s"), brk, c.get("dynamo")))
