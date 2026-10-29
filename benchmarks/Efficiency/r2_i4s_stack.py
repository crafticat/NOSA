"""Rung I4S of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the STACKED arms of the
measured no-kernel levers of the specKV-on-cached-NOSA ladder. Every lever is a hook on L4PTFD (rung I4A,
r2_i4a_lean.py); this file composes the hooks, it adds no new operation:
  ARM L4PTFDGUD   = L4PTFD + I4G (joint-pass section graphs, r2_i4g_graph.py) + I4U (layer-invariant tensors hoisted
                    once per joint pass, r2_i4u_hoist.py; captured inside layer 0's graph) + I4H (draft one-row passes
                    as per-layer graphs, r2_i4h_graph.py).
  ARM L4PTFDGUDKW = the same + I4D's store keep and window loads from the kept store (r2_i4d_keep.py) in I4K's way: the
                    keep's bookkeeping inside the joint graphs (r2_i4k_combo.py).
OPT-IN: benchmarks/Efficiency/r2_i4s_cell.py; no nosi module and no older cell imports this file. NO NEW KERNEL and no
kernel edit: the levers' torch ops, the shipped kernels and torch.cuda.CUDAGraph (through r2_i4g_graph / r2_i4h_graph).
This file imports torch and the six lever modules only.

ABBREVIATIONS (all in benchmarks/Efficiency): LN r2_i4a_lean.py, GR r2_i4g_graph.py, DG r2_i4h_graph.py, HU
r2_i4u_hoist.py, KP r2_i4d_keep.py, CB r2_i4k_combo.py; STF = nosi/nosi/spec_tentative_fork.py.

THE CLASS STACK (stack_classes(STF); each class factory of a lever takes its base as an argument, so the levers stack
by construction):
  _SUa(HU TentStateFDU)          the hoist state whose options (hneg, vs) come from self._uinit (an outer class sets
                                 it before its __init__ chain: GR's and KP's __init__ do not pass them).
  L4PTFDGU   (no draft graphs; tests only)  GR.graph_classes(STF, (_SUa, HU runner)) + _GraphS (below).
  L4PTFDGUD  DG.draft_classes(STF, L4PTFDGU) + the outer __init__ (hneg, vs) and _make_state.
  L4PTFKU    (eager; tests only)  CB's graph-safe keep state TentStateFKg + _SUa (MRO: FKg, KP's FK, _SUa, HU's FDU,
             LN's FD): the keep's section with the hoisted tensors (TentStateFKU._section, below).
  L4PTFDGUK  (no draft graphs; tests only)  GR.graph_classes(STF, L4PTFKU) + _GraphS + I4K's eager parts (_kpre /
             _kpost, the keep's residency in _resid_eager: CB:402-417).
  L4PTFDGUDKW DG.draft_classes(STF, L4PTFDGUK) + the outer __init__ and _make_state.

WHAT IS CAPTURED WHERE (per layer l of a JOINT pass; the one-row passes are I4H's, unchanged):
  EAGER: the residency (GR:218-227 _resid_eager; with the keep and kwin: CB's _resid_joint -> KP _resid_k), because
         pool_update takes the host stamp (spec_tentative.py:461-466). With the keep also _kpre (the device pass index,
         layer 0) and _kpost (the counter flush, layer L - 1): CB:158-179.
  GRAPH l: the section. L4PTFDGUD: HU TentStateFDU._rows_union (HU:318-452), i.e. at l = 0 the hoist of the pass (HU
         _hoist, HU:146-173: qb, qb + 1, P, past, full, res, fullv, p % 64, the cis row clamp, the chain mask, ~act and
         the row lists' last_len, ALL written in place into persistent buffers made at the state's construction), then
         the hoisted section (HU _write_h, _chain_h, _locate_h, _lists_h, the fork call). Layers 1 .. L - 1's graphs read
         the buffers that layer 0's graph writes (the replays run in layer order, as the captures did). L4PTFDGUDKW: the
         combined section TentStateFKU._section = HU's hoisted section with step 5's store fill replaced by the keep (CB
         _keep_fill: the device step, kcur), the slot-ordered ids for the record / check (KP slot_ids), as CB _section.
  So the residency's hoisted ~act / tent (HU _resid in a joint pass) is NOT used under the graphs: the eager residency
  runs the parent's code (GR _resid_eager, as I4G), because layer 0's hoist runs inside layer 0's graph, after layer
  0's residency. The values are the same (HU's claim: bit for bit).
THE HOIST'S HOST COUNTS (HU self.hoist passes / layers): Python statements inside the captured section, so a replay
  skips them. _GraphS measures them at the capture with GR's counter groups plus the group H (the hoist dict) and every
  replay adds the measured delta (GR's mechanism, GR:229-235, 256-279). The expected delta per layer (expected_delta_s):
  GR's SKIPPED plus H {passes 1, layers 1} at l = 0 and {layers 1} at l >= 1 (HU's docstring, "CUDA GRAPHS"). A capture
  with another delta goes to delta_bad and fails <arm>-GRAPH.
THE GRAPH WRAPPER _GraphS._gsection = GR TentStateFDG._rows_union (GR:237-280) line for line with three parameters: the
  section (HU's or the combined one), the stub's attribute name (_resid; with the keep _resid_joint, as CB:419-464), the
  skip-mask copy (ops.stats; with the keep also kacct, never on here); and the counter groups (+ H).
ADDRESSES: the hoist buffers (HU _alloc at the state's construction: R = vs.R) and the keep's map / counters (KP, CB) are
  persistent; vs.stat is GR's InPlaceDict. DG's __init__ sets self._vs = None after the hoist set it; the outer class
  sets it back to the runner's VerifyState (the same object that DG's runner assigns).
NOT SUPPORTED (refused): the accounting option (KP acct: CB keeps it, the stacked keep arm does not need it), CB's test
  control resid_in_graph (the stub is always made here).
NEGATIVE CONTROL (tests only; sneg): "no_hcounts" (a replay does not add the hoist's skipped host counts: the cell's
  <arm>-HOIST gate fails). The levers' controls pass through (gneg, dgneg, hneg, kneg, cneg host_t / pass_row).
HOST SYNCS: none in a replayed pass (the captures synchronize, once per graph).
"""
import torch

import r2_i4a_lean as LN
import r2_i4d_keep as KP
import r2_i4g_graph as GR
import r2_i4h_graph as DG
import r2_i4k_combo as CB
import r2_i4u_hoist as HU

SNEG = (None, "no_hcounts")
CNEG_OK = (None, "host_t", "pass_row")          # CB's controls that the stacked keep arm keeps (tests only)
ARMS = ("L4PTFDGU", "L4PTFDGUD", "L4PTFKU", "L4PTFDGUK", "L4PTFDGUDKW")


def expected_delta_s(stats: bool, H: int, B: int, l: int) -> dict:
    """The host counts per layer that a replay of the joint section must add: GR.expected_delta + the hoist's (group
    H): passes 1 and layers 1 at layer 0, layers 1 at the others (module docstring)."""
    d = GR.expected_delta(stats, H, B)
    d["H"] = dict(passes=1, layers=1) if int(l) == 0 else dict(layers=1)
    return d


def _counts_s(T, vs) -> dict:
    c = GR._counts(T, vs)
    c["H"] = dict(T.hoist)
    return c


_CLASSES = {}


def stack_classes(STF) -> dict:
    """{arm: (state class, runner class)} for ARMS, built on the spec_tentative_fork module STF."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF, _make(STF))
    return _CLASSES[key][1]


def _make(STF):
    SpecError = STF.SpecError
    VR = STF.VR
    lean = LN.lean_classes(STF)
    SU, RU = HU.hoist_classes(STF, lean)
    SKg, RKg = CB.combo_classes(STF)[:2]

    class _SUa(SU):
        """HU's TentStateFDU with (hneg, vs) from self._uinit (module docstring, THE CLASS STACK)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None):
            u = self.__dict__.get("_uinit") or {}
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, hneg=u.get("hneg"),
                             vs=u.get("vs"))

    class _GraphS:
        """GR's per-layer graph wrapper with the hoist's host counts (module docstring); a mixin placed before GR's
        TentStateFDG in the MRO."""

        def _snew(self, sneg):
            if sneg not in SNEG:
                raise SpecError("unknown stack control %r (one of %s)" % (sneg, SNEG))
            self.sneg = sneg

        def _apply(self, delta, vs):
            """GR._apply (GR:229-235) + the hoist's group H."""
            if self.gneg == "no_counts":                               # GR's NEGATIVE CONTROL (tests only)
                return
            super()._apply(delta, vs)
            if self.sneg == "no_hcounts":                              # NEGATIVE CONTROL (tests only)
                return
            for k, v in delta.get("H", {}).items():
                self.hoist[k] = self.hoist.get(k, 0) + v

        def _gsection(self, l, eng, twin, sel, q4, k4, v4, pbs, plan, section, stub: str, copy_skip: bool):
            """GR:237-280 line for line: `section` in place of the parent's _rows_union, the stub named `stub`, the skip
            copy when copy_skip, the counter groups with H (_counts_s / expected_delta_s)."""
            vs = plan.vs
            why = self._why(l, vs, plan)
            key = None if why is not None else self._key(plan)
            g = self._g.get(l)
            if why is None and g is not None and g["key"] != key and self.gneg != "no_key":
                why = "key"
            if why is not None:                                        # EAGER: the section on the real tensors
                self.graph["eager"] += 1
                self.graph["eager_why"][why] = self.graph["eager_why"].get(why, 0) + 1
                return section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            sq, sk, sv, spb, splan = self._static(q4, k4, v4, pbs, plan)
            if g is None and self._warm.get(l) != key:                 # WARM: eager on the static inputs
                self._warm[l] = key
                self.graph["warm"] += 1
                return section(l, eng, twin, sel, sq, sk, sv, spb, splan)
            skip = self._resid_eager(l, eng, twin, sel, plan)          # EAGER: the residency (the host stamp)
            if copy_skip:
                self._sb["skip"].copy_(skip)
            if g is None:                                              # CAPTURE (executes nothing), then replay
                sk0 = self._sb["skip"]
                before = _counts_s(self, vs)
                setattr(self, stub, lambda *a, **k: sk0)                # the stub: the residency ran above
                try:
                    graph, out = self.backend.capture(
                        lambda: section(l, eng, twin, sel, sq, sk, sv, spb, splan), self, vs)
                finally:
                    self.__dict__.pop(stub, None)
                d = GR._delta(before, _counts_s(self, vs))
                want = expected_delta_s(bool(vs.ops.stats), self.H, self.B, l)
                if d != want:
                    self.graph["delta_bad"].append((l, d, want))
                self.graph["deltas"].append((l, d))
                g = self._g[l] = dict(key=key, graph=graph, out=out, delta=d)
                self.graph["captures"] += 1
                self.graph["capture_passes"].append(int(self.calls["joint_passes"]))
                graph.replay()                                         # the counts of this pass ran at the capture
                self.graph["replays"] += 1
                return out
            g["graph"].replay()
            self._apply(g["delta"], vs)
            self.graph["replays"] += 1
            return g["out"]

    # ------------------------------------------------------------------ L4PTFDGU, L4PTFDGUD (no keep)
    SG0, RG0 = GR.graph_classes(STF, (_SUa, RU))

    class TentStateFDGU(_GraphS, SG0):
        """L4PTFD + the joint section graphs + the hoist inside them (no draft graphs)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, hneg=None,
                     vs=None, sneg=None):
            self.__dict__.setdefault("_uinit", dict(hneg=hneg, vs=vs))
            self._snew(self.__dict__.get("_sinit", sneg))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg)

        def _usection(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            return SU._rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan)     # HU:318-452

        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            return self._gsection(l, eng, twin, sel, q4, k4, v4, pbs, plan, self._usection, "_resid",
                                  bool(plan.vs.ops.stats))

    class TentRunnerFDGU(RG0):
        """The runner of L4PTFDGU (tests only). Keyword arguments: GR's (gneg, graph_backend), HU's (hneg), sneg."""

        def __init__(self, *args, sneg=None, **kw):
            if sneg not in SNEG:
                raise SpecError("unknown stack control %r (one of %s)" % (sneg, SNEG))
            self._sneg = sneg
            super().__init__(*args, **kw)
            self.meta.update(stack=dict(arm=self._arm_name(), sneg=sneg))

        def _arm_name(self):
            return "L4PTFDGU"

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGU(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                 gneg=self._gneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg)

    SGD0, RGD0 = DG.draft_classes(STF, (TentStateFDGU, TentRunnerFDGU))

    class TentStateFDGUD(SGD0):
        """The arm L4PTFDGUD's state (module docstring)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, dgneg=None,
                     hneg=None, vs=None, sneg=None):
            self._uinit = dict(hneg=hneg, vs=vs)
            self._sinit = sneg
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg, dgneg=dgneg)
            self._vs = vs                                              # DG set it to None (module docstring)

    class TentRunnerFDGUD(RGD0):
        """The arm L4PTFDGUD. Keyword arguments: DG's (dgneg, graph_backend; GR defaults to r2_i4g_graph), GR's
        (gneg), HU's (hneg), sneg (tests only)."""

        def __init__(self, *args, **kw):
            if kw.get("graph_backend") is None and kw.get("GR") is None:
                kw["GR"] = GR
            super().__init__(*args, **kw)

        def _arm_name(self):
            return "L4PTFDGUD"

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGUD(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                  gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg)

    # ------------------------------------------------------------------ L4PTFKU, L4PTFDGUK, L4PTFDGUDKW (the keep)
    class TentStateFKU(SKg, _SUa):
        """CB's graph-safe keep (TentStateFKg, keep on; kwin as given) with HU's hoist: the combined section (eager
        here; L4PTFDGUK / L4PTFDGUDKW replay it from graphs)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, hneg=None, vs=None,
                     **kk):
            self.__dict__.setdefault("_uinit", dict(hneg=hneg, vs=vs))
            self.__dict__.setdefault("_kinit", dict(kk))
            ki = self.__dict__["_kinit"]
            if ki.get("acct"):
                raise SpecError("the stacked keep arm has no accounting option (acct): r2_i4k_combo's arm has it")
            if ki.get("cneg") not in CNEG_OK:
                raise SpecError("stacked keep: combo control %r not kept (one of %s)" % (ki.get("cneg"), CNEG_OK))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg)

        def _section(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            """HU:318-452 (the hoisted joint section) with step 5's store fill replaced by the keep, as CB:198-325
            (CB _keep_fill, KP slot_ids); acct is off (refused). Returns (B, R, Hq x D): a view of vs.fout."""
            vs = plan.vs
            ops = vs.ops
            rs, B, R, bs = self.rs, self.B, plan.R, self.bs
            E = B * R
            Hq, Dh = q4.shape[2], q4.shape[3]
            chk, st = ops.check, ops.stats
            kern = rs.ops.kernels
            qc = self.qc
            if l == 0 and not (self.hneg == "stale_pass" and self._cur is not None):   # HU NEGATIVE CONTROL
                self._cur = self._hoist(eng, pbs, plan, vs)
            hp = self._cur
            if hp is None or hp.R != R:
                raise SpecError("L4PTFDGUDKW: layer %d of a joint pass without its hoist (R %d)" % (l, R))
            self.hoist["layers"] += 1
            qb, qb3, qb4, P, past = hp.qb, hp.qb3, hp.qb4, hp.P, hp.past
            hasq = (sel == qb4).any(-1)                                # (H, B, R): the head selects block qb
            if self.hooks is not None and hasattr(self.hooks, "on_select"):    # CHECK-ONLY (WS-G)
                for r in range(R):
                    self.hooks.on_select(l, plan.kind, r, pbs[r], plan.verify[r], sel[:, :, r])
            VR._mark(vs, l, "resid")
            # 2. THE RESIDENCY, one per layer and pass: row 0's selection (kept, D1); kwin: from the kept store
            s0 = sel[:, :, 0]
            inj = (s0 > qb3) & ~hasq[:, :, 0:1]
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            self._in_union = True
            try:
                skip0 = self._resid_joint(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, vs)
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
            # 4. the chain images (the hoisted mask)
            self._chain_h(l, eng, hp)
            VR._mark(vs, l, "union")
            # 5. where the other selected blocks live; THE KEEP in place of the scratch fill
            full, res = hp.full, hp.res                                # (R, B)
            if self.fneg == "window_tail":                             # NEGATIVE CONTROL: block qb from the tail slot
                sel_c = torch.where(sel > qb4, vs.minus1, sel)
                is_q = torch.zeros_like(sel, dtype=torch.bool)
            else:
                sel_c = torch.where(sel >= qb4, vs.minus1, sel)
                is_q = sel == qb4
            is_n = sel == hp.qb1_4
            loc = self._locate_h(vs, eng, sel_c, full, res, hp, l)
            present = loc["present"] | is_q | is_n
            VR._mark(vs, l, "fetch")
            lst = loc["lst"]
            lst_v = lst
            lphys, valid = self._keep_fill(l, vs, eng, twin, lst, loc, kern)   # CB:327-354 (KP docstring 2)
            phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), lphys))
            if chk or ops.record:                                      # the slot-ordered ids (check / record)
                lst_v = KP.slot_ids(self.kid, 0 if self.kneg == "layer_blind" else l, vs.minus1)
            vs.calls["gather_host"] += 2
            vs.calls["fills"] += 1
            VR._mark(vs, l, "attn")
            # 6. the row lists (last_len: the hoist's) and ONE fork call for every row of every request
            if ops.neg == "fork_union_rows":                           # NEGATIVE CONTROL (WS-C): VR's lists
                VR._fork_lists(vs, sel, dict(present=present, phys=phys), eng, P)
            else:
                self._lists_h(vs, sel, present, phys)
            if self.tneg == "no_causal":                               # NEGATIVE CONTROL: the own block whole
                vs.plast.fill_(bs)
            qf = q4.reshape(E, Hq, Dh).contiguous()
            VR._fork_call(vs, qf, vs.fout, vs.fork_cu, vs.fork_seq, vs.fork_tab, vs.fork_ext, vs.fork_len, vs.lists,
                          vs.fork_qmax)
            vs.calls["attn"] += 1
            out = vs.fout.reshape(B, R, -1)
            if st:                                                     # ACCOUNTING (device counters), STF:187
                s = vs.stat
                roles = (full | res).t()
                s["rows_full"] += full.sum()
                s["rows_res"] += res.sum()
                s["win_sel"] += ((s0 >= 0) & plan.act[0].view(1, B, 1)).sum()
                s["win_host"] += ((eng._load_mask >= 0) & ~skip0).sum()
                s["past_edge"] += (past & roles).sum()
                s["f4_rows"] += ((~past) & (P % bs == bs - 1) & roles).sum()
            if ops.record:                                             # tests: the byte checks (i3_world)
                vs.log.append(dict(kind="fork_t", layer=l, sel=sel.clone(), win=eng._block_map.clone(),
                                   lst=lst_v.clone(), present=present.clone(), phys=phys.clone(), qb=qb.clone(),
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
            if chk:                                                    # CHECK-ONLY, STF:212-240
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
                vis_cnt, bad, want, role, b0, tail = self._vb(vs, eng, sel, present, lst_v, full, res, P, qb, l)
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
                vs.stat_row["visible"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).view(H, B, R).sum(
                    (0, 1))
                vs.stat_row["selected"] += (sel >= 0).sum(-1).sum((0, 1))
                cbad, crows, creads = self._chain_check(l, eng, sel, P, qb, past, full | res)
                s["chain_bad"] += cbad
                s["chain_rows"] += crows
                s["chain_reads"] += creads
            if vs.fa_check is not None and l == 0 and "F" not in vs.fa_check:
                VR._fork_probe(vs, qf)                                 # CHECK-ONLY: "F" and "RA" (one host sync)
            VR._mark(vs, l, "post")
            return out

    class TentRunnerFKU(RKg, RU):
        """L4PTFKU (eager; tests only): CB's TentRunnerFKg (keep, kwin, kcheck, kneg, kslots, cneg) + HU (hneg)."""

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFKU(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                hneg=self._hneg, vs=self.vs, cneg=self._cneg, **self._k)

    SG1, RG1 = GR.graph_classes(STF, (TentStateFKU, TentRunnerFKU))

    class TentStateFDGUK(_GraphS, SG1):
        """L4PTFKU's combined section from per-layer graphs (no draft graphs), with I4K's eager parts."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, hneg=None,
                     vs=None, sneg=None, **kk):
            self.__dict__.setdefault("_uinit", dict(hneg=hneg, vs=vs))
            self.__dict__.setdefault("_kinit", dict(kk))
            self._snew(self.__dict__.get("_sinit", sneg))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg)

        def _resid_eager(self, l, eng, twin, sel, plan):
            """GR:218-227 with the keep's joint residency (= CB:402-411)."""
            B = self.B
            qb = self.qc // self.bs
            qb3 = qb.view(1, B, 1)
            s0 = sel[:, :, 0]
            hasq0 = (s0 == qb3).any(-1, keepdim=True)
            inj = (s0 > qb3) & ~hasq0
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            return self._resid_joint(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, plan.vs)

        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            self._kpre(l, plan.vs)                                     # EAGER: the pass index, once-made tensors
            out = self._gsection(l, eng, twin, sel, q4, k4, v4, pbs, plan, self._section, "_resid_joint",
                                 bool(plan.vs.ops.stats or self.kacct))
            self._kpost(l)                                             # EAGER: the counter flush (layer L - 1)
            return out

    class TentRunnerFDGUK(RG1):
        """The runner of L4PTFDGUK (tests only). Keyword arguments: KP's / CB's keep options, GR's, HU's, sneg."""

        def __init__(self, *args, sneg=None, **kw):
            if sneg not in SNEG:
                raise SpecError("unknown stack control %r (one of %s)" % (sneg, SNEG))
            self._sneg = sneg
            super().__init__(*args, **kw)
            self.meta.update(stack=dict(arm=self._arm_name(), sneg=sneg))

        def _arm_name(self):
            return "L4PTFDGUK"

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGUK(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                  gneg=self._gneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg, cneg=self._cneg,
                                  **self._k)

    SGD1, RGD1 = DG.draft_classes(STF, (TentStateFDGUK, TentRunnerFDGUK))

    class TentStateFDGUDK(SGD1):
        """The arm L4PTFDGUDKW's state (module docstring)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, dgneg=None,
                     hneg=None, vs=None, sneg=None, **kk):
            self._uinit = dict(hneg=hneg, vs=vs)
            self._kinit = dict(kk)
            self._sinit = sneg
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg, dgneg=dgneg)
            self._vs = vs                                              # DG set it to None (module docstring)

    class TentRunnerFDGUDK(RGD1):
        """The arm L4PTFDGUDKW (kwin=True; the cell passes keep=True, kwin=True, kcheck in check cells). Keyword
        arguments: DG's (dgneg, graph_backend; GR defaults to r2_i4g_graph), GR's, KP's / CB's, HU's, sneg."""

        def __init__(self, *args, **kw):
            if kw.get("graph_backend") is None and kw.get("GR") is None:
                kw["GR"] = GR
            super().__init__(*args, **kw)

        def _arm_name(self):
            return "L4PTFDGUDKW"

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGUDK(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                   gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs,
                                   sneg=self._sneg, cneg=self._cneg, **self._k)

    return dict(L4PTFDGU=(TentStateFDGU, TentRunnerFDGU), L4PTFDGUD=(TentStateFDGUD, TentRunnerFDGUD),
                L4PTFKU=(TentStateFKU, TentRunnerFKU), L4PTFDGUK=(TentStateFDGUK, TentRunnerFDGUK),
                L4PTFDGUDKW=(TentStateFDGUDK, TentRunnerFDGUDK))
