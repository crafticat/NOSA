"""Rung I4K of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the COMBINED arm of
the two measured levers of the specKV-on-cached-NOSA ladder. ARM L4PTFDGKW = L4PTFD (rung I4A) + the store keep and the
window loads from the kept store (rung I4D, r2_i4d_keep.py, arm L4PTFKW) + the per-layer CUDA graphs of the joint
pass's union + fork section (rung I4G, r2_i4g_graph.py, arm L4PTFDG). The keep's map lookup, slot assignment, miss fill
and page-id remap run INSIDE the captured section, so a replay launches them, not the host one by one. OPT-IN:
benchmarks/Efficiency/r2_i4k_cell.py (arm L4PTFDGKW); no nosi module and no older cell imports this file. NO NEW KERNEL
and no kernel edit: torch ops, the shipped gathers and torch.cuda.CUDAGraph (through r2_i4g_graph). This file imports
torch, r2_i4d_keep and r2_i4g_graph only.

ABBREVIATIONS. KP = r2_i4d_keep.py (99c231e), GR = r2_i4g_graph.py (d7997e6), STF = nosi/nosi/spec_tentative_fork.py,
ST = nosi/nosi/spec_tentative.py.

THE SPLIT (CODE). Which keep ops go into layer l's graph and which stay eager:
  CAPTURED (in the section of layer l; the keep adds KP.KEEP_EXTRA_OPS = 30 ops per layer there):
    KP:116-157 keep_assign: the map lookup (one broadcast compare), the eviction order (one stable sort), the slot of
        every miss, the map update (two scatters). Device ops on static shapes, no host read. ONE CHANGE: the step t.
        KP:133 / KP:151 use int(t), t = g x L + l with g the HOST pass index: a replay would keep the captured t (the
        ages and the "alive" test freeze). Here t is a device scalar, kpass_d x L + l (kpass_d: a persistent int64,
        set once per pass, eagerly); keep_assign_d below is keep_assign with that one change.
    KP:465-492 _keep_fill: host_x fill / scatter, the miss gathers (the shipped gathers, the same calls as STF:175-176,
        which I4G's graphs already hold), the counters, the page-id remap. ONE CHANGE: the counter row. KP:480
        self._krow(g) picks the row of kc by the host pass index g: a replay would add into the capture pass's row.
        Here the section adds into the persistent accumulator kcur (8,), and the eager path adds kcur into the pass's
        row of kc once per pass (after layer L - 1) and zeroes it.
    KP:494-502 _acct with KP:160-167 acct_repeat, KP:427-436 _acct_win (acct only): device ops; the same counter change.
        _acct_win reads eng._load_mask (written by the eager residency before the replay) and the static skip mask.
    KP:504-530 _keep_bytes, KP:170-174 slot_ids (check rounds only): device ops. The scratch store _kfresh and the page
        bases _ksb1 are made EAGERLY before the first capture (_kpre). The unmetered gathers come from the meter at call
        time (property k_orig), so a capture sees the kernels the meter sees.
  EAGER (per layer, before the replay):
    KP:381-425 _resid_k (kwin): it IS the joint residency (STF:149): the diff, pool_update with the HOST stamp
        eng._pool_stamp_base (ST:461-466; I4G keeps the residency eager for this reason, a device stamp is a kernel
        edit), pool_swap, the block map, then the lookup of row 0's loads in the map (KP:404-410), the host gather pair
        and the store gather pair (KP:411-416) and its counter (KP:417, a host row index: eager is right). The lookup
        must read the map BEFORE step 5 of the same layer writes it, and after pool_update: it stays in the residency.
        KP.KEEP_WIN_EXTRA_OPS = 12 ops per layer stay eager.
    KP:259-260 the host pass index (self.kpass += 1 at layer 0): a host statement (a replay skips it): moved to _kpre.
    NEW per joint pass: kpass_d.fill_ (layer 0) and the counter flush add_ + zero_ (after layer L - 1): EAGER_PASS_OPS.
  So of the keep's 42 ops per layer (30 + 12), 30 are replayed and 12 stay eager; plus 3 eager ops per pass.

THE GRAPH WRAPPER (TentStateFDGK._graph_section) = GR:237-280 line for line with three changes: (1) the section is the
keep's section (TentStateFKg._section = KP:237-378 without the host pass index, the residency through _resid_joint);
(2) the capture's stub answers _resid_joint (the residency of the keep: _resid_k with kwin, else _resid); (3) the static
skip mask is also copied when acct reads it (KP:269-270). _resid_eager = GR:218-227 with _resid_joint. The policy (when
a layer runs eagerly), the frozen key, the static inputs, the skipped host counts (GR.SKIPPED: the keep adds no host
count to the section), the address rule (InPlaceDict) and the pool: GR's, unchanged.

THE MAP IS PERSISTENT: kid / kage (KP new_map, made in TentStateFK.__init__), kprev (acct), kcur, kpass_d, kchk: made
once per state, written IN PLACE (scatter_, add_, copy_, masked_fill_ on a fresh tensor only). A replay updates them.

THE SAME BITS (the claim; CPU: tests_cpu/test_i4k.py; GPU: the cells' EQF / KEEP-EQ gates): a replay runs the recorded
ops on the same tensors; t has the same integer value as KP's int(t) (kage and the alive test equal); every counter
lands in the same pass row. So L4PTFDGKW = L4PTFKW = L4PTFD in tokens, fingerprints, acceptance record and state
checksums, and its keep record (listed, fetched, hits, window hits per pass) = L4PTFKW's.

NEGATIVE CONTROLS (tests only; cneg): "host_t" (the step t as a host int inside the section: KP.keep_assign, the
captured t freezes), "pass_row" (the counters into kc's row of the host pass index inside the section: a replay adds into
the capture pass's row), "resid_in_graph" (the capture's stub does not answer the keep's residency: the residency is
recorded into the graph as well).
"""
import torch

import r2_i4d_keep as KP
import r2_i4g_graph as GR

KNEG = KP.KNEG
GNEG = GR.GNEG
CNEG = (None, "host_t", "pass_row", "resid_in_graph")
EAGER_PASS_OPS = 3                 # kpass_d.fill_ (layer 0) + the flush add_ and zero_ (layer L - 1), per joint pass
CAPTURED_KEEP_OPS = KP.KEEP_EXTRA_OPS        # per layer, inside the graph (KP's count: keep, acct off, checks off)
EAGER_KEEP_OPS = KP.KEEP_WIN_EXTRA_OPS       # per layer, in the eager residency (kwin)


def keep_assign_d(kid, kage, lst, l: int, td, L: int, kneg=None, block=None, chk=None, stats: bool = False,
                  big: int = 2 ** 31 - 1) -> dict:
    """KP.keep_assign (KP:116-157) line for line, except the step: td is a 0-dim int64 DEVICE tensor (= KP's int(t));
    the alive test reads it and the ages are scattered from it, so a graph replay reads the current pass's step."""
    key_l = 0 if kneg == "layer_blind" else int(l) * KP.KM                # NEGATIVE CONTROL: no layer in the key
    need = lst + key_l                                                  # an invalid entry (big) never matches a key
    k1, a1 = kid[..., 1:], kage[..., 1:]
    E = need.unsqueeze(-1) == k1.unsqueeze(-2)                          # (H, B, needs, slots)
    hv, hs = E.view(torch.uint8).max(-1)                                # keys distinct: at most one 1 per need
    hit = hv.view(torch.bool)
    used = E.any(-2)                                                    # (H, B, slots): holds a needed key
    if kneg == "evict_used":                                            # NEGATIVE CONTROL: nothing protected
        used = torch.zeros_like(used)
    alive = a1 > (td - int(L))                                          # I4K: the device step (KP:133 int(t))
    score = torch.where(alive, KP.ALIVE_BASE - a1, a1)
    prot = used if block is None else (used | block)
    score.masked_fill_(prot, KP.PROTECT)
    order = torch.sort(score, dim=-1, stable=True).indices
    valid = lst < big
    miss = valid & ~hit
    rank = miss.cumsum(-1) - 1
    ms = order.gather(-1, rank.clamp(min=0))
    out = dict(hit=hit, miss=miss, valid=valid)
    if stats:                                                           # a victim that was alive (capacity pressure)
        out["victims_alive"] = (miss & (alive & (k1 >= 0)).gather(-1, ms)).sum()
    if chk is not None:                                                 # CHECK-ONLY: (iii) no overflow
        nfree = (~prot).sum(-1, keepdim=True)
        chk["overflow"] += (miss & (rank >= nfree)).sum()
    slot = torch.where(hit, hs, ms) + 1
    widx = torch.where(miss, slot, 0)
    kid.scatter_(-1, widx, need)
    ws = torch.where(valid, slot, 0)
    kage.scatter_(-1, ws, td.expand(ws.shape))                          # I4K: the device step (KP:151 int(t))
    out.update(slot=slot, widx=widx)
    if chk is not None:                                                 # CHECK-ONLY: (ii) distinct keys, (iv) present
        s = torch.sort(kid[..., 1:], dim=-1).values
        chk["dup"] += ((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).sum()
        chk["missing"] += (valid & (kid.gather(-1, slot) != need)).sum()
    return out


_CLASSES = {}


def combo_classes(STF):
    """(TentStateFKg, TentRunnerFKg, TentStateFDGK, TentRunnerFDGK) built on the spec_tentative_fork module STF.
    FKg: L4PTFKW with the graph-safe keep (eager; tests compare it with KP's L4PTFKW); FDGK: the arm L4PTFDGKW."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF,) + _make(STF)
    return _CLASSES[key][1:]


def _make(STF):
    SpecError = STF.SpecError
    VR = STF.VR
    SK, RK = KP.keep_classes(STF)

    class TentStateFKg(SK):
        """KP's TentStateFK (keep on) whose joint-pass section is graph-safe (module docstring, THE SPLIT)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, cneg=None, **kw):
            kw = dict(getattr(self, "_kinit", None) or {}, **kw)              # the keep options (TentStateFDGK)
            cneg = kw.pop("cneg", cneg)
            if cneg not in CNEG:
                raise SpecError("unknown combo control %r (one of %s)" % (cneg, CNEG))
            if not kw.get("keep", True):
                raise SpecError("the combined arm needs the keep (keep=True)")
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, **kw)
            self.cneg = cneg
            self.kcur = torch.zeros((8,), dtype=torch.int64, device=self.dev)        # the section's counters
            self.kpass_d = torch.full((), -1, dtype=torch.int64, device=self.dev)    # the device pass index

        # the unmetered gathers: the meter's at call time (a capture swaps them for kernel nodes); KP sets k_orig once
        @property
        def k_orig(self):
            m = self.__dict__.get("meter")
            return m.orig if m is not None else self.__dict__.get("_k_orig0")

        @k_orig.setter
        def k_orig(self, v):
            self.__dict__["_k_orig0"] = v

        # ---------------------------------------------------------- the eager parts (per layer)
        def _kpre(self, l, vs):
            """EAGER, before layer l's section: the host pass index and the device one (layer 0); once per state the
            page bases and the check-only scratch store (so that a capture allocates neither)."""
            if l == 0:
                self.kpass += 1                                                # KP:259-260, moved out of the section
                self.kpass_d.fill_(self.kpass)
            if self._ksb1 is None:                                             # KP:486-487, made before any capture
                B = self.B
                self._ksb1 = (vs.store_b.view(1, B, 1, 1), vs.store_b.view(1, B, 1, 1) + 1)
            if self.kcheck and self._kfresh is None:                           # KP:508-510, made before any capture
                self._kfresh = (torch.empty_like(vs.ext_k), torch.empty_like(vs.ext_v), torch.empty_like(vs.ext_bias),
                                torch.full_like(vs.host_x, -1))

        def _kpost(self, l):
            if l == self.L - 1:
                self._kflush()

        def _kflush(self):
            """EAGER: the section's counters of this pass into its row of kc."""
            if self.kpass >= 0:
                self._krow(self.kpass).add_(self.kcur)
                self.kcur.zero_()

        def _resid_joint(self, l, eng, twin, ti, inj, act, drf, qb3, vs):
            """The joint pass's row-0 residency of the keep (KP:265-268): kwin -> _resid_k, else _resid."""
            if self.kwin:
                return self._resid_k(l, eng, twin, ti, inj, act, drf, qb3, vs)
            return self._resid(l, eng, twin, ti, inj, act, drf, qb3)

        def _crow(self):
            """The counter row of the section: kcur (I4K); the control pass_row: kc's row of the host pass index."""
            return self._krow(self.kpass) if self.cneg == "pass_row" else self.kcur

        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            self._kpre(l, plan.vs)
            out = self._section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            self._kpost(l)
            return out

        # ---------------------------------------------------------- the section (capturable)
        def _section(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            """KP:237-378 (TentStateFK._rows_union) line for line with keep on, except: no host pass index (KP:259-260:
            in _kpre) and the residency through _resid_joint (KP:265-268)."""
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
            skip0 = self._resid_joint(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, vs)
            if self.kacct:                                                 # I4D: the window loads the store held
                self._acct_win(l, eng, skip0)
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
            VR._mark(vs, l, "fetch")
            lst = loc["lst"]
            lst_v = lst
            lphys, valid = self._keep_fill(l, vs, eng, twin, lst, loc, kern)   # I4D: THE KEEP (KP docstring 2)
            phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), lphys))
            if chk or ops.record:                                          # the slot-ordered ids (check / record)
                lst_v = KP.slot_ids(self.kid, 0 if self.kneg == "layer_blind" else l, vs.minus1)
            if self.kacct:                                                 # I4D: THE ACCOUNTING (KP docstring 1)
                self._acct(l, lst, valid)
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
                                   lst=lst_v.clone(), present=present.clone(), phys=phys.clone(), qb=qb.clone(),
                                   qc=qc.clone(), p_rows=P.clone(), past=past.clone(), full=full.clone(),
                                   res=res.clone(), pool_map=(eng._pool_map.clone() if self.pool else None),
                                   k_gpu=eng._k_gpu.clone(), v_gpu=eng._v_gpu.clone(), bias_gpu=eng._kv_bias_gpu.clone(),
                                   k_cpu=eng._k_cpu.clone(), v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone(),
                                   tpos=self.tpos.clone(), buf_k=self.k[l].clone(), buf_v=self.v[l].clone(),
                                   buf_bias=self.bias[l].clone(), store_k=vs.ext_k.clone(), store_v=vs.ext_v.clone(),
                                   store_bias=vs.ext_bias.clone(), cb0_k=self.arena.cb0_k.clone(),
                                   cb0_v=self.arena.cb0_v.clone(), cb0_bias=self.arena.cb0_bias.clone(),
                                   tab=vs.fork_tab.clone(), ext=vs.fork_ext.clone(), ulen=vs.fork_len.clone(),
                                   pkey=vs.pkey.clone(), pslot=vs.pslot.clone(), npg=vs.npg.clone(),
                                   plast=vs.plast.clone(), q_e=qf.clone(), out=vs.fout.float().clone(),
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

        def _keep_fill(self, l, vs, eng, twin, lst, loc, kern):
            """KP:465-492 line for line, except the step (a device scalar: keep_assign_d) and the counter row (kcur)."""
            B, bs = self.B, self.bs
            chk = self.kchk if self.checking else None
            if self.cneg == "host_t":                                      # NEGATIVE CONTROL: the host step, frozen
                m = KP.keep_assign(self.kid, self.kage, lst, l, self.kpass * self.L + l, self.L, kneg=self.kneg,
                                   block=self.kblock, chk=chk, stats=self.kacct, big=VR.FORK_BIG)
            else:
                td = self.kpass_d * self.L + l                             # = KP's t, on the device
                m = keep_assign_d(self.kid, self.kage, lst, l, td, self.L, kneg=self.kneg, block=self.kblock, chk=chk,
                                  stats=self.kacct, big=VR.FORK_BIG)
            hx = vs.host_x
            hx.fill_(-1)                                                   # slot 0 (CB1) and every hit: no load
            if self.kneg != "skip_miss":                                   # NEGATIVE CONTROL: a miss not fetched
                hx.scatter_(-1, m["widx"], torch.where(m["miss"], lst, vs.minus1))
            kern["gather_k"](vs.ext_k, eng._k_cpu, hx, bs)
            kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, hx, bs)
            row = self._crow()
            row[0:2].add_(torch.stack((m["valid"], m["miss"])).sum((1, 2, 3)))
            if self.kacct:
                row[3].add_(m["victims_alive"])
            if chk is not None and self.kcheck:
                self._keep_bytes(vs, eng, twin, lst, m["valid"], m["slot"])
            sb, sb1 = self._ksb1                                           # made in _kpre (eager)
            ph = loc["phys"]
            at = (ph - sb1).clamp_(0, self.kcap - 1)                       # the list rank of a listed block
            sq = m["slot"].gather(-1, at.view(at.shape[0], B, -1)).view_as(at)
            return torch.where(loc["in_lst"], sb + sq, ph), m["valid"]

        def _acct(self, l, lst, valid):
            """KP:494-502 (keep on: valid given) with the counter row kcur."""
            rep = KP.acct_repeat(self.kprev[l], lst, valid)
            self._crow()[2].add_(rep.sum())

        def _acct_win(self, l, eng, skip0):
            """KP:427-436 with the counter row kcur."""
            lm = eng._load_mask
            ld = (lm >= 0) & ~skip0
            ids = torch.where(ld, lm, VR.FORK_BIG)
            prev = self.kprev[l]
            at = torch.searchsorted(prev, ids).clamp_(max=prev.shape[-1] - 1)
            rep = (prev.gather(-1, at) == ids) & ld
            self._crow()[4:6].add_(torch.stack((ld, rep)).sum((1, 2, 3)))

        def keep_read(self) -> dict:
            self._kflush()                                                 # a pass that ended early (none in a run)
            return super().keep_read()

    class TentRunnerFKg(RK):
        """KP's TentRunnerFK with TentStateFKg (keep on; tests compare it with KP's runner)."""

        def __init__(self, *args, keep: bool = True, cneg=None, **kw):
            if cneg not in CNEG:
                raise SpecError("unknown combo control %r (one of %s)" % (cneg, CNEG))
            if not keep:
                raise SpecError("the combined arm needs the keep (keep=True)")
            self._cneg = cneg
            super().__init__(*args, keep=True, **kw)
            self.meta.update(combo=dict(cneg=cneg, eager_pass_ops=EAGER_PASS_OPS, captured_keep_ops=CAPTURED_KEEP_OPS,
                                        eager_keep_ops=EAGER_KEEP_OPS))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFKg(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                cneg=self._cneg, **self._k)

    SG, RG = GR.graph_classes(STF, (TentStateFKg, TentRunnerFKg))

    class TentStateFDGK(SG):
        """The arm L4PTFDGKW's state: the graph-safe keep (TentStateFKg) whose section runs from per-layer CUDA graphs
        (GR's policy, key, static inputs, counts, pool)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, **kk):
            self._kinit = dict(kk)                                         # read by TentStateFKg.__init__
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg)

        def _resid_eager(self, l, eng, twin, sel, plan):
            """GR:218-227 with the keep's joint residency (_resid_joint)."""
            B = self.B
            qb = self.qc // self.bs
            qb3 = qb.view(1, B, 1)
            s0 = sel[:, :, 0]
            hasq0 = (s0 == qb3).any(-1, keepdim=True)
            inj = (s0 > qb3) & ~hasq0
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            return self._resid_joint(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, plan.vs)

        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            self._kpre(l, plan.vs)                                         # EAGER: the pass index, once-made tensors
            out = self._graph_section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            self._kpost(l)                                                 # EAGER: the counter flush (layer L - 1)
            return out

        def _graph_section(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            """GR:237-280 line for line, except the section (self._section), the stub's name (_resid_joint) and the
            skip copy (also with acct)."""
            vs = plan.vs
            why = self._why(l, vs, plan)
            key = None if why is not None else self._key(plan)
            g = self._g.get(l)
            if why is None and g is not None and g["key"] != key and self.gneg != "no_key":
                why = "key"
            if why is not None:                                        # EAGER: the keep's section on the real tensors
                self.graph["eager"] += 1
                self.graph["eager_why"][why] = self.graph["eager_why"].get(why, 0) + 1
                return self._section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            sq, sk, sv, spb, splan = self._static(q4, k4, v4, pbs, plan)
            if g is None and self._warm.get(l) != key:                 # WARM: eager on the static inputs
                self._warm[l] = key
                self.graph["warm"] += 1
                return self._section(l, eng, twin, sel, sq, sk, sv, spb, splan)
            skip = self._resid_eager(l, eng, twin, sel, plan)          # step 1: EAGER (the host stamp; kwin's lookup)
            if vs.ops.stats or self.kacct:
                self._sb["skip"].copy_(skip)
            if g is None:                                              # CAPTURE (executes nothing), then replay
                sk0 = self._sb["skip"]
                before = GR._counts(self, vs)
                if self.cneg != "resid_in_graph":                      # NEGATIVE CONTROL: no stub
                    self._resid_joint = lambda *a, **k: sk0             # the stub: the residency ran above
                try:
                    graph, out = self.backend.capture(
                        lambda: self._section(l, eng, twin, sel, sq, sk, sv, spb, splan), self, vs)
                finally:
                    self.__dict__.pop("_resid_joint", None)
                d = GR._delta(before, GR._counts(self, vs))
                want = GR.expected_delta(bool(vs.ops.stats), self.H, self.B)
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

    class TentRunnerFDGK(RG):
        """The arm L4PTFDGKW (kwin=True; kwin False: L4PTFDGK, tests only): L4PTFD + the keep + the section graphs.
        Keyword arguments: KP's (acct, kneg, kslots, kcheck, kwin), GR's (gneg, graph_backend), cneg (tests only)."""

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGK(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                 gneg=self._gneg, cneg=self._cneg, **self._k)

    return TentStateFKg, TentRunnerFKg, TentStateFDGK, TentRunnerFDGK
