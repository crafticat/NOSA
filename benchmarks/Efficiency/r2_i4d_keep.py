"""Rung I4D of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 4 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (R5.5 item 4, the store refetch): the ACCOUNTING option and the
arm L4PTFK = L4PTFD (rung I4A: L4PTF with the lean one-row pass) with a STORE THAT IS KEPT ACROSS ROUNDS. The design,
the rule and the proof: this docstring. OPT-IN: benchmarks/Efficiency/r2_i4d_cell.py (arms L4PTFD with --store-acct 1, and L4PTFK); no nosi
module, no I3 / I4A file and no older cell imports this file. NO NEW KERNEL and no new kernel launch kind: the shipped
host gathers (rs.ops.kernels gather_k / gather_v_bias, the same calls as L4PTF's fill) and torch ops (host bookkeeping
on the device tensors, no host read).

WHY A CLASS FACTORY (as r2_i4a_lean.lean_classes): keep_classes(STF) builds the classes on the GIVEN
spec_tentative_fork module (the GPU cell passes nosi.spec_tentative_fork; the CPU tests pass the CPU world's synthetic
package copy), so this file imports torch and r2_i4a_lean only.

THE PROBLEM (LEDGER R5.5 item 4). L4PTF's store (r2_i3_arena.ChainArena: B x X slots, X = 1 + (K - 1) x 64 = 129, slot
0 = CB1) is ONE store for ALL layers: at every layer of every joint pass, spec_tentative_fork.TentStateF._rows_union
(step 5) writes the layer's listed host blocks (verify_rows._fork_locate's lst: the distinct ids that a verify row
selects and only the host holds, ascending, at most X - 1 per (head, request)) into slots 1 .. n of the store with the
shipped gathers. So a block that rows 1-3 need in round r and again in round r + 1 is fetched again (L4PT's per-row
residency keeps it in the window). MEASURED: host-gather spans L4PTF - L4PT = 3.218 ms per round at C 128, 15.506 ms at
C 73 (same job); NEW host blocks +60.7 % / +12.0 %.

1. THE ACCOUNTING (acct=True; L4PTFD or L4PTFK). Per joint pass g and layer l, per (head h, request b): listed = the
   valid entries of lst; REPEAT = the entries of lst that were in lst of the SAME (layer, head, request) at the
   previous joint pass (= in the scratch store at the previous round's fill of this layer; the store holds one layer
   at a time, so a block of another layer is never a repeat). Sorted search against the previous list (prev: (L, H,
   B, X - 1) int64, FORK_BIG past its end). Per pass: listed, repeat (device counters, one host read in read()).
   PART B (the same option): row 0's WINDOW host loads of the joint pass (TentState._resid: the diff's loads that are
   not skipped) and those whose block the previous pass's store list of the same (layer, head, request) held. WHY: the
   next round's row 0 sits near the old rows 1-3, so blocks that only the store held are loaded AGAIN, now into the
   window (L4PT's per-row residency had loaded them into the window already). Part A cannot see this refetch.
   Bytes = count x BLOCK_BYTES(H = 1) = 64 x D x 2 x 2 + 64 x 2 = 32,896 B per (head, block) at D 128 (K + V + bias;
   the store slot of one request holds the H heads: 65,536 B K + V + 256 B bias at H 2).

2. THE KEEP (keep=True; arm L4PTFK). A BLOCK-ID MAP over the SAME store slots: kid (H, B, X) int64, kage (H, B, X)
   int64; column x = store slot x of (head, request) (column 0 = slot 0 = CB1: never mapped, a write target for masked
   entries only, never read). KEY = layer x 2^32 + block id (a block id < FORK_BIG = 2^31 - 1), so a slot names one
   (layer, block). Per layer l of joint pass g, at step t = g L + l (keep_assign):
     a. HIT: lst's key found in the map (one broadcast compare (H, B, X - 1, X - 1); keys are distinct, so at most
        one slot): the slot's bytes are reused, nothing is fetched.
     b. MISS: a listed block not in the map. Its slot is chosen by THE EVICTION RULE below, its key and age are
        written, and the slot is fetched with the shipped gathers (host_x: the miss's block id at its slot, -1
        elsewhere: the kernels skip -1).
     c. the fork's page id of a listed block (verify_rows._fork_locate: store_b + 1 + its list rank) becomes
        store_b + its keep slot; every other page id is unchanged.
   THE EVICTION RULE (a cyclic-layer approximation of Belady's rule; each layer is visited once per joint pass):
     PROTECTED: a slot that holds a key of THIS layer's list (a hit) is never chosen.
     DEAD first: an empty slot (age EMPTY_AGE) or a slot whose last use t' <= t - L (not used at its layer's most recent
       visit: not predicted to be used again), the oldest first.
     ALIVE last: a slot used at its layer's most recent visit (t' > t - L; its next use is predicted at t' + L), the
       MOST RECENT first (its next use is the furthest).
     Ties: the lower slot first (stable sort). The i-th miss of (h, b) (list order) takes the i-th slot of this order.
   CAPACITY (the proof; CPU tests check every step on random sequences against a pure Python model, and the GPU check
   cell counts every term as a MUST gate):
     (i)   the map and the store are the arena's existing tensors / X - 1 columns: NOTHING GROWS (kid / kage are 2 x H
           x B x X x 8 B of bookkeeping: 1.3 MB at B 318; the store is the arena's, unchanged).
     (ii)  keys in the map are DISTINCT per (h, b): a key is written only on a miss (not in the map), and lst's ids are
           distinct (sorted unique), so the misses of one step are distinct keys.
     (iii) hits occupy distinct slots (ii), so #non-protected slots = (X - 1) - #hits. lst has at most X - 1 entries
           (_fork_locate: rank < cap), so #misses = #valid - #hits <= (X - 1) - #hits = #non-protected slots: the
           rank of every miss is below the number of non-protected slots, the order puts every non-protected slot
           before every protected one (score PROTECT = 2^62 > every other score), so every miss gets a DISTINCT
           NON-PROTECTED slot (the order is a permutation). No needed key is evicted; no overflow.
     (iv)  after the step every listed key of layer l is in the map at the slot that the fork reads (missing = 0).
   THE SAME BYTES: a store slot is written only by the fill of a miss (the shipped gathers) and slot 0 by the chain image
   CB1 (spec_tentative_fork._chain); a listed host block is a committed full block below qb (sel >= qb is removed
   before the list), its host rows (_k_cpu / _v_cpu of layer l) and its bias rows (twin.total_cis at positions < qc)
   do not change after the block is written to the host. So a hit reads the bytes that a fresh fetch writes: L4PTFK's
   tokens, logits, fingerprints, acceptance record and every state checksum equal L4PTFD's (EQF; the store layout is
   not state). CHECK-ONLY (kcheck, the miniature): a FRESH fetch into a scratch store (the original gathers, not
   metered) and a bitwise compare of every listed block's K, V and bias rows (bytes_bad).
3. HOST BOOKKEEPING ONLY: about 30 small torch ops per layer and joint pass (KEEP_EXTRA_OPS, an independent aten op
   counter in the CPU tests); no host read in a pass (the counters are device tensors; read() reads them once).
4. THE WINDOW FROM THE KEPT STORE (kwin=True, needs keep; arm L4PTFKW): in row 0's residency of the joint pass
   (_resid_k = TentState._resid op for op except the gather), a window load whose (layer, block) key the map holds is
   copied from its store slot with the SAME shipped gathers (the store as the source tensor, the slot as the block
   id; the KV meter counts host sources only), the other loads come from the host as before. The window gets the same
   bytes (the slot holds the host block's K, V and bias rows: 2 above). Done BEFORE step 5 of the same layer, so the
   slot is not yet re-used. Ages are not touched by these reads. CHECK-ONLY (kcheck): the loaded window rows against a
   fresh host gather (bytes_bad, win_checked). Extra ops per layer: KEEP_WIN_EXTRA_OPS.

NEGATIVE CONTROLS (tests only; kneg): "layer_blind" (the key without the layer: a hit can return another layer's
bytes), "skip_miss" (a miss is mapped but not fetched), "evict_used" (the protection is off: a miss can overwrite a
slot that this layer's list needs), "win_skip" (kwin: a window load that the map holds is made neither from the store
nor from the host).
"""
import torch

import r2_i4a_lean as LN

KM = 1 << 32                        # key = layer x KM + block id
EMPTY_AGE = -(1 << 40)              # the age of an empty slot (dead, the oldest)
ALIVE_BASE = 1 << 50                # the score of an alive slot: ALIVE_BASE - its last use
PROTECT = 1 << 62                   # the score of a protected slot (never chosen)
KNEG = (None, "layer_blind", "skip_miss", "evict_used", "win_skip")
CHECK_KEYS = ("overflow", "dup", "missing", "bytes_bad", "bytes_checked", "win_checked")
# THE STATED OP COUNT (aten compute ops; tests_cpu/test_i4d.py counts them independently): L4PTFK's joint pass minus
# L4PTFD's, per layer (the keep bookkeeping, its counters and the page-id remap in place of the 2 ops of the scratch
# list copy). Accounting and checks off.
KEEP_EXTRA_OPS = 30
# the same for the window part (kwin: the map lookup of row 0's loads, the second gather pair, its counter)
KEEP_WIN_EXTRA_OPS = 12


def block_bytes(D: int, bs: int = 64, H: int = 1, nbytes: int = 2) -> int:
    """The bytes of one (head, block) of the store: K and V rows (bs x D each) and the bias rows (bs), bf16 on the GPU.
    D 128: 32,896 B."""
    return int(H) * int(bs) * (2 * int(D) + 1) * int(nbytes)


def new_map(H: int, B: int, X: int, dev):
    """(kid, kage): (H, B, X) int64; column x = store slot x; empty: kid -1, kage EMPTY_AGE."""
    kid = torch.full((H, B, X), -1, dtype=torch.int64, device=dev)
    kage = torch.full((H, B, X), EMPTY_AGE, dtype=torch.int64, device=dev)
    return kid, kage


def keep_assign(kid, kage, lst, l: int, t: int, L: int, kneg=None, block=None, chk=None, stats: bool = False,
                big: int = 2 ** 31 - 1) -> dict:
    """ONE keep step (module docstring, 2.a-b and THE EVICTION RULE) for layer l at step t: updates kid / kage IN PLACE.
    lst (H, B, X - 1) int64, ascending, `big` past its end (verify_rows._fork_locate's lst, may be a strided view).
    block (tests only): a bool mask (..., X - 1) of slots that the map must not use (fewer slots than the store).
    chk: a dict of device counters (CHECK_KEYS) or None; stats: also the victims that were alive. Returns dict(slot
    (H, B, X - 1): the store slot 1 .. X - 1 of every list entry (meaningless where not valid), hit, miss, valid, widx
    (the miss's slot, 0 elsewhere), victims_alive (stats) )."""
    key_l = 0 if kneg == "layer_blind" else int(l) * KM                # NEGATIVE CONTROL: no layer in the key
    need = lst + key_l                                                  # an invalid entry (big) never matches a key
    k1, a1 = kid[..., 1:], kage[..., 1:]
    E = need.unsqueeze(-1) == k1.unsqueeze(-2)                          # (H, B, needs, slots)
    hv, hs = E.view(torch.uint8).max(-1)                                # keys distinct: at most one 1 per need
    hit = hv.view(torch.bool)
    used = E.any(-2)                                                    # (H, B, slots): holds a needed key
    if kneg == "evict_used":                                            # NEGATIVE CONTROL: nothing protected
        used = torch.zeros_like(used)
    alive = a1 > (int(t) - int(L))
    score = torch.where(alive, ALIVE_BASE - a1, a1)
    prot = used if block is None else (used | block)
    score.masked_fill_(prot, PROTECT)
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
    kage.scatter_(-1, torch.where(valid, slot, 0), int(t))
    out.update(slot=slot, widx=widx)
    if chk is not None:                                                 # CHECK-ONLY: (ii) distinct keys, (iv) present
        s = torch.sort(kid[..., 1:], dim=-1).values
        chk["dup"] += ((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).sum()
        chk["missing"] += (valid & (kid.gather(-1, slot) != need)).sum()
    return out


def acct_repeat(prev, lst, valid):
    """THE ACCOUNTING of one (pass, layer): the entries of lst that the previous list of the same layer held (prev,
    ascending, big past its end; replaced by lst). Returns the bool mask (H, B, X - 1)."""
    lc = lst.contiguous()
    at = torch.searchsorted(prev, lc).clamp_(max=prev.shape[-1] - 1)
    rep = (prev.gather(-1, at) == lc) & valid
    prev.copy_(lc)
    return rep


def slot_ids(kid, l: int, minus1):
    """(H, B, X - 1): the block id that store slot 1 + j holds for layer l, -1 if it holds another layer or nothing
    (the record / check view of the store: slot-ordered, as _fork_locate's lst is for the scratch store)."""
    k1 = kid[..., 1:]
    return torch.where((k1 >= 0) & ((k1 >> 32) == int(l)), k1 & (KM - 1), minus1)


_CLASSES = {}


def keep_classes(STF):
    """(TentStateFK, TentRunnerFK) built on the spec_tentative_fork module STF (on top of r2_i4a_lean's L4PTFD)."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF,) + _make(STF)
    return _CLASSES[key][1:]


def _make(STF):
    SpecError = STF.SpecError
    VR = STF.VR
    TentStateFD, TentRunnerFD = LN.lean_classes(STF)

    class TentStateFK(TentStateFD):
        """L4PTFD's state whose union joint pass keeps the store across rounds (keep) and / or counts the repeats
        (acct). keep False and acct False: L4PTFD's values and ops (tests only; the cell uses TentStateFD then)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, keep: bool = True,
                     acct: bool = False, kneg=None, kslots=None, kcheck: bool = False, kwin: bool = False):
            if kneg not in KNEG:
                raise SpecError("unknown keep control %r (one of %s)" % (kneg, KNEG))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg)
            X = int(self.arena.X)
            H, B, dev = self.H, self.B, self.dev
            cap = X - 1
            if kslots is not None and not 1 <= int(kslots) <= cap:
                raise SpecError("kslots %r: 1 .. %d (tests only)" % (kslots, cap))
            if kcheck and B > 64:
                raise SpecError("kcheck: the fresh scratch store is the store's size: B <= 64 (got %d)" % B)
            if kwin and not keep:
                raise SpecError("kwin needs the keep (the map)")
            self.kkeep, self.kacct, self.kneg, self.kcheck = bool(keep), bool(acct), kneg, bool(kcheck)
            self.kwin = bool(kwin)
            self.kX, self.kcap = X, cap
            self.kid = self.kage = self.kblock = None
            if self.kkeep:
                self.kid, self.kage = new_map(H, B, X, dev)
                if kslots is not None and int(kslots) < cap:                   # tests only: fewer map slots
                    self.kblock = (torch.arange(cap, device=dev) >= int(kslots)).view(1, 1, cap)
            self.kprev = (torch.full((self.L, H, B, cap), VR.FORK_BIG, dtype=torch.int64, device=dev)
                          if self.kacct else None)
            # per joint pass: 0 listed, 1 misses (fetched), 2 repeat (acct), 3 victims alive (keep and acct), 4 row 0's
            # window host loads (acct), 5 of them in the previous pass's store list of the layer (acct), 6 window loads
            # served from the kept store (kwin), 7 spare
            self.kc = torch.zeros((1024, 8), dtype=torch.int64, device=dev)
            self.kchk = {k: torch.zeros((), dtype=torch.int64, device=dev) for k in CHECK_KEYS}
            self.kpass = -1                                                    # host: the joint pass index
            self.k_orig = None                                                 # the runner sets the unmetered gathers
            self._kfresh = None
            self._ksb1 = None

        def _krow(self, g: int):
            if g >= self.kc.shape[0]:                                          # host decision, no host read
                self.kc = torch.cat((self.kc, torch.zeros_like(self.kc)))
            return self.kc[g]

        # ---------------------------------------------------------- the joint pass (spec_tentative_fork step 5 changed)
        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            """spec_tentative_fork.TentStateF._rows_union, line for line, except step 5's store fill: keep (the map;
            module docstring 2) and / or acct (module docstring 1); the record and the VB check get the slot-ordered
            ids of the kept store. keep False: the scratch fill, unchanged."""
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
            if l == 0:
                self.kpass += 1                                            # I4D: host, the joint pass index
            # 2. THE RESIDENCY, one per layer and pass: row 0's selection (as TentStateU; kept, D1)
            s0 = sel[:, :, 0]
            inj = (s0 > qb3) & ~hasq[:, :, 0:1]
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            if self.kwin:                                                  # I4D: kept blocks from the store (docstring 4)
                skip0 = self._resid_k(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, vs)
            else:
                skip0 = self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)
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
            if not self.kkeep:
                phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), loc["phys"]))
            VR._mark(vs, l, "fetch")
            lst = loc["lst"]
            lst_v = lst
            if self.kkeep:                                                 # I4D: THE KEEP (module docstring 2)
                lphys, valid = self._keep_fill(l, vs, eng, twin, lst, loc, kern)
                phys = torch.where(is_q, self.cb0_id, torch.where(is_n, vs.store_b.view(1, B, 1, 1), lphys))
                if chk or ops.record:                                      # the slot-ordered ids (check / record)
                    lst_v = slot_ids(self.kid, 0 if self.kneg == "layer_blind" else l, vs.minus1)
            else:
                vs.host_x[..., 1:].copy_(torch.where(lst < VR.FORK_BIG, lst, vs.minus1))
                kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
                kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
                valid = None
            if self.kacct:                                                 # I4D: THE ACCOUNTING (module docstring 1)
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

        # ---------------------------------------------------------- I4D: the keep and the accounting
        def _resid_k(self, l, eng, twin, ti, inj, act, drf, qb3, vs):
            """spec_tentative.TentState._resid (the joint pass's row-0 residency), op for op, except the window's host
            gather (module docstring 4): a load whose (layer, block) the map holds is copied from its store slot with the
            SAME shipped gathers (the store as the source: grid (B, H, topk), block id = the store slot), the other loads
            from the host as before. The engine bytes are the same (the slot holds the host block's bytes)."""
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
            tent = (lm >= 0) & (lm > qb3)
            skip = (lm >= 0) & (drf.view(1, B, 1) | tent)
            ids = torch.where(skip, self.minus1, lm)
            key_l = 0 if self.kneg == "layer_blind" else int(l) * KM
            key = torch.where(ids >= 0, ids + key_l, -2)                   # -2: never a key, never an empty slot (-1)
            E = key.unsqueeze(-1) == self.kid[..., 1:].unsqueeze(-2)       # (H, B, topk, X - 1)
            hv, hs = E.view(torch.uint8).max(-1)
            found = hv.view(torch.bool)
            ids_h = torch.where(found, self.minus1, ids)
            ids_s = torch.where(found, hs + 1, self.minus1)
            kern["gather_k"](eng._k_gpu, eng._k_cpu, ids_h, bs)
            kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, ids_h, bs)
            if self.kneg == "win_skip":                                    # NEGATIVE CONTROL: a found load not made
                ids_s = torch.full_like(ids_s, -1)
            kern["gather_k"](eng._k_gpu, vs.ext_k, ids_s, bs)             # the kept store (not a host read)
            kern["gather_v_bias"](eng._v_gpu, vs.ext_v, eng._kv_bias_gpu, vs.ext_bias, ids_s, bs)
            self._krow(self.kpass)[6].add_(found.sum())
            if self.checking and self.kcheck:                              # CHECK-ONLY: the window rows = the host's
                self._win_bytes(eng, twin, ids, found)
            eng._block_map.masked_fill_(skip, -1)                      # the next row loads a skipped block again
            self.skipped[l] += (skip & ~tent).sum()
            if self.checking:                                          # CHECK-ONLY counters
                self.forced[l] += tent.sum()
                self.injected[l] += inj.sum()
            return skip

        def _acct_win(self, l, eng, skip0):
            """The accounting, part B (module docstring 1): row 0's window host loads of this (pass, layer) and those
            whose block the previous pass's store list of the layer held (self.kprev[l], before step 5 replaces it)."""
            lm = eng._load_mask
            ld = (lm >= 0) & ~skip0
            ids = torch.where(ld, lm, VR.FORK_BIG)
            prev = self.kprev[l]
            at = torch.searchsorted(prev, ids).clamp_(max=prev.shape[-1] - 1)
            rep = (prev.gather(-1, at) == ids) & ld
            self._krow(self.kpass)[4:6].add_(torch.stack((ld, rep)).sum((1, 2, 3)))

        def _win_bytes(self, eng, twin, ids, found):
            """CHECK-ONLY (kcheck): the window slots loaded from the kept store against a FRESH host gather (the
            original gathers, not metered) of the same blocks into a scratch window, bit for bit."""
            B, H, D, bs = self.B, self.H, self.D, self.bs
            M = int(ids.shape[-1])
            dev = self.dev
            fk = torch.zeros((B, M * bs, H, D), dtype=eng._k_gpu.dtype, device=dev)
            fv = torch.zeros_like(fk)
            fb = torch.zeros((B, M * bs, H), dtype=eng._kv_bias_gpu.dtype, device=dev)
            fx = torch.where(found, ids, self.minus1)
            gk, gv = self.k_orig
            gk(fk, eng._k_cpu, fx, bs)
            gv(fv, eng._v_cpu, fb, twin.total_cis, fx, bs)
            bi = torch.arange(B, device=dev).view(1, B, 1).expand(H, B, M)
            hi = torch.arange(H, device=dev).view(H, 1, 1).expand(H, B, M)
            mi = torch.arange(M, device=dev).view(1, 1, M).expand(H, B, M)
            bad = torch.zeros((H, B, M), dtype=torch.bool, device=dev)
            for win, fresh in ((eng._k_gpu, fk), (eng._v_gpu, fv)):
                a = win[:, :M * bs].reshape(B, M, bs, H, D)[bi, mi, :, hi, :]
                f = fresh.reshape(B, M, bs, H, D)[bi, mi, :, hi, :]
                bad |= (STF._bits(a) != STF._bits(f)).flatten(3).any(-1)
            a = eng._kv_bias_gpu[:, :M * bs].reshape(B, M, bs, H)[bi, mi, :, hi]
            f = fb.reshape(B, M, bs, H)[bi, mi, :, hi]
            bad |= (STF._bits(a) != STF._bits(f)).any(-1)
            self.kchk["bytes_bad"] += (bad & found).sum()
            self.kchk["win_checked"] += found.sum()

        def _keep_fill(self, l, vs, eng, twin, lst, loc, kern):
            """Module docstring 2: the keep step, the fill of the misses only (the shipped gathers), the counters and the
            fork's page ids of the listed blocks. Returns (page ids (H, B, R, T) with the keep slots, valid)."""
            B, bs = self.B, self.bs
            g = self.kpass
            t = g * self.L + l
            chk = self.kchk if self.checking else None
            m = keep_assign(self.kid, self.kage, lst, l, t, self.L, kneg=self.kneg, block=self.kblock, chk=chk,
                            stats=self.kacct, big=VR.FORK_BIG)
            hx = vs.host_x
            hx.fill_(-1)                                                   # slot 0 (CB1) and every hit: no load
            if self.kneg != "skip_miss":                                   # NEGATIVE CONTROL: a miss not fetched
                hx.scatter_(-1, m["widx"], torch.where(m["miss"], lst, vs.minus1))
            kern["gather_k"](vs.ext_k, eng._k_cpu, hx, bs)
            kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, hx, bs)
            row = self._krow(g)
            row[0:2].add_(torch.stack((m["valid"], m["miss"])).sum((1, 2, 3)))
            if self.kacct:
                row[3].add_(m["victims_alive"])
            if chk is not None and self.kcheck:
                self._keep_bytes(vs, eng, twin, lst, m["valid"], m["slot"])
            if self._ksb1 is None:                                         # host: once per state
                self._ksb1 = (vs.store_b.view(1, B, 1, 1), vs.store_b.view(1, B, 1, 1) + 1)
            sb, sb1 = self._ksb1
            ph = loc["phys"]
            at = (ph - sb1).clamp_(0, self.kcap - 1)                       # the list rank of a listed block
            sq = m["slot"].gather(-1, at.view(at.shape[0], B, -1)).view_as(at)
            return torch.where(loc["in_lst"], sb + sq, ph), m["valid"]

        def _acct(self, l, lst, valid):
            """Module docstring 1: the repeats of this (pass, layer) against the previous pass's list of the layer."""
            if valid is None:
                valid = lst < VR.FORK_BIG
                rep = acct_repeat(self.kprev[l], lst, valid)
                self._krow(self.kpass)[0:3:2].add_(torch.stack((valid, rep)).sum((1, 2, 3)))
            else:
                rep = acct_repeat(self.kprev[l], lst, valid)
                self._krow(self.kpass)[2].add_(rep.sum())

        def _keep_bytes(self, vs, eng, twin, lst, valid, slot):
            """CHECK-ONLY (kcheck): a FRESH fetch of this layer's list into a scratch store (the ORIGINAL gathers, not
            metered) and a bitwise compare of every listed block's 64 K, V and bias rows with its keep slot."""
            B, H, D, bs, X, cap = self.B, self.H, self.D, self.bs, self.kX, self.kcap
            if self._kfresh is None:
                self._kfresh = (torch.empty_like(vs.ext_k), torch.empty_like(vs.ext_v), torch.empty_like(vs.ext_bias),
                                torch.full_like(vs.host_x, -1))
            fk, fv, fb, fx = self._kfresh
            gk, gv = self.k_orig
            fx.fill_(-1)
            fx[..., 1:].copy_(torch.where(valid, lst, vs.minus1))
            gk(fk, eng._k_cpu, fx, bs)
            gv(fv, eng._v_cpu, fb, twin.total_cis, fx, bs)
            dev = self.dev
            bi = torch.arange(B, device=dev).view(1, B, 1).expand(H, B, cap)
            hi = torch.arange(H, device=dev).view(H, 1, 1).expand(H, B, cap)
            j1 = torch.arange(1, X, device=dev).view(1, 1, cap).expand(H, B, cap)
            bad = torch.zeros((H, B, cap), dtype=torch.bool, device=dev)
            for kept, fresh in ((vs.ext_k, fk), (vs.ext_v, fv)):
                a = kept.reshape(B, X, bs, H, D)[bi, slot, :, hi, :]
                f = fresh.reshape(B, X, bs, H, D)[bi, j1, :, hi, :]
                bad |= (STF._bits(a) != STF._bits(f)).flatten(3).any(-1)
            a = vs.ext_bias.reshape(B, X, bs, H)[bi, slot, :, hi]
            f = fb.reshape(B, X, bs, H)[bi, j1, :, hi]
            bad |= (STF._bits(a) != STF._bits(f)).any(-1)
            self.kchk["bytes_bad"] += (bad & valid).sum()
            self.kchk["bytes_checked"] += valid.sum()

        def keep_read(self) -> dict:
            """The host read (after the run): per joint pass (listed, fetched, repeat, victims alive), the totals and
            the check counters."""
            n = self.kpass + 1
            rows = self.kc[:max(n, 0)].tolist()
            listed = [r[0] for r in rows]
            fetched = [r[1] for r in rows] if self.kkeep else list(listed)
            out = dict(keep=self.kkeep, acct=self.kacct, kwin=self.kwin, kneg=self.kneg, passes=n, X=self.kX,
                       slots=(None if self.kblock is None else int((~self.kblock).sum())),
                       listed=listed, fetched=fetched, listed_total=sum(listed), fetched_total=sum(fetched),
                       hits_total=sum(listed) - sum(fetched))
            if self.kacct:
                rep = [r[2] for r in rows]
                wl, wp = [r[4] for r in rows], [r[5] for r in rows]
                out.update(repeat=rep, repeat_total=sum(rep), win_loads=wl, win_loads_total=sum(wl), win_prev=wp,
                           win_prev_total=sum(wp))
                if self.kkeep:
                    va = [r[3] for r in rows]
                    out.update(victims_alive=va, victims_alive_total=sum(va))
            if self.kwin:
                wh = [r[6] for r in rows]
                out.update(win_hits=wh, win_hits_total=sum(wh))
            out["check"] = {k: int(v) for k, v in self.kchk.items()}
            return out

    class TentRunnerFK(TentRunnerFD):
        """L4PTFD (TentRunnerFD) with the kept store (keep, arm L4PTFK) and / or the accounting (acct). kneg: a
        control (KNEG), kslots: fewer map slots (tests only); kcheck: the fresh-fetch byte check (check rounds)."""

        def __init__(self, *args, keep: bool = True, acct: bool = False, kneg=None, kslots=None, kcheck: bool = False,
                     kwin: bool = False, **kw):
            if kneg not in KNEG:                                           # before the meter attaches
                raise SpecError("unknown keep control %r (one of %s)" % (kneg, KNEG))
            if not keep and (kneg is not None or kslots is not None or kcheck or kwin):
                raise SpecError("kneg / kslots / kcheck / kwin belong to the keep (keep=True)")
            self._k = dict(keep=bool(keep), acct=bool(acct), kneg=kneg, kslots=kslots, kcheck=bool(kcheck),
                           kwin=bool(kwin))
            super().__init__(*args, **kw)
            self.T.k_orig = self.meter.orig                                # the unmetered gathers (kcheck)
            self.meta.update(store=("keep" if keep else "scratch"), store_acct=bool(acct),
                             keep=dict(kneg=kneg, kslots=kslots, kcheck=bool(kcheck), kwin=bool(kwin),
                                       extra_ops=KEEP_EXTRA_OPS, extra_ops_win=KEEP_WIN_EXTRA_OPS))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFK(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg, **self._k)

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(keep=self.T.keep_read())
            return out

    return TentStateFK, TentRunnerFK
