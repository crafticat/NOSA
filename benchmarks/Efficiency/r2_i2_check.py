"""Rung I2 (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): WS-G's CHECK PASS for the
runners of THE SCHEDULE OF CONTRACT 4 (spec_tentative.TentRunner = L4PT, spec_tentative_union.TentRunnerU = L4PTU,
spec_tentative.PaperJournalRunner = L4PJ, the EXACT positive control). It runs only in the check cell of r2_i2_cell.py
(cell i2g), never in a timing cell. No kernel, no engine edit.

WHY A NEW CAPTURE. r2_wsg_check.py (WS-G, b1ae579, copied into this tree UNCHANGED) captures through
verify_rows.joint_pass and maps the rows of a joint pass to committed columns by "rows 0 .. m - 1 of the pass, m = the
count increase". In contract 4 that map is wrong: a round also commits tokens in its D slots (target rows of the drain,
outside any joint pass), and TentRunner / TentRunnerU never call verify_rows.joint_pass (TentState runs its own rows;
docs/NOTES_i2.md section 2). ContractCapture (a subclass of r2_wsg_check.FastCapture: the same stores, the same result
keys, so r2_wsg_check.run_ref, chk_parts and chk_gates apply unchanged) maps rows BY POSITION:
  a row at position p predicts committed column c = p - S. After each round (the after_round hook), every row of the
  round whose column lies in [count at the round start, count after it) is a committing row; its selection (every layer,
  every KV head) and, for the sample rows, its logits row are stored at that column. Rows are taken in the order they
  ran, so a later row at the same column (a drain's D-slot target row after a J slot's resident row) overwrites an
  earlier one. TentRunner / TentRunnerU also pass each row's verify mask (the plan): only verify (or target) rows count.
  The warm step (column 0) and the initial fill (round -1) are not committing rows; the round boundary is the runner's
  _set_check(g >= 0) call at the start of each round.
WHERE THE ROWS COME FROM (wrappers, put back by remove()):
  joint rows    verify_rows._select_rows (per layer, (H, B, R, T)) inside a joint pass; the positions from
                TentState.run_pass's plan (tentative) or verify_rows.joint_pass's position_ids (journal)
  one-row rows  ragged_engine._select (per layer, rs.st.ti (H, B, T)); the positions rs.pos_before (set before the
                twin step by the ragged step and by TentState's one-row pass)
  logits        TentState.run_pass's return (every pass) or verify_rows.joint_pass's return (journal)
THE PARTS (r2_wsg_check.chk_parts): MAP, REF, SEL, GREEDY (tie tolerance from the cell), POS, SLOT, and MISS (FAST) or
BITS (EXACT). Adapted inputs:
  MISS  (tentative arms) every committed block (id <= qb = qc // 64) of a VERIFY row's selection is in the window after
        that row's residency (TentState._resid, wrapped: every one-row pass and row 0 of a union pass), plus (L4PTU) the
        union pass's presence recount verify_rows stat "dropped" (rows 1 .. R - 1; VerifyOps(check=True) in this cell)
  POS   tentative arms: after every round the committed position T.qc = S + count and the committed twin counter
        T.ctr_c[0] = T.qc (the twins themselves stand at the chain's frontier between rounds, by design); the journal arm:
        every layer's twin position = S + count (as r2_wsg_check)
  SLOT  r2_wsg_check.slot_errors on the sample rows (every window / pool slot that a map names holds its block's host
        rows); no extras-content part (the union store is scratch, refilled per layer)
CHECK-ONLY OPERATIONS: CHECK_ONLY_OPS below (and r2_wsg_check.CHECK_ONLY_OPS for the reference run).
"""
from __future__ import annotations

import r2_wsg_check as CK

CHECK_ONLY_OPS = (
    "the wrappers of verify_rows._select_rows / joint_pass, ragged_engine._select, TentState.run_pass / _resid and the "
    "runner's _set_check: clones of every row's selection and position per pass (device), the sample rows' logits",
    "the after_round hook: the column map of the round's rows, the host copies of the committing rows' selections, the "
    "committed-position comparison (every round), the slot-content comparison (sample rows, every --chk-slot-every "
    "rounds)",
    "the window-membership comparison after each verify row's residency (MISS)",
)


class ContractCapture(CK.FastCapture):
    """The capture of ONE contract-4 arm (module docstring). install() before runner.run, remove() after it (finally),
    then finish(). RE: the ragged_engine module the runner uses. tentative: TentRunner / TentRunnerU (else
    PaperJournalRunner)."""

    def __init__(self, runner, steps, S, sample, exact, RE, tentative, slot_every=32, keep_logits=True):
        super().__init__(runner, steps, S, sample, exact, slot_every=slot_every, keep_logits=keep_logits)
        import torch
        self.RE, self.tentative = RE, bool(tentative)
        self.rs = runner.rs
        self.cur = None                                   # the pass being captured: dict(pos, verify, sel [layers])
        self.round_rows = []                              # this round's passes, in run order
        self.count0 = None                                # the committed count at the round start
        self.in_joint = False
        self.miss_res = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.chk_res = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.drop0 = None

    # -------------------------------------------------------------- the pass records
    def _open(self, pos, verify):
        self.cur = dict(pos=pos.detach().to(torch_int64()).clone(), verify=(verify.clone() if verify is not None
                                                                              else None), sel=[], lg=None)

    def _close(self, lg=None):
        c = self.cur
        self.cur = None
        if c is None:
            return
        if lg is not None and self.keep_logits:
            c["lg"] = lg[self.sidx].float()
        if len(c["sel"]) != self.L:
            self.map_errors.append("pass with %d selections for %d layers" % (len(c["sel"]), self.L))
            return
        self.round_rows.append(c)
        self.passes += 1

    # -------------------------------------------------------------- the wrappers
    def _sr(self, vs, layer, q, twin):                    # verify_rows._select_rows (joint passes)
        sel = self._orig["_select_rows"](vs, layer, q, twin)
        if self.cur is not None and self.in_joint:
            self.cur["sel"].append(sel.to(dtype=self.sel_store.dtype))
        return sel

    def _rsel(self, rs_, l, layer, score, twin):         # ragged_engine._select (one-row passes)
        self._orig["_select"](rs_, l, layer, score, twin)
        if rs_ is not self.rs or self.in_joint:
            return
        if l == 0:                                        # a new one-row pass (an unclosed one, e.g. the warm step, is
            self._open(rs_.pos_before.view(self.B, 1), self._next_verify)    # dropped: it commits no round column)
            self.cur["one"] = True
        if self.cur is not None and self.cur.get("one"):
            self.cur["sel"].append(rs_.st.ti.view(self.H, self.B, 1, -1).to(dtype=self.sel_store.dtype))
            if len(self.cur["sel"]) == self.L and not self.tentative:
                self._close(None)                        # journal one-row steps: no logits captured

    def _jp(self, model, vs, input_ids, position_ids):   # verify_rows.joint_pass (the journal runner)
        self._open(position_ids, None)
        self.in_joint = True
        try:
            lg = self._orig["joint_pass"](model, vs, input_ids, position_ids)
        finally:
            self.in_joint = False
        self._close(lg)
        return lg

    def _run_pass(self, model, plan):                     # TentState.run_pass (the tentative runners)
        if plan.R > 1:
            self._open(plan.pos, plan.verify)
            self.in_joint = True
            try:
                lg = self._orig["run_pass"](model, plan)
            finally:
                self.in_joint = False
            self._close(lg)
            return lg
        self._next_verify = plan.verify
        self.cur = None
        try:
            lg = self._orig["run_pass"](model, plan)
        finally:
            self._next_verify = None
        if self.cur is not None and self.cur.get("one"):
            self._close(lg)
        return lg

    def _resid(self, l, eng, twin, ti, inj, act, drf, qb3):   # TentState._resid (MISS of the tentative arms)
        skip = self._orig["_resid"](l, eng, twin, ti, inj, act, drf, qb3)
        ver = (act & ~drf).view(1, -1, 1)
        com = (ti >= 0) & (ti <= qb3) & ver
        mem = (ti.unsqueeze(-1) == eng._block_map.unsqueeze(-2)).any(-1)
        miss = com & ~mem
        self.miss_res += miss.sum()
        self.chk_res += com.sum()
        if 0 <= l < self.L:
            self.miss_layer[l] += miss.sum()
        self.miss_req += miss.sum(dim=(0, 2))
        return skip

    def _set_check(self, g, probe):
        self._orig["_set_check"](g, probe)
        if g >= 0:                                        # a round starts (or the initial fill: cleared at round 0)
            self.round_rows = []
            self.count0 = self.runner.count.clone()

    # -------------------------------------------------------------- the round's column map
    def _after(self, runner):
        import torch
        if self.prev_after is not None:
            self.prev_after(runner)
        c0 = (self.count0 if self.count0 is not None else runner.count).cpu()
        c1 = runner.count.detach().cpu()
        for p in self.round_rows:
            pos = p["pos"].cpu()                          # (B, R)
            R = int(pos.shape[1])
            col = pos - self.S0
            sel = torch.stack(p["sel"]).cpu()             # (L, H, B, R, T)
            ver = p["verify"].cpu() if p["verify"] is not None else None   # (R, B)
            for r in range(R):
                mask = (col[:, r] >= c0) & (col[:, r] < c1) & (col[:, r] < self.cap) & (col[:, r] >= 1)
                if ver is not None:
                    mask &= ver[r]
                if not bool(mask.any()):
                    continue
                rows = torch.nonzero(mask).view(-1)
                cols = col[rows, r]
                self.sel_store[rows, cols] = sel[:, :, rows, r].permute(2, 0, 1, 3)
                self.has_sel[rows, cols] = True
                lgc = p["lg"].cpu() if p["lg"] is not None else None
                for i, b in enumerate(self.sample):
                    if bool(mask[b]):                     # the LAST row of the column decides: a later row without
                        key = (b, int(col[b, r]))         # captured logits (a journal D-slot step) drops an earlier
                        if lgc is None:                   # row's logits (a J-slot resident row of a target request)
                            self.lg_store.pop(key, None)
                        else:
                            self.lg_store[key] = lgc[i, r].clone()
        self.round_rows = []
        S = self.S0
        if self.tentative:                                # the committed position (the twins stand at the frontier)
            T = runner.T
            want = S + runner.count
            self.pos_bad += (T.qc != want).sum() + (T.ctr_c[0] != T.qc).sum()
        else:
            want = S + runner.count
            for t in runner.rs.st.twins:
                self.pos_bad += (t.pos != want).sum()
        self.rounds += 1
        if self.slot_every > 0 and self.rounds % self.slot_every == 0:
            self._slots(runner.rs)

    def _finalize(self, count_now):
        return None                                       # the map is per round (_after)

    # -------------------------------------------------------------- install / remove / finish
    def _wrap(self, obj, name, fn):
        """obj.<name> = fn; remove() puts back the instance attribute that was there (a test hook) or deletes it."""
        self._inst.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        self._orig[name] = getattr(obj, name)
        setattr(obj, name, fn)

    def install(self):
        VR, RE, rn = self.VR, self.RE, self.runner
        self._next_verify = None
        self._inst = []
        self._orig = dict(_select_rows=VR._select_rows, joint_pass=VR.joint_pass, _select=RE._select)
        VR._select_rows = self._sr
        RE._select = self._rsel
        self._wrap(rn, "_set_check", self._set_check)
        if self.tentative:
            T = rn.T
            self._wrap(T, "run_pass", self._run_pass)
            self._wrap(T, "_resid", self._resid)
            vs = rn.vs
            self.drop0 = int(vs.stat["dropped"]) if (vs is not None and "dropped" in vs.stat) else None
        else:
            VR.joint_pass = self._jp
        self.prev_after = rn.ops.after_round
        rn.ops.after_round = self._after

    def remove(self):
        if self._orig is None:
            return
        rn = self.runner
        self.VR._select_rows = self._orig["_select_rows"]
        self.VR.joint_pass = self._orig["joint_pass"]
        self.RE._select = self._orig["_select"]
        for obj, name, had, val in reversed(self._inst):
            if had:
                setattr(obj, name, val)
            else:
                delattr(obj, name)                        # the class method again
        self._inst = []
        rn.ops.after_round = self.prev_after

    def finish(self, steps=None):
        rn = self.runner
        if self.tentative:
            vs = rn.vs
            drop = 0
            if self.drop0 is not None:
                drop = int(vs.stat["dropped"]) - self.drop0
            sel_n = int(vs.stat_row["selected"].sum()) if hasattr(vs, "stat_row") else 0
            self.miss += self.miss_res + drop
            self.miss_checked += self.chk_res + sel_n
        RE = self.RE
        out = super().finish(steps)
        self.RE = RE
        return out


def torch_int64():
    import torch
    return torch.int64
