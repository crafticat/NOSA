"""WS-G CHECK PASS (KS/analysis/plan/PLAN_L4P_faithful.md, WS-G; the human's correction of 2026-10-28): the FAST
computation of a speculative arm against SH's computation ON THE SAME COMMITTED PREFIX. It runs only in the check cell
r2_wsg_cell.py, never in a timing cell. No kernel. No engine edit: the capture wraps three functions of the verify_rows
module (joint_pass, _select_rows, _extras) and the round runner's after_round hook while the arm runs, and puts every
one back after the run (try / finally). It works on the GPU and on the CPU stand-in world (tests_cpu/test_wsg.py)
because it takes the modules from the runner (runner.VR) and from the caller (RE).

THE REFERENCE (REF). SH's computation teacher-forced on the arm's OWN committed tokens: ragged_engine.decode_step in
lockstep, every request active, at the cell's C and B, with the configuration of the rg73 cell (gemm natural, twin
fused, body fused, graph on). MEASURED: that step is bitwise SH at C 73 B 318 (SCORING_rg73_rb73.md: RG-BIT, 0 of
162,816 logits fingerprints unequal). REF step c feeds the arm's token of column c - 1 (column 0: SH's first token) at
position S + c, so its logits and its selection are SH's computation on the arm's prefix of column c. MUST <arm>-CHK-REF
checks it again in each cell: on every column whose prefix equals SH's (up to the arm's first divergence from SH) REF's
logits fingerprints equal SH's, bitwise, and REF's error flags are 0.

THE PARTS of <arm>-CHK (MUST; the planted control's is a REPORT, see PLANT):
  MAP     the capture maps each joint-pass row to the right committed column: every column 1 .. count - 1 has a stored
          selection; every stored logits row has the committed token as its argmax and the committed fingerprint's max,
          bitwise (runner.out / runner.fmax); 0 <= rows committed per pass <= R
  REF     above
  SEL     every committed joint-pass row (every request, every layer, every KV head): its selected block SET equals
          REF's selected block set at that column. A changed selection is a semantic bug, not rounding.
  MISS    (FAST) every block that a VERIFY row selects is present for it: in the window after the row-0 residency, in a
          kept extras slot, the tail block or (past a block edge) the next block. Every verify row of every joint pass,
          every request (committed or not), every layer and KV head. The draft row (resident-only by design) is not a
          verify row.
  GREEDY  every committed token of every request is REF's argmax at that column (REF's logit of the committed token
          equals REF's maximum), except where REF's gap (its maximum minus its logit of the committed token) is at most
          the REGISTERED tie tolerance (--chk-tie-logit, default 0: no exemption)
  POS     after every round, every request's twin position in every layer equals S + its committed count (the rollback
          left the committed length)
  SLOT    every window and pool slot that a map names holds the host rows of its block (K, V) and its bias rows, bitwise
          (the residency invariant of tests_cpu/spec_world.residency_errors), on the sample rows, after every
          --chk-slot-every rounds and after the run; and (FAST) every kept extras slot holds the host rows of its block
          on the joint passes of those rounds
  BITS    (EXACT arms only, the positive control of the check) max |dlogit| = 0 on every compared position
REPORT <arm>-CHK-LOGIT: max |dlogit| and KL(REF || arm) per committed position of the sample rows (quantiles), and the
argmax agreement. EMPIRICAL AGREEMENT, scope: the sample rows, the committed positions, this cell; not a proof of an
identical output distribution.

PLANT (--acc-plant ARM): <arm>-CHK becomes a REPORT; MUST <arm>-PLANT-CHK-PRE (MISS > 0: the plant drops verify blocks
on these requests, L34) and MUST <arm>-PLANT-CHK-CAUGHT (<arm>-CHK FAILS).

CHECK-ONLY OPERATIONS (they never run in a timing cell): CHECK_ONLY_OPS below.
"""
from __future__ import annotations

import math

CHECK_ONLY_OPS = (
    "the wrappers of verify_rows.joint_pass / _select_rows / _extras: clones of the selections (every row), of the "
    "window map and the kept extras ids, of the sample rows' logits; the missing-block comparison; the extras-content "
    "comparison; one host read of the committed counts per joint pass",
    "the after_round hook: the twin-position comparison (every round) and the slot-content comparison (sample rows, "
    "every --chk-slot-every rounds), with host copies",
    "the teacher-forced reference run (its own prefill, ragged_engine.attach, steps) and every comparison against it",
    "the host stores of the captured selections and logits rows, and the summary of the comparisons",
)
REQUIRED_OPS = (
    "the residency of every row (diff, victim pool, map commit, gathers), the extras store and its gathers, the joint "
    "pass, accept / truncate / commit and the rollback journal, the write-backs, the rollback's release waits, the "
    "one host sync per round (the termination and the position mirror), the prefetch's events and stream waits (rung "
    "L5): a timing cell keeps all of them (they produce the tokens or keep a buffer alive until its reader is done)",
)
ACCOUNTING_IN_TIMED_PATH = (
    "verify_rows' device statistics (union_sum, union_max, extras, overflow, sel0; VerifyOps(stats=True), the default) "
    "and spec_round's KVMeter counts run inside the timed FAST path today: they are accounting, not required for the "
    "tokens. The DROP part of <arm>-ACC reads the overflow counter. Their cost is not measured (REPORT, open)."
)


def default_rows(B, n=8):
    """n sample rows spread over the batch: (i x B) // n."""
    return sorted({(i * int(B)) // int(n) for i in range(int(n))}) if int(B) >= int(n) else list(range(int(B)))


def _bits(t):
    import torch
    return t.contiguous().view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _same(a, b):
    """Bitwise equality of two tensors of one dtype (else value equality after a cast to float64)."""
    import torch
    if a.dtype == b.dtype:
        return torch.equal(_bits(a), _bits(b))
    return torch.equal(a.double(), b.double())


def slot_errors(rs, rows):
    """Every window slot s < topk - 1 and every pool slot that a map names holds the host rows of its block and the
    bias rows of twin.total_cis, bitwise (spec_world.residency_errors), on the given requests. dict(bad, checked,
    first)."""
    import torch
    bad = checked = 0
    first = None
    for l, lay in enumerate(rs.cache.layers):
        e, tw = lay.cache_engine, rs.st.twins[l]
        bs, topk = int(e.block_size), int(e.topk)
        P = int(e._pool_map.shape[2]) if rs.pool else 0
        ar = torch.arange(bs, dtype=torch.int64)
        for b in rows:
            ids = e._block_map[:, b, :topk - 1]
            slots = list(range(topk - 1))
            if P:
                ids = torch.cat([ids, e._pool_map[:, b]], dim=-1)
                slots += [topk + q for q in range(P)]
            ids = ids.cpu()
            sl = torch.tensor(slots, dtype=torch.int64)
            kg, vg, bg = e._k_gpu[b].cpu(), e._v_gpu[b].cpu(), e._kv_bias_gpu[b].cpu()
            kc, vc, tc = e._k_cpu[b], e._v_cpu[b], tw.total_cis[b].cpu()
            for h in range(ids.shape[0]):
                ok = ids[h] >= 0
                n = int(ok.sum())
                if n == 0:
                    continue
                xv, sv = ids[h][ok], sl[ok]
                hr = (xv.view(-1, 1) * bs + ar.view(1, -1)).reshape(-1)
                gr = (sv.view(-1, 1) * bs + ar.view(1, -1)).reshape(-1)
                for i in range(n):
                    g, c = gr[i * bs:(i + 1) * bs], hr[i * bs:(i + 1) * bs]
                    good = _same(kg[g, h], kc[c, h]) and _same(vg[g, h], vc[c, h]) and _same(bg[g, h], tc[c, h])
                    checked += 1
                    if not good:
                        bad += 1
                        if first is None:
                            first = (l, int(b), h, int(xv[i]), int(sv[i]))
    return dict(bad=bad, checked=checked, first=first)


class FastCapture:
    """The capture of ONE speculative arm (module docstring). Build it after the SpecRunner, install() before
    runner.run, remove() after it (finally), then finish(). It keeps CPU stores only after finish()."""

    def __init__(self, runner, steps, S, sample, exact, slot_every=32, keep_logits=True):
        import torch
        self.runner, self.VR = runner, runner.VR
        rs = runner.rs
        self.B, self.K = int(runner.B), int(runner.K)
        self.L = len(rs.st.twins)
        eng0 = rs.cache.layers[0].cache_engine
        self.H, self.topk = int(eng0.head_num), int(eng0.topk)
        self.steps, self.S0 = int(steps), int(S)
        self.cap = self.steps + self.K
        self.sample = [int(b) for b in sample]
        self.exact = bool(exact)
        self.slot_every = int(slot_every)
        self.keep_logits = bool(keep_logits)
        self.dev = rs.st.cu_q.device
        self.sidx = torch.tensor(self.sample, dtype=torch.int64, device=self.dev)
        self.sel_store = torch.full((self.B, self.cap, self.L, self.H, self.topk), -1, dtype=torch.int16)
        self.has_sel = torch.zeros((self.B, self.cap), dtype=torch.bool)
        self.lg_store = {}
        self.pending = None
        self.layer_sel = []
        self.layer_i = -1
        self.passes = 0
        self.rounds = 0
        z = lambda: torch.zeros((), dtype=torch.int64, device=self.dev)    # noqa: E731
        self.miss, self.miss_checked, self.pos_bad = z(), z(), z()
        self.miss_req = torch.zeros(self.B, dtype=torch.int64, device=self.dev)
        self.miss_layer = torch.zeros(self.L, dtype=torch.int64, device=self.dev)
        self.ext_bad = self.ext_checked = 0
        self.ext_first = None
        self.slot = dict(bad=0, checked=0, first=None, checks=0)
        self.map_errors = []
        self._orig = None
        self.prev_after = None
        self.result = None

    # -------------------------------------------------------------- the wrappers
    def _jp(self, model, vs, input_ids, position_ids):
        import torch
        self._finalize(self.runner.count)            # the previous pass: count is after its round's commit now
        c0 = self.runner.count.clone()
        self.layer_sel, self.layer_i = [], -1
        lg = self._orig["joint_pass"](model, vs, input_ids, position_ids)
        if len(self.layer_sel) != self.L:
            self.map_errors.append("pass %d: %d selections for %d layers" % (self.passes, len(self.layer_sel), self.L))
        sel = torch.stack(self.layer_sel) if self.layer_sel else None              # (L, H, B, R, topk) int16
        self.pending = dict(c0=c0.cpu(), sel=(sel.cpu() if sel is not None else None),
                            lg=(lg[self.sidx].float().cpu() if self.keep_logits else None), R=int(lg.shape[1]))
        self.passes += 1
        return lg

    def _sr(self, vs, layer, q, twin):
        sel = self._orig["_select_rows"](vs, layer, q, twin)
        self.layer_sel.append(sel.to(dtype=self.sel_store.dtype))                # a copy (vs.ti is reused per layer)
        self.layer_i += 1
        return sel

    def _ex(self, vs, eng, twin, sel, m, crossing):
        import torch
        win = eng._block_map.clone()                                             # after the row-0 residency
        ids = self._orig["_extras"](vs, eng, twin, sel, m, crossing)
        H, B = int(win.shape[0]), int(win.shape[1])
        X = int(vs.X)
        tail = m.view(1, B, 1).expand(H, B, 1)
        att = torch.cat([win, ids[..., :X], tail], dim=-1)                       # (H, B, A); ids[..., X - 1] = m + 1
        nver = vs.R - 1 if vs.ops.draft_last else vs.R                           # the draft row is not a verify row
        sv = sel[:, :, :nver]                                                    # (H, B, nver, topk)
        mem = (sv.unsqueeze(-1) == att.view(H, B, 1, 1, -1)).any(-1)
        miss = (~mem) & (sv >= 0)
        self.miss += miss.sum()
        self.miss_checked += (sv >= 0).sum()
        self.miss_req += miss.sum(dim=(0, 2, 3))
        if 0 <= self.layer_i < self.L:
            self.miss_layer[self.layer_i] += miss.sum()
        if self.slot_every > 0 and self.passes % self.slot_every == 0:
            self._extras_contents(vs, eng, twin, ids, X)
        return ids

    def _extras_contents(self, vs, eng, twin, ids, X):
        """The kept extras slots of the sample rows hold the host rows of their blocks and the bias of total_cis."""
        import torch
        bs = int(vs.bs)
        ar = torch.arange(bs, dtype=torch.int64)
        idc = ids[..., :X - 1].cpu()
        for b in self.sample:
            ek, ev, eb = vs.ext_k[b].cpu(), vs.ext_v[b].cpu(), vs.ext_bias[b].cpu()
            kc, vc, tc = eng._k_cpu[b], eng._v_cpu[b], twin.total_cis[b].cpu()
            for h in range(idc.shape[0]):
                for x, k in enumerate(idc[h, b].tolist()):
                    if k < 0:
                        continue
                    g = x * bs + ar
                    c = k * bs + ar
                    good = _same(ek[g, h], kc[c, h]) and _same(ev[g, h], vc[c, h]) and _same(eb[g, h], tc[c, h])
                    self.ext_checked += 1
                    if not good:
                        self.ext_bad += 1
                        if self.ext_first is None:
                            self.ext_first = (self.passes, self.layer_i, int(b), h, int(k))

    def _after(self, runner):
        if self.prev_after is not None:
            self.prev_after(runner)
        want = self.S0 + runner.count
        for t in runner.rs.st.twins:
            self.pos_bad += (t.pos != want).sum()
        self.rounds += 1
        if self.slot_every > 0 and self.rounds % self.slot_every == 0:
            self._slots(runner.rs)

    def _slots(self, rs):
        s = slot_errors(rs, self.sample)
        self.slot["bad"] += s["bad"]
        self.slot["checked"] += s["checked"]
        self.slot["checks"] += 1
        if self.slot["first"] is None and s["first"] is not None:
            self.slot["first"] = (self.rounds,) + tuple(s["first"])

    def _finalize(self, count_now):
        """The rows of the pending pass that the round committed: rows 0 .. m - 1, m = count after - count before."""
        import torch
        p = self.pending
        if p is None:
            return
        self.pending = None
        c0 = p["c0"]
        m = count_now.detach().cpu() - c0
        R = p["R"]
        if bool((m < 0).any()) or bool((m > R).any()):
            self.map_errors.append("pass %d: committed rows per request outside 0 .. %d: %s" % (
                self.passes - 1, R, sorted(set(m.tolist()))[:5]))
        sel = p["sel"]
        for r in range(R):
            mask = (m > r) & (c0 + r < self.cap)
            if not bool(mask.any()) or sel is None:
                continue
            rows = torch.nonzero(mask).view(-1)
            cols = (c0 + r)[rows]
            self.sel_store[rows, cols] = sel[:, :, rows, r].permute(2, 0, 1, 3)
            self.has_sel[rows, cols] = True
        if p["lg"] is not None:
            for i, b in enumerate(self.sample):
                for r in range(max(0, min(int(m[b]), R))):
                    c = int(c0[b]) + r
                    if c < self.cap:
                        self.lg_store[(b, c)] = p["lg"][i, r].clone()

    # -------------------------------------------------------------- install / remove / finish
    def install(self):
        VR = self.VR
        self._orig = dict(joint_pass=VR.joint_pass, _select_rows=VR._select_rows, _extras=VR._extras)
        VR.joint_pass, VR._select_rows = self._jp, self._sr
        if not self.exact:
            VR._extras = self._ex
        self.prev_after = self.runner.ops.after_round
        self.runner.ops.after_round = self._after

    def remove(self):
        if self._orig is not None:
            for k, f in self._orig.items():
                setattr(self.VR, k, f)
        self.runner.ops.after_round = self.prev_after

    def finish(self, steps=None):
        """After the run: the last pending pass, a final slot check, MAP. Drops every reference to the runner."""
        import torch
        rn = self.runner
        self._finalize(rn.count)
        self._slots(rn.rs)
        cnt = [int(c) for c in rn.count.cpu().tolist()]
        out, fmax = rn.out.detach().cpu(), rn.fmax.detach().cpu()
        n = [min(c, self.steps) for c in cnt]
        col = torch.arange(self.cap, dtype=torch.int64).view(1, -1)
        need = (col >= 1) & (col < torch.tensor(cnt, dtype=torch.int64).clamp(max=self.cap).view(-1, 1))
        gaps = (need & ~self.has_sel).nonzero()                   # column 0 is the warm step (no joint pass)
        if gaps.numel():
            self.map_errors.append("%d committed columns have no stored selection (first %s)" % (
                int(gaps.shape[0]), gaps[:3].tolist()))
        lgbad = []
        for (b, c), row in self.lg_store.items():
            if c >= cnt[b]:
                continue
            if int(row.argmax()) != int(out[b, c]) or not _same(row.max().view(1), fmax[b, c].view(1)):
                lgbad.append((b, c))
        if lgbad:
            self.map_errors.append("%d stored logits rows are not the committed rows (first %s)" % (len(lgbad),
                                                                                                  lgbad[:3]))
        self.result = dict(
            exact=self.exact, passes=self.passes, rounds=self.rounds, count=cnt, n=n, sample=self.sample,
            map_errors=self.map_errors, sel_columns=int(self.has_sel.sum()),
            miss=int(self.miss), miss_checked=int(self.miss_checked), miss_layer=self.miss_layer.cpu().tolist(),
            miss_requests=int((self.miss_req > 0).sum()), pos_bad=int(self.pos_bad), slot=dict(self.slot),
            ext=dict(bad=self.ext_bad, checked=self.ext_checked, first=self.ext_first),
            logits_rows=len(self.lg_store))
        self.runner = self.VR = None
        return self.result

    def ref_inputs(self, out, first, sh_inputs):
        """REF's input columns (B, steps + 1): column 0 = the first token, column c + 1 = the arm's committed token of
        column c; past the arm's committed count, SH's columns (any token: REF's later steps are not compared)."""
        x = sh_inputs[:, :self.steps + 1].detach().clone()
        o = out.detach()
        for b in range(self.B):
            k = self.result["n"][b]
            x[b, 0] = first[b]
            x[b, 1:k + 1] = o[b, :k].to(x.device)
        return x


def run_ref(model, rs, RE, inputs_ref, S, steps, cap, fp_sh=None, sh_inputs=None):
    """THE TEACHER-FORCED REFERENCE and every comparison with the capture `cap` (FastCapture after finish()). rs: a
    RaggedState attached after the shipped prefill (RE.attach). One step per column, every request active. Returns the
    REF record (CPU)."""
    import torch
    with torch.inference_mode():
        return _run_ref(model, rs, RE, inputs_ref, S, steps, cap, fp_sh, sh_inputs)


def _run_ref(model, rs, RE, inputs_ref, S, steps, cap, fp_sh, sh_inputs):
    import torch
    B, L = int(rs.B), len(rs.st.twins)
    dev = rs.st.cu_q.device
    H, T = int(rs.st.ti.shape[0]), int(rs.st.ti.shape[-1])
    ref_sel = torch.zeros((L, H, B, T), dtype=torch.int64, device=dev)
    orig = RE._select

    def sel_w(rs_, l, layer, score, twin):
        orig(rs_, l, layer, score, twin)
        if rs_ is rs:
            ref_sel[l].copy_(rs_.st.ti)
    pred = torch.full((B, steps), -1, dtype=torch.int64, device=dev)
    fmax = torch.zeros((B, steps), dtype=torch.float32, device=dev)
    fsum = torch.zeros((B, steps), dtype=torch.float64, device=dev)
    gap = torch.zeros((B, steps), dtype=torch.float32, device=dev)
    sel_bad = torch.zeros(B, dtype=torch.int64, device=dev)
    sel_checked = torch.zeros((), dtype=torch.int64, device=dev)
    sel_cols = torch.zeros((B, steps), dtype=torch.bool, device=dev)
    S_ = list(cap.sample)
    dl = torch.full((len(S_), steps), float("nan"), dtype=torch.float64, device=dev)
    kl = torch.full((len(S_), steps), float("nan"), dtype=torch.float64, device=dev)
    ids_buf = torch.empty((B, 1), dtype=torch.int64, device=dev)
    pos_buf = torch.empty((B, 1), dtype=torch.int32, device=dev)
    inp = inputs_ref.to(dev)
    RE.set_activity(rs, torch.ones(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool))
    RE._select = sel_w
    try:
        for t in range(int(steps)):
            ids_buf.copy_(inp[:, t:t + 1])
            pos_buf.fill_(int(S) + t)
            lg = RE.decode_step(model, rs, ids_buf, pos_buf)[:, -1, :]
            mx = lg.amax(dim=-1)
            pred[:, t] = lg.argmax(dim=-1)
            fmax[:, t] = mx
            fsum[:, t] = lg.sum(dim=-1, dtype=torch.float64)
            gap[:, t] = mx - lg.gather(1, inp[:, t + 1:t + 2]).view(B)
            has = cap.has_sel[:, t] if t < cap.cap else torch.zeros(B, dtype=torch.bool)
            if bool(has.any()):
                hd = has.to(dev)
                fs = cap.sel_store[:, t].to(device=dev, dtype=torch.int64)       # (B, L, H, T)
                rsel = ref_sel.permute(2, 0, 1, 3)
                eq = (fs.sort(dim=-1).values == rsel.sort(dim=-1).values).all(-1)    # (B, L, H)
                sel_bad += ((~eq) & hd.view(B, 1, 1)).sum(dim=(1, 2))
                sel_checked += hd.sum() * L * H
                sel_cols[:, t] = hd & (~eq).any(-1).any(-1)
            for i, b in enumerate(S_):
                row = cap.lg_store.get((b, t))
                if row is None:
                    continue
                f = row.to(device=dev, dtype=torch.float64)
                r = lg[b].double()
                dl[i, t] = (f - r).abs().max()
                p = torch.softmax(r, dim=-1)
                kl[i, t] = (p * (torch.log_softmax(r, dim=-1) - torch.log_softmax(f, dim=-1))).sum()
    finally:
        RE._select = orig
    err = RE.errors(rs)
    out = dict(pred=pred.cpu(), fmax=fmax.cpu(), fsum=fsum.cpu(), gap=gap.cpu(), sel_bad=sel_bad.cpu(),
               sel_checked=int(sel_checked), sel_cols=sel_cols.cpu(), dl=dl.cpu(), kl=kl.cpu(), errors=err, steps=int(steps))
    if fp_sh is not None and sh_inputs is not None:
        out["self"] = ref_self(out, fp_sh, inputs_ref, sh_inputs, cap.result["n"])
    return out


def ref_self(ref, fp_sh, inputs_ref, sh_inputs, n):
    """REF against SH on every column whose prefix equals SH's (columns 0 .. d_b, d_b = the arm's first column that
    differs from SH, else 0 .. n_b - 1): logits fingerprints bitwise. dict(ok, checked, unequal, first)."""
    import torch
    sm, ss = fp_sh[0].detach().cpu(), fp_sh[1].detach().cpu()
    a, s = inputs_ref.detach().cpu(), sh_inputs.detach().cpu()
    checked = unequal = 0
    first = None
    for b in range(a.shape[0]):
        k = min(int(n[b]), int(ref["steps"]), int(sm.shape[1]))
        if k <= 0:
            continue
        neq = (a[b, 1:k + 1] != s[b, 1:k + 1]).nonzero()
        last = int(neq[0, 0]) if neq.numel() else k - 1
        cols = slice(0, last + 1)
        e = ((_bits(ref["fmax"][b, cols]) == _bits(sm[b, cols])) & (_bits(ref["fsum"][b, cols]) == _bits(ss[b, cols])))
        checked += int(e.numel())
        u = int((~e).sum())
        unequal += u
        if u and first is None:
            first = (b, int((~e).nonzero()[0, 0]))
    return dict(ok=checked > 0 and unequal == 0, checked=checked, unequal=unequal, first=first)


def _quant(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return None
    pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]    # noqa: E731
    return dict(n=len(xs), p50=pick(0.5), p90=pick(0.9), p99=pick(0.99), max=xs[-1])


def chk_parts(arm, cap, ref, tie=0.0):
    """The parts of <arm>-CHK from the capture result and the REF record (module docstring)."""
    exact = cap["exact"]
    n = cap["n"]
    B = len(n)
    gaps = ref["gap"].tolist()
    mism = exempt = checked = 0
    worst = None
    first = None
    for b in range(B):
        for c in range(min(int(n[b]), ref["steps"])):
            g = float(gaps[b][c])
            checked += 1
            if g > 0.0:
                if g <= tie:
                    exempt += 1
                else:
                    mism += 1
                    if first is None:
                        first = (b, c, g)
                if worst is None or g > worst:
                    worst = g
    dls = [float(x) for x in ref["dl"].flatten().tolist()]
    kls = [float(x) for x in ref["kl"].flatten().tolist()]
    dq, kq = _quant(dls), _quant(kls)
    rself = ref.get("self") or dict(ok=False, checked=0, unequal=None, first=None)
    err_ok = ref["errors"][0] == 0 and not any(ref["errors"][1])
    sel_bad = int(ref["sel_bad"].sum())
    parts = dict(
        MAP=dict(ok=not cap["map_errors"] and cap["sel_columns"] > 0,
                 detail="%d committed columns with a stored selection; %d logits rows; errors %s" % (
                     cap["sel_columns"], cap["logits_rows"], cap["map_errors"][:3])),
        REF=dict(ok=bool(rself["ok"]) and err_ok,
                 detail="REF (SH's computation, teacher-forced on this arm's tokens) against SH on the shared prefix: "
                        "%s of %s fingerprints unequal (first %s); REF error flags %s" % (
                            rself["unequal"], rself["checked"], rself["first"], ref["errors"])),
        SEL=dict(ok=ref["sel_checked"] > 0 and sel_bad == 0,
                 detail="selected block sets unequal to REF's: %d of %d (column, layer, KV head) over every request; "
                        "columns with a difference %d; requests %d" % (
                            sel_bad, ref["sel_checked"], int(ref["sel_cols"].sum()), int((ref["sel_bad"] > 0).sum()))),
        GREEDY=dict(ok=checked > 0 and mism == 0,
                    detail="committed tokens that are not REF's argmax: %d of %d (first (request, column, gap) %s); "
                           "exempted near ties (gap <= %g) %d; largest gap %s" % (mism, checked, first, tie, exempt,
                                                                                worst)),
        POS=dict(ok=cap["pos_bad"] == 0 and cap["rounds"] > 0,
                 detail="(layer, request) twin positions != S + committed count after a round: %d over %d rounds" % (
                     cap["pos_bad"], cap["rounds"])),
        SLOT=dict(ok=cap["slot"]["bad"] == 0 and cap["slot"]["checked"] > 0 and cap["ext"]["bad"] == 0,
                  detail="named slots whose bytes are not their block's host rows: %d of %d in %d checks (first %s); "
                         "kept extras slots: %d of %d (first %s)" % (
                             cap["slot"]["bad"], cap["slot"]["checked"], cap["slot"]["checks"], cap["slot"]["first"],
                             cap["ext"]["bad"], cap["ext"]["checked"], cap["ext"]["first"])))
    if exact:
        parts["BITS"] = dict(ok=dq is not None and dq["max"] == 0.0,
                             detail="EXACT: max |dlogit| against REF %s (must be 0 on every compared position)" % (
                                 dq["max"] if dq else None))
    else:
        parts["MISS"] = dict(ok=cap["miss"] == 0 and cap["miss_checked"] > 0,
                             detail="verify-row selected blocks not present: %d of %d (requests %d; per layer %s)" % (
                                 cap["miss"], cap["miss_checked"], cap["miss_requests"], cap["miss_layer"]))
    logit = dict(max_abs_dlogit=dq, kl=kq, positions=(dq or {}).get("n", 0),
                 detail="max |dlogit| %s; KL(REF || %s) %s; sample rows %s (EMPIRICAL AGREEMENT, scope: these rows, the "
                        "committed positions, this cell; not a proof)" % (dq, arm, kq, cap["sample"]))
    return parts, logit


def chk_gates(arm, cap, ref, tie=0.0, plant=False):
    """[(name, must, passed, detail)] and the record of one arm's check."""
    parts, logit = chk_parts(arm, cap, ref, tie)
    ok = all(p["ok"] for p in parts.values())
    failed = [k for k, p in parts.items() if not p["ok"]]
    g = [(("%s-CHK" % arm), not plant, ok, "%s against SH's computation on the same committed prefix: %s (failed parts "
                                          "%s)" % (arm, "PASS" if ok else "FAIL", failed))]
    for k, p in parts.items():
        g.append(("%s-CHK-%s" % (arm, k), False, p["ok"], p["detail"]))
    g.append(("R-%s-CHK-LOGIT" % arm, False, True, logit["detail"]))
    if plant:
        pre = cap["miss"] > 0
        g.append(("%s-PLANT-CHK-PRE" % arm, True, pre, "the planted control misses verify blocks on these requests: %d "
                                                       "(L34: not a no-op)" % cap["miss"]))
        g.append(("%s-PLANT-CHK-CAUGHT" % arm, True, not ok, "the check %s the planted control (failed parts %s)" % (
            "CATCHES" if not ok else "MISSES", failed)))
    rec = dict(ok=ok, failed=failed, parts={k: p["ok"] for k, p in parts.items()},
               details={k: p["detail"] for k, p in parts.items()}, logit=logit, capture={
                   k: v for k, v in cap.items() if k not in ("count", "n")})
    return g, rec


def finite(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
