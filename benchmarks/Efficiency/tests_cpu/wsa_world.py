"""The CPU WORLD of tests_cpu/test_wsa.py (WS-A of KS/analysis/plan/PLAN_L4P_faithful.md: the tentative buffers,
nosi/nosi/spec_tentative.py; not a test file). It builds on spec_world.py (read its docstring first: the SHIPPED engine
from its own source, every CUDA op a torch stand-in with the kernel's flat addresses, lesson L28).

THIS FILE ADDS
  tent_runner(...)        ragged_engine.attach on a cache (ops natural / twin torch / body torch: the shipped bits),
                          define_undefined, spec_tentative.TentRunner (FAST, check rounds = every round)
  Recorder                the WS-G hooks (on_select / on_logits): the target selection of every row of every layer and
                          the joint pass's logits; CHECK-ONLY
  reference(...)          SH ON THE SAME COMMITTED PREFIX: a ragged world (bitwise the shipped step: test_ragged_engine,
                          and checked here against the shipped logits) fed the shipped greedy tokens, request b exactly
                          its committed count; it records the selection of every step and layer (the decode_fixed
                          argument topk_idx)
  run_tent(...)           the shipped greedy decode (A), the FAST run (Bc) and the reference (C) from ONE synthetic
                          prefill (precondition: the caches are bitwise equal before step 0); per-round checks; the
                          evidence (tokens, logits deltas, selections, facts, state classes)
  tent_residency_errors   every named window / pool slot holds its block's host rows and cis rows (a cache may hold
                          ANY blocks: only self-consistency); no id twice; the tail id and the attention length follow
                          the COMMITTED position (the twin is at the chain's frontier between rounds)
  host_poison_errors      host rows from the committed block on are still NaN (no tentative or draft row reached them)
  state_classes(...)      the committed state of the FAST run against the reference, field by field, in three classes:
                          EXACT (must be equal), ROUNDED (layer 0 bitwise, every layer within atol: float order),
                          CACHE (may differ: residency contents and order; reported)
  OpCount                 a TorchFunctionMode that counts torch calls (the fixed-cost claim); one_launch() counts a
                          callable that is one GPU launch (the device write-back kernel) but a host loop on the CPU as 1
  JOURNAL, verifying      spec_world.run_vs_shipped(..., sr=JOURNAL) runs spec_tentative.PaperJournalRunner (the EXACT
                          journal runner of the schedule of contract 4); verifying(record) gives the requests of a round
                          that verified drafts (a group or a final group)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SW = _load("spec_world")
W, RW, VW, RE, SR, BC = SW.W, SW.RW, SW.VW, SW.RE, SW.SR, SW.BC
ST = W.pkg_load("spec_tentative")
L, BS, TOPK, HKV, V = W.L, W.BS, W.TOPK, W.HKV, W.V
K0, N0, S0 = SW.K0, SW.N0, SW.S0
TOL = 1.0e-4                    # |logit - SH's|: float order gives about 1e-6 here; a missing / extra row gives >> 1e-3


def bits_equal(a, b) -> bool:
    return SW.bits_equal(a, b)


# ---------------------------------------------------------------------------
# the hooks (WS-G) and the op counter
# ---------------------------------------------------------------------------
class Recorder:
    """CHECK-ONLY hooks: on_select keeps the rows that have a verify request; on_logits every call."""

    def __init__(self):
        self.sel, self.logits = [], []
        self.round = None

    def on_select(self, l, kind, r, pos, verify, sel):
        if bool(verify.any()):
            self.sel.append(dict(round=self.round, l=l, kind=kind, r=r, p=pos.clone(), verify=verify.clone(),
                                 sel=sel.clone()))

    def on_logits(self, rnd, kind, pos, verify, logits):
        self.logits.append(dict(round=rnd, kind=kind, pos=pos.clone(), verify=verify.clone(), logits=logits.clone()))


class OpCount(TorchFunctionMode):
    """Counts the torch calls (functions and Tensor methods) issued while it is active, except while paused."""

    def __init__(self):
        super().__init__()
        self.n = 0
        self.paused = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not self.paused:
            self.n += 1
        return func(*args, **(kwargs or {}))


def count_calls(obj, name, box, mode):
    """Wraps obj.<name>: every call runs inside `mode` and appends its op count to box[name]."""
    orig = getattr(obj, name)

    def f(*a, **k):
        n0 = mode.n
        with mode:
            out = orig(*a, **k)
        box.setdefault(name, []).append(mode.n - n0)
        return out
    setattr(obj, name, f)


def one_launch(body, mode):
    """The device write-back is ONE kernel launch on the GPU; its CPU reference loops on the host over the full
    requests. Count it as one call."""
    orig = body.writeback_kernel

    def f(*a, **k):
        mode.n += 1
        mode.paused = True
        try:
            return orig(*a, **k)
        finally:
            mode.paused = False
    body.writeback_kernel = f


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------
def tent_runner(model, cache, S: int, horizon: int, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None,
                poison=False, hooks=None, check_every=1, after_round=None, record=True):
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
    rs = RE.attach(model, cache, S, horizon, ops)
    RW.define_undefined(rs)
    sops = SR.SpecOps(level=level, K=K, mode="exact", gemm="rowgroup", extra=16, attn=VW.fake_attention_idx,
                      journal=RW.CAP, redirect=True, draft_filter=draft_filter, neg=neg, record=record, probe_every=0,
                      after_round=after_round)
    return ST.TentRunner(model, rs, sops, BC.HostClock(), t0=tok0, hooks=hooks, check_every=check_every,
                         poison=poison, tneg=tneg)


def reference(model, cache, S: int, horizon: int, ref_tok, counts, ref_lg):
    """SH on the same committed prefix (module docstring). counts (B,): request b feeds ref_tok[b, :counts[b]] at
    S .. S + counts[b] - 1. Returns (rs, sel {(i, l): (H, B, topk)}, logits_bad: the (step, request) pairs whose logits
    differ from the shipped ones, bitwise)."""
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
    rs = RE.attach(model, cache, S, horizon, ops)
    RW.define_undefined(rs)
    B = rs.B
    box = dict(step=0, calls=0)
    sel = {}
    orig = rs.body.decode_fixed

    def hook(eng, key_states, value_states, kv_bias, topk_idx, *a, **k):
        l = box["calls"] % L
        sel[(box["step"], l)] = topk_idx.clone()
        box["calls"] += 1
        return orig(eng, key_states, value_states, kv_bias, topk_idx, *a, **k)
    rs.body.decode_fixed = hook
    cnt = torch.as_tensor(counts, dtype=torch.int64)
    bad = []
    n_ref = ref_lg.shape[1]
    for i in range(int(cnt.max())):
        act = i < cnt
        RE.set_activity(rs, act.clone(), act.clone())
        box["step"] = i
        W.TAG[0] = "ref"
        lg = RE.decode_step(model, rs, ref_tok[:, i:i + 1].contiguous(),
                            torch.full((B, 1), S + i, dtype=torch.int32))[:, -1]
        for b in range(B):
            if bool(act[b]) and i < n_ref and not bits_equal(lg[b], ref_lg[b, i]):
                bad.append((i, b))
    del rs.body.decode_fixed
    RE.set_activity(rs, torch.ones(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool))
    return rs, sel, bad


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def tent_residency_errors(rs, qc) -> list:
    """The residency invariants with the COMMITTED position qc (B,) (module docstring)."""
    errs = []
    for l, lay in enumerate(rs.cache.layers):
        e, tw = lay.cache_engine, rs.st.twins[l]
        P = int(e._pool_map.shape[2]) if rs.pool else 0
        for b in range(rs.B):
            q = int(qc[b])
            tail = q // BS
            want = int(rs.body.cache_lens_value(e, torch.tensor([q - 1]))[0])
            if int(e._cache_lens[b]) != want:
                errs.append(("lens", l, b, int(e._cache_lens[b]), want))
            for h in range(HKV):
                win = e._block_map[h, b].tolist()
                pool = e._pool_map[h, b].tolist() if P else []
                if win[TOPK - 1] != tail:
                    errs.append(("tail", l, h, b, win[TOPK - 1], tail))
                ids = [x for x in win[:TOPK - 1] if x >= 0] + [y for y in pool if y >= 0]
                if len(ids) != len(set(ids)) or tail in ids or any(x >= tail for x in ids):
                    errs.append(("dup_or_uncommitted", l, h, b))
                for s, x in enumerate(win[:TOPK - 1]):
                    if x >= 0 and not SW._slot_holds(e, tw, b, h, s, x):
                        errs.append(("win", l, h, b, s, x))
                for qq, y in enumerate(pool):
                    if y >= 0 and not SW._slot_holds(e, tw, b, h, TOPK + qq, y):
                        errs.append(("pool", l, h, b, qq, y))
    return errs


def host_poison_errors(rs, S: int, qc) -> list:
    """Host rows from max(S, 64 (qc // 64)) on are NaN (the poison of the CPU world)."""
    errs = []
    for l, lay in enumerate(rs.cache.layers):
        e = lay.cache_engine
        for b in range(rs.B):
            n0 = max(S, (int(qc[b]) // BS) * BS)
            if not (bool(torch.isnan(e._k_cpu[b, n0:]).all()) and bool(torch.isnan(e._v_cpu[b, n0:]).all())):
                errs.append(("tail_written", l, b, n0))
    return errs


STATE_EXACT = ("twin.pos", "twin.n_comp", "twin.n_ccis", "tail_id", "cache_lens", "host_poison", "residency",
               "committed_counters")
STATE_ROUNDED = ("twin.comp_k", "twin.ccis", "twin.total_cis", "twin.ring_live", "tail_rows.k", "tail_rows.v",
                 "tail_rows.bias", "host_rows.k", "host_rows.v")
STATE_CACHE = ("window.contents", "window.slot_order", "pool.contents", "pool.age_order", "clock")


def state_classes(rsF, rsR, qc, S: int, atol: float = TOL) -> dict:
    """The committed state of the FAST run (its twins reset to the committed counters by the caller) against the
    reference that fed only the committed tokens. Returns dict(exact=[names that differ], l0=[layer-0 names not
    bitwise], close=[names beyond atol], maxabs, cache={name: count of (layer, head, request) rows that differ})."""
    out = dict(exact=[], l0=[], close=[], maxabs=0.0, cache={n: 0 for n in STATE_CACHE})

    def cmp(name, l, x, y):
        if l == 0 and not bits_equal(x, y):
            out["l0"].append((name, l))
        if x.numel():
            d = (x.double() - y.double()).abs()
            fin = torch.isfinite(x) & torch.isfinite(y)
            m = float(d[fin].max()) if bool(fin.any()) else 0.0
            same_nan = bool((torch.isnan(x) == torch.isnan(y)).all())
            out["maxabs"] = max(out["maxabs"], m)
            if m > atol or not same_nan:
                out["close"].append((name, l, m))
    for l in range(L):
        eF, eR = rsF.cache.layers[l].cache_engine, rsR.cache.layers[l].cache_engine
        tF, tR = rsF.st.twins[l], rsR.st.twins[l]
        for n in ("pos", "n_comp", "n_ccis"):
            if not torch.equal(getattr(tF, n), getattr(tR, n)):
                out["exact"].append(("twin." + n, l))
        if not torch.equal(tF.pos.cpu(), torch.as_tensor(qc).cpu()):
            out["exact"].append(("committed_counters", l))
        if not torch.equal(eF._block_map[:, :, TOPK - 1], eR._block_map[:, :, TOPK - 1]):
            out["exact"].append(("tail_id", l))
        if not torch.equal(eF._cache_lens, eR._cache_lens):
            out["exact"].append(("cache_lens", l))
        for b in range(rsF.B):
            q, nk, nc = int(tF.pos[b]), int(tF.n_comp[b]), int(tF.n_ccis[b])
            cmp("twin.comp_k", l, tF.comp_k[b, :nk], tR.comp_k[b, :nk])
            cmp("twin.ccis", l, tF.ccis[:, b, :nc], tR.ccis[:, b, :nc])
            cmp("twin.total_cis", l, tF.total_cis[b, :q], tR.total_cis[b, :q])
            live = torch.arange(16 * nk, q) % 64
            cmp("twin.ring_live", l, tF.ring[b, live], tR.ring[b, live])
            t0, nt = (TOPK - 1) * BS, q % BS
            cmp("tail_rows.k", l, eF._k_gpu[b, t0:t0 + nt], eR._k_gpu[b, t0:t0 + nt])
            cmp("tail_rows.v", l, eF._v_gpu[b, t0:t0 + nt], eR._v_gpu[b, t0:t0 + nt])
            cmp("tail_rows.bias", l, eF._kv_bias_gpu[b, t0:t0 + nt], eR._kv_bias_gpu[b, t0:t0 + nt])
            a0, z = (S // BS) * BS, (q // BS) * BS
            if z > a0:
                cmp("host_rows.k", l, eF._k_cpu[b, a0:z], eR._k_cpu[b, a0:z])
                cmp("host_rows.v", l, eF._v_cpu[b, a0:z], eR._v_cpu[b, a0:z])
            for h in range(HKV):
                wf, wr = eF._block_map[h, b, :TOPK - 1].tolist(), eR._block_map[h, b, :TOPK - 1].tolist()
                if set(x for x in wf if x >= 0) != set(x for x in wr if x >= 0):
                    out["cache"]["window.contents"] += 1
                elif wf != wr:
                    out["cache"]["window.slot_order"] += 1
                if rsF.pool:
                    pf, pr = eF._pool_map[h, b].tolist(), eR._pool_map[h, b].tolist()
                    if set(pf) != set(pr):
                        out["cache"]["pool.contents"] += 1
                    elif not torch.equal(eF._pool_age[h, b].argsort(), eR._pool_age[h, b].argsort()):
                        out["cache"]["pool.age_order"] += 1
        if rsF.pool and eF._pool_stamp_base != eR._pool_stamp_base:
            out["cache"]["clock"] += 1
    if host_poison_errors(rsF, S, qc):
        out["exact"].append(("host_poison", -1))
    if tent_residency_errors(rsF, qc):
        out["exact"].append(("residency", -1))
    return out


# ---------------------------------------------------------------------------
# the driver of the claims
# ---------------------------------------------------------------------------
def verifying(r):
    """(B,) bool: the requests of a round record that verified drafts (L4P: a group or a final; L3: every request)."""
    if "role" not in r:
        return torch.ones_like(r["n"], dtype=torch.bool)
    role = r["role"].to(r["n"].device)
    return (role == ST.GROUP) | (role == ST.FINAL)


# spec_world.run_vs_shipped(..., sr=JOURNAL) builds the EXACT journal runner of the schedule of contract 4
JOURNAL = type(sys)("spec_round_contract4")
JOURNAL.SpecOps, JOURNAL.SpecRunner = SR.SpecOps, ST.PaperJournalRunner


def run_tent(B: int, level="L4P", K=K0, S=S0, N=N0, seed=0, draft_filter=None, neg=None, tneg=None, poison=False,
             oracle_rounds=(-1, 0), max_rounds=0, check_rounds=True, with_reference=True, on_runner=None):
    """THE SHIPPED GREEDY DECODE (A), the FAST run (Bc) and SH ON THE SAME COMMITTED PREFIX (C). After every round
    (check_rounds): tent_residency_errors and host_poison_errors with the committed position (collected). Returns
    the evidence: tokens / logits against SH, the selection mismatches of committed verify rows, the state classes,
    the facts (preconditions), the runner, the reference."""
    W.STATS.clear()
    hz = SW.horizon_for(N, K)
    n_ref = N + 2 * K + 2
    model, A, Bc = SW.two_caches(B, S, hz + 64, seed)
    C = SW.two_caches(B, S, hz + 64, seed)[1]
    pre = VW.engines_equal(A, C)
    assert pre == [], "precondition: the reference cache differs before step 0: %s" % pre[:5]
    g = torch.Generator().manual_seed(seed + 2)
    tok0 = torch.randint(0, V, (B,), generator=g)
    ref_tok, ref_lg, ref_host = SW.shipped_greedy(model, A, B, S, n_ref, tok0)
    if oracle_rounds:
        draft_filter = SW.all_kinds(SW.inject(ref_tok, tuple(oracle_rounds)), draft_filter)
    rec = Recorder()
    ev = dict(resid=[], host=[], round_q=[])

    def after_round(runner):
        q = runner.T.qc.clone()
        ev["round_q"].append(q)
        if check_rounds:
            r = tent_residency_errors(runner.rs, q)
            if r:
                ev["resid"].append((runner.round, r[:5]))
            h = host_poison_errors(runner.rs, S, q)
            if h:
                ev["host"].append((runner.round, h[:5]))
    W.TAG[0] = "spec"
    runner = tent_runner(model, Bc, S, hz, tok0, level=level, K=K, draft_filter=draft_filter, neg=neg, tneg=tneg,
                         poison=poison, hooks=rec, after_round=after_round)
    runner.T.rec = []
    if on_runner is not None:
        on_runner(runner)
    orig_set = runner._set_check

    def set_check(g_, probe):                                        # the recorder knows the round of a row: the
        rec.round = runner.round                                     # runner's round (-1 in the L4P prelude)
        orig_set(g_, probe)
    runner._set_check = set_check
    runner.run(N, max_rounds=max_rounds)
    rd = runner.read()
    cnt = torch.tensor(rd["count"])
    bad_tok, maxd, nrows = [], 0.0, 0
    for b in range(B):
        c = min(int(cnt[b]), n_ref, runner.cap)     # out holds cap columns (the round's writes past cap are masked,
        got, want = runner.out[b, :c].cpu(), ref_tok[b, 1:c + 1]   # spec_tentative.py:510-511); L3 may commit past N
        if not torch.equal(got, want):
            i = int((got != want).nonzero()[0, 0])
            bad_tok.append((b, i, int(got[i]), int(want[i])))
        for i in range(c):
            d = float((runner.lg_out[b, i].double() - ref_lg[b, i].double()).abs().max())
            maxd = max(maxd, d)
            nrows += 1
    recs = runner.records
    # the committed rows of round g, request b: the positions q_g <= p < q_end_g (a verify row past q + m was rejected)
    committed = {}
    for gi, r in enumerate(recs):
        for b in range(B):
            committed[(gi, b)] = (int(r["q"][b]), int(r["q_end"][b]))
    out = dict(runner=runner, rd=rd, rec=rec, records=recs, bad_tokens=bad_tok, max_logit_delta=maxd,
               rows_compared=nrows, ref_tok=ref_tok, ref_lg=ref_lg, ref_host=ref_host, S=S, N=N, K=K, B=B,
               model=model, committed=committed, cnt=cnt)
    out.update(ev)
    tr = rd["tent"]
    facts = dict(rejects=sum(int((verifying(r) & (r["n"] < r["v"])).sum()) for r in recs),
                 full=sum(int((verifying(r) & r["full"]).sum()) for r in recs), skipped=sum(rd["draft_skipped"]),
                 forced=sum(tr["forced"]), injected=sum(tr["injected"]), edge_rows=tr["edge_rows"],
                 fill_rows=tr["fill_rows"], crossed=tr["crossed"], dropped=tr["dropped"],
                 deferred_writes=tr["deferred_writes"], replays=tr["replays"],
                 move_out=W.STATS.get("spec", {}).get("move_out", 0), swap=W.STATS.get("spec", {}).get("swap", 0),
                 ragged=any(len(set(q.tolist())) > 1 for q in ev["round_q"]), min_count=int(cnt.min()),
                 rounds=rd["rounds"])
    out["facts"] = facts
    if with_reference:
        rsR, sel_ref, ref_bad = reference(model, C, S, hz, ref_tok, cnt, ref_lg)
        out.update(ref_rs=rsR, ref_sel=sel_ref, ref_logits_bad=ref_bad)
        mism, compared = [], 0
        for x in rec.sel:
            gi = x["round"]
            for b in range(B):
                if not bool(x["verify"][b]):
                    continue
                p = int(x["p"][b])
                lo, hi = committed.get((gi, b), (0, 0))
                if not lo <= p < hi:
                    continue
                ref = sel_ref.get((p - S, x["l"]))
                if ref is None:
                    continue
                compared += 1
                for h in range(HKV):
                    if sorted(x["sel"][h, b].tolist()) != sorted(ref[h, b].tolist()):
                        mism.append((gi, x["kind"], x["r"], x["l"], h, b, p))
        out.update(sel_mismatch=mism, sel_compared=compared)
    return out
