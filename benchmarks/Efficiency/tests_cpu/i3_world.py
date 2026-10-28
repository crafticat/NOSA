"""The CPU WORLD of tests_cpu/test_i3.py and tests_cpu/test_i3_interp.py (rung I3 of KS/analysis/plan/
PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs: arm L4PTF = the tentative buffers of WS-A with the union
residency of WS-B and ONE fork call per layer of WS-C, nosi/nosi/spec_tentative_fork.py; not a test file). Load it by
file path, as the older tests load their worlds.

It builds on (read their docstrings first):
  i2_world.py    run_tentu / tentu_builder (the I2 world: WS-A's run_tent with a builder), run_teacher_cpu, the Gate-0
                 tools (old_chain, swapped, same_run), run_wsg_cpu
  wsc_world.py   fake_fork (the fork's torch stand-in, the CPU world's bias convention), make_fake_fork, TeeFork,
                 use_kernel_bias (the interpreter child: SH with the real kernels' (q.k + bias) x scale), FORK_CALLS
  wsa_world.py   run_tent (the shipped greedy decode, the FAST run, SH fed the committed tokens), reference, Recorder
Every CUDA op is a torch stand-in with the kernel's flat addresses (lesson L28); the stand-ins do not establish GPU
dispatch, streams, ordering, timing or memory.

THIS FILE ADDS
  STF, IA              nosi/nosi/spec_tentative_fork.py (synthetic package); benchmarks/Efficiency/r2_i3_arena.py
  tentf_builder(...)   a builder with wsa_world.tent_runner's signature: the cache re-homed into a ChainArena (store X =
                       1 + (full_max - 1) x 64 unless X is given) BEFORE ragged_engine.attach, rs.kv_arena, and
                       TentRunnerF with the fork (default: wsc_world.fake_fork)
  run_tentf(...)       wsa_world.run_tent with that builder (the same evidence)
  count_fork_calls     an INDEPENDENT count of the fork's launches inside every joint pass (the callable is wrapped)
  run_teacher_f(...)   the CPU twin of the check cell: TentRunnerF.run_teacher on SH's tokens (warm plain steps first);
                       SH on the same prefix gives the selections; per row and layer the selection sets, the logits
                       deltas, the argmaxes, the counters, the diff count, the "F" / "RA" probes
  chain_bytes(log)     the byte check of the fork_t records: every (row, block, token) that the KERNEL's tables show is
                       mapped to its location and must hold the right bytes (module docstring of the function)
  use_kernel_entry()   the interpreter child only: TentState's own attention calls (call A / call B of the one-row
                       passes: verify_world._entry) with the real kernels' bias convention too
  run_wsg_cpu_f(...)   i2_world.run_wsg_cpu for the fork arms (WF, WFP: the planted control fork_small_store)
  arm_run(...)         Gate 0: one run of a branch arm with this tree's modules or an older chain
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


I = _load("i2_world")
Z = _load("wsc_world")
WW, WB, SW, W, RW, VW, RE, SR, BC, ST, STU, VR = (I.WW, I.WB, I.SW, I.W, I.RW, I.VW, I.RE, I.SR, I.BC, I.ST, I.STU,
                                                  I.VR)
KA = Z.KA
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i3_arena as IA          # noqa: E402

STF = W.pkg_load("spec_tentative_fork")
L, BS, TOPK, HKV, V = W.L, W.BS, W.TOPK, W.HKV, W.V
K0, N0, S0, TOL = WW.K0, WW.N0, WW.S0, WW.TOL
FORK_CALLS = Z.FORK_CALLS


# ---------------------------------------------------------------------------
# builders and drivers
# ---------------------------------------------------------------------------
def chain_arena(cache, B: int, X: int):
    """A ChainArena for a built or prefilled CPU world and the re-home (r2_kv_arena.rehome)."""
    a = IA.for_engine(cache.layers[0].cache_engine, len(cache.layers), B, X)
    KA.rehome(cache, a)
    return a


def tentf_builder(X=None, gemm="rowgroup", check=True, stats=True, vneg=None, fneg=None, vrecord=False, full_max=None,
                  fork=None, attn=None, cls="L4PTF"):
    """A builder with wsa_world.tent_runner's signature (run_tent calls it). cls "L4PTF": TentRunnerF (the cache
    re-homed into a ChainArena first); "L4PTU" / "L4PT": i2_world.tentu_builder's runners (no arena); level L3 always
    gives TentRunner."""
    fa = WB.fake_attention_u if attn is None else attn
    fk = Z.fake_fork if fork is None else fork

    def build(model, cache, S, horizon, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None, poison=False,
              hooks=None, check_every=1, after_round=None, record=True):
        if cls != "L4PTF" or level != "L4P":
            return I.tentu_builder(X=(X or 16), gemm=gemm, check=check, stats=stats, cls=cls, attn=attn)(
                model, cache, S, horizon, tok0, level=level, K=K, draft_filter=draft_filter, neg=neg, tneg=tneg,
                poison=poison, hooks=hooks, check_every=check_every, after_round=after_round, record=record)
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        fm = K if full_max is None else full_max
        arena = chain_arena(cache, B, X or KA.store_slots(TOPK, fm))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level=level, K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, draft_filter=draft_filter, neg=neg, record=record, probe_every=0,
                          after_round=after_round)
        return STF.TentRunnerF(model, rs, sops, BC.HostClock(), fork=fk, t0=tok0, hooks=hooks,
                               check_every=check_every, poison=poison, tneg=tneg, union_check=check,
                               union_stats=stats, full_max=full_max, vneg=vneg, fneg=fneg, vrecord=vrecord)
    return build


def run_tentf(B: int, build=None, **kw):
    """wsa_world.run_tent (its evidence) with tentf_builder's runner (default: L4PTF, the bound X, gemm rowgroup, check
    and stats on)."""
    return I.run_tentu(B, build=(build if build is not None else tentf_builder()), **kw)


def count_fork_calls(runner) -> dict:
    """An INDEPENDENT count of the fork's launches (the callable on the VerifyOps is wrapped; the module's host counter
    is not read) inside every joint pass (TentState.run_pass with R > 1). Returns the box: per_pass (list)."""
    vs = runner.vs
    box = dict(per_pass=[], cur=None)
    fk = vs.ops.fork

    def fork(*a, **k):
        if box["cur"] is not None:
            box["cur"] += 1
        return fk(*a, **k)
    vs.ops.fork = fork
    T = runner.T
    orig = T.run_pass

    def run_pass(model, plan):
        if plan.R == 1:
            return orig(model, plan)
        box["cur"] = 0
        try:
            return orig(model, plan)
        finally:
            box["per_pass"].append(box["cur"])
            box["cur"] = None
    T.run_pass = run_pass
    return box


def run_teacher_f(B: int, G: int = 6, S: int = S0, seed: int = 0, warm: int = 1, gemm: str = "rowgroup", vneg=None,
                  fneg=None, vrecord: bool = False, fork=None, X=None, probe: bool = True, attn=None) -> dict:
    """THE CHECK CELL ON THE CPU (module docstring). Three caches of one synthetic prefill: A runs SH's greedy decode
    (tokens, logits); Bc runs TentRunnerF.run_teacher on SH's tokens (warm plain steps, then G joint passes of R =
    K + 1 verify rows, each committed whole; full_max R, the store 1 + (R - 1) x 64 unless X); C runs the reference
    (SH fed the same tokens; its logits bitwise SH's, checked; it records the selection of every step and layer). Row r
    of pass g is step i = warm + g R + r. Returns the evidence."""
    W.STATS.clear()
    K = K0
    R = K + 1
    n = warm + G * R
    hz = n + 8
    model, A, Bc = SW.two_caches(B, S, hz + 64, seed)
    C = SW.two_caches(B, S, hz + 64, seed)[1]
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    ref_tok, ref_lg, _ = SW.shipped_greedy(model, A, B, S, n, tok0)       # ref_tok (B, n + 1): column 0 = tok0
    rec = WW.Recorder()
    build = tentf_builder(X=X, gemm=gemm, check=True, stats=True, vneg=vneg, fneg=fneg, vrecord=vrecord, full_max=R,
                          fork=fork, attn=attn)
    W.TAG[0] = "spec"
    runner = build(model, Bc, S, hz, tok0, hooks=rec, record=False)
    if probe:
        runner.vs.fa_check = {}
    diffs = I.count_joint_diff(runner)
    forks = count_fork_calls(runner)
    lgs = []
    runner.run_teacher(ref_tok[:, :n], G, on_pass=lambda g, lg: lgs.append(lg.clone()), warm=warm)
    stats = VR.read_stats(runner.vs)
    _, sel_ref, ref_bad = WW.reference(model, C, S, hz, ref_tok, torch.full((B,), n, dtype=torch.int64), ref_lg)
    ev = dict(runner=runner, stats=stats, diffs=diffs["per_pass"], forks=forks["per_pass"], fa=runner.vs.fa_check,
              ref_logits_bad=ref_bad, G=G, R=R, B=B, S=S, n=n, warm=warm, changed=[], compared=0, deltas=[],
              argmax_bad=[], fills=[], past_rows=0)
    for x in rec.sel:                                                # the selections against SH's at each position
        for b in range(B):
            i = int(x["p"][b]) - S
            if i < warm:                                             # the plain steps are SH's own
                continue
            ref = sel_ref.get((i, x["l"]))
            assert ref is not None, (i, x["l"])
            ev["compared"] += 1
            for h in range(HKV):
                a, r_ = sorted(x["sel"][h, b].tolist()), sorted(ref[h, b].tolist())
                if a != r_:
                    ev["changed"].append((i, x["l"], x["r"], h, b, len(set(a) ^ set(r_))))
    for g, lg in enumerate(lgs):                                     # the logits against SH's
        for r in range(R):
            i = warm + g * R + r
            for b in range(B):
                d = float((lg[b, r].double() - ref_lg[b, i].double()).abs().max())
                ev["deltas"].append((g, r, b, d))
                if int(lg[b, r].argmax()) != int(ref_tok[b, i + 1]):
                    ev["argmax_bad"].append((g, r, b))
                p = S + i
                if p // BS == (S + warm + g * R) // BS and p % BS == BS - 1:
                    ev["fills"].append((g, r, b))
                if p // BS > (S + warm + g * R) // BS:               # past the committed block's edge at the pass start
                    ev["past_rows"] += 1
    ev["max_delta"] = max((d for *_, d in ev["deltas"]), default=0.0)
    ev["errors"] = RE.errors(runner.rs)
    return ev


# ---------------------------------------------------------------------------
# the byte check of the fork_t records
# ---------------------------------------------------------------------------
def _beq(a, b) -> bool:
    return torch.equal(W.bits(a), W.bits(b))


def chain_bytes(log, R: int) -> dict:
    """For every fork_t record: every union entry (head h, sequence s, u < len) of the KERNEL's tables is mapped to its
    location; an entry a row with a role (verify or draft) SEES (byte n > 0) must belong to the row's request and name a block of the row's own
    selection, and its tokens 0 .. n - 1 must hold the right bytes:
      a window slot (never the tail slot) or a pool slot, or a listed store slot (1 .. X - 1): the block's HOST rows
      (K, V, bias = total_cis), bit for bit;
      CB0 (block qb): tokens x < qc % 64 = the committed tail slot's rows; tokens x >= qc % 64 = the buffer's row x,
      whose tpos is 64 qb + x and at most the row's position p;
      CB1 (store slot 0, block qb + 1): the buffer's row x, tpos 64 (qb + 1) + x <= p.
    Returns dict(checked, chain_checked, bad_contents, foreign, cross_request, tail_slot)."""
    out = dict(checked=0, chain_checked=0, bad_contents=[], foreign=[], cross_request=[], tail_slot=[])
    for rec in (x for x in log if x["kind"] == "fork_t"):
        sel, win, pm, lst, qb, qc = rec["sel"], rec["win"], rec["pool_map"], rec["lst"], rec["qb"], rec["qc"]
        H, B, R_, T = sel.shape
        topk = win.shape[2]
        ns = rec["k_gpu"].shape[1] // BS
        X = rec["X"]
        base, sbase, cbase = rec["base"], rec["store_base"], rec["cb0_base"]
        nch = rec["tab"].shape[1] // B
        tail0 = (topk - 1) * BS
        for h in range(H):
            for s in range(rec["tab"].shape[1]):
                b = s // nch
                c = s % nch
                for j in range(int(rec["ulen"][h, s])):
                    phys = int(rec["tab"][h, s, j])
                    word = int(rec["ext"][h, s, j])
                    kind = None
                    if base <= phys < base + B * ns:
                        bb, sl_ = divmod(phys - base, ns)
                        if sl_ == topk - 1:
                            kind, bid = "tail", int(qb[bb])
                        else:
                            kind = "host"
                            bid = int(win[h, bb, sl_]) if sl_ < topk else int(pm[h, bb, sl_ - topk])
                            src = (rec["k_gpu"], rec["v_gpu"], rec["bias_gpu"], sl_)
                    elif sbase <= phys < sbase + B * X:
                        bb, x = divmod(phys - sbase, X)
                        if x == 0:
                            kind, bid = "cb1", int(qb[bb]) + 1
                        else:
                            kind, bid = "host", int(lst[h, bb, x - 1])
                            src = (rec["store_k"], rec["store_v"], rec["store_bias"], x)
                    elif cbase <= phys < cbase + B:
                        bb = phys - cbase
                        kind, bid = "cb0", int(qb[bb])
                    else:
                        out["cross_request"].append((rec["layer"], h, s, j, "outside"))
                        continue
                    for u in range(4):
                        n = (word >> (8 * u)) & 0xFF
                        r = c * 4 + u
                        if n == 0 or r >= R_:
                            continue
                        if not bool(rec["full"][r, b] | rec["res"][r, b]):
                            continue                                  # a row without a role (its output is not used)
                        out["checked"] += 1
                        if bb != b:
                            out["cross_request"].append((rec["layer"], h, s, j, u))
                            continue
                        if bid not in sel[h, b, r].tolist():
                            out["foreign"].append((rec["layer"], h, b, r, bid))
                            continue
                        if kind == "tail":
                            out["tail_slot"].append((rec["layer"], h, b, r))
                            continue
                        p = int(rec["p_rows"][b, r])
                        if kind == "host":
                            kk, vv, bi_, slot = src
                            dst, rows = slice(slot * BS, slot * BS + n), slice(bid * BS, bid * BS + n)
                            ok = (_beq(kk[b, dst, h], rec["k_cpu"][b, rows, h])
                                  and _beq(vv[b, dst, h], rec["v_cpu"][b, rows, h])
                                  and _beq(bi_[b, dst, h], rec["total_cis"][b, rows, h].to(bi_.dtype)))
                        else:
                            out["chain_checked"] += 1
                            if kind == "cb0":
                                ck, cv, cb_ = rec["cb0_k"][b], rec["cb0_v"][b], rec["cb0_bias"][b]
                            else:
                                ck, cv, cb_ = (rec["store_k"][b, :BS], rec["store_v"][b, :BS],
                                               rec["store_bias"][b, :BS])
                            cq = int(qc[b]) % BS
                            ok = True
                            for x in range(n):
                                if kind == "cb0" and x < cq:          # committed: the tail slot's row
                                    want = (rec["k_gpu"][b, tail0 + x, h], rec["v_gpu"][b, tail0 + x, h],
                                            rec["bias_gpu"][b, tail0 + x, h])
                                else:                                 # the chain: the buffer row of its position
                                    tp = int(rec["tpos"][b, x])
                                    if tp != bid * BS + x or tp > p:
                                        ok = False
                                        break
                                    want = (rec["buf_k"][b, x, h], rec["buf_v"][b, x, h], rec["buf_bias"][b, x, h])
                                if not (_beq(ck[x, h], want[0]) and _beq(cv[x, h], want[1])
                                        and _beq(cb_[x, h], want[2])):
                                    ok = False
                                    break
                        if not ok:
                            out["bad_contents"].append((rec["layer"], h, b, r, bid, kind))
    return out


# ---------------------------------------------------------------------------
# the interpreter child: the one-row passes' attention with the real kernels' bias convention
# ---------------------------------------------------------------------------
def _entry_kernel(qrow, kk, vv, bias_rows, n, per, Hq, Dq):
    """verify_world._entry with the REAL kernels' (q.k + bias) x scale (flash_fwd_kernel.h:968-971, :1021)."""
    Hk = kk.shape[1]
    qq = qrow.reshape(Hk, Hq // Hk, Dq).float()
    parts = []
    lo = 0
    while lo < n:
        hi = min(lo + per * W.KBLOCK_N, n)
        sc = (torch.einsum("hgd,lhd->hgl", qq, kk[lo:hi].float()) +
              bias_rows[lo:hi].float().t().unsqueeze(1)) * Dq ** -0.5
        mx = sc.amax(-1, keepdim=True)
        e = torch.exp(sc - mx)
        parts.append((mx, e.sum(-1, keepdim=True), torch.einsum("hgl,lhd->hgd", e, vv[lo:hi].float())))
        lo = hi
    mx = torch.stack([p_[0] for p_ in parts]).amax(0)
    num = sum(torch.exp(p_[0] - mx) * p_[2] for p_ in parts)
    den = sum(torch.exp(p_[0] - mx) * p_[1] for p_ in parts)
    return (num / den).reshape(Hq, Dq), (mx + torch.log(den)).reshape(Hq)


def use_kernel_entry():
    """THE INTERPRETER CHILD ONLY (a fresh process, after wsc_world.use_kernel_bias): TentState's one-row attention
    (wsb_world.fake_attention_u with the log-sum-exp: verify_world._entry, read at call time) with the real kernels'
    bias convention, so every attention of the run uses one convention."""
    VW._entry = _entry_kernel


# ---------------------------------------------------------------------------
# WS-G's check for the fork arms
# ---------------------------------------------------------------------------
def run_wsg_cpu_f(kind: str, B: int = 8, X=None, vneg=None, check: bool = True, draft_filter=None, seed: int = 0):
    """i2_world.run_wsg_cpu for kind WF (TentRunnerF, the union checks on) and WFP (vneg fork_small_store at X): the
    arm with r2_i2_check.ContractCapture, then REF teacher-forced on the arm's committed tokens."""
    CK, ICK = I.wsg_modules()
    W.STATS.clear()
    K, N, S = K0, N0, S0
    hz = SW.horizon_for(N, K)
    model, A, Bc = SW.two_caches(B, S, hz + 64, seed)
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    ref_tok, ref_lg, _ = SW.shipped_greedy(model, A, B, S, N, tok0)
    W.TAG[0] = "spec"
    runner = tentf_builder(X=X, gemm="rowgroup", check=True, stats=True, vneg=vneg)(
        model, Bc, S, hz, tok0, draft_filter=draft_filter, check_every=1, record=False)
    origs = (VR._select_rows, VR.joint_pass, RE._select)
    cap = ICK.ContractCapture(runner, N, S, list(range(B)), exact=False, RE=RE, tentative=True,
                              slot_every=1) if check else None
    if cap is not None:
        cap.install()
    try:
        runner.run(N)
    finally:
        if cap is not None:
            cap.remove()
    restored = (VR._select_rows, VR.joint_pass, RE._select) == origs and "_set_check" not in runner.__dict__ and (
        "run_pass" not in runner.T.__dict__ and "_resid" not in runner.T.__dict__)
    rd = runner.read()
    ev = dict(runner=runner, rd=rd, ref_tok=ref_tok, ref_lg=ref_lg, restored=restored, K=K, N=N, S=S, B=B, kind=kind)
    if cap is None:
        return ev
    cap.finish(N)
    _, A2, _ = SW.two_caches(B, S, hz + 64, seed)
    rs2 = RE.attach(model, A2, S, hz, RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True))
    RW.define_undefined(rs2)
    inputs_ref = cap.ref_inputs(runner.out, ref_tok[:, 0], ref_tok)
    fp_sh = (ref_lg.amax(dim=-1), ref_lg.sum(dim=-1, dtype=torch.float64))
    ref = CK.run_ref(model, rs2, RE, inputs_ref, S, N, cap, fp_sh, ref_tok)
    ev.update(cap=cap.result, ref=ref, ref_restored=RE._select is origs[2])
    return ev


# ---------------------------------------------------------------------------
# Gate 0: one run of a branch arm
# ---------------------------------------------------------------------------
def arm_run(arm, B, mods=None):
    """Gate 0: one run of an arm of the merged branches with this tree's modules (mods None) or an older chain
    (i2_world.old_chain: {name: module}). L4PT / L4PTU / L4PJ: I2's arms (run_tent / run_vs_shipped with the journal
    runner); L4XU / L4XF: WS-C's (wsb_world.run_spec_union / wsc_world.run_spec_fork)."""
    filt = SW.corrupt(4, 1, kinds=("draft",), phase=2)
    mods = mods or {}
    ST_, SR_, STU_ = mods.get("spec_tentative", ST), mods.get("spec_round", SR), mods.get("spec_tentative_union", STU)
    sw = I.swapped(mods) if mods else contextlib.nullcontext()
    if arm in ("L4PT", "L4PTU"):
        def build(model, cache, S, horizon, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None,
                  poison=False, hooks=None, check_every=1, after_round=None, record=True):
            ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
            rs = RE.attach(model, cache, S, horizon, ops)
            RW.define_undefined(rs)
            sops = SR_.SpecOps(level=level, K=K, mode="exact", gemm="rowgroup", extra=16, attn=WB.fake_attention_u,
                               journal=RW.CAP, redirect=True, draft_filter=draft_filter, neg=neg, record=record,
                               probe_every=0, after_round=after_round)
            if arm == "L4PT":
                return ST_.TentRunner(model, rs, sops, BC.HostClock(), t0=tok0, hooks=hooks, check_every=check_every,
                                      poison=poison, tneg=tneg)
            return STU_.TentRunnerU(model, rs, sops, BC.HostClock(), t0=tok0, hooks=hooks, check_every=check_every,
                                    poison=poison, tneg=tneg, union=True, union_check=True, union_stats=True)
        with sw, I.world_modules(ST_, SR_):
            return I.run_tentu(B, build=build, draft_filter=filt, with_reference=False)
    if arm == "L4PJ":
        sr = type(sys)("journal_contract4_i3")
        sr.SpecOps, sr.SpecRunner = SR_.SpecOps, ST_.PaperJournalRunner
        with sw:
            return SW.run_vs_shipped(B, level="L4P", mode="exact", gemm="rowgroup", draft_filter=filt, sr=sr)
    sr = SR_ if "spec_round" in mods else None
    with sw:
        if arm == "L4XU":
            return WB.run_spec_union(B, level="L4", extra=16, draft_filter=filt, sr=sr)
        if arm == "L4XF":
            return Z.run_spec_fork(B, level="L4", draft_filter=filt, sr=sr)
    raise ValueError(arm)
