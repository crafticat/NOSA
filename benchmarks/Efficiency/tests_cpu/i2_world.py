"""The CPU WORLD of tests_cpu/test_i2.py (rung I2 of KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs: arm L4PTU = the tentative buffers of WS-A with the union residency of WS-B, nosi/nosi/
spec_tentative_union.py; not a test file). Load it by file path, as the older tests load their worlds.

It builds on (read their docstrings first):
  wsa_world.py   the WS-A world: run_tent (the shipped greedy decode, the FAST run, SH fed only the committed tokens; per
                 round the residency and host-row checks), Recorder (the WS-G hooks), reference, state classes, TOL
  wsb_world.py   the WS-B world: fake_attention_u (flash_attn_nosa with the kernel's EARLY EXIT for a length-0 entry: a
                 store call can have one), union_bytes (the byte check of a union record), old_module (an older commit's
                 module in the synthetic package)
  spec_world.py  run_vs_shipped (the speculative round runners against the shipped decode), the CPU worlds below it
Every CUDA op is a torch stand-in with the kernel's flat addresses (lesson L28); the stand-ins do not establish GPU
dispatch, streams, ordering, timing or memory.

THIS FILE ADDS
  STU                  nosi/nosi/spec_tentative_union.py in the synthetic package
  tentu_builder(...)   a builder with wsa_world.tent_runner's signature: TentRunnerU (or, cls "L4PT", TentRunner) with
                       attn fake_attention_u, the chosen X (store slots), gemm and union options
  run_tentu(...)       wsa_world.run_tent with that builder (the same evidence)
  count_joint_diff     an INDEPENDENT count of the residency kernel (kernels["diff"]) inside every joint pass
  run_teacher_cpu(...) the CPU twin of the check cell: TentRunnerU.run_teacher on SH's tokens; SH on the same prefix
                       (wsa_world.reference) gives the selections; per row and layer the selection sets, the logits
                       deltas, the argmaxes, the union counters, the diff count, VRF-IDX
  swapped(mods)        the synthetic package's modules {name: module} replaced while a block runs (sys.modules and the
                       package attribute): an older module's relative imports, also the run-time ones (spec_round imports
                       verify_rows in SpecRunner.__init__), get the older modules
  old_module_as(...)   wsb_world.old_module with a tag in the module name (two chains of one commit must not share one)
  old_chain(...)       an older commit's modules as ONE coherent set (each loads while the ones before it are swapped in)
  world_modules(...)   wsa_world's module globals ST / SR replaced while a block runs (its run_tent uses them)
  same_run(a, b)       Gate 0: two runs bit for bit (tokens, fingerprints, logits rows, read() without timing, records,
                       engine and twin state, the tentative buffers and the layer-0 writes)
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


WW = _load("wsa_world")
WB = _load("wsb_world")
SW, W, RW, VW, RE, SR, BC, ST = WW.SW, WW.W, WW.RW, WW.VW, WW.RE, WW.SR, WW.BC, WW.ST
VR = VW.VR
STU = W.pkg_load("spec_tentative_union")
L, BS, TOPK, HKV, V = W.L, W.BS, W.TOPK, W.HKV, W.V
K0, N0, S0, TOL = WW.K0, WW.N0, WW.S0, WW.TOL
TIMING = ("wall_s", "phases", "round_ms", "probe_wait_ms", "prelude_ms")


# ---------------------------------------------------------------------------
# builders and drivers
# ---------------------------------------------------------------------------
def tentu_builder(X=16, gemm="rowgroup", union=True, check=True, stats=True, spill=None, vneg=None, uneg=None,
                  vrecord=False, cls="L4PTU", attn=None):
    """A builder with wsa_world.tent_runner's signature (run_tent calls it). cls "L4PTU": TentRunnerU; "L4PT": TentRunner
    with the same stand-in, X and gemm (the I1 arm in the I2 world); level L3 always gives TentRunner."""
    fa = WB.fake_attention_u if attn is None else attn

    def build(model, cache, S, horizon, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None, poison=False,
              hooks=None, check_every=1, after_round=None, record=True):
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level=level, K=K, mode="exact", gemm=gemm, extra=X, attn=fa, journal=RW.CAP, redirect=True,
                          draft_filter=draft_filter, neg=neg, record=record, probe_every=0, after_round=after_round)
        if cls == "L4PT" or level != "L4P":
            return ST.TentRunner(model, rs, sops, BC.HostClock(), t0=tok0, hooks=hooks, check_every=check_every,
                                 poison=poison, tneg=tneg)
        return STU.TentRunnerU(model, rs, sops, BC.HostClock(), t0=tok0, hooks=hooks, check_every=check_every,
                               poison=poison, tneg=tneg, union=union, union_check=check, union_stats=stats,
                               spill=spill, vneg=vneg, uneg=uneg, vrecord=vrecord)
    return build


@contextlib.contextmanager
def _builder(build):
    orig = WW.tent_runner
    WW.tent_runner = build
    try:
        yield
    finally:
        WW.tent_runner = orig


def run_tentu(B: int, build=None, **kw):
    """wsa_world.run_tent (its evidence) with tentu_builder's runner (default: L4PTU, X 16, gemm rowgroup, check and
    stats on)."""
    with _builder(build if build is not None else tentu_builder()):
        return WW.run_tent(B, **kw)


def count_joint_diff(runner) -> dict:
    """An INDEPENDENT count of the residency kernel (not the module's host counter): the calls of kernels["diff"] inside
    each joint pass (a TentState.run_pass with R > 1). Returns the box: per_pass (list of counts)."""
    kern = runner.rs.ops.kernels
    box = dict(per_pass=[], cur=None)
    kd = kern["diff"]

    def diff(*a, **k):
        if box["cur"] is not None:
            box["cur"] += 1
        return kd(*a, **k)
    kern["diff"] = diff
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


def run_teacher_cpu(B: int, G: int = 6, S: int = S0, seed: int = 0, X: int = 16, gemm: str = "rowgroup", vneg=None,
                    uneg=None, vrecord: bool = False, union: bool = True, spill=None) -> dict:
    """THE CHECK CELL ON THE CPU (module docstring). Three caches of one synthetic prefill: A runs SH's greedy decode
    (tokens, logits); Bc runs TentRunnerU.run_teacher on SH's tokens (one plain step, then G joint passes of R = K + 1
    verify rows, each committed whole); C runs the reference (SH fed the same tokens; its logits bitwise SH's, checked;
    it records the selection of every step and layer). Row r of pass g is step i = 1 + g R + r. Returns the evidence."""
    W.STATS.clear()
    K = K0
    R = K + 1
    n = 1 + G * R
    hz = n + 8
    model, A, Bc = SW.two_caches(B, S, hz + 64, seed)
    C = SW.two_caches(B, S, hz + 64, seed)[1]
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    ref_tok, ref_lg, _ = SW.shipped_greedy(model, A, B, S, n, tok0)       # ref_tok (B, n + 1): column 0 = tok0
    rec = WW.Recorder()
    build = tentu_builder(X=X, gemm=gemm, union=union, check=True, stats=True, spill=spill, vneg=vneg, uneg=uneg,
                          vrecord=vrecord)
    W.TAG[0] = "spec"
    runner = build(model, Bc, S, hz, tok0, hooks=rec, record=False)
    runner.vs.fa_check = {}
    diffs = count_joint_diff(runner)
    lgs = []
    runner.run_teacher(ref_tok[:, :n], G, on_pass=lambda g, lg: lgs.append(lg.clone()))
    stats = VR.read_stats(runner.vs)
    _, sel_ref, ref_bad = WW.reference(model, C, S, hz, ref_tok, torch.full((B,), n, dtype=torch.int64), ref_lg)
    ev = dict(runner=runner, stats=stats, diffs=diffs["per_pass"], fa=runner.vs.fa_check, ref_logits_bad=ref_bad,
              G=G, R=R, B=B, S=S, n=n, changed=[], compared=0, deltas=[], argmax_bad=[], fills=[], past_rows=0)
    for x in rec.sel:                                                # the selections against SH's at each position
        for b in range(B):
            i = int(x["p"][b]) - S
            ref = sel_ref.get((i, x["l"]))
            assert ref is not None, (i, x["l"])
            ev["compared"] += 1
            for h in range(HKV):
                a, r_ = sorted(x["sel"][h, b].tolist()), sorted(ref[h, b].tolist())
                if a != r_:
                    ev["changed"].append((i, x["l"], x["r"], h, b, len(set(a) ^ set(r_))))
    for g, lg in enumerate(lgs):                                     # the logits against SH's
        for r in range(R):
            i = 1 + g * R + r
            for b in range(B):
                d = float((lg[b, r].double() - ref_lg[b, i].double()).abs().max())
                ev["deltas"].append((g, r, b, d))
                if int(lg[b, r].argmax()) != int(ref_tok[b, i + 1]):
                    ev["argmax_bad"].append((g, r, b))
                p = S + i
                if p % BS == BS - 1:
                    ev["fills"].append((g, r, b))
                if p // BS > (S + 1 + g * R) // BS:                  # past the committed block's edge at the pass start
                    ev["past_rows"] += 1
    ev["max_delta"] = max((d for *_, d in ev["deltas"]), default=0.0)
    ev["errors"] = RE.errors(runner.rs)
    return ev


def wsg_modules():
    """r2_wsg_check (WS-G b1ae579, copied unchanged) and r2_i2_check from benchmarks/Efficiency."""
    if str(W.EFF) not in sys.path:
        sys.path.insert(0, str(W.EFF))
    import r2_i2_check
    import r2_wsg_check
    return r2_wsg_check, r2_i2_check


def run_wsg_cpu(kind: str, B: int = 8, X: int = 16, vneg=None, check: bool = True, draft_filter=None, seed: int = 0):
    """THE WS-G CHECK ON THE CPU (the cell's i2g arms): SH's greedy decode on a first cache (tokens, logits); the arm
    (kind WJ: PaperJournalRunner; WT: TentRunner; WU: TentRunnerU with the union checks; vneg: its VerifyOps control)
    on a second cache with r2_i2_check.ContractCapture (check); REF = r2_wsg_check.run_ref on a third cache (the ragged
    step, natural ops: bitwise the shipped step) teacher-forced on the arm's committed tokens. Returns the evidence."""
    CK, ICK = wsg_modules()
    W.STATS.clear()
    K, N, S = K0, N0, S0
    hz = SW.horizon_for(N, K)
    model, A, Bc = SW.two_caches(B, S, hz + 64, seed)
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    ref_tok, ref_lg, _ = SW.shipped_greedy(model, A, B, S, N, tok0)
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
    rs = RE.attach(model, Bc, S, hz, ops)
    RW.define_undefined(rs)
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm="rowgroup", extra=X, attn=WB.fake_attention_u,
                      journal=RW.CAP, redirect=True, draft_filter=draft_filter, record=False, probe_every=0)
    W.TAG[0] = "spec"
    if kind == "WJ":
        runner = ST.PaperJournalRunner(model, rs, sops, BC.HostClock(), t0=tok0)
    elif kind == "WT":
        runner = ST.TentRunner(model, rs, sops, BC.HostClock(), t0=tok0, check_every=1)
    else:
        runner = STU.TentRunnerU(model, rs, sops, BC.HostClock(), t0=tok0, check_every=1, union_check=True,
                                 union_stats=True, vneg=vneg)
    origs = (VR._select_rows, VR.joint_pass, RE._select)
    cap = ICK.ContractCapture(runner, N, S, list(range(B)), exact=(kind == "WJ"), RE=RE, tentative=(kind != "WJ"),
                              slot_every=1) if check else None
    if cap is not None:
        cap.install()
    try:
        runner.run(N)
    finally:
        if cap is not None:
            cap.remove()
    restored = (VR._select_rows, VR.joint_pass, RE._select) == origs and "_set_check" not in runner.__dict__ and (
        kind == "WJ" or ("run_pass" not in runner.T.__dict__ and "_resid" not in runner.T.__dict__))
    rd = runner.read()
    ev = dict(runner=runner, rd=rd, ref_tok=ref_tok, ref_lg=ref_lg, restored=restored, K=K, N=N, S=S, B=B)
    if cap is None:
        return ev
    cap.finish(N)
    _, A2, _ = SW.two_caches(B, S, hz + 64, seed)                    # a third cache of the same prefill
    rs2 = RE.attach(model, A2, S, hz, RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True))
    RW.define_undefined(rs2)
    inputs_ref = cap.ref_inputs(runner.out, ref_tok[:, 0], ref_tok)
    fp_sh = (ref_lg.amax(dim=-1), ref_lg.sum(dim=-1, dtype=torch.float64))
    ref = CK.run_ref(model, rs2, RE, inputs_ref, S, N, cap, fp_sh, ref_tok)
    ev.update(cap=cap.result, ref=ref, ref_restored=RE._select is origs[2])
    return ev


# ---------------------------------------------------------------------------
# Gate 0: older modules in the synthetic package
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def swapped(mods: dict):
    """The synthetic package's modules {name: module} replace this tree's while the block runs (module docstring)."""
    pkg = sys.modules[W.PKG]
    saved = {}
    for name, mod in mods.items():
        full = W.PKG + "." + name
        saved[name] = (sys.modules.get(full), name in pkg.__dict__, pkg.__dict__.get(name))
        sys.modules[full] = mod
        setattr(pkg, name, mod)
    try:
        yield
    finally:
        for name, (m, had, attr) in saved.items():
            full = W.PKG + "." + name
            if m is None:
                sys.modules.pop(full, None)
            else:
                sys.modules[full] = m
            if had:
                setattr(pkg, name, attr)
            elif name in pkg.__dict__:
                delattr(pkg, name)


def old_module_as(name: str, commit: str, tmp_path, tag: str, patch_verify: bool = False):
    """wsb_world.old_module with a TAG in the module name (<PKG>.<name>_<commit[:7]>_<tag>): two chains of one commit
    (all three modules, or spec_tentative alone) bind their imports differently, so they must not share a module."""
    full = "%s.%s_%s_%s" % (W.PKG, name, commit[:7], tag)
    if full in sys.modules:
        return sys.modules[full]
    r = WB.git("show", "%s:nosi/nosi/%s.py" % (commit, name))
    assert r.returncode == 0, r.stderr
    p = tmp_path / ("%s_%s_%s.py" % (name, commit[:7], tag))
    p.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location(full, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    if patch_verify:                                                 # verify_world's stand-ins
        mod.torch = W.TP
        mod.nosa_pooling_per_request = W.fake_pool_per_request
    return mod


def old_chain(commit: str, names, tmp_path, tag: str = "chain") -> dict:
    """nosi/nosi/<name>.py of `commit` for each name, in import order, as ONE coherent set: each loads while the ones
    before it are swapped in, so its module-level relative imports get them; verify_rows gets the CPU world's stand-ins
    (as verify_world patches this tree's). Returns {name: module}."""
    out = {}
    for name in names:
        with swapped(out):
            mod = old_module_as(name, commit, tmp_path, tag, patch_verify=(name == "verify_rows"))
        out[name] = mod
    return out


@contextlib.contextmanager
def world_modules(ST_, SR_):
    """wsa_world's module globals ST / SR (used by run_tent, tent_runner, verifying) replaced while the block runs."""
    old = (WW.ST, WW.SR)
    WW.ST, WW.SR = ST_, SR_
    try:
        yield
    finally:
        WW.ST, WW.SR = old


def _drop(rd: dict, verify_keys=()) -> dict:
    """read() without the timing fields; verify_keys: keys dropped from rd["verify"] (the host counts that WS-B added to
    verify_rows.read_stats: an older verify_rows has none)."""
    out = {k: v for k, v in rd.items() if k not in TIMING}
    if verify_keys and isinstance(out.get("verify"), dict):
        out["verify"] = {k: v for k, v in out["verify"].items() if k not in verify_keys}
    return out


def _same_value(u, v) -> bool:
    if torch.is_tensor(u) or torch.is_tensor(v):
        return torch.is_tensor(u) and torch.is_tensor(v) and RW.same(u, v)
    if isinstance(u, dict) and isinstance(v, dict):
        return set(u) == set(v) and all(_same_value(u[k], v[k]) for k in u)
    if isinstance(u, (list, tuple)) and isinstance(v, (list, tuple)):
        return len(u) == len(v) and all(_same_value(x, y) for x, y in zip(u, v))
    return u == v


def same_run(a: dict, b: dict, verify_keys=()) -> list:
    """Gate 0: the differences of two runs (run_tent's or run_vs_shipped's evidence), bit for bit: the committed tokens,
    the fingerprints, the logits rows, fp_ok, read() (without timing; verify_keys dropped from its verify record), the
    round records, the engine and twin state, and (tentative runners) the buffers, tpos, qc, ctr_c and the layer-0 row
    writes. Returns the names that differ."""
    ra, rb = a["runner"], b["runner"]
    bad = []
    for name in ("out", "fmax", "fsum", "lg_out", "fp_ok"):
        x, y = getattr(ra, name, None), getattr(rb, name, None)
        if (x is None) != (y is None) or (x is not None and not RW.same(x, y)):
            bad.append(name)
    if not _same_value(_drop(a["rd"], verify_keys), _drop(b["rd"], verify_keys)):
        da, db = _drop(a["rd"], verify_keys), _drop(b["rd"], verify_keys)
        bad.append(("read", sorted(k for k in set(da) | set(db) if not _same_value(da.get(k), db.get(k)))))
    if not _same_value(a["records"], b["records"]):
        bad.append("records")
    d = RW.diff(RW.state(ra.rs), RW.state(rb.rs))
    if d:
        bad.append(("state", d[:5]))
    Ta, Tb = getattr(ra, "T", None), getattr(rb, "T", None)
    if Ta is not None or Tb is not None:
        if Ta is None or Tb is None:
            bad.append("T")
        else:
            for name in ("k", "v", "bias"):
                if not all(RW.same(x, y) for x, y in zip(getattr(Ta, name), getattr(Tb, name))):
                    bad.append("T." + name)
            for name in ("tpos", "qc", "ctr_c", "skipped", "forced", "injected"):
                if not RW.same(getattr(Ta, name), getattr(Tb, name)):
                    bad.append("T." + name)
            if not _same_value(Ta.rec, Tb.rec):
                bad.append("T.rec")
    return bad
