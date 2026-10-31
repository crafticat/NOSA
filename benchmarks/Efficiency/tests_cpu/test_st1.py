"""CPU tests of ST1 (AREA 3 of the RECON7 follow-up: the commit of the tentative rows captured and moved by row;
benchmarks/Efficiency/r2_st1_commit.py, the cell r2_st1_cell.py, the step script r2_st1_step.sh). The world: tests_cpu/
i4dc_world.py (the D arms; FakeGraphsDCD) plus, here, FakeCommitGraphs (i4g_world's FakeGraphs whose capture also makes
the write-back ONE kernel node: the CPU write-back reads its mask on the host). CPU stand-ins establish the plumbing, the
bit equality of the M arms with their D arms, the counters and the gates; only the GPU cells establish capture legality,
timing and memory.

GATE 0    test_st1_gate0_files: the tree = RECON7 cd0f0c5 + the ST1 files + ONE declaration block in each older Gate-0
          test; nosi/ and every older file are cd0f0c5's; no kernel source; no hand-written kernel; r2_st1_commit imports
          torch and time only; nobody else names the ST1 modules.
          test_st1_copies: run_arm_st1 = r2_i4dc_cell.run_arm_d line for line, plus the lines marked "# ST1" only.
EQUAL     test_st1_equal (both D arms x the three modes; the check scenario for "rows"): the M run = the D run (every
          read() key but "st1" and "compile"; every state tensor; mode "rows": the KV tables outside the tail slots'
          uncommitted rows); the committed tail rows and the committed host rows equal; 1 warm call, 1 capture, every
          later commit replayed; the overflow counter 0; PRECONDITIONS: a block filled in the run, a commit of m >= 2.
CONTROLS  test_st1_controls: "stale", "no_phase2" and "narrow" each CHANGE the run (not a no-op: lesson L34).
ST2       test_st2_equal (both D arms, timing and check): the MT run (M + the twin section in one advance) = the M run,
          every bit, and = the D run outside the uncommitted tail rows; every joint layer ran ST2 (no fallback); the
          replay counters equal; PRECONDITION: deferred replays ran. test_st2_controls: "no_replay_seq" and "no_shift"
          CHANGE the run. test_st2_pass_plan: the step sequence of a planted plan (group with replay, final, fill, done).
CELL      test_st1_cell_pure (arguments, the gates on CPU evidence pass, their plants fail; the verdicts against the
          registration state), test_st1_registration_state (PRED_STAMP_ST1 / PRED_ST1_BY_B and the GRANT_<cell> lines:
          None or a date -u stamp, as the step script's greps see them), test_step_script_st1 (the repaired pattern: no
          interpreter or hash on login, cmp against git show, the child's /usr/bin/sha256sum (exit 6), the output limit
          (exit 7), granted(), the registration stop; DRYRUN on a clean checkout of nosi-r2-st1).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_st1.py
"""
from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "cd0f0c59c1f1cac8a3ef05f6c3abcc0a2079d926"          # nosi-r2-recon7: RECON7 X1 notes (the ST1 base)
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-st1"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ ="benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
MODULES = ("r2_st1_commit.py", "r2_st1_cell.py")
ST1_NEW = tuple(E_ + m for m in MODULES) + (E_ + "r2_st1_step.sh", T_ + "test_st1.py")
ST1_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                    "i4h", "i4s", "i4cc", "i4dc", "recon7"))
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
# INT (nosi-r2-int: J1 + ST1 + BF1 in one arm, r2_int_cell.py): the J1 and BF1 rungs' files and its own (its own Gate
# 0: tests_cpu/test_int.py); ST1_LATER: its modules import the ST1 modules (scan (f) skips them)
ST1_NEW = ST1_NEW + (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", E_ + "r2_j1_forkbench.py",
                     T_ + "test_j1.py", "docs/NOTES_j1.md",
                     E_ + "r2_bf1_fetch.py", E_ + "r2_bf1_cell.py", E_ + "r2_bf1_step.sh", T_ + "test_bf1.py",
                     E_ + "r2_int_arms.py", E_ + "r2_int_cell.py", E_ + "r2_int_step.sh", T_ + "test_int.py",
                     "docs/NOTES_int.md")                                                         # INT
ST1_LATER = ("r2_int_arms.py", "r2_int_cell.py")                                                  # INT
N_RUN = 48                             # committed tokens per request: about 16 rounds + the drain


class Evidence(UserWarning):
    """An evidence line (the CPU template runs pytest without -s: the warnings summary carries these lines)."""


def _ev(msg: str) -> None:
    print(msg)
    warnings.warn(Evidence(msg), stacklevel=2)


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4dc_world")
TS = _load("test_i4s")
XG = X.XG
BC, L = X.BC, X.L


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _mod(name):
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    return importlib.import_module(name)


# ------------------------------------------------------------------ GATE 0
def test_st1_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(ST1_NEW) | set(ST1_DECL), sorted(changed - set(ST1_NEW) - set(ST1_DECL))      # (a)
    assert all((ROOT / f).is_file() for f in ST1_NEW), [f for f in ST1_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: cd0f0c5's
    for f in ST1_DECL:                                                    # (c) one ST1 block each
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} == set(), (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert not [x for x in body if x.startswith("-")], (f, body[:5])
        assert any("r2_st1_cell.py" in x for x in body) and any("ST1" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    for name in MODULES:                                                   # (d) NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    tree = ast.parse((EFF / "r2_st1_commit.py").read_text())           # (e) the commit module imports torch only
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch", "time"}, mods
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (f) nobody else names them
        if p.name.startswith("r2_st1_") or p.name in ST1_LATER:                                     # INT
            continue
        assert "r2_st1" not in p.read_text(), p.name


def _body_lines(fn):
    """The source lines of a function after its docstring."""
    src = inspect.getsource(fn)
    lines = src.splitlines()
    node = ast.parse(src).body[0]
    first = node.body[0]
    start = first.end_lineno if (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                                 and isinstance(first.value.value, str)) else first.lineno - 1
    return lines[start:]


def test_st1_copies():
    C = _mod("r2_st1_cell")
    nb, ob = _body_lines(C.run_arm_st1), _body_lines(C.DCC.run_arm_d)
    core = [x for x in nb if not x.rstrip().endswith("# ST1")]
    extra = [x for x in nb if x.rstrip().endswith("# ST1")]
    assert core == ob, [(i, a, b) for i, (a, b) in enumerate(zip(core, ob)) if a != b][:3] or (len(core), len(ob))
    assert len(extra) == 2, extra
    _ev("ST1 COPIES: run_arm_st1 = run_arm_d + %d ST1 lines" % len(extra))


# ------------------------------------------------------------------ the M arms (CPU world)
class FakeCommitGraphs(XG.FakeGraphs):
    """i4g_world's FakeGraphs whose capture also records the write-back (rs.body.writeback_kernel) as ONE kernel node
    (the CPU write-back reads its mask on the host; on the GPU the write-back is one captured kernel)."""

    def capture(self, fn, T, vs):
        rec = XG._Recorder()
        body = T.rs.body
        orig = body.writeback_kernel
        body.writeback_kernel = rec.node(orig)
        try:
            with rec:
                out = fn()
        finally:
            body.writeback_kernel = orig
            for t, c in reversed(rec.saved):                          # property 1: the capture executes nothing
                t.copy_(c)
            self.restored += len(rec.saved)
        g = XG.FakeGraph(rec.ops)
        self.graphs.append(g)
        return g, out


def m_builder(arm, mode="rows", sneg=None, check=False, stats=False, kcheck=False, phase_marks=False, probe_every=0,
              twin=False, tneg2=None):
    """The M arm's runner (twin: the MT arm's; the D arm's runner arguments of i4dc_world.d_builder) with
    FakeCommitGraphs."""
    STC = _mod("r2_st1_commit")
    W, WB, Z, I3, KA, RE, RW, SR = X.W, X.WB, X.Z, X.I3, X.KA, X.RE, X.RW, X.SR
    fa, fk = WB.fake_attention_u, Z.fake_fork
    Runner = STC.st1_runner(X.DCL[X.DARMS[arm]][1], mode, sneg, twin=twin, tneg2=tneg2)

    def build(model, cache, S, horizon, tok0, K=X.K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(X.TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm="natural", extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, record=record, probe_every=probe_every)
        kw = dict(fork=fk, t0=tok0, hooks=None, check_every=check_every, union_check=check, union_stats=stats,
                  phase_marks=phase_marks, dneg=None, hneg=None, gneg=None, sneg=None, dgneg=None,
                  cc_backend="inductor", ccneg=None, cdneg=None)
        if arm in X.KEEP:
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=None, kslots=None, cneg=None)
        be = X.FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, st1_backend=FakeCommitGraphs(), **kw)
        be.T = run.T
        return run
    return build


def run_m(arm, mode="rows", sneg=None, scen="timing", B=8, N=N_RUN, seed=0, twin=False, tneg2=None):
    """One run of the M arm (twin: the MT arm) of `arm` (GUDCD, GUDKCD) in i4dc_world's scenario."""
    X.CC.configure()
    s = X.SCEN[scen]
    model, c, S, hz, tok0 = X.world(B, N, seed)
    kc = bool(s["check"]) and arm in X.KEEP
    run = m_builder(arm, mode, sneg, check=s["check"], stats=s["stats"], kcheck=kc, phase_marks=s["phase_marks"],
                    probe_every=s["probe_every"], twin=twin, tneg2=tneg2)(model, c, S, hz, tok0,
                                                                         check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm=arm)


@functools.lru_cache(maxsize=None)
def _plain(arm, scen="timing"):
    return X.run_arm(arm, scen, B=8, N=N_RUN)


_M = {}


def _m(arm, mode, sneg=None, scen="timing", twin=False, tneg2=None):
    key = (arm, mode, sneg, scen, twin, tneg2)
    if key not in _M:
        _plain({"GUDCD": "GUDC", "GUDKCD": "GUDKC"}[arm], scen)        # the C arm first: it compiles the joint helpers
        _plain(arm, scen)                                              # the D arm (the reference)
        _M[key] = run_m(arm, mode, sneg, scen, twin=twin, tneg2=tneg2)
    return _M[key]


def _uncommitted(T):
    """(B, 64) bool: the tail-slot rows that hold no committed position (rows >= qc % 64)."""
    return ~(T.ar64.view(1, T.bs) < (T.qc % T.bs).view(T.B, 1))


def _kv_masked(run):
    """Every layer's K, V and bias tables with the tail slots' uncommitted rows zeroed (bits)."""
    T = run.T
    drop = _uncommitted(T)
    out = []
    for lay in run.rs.cache.layers:
        e = lay.cache_engine
        for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu):
            x = X.W.bits(t).clone()
            x[:, T.base:T.base + T.bs].masked_fill_(drop.view((T.B, T.bs) + (1,) * (x.dim() - 2)), 0)
            out.append(x)
    return out


def _host_committed(run):
    """Every layer's host K / V rows below each request's committed position (bits)."""
    T = run.T
    out = []
    for lay in run.rs.cache.layers:
        e = lay.cache_engine
        n = e._k_cpu.shape[1]
        keep = torch.arange(n).view(1, n) < T.qc.cpu().view(-1, 1)
        for t in (e._k_cpu, e._v_cpu):
            x = X.W.bits(t).clone()
            x.masked_fill_(~keep.view((T.B, n) + (1,) * (x.dim() - 2)), 0)
            out.append(x)
    return out


KV_KEYS = tuple("%s%d" % (n, l) for n in ("kg", "vg", "bg") for l in range(L))
EQ_CASES = [("GUDCD", "rows", "timing"), ("GUDCD", "rowsx", "timing"), ("GUDCD", "where", "timing"),
            ("GUDKCD", "rows", "timing"), ("GUDKCD", "rowsx", "timing"), ("GUDKCD", "where", "timing"),
            ("GUDCD", "rows", "check"), ("GUDKCD", "rows", "check")]


@pytest.mark.parametrize("arm,mode,scen", EQ_CASES)
def test_st1_equal(arm, mode, scen):
    STC = _mod("r2_st1_commit")
    base = _plain(arm, scen)
    ev = _m(arm, mode, None, scen)
    rd, rb = ev["rd"], base["rd"]
    tb = rb["tent"]
    assert tb["crossed"] > 0, "precondition: a committed block filled (the write-back and phase 2 ran)"
    assert sum(rb["acc_hist"][1:]) > 0, "precondition: a group accepted n >= 1 drafts (a commit of m >= 2 rows)"
    skip_t = KV_KEYS if mode == "rows" else ()
    bad = TS._diff(base, ev, skip=("st1", "compile"), tensors_skip=skip_t)
    assert bad == [], bad[:10]                                         # THE CLAIM: the same run
    if mode == "rows":                                                 # the KV tables outside uncommitted tail rows
        a, b = _kv_masked(base["runner"]), _kv_masked(ev["runner"])
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert STC.tail_checksum(ev["runner"].T) == STC.tail_checksum(base["runner"].T)
    assert all(torch.equal(x, y) for x, y in zip(_host_committed(base["runner"]), _host_committed(ev["runner"])))
    s1 = rd["st1"]
    n = rd["tent"]["calls"]["commit"]
    assert s1["mode"] == mode and s1["warm"] == 1 and s1["captures"] == 1 and s1["commits"] == n and n >= 3
    assert s1["replays"] == n - 1 and s1["overflow"] == 0 and s1["J"] == X.K0 + 2, s1
    _ev("ST1 EQUAL %s %s %s: rounds %d, commits %d, crossed %d, appended %d; st1 %s" % (
        arm, mode, scen, rd["rounds"], n, tb["crossed"], tb["appended"], s1))


@pytest.mark.parametrize("sneg", ["stale", "no_phase2", "narrow"])
def test_st1_controls(sneg):
    base = _plain("GUDCD")
    tb = base["rd"]["tent"]
    assert tb["crossed"] > 0 and sum(base["rd"]["acc_hist"][1:]) > 0, "precondition (as test_st1_equal)"
    try:
        ev = _m("GUDCD", "rows", sneg)
    except Exception as e:                                             # a control that breaks the run is caught too
        _ev("ST1 CONTROL %s: the run raised %s: %s" % (sneg, type(e).__name__, str(e)[:200]))
        return
    bad = TS._diff(base, ev, skip=("st1", "compile"), tensors_skip=KV_KEYS)
    a, b = _kv_masked(base["runner"]), _kv_masked(ev["runner"])
    kv_same = all(torch.equal(x, y) for x, y in zip(a, b))
    assert bad != [] or not kv_same, "the control %s changed nothing (a no-op plant)" % sneg
    if sneg == "narrow":
        assert ev["rd"]["st1"]["overflow"] > 0 and ev["rd"]["st1"]["J"] == 1
    _ev("ST1 CONTROL %s: differences %d (first %s); KV equal %s; st1 %s" % (sneg, len(bad), bad[:4], kv_same,
                                                                           ev["rd"]["st1"]))


# ------------------------------------------------------------------ ST2: the twin section in one advance
@pytest.mark.parametrize("arm,scen", [("GUDCD", "timing"), ("GUDKCD", "timing"), ("GUDCD", "check"),
                                      ("GUDKCD", "check")])
def test_st2_equal(arm, scen):
    m = _m(arm, "rows", None, scen)                                    # the M arm: the reference of the MT arm
    ev = _m(arm, "rows", None, scen, twin=True)
    base = _plain(arm, scen)
    tb = base["rd"]["tent"]
    assert tb["replays"] > 0 and tb["calls"]["replay_calls"] > 0, "precondition: deferred replays ran"
    assert tb["calls"]["joint_passes"] > 0
    bad = TS._diff(m, ev, skip=("st1", "st2", "compile"))
    assert bad == [], bad[:10]                                         # THE CLAIM: the MT run = the M run, every bit
    bad_d = TS._diff(base, ev, skip=("st1", "st2", "compile"), tensors_skip=KV_KEYS)
    assert bad_d == [], bad_d[:10]                                     # and = the D run (KV outside uncommitted rows)
    assert all(torch.equal(x, y) for x, y in zip(_kv_masked(base["runner"]), _kv_masked(ev["runner"])))
    s2, rd = ev["rd"]["st2"], ev["rd"]
    jp = rd["tent"]["calls"]["joint_passes"]
    assert s2["passes"] == jp and s2["layers"] == jp * L and s2["fallback"] == 0 and s2["neg"] is None, s2
    assert rd["tent"]["replays"] == tb["replays"] and rd["tent"]["calls"]["replay_calls"] == tb["calls"]["replay_calls"]
    _ev("ST2 EQUAL %s %s: joint passes %d, replays %d; st2 %s" % (arm, scen, jp, tb["replays"], s2))


@pytest.mark.parametrize("tneg2", ["no_replay_seq", "no_shift"])
def test_st2_controls(tneg2):
    m = _m("GUDCD", "rows")
    try:
        ev = _m("GUDCD", "rows", twin=True, tneg2=tneg2)
    except Exception as e:                                             # a control that breaks the run is caught too
        _ev("ST2 CONTROL %s: the run raised %s: %s" % (tneg2, type(e).__name__, str(e)[:200]))
        return
    bad = TS._diff(m, ev, skip=("st1", "st2", "compile"))
    assert bad != [], "the control %s changed nothing (a no-op plant)" % tneg2
    _ev("ST2 CONTROL %s: differences %d (first %s)" % (tneg2, len(bad), bad[:4]))


def test_st2_pass_plan():
    """The per-pass step sequence on a planted plan: a group request (v 2: rows 0, 1 verify, the replay, row 2 the
    draft row), a final request (3 rows), a fill request (1 row), a done request (none); R 4."""
    STC = _mod("r2_st1_commit")

    class _P:
        pass

    class _T(STC.TwinST2):
        def __init__(self):
            self.B, self.dev, self.bs, self.tneg = 4, torch.device("cpu"), 64, None
            self._st2 = dict(neg=None, passes=0)
            self._st2_cur = None

    t = _T()
    p = _P()
    p.R = 4
    p.act = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)   # (R, B)
    p.replay = torch.zeros((4, 4), dtype=torch.bool)
    p.replay[2, 0] = True
    p.replay_rows = (2,)
    p.deferred = torch.tensor([130, 7, 9, 11])
    pp = t._st2_pass(p)
    assert pp["n"].tolist() == [4, 3, 1, 0]
    assert pp["idx"][0].tolist() == [0, 1, 4, 2, 3]                    # rows 0, 1, the replay (slot R), row 2
    assert pp["idx"][1, :3].tolist() == [0, 1, 2]
    assert pp["S"][:, 0].tolist() == [0, 1, 3, 4] and pp["S"][:, 1].tolist() == [0, 1, 2, 3]
    assert pp["i"].tolist() == [2, 7, 9, 11] and int(pp["nrep"]) == 1
    assert t._st2_pass(p) is pp                                        # cached per plan
    C, STC = _mod("r2_st1_cell"), _mod("r2_st1_commit")
    base = ["--out", "/nonexistent/x", "--workload", "/nonexistent/w"]
    a = C.parse_args(base + ["--arms", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDM"])
    assert a.st1_commit == "rows" and a.arms[-1] == "L4PTFDGUDCDM"
    for bad in ("L4PTFD,L4PTFDGUDC,L4PTFDGUDCDM", "L4PTFDGUDCDM,L4PTFD", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDX"):
        with pytest.raises(SystemExit):
            C.parse_args(base + ["--arms", bad])
    with pytest.raises(SystemExit):
        C.parse_args(base + ["--st1-commit", "fast"])
    assert C.ARMS[-4:] == ("L4PTFDGUDCDM", "L4PTFDGUDKWCDM", "L4PTFDGUDCDMT", "L4PTFDGUDKWCDMT")
    with pytest.raises(SystemExit):                                    # an MT arm needs its M arm
        C.parse_args(base + ["--arms", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDMT"])
    assert set(STC.MODES) == {"rows", "rowsx", "where"}
    # the gates on the CPU evidence (the GPU cell's inputs): the D arm's gates under the M name + PEQ, DCOUNTS, TAIL-EQ,
    # CGRAPH; every MUST passes
    for arm, name, cname in (("GUDCD", "L4PTFDGUDCDM", "L4PTFDGUDC"), ("GUDKCD", "L4PTFDGUDKWCDM", "L4PTFDGUDKWC")):
        x = C.ST1[name]
        ev, d, cc, dd = _m(arm, "rows"), _plain(arm), _plain({"GUDCD": "GUDC", "GUDKCD": "GUDKC"}[arm]), _plain("D")

        def cpu(e):
            out = X.cell_inputs(e, C)
            out["tails"] = STC.tail_checksum(e["runner"].T)
            return out
        keepd = {"L4PTFD": cpu(dd), x: cpu(d)}
        if arm == "GUDKCD":
            keepd[cname] = cpu(cc)
        rd = ev["rd"]
        rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
        me = cpu(ev)
        gs = C.arm_gates_st1(name, rd, rec, me, keepd, False, 8, 10, N_RUN, L, 1.0)
        names = [g[0] for g in gs]
        assert all(n.startswith(name + "-") for n in names), names
        for want in ("-PEQ", "-DCOUNTS", "-TAIL-EQ", "-CGRAPH", "-EQF"):
            assert name + want in names, (want, names)
        assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
        # the plants: each gate fails on a planted difference
        bad_out = dict(me, out=me["out"].clone())
        bad_out["out"][0, 5] += 1                                      # a committed token (column 5 < N_RUN)
        assert not C.peq_gate(name, bad_out, keepd[x], x, N_RUN)[2]
        assert not C.peq_gate(name, dict(me, state=list(me["state"][:-1]) + [me["state"][-1] + 1]), keepd[x], x,
                              N_RUN)[2]
        assert not C.tail_gate(name, [t + 1 for t in me["tails"]], keepd[x]["tails"], x)[2]
        assert not C.tail_gate(name, me["tails"], None, x)[2]
        s1 = dict(rd["st1"])
        n = rd["tent"]["calls"]["commit"]
        assert C.cgraph_gate(name, s1, n)[2]
        for plant in (dict(overflow=1), dict(captures=2), dict(replays=s1["replays"] - 1), dict(warm=0)):
            assert not C.cgraph_gate(name, dict(s1, **plant), n)[2], plant
        assert not C.cgraph_gate(name, s1, n + 1)[2]
        # the MT arm: the D arm's gates under the MT name, PEQ / DCOUNTS / TAIL-EQ against the M arm, CGRAPH, TWIN
        tname = name + "T"
        et = _m(arm, "rows", twin=True)
        keepd[name] = me
        rdt = et["rd"]
        rect = dict(graph=rdt["graph"], dgraph=rdt["dgraph"], gpu_peak_alloc_GB=1.0,
                    mem_arena=dict(prefill_peak_GB=1.0))
        gt = C.arm_gates_st1(tname, rdt, rect, cpu(et), keepd, False, 8, 10, N_RUN, L, 1.0)
        namest = [g[0] for g in gt]
        assert all(n_.startswith(tname + "-") for n_ in namest), namest
        for want in ("-PEQ", "-DCOUNTS", "-TAIL-EQ", "-CGRAPH", "-TWIN", "-EQF"):
            assert tname + want in namest, (want, namest)
        assert all(g[2] for g in gt if g[1]), [g for g in gt if g[1] and not g[2]]
        jp = rdt["tent"]["calls"]["joint_passes"]
        assert C.twin_gate(tname, rdt["st2"], jp, L)[2]
        for plant in (dict(fallback=1), dict(layers=rdt["st2"]["layers"] - 1), dict(passes=jp + 1)):
            assert not C.twin_gate(tname, dict(rdt["st2"], **plant), jp, L)[2], plant
        assert not C.twin_gate(tname, None, jp, L)[2]
    # the predictions: REPORT while nothing is registered, NOT_RUN without an M arm
    assert C.verdicts({}, 216) == [("ST1-RUN", "NOT_RUN", "no M arm ran")]
    sp = {"L4PTFDGUDCD": dict(rates=dict(median_tok_s=100.0, round_ms_median=10.0),
                              phases_median=dict(commit=6.0, joint=5.0, draft=4.0), peak_with_prefill_GB=80.0),
          "L4PTFDGUDCDM": dict(rates=dict(median_tok_s=101.0, round_ms_median=9.9),
                               phases_median=dict(commit=4.5, joint=5.0, draft=4.0), peak_with_prefill_GB=80.0)}
    v = {p[0]: p for p in C.verdicts(sp, 216)}
    P = C.PRED_ST1_BY_B[216]

    def want(key, val):                                                # REPORT while the key is not registered
        b = P.get(key)
        return "REPORT" if b is None else ("CONFIRMED" if b[0] <= val <= b[1] else "REFUTED")
    assert v["ST1-COMMIT-SAVING"][1] == want("ST1_commit_saving", 1.5) and "1.5000" in v["ST1-COMMIT-SAVING"][2]
    assert v["ST1-VS-REF"][1] == want("ST1_vs_ref", 1.01)


def test_st1_registration_state():
    """The registration and the grants: None (the builder's state) or a `date -u` stamp, as the step script's greps
    see them in the source."""
    C = _mod("r2_st1_cell")
    src = (EFF / "r2_st1_cell.py").read_text()
    if C.PRED_STAMP_ST1 is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_ST1_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_ST1)
    assert set(C.PRED_ST1_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_ST1_BY_B.values())
    assert ("\nPRED_STAMP_ST1 = None\n" in src) == (C.PRED_STAMP_ST1 is None)   # the step script's grep sees it
    for cell in ("st1m", "st1a", "st1b"):                               # the grants: None or a date -u stamp
        g = getattr(C, "GRANT_%s" % cell)
        assert g is None or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", g)
        assert (("\nGRANT_%s = None\n" % cell) in src) == (g is None)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_st1_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


CELLS = (("st1m", 10, 8, 72, 1, 1, 35, "100G",
          "L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,"
          "L4PTFDGUDKWCDMT"),
         ("st1a", 65, 216, 512, 0, 0, 160, "240G", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDM,L4PTFDGUDCDMT"),
         ("st1b", 10, 318, 512, 0, 0, 240, "340G", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT"))


def test_step_script_st1():
    p = EFF / "r2_st1_step.sh"
    src = p.read_text()
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    assert src.rstrip().endswith('if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi')
    for word in ("python", "perl", "awk", "sed", "sha256sum", "hash-object"):   # no interpreter / hash on login
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % re.escape(word), x)
                 and "/venv/nosa/bin/python" not in x and "/usr/bin/sha256sum" not in x and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert 'cmp -s -- "$0" <(git -C "$NOSI" show "$NSHA:$SELF")' in src                 # the bytes, not a hash
    assert src.count("/usr/bin/sha256sum -- ") == 2 and "exit 6" in src                 # the child hashes (GPU node)
    assert 'ulimit -f "$KB"' in src and 'du -sb -- "$t"' in src and "exit 7" in src     # the output limit
    assert "NOSI=%s " % STEP_ROOT in src and "FORK_BLOB=%s" % FORK_BLOB in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/st1_jit " in src and "CELLPY=%sr2_st1_cell.py" % E_ in src
    for f in ("r2_st1_commit.py", "r2_i4dc_cell.py", "r2_i4dc_compile.py", "r2_i4h_graph.py", "r2_i4g_graph.py",
              "r2_writeback.py", "r2_twin_fused.py", "spec_tentative.py", "r2_positional_twin.py"):
        assert f in src, f
    assert "OUTROOT/sp_active_$JOB" in src and "granted \"$CELL\" <(git -C \"$NOSI\" show \"$NSHA:$CELLPY\")" in src
    assert "grep -q '^PRED_STAMP_ST1 = None$' && REGISTERED=0" in src                   # the registration stop
    C = _mod("r2_st1_cell")
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:   # every cell's arguments parse
        a = C.parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                          "--pool", str(pool), "--arms", arms, "--check-every", str(check),
                          "--require-stress", str(stress)])
        assert a.arms == arms.split(",")
        assert ("%s) POOL=%d BATCH=%d STEPS=%d" % (cell, pool, batch, steps)) in src and ("STEP_MIN=%d MEM=%s" % (
            smin, mem)) in src, cell
        assert "ARMS=" + arms in src, cell
    for bad in ("i4dca", "st1", "st1c"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    if str(ROOT) != STEP_ROOT or _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    sha = _git("rev-parse", "HEAD").stdout.strip()
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if getattr(C, "GRANT_%s" % cell) is None:
            assert r.returncode == 1 and "is not granted" in r.stderr, (cell, r.returncode, r.stderr[-300:])
        assert ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3" % (cell, 63 + pool, pool, batch, arms, steps)
                in r.stdout), (cell, r.stdout[-600:])
    _ev("ST1 DRYRUN: 3 cells printed their step command; grants %s" % ({c[0]: getattr(C, "GRANT_%s" % c[0]) for c in
                                                                       CELLS}))
