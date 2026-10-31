"""Rung INT CPU tests (benchmarks/Efficiency/r2_int_arms.py, r2_int_cell.py, r2_int_step.sh; docs/NOTES_int.md):
  Gate 0      the changed files = INT's + the three rungs' (J1, ST1, BF1) + one INT block in each older Gate-0 test and in
              test_j1 / test_st1 / test_bf1; nosi and every older module are cd0f0c5's; no kernel source; no hand-written
              kernel; r2_int_arms imports torch and the three rungs' modules only; nobody else names the INT modules
  COPY        IntLayer._layer = r2_st1_commit.TwinST2._layer except the 2 lines marked "# INT"
  CLASSES     the owners of _layer / commit, J3 only in the C 128 arm, BF1 only in the C 73 arm, F3 never with BF1
  EQUAL       both INT arms on the CPU world (i4dc_world, the scenarios timing and check) = their D arm and L4PTFD bit for
              bit (outside the records and the uncommitted tail rows of ST1's row commit), every component ran (J1 calls
              = ST2 layers = joint passes x L, F2 in the draft graphs, F6, F3 / BF1, the commit graph), and every MUST
              gate of the cell passes
  CONTROLS    "stacked" (no merged _layer: the run is silent, J1-COMPILE / J1-PEQ / INT-CLASSES / INT-LAYER catch it),
              "nolock" (BF1's thread worker: INT-LOCK catches the unwrapped ST1 backend; the wrapped one adds exactly the
              commit capture to BF1's locked captures)
  CELL        the private copy's tables, runner choice / class, the arm rules, the verdicts (registration-aware), main's
              refusal, the registration state, the step script (no interpreter or hash on login, the cells, DRYRUN)
Run on amd (srun) by the lead: pytest -q benchmarks/Efficiency/tests_cpu/test_int.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
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
BASE_SHA = "cd0f0c59c1f1cac8a3ef05f6c3abcc0a2079d926"          # nosi-r2-recon7: RECON7 X1 notes (the three rungs' base)
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-int"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
INT_NEW = (E_ + "r2_int_arms.py", E_ + "r2_int_cell.py", E_ + "r2_int_step.sh", T_ + "test_int.py", "docs/NOTES_int.md")
RUNGS = (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", E_ + "r2_j1_forkbench.py", T_ + "test_j1.py",
         "docs/NOTES_j1.md", E_ + "r2_st1_commit.py", E_ + "r2_st1_cell.py", E_ + "r2_st1_step.sh", T_ + "test_st1.py",
         E_ + "r2_bf1_fetch.py", E_ + "r2_bf1_cell.py", E_ + "r2_bf1_step.sh", T_ + "test_bf1.py")
OLD_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u", "i4h",
                                                    "i4s", "i4cc", "i4dc", "recon7"))
RUNG_TESTS = dict(j1="J1_LATER", st1="ST1_LATER", bf1="BF1_LATER")
ARMS_IMPORTS = {"torch", "r2_bf1_fetch", "r2_j1_select", "r2_st1_commit"}
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
N_RUN = 48                             # committed tokens per request (as test_st1 / test_bf1: blocks fill, drafts accept)


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
TJ = _load("test_j1")                  # its module code registers the J1 helpers' capture stand-ins (MUT)
TST = _load("test_st1")
TB = _load("test_bf1")
L = X.L


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _mod(name):
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import importlib
    return importlib.import_module(name)


# ------------------------------------------------------------------ GATE 0
def test_int_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    allowed = set(INT_NEW) | set(RUNGS) | set(OLD_DECL)
    assert changed <= allowed, sorted(changed - allowed)                                    # (a)
    assert all((ROOT / f).is_file() for f in INT_NEW + RUNGS), [f for f in INT_NEW + RUNGS if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: cd0f0c5's
    for f in OLD_DECL:                                                    # (c) one INT block each, only added lines
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        assert set(old) == set(new) and all(old[k] == new[k] for k in old), (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert body and all(x.startswith("+") for x in body), (f, body[:5])
        assert any("r2_int_cell.py" in x for x in body) and any("# INT" in x for x in body), (f, body[:5])
    for rung, later in RUNG_TESTS.items():                                # (c') the three rungs' Gate-0 tests
        text = (ROOT / (T_ + "test_%s.py" % rung)).read_text()
        assert '%s = ("r2_int_arms.py", "r2_int_cell.py")' % later in text and "p.name in %s:" % later in text, rung
        assert "r2_int_cell.py" in text and "test_int.py" in text, rung
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    for name in ("r2_int_arms.py", "r2_int_cell.py"):                     # (d) NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    tree = ast.parse((EFF / "r2_int_arms.py").read_text())               # (e) the classes' imports
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == ARMS_IMPORTS, mods
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (f) nobody else names them
        if p.name.startswith("r2_int_"):
            continue
        assert "r2_int" not in p.read_text(), p.name


# ------------------------------------------------------------------ the merged _layer and the classes
def test_int_layer_copy():
    INTA = _mod("r2_int_arms")
    STC = _mod("r2_st1_commit")
    mine, orig = TJ._dedented(INTA.IntLayer._layer), TJ._dedented(STC.TwinST2._layer)
    assert len(mine) == len(orig), (len(mine), len(orig))
    diff = [(a, b) for a, b in zip(mine, orig) if a != b]
    assert len(diff) == 2 and all("# INT" in a for a, _ in diff), diff
    assert diff[0][1].strip() == "VR = self._st2_vr" and diff[0][0].strip().startswith("VR = None"), diff[0]
    assert "VR._select_rows(plan.vs, layer, q, twin)" in diff[1][1], diff[1]
    assert "self._j1_select(plan.vs, layer, q, twin, l)" in diff[1][0], diff[1]
    assert sum("# INT" in x for x in mine) == 2
    _ev("INT COPY: IntLayer._layer = TwinST2._layer except %d marked lines" % len(diff))


def test_int_classes():
    INTA = _mod("r2_int_arms")
    for arm in INTA.ARMS_INT:
        X.CC.configure()
        model, c, S, hz, tok0 = X.world(2, 8, 0)
        run = _make(arm, "timing")(model, c, S, hz, tok0)
        cls = type(run.T)
        names = [k.__name__ for k in cls.__mro__]
        assert INTA.owner(cls, "_layer") == "IntLayer" and INTA.owner(cls, "commit") == "CommitST1", names
        assert ("_J3Mix" in names) == (arm == "L4PTFDGUDCDI"), names                       # F3: the C 128 arm only
        assert ("TentStateBF" in names) == (arm == "L4PTFDGUDKWCDI"), names                # BF1: the C 73 arm only
        assert "_J1Mix" in names and "TwinST2" in names and names.index("IntLayer") < names.index("TwinST2")
        assert run._arm_name() == arm and run.T._st2_base.__qualname__.endswith("_J1Mix._layer")
        rec = run.int_record()
        assert rec["layer_owner"] == "IntLayer" and rec["j3"] == (arm == "L4PTFDGUDCDI") and rec["ineg"] is None
        s = _make(arm, "timing", ineg="stacked")(*X.world(2, 8, 0)[:5])
        assert INTA.owner(type(s.T), "_layer") == "TwinST2"                                # the control: stacked
        assert run.T._st2_vr is X.STF.VR and s.T._st2_vr is X.STF.VR      # ST2's VR: verify_rows, on a J1 base too
    with pytest.raises(INTA.IntError):
        INTA.int_runner(X.STF, "L4PTFDGUDCDI", ineg="nope")
    with pytest.raises(INTA.IntError):
        INTA.base_runner(X.STF, "L4PTFDGUDCD")


# ------------------------------------------------------------------ the arms on the CPU world
INT_ARMS = dict(C128="L4PTFDGUDCDI", C73="L4PTFDGUDKWCDI")
D_OF = dict(C128="GUDCD", C73="GUDKCD")
C_OF = dict(C128="GUDC", C73="GUDKC")


def _make(arm, scen, ineg=None, worker="inline", check=None, kcheck=None, late="drop"):
    """i4dc_world.d_builder with the INT runner (the cell's runner arguments): FakeGraphsJ3 for the F3 arm, test_st1's
    FakeCommitGraphs for the commit, BF1 with test_bf1's CPU options for the keep arm."""
    INTA = _mod("r2_int_arms")
    s = X.SCEN[scen]
    check = s["check"] if check is None else check
    W, WB, Z, I3, KA, RE, RW, SR, BC = X.W, X.WB, X.Z, X.I3, X.KA, X.RE, X.RW, X.SR, X.BC
    Runner = INTA.int_runner(X.STF, arm, "rows", ineg)
    keep = arm in INTA.KEEP_INT

    def build(model, cache, S, horizon, tok0, K=X.K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(X.TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm="natural", extra=arena.X, attn=WB.fake_attention_u,
                          journal=RW.CAP, redirect=True, record=record, probe_every=s["probe_every"])
        kw = dict(fork=Z.fake_fork, t0=tok0, hooks=None, check_every=check_every, union_check=check,
                  union_stats=s["stats"], phase_marks=s["phase_marks"], dneg=None, hneg=None, gneg=None, sneg=None,
                  dgneg=None, cc_backend="inductor", ccneg=None, cdneg=None, j1neg=None, j1draft=True)
        if keep:
            kw.update(keep=True, kwin=True, kcheck=bool(check) if kcheck is None else kcheck, kneg=None, kslots=None,
                      cneg=None, bf=dict(mode="ce", worker=worker, acct=True, bneg=None, n_cap=4096, ring=4,
                                         packer="torch", gf=None, threads=2, late=late))
        be = X.FakeGraphsDCD() if keep else TJ.FakeGraphsJ3()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, st1_backend=TST.FakeCommitGraphs(), **kw)
        be.T = run.T
        return run
    return build


def _builder(arm, scen, ineg=None, worker="inline", late="drop"):
    X.CC.configure()
    s = X.SCEN[scen]
    model, c, S, hz, tok0 = X.world(8, N_RUN, 0)
    return _make(arm, scen, ineg=ineg, worker=worker, late=late)(model, c, S, hz, tok0, check_every=s["check_every"])


def run_int(arm, scen="timing", ineg=None, worker="inline", late="drop"):
    s = X.SCEN[scen]
    run = _builder(arm, scen, ineg=ineg, worker=worker, late=late)
    if s["fa"]:
        run.vs.fa_check = {}
    run.run(N_RUN)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N_RUN, B=8, scen=scen, arm=arm)


@functools.lru_cache(maxsize=None)
def _plain(arm, scen="timing"):
    return X.run_arm(arm, scen, B=8, N=N_RUN)


_I = {}


def _int(key, scen="timing", ineg=None, worker="inline", late="drop"):
    k = (key, scen, ineg, worker, late)
    if k not in _I:
        _plain("D", scen)
        _plain(C_OF[key], scen)                                        # the C arm first: the joint helpers
        _plain(D_OF[key], scen)                                        # the D arm: the draft helpers, the reference
        _I[k] = run_int(INT_ARMS[key], scen, ineg=ineg, worker=worker, late=late)
    return _I[k]


_F = {}


def _int_full(key, scen="timing"):
    """The cell's INT arm on the CPU world: the INT runner + ST3 (the keep arm) + ST4, attached in the cell's order
    (r2_int_cell.runner_class: st3_runner then st4_runner over int_runner; st*_attach = their __init__ step)."""
    if (key, scen) not in _F:
        STC = _mod("r2_st1_commit")
        _int(key, scen)                                                # its C / D arms first (test_int's order)
        run = _builder(INT_ARMS[key], scen)
        if key == "C73":
            STC.st3_attach(run)
        STC.st4_attach(run)
        if X.SCEN[scen]["fa"]:
            run.vs.fa_check = {}
        run.run(N_RUN)
        rd = run.read()
        if key == "C73":
            rd["st3"] = run.T.st3_report()
        rd["st4"] = run.T.st4_report()
        _F[(key, scen)] = dict(runner=run, rd=rd, N=N_RUN, B=8, scen=scen, arm=INT_ARMS[key])
    return _F[(key, scen)]


def _cell():
    C = _mod("r2_int_cell")
    return C, C.configure()


def _cpu(ev, D):
    STC = _mod("r2_st1_commit")
    c = X.cell_inputs(ev, C=D)
    c["tails"] = STC.tail_checksum(ev["runner"].T)
    return c


def _gates(key, scen, ev):
    C, D = _cell()
    J1C = _mod("r2_j1_cell")
    J1C.SH_J1["emitted"] = True                                        # J1-SHEQ: test_j1's test_sh_j1
    keepd = {"L4PTFD": _cpu(_plain("D", scen), D)}
    keepd[X.DARMS[D_OF[key]]] = _cpu(_plain(D_OF[key], scen), D)
    if key == "C73":
        keepd["L4PTFDGUDKWC"] = _cpu(_plain("GUDKC", scen), D)
    rd = ev["rd"]
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    return D.arm_gates(INT_ARMS[key], dict(rd, tails=_cpu(ev, D)["tails"]), rec, _cpu(ev, D), keepd, scen == "check", 8,
                       10, N_RUN, L, 1.0), rec


SKIP_REC = ("compile", "j1", "j3", "st1", "st2", "int")
BF_REC = ("keep", "round_blocks", "bf")
BF_T = ("kid", "kage", "blocks")


@pytest.mark.parametrize("key,scen", [("C128", "timing"), ("C128", "check"), ("C73", "timing"), ("C73", "check")])
def test_int_equal(key, scen):
    ev = _int(key, scen)
    d, ref = _plain("D", scen), _plain(D_OF[key], scen)
    rd = ev["rd"]
    jp = rd["tent"]["calls"]["joint_passes"]
    tb = ref["rd"]["tent"]
    assert tb["crossed"] > 0 and sum(ref["rd"]["acc_hist"][1:]) > 0 and tb["replays"] > 0, "preconditions (L34)"
    keep = key == "C73"
    skip = SKIP_REC + (BF_REC if keep else ())
    ts = TST.KV_KEYS + (BF_T if keep else ())
    bad = TS._diff(ref, ev, skip=skip, tensors_skip=ts)
    assert bad == [], bad[:10]                                         # THE CLAIM: its D arm's run
    bad = TS._diff(d, ev, skip=skip + ("hoist",) + (("keep", "round_blocks") if keep else ()), tensors_skip=ts)
    assert bad == [], bad[:10]                                         # and L4PTFD's
    assert all(torch.equal(a, b) for a, b in zip(TST._kv_masked(ref["runner"]), TST._kv_masked(ev["runner"])))
    STC = _mod("r2_st1_commit")
    assert STC.tail_checksum(ev["runner"].T) == STC.tail_checksum(ref["runner"].T)
    j, s1, s2 = rd["j1"], rd["st1"], rd["st2"]
    assert jp > 0 and j["calls"] == jp * L and s2["layers"] == jp * L and s2["fallback"] == 0      # F1 and ST2: both
    assert j["peq_layers"] > 0 and j["peq_bad"] == 0 and j["f6"] and j["f6calls"] == rd["compile"]["sections"] > 0
    assert j["dwarm"] == 1 and j["dlayers"] == rd["dgraph"]["capture_passes"] * L > 0 and j["dpeq_bad"] == 0   # F2
    n = rd["tent"]["calls"]["commit"]
    assert s1["warm"] == 1 and s1["captures"] == 1 and s1["replays"] == n - 1 and s1["overflow"] == 0   # ST1
    if keep:
        bf = rd["bf"]
        assert bf["committed"] > 0 and bf["used_store"] + bf["used_win"] > 0 and bf["error"] is None, bf   # BF1
        assert "j3" not in rd                                          # F3 is not in the BF1 arm
    else:
        assert rd["j3"]["joint_captures"] > 0 and rd["j3"]["draft_layers"] > 0, rd["j3"]               # F3
    evg = _int_full(key, scen)                                         # the cell's INT arm: + ST4 (+ ST3 in the keep arm)
    bad = TS._diff(ev, evg, skip=SKIP_REC + ("st3", "st4", "bf"))
    assert bad == [], bad[:10]                                         # = the INT run, every bit (BF1's timers aside)
    if keep:
        assert all(evg["rd"]["bf"][k] == rd["bf"][k] for k in ("committed", "used_store", "used_win"))
    gs, rec = _gates(key, scen, evg)
    names = {g[0][len(INT_ARMS[key]) + 1:] for g in gs if g[0].startswith(INT_ARMS[key] + "-")}
    want = {"MEM", "GRAPH", "DGRAPH", "HOIST", "COMPILE", "EQF", "J1-PEQ", "J1-COMPILE", "J1-DRAFT", "PEQ", "TAIL-EQ",
            "CGRAPH", "TWIN", "INT-CLASSES", "INT-LAYER", "COUNTS", "DRESID"}
    want |= {"KEEP", "LISTED", "BF", "INT-LOCK", "KFILL"} | ({"CAP"} if scen == "check" else set()) if keep else \
        {"DCOUNTS", "J1-RESID"}
    assert names == want, (sorted(names - want), sorted(want - names))
    assert not keep or not ({"KEEP-EQ", "KEEP-REF"} & names)           # BF1 changes them by design (dropped)
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert rec["int"]["layer_owner"] == "IntLayer" and rec["st2"]["layers"] == jp * L
    _ev("INT EQUAL %s %s: rounds %d, joint passes %d, commits %d; j1 calls %d f6 %d dlayers %d; st2 %s; st1 %s; %s; "
        "%d MUST gates pass" % (key, scen, rd["rounds"], jp, n, j["calls"], j["f6calls"], j["dlayers"], s2, s1,
                                ("bf committed %d used %d" % (rd["bf"]["committed"], rd["bf"]["used_store"] +
                                                              rd["bf"]["used_win"])) if keep else ("j3 %s" % rd["j3"]),
                                sum(1 for g in gs if g[1])))


def test_int_stacked():
    """The control "stacked" (no merged _layer): the run is the same (silent), the J1 selection never runs in a joint
    pass; J1-COMPILE, J1-PEQ, INT-CLASSES and INT-LAYER catch it."""
    INTA = _mod("r2_int_arms")
    J1 = _mod("r2_j1_select")
    ref = _plain("GUDCD")
    ev = _int("C128", "timing", ineg="stacked")
    rd = ev["rd"]
    jp = rd["tent"]["calls"]["joint_passes"]
    assert TS._diff(ref, ev, skip=SKIP_REC, tensors_skip=TST.KV_KEYS) == []          # silent: EQF cannot catch it
    assert rd["j1"]["calls"] == 0 and rd["st2"]["layers"] == jp * L > 0              # L34: the plant did act
    g = {x[0].split("-", 1)[1]: x for x in J1.j1_gates("L4PTFDGUDCDI", rd, L) + INTA.int_gates("L4PTFDGUDCDI", rd, L)}
    for n in ("J1-PEQ", "J1-COMPILE", "INT-CLASSES", "INT-LAYER"):
        assert g[n][1] is True and g[n][2] is False, g[n]
    _ev("INT CONTROL stacked: silent run; caught by %s" % [n for n, x in g.items() if not x[2]])


def test_int_lock():
    """BF1's thread worker: the INT arm wraps the ST1 commit backend with BF1's capture lock (exactly one more locked
    capture: the commit's); the control "nolock" leaves it unwrapped and INT-LOCK catches it."""
    INTA = _mod("r2_int_arms")
    a = _int("C73", "timing", worker="thread")
    b = _int("C73", "timing", ineg="nolock", worker="thread")
    ra, rb = a["rd"], b["rd"]
    assert ra["int"]["lock_applies"] and ra["int"]["st1_locked"] and not rb["int"]["st1_locked"]
    assert ra["bf"]["error"] is None and rb["bf"]["error"] is None
    assert ra["bf"]["captures_locked"] - rb["bf"]["captures_locked"] == 1, (ra["bf"]["captures_locked"],
                                                                            rb["bf"]["captures_locked"])
    ga = {x[0]: x for x in INTA.int_gates("L4PTFDGUDKWCDI", ra, L)}
    gb = {x[0]: x for x in INTA.int_gates("L4PTFDGUDKWCDI", rb, L)}
    assert ga["L4PTFDGUDKWCDI-INT-LOCK"][2] is True and gb["L4PTFDGUDKWCDI-INT-LOCK"][2] is False
    assert TS._diff(_plain("GUDKCD"), a, skip=SKIP_REC + BF_REC, tensors_skip=TST.KV_KEYS + BF_T) == []
    _ev("INT LOCK: captures under BF1's lock %d (wrapped) vs %d (nolock)" % (ra["bf"]["captures_locked"],
                                                                           rb["bf"]["captures_locked"]))


def test_int_defer():
    """The C 73 INT arm with BF1's production late mode ("defer": a job not issued at the joint pass's start commits at
    ITS layer, in the joint residency) and the thread worker: = the D arm's run (the BF1 record aside), no worker error,
    the INT gates pass (INT-LOCK: the commit backend under BF1's lock)."""
    INTA = _mod("r2_int_arms")
    ev = _int("C73", "timing", worker="thread", late="defer")
    rd = ev["rd"]
    assert rd["bf"]["late_mode"] == "defer" and rd["bf"]["error"] is None and rd["bf"]["committed"] > 0, rd["bf"]
    assert TS._diff(_plain("GUDKCD"), ev, skip=SKIP_REC + BF_REC, tensors_skip=TST.KV_KEYS + BF_T) == []
    assert all(g[2] for g in INTA.int_gates("L4PTFDGUDKWCDI", rd, L))
    _ev("INT DEFER: bf late %s, issued %s, committed %s, used %s" % (rd["bf"].get("late"), rd["bf"].get("issued"),
                                                                   rd["bf"]["committed"], rd["bf"]["used_store"] +
                                                                   rd["bf"]["used_win"]))


def test_int_gate_plants():
    INTA = _mod("r2_int_arms")
    rd = dict(tent=dict(calls=dict(joint_passes=3)), j1=dict(calls=3 * L), st2=dict(layers=3 * L, fallback=0),
              bf=dict(captures_locked=5),
              int=dict(arm="L4PTFDGUDKWCDI", layer_owner="IntLayer", commit_owner="CommitST1", j3=False, bf=True,
                       ineg=None, lock_applies=True, st1_locked=True))
    assert all(g[2] for g in INTA.int_gates("L4PTFDGUDKWCDI", rd, L))
    for path, val in ((("int", "layer_owner"), "TwinST2"), (("int", "commit_owner"), "TentState"), (("int", "j3"), True),
                      (("int", "bf"), False), (("int", "ineg"), "stacked"), (("int", "st1_locked"), False),
                      (("j1", "calls"), 3 * L - 1), (("st2", "layers"), 0), (("st2", "fallback"), 1)):
        r2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rd.items()}
        r2[path[0]][path[1]] = val
        assert not all(g[2] for g in INTA.int_gates("L4PTFDGUDKWCDI", r2, L)), (path, val)
    r3 = dict(rd, int=dict(rd["int"], arm="L4PTFDGUDCDI", j3=True, bf=False))
    assert [g[0] for g in INTA.int_gates("L4PTFDGUDCDI", r3, L)] == ["L4PTFDGUDCDI-INT-CLASSES", "L4PTFDGUDCDI-INT-LAYER"]
    assert all(g[2] for g in INTA.int_gates("L4PTFDGUDCDI", r3, L))


# ------------------------------------------------------------------ CELL
def test_int_cell_tables(monkeypatch):
    C, D = _cell()
    I4DC = _mod("r2_i4dc_cell")
    INTA = _mod("r2_int_arms")
    assert D is not I4DC and D.__name__ == C.DCC_NAME and C.configure() is D
    assert "L4PTFDGUDCDI" not in I4DC.ARMS and D.ARMS == I4DC.ARMS + C.NEW_ARMS
    assert D.ARMS[-2:] == ("L4PTFDGUDCDI", "L4PTFDGUDKWCDI")
    assert D.BASE_D["L4PTFDGUDCDI"] == "L4PTFDGUDCD" and D.BASE_D["L4PTFDGUDKWCDI"] == "L4PTFDGUDKWCD"
    assert D.ROOT_ARM["L4PTFDGUDKWCDI"] == "L4PTFDGUDKW" and "L4PTFDGUDKWCDI" in D.KEEP_ARMS
    assert D.NAMES["L4PTFDGUDCDI"] == tuple(n for n in I4DC.NAMES["L4PTFDGUDCD"] if n != "loc1")   # F6
    k1, kw1 = D.runner_choice("L4PTFDGUDCDI", False)
    k2, kw2 = D.runner_choice("L4PTFDGUDKWCDI", True)
    assert (k1, kw1) == ("SCD", dict(cc_backend="inductor")) and k2 == "SKCD" and kw2["kcheck"] is True
    assert kw2["bf"] == dict(mode="ce", n_cap=2048, ring=8, worker="thread", acct=False, close_wait_ms=0.0,
                             packer="gilfree", threads=14, gf=dict(cap_groups=1024), late="defer", late_wait_ms=250.0)
    BFC = _mod("r2_bf1_cell")                                          # the same keywords as the BF1 cell's
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/tmp/x")
    ab = BFC.parse_args(["--out", "x", "--workload", "w", "--arms", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDB"])
    assert set(BFC.bf_opt(ab, "L4PTFDGUDKWCDB")) == set(kw2["bf"])
    assert D.runner_choice("L4PTFDGUDKWCDX", False)[1]["bf"]["mode"] == "off"
    STC = _mod("r2_st1_commit")
    MTk = STC.st1_runner(X.DCL["L4PTFDGUDKWCD"][1], "rows", twin=True)
    MTc = STC.st1_runner(X.DCL["L4PTFDGUDCD"][1], "rows", twin=True)
    for arm, base in (("L4PTFDGUDCDI", "SCD"), ("L4PTFDGUDKWCDI", "SKCD"), ("L4PTFDGUDKWCDMTK", "SKCD"),
                      ("L4PTFDGUDCDMTR", "SCD"), ("L4PTFDGUDKWCDMTKR", "SKCD")):
        C._CUR["arm"] = arm
        try:
            R = D.runner_class(base, X.STF)                            # R.__mro__[0]: the tails read
        finally:
            C._CUR["arm"] = None
        RI = INTA.int_runner(X.STF, arm, "rows", None) if arm in C.ARMS_INT else None
        want = dict(L4PTFDGUDCDI=[STC.st4_runner(RI), RI] if RI else None,
                    L4PTFDGUDKWCDI=[STC.st4_runner(STC.st3_runner(RI)), STC.st3_runner(RI), RI] if RI else None,
                    L4PTFDGUDKWCDMTK=[STC.st3_runner(MTk), MTk], L4PTFDGUDCDMTR=[STC.st4_runner(MTc), MTc],
                    L4PTFDGUDKWCDMTKR=[STC.st4_runner(STC.st3_runner(MTk)), STC.st3_runner(MTk), MTk])[arm]
        assert list(R.__mro__[1:1 + len(want)]) == want, (arm, R.__mro__[:4])
    assert C.hooks("L4PTFDGUDKWCDI")[-2:] == ("ST3", "ST4") and C.hooks("L4PTFDGUDCDI")[-1] == "ST4"
    assert "ST3" not in C.hooks("L4PTFDGUDCDI")
    assert C.ST_ARMS["L4PTFDGUDKWCDMTK"] == "L4PTFDGUDKWCDMT" and "L4PTFDGUDKWCDMTKR" in D.KEEP_ARMS
    assert C.ST_ARMS["L4PTFDGUDCDMTR"] == "L4PTFDGUDCDMT" and C.ST_ARMS["L4PTFDGUDKWCDMTKR"] == "L4PTFDGUDKWCDMTK"
    assert D.runner_class("SCD", X.STF).__mro__[1] is X.DCL["L4PTFDGUDCD"][1]       # every arm: the tails read
    base = ["--out", "x", "--workload", "w"]
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/tmp/x")
    for arms in ("L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDI",
                 "L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDI,L4PTFDGUDKWCDI",
                 "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDB,"
                 "L4PTFDGUDKWCDI"):
        own, rest = C.parse_own(base + ["--arms", arms])
        a = D.parse_args(rest)
        assert a.arms == arms.split(",") and C.check_arms(a.arms, own, D) is None, arms
    with pytest.raises(SystemExit):
        D.parse_args(base + ["--arms", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCDI"])          # the INT arm needs its D arm
    own, _ = C.parse_own(base)
    assert "needs L4PTFDGUDKWCDM earlier" in C.check_arms(["L4PTFD", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDMT"], own, D)
    assert "needs L4PTFDGUDKWCDMT earlier" in C.check_arms(["L4PTFD", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDM",
                                                            "L4PTFDGUDKWCDMTK"], own, D)
    assert "needs L4PTFDGUDKWCD earlier" in C.check_arms(["L4PTFD", "L4PTFDGUDKWCDB"], own, D)
    assert C.check_arms(["L4PTFD", "L4PTFDGUDKWC", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDI"], own, D) is None
    own2, _ = C.parse_own(base + ["--bf-store", "5"])
    assert "--bf-store 5" in C.check_arms(["L4PTFD"], own2, D)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR")
    assert "TORCH_EXTENSIONS_DIR" in C.check_arms(["L4PTFD", "L4PTFDGUDKWC", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDI"], own, D)
    assert C.check_arms(["L4PTFD", "L4PTFDGUDC", "L4PTFDGUDCD", "L4PTFDGUDCDI"], own, D) is None
    own3, rest3 = C.parse_own(base + ["--sh-j1", "0", "--st1-commit", "rowsx", "--bf-ncap", "7", "--batch", "8"])
    assert own3.sh_j1 == 0 and own3.st1_commit == "rowsx" and own3.bf_ncap == 7 and rest3 == base + ["--batch", "8"]
    assert C.main(["--out", "x", "--workload", "w", "--model", "", "--arms", "L4PTFD"]) == 20   # no model: refused
    assert C.main(base + ["--arms", "L4PTFD,L4PTFDGUDKWCDB"]) == 2                   # an arm rule: refused
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    mk = lambda tok, jt, dr, cm: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt, commit=cm),  # noqa
                                      gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0))
    sp = dict(L4PTFD=mk(400.0, 200.0, 140.0, 6.0), L4PTFDGUDCD=mk(440.0, 186.0, 130.0, 6.0),
              L4PTFDGUDCDI=mk(480.0, 170.0, 124.0, 2.0))
    v = {y[0]: y for y in C.verdicts(sh2, sp, 216, D)}
    if C.PRED_STAMP_INT is None:
        assert all(y[1] == "REPORT" for y in v.values()), v
    assert "16.0000" in v["INT-JOINT-SAVING"][2] and "4.0000" in v["INT-COMMIT-SAVING"][2] and "1.0909" in \
        v["INT-VS-REF"][2], v
    assert C.verdicts(sh2, dict(L4PTFD=sp["L4PTFD"]), 216, D) == [("INT-RUN", "NOT_RUN", "no INT arm ran")]
    v3 = {y[0]: y for y in C.verdicts(sh2, dict(sp, L4PTFDGUDKWCDMT=mk(400.0, 300.0, 140.0, 3.0),
                                                L4PTFDGUDKWCDMTK=mk(410.0, 293.0, 140.0, 3.0)), 318, D)}
    assert "7.0000" in v3["INT-ST3-JOINT-SAVING"][2] and "1.0250" in v3["INT-ST3-VS-MT"][2], v3
    vv = [y[0] for y in D.verdicts(sh2, sp, 216)]
    assert "I4DC-VS-REF" in vv and "INT-VS-REF" in vv and "J1-JOINT-SAVING" not in vv   # J1 verdicts: J1 arms only


def test_int_registration_state():
    C = _mod("r2_int_cell")
    src = (EFF / "r2_int_cell.py").read_text()
    if C.PRED_STAMP_INT is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_INT_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_INT)
    assert set(C.PRED_INT_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS_INT) for x in C.PRED_INT_BY_B.values())
    assert ("\nPRED_STAMP_INT = None\n" in src) == (C.PRED_STAMP_INT is None)   # the step script's grep sees it
    for cell in ("intm", "inta", "intb"):                               # the grants: None or a date -u stamp
        g = getattr(C, "GRANT_%s" % cell)
        assert g is None or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", g)
        assert (("\nGRANT_%s = None\n" % cell) in src) == (g is None)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_int_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


CELLS = (("intm", 10, 8, 72, 1, 1, 4, 1, 0, 55, "100G",
          "L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,"
          "L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDCDMTR,L4PTFDGUDKWCDMTKR,L4PTFDGUDCDI,L4PTFDGUDKWCDI"),
         ("inta", 65, 216, 512, 0, 0, 16, 0, 0, 125, "240G", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI"),
         ("intb", 10, 318, 512, 0, 0, 16, 0, 257, 270, "340G",
          "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDKWCDI"))


def test_step_script_int(monkeypatch):
    src = (EFF / "r2_int_step.sh").read_text()
    assert src.rstrip().endswith('if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi')
    for word in ("python", "perl", "awk", "sed", "sha256sum", "hash-object"):   # no interpreter / hash on login
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % re.escape(word), x)
                 and "/venv/nosa/bin/python" not in x and "/usr/bin/sha256sum" not in x and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/int_jit " in src
    for f in ("r2_int_arms.py", "r2_j1_select.py", "r2_j1_cell.py", "r2_st1_commit.py", "r2_st1_cell.py",
              "r2_bf1_fetch.py", "r2_bf1_cell.py", "gilfree_pack.py", "r2_transport_gilfree_v3.py", "r2_i4dc_cell.py",
              "r2_i4dc_compile.py", "max_pooling_per_request.py", "verify_rows.py", "spec_tentative.py"):
        assert f in src, f
    assert "OUTROOT/sp_active_$JOB" in src and "granted \"$CELL\" <(git -C \"$NOSI\" show \"$NSHA:$CELLPY\")" in src
    assert "--env TORCH_EXTENSIONS_DIR=\"$JIT/torch_ext\"" in src and "--bf-packer gilfree" in src
    C = _mod("r2_int_cell")
    D = C.configure()
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/tmp/x")
    for cell, pool, batch, steps, check, stress, pe, acct, bfx, smin, mem, arms in CELLS:   # every cell's arguments
        argv = ["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps), "--pool", str(pool),
                "--arms", arms, "--check-every", str(check), "--require-stress", str(stress), "--probe-every", str(pe),
                "--st1-commit", "rows", "--bf-store", str(bfx), "--bf-ncap", "2048", "--bf-ring", "16", "--bf-worker",
                "thread", "--bf-acct", str(acct), "--bf-close-wait-ms", "0", "--bf-packer", "gilfree",
                "--bf-threads", "14", "--bf-cap-groups", "1024", "--bf-late", "defer", "--bf-late-wait-ms", "250"]
        own, rest = C.parse_own(argv)
        a = D.parse_args(rest)
        assert a.arms == arms.split(",") and C.check_arms(a.arms, own, D) is None, cell
        assert (own.bf_ring, own.bf_threads, own.bf_cap_groups, own.bf_late, own.bf_late_wait_ms) == (16, 14, 1024,
                                                                                                   "defer", 250.0)
    bfs = (EFF / "r2_bf1_step.sh").read_text()                         # the same BF1 arguments as r2_bf1_step.sh
    for w in ("--bf-ring 16", "--bf-threads 14", "--bf-cap-groups 1024", "--bf-late defer", "--bf-late-wait-ms 250",
              "--bf-packer gilfree", "--bf-worker thread", "--bf-close-wait-ms 0"):
        assert w in bfs and w in " ".join(src.split()), w
        assert ("%s) POOL=%d BATCH=%d STEPS=%d" % (cell, pool, batch, steps)) in src, cell
        assert ("STEP_MIN=%d MEM=%s" % (smin, mem)) in src and arms in src, cell
    assert "intm) POOL=10 BATCH=8 STEPS=72 CHECK=1 STRESS=1 PE=4 ACCT=1" in src and "BFX=257 STEP_MIN=270" in src
    for bad in ("j1m", "int", "intc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    if str(ROOT) != STEP_ROOT or _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    sha = _git("rev-parse", "HEAD").stdout.strip()
    for cell, pool, batch, steps, check, stress, pe, acct, bfx, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if getattr(C, "GRANT_%s" % cell) is None:
            assert r.returncode == 1 and "is not granted" in r.stderr, (cell, r.returncode, r.stderr[-300:])
        assert ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 probe %d" % (cell, 63 + pool, pool, batch, arms,
                                                                                 steps, pe) in r.stdout), \
            (cell, r.stdout[-600:])
    _ev("INT DRYRUN: 3 cells printed their step command; grants %s" % ({c[0]: getattr(C, "GRANT_%s" % c[0]) for c in
                                                                       CELLS}))
