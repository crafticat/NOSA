"""CPU tests of rung J1 (KS/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md fix F1; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
the arms L4PTFDGUDCDJ1 / L4PTFDGUDKWCDJ1 (the I4DC best arms with the joint pass's block pooling compiled), benchmarks/
Efficiency/r2_j1_select.py; the cell r2_j1_cell.py; the step script r2_j1_step.sh; docs/NOTES_j1.md. The world:
tests_cpu/i4dc_world.py (its stand-ins: the pooling kernel = ragged_world.fake_pool_per_request, the kernel's flat-load
arithmetic; stage 1 = the fake op; CUDA graphs = FakeGraphsDCD).
CUDA GRAPHS DO NOT RUN ON THE CPU AND INDUCTOR'S CPU BACKEND IS NOT ITS GPU BACKEND: these tests establish the plumbing,
the bit equality on the CPU backend (the same graph, another code generator), the gates and the plants; only the GPU
miniature j1m establishes Triton's bits on live inputs (J1-PEQ) and the timing cells the saving.

GATE 0    test_j1_gate0_files: the tree = RECON7 cd0f0c5 + the J1 files + ONE module-level J1 block in each older Gate-0
          test (15 files, test_recon7.py included), only added lines; nosi/ and every other file are cd0f0c5's bytes; no
          kernel source; the fork is fia6's blob; the select module imports torch, time and the I4DC compile module
          only; no hand-written kernel in the J1 modules; nobody else names them.
          test_j1_layer_copy: the J1 state's _layer = spec_tentative.TentState._layer line for line except the lines
          marked "# J1" (the R 1 delegation and the one selection line).
POOL      test_pool_fn_equal: pool_fn (eager and Inductor-compiled) = the kernel's CPU stand-in (fake_pool_per_request on
          the copied cis rows, the shipped call's inputs) bit for bit, and = max_pooling_per_request.pool_reference, on
          varied lengths (n_used 0, 1, mid, full; M_b small, mid, M), init / local edges, bf16 and float32, a NaN
          payload outside every live window, widths Nq = round128(Nc) > Nc.
EQUAL     test_j1_equals (timing B 8, check B 8, both arms): the J1 run = L4PTFD and = its I4DC arm BIT FOR BIT
          (tokens, fingerprints, every state tensor) except the J1 record; J1-PEQ ran and found 0 differing values;
          J1-COMPILE: one compile event at most, calls = joint passes x L, no fallback; the I4DC cell gates pass on the
          CPU evidence under the J1 name.
PLANT     test_j1_plant_wrong_req: the cis of the next request: PEQ > 0 and EQF FAILS (not a no-op, L34).
F2        test_shpool_equal (ShPool = the shipped scalar kernel's stand-in), test_sh_j1 (SHJ1 on a base-cell stand-in,
          J1-SHEQ, the verdicts); the draft preconditions in test_j1_equals (J1-DRAFT).
F3        test_stamp_frame_equal (the shifted-age frame = the shipped stamps on a toy LRU kernel; the shift's plant),
          test_j3_gsection_copy (the I4S _gsection copy), test_j3_equals (L4PTFDGUDCDJ3 = the J1 arm and = L4PTFD bit
          for bit, ages and maps included; J1-RESID and its plants). The capture stand-in FakeGraphsJ3 records the
          residency's kernels as kernel nodes.
CELL      test_j1_cell_tables (the private copy; the I4DC cell module itself unchanged), test_j1_registration_state,
          test_step_script_j1.
J1F       test_forkbench_pure; test_forkbench_nonfinite (planted NaN / +Inf / -Inf in output, reference, both: the new
          NUM and EXACT gates FAIL with count and index; the j1f source's old accumulation, loaded from git, PASSED the
          identical-NaN plant: the control is not a no-op); test_forkbench_out_view (the flash controls' direct out_
          write: layout = the op's own return, the argument list = the installed wrapper's); test_forkbench_timing_
          plumbing (the rotated timing loop with a fake timer; kernel / copy / total per flash control).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_j1.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "cd0f0c59c1f1cac8a3ef05f6c3abcc0a2079d926"          # nosi-r2-recon7: RECON7 + the X1 notes
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-j1"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
J1_NEW = (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", T_ + "test_j1.py", "docs/NOTES_j1.md",
          E_ + "r2_j1_forkbench.py", E_ + "fork_vllm_unified_attention_ahead.py")
FORK_AHEAD = E_ + "fork_vllm_unified_attention_ahead.py"           # the fia6 fork + the [ahead] lines (a sibling)
J1_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u", "i4h",
                                                   "i4s", "i4cc", "i4dc", "recon7"))
# INT (nosi-r2-int: J1 + ST1 + BF1 in one arm, r2_int_cell.py): the ST1 and BF1 rungs' files and its own (its own Gate
# 0: tests_cpu/test_int.py); J1_LATER: its modules import the J1 modules (scan (e) skips them)
J1_NEW = J1_NEW + (E_ + "r2_st1_commit.py", E_ + "r2_st1_cell.py", E_ + "r2_st1_step.sh", T_ + "test_st1.py",
                   E_ + "r2_bf1_fetch.py", E_ + "r2_bf1_cell.py", E_ + "r2_bf1_step.sh", T_ + "test_bf1.py",
                   E_ + "r2_int_arms.py", E_ + "r2_int_cell.py", E_ + "r2_int_step.sh", T_ + "test_int.py",
                   "docs/NOTES_int.md")                                                           # INT
J1_LATER = ("r2_int_arms.py", "r2_int_cell.py")                                                   # INT
# R5 (review 5, ST: the yield / window population repair, r2_i4h_cell.population_repair; its tests in test_int.py)
J1_NEW = J1_NEW + (E_ + "r2_i4h_cell.py",)                                                         # R5
SELECT_IMPORTS = {"torch", "time", "types", "r2_i4dc_compile"}
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
LATER_NAMES = ("r2_i4a", "r2_i4cc", "r2_i4h", "r2_i4g", "r2_i4s", "r2_i4k", "r2_i4u", "r2_recon7")


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
CD, CC, L, N0 = X.CD, X.CC, X.L, X.N0
X.XC.MUT.update(j1pool=(8, 9))         # the written arguments of the J1 pooling call (pq, pc): the capture stand-in
X.XC.MUT.update(j1loc=())              # F6's locate writes no argument (it returns new tensors)


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
def test_j1_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(J1_NEW) | set(J1_DECL), sorted(changed - set(J1_NEW) - set(J1_DECL))      # (a)
    assert all((ROOT / f).is_file() for f in J1_NEW), [f for f in J1_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: cd0f0c5's
    for f in J1_DECL:                                                     # (c) one J1 block each, only added lines
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        assert set(old) == set(new) and all(old[k] == new[k] for k in old), (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert [x for x in body if x.startswith("-")] == [], f
        assert any("r2_j1_select.py" in x for x in body) and any("# J1" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    tree = ast.parse((EFF / "r2_j1_select.py").read_text())             # (d) the select module's imports
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert {m for m in mods if not m.startswith("torch.")} == SELECT_IMPORTS, mods
    for name in ("r2_j1_select.py", "r2_j1_cell.py", "r2_j1_forkbench.py"):   # NO HAND-WRITTEN KERNEL; no later rung
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
        assert not [w for w in LATER_NAMES if w in text], (name, [w for w in LATER_NAMES if w in text])
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names them
        if p.name.startswith("r2_j1_") or p.name in J1_LATER:                                       # INT
            continue
        assert "r2_j1" not in p.read_text(), p.name


def _dedented_body(fn):
    """The source lines of a function (method) after its def line, dedented."""
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    lines = src.splitlines()
    return lines[node.body[0].lineno - 1:]


def test_j1_layer_copy():
    J1 = _mod("r2_j1_select")
    S = J1.j1_classes(X.STF)["L4PTFDGUDCDJ1"][0]
    mine = _dedented_body(S._layer)
    orig = _dedented_body(X.STF.ST.TentState._layer)
    core = [x for x in mine if not x.rstrip().endswith("# J1")]
    extra = [x for x in mine if x.rstrip().endswith("# J1")]
    gone = [x for x in orig if "VR._select_rows(plan.vs, layer, q, twin)" in x]
    assert len(gone) == 1 and len(extra) == 3, (gone, extra)
    assert core == [x for x in orig if x not in gone], [(i, a, b) for i, (a, b) in enumerate(zip(core, orig)) if a != b][:3]
    assert "self._j1_select(plan.vs, layer, q, twin, l)" in extra[-1]
    assert S._layer.__qualname__.endswith("_J1Mix._layer")
    _ev("J1 COPY: _layer = TentState._layer + %d J1 lines (the R 1 delegation, the selection call); removed %s" % (
        len(extra), gone[0].strip()))


# ------------------------------------------------------------------ the pooling alone
def _pinputs(seed: int, dtype):
    """Shipped-call inputs at world-like shapes: H 2, B 3, R 4 (E 12), Nc 70, Nq 128 = round128(Nc), M 20."""
    g = torch.Generator().manual_seed(seed)
    H, B, R, Nc, M = 2, 3, 4, 70, 20
    E, Nq = B * R, 128
    score = torch.rand(H, E, Nq, generator=g).to(dtype)
    ccis = torch.randn(H, B, Nc, generator=g).to(dtype)
    n_used = torch.tensor([0, 1, 7, 70, 33, 34, 35, 36, 69, 70, 70, 12], dtype=torch.int32)
    M_b = torch.tensor([1, 2, 3, 20, 9, 9, 10, 10, 18, 19, 20, 4], dtype=torch.int32)
    if dtype == torch.bfloat16:                                          # a NaN payload outside every live window
        score.view(torch.int16)[0, 0, 100] = 0x7FC1
    return dict(H=H, B=B, R=R, E=E, Nc=Nc, Nq=Nq, M=M, score=score, ccis=ccis, n_used=n_used, M_b=M_b)


def _shipped(I, init, local):
    """The shipped call: the cis copied to the R rows, the kernel's CPU stand-in."""
    H, B, R, E, Nc, M = I["H"], I["B"], I["R"], I["E"], I["Nc"], I["M"]
    rows = I["ccis"].unsqueeze(2).expand(H, B, R, Nc).contiguous().view(H, E, Nc)
    pq = torch.empty((H, E, M), dtype=I["score"].dtype)
    pc = torch.empty((H, E, M), dtype=I["score"].dtype)
    X.W.fake_pool_per_request(I["score"], rows, I["n_used"], I["M_b"], pq, pc, init_blocks=init, local_blocks=local)
    return pq, pc, rows


def _fresh(fn):
    """A copy of fn's code object (dynamo keeps its cache on the code object: the test's compiles leave the arm's
    helper cache as it is)."""
    import types
    co = fn.__code__.replace(co_name=fn.__code__.co_name + "_test")
    return types.FunctionType(co, fn.__globals__, co.co_name, fn.__defaults__, fn.__closure__)


def _mine(I, fn, init, local, req_shift=0):
    J1 = _mod("r2_j1_select")
    H, B, R, E, M = I["H"], I["B"], I["R"], I["E"], I["M"]
    ar = torch.arange(E)
    req = (ar // R + req_shift) % B
    m = torch.arange(M)
    nidx = m.view(M, 1) * (J1.SET_SIZE - 1) - J1.PADDING + torch.arange(J1.SET_SIZE).view(1, J1.SET_SIZE)
    pq = torch.full((H, E, M), 7.0, dtype=I["score"].dtype)
    pc = torch.full((H, E, M), 7.0, dtype=I["score"].dtype)
    p0 = (pq.data_ptr(), pc.data_ptr())
    fn(I["score"], I["ccis"], ar.view(E, 1, 1), req.view(E, 1, 1).contiguous(), nidx.contiguous(), m.view(1, M),
       I["n_used"], I["M_b"], pq, pc, int(init), int(local) + 1)
    assert (pq.data_ptr(), pc.data_ptr()) == p0                         # in place
    return pq, pc


def test_pool_fn_equal():
    J1 = _mod("r2_j1_select")
    CC.configure()
    MP = X.W.pkg_load("max_pooling_per_request")
    comp = torch.compile(_fresh(J1.pool_fn), fullgraph=True, dynamic=False, backend="inductor")
    assert J1.compiled("pool", "inductor") is J1.compiled("pool", "inductor")         # one wrapper per process
    rows = []
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float32):
            for seed in range(2):
                for init, local in ((1, 16), (0, 2), (2, 0)):
                    I = _pinputs(seed, dtype)
                    rq, rc, crow = _shipped(I, init, local)
                    fq, fc = MP.pool_reference(I["score"][..., :I["Nc"]].contiguous(), crow, I["n_used"], I["M_b"],
                                               I["M"], init_blocks=init, local_blocks=local)   # equal widths
                    assert torch.equal(CD.bits(rq), CD.bits(fq)) and torch.equal(CD.bits(rc), CD.bits(fc))
                    for fn in (J1.pool_fn, comp):
                        mq, mc = _mine(I, fn, init, local)
                        assert torch.equal(CD.bits(mq), CD.bits(rq)), (dtype, seed, init, local, fn)
                        assert torch.equal(CD.bits(mc), CD.bits(rc)), (dtype, seed, init, local, fn)
                    bq, bc = _mine(I, J1.pool_fn, init, local, req_shift=1)          # the plant's arithmetic
                    assert not torch.equal(CD.bits(bc), CD.bits(rc))                 # differs (L34 precondition)
                    rows.append((str(dtype), seed, init, local))
    _ev("J1 POOL: %d variants (dtype, seed, init, local) bit-equal (eager and Inductor) to the kernel's stand-in and to "
        "pool_reference; dynamo %s" % (len(rows), CC.dynamo_counts()))


# ------------------------------------------------------------------ the arms
ARMS = dict(GUDCDJ1="L4PTFDGUDCDJ1", GUDKCDJ1="L4PTFDGUDKWCDJ1", GUDCDJ3="L4PTFDGUDCDJ3")
BASE = dict(GUDCDJ1="GUDCD", GUDKCDJ1="GUDKCD", GUDCDJ3="GUDCD")
C_OF = dict(GUDCD="GUDC", GUDKCD="GUDKC")
RESID_KERNELS = ("diff", "pool_update", "pool_swap")


class FakeGraphsJ3(X.FakeGraphsDCD):
    """FakeGraphsDCD for F3: the residency runs INSIDE the joint and the draft captures, so its kernels (diff,
    pool_update = the stamp frame, pool_swap) and the meter's gathers are kernel nodes of the capture (one recorded
    call each; their CPU stand-ins may read data-dependent shapes, as a GPU kernel never does). capture() is
    i4cc_world.FakeGraphsDC.capture with the kernel nodes added (the lines marked F3)."""

    def capture(self, fn, T, vs):
        rec = X.XG._Recorder()
        meter = T.meter
        kern = T.rs.ops.kernels                                                         # F3
        orig_k = {k: kern[k] for k in RESID_KERNELS if kern.get(k) is not None}         # F3
        orig_m, orig_f = meter.orig, vs.ops.fork
        meter.orig = tuple(rec.node(f) for f in orig_m)
        vs.ops.fork = rec.node(orig_f)
        kern.update({k: rec.node(f) for k, f in orig_k.items()})                       # F3
        T._cc_wrap = X.XC._cnode(rec)
        try:
            with rec:
                out = fn()
        finally:
            T._cc_wrap = None
            meter.orig, vs.ops.fork = orig_m, orig_f
            kern.update(orig_k)                                                         # F3
            for t, c in reversed(rec.saved):
                t.copy_(c)
            self.restored += len(rec.saved)
        g = X.XG.FakeGraph(rec.ops)
        self.graphs.append(g)
        return g, out

    def begin(self):
        super().begin()
        rec = self._open[0]
        T = self.T
        kern = T.rs.ops.kernels
        self._j3_k = {k: kern[k] for k in RESID_KERNELS if kern.get(k) is not None}
        self._j3_m = T.meter.orig
        kern.update({k: rec.node(f) for k, f in self._j3_k.items()})
        T.meter.orig = tuple(rec.node(f) for f in self._j3_m)

    def end(self):
        try:
            return super().end()
        finally:
            self.T.rs.ops.kernels.update(self._j3_k)
            self.T.meter.orig = self._j3_m


def j1_builder(arm, check=False, stats=False, kcheck=False, phase_marks=False, probe_every=0, j1neg=None,
               j1draft=True):
    """i4dc_world.d_builder with the J1 runner (the same arguments; the GPU cell's runner arguments)."""
    J1 = _mod("r2_j1_select")
    Runner = J1.j1_classes(X.STF)[ARMS[arm]][1]
    W, WB, Z, I3, KA, RE, RW, SR, BC = X.W, X.WB, X.Z, X.I3, X.KA, X.RE, X.RW, X.SR, X.BC

    def build(model, cache, S, horizon, tok0, K=X.K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(X.TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm="natural", extra=arena.X, attn=WB.fake_attention_u,
                          journal=RW.CAP, redirect=True, record=record, probe_every=probe_every)
        kw = dict(fork=Z.fake_fork, t0=tok0, hooks=None, check_every=check_every, union_check=check,
                  union_stats=stats, phase_marks=phase_marks, dneg=None, hneg=None, gneg=None, sneg=None, dgneg=None,
                  cc_backend="inductor", ccneg=None, cdneg=None, j1neg=j1neg, j1draft=j1draft)
        if arm == "GUDKCDJ1":
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=None, kslots=None, cneg=None)
        be = FakeGraphsJ3() if arm == "GUDCDJ3" else X.FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def run_j1(arm, scen="timing", B=8, j1neg=None, j1draft=True):
    CC.configure()
    s = X.SCEN[scen]
    N = N0
    model, c, S, hz, tok0 = X.world(B, N, 0)
    kc = bool(s["check"]) and arm == "GUDKCDJ1"
    run = j1_builder(arm, check=s["check"], stats=s["stats"], kcheck=kc, phase_marks=s["phase_marks"],
                     probe_every=s["probe_every"], j1neg=j1neg, j1draft=j1draft)(model, c, S, hz, tok0,
                                                                                 check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm=arm)


@functools.lru_cache(maxsize=None)
def _run(arm, scen="timing", B=8):
    if arm in ARMS:
        return run_j1(arm, scen, B)
    return X.run_arm(arm, scen, B=B)


CASES = dict(gudcd_timing=("GUDCDJ1", "timing"), gudcd_check=("GUDCDJ1", "check"),
             gudkcd_timing=("GUDKCDJ1", "timing"), gudkcd_check=("GUDKCDJ1", "check"))


def _cell():
    C = _mod("r2_j1_cell")
    return C, C.configure()


@pytest.mark.parametrize("case", list(CASES))
def test_j1_equals(case):
    arm, scen = CASES[case]
    keep = arm == "GUDKCDJ1"
    name = ARMS[arm]
    d = _run("D", scen)
    _run(C_OF[BASE[arm]], scen)                                        # the C arm first: the joint helpers
    b = _run(BASE[arm], scen)                                          # the I4DC arm: the draft helpers
    x = _run(arm, scen)
    J1 = _mod("r2_j1_select")
    rd = x["rd"]
    j = rd["j1"]
    jp = rd["tent"]["calls"]["joint_passes"]
    _ev("J1 %s %s: rounds %d; joint passes %d; draft capture passes %d; J1 %s" % (
        case, name, rd["rounds"], jp, rd["dgraph"]["capture_passes"], j))
    assert jp > 0 and j["calls"] == jp * L and not j["fallback"] and j["peq_layers"] > 0      # L34 preconditions
    assert j["peq_bad"] == 0 and j["peq_vals"] > 0
    assert j["draft"] and j["dwarm"] == 1 and j["dlayers"] == rd["dgraph"]["capture_passes"] * L > 0   # F2 ran
    assert j["dpeq_layers"] > 0 and j["dpeq_bad"] == 0 and rd["dgraph"]["replay_passes"] >= 10
    assert not any(e["capture"] for e in j["events"]) and {e["kind"] for e in j["events"]} <= {"joint", "draft_warm",
                                                                                              "f6"}
    assert j["f6"] and j["f6calls"] == rd["compile"]["sections"] > 0                     # F6 ran in every section
    assert rd["compile"]["calls"].get("loc1", 0) == 0                                     # the I4CC loc1: not called
    bad = TS._diff(b, x, skip=("compile", "j1"))
    assert bad == [], bad[:10]                                         # = its I4DC arm in everything but the records
    skip = ("hoist", "keep", "round_blocks", "compile", "j1") if keep else ("hoist", "compile", "j1")
    ts = ("kid", "kage", "blocks") if keep else ()
    bad = TS._diff(d, x, skip=skip, tensors_skip=ts)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    C, D = _cell()
    C.SH_J1["emitted"] = True                                          # J1-SHEQ: test_sh_j1 (emitted once per job)
    fd, fb, fx = X.cell_inputs(d, C=D), X.cell_inputs(b, C=D), X.cell_inputs(x, C=D)
    keepd = dict(L4PTFD=fd, **({"L4PTFDGUDKWCD": fb} if keep else {}))
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = D.arm_gates(name, rd, rec, fx, keepd, scen == "check", 8, 10, N0, L, 1.0)
    names = [g[0] for g in gs]
    want = ["MEM", "GRAPH", "COUNTS", "DGRAPH", "HOIST"] + (["KEEP"] + (["CAP"] if scen == "check" else []) +
                                                            ["KEEP-EQ", "KEEP-REF"] if keep else []) + [
        "COMPILE", "EQF", "J1-PEQ", "J1-COMPILE", "J1-DRAFT"]
    assert names == ["%s-%s" % (name, w) for w in want], names
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert rec["j1"] == j                                              # the summary carries the J1 record
    if keep:
        kq = next(g for g in gs if g[0].endswith("-KEEP-EQ"))
        assert kq[1] is True and "L4PTFDGUDKWCD" in kq[3], kq
    ev0 = dict(graphs=1, s=1.0, layer=0, capture=False, joint_pass=0, kind="joint")
    for plant in (dict(peq_bad=1), dict(peq_layers=0), dict(fallback=dict(record=1)), dict(calls=j["calls"] - 1),
                  dict(events=[dict(ev0, capture=True)]), dict(events=[ev0, ev0]),
                  dict(events=[dict(ev0, kind="draft")]), dict(dlayers=j["dlayers"] - 1), dict(dwarm=2),
                  dict(dpeq_bad=1), dict(dpeq_layers=0)):
        g = J1.j1_gates(name, dict(rd, j1=dict(j, **plant)), L)
        assert not all(x[2] for x in g), plant
    g = J1.j1_gates(name, dict(rd, j1=dict(j, draft=False)), L)        # F1 alone: no draft gate
    assert [x[0] for x in g] == ["%s-J1-PEQ" % name, "%s-J1-COMPILE" % name] and all(x[2] for x in g)


def test_j1_plant_wrong_req():
    C, D = _cell()
    d = _run("D")
    _run("GUDC")
    _run("GUDCD")
    p = run_j1("GUDCDJ1", "timing", j1neg="wrong_req")
    fd = X.cell_inputs(d, C=D)
    e = D.S4.eq_gate("L4PTFDGUDCDJ1", X.cell_inputs(p, C=D), fd, N0)
    df = TS._diff(d, p, skip=("hoist", "compile", "j1"))
    j = p["rd"]["j1"]
    _ev("J1 PLANT wrong_req: differs %s; EQF %s; PEQ bad %s of %s; draft PEQ bad %s of %s" % (
        df[:4], e[2], j["peq_bad"], j["peq_vals"], j["dpeq_bad"], j["dpeq_vals"]))
    assert df != [] and e[2] is False and j["peq_bad"] > 0             # not a no-op (L34); CAUGHT by EQF and PEQ
    assert j["dpeq_bad"] > 0                                           # the draft pooling's plant: CAUGHT by its PEQ
    q = run_j1("GUDCDJ1", "timing", j1neg="wrong_req", j1draft=False)  # F1 alone: the draft path is the shipped one
    assert q["rd"]["j1"]["dlayers"] == 0 and q["rd"]["j1"]["dpeq_vals"] == 0 and q["rd"]["j1"]["peq_bad"] > 0
    J1 = _mod("r2_j1_select")
    assert J1.j1_gates("L4PTFDGUDCDJ1", p["rd"], L)[0][2] is False
    with pytest.raises(X.SR.SpecError):
        j1_builder("GUDCDJ1", j1neg="nope")(*X.world(2, 8)[:5])


# ------------------------------------------------------------------ F6: the locate without the map sorts
def _linputs(seed: int, pool: bool):
    """loc1's inputs at world-like shapes with adversarial maps: H 2, B 5, R 4, T 8 (topk), P 6 (pool), nb 40 block
    ids; window ids unique per (head, request), pool ids unique and disjoint from the window's, -1 holes; row ids with
    -1, ids in the window, in the pool, in neither; fullv random."""
    g = torch.Generator().manual_seed(seed)
    H, B, R, T, P, nb = 2, 5, 4, 8, 6, 40
    bm = torch.full((H, B, T), -1, dtype=torch.int64)
    pm = torch.full((H, B, P), -1, dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            perm = torch.randperm(nb, generator=g)
            nw, npool = int(torch.randint(3, T + 1, (1,), generator=g)), int(torch.randint(0, P + 1, (1,), generator=g))
            slots = torch.randperm(T, generator=g)[:nw]
            bm[h, b, slots] = perm[:nw]
            ps = torch.randperm(P, generator=g)[:npool]
            pm[h, b, ps] = perm[nw:nw + npool]
    v = torch.randint(-1, nb, (H, B, R * T), generator=g)
    fullv = torch.rand(H, B, R * T, generator=g) > 0.3
    return dict(v=v, bm=bm, pm=pm if pool else None, fullv=fullv, big=torch.tensor(2 ** 31 - 1),
                arw=torch.arange(T), arp=torch.arange(P), topk=T, nb=nb)


def test_loc_fn_equal():
    """F6: loc_fn (eager and Inductor) = the I4CC loc1 helper on the sorted maps, for every consumed value (valid,
    in_win, in_pool, keyh everywhere; wslot under in_win, pslot under in_pool), with and without the pool."""
    J1 = _mod("r2_j1_select")
    CC.configure()
    comp = torch.compile(_fresh(J1.loc_fn), fullgraph=True, dynamic=False, backend="inductor")
    n = 0
    with torch.inference_mode():
        for seed in range(4):
            for pool in (True, False):
                I = _linputs(seed, pool)
                ws, wi = torch.sort(I["bm"], dim=-1)
                ps, pi = torch.sort(I["pm"], dim=-1) if pool else (None, None)
                ref = CC.loc1_fn(I["v"], ws, wi, ps, pi, I["fullv"], I["big"], I["topk"])
                for fn in (J1.loc_fn, comp):
                    got = fn(I["v"], I["bm"], I["pm"], I["fullv"], I["big"], I["arw"], I["arp"], I["topk"], I["nb"])
                    for k in (0, 1, 3, 5):                             # valid, in_win, in_pool, keyh
                        assert torch.equal(got[k], ref[k]), (seed, pool, k)
                    assert torch.equal(torch.where(ref[1], got[2], 0), torch.where(ref[1], ref[2], 0))
                    assert torch.equal(torch.where(ref[3], got[4], 0), torch.where(ref[3], ref[4], 0))
                assert ref[1].any() and (ref[3].any() or not pool) and (I["v"] < 0).any()    # L34: every case met
                n += 1
    _ev("J1 F6 LOC: %d variants (seed, pool): loc_fn = I4CC loc1 on the sorted maps for every consumed value" % n)


def test_cc_locate_copy():
    J1 = _mod("r2_j1_select")
    S = J1.j1_classes(X.STF)["L4PTFDGUDCDJ1"][0]
    M = next(c for c in S.__mro__ if c.__name__ == "_CCMix")
    mine, orig = _dedented(S._cc_locate), _dedented(M._cc_locate)
    core = [x for x in mine if not x.rstrip().endswith("# J1")]
    extra = [x for x in mine if x.rstrip().endswith("# J1")]
    i0 = next(i for i, x in enumerate(orig) if "ws, wi = torch.sort(eng._block_map" in x)
    gone = orig[i0:i0 + 7]                                            # the two sorts and the loc1 call
    assert "loc1" in gone[-2] and len(extra) == 3 and "_j1_loc(" in extra[-1], (gone, extra)
    assert core == orig[:i0] + orig[i0 + 7:], [(i, a, b) for i, (a, b) in enumerate(zip(core, orig)) if a != b][:3]


# ------------------------------------------------------------------ F3: the residency in the graphs
def _toy_pool_update(bm, lm, pm, age, tgt, act, stamp, tail):
    """A toy LRU kernel with pool_update's contract: per (head, request) the oldest REAL age (an empty, < 0, first) is
    replaced by stamp + (the first load id, clamped at 0); the chosen slot into tgt. Only comparisons of ages and
    stamp + m writes (the property the frame relies on)."""
    key = torch.where(age >= 0, age, age - (1 << 62))
    slot = key.argmin(-1, keepdim=True)
    age.scatter_(-1, slot, int(stamp) + lm[..., :1].clamp(min=0).to(age.dtype))
    tgt.copy_(slot.expand_as(tgt))


def test_stamp_frame_equal():
    """StampFrame = the shipped stamp sequence: the same evictions at every call and, after export, the same ages and
    host stamps; the first call inside a capture is refused; export re-arms the frame."""
    import types
    J1 = _mod("r2_j1_select")
    g = torch.Generator().manual_seed(11)
    H, B, P, Lyr, topk = 2, 3, 6, 2, 64

    def engines(seed_age):
        return [types.SimpleNamespace(_pool_age=seed_age.clone(), _pool_stamp_base=1000) for _ in range(Lyr)]
    age0 = torch.randint(-3, 900, (H, B, P), generator=g)
    age0[0, 0, :2] = torch.tensor([-1, -2])                            # empty slots
    ship, fr_e = engines(age0), engines(age0)
    frame = J1.StampFrame(_toy_pool_update, fr_e, topk)
    bad = 0
    for t in range(40):
        lm = torch.randint(-1, 50, (H, B, 4), generator=g)
        for l in range(Lyr):
            ts, tf = torch.zeros(H, B, 1, dtype=torch.int64), torch.zeros(H, B, 1, dtype=torch.int64)
            e = ship[l]
            _toy_pool_update(None, lm, None, e._pool_age, ts, None, e._pool_stamp_base, 7)
            e._pool_stamp_base += topk
            f = fr_e[l]
            frame(None, lm, None, f._pool_age, tf, None, f._pool_stamp_base, 7)
            f._pool_stamp_base += topk if t == 0 else 0                 # a captured caller moves the host int once
            bad += int((ts != tf).sum())
    assert bad == 0                                                     # the same evictions at every call
    rec = frame.export()
    for l in range(Lyr):
        assert torch.equal(ship[l]._pool_age, fr_e[l]._pool_age) and ship[l]._pool_stamp_base == fr_e[l]._pool_stamp_base
    assert rec["calls"] == [40] * Lyr and rec["first"] == [1000] * Lyr and frame.first == [None] * Lyr
    frame2 = J1.StampFrame(_toy_pool_update, engines(age0), topk)       # PLANT: no per-call shift -> evictions differ
    f2 = frame2.engines
    orig_sub = torch.Tensor.sub_
    try:
        torch.Tensor.sub_ = lambda self, *a, **k: self                  # the frame's shift disabled
        ok = True
        for t in range(40):
            lm = torch.randint(-1, 50, (H, B, 4), generator=g)
            ts, tf = torch.zeros(H, B, 1, dtype=torch.int64), torch.zeros(H, B, 1, dtype=torch.int64)
            _toy_pool_update(None, lm, None, ship[0]._pool_age, ts, None, ship[0]._pool_stamp_base, 7)
            ship[0]._pool_stamp_base += topk
            frame2(None, lm, None, f2[0]._pool_age, tf, None, 0, 7)
            ok = ok and torch.equal(ts, tf)
    finally:
        torch.Tensor.sub_ = orig_sub
    assert not ok                                                       # the shift is not a no-op (L34)
    _ev("J1 F3 FRAME: 40 calls x %d layers bit-equal to the shipped stamps; export %s; plant (no shift) caught" % (
        Lyr, rec))


def _dedented(fn):
    """The body lines of a function (method) after its docstring, dedented (test_recon7._body_lines' docstring rule)."""
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    first = node.body[0]
    doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(
        first.value.value, str)
    start = first.end_lineno if doc else first.lineno - 1
    return src.splitlines()[start:]


def test_j3_gsection_copy():
    J1 = _mod("r2_j1_select")
    S3 = J1.j1_classes(X.STF)["L4PTFDGUDCDJ3"][0]
    G = next(c for c in S3.__mro__ if c.__name__ == "_GraphS")
    mine, orig = _dedented(S3._gsection), _dedented(G._gsection)
    core = [x for x in mine if not x.rstrip().endswith("# J3")]
    gone = [x for x in orig if any(w in x for w in ("self._resid_eager(", "if copy_skip:", 'self._sb["skip"].copy_(skip)',
                                                    'sk0 = self._sb["skip"]', "setattr(self, stub,",
                                                    "want = expected_delta_s("))]
    assert len(gone) == 6, gone
    assert core == [x for x in orig if x not in gone], [(i, a, b) for i, (a, b) in enumerate(zip(core, orig)) if a != b][:3]
    extra = [x for x in mine if x.rstrip().endswith("# J3")]
    assert len(extra) == 4 and "expected_delta_j3(" in extra[2], extra


def test_j3_equals():
    """The J3 arm (F1 + F2 + F3) = the J1 arm and = L4PTFD BIT FOR BIT (every state tensor, the pool ages and maps
    included, after the frame's export), with the residency inside every joint and draft capture."""
    C, D = _cell()
    J1 = _mod("r2_j1_select")
    d = _run("D")
    _run("GUDC")
    _run("GUDCD")
    j = _run("GUDCDJ1")
    x = _run("GUDCDJ3")
    rd = x["rd"]
    _ev("J1 F3 timing: rounds %d; j3 %s; graph captures %s warm %s replays %s eager %s; dgraph keys %s captures %s "
        "capture passes %s replay passes %s eager %s" % (
            rd["rounds"], rd["j3"], rd["graph"]["captures"], rd["graph"]["warm"], rd["graph"]["replays"],
            rd["graph"]["eager_why"], rd["dgraph"]["keys"], rd["dgraph"]["captures"], rd["dgraph"]["capture_passes"],
            rd["dgraph"]["replay_passes"], rd["dgraph"]["eager_why"]))
    assert rd["dgraph"]["replay_passes"] >= 10 and rd["graph"]["replays"] > 0         # L34: replays ran
    bad = TS._diff(j, x, skip=("compile", "j1", "j3"))
    assert bad == [], bad[:10]                                         # = the J1 arm (ages after the export too)
    bad = TS._diff(d, x, skip=("hoist", "compile", "j1", "j3"))
    assert bad == [], bad[:10]                                         # = L4PTFD
    g = J1.j3_gates("L4PTFDGUDCDJ3", rd, L)
    assert g[0][2] is True, g
    C.SH_J1["emitted"] = True
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = D.arm_gates("L4PTFDGUDCDJ3", rd, rec, X.cell_inputs(x, C=D), dict(L4PTFD=X.cell_inputs(d, C=D)), False, 8,
                     10, N0, L, 1.0)
    assert [g_[0] for g_ in gs][-4:] == ["L4PTFDGUDCDJ3-%s" % w for w in ("J1-PEQ", "J1-COMPILE", "J1-DRAFT",
                                                                          "J1-RESID")]
    assert all(g_[2] for g_ in gs if g_[1]), [g_ for g_ in gs if g_[1] and not g_[2]]
    for plant in (dict(frame=[]), dict(joint_captures=L - 1), dict(draft_layers=0), dict(delta_fixed=0),
                  dict(frame=[dict(rd["j3"]["frame"][0], calls=[1, 2] + [1] * (L - 2))])):
        assert J1.j3_gates("x", dict(rd, j3=dict(rd["j3"], **plant)), L)[0][2] is False, plant


# ------------------------------------------------------------------ F2 on the baseline (SHJ1)
def test_shpool_equal():
    """ShPool = the shipped scalar kernel's stand-in (ragged_world.fake_nosa_pooling) bit for bit, over varied max_seqlen
    (0, small, mid, full), widths M and batches (dynamic shapes), init / local edges, bf16 and float32; its eager PEQ
    counts every call; a wrong shipped function is CAUGHT by its PEQ."""
    J1 = _mod("r2_j1_select")
    fake = X.W.fake_nosa_pooling
    sp = J1.ShPool(fake)
    g = torch.Generator().manual_seed(5)
    n = 0
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float32):
            for B, N, M, mx, init, local in ((3, 128, 20, 70, 1, 16), (3, 128, 20, 5, 0, 2), (5, 256, 40, 150, 1, 16),
                                             (5, 256, 40, 0, 2, 0), (2, 128, 33, 128, 1, 3)):
                q = torch.rand(2, B, N, generator=g).to(dtype)
                c = torch.randn(2, B, N, generator=g).to(dtype)
                rq, rc = torch.empty(2, B, M, dtype=dtype), torch.empty(2, B, M, dtype=dtype)
                mq, mc = torch.full((2, B, M), 7.0, dtype=dtype), torch.full((2, B, M), 7.0, dtype=dtype)
                fake(q, c, mx, rq, rc, block_size=32, stride=16, local_blocks=local, init_blocks=init)
                sp(q, c, mx, mq, mc, block_size=32, stride=16, local_blocks=local, init_blocks=init)
                assert torch.equal(CD.bits(mq), CD.bits(rq)) and torch.equal(CD.bits(mc), CD.bits(rc)), (dtype, B, M, mx)
                n += 1
    r = sp.report()
    assert r["calls"] == n == r["peq_calls"] and r["peq_bad"] == 0 and r["captured"] == 0 and r["peq_vals"] > 0
    assert len(r["events"]) <= 4, r["events"]                         # dynamic shapes: no compile per shape
    with pytest.raises(ValueError):
        sp(q, c, 5, mq, mc, padding=2)

    def wrong(q_, c_, mx_, oq, oc, **kw):                              # PLANT: a shipped function that differs
        fake(q_, c_, max(int(mx_) - 1, 0), oq, oc, **kw)
    bad = J1.ShPool(wrong)
    with torch.inference_mode():
        bad(q, c, 100, mq, mc, block_size=32, stride=16, local_blocks=3, init_blocks=1)
    assert bad.report()["peq_bad"] > 0
    _ev("J1 SHPOOL: %d variants bit-equal to the shipped kernel's stand-in; events %s; plant caught %s" % (
        n, r["events"], bad.report()["peq_bad"]))


class _FakeBC:
    """A base-cell stand-in for run_sh_j1: one eager pooling call through NL per run (the shipped warm-up's call)."""

    def __init__(self, NL, tokens):
        self.NL, self.tokens, self.seen = NL, tokens, []
        BCm = _mod("r2_base_cell")
        self.period_stats, self.rates = BCm.period_stats, BCm.rates

    def run_shipped(self, model, ids, steps, clock, fp=None):
        q = torch.rand(2, 3, 128)
        oq, oc = torch.empty(2, 3, 20), torch.empty(2, 3, 20)
        self.NL.nosa_pooling(q, q.clone(), 70, oq, oc, block_size=32, stride=16, local_blocks=16, init_blocks=1)
        self.seen.append(type(self.NL.nosa_pooling).__name__)
        return dict(inputs=self.tokens.clone(), periods=[10.0] * steps, rewarms=[(0, "warm")])

    def free(self, host=False):
        return False


def test_sh_j1():
    C, D = _cell()
    J1 = _mod("r2_j1_select")
    import types
    ns = types.SimpleNamespace(nosa_pooling=X.W.fake_nosa_pooling)
    tok = torch.arange(6).view(3, 2)
    fb = _FakeBC(ns, tok)
    with torch.inference_mode():
        rec = C.run_sh_j1(fb, None, tok, 8, None, dict(inputs=tok.clone()), NL=ns)
    assert ns.nosa_pooling is X.W.fake_nosa_pooling and fb.seen == ["ShPool"]          # swapped for the run, restored
    assert rec["ran"] and rec["tokens_equal"] and rec["pool"]["peq_calls"] == 1 and rec["pool"]["peq_bad"] == 0
    assert rec["rates"]["steady_step_ms"] == 10.0
    with torch.inference_mode():
        rec2 = C.run_sh_j1(_FakeBC(ns, tok + 1), None, tok, 8, None, dict(inputs=tok.clone()), NL=ns)   # PLANT
    assert rec2["ran"] and rec2["tokens_equal"] is False
    saved = dict(C.SH_J1)
    try:
        for r, want in ((rec, False), (dict(rec, pool=dict(rec["pool"], captured=4)), True),     # CPU: no capture
                        (dict(rec2, pool=dict(rec2["pool"], captured=4)), False), (dict(ran=False, error="x"), False)):
            C.SH_J1.update(result=r)
            g = C.sh_gate()
            assert g[0] == "J1-SHEQ" and g[1] is True and g[2] is want, (g, want)
        C.SH_J1.update(result=None)
        assert C.sh_gate()[1] is False                                 # not run: REPORT
        C.SH_J1.update(result=dict(rec, pool=dict(rec["pool"], captured=4)), emitted=False)
        rd = _run("GUDCDJ1")["rd"]
        gs = D.arm_gates("L4PTFDGUDCDJ1", rd, dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0,
                                                   mem_arena=dict(prefill_peak_GB=1.0)),
                         X.cell_inputs(_run("GUDCDJ1"), C=D), dict(L4PTFD=X.cell_inputs(_run("D"), C=D)), False, 8, 10,
                         N0, L, 1.0)
        assert gs[-1][0] == "J1-SHEQ" and gs[-1][2] is True and C.SH_J1["emitted"]
        gs2 = D.arm_gates("L4PTFDGUDCDJ1", rd, dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0,
                                                    mem_arena=dict(prefill_peak_GB=1.0)),
                          X.cell_inputs(_run("GUDCDJ1"), C=D), dict(L4PTFD=X.cell_inputs(_run("D"), C=D)), False, 8, 10,
                          N0, L, 1.0)
        assert "J1-SHEQ" not in [x[0] for x in gs2]                    # once per job
        sh = dict(rates=dict(steady_step_ms=13.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
        mk = lambda tok_, jt, dr: dict(rates=dict(median_tok_s=tok_), phases_median=dict(draft=dr, joint=jt),  # noqa
                                       gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0), j1=dict(compile_s=2.0))
        sp = dict(L4PTFD=mk(400.0, 200.0, 140.0), L4PTFDGUDCD=mk(440.0, 186.0, 130.0),
                  L4PTFDGUDCDJ1=mk(455.0, 174.0, 124.0))
        v = {y[0]: y for y in C.verdicts(sh, sp, 216)}
        assert "6.0000" in v["J1-DRAFT-SAVING"][2] and "3.0000" in v["J1-SH-SAVING"][2], v      # 13 - 10 ms per step
        assert "J1-VS-SHJ1" in v and "MODEL" in v["J1-STRONGEST-MATCHED"][2]
    finally:
        C.SH_J1.clear()
        C.SH_J1.update(saved)
    assert C.main(["--sh-j1", "2", "--out", "x", "--workload", "w"]) == 20


# ------------------------------------------------------------------ the exported factories (no behavior change)
def test_j1_wrap_export():
    J1 = _mod("r2_j1_select")
    STF = X.STF
    arms = J1.j1_classes(STF)
    DCL = X.CD.dc_classes(STF)
    for arm, base in J1.ARMS_J1.items():                               # the arms ARE the factory's (cached) pairs
        assert J1.j1_wrap(STF, *DCL[base], arm) == arms[arm]
    for arm, base in J1.ARMS_J3.items():
        assert J1.j3_wrap(STF, *arms[base], arm) == arms[arm]
    mix1 = arms["L4PTFDGUDCDJ1"][0].__mro__[1]
    mix3 = arms["L4PTFDGUDCDJ3"][0].__mro__[1]
    SB, RB = DCL["L4PTFDGUDKWCD"]
    S, R = J1.j1_wrap(STF, SB, RB, "XJ1")                              # a new arm on another base: the same mixin
    assert S.__mro__[1] is mix1 and S.__mro__[2] is SB and issubclass(R, RB) and R._arm_name(None) == "XJ1"
    assert J1.j1_wrap(STF, SB, RB, "XJ1") == (S, R)                    # cached
    S3, R3 = J1.j3_wrap(STF, S, R, "XJ3")
    assert S3.__mro__[1] is mix3 and S3.__mro__[2] is S and issubclass(R3, R) and R3._arm_name(None) == "XJ3"
    assert arms["L4PTFDGUDCDJ1"][0].__doc__ == "L4PTFDGUDCD's state with the compiled joint pooling."


# ------------------------------------------------------------------ the fork microbench (pure parts)
def test_forkbench_pure():
    FB = _mod("r2_j1_forkbench")
    SEL = _mod("r2_j1_select")
    VRm = X.W.pkg_load("verify_rows")
    assert FB.FORK_LAUNCH == VRm.FORK_LAUNCH and FB.MASK_BIAS == VRm.MASK_BIAS and FB.R == VRm.FORK_ROWS
    assert FB.UNION == 72 and FB.BLOCK_BYTES == 32896 and FB.GRANT_j1f is None and not FB.granted("j1f")
    assert FB.main(["--out", "x", "--cell", "j1f"]) == 20                # default deny
    pages, last = FB.row_pages(3, 2)
    assert pages[0] == 3 * 72 + 5 and len(pages) == 64 and pages == sorted(pages) and last == 42
    u = set()
    for r in range(4):                                                  # the rows' union = 72 pages per request
        u |= set(FB.row_pages(0, r)[0])
    assert u == set(range(72))
    B = 216
    w = {v: FB.work(v, B) for v in FB.variants((2, 4))}
    assert w["FORK"] == w["FORKT"] == w["FORK3D2"] == dict(rows=864, pages=2 * 216 * 72)
    assert w["FLASHU1"]["pages"] == w["FORK"]["pages"] and w["FLASHU1"]["rows"] == 216        # equal bytes
    assert w["FORK1R"] == w["NS1R1"] == dict(rows=216, pages=2 * 216 * 64)                     # equal rows and bytes
    assert w["NS1x4"] == dict(rows=864, pages=2 * 864 * 64)                                     # equal rows
    assert w["NSRU"]["pages"] == 2 * 216 * (64 + 67 + 69 + 72)
    for x in FB.FLASH:                                                  # the j1f form: X's work; the copy: none
        assert w[x + "_CP"] == w[x] and w[x + "_CO"] == dict(rows=0, pages=0), x
    assert FB.variants(()).index("NS1R1_CP") < FB.variants(()).index("NS1R1_CO")   # the copy's source runs first
    assert FB.rotation(["a", "b", "c"], 4) == ["b", "c", "a"]
    times = {v: [32.0 * (1.0 + 0.1 * i)] * 3 for i, v in enumerate(FB.variants((2,)))}
    times["FORK"] = [32.0 * 1.2] * 3
    times["FLASHU1"] = [32.0 * 0.6] * 3
    times["FORK3D2"] = [32.0 * 1.08] * 3
    valid = {v: True for v in times}
    valid["NSRU"] = False
    s = FB.summarize(times, B, 32, valid, (2,))
    assert s["per"]["FORK"]["ms_call"] == 1.2 and s["ratios"]["FORK/FLASHU1"]["ratio"] == 2.0
    assert s["per"]["FORK"]["logical_tb_s"] == round(2 * 216 * 72 * 32896 / (1.2 * 1e-3) / 1e12, 4)
    assert s["ratios"]["NSRU/FORK"]["valid"] is False and "INVALID" in s["ratios"]["NSRU/FORK"]["note"]
    assert s["fork_minus_flashu1_ms_per_pass"] == round(0.6 * 32, 3)
    assert s["best_3d"] == dict(variant="FORK3D2", ratio=0.9, saving_ms_per_pass=round(0.12 * 32, 3))
    assert {"FORK/FLASHU1", "FORK1R/NS1R1", "FORK/NS1x4", "FORKT/FORK", "NSRU/FORK", "FORK3D2/FORK", "FORK/FLASHU1_CP",
            "FORK1R/NS1R1_CP", "FORK/NS1x4_CP", "NSRU_CP/FORK", "FORKA1/FORK", "FORKA2/FORK", "FORKA3/FORK"} == set(
        s["ratios"])
    assert w["FORKA1"] == w["FORKA2"] == w["FORKA3"] == w["FORK"]
    assert s["per"]["NS1R1_CO"]["logical_tb_s"] is None and s["per"]["NS1R1_CO"]["bytes"] == 0
    assert set(s["flash_timing"]) == set(FB.FLASH) and s["flash_timing"]["NSRU"]["valid"] is False
    for bad in (["--out", "x", "--segments", "1"], ["--out", "x", "--reps", "2"], ["--out", "x", "--shapes", "a"]):
        with pytest.raises(SystemExit):
            FB.parse_args(bad)
    assert SEL is not None
    _ev("J1F PURE: work per variant %s; ratios %s" % (w, {k: v["ratio"] for k, v in s["ratios"].items()}))


# ------------------------------------------------------------------ the fork microbench: the j1f source follow-up
J1F_OLD = "cd0e2865924ed59a254fae1e283b1a2fcac4ad52"                  # the j1f source before the follow-up


def _fb_old():
    """r2_j1_forkbench.py as committed at J1F_OLD (the j1f run's source), loaded as a separate module object."""
    import types
    if not _have(J1F_OLD):
        pytest.skip("no git or no commit %s in %s" % (J1F_OLD, ROOT))
    src = _git("show", "%s:benchmarks/Efficiency/r2_j1_forkbench.py" % J1F_OLD).stdout
    assert "err = max(err, float((got.double() - want).abs().max()))" in src     # the old accumulation
    mod = types.ModuleType("r2_j1_forkbench_old")
    mod.__file__ = str(EFF / "r2_j1_forkbench.py")
    exec(compile(src, "r2_j1_forkbench@%s" % J1F_OLD[:7], "exec"), mod.__dict__)
    return mod


def _fb_world(FB, B=3, seed=11):
    """Outputs of FORK (4 rows per request), NS1R1 (= FORK's row 0) and FLASHU1, and float64 references 1e-5 away."""
    g = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    fork = (torch.randn(B * FB.R, FB.HQ, FB.HD, generator=g) * 0.05).to(bf)
    outs = dict(FORK=fork, NS1R1=fork.view(B, FB.R, FB.HQ, FB.HD)[:, 0].reshape(B, 1, FB.HQ, FB.HD).clone(),
                FLASHU1=(torch.randn(B, 1, FB.HQ, FB.HD, generator=g) * 0.05).to(bf))
    ref = {}
    for b in range(B):
        for r in range(FB.R):
            ref[(b, r)] = fork[b * FB.R + r].double() + 1e-5
        ref[(b, "U")] = outs["FLASHU1"].view(B, FB.HQ, FB.HD)[b].double() + 1e-5
    return outs, ref


# where the plant goes: variant -> (reference key, the output row index in out.view(-1, HQ, HD))
PLANT_AT = dict(FORK=((1, 2), 1 * 4 + 2), NS1R1=((2, 0), 2), FLASHU1=((0, "U"), 0))


def test_forkbench_nonfinite():
    """PLANTED NON-FINITE CONTROL (reviewer finding): NaN and +Inf (and -Inf) in the candidate output, in the float64
    reference, and in both (identical values): every case FAILS the new NUM gate with its count and first index; the
    finite case PASSES. The OLD accumulation (the module at J1F_OLD) PASSES the identical-NaN case (and others): the
    control is not a no-op (L34). Repeatability: identical non-finite bits pass the old bit test and FAIL the new one."""
    FB = _mod("r2_j1_forkbench")
    OLD = _fb_old()
    B = 3
    h, d = 5, 7
    old_pass = {}
    for name, (key, row) in PLANT_AT.items():
        outs, ref = _fb_world(FB, B)
        ok, det = FB.num_verdict(FB.num_check(name, outs[name], ref, B))           # the finite case
        assert ok and "max |out - float64|" in det, (name, det)
        assert OLD.num_err(name, outs[name], ref, B) <= OLD.TOL                     # the old code agrees here
        for val in (float("nan"), float("inf"), float("-inf")):
            for where in ("out", "ref", "both"):
                outs, ref = _fb_world(FB, B)
                o = outs[name]
                if where in ("out", "both"):
                    o.view(-1, FB.HQ, FB.HD)[row, h, d] = val
                if where in ("ref", "both"):
                    ref[key][h, d] = val
                chk = FB.num_check(name, o, ref, B)
                ok, det = FB.num_verdict(chk)
                assert not ok and chk["nonfinite"] == 1, (name, val, where, chk, det)
                assert det.startswith("NON-FINITE: 1 value(s)"), det
                assert "request %s row %s head %d dim %d" % (key[0], key[1], h, d) in det, det
                assert chk["err"] == chk["err"] and chk["err"] < 1e-3, chk      # the max stays finite and honest
                old_pass[(name, repr(val), where)] = OLD.num_err(name, o, ref, B) <= OLD.TOL
    for name in PLANT_AT:                                                   # the old hole, per variant
        assert old_pass[(name, "nan", "both")], (name, old_pass)            # identical NaN: the old gate PASSED
        assert old_pass[(name, "nan", "out")] and old_pass[(name, "nan", "ref")]
        assert old_pass[(name, "inf", "both")]                             # inf - inf = NaN: vanished too
        assert not old_pass[(name, "inf", "out")]                          # a lone Inf was already caught
    # repeatability: identical bits that hold a non-finite value
    for val in (float("nan"), float("inf")):
        outs, _ = _fb_world(FB, B)
        o = outs["FORK"]
        o.view(-1, FB.HQ, FB.HD)[6, h, d] = val
        first = o.clone()
        assert bool((OLD.bits(o) == OLD.bits(first)).all())                # the old EXACT: PASS (a no-op on NaN)
        ok, det = FB.repeat_check(o, first)
        assert not ok and "bit for bit: True" in det and "last 1 (first at (6, %d, %d))" % (h, d) in det, det
    outs, _ = _fb_world(FB, B)
    o = outs["FORK"]
    assert FB.repeat_check(o, o.clone())[0]                                 # finite and identical: PASS
    o2 = o.clone()
    o2.view(-1)[123] += 0.5
    assert not FB.repeat_check(o2, o)[0]                                    # finite and different: FAIL
    outs, ref = _fb_world(FB, B)                                            # no reference row: FAIL (fail closed)
    ok, det = FB.num_verdict(FB.num_check("FORK1R", outs["NS1R1"], {k: v for k, v in ref.items() if k[1] != 0}, B))
    assert not ok and "no row checked" in det
    # the decision lines: a planted NaN makes the variant INVALID in the ratios and the flash timing
    outs, ref = _fb_world(FB, B)
    outs["NS1R1"].view(-1)[0] = float("nan")
    valid = {v: True for v in FB.variants((2,))}
    valid["NS1R1"] = FB.num_verdict(FB.num_check("NS1R1", outs["NS1R1"], ref, B))[0]
    assert valid["NS1R1"] is False
    s = FB.summarize({v: [32.0, 33.0, 34.0] for v in valid}, 216, 32, valid, (2,))
    assert s["ratios"]["FORK1R/NS1R1"]["valid"] is False and "INVALID" in s["ratios"]["FORK1R/NS1R1"]["note"]
    assert s["ratios"]["FORK1R/NS1R1_CP"]["valid"] is True and s["flash_timing"]["NS1R1"]["valid"] is False
    valid["FLASHU1"] = False
    assert "fork_minus_flashu1_ms_per_pass" not in FB.summarize({v: [32.0] * 3 for v in valid}, 216, 32, valid, (2,))
    _ev("J1F NONFINITE: new NUM fails every plant (NaN/+Inf/-Inf x out/ref/both, 3 variants); the old accumulation "
        "passed %d of %d plants: %s" % (sum(old_pass.values()), len(old_pass),
                                        sorted(k for k, v in old_pass.items() if v)))


def test_forkbench_best3d_validity():
    """REVIEW 5: the 3D saving follows the ratios' validity rule. PLANT: FORK invalid -> best_3d None, the note says
    INVALID and the decision line is SUPPRESSED (no number); the j1f source (cd0e286) still reported the saving (the
    control is not a no-op). The fastest 3D member invalid -> it is not chosen. FORK - FLASHU1 likewise."""
    FB = _mod("r2_j1_forkbench")
    OLD = _fb_old()
    segs = (2, 4)
    times = {"FORK": [32.0] * 3, "FLASHU1": [20.0] * 3, "FORK3D2": [28.0] * 3, "FORK3D4": [30.0] * 3}
    ok = {v: True for v in times}
    s = FB.summarize(times, 216, 32, ok, segs)
    assert s["best_3d"]["variant"] == "FORK3D2" and s["best_3d"]["saving_ms_per_pass"] == 4.0 and s["best_3d_note"] is None
    bad = dict(ok, FORK=False)                                                # PLANT: FORK invalid
    s = FB.summarize(times, 216, 32, bad, segs)
    assert s["best_3d"] is None and s["best_3d_note"].startswith("INVALID: FORK")
    assert "fork_minus_flashu1_ms_per_pass" not in s
    lines = FB.decision_lines("C128_B216", s)
    assert [x for x in lines if "best 3D SUPPRESSED (INVALID: FORK" in x]
    assert not [x for x in lines if "saving_ms_per_pass" in x or ("best 3D" in x and "FORK3D2" in x)], lines
    assert all(x.endswith(" INVALID") for x in lines if "RATIO FORK3D" in x), lines            # the ratio: INVALID
    assert [x for x in lines if "FORK - FLASHU1 = SUPPRESSED" in x]
    assert all(v["valid"] is False for k, v in s["ratios"].items() if "FORK" in k.split("/"))
    so = OLD.summarize(times, 216, 32, bad, segs)                             # the old rule: the saving survived
    assert so["best_3d"]["saving_ms_per_pass"] == 4.0 and so["fork_minus_flashu1_ms_per_pass"] == 12.0
    s = FB.summarize(times, 216, 32, dict(ok, FORK3D2=False), segs)           # the fastest 3D member invalid
    assert s["best_3d"]["variant"] == "FORK3D4"
    s = FB.summarize(times, 216, 32, dict(ok, FORK3D2=False, FORK3D4=False), segs)
    assert s["best_3d"] is None and s["best_3d_note"] == "INVALID: no valid 3D variant"
    _ev("J1F BEST3D: FORK invalid -> saving suppressed (old source reported %s ms)" % so["best_3d"]["saving_ms_per_pass"])


def test_forkbench_rows_sweep_pure():
    """TRACK 2: the rows sweep's work, row mapping, bit check, slopes, suppression and decision lines (CPU)."""
    FB = _mod("r2_j1_forkbench")
    B = 287
    v = FB.variants((2,))
    assert all(x in v for x in FB.ROWS) and FB.ROWS == ("RSM1", "RSM2", "RSM3", "RSM4", "RSQ1", "RSQ2", "RSQ4")
    for x in FB.ROWS:                                                    # fixed union bytes, n active rows
        assert FB.work(x, B) == dict(rows=int(x[3:]) * B, pages=2 * B * 72), x
    Bs = 3
    g = torch.Generator().manual_seed(5)
    forkt = (torch.randn(Bs * 4, FB.HQ, FB.HD, generator=g) * 0.05).to(torch.bfloat16)
    for n in (1, 2, 3, 4):
        got = forkt.view(Bs, 4, FB.HQ, FB.HD)[:, :n].contiguous().view(Bs * n, FB.HQ, FB.HD)
        for pre in ("RSM", "RSQ") if n != 3 else ("RSM",):
            ok, det = FB.rows_eq("%s%d" % (pre, n), got, forkt, Bs)
            assert ok and "0 values differ" in det, det
            assert torch.equal(FB.out_row("%s%d" % (pre, n), got, (2, n - 1), Bs), forkt[2 * 4 + n - 1])
            assert FB.out_row("%s%d" % (pre, n), got, (2, n), Bs) is None if n < 4 else True
        bad = got.clone()
        bad.view(-1)[7] += 0.25                                         # PLANT: one value off
        assert not FB.rows_eq("RSM%d" % n, bad, forkt, Bs)[0]
        bad = got.clone()
        bad.view(-1)[7] = float("nan")                                  # PLANT: NaN
        assert "non-finite 1" in FB.rows_eq("RSM%d" % n, bad, forkt, Bs)[1]
    ms = dict(FORK=34.0, FORKT=33.0, FORK1R=22.0, RSM1=30.0, RSM2=31.0, RSM3=32.0, RSM4=33.0, RSQ1=24.0, RSQ2=27.0,
              RSQ4=33.0)
    valid = {x: True for x in ms}
    s = FB.summarize({x: [t] * 3 for x, t in ms.items()}, B, 32, valid, ())
    rs = s["rows_sweep"]
    assert rs["masked"]["slope_ms_per_row"] == 1.0 and rs["masked"]["rsm4_minus_forkt"] == 0.0
    assert rs["tile"] == dict(ms_pass={1: 24.0, 2: 27.0, 4: 33.0}, per_row_1_to_2=3.0, per_row_2_to_4=3.0)
    assert rs["e_fork_ms_pass"] == 2.0 and rs["tile4_fork_ms_pass"] == 9.0
    s = FB.summarize({x: [t] * 3 for x, t in ms.items()}, B, 32, dict(valid, RSM2=False, RSQ1=False), ())
    rs = s["rows_sweep"]
    assert rs["masked"] is None and rs["tile"] is None and rs["e_fork_ms_pass"] is None
    lines = FB.decision_lines("C81_B287", s)
    assert [x for x in lines if "ROWS masked SUPPRESSED" in x] and [x for x in lines if "ROWS e_fork_ms_pass SUPP" in x]
    src = (EFF / "r2_j1_forkbench.py").read_text()
    assert "fork_call(qn, oM, cun, n, *tT, None, block_q=R)" in src and "fork_call(qn, oQ, cun, n, *tT, None)" in src
    _ev("J1F ROWS: rows sweep plumbing; planted value / NaN detected")


NCU_QUERY = """Device NVIDIA A100
  dram__bytes_read                 # of bytes read from DRAM
  gpu__time_duration               total duration in nanoseconds
  lts__t_sectors_srcunit_tex_op_read   # of LTS sectors from unit TEX for reads
  sm__warps_active                 cumulative # of warps in flight
  launch__registers_per_thread     registers per thread
"""


def test_forkbench_ncu_plumbing(tmp_path):
    """TRACK 2: the ncu argv, the metric query filter, the raw CSV parser (the fia6 record's real CSV when present,
    else a written one) and the per-arm summary; the refusal of a missing ncu binary; the j1n arguments."""
    FB = _mod("r2_j1_forkbench")
    names = FB.ncu_metric_names(NCU_QUERY)
    missing, metrics = FB.ncu_metric_list(names)
    assert missing == [] and metrics[:3] == list(FB.NCU_REQUIRED)
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in metrics and "launch__registers_per_thread" in metrics
    assert "dram__bytes_write.sum" not in metrics                        # not in this query: not asked
    assert FB.ncu_metric_list(names - {"dram__bytes_read"})[0] == ["dram__bytes_read.sum"]
    a = FB.parse_args(["--out", "x", "--cell", "j1n", "--shapes", "81:287", "--segments", "2", "--ncu", "/n/ncu"])
    argv = FB.ncu_argv("/n/ncu", a, 81, 287, "RSQ2", "/o/child", "/o/rep", metrics)
    for want in (["--profile-from-start", "off"], ["--cache-control", "all"], ["--replay-mode", "kernel"],
                 ["--mode", "ncu-cell"], ["--ncu-arms", "RSQ2"], ["--shapes", "81:287"], ["--cell", "j1n"]):
        i = argv.index(want[0])
        assert argv[i:i + 2] == want, (want, argv)
    child = FB.parse_args(argv[argv.index(sys.executable) + 2:])        # the child's own arguments parse
    assert child.mode == "ncu-cell" and child.ncu_arms == ("RSQ2",) and child.shapes == {81: 287}
    for bad in (["--out", "x", "--ncu-arms", "NOPE"], ["--out", "x", "--mode", "ncu-cell", "--shapes", "81:287,73:318",
                                                       "--ncu-arms", "FORK"]):
        with pytest.raises(SystemExit):
            FB.parse_args(bad)
    real = Path("/mnt/beegfs/ojerbi/r2-runs/fiattn/fia6_l2a_2195574/ncu_216x1_raw.csv")
    if real.is_file():                                                   # the real format (read only)
        rows = FB.parse_ncu_csv(real.read_text(errors="replace"), ["gpu__time_duration.sum", "dram__bytes_read.sum"])
        assert rows and all(r["gpu__time_duration.sum"] for r in rows), rows[:2]
        assert rows[0]["name"] == "_bi_split_kernel" and rows[0]["dram__bytes_read.sum"] > 0
    head = '"ID","Process ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum",' \
           '"lts__t_sectors_srcunit_tex_op_read.sum","sm__warps_active.avg.pct_of_peak_sustained_active",' \
           '"launch__registers_per_thread"'
    text = "\n".join([head, '"","","","ns","byte","sector","%","register/thread"',
                      '"0","1","kernel_unified_attention","1,200,000","1000000000","40000000","12.4","255"',
                      '"1","1","kernel_unified_attention","300000","50000000","2000000","10.0","255"'])
    rows = FB.parse_ncu_csv(text, metrics)
    s = FB.ncu_arm_summary(rows, 1_000_000_000)
    assert s["n_kernels"] == 2 and s["time_us"] == 1500.0 and s["dram_read"] == 1.05e9
    assert s["dram_over_logical"] == 1.05 and s["l2sm_bytes"] == 42e6 * 32 and s["occupancy_pct"] == 12.4
    assert s["registers"] == 255.0
    with pytest.raises(ValueError):
        FB.parse_ncu_csv("no header", metrics)
    old = FB.GRANT_j1n
    FB.GRANT_j1n = "2026-01-01T00:00:00Z"                               # a stand-in grant (restored below)
    try:
        assert FB.main(["--out", str(tmp_path / "o"), "--cell", "j1n", "--ncu", str(tmp_path / "no_ncu")]) == 20
        assert not (tmp_path / "o").exists()
    finally:
        FB.GRANT_j1n = old
    assert FB.GRANT_j1n is None and not FB.granted("j1n")
    _ev("J1F NCU: argv %s ... ; summary %s" % (argv[:12], s))


def test_forkbench_out_view():
    """The direct write: swapped_out_view puts the op's swapped output (N, HQ/HKV, HKV, HD) where the op's own return
    (flash_api.cpp: out.transpose(1, 2).reshape(N, 1, HQ, HD)) puts it, with the strides of the wrapper's own output
    (torch.empty_like of the swapped q); the call passes it at the op's out_ position, every other argument = the
    wrapper's; the installed wrapper passes None there; only the _CP variants copy_."""
    FB = _mod("r2_j1_forkbench")
    N, G = 3, FB.HQ // FB.HKV
    out = torch.zeros(N, 1, FB.HQ, FB.HD, dtype=torch.bfloat16)
    v = FB.swapped_out_view(out, FB.HKV)
    q = torch.randn(N, 1, FB.HQ, FB.HD).to(torch.bfloat16)
    qs = q.reshape(N, FB.HKV, G, FB.HD).transpose(1, 2)                   # flash_api.cpp: the swapped q
    assert tuple(v.shape) == tuple(qs.shape) == (N, G, FB.HKV, FB.HD) and v.data_ptr() == out.data_ptr()
    assert v.stride() == torch.empty_like(qs).stride() and v.stride(-1) == 1   # = the wrapper's output layout
    S = torch.randn(N, G, FB.HKV, FB.HD).to(torch.bfloat16)                # what the kernel writes (swapped)
    v.copy_(S)
    assert torch.equal(out, S.transpose(1, 2).reshape(N, 1, FB.HQ, FB.HD))   # = the op's own return transform
    for n, g, k in ((0, 0, 0), (1, 3, 1), (2, G - 1, FB.HKV - 1)):         # head k * G + g = swapped [g, k]
        assert torch.equal(out[n, 0, k * G + g], S[n, g, k]) and torch.equal(qs[n, g, k], q[n, 0, k * G + g])
    for bad in (torch.zeros(N, 2, FB.HQ, FB.HD), torch.zeros(N, 1, FB.HQ, FB.HD).transpose(2, 3)):
        with pytest.raises(AssertionError):
            FB.swapped_out_view(bad, FB.HKV)
    src = (EFF / "r2_j1_forkbench.py").read_text()                       # the call: the op's out_ = ov
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
             n.func.id == "FAOP"]
    assert len(calls) == 1 and len(calls[0].args) == 21 and not calls[0].keywords
    got = [ast.unparse(a) for a in calls[0].args]
    want = ["qx", "kc", "vc", "bc", "None", "None", "lens", "None", "None", "cbi", "None", "None", "None", "ov", "scale",
            "False", "-1", "-1", "0.0", "True", "SPLITS"]
    assert got == want, got
    assert src.count(".copy_(FA(") == 1 and "def wrapper_cp():\n            o_cp.copy_(FA(" in src   # only _CP copies
    assert "FAOP = FAI.flash_attn_cuda.fwd_kvcache" in src
    spec = importlib.util.find_spec("flash_attn_nosa")                    # the installed wrapper (not imported)
    if spec is None or not spec.origin:
        pytest.skip("flash_attn_nosa not installed here")
    itxt = (Path(spec.origin).parent / "flash_attn_interface.py").read_text()
    assert "import flash_attn_2_cuda_nosa as flash_attn_cuda" in itxt
    m = re.search(r"out, softmax_lse = flash_attn_cuda\.fwd_kvcache\((.*?)\n    \)", itxt, re.S)
    wargs = [x.strip() for x in m.group(1).split(",") if x.strip()]
    assert wargs == ["q", "k_cache", "v_cache", "kv_bias", "k", "v", "cache_seqlens", "rotary_cos", "rotary_sin",
                     "cache_batch_idx", "cache_leftpad", "block_table", "alibi_slopes", "None", "softmax_scale",
                     "causal", "window_size[0]", "window_size[1]", "softcap", "rotary_interleaved", "num_splits"], wargs
    for line in ("softcap=0.0", "rotary_interleaved=True", "window_size=(-1, -1)", "causal=False",
                 "softmax_scale = q.shape[-1] ** (-0.5)"):                 # the wrapper's defaults = the direct call's
        assert line in itxt, line
    cpp = ROOT / "dependencies/flash-attention-nosa/csrc/flash_attn_nosa/flash_api.cpp"
    if cpp.is_file():
        ctxt = cpp.read_text()
        sig = ctxt[ctxt.index("mha_fwd_kvcache(at::Tensor &q"):]
        sig = sig[:sig.index(") {")]
        params = [x.split("//")[0].strip() for x in sig.split("\n")]
        names_ = [re.findall(r"&?(\w+)\s*,?$", p)[0] for p in params if p]
        assert names_.index("out_") == 13 and len(names_) == 21, names_
        assert "out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});" in ctxt
        assert "CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);" in ctxt
    _ev("J1F OUTVIEW: the direct call passes the out_ view at position 13 of the installed op %s; layout = the op's "
        "return" % spec.origin)


def test_forkbench_timing_plumbing():
    """The timing loop with a FAKE timer (CUDA events do not run on the CPU): warm-up passes first, then the rotated
    rounds, `layers` calls per pass, the stop after the budget; summarize's kernel / copy / total split per control."""
    FB = _mod("r2_j1_forkbench")
    names = list(FB.variants((2,)))
    seen = []
    parts = {v: (lambda v=v: seen.append(v)) for v in names}
    ms = {v: 10.0 + i for i, v in enumerate(names)}
    for x in FB.FLASH:
        ms[x], ms[x + "_CP"], ms[x + "_CO"] = 20.0, 24.0, 1.5
    by_fn = {id(parts[v]): v for v in names}

    def fake_timer(fn, layers):
        for _ in range(layers):
            fn()
        return ms[by_fn[id(fn)]]
    times, warm, stop = FB.time_variants(parts, names, 3, 2, 4, fake_timer, 0.0, 1e9, clock=lambda: 1.0)
    assert stop is None and all(len(warm[v]) == 2 and len(times[v]) == 4 for v in names)
    k = 2 * 3 * len(names)                                                  # the warm-up calls, in list order
    assert seen[:k] == [v for v in names for _ in range(2 * 3)]
    rounds = [seen[k + i * 3 * len(names):k + (i + 1) * 3 * len(names):3] for i in range(4)]
    assert rounds == [FB.rotation(names, r) for r in range(4)]
    assert len(seen) == 3 * len(names) * 6
    t2, _, stop2 = FB.time_variants(parts, names, 1, 1, 10, fake_timer, 0.0, 5.0, clock=iter([1.0, 6.0, 9.0]).__next__)
    assert stop2 == "wall clock above 5 s after pass 1" and all(len(t2[v]) == 2 for v in names)
    s = FB.summarize(times, 216, 32, {v: True for v in names}, (2,))
    for x in FB.FLASH:
        assert s["flash_timing"][x] == dict(direct_e2e_ms_pass=20.0, copy_only_ms_pass=1.5,
                                            total_j1f_form_ms_pass=24.0, residual_ms_pass=2.5, valid=True), \
            s["flash_timing"][x]
    lines = FB.decision_lines("C128_B216", s)
    assert any("FLASHT NS1R1" in x and "direct (end-to-end, preallocated output) 20.000" in x and "residual 2.500" in x
               for x in lines), lines
    assert not [x for x in lines if "FLASHT" in x and ("kernel" in x or " rest " in x)], lines
    assert not [x for x in lines if "per-byte efficiency" in x or "optimal" in x], lines
    assert s["ratios"]["FORK1R/NS1R1"]["ratio"] == round(ms["FORK1R"] / 20.0, 4)
    assert s["ratios"]["FORK1R/NS1R1_CP"]["ratio"] == round(ms["FORK1R"] / 24.0, 4)
    src = (EFF / "r2_j1_forkbench.py").read_text()                       # run_shape times through this loop
    assert "time_variants(parts, names, a.layers, a.warmup, a.reps, cuda_timer, t0, a.max_shape_s)" in src
    _ev("J1F TIMING: fake-timer plumbing; flash split %s" % s["flash_timing"]["NS1R1"])


# ------------------------------------------------------------------ the fork with its loads ahead (lead's edit plan)
def test_ahead_copy():
    """fork_vllm_unified_attention_ahead.py = the fia6 fork + lines tagged [ahead]: without the tagged lines and with
    the indentation stripped, its lines are the fork's lines in order (an original line may only move into an
    `if AHEAD` / `else` block). The fork itself stays the fia6 blob (Gate 0). The flag defaults to 0."""
    fork = [x for x in (EFF / "fork_vllm_unified_attention.py").read_text().splitlines() if x.strip()]
    ahead = [x for x in (ROOT / FORK_AHEAD).read_text().splitlines() if x.strip()]    # blank lines aside
    kept = [x.strip() for x in ahead if "[ahead]" not in x]
    assert kept == [x.strip() for x in fork], next(
        ((i, a, b) for i, (a, b) in enumerate(zip(kept, [x.strip() for x in fork])) if a != b), len(kept))
    tagged = [x for x in ahead if "[ahead]" in x]
    assert len(tagged) == len(ahead) - len(fork) and len(tagged) > 40, len(tagged)
    moved = [x for x, y in zip([x for x in ahead if "[ahead]" not in x], fork) if x != y]
    assert all(x.startswith(" ") for x in moved) and len(moved) <= 8, moved     # re-indented lines only, few
    assert any(x.startswith("    ahead=0,  # [ahead]") for x in ahead) and "        AHEAD=int(ahead),  # [ahead]" in ahead
    for w in ("tl.dot", "softmax_step", "tl.store", "tl.exp"):              # no arithmetic in the tagged lines
        assert not [x for x in tagged if w in x and not x.lstrip().startswith("#")], w
    _ev("J1 AHEAD COPY: %d tagged lines, %d re-indented fork lines" % (len(tagged), len(moved)))


AHEAD_CHILD = r'''
import os, sys, importlib.util
assert os.environ.get("TRITON_INTERPRET") == "1" and "triton" not in sys.modules
sys.dont_write_bytecode = True
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FK = load("fork_fia6", sys.argv[1])
FA = load("fork_ahead", sys.argv[2])
HQ, HKV, HD, BS, R, B, P = 32, 2, 128, 64, 4, 2, 5          # the real heads and block; 4 rows; 2 requests; 5 pages
SHIFT = ((0, 1, 2, 3), (3, 2, 1, 0))                         # per KV head: row r starts SHIFT[h][r] pages in
LAST = (40, 64, 17, 1)                                       # tokens of each row's last page
U = P + 3                                                    # union entries per (KV head, request): 8
E, NB = B * R, B * U + 4
NANPAGE = NB - 1                                             # unreferenced; the tables' tail points at it
torch.manual_seed(3)
ak, av = torch.randn(NB, BS, HKV, HD) * 0.5, torch.randn(NB, BS, HKV, HD)
ab = torch.randn(NB, BS, HKV) * 0.5
for t in (ak, av, ab):
    t[NANPAGE] = float("nan")
q = torch.randn(E, HQ, HD)
scale = HD ** -0.5
i32 = torch.int32
key = torch.zeros((E, HKV, P), dtype=i32)
for b in range(B):
    for r in range(R):
        for h in range(HKV):
            key[b * R + r, h] = torch.arange(P, dtype=i32) + b * U + SHIFT[h][r]
lists = dict(page_key=key, page_table=key.clone(), num_pages=torch.full((E, HKV), P, dtype=i32),
             last_len=torch.tensor(LAST * B, dtype=i32), seq_idx=torch.zeros(E, dtype=i32), slots_per_entry=1)
cu = torch.arange(0, E + 1, R, dtype=i32)
seq = torch.zeros(B, dtype=i32)
W = R * 8                                                    # BLOCK_Q x P_PAD


def tables():
    return (torch.full((HKV, B, W), NANPAGE, dtype=i32), torch.zeros((HKV, B, W), dtype=i32),
            torch.zeros((HKV, B), dtype=i32))


def run(mod, tab, ext, ln, build, bias, tile, **kw):
    out = torch.zeros(E, HQ, HD)
    mod.unified_attention(q, ak, av, out, cu, R, seq, BS, scale, True, (-1, -1), tab, 0, None, None, None,
                          union_ext=ext, union_len=ln, cis_bias=bias, build_union=lists if build else None,
                          rows_per_q_block=R, num_warps=4, num_stages=3, tile_size=tile, **kw)
    return out


def ref():
    out = torch.zeros(E, HQ, HD, dtype=torch.float64)
    g = HQ // HKV
    for e in range(E):
        for hq in range(HQ):
            h = hq // g
            pages = key[e, h].long()
            n = (P - 1) * BS + LAST[e % R]
            K = ak[pages, :, h].double().reshape(-1, HD)[:n]
            V = av[pages, :, h].double().reshape(-1, HD)[:n]
            Bi = ab[pages, :, h].double().reshape(-1)[:n]
            s = (K @ q[e, hq].double() + Bi) * scale
            out[e, hq] = torch.softmax(s, 0) @ V
    return out


def same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


fails, lines = [], []
REF = ref()
for tile in (64, 32):
    for bias in (ab, None):
        bname = "bias" if bias is not None else "nobias"
        t0 = tables()
        f_build = run(FK, *t0, True, bias, tile)                  # the fork, union built in the kernel
        if bias is not None:
            err = float((f_build.double() - REF).abs().max())
            lines.append("REF tile %d: max |fork - float64| %.2e" % (tile, err))
            if not err <= 1e-4:
                fails.append(("fork_vs_ref", tile, err))
        if not bool((t0[2] == U).all()):
            fails.append(("union_len", tile, t0[2].tolist()))
        f_table = run(FK, *(x.clone() for x in t0), False, bias, tile)   # the fork reading those tables
        for lvl in (0, 1, 2, 3):
            for build, want in ((True, f_build), (False, f_table)):
                t = tables() if build else tuple(x.clone() for x in t0)
                got = run(FA, *t, build, bias, tile, ahead=lvl)
                ok = same(got, want) and bool(torch.isfinite(got).all())
                nd = int((got.view(torch.int32) != want.view(torch.int32)).sum())
                lines.append("EQ tile %d %s ahead %d %s: bitwise %s (%d differ)" % (
                    tile, bname, lvl, "build" if build else "table", ok, nd))
                if not ok:
                    fails.append(("EQ", tile, bname, lvl, build, nd))
                if build and not torch.equal(t[0][:, :, :U], t0[0][:, :, :U]):
                    fails.append(("tables", tile, bname, lvl))
        # THE ROWS SWEEP (r2_j1_forkbench RSMn / RSQn): n query rows per request on the same tables; BLOCK_Q 4
        # (RSM) and BLOCK_Q n (RSQ, n in 1, 2, 4): rows 0 .. n-1 = the 4-row table read's rows
        for n in (1, 2, 3, 4):
            qn = q.view(B, R, HQ, HD)[:, :n].contiguous().view(B * n, HQ, HD)
            cun = torch.arange(0, B * n + 1, n, dtype=i32)
            for bq in ((R, n) if n in (1, 2) else (R,)):
                on = torch.zeros(B * n, HQ, HD)
                tq = tuple(x.clone() for x in t0)
                FK.unified_attention(qn, ak, av, on, cun, n, seq, BS, scale, True, (-1, -1), tq[0], 0, None, None,
                                     None, union_ext=tq[1], union_len=tq[2], cis_bias=bias, build_union=None,
                                     rows_per_q_block=bq, num_warps=4, num_stages=3, tile_size=tile)
                ok = same(on.view(B, n, HQ, HD).contiguous(), f_table.view(B, R, HQ, HD)[:, :n].contiguous())
                lines.append("ROWS tile %d %s n %d BLOCK_Q %d: rows = the 4-row read's rows bitwise %s" % (
                    tile, bname, n, bq, ok))
                if not ok and bq == R:                        # RSM: MUST; RSQ (another BLOCK_M): REPORT
                    fails.append(("ROWS", tile, bname, n, bq))
        # PLANT (L34): the failure mode of a wrong shift = tile j reads tile j + 1's page: must differ
        tp = tuple(x.clone() for x in t0)
        tp[0][0, 0, 2] = tp[0][0, 0, 3]
        got = run(FA, *tp, False, bias, tile, ahead=1)
        nd = int((got.view(torch.int32) != f_table.view(torch.int32)).sum())
        lines.append("PLANT stale page tile %d %s: %d values differ" % (tile, bname, nd))
        if nd == 0:
            fails.append(("PLANT_stale_page_not_detected", tile, bname))
        # PLANT (REPORT): two union entries swapped (same set, other tile order): the bits see the order
        tp = tuple(x.clone() for x in t0)
        tp[0][0, 0, [1, 2]] = tp[0][0, 0, [2, 1]]
        tp[1][0, 0, [1, 2]] = tp[1][0, 0, [2, 1]]
        got = run(FA, *tp, False, bias, tile, ahead=1)
        nd = int((got.view(torch.int32) != f_table.view(torch.int32)).sum())
        lines.append("REPORT swapped order tile %d %s: %d values differ, max |d| %.2e" % (
            tile, bname, nd, float((got - f_table).abs().max())))
print("\n".join(lines))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("AHEAD-INTERP OK")
'''


def test_ahead_interp():
    """THE BIT-PRESERVATION CHECK ON THE CPU (lead's plan, edit A1/A2): under TRITON_INTERPRET=1 (a fresh child, L36),
    real heads (32 query, 2 KV, dim 128, block 64), 2 requests x 4 rows, per-row shifted pages (union 8), partial last
    pages, a NaN page behind the tables' tail: the ahead copy with ahead 0, 1, 2 = the fia6 fork BIT FOR BIT, union
    built in the kernel and read from tables, with and without the bias, tile 64 and 32 (2 tiles per block). The fork
    = the float64 reference within 1e-4 (the inputs exercise attention). PLANT: a stale page (tile j reads tile j + 1's
    page) is DETECTED. LIMIT: the interpreter runs numpy; it shows the same values in the same operations. It does not
    show the GPU code generator's reduction order; the GPU gate FORKAn-EQ (r2_j1_forkbench) does."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", AHEAD_CHILD, str(EFF / "fork_vllm_unified_attention.py"),
                        str(ROOT / FORK_AHEAD)], env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["AHEAD-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
    _ev("J1 AHEAD INTERP:\n" + r.stdout.strip())


COMPILE_CHILD = r'''
import os, re, subprocess, sys, importlib.util, tempfile
assert os.environ.get("TRITON_INTERPRET", "0") != "1"
sys.dont_write_bytecode = True
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver


class FakeDriver:                                    # a CPU node: no CUDA driver; compile only (warmup, no launch)
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return GPUTarget("cuda", 80, 32)             # A100 (sm_80), warp 32: the record's GPU


driver.set_active(FakeDriver())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FK = load("fork_fia6", sys.argv[1])
FA = load("fork_ahead", sys.argv[2])
BIN = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")
HQ, HKV, HD, BS, R, B, P = 32, 2, 128, 64, 4, 216, 64         # the j1f / production shapes (B 216, 64 pages per row)
E, NB = B * R, 16                                             # the arena size does not enter the code (strides do)
bf, i32 = torch.bfloat16, torch.int32


class Capture:                                                 # kernel[grid](**kw) -> warmup (compile, no launch)
    def __init__(self, jit):
        self.jit, self.kernels = jit, []

    def __getitem__(self, grid):
        def go(**kw):
            k = self.jit.warmup(grid=grid, **kw)
            self.kernels.append(k)
            return k
        return go


def compiled(mod, bias_dtype, build, **kw):
    cap = Capture(mod.kernel_unified_attention)
    real, mod.kernel_unified_attention = mod.kernel_unified_attention, cap
    try:
        q = torch.zeros(E, HQ, HD, dtype=bf)
        k = torch.zeros(NB, BS, HKV, HD, dtype=bf)
        key = torch.zeros((E, HKV, P), dtype=i32)
        lists = dict(page_key=key, page_table=key.clone(), num_pages=torch.zeros((E, HKV), dtype=i32),
                     last_len=torch.zeros(E, dtype=i32), seq_idx=torch.zeros(E, dtype=i32), slots_per_entry=1)
        tab = torch.zeros((HKV, B, R * P), dtype=i32)
        mod.unified_attention(q, k, k.clone(), torch.zeros_like(q), torch.arange(0, E + 1, R, dtype=i32), R,
                              torch.zeros(B, dtype=i32), BS, HD ** -0.5, True, (-1, -1), tab, 0, None, None, None,
                              union_ext=torch.zeros_like(tab), union_len=torch.zeros((HKV, B), dtype=i32),
                              cis_bias=torch.zeros(NB, BS, HKV, dtype=bias_dtype), build_union=lists if build else None,
                              rows_per_q_block=R, num_warps=4, num_stages=3, tile_size=64, **kw)
    finally:
        mod.kernel_unified_attention = real
    return cap.kernels[-1]


def usage(kern):
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(kern.asm["cubin"])
    out = subprocess.run([os.path.join(BIN, "cuobjdump"), "--dump-resource-usage", f.name], capture_output=True,
                         text=True).stdout
    os.unlink(f.name)
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", out)
    return dict(reg=int(m.group(1)), stack=int(m.group(2)), local=int(m.group(4)), shared=kern.metadata.shared)


def sass(kern):
    """The cubin's SASS instruction lines (cuobjdump -sass), without addresses and encodings."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(kern.asm["cubin"])
    out = subprocess.run([os.path.join(BIN, "cuobjdump"), "-sass", f.name], capture_output=True, text=True).stdout
    os.unlink(f.name)
    return [re.sub(r"/\*[0-9a-f]+\*/|/\* 0x[0-9a-f]+ \*/", "", x).strip() for x in out.splitlines()
            if re.search(r"/\*[0-9a-f]{4}\*/", x)]


def loop_ops(ttgir):
    """The tile loop = the scf.for whose body holds the tt.dot ops: its tt.load / async copy / async_wait counts."""
    lines = ttgir.splitlines()
    first_dot = next(i for i, x in enumerate(lines) if "tt.dot" in x)
    start = max(i for i in range(first_dot) if "scf.for" in lines[i])
    ind = len(lines[start]) - len(lines[start].lstrip())
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("}")
               and len(lines[i]) - len(lines[i].lstrip()) == ind)
    body = lines[start:end]
    return dict(sync_load=sum(" tt.load " in x for x in body), async_copy=sum("async_copy_global_to_local" in x
                for x in body), async_wait=sum("async_wait" in x for x in body), dots=sum("tt.dot" in x for x in body))


LOC = re.compile(r' ?loc\((?:"[^"]*"(?::\d+:\d+)?|#loc\d*|[^()]*)\)')


def strip_loc(s):
    """TTGIR without source locations (file paths and line numbers differ between the two files)."""
    return "\n".join(LOC.sub("", x) for x in s.splitlines() if not x.startswith("#loc"))


def ptx_code(s):
    """PTX without its debug information (.file / .loc lines, the .debug sections) and comments."""
    s = s.split(".section", 1)[0]
    return [x for x in s.splitlines() if x.strip() and not x.strip().startswith((".loc", ".file", "//", "$L__"))]


def canon(lines):
    """SASS with registers renamed in order of first use: the same instructions on the same data flow."""
    names = {}

    def ren(m):
        return names.setdefault(m.group(0), "%s%d" % (m.group(1), len(names)))
    return [re.sub(r"\b(UR|R|UP|P)(\d+)\b", ren, x) for x in lines]


CTRL = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fork_ctrl_%d.py" % os.getpid())   # the fork at another path
with open(sys.argv[1]) as f, open(CTRL, "w") as g:
    g.write(f.read())
FC = load("fork_ctrl", CTRL)


def compare(x, y):
    gx, gy = strip_loc(x.asm["ttgir"]).splitlines(), strip_loc(y.asm["ttgir"]).splitlines()
    px, py = ptx_code(x.asm["ptx"]), ptx_code(y.asm["ptx"])
    sx, sy = sass(x), sass(y)
    first = lambda a, b: next(((i, u, v) for i, (u, v) in enumerate(zip(a, b)) if u != v), None)  # noqa: E731
    return dict(ttgir=gx == gy, ptx=px == py, sass=sx == sy, sass_canon=canon(sx) == canon(sy), n_ttgir=len(gx),
                n_ptx=len(px), n_sass=len(sx), d_ttgir=first(gx, gy), d_ptx=first(px, py), d_sass=first(sx, sy))


rows = []
for bias in (torch.bfloat16, torch.float32):
    base = compiled(FK, bias, True)
    a0 = compiled(FA, bias, True, ahead=0)
    for what, other in (("ahead0_vs_fork", a0), ("CONTROL_fork_copy_vs_fork", compiled(FC, bias, True))):
        c = compare(base, other)
        same = c["ttgir"] and c["n_ttgir"] > 1000                     # MUST: the same TTGIR (PTX / SASS: REPORT)
        print("%s bias %s %s: ttgir %s (%d lines) ptx %s (%d lines) sass %s, sass with registers renamed %s (%d)" % (
            "SAME_CODE" if what.startswith("ahead") else "REPORT", str(bias)[6:], what, c["ttgir"], c["n_ttgir"],
            c["ptx"], c["n_ptx"], c["sass"], c["sass_canon"], c["n_sass"]) + (" " + str(same) if
                                                                                what.startswith("ahead") else ""))
        if not (c["ttgir"] and c["ptx"] and c["sass"]):
            print("DIFF %s ttgir %s | ptx %s | sass %s" % (what, str(c["d_ttgir"])[:300], str(c["d_ptx"])[:300],
                                                         str(c["d_sass"])[:300]))
    for name, kern in (("fork", base), ("ahead0", a0), ("ahead1", compiled(FA, bias, True, ahead=1)),
                       ("ahead2", compiled(FA, bias, True, ahead=2)), ("ahead3", compiled(FA, bias, True, ahead=3)),
                       ("fork_table", compiled(FK, bias, False)), ("ahead2_table", compiled(FA, bias, False, ahead=2)),
                       ("ahead3_table", compiled(FA, bias, False, ahead=3))):
        u, lo = usage(kern), loop_ops(kern.asm["ttgir"])
        print("WITNESS bias %s %-12s reg %d stack %d local %d shared %d | loop sync tt.load %d async copies %d "
              "async_wait %d dots %d" % (str(bias)[6:], name, u["reg"], u["stack"], u["local"], u["shared"],
                                         lo["sync_load"], lo["async_copy"], lo["async_wait"], lo["dots"]))
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"                  # the same pair without line info (fresh JIT objects)
FK2, FA2 = load("fork_fia6_noline", sys.argv[1]), load("fork_ahead_noline", sys.argv[2])
for bias in (torch.bfloat16, torch.float32):
    c = compare(compiled(FK2, bias, True), compiled(FA2, bias, True, ahead=0))
    print("REPORT bias %s ahead0_vs_fork WITHOUT LINE INFO: ttgir %s ptx %s sass %s, renamed %s" % (
        str(bias)[6:], c["ttgir"], c["ptx"], c["sass"], c["sass_canon"]))
os.unlink(CTRL)
print("COMPILE-WITNESS OK")
'''


def test_ahead_compile_witness():
    """CPU COMPILE WITNESS (no GPU, no launch): Triton 3.2 compiles the fia6 fork and the ahead copy for sm_80 (A100)
    through JITFunction.warmup with a stand-in driver (target only); cuobjdump --dump-resource-usage reads registers,
    stack and shared memory from the cubin; the TTGIR tile loop's synchronous tt.load / async copies are counted. MUST:
    ahead 0 compiles to the fork's TTGIR and PTX (the flag off = the fork's code). REPORT: the rest (evidence for the
    lead's GPU decision; timing is the GPU's)."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(PYTHONDONTWRITEBYTECODE="1", TRITON_CACHE_DIR=str(Path(os.environ.get("TMPDIR", "/tmp")) /
                                                                  ("j1_ahead_cache_%d" % os.getpid())))
    r = subprocess.run([sys.executable, "-c", COMPILE_CHILD, str(EFF / "fork_vllm_unified_attention.py"),
                        str(ROOT / FORK_AHEAD)], env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    out = r.stdout.strip().splitlines()
    assert r.returncode == 0 and out[-1:] == ["COMPILE-WITNESS OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
    _ev("J1 AHEAD COMPILE WITNESS (sm_80, w4 s3 tile 64, B 216):\n" + "\n".join(out))
    assert [x for x in out if x.startswith("SAME_CODE")] and all(x.endswith(" True") for x in out
                                                                 if x.startswith("SAME_CODE")), "\n".join(out)


# ------------------------------------------------------------------ CELL
def test_j1_cell_tables():
    C, D = _cell()
    I4DC = _mod("r2_i4dc_cell")
    assert D is not I4DC and D.__name__ == C.DCC_NAME
    assert os.path.realpath(D.__file__) == os.path.realpath(I4DC.__file__)   # the same file, a second module object
    assert "L4PTFDGUDCDJ1" not in I4DC.ARMS and "L4PTFDGUDCDJ1" not in I4DC.BASE_D       # the import: unchanged
    assert D.ARMS == I4DC.ARMS + C.ARMS_J1 + C.ARMS_J3 and D.BASE_D["L4PTFDGUDKWCDJ1"] == "L4PTFDGUDKWCD"
    assert D.BASE_D["L4PTFDGUDCDJ3"] == "L4PTFDGUDCD" and D.ROOT_ARM["L4PTFDGUDCDJ3"] == "L4PTFDGUD"
    assert "L4PTFDGUDCDJ3" in D.GRAPH_ARMS and "L4PTFDGUDCDJ3" not in D.KEEP_ARMS
    assert D.runner_choice("L4PTFDGUDCDJ3", False) == ("SCD", dict(cc_backend="inductor"))
    assert D.ROOT_ARM["L4PTFDGUDCDJ1"] == "L4PTFDGUD" and D.ROOT_ARM["L4PTFDGUDKWCDJ1"] == "L4PTFDGUDKW"
    assert D.NAMES["L4PTFDGUDCDJ1"] == tuple(n for n in I4DC.NAMES["L4PTFDGUDCD"] if n != "loc1")   # F6
    assert "L4PTFDGUDKWCDJ1" in D.KEEP_ARMS
    assert C.configure() is D                                          # idempotent
    base = ["--out", "x", "--workload", "w"]
    for arms in ("L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1",
                 "L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1"):
        assert D.parse_args(base + ["--arms", arms]).arms == arms.split(",")
    for bad in ("L4PTFD,L4PTFDGUDC,L4PTFDGUDCDJ1", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCDJ1,L4PTFDGUDCD"):
        with pytest.raises(SystemExit):
            D.parse_args(base + ["--arms", bad])
    assert D.runner_choice("L4PTFDGUDCDJ1", False) == ("SCD", dict(cc_backend="inductor"))
    J1 = _mod("r2_j1_select")
    C._CUR["arm"] = "L4PTFDGUDKWCDJ1"
    try:
        assert D.runner_class("SKCD", X.STF) is J1.j1_classes(X.STF)["L4PTFDGUDKWCDJ1"][1]
    finally:
        C._CUR["arm"] = None
    assert D.runner_class("SKCD", X.STF) is X.DCL["L4PTFDGUDKWCD"][1]
    assert D.keep_ref_arm("L4PTFDGUDKWCDJ1", {"L4PTFD": 1, "L4PTFDGUDKWC": 1, "L4PTFDGUDKWCD": 1}) == "L4PTFDGUDKWC"
    assert D.keep_ref_arm("L4PTFDGUDKWCDJ1", {"L4PTFD": 1}) is None
    assert D.keep_ref_arm("L4PTFDGUDKWCD", {"L4PTFD": 1, "L4PTFDGUDKWC": 1}) == "L4PTFDGUDKWC"
    assert D.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    mk = lambda tok, jt, dr: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa: E731
                                  gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0), j1=dict(compile_s=2.0))
    sp = dict(L4PTFD=mk(400.0, 200.0, 140.0), L4PTFDGUDCD=mk(440.0, 186.0, 130.0), L4PTFDGUDCDJ1=mk(455.0, 174.0, 130.0))
    v = {y[0]: y for y in C.verdicts(sh2, sp, 216)}
    if C.PRED_STAMP_J1 is None:                                         # unregistered: every verdict REPORT
        assert all(y[1] == "REPORT" for y in v.values()), v
    else:                                                               # registered: a band gives CONFIRMED / REFUTED
        assert all(y[1] in ("REPORT", "CONFIRMED", "REFUTED") for y in v.values()), v
        b = C.PRED_J1_BY_B[216].get("J1_joint_saving")
        if b is not None:
            assert v["J1-JOINT-SAVING"][1] == ("CONFIRMED" if b[0] <= 12.0 <= b[1] else "REFUTED"), v
    assert "12.0000" in v["J1-JOINT-SAVING"][2], v
    assert C.verdicts(sh2, dict(L4PTFD=sp["L4PTFD"]), 216) == [("J1-RUN", "NOT_RUN", "no J1 arm ran")]
    vv = [y[0] for y in D.verdicts(sh2, sp, 216)]
    assert "I4DC-VS-REF" in vv and "J1-JOINT-SAVING" in vv             # the I4DC verdicts + the J1 verdicts


def test_j1_registration_state():
    C = _mod("r2_j1_cell")
    src = (EFF / "r2_j1_cell.py").read_text()
    if C.PRED_STAMP_J1 is None:                                         # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_J1_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_J1)
    assert set(C.PRED_J1_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS_J1) for x in C.PRED_J1_BY_B.values())
    assert ("\nPRED_STAMP_J1 = None\n" in src) == (C.PRED_STAMP_J1 is None)   # the step script's grep sees it
    for cell in ("j1m", "j1a", "j1b"):                                  # the grants: None or a date -u stamp
        g = getattr(C, "GRANT_%s" % cell)
        assert g is None or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", g)
        assert (("\nGRANT_%s = None\n" % cell) in src) == (g is None)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_j1_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


CELLS = (("j1m", 10, 8, 72, 1, 1, 35, "100G",
          "L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1,L4PTFDGUDCDJ3"),
         ("j1a", 65, 216, 512, 0, 0, 125, "240G", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1,L4PTFDGUDCDJ3"),
         ("j1b", 10, 318, 512, 0, 0, 170, "340G", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1"))


def test_step_script_j1():
    src = (EFF / "r2_j1_step.sh").read_text()
    assert src.rstrip().endswith('if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi')
    for word in ("python", "perl", "awk", "sed", "sha256sum", "hash-object"):   # no interpreter / hash on login
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % re.escape(word), x)
                 and "/venv/nosa/bin/python" not in x and "/usr/bin/sha256sum" not in x and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/j1_jit " in src
    for f in ("r2_j1_select.py", "r2_i4dc_cell.py", "r2_i4dc_compile.py", "r2_i4h_graph.py", "r2_i4g_graph.py",
              "r2_i4a_lean.py", "max_pooling_per_request.py", "verify_rows.py", "spec_tentative.py"):
        assert f in src, f
    assert "OUTROOT/sp_active_$JOB" in src and "granted \"$CELL\" <(git -C \"$NOSI\" show \"$NSHA:$CELLPY\")" in src
    C = _mod("r2_j1_cell")
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:   # every cell's arguments parse
        a = C.configure().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                      "--pool", str(pool), "--arms", arms, "--check-every", str(check),
                                      "--require-stress", str(stress)])
        assert a.arms == arms.split(",")
        assert ("%s) POOL=%d BATCH=%d STEPS=%d" % (cell, pool, batch, steps)) in src and ("STEP_MIN=%d MEM=%s" % (
            smin, mem)) in src, cell
        assert arms in src, cell
    assert "j1f) POOL=65 BATCH=0 STEPS=0 STEP_MIN=25 MEM=64G NEEDWL=0 OUT_MB=60" in src           # the microbench
    assert "CELLPY=benchmarks/Efficiency/r2_j1_forkbench.py" in src and '"${WARGS[@]}"' in src
    FB = _mod("r2_j1_forkbench")
    fa = FB.parse_args(["--out", "x", "--cell", "j1f", "--shapes", "128:216,73:318", "--segments", "2,4,8", "--layers",
                        "32", "--reps", "10", "--warmup", "2", "--seed", "7", "--sample", "8", "--max-shape-s", "300",
                        "--max-run-s", "900"])                          # the step script's j1f arguments parse
    assert fa.shapes == {128: 216, 73: 318} and fa.segments == (2, 4, 8)
    assert "j1n) POOL=18 BATCH=0 STEPS=0 STEP_MIN=50 MEM=64G NEEDWL=0 OUT_MB=400" in src          # TRACK 2
    assert "NCU=/opt/nvidia/nsight-compute/2025.1.0/ncu " in src and '--ncu "$NCU" --ncu-cap 240' in src
    fn = FB.parse_args(["--out", "x", "--cell", "j1n", "--shapes", "81:287", "--segments", "2", "--layers", "32",
                        "--reps", "6", "--warmup", "2", "--seed", "7", "--sample", "8", "--max-shape-s", "400",
                        "--max-run-s", "600", "--ncu", "/opt/nvidia/nsight-compute/2025.1.0/ncu", "--ncu-cap", "240"])
    assert fn.shapes == {81: 287} and fn.ncu_arms == FB.NCU_ARMS and fn.mode == "main"
    for bad in ("i4dca", "j1", "j1c"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    if str(ROOT) != STEP_ROOT or _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    sha = _git("rev-parse", "HEAD").stdout.strip()
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if not hasattr(C, "GRANT_%s" % cell) or getattr(C, "GRANT_%s" % cell) is None:
            assert r.returncode == 1 and "is not granted" in r.stderr, (cell, r.returncode, r.stderr[-300:])
        assert ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3" % (cell, 63 + pool, pool, batch, arms, steps)
                in r.stdout), (cell, r.stdout[-600:])
    _ev("J1 DRYRUN: 3 cells printed their step command; grants %s" % ({c[0]: getattr(C, "GRANT_%s" % c[0]) for c in
                                                                      CELLS}))
