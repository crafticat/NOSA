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
J1_NEW = (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", T_ + "test_j1.py", "docs/NOTES_j1.md")
J1_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u", "i4h",
                                                   "i4s", "i4cc", "i4dc", "recon7"))
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
    for name in ("r2_j1_select.py", "r2_j1_cell.py"):                  # NO HAND-WRITTEN KERNEL; no later rung's name
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
        assert not [w for w in LATER_NAMES if w in text], (name, [w for w in LATER_NAMES if w in text])
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names them
        if p.name.startswith("r2_j1_"):
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
    assert all(y[1] == "REPORT" for y in v.values()) and "12.0000" in v["J1-JOINT-SAVING"][2], v
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
