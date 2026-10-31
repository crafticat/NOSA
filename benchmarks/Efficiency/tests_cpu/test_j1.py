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
SELECT_IMPORTS = {"torch", "time", "r2_i4dc_compile"}
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
ARMS = dict(GUDCDJ1="L4PTFDGUDCDJ1", GUDKCDJ1="L4PTFDGUDKWCDJ1")
BASE = dict(GUDCDJ1="GUDCD", GUDKCDJ1="GUDKCD")
C_OF = dict(GUDCD="GUDC", GUDKCD="GUDKC")


def j1_builder(arm, check=False, stats=False, kcheck=False, phase_marks=False, probe_every=0, j1neg=None):
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
                  cc_backend="inductor", ccneg=None, cdneg=None, j1neg=j1neg)
        if arm == "GUDKCDJ1":
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=None, kslots=None, cneg=None)
        be = X.FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def run_j1(arm, scen="timing", B=8, j1neg=None):
    CC.configure()
    s = X.SCEN[scen]
    N = N0
    model, c, S, hz, tok0 = X.world(B, N, 0)
    kc = bool(s["check"]) and arm == "GUDKCDJ1"
    run = j1_builder(arm, check=s["check"], stats=s["stats"], kcheck=kc, phase_marks=s["phase_marks"],
                     probe_every=s["probe_every"], j1neg=j1neg)(model, c, S, hz, tok0, check_every=s["check_every"])
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
    _ev("J1 %s %s: rounds %d; joint passes %d; J1 %s" % (case, name, rd["rounds"], jp, j))
    assert jp > 0 and j["calls"] == jp * L and not j["fallback"] and j["peq_layers"] > 0      # L34 preconditions
    assert j["peq_bad"] == 0 and j["peq_vals"] > 0 and len(j["events"]) <= 1
    bad = TS._diff(b, x, skip=("compile", "j1"))
    assert bad == [], bad[:10]                                         # = its I4DC arm in everything but the records
    skip = ("hoist", "keep", "round_blocks", "compile", "j1") if keep else ("hoist", "compile", "j1")
    ts = ("kid", "kage", "blocks") if keep else ()
    bad = TS._diff(d, x, skip=skip, tensors_skip=ts)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    C, D = _cell()
    fd, fb, fx = X.cell_inputs(d, C=D), X.cell_inputs(b, C=D), X.cell_inputs(x, C=D)
    keepd = dict(L4PTFD=fd, **({"L4PTFDGUDKWCD": fb} if keep else {}))
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = D.arm_gates(name, rd, rec, fx, keepd, scen == "check", 8, 10, N0, L, 1.0)
    names = [g[0] for g in gs]
    want = ["MEM", "GRAPH", "COUNTS", "DGRAPH", "HOIST"] + (["KEEP"] + (["CAP"] if scen == "check" else []) +
                                                            ["KEEP-EQ", "KEEP-REF"] if keep else []) + [
        "COMPILE", "EQF", "J1-PEQ", "J1-COMPILE"]
    assert names == ["%s-%s" % (name, w) for w in want], names
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert rec["j1"] == j                                              # the summary carries the J1 record
    if keep:
        kq = next(g for g in gs if g[0].endswith("-KEEP-EQ"))
        assert kq[1] is True and "L4PTFDGUDKWCD" in kq[3], kq
    for plant in (dict(peq_bad=1), dict(peq_layers=0), dict(fallback=dict(record=1)), dict(calls=j["calls"] - 1),
                  dict(events=[dict(graphs=1, s=1.0, layer=0, capture=True, joint_pass=0)]),
                  dict(events=[dict(graphs=1, s=1.0, layer=0, capture=False, joint_pass=0)] * 2)):
        g = J1.j1_gates(name, dict(rd, j1=dict(j, **plant)), L)
        assert not (g[0][2] and g[1][2]), plant


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
    _ev("J1 PLANT wrong_req: differs %s; EQF %s; PEQ bad %s of %s" % (df[:4], e[2], j["peq_bad"], j["peq_vals"]))
    assert df != [] and e[2] is False and j["peq_bad"] > 0             # not a no-op (L34); CAUGHT by EQF and PEQ
    J1 = _mod("r2_j1_select")
    assert J1.j1_gates("L4PTFDGUDCDJ1", p["rd"], L)[0][2] is False
    with pytest.raises(X.SR.SpecError):
        j1_builder("GUDCDJ1", j1neg="nope")(*X.world(2, 8)[:5])


# ------------------------------------------------------------------ CELL
def test_j1_cell_tables():
    C, D = _cell()
    I4DC = _mod("r2_i4dc_cell")
    assert D is not I4DC and D.__name__ == C.DCC_NAME
    assert os.path.realpath(D.__file__) == os.path.realpath(I4DC.__file__)   # the same file, a second module object
    assert "L4PTFDGUDCDJ1" not in I4DC.ARMS and "L4PTFDGUDCDJ1" not in I4DC.BASE_D       # the import: unchanged
    assert D.ARMS == I4DC.ARMS + C.ARMS_J1 and D.BASE_D["L4PTFDGUDKWCDJ1"] == "L4PTFDGUDKWCD"
    assert D.ROOT_ARM["L4PTFDGUDCDJ1"] == "L4PTFDGUD" and D.ROOT_ARM["L4PTFDGUDKWCDJ1"] == "L4PTFDGUDKW"
    assert D.NAMES["L4PTFDGUDCDJ1"] == I4DC.NAMES["L4PTFDGUDCD"] and "L4PTFDGUDKWCDJ1" in D.KEEP_ARMS
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


CELLS = (("j1m", 10, 8, 72, 1, 1, 30, "100G",
          "L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1"),
         ("j1a", 65, 216, 512, 0, 0, 110, "240G", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1"),
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
