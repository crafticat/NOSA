"""CPU tests of rung I4CC (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the COMPILED
arms L4PTFDGUDC and L4PTFDGUDKWC (torch.compile / Inductor over the joint section's integer / mask / index / bit-copy
helpers), benchmarks/Efficiency/r2_i4cc_compile.py; the cell r2_i4cc_cell.py; the step script r2_i4cc_step.sh. The
world: tests_cpu/i4cc_world.py over i4s_world (FakeGraphsDC: the compiled calls as kernel nodes of the joint capture).
CUDA GRAPHS DO NOT RUN ON THE CPU AND INDUCTOR'S CPU BACKEND IS NOT ITS GPU BACKEND: these tests establish the plumbing,
the bit equality on the CPU backend (the same graphs and fusion decisions, another code generator), no graph break, the
compile-event record and the gates; only the GPU miniature i4cm2 establishes Triton's bits and capture legality, and the
timing cells the saving.

GATE 0    test_i4cc_gate0_files: the tree = I4S 8f236f8 + the I4CC files + ONE module-level declaration block in each
          older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py, test_i4a.py, test_i4g.py,
          test_i4d.py, test_i4k.py, test_i4u.py, test_i4h.py, test_i4s.py) and ONE line of test_i4s's name scan; nosi/ and
          every other file are 8f236f8's bytes; no kernel source; the fork is fia6's blob; the compile module imports
          torch, time and r2_i4s_stack only, no hand-written kernel in the I4CC modules; nobody else names them.
HELPERS   test_helpers_equal_eager: every compiled helper (Inductor, CPU) = the same Python run eagerly, bit for bit, on
          adversarial inputs (duplicate ids, -1 entries, sort-key ties, every variant: pool / no pool, fetch, stats); the
          written buffers keep their addresses (in place) and their unwritten bytes. REPORT: aten ops vs generated kernels.
EQUAL     test_compiled_equals (timing B 8, check B 8, timing B 2 rowgroup, both C arms): the run = L4PTFD BIT FOR BIT and
          = its base arm (L4PTFDGUD / L4PTFDGUDKW) in everything but the compile record; the cell's gates pass on the CPU
          evidence (EQF, LEAN, HOIST, COUNTS, GRAPH, DGRAPH, the keep gates, COMPILE). Preconditions (L34): every section
          through the compiled helpers, replays, every compile event before the first capture.
PLANTS    test_compile_plants_caught: lost_lists (EQF), late (COMPILE: a compile inside a capture), a lever control that
          forces STK's eager section (COMPILE: fallback).
WIRING    test_compile_refusals.
CELL      test_i4cc_cell_arguments, test_i4cc_cell_pure_parts, test_i4cc_registration_state, test_step_script_i4cc.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_i4cc.py
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
from torch.utils._python_dispatch import TorchDispatchMode

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "8f236f872faa2f1e176333220bd896855531c6f0"         # nosi-r2-i4s: I4S with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4cc"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
I4CC_NEW = (E_ + "r2_i4cc_compile.py", E_ + "r2_i4cc_cell.py", E_ + "r2_i4cc_step.sh", T_ + "i4cc_world.py",
            T_ + "test_i4cc.py")
I4CC_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                     "i4h", "i4s"))
I4CC_EDITED = {T_ + "test_i4s.py": "test_i4s_gate0_files"}
COMPILE_IMPORTS = {"torch", "time", "r2_i4s_stack"}
TIMING_KEYS = {"phases", "round_ms", "wall_s", "prelude_ms", "probe_wait_ms", "graph", "dgraph"}
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
# I4DC (nosi-r2-i4dc: I4CC d75a946 + the compiled draft helpers): its files (its own Gate 0: tests_cpu/test_i4dc.py);
# I4CC_LATER: its modules import the I4CC modules (scan (e) skips them)
I4CC_NEW = I4CC_NEW + (E_ + "r2_i4dc_compile.py", E_ + "r2_i4dc_cell.py", E_ + "r2_i4dc_step.sh", T_ + "i4dc_world.py",
                       T_ + "test_i4dc.py")                                                       # I4DC
I4CC_LATER = ("r2_i4dc_compile.py", "r2_i4dc_cell.py")                                          # I4DC
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py); I4CC_LATER: its modules name the I4CC modules (scan (e) skips them)
I4CC_NEW = I4CC_NEW + (E_ + "r2_recon7_protocol.py", E_ + "r2_recon7_probe.py", E_ + "r2_recon7_cell.py",
                       E_ + "r2_recon7_streams.py", E_ + "r2_recon7_mem.py", E_ + "r2_recon7_step.sh",
                       E_ + "r2_i4c_cell.py", E_ + "r2_i4m_cell.py", E_ + "r2_wsm_cell.py", T_ + "test_recon7.py",
                       "docs/NOTES_recon7.md")                                                    # RECON7
I4CC_LATER = I4CC_LATER + ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py",
                           "r2_recon7_streams.py", "r2_recon7_mem.py", "r2_i4c_cell.py", "r2_i4m_cell.py",
                           "r2_wsm_cell.py")                                                      # RECON7
# J1 (nosi-r2-j1: RECON7 cd0f0c5 + the compiled joint pooling of AREA 1): its files
# (its own Gate 0: tests_cpu/test_j1.py)
I4CC_NEW = I4CC_NEW + (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", T_ + "test_j1.py",
                       "docs/NOTES_j1.md")                                                        # J1
I4CC_NEW = I4CC_NEW + (E_ + "r2_j1_forkbench.py",)                                                # J1


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


X = _load("i4cc_world")
TS = _load("test_i4s")
CC, L, N0 = X.CC, X.L, X.N0


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4cc_cell
    return r2_i4cc_cell


# ------------------------------------------------------------------ GATE 0
def test_i4cc_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4CC_NEW) | set(I4CC_DECL), sorted(changed - set(I4CC_NEW) - set(I4CC_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4CC_NEW), [f for f in I4CC_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: 8f236f8's
    for f in I4CC_DECL:                                                   # (c) one I4CC block each (+ one line)
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        allowed = {I4CC_EDITED[f]} if f in I4CC_EDITED else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert len(minus) <= (1 if f in I4CC_EDITED else 0), (f, minus)
        assert any("r2_i4cc_compile.py" in x for x in body) and any("I4CC" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    tree = ast.parse((EFF / "r2_i4cc_compile.py").read_text())           # (d) the compile module
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert {m for m in mods if not m.startswith("torch.")} == COMPILE_IMPORTS, mods
    for name in ("r2_i4cc_compile.py", "r2_i4cc_cell.py"):              # NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4cc_") or p.name in I4CC_LATER:
            continue
        assert "r2_i4cc" not in p.read_text(), p.name


# ------------------------------------------------------------------ the helpers, alone
class _Count(TorchDispatchMode):
    """aten compute ops (views and empty allocations excluded): the eager launches of a helper."""
    SKIP = {"view", "_unsafe_view", "expand", "slice", "select", "permute", "t", "transpose", "reshape", "alias",
            "empty", "empty_like", "empty_strided", "as_strided", "detach", "unsqueeze", "squeeze", "lift_fresh"}

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket.__name__ not in self.SKIP:
            self.n += 1
        return func(*args, **(kwargs or {}))


def _inputs(seed: int, pool: bool):
    """Adversarial helper inputs at world-like shapes: H 2, B 5, R 4, T 8, a window of 8 slots, a pool of 6, cap 6."""
    g = torch.Generator().manual_seed(seed)
    H, B, R, T, Wn, Pn, cap, S, D, NC = 2, 5, 4, 8, 8, 6, 6, 64, 4, 70
    ri = lambda lo, hi, shape: torch.randint(lo, hi, shape, generator=g)   # noqa: E731
    qb = ri(3, 9, (B,))
    sel = ri(-1, 14, (H, B, R, T))
    sel[0, 0, :, :] = sel[0, 0, 0:1, :]                                  # duplicate ids across rows
    sel[1, 1, 2, :] = -1                                                 # a row with no valid id
    bm = ri(-1, 14, (H, B, Wn))
    pm = ri(-1, 14, (H, B, Pn))
    big = torch.tensor(2 ** 31 - 1)
    full = torch.rand(R, B, generator=g) > 0.4
    fullv = full.t().reshape(1, B, R, 1).expand(H, B, R, T).reshape(H, B, R * T).contiguous()
    kb = torch.randn(B, S, H, D, generator=g).to(torch.bfloat16)
    kb.view(torch.int16)[0, 0, 0, 0] = 0x7FC1                             # a NaN payload: must survive bit for bit
    return dict(H=H, B=B, R=R, T=T, cap=cap, qb=qb, sel=sel, bm=bm, pm=pm if pool else None, big=big, fullv=fullv,
                minus1=torch.tensor(-1), cap_t=torch.tensor(cap), eb=ri(100, 200, (1, B, 1)),
                sb=ri(1000, 2000, (1, B, 1)), cb0=ri(5000, 6000, (1, B, 1, 1)), kb=kb,
                vb=torch.randn(B, S, H, D, generator=g).to(torch.bfloat16),
                bb=torch.randn(B, S, H, generator=g).to(torch.bfloat16),
                cis=torch.randn(B, NC, H, generator=g).to(torch.bfloat16), i_all=ri(0, S, (R, B)).to(torch.int32),
                prow=ri(0, NC, (R, B)).to(torch.int32), act=torch.rand(R, B, generator=g) > 0.3,
                k4=torch.randn(B, R, H, D, generator=g).to(torch.bfloat16),
                v4=torch.randn(B, R, H, D, generator=g).to(torch.bfloat16), topk=Wn)


def _chain(I, fns, fetch, stats):
    """The helpers in the section's order on fresh copies of I (fns: name -> callable); the eager sorts between them.
    Returns every output and every written buffer."""
    H, B, R, T, cap = I["H"], I["B"], I["R"], I["T"], I["cap"]
    qb = I["qb"]
    qb3, qb4, qb1_4 = qb.view(1, B, 1), qb.view(1, B, 1, 1), (qb + 1).view(1, B, 1, 1)
    kb, vb, bb = I["kb"].clone(), I["vb"].clone(), I["bb"].clone()
    buf = torch.zeros((H, B, cap + 1), dtype=torch.int64)
    host_x = torch.full((H, B, cap + 1), -1, dtype=torch.int64)
    pkey = torch.zeros((B * R, H, T), dtype=torch.int32)
    pslot = torch.zeros((B * R, H, T), dtype=torch.int32)
    npg = torch.zeros((B * R, H), dtype=torch.int32)
    ptrs = [t.data_ptr() for t in (kb, vb, bb, buf, host_x, pkey, pslot, npg)]
    out = {}
    ti, inj, sel_c, is_q, is_n = fns["pre"](I["sel"], qb3, qb4, qb1_4, I["minus1"])
    out.update(ti=ti, inj=inj, sel_c=sel_c, is_q=is_q, is_n=is_n)
    fns["write"](CC.bits(kb), CC.bits(vb), CC.bits(bb), CC.bits(I["cis"]), torch.arange(B), I["i_all"], I["prow"],
                 I["act"], CC.bits(I["k4"]), CC.bits(I["v4"]))
    v = sel_c.view(H, B, R * T)
    ws, wi = torch.sort(I["bm"], dim=-1)
    ps, pi = torch.sort(I["pm"], dim=-1) if I["pm"] is not None else (None, None)
    l1 = fns["loc1"](v, ws, wi, ps, pi, I["fullv"], I["big"], I["topk"])
    out.update({"l1_%d" % k: t for k, t in enumerate(l1)})
    s = torch.sort(l1[5], dim=-1).values
    l2 = fns["loc2"](s, v, l1[0], l1[1], l1[2], l1[3], l1[4], I["sel"], is_q, is_n, I["eb"], I["sb"] + 1, I["cb0"],
                     I["sb"].view(1, B, 1, 1), buf, host_x, I["minus1"], I["big"], I["cap_t"], cap, 2 ** 31 - 1,
                     fetch, stats)
    out.update({"l2_%d" % k: t for k, t in enumerate(l2)})
    if fetch:
        phys, keys_in = l2[1], l2[2]
    else:
        phys, keys_in = fns["kphys"](is_q, is_n, I["cb0"], I["sb"].view(1, B, 1, 1), l2[1], l2[0], I["sel"], I["big"])
        out.update(kphys=phys, kkeys=keys_in)
    keys, order = torch.sort(keys_in, dim=-1)
    fns["lists"](keys, order, phys, l2[0], pkey, pslot, npg)
    out.update(kb=kb, vb=vb, bb=bb, buf=buf, host_x=host_x, pkey=pkey, pslot=pslot, npg=npg)
    same = ptrs == [t.data_ptr() for t in (kb, vb, bb, buf, host_x, pkey, pslot, npg)]
    return out, same


def test_helpers_equal_eager():
    CC.configure()
    comp = {n: CC.compiled(n, "inductor") for n in CC.HELPERS}
    rows = []
    with torch.inference_mode():
        for seed in range(3):
            for pool in (True, False):
                for fetch in (True, False):
                    for stats in (False, True):
                        I = _inputs(seed, pool)
                        e, se = _chain(I, CC.FNS, fetch, stats)
                        c, sc = _chain(I, comp, fetch, stats)
                        assert se and sc                               # every write in place (the same addresses)
                        assert set(e) == set(c)
                        bad = [k for k in e if not torch.equal(CC.bits(e[k]), CC.bits(c[k]))]   # bf16 NaN too
                        assert bad == [], (seed, pool, fetch, stats, bad)
                        rows.append((seed, pool, fetch, stats))
    I = _inputs(0, True)                                               # REPORT: eager aten ops vs generated kernels
    with torch.inference_mode():
        cnt = _Count()
        with cnt:
            _chain(I, CC.FNS, True, False)
    _ev("I4CC HELPERS: %d variants bit-equal (seed, pool, fetch, stats); eager aten ops of the chain incl. the 4 sorts "
        "%d; dynamo %s" % (len(rows), cnt.n, CC.dynamo_counts()))


# ------------------------------------------------------------------ the arms
@functools.lru_cache(maxsize=None)
def _run(arm, scen="timing", B=8, gemm="natural"):
    return X.run_arm(arm, scen, B=B, gemm=gemm)


CASES = dict(gudc_timing=("GUDC", "timing", 8, "natural"), gudc_check=("GUDC", "check", 8, "natural"),
             gudc_b2=("GUDC", "timing", 2, "rowgroup"), gudkc_timing=("GUDKC", "timing", 8, "natural"),
             gudkc_check=("GUDKC", "check", 8, "natural"), gudkc_b2=("GUDKC", "timing", 2, "rowgroup"))
BASE_OF = dict(GUDC="GUD", GUDKC="GUDK")


@pytest.mark.parametrize("case", list(CASES))
def test_compiled_equals(case):
    arm, scen, B, gemm = CASES[case]
    keep = arm == "GUDKC"
    name = X.CARMS[arm]
    d = _run("D", scen, B, gemm)
    b = _run(BASE_OF[arm], scen, B, gemm)
    x = _run(arm, scen, B, gemm)
    C = _cell()
    rd = x["rd"]
    c, gr, dr = rd["compile"], rd["graph"], rd["dgraph"]
    _ev("I4CC %s %s: rounds %d; compile calls %s, events %s, sections %s, fallback %s, first capture pass %s, dynamo %s; "
        "joint graphs %s; draft replay passes %s" % (
            case, name, rd["rounds"], c["calls"], [(e["name"], e["joint_pass"], e["layer"], e["capture"])
                                                   for e in c["events"]], c["sections"], c["fallback"],
            c["first_capture_pass"], c["dynamo"], (gr["captures"], gr["warm"], gr["replays"], gr["eager"]),
            dr["replay_passes"]))
    assert c["sections"] > 0 and not c["fallback"] and gr["replays"] >= 3 * L and dr["replay_passes"] >= 10, c
    assert all(not e["capture"] and e["joint_pass"] < c["first_capture_pass"] for e in c["events"]), c["events"]
    skip = ("hoist", "keep", "round_blocks", "compile") if keep else ("hoist", "compile")
    ts = ("kid", "kage", "blocks") if keep else ()
    bad = TS._diff(d, x, skip=skip, tensors_skip=ts)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    bad = TS._diff(b, x, skip=("compile",))
    assert bad == [], bad[:10]                                         # = the base arm in everything but the compile
    assert rd["hoist"] == b["rd"]["hoist"] and (rd.get("keep") == b["rd"].get("keep"))
    fd, fx = X.cell_inputs(d, C), X.cell_inputs(x, C)
    keepd = dict(L4PTFD=fd)
    if keep:
        keepd["L4PTFDGUDKW"] = X.cell_inputs(b, C)
    rec = dict(graph=gr, dgraph=dr, gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = C.arm_gates(name, rd, rec, fx, keepd, scen == "check", B, 10, N0, L, 1.0)
    names = [g[0] for g in gs]
    want = ["MEM", "GRAPH", "COUNTS", "DGRAPH", "HOIST"] + (["KEEP"] + (["CAP"] if scen == "check" else []) +
                                                            ["KEEP-EQ", "KEEP-REF"] if keep else []) + ["COMPILE", "EQF"]
    assert names == ["%s-%s" % (name, w) for w in want], names
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert C.IA.lean_gate(name, dict(rd, K=X.K0), L)[2] is True


# ------------------------------------------------------------------ PLANTS
def test_compile_plants_caught():
    C = _cell()
    d = _run("D")
    fd = X.cell_inputs(d, C)
    p = X.run_arm("GUDC", "timing", ccneg="lost_lists")                 # the slots into a copy of vs.pslot
    e = C.S4.eq_gate("L4PTFDGUDC", X.cell_inputs(p, C), fd, N0)
    _ev("I4CC PLANT lost_lists: differs %s; EQF %s" % (TS._diff(d, p, skip=("hoist", "compile"))[:4], e[2]))
    assert TS._diff(d, p, skip=("hoist", "compile")) != [] and e[2] is False    # not a no-op (L34); CAUGHT by EQF
    p = X.run_arm("GUDC", "timing", ccneg="late")                       # a new constant inside the capture
    g = CC.compile_gate("L4PTFDGUDC", p["rd"], L, C.NAMES["L4PTFDGUDC"])
    ev = p["rd"]["compile"]["events"]
    _ev("I4CC PLANT late: events %s; COMPILE %s" % ([(x["name"], x["joint_pass"], x["capture"]) for x in ev], g[2]))
    assert any(x["capture"] for x in ev) and g[2] is False
    p = X.run_arm("GUDC", "timing", hneg="no_plast")                    # a lever control: STK's eager section
    g = CC.compile_gate("L4PTFDGUDC", p["rd"], L, C.NAMES["L4PTFDGUDC"])
    _ev("I4CC PLANT fallback (hneg no_plast): fallback %s; COMPILE %s" % (p["rd"]["compile"]["fallback"], g[2]))
    assert p["rd"]["compile"]["fallback"].get("control", 0) > 0 and g[2] is False


def test_compile_refusals():
    for kw in (dict(ccneg="nope"), dict(cc_backend="cudagraphs")):
        with pytest.raises(X.SR.SpecError):
            X.c_builder("GUDC", **kw)(*X.world(2, 8)[:5])
    with pytest.raises(ValueError):
        CC.compiled("pre", "nope")
    run = X.c_builder("GUDKC")(*X.world(2, 8)[:5])
    assert run.meta["compile"]["arm"] == "L4PTFDGUDKWC" and run.T.ccneg is None and run.T._cc_ok
    assert type(run.T).__name__ == "TentStateFDGUDKC" and run.T.kwin


# ------------------------------------------------------------------ CELL
def test_i4cc_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.check_every, a.probe_every) == (
        ["L4PTFD", "L4PTFDG", "L4PTFDGUD", "L4PTFDGUDC"], 216, 512, 65, 3, 0, 16)
    for bad in (["--arms", "L4PTFD,L4PTFDGUDC"], ["--arms", "L4PTFD,L4PTFDGUDKWC"], ["--arms", "L4PTFDGUDC,L4PTFD"],
                ["--arms", "L4PTFD,L4PTFDGUDC,L4PTFDGUD"], ["--arms", "L4PTFD,L4PTFDGUDX"], ["--spec", "4"],
                ["--arms", "L4PTFD,L4PTFKW", "--check-every", "1", "--batch", "216"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    for arms in ("L4PTFD,L4PTFKW,L4PTFDGUD,L4PTFDGUDKW,L4PTFDGUDC,L4PTFDGUDKWC", "L4PTFD,L4PTFDG,L4PTFDGUD,L4PTFDGUDC",
                 "L4PTFD,L4PTFDGUDKW,L4PTFDGUDKWC"):
        assert C.parse_args(base + ["--arms", arms]).arms == arms.split(",")
    assert C.runner_choice("L4PTFDGUDC", False) == ("SC", dict(cc_backend="inductor"))
    assert C.runner_choice("L4PTFDGUDKWC", True) == ("SKC", dict(keep=True, kwin=True, kcheck=True,
                                                                 cc_backend="inductor"))
    assert C.runner_choice("L4PTFDGUD", False) == ("S", {})
    assert C.runner_class("SC", X.STF) is X.CCL["L4PTFDGUDC"][1]
    assert C.runner_class("SKC", X.STF) is X.CCL["L4PTFDGUDKWC"][1]
    assert C.runner_class("S", X.STF) is X.XS.CLS["L4PTFDGUD"][1]
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused
    assert C.keep_ref_arm("L4PTFDGUDKWC", {"L4PTFD": 1, "L4PTFDGUDKW": 1}) == "L4PTFDGUDKW"
    assert C.keep_ref_arm("L4PTFDGUDKWC", {"L4PTFD": 1, "L4PTFKW": 1, "L4PTFDGUDKW": 1}) == "L4PTFKW"
    assert C.keep_ref_arm("L4PTFDGUDKW", {"L4PTFD": 1, "L4PTFDGUDKW": 1}) is None


def test_i4cc_cell_pure_parts():
    C = _cell()
    d, b, x = _run("D"), _run("GUDK"), _run("GUDKC")
    fd, fb, fx = X.cell_inputs(d, C), X.cell_inputs(b, C), X.cell_inputs(x, C)
    rec = dict(graph=b["rd"]["graph"], dgraph=b["rd"]["dgraph"], gpu_peak_alloc_GB=1.0,
               mem_arena=dict(prefill_peak_GB=1.0))
    gs = C.arm_gates("L4PTFDGUDKW", b["rd"], rec, fb, dict(L4PTFD=fd), False, 8, 10, N0, L, 1.0)
    kq = next(g for g in gs if g[0] == "L4PTFDGUDKW-KEEP-EQ")
    assert kq[1] is False and "REPORT" in kq[3]                        # no keep reference in i4ccb: REPORT only
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    rd = x["rd"]
    good = CC.compile_gate("L4PTFDGUDKWC", rd, L, C.NAMES["L4PTFDGUDKWC"])
    assert good[2] is True, good
    for plant in (dict(events=rd["compile"]["events"] + [dict(name="pre", graphs=1, s=1.0, layer=0, joint_pass=99,
                                                              capture=False)]),   # a late compile
                  dict(fallback=dict(record=1)), dict(sections=rd["compile"]["sections"] - 1),
                  dict(calls=dict(rd["compile"]["calls"], kphys=0)), dict(dynamo=dict(graph_breaks=1)),
                  dict(dtype_ok=False)):
        r2 = dict(rd, compile=dict(rd["compile"], **plant))
        assert CC.compile_gate("L4PTFDGUDKWC", r2, L, C.NAMES["L4PTFDGUDKWC"])[2] is False, plant
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh2, {}, 216)[-1] == ("I4C2-RUN", "NOT_RUN", "no C arm ran")
    mk = lambda tok, dr_, jt, cs: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr_, joint=jt),  # noqa
                                       vs_strongest=0.5, gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                       graph=dict(pool_bytes=50e6, capture_s=3.0), dgraph=dict(dcapture_s=4.0),
                                       cc=dict(compile_s=cs))
    sp = dict(L4PTFD=mk(400.0, 140.0, 200.0, None), L4PTFDGUD=mk(420.0, 136.0, 189.0, None),
              L4PTFDGUDC=mk(430.0, 136.0, 181.0, 42.0))
    saved = {k: dict(v) for k, v in C.PRED_I4C2_BY_B.items()}
    try:
        for k in C.PRED_I4C2_BY_B:
            C.PRED_I4C2_BY_B[k].update({kk: None for kk in C.PRED_KEYS})
        v = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
        C.PRED_I4C2_BY_B[216].update(I4C2_vs_ref=(1.03, 1.05), I4C2_joint_saving=(3, 12), I4C2_above_best=True)
        w = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
    finally:
        for k in saved:
            C.PRED_I4C2_BY_B[k].clear()
            C.PRED_I4C2_BY_B[k].update(saved[k])
    assert all(y[1] == "REPORT" for y in v.values()), v
    assert "8.0000" in v["I4C2-JOINT-SAVING"][2] and "42.0000" in v["I4C2-COMPILE-S"][2] and "1.0238" in v[
        "I4C2-VS-REF"][2] and "7.0000" in v["I4C2-CAPTURE-S"][2], v
    assert "X = L4PTFDGUDC, REF = L4PTFDGUD" in v["I4C2-ARM"][2]
    assert w["I4C2-VS-REF"][1] == "REFUTED" and w["I4C2-JOINT-SAVING"][1] == "CONFIRMED" and w["I4C2-ABOVE-BEST"][
        1] == "CONFIRMED"
    assert C.c_arm(dict(sp, L4PTFDGUDKWC=1)) == "L4PTFDGUDKWC" and C.c_arm(dict(L4PTFD=1)) is None
    r = dict(graph=dict(capture_s=2.0), dgraph=dict(dcapture_s=1.0), compile=dict(compile_s=7.0), wall_s=20.0,
             count=[50, 50])
    assert C.effective_wo_setup({}, r) == 10.0


def test_i4cc_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4C2 is None:                                       # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4C2_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4C2)
    assert set(C.PRED_I4C2_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4C2_BY_B.values())
    src = (EFF / "r2_i4cc_cell.py").read_text()
    assert ("\nPRED_STAMP_I4C2 = None\n" in src) == (C.PRED_STAMP_I4C2 is None)   # the step script's grep sees it


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4cc_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("i4cm2", 10, 8, 72, 1, 1, 40, "100G", "L4PTFD,L4PTFKW,L4PTFDGUD,L4PTFDGUDKW,L4PTFDGUDC,L4PTFDGUDKWC"),
         ("i4cca", 65, 216, 512, 0, 0, 130, "240G", "L4PTFD,L4PTFDG,L4PTFDGUD,L4PTFDGUDC"),
         ("i4ccb", 10, 318, 512, 0, 0, 160, "340G", "L4PTFD,L4PTFDGUDKW,L4PTFDGUDKWC"))


def test_step_script_i4cc():
    src = (EFF / "r2_i4cc_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4cc_jit " in src          # the caches: never in HOME
    for env in ('TORCHINDUCTOR_CACHE_DIR="$CCJIT/inductor"', 'TRITON_CACHE_DIR="$CCJIT/triton"',
                'TRITON_HOME="$CCJIT/triton_home"'):
        assert env in src, env
    for f in ("r2_i4cc_compile.py", "r2_i4s_cell.py", "r2_i4s_stack.py", "r2_i4h_graph.py", "r2_i4h_cell.py",
              "r2_i4k_combo.py", "r2_i4d_keep.py", "r2_i4u_hoist.py", "r2_i4g_graph.py", "r2_i4a_lean.py"):
        assert "benchmarks/Efficiency/%s" % f in src, f
    assert "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4sm", "i4cc", "i4ccc", "i4cm"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4C2 = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4cc_cell.py" % sha).stdout
    ran = 0
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if cell in ("i4cca", "i4ccb") and not registered:              # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4C2 is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.stderr[-400:])
        ran += 1
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 fork-store 0 check %d stress %d STEP_MIN %d mem %s "
                "claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s compile caches "
                "/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4cc_jit/i4cc_t1_%s_%s" % (
                    cell, 63 + pool, pool, batch, arms, steps, check, stress, smin, mem, HOLDER, HOLDER, cell))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4cc_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "TORCHINDUCTOR_CACHE_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4cc_jit/i4cc_t1_%s_%s/inductor"
                     % (HOLDER, cell), "i4cc_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms %s " % arms.replace(",", r"\\?,"), r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == arms.split(",")                               # every cell's arguments parse
    assert ran >= 1
    _ev("I4CC DRYRUN: %d cell(s) printed their step command (registered %s)" % (ran, registered))
