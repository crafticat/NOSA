"""CPU tests of rung I4K (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the combined
arm L4PTFDGKW = L4PTFD + the store keep + the window loads from the kept store (I4D) + the per-layer CUDA graphs of the
joint pass's section (I4G), with the keep's bookkeeping inside the graphs (benchmarks/Efficiency/r2_i4k_combo.py; the
cell r2_i4k_cell.py; the step script r2_i4k_step.sh). The world: tests_cpu/i4k_world.py over i4g_world (FakeGraphs: a
CUDA graph stand-in that captures without executing and replays the recorded ops on the captured tensors). CUDA GRAPHS
DO NOT RUN ON THE CPU: these tests establish the plumbing (the device step, the counter rows, the persistent map, the
keep's residency stub, the eager parts, the counts); only the GPU miniature (i4km) establishes capture legality of the
keep's ops on the device (sort, the broadcast compare, scatter with an expanded source), the replay's bits on the GPU,
timing and memory.

GATE 0    test_i4k_gate0_files: the tree = I4A's acbe433 + I4D's 99c231e files + I4G's d7997e6 files + the I4K files;
          in each older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py, test_i4a.py) only
          added declaration blocks (I4D, I4G, I4K; test_i4a.py also ONE line of test_i4a_gate0_files); test_i4d.py /
          test_i4g.py = their tips plus ONE declaration block and ONE line of their Gate-0 test each; every other I4D /
          I4G file and nosi/ = their tips' bytes; the fork is fia6's blob; the combo module imports torch, r2_i4d_keep and
          r2_i4g_graph only, no "triton"; nobody else names it.
STEP      test_keep_assign_d_equals_keep_assign: the device-step keep_assign_d = r2_i4d_keep.keep_assign (every output,
          the map, the ages, the check counters) on random sequences, with the controls, a blocked slot set and stats.
EAGER     test_graph_safe_keep_equals_keep: TentRunnerFKg (the graph-safe keep, eager) = I4D's TentRunnerFK bit for bit
          (kwin on and off; timing; check with acct and the byte checks), the keep record per pass included.
EQUAL     test_l4ptfdgkw_equals (timing B 8, check B 8 with acct, timing B 2 rowgroup, pressure B 8, no-kwin B 8): the
          combined arm = L4PTFD bit for bit (tokens, fingerprints, read() but its keep / meter / graph parts, every
          engine / twin / buffer tensor, the union stats), = L4PTFKW in its keep record per pass and its counts; the
          cell's EQF, COUNTS, GRAPH, KEEP, CAP, ACCT, ACCT-EQ, KEEP-EQ pass on the CPU evidence. Preconditions (L34): L
          graphs, replays after the capture pass, store hits and window hits > 0, alive victims under pressure.
PLANTS    test_combo_plants_caught: host_t (the captured host step: KEEP-EQ), pass_row (the counter row of the capture
          pass: KEEP-EQ), resid_in_graph (no stub for the keep's residency: GRAPH or EQF or a refused capture),
          stale_input of I4G on the combined arm (EQF).
COUNT     test_i4k_eager_ops: an INDEPENDENT count of the eager launches of a replayed joint pass: L4PTFDGKW - L4PTFDG =
          L x KEEP_WIN_EXTRA_OPS + EAGER_PASS_OPS (the keep's other ops are replayed).
WIRING    test_combo_refusals_and_no_host_sync.
CELL      test_i4k_cell_arguments, test_i4k_cell_pure_parts, test_i4k_registration_state, test_step_script_i4k.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4k.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import random
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
I4A_TIP = "acbe433da7f07db7f7b08791c036372d6d8db885"         # nosi-r2-i4a: rung I4A with the lead's registration
I4D_TIP = "99c231ecf4b4b110293f6a5540007ba0742748ee"         # nosi-r2-i4d: rung I4D with the lead's registration
I4G_TIP = "d7997e688f89b5e6d22b2ca4d2d4472679c69a0b"         # nosi-r2-i4g: rung I4G with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4k"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4K_NEW = ("benchmarks/Efficiency/r2_i4k_combo.py", "benchmarks/Efficiency/r2_i4k_cell.py",
           "benchmarks/Efficiency/r2_i4k_step.sh", T_ + "i4k_world.py", T_ + "test_i4k.py")
I4D_FILES = ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
             "benchmarks/Efficiency/r2_i4d_step.sh")
I4G_FILES = ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
             "benchmarks/Efficiency/r2_i4g_step.sh", T_ + "i4g_world.py", "docs/NOTES_i4g.md")
I4K_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py",
            T_ + "test_i4a.py")
I4K_EDITED = {T_ + "test_i4d.py": (I4D_TIP, "test_i4d_gate0_files", '-        if p.name.startswith("r2_i4d_"):'),
              T_ + "test_i4g.py": (I4G_TIP, "test_i4g_gate0_files", '-        if p.name.startswith("r2_i4g_"):')}
TIMING_KEYS = {"phases", "round_ms", "wall_s", "prelude_ms", "probe_wait_ms", "graph"}
# I4S (nosi-r2-i4s: I4K 70059e5 + I4U 832dca5 + I4H fc6c680 + the stacked arms): the later files in this tree (their
# own Gate 0: tests_cpu/test_i4u.py, test_i4h.py, test_i4s.py); I4K_LATER: the later modules that import the combo
# module (scan (g) skips them)
I4K_NEW = I4K_NEW + ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
                     "benchmarks/Efficiency/r2_i4u_step.sh", T_ + "i4u_world.py", T_ + "test_i4u.py",
                     "benchmarks/Efficiency/r2_i4h_graph.py", "benchmarks/Efficiency/r2_i4h_cell.py",
                     "benchmarks/Efficiency/r2_i4h_step.sh", T_ + "i4h_world.py", T_ + "test_i4h.py",
                     "docs/NOTES_i4h.md", "benchmarks/Efficiency/r2_i4s_stack.py",
                     "benchmarks/Efficiency/r2_i4s_cell.py", "benchmarks/Efficiency/r2_i4s_step.sh",
                     T_ + "i4s_world.py", T_ + "test_i4s.py")                                    # I4S
I4K_LATER = ("r2_i4s_stack.py", "r2_i4s_cell.py")                                                # I4S
# I4CC (nosi-r2-i4cc: I4S 8f236f8 + the compiled arms): its files (its own Gate 0: tests_cpu/test_i4cc.py); its modules
# name the I4S modules only
I4K_NEW = I4K_NEW + ("benchmarks/Efficiency/r2_i4cc_compile.py", "benchmarks/Efficiency/r2_i4cc_cell.py",
                     "benchmarks/Efficiency/r2_i4cc_step.sh", T_ + "i4cc_world.py", T_ + "test_i4cc.py")  # I4CC
# I4DC (nosi-r2-i4dc: I4CC d75a946 + the compiled draft helpers): its files (its own Gate 0: tests_cpu/test_i4dc.py);
# I4K_LATER: its modules may name the I4K modules (scan (f) skips them)
I4K_NEW = I4K_NEW + ("benchmarks/Efficiency/r2_i4dc_compile.py", "benchmarks/Efficiency/r2_i4dc_cell.py",
                     "benchmarks/Efficiency/r2_i4dc_step.sh", T_ + "i4dc_world.py", T_ + "test_i4dc.py")  # I4DC
I4K_LATER = I4K_LATER + ("r2_i4dc_compile.py", "r2_i4dc_cell.py")                              # I4DC
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py); I4K_LATER: its modules and copies may name the I4K modules
I4K_NEW = I4K_NEW + ("benchmarks/Efficiency/r2_recon7_protocol.py", "benchmarks/Efficiency/r2_recon7_probe.py",
                     "benchmarks/Efficiency/r2_recon7_cell.py", "benchmarks/Efficiency/r2_recon7_streams.py",
                     "benchmarks/Efficiency/r2_recon7_mem.py", "benchmarks/Efficiency/r2_recon7_step.sh",
                     "benchmarks/Efficiency/r2_i4c_cell.py", "benchmarks/Efficiency/r2_i4m_cell.py",
                     "benchmarks/Efficiency/r2_wsm_cell.py", T_ + "test_recon7.py", "docs/NOTES_recon7.md")  # RECON7
I4K_LATER = I4K_LATER + ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py", "r2_recon7_streams.py",
                         "r2_recon7_mem.py", "r2_i4c_cell.py", "r2_i4m_cell.py", "r2_wsm_cell.py")       # RECON7


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


X = _load("i4k_world")
G4, KP, CB, GR, W, RW, VR = X.G4, X.KP, X.CB, X.GR, X.W, X.RW, X.VR
L, N0 = X.L, X.N0
BIG = VR.FORK_BIG


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4k_cell
    return r2_i4k_cell


def _units(src: str) -> dict:
    """{name: ast.dump}: top-level functions, class methods (Class.method), the other statements (#i)."""
    out, k = {}, 0
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                out["%s.%s" % (n.name, getattr(m, "name", "#"))] = ast.dump(m)
        else:
            out["#%d" % k] = ast.dump(n)
            k += 1
    return out


def _fn_units(sha, f):
    return {k: v for k, v in _units(_git("show", "%s:%s" % (sha, f)).stdout).items() if not k.startswith("#")}


# ------------------------------------------------------------------ GATE 0
def test_i4k_gate0_files():
    if not (_have(I4A_TIP) and _have(I4D_TIP) and _have(I4G_TIP)):
        pytest.skip("no git or no commits %s / %s / %s in %s" % (I4A_TIP, I4D_TIP, I4G_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I4A_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    allowed = set(I4K_NEW) | set(I4D_FILES) | set(I4G_FILES) | set(I4K_DECL) | set(I4K_EDITED)
    assert changed <= allowed, sorted(changed - allowed)                                 # (a)
    assert all((ROOT / f).is_file() for f in I4K_NEW), [f for f in I4K_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I4A_TIP, "--", "nosi").stdout.strip() == ""       # (b) nosi: acbe433's bytes
    assert _git("diff", "--name-only", I4D_TIP, "--", *I4D_FILES).stdout.strip() == ""   # (c) the levers' bytes
    assert _git("diff", "--name-only", I4G_TIP, "--", *I4G_FILES).stdout.strip() == ""
    for f in I4K_DECL:                                                    # (d) added declaration blocks only
        old, new = _fn_units(I4A_TIP, f), {k: v for k, v in _units((ROOT / f).read_text()).items()
                                           if not k.startswith("#")}
        allowed_fn = {"test_i4a_gate0_files"} if f.endswith("test_i4a.py") else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} == allowed_fn, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I4A_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert minus == (['-        if p.name.startswith("r2_i4a_"):'] if f.endswith("test_i4a.py") else []), (f, minus)
        for word in ("r2_i4d_keep.py", "r2_i4g_graph.py", "r2_i4k_combo.py", "I4K"):
            assert any(word in x for x in body), (f, word)
    for f, (tip, fn, line) in I4K_EDITED.items():                         # (e) one block + one line of Gate 0
        old, new = _fn_units(tip, f), {k: v for k, v in _units((ROOT / f).read_text()).items()
                                       if not k.startswith("#")}
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} == {fn}, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", tip, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert [x for x in body if x.startswith("-")] == [line], (f, body[:6])
        assert any("r2_i4k_combo.py" in x for x in body) and any("I4K" in x for x in body), (f, body[:6])
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4k_combo.py").read_text()                          # (f) the combo module: no kernel
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch", "r2_i4d_keep", "r2_i4g_graph"}, mods
    for name in ("r2_i4k_combo.py", "r2_i4k_cell.py"):                   # NO NEW KERNEL
        assert "triton" not in (EFF / name).read_text(), name
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (g) nobody else names it
        if p.name.startswith("r2_i4k_") or p.name in I4K_LATER:
            continue
        assert "r2_i4k" not in p.read_text(), p.name


# ------------------------------------------------------------------ the device step = the host step
def _lst(rows, cap):
    H, B = len(rows), len(rows[0])
    out = torch.full((H, B, cap), BIG, dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            v = sorted(set(rows[h][b]))
            out[h, b, :len(v)] = torch.tensor(v, dtype=torch.int64)
    return out


def _chk():
    return {k: torch.zeros((), dtype=torch.int64) for k in KP.CHECK_KEYS}


@pytest.mark.parametrize("seed,cap,usable,L_,idmax,kneg", [(0, 8, None, 3, 14, None), (1, 6, 4, 4, 9, None),
                                                           (2, 5, None, 2, 40, "layer_blind"),
                                                           (3, 6, None, 3, 12, "evict_used"), (4, 12, 7, 3, 20, None)])
def test_keep_assign_d_equals_keep_assign(seed, cap, usable, L_, idmax, kneg):
    rnd = random.Random(seed)
    H, B = 2, 3
    lim = cap if usable is None else usable
    maps = [KP.new_map(H, B, cap + 1, "cpu") for _ in range(2)]
    block = None if usable is None else (torch.arange(cap) >= usable).view(1, 1, cap)
    chks = [_chk(), _chk()]
    evictions = 0
    for g in range(8):
        for l in range(L_):
            t = g * L_ + l
            rows = [[rnd.sample(range(idmax), rnd.randint(0, lim)) for _ in range(B)] for _ in range(H)]
            lst = _lst(rows, cap)
            a = KP.keep_assign(maps[0][0], maps[0][1], lst, l, t, L_, kneg=kneg, block=block, chk=chks[0], stats=True,
                               big=BIG)
            td = torch.tensor(g, dtype=torch.int64) * L_ + l                    # as the combo state computes it
            b = CB.keep_assign_d(maps[1][0], maps[1][1], lst, l, td, L_, kneg=kneg, block=block, chk=chks[1],
                                 stats=True, big=BIG)
            assert set(a) == set(b)
            for k in a:
                assert torch.equal(a[k], b[k]), (g, l, k)
            assert torch.equal(maps[0][0], maps[1][0]) and torch.equal(maps[0][1], maps[1][1]), (g, l)
            evictions += int(a["victims_alive"])
    assert {k: int(v) for k, v in chks[0].items()} == {k: int(v) for k, v in chks[1].items()}
    assert int((maps[0][0] >= 0).sum()) > 0                               # precondition (L34): the map holds keys
    _ev("I4K STEP seed %d kneg %s: keep_assign_d = keep_assign over 8 passes x %d layers; alive victims %d; check %s" % (
        seed, kneg, L_, evictions, {k: int(v) for k, v in chks[1].items()}))


# ------------------------------------------------------------------ the graph-safe keep (eager) = I4D's keep
def _tensors(run):
    """Every state tensor of a run (as test_i4g._tensors) plus the keep's map when the run has one."""
    T, rs, vs = run.T, run.rs, run.vs
    out = dict(out=run.out, fmax=run.fmax, fsum=run.fsum, fp_ok=run.fp_ok, tpos=T.tpos, qc=T.qc, ctr_c=T.ctr_c,
               skipped=T.skipped, forced=T.forced, injected=T.injected, blocks=run.meter.blocks, fout=vs.fout)
    out.update({"c." + k: v for k, v in T.c.items()})
    out.update({"stat." + k: v for k, v in vs.stat.items()})
    for l, lay in enumerate(rs.cache.layers):
        e, tw = lay.cache_engine, rs.st.twins[l]
        out.update({"%s%d" % (n, l): t for n, t in (
            ("bk", T.k[l]), ("bv", T.v[l]), ("bb", T.bias[l]), ("map", e._block_map), ("pmap", e._pool_map),
            ("page", e._pool_age), ("kg", e._k_gpu), ("vg", e._v_gpu), ("bg", e._kv_bias_gpu), ("len", e._cache_lens),
            ("pos", tw.pos), ("ncomp", tw.n_comp), ("nccis", tw.n_ccis), ("compk", tw.comp_k), ("ring", tw.ring),
            ("tcis", tw.total_cis), ("ccis", tw.ccis))})
    if getattr(T, "kid", None) is not None:
        out.update(kid=T.kid[..., 1:], kage=T.kage[..., 1:])              # column 0: a write target only (never read)
    return out


def _diff(a, b, skip=(), tensors_skip=()):
    bad = [k for k in sorted(set(a["rd"]) | set(b["rd"])) if k not in TIMING_KEYS and k not in skip
           and a["rd"].get(k) != b["rd"].get(k)]
    ta, tb = _tensors(a["runner"]), _tensors(b["runner"])
    for k in sorted(set(ta) | set(tb)):
        if k in tensors_skip:
            continue
        if k not in ta or k not in tb or not torch.equal(W.bits(ta[k]), W.bits(tb[k])):
            bad.append(k)
    return bad


@functools.lru_cache(maxsize=None)
def _run(arm, scen="timing", B=8, gemm="natural", acct=False, kslots=None, cneg=None, gneg=None):
    return X.run_arm(arm, scen, B=B, gemm=gemm, acct=acct, kslots=kslots, cneg=cneg, gneg=gneg)


@pytest.mark.parametrize("arm,ref,scen,acct", [("FKWg", "FKW", "timing", False), ("FKWg", "FKW", "check", True)])
def test_graph_safe_keep_equals_keep(arm, ref, scen, acct):
    a, b = _run(ref, scen, acct=acct), _run(arm, scen, acct=acct)
    kp = b["rd"]["keep"]
    assert kp["hits_total"] > 0 and kp["win_hits_total"] > 0                 # preconditions (L34)
    bad = _diff(a, b, skip=("keep",))
    assert bad == [], bad[:10]                                               # the same run, map included
    ka, kb = a["rd"]["keep"], kp
    assert ka == kb, sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))   # the keep record per pass
    _ev("I4K EAGER %s %s acct %s: = %s bit for bit; passes %d listed %d hits %d window hits %d check %s" % (
        arm, scen, acct, ref, kp["passes"], kp["listed_total"], kp["hits_total"], kp["win_hits_total"], kp["check"]))


# ------------------------------------------------------------------ L4PTFDGKW = L4PTFD (= L4PTFKW in the keep)
CASES = dict(timing=("timing", 8, "natural", False, None, "FDGKW", "FKW"),
             check=("check", 8, "natural", True, None, "FDGKW", "FKW"),
             timing_b2=("timing", 2, "rowgroup", False, None, "FDGKW", "FKW"),
             pressure=("timing", 8, "natural", True, "most", "FDGKW", "FKW"),
             nokwin=("timing", 8, "natural", False, None, "FDGK", "FK"))


def _kslots(spec, B, gemm):
    if spec != "most":
        return spec
    return max(1, int(_run("FD", "check", B=B, gemm=gemm)["rd"]["union"]["extras_max"]))


@pytest.mark.parametrize("case", list(CASES))
def test_l4ptfdgkw_equals(case):
    scen, B, gemm, acct, ks, arm, ref = CASES[case]
    ks = _kslots(ks, B, gemm)
    d = _run("FD", scen, B=B, gemm=gemm)
    k = _run(ref, scen, B=B, gemm=gemm, acct=acct, kslots=ks)
    x = _run(arm, scen, B=B, gemm=gemm, acct=acct, kslots=ks)
    C = _cell()
    gr, kp = x["rd"]["graph"], x["rd"]["keep"]
    jp = x["rd"]["tent"]["calls"]["joint_passes"]
    _ev("I4K %s %s: rounds %d, joint passes %d; graphs %s, warm %s, replays %s, eager %s %s; keep listed %d hits %d "
        "window hits %s victims alive %s; check %s" % (
            case, arm, x["rd"]["rounds"], jp, gr["captures"], gr["warm"], gr["replays"], gr["eager"], gr["eager_why"],
            kp["listed_total"], kp["hits_total"], kp.get("win_hits_total"), kp.get("victims_alive_total"), kp["check"]))
    assert gr["captures"] == L and gr["warm"] == L and gr["replays"] >= 3 * L, gr      # preconditions (L34)
    assert jp >= 5 and gr["deltas"] and not gr["delta_bad"], gr
    assert kp["hits_total"] > 0 and (arm == "FDGK" or kp["win_hits_total"] > 0), kp
    if ks is not None:
        assert kp["slots"] == ks and kp["victims_alive_total"] > 0, kp              # pressure: alive slots evicted
    if scen == "timing":
        assert gr["eager_why"].get("marks", 0) >= L and max(gr["capture_passes"]) < 4 <= jp - 2, gr
    else:
        assert gr["eager_why"].get("probe_F") == 1 and kp["check"]["bytes_checked"] == kp["listed_total"], (gr, kp)
    bad = _diff(d, x, skip=("keep", "round_blocks"), tensors_skip=("kid", "kage", "blocks"))
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    bad = _diff(k, x)
    assert bad == [], bad[:10]                                         # = L4PTFKW in everything (map, meter, keep)
    fd, fk, fx = X.cell_inputs(d, C), X.cell_inputs(k, C), X.cell_inputs(x, C)
    name = "L4PTFDGKW"
    gates = [C.eq_gate(name, fx, fd, N0), C.counts_gate(name, fx["counts"], fk["counts"], "L4PTFKW"),
             C.counts_gate(name, fx["counts"], fd["counts"], "L4PTFD", skip=("round_blocks",)),
             C.GC.graph_gate(name, x["rd"], L), C.keep_eq_gate(name, fx["keep"], fk["keep"]),
             C.DC.keep_gate(name, x["rd"], scen == "check")]
    if scen == "check":
        gates.append(C.DC.cap_gate(name, x["rd"]))
    if acct:
        gates.append(C.DC.acct_gate(name, x["rd"]))
    for g in gates:
        assert g[2] is True, g
    assert C.counts_gate(name, fx["counts"], fd["counts"], "L4PTFD")[2] is (
        fx["counts"]["round_blocks"] == fd["counts"]["round_blocks"])           # the meter: the keep fetches less
    lg = C.A4.lean_gate(name, dict(x["rd"], K=X.K0), L)
    assert lg[2], lg
    if acct and scen == "check":                                       # ACCT-EQ against an accounting L4PTFD
        da = _run("FDa", scen, B=B, gemm=gemm, acct=True)
        assert _diff(d, da, skip=("keep",)) == []                      # the accounting L4PTFD = L4PTFD
        g = C.DC.acct_eq_gate(da["rd"]["keep"], x["rd"]["keep"], name)
        assert g[0] == "ACCT-EQ-L4PTFDGKW" and g[2] is True, g


# ------------------------------------------------------------------ PLANTS
def test_combo_plants_caught():
    C = _cell()
    ks = _kslots("most", 8, "natural")
    k = _run("FKW", "timing", acct=True, kslots=ks)
    d = _run("FD", "timing")
    fk, fd = X.cell_inputs(k, C), X.cell_inputs(d, C)
    for cneg in ("host_t", "pass_row"):
        p = X.run_arm("FDGKW", "timing", acct=True, kslots=ks, cneg=cneg)
        fp_ = X.cell_inputs(p, C)
        g = C.keep_eq_gate("L4PTFDGKW", fp_["keep"], fk["keep"])
        e = C.eq_gate("L4PTFDGKW", fp_, fd, N0)
        _ev("I4K PLANT %s: KEEP-EQ %s (%s); EQF %s" % (cneg, g[2], g[3][:160], e[2]))
        assert p["rd"]["keep"] != k["rd"]["keep"]                        # not a no-op (L34)
        assert g[2] is False                                             # CAUGHT by KEEP-EQ
    try:                                                                 # resid_in_graph: the residency recorded too
        p = X.run_arm("FDGKW", "timing", cneg="resid_in_graph")
    except RuntimeError as exc:                                          # FakeGraphs refuses a host read in a capture
        _ev("I4K PLANT resid_in_graph: the capture is refused (%s)" % str(exc)[:120])
        assert "not capturable" in str(exc)
    else:
        fp_ = X.cell_inputs(p, C)
        g = C.GC.graph_gate("L4PTFDGKW", p["rd"], L)
        e = C.eq_gate("L4PTFDGKW", fp_, fd, N0)
        _ev("I4K PLANT resid_in_graph: GRAPH %s (%s); EQF %s" % (g[2], g[3][-120:], e[2]))
        assert g[2] is False or e[2] is False
    p = X.run_arm("FDGKW", "timing", gneg="stale_input")                # I4G's plant on the combined arm
    e = C.eq_gate("L4PTFDGKW", X.cell_inputs(p, C), fd, N0)
    _ev("I4K PLANT stale_input: EQF %s" % e[2])
    assert e[2] is False


# ------------------------------------------------------------------ the eager launches
def test_i4k_eager_ops():
    boxes, ev = {}, {}
    for arm in ("FDG", "FDGKW"):
        boxes[arm] = {}
        try:
            ev[arm] = X.run_arm(arm, "timing", on_runner=lambda r, a=arm: boxes[a].update(box=G4.eager_op_count(r)))
        finally:
            G4.restore_replay()
    g, x = boxes["FDG"]["box"], boxes["FDGKW"]["box"]
    cap = set(ev["FDGKW"]["rd"]["graph"]["capture_passes"]) | set(ev["FDG"]["rd"]["graph"]["capture_passes"])
    rep = [i for i, (a, b) in enumerate(zip(g["replays"], x["replays"])) if a == L and b == L and i not in cap]
    assert rep and len(g["joint"]) == len(x["joint"]), (g["replays"], x["replays"])
    extra = [x["joint"][i] - g["joint"][i] for i in rep]
    want = L * CB.EAGER_KEEP_OPS + CB.EAGER_PASS_OPS
    _ev("I4K EAGER OPS per replayed joint pass: L4PTFDG %s; L4PTFDGKW %s; extra %s (stated L x %d + %d = %d); the keep's "
        "%d ops per layer replayed" % ([g["joint"][i] for i in rep], [x["joint"][i] for i in rep], extra,
                                        CB.EAGER_KEEP_OPS, CB.EAGER_PASS_OPS, want, CB.CAPTURED_KEEP_OPS))
    assert all(e == want for e in extra), extra


# ------------------------------------------------------------------ WIRING
def test_combo_refusals_and_no_host_sync():
    model, c, S, hz, tok0 = X.world(2, 8)
    with pytest.raises(X.SR.SpecError):
        X.k4_builder("FDGKW", cneg="nope")(model, c, S, hz, tok0)
    with pytest.raises(X.SR.SpecError):
        X.k4_builder("FDGKW", gneg="nope")(*X.world(2, 8)[:5])
    with pytest.raises(X.SR.SpecError):
        X.k4_builder("FDGKW", kneg="nope")(*X.world(2, 8)[:5])
    with pytest.raises(X.SR.SpecError):
        X.TentRunnerFKg(*([None] * 4), keep=False)
    model, c, S, hz, tok0 = X.world(2, N0)
    run = X.k4_builder("FDGKW", acct=True, probe_every=0)(model, c, S, hz, tok0, check_every=0)
    assert isinstance(run.T, X.TentStateFDGK) and isinstance(run.T, X.TentStateFK) and isinstance(run, X.TentRunnerFD)
    with RW.SyncTrap(Path(CB.__file__)) as trap:
        run.run(N0)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the combo module
    assert run.T.graph["replays"] > 0 and run.meta["combo"]["eager_pass_ops"] == CB.EAGER_PASS_OPS
    kp = run.read()["keep"]
    assert kp["passes"] == run.T.kpass + 1 and int(run.T.kcur.abs().sum()) == 0      # every pass flushed
    run.detach()
    assert run.T._g == {} and run.T._sb is None                        # the arm's end drops the graphs


# ------------------------------------------------------------------ CELL
def test_i4k_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.store_acct, a.fork_store, a.check_every) == (
        ["L4PTFD", "L4PTFDG", "L4PTFDGKW"], 216, 512, 65, 3, 0, 0, 0)
    for bad in (["--arms", "L4PTFDGKW,L4PTFD"], ["--arms", "L4PTFDGKW,L4PTFKW"], ["--arms", "L4PTFK"],
                ["--spec", "4"], ["--fork-store", "100"], ["--arms", "L4PTFD,L4PTFD"], ["--pool", "0"],
                ["--steps", "4"], ["--check-every", "1", "--batch", "216"], ["--store-acct", "2"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1", "--store-acct",
                             "1", "--require-stress", "1", "--arms", "L4PTFD,L4PTFKW,L4PTFDGKW"])
    assert (a.check_every, a.require_stress, a.store_acct, a.arms) == (1, 1, 1, ["L4PTFD", "L4PTFKW", "L4PTFDGKW"])
    assert C.runner_choice("L4PTFD", False, False) == ("FD", {})
    assert C.runner_choice("L4PTFD", True, False) == ("FK", dict(keep=False, acct=True))
    assert C.runner_choice("L4PTFDG", True, True) == ("FDG", {})
    assert C.runner_choice("L4PTFKW", True, True) == ("FK", dict(keep=True, acct=True, kcheck=True, kwin=True))
    assert C.runner_choice("L4PTFDGKW", False, False) == ("FDGK", dict(keep=True, acct=False, kcheck=False, kwin=True))
    assert C.runner_class("FDGK", X.STF) is X.TentRunnerFDGK and C.runner_class("FDG", X.STF) is X.TentRunnerFDG
    assert C.runner_class("FK", X.STF) is X.TentRunnerFK and C.runner_class("FD", X.STF) is X.TentRunnerFD
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4k_cell_pure_parts():
    C = _cell()
    d, k, x = _run("FD", "timing"), _run("FKW", "timing"), _run("FDGKW", "timing")
    fd, fk, fx = X.cell_inputs(d, C), X.cell_inputs(k, C), X.cell_inputs(x, C)
    assert C.eq_gate("L4PTFDGKW", fx, fd, N0)[2] is True
    planted = dict(fx, out=fx["out"].clone())
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    n, m, ok, det = C.eq_gate("L4PTFDGKW", planted, fd, N0)
    assert (n, m, ok) == ("L4PTFDGKW-EQF", True, False) and "L4PTFD's" in det, det
    assert C.eq_gate("L4PTFDGKW", fx, None, N0)[2] is False
    pc = dict(fx["counts"], tent=dict(fx["counts"]["tent"], calls=dict(fx["counts"]["tent"]["calls"], chain=0)))
    assert C.counts_gate("L4PTFDGKW", pc, fk["counts"], "L4PTFKW")[2] is False      # CONTROL: one host count changed
    assert C.counts_gate("L4PTFDGKW", fx["counts"], None, "L4PTFKW")[2] is False
    g = C.counts_gate("L4PTFDGKW", fx["counts"], fd["counts"], "L4PTFD", skip=("round_blocks",))
    assert g[2] is True and "round_blocks" in g[3], g
    kb = dict(fx["keep"], fetched=[v + 1 for v in fx["keep"]["fetched"]])
    assert C.keep_eq_gate("L4PTFDGKW", kb, fk["keep"])[2] is False        # CONTROL: one pass row changed
    assert C.keep_eq_gate("L4PTFDGKW", fx["keep"], None)[2] is False
    assert C.keep_eq_gate("L4PTFDGKW", fx["keep"], fk["keep"])[2] is True
    r = C.keep_ref_gate("L4PTFDGKW", fx["keep"], 8, 10, 72)
    assert r[1] is False and r[2] is True and "no reference" in r[3]   # REPORT, no verdict without a reference
    ref = C.KEEP_REF[(216, 65, 512)]
    good = dict(listed_total=ref["listed"], fetched_total=ref["fetched"], win_hits_total=ref["win_hits"])
    assert C.keep_ref_gate("X", good, 216, 65, 512)[2] is True
    assert C.keep_ref_gate("X", dict(good, fetched_total=1), 216, 65, 512)[2] is False
    rec = dict(gpu_peak_alloc_GB=82.57, mem_arena=dict(prefill_peak_GB=78.13, decode_peak_reserved_GB=77.1),
               graph=dict(pool_bytes=30_000_000, static_bytes=8_000_000))
    assert C.GC.mem_gate("L4PTFDGKW", rec)[2] is True
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh, {}, 216) == [("L4PTFDGKW-RUN", "NOT_RUN", "the arm did not run")]
    mk = lambda tok, dr, jt, wt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                      kv_at_verify=dict(new_blocks_per_round=10.0, wait_ms_median=wt),
                                      tokens=dict(equal_fraction=0.4, first=(1, 1)), vs_strongest=0.5,
                                      gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                      graph=dict(pool_bytes=5e6, capture_s=3.0))
    saved = {b: dict(C.PRED_I4K_BY_B[b]) for b in C.PRED_I4K_BY_B}
    try:
        for b in C.PRED_I4K_BY_B:
            C.PRED_I4K_BY_B[b].update({kk: None for kk in C.PRED_KEYS})
        v = {y[0]: y for y in C.verdicts(sh, dict(L4PTFD=mk(400.0, 20.0, 200.0, 20.0),
                                                  L4PTFDG=mk(410.0, 20.0, 193.0, 20.0),
                                                  L4PTFDGKW=mk(420.0, 20.0, 192.0, 14.0)), 216, eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4K_BY_B[b].clear()
            C.PRED_I4K_BY_B[b].update(saved[b])
    assert all(y[1] == "REPORT" for y in v.values()), v               # registration None: every key REPORT
    assert "8.0000" in v["I4K-JOINT-SAVING"][2] and "1.0500" in v["I4K-VS-D"][2] and "6.0000" in v[
        "I4K-WAIT-SAVING"][2] and "no value" in v["I4K-VS-KW"][2] and "1.0244" in v["I4K-VS-G"][2], v
    assert "True" in v["I4K-ABOVE-BEST"][2] or "420.0" in v["I4K-ABOVE-BEST"][2]
    assert set(v) == {"I4G-VS-D", "I4K-JOINT-SAVING", "I4K-JOINT-RATIO", "I4K-VS-D", "I4K-VS-G", "I4K-VS-KW",
                      "I4K-RATIO", "I4K-WAIT-SAVING", "I4K-DRAFT-RATIO", "I4K-PEAK-GB", "I4K-POOL-MB", "I4K-CAPTURE-S",
                      "I4K-EQ", "I4K-ABOVE-BEST", "I4K-VS-STRONGEST"}, sorted(v)


def test_i4k_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4K is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4K_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4K)
    assert set(C.PRED_I4K_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4K_BY_B.values())
    src = (EFF / "r2_i4k_cell.py").read_text()
    assert ("\nPRED_STAMP_I4K = None\n" in src) == (C.PRED_STAMP_I4K is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4k_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("i4km", 10, 8, 72, "L4PTFD,L4PTFKW,L4PTFDGKW", 1, 1, 1, 40, "100G"),
         ("i4ka", 65, 216, 512, "L4PTFD,L4PTFDG,L4PTFDGKW", 0, 0, 0, 80, "240G"),
         ("i4kb", 10, 318, 512, "L4PTFD,L4PTFKW,L4PTFDGKW", 0, 0, 0, 110, "340G"))


def test_step_script_i4k():
    src = (EFF / "r2_i4k_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    for f in ("r2_i4k_combo.py", "r2_i4d_keep.py", "r2_i4g_graph.py", "r2_i4d_cell.py", "r2_i4g_cell.py"):
        assert "benchmarks/Efficiency/" + f in src, f
    assert "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4gm", "i4dm", "i4k", "i4kc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4K = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4k_cell.py" % sha).stdout
    for cell, pool, batch, steps, arms, acct, check, stress, smin, mem in CELLS:
        r = _step(cell, sha)
        if cell in ("i4ka", "i4kb") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4K is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.returncode, r.stderr[-400:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 fork-store 0 acct %d check %d stress %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, arms, steps, acct, check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--store-acct %d" % acct, "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4k_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4k_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms %s " % arms.replace(",", r"\\?,"), r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--store-acct", str(acct), "--check-every",
                                str(check), "--require-stress", str(stress)])
        assert a.arms == arms.split(",")                                # every cell's arguments parse
