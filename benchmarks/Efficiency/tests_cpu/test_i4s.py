"""CPU tests of rung I4S (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the STACKED
arms L4PTFDGUD (L4PTFD + I4G joint graphs + I4U hoist + I4H draft graphs) and L4PTFDGUDKW (the same + the I4D keep, I4K's
way), benchmarks/Efficiency/r2_i4s_stack.py; the cell r2_i4s_cell.py; the step script r2_i4s_step.sh. The world:
tests_cpu/i4s_world.py over i4h_world (FakeGraphsD) and i4k_world. CUDA GRAPHS DO NOT RUN ON THE CPU: these tests
establish the class stack, the hoist inside layer 0's graph, the host-count bookkeeping, the keep inside the graphs and the
draft graphs on top; only the GPU miniature (i4sm) establishes capture legality, the replay's bits on the GPU, timing and
memory.

GATE 0    test_i4s_gate0_files: the tree = the merge db3898c (I4K 70059e5 + I4U 832dca5 + I4H fc6c680) + the I4S files + ONE
          module-level declaration block in each older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py,
          test_i3.py, test_i4a.py, test_i4g.py, test_i4d.py, test_i4k.py, test_i4u.py, test_i4h.py); ONE line of the
          name scan in test_i4a / test_i4h / test_i4k / test_i4u's Gate 0 (test_i4a: the merge's line back to the I4G /
          I4U form, and the I4U line back to its own bytes); nosi/ and every lever file are their tips' bytes; the fork is
          fia6's blob; the stack module imports torch and the lever modules only (no kernel); nobody else names it.
STACK     test_stack_classes: the MRO of every stacked class holds each lever's class once, in the stated order.
EQUAL     test_stacked_equals (timing B 8, check B 8, timing B 2 rowgroup, for both stacked arms; plus the intermediate
          stacks L4PTFDGU, L4PTFKU, L4PTFDGUK): the run = L4PTFD's BIT FOR BIT (tokens, fingerprints, read() except the
          host clock, the graph reports and the new records, every engine / twin / buffer tensor, the union stats); the
          keep arms = L4PTFKW's run in everything (the map, the meter, the keep record per pass); the cell's EQF, COUNTS,
          GRAPH, DGRAPH, HOIST, LEAN, KEEP, CAP, KEEP-EQ pass on the CPU evidence. Preconditions (L34): joint graphs
          captured, the hoist of layer 0 in a graph, draft replay passes, keep hits.
PLANTS    test_stack_plants_caught: no_hcounts (HOIST), stale_pass (EQF: the hoist must be renewed inside layer 0's graph),
          stale_input / stale_ids (EQF), host_t (KEEP-EQ), no_plast (EQF).
COUNT     test_i4s_eager_ops: an INDEPENDENT count of the eager launches: per replayed joint pass L4PTFDGUD = L4PTFDGD and
          L4PTFDGUDKW = L4PTFDGKW (the hoist adds no eager launch), per replayed one-row pass L4PTFDGUD = L4PTFDGD.
WIRING    test_stack_refusals_and_no_host_sync.
CELL      test_i4s_cell_arguments, test_i4s_cell_pure_parts, test_i4s_registration_state, test_step_script_i4s.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4s.py
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
MERGE = "db3898c5ae04865045200182a580c8853ef1e57f"           # nosi-r2-i4s: I4K + I4U + I4H, merged (no I4S file)
TIPS = dict(I4A="acbe433da7f07db7f7b08791c036372d6d8db885", I4G="d7997e688f89b5e6d22b2ca4d2d4472679c69a0b",
            I4D="99c231ecf4b4b110293f6a5540007ba0742748ee", I4K="70059e55042b6b31dd2dd671741dc834f146a143",
            I4U="832dca50ce13a0d8979d28437cb728f6c897834d", I4H="fc6c680d748c56bff8507759f5d0ce52db6ae5f1")
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4s"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
I4S_NEW = (E_ + "r2_i4s_stack.py", E_ + "r2_i4s_cell.py", E_ + "r2_i4s_step.sh", T_ + "i4s_world.py",
           T_ + "test_i4s.py")
I4S_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                    "i4h"))
I4S_EDITED = {T_ + "test_i4a.py": "test_i4a_gate0_files", T_ + "test_i4h.py": "test_i4h_gate0_files",
              T_ + "test_i4k.py": "test_i4k_gate0_files", T_ + "test_i4u.py": "test_i4u_gate0_files"}
LEVER_FILES = dict(
    I4A=(E_ + "r2_i4a_lean.py", E_ + "r2_i4a_cell.py", E_ + "r2_i4a_step.sh", T_ + "i4a_world.py", "docs/NOTES_i4a.md"),
    I4G=(E_ + "r2_i4g_graph.py", E_ + "r2_i4g_cell.py", E_ + "r2_i4g_step.sh", T_ + "i4g_world.py", "docs/NOTES_i4g.md"),
    I4D=(E_ + "r2_i4d_keep.py", E_ + "r2_i4d_cell.py", E_ + "r2_i4d_step.sh"),
    I4K=(E_ + "r2_i4k_combo.py", E_ + "r2_i4k_cell.py", E_ + "r2_i4k_step.sh", T_ + "i4k_world.py"),
    I4U=(E_ + "r2_i4u_hoist.py", E_ + "r2_i4u_cell.py", E_ + "r2_i4u_step.sh", T_ + "i4u_world.py"),
    I4H=(E_ + "r2_i4h_graph.py", E_ + "r2_i4h_cell.py", E_ + "r2_i4h_step.sh", T_ + "i4h_world.py", "docs/NOTES_i4h.md"))
STACK_IMPORTS = {"torch", "r2_i4a_lean", "r2_i4d_keep", "r2_i4g_graph", "r2_i4h_graph", "r2_i4k_combo", "r2_i4u_hoist"}
TIMING_KEYS = {"phases", "round_ms", "wall_s", "prelude_ms", "probe_wait_ms", "graph", "dgraph"}


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


X = _load("i4s_world")
SK, W, RW, L, N0 = X.SK, X.W, X.RW, X.L, X.N0


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4s_cell
    return r2_i4s_cell


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


def _fn_units(src: str) -> dict:
    return {k: v for k, v in _units(src).items() if not k.startswith("#")}


# ------------------------------------------------------------------ GATE 0
def test_i4s_gate0_files():
    if not all(_have(s) for s in [MERGE] + list(TIPS.values())):
        pytest.skip("no git or no commits %s in %s" % ([MERGE] + list(TIPS.values()), ROOT))
    changed = {x for x in _git("diff", "--name-only", MERGE, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4S_NEW) | set(I4S_DECL), sorted(changed - set(I4S_NEW) - set(I4S_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4S_NEW), [f for f in I4S_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", TIPS["I4A"], "--", "nosi").stdout.strip() == ""      # (b) nosi: acbe433's
    for lever, files in LEVER_FILES.items():                              # (c) every lever file: its tip's bytes
        assert _git("diff", "--name-only", TIPS[lever], "--", *files).stdout.strip() == "", lever
    for f in I4S_DECL:                                                    # (d) one I4S block each (+ one line)
        old, new = _fn_units(_git("show", "%s:%s" % (MERGE, f)).stdout), _fn_units((ROOT / f).read_text())
        allowed = {I4S_EDITED[f]} if f in I4S_EDITED else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", MERGE, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        nmax = 2 if f.endswith("test_i4a.py") else (1 if f in I4S_EDITED else 0)
        assert len(minus) <= nmax, (f, minus)
        assert any("r2_i4s_stack.py" in x for x in body) and any("I4S" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", TIPS["I4A"], "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4s_stack.py").read_text()                          # (e) the stack module: no kernel
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == STACK_IMPORTS, mods
    for name in ("r2_i4s_stack.py", "r2_i4s_cell.py"):                   # NO NEW KERNEL
        text = (EFF / name).read_text()
        assert "triton" not in text and "load_inline" not in text and "cpp_extension" not in text, name
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (f) nobody else names it
        if p.name.startswith("r2_i4s_"):
            continue
        assert "r2_i4s" not in p.read_text(), p.name


# ------------------------------------------------------------------ the class stack
def test_stack_classes():
    C = X.CLS
    S, R = C["L4PTFDGUD"]
    mro = S.__mro__
    names = [k.__name__ for k in mro]
    want =["TentStateFDGUD", "TentStateFDGD", "TentStateFDGU", "_GraphS", "TentStateFDG", "_SUa", "TentStateFDU",
            "TentStateFD"]
    assert names[:len(want)] == want, names[:10]
    for n in ("TentStateFD", "TentStateFDU", "TentStateFDG", "TentStateFDGD"):
        assert names.count(n) == 1, (n, names)
    S2, R2 = C["L4PTFDGUDKW"]
    names2 = [k.__name__ for k in S2.__mro__]
    want2 = ["TentStateFDGUDK", "TentStateFDGD", "TentStateFDGUK", "_GraphS", "TentStateFDG", "TentStateFKU",
             "TentStateFKg", "TentStateFK", "_SUa", "TentStateFDU", "TentStateFD"]
    assert names2[:len(want2)] == want2, names2[:12]
    for n in want2[1:]:
        assert names2.count(n) == 1, (n, names2)
    lean =X.LN.lean_classes(X.STF)[0]
    assert lean in S.__mro__ and lean in S2.__mro__ and X.KP.keep_classes(X.STF)[0] in S2.__mro__
    rn = [k.__name__ for k in R2.__mro__]
    for n in ("TentRunnerFDGUDK", "TentRunnerFDGD", "TentRunnerFDGUK", "TentRunnerFDG", "TentRunnerFKU", "TentRunnerFKg",
              "TentRunnerFK", "TentRunnerFDU", "TentRunnerFD"):
        assert rn.count(n) == 1, (n, rn)
    _ev("I4S STACK L4PTFDGUD %s; L4PTFDGUDKW %s" % (names[:8], names2[:11]))


# ------------------------------------------------------------------ the stacked arms = L4PTFD (= L4PTFKW in the keep)
def _tensors(run):
    """Every state tensor of a run (as test_i4k._tensors): engine, twins, buffers, the union stats, the meter, the
    outputs, the keep's map when the run has one."""
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
def _run(arm, scen="timing", B=8, gemm="natural"):
    return X.run_arm(arm, scen, B=B, gemm=gemm)


CASES = dict(gud_timing=("GUD", "timing", 8, "natural"), gud_check=("GUD", "check", 8, "natural"),
             gud_b2=("GUD", "timing", 2, "rowgroup"), gudk_timing=("GUDK", "timing", 8, "natural"),
             gudk_check=("GUDK", "check", 8, "natural"), gudk_b2=("GUDK", "timing", 2, "rowgroup"),
             gu_timing=("GU", "timing", 8, "natural"), fku_check=("FKU", "check", 8, "natural"),
             guk_timing=("GUK", "timing", 8, "natural"))
CELL_NAME = dict(GUD="L4PTFDGUD", GUDK="L4PTFDGUDKW", GU="L4PTFDGU", FKU="L4PTFKU", GUK="L4PTFDGUK")


@pytest.mark.parametrize("case", list(CASES))
def test_stacked_equals(case):
    arm, scen, B, gemm = CASES[case]
    keep = arm in X.KEEP
    d = _run("D", scen, B, gemm)
    x = _run(arm, scen, B, gemm)
    C = _cell()
    name = CELL_NAME[arm]
    rd = x["rd"]
    jp = rd["tent"]["calls"]["joint_passes"]
    h = rd["hoist"]
    gr, dr, kp = rd.get("graph"), rd.get("dgraph"), rd.get("keep")
    _ev("I4S %s %s: rounds %d, joint passes %d; hoist %s; joint graphs %s; draft %s; keep %s" % (
        case, name, rd["rounds"], jp, h, None if gr is None else (gr["captures"], gr["warm"], gr["replays"], gr["eager"],
                                                                  gr["eager_why"], gr["deltas"][:2]),
        None if dr is None else (dr["keys"], dr["capture_passes"], dr["replay_passes"], dr["eager"]),
        None if kp is None else (kp["listed_total"], kp["hits_total"], kp.get("win_hits_total"), kp["check"])))
    assert jp >= 5 and h["passes"] == jp and h["layers"] == L * jp, (jp, h)         # every joint pass hoisted
    if gr is not None:                                                 # preconditions (L34): graphs and the hoist
        assert gr["captures"] == L and gr["warm"] == L and gr["replays"] >= 3 * L and not gr["delta_bad"], gr
        assert [dd for l_, dd in gr["deltas"] if l_ == 0][0]["H"] == dict(passes=1, layers=1), gr["deltas"]
        assert all(dd["H"] == dict(layers=1) for l_, dd in gr["deltas"] if l_ > 0), gr["deltas"]
    if dr is not None:
        assert dr["replay_passes"] >= 10 and not dr["delta_bad"], dr
    if keep:
        assert kp["hits_total"] > 0 and kp["win_hits_total"] > 0, kp
    bad = _diff(d, x, skip=("hoist", "keep", "round_blocks") if keep else ("hoist",),
                tensors_skip=("kid", "kage", "blocks") if keep else ())
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    fd, fx = X.cell_inputs(d, C), X.cell_inputs(x, C)
    gates = [C.eq_gate(name, fx, fd, N0), C.IA.lean_gate(name, dict(rd, K=X.K0), L),
             C.UC.hoist_gate(name, rd, L, scen == "check")]
    if keep:
        k = _run("FKW", scen, B, gemm)
        bad = _diff(k, x, skip=("hoist",))
        assert bad == [], bad[:10]                                     # = L4PTFKW in everything (map, meter, keep)
        fk = X.cell_inputs(k, C)
        gates += [C.KC.counts_gate(name, fx["counts"], fk["counts"], "L4PTFKW"), C.DC.keep_gate(name, rd, scen == "check"),
                  C.keep_eq_gate(name, fx["keep"], fk["keep"], "L4PTFKW")]
        if scen == "check":
            gates.append(C.DC.cap_gate(name, rd))
    else:
        gates.append(C.KC.counts_gate(name, fx["counts"], fd["counts"], "L4PTFD"))
    if gr is not None:
        gates.append(C.GC.graph_gate(name, rd, L))
    if dr is not None:
        gates.append(C.HC.dgraph_gate(name, rd, L))
    for g in gates:
        assert g[2] is True, g


# ------------------------------------------------------------------ PLANTS
def test_stack_plants_caught():
    C = _cell()
    d, x = _run("D"), _run("GUD")
    fd = X.cell_inputs(d, C)
    p = X.run_arm("GUD", "timing", sneg="no_hcounts")                  # no_hcounts: the hoist's counts not re-added
    hg = C.UC.hoist_gate("L4PTFDGUD", p["rd"], L, False)
    cg = C.KC.counts_gate("L4PTFDGUD", X.cell_inputs(p, C)["counts"], fd["counts"], "L4PTFD")
    _ev("I4S PLANT no_hcounts: hoist %s; HOIST %s; COUNTS %s" % (p["rd"]["hoist"], hg[2], cg[2]))
    assert p["rd"]["hoist"] != x["rd"]["hoist"] and hg[2] is False and cg[2] is True   # only the hoist's counts move
    for kw in (dict(hneg="stale_pass"), dict(gneg="stale_input"), dict(dgneg="stale_ids"), dict(hneg="no_plast")):
        p = X.run_arm("GUD", "timing", **kw)
        e = C.eq_gate("L4PTFDGUD", X.cell_inputs(p, C), fd, N0)
        _ev("I4S PLANT %s: differs %s; EQF %s" % (kw, _diff(d, p, skip=("hoist",))[:4], e[2]))
        assert _diff(d, p, skip=("hoist",)) != [] and e[2] is False, (kw, e)    # not a no-op (L34); CAUGHT by EQF
    ks = max(1, int(_run("D", "check")["rd"]["union"]["extras_max"]))   # fewer map slots: eviction pressure
    k = X.run_arm("FKW", "timing", kslots=ks)
    fk = X.cell_inputs(k, C)
    x2 = X.run_arm("GUDK", "timing", kslots=ks)                         # the clean stacked arm under pressure
    assert _diff(k, x2, skip=("hoist",)) == [] and C.keep_eq_gate("L4PTFDGUDKW", X.cell_inputs(x2, C)["keep"],
                                                                   fk["keep"], "L4PTFKW")[2] is True
    p = X.run_arm("GUDK", "timing", kslots=ks, cneg="host_t")          # host_t: the keep's step frozen in the graphs
    g = C.keep_eq_gate("L4PTFDGUDKW", X.cell_inputs(p, C)["keep"], fk["keep"], "L4PTFKW")
    _ev("I4S PLANT host_t (kslots %d): differs %s; KEEP-EQ %s (%s)" % (ks, _diff(k, p, skip=("hoist",))[:4], g[2],
                                                                      g[3][:160]))
    assert _diff(k, p, skip=("hoist",)) != [] and g[2] is False


# ------------------------------------------------------------------ the eager launches
def test_i4s_eager_ops():
    boxes, ev = {}, {}
    for arm in ("GD", "GUD", "FDGKW", "GUDK"):
        try:
            ev[arm] = X.run_arm(arm, "timing", on_runner=lambda r, a=arm: boxes.update({a: X.XG.eager_op_count(r)}))
        finally:
            X.XH.restore_replay()
    for a, b in (("GD", "GUD"), ("FDGKW", "GUDK")):
        ga, gb = boxes[a], boxes[b]
        cap = set(ev[a]["rd"]["graph"]["capture_passes"]) | set(ev[b]["rd"]["graph"]["capture_passes"])
        rep = [i for i, (u, v) in enumerate(zip(ga["replays"], gb["replays"])) if u == L and v == L and i not in cap]
        assert len(rep) >= 3 and len(ga["joint"]) == len(gb["joint"]), (ga["replays"], gb["replays"])
        _ev("I4S EAGER OPS per replayed joint pass: %s %s; %s %s" % (a, [ga["joint"][i] for i in rep], b,
                                                                     [gb["joint"][i] for i in rep]))
        assert all(ga["joint"][i] == gb["joint"][i] for i in rep)      # the hoist adds no eager launch
    db = {}
    for arm in ("GD", "GUD"):
        try:
            X.run_arm(arm, "timing", on_runner=lambda r, a=arm: db.update({a: X.XH.draft_op_count(r)}))
        finally:
            X.XH.restore_replay()
    a, b = db["GD"], db["GUD"]
    rep = [i for i in range(min(len(a["passes"]), len(b["passes"]))) if a["replays"][i] == b["replays"][i] == 2 * L
           and a["captures"][i] == b["captures"][i] == 0]
    _ev("I4S EAGER OPS per replayed one-row pass: L4PTFDGD %s; L4PTFDGUD %s" % ([a["passes"][i] for i in rep[:6]],
                                                                              [b["passes"][i] for i in rep[:6]]))
    assert len(rep) >= 10 and all(a["passes"][i] == b["passes"][i] for i in rep)


# ------------------------------------------------------------------ WIRING
def test_stack_refusals_and_no_host_sync():
    for kw in (dict(sneg="nope"), dict(cneg="resid_in_graph"), dict(kneg="nope"), dict(gneg="nope"), dict(dgneg="nope"),
               dict(hneg="nope")):
        with pytest.raises(X.SR.SpecError):
            X.s_builder("GUDK", **kw)(*X.world(2, 8)[:5])
    with pytest.raises(X.SR.SpecError, match="accounting"):            # no accounting in the stacked keep arm
        _acct_refused(*X.world(2, 8)[:5])
    model, c, S, hz, tok0 = X.world(2, N0)
    run = X.s_builder("GUDK", probe_every=0)(model, c, S, hz, tok0, check_every=0)
    assert isinstance(run.T, X.CLS["L4PTFDGUDKW"][0]) and run.T._vs is run.vs and run.T.kwin and not run.T.kacct
    with RW.SyncTrap(Path(SK.__file__)) as trap:
        run.run(N0)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the stack module
    rd = run.read()
    assert rd["graph"]["replays"] > 0 and rd["dgraph"]["replay_passes"] > 0 and rd["hoist"]["passes"] > 0
    assert run.meta["stack"] == dict(arm="L4PTFDGUDKW", sneg=None)
    assert "_resid_joint" not in run.T.__dict__ and "_resid_lean" not in run.T.__dict__   # the stubs are gone
    assert int(run.T.kcur.abs().sum()) == 0                            # every pass flushed
    run.detach()
    assert run.T._g == {} and run.T._sb is None and run.T._dg == {}    # the arm's end drops every graph


def _acct_refused(model, c, S, hz, tok0):
    """Builds L4PTFDGUDKW with acct=True (the stack refuses it)."""
    fa, fk = X.WB.fake_attention_u, X.Z.fake_fork
    B = int(c.layers[0].cache_engine._k_gpu.shape[0])
    arena = X.I3.chain_arena(c, B, X.KA.store_slots(X.TOPK, X.K0))
    ops = X.RE.RaggedOps(X.W.KERNELS, twin="torch", body="torch", graph=True)
    rs = X.RE.attach(model, c, S, hz, ops)
    X.KA.attach(rs, arena)
    X.RW.define_undefined(rs)
    sops = X.SR.SpecOps(level="L4P", K=X.K0, mode="exact", gemm="natural", extra=arena.X, attn=fa, journal=X.RW.CAP,
                        redirect=True, record=False, probe_every=0)
    be = X.FakeGraphsD()
    return X.CLS["L4PTFDGUDKW"][1](model, rs, sops, X.BC.HostClock(), fork=fk, t0=tok0, hooks=None, check_every=0,
                                   union_check=False, union_stats=False, phase_marks=False, keep=True, kwin=True,
                                   acct=True, graph_backend=be)


# ------------------------------------------------------------------ CELL
def test_i4s_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.fork_store, a.prefill_hook, a.tu_gemm, a.check_every,
            a.require_stress, a.probe_every) == (["L4PTFD", "L4PTFDG", "L4PTFDGUD"], 216, 512, 65, 3, 0, "nocopy",
                                                 "natural", 0, 0, 16)
    for bad in (["--arms", "L4PTFDGUD,L4PTFD"], ["--arms", "L4PTFDGUD"], ["--arms", "L4PTFD,L4PTFDGUDKW"],
                ["--arms", "L4PTFD,L4PTFDGUDKW,L4PTFDGKW"], ["--arms", "L4PTFDGD"], ["--spec", "4"],
                ["--fork-store", "100"], ["--arms", "L4PTFD,L4PTFD"], ["--pool", "0"], ["--steps", "4"],
                ["--arms", "L4PTFD,L4PTFKW", "--check-every", "1", "--batch", "216"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    for arms in ("L4PTFD,L4PTFKW,L4PTFDGUD,L4PTFDGUDKW", "L4PTFD,L4PTFDGKW,L4PTFDGUDKW", "L4PTFD,L4PTFDG,L4PTFDGUD"):
        assert C.parse_args(base + ["--arms", arms]).arms == arms.split(",")
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1",
                             "--require-stress", "1", "--arms", "L4PTFD,L4PTFKW,L4PTFDGUD,L4PTFDGUDKW"])
    assert (a.check_every, a.require_stress) == (1, 1)
    assert C.runner_choice("L4PTFDGUDKW", True) == ("SK", dict(keep=True, kwin=True, kcheck=True))
    assert C.runner_choice("L4PTFDGUD", False) == ("S", {}) and C.runner_choice("L4PTFKW", False)[0] == "FK"
    assert C.runner_class("SK", X.STF) is X.CLS["L4PTFDGUDKW"][1] and C.runner_class("S", X.STF) is X.CLS[
        "L4PTFDGUD"][1]
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4s_cell_pure_parts():
    C = _cell()
    d, x = _run("D"), _run("GUDK")
    k = _run("FKW")
    fd, fx, fk = X.cell_inputs(d, C), X.cell_inputs(x, C), X.cell_inputs(k, C)
    planted = dict(fx, out=fx["out"].clone())
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    n, m, ok, det = C.eq_gate("L4PTFDGUDKW", planted, fd, N0)
    assert (n, m, ok) == ("L4PTFDGUDKW-EQF", True, False), det
    assert C.keep_ref_arm("L4PTFDGUDKW", {"L4PTFD": 1, "L4PTFDGKW": 1}) == "L4PTFDGKW"
    assert C.keep_ref_arm("L4PTFDGUDKW", {"L4PTFD": 1, "L4PTFKW": 1, "L4PTFDGKW": 1}) == "L4PTFKW"
    assert C.keep_ref_arm("L4PTFDGKW", {"L4PTFD": 1, "L4PTFDGKW": 1}) is None
    assert C.keep_eq_gate("L4PTFDGUDKW", fx["keep"], fk["keep"], "L4PTFDGKW")[2] is True
    kp2 = dict(fx["keep"], fetched=list(fx["keep"]["fetched"]))
    kp2["fetched"][-1] += 1                                            # CONTROL: one pass fetched one block more
    assert C.keep_eq_gate("L4PTFDGUDKW", kp2, fk["keep"], "L4PTFKW")[2] is False
    assert "no L4PTFDGKW keep record" in C.keep_eq_gate("L4PTFDGUDKW", fx["keep"], None, "L4PTFDGKW")[3]
    rd = x["rd"]                                                       # the cell's arm gates on the CPU evidence
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    keep = dict(L4PTFD=fd, L4PTFKW=fk)
    gs = C.arm_gates("L4PTFDGUDKW", rd, rec, fx, keep, False, 8, 10, N0, L, 1.0)
    names = [g[0] for g in gs]
    assert names == ["L4PTFDGUDKW-MEM", "L4PTFDGUDKW-GRAPH", "L4PTFDGUDKW-COUNTS", "L4PTFDGUDKW-DGRAPH",
                     "L4PTFDGUDKW-HOIST", "L4PTFDGUDKW-KEEP", "L4PTFDGUDKW-KEEP-EQ", "L4PTFDGUDKW-KEEP-REF",
                     "L4PTFDGUDKW-EQF"], names
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    gs = C.arm_gates("L4PTFDGUDKW", rd, rec, fx, dict(L4PTFD=fd), False, 8, 10, N0, L, 1.0)
    assert ("L4PTFDGUDKW-KEEP-EQ", True, False) == next(g for g in gs if g[0].endswith("KEEP-EQ"))[:3]
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh2, {}, 216)[-1] == ("I4S-RUN", "NOT_RUN", "no stacked arm ran")
    mk = lambda tok, dr, jt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                  vs_strongest=0.5, gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                  graph=dict(pool_bytes=50e6, capture_s=3.0), dgraph=dict(dcapture_s=4.0))
    sp = dict(L4PTFD=mk(400.0, 140.0, 200.0), L4PTFDG=mk(410.0, 140.0, 192.0), L4PTFDGUD=mk(420.0, 136.0, 189.0))
    saved = {b: dict(C.PRED_I4S_BY_B[b]) for b in C.PRED_I4S_BY_B}
    try:
        for b in C.PRED_I4S_BY_B:
            C.PRED_I4S_BY_B[b].update({kk: None for kk in C.PRED_KEYS})
        v = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
        C.PRED_I4S_BY_B[216].update(I4S_vs_ref=(1.0, 1.01), I4S_above_best=True, I4S_eq=True)
        w = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4S_BY_B[b].clear()
            C.PRED_I4S_BY_B[b].update(saved[b])
    assert all(y[1] == "REPORT" for y in v.values()), v                # registration None: every key REPORT
    assert "3.0000" in v["I4S-JOINT-SAVING"][2] and "4.0000" in v["I4S-DRAFT-SAVING"][2] and "1.0244" in v[
        "I4S-VS-REF"][2] and "1.0500" in v["I4S-VS-D"][2] and "7.0000" in v["I4S-CAPTURE-S"][2], v
    assert "X = L4PTFDGUD, REF = L4PTFDG" in v["I4S-ARM"][2]
    assert w["I4S-VS-REF"][1] == "REFUTED" and w["I4S-ABOVE-BEST"][1] == "CONFIRMED" and w["I4S-EQ"][1] == "CONFIRMED"
    sp2 = dict(sp, L4PTFDGKW=mk(430.0, 140.0, 190.0), L4PTFDGUDKW=mk(425.0, 136.0, 188.0))
    v2 = {y[0]: y for y in C.verdicts(sh2, sp2, 318)}
    assert "X = L4PTFDGUDKW, REF = L4PTFDGKW" in v2["I4S-ARM"][2] and "I4K-VS-D" in v2
    assert C.capture_s(dict(graph=dict(capture_s=1.5), dgraph=dict(dcapture_s=2.0))) == 3.5 and C.capture_s({}) is None
    assert C.stacked_arm(sp2) == "L4PTFDGUDKW" and C.stacked_arm(dict(L4PTFD=1)) is None


def test_i4s_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4S is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4S_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4S)
    assert set(C.PRED_I4S_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4S_BY_B.values())
    src = (EFF / "r2_i4s_cell.py").read_text()
    assert ("\nPRED_STAMP_I4S = None\n" in src) == (C.PRED_STAMP_I4S is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4s_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("i4sm", 10, 8, 72, 1, 1, 40, "100G", "L4PTFD,L4PTFKW,L4PTFDGUD,L4PTFDGUDKW"),
         ("i4sa", 65, 216, 512, 0, 0, 100, "240G", "L4PTFD,L4PTFDG,L4PTFDGUD"),
         ("i4sb", 10, 318, 512, 0, 0, 150, "340G", "L4PTFD,L4PTFDGKW,L4PTFDGUDKW"))


def test_step_script_i4s():
    src = (EFF / "r2_i4s_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    for f in ("r2_i4s_stack.py", "r2_i4h_graph.py", "r2_i4h_cell.py", "r2_i4k_combo.py", "r2_i4k_cell.py",
              "r2_i4d_keep.py", "r2_i4d_cell.py", "r2_i4u_hoist.py", "r2_i4u_cell.py", "r2_i4g_graph.py",
              "r2_i4g_cell.py", "r2_i4a_lean.py", "r2_i4a_cell.py"):
        assert "benchmarks/Efficiency/%s" % f in src, f
    assert "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4hm", "i4ka", "i4s", "i4sc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4S = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4s_cell.py" % sha).stdout
    ran = 0
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if cell in ("i4sa", "i4sb") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4S is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.stderr[-400:])
        ran += 1
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 fork-store 0 check %d stress %d STEP_MIN %d mem %s "
                "claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (cell, 63 + pool, pool, batch, arms, steps,
                                                                        check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4s_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4s_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms %s " % arms.replace(",", r"\\?,"), r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == arms.split(",")                               # every cell's arguments parse
    assert ran >= 1
    _ev("I4S DRYRUN: %d cell(s) printed their step command (registered %s)" % (ran, registered))
