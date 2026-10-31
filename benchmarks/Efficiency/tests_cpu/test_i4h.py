"""CPU tests of rung I4H (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), the
"draft-step graph": the arm L4PTFDGD = L4PTFDG with the one-row (draft) passes replayed from per-layer CUDA graphs
(benchmarks/Efficiency/r2_i4h_graph.py; the cell r2_i4h_cell.py; the step script r2_i4h_step.sh). The world:
tests_cpu/i4h_world.py over i4g_world (FakeGraphsD: a CUDA graph stand-in with the split capture, begin() / end(), that
captures without executing and replays the recorded ops on the captured tensors). CUDA GRAPHS DO NOT RUN ON THE CPU:
these tests establish the split at the residency, the static-input plumbing, the host-count bookkeeping, the keys, the
eager fallback and the L61 steady-set fix; only the GPU miniature (i4hm) establishes capture legality on the device, the
replay's bits on the GPU, timing and memory.

GATE 0    test_i4h_gate0_files: the tree = I4G's d7997e6 except the I4H files and ONE module-level declaration block in
          each older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py, test_i4a.py,
          test_i4g.py; in test_i4g.py also ONE line of test_i4g_gate0_files, as I4G did in test_i4a.py); nosi/ and every
          I4A / I4G file are d7997e6's bytes; the fork is fia6's blob; the graph module imports torch and time only (no
          kernel); nobody else names it.
FAKE      test_fake_graphs_d_semantics: the split capture's properties, the result node and the ILLEGAL guard (each with a
          control).
EQUAL     test_l4ptfdgd_equals_l4ptfd (timing B 8, check B 8, timing B 2 rowgroup): L4PTFDGD's run = L4PTFD's run BIT FOR
          BIT (tokens, fingerprints, read() except the host clock and the graph reports, every engine / twin / buffer
          tensor, the union stats, the meter's blocks) and the cell's EQF, COUNTS, GRAPH, DGRAPH, LEAN pass on the CPU
          evidence. Preconditions (L34): graphs captured, replay passes after the capture pass, two keys at least (a check
          key in the probe rounds; timing B 2: the snap key of the one-row J slots).
PLANTS    test_draft_graph_plants_caught: stale_ids (EQF), no_counts (COUNTS and LEAN), no_key (COUNTS: the check-only
          counters of a check round's one-row pass), live_skip (EQF).
COUNT     test_i4h_eager_ops: an INDEPENDENT count of the eager launches of a one-row pass (a replay = one launch).
WIRING    test_draft_graph_refusals_and_no_host_sync.
CELL      test_i4h_cell_arguments, test_i4h_cell_pure_parts (incl. the L61 fix), test_i4h_registration_state,
          test_step_script_i4h, test_notes_i4h.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4h.py
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
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
I4G_TIP = "d7997e688f89b5e6d22b2ca4d2d4472679c69a0b"        # nosi-r2-i4g: rung I4G with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4h"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4H_NEW = ("benchmarks/Efficiency/r2_i4h_graph.py", "benchmarks/Efficiency/r2_i4h_cell.py",
           "benchmarks/Efficiency/r2_i4h_step.sh", T_ + "i4h_world.py", T_ + "test_i4h.py", "docs/NOTES_i4h.md")
I4H_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py",
            T_ + "test_i4a.py", T_ + "test_i4g.py")
OLDER_FILES = ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
               "benchmarks/Efficiency/r2_i4g_step.sh", T_ + "i4g_world.py", "docs/NOTES_i4g.md",
               "benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
               "benchmarks/Efficiency/r2_i4a_step.sh", T_ + "i4a_world.py", "docs/NOTES_i4a.md")
TIMING_KEYS = {"phases", "round_ms", "wall_s", "prelude_ms", "probe_wait_ms", "graph", "dgraph"}
# I4S (nosi-r2-i4s: I4K 70059e5 + I4U 832dca5 + I4H fc6c680 + the stacked arms): the later files in this tree (their
# own Gate 0: tests_cpu/test_i4d.py, test_i4k.py, test_i4u.py, test_i4s.py); I4H_LATER: the later modules that import
# the draft-graph module (scan (e) skips them)
I4H_NEW = I4H_NEW + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                     "benchmarks/Efficiency/r2_i4d_step.sh", T_ + "test_i4d.py", "benchmarks/Efficiency/r2_i4k_combo.py",
                     "benchmarks/Efficiency/r2_i4k_cell.py", "benchmarks/Efficiency/r2_i4k_step.sh",
                     T_ + "i4k_world.py", T_ + "test_i4k.py", "benchmarks/Efficiency/r2_i4u_hoist.py",
                     "benchmarks/Efficiency/r2_i4u_cell.py", "benchmarks/Efficiency/r2_i4u_step.sh",
                     T_ + "i4u_world.py", T_ + "test_i4u.py", "benchmarks/Efficiency/r2_i4s_stack.py",
                     "benchmarks/Efficiency/r2_i4s_cell.py", "benchmarks/Efficiency/r2_i4s_step.sh",
                     T_ + "i4s_world.py", T_ + "test_i4s.py")                                    # I4S
I4H_LATER = ("r2_i4s_stack.py", "r2_i4s_cell.py")                                                # I4S
# I4CC (nosi-r2-i4cc: I4S 8f236f8 + the compiled arms): its files (its own Gate 0: tests_cpu/test_i4cc.py); its modules
# name the I4S modules only
I4H_NEW = I4H_NEW + ("benchmarks/Efficiency/r2_i4cc_compile.py", "benchmarks/Efficiency/r2_i4cc_cell.py",
                     "benchmarks/Efficiency/r2_i4cc_step.sh", T_ + "i4cc_world.py", T_ + "test_i4cc.py")  # I4CC
# I4DC (nosi-r2-i4dc: I4CC d75a946 + the compiled draft helpers): its files (its own Gate 0: tests_cpu/test_i4dc.py);
# I4H_LATER: its modules cite the I4H module's lines (scan (e) skips them)
I4H_NEW = I4H_NEW + ("benchmarks/Efficiency/r2_i4dc_compile.py", "benchmarks/Efficiency/r2_i4dc_cell.py",
                     "benchmarks/Efficiency/r2_i4dc_step.sh", T_ + "i4dc_world.py", T_ + "test_i4dc.py")  # I4DC
I4H_LATER = I4H_LATER + ("r2_i4dc_compile.py", "r2_i4dc_cell.py")                              # I4DC
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py); I4H_LATER: its modules cite the I4H module's lines (scan (e) skips them)
I4H_NEW = I4H_NEW + ("benchmarks/Efficiency/r2_recon7_protocol.py", "benchmarks/Efficiency/r2_recon7_probe.py",
                     "benchmarks/Efficiency/r2_recon7_cell.py", "benchmarks/Efficiency/r2_recon7_streams.py",
                     "benchmarks/Efficiency/r2_recon7_mem.py", "benchmarks/Efficiency/r2_recon7_step.sh",
                     "benchmarks/Efficiency/r2_i4c_cell.py", "benchmarks/Efficiency/r2_i4m_cell.py",
                     "benchmarks/Efficiency/r2_wsm_cell.py", T_ + "test_recon7.py", "docs/NOTES_recon7.md")  # RECON7
I4H_LATER = I4H_LATER + ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py", "r2_recon7_streams.py",
                         "r2_recon7_mem.py", "r2_i4c_cell.py", "r2_i4m_cell.py", "r2_wsm_cell.py")       # RECON7
# J1 (nosi-r2-j1: RECON7 cd0f0c5 + the compiled joint pooling of AREA 1): its files
# (its own Gate 0: tests_cpu/test_j1.py)
I4H_NEW = I4H_NEW + ("benchmarks/Efficiency/r2_j1_select.py", "benchmarks/Efficiency/r2_j1_cell.py",
                     "benchmarks/Efficiency/r2_j1_step.sh", T_ + "test_j1.py", "docs/NOTES_j1.md")          # J1


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4h_world")
DGR, W, RW, L, N0 = X.DGR, X.W, X.RW, X.L, X.N0


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4h_cell
    return r2_i4h_cell


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


# ------------------------------------------------------------------ GATE 0
def test_i4h_gate0_files():
    if not _have(I4G_TIP):
        pytest.skip("no git or no commit %s in %s" % (I4G_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I4G_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4H_NEW) | set(I4H_DECL), sorted(changed - set(I4H_NEW) - set(I4H_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4H_NEW), [f for f in I4H_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I4G_TIP, "--", "nosi").stdout.strip() == ""   # (b) nosi: d7997e6's bytes
    assert _git("diff", "--name-only", I4G_TIP, "--", *OLDER_FILES).stdout.strip() == ""   # L4PTFD, L4PTFDG: their code
    for f in I4H_DECL:                                                    # (c) one added declaration block each
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I4G_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        allowed = {"test_i4g_gate0_files"} if f.endswith("test_i4g.py") else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I4G_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert body and len(minus) <= (1 if f.endswith("test_i4g.py") else 0), (f, minus)
        assert any("r2_i4h_graph.py" in x for x in body) and any("I4H" in x for x in body), (f, body[:5])
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4h_graph.py").read_text()                         # (d) the graph module: torch and time only
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch", "time"}, mods
    assert "@triton.jit" not in src and "import triton" not in src      # NO NEW KERNEL
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4h_") or p.name in I4H_LATER:
            continue
        assert "r2_i4h" not in p.read_text(), p.name


# ------------------------------------------------------------------ the stand-in
def test_fake_graphs_d_semantics():
    with torch.inference_mode():
        a = torch.zeros(4)
        xs = torch.ones(4)
        cnt = dict(n=0)
        seen = []

        def attn(t):                                                   # a result node that reads on the host
            seen.append(float(t.sum()))
            return t * 10
        T = types.SimpleNamespace(attn=attn, rs=types.SimpleNamespace(ops=types.SimpleNamespace(kernels={})))
        G = X.FakeGraphsD()
        G.T = T
        G.begin()
        cnt["n"] += 1                                                  # a host count: moves at the capture only
        y = xs * 2                                                     # out of place
        a.add_(y)                                                      # a write
        r = T.attn(a)                                                  # the node: ONE call, its result held
        z = r + 1
        g = G.end()
        assert torch.equal(a, torch.zeros(4)) and T.attn is attn       # the capture executes nothing; node restored
        assert seen == [8.0] and torch.equal(z, torch.full((4,), 21.0))   # values of the capture-time call only
        g.replay()
        assert torch.equal(a, torch.full((4,), 2.0)) and torch.equal(z, torch.full((4,), 21.0))
        xs.fill_(3.0)                                                  # the replay reads the static input now
        g.replay()
        assert torch.equal(a, torch.full((4,), 8.0)) and torch.equal(z, torch.full((4,), 81.0)) and cnt["n"] == 1
        assert seen[-2:] == [8.0, 32.0]                                # the node reads its tensors at each replay
        G.begin()
        with pytest.raises(RuntimeError, match="not capturable"):      # CONTROL: a host read outside a node
            float(a.sum())
        G.abort()
        assert G._open is None and T.attn is attn and torch.equal(a, torch.full((4,), 8.0))
        assert G.side(lambda: 5) == 5 and G.sides == 1


# ------------------------------------------------------------------ L4PTFDGD = L4PTFD, bit for bit
@functools.lru_cache(maxsize=None)
def _pair(scen, B=8, gemm="natural"):
    return X.run_arm("D", scen, B=B, gemm=gemm), X.run_arm("GD", scen, B=B, gemm=gemm)


def _tensors(run):
    """Every state tensor of a run (the engine, twins, buffers, the union stats, the meter, the outputs)."""
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
    return out


def _diff(d, g):
    bad = [k for k in sorted(set(d["rd"]) | set(g["rd"])) if k not in TIMING_KEYS and d["rd"].get(k) != g["rd"].get(k)]
    td, tg = _tensors(d["runner"]), _tensors(g["runner"])
    bad += [k for k in sorted(set(td) | set(tg)) if k not in td or k not in tg or not torch.equal(W.bits(td[k]),
                                                                                                    W.bits(tg[k]))]
    return bad


CASES = dict(timing=("timing", 8, "natural"), check=("check", 8, "natural"), timing_b2=("timing", 2, "rowgroup"))


@pytest.mark.parametrize("case", list(CASES))
def test_l4ptfdgd_equals_l4ptfd(case):
    scen, B, gemm = CASES[case]
    d, g = _pair(scen, B, gemm)
    C = _cell()
    dr, gr = g["rd"]["dgraph"], g["rd"]["graph"]
    passes = g["rd"]["lean"]["passes"]
    print("I4H %s: rounds %d, one-row passes %d; draft keys %s (warm %s); captures %s, capture passes %s, replay passes "
          "%s, eager %s; capture rounds %s; joint capture rounds %s; joint graphs %s" % (
              case, g["rd"]["rounds"], passes, dr["keys"], dr["warm_keys"], dr["captures"], dr["capture_passes"],
              dr["replay_passes"], dr["eager"], dr["capture_rounds"], dr["joint_capture_rounds"], gr["captures"]))
    assert dr["replay_passes"] >= 10 and not dr["delta_bad"] and gr["captures"] == L, (dr, gr)      # (L34)
    assert dr["capture_rounds"][0][0] == -1                           # the first key is captured in the prelude
    if scen == "timing":
        assert [False, False] in dr["keys"] and [True, False] in dr["keys"], dr    # the probe rounds' check key
    else:
        assert dr["keys"] == [[True, False]], dr                       # every round a check round
    if case == "timing_b2":
        assert [False, True] in dr["keys"], dr                         # the snap key (one-row J slots)
    bad = _diff(d, g)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    fd, fg = X.cell_inputs(d, C), X.cell_inputs(g, C)
    n, m, ok, det = C.eq_gate("L4PTFDGD", fg, fd, N0)
    assert (n, m, ok) == ("L4PTFDGD-EQF", True, True), det
    n, m, ok, det = C.GC.counts_gate("L4PTFDGD", fg["counts"], fd["counts"])
    assert ok, det
    assert C.GC.graph_gate("L4PTFDGD", g["rd"], L)[2], C.GC.graph_gate("L4PTFDGD", g["rd"], L)
    n, m, ok, det = C.dgraph_gate("L4PTFDGD", g["rd"], L)
    assert ok, det
    assert C.dgraph_gate("L4PTFDG", dict(d["rd"], graph={}), L)[2] is False   # CONTROL: no draft graph record
    lg = C.IA.lean_gate("L4PTFDGD", dict(g["rd"], K=X.K0), L)
    assert lg[2], lg


# ------------------------------------------------------------------ PLANTS
def test_draft_graph_plants_caught():
    C = _cell()
    d, g = _pair("timing")
    fd = X.cell_inputs(d, C)
    p = X.run_arm("GD", "timing", dgneg="stale_ids")                   # stale_ids: the replay reads old token ids
    assert _diff(d, p) != []                                           # not a no-op (L34)
    ok = C.eq_gate("L4PTFDGD", X.cell_inputs(p, C), fd, N0)[2]
    print("I4H PLANT stale_ids: EQF %s" % ok)
    assert ok is False
    p = X.run_arm("GD", "timing", dgneg="no_counts")                   # no_counts: the skipped host counts
    fp_ = X.cell_inputs(p, C)
    n, m, ok, det = C.GC.counts_gate("L4PTFDGD", fp_["counts"], fd["counts"])
    lg = C.IA.lean_gate("L4PTFDGD", dict(p["rd"], K=X.K0), L)
    print("I4H PLANT no_counts: COUNTS %s (%s); LEAN %s; EQF %s" % (ok, det[:160], lg[2], C.eq_gate(
        "L4PTFDGD", fp_, fd, N0)[2]))
    assert ok is False and lg[2] is False
    assert d["rd"]["tent"]["fill_rows"] + d["rd"]["tent"]["edge_rows"] > 0   # precondition: check-only counters move
    p = X.run_arm("GD", "timing", dgneg="no_key")                      # no_key: a check round replays another key
    ok = C.GC.counts_gate("L4PTFDGD", X.cell_inputs(p, C)["counts"], fd["counts"])[2]
    print("I4H PLANT no_key: differs %s; keys %s; COUNTS %s" % (_diff(d, p)[:4], p["rd"]["dgraph"]["keys"], ok))
    assert p["rd"]["dgraph"]["keys"] == [[False, False]] and ok is False
    p = X.run_arm("GD", "timing", dgneg="live_skip")                   # live_skip: B(l) reads the capture's mask
    ok = C.eq_gate("L4PTFDGD", X.cell_inputs(p, C), fd, N0)[2]
    print("I4H PLANT live_skip: differs %s; EQF %s" % (_diff(d, p)[:4], ok))
    assert ok is False


# ------------------------------------------------------------------ the eager launches
def test_i4h_eager_ops():
    boxes = {}
    for kind in ("D", "GD"):
        try:
            ev = X.run_arm(kind, "timing", on_runner=lambda r, k=kind: boxes.update({k: X.draft_op_count(r)}))
        finally:
            X.restore_replay()
    d, g = boxes["D"], boxes["GD"]
    assert len(d["passes"]) == len(g["passes"]) == ev["rd"]["lean"]["passes"], (len(d["passes"]), len(g["passes"]))
    rep = [i for i, n in enumerate(g["replays"]) if n == 2 * L and g["captures"][i] == 0]    # replay passes only
    assert len(rep) >= 10 and sum(g["captures"]) == ev["rd"]["dgraph"]["captures"], (g["replays"], g["captures"])
    per_layer = [(d["passes"][i] - g["passes"][i]) / L for i in rep]
    print("I4H EAGER OPS per one-row pass: L4PTFD %s; L4PTFDGD %s (replays %s); removed per layer in replayed passes %s; "
          "eager names left %s" % (d["passes"][:6], g["passes"][:6], g["replays"][:6], sorted(set(per_layer)),
                                   dict(g["names"][rep[0]])))
    assert all(x > 0 for x in per_layer)
    assert all(g["passes"][i] + g["replays"][i] < d["passes"][i] for i in rep)


# ------------------------------------------------------------------ WIRING
def test_draft_graph_refusals_and_no_host_sync():
    model, c, S, hz, tok0 = X.XG.world(2, 8)
    with pytest.raises(X.SR.SpecError):
        X.arm_builder("GD", dgneg="nope")(model, c, S, hz, tok0)
    model, c, S, hz, tok0 = X.XG.world(2, N0)
    run = X.arm_builder("GD", probe_every=0)(model, c, S, hz, tok0, check_every=0)
    assert isinstance(run.T, X.TentStateFDGD) and isinstance(run.T, X.TentStateFDG)
    with RW.SyncTrap(Path(DGR.__file__)) as trap:
        run.run(N0)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the hook
    rd = run.read()
    assert rd["dgraph"]["replay_passes"] > 0 and run.meta["draft_graph"]["backend"] == "FakeGraphsD"
    assert "_resid_lean" not in run.T.__dict__ and run.rs.graphs[0] is not None and not isinstance(
        run.rs.graphs[0], DGR._Inline)                                 # the stub and the inline selection are gone
    run.detach()
    assert run.T._dg == {} and run.T._dsb is None and run.T._g == {}   # the arm's end drops every graph


# ------------------------------------------------------------------ CELL
def test_i4h_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.fork_store, a.prefill_hook, a.tu_gemm, a.check_every,
            a.require_stress, a.probe_every) == (["L4PTFD", "L4PTFDG", "L4PTFDGD"], 216, 512, 65, 3, 0, "nocopy",
                                                 "natural", 0, 0, 16)
    for bad in (["--arms", "L4PTFDG,L4PTFD"], ["--arms", "L4PTFDGD"], ["--arms", "L4PTF"], ["--spec", "4"],
                ["--fork-store", "100"], ["--arms", "L4PTFD,L4PTFD"], ["--pool", "0"], ["--steps", "4"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    assert C.parse_args(base + ["--arms", "L4PTFD,L4PTFDGD"]).arms == ["L4PTFD", "L4PTFDGD"]
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1",
                             "--require-stress", "1"])
    assert (a.check_every, a.require_stress) == (1, 1)
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4h_cell_pure_parts():
    C = _cell()
    d, g = _pair("timing")
    fd, fg = X.cell_inputs(d, C), X.cell_inputs(g, C)
    planted = dict(fg, out=fg["out"].clone())
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    n, m, ok, det = C.eq_gate("L4PTFDGD", planted, fd, N0)
    assert (n, m, ok) == ("L4PTFDGD-EQF", True, False) and "L4PTFD's" in det, det
    assert "no L4PTFD record" in C.eq_gate("L4PTFDGD", fg, None, N0)[3]
    rd = g["rd"]
    assert C.dgraph_gate("L4PTFDGD", rd, L)[2] is True

    def plant(**kw):
        return dict(rd, dgraph=dict(rd["dgraph"], **kw))
    assert C.dgraph_gate("L4PTFDGD", plant(delta_bad=[(0, {}, {})]), L)[2] is False          # CONTROL: counts
    assert C.dgraph_gate("L4PTFDGD", plant(eager=1, eager_why={"hooks": 1}), L)[2] is False  # CONTROL: an eager pass
    assert C.dgraph_gate("L4PTFDGD", plant(replay_passes=rd["dgraph"]["replay_passes"] - 1), L)[2] is False
    assert C.dgraph_gate("L4PTFDGD", plant(captures=rd["dgraph"]["captures"] - 2), L)[2] is False
    cx = C.capture_rounds_x(rd)
    assert cx["mapped"] and cx["joint"] == [1] and min(cx["draft"]) > 0, cx      # the prelude's capture is round -1
    cg = C.capture_rounds_x(dict(graph=dict(capture_passes=[1, 1]), tent=dict(calls=dict(j1_passes=0))))
    assert cg == dict(joint=[1], draft=[], mapped=True)
    assert C.capture_rounds_x(dict(graph=dict(capture_passes=[1]), tent=dict(calls=dict(j1_passes=2))))["mapped"] is False
    assert C.excluded_rounds(dict(a=dict(joint=[1], draft=[4]), b=dict(joint=[1], draft=[]))) == [1, 4]
    G = 20                                                             # THE L61 FIX on a synthetic record
    ms = [300.0] * G
    ms[1] = 7000.0                                                     # the capture round
    rdx = dict(rounds=G, round_committed=[600] * G, round_ms=ms, wall_s=10.0, count=[600] * 2,
               phases=[[0.1, 200.0 if g != 1 else 6900.0, 6.0, 90.0] for g in range(G)])
    sh = dict(rates=dict(steady_step_ms=60.0, steady_tok_s=3600.0, effective_tok_s=3000.0), steady=dict(mean_ms=60.0))
    rec = dict(rates=C.SC.round_rates(rdx["round_committed"], ms, C.SC.steady_rounds(G, 16), 10.0, 1200),
               phases_median=dict(joint=200.0))
    mean0 = rec["rates"]["mean_tok_s"]
    C.restrict(rec, rdx, sh, 216, 3, 16, [1])
    assert rec["steady_excluded_rounds"] == [1] and rec["steady_rounds_used"] == G - 3        # rounds 0, 1, 16 out
    assert rec["rates"]["mean_tok_s"] == 2000.0 and rec["rates_incl_capture"]["mean_tok_s"] == mean0 < 2000.0
    assert rec["phases_median"]["joint"] == 200.0 and rec["draft_step_ms"] == 45.0 and rec["joint_sh_steps"] == 3.3333
    assert C.effective_wo_capture(rec, dict(rdx, graph=dict(capture_s=2.0), dgraph=dict(dcapture_s=3.0))) == 240.0
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh2, {}, 216) == [("L4PTFDGD-RUN", "NOT_RUN", "the arm did not run")]
    mk = lambda tok, dr, jt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                  vs_strongest=0.5, gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                  graph=dict(pool_bytes=50e6, capture_s=3.0), dgraph=dict(dcapture_s=4.0))
    saved = {b: dict(C.PRED_I4H_BY_B[b]) for b in C.PRED_I4H_BY_B}
    try:
        for b in C.PRED_I4H_BY_B:
            C.PRED_I4H_BY_B[b].update({k: None for k in C.PRED_KEYS})
        v = {x[0]: x for x in C.verdicts(sh2, dict(L4PTFD=mk(400.0, 140.0, 200.0), L4PTFDG=mk(410.0, 140.0, 192.0),
                                                   L4PTFDGD=mk(420.0, 136.0, 192.0)), 216, eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4H_BY_B[b].clear()
            C.PRED_I4H_BY_B[b].update(saved[b])
    assert all(x[1] == "REPORT" for x in v.values()), v                # registration None: every key REPORT
    assert "4.0000" in v["I4H-DRAFT-SAVING"][2] and "1.0000" in v["I4H-JOINT-RATIO"][2] and "1.0244" in v[
        "I4H-VS-G"][2] and "1.0500" in v["I4H-VS-D"][2] and "50.0000" in v["I4H-POOL-MB"][2], v
    assert set(v) == {"I4H-DRAFT-SAVING", "I4H-DRAFT-RATIO", "I4H-JOINT-RATIO", "I4H-VS-G", "I4H-VS-D", "I4H-RATIO",
                      "I4H-PEAK-GB", "I4H-POOL-MB", "I4H-CAPTURE-S", "I4H-EQ", "I4H-VS-STRONGEST", "I4G-VS-D"}


def test_i4h_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4H is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4H_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4H)
    assert set(C.PRED_I4H_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4H_BY_B.values())
    src = (EFF / "r2_i4h_cell.py").read_text()
    assert ("\nPRED_STAMP_I4H = None\n" in src) == (C.PRED_STAMP_I4H is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4h_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i4h():
    src = (EFF / "r2_i4h_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "benchmarks/Efficiency/r2_i4h_graph.py" in src and "OUTROOT/sp_active_$JOB" in src
    assert "benchmarks/Efficiency/r2_i4g_graph.py" in src and "benchmarks/Efficiency/r2_i4g_cell.py" in src
    for bad in ("i4gm", "i4ga", "i4h", "i4hc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4H = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4h_cell.py" % sha).stdout
    ran = 0
    for cell, pool, batch, steps, check, stress, smin, mem in (
            ("i4hm", 10, 8, 72, 1, 1, 40, "100G"),
            ("i4ha", 65, 216, 512, 0, 0, 110, "240G"),
            ("i4hb", 10, 318, 512, 0, 0, 160, "340G")):
        r = _step(cell, sha)
        if cell in ("i4ha", "i4hb") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4H is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.stderr[-400:])
        ran += 1
        line = ("cell %s: C %d (pool %d) B %d arms SH+L4PTFD,L4PTFDG,L4PTFDGD steps %d K 3 fork-store 0 check %d stress "
                "%d STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4h_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4h_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms L4PTFD\\?,L4PTFDG\\?,L4PTFDGD ", r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", "L4PTFD,L4PTFDG,L4PTFDGD", "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == ["L4PTFD", "L4PTFDG", "L4PTFDGD"]              # every cell's arguments parse
    assert ran >= 1
    print("I4H DRYRUN: %d cell(s) printed their step command (registered %s)" % (ran, registered))


def test_notes_i4h():
    doc = (ROOT / "docs/NOTES_i4h.md").read_text()
    for word in ("test_i4h.py", "PRED_I4H_BY_B", "PRED_STAMP_I4H", "INFERENCE", "MEASURED", "DERIVED", "CODE", "MODEL",
                 "i4hm", "i4ha", "i4hb", "84.0", "0.36", "spec_tentative.py:461-466", "expected_delta", "FakeGraphsD",
                 "STEP_MIN", "L61", "capture_rounds", "i4c_c3a_2195575_i4ca", "i4c_c4b_2195576_i4cb", "draftgap.py"):
        assert word in doc, word
