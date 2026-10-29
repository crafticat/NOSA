"""CPU tests of rung I4G (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), lever 3: the
arm L4PTFDG = L4PTFD with the per-layer union + fork section of the joint pass replayed from CUDA graphs
(benchmarks/Efficiency/r2_i4g_graph.py; the cell r2_i4g_cell.py; the step script r2_i4g_step.sh). The world:
tests_cpu/i4g_world.py over i4a_world (FakeGraphs: a CUDA graph stand-in that captures without executing and replays the
recorded ops on the captured tensors). CUDA GRAPHS DO NOT RUN ON THE CPU: these tests establish the static-input
plumbing, the host-count bookkeeping, the frozen branches, the address rule and the eager fallback; only the GPU
miniature (i4gm) establishes capture legality on the device, the replay's bits on the GPU, timing and memory.

GATE 0    test_i4g_gate0_files: the tree = I4A's acbe433 except the I4G files and ONE module-level declaration block in
          each older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py, test_i4a.py; in
          test_i4a.py also ONE line of test_i4a_gate0_files, as I4C did); nosi/ and every I4A file are acbe433's bytes;
          the fork is fia6's blob; the graph module imports torch and time only (no kernel); nobody else names it.
FAKE      test_fake_graphs_semantics: the stand-in's three properties and its ILLEGAL guard (each with a control).
EQUAL     test_l4ptfdg_equals_l4ptfd (timing B 8, check B 8, timing B 2 rowgroup): L4PTFDG's run = L4PTFD's run BIT FOR
          BIT (tokens, fingerprints, read() except the host clock and the graph report, every engine / twin / buffer
          tensor, the union stats, the meter's blocks) and the cell's L4PTFDG-EQF, -COUNTS, -GRAPH pass on the CPU
          evidence. Preconditions (L34): L graphs captured, replays after the capture pass, probe passes eager between
          replays (timing), the checks and the "F" probe on (check).
PLANTS    test_graph_plants_caught: stale_input (caught by EQF), no_counts (caught by COUNTS and ATTN1 / FILL1), no_key
          (caught by COUNTS: the check-only counters of a check round), plain_stats (the stats lose their addresses).
COUNT     test_i4g_eager_ops: an INDEPENDENT count of the eager launches of a joint pass (a replay = one launch).
WIRING    test_graph_refusals_and_no_host_sync.
CELL      test_i4g_cell_arguments, test_i4g_cell_pure_parts, test_i4g_registration_state, test_step_script_i4g,
          test_notes_i4g.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4g.py
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
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
I4A_TIP = "acbe433da7f07db7f7b08791c036372d6d8db885"        # nosi-r2-i4a: rung I4A with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4g"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4G_NEW = ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
           "benchmarks/Efficiency/r2_i4g_step.sh", T_ + "i4g_world.py", T_ + "test_i4g.py", "docs/NOTES_i4g.md")
I4G_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py",
            T_ + "test_i4a.py")
# I4K (nosi-r2-i4k: I4D 99c231e + I4G d7997e6 + the arm L4PTFDGKW): the later files in this tree (their own Gate 0:
# tests_cpu/test_i4d.py, tests_cpu/test_i4k.py); I4G_LATER: the later modules that name the I4G modules (scan (e) skips
# them)
I4G_NEW = I4G_NEW + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                     "benchmarks/Efficiency/r2_i4d_step.sh", T_ + "test_i4d.py", "benchmarks/Efficiency/r2_i4k_combo.py",
                     "benchmarks/Efficiency/r2_i4k_cell.py", "benchmarks/Efficiency/r2_i4k_step.sh",
                     T_ + "i4k_world.py", T_ + "test_i4k.py")                                    # I4K
I4G_LATER = ("r2_i4k_combo.py", "r2_i4k_cell.py")                                                # I4K
I4A_FILES = ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
             "benchmarks/Efficiency/r2_i4a_step.sh", T_ + "i4a_world.py", "docs/NOTES_i4a.md")
TIMING_KEYS = {"phases", "round_ms", "wall_s", "prelude_ms", "probe_wait_ms", "graph"}
# I4H (nosi-r2-i4h: I4G d7997e6 + the arm L4PTFDGD): its files (its own Gate 0: tests_cpu/test_i4h.py); I4G_LATER: the
# later modules that import or name the I4G modules (the name scan (e) skips them)
I4G_NEW = I4G_NEW + ("benchmarks/Efficiency/r2_i4h_graph.py", "benchmarks/Efficiency/r2_i4h_cell.py",
                     "benchmarks/Efficiency/r2_i4h_step.sh", T_ + "i4h_world.py", T_ + "test_i4h.py",
                     "docs/NOTES_i4h.md")                                                        # I4H
I4G_LATER = I4G_LATER + ("r2_i4h_cell.py", "r2_i4h_graph.py")                                    # I4H


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4g_world")
GR, W, RW, L, N0 = X.GR, X.W, X.RW, X.L, X.N0


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4g_cell
    return r2_i4g_cell


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
def test_i4g_gate0_files():
    if not _have(I4A_TIP):
        pytest.skip("no git or no commit %s in %s" % (I4A_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I4A_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4G_NEW) | set(I4G_DECL), sorted(changed - set(I4G_NEW) - set(I4G_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4G_NEW), [f for f in I4G_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I4A_TIP, "--", "nosi").stdout.strip() == ""   # (b) nosi: acbe433's bytes
    assert _git("diff", "--name-only", I4A_TIP, "--", *I4A_FILES).stdout.strip() == ""   # L4PTFD: I4A's code
    for f in I4G_DECL:                                                    # (c) one added declaration block each
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I4A_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        allowed = {"test_i4a_gate0_files"} if f.endswith("test_i4a.py") else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I4A_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert body and len(minus) <= (1 if f.endswith("test_i4a.py") else 0), (f, minus)
        assert any("r2_i4g_graph.py" in x for x in body) and any("I4G" in x for x in body), (f, body[:5])
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4g_graph.py").read_text()                         # (d) the graph module: torch and time only
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch", "time"}, mods
    assert "@triton.jit" not in src and "import triton" not in src      # NO NEW KERNEL
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4g_") or p.name in I4G_LATER:
            continue
        assert "r2_i4g" not in p.read_text(), p.name


# ------------------------------------------------------------------ the stand-in
class _Obj:
    pass


def test_fake_graphs_semantics():
    with torch.inference_mode():
        a = torch.zeros(4)
        xs = torch.ones(4)
        host = torch.zeros(1)
        seen = []
        cnt = dict(n=0)

        def kern(t):                                                  # a kernel stand-in that reads on the host
            seen.append(float(t.sum()))
            host.add_(t.sum())
        T, vs = _Obj(), _Obj()
        T.meter, vs.ops = _Obj(), _Obj()
        T.meter.orig, vs.ops.fork = (kern, kern), kern

        def fn():
            cnt["n"] += 1                                              # a host count: moves at the capture only
            y = xs * 2                                                 # out of place
            a.add_(y)                                                  # a write
            T.meter.orig[0](a)                                         # a kernel node
            return a.view(2, 2)                                        # a view
        G = X.FakeGraphs()
        g, out = G.capture(fn, T, vs)
        assert torch.equal(a, torch.zeros(4)) and torch.equal(host, torch.zeros(1))   # 1: the capture executes nothing
        assert cnt["n"] == 1 and T.meter.orig == (kern, kern) and vs.ops.fork is kern
        g.replay()
        assert torch.equal(a, torch.full((4,), 2.0)) and torch.equal(out, a.view(2, 2))
        xs.fill_(3.0)                                                  # 2: the replay reads the static input now
        g.replay()
        assert torch.equal(a, torch.full((4,), 8.0)) and cnt["n"] == 1
        assert seen[-2:] == [8.0, 32.0] and float(host) == 40.0     # 3: the node reads its tensors at every replay
        xs2 = torch.ones(4)

        def stale():                                                   # CONTROL: an input not copied into the static
            a.add_(xs2)                                                # tensor: the replay keeps the captured object
        g2, _ = G.capture(stale, T, vs)
        xs2 = torch.full((4,), 100.0)                                  # noqa: F841 (a NEW tensor, as a GEMM output)
        g2.replay()
        assert torch.equal(a, torch.full((4,), 9.0))

        b = torch.zeros(2, 4)
        col = torch.ones(2)

        def copy_view():                                               # reshape of an expanded tensor COPIES
            b.add_(col.view(2, 1).expand(2, 4).reshape(2, 4))
        g3, _ = G.capture(copy_view, T, vs)
        col.fill_(5.0)
        g3.replay()
        assert torch.equal(b, torch.full((2, 4), 5.0))                # replayed out of place, not as a stale alias

        def host_read():                                               # CONTROL: a host read outside a node
            return float(a.sum())
        with pytest.raises(RuntimeError, match="not capturable"):
            G.capture(host_read, T, vs)
        assert torch.equal(a, torch.full((4,), 9.0))


# ------------------------------------------------------------------ L4PTFDG = L4PTFD, bit for bit
@functools.lru_cache(maxsize=None)
def _pair(scen, B=8, gemm="natural"):
    return X.run_arm(False, scen, B=B, gemm=gemm), X.run_arm(True, scen, B=B, gemm=gemm)


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
def test_l4ptfdg_equals_l4ptfd(case):
    scen, B, gemm = CASES[case]
    d, g = _pair(scen, B, gemm)
    C = _cell()
    gr = g["rd"]["graph"]
    jp = g["rd"]["tent"]["calls"]["joint_passes"]
    print("I4G %s: rounds %d, joint passes %d; graphs %s, warm %s, replays %s, eager %s %s; capture passes %s" % (
        case, g["rd"]["rounds"], jp, gr["captures"], gr["warm"], gr["replays"], gr["eager"], gr["eager_why"],
        sorted(set(gr["capture_passes"]))))
    assert gr["captures"] == L and gr["warm"] == L and gr["replays"] >= 3 * L, gr      # preconditions (L34)
    assert jp >= 5 and gr["deltas"] and not gr["delta_bad"], gr
    if scen == "timing":
        assert gr["eager_why"].get("marks", 0) >= L and jp >= 6, gr     # probe passes eager BETWEEN replays
        assert max(gr["capture_passes"]) < 4 <= jp - 2                 # the capture before probe round 4, replays after
    else:
        u = g["rd"]["union"]
        assert gr["eager_why"].get("probe_F") == 1 and u["vb_rows"] > 0 and u["chain_rows"] > 0, (gr, u)
        assert u["vb_bad"] == 0 and u["dropped"] == 0 and u["chain_bad"] == 0 and u["tail_reads"] == 0, u
    bad = _diff(d, g)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    fd, fg = X.cell_inputs(d, C), X.cell_inputs(g, C)
    n, m, ok, det = C.eq_gate(fg, fd, N0)
    assert (n, m, ok) == ("L4PTFDG-EQF", True, True), det
    n, m, ok, det = C.counts_gate("L4PTFDG", fg["counts"], fd["counts"])
    assert ok, det
    n, m, ok, det = C.graph_gate("L4PTFDG", g["rd"], L)
    assert ok, det
    assert C.graph_gate("L4PTFD", d["rd"], L)[2] is False               # CONTROL: L4PTFD has no graph
    lg = C.IA.lean_gate("L4PTFDG", dict(g["rd"], K=X.K0), L)
    assert lg[2], lg


# ------------------------------------------------------------------ PLANTS
def test_graph_plants_caught():
    C = _cell()
    d, g = _pair("timing")
    fd = X.cell_inputs(d, C)
    p = X.run_arm(True, "timing", gneg="stale_input")                  # stale_input: the replay reads an old q
    assert _diff(d, p) != []                                           # not a no-op (L34)
    ok = C.eq_gate(X.cell_inputs(p, C), fd, N0)[2]
    print("I4G PLANT stale_input: EQF %s" % ok)
    assert ok is False
    p = X.run_arm(True, "timing", gneg="no_counts")                    # no_counts: the skipped host counts
    fp_ = X.cell_inputs(p, C)
    n, m, ok, det = C.counts_gate("L4PTFDG", fp_["counts"], fd["counts"])
    fg = {x[0]: x[2] for x in C.I3C.fork_gates("L4PTFDG", dict(read=p["rd"]), L, False)}
    print("I4G PLANT no_counts: COUNTS %s (%s); fork gates %s; EQF %s" % (ok, det[:160], fg,
                                                                         C.eq_gate(fp_, fd, N0)[2]))
    assert ok is False and fg["L4PTFDG-ATTN1"] is False and fg["L4PTFDG-FILL1"] is False
    dt, gt = _pair("toggle")                                           # no_key: the check round's frozen branch
    assert gt["rd"]["graph"]["eager_why"].get("key", 0) > 0            # precondition: the key toggles
    assert _diff(dt, gt) == [] and dt["rd"]["tent"]["fill_rows"] + dt["rd"]["tent"]["edge_rows"] > 0
    assert C.graph_gate("L4PTFDG", gt["rd"], L)[2] is False            # a key change is not an expected eager reason
    p = X.run_arm(True, "toggle", gneg="no_key")
    fdt = X.cell_inputs(dt, C)
    ok = C.counts_gate("L4PTFDG", X.cell_inputs(p, C)["counts"], fdt["counts"])[2]
    print("I4G PLANT no_key: differs %s; COUNTS %s" % (_diff(dt, p)[:4], ok))
    assert ok is False
    dc, gc_ = _pair("check")                                           # plain_stats: the address rule
    assert isinstance(gc_["runner"].vs.stat, GR.InPlaceDict)
    p = X.run_arm(True, "check", gneg="plain_stats")
    keys0 = ("extras_max", "union_max")
    moved = [k for k in keys0 if id(p["runner"].vs.stat[k]) != p["runner"]._ids0[k]]
    kept = [k for k in gc_["runner"]._ids0 if id(gc_["runner"].vs.stat[k]) == gc_["runner"]._ids0[k]]
    vals = {k: (int(p["runner"].vs.stat[k]), int(dc["runner"].vs.stat[k])) for k in keys0}
    print("I4G PLANT plain_stats: entries rebound %s; (plant, L4PTFD) values %s; InPlaceDict ids kept %d of %d" % (
        moved, vals, len(kept), len(gc_["runner"]._ids0)))
    assert moved == list(keys0)                                        # the plant rebinds both (not a no-op)
    assert len(kept) == len(gc_["runner"]._ids0)                       # the hook keeps every address


@pytest.fixture(autouse=True)
def _record_ids(monkeypatch):
    """The ids of vs.stat's entries right after the runner is built (the address rule's reference)."""
    orig = X.TentRunnerFDG.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self._ids0 = {kk: id(v) for kk, v in self.vs.stat.items()}
    monkeypatch.setattr(X.TentRunnerFDG, "__init__", init)
    yield


# ------------------------------------------------------------------ the eager launches
def test_i4g_eager_ops():
    boxes = {}
    for graph in (False, True):
        boxes[graph] = {}
        try:
            ev = X.run_arm(graph, "timing", on_runner=lambda r, g=graph: boxes[g].update(box=X.eager_op_count(r)))
        finally:
            X.restore_replay()
    cap = set(ev["rd"]["graph"]["capture_passes"])
    d, g = boxes[False]["box"], boxes[True]["box"]
    rep = [i for i, n in enumerate(g["replays"]) if n == L and i not in cap]   # every layer replayed, no capture
    assert rep and len(d["joint"]) == len(g["joint"]), (g["replays"], len(d["joint"]))
    per_layer = [(d["joint"][i] - g["joint"][i]) / L for i in rep]
    print("I4G EAGER OPS per joint pass: L4PTFD %s; L4PTFDG %s (replays %s); removed per layer in replayed passes %s" % (
        d["joint"], g["joint"], g["replays"], per_layer))
    assert all(x > 0 for x in per_layer)
    assert all(g["joint"][i] + g["replays"][i] < d["joint"][i] for i in rep)


# ------------------------------------------------------------------ WIRING
def test_graph_refusals_and_no_host_sync():
    model, c, S, hz, tok0 = X.world(2, 8)
    with pytest.raises(X.SR.SpecError):
        X.gfd_builder(True, gneg="nope")(model, c, S, hz, tok0)
    model, c, S, hz, tok0 = X.world(2, N0)
    run = X.gfd_builder(True, probe_every=0)(model, c, S, hz, tok0, check_every=0)
    assert isinstance(run.T, X.TentStateFDG) and isinstance(run, X.TentRunnerFD)
    with RW.SyncTrap(Path(GR.__file__)) as trap:
        run.run(N0)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the hook
    assert run.T.graph["replays"] > 0 and run.meta["section_graph"]["backend"] == "FakeGraphs"
    run.detach()
    assert run.T._g == {} and run.T._sb is None                        # the arm's end drops the graphs


# ------------------------------------------------------------------ CELL
def test_i4g_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.fork_store, a.prefill_hook, a.tu_gemm, a.check_every,
            a.require_stress, a.probe_every) == (["L4PTFD", "L4PTFDG"], 216, 512, 65, 3, 0, "nocopy", "natural", 0, 0,
                                                 16)
    for bad in (["--arms", "L4PTFDG,L4PTFD"], ["--arms", "L4PTF"], ["--spec", "4"], ["--fork-store", "100"],
                ["--arms", "L4PTFD,L4PTFD"], ["--pool", "0"], ["--steps", "4"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1",
                             "--require-stress", "1"])
    assert (a.check_every, a.require_stress) == (1, 1)
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4g_cell_pure_parts():
    C = _cell()
    d, g = _pair("timing")
    fd, fg = X.cell_inputs(d, C), X.cell_inputs(g, C)
    assert C.eq_gate(fg, fd, N0)[2] is True
    planted = dict(fg, out=fg["out"].clone())
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    n, m, ok, det = C.eq_gate(planted, fd, N0)
    assert (n, m, ok) == ("L4PTFDG-EQF", True, False) and "L4PTFD's" in det, det
    assert C.eq_gate(fg, None, N0)[2] is False and "no L4PTFD record" in C.eq_gate(fg, None, N0)[3]
    pc = dict(fg["counts"], tent=dict(fg["counts"]["tent"], calls=dict(fg["counts"]["tent"]["calls"], chain=0)))
    assert C.counts_gate("L4PTFDG", pc, fd["counts"])[2] is False      # CONTROL: one host count changed
    assert C.counts_gate("L4PTFDG", fg["counts"], None)[2] is False
    rd = dict(g["rd"], graph=dict(g["rd"]["graph"], eager_why=dict(g["rd"]["graph"]["eager_why"], key=1),
                                  eager=g["rd"]["graph"]["eager"] + 1))
    assert C.graph_gate("L4PTFDG", rd, L)[2] is False                   # CONTROL: an unexpected eager layer
    rd = dict(g["rd"], graph=dict(g["rd"]["graph"], delta_bad=[(0, {}, {})]))
    assert C.graph_gate("L4PTFDG", rd, L)[2] is False                   # CONTROL: a skipped-count mismatch
    rec = dict(gpu_peak_alloc_GB=82.57, mem_arena=dict(prefill_peak_GB=78.13, decode_peak_reserved_GB=77.1),
               graph=dict(pool_bytes=12_000_000, static_bytes=8_000_000))
    n, m, ok, det = C.mem_gate("L4PTFDG", rec)
    assert (n, m, ok) == ("L4PTFDG-MEM", True, True) and "83.891 GB" in det and "graph pool 12.0 MB" in det, det
    rec["mem_arena"] = dict(prefill_peak_GB=78.3)
    assert C.mem_gate("L4PTFDG", rec)[2] is False                       # 84.073 GB > 84.0
    assert C.capture_rounds(dict(graph=dict(capture_passes=[1, 1, 2]), phases=[[0, 5.0], [0, 900.0], [0, 7.0]])) == [
        (1, 900.0), (2, 7.0)]
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh, {}, 216) == [("L4PTFDG-RUN", "NOT_RUN", "the arm did not run")]
    rec = lambda tok, dr, jt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                   tokens=dict(equal_fraction=0.4, first=(1, 1)), vs_strongest=0.5,
                                   gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                   graph=dict(pool_bytes=5e6, capture_s=3.0))
    saved = {b: dict(C.PRED_I4G_BY_B[b]) for b in C.PRED_I4G_BY_B}
    try:
        for b in C.PRED_I4G_BY_B:
            C.PRED_I4G_BY_B[b].update({k: None for k in C.PRED_KEYS})
        v = {x[0]: x for x in C.verdicts(sh, dict(L4PTFD=rec(400.0, 20.0, 200.0), L4PTFDG=rec(410.0, 20.0, 192.0)),
                                         216, eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4G_BY_B[b].clear()
            C.PRED_I4G_BY_B[b].update(saved[b])
    assert all(x[1] == "REPORT" for x in v.values()), v                # registration None: every key REPORT
    assert "8.0000" in v["I4G-JOINT-SAVING"][2] and "0.9600" in v["I4G-JOINT-RATIO"][2] and "1.0250" in v[
        "I4G-VS-D"][2] and "5.0000" in v["I4G-POOL-MB"][2], v
    assert set(v) == {"I4G-JOINT-SAVING", "I4G-JOINT-RATIO", "I4G-VS-D", "I4G-RATIO", "I4G-DRAFT-RATIO",
                      "I4G-PEAK-GB", "I4G-POOL-MB", "I4G-CAPTURE-S", "I4G-EQ", "I4G-VS-STRONGEST"}


def test_i4g_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4G is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4G_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4G)
    assert set(C.PRED_I4G_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4G_BY_B.values())
    src = (EFF / "r2_i4g_cell.py").read_text()
    assert ("\nPRED_STAMP_I4G = None\n" in src) == (C.PRED_STAMP_I4G is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4g_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i4g():
    src = (EFF / "r2_i4g_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "benchmarks/Efficiency/r2_i4g_graph.py" in src and "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4am", "i4aa", "i4g", "i4gc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4G = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4g_cell.py" % sha).stdout
    for cell, pool, batch, steps, check, stress, smin, mem in (
            ("i4gm", 10, 8, 72, 1, 1, 40, "100G"),
            ("i4ga", 65, 216, 512, 0, 0, 80, "240G"),
            ("i4gb", 10, 318, 512, 0, 0, 110, "340G")):
        r = _step(cell, sha)
        if cell in ("i4ga", "i4gb") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4G is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.stderr[-400:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L4PTFD,L4PTFDG steps %d K 3 fork-store 0 check %d stress %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4g_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4g_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert "--eq-must" not in r.stdout
        assert re.search(r"--arms L4PTFD\\?,L4PTFDG ", r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", "L4PTFD,L4PTFDG", "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == ["L4PTFD", "L4PTFDG"]                          # every cell's arguments parse


def test_notes_i4g():
    doc = (ROOT / "docs/NOTES_i4g.md").read_text()
    for word in ("test_i4g.py", "PRED_I4G_BY_B", "PRED_STAMP_I4G", "INFERENCE", "MEASURED", "DERIVED", "CODE", "i4gm",
                 "i4ga", "i4gb", "84.0", "0.359", "STF:150-186", "spec_tentative.py:461-466", "SKIPPED", "FakeGraphs",
                 "MODEL", "STEP_MIN", "i4cb"):
        assert word in doc, word
