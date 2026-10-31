"""CPU tests of rung I4U (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), lever L5 of
the specKV-on-cached-NOSA ladder: the arm L4PTFDU = L4PTFD (rung I4A) whose joint pass computes its layer-invariant
tensors once per pass (benchmarks/Efficiency/r2_i4u_hoist.py; the cell r2_i4u_cell.py; the step script
r2_i4u_step.sh). The world: tests_cpu/i4u_world.py over i4a_world / i3_world (the SHIPPED engine from its own source;
every CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). CPU stand-ins do not establish GPU
dispatch, ordering, timing or memory (the cells i4um / i4ua / i4ub do). FAST is checked against L4PTFD (bit for bit)
and against SH (tokens, TOL), not certified by an EXACT arm.

GATE 0    test_i4u_gate0_files: the tree = I4A's acbe433 except the I4U files, ONE module-level declaration block in
          each older Gate-0 test (test_wsa / wsb / wsc / i2 / i3 / i4a: only added lines) and test_i4a's name-scan skip
          line; nosi/ and every I4A / I3 file are acbe433's bytes; the fork is fia6's blob; the hoist module imports
          torch only (no kernel source); no nosi module and no older cell names it.
EQUAL     test_l4ptfdu_equals_l4ptfd (B 1, B 8 rowgroup, B 8 natural; checks and stats on) and
          test_l4ptfdu_equals_timing_config (B 8, checks and stats off: the timing cells' path): L4PTFDU's run =
          L4PTFD's BIT FOR BIT (i2_world.same_run: tokens, fingerprints, logits rows, read() incl. the trace and the
          counters, the round records, the engine and twin state, the buffers, tpos, qc, the layer-0 row writes); SH's
          tokens; one hoist per joint pass, L hoisted layers per pass, is_next recounted 0. Preconditions (L34): joint
          passes with inactive rows and rows past the committed edge.
COUNT     test_l4ptfdu_op_counts: an INDEPENDENT aten op counter (i4a_world.OpCount) per joint pass and layer: L4PTFD -
          L4PTFDU = SAVED_PER_LAYER in every layer after the first, SAVED_LAYER0 in the first, saved_per_pass(L) per
          joint pass; the one-row passes unchanged; the named removals.
PLANTS    test_hoist_plants_caught: stale_pass, row0_index, no_plast each change the run (not a no-op) and are CAUGHT by
          the cell's L4PTFDU-EQF.
WIRING    test_hoist_refusals_sync_and_graph_seams: refusals; no host read in the hoist module; the joint pass calls
          self._resid (I4G's capture stub point); the hoisted buffers keep their addresses over the run.
CELL      test_i4u_cell_arguments, test_i4u_cell_pure_parts, test_i4u_registration_state, test_step_script_i4u.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4u.py
"""
from __future__ import annotations

import ast
import collections
import functools
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
I4A_TIP = "acbe433da7f07db7f7b08791c036372d6d8db885"         # nosi-r2-i4a: rung I4A with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4u"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4U_NEW = ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
           "benchmarks/Efficiency/r2_i4u_step.sh", T_ + "i4u_world.py", T_ + "test_i4u.py")
I4U_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py",
            T_ + "test_i4a.py")
# I4S (nosi-r2-i4s: I4K 70059e5 + I4U 832dca5 + I4H fc6c680 + the stacked arms): the later files in this tree (their
# own Gate 0: tests_cpu/test_i4d.py, test_i4g.py, test_i4k.py, test_i4h.py, test_i4s.py); I4U_LATER: the later modules
# that import the hoist module (scan (e) skips them)
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                     "benchmarks/Efficiency/r2_i4d_step.sh", T_ + "test_i4d.py",
                     "benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
                     "benchmarks/Efficiency/r2_i4g_step.sh", T_ + "i4g_world.py", T_ + "test_i4g.py",
                     "docs/NOTES_i4g.md", "benchmarks/Efficiency/r2_i4k_combo.py",
                     "benchmarks/Efficiency/r2_i4k_cell.py", "benchmarks/Efficiency/r2_i4k_step.sh",
                     T_ + "i4k_world.py", T_ + "test_i4k.py", "benchmarks/Efficiency/r2_i4h_graph.py",
                     "benchmarks/Efficiency/r2_i4h_cell.py", "benchmarks/Efficiency/r2_i4h_step.sh",
                     T_ + "i4h_world.py", T_ + "test_i4h.py", "docs/NOTES_i4h.md",
                     "benchmarks/Efficiency/r2_i4s_stack.py", "benchmarks/Efficiency/r2_i4s_cell.py",
                     "benchmarks/Efficiency/r2_i4s_step.sh", T_ + "i4s_world.py", T_ + "test_i4s.py")    # I4S
I4U_LATER = ("r2_i4s_stack.py", "r2_i4s_cell.py")                                                # I4S
# I4CC (nosi-r2-i4cc: I4S 8f236f8 + the compiled arms): its files (its own Gate 0: tests_cpu/test_i4cc.py); its modules
# name the I4S modules only
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_i4cc_compile.py", "benchmarks/Efficiency/r2_i4cc_cell.py",
                     "benchmarks/Efficiency/r2_i4cc_step.sh", T_ + "i4cc_world.py", T_ + "test_i4cc.py")  # I4CC
# I4DC (nosi-r2-i4dc: I4CC d75a946 + the compiled draft helpers): its files (its own Gate 0: tests_cpu/test_i4dc.py);
# I4U_LATER: its modules may name the I4U modules (scan (f) skips them)
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_i4dc_compile.py", "benchmarks/Efficiency/r2_i4dc_cell.py",
                     "benchmarks/Efficiency/r2_i4dc_step.sh", T_ + "i4dc_world.py", T_ + "test_i4dc.py")  # I4DC
I4U_LATER = I4U_LATER + ("r2_i4dc_compile.py", "r2_i4dc_cell.py")                              # I4DC
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py); I4U_LATER: its modules and copies may name the I4U modules
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_recon7_protocol.py", "benchmarks/Efficiency/r2_recon7_probe.py",
                     "benchmarks/Efficiency/r2_recon7_cell.py", "benchmarks/Efficiency/r2_recon7_streams.py",
                     "benchmarks/Efficiency/r2_recon7_mem.py", "benchmarks/Efficiency/r2_recon7_step.sh",
                     "benchmarks/Efficiency/r2_i4c_cell.py", "benchmarks/Efficiency/r2_i4m_cell.py",
                     "benchmarks/Efficiency/r2_wsm_cell.py", T_ + "test_recon7.py", "docs/NOTES_recon7.md")  # RECON7
I4U_LATER = I4U_LATER + ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py", "r2_recon7_streams.py",
                         "r2_recon7_mem.py", "r2_i4c_cell.py", "r2_i4m_cell.py", "r2_wsm_cell.py")       # RECON7
# J1 (nosi-r2-j1: RECON7 cd0f0c5 + the compiled joint pooling of AREA 1): its files
# (its own Gate 0: tests_cpu/test_j1.py)
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_j1_select.py", "benchmarks/Efficiency/r2_j1_cell.py",
                     "benchmarks/Efficiency/r2_j1_step.sh", T_ + "test_j1.py", "docs/NOTES_j1.md")          # J1
I4U_NEW = I4U_NEW + ("benchmarks/Efficiency/r2_j1_forkbench.py",)                                         # J1


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4u_world")
WT = _load("wsa_tests_lib", HERE / "test_wsa.py")           # WS-A's test helpers (the draft filters)
X4, I, W, RW, SR, HU, LN = X.X4, X.I, X.W, X.RW, X.SR, X.HU, X.LN
K, N, L, BS, TOL = X.K0, X.N0, X.L, X.BS, X.TOL
R = K + 1


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4u_cell
    return r2_i4u_cell


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
def test_i4u_gate0_files():
    if not _have(I4A_TIP):
        pytest.skip("no git or no commit %s in %s" % (I4A_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I4A_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4U_NEW) | set(I4U_DECL), sorted(changed - set(I4U_NEW) - set(I4U_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4U_NEW), [f for f in I4U_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I4A_TIP, "--", "nosi").stdout.strip() == ""    # (b) L4PTFD's code: acbe433's
    old_files = [f for f in _git("ls-tree", "-r", "--name-only", I4A_TIP, "--", "benchmarks/Efficiency").stdout.split()
                 if ("r2_i4a" in f or "r2_i3" in f or "fork_vllm" in f or "r2_kv_arena" in f or "i4a_world" in f
                     or "i3_world" in f)]
    assert len(old_files) >= 8 and _git("diff", "--name-only", I4A_TIP, "--", *old_files).stdout.strip() == ""
    for f in I4U_DECL:                                                    # (c) one added declaration block each
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I4A_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        diff_units = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        d = _git("diff", "-U0", I4A_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        if f.endswith("test_i4a.py"):                                    # its name-scan skip line, nothing else
            assert diff_units == ["test_i4a_gate0_files"], diff_units
            minus = [x for x in body if x.startswith("-")]
            assert minus == ['-        if p.name.startswith("r2_i4a_"):'], minus
            assert '+        if p.name.startswith("r2_i4a_") or p.name in I4A_LATER:' in body, body
        else:
            assert diff_units == [], (f, diff_units)
            assert body and all(x.startswith("+") for x in body), (f, body[:5])
        assert any("r2_i4u_hoist.py" in x for x in body) and any("I4U" in x for x in body), (f, body[:5])
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4u_hoist.py").read_text()
    tree = ast.parse(src)                                                 # (d) the hoist module: torch only
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch"}, mods
    assert "triton" not in src and "load_inline" not in src and "cpp_extension" not in src      # NO NEW KERNEL
    assert not any(f.endswith((".cu", ".cpp", ".cuh")) or "triton" in f for f in changed), sorted(changed)
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4u_") or p.name in I4U_LATER:
            continue
        assert "r2_i4u" not in p.read_text(), p.name


# ------------------------------------------------------------------ L4PTFDU = L4PTFD, bit for bit
@functools.lru_cache(maxsize=None)
def _pair(B, gemm="rowgroup", filt="periodic", check=True):
    """(L4PTFD's evidence, L4PTFDU's evidence) of one world; L4PTFDU's carries the joint-pass recorder."""
    d = X.run_fd(B, gemm=gemm, filt=WT._filter(filt), check=check, stats=check)
    box = {}
    u = X.run_fdu(B, gemm=gemm, filt=WT._filter(filt), check=check, stats=check,
                  on_runner=lambda r: box.update(f=X.joint_facts(r)))
    u["facts_j"] = box["f"]
    return d, u


CASES = dict(B1=(1, "rowgroup"), B8=(8, "rowgroup"), B8nat=(8, "natural"))


def _equal_checks(d, u, B, check):
    rd, jf = u["rd"], u["facts_j"]
    c = rd["tent"]["calls"]
    h = rd["hoist"]
    bad = I.same_run(d, X.without_hoist(u))
    assert bad == [], bad                                              # THE CLAIM: L4PTFD's run, bit for bit
    assert jf["passes"] == c["joint_passes"] > 0 and jf["inactive"] > 0, (jf, c)     # preconditions (L34)
    if B > 1:
        assert jf["past"] > 0, jf                                      # joint rows past the committed edge
    assert h["passes"] == c["joint_passes"] and h["layers"] == L * c["joint_passes"], (h, c)
    assert h["next_seen"] == 0, h                                      # the dropped is_next test: never true
    assert rd["count"] == [N] * B, rd["count"]                        # exactly N (contract 4)
    assert u["bad_tokens"] == [] and u["max_logit_delta"] <= TOL, (u["bad_tokens"][:3], u["max_logit_delta"])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    if check:
        un = rd["union"]
        assert un["dropped"] == 0 and un["vb_bad"] == 0 and un["tail_reads"] == 0 and un["chain_bad"] == 0, un
    assert "hoist" not in d["rd"] and u["runner"].meta["joint_hoist"]["hneg"] is None


@pytest.mark.parametrize("case", list(CASES))
def test_l4ptfdu_equals_l4ptfd(case):
    B, gemm = CASES[case]
    d, u = _pair(B, gemm)
    print("I4U %s: max |logit - SH's| %.3e (TOL %.0e); joint passes %d: inactive rows %d, rows past the edge %d, "
          "inactive rows at an active row's position %d; hoist %s" % (
              case, u["max_logit_delta"], TOL, u["facts_j"]["passes"], u["facts_j"]["inactive"], u["facts_j"]["past"],
              u["facts_j"]["dup"], u["rd"]["hoist"]))
    _equal_checks(d, u, B, True)


def test_l4ptfdu_equals_timing_config():
    d, u = _pair(8, "rowgroup", "periodic", False)                     # check and stats OFF (the timing cells' path)
    assert u["runner"].vs.ops.check is False and u["runner"].vs.ops.stats is False
    print("I4U timing config B 8: max |logit - SH's| %.3e; joint facts %s; hoist %s" % (
        u["max_logit_delta"], u["facts_j"], u["rd"]["hoist"]))
    _equal_checks(d, u, 8, False)


# ------------------------------------------------------------------ the op counts
def _count(arm, B=4):
    model, c, S, hz, tok0 = X4.small_world(B=B)
    build = (X4.tentfd_builder(check=False, stats=False) if arm == "L4PTFD" else
             X.tentfdu_builder(check=False, stats=False))
    run = build(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)
    box = X.count_joint_ops(run)
    run.run(8, rounds=3)
    return box, run


def test_l4ptfdu_op_counts():
    bd, rd_ = _count("L4PTFD")
    bu, ru = _count("L4PTFDU")
    assert torch.equal(rd_.out, ru.out) and torch.equal(W.bits(rd_.fmax), W.bits(ru.fmax))   # the same run
    nj = len(bd["joint"])
    assert nj > 0 and len(bu["joint"]) == nj and len(bd["layers"]) == nj == len(bu["layers"]), (nj, bu["joint"])
    assert all(len(x) == L for x in bd["layers"] + bu["layers"])
    assert rd_.rs.pool and ru.rs.pool                                   # the stated counts are for the pool on
    dl = [[a - b for a, b in zip(x, y)] for x, y in zip(bd["layers"], bu["layers"])]
    dp = [a - b for a, b in zip(bd["joint"], bu["joint"])]
    gone = collections.Counter()
    for pd, pu in zip(bd["names"], bu["names"]):
        for nd, nu in zip(pd[1:], pu[1:]):                              # layers >= 1
            gone.update({k: nd[k] - nu.get(k, 0) for k in set(nd) | set(nu) if nd.get(k, 0) != nu.get(k, 0)})
    print("I4U COUNT: %d joint passes of L %d; aten compute ops per joint pass L4PTFD %s, L4PTFDU %s; _rows per layer "
          "L4PTFD %s, L4PTFDU %s; L4PTFD - L4PTFDU per layer %s, per pass %s (stated %d per layer, %d in layer 0, %d "
          "per pass); one-row passes %s / %s; ops removed by name in layers >= 1 (all passes) %s" % (
              nj, L, bd["joint"][:3], bu["joint"][:3], bd["layers"][0], bu["layers"][0], dl[0], dp[:3],
              HU.SAVED_PER_LAYER, HU.SAVED_LAYER0, HU.saved_per_pass(L), bd["one"][:2], bu["one"][:2],
              dict(sorted(gone.items()))))
    assert all(x[0] == HU.SAVED_LAYER0 and all(y == HU.SAVED_PER_LAYER for y in x[1:]) for x in dl), dl[:3]
    assert all(x == HU.saved_per_pass(L) for x in dp), dp[:5]           # nothing outside _rows changed
    assert bd["one"] == bu["one"] and len(bd["one"]) > 0                 # the one-row passes: L4PTFD's
    assert HU.saved_per_pass(32) == HU.SAVED_PER_LAYER * 31 + HU.SAVED_LAYER0
    per = nj * (L - 1)                                                   # the named removals, per layer >= 1
    assert gone.get("stack") == per and gone.get("clamp") == R * per, gone
    assert all(v > 0 for v in gone.values()), gone                       # only removals, no op added per layer


# ------------------------------------------------------------------ PLANTS
def _cpu(ev):
    """The cell's EQF input (r2_i4a_cell.eq_gate) from a CPU run."""
    r = ev["runner"]
    IA = _cell().IA
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=IA.acct_record(ev["rd"]),
                state=IA.state_checksum(r))


@pytest.mark.parametrize("hneg", ["stale_pass", "row0_index", "no_plast"])
def test_hoist_plants_caught(hneg):
    d, u = _pair(8)
    p = X.run_fdu(8, filt=WT._filter("periodic"), hneg=hneg)
    bad = I.same_run(d, X.without_hoist(p))
    sh_caught = p["bad_tokens"] != [] or p["max_logit_delta"] > TOL
    n, m, ok, det = _cell().eq_gate(_cpu(p), _cpu(d), N)
    print("I4U PLANT %s: differs from L4PTFD in %s; caught by SH's tokens / TOL %s (bad tokens %d, max |logit - SH's| "
          "%.3e); EQF: %s" % (hneg, [b if isinstance(b, str) else b[0] for b in bad][:8], sh_caught,
                              len(p["bad_tokens"]), p["max_logit_delta"], det))
    assert bad != [], hneg                                             # not a no-op (L34)
    assert (n, m) == ("L4PTFDU-EQF", True) and ok is False, (hneg, det)   # CAUGHT by L4PTFDU-EQF


# ------------------------------------------------------------------ WIRING
def test_hoist_refusals_sync_and_graph_seams():
    model, c, S, hz, tok0 = X4.small_world(B=2)
    with pytest.raises(SR.SpecError):
        X.tentfdu_builder(hneg="nope")(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)
    model, c, S, hz, tok0 = X4.small_world(B=2)
    run = X.tentfdu_builder(check=False, stats=False)(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0,
                                                      record=False)
    assert isinstance(run.T, X.TentStateFDU) and isinstance(run, X.TentRunnerFD) and isinstance(run.T, X.TentStateFD)
    T = run.T
    seen = dict(resid=0, ptrs=None, same=True)
    resid0 = T._resid

    def resid(*a, **k):                                                # I4G's capture stub point: an instance attribute
        seen["resid"] += 1
        return resid0(*a, **k)
    T._resid = resid
    rows0 = T._rows

    def rows(l, eng, twin, sel, q4, k4, v4, pbs, plan):
        out = rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        if plan.R > 1 and l == 0:                                      # the hoisted buffers keep their addresses
            hp = T._cur
            p = [t.data_ptr() for t in (hp.qb, hp.qb1, hp.P, hp.past, hp.full, hp.res, hp.fullv, hp.i, hp.prow, hp.low,
                                        hp.nam, run.vs.plast)]
            if seen["ptrs"] is None:
                seen["ptrs"] = p
            seen["same"] = seen["same"] and p == seen["ptrs"]
        return out
    T._rows = rows
    with RW.SyncTrap(Path(HU.__file__)) as trap:
        run.run(8, rounds=3)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the hoist module
    jp = T.calls["joint_passes"]
    assert jp > 1 and seen["resid"] == L * jp, (seen, jp)              # the joint pass's residency = self._resid
    assert seen["ptrs"] is not None and seen["same"], seen
    assert T.hoist["passes"] == jp and T.hoist["layers"] == L * jp
    assert run.read()["hoist"] == dict(T.hoist, next_seen=0)
    assert run.meta["joint_hoist"] == dict(hneg=None, saved_per_layer=HU.SAVED_PER_LAYER,
                                           saved_layer0=HU.SAVED_LAYER0)
    with RW.SyncTrap(Path(__file__)) as trap:                          # CONTROL: the trap is not blind
        torch.zeros(2).sum().item()
    assert [h[2] for h in trap.hits] == ["item"], trap.hits


# ------------------------------------------------------------------ CELL
def test_i4u_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.fork_store, a.prefill_hook, a.tu_gemm, a.check_every,
            a.require_stress, a.probe_every) == (["L4PTFD", "L4PTFDU"], 216, 512, 65, 3, 0, "nocopy", "natural", 0, 0,
                                                 16)
    for bad in (["--arms", "L4PTFDU,L4PTFD"], ["--arms", "L4PTF"], ["--spec", "4"], ["--fork-store", "100"],
                ["--arms", "L4PTFD,L4PTFD"], ["--pool", "0"], ["--steps", "4"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1",
                             "--require-stress", "1"])
    assert (a.check_every, a.require_stress) == (1, 1)
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4u_cell_pure_parts():
    C = _cell()
    d, u = _pair(8)
    n, m, ok, det = C.eq_gate(_cpu(u), _cpu(d), N)
    assert (n, m, ok) == ("L4PTFDU-EQF", True, True), det
    planted = _cpu(u)
    planted["state"] = planted["state"][:-1] + [planted["state"][-1] + 1]
    assert C.eq_gate(planted, _cpu(d), N)[2] is False                   # CONTROL: one state checksum changed
    assert C.eq_gate(_cpu(u), None, N)[2] is False
    cu, cd = C.counts_record(u["rd"]), C.counts_record(d["rd"])
    n, m, ok, det = C.counts_gate("L4PTFDU", cu, cd)
    assert (n, m, ok) == ("L4PTFDU-COUNTS", True, True), det
    assert cu["tent"] is not None and cu["union"] is not None and cu["lean"] is not None
    pl = dict(cu, tent=dict(cu["tent"], calls=dict(cu["tent"]["calls"], chain=cu["tent"]["calls"]["chain"] + 1)))
    assert C.counts_gate("L4PTFDU", pl, cd)[2] is False                  # CONTROL: one host count changed
    assert C.counts_gate("L4PTFDU", cu, None)[2] is False
    n, m, ok, det = C.hoist_gate("L4PTFDU", u["rd"], L, True)
    assert (n, m, ok) == ("L4PTFDU-HOIST", True, True), det
    bad = dict(u["rd"], hoist=dict(u["rd"]["hoist"], next_seen=1))
    assert C.hoist_gate("L4PTFDU", bad, L, True)[2] is False             # CONTROL: is_next seen
    assert C.hoist_gate("L4PTFDU", bad, L, False)[2] is True             # (the checks off: not read)
    bad = dict(u["rd"], hoist=dict(u["rd"]["hoist"], passes=u["rd"]["hoist"]["passes"] - 1))
    assert C.hoist_gate("L4PTFDU", bad, L, False)[2] is False            # CONTROL: a pass without its hoist
    assert C.hoist_gate("L4PTFDU", d["rd"], L, False)[2] is False        # CONTROL: L4PTFD has no hoist
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh, {}, 216) == [("L4PTFDU-RUN", "NOT_RUN", "the arm did not run")]
    rec = lambda tok, dr, jt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                   tokens=dict(equal_fraction=0.4, first=(1, 1)), vs_strongest=0.5,
                                   gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0))
    saved = {b: dict(C.PRED_I4U_BY_B[b]) for b in C.PRED_I4U_BY_B}        # the REPORT check runs on an all-None block
    try:
        for b in C.PRED_I4U_BY_B:
            C.PRED_I4U_BY_B[b].update({k: None for k in C.PRED_KEYS})
        v = {x[0]: x for x in C.verdicts(sh, dict(L4PTFD=rec(400.0, 20.0, 200.0), L4PTFDU=rec(404.0, 20.0, 196.0)),
                                         216, eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4U_BY_B[b].clear()
            C.PRED_I4U_BY_B[b].update(saved[b])
    assert all(x[1] == "REPORT" for x in v.values()), v                # registration None: every key REPORT
    assert "4.0000" in v["I4U-JOINT-SAVING"][2] and "0.9800" in v["I4U-JOINT-RATIO"][2] and "1.0100" in v[
        "I4U-VS-D"][2], v
    assert set(v) == {"I4U-JOINT-SAVING", "I4U-JOINT-RATIO", "I4U-VS-D", "I4U-RATIO", "I4U-DRAFT-RATIO",
                      "I4U-PEAK-GB", "I4U-EQ", "I4U-VS-STRONGEST"}


def test_i4u_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4U is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4U_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4U)
    assert set(C.PRED_I4U_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4U_BY_B.values())
    src = (EFF / "r2_i4u_cell.py").read_text()
    assert ("\nPRED_STAMP_I4U = None\n" in src) == (C.PRED_STAMP_I4U is None)  # the step script's grep sees the state
    assert "%d x 31 + SAVED_LAYER0 = %s" % (HU.SAVED_PER_LAYER, format(HU.saved_per_pass(32), ",")) in src   # MODEL


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_i4u_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i4u():
    src = (EFF / "r2_i4u_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "benchmarks/Efficiency/r2_i4u_hoist.py" in src and "benchmarks/Efficiency/r2_i4a_lean.py" in src
    assert "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4am", "i4aa", "i4u", "i4uc"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4U = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4u_cell.py" % sha).stdout
    ran = 0
    for cell, pool, batch, steps, check, stress, smin, mem in (
            ("i4um", 10, 8, 72, 1, 1, 40, "100G"),
            ("i4ua", 65, 216, 512, 0, 0, 80, "240G"),
            ("i4ub", 10, 318, 512, 0, 0, 110, "340G")):
        r = _step(cell, sha)
        if cell in ("i4ua", "i4ub") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4U is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L4PTFD,L4PTFDU steps %d K 3 fork-store 0 check %d stress %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4u_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4u_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms L4PTFD\\?,L4PTFDU ", r.stdout), (cell, r.stdout[-600:])   # printf %q may quote ","
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", "L4PTFD,L4PTFDU", "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == ["L4PTFD", "L4PTFDU"]                          # every cell's arguments parse
        ran += 1
    assert ran >= 1
