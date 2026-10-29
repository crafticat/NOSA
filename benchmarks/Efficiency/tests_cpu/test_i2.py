"""CPU tests of rung I2 (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the arm L4PTU =
WS-A's tentative buffers on the schedule of contract 4 (spec_tentative.TentRunner) with WS-B's union residency (ONE
residency per layer for the union of the joint pass's rows): nosi/nosi/spec_tentative_union.py, the cell
benchmarks/Efficiency/r2_i2_cell.py, the step script r2_i2_step.sh. The world: tests_cpu/i2_world.py over wsa_world /
wsb_world / spec_world (the SHIPPED engine from its own source; every CUDA op a torch stand-in with the kernel's flat
addresses, lesson L28). B 1 and B 8, K 3. No CUDA, no Triton launch. CPU stand-ins do not establish GPU dispatch,
ordering, timing or memory (the cells i2m / i2a / i2b / i2ca / i2cb do). FAST is checked on itself against SH (ordinary
NOSA at the same configuration), not certified by an EXACT arm.

GATE 0   test_i2_gate0_files: the tree = 40ac348 except the declared WS-A, WS-B and I2 files; WS-A's files that I2 does
         not edit are the 83dac96 bytes, WS-B's files the 1a8ce68 bytes (verify_rows.py, spec_round.py among them);
         test_spec_prefetch.py = both sides' lines, all of them; the older Gate-0 tests only add lines naming
         spec_tentative_union; in test_wsa.py / test_wsb.py only the Gate-0 function changed (AST); spec_tentative.py =
         83dac96's AST outside the declared seams; the name scans. test_i2_gate0_tent_arms (L4PT B 1 / B 8, L3T, L4PJ,
         L4P legacy): every arm of WS-A's branch bit for bit against the 83dac96 SOURCE loaded from git (chain "full":
         spec_tentative + spec_round + verify_rows as one set; "st": spec_tentative alone, which isolates the seams).
         test_i2_gate0_wsb_arms (L4X, L4XU): WS-B's arms bit for bit against the 1a8ce68 source.
TOKENS   test_l4ptu_commits_sh_tokens (B 1, B 8 gemm rowgroup; B 8 gemm natural): L4PTU commits SH's tokens (MUST), exactly
         N per request; the logits within TOL; the selection of every committed verify / target row, layer and head = SH's
         (the reference fed SH's tokens; bitwise SH's logits, asserted); residency and host rows after every round; every
         selected block of a verify row present (dropped 0), the visible count (vis_bad 0). Preconditions: rejects and
         full accepts, an edge crossed, rows past an edge, F4 fill rows, draft and forced skips, pool moves, store entries.
COUNT    test_l4ptu_one_residency_per_layer: one residency per layer and joint pass (the module's counters and an
         INDEPENDENT count of the diff kernel). CONTROL: union False (L4PT's rows) gives (K + 1) L; the cell's RESID gate
         catches it.
D1 / D3  test_l4ptu_deferred_and_residency (B 1, B 8): no deferred-row write, K + 1 rows per joint pass, one replay per
         group from the draft row's buffer entry, the J slots' row writes = the trace's fed positions, no residency undo.
         PLANT recompute_deferred: CAUGHT. test_l4ptu_rollback_fixed_cost: reset / commit a fixed number of torch calls; a
         reset writes no engine table; the dropped entries recounted from the buffer writes. PLANT undo_residency: CAUGHT.
SCHEDULE test_l4ptu_schedule_trace: the contract-4 trace = the test model's (WS-A's spec_trace), exactly N, the rows that
         ran = the trace, SH's tokens across rejects, refills, final groups and target rows.
STORE    test_l4ptu_store_and_union_plants (X 4: the union does not fit one fill): spill > 0, dropped 0, vis_bad 0, SH's
         tokens. PLANTS (WS-B's) drop_spill, union_attention, short_store_len: CAUGHT by their counters.
STATE    test_l4ptu_committed_state (B 1, B 8): WS-A's state classes against SH fed the committed tokens; host rows of the
         complete committed blocks = SH's.
LIFETIME test_l4ptu_buffer_lifetime: dead buffer entries poisoned change no token. PLANTS no_causal / no_lower: CAUGHT.
TEACHER  test_l4ptu_teacher_forced_vs_sh (B 1, B 8): the CPU twin of the check cell: every row's selection SET = SH's at
         its position (every layer, head, request), argmax = SH's token, logits within TOL, dropped 0, vis_bad 0, one
         residency per layer and pass (independent count), VRF-IDX of the window, the first store and the buffer call.
         test_l4ptu_store_bytes_and_tentative_plant: the byte check of every store and window slot (wsb_world.union_bytes);
         PLANT list_tentative (the tentative block in the union list): CAUGHT by the visible count.
WIRING   test_l4ptu_refusals_and_no_host_sync.
CELL     test_i2_cell_arguments, test_i2_cell_report_and_gates, test_step_script_i2.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i2.py
"""
from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
SPEC5 = "40ac348870e89b43ac5278458b660dea8d360508"          # nosi-r2-spec5: the base of both workstreams
WSA = "83dac96becf599181913e207dae7174c7f12c147"            # r2-wsareg: WS-A (I1) with the lead's registration
WSB = "1a8ce68d9d6852fecf45b916c9ef1e6a5a80e098"            # r2-wsb: WS-B with the lead's registration
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i2"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
SAME_AS_WSA = ("benchmarks/Efficiency/r2_wsa_cell.py", "benchmarks/Efficiency/r2_wsa_step.sh", T_ + "wsa_world.py",
               "docs/NOTES_wsa.md")
SAME_AS_WSB = ("nosi/nosi/verify_rows.py", "nosi/nosi/spec_round.py", "benchmarks/Efficiency/r2_wsb_cell.py",
               "benchmarks/Efficiency/r2_wsb_step.sh", T_ + "wsb_world.py", "docs/NOTES_wsb.md")
OLDER_GATE0 = (T_ + "test_ragged_engine.py", T_ + "test_verify_rows.py", T_ + "test_rollback.py", T_ + "test_spec_round.py")
I2_EDITED = ("nosi/nosi/spec_tentative.py", T_ + "test_wsa.py", T_ + "test_wsb.py") + OLDER_GATE0
MERGED = (T_ + "test_spec_prefetch.py",)
I2_NEW = ("nosi/nosi/spec_tentative_union.py", "benchmarks/Efficiency/r2_i2_cell.py", "benchmarks/Efficiency/r2_i2_step.sh",
          T_ + "i2_world.py", T_ + "test_i2.py", "docs/NOTES_i2.md", "benchmarks/Efficiency/r2_i2_check.py",
          "benchmarks/Efficiency/r2_wsg_check.py")
WSG = "b1ae579"                                              # r2-wsgreg: WS-G with the lead's registration
WSG_COPY = "benchmarks/Efficiency/r2_wsg_check.py"          # copied from WSG unchanged
# I3 (nosi-r2-i3 = this tree 63da01d + WS-C2 db21536 + the arm L4PTF): WS-C's files and the I3 files are declared too
# (their own Gate 0s: tests_cpu/test_wsc.py, tests_cpu/test_i3.py); WS-C edited two of WS-B's files (verify_rows.py,
# spec_round.py): they are WS-C2's bytes here
WSC2 = "db21536c4e97f651cbac8cc9b95a376ff5dc76db"
WSC_EDITED = ("nosi/nosi/verify_rows.py", "nosi/nosi/spec_round.py")
I3_FILES = ("benchmarks/Efficiency/fork_vllm_unified_attention.py", "benchmarks/Efficiency/r2_kv_arena.py",
            "benchmarks/Efficiency/r2_wsc_cell.py", "benchmarks/Efficiency/r2_wsc_step.sh", T_ + "wsc_world.py",
            T_ + "test_wsc.py", T_ + "test_wsc_interp.py", "docs/NOTES_wsc.md",                       # WS-C
            "nosi/nosi/spec_tentative_fork.py", "benchmarks/Efficiency/r2_i3_arena.py",
            "benchmarks/Efficiency/r2_i3_cell.py", "benchmarks/Efficiency/r2_i3_step.sh", T_ + "i3_world.py",
            T_ + "test_i3.py", T_ + "test_i3_interp.py", "docs/NOTES_i3.md")                          # I3
# I4A (nosi-r2-i4a: I3 c4c64b2 + the arm L4PTFD): its files (its own Gate 0: tests_cpu/test_i4a.py)
I3_FILES = I3_FILES + ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
                       "benchmarks/Efficiency/r2_i4a_step.sh", T_ + "i4a_world.py", T_ + "test_i4a.py",
                       "docs/NOTES_i4a.md")                                                     # I4A
# I4U (nosi-r2-i4u: I4A acbe433 + the arm L4PTFDU): its files (its own Gate 0: tests_cpu/test_i4u.py)
I3_FILES = I3_FILES + ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
                       "benchmarks/Efficiency/r2_i4u_step.sh", T_ + "i4u_world.py", T_ + "test_i4u.py")   # I4U
SEAM_CHANGED = {"#0", "TentState._layer", "TentState._row", "TentRunner.__init__"}
SEAM_ADDED = {"TentState._rows", "TentState._resid", "TentState._write", "TentRunner._joint_ops", "TentRunner._make_state"}


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


I = _load("i2_world")
WT = _load("wsa_tests_lib", HERE / "test_wsa.py")           # WS-A's test helpers (spec_trace, the row checks, the cost run)
WW, WB, SW, W, RW, VW, RE, SR, BC, ST, STU, VR = (I.WW, I.WB, I.SW, I.W, I.RW, I.VW, I.RE, I.SR, I.BC, I.ST, I.STU,
                                                  I.VR)
K, N, L, BS, TOL = I.K0, I.N0, I.L, I.BS, I.TOL
FILTER = WT.FILTER
S1 = [4658]


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i2_cell
    return r2_i2_cell


# ------------------------------------------------------------------ GATE 0: the files
def _units(src: str) -> dict:
    """{name: ast.dump}: top-level functions, class methods (Class.method), each class's bases and other statements
    (Class.<bases>, Class.#i), the other top-level statements (#i in order; #0 the docstring)."""
    out, k = {}, 0
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.ClassDef):
            out[n.name + ".<bases>"] = ast.dump(ast.Tuple(elts=list(n.bases), ctx=ast.Load()))
            j = 0
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    out["%s.%s" % (n.name, m.name)] = ast.dump(m)
                else:
                    out["%s.#%d" % (n.name, j)] = ast.dump(m)
                    j += 1
        else:
            out["#%d" % k] = ast.dump(n)
            k += 1
    return out


def _changed_lines(*rev, f):
    d = _git("diff", "-U0", *rev, "--", f).stdout.splitlines()
    return [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]


def test_i2_gate0_files():
    if not (_have(SPEC5) and _have(WSA) and _have(WSB)):
        pytest.skip("no git or no commit %s / %s / %s in %s" % (SPEC5, WSA, WSB, ROOT))
    wsa_files = [x for x in _git("diff", "--name-only", SPEC5, WSA).stdout.split("\n") if x]
    wsb_files = [x for x in _git("diff", "--name-only", SPEC5, WSB).stdout.split("\n") if x]
    r = _git("diff", "--name-only", SPEC5, "--", ".", ":(exclude,glob)**/__pycache__/**")
    changed = {x for x in r.stdout.split("\n") if x}
    declared = set(wsa_files) | set(wsb_files) | set(I2_EDITED) | set(MERGED) | set(I2_NEW) | set(I3_FILES)   # I3
    assert changed <= declared, sorted(changed - declared)                 # (a) nothing else moved
    assert set(SAME_AS_WSA) <= set(wsa_files) and set(SAME_AS_WSB) <= set(wsb_files)
    assert _git("diff", "--name-only", WSA, "--", *SAME_AS_WSA).stdout.strip() == ""    # (b) the branches' bytes
    assert _git("diff", "--name-only", WSB, "--", *[f for f in SAME_AS_WSB if f not in WSC_EDITED]).stdout.strip() == ""
    if _have(WSC2):                                                        # I3: WS-C's two edited WS-B files
        assert _git("diff", "--name-only", WSC2, "--", *WSC_EDITED).stdout.strip() == ""
    assert set(wsa_files) | set(wsb_files) <= set(SAME_AS_WSA) | set(SAME_AS_WSB) | set(I2_EDITED) | set(MERGED) | set(
        I2_NEW), "a branch file is neither kept nor declared as edited"
    f = MERGED[0]                                                          # (c) test_spec_prefetch.py: both sides
    a, b, m = set(_changed_lines(SPEC5, WSA, f=f)), set(_changed_lines(SPEC5, WSB, f=f)), set(_changed_lines(SPEC5, f=f))
    assert a and b and m == a | b, (sorted(m - (a | b))[:5], sorted((a | b) - m)[:5])
    for f in OLDER_GATE0:                                                  # (d) one added line each
        body = _changed_lines(WSA, f=f)
        assert body and all(x.startswith("+") and "spec_tentative_union" in x for x in body), (f, body[:5])
    for f, sha, gate in ((T_ + "test_wsa.py", WSA, "test_wsa_flag_off_is_spec5"),   # (e) only the Gate-0 function
                         (T_ + "test_wsb.py", WSB, "test_gate0_files")):
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (sha, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= {gate}, f
    old = _units(_git("show", "%s:nosi/nosi/spec_tentative.py" % WSA).stdout)     # (f) the seams only
    new = _units((ROOT / "nosi/nosi/spec_tentative.py").read_text())
    assert set(old) <= set(new) and set(new) - set(old) == SEAM_ADDED, sorted(set(new) ^ set(old))
    diff = {k for k in old if old[k] != new[k]}
    assert diff <= SEAM_CHANGED and {"TentState._row", "TentRunner.__init__"} <= diff, sorted(diff)
    if _have(WSG):                                                     # (h) WS-G's check module: its bytes
        assert _git("diff", "--name-only", WSG, "--", WSG_COPY).stdout.strip() == "", WSG_COPY
    r = _git("ls-files", "--others", "--exclude-standard", "--", "nosi", ":(exclude,glob)**/__pycache__/**")
    assert r.stdout.strip() == "", "untracked files in nosi/: %s" % r.stdout
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}           # (g) the name scans
    for name in ("spec_round", "ragged_rollback", "state_snapshot", "spec_prefetch"):
        assert name not in src["spec_tentative_union.py"], name
    assert sorted(n for n, t in src.items() if "spec_tentative_union" in t and n != "spec_tentative_union.py") == [
        "spec_tentative.py", "spec_tentative_fork.py"]                 # the seams' docstring; I3's module imports it
    for c in ("r2_spec_cell.py", "r2_spec5_cell.py", "r2_base_cell.py", "r2_verify_cell.py", "r2_wsa_cell.py",
              "r2_wsb_cell.py"):
        assert "spec_tentative_union" not in (EFF / c).read_text(), c


# ------------------------------------------------------------------ GATE 0: the arms of both branches, bit for bit
def _arm_run(arm, B, mods):
    """One run of a branch arm with this tree's modules (mods None) or the older ones (i2_world.old_chain)."""
    if arm in ("L4PT", "L3T"):
        kw = dict(level=("L4P" if arm == "L4PT" else "L3"), draft_filter=FILTER, with_reference=False)
        if mods is None:
            return WW.run_tent(B, **kw)
        with I.swapped(mods), I.world_modules(mods["spec_tentative"], mods.get("spec_round", SR)):
            return WW.run_tent(B, **kw)
    ST_, SR_ = (mods or {}).get("spec_tentative", ST), (mods or {}).get("spec_round", SR)
    if arm == "L4PJ":                                                  # the EXACT journal runner of contract 4
        sr = type(sys)("journal_contract4")
        sr.SpecOps, sr.SpecRunner = SR_.SpecOps, ST_.PaperJournalRunner
    else:                                                              # L4P legacy: the round runner's own L4P
        sr = SR_ if (mods and "spec_round" in mods) else None
    with (I.swapped(mods) if mods else contextlib.nullcontext()):
        return SW.run_vs_shipped(B, level="L4P", mode="exact", gemm="rowgroup", draft_filter=FILTER, sr=sr)


CASES0 = [("L4PT", 1, "full"), ("L4PT", 8, "full"), ("L4PT", 8, "st"), ("L3T", 8, "full"), ("L4PJ", 8, "full"),
          ("L4PJ", 8, "st"), ("L4P", 8, "full")]


@pytest.mark.parametrize("arm,B,chain", CASES0, ids=["%s-B%d-%s" % c for c in CASES0])
def test_i2_gate0_tent_arms(arm, B, chain, tmp_path):
    if not _have(WSA):
        pytest.skip("no git or no commit %s in %s" % (WSA, ROOT))
    names = ["verify_rows", "spec_round", "spec_tentative"] if chain == "full" else ["spec_tentative"]
    mods = I.old_chain(WSA, names, tmp_path, tag=chain)
    assert mods["spec_tentative"].SR is mods.get("spec_round", SR)       # precondition: the chain is coherent
    assert mods["spec_tentative"].VR is mods.get("verify_rows", VR)
    new, old = _arm_run(arm, B, None), _arm_run(arm, B, mods)
    # an older verify_rows has no host invocation counts in read_stats (WS-B added "calls" / "passes")
    bad = I.same_run(new, old, verify_keys=(("calls", "passes") if chain == "full" else ()))
    assert bad == [], (arm, B, chain, bad)
    f = new["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f                      # both accept paths ran (L34)


def _wsb_run(arm, mods):
    sr = mods["spec_round"] if mods else None
    with (I.swapped(mods) if mods else contextlib.nullcontext()):
        if arm == "L4X":
            return SW.run_vs_shipped(8, level="L4", mode="exact", gemm="rowgroup", draft_filter=FILTER, sr=sr)
        return WB.run_spec_union(8, level="L4", extra=16, draft_filter=FILTER, sr=sr)


@pytest.mark.parametrize("arm", ["L4X", "L4XU"])
def test_i2_gate0_wsb_arms(arm, tmp_path):
    if not _have(WSB):
        pytest.skip("no git or no commit %s in %s" % (WSB, ROOT))
    mods = I.old_chain(WSB, ["verify_rows", "spec_round"], tmp_path, tag="wsb")
    new, old = _wsb_run(arm, None), _wsb_run(arm, mods)
    assert I.same_run(new, old) == [], arm
    f = new["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f


# ------------------------------------------------------------------ L4PTU on the stand-ins
@functools.lru_cache(maxsize=None)
def _evu(B, gemm="rowgroup", X=16, filt="periodic", tneg=None, poison=False, ref=True, union=True, vneg=None,
         max_rounds=0, cls="L4PTU"):
    """One run_tent of L4PTU (or cls "L4PT") per parameter set; ev["diff"]: the independent diff-kernel count."""
    build = I.tentu_builder(X=X, gemm=gemm, union=union, vneg=vneg, cls=cls)
    box = {}

    def on_runner(runner):
        if runner.vs is not None:
            box["diff"] = I.count_joint_diff(runner)
    ev = I.run_tentu(B, build=build, draft_filter=WT._filter(filt), tneg=tneg, poison=poison, with_reference=ref,
                     max_rounds=max_rounds, on_runner=on_runner)
    ev["diff"] = box.get("diff")
    return ev


CASES_T = dict(B1=(1, "rowgroup"), B8=(8, "rowgroup"), B8nat=(8, "natural"))


@pytest.mark.parametrize("case", list(CASES_T))
def test_l4ptu_commits_sh_tokens(case):
    B, gemm = CASES_T[case]
    ev = _evu(B, gemm=gemm)
    f, rd = ev["facts"], ev["rd"]
    u = rd["union"]
    assert f["rejects"] > 0 and f["full"] > 0, f                      # preconditions (L34)
    assert f["crossed"] > 0 and f["edge_rows"] > 0 and f["fill_rows"] > 0, f
    assert f["skipped"] > 0 and f["forced"] > 0 and f["move_out"] + f["swap"] > 0, f
    assert u["passes"] > 0 and u["union_samples"] > 0, u
    if B > 1:
        assert f["ragged"] and u["extras"] > 0 and u["fills_used"] > 0, (f, u)   # rows read store blocks
    assert rd["count"] == [N] * B, rd["count"]                        # exactly N (contract 4)
    assert ev["ref_logits_bad"] == [], ev["ref_logits_bad"][:5]        # the reference IS SH
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]                # THE CLAIM: SH's tokens
    print("I2 L4PTU %s: max |logit - SH's| over %d committed rows = %.3e (TOL %.0e); selections compared %d; union "
          "extras %d, pool %d, host %d, spill %d" % (case, ev["rows_compared"], ev["max_logit_delta"], TOL,
                                                    ev["sel_compared"], u["extras"], u["extras_pool"],
                                                    u["extras_host"], u["spill"]))
    assert ev["max_logit_delta"] <= TOL, ev["max_logit_delta"]
    assert ev["sel_compared"] > 0 and ev["sel_mismatch"] == [], ev["sel_mismatch"][:5]
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert u["dropped"] == 0 and u["unserved"] == 0 and u["vis_bad"] == 0, u      # present; union never attended
    assert u["row_visible"] == [s - k for s, k in zip(u["row_selected"], u["row_skipped"])], u
    assert u["row_dropped"] == [0] * (K + 1), u


def test_l4ptu_one_residency_per_layer():
    ev = _evu(8)
    c, u = ev["rd"]["tent"]["calls"], ev["rd"]["union"]
    jp = c["joint_passes"]
    assert jp > 0 and ev["diff"]["per_pass"] == [L] * jp, (jp, ev["diff"]["per_pass"][:5])   # INDEPENDENT count
    assert c["joint_resid"] == L * jp and c["union_layers"] == L * jp, c
    assert u["calls"]["resid"] == L * jp and u["passes"] == jp, u["calls"]
    ser = _evu(8, union=False, ref=False)                              # NEGATIVE CONTROL: the rows one by one
    cs = ser["rd"]["tent"]["calls"]
    assert ser["diff"]["per_pass"] == [(K + 1) * L] * cs["joint_passes"] and cs["union_layers"] == 0, cs
    assert ser["bad_tokens"] == []                                     # (L4PT's code: SH's tokens)
    g = {n: ok for n, _, ok, _ in _cell().union_gates("L4PTU", dict(read=ser["rd"]), L, check=False)}
    assert g["L4PTU-RESID"] is False                                   # the cell's gate catches it
    g = {n: ok for n, _, ok, _ in _cell().union_gates("L4PTU", dict(read=ev["rd"]), L, check=False)}
    assert g["L4PTU-RESID"] is True


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4ptu_deferred_and_residency(B):
    ev = _evu(B)
    tr, log = ev["rd"]["tent"], ev["runner"].T.rec
    c = tr["calls"]
    assert tr["deferred_writes"] == 0, tr["deferred_writes"]             # D3 (the cell's MUST gate)
    assert c["joint_rows"] == (K + 1) * c["joint_passes"] and c["joint_passes"] > 0, c
    assert c["joint_passes"] + c["j1_passes"] == ev["rd"]["rounds"], c
    assert tr["replays"] == WT._groups(ev) > 0, (tr["replays"], WT._groups(ev))
    assert ev["rd"]["tokens_acct"]["remade_rows"] == 0
    assert WT._joint_row_errors(log, ev["rd"]["trace_all"], B) == []
    errs, seen = WT._replay_errors(log, B)
    assert errs == [] and seen == WT._groups(ev), (errs[:5], seen)
    assert WT._pos_errors(log) == []
    assert c["residency_undo"] == 0                                    # D1
    p = _evu(B, tneg="recompute_deferred", ref=False, max_rounds=12)   # PLANT: EXACT's K + 2 rows
    pt = p["rd"]["tent"]
    assert pt["deferred_writes"] > 0 and pt["calls"]["joint_rows"] == (K + 2) * pt["calls"]["joint_passes"], pt
    assert p["bad_tokens"] == []                                       # a COST plant: the tokens stay SH's


def test_l4ptu_rollback_fixed_cost():
    with I._builder(I.tentu_builder()):
        runs = [WT._cost_run(WT.FILTER), WT._cost_run(WT.FILTER0)]
        ev, _, facts = WT._cost_run(WT.FILTER, tneg="undo_residency")  # PLANT: a residency undo
    resets = {n for _, box, _ in runs for n in box["reset"]}
    commits = {n for _, box, _ in runs for n in box["commit"]}
    drops = [e["rd"]["tent"]["dropped"] for e, _, _ in runs]
    print("I2 COST: torch calls per reset %s, per commit %s; dropped entries %s" % (sorted(resets), sorted(commits),
                                                                                   drops))
    assert drops[0] != drops[1], drops                                  # precondition
    assert len(resets) == 1 and len(commits) == 1, (resets, commits)
    for e, _, fx in runs:
        assert fx["changed"] > 0 and fx["reset_wrote"] == [], fx["reset_wrote"][:3]
        assert e["rd"]["tent"]["dropped"] == WT._dropped_recount(e)
        assert e["rd"]["tent"]["calls"]["residency_undo"] == 0 and e["bad_tokens"] == []
        assert e["rd"]["tent"]["calls"]["union_layers"] > 0            # the union pass ran in these runs
    assert facts["runner"].T.calls["residency_undo"] > 0 and facts["reset_wrote"] != []   # CAUGHT


def test_l4ptu_schedule_trace():
    B = 8
    ev = _evu(B, filt="end", ref=False)
    rd = ev["rd"]
    acct = rd["tokens_acct"]
    assert acct["corrected"] > 0 and acct["fills"] > 0 and acct["finals"] == B and acct["targets"] > 0, acct
    assert rd["count"] == [N] * B, rd["count"]
    errs, remade = WT._trace_errors(ev, B, False)
    assert errs == [] and remade == [], (errs[:3], remade[:3])
    log, tr_all = ev["runner"].T.rec, rd["trace_all"]
    assert WT._joint_row_errors(log, tr_all, B) == []
    assert WT._slot_row_errors(log, tr_all, B) == []
    assert ev["bad_tokens"] == [] and ev["max_logit_delta"] <= TOL, (ev["bad_tokens"][:3], ev["max_logit_delta"])
    assert ev["resid"] == [] and ev["host"] == []
    for b in range(B):
        t = tr_all[b]
        k = max(i for i, x in enumerate(t) if x["role"] == "final")
        assert {x["role"] for x in t[k + 1:]} <= {"target", "idle"}
        assert sum(x["committed"] for x in t) == N - rd["warm"]


def test_l4ptu_store_and_union_plants():
    clean = _evu(8, X=4, ref=False, max_rounds=12)
    u = clean["rd"]["union"]
    assert u["spill"] > 0 and u["spill_fills_used"] > 0 and u["extras"] > 0, u      # precondition: spill
    assert u["dropped"] == 0 and u["unserved"] == 0 and u["vis_bad"] == 0, u
    assert clean["bad_tokens"] == [] and clean["max_logit_delta"] <= TOL
    d = _evu(8, X=4, ref=False, max_rounds=12, vneg="drop_spill")["rd"]["union"]
    assert d["unserved"] > 0 and d["dropped"] > 0, d                                 # CAUGHT (presence)
    a = _evu(8, X=4, ref=False, max_rounds=12, vneg="union_attention")
    assert a["rd"]["union"]["vis_bad"] > 0, a["rd"]["union"]                         # CAUGHT (visible count)
    assert a["bad_tokens"] != [] or a["max_logit_delta"] > TOL                       # and the logits leave SH's
    s = _evu(8, X=4, ref=False, max_rounds=12, vneg="short_store_len")["rd"]["union"]
    assert s["vis_bad"] > 0 and s["dropped"] == 0, s                                 # CAUGHT (a read, not a list)
    g = {n: ok for n, _, ok, _ in _cell().union_gates("L4PTU", dict(read=clean["rd"], fa_check={}), L, check=True,
                                                        require_stress=True)}
    assert g["L4PTU-PRESENT"] and g["L4PTU-VIS"] and g["L4PTU-USTRESS"] and not g["L4PTU-IDX"], g


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4ptu_committed_state(B):
    ev = _evu(B)
    run = ev["runner"]
    assert ev["facts"]["rejects"] > 0 and ev["facts"]["crossed"] > 0
    run.T.reset()                                 # the twins to the committed counters (the chain is not committed)
    qc = run.T.qc.clone()
    assert torch.equal(qc.cpu(), ev["S"] + ev["cnt"])
    sc = WW.state_classes(run.rs, ev["ref_rs"], qc, ev["S"])
    print("I2 STATE B %d: EXACT differ %s; layer-0 not bitwise %s; beyond TOL %s; max |delta| %.3e; CACHE %s" % (
        B, sc["exact"], sc["l0"], sc["close"], sc["maxabs"], sc["cache"]))
    assert sc["exact"] == [] and sc["l0"] == [] and sc["close"] == [], (sc["exact"], sc["l0"], sc["close"][:5])
    assert WT._host_vs_shipped(ev) == []                              # the complete committed blocks: SH's host rows


def test_l4ptu_buffer_lifetime():
    ev = _evu(8, poison=True, ref=False)
    assert any(bool((t == ST.POISON_V).any()) for t in ev["runner"].T.v)    # precondition: the poison acted
    assert ev["bad_tokens"] == [] and ev["max_logit_delta"] <= TOL, (ev["bad_tokens"][:3], ev["max_logit_delta"])
    for tneg in ("no_causal", "no_lower"):
        p = _evu(8, poison=True, tneg=tneg, ref=False)
        assert p["bad_tokens"] != [] or p["max_logit_delta"] > TOL, tneg   # CAUGHT


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4ptu_teacher_forced_vs_sh(B):
    G = 6
    ev = I.run_teacher_cpu(B, G=G)
    st = ev["stats"]
    assert ev["ref_logits_bad"] == [], ev["ref_logits_bad"][:5]
    assert ev["fills"] and ev["past_rows"] > 0, (ev["fills"], ev["past_rows"])     # preconditions: an edge in the passes
    if B > 1:
        assert st["extras"] > 0, st
    assert ev["compared"] == G * L * (K + 1) * B, ev["compared"]
    print("I2 TEACHER B %d: max |logit - SH's| %.3e over %d rows; selection sets changed %d of %d; extras %d" % (
        B, ev["max_delta"], len(ev["deltas"]), len(ev["changed"]), ev["compared"] * W.HKV, st["extras"]))
    assert ev["changed"] == [], ev["changed"][:8]                     # THE CLAIM: SH's selections, every row
    assert ev["argmax_bad"] == [] and ev["max_delta"] <= TOL, (ev["argmax_bad"][:5], ev["max_delta"])
    assert st["dropped"] == 0 and st["unserved"] == 0 and st["vis_bad"] == 0, st
    assert st["row_visible"] == st["row_selected"] and st["rows_full"] == G * L * (K + 1) * B, st
    assert ev["diffs"] == [L] * G and st["calls"]["resid"] == L * G and st["passes"] == G, (ev["diffs"], st["calls"])
    fa = ev["fa"]
    assert fa["A"]["entries"] and all(fa["A"]["equal"]) and fa["T"]["entries"] and all(fa["T"]["equal"]), fa
    if st["extras"] > 0:
        assert fa["B"]["entries"] and all(fa["B"]["equal"]), fa
    assert ev["errors"] == (0, [0] * L), ev["errors"]
    res = dict(stats=st, errors=ev["errors"], fa_check=fa, sel=dict(compared=ev["compared"], changed=len(ev["changed"]),
                                                                    by_layer_nonzero={}, symdiff_hist={}, first=[]))
    g = {n: ok for n, _, ok, _ in _cell().check_gates_u("TUC", res, L=L)}
    if B > 1:
        assert all(g.values()), g                                     # the check cell's gates on the CPU evidence
    bad = dict(res, sel=dict(res["sel"], changed=1))
    assert {n: ok for n, _, ok, _ in _cell().check_gates_u("TUC", bad, L=L)}["TUC-SEL"] is False


def test_l4ptu_store_bytes_and_tentative_plant():
    ev = I.run_teacher_cpu(8, G=6, X=4, vrecord=True)
    st = ev["stats"]
    assert st["spill"] > 0, st                                         # precondition: the union did not fit one fill
    ub = WB.union_bytes(ev["runner"].vs.log, 4)
    assert ub["checked"] > 0 and ub["fills"] > 0, ub
    assert ub["missing_full"] == [] and ub["contents_bad"] == [], (ub["missing_full"][:5], ub["contents_bad"][:5])
    assert st["dropped"] == 0 and st["vis_bad"] == 0 and ev["changed"] == [] and ev["argmax_bad"] == []
    clean = I.run_teacher_cpu(1, G=5)                                  # the 5th pass is the first past the edge
    plant = I.run_teacher_cpu(1, G=5, uneg="list_tentative")           # PLANT: the tentative block in the union list
    assert clean["past_rows"] > 0 and clean["stats"]["vis_bad"] == 0, clean["stats"]
    assert plant["stats"]["vis_bad"] > 0, plant["stats"]               # CAUGHT


def _small(B=2, N_=8, **kw):
    S = I.S0
    hz = SW.horizon_for(N_, K)
    model, _, c = SW.two_caches(B, S, hz + 64, 0)
    tok0 = torch.randint(0, I.V, (B,), generator=torch.Generator().manual_seed(2))
    return I.tentu_builder(**kw)(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)


def test_l4ptu_refusals_and_no_host_sync():
    model, rs = RW.build(S1)
    ops = lambda **k: SR.SpecOps(**dict(dict(level="L4P", K=K, attn=WB.fake_attention_u), **k))   # noqa: E731
    for bad, kw in ((dict(level="L3"), {}), ({}, dict(uneg="nope")), ({}, dict(vneg="no_extras")),
                    ({}, dict(spill=0)), (dict(extra=1), {}), ({}, dict(tneg="nope"))):
        with pytest.raises(SR.SpecError):
            STU.TentRunnerU(model, rs, ops(**bad), BC.HostClock(), **kw)
    assert rs.ops.kernels["gather_k"] is W.KERNELS["gather_k"]         # every refusal before the meter attached
    for path, allowed in ((Path(STU.__file__), set()), (Path(ST.__file__), {"_run_paper", "_ps_after"}),
                          (Path(VR.__file__), set())):
        run = _small(check=False, stats=False)
        with RW.SyncTrap(path) as trap:
            run.run(8, rounds=3)
        fns = {h[0] for h in trap.hits}
        assert fns <= allowed, (path.name, trap.hits[:10])             # only the per-round sync
        if allowed:
            assert fns, "the trap is not blind"
        assert run.T.calls["union_layers"] > 0                         # the union pass ran


# ------------------------------------------------------------------ WS-G's check (cell i2g) on the stand-ins
@functools.lru_cache(maxsize=None)
def _wsg(kind, X=16, vneg=None, check=True):
    return I.run_wsg_cpu(kind, X=X, vneg=vneg, check=check, draft_filter=WT._filter("end"))


def test_i2_wsg_positive_control_and_arms():
    C = _cell()
    ex = _wsg("WJ")
    assert ex["restored"] and ex["ref_restored"]                       # every wrapper is back after the run
    acct = ex["rd"]["tokens_acct"]
    assert acct["corrected"] > 0 and acct["targets"] > 0 and acct["finals"] > 0, acct   # preconditions: D-slot commits
    g, rec = C.wsg_gates("WJ", ex["cap"], ex["ref"])
    print("I2 WS-G WJ: %s" % rec["details"])
    assert rec["parts"] == dict(MAP=True, REF=True, SEL=True, GREEDY=True, POS=True, SLOT=True, BITS=True), rec["details"]
    gd = {n: (m, ok) for n, m, ok, _ in g}
    assert gd["WJ-CHK"] == (True, True)                                # the positive control: MUST and PASS
    assert ex["cap"]["slot"]["checked"] > 0 and ex["ref"]["sel_checked"] > 0 and ex["ref"]["self"]["checked"] > 0
    for kind in ("WT", "WU"):
        ev = _wsg(kind)
        assert ev["restored"] and ev["ref_restored"], kind
        assert ev["rd"]["tokens_acct"]["targets"] > 0, kind
        g, rec = C.wsg_gates(kind, ev["cap"], ev["ref"])
        print("I2 WS-G %s: %s; logit %s" % (kind, rec["details"], rec["logit"]["detail"]))
        assert all(not m for _, m, _, _ in g), kind                    # every gate of a FAST arm is a REPORT
        assert ev["cap"]["miss_checked"] > 0 and ev["cap"]["sel_columns"] > 0, ev["cap"]
        assert all(rec["parts"].values()), (kind, rec["details"])      # on the stand-ins FAST passes every part
        assert "MISS" in rec["parts"] and "BITS" not in rec["parts"]
    a, b = _wsg("WU", check=False), _wsg("WU")                         # the capture changes nothing (read-only)
    for x, y in ((a["runner"].out, b["runner"].out), (a["runner"].fmax, b["runner"].fmax)):
        assert RW.same(x, y)
    assert a["rd"]["count"] == b["rd"]["count"] and a["rd"]["tent"]["calls"] == b["rd"]["tent"]["calls"]


def test_i2_wsg_planted_control():
    C = _cell()
    p = _wsg("WU", X=4, vneg="drop_spill")
    assert p["rd"]["union"]["unserved"] > 0, p["rd"]["union"]          # precondition: the plant dropped blocks
    g, rec = C.wsg_gates("WUP", p["cap"], p["ref"])
    gd = {n: (m, ok) for n, m, ok, _ in g}
    print("I2 WS-G WUP (planted): failed parts %s; miss %d" % (rec["failed"], p["cap"]["miss"]))
    assert p["cap"]["miss"] > 0 and not rec["parts"]["MISS"] and not rec["ok"], rec["details"]
    assert gd["WUP-CHK"] == (False, False)                             # a REPORT
    assert gd["WUP-PLANT-CHK-PRE"] == (True, True) and gd["WUP-PLANT-CHK-CAUGHT"] == (True, True)   # MUST, CAUGHT


# ------------------------------------------------------------------ CELL
def test_i2_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base + ["--joint-residency", "union"])
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.joint_extra, a.joint_spill, a.tu_gemm, a.check_passes,
            a.check_every, a.require_stress, a.spec_twin, a.spec_body, a.probe_every) == (
        ["L4PT", "L4PTU"], 216, 512, 65, 3, 48, 0, "natural", 32, 0, 0, "fused", "fused", 16)
    assert C.parse_args(base + ["--arms", "L4PT"]).joint_residency == "serial"         # the flag is off by default
    for bad in ([], ["--joint-residency", "union", "--arms", "L4PT"], ["--joint-residency", "union", "--arms", "TUC,SHR"],
                ["--joint-residency", "union", "--arms", "TUC"], ["--joint-residency", "union", "--spec", "2"],
                ["--joint-residency", "union", "--arms", "SHR,TUC", "--check-passes", "200"],
                ["--joint-residency", "union", "--joint-extra", "1"], ["--joint-residency", "union", "--pool", "0"],
                ["--joint-residency", "union", "--arms", "L4PTU,L4PTU"], ["--arms", "L4X"],
                ["--joint-residency", "union", "--arms", "WU,WUP"], ["--arms", "WJ,WU"],
                ["--joint-residency", "union", "--arms", "WJ,WUP", "--chk-rows", "999"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--joint-residency", "union", "--arms", "L4PT,L4PTU,SHR,TUC", "--steps", "64",
                             "--check-passes", "15"])
    assert a.arms == ["L4PT", "L4PTU", "SHR", "TUC"] and a.check_passes == 15
    a = C.parse_args(base + ["--joint-residency", "union", "--arms", "WJ,WT,WU,WUP", "--batch", "100", "--steps", "64",
                             "--chk-slot-every", "4"])
    assert (a.arms, a.chk_slot_every, a.chk_plant_extra, a.sample) == (["WJ", "WT", "WU", "WUP"], 4, 4, None)
    assert C.parse_args(base + ["--arms", "WJ,WT"]).joint_residency == "serial"   # no union arm: no flag needed
    assert C.main(["--out", "x", "--workload", "w", "--arms", "L4PT", "--model", ""]) == 20     # no model: refused


def test_i2_cell_report_and_gates(monkeypatch):
    C = _cell()
    B, steps = 8, N
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    dims = dict(L=L, H=I.HKV, D=W.D, topk=I.TOPK, block_bytes=64 * W.D * 4 * 2 + 64 * 4)
    recs = {}
    for arm, ev in (("L4PTU", _evu(8)), ("L4PT", _evu(8, cls="L4PT", ref=False))):
        res = WT._res_of(ev, False)
        res.update(joint_phases=C.union_phase_summary([], []), fa_check={})
        ref = _evu(8)                                                  # SH's tokens and logits (the same prefill)
        inputs = ref["ref_tok"][:, :steps + 1]
        fp_sh = (ref["ref_lg"][:, :steps].amax(-1), ref["ref_lg"][:, :steps].sum(-1, dtype=torch.float64))
        if arm == "L4PTU":
            rec = C.summarize_u(arm, res, inputs, fp_sh, sh, B, steps, dims)
            g = C.arm_gates_u(arm, res, rec, steps, L, check=False)
            names = {x[0] for x in g}
            assert {"L4PTU-ERR", "L4PTU-COMMIT-N", "L4PTU-D1", "L4PTU-D3", "L4PTU-SPAN", "L4PTU-RESID"} <= names, names
            assert not any(n.startswith("L4PT-") for n in names), names
            assert rec["union"]["resid_per_joint_pass"] == L and rec["arm"] == "L4PTU", rec["union"]
        else:
            rec = C.WA.summarize_w(arm, res, inputs, fp_sh, sh, B, steps, dims)
            g = C.WA.arm_gates_w(arm, res, rec, steps)
        assert all(x[2] for x in g), [x for x in g if not x[2]]
        assert rec["tokens"]["n_unequal"] == 0 and rec["tokens"]["n_checked"] == B * steps, rec["tokens"]
        recs[arm] = rec
    fa = dict(A=dict(entries=[0], equal=[True]), B=dict(entries=[1], equal=[True]), T=dict(entries=[0], equal=[True]))
    res = dict(read=_evu(8)["rd"], fa_check=fa)
    g = {n: ok for n, _, ok, _ in C.union_gates("L4PTU", res, L, check=True)}
    assert g["L4PTU-IDX"] and g["L4PTU-PRESENT"] and g["L4PTU-VIS"], g
    res["fa_check"] = dict(fa, T=dict(entries=[0], equal=[False]))
    assert not {n: ok for n, _, ok, _ in C.union_gates("L4PTU", res, L, check=True)}["L4PTU-IDX"]   # not blind
    p = C.verdicts(sh, recs, B)                                        # B 8: the flat PRED_I2, all None: REPORT
    assert all(v in ("REPORT", "NOT_RUN") for _, v, _ in p), p
    assert C.PRED_STAMP_I2 is None or isinstance(C.PRED_STAMP_I2, str)   # None until the lead registers
    assert all(C.PRED_I2[k] is None for k in C.PRED_KEYS) and set(C.PRED_I2_BY_B) == {216, 318}   # flat block: None
    assert all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I2_BY_B.values())
    r2 = recs["L4PTU"]["rates"]["median_tok_s"] / C.SC.sh_rates(B, sh)["median_tok_s"]
    monkeypatch.setitem(C.PRED_I2_BY_B, B, dict(I2_ratio=(0.0, 1e9), I2_plan=r2 * 1.5, I2_above_I1=True,
                                                 I2_agree=0.5))
    v = {n: x for n, x, _ in C.verdicts(sh, recs, B)}                  # the per-batch entry overrides the flat None
    assert v["I2-RATIO"] == "CONFIRMED" and v["I2-PLAN-15"] == "REFUTED" and v["I2-AGREE"] == "CONFIRMED", v
    assert v["I2-ABOVE-I1"] in ("CONFIRMED", "REFUTED") and v["I2-VS-I1"] == "REPORT", v
    assert C.verdicts(sh, {}, B) == [("L4PTU-RUN", "NOT_RUN", "the arm did not run")]
    ph = dict(resid=1.0, write=0.5, union=0.25, fetch=2.0, attn=3.0, spill=0.5, buffer=1.0, merge=0.25, post=4.0)
    gr = C.union_groups(ph)
    assert gr["attention"] == 4.25 and gr["residency"] == 1.0 and gr["dense_next"] == 4.0 and gr["total"] == 12.5


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_i2_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i2():
    for bad in ("wa1m", "wb1m", "i2c", "i2"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I2 = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i2_cell.py" % sha).stdout
    for cell, pool, batch, steps, arms, X, passes, check, stress, smin, mem, slot in (
            ("i2m", 10, 8, 64, "L4PT,L4PTU,SHR,TUC", 4, 15, 1, 1, 45, "100G", 32),
            ("i2a", 65, 216, 512, "L4PT,L4PTU", 48, 32, 0, 0, 70, "240G", 32),
            ("i2b", 10, 318, 512, "L4PT,L4PTU", 48, 32, 0, 0, 95, "340G", 32),
            ("i2ca", 65, 216, 129, "SHR,TUC", 48, 32, 0, 0, 70, "240G", 32),
            ("i2cb", 10, 318, 129, "SHR,TUC", 48, 32, 0, 0, 95, "340G", 32),
            ("i2g", 10, 100, 64, "WJ,WT,WU,WUP", 48, 32, 0, 0, 90, "200G", 4)):
        r = _step(cell, sha)
        if cell in ("i2a", "i2b") and not registered:                 # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I2 is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 X %d passes %d check %d stress %d slot %d STEP_MIN "
                "%d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (cell, 63 + pool, pool, batch, arms,
                                                                                 steps, X, passes, check, stress, slot,
                                                                                 smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        assert "--arms %s " % arms.replace(",", "\\,") in r.stdout, (cell, "printf %q escapes the commas")
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--chk-slot-every %d" % slot, "--chk-plant-extra 4",
                     "--joint-residency union", "--joint-extra %d" % X, "--tu-gemm natural",
                     "--check-passes %d" % passes, "--check-every %d" % check, "--require-stress %d" % stress,
                     "--spec-twin fused --spec-body fused", STEP_ROOT + "/benchmarks/Efficiency/r2_i2_cell.py",
                     "--mem=%s" % mem, "--time=%d" % smin, "i2_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
