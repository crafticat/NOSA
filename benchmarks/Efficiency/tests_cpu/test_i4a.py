"""CPU tests of rung I4A (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 1 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5: the arm L4PTFD = L4PTF (rung I3) with the LEAN ONE-ROW PASS
(benchmarks/Efficiency/r2_i4a_lean.py; the cell r2_i4a_cell.py; the step script r2_i4a_step.sh). The world:
tests_cpu/i4a_world.py over i3_world (the SHIPPED engine from its own source; every CUDA op a torch stand-in with the
kernel's flat addresses, lesson L28). CPU stand-ins do not establish GPU dispatch, ordering, timing or memory (the
cells i4am / i4aa / i4ab do). FAST is checked on itself against SH and against L4PTF, not certified by an EXACT arm.

GATE 0    test_i4a_gate0_files: the tree = I3's c4c64b2 except the I4A files and ONE module-level declaration block in
          each older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py: only added lines, no
          function changed); nosi/ and every I3 file are c4c64b2's bytes (L4PTF's code, bit for bit); the fork is
          fia6's blob; the lean module imports torch only; no nosi module and no older cell names it.
EQUAL     test_l4ptfd_equals_l4ptf (B 1, B 8 rowgroup, B 8 natural): L4PTFD's run = L4PTF's run BIT FOR BIT (i2_world.
          same_run: tokens, fingerprints, logits rows, read() incl. the trace and the counters, the round records, the
          engine and twin state, the buffers, tpos, qc, the layer-0 row writes); and on its own: SH's tokens (MUST),
          exactly N, logits within TOL, the selections of committed rows = SH's, the residency and host rows after
          every round, PRESENT / VB / TAIL / CHAIN 0. Preconditions (L34): rejects, full accepts, edges crossed, draft
          and tentative-block skips, pool moves; ONE-ROW passes with rows in the committed block, rows PAST the edge
          (B 8) and draft skips (a test-side recorder).
CONTRACT  test_l4ptfd_trace_d1_d3: the contract-4 trace (WS-A's checks), the D slots' and J slots' row writes, the
          deferred replays (D3, deferred writes 0), no residency undo (D1), exactly N; the cell's -D1 / -D3 / -SPAN /
          -COMMIT-N / -LEAN gates on the CPU evidence; the LEAN gate fails on L4PTF's record (CONTROL).
COUNT     test_l4ptfd_op_counts: an INDEPENDENT aten op counter (TorchDispatchMode; views and empty allocations
          excluded) per one-row pass and layer: L4PTF - L4PTFD = SAVED_PER_LAYER in every layer after the first and
          SAVED_LAYER0 in the first (r2_i4a_lean's stated numbers), saved_per_pass(L) per pass, nothing else changed
          (the joint passes' counts equal; the tokens equal); the named removals (index_put_, stack, the mask copy).
PLANTS    test_lean_plants_caught: mask_before_write, live_bias, stale_pass each change the run (not a no-op) and are
          CAUGHT by the cell's L4PTFD-EQF (bit for bit against L4PTF, the acceptance record included). SH's tokens
          are reported only: a draft-only error changes the drafts, and the joint pass verifies every draft.
WIRING    test_lean_refusals_and_no_host_sync.
CELL      test_i4a_cell_arguments, test_i4a_cell_pure_parts, test_i4a_registration_state, test_step_script_i4a,
          test_notes_i4a.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4a.py
"""
from __future__ import annotations

import ast
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
I3_TIP = "c4c64b22ca1bc41d7f4b23168a526058d2f412ff"         # nosi-r2-i3: rung I3 with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4a"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4A_NEW = ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
           "benchmarks/Efficiency/r2_i4a_step.sh", T_ + "i4a_world.py", T_ + "test_i4a.py", "docs/NOTES_i4a.md")
I4A_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py")
# I4D (nosi-r2-i4d: this tree acbe433 + the accounting and the arm L4PTFK): its files (its own Gate 0:
# tests_cpu/test_i4d.py); its two modules build on r2_i4a_lean / r2_i4a_cell (they may name them, scan (e))
I4A_NEW = I4A_NEW + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                     "benchmarks/Efficiency/r2_i4d_step.sh", T_ + "test_i4d.py")                 # I4D
I4A_USERS = ("r2_i4d_keep.py", "r2_i4d_cell.py")                                                 # I4D
# I4G (nosi-r2-i4g: I4A acbe433 + the arm L4PTFDG): its files (its own Gate 0: tests_cpu/test_i4g.py); I4A_LATER: the
# later modules that import or name the I4A modules (the name scan (e) skips them)
I4A_NEW = I4A_NEW + ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
                     "benchmarks/Efficiency/r2_i4g_step.sh", T_ + "i4g_world.py", T_ + "test_i4g.py",
                     "docs/NOTES_i4g.md")                                                        # I4G
I4A_LATER = ("r2_i4g_cell.py", "r2_i4g_graph.py")                                                # I4G
# I4K (nosi-r2-i4k: I4D 99c231e + I4G d7997e6 + the arm L4PTFDGKW): its files (its own Gate 0: tests_cpu/test_i4k.py);
# its cell and combo module name the I4A modules (scan (e) skips them)
I4A_NEW = I4A_NEW + ("benchmarks/Efficiency/r2_i4k_combo.py", "benchmarks/Efficiency/r2_i4k_cell.py",
                     "benchmarks/Efficiency/r2_i4k_step.sh", T_ + "i4k_world.py", T_ + "test_i4k.py")    # I4K
I4A_LATER = I4A_LATER + ("r2_i4k_combo.py", "r2_i4k_cell.py")                                    # I4K


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4a_world")
WT = _load("wsa_tests_lib", HERE / "test_wsa.py")           # WS-A's test helpers (filters, the trace and row checks)
I3, I, W, RW, SW, SR, BC, VR, KA, LN = X.I3, X.I, X.W, X.RW, X.SW, X.SR, X.BC, X.VR, X.KA, X.LN
K, N, L, BS, TOL = X.K0, X.N0, X.L, X.BS, X.TOL
R = K + 1


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4a_cell
    return r2_i4a_cell


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
def test_i4a_gate0_files():
    if not _have(I3_TIP):
        pytest.skip("no git or no commit %s in %s" % (I3_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I3_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4A_NEW) | set(I4A_DECL), sorted(changed - set(I4A_NEW) - set(I4A_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4A_NEW), [f for f in I4A_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I3_TIP, "--", "nosi").stdout.strip() == ""    # (b) L4PTF's code: c4c64b2's
    i3_files = [f for f in _git("ls-tree", "-r", "--name-only", I3_TIP, "--", "benchmarks/Efficiency").stdout.split()
                if "r2_i3" in f or "fork_vllm" in f or "r2_kv_arena" in f or "r2_wsa_cell" in f or "r2_i2_cell" in f]
    assert len(i3_files) >= 6 and _git("diff", "--name-only", I3_TIP, "--", *i3_files).stdout.strip() == ""
    for f in I4A_DECL:                                                    # (c) one added declaration block each
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I3_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        assert old == new, (f, sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I3_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert body and all(x.startswith("+") for x in body), (f, body[:5])
        assert any("r2_i4a_lean.py" in x for x in body) and any("I4A" in x for x in body), (f, body[:5])
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    tree = ast.parse((EFF / "r2_i4a_lean.py").read_text())             # (d) the lean module: torch only
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch"}, mods
    assert "triton" not in (EFF / "r2_i4a_lean.py").read_text()        # NO NEW KERNEL
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4a_") or p.name in I4A_USERS + I4A_LATER:
            continue
        assert "r2_i4a" not in p.read_text(), p.name


# ------------------------------------------------------------------ L4PTFD = L4PTF, bit for bit
@functools.lru_cache(maxsize=None)
def _pair(B, gemm="rowgroup", filt="periodic"):
    """(L4PTF's evidence, L4PTFD's evidence) of one world; L4PTFD's carries the one-row recorder (lean_rec)."""
    f = X.run_f(B, gemm=gemm, filt=WT._filter(filt))
    box = {}
    d = X.run_fd(B, gemm=gemm, filt=WT._filter(filt), on_runner=lambda r: box.update(rec=X.past_rows(r)))
    d["lean_rec"] = box["rec"]
    return f, d


CASES = dict(B1=(1, "rowgroup"), B8=(8, "rowgroup"), B8nat=(8, "natural"))


@pytest.mark.parametrize("case", list(CASES))
def test_l4ptfd_equals_l4ptf(case):
    B, gemm = CASES[case]
    f, d = _pair(B, gemm)
    fx, rd, u, lr = d["facts"], d["rd"], d["rd"]["union"], d["lean_rec"]
    c = rd["tent"]["calls"]
    print("I4A %s: max |logit - SH's| %.3e over %d rows (TOL %.0e); one-row passes %d (recorded %d): rows in the "
          "committed block %d, past the edge %d, draft skips %d; lean %s" % (
              case, d["max_logit_delta"], d["rows_compared"], TOL, rd["lean"]["passes"], lr["passes"], lr["inblock"],
              lr["past"], lr["skipped"], rd["lean"]))
    assert fx["rejects"] > 0 and fx["full"] > 0 and fx["crossed"] > 0, fx          # preconditions (L34)
    assert fx["skipped"] > 0 and fx["forced"] > 0 and fx["move_out"] + fx["swap"] > 0, fx
    assert lr["passes"] == rd["lean"]["passes"] > 0 and lr["inblock"] > 0 and lr["skipped"] > 0, lr
    if B > 1:
        assert lr["past"] > 0, lr                                       # one-row rows past the committed edge
    bad = I.same_run(X.without_lean(f), X.without_lean(d))
    assert bad == [], bad                                              # THE CLAIM: L4PTF's run, bit for bit
    assert rd["count"] == [N] * B, rd["count"]                        # exactly N (contract 4)
    assert d["ref_logits_bad"] == [] and d["bad_tokens"] == [], (d["ref_logits_bad"][:3], d["bad_tokens"][:3])
    assert d["max_logit_delta"] <= TOL, d["max_logit_delta"]
    assert d["sel_compared"] > 0 and d["sel_mismatch"] == [], d["sel_mismatch"][:5]
    assert d["resid"] == [] and d["host"] == [], (d["resid"][:2], d["host"][:2])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert u["dropped"] == 0 and u["vb_bad"] == 0 and u["tail_reads"] == 0 and u["chain_bad"] == 0, u
    assert rd["lean"]["layers"] == L * rd["lean"]["passes"], rd["lean"]
    assert rd["lean"]["passes"] == c["draft_steps"] + c["mixed_steps"] + c["j1_passes"], (rd["lean"], c)
    assert "lean" not in f["rd"] and f["runner"].meta.get("draft_step") is None     # L4PTF has no lean path


def test_l4ptfd_trace_d1_d3():
    B = 8
    f, d = _pair(B)
    ev = d
    tr, log = ev["rd"]["tent"], ev["runner"].T.rec
    c = tr["calls"]
    assert tr["deferred_writes"] == 0                                  # D3
    assert c["joint_rows"] == R * c["joint_passes"] and c["joint_passes"] > 0, c
    assert c["joint_passes"] + c["j1_passes"] == ev["rd"]["rounds"], c
    assert tr["replays"] == WT._groups(ev) > 0, (tr["replays"], WT._groups(ev))
    assert c["residency_undo"] == 0                                    # D1
    assert ev["rd"]["tokens_acct"]["remade_rows"] == 0
    errs, remade = WT._trace_errors(ev, B, False)
    assert errs == [] and remade == [], (errs[:3], remade[:3])
    assert WT._joint_row_errors(log, ev["rd"]["trace_all"], B) == []
    assert WT._slot_row_errors(log, ev["rd"]["trace_all"], B) == []   # the D slots' rows: the trace's positions
    rerr, seen = WT._replay_errors(log, B)
    assert rerr == [] and seen == WT._groups(ev), (rerr[:5], seen)
    assert WT._pos_errors(log) == []
    assert ev["rd"]["trace_all"] == f["rd"]["trace_all"]               # the trace = L4PTF's
    C = _cell()
    WAc = C.WA
    res = dict(read=ev["rd"], K=K)
    g = {n: ok for n, _, ok, _ in WAc.tent_gates("L4PT", res, None)}
    assert all(g.values()) and set(g) == {"L4PT-D1", "L4PT-D3", "L4PT-SPAN"}, g
    assert WAc.commit_n_gate("L4PTFD", res, N)[2] is True
    n, m, ok, det = C.lean_gate("L4PTFD", dict(ev["rd"], K=K), L)
    assert (n, m, ok) == ("L4PTFD-LEAN", True, True), det
    assert C.lean_gate("L4PTF", dict(f["rd"], K=K), L)[2] is False    # CONTROL: L4PTF's record has no lean path


# ------------------------------------------------------------------ the op counts
def _count(arm, B=4):
    model, c, S, hz, tok0 = X.small_world(B=B)
    build = (I3.tentf_builder(check=False, stats=False) if arm == "L4PTF" else
             X.tentfd_builder(check=False, stats=False))
    run = build(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)
    box = X.count_one_row_ops(run)
    run.run(8, rounds=3)
    return box, run


def test_l4ptfd_op_counts():
    bf, rf = _count("L4PTF")
    bd, rdn = _count("L4PTFD")
    assert torch.equal(rf.out, rdn.out) and torch.equal(W.bits(rf.fmax), W.bits(rdn.fmax))   # the same run
    np_ = len(bf["passes"])
    assert np_ > 0 and len(bd["passes"]) == np_ and len(bf["layers"]) == np_ == len(bd["layers"]), (np_, bd["passes"])
    assert all(len(x) == L for x in bf["layers"] + bd["layers"])
    dl = [[a - b for a, b in zip(x, y)] for x, y in zip(bf["layers"], bd["layers"])]
    dp = [a - b for a, b in zip(bf["passes"], bd["passes"])]
    gone = {k: bf["names"][k] - bd["names"].get(k, 0) for k in bf["names"] if bf["names"][k] != bd["names"].get(k, 0)}
    print("I4A COUNT: %d one-row passes of L %d; aten compute ops per pass L4PTF %s, L4PTFD %s; _rows per layer "
          "L4PTF %s, L4PTFD %s; L4PTF - L4PTFD per layer %s, per pass %s (stated %d per layer, %d in layer 0, %d per "
          "pass); joint passes %s / %s; ops removed by name %s" % (
              np_, L, bf["passes"][:3], bd["passes"][:3], bf["layers"][0], bd["layers"][0], dl[0], dp[:3],
              LN.SAVED_PER_LAYER, LN.SAVED_LAYER0, LN.saved_per_pass(L), bf["joint"][:2], bd["joint"][:2],
              dict(sorted(gone.items()))))
    assert all(x[0] == LN.SAVED_LAYER0 and all(y == LN.SAVED_PER_LAYER for y in x[1:]) for x in dl), dl[:3]
    assert all(x == LN.saved_per_pass(L) for x in dp), dp[:5]          # nothing outside _rows changed
    assert all(b < a for a, b in zip(bf["passes"], bd["passes"]))
    assert bf["joint"] == bd["joint"] and len(bf["joint"]) > 0, (bf["joint"], bd["joint"])   # the joint pass: L4PTF's
    assert LN.saved_per_pass(32) == 37 * 31 + 11 == 1158
    assert gone.get("index_put_") == np_ * (3 * L + 1) and gone.get("stack") == np_ * L, gone   # the named removals
    assert gone.get("masked_fill_") == np_ * L and gone.get("copy_") == np_ * L, gone            # (the mask copy)


# ------------------------------------------------------------------ PLANTS
def _cpu(ev):
    """The cell's EQF input (r2_i4a_cell.eq_gate) from a CPU run."""
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=_cell().acct_record(ev["rd"]),
                state=_cell().state_checksum(r))


@pytest.mark.parametrize("dneg", ["mask_before_write", "live_bias", "stale_pass"])
def test_lean_plants_caught(dneg):
    """A planted error in the draft step changes the drafts. The joint pass verifies every draft, so a draft-only error
    may leave every committed token SH's (speculation hides it): SH's tokens are REPORTED here, not required to catch
    it. The catching gate is the cell's L4PTFD-EQF (bit for bit against L4PTF, the acceptance record included)."""
    f, d = _pair(8)
    assert d["lean_rec"]["skipped"] > 0                                # precondition of live_bias: draft skips
    p = X.run_fd(8, filt=WT._filter("periodic"), dneg=dneg)
    bad = I.same_run(X.without_lean(f), X.without_lean(p))
    sh_caught = p["bad_tokens"] != [] or p["max_logit_delta"] > TOL
    n, m, ok, det = _cell().eq_gate("L4PTFD", _cpu(p), _cpu(f), N, True)
    print("I4A PLANT %s: differs from L4PTF in %s; caught by SH's tokens / TOL %s (bad tokens %d, max |logit - SH's| "
          "%.3e); EQF: %s" % (dneg, [b if isinstance(b, str) else b[0] for b in bad][:8], sh_caught,
                              len(p["bad_tokens"]), p["max_logit_delta"], det))
    assert bad != [], dneg                                             # not a no-op (L34)
    assert ok is False, (dneg, det)                                    # CAUGHT by L4PTFD-EQF
    assert p["rd"]["count"] == [N] * 8                                 # exactly N even with the plant (contract 4)


# ------------------------------------------------------------------ WIRING
def test_lean_refusals_and_no_host_sync():
    model, c, S, hz, tok0 = X.small_world(B=2)
    with pytest.raises(SR.SpecError):
        X.tentfd_builder(dneg="nope")(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)
    model, c, S, hz, tok0 = X.small_world(B=2)
    run = X.tentfd_builder(check=False, stats=False)(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0,
                                                     record=False)
    assert isinstance(run.T, X.TentStateFD) and isinstance(run, X.STF.TentRunnerF)
    with RW.SyncTrap(Path(LN.__file__)) as trap:
        run.run(8, rounds=3)
    assert trap.hits == [], trap.hits[:10]                             # no host read in the lean pass
    assert run.T.lean["passes"] > 0 and run.T.lean["layers"] == L * run.T.lean["passes"]
    assert run.meta["draft_step"] == "lean" and run.read()["lean"] == run.T.lean
    with RW.SyncTrap(Path(__file__)) as trap:                          # CONTROL: the trap is not blind (a host read
        torch.zeros(2).sum().item()                                    # from the traced file is recorded)
    assert [h[2] for h in trap.hits] == ["item"], trap.hits


# ------------------------------------------------------------------ CELL
def test_i4a_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.fork_store, a.prefill_hook, a.tu_gemm, a.check_every,
            a.require_stress, a.eq_must, a.probe_every) == (
        ["L4PTF", "L4PTFD"], 216, 512, 65, 3, 0, "nocopy", "natural", 0, 0, 0, 16)
    for bad in (["--arms", "L4PTFD,L4PTF"], ["--arms", "L4PT"], ["--arms", "L4PTFD", "--eq-must", "1"],
                ["--spec", "4"], ["--fork-store", "100"], ["--arms", "L4PTF,L4PTF"], ["--pool", "0"],
                ["--steps", "4"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1",
                             "--require-stress", "1", "--eq-must", "1"])
    assert (a.check_every, a.require_stress, a.eq_must) == (1, 1, 1)
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4a_cell_pure_parts():
    C = _cell()
    f, d = _pair(8)
    n, m, ok, det = C.eq_gate("L4PTFD", _cpu(d), _cpu(f), N, True)
    assert (n, m, ok) == ("L4PTFD-EQF", True, True), det
    planted = _cpu(d)
    planted["out"] = planted["out"].clone()
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    assert C.eq_gate("L4PTFD", planted, _cpu(f), N, False)[2] is False
    planted = _cpu(d)
    planted["acct"] = dict(planted["acct"], round_accepted=planted["acct"]["round_accepted"][:-1] + [-1])
    assert C.eq_gate("L4PTFD", planted, _cpu(f), N, False)[2] is False   # CONTROL: the acceptance record changed
    planted = _cpu(d)
    planted["state"] = planted["state"][:-1] + [planted["state"][-1] + 1]
    assert C.eq_gate("L4PTFD", planted, _cpu(f), N, False)[2] is False   # CONTROL: one state checksum changed
    assert len(_cpu(d)["state"]) == 3 + L * 7                          # tpos, qc, ctr_c; 7 per layer (with the pool)
    assert C.eq_gate("L4PTFD", _cpu(d), None, N, True)[2] is False
    n, m, ok, det = C.mem_record("L4PTFD", dict(prefill_peak_GB=78.13), 82.57)
    assert ok and "83.891 GB" in det and "usable ceiling 84.25 GB" in det, det
    assert C.peak_gb(dict(gpu_peak_alloc_GB=82.559, mem_arena=dict(prefill_peak_GB=78.13))) == 83.891
    assert C.peak_gb(dict(gpu_peak_alloc_GB=76.932, mem_arena=dict(prefill_peak_GB=72.35))) == 77.685
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh, {}, 216) == [("L4PTFD-RUN", "NOT_RUN", "the arm did not run")]
    rec = lambda tok, dr, jt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr, joint=jt),  # noqa
                                   tokens=dict(equal_fraction=0.4, first=(1, 1)), vs_strongest=0.5,
                                   gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0))
    saved = {b: dict(C.PRED_I4A_BY_B[b]) for b in C.PRED_I4A_BY_B}        # the REPORT check runs on an all-None block
    try:
        for b in C.PRED_I4A_BY_B:
            C.PRED_I4A_BY_B[b].update({k: None for k in C.PRED_KEYS})
        v = {x[0]: x for x in C.verdicts(sh, dict(L4PTF=rec(400.0, 20.0, 30.0), L4PTFD=rec(440.0, 16.0, 30.0)), 216,
                                         eqf=True)}
    finally:
        for b in saved:
            C.PRED_I4A_BY_B[b].clear()
            C.PRED_I4A_BY_B[b].update(saved[b])
    assert all(x[1] == "REPORT" for x in v.values()), v                # registration None: every key REPORT
    assert "1.1000" in v["I4A-VS-I3"][2] and "4.0000" in v["I4A-DRAFT-SAVING"][2] and "0.8000" in v[
        "I4A-DRAFT-STEP-SH"][2], v
    assert set(v) == {"I3-RATIO", "I4A-RATIO", "I4A-VS-I3", "I4A-DRAFT-STEP-SH", "I4A-DRAFT-SAVING",
                      "I4A-JOINT-RATIO", "I4A-PEAK-GB", "I4A-AGREE", "I4A-EQ-I3", "I4A-ABOVE-I3", "I4A-VS-STRONGEST"}


def test_i4a_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4A is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4A_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4A)
    assert set(C.PRED_I4A_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4A_BY_B.values())
    src = (EFF / "r2_i4a_cell.py").read_text()
    assert ("\nPRED_STAMP_I4A = None\n" in src) == (C.PRED_STAMP_I4A is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_i4a_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i4a():
    src = (EFF / "r2_i4a_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "benchmarks/Efficiency/r2_i4a_lean.py" in src and "OUTROOT/sp_active_$JOB" in src
    for bad in ("i3m", "i3a", "i4a", "i4ac"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4A = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4a_cell.py" % sha).stdout
    for cell, pool, batch, steps, check, stress, eq, smin, mem in (
            ("i4am", 10, 8, 72, 1, 1, 1, 40, "100G"),
            ("i4aa", 65, 216, 512, 0, 0, 0, 80, "240G"),
            ("i4ab", 10, 318, 512, 0, 0, 0, 110, "340G")):
        r = _step(cell, sha)
        if cell in ("i4aa", "i4ab") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4A is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L4PTF,L4PTFD steps %d K 3 fork-store 0 check %d stress %d eq %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, check, stress, eq, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress, "--eq-must %d" % eq,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4a_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4a_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms L4PTF\\?,L4PTFD ", r.stdout), (cell, r.stdout[-600:])   # printf %q may quote ","
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", "L4PTF,L4PTFD", "--check-every", str(check),
                                "--require-stress", str(stress), "--eq-must", str(eq)])
        assert a.arms == ["L4PTF", "L4PTFD"]                            # every cell's arguments parse


def test_notes_i4a():
    doc = (ROOT / "docs/NOTES_i4a.md").read_text()
    for word in ("test_i4a.py", "PRED_I4A_BY_B", "PRED_STAMP_I4A", "INFERENCE", "MEASURED", "CODE", "i4am", "i4aa",
                 "i4ab", "REQUIRED", "REMOVABLE", "84.25", "L59", "spec_tentative.py:395-448", "SAVED_PER_LAYER",
                 "1,158", "CODE_EXTRA_PER_LAYER"):
        assert word in doc, word
