"""CPU tests of WS-A (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE TENTATIVE
BUFFERS (nosi/nosi/spec_tentative.py TentRunner, opt-in --spec-rollback tentative of r2_wsa_cell.py; the FAST form) and
THE SCHEDULE OF CONTRACT 4 (the plan's human decisions (a) refill after a reject, (b) drain at the output limit) in both
runners: TentRunner (FAST) and PaperJournalRunner (EXACT, spec_round's journal). FAST is checked ON ITSELF against SH (the
shipped decode at the same configuration), not certified by the EXACT path. The world: tests_cpu/wsa_world.py over
spec_world.py (the SHIPPED engine from its own source, every CUDA op a torch stand-in with the kernel's flat addresses,
L28). B 1 and B 8. No CUDA, no Triton launch. K = 3. THE CPU STAND-INS MODEL THE ARITHMETIC ONLY: GPU dispatch, streams,
ordering, timing and memory need the GPU cell wa1.

GATE 0   test_wsa_flag_off_is_spec5 (B 1 / B 8): the tree is nosi-r2-spec5 40ac348 except the declared WS-A files; the
         five older Gate 0 tests only ADD lines that name spec_tentative; no other nosi module names spec_tentative; it
         names none of the modules the older name scans guard; the cell's flag is off by default and its runners are
         chosen under their flags; an attached TentRunner / PaperJournalRunner and no round: 16 plain steps bitwise the
         plain world, detach() restores every callable.
SPEC     test_paper_spec_model_literal: the test's own plain-rule model of the schedule gives Figure 3 (fill d1 d2 d3;
         verify d1 d2, defer d3, draft d4; next group d3 d4 d5, defer d6), decision (a) (after a reject: draft d1' d2' d3',
         then verify d1' d2', defer d3', draft d4') and decision (b) (a final group verifies every pending draft up to the
         limit, ordinary target rows after a reject there, exactly N tokens).
SCHEDULE test_paper_schedule_trace (FAST / EXACT, B 1 / B 8): the runner's per-pass position trace (fed, verified,
         deferred, drafted positions, accepted and committed counts) equals the model's, driven by the run's own accepted
         counts; FAST remakes nothing, EXACT remakes the deferred row of a group and the earlier positions of a fill;
         every request commits exactly N tokens; the committed tokens are SH's across the rejects, the refills and the
         end (EXACT: the logits rows bitwise too, residency and host rows after every round). The trace is checked
         against the rows that ran, not the host plan: FAST every J / P / D slot's layer-0 row writes, EXACT the joint
         pass's q and every active D slot's twin position (the prelude excepted); a row or slot one position off: CAUGHT.
         Preconditions: refills, final groups, target rows after a final reject.
TOKENS   test_tent_commits_sh_tokens (L4P / L3 at B 1 / B 8): FAST commits SH's tokens; the logits of every committed row
         are within TOL of SH's (printed); the TARGET SELECTION of every committed verify / target row, layer and head
         equals SH's at the same position (the reference fed SH's tokens is checked bitwise against SH's logits); the
         residency and the host rows after every round; FAST's tokens equal the EXACT run's. Preconditions: rejects and
         full accepts, commits across a block edge, rows past an edge, F4 fill rows, forced and draft skips, pool moves,
         (B 8) ragged positions.
STATE    test_tent_committed_state (B 1 / B 8): the committed state against SH that fed only the committed tokens, field by
         field: EXACT fields equal, ROUNDED fields layer-0 bitwise and within TOL, CACHE fields reported.
DEFER    test_tent_deferred_not_recomputed (B 1 / B 8): no joint pass writes the deferred position; a joint pass has K + 1
         rows; one replay per group from the entry that a draft row wrote, unchanged; twin position = rope position. Plants
         recompute_deferred and no_replay: CAUGHT.
HOST     test_tent_reject_keeps_host_rows (B 1 / B 8): a chain fill that its commit did not keep; no host row past the
         committed block is written; the complete committed blocks are SH's. Plant writeback_tentative: CAUGHT.
COST     test_tent_rollback_is_fixed_cost (B 8): torch calls per reset and per commit constant over two runs with different
         reject counts; a reset writes no engine table; the dropped entries = a recount from the layer-0 buffer writes.
         Plant undo_residency: CAUGHT.
LIFETIME test_tent_buffer_lifetime (B 1 / B 8): dead entries poisoned change nothing. Plants no_causal, no_lower: CAUGHT.
WIRING   test_tent_refusals_hooks_check_rounds, test_tent_no_host_sync_in_round (L4P / L3).
CELL     test_wsa_cell_arguments_and_report, test_step_script_wa1.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsa.py
"""
from __future__ import annotations

import ast
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
SPEC5 = "40ac348870e89b43ac5278458b660dea8d360508"          # nosi-r2-spec5: the base of every workstream
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-wsa"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
DECLARED = ("nosi/nosi/spec_tentative.py", "benchmarks/Efficiency/r2_wsa_cell.py", "benchmarks/Efficiency/r2_wsa_step.sh",
            "benchmarks/Efficiency/tests_cpu/wsa_world.py", "benchmarks/Efficiency/tests_cpu/test_wsa.py",
            "docs/NOTES_wsa.md")
GATE0_TESTS = ("benchmarks/Efficiency/tests_cpu/test_ragged_engine.py", "benchmarks/Efficiency/tests_cpu/test_verify_rows.py",
               "benchmarks/Efficiency/tests_cpu/test_rollback.py", "benchmarks/Efficiency/tests_cpu/test_spec_round.py",
               "benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py")
# I2 (nosi-r2-i2: r2-wsareg 83dac96 + r2-wsb 1a8ce68 + the arm L4PTU): the WS-B files and the I2 files are declared too
# (their own Gate 0s: tests_cpu/test_wsb.py, tests_cpu/test_i2.py); in the Gate 0 tests above, the lines of WS-B's own
# diff (40ac348 -> 1a8ce68) are allowed besides the added lines that name spec_tentative
WSB_TIP = "1a8ce68d9d6852fecf45b916c9ef1e6a5a80e098"
DECLARED = DECLARED + ("nosi/nosi/verify_rows.py", "nosi/nosi/spec_round.py", "benchmarks/Efficiency/r2_wsb_cell.py",
                       "benchmarks/Efficiency/r2_wsb_step.sh", "benchmarks/Efficiency/tests_cpu/wsb_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_wsb.py", "docs/NOTES_wsb.md",              # WS-B
                       "nosi/nosi/spec_tentative_union.py", "benchmarks/Efficiency/r2_i2_cell.py",
                       "benchmarks/Efficiency/r2_i2_step.sh", "benchmarks/Efficiency/tests_cpu/i2_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i2.py", "docs/NOTES_i2.md",                # I2
                       "benchmarks/Efficiency/r2_i2_check.py", "benchmarks/Efficiency/r2_wsg_check.py")  # I2
# I3 (nosi-r2-i3: I2 63da01d + WS-C2 db21536 + the arm L4PTF): WS-C's files and the I3 files are declared too (their own
# Gate 0s: tests_cpu/test_wsc.py, tests_cpu/test_i3.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/fork_vllm_unified_attention.py", "benchmarks/Efficiency/r2_kv_arena.py",
                       "benchmarks/Efficiency/r2_wsc_cell.py", "benchmarks/Efficiency/r2_wsc_step.sh",
                       "benchmarks/Efficiency/tests_cpu/wsc_world.py", "benchmarks/Efficiency/tests_cpu/test_wsc.py",
                       "benchmarks/Efficiency/tests_cpu/test_wsc_interp.py", "docs/NOTES_wsc.md",       # WS-C
                       "nosi/nosi/spec_tentative_fork.py", "benchmarks/Efficiency/r2_i3_arena.py",
                       "benchmarks/Efficiency/r2_i3_cell.py", "benchmarks/Efficiency/r2_i3_step.sh",
                       "benchmarks/Efficiency/tests_cpu/i3_world.py", "benchmarks/Efficiency/tests_cpu/test_i3.py",
                       "benchmarks/Efficiency/tests_cpu/test_i3_interp.py", "docs/NOTES_i3.md")         # I3
# I4A (nosi-r2-i4a: I3 c4c64b2 + the arm L4PTFD): its files (its own Gate 0: tests_cpu/test_i4a.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
                       "benchmarks/Efficiency/r2_i4a_step.sh", "benchmarks/Efficiency/tests_cpu/i4a_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4a.py", "docs/NOTES_i4a.md")                # I4A
# I4D (nosi-r2-i4d: I4A acbe433 + the accounting and the arm L4PTFK): its files (its own Gate 0: tests_cpu/test_i4d.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                       "benchmarks/Efficiency/r2_i4d_step.sh", "benchmarks/Efficiency/tests_cpu/test_i4d.py")  # I4D
# I4G (nosi-r2-i4g: I4A acbe433 + the arm L4PTFDG): its files (its own Gate 0: tests_cpu/test_i4g.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
                       "benchmarks/Efficiency/r2_i4g_step.sh", "benchmarks/Efficiency/tests_cpu/i4g_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4g.py", "docs/NOTES_i4g.md")                # I4G
# I4K (nosi-r2-i4k: I4D 99c231e + I4G d7997e6 + the arm L4PTFDGKW): its files (its own Gate 0: tests_cpu/test_i4k.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4k_combo.py", "benchmarks/Efficiency/r2_i4k_cell.py",
                       "benchmarks/Efficiency/r2_i4k_step.sh", "benchmarks/Efficiency/tests_cpu/i4k_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4k.py")                                     # I4K
# I4U (nosi-r2-i4u: I4A acbe433 + the arm L4PTFDU): its files (its own Gate 0: tests_cpu/test_i4u.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
                       "benchmarks/Efficiency/r2_i4u_step.sh", "benchmarks/Efficiency/tests_cpu/i4u_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4u.py")                                     # I4U
# I4H (nosi-r2-i4h: I4G d7997e6 + the arm L4PTFDGD): its files (its own Gate 0: tests_cpu/test_i4h.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4h_graph.py", "benchmarks/Efficiency/r2_i4h_cell.py",
                       "benchmarks/Efficiency/r2_i4h_step.sh", "benchmarks/Efficiency/tests_cpu/i4h_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4h.py", "docs/NOTES_i4h.md")                # I4H
# I4S (nosi-r2-i4s: I4K 70059e5 + I4U 832dca5 + I4H fc6c680 + the stacked arms): its files (its own Gate 0:
# tests_cpu/test_i4s.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4s_stack.py", "benchmarks/Efficiency/r2_i4s_cell.py",
                       "benchmarks/Efficiency/r2_i4s_step.sh", "benchmarks/Efficiency/tests_cpu/i4s_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4s.py")                                     # I4S
# I4CC (nosi-r2-i4cc: I4S 8f236f8 + the compiled arms): its files (its own Gate 0: tests_cpu/test_i4cc.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4cc_compile.py", "benchmarks/Efficiency/r2_i4cc_cell.py",
                       "benchmarks/Efficiency/r2_i4cc_step.sh", "benchmarks/Efficiency/tests_cpu/i4cc_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4cc.py")                                    # I4CC
# I4DC (nosi-r2-i4dc: I4CC d75a946 + the compiled draft helpers): its files (its own Gate 0: tests_cpu/test_i4dc.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4dc_compile.py", "benchmarks/Efficiency/r2_i4dc_cell.py",
                       "benchmarks/Efficiency/r2_i4dc_step.sh", "benchmarks/Efficiency/tests_cpu/i4dc_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4dc.py")                                    # I4DC
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_recon7_protocol.py", "benchmarks/Efficiency/r2_recon7_probe.py",
                       "benchmarks/Efficiency/r2_recon7_cell.py", "benchmarks/Efficiency/r2_recon7_streams.py",
                       "benchmarks/Efficiency/r2_recon7_mem.py", "benchmarks/Efficiency/r2_recon7_step.sh",
                       "benchmarks/Efficiency/r2_i4c_cell.py", "benchmarks/Efficiency/r2_i4m_cell.py",
                       "benchmarks/Efficiency/r2_wsm_cell.py", "benchmarks/Efficiency/tests_cpu/test_recon7.py",
                       "docs/NOTES_recon7.md")                                                             # RECON7
# J1 (nosi-r2-j1: RECON7 cd0f0c5 + the compiled joint pooling of AREA 1): its files
# (its own Gate 0: tests_cpu/test_j1.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_j1_select.py", "benchmarks/Efficiency/r2_j1_cell.py",
                       "benchmarks/Efficiency/r2_j1_step.sh", "benchmarks/Efficiency/tests_cpu/test_j1.py",
                       "docs/NOTES_j1.md")                                                                 # J1
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_j1_forkbench.py",)                                       # J1
# ST1 (nosi-r2-st1: RECON7 cd0f0c5 + AREA 3, the captured row commit of the best arms, r2_st1_cell.py): its files (its
# own Gate 0: tests_cpu/test_st1.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_st1_commit.py", "benchmarks/Efficiency/r2_st1_cell.py",
                       "benchmarks/Efficiency/r2_st1_step.sh", "benchmarks/Efficiency/tests_cpu/test_st1.py")     # ST1
# INT (nosi-r2-int: J1 + ST1 + BF1 in one arm, r2_int_cell.py): its files (its own Gate 0: tests_cpu/test_int.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_int_arms.py", "benchmarks/Efficiency/r2_int_cell.py",
                       "benchmarks/Efficiency/r2_int_step.sh", "benchmarks/Efficiency/tests_cpu/test_int.py",
                       "docs/NOTES_int.md")                                                                # INT
# BF1 (nosi-r2-bf1: RECON7 cd0f0c5 + the draft-issued background fetch): its files (its own Gate 0: tests_cpu/test_bf1.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_bf1_fetch.py", "benchmarks/Efficiency/r2_bf1_cell.py",
                       "benchmarks/Efficiency/r2_bf1_step.sh", "benchmarks/Efficiency/tests_cpu/test_bf1.py")  # BF1


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


WW = _load("wsa_world")
SW, W, RW, VW, RE, SR, BC, ST = WW.SW, WW.W, WW.RW, WW.VW, WW.RE, WW.SR, WW.BC, WW.ST
K, N, L, BS, TOL = WW.K0, WW.N0, WW.L, WW.BS, WW.TOL
ST_PATH = W.N / "spec_tentative.py"
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)       # as test_spec_paper.py: draft 1 wrong in some rounds
FILTER0 = SW.corrupt(3, 0, kinds=("draft",), phase=1)      # draft 0 wrong: early rejects
S1 = [4658]
S8 = [4659, 4658, 4657, 4723, 4722, 4721, 4630, 4691]      # test_rollback's: fills within 16 plain steps
FIELDS = ("round", "slot", "role", "q", "fed_v", "fed_d", "verify", "deferred", "drafted", "n", "committed")


def _end_corrupt(N_):
    """Every draft whose committed column would be N - 3 or later is wrong: the final groups reject, so target rows follow
    (decision (b)'s ordinary decoding)."""
    def f(runner, kind, j, tok):
        if runner.next_col is None:
            return tok
        hit = (runner.next_col.to(tok.device) >= N_ - 3)
        return torch.where(hit, (tok + 1) % WW.V, tok)
    return f


def _filter(name):
    return dict(periodic=FILTER, zero=FILTER0, none=None, end=SW.all_kinds(FILTER, _end_corrupt(N)))[name]


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_wsa_cell
    return r2_wsa_cell


def _ev(level, B, filt="periodic", tneg=None, poison=False, ref=True, S=WW.S0):
    """One FAST run_tent per parameter set (shared by the claims of this module; the key is normalised, so a keyword
    and a default give the same run)."""
    return _ev_cached(level, B, filt, tneg, poison, ref, S)


@functools.lru_cache(maxsize=None)
def _ev_cached(level, B, filt, tneg, poison, ref, S):
    cap = tneg is not None and tneg != "writeback_tentative"           # the HOST plant runs to its clean run's fill
    return WW.run_tent(B, level=level, draft_filter=_filter(filt), tneg=tneg, poison=poison, with_reference=ref, S=S,
                       max_rounds=(12 if cap else 0))


def _exact(level, B, filt="periodic"):
    """The EXACT journal run: L4P the schedule of contract 4 (PaperJournalRunner), L3 spec_round's L3."""
    return _exact_cached(level, B, filt)


@functools.lru_cache(maxsize=None)
def _exact_cached(level, B, filt):
    return SW.run_vs_shipped(B, level=level, mode="exact", gemm="rowgroup", draft_filter=_filter(filt),
                             sr=(WW.JOURNAL if level == "L4P" else None))


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


# ------------------------------------------------------------------ GATE 0
def _guarded_calls(fn, attr: str, test_src: str) -> tuple:
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == attr]
    under = [c for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == test_src
             for c in calls if any(c is x for s in n.body for x in ast.walk(s))]
    return len(calls), len(under)


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_wsa_flag_off_is_spec5(B):
    if _have(SPEC5):
        r = _git("diff", "--name-only", SPEC5, "--", ".", *[":(exclude)" + f for f in DECLARED + GATE0_TESTS],
                 ":(exclude,glob)**/__pycache__/**")
        assert r.returncode == 0 and r.stdout.strip() == "", "files differ from %s:\n%s" % (SPEC5, r.stdout)
        for f in GATE0_TESTS:                                 # their Gate 0s: only added lines naming spec_tentative
            d = _git("diff", "-U0", SPEC5, "--", f).stdout.splitlines()
            body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
            wsb = set(_git("diff", "-U0", SPEC5, WSB_TIP, "--", f).stdout.splitlines()) if _have(WSB_TIP) else set()
            body = [x for x in body if x not in wsb]          # I2: WS-B's own lines of this file (its Gate 0)
            assert body and all(x.startswith("+") and "spec_tentative" in x for x in body), (f, body[:5])
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    assert [n for n, t in src.items() if "spec_tentative" in t and n not in (
        "spec_tentative.py", "spec_tentative_union.py", "spec_tentative_fork.py")] == []   # I2, I3
    for name in ("ragged_rollback", "state_snapshot", "spec_prefetch"):   # the older Gate 0 name scans stay true
        assert name not in src["spec_tentative.py"], name
    for f in ("r2_spec_cell.py", "r2_spec5_cell.py", "r2_base_cell.py", "r2_verify_cell.py"):
        assert "spec_tentative" not in (EFF / f).read_text(), f
    C = _cell()
    base = ["--out", "x", "--workload", "w", "--spec", "3"]
    a = C.parse_args(base)
    assert a.spec_rollback == "journal" and a.spec_arms == ["L4P"]           # the flag is off by default
    with pytest.raises(SystemExit):
        C.parse_args(base + ["--spec-arms", "L4P,L4PT"])                   # a tentative arm needs the flag
    tree = ast.parse((EFF / "r2_wsa_cell.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_arm")
    assert _guarded_calls(fn, "TentRunner", "tent") == (1, 1)
    assert _guarded_calls(fn, "PaperJournalRunner", "paper") == (1, 1)
    S_list = S1 if B == 1 else S8
    for make in (lambda m_, y_: ST.TentRunner(m_, y_, SR.SpecOps(level="L4P", K=K, attn=VW.fake_attention_idx,
                                                                 journal=RW.CAP), BC.HostClock()),
                 lambda m_, y_: ST.PaperJournalRunner(m_, y_, SR.SpecOps(level="L4P", K=K, attn=VW.fake_attention_idx,
                                                                         journal=RW.CAP), BC.HostClock())):
        (mx, x), (my, y) = RW.build(S_list), RW.build(S_list)
        assert RW.diff(RW.state(x), RW.state(y)) == [], "precondition: two worlds of one seed differ before step 0"
        orig = dict(y.ops.kernels)
        lin = y.ops.linear
        runner = make(my, y)
        tok = RW.tokens(0, B)
        dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
        h0 = RW.state(x)
        for i in range(16):
            lx, ly = dx.plain(1)[0], dy.plain(1)[0]
            assert torch.equal(W.bits(lx), W.bits(ly)), "step %d: the attached world's logits differ" % i
        assert RW.diff(RW.state(x), RW.state(y)) == []
        assert any(n.endswith("_k_cpu") for n in RW.diff(h0, RW.state(x))), "precondition: a block fill in 16 steps"
        runner.detach()
        assert all(y.ops.kernels[k] is v for k, v in orig.items()) and y.ops.linear is lin
        assert "decode_fixed" not in y.body.__dict__ and getattr(y, "rollback", None) is None
        assert all("step_unchecked" not in t.__dict__ for t in y.st.twins)


# ------------------------------------------------------------------ SPEC: the test's own model of the schedule
def spec_trace(q0: int, left: int, K_: int, rounds: int, outcomes) -> list:
    """THE SCHEDULE OF CONTRACT 4 as plain rules, written from the plan's text and Figure 3 (not from the runner):
    q is the committed position (the last committed token at q, not fed), pend the drafted tokens at q + 1 .. q + pend
    not yet verified, left the tokens still to commit. Slots: K draft slots P (the initial fill), then per round a joint
    slot J and K - 1 draft slots D. outcomes: the accepted counts n of this request's verification groups (group and
    final), in order. Returns the entries (FIELDS)."""
    out, it = [], iter(outcomes)
    q, pend, ordinary = q0, 0, False

    def e(rnd, slot, role, **kw):
        x = dict(round=rnd, slot=slot, role=role, q=(q if slot == "J" else None), fed_v=[], fed_d=[], verify=[],
                 deferred=None, drafted=None, n=None, committed=0)
        x.update(kw)
        return x
    for j in range(K_):                                              # the initial fill: d1, d2, ..., dK
        out.append(e(-1, "P%d" % j, "draft", fed_d=[q + pend], drafted=q + pend + 1))
        pend += 1
    for g in range(rounds):
        if left <= 0:
            continue
        wait = False
        if ordinary:                                                 # (b) ordinary decoding after the final group
            out.append(e(g, "J", "target", fed_v=[q], committed=1))
            q, left = q + 1, left - 1
        elif left <= K_:                                             # (b) the final group: every pending draft up to N
            v = min(pend, left)
            n = next(it)
            m = v if n == v else n + 1
            out.append(e(g, "J", "final", fed_v=list(range(q, q + v)), verify=list(range(q + 1, q + v + 1)), n=n,
                         committed=m))
            q, left, pend, ordinary, wait = q + m, left - m, 0, True, True
        elif pend < K_:                                              # (a) the new fill's last draft, then wait
            out.append(e(g, "J", "fill", fed_d=[q + pend], drafted=q + pend + 1))
            pend, wait = pend + 1, True
        else:                                                        # Figure 3: verify all but the newest, defer it
            v = pend - 1
            n = next(it)
            m = v if n == v else n + 1
            out.append(e(g, "J", "group", fed_v=list(range(q, q + v)), fed_d=[q + pend],
                         verify=list(range(q + 1, q + v + 1)), deferred=q + pend, drafted=q + pend + 1, n=n,
                         committed=m))
            if n == v:
                q, pend = q + v, 2                                   # the deferred token and the new draft stay
            else:
                q, pend = q + m, 0                                   # (a) a reject starts a new fill
            left -= m
        for j in range(K_ - 1):
            if wait:
                out.append(e(g, "D%d" % j, "idle"))
            elif ordinary:
                if left > 0:
                    out.append(e(g, "D%d" % j, "target", fed_v=[q], committed=1))
                    q, left = q + 1, left - 1
                else:
                    out.append(e(g, "D%d" % j, "idle"))
            else:
                out.append(e(g, "D%d" % j, "draft", fed_d=[q + pend], drafted=q + pend + 1))
                pend += 1
    return out


def _pick(es, round_, slot):
    return next(x for x in es if x["round"] == round_ and x["slot"] == slot)


def test_paper_spec_model_literal():
    q = 100                                                          # d_i is the token of position q + i
    es = spec_trace(q, 100, 3, 2, [2, 3])                           # Figure 3, K = 3, full accepts
    assert [x["fed_d"] for x in es[:3]] == [[100], [101], [102]] and es[2]["drafted"] == 103   # fill d1, d2, d3
    j0 = _pick(es, 0, "J")
    assert (j0["verify"], j0["deferred"], j0["drafted"], j0["fed_v"]) == ([101, 102], 103, 104, [100, 101])
    assert [_pick(es, 0, s)["drafted"] for s in ("D0", "D1")] == [105, 106]
    j1 = _pick(es, 1, "J")                                           # "d3 is deferred to the next group: d3, d4, d5"
    assert (j1["verify"], j1["deferred"], j1["drafted"]) == ([103, 104, 105], 106, 107)
    es = spec_trace(q, 100, 3, 3, [0, 2])                            # decision (a): a reject at d1
    j0 = _pick(es, 0, "J")
    assert j0["committed"] == 1                                      # the correction at 101; q' = 101
    assert [(x["role"], x["fed_d"], x["drafted"]) for x in (_pick(es, 0, "D0"), _pick(es, 0, "D1"))] == [
        ("draft", [101], 102), ("draft", [102], 103)]                # d1' = 102, d2' = 103
    j1 = _pick(es, 1, "J")
    assert (j1["role"], j1["fed_d"], j1["drafted"]) == ("fill", [103], 104)       # d3' = 104
    assert [_pick(es, 1, s)["role"] for s in ("D0", "D1")] == ["idle", "idle"]
    j2 = _pick(es, 2, "J")                                           # verify d1', d2', defer d3', draft d4'
    assert (j2["role"], j2["verify"], j2["deferred"], j2["drafted"]) == ("group", [102, 103], 104, 105)
    es = spec_trace(q, 4, 3, 3, [2, 0])                              # decision (b): 4 tokens left
    j1 = _pick(es, 1, "J")                                           # left 2 <= K: the final group, 2 of 4 pending
    assert (j1["role"], j1["verify"], j1["committed"]) == ("final", [103, 104], 1)
    j2 = _pick(es, 2, "J")                                           # a reject there: ordinary target rows
    assert (j2["role"], j2["fed_v"], j2["committed"]) == ("target", [103], 1)
    assert sum(x["committed"] for x in es) == 4                     # exactly the requested number


# ------------------------------------------------------------------ SCHEDULE (both runners)
def _trace_errors(ev, B, exact: bool) -> tuple:
    rd = ev["rd"]
    S, n_left = ev["S"], ev["N"] - rd["warm"]
    errs, remade_err = [], []
    for b in range(B):
        act = rd["trace_all"][b]
        outs = [x["n"] for x in act if x["role"] in ("group", "final")]
        want = spec_trace(S + rd["warm"], n_left, K, rd["rounds"], outs)
        got = [{k: x[k] for k in FIELDS} for x in act]
        if got != want:
            i = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            errs.append((b, i, got[i] if i < len(got) else None, want[i] if i < len(want) else None))
        for x in act:                      # EXACT remakes a group's deferred row (q + v) and a fill's earlier positions
            w = []
            if exact and x["role"] == "group":
                w = [x["q"] + len(x["verify"])]
            elif exact and x["role"] == "fill":
                w = list(range(x["q"], x["fed_d"][0]))
            if x["remade"] != w:
                remade_err.append((b, x["round"], x["role"], x["remade"], w))
    return errs, remade_err


def _slot_row_errors(log, trace, B) -> list:
    """FAST: the layer-0 row of every P / D slot (T.rec kind "draft", in slot order; the slots that ran are a prefix,
    any_d) = the trace's fed position; an idle slot writes no row; the row count per request = the fed count."""
    by = {}
    for x in log:
        if x["kind"] == "draft":
            by.setdefault(x["round"], []).append(x)
    errs = []
    for b in range(B):
        n_rows = sum(int(bool(x["act"][b])) for xs in by.values() for x in xs)
        n_fed = 0
        for e in trace[b]:
            if e["slot"] == "J":
                continue
            xs, j = by.get(e["round"], []), int(e["slot"][1:])
            got = [int(xs[j]["p"][b])] if j < len(xs) and bool(xs[j]["act"][b]) else []
            want = e["fed_v"] + e["fed_d"]
            n_fed += len(want)
            if got != want:
                errs.append((e["round"], b, e["slot"], got, want))
        if n_rows != n_fed:
            errs.append(("rows", b, n_rows, n_fed))
    return errs


def _exact_slot_errors(recs, trace, B) -> list:
    """EXACT: the J slot's q = the twin's position at the joint pass (records q); the fed position of every active D
    slot = the twin's position before its step (records dpos, check-only). The prelude is spec_round's _prelude: its
    positions are not recorded."""
    errs = []
    for b in range(B):
        for e in trace[b]:
            g = e["round"]
            if g < 0:
                continue
            r = recs[g]
            if e["slot"] == "J":
                if int(r["q"][b]) != e["q"]:
                    errs.append((g, b, "J", int(r["q"][b]), e["q"]))
                continue
            want = e["fed_v"] + e["fed_d"]
            if not want:
                continue
            dp = r["dpos"][int(e["slot"][1:])]
            got = [int(dp[b])] if dp is not None else []
            if got != want:
                errs.append((g, b, e["slot"], got, want))
    return errs


def _shift_first(xs, pick, field):
    """CONTROL: a copy of xs with one entry's field one position off (the first entry that pick accepts)."""
    i = next(k for k, x in enumerate(xs) if pick(x))
    out = list(xs)
    out[i] = dict(xs[i])
    if field == "dpos":
        j = next(k for k, t in enumerate(xs[i]["dpos"]) if t is not None)
        out[i]["dpos"] = list(xs[i]["dpos"])
        out[i]["dpos"][j] = xs[i]["dpos"][j] + 1
    else:
        out[i][field] = xs[i][field] + 1
    return out


CASES_S = [("fast", 1), ("fast", 8), ("exact", 1), ("exact", 8)]


@pytest.mark.parametrize("mode,B", CASES_S, ids=["FAST-B1", "FAST-B8", "EXACT-B1", "EXACT-B8"])
def test_paper_schedule_trace(mode, B):
    ev = _ev("L4P", B, filt="end", ref=False) if mode == "fast" else _exact("L4P", B, filt="end")
    rd = ev["rd"]
    acct = rd["tokens_acct"]
    assert acct["corrected"] > 0 and acct["fills"] > 0, acct           # preconditions: rejects and refills (a)
    assert acct["finals"] == B and acct["targets"] > 0, acct           # every request's final group; target rows (b)
    assert rd["count"] == [N] * B, rd["count"]                        # THE CLAIM (b): exactly N tokens
    errs, remade = _trace_errors(ev, B, mode == "exact")
    assert errs == [], errs[:3]                                        # THE CLAIM: the specified position trace
    assert remade == [], remade[:3]                                    # FAST remakes nothing (D3); EXACT as described
    tr_all = rd["trace_all"]                                           # the trace = the rows that ran (not the plan)
    if mode == "fast":
        log = ev["runner"].T.rec
        assert _joint_row_errors(log, tr_all, B) == []                 # J slots: the layer-0 row writes
        errs = _slot_row_errors(log, tr_all, B)
        assert errs == [], errs[:3]                                    # P / D slots: the layer-0 row writes
        bad = _shift_first(log, lambda x: x["kind"] == "draft" and x["round"] >= 0 and bool(x["act"].any()), "p")
        assert _slot_row_errors(bad, tr_all, B) != []                  # CONTROL: one row one position off: CAUGHT
    else:
        recs = ev["records"]
        assert any(t is not None for r in recs for t in r["dpos"])      # precondition: D slots ran
        errs = _exact_slot_errors(recs, tr_all, B)
        assert errs == [], errs[:3]                                    # J q and D slots: the twin's positions
        bad = _shift_first(recs, lambda r: any(t is not None for t in r["dpos"]), "dpos")
        assert _exact_slot_errors(bad, tr_all, B) != []                # CONTROL: one slot one position off: CAUGHT
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]                # SH's tokens across rejects, refills, the end
    if mode == "exact":
        assert ev["bad_logits"] == [], ev["bad_logits"][:5]            # the proof: bitwise SH's logits rows
        assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    else:
        assert ev["max_logit_delta"] <= TOL, ev["max_logit_delta"]
        assert ev["resid"] == [] and ev["host"] == []
    for b in range(B):                                                 # after the final group only target / idle
        tr = rd["trace_all"][b]
        k = max(i for i, x in enumerate(tr) if x["role"] == "final")
        assert {x["role"] for x in tr[k + 1:]} <= {"target", "idle"}, (b, tr[k + 1:][:3])
        assert sum(x["committed"] for x in tr) == N - rd["warm"], b


# ------------------------------------------------------------------ TOKENS, SELECTIONS, LOGITS against SH
CASES = [("L4P", 1), ("L4P", 8), ("L3", 1), ("L3", 8)]


@pytest.mark.parametrize("level,B", CASES, ids=["L4P-B1", "L4P-B8", "L3-B1", "L3-B8"])
def test_tent_commits_sh_tokens(level, B):
    ev = _ev(level, B)
    f = ev["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f                      # preconditions (L34): both paths of a round ran
    assert f["crossed"] > 0 and f["edge_rows"] > 0 and f["fill_rows"] > 0, f    # a commit across an edge, rows past
    assert f["skipped"] > 0 and f["forced"] > 0, f                    # an edge, F4 fill rows; draft and forced skips
    assert f["move_out"] + f["swap"] > 0, f                           # the victim pool moved blocks
    if B > 1:
        assert f["ragged"], f
    assert f["min_count"] >= N, f
    if level == "L4P":
        assert ev["rd"]["count"] == [N] * B
    assert ev["ref_logits_bad"] == [], ev["ref_logits_bad"][:5]        # the reference IS SH (bitwise its logits)
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]                # THE CLAIM: SH's tokens
    print("WS-A %s B %d: max |logit - SH's| over %d committed rows = %.3e (TOL %.0e); selections compared %d" % (
        level, B, ev["rows_compared"], ev["max_logit_delta"], TOL, ev["sel_compared"]))
    assert ev["max_logit_delta"] <= TOL, ev["max_logit_delta"]         # the logit deltas against SH
    assert ev["sel_compared"] > 0
    assert ev["sel_mismatch"] == [], ev["sel_mismatch"][:5]            # SH's selections on the same committed prefix
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])
    assert ev["rd"]["errors"] == (0, [0] * L), ev["rd"]["errors"]
    ex = _exact(level, B)                                              # FAST commits EXACT's tokens
    assert ex["bad_tokens"] == []
    ce = ex["rd"]["count"]
    for b in range(B):
        c = min(int(ev["cnt"][b]), int(ce[b]), N)
        assert c >= N and torch.equal(ev["runner"].out[b, :c].cpu(), ex["runner"].out[b, :c].cpu()), b


# ------------------------------------------------------------------ STATE
@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_tent_committed_state(B):
    ev = _ev("L4P", B)
    run = ev["runner"]
    assert ev["facts"]["rejects"] > 0 and ev["facts"]["crossed"] > 0, ev["facts"]   # preconditions
    run.T.reset()                                    # the twins to the committed counters (the chain is not committed)
    qc = run.T.qc.clone()
    assert torch.equal(qc.cpu(), ev["S"] + ev["cnt"])                   # the committed position = S + the count
    sc = WW.state_classes(run.rs, ev["ref_rs"], qc, ev["S"])
    print("WS-A STATE B %d: EXACT differ %s; layer-0 not bitwise %s; beyond TOL %s; max |delta| %.3e; CACHE (rows "
          "that differ, not semantic) %s" % (B, sc["exact"], sc["l0"], sc["close"], sc["maxabs"], sc["cache"]))
    assert sc["exact"] == [], sc["exact"]
    assert sc["l0"] == [], sc["l0"]
    assert sc["close"] == [], sc["close"][:5]


# ------------------------------------------------------------------ DEFER (D3)
def _joint_row_errors(log, trace, B) -> list:
    """The layer-0 row writes of each J slot = the trace's fed positions (FAST: never the deferred row's)."""
    errs = []
    for b in range(B):
        for x in trace[b]:
            if x["slot"] != "J":
                continue
            got = sorted(int(r["p"][b]) for r in log if r["kind"] == "joint" and r["round"] == x["round"]
                         and bool(r["act"][b]))
            want = sorted(x["fed_v"] + x["fed_d"])
            if got != want:
                errs.append((x["round"], b, got, want))
    return errs


def _replay_errors(log, B):
    """Every replay used the newest buffer write of its position, a draft row's, unchanged."""
    last, errs, seen = {}, [], 0
    for x in log:
        if x["kind"] != "replay":
            for b in range(B):
                if bool(x["act"][b]):
                    last[(b, int(x["p"][b]))] = (bool(x["draft"][b]), x["k"][b], x["c"][b])
            continue
        for b in range(B):
            if not bool(x["act"][b]):
                continue
            w = last.get((b, int(x["p"][b])))
            if w is None or not w[0] or not (SW.bits_equal(w[1], x["k"][b]) and SW.bits_equal(w[2], x["c"][b])):
                errs.append((x["round"], b, int(x["p"][b]), None if w is None else w[0]))
            else:
                seen += 1
    return errs, seen


def _pos_errors(log):
    return [(x["round"], x["kind"], x["r"]) for x in log if x["kind"] != "replay"
            and not torch.equal(x["p"][x["act"]], x["rope"][x["act"]])]


def _groups(ev) -> int:
    return sum(int((r["role"] == ST.GROUP).sum()) for r in ev["records"])


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_tent_deferred_not_recomputed(B):
    ev = _ev("L4P", B)
    tr, log = ev["rd"]["tent"], ev["runner"].T.rec
    c = tr["calls"]
    assert tr["deferred_writes"] == 0, tr["deferred_writes"]             # the D3 counter (the cell's MUST gate)
    assert c["joint_rows"] == (K + 1) * c["joint_passes"] and c["joint_passes"] > 0, c
    assert c["joint_passes"] + c["j1_passes"] == ev["rd"]["rounds"], c
    assert tr["replays"] == _groups(ev) > 0, (tr["replays"], _groups(ev))    # one replay per group
    assert ev["rd"]["tokens_acct"]["remade_rows"] == 0
    errs = _joint_row_errors(log, ev["rd"]["trace_all"], B)
    assert errs == [], errs[:5]                                           # v verify rows + the draft row, none at q + v
    errs, seen = _replay_errors(log, B)
    assert errs == [] and seen == _groups(ev), (errs[:5], seen)
    assert _pos_errors(log) == []                                         # twin position = rope position, every row
    p = _ev("L4P", B, tneg="recompute_deferred", ref=False)              # PLANT: EXACT's K + 2 rows
    pt = p["rd"]["tent"]
    assert pt["deferred_writes"] > 0 and pt["calls"]["joint_rows"] == (K + 2) * pt["calls"]["joint_passes"], pt
    assert p["bad_tokens"] == []                                          # a COST plant: the tokens stay SH's
    q = _ev("L4P", B, tneg="no_replay", ref=False)                       # PLANT: no replay
    assert q["rd"]["tent"]["replays"] == 0 and _pos_errors(q["runner"].T.rec) != []      # CAUGHT


# ------------------------------------------------------------------ HOST (reject at a block edge)
def _uncommitted_fills(ev) -> list:
    """(round, request): the chain passed the end of the committed block, the commit did not."""
    out = []
    for g, r in enumerate(ev["records"]):
        for b in range(ev["B"]):
            q, m, fr = int(r["q"][b]), int(r["m"][b]), int(r["frontier"][b])
            if fr // BS > q // BS and (q + m) // BS == q // BS:
                out.append((g, b))
    return out


def _with_fill(B, **kw):
    """The first prompt length in S0 .. S0 + 3 whose run has an uncommitted fill (the precondition of the claim)."""
    for S in range(WW.S0, WW.S0 + 4):
        ev = _ev("L4P", B, S=S, **kw)
        if _uncommitted_fills(ev):
            return ev
    return None


def _host_vs_shipped(ev) -> list:
    errs = []
    rs, S = ev["runner"].rs, ev["S"]
    for l, lay in enumerate(rs.cache.layers):
        e = lay.cache_engine
        rk, rv = ev["ref_host"][l]
        for b in range(ev["B"]):
            a0, z = (S // BS) * BS, ((S + int(ev["cnt"][b])) // BS) * BS
            if z <= a0:
                continue
            for name, x, y in (("k", e._k_cpu[b, a0:z], rk[b, a0:z]), ("v", e._v_cpu[b, a0:z], rv[b, a0:z])):
                if l == 0 and not SW.bits_equal(x, y):
                    errs.append(("layer0", name, b))
                if float((x.double() - y.double()).abs().max()) > TOL:
                    errs.append(("tol", name, l, b))
    return errs


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_tent_reject_keeps_host_rows(B):
    ev = _with_fill(B)
    assert ev is not None, "precondition: no run in S0 .. S0 + 3 with a chain fill that its commit did not keep"
    assert ev["host"] == [] and ev["resid"] == [], (ev["host"][:2], ev["resid"][:2])
    assert ev["facts"]["crossed"] > 0                                     # a committed fill reached the host
    assert _host_vs_shipped(ev) == []                                     # SH's host rows (layer 0 bitwise, TOL)
    p = _ev("L4P", B, S=ev["S"], tneg="writeback_tentative", ref=False)  # PLANT: the chain's fill is written back
    assert _uncommitted_fills(p), "precondition: the plant run has an uncommitted fill"
    assert p["host"] != []                                                # CAUGHT


# ------------------------------------------------------------------ COST (fixed, O(1) per rejected token)
def _engine_snapshot(rs) -> dict:
    out = {}
    for l, lay in enumerate(rs.cache.layers):
        e = lay.cache_engine
        for n in ("_block_map", "_pool_map", "_pool_age", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu",
                  "_cache_lens"):
            out["L%d.%s" % (l, n)] = getattr(e, n).clone()
        out["L%d.clock" % l] = e._pool_stamp_base
    return out


def _snap_diff(a, b) -> list:
    return [n for n in a if not RW.same(a[n], b[n])]


def _cost_run(filt, tneg=None):
    box, mode = {}, WW.OpCount()
    facts = dict(changed=0, reset_wrote=[], last=None)

    def on_runner(runner):
        facts["runner"] = runner
        WW.one_launch(runner.rs.body, mode)
        T = runner.T
        orig = T.reset

        def reset(mask=None):
            before = _engine_snapshot(runner.rs)
            if facts["last"] is not None and any(n.endswith(("_block_map", "_pool_map"))
                                                 for n in _snap_diff(facts["last"], before)):
                facts["changed"] += 1
            n0 = mode.n
            with mode:
                orig(mask)
            box.setdefault("reset", []).append(mode.n - n0)
            after = _engine_snapshot(runner.rs)
            d = _snap_diff(before, after)
            if d:
                facts["reset_wrote"].append((runner.round, d[:4]))
            facts["last"] = after
        T.reset = reset
        WW.count_calls(T, "commit", box, mode)
    try:
        ev = WW.run_tent(8, level="L4P", draft_filter=filt, tneg=tneg, with_reference=False, on_runner=on_runner,
                         max_rounds=(12 if tneg else 0))
    except Exception as e:           # a plant may break a stand-in's precondition AFTER its reset was recorded
        if tneg is None:
            raise
        ev = dict(error=repr(e))
    return ev, box, facts


def _dropped_recount(ev) -> int:
    """The dropped entries counted from the buffer WRITES (T.rec, layer 0), not from the runner's formula: per request
    the written positions not yet committed; at a round's commit (after its J slot, before its D slots) a group's reject
    or a final drops every written position at or past the committed position q_end; otherwise the positions below
    q_end leave (committed) and the others stay."""
    B = ev["B"]
    by = {}
    for x in ev["runner"].T.rec:
        if x["kind"] != "replay":
            by.setdefault((x["round"], x["kind"] == "joint"), []).append(x)

    def add(live, xs):
        for x in xs:
            for b in range(B):
                if bool(x["act"][b]):
                    live[b].add(int(x["p"][b]))
    live = [set() for _ in range(B)]
    n = 0
    for g in range(-1, len(ev["records"])):
        add(live, by.get((g, True), []))                               # the J slot's rows
        if g >= 0:
            r = ev["records"][g]
            for b in range(B):
                qe, role = int(r["q_end"][b]), int(r["role"][b])
                ge = {p for p in live[b] if p >= qe}
                if role == ST.FINAL or (role == ST.GROUP and not bool(r["full"][b])):
                    n += len(ge)
                    ge = set()
                live[b] = ge
        add(live, by.get((g, False), []))                              # the D slots' rows (g = -1: the prelude)
    return n


def test_tent_rollback_is_fixed_cost():
    runs = [_cost_run(FILTER), _cost_run(FILTER0)]
    resets = {n for _, box, _ in runs for n in box["reset"]}
    commits = {n for _, box, _ in runs for n in box["commit"]}
    drops = [ev["rd"]["tent"]["dropped"] for ev, _, _ in runs]
    print("WS-A COST: torch calls per reset %s, per commit %s (the device write-back = 1); dropped entries %s" % (
        sorted(resets), sorted(commits), drops))
    assert drops[0] != drops[1], drops                                    # precondition: different reject counts
    assert len(resets) == 1 and len(commits) == 1, (resets, commits)      # THE CLAIM: a fixed cost per call
    for ev, box, facts in runs:
        assert facts["changed"] > 0, facts["changed"]                    # precondition: residency changed between resets
        assert facts["reset_wrote"] == [], facts["reset_wrote"][:3]      # a reset writes no engine table
        assert ev["rd"]["tent"]["dropped"] == _dropped_recount(ev)       # the drop is a counter, counted per token
        assert ev["rd"]["tent"]["calls"]["residency_undo"] == 0
        assert ev["bad_tokens"] == []
    ev, box, facts = _cost_run(FILTER, tneg="undo_residency")             # PLANT: a residency undo
    assert facts["runner"].T.calls["residency_undo"] > 0                  # precondition: the plant ran
    assert facts["reset_wrote"] != []                                     # CAUGHT


# ------------------------------------------------------------------ LIFETIME
@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_tent_buffer_lifetime(B):
    ev = _ev("L4P", B, poison=True, ref=False)
    assert any(bool((t == ST.POISON_V).any()) for t in ev["runner"].T.v)   # precondition: the poison acted
    assert ev["bad_tokens"] == [] and ev["max_logit_delta"] <= TOL, (ev["bad_tokens"][:3], ev["max_logit_delta"])
    for tneg in ("no_causal", "no_lower"):
        p = _ev("L4P", B, poison=True, tneg=tneg, ref=False)
        assert p["bad_tokens"] != [] or p["max_logit_delta"] > TOL, tneg     # CAUGHT


# ------------------------------------------------------------------ WIRING
def _small(level, B=2, hooks=None, check_every=1, record=True, N_=8):
    S = WW.S0
    hz = SW.horizon_for(N_, K)
    model, _, c = SW.two_caches(B, S, hz + 64, 0)
    tok0 = torch.randint(0, WW.V, (B,), generator=torch.Generator().manual_seed(2))
    return WW.tent_runner(model, c, S, hz, tok0, level=level, hooks=hooks, check_every=check_every, record=record)


def test_tent_refusals_hooks_check_rounds():
    model, rs = RW.build(S1)
    ops = lambda **k: SR.SpecOps(**dict(dict(level="L4P", K=K, attn=VW.fake_attention_idx), **k))   # noqa: E731
    for bad, kw in ((dict(level="L4", K=K), {}), (dict(prefetch=SW.Recorder()), {}), (dict(pin=True), {}),
                    (dict(level="L3", K=32), {}), (dict(level="L3", neg="keep_rejected"), {}),
                    ({}, dict(tneg="nope")), (dict(level="L3"), dict(tneg="recompute_deferred")),
                    ({}, dict(check_every=-1))):
        with pytest.raises(SR.SpecError):
            ST.TentRunner(model, rs, ops(**bad), BC.HostClock(), **kw)
    for bad in (dict(level="L3"), dict(prefetch=SW.Recorder()), dict(pin=True)):
        with pytest.raises(SR.SpecError):
            ST.PaperJournalRunner(model, rs, ops(**bad), BC.HostClock())
    assert rs.ops.kernels["gather_k"] is W.KERNELS["gather_k"]           # every refusal before a hook attached
    rec = WW.Recorder()                                                   # check_every 0: no hook, no fingerprint
    run = _small("L4P", hooks=rec, check_every=0)
    run.run(8)
    assert rec.sel == [] and rec.logits == []
    assert int(run.fp_ok.sum()) == run.B * run.warm_steps
    rec = WW.Recorder()                                                   # check_every 2: the rounds 2, 4, ... only
    run = _small("L4P", hooks=rec, check_every=2, N_=24)
    run.run(24)
    rounds = sorted({x["round"] for x in rec.logits})
    assert rounds and all(g > 0 and g % 2 == 0 for g in rounds), rounds
    x = next(y for y in rec.logits if y["logits"].shape[1] > 1)
    assert tuple(x["logits"].shape[:2]) == (run.B, K + 1) and tuple(x["verify"].shape) == (K + 1, run.B)
    ev = _ev("L4P", 8)                                                    # check_every 1: on_select, every layer
    s = ev["rec"].sel[0]
    assert tuple(s["sel"].shape) == (WW.HKV, 8, WW.TOPK) and {x["l"] for x in ev["rec"].sel} == set(range(L))


@pytest.mark.parametrize("level", ["L4P", "L3"])
def test_tent_no_host_sync_in_round(level):
    run = _small(level, record=False, check_every=0)
    with RW.SyncTrap(ST_PATH) as trap:
        run.run(8, rounds=3)
    fns = {h[0] for h in trap.hits}
    allowed = {"_run_paper", "_ps_after"} if level == "L4P" else {"_run_l3"}
    assert fns and fns <= allowed, trap.hits[:10]                         # the per-round sync only (and not blind)


# ------------------------------------------------------------------ CELL
def _res_of(ev, exact: bool):
    run = ev["runner"]
    return dict(read=ev["rd"], out=run.out, fp=(run.fmax, run.fsum), fp_ok=(None if exact else run.fp_ok), K=K,
                probe_every=16, prefill_s=0.0, first_token_agreement=ev["B"], meta={},
                undo=(dict(undo_steps=1, commits=1, truncates=1, rejects=1) if exact else None),
                mem=dict(start_GB=16.0, prefill_peak_GB=82.0, decode_peak_GB=80.0))


def test_wsa_cell_arguments_and_report():
    C = _cell()
    base = ["--out", "x", "--workload", "w", "--spec", "3"]
    a = C.parse_args(base + ["--spec-arms", "L4P,L4PJ,L4PT", "--spec-rollback", "tentative", "--check-every", "1",
                             "--require-stress", "1"])
    assert (a.spec_arms, a.spec_rollback, a.check_every, a.require_stress) == (["L4P", "L4PJ", "L4PT"], "tentative", 1,
                                                                               1)
    for bad in (["--spec-arms", "L4PT"],                                   # a tentative arm needs the flag
                ["--spec-rollback", "tentative"],                          # the flag needs a tentative arm
                ["--spec-arms", "L4X"], ["--check-every", "-1"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    with pytest.raises(SystemExit):
        C.parse_args(["--out", "x", "--workload", "w", "--spec", "2"])    # L4P needs K >= 3
    assert C.main(["--out", "x", "--workload", "w"]) == 20                # --spec 0: refused (the flag of the cell)
    assert C.fit_batch(216, 16.0, 82.0) == int(216 * 68 / 66) and C.fit_batch(8, None, 1.0) is None
    B, steps = 8, N
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    dims = dict(L=L, H=WW.HKV, D=W.D, topk=WW.TOPK, block_bytes=64 * W.D * 4 * 2 + 64 * 4)
    recs = {}
    for arm, ev, exact in (("L4PT", _ev("L4P", 8), False), ("L4PJ", _exact("L4P", 8), True)):
        res = _res_of(ev, exact)
        inputs = ev["ref_tok"][:, :steps + 1]
        fp_sh = (ev["ref_lg"][:, :steps].amax(-1), ev["ref_lg"][:, :steps].sum(-1, dtype=torch.float64))
        rec = C.summarize_w(arm, res, inputs, fp_sh, sh, B, steps, dims)
        recs[arm] = rec
        assert rec["tokens"]["n_unequal"] == 0 and rec["tokens"]["n_checked"] == B * steps, rec["tokens"]
        assert rec["schedule_record"]["tokens_acct"]["committed"] == B * (steps - 1)
        assert len(rec["schedule_record"]["trace_head"]) == B
        g = C.arm_gates_w(arm, res, rec, steps, require_stress=not exact)
        names = {x[0]: x for x in g}
        want = {"%s-ERR" % arm, "%s-COMMIT-N" % arm} | ({"L4PJ-TOK", "L4PJ-FP"} if exact else
                                                         {"L4PT-D1", "L4PT-D3", "L4PT-SPAN", "L4PT-STRESS"})
        assert want <= set(names) and "%s-COMMIT" % arm not in names, sorted(names)
        assert all(x[2] for x in g), [x for x in g if not x[2]]
    t = recs["L4PT"]
    assert t["fingerprints"]["n"] > 0 and t["fingerprints"]["max_abs_dmax"] <= TOL
    assert t["calls"]["deferred_recompute_rows"] == 0 and t["calls"]["residency_undo_calls"] == 0
    assert t["calls"]["remade_rows"] == 0 and recs["L4PJ"]["calls"]["remade_rows"] > 0
    assert t["rollback_per_round"]["phases"] == ["reset", "commit"]
    assert recs["L4PJ"]["rollback_per_round"]["phases"] == ["truncate", "commit"]
    assert t["bytes_per_round"]["published"] > 0 and t["bytes_per_round"]["residency_restore"] == 0
    assert t["memory"]["fit_batch_decode_MODEL"] == C.fit_batch(B, 16.0, 80.0)
    res = _res_of(_ev("L4P", 8), False)
    bad = dict(res, read=dict(res["read"], tent=dict(res["read"]["tent"], deferred_writes=1)))
    assert not {x[0]: x for x in C.arm_gates_w("L4PT", bad, t, steps)}["L4PT-D3"][2]  # the gate is not blind
    bad = dict(res, read=dict(res["read"], count=[steps + 1] + res["read"]["count"][1:]))
    assert not {x[0]: x for x in C.arm_gates_w("L4PT", bad, t, steps)}["L4PT-COMMIT-N"][2]
    p = C.verdicts(sh, recs, B)
    assert all(v in ("REPORT", "NOT_RUN") for _, v, _ in p), p   # the flat PRED_WSA is None and B 8 has no entry
    assert set(C.PRED_WSA_BY_B) == {216, 318} and all(set(b) < set(C.PRED_WSA) for b in C.PRED_WSA_BY_B.values())
    old = C.PRED_WSA_BY_B.get(B)
    try:
        C.PRED_WSA_BY_B[B] = dict(L4PJ_tok=False)
        p = {n: v for n, v, _ in C.verdicts(sh, recs, B)}
        assert p["L4PJ-TOK"] == "REFUTED", p                         # the per-batch entry overrides the flat None
    finally:
        if old is None:
            del C.PRED_WSA_BY_B[B]
        else:
            C.PRED_WSA_BY_B[B] = old


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_wsa_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_wa1():
    for bad in ("sp3m", "sp5a", "wa1c", "mb1"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_WSA = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_wsa_cell.py" % sha).stdout
    for cell, pool, batch, steps, journal, check, stress, smin, mem in (
            ("wa1m", 10, 8, 64, 8, 1, 1, 30, "100G"), ("wa1a", 65, 216, 512, 256, 0, 0, 75, "240G"),
            ("wa1b", 10, 318, 512, 256, 0, 0, 110, "340G")):
        r = _step(cell, sha)
        if cell != "wa1m" and not registered:                         # the big cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_WSA is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L4P,L4PJ,L4PT steps %d K 3 journal %d check %d stress %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, journal, check, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--spec 3",
                     "--spec-rollback tentative", "--spec-journal-blocks %d" % journal, "--check-every %d" % check,
                     "--require-stress %d" % stress, "--spec-twin fused --spec-body fused",
                     STEP_ROOT + "/benchmarks/Efficiency/r2_wsa_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "wa_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
