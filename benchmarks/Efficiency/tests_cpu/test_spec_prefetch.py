"""CPU tests of rung L5 of Phase 2 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs): THE BACKGROUND PREFETCH of the draft misses with the soft pin (nosi/nosi/spec_prefetch.py), wired into
the speculative round runner (nosi/nosi/spec_round.py, levels L3 and L4P), the cell r2_spec5_cell.py and the step script
r2_spec5_step.sh. The world: tests_cpu/spec5_world.py over spec_world.py (the SHIPPED engine from its own source, every
CUDA op a torch stand-in with the kernel's flat addresses, L28). B 1 and B 8. No CUDA, no Triton launch. K = 3.

GATE 0   test_flags_off_is_specp (B 1 / B 8): the tree is nosi-r2-specp a6c9d10 except the declared rung-L5 files; the
         four older Gate 0 tests only add spec_prefetch.py to their exclude lists; spec_round.py is a6c9d10's AST plus
         the guarded seams (docstrings aside); no engine module but spec_round.py names spec_prefetch; with
         SpecOps(prefetch=None) the round runner of THIS tree and the a6c9d10 source give the same tokens, logits rows,
         counters, round records and final state, bit for bit (L3 and L4P).
TOKENS   test_prefetch_tokens_equal_shipped (L3 / L4P at B 1 / B 8, L4PP = L4P + the requested-KV pin at B 8): every
         committed token is the SHIPPED greedy decode and every committed logits row is the shipped step's, bitwise;
         the residency (every named window / pool slot holds its host rows and cis rows: the prefetched blocks too) and
         the host rows hold after every round; no soft pin stays after a round; pin conservation. Preconditions: the
         prefetch acted (jobs, prefetched verify hits), rejects and full accepts, a block edge, draft skips, pool moves,
         (B 8) ragged positions.
OVERFLOW test_prefetch_tiny_journal_equals_shipped (L3 / L4P, B 8, a journal of 1 slot per layer and step): the same
         claims. Preconditions: journal overflow (live and dead), prefetched hits, and a prefetched hit in a verify step
         / row that the round UNDID (the undo moved a prefetched block back into its pinned slot).
HIT      test_prefetched_block_is_a_pool_hit (L3 / L4P, B 8): in round 0 (the same history in both runs) every block
         that the verify found as a prefetched pool hit is a HOST LOAD of the same verify call in the run without the
         prefetch (the control); the prefetch delivered it into a pool slot with its host bytes (the residency check).
SPLIT    test_wait_split_adds_up (L3 / L4P, ready / in flight): an independent recount of every verify-selected slot from
         the residency itself (window W, pool P, prefetched F0 / F1, new N) equals the device split: sel = READY_resident
         + READY_prefetched + IN_FLIGHT + NEW; NEW = the load mask's host loads; ready transport: no IN FLIGHT; held
         transport: every prefetched hit IN FLIGHT and one commit wait per layer in each committing round.
PIN      test_dropped_pin_is_caught (L3, B 8): every tagged slot carries a prefetch pin age before each verify call, and no
         prefetched block is evicted while an unpinned unclaimed slot exists (evict_free = 0; the cell's PF-PIN gate).
         PLANT no_pin (the slot keeps the victim's age): CAUGHT by both (preconditions: tagged slots, evictions of them).
GROUP    test_commit_inside_group_is_caught (L3, B 8): on_verify / on_release refuse inside a group. PLANT
         commit_mid_group (the epoch's commit inside the verify group, at step K - 1): the undo of step 1 names a pool
         slot with the bytes of another block: CAUGHT by the residency or token check (preconditions: named victims, a
         request that undid step 1). The clean run with the same drafts is exact.
WIRING   test_seams_and_refusals, test_host_syncs_are_declared (the prefetch module reads host values only in
         _issue_snap; the round runner's syncs are a6c9d10's).
PACKER   test_real_packer_tokens_equal_shipped (L3 / L4P, B 8): the REAL packer v3 (C++ worker, issue_desc, flat lanes) in
         CPU mode, built here into a pytest temporary directory: the same token, residency and conservation claims.
CELL     test_cell5_arguments_and_report, test_step_script_sp5.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py
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
SPECP = "a6c9d10500aa3070ad6c0fb781e7c05e1438325b"          # nosi-r2-specp: the tree of rung L4P with the undo fix
# WS-B (nosi-r2-wsb): parts (a) and (b) of Gate 0 read the rung-L5 tip 40ac348 (the change of rung L5); the WS-B change
# on top of it is pinned by tests_cpu/test_wsb.py (test_gate0_files, test_gate0_*_unchanged)
SPEC5_TIP = "40ac348870e89b43ac5278458b660dea8d360508"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-spec5"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
DECLARED = ("nosi/nosi/spec_round.py", "nosi/nosi/spec_prefetch.py", "benchmarks/Efficiency/r2_spec5_cell.py",
            "benchmarks/Efficiency/r2_spec5_step.sh", "benchmarks/Efficiency/tests_cpu/spec_world.py",
            "benchmarks/Efficiency/tests_cpu/spec5_world.py", "benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py",
            "docs/NOTES_spec5.md")
GATE0_TESTS = ("benchmarks/Efficiency/tests_cpu/test_ragged_engine.py", "benchmarks/Efficiency/tests_cpu/test_verify_rows.py",
               "benchmarks/Efficiency/tests_cpu/test_rollback.py", "benchmarks/Efficiency/tests_cpu/test_spec_round.py")


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SW5 = _load("spec5_world")
SW = SW5.SW
W, RW, VW, RE, RR, SR, BC, SP = SW.W, SW.RW, SW.VW, SW.RE, SW.RR, SW.SR, SW.BC, SW5.SP
K, N, L = SW.K0, SW.N0, SW.L
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)       # as test_spec_round.py: draft 1 wrong in some rounds
FILTER0 = SW.corrupt(4, 0, kinds=("draft",), phase=2)      # draft 0 wrong: m = 1, the verify steps 1 .. K - 1 undone
LANES = 16
CAP_TINY = 1


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have_specp() -> bool:
    return _git("rev-parse", "-q", "--verify", SPECP + "^{commit}").returncode == 0


@functools.lru_cache(maxsize=None)
def _ev(level, B, hold=False, lanes=LANES, neg=None, journal=None, pin=False, filt="periodic", prefetch=True, seed=0):
    """One run per parameter set (shared by the claims of this module)."""
    f = dict(periodic=FILTER, zero=FILTER0, none=None)[filt]
    return SW5.run_pf(B, level=level, prefetch=prefetch, hold=hold, lanes=lanes, neg=neg, journal=journal, pin=pin,
                      filt=f, seed=seed, max_rounds=(12 if neg else 0))


def _clean(ev):
    return ev["bad_tokens"] == [] and ev["bad_logits"] == []


def _tot(ev):
    return ev["rep"]["spec"]["totals"]


def _hits(ev):
    t = _tot(ev)
    return t["read_ready"] + t["read_inflight"]


# ------------------------------------------------------------------ GATE 0
class _Strip(ast.NodeTransformer):
    """Docstrings out; the guarded seams of rung L5 out (an `if` without `else` whose test names prefetch and hasattr)."""

    def _body(self, body):
        out = []
        for k, n in enumerate(body):
            if k == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,
                                                                                                         str):
                continue
            if isinstance(n, ast.If) and not n.orelse and "prefetch" in ast.unparse(n.test) and \
                    "hasattr" in ast.unparse(n.test):
                continue
            out.append(n)
        return out

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            node.body = self._body(node.body)
        return node


def _stripped(src: str) -> str:
    return ast.dump(_Strip().visit(ast.parse(src)))


def _old_spec_round(tmp_path):
    """The a6c9d10 source of spec_round.py as a module of the synthetic package (its relative imports resolve to the
    same patched ragged_engine / ragged_rollback / verify_rows)."""
    name = W.PKG + ".spec_round_a6c9d10"
    if name in sys.modules:
        return sys.modules[name]
    r = _git("show", "%s:nosi/nosi/spec_round.py" % SPECP)
    assert r.returncode == 0, r.stderr
    p = tmp_path / "spec_round_a6c9d10.py"
    p.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TIMING = ("wall_s", "phases", "round_ms", "probe_wait_ms", "prelude_ms")


def _same_records(a, b) -> list:
    bad = []
    if len(a) != len(b):
        return ["record count %d != %d" % (len(a), len(b))]
    for g, (x, y) in enumerate(zip(a, b)):
        for k in sorted(set(x) | set(y)):
            u, v = x.get(k), y.get(k)
            if torch.is_tensor(u) or torch.is_tensor(v):
                if not (torch.is_tensor(u) and torch.is_tensor(v) and torch.equal(W.bits(u), W.bits(v))):
                    bad.append((g, k))
            elif u != v:
                bad.append((g, k))
    return bad


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_flags_off_is_specp(B, tmp_path):
    if not _have_specp() or _git("rev-parse", "-q", "--verify", SPEC5_TIP + "^{commit}").returncode != 0:
        pytest.skip("no git or no commit %s / %s in %s" % (SPECP, SPEC5_TIP, ROOT))
    # (a) the rung-L5 tip = a6c9d10 except the declared files of rung L5
    r = _git("diff", "--name-only", SPECP, SPEC5_TIP, "--", ".", *[":(exclude)" + f for f in DECLARED + GATE0_TESTS],
             ":(exclude,glob)**/__pycache__/**")
    assert r.returncode == 0 and r.stdout.strip() == "", "files differ from %s:\n%s" % (SPECP, r.stdout)
    for f in GATE0_TESTS:                                     # their exclude lists only: one added line each
        d = _git("diff", "-U0", SPECP, SPEC5_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert body and all(x.startswith("+") and "spec_prefetch" in x for x in body), (f, body[:5])
    # (b) spec_round.py of the rung-L5 tip = a6c9d10's AST + the guarded seams
    old = _git("show", "%s:nosi/nosi/spec_round.py" % SPECP).stdout
    assert _stripped(old) == _stripped(_git("show", "%s:nosi/nosi/spec_round.py" % SPEC5_TIP).stdout)
    # (c) no engine module but spec_round.py names spec_prefetch; no older cell does
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    assert [n for n, t in src.items() if "spec_prefetch" in t and n not in ("spec_round.py", "spec_prefetch.py")] == []
    for n in ("spec_round", "ragged_rollback", "state_snapshot"):  # the older Gate 0 name scans stay true
        assert n not in src["spec_prefetch.py"], n
    for f in ("r2_spec_cell.py", "r2_base_cell.py", "r2_poolpf_cell.py", "r2_spec_step.sh"):
        t = (EFF / f).read_text()                             # (r2_spec_cell.py has the argument a.spec_prefetch)
        assert "SpecPrefetch" not in t and not re.search(r"(import|from)\s+[\w.]*\bspec_prefetch\b", t), f
    # (d) prefetch None: this tree's round runner = the a6c9d10 source, bit for bit
    OLD = _old_spec_round(tmp_path)
    for level in ("L3", "L4P"):
        evs = [SW.run_vs_shipped(B, level=level, mode="exact", gemm="rowgroup", draft_filter=FILTER, sr=m)
               for m in (None, OLD)]
        new, old_ = evs
        assert _clean(new) and _clean(old_), (level, new["bad_tokens"][:3], old_["bad_tokens"][:3])
        rn, ro = new["runner"], old_["runner"]
        for x, y in ((rn.out, ro.out), (rn.fmax, ro.fmax), (rn.fsum, ro.fsum), (rn.lg_out, ro.lg_out)):
            assert torch.equal(W.bits(x), W.bits(y)), level
        dn = {k: v for k, v in new["rd"].items() if k not in TIMING}
        do = {k: v for k, v in old_["rd"].items() if k not in TIMING}
        assert dn == do, (level, [k for k in dn if dn[k] != do.get(k)])
        assert _same_records(new["records"], old_["records"]) == [], level
        assert RW.diff(RW.state(rn.rs), RW.state(ro.rs)) == [], level
        assert new["facts"]["rejects"] > 0 and new["facts"]["full"] > 0, new["facts"]   # both paths ran (L34)


# ------------------------------------------------------------------ TOKENS
@pytest.mark.parametrize("level,B,pin", [("L3", 1, False), ("L3", 8, False), ("L4P", 1, False), ("L4P", 8, False),
                                         ("L4P", 8, True)], ids=["L3-B1", "L3-B8", "L4P-B1", "L4P-B8", "L4PP-B8"])
def test_prefetch_tokens_equal_shipped(level, B, pin):
    ev = _ev(level, B, pin=pin)
    f, rep = ev["facts"], ev["rep"]
    assert rep["stats"]["jobs"] > 0 and _hits(ev) > 0, (rep["stats"], _tot(ev))   # precondition: the prefetch acted
    assert f["rejects"] > 0 and f["full"] > 0, f
    assert f["edge_crossed"] and f["skipped"] > 0, f
    assert f["move_out"] + f["swap"] > 0, f
    if B > 1:
        assert f["ragged"], f
    assert f["min_count"] >= N, f
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]              # THE CLAIM: the shipped greedy tokens
    assert ev["bad_logits"] == [], ev["bad_logits"][:5]              # and the shipped logits rows
    assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    assert ev["pinned"] == [], ev["pinned"][:3]                      # no soft pin (prefetch or request) after a round
    assert ev["rd"]["errors"] == (0, [0] * L), ev["rd"]["errors"]
    cons = rep["spec"]["conservation"]
    assert cons["conserved"], cons                                   # PIN CONSERVATION
    assert _tot(ev)["evict_free"] == 0, _tot(ev)
    if pin:
        assert ev["rd"]["pin"]["applied"] > 0, ev["rd"]["pin"]       # the requested-KV pin acted too (L4PP)


# ------------------------------------------------------------------ OVERFLOW
@functools.lru_cache(maxsize=None)
def _tiny(level, seeds=3):
    """The first world seed (0 .. seeds - 1) with an undone prefetched hit, and its evidence (the last seed's if none)."""
    for seed in range(seeds):
        ev = _ev(level, 8, journal=CAP_TINY, filt=("zero" if level == "L3" else "periodic"), seed=seed)
        if ev["recount"].undone_pf_hits(ev["records"]) > 0:
            break
    return seed, ev


@pytest.mark.parametrize("level", ["L3", "L4P"])
def test_prefetch_tiny_journal_equals_shipped(level):
    seed, ev = _tiny(level)
    st = ev["rd"]["journal"]
    assert sum(st["overflow"]) > 0 and sum(st["overflow_dead"]) > 0, st   # precondition: the journal overflows
    assert _hits(ev) > 0, _tot(ev)                                    # precondition: prefetched verify hits
    assert ev["recount"].undone_pf_hits(ev["records"]) > 0, (seed, ev["recount"].tot)   # precondition: an undone one
    assert _clean(ev), (seed, ev["bad_tokens"][:3], ev["bad_logits"][:3])               # THE CLAIM
    assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    assert ev["pinned"] == [] and ev["rep"]["spec"]["conservation"]["conserved"], ev["rep"]["spec"]["conservation"]


# ------------------------------------------------------------------ HIT
@pytest.mark.parametrize("level", ["L3", "L4P"])
def test_prefetched_block_is_a_pool_hit(level):
    on = _ev(level, 8)
    off = _ev(level, 8, prefetch=False)
    assert off["rep"] is None and _clean(on) and _clean(off)
    pf0 = {k: c for k, c in on["recount"].cats.items() if k[0] == 0 and c in ("F0", "F1")}
    assert pf0, "precondition (L34): round 0's verify found a prefetched block"
    miss = {k: off["recount"].cats.get(k) for k in pf0 if off["recount"].cats.get(k) != "N"}
    assert miss == {}, list(miss.items())[:5]                         # THE CONTROL: without the prefetch, host loads
    assert on["resid"] == [], on["resid"][:2]                         # every named pool slot holds its host bytes


# ------------------------------------------------------------------ SPLIT
@pytest.mark.parametrize("level,hold", [("L3", False), ("L3", True), ("L4P", False), ("L4P", True)],
                         ids=["L3-ready", "L3-inflight", "L4P-ready", "L4P-inflight"])
def test_wait_split_adds_up(level, hold):
    ev = _ev(level, 8, hold=hold)
    rc, t, sp = ev["recount"].tot, _tot(ev), ev["rep"]["spec"]
    assert rc["W"] + rc["P"] + rc["F0"] + rc["F1"] + rc["N"] == rc["sel"]          # the recount adds up
    assert t["sel"] == rc["sel"], (t, rc)                              # THE SPLIT = the independent recount
    assert t["new"] == rc["N"] == rc["new_lm"], (t, rc)
    assert t["read_ready"] == rc["F0"] and t["read_inflight"] == rc["F1"], (t, rc)
    assert sp["ready_resident"] == rc["W"] + rc["P"], (sp["ready_resident"], rc)
    assert sp["ready_resident"] + t["read_ready"] + t["read_inflight"] + t["new"] == t["sel"]
    rounds = [r for r in sp["rounds"] if "split" in r]
    assert rounds and all(r["split"]["ready_resident"] >= 0 for r in rounds)
    assert all(r["split"]["sel"] == r["split"]["ready_resident"] + r["split"]["read_ready"] +
               r["split"]["read_inflight"] + r["split"]["new"] for r in rounds)
    commits = [r for r in ev["pf"].records if r["jobs_ready"] + r["jobs_inflight"] > 0]
    assert commits and all(len(r["waits"]) == r["layers"] >= 1 for r in commits), \
        [(len(r["waits"]), r["layers"]) for r in commits]               # one commit wait per layer with a commit
    if hold:                                                           # the jobs land only when the commit waits them
        assert rc["F0"] == 0 and rc["F1"] > 0, rc
        assert all(r["jobs_ready"] == 0 and r["jobs_inflight"] > 0 and r["items_inflight"] > 0 for r in commits)
        assert "wait_submitted" in ev["pf"].tp.by                     # the in-flight path waited the submission
    else:
        assert rc["F1"] == 0 and rc["F0"] > 0, rc
        assert all(r["jobs_inflight"] == 0 for r in commits)
    assert _clean(ev) and ev["resid"] == [], (ev["bad_tokens"][:3], ev["resid"][:2])
    assert sp["conservation"]["conserved"], sp["conservation"]


# ------------------------------------------------------------------ PIN
def _cell5():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_spec5_cell
    return r2_spec5_cell


def _res(ev):
    runner = ev["runner"]
    return dict(read=ev["rd"], out=runner.out, fp=(runner.fmax, runner.fsum), K=K, probe_every=0, prefill_s=0.0,
                first_token_agreement=ev["B"], meta={}, prefetch=ev["rep"])


def _rec(C5, arm, ev):
    lg = ev["ref_lg"]
    fp_sh = (lg.amax(dim=-1), lg.sum(dim=-1, dtype=torch.float64))
    return C5.summarize(arm, _res(ev), ev["ref_tok"], fp_sh, None, ev["B"], N,
                        dict(L=L, H=W.HKV, topk=W.TOPK, block_bytes=1))


def test_dropped_pin_is_caught():
    C5 = _cell5()
    clean = _ev("L3", 8)
    rc = clean["recount"].tot
    assert rc["tagged_seen"] > 0, rc                                   # precondition: tagged slots before verify calls
    assert rc["tagged_unpinned"] == 0, rc                              # THE CLAIM: every one carries a prefetch pin
    assert _tot(clean)["evict_free"] == 0, _tot(clean)                 # and none was evicted with a free slot left
    g = {n: ok for n, _, ok, _ in C5.arm_gates("L3P", _res(clean), _rec(C5, "L3P", clean))}
    assert g["L3P-PF-PIN"] and g["L3P-PF-CONS"], g
    p = _ev("L3", 8, neg="no_pin")                                     # PLANT: the slot keeps the victim's age
    pr, pt = p["recount"].tot, _tot(p)
    assert pr["tagged_seen"] > 0, pr                                   # precondition (L34)
    assert pt["evict_free"] + pt["evict_overflow"] > 0, pt            # precondition (L34): tagged slots were evicted
    assert pr["tagged_unpinned"] > 0, pr                               # CAUGHT (the mechanism)
    assert pt["evict_free"] > 0, pt                                    # CAUGHT (the effect)
    gp = {n: ok for n, _, ok, _ in C5.arm_gates("L3P", _res(p), _rec(C5, "L3P", p))}
    assert gp["L3P-PF-PIN"] is False, gp                               # CAUGHT (the cell's MUST gate)
    assert _clean(p), p["bad_tokens"][:3]                              # the pin is a cost property, not exactness


# ------------------------------------------------------------------ GROUP
def test_commit_inside_group_is_caught():
    box = {}
    hz = SW.horizon_for(8, K)
    model, A, Bc = SW.two_caches(8, SW.S0, hz + 64)
    runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(8, dtype=torch.int64), level="L3",
                            make_prefetch=SW5.make_pf(box=box))
    pf = box["pf"]
    runner.rb.begin()
    with pytest.raises(SP.SpecPrefetchError):
        pf.on_verify()                                                 # inside a group: refused
    with pytest.raises(SP.SpecPrefetchError):
        pf.on_release()
    runner.rb.reject_all()
    clean = _ev("L3", 8, filt="zero")
    assert _clean(clean) and clean["resid"] == [], (clean["bad_tokens"][:3], clean["resid"][:2])
    p = _ev("L3", 8, neg="commit_mid_group", filt="zero")              # PLANT: the commit inside the verify group
    m1 = sum(int((r["m"] == 1).sum()) for r in p["records"])
    vn = p["rep"]["spec"]["totals_cs"]["victim_named"]
    assert p["rep"]["counters"]["committed"] > 0 and vn > 0 and m1 > 0, (p["rep"]["counters"], vn, m1)   # (L34)
    assert not (_clean(p) and p["resid"] == []), "commit_mid_group was not caught"                      # CAUGHT


# ------------------------------------------------------------------ WIRING
def test_seams_and_refusals():
    with pytest.raises(SP.SpecPrefetchError):
        SW5.make_pf(neg="other")(None)                                 # unknown control (checked first)
    (m0, x0) = RW.build([4910], nopool=True)
    with pytest.raises(SP.SpecPrefetchError):
        SP.SpecPrefetch(x0, transport=SW5.FakeTransport(), ops=SW5.TWIN_OPS)    # no victim pool
    hz = SW.horizon_for(8, K)
    model, A, Bc = SW.two_caches(1, SW.S0, hz + 64)
    with pytest.raises(SP.SpecPrefetchError):                         # L4 has no on_verify seam
        SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(1, dtype=torch.int64), level="L4", make_prefetch=SW5.make_pf())
    for level, want in (("L3", None), ("L4P", None)):
        box, calls = {}, []
        model, A, Bc = SW.two_caches(1, SW.S0, hz + 64)
        runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(1, dtype=torch.int64), level=level,
                                make_prefetch=SW5.make_pf(box=box))
        pf = box["pf"]
        for name in ("on_draft", "on_verify", "on_release", "on_round"):
            def wrap(*a, _n=name, _o=getattr(pf, name)):
                calls.append(_n if _n != "on_round" else a[0])
                return _o(*a)
            setattr(pf, name, wrap)
        with pytest.raises(SP.SpecPrefetchError):
            pf.bind(runner)                                            # bound once
        for bad in (pf.attach, pf.begin_step, lambda: pf.prefetch(torch.zeros((0, 4), dtype=torch.int64))):
            with pytest.raises(SP.SpecPrefetchError):
                bad()
        runner.run(8, max_rounds=3)
        G = runner.rounds_run
        per = (["begin"] + ["on_draft"] * K + ["on_verify", "on_release", "end"] if level == "L3"
               else ["begin", "on_verify"] + ["on_draft"] * K + ["on_release", "end"])
        pre = [] if level == "L3" else ["on_draft"] * K               # the L4P prelude's K drafts
        assert calls == pre + per * G, calls
        with pytest.raises(SP.SpecPrefetchError):
            pf.close()                                                 # unbind first
        body = runner.rs.body
        assert body.__dict__.get("decode_fixed") == pf._hook
        runner.detach()
        assert "decode_fixed" not in body.__dict__ and runner.rs.rollback is None and runner.rs.draft is None
        pf.close()
        assert pf.tp.closed and pf.pins_live_prefetch() == 0


@pytest.mark.parametrize("level", ["L3", "L4P"])
def test_host_syncs_are_declared(level):
    """Two runs (one trap each: a trap records only the calls whose DIRECT caller is its file)."""
    hz = SW.horizon_for(8, K)
    hits = []
    for path in (SP.__file__, SR.__file__):
        box = {}
        model, A, Bc = SW.two_caches(8, SW.S0, hz + 64)
        runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(8, dtype=torch.int64), level=level,
                                make_prefetch=SW5.make_pf(box=box))
        with RW.SyncTrap(path) as trap:
            runner.run(8, max_rounds=3)
        assert runner.rounds_run >= 1 and box["pf"].issue["sent"] > 0, box["pf"].issue   # precondition: jobs ran
        hits.append(trap.hits)
    assert {h[0] for h in hits[0]} == {"_issue_snap"}, sorted({h[0] for h in hits[0]})   # the declared host reads
    assert {h[0] for h in hits[1]} <= {"run", "_edge_before"}, sorted({h[0] for h in hits[1]})
    assert any(h[0] == "run" for h in hits[1])                        # the trap is not blind


# ------------------------------------------------------------------ PACKER
@pytest.fixture(scope="module")
def ext(tmp_path_factory):
    """Packer v3 of THIS tree, built (OpenMP), as test_pool_prefetch.py builds it."""
    spec = importlib.util.spec_from_file_location("gilfree_pack_v3_spec5", str(EFF / "gilfree_pack.py"))
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    e = gp.load(str(tmp_path_factory.mktemp("gilfree3_build_spec5")))
    assert int(e.VERSION) == 3 and int(e.DESC) >= 2
    return e


@pytest.mark.parametrize("level", ["L3", "L4P"])
def test_real_packer_tokens_equal_shipped(level, ext):
    ev = SW5.run_pf(8, level=level, ext=ext, lanes=LANES, filt=FILTER)
    pf = ev["pf"]
    try:
        assert ev["rep"]["stats"]["jobs"] > 0 and _hits(ev) > 0, (ev["rep"]["stats"], _tot(ev))   # precondition
        assert _clean(ev), (ev["bad_tokens"][:3], ev["bad_logits"][:3])                          # THE CLAIM
        assert ev["resid"] == [] and ev["host"] == [] and ev["pinned"] == [], (ev["resid"][:2], ev["pinned"][:2])
        assert ev["rep"]["spec"]["conservation"]["conserved"], ev["rep"]["spec"]["conservation"]
        rc, t = ev["recount"].tot, _tot(ev)
        assert t["new"] == rc["N"] and t["read_ready"] + t["read_inflight"] == rc["F0"] + rc["F1"], (t, rc)
    finally:
        ev["recount"].detach()
        ev["runner"].detach()
        pf.close()


# ------------------------------------------------------------------ CELL
def test_cell5_arguments_and_report(monkeypatch):
    C5 = _cell5()
    base = ["--out", "x", "--workload", "w"]
    a = C5.parse_args(base + ["--spec", "3", "--spec-prefetch", "on"])
    assert (a.spec_arms, a.spec_pf_lanes, a.spec_pf_threads, a.require_stress) == (["L3", "L3P", "L4P", "L4PP"], 128,
                                                                                   8, 0)
    for bad in (["--spec", "3"],                                       # the prefetch arms need --spec-prefetch on
                ["--spec", "3", "--spec-arms", "L3,L4P", "--spec-prefetch", "on"],      # on without a prefetch arm
                ["--spec", "2", "--spec-prefetch", "on"], ["--spec", "3", "--spec-prefetch", "on", "--pool", "0"],
                ["--spec", "3", "--spec-prefetch", "on", "--spec-arms", "L4X"],
                ["--spec", "3", "--spec-prefetch", "on", "--spec-pf-lanes", "0"]):
        with pytest.raises(SystemExit):
            C5.parse_args(base + bad)
    assert C5.main(base + ["--spec-arms", "L3,L4P"]) == 20            # --spec 0: refused (flag off)
    assert C5.BASE_ARM == dict(L3="L3", L3P="L3", L4P="L4P", L4PP="L4P") and set(C5.EXACT_ARMS5) == set(C5.ARMS5)
    assert {"stamp", "L3P_tok", "L4PP_tok", "L3P_vs_L3", "L4PP_vs_L4P", "L3P_served", "L4PP_served",
            "L3P_inflight_ms", "L4PP_inflight_ms"} == set(C5.PRED_SPEC5)
    assert C5.pick_cores(2, {0, 1, 2, 3}) == [2, 3] and C5.pick_cores(4, {0, 1}) is None
    # the report and the gates on CPU runs (the shipped reference as SH)
    ev = _ev("L3", 8)
    rec = _rec(C5, "L3P", ev)
    assert rec["arm"] == "L3P" and rec["base_arm"] == "L3" and rec["tokens"]["n_unequal"] == 0, rec["tokens"]
    pf = rec["prefetch"]
    assert pf["released_rounds"] > 0 and pf["blocks_per_round"]["READY_prefetched"] is not None, pf
    assert pf["useful"]["committed"] == pf["useful"]["ended"] > 0, pf["useful"]
    assert pf["waits"]["ready_ms"] == 0.0 and pf["conservation"]["conserved"], pf["waits"]
    gates = {n: ok for n, _, ok, _ in C5.arm_gates("L3P", _res(ev), rec, require_stress=True)}
    assert gates == {"L3P-ERR": True, "L3P-COMMIT": True, "L3P-TOK": True, "L3P-FP": True, "L3P-PF-ACT": True,
                     "L3P-PF-CONS": True, "L3P-PF-PIN": True, "L3P-PF-NEW": True,
                     "L3P-STRESS": False}, gates                      # journal 1024: no overflow, so STRESS fails
    st = _ev("L3", 8, journal=CAP_TINY, lanes=4)                       # the stressed miniature: tiny journal, 4 lanes
    gs = {n: ok for n, _, ok, _ in C5.arm_gates("L3P", _res(st), _rec(C5, "L3P", st), require_stress=True)}
    assert all(gs.values()), gs                                        # STRESS passes there, and every other gate
    assert st["rep"]["stats"]["lane_refused"] > 0, st["rep"]["stats"]  # precondition: lanes overflowed
    e4 = _ev("L4P", 8, pin=True)
    g4 = {n: ok for n, _, ok, _ in C5.arm_gates("L4PP", _res(e4), _rec(C5, "L4PP", e4))}
    assert g4 == {"L4PP-ERR": True, "L4PP-COMMIT": True, "L4PP-TOK": True, "L4PP-FP": True, "L4PP-PF-ACT": True,
                  "L4PP-PF-CONS": True, "L4PP-PF-PIN": True, "L4PP-PF-NEW": True}, g4
    g3 = {n: ok for n, _, ok, _ in C5.arm_gates("L3", _res(ev), _rec(C5, "L3", ev))}
    assert set(g3) == {"L3-ERR", "L3-COMMIT", "L3-TOK", "L3-FP"}      # a base arm: no prefetch gate
    missing = {n: ok for n, _, ok, _ in C5.arm_gates("L3P", dict(_res(ev), prefetch=None), rec)}
    assert not missing["L3P-PF-ACT"] and not missing["L3P-PF-CONS"]  # no report: the prefetch gates FAIL
    for k in C5.PRED_SPEC5:
        monkeypatch.setitem(C5.PRED_SPEC5, k, None)
    sp = {"L3": rec, "L3P": rec}
    v = {n: x for n, x, _ in C5.verdicts(None, sp, 8)}
    assert v["L3P-TOK"] == "REPORT" and v["L4PP-RUN"] == "NOT_RUN" and v["L5-VS-L4"] == "REPORT", v
    monkeypatch.setitem(C5.PRED_SPEC5, "L3P_tok", True)
    monkeypatch.setitem(C5.PRED_SPEC5, "L3P_vs_L3", (0.9, 1.1))
    v = {n: x for n, x, _ in C5.verdicts(None, sp, 8)}
    assert v["L3P-TOK"] == "CONFIRMED" and v["L3P-VS-L3"] == "CONFIRMED", v
    src = (EFF / "r2_spec5_cell.py").read_text()
    assert "gates.extend(arm_gates(arm, res, rec, require_stress=bool(a.require_stress)))" in src


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_spec5_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_sp5():
    for bad in ("sp3m", "sp5c", "mb1", "rg73"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP5 = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_spec5_cell.py" % sha).stdout
    for cell, pool, batch, steps, journal, lanes, stress, smin, mem in (
            ("sp5m", 10, 8, 64, 8, 8, 1, 40, "100G"), ("sp5a", 65, 216, 512, 256, 128, 0, 110, "240G"),
            ("sp5b", 10, 318, 512, 256, 1024, 0, 160, "340G")):
        r = _step(cell, sha)
        if cell != "sp5m" and not registered:                         # the big cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP5 is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L3,L3P,L4P,L4PP steps %d K 3 journal %d lanes %d stress %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, steps, journal, lanes, stress, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--spec 3",
                     "--spec-prefetch on", "--spec-pf-lanes %d" % lanes, "--spec-journal-blocks %d" % journal,
                     "--require-stress %d" % stress, "--spec-twin fused --spec-body fused",
                     STEP_ROOT + "/benchmarks/Efficiency/r2_spec5_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "sp_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
