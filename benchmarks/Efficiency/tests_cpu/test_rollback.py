"""CPU tests of Task 1.5 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
accept, correct and roll back per request in the cached-NOSA engine with per-request lengths, the victim pool included
(nosi/nosi/ragged_rollback.py; r2_base_cell.py --rollback; r2_base_rollback_step.sh cells rb1 / rb1x). The world:
tests_cpu/rollback_world.py over tests_cpu/ragged_world.py (the SHIPPED engine from its own source, every CUDA op a
torch stand-in with the kernel's flat addresses, L28). B 1 and B 8 (the largest CPU batch) in every claim.

GATE 0   test_flag_off_is_shipped: the engine sources are the Task 1.3 bytes (only ragged_rollback.py is new; only the
         pool names are new in state_snapshot.py, which no engine module imports); no engine module imports the
         rollback; r2_base_cell without --rollback parses as before and every rollback call in main sits under
         `if a.rollback`; a world with the Rollback attached but no group is the plain world bit for bit (logits and
         every state tensor, through a block fill); detach() puts every original callable back.
ACCEPT   test_accept_all_equals_shipped: groups of K = 3 tentative steps, every one kept (n_acc = K): the logits are
         BITWISE the SHIPPED engine's (Llama.decode_inference on InfLLMv2Cache) at every step, and the whole state is
         the plain ragged world's after K x groups steps (block fill inside a group: the deferred write-back and the
         redirect run).
REJECT   test_reject_all_restores_snapshot_with_pool: n_acc = 0: every state tensor (pool tensors, the LRU clock, the
         host rows, the twin, the scratch, the dead bytes) equals the snapshot before the group, bitwise; the ported
         state_snapshot.CacheSnapshot (now with the pool fields) gives equal takes; the next steps equal a world that
         never ran the group. Pool on (B 1, B 8) and off (B 8).
PARTIAL  test_partial_accept_equals_shipped_prefix: a different n_acc per request (0 .. K, every value at B 8): each
         request's rows equal the rows of a world that ran n_acc[b] plain steps (group 0, scratch included), and the
         whole state equals the reference that ran only the kept steps (every group, scratch excepted: the reference's
         inactive steps rewrite it); the kept logits equal.
EDGE     test_reject_at_block_edge_keeps_host_rows: a rejected step fills a block: the host rows of that block never
         change, at any step of the group and after the commit; a kept fill reaches the host rows at the commit only.
         THE dr1 CASE (SCORING_dr1.md section 7): draft steps with wrong tokens write part of the tail block, are
         rejected, and a later PLAIN step fills the block and writes all 64 rows back: those rows are the kept rows only
         (bitwise the world that never ran the drafts).
FREE     test_rollback_frees_after_last_reader: with a deferred executor, the next step of layer l waits layer l's
         release event before it writes, and the result equals the immediate executor's.
SYNC     test_no_host_sync_in_commit_path: the static scan of COMMIT_PATH finds 0 host syncs; the run-time trap records
         0 tensor-to-host reads from the module during the tentative steps, commit and reject_all.
OVERFLOW test_journal_overflow_falls_back_to_host: with 1 journal slot per layer and step, the live state is exact (the
         lost blocks come back from the host) and the next steps equal a world without the group.
FUSED    test_fused_body_rollback_under_the_interpreter (needs triton): the rollback with FusedBody (the GPU cell's
         body) equals the reference with the torch body.
CELL     test_rollback_cell_arguments_schedule_compare_verdicts, test_accept_perreq, test_refusals,
         test_step_script_rb1.
NEGATIVE CONTROLS, each with a precondition that the plant changes something on these data (L34): no_slot_restore,
no_tail_restore, no_meta_restore, no_pool_restore, no_twin_restore (REJECT), no_defer, no_redirect and no_rollback (the
dr1 flow: the draft rows reach the host at the edge) (EDGE), skip_release_wait (FREE), host_sync and a planted source
line (SYNC), no_host_fallback (OVERFLOW). The rejected steps of PARTIAL and EDGE read WRONG tokens (as a rejected draft
does), so their rows differ from the rows that the kept steps write later at the same positions.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_rollback.py
"""
from __future__ import annotations

import ast
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
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-rollback"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
BASE = "0bdc0457c779268db68163d3417277f88bb5dec6"          # the Task 1.3 tree this branch starts from


def _world():
    if "rollback_world" not in sys.modules:
        spec = importlib.util.spec_from_file_location("rollback_world", str(HERE / "rollback_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rollback_world"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["rollback_world"]


RW = _world()
W, RE, RR, SS = RW.W, RW.RE, RW.RR, RW.SS
K, CAP, L = RW.K, RW.CAP, RW.L
W0 = 12                                # plain steps before the first group: the pools fill (live victims)
# S % 64 = 51, 50, 49, 51, 50, 49, 22, 19: after the W0 = 12 warm-up steps the first group starts at p % 64 = 63, 62, 61
# (a fill at tentative step 0 / 1 / 2), 63, 62, 61, 34, 31; 72-74 blocks per request, so the selection (64 slots)
# leaves blocks out and the pool moves blocks every step
S8 = [4659, 4658, 4657, 4723, 4722, 4721, 4630, 4691]
S1 = [4658]                            # after W0: p % 64 = 62, a fill at tentative step 1 of the first group
assert len(S8) == W.B_MAX and all((S + W0) % W.BS in (61, 62, 63) for S in S8[:6] + S1)


def _rb(rs, **kw):
    kw.setdefault("capacity", CAP)
    return RR.Rollback(rs, K, **kw)


def _equal_before_step0(*states):
    for s in states[1:]:
        assert RW.diff(states[0], s) == [], "precondition: two worlds of one seed differ before step 0"


# ------------------------------------------------------------------ GATE 0
def _guarded(fn, is_target, is_guard):
    """Line numbers of the nodes of fn with is_target(node) that no `if <is_guard>:` body contains (Task 1.3's helper)."""
    bad = []

    def walk(node, guarded):
        for child in ast.iter_child_nodes(node):
            g = guarded or (isinstance(node, ast.If) and is_guard(node.test) and child in node.body)
            if is_target(child) and not g:
                bad.append(getattr(child, "lineno", None))
            walk(child, g)
    walk(fn, False)
    return bad


@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_flag_off_is_shipped(S_list):
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", BASE + "^{commit}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        # The integration tree nosi-r2-spec merges three more hooks; each file is excluded here and checked by its own
        # Gate 0: verify_rows.py (test_verify_rows.py), cache_engine.py's appended draft block (test_draft_pool.py),
        # pool_prefetch.py / r2_transport_gilfree_v3.py (test_pool_prefetch.py), spec_round.py (test_spec_round.py).
        r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", BASE, "--", "nosi/nosi", "dependencies",
                            ":(exclude)nosi/nosi/ragged_rollback.py", ":(exclude)nosi/nosi/state_snapshot.py",
                            ":(exclude)nosi/nosi/verify_rows.py", ":(exclude)nosi/nosi/cache_engine.py",
                            ":(exclude)nosi/nosi/pool_prefetch.py", ":(exclude)nosi/nosi/r2_transport_gilfree_v3.py",
                            ":(exclude)nosi/nosi/spec_prefetch.py",  # rung L5 (Gate 0: test_spec_prefetch.py)
                            ":(exclude)nosi/nosi/spec_tentative.py", # WS-A (Gate 0: test_wsa.py)
                            ":(exclude)nosi/nosi/spec_tentative_union.py",   # I2 (Gate 0: test_i2.py)
                            ":(exclude)nosi/nosi/spec_round.py", ":(exclude,glob)**/__pycache__/**"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "", "the engine sources differ from %s:\n%s" % (BASE, r.stdout)
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    assert "ragged_rollback.py" in src and "state_snapshot.py" in src
    # spec_round.py (Phase 2, opt-in: only r2_spec_cell.py and the tests import it) is the one module that uses it
    assert [n for n, t in src.items() if "ragged_rollback" in t and n not in ("ragged_rollback.py",
                                                                               "spec_round.py")] == []
    assert [n for n, t in src.items() if "state_snapshot" in t and n != "state_snapshot.py"] == []
    assert {"_pool_map", "_pool_age", "_pool_target", "_pool_action"} <= set(SS._ENGINE_TENSORS)
    assert "_pool_stamp_base" in SS._ENGINE_SCALARS
    BC = W.bc()
    a = BC.parse_args(["--out", "x", "--workload", "w", "--batch", "216", "--arms", "SH", "--pool", "65", "--mbase"])
    assert not a.rollback
    tree = ast.parse((EFF / "r2_base_cell.py").read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {"run_rollback", "verdicts_rollback", "rollback_compare", "rollback_schedule", "rollback_agreement"}

    def is_call(n):
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in calls

    def is_a_rollback(t):
        return isinstance(t, ast.Attribute) and t.attr == "rollback" and isinstance(t.value, ast.Name) and t.value.id == "a"
    assert _guarded(fns["main"], is_call, is_a_rollback) == []
    assert sum(1 for n in ast.walk(fns["main"]) if is_call(n)) == 5
    # behaviour: the Rollback attached, no group: the plain world bit for bit, through a block fill
    worlds = [RW.build(S_list) for _ in range(2)]
    _equal_before_step0(*(RW.state(rs) for _, rs in worlds))
    (mx, x), (my, y) = worlds
    rb = _rb(y)
    tok = RW.tokens(0, len(S_list))
    dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
    h0 = RW.state(x)
    n = 16                                            # S1 fills at step 13 (p 4671); S8's first request at step 12
    for i in range(n):
        lx, ly = dx.plain(1)[0], dy.plain(1)[0]
        assert torch.equal(W.bits(lx), W.bits(ly)), "step %d: the attached world's logits differ" % i
    assert RW.diff(RW.state(x), RW.state(y)) == []
    assert any(n_.endswith("_k_cpu") for n_ in RW.diff(h0, RW.state(x))), \
        "precondition: a block fill wrote host rows in these %d steps" % n
    o = rb._orig
    rb.detach()
    assert y.ops.kernels["gather_k"] is o["gather_k"] and y.ops.kernels["gather_v_bias"] is o["gather_v_bias"]
    assert y.ops.kernels["pool_update"] is o["pool_update"] and y.body.writeback_kernel is o["writeback"]
    assert y.ops.linear is o["linear"] and y.rollback is None
    assert all("step_unchecked" not in t.__dict__ for t in y.st.twins)


# ------------------------------------------------------------------ ACCEPT
@pytest.mark.parametrize("B", [1, W.B_MAX], ids=["B1", "B8"])
def test_accept_all_equals_shipped(B):
    S, G = 4607, 4                                    # S % 64 = 63: the first tentative step fills block 71; 72
    t0 = (S // W.BS) * W.BS                           # blocks, so the pool moves blocks (the shipped prefill path)
    W.STATS.clear()
    model = W.make_model(0)
    rows = W.prompt_rows(1, [S] * B)
    caches = []
    for _ in range(3):
        c = W.new_cache(G * K + 64)
        W.shipped_prefill(c, rows, S, B)
        W.poison_shipped(c, S)
        caches.append(c)
    A, Cb, Cc = caches
    xs = RE.attach(model, Cb, S, G * K, RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True))
    ys = RE.attach(model, Cc, S, G * K, RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True))
    RW.define_undefined(xs)
    RW.define_undefined(ys)
    _equal_before_step0(RW.state(xs), RW.state(ys))
    rb = _rb(xs)
    g = torch.Generator().manual_seed(2)
    tok = torch.randint(0, W.V, (B, 1), generator=g)
    pos = torch.full((B, 1), S, dtype=torch.int32)
    warm = dict(state=None, rewarms=[])
    host0 = xs.cache.layers[0].cache_engine._k_cpu[:, t0:t0 + W.BS].clone()
    deferred_seen = False
    for grp in range(G):
        rb.begin()
        for u in range(K):
            W.TAG[0] = "shipped"
            la = W.shipped_step(model, A, tok, pos, warm)
            W.TAG[0] = "ragged"
            lb = RE.decode_step(model, xs, tok, pos)
            lc = RE.decode_step(model, ys, tok, pos)
            assert bool(torch.isfinite(la).all()), "group %d step %d: the shipped logits are not finite" % (grp, u)
            assert torch.equal(W.bits(la), W.bits(lb)), "group %d step %d: the rollback world differs from the " \
                                                        "shipped step" % (grp, u)
            assert torch.equal(W.bits(lb), W.bits(lc)), "group %d step %d: rollback vs plain ragged" % (grp, u)
            if grp == 0:                                  # block 71 fills at step 0: deferred until the commit
                assert torch.equal(W.bits(xs.cache.layers[0].cache_engine._k_cpu[:, t0:t0 + W.BS]), W.bits(host0)), \
                    "group 0 step %d: a tentative fill reached the host rows" % u
                deferred_seen = deferred_seen or not torch.equal(
                    W.bits(ys.cache.layers[0].cache_engine._k_cpu[:, t0:t0 + W.BS]), W.bits(host0))
            tok = la[:, -1].argmax(-1, keepdim=True)
            pos = pos + 1
        rb.commit(torch.full((B,), K, dtype=torch.int64), torch.full((B,), K, dtype=torch.int64))
    assert RW.diff(RW.state(xs), RW.state(ys)) == []
    for l in range(L):                                # the engine tables against the shipped engine (run_equal's list)
        ea, eb = A.layers[l].cache_engine, xs.cache.layers[l].cache_engine
        for name in W.ENGINE_TABLES:
            assert torch.equal(W.bits(getattr(ea, name)), W.bits(getattr(eb, name))), "layer %d: %s vs shipped" % (l, name)
        assert ea._pool_stamp_base == eb._pool_stamp_base
    st = rb.stats()
    assert deferred_seen                              # precondition: the plain world wrote block 71 during group 0
    assert sum(st["redirected"]) > 0                  # precondition: a later step loaded the staged block
    assert W.STATS["shipped"].get("move_out", 0) > 0  # precondition: the pool parked blocks
    assert sum(st["overflow"]) == 0 and sum(st["destroyed"]) > 0


# ------------------------------------------------------------------ REJECT
def _reject_run(S_list, neg=None, capacity=CAP, nopool=False):
    """Two worlds of one seed; W0 plain steps; world x: a group of K steps, then reject_all. Returns the evidence."""
    (mx, x), (my, y) = RW.build(S_list, nopool=nopool), RW.build(S_list, nopool=nopool)
    _equal_before_step0(RW.state(x), RW.state(y))
    rb = _rb(x, neg=neg, capacity=capacity)
    tok = RW.tokens(0, len(S_list))
    dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
    dx.plain(W0)
    dy.plain(W0)
    snap = RW.state(x)
    cs0 = SS.CacheSnapshot(x.cache).take()
    facts, pre = {}, {}

    def before_commit():
        facts.update(RW.journal_facts(rb, torch.zeros(len(S_list), dtype=torch.int64)))
        pre.update(RW.state(x))
    rb.begin()
    for u in range(K):
        dx.step(u)
    before_commit()
    rb.reject_all()
    cs1 = SS.CacheSnapshot(x.cache).take()
    return dict(x=x, y=y, dx=dx, dy=dy, rb=rb, snap=snap, pre=pre, post=RW.state(x), facts=facts, cs0=cs0, cs1=cs1,
                stats=rb.stats())


@pytest.mark.parametrize("S_list,nopool", [(S1, False), (S8, False), (S8, True)], ids=["B1", "B8", "B8-nopool"])
def test_reject_all_restores_snapshot_with_pool(S_list, nopool):
    ev = _reject_run(S_list, nopool=nopool)
    assert RW.diff(ev["post"], ev["snap"]) == []                       # every tensor and the clock, bitwise
    assert RW.snapshot_equal(ev["cs0"], ev["cs1"]) == []               # the ported CacheSnapshot agrees
    changed = RW.diff(ev["pre"], ev["snap"])                           # preconditions: the group changed each family
    for fam in ("slots", "meta", "twin") + (() if nopool else ("pool",)):
        assert RW.in_family(changed, fam), (fam, changed)
    assert sum(ev["stats"]["overflow"]) == 0 and ev["facts"]["lost"] > 0
    if not nopool:
        assert ev["facts"]["swap"] + ev["facts"]["move_out"] > 0, ev["facts"]
        assert any(n.endswith("_pool_stamp_base") for n in changed)    # the clock moved and came back
    for i in range(3):                                                 # the next steps: as if the group never ran
        lx, ly = ev["dx"].plain(1)[0], ev["dy"].plain(1)[0]
        assert torch.equal(W.bits(lx), W.bits(ly)), "step %d after reject_all differs" % i
    assert RW.diff(RW.state(ev["x"]), RW.state(ev["y"])) == []


@pytest.mark.parametrize("neg", sorted(RW.NEG_FAMILY))
def test_negctl_reject_all_plants(neg):
    fam = RW.NEG_FAMILY[neg]
    ev = _reject_run(S8, neg=neg)
    changed = RW.diff(ev["pre"], ev["snap"])
    assert RW.in_family(changed, fam), "precondition: the group changed the %s family (%s)" % (fam, changed)
    if neg == "no_slot_restore":
        assert ev["facts"]["lost"] + ev["facts"]["swap"] + ev["facts"]["move_out"] > 0, ev["facts"]
    bad = RW.diff(ev["post"], ev["snap"])
    assert bad and RW.in_family(bad, fam), (neg, bad)                  # CAUGHT
    if neg == "no_pool_restore":                                       # the ported snapshot sees the pool now
        assert any(".engine._pool_" in n for n in RW.snapshot_equal(ev["cs0"], ev["cs1"]))


# ------------------------------------------------------------------ PARTIAL
def _partial_run(S_list, groups, body="torch"):
    """World x: the rollback; world r: the reference that runs only the kept steps; world c: K plain steps after the
    warm-up (the states of v = 0 .. K steps). Group 0 also checks each request against c; every group against r."""
    worlds = [RW.build(S_list, ops=W.ragged_ops(body=body if i == 0 else "torch")) for i in range(3)]
    _equal_before_step0(*(RW.state(rs) for _, rs in worlds))
    (mx, x), (mr, r), (mc, c) = worlds
    rb = _rb(x)
    tok = RW.tokens(0, len(S_list))
    dx, dr, dc = (RW.Driver(m, rs, S_list, tok) for m, rs in worlds)
    for d in (dx, dr, dc):
        d.plain(W0)
    assert RW.diff(RW.state(x), RW.state(r)) == [] and RW.diff(RW.state(x), RW.state(c)) == []
    clones = [RW.state(c)]
    for _ in range(K):
        dc.plain(1)
        clones.append(RW.state(c))
    ev = dict(facts=[], fills=[], n_acc=[])
    for g, n_acc in enumerate(groups):
        n_acc = torch.as_tensor(n_acc, dtype=torch.int64)
        ev["fills"] += RW.fills(S_list, dx.k, n_acc)
        lx = dx.group(rb, n_acc, before_commit=lambda: ev["facts"].append(RW.journal_facts(rb, n_acc)), wrong=True)
        lr = dr.masked(n_acc)
        assert RW.kept_logits_equal(lx, lr, n_acc) == [], "group %d: kept logits differ" % g
        sx, sr = RW.state(x), RW.state(r)
        assert RW.diff(sx, sr, skip=RW.SCRATCH) == [], "group %d: %s" % (g, RW.diff(sx, sr, skip=RW.SCRATCH))
        if g == 0:
            for b in range(len(S_list)):
                want = RW.rows_of(clones[int(n_acc[b])], b)
                got = RW.rows_of(sx, b)
                bad = RW.diff(got, want)
                assert bad == [], "request %d (n_acc %d) differs from %d plain steps: %s" % (b, int(n_acc[b]),
                                                                                           int(n_acc[b]), bad)
        ev["n_acc"].append(n_acc)
    ev.update(stats=rb.stats(), x=x)
    return ev


@pytest.mark.parametrize("S_list,groups", [
    (S1, [[1], [3], [0], [2]]),
    (S8, [[0, 1, 2, 3, 0, 1, 2, 3], [3, 0, 1, 2, 2, 3, 0, 1], [1, 1, 0, 3, 3, 0, 2, 2], [2, 3, 3, 0, 1, 2, 1, 0]])],
    ids=["B1", "B8"])
def test_partial_accept_equals_shipped_prefix(S_list, groups):
    ev = _partial_run(S_list, groups)
    vals = {int(v) for na in ev["n_acc"] for v in na}
    assert vals == set(range(K + 1))                                   # precondition: every n_acc value 0 .. K
    fl = ev["fills"]
    assert any(kept for _, _, kept in fl) and any(not kept for _, _, kept in fl), fl   # kept and rejected fills
    tot = {k: sum(f[k] for f in ev["facts"]) for k in ev["facts"][0]}
    assert tot["lost"] > 0 and tot["swap"] + tot["move_out"] > 0, tot   # rejected steps moved / loaded slots
    if len(S_list) > 1:
        assert tot["live_victims"] > 0, tot                            # a rejected MOVE_OUT evicted a live block
    st = ev["stats"]
    assert sum(st["overflow"]) == 0 and sum(st["redirected"]) > 0, st


# ------------------------------------------------------------------ EDGE
S8E = [4607, 4606, 4605, 4671, 4670, 4669, 4600, 4632]                 # fills at u 0, 1, 2, 0, 1, 2, none, none
EDGE = {"B1": ([4606], [[1], [3]]),                                    # fill at u 1: rejected (n_acc 1), then kept
        "B8": (S8E, [[0, 1, 3, 1, 2, 2, 3, 0], [3, 3, 3, 3, 3, 3, 3, 3]])}


def _edge_run(S_list, groups, neg=None):
    """World x: the rollback (neg); world r: the reference of the kept steps. After every tentative step the host rows
    of each request's tail block at the group start are those of the start; after the commit the state equals r."""
    (mx, x), (mr, r) = RW.build(S_list), RW.build(S_list)
    rb = _rb(x, neg=neg)
    tok = RW.tokens(0, len(S_list))
    dx, dr = RW.Driver(mx, x, S_list, tok), RW.Driver(mr, r, S_list, tok)
    ev = dict(fills=[], host_moved=0)
    for g, n_acc in enumerate(groups):
        n_acc = torch.as_tensor(n_acc, dtype=torch.int64)
        ev["fills"] += [(g,) + f for f in RW.fills(S_list, dx.k, n_acc)]
        blk = (dx.positions() // W.BS).tolist()
        host0 = [[(lay.cache_engine._k_cpu[b, t * 64:(t + 1) * 64].clone(),
                   lay.cache_engine._v_cpu[b, t * 64:(t + 1) * 64].clone()) for b, t in enumerate(blk)]
                 for lay in x.cache.layers]

        def after_step(u, g=g, blk=blk, host0=host0):
            for l, lay in enumerate(x.cache.layers):
                for b, t in enumerate(blk):
                    k0, v0 = host0[l][b]
                    now_k = lay.cache_engine._k_cpu[b, t * 64:(t + 1) * 64]
                    now_v = lay.cache_engine._v_cpu[b, t * 64:(t + 1) * 64]
                    assert torch.equal(W.bits(now_k), W.bits(k0)) and torch.equal(W.bits(now_v), W.bits(v0)), \
                        "group %d step %d: request %d's host rows of block %d changed inside the group" % (g, u, b, t)
        lx = dx.group(rb, n_acc, after_step=after_step, wrong=True)
        lr = dr.masked(n_acc)
        bad = RW.kept_logits_equal(lx, lr, n_acc)
        assert bad == [], "group %d: kept logits differ at %s" % (g, bad)
        assert RW.diff(RW.state(x), RW.state(r), skip=RW.SCRATCH) == [], "group %d: the state differs" % g
    ev.update(stats=rb.stats(), x=x, r=r)
    return ev


WD = 6                                     # plain steps before the drafts: S1 at p % 64 = 56; S8 at 57, 56, 55, ..., 28, 25
DRAFT_EDGE = {"B1": S1, "B8": S8}


def _draft_edge_run(S_list, neg=None):
    """THE dr1 CASE (KS/analysis/plan/SCORING_dr1.md section 7): K DRAFT steps with WRONG tokens write part of the tail
    block (no fill: the DraftBlockEdge case), reject_all, then plain steps with the true tokens: a plain (shipped) step
    fills the block and writes all 64 rows to the host. World r runs the same plain steps and never the drafts. Returns
    the evidence; it asserts nothing about the result (the plant no_rollback must run to the end)."""
    (mx, x), (mr, r) = RW.build(S_list), RW.build(S_list)
    _equal_before_step0(RW.state(x), RW.state(r))
    rb = _rb(x, neg=neg)
    B = len(S_list)
    tok = RW.tokens(0, B)
    dx, dr = RW.Driver(mx, x, S_list, tok), RW.Driver(mr, r, S_list, tok)
    dx.plain(WD)
    dr.plain(WD)
    p0 = dx.positions()
    blk = p0 // W.BS
    edge = [b for b in range(B) if int(p0[b] % W.BS) >= 40]           # these requests reach their block edge below
    n_plain = max(W.BS - int(p0[b] % W.BS) for b in edge) + 1
    ev = dict(edge=edge, draft_fill=bool(((p0 + K) // W.BS != blk).any()), host_moved_in_drafts=[])   # a draft p
    #                                                                     with (p + 1) % 64 == 0 would fill the block
    before = RW.state(x)
    rows = [[W.TAIL0 + int((p0[b] + u) % W.BS) for u in range(K)] for b in range(B)]
    rb.begin()
    for u in range(K):
        dx.step(u, wrong=torch.ones(B, dtype=torch.bool))
        ev["host_moved_in_drafts"] += [n for n in RW.diff(before, RW.state(x)) if n.endswith(("_k_cpu", "_v_cpu"))]
    e0 = x.cache.layers[0].cache_engine
    draft_k = [e0._k_gpu[b, rows[b]].clone() for b in range(B)]          # the draft rows in the tail block
    rb.reject_all()
    ev["logits_equal"] = []
    for i in range(n_plain):
        lx, lr = dx.plain(1)[0], dr.plain(1)[0]
        ev["logits_equal"].append(torch.equal(W.bits(lx), W.bits(lr)))
        if i == K - 1:                                                   # r wrote the same positions with true tokens
            er = r.cache.layers[0].cache_engine
            ev["drafts_differ"] = [not torch.equal(W.bits(draft_k[b]), W.bits(er._k_gpu[b, rows[b]]))
                                   for b in range(B)]
    ev["edge_written"] = []
    ev["host_rows_equal"] = []
    for b in edge:
        t = int(blk[b])
        for l in range(L):
            ex, er = x.cache.layers[l].cache_engine, r.cache.layers[l].cache_engine
            hx = [getattr(ex, n)[b, t * W.BS:(t + 1) * W.BS] for n in ("_k_cpu", "_v_cpu")]
            hr = [getattr(er, n)[b, t * W.BS:(t + 1) * W.BS] for n in ("_k_cpu", "_v_cpu")]
            ev["edge_written"].append(all(bool(torch.isfinite(h).all()) for h in hr))
            ev["host_rows_equal"].append(all(torch.equal(W.bits(a), W.bits(c)) for a, c in zip(hx, hr)))
    ev["diff"] = RW.diff(RW.state(x), RW.state(r))
    return ev


@pytest.mark.parametrize("key", sorted(EDGE))
def test_reject_at_block_edge_keeps_host_rows(key):
    # (1) fills inside groups: a rejected fill never reaches the host rows; a kept fill reaches them at the commit only
    S_list, groups = EDGE[key]
    ev = _edge_run(S_list, groups)
    rej = [f for f in ev["fills"] if not f[3]]
    kept = [f for f in ev["fills"] if f[3]]
    assert rej and kept, ev["fills"]                                   # precondition: a rejected and a kept fill
    assert any(u < K - 1 for _, _, u, _ in ev["fills"])                # a fill before the last step: the redirect ran
    assert sum(ev["stats"]["redirected"]) > 0 and sum(ev["stats"]["overflow"]) == 0
    # (2) the dr1 case: rejected draft rows in a part-filled tail, then a plain edge step writes the whole block back
    d = _draft_edge_run(DRAFT_EDGE[key])
    assert d["edge"] and not d["draft_fill"]                           # precondition: no draft fills; a plain edge
    assert all(d["drafts_differ"]), d["drafts_differ"]                 # precondition: the draft rows are not the kept
    assert all(d["edge_written"]), d["edge_written"]                   # precondition: the edge wrote the host block
    assert d["host_moved_in_drafts"] == []                             # no host row moved during the drafts
    assert all(d["host_rows_equal"]), d["host_rows_equal"]             # the edge wrote kept rows only
    assert all(d["logits_equal"]) and d["diff"] == [], (d["logits_equal"], d["diff"])


@pytest.mark.parametrize("key", sorted(DRAFT_EDGE))
def test_negctl_no_rollback_leaks_draft_rows_at_the_edge(key):
    """PLANT no_rollback = the Task 1.1 draft flow of dr1: the draft rows stay in the tail and the plain edge step
    writes them to the host."""
    n = _draft_edge_run(DRAFT_EDGE[key], neg="no_rollback")
    assert n["edge"] and not n["draft_fill"] and all(n["drafts_differ"])    # preconditions (as the claim's)
    assert n["host_moved_in_drafts"] == []                             # the drafts themselves wrote no host row
    assert not all(n["host_rows_equal"])                               # CAUGHT: draft rows reached the host rows
    assert any(name.endswith("_k_cpu") for name in n["diff"]), n["diff"]


@pytest.mark.parametrize("key", sorted(EDGE))
def test_negctl_no_defer_writes_host_rows(key):
    S_list, groups = EDGE[key]
    # precondition (L34): the rows a fill writes differ from the host rows before it (the poison past S is NaN; the
    # filled block holds the decode rows), so a write-back inside the group is visible
    (m, x) = RW.build(S_list)
    assert all(bool(torch.isnan(x.cache.layers[0].cache_engine._k_cpu[b, S]).all()) for b, S in enumerate(S_list))
    with pytest.raises(AssertionError, match="host rows of block"):
        _edge_run(S_list, groups, neg="no_defer")


@pytest.mark.parametrize("key", sorted(EDGE))
def test_negctl_no_redirect_reads_stale_host_rows(key):
    S_list, groups = EDGE[key]
    ev = _edge_run(S_list, groups)
    assert sum(ev["stats"]["redirected"]) > 0                          # precondition: the staged block was loaded
    with pytest.raises(AssertionError, match="kept logits differ|the state differs"):
        _edge_run(S_list, groups, neg="no_redirect")


# ------------------------------------------------------------------ FREE
def _deferred_run(S_list, plant):
    """World x: the rollback with DeferredStream (and the plant skip_release_wait); world y: the immediate executor.
    The same calls: W0 plain steps, group 1 (K steps, commit), group 2 (one step, commit); then (no plant) one plain
    step. The plant's state is compared before that step: a corrupt state may break the stand-ins' own checks."""
    (mx, x), (my, y) = RW.build(S_list), RW.build(S_list)
    ex = RW.DeferredStream()
    rbx = _rb(x, stream=ex, neg="skip_release_wait" if plant else None)
    rby = _rb(y)
    tok = RW.tokens(0, len(S_list))
    dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
    dx.plain(W0)
    dy.plain(W0)
    B = len(S_list)
    n1 = torch.tensor([0, 2, 1, 3, 0, 1, 2, 0][:B], dtype=torch.int64)    # request 0 rejects step 0
    n2 = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1][:B], dtype=torch.int64)
    dx.group(rbx, n1)
    dy.group(rby, n1)
    ev = dict(pending_after_commit=ex.pending(), events=list(ex.events))
    j0 = rbx.J[0][0]["ctr"].clone()
    rbx.begin()
    rby.begin()
    dx.step(0)
    dy.step(0)
    ev.update(done_after_step=[e.done for e in ev["events"]], ran_by=[e.ran_by for e in ev["events"]],
              journal_overwritten=not torch.equal(j0, rbx.J[0][0]["ctr"]))
    rbx.commit(n2.clone(), n2.clone())
    rby.commit(n2.clone(), n2.clone())
    dx.k += n2.clamp(0, 1)
    dy.k += n2.clamp(0, 1)
    ex.drain()
    ev["diff"] = RW.diff(RW.state(x), RW.state(y))
    if not plant:
        lx, ly = dx.plain(1)[0], dy.plain(1)[0]
        ex.drain()
        ev["diff_after_step"] = RW.diff(RW.state(x), RW.state(y))
        ev["logits_equal"] = torch.equal(W.bits(lx), W.bits(ly))
    return ev


@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_rollback_frees_after_last_reader(S_list):
    ev = _deferred_run(S_list, plant=False)
    assert ev["pending_after_commit"] == L                             # precondition: the commit's work was deferred
    assert ev["ran_by"] == ["wait"] * L                                # the next step waited every layer's event
    assert ev["diff"] == [] and ev["diff_after_step"] == [] and ev["logits_equal"], ev
    n = _deferred_run(S_list, plant=True)                              # PLANT: the next step does not wait
    assert n["pending_after_commit"] == L and n["done_after_step"] == [False] * L   # the step ran first
    assert n["journal_overwritten"]                                    # and overwrote the journal the undo reads
    assert n["diff"] != []                                             # CAUGHT


# ------------------------------------------------------------------ SYNC
def test_no_host_sync_in_commit_path():
    src = Path(RR.__file__).read_text()
    found, hits = RW.scan_source(src, RR.COMMIT_PATH)
    assert found == sorted(RR.COMMIT_PATH), set(RR.COMMIT_PATH) - set(found)
    assert hits == [], hits                                            # 0 host syncs by the static scan
    old = "        nacc = n_acc.to(torch.int64).clamp(0, n_run)\n"
    assert src.count(old) == 1
    planted = src.replace(old, old + "        _ = int(nacc.max())\n")
    ph = RW.scan_source(planted, RR.COMMIT_PATH)[1]
    assert [h[0] for h in ph] == ["commit"] and ph[0][2] == "int()", ph   # precondition: the scan sees a planted sync
    for S_list in (S1, S8):
        m, rs = RW.build(S_list)
        rb = _rb(rs)
        d = RW.Driver(m, rs, S_list, RW.tokens(0, len(S_list)))
        d.plain(2)
        n_acc = torch.arange(len(S_list), dtype=torch.int64) % (K + 1)
        with RW.SyncTrap(RR.__file__) as trap:
            d.group(rb, n_acc)
            rb.begin()
            d.step(0)
            rb.reject_all()
            d.plain(1)
        assert trap.hits == [], trap.hits                              # 0 tensor-to-host reads from the module
        m2, rs2 = RW.build(S_list)
        rb2 = _rb(rs2, neg="host_sync")
        d2 = RW.Driver(m2, rs2, S_list, RW.tokens(0, len(S_list)))
        with RW.SyncTrap(RR.__file__) as trap2:                         # PLANT: one host read of n_acc in commit
            d2.group(rb2, n_acc)
        assert any(h[0] == "_neg_host_sync" for h in trap2.hits), trap2.hits   # the trap is not blind


# ------------------------------------------------------------------ OVERFLOW
@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_journal_overflow_falls_back_to_host(S_list):
    ev = _reject_run(S_list, capacity=1)
    assert sum(ev["stats"]["overflow"]) > 0 and ev["facts"]["unsaved_live"] > 0, (ev["stats"], ev["facts"])
    assert RW.diff(ev["post"], ev["snap"], skip=RW.SLOTS) == []        # every other tensor and the clock: exact
    assert RW.live_slot_diff(ev["post"], ev["snap"]) == []             # the live slot bytes: exact
    for i in range(3):
        lx, ly = ev["dx"].plain(1)[0], ev["dy"].plain(1)[0]
        assert torch.equal(W.bits(lx), W.bits(ly)), "step %d after the overflowed rollback differs" % i
    n = _reject_run(S_list, capacity=1, neg="no_host_fallback")       # PLANT: no host fallback
    assert n["facts"]["unsaved_live"] > 0                              # precondition: a lost live block was unsaved
    assert RW.diff(n["post"], n["snap"], skip=RW.SLOTS) == []
    assert RW.live_slot_diff(n["post"], n["snap"]) != []               # CAUGHT


# ------------------------------------------------------------------ FUSED (Triton interpreter, fresh process)
FUSED_CHILD = r'''
import importlib.util, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton was imported before the interpreter flag was read"
HERE = sys.argv[1]
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("test_rollback", os.path.join(HERE, "test_rollback.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
FB = T.W.pkg_load("r2_body_fused")
assert FB.triton is not None, "child: triton did not import"
ev = T._partial_run([4659, 4658, 4657, 4723, 4722, 4630], [[0, 1, 3, 1, 2, 3], [3, 2, 0, 1, 3, 1]], body="fused")
assert sum(ev["stats"]["redirected"]) > 0 and sum(ev["stats"]["overflow"]) == 0, ev["stats"]
print("RBF OK the rollback with FusedBody == the reference with the torch body, bitwise: %s" % (ev["stats"],))
'''


def test_fused_body_rollback_under_the_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1", NOSI_REPO=str(ROOT))
    r = subprocess.run([sys.executable, "-c", FUSED_CHILD, str(HERE)], env=env, cwd=str(ROOT), capture_output=True,
                       text=True, timeout=3000)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith("RBF OK"), (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-6000:]))


# ------------------------------------------------------------------ CELL
def test_rollback_cell_arguments_schedule_compare_verdicts(monkeypatch):
    BC = W.bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--mbase", "--rollback"])
    assert (a.rollback, a.rb_k, a.rb_groups, a.rb_seed, a.rb_journal_blocks) == (True, 3, 128, 0, 256)
    assert BC.parse_args(base + ["--arms", "SH", "--rollback", "--rb-groups", "169"]).rb_groups == 169   # 170 x 3 <= 512
    for bad in (["--rollback"], ["--arms", "SH", "--rollback", "--rb-k", "0"], ["--arms", "SH", "--rollback", "--rb-k",
                                                                                "65"],
                ["--arms", "SH", "--rollback", "--rb-groups", "170"], ["--arms", "SH", "--rollback", "--rb-journal-blocks",
                                                                         "0"]):
        with pytest.raises(SystemExit):
            BC.parse_args(base + bad)
    n_acc, kc = BC.rollback_schedule(216, 128, 3, 0)
    assert tuple(n_acc.shape) == (128, 216) and int(n_acc.min()) == 0 and int(n_acc.max()) == 3
    assert torch.equal(n_acc, BC.rollback_schedule(216, 128, 3, 0)[0])
    assert not torch.equal(n_acc, BC.rollback_schedule(216, 128, 3, 1)[0])
    assert int(kc[0].abs().sum()) == 0 and torch.equal(kc[-1], n_acc.sum(0)) and int(kc.max()) + 3 <= 512
    g = torch.Generator().manual_seed(0)
    G, Kk, Bn = 3, 3, 4
    ref = (torch.randn(G, Kk, Bn, generator=g), torch.randn(G, Kk, Bn, generator=g, dtype=torch.float64),
           torch.randint(0, 9, (G, Kk, Bn), generator=g))
    na = torch.tensor([[0, 1, 2, 3], [3, 3, 0, 0], [1, 0, 2, 1]])
    n_kept = int((torch.arange(Kk).view(1, Kk, 1) < na.view(G, 1, Bn)).sum())
    assert BC.rollback_compare(ref, tuple(x.clone() for x in ref), na) == dict(
        n_checked=n_kept, n_unequal=0, equal_fraction=1.0, first=None, groups_unequal=0)
    got = [x.clone() for x in ref]
    got[0][1, 2, 0] = torch.nextafter(got[0][1, 2, 0], torch.tensor(float("inf")))   # kept: u 2 < n_acc[1, 0] = 3
    got[1][0, 2, 0] += 1.0                                                           # not kept: n_acc[0, 0] = 0
    c = BC.rollback_compare(ref, tuple(got), na)
    assert c["n_unequal"] == 1 and c["first"] == (1, 2, 0) and c["groups_unequal"] == 1
    inputs = torch.arange(Bn * 40).view(Bn, 40)
    kc2 = torch.zeros((G + 1, Bn), dtype=torch.int64)
    kc2[1:] = torch.cumsum(na, 0)
    arg = torch.stack([torch.stack([inputs[torch.arange(Bn), kc2[gg] + u + 1] for u in range(Kk)]) for gg in range(G)])
    assert BC.rollback_agreement(arg, inputs, (na, kc2)) == dict(n_kept=n_kept, agreement=1.0)
    rb = {"compare": c, "RBR": dict(steady=dict(median_ms=60.0)),
          "RBS": dict(steady=dict(median_ms=66.0), commit=dict(median_ms=5.0, mean_ms=5.5, p90_ms=7.0, n=10),
                      journal=dict(overflow=[0, 0], destroyed=[10, 12]))}
    sh = dict(rates=dict(steady_step_ms=64.0))
    for key in ("RB_bitwise", "RB_step_ratio", "RB_commit_ms", "RB_overflow_zero"):   # the builder's state: TBD
        monkeypatch.setitem(BC.PRED_ROLLBACK, key, None)
    v = {n: x for n, x, _ in BC.verdicts_rollback(sh, rb)}
    assert v == {"RB-BIT": "REPORT", "RB-STEP": "REPORT", "RB-COMMIT": "REPORT", "RB-OVF": "REPORT"}, v
    for key, val in (("RB_bitwise", True), ("RB_step_ratio", (1.0, 1.2)), ("RB_commit_ms", (1.0, 4.0)),
                     ("RB_overflow_zero", True)):
        monkeypatch.setitem(BC.PRED_ROLLBACK, key, val)
    v = {n: x for n, x, _ in BC.verdicts_rollback(sh, rb)}
    assert v == {"RB-BIT": "REFUTED", "RB-STEP": "CONFIRMED", "RB-COMMIT": "REFUTED", "RB-OVF": "CONFIRMED"}, v
    assert BC.verdicts_rollback(sh, {})[0][1] == "NOT_RUN"


def test_accept_perreq():
    v = torch.tensor([[5, 6, 7, 8], [5, 9, 7, 8], [1, 2, 3, 4]])
    d = torch.tensor([[5, 6, 7], [5, 6, 7], [0, 2, 3]])
    n_acc, keep, committed, nxt = RR.accept_perreq(v, d)
    assert n_acc.tolist() == [3, 1, 0] and keep.tolist() == [4, 2, 1]
    assert committed.tolist() == [[5, 6, 7, 8], [5, 9, -1, -1], [1, -1, -1, -1]] and nxt.view(-1).tolist() == [8, 9, 1]
    with RW.SyncTrap(RR.__file__) as trap:
        RR.accept_perreq(v, d)
    assert trap.hits == []


def test_refusals():
    m, rs = RW.build(S1)
    for kw in (dict(neg="nope"), dict(capacity=0)):
        with pytest.raises(RR.RollbackError):
            RR.Rollback(rs, K, **kw)
    with pytest.raises(RR.RollbackError):
        RR.Rollback(rs, 65)
    rb = _rb(rs)
    with pytest.raises(RR.RollbackError):
        _rb(rs)                                                        # one Rollback per RaggedState
    with pytest.raises(RR.RollbackError):
        rb.commit(torch.zeros(1, dtype=torch.int64))                   # outside a group
    d = RW.Driver(m, rs, S1, RW.tokens(0, 1))
    rb.begin()
    with pytest.raises(RR.RollbackError):
        rb.begin()
    for u in range(K):
        d.step(u)
    with pytest.raises(RR.RollbackError):
        d.step(K)                                                      # more than K tentative steps
    with pytest.raises(RR.RollbackError):
        rb.commit(torch.zeros(2, dtype=torch.int64))                   # a wrong shape
    with pytest.raises(RR.RollbackError):
        rb.detach()                                                    # inside a group


# ------------------------------------------------------------------ the step script
def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_base_rollback_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_rb1():
    for bad in ("rb2", "mb1", "rg1"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    for cell, gemm in (("rb1", "natural"), ("rb1x", "exact")):
        r = _step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C 128 (pool 65) B 216 arms SH+RBR+RBS steps 512 gemm %s K 3 groups 128 STEP_MIN 75 mem 240G"
                % (cell, gemm))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=65", "--pool 65", "--batch 216", "--mbase", "--rollback --rb-k 3 --rb-groups 128",
                     "--rb-seed 0 --rb-journal-blocks 256", "--ragged-twin torch --ragged-body fused",
                     "--ragged-gemm %s" % gemm, "--arms SH", "--mem=240G", "--time=75", "rb_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
