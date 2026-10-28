"""CPU tests of Phase 2, rungs L3 and L4 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs): the speculative round runner (nosi/nosi/spec_round.py; r2_spec_cell.py --spec K; r2_spec_step.sh cells
sp3m / sp3a / sp3b) and the Phase 2 additions of ragged_rollback.py (split journal, truncate, FAST journal) and
verify_rows.py (draft_last, hooks). The world: tests_cpu/spec_world.py over the worlds of the four hooks (the SHIPPED
engine from its own source, every CUDA op a torch stand-in with the kernel's flat addresses, L28; nosa_linear's
stand-in per row). B 1 and B 8 (the largest CPU batch) in the claims. No CUDA, no Triton launch.

GATE 0   test_flags_off_is_shipped: nosi/nosi and dependencies/ are the bytes of the merged hook trees (3a493d2) except
         spec_round.py (new) and the declared Phase 2 additions of ragged_rollback.py / verify_rows.py; no other module
         names spec_round; the cell refuses --spec 0; a SpecRunner attached (L3 and L4) and no round leaves every plain
         step bitwise the plain world's (logits and state, through a block fill); detach() puts every callable back.
         Negative control: one committed DRAFT step in the attached world (precondition: it skipped host loads).
TOKENS   test_spec_tokens_equal_shipped_greedy (L3 / L4 EXACT, B 1 / B 8): every request's committed tokens are the
         SHIPPED greedy decode (Llama.decode_inference, re-warmed), and every committed row's logits are the shipped
         step's, bitwise; after every round the residency invariants and the host rows hold. Preconditions: rejects and
         full accepts, a block edge, skipped host loads, pool moves, (B 8) ragged positions.
OVERFLOW test_tiny_journal_equals_shipped (L3 / L4 / L4P EXACT, B 8; the GPU cell sp3b): the same claims with a journal of
         1 slot per layer and step, and the residency right after every reject_all (L4: shipped-token drafts in every
         round, full accepts only). Preconditions: overflow, and a pool slot that a MOVE_IN emptied and a later step of
         the group overwrote by a MOVE_OUT with another block, unsaved. Negative control
         undo_move_in_none (the rollback's undo table of 6de1cfd): CAUGHT.
         test_draft_group_tiny_journal_restores_live_slots (no runner): one group of 6 draft steps, 1-slot journal,
         reject_all: the snapshot's maps and live slot bytes, the next plain steps bitwise. The same plant: CAUGHT.
REJECT   test_reject_keeps_accepted_prefix_and_correction (L3 / L4, B 8): a reject commits the accepted drafts and the
         target's correction (m = n + 1). Negative controls commit_draft_at_reject and keep_rejected: CAUGHT.
FULL     test_full_accept_no_bonus_token (L3 / L4, B 8): a full accept commits exactly K tokens, the drafts. Negative
         control bonus_on_full_accept (L4: the draft row's token counted): CAUGHT.
HOST     test_no_draft_row_reaches_host (L3 / L4, B 1 / B 8): after every round, the blocks completed during the decode
         are the shipped run's host rows and every host row from the committed block on is still the poison (the dr1
         leak cannot happen). Precondition: a block filled by a step that its round does not keep. Negative control:
         the rollback's no_defer: CAUGHT.
COUNTS   test_accept_counts_reported (L3 / L4, B 8): the device counters equal a host recount of the round records.
         Negative control miscount: CAUGHT.
TRUNCATE test_truncate_then_commit_equals_reference (B 1 / B 8): truncate inside a group, more steps, commit: the state
         and the kept logits equal a world that ran only the kept steps. Negative control no_truncate_mark: CAUGHT.
FAST     test_fast_rounds_keep_residency (L4 FAST, B 1 / B 8): the FAST joint pass under the rollback keeps the
         residency, host-row and twin invariants after every round; a joint pass without the FAST hooks is REFUSED.
DRAFT    test_draft_resident_skips_and_masks: the draft step skips the host loads, masks them in the map and in a
         scratch bias, never writes the live bias, restores the attended view; reject_all gives the snapshot back.
WIRING   test_prefetch_wiring (rung L5 hook, off in the cells); test_no_host_sync_in_round (only run() and the host
         redirect flag read host values; the rollback none).
CELL     test_cell_arguments_and_verdicts, test_step_script_sp3. Tree nosi-r2-specp (rung L4P): the verdicts include
         the L4P keys; the step script has the fifth arm L4P, STEP_MIN 30 / 100 / 140, --spec-pin off and this tree as
         ROOT. The L4P claims are in tests_cpu/test_spec_paper.py.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_round.py
"""
from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-specp"          # r2-specp: the step script's ROOT is this tree (rung L4P)
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
INTEG_BASE = "3a493d29da11193712c4a0cc9a887f8479c90eda"      # the merge of the four hook trees (before Phase 2)


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SW = _load("spec_world")
W, RW, VW, RE, RR, SR, BC = SW.W, SW.RW, SW.VW, SW.RE, SW.RR, SW.SR, SW.BC
K, N, L = SW.K0, SW.N0, SW.L
S8 = [4659, 4658, 4657, 4723, 4722, 4721, 4630, 4691]          # test_rollback's: fills within 16 plain steps
S1 = [4658]
SM8 = [4910, 4911, 4930, 4900, 4925, 4950, 4905, 4915]           # 76-77 complete blocks: misses after the warm-up


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_spec_cell
    return r2_spec_cell


# THE FILTERS. The proof holds for any draft tokens; these force rejects at known places. Request b's draft 1 is wrong
# in the rounds g with (g + b) % 4 == 2 (the L4 prelude is round -1: request 0's first two rounds keep its natural
# drafts), so every request has rounds with its natural drafts (full accepts are possible) and a forced reject.
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)
S_EDGE = 4925                          # S % 64 = 61: after the warm step the first round starts 1 position before the
#                                        block edge (position 4927), so a draft (L3) or a joint-pass row (L4) fills it


def _ev(level, B, neg=None, rb_neg=None, filt="periodic", mode="exact", S=SW.S0):
    """One run_vs_shipped per parameter set (shared by the claims of this module; the key is normalised, so a keyword
    and a default give the same run)."""
    return _ev_cached(level, B, neg, rb_neg, filt, mode, S)


@functools.lru_cache(maxsize=None)
def _ev_cached(level, B, neg, rb_neg, filt, mode, S):
    f = dict(periodic=FILTER, edge=SW.all_kinds(SW.corrupt_before_edge(K), FILTER),
             row=SW.corrupt(1, 0, kinds=("draft_row",)), none=None)[filt]
    gemm = "rowgroup" if mode == "exact" else "natural"
    plant = neg is not None or rb_neg is not None
    return SW.run_vs_shipped(B, level=level, mode=mode, gemm=gemm, draft_filter=f, neg=neg, rb_neg=rb_neg, S=S,
                             max_rounds=(12 if plant else 0))


def _clean(ev):
    return ev["bad_tokens"] == [] and ev["bad_logits"] == []


# ------------------------------------------------------------------ GATE 0
@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_flags_off_is_shipped(S_list):
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", INTEG_BASE + "^{commit}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", INTEG_BASE, "--", "nosi/nosi", "dependencies",
                            ":(exclude)nosi/nosi/spec_round.py", ":(exclude)nosi/nosi/ragged_rollback.py",
                            ":(exclude)nosi/nosi/spec_prefetch.py",  # rung L5 (Gate 0: test_spec_prefetch.py)
                            ":(exclude)nosi/nosi/spec_tentative.py", # WS-A (Gate 0: test_wsa.py)
                            ":(exclude)nosi/nosi/spec_tentative_union.py",   # I2 (Gate 0: test_i2.py)
                            ":(exclude)nosi/nosi/verify_rows.py", ":(exclude,glob)**/__pycache__/**"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "", "engine sources differ from %s:\n%s" % (INTEG_BASE,
                                                                                                    r.stdout)
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    assert "spec_round.py" in src
    src.pop("spec_tentative.py", None)        # WS-A: spec_tentative imports spec_round (opt-in; Gate 0: test_wsa.py)
    assert [n for n, t in src.items() if "spec_round" in t and n != "spec_round.py"] == []
    for f in ("r2_base_cell.py", "r2_verify_cell.py", "r2_poolpf_cell.py"):
        assert "spec_round" not in (EFF / f).read_text(), f
    assert _cell().main(["--out", "x", "--workload", "w"]) == 20         # --spec 0: the cell refuses (flag off)
    B = len(S_list)
    for level in ("L3", "L4"):
        (mx, x), (my, y) = RW.build(S_list), RW.build(S_list)
        assert RW.diff(RW.state(x), RW.state(y)) == [], "precondition: two worlds of one seed differ before step 0"
        orig = dict(y.ops.kernels)
        lin = y.ops.linear
        runner = SR.SpecRunner(my, y, SR.SpecOps(level=level, K=K, attn=VW.fake_attention_idx, journal=RW.CAP),
                               BC.HostClock())
        tok = RW.tokens(0, B)
        dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
        h0 = RW.state(x)
        for i in range(16):                                   # S1 fills at step 13, S8's first request at step 12
            lx, ly = dx.plain(1)[0], dy.plain(1)[0]
            assert torch.equal(W.bits(lx), W.bits(ly)), "%s step %d: the attached world's logits differ" % (level, i)
        assert RW.diff(RW.state(x), RW.state(y)) == []
        assert any(n.endswith("_k_cpu") for n in RW.diff(h0, RW.state(x))), "precondition: a block fill in 16 steps"
        runner.detach()
        assert "decode_fixed" not in y.body.__dict__ and y.rollback is None and y.draft is None
        assert all(y.ops.kernels[k] is v for k, v in orig.items()) and y.ops.linear is lin
        assert all("step_unchecked" not in t.__dict__ for t in y.st.twins)
    # NEGATIVE CONTROL: one DRAFT step committed in the attached world (after two plain steps load the window)
    (mx, x), (my, y) = RW.build(SM8[:B])[:2], RW.build(SM8[:B])[:2]
    runner = SR.SpecRunner(my, y, SR.SpecOps(level="L3", K=K, journal=RW.CAP), BC.HostClock())
    tok = RW.tokens(0, B)
    dx, dy = RW.Driver(mx, x, SM8[:B], tok), RW.Driver(my, y, SM8[:B], tok)
    dx.plain(2)
    dy.plain(2)
    caught = False
    for _ in range(6):                                        # until a draft step skips a host load
        runner.rb.begin()
        runner.draft.begin("all")
        ly = dy.step(0)
        runner.draft.end()
        runner.rb.commit(torch.ones(B, dtype=torch.int64))
        lx = dx.step(0)
        dx.k += 1
        dy.k += 1
        if int(runner.draft.skipped.sum()) > 0:
            caught = not torch.equal(W.bits(lx), W.bits(ly))
            break
    assert int(runner.draft.skipped.sum()) > 0, "precondition (L34): a draft step skipped a host load"
    assert caught, "the committed draft step was not seen"


# ------------------------------------------------------------------ TOKENS
@pytest.mark.parametrize("level,B", [("L3", 1), ("L3", 8), ("L4", 1), ("L4", 8)], ids=["L3-B1", "L3-B8", "L4-B1",
                                                                                         "L4-B8"])
def test_spec_tokens_equal_shipped_greedy(level, B):
    ev = _ev(level, B)
    f = ev["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f                    # preconditions (L34): both paths of a round ran
    assert f["edge_crossed"] and f["skipped"] > 0, f                 # a block edge; the drafts skipped host loads
    assert f["move_out"] + f["swap"] > 0, f                          # the victim pool moved blocks
    if B > 1:
        assert f["ragged"], f                                        # requests at different positions
    assert f["min_count"] >= N, f
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]              # THE CLAIM: the shipped greedy tokens
    assert ev["bad_logits"] == [], ev["bad_logits"][:5]              # and the shipped logits of every committed row
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])
    rd = ev["rd"]
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert sum(rd["journal"]["overflow"]) == 0


# ------------------------------------------------------------------ OVERFLOW (GPU cell sp3b at 6de1cfd)
# sp3b (C 73, B 318): the journal saturated (16.7 - 30.4 M live slots past N = 256 went to the host fallback) and the
# EXACT arms L3 / L4X lost the shipped tokens from column 1 on; sp3m (C 128, B 8, overflow 0) was bitwise. This world
# has the same regime with a journal of CAP_TINY slots: almost every lost slot of a step is past N.
CAP_TINY = 1


ORACLE_ALL = tuple(range(-1, N + 9))    # every round of a run (and the L4 prelude, round -1)


@functools.lru_cache(maxsize=None)
def _ev_tiny(level, B, rb_neg, seed=0):
    """run_vs_shipped with a journal of CAP_TINY slots and spec_world.watch_reuse (the sp3b pattern in every group
    before its undo; the residency right after every reject_all). L3: the periodic FILTER (rejects and full accepts).
    L4: the drafts are the shipped tokens in EVERY round (no filter), so every request accepts every round: in L4 only a
    full-accept request runs three skip-making draft steps in one group (the draft row, then 2 drafts; after a reject a
    request has K - 1 valid drafts and never accepts in full again), and the pattern needs three. L4P (r2-specp, the
    paper's schedule) as L3: the periodic FILTER; its groups have three or more draft residencies after a full accept
    (the deferred row, the draft row, the 2 drafts) and after a reject (the draft rows and pads, the 2 drafts). No round
    cap (the plant runs as many rounds as the claim run, so their groups are the same until the first harmful reuse)."""
    box = {}
    filt = level != "L4"                 # L3 / L4P: the periodic FILTER (rejects and full accepts); L4: shipped drafts
    ev = SW.run_vs_shipped(B, level=level, mode="exact", gemm="rowgroup", draft_filter=(FILTER if filt else None),
                           rb_neg=rb_neg, journal=CAP_TINY, on_runner=lambda r: SW.watch_reuse(r, box), seed=seed,
                           oracle_rounds=((-1, 0) if filt else ORACLE_ALL), max_rounds=0)
    ev["reuse"] = box
    return ev


def _tiny_seed(level, B=8, seeds=3):
    """The first world seed (0 .. seeds - 1) whose run has the sp3b pattern (move_in_reuse: unsaved > 0), and its
    evidence; the last seed's evidence if none has it (the precondition assert then fails with the facts)."""
    for seed in range(seeds):
        ev = _ev_tiny(level, B, None, seed)
        if ev["reuse"].get("unsaved", 0) > 0:
            break
    return seed, ev


@pytest.mark.parametrize("level", ["L3", "L4", "L4P"])
def test_tiny_journal_equals_shipped(level):
    """THE sp3b DIVERGENCE ON CPU. EXACT rounds with a saturated journal: the committed tokens and logits rows are the
    shipped greedy decode's, the residency holds after every reject_all and every round, the host rows hold.
    Preconditions (L34): the journal overflows (live and dead slots past N), and some group has a pool slot that a
    MOVE_IN emptied and a later step's MOVE_OUT overwrote, unsaved (spec_world.move_in_reuse)."""
    seed, ev = _tiny_seed(level)
    st, f, ru = ev["rd"]["journal"], ev["facts"], ev["reuse"]
    assert sum(st["overflow"]) > 0 and sum(st["overflow_dead"]) > 0, st
    assert ru.get("unsaved", 0) > 0, (seed, ru, f)
    assert f["skipped"] > 0 and f["full"] > 0 and f["min_count"] >= N, f
    if level != "L4":
        assert f["rejects"] > 0, f                                     # L4 here: full accepts only (_ev_tiny)
    got = (ev["reuse"].get("resid_reject", [])[:2], ev["resid"][:2], ev["bad_tokens"][:5], ev["bad_logits"][:5],
           ev["host"][:2], ev["host_c"][:2])                          # host_c: L4P's committed-length host check
    assert got == ([], [], [], [], [], []), got
    # PLANT undo_move_in_none: the inverse of MOVE_IN is no move (the table of 6de1cfd, the sp3b code)
    p = _ev_tiny(level, 8, "undo_move_in_none", seed)
    assert p["reuse"].get("unsaved", 0) > 0, (seed, p["reuse"])        # precondition (L34) of the plant run
    caught = (p["reuse"].get("resid_reject", []), p["resid"], p["bad_tokens"], p["bad_logits"])
    assert caught != ([], [], [], []), "undo_move_in_none was not caught"     # CAUGHT


def _draft_group_run(seed, neg=None, steps=6):
    """THE ROOT CAUSE WITHOUT THE RUNNER. World x: 4 plain steps (the window is loaded), ONE rollback group of `steps`
    DRAFT steps (DraftResident "all") with a journal of CAP_TINY slots, reject_all. World y: the same 4 plain steps, no
    group. Returns the facts of the group (spec_world.move_in_reuse, before the undo), the journal counters, the
    non-slot differences from the snapshot, the LIVE slot differences (rollback_world.live_slot_diff: the window and pool
    slots that the maps name) and whether the next 3 plain steps of x and y are bitwise equal."""
    (mx, x), (my, y) = RW.build(SM8, horizon=128, seed=seed), RW.build(SM8, horizon=128, seed=seed)
    B = len(SM8)
    rb = RR.Rollback(x, steps, capacity=CAP_TINY, neg=neg)
    ds = SR.DraftResident(x, rb)
    tok = RW.tokens(seed, B, n=128)
    dx, dy = RW.Driver(mx, x, SM8, tok), RW.Driver(my, y, SM8, tok)
    dx.plain(4)
    dy.plain(4)
    snap = RW.state(x)
    rb.begin()
    for u in range(steps):
        ds.begin("all")
        dx.step(u)
        ds.end()
    facts = SW.move_in_reuse(rb, torch.zeros(B, dtype=torch.int64))
    rb.reject_all()
    post = RW.state(x)
    ev = dict(facts=facts, stats=rb.stats(), other=RW.diff(post, snap, skip=RW.SLOTS))
    ev["live"] = RW.live_slot_diff(post, snap) if ev["other"] == [] else None
    ev["after"] = [torch.equal(W.bits(dx.plain(1)[0]), W.bits(dy.plain(1)[0])) for _ in range(3)]
    return ev


def test_draft_group_tiny_journal_restores_live_slots():
    """One group of 6 draft steps with a 1-slot journal, then reject_all: every map, table, counter and the clock are
    the snapshot's, every LIVE slot (named by a map) holds its snapshot bytes, and the next 3 plain steps equal a world
    that never ran the group. Precondition (L34): the group has the sp3b pattern (a pool slot that a MOVE_IN emptied and
    a later draft step overwrote by a MOVE_OUT, unsaved); the first world seed in 0 .. 2 with it. Negative control
    undo_move_in_none (the undo table of 6de1cfd) on the same seed: CAUGHT by the live slots."""
    for seed in range(3):
        ev = _draft_group_run(seed)
        if ev["facts"]["unsaved"] > 0:
            break
    assert ev["facts"]["unsaved"] > 0 and sum(ev["stats"]["overflow_dead"]) > 0, (seed, ev["facts"], ev["stats"])
    assert ev["other"] == [] and ev["live"] == [] and all(ev["after"]), (seed, ev["other"][:5], ev["live"], ev["after"])
    p = _draft_group_run(seed, neg="undo_move_in_none")
    assert p["facts"]["unsaved"] > 0, (seed, p["facts"])               # precondition (L34) of the plant run
    assert p["other"] == [], p["other"][:5]                              # the maps are right; only bytes are wrong
    assert p["live"] != [], "undo_move_in_none was not caught"          # CAUGHT


# ------------------------------------------------------------------ REJECT
@pytest.mark.parametrize("level", ["L3", "L4"])
def test_reject_keeps_accepted_prefix_and_correction(level):
    ev = _ev(level, 8)
    real = 0
    for r in ev["records"]:
        for b in range(8):
            n, v, m = int(r["n"][b]), int(r["v"][b]), int(r["m"][b])
            Db, Gb, Cb = r["D"][b], r["G"][b], r["committed"][b]
            assert m == min(n + 1, K), (n, m)
            assert torch.equal(Cb[:n], Db[:n]), "the accepted prefix is not the drafts"
            if n < K:
                assert int(Cb[n]) == int(Gb[n]), "the token after the accepted prefix is not the target's"
                if n < v:
                    assert int(Gb[n]) != int(Db[n])
                    real += 1
    assert real > 0                                                   # precondition: a reject of a valid draft
    assert _clean(ev)
    for neg in ("commit_draft_at_reject", "keep_rejected"):
        p = _ev(level, 8, neg=neg)
        assert p["facts"]["rejects"] > 0 and p["facts"]["partial_keep"] > 0, (neg, p["facts"])   # preconditions
        assert not _clean(p), "%s was not caught" % neg               # CAUGHT by the token / logits check


# ------------------------------------------------------------------ FULL
@pytest.mark.parametrize("level", ["L3", "L4"])
def test_full_accept_no_bonus_token(level):
    ev = _ev(level, 8)
    fulls = 0
    for r in ev["records"]:
        for b in range(8):
            if bool(r["full"][b]):
                assert int(r["m"][b]) == K and int(r["n"][b]) == K
                assert torch.equal(r["committed"][b], r["D"][b]), "a full accept commits the K drafts"
                fulls += 1
    assert fulls > 0                                                  # precondition: full accepts happened
    assert _clean(ev)
    if level == "L4":
        p = _ev("L4", 8, neg="bonus_on_full_accept", filt="row")      # every draft row's token is wrong
        assert p["facts"]["full"] > 0, p["facts"]                     # precondition: a full accept (its bonus is wrong)
        assert not _clean(p), "bonus_on_full_accept was not caught"   # CAUGHT


# ------------------------------------------------------------------ HOST
def _unkept(ev):
    """Fills of steps that their round does not keep, and that no kept step of the same round fills again."""
    return [f for f in SW.unkept_fills(ev) if f[2] >= int(ev["records"][f[0]]["m"][f[1]])]


@pytest.mark.parametrize("level,B", [("L3", 1), ("L3", 8), ("L4", 1), ("L4", 8)], ids=["L3-B1", "L3-B8", "L4-B1",
                                                                                         "L4-B8"])
def test_no_draft_row_reaches_host(level, B):
    ev = _ev(level, B, filt="edge", S=S_EDGE)
    assert _unkept(ev), "precondition (L34): a block filled by a step that its round does not keep"
    assert ev["facts"]["edge_crossed"]
    assert ev["host"] == [], ev["host"][:3]                           # THE CLAIM, after every round
    assert ev["resid"] == [] and _clean(ev)
    p = _ev(level, B, rb_neg="no_defer", filt="edge", S=S_EDGE)       # PLANT: a fill goes to the host inside a group
    assert _unkept(p), "precondition (L34) of the plant run"
    assert p["host"] != [], "no_defer was not caught"                 # CAUGHT


# ------------------------------------------------------------------ COUNTS
@pytest.mark.parametrize("level", ["L3", "L4"])
def test_accept_counts_reported(level):
    for neg in (None, "miscount"):
        ev = _ev(level, 8, neg=neg)
        rd, recs = ev["rd"], ev["records"]
        hist = [0] * (K + 1)
        for r in recs:
            for x in r["n"].tolist():
                hist[x] += 1
        host = dict(valid=sum(int(r["v"].sum()) for r in recs), accepted=sum(int(r["n"].sum()) for r in recs),
                    committed=sum(int(r["m"].sum()) for r in recs),
                    full=sum(int((r["n"] >= r["v"]).sum()) for r in recs))
        got = dict(valid=rd["valid"], accepted=rd["accepted"], committed=rd["committed"], full=rd["full"])
        assert rd["acc_hist"] == hist and rd["round_committed"] == [int(r["m"].sum()) for r in recs]
        assert sum(rd["count"]) == host["committed"] + rd["warm"] * 8
        if neg is None:
            assert got == host, (got, host)
        else:
            assert any(bool((r["m"] != r["n"]).any()) for r in recs)   # precondition: a round where m != n
            assert got != host, "miscount was not caught"                # CAUGHT


# ------------------------------------------------------------------ TRUNCATE
W0 = 12                                # test_rollback's warm-up: the first group starts at p % 64 = 63, 62, 61, ...


def _truncate_run(S_list, neg=None):
    """World x: begin, 3 steps (the steps that x truncates read WRONG tokens), truncate(keep1), 2 more steps (wrong
    tokens: they are undone), commit(nfin); world r: the reference that runs only the kept steps (kept = 2 for a
    request with keep1 = 3, else keep1) over 5 masked steps (steps 3 and 4 have no active request: they advance the LRU
    clock by topk each, as x's 5 tentative steps do; commit keeps the clock). The host position mirror goes through
    truncate / commit (their host copies). Returns the evidence."""
    (mx, x), (mr, r) = RW.build(S_list), RW.build(S_list)
    assert RW.diff(RW.state(x), RW.state(r)) == [], "precondition: two worlds of one seed differ before step 0"
    rb = RR.Rollback(x, 6, capacity=RW.CAP, neg=neg)
    B = len(S_list)
    tok = RW.tokens(0, B)
    dx, dr = RW.Driver(mx, x, S_list, tok), RW.Driver(mr, r, S_list, tok)
    dx.plain(W0)
    dr.plain(W0)
    # request 0 truncates all 3 steps (B 1: the plant then has steps to undo twice); B 8: fills at u 0 / 1 / 2 (S8 after
    # W0) meet kept and truncated steps, and a fill that truncate keeps for a truncated request (request 3) stays staged
    # until the commit
    keep1 = torch.tensor([0, 3, 1, 2, 3, 1, 2, 0][:B], dtype=torch.int64)
    kept = torch.where(keep1 == 3, torch.full_like(keep1, 2), keep1)
    rb.begin()
    lx = [dx.step(u, wrong=(u >= kept)) for u in range(3)]
    facts = RW.journal_facts(rb, keep1)                              # the steps that truncate undoes (u >= keep1)
    rb.truncate(keep1, keep1.clone())
    for u in range(3, 5):
        dx.step(u, wrong=torch.ones(B, dtype=torch.bool))
    nfin = torch.where(keep1 == 3, torch.full_like(keep1, 2), torch.full_like(keep1, 3))
    rb.commit(nfin, nfin.clone())
    dx.k += kept
    lr = dr.masked(kept, steps=5)
    ev = dict(bad_logits=RW.kept_logits_equal(lx, lr, kept), diff=RW.diff(RW.state(x), RW.state(r), skip=RW.SCRATCH),
              facts=facts, fills=RW.fills(S_list, dr.k - kept, kept), stats=rb.stats(), truncations=rb.truncations)
    if not ev["diff"]:
        after = []
        for _ in range(3):
            a, c = dx.plain(1)[0], dr.plain(1)[0]
            after.append(torch.equal(W.bits(a), W.bits(c)))
        ev["after"] = after
        ev["diff_after"] = RW.diff(RW.state(x), RW.state(r), skip=RW.SCRATCH)
    return ev


def _truncate_reject_run(S_list, neg=None):
    """World x: begin, 3 steps (the truncated ones read WRONG tokens), truncate(keep1), 2 more steps, then reject_all:
    the snapshot before the group must come back bit for bit (the clock too). Here the TRUNCATE MARK matters: the final
    keep count (0) is below every truncated step, so without the mark reject_all undoes those steps a second time. World
    y never runs the group (the next steps must equal). Returns the evidence."""
    (mx, x), (my, y) = RW.build(S_list), RW.build(S_list)
    assert RW.diff(RW.state(x), RW.state(y)) == [], "precondition: two worlds of one seed differ before step 0"
    rb = RR.Rollback(x, 6, capacity=RW.CAP, neg=neg)
    B = len(S_list)
    tok = RW.tokens(0, B)
    dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
    dx.plain(W0)
    dy.plain(W0)
    snap = RW.state(x)
    keep1 = torch.tensor([0, 3, 1, 2, 3, 1, 2, 0][:B], dtype=torch.int64)
    # host facts: (request, step) of a fill that truncate KEEPS for a TRUNCATED request (keep1 < 3); reject_all then
    # undoes that step, so its block must not reach the host rows (CPU job sp_20202f8_a, B 8: request 3, block 73)
    kept_by_truncate = [(b, u) for b, u, kept in RW.fills(S_list, dx.k, keep1) if kept and int(keep1[b]) < 3]
    rb.begin()
    for u in range(3):
        dx.step(u, wrong=(u >= keep1))
    facts = RW.journal_facts(rb, keep1)                              # the truncated (request, step) pairs
    pre = RW.state(x)
    rb.truncate(keep1, keep1.clone())
    for u in range(3, 5):
        dx.step(u, wrong=torch.ones(B, dtype=torch.bool))
    rb.reject_all()
    ev = dict(diff=RW.diff(RW.state(x), snap), facts=facts, changed=RW.diff(pre, snap), stats=rb.stats(),
              kept_by_truncate=kept_by_truncate)
    if not ev["diff"]:
        ev["after"] = [torch.equal(W.bits(dx.plain(1)[0]), W.bits(dy.plain(1)[0])) for _ in range(3)]
        ev["diff_after"] = RW.diff(RW.state(x), RW.state(y))
    return ev


@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_truncate_then_commit_equals_reference(S_list):
    ev = _truncate_run(S_list)
    assert ev["bad_logits"] == [] and ev["diff"] == [], (ev["bad_logits"], ev["diff"][:5])
    assert all(ev["after"]) and ev["diff_after"] == [], ev["diff_after"][:5]
    # preconditions: truncated steps moved slots by a SWAP or a MOVE_OUT (no window slot is empty in these groups, so
    # they have no MOVE_IN)
    assert ev["facts"]["lost"] > 0 and ev["facts"]["swap"] + ev["facts"]["move_out"] > 0, ev["facts"]
    if len(S_list) > 1:
        assert any(kept for _, _, kept in ev["fills"]) and any(not kept for _, _, kept in ev["fills"]), ev["fills"]
    assert sum(ev["stats"]["overflow"]) == 0 and ev["truncations"] == 1
    # THE TRUNCATE MARK. In the commit above (and in spec_round's L4 commit) every truncated request keeps 3 steps,
    # more than its truncated step indices, so commit never selects a truncated step again: the mark cannot act there
    # (CPU job sp_a6eb5dd_a: the plant no_truncate_mark gave no difference). It acts when a later commit's keep count is
    # at or below a truncated step: truncate, more steps, then reject_all.
    ev = _truncate_reject_run(S_list)
    if len(S_list) > 1:                                               # precondition (L34) of the host-row part: a fill
        assert ev["kept_by_truncate"], ev["kept_by_truncate"]        # that truncate keeps and reject_all undoes
    assert ev["diff"] == [], ev["diff"][:5]                           # the snapshot, bit for bit (host rows included)
    assert all(ev["after"]) and ev["diff_after"] == [], ev["diff_after"][:5]
    assert RW.in_family(ev["changed"], "slots") and RW.in_family(ev["changed"], "twin"), ev["changed"][:5]
    assert sum(ev["stats"]["overflow"]) == 0
    p = _truncate_reject_run(S_list, neg="no_truncate_mark")         # PLANT: reject_all undoes truncated steps again
    assert p["facts"]["swap"] + p["facts"]["move_out"] > 0, p["facts"]    # precondition (L34): a truncated step of a
    #                                                                       truncated request swapped / parked a block
    assert p["diff"] != [], "no_truncate_mark was not caught"         # CAUGHT


# ------------------------------------------------------------------ FAST
@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_fast_rounds_keep_residency(B):
    ev = _ev("L4", B, mode="fast")
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])
    rd = ev["rd"]
    assert rd["errors"] == (0, [0] * L) and sum(rd["journal"]["overflow"]) == 0
    assert ev["facts"]["rejects"] > 0 and ev["facts"]["edge_crossed"], ev["facts"]   # preconditions
    assert rd["verify"]["extras"] >= 0 and rd["verify"]["overflow"] == 0
    for l in range(L):                                                # the twin's counter invariants
        ev["runner"].rs.st.twins[l].check_invariants()
    # a FAST joint pass whose hooks are gone: the group's truncate REFUSES (no silent wrong journal)
    hz = SW.horizon_for(8, K)
    model, A, Bc = SW.two_caches(B, SW.S0, hz + 64)
    runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(B, dtype=torch.int64), level="L4", mode="fast",
                            gemm="natural")
    runner.vs.hooks = None
    with pytest.raises(RR.RollbackError, match="without its engine journal"):
        runner.run(8, max_rounds=2)


# ------------------------------------------------------------------ DRAFT
@pytest.mark.parametrize("S_list", [SM8[:1], SM8], ids=["B1", "B8"])
def test_draft_resident_skips_and_masks(S_list):
    m, x = RW.build(S_list, horizon=128)
    B = len(S_list)
    calls = []
    kern = x.ops.kernels
    base_k = kern["gather_k"]

    def counting(gpu, cpu, ids, bs):
        calls.append(int((ids >= 0).sum()))
        base_k(gpu, cpu, ids, bs)
    kern["gather_k"] = counting                                       # under the rollback: its "original" gather
    rb = RR.Rollback(x, 6, capacity=RW.CAP)
    ds = SR.DraftResident(x, rb)
    d = RW.Driver(m, x, S_list, RW.tokens(0, B, n=128))
    d.plain(4)                                                        # the window is loaded
    live = [lay.cache_engine._kv_bias_gpu_att for lay in x.cache.layers]
    snap = RW.state(x)
    rb.begin()
    seen = None
    for u in range(6):
        del calls[:]
        last0 = int(ds.skipped[L - 1])
        ds.begin("all")
        d.step(u)
        ds.end()
        if int(ds.skipped[L - 1]) > last0 and seen is None:           # the LAST layer skipped (the scratch holds it)
            e = x.cache.layers[L - 1].cache_engine
            skip = e._load_mask >= 0
            seen = dict(calls=sum(calls), miss=torch.equal(ds.miss_ids[L - 1], torch.where(skip, e._load_mask,
                                                                                             torch.full_like(
                                                                                                 e._load_mask, -1))),
                        map=bool((e._block_map[skip] == -1).all()), n=int(skip.sum()),
                        rows=skip.permute(1, 2, 0).unsqueeze(2).expand(B, W.TOPK, W.BS, W.HKV).reshape(
                            B, W.TOPK * W.BS, W.HKV),
                        scratch=ds.scratch_att.clone(), live=e._kv_bias_gpu_att.clone())
    assert all(lay.cache_engine._kv_bias_gpu_att is v for lay, v in zip(x.cache.layers, live))
    assert seen is not None, "precondition (L34): a draft step skipped a host load"
    assert seen["calls"] == 0, "a draft step ran a host gather"
    assert seen["miss"] and seen["map"] and seen["n"] > 0
    assert bool((seen["scratch"][seen["rows"]] == SR.MASK_BIAS).all())
    assert torch.equal(W.bits(seen["scratch"][~seen["rows"]]), W.bits(seen["live"][~seen["rows"]]))
    assert not bool((seen["live"][seen["rows"]] == SR.MASK_BIAS).all())   # the live bias never got the mask
    rb.reject_all()
    assert RW.diff(RW.state(x), snap) == []
    with pytest.raises(SR.SpecError):
        ds.begin("all")                                               # outside a group: refused (the dr1 leak)


# ------------------------------------------------------------------ WIRING
def test_prefetch_wiring():
    rec = SW.Recorder()
    hz = SW.horizon_for(8, K)
    model, A, Bc = SW.two_caches(1, SW.S0, hz + 64)
    runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(1, dtype=torch.int64), level="L3", prefetch=rec)
    runner.run(8, max_rounds=3)
    G = runner.rounds_run
    assert rec.rounds == ["begin", "end"] * G
    assert len(rec.drafts) == G * K and all(len(x) == L for x in rec.drafts)
    assert all(tuple(t.shape) == (W.HKV, 1, W.TOPK) for x in rec.drafts for t in x)
    assert SR.SpecOps().prefetch is None                              # off by default (the cells keep it off)


@pytest.mark.parametrize("level", ["L3", "L4"])
def test_no_host_sync_in_round(level):
    """Two runs (one trap each: a trap records only the calls whose DIRECT caller is its file, so two nested traps
    would hide each other)."""
    hz = SW.horizon_for(8, K)
    hits = []
    for path in (SR.__file__, RR.__file__):
        model, A, Bc = SW.two_caches(8, SW.S0, hz + 64)
        runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(8, dtype=torch.int64), level=level)
        with RW.SyncTrap(path) as trap:
            runner.run(8, max_rounds=3)
        assert runner.rounds_run >= 1
        hits.append(trap.hits)
    assert {h[0] for h in hits[0]} <= {"run", "_edge_before"}, sorted({h[0] for h in hits[0]})
    assert any(h[0] == "run" for h in hits[0])                        # the trap is not blind: run's per-round sync
    assert hits[1] == [], hits[1][:5]                                 # the rollback: none (truncate included)


# ------------------------------------------------------------------ CELL
def test_cell_arguments_and_verdicts(monkeypatch):
    SC = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = SC.parse_args(base + ["--spec", "3"])
    assert (a.spec, a.spec_arms, a.batch, a.steps, a.pool, a.spec_journal_blocks, a.spec_twin) == (
        3, ["L3", "L4X", "L4F"], 216, 512, 65, 256, "fused")             # the fused twin by default (SCORING_rg1.md)
    for bad in (["--spec", "1"], ["--spec", "19"], ["--spec", "3", "--spec-arms", "L5"], ["--spec", "3", "--spec-arms",
                                                                                        "L3,L3"]):
        with pytest.raises(SystemExit):
            SC.parse_args(base + bad)
    assert SC.main(base + ["--spec", "3", "--spec-prefetch", "on"]) == 20          # rung L5: refused
    assert SC.rounds_for(512, 3) == 171 and SC.rounds_for(512, 3, 40) == 40
    assert SC.steady_rounds(20, 16) == [g for g in range(1, 20) if g != 16]
    inputs = torch.arange(3 * 11).view(3, 11)
    out = torch.full((3, 14), -1, dtype=torch.int64)
    out[:, :10] = inputs[:, 1:11]
    tc = SC.token_check(out, [10, 10, 4], inputs, 10)
    assert tc == dict(n_checked=24, n_unequal=0, equal_fraction=1.0, first=None, min_checked=4)
    out[1, 6] += 1
    out[2, 7] += 1                                                    # past request 2's count: not compared
    tc = SC.token_check(out, [10, 10, 4], inputs, 10)
    assert tc["n_unequal"] == 1 and tc["first"] == (1, 6)
    g = torch.Generator().manual_seed(0)
    fm, fs = torch.randn(2, 6, generator=g), torch.randn(2, 6, generator=g, dtype=torch.float64)
    fc = SC.fingerprint_check((fm.clone(), fs.clone()), [6, 3], (fm, fs), 6)
    assert fc["n_unequal"] == 0 and fc["n_checked"] == 9
    fm2 = fm.clone()
    fm2[0, 2] = torch.nextafter(fm2[0, 2], torch.tensor(float("inf")))
    assert SC.fingerprint_check((fm2, fs.clone()), [6, 3], (fm, fs), 6)["first"] == (0, 2)
    rr = SC.round_rates([10, 12, 12, 9], [100.0, 100.0, 120.0, 90.0], [1, 2, 3], 1.0, 43)
    assert rr["median_tok_s"] == 100.0 and rr["mean_tok_s"] == round(1e3 * 33 / 310.0, 1) and rr["effective_tok_s"] == 43
    sh = dict(steady=dict(median_ms=60.0, mean_ms=62.0), rates=dict(steady_tok_s=1000.0, effective_tok_s=900.0))
    spr = dict(tokens=dict(n_checked=10, n_unequal=0, equal_fraction=1.0, first=None),
               fingerprints=dict(n_checked=10, n_unequal=0), rates=dict(median_tok_s=950.0),
               acceptance=dict(rate=0.97))
    sp = {"L3": dict(spr), "L4X": dict(spr), "L4F": dict(spr, rates=dict(median_tok_s=1100.0)),
          "L4P": dict(spr, rates=dict(median_tok_s=900.0), acceptance=dict(rate=0.95))}      # rung L4P (r2-specp)
    for k in ("L3_tok", "L4X_tok", "L4F_agree", "L3_ratio", "L4X_ratio", "L4F_ratio", "accept", "L4F_vs_L3",
              "L4P_tok", "L4P_ratio", "L4P_accept", "L4P_vs_L4X"):
        monkeypatch.setitem(SC.PRED_SPEC, k, None)
    v = {n: x for n, x, _ in SC.verdicts_spec(sh, sp, 8)}
    assert set(v.values()) == {"REPORT"}, v
    for k, val in (("L3_tok", True), ("L4X_tok", True), ("L4F_agree", 0.99), ("L3_ratio", (0.9, 1.0)),
                   ("L4X_ratio", (1.0, 1.2)), ("L4F_ratio", (1.0, 1.2)), ("accept", (0.9, 1.0)),
                   ("L4F_vs_L3", (1.0, 1.1)), ("L4P_tok", True), ("L4P_ratio", (0.8, 0.95)),
                   ("L4P_accept", (0.97, 1.0)), ("L4P_vs_L4X", (1.0, 1.2))):
        monkeypatch.setitem(SC.PRED_SPEC, k, val)
    v = {n: x for n, x, _ in SC.verdicts_spec(sh, sp, 8)}
    assert v == {"L3-TOK": "CONFIRMED", "L3-RATIO": "CONFIRMED", "L4X-TOK": "CONFIRMED", "L4X-RATIO": "REFUTED",
                 "L4F-AGREE": "CONFIRMED", "L4F-RATIO": "CONFIRMED", "SP-ACCEPT": "CONFIRMED",
                 "L4F-VS-L3": "REFUTED", "L4P-TOK": "CONFIRMED", "L4P-RATIO": "CONFIRMED", "L4P-ACCEPT": "REFUTED",
                 "L4P-VS-L4X": "REFUTED"}, v
    rd = dict(errors=(0, [0, 0]), count=[10, 13], warm=1, rounds=4, round_committed=[5, 6, 5, 5])
    rec = dict(tokens=dict(spr["tokens"], min_checked=10), fingerprints=dict(spr["fingerprints"], first=None))

    def gates_of(arm, r=rec):
        out = SC.arm_gates(arm, dict(read=rd), r)
        assert all(must for _, must, _, _ in out), out                # every gate of an arm is a MUST gate
        return {n: ok for n, _, ok, _ in out}
    assert gates_of("L3") == {"L3-ERR": True, "L3-COMMIT": True, "L3-TOK": True, "L3-FP": True}
    assert gates_of("L4X") == {"L4X-ERR": True, "L4X-COMMIT": True, "L4X-TOK": True, "L4X-FP": True}
    assert gates_of("L4F") == {"L4F-ERR": True, "L4F-COMMIT": True}   # FAST: no token gate (not bitwise by design)
    assert "L4P" in SC.EXACT_ARMS                                      # r2-specp: L4P (EXACT) gets the token gates
    assert gates_of("L4P") == {"L4P-ERR": True, "L4P-COMMIT": True, "L4P-TOK": True, "L4P-FP": True}
    # THE TOKEN GATES FAIL THE CELL (sp3b at 6de1cfd: 122,604 unequal tokens and 0 failed MUST gates)
    one_tok = dict(rec, tokens=dict(rec["tokens"], n_unequal=1, first=(124, 1)))
    assert gates_of("L3", one_tok) == {"L3-ERR": True, "L3-COMMIT": True, "L3-TOK": False, "L3-FP": True}
    one_fp = dict(rec, fingerprints=dict(rec["fingerprints"], n_unequal=1, first=(1, 1)))
    assert gates_of("L4X", one_fp)["L4X-FP"] is False and gates_of("L4X", one_fp)["L4X-TOK"] is True
    none_checked = dict(tokens=dict(rec["tokens"], n_checked=0), fingerprints=dict(rec["fingerprints"], n_checked=0))
    assert gates_of("L3", none_checked)["L3-TOK"] is False and gates_of("L3", none_checked)["L3-FP"] is False
    assert gates_of("L3", None)["L3-TOK"] is False and gates_of("L4X", None)["L4X-FP"] is False
    assert gates_of("L4P", one_tok)["L4P-TOK"] is False and gates_of("L4P", one_fp)["L4P-FP"] is False
    assert gates_of("L4P", None)["L4P-TOK"] is False and gates_of("L4P", None)["L4P-FP"] is False
    assert gates_of("L4F", None) == {"L4F-ERR": True, "L4F-COMMIT": True}
    rd["round_committed"] = [5, 6, 5, 4]
    assert gates_of("L3")["L3-COMMIT"] is False
    src = (EFF / "r2_spec_cell.py").read_text()
    assert "gates.extend(arm_gates(arm, res, rec))" in src                 # main passes the record to the gates


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_spec_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
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


def test_step_script_sp3():
    for bad in ("sp3c", "mb1", "rb1", "rg1f", "rb74"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    # r2-specp (rung L4P): the fifth arm L4P, STEP_MIN 30 / 100 / 140, --spec-pin off, ROOT nosi-r2-specp
    for cell, pool, batch, steps, smin, mem in (("sp3m", 65, 8, 64, 30, "100G"), ("sp3a", 65, 216, 512, 100, "240G"),
                                                ("sp3b", 10, 318, 512, 140, "340G")):
        r = _step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+L3,L4X,L4F,L4P steps %d K 3 STEP_MIN %d mem %s claim "
                "/mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (cell, 63 + pool, pool, batch, steps, smin, mem,
                                                                    HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--spec 3",
                     "--spec-arms L3", "L4X", "L4F", "L4P", "--spec-prefetch off", "--spec-pin off",
                     "--spec-twin fused --spec-body fused", "/mnt/beegfs/ojerbi/nosi-r2-specp/benchmarks/Efficiency/"
                     "r2_spec_cell.py", "--mem=%s" % mem, "--time=%d" % smin, "sp_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
    # THE DISCRIMINATING CELLS of the sp3b divergence: r2_base_cell.py at sp3b's C 73 B 318, fused twin, no speculation
    eng = "--ragged-twin fused --ragged-body fused --ragged-gemm natural --ragged-graph on"
    for cell, desc, args in (
            ("rg73", "arms SH+RG+RGS steps 512 twin fused (no speculation)",
             ("--arms SH --ragged --ragged-spread 64 " + eng,)),
            ("rb73", "arms SH+RBR+RBS steps 512 K 3 groups 128 journal 256 twin fused (no draft)",
             ("--arms SH --rollback --rb-k 3 --rb-groups 128 --rb-seed 0 --rb-journal-blocks 256 " + eng,))):
        r = _step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C 73 (pool 10) B 318 %s STEP_MIN 90 mem 340G claim /mnt/beegfs/ojerbi/r2-runs/spec/"
                "sp_active_%s" % (cell, desc, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=10", "--pool 10", "--batch 318", "--steps 512", "--warmup 2", "--mem=340G",
                     "--time=90", "sp_t1_%s_%s" % (HOLDER, cell), "benchmarks/Efficiency/r2_base_cell.py") + args:
            assert want in r.stdout, (cell, want)
        for no in ("--spec", "r2_spec_cell.py", "--mbase"):
            assert no not in r.stdout, (cell, no)
