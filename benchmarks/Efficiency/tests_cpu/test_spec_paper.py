"""CPU tests of rung L4P of Phase 2 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs): THE PAPER'S SCHEDULE in the speculative round runner (nosi/nosi/spec_round.py level "L4P";
r2_spec_cell.py arm L4P; r2_spec_step.sh cells sp3m / sp3a / sp3b). Paper section 3 and Figure 3: drafting is
continuous; every K draft steps ONE joint pass verifies every tentative draft but the newest (the newest is deferred to
the next verification group) and drafts the next token; section 3.4: greedy accept, no bonus token, a reject discards
every tentative token from the rejected position on (the joint pass's draft too). The world: tests_cpu/spec_world.py (the
SHIPPED engine from its own source, every CUDA op a torch stand-in with the kernel's flat addresses, L28). B 1 and B 8.
No CUDA, no Triton launch. K = 3 (spec_world.K0).

TOKENS   test_paper_sched_tokens_equal_shipped_greedy (B 1 / B 8): every request's committed tokens are the SHIPPED
         greedy decode and every committed row's logits are the shipped step's, bitwise; residency and host rows hold
         after every round. Preconditions: rejects and full accepts, the three verify-row counts K - 1 / K / K - 2, a
         block edge, draft skips, pool moves, (B 8) ragged positions. Plant commit_draft_at_reject: CAUGHT.
DEFER    test_paper_sched_defers_newest (B 1 / B 8): the first joint pass verifies exactly d_1 .. d_{K-1} (d_K is not
         compared); after a full accept the next group starts with the deferred token and verifies it with a target row;
         the draft row's token is the group's second token; the deferred row and the draft row ran resident-only (per
         request), row 0 never. Plant verify_newest: CAUGHT.
REJECT   test_paper_reject_discards_joint_draft (B 1 / B 8): a reject keeps the accepted prefix and the target's
         correction; at the truncate the engine is at q + m (the joint draft and the rejected rows undone); drafting
         resumes from the correction; the next group has K - 1 drafts, none of the discarded ones. A full accept keeps the
         deferred row and the draft row and drafts on from e_new. Plant keep_joint_draft: CAUGHT.
HOST     test_paper_no_draft_row_reaches_host (B 1 / B 8): S 4924, so round 0's DEFERRED row fills block 76 in a full
         accept with a pad row (the truncate keeps it tentative). After every round: no tentative or draft row in the host
         rows (twin position and committed length). Plants commit_draft_rows and the rollback's no_defer: CAUGHT.
COUNTS   test_paper_commit_counts (B 8): per joint pass and request: m = n + (1 if n < v), n <= v, m <= v <= K, and
         v = K - 1 in round 0 (the first group commits at most K - 1); the device counters (valid, accepted, committed,
         full, the n histogram, the v histogram, accepted per round) equal a host recount. Plants miscount and
         commit_deferred: CAUGHT.
PIN      test_paper_pin_flag (B 8): the requested-KV pin is off by default (no soft pin ever); on: pins applied, released
         after every round, the tokens unchanged. Plant pin_no_release: CAUGHT.
ROWS     test_paper_rows_mode_per_request (B 1 / B 8): DraftResident mode "rows" gives a draft request the draft
         residency of mode "all" and a verify request the target residency, bit for bit per request.
GATE 0   test_paper_attached_runner_is_shipped (B 1 / B 8): an attached L4P runner (pin off and on) and no round: 16
         plain steps bitwise the plain world's; detach() puts every callable back.
WIRING   test_paper_ops_refusals, test_paper_no_host_sync_in_round (pin off / on), test_paper_prefetch_wiring.
CELL     test_paper_cell_arguments_and_report: the L4P arguments; the cell's summarize and gates on a CPU L4P run.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_paper.py
"""
from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SW = _load("spec_world")
W, RW, VW, RE, RR, SR, BC = SW.W, SW.RW, SW.VW, SW.RE, SW.RR, SW.SR, SW.BC
K, N, L, BS = SW.K0, SW.N0, SW.L, SW.BS
R = K + 2                              # the joint pass's rows: K verify (at most), the deferred row, the draft row
S1 = [4658]
S8 = [4659, 4658, 4657, 4723, 4722, 4721, 4630, 4691]          # test_rollback's: fills within 16 plain steps
SM8 = [4910, 4911, 4930, 4900, 4925, 4950, 4905, 4915]           # 76-77 complete blocks: misses after the warm-up
S_DEF = 4924                           # S % 64 = 60: round 0 starts at q = S + 1 (after the warm step); its deferred
#                                        row (q + K - 1 = 4927, p % 64 = 63) fills block 76
# THE FILTER (as test_spec_round.py): request b's draft 1 is wrong in the rounds g with (g + b) % 4 == 2. In L4P draft 1
# is the newest draft of a group (deferred), so the next-but-one joint pass rejects it.
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_spec_cell
    return r2_spec_cell


def _ev(B, neg=None, rb_neg=None, S=SW.S0, pin=False):
    """One run_vs_shipped of level L4P per parameter set (shared by the claims of this module)."""
    return _ev_cached(B, neg, rb_neg, S, pin)


@functools.lru_cache(maxsize=None)
def _ev_cached(B, neg, rb_neg, S, pin):
    plant = neg is not None or rb_neg is not None
    return SW.run_vs_shipped(B, level="L4P", mode="exact", gemm="rowgroup", draft_filter=FILTER, neg=neg,
                             rb_neg=rb_neg, S=S, max_rounds=(12 if plant else 0), pin=pin)


def _clean(ev):
    return ev["bad_tokens"] == [] and ev["bad_logits"] == []


def _i(t, *ix):
    return int(t[ix])


# ------------------------------------------------------------------ TOKENS
@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_paper_sched_tokens_equal_shipped_greedy(B):
    ev = _ev(B)
    f = ev["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f                    # preconditions (L34): both paths of a round ran
    vs = {int(x) for r in ev["records"] for x in r["v"]}
    assert vs == {K - 2, K - 1, K}, vs                               # round 0, after a full accept, after a reject
    assert f["edge_crossed"] and f["skipped"] > 0, f                 # a block edge; the drafts skipped host loads
    assert f["move_out"] + f["swap"] > 0, f                          # the victim pool moved blocks
    if B > 1:
        assert f["ragged"], f                                        # requests at different positions
    assert f["min_count"] >= N, f
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]              # THE CLAIM: the shipped greedy tokens
    assert ev["bad_logits"] == [], ev["bad_logits"][:5]              # and the shipped logits of every committed row
    assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    rd = ev["rd"]
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert sum(rd["journal"]["overflow"]) == 0
    assert rd["pin"] is None and ev["pinned"] == []                  # the pin is off by default
    p = _ev(B, neg="commit_draft_at_reject")                         # PLANT: a reject commits the draft token
    assert p["facts"]["rejects"] > 0, p["facts"]                     # precondition (L34): a reject of a valid draft
    assert not _clean(p), "commit_draft_at_reject was not caught"   # CAUGHT


# ------------------------------------------------------------------ DEFER
def _defer_errors(recs, B):
    """The schedule of the deferral, from the round records. Returns (errors, the number of deferred tokens that a later
    group verified with a target row)."""
    errs, seen = [], 0
    r0 = recs[0]
    for b in range(B):
        if _i(r0["u"], b) != K or _i(r0["v"], b) != K - 1:
            errs.append(("round 0 is not u = K, v = K - 1", b, _i(r0["u"], b), _i(r0["v"], b)))
        if not torch.equal(r0["Dv"][b, :K - 1], r0["D"][b, :K - 1]) or (r0["Dv"][b, K - 1:] != -1).any():
            errs.append(("round 0 does not compare exactly d_1 .. d_{K-1}", b, r0["Dv"][b].tolist()))
    for g, r in enumerate(recs):
        rr = torch.arange(R).view(R, 1)
        if not torch.equal(r["drows"], rr >= r["v"].view(1, B)):
            errs.append(("the resident-only rows are not rows >= v", g))
        nxt = recs[g + 1] if g + 1 < len(recs) else None
        for b in range(B):
            if not bool(r["full"][b]) or nxt is None:
                continue
            u = _i(r["u"], b)
            if _i(nxt["D"], b, 0) != _i(r["D"], b, u - 1):
                errs.append(("the next group does not start with the deferred token", g, b))
            elif _i(nxt["D"], b, 1) != _i(r["e_new"], b):
                errs.append(("the draft row's token is not the next group's second token", g, b))
            elif _i(nxt["v"], b) < 1 or bool(nxt["drows"][0, b]) or _i(nxt["Dv"], b, 0) != _i(r["D"], b, u - 1):
                errs.append(("the deferred token is not verified by a target row in the next group", g, b))
            else:
                seen += 1
    return errs, seen


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_paper_sched_defers_newest(B):
    ev = _ev(B)
    recs = ev["records"]
    assert all(_i(recs[0]["D"], b, K - 1) >= 0 for b in range(B))  # precondition: the newest draft d_K exists
    errs, seen = _defer_errors(recs, B)
    assert errs == [], errs[:5]                                       # THE CLAIM
    assert seen > 0                                                   # precondition: a deferred token verified later
    rs = ev["rd"]["row_skipped"]                                      # skipped slots per joint-pass row (draft pairs)
    assert rs[0] == 0, rs                                             # row 0 is always a verify row (v >= K - 2 >= 1)
    assert sum(rs[K - 2:]) > 0, rs                                    # the deferred / draft rows were resident-only
    assert _clean(ev)
    p = _ev(B, neg="verify_newest")                                   # PLANT: no deferral (verify d_K too)
    assert all(_i(p["records"][0]["D"], b, K - 1) >= 0 for b in range(B))   # precondition: d_K exists to be verified
    assert _defer_errors(p["records"], B)[0] != [], "verify_newest was not caught"   # CAUGHT


# ------------------------------------------------------------------ REJECT
def _reject_errors(recs, B):
    """Per round and request (records): the accept, the truncate position, the first draft input and the next drafts.
    Returns (errors, rejects, full accepts)."""
    errs, n_rej, n_full = [], 0, 0
    for g, r in enumerate(recs):
        for b in range(B):
            n, v, m, u = _i(r["n"], b), _i(r["v"], b), _i(r["m"], b), _i(r["u"], b)
            p0, pt, x0 = _i(r["pos"], b), _i(r["pos_trunc"], b), _i(r["x0"], b)
            Dn, un = r["Dn"][b], _i(r["un"], b)
            if not torch.equal(r["committed"][b, :n], r["D"][b, :n]):
                errs.append(("the accepted prefix is not the drafts", g, b))
            if n < v:                                                 # a reject
                n_rej += 1
                if m != n + 1 or _i(r["committed"], b, n) != _i(r["G"], b, n) or _i(r["G"], b, n) == _i(r["D"], b, n):
                    errs.append(("a reject does not commit the accepted prefix and the target's correction", g, b))
                if pt != p0 + m:
                    errs.append(("the joint draft / the rejected rows are not undone at the truncate", g, b, pt - p0,
                                 m))
                if x0 != _i(r["committed"], b, m - 1):
                    errs.append(("drafting does not resume from the correction", g, b))
                if un != K - 1 or bool((Dn[:K - 1] < 0).any()) or bool((Dn[K - 1:] != -1).any()):
                    errs.append(("the next group is not K - 1 new drafts", g, b, Dn.tolist()))
            else:                                                     # a full accept: no bonus, the newer drafts stay
                n_full += 1
                if m != v or pt != p0 + u + 1 or x0 != _i(r["e_new"], b) or un != K + 1:
                    errs.append(("a full accept does not keep the deferred row and the draft row", g, b))
                if _i(Dn, 0) != _i(r["D"], b, u - 1) or _i(Dn, 1) != _i(r["e_new"], b):
                    errs.append(("a full accept does not keep the deferred token and e_new", g, b))
        nxt = recs[g + 1] if g + 1 < len(recs) else None
        if nxt is not None and not (torch.equal(nxt["D"], r["Dn"]) and torch.equal(nxt["u"], r["un"])):
            errs.append(("the next round does not verify the drafts of this round", g))
    return errs, n_rej, n_full


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_paper_reject_discards_joint_draft(B):
    ev = _ev(B)
    errs, n_rej, n_full = _reject_errors(ev["records"], B)
    assert n_rej > 0 and n_full > 0, (n_rej, n_full)                 # preconditions: both paths
    assert errs == [], errs[:5]                                       # THE CLAIM
    assert _clean(ev) and ev["resid"] == []
    p = _ev(B, neg="keep_joint_draft")                                # PLANT: a reject keeps the joint pass's draft
    perrs, p_rej, _ = _reject_errors(p["records"], B)
    assert p_rej > 0, p["facts"]                                      # precondition (L34): a reject in the plant run
    assert any("undone at the truncate" in e[0] for e in perrs), perrs[:3]   # CAUGHT


# ------------------------------------------------------------------ HOST
def _draft_row_fills(ev):
    """(round, request, row) where a full-accept request's DEFERRED row (row v) or DRAFT row (row u) fills a block:
    the truncate keeps that row tentative; only the round's commit may decide its fill (it must drop it)."""
    out = []
    for g, r in enumerate(ev["records"]):
        for b in range(ev["B"]):
            if not bool(r["full"][b]):
                continue
            p, v, u = _i(r["pos"], b), _i(r["v"], b), _i(r["u"], b)
            out += [(g, b, row) for row in (v, u) if (p + row + 1) % BS == 0]
    return out


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_paper_no_draft_row_reaches_host(B):
    ev = _ev(B, S=S_DEF)
    fills = _draft_row_fills(ev)
    assert fills and fills[0][0] == 0, fills[:3]                      # precondition (L34): round 0's deferred row fills
    assert any(u + 1 < R for u in ev["records"][0]["u"].tolist())    # with a pad row: the truncate keeps it tentative
    assert SW.unkept_fills(ev) and ev["facts"]["edge_crossed"]
    assert ev["host"] == [] and ev["host_c"] == [], (ev["host"][:3], ev["host_c"][:3])     # THE CLAIM, every round
    assert ev["resid"] == [] and _clean(ev)
    for plant in (dict(neg="commit_draft_rows"), dict(rb_neg="no_defer")):
        p = _ev(B, S=S_DEF, **plant)                                   # PLANTS: the commit keeps the draft rows; the
        assert _draft_row_fills(p), ("precondition (L34) of the plant run", plant)    # rollback writes in the group
        assert p["host"] != [] or p["host_c"] != [], "%s was not caught" % plant     # CAUGHT


# ------------------------------------------------------------------ COUNTS
def _count_errors(recs, B):
    errs = []
    for g, r in enumerate(recs):
        for b in range(B):
            n, v, m = _i(r["n"], b), _i(r["v"], b), _i(r["m"], b)
            if not (n <= v and m == n + (1 if n < v else 0) and m <= v <= K):
                errs.append(("per pass: m = n + correction, m <= v <= K", g, b, n, v, m))
            if g == 0 and v != K - 1:
                errs.append(("round 0 verifies K - 1", b, v))
    return errs


def test_paper_commit_counts():
    for neg in (None, "miscount", "commit_deferred"):
        ev = _ev(8, neg=neg)
        rd, recs = ev["rd"], ev["records"]
        hist, vhist = [0] * (K + 1), [0] * (K + 1)
        for r in recs:
            for x in r["n"].tolist():
                hist[x] += 1
            for x in r["v"].tolist():
                vhist[x] += 1
        host = dict(valid=sum(int(r["v"].sum()) for r in recs), accepted=sum(int(r["n"].sum()) for r in recs),
                    committed=sum(int(r["m"].sum()) for r in recs),
                    full=sum(int((r["n"] >= r["v"]).sum()) for r in recs))
        got = dict(valid=rd["valid"], accepted=rd["accepted"], committed=rd["committed"], full=rd["full"])
        assert rd["acc_hist"] == hist and rd["verify_hist"] == vhist
        assert rd["round_accepted"] == [int(r["n"].sum()) for r in recs]
        assert rd["round_committed"] == [int(r["m"].sum()) for r in recs]
        if neg is None:
            assert got == host, (got, host)
            assert _count_errors(recs, 8) == []                      # THE CLAIM
            assert sum(rd["count"]) == host["committed"] + rd["warm"] * 8
            assert any(int((r["m"] == K).sum()) for r in recs)      # precondition: a pass committed K (steady state)
            assert ev["facts"]["rejects"] > 0
            print("L4P accepted per round (B 8): %s; n histogram %s; verify rows histogram %s" % (
                rd["round_accepted"], rd["acc_hist"], rd["verify_hist"]))
        elif neg == "miscount":
            assert any(bool((r["m"] != r["n"]).any()) for r in recs)   # precondition: a round where m != n
            assert got != host, "miscount was not caught"                # CAUGHT
        else:
            assert ev["facts"]["full"] > 0                              # precondition: a full accept (it commits more)
            assert _count_errors(recs, 8) != [], "commit_deferred was not caught"   # CAUGHT


# ------------------------------------------------------------------ PIN
def test_paper_pin_flag():
    assert SR.SpecOps(level="L4P").pin is False                        # off by default
    assert SR.PIN_BASE == 1 << 60
    assert "PIN_BASE = 1 << 60" in (ROOT / "nosi/nosi/pool_prefetch.py").read_text()   # Task 1.2's soft pin value
    off = _ev(8)
    assert off["rd"]["pin"] is None and off["pinned"] == []
    on = _ev(8, pin=True)
    pin = on["rd"]["pin"]
    assert pin["on"] and pin["recorded"] > 0 and pin["applied"] > 0, pin   # precondition: requested blocks in the pool
    assert pin["applies"] == pin["releases"] == on["rd"]["rounds"], pin
    assert on["pinned"] == []                                         # THE CLAIM: no pin stays after its round
    assert _clean(on) and on["resid"] == [] and on["host"] == [], (on["bad_tokens"][:3], on["resid"][:2])
    p = _ev(8, neg="pin_no_release", pin=True)                        # PLANT: the pins are not released
    assert p["rd"]["pin"]["applied"] > 0                              # precondition
    assert p["pinned"] != [], "pin_no_release was not caught"         # CAUGHT


# ------------------------------------------------------------------ ROWS
@pytest.mark.parametrize("S_list,masks", [(SM8[:1], ([True], [False])), (SM8, ([b % 2 == 0 for b in range(8)],))],
                         ids=["B1", "B8"])
def test_paper_rows_mode_per_request(S_list, masks):
    B = len(S_list)
    for mk in masks:
        dm = torch.tensor(mk, dtype=torch.bool)
        worlds = [RW.build(S_list, horizon=128) for _ in range(3)]  # 0: target, 1: draft "all", 2: mode "rows"
        assert RW.diff(RW.state(worlds[0][1]), RW.state(worlds[2][1])) == [], "precondition: equal worlds"
        rbs = [RR.Rollback(x, 6, capacity=RW.CAP) for _, x in worlds]
        drs = [SR.DraftResident(x, rb) for (_, x), rb in zip(worlds, rbs)]
        tok = RW.tokens(0, B, n=128)
        dvs = [RW.Driver(m, x, S_list, tok) for m, x in worlds]
        for d in dvs:
            d.plain(4)                                                # the window is loaded
        skipped = loaded = 0
        for u in range(8):
            for rb in rbs:
                rb.begin()
            lt = dvs[0].step(0)
            drs[1].begin("all")
            la = dvs[1].step(0)
            drs[1].end()
            s0 = int(drs[2].skipped.sum())
            drs[2].begin("rows", rows=dm.view(1, B))
            lr = dvs[2].step(0)
            drs[2].end()
            st = [RW.state(x) for _, x in worlds]
            for b in range(B):
                ref, lref = (st[1], la) if mk[b] else (st[0], lt)
                assert torch.equal(W.bits(lr[b]), W.bits(lref[b])), ("logits", mk[b], u, b)
                assert RW.diff(RW.rows_of(st[2], b), RW.rows_of(ref, b)) == [], ("state", mk[b], u, b)
                if mk[b]:
                    for l in range(L):
                        assert torch.equal(drs[2].miss_ids[l][:, b], drs[1].miss_ids[l][:, b])
                else:
                    loaded += sum(int((st[0]["L%d._load_mask" % l][:, b] >= 0).sum()) for l in range(L))
            skipped += int(drs[2].skipped.sum()) - s0
            for rb in rbs:
                rb.commit(torch.ones(B, dtype=torch.int64))
            for d in dvs:
                d.k += 1
            if (skipped > 0 or not any(mk)) and (loaded > 0 or all(mk)):
                break
        assert skipped > 0 or not any(mk), "precondition (L34): a draft request skipped a host load"
        assert loaded > 0 or all(mk), "precondition (L34): a verify request loaded a host block"
        assert int(drs[2].row_skipped.sum()) == int(drs[2].row_skipped[0]) == int(drs[2].skipped.sum())
        assert int(drs[2].rows_draft) == drs[2].rows_calls * int(dm.sum()) and drs[2].rows_calls > 0
        x2 = worlds[2][1]
        with pytest.raises(SR.SpecError):
            drs[2].begin("rows")                                      # no rows
        rbs[2].begin()
        for bad in (dict(rows=dm.view(1, B).long()), dict(rows=dm.view(B, 1)) if B > 1 else dict(rows=dm.view(1, 1, 1)),
                    dict(rows=torch.zeros((SR.ROWS_MAX + 1, B), dtype=torch.bool)),
                    dict(rows=dm.view(1, B), record=torch.zeros((2, B), dtype=torch.bool))):
            with pytest.raises(SR.SpecError):
                drs[2].begin("rows", **bad)
        with pytest.raises(SR.SpecError):
            drs[2].begin("all", rows=dm.view(1, B))                   # rows belong to mode "rows"
        rbs[2].reject_all()
        assert all(lay.cache_engine._kv_bias_gpu_att.data_ptr() != drs[2].scratch.data_ptr()
                   for lay in x2.cache.layers)                        # end() restored the live views


# ------------------------------------------------------------------ GATE 0
@pytest.mark.parametrize("S_list", [S1, S8], ids=["B1", "B8"])
def test_paper_attached_runner_is_shipped(S_list):
    B = len(S_list)
    for pin in (False, True):
        (mx, x), (my, y) = RW.build(S_list), RW.build(S_list)
        assert RW.diff(RW.state(x), RW.state(y)) == [], "precondition: two worlds of one seed differ before step 0"
        orig = dict(y.ops.kernels)
        lin = y.ops.linear
        runner = SR.SpecRunner(my, y, SR.SpecOps(level="L4P", K=K, attn=VW.fake_attention_idx, journal=RW.CAP, pin=pin),
                               BC.HostClock())
        assert runner.rb.K == 2 * K + 1 and runner.vs.R == R and (runner.pin is not None) == pin
        tok = RW.tokens(0, B)
        dx, dy = RW.Driver(mx, x, S_list, tok), RW.Driver(my, y, S_list, tok)
        h0 = RW.state(x)
        for i in range(16):                                   # S1 fills at step 13, S8's first request at step 12
            lx, ly = dx.plain(1)[0], dy.plain(1)[0]
            assert torch.equal(W.bits(lx), W.bits(ly)), "pin %s step %d: the attached world's logits differ" % (pin, i)
        assert RW.diff(RW.state(x), RW.state(y)) == []
        assert any(n.endswith("_k_cpu") for n in RW.diff(h0, RW.state(x))), "precondition: a block fill in 16 steps"
        runner.detach()
        assert "decode_fixed" not in y.body.__dict__ and y.rollback is None and y.draft is None
        assert all(y.ops.kernels[k] is v for k, v in orig.items()) and y.ops.linear is lin
        assert all("step_unchecked" not in t.__dict__ for t in y.st.twins)


# ------------------------------------------------------------------ WIRING
def test_paper_ops_refusals():
    for kw in (dict(K=2), dict(K=31), dict(mode="fast"), dict(neg="keep_rejected"), dict(neg="bonus_on_full_accept"),
               dict(neg="pin_no_release"), dict(gemm="exact")):
        with pytest.raises(SR.SpecError):
            SR.SpecOps(level="L4P", **kw)
    for lv in ("L3", "L4"):
        for neg in SR.NEG_L4P:
            with pytest.raises(SR.SpecError):
                SR.SpecOps(level=lv, K=3, neg=neg)
        with pytest.raises(SR.SpecError):
            SR.SpecOps(level=lv, K=3, pin=True)
    d = SR.SpecOps(level="L4P", K=3).describe()
    assert (d["level"], d["mode"], d["gemm"], d["pin"]) == ("L4P", "exact", "rowgroup", False)
    assert SR.PHASES["L4P"] == ("joint", "truncate", "draft", "commit")
    for neg in SR.NEG_L4P:
        SR.SpecOps(level="L4P", K=3, neg=neg, pin=(neg == "pin_no_release"))     # every L4P plant is accepted there


@pytest.mark.parametrize("pin", [False, True], ids=["pin-off", "pin-on"])
def test_paper_no_host_sync_in_round(pin):
    """Two runs (one trap each: a trap records only the calls whose DIRECT caller is its file)."""
    hz = SW.horizon_for(8, K)
    hits = []
    for path in (SR.__file__, RR.__file__):
        model, A, Bc = SW.two_caches(8, SW.S0, hz + 64)
        runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(8, dtype=torch.int64), level="L4P", pin=pin)
        with RW.SyncTrap(path) as trap:
            runner.run(8, max_rounds=3)
        assert runner.rounds_run >= 1
        hits.append(trap.hits)
    assert {h[0] for h in hits[0]} <= {"run", "_edge_before"}, sorted({h[0] for h in hits[0]})
    assert any(h[0] == "run" for h in hits[0])                        # the trap is not blind: run's per-round sync
    assert hits[1] == [], hits[1][:5]                                 # the rollback: none


def test_paper_prefetch_wiring():
    rec = SW.Recorder()
    hz = SW.horizon_for(8, K)
    model, A, Bc = SW.two_caches(1, SW.S0, hz + 64)
    runner = SW.spec_runner(model, Bc, SW.S0, hz, torch.zeros(1, dtype=torch.int64), level="L4P", prefetch=rec)
    runner.run(8, max_rounds=3)
    G = runner.rounds_run
    assert rec.rounds == ["begin", "end"] * G
    # the prelude's K drafts; per round the joint pass (its draft row's misses) and K - 1 drafts
    assert len(rec.drafts) == K + G * K and all(len(x) == L for x in rec.drafts)
    assert all(tuple(t.shape) == (W.HKV, 1, W.TOPK) for x in rec.drafts for t in x)


# ------------------------------------------------------------------ CELL
def test_paper_cell_arguments_and_report():
    SC = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = SC.parse_args(base + ["--spec", "3", "--spec-arms", "L4P"])
    assert a.spec_arms == ["L4P"] and a.spec_pin == "off"
    assert SC.parse_args(base + ["--spec", "3", "--spec-arms", "L3,L4P", "--spec-pin", "on"]).spec_pin == "on"
    for bad in (["--spec", "2", "--spec-arms", "L3,L4P"], ["--spec", "3", "--spec-pin", "on"],
                ["--spec", "3", "--spec-arms", "L4P", "--spec-pin", "yes"]):
        with pytest.raises(SystemExit):
            SC.parse_args(base + bad)
    assert "L4P" in SC.ARMS and "L4P" in SC.EXACT_ARMS
    assert SC.ARM_OPS["L4P"] == dict(level="L4P", mode="exact", gemm="rowgroup")
    assert {"stamp_L4P", "L4P_tok", "L4P_ratio", "L4P_accept", "L4P_vs_L4X"} <= set(SC.PRED_SPEC)
    # the cell's report path on a CPU L4P run (the shipped reference as SH)
    ev = _ev(8)
    runner, rd = ev["runner"], ev["rd"]
    lg = ev["ref_lg"]
    fp_sh = (lg.amax(dim=-1), lg.sum(dim=-1, dtype=torch.float64))
    res = dict(read=rd, out=runner.out, fp=(runner.fmax, runner.fsum), K=K, probe_every=0, prefill_s=0.0,
               first_token_agreement=8, meta={})
    rec = SC.summarize("L4P", res, ev["ref_tok"], fp_sh, None, 8, N, dict(L=L, H=W.HKV, topk=W.TOPK, block_bytes=1))
    assert rec["tokens"]["n_checked"] >= 8 * N and rec["tokens"]["n_unequal"] == 0, rec["tokens"]
    assert rec["fingerprints"]["n_unequal"] == 0, rec["fingerprints"]
    assert set(rec["phases_median"]) == {"joint", "truncate", "draft", "commit"}
    pp = rec["paper"]
    assert pp["verify_rows_hist"] == rd["verify_hist"] and pp["accepted_per_round"] == rd["round_accepted"]
    assert pp["pin"] is None and rec["draft"]["skip_per_stream_per_call"] > 0
    assert rec["acceptance"]["valid"] == rd["valid"] and rec["kv_at_verify"]["in_flight_blocks"] == 0
    gates = {n: ok for n, _, ok, _ in SC.arm_gates("L4P", res)}
    assert gates == {"L4P-ERR": True, "L4P-COMMIT": True}, gates
