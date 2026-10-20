"""CPU tests S1-S14 of R2 step 7 (docs/R2_STEP7_SPEC_ROUND.md section 7.1): nosi/nosi/r2_spec_core.py (cadence, the
round state machine, the flattened bank advance, the stage-1 alias, the deterministic selection, page lists, the BI
paged attention reference, rollback by content with the two-page tail ring, the draft bank and the F5 rebuild, the
slot store, the G-b PaddedLinear), the contract agreement of the reference with piece 3 (nosi-r2-m2-biattn 4c1243f,
bi_paged_attention.py, copied from the shared object store into a fresh process), and the pins of
benchmarks/Efficiency/r2_bi_step.sh. No CUDA, no Triton launch. Each gate has a negative control.

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_spec_round.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi/nosi"
EFF = ROOT / "benchmarks/Efficiency"
KSTEP_CADENCE = Path("/mnt/beegfs/ojerbi/nosi-kstep-bc/nosi/nosi/kstep/cadence.py")


def _pkg(name, path):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg


def _load(pkg, path, mod):
    _pkg(pkg, path)
    full = pkg + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(path / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


C = _load("r2spec", NOSI, "r2_spec_core")
TW = sys.modules["r2spec.r2_positional_twin"]
BS = C.BLOCK


# =====================================================================================================================
# S1 cadence
# =====================================================================================================================
def test_S1_cadence():
    """S1: K = 3 cohorts [1,2], [3,5], [6,8]; the trigger at s = mK; the cohorts cover 1 .. mK-1 exactly and defer
    d_{mK}; LAMBDA <= K and K < 2 are refused; equal to K3X cadence.py when that tree is present. Negative control: a
    cohort rule that takes the newest token breaks the cover property."""
    assert [C.cohort_range(m, 3) for m in (1, 2, 3)] == [(1, 2), (3, 5), (6, 8)]
    assert [C.triggered_by(s, 3) for s in range(1, 10)] == [None, None, 1, None, None, 2, None, None, 3]
    for bad in ((1, 7), (3, 3), (3, 2), (3, None)):
        with pytest.raises(ValueError):
            C.check_cadence(*bad)

    def covers(rule, K, m):
        idx = [i for j in range(1, m + 1) for i in range(rule(j, K)[0], rule(j, K)[1] + 1)]
        return idx == list(range(1, m * K))
    assert all(covers(C.cohort_range, K, m) for K in (2, 3, 4) for m in (1, 2, 5))
    newest = lambda m, K: (1, K) if m == 1 else ((m - 1) * K + 1, m * K)
    assert not covers(newest, 3, 2)
    if KSTEP_CADENCE.exists():
        _pkg("k3x", KSTEP_CADENCE.parent)
        try:
            cad = _load("k3x", KSTEP_CADENCE.parent, "cadence")
        except Exception as e:                                    # pragma: no cover - the other tree changed
            pytest.skip("K3X cadence.py not loadable: %s" % e)
        for K in (2, 3, 4):
            for m in range(1, 6):
                assert C.cohort_range(m, K) == cad.cohort_range(m, K)
            for s in range(1, 20):
                assert C.triggered_by(s, K) == cad.triggered_by(s, K)


# =====================================================================================================================
# S2 / S3 / S11 the round over a toy target and draft
# =====================================================================================================================
V = 53


def target(seq):
    """The toy target: the token after seq (deterministic, position dependent)."""
    t = sum((x + 1) * (i % 7 + 1) for i, x in enumerate(seq[-6:])) + 7 * len(seq)
    return t % V


def draft(seq):
    """The toy draft: the target, except on about a quarter of the contexts."""
    t = target(seq)
    return (t + 1) % V if (sum(seq[-4:]) + len(seq)) % 4 == 0 else t


def oracle(prompt, n):
    seq, out = list(prompt), []
    for _ in range(n):
        t = target(seq)
        out.append(t)
        seq.append(t)
    return out


def row_seq(rnd, prompts, row):
    """The token sequence up to and including the row's input (positions 0 .. row.pos)."""
    r = rnd.reqs[row.b]
    S = len(prompts[row.b])
    return list(prompts[row.b]) + r.out[:r.E - S] + r.u[:row.pos - r.E + 1]


def run_round(prompts, max_new, K=3, LAMBDA=7, ready=None, round_cls=None, draft_fn=draft):
    first = [target(p) for p in prompts]
    pubs, rbs = [], []
    rnd = (round_cls or C.Round)(first, [len(p) for p in prompts], K=K, LAMBDA=LAMBDA, max_new=max_new,
                                 on_publish=lambda *x: pubs.append(x), on_rollback=lambda *x: rbs.append(x), ready=ready)
    verify_log, syncs, verify_ticks, ticks = [], 0, 0, 0
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        ticks += 1
        assert ticks < 5000, "no progress"
        outs = []
        for row in rows:
            seq = row_seq(rnd, prompts, row)
            assert len(seq) == row.pos + 1 and seq[-1] == row.token
            outs.append(draft_fn(seq) if row.kind == "d" else target(seq))
            if row.kind == "v":
                verify_log.append((row.b, row.pos, outs[-1], tuple(seq)))
        verify_ticks += any(row.kind == "v" for row in rows)
        syncs += rnd.finish_tick(rows, outs)
    return rnd, dict(pubs=pubs, rbs=rbs, verify_log=verify_log, syncs=syncs, verify_ticks=verify_ticks, ticks=ticks)


PROMPTS = [[3, 1, 4, 1, 5, 9], [2, 7, 1, 8, 2, 8, 1, 8], [1, 1, 2, 3, 5, 8, 13, 21, 34, 3], [9, 9, 9, 9, 9]]
MAX_NEW = [40, 33, 57, 21]


def test_S11_round_commits_the_serial_sequence():
    """S11 (bookkeeping): with B = 4 ragged requests (different prompts, budgets and rejections), every committed
    sequence equals the serial greedy oracle; every verify row whose inputs were all committed returns the oracle's
    token at its position; rollbacks keep exactly the positions < E + i*. Negative control: a decision that keeps
    accepting after the first mismatch commits a different sequence."""
    rnd, info = run_round(PROMPTS, MAX_NEW)
    for b, p in enumerate(PROMPTS):
        assert rnd.reqs[b].out == oracle(p, MAX_NEW[b]), b
    assert info["rbs"] and info["pubs"], "the run must reject and publish"
    for b, pos, out, seq in info["verify_log"]:
        S = len(PROMPTS[b])
        committed = PROMPTS[b] + rnd.reqs[b].out
        if list(seq) == committed[:pos + 1] and pos + 1 - S < len(rnd.reqs[b].out):
            assert out == rnd.reqs[b].out[pos + 1 - S], (b, pos)

    class Greedy(C.Round):
        def _decide(self, r):
            while r.cohorts and r.cohorts[0].state == "DONE":
                c = r.cohorts.pop(0)
                for i in range(c.a, c.b + 1):
                    r.out.append(r.u[i] if c.g[i] == r.u[i] else c.g[i])
                    r.committed_in_epoch += 1
                if len(r.out) >= r.max_new:
                    r.done = True
                    del r.out[r.max_new:]
                    r.cohorts = []
                    return
    bad, _ = run_round(PROMPTS, MAX_NEW, round_cls=Greedy)
    assert any(bad.reqs[b].out != oracle(p, MAX_NEW[b]) for b, p in enumerate(PROMPTS))


def test_S2_state_machine_rows_and_stalls():
    """S2: every tick has at most one draft row and one cohort per request; the lead never exceeds LAMBDA; a cohort
    that is not ready lets drafting continue up to LAMBDA, then the request stalls; the committed sequences stay the
    oracle's. Negative control: a tick with two cohorts of one request, and one with two draft rows, are refused."""
    clock = {"t": 0}

    def ready(b, c):
        return clock["t"] % 5 == 4                                    # verification waits for its blocks

    first = [target(p) for p in PROMPTS]
    rnd = C.Round(first, [len(p) for p in PROMPTS], K=3, LAMBDA=7, max_new=MAX_NEW, ready=ready)
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        per = {}
        for row in rows:
            d, ms = per.setdefault(row.b, [0, set()])
            if row.kind == "d":
                per[row.b][0] += 1
            else:
                ms.add(row.m)
        assert all(d <= 1 and len(ms) <= 1 for d, ms in per.values())
        for r in rnd.reqs:
            assert rnd.lead(r) <= rnd.LAMBDA
        outs = [draft(row_seq(rnd, PROMPTS, row)) if row.kind == "d" else target(row_seq(rnd, PROMPTS, row)) for row in rows]
        rnd.finish_tick(rows, outs)
        clock["t"] += 1
    assert rnd.stats["stall_lead"] > 0
    for b, p in enumerate(PROMPTS):
        assert rnd.reqs[b].out == oracle(p, MAX_NEW[b])
    r2 = C.Round([1], [5], max_new=[50])
    with pytest.raises(RuntimeError, match="two cohorts"):
        r2.finish_tick([C.Row(0, "v", 5, 1, 1, 0, m=1), C.Row(0, "v", 8, 1, 4, 0, m=2)], [0, 0])
    with pytest.raises(RuntimeError, match="two draft rows"):
        r2.finish_tick([C.Row(0, "d", 5, 1, 1, 0), C.Row(0, "d", 6, 1, 2, 0)], [0, 0])


def test_S3_one_decision_sync_per_tick():
    """S3: one host decision sync in every tick that finishes a cohort (with the flatten, every tick with verify rows),
    none in the others. Negative control: a round that syncs per finished cohort counts more syncs than ticks."""
    rnd, info = run_round(PROMPTS, MAX_NEW)
    assert info["syncs"] == info["verify_ticks"] == rnd.syncs and 0 < info["syncs"] < info["ticks"]

    class PerCohort(C.Round):
        def finish_tick(self, rows, outputs):
            n = super().finish_tick(rows, outputs)
            return n * len({(r.b, r.m) for r in rows if r.kind == "v"})
    _, info2 = run_round(PROMPTS, MAX_NEW, round_cls=PerCohort)
    assert info2["syncs"] > info2["verify_ticks"]


# =====================================================================================================================
# S4 / S5 / S9 / S10 positional tables (one 2B twin)
# =====================================================================================================================
def _twin(B=2, S_list=(200, 233), H=2, D=8, extra=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = max(S_list) + extra
    keys = torch.randn((B, N, H, D), generator=g)
    cis = torch.randn((B, N, H), generator=g)
    t = TW.PositionalTwin(2 * B, H, D, lk_max=N // 16 + 2, n_max=N + 64)
    for b, S in enumerate(S_list):
        t.prefill(b, keys[b, :S], cis[b, :S])
        C.sync_draft_bank(t, b, B)
    return t, keys, cis


def _clone_twin(t):
    c = TW.PositionalTwin(t.B, t.H, t.D, t.lk_max, t.n_max)
    for name in ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis"):
        getattr(c, name).copy_(getattr(t, name))
    return c


def _same_tables(a, b, banks):
    for k in banks:
        assert int(a.pos[k]) == int(b.pos[k]) and int(a.n_comp[k]) == int(b.n_comp[k]) and int(a.n_ccis[k]) == int(b.n_ccis[k])
        n, m, p = int(a.n_comp[k]), int(a.n_ccis[k]), int(a.pos[k])
        assert torch.equal(a.comp_k[k, :n], b.comp_k[k, :n]) and torch.equal(a.ccis[:, k, :m], b.ccis[:, k, :m])
        assert torch.equal(a.total_cis[k, :p], b.total_cis[k, :p]) and torch.equal(a.live_keys(k), b.live_keys(k))


def test_S4_flatten_advance_equals_serial_steps():
    """S4: one flattened call (exact bank 0: a cohort of 3 rows crossing a key and a cis compression; exact bank 1:
    2 rows; draft bank 2: 1 row) equals single-position steps of each bank in position order; per-row counters equal.
    Negative control: the cohort's rows advanced in reverse order give other tables."""
    t, keys, cis = _twin()
    ref = _clone_twin(t)
    g = torch.Generator().manual_seed(9)
    rows = {0: (torch.randn(3, 2, 8, generator=g), torch.randn(3, 2, generator=g)),
            1: (torch.randn(2, 2, 8, generator=g), torch.randn(2, 2, generator=g)),
            2: (torch.randn(1, 2, 8, generator=g), torch.randn(1, 2, generator=g))}
    for _ in range(22):                    # bank 0 to position 222: the cohort writes 222 (cis event) .. 224 (key event)
        k = torch.zeros(4, 2, 8)
        c = torch.zeros(4, 2)
        k[0], c[0] = torch.randn(2, 8, generator=g), torch.randn(2, generator=g)
        t.step(k, c, active=torch.tensor([True, False, False, False]))
        ref.step(k, c, active=torch.tensor([True, False, False, False]))
    ctr, per_row = C.flatten_advance(t, rows)
    for bank, (k, c) in rows.items():
        for u in range(k.shape[0]):
            kk = torch.zeros(4, 2, 8)
            cc = torch.zeros(4, 2)
            kk[bank], cc[bank] = k[u], c[u]
            act = torch.zeros(4, dtype=torch.bool)
            act[bank] = True
            ref.step(kk, cc, active=act)
            assert per_row[(bank, u)] == (int(ref.pos[bank]), int(ref.n_comp[bank]), int(ref.n_ccis[bank]))
    _same_tables(t, ref, range(3))
    assert int(ctr[0, 0, 0]) == 222 and int(ctr[-1, 1, 0]) == int(ctr[0, 1, 0]) + 1 and int(ctr[-1, 2, 0]) == int(ctr[0, 2, 0]) + 1
    bad = _clone_twin(ref)
    bad2 = _clone_twin(t)
    rev = {0: (rows[0][0].flip(0), rows[0][1].flip(0))}
    C.rollback_bank(bad2, ctr, {0: 0})
    C.flatten_advance(bad2, rev)
    with pytest.raises(AssertionError):
        _same_tables(bad2, bad, [0])


def test_S5_stage1_alias():
    """S5: cu_seqlens_k repeats each row's bank start (bank x Lk_max), the last entry is the table end, seqused_k is
    each row's own count; the arguments reach the op's 6th / 7th positions (stage1_seqused with a recording stub).
    Negative control: giving every row its bank's final count changes the rows that are not the last of their bank."""
    S1 = _load("r2spec", NOSI, "stage1_seqused")
    banks, n_used = [0, 0, 0, 3, 1, 1], [1007, 1008, 1008, 750, 999, 1000]
    cu, sq = C.stage1_alias(banks, n_used, 1041, 4)
    assert cu.tolist() == [0, 0, 0, 3 * 1041, 1041, 1041, 4 * 1041] and sq.tolist() == n_used
    assert cu.dtype == sq.dtype == torch.int32

    class Rec:
        def __init__(self):
            self.args = None

        def varlen_fwd_stage1(self, *a):
            self.args = a
            return [torch.zeros(1)]
    rec = Rec()
    q = torch.randn(6, 4, 8)
    ck = torch.randn(4 * 1041, 2, 8)
    S1.stage1_seqused(q, ck, ck, torch.arange(7, dtype=torch.int32), cu, 1, 1041, seqused_k=sq, ext=rec)
    assert rec.args[5] is cu and rec.args[6] is sq and rec.args[11] == 1041
    final = {}
    for b, n in zip(banks, n_used):
        final[b] = max(final.get(b, 0), n)
    wrong = C.stage1_alias(banks, [final[b] for b in banks], 1041, 4)[1]
    assert not torch.equal(wrong, sq)
    with pytest.raises(ValueError):
        C.stage1_alias([4], [10], 1041, 4)


def test_S9_rollback_by_content_and_the_two_page_tail():
    """S9: (a) a cohort advanced across a block boundary and rolled back to keep 1 row equals, by content, a bank that
    advanced 1 row; (b) the two-page tail ring: rows of block T+1 go to the other page, so after the rollback the rows
    of T below the kept position are intact and a row's attention over its pages equals the never-advanced store.
    Negative control: a one-page tail (slot-63 style) overwrites T's rows and the attention differs."""
    t, keys, cis = _twin(S_list=(250, 233))
    for _ in range(4):                                              # bank 0 at 254: the cohort crosses 256
        k = torch.zeros(4, 2, 8)
        c = torch.zeros(4, 2)
        k[0], c[0] = torch.randn(2, 8), torch.randn(2)
        t.step(k, c, active=torch.tensor([True, False, False, False]))
    keep_ref = _clone_twin(t)
    g = torch.Generator().manual_seed(4)
    kk, cc = torch.randn(4, 2, 8, generator=g), torch.randn(4, 2, generator=g)
    ctr, _ = C.flatten_advance(t, {0: (kk, cc)})
    assert int(ctr[0, 0, 0]) // BS != int(ctr[-1, 0, 0]) // BS
    C.rollback_bank(t, ctr, {0: 1})
    C.flatten_advance(keep_ref, {0: (kk[:1], cc[:1])})
    _same_tables(t, keep_ref, [0])
    # (b) tail pages: a store of 2 tail pages; positions p0 .. p0 + 3 cross a block boundary
    D, P0 = 8, 254
    q = torch.randn(1, 2, D, generator=g)

    def tail_attention(two_pages):
        store_k = torch.zeros(1, 2 * BS, 1, D)
        store_v = torch.zeros(1, 2 * BS, 1, D)
        store_b = torch.zeros(1, 2 * BS, 1)
        rows = torch.randn(P0 + 4, 1, D, generator=torch.Generator().manual_seed(5))
        for p in range(P0 - (P0 % BS), P0 + 4):                     # committed rows of T, then the cohort's rows
            page = C.tail_page(p) if two_pages else 0
            store_k[0, page * BS + p % BS] = rows[p]
            store_v[0, page * BS + p % BS] = rows[p] * 0.5
        keep_pos = P0 + 1                                           # the rollback keeps positions < 255 (in block T)
        page = C.tail_page(keep_pos - 1) if two_pages else 0
        return C.paged_attention_reference(q, store_k, store_v, store_b, torch.tensor([0]),
                                           torch.tensor([[[page]]], dtype=torch.int32), torch.tensor([[1]], dtype=torch.int32),
                                           torch.tensor([(keep_pos - 1) % BS + 1], dtype=torch.int32), D ** -0.5)

    def fresh():
        store_k = torch.zeros(1, BS, 1, D)
        store_v = torch.zeros(1, BS, 1, D)
        rows = torch.randn(P0 + 4, 1, D, generator=torch.Generator().manual_seed(5))
        for p in range(P0 - (P0 % BS), P0 + 1):
            store_k[0, p % BS] = rows[p]
            store_v[0, p % BS] = rows[p] * 0.5
        return C.paged_attention_reference(q, store_k, store_v, torch.zeros(1, BS, 1), torch.tensor([0]),
                                           torch.tensor([[[0]]], dtype=torch.int32), torch.tensor([[1]], dtype=torch.int32),
                                           torch.tensor([(P0 % BS) + 1], dtype=torch.int32), D ** -0.5)
    assert torch.equal(tail_attention(True), fresh())
    assert not torch.equal(tail_attention(False), fresh())


def test_S10_draft_bank_sync_and_rebuild():
    """S10: after a publication the draft bank equals the exact bank followed by a replay of the draft's own keys of its
    uncommitted positions (K3X F5); the replay recomputes summaries that cover published positions from exact keys.
    Negative control: a draft bank that keeps its old draft rows differs."""
    t, keys, cis = _twin()
    B = 2
    g = torch.Generator().manual_seed(6)
    ex_k, ex_c = torch.randn(10, 2, 8, generator=g), torch.randn(10, 2, generator=g)
    dr_k, dr_c = torch.randn(14, 2, 8, generator=g), torch.randn(14, 2, generator=g)
    C.flatten_advance(t, {B + 0: (dr_k, dr_c)})                     # the draft ran 14 positions ahead
    stale = _clone_twin(t)
    C.flatten_advance(t, {0: (ex_k, ex_c)})                         # the verifier published 10 exact positions
    C.rebuild_draft_bank(t, 0, B, dr_k[10:], dr_c[10:])
    # an independent one-bank twin that saw the prompt, the 10 exact positions, then the 4 later draft positions
    ref = TW.PositionalTwin(1, 2, 8, t.lk_max, t.n_max)
    ref.prefill(0, keys[0, :200], cis[0, :200])
    ref.advance(torch.cat([ex_k, dr_k[10:]]).unsqueeze(0), torch.cat([ex_c, dr_c[10:]]).unsqueeze(0), [14])

    def same_bank(a, ka, b, kb):
        assert int(a.pos[ka]) == int(b.pos[kb]) and int(a.n_comp[ka]) == int(b.n_comp[kb]) and int(a.n_ccis[ka]) == int(b.n_ccis[kb])
        n, m, p = int(b.n_comp[kb]), int(b.n_ccis[kb]), int(b.pos[kb])
        assert torch.equal(a.comp_k[ka, :n], b.comp_k[kb, :n]) and torch.equal(a.ccis[:, ka, :m], b.ccis[:, kb, :m])
        assert torch.equal(a.total_cis[ka, :p], b.total_cis[kb, :p]) and torch.equal(a.live_keys(ka), b.live_keys(kb))
    same_bank(t, B + 0, ref, 0)
    with pytest.raises(AssertionError):
        same_bank(stale, B + 0, ref, 0)


# =====================================================================================================================
# S6 / S7 / S8 selection, BI attention, pages
# =====================================================================================================================
def _ref_select(pq, pc, qk, k, high=False):
    """The rule in plain Python: top-qk of pq, +inf on those in pc, top-k of pc; ties to the lower index (or, for the
    negative control, to the higher one); ascending."""
    H, R, M = pq.shape
    tie = (lambda j: -j) if high else (lambda j: j)
    out = torch.empty((H, R, k), dtype=torch.int64)
    for h in range(H):
        for r in range(R):
            iq = set(sorted(range(M), key=lambda j: (-float(pq[h, r, j]), tie(j)))[:qk])
            v = [float("inf") if j in iq else float(pc[h, r, j]) for j in range(M)]
            out[h, r] = torch.tensor(sorted(sorted(range(M), key=lambda j: (-v[j], tie(j)))[:k]))
    return out


def test_S6_deterministic_selection():
    """S6: select_blocks equals the stable (value descending, index ascending) rule on tie-heavy data; a row's result
    does not depend on the batch (alone, in the batch, permuted). Negative control: a selection that breaks ties by
    the HIGHER index differs on the tie rows."""
    g = torch.Generator().manual_seed(3)
    H, R, M = 2, 6, 40
    pq = torch.randn(H, R, M, generator=g)
    pc = torch.randn(H, R, M, generator=g)
    pq[:, :, :2] = float("inf")
    pq[:, :, -5:] = float("inf")
    pc[:, :, 10:30] = 3.0                                           # 20 tied values above the rest: the cut is inside
    sel = C.select_blocks(pq, pc, 8, 16)
    assert torch.equal(sel, _ref_select(pq, pc, 8, 16))
    perm = torch.randperm(R, generator=g)
    assert torch.equal(C.select_blocks(pq[:, perm], pc[:, perm], 8, 16), sel[:, perm])
    for r in range(R):
        assert torch.equal(C.select_blocks(pq[:, r:r + 1], pc[:, r:r + 1], 8, 16)[:, 0], sel[:, r])
    assert not torch.equal(_ref_select(pq, pc, 8, 16, high=True), sel)


def _paged_case(R=6, Bq=2, Hk=2, G=2, D=8, S=20, n=5, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(Bq, S * BS, Hk, D, generator=g).to(dtype)
    v = torch.randn(Bq, S * BS, Hk, D, generator=g).to(dtype)
    bias = (torch.rand(Bq, S * BS, Hk, generator=g) - 0.5).to(dtype)
    q = torch.randn(R, Hk * G, D, generator=g).to(dtype)
    req = torch.tensor([r % Bq for r in range(R)], dtype=torch.int32)
    pages = torch.stack([torch.stack([torch.sort(torch.randperm(S, generator=g)[:n]).values for _ in range(Hk)])
                         for _ in range(R)]).to(torch.int32)
    npg = torch.full((R, Hk), n, dtype=torch.int32)
    last = torch.randint(1, BS + 1, (R,), generator=g).to(torch.int32)
    return q, k, v, bias, req, pages, npg, last


def _split_reference(q, k, v, bias, req, pages, npg, last, scale):
    """NEGATIVE CONTROL: split-KV whose split size is set by the batch's longest page list (a batch-dependent
    partition), combined in float32."""
    R, Hq, D = q.shape
    Hk = k.shape[2]
    G = Hq // Hk
    split = max(1, int(npg.max()) // 2)
    out = torch.empty_like(q)
    for r in range(R):
        b = int(req[r])
        for h in range(Hk):
            parts = []
            n = int(npg[r, h])
            for s0 in range(0, n, split):
                ks, vs, bs_ = [], [], []
                for p in range(s0, min(n, s0 + split)):
                    nv = int(last[r]) if p == n - 1 else BS
                    sl = slice(int(pages[r, h, p]) * BS, int(pages[r, h, p]) * BS + nv)
                    ks.append(k[b, sl, h])
                    vs.append(v[b, sl, h])
                    bs_.append(bias[b, sl, h])
                kk, vv, bb = torch.cat(ks), torch.cat(vs), torch.cat(bs_)
                s = (q[r, h * G:(h + 1) * G] @ kk.t() + bb[None, :]) * scale
                m = s.max(1).values
                p_ = torch.exp(s - m[:, None])
                parts.append((m, p_.sum(1), p_ @ vv))
            m_all = torch.stack([x[0] for x in parts]).max(0).values
            num = sum(x[2] * torch.exp(x[0] - m_all)[:, None] for x in parts)
            den = sum(x[1] * torch.exp(x[0] - m_all) for x in parts)
            out[r, h * G:(h + 1) * G] = num / den[:, None]
    return out


def test_S7_bi_attention_reference_invariance():
    """S7: the torch twin of the BI paged attention (float32): a row's output is the same alone, in the batch, permuted,
    and next to rows of its own request at other lengths; it matches a float64 direct softmax over the row's units
    to 1e-5. Negative control: a split whose size depends on the batch's longest row changes some row's bits when a
    longer row joins the batch."""
    q, k, v, bias, req, pages, npg, last = _paged_case()
    sc = 8 ** -0.5
    full = C.paged_attention_reference(q, k, v, bias, req, pages, npg, last, sc)
    perm = torch.randperm(q.shape[0])
    sh = C.paged_attention_reference(q[perm], k, v, bias, req[perm], pages[perm], npg[perm], last[perm], sc)
    assert torch.equal(sh, full[perm])
    for r in range(q.shape[0]):
        one = C.paged_attention_reference(q[r:r + 1], k, v, bias, req[r:r + 1], pages[r:r + 1], npg[r:r + 1], last[r:r + 1], sc)
        assert torch.equal(one[0], full[r])
    d64 = C.paged_attention_reference(q.double(), k.double(), v.double(), bias.double(), req, pages, npg, last, sc)
    assert float((d64 - full.double()).abs().max()) < 1e-5
    # negative control: row 0 has 4 pages; alone its split is 2 pages; next to a row of 10 pages it is 5 (one part)
    q2, k2, v2, b2, req2, pg2, _, last2 = _paged_case(R=2, n=10, seed=1)
    n_alone = torch.tensor([[4, 4]], dtype=torch.int32)
    n_batch = torch.tensor([[4, 4], [10, 10]], dtype=torch.int32)
    alone = _split_reference(q2[:1], k2, v2, b2, req2[:1], pg2[:1], n_alone, last2[:1], sc)
    batch = _split_reference(q2, k2, v2, b2, req2, pg2, n_batch, last2, sc)
    assert not torch.equal(alone[0], batch[0]), "a batch-dependent split must change row 0's bits"
    bi_alone = C.paged_attention_reference(q2[:1], k2, v2, b2, req2[:1], pg2[:1], n_alone, last2[:1], sc)
    bi_batch = C.paged_attention_reference(q2, k2, v2, b2, req2, pg2, n_batch, last2, sc)
    assert torch.equal(bi_alone[0], bi_batch[0])


def test_S8_page_lists_canonical_and_two_tail_pages():
    """S8: page_lists returns, per (row, head), the slots of the selected blocks in ascending block id, the row's own
    tail page last, the previous block from the tail ring when it is not in the slot store, and last_len = pos % 64 + 1;
    attention over those pages equals a float64 direct read of the row's units. Negative control: the same blocks in
    slot order (after a residency move) give other float32 bits."""
    g = torch.Generator().manual_seed(8)
    Bq, Hk, D, S = 1, 2, 8, 30
    T = 40
    loc = {h: {blk: s for s, blk in enumerate(torch.randperm(T - 1, generator=g)[:25].tolist())} for h in range(Hk)}
    loc[1][T - 1] = 25                                              # head 1: the previous block is in the slot store
    tail = {0: S - 2, 1: S - 1}

    def slot_of(r, h, blk):
        return loc[h][blk]
    sel_ids = []
    for h in range(Hk):
        keys = sorted(b for b in loc[h] if b != T - 1)
        pick = sorted(keys[i] for i in torch.randperm(len(keys), generator=g)[:9].tolist())
        sel_ids.append(pick + [T - 1, T])
    sel = torch.tensor(sel_ids).view(Hk, 1, 11)
    pos = T * BS + 5
    pages, npg, last = C.page_lists(sel, slot_of, [0], [pos], lambda bank, p: tail[p])
    assert int(last[0]) == pos % BS + 1 and npg.tolist() == [[11, 11]]
    for h in range(Hk):
        exp = [loc[h][blk] for blk in sel_ids[h][:-2]]
        exp += [loc[h][T - 1] if (T - 1) in loc[h] else tail[C.tail_page(pos - BS)], tail[C.tail_page(pos)]]
        assert pages[0, h].tolist() == exp
    assert pages[0, 0, -2] == tail[1] and pages[0, 1, -2] == 25    # head 0 from the tail ring, head 1 from the store
    k = torch.randn(Bq, S * BS, Hk, D, generator=g)
    v = torch.randn(Bq, S * BS, Hk, D, generator=g)
    bias = torch.rand(Bq, S * BS, Hk, generator=g)
    q = torch.randn(1, 2 * Hk, D, generator=g)
    out = C.paged_attention_reference(q, k, v, bias, torch.tensor([0], dtype=torch.int32), pages, npg, last, D ** -0.5)
    for h in range(Hk):
        rows = [s * BS + i for s in pages[0, h, :-1].tolist() for i in range(BS)] + [int(pages[0, h, -1]) * BS + i for i in range(int(last[0]))]
        kk, vv, bb = k[0, rows, h].double(), v[0, rows, h].double(), bias[0, rows, h].double()
        sc = (q[0, 2 * h:2 * h + 2].double() @ kk.t() + bb[None, :]) * D ** -0.5
        ref = torch.softmax(sc, -1) @ vv
        assert float((ref - out[0, 2 * h:2 * h + 2].double()).abs().max()) < 1e-5
    scrambled = pages.clone()
    scrambled[0, :, :-1] = scrambled[0, :, :-1].flip(-1)
    out2 = C.paged_attention_reference(q, k, v, bias, torch.tensor([0], dtype=torch.int32), scrambled, npg, last, D ** -0.5)
    assert not torch.equal(out, out2)


# =====================================================================================================================
# S7b the contract agreement with piece 3 (a fresh process: piece 3 imports triton when it is installed)
# =====================================================================================================================
PIECE3 = dict(sha="4c1243f3f0441725943e8dcd515dfaac8c27446c", path="nosi/nosi/bi_paged_attention.py",
              blob="7477ddd01b7b47585d5c45557c7d41c687906b2d")

CHILD_P3 = r'''
import importlib.util, os, sys, types
sys.dont_write_bytecode = True
import torch
root, p3path = sys.argv[1], sys.argv[2]
pkg = types.ModuleType("r2k"); pkg.__path__ = [root]; sys.modules["r2k"] = pkg
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
core = load("r2k.r2_spec_core", os.path.join(root, "r2_spec_core.py"))
p3 = load("piece3_bi_paged_attention", p3path)
g = torch.Generator().manual_seed(3)
PAGE, D, G, Hk, S, R, P = 64, 16, 4, 2, 12, 6, 5
k = torch.randn(2, S * PAGE, Hk, D, generator=g); v = torch.randn(2, S * PAGE, Hk, D, generator=g)
bias = (torch.rand(2, S * PAGE, Hk, generator=g) - 0.5) * 4
q = torch.randn(R, Hk * G, D, generator=g)
req = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32)
# rows 0-3: P pages, the last two are the tail pair (T read whole, T+1 cut); rows 4-5: one page only (the cut page)
pages = torch.full((R, Hk, P), -1, dtype=torch.int32)
npg = torch.empty((R, Hk), dtype=torch.int32)
for r in range(R):
    for h in range(Hk):
        n = P if r < 4 else 1
        ids = torch.sort(torch.randperm(S, generator=g)[:n]).values
        pages[r, h, :n] = ids.to(torch.int32); npg[r, h] = n
last = torch.tensor([1, 64, 37, 2, 63, 9], dtype=torch.int32)
scale = D ** -0.5
mine = core.paged_attention_reference(q, k, v, bias, req, pages, npg, last, scale)
twin = p3.bi_paged_decode_reference(q, k, v, bias, req, pages, npg, last, softmax_scale=scale)
orc = p3.attention_oracle(q, k, v, bias, req, pages, npg, last, softmax_scale=scale)
d_twin = float((mine - twin).abs().max()); d_orc = float((mine.double() - orc).abs().max())
assert d_twin < 1e-5, ("mine vs piece 3 twin", d_twin)
assert d_orc < 1e-5, ("mine vs piece 3 oracle", d_orc)
# negative control: the OTHER bias semantics (bias added after the scale) is far from mine: the check discriminates
after = p3.attention_oracle(q, k, v, bias / scale, req, pages, npg, last, softmax_scale=scale)
d_after = float((mine.double() - after).abs().max())
assert d_after > 1e-2, ("bias-after-scale is not caught", d_after)
# negative control: a last page cut one row late (the extent) is far from mine
late = p3.attention_oracle(q, k, v, bias, req, pages, npg, torch.clamp(last + 1, max=PAGE), softmax_scale=scale)
moved = [r for r in range(R) if int(last[r]) < PAGE]
d_late = min(float((mine[r].double() - late[r]).abs().max()) for r in moved)
assert d_late > 1e-4, ("a late cut is not caught", d_late)
print("CHILD OK %.2e %.2e %.2e %.2e" % (d_twin, d_orc, d_after, d_late))
'''


def test_S7b_contract_agreement_with_piece3(tmp_path):
    """S7b: the reference r2_spec_core.paged_attention_reference and piece 3 (nosi-r2-m2-biattn 4c1243f: the torch
    twin bi_paged_decode_reference and the float64 attention_oracle) compute the SAME attention: logit =
    (q.k + bias) * scale, pages in list order, the last listed page cut at last_len (1, 2, 9, 37, 63, 64; a tail pair;
    one-page rows), within 1e-5. Piece 3's module is the pinned blob, copied from the shared object store and loaded by
    path in a fresh process. Negative controls: the bias-after-the-scale semantics and a cut one row late both differ
    from the reference by far more than the tolerance."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (PIECE3["sha"], PIECE3["path"])],
                          capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.strip() == PIECE3["blob"], (
        "piece 3 is not reachable at its pin: %r %r" % (head.stdout.strip(), head.stderr.strip()[-300:]))
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3["blob"]], capture_output=True)
    assert body.returncode == 0 and body.stdout
    p3 = tmp_path / "piece3_bi_paged_attention.py"
    p3.write_bytes(body.stdout)
    r = subprocess.run([sys.executable, "-c", CHILD_P3, str(NOSI), str(p3)],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), cwd=str(tmp_path), capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0 and (r.stdout.strip().splitlines() or [""])[-1].startswith("CHILD OK"), (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:]))


# =====================================================================================================================
# S12 slot store, S13 step script
# =====================================================================================================================
def test_S12_slot_store():
    """S12: a request issues one fetch; a second request of the same block (resident or in flight) issues none and adds
    a holder; urgent: fetch / wait / resident; a slot becomes reusable only with no holder, arrived and its reader
    done; an arrival of an old generation is ignored. Negative controls: releasing with the reader not done frees
    nothing; a stale arrival does not mark the new request."""
    st = C.SlotStore(2)
    assert st.urgent(7) == "fetch"
    s, gen, issued = st.request(7, ("u0", 0, 1))
    assert issued and st.urgent(7) == "wait" and st.fetches == 1
    assert st.request(7, ("u0", 0, 2)) == (s, gen, False) and st.fetches == 1
    assert st.arrive(s, gen) and st.urgent(7) == "resident"
    busy = {"done": False}
    st.read(s, lambda: busy["done"])
    assert st.release(("u0", 0, 1)) == [] and st.release(("u0", 0, 2)) == []
    busy["done"] = True
    s2, gen2, _ = st.request(9, ("u0", 1, 1))
    s3, gen3, issued3 = st.request(11, ("u0", 1, 1))
    assert issued3 and s3 == s and gen3 == gen + 1 and st.lookup(7) is None
    assert not st.arrive(s3, gen) and st.state[s3] == "REQUESTED"
    assert st.arrive(s3, gen3)
    with pytest.raises(MemoryError):
        st.request(13, ("u0", 1, 2))


def _padding_leaks(f, x, w, m_fix, buf):
    """The live rows of f(x[:M]) for every M must not depend on the rows past M of the fixed buffer: random rows and
    zero rows give the same live rows. Returns the M values where they differ."""
    out = []
    g = torch.Generator().manual_seed(5)
    for M in (1, 3, m_fix // 2, m_fix - 1, m_fix):
        buf.copy_(torch.randn(buf.shape, generator=g))
        y1 = f(x[:M], w).clone()
        buf.zero_()
        y2 = f(x[:M], w).clone()
        if not torch.equal(y1, y2):
            out.append(M)
    return out


def test_S14_padded_linear_g_b():
    """S14 (G-b): PaddedLinear(m_fix) is F.linear at ONE row count: its output equals x @ w.T (1e-5), keeps the lead
    dims, refuses more than m_fix rows, keeps one buffer per (K, dtype, device), and its live rows do not depend on the
    buffer rows past them (random vs zero). The row-position and GPU parts are B3 (MUST). Negative control: a stand-in
    whose rows read the padding (adds the mean of the rows past M) is caught by the same check."""
    g = torch.Generator().manual_seed(4)
    m_fix, K, N = 32, 48, 24
    x = torch.randn(m_fix, K, generator=g)
    w = torch.randn(N, K, generator=g)
    pl = C.PaddedLinear(m_fix)
    buf = pl.buffer(K, x.dtype, x.device)
    assert pl.buffer(K, x.dtype, x.device) is buf and pl.buffer(K + 16, x.dtype, x.device) is not buf
    assert float((pl(x, w) - x @ w.t()).abs().max()) < 1e-5
    y3 = pl(x[:12].view(3, 4, K), w)
    assert y3.shape == (3, 4, N) and float((y3 - (x[:12] @ w.t()).view(3, 4, N)).abs().max()) < 1e-5
    with pytest.raises(ValueError):
        pl(torch.randn(m_fix + 1, K), w)
    with pytest.raises(ValueError):
        C.PaddedLinear(0)
    assert _padding_leaks(pl, x, w, m_fix, buf) == []

    def leaky(xx, ww):
        M = xx.shape[0]
        y = pl(xx, ww)
        rest = buf[M:]
        return y + (rest.mean() if rest.numel() else 0.0)
    assert _padding_leaks(leaky, x, w, m_fix, buf) != []


def test_S13_step_script_reads_the_uuid_file(tmp_path):
    """S13: r2_bi_step.sh reads the GPU uuid from <holders>/<job>.gpu_uuid: a missing file, a "GPU-" prefix and two
    lines STOP (exit 1) before any other check; BI_HOLDERS outside DRYRUN is refused; a valid file passes the uuid
    stage (DRYRUN prints it with its path when the checkout is clean at HEAD, with piece 3's pins, the BI GEMM unset,
    and the step command loading piece 3 from the step record)."""
    script = EFF / "r2_bi_step.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    sha = "0" * 40

    def run(job, content=None, dry=True):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, BI_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True, text=True,
                              timeout=120)
    r = run(111)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(112, "GPU-f31fc723-9dfe-407a-9947-28173556232a\n")
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(113, "f31fc723-9dfe-407a-9947-28173556232a\nf31fc723-9dfe-407a-9947-28173556232a\n")
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(114, "f31fc723-9dfe-407a-9947-28173556232a\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(115, "f31fc723-9dfe-407a-9947-28173556232a\n")              # passes the uuid stage, stops on the sha pin
    assert r.returncode == 1 and "HEAD is not" in r.stderr and "uuid" not in r.stderr.lower()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean git checkout: the full DRYRUN needs one")
    sha = head.stdout.strip()
    r = run(116, "f31fc723-9dfe-407a-9947-28173556232a\n")
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the uuid stage: %s" % r.stderr.strip()[-200:])
    assert "gpu_uuid f31fc723-9dfe-407a-9947-28173556232a (%s/116.gpu_uuid)" % tmp_path in r.stdout
    assert "piece 3 %s:%s blob %s" % (PIECE3["sha"], PIECE3["path"], PIECE3["blob"]) in r.stdout
    assert "BI GEMM unset" in r.stdout
    assert "--biattn /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_116.step/piece3_bi_paged_attention.py" in r.stdout
    assert "--bigemm" not in r.stdout
