"""The CPU WORLD of tests_cpu/test_spec_round.py (Phase 2, rungs L3 / L4: nosi/nosi/spec_round.py; not a test file).
Load it by file path, as test_rollback.py loads rollback_world.py.

It builds on the worlds of the four hooks (read their docstrings first):
  ragged_world.py   the SHIPPED engine from its own source, the ragged engine and the R2 modules from theirs, every CUDA
                    op a torch stand-in with the kernel's flat addresses (lesson L28); fake_nosa_linear runs ROW BY ROW
                    (verify1 fix 6adf18d: the kernel's one program per (row, head)), so a row's bits never depend on
                    the row count (the joint pass has 4 B rows, the serial step B rows)
  rollback_world.py define_undefined (the KNOWN PITFALL: _new_block_map_buf / _load_mask are torch.empty scratch),
                    state / diff, Driver, journal_facts, SyncTrap, scan_source
  verify_world.py   verify_rows with the stand-ins, fake_attention_idx (the joint pass's attention), engines_equal
spec_round.py loads through the same synthetic package, so its ragged_engine / ragged_rollback / verify_rows are the
modules these worlds patched.

THIS FILE ADDS
  two_caches(B, S, max_gen)    one synthetic prefill (the shipped prefill_update with the pool), two caches, poisoned;
                               precondition (asserted): the two caches are bitwise equal before step 0
  shipped_greedy(...)          the SHIPPED greedy decode (ragged_world.shipped_step: the re-warm rule, decode_inference):
                               tokens, logits and the final host rows
  spec_runner(...)             ragged_engine.attach on the second cache (ops natural / twin torch / body torch: the
                               shipped bits), define_undefined, spec_round.SpecRunner
  run_vs_shipped(...)          both, then the comparison per request and the run's facts (the preconditions of claims)
  residency_errors(rs)         every window / pool slot that a map names holds the host rows and cis rows of that block;
                               no id twice; the tail id and the attention length follow the twin position
  host_rows_errors(...)        blocks completed during the decode equal the shipped run's host rows; the host rows from
                               the committed position's block on are still the poison (no tentative or draft row
                               reached the host rows: the dr1 leak)
  corrupt(...), corrupt_before_edge(...)   deterministic draft corruptions (the token proof holds for ANY drafts; these
                               force rejects at chosen places)
  Recorder                     a stand-in of a rung-L5 prefetcher (on_draft / on_round), for the wiring test
RUNG L4P (tests_cpu/test_spec_paper.py; level "L4P" of spec_round.py): run_vs_shipped(pin=...) passes the
requested-KV pin; after every round it records the soft pins that stay (ev["pinned"]: pinned_slots) and, for L4P, the
host-row check with q = S + the committed count (ev["host_c"]); unkept_fills knows the L4P round.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


RW = _load("rollback_world")
VW = _load("verify_world")
W = RW.W
RE, RR, VR = W.RE, RW.RR, VW.VR
SR = W.pkg_load("spec_round")
BC = W.bc()
L, BS, TOPK, HKV, V = W.L, W.BS, W.TOPK, W.HKV, W.V
S0 = 4910                      # 76 complete blocks: the 64 slots leave blocks out, the pool (P 8) moves blocks
N0 = 24                        # committed tokens per request: the run crosses the block edge at 4927 (p % 64 = 63)
K0 = 3


def bits_equal(a, b) -> bool:
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(W.bits(a), W.bits(b))


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------
def horizon_for(N: int, K: int) -> int:
    """Positions a run may reach: up to N x K committed tokens (a fast request) plus the tentative rows of a round."""
    return N * K + 4 * K + 8


def two_caches(B: int, S: int, max_gen: int, seed: int = 0):
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, [S] * B)
    caches = []
    for _ in range(2):
        c = W.new_cache(max_gen)
        W.shipped_prefill(c, rows, S, B)
        W.poison_shipped(c, S)
        caches.append(c)
    pre = VW.engines_equal(caches[0], caches[1])
    assert pre == [], "precondition: the two caches differ before step 0: %s" % pre[:5]
    return model, caches[0], caches[1]


def shipped_greedy(model, cache, B: int, S: int, N: int, tok0):
    """N steps of the shipped decode, greedy from tok0 (B,). Returns (tokens (B, N + 1): column 0 = tok0, logits
    (B, N, V), host rows [(k_cpu, v_cpu)] per layer at the end)."""
    tok = tok0.view(B, 1).clone()
    pos = torch.full((B, 1), S, dtype=torch.int32)
    warm = dict(state=None, rewarms=[])
    toks, lgs = [tok0.view(B).clone()], []
    for _ in range(N):
        W.TAG[0] = "shipped"
        la = W.shipped_step(model, cache, tok, pos, warm)[:, -1]
        assert bool(torch.isfinite(la).all()), "the shipped logits are not finite"
        lgs.append(la.clone())
        tok = la.argmax(-1, keepdim=True)
        toks.append(tok.view(B).clone())
        pos = pos + 1
    host = [(lay.cache_engine._k_cpu.clone(), lay.cache_engine._v_cpu.clone()) for lay in cache.layers]
    return torch.stack(toks, 1), torch.stack(lgs, 1), host


def spec_runner(model, cache, S: int, horizon: int, tok0, level="L3", K=K0, mode="exact", gemm="rowgroup",
                draft_filter=None, neg=None, rb_neg=None, record=True, prefetch=None, after_round=None,
                journal=RW.CAP, body="torch", redirect=True, pin=False):
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body=body, graph=True)
    rs = RE.attach(model, cache, S, horizon, ops)
    RW.define_undefined(rs)
    sops = SR.SpecOps(level=level, K=K, mode=mode, gemm=gemm, extra=16, attn=VW.fake_attention_idx, journal=journal,
                      redirect=redirect, draft_filter=draft_filter, prefetch=prefetch, rb_neg=rb_neg, neg=neg,
                      record=record, probe_every=0, after_round=after_round, pin=pin)
    return SR.SpecRunner(model, rs, sops, BC.HostClock(), t0=tok0)


def pinned_slots(rs) -> int:
    """Pool slots whose age is a soft pin (>= spec_round.PIN_BASE), over every layer (0 without a pool)."""
    if not rs.pool:
        return 0
    return sum(int((lay.cache_engine._pool_age >= SR.PIN_BASE).sum()) for lay in rs.cache.layers)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def _slot_holds(e, tw, b, h, s, x) -> bool:
    g, c = slice(s * BS, (s + 1) * BS), slice(x * BS, (x + 1) * BS)
    return (bits_equal(e._k_gpu[b, g, h], e._k_cpu[b, c, h]) and bits_equal(e._v_gpu[b, g, h], e._v_cpu[b, c, h])
            and bits_equal(e._kv_bias_gpu[b, g, h], tw.total_cis[b, c, h]))


def residency_errors(rs) -> list:
    """The residency invariants of every layer and (head, request) after a round (module docstring)."""
    errs = []
    for l, lay in enumerate(rs.cache.layers):
        e, tw = lay.cache_engine, rs.st.twins[l]
        P = int(e._pool_map.shape[2]) if rs.pool else 0
        for b in range(rs.B):
            p = int(tw.pos[b])
            tail = p // BS
            if int(e._cache_lens[b]) != (TOPK - 1) * BS + p % BS:
                errs.append(("lens", l, b, int(e._cache_lens[b]), p))
            for h in range(HKV):
                win = e._block_map[h, b].tolist()
                pool = e._pool_map[h, b].tolist() if P else []
                if win[TOPK - 1] != tail:
                    errs.append(("tail", l, h, b, win[TOPK - 1], tail))
                ids = [x for x in win[:TOPK - 1] if x >= 0] + [y for y in pool if y >= 0]
                if len(ids) != len(set(ids)) or tail in ids:
                    errs.append(("dup", l, h, b))
                for s, x in enumerate(win[:TOPK - 1]):
                    if x >= 0 and not _slot_holds(e, tw, b, h, s, x):
                        errs.append(("win", l, h, b, s, x))
                for q, y in enumerate(pool):
                    if y >= 0 and not _slot_holds(e, tw, b, h, TOPK + q, y):
                        errs.append(("pool", l, h, b, q, y))
    return errs


def host_rows_errors(rs, S: int, ref_host, n_ref: int, compare_complete: bool = True, q_list=None) -> list:
    """(A, compare_complete) the blocks completed during the decode (inside the reference's n_ref steps) equal the
    reference's host rows (EXACT runs only: FAST rows are not the shipped bits); (B) every host row from
    max(S, 64 x (q // 64)) on is still NaN, q = the request's twin position (its committed length): no tentative or
    draft row reached the host rows. q_list (B,) ints: q from the caller (L4P: S + the committed count, so a wrong
    twin position cannot hide a write)."""
    errs = []
    for l, lay in enumerate(rs.cache.layers):
        e = lay.cache_engine
        rk, rv = ref_host[l]
        for b in range(rs.B):
            q = int(rs.st.twins[l].pos[b]) if q_list is None else int(q_list[b])
            done = q // BS
            a0 = (S // BS) * BS
            z = min(done, (S + n_ref) // BS) * BS
            if compare_complete and z > a0 and not (bits_equal(e._k_cpu[b, a0:z], rk[b, a0:z])
                                                    and bits_equal(e._v_cpu[b, a0:z], rv[b, a0:z])):
                errs.append(("complete", l, b, a0, z))
            n0 = max(S, done * BS)
            if not (bool(torch.isnan(e._k_cpu[b, n0:]).all()) and bool(torch.isnan(e._v_cpu[b, n0:]).all())):
                errs.append(("tail_written", l, b, n0))
    return errs


# ---------------------------------------------------------------------------
# draft corruptions (tests only; the token proof holds for any drafts)
# ---------------------------------------------------------------------------
def corrupt(period: int = 3, j_bad: int = 1, kinds=("draft",), phase: int = 0):
    """Request b's draft j_bad (of the kinds given) becomes (token + 1) % V in the rounds g with (g + b) % period ==
    phase (the L4 prelude has round -1; in L4 the drafts made in round g are verified in round g + 1)."""
    def f(runner, kind, j, tok):
        if kind not in kinds or j != j_bad:
            return tok
        b = torch.arange(tok.shape[0])
        hit = ((runner.round + b) % period == phase).to(tok.device)
        return torch.where(hit, (tok + 1) % V, tok)
    return f


def corrupt_before_edge(K: int):
    """L3: draft 0 of a request whose round positions p .. p + K - 1 reach its block edge at offset 1 .. K - 1 becomes
    wrong, so its verify rejects at once (m = 1) and the fill of that round is NOT kept."""
    def f(runner, kind, j, tok):
        if kind != "draft" or j != 0:
            return tok
        off = (BS - 1) - runner.pos_host % BS
        hit = ((off >= 1) & (off <= K - 1)).to(tok.device)
        return torch.where(hit, (tok + 1) % V, tok)
    return f


def inject(ref_tok, rounds, kinds=("draft", "draft_row")):
    """In the rounds given, the drafts ARE the shipped tokens: the draft token for committed column c (runner.next_col,
    set by the runner for each filter call) becomes ref_tok[b, c + 1] (ref_tok column 0 = the first input). Those
    drafts are all accepted (EXACT runs), so the rounds that verify them are full accepts by construction."""
    n = ref_tok.shape[1] - 1

    def f(runner, kind, j, tok):
        if kind not in kinds or runner.round not in rounds or runner.next_col is None:
            return tok
        col = runner.next_col.to("cpu") + 1
        want = ref_tok[torch.arange(ref_tok.shape[0]), col.clamp(max=n)]
        return torch.where((col <= n).to(tok.device), want.to(tok.device), tok)
    return f


def all_kinds(*fs):
    def f(runner, kind, j, tok):
        for g in fs:
            if g is not None:
                tok = g(runner, kind, j, tok)
        return tok
    return f


class Recorder:
    """A stand-in of a rung-L5 prefetcher: it records the calls of the runner."""

    def __init__(self):
        self.drafts, self.rounds = [], []

    def on_draft(self, miss_ids):
        self.drafts.append([m.clone() for m in miss_ids])

    def on_round(self, phase):
        self.rounds.append(phase)


# ---------------------------------------------------------------------------
# the driver of the claims
# ---------------------------------------------------------------------------
def run_vs_shipped(B: int, level="L3", mode="exact", gemm="rowgroup", K=K0, S=S0, N=N0, seed=0, draft_filter=None,
                   neg=None, rb_neg=None, check_rounds=True, max_rounds=0, body="torch", redirect=True,
                   oracle_rounds=(-1, 0), pin=False):
    """THE SHIPPED GREEDY DECODE against the SPECULATIVE RUN on a second cache from the same prefill. After every round
    (check_rounds): residency_errors and host_rows_errors (collected, not raised). oracle_rounds: in these rounds the
    drafts are the shipped tokens (inject; the L4 prelude is round -1), then draft_filter acts: the first rounds are
    full accepts by construction (precondition of the full-accept path), later rounds use the natural drafts. Returns the
    evidence: per request the first unequal committed token / logits row, the round records, the facts
    (preconditions) and the runner."""
    W.STATS.clear()
    hz = horizon_for(N, K)
    model, A, Bc = two_caches(B, S, hz + 64, seed)
    g = torch.Generator().manual_seed(seed + 2)
    tok0 = torch.randint(0, V, (B,), generator=g)
    ref_tok, ref_lg, ref_host = shipped_greedy(model, A, B, S, N, tok0)
    if oracle_rounds:
        draft_filter = all_kinds(inject(ref_tok, tuple(oracle_rounds)), draft_filter)
    ev = dict(resid=[], host=[], round_pos=[], pinned=[], host_c=[])

    def after_round(runner):
        ev["round_pos"].append(runner.pos_host.clone())
        npin = pinned_slots(runner.rs)                    # L4P: no soft pin may stay after a round (RequestPin.release)
        if npin:
            ev["pinned"].append((runner.round, npin))
        if check_rounds:
            r = residency_errors(runner.rs)
            if r:
                ev["resid"].append((runner.round, r[:5]))
            h = host_rows_errors(runner.rs, S, ref_host, N, compare_complete=(mode == "exact"))
            if h:
                ev["host"].append((runner.round, h[:5]))
            if level == "L4P":                            # the committed length: S + the committed tokens
                qc = [S + c for c in runner.count.tolist()]
                h = host_rows_errors(runner.rs, S, ref_host, N, compare_complete=(mode == "exact"), q_list=qc)
                if h:
                    ev["host_c"].append((runner.round, h[:5]))
    W.TAG[0] = "spec"
    runner = spec_runner(model, Bc, S, hz, tok0, level=level, K=K, mode=mode, gemm=gemm, draft_filter=draft_filter,
                         neg=neg, rb_neg=rb_neg, after_round=after_round, body=body, redirect=redirect, pin=pin)
    runner.run(N, max_rounds=max_rounds)
    rd = runner.read()
    cnt = torch.tensor(rd["count"])
    bad_tok, bad_lg = [], []
    for b in range(B):
        c = min(int(cnt[b]), N)
        got, want = runner.out[b, :c].cpu(), ref_tok[b, 1:c + 1]
        if not torch.equal(got, want):
            i = int((got != want).nonzero()[0, 0])
            bad_tok.append((b, i, int(got[i]), int(want[i])))
        for i in range(c):
            if not bits_equal(runner.lg_out[b, i].cpu(), ref_lg[b, i]):
                bad_lg.append((b, i))
                break
    recs = runner.records
    facts = dict(rejects=sum(int(((r["n"] < r["v"]) & (r["v"] > 0)).sum()) for r in recs),
                 full=sum(int(r["full"].sum()) for r in recs),
                 partial_keep=sum(int((r["m"] < K).sum()) for r in recs),
                 skipped=sum(rd["draft_skipped"]), redirected=sum(rd["draft_redirected"]),
                 edge_crossed=any(int(c) >= ((S // BS + 1) * BS - S) for c in cnt),
                 ragged=any(len(set(p.tolist())) > 1 for p in ev["round_pos"]),
                 move_out=W.STATS.get("spec", {}).get("move_out", 0), swap=W.STATS.get("spec", {}).get("swap", 0),
                 min_count=int(cnt.min()), rounds=rd["rounds"])
    ev.update(runner=runner, rd=rd, bad_tokens=bad_tok, bad_logits=bad_lg, records=recs, facts=facts, ref_tok=ref_tok,
              ref_lg=ref_lg, ref_host=ref_host, S=S, N=N, K=K, B=B)
    return ev


def unkept_fills(ev) -> list:
    """Host facts from the round records: (round, request, offset) of a tentative position that FILLS a block
    ((p + 1) % 64 == 0) in a step that the round does not keep (a draft step, or a verify step / row at or past m)."""
    out = []
    K = ev["K"]
    level = ev["runner"].ops.level
    for g, r in enumerate(ev["records"]):
        for b in range(ev["B"]):
            p, m, full = int(r["pos"][b]), int(r["m"][b]), bool(r["full"][b])
            if level == "L3":
                offs = list(range(K))                                     # the drafts at p .. p + K - 1 (all undone)
                offs += [u for u in range(m, K)]                          # the verify steps past m
            elif level == "L4P":                                          # the paper's schedule (spec_round, L4P)
                u_ = int(r["u"][b])
                offs = list(range(m, K + 2))                              # joint-pass rows that the commit undoes
                start = u_ + 1 if full else m                             # the drafts after the truncate
                offs += [start + j for j in range(K - 1)]
            else:
                offs = [u for u in range(m, K + 1)] if not full else []  # the joint pass's undone rows
                start = p + K + 1 if full else p + m                      # the drafts after the joint pass
                offs += [start - p + j for j in range(K - 1)]
                if full:
                    offs.append(K)                                        # the draft row (kept until the group end)
            for u in offs:
                if (p + u + 1) % BS == 0:
                    out.append((g, b, u))
    return out
