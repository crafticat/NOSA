"""R2 milestone 2: PER-REQUEST speculative rounds (nosi/nosi/spec_loop.py, the section PER-REQUEST ROUNDS).

Run on a CPU node, inside a job (never on the login node); one node per test, for example:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        "benchmarks/Efficiency/tests_cpu/test_spec_perreq.py::test_G1_equal_acceptance_equals_lockstep_bitwise"

THE TABLES are the positional twin of nosi-r2-m2 (nosi/nosi/r2_positional_twin.py; the interface aligned with is the one
at 65b7341: counters() (4, B), advance() -> (T + 1, 4, B), commit(ctr, keep)). It is loaded by path: NOSI_R2_TWIN if
set, else this tree's nosi/nosi/r2_positional_twin.py (after a merge), else
/mnt/beegfs/ojerbi/nosi-r2-m2/nosi/nosi/r2_positional_twin.py. A test that needs the twin FAILS (never skips) when none
is found.

THE PIPELINE (one round over NL = 2 layers, the order of the lockstep loop): K draft steps (the tables' step, the tail
write, the draft plan, the twin of the two gathers, the draft bias), the restore, the verify (the tables' advance, the
tail write, the round plan, the gathers' twin), accept, commit. _round_lockstep runs it with the lockstep functions (one
seq_length per engine, int tail ids), _round_perreq with the per-request functions (PerReqSpecContext, engine._pos).

GATE 1 (test_G1): every request at the same position, every request accepting equally. After every round the
per-request pipeline equals the lockstep pipeline bit for bit: the engines (slots, host rows, map, cache_lens,
counters), the store, the tables, and every plan, bias, mask, tail result and commit.
GATE 2 (test_G2): requests at different positions, with different acceptance. After every round the batched
per-request pipeline equals B independent B = 1 lockstep pipelines, bit for bit, per request.
NEGATIVE CONTROLS: one shared counter (the batch commits keep.min()) fails gate 2 and passes gate 1; the masked
restore removed fails both gates, at the first round that undoes a rollover.
UNITS: the write and the truncate against the lockstep functions for every keep; a ragged write and truncate against
B = 1 writes; accept; the live tail of the round plan per request; to_perreq poisons the lockstep counters; refusals
change nothing; the module imports only torch, typing and collections.

DATA: integer-valued float32 in [-8, 8] for keys, values and cis (exact in bf16; every window sum of the twin is
exact, so torch.equal does not depend on the reduction order).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import types
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "nosi/nosi/spec_loop.py"
TAIL_WRITE = ROOT / "nosi/nosi/verify/tail_write.py"
TWIN_DEFAULT = Path("/mnt/beegfs/ojerbi/nosi-r2-m2/nosi/nosi/r2_positional_twin.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sl = _load(SPEC, "nosi_spec_loop_perreq_under_test")
tw = _load(TAIL_WRITE, "nosi_tail_write_for_perreq_test")


def _twin_path() -> Path:
    env = os.environ.get("NOSI_R2_TWIN")
    if env:
        return Path(env)
    here = ROOT / "nosi/nosi/r2_positional_twin.py"
    return here if here.is_file() else TWIN_DEFAULT


@pytest.fixture(scope="module")
def twin():
    p = _twin_path()
    if not p.is_file():
        pytest.fail("the positional twin is missing at %s (set NOSI_R2_TWIN)" % p)
    return _load(p, "nosi_r2_positional_twin_for_perreq_test")


BS, TOPK, R, H, D = 64, 8, 5, 2, 4      # W = 15 slots
NL, KSEL = 2, 6                         # layers, selection width
LAY = sl.store_layout(TOPK, R)
EDT = torch.bfloat16


def _g(*parts):
    s = 0
    for p in parts:
        s = (s * 1000003 + int(p)) % (2 ** 61 - 1)
    return torch.Generator().manual_seed(s)


def _ints(shape, g):
    return torch.randint(-8, 9, tuple(shape), generator=g).to(torch.float32)


# ===========================================================================
# fixtures: engines, tables, a universe (engines + tables + store + context)
# ===========================================================================
def _engine(pos, rids, seed, l, n_max, perreq):
    """One layer's engine for requests ``rids`` at positions ``pos``. Request rid's data comes from its own generator,
    so a B = 1 engine of request rid equals slice b of a batched one."""
    B = len(pos)
    eng = types.SimpleNamespace(block_size=BS, topk=TOPK, head_num=H, head_dim=D, _tail_block_idx_on_gpu=TOPK - 1)
    parts = {k: [] for k in ("k_gpu", "v_gpu", "b_gpu", "k_cpu", "v_cpu")}
    eng._block_map = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    for b, (p, rid) in enumerate(zip(pos, rids)):
        g = _g(seed, rid, l, 1)
        parts["k_gpu"].append(_ints((LAY.W * BS, H, D), g))
        parts["v_gpu"].append(_ints((LAY.W * BS, H, D), g))
        parts["b_gpu"].append(_ints((LAY.W * BS, H), g))
        parts["k_cpu"].append(_ints((n_max, H, D), g))
        parts["v_cpu"].append(_ints((n_max, H, D), g))
        T = p // BS
        assert T >= TOPK - 1
        for h in range(H):
            eng._block_map[h, b, :TOPK - 1] = torch.randperm(T, generator=g)[:TOPK - 1]
        eng._block_map[:, b, TOPK - 1] = T
    eng._k_gpu = torch.stack(parts["k_gpu"]).to(EDT)
    eng._v_gpu = torch.stack(parts["v_gpu"]).to(EDT)
    eng._kv_bias_gpu = torch.stack(parts["b_gpu"]).to(EDT)
    eng._k_cpu = torch.stack(parts["k_cpu"]).to(EDT)
    eng._v_cpu = torch.stack(parts["v_cpu"]).to(EDT)
    pt = torch.tensor(pos, dtype=torch.int64)
    eng._cache_lens = ((TOPK - 1) * BS + pt % BS).to(torch.int32)
    if len(set(pos)) == 1:
        eng.seq_length, eng._tail_block_len_on_gpu = pos[0], pos[0] % BS
        if perreq:
            sl.to_perreq(eng)
    else:
        assert perreq, "a lockstep engine holds one seq_length"
        eng.seq_length = eng._tail_block_len_on_gpu = None
        eng._pos = pt.clone()
        sl.check_tail_state(eng, eng._pos)
    return eng


def _tables(twin, pos, rids, seed, l, n_max):
    t = twin.PositionalTwin(len(pos), H, D, lk_max=n_max // 16, n_max=n_max)
    for b, (S, rid) in enumerate(zip(pos, rids)):
        g = _g(seed, rid, l, 2)
        t.prefill(b, _ints((S, H, D), g), _ints((S, H), g))
    return t


class Uni:
    """One batch: NL engines (one per layer, as in the shipped model), NL twins, one store, one context."""

    def __init__(self, twin, pos, rids, seed, n_max, perreq):
        self.B, self.perreq = len(pos), perreq
        self.engines = [_engine(pos, rids, seed, l, n_max, perreq) for l in range(NL)]
        self.twins = [_tables(twin, pos, rids, seed, l, n_max) for l in range(NL)]
        self.store = sl.Store(NL, H, self.B, LAY)
        cls = sl.PerReqSpecContext if perreq else sl.SpecContext
        self.ctx = cls(self.store, side=None, make_event=None, current_stream=None, stream_ctx=None,
                       num_layers=NL, block_size=BS, kernel_size=32, kernel_stride=16)
        tick = self.ctx.next_tick()
        for l, e in enumerate(self.engines):
            self.store.sync_window(l, e._block_map, tick)
        self.cache = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e, tables=t)
                                                   for e, t in zip(self.engines, self.twins)])
        self.scratch = [torch.empty_like(e._kv_bias_gpu) for e in self.engines]
        self.deny_buf = [torch.empty((self.B, LAY.W, 1, H), dtype=torch.bool) for _ in self.engines]

    def pos(self):
        e = self.engines[0]
        return e._pos.clone() if self.perreq else torch.full((self.B,), int(e.seq_length), dtype=torch.int64)


def _gather(eng, issue, kv_bias):
    """The effect of the two shipped Triton gathers (flash_h2d_mask.py, flash_h2d_mask_bias.py): slot w of (h, b) <-
    host block issue[h, b, w]; the bias rows from the layer's total_cis."""
    for h, b, w in (issue >= 0).nonzero().tolist():
        blk = int(issue[h, b, w])
        src, dst = slice(blk * BS, (blk + 1) * BS), slice(w * BS, (w + 1) * BS)
        eng._k_gpu[b, dst, h] = eng._k_cpu[b, src, h]
        eng._v_gpu[b, dst, h] = eng._v_cpu[b, src, h]
        eng._kv_bias_gpu[b, dst, h] = kv_bias[b, src, h].to(eng._kv_bias_gpu.dtype)


def _sel(g, maxid):
    """(H, KSEL) int64: distinct block ids in [0, maxid] per head (a top-k); the last entry is -1 on odd draws."""
    rows = []
    for _ in range(H):
        r = torch.randperm(maxid + 1, generator=g)[:KSEL].clone()
        if int(torch.randint(0, 2, (1,), generator=g)) == 1:
            r[-1] = -1
        rows.append(r)
    return torch.stack(rows)


def _inputs(seed, rnd, rids, pos0, K, n_acc):
    """The inputs of round ``rnd``, per request from its own generator, stacked over the batch. A selection at the
    position p names blocks <= p // BS (the pooled width of the position)."""
    U = K + 1
    per = []
    for rid, p, na in zip(rids, pos0.tolist(), n_acc):
        g = _g(seed, rid, 1000 + rnd, 3)
        d = dict(dk=_ints((K, H, D), g), dv=_ints((K, H, D), g), dc=_ints((K, H), g),
                 vk=_ints((U, H, D), g), vv=_ints((U, H, D), g), vc=_ints((U, H), g),
                 sd=[[_sel(g, (p + j) // BS) for _ in range(NL)] for j in range(K)],
                 sv=[torch.stack([_sel(g, (p + u) // BS) for u in range(U)]) for _ in range(NL)])
        drafted = torch.randint(0, 1000, (K,), generator=g)
        va = torch.randint(0, 1000, (U,), generator=g)
        va[:K] = drafted
        if na < K:
            va[na] = (drafted[na] + 1) % 1000
        d["drafted"], d["va"] = drafted, va
        per.append(d)
    return types.SimpleNamespace(
        K=K, U=U,
        dk=[torch.stack([x["dk"][j] for x in per]) for j in range(K)],                      # (B, H, D)
        dv=[torch.stack([x["dv"][j] for x in per]) for j in range(K)],
        dc=[torch.stack([x["dc"][j] for x in per]) for j in range(K)],                      # (B, H)
        vk=torch.stack([x["vk"] for x in per]), vv=torch.stack([x["vv"] for x in per]),     # (B, U, H, D)
        vc=torch.stack([x["vc"] for x in per]),                                             # (B, U, H)
        sd=[[torch.stack([x["sd"][j][l] for x in per], 1) for l in range(NL)] for j in range(K)],   # (H, B, KSEL)
        sv=[torch.stack([x["sv"][l] for x in per], 2) for l in range(NL)],                   # (U, H, B, KSEL)
        drafted=torch.stack([x["drafted"] for x in per]), va=torch.stack([x["va"] for x in per]))


# ===========================================================================
# the two pipelines
# ===========================================================================
def _round_lockstep(uv, inp):
    ctx, st, B = uv.ctx, uv.store, uv.B
    K, U = inp.K, inp.U
    T = int(uv.engines[0].seq_length) // BS
    tc = [sl.capture_tail_counters(e, T) for e in uv.engines]
    t0 = [t.counters().unsqueeze(0) for t in uv.twins]
    ctx.gen_round0 = ctx.gen + 1
    rolled = [False] * NL
    rec = dict(draft=[], round=[], tails=[], tr=[])
    done = ctx.gen
    for j in range(K):
        ctx.next_tick()
        st.mark_arrived(done)
        done = ctx.gen
        for l in range(NL):
            e, t = uv.engines[l], uv.twins[l]
            t.step(inp.dk[j], inp.dc[j])
            rolled[l] = sl.draft_tail_write(e, inp.dk[j].unsqueeze(1), inp.dv[j].unsqueeze(1), t.total_cis, LAY.mirror_hi, rolled[l])
            plan = sl.plan_draft(st, l, inp.sd[j][l], T, rolled[l], ctx.next_gen(), ctx.tick, T)
            _gather(e, plan.issue, t.total_cis)
            sl.apply_draft_bias(uv.scratch[l], e._kv_bias_gpu, plan.deny, uv.deny_buf[l], LAY, BS, e._tail_block_len_on_gpu, rolled[l])
            rec["draft"].append(dict(plan=plan, bias=uv.scratch[l].clone(), rolled=torch.full((B,), rolled[l])))
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        sl.restore_tail_counters(e, tc[l], rolled[l])
        t.commit(t0[l], torch.zeros(B, dtype=torch.int64))
    was = (st.state == sl.INFLIGHT).clone()
    st.mark_arrived(ctx.gen)
    ctx.next_tick()
    tails, ctrs = [], []
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        ctrs.append(t.advance(inp.vk, inp.vc, torch.full((B,), U, dtype=torch.int64)))
        tail = tw.write_tail(e, inp.vk, inp.vv, t.total_cis, LAY.mirror_lo)
        rp = sl.plan_round(st, l, inp.sv[l], T, tail.rollover_at, ctx.gen_round0, ctx.next_gen(), ctx.tick, was[l])
        _gather(e, rp.fetch, t.total_cis)
        tails.append(tail)
        rec["round"].append(rp)
        rec["tails"].append(dict(rolled_over=torch.full((B,), tail.rolled_over),
                                 rollover_at=torch.full((B,), tail.rollover_at, dtype=torch.int64),
                                 seqused_k=torch.full((B,), tail.seqused_k, dtype=torch.int32)))
    acc = sl.accept_commit(inp.va, inp.drafted)
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        t.commit(ctrs[l], torch.full((B,), acc.c, dtype=torch.int64))
        tr = sl.truncate_tail(e, tails[l], acc.c, LAY.mirror_lo)
        rec["tr"].append(dict(undo=torch.full((B,), tr.undo_rollover),
                              restored=torch.full((B,), tr.rows_restored, dtype=torch.int64)))
    rec["keep"] = torch.full((B,), acc.c, dtype=torch.int64)
    rec["committed"] = torch.cat([acc.committed, torch.full((B, U - acc.c), -1, dtype=torch.int64)], 1)
    rec["next"] = acc.next_token
    return rec


def _round_perreq(uv, inp, keep_fn=None):
    ctx, st, B = uv.ctx, uv.store, uv.B
    K, U = inp.K, inp.U
    ctx.begin_round(uv.cache)
    T = ctx.tail_id
    rec = dict(draft=[], round=[], tails=[], tr=[])
    done = ctx.gen
    for j in range(K):
        ctx.next_tick()
        st.mark_arrived(done)
        done = ctx.gen
        for l in range(NL):
            e, t = uv.engines[l], uv.twins[l]
            t.step(inp.dk[j], inp.dc[j])
            ctx.rolled[l] = sl.draft_tail_write_perreq(e, inp.dk[j].unsqueeze(1), inp.dv[j].unsqueeze(1), t.total_cis, LAY.mirror_hi, T)
            tl, rl = sl.draft_tail_state(e._pos, T, BS)
            assert torch.equal(rl, ctx.rolled[l])
            plan = sl.plan_draft_perreq(st, l, inp.sd[j][l], T, rl, ctx.next_gen(), ctx.tick, T)
            _gather(e, plan.issue, t.total_cis)
            sl.apply_draft_bias_perreq(uv.scratch[l], e._kv_bias_gpu, plan.deny, uv.deny_buf[l], LAY, BS, tl, rl)
            rec["draft"].append(dict(plan=plan, bias=uv.scratch[l].clone(), rolled=rl.clone()))
    ctx.restore_before_verify(uv.cache)
    was = (st.state == sl.INFLIGHT).clone()
    st.mark_arrived(ctx.gen)
    ctx.next_tick()
    tails, ctrs = [], []
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        ctrs.append(t.advance(inp.vk, inp.vc, torch.full((B,), U, dtype=torch.int64)))
        tail = sl.write_tail_perreq(e, inp.vk, inp.vv, t.total_cis, LAY.mirror_lo)
        rp = sl.plan_round_perreq(st, l, inp.sv[l], tail.tail_id, tail.rollover_at, ctx.gen_round0, ctx.next_gen(), ctx.tick, was[l])
        _gather(e, rp.fetch, t.total_cis)
        tails.append(tail)
        rec["round"].append(rp)
        rec["tails"].append(dict(rolled_over=tail.rolled_over, rollover_at=tail.rollover_at, seqused_k=tail.seqused_k))
    acc = sl.accept_perreq(inp.va, inp.drafted)
    keep = acc.keep if keep_fn is None else keep_fn(acc.keep)
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        tr = sl.commit_layer_perreq(e, tails[l], t, ctrs[l], keep, LAY.mirror_lo)
        assert torch.equal(t.pos, e._pos)
        rec["tr"].append(dict(undo=tr.undo_rollover, restored=tr.rows_restored))
    rec["keep"] = keep
    rec["committed"] = torch.where(torch.arange(U).view(1, U) < keep.view(B, 1), inp.va, torch.full_like(inp.va, -1))
    if keep_fn is None:
        assert torch.equal(rec["committed"], acc.committed)
    rec["next"] = inp.va.gather(1, (keep - 1).view(B, 1))
    return rec


# ===========================================================================
# per-request views and the comparison
# ===========================================================================
TWIN_FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis")
PLAN_FIELDS = ("deny", "issue", "n_sel", "n_hit", "n_inflight", "n_issue", "n_unrequested", "n_forbidden", "n_evicted_unused")
ROUND_FIELDS = ("fetch", "overflow", "invalid", "n_new", "n_need", "n_hit_old", "n_timely", "n_late_wait", "n_late_fetch", "union_size")


def _state(uv, b):
    """Everything of request b: per layer the engine (counter, tail length, slots, host rows, map, cache_lens) and the
    tables; the store."""
    d = {}
    for l, (e, t) in enumerate(zip(uv.engines, uv.twins)):
        pos = int(e._pos[b]) if uv.perreq else int(e.seq_length)
        tl = pos % BS if uv.perreq else int(e._tail_block_len_on_gpu)
        d["L%d.pos" % l] = torch.tensor(pos)
        d["L%d.tail_len" % l] = torch.tensor(tl)
        for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_cache_lens"):
            d["L%d.%s" % (l, name)] = getattr(e, name)[b]
        d["L%d._block_map" % l] = e._block_map[:, b]
        for name in TWIN_FIELDS:
            d["L%d.tables.%s" % (l, name)] = getattr(t, name)[b]
        d["L%d.tables.ccis" % l] = t.ccis[:, b]
    st = uv.store
    for name in ("ids", "state", "gen", "stamp", "used") + st.COUNTERS:
        d["store.%s" % name] = getattr(st, name)[:, :, b]
    return d


def _rec_state(rec, b, U):
    d = {}
    for i, x in enumerate(rec["draft"]):
        for f in PLAN_FIELDS:
            d["draft%d.%s" % (i, f)] = getattr(x["plan"], f)[:, b]
        d["draft%d.bias" % i] = x["bias"][b]
        d["draft%d.rolled" % i] = x["rolled"][b]
    for l, rp in enumerate(rec["round"]):
        d["round%d.mask" % l] = rp.mask[:, b * U:(b + 1) * U]
        for f in ROUND_FIELDS:
            d["round%d.%s" % (l, f)] = getattr(rp, f)[:, b]
    for l, x in enumerate(rec["tails"]):
        for k, v in x.items():
            d["tail%d.%s" % (l, k)] = v[b]
    for l, x in enumerate(rec["tr"]):
        for k, v in x.items():
            d["truncate%d.%s" % (l, k)] = v[b]
    for k in ("keep", "committed", "next"):
        d[k] = rec[k][b]
    return d


def _first_diff(da, db):
    if list(da) != list(db):
        return "keys differ: %s" % sorted(set(da) ^ set(db))
    for k in da:
        a, b = da[k], db[k]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return "%s differs" % k
    return None


def _cover(cov, rec, U):
    for x, y in zip(rec["tails"], rec["tr"]):
        cov["rolled"] += int(x["rolled_over"].sum())
        cov["kept_fill"] += int((x["rolled_over"] & ~y["undo"]).sum())
        cov["undo_restored"] += int((y["undo"] & (y["restored"] > 0)).sum())
        cov["undo_fill_on_last_token"] += int((y["undo"] & (y["restored"] == 0)).sum())
    for x in rec["draft"]:
        cov["draft_rolled"] += int(x["rolled"].sum())
        cov["draft_inflight"] += int(x["plan"].n_inflight.sum())
        cov["draft_prefetch"] += int(x["plan"].n_issue.sum())
    for rp in rec["round"]:
        cov["late_fetch"] += int(rp.n_late_fetch.sum())
        cov["late_wait"] += int(rp.n_late_wait.sum())
        cov["timely"] += int(rp.n_timely.sum())
    cov["keep_U"] += int((rec["keep"] == U).sum())
    cov["keep_1"] += int((rec["keep"] == 1).sum())
    cov["ragged_rounds"] += int(len(set(rec["keep"].tolist())) > 1)


# ===========================================================================
# the gates
# ===========================================================================
def _gate1(twin, start_tl, K, seed, n_rounds=8, keep_fn=None, n_acc_seq=()):
    """Every request at one position, accepting equally: per-request == lockstep. Round r takes n_acc_seq[r] when
    given, else a draw from the cell's generator. -> (ok, detail, coverage)."""
    B, U = 3, K + 1
    pos = [9 * BS + start_tl] * B
    rids = list(range(B))
    n_max = pos[0] + n_rounds * U + 2 * BS
    A = Uni(twin, pos, rids, seed, n_max, perreq=False)
    P = Uni(twin, pos, rids, seed, n_max, perreq=True)
    g = _g(seed, 77)
    cov = Counter()
    for r in range(n_rounds):
        pos0 = A.pos()
        if not torch.equal(pos0, P.pos()):
            return False, "round %d: positions differ" % r, cov
        na = n_acc_seq[r] if r < len(n_acc_seq) else int(torch.randint(0, K + 1, (1,), generator=g))
        inp = _inputs(seed, r, rids, pos0, K, [na] * B)
        ra = _round_lockstep(A, inp)
        rp = _round_perreq(P, inp, keep_fn)
        for b in range(B):
            bad = _first_diff(_state(A, b), _state(P, b)) or _first_diff(_rec_state(ra, b, U), _rec_state(rp, b, U))
            if bad:
                return False, "round %d request %d: %s" % (r, b, bad), cov
        _cover(cov, rp, U)
    return True, "", cov


def _gate2(twin, pos, K, seed, n_rounds=16, keep_fn=None, n_acc_seq=()):
    """Requests at different positions, accepting differently: batched per-request == B independent B = 1 lockstep
    runs, per request. Round r takes the (B,) list n_acc_seq[r] when given, else draws. -> (ok, detail, coverage)."""
    B, U = len(pos), K + 1
    rids = list(range(B))
    n_max = max(pos) + n_rounds * U + 2 * BS
    P = Uni(twin, pos, rids, seed, n_max, perreq=True)
    refs = [Uni(twin, [pos[b]], [b], seed, n_max, perreq=False) for b in range(B)]
    g = _g(seed, 78)
    cov = Counter()
    for r in range(n_rounds):
        pos0 = P.pos()
        for b in range(B):
            if int(refs[b].pos()[0]) != int(pos0[b]):
                return False, "round %d request %d: position %d vs %d" % (r, b, int(pos0[b]), int(refs[b].pos()[0])), cov
        na = list(n_acc_seq[r]) if r < len(n_acc_seq) else torch.randint(0, K + 1, (B,), generator=g).tolist()
        inp = _inputs(seed, r, rids, pos0, K, na)
        rp = _round_perreq(P, inp, keep_fn)
        for b in range(B):
            rb = _round_lockstep(refs[b], _inputs(seed, r, [b], pos0[b:b + 1], K, [na[b]]))
            bad = _first_diff(_state(refs[b], 0), _state(P, b)) or _first_diff(_rec_state(rb, 0, U), _rec_state(rp, b, U))
            if bad:
                return False, "round %d request %d: %s" % (r, b, bad), cov
        _cover(cov, rp, U)
    return True, "", cov


# Gate 1 cells: (tail_len0, K, n_acc of round 0). Round 0 decides the rollover case of the cell (with K = 3: 61 and
# n_acc 0 undoes a fill and restores 1 row, 60 undoes a fill on the last token, 63 keeps its fill; K = 1: 62 and
# n_acc 0 undoes a fill on the last token); n_acc K gives keep = U. Later rounds draw n_acc.
G1_CELLS = [(tl, K, first) for tl in (0, 37, 58, 60, 61, 62, 63) for K in (1, 3) for first in (0, K)]
# Gate 2 cells: (positions, K, n_acc of round 0 per request). Cell 0, K = 3: tail_len0 60 keep 1 (undo, fill on the
# last token), 3 keep 4 (= U), 63 keep 2 (kept fill), 0 keep 3, 61 keep 1 (undo, 1 row restored). Cell 1, K = 1:
# 62 keep 1 (undo, fill on the last token), 63 keep 1 (kept fill).
G2_CELLS = [([9 * BS + 60, 10 * BS + 3, 9 * BS + 63, 11 * BS + 0, 9 * BS + 61], 3, [0, 3, 1, 2, 0]),
            ([9 * BS + 62, 12 * BS + 40, 10 * BS + 63, 9 * BS + 1], 1, [0, 1, 0, 1])]


def test_G1_equal_acceptance_equals_lockstep_bitwise(twin):
    cov = Counter()
    for i, (tl, K, first) in enumerate(G1_CELLS):
        ok, why, c = _gate1(twin, tl, K, seed=100 + i, n_acc_seq=[first])
        assert ok, "cell tail_len0=%d K=%d n_acc0=%d: %s" % (tl, K, first, why)
        cov.update(c)
    for k in ("rolled", "kept_fill", "undo_restored", "undo_fill_on_last_token", "draft_rolled", "draft_inflight",
              "draft_prefetch", "late_fetch", "late_wait", "timely", "keep_U", "keep_1"):
        assert cov[k] > 0, "gate 1 never exercised %s: %s" % (k, dict(cov))
    assert cov["ragged_rounds"] == 0


def test_G2_ragged_round_equals_independent_B1_runs(twin):
    cov = Counter()
    for i, (pos, K, first) in enumerate(G2_CELLS):
        ok, why, c = _gate2(twin, pos, K, seed=200 + i, n_acc_seq=[first])
        assert ok, "cell %d (K=%d): %s" % (i, K, why)
        cov.update(c)
    for k in ("ragged_rounds", "rolled", "kept_fill", "undo_restored", "undo_fill_on_last_token", "draft_rolled",
              "draft_inflight", "late_fetch", "late_wait", "timely", "keep_U", "keep_1"):
        assert cov[k] > 0, "gate 2 never exercised %s: %s" % (k, dict(cov))


def _shared(keep):
    """NEGATIVE CONTROL: one counter for the batch (the lockstep rule, keep.min() for every request)."""
    return keep.min().expand(keep.shape[0]).clone()


def test_NC1_shared_counter_fails_gate2_and_passes_gate1(twin):
    pos, K, first = G2_CELLS[0]
    ok, why, _ = _gate2(twin, pos, K, seed=200, keep_fn=_shared, n_acc_seq=[first])
    # round 0 is ragged (keep 1, 4, 2, 3, 1): request 0 keeps the minimum and matches, request 1 does not
    assert not ok and why.startswith("round 0 request 1: L0.pos differs"), why
    for i, (tl, K1, first1) in enumerate(G1_CELLS):
        if tl in (60, 61):
            ok1, why1, _ = _gate1(twin, tl, K1, seed=100 + i, keep_fn=_shared, n_acc_seq=[first1])
            assert ok1, why1                                      # equal acceptance: the shared counter is the right one


def _no_restore(engine, tail_slot, mirror_lo, nrows):
    """NEGATIVE CONTROL: the masked restore of truncate_perreq removed."""
    return None


def test_NC2_missing_masked_restore_fails_both_gates_at_the_undone_rollover(twin, monkeypatch):
    # tail_len0 61, K 3 (U 4): token 2 fills T, token 3 lands in T+1 at slot row 0; n_acc 0 keeps 1 token -> undo, 1 row
    ok, why, cov = _gate1(twin, 61, 3, seed=300, n_rounds=3, n_acc_seq=[0, 3, 3])
    assert ok and cov["undo_restored"] > 0, (why, dict(cov))       # the scenario passes with the restore
    monkeypatch.setattr(sl, "_masked_tail_restore", _no_restore)
    ok, why, _ = _gate1(twin, 61, 3, seed=300, n_rounds=3, n_acc_seq=[0, 3, 3])
    assert not ok and why.startswith("round 0 ") and ("_k_gpu" in why or "_v_gpu" in why or "_kv_bias_gpu" in why), why
    pos = [9 * BS + 61, 10 * BS + 3]
    ok, why, _ = _gate2(twin, pos, 3, seed=301, n_rounds=2, n_acc_seq=[[0, 2], [3, 3]])
    assert not ok and why.startswith("round 0 request 0") and "_gpu" in why, why
    monkeypatch.undo()
    ok, why, _ = _gate2(twin, pos, 3, seed=301, n_rounds=2, n_acc_seq=[[0, 2], [3, 3]])
    assert ok, why


# ===========================================================================
# units
# ===========================================================================
def _same_engine(a, p, where):
    """A lockstep engine ``a`` against a per-request engine ``p`` of the same batch: every tensor and the counters."""
    for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_block_map", "_cache_lens"):
        x, y = getattr(a, name), getattr(p, name)
        assert x.dtype == y.dtype and torch.equal(x, y), "%s: %s differs" % (where, name)
    assert p._pos.tolist() == [int(a.seq_length)] * a._k_gpu.shape[0], where
    assert (p._pos % BS).tolist() == [int(a._tail_block_len_on_gpu)] * a._k_gpu.shape[0], where


def _slice_engine(p, b, a, where):
    """Request b of a per-request engine ``p`` against a B = 1 lockstep engine ``a``."""
    for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_cache_lens"):
        assert torch.equal(getattr(p, name)[b], getattr(a, name)[0]), "%s: %s differs" % (where, name)
    assert torch.equal(p._block_map[:, b], a._block_map[:, 0]), "%s: _block_map differs" % where
    assert int(p._pos[b]) == int(a.seq_length) and int(p._pos[b]) % BS == int(a._tail_block_len_on_gpu), where


@pytest.mark.parametrize("tl0,U", [(10, 3), (62, 3), (63, 2), (61, 4), (60, 4), (0, 1), (63, 1), (5, 9), (56, 9), (63, 9)])
def test_U1_write_and_truncate_equal_the_lockstep_functions_for_every_keep(tl0, U):
    B = 3
    pos = [9 * BS + tl0] * B
    n_max = pos[0] + U + 2 * BS
    g = _g(5, tl0, U)
    k, v, kvb = _ints((B, U, H, D), g), _ints((B, U, H, D), g), _ints((B, n_max, H), g)
    for keep in range(U + 1):
        a = _engine(pos, list(range(B)), 11, 0, n_max, perreq=False)
        p = _engine(pos, list(range(B)), 11, 0, n_max, perreq=True)
        ta = tw.write_tail(a, k, v, kvb, LAY.mirror_lo)
        tp = sl.write_tail_perreq(p, k, v, kvb, LAY.mirror_lo)
        _same_engine(a, p, "after the write")
        assert tp.rolled_over.tolist() == [ta.rolled_over] * B and tp.rollover_at.tolist() == [ta.rollover_at] * B
        assert tp.tail_len_after.tolist() == [ta.tail_len_after] * B and tp.last_used_slot.tolist() == [ta.last_used_slot] * B
        assert tp.seqused_k.dtype == torch.int32 and tp.seqused_k.tolist() == [ta.seqused_k] * B
        assert tp.ctr.tolist() == [[pos[0] + u] * B for u in range(U + 1)]
        ra = sl.truncate_tail(a, ta, keep, LAY.mirror_lo)
        rp = sl.truncate_perreq(p, tp, torch.full((B,), keep, dtype=torch.int64), LAY.mirror_lo)
        _same_engine(a, p, "keep %d" % keep)
        assert rp.undo_rollover.tolist() == [ra.undo_rollover] * B and rp.rows_restored.tolist() == [ra.rows_restored] * B
        assert rp.tail_len.tolist() == [ra.tail_len] * B and rp.tail_id.tolist() == [ra.tail_id] * B


def test_U2_ragged_write_and_truncate_equal_independent_B1_writes():
    pos = [9 * BS + 61, 10 * BS + 2, 9 * BS + 60, 11 * BS + 63, 9 * BS + 63]
    keep = [1, 4, 3, 2, 0]
    B, U = len(pos), 4
    n_max = max(pos) + U + 2 * BS
    g = _g(6)
    k, v, kvb = _ints((B, U, H, D), g), _ints((B, U, H, D), g), _ints((B, n_max, H), g)
    p = _engine(pos, list(range(B)), 12, 0, n_max, perreq=True)
    tp = sl.write_tail_perreq(p, k, v, kvb, LAY.mirror_lo)
    after_write = [_engine([pos[b]], [b], 12, 0, n_max, perreq=False) for b in range(B)]
    tas = [tw.write_tail(a, k[b:b + 1], v[b:b + 1], kvb[b:b + 1], LAY.mirror_lo) for b, a in enumerate(after_write)]
    for b, (a, ta) in enumerate(zip(after_write, tas)):
        _slice_engine(p, b, a, "write, request %d" % b)
        assert (bool(tp.rolled_over[b]), int(tp.rollover_at[b]), int(tp.seqused_k[b]), int(tp.tail_len_after[b])) == (
            ta.rolled_over, ta.rollover_at, ta.seqused_k, ta.tail_len_after), b
    assert len(set(tp.seqused_k.tolist())) > 1                    # the verify gets per-request lengths
    rp = sl.truncate_perreq(p, tp, torch.tensor(keep), LAY.mirror_lo)
    for b, (a, ta) in enumerate(zip(after_write, tas)):
        ra = sl.truncate_tail(a, ta, keep[b], LAY.mirror_lo)
        _slice_engine(p, b, a, "truncate, request %d" % b)
        assert (bool(rp.undo_rollover[b]), int(rp.rows_restored[b])) == (ra.undo_rollover, ra.rows_restored), b
    # tail_len0 61 keep 1: undo, 1 row; 2: no fill; 60 keep 3: undo of a fill on the last token; 63 keep 2: kept; 63 keep 0: undo, 3 rows
    assert rp.undo_rollover.tolist() == [True, False, True, False, True]
    assert rp.rows_restored.tolist() == [1, 0, 0, 0, 3]
    assert rp.pos.tolist() == [x + y for x, y in zip(pos, keep)]


def test_U3_accept_perreq_commits_one_plus_n_acc_per_request_without_a_host_sync():
    drafted = torch.tensor([[5, 6, 7], [5, 6, 7], [1, 2, 3], [9, 9, 9]])
    va = torch.tensor([[5, 6, 7, 8], [5, 0, 7, 8], [1, 2, 0, 4], [0, 9, 9, 9]])
    acc = sl.accept_perreq(va, drafted)
    assert all(torch.is_tensor(x) for x in acc)                    # nothing was read to the host
    assert acc.n_acc.tolist() == [3, 1, 2, 0] and acc.keep.tolist() == [4, 2, 3, 1]
    assert acc.committed.tolist() == [[5, 6, 7, 8], [5, 0, -1, -1], [1, 2, 0, -1], [0, -1, -1, -1]]
    assert acc.next_token.tolist() == [[8], [0], [0], [0]]
    # every request accepting equally: the lockstep commit
    va2 = torch.tensor([[5, 6, 1, 8], [5, 6, 0, 8], [1, 2, 9, 4], [9, 9, 0, 9]])
    acc2 = sl.accept_perreq(va2, drafted)
    lock = sl.accept_commit(va2, drafted)
    assert lock.c == 3 and acc2.keep.tolist() == [3] * 4
    assert torch.equal(acc2.committed[:, :3], lock.committed) and bool((acc2.committed[:, 3:] == -1).all())
    assert torch.equal(acc2.next_token, lock.next_token)


def _store_slice(st, b):
    s1 = sl.Store(st.num_layers, st.H, 1, st.lay)
    for n in ("ids", "state", "gen", "stamp", "used") + st.COUNTERS:
        getattr(s1, n).copy_(getattr(st, n)[:, :, b:b + 1])
    return s1


def test_U4_round_plan_live_tail_is_per_request():
    """Request 0 (T 9, no rollover) names T + 1 at position 0: invalid. Request 1 (T 10, rollover_at 2) names T + 1
    from position 2: mirror_hi, valid. Each request's plan equals the lockstep plan of its own B = 1 store."""
    U, B = 4, 2
    T = torch.tensor([9, 10])
    ra = torch.tensor([4, 2])
    g = _g(7)
    st = sl.Store(1, H, B, LAY)
    bm = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    for b in range(B):
        for h in range(H):
            bm[h, b, :TOPK - 1] = torch.randperm(int(T[b]), generator=g)[:TOPK - 1]
        bm[:, b, TOPK - 1] = T[b]
    st.sync_window(0, bm, 1)
    sel = torch.stack([torch.stack([_sel(g, int(T[b]) + int(u >= int(ra[b]))) for b in range(B)], 1) for u in range(U)])
    sel[0, 0, 0, 0] = int(T[0]) + 1                                    # request 0, position 0, head 0: above its live tail
    sel[2, 1, 1, 0] = int(T[1]) + 1                                    # request 1, position 2, head 1: its live tail T + 1
    was = torch.zeros((H, B, LAY.W), dtype=torch.bool)
    refs = [_store_slice(st, b) for b in range(B)]
    rp = sl.plan_round_perreq(st, 0, sel, T, ra, 1, 2, 3, was)
    assert bool(rp.invalid[0, 0]) and not bool(rp.invalid[:, 1].any())
    assert int(rp.mask[1, 1 * U + 2, 0]) == LAY.mirror_hi
    for b in range(B):
        rb = sl.plan_round(refs[b], 0, sel[:, :, b:b + 1], int(T[b]), int(ra[b]), 1, 2, 3, was[:, b:b + 1])
        assert torch.equal(rp.mask[:, b * U:(b + 1) * U], rb.mask), b
        for f in ROUND_FIELDS:
            assert torch.equal(getattr(rp, f)[:, b:b + 1], getattr(rb, f)), (b, f)
        for n in ("ids", "state", "gen", "stamp", "used") + st.COUNTERS:
            assert torch.equal(getattr(st, n)[:, :, b:b + 1], getattr(refs[b], n)), (b, n)


def test_U5_to_perreq_poisons_the_lockstep_counters_and_checks_the_tail_state():
    B, U = 3, 2
    pos = [9 * BS + 10] * B
    n_max = pos[0] + 2 * BS
    g = _g(8)
    k, v, kvb = _ints((B, U, H, D), g), _ints((B, U, H, D), g), _ints((B, n_max, H), g)
    p = _engine(pos, list(range(B)), 13, 0, n_max, perreq=True)
    assert p.seq_length is None and p._tail_block_len_on_gpu is None and p._pos.tolist() == pos
    before = {n: getattr(p, n).clone() for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_cache_lens", "_pos")}
    with pytest.raises(TypeError):
        tw.write_tail(p, k, v, kvb, LAY.mirror_lo)
    with pytest.raises(TypeError):
        sl.capture_tail_counters(p, 9)
    with pytest.raises(TypeError):
        sl.draft_tail_write(p, k[:, :1], v[:, :1], kvb, LAY.mirror_hi, False)
    for n, t in before.items():
        assert torch.equal(getattr(p, n), t), n
    with pytest.raises(ValueError):
        sl.to_perreq(p)                                              # per-request already
    a = _engine(pos, list(range(B)), 13, 0, n_max, perreq=False)
    a._block_map[1, 2, TOPK - 1] += 1                                # one head of one request disagrees with the counter
    with pytest.raises(ValueError):
        sl.to_perreq(a)
    assert a.seq_length == pos[0]                                    # refused: still lockstep
    a = _engine(pos, list(range(B)), 13, 0, n_max, perreq=False)
    a._cache_lens[0] += 1
    with pytest.raises(ValueError):
        sl.to_perreq(a)


def _snap(*objs):
    out = []
    for o in objs:
        out.append({n: v.clone() for n, v in vars(o).items() if torch.is_tensor(v)})
    return out


def _unchanged(objs, snaps):
    for o, s in zip(objs, snaps):
        for n, v in s.items():
            assert torch.equal(getattr(o, n), v), n


def test_U6_refusals_change_nothing(twin):
    B, U = 2, 4
    pos = [9 * BS + 61, 10 * BS + 63]
    n_max = max(pos) + 80 + 2 * BS
    g = _g(9)
    p = _engine(pos, [0, 1], 14, 0, n_max, perreq=True)
    t = _tables(twin, pos, [0, 1], 14, 0, n_max)
    big = 66                                                       # 63 + 66 >= 128: a second rollover for request 1
    kb, vb = _ints((B, big, H, D), g), _ints((B, big, H, D), g)
    kvb = _ints((B, n_max, H), g)
    s = _snap(p)
    with pytest.raises(ValueError):
        sl.write_tail_perreq(p, kb, vb, kvb, LAY.mirror_lo)
    with pytest.raises(ValueError):
        sl.write_tail_perreq(p, kb[:, :U], vb[:, :U], kvb[:, :pos[1] + U - 1], LAY.mirror_lo)    # bias table too short
    with pytest.raises(ValueError):
        sl.write_tail_perreq(p, kb[:, :U], vb[:, :U], kvb, LAY.mirror_hi)                       # not W-2
    _unchanged([p], s)
    # a round, then the commit refusals
    k, v, c = kb[:, :U], vb[:, :U], _ints((B, U, H), g)
    ctr = t.advance(k, c, torch.full((B,), U, dtype=torch.int64))
    tail = sl.write_tail_perreq(p, k, v, t.total_cis, LAY.mirror_lo)
    s = _snap(p, t)
    for bad_keep in (torch.tensor([1, U + 1]), torch.tensor([-1, 1]), torch.tensor([1, 1, 1]), 1):
        with pytest.raises(ValueError):
            sl.commit_layer_perreq(p, tail, t, ctr, bad_keep, LAY.mirror_lo)
    shifted = ctr.clone()
    shifted[:, 0, 1] += 1                                          # the tables' counter is not the tail's
    with pytest.raises(ValueError):
        sl.commit_layer_perreq(p, tail, t, shifted, torch.tensor([1, 2]), LAY.mirror_lo)
    with pytest.raises(ValueError):
        sl.commit_layer_perreq(p, tail, t, ctr[:U], torch.tensor([1, 2]), LAY.mirror_lo)       # wrong U
    _unchanged([p, t], s)
    tr = sl.commit_layer_perreq(p, tail, t, ctr, torch.tensor([1, 2]), LAY.mirror_lo)
    assert tr.pos.tolist() == [pos[0] + 1, pos[1] + 2] and torch.equal(t.pos, p._pos)
    s = _snap(p, t)
    with pytest.raises(ValueError):
        sl.truncate_perreq(p, tail, torch.tensor([1, 2]), LAY.mirror_lo)                       # truncated already
    with pytest.raises(ValueError):
        sl.plan_round_perreq(sl.Store(1, H, B, LAY), 0, torch.full((U, H, B, KSEL), -1, dtype=torch.int64),
                             torch.tensor([9, 10]), torch.tensor([0, 2]), 1, 2, 3, torch.zeros((H, B, LAY.W), dtype=torch.bool))
    with pytest.raises(ValueError):
        sl.plan_draft_perreq(sl.Store(1, H, B, LAY), 0, torch.full((H, B, KSEL), -1, dtype=torch.int64), 9,
                             torch.zeros(B, dtype=torch.bool), 1, 1, torch.tensor([9, 10]))   # an int tail id is lockstep
    _unchanged([p, t], s)
    # the draft: a write into block T + 2 is refused
    T0 = p._pos // BS
    p._pos.copy_((T0 + 2) * BS)
    p._block_map[..., TOPK - 1] = (T0 + 1).view(1, B)
    s = _snap(p)
    with pytest.raises(ValueError):
        sl.draft_tail_write_perreq(p, k[:, :1], v[:, :1], kvb, LAY.mirror_hi, T0)
    _unchanged([p], s)


def test_U7_context_begin_round_refuses_misaligned_layers(twin):
    pos = [9 * BS + 5, 10 * BS + 7]
    n_max = max(pos) + 2 * BS
    uv = Uni(twin, pos, [0, 1], 15, n_max, perreq=True)
    uv.ctx.begin_round(uv.cache)
    assert uv.ctx.tail_id.tolist() == [9, 10] and [r.tolist() for r in uv.ctx.rolled] == [[False, False]] * NL
    with pytest.raises(ValueError):
        uv.ctx.begin_round(uv.cache, tail_id=torch.tensor([9, 11]))
    uv.engines[1]._pos[0] += 1                                     # layer 1's counter moved alone
    with pytest.raises(ValueError):
        uv.ctx.begin_round(uv.cache)
    uv.engines[1]._pos[0] -= 1
    uv.twins[0].pos[1] += 1                                        # the tables of layer 0 moved alone
    with pytest.raises(ValueError):
        uv.ctx.begin_round(uv.cache)


def test_U8_spec_loop_imports_only_torch_typing_and_collections():
    tree = ast.parse(SPEC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in ("torch", "typing", "collections"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in ("torch", "typing", "collections", "__future__"), node.module
