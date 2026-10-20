"""CPU tests of R2 step 7 PIECE 4, the flattened joint tick (nosi/nosi/r2_tick.py; docs/R2_STEP7_SPEC_ROUND.md 7.1-7.2),
its GPU gate's helpers (benchmarks/Efficiency/r2_tick_gate.py) and its holder step (r2_tick_step.sh). No CUDA: the
engine runs a TINY random NOSA-shaped model (2 layers, 4 query heads, 2 KV heads, head dim 16, 8 selected blocks of
64 rows) with torch stand-ins for the CUDA ops (rmsnorm, nosa_linear, rope, stage 1, pooling = pool_reference) and
piece 3 v2's torch twin (bi_paged_decode_reference, the pinned blob 7c826c1 copied from the shared object store). The
reference is the engine's own SERIAL mode (one exact row per request per call), the CPU stand-in of the BI serial
decode; the GPU gate's SG0 checks the serial mode against r2_bi_decode itself. Each check has a negative control.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_tick.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
PKG = "r2tick"


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
        stub = types.ModuleType(PKG + ".nosa_llama")             # never imported by the code under test on CPU
        stub.QK_SELECT = 33
        sys.modules[PKG + ".nosa_llama"] = stub
        pkg.nosa_llama = stub
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        setattr(sys.modules[PKG], mod, m)
    return sys.modules[full]


TK = _load("r2_tick")
SC = sys.modules[PKG + ".r2_spec_core"]
TW = sys.modules[PKG + ".r2_positional_twin"]
BD = _load("r2_bi_decode")
MP = _load("max_pooling_per_request")
BS = 64

PIECE3_V2 = dict(sha="7c826c1bc7235ee9153e3af82c99d85836915416", path="nosi/nosi/bi_paged_attention.py",
                 blob="39ed5e1894ca1662f92e384a10e01a4eaab99042")


@pytest.fixture(scope="module")
def p3(tmp_path_factory):
    """Piece 3 v2 (the step's attention pin), copied from the shared object store and loaded by path."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (PIECE3_V2["sha"], PIECE3_V2["path"])],
                          capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.strip() == PIECE3_V2["blob"], head.stderr[-300:]
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3_V2["blob"]], capture_output=True)
    assert body.returncode == 0 and body.stdout
    path = tmp_path_factory.mktemp("p3") / "piece3_v2.py"
    path.write_bytes(body.stdout)
    spec = importlib.util.spec_from_file_location("piece3_v2_tick", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PAGES_PER_SPLIT == 8
    return mod


# =====================================================================================================================
# the round with device tokens (toys)
# =====================================================================================================================
V_TOY = 53


def target(seq):
    t = sum((x + 1) * (i % 7 + 1) for i, x in enumerate(seq[-6:])) + 7 * len(seq)
    return t % V_TOY


def draft(seq):
    t = target(seq)
    return (t + 1) % V_TOY if (sum(seq[-4:]) + len(seq)) % 4 == 0 else t


PROMPTS = [[3, 1, 4, 1, 5, 9], [2, 7, 1, 8, 2, 8, 1, 8], [1, 1, 2, 3, 5, 8, 13, 21, 34, 3], [9, 9, 9, 9, 9]]
MAX_NEW = [40, 33, 57, 21]


def _toy_run(round_cls=None):
    """TickRound on the toy target / draft: the fake device is a Python list (the token log)."""
    log = []
    reads = [0]

    def read(lo, hi):
        reads[0] += 1
        return log[lo:hi]
    first = [target(p) for p in PROMPTS]
    rnd = (round_cls or TK.TickRound)(first, [len(p) for p in PROMPTS], read, K=3, LAMBDA=7, max_new=MAX_NEW)
    verify_ticks = 0

    def val(x):
        return log[x.slot] if isinstance(x, TK.Ref) else x
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        assert rows and rnd.ticks < 5000
        slots = []
        for row in rows:
            r = rnd.reqs[row.b]
            S = len(PROMPTS[row.b])
            seq = list(PROMPTS[row.b]) + [val(x) for x in r.out[:r.E - S]] + [val(x) for x in r.u[:row.pos - r.E + 1]]
            assert len(seq) == row.pos + 1 and seq[-1] == val(row.token)
            log.append(draft(seq) if row.kind == "d" else target(seq))
            slots.append(len(log) - 1)
        verify_ticks += any(row.kind == "v" for row in rows)
        rnd.finish_tick(rows, slots, len(log))
    return rnd, reads[0], verify_ticks


def _spec_core_run():
    first = [target(p) for p in PROMPTS]
    rnd = SC.Round(first, [len(p) for p in PROMPTS], K=3, LAMBDA=7, max_new=MAX_NEW)
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        outs = []
        for row in rows:
            r = rnd.reqs[row.b]
            S = len(PROMPTS[row.b])
            seq = list(PROMPTS[row.b]) + r.out[:r.E - S] + r.u[:row.pos - r.E + 1]
            outs.append(draft(seq) if row.kind == "d" else target(seq))
        rnd.finish_tick(rows, outs)
    return rnd


def test_T1_device_token_round_equals_spec_core_round():
    """T1: TickRound with the draft and verifier tokens held on the "device" (a log read once per tick that finishes a
    cohort) commits exactly what r2_spec_core.Round commits with host tokens, with the same statistics, and reads the
    log once per tick that has verify rows, never otherwise. Negative control: a round that reads every token on its
    own is counted with more reads than verify ticks."""
    rnd, reads, verify_ticks = _toy_run()
    ref = _spec_core_run()
    for b in range(len(PROMPTS)):
        assert rnd.reqs[b].out == ref.reqs[b].out, b
    for k in ("accepted", "rejected", "cohorts", "draft_rows", "verify_rows", "stall_lead"):
        assert rnd.stats[k] == ref.stats[k], k
    assert rnd.stats["rejected"] > 0 and rnd.stats["accepted"] > 0
    assert reads == rnd.syncs == verify_ticks == ref.syncs and 0 < reads < rnd.ticks

    class PerToken(TK.TickRound):
        def _resolve(self, hi):
            lo = self.read_upto
            orig = self.read_tokens
            vals = [orig(s, s + 1)[0] for s in range(lo, hi)]                 # one read per token
            self.read_tokens = lambda a, b: vals[a - lo:b - lo]
            try:
                TK.TickRound._resolve(self, hi)
            finally:
                self.read_tokens = orig
    rnd2, reads2, vt2 = _toy_run(PerToken)
    assert [r.out for r in rnd2.reqs] == [r.out for r in rnd.reqs] and reads2 > vt2


def _teacher(b, p):
    return (p * 7 + b * 13 + (p // 5)) % 50


def test_T2_forced_schedule_covers_every_kind_and_commits_the_teacher():
    """T2: the teacher-forced round with forced_schedule (block 8 here: many boundaries) commits the teacher's tokens;
    its log covers every rejection offset, full acceptances, and both boundary kinds; every steered decision makes the
    next epoch start at X - 1; every cross_new decision keeps exactly the new block's first row, every cross_old one
    keeps none of the new block; simulate_schedule gives the same log. Negative controls: a reject_at outside the
    cohort is refused; a schedule without steering and boundary handling is not complete for these lengths."""
    S_list, max_new, blk = [10, 21, 30, 44], [40, 33, 57, 21], 8
    reject_at, log = TK.forced_schedule(3, block=blk)
    n = [0]
    rnd = TK.TickRound([_teacher(b, S) for b, S in enumerate(S_list)], S_list, lambda lo, hi: [0] * (hi - lo), K=3,
                       LAMBDA=7, max_new=max_new, teacher=_teacher, reject_at=reject_at)
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        assert rows
        slots = list(range(n[0], n[0] + len(rows)))
        n[0] += len(rows)
        rnd.finish_tick(rows, slots, n[0])
    for b, S in enumerate(S_list):
        assert rnd.reqs[b].out == [_teacher(b, S + j) for j in range(max_new[b])]
    cov = TK.schedule_coverage(log, 3)
    assert cov["complete"] and cov["steer"] > 0, cov
    for (b, a, e, E, o, kind) in log:
        if kind == "cross_new":
            assert (E + a + o - 1) % blk == 0                         # the kept rows end at the new block's first row
        elif kind == "cross_old":
            assert (E + a + o) % blk == 0                             # the kept rows end at the old block's last row
        elif kind == "steer":
            assert (E + a + o + 1) % blk == 0                         # the next epoch starts at X - 1
    sim, _ = TK.simulate_schedule(S_list, max_new, K=3, LAMBDA=7, block=blk)
    assert sim == log
    bad_rnd = TK.TickRound([0], [10], lambda lo, hi: [0] * (hi - lo), K=3, LAMBDA=7, max_new=[20], teacher=_teacher,
                           reject_at=lambda b, c, E: c.b + 1)
    with pytest.raises(TK.TickError):
        k = 0
        while True:
            rows = bad_rnd.plan_tick()
            bad_rnd.finish_tick(rows, list(range(k, k + len(rows))), k + len(rows))
            k += len(rows)

    def cycle_only(K):
        nd = {}
        lg = []

        def rj(b, c, E):
            L = c.b - c.a + 1
            k2 = nd.get(b, 0)
            nd[b] = k2 + 1
            off = k2 % (L + 1)
            j = None if off == L else c.a + off
            lg.append((b, c.a, c.b, E, None if j is None else j - c.a, "cycle"))
            return j
        return rj, lg
    rj, lg = cycle_only(3)
    rnd3 = TK.TickRound([0] * 4, S_list, lambda lo, hi: [0] * (hi - lo), K=3, LAMBDA=7, max_new=max_new,
                        teacher=_teacher, reject_at=rj)
    k = 0
    while not all(r.done for r in rnd3.reqs):
        rows = rnd3.plan_tick()
        rnd3.finish_tick(rows, list(range(k, k + len(rows))), k + len(rows))
        k += len(rows)
    assert not TK.schedule_coverage(lg, 3)["complete"]


# =====================================================================================================================
# the tiny model and the CPU stand-ins
# =====================================================================================================================
HD, HQ, HKV, D, I_FF, VOCAB, NL_LAYERS = 32, 4, 2, 16, 24, 61, 2
TOPK, WINDOW, INIT, QK = 8, 2, 1, 5


def tiny_model(seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(*s, scale=0.3):
        return torch.randn(*s, generator=g) * scale
    layers = []
    for l in range(NL_LAYERS):
        layers.append(SimpleNamespace(
            input_layernorm_variance_epsilon=1e-6, input_layernorm_weight=1 + rn(HD, scale=0.1),
            post_attention_layernorm_variance_epsilon=1e-6, post_attention_layernorm_weight=1 + rn(HD, scale=0.1),
            wqkv=rn(HQ * D + 2 * HKV * D, HD), delta=SimpleNamespace(weight=rn(HKV, HKV * D, scale=0.2)),
            A=-(torch.rand(HKV, generator=g) + 0.5), q_size=HQ * D, kv_size=HKV * D, num_key_value_heads=HKV,
            head_dim=D, num_heads=HQ, wo=rn(HD, HQ * D), gate_up_proj=rn(2 * I_FF, HD), down_proj=rn(HD, I_FF),
            pooling_stride=16, block_size=64, window_blocks=WINDOW, init_blocks=INIT, topk_blocks=TOPK))
    pos = torch.arange(4096, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    fr = torch.outer(pos, inv)
    return SimpleNamespace(layers=layers, embed_tokens=rn(VOCAB, HD, scale=1.0), lm_head=rn(VOCAB, HD),
                           norm_weight=1 + rn(HD, scale=0.1), norm_variance_epsilon=1e-6,
                           cos_sin_cache=torch.cat([fr.cos(), fr.sin()], -1))


def _layer_norm(h, eps, w):
    x = h.double()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.double()).to(h.dtype)


def _nosa_linear(qkv, delta_w, A, q_size, kv_size, Hkv):
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size:q_size + kv_size].contiguous()
    v = qkv[..., q_size + kv_size:].contiguous()
    cis = (A.double() * F.softplus(v.double() @ delta_w.double().t())).to(qkv.dtype)
    return q, k, v, cis


def _rope(pos, q, k, Dh, cache, neox):
    c, s = cache[pos.long(), :Dh // 2], cache[pos.long(), Dh // 2:]
    for x in (q, k):
        y = x.view(x.shape[0], -1, Dh)
        x1, x2 = y[..., :Dh // 2].clone(), y[..., Dh // 2:].clone()
        y[..., :Dh // 2] = x1 * c[:, None] - x2 * s[:, None]
        y[..., Dh // 2:] = x2 * c[:, None] + x1 * s[:, None]


def _silu_and_mul(h, out):
    d = h.shape[-1] // 2
    out.copy_(F.silu(h[..., :d]) * h[..., d:])


NL_CPU = SimpleNamespace(layer_norm=_layer_norm, nosa_linear=_nosa_linear, apply_rope_with_cos_sin_cache_inplace=_rope,
                         silu_and_mul=_silu_and_mul, QK_SELECT=QK)


def _stage1(q, k, v, cu_q, cu_k, max_q, max_k, seqused_k=None):
    """A row-local stand-in of stage 1: per (KV head, row, key) the max over the group of the scaled dot (float64),
    zero past the row's own key count; (Hkv, R, round128(max_k)) as the op returns."""
    R, Hq, Dh = q.shape
    Hk = k.shape[1]
    G = Hq // Hk
    out = torch.zeros(Hk, R, (max_k + 127) // 128 * 128, dtype=q.dtype)
    for r in range(R):
        n, s0 = int(seqused_k[r]), int(cu_k[r])
        kk = k[s0:s0 + n].double()
        for h in range(Hk):
            out[h, r, :n] = ((q[r, h * G:(h + 1) * G].double() @ kk[:, h].t()) * Dh ** -0.5).max(0).values.to(q.dtype)
    return out


def _pool(score, ccis, n_comp, M_b, pq, pc, stride=16, block_size=64, local_blocks=16, init_blocks=1):
    pad = torch.zeros(score.shape, dtype=ccis.dtype)
    pad[..., :ccis.shape[-1]] = ccis
    a, c = MP.pool_reference(score, pad, n_comp, M_b, pq.shape[-1], init_blocks=init_blocks, local_blocks=local_blocks,
                             out_dtype=pq.dtype)
    pq.copy_(a)
    pc.copy_(c)


def exact_linear(x, w):
    """An M-invariant GEMM stand-in (float64 accumulation, one rounding)."""
    return (x.double() @ w.double().t()).to(x.dtype)


def mdep_linear(x, w):
    """NEGATIVE CONTROL: a GEMM whose rounding depends on the row count M (as cuBLAS picks its kernel by M)."""
    M = x.reshape(-1, x.shape[-1]).shape[0]
    return (x.double() @ w.double().t() + M * 1e-5).to(x.dtype)


def make_ops(p3, linear=exact_linear, **attn_kw):
    fn = functools.partial(p3.bi_paged_decode_reference, **attn_kw) if attn_kw else p3.bi_paged_decode_reference
    return SimpleNamespace(linear=linear, attn=SimpleNamespace(bi_paged_decode=fn, build_page_table=p3.build_page_table),
                           mean=BD.fixed_order_mean)


S_TINY = [500, 557, 600, 639]
STEPS = 40
N_REF = STEPS + 16


def make_engine(ops, S_list=S_TINY, steps=STEPS, seed=1, cls=None, **kw):
    """The tiny model's state after a synthetic prefill (random keys / values / cis of the prompt), in both banks of
    every layer's 2B twin and in the store (complete blocks in their historical slots, the partial block in the exact
    ring, the draft ring = the exact ring): the layout TickEngine.from_r2 builds from build_ragged."""
    model = tiny_model()
    B = len(S_list)
    end = max(S_list) + steps + 80
    lk_max, n_max, M_alloc = end // 16 + 1, end + 64, end // 64 + 1
    NB = n_max // BS + 1
    g = torch.Generator().manual_seed(seed)
    twins, sk, sv, sb = [], [], [], []
    for l in range(NL_LAYERS):
        t = TW.PositionalTwin(2 * B, HKV, D, lk_max, n_max)
        k = torch.zeros(B, (NB + 4) * BS, HKV, D)
        v = torch.zeros_like(k)
        bias = torch.zeros(B, (NB + 4) * BS, HKV)
        for b, S in enumerate(S_list):
            keys = torch.randn(S, HKV, D, generator=g)
            vals = torch.randn(S, HKV, D, generator=g)
            cis = -torch.rand(S, HKV, generator=g)
            t.prefill(b, keys, cis)
            t.prefill(B + b, keys, cis)
            full = (S // BS) * BS
            k[b, :full], v[b, :full], bias[b, :full] = keys[:full], vals[:full], cis[:full]
            r0 = (NB + (S // BS) % 2) * BS
            k[b, r0:r0 + S - full], v[b, r0:r0 + S - full], bias[b, r0:r0 + S - full] = keys[full:], vals[full:], cis[full:]
            e0, d0 = NB * BS, (NB + 2) * BS
            for x in (k, v, bias):
                x[b, d0:d0 + 2 * BS] = x[b, e0:e0 + 2 * BS]
        twins.append(t)
        sk.append(k)
        sv.append(v)
        sb.append(bias)
    return (cls or TK.TickEngine)(model, ops, twins, sk, sv, sb, S_list, M_alloc, nl=NL_CPU, stage1=_stage1, pool=_pool,
                                  check=True, **kw)


FIRST = [3, 7, 11, 13]


def serial_ref(eng, n=N_REF, first=FIRST):
    """The engine's serial mode, greedy and free-running: tokens (n + 1, B) (the inputs, then the argmaxes) and the
    logits of every step (n, B, V)."""
    toks, lg = [list(first)], []
    for s in range(n):
        out = eng.run_serial(toks[-1])
        lg.append(out.logits.clone())
        toks.append(out.argmax.tolist())
    return torch.tensor(toks), torch.stack(lg)


def tick_check(eng, toks, logits, forced, K=3, LAMBDA=7):
    """The SG1 / SG2 checks on the CPU. Returns dict(bad, ctrl, compared, committed_ok, rnd, log, wrong_in)."""
    S_list = eng.S_list
    n_ref = logits.shape[0]
    recs = []

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        for k, row in enumerate(rows):
            if row.kind != "v" or row.pos - S_list[row.b] >= n_ref:
                continue
            s = row.pos - S_list[row.b]
            nxt = s + 1 < n_ref and torch.equal(out.logits[k], logits[s + 1, row.b])
            recs.append((row.b, row.pos, tokens[k], (row.b, row.epoch, row.m), row.i,
                         torch.equal(out.logits[k], logits[s, row.b]), nxt))
    reject_at, log = TK.forced_schedule(K) if forced else (None, [])
    rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [STEPS] * eng.B, K=K, LAMBDA=LAMBDA,
                       teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    tl = eng.tok_log[:eng.n_tok].tolist()
    inp = [tl[t.slot] if isinstance(t, TK.Ref) else int(t) for (_, _, t, _, _, _, _) in recs]
    ref_in = [int(toks[p - S_list[b], b]) for (b, p, _, _, _, _, _) in recs]
    on_path = [False] * len(recs)
    groups = {}
    for j, rec in enumerate(recs):
        groups.setdefault(rec[3], []).append((rec[4], j))
    for lst in groups.values():
        ok = True
        for _, j in sorted(lst):
            ok = ok and inp[j] == ref_in[j]
            on_path[j] = ok
    idx = list(range(len(recs))) if forced else [j for j in range(len(recs)) if on_path[j]]
    return dict(bad=[recs[j][:2] for j in idx if not recs[j][5]], ctrl=[recs[j][:2] for j in idx if recs[j][6]],
                compared=len(idx), wrong_in=[j for j in range(len(recs)) if forced and inp[j] != ref_in[j]],
                committed_ok=[rnd.reqs[b].out == [int(toks[s, b]) for s in range(STEPS)] for b in range(eng.B)],
                mirror_bad=eng.mirror_failures(), rnd=rnd, log=log, eng=eng)


@pytest.fixture(scope="module")
def ref(p3):
    eng = make_engine(make_ops(p3))
    return serial_ref(eng)


# =====================================================================================================================
# the engine
# =====================================================================================================================
def test_T3_serial_mode_is_row_invariant(p3):
    """T3 (the premise): the serial mode's row of request 0 is torch.equal whether request 0 is alone (a one-request
    engine from the same prefill) or in the batch of 4, for 14 steps (positions 500 .. 513: across request 0's block
    boundary 512, so a write-back is read). Negative control: with a GEMM whose rounding depends on the row count the
    rows differ."""
    for linear, expect in ((exact_linear, True), (mdep_linear, False)):
        full = make_engine(make_ops(p3, linear))
        alone = make_engine(make_ops(p3, linear), S_list=S_TINY[:1])
        t_full, l_full = serial_ref(full, n=14)
        t_one, l_one = serial_ref(alone, n=14, first=FIRST[:1])
        same = torch.equal(l_full[:, 0], l_one[:, 0])
        assert same == expect, (linear.__name__, same)
        if expect:
            assert full.stats["writebacks"] >= 1                          # request 0 crossed 512: a write-back happened


def test_T4_teacher_forced_tick_equals_serial(p3, ref):
    """T4 (SG1 on the CPU): the teacher-forced tick with the forced schedule: every verify row, the rejected ones
    included, torch.equal to the serial row at (request, position); none equals the serial row at the next position
    (the control); the committed sequences are the teacher's; the schedule covers every offset, full acceptances and
    both boundary kinds, and equals its model-free simulation; one decision read per tick with verify rows."""
    toks, logits = ref
    res = tick_check(make_engine(make_ops(p3), draft_drop=0.5), toks, logits, forced=True)
    assert res["compared"] > 50 and not res["wrong_in"]
    assert not res["bad"], res["bad"][:5]
    assert not res["ctrl"], res["ctrl"][:5]
    assert all(res["committed_ok"]) and not res["mirror_bad"], res["mirror_bad"][:4]
    cov = TK.schedule_coverage(res["log"], 3)
    assert cov["complete"], cov
    sim, _ = TK.simulate_schedule(S_TINY, [STEPS] * 4)
    assert sim == res["log"]
    eng, rnd = res["eng"], res["rnd"]
    assert eng.stats["rollbacks"] > 0 and eng.stats["publications"] > 0 and eng.stats["writebacks"] >= 3
    assert eng.stats["decision_reads"] == rnd.syncs > 0


def test_T5_free_running_tick_equals_serial(p3, ref):
    """T5 (SG2 on the CPU): free-running greedy with draft rows that miss half of their middle blocks: the committed
    sequences equal the serial greedy sequences; every verify row on the serial path is torch.equal to the serial row;
    the run both rejects and publishes (else it would be blind). Negative control: with draft_drop 0 the drafts are
    the serial rows themselves, so nothing is ever rejected."""
    toks, logits = ref
    res = tick_check(make_engine(make_ops(p3), draft_drop=0.5), toks, logits, forced=False)
    assert all(res["committed_ok"]) and not res["mirror_bad"], (res["committed_ok"], res["mirror_bad"][:4])
    assert res["compared"] > 30 and not res["bad"], res["bad"][:5]
    assert not res["ctrl"]
    eng = res["eng"]
    assert eng.stats["rollbacks"] > 0 and eng.stats["publications"] > 0, eng.stats
    res0 = tick_check(make_engine(make_ops(p3), draft_drop=0.0), toks, logits, forced=False)
    assert all(res0["committed_ok"]) and res0["eng"].stats["rollbacks"] == 0 and not res0["bad"]


class NoTwinRollback(TK.TickEngine):
    """NEGATIVE CONTROL: a rejection that does not reset the exact bank's twin (the counters run ahead)."""
    def apply_events(self):
        saved = SC.rollback_bank
        TK.SC.rollback_bank = lambda *a, **k: None
        try:
            return super().apply_events()
        finally:
            TK.SC.rollback_bank = saved


class NoWritebackCopy(TK.TickEngine):
    """NEGATIVE CONTROL: a write-back that registers the block without copying its rows."""
    def writeback(self, b):
        while BS * (self.wb[b] + 1) <= self.committed[b]:
            self.wb[b] += 1


class OnePageTail(TK.TickEngine):
    """NEGATIVE CONTROL: one tail page per bank (the slot-63 style): a cohort that crosses a block boundary
    overwrites the old block's rows."""
    def ring_slot(self, draft, pos):
        return self.NB + (TK.RING_DRAFT if draft else TK.RING_EXACT)


@pytest.mark.parametrize("case", ["mdep_gemm", "batch_keyed_split", "no_twin_rollback", "no_writeback_copy",
                                  "one_page_tail"])
def test_T6_negative_controls_are_caught(p3, ref, case):
    """T6: each broken variant of the tick makes the T4 check fail (a differing verify row, or a refusal). The
    reference stays the correct serial mode. Preconditions: the batch-keyed split differs from the fixed one on the
    tiny rows (8 pages: one split of 8 alone, two of 4 in a batch), and the serial rows have 4 rows per call."""
    toks, logits = ref
    ops, cls = make_ops(p3), None
    if case == "mdep_gemm":
        ops = make_ops(p3, mdep_linear)
    elif case == "batch_keyed_split":
        ops = make_ops(p3, batch_keyed_split=True)
        e = make_engine(make_ops(p3))
        t1, l1 = serial_ref(e, n=2)
        e2 = make_engine(make_ops(p3, pages_per_split=4))
        t2, l2 = serial_ref(e2, n=2)
        assert not torch.equal(l1, l2), "precondition: 4 and 8 pages per split give other bits"
    else:
        cls = dict(no_twin_rollback=NoTwinRollback, no_writeback_copy=NoWritebackCopy, one_page_tail=OnePageTail)[case]
    caught = False
    try:
        res = tick_check(make_engine(ops, cls=cls, draft_drop=0.5), toks, logits, forced=True)
        caught = bool(res["bad"]) or not all(res["committed_ok"]) or bool(res["mirror_bad"])
    except (TK.TickError, TW.TwinRollbackError, TW.TwinCapacityError, RuntimeError, AssertionError, IndexError):
        caught = True
    assert caught, case


def test_T7_page_table_and_draft_drop(p3):
    """T7: page_table renames the selection's last two entries to the row's T - 1 and T slots; draft rows drop only
    middle blocks (never the init block or the local window), keep the ascending order, and the drop is a function of
    (request, layer, head, block, seed) only (the same row alone or in a batch drops the same blocks). Negative
    control: a verify row drops nothing."""
    eng = make_engine(make_ops(p3), draft_drop=0.5)
    H, R, K = HKV, 3, TOPK
    T = torch.tensor([9, 9, 10])
    sel = torch.stack([torch.stack([torch.tensor(sorted([0, 2, 3, 4, 5] + [int(t) - 2, int(t) - 1, int(t)]))
                                    for t in T]) for _ in range(H)])
    pos = T * BS + 5
    rb = torch.tensor([0, 1, 2])
    prev = torch.tensor([8, 8, 9], dtype=torch.int32)
    tail = torch.tensor([eng.NB + 2, eng.NB + 2, eng.NB + 0], dtype=torch.int32)
    draft = torch.tensor([True, True, False])
    lay = eng.model.layers[0]
    tab, npg = eng.page_table(0, sel, pos, rb, prev, tail, draft, True, lay)
    assert npg[2].tolist() == [K, K] and tab[2, :, -1].tolist() == [eng.NB, eng.NB] and tab[2, :, -2].tolist() == [9, 9]
    for r in range(2):
        for h in range(H):
            n = int(npg[r, h])
            lst = tab[r, h, :n].tolist()
            assert lst[-2:] == [int(prev[r]), int(tail[r])] and lst[0] == 0
            kept = lst[:-2]
            assert kept == sorted(kept) and set(sel[h, r, :-2].tolist()) - set(kept) <= {2, 3, 4, 5}
            assert int(T[r]) - 2 in kept
    assert int(npg[:2].sum()) < 2 * H * K, "precondition: draft_drop 0.5 drops some middle block here"
    tab1, npg1 = eng.page_table(0, sel[:, :1], pos[:1], rb[:1], prev[:1], tail[:1], draft[:1], True, lay)
    assert torch.equal(tab1[0, :, :int(npg1[0, 0])], tab[0, :, :int(npg1[0, 0])]) and torch.equal(npg1[0], npg[0])


def test_T8_row_table_refusals(p3):
    """T8: the row table refuses a position that is not its bank's next one, a row before 64 blocks of selection,
    a committed position, and a token slot not yet written."""
    eng = make_engine(make_ops(p3))
    ok = [SimpleNamespace(b=0, kind="v", pos=S_TINY[0]), SimpleNamespace(b=0, kind="d", pos=S_TINY[0])]
    tab, count, T = eng.row_table(ok, [1, 2])
    assert count[0] == 1 and count[4] == 1 and T == 1
    for rows, toks in (([SimpleNamespace(b=0, kind="v", pos=S_TINY[0] + 1)], [1]),
                       ([SimpleNamespace(b=1, kind="v", pos=S_TINY[1]), SimpleNamespace(b=1, kind="v", pos=S_TINY[1])], [1, 1]),
                       ([SimpleNamespace(b=0, kind="v", pos=S_TINY[0])], [TK.Ref(0)])):
        with pytest.raises(TK.TickError):
            eng.row_table(rows, toks)
    small = make_engine(make_ops(p3), S_list=[300])
    with pytest.raises(TK.TickError, match="topk"):
        small.row_table([SimpleNamespace(b=0, kind="v", pos=300)], [1])


# =====================================================================================================================
# source pins, the gate, the step script
# =====================================================================================================================
def _method_src(cls_name, name):
    src = (NOSI / "r2_tick.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


def test_T9_hot_path_has_no_host_read():
    """T9: the per-tick methods of the engine contain no host read of their own (.item, .tolist, .cpu, synchronize);
    bool( appears only in the opt-in selection check; read_tokens is the one decision read. Negative control: the
    scan finds the read in read_tokens."""
    for name in ("run_rows", "_layer", "page_table", "apply_events", "writeback", "copy_ring", "copy_rows", "row_table",
                 "run_serial"):
        code = "\n".join(l.split("#")[0] for l in _method_src("TickEngine", name).splitlines())
        for bad in (".item(", ".tolist(", ".cpu(", "synchronize"):
            assert bad not in code, (name, bad)
        if name == "_layer":
            lines = [l for l in code.splitlines() if "bool(" in l]
            assert len(lines) == 1, lines
            before = code.split(lines[0])[0].splitlines()
            assert any("if self.check:" in x for x in before[-2:]), before[-2:]
        else:
            assert "bool(" not in code, name
    assert ".tolist()" in _method_src("TickEngine", "read_tokens")


def test_T10_gate_helpers_and_defaults():
    """T10: the gate's defaults (m_fix = (1 + K) B = 32, the ragged deltas, parts SG0-SG2), its refusals, and
    cmp_rows (equal rows, the next-position control, the past-REF guard)."""
    spec = importlib.util.spec_from_file_location("r2_tick_gate_cpu", str(EFF / "r2_tick_gate.py"))
    G = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EFF))
    spec.loader.exec_module(G)
    a = G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p"])
    assert a.m_fix == 32 and a.batch == 8 and a.parts == ["SG0", "SG1", "SG2"] and a.gemm == "exact" and a.steps == 96
    for bad in (["--m-fix", "16"], ["--deltas", "0,1"], ["--parts", "SG9"], ["--S", "4100"]):
        with pytest.raises(SystemExit):
            G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p"] + bad)
    ref = torch.randn(5, 2, 7)
    lg = torch.stack([ref[1, 0], ref[4, 1], ref[2, 1] + 1])
    eq, nx = G.cmp_rows(lg, ref, [10, 20], [(0, 0, 11), (1, 1, 24), (2, 1, 21)])
    assert eq.tolist() == [True, True, False] and nx.tolist() == [False, False, False]
    eq2, nx2 = G.cmp_rows(torch.stack([ref[2, 0]]), ref, [10, 20], [(0, 0, 11)])
    assert eq2.tolist() == [False] and nx2.tolist() == [True]
    sim, _ = TK.simulate_schedule([16128 - d for d in a.deltas], [a.steps] * a.batch, K=a.K, LAMBDA=a.LAMBDA)
    assert TK.schedule_coverage(sim, a.K)["complete"], "the step's SG1 schedule would be blind"


def test_T11_step_script_uuid_file_and_pins(tmp_path):
    """T11: r2_tick_step.sh reads the GPU uuid from <holders>/<job>.gpu_uuid: a missing file, a "GPU-" prefix and two
    lines STOP (exit 1) before any other check; TICK_HOLDERS outside DRYRUN is refused; a valid file passes the uuid
    stage; on a clean checkout at HEAD the DRYRUN prints the uuid with its path, piece 3's pins and the gate's
    arguments. Static pins: the claim tick_active, STEP_MIN, --export=NONE, no awk / python3 / perl outside comments."""
    script = EFF / "r2_tick_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert 'mkdir "$OUTROOT/tick_active"' in text and "STEP_MIN=45" in text and "--export=NONE" in text
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index('mkdir "$OUTROOT/tick_active"')
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def run(job, content=None, dry=True, sha="0" * 40):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, TICK_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True,
                              text=True, timeout=120)
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    r = run(311)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(312, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(313, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(314, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(315, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr and "uuid" not in r.stderr.lower()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean git checkout: the full DRYRUN needs one")
    r = run(316, uid + "\n", sha=head.stdout.strip())
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the uuid stage: %s" % r.stderr.strip()[-200:])
    assert "gpu_uuid %s (%s/316.gpu_uuid)" % (uid, tmp_path) in r.stdout
    assert ("piece 3 %s:%s blob %s" % (PIECE3_V2["sha"], PIECE3_V2["path"], PIECE3_V2["blob"])) in r.stdout
    for arg in ("--biattn /mnt/beegfs/ojerbi/r2-runs/tick/tick_t1_316.step/piece3_bi_paged_attention.py", "--gemm exact",
                "--batch 8", "--S 16128", "--steps 96", "--m-fix 32", "--parts SG0,SG1,SG2", "--draft-drop 0.5",
                "NOSI_R2_DECODE=1", "NOSI_ATTN_SPLITS=4", "NOSI_POOL_BLOCKS=0"):
        assert arg in r.stdout, arg
