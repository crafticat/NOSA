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
import re
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


def test_T2a_forced_schedule_kinds():
    """T2a: forced_schedule's four kinds on constructed cohorts (block 8; row i wrote position E + i - 1; a rejection
    at j keeps the positions < E + j): a cohort far from a boundary cycles its offsets (0, then 1, then accepted
    whole); a cohort ending 1 or 2 positions before the boundary X is STEERED to j = X - 1 - E (the next epoch starts at
    X - 1); a crossing cohort is rejected at the row that wrote X (cross_new: X kept) and, the next time, at the row
    before it (cross_old: X rolled back)."""
    C = SC.Cohort
    rj, log = TK.forced_schedule(3, block=8)
    assert rj(0, C(1, 1, 2), 0) == 1 and rj(0, C(1, 1, 2), 0) == 2 and rj(0, C(1, 1, 2), 0) is None   # cycle 0, 1, L
    assert rj(1, C(1, 1, 2), 5) == 2                                    # rows at 5, 6; X = 8: steer, keep < 7
    assert rj(1, C(2, 3, 5), 3) == 4                                    # rows at 5, 6, 7; X = 8: steer, keep < 7
    assert rj(2, C(1, 1, 2), 7) == 2                                    # rows at 7, 8: cross_new, keep < 9 (8 kept)
    assert rj(3, C(2, 3, 5), 4) == 4                                    # rows at 6, 7, 8: cross_old, keep < 8
    assert [e[5] for e in log] == ["cycle"] * 3 + ["steer", "steer", "cross_new", "cross_old"]
    assert [e[4] for e in log] == [0, 1, None, 1, 1, 1, 1]


def test_T2_forced_schedule_covers_every_kind_and_commits_the_teacher():
    """T2: the teacher-forced round with forced_schedule (block 32 here: several boundaries per request, room for the
    cycle between them) commits the teacher's tokens; its log covers every rejection offset, full acceptances, and
    both boundary kinds; every steered decision makes the next epoch start at X - 1; every cross_new decision keeps
    exactly the new block's first row, every cross_old one keeps none of the new block; simulate_schedule gives the
    same log. Negative controls: a reject_at outside the cohort is refused; a schedule without steering and boundary
    handling is not complete for these lengths. (Block 8, CPU job 2190867: almost every cohort crosses or is steered,
    so the cycle never reached every offset: the schedule needs room between boundaries, as at block 64.)"""
    S_list, max_new, blk = [10, 21, 30, 44], [40, 33, 57, 21], 32
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
    assert cov["complete"], cov
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
    out.copy_((F.silu(h[..., :d].double()) * h[..., d:].double()).to(out.dtype))   # float64: no vector-tail path


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


def make_engine(ops, S_list=S_TINY, steps=STEPS, seed=1, cls=None, width_S=max(S_TINY), **kw):
    """The tiny model's state after a synthetic prefill (random keys / values / cis of the prompt), in both banks of
    every layer's 2B twin and in the store (complete blocks in their historical slots, the partial block in the exact
    ring, the draft ring = the exact ring): the layout TickEngine.from_r2 builds from build_ragged. Request b's data
    of layer l come from their OWN seeded generator and the widths (Lk_max, N_max, M_alloc) from width_S, so a request
    has the same state in any batch (CPU job 2190867: one shared generator gave request 0 other layer-1 data alone)."""
    model = tiny_model()
    B = len(S_list)
    end = max(width_S, max(S_list)) + steps + 80
    lk_max, n_max, M_alloc = end // 16 + 1, end + 64, end // 64 + 1
    NB = n_max // BS + 1
    twins, sk, sv, sb = [], [], [], []
    for l in range(NL_LAYERS):
        t = TW.PositionalTwin(2 * B, HKV, D, lk_max, n_max)
        k = torch.zeros(B, (NB + 4) * BS, HKV, D)
        v = torch.zeros_like(k)
        bias = torch.zeros(B, (NB + 4) * BS, HKV)
        for b, S in enumerate(S_list):
            g = torch.Generator().manual_seed(seed * 100003 + l * 1009 + b)
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


RES = _load("r2_residency")
RSB = _load("r2_residency_sb")
RSC = _load("r2_residency_sc")
OWN2 = _load("r2_own2")                                                    # [b2] OwnResidency (B's B2)
KTR = sys.modules[PKG + ".kstep.transfer"]


class Ev:
    """A CPU event (K3X's backend shape: .done); the CPU's reads are complete when recorded."""

    def __init__(self, done=True):
        self.done = done


class Be:
    def __init__(self):
        self.waits = 0

    def host_wait(self, ev):
        if ev is not None and not ev.done:
            ev.done = True
            self.waits += 1


def make_res_engine(ops, kind="v1", C=12, X=24, R_ver=4, S_list=S_TINY, steps=STEPS, seed=1, backend_cls=None,
                    mode="delay", svc_kw=None, **kw):
    """make_engine's state on the RESIDENCY layout (TickEngine.from_r2(hist_slots=C + X)): C + X empty historical
    slots per stream, the same tail rings; the prompt's K / V in the host rows; piece 5's service (kind v1 / sb / sc,
    C's 2ec08da; [b2] b2 = OwnResidency, checked unless svc_kw says otherwise) with K3X's adversarial transport (seeded
    delays, reordering, bursts) over the engine's store, the cis tables = the twins' exact banks; a ResidencyBackend
    attached. [b2] svc_kw: more service arguments (b2: checked, b2_mut)."""
    ref = make_engine(ops, S_list=S_list, steps=steps, seed=seed)          # the twins and the data, then re-laid out
    B, NBr = len(S_list), C + X
    n_host = (ref.NB + 1) * BS
    host_k = [torch.zeros(B, n_host, HKV, D) for _ in range(NL_LAYERS)]
    host_v = [torch.zeros(B, n_host, HKV, D) for _ in range(NL_LAYERS)]
    sk, sv, sb = [], [], []
    for l in range(NL_LAYERS):
        k = torch.zeros(B, (NBr + 4) * BS, HKV, D)
        v = torch.zeros_like(k)
        bias = torch.zeros(B, (NBr + 4) * BS, HKV)
        for b, S in enumerate(S_list):
            full = (S // BS) * BS
            host_k[l][b, :full] = ref.store_k[l][b, :full]                  # the prompt's complete blocks
            host_v[l][b, :full] = ref.store_v[l][b, :full]
            e_ref, e_new = ref.NB * BS, NBr * BS                           # the four tail pages, as laid out
            k[b, e_new:e_new + 4 * BS] = ref.store_k[l][b, e_ref:e_ref + 4 * BS]
            v[b, e_new:e_new + 4 * BS] = ref.store_v[l][b, e_ref:e_ref + 4 * BS]
            bias[b, e_new:e_new + 4 * BS] = ref.store_bias[l][b, e_ref:e_ref + 4 * BS]
            if S > full:                                                   # the partial block's prompt rows too
                r0 = (ref.NB + (S // BS) % 2) * BS
                host_k[l][b, full:S] = ref.store_k[l][b, r0:r0 + S - full]
                host_v[l][b, full:S] = ref.store_v[l][b, r0:r0 + S - full]
        sk.append(k)
        sv.append(v)
        sb.append(bias)
    eng = TK.TickEngine(ref.model, ops, ref.twins, sk, sv, sb, S_list, ref.M_alloc, nl=NL_CPU, stage1=_stage1,
                        pool=_pool, check=True, **kw)
    be = Be()
    tp = KTR.AdversarialTransport(be, host_k, host_v, sk, sv, BS, seed, mode=mode, max_delay=3)
    gather = RES.TorchGather(host_k, host_v, sk, sv)
    cis = [t.total_cis[:B] for t in eng.twins]
    args = (NL_LAYERS, B, HKV, C, X, R_ver, tp, gather)
    skw = dict(store_b=sb, cis=cis, block_bytes=BS * 2 * D * 4, host_wait=be.host_wait)
    skw.update(svc_kw or {})                                                # [b2]
    if kind == "v1":
        svc = RES.Residency(*args, **skw)
    elif kind == "sb":
        svc = RSB.ResidencySB(*args, device="cpu", nblk=n_host // BS, **skw)
    elif kind == "b2":                                                      # [b2] B2 on the CPU stand-in device
        svc = OWN2.OwnResidency(*args, device="cpu", nblk=n_host // BS, **skw)
    else:
        svc = RSC.ResidencySC(*args, device="cpu", nblk=n_host // BS, **skw)
    eng.attach_backend((backend_cls or TK.ResidencyBackend)(eng, svc, host_k, host_v, lambda: Ev(True), 3))
    eng.res_world = SimpleNamespace(svc=svc, tp=tp, be=be, host_k=host_k, host_v=host_v)
    return eng


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


def tick_check(eng, toks, logits, forced, K=3, LAMBDA=7, ready=None):
    """The SG1 / SG2 / SG3 checks on the CPU. Returns dict(bad, ctrl, compared, committed_ok, rnd, log, wrong_in,
    comps: per compared row (own draft, other verify, other draft, R, equal))."""
    S_list = eng.S_list
    n_ref = logits.shape[0]
    recs = []
    comps = []

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        comp = TK.compositions(rows)
        for k, row in enumerate(rows):
            if row.kind != "v" or row.pos - S_list[row.b] >= n_ref:
                continue
            s = row.pos - S_list[row.b]
            nxt = s + 1 < n_ref and torch.equal(out.logits[k], logits[s + 1, row.b])
            same = torch.equal(out.logits[k], logits[s, row.b])
            recs.append((row.b, row.pos, tokens[k], (row.b, row.epoch, row.m), row.i, same, nxt))
            comps.append(comp[k][1:] + (same,))
    reject_at, log = TK.forced_schedule(K) if forced else (None, [])
    rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [STEPS] * eng.B, K=K, LAMBDA=LAMBDA,
                       teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick, ready=ready)
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
                mirror_bad=eng.mirror_failures(), rnd=rnd, log=log, eng=eng, comps=comps)


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


def test_T13_phase_mix_equals_serial(p3, ref):
    """T13 (SG3 on the CPU): the teacher-forced tick with the requests' cohorts staggered over 3 ticks (phase_ready):
    every verify row torch.equal to the serial row in every composition class; the run holds verify rows with and
    without their own draft row in the tick, with and without another request's verify rows, and >= 3 tick sizes; it
    has empty ticks or stalls handled (the round completes). Negative control: with the M-dependent GEMM some class
    differs."""
    toks, logits = ref
    res = tick_check(make_engine(make_ops(p3), draft_drop=0.5), toks, logits, forced=True, ready=TK.phase_ready(3))
    assert res["compared"] > 50 and not res["bad"], res["bad"][:5]
    assert all(res["committed_ok"]) and not res["mirror_bad"] and not res["ctrl"]
    comps = res["comps"]
    assert any(c[0] for c in comps) and any(not c[0] for c in comps), "own draft row with / without"
    assert any(c[1] for c in comps) and any(not c[1] for c in comps), "another request's verify rows with / without"
    assert len({c[3] for c in comps}) >= 3
    neg = tick_check(make_engine(make_ops(p3, mdep_linear), draft_drop=0.5), toks, logits, forced=True,
                     ready=TK.phase_ready(3))
    assert neg["bad"]


def test_T14_negative_ops_and_phase_helpers(p3):
    """T14: the SG4 negatives' ops and the SG3 helpers. BatchKeyedSplit runs piece 3 with 8 pages per split for
    <= `rows` rows and 4 above (equal to the direct calls; the two differ on 8-page rows: the precondition);
    topk_select equals select_blocks where no value ties at a cut and returns ascending ids (on tie-heavy rows the
    two may differ: reported, not required, as the CPU kernel may break ties like the stable rule); phase_ready gates by
    (tick + request) % period; compositions classifies the rows."""
    g = torch.Generator().manual_seed(14)
    R, Hk, Dh, S = 6, 2, 16, 12
    q = torch.randn(R, Hk * 2, Dh, generator=g)
    k = torch.randn(2, S * BS, Hk, Dh, generator=g)
    v = torch.randn(2, S * BS, Hk, Dh, generator=g)
    bias = torch.rand(2, S * BS, Hk, generator=g)
    req = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32)
    pt = torch.stack([torch.stack([torch.sort(torch.randperm(S, generator=g)[:8]).values for _ in range(Hk)])
                      for _ in range(R)]).to(torch.int32)
    npg = torch.full((R, Hk), 8, dtype=torch.int32)
    last = torch.tensor([5, 64, 1, 33, 17, 40], dtype=torch.int32)
    bk = TK.BatchKeyedSplit(SimpleNamespace(bi_paged_decode=p3.bi_paged_decode_reference,
                                            build_page_table=p3.build_page_table), rows=4)
    args = (k, v, bias, req, pt, npg, last)
    alone = bk.bi_paged_decode(q[:4], k, v, bias, req[:4], pt[:4], npg[:4], last[:4])
    full = bk.bi_paged_decode(q, *args)
    assert torch.equal(alone, p3.bi_paged_decode_reference(q[:4], k, v, bias, req[:4], pt[:4], npg[:4], last[:4],
                                                           pages_per_split=8))
    assert torch.equal(full, p3.bi_paged_decode_reference(q, *args, pages_per_split=4))
    assert not torch.equal(full[:4], alone), "precondition: 4 and 8 pages per split give other bits on 8-page rows"
    pq = torch.randn(2, 5, 40, generator=g)
    pc = torch.randn(2, 5, 40, generator=g)
    pq[..., -3:] = float("inf")
    assert torch.equal(TK.topk_select(pq, pc, 6, 12), SC.select_blocks(pq, pc, 6, 12))
    out = TK.topk_select(pq, pc, 6, 12)
    assert torch.equal(out, torch.sort(out, -1).values)
    rnd = SimpleNamespace(ticks=4)
    pr = TK.phase_ready(3)
    assert [pr(rnd, b, None) for b in range(6)] == [False, False, True, False, False, True]
    rows = [SimpleNamespace(b=0, kind="d"), SimpleNamespace(b=0, kind="v"), SimpleNamespace(b=1, kind="v"),
            SimpleNamespace(b=2, kind="d")]
    assert TK.compositions(rows) == [("d", True, True, True, 4), ("v", True, True, True, 4), ("v", False, True, True, 4),
                                     ("d", True, True, True, 4)]


class NoHostCopy(TK.ResidencyBackend):
    """NEGATIVE CONTROL: a write-back that registers the block (history_written) without copying its rows to the host."""

    def commit_block(self, b, blk):
        self.svc.history_written(b, blk, self.new_event())


@pytest.mark.parametrize("kind", ["v1", "sb", "sc", "b2"])                    # [b2] b2: OwnResidency, checked
def test_T15_residency_backend_equals_the_resident_store(p3, ref, kind):
    """T15 (the SG5 seam): the teacher-forced tick on piece 5's Residency (v1 / S-B / S-C, C's 2ec08da; B2) under K3X's
    adversarial transport: every verify row torch.equal to the serial row (= the ResidentStore tick, T4), the committed
    sequences the teacher's, the twins' counters the mirrors; the service fetched blocks (transfers issued, urgent misses
    or waits seen), the decoded blocks went through the host rows (write-backs registered), drafts read READY blocks
    only. Negative control: a write-back that does not copy the rows to the host is caught."""
    toks, logits = ref
    eng = make_res_engine(make_ops(p3), kind=kind)
    res = tick_check(eng, toks, logits, forced=True)
    assert res["compared"] > 50 and not res["bad"], res["bad"][:5]
    assert all(res["committed_ok"]) and not res["mirror_bad"] and not res["ctrl"]
    svc = eng.res_world.svc
    k43 = svc.metrics.k43
    assert sum(svc.metrics.transferred.values()) > 0 and k43["layer_calls"] > 0
    assert eng.stats["writebacks"] >= 3 and not svc.uid_b                    # write-backs; every request ended
    bad = make_res_engine(make_ops(p3), kind=kind, backend_cls=NoHostCopy)
    caught = False
    try:
        res2 = tick_check(bad, toks, logits, forced=True)
        caught = bool(res2["bad"])
    except (TK.TickError, RuntimeError):
        caught = True
    assert caught


def test_T16_overflow_voids_the_tick_and_the_capacity_check(p3, ref):
    """T16: the twins advance with advance_unchecked; a set overflow flag (any layer) is read WITH the decision read and
    VOIDS the tick: TickError, nothing committed (the committed counters stay the prompt's), void_ticks counted.
    Negative control: the same run without the flag completes (T4). The row table's ONE capacity check refuses a row
    whose compressed rows would reach Lk_max."""
    toks, logits = ref
    eng = make_engine(make_ops(p3), draft_drop=0.5)
    eng.twins[1].overflow.fill_(True)
    with pytest.raises(TK.TickError, match="VOID"):
        tick_check(eng, toks, logits, forced=True)
    assert eng.void and eng.stats["void_ticks"] == 1 and eng.committed == S_TINY and eng.stats["publications"] == 0
    eng2 = make_engine(make_ops(p3))
    for t in eng2.twins:
        t.lk_max = S_TINY[0] // 16 + 1
    with pytest.raises(TK.TickError, match="capacity"):
        eng2.row_table([SimpleNamespace(b=0, kind="v", pos=S_TINY[0])], [1])
    src = (NOSI / "r2_tick.py").read_text()
    assert "twin.advance_unchecked(keys, cis_b, n_dev)" in src and "twin.advance(keys" not in src


def _planted(p3, ref, select, requests=(3,), layers=(0,), steps=20, apply=True):
    """A teacher-forced tick with ties planted at `layers` for `requests`: (verify logits in order, tie counters)."""
    toks, logits = ref
    eng = make_engine(make_ops(p3), select=select)
    eng.count_ties()
    eng.plant_ties(requests, layers)
    S_list = eng.S_list
    out_l = []

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        vi = [k for k, r in enumerate(rows) if r.kind == "v"]
        if vi:
            out_l.append(out.logits[vi].clone())
    reject_at, _ = TK.forced_schedule(3)
    saved = TK.plant_tie
    if not apply:
        TK.plant_tie = functools.partial(saved, apply=False)
    try:
        TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [steps] * eng.B, teacher=teacher,
                     reject_at=reject_at, on_tick=on_tick)
    finally:
        TK.plant_tie = saved
    return torch.cat(out_l), eng.tie_report()


def test_T17_planted_ties_precondition_and_detection(p3, ref):
    """T17 (SG4c PLANTED on the CPU): with ties planted across the top-k cut (request 3, layer 0), the engine counts a
    top-k tie for EVERY planted (row, head) pair (the precondition), and the higher-id tie rule then changes some verify
    row against the stable rule (detected by construction); torch.topk on the CPU is reported, not required (its tie
    order is the kernel's). Negative control of the precondition: a plant that reports its pairs but does not write the
    tie (apply False) leaves fewer ties than pairs; the higher-id rule's control: without any plant the counters see no
    planted pair."""
    st, ties = _planted(p3, ref, SC.select_blocks)
    planted, got = ties[0][3], ties[0][4]
    assert planted > 0 and got == planted, ties[0]
    assert all(t[3] == 0 for t in ties[1:])
    hi, _ = _planted(p3, ref, TK.high_select)
    assert hi.shape == st.shape and bool((hi != st).any(-1).any())
    _, ties0 = _planted(p3, ref, SC.select_blocks, apply=False)
    assert ties0[0][3] > 0 and ties0[0][4] < ties0[0][3], ties0[0]
    _, tiesn = _planted(p3, ref, SC.select_blocks, layers=())
    assert all(t[3] == 0 for t in tiesn)
    tk, _ = _planted(p3, ref, TK.topk_select)
    print("T17 REPORT: torch.topk on the CPU under the planted ties: %d of %d verify rows differ"
          % (int((tk != st).any(-1).sum()) if tk.shape == st.shape else -1, st.shape[0]))


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


def _same_bank(a, ka, b, kb):
    assert int(a.pos[ka]) == int(b.pos[kb]) and int(a.n_comp[ka]) == int(b.n_comp[kb]) and int(a.n_ccis[ka]) == int(b.n_ccis[kb])
    n, m, p = int(b.n_comp[kb]), int(b.n_ccis[kb]), int(b.pos[kb])
    assert torch.equal(a.comp_k[ka, :n], b.comp_k[kb, :n]) and torch.equal(a.ccis[:, ka, :m], b.ccis[:, kb, :m])
    assert torch.equal(a.total_cis[ka, :p], b.total_cis[kb, :p]) and torch.equal(a.live_keys(ka), b.live_keys(kb))


def test_T12_rebuild_from_own_rows():
    """T12 (C's Q5): after a publication up to p = 205, the draft bank rebuilt from its OWN unverified rows (positions
    205 .. 211, captured BEFORE the sync) equals, by content, a bank re-advanced from an INDEPENDENT record of the
    same keys and cis (the prompt, the 5 exact rows, then the 7 draft rows); the replayed span holds a cis compression
    (position 207) and a key compression (208), whose windows now cover the exact rows 200 .. 204; fixed_order_mean
    is the twins' mean. Negative control: rows read AFTER the sync (the exact bank's rows) differ, and so does the
    bank re-advanced with them."""
    B, H, Dh, S = 2, 2, 8, 200

    def build():
        g = torch.Generator().manual_seed(12)
        pk, pc = torch.randn(S, H, Dh, generator=g), torch.randn(S, H, generator=g)
        ek, ec = torch.randn(5, H, Dh, generator=g), torch.randn(5, H, generator=g)
        dk, dc = torch.randn(12, H, Dh, generator=g), torch.randn(12, H, generator=g)
        t = TW.PositionalTwin(2 * B, H, Dh, 40, 400)
        t.mean_fn = BD.fixed_order_mean
        for bank in range(2 * B):
            t.prefill(bank, pk, pc)
        SC.flatten_advance(t, {B + 0: (dk, dc)})                        # the draft ran ahead: positions 200 .. 211
        SC.flatten_advance(t, {0: (ek, ec)})                            # the verifier published 200 .. 204
        return t, (pk, pc, ek, ec, dk, dc)
    t, (pk, pc, ek, ec, dk, dc) = build()
    assert int(t.n_comp[0]) == 11 and int(t.n_ccis[0]) == 11 and int(t.pos[B]) == 212    # events due at 207 / 208
    got_k, got_c = TK.rebuild_from_own_rows(t, 0, B, 205, 212)
    assert torch.equal(got_k, dk[5:]) and torch.equal(got_c, dc[5:])
    ref = TW.PositionalTwin(1, H, Dh, 40, 400)
    ref.mean_fn = BD.fixed_order_mean
    ref.prefill(0, pk, pc)
    ref.advance(torch.cat([ek, dk[5:]]).unsqueeze(0), torch.cat([ec, dc[5:]]).unsqueeze(0), [12])
    assert int(ref.n_comp[0]) == 12 and int(ref.n_ccis[0]) == 12                          # both events replayed
    _same_bank(t, B + 0, ref, 0)
    bad, _ = build()                                                    # NEGATIVE: read after the sync
    SC.sync_draft_bank(bad, 0, B)
    idx = torch.arange(205, 212)
    wk, wc = bad.ring[B, idx % BS].clone(), bad.total_cis[B, idx].clone()
    assert not torch.equal(wk, dk[5:]) and not torch.equal(wc, dc[5:])
    cnt = torch.zeros(2 * B, dtype=torch.int64)
    cnt[B] = 7
    kk = torch.zeros(2 * B, 7, H, Dh)
    cc = torch.zeros(2 * B, 7, H)
    kk[B], cc[B] = wk, wc
    bad.advance(kk, cc, cnt)
    with pytest.raises(AssertionError):
        _same_bank(bad, B + 0, ref, 0)


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
    assert a.m_fix == 32 and a.batch == 8 and a.parts == ["SG0", "SG1", "SG2", "SG3", "SG4"] and a.gemm == "exact"
    assert a.steps == 96 and a.sg4_steps == 32 and a.phase == 3
    for bad in (["--m-fix", "16"], ["--deltas", "0,1"], ["--parts", "SG9"], ["--S", "4100"], ["--phase", "1"],
                ["--sg4-steps", "200"]):
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
    assert 'mkdir "$OUTROOT/tick_active"' in text and "STEP_MIN=60" in text and "--export=NONE" in text
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
                "--batch 8", "--S 16128", "--steps 96", "--m-fix 32", "--draft-drop 0.5",
                "NOSI_R2_DECODE=1", "NOSI_ATTN_SPLITS=4", "NOSI_POOL_BLOCKS=0"):
        assert arg in r.stdout, arg
    # printf %q writes a comma as "\," (CPU job 2190867; C's G8 the same): accept that form and the plain one
    assert "--sg4-steps 32" in r.stdout and "--phase 3" in r.stdout
    for plain in ("--parts SG0,SG1,SG2,SG3,SG4", "--deltas 0,1,17,63,64,65,777,4097"):
        assert plain in r.stdout or plain.replace(",", "\\,") in r.stdout, plain


# =====================================================================================================================
# SG5: the harness and its step script
# =====================================================================================================================
def _sg5_gate():
    spec = importlib.util.spec_from_file_location("r2_tick_sg5_cpu", str(EFF / "r2_tick_sg5.py"))
    G = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EFF))
    spec.loader.exec_module(G)
    return G


def test_T18_sg5_harness_on_the_cpu(p3):
    """T18: the SG5 harness's pieces on the tiny model: its defaults (B 64, the primary point C 96 / X 32 / R_ver 16,
    m_fix 4B, the three backends, both modes) and refusals; the draft-only tick on the S-C Residency with the CPU
    TickTimer (every category timed per tick: gemm, twin, sel, store, res, att, tick; host: rows, backend); its
    summary and the budget (a run against itself: every term 0). Negative control: an unknown backend is refused."""
    G = _sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base)
    assert (a.batch, a.C, a.X, a.R_ver, a.m_fix, a.backends, a.modes) == (64, 96, 32, 16, 256, ["sc", "sb", "v1"],
                                                                         ["exact", "fast"])
    assert G.parse_args(base + ["--sweep", "64:16,96:64"]).sweep_pts == [(64, 16, 8), (96, 64, 32)]
    assert a.transport == "gilfree" and G.parse_args(base + ["--transport", "gather"]).transport == "gather"
    for bad in (["--backends", "s9"], ["--modes", "slow"], ["--m-fix", "8"], ["--R-ver", "40"], ["--transport", "s9"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    gsrc = ast.get_source_segment(src_all := (EFF / "r2_tick_sg5.py").read_text(), next(
        n for n in ast.parse(src_all).body if isinstance(n, ast.ClassDef) and n.name == "GatherTransport"))
    for need in ("self.side.wait_event(ev)", "tr.hist_ev, tr.reader_ev", "def stream_wait", "flash_h2d_from_mask"):
        assert need in gsrc, need
    assert a.preflight_steps == 4                                       # the PREFLIGHT runs before the long arms
    assert src_all.index("backends = preflight(") < src_all.index("for kind in backends:")
    for bad in ("Thread(", "ThreadPool", "start_new_thread", "concurrent.futures"):   # the harness starts no thread
        assert bad not in src_all, bad                                  # of its own ([hostfix] it may NAME the current
                                                                        # thread in its gc callback)
    src = (EFF / "r2_tick_sg5.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "run_draft_only")
    assert "from nosi." not in ast.get_source_segment(src, fn)        # the engine's own module (job 2190926)
    eng = make_res_engine(make_ops(p3), kind="sc")
    eng.timer = TK.TickTimer(cuda=False)
    G.run_draft_only(eng, FIRST, 6, 3)
    rep = eng.timer.report()
    assert rep["ticks"] == 6
    for c in ("gemm", "twin", "sel", "store", "res", "att", "tick"):
        assert c in rep["device_ms"] and len(rep["device_ms"][c]) == 6, c
    for c in ("rows", "backend"):
        assert c in rep["host_ms"], c
    s = G.summarize(rep)
    assert s["device_ms"]["tick"]["n"] == 5 and s["residual_ms"] is not None
    assert all(abs(v) < 1e-9 for v in G.budget(s, s).values())
    assert eng.stats["draft_rows"] == 6 * eng.B and not eng.mirror_failures()


def test_T19_sg5_step_script_uuid_file_and_pins(tmp_path):
    """T19: r2_tick_sg5_step.sh: the uuid file rules (missing, "GPU-" prefix, two lines: STOP before any other check;
    SG5_HOLDERS outside DRYRUN refused); the pins (piece 3 v2, the packer's two blobs of f2b981a); on a clean checkout
    at HEAD the DRYRUN prints the uuid, the pins and the arguments of the primary point and the sweep, and the cells
    (_t19_cells: default = p3's arguments, p4a, p4b-<b>, p4sc64, [b2] p5b2, p6 = p5b2's EXACT with the four --variants;
    no other cell has --variants); in the step's own ROOT a DRYRUN stop FAILS.
    Static: the claim sg5_active_<holder> ([sg5par]), STEP_MIN, --export=NONE, no awk / python3 / perl outside
    comments."""
    script = EFF / "r2_tick_sg5_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    # [sg5par] the claim is per holder: CLAIM=$OUTROOT/sg5_active_<holder job>; the real mkdir / rmdir use CLAIM
    # (test_r2_evalsplit.py EV14 takes the claim through SG5_CLAIM_PROBE, with a negative control)
    assert "CLAIMNAME=sg5_active_$JOB " in text and "CLAIM=$OUTROOT/$CLAIMNAME\n" in text
    assert 'mkdir "$CLAIM" 2>/dev/null ||' in text and 'rmdir "$CLAIM" || ok=0' in text
    assert 'mkdir "$OUTROOT/sg5_active"' not in text and 'rmdir "$OUTROOT/sg5_active"' not in text
    assert "STEP_MIN=120" in text and "--export=NONE" in text
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index('mkdir "$CLAIM"')
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def run(job, content=None, dry=True, sha="0" * 40):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, SG5_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True,
                              text=True, timeout=120)
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    r = run(411)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(412, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(413, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(414, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(415, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr and "uuid" not in r.stderr.lower()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", "nosi"],
                               capture_output=True, text=True)                 # the step's own precondition (e.g. .pyc)
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the full DRYRUN needs one")
    nosi_root = next(ln for ln in lines if ln.startswith("NOSI=")).split()[0][len("NOSI="):]
    at_root = Path(nosi_root).resolve() == ROOT.resolve()             # this checkout IS the step's ROOT
    r = run(416, uid + "\n", sha=head.stdout.strip())
    if r.returncode != 0:
        # in the step's own ROOT a stop here is a FAILURE (a broken pin), never a skip
        assert not at_root, "DRYRUN stopped in the step's ROOT %s: %s" % (nosi_root, r.stderr.strip()[-300:])
        pytest.skip("DRYRUN stopped on a pin outside the uuid stage: %s" % r.stderr.strip()[-200:])
    assert "gpu_uuid %s (%s/416.gpu_uuid)" % (uid, tmp_path) in r.stdout
    assert "piece 3 %s:%s blob %s" % (PIECE3_V2["sha"], PIECE3_V2["path"], PIECE3_V2["blob"]) in r.stdout
    assert "45c56f679b017436810bea8e16fa2719d642ba54" in r.stdout and "df3e3c8026fb0a48cc2fe2597013c3f20ab1338b" in r.stdout
    for arg in ("--batch 64", "--S 16128", "--steps 48", "--C 96", "--X 32", "--R-ver 16", "--sweep-steps 24",
                "--pool-baseline", "--threads 8", "NOSI_R2_DECODE=1", "NOSI_POOL_BLOCKS=0",
                "--gilfree /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_416.step/gilfree"):
        assert arg in r.stdout, arg
    for plain in ("--modes exact,fast", "--backends sc,sb,v1", "--sweep 64:16,64:32,64:64,80:16,80:32,80:64,96:16,96:32,96:64"):
        assert plain in r.stdout or plain.replace(",", "\\,") in r.stdout, plain
    assert "--transport gilfree" in r.stdout and "--preflight-steps 4" in r.stdout
    e = dict(env, SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    r = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "416", "t1", "gather"], env=e,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--transport gather" in r.stdout and "transport gather" in r.stdout
    r = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "416", "t1", "s9"], env=e,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "usage" in r.stderr
    _t19_cells(script, e, head.stdout.strip())


# p3's frozen r2_tick_sg5.py arguments: /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p3_2190876.step/command.txt, the words after
# r2_tick_sg5.py, with "p3_2190876" -> {n} and printf %q's "\," -> ","
P3_SG5_ARGS = ("--out /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_{n} --model /mnt/beegfs/ojerbi/models/NOSA-8B "
               "--workload /mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k "
               "--biattn /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_{n}.step/piece3_bi_paged_attention.py "
               "--gilfree /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_{n}.step/gilfree "
               "--gilfree-build /mnt/beegfs/ojerbi/nosa-env/jit/sg5_{n}/gilfree_build --threads 8 --transport gilfree "
               "--batch 64 --S 16128 --steps 48 --K 3 --LAMBDA 7 --modes exact,fast --backends sc,sb,v1 --C 96 --X 32 "
               "--R-ver 16 --sweep 64:16,64:32,64:64,80:16,80:32,80:64,96:16,96:32,96:64 --sweep-steps 24 "
               "--draft-ticks 12 --shipped-steps 24 --pool-baseline --preflight-steps 4")
P3_SG5_SWEEP = " --sweep 64:16,64:32,64:64,80:16,80:32,80:64,96:16,96:32,96:64"
P3_SG5_POINTS = "--modes exact,fast --backends sc,sb,v1"


def _t19_cells(script, env, sha):
    """T19, the CELLS (the lead's two-step p4 plan): the DEFAULT prints p3's arguments token for token; p4a = p3's
    arguments without the sweep (the harness's --sweep default "" gives no sweep point); p4b-<b> = --modes exact
    --backends <b> with the 9-point sweep; p4sc64 = --backends sc --X 64, both modes, no sweep, and the harness's
    equal-memory pool C + X + 4 - 64 = 100 (68 elsewhere; printed by the DRYRUN, the harness's line pinned); [b2] p5b2
    = p4sc64's arguments with --backends b2 (unchecked: no --b2-checked), pool 100; the cell and the transport in either
    order; the cell's names (out, record, JIT, job name). Negative controls: every cell's arguments differ from p3's
    (the comparison discriminates); a bare p4b, an unknown cell or backend (p4sc32, p5sc, p5b2c), two cells (p5b2 with
    p4sc64 too), two transports, or 7 words: usage, exit 2."""
    def dry(*words):
        return subprocess.run(["bash", str(script), "--nosi", sha, "416", "t1", *words], env=env, capture_output=True,
                              text=True, timeout=120)

    def args(out):
        line = out.strip().splitlines()[-1]
        return line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    p3 = P3_SG5_ARGS.format(n="t1_416")
    r = dry()
    assert r.returncode == 0 and args(r.stdout) == p3, args(r.stdout)
    assert "--job-name=sg5-t1-416 " in r.stdout and "cell default" in r.stdout
    assert "TRITON_CACHE_DIR=/mnt/beegfs/ojerbi/nosa-env/jit/sg5_t1_416/triton" in r.stdout
    G = _sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).sweep == "" and G.parse_args(base).sweep_pts == []     # p4a: no --sweep = no sweep
    # the equal-memory baseline: the harness derives NOSI_POOL_BLOCKS from the parsed C and X (main, --pool-baseline)
    src = (EFF / "r2_tick_sg5.py").read_text()
    assert "if a.pool_baseline:\n            pool = a.C + a.X + 4 - 64\n" in src
    assert "NOSI_POOL_BLOCKS = C + X + 4 - 64 = 68" in r.stdout and "C 96 X 32 R_ver 16" in r.stdout
    want = {"p4a": P3_SG5_ARGS.format(n="t1_416_p4a").replace(P3_SG5_SWEEP, "")}
    shape = {"p4a": (["sc", "sb", "v1"], ["exact", "fast"], 0, 32)}            # backends, modes, sweep points, X
    for b in ("sc", "sb", "v1"):
        want["p4b-" + b] = P3_SG5_ARGS.format(n="t1_416_p4b-" + b).replace(P3_SG5_POINTS,
                                                                           "--modes exact --backends " + b)
        shape["p4b-" + b] = ([b], ["exact"], 9, 32)
    want["p4sc64"] = (P3_SG5_ARGS.format(n="t1_416_p4sc64").replace(P3_SG5_SWEEP, "")
                      .replace(P3_SG5_POINTS, "--modes exact,fast --backends sc").replace(" --X 32 ", " --X 64 "))
    shape["p4sc64"] = (["sc"], ["exact", "fast"], 0, 64)
    want["p5b2"] = want["p4sc64"].replace("t1_416_p4sc64", "t1_416_p5b2").replace("--backends sc ",      # [b2]
                                                                                   "--backends b2 ")
    shape["p5b2"] = (["b2"], ["exact", "fast"], 0, 64)                                             # [b2]
    P6_VARIANTS = "torch/per_call,fused/per_call,torch/batched,fused/batched"                    # [batched-rebuild]
    want["p6"] = (want["p5b2"].replace("t1_416_p5b2", "t1_416_p6").replace("--modes exact,fast ", "--modes exact ")
                  + " --variants " + P6_VARIANTS)                                                # [batched-rebuild]
    shape["p6"] = (["b2"], ["exact"], 0, 64)                                                     # [batched-rebuild]
    for cell, exp in want.items():                         # [batched-rebuild] --variants in p6 only: every other
        assert ("--variants" in exp) == (cell == "p6"), cell     # cell's arguments stay p3's / p4's / p5b2's tokens
    assert "--variants" not in p3
    assert "--b2-checked" not in want["p5b2"] and want["p5b2"] != want["p4sc64"]                  # [b2] unchecked
    for cell, exp in want.items():
        assert exp != p3 and (P3_SG5_SWEEP in exp) == (shape[cell][2] == 9)
        for words in ((cell,), (cell, "gilfree"), ("gilfree", cell)):
            r = dry(*words)
            assert r.returncode == 0, (words, r.stderr.strip()[-300:])
            got = args(r.stdout)
            assert got == exp and got != p3, (words, got)
            assert "--job-name=sg5-t1-416-%s " % cell in r.stdout and "cell %s" % cell in r.stdout
            assert "TRITON_CACHE_DIR=/mnt/beegfs/ojerbi/nosa-env/jit/sg5_t1_416_%s/triton" % cell in r.stdout
            pa = G.parse_args(got.split())                             # the harness's own reading of the cell
            backends, modes, n_sweep, X = shape[cell]
            assert pa.backends == backends and pa.modes == modes, (cell, pa.backends, pa.modes)
            assert len(pa.sweep_pts) == n_sweep and pa.steps == 48 and pa.sweep_steps == 24
            assert (pa.C, pa.X, pa.R_ver, pa.batch, pa.S, pa.transport, pa.threads) == (96, X, 16, 64, 16128,
                                                                                         "gilfree", 8)
            pool = pa.C + pa.X + 4 - 64                                # the harness's rule (asserted above)
            assert pool == (100 if cell in ("p4sc64", "p5b2", "p6") else 68)                 # [b2] p5b2, p6
            if cell == "p6":                                                                    # [batched-rebuild]
                assert pa.variant_list == [tuple(v.split("/")) for v in P6_VARIANTS.split(",")]
                assert pa.rebuild == "per_call" and pa.twin == "torch"         # the first arm: p5b2's replica
            else:                                                                               # [batched-rebuild]
                assert pa.variants == "" and pa.variant_list == [("torch", "per_call")]
            if cell in ("p5b2", "p6"):                                                          # [b2]
                assert pa.b2_checked is False                                               # the timed path
            assert "NOSI_POOL_BLOCKS = C + X + 4 - 64 = %d" % pool in r.stdout, cell
            assert "C 96 X %d R_ver 16" % X in r.stdout
    r = dry("gather", "p4a")
    assert r.returncode == 0 and args(r.stdout) == want["p4a"].replace("--transport gilfree", "--transport gather")
    for bad in (("p4b",), ("p4c",), ("p4b-s9",), ("p4a", "p4a"), ("p4a", "p4b-sb"), ("gather", "gilfree"),
                ("p4a", "gather", "x"), ("P4A",), ("p4sc32",), ("p4sc64", "p4a"),
                ("p5sc",), ("p5b2c",), ("p5b2", "p4sc64"),                                     # [b2]
                ("p6", "p5b2"), ("p6x",), ("P6",)):                                            # [batched-rebuild]
        r = dry(*bad)
        assert r.returncode == 2 and "usage" in r.stderr, bad


def test_T20_the_sweep_runs_on_the_first_backend_that_passed_its_preflight():
    """T20 (the p3 sweep flaw): with --backends sc,sb,v1 and S-C failing its stand-in preflight (an engine stand-in that
    raises a CapacityRefusal-like error for sc), preflight returns [sb, v1] and the sweep runs every point on sb,
    recording sweep_backend sb; with no backend passed it runs nothing and records the REPORT "sweep skipped: no
    backend passed"; with all passed it uses the list's first. Negative controls: a planted chooser that takes
    order[0] (p3's rule) fails the same property check; run_sweep and main contain no a.backends[0]."""
    G = _sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--sweep", "64:16,96:32", "--transport", "gather"])
    assert a.backends == ["sc", "sb", "v1"]

    class Eng:
        B = 2

        def mirror_failures(self):
            return []

    class Tp:
        def close(self):
            pass

    def engine_fn(a_, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=None, transport=None):
        if kind == "sc":
            raise RuntimeError("CapacityRefusal (stand-in): the verifier's urgent misses found no slot")
        return Eng(), None, Tp()
    gates, res = [], {}
    passed = G.preflight(a, None, None, None, [0, 0], None, None, gates, res, engine_fn=engine_fn,
                         round_fn=lambda eng, first, max_new, K, LAMBDA: SimpleNamespace(ticks=9, syncs=5))
    assert passed == ["sb", "v1"]
    pf = {g["name"]: g for g in gates}
    assert not pf["SG5_preflight_gather_sc"]["passed"] and pf["SG5_preflight_gather_sb"]["passed"]
    calls = []

    def point(ops, kind, C, X, R_ver, label):
        calls.append((kind, C, X, R_ver, label, ops))
        return dict(kind=kind, C=C)
    G.run_sweep(a, lambda: "OPS", passed, gates, res, lambda: None, point)
    assert [c[0] for c in calls] == ["sb", "sb"] and res["sweep_backend"] == "sb"
    assert [c[4] for c in calls] == ["SG5sweep_sb_C64_X16", "SG5sweep_sb_C96_X32"] and [c[3] for c in calls] == [8, 16]
    assert all(c[5] == "OPS" for c in calls) and len(res["sweep"]) == 2
    g2, r2, c2 = [], {}, []
    G.run_sweep(a, lambda: "OPS", [], g2, r2, lambda: None, lambda *x: c2.append(x))
    assert not c2 and r2["sweep_backend"] is None and r2["sweep"] == []
    assert g2[-1]["name"] == "R_SG5sweep_skipped" and not g2[-1]["must"] and g2[-1]["passed"]
    assert "sweep skipped: no backend passed" in g2[-1]["detail"]
    g3, r3, c3 = [], {}, []
    G.run_sweep(a, lambda: "OPS", ["v1", "sc", "sb"], g3, r3, lambda: None, lambda ops, kind, *x: c3.append(kind))
    assert c3 == ["sc", "sc"] and r3["sweep_backend"] == "sc"

    def property_holds(choose):
        import itertools
        for order in itertools.permutations(["sc", "sb", "v1"]):
            for n in range(4):
                for passed_ in itertools.combinations(order, n):
                    want = next((k for k in order if k in passed_), None)
                    if choose(list(order), list(passed_)) != want:
                        return False
        return True
    assert property_holds(G.sweep_backend)
    assert not property_holds(lambda order, passed_: order[0])          # NEGATIVE CONTROL: p3's rule
    src = (EFF / "r2_tick_sg5.py").read_text()
    for name in ("run_sweep", "main"):
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert "a.backends[0]" not in ast.get_source_segment(src, fn), name


def test_T21_sc_counters_are_a_report_only_snapshot(p3):
    """T21 (REPORT only, analysis/sg5_p4/RES_SC.md section 6): sc_counters reads the service's OWN counters after a
    run. S-C on the CPU stand-in after 6 draft-only ticks: mirror_uploads >= 1 and mirror_slots_uploaded > 0, equal to
    the service's attributes; row_uploads = one per tick; journal_counts holds the takes; devops = CpuDeviceSim's
    waited / gathered. Behind _LazyOps (the GPU path) it reads CudaDeviceOps' events_waited / submission_waits. v1 and
    sb: None. Negative controls: a planted change of the live counters shows in a new read (it reads the live values)
    and not in the earlier snapshot (it copies, journal_counts included). Static: one_point stores it as
    draft_only["sc"] and residency["sc"]; sc_counters makes no gate (no GB.gate / gates in its source) and one_point's
    gate names are unchanged."""
    G = _sg5_gate()
    eng = make_res_engine(make_ops(p3), kind="sc")
    eng.timer = TK.TickTimer(cuda=False)
    G.run_draft_only(eng, FIRST, 6, 3)
    svc = eng.res_world.svc
    c = G.sc_counters(svc)
    assert c["service"] == "ResidencySC" and c["mirror_uploads"] >= 1
    assert c["mirror_slots_uploaded"] == svc.mirror_slots_uploaded > 0 and c["mirror_uploads"] == svc.mirror_uploads
    assert c["row_uploads"] == svc.row_uploads == 6                     # ONE row copy per tick (_tick_rows)
    assert c["journal_counts"] == dict(svc.journal_counts) and c["journal_counts"].get("take", 0) > 0
    assert c["devops"]["kind"] == "CpuDeviceSim" and set(c["devops"]) >= {"waited", "gathered"}
    svc.mirror_slots_uploaded += 7                                       # NEGATIVE CONTROL: planted live changes
    svc.journal_counts["take"] = svc.journal_counts["take"] + 1
    c2 = G.sc_counters(svc)
    assert c2["mirror_slots_uploaded"] == c["mirror_slots_uploaded"] + 7
    assert c2["journal_counts"]["take"] == c["journal_counts"]["take"] + 1
    assert c["mirror_slots_uploaded"] == svc.mirror_slots_uploaded - 7   # the snapshot did not move
    lazy = G._LazyOps()
    lazy.impl = SimpleNamespace(events_waited=5, submission_waits=2)
    fake = SimpleNamespace(mirror_uploads=3, mirror_slots_uploaded=40, row_uploads=3, journal_counts={"touch": 9},
                           dev=lazy)
    assert G.sc_counters(fake)["devops"] == dict(events_waited=5, submission_waits=2, kind="SimpleNamespace")
    for kind in ("v1", "sb"):
        e2 = make_res_engine(make_ops(p3), kind=kind)
        assert G.sc_counters(e2.res_world.svc) is None, kind
    src = (EFF / "r2_tick_sg5.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sc_counters")
    seg = ast.get_source_segment(src, fn)
    assert "GB.gate" not in seg and "gates" not in seg
    op = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                           and n.name == "one_point"))
    assert 'info["draft_only"]["sc"] = sc_counters(svc)' in op and "sc=sc_counters(svc)" in op
    names = sorted(set(re.findall(r'GB\.gate\("%s_([A-Za-z0-9_]+)"', op)))      # [integ2] any case: SKV_ran too
    assert names == ["SKV_ran", "bookkeeping", "completed", "gilfree_team"], names  # [integ] SKV_ran: --attn shared
    assert not re.findall(r'GB\.gate\("%s_[^A-Za-z0-9_"]', op), "a gate name the pattern above would not read"
    # [integ3] review-integ2 m6: every GB.gate( in one_point uses that "%s_<name>" form (no label + "...", no "R_%s",
    # no other first argument); the helpers' gates (fused_gates, stack_acted_gates) come in through gates.extend
    assert op.count("GB.gate(") == len(re.findall(r'GB\.gate\("%s_[A-Za-z0-9_]+" % label', op)), \
        "a GB.gate in one_point that the name check does not read"
    assert sorted(set(re.findall(r"gates\.extend\((\w+)\(", op))) == ["fused_gates", "stack_acted_gates"]
