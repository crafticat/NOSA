"""CPU tests of the section 4.5 joint-HBM harness: benchmarks/Efficiency/joint_hbm_plan.py (accounting, layouts, write
plans, parsers, summary) and the CPU-only parts of benchmarks/Efficiency/joint_hbm.py (arguments, summarize). The
central test (J3) runs every form's window layout through a reference of the attention kernel's arithmetic (bias added
before the softmax scale, rows below cache_seqlens, cache_batch_idx, the bias addressed by the query row) and compares
it with a DIRECT statement of each row's reads (no window), with a version-mixup negative control.

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_joint_hbm.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
EFF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EFF))
import joint_hbm_plan as JP  # noqa: E402
import joint_hbm as JH  # noqa: E402

BS = JP.BS
SHIPPED_S1 = EFF.parents[1] / "dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py"


# ------------------------------------------------------------------ J1 / J2 accounting
def test_J1_accounting_hand_case():
    """J1: B 1, H 1, K 3, T 10, t0 5; selections v0 {0,1,2}, v1 {1,2,3}, v2 {2,3,4}, d {0,4,5}. sum: c 4 x (3 x 64 + 5),
    x 1 + 2 + 3, d 4; union_sep: verify {0..4} + tail 5, x 3; draft 3 x 64 + 5, d 4; union_request {0..5} + 5."""
    hist = torch.tensor([[[[0, 1, 2]], [[1, 2, 3]], [[2, 3, 4]], [[0, 4, 5]]]])        # (1, 4, 1, 3)
    a = JP.account_layer(hist, K=3, t0=5, T=10)
    assert a["sum"] == dict(c=4 * (3 * 64 + 5), x=6, d=4, hist_blocks=12)
    assert a["union_sep"] == dict(c=(5 * 64 + 5) + (3 * 64 + 5), x=3, d=4, hist_blocks=8)
    assert a["union_request"] == dict(c=6 * 64 + 5, x=3, d=4, hist_blocks=6)


def test_J2_accounting_identities_and_controls():
    """J2: on random selections sum >= union_sep >= union_request per version; identical selections in every row give
    a request union of n blocks (the K = 1 ratio-1 control of union_K16); t0 = 0 drops the c tail unit."""
    g = torch.Generator().manual_seed(2)
    hist = torch.stack([torch.stack([torch.stack([torch.randperm(40, generator=g)[:12] for _ in range(2)])
                                     for _ in range(4)]) for _ in range(3)])                  # (3, 4, 2, 12)
    a = JP.account_layer(hist, K=3, t0=7, T=40)
    for v in JP.VERSIONS:
        assert a["sum"][v] >= a["union_sep"][v] >= a["union_request"][v]
    same = hist[:, :1].expand(3, 4, 2, 12).contiguous()
    s = JP.account_layer(same, K=3, t0=7, T=40)
    assert s["union_request"]["hist_blocks"] == 3 * 2 * 12 and s["sum"]["hist_blocks"] == 4 * 3 * 2 * 12
    z = JP.account_layer(same, K=3, t0=0, T=40)
    assert z["union_request"]["c"] == 3 * 2 * 12 * 64


def test_J6_check_block():
    assert JP.check_block(16128, 20, 3) == (252, 20)
    with pytest.raises(ValueError, match="crosses a tail fill"):
        JP.check_block(16128, 61, 3)
    assert JP.check_block(16128 + 10, 50, 3) == (252, 60)                             # 10 + 50 + 4 = 64: allowed


# ------------------------------------------------------------------ J3 / J4 / J5 layouts against a direct reference
def attention_reference(q, kwin, vwin, bias, extent, entry, scale):
    """The kernel's arithmetic: row r (batch entry of q) reads window entry[r] rows [0, extent[r]); bias row r (by the
    QUERY row, bias.stride(0)); softmax(scale * (q k + bias)); q head j uses KV head j // (Hq / H)."""
    R, Hq, D = q.shape
    H = kwin.shape[2]
    G = Hq // H
    out = torch.empty_like(q)
    for r in range(R):
        e, n = int(entry[r]), int(extent[r])
        for j in range(Hq):
            h = j // G
            s = (kwin[e, :n, h] @ q[r, j] + bias[r, :n, h]) * scale
            out[r, j] = torch.softmax(s, 0) @ vwin[e, :n, h]
    return out


class World:
    """A random token store (B requests, T historical blocks + the tail block), per-step block maps (slots 0 .. 62 a
    permutation of the row's 63 ids, slot 63 = T), fresh x rows (the verifier rows' K / V / bias of this forward) and
    the draft's own row; the store's tail rows t0 .. t0 + K - 1 are the d version."""

    def __init__(self, B=2, K=3, T=80, t0=9, H=2, Hq=4, D=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.B, self.K, self.T, self.t0, self.H, self.Hq, self.D = B, K, T, t0, H, Hq, D
        n = (T + 1) * BS
        f = dict(dtype=torch.float64)
        self.k = torch.randn((B, n, H, D), generator=g, **f)
        self.v = torch.randn((B, n, H, D), generator=g, **f)
        self.c = torch.randn((B, n, H), generator=g, **f)
        self.k[:, T * BS + t0 + K + 1:] = 0
        self.v[:, T * BS + t0 + K + 1:] = 0
        bm = torch.empty((K + 1, H, B, JP.SLOTS), dtype=torch.int64)
        base = torch.randperm(T, generator=g)[:63]
        for o in range(K + 1):
            for h in range(H):
                for b in range(B):
                    ids = base.clone()
                    swap = torch.randperm(T, generator=g)[:8]                    # overlap but not identity
                    ids[torch.randperm(63, generator=g)[:8]] = swap
                    ids = torch.unique(ids)
                    extra = [x for x in torch.randperm(T, generator=g).tolist() if x not in set(ids.tolist())]
                    ids = torch.cat([ids, torch.tensor(extra[:63 - ids.numel()], dtype=torch.int64)])
                    bm[o, h, b, :63] = ids[torch.randperm(63, generator=g)]
                    bm[o, h, b, 63] = T
        self.bm = bm
        self.hist = bm[..., :63].permute(2, 0, 1, 3).contiguous()                  # (B, K + 1, H, 63)
        self.xk = torch.randn((B, K, H, D), generator=g, **f)
        self.xv = torch.randn((B, K, H, D), generator=g, **f)
        self.xc = torch.randn((B, K, H), generator=g, **f)
        self.dk = torch.randn((B, H, D), generator=g, **f)
        self.dv = torch.randn((B, H, D), generator=g, **f)
        self.dc = torch.randn((B, H), generator=g, **f)
        self.q = torch.randn((B, K + 1, Hq, D), generator=g, **f)
        self.scale = D ** -0.5

    def own(self, b, o):
        if o < self.K:
            return self.xk[b, o], self.xv[b, o], self.xc[b, o]
        return self.dk[b], self.dv[b], self.dc[b]

    def direct(self, b, o):
        """Row (b, o) from its read units: no window, no mask."""
        out = torch.empty((self.Hq, self.D), dtype=torch.float64)
        G = self.Hq // self.H
        T, t0, K = self.T, self.t0, self.K
        for j in range(self.Hq):
            h = j // G
            ks, vs, cs = [], [], []
            for blk in sorted(self.hist[b, o, h].tolist()):
                sl = slice(blk * BS, (blk + 1) * BS)
                ks.append(self.k[b, sl, h]); vs.append(self.v[b, sl, h]); cs.append(self.c[b, sl, h])
            sl = slice(T * BS, T * BS + t0)
            ks.append(self.k[b, sl, h]); vs.append(self.v[b, sl, h]); cs.append(self.c[b, sl, h])
            if o < K:
                for jj in range(o + 1):
                    ks.append(self.xk[b, jj, h:h + 1]); vs.append(self.xv[b, jj, h:h + 1]); cs.append(self.xc[b, jj, h:h + 1])
            else:
                sl = slice(T * BS + t0, T * BS + t0 + K)
                ks.append(self.k[b, sl, h]); vs.append(self.v[b, sl, h]); cs.append(self.c[b, sl, h])
                ks.append(self.dk[b, h:h + 1]); vs.append(self.dv[b, h:h + 1]); cs.append(self.dc[b, h:h + 1])
            kk, vv, cc = torch.cat(ks), torch.cat(vs), torch.cat(cs)
            out[j] = torch.softmax((kk @ self.q[b, o, j] + cc) * self.scale, 0) @ vv
        return out


def run_call_reference(w: World, rows, layout, bm=None, hist=None, tamper=None):
    """Build the call's windows with the plan functions, apply its write plan with the rows' own K / V / bias, run the
    reference kernel. Returns (out (R, Hq, D), plan)."""
    K, H = w.K, w.H
    R = len(rows)
    if layout == "private":
        E = w.B * (K + 1)
        req = [b for b in range(w.B) for _ in range(K + 1)]
        tok = JP.private_tok(w.bm if bm is None else bm)
        kw = JP.gather_window(w.k, tok, req)
        vw = JP.gather_window(w.v, tok, req)
        pc = JP.private_call(rows, K, w.t0)
        store = torch.zeros((E, pc["S_e"], H), dtype=torch.float64)
        rr = torch.tensor([JP.rid(b, o, K) for b, o in rows])
        store[:R] = JP.gather_window(w.c, tok[rr], [b for b, _ in rows])
        bias_view = store
        plan = pc
    else:
        sc = JP.shared_call(w.hist if hist is None else hist, rows, K, w.t0, w.T)
        kw = JP.gather_window(w.k, sc["tok"])
        vw = JP.gather_window(w.v, sc["tok"])
        vals = JP.gather_window(w.c, sc["tok"])
        rb = torch.tensor([b for b, _ in rows])
        vis = sc["visible"].clone()
        if tamper:
            tamper(sc, vis)
        store = torch.where(vis, vals[rb], torch.full_like(vals[rb], JP.MASK))
        bias_view = store[:w.B]
        plan = sc
    ok = torch.stack([w.own(b, o)[0] for b, o in rows])
    ov = torch.stack([w.own(b, o)[1] for b, o in rows])
    oc = torch.stack([w.own(b, o)[2] for b, o in rows])
    S_e = plan["S_e"]
    kw.view(-1, H, w.D).index_copy_(0, torch.tensor(plan["kv_dst"]), ok[torch.tensor(plan["kv_src"])])
    vw.view(-1, H, w.D).index_copy_(0, torch.tensor(plan["kv_dst"]), ov[torch.tensor(plan["kv_src"])])
    store.view(-1, H).index_copy_(0, torch.tensor(plan["b_dst"]), oc[torch.tensor(plan["b_src"])])
    q = torch.stack([w.q[b, o] for b, o in rows])
    full_bias = store                                         # the kernel reads row r of the STORAGE (stride(0))
    assert bias_view.shape[0] == kw.shape[0] and bias_view.stride(0) == S_e * H
    out = attention_reference(q, kw, vw, full_bias, plan["extent"], plan["entry"], w.scale)
    return out, plan


@pytest.mark.parametrize("form", JP.FORMS)
def test_J3_every_form_computes_each_rows_direct_attention(form):
    """J3: for every call of every form, the window layout + the write plan + the kernel's arithmetic equal the row's
    DIRECT attention over its read units (historical blocks, committed tail rows, x rows 0 .. o for a verifier row, d
    rows + its own row for the draft row) to 1e-10, at two seeds; so every form does the same work per row."""
    for seed in (0, 1):
        w = World(seed=seed)
        for name, kinds, layout in JP.FORM_CALLS[form]:
            rows = JP.call_rows(w.B, w.K, kinds)
            out, _ = run_call_reference(w, rows, layout)
            for p, (b, o) in enumerate(rows):
                assert torch.allclose(out[p], w.direct(b, o), atol=1e-10, rtol=0), (form, name, b, o)


def test_J3b_version_mixup_is_detected():
    """J3b (negative control of J3): in joint_shared, letting the draft row see the x-tail rows (the exact version of
    its uncommitted positions) changes its output; letting a verifier row see one d-tail row changes its output."""
    w = World(seed=3)
    rows = JP.call_rows(w.B, w.K, "vd")
    good, sc = run_call_reference(w, rows, "shared")

    def draft_sees_x(sc, vis):
        n = sc["n_hist"] * BS
        for p, (b, o) in enumerate(rows):
            if o == w.K:
                vis[p, n:n + w.t0 + w.K] = True

    bad, _ = run_call_reference(w, rows, "shared", tamper=draft_sees_x)
    dp = [p for p, (b, o) in enumerate(rows) if o == w.K]
    assert all(not torch.allclose(bad[p], good[p]) for p in dp)
    vp = [p for p, (b, o) in enumerate(rows) if o < w.K]
    assert all(torch.allclose(bad[p], good[p], atol=1e-12, rtol=0) for p in vp)


def test_J4_shared_layout_shape_rules():
    """J4: n_hist = the largest union over (b, h); S_e a multiple of 128 (the pad slot is invisible and past every
    extent); a verifier row's extent ends in the x-tail at t0 + o + 1, the draft row's in the d-tail at t0 + K + 1; the
    draft row sees nothing of the x-tail; padding slots (beyond a (b, h) union) are invisible to every row."""
    w = World(seed=4)
    for kinds in ("v", "d", "vd"):
        rows = JP.call_rows(w.B, w.K, kinds)
        sc = JP.shared_call(w.hist, rows, w.K, w.t0, w.T)
        U = torch.zeros((w.B, w.H), dtype=torch.int64)
        for b in range(w.B):
            for h in range(w.H):
                U[b, h] = len(set().union(*[set(w.hist[b, o, h].tolist()) for bb, o in rows if bb == b]))
        assert torch.equal(sc["n_union"].cpu(), U) and sc["n_hist"] == int(U.max())
        assert sc["S_e"] % 128 == 0 and sc["S_e"] >= sc["n_slots"] * BS
        nh = sc["n_hist"] * BS
        for p, (b, o) in enumerate(rows):
            vis = sc["visible"][p]
            assert not bool(vis[sc["extent"][p]:].any())
            if o < w.K:
                assert sc["extent"][p] == nh + w.t0 + o + 1
                assert bool(vis[nh:nh + w.t0 + o + 1].all())
            else:
                d0 = nh + (BS if "v" in kinds else 0)
                assert sc["extent"][p] == d0 + w.t0 + w.K + 1
                if "v" in kinds:
                    assert not bool(vis[nh:nh + BS].any())
                assert bool(vis[d0:d0 + w.t0 + w.K + 1].all())
            for h in range(w.H):
                for s in range(int(sc["n_union"][b, h]), sc["n_hist"]):
                    assert not bool(vis[s * BS:(s + 1) * BS, h].any())
                sel = set(w.hist[b, o, h].tolist())
                for s in range(int(sc["n_union"][b, h])):
                    blk = int(sc["order"][b, h, s])
                    assert bool(vis[s * BS, h]) == (blk in sel)
                    assert int(sc["tok"][b, s * BS + 5, h]) == blk * BS + 5


def test_J5_private_write_plan_and_extents():
    """J5: private: verifier row o's own K / V lands in the tail slot row t0 + o of the entries of rows o .. K - 1 of
    its request (and its bias in those rows' bias rows); the draft row writes only its own entry at t0 + K; extents are
    63 x 64 + t0 + o + 1; the entry is rid(b, o)."""
    K, t0 = 3, 9
    rows = JP.call_rows(2, K, "vd")
    pc = JP.private_call(rows, K, t0)
    S_e, tail = pc["S_e"], 63 * BS
    assert S_e == 64 * BS and pc["entry"] == [JP.rid(b, o, K) for b, o in rows]
    assert pc["extent"] == [tail + t0 + o + 1 for _, o in rows]
    pos = {r: p for p, r in enumerate(rows)}
    want = []
    for p, (b, o) in enumerate(rows):
        readers = [(b, K)] if o == K else [(b, oo) for oo in range(o, K)]
        want += [(p, JP.rid(*r, K) * S_e + tail + t0 + o, pos[r] * S_e + tail + t0 + o) for r in readers]
    assert list(zip(pc["kv_src"], pc["kv_dst"], pc["b_dst"])) == want
    vonly = JP.private_call(JP.call_rows(2, K, "v"), K, t0)
    assert all(d // S_e % (K + 1) < K for d in vonly["kv_dst"])


def test_J5b_layout_metrics():
    """J5b: private calls attend no masked token; shared calls attend >= useful and their window bound is the largest
    extent per entry."""
    w = World(seed=5)
    for layout, kinds in (("private", "vd"), ("shared", "vd"), ("shared", "v")):
        rows = JP.call_rows(w.B, w.K, kinds)
        plan = JP.private_call(rows, w.K, w.t0) if layout == "private" else JP.shared_call(w.hist, rows, w.K, w.t0, w.T)
        useful = JP.call_useful_tokens(w.hist, rows, w.K, w.t0, w.T)
        m = JP.layout_metrics(plan, useful, w.H)
        assert m["attended_tokens"] == w.H * sum(plan["extent"])
        if layout == "private":
            assert m["masked_tokens"] == 0
        else:
            assert m["masked_tokens"] > 0 and m["window_tokens"] < m["attended_tokens"]


# ------------------------------------------------------------------ J12 the stage-1 call
class RecordingExt:
    def __init__(self):
        self.calls = []

    def varlen_fwd_stage1(self, *args, **kw):
        self.calls.append((args, kw))
        return [torch.full((1,), float(len(self.calls)))]


class NoNvtx:
    def range_push(self, *_):
        pass

    def range_pop(self):
        pass


@pytest.mark.parametrize("U", [1, 3, 4])
def test_J12_stage1_direct_is_the_shipped_call(U):
    """J12: the SHIPPED function body of infllmv2_attn_stage1_fast (taken from the source file by AST, run with a
    recording stub extension and a no-op nvtx), called as the decode calls it (v = k, causal False) with U rows per
    request, and joint_hbm_plan.stage1_direct with the same inputs pass the SAME 21 positional arguments (tensors
    torch.equal, the rest ==) and no keyword; both return the op's first result."""
    fn = next(n for n in ast.parse(SHIPPED_S1.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "infllmv2_attn_stage1_fast")
    shipped, mine = RecordingExt(), RecordingExt()
    ns = dict(torch=torch, nvtx=NoNvtx(), infllm_cuda=shipped)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SHIPPED_S1), "exec"), ns)
    g = torch.Generator().manual_seed(12)
    B, HQ, HK, D, N = 3, 32, 2, 16, 7
    q = torch.randn((B * U, HQ, D), generator=g)
    k = torch.randn((B * N, HK, D), generator=g)
    cu_q = torch.arange(B + 1, dtype=torch.int32) * U
    cu_k = torch.arange(B + 1, dtype=torch.int32) * N
    r0 = ns["infllmv2_attn_stage1_fast"](q, k, k, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=U, max_seqlen_k=N,
                                         causal=False)
    r1 = JP.stage1_direct(mine, q, k, cu_q, cu_k, U, N)
    (a, ka), (b, kb) = shipped.calls[0], mine.calls[0]
    assert ka == kb == {} and len(a) == len(b) == 21
    for i, (x, y) in enumerate(zip(a, b)):
        if torch.is_tensor(x):
            assert torch.is_tensor(y) and torch.equal(x, y), i
        else:
            assert x == y and type(x) is type(y), (i, x, y)
    assert float(r0) == 1.0 and float(r1) == 1.0


# ------------------------------------------------------------------ J7 - J9 parsers
HEADER = ['"ID"', '"Process ID"', '"thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"',
          '"Kernel Name"', '"dram__bytes_read.sum"', '"dram__bytes_write.sum"', '"gpu__time_duration.sum"']


def csv_text(kernels):
    lines = [",".join(HEADER), ",".join(['""', '""', '""', '""', '"byte"', '"byte"', '"nsecond"'])]
    for i, (name, tag, rd, wr, ns) in enumerate(kernels):
        rng = "<default>:%s:::::::" % tag if tag else ""
        lines.append(",".join('"%s"' % x for x in (i, 7, rng, name, rd, wr, ns)))
    return "==PROF== Connected\n" + "\n".join(lines) + "\n"


def synthetic_kernels():
    ks = [("cuda::spin_kernel(long)", None, 10, 0, 5)]
    ks += [("ampere_bf16_s16816gemm", "jh|sep_private|draft|norm_qkv", 1000, 10, 50),
           ("flash_fwd_splitkv_kernel", "jh|sep_private|verify|attn", 500, 5, 40)]
    ks += [("cuda::spin_kernel(long)", None, 10, 0, 5)] * 2
    ks += [("ampere_bf16_s16816gemm", "jh|joint_private|joint|norm_qkv", 900, 9, 45),
           ("flash_fwd_splitkv_kernel", "jh|joint_private|joint|attn", 450, 4, 35),
           ("elementwise", None, 7, 1, 2)]
    return ks


def test_J7_ncu_windows_and_tags():
    """J7: the units row is skipped; marker runs of 1 and 2 open windows 1 and 2; the NVTX tag is read from the
    push/pop column; per part and total DRAM sums; an untagged kernel is counted."""
    rows = JP.ncu_rows(csv_text(synthetic_kernels()))
    assert len(rows) == 8 and rows[1]["tag"] == ("sep_private", "draft", "norm_qkv")
    wins = JP.ncu_windows(rows)
    assert [w_["markers"] for w_ in wins] == [1, 2]
    s1, s2 = (JP.window_summary(w_) for w_ in wins)
    assert s1["forms"] == ["sep_private"] and s1["calls"] == ["draft", "verify"] and s1["untagged"] == 0
    assert s1["total"]["dram__bytes_read.sum"] == 1500 and s1["parts"]["attn"]["kernels"] == 1
    assert s2["untagged"] == 1 and s2["total"]["kernels"] == 3 and s2["total"]["dram__bytes_read.sum"] == 1357
    renamed = csv_text([("cuda::spin_kernel(long)", None, 1, 0, 1), ("jh|joint_shared|joint|attn/flash_fwd", None, 3, 0, 1)])
    assert JP.ncu_rows(renamed)[1]["tag"] == ("joint_shared", "joint", "attn")


def test_J8_ncu_passes():
    log = ('==PROF== Profiling "spin_kernel" - 0: 0%....50%....100% - 1 pass\n'
           '==PROF== Profiling "flash_fwd" - 1: 0%....50%....100% - 3 passes\n')
    assert JP.ncu_passes(log) == [1, 3]


def test_J9_trace_busy():
    """J9: overlapping activities merge; idle = span - busy; the largest gap; CPU events ignored."""
    ev = [dict(ph="X", cat="kernel", ts=0, dur=10), dict(ph="X", cat="kernel", ts=5, dur=10),
          dict(ph="X", cat="gpu_memcpy", ts=30, dur=5), dict(ph="X", cat="cpu_op", ts=0, dur=100)]
    t = JP.trace_busy(dict(traceEvents=ev))
    assert t["n"] == 3 and t["busy_ms"] == pytest.approx(0.020) and t["span_ms"] == pytest.approx(0.035)
    assert t["idle_ms"] == pytest.approx(0.015) and t["max_gap_ms"] == pytest.approx(0.015)


# ------------------------------------------------------------------ J10 / J11 summarize and arguments (joint_hbm.py)
def synthetic_u(forms):
    an = {}
    for f in forms:
        an[f] = dict(calls=[c[0] for c in JP.FORM_CALLS[f]], rows=8, forwards=len(JP.FORM_CALLS[f]),
                     weight_bytes=16e9 * len(JP.FORM_CALLS[f]), ck_bytes=1e8, act_bytes=1e7,
                     kv_sum=dict(c=1000, x=6, d=4, hist_blocks=12), kv_form=dict(c=900, x=3, d=4, hist_blocks=10),
                     kv_request=dict(c=800, x=3, d=4, hist_blocks=8),
                     layout=dict(attended_tokens=2000, window_tokens=1500, useful_tokens=1900, masked_tokens=100))
    return dict(meta=dict(B=2, K=3), forms=forms, gates=[dict(name="G1_capture", passed=True, must=True, detail="ok")],
                timing={f: dict(median_ms=10.0 + i, p95_ms=11.0 + i, host_median_ms=3.0, per_call_median_ms=[5.0])
                        for i, f in enumerate(forms)},
                kineto={f: dict(n=100, busy_ms=8.0, idle_ms=2.0, span_ms=10.0, max_gap_ms=0.1) for f in forms},
                analytic=an)


def test_J10_summarize_end_to_end(tmp_path):
    """J10: joint_hbm.py --mode summarize over a synthetic run directory: every MUST gate passes -> exit 0, the
    measured windows join their forms, the comparisons carry the DRAM ratio; a missing window -> a failed MUST gate
    -> exit 1; the limits are in the summary."""
    forms = ["sep_private", "joint_private"]
    out = tmp_path / "run"
    (out / "U").mkdir(parents=True)
    (out / "P_none").mkdir()
    (out / "U" / "result.json").write_text(json.dumps(synthetic_u(forms)))
    (out / "P_none" / "windows.json").write_text(json.dumps(dict(windows=[
        dict(form="sep_private", group="private", markers=1, order=1, cache_control="none"),
        dict(form="joint_private", group="private", markers=2, order=2, cache_control="none")])))
    ks = [k for k in synthetic_kernels() if k[0] != "elementwise"]
    (out / "none_raw.csv").write_text(csv_text(ks))
    (out / "P_none.log").write_text('==PROF== Profiling "a" - 0: 0%....100% - 1 pass\n' * 3)
    rc = JH.main(["--mode", "summarize", "--out", str(out), "--variants", "none", "--forms", ",".join(forms)])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0, [g for g in s["gates"] if not g["passed"]]
    m = s["forms"]["joint_private"]["measured"]["none"]
    assert m["dram_read_bytes"] == 1350 and m["parts"]["attn"]["dram_read_bytes"] == 450 and m["window"] == 2
    c = [x for x in s["comparisons"] if x["base"] == "sep_private" and x["other"] == "joint_private"][0]
    assert c["dram_read_ratio_none"] == pytest.approx(1350 / 1500, abs=1e-4)
    assert c["attn_dram_read_ratio_none"] == pytest.approx(450 / 500, abs=1e-4) and s["limits"] == list(JP.LIMITS)
    assert "joint_private" in (out / "summary.md").read_text()
    (out / "none_raw.csv").write_text(csv_text(ks[:3]))
    assert JH.main(["--mode", "summarize", "--out", str(out), "--variants", "none", "--forms", ",".join(forms)]) == 1


class _FakeEvent:
    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


class _FakeProfile:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def export_chrome_trace(self, path):
        Path(path).write_text(json.dumps(dict(traceEvents=[])))


class _FakeOps:
    def sync(self):
        pass

    def event(self):
        return _FakeEvent()

    def profile(self):
        return _FakeProfile()

    def release(self):
        import gc
        gc.collect()

    def allocated(self):
        return 0


def _fake_engine(monkeypatch, refs, kept):
    """Stand-ins for the GPU functions run_group calls; every plan tensor is registered by a weakref."""
    import weakref

    def build(S, group, forms):
        plans = {}
        for f in forms:
            calls = []
            for name, kinds, layout in JP.FORM_CALLS[f]:
                t = torch.zeros(4, 3)
                refs.append(weakref.ref(t))
                calls.append(dict(name=name, layout=layout, metrics=dict(masked_tokens=0), layers=[dict(kwin=t)], R=2))
            plans[f] = calls
        if S.get("keep"):
            kept.append(plans)                           # what the old inline loop's variable did
        return plans, []

    def run_form(S, plans, f, events=None):
        outs = []
        for i, cp in enumerate(plans[f]):
            outs.append(torch.zeros(2, 5))
            if events is not None:
                events[i].record()
        return outs

    monkeypatch.setattr(JH, "build_group", build)
    monkeypatch.setattr(JH, "run_form", run_form)
    monkeypatch.setattr(JH, "run_call", lambda S, cp, unmasked=False: torch.ones(2, 5))
    monkeypatch.setattr(JH, "logits_gate", lambda S, plans, f, outs: dict(max_abs=0.0, argmax_agree=1.0))
    monkeypatch.setattr(JH, "selection_agreement", lambda S, plans, f: {})
    monkeypatch.setattr(JH, "analytic", lambda S, plans, f, acc: {})


def _R():
    return dict(timing={}, kineto={}, analytic={}, logits={}, selection={}, issues2=[], bad_acc=[], mask_ctl=None,
                ref_out={}, order_log=[], release=[])


def test_J13_run_group_releases_its_plans(tmp_path, monkeypatch):
    """J13: run_group (the per-group body of mode_cell) keeps NO reference to its group's plans: after it returns, the
    weakref of every plan tensor is dead, so ops.release() frees the windows before the next group (job b64 ran out of
    memory in the shared group: a loop variable of the former inline loop kept a private plan, 52.8 GB, alive). It
    still fills the results (timing, kernel trace, the mask control, CPU logits). Negative control: a reference kept
    outside (the old loop variable) keeps the tensors alive and the check sees it."""
    import gc
    import types
    a = types.SimpleNamespace(warm=1, reps=2, out=str(tmp_path))
    refs, kept = [], []
    _fake_engine(monkeypatch, refs, kept)
    R = _R()
    JH.run_group({}, a, "shared", ["sep_shared", "joint_shared"], torch.zeros(4, dtype=torch.uint8), {}, R, _FakeOps())
    gc.collect()
    assert len(refs) == 3 and all(r() is None for r in refs)
    assert set(R["timing"]) == {"sep_shared", "joint_shared"} and R["timing"]["sep_shared"]["per_call_median_ms"] == [1.0, 1.0]
    assert R["kineto"]["joint_shared"]["n"] == 0 and R["mask_ctl"]["differs"] is True
    assert R["ref_out"]["joint_shared"][0].device.type == "cpu" and len(R["order_log"]) == 4
    refs.clear()
    JH.run_group({"keep": True}, a, "private", ["sep_private", "joint_private"], torch.zeros(4, dtype=torch.uint8), {},
                 _R(), _FakeOps())
    gc.collect()
    assert refs and all(r() is not None for r in refs)
    kept.clear()
    gc.collect()
    assert all(r() is None for r in refs)


def test_J14_mode_cell_builds_groups_only_through_run_group():
    """J14 (AST of joint_hbm.py): mode_cell and mode_ncu_cell call build_group only through run_group / ncu_group,
    and release after each group (ops.release() right after the call inside the loop over JP.GROUPS)."""
    tree = ast.parse((EFF / "joint_hbm.py").read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for mode, helper in (("mode_cell", "run_group"), ("mode_ncu_cell", "ncu_group")):
        src = ast.unparse(fns[mode])
        assert "build_group(" not in src, mode
        loop = next(n for n in ast.walk(fns[mode]) if isinstance(n, ast.For) and "JP.GROUPS" in ast.unparse(n.iter))
        calls = [ast.unparse(s) for s in loop.body if isinstance(s, ast.Expr)]
        i = next(k for k, c in enumerate(calls) if c.startswith(helper + "("))
        assert calls[i + 1] == "ops.release()", (mode, calls)
    for helper in ("run_group", "ncu_group"):
        assert "build_group(" in ast.unparse(fns[helper])


def test_J11_arguments_and_child_argv(tmp_path):
    """J11: unknown forms and variants are refused; the child command carries every knob of the parent; orchestrate
    refuses an existing --out (exit 2) and missing model / workload (exit 2)."""
    with pytest.raises(SystemExit):
        JH.parse_args(["--mode", "cell", "--out", "x", "--forms", "sep_private,nope"])
    with pytest.raises(SystemExit):
        JH.parse_args(["--mode", "cell", "--out", "x", "--variants", "warm"])
    a = JH.parse_args(["--mode", "orchestrate", "--out", str(tmp_path), "--model", "m", "--workload", "w", "--batch", "16",
                       "--W", "21", "--reps", "4", "--forms", "joint_shared"])
    argv = JH.child_argv(a, "cell", "/o/U")
    for flag, val in (("--batch", "16"), ("--W", "21"), ("--reps", "4"), ("--forms", "joint_shared"), ("--mode", "cell")):
        assert argv[argv.index(flag) + 1] == val
    assert JH.main(["--mode", "orchestrate", "--out", str(tmp_path), "--model", "m", "--workload", "w"]) == 2
    assert JH.main(["--mode", "cell", "--out", str(tmp_path / "new"), "--model", "", "--workload", ""]) == 2
