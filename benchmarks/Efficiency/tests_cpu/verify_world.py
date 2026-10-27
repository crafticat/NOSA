"""The CPU WORLD of tests_cpu/test_verify_rows.py (Task 1.4 route 1; not a test file). It extends ragged_world.py
(Task 1.3: the SHIPPED engine from its own source, every CUDA op a torch stand-in with the kernel's flat addresses,
lesson L28) with:
  - nosi/nosi/verify_rows.py loaded through the same synthetic package, with the same stand-ins (torch.topk = CUDA's
    sorted=False order, the per-request pooling, the stage-1 op);
  - fake_attention_idx: flash_attn_nosa_with_kvcache WITH cache_batch_idx and return_softmax_lse. It models the two
    facts the FAST route relies on: the API checks the bias against the CACHE batch (flash_api.cpp:366), and the
    kernel reads the bias with the QUERY entry index at FLAT addresses (flash_fwd_kernel.h:968-970), the K / V with
    cache_batch_idx (:649-659). Its per-entry arithmetic is ragged_world.fake_attention's, op for op (a test checks
    the bits);
  - the drivers: run_joint_vs_shipped (attach after the shipped prefill, the shipped decode as the reference),
    run_joint_vs_ragged (build, ragged lengths, the ragged serial step as the reference), run_fast_vs_exact (two
    joint worlds), run_flag_off (a VerifyState attached and never used);
  - check_union (every row's selected block present, with its host rows) and looped_select_rows (the per-position
    selection loop of the pilot: a negative control that lives here, not in the engine).
Load it by file path, as test_ragged_engine.py loads ragged_world.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _ragged_world():
    if "ragged_world" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ragged_world", str(HERE / "ragged_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ragged_world"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ragged_world"]


W = _ragged_world()
RE = W.RE
VR = W.pkg_load("verify_rows")
VR.torch = W.TP                                       # torch.topk -> CUDA's sorted=False order
VR.nosa_pooling_per_request = W.fake_pool_per_request
STAGE1 = VR.stage1_seqused                            # the real wrapper (it calls the stand-in op through ext)
POOLING = VR.nosa_pooling_per_request
SELECT_ROWS = VR._select_rows
VR_PATH = W.N / "verify_rows.py"


# ---------------------------------------------------------------------------
# flash_attn_nosa_with_kvcache with cache_batch_idx and the log-sum-exp
# ---------------------------------------------------------------------------
def _entry(qrow, kk, vv, bias_rows, n, per, Hq, Dq):
    """ragged_world.fake_attention's loop body for ONE entry, op for op: qrow (Hq, Dq), kk / vv (rows, Hk, Dq),
    bias_rows (rows, Hk); n attended rows; per n-blocks per split. Returns (out (Hq, Dq) float32, lse (Hq,))."""
    Hk = kk.shape[1]
    qq = qrow.reshape(Hk, Hq // Hk, Dq).float()
    parts = []
    lo = 0
    while lo < n:
        hi = min(lo + per * W.KBLOCK_N, n)
        sc = torch.einsum("hgd,lhd->hgl", qq, kk[lo:hi].float()) * Dq ** -0.5 + \
            bias_rows[lo:hi].float().t().unsqueeze(1)
        mx = sc.amax(-1, keepdim=True)
        e = torch.exp(sc - mx)
        parts.append((mx, e.sum(-1, keepdim=True), torch.einsum("hgl,lhd->hgd", e, vv[lo:hi].float())))
        lo = hi
    mx = torch.stack([p_[0] for p_ in parts]).amax(0)
    num = sum(torch.exp(p_[0] - mx) * p_[2] for p_ in parts)
    den = sum(torch.exp(p_[0] - mx) * p_[1] for p_ in parts)
    return (num / den).reshape(Hq, Dq), (mx + torch.log(den)).reshape(Hq)


def fake_attention_idx(q, k_cache, v_cache, kv_bias, cache_seqlens=None, cache_batch_idx=None, num_splits=0,
                       return_softmax_lse=False):
    """flash_attn_nosa_with_kvcache (module docstring). Without cache_batch_idx and the log-sum-exp it IS
    ragged_world.fake_attention (the shipped path's stand-in)."""
    if cache_batch_idx is None and not return_softmax_lse:
        return W.fake_attention(q, k_cache, v_cache, kv_bias, cache_seqlens, num_splits)
    Bq, sq, Hq, Dq = q.shape
    Bc, rows, Hk = k_cache.shape[0], k_cache.shape[1], k_cache.shape[2]
    assert sq == 1, "decode rows only (seqlen_q 1)"
    assert tuple(kv_bias.shape) == (Bc, rows, Hk), "flash_api.cpp:366: bcache must be (batch_size_c, seqlen_k, H_k)"
    assert tuple(cache_seqlens.shape) == (Bq,) and cache_seqlens.dtype == torch.int32
    if cache_batch_idx is not None:
        assert tuple(cache_batch_idx.shape) == (Bq,) and cache_batch_idx.dtype == torch.int32
        assert cache_batch_idx.is_contiguous()
        cbi = [int(x) for x in cache_batch_idx]
        assert all(0 <= x < Bc for x in cbi), "cache_batch_idx past the cache batch"
    else:
        cbi = list(range(Bq))
    _, per = W.split_partition(rows, num_splits)
    flat, off0, st = W._flat(kv_bias)
    r_ = torch.arange(rows).view(rows, 1)
    h_ = torch.arange(Hk).view(1, Hk)
    out = torch.empty_like(q)
    lse = torch.empty((Bq, Hq, 1), dtype=torch.float32)
    for e in range(Bq):
        idx = off0 + e * st[0] + r_ * st[1] + h_ * st[2]                 # the bias row of ENTRY e (bidb)
        assert int(idx.max()) < flat.numel(), "the bias read of entry %d is past the bias storage" % e
        b_e = flat[idx]
        o, l_ = _entry(q[e, 0], k_cache[cbi[e]], v_cache[cbi[e]], b_e, int(cache_seqlens[e]), per, Hq, Dq)
        out[e, 0] = o.to(q.dtype)
        lse[e, :, 0] = l_
    return (out, lse) if return_softmax_lse else out


def looped_select_rows(vs, layer, q, twin):
    """NEGATIVE CONTROL (the pilot's per-position selection, P0_PILOT_AUTOPSY.md V1): ONE stage-1 / pooling / top-k
    pass PER ROW over the B requests, the results copied into the batched buffer. The values are the batched ones
    (every op is per entry); only the call count changes."""
    rs = vs.rs
    st = rs.st
    B, R, H = vs.B, vs.R, vs.H
    sel = vs.ti.view(H, B, R, layer.topk_blocks)
    q4 = q.view(B, R, q.shape[1], q.shape[2])
    ck = twin.comp_k_flat()
    for r in range(R):
        rs.n_used.copy_(vs.n_used[:, r])
        rs.M_b.copy_(vs.M_b[:, r])
        score = VR.stage1_seqused(q4[:, r].contiguous(), ck, ck, st.cu_q, twin.cu, 1, twin.lk_max,
                                  seqused_k=rs.n_used, ext=rs.ops.kernels.get("stage1_ext"))
        VR.nosa_pooling_per_request(score, twin.ccis, rs.n_used, rs.M_b, st.pq, st.pc, stride=layer.pooling_stride,
                                    block_size=layer.block_size, local_blocks=layer.window_blocks,
                                    init_blocks=layer.init_blocks)
        W.TP.topk(st.pq, rs.nl.QK_SELECT, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
        st.mask.zero_()
        st.mask.scatter_(2, st.ti_q, True)
        st.pc.masked_fill_(st.mask, float("inf"))
        W.TP.topk(st.pc, layer.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))
        sel[:, :, r].copy_(st.ti)
    return sel


# ---------------------------------------------------------------------------
# preconditions and checks
# ---------------------------------------------------------------------------
TWIN_TABLES = ("pos", "n_comp", "n_ccis", "comp_k", "ring", "total_cis", "ccis")


def engines_equal(ca, cb, twins=None):
    """Bitwise: every engine table (ragged_world.ENGINE_TABLES) of two caches, layer by layer, and the LRU stamp;
    twins = (twins_a, twins_b): the twin tables too. Returns the list of differences (empty: equal). The per-step
    scratch (_new_block_map_buf, _load_mask) is not compared: diff_offload writes every entry before a read."""
    bad = []
    for l in range(W.L):
        ea, eb = ca.layers[l].cache_engine, cb.layers[l].cache_engine
        for name in W.ENGINE_TABLES:
            if not torch.equal(W.bits(getattr(ea, name)), W.bits(getattr(eb, name))):
                bad.append((l, name))
        if ea._pool_stamp_base != eb._pool_stamp_base:
            bad.append((l, "_pool_stamp_base"))
        if twins is not None:
            for name in TWIN_TABLES:
                if not torch.equal(W.bits(getattr(twins[0][l], name)), W.bits(getattr(twins[1][l], name))):
                    bad.append((l, "twin." + name))
    return bad


def _compare_rows(grp, la_rows, lb, compare, tol, ev, skip=None):
    """la_rows: R reference logits (B, V); lb: the joint logits (B, R, V). compare "bits" raises at the first unequal
    bit; "close": |a - b| <= tol * (1 + |a|) and the same argmax, rows in `skip` (r, b) excepted; None: no check."""
    if compare is None:
        return
    for r, la in enumerate(la_rows):
        got = lb[:, r]
        assert bool(torch.isfinite(la).all()), "group %d row %d: the reference logits are not finite" % (grp, r)
        if compare == "bits":
            assert torch.equal(W.bits(la), W.bits(got)), \
                "group %d row %d: the joint logits differ from the serial step" % (grp, r)
            continue
        for b in range(la.shape[0]):
            if skip is not None and (r, b) in skip:
                continue
            d = float(((la[b] - got[b]).abs() / (1 + la[b].abs())).max())
            ev["max_rel"] = max(ev["max_rel"], d)
            assert d <= tol, "group %d row %d request %d: FAST logits off by %.3g (tol %.1g)" % (grp, r, b, d, tol)
            assert int(la[b].argmax()) == int(got[b].argmax()), \
                "group %d row %d request %d: FAST argmax differs" % (grp, r, b)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def run_joint_vs_shipped(B, S, groups, K=3, seed=0, mode="exact", gemm="rowgroup", neg=None, record=False,
                         compare="bits", compare_state=None, extra=16, linear=None, tol=1e-4):
    """THE SHIPPED DECODE (Llama.decode_inference, serial, re-warmed as r2_base_cell.run_shipped) against the JOINT
    PASS on a second cache from the same synthetic prefill (ragged_engine.attach, then verify_rows.attach),
    teacher-forced on the shipped tokens: per group of R positions, R shipped steps, then ONE joint pass; row r against
    the shipped step g * R + r. Precondition (asserted): the two caches are equal before step 0. compare_state (default:
    EXACT mode): every engine table and the twin against the shipped tables at the end. Returns the evidence."""
    W.STATS.clear()
    R = K + 1
    steps = groups * R
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, [S] * B)
    caches = []
    for _ in range(2):
        c = W.new_cache(steps + 64)
        W.shipped_prefill(c, rows, S, B)
        W.poison_shipped(c, S)
        caches.append(c)
    A, Bc = caches
    pre = engines_equal(A, Bc)
    assert pre == [], "precondition: the two caches differ before step 0: %s" % pre[:5]
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=False)
    rs = RE.attach(model, Bc, S, steps, ops)
    vops = VR.VerifyOps(K=K, mode=mode, gemm=gemm, linear=linear, extra=extra, attn=fake_attention_idx, neg=neg,
                        record=record)
    vs = VR.attach(model, rs, vops)
    g = torch.Generator().manual_seed(seed + 2)
    tok = torch.randint(0, W.V, (B, 1), generator=g)
    pos = torch.full((B, 1), S, dtype=torch.int32)
    warm = dict(state=None, rewarms=[])
    ev = dict(max_rel=0.0, fills=[], groups=groups, R=R)
    for grp in range(groups):
        ins, poss, las = [], [], []
        for r in range(R):
            if (S + grp * R + r) % W.BS == W.BS - 1:
                ev["fills"].append((grp, r))
            W.TAG[0] = "shipped"
            la = W.shipped_step(model, A, tok, pos, warm)
            ins.append(tok)
            poss.append(pos)
            las.append(la[:, -1])
            tok = la[:, -1].argmax(-1, keepdim=True)
            pos = pos + 1
        W.TAG[0] = "joint"
        lb = VR.joint_pass(model, vs, torch.cat(ins, 1), torch.cat(poss, 1))
        _compare_rows(grp, las, lb, compare, tol, ev)
    ev.update(rewarms=list(warm["rewarms"]), stats={k: dict(v) for k, v in W.STATS.items()}, vs=vs, rs=rs,
              verify=VR.read_stats(vs), errors=RE.errors(rs))
    if compare_state is None:
        compare_state = mode == "exact"
    if compare_state:
        for l in range(W.L):
            ea, eb = A.layers[l].cache_engine, Bc.layers[l].cache_engine
            for name in W.ENGINE_TABLES:
                assert torch.equal(W.bits(getattr(ea, name)), W.bits(getattr(eb, name))), \
                    "layer %d: %s differs from the shipped engine" % (l, name)
            assert ea._pool_stamp_base == eb._pool_stamp_base, "layer %d: _pool_stamp_base differs" % l
            la_, tw = A.layers[l], rs.st.twins[l]
            n = int(la_.cached_compressed_max_seqlen)
            assert torch.equal(tw.n_comp, torch.full((B,), n)), "layer %d: n_comp vs the shipped length" % l
            assert torch.equal(W.bits(tw.comp_k[:, :n]), W.bits(la_.compress_k_cache_varlen)), "layer %d: comp_k" % l
            p_end = S + steps
            assert torch.equal(W.bits(tw.total_cis[:, :p_end]), W.bits(la_.total_cis[:, :p_end])), \
                "layer %d: total_cis" % l
    return ev


def exact_ragged_ops():
    """EXACT mode of the ragged engine on the CPU (ragged_world.ragged_ops): row-wise GEMM, fixed-order twin means."""
    return RE.RaggedOps(W.KERNELS, gemm="exact", linear=W.row_linear, twin="bi", body="torch", graph=True)


def _two_built(S_list, steps, seed):
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, S_list)
    worlds = []
    for _ in range(2):
        s = RE.build(model, W.new_cache(0), S_list, steps, exact_ragged_ops(), lambda b, l: rows[(b, l)], pin=False)
        W.poison_built(s, S_list)
        worlds.append(s)
    a, b = worlds
    pre = engines_equal(a.cache, b.cache, (a.st.twins, b.st.twins))
    assert pre == [], "precondition: the two built worlds differ before step 0: %s" % pre[:5]
    return model, a, b


def run_joint_vs_ragged(S_list, groups, K=3, seed=0, mode="exact", gemm="exact", neg=None, record=False, extra=16,
                        compare="bits", tol=1e-4, linear=None):
    """RAGGED LENGTHS: the ragged engine's SERIAL step (EXACT ops) against the joint pass on a second build of the
    same prompts, teacher-forced on one random token stream; row r against serial step g * R + r of the same request.
    Precondition (asserted): the two worlds are equal before step 0. EXACT mode with compare "bits": then every engine
    table and twin. linear: the GEMM callable of the joint pass (default row_linear for gemm "exact", else F.linear)."""
    W.STATS.clear()
    R = K + 1
    steps = groups * R
    B = len(S_list)
    model, A, Bw = _two_built(S_list, steps, seed)
    lin = linear if linear is not None else (W.row_linear if gemm == "exact" else None)
    vops = VR.VerifyOps(K=K, mode=mode, gemm=gemm, linear=lin, extra=extra, attn=fake_attention_idx, neg=neg,
                        record=record)
    vs = VR.attach(model, Bw, vops)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    ev = dict(max_rel=0.0, fills=[], groups=groups, R=R)
    for grp in range(groups):
        las = []
        for r in range(R):
            t = grp * R + r
            p = S_t + t
            for b in range(B):
                if int(p[b]) % W.BS == W.BS - 1:
                    ev["fills"].append((grp, r, b))
            W.TAG[0] = "serial"
            la = RE.decode_step(model, A, tokens[:, t:t + 1], p.view(B, 1).to(torch.int32))
            las.append(la[:, -1])
        W.TAG[0] = "joint"
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        lb = VR.joint_pass(model, vs, tokens[:, grp * R:(grp + 1) * R], pos)
        _compare_rows(grp, las, lb, compare, tol, ev)
    ev.update(stats={k: dict(v) for k, v in W.STATS.items()}, vs=vs, verify=VR.read_stats(vs), errors=RE.errors(Bw))
    if mode == "exact" and compare == "bits":
        bad = engines_equal(A.cache, Bw.cache, (A.st.twins, Bw.st.twins))
        assert bad == [], "the joint world's tables differ from the serial world's: %s" % bad[:5]
    return ev


def run_fast_vs_exact(S_list, groups, K=3, seed=0, neg=None, extra=16, tol=1e-5, gemm="exact", compare=True,
                      max_flips=2):
    """FAST against EXACT, two joint worlds of the same build, the same tokens (EXACT is bitwise the serial step:
    test_verify_equals_serial_shipped / _ragged). Both record their selections. Layer 0 gets the same inputs in both
    worlds, so its selections must be BITWISE equal (asserted). A later layer's q carries FAST's attention rounding, so
    a near-tie may move one block: such a row is COUNTED (ev["flips"]) and skipped; more than max_flips moves raise
    (a real difference moves many). Every other row must be close (tol) with the same argmax. gemm "exact" isolates
    the attention (the GEMMs give the same bits in both worlds). compare False: run and record only."""
    W.STATS.clear()
    R = K + 1
    steps = groups * R
    B = len(S_list)
    model, X, Fw = _two_built(S_list, steps, seed)
    lin = W.row_linear if gemm == "exact" else None
    vx = VR.attach(model, X, VR.VerifyOps(K=K, mode="exact", gemm="exact", linear=W.row_linear, extra=extra,
                                          attn=fake_attention_idx, record=True))
    vf = VR.attach(model, Fw, VR.VerifyOps(K=K, mode="fast", gemm=gemm, linear=lin, extra=extra,
                                           attn=fake_attention_idx, neg=neg, record=True))
    vf.fa_check = {}                                  # the cell's VRF-IDX gate: first pass, layer 0
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    ev = dict(max_rel=0.0, flips=[], groups=groups, R=R)
    for grp in range(groups):
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        ids = tokens[:, grp * R:(grp + 1) * R]
        nx, nf = len(vx.log), len(vf.log)
        W.TAG[0] = "exact"
        lx = VR.joint_pass(model, vx, ids, pos)
        W.TAG[0] = "fast"
        lf = VR.joint_pass(model, vf, ids, pos)
        sel_x = {}
        for rec in vx.log[nx:]:
            sel_x.setdefault(rec["layer"], {})[rec["row"]] = rec["sel"]
        skip = set()
        for rec in vf.log[nf:]:
            l = rec["layer"]
            for r in range(R):
                same = rec["sel"][:, :, r] == sel_x[l][r]
                for b in range(B):
                    if not bool(same[:, b].all()):
                        if compare:
                            assert l > 0, "group %d layer 0 row %d request %d: FAST selection differs" % (grp, r, b)
                        skip.add((r, b))
                        ev["flips"].append((grp, l, r, b))
        if not compare:
            continue
        assert len(ev["flips"]) <= max_flips, "FAST: %d selection moves in later layers (more than %d near-ties): %s" % (
            len(ev["flips"]), max_flips, ev["flips"][:8])
        _compare_rows(grp, [lx[:, r] for r in range(R)], lf, "close", tol, ev, skip=skip)
    ev.update(vx=vx, vf=vf, fast=VR.read_stats(vf), exact=VR.read_stats(vx), errors=(RE.errors(X), RE.errors(Fw)),
              stats={k: dict(v) for k, v in W.STATS.items()})
    return ev


def run_flag_off(B, S, steps, seed=0, plant=None):
    """FLAG OFF: the shipped step against the ragged engine (ragged_world.run_equal's comparison) with a VerifyState
    (FAST and EXACT) ATTACHED and never used. Asserts: the attach changes no engine or twin table (bitwise), and every
    ragged step is bitwise the shipped step. plant "touch" (NEGATIVE CONTROL): the attach also adds 1 to the first
    tail row of request 0, layer 0 (a written, finite row when S % 64 > 0). Returns the evidence."""
    W.STATS.clear()
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, [S] * B)
    caches = []
    for _ in range(2):
        c = W.new_cache(steps + 64)
        W.shipped_prefill(c, rows, S, B)
        W.poison_shipped(c, S)
        caches.append(c)
    A, Bc = caches
    ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
    rs = RE.attach(model, Bc, S, steps, ops)
    snap = [{name: getattr(lay.cache_engine, name).clone() for name in W.ENGINE_TABLES} for lay in Bc.layers]
    tsnap = [{name: getattr(t, name).clone() for name in TWIN_TABLES} for t in rs.st.twins]
    states = [VR.attach(model, rs, VR.VerifyOps(K=3, mode=m, attn=fake_attention_idx)) for m in ("fast", "exact")]
    if plant == "touch":
        assert S % W.BS > 0 and bool(torch.isfinite(Bc.layers[0].cache_engine._k_gpu[0, W.TAIL0]).all())
        Bc.layers[0].cache_engine._k_gpu[0, W.TAIL0] += 1.0
    for l, lay in enumerate(Bc.layers):
        for name in W.ENGINE_TABLES:
            assert torch.equal(W.bits(snap[l][name]), W.bits(getattr(lay.cache_engine, name))), \
                "layer %d: the verify attach changed %s" % (l, name)
        for name in TWIN_TABLES:
            assert torch.equal(W.bits(tsnap[l][name]), W.bits(getattr(rs.st.twins[l], name))), \
                "layer %d: the verify attach changed twin.%s" % (l, name)
    g = torch.Generator().manual_seed(seed + 2)
    tok = torch.randint(0, W.V, (B, 1), generator=g)
    pos = torch.full((B, 1), S, dtype=torch.int32)
    warm = dict(state=None, rewarms=[])
    for i in range(steps):
        W.TAG[0] = "shipped"
        la = W.shipped_step(model, A, tok, pos, warm)
        W.TAG[0] = "ragged"
        lb = RE.decode_step(model, rs, tok, pos)
        assert torch.equal(W.bits(la), W.bits(lb)), "step %d: the ragged logits differ from the shipped step" % i
        tok = la[:, -1].argmax(-1, keepdim=True)
        pos = pos + 1
    return dict(states=states, rewarms=list(warm["rewarms"]), stats={k: dict(v) for k, v in W.STATS.items()})


# ---------------------------------------------------------------------------
# block 0, the union check (logs of record=True runs)
# ---------------------------------------------------------------------------
def rows_without_block0(log):
    """The (kind, layer, row, head, request) of every recorded row selection that does NOT list block 0, and the
    number of row selections checked."""
    bad, n = [], 0
    for rec in log:
        sel = rec["sel"]
        if rec["kind"] == "exact":                                  # (H, B, topk): row rec["row"]
            sel = sel.unsqueeze(2)
            rows = [rec["row"]]
        else:                                                       # (H, B, R, topk)
            rows = list(range(sel.shape[2]))
        has0 = (sel == 0).any(-1)                                   # (H, B, R')
        n += has0.numel()
        for h, b, i in (~has0).nonzero().tolist():
            bad.append((rec["kind"], rec["layer"], rows[i], h, b))
    return bad, n


def exact_windows_hold_rows(log):
    """EXACT records: when row r attends, the window's slots 0 .. topk - 2 plus row r's own tail block p_r // 64 equal
    row r's selection as a set (decode_fixed names the NEXT block in the tail slot after a fill row, findings F4).
    Returns the (layer, row, head, request) where it does not, and the count checked."""
    bad, n = [], 0
    for rec in log:
        if rec["kind"] != "exact":
            continue
        w = rec["win"].clone()
        w[..., -1] = (rec["pos"] // W.BS).view(1, -1)
        a = w.sort(-1).values
        b_ = rec["sel"].sort(-1).values
        eq = (a == b_).all(-1)                                      # (H, B)
        n += eq.numel()
        for h, b in (~eq).nonzero().tolist():
            bad.append((rec["layer"], rec["row"], h, b))
    return bad, n


# the union check (FAST logs)
def check_union(log, X, bs=None):
    """For every FAST record (one layer of one pass) and every (head h, request b, row r) and every block id k of row
    r's selection: k must be present for row r. Where: the tail block m_b in slot topk - 1; the next block m_b + 1 in
    extras slot X - 1 (rows past the edge); a window slot s < topk - 1 with map id k, whose K / V rows equal the host
    rows of block k and whose bias rows equal total_cis; or an extras slot x < X - 1 with id k, the same contents.
    Returns dict(checked, missing (distinct (record, h, b, k) not present), contents_bad, missing_list)."""
    bs = bs or W.BS
    checked, missing, contents_bad, miss = 0, set(), [], []
    for ri, rec in enumerate(x for x in log if x["kind"] == "fast"):
        sel, win, ids = rec["sel"], rec["win"], rec["ids"]
        H, B, R, T = sel.shape
        tail = win.shape[2] - 1
        for h in range(H):
            for b in range(B):
                m = int(rec["m"][b])
                wl = win[h, b].tolist()
                xl = ids[h, b].tolist()
                for r in range(R):
                    in_tail = bool(rec["in_tail"][b, r])
                    for k in sel[h, b, r].tolist():
                        checked += 1
                        if k == m:
                            if wl[tail] != m:
                                missing.add((ri, h, b, k))
                            continue
                        if k == m + 1 and not in_tail:
                            if xl[X - 1] != m + 1:
                                missing.add((ri, h, b, k))
                            continue
                        rows = slice(k * bs, (k + 1) * bs)
                        if k in wl[:tail]:
                            s = wl.index(k)
                            src = (rec["k_gpu"], rec["v_gpu"], rec["bias_gpu"])
                        elif k in xl[:X - 1]:
                            s = xl.index(k)
                            src = (rec["ext_k"], rec["ext_v"], rec["ext_bias"])
                        else:
                            missing.add((ri, h, b, k))
                            miss.append((ri, rec["layer"], h, b, r, k))
                            continue
                        dst = slice(s * bs, (s + 1) * bs)
                        ok = (torch.equal(W.bits(src[0][b, dst, h]), W.bits(rec["k_cpu"][b, rows, h]))
                              and torch.equal(W.bits(src[1][b, dst, h]), W.bits(rec["v_cpu"][b, rows, h]))
                              and torch.equal(W.bits(src[2][b, dst, h]), W.bits(rec["total_cis"][b, rows, h])))
                        if not ok:
                            contents_bad.append((ri, rec["layer"], h, b, r, k))
    return dict(checked=checked, missing=len(missing), contents_bad=contents_bad, missing_list=miss)
