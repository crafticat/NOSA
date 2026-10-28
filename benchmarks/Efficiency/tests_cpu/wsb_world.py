"""The CPU WORLD of tests_cpu/test_wsb.py (WS-B of KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs: the joint pass with ONE residency for the union, verify_rows mode "union"; not a test file). Load it by
file path, as the older tests load their worlds.

It builds on (read their docstrings first):
  ragged_world.py   the SHIPPED engine from its own source, every CUDA op a torch stand-in with the kernel's flat
                    addresses (lesson L28); fake_nosa_linear row by row (verify1 fix 6adf18d)
  verify_world.py   verify_rows with the stand-ins; fake_attention_idx (cache_batch_idx, return_softmax_lse)
  spec_world.py     the speculative round runner against the shipped greedy decode (residency / host-row checks)

THIS FILE ADDS
  fake_attention_u     flash_attn_nosa_with_kvcache as verify_world.fake_attention_idx, PLUS the kernel's EARLY EXIT
                       for an entry whose cache_seqlens is 0 (flash_fwd_kernel.h:603-641: each split writes out 0 and
                       lse -inf; the combine, :1331-1339, writes lse +inf and out 0). Every other entry is
                       verify_world._entry, op for op (test_wsb.py checks the bits against fake_attention_idx).
  SelRecorder          the SH reference's selection: a hook on the body's decode_fixed of the SERIAL world records the
                       top-k ids of every layer of every step (the step's own st.ti; no change to the step).
  run_union_vs_sh      THE CHECK OF FAST ITSELF AGAINST SH (the coordinator's binding correction, 2026-10-28): two
                       worlds of one build; world A runs the SERIAL ragged step (ordinary NOSA per request; at equal
                       lengths with natural ops it is bitwise the shipped step, test_ragged_engine.py); world B runs the
                       union joint pass on the same tokens (teacher-forced). Per pass: every row's selection set
                       against SH's at its position (layer by layer), the presence and visible-count counters, the
                       logits deltas per verified position, row 0 of the first pass bitwise.
  union_bytes          the byte check of a record: every selected block of a full row is in the window or in a store
                       slot of some fill (K, V and bias rows equal the host rows and total_cis), or it is the tail /
                       next block; the resident rows' blocks too, or they are skipped.
  sh_attention_ref     the attention of SH's semantics in float64, ORDER-FREE, from a union record: row r reads the
                       complete blocks of its selection from the host rows, its tail block's rows up to its own
                       position from the tail slot (F4 kept: the fill row reads none), and past a block edge the block
                       m in full plus the next block's rows up to its own position.
  run_spec_union       spec_world.run_vs_shipped with the union stand-in and a chosen X (spec_world's runner pins
                       extra 16 and fake_attention_idx).
  old_module           a module of an OLDER commit's source (git show), loaded into the synthetic package (Gate 0).
"""
from __future__ import annotations

import contextlib
import importlib.util
import subprocess
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


SW = _load("spec_world")
V = SW.VW
W, RE, VR, SR, RW, BC = V.W, V.RE, V.VR, SW.SR, SW.RW, SW.BC
ROOT = W.ROOT
L, BS, TOPK, HKV, HQ, D = W.L, W.BS, W.TOPK, W.HKV, W.HQ, W.D


# ---------------------------------------------------------------------------
# the attention stand-in with the kernel's early exit
# ---------------------------------------------------------------------------
def fake_attention_u(q, k_cache, v_cache, kv_bias, cache_seqlens=None, cache_batch_idx=None, num_splits=0,
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
        n = int(cache_seqlens[e])
        assert 0 <= n <= rows, "cache_seqlens past the cache rows"
        if n == 0:                                                   # the kernel's early exit and the combine
            out[e, 0] = 0
            lse[e, :, 0] = float("inf")
            continue
        idx = off0 + e * st[0] + r_ * st[1] + h_ * st[2]             # the bias row of ENTRY e (bidb)
        assert int(idx.max()) < flat.numel(), "the bias read of entry %d is past the bias storage" % e
        o, l_ = V._entry(q[e, 0], k_cache[cbi[e]], v_cache[cbi[e]], flat[idx], n, per, Hq, Dq)
        out[e, 0] = o.to(q.dtype)
        lse[e, :, 0] = l_
    return (out, lse) if return_softmax_lse else out


# ---------------------------------------------------------------------------
# the SH reference's selection
# ---------------------------------------------------------------------------
class SelRecorder:
    """A hook on rs.body.decode_fixed of the SERIAL world: each call records topk_idx (the step's own top-k ids,
    (H, B, topk)) under (step, layer). attach() / detach() as DraftResident does (an instance attribute)."""

    def __init__(self, rs):
        self.rs = rs
        self.lay = {lay.cache_engine._k_gpu.data_ptr(): l for l, lay in enumerate(rs.cache.layers)}
        self.step = 0
        self.sel = {}                                                 # (step, layer) -> (H, B, topk)
        body = rs.body
        self.orig = body.decode_fixed
        self.own = "decode_fixed" in body.__dict__
        body.decode_fixed = self._hook

    def _hook(self, eng, key_states, value_states, kv_bias, topk_idx, *args):
        self.sel[(self.step, self.lay[eng._k_gpu.data_ptr()])] = topk_idx.clone()
        return self.orig(eng, key_states, value_states, kv_bias, topk_idx, *args)

    def detach(self):
        if self.own:
            self.rs.body.decode_fixed = self.orig
        else:
            del self.rs.body.decode_fixed


def _sorted_eq(a, b):
    """(H, B) bool: the two (H, B, T) id sets are equal; and the symmetric-difference size (H, B)."""
    sa, sb = a.sort(-1).values, b.sort(-1).values
    eq = (sa == sb).all(-1)
    only_a = ~(a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)
    only_b = ~(b.unsqueeze(-1) == a.unsqueeze(-2)).any(-1)
    return eq, only_a.sum(-1) + only_b.sum(-1)


# ---------------------------------------------------------------------------
# the check of FAST itself against SH
# ---------------------------------------------------------------------------
def run_union_vs_sh(S_list, groups, K=3, seed=0, extra=16, spill=None, neg=None, gemm="exact", record=True,
                    draft_last=False, natural=False, count_diff=True):
    """Two worlds of one build (module docstring). World A: the SERIAL ragged step, row by row (SH); world B: the union
    joint pass (draft_last False: every row a verify row, so every row's K / V stays committed state as SH's). ops:
    natural False = verify_world.exact_ragged_ops (row_linear, twin bi) in both worlds and gemm "exact" (row_linear) in
    the joint pass: every op but the attention is row-invariant, so a difference is the union attention's; natural
    True = ops natural / twin torch (the shipped bits at equal lengths) and the given gemm (rowgroup: the shipped GEMM
    shape). Returns the evidence: per (group, layer, row) the selection changes (with the symmetric-difference size),
    per (group, row, request) the logits delta, row 0 of group 0 bitwise or not, the verify stats, the diff-kernel
    calls per pass (count_diff), the log."""
    W.STATS.clear()
    R = K + 1
    steps = groups * R
    B = len(S_list)
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, S_list)
    ops_of = (lambda: RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)) if natural else V.exact_ragged_ops
    worlds = []
    for _ in range(2):
        s = RE.build(model, W.new_cache(0), S_list, steps, ops_of(), lambda b, l: rows[(b, l)], pin=False)
        W.poison_built(s, S_list)
        worlds.append(s)
    A, Bw = worlds
    pre = V.engines_equal(A.cache, Bw.cache, (A.st.twins, Bw.st.twins))
    assert pre == [], "precondition: the two built worlds differ before step 0: %s" % pre[:5]
    lin = W.row_linear if gemm == "exact" else None
    vops = VR.VerifyOps(K=K, mode="union", gemm=gemm, linear=lin, extra=extra, attn=fake_attention_u, neg=neg,
                        record=record, check=True, spill=spill, draft_last=draft_last)
    vs = VR.attach(model, Bw, vops)
    vs.fa_check = {}
    diff_calls = []
    if count_diff:                                                   # an independent count of the residency kernel
        kd = Bw.ops.kernels["diff"]

        def counted(*a, **kw):
            diff_calls.append(W.TAG[0])
            return kd(*a, **kw)
        Bw.ops.kernels["diff"] = counted
    rec = SelRecorder(A)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    ev = dict(changed=[], deltas=[], row0_bits=None, groups=groups, R=R, fills=[], diff_per_pass=[], argmax_bad=[],
              compared=0)
    for grp in range(groups):
        las = []
        for r in range(R):
            t = grp * R + r
            p = S_t + t
            for b in range(B):
                if int(p[b]) % W.BS == W.BS - 1:
                    ev["fills"].append((grp, r, b))
            rec.step = t
            W.TAG[0] = "serial"
            la = RE.decode_step(model, A, tokens[:, t:t + 1], p.view(B, 1).to(torch.int32))
            las.append(la[:, -1])
        W.TAG[0] = "joint"
        n0 = len(diff_calls)
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        lb = VR.joint_pass(model, vs, tokens[:, grp * R:(grp + 1) * R], pos)
        ev["diff_per_pass"].append(len(diff_calls) - n0)
        full = vs.full_default if vs.full_rows is None else vs.full_rows
        for l in range(L):
            for r in range(R):
                if not bool(full[r].any()):
                    continue
                eq, sym = _sorted_eq(vs.chk_sel[l][:, :, r], rec.sel[(grp * R + r, l)])
                ev["compared"] += int(full[r].sum()) * eq.shape[0]
                for h, b in (~eq).nonzero().tolist():
                    if bool(full[r, b]):
                        ev["changed"].append((grp, l, r, h, b, int(sym[h, b])))
        for r in range(R):
            la, got = las[r], lb[:, r]
            assert bool(torch.isfinite(la).all()) and bool(torch.isfinite(got).all()), (grp, r)
            for b in range(B):
                if not bool(full[r, b]):                             # a resident row is a draft: not checked here
                    continue
                d = float(((la[b] - got[b]).abs() / (1 + la[b].abs())).max())
                ev["deltas"].append((grp, r, b, d))
                if int(la[b].argmax()) != int(got[b].argmax()):
                    ev["argmax_bad"].append((grp, r, b))
            if grp == 0 and r == 0:
                ev["row0_bits"] = torch.equal(W.bits(la), W.bits(got))
    rec.detach()
    ev.update(vs=vs, A=A, Bw=Bw, model=model, stats=VR.read_stats(vs), errors=(RE.errors(A), RE.errors(Bw)),
              max_delta=max((d for *_, d in ev["deltas"]), default=0.0))
    return ev


# ---------------------------------------------------------------------------
# the byte check and the order-free reference attention (union records)
# ---------------------------------------------------------------------------
def _bits_eq(a, b) -> bool:
    return tuple(a.shape) == tuple(b.shape) and torch.equal(W.bits(a), W.bits(b))


def union_bytes(log, X) -> dict:
    """For every union record and every (head h, request b, row r) of a FULL or RESIDENT row and every id k of its
    selection: k is the tail block m (the tail slot names it), or the next block m + 1 of a row past the block edge
    (store slot 0), or a window slot names k and holds the host rows / cis rows of k, or store slot x of a fill holds
    listed id k with those rows. Returns dict(checked, missing_full (list), missing_res (count), contents_bad (list),
    fills (store fills recorded))."""
    out = dict(checked=0, missing_full=[], missing_res=0, contents_bad=[], fills=0)
    for rec in (x for x in log if x["kind"] == "union"):
        sel, win, full, res = rec["sel"], rec["win"], rec["full"], rec["res"]
        H, B, R, T = sel.shape
        tail = win.shape[2] - 1
        fills = rec["fills"]
        out["fills"] += len(fills)
        where = {}                                                   # (h, b, k) -> (source tensors, slot)
        for f in fills:
            for h in range(H):
                for b in range(B):
                    for x, k in enumerate(f["lst"][h, b].tolist()):
                        if k >= 0:
                            where[(h, b, k)] = ((f["ext_k"], f["ext_v"], f["ext_bias"]), x + 1)
        for h in range(H):
            for b in range(B):
                m = int(rec["m"][b])
                wl = win[h, b].tolist()
                for r in range(R):
                    is_full, is_res = bool(full[r, b]), bool(res[r, b])
                    if not (is_full or is_res):
                        continue
                    for k in sel[h, b, r].tolist():
                        if k < 0:
                            continue
                        out["checked"] += 1
                        if k == m and wl[tail] == m:
                            continue
                        if k == m + 1 and not bool(rec["in_tail"][b, r]):
                            continue
                        if k in wl[:tail]:
                            src, s = (rec["k_gpu"], rec["v_gpu"], rec["bias_gpu"]), wl.index(k)
                        elif (h, b, k) in where:
                            src, s = where[(h, b, k)]
                        else:
                            if is_full:
                                out["missing_full"].append((rec["layer"], h, b, r, k))
                            else:
                                out["missing_res"] += 1
                            continue
                        dst, rows = slice(s * BS, (s + 1) * BS), slice(k * BS, (k + 1) * BS)
                        ok = (_bits_eq(src[0][b, dst, h], rec["k_cpu"][b, rows, h])
                              and _bits_eq(src[1][b, dst, h], rec["v_cpu"][b, rows, h])
                              and _bits_eq(src[2][b, dst, h], rec["total_cis"][b, rows, h].to(src[2].dtype)))
                        if not ok:
                            out["contents_bad"].append((rec["layer"], h, b, r, k))
    return out


def sh_attention_ref(rec, R: int):
    """The attention of SH's semantics for every FULL row of one union record, float64, order-free (module
    docstring). Returns (ref (E, Hq, D) float64, mask (E,) bool: the entries computed)."""
    sel, in_tail, p_rows = rec["sel"], rec["in_tail"], rec["p_rows"]
    H, B, R_, T = sel.shape
    q = rec["q_e"].double()                                          # (E, 1, Hq, D)
    E, _, Hq, Dq = q.shape
    g = Hq // H
    tail0 = (TOPK - 1) * BS
    ref = torch.zeros((E, Hq, Dq), dtype=torch.float64)
    done = torch.zeros(E, dtype=torch.bool)
    slot0 = rec["fills"][0]
    for b in range(B):
        m = int(rec["m"][b])
        for r in range(R):
            if not bool(rec["full"][r, b]):
                continue
            e = b * R + r
            p = int(p_rows[b, r])
            for h in range(H):
                ks, vs_, bs_ = [], [], []
                ids = [k for k in sel[h, b, r].tolist() if k >= 0]
                for k in ids:
                    if k in (m, m + 1):
                        continue
                    rows = slice(k * BS, (k + 1) * BS)
                    ks.append(rec["k_cpu"][b, rows, h])
                    vs_.append(rec["v_cpu"][b, rows, h])
                    bs_.append(rec["total_cis"][b, rows, h])
                if bool(in_tail[b, r]):
                    n_t = (p + 1) % BS                               # F4: the shipped length (r2_decode_body:93)
                    ks.append(rec["k_gpu"][b, tail0:tail0 + n_t, h])
                    vs_.append(rec["v_gpu"][b, tail0:tail0 + n_t, h])
                    bs_.append(rec["bias_gpu"][b, tail0:tail0 + n_t, h])
                else:
                    if m in ids:                                     # block m, complete, in the tail slot
                        ks.append(rec["k_gpu"][b, tail0:tail0 + BS, h])
                        vs_.append(rec["v_gpu"][b, tail0:tail0 + BS, h])
                        bs_.append(rec["bias_gpu"][b, tail0:tail0 + BS, h])
                    n1 = p % BS + 1                                  # the next block's rows up to the own position
                    ks.append(slot0["ext_k"][b, :n1, h])
                    vs_.append(slot0["ext_v"][b, :n1, h])
                    bs_.append(slot0["ext_bias"][b, :n1, h])
                kk = torch.cat(ks).double()
                vv = torch.cat(vs_).double()
                bb = torch.cat(bs_).double()
                qq = q[e, 0, h * g:(h + 1) * g]                      # (g, D)
                sc = qq @ kk.t() * Dq ** -0.5 + bb.view(1, -1)
                ref[e, h * g:(h + 1) * g] = torch.softmax(sc, -1) @ vv
            done[e] = True
    return ref, done


# ---------------------------------------------------------------------------
# the speculative runner with the union stand-in
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _union_runner(extra, joint_stats=True):
    """spec_world.spec_runner with attn=fake_attention_u, the given X and joint_stats (spec_world pins extra 16 and
    fake_attention_idx, which has no early exit for a length-0 entry)."""
    orig = SW.spec_runner

    def spec_runner(model, cache, S, horizon, tok0, level="L3", K=SW.K0, mode="exact", gemm="rowgroup",
                    draft_filter=None, neg=None, rb_neg=None, record=True, prefetch=None, after_round=None,
                    journal=RW.CAP, body="torch", redirect=True, pin=False, make_prefetch=None, sr=None):
        SRm = SR if sr is None else sr
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body=body, graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        RW.define_undefined(rs)
        if make_prefetch is not None:
            prefetch = make_prefetch(rs)
        sops = SRm.SpecOps(level=level, K=K, mode=mode, gemm=gemm, extra=extra, attn=fake_attention_u,
                           journal=journal, redirect=redirect, draft_filter=draft_filter, prefetch=prefetch,
                           rb_neg=rb_neg, neg=neg, record=record, probe_every=0, after_round=after_round, pin=pin,
                           joint_stats=joint_stats)
        return SRm.SpecRunner(model, rs, sops, BC.HostClock(), t0=tok0)
    SW.spec_runner = spec_runner
    try:
        yield
    finally:
        SW.spec_runner = orig


def run_spec_union(B, level="L4", extra=16, gemm="natural", **kw):
    """spec_world.run_vs_shipped(level, mode "union") with the union runner (the evidence of run_vs_shipped)."""
    with _union_runner(extra):
        return SW.run_vs_shipped(B, level=level, mode="union", gemm=gemm, **kw)


# ---------------------------------------------------------------------------
# Gate 0: an older commit's module in the synthetic package
# ---------------------------------------------------------------------------
def git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def old_module(name: str, commit: str, tmp_path, patch_verify=False):
    """nosi/nosi/<name>.py at `commit`, as the module <PKG>.<name>_<commit[:7]> of the synthetic package (its relative
    imports resolve to THIS tree's patched modules). patch_verify: the stand-ins of verify_world (torch.topk in CUDA's
    order, the per-request pooling)."""
    full = "%s.%s_%s" % (W.PKG, name, commit[:7])
    if full in sys.modules:
        return sys.modules[full]
    r = git("show", "%s:nosi/nosi/%s.py" % (commit, name))
    assert r.returncode == 0, r.stderr
    p = tmp_path / ("%s_%s.py" % (name, commit[:7]))
    p.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location(full, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    if patch_verify:
        mod.torch = W.TP
        mod.nosa_pooling_per_request = W.fake_pool_per_request
    return mod


def joint_run(VRm, S_list, groups, mode, gemm, K=3, seed=0, extra=16):
    """Joint passes of module VRm (a verify_rows source) on a built world (EXACT ragged ops), random tokens: returns
    (the logits of every pass, the world). The Gate 0 driver: the same call on two sources of verify_rows."""
    R = K + 1
    steps = groups * R
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, S_list)
    s = RE.build(model, W.new_cache(0), S_list, steps, V.exact_ragged_ops(), lambda b, l: rows[(b, l)], pin=False)
    W.poison_built(s, S_list)
    lin = W.row_linear if gemm == "exact" else None
    vs = VRm.attach(model, s, VRm.VerifyOps(K=K, mode=mode, gemm=gemm, linear=lin, extra=extra,
                                            attn=V.fake_attention_idx))
    B = len(S_list)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    out = []
    for grp in range(groups):
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        out.append(VRm.joint_pass(model, vs, tokens[:, grp * R:(grp + 1) * R], pos))
    return out, s
