"""The CPU WORLD of tests_cpu/test_wsc.py and tests_cpu/test_wsc_interp.py (WS-C of KS/analysis/plan/
PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs: the union joint pass with ONE call of the shared-KV fork
per layer, verify_rows mode "union" with joint_attn "fork"; not a test file). Load it by file path.

It builds on wsb_world.py (over spec_world / verify_world / ragged_world: the SHIPPED engine from its own source, every
CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). Read their docstrings first.

THIS FILE ADDS
  fake_fork            the fork's unified_attention (benchmarks/Efficiency/fork_vllm_unified_attention.py, a byte copy
                       of fia6's) as a torch stand-in, path BUILD_UNION + 2D + extents + CIS bias only:
                         the launcher's checks (fork :952-957, :856-891, :1018-1021; BLOCK_M >= 16 only when
                         block_m_min asks for it: the CPU world has 2 query heads per KV head, BLOCK_M 8);
                         fork_build_union (:271-344): per (sequence, KV head) the DISTINCT values block id << 32 | slot
                         of the rows' lists in ascending order (each list strictly ascending: the precondition, :296-298,
                         ASSERTED here), physical block = entry x slots_per_entry + slot, the row bytes (64; the row's
                         LAST listed page: last_len), the length;
                         the tile loop (:570-689): one tile = one block, S = (q.k + bias) x scale, the rows' extents
                         (-inf), V read as 0 past the largest byte of the entry, the online softmax of softmax_step
                         (:218-241: M -inf and L 1 at the start, m_j 0 when the row has seen nothing yet), acc / L.
                       Float32, ROW BY ROW with fixed shapes, so a row's bits do not depend on its siblings (the
                       kernel's MMA rows are independent too; fia6 RA measured 0 of 16 rows differ).
  FORK_CALLS           one entry (rows, sequences) per launch of fake_fork (the call count of a pass)
  fork_ref64           the attention of the same lists in float64, ORDER-FREE (each row: one softmax over the tokens
                       its own list shows): the reference of fake_fork and of the real kernel
  union_ref            the union tables of row lists, built independently (torch.unique per (sequence, head))
  TeeFork              a fork callable that runs TWO forks on the same inputs (separate outputs and tables) and keeps
                       the largest output difference and the table differences (the interpreter test: real vs stand-in)
  run_fork_vs_sh       wsb_world.run_union_vs_sh with the K / V arena and joint_attn "fork": THE CHECK OF FAST ITSELF
                       AGAINST SH (selections, presence, VIS / VB from the kernel's tables, the "F" / "RA" probes, the
                       logits deltas)
  fork_bytes           the byte check of a fork record: every (row, block) that the KERNEL's tables show holds the
                       block's host rows (K, V, bias), or it is the tail / next block
  run_spec_fork        spec_world.run_vs_shipped with the arena and joint_attn "fork" (L4XF, L4P fork)
  joint_run_union      joint passes of mode union (joint_attn "store") of a verify_rows source (the Gate 0 driver)
"""
from __future__ import annotations

import contextlib
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


X_ = _load("wsb_world")
SW, V, W, RE, VR, SR, RW, BC = X_.SW, X_.V, X_.W, X_.RE, X_.VR, X_.SR, X_.RW, X_.BC
ROOT, EFF = W.ROOT, W.EFF
if str(EFF) not in sys.path:
    sys.path.insert(0, str(EFF))
import r2_kv_arena as KA          # noqa: E402

FORK_CALLS = []
NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# the fork's stand-in
# ---------------------------------------------------------------------------
def _p2(x: int) -> int:
    p = 1
    while p < int(x):
        p *= 2
    return p


def _union_of(build, r0, n, h, S):
    """fork_build_union for one (sequence, KV head): (values in ascending order, {value: (physical block, [bytes of
    the BLOCK_Q rows])}). The precondition (strictly ascending block ids per row) is asserted."""
    key, slot, npg, last, entry = (build[k] for k in ("page_key", "page_table", "num_pages", "last_len", "seq_idx"))
    vals = {}
    for u in range(n):
        e = r0 + u
        np_ = int(npg[e, h])
        ks = key[e, h, :np_].tolist()
        ss = slot[e, h, :np_].tolist()
        assert all(ks[i] < ks[i + 1] for i in range(np_ - 1)), "precondition (fork :296-298): row %d head %d" % (e, h)
        assert all(x >= 0 for x in ss), "a negative slot in row %d head %d" % (e, h)
        for i in range(np_):
            vv = (int(ks[i]) << 32) | int(ss[i])
            phys = int(entry[e]) * S + int(ss[i])
            got = vals.setdefault(vv, [phys, [0] * 4])
            assert got[0] == phys
            got[1][u] = int(last[e]) if i == np_ - 1 else None
    return sorted(vals), vals


def make_fake_fork(block_m_min: int = 1):
    """fake_fork (module docstring); block_m_min 16 = the real launcher's BLOCK_M check."""

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                          window_size, block_table, softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None,
                          num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None,
                          softmax_segm_expsum=None, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
                          mm_prefix_range=None, use_alibi_sqrt=False, kv_quant_mode=0, k_scale_cache=None,
                          v_scale_cache=None, chunk_lookback=-1, union_ext=None, union_len=None, cis_bias=None,
                          build_union=None, rows_per_q_block=4, num_warps=None, num_stages=None, tile_size=None,
                          use_extents=True):
        assert causal and q_descale is None and window_size[0] < 0 and chunk_lookback == -1
        assert alibi_slopes is None and qq_bias is None and mm_prefix_range is None and sinks is None
        assert int(kv_quant_mode) == 0 and output_scale is None and softcap == 0 and seq_threshold_3D is None
        assert union_ext is not None and union_len is not None, "[fork] the union tables are required"
        assert build_union is not None and use_extents, "the stand-in models BUILD_UNION with extents only"
        nb, bs, hkv, d = k.shape
        num_seqs = len(seqused_k)
        hq = q.shape[1]
        g = hq // hkv
        BQ = int(rows_per_q_block)
        assert BQ in (1, 2, 4) and int(max_seqlen_q) <= BQ, "[fork] one q block (<= 4 rows) per request"
        assert BQ * g >= block_m_min and _p2(BQ * g) == BQ * g, "[fork] the stacked tile rows (BLOCK_M)"
        assert bs == _p2(bs) and bs <= 128, "[fork] extents fit 8 bits"
        assert tile_size in (None, bs), "the stand-in: one tile = one block (FV6: tile 64)"
        Wd = block_table.shape[-1]
        for t in (block_table, union_ext):
            assert t.dtype == torch.int32 and t.is_contiguous() and tuple(t.shape) == (hkv, num_seqs, Wd)
        assert union_len.dtype == torch.int32 and union_len.is_contiguous() and tuple(union_len.shape) == (hkv, num_seqs)
        assert cis_bias is not None and tuple(cis_bias.shape) == (nb, bs, hkv) and cis_bias.is_floating_point()
        key = build_union["page_key"]
        r_, hk2, p_ = key.shape
        for name in ("page_key", "page_table", "num_pages", "last_len", "seq_idx"):
            t = build_union[name]
            assert t.dtype == torch.int32 and t.is_contiguous(), "[fork] the row lists: contiguous int32 (%s)" % name
        assert hk2 == hkv and r_ == q.shape[0] and tuple(build_union["num_pages"].shape) == (r_, hkv)
        assert Wd >= BQ * _p2(p_), "[fork] the table width W must hold BLOCK_Q x P_PAD entries"
        S = int(build_union["slots_per_entry"])
        FORK_CALLS.append((r_, num_seqs))
        cu = [int(x) for x in cu_seqlens_q.tolist()]
        assert len(cu) == num_seqs + 1 and cu[0] == 0 and cu[-1] <= r_
        scale = float(softmax_scale)
        tok = torch.arange(bs)
        for s in range(num_seqs):
            r0, n = cu[s], cu[s + 1] - cu[s]
            if n == 0:                                           # every program of the sequence returns at once
                continue
            assert n <= BQ
            for h in range(hkv):
                order, vals = _union_of(build_union, r0, n, h, S)
                assert len(order) <= BQ * _p2(p_)
                for j, vv in enumerate(order):
                    phys, byts = vals[vv]
                    block_table[h, s, j] = phys
                    union_ext[h, s, j] = sum(int(b or bs) << (8 * u) for u, b in enumerate(byts) if b != 0)
                union_len[h, s] = len(order)
                ents = []
                for vv in order:                                 # per entry: block, the rows' bytes, the reach
                    phys, byts = vals[vv]
                    bb = [(bs if b is None else int(b)) for b in byts]
                    ents.append((phys, bb, max(bb)))
                for u in range(n):
                    e = r0 + u
                    qh = q[e, h * g:(h + 1) * g].float()
                    M = torch.full((g,), NEG_INF)
                    L = torch.ones(g)
                    acc = torch.zeros((g, d))
                    for phys, bb, reach in ents:
                        kt = k[phys, :, h].float()
                        vt = torch.where((tok < reach).view(bs, 1), v[phys, :, h].float(), torch.zeros(()))
                        bt = cis_bias[phys, :, h].float()
                        Sx = (qh @ kt.t() + bt.view(1, bs)) * scale
                        Sx = torch.where((tok < bb[u]).view(1, bs), Sx, torch.full((), NEG_INF))
                        m_j = torch.maximum(M, Sx.max(-1).values)
                        m_j = torch.where(m_j > NEG_INF, m_j, torch.zeros(()))
                        P = torch.exp(Sx - m_j.view(g, 1))
                        l_j = P.sum(-1)
                        alpha = torch.exp(M - m_j)
                        L = L * alpha + l_j
                        acc = acc * alpha.view(g, 1)
                        acc = acc + P @ vt
                        M = m_j
                    out[e, h * g:(h + 1) * g] = (acc / L.view(g, 1)).to(out.dtype)
        return None

    unified_attention.__module__ = "wsc_world.fake_fork"
    return unified_attention


fake_fork = make_fake_fork(1)


def fork_ref64(q, k, v, cis_bias, build, cu, scale, rows=None):
    """float64, order-free: each row's softmax over the tokens of its own list (64 per page, last_len on its last
    listed page), physical block = entry x S + slot. Returns (rows, Hq, D) float64 for `rows` (default: all)."""
    key, slot, npg, last, entry = (build[x] for x in ("page_key", "page_table", "num_pages", "last_len", "seq_idx"))
    S = int(build["slots_per_entry"])
    nb, bs, hkv, d = k.shape
    hq = q.shape[1]
    g = hq // hkv
    rows = list(range(q.shape[0])) if rows is None else list(rows)
    out = torch.zeros((len(rows), hq, d), dtype=torch.float64)
    for i, e in enumerate(rows):
        for h in range(hkv):
            ks, vs_, bs_ = [], [], []
            np_ = int(npg[e, h])
            for j in range(np_):
                phys = int(entry[e]) * S + int(slot[e, h, j])
                n = int(last[e]) if j == np_ - 1 else bs
                ks.append(k[phys, :n, h])
                vs_.append(v[phys, :n, h])
                bs_.append(cis_bias[phys, :n, h])
            if not ks:
                continue
            kk, vv, bb = torch.cat(ks).double(), torch.cat(vs_).double(), torch.cat(bs_).double()
            sc = (q[e, h * g:(h + 1) * g].double() @ kk.t() + bb.view(1, -1)) * scale
            out[i, h * g:(h + 1) * g] = torch.softmax(sc, -1) @ vv
    return out


def union_ref(build, cu, hkv, width, bs=64):
    """The union tables (table, ext, len) of row lists, built independently with torch.unique per (sequence, head)."""
    key, slot, npg, last, entry = (build[x] for x in ("page_key", "page_table", "num_pages", "last_len", "seq_idx"))
    S = int(build["slots_per_entry"])
    ns = len(cu) - 1
    tab = torch.zeros((hkv, ns, width), dtype=torch.int64)
    ext = torch.zeros((hkv, ns, width), dtype=torch.int64)
    ln = torch.zeros((hkv, ns), dtype=torch.int64)
    for s in range(ns):
        for h in range(hkv):
            r0, r1 = int(cu[s]), int(cu[s + 1])
            parts = []
            for e in range(r0, r1):
                n = int(npg[e, h])
                parts.append(key[e, h, :n].to(torch.int64) * (1 << 32) + slot[e, h, :n].to(torch.int64))
            if not parts:
                continue
            u = torch.unique(torch.cat(parts))                  # sorted
            ln[h, s] = u.numel()
            for j, vv in enumerate(u.tolist()):
                sl_ = vv & 0xFFFFFFFF
                e0 = None
                word = 0
                for r, e in enumerate(range(r0, r1)):
                    n = int(npg[e, h])
                    vals = (key[e, h, :n].to(torch.int64) * (1 << 32) + slot[e, h, :n].to(torch.int64)).tolist()
                    if vv in vals:
                        i = vals.index(vv)
                        byte = int(last[e]) if i == n - 1 else bs
                        word |= byte << (8 * r)
                        e0 = e
                tab[h, s, j] = int(entry[e0]) * S + sl_
                ext[h, s, j] = word
    return tab, ext, ln


class TeeFork:
    """Two fork callables on the same inputs: `main` writes the caller's output and tables; `other` writes copies.
    Keeps per call the largest relative output difference and whether the tables (entries below the length) and the
    lengths are equal."""

    def __init__(self, main, other):
        self.main, self.other = main, other
        self.diffs, self.tables_equal = [], []
        self.__module__ = "wsc_world.TeeFork"

    def __call__(self, q, k, v, out, cu, qmax, seq, mk, scale, causal, win, tab, softcap, qd, kd, vd, **kw):
        o2 = torch.empty_like(out)
        t2, x2, l2 = torch.zeros_like(tab), torch.zeros_like(kw["union_ext"]), torch.zeros_like(kw["union_len"])
        kw2 = dict(kw, union_ext=x2, union_len=l2)
        self.other(q, k, v, o2, cu, qmax, seq, mk, scale, causal, win, t2, softcap, qd, kd, vd, **kw2)
        self.main(q, k, v, out, cu, qmax, seq, mk, scale, causal, win, tab, softcap, qd, kd, vd, **kw)
        ln = kw["union_len"]
        live = torch.arange(tab.shape[-1]).view(1, 1, -1) < ln.to(torch.int64).unsqueeze(-1)
        eq = torch.equal(ln, l2) and torch.equal(tab[live], t2[live]) and torch.equal(kw["union_ext"][live], x2[live])
        self.tables_equal.append(bool(eq))
        a, b = out.double(), o2.double()
        self.diffs.append(float(((a - b).abs() / (1 + b.abs())).max()) if out.numel() else 0.0)


# ---------------------------------------------------------------------------
# the arena on CPU worlds
# ---------------------------------------------------------------------------
def arena_for(cache, B: int, X: int):
    """A K / V arena for a built or prefilled CPU world (r2_kv_arena.for_engine on engine 0) and the re-home."""
    a = KA.for_engine(cache.layers[0].cache_engine, len(cache.layers), B, X)
    KA.rehome(cache, a)
    return a


# ---------------------------------------------------------------------------
# the check of FAST (fork) itself against SH
# ---------------------------------------------------------------------------
def run_fork_vs_sh(S_list, groups, K=3, seed=0, X=None, neg=None, gemm="exact", record=True, draft_last=False,
                   natural=False, fork=None, probe=True, count_diff=True):
    """wsb_world.run_union_vs_sh (read it) with the arena (world B re-homed after the build) and joint_attn "fork".
    Returns its evidence plus: the fork calls per pass (FORK_CALLS), rehome_equal (world B's engine tables after the
    re-home = world A's, bitwise), the arena."""
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
    full_max = R - 1 if draft_last else R
    arena = arena_for(Bw.cache, B, X or KA.store_slots(W.TOPK, full_max))
    KA.attach(Bw, arena)
    rehome_equal = V.engines_equal(A.cache, Bw.cache, (A.st.twins, Bw.st.twins)) == []
    lin = W.row_linear if gemm == "exact" else None
    vops = VR.VerifyOps(K=K, mode="union", gemm=gemm, linear=lin, extra=16, attn=X_.fake_attention_u, neg=neg,
                        record=record, check=True, draft_last=draft_last, joint_attn="fork",
                        fork=(fork if fork is not None else fake_fork))
    vs = VR.attach(model, Bw, vops)
    vs.fa_check = {} if probe else None
    diff_calls = []
    if count_diff:                                                   # an independent count of the residency kernel
        kd = Bw.ops.kernels["diff"]

        def counted(*a, **kw):
            diff_calls.append(W.TAG[0])
            return kd(*a, **kw)
        Bw.ops.kernels["diff"] = counted
    rec = X_.SelRecorder(A)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    ev = dict(changed=[], deltas=[], row0_bits=None, groups=groups, R=R, fills=[], diff_per_pass=[], argmax_bad=[],
              compared=0, fork_per_pass=[], rehome_equal=rehome_equal, arena=arena)
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
        n0, f0 = len(diff_calls), len(FORK_CALLS)
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        lb = VR.joint_pass(model, vs, tokens[:, grp * R:(grp + 1) * R], pos)
        ev["diff_per_pass"].append(len(diff_calls) - n0)
        ev["fork_per_pass"].append(len(FORK_CALLS) - f0)
        full = vs.full_default if vs.full_rows is None else vs.full_rows
        for l in range(W.L):
            for r in range(R):
                if not bool(full[r].any()):
                    continue
                eq, sym = X_._sorted_eq(vs.chk_sel[l][:, :, r], rec.sel[(grp * R + r, l)])
                ev["compared"] += int(full[r].sum()) * eq.shape[0]
                for h, b in (~eq).nonzero().tolist():
                    if bool(full[r, b]):
                        ev["changed"].append((grp, l, r, h, b, int(sym[h, b])))
        for r in range(R):
            la, got = las[r], lb[:, r]
            assert bool(torch.isfinite(la).all()) and bool(torch.isfinite(got).all()), (grp, r)
            for b in range(B):
                if not bool(full[r, b]):
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
# the byte check of a fork record
# ---------------------------------------------------------------------------
def fork_bytes(log, R: int) -> dict:
    """For every fork record: every union entry (head h, sequence s, u < len) of the KERNEL's tables is mapped to its
    location (layer view slot or store slot) and its block id (window / pool / store list / next block). An entry a
    row SEES (byte > 0) must: belong to the row's request; name a block of the row's own selection; hold the block's
    host rows (K, V, bias = total_cis) unless it is the tail block m (the tail slot) or the next block m + 1 (store
    slot 0). Returns dict(checked, bad_contents, foreign, cross_request)."""
    out = dict(checked=0, bad_contents=[], foreign=[], cross_request=[])
    for rec in (x for x in log if x["kind"] == "fork"):
        sel, win, pm, lst, m = rec["sel"], rec["win"], rec["pool_map"], rec["lst"], rec["m"]
        H, B, R_, T = sel.shape
        topk = win.shape[2]
        ns = rec["k_gpu"].shape[1] // W.BS
        X = rec["fills"][0]["ext_k"].shape[1] // W.BS
        base, sbase = rec["base"], rec["store_base"]
        nch = rec["tab"].shape[1] // B
        for h in range(H):
            for s in range(rec["tab"].shape[1]):
                b = s // nch
                c = s % nch
                for j in range(int(rec["ulen"][h, s])):
                    phys = int(rec["tab"][h, s, j])
                    word = int(rec["ext"][h, s, j])
                    if base <= phys < base + B * ns:
                        bb, sl_ = divmod(phys - base, ns)
                        bid = int(win[h, bb, sl_]) if sl_ < topk else int(pm[h, bb, sl_ - topk])
                        src = (rec["k_gpu"], rec["v_gpu"], rec["bias_gpu"], sl_)
                    elif sbase <= phys < sbase + B * X:
                        bb, x = divmod(phys - sbase, X)
                        bid = int(m[bb]) + 1 if x == 0 else int(lst[h, bb, x - 1])
                        f = rec["fills"][0]
                        src = (f["ext_k"], f["ext_v"], f["ext_bias"], x)
                    else:
                        out["cross_request"].append((rec["layer"], h, s, j, "outside"))
                        continue
                    for u in range(4):
                        byte = (word >> (8 * u)) & 0xFF
                        r = c * 4 + u
                        if byte == 0 or r >= R_:
                            continue
                        out["checked"] += 1
                        if bb != b:
                            out["cross_request"].append((rec["layer"], h, s, j, u))
                            continue
                        if bid not in sel[h, b, r].tolist():
                            out["foreign"].append((rec["layer"], h, b, r, bid))
                            continue
                        if bid in (int(m[b]), int(m[b]) + 1):
                            continue
                        kk, vv, bi_, slot = src
                        dst, rows = slice(slot * W.BS, (slot + 1) * W.BS), slice(bid * W.BS, (bid + 1) * W.BS)
                        ok = (X_._bits_eq(kk[b, dst, h], rec["k_cpu"][b, rows, h])
                              and X_._bits_eq(vv[b, dst, h], rec["v_cpu"][b, rows, h])
                              and X_._bits_eq(bi_[b, dst, h], rec["total_cis"][b, rows, h].to(bi_.dtype)))
                        if not ok:
                            out["bad_contents"].append((rec["layer"], h, b, r, bid))
    return out


# ---------------------------------------------------------------------------
# the speculative runner with the arena and the fork
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _fork_runner(fork=None, X=None, joint_stats=True):
    """spec_world.spec_runner with the cache re-homed into an arena before ragged_engine.attach, rs.kv_arena, and
    SpecOps(joint_attn="fork", fork=...) (the store: the bound for K full rows unless X is given)."""
    orig = SW.spec_runner

    def spec_runner(model, cache, S, horizon, tok0, level="L3", K=SW.K0, mode="exact", gemm="rowgroup",
                    draft_filter=None, neg=None, rb_neg=None, record=True, prefetch=None, after_round=None,
                    journal=RW.CAP, body="torch", redirect=True, pin=False, make_prefetch=None, sr=None):
        SRm = SR if sr is None else sr
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = arena_for(cache, B, X or KA.store_slots(W.TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body=body, graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        if make_prefetch is not None:
            prefetch = make_prefetch(rs)
        sops = SRm.SpecOps(level=level, K=K, mode=mode, gemm=gemm, extra=16, attn=X_.fake_attention_u,
                           journal=journal, redirect=redirect, draft_filter=draft_filter, prefetch=prefetch,
                           rb_neg=rb_neg, neg=neg, record=record, probe_every=0, after_round=after_round, pin=pin,
                           joint_stats=joint_stats, joint_attn="fork", fork=(fork if fork is not None else fake_fork))
        return SRm.SpecRunner(model, rs, sops, BC.HostClock(), t0=tok0)
    SW.spec_runner = spec_runner
    try:
        yield
    finally:
        SW.spec_runner = orig


def run_spec_fork(B, level="L4", gemm="natural", fork=None, X=None, **kw):
    """spec_world.run_vs_shipped(level, mode "union") with the fork runner (the evidence of run_vs_shipped)."""
    with _fork_runner(fork, X):
        return SW.run_vs_shipped(B, level=level, mode="union", gemm=gemm, **kw)


# ---------------------------------------------------------------------------
# Gate 0: joint passes of mode union (joint_attn "store") of a verify_rows source
# ---------------------------------------------------------------------------
def joint_run_union(VRm, S_list, groups, K=3, seed=0, extra=16, gemm="exact"):
    """Joint passes of mode union (the WS-B store attention) of module VRm on a built world (EXACT ragged ops, check
    on), random tokens: returns (the logits of every pass, the world, the read_stats)."""
    R = K + 1
    steps = groups * R
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, S_list)
    s = RE.build(model, W.new_cache(0), S_list, steps, V.exact_ragged_ops(), lambda b, l: rows[(b, l)], pin=False)
    W.poison_built(s, S_list)
    lin = W.row_linear if gemm == "exact" else None
    vs = VRm.attach(model, s, VRm.VerifyOps(K=K, mode="union", gemm=gemm, linear=lin, extra=extra,
                                            attn=X_.fake_attention_u, check=True))
    B = len(S_list)
    g = torch.Generator().manual_seed(seed + 2)
    tokens = torch.randint(0, W.V, (B, steps), generator=g)
    S_t = torch.tensor(S_list, dtype=torch.int64)
    offs = torch.arange(R, dtype=torch.int64)
    out = []
    for grp in range(groups):
        pos = (S_t.view(B, 1) + grp * R + offs.view(1, R)).to(torch.int32)
        out.append(VRm.joint_pass(model, vs, tokens[:, grp * R:(grp + 1) * R], pos))
    return out, s, VRm.read_stats(vs)
