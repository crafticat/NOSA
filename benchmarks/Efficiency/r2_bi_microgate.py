"""R2 step 7: the batch-invariance MICROGATES B1-B5 and the cost tables (docs/R2_STEP7_SPEC_ROUND.md sections 3 and 7.2).
Random bf16 inputs with NOSA-8B's shapes; no model, no workload. One process, one GPU.

GATES (MUST unless marked; "row-equal" = torch.equal of a row's output whatever the other rows hold, computed alone
(one row), in the full batch and in a shuffled batch):
  B1  rmsnorm (flashinfer), nosa_linear (Triton), rope (flashinfer): row-equal at R in {B, 4B}.
  B2  stage 1 (stage1_seqused) on a flattened call: rows share their bank's compressed-key table (cu_seqlens_k
      repeated, seqused_k per row, r2_spec_core.stage1_alias); every row's score row-equal to its call alone.
  B3  GEMMs for wqkv (4608 x 4096), wo (4096 x 4096), gate_up (28672 x 4096), down (4096 x 14336), at M in
      {B, 2B, 3B, 4B} live rows (the tick has B draft rows plus 0 .. U B verify rows; U = 3):
        natural  cuBLAS (F.linear) at M = the live row count (REPORT: the number of rows that differ from the same
                 rows at M = 4B; the M-dependence premise of the design),
        padded   option G-b, r2_spec_core.PaddedLinear: cuBLAS at ONE fixed M_fix = 4B. MUST: every live row equal to
                 its row in the all-live call, with random rows past M and with the live rows in a shuffled order
                 (zero rows past M),
        bi       option G-a, the BI matmul of the bigemm branch (nosi-r2-m2-bigemm), loaded BY PATH (--bigemm,
                 --bigemm-fn; ASSUMED fn(x (M, K), w (N, K)) -> (M, N), F.linear's layout, until its interface is
                 forwarded): row-equal at every M (MUST when given; absent = no BI column).
      COST (REPORT): CUDA-event median over the reps per (shape, B, M, variant); ratios against natural. The padded
      time includes the copy of the live rows into the fixed buffer (the real G-b cost).
  B4  piece 3's BI paged attention (nosi-r2-m2-biattn 4c1243f, bi_paged_attention.bi_paged_decode, loaded BY PATH:
      --biattn): R = B x (1 + U) rows, U + 1 rows of one request at consecutive positions in its tail page, each row
      63 historical pages (the request's base selection, 6 pages swapped per verify row) + the tail page. row-equal
      (MUST); close to the INDEPENDENT torch twin
      r2_spec_core.paged_attention_reference (max |d| <= 0.02, MUST: the two implementations agree on the contract,
      bias semantics included: logit = (q.k + bias) * scale); piece 3's float64 attention_oracle (REPORT).
      COST (REPORT): (a) the same R rows through FA2-NOSA flash_attn_with_kvcache (each row its 64 pages copied
      into a PRIVATE contiguous 4096-row window, the bias as kv_bias; so FA2 gets no L2 reuse between the rows of one
      request, piece 3 may) at num_splits 4 and 1; (b) THE SG5 ATTENTION TERM:
      BI over B x (1 + u) rows, 1 + u in {1, 2, U + 1}, against FA2-NOSA over the B draft rows only at num_splits 4
      and 0 (0 = the shipped default heuristic), next to piece 3's registered ratios.
  B5  the deterministic selection r2_spec_core.select_blocks on the GPU: row-equal and equal to the CPU result (MUST).

OUTPUT: --out/summary.json (gates, cost tables, environment); one GATE / REPORT line per gate on stdout; exit = the
number of failed MUST gates, 2 = refused (--out exists).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback

SHAPES = dict(wqkv=(4608, 4096), wo=(4096, 4096), gate_up=(28672, 4096), down=(4096, 14336))
# piece 3's registered time ratios (the coordinator's message of 2026-10-20): BI over B x (1 + u) rows against
# FA2-NOSA split 4 over the B rows, keyed by (B, 1 + u)
PIECE3_REGISTERED = {(64, 1): 1.4, (64, 2): 1.9, (64, 4): 2.8, (128, 1): 1.1, (128, 2): 1.6, (128, 4): 2.4}


def log(msg):
    print("[r2_bi %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


def load_by_path(path, name):
    """A module of another branch, copied into the step record by the step script (pinned by blob id)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def timed(fn, warm, reps):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        xs.append(e0.elapsed_time(e1))
    return statistics.median(xs)


def row_equal(f, R, rows, gen):
    """f(idx) computes the outputs of the rows idx (a list), in that order, as ONE call. Returns the rows (of `rows`)
    whose output differs between the full batch, the row alone and a shuffled batch."""
    import torch
    full = f(list(range(R)))
    perm = torch.randperm(R, generator=gen).tolist()
    shuf = f(perm)
    pos = {r: j for j, r in enumerate(perm)}
    bad = []
    for r in rows:
        alone = f([r])
        if not (torch.equal(full[r], alone[0]) and torch.equal(full[r], shuf[pos[r]])):
            bad.append(int(r))
    return bad


def b1(B, gen):
    import torch
    from flashinfer.norm import rmsnorm
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    from nosi.nosa_linear import nosa_linear
    dev = "cuda"
    out = {}
    w = torch.randn(4096, generator=gen).to(dev, torch.bfloat16)
    delta = (torch.randn(2, 256, generator=gen) * 0.05).to(dev, torch.bfloat16)
    A = torch.rand(2, generator=gen).to(dev, torch.bfloat16)
    cs = torch.randn(140000, 128, generator=gen).to(dev, torch.float32)
    for R in (B, 4 * B):
        rows = list(range(0, R, max(1, R // 8)))
        x = torch.randn(R, 4096, generator=gen).to(dev, torch.bfloat16)
        out["rmsnorm_R%d" % R] = row_equal(lambda idx: rmsnorm(x[idx].contiguous(), w, 1e-5), R, rows, gen)
        qkv = torch.randn(R, 1, 4608, generator=gen).to(dev, torch.bfloat16)

        def nl(idx):
            q, k, v, c = nosa_linear(qkv[idx].contiguous(), delta, A, 4096, 256, 2)
            return torch.cat([q.flatten(1), k.flatten(1), v.flatten(1), c.flatten(1)], 1)
        out["nosa_linear_R%d" % R] = row_equal(nl, R, rows, gen)
        pos = torch.randint(0, 130000, (R,), generator=gen).to(dev, torch.int32)
        qq = torch.randn(R, 4096, generator=gen).to(dev, torch.bfloat16)
        kk = torch.randn(R, 256, generator=gen).to(dev, torch.bfloat16)

        def rope(idx):
            q, k = qq[idx].contiguous(), kk[idx].contiguous()
            apply_rope_with_cos_sin_cache_inplace(pos[idx].contiguous(), q, k, 128, cs, True)
            return torch.cat([q, k], 1)
        out["rope_R%d" % R] = row_equal(rope, R, rows, gen)
    bad = {k: v for k, v in out.items() if v}
    return gate("B1_row_invariant_norm_linear_rope", not bad, True, json.dumps(bad) if bad else json.dumps(sorted(out)))


def b2(B, gen):
    import torch
    from nosi.r2_spec_core import stage1_alias
    from nosi.stage1_seqused import stage1_seqused
    dev = "cuda"
    NB, Lk = 2 * B, 1041
    ck = torch.randn(NB * Lk, 2, 128, generator=gen).to(dev, torch.bfloat16)
    R = 4 * B
    bank = [int(x) for x in torch.randint(0, NB, (R,), generator=gen)]
    n_used = [int(x) for x in torch.randint(900, Lk + 1, (R,), generator=gen)]
    q = torch.randn(R, 32, 128, generator=gen).to(dev, torch.bfloat16)

    def call(idx):
        cu, sq = stage1_alias([bank[i] for i in idx], [n_used[i] for i in idx], Lk, NB)
        cq = torch.arange(len(idx) + 1, dtype=torch.int32, device=dev)
        return stage1_seqused(q[idx], ck, ck, cq, cu.to(dev), 1, Lk, seqused_k=sq.to(dev))
    full = call(list(range(R)))
    perm = torch.randperm(R, generator=gen).tolist()
    shuf = call(perm)
    bad = []
    for j, r in enumerate(perm):
        if not torch.equal(shuf[:, j], full[:, r]):
            bad.append(r)
    for r in range(0, R, max(1, R // 16)):
        if not torch.equal(call([r])[:, 0], full[:, r]):
            bad.append(r)
    return gate("B2_row_invariant_stage1_alias", not bad, True,
                "R %d rows over %d banks, Lk %d, seqused 900..%d; differing rows %s" % (R, NB, Lk, Lk, sorted(set(bad))[:10]))


def b3(Bs, gen, warm, reps, bigemm=None):
    import torch
    import torch.nn.functional as F
    from nosi.r2_spec_core import PaddedLinear
    dev = "cuda"
    inv, cost = {}, []
    for name, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, generator=gen) * 0.02).to(dev, torch.bfloat16)
        for B in Bs:
            Mfix = 4 * B
            x = torch.randn(Mfix, K, generator=gen).to(dev, torch.bfloat16)
            garbage = torch.randn(Mfix, K, generator=gen).to(dev, torch.bfloat16)
            y_nat_full = F.linear(x, W)
            pl = PaddedLinear(Mfix)
            buf = pl.buffer(K, torch.bfloat16, dev)
            y_pad_full = pl(x, W).clone()                                   # all M_fix rows live
            y_bi_full = bigemm(x, W) if bigemm is not None else None
            for M in (B, 2 * B, 3 * B, 4 * B):
                y_nat = F.linear(x[:M], W)
                nat_bad = int((~(y_nat == y_nat_full[:M]).all(1)).sum())
                buf.copy_(garbage)                                          # random rows past M
                y1 = pl(x[:M], W).clone()
                buf.zero_()                                                 # zero rows past M, live rows shuffled
                perm = torch.randperm(M, generator=gen).to(dev)
                y2 = pl(x[:M][perm], W).clone()
                ok = (y1 == y_pad_full[:M]).all(1)
                ok2 = torch.empty_like(ok)
                ok2[perm] = (y2 == y_pad_full[:M][perm]).all(1)
                pad_bad = int((~(ok & ok2)).sum())
                row = dict(natural_rows_differing=nat_bad, padded_rows_differing=pad_bad)
                if bigemm is not None:
                    row["bi_rows_differing"] = int((~(bigemm(x[:M], W) == y_bi_full[:M]).all(1)).sum())
                inv["%s_B%d_M%d" % (name, B, M)] = row
                xm = x[:M].contiguous()
                t_nat = timed(lambda: F.linear(xm, W), warm, reps)
                t_pad = timed(lambda: pl(xm, W), warm, reps)
                c = dict(shape=name, N=N, K=K, B=B, M=M, M_fix=Mfix, natural_ms=round(t_nat, 4), padded_ms=round(t_pad, 4),
                         padded_ratio=round(t_pad / t_nat, 3))
                msg = "B3 %s B %d M %d: natural %.3f ms, padded %.3f (x%.2f)" % (name, B, M, t_nat, t_pad, t_pad / t_nat)
                if bigemm is not None:
                    t_bi = timed(lambda: bigemm(xm, W), warm, reps)
                    c.update(bi_ms=round(t_bi, 4), bi_ratio=round(t_bi / t_nat, 3))
                    msg += ", bi %.3f (x%.2f)" % (t_bi, t_bi / t_nat)
                cost.append(c)
                log(msg + "; rows differing %s" % json.dumps(row))
            del pl, buf, garbage
        del W
        torch.cuda.empty_cache()
    gates = [gate("B3_padded_row_invariant", all(v["padded_rows_differing"] == 0 for v in inv.values()), True,
                  json.dumps({k: v["padded_rows_differing"] for k, v in inv.items() if v["padded_rows_differing"]}))]
    if bigemm is not None:
        gates.append(gate("B3_bi_row_invariant", all(v["bi_rows_differing"] == 0 for v in inv.values()), True,
                          json.dumps({k: v["bi_rows_differing"] for k, v in inv.items() if v["bi_rows_differing"]})))
    else:
        gates.append(gate("R_B3_bi_not_run", True, False, "no --bigemm: the BI GEMM pins of the step are unset"))
    gates.append(gate("R_B3_natural_M_dependence", True, False,
                      "rows of cuBLAS at the natural M that differ from the same rows at M = 4B: "
                      + json.dumps({k: v["natural_rows_differing"] for k, v in inv.items()})))
    return gates, cost, inv


def make_attention_case(B, U, S_slots, gen, dev="cuda", n_swap=6):
    """Rows: for each request b, U + 1 rows at consecutive positions in its tail page (row b * (U + 1) + u). Per
    (request, head) one base selection of 63 distinct historical slots; row u = 0 (the draft row) lists the base, each
    row u > 0 swaps n_swap of its pages for others (consecutive positions select nearly the same blocks), so the rows of
    one request share most pages, as in a tick. Every list is ascending (canonical) and ends with the tail page (slot
    S_slots - 1), cut at the row's own last_len."""
    import torch
    Hq, Hk, D = 32, 2, 128
    k = torch.randn(B, S_slots * 64, Hk, D, generator=gen).to(dev, torch.bfloat16)
    v = torch.randn(B, S_slots * 64, Hk, D, generator=gen).to(dev, torch.bfloat16)
    bias = (torch.rand(B, S_slots * 64, Hk, generator=gen) * 2 - 1).to(dev, torch.bfloat16)
    R = B * (U + 1)
    req = torch.arange(B).repeat_interleave(U + 1)
    pages = torch.empty(R, Hk, 64, dtype=torch.int32)
    for b in range(B):
        for h in range(Hk):
            perm = torch.randperm(S_slots - 1, generator=gen)
            base, spare = perm[:63], perm[63:]
            for u in range(U + 1):
                sel = base.clone()
                if u > 0 and n_swap > 0:
                    sel[torch.randperm(63, generator=gen)[:n_swap]] = spare[torch.randperm(spare.numel(), generator=gen)[:n_swap]]
                pages[b * (U + 1) + u, h, :63] = torch.sort(sel).values.to(torch.int32)
                pages[b * (U + 1) + u, h, 63] = S_slots - 1
    t0 = int(torch.randint(1, 60 - U, (1,), generator=gen))
    last = torch.tensor([t0 + (r % (U + 1)) + 1 for r in range(R)], dtype=torch.int32)
    q = torch.randn(R, Hq, D, generator=gen).to(dev, torch.bfloat16)
    return dict(q=q, k=k, v=v, bias=bias, req=req.to(dev, torch.int32), pages=pages.to(dev).contiguous(),
                n_pages=torch.full((R, Hk), 64, dtype=torch.int32, device=dev), last=last.to(dev), R=R)


def b4(Bs, U, gen, warm, reps, attn):
    import torch
    from flash_attn_nosa import flash_attn_with_kvcache
    from nosi.r2_spec_core import paged_attention_reference
    bpd = attn.bi_paged_decode
    gates, cost, budget = [], [], []
    for B in Bs:
        c = make_attention_case(B, U, 80, gen)
        R = c["R"]

        def run(idx):
            idx_t = torch.tensor(idx, device="cuda")
            return bpd(c["q"][idx_t].contiguous(), c["k"], c["v"], c["bias"], c["req"][idx_t].contiguous(),
                       c["pages"][idx_t].contiguous(), c["n_pages"][idx_t].contiguous(), c["last"][idx_t].contiguous())
        full = run(list(range(R)))
        perm = torch.randperm(R, generator=gen).tolist()
        shuf = run(perm)
        bad = [r for j, r in enumerate(perm) if not torch.equal(shuf[j], full[r])]
        bad += [r for r in range(0, R, max(1, R // 16)) if not torch.equal(run([r])[0], full[r])]
        gates.append(gate("B4_bi_attention_row_invariant_B%d" % B, not bad, True, "R %d rows; differing %s" % (R, sorted(set(bad))[:10])))
        sub = list(range(0, R, max(1, R // 8)))
        sub_t = torch.tensor(sub, device="cuda")
        args_sub = (c["q"][sub_t].contiguous(), c["k"], c["v"], c["bias"], c["req"][sub_t].contiguous(),
                    c["pages"][sub_t].contiguous(), c["n_pages"][sub_t].contiguous(), c["last"][sub_t].contiguous())
        ref = paged_attention_reference(*[a.cpu() for a in args_sub], 128 ** -0.5)
        d = float((full[sub].float().cpu() - ref.float()).abs().max())
        gates.append(gate("B4_bi_attention_vs_independent_twin_B%d" % B, d <= 0.02, True,
                          "max |piece 3 - r2_spec_core twin| %.5f over %d rows" % (d, len(sub))))
        orc = attn.attention_oracle(*args_sub)
        d_or = float((full[sub].double() - orc).abs().max())
        gates.append(gate("R_B4_bi_attention_vs_oracle_B%d" % B, True, False, "max |piece 3 - float64 oracle| %.5f" % d_or))
        # (a) the same R rows through FA2-NOSA: each row attends a contiguous 4096-row window (its 64 pages in order)
        win_k = torch.empty(R, 4096, 2, 128, device="cuda", dtype=torch.bfloat16)
        win_v = torch.empty_like(win_k)
        win_b = torch.empty(R, 4096, 2, device="cuda", dtype=torch.bfloat16)
        rows = (c["pages"].long().unsqueeze(-1) * 64 + torch.arange(64, device="cuda")).reshape(R, 2, 4096).permute(0, 2, 1)
        bidx = c["req"].long().view(R, 1, 1)
        hidx = torch.arange(2, device="cuda").view(1, 1, 2)
        win_k.copy_(c["k"][bidx, rows, hidx])
        win_v.copy_(c["v"][bidx, rows, hidx])
        win_b.copy_(c["bias"][bidx, rows, hidx])
        seqlens = (63 * 64 + c["last"]).to(torch.int32)
        q4 = c["q"].unsqueeze(1).contiguous()
        fa = {}
        for splits in (4, 1):
            fa[splits] = timed(lambda: flash_attn_with_kvcache(q4, win_k, win_v, win_b, cache_seqlens=seqlens, num_splits=splits),
                               warm, reps)
        o_fa = flash_attn_with_kvcache(q4, win_k, win_v, win_b, cache_seqlens=seqlens, num_splits=1)[:, 0]
        t_bi = timed(lambda: bpd(c["q"], c["k"], c["v"], c["bias"], c["req"], c["pages"], c["n_pages"], c["last"]), warm, reps)
        dfa = float((o_fa.float() - full.float()).abs().max())
        cost.append(dict(B=B, U=U, R=R, bi_ms=round(t_bi, 4), fa2_split4_ms=round(fa[4], 4), fa2_split1_ms=round(fa[1], 4),
                         bi_ratio_vs_split4=round(t_bi / fa[4], 3), bi_ratio_vs_split1=round(t_bi / fa[1], 3),
                         max_abs_bi_vs_fa2_split1=round(dfa, 5)))
        log("B4 B %d R %d: bi %.3f ms, fa2 split4 %.3f, split1 %.3f; max |bi - fa2| %.4f" % (B, R, t_bi, fa[4], fa[1], dfa))
        # (b) the SG5 attention term: BI over B x (1 + u) rows against FA2-NOSA over the B draft rows (u = 0)
        d_idx = torch.arange(0, R, U + 1, device="cuda")
        qd, wkd, wvd = q4[d_idx].contiguous(), win_k[d_idx].contiguous(), win_v[d_idx].contiguous()
        wbd, sd = win_b[d_idx].contiguous(), seqlens[d_idx].contiguous()
        fa_b = {s: timed(lambda: flash_attn_with_kvcache(qd, wkd, wvd, wbd, cache_seqlens=sd, num_splits=s), warm, reps)
                for s in (4, 0)}
        for n1 in sorted({1, 2, U + 1}):
            if n1 > U + 1:
                continue
            sel = torch.nonzero(torch.arange(R, device="cuda") % (U + 1) < n1).flatten()
            a_n = (c["q"][sel].contiguous(), c["k"], c["v"], c["bias"], c["req"][sel].contiguous(),
                   c["pages"][sel].contiguous(), c["n_pages"][sel].contiguous(), c["last"][sel].contiguous())
            t_n = timed(lambda: bpd(*a_n), warm, reps)
            budget.append(dict(B=B, rows_per_request=n1, R=int(sel.numel()), bi_ms=round(t_n, 4),
                               fa2_B_rows_split4_ms=round(fa_b[4], 4), fa2_B_rows_split0_ms=round(fa_b[0], 4),
                               ratio_vs_split4=round(t_n / fa_b[4], 3), ratio_vs_split0=round(t_n / fa_b[0], 3),
                               piece3_registered_vs_split4=PIECE3_REGISTERED.get((B, n1))))
            log("B4 budget B %d 1+u %d: bi %.3f ms over %d rows; fa2 over %d rows split4 %.3f / split0 %.3f; x%.2f (piece 3 "
                "registered %s)" % (B, n1, t_n, sel.numel(), B, fa_b[4], fa_b[0], t_n / fa_b[4], PIECE3_REGISTERED.get((B, n1))))
        del win_k, win_v, win_b, wkd, wvd, wbd, c
        torch.cuda.empty_cache()
    return gates, cost, budget


def b5(B, gen):
    import torch
    from nosi.r2_spec_core import select_blocks
    R, M = 4 * B, 261
    pq = torch.randn(2, R, M, generator=gen).to(torch.bfloat16)
    pc = torch.randn(2, R, M, generator=gen).to(torch.bfloat16)
    pq[:, :, :3] = float("inf")
    pq[:, :, -17:] = float("inf")
    pc[:, :, 100:140] = pc[:, :, 100:101]                            # ties at the cut
    cpu = select_blocks(pq.float(), pc.float(), 33, 64)
    gpu = select_blocks(pq.float().cuda(), pc.float().cuda(), 33, 64)
    bad = [] if torch.equal(gpu.cpu(), cpu) else ["gpu_vs_cpu"]
    for r in range(0, R, max(1, R // 8)):
        alone = select_blocks(pq[:, r:r + 1].float().cuda(), pc[:, r:r + 1].float().cuda(), 33, 64)
        if not torch.equal(alone[:, 0].cpu(), cpu[:, r]):
            bad.append(r)
    return gate("B5_deterministic_selection", not bad, True, "R %d, ties at the cut; differing %s" % (R, bad[:10]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="64,128")
    ap.add_argument("--U", type=int, default=3)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--biattn", default=None, help="piece 3's bi_paged_attention.py (a pinned copy)")
    ap.add_argument("--bigemm", default=None, help="the BI GEMM module of the bigemm branch (a pinned copy)")
    ap.add_argument("--bigemm-fn", default=None, help="its function: fn(x (M, K), w (N, K)) -> (M, N)")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if (a.bigemm is None) != (a.bigemm_fn is None):
        print("REFUSED: --bigemm and --bigemm-fn go together", file=sys.stderr)
        return 2
    import torch
    os.makedirs(a.out)
    Bs = [int(x) for x in a.batches.split(",") if x]
    gen = torch.Generator().manual_seed(a.seed)
    gates, tables = [], {}
    for name, fn in (("B1", lambda: [b1(Bs[0], gen)]), ("B2", lambda: [b2(Bs[0], gen)]), ("B5", lambda: [b5(Bs[0], gen)])):
        try:
            gates += fn()
        except Exception as e:
            traceback.print_exc()
            gates.append(gate(name, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    try:
        bigemm = getattr(load_by_path(a.bigemm, "r2_bigemm_by_path"), a.bigemm_fn) if a.bigemm else None
        g3, tables["gemm_cost"], tables["gemm_invariance"] = b3(Bs, gen, a.warm, a.reps, bigemm)
        gates += g3
    except Exception as e:
        traceback.print_exc()
        gates.append(gate("B3", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    try:
        if not a.biattn:
            raise ValueError("no --biattn (piece 3's module)")
        attn = load_by_path(a.biattn, "piece3_bi_paged_attention")
        g4, tables["attention_cost"], tables["attention_sg5_term"] = b4(Bs, a.U, gen, a.warm, a.reps, attn)
        gates += g4
    except Exception as e:
        traceback.print_exc()
        gates.append(gate("B4", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    meta = dict(batches=Bs, U=a.U, warm=a.warm, reps=a.reps, seed=a.seed, shapes=SHAPES, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), commit=os.environ.get("BI_COMMIT", "unknown"), biattn=a.biattn,
                bigemm=a.bigemm, bigemm_fn=a.bigemm_fn, piece3_registered={"%d_%d" % k: v for k, v in PIECE3_REGISTERED.items()})
    json.dump(dict(meta=meta, gates=gates, tables=tables), open(os.path.join(a.out, "summary.json"), "w"), indent=1,
              sort_keys=True, default=str)
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL", g["detail"][:400]))
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("r2_bi_microgate: %d MUST gate(s) failed" % nfail)
    return min(nfail, 125)


if __name__ == "__main__":
    sys.exit(main())
