"""R2 step 7: the batch-invariance MICROGATES B1-B5 and the cost tables (docs/R2_STEP7_SPEC_ROUND.md sections 3 and 7.2).
Random bf16 inputs with NOSA-8B's shapes; no model, no workload. One process, one GPU.

GATES (MUST unless marked; "row-equal" = torch.equal of a row's output whatever the other rows hold, computed alone
(one row), in the full batch and in a shuffled batch):
  B1  rmsnorm (flashinfer), nosa_linear (Triton), rope (flashinfer): row-equal at R in {B, 4B}.
  B2  stage 1 (stage1_seqused) on a flattened call: rows share their bank's compressed-key table (cu_seqlens_k
      repeated, seqused_k per row, r2_spec_core.stage1_alias); every row's score row-equal to its call alone.
  B3  GEMMs for wqkv (4608 x 4096), wo (4096 x 4096), gate_up (32768 x 4096), down (4096 x 16384) (NOSA-8B:
      SHAPES = shapes_from_config(models/NOSA-8B/config.json), the CPU test S15 checks it), at M in
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

--gemm-scan (r2-mbase cell gm1; paper-notes/reconciliation-17pct.md row 7, "GEMM above M 320"): ONLY the GEMM cost
scan, no B1-B5. The five decode linears (SHAPES + LM_HEAD, NOSA-8B) with random bf16 weights, CUDA-event medians
(timed, --warm / --reps):
  natural  cuBLAS F.linear at M live rows, for M in --ms (default SCAN_MS: 64 .. 1148) and every live count of the
           EXACT pairs below;
  padded   r2_spec_core.PaddedLinear(m_fix) called with M live rows (the copy into the fixed buffer included), for
           (M, M) with M in --ms (all rows live) and the EXACT tick's pairs (u B, 4 B), u = 1 .. 4, B in --exact-b
           (default 287, 216, 64): the tick's EXACT mode pads EVERY call to M_fix = 4 B (REG "step 7 budget").
Per-step totals: 32 x (wqkv + wo + gate_up + down) + 1 x lm_head (the old M-scan's projection, 2175525). The dense
verify term of the +17 % model at B 287, K 3: FAST = 3 natural(287) + natural(861); EXACT = 3 padded(287, 1148) +
padded(861, 1148). MUST: GM-RAN (every (shape, M) timed, finite and > 0). REPORT: the padded rows' invariance at the
largest pair (garbage vs zero past the live rows). PREDICTIONS: PRED_GM1 (natural per-step at each M; the old M-scan
curve x the eager factor), PRED_GM1_RATIO (EXACT / natural at B 287), PRED_GM1_DENSE (FAST / EXACT dense vs HBM 213.90).
--device cpu and --shape-div N exist for the CPU test only (host timer, shapes / N); the meta records both.

--gemm-scan --bigemm-c {strict,bucket,both} (r2-bigemmc cell bigc-time, the gm1 protocol): adds G-c columns
(nosi.bi_linear_c; --bigc-table / --bigc-table-sha: a frozen table file, else the source TABLE; its digest is in the
meta) timed at every natural M: strict = THE CANDIDATE (one launch per weight shape); bucket = the RESEARCH ARM
(launch by M bucket; the relaxed law is NOT approved; its verdicts are labelled research). Per-step totals as above.
REGISTERED PREDICTIONS (forecast costs, not measured speedups; design note 5.3): G-c / natural per step at M 287 and
861: point 1.15, band [1.00, 1.35]; M 64: point 1.35, band [1.00, 1.70]; dense' = 3 G-c(287) + G-c(861): point 178 ms,
band [155, 210]. FALSIFIER: dense' >= 213.904 ms (the HBM term; then exactness costs Q). MUST (GPU, shape-div 1 only):
GC-NAT-REPEATS-GM1, the natural arm within 5 % of gm1's measured per-step totals (run 2190876) at every M both have;
else the cell does not compare.
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

# (N, K) of F.linear's weight for NOSA-8B's decode linears (nosa_llama.py: wqkv, wo, gate_up_proj = cat(gate, up),
# down_proj). The values follow /mnt/beegfs/ojerbi/models/NOSA-8B/config.json (hidden 4096, 32 heads, 2 KV heads,
# head_dim 128, intermediate 16384) through shapes_from_config; test S15 keeps them equal. They are also the values
# of the bigemm branch's bi_linear.NOSA8B_SHAPES (without lm_head).
SHAPES = dict(wqkv=(4608, 4096), wo=(4096, 4096), gate_up=(32768, 4096), down=(4096, 16384))


def shapes_from_config(cfg: dict) -> dict:
    """The four decode GEMM shapes (N, K) from a HF config dict."""
    hidden, heads, kv_heads = int(cfg["hidden_size"]), int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    inter = int(cfg["intermediate_size"])
    return dict(wqkv=(heads * head_dim + 2 * kv_heads * head_dim, hidden), wo=(hidden, heads * head_dim),
                gate_up=(2 * inter, hidden), down=(hidden, inter))
# piece 3's registered time ratios (the coordinator's message of 2026-10-20): BI over B x (1 + u) rows against
# FA2-NOSA split 4 over the B rows, keyed by (B, 1 + u)
PIECE3_REGISTERED = {(64, 1): 1.4, (64, 2): 1.9, (64, 4): 2.8, (128, 1): 1.1, (128, 2): 1.6, (128, 4): 2.4}

# --gemm-scan (gm1). lm_head (vocab_size, hidden_size) of /mnt/beegfs/ojerbi/models/NOSA-8B/config.json (test GM1).
LM_HEAD = (73448, 4096)
LAYERS = 32
SCAN_MS = (64, 128, 256, 320, 512, 768, 861, 1024, 1148)
SCAN_EXACT_B = (287, 216, 64)
SCAN_U = 3
# The old M-scan (retroinfer-eval docs/evidence/rows_bench_2175525/gemm_mscan.md, nova07, graph timing with an L2
# flush): per-step ms. bigemm_v1 (2188818, this GPU f31fc723, eager CUDA events as here) measured cuBLAS natural
# 16.13 / 22.00 / 28.31 at M 128 / 256 / 320: 1.106 / 1.098 / 1.112 x the old curve -> EAGER_FACTOR 1.105.
OLD_MSCAN = {128: 14.59, 256: 20.03, 320: 25.46, 384: 27.26, 512: 34.62, 640: 42.93, 672: 51.68, 768: 52.07}
EAGER_FACTOR = 1.105
# natural per-step (point, low, high), ms. 64: bigemm_v1's 14.46. 128 .. 768: OLD x 1.105, +-10 %. Past 768
# (extrapolated by 256-row tiles, about 17.45 ms per tile from 512 -> 768): 861 between 768 and 1024, 1024 = 4 tiles,
# 1148 = 4.5-5 tiles; wider bands.
PRED_GM1 = {64: (14.46, 13.0, 16.0), 128: (16.12, 14.5, 17.8), 256: (22.13, 19.9, 24.4), 320: (28.13, 25.3, 31.0),
            512: (38.26, 34.4, 42.1), 768: (57.54, 51.8, 63.3), 861: (67.2, 57.5, 77.0), 1024: (76.8, 65.0, 88.0),
            1148: (96.1, 77.0, 106.0)}
# EXACT / natural per-step at B 287 (live rows u x 287, m_fix 1148): (point, low, high)
PRED_GM1_RATIO = {287: (3.8, 2.8, 4.6), 861: (1.43, 1.1, 1.8)}
# the dense verify term (ms) at B 287, K 3 against the model's HBM term 213.90 ms (FC:26, recon row 7):
PRED_GM1_DENSE = dict(fast=(142.5, 120.0, 165.0), exact=(385.0, 300.0, 430.0), hbm_ms=213.90, model_dense_ms=126.34)

# --bigemm-c (bigc-time). gm1's MEASURED natural per-step totals, ms (run 2190876, r2-runs/bi/bi_gma_2190876_gm1;
# kstep-cpu-runs/analysis/mbase/SCORING_gm1_mb1.md): the natural arm must repeat them within GM1_REPEAT_TOL.
GM1_NATURAL = {64: 14.225, 128: 15.849, 256: 21.791, 287: 27.689, 320: 28.214, 512: 38.119, 768: 56.132, 861: 72.006,
               1024: 77.023, 1148: 82.882}
GM1_REPEAT_TOL = 0.05
BIGC_VARIANTS = ("strict", "bucket")
# REGISTERED (design note 5.3; forecast costs): G-c / natural per step (point, low, high), and dense' (ms)
PRED_BIGC_RATIO = {64: (1.35, 1.00, 1.70), 287: (1.15, 1.00, 1.35), 861: (1.15, 1.00, 1.35)}
PRED_BIGC_DENSE = dict(point=178.0, low=155.0, high=210.0, falsifier_ms=213.904)


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


# ------------------------------------------------------------------ --gemm-scan (gm1)
def lm_head_from_config(cfg: dict):
    """(N, K) of lm_head from a HF config dict."""
    return (int(cfg["vocab_size"]), int(cfg["hidden_size"]))


def scan_shapes(div: int = 1) -> dict:
    """The five linears (SHAPES + lm_head), every dimension divided by div (the CPU test only)."""
    out = dict(SHAPES, lm_head=LM_HEAD)
    return {k: (n // div, kk // div) for k, (n, kk) in out.items()}


def scan_plan(ms, exact_bs, U: int = SCAN_U):
    """(natural live counts, padded (live, m_fix) pairs), both sorted and without repeats: natural at every M of ms and
    every live count u B (u = 1 .. U + 1) of the EXACT pairs; padded (M, M) for M in ms and (u B, (U + 1) B)."""
    ms, bs = [int(m) for m in ms], [int(b) for b in exact_bs]
    if not ms or min(ms) < 1 or any(b < 1 for b in bs):
        raise ValueError("--ms and --exact-b need positive row counts")
    nat = set(ms)
    pad = {(m, m) for m in ms}
    for b in bs:
        for u in range(1, U + 2):
            nat.add(u * b)
            pad.add((u * b, (U + 1) * b))
    return sorted(nat), sorted(pad)


def per_step_totals(rows, key: str, layers: int = LAYERS):
    """{group: layers x (wqkv + wo + gate_up + down) + lm_head} over the rows (dicts with 'shape', the group key and
    'ms'). Refuses a group that lacks one of the five shapes or holds one twice."""
    need = set(SHAPES) | {"lm_head"}
    acc = {}
    for r in rows:
        g = acc.setdefault(r[key] if not isinstance(r[key], list) else tuple(r[key]), {})
        if r["shape"] in g:
            raise ValueError("group %s: shape %s twice" % (r[key], r["shape"]))
        g[r["shape"]] = float(r["ms"])
    out = {}
    for g, d in acc.items():
        if set(d) != need:
            raise ValueError("group %s: shapes %s, need %s" % (g, sorted(d), sorted(need)))
        out[g] = sum(v * (1 if s == "lm_head" else layers) for s, v in d.items())
    return out


def gemm_scan(ms, exact_bs, gen, warm, reps, U: int = SCAN_U, div: int = 1, dev: str = "cuda", timer=None,
              bigc=None, bigc_rows=None):
    """The gm1 timing scan. timer(fn) -> ms (default: the CUDA-event median timed(fn, warm, reps)). Returns the
    natural rows, the padded rows and the padded-invariance report at the largest pair. [bigemmc] bigc {variant: fn}:
    each G-c variant timed at every natural M on the same W and x, rows appended to bigc_rows."""
    import torch
    import torch.nn.functional as F
    from nosi.r2_spec_core import PaddedLinear
    timer = timer or (lambda fn: timed(fn, warm, reps))
    nat_ms, pad_pairs = scan_plan(ms, exact_bs, U)
    rows_max = max(max(nat_ms), max(f for _, f in pad_pairs))
    natural, padded, inv = [], [], {}
    big = max(pad_pairs, key=lambda p: (p[1], -p[0]))                 # the largest m_fix, its smallest live count
    for name, (N, K) in scan_shapes(div).items():
        W = (torch.randn(N, K, generator=gen) * 0.02).to(dev, torch.bfloat16)
        x = torch.randn(rows_max, K, generator=gen).to(dev, torch.bfloat16)
        for M in nat_ms:
            xm = x[:M].contiguous()
            t = timer(lambda: F.linear(xm, W))
            natural.append(dict(shape=name, N=N, K=K, M=M, ms=round(t, 5)))
            for v, fn in (bigc or {}).items():                         # [bigemmc] the same xm and W
                t = timer(lambda: fn(xm, W))
                bigc_rows.append(dict(shape=name, N=N, K=K, M=M, variant=v, ms=round(t, 5)))
        pls = {}
        for live, mfix in pad_pairs:
            pl = pls.setdefault(mfix, PaddedLinear(mfix))
            xm = x[:live].contiguous()
            t = timer(lambda: pl(xm, W))
            padded.append(dict(shape=name, N=N, K=K, M=live, M_fix=mfix, pair=[live, mfix], ms=round(t, 5)))
        live, mfix = big
        pl = pls[mfix]
        buf = pl.buffer(K, torch.bfloat16, dev)
        full = pl(x[:mfix], W).clone()
        buf.copy_(torch.randn(mfix, K, generator=gen).to(dev, torch.bfloat16))
        y1 = pl(x[:live], W).clone()
        buf.zero_()
        y2 = pl(x[:live], W).clone()
        inv[name] = dict(live=live, m_fix=mfix, rows_differing_garbage=int((~(y1 == full[:live]).all(1)).sum()),
                         rows_differing_zero=int((~(y2 == full[:live]).all(1)).sum()))
        log("gm1 %s (%d x %d): natural %s; padded %s" % (name, N, K, [r["ms"] for r in natural if r["shape"] == name],
                                                         [r["ms"] for r in padded if r["shape"] == name]))
        del W, x, pls, buf
        if dev == "cuda":
            torch.cuda.empty_cache()
    return natural, padded, inv


def gm1_verdicts(nat_step, pad_step, U: int = SCAN_U):
    """The registered gm1 predictions against the per-step totals (nat_step {M: ms}, pad_step {(live, m_fix): ms}):
    [(name, verdict, detail)]; REPORT where a needed total is absent."""
    out = []
    for M, (pt, lo, hi) in sorted(PRED_GM1.items()):
        v = nat_step.get(M)
        out.append(("GM-NAT-%d" % M, "REPORT" if v is None else ("CONFIRMED" if lo <= v <= hi else "REFUTED"),
                    "cuBLAS natural per step %s ms at M %d (point %s, band [%s, %s])" % (
                        "n/a" if v is None else "%.3f" % v, M, pt, lo, hi)))
    for B in SCAN_EXACT_B:
        mfix = (U + 1) * B
        for u in range(1, U + 2):
            n, p = nat_step.get(u * B), pad_step.get((u * B, mfix))
            if n is None or p is None:
                continue
            reg = PRED_GM1_RATIO.get(u * B) if B == 287 else None
            r = p / n
            v = "REPORT" if reg is None else ("CONFIRMED" if reg[1] <= r <= reg[2] else "REFUTED")
            out.append(("GM-EXACT-%d-%d" % (u * B, mfix), v, "EXACT (live %d, M_fix %d) %.3f ms / natural %.3f ms = "
                        "%.3f%s" % (u * B, mfix, p, n, r, "" if reg is None else " (point %s, band [%s, %s])" % reg)))
    B, K = 287, U
    mfix = (U + 1) * B
    need_n, need_p = (B, K * B), ((B, mfix), (K * B, mfix))
    hbm = PRED_GM1_DENSE["hbm_ms"]
    if all(m in nat_step for m in need_n):
        fast = 3 * nat_step[B] + nat_step[K * B]
        pt, lo, hi = PRED_GM1_DENSE["fast"]
        out.append(("GM-DENSE-FAST", "CONFIRMED" if lo <= fast <= hi else "REFUTED",
                    "3 x natural(%d) + natural(%d) = %.2f ms (point %s, band [%s, %s]; model %s; HBM %s: %s)" % (
                        B, K * B, fast, pt, lo, hi, PRED_GM1_DENSE["model_dense_ms"], hbm,
                        "BINDING" if fast > hbm else "not binding")))
    if all(p in pad_step for p in need_p):
        ex = 3 * pad_step[need_p[0]] + pad_step[need_p[1]]
        pt, lo, hi = PRED_GM1_DENSE["exact"]
        out.append(("GM-DENSE-EXACT", "CONFIRMED" if lo <= ex <= hi else "REFUTED",
                    "3 x padded(%d, %d) + padded(%d, %d) = %.2f ms (point %s, band [%s, %s]; HBM %s: %s)" % (
                        B, mfix, K * B, mfix, ex, pt, lo, hi, hbm, "BINDING" if ex > hbm else "not binding")))
    return out


def bigc_verdicts(nat_step, bigc_step, U: int = SCAN_U):
    """[bigemmc] The registered G-c predictions (FORECAST COSTS) against per-step totals: nat_step {M: ms},
    bigc_step {variant: {M: ms}}. [(name, verdict, detail)]; the research arm's names carry RESEARCH."""
    out = []
    B = 287
    for v, steps in sorted(bigc_step.items()):
        lab = "GC-%s%s" % (v.upper(), "-RESEARCH" if v == "bucket" else "")
        for M in sorted(set(steps) & set(nat_step)):
            r = steps[M] / nat_step[M]
            reg = PRED_BIGC_RATIO.get(M)
            verdict = "REPORT" if reg is None else ("CONFIRMED" if reg[1] <= r <= reg[2] else "REFUTED")
            out.append(("%s-RATIO-%d" % (lab, M), verdict, "G-c %s %.3f ms / natural %.3f ms = %.3f per step%s" % (
                v, steps[M], nat_step[M], r, "" if reg is None else " (point %s, band [%s, %s])" % reg)))
        if B in steps and U * B in steps:
            d = 3 * steps[B] + steps[U * B]
            p = PRED_BIGC_DENSE
            out.append(("%s-DENSE" % lab, "CONFIRMED" if p["low"] <= d <= p["high"] else "REFUTED",
                        "dense' = 3 x G-c(%d) + G-c(%d) = %.2f ms (point %s, band [%s, %s])" % (
                            B, U * B, d, p["point"], p["low"], p["high"])))
            out.append(("%s-FALSIFIER" % lab, "FALSIFIED" if d >= p["falsifier_ms"] else "NOT-FALSIFIED",
                        "dense' %.2f ms %s HBM %.3f ms%s" % (d, ">=" if d >= p["falsifier_ms"] else "<",
                                                             p["falsifier_ms"], ": exactness costs Q"
                                                             if d >= p["falsifier_ms"] else "")))
    return out


def gm1_repeat(nat_step):
    """[bigemmc] {M: (measured, gm1, relative difference)} at every M that gm1 measured."""
    return {M: (v, GM1_NATURAL[M], v / GM1_NATURAL[M] - 1.0) for M, v in sorted(nat_step.items()) if M in GM1_NATURAL}


def run_gemm_scan(a) -> int:
    """--gemm-scan: the gm1 cell (no B1-B5). Writes --out/summary.json; exit = the failed MUST gates."""
    import math
    import torch
    os.makedirs(a.out)
    gen = torch.Generator().manual_seed(a.seed)
    ms = [int(x) for x in a.ms.split(",") if x]
    bs = [int(x) for x in a.exact_b.split(",") if x]
    gates, preds, tables = [], [], {}
    timer = None
    if a.device == "cpu":
        def timer(fn):
            for _ in range(a.warm):
                fn()
            xs = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                fn()
                xs.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(xs)
    variants = list(BIGC_VARIANTS) if a.bigemm_c == "both" else ([a.bigemm_c] if a.bigemm_c else [])
    bigc, bigc_rows, bigc_meta = {}, [], None
    try:
        if variants:                                                     # [bigemmc]
            import nosi.bi_linear_c as BLC
            if a.bigc_table:
                BLC.use_table(BLC.load_table(a.bigc_table, a.bigc_table_sha))
            for v in variants:
                fn = dict(strict=BLC.bi_c_linear_strict, bucket=BLC.bi_c_linear_bucket)[v]
                if a.device == "cpu":                                    # the interpreter (CPU test): fp32 operands
                    fn = (lambda f: lambda x, w: f(x.float(), w.float()))(fn)
                bigc[v] = fn
            bigc_meta = dict(variants=variants, describe={v: BLC.describe(v) for v in variants}, table=a.bigc_table,
                             table_sha_pinned=a.bigc_table_sha, cpu_fp32_upcast=a.device == "cpu",
                             gm1_natural=GM1_NATURAL, gm1_repeat_tol=GM1_REPEAT_TOL,
                             predictions=dict(ratio={str(k): v for k, v in PRED_BIGC_RATIO.items()},
                                              dense=PRED_BIGC_DENSE, note="forecast costs, not measured speedups"))
        natural, padded, inv = gemm_scan(ms, bs, gen, a.warm, a.reps, a.U, a.shape_div, a.device, timer,
                                         bigc=bigc, bigc_rows=bigc_rows)
        nat_ms, pad_pairs = scan_plan(ms, bs, a.U)
        want = len(scan_shapes(a.shape_div)) * (len(nat_ms) + len(pad_pairs))
        got = natural + padded
        bad = [r for r in got if not (isinstance(r["ms"], float) and math.isfinite(r["ms"]) and r["ms"] > 0)]
        gates.append(gate("GM-RAN", len(got) == want and not bad, True,
                          "%d of %d timings, %d not finite or <= 0" % (len(got), want, len(bad))))
        nat_step = per_step_totals(natural, "M", LAYERS)
        pad_step = per_step_totals(padded, "pair", LAYERS)
        tables = dict(natural=natural, padded=padded, invariance=inv,
                      per_step_natural={str(k): round(v, 4) for k, v in sorted(nat_step.items())},
                      per_step_padded={"%d_%d" % k: round(v, 4) for k, v in sorted(pad_step.items())})
        nbad = {k: v for k, v in inv.items() if v["rows_differing_garbage"] or v["rows_differing_zero"]}
        gates.append(gate("R_GM_padded_invariance", True, False, "largest pair, rows differing: %s" % (
            json.dumps(nbad) if nbad else "none (%s)" % json.dumps(inv))))
        preds = gm1_verdicts(nat_step, pad_step, a.U)
        if variants:                                                     # [bigemmc]
            want_c = len(scan_shapes(a.shape_div)) * len(nat_ms) * len(variants)
            bad_c = [r for r in bigc_rows if not (isinstance(r["ms"], float) and math.isfinite(r["ms"]) and r["ms"] > 0)]
            gates.append(gate("GC-RAN", len(bigc_rows) == want_c and not bad_c, True,
                              "%d of %d G-c timings, %d not finite or <= 0" % (len(bigc_rows), want_c, len(bad_c))))
            bigc_step = {v: per_step_totals([r for r in bigc_rows if r["variant"] == v], "M", LAYERS) for v in variants}
            rep = gm1_repeat(nat_step)
            off = {M: x for M, x in rep.items() if abs(x[2]) > GM1_REPEAT_TOL}
            applies = a.device == "cuda" and a.shape_div == 1
            gates.append(gate("GC-NAT-REPEATS-GM1" if applies else "R_GC-NAT-REPEATS-GM1", (not off and bool(rep))
                              if applies else True, applies,
                              ("natural vs gm1 2190876 at %d M, tolerance %.0f %%: outside %s" % (
                                  len(rep), 100 * GM1_REPEAT_TOL, json.dumps({str(k): [round(y, 4) for y in v]
                                                                             for k, v in off.items()}) if off else "none"))
                              if applies else "not applicable (device %s, shape-div %d)" % (a.device, a.shape_div)))
            tables.update(bigc=bigc_rows, per_step_bigc={v: {str(k): round(x, 4) for k, x in sorted(s.items())}
                                                         for v, s in bigc_step.items()},
                          gm1_repeat={str(k): [round(y, 4) for y in v] for k, v in rep.items()})
            preds += bigc_verdicts(nat_step, bigc_step, a.U)
            for v, s in sorted(bigc_step.items()):
                for M, x in sorted(s.items()):
                    log("bigc per step %s M %d: %.3f ms (natural %.3f)" % (v, M, x, nat_step[M]))
        for M, v in sorted(nat_step.items()):
            log("gm1 per step natural M %d: %.3f ms" % (M, v))
        for p, v in sorted(pad_step.items()):
            log("gm1 per step padded live %d M_fix %d: %.3f ms" % (p[0], p[1], v))
    except Exception as e:
        traceback.print_exc()
        gates.append(gate("GM-RAN", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    meta = dict(cell="bigc-time" if variants else "gm1", bigc=bigc_meta,
                ms=ms, exact_b=bs, U=a.U, warm=a.warm, reps=a.reps, seed=a.seed, device=a.device,
                shape_div=a.shape_div, shapes=scan_shapes(a.shape_div), layers=LAYERS, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
                commit=os.environ.get("BI_COMMIT", "unknown"), old_mscan=OLD_MSCAN, eager_factor=EAGER_FACTOR,
                predictions=dict(natural={str(k): v for k, v in PRED_GM1.items()},
                                 ratio={str(k): v for k, v in PRED_GM1_RATIO.items()}, dense=PRED_GM1_DENSE))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, gates=gates, tables=tables,
                       predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds]),
                  f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                g["detail"][:400]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("r2_bi_microgate gm1: %d MUST gate(s) failed" % nfail)
    return min(nfail, 125)


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
    ap.add_argument("--gemm-scan", action="store_true", help="the gm1 cell: the GEMM cost scan only (no B1-B5)")
    ap.add_argument("--ms", default=",".join(str(m) for m in SCAN_MS))
    ap.add_argument("--exact-b", default=",".join(str(b) for b in SCAN_EXACT_B))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cpu: the CPU test only (--gemm-scan)")
    ap.add_argument("--shape-div", type=int, default=1, help="shapes / N: the CPU test only (--gemm-scan)")
    ap.add_argument("--bigemm-c", default=None, choices=("strict", "bucket", "both"),
                    help="[bigemmc] with --gemm-scan: G-c columns (strict = the candidate; bucket = the research arm)")
    ap.add_argument("--bigc-table", default=None, help="[bigemmc] a frozen table file (bigc_tune.py's table.json)")
    ap.add_argument("--bigc-table-sha", default=None, help="[bigemmc] its canonical digest")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if (a.bigemm_c or a.bigc_table or a.bigc_table_sha) and not a.gemm_scan:
        print("REFUSED: --bigemm-c / --bigc-table go with --gemm-scan", file=sys.stderr)
        return 2
    if (a.bigc_table is None) != (a.bigc_table_sha is None) or (a.bigc_table and not a.bigemm_c):
        print("REFUSED: --bigc-table and --bigc-table-sha go together, with --bigemm-c", file=sys.stderr)
        return 2
    if (a.device != "cuda" or a.shape_div != 1) and not a.gemm_scan:
        print("REFUSED: --device / --shape-div go with --gemm-scan", file=sys.stderr)
        return 2
    if a.shape_div < 1:
        print("REFUSED: --shape-div >= 1", file=sys.stderr)
        return 2
    if a.gemm_scan:
        if a.biattn or a.bigemm:
            print("REFUSED: --gemm-scan takes no --biattn / --bigemm", file=sys.stderr)
            return 2
        return run_gemm_scan(a)
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
