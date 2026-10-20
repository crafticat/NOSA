"""GPU gate of the BATCH-INVARIANT PAGED decode attention (nosi/nosi/bi_paged_attention.py; R2-M2 step 7, piece 3;
design note docs/R2_STEP7_SPEC_ROUND.md sections 3.4 and 7.2 B4). Holder step: benchmarks/Efficiency/biattn_step.sh.
CPU tests: benchmarks/Efficiency/tests_cpu/test_bi_paged_attention.py.

SYNTHETIC rows at L = 16128 (NOSA-8B geometry: 32 query heads, 2 KV heads, head dim 128, bf16): B requests, 1 + U rows
per request at positions L + 10 + u (u = 0 .. U). Per (request, KV head) 63 historical blocks of 252 plus U extra
blocks live in slots in a random order (so slot order is not block order); the row at u lists base[u:] + extra[:u]
(63 blocks) and the live tail block 252 (its tail page, cut at last_len = 11 + u): 64 pages, canonical order
(build_page_table). The FA2-NOSA reference is the shipped decode call (nosa_llama.decode_forward): q (B, 1, 32, 128),
the 64-slot window (B, 4096, 2, 128), the bias (B, 4096, 2), cache_seqlens 4032 + 11, num_splits 4 (the R2 gates'
setting) and 1.

MUST gates (each counted in the exit code):
  G1 semantics: sampled rows against the float64 oracle (attention_oracle), max |d| <= 3e-2.
  G2 row invariance (torch.equal) at B = 64 and 128 with 1 + U = 4: every sampled row ALONE (a fresh one-row call),
     the batch PERMUTED, new NEIGHBOURS (every other row's q and page list replaced), a WIDER page table (-1 padded),
     PREFIX batches of 1, 7, 64 and R / 2 rows.
  G3 NaN poison: every row nobody lists set to NaN (unlisted slots; tail-page rows at or beyond the request's largest
     extent): the batch output torch.equal the clean one and finite; and per sampled row, alone, every row beyond
     ITS extent poisoned: equal.
  G4 negative control: the split size chosen by the batch (8 pages per split for a row alone, 4 in the batch) must
     differ somewhere; its PRECONDITION first (8 and 4 pages per split give different bits on the same batch); if either
     fails, the invariance check is blind and the gate FAILS. (Version 1's G4 used a 2-page tile; version 2 has no
     tile knob.)
REPORT: G5 page order (rows that change when the same pages are listed in slot order), the timing table (median of 20
CUDA-event timed calls after 3 warm-ups, as for version 1; BI with a preallocated workspace) with the BI time /
FA2-NOSA split-4 time ratio against the prediction registered in the commit message (PRED below); also the program
count and a back-to-back time (10 calls per event pair: the host launch gap mostly drops out), report only.
OUTPUT: --out DIR/biattn_gate.json and .md; the last stdout line is BIATTN_GATE_DONE failed=<n>; exit = n (2 = refused).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "nosi/nosi/bi_paged_attention.py"
L = 16128
HQ, HKV, D = 32, 2, 128
NBLK = L // 64                   # 252 complete historical blocks; the live tail block is 252
HIST = 63
S_SLOTS = 72                     # 63 base + up to 3 extra + spare; slot 71 is the tail page
TAIL_SLOT = 71
TAIL_ROWS0 = 10                  # position L + 10 + u -> last_len 11 + u
BATCHES = (64, 128)
ROWS_PER_REQ = (1, 2, 4)
# REGISTERED PREDICTION for VERSION 2 (the fixed-size split-KV; commit message): BI time over B*(1+U) rows /
# FA2-NOSA split-4 time over B rows, (central, low, high)
PRED = {(64, 1): (1.8, 1.1, 3.5), (64, 2): (2.4, 1.4, 4.5), (64, 4): (3.6, 2.0, 7.0),
        (128, 1): (1.7, 1.0, 3.5), (128, 2): (2.3, 1.3, 4.5), (128, 4): (3.4, 1.9, 7.0)}
# VERSION 1's prediction (c866d7c), REFUTED by run biattn_v1_2188818: measured 8.82 / 13.42 / 21.71 (B 64) and
# 8.95 / 13.25 / 22.13 (B 128). Kept for the record; not used.
PRED_V1 = {(64, 1): (1.4, 0.9, 2.5), (64, 2): (1.9, 1.2, 3.2), (64, 4): (2.8, 1.8, 4.5),
           (128, 1): (1.1, 0.8, 1.8), (128, 2): (1.6, 1.0, 2.6), (128, 4): (2.4, 1.5, 3.8)}


def load_module():
    spec = importlib.util.spec_from_file_location("bi_paged_attention", str(MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(bi, B, U1, seed):
    """The synthetic batch (module docstring) on the GPU."""
    dev = "cuda"
    gc = torch.Generator(device="cpu").manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed)
    bf = torch.bfloat16
    k = (torch.randn(B, S_SLOTS * 64, HKV, D, generator=gd, device=dev) * 0.5).to(bf)
    v = (torch.randn(B, S_SLOTS * 64, HKV, D, generator=gd, device=dev) * 0.5).to(bf)
    bias = (torch.rand(B, S_SLOTS * 64, HKV, generator=gd, device=dev) - 0.5).to(bf)
    R = B * U1
    q = torch.randn(R, HQ, D, generator=gd, device=dev).to(bf)
    ids = torch.empty(R, HKV, HIST + 1, dtype=torch.int64)
    slots = torch.empty(R, HKV, HIST + 1, dtype=torch.int64)
    for b in range(B):
        for h in range(HKV):
            perm = torch.randperm(NBLK, generator=gc)
            base, extra = perm[:HIST], perm[HIST:HIST + 3]
            for u in range(U1):
                r = b * U1 + u
                ids[r, h, :HIST] = torch.cat([base[u:], extra[:u]])
                slots[r, h, :HIST] = torch.cat([torch.arange(u, HIST), torch.arange(HIST, HIST + u)])
                ids[r, h, HIST] = NBLK
                slots[r, h, HIST] = TAIL_SLOT
    table, num_pages = bi.build_page_table(ids.to(dev), slots.to(dev))
    seq_idx = torch.arange(B, device=dev, dtype=torch.int32).repeat_interleave(U1)
    pos = L + TAIL_ROWS0 + torch.arange(U1, device=dev).repeat(B)
    last_len = bi.last_len_of(pos)
    return dict(q=q, k=k, v=v, bias=bias, seq_idx=seq_idx.contiguous(), table=table, num_pages=num_pages,
                last_len=last_len, B=B, U1=U1, R=R)


def call(bi, c, rows=None, table=None, k=None, v=None, bias=None, q=None, pages_per_split=None):
    t = c["table"] if table is None else table
    qq = c["q"] if q is None else q
    if rows is None:
        args = (qq, t, c["seq_idx"], c["num_pages"], c["last_len"])
    else:
        idx = torch.as_tensor(rows, dtype=torch.int64, device=qq.device)
        args = tuple(x.index_select(0, idx).contiguous() for x in (qq, t, c["seq_idx"], c["num_pages"], c["last_len"]))
    return bi.bi_paged_decode(args[0], c["k"] if k is None else k, c["v"] if v is None else v,
                              c["bias"] if bias is None else bias, args[2], args[1], args[3], args[4],
                              pages_per_split=bi.PAGES_PER_SPLIT if pages_per_split is None else pages_per_split)


def used_rows(bi, c, own_rows_only=None):
    """(N, S*64, Hkv) bool: the rows some row of the batch (or only the given row) reads."""
    N, SR, H, _ = c["k"].shape
    dev = c["k"].device
    rows = range(c["R"]) if own_rows_only is None else [own_rows_only]
    used = torch.zeros(N, SR, H, dtype=torch.bool, device=dev)
    ar = torch.arange(64, device=dev)
    for r in rows:
        n = int(c["seq_idx"][r])
        last = int(c["last_len"][r])
        for h in range(H):
            npg = int(c["num_pages"][r, h])
            sl = c["table"][r, h, :npg].to(torch.int64)
            full = (sl[:-1].unsqueeze(1) * 64 + ar.unsqueeze(0)).reshape(-1)
            used[n, full, h] = True
            used[n, int(sl[-1]) * 64 + torch.arange(last, device=dev), h] = True
    return used


def poison(c, used):
    k, v, b = c["k"].clone(), c["v"].clone(), c["bias"].clone()
    nan = float("nan")
    k[~used] = nan
    v[~used] = nan
    b[~used] = nan
    return k, v, b


def time_ms(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def time_b2b_ms(fn, calls=10, iters=10, warm=3):
    """REPORT only: the median over `iters` of (one event pair around `calls` back-to-back calls) / calls. The host
    enqueues ahead of the GPU, so the launch gap that time_ms sees after each sync mostly drops out."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(calls):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / calls)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print("[biattn] REFUSED: %s exists" % args.out, flush=True)
        return 2
    os.makedirs(args.out)
    bi = load_module()
    from flash_attn_nosa import flash_attn_with_kvcache as fa2_nosa
    gates, report = [], dict(timing=[], page_order=[], meta=dict(
        L=L, HQ=HQ, HKV=HKV, D=D, pages_per_row=HIST + 1, page=bi.PAGE, num_warps=bi.NUM_WARPS,
        num_stages=bi.NUM_STAGES, pages_per_split=bi.PAGES_PER_SPLIT, torch=torch.__version__,
        device=torch.cuda.get_device_name(0)))

    def gate(name, ok, detail):
        gates.append(dict(gate=name, passed=bool(ok), detail=detail))
        print("[biattn] %s %s: %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)

    gsel = torch.Generator(device="cpu").manual_seed(5)
    for B in BATCHES:
        c = build(bi, B, 4, seed=B)
        R = c["R"]
        full = call(bi, c)
        torch.cuda.synchronize()
        # G1 semantics on 16 rows
        sample = sorted(set(torch.randperm(R, generator=gsel)[:16].tolist()))
        sub = dict(c, q=c["q"][sample].contiguous(), seq_idx=c["seq_idx"][sample].contiguous(),
                   table=c["table"][sample].contiguous(), num_pages=c["num_pages"][sample].contiguous(),
                   last_len=c["last_len"][sample].contiguous())
        ref = bi.attention_oracle(sub["q"], c["k"], c["v"], c["bias"], sub["seq_idx"], sub["table"],
                                  sub["num_pages"], sub["last_len"])
        d = float((full[sample].double() - ref).abs().max())
        gate("G1-semantics[B%d]" % B, d <= 3e-2, "max |kernel - float64 oracle| = %.3e over %d rows" % (d, len(sample)))
        # G2 invariance
        rows64 = sorted(set(torch.randperm(R, generator=gsel)[:64].tolist()))
        bad = []
        for r in rows64:
            if not torch.equal(call(bi, c, rows=[r])[0], full[r]):
                bad.append(("alone", r))
        perm = torch.randperm(R, generator=gsel).tolist()
        pout = call(bi, c, rows=perm)
        bad += [("permuted", r) for i, r in enumerate(perm) if not torch.equal(pout[i], full[r])]
        q2 = torch.randn_like(c["q"], dtype=torch.float32).to(c["q"].dtype)
        t2 = torch.roll(c["table"], shifts=1, dims=0)              # another valid row's pages (same entry layout)
        keep = torch.zeros(R, dtype=torch.bool, device=full.device)
        keep[rows64] = True
        q_mix = torch.where(keep.view(R, 1, 1), c["q"], q2).contiguous()
        t_mix = torch.where(keep.view(R, 1, 1), c["table"], t2).contiguous()
        np_mix = torch.where(keep.view(R, 1), c["num_pages"], torch.roll(c["num_pages"], 1, 0)).contiguous()
        nout = bi.bi_paged_decode(q_mix, c["k"], c["v"], c["bias"], c["seq_idx"], t_mix, np_mix, c["last_len"])
        bad += [("neighbours", r) for r in rows64 if not torch.equal(nout[r], full[r])]
        wide = torch.cat([c["table"], torch.full_like(c["table"], -1)[..., :16]], dim=-1).contiguous()
        wout = call(bi, c, table=wide)
        bad += [("wider-table", r) for r in range(R) if not torch.equal(wout[r], full[r])]
        for pre in (1, 7, 64, R // 2):
            pout = call(bi, c, rows=list(range(pre)))
            bad += [("prefix%d" % pre, r) for r in range(pre) if not torch.equal(pout[r], full[r])]
        gate("G2-invariance[B%d,1+U=4]" % B, not bad, "%d differing (kind, row) of %d checks%s" % (
            len(bad), len(rows64) * 2 + R * 2 + 1 + 7 + 64 + R // 2, ("; first %r" % (bad[:4],)) if bad else ""))
        # G3 NaN poison
        k3, v3, b3 = poison(c, used_rows(bi, c))
        dirty = call(bi, c, k=k3, v=v3, bias=b3)
        ok = bool(torch.isfinite(dirty).all()) and torch.equal(dirty, full)
        own_bad = []
        for r in rows64[:8]:
            k4, v4, b4 = poison(c, used_rows(bi, c, own_rows_only=r))
            if not torch.equal(call(bi, c, rows=[r], k=k4, v=v4, bias=b4)[0], full[r]):
                own_bad.append(r)
        gate("G3-nan-poison[B%d]" % B, ok and not own_bad, "batch equal and finite: %s; rows alone with every unread "
             "row poisoned that differ: %s" % (ok, own_bad))
        # G4 negative control: a split size chosen by the batch (8 alone, 4 in the batch); its precondition first
        neg = call(bi, c, pages_per_split=4)
        pre = not torch.equal(call(bi, c, pages_per_split=8), neg)
        caught = [r for r in rows64 if not torch.equal(call(bi, c, rows=[r], pages_per_split=8)[0], neg[r])] if pre else []
        gate("G4-negative-split-caught[B%d]" % B, pre and bool(caught), "precondition (8 vs 4 pages per split differ "
             "on the batch): %s; rows where 4 pages per split in the batch differ from 8 alone: %d of %d" % (
                 pre, len(caught), len(rows64)))
        # G5 page order (report)
        so = c["table"].clone()
        head = torch.sort(so[..., :HIST], dim=-1).values                  # slot order (the tail page stays last)
        so[..., :HIST] = head
        sout = call(bi, c, table=so.contiguous())
        moved = sum(1 for r in range(R) if not torch.equal(sout[r], full[r]))
        report["page_order"].append(dict(B=B, rows=R, rows_changed_by_slot_order=moved))
        print("[biattn] G5-page-order[B%d] (report): %d of %d rows change when listed in slot order" % (B, moved, R),
              flush=True)
        del c, full
        torch.cuda.empty_cache()

    # TIMING (report)
    for B in BATCHES:
        bf = torch.bfloat16
        gd = torch.Generator(device="cuda").manual_seed(77)
        kf = (torch.randn(B, 4096, HKV, D, generator=gd, device="cuda") * 0.5).to(bf)
        vf = (torch.randn(B, 4096, HKV, D, generator=gd, device="cuda") * 0.5).to(bf)
        bfias = (torch.rand(B, 4096, HKV, generator=gd, device="cuda") - 0.5).to(bf)
        qf = torch.randn(B, 1, HQ, D, generator=gd, device="cuda").to(bf)
        cl = torch.full((B,), 4032 + TAIL_ROWS0 + 1, dtype=torch.int32, device="cuda")
        fa = {}
        for sp in (4, 1):
            fa[sp] = time_ms(lambda: fa2_nosa(qf, kf, vf, bfias, cache_seqlens=cl, num_splits=sp))
        fa_b2b = time_b2b_ms(lambda: fa2_nosa(qf, kf, vf, bfias, cache_seqlens=cl, num_splits=4))
        for U1 in ROWS_PER_REQ:
            c = build(bi, B, U1, seed=1000 + B + U1)
            out = torch.empty_like(c["q"])
            nsp = bi.splits_for(c["table"].shape[2])
            bh = bi.block_h_for(HQ // HKV)
            ws = (torch.empty((c["R"], HKV, nsp, bh, D), dtype=torch.float32, device="cuda"),   # a fixed buffer, as
                  torch.empty((c["R"], HKV, nsp, 2, bh), dtype=torch.float32, device="cuda"))   # a graphed tick has
            fn = lambda: bi.bi_paged_decode(c["q"], c["k"], c["v"], c["bias"], c["seq_idx"], c["table"],
                                            c["num_pages"], c["last_len"], out=out, workspace=ws)
            t_bi = time_ms(fn)
            t_b2b = time_b2b_ms(fn)
            rows_read = c["R"] * HKV * (HIST * 64 + TAIL_ROWS0 + 1)
            gbs = rows_read * D * 2 * 2 / (t_bi * 1e-3) / 1e9
            ratio = t_bi / fa[4]
            cen, lo, hi = PRED[(B, U1)]
            row = dict(B=B, rows_per_request=U1, rows=c["R"], bi_ms=t_bi, fa2_split4_ms=fa[4], fa2_split1_ms=fa[1],
                       ratio_bi_over_fa2_split4=ratio, ratio_bi_over_fa2_split1=t_bi / fa[1],
                       bi_b2b_ms=t_b2b, fa2_split4_b2b_ms=fa_b2b, ratio_b2b=t_b2b / fa_b2b, splits=nsp,
                       programs=c["R"] * HKV * nsp, bi_logical_GBps=gbs, predicted=cen, predicted_band=[lo, hi],
                       inside_band=lo <= ratio <= hi)
            report["timing"].append(row)
            print("[biattn] timing B=%d 1+U=%d rows=%d: BI %.3f ms, FA2-NOSA split4 %.3f / split1 %.3f ms; ratio %.2f "
                  "(predicted %.1f, band [%.1f, %.1f]: %s); back-to-back BI %.3f / FA2 %.3f ms (ratio %.2f, report); "
                  "%d programs; BI logical %.0f GB/s" % (
                      B, U1, c["R"], t_bi, fa[4], fa[1], ratio, cen, lo, hi, "inside" if lo <= ratio <= hi else "OUTSIDE",
                      t_b2b, fa_b2b, t_b2b / fa_b2b, c["R"] * HKV * nsp, gbs), flush=True)
            del c, out, ws, fn
            torch.cuda.empty_cache()
        del kf, vf, bfias, qf

    nfail = sum(1 for g in gates if not g["passed"])
    lines = ["# BI paged attention gate", "", "| gate | result | detail |", "|---|---|---|"]
    lines += ["| %s | %s | %s |" % (g["gate"], "PASS" if g["passed"] else "FAIL", g["detail"]) for g in gates]
    lines += ["", "| B | 1+U | rows | programs | BI ms | FA2-NOSA split4 ms | split1 ms | ratio vs split4 | "
              "predicted [band] | inside | b2b BI ms | b2b FA2 ms | b2b ratio |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += ["| %d | %d | %d | %d | %.3f | %.3f | %.3f | %.2f | %.1f [%.1f, %.1f] | %s | %.3f | %.3f | %.2f |" % (
        t["B"], t["rows_per_request"], t["rows"], t["programs"], t["bi_ms"], t["fa2_split4_ms"], t["fa2_split1_ms"],
        t["ratio_bi_over_fa2_split4"], t["predicted"], t["predicted_band"][0], t["predicted_band"][1], t["inside_band"],
        t["bi_b2b_ms"], t["fa2_split4_b2b_ms"], t["ratio_b2b"])
        for t in report["timing"]]
    lines += ["", "failed MUST gates: %d" % nfail]
    with open(os.path.join(args.out, "biattn_gate.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "biattn_gate.json"), "w") as f:
        json.dump(dict(gates=gates, failed=nfail, **report), f, indent=1, default=str)
    print("\n".join(lines), flush=True)
    print("BIATTN_GATE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
