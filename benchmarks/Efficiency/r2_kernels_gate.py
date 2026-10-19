"""R2 milestone 2, step 4 (KERNELS), item 3: the GPU gate of the per-request stage-1 wrapper (nosi/nosi/
stage1_seqused.py) and the per-request pooling kernel (nosi/nosi/max_pooling_per_request.py), EAGER (no graph), on one
A100 in the holder (benchmarks/Efficiency/r2_kernels_step.sh runs it). Random inputs of NOSA-8B's shapes (HQ 32, HKV 2,
D 128, bf16; L 16128: n_comp 1007 compressed keys of kernel 32 / stride 16, M = 253 pooled blocks of 64), seeded.

Every comparison is BITWISE (the bf16 bit patterns, beq: torch.equal is False on any NaN, and the op's masked columns
may hold NaN). MUST gates (exit code = the number of failed MUST gates; summary.json records every gate):
  G1 stage1_seqused(seqused_k=None) == the shipped infllmv2_attn_stage1_fast (the same op call).
  G2 stage1_seqused(seqused_k = every request's full length) == the shipped call (seqused_k equal to the cu_seqlens_k
     extents changes nothing).
  G3 PER-REQUEST LENGTHS: a padded table (B, Mcap = 1007) with cu_seqlens_k = arange(B + 1) Mcap and seqused_k = n_b
     (varied, 64 .. 1007): row b's first n_b columns == a SEPARATE B = 1 shipped call on that request's own n_b keys,
     for every b. G3-tail (REPORT): the columns >= n_b hold only the op's zero fill.
  G4 nosa_pooling_per_request with every n_comp[b] = the shipped scalar and every M_b[b] = M == the shipped nosa_pooling,
     on both outputs (q and cis).
  G5 PER-REQUEST POOLING: n_comp[b], M_b[b] varied: row b's first M_b columns == a SEPARATE shipped nosa_pooling of that
     request alone (its own scalar n_comp[b] and a buffer of width M_b[b]); the columns >= M_b are -inf.
  G6 the per-request kernel == max_pooling_per_request.pool_reference (the torch twin) on the device.
  G7 THE topk(sorted=False) ORDER QUESTION (the decode's captured top-k, nosa_llama.py:460-490): torch.topk(k,
     sorted=False) of the shipped pooled rows of width M and of the same rows padded with -inf to M + 64 select the
     same MULTISET OF VALUES at k = 33 and 64 (MUST: a top-k is defined by its values). G7-order (REPORT, the question
     itself): per k, the rows whose selected index SET and whose RAW order are identical, and the rows with a tie at the
     cut. The plan's mitigation (an explicit sort of the selected ids on both paths) makes the order identical; only a
     tie at the cut can change the SET.
Nothing is timed; no model is loaded. Run (inside the image, one GPU):
    python benchmarks/Efficiency/r2_kernels_gate.py --out <new dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NOSI_PKG = os.path.join(HERE, "..", "..", "nosi", "nosi")
HQ, HKV, D = 32, 2, 128
B = 64
NCOMP = (16128 - 32) // 16 + 1          # 1007 compressed keys at L = 16128 (kernel 32, stride 16)
M_POOL = (16128 + 1 + 63) // 64         # 253 pooled blocks (nosa_llama decode_inference out_len)
TOPK, QK_SELECT, INIT, LOCAL = 64, 33, 1, 16


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NOSI_PKG, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(t):
    """The raw 16-bit patterns of a bf16 tensor: a BITWISE comparison (NaN payloads included; torch.equal is False on
    any NaN, and the op's masked score columns may hold NaN: the slow wrapper replaces them, the fast one does not)."""
    return t.contiguous().view(torch.int16)


def beq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def _first_diff(a, b):
    if a.shape != b.shape:
        return "shape %s != %s" % (tuple(a.shape), tuple(b.shape))
    d = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
    if not bool(d.any()):
        return None
    i = tuple(int(x) for x in d.nonzero()[0])
    return "first difference at %s: %r != %r" % (i, float(a[i]), float(b[i]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("r2 kernels gate: --out %s exists" % a.out, file=sys.stderr)
        return 2
    os.makedirs(a.out)
    from infllm_v2 import infllmv2_attn_stage1_fast as shipped_stage1
    s1 = _load("nosi_stage1_seqused_gate", "stage1_seqused.py")
    pool_new = _load("nosi_max_pooling_per_request_gate", "max_pooling_per_request.py")
    pool_old = _load("nosi_max_pooling_fused_gate", "max_pooling_fused.py")
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(20261019)
    rec = dict(t0=time.time(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, gates={}, report={})
    nfail = 0

    def gate(name, ok, detail, must=True):
        nonlocal nfail
        rec["gates"][name] = dict(passed=bool(ok), must=must, detail=detail)
        print("%s %s %s" % (name, "PASS" if ok else ("FAIL" if must else "REPORT"), detail), flush=True)
        if must and not ok:
            nfail += 1

    bf = torch.bfloat16
    q = torch.randn((B, HQ, D), generator=g, device=dev, dtype=torch.float32).to(bf)
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)
    # ---- G1 / G2: equal lengths
    ck = torch.randn((B * NCOMP, HKV, D), generator=g, device=dev, dtype=torch.float32).to(bf)
    cu_k = torch.arange(B + 1, dtype=torch.int32, device=dev) * NCOMP
    ref = shipped_stage1(q, ck, ck, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=NCOMP, causal=False)
    new = s1.stage1_seqused(q, ck, ck, cu_q, cu_k, 1, NCOMP, seqused_k=None)
    torch.cuda.synchronize()
    gate("G1", beq(new, ref), _first_diff(new, ref) or "bitwise equal %s %s" % (tuple(new.shape), new.dtype))
    full = torch.full((B,), NCOMP, dtype=torch.int32, device=dev)
    new2 = s1.stage1_seqused(q, ck, ck, cu_q, cu_k, 1, NCOMP, seqused_k=full)
    torch.cuda.synchronize()
    gate("G2", beq(new2, ref), _first_diff(new2, ref) or "bitwise equal")
    # ---- G3: per-request lengths vs B separate B = 1 calls
    nb = [max(64, NCOMP - 13 * b) for b in range(B)]
    seqused = torch.tensor(nb, dtype=torch.int32, device=dev)
    out = s1.stage1_seqused(q, ck, ck, cu_q, cu_k, 1, NCOMP, seqused_k=seqused)
    bad, zero_bad = [], []
    for b in range(B):
        kb = ck[b * NCOMP:b * NCOMP + nb[b]]
        r = shipped_stage1(q[b:b + 1], kb, kb, cu_seqlens_q=cu_q[:2], cu_seqlens_k=torch.tensor([0, nb[b]], dtype=torch.int32,
                           device=dev), max_seqlen_q=1, max_seqlen_k=nb[b], causal=False)
        if not beq(out[:, b, :nb[b]], r[:, 0, :nb[b]]):
            bad.append((b, nb[b], _first_diff(out[:, b, :nb[b]], r[:, 0, :nb[b]])))
        if bool((out[:, b, nb[b]:] != 0).any()):
            zero_bad.append(b)
    torch.cuda.synchronize()
    gate("G3", not bad, "%d requests, n_b %d .. %d; rows whose first n_b columns differ bitwise: %s"
         % (B, min(nb), max(nb), bad[:3]))
    gate("G3-tail", not zero_bad, "columns >= n_b hold only the op's zero fill: rows with another value %s (REPORT: "
         "the per-request pooling never reads them)" % zero_bad[:5], must=False)
    # ---- G4: pooling, all lengths equal
    W = ref.shape[-1]
    score = ref.contiguous()                                        # (HKV, B, W): real stage-1 outputs as the input
    cis = torch.randn((HKV, B, W), generator=g, device=dev, dtype=torch.float32).to(bf)
    pq_old = torch.empty((HKV, B, M_POOL), dtype=bf, device=dev)
    pc_old = torch.empty_like(pq_old)
    pool_old.nosa_pooling(score, cis, NCOMP, pq_old, pc_old, stride=16, block_size=64, local_blocks=LOCAL,
                          init_blocks=INIT)
    pq_new = torch.empty_like(pq_old)
    pc_new = torch.empty_like(pq_old)
    pool_new.nosa_pooling_per_request(score, cis, full, torch.full((B,), M_POOL, dtype=torch.int32, device=dev),
                                      pq_new, pc_new, stride=16, block_size=64, local_blocks=LOCAL, init_blocks=INIT)
    torch.cuda.synchronize()
    gate("G4", beq(pq_new, pq_old) and beq(pc_new, pc_old),
         "q: %s; cis: %s" % (_first_diff(pq_new, pq_old) or "equal", _first_diff(pc_new, pc_old) or "equal"))
    # ---- G5: per-request pooling vs B separate shipped calls; G6: vs the torch twin
    mb = [max(40, M_POOL - 3 * b) for b in range(B)]
    ncb = [min(NCOMP, 4 * m - 2) for m in mb]                         # a request's windows for its own pooled width
    ncomp_t = torch.tensor(ncb, dtype=torch.int32, device=dev)
    mb_t = torch.tensor(mb, dtype=torch.int32, device=dev)
    pq5 = torch.empty_like(pq_old)
    pc5 = torch.empty_like(pq_old)
    pool_new.nosa_pooling_per_request(score, cis, ncomp_t, mb_t, pq5, pc5, stride=16, block_size=64,
                                      local_blocks=LOCAL, init_blocks=INIT)
    bad5 = []
    for b in range(B):
        po = torch.empty((HKV, 1, mb[b]), dtype=bf, device=dev)
        pc = torch.empty_like(po)
        pool_old.nosa_pooling(score[:, b:b + 1].contiguous(), cis[:, b:b + 1].contiguous(), ncb[b], po, pc, stride=16,
                              block_size=64, local_blocks=LOCAL, init_blocks=INIT)
        ok = (beq(pq5[:, b, :mb[b]], po[:, 0]) and beq(pc5[:, b, :mb[b]], pc[:, 0])
              and bool((pq5[:, b, mb[b]:] == float("-inf")).all()) and bool((pc5[:, b, mb[b]:] == float("-inf")).all()))
        if not ok:
            bad5.append(b)
    torch.cuda.synchronize()
    gate("G5", not bad5, "%d requests, M_b %d .. %d, n_comp %d .. %d; mismatching rows %s"
         % (B, min(mb), max(mb), min(ncb), max(ncb), bad5[:5]))
    rq, rc = pool_new.pool_reference(score, cis, ncomp_t, mb_t, M_POOL, init_blocks=INIT, local_blocks=LOCAL)
    gate("G6", beq(rq, pq5) and beq(rc, pc5),
         "q: %s; cis: %s" % (_first_diff(rq, pq5) or "equal", _first_diff(rc, pc5) or "equal"))
    # ---- G7: topk(sorted=False) under a wider row (the pooled rows of G4 and the same rows + 64 -inf columns)
    pad = torch.full((HKV, B, 64), float("-inf"), dtype=bf, device=dev)
    wide = torch.cat([pq_old, pad], dim=-1)
    vals_ok = 0
    for k in (QK_SELECT, TOPK):
        t0 = torch.topk(pq_old, k, dim=-1, sorted=False)
        t1 = torch.topk(wide, k, dim=-1, sorted=False)
        vals_ok += int(beq(t0.values.sort(-1).values, t1.values.sort(-1).values))
        rows = int(t0.indices.shape[0] * t0.indices.shape[1])
        rec["report"]["k%d" % k] = dict(
            rows=rows,
            same_index_set=int((t0.indices.sort(-1).values == t1.indices.sort(-1).values).all(-1).sum()),
            same_raw_order=int((t0.indices == t1.indices).all(-1).sum()),
            ties_at_the_cut=int((t0.values.min(-1).values.unsqueeze(-1) == pq_old).sum(-1).gt(1).sum()))
    gate("G7", vals_ok == 2, "topk(sorted=False) of the width-%d rows and of the same rows padded to %d select the same "
         "multiset of VALUES at k %d and %d: %d of 2" % (M_POOL, M_POOL + 64, QK_SELECT, TOPK, vals_ok))
    rp = rec["report"]
    gate("G7-order", all(v["same_raw_order"] == v["rows"] and v["same_index_set"] == v["rows"] for v in rp.values()),
         "REPORT (the question): rows with the same index SET / the same RAW order / a tie at the cut: %s; the plan's "
         "mitigation (an explicit sort of the selected ids on both paths) makes the order equal, and only a tie at the "
         "cut can change the SET" % {k: (v["same_index_set"], v["same_raw_order"], v["ties_at_the_cut"], v["rows"])
                                     for k, v in rp.items()}, must=False)
    rec.update(failed=nfail, t1=time.time())
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("r2 kernels gate: %d MUST gate(s) failed" % nfail, flush=True)
    return nfail


if __name__ == "__main__":
    sys.exit(main())
