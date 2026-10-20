"""GPU gate of the M-INVARIANT linear (nosi/nosi/bi_linear.py; R2-M2 step 7, design note section 3.3 option G-a,
microgate B3). Holder step: benchmarks/Efficiency/bigemm_step.sh. CPU tests: benchmarks/Efficiency/tests_cpu/
test_bi_linear.py.

NOSA-8B's five decode linears (bi_linear.NOSA8B_SHAPES: wqkv, wo, gate_up, down, lm_head) with random bf16 weights
(no model is loaded: the gate is about shapes and bits), x (320, K) bf16.
MUST gates (each counted in the exit code), per linear:
  G1 semantics: 8 rows against the float64 product: max |d| <= 0.02 * max|ref| + 1e-3.
  G2 row invariance (torch.equal): bi_linear(x[:M]) == bi_linear(x)[:M] for M in MS; a permutation of the 320 rows;
     new neighbours (every row but every 5th replaced); a preallocated, larger, NaN-filled workspace.
  G3 negative control: a split-K chosen by M (no split up to 64 rows, 2 splits above) must differ between M = 64
     and M = 320; its PRECONDITION (split 1 and split 2 give different bits on the same rows) is checked first. If
     either fails, the invariance check is blind for this linear and the gate FAILS.
REPORT: G4 cuBLAS's own row variance (F.linear(x[:M]) against F.linear(x)[:M], rows that differ); the timing table
(median of 20 CUDA-event calls after 3 warm-ups): bi_linear at M, F.linear at the natural M, F.linear on rows padded
to M_FIX; per-step totals 32 x (wqkv + wo + gate_up + down) + lm_head against the registered prediction (PRED).
OUTPUT: --out DIR/bigemm_gate.json and .md; the last stdout line is BIGEMM_GATE_DONE failed=<n>; exit = n (2 = refused).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "nosi/nosi/bi_linear.py"
MS = (1, 64, 128, 192, 256, 320)
M_FIX = 320
LAYERS = 32
# REGISTERED PREDICTION (commit message): per-step GEMM time ratios, (central, low, high)
PRED_NATURAL = {1: (1.3, 0.9, 2.0), 64: (1.25, 0.9, 1.8), 128: (1.35, 0.95, 2.0), 192: (1.45, 1.0, 2.2),
                256: (1.5, 1.0, 2.3), 320: (1.5, 1.0, 2.4)}
PRED_PADDED = {64: (0.95, 0.6, 1.5), 256: (1.35, 0.9, 2.0)}


def load_module():
    spec = importlib.util.spec_from_file_location("bi_linear", str(MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print("[bigemm] REFUSED: %s exists" % args.out, flush=True)
        return 2
    os.makedirs(args.out)
    bl = load_module()
    dev = "cuda"
    bf = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(1234)
    gates, timing, cublas_var = [], [], []

    def gate(name, ok, detail):
        gates.append(dict(gate=name, passed=bool(ok), detail=detail))
        print("[bigemm] %s %s: %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)

    for name, (N, K) in bl.NOSA8B_SHAPES.items():
        cfg = bl.config_for(N, K)
        w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(bf)
        x = torch.randn(M_FIX, K, generator=g, device=dev).to(bf)
        full = bl.bi_linear(x, w)
        # G1
        ref = x[:8].double() @ w.double().t()
        d = float((full[:8].double() - ref).abs().max())
        lim = 0.02 * float(ref.abs().max()) + 1e-3
        gate("G1-semantics[%s]" % name, d <= lim, "max |bi - float64| %.3e (limit %.3e); config %s" % (d, lim, tuple(cfg)))
        # G2
        bad = [M for M in MS if not torch.equal(bl.bi_linear(x[:M], w), full[:M])]
        perm = torch.randperm(M_FIX, device=dev)
        if not torch.equal(bl.bi_linear(x[perm], w), full[perm]):
            bad.append("permutation")
        x2 = torch.randn(M_FIX, K, generator=g, device=dev).to(bf)
        x2[::5] = x[::5]
        if not torch.equal(bl.bi_linear(x2, w)[::5], full[::5]):
            bad.append("neighbours")
        if cfg.split_k > 1:
            ws = torch.full((cfg.split_k, 400, N), float("nan"), dtype=torch.float32, device=dev)
            if not torch.equal(bl.bi_linear(x[:200], w, workspace=ws), full[:200]):
                bad.append("workspace")
        gate("G2-row-invariance[%s]" % name, not bad, "M in %s, permutation, neighbours%s: differing %s" % (
            list(MS), ", workspace" if cfg.split_k > 1 else "", bad))
        # G3 negative control: an M-dependent split-K (1 at M <= 64, 2 above). PRECONDITION first: split 1 and split
        # 2 give different bits on the same 64 rows (K / 64 >= 64 chunks here, so the groupings differ; at 3 chunks
        # every in-order split is the same fold and a control is blind: CPU job 2190757)
        one, two = cfg._replace(split_k=1), cfg._replace(split_k=2)
        pre = not torch.equal(bl.bi_linear(x[:64], w, config=one), bl.bi_linear(x[:64], w, config=two))
        neg_small = bl.bi_linear(x[:64], w, config=one)
        neg_big = bl.bi_linear(x, w, config=two)[:64]
        caught = pre and not torch.equal(neg_small, neg_big)
        gate("G3-negative-split-caught[%s]" % name, caught,
             "precondition (split 1 and 2 differ on the rows): %s; the M-dependent split-K %s" % (
                 pre, "differs (detected)" if caught else "is NOT detected: the invariance check is blind"))
        # G4 cuBLAS report
        cf = F.linear(x, w)
        var = {M: int((F.linear(x[:M], w) != cf[:M]).any(dim=1).sum()) for M in MS}
        cublas_var.append(dict(linear=name, rows_differing_from_M320=var))
        print("[bigemm] G4-cublas-variance[%s] (report): rows of F.linear(x[:M]) not equal to F.linear(x)[:M]: %s"
              % (name, var), flush=True)
        # timing
        for M in MS:
            xm = x[:M].contiguous()
            out = torch.empty(M, N, dtype=bf, device=dev)
            ws = torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=dev) if cfg.split_k > 1 else None
            t_bi = time_ms(lambda: bl.bi_linear(xm, w, out=out, workspace=ws))
            t_cb = time_ms(lambda: F.linear(xm, w))
            t_pad = time_ms(lambda: F.linear(x, w))
            timing.append(dict(linear=name, N=N, K=K, M=M, bi_ms=t_bi, cublas_ms=t_cb, cublas_padded_ms=t_pad,
                               ratio_natural=t_bi / t_cb, ratio_padded=t_bi / t_pad))
            print("[bigemm] timing %s M=%d: bi %.4f ms, cuBLAS %.4f ms (natural), %.4f ms (padded to %d): ratios %.2f / %.2f"
                  % (name, M, t_bi, t_cb, t_pad, M_FIX, t_bi / t_cb, t_bi / t_pad), flush=True)
        del w, x, full, cf
        torch.cuda.empty_cache()

    steps = []
    for M in MS:
        def tot(key):
            s = 0.0
            for t in timing:
                if t["M"] == M:
                    s += t[key] * (1 if t["linear"] == "lm_head" else LAYERS)
            return s
        bi, cb, pad = tot("bi_ms"), tot("cublas_ms"), tot("cublas_padded_ms")
        cen, lo, hi = PRED_NATURAL[M]
        row = dict(M=M, bi_ms=bi, cublas_ms=cb, cublas_padded_ms=pad, ratio_natural=bi / cb, ratio_padded=bi / pad,
                   predicted_natural=cen, band_natural=[lo, hi], inside_natural=lo <= bi / cb <= hi)
        if M in PRED_PADDED:
            c2, l2, h2 = PRED_PADDED[M]
            row.update(predicted_padded=c2, band_padded=[l2, h2], inside_padded=l2 <= bi / pad <= h2)
        steps.append(row)
        print("[bigemm] per-step GEMMs M=%d: bi %.2f ms, cuBLAS %.2f ms, padded %.2f ms; ratio %.2f (predicted %.2f [%.2f, "
              "%.2f]: %s), padded ratio %.2f" % (M, bi, cb, pad, bi / cb, cen, lo, hi,
                                                 "inside" if lo <= bi / cb <= hi else "OUTSIDE", bi / pad), flush=True)

    nfail = sum(1 for x in gates if not x["passed"])
    lines = ["# M-invariant linear gate", "", "| gate | result | detail |", "|---|---|---|"]
    lines += ["| %s | %s | %s |" % (x["gate"], "PASS" if x["passed"] else "FAIL", x["detail"]) for x in gates]
    lines += ["", "| linear | M | bi ms | cuBLAS ms | cuBLAS padded ms | ratio | padded ratio |", "|---|---|---|---|---|---|---|"]
    lines += ["| %s | %d | %.4f | %.4f | %.4f | %.2f | %.2f |" % (t["linear"], t["M"], t["bi_ms"], t["cublas_ms"],
              t["cublas_padded_ms"], t["ratio_natural"], t["ratio_padded"]) for t in timing]
    lines += ["", "| M | per-step bi ms | cuBLAS ms | padded ms | ratio | predicted [band] | inside | padded ratio |",
              "|---|---|---|---|---|---|---|---|"]
    lines += ["| %d | %.2f | %.2f | %.2f | %.2f | %.2f [%.2f, %.2f] | %s | %.2f |" % (
        s["M"], s["bi_ms"], s["cublas_ms"], s["cublas_padded_ms"], s["ratio_natural"], s["predicted_natural"],
        s["band_natural"][0], s["band_natural"][1], s["inside_natural"], s["ratio_padded"]) for s in steps]
    lines += ["", "failed MUST gates: %d" % nfail]
    with open(os.path.join(args.out, "bigemm_gate.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "bigemm_gate.json"), "w") as f:
        json.dump(dict(gates=gates, failed=nfail, timing=timing, per_step=steps, cublas_variance=cublas_var,
                       configs={n: tuple(bl.config_for(*s)) for n, s in bl.NOSA8B_SHAPES.items()},
                       device=torch.cuda.get_device_name(0), torch=torch.__version__), f, indent=1)
    print("\n".join(lines), flush=True)
    print("BIGEMM_GATE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
