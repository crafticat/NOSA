"""GPU cell bigc-gate: batch invariance of G-c (nosi/nosi/bi_linear_c.py; design note
/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/bi-gemm-design.md section 5.2). Holder step: r2_bi_step.sh ... bigc-gate.
CPU tests: benchmarks/Efficiency/tests_cpu/test_bi_linear_c.py.

VARIANTS (--variants, default strict,bucket):
  strict  THE CANDIDATE (one launch per weight shape for every M): its gates are MUST.
  bucket  RESEARCH ARM (launch chosen by M bucket; the relaxed law is NOT approved): the same checks, all REPORT (R_*),
          never cited as satisfying the contract.
NOSA-8B's five decode linears (wqkv, wo, gate_up, down, lm_head), random bf16 weights (N(0, 1) x 0.02), X (1148, K):
input sets "randn" (N(0, 1)), "outlier" (randn, 8 channels x 50) and "planted" (randn with order-sensitive rows 0-3
x cols 0-7: bi_linear_c.plant_order_rows; the controls' input set, lesson L34).

GATES (per shape; MUST for strict):
  G1   semantics: 8 rows against the float64 product, max |d| <= 0.02 max|ref| + 1e-3.
  G2   row equality across M (bitwise): f(X[:M]) == f(X)[:M] for M in MS, every input set.
  G2w  windows: f(X[a:a+M]) == f(X)[a:a+M], a in {1, 63, 129}, M in {7, 65, 287}; one row permutation of all 1148.
  G2g  CUDA graph: a captured call at M 287 and 861 == the eager call == f(X)[:M], bitwise (randn).
  G3a  NEGATIVE CONTROL, split-K 2 chosen by M (M > 64): PRECONDITION first (kernel vs kernel on the planted rows:
       split 2 != contract at M 7); then the same row checker as G2 must report >= 1 differing row. A control whose
       precondition fails is BLIND and the gate FAILS. On randn the precondition is REPORTED (effective / no-op).
  G3b  NEGATIVE CONTROL, natural cuBLAS (F.linear): PRECONDITION: the profiler shows >= 2 kernel names over MS; then
       the checker reports >= 1 differing row (randn).
  G3c  NEGATIVE CONTROL, K reversed chosen by M (M > 64): as G3a.
REPORTS: R1 BLOCK_K 32 vs 64 bits at M 287 (expected equal; NOT a control). R3 PTX of the strict launch: count of
mma.sync.aligned.m16n8k16 and of red. / atom. lines. R4 strict vs bucket bits at every M (launch-parameter
invariance; a measurement, not an approval). R5 the REFERENCE CHANGE: rows of G-c strict at M 1148 that differ from
F.linear (cuBLAS, G-b's arithmetic), with the max |d|: G-c is a NEW target reference.

WHAT A PASS PROVES: for these five shapes, these M, windows and permutation, these three input sets, this GPU, this
torch / Triton build and this table digest, the strict variant's rows are bitwise independent of the row count and the
row position, and the checker sees the three planted M-dependences. It does NOT prove other M, other inputs, another
Triton version, or equality to the G-b reference (R5 measures that they differ).
OUTPUT: --out DIR/bigc_gate.json; last stdout line BIGC_GATE_DONE failed=<n>; exit = n (2 = refused).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "nosi/nosi/bi_linear_c.py"
MS = (1, 7, 16, 17, 64, 65, 256, 257, 287, 861, 1148)
ROWS = 1148
WINDOWS = ((1, 7), (63, 65), (129, 287), (1, 287), (63, 7), (129, 65))
GRAPH_MS = (287, 861)
SPLIT_AT = 64                     # the controls change their order above this row count
PLANT_ROWS = (0, 1, 2, 3)
PLANT_COLS = tuple(range(8))
VARIANTS = ("strict", "bucket")


def load_module():
    spec = importlib.util.spec_from_file_location("bi_linear_c_gate", str(MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row_check(f, X, ms=MS, windows=WINDOWS, perm=None):
    """THE CHECKER (the same for the variants and the controls): {case: rows differing} for f(X[:M]) vs f(X)[:M],
    windows f(X[a:a+M]) vs f(X)[a:a+M] and a permutation. Empty = invariant on these cases."""
    import torch
    full = f(X)
    bad = {}
    for M in ms:
        n = int((f(X[:M]) != full[:M]).any(dim=1).sum())
        if n:
            bad["M%d" % M] = n
    for a, M in windows:
        n = int((f(X[a:a + M]) != full[a:a + M]).any(dim=1).sum())
        if n:
            bad["w%d+%d" % (a, M)] = n
    if perm is not None:
        n = int((f(X[perm]) != full[perm]).any(dim=1).sum())
        if n:
            bad["perm"] = n
    del full
    torch.cuda.empty_cache()
    return bad


def kernel_names(fn):
    """CUDA kernel names of one call (torch.profiler). None when the profiler is unavailable."""
    import torch
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        cuda = getattr(torch.autograd, "DeviceType").CUDA
        return sorted({e.name for e in p.events() if getattr(e, "device_type", None) == cuda})
    except Exception as e:                                                  # noqa: BLE001 (a report, not a gate)
        print("[bigc] profiler unavailable: %s" % e, flush=True)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="strict,bucket")
    ap.add_argument("--table", default=None, help="a frozen table file (bigc_tune.py's table.json)")
    ap.add_argument("--table-sha", default=None, help="its canonical digest (bi_linear_c.table_digest)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    variants = [v for v in a.variants.split(",") if v]
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        print("[bigc] REFUSED: --variants in %s" % (VARIANTS,), flush=True)
        return 2
    if (a.table is None) != (a.table_sha is None):
        print("[bigc] REFUSED: --table and --table-sha go together", flush=True)
        return 2
    if os.path.exists(a.out):
        print("[bigc] REFUSED: %s exists" % a.out, flush=True)
        return 2
    import torch
    import torch.nn.functional as F
    bl = load_module()
    if a.table:
        bl.use_table(bl.load_table(a.table, a.table_sha))
    os.makedirs(a.out)
    dev, bf = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(a.seed)
    fns = dict(strict=bl.bi_c_linear_strict, bucket=bl.bi_c_linear_bucket)
    gates, reports = [], []

    def gate(name, ok, must, detail):
        gates.append(dict(gate=name, passed=bool(ok), must=bool(must), detail=detail))
        print("[bigc] %s %s %s: %s" % ("GATE" if must else "REPORT", name, "PASS" if ok else "FAIL", detail), flush=True)

    def report(name, detail):
        reports.append(dict(report=name, detail=detail))
        print("[bigc] REPORT %s: %s" % (name, json.dumps(detail, default=str)), flush=True)

    perm = torch.randperm(ROWS, device=dev, generator=g)
    for name, (N, K) in bl.NOSA8B_SHAPES.items():
        w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(bf)
        Xr = torch.randn(ROWS, K, generator=g, device=dev).to(bf)
        Xo = Xr.clone()
        Xo[:, torch.randperm(K, device=dev, generator=g)[:8]] *= 50
        Xp, wp = Xr.clone(), w.clone()
        bl.plant_order_rows(Xp, wp, PLANT_ROWS, PLANT_COLS)
        sets = dict(randn=(Xr, w), outlier=(Xo, w), planted=(Xp, wp))
        strict = bl.strict_launch_for(N, K)
        for v in variants:
            f = fns[v]
            must = v == "strict"
            tag = ("" if must else "R_") + "%s[%s:%s]"
            full = f(Xr, w)
            ref = Xr[:8].double() @ w.double().t()
            d = float((full[:8].double() - ref).abs().max())
            lim = 0.02 * float(ref.abs().max()) + 1e-3
            gate(tag % ("G1-semantics", v, name), d <= lim, must, "max |bi - float64| %.3e (limit %.3e)" % (d, lim))
            bad = {s: row_check(lambda x, ww=ww: f(x, ww), X, ms=MS, windows=(), perm=None) for s, (X, ww) in sets.items()}
            bad = {s: b for s, b in bad.items() if b}
            gate(tag % ("G2-rows-across-M", v, name), not bad, must, "M in %s, sets randn / outlier / planted: "
                 "differing %s" % (list(MS), bad or "none"))
            badw = row_check(lambda x: f(x, w), Xr, ms=(), windows=WINDOWS, perm=perm)
            gate(tag % ("G2w-windows-perm", v, name), not badw, must, "windows %s + a permutation: differing %s"
                 % (list(WINDOWS), badw or "none"))
            gbad = {}
            for M in GRAPH_MS:
                xs = Xr[:M].clone()
                oe = torch.empty(M, N, dtype=bf, device=dev)
                og = torch.empty(M, N, dtype=bf, device=dev)
                f(xs, w, out=oe)                                       # compile + warm outside the capture
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    f(xs, w, out=og)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    f(xs, w, out=og)
                og.zero_()
                graph.replay()
                torch.cuda.synchronize()
                n = int((og != oe).any(dim=1).sum()) + int((og != full[:M]).any(dim=1).sum())
                if n:
                    gbad["M%d" % M] = n
                del graph, xs, oe, og
            gate(tag % ("G2g-cuda-graph", v, name), not gbad, must, "captured vs eager vs f(X)[:M] at M %s: differing %s"
                 % (list(GRAPH_MS), gbad or "none"))
            del full
        # ---- negative controls (on the strict launch; the checker is row_check) ----
        def mdep(control):
            def f(x, ww):
                return control(x, ww, strict) if x.shape[0] > SPLIT_AT else bl.bi_c_linear_strict(x, ww)
            return f
        for gid, cname, control in (("G3a", "split-K-2", lambda x, ww, l: bl.control_split_k(x, ww, l, 2)),
                                    ("G3c", "K-reversed", bl.control_k_reversed)):
            pre = {}
            for s in ("planted", "randn"):
                X, ww = sets[s]
                pre[s] = not torch.equal(control(X[:7], ww, strict), bl.bi_c_linear_strict(X[:7], ww))
            seen = {}
            if pre["planted"]:
                X, ww = sets["planted"]
                seen = row_check(mdep(control), X, ms=(7,), windows=(), perm=None)
            ok = pre["planted"] and bool(seen)
            gate("%s-negative-%s-caught[%s]" % (gid, cname, name), ok, True,
                 "precondition planted (control != contract at M 7): %s; randn: %s; the checker %s" % (
                     pre["planted"], "effective" if pre["randn"] else "NO-OP on randn (reported, not counted)",
                     ("reports %s (DETECTED)" % seen) if seen else "reports nothing: BLIND"))
        names = {M: kernel_names(lambda M=M: F.linear(Xr[:M], w)) for M in MS}
        distinct = sorted({n for v in names.values() if v for n in v})
        pre = all(v is not None for v in names.values()) and len({tuple(v) for v in names.values()}) >= 2
        seen = row_check(lambda x: F.linear(x, w), Xr, ms=MS, windows=(), perm=None) if pre else {}
        gate("G3b-negative-natural-cuBLAS-caught[%s]" % name, pre and bool(seen), True,
             "precondition (>= 2 kernel sets over M): %s (%d distinct names); the checker %s" % (
                 pre, len(distinct), ("reports %s (DETECTED)" % seen) if seen else "reports nothing: BLIND"))
        report("G3b-kernel-names[%s]" % name, {str(M): v for M, v in names.items()})
        # ---- reports ----
        x287 = Xr[:287]
        r1 = torch.equal(bl.bi_c_with_launch(x287, w, strict, block_k=32), bl.bi_c_with_launch(x287, w, strict))
        report("R1-blockk32-vs-64[%s]" % name, dict(equal=r1, note="expected equal; NOT a control"))
        try:
            o2 = torch.empty(287, N, dtype=bf, device=dev)
            h = bl._launch(x287.contiguous(), w, o2, strict, 0, K)
            ptx = h.asm.get("ptx", "") if hasattr(h, "asm") else ""
            report("R3-ptx[%s]" % name, dict(mma_m16n8k16=ptx.count("mma.sync.aligned.m16n8k16"),
                                             red_or_atom_lines=sum(1 for ln in ptx.splitlines()
                                                                   if "red." in ln or "atom." in ln),
                                             ptx_bytes=len(ptx)))
        except Exception as e:                                              # noqa: BLE001
            report("R3-ptx[%s]" % name, dict(error=str(e)[:200]))
        if "bucket" in variants and "strict" in variants:
            diff = {}
            for M in MS:
                n = int((bl.bi_c_linear_bucket(Xr[:M], w) != bl.bi_c_linear_strict(Xr[:M], w)).any(dim=1).sum())
                if n:
                    diff["M%d" % M] = n
            report("R4-strict-vs-bucket-bits[%s]" % name, dict(differing=diff, launches={
                str(M): list(bl.launch_for(N, K, M)) for M in MS}, strict=list(strict)))
        ys, yc = bl.bi_c_linear_strict(Xr, w), F.linear(Xr, w)
        report("R5-reference-change-vs-cuBLAS[%s]" % name, dict(
            rows_differing=int((ys != yc).any(dim=1).sum()), of=ROWS,
            max_abs=float((ys.float() - yc.float()).abs().max()), note=bl.REFERENCE_NOTE))
        del w, Xr, Xo, Xp, wp, sets, ys, yc
        torch.cuda.empty_cache()

    nfail = sum(1 for x in gates if x["must"] and not x["passed"])
    meta = dict(cell="bigc-gate", variants=variants, ms=list(MS), windows=list(WINDOWS), graph_ms=list(GRAPH_MS),
                rows=ROWS, seed=a.seed, table=a.table, table_sha_pinned=a.table_sha,
                describe={v: bl.describe(v) for v in variants}, device=torch.cuda.get_device_name(0),
                torch=torch.__version__, commit=os.environ.get("BI_COMMIT", "unknown"),
                proves=("strict: rows bitwise independent of row count and position for these shapes / M / windows / "
                        "inputs / GPU / build / table digest, with the three planted M-dependences detected; NOT other "
                        "M or inputs, NOT another Triton, NOT equality to the G-b reference (R5)"))
    with open(os.path.join(a.out, "bigc_gate.json"), "w") as f:
        json.dump(dict(meta=meta, gates=gates, reports=reports, failed=nfail), f, indent=1, default=str)
    print("BIGC_GATE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
