"""GPU cell bigc-tune: pick and FREEZE G-c's launch table (nosi/nosi/bi_linear_c.py). Holder step: r2_bi_step.sh ...
bigc-tune. Run it BEFORE bigc-gate and bigc-time; they then run on the frozen table (--table / --table-sha, or after the
table is committed into bi_linear_c.TABLE). CPU test: test_bi_linear_c.py (BC-T*).

For each of NOSA-8B's five decode linears (random bf16 weights, X (1148, K) N(0, 1)) every candidate launch of GRID
(BLOCK_M x BLOCK_N x warps x stages, GROUP_M 8, the contract fixed: BLOCK_K 64, split 1, ascending K) is timed
(CUDA-event median, --warm / --reps) at every M of TUNE_MS. A candidate that does not compile or launch (shared
memory) is recorded and skipped.
  strict (THE CANDIDATE): per shape the ONE launch that minimises 3 t(287) + t(861) (the dense' verify term's weights).
  bucket (RESEARCH ARM only; the relaxed law is NOT approved): per (shape, bucket) the launch that minimises the sum of
          t over the bucket's representative M (BUCKET_REPS).
CROSS-LAUNCH BITS (REPORT, evidence about Claim R, not a gate of the strict variant): every candidate's output at every
M against the source strict launch's output, bitwise; the count of differing (candidate, M) pairs is recorded. A
nonzero count means launch parameters DO change bits here: the bucket arm is then not batch invariant, and the table
records "bucket_invariant": false.
OUTPUT: --out DIR/tune.json (every timing, failures, mismatches), DIR/table.json (the frozen table, canonical JSON,
"tuned": true) and its digest on the line "BIGC_TABLE_SHA <hex>"; last line BIGC_TUNE_DONE failed=<n> (MUST: every shape
has >= 1 working candidate); exit = n (2 = refused). All times are measurements of this cell; any ratio derived from
them for the tick is a PREDICTION until bigc-time measures it.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "nosi/nosi/bi_linear_c.py"
TUNE_MS = (1, 16, 64, 256, 287, 861, 1148)
ROWS = 1148
GRID = dict(block_m=(16, 32, 64, 128), block_n=(32, 64, 128, 256), num_warps=(4, 8), num_stages=(3, 4))
GROUP_M = 8
STRICT_WEIGHTS = {287: 3.0, 861: 1.0}
# the representative M per bucket of TABLE["buckets"] = [16, 64, 256, 512] (+ the open bucket)
BUCKET_REPS = ((1, 16), (64,), (256,), (287,), (861, 1148))


def load_module():
    spec = importlib.util.spec_from_file_location("bi_linear_c_tune", str(MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidates(bl):
    out = []
    for bm in GRID["block_m"]:
        for bn in GRID["block_n"]:
            for nw in GRID["num_warps"]:
                for ns in GRID["num_stages"]:
                    out.append(bl.Launch(bm, bn, nw, ns, GROUP_M))
    return out


def pick(times, weights):
    """The launch (a key of times) with the least weighted sum over weights' M; launches missing an M are skipped.
    times: {launch: {M: ms}}. Returns (launch, score) or (None, None)."""
    best, score = None, None
    for l, t in times.items():
        if not all(M in t for M in weights):
            continue
        s = sum(w * t[M] for M, w in weights.items())
        if score is None or s < score or (s == score and list(l) < list(best)):
            best, score = l, s
    return best, score


def build_table(bl, prior, strict_pick, bucket_pick, bucket_invariant, note):
    t = json.loads(json.dumps(prior))
    t["tuned"] = True
    t["source"] = note
    t["strict"] = {k: list(v) for k, v in strict_pick.items()}
    t["shapes"] = {k: [list(e) for e in v] for k, v in bucket_pick.items()}
    t["bucket_invariant"] = bool(bucket_invariant)
    return bl.check_table(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("[bigc-tune] REFUSED: %s exists" % a.out, flush=True)
        return 2
    import torch
    bl = load_module()
    if len(BUCKET_REPS) != len(bl.TABLE["buckets"]) + 1 or any(
            bl.bucket_index(M, bl.TABLE["buckets"]) != i for i, reps in enumerate(BUCKET_REPS) for M in reps):
        print("[bigc-tune] REFUSED: BUCKET_REPS do not match TABLE['buckets']", flush=True)
        return 2
    os.makedirs(a.out)
    dev, bf = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def timed(fn):
        for _ in range(a.warm):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(a.reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            xs.append(e0.elapsed_time(e1))
        return statistics.median(xs)

    cands = candidates(bl)
    rec = dict(timings=[], failures=[], mismatches=[])
    strict_pick, bucket_pick, failed_shapes = {}, {}, []
    for name, (N, K) in bl.NOSA8B_SHAPES.items():
        key = bl.shape_key(N, K)
        w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(bf)
        X = torch.randn(ROWS, K, generator=g, device=dev).to(bf)
        ref = {M: bl.bi_c_with_launch(X[:M], w, bl.strict_launch_for(N, K)) for M in TUNE_MS}
        times = {}
        for l in cands:
            t = {}
            try:
                for M in TUNE_MS:
                    xm = X[:M].contiguous()
                    out = torch.empty(M, N, dtype=bf, device=dev)
                    bl.bi_c_with_launch(xm, w, l, out=out)
                    torch.cuda.synchronize()
                    n = int((out != ref[M]).any(dim=1).sum())
                    if n:
                        rec["mismatches"].append(dict(shape=name, launch=list(l), M=M, rows=n))
                    t[M] = timed(lambda: bl.bi_c_with_launch(xm, w, l, out=out))
                    rec["timings"].append(dict(shape=name, launch=list(l), M=M, ms=round(t[M], 5)))
            except Exception as e:                                          # noqa: BLE001 (out of resources etc.)
                rec["failures"].append(dict(shape=name, launch=list(l), error="%s: %s" % (type(e).__name__, str(e)[:160])))
                torch.cuda.synchronize()
                continue
            times[tuple(l)] = t
        sp, ss = pick(times, STRICT_WEIGHTS)
        if sp is None:
            failed_shapes.append(name)
            print("[bigc-tune] GATE TUNE-RAN[%s] FAIL: no working candidate" % name, flush=True)
            continue
        strict_pick[key] = sp
        bucket_pick[key] = [pick(times, {M: 1.0 for M in reps})[0] for reps in BUCKET_REPS]
        print("[bigc-tune] %s: strict %s (3 t(287) + t(861) = %.4f ms); buckets %s" % (name, list(sp), ss,
              [list(x) for x in bucket_pick[key]]), flush=True)
        del w, X, ref
        torch.cuda.empty_cache()
    nfail = len(failed_shapes)
    out = dict(meta=dict(cell="bigc-tune", tune_ms=list(TUNE_MS), grid=GRID, group_m=GROUP_M, strict_weights=STRICT_WEIGHTS,
                         bucket_reps=[list(r) for r in BUCKET_REPS], warm=a.warm, reps=a.reps, seed=a.seed,
                         prior_digest=bl.table_digest(bl.TABLE), device=torch.cuda.get_device_name(0),
                         torch=torch.__version__, triton=bl.describe("strict")["triton"],
                         commit=os.environ.get("BI_COMMIT", "unknown")), failed_shapes=failed_shapes, **rec)
    if not nfail:
        note = "bigc_tune %s on %s (torch %s, triton %s); cross-launch mismatches %d" % (
            os.environ.get("BI_COMMIT", "unknown"), out["meta"]["device"], torch.__version__, out["meta"]["triton"],
            len(rec["mismatches"]))
        table = build_table(bl, bl.TABLE, strict_pick, bucket_pick, not rec["mismatches"], note)
        with open(os.path.join(a.out, "table.json"), "w") as f:
            json.dump(table, f, sort_keys=True, separators=(",", ":"))
        out["table_digest"] = bl.table_digest(table)
        print("BIGC_TABLE_SHA %s" % out["table_digest"], flush=True)
    print("[bigc-tune] REPORT cross-launch bits: %d (shape, launch, M) triples differ from the source strict launch%s"
          % (len(rec["mismatches"]), "" if not rec["mismatches"] else ": launch parameters change bits here; the bucket "
             "arm is NOT batch invariant (table bucket_invariant false)"), flush=True)
    with open(os.path.join(a.out, "tune.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("BIGC_TUNE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
