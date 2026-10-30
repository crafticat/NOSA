"""RECON7 review V2 (read-only): independent parse of the Kineto traces of the profiled L4PTFDP passes.
Checks: device busy, streams, op overlap, H2D overlap, GEMM count / device ms / grids, copy kinds, sync calls.
CPU only (srun on amd)."""
import gzip
import json
import re
from collections import defaultdict

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
GPU = ("kernel", "gpu_memcpy", "gpu_memset")
GEMM_RX = re.compile(r"gemm|cutlass|xmma|cublas|gemv", re.I)
H2D_RX = re.compile(r"h2d", re.I)


def union_len(iv):
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def merged(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inter(A, B):
    i = j = 0
    t = 0.0
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if hi > lo:
            t += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return t


def one(run, g, slot, B):
    p = R + run + "/trace_L4PTFDP_g%d_%s.json.gz" % (g, slot)
    with gzip.open(p, "rb") as f:
        tr = json.loads(f.read().decode("utf-8"))
    X = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
    ann = [e for e in X if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("i4c.")
           and "." not in str(e.get("name"))[4:]]
    names = defaultdict(int)
    for e in ann:
        names[e["name"]] += 1
    rng = [e for e in ann if e["name"] == "i4c." + slot]
    print("=" * 10, run, "g%d" % g, slot, "top-level i4c annotations", dict(names))
    h0 = float(rng[0]["ts"])
    h1 = h0 + float(rng[0]["dur"])
    rt = {}
    rtnames = defaultdict(lambda: [0, 0.0])
    for e in X:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            ts = float(e["ts"])
            if c is not None:
                rt[c] = ts
            if h0 <= ts <= h1:
                rtnames[e["name"]][0] += 1
                rtnames[e["name"]][1] += float(e.get("dur", 0))
    gall = [e for e in X if e.get("cat") in GPU]
    sel = [e for e in gall if h0 <= rt.get((e.get("args") or {}).get("correlation"), -1) <= h1]
    d0 = min(float(e["ts"]) for e in sel)
    d1 = max(float(e["ts"]) + float(e.get("dur", 0)) for e in sel)
    inwin = [e for e in gall if float(e["ts"]) >= d0 - 1e-3 and float(e["ts"]) + float(e.get("dur", 0)) <= d1 + 1e-3]
    print("  host range %.3f ms; GPU ops launched in range %d; GPU ops in device window %d (all GPU ops in file %d);"
          " device window %.3f ms" % ((h1 - h0) / 1e3, len(sel), len(inwin), len(gall), (d1 - d0) / 1e3))
    iv = [(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in inwin]
    busy = union_len(iv)
    sdur = sum(b - a for a, b in iv)
    streams = defaultdict(lambda: [0, 0.0])
    for e in inwin:
        k = (e.get("pid"), e.get("tid"), (e.get("args") or {}).get("stream"))
        streams[k][0] += 1
        streams[k][1] += float(e.get("dur", 0)) / 1e3
    print("  busy %.3f ms; sum of durations %.3f ms; overlap (sum - busy) %.4f ms; idle in window %.3f ms" % (
        busy / 1e3, sdur / 1e3, (sdur - busy) / 1e3, (d1 - d0 - busy) / 1e3))
    print("  streams (pid, tid, args.stream): ", {str(k): (v[0], round(v[1], 3)) for k, v in streams.items()})
    h2d = [e for e in inwin if e.get("cat") == "kernel" and H2D_RX.search(e["name"])]
    oth = [e for e in inwin if not (e.get("cat") == "kernel" and H2D_RX.search(e["name"]))]
    mh = merged([(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in h2d])
    mo = merged([(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in oth])
    print("  H2D-named kernels %d, %.3f ms; overlap with other ops %.4f ms" % (
        len(h2d), sum(float(e.get("dur", 0)) for e in h2d) / 1e3, inter(mh, mo) / 1e3))
    hn = defaultdict(lambda: [0, 0.0, set()])
    for e in h2d:
        k = e["name"][:60]
        hn[k][0] += 1
        hn[k][1] += float(e.get("dur", 0)) / 1e3
        hn[k][2].add(str((e.get("args") or {}).get("grid")))
    for k, v in hn.items():
        print("    h2d kernel %-60s n %d ms %.3f grids %s" % (k, v[0], v[1], sorted(v[2])[:4]))
    cps = defaultdict(lambda: [0, 0, 0.0])
    for e in inwin:
        if e.get("cat") != "kernel":
            k = (e.get("cat"), e["name"][:40])
            cps[k][0] += 1
            cps[k][1] += int((e.get("args") or {}).get("bytes") or 0)
            cps[k][2] += float(e.get("dur", 0)) / 1e3
    print("  copies / memsets:", {"%s|%s" % k: (v[0], v[1], round(v[2], 3)) for k, v in cps.items()})
    gm = [e for e in inwin if e.get("cat") == "kernel" and GEMM_RX.search(e["name"])]
    print("  GEMM-named kernels %d, device %.3f ms" % (len(gm), sum(float(e.get("dur", 0)) for e in gm) / 1e3))
    ag = defaultdict(lambda: [0, 0.0])
    for e in gm:
        a = e.get("args") or {}
        k = (re.sub(r"<.*", "", e["name"])[:70], str(a.get("grid")), str(a.get("block")))
        ag[k][0] += 1
        ag[k][1] += float(e.get("dur", 0)) / 1e3
    for k, v in sorted(ag.items(), key=lambda kv: -kv[1][1]):
        m = re.search(r"_(\d+)x(\d+)_", k[0])
        tiles = (int(m.group(1)), int(m.group(2))) if m else None
        print("    %-70s grid %-16s block %-14s n %4d ms %8.3f tile %s" % (k[0], k[1], k[2], v[0], v[1], tiles))
    syncs = {k: v for k, v in rtnames.items() if re.search(r"Synchron|StreamQuery|EventQuery|cudaMemcpy$|DtoH", k)}
    print("  sync-like runtime calls in host range:", {k: (v[0], round(v[1] / 1e3, 3)) for k, v in syncs.items()})
    launches = {k: v[0] for k, v in rtnames.items() if "aunch" in k or "Memcpy" in k or "Memset" in k}
    print("  launch-like runtime calls:", launches)
    big = defaultdict(lambda: [0, 0.0])
    for e in inwin:
        if e.get("cat") != "kernel":
            continue
        n = e["name"]
        if "unified_attention" in n:
            k = "fork_attention"
        elif GEMM_RX.search(n):
            k = "GEMM"
        elif H2D_RX.search(n):
            k = "H2D"
        elif re.search(r"flash_fwd|flash_attn", n, re.I):
            k = "flash_attention"
        else:
            k = "other"
        big[k][0] += 1
        big[k][1] += float(e.get("dur", 0)) / 1e3
    print("  coarse classes:", {k: (v[0], round(v[1], 3)) for k, v in big.items()})
    return busy / 1e3


for run, B in (("i4c_e4a_2195576_i4ca", 216), ("i4c_c4b_2195576_i4cb", 318)):
    for g in (112, 96):
        for slot in ("joint", "draft0", "draft1"):
            try:
                one(run, g, slot, B)
            except Exception as ex:
                print("FAIL", run, g, slot, type(ex).__name__, ex)
