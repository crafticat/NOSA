"""RECON7 R2 part 2 (read-only): work counts and the device timeline of the profiled passes of L4PTFDP (I4C cells;
joint pass = I3's joint code path, NOTES: core files identical in nosi-r2-i3 / -i4c / -i4dc). Per trace: clock base,
host range, device window, streams, classes, GEMM / attention / selection / H2D kernels with grid and block, memcpy
bytes, runtime calls (launches, syncs), busy / idle, idle split host-exposed vs device-side, concurrency (overlap of
H2D-class ops with other ops), and (labelled rounds) per label device ms, spans and gaps. Run inside srun on amd."""
import bisect
import gzip
import json
import re
import sys
from collections import defaultdict

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
RUNS = ["i4c_e4a_2195576_i4ca", "i4c_c3a_2195575_i4ca", "i4c_c4b_2195576_i4cb"]
GPU = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH = ("cuda_runtime", "cuda_driver")
LP = "i4cL."
RULES = (
    ("TWINF", r"_twin_kernel"), ("BODYF", r"_body_kernel"), ("STAGE1", r"splitkv_stage1|stage1_kernel|infllm"),
    ("ATTN", r"flash_fwd|flash_attn"), ("GEMM", r"gemm|gemv|xmma|cutlass|cublas"), ("POOL", r"pool"),
    ("SELECT", r"topk|sort|radix"), ("H2D", r"h2d"), ("DIFF", r"diff"), ("WB", r"_wb_kernel|writeback"),
    ("FUSED", r"rmsnorm|rotary|rope|act_and_mul|silu|nosa_linear"),
    ("SMALL", r"elementwise|index|reduce|catarray|copy|fill|where|gather|scatter|arange|clamp|compare"),
)
SYNC_RX = r"Synchronize|cudaMemcpy$|cudaMemcpy\b(?!Async)|cuMemcpyDtoH|cudaStreamQuery"


def cls(name, cat):
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "COPY"
    for c, rx in RULES:
        if re.search(rx, name, re.IGNORECASE):
            return c
    return "OTHER"


def short(name, n=70):
    s = name
    for pat, rep in ((r"void at::native::\(anonymous namespace\)::", ""), (r"void at::native::", ""),
                     (r"void at_cuda_detail::cub::", "cub::"), (r"<.*", "")):
        s = re.sub(pat, rep, s)
    return s[:n]


def ranges(X, prefix):
    by = {}
    for e in X:
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith(prefix):
            s = float(e["ts"])
            by.setdefault((e.get("pid"), e.get("tid")), []).append((s, s + float(e.get("dur", 0.0)), e["name"]))
    out = {}
    for k, rs in by.items():
        rs.sort()
        out[k] = ([r[0] for r in rs], rs)
    return out


def innermost(index, pid, tid, ts):
    if (pid, tid) not in index:
        return None
    starts, rs = index[(pid, tid)]
    best = None
    j = bisect.bisect_right(starts, ts) - 1
    for k in range(j, max(-1, j - 64), -1):
        s, t, nm = rs[k]
        if s <= ts <= t and (best is None or t - s < best[1] - best[0]):
            best = (s, t, nm)
    return best


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def inter(a, b):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyse(path, slot):
    with gzip.open(path, "rb") as f:
        tr = json.loads(f.read().decode("utf-8"))
    meta = {k: (v if len(repr(v)) < 300 else "<%d chars>" % len(repr(v))) for k, v in tr.items() if k != "traceEvents"}
    ev = tr.get("traceEvents", [])
    X = [e for e in ev if e.get("ph") == "X"]
    rng = [e for e in X if e.get("cat") == "user_annotation" and e.get("name") == "i4c." + slot]
    h0 = float(rng[0]["ts"]) if rng else None
    h1 = h0 + float(rng[0].get("dur", 0.0)) if rng else None
    pidx = ranges(X, "i4c." + slot)
    lidx = ranges(X, LP)
    launch = {}
    rt = defaultdict(lambda: [0, 0.0])
    for e in X:
        if e.get("cat") in LAUNCH:
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launch[c] = (e.get("pid"), e.get("tid"), float(e["ts"]), float(e.get("dur", 0.0)))
            if h0 is not None and h0 <= float(e["ts"]) <= h1:
                x = rt[str(e.get("name", ""))]
                x[0] += 1
                x[1] += float(e.get("dur", 0.0))
    ops = []
    for e in X:
        cat = e.get("cat")
        if cat not in GPU:
            continue
        a = e.get("args") or {}
        c = a.get("correlation")
        lab = None
        inr = False
        lts = lend = None
        if c in launch:
            pid, tid, ts, dur = launch[c]
            lts, lend = ts, ts + dur
            inr = innermost(pidx, pid, tid, ts) is not None
            b = innermost(lidx, pid, tid, ts)
            if b is not None:
                lab = b[2][len(LP):]
        name = str(e.get("name", ""))
        ops.append(dict(ts=float(e["ts"]), dur=float(e.get("dur", 0.0)), name=name, cat=cat, cls=cls(name, cat),
                        stream=(e.get("pid"), e.get("tid")), grid=tuple(a.get("grid", ()) or ()),
                        block=tuple(a.get("block", ()) or ()), bytes=a.get("bytes"), inr=inr, lab=lab, lts=lts,
                        lend=lend, regs=a.get("registers per thread"), occ=a.get("est. achieved occupancy %")))
    inr = [o for o in ops if o["inr"]]
    if not inr:
        print("  no ops in range")
        return None
    d0 = min(o["ts"] for o in inr)
    d1 = max(o["ts"] + o["dur"] for o in inr)
    win = sorted([o for o in ops if o["ts"] >= d0 and o["ts"] + o["dur"] <= d1 + 1e-3], key=lambda o: o["ts"])
    mg = merge([(o["ts"], o["ts"] + o["dur"]) for o in win])
    busy = sum(b - a for a, b in mg)
    sumdur = sum(o["dur"] for o in win)
    # gaps: host-exposed (the op that ends the gap was launched after the device became free) vs device-side
    endmax = win[0]["ts"]
    gap = hexp = 0.0
    big = []
    gl = defaultdict(lambda: [0.0, 0.0, 0])
    for o in win:
        g = o["ts"] - endmax
        if g > 0:
            le = o["lend"] if o["lend"] is not None else endmax
            h = min(g, max(0.0, le - endmax))
            gap += g
            hexp += h
            k = (o["lab"] or "?").rpartition("@")[0] or (o["lab"] or "?")
            gl[k][0] += g
            gl[k][1] += h
            gl[k][2] += 1
            if g >= 1000.0:
                big.append((round((endmax - d0) / 1e3, 3), round(g / 1e3, 3), k, short(o["name"], 40)))
        endmax = max(endmax, o["ts"] + o["dur"])
    print("  META", meta)
    print("  host range ts %.1f dur_ms %.4f; device window [%.1f, %.1f] ms %.4f; start-after-host %.4f ms; "
          "end-after-host %.4f ms" % (h0, (h1 - h0) / 1e3, d0, d1, (d1 - d0) / 1e3, (d0 - h0) / 1e3, (d1 - h1) / 1e3))
    print("  ops %d busy_ms %.4f sum_dur_ms %.4f (concurrency %.4f ms) idle_ms %.4f = host-exposed %.4f + device-side "
          "%.4f; gaps >= 1 ms: %s" % (len(win), busy / 1e3, sumdur / 1e3, (sumdur - busy) / 1e3, gap / 1e3, hexp / 1e3,
                                       (gap - hexp) / 1e3, big[:8]))
    st = defaultdict(lambda: [0, 0.0])
    for o in win:
        st[o["stream"]][0] += 1
        st[o["stream"]][1] += o["dur"]
    print("  streams", {str(k): (v[0], round(v[1] / 1e3, 4)) for k, v in st.items()})
    h2d = merge([(o["ts"], o["ts"] + o["dur"]) for o in win if o["cls"] == "H2D"])
    oth = merge([(o["ts"], o["ts"] + o["dur"]) for o in win if o["cls"] != "H2D"])
    print("  H2D-class busy %.4f ms; other busy %.4f ms; H2D overlap with other ops %.4f ms" % (
        sum(b - a for a, b in h2d) / 1e3, sum(b - a for a, b in oth) / 1e3, inter(h2d, oth) / 1e3))
    cl = defaultdict(lambda: [0, 0.0])
    for o in win:
        cl[o["cls"]][0] += 1
        cl[o["cls"]][1] += o["dur"]
    print("  classes", {k: (v[0], round(v[1] / 1e3, 4)) for k, v in sorted(cl.items(), key=lambda kv: -kv[1][1])})
    ag = defaultdict(lambda: [0, 0.0, None, None])
    for o in win:
        if o["cls"] in ("GEMM", "ATTN", "OTHER", "STAGE1", "POOL", "SELECT", "H2D", "TWINF", "DIFF", "FUSED", "COPY"):
            k = (o["cls"], short(o["name"], 60), o["grid"], o["block"])
            x = ag[k]
            x[0] += 1
            x[1] += o["dur"]
            x[2] = o["regs"]
            x[3] = o["occ"]
    print("  KERNELS (class, name, grid, block): count, ms, us each, regs, est occupancy %")
    for k, v in sorted(ag.items(), key=lambda kv: -kv[1][1])[:40]:
        print("    %-6s %-60s %-16s %-12s %5d %9.4f %8.2f %s %s" % (k[0], k[1], k[2], k[3], v[0], v[1] / 1e3,
                                                                   v[1] / v[0], v[2], v[3]))
    cp = defaultdict(lambda: [0, 0, 0.0])
    for o in win:
        if o["cat"] != "kernel":
            k = (o["cat"], short(o["name"], 50))
            cp[k][0] += 1
            cp[k][1] += int(o["bytes"] or 0)
            cp[k][2] += o["dur"]
    print("  COPIES (cat, name): count, bytes, ms", {"%s|%s" % k: (v[0], v[1], round(v[2] / 1e3, 4))
                                                      for k, v in cp.items()})
    sync = {k: (v[0], round(v[1] / 1e3, 4)) for k, v in rt.items() if re.search(SYNC_RX, k)}
    print("  RUNTIME calls in the host range:", {k: (v[0], round(v[1] / 1e3, 4)) for k, v in
                                                 sorted(rt.items(), key=lambda kv: -kv[1][0])[:14]})
    print("  SYNC-like runtime calls:", sync)
    print("  idle by the label of the op that ends the gap (gap ms, host-exposed ms, n):",
          {k: (round(v[0] / 1e3, 4), round(v[1] / 1e3, 4), v[2]) for k, v in sorted(gl.items(), key=lambda kv: -kv[1][0])})
    labs = [o for o in win if o["lab"]]
    if labs:
        per = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        inst = defaultdict(list)
        for o in labs:
            lab, _, lay = o["lab"].rpartition("@")
            per[lab][o["cls"]][0] += 1
            per[lab][o["cls"]][1] += o["dur"]
            inst[(lab, int(lay))].append(o)
        print("  LABELS: device ms per class; spans (first op start .. last op end, summed over instances); busy in "
              "spans")
        for lab in sorted(per):
            tot = sum(v[1] for v in per[lab].values())
            n = sum(v[0] for v in per[lab].values())
            sp = bs = 0.0
            for (lb, lay), os_ in inst.items():
                if lb != lab:
                    continue
                a = min(o["ts"] for o in os_)
                b = max(o["ts"] + o["dur"] for o in os_)
                sp += b - a
                bs += sum(y - x for x, y in merge([(o["ts"], o["ts"] + o["dur"]) for o in os_]))
            print("    %-7s ops %5d dev_ms %8.4f span_ms %8.4f busy_in_span %8.4f | %s" % (
                lab, n, tot / 1e3, sp / 1e3, bs / 1e3,
                ", ".join("%s %d/%.3f" % (c, v[0], v[1] / 1e3) for c, v in sorted(per[lab].items()))))
        # one middle layer, label by label: start offset, span, device ms
        lay = 5
        seq = sorted({k for k in inst if k[1] == lay}, key=lambda k: min(o["ts"] for o in inst[k]))
        t0 = min(min(o["ts"] for o in inst[k]) for k in seq) if seq else 0.0
        print("  LAYER %d sequence (label: start offset ms, span ms, device ms, ops)" % lay)
        for k in seq:
            os_ = inst[k]
            a = min(o["ts"] for o in os_)
            b = max(o["ts"] + o["dur"] for o in os_)
            print("    %-7s %8.4f %8.4f %8.4f %4d" % (k[0], (a - t0) / 1e3, (b - a) / 1e3,
                                                     sum(o["dur"] for o in os_) / 1e3, len(os_)))
        gm = [o for o in win if o["cls"] == "GEMM"][:6]
        print("  first GEMMs (label, grid, block, us):", [(o["lab"], o["grid"], o["block"], round(o["dur"], 1))
                                                          for o in gm])
    return dict(h0=h0, h1=h1, d0=d0, d1=d1, busy=busy, gap=gap, hexp=hexp)


def main():
    for run in RUNS:
        for g in (112, 96):
            tl = {}
            for slot in ("joint", "draft0", "draft1"):
                p = R + run + "/trace_L4PTFDP_g%d_%s.json.gz" % (g, slot)
                print("=" * 24, run, "g%d" % g, slot)
                try:
                    tl[slot] = analyse(p, slot)
                except Exception as ex:
                    print("  FAIL", type(ex).__name__, ex)
            ok = {k: v for k, v in tl.items() if v}
            if len(ok) == 3:
                j, a, b = ok["joint"], ok["draft0"], ok["draft1"]
                print("STITCH %s g%d: device windows on one clock? joint [%.1f, %.1f], draft0 [%.1f, %.1f], draft1 "
                      "[%.1f, %.1f]; joint end -> draft0 start %.3f ms; draft0 end -> draft1 start %.3f ms" % (
                          run, g, j["d0"], j["d1"], a["d0"], a["d1"], b["d0"], b["d1"], (a["d0"] - j["d1"]) / 1e3,
                          (b["d0"] - a["d1"]) / 1e3))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
