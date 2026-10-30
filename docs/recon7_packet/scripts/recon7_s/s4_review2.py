# RECON7 synthesis, review-2 arithmetic with the author's model inputs (2026-10-30). CPU only (srun on amd).
# Author inputs: U, I, traffic, overflow, td points, backbone points, memory rule (coordinator message, review 2).
# Measured inputs: R2, R3, SI, V4 (as cited in LEDGER_rev7.md); J91 snapshot column of output2_memory.csv (C63 rows).

def lin(a, b, t):
    return a + (b - a) * t

def interp(pts, x):
    ks = sorted(pts)
    if x <= ks[0]:
        lo, hi = ks[0], ks[1]
    elif x >= ks[-1]:
        lo, hi = ks[-2], ks[-1]
    else:
        lo = max(k for k in ks if k <= x)
        hi = min(k for k in ks if k > x) if x != lo else lo
        if hi == lo:
            return pts[lo]
    return pts[lo] + (pts[hi] - pts[lo]) * (x - lo) / (hi - lo)

U = 69.16294642857143
I = 60.193917410714285
UNIT_MB = 32 * 2 * 32768 / 1e6
CFG = {"anc": dict(C=81, B=287, traffic=6.465828571428571, ovf=1.6384),
       "c128": dict(C=128, B=216, traffic=2.910500571428572, ovf=0.0),
       "c73": dict(C=73, B=318, traffic=9.097801142857142, ovf=5.332992)}
TD = {64: 27.376640, 128: 37.512192, 256: 62.678017, 344: 80.585732}
BACK = {64: 29.814030, 128: 41.247703, 256: 65.727758, 344: 83.870220}
SNAP = {64: 1.247945, 128: 2.49589, 256: 4.991779, 344: 6.707703}  # J91 output2_memory.csv C63 rows, staging snapshot
GS = {128: 14.5920097, 256: 20.027118, 320: 25.4561339, 384: 27.26451, 512: 34.6211361, 640: 42.9303249, 672: 51.6753099, 768: 52.0687711}  # V3:11
W = 15.768289
LV = {336: (0.07378858, 11.133855), 64: (0.14626987, 8.859767)}  # V3:13

def gemm(M):
    if M > 768:
        return GS[768] * M / 768.0
    return interp(GS, M)

def s_r(B):
    t = (B - 64.0) / (336 - 64.0)
    return lin(LV[64][0], LV[336][0], t), lin(LV[64][1], LV[336][1], t)

print("U - I = %.9f groups; unit %.6f MB; extra bytes per request per cohort %.4f MB" % (U - I, UNIT_MB, (U - I) * UNIT_MB))
print("\n== 1. model columns with the author's inputs ==")
M = {}
for k in ("anc", "c128", "c73"):
    c = CFG[k]
    B = c["B"]
    td = interp(TD, B)
    f = W / (W + B * 0.132637)
    extra = td * (1 - f) * (U - I) / 63
    gb, g3b, g4b = gemm(B), gemm(3 * B), gemm(4 * B)
    s, r = s_r(B)
    hbm = 3 * td + extra
    dense = 3 * gb + g3b
    gpu = max(hbm, dense) * (1 + s)
    xfer = B * c["traffic"] / r
    interval = max(gpu, xfer)
    Q = 1000.0 * B * 3 / interval
    jb = max(td + extra, gb + g3b) * (1 + s)
    perpass = 2 * max(td, gb) * (1 + s) + jb
    M[k] = dict(td=td, f=f, extra=extra, gb=gb, g3b=g3b, g4b=g4b, s=s, r=r, gpu=gpu, xfer=xfer, interval=interval, Q=Q,
                jb=jb, perpass=perpass, ov=td + extra - gb, serial=td + extra + g3b)
    print("%s B%d: td %.4f f %.5f extra %.4f GEMM(B) %.4f GEMM(3B) %.4f GEMM(4B) %.4f s %.5f r %.5f" % (k, B, td, f, extra, gb, g3b, g4b, s, r))
    print("   cohort HBM %.4f dense %.4f GPU %.4f; transfer %.4f ms (%.4f MB per request); interval %.4f (%s binds); Q %.2f; gain vs 3160.822684 %+.3f %%" % (
        hbm, dense, gpu, xfer, c["traffic"], interval, "TRANSFER" if xfer > gpu else "GPU", Q, 100 * (Q / 3160.822684 - 1)))
    print("   joint budget (REQ:40) %.4f; per-pass reading %.4f; overlap term td + extra - GEMM(B) %.4f; serial form %.4f" % (jb, perpass, td + extra - gb, td + extra + g3b))
    print("   Q / strongest 3426.2 %.4f, 3410.1 %.4f" % (Q / 3426.2, Q / 3410.1))

print("\n== 2. memory with the author's rule (decimal GB; aliased retention: no copied union) ==")
STATE_MB = (2 * (3 + 1 + 2) * 32768 + 3 * 8192) / 1e6
LAND = 3 * 17039360 / 1e9
print("state_MB %.6f; landing %.6f GB" % (STATE_MB, LAND))

def mem(C, B, ovf, copied=0.0, drop_snap=False):
    base = interp(BACK, B) + (C - 63) * B * 2105344 / 1e9 + 0.12740196
    if drop_snap:
        base -= interp(SNAP, B)
    return base + B * (ovf * (1 + 128 / 32768.0) + STATE_MB + copied) / 1000.0 + LAND

def bmax(C, ovf, cap=84.0, drop_snap=False):
    lo, hi = 1.0, 2000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mem(C, mid, ovf, drop_snap=drop_snap) <= cap:
            lo = mid
        else:
            hi = mid
    return lo

for k in ("anc", "c128", "c73"):
    c = CFG[k]
    C, B, ovf = c["C"], c["B"], c["ovf"]
    tot = mem(C, B, ovf)
    tot_ns = mem(C, B, ovf, drop_snap=True)
    print("%s C%d B%d: backbone %.6f + const + slots %.6f + per-request %.6f + landing -> total %.6f GB; largest B under 84.0: %.2f" % (
        k, C, B, interp(BACK, B), (C - 63) * B * 2105344 / 1e9, B * (ovf * (1 + 128 / 32768.0) + STATE_MB) / 1000.0, tot, bmax(C, ovf)))
    print("   snapshot at B %.6f GB; total without snapshot %.6f; largest B without snapshot %.2f (MODEL)" % (interp(SNAP, B), tot_ns, bmax(C, ovf, drop_snap=True)))
    print("   each 1 MB per request of copied union adds %.4f GB at this B" % (B / 1000.0))
print("our allocated peaks with prefill: C128 83.891 / 83.900; C73 77.685 / 77.694 (I3 / BEST); fit lines 19.020 + 0.30033 B, 19.020 + 0.18448 B")

print("\n== 3. measured bytes vs the author's traffic ==")
for nm, ours, k in (("C128 all arms", 2.581, "c128"), ("C73 I3", 10.527, "c73"), ("C73 BEST keep + window", 8.065, "c73")):
    print("%s: ours %.3f MB per request per round vs model %.4f: x%.3f" % (nm, ours, CFG[k]["traffic"], ours / CFG[k]["traffic"]))

print("\n== 4. point 1: resource service per joint pass (MODEL floors; references named) ==")
# references: 312 TF bf16 peak (vendor, CITED R2:165); achieved logical-byte rate of the one-row flash call A 1.527 / 1.543 TB/s
# (R2:166); bulk pinned H2D 25.2 GB/s (CITED R2:171); FLOPs 13.624 / 20.057 TF (RV row 19); KV logical 32.85 / 48.36 GB (R2:133,
# C128 from the C73 page count); H2D 557.46 / 3347.6 MB (R2:136); selection / tentative / small measured (R2:228-244).
for k, flops, bw, kv, h2d, gemm_m, fork_m, h2d_m, sel, tent, small, joint in (
        ("C128", 13.624, 1.527, 32.85, 557.46, 64.800, 37.118, 21.302, 27.310, 28.908, 4.386, 207.377),
        ("C73", 20.057, 1.543, 48.36, 3347.6, 79.879, 52.165, 160.136, 38.472, 31.118, 6.096, 382.041)):
    g_c = flops / 312.0 * 1000
    g_w = W / bw
    a_m = kv / bw
    p = h2d / 25.2
    S = sel + tent + small
    serial_floor = max(g_c, g_w) + a_m + p + S
    split_floor = max(g_c, a_m + g_w, p) + S
    print("%s: GEMM stage floor max(compute %.2f, weights %.2f) = %.2f (measured %.2f); attention floor (logical bytes at the flash rate) %.2f (measured fork %.2f); PCIe floor %.2f (measured %.2f)" % (
        k, g_c, g_w, max(g_c, g_w), gemm_m, a_m, fork_m, p, h2d_m))
    print("   unmodelled stages measured (selection + tentative + small) %.2f" % S)
    print("   A dependent single schedule, floors + measured rest: %.2f ms; B two-half cross-request split, ideal interleave, weights read twice, no fill: %.2f ms; measured joint %.2f" % (
        serial_floor, split_floor, joint))

print("\n== 5. point 3: capture amortization (V4; S3) ==")
for nm, com, wall, pre, cap, i3cw in (("C128", 110376, 85.569, 2.457, 9.594118, 1394.0), ("C73", 162498, 131.034, 2.562, 10.140224, 1272.8)):
    base = wall - cap
    i3_time = com / i3cw
    k_cap = cap / (i3_time - base)
    k_both = (cap + pre) / (i3_time - (wall - cap - pre))
    print("%s: BEST with capture amortized over k runs: rate(k) = %d / (%.3f + %.3f / k); break-even vs I3 (%.1f) at k = %.2f runs; amortizing capture + prelude: k = %.2f" % (
        nm, com, base, cap, i3cw, k_cap, k_both))
    for kk in (1, 2, 5, 10):
        print("   k %d: %.1f tok/s" % (kk, com / (base + cap / kk)))

print("\n== 6. point 4: dispatch shapes and draft accounting ==")
for k, gd_m, gj_m in (("c128", 19.785, 64.800), ("c73", 25.705, 79.879)):
    m = M[k]
    print("%s: model curve 3 GEMM(B) + GEMM(3B) %.3f vs our shape 2 GEMM(B) + GEMM(4B) %.3f (diff %+.3f); measured device 2 x %.3f + %.3f = %.3f" % (
        k, 3 * m["gb"] + m["g3b"], 2 * m["gb"] + m["g4b"], 2 * m["gb"] + m["g4b"] - 3 * m["gb"] - m["g3b"], gd_m, gj_m, 2 * gd_m + gj_m))
for nm, p, nbar, ymeas in (("C128", 1058 / 37614.0, 0.871, 2.855), ("C73", 2250 / 55866.0, 0.812, 2.798)):
    print("%s: p %.5f; fixed-lead n = 3 + 3p = %.4f -> Y = 9 / n = %.4f; empirical n = 3 + 5.1p = %.4f -> %.4f; measured active-request %.3f; dropped drafts per reject 4 - n = %.3f (model 1 + lag = 3)" % (
        nm, p, 3 + 3 * p, 9 / (3 + 3 * p), 3 + 5.1 * p, 9 / (3 + 5.1 * p), ymeas, 4 - nbar))

print("\n== 7. revised model at the anchor with interference charged ONCE: Vx = (R - removed) x (1 + s) ==")
T = (287.0 - 216.0) / (318.0 - 216.0)
I3 = {"c128": dict(body=15.294, attn=11.649, sel=17.830, tent=25.915, book=0.775, small=2.743 + 8.454, fork=15.601, gj=64.799, gd=19.785, draft=151.363, c=6.114),
      "c73": dict(body=18.203, attn=14.650, sel=24.434, tent=27.171, book=0.089, small=4.084 - 1.873, fork=20.827, gj=79.879, gd=25.705, draft=194.550, c=8.466)}
BEST = {"c128": dict(d_attn=-0.606, d_tent=-17.142, d_dense=-0.372, d_nonbusy=-2.131, draft=132.287, c=6.178),
        "c73": dict(d_attn=-1.135, d_tent=-17.445, d_dense=0.070, d_nonbusy=-9.610, draft=176.437, c=8.546)}
a = M["anc"]
tdm = {"c128": M["c128"]["td"], "c73": M["c73"]["td"]}
body = lin(I3["c128"]["body"] / tdm["c128"], I3["c73"]["body"] / tdm["c73"], T) * a["td"]
gj = lin(64.799, 79.879, (4 * 287 - 864) / 408.0)
gd = lin(19.785, 25.705, T)
dense_ex = (gj - gd) - a["g3b"]
attn = lin(I3["c128"]["attn"] / 216, I3["c73"]["attn"] / 318, T) * 287
fork = lin(I3["c128"]["fork"] / 216, I3["c73"]["fork"] / 318, T) * 287
sel = lin(I3["c128"]["sel"] / 648, I3["c73"]["sel"] / 954, T) * 861
tent = lin(I3["c128"]["tent"] / 216, I3["c73"]["tent"] / 318, T) * 287
book = lin(I3["c128"]["book"], I3["c73"]["book"], T)
small = lin(I3["c128"]["small"], I3["c73"]["small"], T)
c_i3 = lin(6.114, 8.466, T)
c_b = lin(6.178, 8.546, T)
bd = {key: lin(BEST["c128"][key] / 216, BEST["c73"][key] / 318, T) * 287 for key in ("d_attn", "d_tent", "d_dense", "d_nonbusy")}
d_i3 = lin((151.363 / 2) / tdm["c128"], (194.550 / 2) / tdm["c73"], T) * a["td"]
d_b = lin((132.287 / 2) / tdm["c128"], (176.437 / 2) / tdm["c73"], T) * a["td"]
serial = a["serial"]
drain = lin(0.950, 16.454, T)
corr = lin(0.02525, 0.03385, T)
print("anchor terms: td %.4f; d I3 %.3f, best %.3f (alt 84.399); serial %.3f; body %.3f; dense_ex %.3f; attn %.3f (fork %.3f); sel %.3f; tent %.3f; book %.3f; small %.3f; c %.3f / %.3f; drain %.3f" % (
    a["td"], d_i3, d_b, serial, body, dense_ex, attn, fork, sel, tent, book, small, c_i3, c_b, drain))
for nm, code, d, rate, Y in (("I3 code, 27.6 GB/s, Y 2.864", "I3", d_i3, 27.6, 2.864), ("I3 code, 22.0 GB/s, Y 2.808", "I3", d_i3, 22.0, 2.808),
                             ("best code, d 81.56, 27.6 GB/s, Y 2.864", "B", d_b, 27.6, 2.864), ("best code, d 84.40, 22.0 GB/s, Y 2.808", "B", 84.399, 22.0, 2.808)):
    x = 287 * CFG["anc"]["traffic"] / rate
    J = serial + body + dense_ex + attn + sel + tent + book + x + small
    cc = c_i3
    if code == "B":
        J += bd["d_attn"] + bd["d_tent"] + bd["d_dense"] + bd["d_nonbusy"]
        cc = c_b
    R = 2 * d + J + cc
    rows = [("R", R), ("R + drain", R + drain), ("V1", (R - x) * (1 + a["s"])), ("V2", (R - x - a["ov"]) * (1 + a["s"])),
            ("V3", (R - x - a["ov"] - fork) * (1 + a["s"]))]
    print(nm + " (exposed transfer %.3f ms; joint %.2f)" % (x, J))
    for lab, iv in rows:
        qa = 1000 * 287 * Y / iv
        qc = 1000 * 287 * (Y + corr) / iv
        print("   %-9s %.2f ms: accepted %.1f (%.4f / %.4f); committed %.1f (%.4f / %.4f)" % (lab, iv, qa, qa / 3426.2, qa / 3410.1, qc, qc / 3426.2, qc / 3410.1))

print("\n== 8. measured-config variants, interference once ==")
for nm, k, B, y, R, x, fork_m, den in (("I3 C128", "c128", 216, 2.8926, 365.423, 19.856, 15.601, 3410.1), ("I3 C73", "c73", 318, 2.8091, 585.463, 158.012, 20.827, 3426.2),
                                       ("BEST C128", "c128", 216, 2.894, 324.883, 20.25, 15.601, 3426.2), ("BEST C73", "c73", 318, 2.803, 524.659, 115.64, 20.827, 3410.1)):
    s = M[k]["s"]
    ov = M[k]["ov"]
    out = []
    for lab, iv in (("V1", (R - x) * (1 + s)), ("V2", (R - x - ov) * (1 + s)), ("V3", (R - x - ov - fork_m) * (1 + s))):
        q = 1000 * B * y / iv
        out.append("%s %.2f ms Q %.1f (/strong %.4f)" % (lab, iv, q, q / den))
    print("%s: %s" % (nm, "; ".join(out)))

print("\n== 9. round vs model cohort with transfer binding at C73 ==")
for nm, k, dr, jo, c, rnd in (("I3 C128", "c128", 151.363, 207.377, 6.114, 365.423), ("I3 C73", "c73", 194.550, 382.041, 8.466, 585.463),
                              ("BEST C128", "c128", 132.287, 186.011, 6.178, 324.883), ("BEST C73", "c73", 176.437, 338.961, 8.546, 524.659)):
    m = M[k]
    parts = [("draft - 2 max(td, G(B))(1 + s)", dr - 2 * max(m["td"], m["gb"]) * (1 + m["s"])), ("joint - joint budget", jo - m["jb"]),
             ("commit + reset", c), ("lending: per-pass - GPU", m["perpass"] - m["gpu"]), ("transfer binding: GPU - interval", m["gpu"] - m["interval"]),
             ("round - phase sum", rnd - (dr + jo + c))]
    tot = sum(v for _, v in parts)
    print("%s: round %.3f - model interval %.4f = %.3f; %s; sum %.3f" % (nm, rnd, m["interval"], rnd - m["interval"], "; ".join("%s %+.3f" % p for p in parts), tot))
