"""RECON7 R1: reproduce the October 21 forecast numbers and evaluate the same formulas at C128/B216 and C73/B318.

Inputs are read from the archived evidence files (paths below). No GPU, no network. Prints every intermediate.
Run inside srun on amd (the nosa-env image), python -I -B.
"""
import csv
import json
from fractions import Fraction

HBM_O3 = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output3_timing_bw.csv"
GEMM = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/rows_bench_2175525/gemm_mscan.json"
LAYOUT = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/cpupack_confirm_2179735/scoring/layout_variants.csv"
GM1 = "/mnt/beegfs/ojerbi/r2-runs/bi/bi_gma_2190876_gm1/summary.json"


def lin(x0, y0, x1, y1, x):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def p(label, v, nd=10):
    print(f"{label:<60s} {v:.{nd}f}" if isinstance(v, float) else f"{label:<60s} {v}")


print("=== 1. Resident C63 curve (2175791, eager median, mode resident) ===")
res = {}
with open(HBM_O3) as fh:
    for row in csv.DictReader(fh):
        if row["C"] == "63" and row["mode"] == "resident":
            res[int(row["B"])] = float(row["eager_median_ms"])
for b in sorted(res):
    print(f"  C63 resident B{b}: eager median {res[b]:.6f} ms (n rows csv)")


def td_of(B):
    pts = sorted(res)
    for lo, hi in zip(pts, pts[1:]):
        if lo <= B <= hi:
            return lin(lo, res[lo], hi, res[hi], B), (lo, hi)
    raise ValueError(B)


print("=== 2. GEMM step sums (2175525, graph-timed, default variant, env base) ===")
g = json.load(open(GEMM))
rows = g["env_variants"]["base"]["rows"]
step = {}
eager_step = {}
per_shape = {}
for r in rows:
    if r.get("variant") != "default" or r.get("op") != "linear":
        continue
    m = r["M"]
    step[m] = step.get(m, 0.0) + r["per_step"] * r["graph_ms"]
    eager_step[m] = eager_step.get(m, 0.0) + r["per_step"] * r["eager_median_ms"]
    per_shape.setdefault(r["shape"], {})[m] = r["per_step"] * r["graph_ms"]
for m in sorted(step):
    print(f"  M{m}: graph step {step[m]:.7f} ms; eager-median step {eager_step[m]:.7f} ms")
Ms = sorted(step)


def gemm_interp(M):
    for lo, hi in zip(Ms, Ms[1:]):
        if lo <= M <= hi:
            return lin(lo, step[lo], hi, step[hi], M), f"interp {lo}-{hi}"
    return None, "outside"


print("--- candidate extrapolation rules for M > 768 (target GEMM(861) = 58.3739738593) ---")
cands = {}
cands["last2 (672,768)"] = lambda M: lin(672, step[672], 768, step[768], M)
cands["640-768"] = lambda M: lin(640, step[640], 768, step[768], M)
cands["640-672"] = lambda M: lin(640, step[640], 672, step[672], M)
cands["512-768"] = lambda M: lin(512, step[512], 768, step[768], M)
cands["256-768"] = lambda M: lin(256, step[256], 768, step[768], M)
cands["128-768"] = lambda M: lin(128, step[128], 768, step[768], M)


def lsq(ms):
    n = len(ms)
    sx = sum(ms)
    sy = sum(step[m] for m in ms)
    sxx = sum(m * m for m in ms)
    sxy = sum(m * step[m] for m in ms)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


for name, sub in [("lsq all", Ms), ("lsq M>=256", [m for m in Ms if m >= 256]), ("lsq M>=320", [m for m in Ms if m >= 320]),
                  ("lsq M>=384", [m for m in Ms if m >= 384]), ("lsq M>=512", [m for m in Ms if m >= 512])]:
    a, b = lsq(sub)
    cands[name] = (lambda a, b: (lambda M: a + b * M))(a, b)


def per_shape_last2(M):
    t = 0.0
    for s, d in per_shape.items():
        t += lin(672, d[672], 768, d[768], M)
    return t


cands["per-shape last2"] = per_shape_last2
cands["ratio M861/768 x step768"] = lambda M: step[768] * M / 768
for name, f in cands.items():
    print(f"  {name:<28s} GEMM(861) = {f(861):.7f}   GEMM(954) = {f(954):.7f}")

print("=== 3. Transfer rate and interference (2179735, layout cpu8@hm>orig:group/row) ===")
lay = {}
with open(LAYOUT) as fh:
    for row in csv.DictReader(fh):
        if row["arm"] == "cpu8@hm>orig:group/row":
            lay[int(row["batch"])] = row
for b in sorted(lay):
    r = lay[b]
    print(f"  B{b}: gbps_alone {r['gbps_alone_p50']}  gbps_conc {r['gbps_conc_p50']}  slowdown_pct {r['slowdown_pct']}  mean_step_slowdown {r['mean_step_slowdown']}")


def r_of(B):
    return lin(64, float(lay[64]["gbps_conc_p50"]), 336, float(lay[336]["gbps_conc_p50"]), B)


def s_of(B):
    return lin(64, float(lay[64]["slowdown_pct"]), 336, float(lay[336]["slowdown_pct"]), B) / 100.0


print("=== 4. Rational structure of U, I, bytes (forecast inputs, not re-derived) ===")
U = 69.1629464286
I = 60.1939174107
BYTES_MB = 6.4658285714
print("  U*448 =", U * 448, " I*3584 =", I * 3584, " bytes*7 =", BYTES_MB * 7)
Uf = Fraction(30985, 448)
If = Fraction(215735, 3584)
print("  U = 30985/448 =", float(Uf), "  I = 215735/3584 =", float(If), "  U-I =", float(Uf - If))
print("  bytes MB / (3 x 64 x 32768 B) =", BYTES_MB * 1e6 / (3 * 64 * 32768), "groups per stream-step")
print("  63 x 64 x 32896 B =", 63 * 64 * 32896, "(f's per-request KV bytes)")

W_GB = 15.768289
KV_GB_PER_REQ = 0.132637
K = 3


def model(C, B, gemmB, gemmBK, label, bytes_mb=BYTES_MB, U=float(Uf), I=float(If), lag=2, pr=0.0):
    print(f"--- model at {label}: C{C} B{B} K{K} lag{lag} p{pr} ---")
    td, seg = td_of(B)
    f = W_GB / (W_GB + B * KV_GB_PER_REQ)
    extra = td * (1 - f) * (U - I) / 63
    hbm = 3 * td + extra
    dense = 3 * gemmB + gemmBK
    s = s_of(B)
    r = r_of(B)
    gpu = max(hbm, dense) * (1 + s)
    n = K + pr * (1 + lag)
    transfer = B * bytes_mb * (n / K) / r
    interval = max(gpu, transfer)
    Q = 1000 * B * 3 / interval
    joint_hbm = td + extra
    joint_dense = gemmB + gemmBK
    budget = max(joint_hbm, joint_dense) * (1 + s)
    p("td (C63 resident interp " + str(seg) + ")", td)
    p("f", f)
    p("1-f", 1 - f)
    p("td(1-f)", td * (1 - f))
    p("extra = td(1-f)(U-I)/63", extra)
    p("cohort HBM = 3 td + extra", hbm)
    p("GEMM(B)", gemmB)
    p("GEMM(BK)", gemmBK)
    p("cohort dense = 3 GEMM(B) + GEMM(BK)", dense)
    p("s (interference)", s)
    p("GPU = max(HBM,dense)(1+s)", gpu)
    p("r GB/s", r)
    p("transfer = B bytes (n/K)/r", transfer)
    p("interval", interval)
    p("Q accepted-draft tok/s", Q)
    p("joint HBM proxy td + extra", joint_hbm)
    p("joint dense GEMM(B)+GEMM(BK)", joint_dense)
    p("joint budget max(...)(1+s)", budget)
    p("2 draft-only passes HBM (2 td)", 2 * td)
    p("2 draft-only passes dense (2 GEMM(B))", 2 * gemmB)
    p("per-pass draft-only GPU max(td,GEMM(B))(1+s)", max(td, gemmB) * (1 + s))
    p("sum of per-pass maxes: 2 draft + joint, x(1+s)", (2 * max(td, gemmB) + max(joint_hbm, joint_dense)) * (1 + s))
    p("draft weight bytes share td*f per pass (ms)", td * f)
    p("draft KV share td*(1-f) per pass (ms)", td * (1 - f))
    return dict(td=td, f=f, extra=extra, hbm=hbm, dense=dense, s=s, gpu=gpu, r=r, transfer=transfer, interval=interval, Q=Q,
                joint_hbm=joint_hbm, joint_dense=joint_dense, budget=budget)


print("=== 5. Original anchor C81 / B287 ===")
g287, s287 = gemm_interp(287)
print("  GEMM(287)", g287, s287)
for name, fx in cands.items():
    pass
# the rule chosen below is printed; the selection is by exact match to the forecast's 58.3739738593
best = min(cands.items(), key=lambda kv: abs(kv[1](861) - 58.3739738593))
print("  closest extrapolation rule to 58.3739738593:", best[0], best[1](861))
ext = best[1]
A = model(81, 287, g287, ext(861), "original anchor")
print("  forecast check: Q/3160.822684 - 1 =", A["Q"] / 3160.822684 - 1)
print("  no-interference Q =", 1000 * 287 * 3 / max(A["hbm"], A["dense"]), " gain", 1000 * 287 * 3 / max(A["hbm"], A["dense"]) / 3160.822684 - 1)
# sharing sensitivity (FC:35)
td = A["td"]
f = A["f"]
for nm, V in [("none (V=U+63? see note)", None), ("weights only V=U", float(Uf)), ("weights+half KV V=U-I/2", float(Uf) - float(If) / 2),
              ("weights+full KV V=U-I", float(Uf) - float(If))]:
    if V is None:
        continue
    ex = td * (1 - f) * V / 63
    hbm = 3 * td + ex
    gpu = max(hbm, A["dense"]) * (1 + A["s"])
    print(f"  sensitivity {nm}: extra {ex:.4f} HBM {hbm:.4f} Q {1000*287*3/max(gpu, A['transfer']):.4f}")
# no sharing at all: verifier also pays its own weight pass: HBM = 3td + td (f + (1-f) U/63)
ex = td * (f + (1 - f) * float(Uf) / 63)
hbm = 3 * td + ex
gpu = max(hbm, A["dense"]) * (1 + A["s"])
print(f"  sensitivity none (verifier pays its own weight pass + U): extra {ex:.4f} HBM {hbm:.4f} Q {1000*287*3/max(gpu, A['transfer']):.4f}")
# p correction
for pr in (0.0, 0.109):
    n_old = K + pr * 2
    n_new = K + pr * 3
    print(f"  p={pr}: n_old K+p*lag = {n_old}, n_new K+p(1+lag) = {n_new}, transfer_new {A['transfer']*n_new/K:.4f}")

print("=== 6. C128 / B216 ===")
g216, s216 = gemm_interp(216)
g648, s648 = gemm_interp(648)
print("  GEMM(216)", g216, s216, " GEMM(648)", g648, s648)
Bm = model(128, 216, g216, g648, "C128/B216 (U, I, bytes carried from C81)")
print("=== 7. C73 / B318 ===")
g318, s318 = gemm_interp(318)
g954 = ext(954)
print("  GEMM(318)", g318, s318, " GEMM(954) extrapolated by", best[0], g954)
Cm = model(73, 318, g318, g954, "C73/B318 (U, I, bytes carried from C81)")

print("=== 8. Human's retrospective conversion (I3 cells) ===")
for lab, B, tok, ownsh, jsh, budget in [("C128/B216", 216, 1709.8, 0.5128, 3.201, Bm["budget"]), ("C73/B318", 318, 1525.8, 0.5925, 3.094, Cm["budget"])]:
    sh = 1000 * B * ownsh / tok
    joint = jsh * sh
    print(f"  {lab}: SH_ms = 1000*{B}*{ownsh}/{tok} = {sh:.4f}; joint = {jsh}*SH = {joint:.3f}; joint/budget = {joint/budget:.4f}; gap {joint-budget:.3f} ms")

print("=== 9. Per round of the implemented schedule vs per model cohort (MODEL terms only) ===")
for lab, M, B in [("anchor C81/B287", A, 287), ("C128/B216", Bm, 216), ("C73/B318", Cm, 318)]:
    # implemented round: 2 draft-only passes (B rows) + 1 joint pass (4B rows: 3 verify + 1 deferred draft)
    # model terms mapped: draft pass = max(td, GEMM(B)); joint pass HBM = td + extra (weights read once), dense = GEMM(4B)
    print(f"  {lab}: model cohort interval {M['interval']:.4f} ms for 3 accepted drafts/request -> {M['interval']/3:.4f} ms per accepted token/request")

print("=== 10. Optional: gm1 eager GEMM inputs (MEASURED later, nova05) for comparison ===")
try:
    gm = json.load(open(GM1))
    print("  gm1 keys:", list(gm.keys())[:20])
except Exception as e:  # noqa
    print("  gm1 not read:", e)
