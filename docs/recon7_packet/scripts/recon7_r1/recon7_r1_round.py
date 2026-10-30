"""RECON7 R1 part round: the model's own curves evaluated on the IMPLEMENTED round (2 draft passes of B rows + one
joint pass of 4B rows), next to the model cohort (3 passes; dense charged as 3 GEMM(B) + GEMM(3B)).
Same inputs and the same interpolation / extrapolation rules as recon7_r1_model.py (GEMM above M 768 = GEMM(768) x M / 768).
"""
import csv
import json

HBM_O3 = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output3_timing_bw.csv"
GEMM = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/rows_bench_2175525/gemm_mscan.json"
LAYOUT = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/cpupack_confirm_2179735/scoring/layout_variants.csv"


def lin(x0, y0, x1, y1, x):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


res = {}
for row in csv.DictReader(open(HBM_O3)):
    if row["C"] == "63" and row["mode"] == "resident":
        res[int(row["B"])] = float(row["eager_median_ms"])
step = {}
for r in json.load(open(GEMM))["env_variants"]["base"]["rows"]:
    if r.get("variant") == "default" and r.get("op") == "linear":
        step[r["M"]] = step.get(r["M"], 0.0) + r["per_step"] * r["graph_ms"]
Ms = sorted(step)
lay = {}
for row in csv.DictReader(open(LAYOUT)):
    if row["arm"] == "cpu8@hm>orig:group/row":
        lay[int(row["batch"])] = row


def interp(tab, x):
    ks = sorted(tab)
    for lo, hi in zip(ks, ks[1:]):
        if lo <= x <= hi:
            return lin(lo, tab[lo], hi, tab[hi], x), f"interp {lo}-{hi}"
    return None, "outside"


def gemm(M):
    v, how = interp(step, M)
    if v is None and M > 768:
        return step[768] * M / 768, "extrap GEMM(768) x M/768"
    if v is None and M < 128:
        return None, "below 128: no row"
    return v, how


def s_of(B):
    return lin(64, float(lay[64]["slowdown_pct"]), 336, float(lay[336]["slowdown_pct"]), B) / 100.0


U, I = 30985 / 448, 215735 / 3584
W, KVR = 15.768289, 0.132637
for C, B in [(81, 287), (128, 216), (73, 318)]:
    td, how_td = interp(res, B)
    f = W / (W + B * KVR)
    s = s_of(B)
    gB, hB = gemm(B)
    g3B, h3B = gemm(3 * B)
    g4B, h4B = gemm(4 * B)
    extra = td * (1 - f) * (U - I) / 63
    print(f"=== C{C} B{B} ===")
    print(f"  td {td:.4f} ({how_td}); GEMM(B={B}) {gB:.4f} ({hB}); GEMM(3B={3*B}) {g3B:.4f} ({h3B}); GEMM(4B={4*B}) {g4B:.4f} ({h4B})")
    print(f"  model joint dense GEMM(B)+GEMM(3B) {gB+g3B:.4f}; one call GEMM(4B) {g4B:.4f}; difference {g4B-(gB+g3B):+.4f}")
    # implemented round under the model's terms, per-round max (no cross-pass overlap)
    draft_pass = max(td, gB)
    joint_model = max(td + extra, gB + g3B)
    joint_4b = max(td + extra, g4B)
    print(f"  per-pass max: draft {draft_pass:.4f}; joint (model dense) {joint_model:.4f}; joint (GEMM(4B)) {joint_4b:.4f}")
    print(f"  round = 2 draft + joint, x(1+s={s:.6f}): model dense {(2*draft_pass+joint_model)*(1+s):.4f}; GEMM(4B) {(2*draft_pass+joint_4b)*(1+s):.4f}")
    hbm = 3 * td + extra
    print(f"  cohort max over whole cohort: HBM {hbm:.4f}; dense(model) {3*gB+g3B:.4f}; dense(2GEMM(B)+GEMM(4B)) {2*gB+g4B:.4f}; GPU {max(hbm, 3*gB+g3B)*(1+s):.4f}")
    # rows / work units
    print(f"  rows per cohort: draft {3*B}, verify {3*B}, total {6*B}; weight passes 3; weight bytes read 3 x {W} GB = {3*W:.6f} GB")
    print(f"  KV groups per request per cohort (model): draft 3 x 63 = 189, verifier extra U-I = {U-I:.6f}; total {189+U-I:.6f}")
    print(f"  KV bytes per request per cohort (x 64 layer-heads x 32896 B): draft {189*64*32896/1e6:.4f} MB; verifier extra {(U-I)*64*32896/1e6:.4f} MB")
    for y in (3.0, 2.8926, 2.8091):
        print(f"    yield {y}: tokens per request per round; model GPU-bound rate at the cohort interval = {1000*B*y/(max(hbm, 3*gB+g3B)*(1+s)):.2f} tok/s")
