"""RECON7 review V3 (read-only): rebuild the October 21 model terms from the raw calibration files (job 2175791 CSVs,
job 2175525 GEMM scan JSON, job 2179735 layout variants CSV) and evaluate them at C128 / B216, C73 / B318, C81 / B287.
Independent of the R1 scripts. CPU only."""
import csv
import json
from collections import defaultdict

E = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/"
O3 = E + "hbm_sweep_2175791/scoring/output3_timing_bw.csv"
O2 = E + "hbm_sweep_2175791/scoring/output2_memory.csv"
GS = E + "rows_bench_2175525/gemm_mscan.json"
LV = E + "cpupack_confirm_2179735/scoring/layout_variants.csv"

# ---- td: eager median of the C63 RESIDENT step, linear in B
res = {}
for r in csv.DictReader(open(O3)):
    if r["C"] == "63" and r["mode"] == "resident":
        res[int(r["B"])] = float(r["eager_median_ms"])
print("td points", res)


def lin(pts, x):
    ks = sorted(pts)
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            return pts[a] + (x - a) * (pts[b] - pts[a]) / (b - a)
    raise ValueError(x)


# ---- GEMM(M): sum over shapes of per_step x graph_ms; linear between scan points; above 768 proportional
g = json.load(open(GS))
rows = g["env_variants"]["base"]["rows"]
step = defaultdict(float)
seen = defaultdict(list)
for r in rows:
    if r.get("variant") == "default" and r.get("op") == "linear":
        step[r["M"]] += r["per_step"] * r["graph_ms"]
        seen[r["M"]].append(r["shape"])
print("GEMM step sums by M", {k: round(v, 7) for k, v in sorted(step.items())})
print("shapes per M", dict(seen))


def gemm(M):
    if M > 768:
        return step[768] * M / 768.0
    return lin(dict(step), M)


# ---- s and r: layout variant cpu8@hm>orig:group/row, B64 and B336
lv = {}
for r in csv.DictReader(open(LV)):
    if r["arm"] == "cpu8@hm>orig:group/row":
        lv[int(r["batch"])] = (float(r["slowdown_pct"]) / 100.0, float(r["gbps_conc_p50"]))
print("LV", lv)


def s_of(B):
    return lin({k: v[0] for k, v in lv.items()}, B)


def r_of(B):
    return lin({k: v[1] for k, v in lv.items()}, B)


W, PER = 15.768289, 0.132637
U, I = 30985 / 448.0, 215735 / 3584.0
MB = 6.4658285714
print("U %.7f I %.7f U-I %.6f" % (U, I, U - I))
for C, B in ((81, 287), (128, 216), (73, 318)):
    td = lin(res, B)
    f = W / (W + B * PER)
    ex = td * (1 - f) * (U - I) / 63
    gb, g3, g4 = gemm(B), gemm(3 * B), gemm(4 * B)
    s, r = s_of(B), r_of(B)
    hbm_j, den_j = td + ex, gb + g3
    bud = max(hbm_j, den_j) * (1 + s)
    hbm_c, den_c = 3 * td + ex, 3 * gb + g3
    gpu = max(hbm_c, den_c) * (1 + s)
    tr = B * MB / r
    iv = max(gpu, tr)
    q = 1000 * B * 3 / iv
    perpass = 2 * max(td, gb) * (1 + s) + bud
    print("C%d B%d: td %.7f f %.7f extra %.7f GEMM(B) %.7f GEMM(3B) %.7f GEMM(4B) %.7f s %.8f r %.8f" % (
        C, B, td, f, ex, gb, g3, g4, s, r))
    print("   joint: HBM %.7f dense %.7f budget %.7f (binds %s) allowance s x dense %.4f; budget w/ GEMM(4B) %.4f" % (
        hbm_j, den_j, bud, "dense" if den_j > hbm_j else "HBM", s * den_j, max(hbm_j, g4) * (1 + s)))
    print("   cohort: HBM %.7f dense %.7f GPU %.7f transfer %.7f interval %.7f Q %.4f; per-pass sum %.4f;"
          " R3 split: 2 td (1+s) %.4f + (td+extra)(1+s) %.4f" % (hbm_c, den_c, gpu, tr, iv, q, perpass,
                                                                 2 * td * (1 + s), (td + ex) * (1 + s)))
    print("   serial form td + extra + GEMM(3B) %.4f; serial - budget %.4f" % (td + ex + g3, td + ex + g3 - bud))
    if C == 81:
        print("   gain vs 3160.822684: %+.6f %%" % (100 * (q / 3160.822684 - 1)))

# I3 measured joints and the ratio
for C, B, j_req, j_raw in ((128, 216, 207.368, 207.377), (73, 318, 382.066, 382.041)):
    td = lin(res, B)
    s = s_of(B)
    bud = max(td + td * (1 - W / (W + B * PER)) * (U - I) / 63, gemm(B) + gemm(3 * B)) * (1 + s)
    print("C%d: joint/budget %.4f (gap %.3f) using REQ joint; %.4f (gap %.3f) using record joint" % (
        C, j_req / bud, j_req - bud, j_raw / bud, j_raw - bud))

# ---- memory formula (R1 reading): backbone = device_used line (B256..344) + const; const back-solved at C81 B287
mem = {}
for r in csv.DictReader(open(O2)):
    if r["C"] == "63" and r["kind"] == "full cell":
        mem[int(r["B"])] = r
slot = 2105344e-9
ovf = (1.6384 + 0.417792) / 1000.0
land = 3 * 17039360e-9


def line(col, B, a=256, b=344):
    va, vb = float(mem[a][col]), float(mem[b][col])
    return va + (B - a) * (vb - va) / (b - a)


raw = line("device_used_GB", 287) + 18 * 287 * slot + 287 * ovf + land
const = 83.7655 - raw
print("memory: device_used(B287, 256..344 line) %.6f; const to hit 83.7655 = %.6f GB" % (line("device_used_GB", 287), const))
for C, B in ((81, 287), (128, 216), (73, 318)):
    tot = line("device_used_GB", B) + const + (C - 63) * B * slot + B * ovf + land
    slope = (float(mem[344]["device_used_GB"]) - float(mem[256]["device_used_GB"])) / 88 + (C - 63) * slot + ovf
    bmax = B + (84.0 - tot) / slope
    print("  C%d B%d: model total %.6f GB; per-request slope %.6f; largest B under 84.0 = %.2f; staging snapshot at B %.4f"
          % (C, B, tot, slope, bmax, line("staging_light_restore_snapshot_GB", B)))
# basis decomposition at B287 on the 256..344 segment
for col in ("device_used_GB", "reserved_GB", "allocated_GB", "peak_alloc_GB_process_max", "peak_reserved_GB_process_max",
            "staging_light_restore_snapshot_GB", "allocator_slack_cached_free_GB", "other_overhead_outside_allocator_GB"):
    print("  B287 on 256..344 segment %-40s %.4f   (64..344 line %.4f)" % (col, line(col, 287), line(col, 287, 64, 344)))
bb = line("device_used_GB", 287) + const
print("  model backbone %.4f; backbone - allocated %.4f; - peak alloc %.4f; slack + outside + const = %.4f" % (
    bb, bb - line("allocated_GB", 287), bb - line("peak_alloc_GB_process_max", 287),
    line("allocator_slack_cached_free_GB", 287) + line("other_overhead_outside_allocator_GB", 287) + const))
# like-for-like: the model formula on the allocated (process peak) basis, with and without the staging snapshot
for basis in ("peak_alloc_GB_process_max", "allocated_GB"):
    for drop_stage in (False, True):
        for C, B in ((81, 287), (128, 216), (73, 318)):
            def tot_at(BB):
                v = line(basis, BB, 64, 344)
                if drop_stage:
                    v -= line("staging_light_restore_snapshot_GB", BB, 64, 344)
                return v + (C - 63) * BB * slot + BB * ovf + land
            t0, t1 = tot_at(B), tot_at(B + 1)
            bmax = B + (84.0 - t0) / (t1 - t0)
            print("  basis %-28s drop staging %-5s C%d B%d: total %.3f GB, largest B under 84.0 %.1f" % (
                basis, drop_stage, C, B, t0, bmax))
