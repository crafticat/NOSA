# REOPT 2026-11-02: re-optimization of (C, B, K=3, L=2, arm) under the 84 GB cap with the MEASURED costs.
# Analysis only. NO NEW KERNELS (this script builds none and proposes none). No GPU. Reads immutable run records.
# Run only on amd:
#   srun -p amd -c 4 --mem=16G -t 30 python3 -I /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/reopt_1102/s3_reopt.py \
#        > /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/reopt_1102/s3_reopt.out
# Method (ORIGINAL_CALCULATIONS.md "Memory and finding a new optimum"): per (C, arm), for each integer B: memory first,
# discard over the cap, then the time terms, then Q; compare nearby candidates. Q_committed = 1000 B Y_c / T_round.
# Population: ratio of sums on the steady set (r2_spec_cell.py:206-217: mean_tok_s = steady tokens / steady time), so
# T_mean = tokens_per_round_mean / mean_tok_s and Y_c = tokens_per_round_mean / B (= the matched committed yield,
# LEDGER_rev7.md:774-776). The median basis is printed beside it for continuity with the ledger (never mixed).
# Labels: MEASURED (record field), DERIVED (arithmetic on MEASURED), MODEL (stated assumption), AUTHOR (the original
# author's inputs, LEDGER_rev7.md:40-55).
import json

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
MBS = "/mnt/beegfs/ojerbi/r2-runs/mbstrong/"


def load(p):
    with open(p) as f:
        return json.load(f)


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ---------------------------------------------------------------- caps (task statement; lesson L60)
CAP = 84.0           # GB, allocated basis (the registered budget)
USABLE = 83.75       # GB, usable ceiling after another arm in process (L60; RECON7_R3:159-161)
MARGIN = 0.6         # GB, fit margin for MODEL configurations (L60)
STRICT = USABLE - MARGIN   # 83.15 GB: a MODEL configuration is FEASIBLE only below this
# A MEASURED configuration that ran is feasible by demonstration even above STRICT (e.g. inta 83.909 GB).

PH = ("draft", "joint", "commit", "reset")

# ---------------------------------------------------------------- the INT cells (MEASURED)
INT = {128: ("int_a1_2195630_inta", "L4PTFDGUDCDI"),
       81: ("int_c1_2195631_intc", "L4PTFDGUDCDI"),
       73: ("int_d1_2195630_intd", "L4PTFDGUDKWCDIN")}
# plain-arm (L4PTFD) B points per C for the B scaling (MEASURED; mixed jobs: MODEL when combined)
PLAIN_EXTRA = {128: [("i4m_m3p128_2195575_i4mp128", "ARM", None)],
               81: [("recon7_mem1_2195630_r7mem", "per_batch", "64"), ("recon7_mem1_2195630_r7mem", "per_batch", "192")],
               73: [("i4m_m3p73_2195575_i4mp73", "ARM", None)]}
FITREC = {128: "i4m_m3p128_2195575_i4mp128", 81: "recon7_mem1_2195630_r7mem", 73: "i4m_m3p73_2195575_i4mp73"}

cells = {}
for C, (run, arm) in INT.items():
    T = load(R + run + "/summary.json")
    rec = T["SPEC"][arm]
    pl = T["SPEC"]["L4PTFD"]
    B0 = T["meta"]["batch"]
    rt = rec["rates"]
    cell = dict(run=run, arm=arm, B0=B0, rates=rt, ph={p: rec["phases_median"][p] for p in PH},
                peak=rec["peak_with_prefill_GB"], plain_peak=pl["peak_with_prefill_GB"],
                plain_ph={p: pl["phases_median"][p] for p in PH},
                wait=g(rec, "kv_at_verify", "wait_ms_median"), new_mb=g(rec, "kv_at_verify", "new_MB_per_round"),
                jgroups=g(rec, "joint_phases_fork", "groups_median"), store=g(rec, "store"),
                sh=T["SH"], draft_step=rec.get("draft_step_ms"))
    cell["Yc"] = rt["tokens_per_round_mean"] / B0                        # DERIVED (= matched committed yield)
    cell["Tmean0"] = 1e3 * rt["tokens_per_round_mean"] / rt["mean_tok_s"]  # DERIVED (ratio of sums)
    pts = [(B0, cell["plain_ph"], "%s L4PTFD" % run)]
    for (r2, key, sub) in PLAIN_EXTRA[C]:
        T2 = load(R + r2 + "/summary.json")
        a = T2[key] if sub is None else T2[key][sub]["ARM"]
        pts.append((int(sub) if sub else T2["meta"]["batch"], {p: a["phases_median"][p] for p in PH},
                    "%s %s" % (r2, sub or "ARM")))
    pts.sort(key=lambda x: x[0])
    cell["pts"] = pts
    F = load(R + FITREC[C] + "/summary.json")["fit"]
    cell["line"] = F["prefill_line"]                                      # MEASURED fit (allocated, prefill binds)
    cell["fitrec"] = FITREC[C]
    cells[C] = cell

# ---------------------------------------------------------------- B scaling of the phases (MODEL)
# L_p(B): piecewise linear through the plain-arm points; beyond the last point, a local slope.
# C81 has 3 points: local slope = last segment (192 -> 287). C128 / C73 have 2 points (64, B0): the chord under-
# states the local slope (convex: weight-bound GEMMs at small B), so the central local slope = chord x kappa_p, with
# kappa_p = C81's (last segment / chord 64 -> 287) per phase (MODEL: shape transferred from C81). Variants:
# lo = chord (C128 / C73) or last segment (C81); hi = central x 1.15.


def seg_slope(pts, i, j, p):
    return (pts[j][1][p] - pts[i][1][p]) / float(pts[j][0] - pts[i][0])


c81 = cells[81]["pts"]
KAPPA = {p: (seg_slope(c81, 1, 2, p) / seg_slope(c81, 0, 2, p) if seg_slope(c81, 0, 2, p) > 0 else 1.0) for p in PH}
for C, cell in cells.items():
    pts = cell["pts"]
    n = len(pts)
    loc = {}
    for p in PH:
        last = seg_slope(pts, n - 2, n - 1, p)
        if n >= 3:
            loc[p] = {"lo": last, "c": last, "hi": last * 1.15}
        else:
            loc[p] = {"lo": last, "c": last * KAPPA[p], "hi": last * KAPPA[p] * 1.15}
    cell["loc"] = loc
    cell["rho"] = {p: cell["ph"][p] / pts[-1][1][p] for p in PH}          # INT / plain at B0 (DERIVED)


def L(cell, p, B, var):
    pts = cell["pts"]
    Bl, phl = pts[-1][0], pts[-1][1]
    if B >= Bl:
        return phl[p] + cell["loc"][p][var] * (B - Bl)
    # below the last point: central = piecewise linear (chord); 'lo'-T variant = backward local slope (convex truth
    # lies between the two)
    lin = None
    for i in range(len(pts) - 1):
        if pts[i][0] <= B <= pts[i + 1][0]:
            lin = pts[i][1][p] + seg_slope(pts, i, i + 1, p) * (B - pts[i][0])
    back = phl[p] - cell["loc"][p]["c"] * (Bl - B)
    if var == "c":
        return lin
    return min(lin, back) if var == "lo" else max(lin, back)


def M(cell, B, var):
    """Model phase sum of the INT arm at batch B (ms; median-phase basis)."""
    return sum(cell["rho"][p] * L(cell, p, B, var) for p in PH)


def t_int(cell, B, var):
    m0 = M(cell, cell["B0"], "c")
    m = M(cell, B, var)
    return cell["Tmean0"] * m / m0, cell["rates"]["round_ms_median"] * m / m0


# ---------------------------------------------------------------- track 1 S (BF MODEL, INTEGRATION_NOTES:1116-1130)
# saving in J = wait x a (a = c_req x q); costs: draft 6.2 + dk x r', J-side jk x r'; r' = NEW(C, B) / NEW(anchor).
# Calibrated so that the anchor reproduces the notes' S - X = -6 .. +21, central +7 ms per round.
S_VAR = {"c": dict(a=0.38, dk=3.8, jk=17.0), "best": dict(a=0.48, dk=2.8, jk=14.0),
         "worst": dict(a=0.28, dk=4.8, jk=20.0)}
NEW_ANCHOR = cells[81]["new_mb"]


def s_net(cell, B, v):
    """ms per round SAVED by S (positive = faster); MODEL."""
    q = S_VAR[v]
    f = float(B) / cell["B0"]
    r = cell["new_mb"] / NEW_ANCHOR * f
    return cell["wait"] * f * q["a"] - (6.2 + q["dk"] * r) - q["jk"] * r


# BF parts in memory (INTEGRATION_NOTES:1089-1096): landing ring 1.208 + transport 0.136 + counters 0.015 GB fixed;
# per request: snapshot ring + ledger bitmaps. The notes used H = 8 (0.113 GB at B287); the lead correction (:1104)
# says H = 2 KV heads (0.028 GB at B287). We use the H = 8 value (conservative, +0.085 GB at B287).
BF_FIXED = 1.208 + 0.136 + 0.015
BF_PER_REQ = (0.075 + 0.038) / 287.0
# The kept store X 129 (8.487 MB / request) is ALREADY in every INT arm's arena (record field store.store_GB and
# memory.arena.store_GB; e.g. intc 2.4358 GB): it is NOT added again (this corrects INTEGRATION_NOTES:1107-1109).

# ---------------------------------------------------------------- track 2 J (INTEGRATION_NOTES:1283-1295; j1f)
# FORK / FORKT per pass at B216 and B318 (j1f synthetic shapes, union 72; track2_numbers.out lines 11, 15),
# linear in B (MODEL, as the notes do at B287). Ahead levels -15 % .. +10 % of FORK; F7 ceiling FORKT - FORK.
J_VAR = {"c": (-0.025, 0.5), "best": (-0.15, 1.0), "worst_raw": (0.10, 0.0), "worst_adopt": (0.0, 0.0)}


def fork(B):
    return 34.41 + (51.10 - 34.41) * (B - 216) / 102.0


def forkt(B):
    return 33.38 + (49.82 - 33.38) * (B - 216) / 102.0


def j_delta(B, v):
    fa, gf = J_VAR[v]
    return fa * fork(B) + gf * (forkt(B) - fork(B))       # ms per round ADDED (negative = faster); one joint pass


# ---------------------------------------------------------------- memory (allocated basis, ours)
def mem(cell, B, arm):
    a, b = cell["line"]
    d_int = cell["peak"] - cell["plain_peak"]          # INT hooks over the plain arm, same job (MEASURED)
    m = a + b * B + d_int
    if arm == "S":
        m += BF_FIXED + BF_PER_REQ * B
    return m


# AUTHOR basis (device-used; LEDGER_rev7.md:50-55): interp(C63 backbone) + (C - 63) B 2,105,344 + 0.12740196 +
# B [overflow (1 + 128 / 32768) + state 0.417792] / 1000 + 3 x 17,039,360. Overflow MB per request: AUTHOR (:40-46).
BB = [(64, 29.814030), (128, 41.247703), (256, 65.727758), (344, 83.870220)]
OVF = {128: 0.0, 81: 1.6384, 73: 5.332992}


def author_mem(C, B):
    for i in range(len(BB) - 1):
        if BB[i][0] <= B <= BB[i + 1][0] or (i == len(BB) - 2 and B > BB[-1][0]):
            b0, v0 = BB[i]
            b1, v1 = BB[i + 1]
            back = v0 + (v1 - v0) * (B - b0) / float(b1 - b0)
            break
    return back + (C - 63) * B * 2105344 / 1e9 + 0.12740196 + B * (OVF[C] * (1 + 128 / 32768.0) + 0.417792) / 1000 \
        + 3 * 17039360 / 1e9


def author_bmax(C):
    B = 64
    while author_mem(C, B + 1) <= CAP:
        B += 1
    return B, author_mem(C, B)


# ---------------------------------------------------------------- baseline SH / SHJ1 (MEASURED points)
SHP = [  # (C, B, record, label)
    (128, 216, MBS + "mbs_o2s1_2195574_mb1", "mbstrong mb1 plain"),
    (128, 216, MBS + "mbs_o2s1_2195574_mb1n", "mbstrong mb1n nogc"),
    (128, 216, MBS + "mbs_j1s1_2195573_mb1", "mbstrong j1s1 mb1"),
    (128, 217, R + "wsm_v1128_2195573_wm128", "wsm wm128 SH"),
    (128, 222, MBS + "mbs_o2x128_2195574_c128b222n", "mbstrong c128b222n"),
    (81, 290, MBS + "mbs_o2x81_2195574_c81b290", "mbstrong c81b290"),
    (81, 290, MBS + "mbs_o2x81_2195574_c81b290n", "mbstrong c81b290n"),
    (81, 326, MBS + "mbs_o2x81_2195574_c81b326n", "mbstrong c81b326n"),
    (81, 336, MBS + "mbs_o2x81_2195574_c81b336n", "mbstrong c81b336n"),
    (73, 318, MBS + "mbs_o2x73_2195574_c73b318", "mbstrong c73b318"),
    (73, 318, MBS + "mbs_o2x73_2195574_c73b318n", "mbstrong c73b318n"),
    (73, 351, R + "i4m_n373_2195575_i4m73", "i4m n373 SH (L4PTFD OOM after it)"),
    (73, 355, R + "wsm_v173_2195573_wm73", "wsm wm73 SH"),
    (73, 358, MBS + "mbs_o2x73_2195574_c73b358n", "mbstrong c73b358n"),
    (63, 344, MBS + "mbs_o2x63_2195574_c63b344", "mbstrong c63b344"),
    (63, 406, MBS + "mbs_o2x63_2195574_c63b406n", "mbstrong c63b406n"),
]
sh = []
for C, B, p, lab in SHP:
    T = load(p + "/summary.json")
    S = T["SH"]
    med = g(S, "steady", "median_ms") or g(S, "rates", "steady_step_ms")
    mean = g(S, "steady", "mean_ms")
    sh.append(dict(C=C, B=B, med=med, mean=mean, peak=S.get("gpu_peak_alloc_GB"), lab=lab))
for C, cell in cells.items():   # same-job SH of the INT cells (peak field there is not the decode peak: not used)
    S = cell["sh"]
    sh.append(dict(C=C, B=cell["B0"], med=S["steady"]["median_ms"], mean=S["steady"]["mean_ms"], peak=None,
                   lab="%s same-job SH" % cell["run"]))
# SHJ1 serialized (j1a C128 B216, j1b C73 B318) with their same-job SH
shj1 = []
for run, C in (("j1_a1_2195631_j1a", 128), ("j1_b1_2195631_j1b", 73)):
    T = load(R + run + "/summary.json")
    for arm, rec in T["SPEC"].items():
        if isinstance(rec, dict) and "sh_j1" in rec:
            s = rec["sh_j1"]
            shj1.append(dict(C=C, B=T["meta"]["batch"], med=s["steady"]["median_ms"], mean=s["steady"]["mean_ms"],
                             sh_med=T["SH"]["steady"]["median_ms"], sh_mean=T["SH"]["steady"]["mean_ms"], run=run))
            break
# per-row SHJ1 saving (MODEL: the pooling saving scales with B; DERIVED from j1a / j1b)
K_MED = sum((x["sh_med"] - x["med"]) / x["B"] for x in shj1) / len(shj1)
K_MEAN = sum((x["sh_mean"] - x["mean"]) / x["B"] for x in shj1) / len(shj1)

# ---------------------------------------------------------------- outputs
print("REOPT 2026-11-02 (analysis only; NO NEW KERNELS). Caps: cap %.2f, usable %.2f, margin %.2f -> STRICT %.2f GB"
      % (CAP, USABLE, MARGIN, STRICT))
print()
print("== 1. INT cells at the measured B (MEASURED; DERIVED Y_c and T_mean) ==")
for C in (128, 81, 73):
    c = cells[C]
    rt = c["rates"]
    print("C%d B%d %s %s: median %.1f mean %.1f tok/s; tokens/round %.3f -> Y_c %.6f; round median %.3f, T_mean %.2f ms;"
          " phases %s; draft step %.3f; peak %.3f (plain %.3f); wait %.3f ms, NEW %.1f MB/round; store %s"
          % (C, c["B0"], c["run"], c["arm"], rt["median_tok_s"], rt["mean_tok_s"], rt["tokens_per_round_mean"],
             c["Yc"], rt["round_ms_median"], c["Tmean0"], c["ph"], c["draft_step"], c["peak"], c["plain_peak"],
             c["wait"], c["new_mb"], c["store"]))
    print("    joint probe-pass groups (ms): %s" % c["jgroups"])
    print("    plain points: %s" % [(b, {p: round(v[p], 3) for p in PH}, s) for (b, v, s) in c["pts"]])
    print("    rho (INT / plain at B0): %s" % {p: round(c["rho"][p], 4) for p in PH})
    print("    local slopes ms/row %s" % {p: {k: round(c["loc"][p][k], 4) for k in c["loc"][p]} for p in PH})
    print("    memory line (allocated, prefill) %s from %s; INT delta %.3f GB" % (c["line"], c["fitrec"],
                                                                                 c["peak"] - c["plain_peak"]))
print("KAPPA (C81 last segment / chord 64->287) per phase: %s" % {p: round(KAPPA[p], 4) for p in PH})
print()

# anchor check of the S calibration
c = cells[81]
print("S calibration at the anchor C81 B287: S - X saving central %.2f, best %.2f, worst %.2f ms (notes: +7, +21, -6)"
      % (s_net(c, 287, "c"), s_net(c, 287, "best"), s_net(c, 287, "worst")))
print("J at B216 / B287 / B318: FORK %.3f / %.3f / %.3f ms; central %.3f / %.3f / %.3f; best %.3f / %.3f / %.3f;"
      " worst_raw %.3f / %.3f / %.3f" % tuple([fork(b) for b in (216, 287, 318)] +
                                             [j_delta(b, v) for v in ("c", "best", "worst_raw") for b in (216, 287, 318)]))
print()


def q_of(cell, B, arm, tv, sv=None, jv=None):
    tm, tmed = t_int(cell, B, tv)
    d = 0.0
    if arm == "S":
        d = -s_net(cell, B, sv)
    elif arm == "J":
        d = j_delta(B, jv)
    tm += d
    tmed += d
    qmean = 1e3 * B * cell["Yc"] / tm
    qmed = cell["rates"]["median_tok_s"] * (float(B) / cell["B0"]) * (cell["rates"]["round_ms_median"] / tmed)
    return qmean, qmed, tm, tmed


VARS = {"INT": [("c", None, None), ("lo", None, None), ("hi", None, None)],
        "S": [("c", "c", None)] + [(t, s, None) for t in ("lo", "hi") for s in ("best", "worst")],
        "J": [("c", None, "c")] + [(t, None, j) for t in ("lo", "hi") for j in ("best", "worst_raw", "worst_adopt")]}

print("== 2. Sweep per (C, arm): memory first, then T, then Q (mean basis primary) ==")
best = {}
for C in (128, 81, 73):
    cell = cells[C]
    for arm in ("INT", "S", "J"):
        Bcap = 64
        while mem(cell, Bcap + 1, arm) <= CAP:
            Bcap += 1
        rows = []
        for B in range(64, Bcap + 1):
            m = mem(cell, B, arm)
            demo = (arm in ("INT", "J") and B == cell["B0"])        # J does not change memory (MODEL)
            feas_strict = m <= STRICT or demo
            feas_loose = m <= USABLE or demo
            qc = q_of(cell, B, arm, *VARS[arm][0])
            qs = [q_of(cell, B, arm, *v)[0] for v in VARS[arm]]
            qms = [q_of(cell, B, arm, *v)[1] for v in VARS[arm]]
            rows.append(dict(B=B, m=m, fs=feas_strict, fl=feas_loose, demo=demo, qc=qc[0], qmed=qc[1], tm=qc[2],
                             tmed=qc[3], lo=min(qs), hi=max(qs), qmed_lo=min(qms), qmed_hi=max(qms)))
        fs = [r for r in rows if r["fs"]]
        fl = [r for r in rows if r["fl"]]
        bs = max(fs, key=lambda r: r["qc"]) if fs else None
        bl = max(fl, key=lambda r: r["qc"]) if fl else None
        best[(C, arm)] = (bs, bl, rows)
        print("C%d %s: B at the 84.0 cap %d; STRICT best B %s; LOOSE best B %s" % (
            C, arm, Bcap, bs and bs["B"], bl and bl["B"]))
        near = [r for r in rows if bs and abs(r["B"] - bs["B"]) <= 12 and r["B"] % 4 == bs["B"] % 4] + \
            [r for r in rows if r["B"] in (cell["B0"], Bcap) or (bl and r["B"] == bl["B"])]
        seen = set()
        for r in sorted(near, key=lambda r: r["B"]):
            if r["B"] in seen:
                continue
            seen.add(r["B"])
            print("   B%3d mem %.3f GB strict %s loose %s demo %s | T_mean %.1f T_med %.1f ms | Q_mean %.1f [%.1f, %.1f]"
                  " | Q_med %.1f [%.1f, %.1f] | author-basis %.2f GB"
                  % (r["B"], r["m"], r["fs"], r["fl"], r["demo"], r["tm"], r["tmed"], r["qc"], r["lo"], r["hi"],
                     r["qmed"], r["qmed_lo"], r["qmed_hi"], author_mem(C, r["B"])))
print()

print("== 3. Per-C cost functions at B0 and at the STRICT best B (INT arm, central; MODEL away from B0) ==")
for C in (128, 81, 73):
    cell = cells[C]
    for B in sorted(set([cell["B0"], best[(C, "INT")][0]["B"]])):
        parts = {p: cell["rho"][p] * L(cell, p, B, "c") for p in PH}
        w = cell["wait"] * float(B) / cell["B0"]
        tm, tmed = t_int(cell, B, "c")
        print("C%d B%d: d(B) per draft step %.2f, J(B) %.2f (of it the gather wait about %.1f, MODEL x B/B0), commit "
              "%.2f, reset %.2f; T_med %.1f, T_mean %.1f ms; S saving c/best/worst %.1f/%.1f/%.1f; J delta c/best/raw "
              "%.2f/%.2f/%.2f ms" % (C, B, parts["draft"] / 2, parts["joint"], w, parts["commit"], parts["reset"], tmed,
                                    tm, s_net(cell, B, "c"), s_net(cell, B, "best"), s_net(cell, B, "worst"),
                                    j_delta(B, "c"), j_delta(B, "best"), j_delta(B, "worst_raw")))
print()

print("== 4. Baseline SH / SHJ1 (MEASURED points; SHJ1 MODEL away from j1a / j1b) ==")
print("SHJ1 per-row saving (DERIVED from j1a / j1b): median %.5f, mean %.5f ms per request" % (K_MED, K_MEAN))
for x in shj1:
    print("  SHJ1 MEASURED %s C%d B%d: median %.3f ms (%.1f tok/s), mean %.3f ms (%.1f tok/s); same-job SH %.3f / %.3f"
          % (x["run"], x["C"], x["B"], x["med"], 1e3 * x["B"] / x["med"], x["mean"], 1e3 * x["B"] / x["mean"],
             x["sh_med"], x["sh_mean"]))
shbest = {}
for x in sorted(sh, key=lambda x: (x["C"], x["B"])):
    qm = 1e3 * x["B"] / x["med"]
    qn = 1e3 * x["B"] / x["mean"] if x["mean"] else None
    j_med = 1e3 * x["B"] / (x["med"] - K_MED * x["B"])
    j_mean = 1e3 * x["B"] / (x["mean"] - K_MEAN * x["B"]) if x["mean"] else None
    fs = x["peak"] is not None and x["peak"] <= STRICT
    print("  C%d B%d %-40s median %.3f (%.1f) mean %s (%s) peak %s strict %s | SHJ1 MODEL median %.1f mean %s"
          % (x["C"], x["B"], x["lab"], x["med"], qm, x["mean"], qn and round(qn, 1), x["peak"], fs, j_med,
             j_mean and round(j_mean, 1)))
    k = x["C"]
    if k not in shbest or qm > shbest[k][1]:
        shbest[k] = (x, qm)
# SH memory lines per C (DERIVED: two highest-B measured peaks) and the STRICT B cap
print("SH memory lines (allocated peak, DERIVED from the measured peaks) and STRICT B caps:")
for C in (128, 81, 73, 63):
    pk = sorted([x for x in sh if x["C"] == C and x["peak"]], key=lambda x: x["B"])
    lo, hi = pk[0], pk[-1]
    s = (hi["peak"] - lo["peak"]) / float(hi["B"] - lo["B"])
    bcap = int(hi["B"] + (STRICT - hi["peak"]) / s)
    print("  C%d: slope %.4f GB/request (B%d %.3f .. B%d %.3f); STRICT B cap %d; usable cap %d" % (
        C, s, lo["B"], lo["peak"], hi["B"], hi["peak"], bcap, int(hi["B"] + (USABLE - hi["peak"]) / s)))
print()

print("== 5. AUTHOR basis (device-used) largest B under 84.0 per C, and at the chosen configurations ==")
for C in (128, 81, 73):
    print("  C%d: author B_max %d (%.4f GB); at B0 %d: %.4f GB" % ((C,) + author_bmax(C) + (cells[C]["B0"],
                                                                                         author_mem(C, cells[C]["B0"]))))
print("  anchor check C81 B287: %.4f GB (document 83.7655)" % author_mem(81, 287))
print()

print("== 6. Verdict table (STRICT feasibility; mean basis primary) ==")
jb = [x for x in shj1 if x["C"] == 128][0]
base_mean = 1e3 * jb["B"] / jb["mean"]
base_med = 1e3 * jb["B"] / jb["med"]
print("best baseline: SHJ1 C128 B216 (j1a MEASURED): mean %.1f, median %.1f tok/s; inta same-job SHJ1 median DERIVED "
      "%.1f (2,096.1 / 0.5995, LEDGER_rev7.md:758-761); graph + J1 MODEL 3,540.9 (j1a prediction 26)"
      % (base_mean, base_med, 2096.1 / 0.5995))
for arm in ("INT", "S", "J"):
    cand = [(C, best[(C, arm)][0]) for C in (128, 81, 73) if best[(C, arm)][0]]
    for C, r in sorted(cand, key=lambda x: -x[1]["qc"]):
        print("  %-3s C%d B%d mem %.2f GB: Q_mean %.1f [%.1f, %.1f] = %.3f [%.3f, %.3f] x SHJ1 mean; Q_med %.1f [%.1f,"
              " %.1f] = %.3f x SHJ1 median (j1a)"
              % (arm, C, r["B"], r["m"], r["qc"], r["lo"], r["hi"], r["qc"] / base_mean, r["lo"] / base_mean,
                 r["hi"] / base_mean, r["qmed"], r["qmed_lo"], r["qmed_hi"], r["qmed"] / base_med))
print()

print("== 7. Flip thresholds (DERIVED from the central model) ==")
best_int = best[(128, "INT")][0]
q_star = best_int["qc"]
for C in (81, 73):
    r = best[(C, "S")][0]
    tm_int, _ = t_int(cells[C], r["B"], "c")
    need = 1e3 * r["B"] * cells[C]["Yc"] / q_star
    w = cells[C]["wait"] * float(r["B"]) / cells[C]["B0"]
    print("  C%d B%d (STRICT with the BF parts): INT T_mean %.1f ms; to pass C128 INT (%.1f) T_mean <= %.1f ms: S must "
          "save >= %.1f ms per round = %.2f of the gather wait %.1f ms (BF MODEL best %.1f)"
          % (C, r["B"], tm_int, q_star, need, tm_int - need, (tm_int - need) / w, w, s_net(cells[C], r["B"], "best")))
c = cells[128]
need = 1e3 * c["B0"] * c["Yc"] / base_mean
print("  C128 B216: to tie SHJ1 (mean %.1f) T_mean <= %.1f ms vs measured %.1f (gap %.1f ms per round); two drafts "
      "%.1f ms = 2 x %.2f vs SHJ1 step median %.3f (ratio %.3f)" % (base_mean, need, c["Tmean0"], c["Tmean0"] - need,
                                                                  c["ph"]["draft"], c["draft_step"], jb["med"],
                                                                  c["draft_step"] / jb["med"]))
for C in (81, 73):
    cc = cells[C]
    print("  C%d B%d: draft step %.3f vs same-job SH step %.3f (ratio %.3f)" % (
        C, cc["B0"], cc["draft_step"], cc["sh"]["steady"]["median_ms"], cc["draft_step"] / cc["sh"]["steady"]["median_ms"]))
inta_shj1_med = 2096.1 / 0.5995
inta_shj1_mean = 1e3 * 216 / (cells[128]["sh"]["steady"]["mean_ms"] - K_MEAN * 216)
print("  same job (inta): INT / SHJ1 median %.4f (DERIVED SHJ1 %.1f); mean %.4f (SHJ1 mean MODEL %.1f = SH mean step "
      "%.3f - %.5f x 216)" % (2096.1 / inta_shj1_med, inta_shj1_med, 1910.0 / inta_shj1_mean, inta_shj1_mean,
                              cells[128]["sh"]["steady"]["mean_ms"], K_MEAN))
print("  INT accepted-draft Q at C128 B216 (DERIVED, accepted 2.617130 per request per round, LEDGER_rev7.md:774): %.1f"
      % (1e3 * 216 * 2.617130 / cells[128]["Tmean0"]))
print("  store already in every INT arm (record field store.store_GB): %s" % {C: cells[C]["store"] for C in cells})
print("  BF parts at B287: H = 8 basis %.3f GB; H = 2 basis %.3f GB" % (
    BF_FIXED + BF_PER_REQ * 287, BF_FIXED + (65536 + 2 * 32 * 2 * 262) * 287 / 1e9))
