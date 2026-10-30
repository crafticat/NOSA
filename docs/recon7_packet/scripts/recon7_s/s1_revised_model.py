# RECON7 synthesis: revised model arithmetic (CPU only; run via srun on amd).
# Every input is copied from a cited input file; see the comment per value.
# R1 = analysis/plan/RECON7_R1_model.md, R2 = RECON7_R2_joint.md, R3 = RECON7_R3_draft_transfer_capacity_schedule.md,
# SI = SCORING_i4.md, CP = CHECKPOINTS.md.

def lin(a, b, t):
    return a + (b - a) * t

B = {"anc": 287.0, "c128": 216.0, "c73": 318.0}
T = (287.0 - 216.0) / (318.0 - 216.0)  # interpolation weight from C128/B216 to C73/B318 (MODEL rule)
print("T (B interpolation weight 216 -> 318 at 287) = %.6f" % T)

# ---- original model terms (R1 sections 1.2, 1.3, 4) ----
td = {"anc": 68.9864166, "c128": 54.8136967, "c73": 75.2948162}
GB = {"anc": 22.6567976, "c128": 18.3286467, "c73": 25.2864772}
G3B = {"anc": 58.3739739, "c128": 45.1165712, "c73": 64.6791766}
extra = {"anc": 6.9446356, "c128": 5.0333204, "c73": 7.8024662}
s = {"anc": 0.0868458712, "c128": 0.105765620, "c73": 0.078585136}
Q_orig = {"anc": 3703.5355537, "c128": 3457.86, "c73": 3784.94}  # R1 section 4 (M)
GPU_orig = {"anc": 232.4805547, "c128": 187.3990, "c73": 252.0512}

print("\n== original-model joint pieces (DERIVED from R1 terms) ==")
ov = {}
for k in ("anc", "c128", "c73"):
    hbm = td[k] + extra[k]
    dense = GB[k] + G3B[k]
    budget = max(hbm, dense) * (1 + s[k])
    serial = td[k] + extra[k] + G3B[k]
    ov[k] = serial - budget
    print("%s: joint HBM %.4f dense %.4f budget %.4f serial %.4f overlap-class (serial - budget) %.4f" % (k, hbm, dense, budget, serial, ov[k]))

# ---- measured I3 terms at the measured configs (R2 section 3.3, serial chain) ----
I3 = {
    "c128": dict(body=15.294, dense=-0.101, attn=11.649, sel=17.830, resfetch=20.631, h2d=19.856,
                 tent=25.915, small=2.743, nonbusy=8.454, joint=207.377, draft=151.363, c=6.004 + 0.110,
                 round=365.423, Q=1709.8, fork_ineff=15.601, gemm_joint=64.799, gemm_draft=19.785,
                 draft0_busy=70.107),
    "c73": dict(body=18.203, dense=-10.501, attn=14.650, sel=24.434, resfetch=158.101, h2d=158.012,
                tent=27.171, small=4.084, nonbusy=-1.873, joint=382.041, draft=194.550, c=8.351 + 0.115,
                round=585.463, Q=1525.8, fork_ineff=20.827, gemm_joint=79.879, gemm_draft=25.705,
                draft0_busy=93.498),
}
# best arm (R2 3.4 label deltas; R3 rows; SI section 12)
BEST = {
    "c128": dict(d_attn=-0.606, d_res=-1.135, d_tent=-17.142, d_dense=-0.372, d_nonbusy=-2.131,
                 joint=186.011, draft=132.287, c=6.178, round=324.883, Q=1924.1, y=2.894, y_acc=2.889,
                 exposed=20.25, gpu_strong=3426.2),
    "c73": dict(d_attn=-1.135, d_res=-16.807, d_tent=-17.445, d_dense=0.070, d_nonbusy=-9.610,
                joint=338.961, draft=176.437, c=8.546, round=524.659, Q=1699.0, y=2.803, y_acc=2.780,
                exposed=115.64, gpu_strong=3410.1),
}
I3["c128"]["gpu_strong"] = 3410.1  # I3 r4a ran on nova07 (R2 data line; CP 1e)
I3["c73"]["gpu_strong"] = 3426.2   # I3 r3b ran on nova05

print("\n== check: I3 serial chain sums to the measured joint (R2 3.3) ==")
for k in ("c128", "c73"):
    m = I3[k]
    serial = td[k] + extra[k] + G3B[k]
    tot = serial + m["body"] + m["dense"] + m["attn"] + m["sel"] + m["resfetch"] + m["tent"] + m["small"] + m["nonbusy"]
    print("%s: serial form %.3f + corrections = %.3f vs measured joint %.3f (residual %.3f)" % (k, serial, tot, m["joint"], m["joint"] - tot))
    y = m["Q"] * m["round"] / B[k] / 1000.0
    m["y"] = y
    print("%s: I3 committed y_med = Q x round / B = %.4f; phase sum %.3f vs round %.3f (residual %+.3f)" % (
        k, y, m["draft"] + m["joint"] + m["c"], m["round"], m["round"] - (m["draft"] + m["joint"] + m["c"])))

# ---- measured-config variants (MODEL removals; not established) ----
print("\n== measured configs: revised model = measured; mechanism variants (MODEL upper bounds) ==")
for arm, D in (("I3", I3), ("BEST", BEST)):
    for k in ("c128", "c73"):
        m = D[k]
        y = m["y"]
        Rr = m["round"]
        if arm == "I3":
            h2d = m["h2d"]
            fork = m["fork_ineff"]
        else:
            h2d = m["exposed"]
            fork = I3[k]["fork_ineff"]  # MODEL: best arm keeps I3's fork kernel (same blob)
        v1 = Rr - h2d
        v2 = v1 - ov[k]
        v3 = v2 - fork
        rows = [("as measured (round median)", Rr), ("V1 transfer hidden", v1), ("V2 + max() overlap in joint", v2),
                ("V3 + fork at the flash byte rate", v3)]
        for name, iv in rows:
            q = 1000.0 * B[k] * y / iv
            print("%s %s %-36s interval %.3f ms  Q %.1f tok/s  /strong(same GPU %.1f) %.4f  /old-ref 3160.82 %.4f" % (
                arm, k, name, iv, q, m["gpu_strong"], q / m["gpu_strong"], q / 3160.822684))

# ---- anchor C81 / B287: revised terms (MODEL, interpolation in B between the measured configs) ----
print("\n== anchor C81 / B287: revised terms (MODEL) ==")
a = {}
# draft step (elapsed), ratio to td interpolated in B
r_i3 = lin((I3["c128"]["draft"] / 2) / td["c128"], (I3["c73"]["draft"] / 2) / td["c73"], T)
r_best = lin((BEST["c128"]["draft"] / 2) / td["c128"], (BEST["c73"]["draft"] / 2) / td["c73"], T)
d_i3 = r_i3 * td["anc"]
d_best_lo = r_best * td["anc"]
d_best_hi = 77.799 + 6.6  # R3 row 1: DR81 measured 77.799 + tentative increment 6.6 (MODEL)
print("draft step ratio to td: I3 %.4f -> d %.3f ms; best %.4f -> d %.3f ms (alt DR81 + 6.6 = %.3f)" % (r_i3, d_i3, r_best, d_best_lo, d_best_hi))
# joint pieces
body = lin(I3["c128"]["body"] / td["c128"], I3["c73"]["body"] / td["c73"], T) * td["anc"]
gj = lin(I3["c128"]["gemm_joint"], I3["c73"]["gemm_joint"], (4 * 287 - 864) / (1272.0 - 864))
gd = lin(I3["c128"]["gemm_draft"], I3["c73"]["gemm_draft"], T)
dense_ex = (gj - gd) - G3B["anc"]
attn_pr = lin(I3["c128"]["attn"] / 216, I3["c73"]["attn"] / 318, T) * 287
fork_pr = lin(I3["c128"]["fork_ineff"] / 216, I3["c73"]["fork_ineff"] / 318, T) * 287
sel = lin(I3["c128"]["sel"] / 648, I3["c73"]["sel"] / 954, T) * 861
tent = lin(I3["c128"]["tent"] / 216, I3["c73"]["tent"] / 318, T) * 287
book = lin(I3["c128"]["resfetch"] - I3["c128"]["h2d"], I3["c73"]["resfetch"] - I3["c73"]["h2d"], T)
small = lin(I3["c128"]["small"] + I3["c128"]["nonbusy"], I3["c73"]["small"] + I3["c73"]["nonbusy"], T)
c = lin(I3["c128"]["c"], I3["c73"]["c"], T)
c_best = lin(BEST["c128"]["c"], BEST["c73"]["c"], T)
serial = td["anc"] + extra["anc"] + G3B["anc"]
print("joint serial form td + extra + G(3B) = %.3f" % serial)
print("body above td %.3f; dense excess (measured GEMM curve at M 1148 %.3f - draft GEMM %.3f - model G(3B)) %.3f" % (body, gj, gd, dense_ex))
print("attention above model %.3f (of it fork inefficiency %.3f); verify-row selection %.3f; tentative + union %.3f" % (attn_pr, fork_pr, sel, tent))
print("residency bookkeeping (non-H2D) %.3f; small + non-busy %.3f; commit + reset I3 %.3f best %.3f" % (book, small, c, c_best))
# best-arm deltas, per-request interpolation
bd = {}
for key in ("d_attn", "d_res", "d_tent", "d_dense", "d_nonbusy"):
    bd[key] = lin(BEST["c128"][key] / 216, BEST["c73"][key] / 318, T) * 287
print("best-arm label deltas at anchor (MODEL, per request): %s" % ", ".join("%s %.3f" % (k2, v) for k2, v in bd.items()))
# transfer options
mb_model = 6.4658285714
mb_i3 = lin(2.581, 10.527, (128 - 81) / (128.0 - 73))
mb_best = lin(2.581, 8.065, (128 - 81) / (128.0 - 73))
print("bytes per request per round: model %.4f MB; C-interpolated I3 %.4f MB, best %.4f MB" % (mb_model, mb_i3, mb_best))

def xfer(mb, rate):
    return 287 * mb / rate  # MB / (GB/s) = ms

for nm, mb in (("model", mb_model), ("I3 C-interp", mb_i3), ("best C-interp", mb_best)):
    print("  exposed transfer %s: %.3f ms (27.6 GB/s) .. %.3f ms (22.0 GB/s)" % (nm, xfer(mb, 27.6), xfer(mb, 22.0)))

Y_lo, Y_hi = 2.808, 2.864  # R3 row 4 Y_rev (C73 p .. C128 p), accepted drafts per request-round
corr = lin(0.02525, 0.03385, T)  # R1 4: corrections per request-round
print("Y_rev accepted %.3f .. %.3f; corrections per request-round (interp) %.5f" % (Y_lo, Y_hi, corr))

def anchor(code, mb, rate, d, Y, committed):
    if code == "I3":
        J = serial + body + dense_ex + attn_pr + sel + tent + book + xfer(mb, rate) + small
        cc = c
    else:
        # d_res is NOT applied: the keep's byte cut is already inside the best-arm byte value (no double count)
        J = (serial + body + dense_ex + attn_pr + sel + tent + book + xfer(mb, rate) + small
             + bd["d_attn"] + bd["d_tent"] + bd["d_dense"] + bd["d_nonbusy"])
        cc = c_best
    R = 2 * d + J + cc
    v1 = R - xfer(mb, rate)
    v2 = v1 - ov["anc"]
    v3 = v2 - fork_pr
    y = Y + (corr if committed else 0.0)
    return J, R, v1, v2, v3, y

print("\n== anchor results (MODEL). Q accepted = 1000 x 287 x Y / interval; committed adds corrections ==")
cases = [
    ("I3 code, model bytes, 27.6 GB/s, Y hi", "I3", mb_model, 27.6, d_i3, Y_hi),
    ("I3 code, I3 C-interp bytes, 22.0 GB/s, Y lo", "I3", mb_i3, 22.0, d_i3, Y_lo),
    ("best code, model bytes, 27.6 GB/s, d ratio, Y hi", "BEST", mb_model, 27.6, d_best_lo, Y_hi),
    ("best code, best C-interp bytes, 22.0 GB/s, d DR81, Y lo", "BEST", mb_best, 22.0, d_best_hi, Y_lo),
]
for name, code, mb, rate, d, Y in cases:
    for committed in (False, True):
        J, R, v1, v2, v3, y = anchor(code, mb, rate, d, Y, committed)
        tag = "committed" if committed else "accepted "
        out = []
        for lab, iv in (("R", R), ("V1", v1), ("V2", v2), ("V3", v3)):
            q = 1000 * 287 * y / iv
            out.append("%s %.2f ms Q %.1f (/3426.2 %.4f, /3410.1 %.4f, /orig-model %.4f)" % (lab, iv, q, q / 3426.2, q / 3410.1, q / Q_orig["anc"]))
        print("%s [%s] joint %.2f | %s" % (name, tag, J, " | ".join(out)))

# what stays above the per-pass reading (b) of the model (R1: 238.0232) in the best-code V3 case, model bytes, d ratio
print("\n== anchor, best code, V3 (all three mechanisms realised): what remains above the model (MODEL) ==")
pb = 2 * td["anc"] * (1 + s["anc"]) + (serial - ov["anc"])
v3 = anchor("BEST", mb_model, 27.6, d_best_lo, Y_hi, False)[4]
parts = [("2 x (d - td(1 + s)) draft excess", 2 * (d_best_lo - td["anc"] * (1 + s["anc"]))),
         ("commit + reset", c_best),
         ("joint body above td", body),
         ("dense excess (+ best delta)", dense_ex + bd["d_dense"]),
         ("attention other than fork inefficiency (+ best delta)", attn_pr - fork_pr + bd["d_attn"]),
         ("verify-row selection", sel),
         ("tentative + union (+ best delta)", tent + bd["d_tent"]),
         ("residency bookkeeping", book),
         ("small + non-busy (+ best delta)", small + bd["d_nonbusy"])]
tot = 0.0
for nm, v in parts:
    tot += v
    print("  %-55s %+.3f" % (nm, v))
print("per-pass model (b) %.3f + parts %.3f = %.3f vs V3 %.3f (check %.4f); cohort model (a) 232.4806" % (pb, tot, pb + tot, v3, v3 - pb - tot))

# original model with corrected single terms at the anchor (R3 row 7 check)
print("\n== anchor: original-model formula with the measured yield only (R3 Y_rev) ==")
for Y in (Y_lo, Y_hi):
    print("Y %.3f: Q = 3703.5355537 x Y / 3 = %.1f" % (Y, 3703.5355537 * Y / 3))
