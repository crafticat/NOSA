# RECON7 synthesis, review response arithmetic. CPU only (srun on amd).
# Inputs: reviewer logs V3 / V4 / V4B (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_v_v3.txt, _v4.txt, _v4b.txt),
# R2 / R3 values, and S1 (recon7_s1.txt) anchor results. Each value carries its source in a comment.

def lin(a, b, t):
    return a + (b - a) * t

T = (287.0 - 216.0) / (318.0 - 216.0)

print("== 1. whole-run window (V4:19-25) ==")
runs = [
    # name, committed, wall_s, prelude_s, capture_s, rounds G, round_ms_median, median tok/s, effective field, GPU
    ("I3 C128 r4a", 110376, 79.180, 0.218, 0.0, 194, 365.423, 1709.8, 1396.7, "nova07"),
    ("I3 C73 r3b", 162498, 127.667, 0.285, 0.0, 209, 585.463, 1525.8, 1275.3, "nova05"),
    ("BEST C128 d3a", 110376, 85.569, 2.457, 9.594118, 194, 324.883, 1924.1, 1292.4, "nova05"),
    ("BEST C73 d4b", 162498, 131.034, 2.562, 10.140224, 209, 524.659, 1699.0, 1242.5, "nova07"),
]
for nm, com, wall, pre, cap, G, rmed, qmed, eff, gpu in runs:
    per_round = (wall - pre) * 1000.0 / G
    per_round_nocap = (wall - pre - cap) * 1000.0 / G
    print("%s (%s): committed / wall %.1f (record effective %.1f); median %.1f; effective / median %.4f" % (nm, gpu, com / wall, eff, qmed, eff / qmed))
    print("   mean wall per round (wall - prelude) / G %.2f ms; without capture %.2f ms; median round %.3f; excess (no capture) %+.2f ms" % (
        per_round, per_round_nocap, rmed, per_round_nocap - rmed))
    print("   committed / (wall - capture) %.1f; committed / (wall - capture - prelude) %.1f" % (com / (wall - cap), com / (wall - cap - pre)))
    if gpu == "nova07":
        print("   effective / strongest EFFECTIVE nova07 2778.3 (CP:12) = %.4f" % (eff / 2778.3))

print("\n== 2. accepted vs committed basis per round (R3:60; V1 via review row 27) ==")
for nm, rmed, B, yacc, ycom, model in (("I3 C128", 365.423, 216, 2.8889, 2.8926, 187.399), ("I3 C73", 585.463, 318, 2.7799, 2.8091, 252.051),
                                       ("BEST C128", 324.883, 216, 2.8889, 2.894, 187.399), ("BEST C73", 524.659, 318, 2.7799, 2.803, 252.051)):
    qa = 1000 * B * yacc / rmed
    print("%s: accepted-draft Q (median round, accepted 2.8889 / 2.7799 per request) %.1f; per 3 accepted %.2f ms = %.3f x model %.2f; committed basis per 3 %.2f" % (
        nm, qa, rmed * 3 / yacc, rmed * 3 / yacc / model, model, rmed * 3 / ycom))

print("\n== 3. re-netting the interference allowance against serialization (review finding 16) ==")
for k, hbm, gb, allow, h2d_inc, h2d_tot, best_wait in (("C128", 59.8470171, 18.3286467, 6.7103, 19.856, 21.302, 20.25),
                                                      ("C73", 83.0972824, 25.2864772, 7.0700, 158.012, 160.136, 115.64)):
    pure = hbm - gb
    print("%s: overlap NOT REALISED (td + extra - GEMM(B)) %.3f; serialization (H2D increment - allowance) %.3f; "
          "joint-boundary H2D total - allowance %.3f; BEST exposed wait - allowance %.3f" % (k, pure, h2d_inc - allow, h2d_tot - allow, best_wait - allow))
anc_pure = 75.9310522 - 22.6567976
print("ANCHOR: overlap not realised %.3f; allowance s x joint dense 7.0372 (V3:16)" % anc_pure)

print("\n== 4. drain in the steady window (V4B:32-34, :56-58) ==")
for k, pall, pfree, steady in (("C128", 210.182, 211.132, 207.094), ("C73", 374.565, 391.019, 383.691)):
    print("%s: probe median all %.3f, drain-free %.3f (shift %+.3f); steady median (drain incl.) %.3f; drain-free probe - steady %+.3f" % (
        k, pall, pfree, pfree - pall, steady, pfree - steady))
print("steady-window drain share: C128 24 / 181 = %.3f, C73 39 / 195 = %.3f; probe-set drain share 2 / 10 = 0.200, 3 / 11 = %.3f" % (24 / 181, 39 / 195, 3 / 11))
shift_anchor = lin(210.182 - 210.182 + (211.132 - 210.182), 391.019 - 374.565, T)
print("anchor sensitivity: per-round joint upward shift if drain rounds were out (MODEL, interpolated probe shift) %.3f ms" % shift_anchor)

print("\n== 5. anchor revised model after the review (S1:55-62 values; V1 now pays the interference allowance) ==")
allow_anc = 7.0372
cases = [
    # name, R, V1(old), V2, V3, Y accepted, committed add
    ("I3 code, model bytes, 27.6 GB/s, Y 2.864", 470.97, 403.74, 357.50, 338.12, 2.864),
    ("I3 code, C-interp bytes, 22.0 GB/s, Y 2.808", 525.99, 403.74, 357.50, 338.12, 2.808),
    ("best code, model bytes, 27.6 GB/s, d 81.56, Y 2.864", 426.35, 359.12, 312.88, 293.50, 2.864),
    ("best code, C-interp bytes, 22.0 GB/s, d 84.40, Y 2.808", 459.60, 364.79, 318.56, 299.17, 2.808),
]
corr = 0.03124  # S1:52
for nm, R, v1, v2, v3, Y in cases:
    v1n = v1 + allow_anc
    out = []
    for lab, iv in (("R", R), ("R + drain shift", R + shift_anchor), ("V1", v1n), ("V2", v2), ("V3", v3)):
        qa = 1000 * 287 * Y / iv
        qc = 1000 * 287 * (Y + corr) / iv
        out.append("%s %.2f ms: accepted %.1f (%.4f / %.4f) committed %.1f (%.4f / %.4f)" % (lab, iv, qa, qa / 3426.2, qa / 3410.1, qc, qc / 3426.2, qc / 3410.1))
    print(nm)
    for o in out:
        print("   " + o)

print("\n== 6. measured-config variants with V1 paying the allowance (S1:24-39 values) ==")
for nm, B, y, R, v1, v2, v3, den, allow in (("I3 C128", 216, 2.8926, 365.423, 345.567, 310.759, 295.158, 3410.1, 6.7103),
                                            ("I3 C73", 318, 2.8091, 585.463, 427.451, 376.710, 355.883, 3426.2, 7.0700),
                                            ("BEST C128", 216, 2.894, 324.883, 304.633, 269.825, 254.224, 3426.2, 6.7103),
                                            ("BEST C73", 318, 2.803, 524.659, 409.019, 358.278, 337.451, 3410.1, 7.0700)):
    v1n = v1 + allow
    print("%s: V1 %.3f ms -> Q %.1f (/strong %.4f); V2 %.4f; V3 %.4f (committed y, unchanged)" % (
        nm, v1n, 1000 * B * y / v1n, 1000 * B * y / v1n / den, 1000 * B * y / v2 / den, 1000 * B * y / v3 / den))

print("\n== 7. reference baselines (REQ:17; review finding 18) ==")
sh128 = 3334.0  # I3 record same-job SH (review finding 18)
sh73 = 1525.8 / 0.5925  # I3 DIAG ratio (CP:69): SH = median / ratio, rounded ratio
print("I3 same-job SH: C128 %.1f; C73 %.1f (from 1525.8 / 0.5925, rounded ratio)" % (sh128, sh73))
print("model Q anchor 3703.5356 / own SH C128 %.4f; model Q at C128 3457.8631 / own SH C128 %.4f; model Q at C73 3784.9448 / own SH C73 %.4f" % (
    3703.5355537 / sh128, 3457.8631 / sh128, 3784.9448 / sh73))
print("model Q at C128 / strongest n05 %.4f, n07 %.4f; at C73 %.4f, %.4f" % (3457.8631 / 3426.2, 3457.8631 / 3410.1, 3784.9448 / 3426.2, 3784.9448 / 3410.1))

print("\n== 8. capacity on one basis (V3:30-54) ==")
for nm, ours_lo, ours_hi, mp, ma, mps, mas in (("C128", 216.0, 216.4, 213.2, 215.7, 227.6, 230.0), ("C73", 318.0, 352.0, 340.9, 342.4, 379.1, 380.0)):
    print("%s: ours %.1f-%.1f; model allocated basis %.1f-%.1f -> margin ours - model %+.1f..%+.1f; snapshot removed %.1f-%.1f -> %+.1f..%+.1f" % (
        nm, ours_lo, ours_hi, mp, ma, ours_lo - ma, ours_hi - mp, mps, mas, ours_lo - mas, ours_hi - mps))
slope81 = lin(0.18448, 0.30033, (81 - 73) / (128.0 - 73))
print("C81 slope (linear in C between the fit lines) %.5f GB per request; B_fit at 84.0 GB %.1f; at 83.75 GB %.1f" % (
    slope81, (84.0 - 19.020) / slope81, (83.75 - 19.020) / slope81))
print("C81 model: allocated basis 313.6-315.5; snapshot removed 345.6-347.1 (V3:43, :46, :49, :52)")
print("basis gap at B287 (B256-344 segment): 72.2480 - 66.3914 = %.4f = slack 4.6087 + outside 1.1187 + const 0.1292 = %.4f" % (72.2480 - 66.3914, 4.6087 + 1.1187 + 0.1292))

print("\n== 9. small checks ==")
print("GEMM measured / GEMM(4B) curve: C128 64.800 / 58.5774 = %.4f (%+.3f ms); C73 79.879 / 86.2389 = %.4f (%+.3f ms)" % (
    64.800 / 58.5774, 64.800 - 58.5774, 79.879 / 86.2389, 79.879 - 86.2389))
print("one-step body above td by arm (C128 / C73): I3 L4PTF elapsed %.2f / %.2f; L4PTFD elapsed %.2f / %.2f; L4PTFDP draft0 busy %.2f / %.2f; BEST elapsed %.2f / %.2f" % (
    151.363 / 2 - 54.8137, 194.550 / 2 - 75.2948, 71.632 - 54.8137, 94.772 - 75.2948, 70.107 - 54.8137, 93.498 - 75.2948, 66.144 - 54.8137, 88.219 - 75.2948))
print("r4a vs d3a /SH: 0.5750 / 0.5700 = %.4f" % (0.5750 / 0.5700))
print("final groups commit 459 / 216 = %.3f tokens (C128)" % (459 / 216))
