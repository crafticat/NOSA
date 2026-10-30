# RECON7 synthesis: per-round split of (measured round - model cohort interval). CPU only, srun on amd.
# Inputs: R1 section 4 (model per-pass and cohort values), R2 3.1 (I3 phases), R3 section 5 (best-arm phases),
# SI section 12 (best-arm rounds), and the record i4dc_r4a (stdout PREDICTION lines 92, 98).
model = {
    "c128": dict(draft2=121.2222, joint=70.1555407, b=191.3777, a=187.3990),
    "c73": dict(draft2=162.4237, joint=97.0356169, b=259.4594, a=252.0512),
}
meas = [
    ("I3 r4a C128", "c128", 151.363, 207.377, 6.004 + 0.110, 365.423, 1709.8, 216),
    ("I3 r3b C73", "c73", 194.550, 382.041, 8.351 + 0.115, 585.463, 1525.8, 318),
    ("BEST d3a C128", "c128", 132.287, 186.011, 6.178, 324.883, 1924.1, 216),
    ("BEST d4b C73", "c73", 176.437, 338.961, 8.546, 524.659, 1699.0, 318),
]
for name, k, dr, jo, c, rnd, q, B in meas:
    m = model[k]
    parts = [("draft phase - 2 td (1 + s)", dr - m["draft2"]), ("joint - joint budget", jo - m["joint"]),
             ("commit + reset (model 0)", c), ("cross-pass lending (b) - (a)", m["b"] - m["a"]),
             ("round median - phase sum", rnd - (dr + jo + c))]
    tot = sum(v for _, v in parts)
    y = q * rnd / B / 1000.0
    per3 = rnd * 3.0 / y
    print("%s: round %.3f - model cohort %.4f = %.3f; parts %s; sum %.3f (check %.4f)" % (
        name, rnd, m["a"], rnd - m["a"], "; ".join("%s %+.3f" % p for p in parts), tot, rnd - m["a"] - tot))
    print("   y_med committed %.4f; round per 3 committed tokens per request %.2f ms = %.3f x model cohort %.2f" % (
        y, per3, per3 / m["a"], m["a"]))
# I4DC r4a replicate (record i4dc_r4a_2195576_i4dca, stdout:92, :98), nova07
for den, lab in ((3410.1, "same-GPU strongest nova07"), (3402.1, "same-day re-check nova07"), (3426.2, "nova05 strongest")):
    print("r4a L4PTFDGUDCD 1916.8 / %.1f (%s) = %.4f" % (den, lab, 1916.8 / den))
# original model Q against the current strongest (not the old reference)
for den in (3426.2, 3410.1, 3160.822684):
    print("original Q 3703.5355537 / %.6f = %.5f" % (den, 3703.5355537 / den))
