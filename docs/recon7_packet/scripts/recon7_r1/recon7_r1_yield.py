"""RECON7 R1 part yield: model yield terms against the I3 token accounting (SCORING_i3.md:221-231, CITED counts)."""
cells = {
    # label: B, rounds (all), accepted, corrected, committed, steady tokens per round mean, median-round yield, transfer ms, gpu ms (model)
    "C128/B216": dict(B=216, rounds=194, acc=109318, cor=1058, com=110376, tpr=570.878, ymed=2.8926, transfer=137.8616870682, gpu=187.3989765171),
    "C73/B318": dict(B=318, rounds=209, acc=160248, cor=2250, com=162498, tpr=779.995, ymed=2.8091, transfer=187.2043490254, gpu=252.0512328135),
}
K, lag = 3, 2
for lab, c in cells.items():
    rr = c["B"] * c["rounds"]
    p = c["cor"] / rr
    print(f"{lab}: request-rounds {rr}; corrections per request-round p_rr = {c['cor']}/{rr} = {p:.6f}")
    print(f"   accepted per request-round {c['acc']/rr:.6f}; committed per request-round {c['com']/rr:.6f}; "
          f"steady mean tokens per request-round {c['tpr']/c['B']:.6f}; median-round yield {c['ymed']}")
    print(f"   correction share of committed {c['cor']/c['com']:.6f}")
    n_old, n_new = K + p * lag, K + p * (1 + lag)
    print(f"   n old {n_old:.6f}; n corrected {n_new:.6f}; transfer (C81 bytes) x n/K: {c['transfer']*n_new/K:.4f} ms vs GPU {c['gpu']:.4f}")
    print(f"   model Q with Y = 3 at the model interval {1000*c['B']*3/c['gpu']:.2f}; with Y = accepted per request-round "
          f"{1000*c['B']*(c['acc']/rr)/c['gpu']:.2f}; with Y = median-round yield {1000*c['B']*c['ymed']/c['gpu']:.2f}")
