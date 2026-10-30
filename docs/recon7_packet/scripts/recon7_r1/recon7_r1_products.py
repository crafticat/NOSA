"""RECON7 R1 part products: the few combinations quoted in RECON7_R1_model.md section 4 (inputs = logged values)."""
rows = {
    "C81/B287": dict(td=68.9864166023, extra=6.9446356413, g4b=77.8320, gB=22.6567976038, s=0.0868458712, hbm=213.9038854481),
    "C128/B216": dict(td=54.8136966875, extra=5.0333204070, g4b=58.5774, gB=18.3286466996, s=0.1057656197, hbm=169.4744104695),
    "C73/B318": dict(td=75.2948162045, extra=7.8024662261, g4b=86.2389, gB=25.2864771605, s=0.0785851360, hbm=233.6869148397),
}
for k, r in rows.items():
    jh = r["td"] + r["extra"]
    b4 = max(jh, r["g4b"]) * (1 + r["s"])
    dpass = max(r["td"], r["gB"]) * (1 + r["s"])
    print(f"{k}: 3 td = {3*r['td']:.4f} (HBM - extra = {r['hbm']-r['extra']:.4f}); D phase 2 x pass = {2*dpass:.4f}; "
          f"joint budget with GEMM(4B) = max({jh:.4f}, {r['g4b']:.4f}) x (1 + s) = {b4:.4f} ({'dense' if r['g4b'] > jh else 'HBM'} binds)")
