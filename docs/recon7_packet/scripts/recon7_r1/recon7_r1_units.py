"""RECON7 R1 part units: memory formula at the measured configs, the model's largest batch under 84 GB, and work units."""
import csv

P = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv"
rows = {int(r["B"]): r for r in csv.DictReader(open(P)) if r["C"] == "63" and r["kind"] == "full cell"}
UNIT = 2105344
EXTRA_REQ = (1.6384 + 0.417792) / 1e3
LAND = 3 * 17039360 / 1e9
y256, y344 = float(rows[256]["device_used_GB"]), float(rows[344]["device_used_GB"])
slope = (y344 - y256) / 88
guard = 83.765543616 - (y256 + 31 * slope) - 287 * ((81 - 63) * UNIT / 1e9 + EXTRA_REQ) - LAND
print(f"backbone slope {slope:.9f} GB/request; guard constant {guard:.9f} GB")
stg = {b: float(rows[b]["staging_light_restore_snapshot_GB"]) for b in (256, 344)}


def mem(C, B):
    return y256 + (B - 256) * slope + guard + B * ((C - 63) * UNIT / 1e9 + EXTRA_REQ) + LAND


for C, B in [(81, 287), (81, 288), (128, 216), (73, 318)]:
    per_req = (C - 63) * UNIT / 1e9 + EXTRA_REQ
    tot_slope = slope + per_req
    bmax = B + (84.0 - mem(C, B)) / tot_slope
    st = stg[256] + (B - 256) * (stg[344] - stg[256]) / 88
    print(f"C{C} B{B}: model memory {mem(C, B):.6f} GB; marginal per request {tot_slope:.6f}; largest B under 84.0 GB = {bmax:.3f}; "
          f"harness staging snapshot inside the backbone {st:.4f} GB; memory without it {mem(C, B) - st:.6f}")

print("--- work units per round / cohort ---")
W = 15.768289
for C, B, new_mb in [(81, 287, None), (128, 216, 557.462), (73, 318, 3347.635)]:
    print(f"C{C} B{B}: weight bytes per pass {W} GB; per cohort (3 passes) {3*W:.6f} GB; draft rows {3*B}; verify rows {3*B}; "
          f"query positions per request per cohort 6 (3 draft + 3 verify)")
    print(f"   model KV read per pass (draft, 63 groups x 64 layer-heads x 32896 B x B) {63*64*32896*B/1e9:.4f} GB; "
          f"verifier extra per cohort {(30985/448-215735/3584)*64*32896*B/1e9:.4f} GB")
    print(f"   model H2D per cohort (C81 bytes carried) {6.4658285714*B/1e3:.4f} GB = {6.4658285714:.4f} MB per request")
    if new_mb is not None:
        print(f"   MEASURED (SCORING_i3.md:209, L4PTF) NEW host MB per round {new_mb} = {new_mb/B:.4f} MB per request per round; "
              f"ratio to the model's C81 bytes {new_mb/B/6.4658285714:.4f}")
print("--- p correction at measured acceptance (n = K + p(1+lag) vs K + p lag), p = 1 - full-accept fraction unknown; "
      "illustration with p = rejected cohorts / cohorts from token counts is NOT available here ---")
