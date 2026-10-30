"""RECON7 R1 part mem: search the 2175791 memory table for the 'guarded C63 backbone' behind 83.765543616 GB."""
import csv

P = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv"
rows = [r for r in csv.DictReader(open(P)) if r["C"] == "63" and r["kind"] == "full cell"]
byB = {int(r["B"]): r for r in rows}
UNIT = 2105344
EXTRA_REQ_MB = 1.6384 + 0.417792
LAND = 3 * 17039360 / 1e9


def per_req(C):
    return (C - 63) * UNIT / 1e9 + EXTRA_REQ_MB / 1e3


target_backbone = 83.765543616 - 287 * per_req(81) - LAND
print("back-solved backbone(287) =", target_backbone)
print("per-request non-backbone at C81 =", per_req(81), " landing =", LAND)
for col in rows[0].keys():
    try:
        y256 = float(byB[256][col]); y344 = float(byB[344][col])
    except (ValueError, KeyError):
        continue
    slope = (y344 - y256) / 88
    v287 = y256 + 31 * slope
    print(f"{col:<55s} B256 {y256:12.6f} B344 {y344:12.6f} slope {slope:.6f} interp287 {v287:.6f} diff_to_target {target_backbone - v287:.6f}")


def model_mem(C, B, backbone_fn):
    return backbone_fn(B) + B * per_req(C) + LAND


du = lambda B: float(byB[256]["device_used_GB"]) + (B - 256) * (float(byB[344]["device_used_GB"]) - float(byB[256]["device_used_GB"])) / 88
off = target_backbone - du(287)
print("offset (guard) =", off)
for C, B in [(81, 287), (81, 288), (128, 216), (73, 318)]:
    bb = du(B) + off
    print(f"C{C} B{B}: backbone(device_used interp/extrap + offset) {bb:.6f}  per-request (C-63) part {B*(C-63)*UNIT/1e9:.6f}  overflow+reserve {B*EXTRA_REQ_MB/1e3:.6f}  landing {LAND:.6f}  total {bb + B*per_req(C) + LAND:.6f}")
