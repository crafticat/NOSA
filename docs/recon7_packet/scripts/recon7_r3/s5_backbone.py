"""RECON7 R3 part 5: which job-2175791 memory line reproduces the model's 'guarded C 63 backbone' (72.248 GB at
B 287, DERIVED in s4 from the October 21 anchor)? Read-only CPU check on output2_memory.csv (MEASURED)."""
import csv

P = "/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv"
rows = list(csv.DictReader(open(P)))
cols = list(rows[0].keys())
print("columns:", cols)
c63 = [r for r in rows if r["C"] == "63" and r["kind"] == "full cell"]
for r in c63:
    print("C 63 B", r["B"], {k: r[k] for k in cols if k not in ("C", "kind", "point") and not k.endswith("_pct")
                              and r[k] not in ("", "0", "0.0")})


def num(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError):
        return None


TARGET = 72.2481
by_b = {int(r["B"]): r for r in c63}
print("B values:", sorted(by_b))
for k in cols:
    if k in ("C", "B", "kind", "point") or k.endswith("_pct"):
        continue
    v64, v344 = num(by_b.get(64, {}), k) if 64 in by_b else None, num(by_b.get(344, {}), k) if 344 in by_b else None
    if v64 is None or v344 is None:
        continue
    s = (v344 - v64) / 280.0
    at = v64 + (287 - 64) * s
    print("%-55s B64 %9.4f B344 %9.4f slope/req %8.5f GB  at B287 %9.4f  (model backbone %.4f; diff %+.4f)" % (
        k, v64, v344, s, at, TARGET, at - TARGET))
