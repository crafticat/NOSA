"""RECON7 review V1b (read-only): locate the TFC union / fork counters in i3_r2cb and recompute the per-(layer, KV head,
request) page counts. CPU only."""
import json

d = json.load(open("/mnt/beegfs/ojerbi/r2-runs/spec/i3_r2cb_2195574_i3cb/summary.json"))
print("top keys", list(d.keys()))
WANT = ("fork_loaded", "union_sum", "union_samples", "union_mean", "union_max", "union_layers", "visible_blocks",
        "compared", "vb_rows", "passes", "B", "batch", "capacity_C", "arm", "rows", "R", "E")
hits = []


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in WANT and not isinstance(v, (dict, list)):
                hits.append((p + k, v))
            walk(v, p + k + ".")
    elif isinstance(o, list) and len(o) < 64:
        for i, v in enumerate(o):
            walk(v, p + "[%d]." % i)


walk(d)
for k, v in hits:
    print("  ", k, v)
fl = [v for k, v in hits if k.endswith("fork_loaded")]
us = [v for k, v in hits if k.endswith("union_sum")]
sa = [v for k, v in hits if k.endswith("union_samples")]
vb = [v for k, v in hits if k.endswith("visible_blocks")]
cp = [v for k, v in hits if k.endswith(".compared")]
print("fork_loaded", fl, "union_sum", us, "samples", sa, "visible", vb, "compared", cp)
if fl and sa:
    print("fork_loaded / samples = %.4f; union_sum / samples = %.4f; diff per sample = %.4f" % (
        fl[0] / sa[0], us[0] / sa[0], (fl[0] - us[0]) / sa[0]))
    print("samples / (318 x 2) = %.2f (layer-passes)" % (sa[0] / 636.0))
if vb and cp:
    print("visible / compared = %.4f" % (vb[0] / cp[0]))
U, I = 30985 / 448, 215735 / 3584
print("model: 63 + U - I = %.4f ; U = %.4f ; I = %.4f" % (63 + U - I, U, I))
if fl and sa:
    print("measured fork_loaded / model(63+U-I) = %.4f; union_mean / model = %.4f; fork_loaded / (64 + U - I) = %.4f" % (
        fl[0] / sa[0] / (63 + U - I), us[0] / sa[0] / (63 + U - I), fl[0] / sa[0] / (64 + U - I)))
