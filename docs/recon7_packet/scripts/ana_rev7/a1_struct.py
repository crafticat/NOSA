"""ana_rev7 part a1: print the key structure of selected summary.json files (read-only)."""
import json

FILES = [
    "/mnt/beegfs/ojerbi/r2-runs/spec/i4dc_d3a_2195575_i4dca/summary.json",
    "/mnt/beegfs/ojerbi/r2-runs/gstep/gstep_e3r_2195575_gs1r/summary.json",
]


def walk(o, path, depth, out):
    if depth > 4:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + "." + str(k)
            if isinstance(v, (dict, list)):
                n = len(v)
                out.append("%s  <%s len %d>" % (p, type(v).__name__, n))
                if isinstance(v, list):
                    if n and isinstance(v[0], (dict, list)):
                        walk(v[0], p + "[0]", depth + 1, out)
                else:
                    walk(v, p, depth + 1, out)
            else:
                s = repr(v)
                out.append("%s = %s" % (p, s[:100]))
    elif isinstance(o, list):
        if o and isinstance(o[0], (dict, list)):
            walk(o[0], path + "[0]", depth + 1, out)


for f in FILES:
    print("=" * 20, f)
    try:
        with open(f) as fh:
            d = json.load(fh)
    except FileNotFoundError:
        print("missing")
        continue
    out = []
    walk(d, "", 0, out)
    # drop long per-round arrays' repetition: print at most 400 lines
    for line in out[:600]:
        print(line)
    print("lines", len(out))
