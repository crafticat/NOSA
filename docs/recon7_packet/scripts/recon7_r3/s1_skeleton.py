"""RECON7 R3 part 1: print the key skeleton of selected summary.json records (read-only).
Lists longer than 6 print their length and the first 3 items; scalars print in full (strings cut at 160 chars)."""
import json
import sys

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
RUNS = sys.argv[1:] or ["i4dc_d3a_2195575_i4dca"]


def short(v):
    s = json.dumps(v)
    return s if len(s) <= 160 else s[:160] + "..."


def walk(x, path, depth):
    if isinstance(x, dict):
        if depth > 9:
            print(path, "{...%d keys}" % len(x))
            return
        for k, v in x.items():
            walk(v, path + "/" + str(k), depth + 1)
    elif isinstance(x, list):
        if len(x) > 6:
            print(path, "LIST len %d first3 %s" % (len(x), short(x[:3])))
        else:
            if all(not isinstance(e, (dict, list)) for e in x):
                print(path, short(x))
            else:
                for i, e in enumerate(x):
                    walk(e, path + "[%d]" % i, depth + 1)
    else:
        print(path, short(x))


for run in RUNS:
    with open(R + run + "/summary.json") as f:
        d = json.load(f)
    print("=" * 30, run)
    walk(d, "", 0)
