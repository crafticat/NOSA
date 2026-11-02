# REOPT 2026-11-02, step 4: the serialized SHJ1 records (j1a / j1b) and any string that names SHJ1. Read-only.
# Analysis only; NO NEW KERNELS. Run only on amd: srun -p amd -c 4 --mem=16G -t 30 python3 -I s4_shj1.py > s4_shj1.out
import json

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
for run in ("j1_a1_2195631_j1a", "j1_b1_2195631_j1b", "int_a1_2195630_inta", "int_c1_2195631_intc",
            "int_d1_2195630_intd"):
    with open(R + run + "/summary.json") as f:
        T = json.load(f)
    print("RUN", run)
    for arm, rec in (T.get("SPEC") or {}).items():
        if isinstance(rec, dict) and "sh_j1" in rec:
            s = rec["sh_j1"]
            print("  arm %s sh_j1 keys %s" % (arm, sorted(s.keys()) if isinstance(s, dict) else s))
            if isinstance(s, dict):
                for k in s:
                    v = s[k]
                    if isinstance(v, dict):
                        print("    %s: %s" % (k, {a: v[a] for a in v if not isinstance(v[a], (list, dict))}))
                    elif not isinstance(v, list):
                        print("    %s = %s" % (k, v))
            break

    def walk(d, path):
        if isinstance(d, dict):
            for k in d:
                walk(d[k], path + "/" + str(k))
        elif isinstance(d, list):
            for i, x in enumerate(d[:400]):
                walk(x, path + "[%d]" % i)
        elif isinstance(d, str) and ("SHJ1" in d or "sh_j1" in d):
            print("  STR %s: %s" % (path, d[:400]))
    walk(T.get("predictions"), "/predictions")
    walk(T.get("gates"), "/gates")
