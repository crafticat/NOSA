# REOPT 2026-11-02, step 0: structure dump of run records (read-only). Analysis only; NO NEW KERNELS.
# Run only on amd: srun -p amd -c 4 --mem=16G -t 30 python3 -I s0_struct.py <summary.json> [maxdepth]
import json
import sys


def show(d, path, depth, maxd):
    if isinstance(d, dict):
        for k in d:
            v = d[k]
            p = path + "/" + str(k)
            if isinstance(v, (dict, list)) and depth < maxd:
                if isinstance(v, list):
                    n = len(v)
                    if n > 0 and not isinstance(v[0], (dict, list)):
                        print("%s : list[%d] head %s" % (p, n, repr(v[:4])[:160]))
                        continue
                    print("%s : list[%d]" % (p, n))
                    if n > 0:
                        show(v[0], p + "[0]", depth + 1, maxd)
                    continue
                show(v, p, depth + 1, maxd)
            elif isinstance(v, (dict, list)):
                print("%s : %s[%d]" % (p, type(v).__name__, len(v)))
            else:
                print("%s = %s" % (p, repr(v)[:160]))
    elif isinstance(d, list):
        print("%s : list[%d]" % (path, len(d)))


path = sys.argv[1]
maxd = int(sys.argv[2]) if len(sys.argv) > 2 else 3
with open(path) as f:
    T = json.load(f)
print("FILE", path)
show(T, "", 0, maxd)
