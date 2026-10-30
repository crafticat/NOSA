"""RECON7 R3 part 3: kernel time and kernel count per group of the job-2175791 ncu records (the td proxy's own
source: the C 63 resident target step), per attribution method, for the cells that bracket the ladder batches.
Read-only; the CSV is MEASURED (ncu kernel time, ms per step); sums are DERIVED."""
import csv
import collections

P = ("/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/"
     "output1_hbm_families_long.csv")
TAGS = ("c63_b64", "c63_b344", "c128_b216", "c128_b64")
rows = list(csv.DictReader(open(P)))
print("columns", list(rows[0].keys()))
seen = collections.OrderedDict()
for r in rows:
    seen[(r["tag"], r["mode"])] = 1
print("cells", list(seen.keys()))
for (tag, mode) in seen:
    if tag not in TAGS:
        continue
    for meth in ("nvtx-module", "kernel-family"):
        g = collections.OrderedDict()
        n = collections.OrderedDict()
        rd = collections.OrderedDict()
        for r in rows:
            if r["tag"] == tag and r["mode"] == mode and r["attribution_method"] == meth:
                k = r["group"]
                g[k] = g.get(k, 0.0) + float(r["kernel_time_ms"] or 0)
                n[k] = n.get(k, 0) + int(float(r["n_kernels"] or 0))
                rd[k] = rd.get(k, 0.0) + float(r["dram_read_GB"] or 0)
        tot = sum(g.values())
        print("%s %s %s: total %.3f ms, %d kernels, DRAM read %.3f GB" % (tag, mode, meth, tot, sum(n.values()),
                                                                         sum(rd.values())))
        for k in g:
            print("    %-18s %8.3f ms  %6d kernels  %7.3f GB read" % (k, g[k], n[k], rd[k]))
