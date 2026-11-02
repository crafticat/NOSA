# REOPT 2026-11-02, step 2: probe records with a top-level ARM (i4m / wsm probes) and the INT arms' sub-dicts.
# Read-only. Analysis only; NO NEW KERNELS.
# Run only on amd: srun -p amd -c 4 --mem=16G -t 30 python3 -I s2_probe.py > s2_probe.out
import json

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
PROBES = ["i4m_m3p128_2195575_i4mp128", "i4m_m3p73_2195575_i4mp73", "i4m_n373_2195575_i4m73",
          "i4m_n3128_2195575_i4m128", "i4m_v4128_2195576_i4m128",
          "wsm_m1128_2195573_wp128", "wsm_m173_2195573_wp73", "wsm_v1128_2195573_wm128", "wsm_v173_2195573_wm73"]
INT = [("int_a1_2195630_inta", "L4PTFDGUDCDI"), ("int_c1_2195631_intc", "L4PTFDGUDCDI"),
       ("int_d1_2195630_intd", "L4PTFDGUDKWCDIN"), ("int_c1_2195631_intc", "L4PTFD")]


def flat(d, pre, out, depth):
    if isinstance(d, dict):
        for k in d:
            flat(d[k], pre + "/" + str(k), out, depth + 1)
    elif isinstance(d, list):
        if len(d) <= 6 and all(not isinstance(x, (dict, list)) for x in d):
            out.append("%s = %s" % (pre, d))
        elif all(not isinstance(x, (dict, list)) for x in d):
            out.append("%s : list[%d] head %s" % (pre, len(d), repr(d[:5])[:120]))
        else:
            out.append("%s : list[%d] of %s" % (pre, len(d), type(d[0]).__name__))
    else:
        out.append("%s = %s" % (pre, repr(d)[:140]))


for run in PROBES:
    p = R + run + "/summary.json"
    try:
        with open(p) as f:
            T = json.load(f)
    except Exception as e:
        print("RUN", run, "ERROR", repr(e)[:100])
        continue
    print("RUN", run, "meta C", T["meta"].get("capacity_C"), "B", T["meta"].get("batch"),
          "arm", (T["meta"].get("opt") or {}).get("arm"))
    out = []
    for key in ("fit", "verdict", "inferred_cross_c"):
        if key in T:
            flat(T[key], "/" + key, out, 0)
    A = T.get("ARM")
    if isinstance(A, dict):
        for k in ("rates", "phases_median", "draft_step_ms", "gpu_peak_alloc_GB", "peak_with_prefill_GB",
                  "rounds", "arm", "steady_rounds_used", "memory", "mem_arena"):
            if k in A:
                flat(A[k], "/ARM/" + k, out, 0)
    S = T.get("SH")
    if isinstance(S, dict):
        for k in ("rates", "steady", "gpu_peak_alloc_GB", "memory"):
            if k in S:
                flat(S[k], "/SH/" + k, out, 0)
    for line in out:
        print("   ", line)

for run, arm in INT:
    with open(R + run + "/summary.json") as f:
        T = json.load(f)
    rec = T["SPEC"][arm]
    print("INT", run, arm)
    out = []
    for k in ("rates", "rates_incl_capture", "phases_median", "memory", "mem_arena", "store", "kv_at_verify", "int",
              "paper", "tokens", "acceptance", "draft", "bytes_per_round", "vs_sh", "union", "joint_phases_fork"):
        if k in rec:
            flat(rec[k], "/" + k, out, 0)
    for line in out:
        print("   ", line)
    print("    SH:", json.dumps({k: T["SH"][k] for k in ("rates", "steady", "tok_s")})[:600])
