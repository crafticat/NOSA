# REOPT 2026-11-02, step 1: index of run records (read-only). Analysis only; NO NEW KERNELS.
# Run only on amd: srun -p amd -c 4 --mem=16G -t 30 python3 -I s1_index.py <root> [<root> ...] > s1_index.out
# For each summary.json below each root: C, B, arms; per arm: round median, phase medians, draft step, peaks.
import json
import os
import sys


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(x):
    if isinstance(x, float):
        return "%.3f" % x
    return str(x)


def arm_line(name, rec):
    if not isinstance(rec, dict):
        return "    %s: (not a dict)" % name
    rt = rec.get("rates") or {}
    ph = rec.get("phases_median") or {}
    phs = " ".join("%s=%s" % (k, fmt(ph[k])) for k in sorted(ph)) if isinstance(ph, dict) else str(ph)
    return ("    %s: med_tok_s %s mean_tok_s %s round_ms_med %s tpr_mean %s | phases %s | draft_step %s | "
            "peak_alloc %s peak_w_prefill %s rounds %s used %s" % (
                name, fmt(rt.get("median_tok_s")), fmt(rt.get("mean_tok_s")), fmt(rt.get("round_ms_median")),
                fmt(rt.get("tokens_per_round_mean")), phs, fmt(rec.get("draft_step_ms")),
                fmt(rec.get("gpu_peak_alloc_GB")), fmt(rec.get("peak_with_prefill_GB")), rec.get("rounds"),
                rec.get("steady_rounds_used")))


for root in sys.argv[1:]:
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
        if "summary.json" not in fn:
            continue
        p = os.path.join(dp, "summary.json")
        try:
            with open(p) as f:
                T = json.load(f)
        except Exception as e:  # report and go on
            print("FILE", p, "ERROR", repr(e)[:120])
            continue
        if not isinstance(T, dict):
            print("FILE", p, "not a dict")
            continue
        M = T.get("meta") or {}
        print("FILE %s | C %s B %s batches %s arms %s | top %s" % (
            p, M.get("capacity_C"), M.get("batch"), M.get("batches"), M.get("arms"), sorted(T.keys())[:12]))
        sh = T.get("SH")
        if isinstance(sh, dict):
            print("    SH: steady_tok_s %s step_ms %s mean_tok_s %s peak %s" % (
                g(sh, "rates", "steady_tok_s"), g(sh, "rates", "steady_step_ms"), g(sh, "tok_s", "mean_tok_s"),
                sh.get("gpu_peak_alloc_GB")))
        S = T.get("SPEC")
        if isinstance(S, dict):
            for a in S:
                print(arm_line(a, S[a]))
        PB = T.get("per_batch")
        if isinstance(PB, dict):
            for b in PB:
                for a in PB[b]:
                    r = PB[b][a]
                    if a in ("ARM",):
                        print(arm_line("B%s/%s" % (b, a), r))
                    elif a == "SH" and isinstance(r, dict):
                        print("    B%s/SH: %s" % (b, {k: r[k] for k in r if not isinstance(r[k], (dict, list))}))
        F = T.get("fit")
        if isinstance(F, dict):
            print("    fit: %s" % {k: F[k] for k in F if not isinstance(F[k], dict)})
