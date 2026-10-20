"""R2-M2 r2base P3: the OFFLINE RE-ANALYSIS of an M-P3 run (r2_p3_profile.py), on a CPU node (no GPU, no model; it
imports r2_p3_profile, whose helpers import torch). It re-reads the run's gzipped Kineto traces (trace_SH / trace_R2G /
trace_R2E .json.gz) and its summary.json, and writes ONE new JSON file (--out; refused if it exists). The run directory
is only read.

  REPARSE  every trace re-parsed with the run's OWN recorded class rules (summary.json meta.class_rules) reproduces the
           run's recorded profile exactly (per arm: the total, the classes, the ranges): the re-analysis counts the
           same ops the run counted. MUST.
  COVER    under the corrected CLASS_RULES (r2_p3_profile.py: STAGE1 and ATTN before GEMM) no kernel whose name holds
           "flash_fwd" is GEMM. MUST.
  RECLASS  the profile and the registered Q1-Q6 again with the corrected rules (run 2190876 counted the flash kernels
           as GEMM: Q4 had ATTN 0 / 0). REPORT.
  RANGES   per range ('-' = none), every kernel name with ops and device ms per step, the classes per range, and
           r2.body - sh.kv per name: where the R2 body's extra device time goes. REPORT.
  SEQ      per range, its INSTANCES' kernel sequences in device order (an instance = one call of the ranged function,
           one layer): the distinct sequences with their instance counts and, per position, the median device us. The
           positions follow the ranged function's program order, so each maps to a source line
           (r2_decode_body.decode_fixed, r2_positional_twin.step_fixed, the shipped decode_update_kv). REPORT.
  GRAPH    the R2G replays carry no range (one graph launch per step). Every profiled step is cut at its first kernel,
           the embedding gather (STEP_MARK, once per step in every arm); each R2G step's kernel names are aligned to the
           R2E step of the same index (difflib, exact runs only) and a matched R2G kernel takes the range of its R2E
           partner: device ms per range IN THE GRAPH. REPORT, with the matched fraction (UNDETERMINED below MATCH_MIN
           or when the step marks are not N_WIN per arm).
EXIT: the number of failed MUST gates (READ, REPARSE per arm, COVER); 20 refused (arguments, --out exists, no summary).

    python benchmarks/Efficiency/r2_p3_reclass.py --run /mnt/beegfs/ojerbi/r2-runs/p3/p3_mp3_2190876 \
        --out /mnt/beegfs/ojerbi/r2-runs/p3/reclass_mp3_2190876.json
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import re
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_p3_profile as P          # noqa: E402

ARMS = ("SH", "R2G", "R2E")
STEP_MARK = r"indexSelect"         # F.embedding's gather: the first kernel of every step (2190876: 1.0 per step per arm)
SEQ_RANGES = ("r2.body", "r2.twin", "r2.stage1", "r2.pool", "sh.kv", "sh.k", "sh.cis")
MATCH_MIN = 0.99


def log(msg):
    print("[r2_p3_reclass %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an M-P3 run directory (summary.json, trace_<arm>.json.gz)")
    ap.add_argument("--out", required=True, help="a NEW JSON file")
    ap.add_argument("--top", type=int, default=40, help="names per range in the report")
    ap.add_argument("--variants", type=int, default=3, help="sequence variants per range")
    a = ap.parse_args(argv)
    if a.top < 1 or a.variants < 1:
        ap.error("--top and --variants >= 1")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested: tests/test_r2_p3.py P6)
def rules_of(summary):
    """The class rules a run recorded (meta.class_rules: [[class, regex], ...]) as a tuple of pairs."""
    return tuple((str(c), str(rx)) for c, rx in summary["meta"]["class_rules"])


def reclassified(ops, rules):
    """A copy of ops with cls under `rules` (one classify per distinct (name, cat))."""
    memo = {}
    out = []
    for o in ops:
        k = (o["name"], o["cat"])
        if k not in memo:
            memo[k] = P.classify(o["name"], o["cat"], rules)
        out.append(dict(o, cls=memo[k]))
    return out


def same_profile(mine, recorded, tol=1e-9):
    """The differences between two aggregates (total, classes, ranges; kernels and ms per step): [] when equal."""
    diffs = []
    for part in ("total", "classes", "ranges"):
        a, b = mine.get(part, {}), recorded.get(part, {})
        if part == "total":
            a, b = {"*": a}, {"*": b}
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            if x is None or y is None:
                diffs.append("%s %s: %s / %s" % (part, k, x, y))
                continue
            for f in ("kernels", "ms"):
                if abs(float(x[f]) - float(y[f])) > tol:
                    diffs.append("%s %s %s: %s / %s" % (part, k, f, x[f], y[f]))
    return diffs


def by_range_name(ops, steps):
    """{range ('-' none): {name: [ops, us, cls]}} over the window."""
    d = {}
    for o in ops:
        x = d.setdefault(o["range"] or "-", {}).setdefault(o["name"], [0, 0.0, o["cls"]])
        x[0] += 1
        x[1] += o["dur_us"]
    return d


def table(names, steps, top=None):
    """[dict(name, cls, kernels, ms)] per step, by device time."""
    rows = sorted(names.items(), key=lambda kv: (-kv[1][1], kv[0]))
    if top is not None:
        rows = rows[:top]
    return [dict(name=n[:200], cls=v[2], kernels=round(v[0] / steps, 3), ms=round(v[1] / 1e3 / steps, 4))
            for n, v in rows]


def class_by_range(ops, steps):
    """{range: {class: dict(kernels, ms)}} per step."""
    d = {}
    for o in ops:
        x = d.setdefault(o["range"] or "-", {}).setdefault(o["cls"], [0, 0.0])
        x[0] += 1
        x[1] += o["dur_us"]
    return {r: {c: dict(kernels=round(v[0] / steps, 3), ms=round(v[1] / 1e3 / steps, 4)) for c, v in sorted(cs.items())}
            for r, cs in sorted(d.items())}


def name_delta(a, b, steps):
    """Per name, a - b (two {name: [ops, us, cls]} of one range each): [dict(name, cls, kernels, ms, ms_a, ms_b)] per
    step, by the ms difference (largest first)."""
    out = []
    for n in set(a) | set(b):
        x, y = a.get(n, [0, 0.0, None]), b.get(n, [0, 0.0, None])
        out.append(dict(name=n[:200], cls=x[2] or y[2], kernels=round((x[0] - y[0]) / steps, 3),
                        ms=round((x[1] - y[1]) / 1e3 / steps, 4), ms_a=round(x[1] / 1e3 / steps, 4),
                        ms_b=round(y[1] / 1e3 / steps, 4)))
    out.sort(key=lambda r: (-r["ms"], r["name"]))
    return out


def seq_variants(ops, rng, nvar):
    """The instances of range rng (rid) as kernel sequences in device order; the distinct sequences by instance count
    (top nvar), each with the per-position median device us."""
    inst = {}
    for o in ops:
        if o["range"] == rng and o.get("rid") is not None:
            inst.setdefault(o["rid"], []).append(o)
    groups, cls = {}, {}
    for lst in inst.values():
        lst.sort(key=lambda o: o["ts"])
        key = tuple(o["name"] for o in lst)
        for o in lst:
            cls[o["name"]] = o["cls"]
        groups.setdefault(key, []).append([o["dur_us"] for o in lst])
    top = []
    for key, durs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), len(kv[0])))[:nvar]:
        med = [statistics.median(d[i] for d in durs) for i in range(len(key))]
        top.append(dict(instances=len(durs), ops=len(key), median_us=round(sum(med), 3),
                        seq=[dict(i=i, name=key[i][:160], cls=cls[key[i]], us=round(med[i], 3))
                             for i in range(len(key))]))
    return dict(range=rng, instances=len(inst), variants=len(groups), top=top)


def cut_steps(ops, mark=STEP_MARK):
    """The ops in device order, cut BEFORE every op whose name matches mark: (steps [[op]], ops before the first)."""
    seq = sorted(ops, key=lambda o: o["ts"])
    idx = [i for i, o in enumerate(seq) if re.search(mark, o["name"])]
    if not idx:
        return [], seq
    return [seq[a:b] for a, b in zip(idx, idx[1:] + [len(seq)])], seq[:idx[0]]


def transfer_ranges(g_ops, e_ops, steps, mark=STEP_MARK):
    """GRAPH (module docstring): (status, detail, R2G ops with a range each: the matched R2E kernel's range ('-' none),
    '?unmatched', or '?pre' before the first mark)."""
    sg, g_pre = cut_steps(g_ops, mark)
    se, _ = cut_steps(e_ops, mark)
    if len(sg) != steps or len(se) != steps:
        return "UNDETERMINED", "step marks R2G %d / R2E %d (expected %d each)" % (len(sg), len(se), steps), None
    code = {}
    out = [dict(o, range="?pre") for o in g_pre]
    matched = total = 0
    per_step = []
    for a_ops, b_ops in zip(sg, se):
        a = [code.setdefault(o["name"], len(code)) for o in a_ops]
        b = [code.setdefault(o["name"], len(code)) for o in b_ops]
        rng = [None] * len(a)
        for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                rng[blk.a + k] = b_ops[blk.b + k]["range"] or "-"
        m = sum(1 for r in rng if r is not None)
        matched += m
        total += len(a)
        per_step.append([len(a), len(b), m])
        out.extend(dict(o, range=(r if r is not None else "?unmatched")) for o, r in zip(a_ops, rng))
    frac = matched / total if total else 0.0
    status = "OK" if frac >= MATCH_MIN else "UNDETERMINED"
    detail = "matched %d of %d R2G ops in steps (%.4f; min %.2f); %d before the first mark; per step [R2G, R2E, " \
             "matched] %s" % (matched, total, frac, MATCH_MIN, len(g_pre), per_step[:4])
    return status, detail, out


# ------------------------------------------------------------------ the run
def load_trace(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    spath = os.path.join(a.run, "summary.json")
    if not os.path.isfile(spath):
        print("REFUSED: no %s" % spath, file=sys.stderr)
        return 20
    with open(spath) as f:
        summary = json.load(f)
    old_rules = rules_of(summary)
    steps = P.N_WIN
    gates, ops, agg_new, agg_old = [], {}, {}, {}
    for tag in ARMS:
        path = os.path.join(a.run, "trace_%s.json.gz" % tag)
        try:
            t0 = time.time()
            trace = load_trace(path)
            ops[tag] = P.parse_trace(trace)                              # the corrected CLASS_RULES
            del trace
            log("%s: %d GPU ops in %.0f s" % (tag, len(ops[tag]), time.time() - t0))
        except Exception as e:                                           # noqa: BLE001
            traceback.print_exc()
            gates.append(("READ_%s" % tag, True, False, "%s: %s: %s" % (path, type(e).__name__, str(e)[:200])))
            continue
        gates.append(("READ_%s" % tag, True, True, "%s: %d GPU ops" % (path, len(ops[tag]))))
        agg_new[tag] = P.aggregate(ops[tag], steps)
        agg_old[tag] = P.aggregate(reclassified(ops[tag], old_rules), steps)
        rec = summary.get("profile", {}).get(tag)
        diffs = same_profile(agg_old[tag], rec) if rec else ["no recorded profile"]
        gates.append(("REPARSE_%s" % tag, True, not diffs, "the run's own rules reproduce its recorded profile: %s"
                      % ("yes" if not diffs else "NO: %s" % diffs[:8])))
    flash_gemm = sorted({o["name"][:120] for t in ops for o in ops[t] if "flash_fwd" in o["name"] and o["cls"] == "GEMM"})
    gates.append(("COVER", True, not flash_gemm, "flash kernels left in GEMM: %s" % (flash_gemm or "none")))
    preds = P.q_verdicts(agg_new)
    rn = {t: by_range_name(ops[t], steps) for t in ops}
    res = dict(meta=dict(run=a.run, steps=steps, rules_recorded=old_rules, rules_corrected=P.CLASS_RULES,
                         step_mark=STEP_MARK, match_min=MATCH_MIN, recorded_commit=summary["meta"].get("nosi_commit"),
                         reclass_commit=os.environ.get("RECLASS_COMMIT", "unknown")),
               profile=agg_new,
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds],
               predictions_recorded=summary.get("predictions"),
               class_by_range={t: class_by_range(ops[t], steps) for t in ops},
               names_by_range={t: {r: table(d, steps, a.top) for r, d in rn[t].items()} for t in ops},
               seq={t: [seq_variants(ops[t], r, a.variants) for r in SEQ_RANGES
                        if any(o["range"] == r for o in ops[t])] for t in ops})
    if "R2E" in rn and "SH" in rn:
        res["body_minus_shkv"] = name_delta(rn["R2E"].get("r2.body", {}), rn["SH"].get("sh.kv", {}), steps)
    if "R2G" in ops and "R2E" in ops:
        status, detail, g_ops = transfer_ranges(ops["R2G"], ops["R2E"], steps)
        res["graph"] = dict(status=status, detail=detail)
        if g_ops is not None:
            gagg = P.aggregate(g_ops, steps)
            res["graph"].update(ranges=gagg["ranges"], class_by_range=class_by_range(g_ops, steps))
            if "SH" in rn:
                res["graph"]["body_minus_shkv"] = name_delta(by_range_name(g_ops, steps).get("r2.body", {}),
                                                             rn["SH"].get("sh.kv", {}), steps)
    res["gates"] = [dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:400]))
    for p in preds:
        print("PREDICTION (corrected classes) %s %s: %s" % p)
    for t in ARMS:
        if t in agg_new:
            print("REPORT %s classes (ops/ms per step): %s" % (t, ", ".join(
                "%s %.1f/%.3f" % (k, v["kernels"], v["ms"]) for k, v in agg_new[t]["classes"].items())))
    for r2, sh in (("r2.body", "sh.kv"),):
        for t, cbr in (("R2E", res["class_by_range"].get("R2E", {})), ("SH", res["class_by_range"].get("SH", {}))):
            rng = r2 if t == "R2E" else sh
            print("REPORT %s %s by class: %s" % (t, rng, ", ".join(
                "%s %.1f/%.3f" % (k, v["kernels"], v["ms"]) for k, v in cbr.get(rng, {}).items())))
    for row in res.get("body_minus_shkv", [])[:25]:
        print("REPORT r2.body - sh.kv %+.4f ms (%+.1f ops) [%s] %s" % (row["ms"], row["kernels"], row["cls"],
                                                                        row["name"][:110]))
    for t in ("R2E", "SH"):
        for sv in res["seq"].get(t, []):
            if sv["range"] in ("r2.body", "r2.twin", "sh.kv") and sv["top"]:
                v = sv["top"][0]
                print("REPORT SEQ %s %s: %d instances, %d variants; top variant x%d: %d ops, %.1f us" % (
                    t, sv["range"], sv["instances"], sv["variants"], v["instances"], v["ops"], v["median_us"]))
                for s in v["seq"]:
                    print("    %3d %8.2f us [%s] %s" % (s["i"], s["us"], s["cls"], s["name"][:110]))
    if "graph" in res:
        g = res["graph"]
        print("REPORT GRAPH %s: %s" % (g["status"], g["detail"][:400]))
        for k, v in sorted(g.get("ranges", {}).items()):
            print("REPORT GRAPH range %s: %.1f ops / %.4f ms per step" % (k, v["kernels"], v["ms"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_p3_reclass: %d MUST gate(s) failed; wrote %s" % (nfail, a.out))
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
