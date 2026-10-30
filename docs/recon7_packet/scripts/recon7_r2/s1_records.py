"""RECON7 R2 part 1 (read-only): the arm records that bound the joint pass. For each run: meta dims / batch / C /
pool; SH steady; per arm the phase medians, rates, kv_at_verify, calls, union / fork / tent counters, tokens, probe
splits (joint_phases_fork, probe_joint, probe_draft labels), graph / compile records, and every scalar under a key
that names union / extras / reads. Run inside srun on amd (no GPU)."""
import json
import sys

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
RUNS = [
    ("i3_r4a_2195576_i3a", ("L4PTF",)),
    ("i3_r3b_2195575_i3b", ("L4PTF",)),
    ("i3_r2cb_2195574_i3cb", None),
    ("i4c_c3a_2195575_i4ca", ("L4PTFD", "L4PTFDP")),
    ("i4c_e4a_2195576_i4ca", ("L4PTFD", "L4PTFDP")),
    ("i4c_c4b_2195576_i4cb", ("L4PTFD", "L4PTFDP")),
    ("i4dc_d3a_2195575_i4dca", None),
    ("i4dc_d4b_2195576_i4dcb", None),
    ("i4cc_c3a_2195575_i4cca", None),
    ("i4cc_c4b_2195576_i4ccb", None),
    ("i4s_s3a_2195575_i4sa", None),
    ("i4s_s4b_2195576_i4sb", None),
]
KEYS = ("phases_median", "rates", "kv_at_verify", "joint_sh_steps", "draft_step_ms", "calls", "union", "fork", "tent",
        "tokens", "joint_phases_fork", "rollback_per_round", "lean", "graph", "capture_rounds", "compile", "store",
        "bytes_per_round", "acceptance", "rounds", "wall_s", "prelude_ms", "schedule_record", "hoist", "store_record",
        "level", "opt", "K")
HUNT = ("union", "extra", "reads", "loaded", "visible", "selected", "listed", "fetched", "hits", "window")


def short(v, n=2500):
    s = repr(v)
    return s if len(s) <= n else s[:n] + " ...(%d chars)" % len(s)


def hunt(tag, o, depth=0, out=None):
    """Every scalar (or short list) under a key that contains one of HUNT."""
    if out is None:
        out = []
    if depth > 6:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            t = tag + "." + str(k)
            if any(h in str(k).lower() for h in HUNT) and (not isinstance(v, (dict, list)) or len(repr(v)) < 600):
                out.append((t, v))
            if isinstance(v, (dict, list)):
                hunt(t, v, depth + 1, out)
    elif isinstance(o, list) and len(o) < 50:
        for i, v in enumerate(o):
            if isinstance(v, (dict, list)):
                hunt(tag + "[%d]" % i, v, depth + 1, out)
    return out


def main():
    for run, arms in RUNS:
        try:
            d = json.load(open(R + run + "/summary.json"))
        except Exception as ex:
            print("FAIL", run, type(ex).__name__, ex)
            continue
        print("=" * 30, run)
        m = d.get("meta") or {}
        for k in ("batch", "capacity_C", "pool_blocks", "spec_k", "dims", "S0", "attn_splits", "steps", "probe_every",
                  "profile_rounds", "label_rounds", "strongest_tok_s", "spec_graph", "fork_store_slots", "arms"):
            if k in m:
                print("META", k, short(m[k], 800))
        sh = d.get("SH") or {}
        print("SH rates", short(sh.get("rates")), "steady", short(sh.get("steady")))
        spec = d.get("SPEC") or {}
        use = arms if arms else tuple(spec.keys())
        for a in use:
            rec = spec.get(a)
            if rec is None:
                print("ARM", a, "absent")
                continue
            print("-- ARM", a)
            for k in KEYS:
                if k in rec:
                    print("  ", a, k, short(rec[k]))
            for k in ("probe_joint", "probe_draft"):
                pj = rec.get(k)
                if isinstance(pj, dict):
                    keep = {x: pj[x] for x in pj if x not in ("per_round", "row_ms", "total", "probe_joint", "plan")}
                    print("  ", a, k, short(keep, 3000))
            for t, v in hunt(a, {x: rec[x] for x in rec if x not in ("profile",)}):
                print("   HUNT", t, short(v, 600))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
