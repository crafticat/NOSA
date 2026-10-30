"""RECON7 review V4 (read-only): (a) the probe-round joint values behind R2's -12.7 ms 'sampling artefact' claim, with
their round numbers and each request's finish round; (b) the whole-run window of the I3 arms (wall, prelude, rounds,
committed) against the steady medians. CPU only."""
import json
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"


def load(run):
    with open(R + run + "/summary.json") as f:
        return json.load(f)


for run, B in (("i4c_c4b_2195576_i4cb", 318), ("i4c_e4a_2195576_i4ca", 216)):
    d = load(run)
    for arm in ("L4PTFDP", "L4PTFD"):
        a = d["SPEC"][arm]
        pj = a.get("probe_joint")
        if not isinstance(pj, dict):
            print(run, arm, "no probe_joint", type(pj))
            continue
        print("==", run, arm, "probe_joint keys", list(pj.keys()))
        pr = pj.get("per_round")
        if isinstance(pr, list):
            for x in pr[:20]:
                if isinstance(x, dict):
                    print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()
                                  if not isinstance(v, (dict, list))})
                else:
                    print("   ", repr(x)[:200])
        elif isinstance(pr, dict):
            for k, v in list(pr.items())[:20]:
                print("   ", k, repr(v)[:200])
        tot = pj.get("total")
        print("   total", repr(tot)[:600])
        tt = a["schedule_record"]["trace_tail"]
        fin = sorted(max(e["round"] for e in req) for req in tt)
        print("   finish rounds: min %d, 10%% %d, median %d, max %d; rounds %d" % (
            fin[0], fin[len(fin) // 10], fin[len(fin) // 2], fin[-1], a["rounds"]))

print("== whole run vs steady (I3 arms and best arms)")
for run, arm, B in (("i3_r4a_2195576_i3a", "L4PTF", 216), ("i3_r3b_2195575_i3b", "L4PTF", 318),
                    ("i4dc_d3a_2195575_i4dca", "L4PTFDGUDCD", 216), ("i4dc_d4b_2195576_i4dcb", "L4PTFDGUDKWCD", 318)):
    a = load(run)["SPEC"][arm]
    rt = a["rates"]
    t = a["schedule_record"]["tokens_acct"]
    G = a["rounds"]
    wall = a["wall_s"]
    pre = a["prelude_ms"] / 1000.0
    print("  %s %s: committed %d, wall_s %.3f, prelude_s %.3f, rounds %d, committed / wall %.1f (effective %s), "
          "mean wall per round (wall - prelude) / G %.3f ms, round_ms_median %.3f, steady_rounds %s, median tok/s %s, "
          "effective / median %.4f, capture_rounds %s, phases_median_incl_capture %s" % (
              run, arm, t["committed"], wall, pre, G, t["committed"] / wall, rt["effective_tok_s"],
              1000 * (wall - pre) / G, rt["round_ms_median"], rt.get("steady_rounds"), rt["median_tok_s"],
              rt["effective_tok_s"] / rt["median_tok_s"], a.get("capture_rounds"), a.get("phases_median_incl_capture")))
    apr = a["paper"]["accepted_per_round"]
    print("    accepted in rounds >= 170: %d of %d (%.2f %%); rounds >= 170: %d of %d" % (
        sum(apr[170:]), sum(apr), 100.0 * sum(apr[170:]) / sum(apr), max(0, G - 170), G))
