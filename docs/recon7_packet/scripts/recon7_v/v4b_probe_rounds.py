"""RECON7 review V4b (read-only): the round numbers of the L4PTFDP probe values and the medians with / without drain
rounds; plus the I3 L4PTF 'joint_phases_fork' probe sample if any. CPU only."""
import json
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
for run in ("i4c_c4b_2195576_i4cb", "i4c_e4a_2195576_i4ca"):
    d = json.load(open(R + run + "/summary.json"))
    a = d["SPEC"]["L4PTFDP"]
    pj = a["probe_joint"]
    print("==", run)
    for k in ("rounds", "n", "probe_joint", "probe_joint_median", "total_median", "plan_median", "span_median",
              "lead_median", "tail_median", "uncovered_median"):
        print("  ", k, repr(pj.get(k))[:400])
    rs, tot = pj.get("rounds"), pj.get("total")
    pjl = pj.get("probe_joint")
    if isinstance(rs, list) and isinstance(tot, list) and len(rs) == len(tot):
        tt = a["schedule_record"]["trace_tail"]
        fin = [max(e["round"] for e in req) for req in tt]
        for r_, v in zip(rs, tot):
            done = sum(1 for x in fin if x < r_)
            print("   round %4s total %.3f; requests already finished %d of %d" % (r_, v, done, len(fin)))
        keep = [v for r_, v in zip(rs, tot) if r_ < 170]
        print("  median all %.3f; median rounds < 170 %.3f (n %d)" % (st.median(tot), st.median(keep), len(keep)))
        if isinstance(pjl, list) and len(pjl) == len(rs):
            keep2 = [v for r_, v in zip(rs, pjl) if r_ < 170]
            print("  probe_joint list median all %.3f; rounds < 170 %.3f" % (st.median(pjl), st.median(keep2)))
    print("  steady joint median %.3f" % a["phases_median"]["joint"])
