"""RECON7 review V1 (read-only): re-derive record numbers of R1 / R2 / R3 from raw summary.json files.
Independent of the agents' scripts. Prints operands and results. CPU only (srun on amd)."""
import json
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"


def load(run):
    with open(R + run + "/summary.json") as f:
        return json.load(f)


def shape(o, depth=0, pre=""):
    if depth > 1:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, dict):
                print("   %s%s: dict(%d)" % (pre, k, len(v)))
                shape(v, depth + 1, pre + k + ".")
            elif isinstance(v, list):
                print("   %s%s: list(%d) first %r" % (pre, k, len(v), repr(v[:2])[:120]))
            else:
                print("   %s%s: %r" % (pre, k, repr(v)[:120]))


print("=" * 20, "A. I3 conversions (human REQ:37-41)")
for run, B, req_tok, req_osh, req_sj in (("i3_r4a_2195576_i3a", 216, 1709.8, 0.5128, 3.201),
                                         ("i3_r3b_2195575_i3b", 318, 1525.8, 0.5925, 3.094)):
    d = load(run)
    a = d["SPEC"]["L4PTF"]
    sh = d["SH"]
    ph, rt = a["phases_median"], a["rates"]
    osh = rt["median_tok_s"] / sh["rates"]["steady_tok_s"]
    print(run, "B", d["meta"].get("batch"), "C", d["meta"].get("capacity_C"))
    print("  SH steady_tok_s %.1f steady_step_ms %.3f; I3 median_tok_s %.1f -> own-SH %.5f (req %.4f)" % (
        sh["rates"]["steady_tok_s"], sh["rates"]["steady_step_ms"], rt["median_tok_s"], osh, req_osh))
    shms_req = 1000 * B * req_osh / req_tok
    shms_raw = 1000 * B * osh / rt["median_tok_s"]
    print("  SH_ms (req rounded inputs) %.4f; SH_ms (raw) %.4f; record steady_step_ms %.3f" % (
        shms_req, shms_raw, sh["rates"]["steady_step_ms"]))
    print("  joint_sh_steps %.4f; joint (req) %.3f = %.3f x %.4f; joint (raw) %.3f; joint median record %.3f" % (
        a["joint_sh_steps"], req_sj * shms_req, req_sj, shms_req, a["joint_sh_steps"] * sh["rates"]["steady_step_ms"],
        ph["joint"]))
    s = sum(ph.values())
    print("  phases %s sum %.3f round_ms_median %.3f residual %+.3f" % (ph, s, rt["round_ms_median"],
                                                                       rt["round_ms_median"] - s))
    kv = a["kv_at_verify"]
    print("  kv NEW blocks %.1f MB %.3f -> MB/request %.4f; blocks/stream %.4f; MB check %.3f; wait_med %.3f;"
          " rate %.2f GB/s" % (kv["new_blocks_per_round"], kv["new_MB_per_round"], kv["new_MB_per_round"] / B,
                               kv["new_blocks_per_round"] / (64 * B), kv["new_blocks_per_round"] * 32896 / 1e6,
                               kv["wait_ms_median"], kv["new_MB_per_round"] / kv["wait_ms_median"]))
    print("  ready / selected %.5f; NEW / selected %.5f" % (kv["ready_slots_per_round"] / kv["selected_slots_per_round"],
                                                           kv["new_blocks_per_round"] / kv["selected_slots_per_round"]))
    print("  y_med = median_tok_s x round / B = %.4f" % (rt["median_tok_s"] * rt["round_ms_median"] / 1000 / B))
    print("  whole run: effective_tok_s %s, mean_tok_s %s, wall_s %s, prelude_ms %s, rounds %s" % (
        rt.get("effective_tok_s"), rt.get("mean_tok_s"), a.get("wall_s"), a.get("prelude_ms"), a.get("rounds")))
    if run.startswith("i3_r4a"):
        print("  -- arm key shape (I3 C128 L4PTF)")
        shape(a)

print("=" * 20, "B. R2 round span sums (L4PTFD of the profile jobs)")
for run in ("i4c_e4a_2195576_i4ca", "i4c_c4b_2195576_i4cb", "i4c_c3a_2195575_i4ca"):
    d = load(run)
    for arm in ("L4PTFD", "L4PTFDP"):
        a = d["SPEC"][arm]
        ph, rt = a["phases_median"], a["rates"]
        s = sum(ph.values())
        print("  %s %s phases %s sum %.3f round %.3f residual %+.3f draft/2 %.3f draft_step_ms %s" % (
            run, arm, ph, s, rt["round_ms_median"], rt["round_ms_median"] - s, ph["draft"] / 2, a.get("draft_step_ms")))
    pr = d["SPEC"]["L4PTFDP"].get("profile") or {}
    print("  profile keys", list(pr.keys())[:30])

print("=" * 20, "C. TFC union counters (72.24 claim)")
d = load("i3_r2cb_2195574_i3cb")
for arm, a in d["SPEC"].items():
    found = {}

    def walk(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("fork_loaded", "union_sum", "union_samples", "union_mean", "union_max", "union_layers",
                         "visible_blocks", "compared", "vb_rows", "chain_reads"):
                    found[p + k] = v
                walk(v, p + k + ".")
    walk(a)
    if found:
        print("  arm", arm, found)
        fl, us, sa = None, None, None
        for k, v in found.items():
            if k.endswith("fork_loaded"):
                fl = v
            if k.endswith("union_sum"):
                us = v
            if k.endswith("union_samples"):
                sa = v
        if fl and sa:
            print("   fork_loaded / samples %.4f; union_sum / samples %.4f; (fork_loaded - union_sum) / samples %.4f;"
                  " samples / (B 318 x 2) %.1f" % (fl / sa, us / sa, (fl - us) / sa, sa / 636))
U, I = 30985 / 448, 215735 / 3584
print("  model 63 + U - I = %.4f; 64 + U - I = %.4f" % (63 + U - I, 64 + U - I))

print("=" * 20, "D. Best arms (R3)")
BEST = (("i4dc_d3a_2195575_i4dca", "L4PTFDGUDCD", 216, 54.8137), ("i4dc_d4b_2195576_i4dcb", "L4PTFDGUDKWCD", 318, 75.2948))
for run, arm, B, td in BEST:
    d = load(run)
    print("--", run, "arms", list(d["SPEC"].keys()))
    for an in d["SPEC"]:
        a = d["SPEC"][an]
        ph, rt = a["phases_median"], a["rates"]
        kv = a.get("kv_at_verify") or {}
        print("  %s draft_step_ms %s (draft/2 %.3f, -td %+.3f, /td %.4f) phases %s round %.3f residual %+.3f "
              "median_tok_s %s NEW MB %s (/B %.4f) wait %s rate %.2f peak_prefill %s peak_alloc %s" % (
                  an, a.get("draft_step_ms"), ph["draft"] / 2, ph["draft"] / 2 - td, ph["draft"] / 2 / td, ph,
                  rt["round_ms_median"], rt["round_ms_median"] - sum(ph.values()), rt["median_tok_s"],
                  kv.get("new_MB_per_round"), (kv.get("new_MB_per_round") or 0) / B, kv.get("wait_ms_median"),
                  (kv.get("new_MB_per_round") or 0) / (kv.get("wait_ms_median") or 1), a.get("peak_with_prefill_GB"),
                  a.get("gpu_peak_alloc_GB")))
    a = d["SPEC"][arm]
    t = a["schedule_record"]["tokens_acct"]
    print("  tokens_acct", t)
    G = a["rounds"]
    acc = a.get("acceptance") or {}
    print("  acceptance", {k: v for k, v in acc.items()})
    ngroups = t["groups"] + t["finals"]
    print("  p = corrected / (groups + finals) = %d / %d = %.5f; corrected / groups only = %.5f" % (
        t["corrected"], ngroups, t["corrected"] / ngroups, t["corrected"] / t["groups"]))
    print("  accepted / verified %.5f; corrected / committed %.5f; committed - accepted - corrected = %d" % (
        t["accepted"] / t["verified"], t["corrected"] / t["committed"], t["committed"] - t["accepted"] - t["corrected"]))
    print("  whole run accepted per request-round = %d / (%d x %d) = %.4f; committed %.4f" % (
        t["accepted"], B, G, t["accepted"] / (B * G), t["committed"] / (B * G)))
    apr = a["paper"]["accepted_per_round"]
    ex = set(a.get("steady_excluded_rounds") or [])
    print("  accepted_per_round len %d; steady_excluded_rounds %s; steady_rounds_used %s" % (
        len(apr), sorted(ex)[:20], a.get("steady_rounds_used")))
    # three readings of the median-round yield
    allr = [x for i, x in enumerate(apr)]
    nopr = [x for i, x in enumerate(apr) if i > 0 and i % 16 != 0 and i not in ex]
    print("  median accepted/round / B: all rounds %.4f; excl round 0, probe (i%%16==0), excluded %.4f; mean all %.4f" % (
        st.median(allr) / B, st.median(nopr) / B, sum(allr) / len(allr) / B))
    print("  y_med (median_tok_s x round / B) %.4f" % (a["rates"]["median_tok_s"] * a["rates"]["round_ms_median"] / 1000 / B))
    # recovery: independent count from trace_tail
    tt = a["schedule_record"]["trace_tail"]
    print("  trace_tail: %d requests; first request first 6 events %s" % (len(tt), tt[0][:6]))
    roles = {}
    for req in tt:
        for e in req:
            roles[e.get("role")] = roles.get(e.get("role"), 0) + 1
    print("  roles count", roles)
    fin = []
    for req in tt:
        fin.append(max(e["round"] for e in req))
    act = sum(x + 1 for x in fin)
    print("  last round per request: min %d max %d; active request-rounds %d; done %d (%.4f of B x G)" % (
        min(fin), max(fin), act, B * G - act, (B * G - act) / (B * G)))
    ideal = 1 + (511 - 2) / 3.0
    print("  mean active rounds %.3f; ideal %.3f; excess per reject %.4f" % (
        act / B, ideal, (act / B - ideal) / (t["corrected"] / B)))
    print("  memory: peak_with_prefill %s, gpu_peak_alloc %s; SH gpu_peak_alloc %s" % (
        a.get("peak_with_prefill_GB"), a.get("gpu_peak_alloc_GB"), d["SH"].get("gpu_peak_alloc_GB")))
