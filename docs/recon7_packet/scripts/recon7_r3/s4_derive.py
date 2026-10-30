"""RECON7 R3 part 4: every DERIVED number of the four R3 rows, computed from record values (read-only).
Each printed line names its operands. MODEL lines say so. No GPU, no write outside stdout."""
import csv
import json
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"


def load(run):
    with open(R + run + "/summary.json") as f:
        return json.load(f)


def p(label, val, how):
    if isinstance(val, float):
        val = round(val, 4)
    print("%-58s %s   <- %s" % (label, val, how))


# ---------------------------------------------------------------- inputs (MEASURED records)
C128 = load("i4dc_d3a_2195575_i4dca")
C73 = load("i4dc_d4b_2195576_i4dcb")
BEST = {128: (C128, "L4PTFDGUDCD", 216), 73: (C73, "L4PTFDGUDKWCD", 318)}
REF = {128: (C128, "L4PTFD", 216), 73: (C73, "L4PTFD", 318)}
TD = {128: 54.8137, 73: 75.2948}                 # MODEL td at the measured configuration (human request :35, :38)
TD_ANCHOR, B_ANCHOR = 68.9864166023, 287         # MODEL anchor (request :18)

# ---------------------------------------------------------------- ROW 1: standalone drafting
print("=" * 30, "ROW 1 standalone drafting")
# the td proxy's own kernel classes (job 2175791 ncu, C 63 resident), linear in B between B 64 and B 344 (MODEL)
P = ("/mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/"
     "output1_hbm_families_long.csv")
rows = list(csv.DictReader(open(P)))
res = {}
for tag in ("c63_b64", "c63_b344"):
    g = {}
    for r in rows:
        if r["tag"] == tag and r["mode"] == "resident":
            g[r["group"]] = g.get(r["group"], 0.0) + float(r["kernel_time_ms"])
    res[tag] = g
    p("resident %s kernel ms by group (both attributions)" % tag, {k: round(v, 3) for k, v in g.items()},
      "CSV sum")
    p("resident %s kernel total" % tag, sum(g.values()), "sum")


def interp(B):
    f = (B - 64) / 280.0
    return {k: res["c63_b64"][k] + f * (res["c63_b344"][k] - res["c63_b64"][k]) for k in res["c63_b64"]}


for B in (216, 318, 287):
    g = interp(B)
    p("MODEL resident kernel ms at B %d (linear 64..344)" % B, {k: round(v, 3) for k, v in g.items()}, "interp")
    p("MODEL resident kernel total at B %d" % B, sum(g.values()), "sum")
for c in (128, 73):
    p("td / MODEL resident kernel total, C %d" % c, TD[c] / sum(interp(BEST[c][2]).values()), "eager / kernel")

# our draft pass classes (I4C profile, L4PTFDP eager, MEASURED one pass)
for run, c in (("i4c_c3a_2195575_i4ca", 128), ("i4c_c4b_2195576_i4cb", 73)):
    d = load(run)
    a = d["SPEC"]["L4PTFDP"]
    dp = a["profile"]["draft_passes"]
    for x in dp:
        cl = {k: round(v["ms"], 3) for k, v in x["classes"].items()}
        nk = {k: int(v["kernels"]) for k, v in x["classes"].items()}
        p("C %d draft pass %s round %s busy / device / kernels / launches" % (c, x["slot"], x["round"]),
          (round(x["busy_ms"], 3), round(x["device_ms"], 3), x["n_kernels"], x["launch_calls"]), "record")
        p("C %d draft pass %s classes ms" % (c, x["slot"]), cl, "record")
        p("C %d draft pass %s classes kernels" % (c, x["slot"]), nk, "record")
        tops = {t["name"][:40]: round(t["ms"], 3) for t in x.get("top_names", []) if t["cls"] in ("POOL", "H2D")}
        p("C %d draft pass %s POOL / H2D top names" % (c, x["slot"]), tops, "record")
        B = BEST[c][2]
        g = interp(B)
        C_ = x["classes"]
        ms = lambda k: C_.get(k, {}).get("ms", 0.0)
        top = {t["name"]: t["ms"] for t in x.get("top_names", [])}
        swap = sum(v for k, v in top.items() if k.startswith("pool_swap"))
        npool = sum(v for k, v in top.items() if k.startswith("nosa_pool_kernel"))
        ours = dict(gemm=ms("GEMM"), attn=ms("ATTN"), selection=ms("STAGE1") + ms("SELECT") + npool,
                    residency=ms("POOL") - npool + ms("DIFF") + ms("H2D"),
                    small=ms("SMALL") + ms("FUSED") + ms("COPY") + ms("TWINF"))
        theirs = dict(gemm=g["gemm"], attn=g["attn_stage2"], selection=g["selection"],
                      residency=g["offload_metadata"] + g["gather_arrivals"], small=g["act_misc"])
        p("C %d draft pass %s pool_swap / nosa_pool ms" % (c, x["slot"]), (round(swap, 3), round(npool, 3)), "top")
        for k in ours:
            p("  C %d %s: ours %.3f vs resident(B %d) %.3f" % (c, k, ours[k], B, theirs[k]), round(ours[k] - theirs[k], 3),
              "difference")
        p("  C %d ours sum vs resident sum" % c, (round(sum(ours.values()), 3), round(sum(theirs.values()), 3),
                                                 round(sum(ours.values()) - sum(theirs.values()), 3)), "sums")
    pd = a.get("probe_draft") or {}
    p("C %d probe_draft labels_median" % c, pd.get("labels_median"), "record")
    p("C %d probe_draft total_median / row_ms_median" % c, (pd.get("total_median"), pd.get("row_ms_median")), "record")
    pr = a["profile"]
    p("C %d draft busy/steady median" % c, pr.get("draft_busy_over_steady_median"), "record")

for c in (128, 73):
    d, arm, B = BEST[c]
    dr, armr, _ = REF[c]
    a, ar = d["SPEC"][arm], dr["SPEC"][armr]
    sh = d["SH"]["rates"]["steady_step_ms"]
    p("C %d best draft step ms / SH / td" % c, (a["draft_step_ms"], sh, TD[c]), "record / record / MODEL")
    p("C %d best draft - td (ms, ratio)" % c, (round(a["draft_step_ms"] - TD[c], 3),
                                             round(a["draft_step_ms"] / TD[c], 4)), "difference")
    p("C %d L4PTFD draft - best draft (graphs + compile)" % c, ar["draft_step_ms"] - a["draft_step_ms"], "difference")
    p("C %d 2 x td vs best draft phase" % c, (round(2 * TD[c], 3), a["phases_median"]["draft"],
                                             round(a["phases_median"]["draft"] - 2 * TD[c], 3)), "per round")
    sk = a["draft"]["skip_per_stream_per_call"]
    streams = 32 * 2 * B
    p("C %d draft skips per stream per call; window blocks read by call A" % c,
      (round(sk, 4), round(64 - sk, 4)), "record; 64 - skip")
    p("C %d call-A KV bytes per draft step (GB)" % c, (64 - sk) * 32768 * streams / 1e9, "(64 - skip) x 32768 x 64 B")
    p("C %d resident KV bytes per step (GB, 64 blocks)" % c, 64 * 32768 * streams / 1e9, "64 x 32768 x 64 B")
# the anchor: ratio interpolated in B between the two measured cells (MODEL; C not separated)
r128 = BEST[128][0]["SPEC"][BEST[128][1]]["draft_step_ms"] / TD[128]
r73 = BEST[73][0]["SPEC"][BEST[73][1]]["draft_step_ms"] / TD[73]
ra = r128 + (B_ANCHOR - 216) / (318 - 216) * (r73 - r128)
p("MODEL best-draft / td ratio at B 287 (linear in B)", ra, "r128 %.4f, r73 %.4f" % (r128, r73))
p("MODEL best draft at the anchor (ms)", ra * TD_ANCHOR, "ratio x 68.9864")
p("MODEL Q at the anchor with that td (others fixed)", 255494 / (ra * TD_ANCHOR), "255,494 / td (reconciliation row 3)")

# ---------------------------------------------------------------- ROW 2: transfers / readiness
print("=" * 30, "ROW 2 packing / transfers / readiness")
MB_MODEL, R_MODEL, S_MODEL = 6.4658285714, 10.7241844536, 0.0868458712132
p("model transfer at the anchor (ms)", 287 * MB_MODEL / R_MODEL, "B x MB / r")
p("model blocks per stream per cohort", MB_MODEL * 1e6 / 2105344, "MB / 2,105,344 B")
for run, arm, c, B in (("i4dc_d3a_2195575_i4dca", "L4PTFD", 128, 216), ("i4dc_d3a_2195575_i4dca", "L4PTFDGUDCD", 128, 216),
                       ("i4d_d3a_2195575_i4da", "L4PTFKW", 128, 216),
                       ("i4dc_d4b_2195576_i4dcb", "L4PTFD", 73, 318), ("i4d_d4b_2195576_i4db", "L4PTFKW", 73, 318),
                       ("i4dc_d4b_2195576_i4dcb", "L4PTFDGUDKWCD", 73, 318)):
    d = load(run)
    a = d["SPEC"][arm]
    kv = a["kv_at_verify"]
    nb, sel, w = kv["new_blocks_per_round"], kv["selected_slots_per_round"], kv["wait_ms_median"]
    mb = kv["new_MB_per_round"]
    rnd = a["rates"]["round_ms_median"]
    y = a["rates"]["median_tok_s"] * rnd / 1000 / B
    print("--", run, arm)
    p("  NEW blocks / MB per round; wait ms", (nb, mb, w), "record")
    p("  MB per request per round", mb / B, "MB / B")
    p("  MB per request per committed token (median round)", mb / B / y, "/ y_med %.4f" % y)
    p("  NEW blocks per stream per round", nb / (64 * B), "/ (32 x 2 x B)")
    p("  not-ready share of verify-selected slots", nb / sel, "NEW / selected")
    p("  delivered rate GB/s (bytes / wait)", mb / w, "MB / ms")
    p("  model transfer with these bytes at r (ms)", mb / R_MODEL, "MB / 10.724")
    p("  wait share of the median round", w / rnd, "wait / round")
    p("  model interference share of a GPU interval", S_MODEL / (1 + S_MODEL), "s / (1 + s)")
    sk = a["draft"]["skip_per_stream_per_call"]
    p("  draft skips per round (all draft rows) vs NEW", (round(sk * a["draft"]["rows"] * 64 / a["rounds"], 1), nb),
      "skip x rows x 64 / rounds")
for run, c, B in (("i4d_d3c_2195575_i4dc", 128, 216), ("i4d_d4d_2195576_i4dd", 73, 318)):
    a = load(run)["SPEC"]["L4PTFD"]
    s = a["store_record"]
    p("acct C %d listed / repeat share / win loads / win prev share" % c,
      (s["listed_pass_after_first"], s["repeat_share_after_first"], s["win_loads_pass_after_first"],
       s["win_prev_share_after_first"]), "record (128-step cell)")
    p("acct C %d listed per (layer, head, request) of X - 1 = 128" % c, s["listed_pass_after_first"] / (64 * B),
      "listed / (32 x 2 x B)")
    p("acct C %d avoidable refetch per pass (repeat + win prev)" % c,
      s["listed_pass_after_first"] * s["repeat_share_after_first"] + s["win_prev_pass_after_first"], "sum")
    p("acct C %d NEW per round of this cell" % c, a["kv_at_verify"]["new_blocks_per_round"], "record")

# ---------------------------------------------------------------- ROW 3: memory
print("=" * 30, "ROW 3 retention / state / capacity")
GB = 1e9
slot_req = 32 * 2 * 32896                       # one slot in every stream of one request
p("one slot per request, all layers and heads (B)", slot_req, "32 x 2 x 32,896")
model_marg = 84.0117 - 83.765543616
model_fixed = 83.765543616 - 287 * model_marg
p("model marginal GB per request at C 81 (B288 - B287)", model_marg, "84.0117 - 83.7655")
p("model implied fixed GB", model_fixed, "83.7655 - 287 x marginal")
p("model backbone C63 GB at B 287", 83.765543616 - 18 * 287 * slot_req / GB - 287 * (1.6384 + 0.417792) / 1000
  - 3 * 17039360 / GB, "peak - 18 x B x slot - B x 2.0562 MB - 3 landing")
for C, B in ((128, 216), (73, 318), (81, 287)):
    m = model_marg + (C - 81) * slot_req / GB
    p("model marginal at C %d (GB/req)" % C, m, "0.24616 + (C - 81) x 2.105 MB")
    p("model peak at C %d B %d (GB)" % (C, B), model_fixed + B * m, "fixed + B x marginal")
    p("model B at 84.0 / 83.75 GB, C %d" % C, ((84.0 - model_fixed) / m, (83.75 - model_fixed) / m), "(cap - fixed) / m")
a73, b73, a128, b128 = 19.020009, 0.1844811, 19.020232, 0.3003276
p("our prefill-line slope per C (GB/req)", (b128 - b73) / 55, "(b128 - b73) / 55 vs slot %.6f" % (slot_req / GB))
for C in (63, 73, 81, 128):
    b = b73 + (C - 73) * (b128 - b73) / 55
    p("MODEL our prefill slope at C %d (GB/req)" % C, b, "linear in C through the two fit lines")
    p("MODEL our B_fit at C %d at 84.0 / 83.75 GB" % C, ((84.0 - a73) / b, (83.75 - a73) / b), "(cap - 19.02) / b")
    p("MODEL our prefill peak at C %d B 287" % C, a73 + 287 * b, "19.02 + 287 b")
for c in (128, 73):
    d, arm, B = BEST[c]
    a = d["SPEC"][arm]
    ar = a["memory"]["arena"]
    shp = d["SH"]["gpu_peak_alloc_GB"]
    p("C %d best arm prefill peak / decode peak / SH peak" % c,
      (a["peak_with_prefill_GB"], a["gpu_peak_alloc_GB"], shp), "record")
    p("C %d spec - SH prefill peak (GB) vs store + CB0" % c,
      (round(a["peak_with_prefill_GB"] - shp, 4), round(ar["store_GB"] + ar["cb0_GB"], 4)), "difference / arena")
    p("C %d arena per request MB: engine / store / CB0" % c,
      (round((ar["store_base"]) * 65792 / B / 1e6, 3), round(ar["store_GB"] * 1000 / B, 3),
       round(ar["cb0_GB"] * 1000 / B, 4)), "blocks x 65,792 B / B")
    p("C %d tentative buffer per request (MB, 32 layers x 64 rows x 1028 B)" % c, 32 * 64 * 1028 / 1e6, "code")
    p("C %d graph pool MB" % c, (a.get("graph") or {}).get("pool_bytes", 0) / 1e6, "record")
    p("C %d prefill slope - arena per request (MB)" % c,
      ((b128 if c == 128 else b73) * 1000 - ar["arena_GB"] * 1000 / B), "fit slope - arena / B")
# OOM record arithmetic (lesson L60)
GiB = 1024 ** 3
p("OOM: capacity / in use / PyTorch alloc / reserved-free / free (GB)",
  (round(79.25 * GiB / GB, 3), round(78.62 * GiB / GB, 3), round(77.90 * GiB / GB, 3), round(212.11 * 2**20 / GB, 3),
   round(119.88 * 2**20 / GB, 3)), "the n373 exception text")
p("OOM: non-PyTorch memory (GB)", (78.62 - 77.90) * GiB / GB - 212.11 * 2**20 / GB, "in use - alloc - reserved-free")
p("OOM: predicted prefill peak at B 351 (GB)", a73 + 351 * b73, "fit line")

# ---------------------------------------------------------------- ROW 4: schedule / yield
print("=" * 30, "ROW 4 schedule / yield / recovery")
for c in (128, 73):
    d, arm, B = BEST[c]
    a = d["SPEC"][arm]
    t = a["schedule_record"]["tokens_acct"]
    G = a["rounds"]
    N = 512 - 1                                            # committed after the warm step
    vfin = t["finals"]
    p("C %d tokens_acct" % c, t, "record")
    p("C %d groups + finals (verification groups)" % c, t["groups"] + vfin, "sum")
    p("C %d rejection rate per verification group (p)" % c, t["corrected"] / (t["groups"] + vfin), "corrected / groups")
    p("C %d acceptance per verified token" % c, t["accepted"] / t["verified"], "accepted / verified")
    p("C %d corrections share of committed" % c, t["corrected"] / t["committed"], "corrected / committed")
    p("C %d accepted per verification group" % c, t["accepted"] / (t["groups"] + vfin), "")
    p("C %d proposed per accepted" % c, t["proposed"] / t["accepted"], "")
    p("C %d model-corrected n = K + p (1 + lag)" % c, 3 + 3 * t["corrected"] / (t["groups"] + vfin), "p from above")
    p("C %d committed per request-round, whole run" % c, t["committed"] / (B * G), "committed / (B x G)")
    p("C %d accepted per request-round, whole run" % c, t["accepted"] / (B * G), "")
    tt = a["schedule_record"]["trace_tail"]
    fin = [max(e["round"] for e in req if e["role"] in ("final", "target")) for req in tt]
    act = sum(x + 1 for x in fin)
    p("C %d active request-rounds / done (drain) request-rounds" % c, (act, B * G - act), "sum(finish + 1)")
    p("C %d mean active rounds per request" % c, act / B, "")
    p("C %d ideal rounds per request (v=2 first group, then 3 per round)" % c, 1 + (N - 2) / 3.0, "1 + (511 - 2) / 3")
    p("C %d excess rounds per reject" % c, (act / B - (1 + (N - 2) / 3.0)) / (t["corrected"] / B), "")
    p("C %d drain share of request-rounds" % c, (B * G - act) / (B * G), "")
    p("C %d rounds: ideal / mean active / batch G" % c, (round(1 + (N - 2) / 3.0, 2), round(act / B, 2), G), "")
    rt = a["rates"]
    y = rt["median_tok_s"] * rt["round_ms_median"] / 1000 / B
    p("C %d y_med committed per request in the median round" % c, y, "median tok/s x round / B")
    apr = a["paper"]["accepted_per_round"]
    ex = set(a.get("steady_excluded_rounds") or [])
    stl = [x for i, x in enumerate(apr) if i > 0 and i % 16 != 0 and i not in ex]
    p("C %d median accepted per steady-like round / B" % c, st.median(stl) / B, "")
    p("C %d accepted per round, full list" % c, apr, "record")
    rows_comp = a["calls"]["joint_passes"] * 4 * B + a["tent"]["calls"]["draft_steps"] * B
    rows_act = a["tent"]["rows"]
    p("C %d GEMM rows computed / active rows / padded share" % c,
      (rows_comp, rows_act, round(1 - rows_act / rows_comp, 4)), "4B x joint + B x draft passes; tent rows")
    p("C %d rows computed / active per committed token" % c,
      (round(rows_comp / t["committed"], 4), round(rows_act / t["committed"], 4)), "model 6B / 3B = 2.0")
    p("C %d drain rows (6 per done request-round)" % c, 6 * (B * G - act), "MODEL count")
    ph = a["phases_median"]
    p("C %d phases median (reset, joint, commit, draft) and sum vs round" % c,
      (ph, round(sum(ph.values()), 3), rt["round_ms_median"]), "record")
    p("C %d round per 3 committed tokens per request (ms)" % c, rt["round_ms_median"] * 3 / y, "round x 3 / y_med")
HB = {128: (3 * 54.8137 + 5.0333, 0.105766), 73: (3 * 75.2948 + 7.8025, 0.078585)}
for c, (h, s) in HB.items():
    B = BEST[c][2]
    p("MODEL cohort interval at C %d (HBM binds)" % c, h * (1 + s), "(3 td + extra) x (1 + s)")
    p("MODEL accepted-draft tok/s at C %d" % c, 1000 * B * 3 / (h * (1 + s)), "1000 B 3 / interval")
