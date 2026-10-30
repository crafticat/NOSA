"""ana_rev7 part a2: DERIVED numbers for SCORING_i4.md / LEDGER_rev7.md from the I4 run records (read-only).

Per run: start (sacct elapsed back from exit.txt utc_end), registration stamp check, SH row, every arm's rates,
ratios vs same-job SH, vs the same-GPU strongest (nova05 3,426.2 / nova07 3,410.1) and vs the same-day re-checks
(nova05 3,421.6 e3r / nova07 3,402.1 e4r). Then per (family, cell, arm): replicate lists and spread.
"""
import datetime as dt
import glob
import json
import os
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
G = "/mnt/beegfs/ojerbi/r2-runs/gstep/"
GPU = {"f31fc723": "nova05", "33cfb929": "nova07"}
STRONG = {"nova05": 3426.2, "nova07": 3410.1}
RECHK = {"nova05": 3421.6, "nova07": 3402.1}
# STAMPs (registration constant in the cell file) and registration commit times (git log, UTC), MEASURED
REG = {
    "i4a": ("2026-10-29T04:25:58Z", "2026-10-29T04:26:23Z"),
    "i4b": (None, "2026-10-29T07:40:29Z"),
    "i4c": ("2026-10-29T08:57:45Z", "2026-10-29T08:58:05Z"),
    "i4d": ("2026-10-29T09:41:32Z", "2026-10-29T09:41:54Z"),
    "i4g": ("2026-10-29T12:00:39Z", "2026-10-29T12:01:00Z"),
    "i4k": ("2026-10-29T14:57:28Z", "2026-10-29T14:57:45Z"),
    "i4m": ("2026-10-29T12:48:07Z", "2026-10-29T12:48:24Z"),
    "i4u": ("2026-10-29T18:16:01Z", "2026-10-29T18:16:15Z"),
    "i4h": ("2026-10-29T18:47:25Z", "2026-10-29T18:47:41Z"),
    "i4s": ("2026-10-29T22:50:04Z", "2026-10-29T22:50:26Z"),
    "i4cc": ("2026-10-30T04:10:18Z", "2026-10-30T04:10:35Z"),
    "i4dc": ("2026-10-30T07:27:31Z", "2026-10-30T07:27:47Z"),
}


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def hms(s):
    p = s.split(":")
    if "-" in p[0]:
        d, h = p[0].split("-")
        return int(d) * 86400 + int(h) * 3600 + int(p[1]) * 60 + int(p[2])
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])


def rd(x, n=4):
    return None if x is None else round(x, n)


# same-day re-checks
for name in ("gstep_e3r_2195575_gs1r", "gstep_e4r_2195576_gs1r"):
    try:
        g = json.load(open(G + name + "/summary.json"))
        print("RECHECK", name, "gs steady", g["arms"]["gs"]["rates"]["steady_tok_s"], "step ms",
              g["arms"]["gs"]["rates"]["steady_step_ms"], "sh steady", g["arms"]["sh"]["rates"]["steady_tok_s"],
              "verdicts", [(p["name"], p["verdict"]) for p in g["predictions"]])
    except Exception as e:  # noqa: BLE001
        print("RECHECK MISSING", name, e)

AGG = {}
ROWS = {}
steps = sorted(glob.glob(R + "i4*.step"))
print("\n" + "=" * 20, "RUNS", len(steps))
for sdir in steps:
    run = os.path.basename(sdir)[:-5]
    fam = run.split("_")[0]
    cmd = open(sdir + "/command.txt").read().splitlines()
    meta = {}
    for tok in cmd[-1].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            meta[k] = v
    gpu = GPU.get(meta.get("gpu_uuid", "")[:8], "?")
    B = int(meta.get("batch", "0"))
    start = meta.get("utc_start")
    ex = open(sdir + "/exit.txt").read().split() if os.path.exists(sdir + "/exit.txt") else None
    sa = open(sdir + "/sacct.txt").read().strip().split("|") if os.path.exists(sdir + "/sacct.txt") else None
    line = "%-34s %-5s %s B%-4s nosi %s start %s" % (run, fam, gpu, B, meta.get("nosi", "?")[:7], start)
    if ex is None:
        print(line, "| PENDING (no exit.txt)")
        continue
    end = ex[1].split("=")[1]
    exitv = ex[0]
    state = "%s %s el %s" % (sa[2], sa[3], sa[4]) if sa else "no sacct"
    sstart = ts(end) - dt.timedelta(seconds=hms(sa[4])) if sa else None
    stamp, rcommit = REG[fam]
    okreg = None
    if stamp:
        okreg = ts(stamp) < ts(start) and ts(rcommit) < ts(start) and (sstart is None or ts(rcommit) < sstart)
    print(line, "| %s end %s | sacct %s | sacct-start %s (cmd start %s, diff %s s) | stamp<start %s"
          % (exitv, end, state, sstart.strftime("%H:%M:%S") if sstart else None, start[11:19],
             int((sstart - ts(start)).total_seconds()) if sstart else None, okreg))
    sj = R + run + "/summary.json"
    if not os.path.exists(sj):
        print("   no summary.json")
        continue
    d = json.load(open(sj))
    if "SPEC" not in d or "SH" not in d:
        print("   top keys", list(d.keys())[:30])
        for k, v in d.items():
            if isinstance(v, dict) and ("rates" in v or "tok_s" in v):
                print("   ", k, {kk: v.get(kk) for kk in ("rates", "tok_s") if kk in v})
        if "verdict" in d:
            print("   verdict", d["verdict"])
        continue
    sh = d["SH"]
    if not sh or not sh.get("rates") or not sh.get("tok_s"):
        print("   SH has no rates (sweep or accounting cell); SPEC arms", list(d["SPEC"].keys()))
        continue
    T = sh["rates"]["steady_step_ms"]
    shmed = sh["tok_s"]["median_tok_s"]
    print("   SH median %.1f mean %.1f eff %.1f step %.3f ms; SH / strongest %.4f; SH / recheck %.4f; peak %.3f"
          % (shmed, sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], T, shmed / STRONG[gpu],
             shmed / RECHK[gpu], sh.get("gpu_peak_alloc_GB", 0)))
    order = list(d["SPEC"].keys())
    for arm, a in d["SPEC"].items():
        rt = a.get("rates") or {}
        med = rt.get("median_tok_s")
        if med is None:
            print("   %-14s no rates" % arm)
            continue
        rnd = rt["round_ms_median"]
        ph = a["phases_median"]
        y = med * rnd / 1000.0 / B
        bud = y * T
        dd = ph["draft"] / 2.0
        j1 = (ph["joint"] - dd) / 3.0
        J0 = dd - j1
        c = ph["reset"] + ph["commit"]
        m = dd + j1
        aa = y / 3.0
        kv = a.get("kv_at_verify") or {}
        g = (a.get("joint_phases_fork") or {}).get("groups_median") or {}
        rec = dict(run=run, fam=fam, arm=arm, B=B, gpu=gpu, T=T, shmed=shmed, med=med, rnd=rnd, y=y, bud=bud,
                   r_sh=med / shmed, r_st=med / STRONG[gpu], r_rc=med / RECHK[gpu], mean=rt.get("mean_tok_s"),
                   eff=rt.get("effective_tok_s"), effwc=a.get("effective_wo_capture_tok_s"),
                   effws=a.get("effective_wo_setup_tok_s"), draft=ph["draft"], joint=ph["joint"], c=c, d=dd, j1=j1,
                   J0=J0, m=m, aa=aa, g=g, kv=kv, peak=a.get("peak_with_prefill_GB"), pos=order.index(arm),
                   dstep=a.get("draft_step_sh"), jsh=a.get("joint_sh_steps"), nst=rt.get("steady_rounds"),
                   excl=a.get("steady_excluded_rounds"), rounds=a.get("rounds"))
        ROWS[(run, arm)] = rec
        AGG.setdefault((fam, B, arm), []).append(rec)
        print("   %-14s pos %d med %7.1f /SH %.4f /strong %.4f /recheck %.4f | mean %s eff %s effwc %s effws %s | "
              "round %.3f draft %.3f joint %.3f c %.3f | dstep %s SH joint %s SH | peak %s | steady %s excl %s"
              % (arm, rec["pos"], med, rec["r_sh"], rec["r_st"], rec["r_rc"], rec["mean"], rec["eff"], rec["effwc"],
                 rec["effws"], rnd, ph["draft"], ph["joint"], c, rec["dstep"], rec["jsh"], rec["peak"], rec["nst"],
                 rec["excl"]))

print("\n" + "=" * 20, "AGGREGATE per (family, B, arm): median tok/s and ratios over replicates")
for (fam, B, arm), lst in sorted(AGG.items()):
    meds = [x["med"] for x in lst]
    rsh = [x["r_sh"] for x in lst]
    rst = [x["r_st"] for x in lst]
    rrc = [x["r_rc"] for x in lst]
    n = len(lst)
    sp = (max(rsh) - min(rsh)) / st.mean(rsh) * 100 if n > 1 else 0.0
    spm = (max(meds) - min(meds)) / st.mean(meds) * 100 if n > 1 else 0.0
    print("%-5s B%-3d %-14s n %d | med %s (spread %.2f %%) | /SH %s mean %.4f (spread %.2f %%) | /strong %s mean %.4f"
          " | /recheck %s | runs %s"
          % (fam, B, arm, n, [round(v, 1) for v in meds], spm, [rd(v) for v in rsh], st.mean(rsh), sp,
             [rd(v) for v in rst], st.mean(rst), [rd(v) for v in rrc], [x["run"].split("_")[1] + "/" + x["gpu"]
                                                                       for x in lst]))

print("\n" + "=" * 20, "ARM / REF ratios inside the same job (median tok/s)")
PAIRS = [("L4PTFD", "L4PTF"), ("L4PTFK", "L4PTFD"), ("L4PTFKW", "L4PTFD"), ("L4PTFDG", "L4PTFD"),
         ("L4PTFDGKW", "L4PTFD"), ("L4PTFDGKW", "L4PTFDG"), ("L4PTFDGKW", "L4PTFKW"), ("L4PTFDU", "L4PTFD"),
         ("L4PTFDGD", "L4PTFDG"), ("L4PTFDGD", "L4PTFD"), ("L4PTFDGUD", "L4PTFDG"), ("L4PTFDGUD", "L4PTFD"),
         ("L4PTFDGUDKW", "L4PTFDGKW"), ("L4PTFDGUDKW", "L4PTFD"), ("L4PTFDGUDC", "L4PTFDGUD"),
         ("L4PTFDGUDC", "L4PTFD"), ("L4PTFDGUDKWC", "L4PTFDGUDKW"), ("L4PTFDGUDKWC", "L4PTFD"),
         ("L4PTFDGUDCD", "L4PTFDGUDC"), ("L4PTFDGUDCD", "L4PTFD"), ("L4PTFDGUDKWCD", "L4PTFDGUDKWC"),
         ("L4PTFDGUDKWCD", "L4PTFD")]
runs = sorted(set(k[0] for k in ROWS))
for x, y in PAIRS:
    vals = []
    for run in runs:
        if (run, x) in ROWS and (run, y) in ROWS:
            a, b = ROWS[(run, x)], ROWS[(run, y)]
            vals.append((run.split("_")[0] + "_" + run.split("_")[1], a["B"], round(a["med"] / b["med"], 4),
                         round(b["joint"] - a["joint"], 3), round(b["draft"] - a["draft"], 3),
                         round(b["rnd"] - a["rnd"], 3)))
    if vals:
        print("%s / %s: (run, B, ratio, joint saving ms, draft saving ms, round saving ms)" % (x, y))
        for v in vals:
            print("    ", v)

print("\n" + "=" * 20, "FIRST-ARM / EFFECTIVE view (L62): per run, arm position, effective and eff-without-capture")
for run in runs:
    lst = sorted([v for k, v in ROWS.items() if k[0] == run], key=lambda r: r["pos"])
    if lst and lst[0]["B"] in (216, 318):
        print("%-34s" % run, " | ".join("%d:%s med %.1f mean %s eff %s effwc %s" % (r["pos"], r["arm"], r["med"],
                                                                                   r["mean"], r["eff"], r["effwc"])
                                        for r in lst))

print("\n" + "=" * 20, "ROUND DECOMPOSITION + PER-TOKEN MODEL (rev-6 method) for chosen (run, arm)")
CHOSEN = [("i4g_g3a_2195575_i4ga", "L4PTFDG"), ("i4k_k4b_2195576_i4kb", "L4PTFDGKW")]
for run in runs:
    if run.startswith("i4dc_") or run.startswith("i4cc_") or run.startswith("i4s_"):
        for k in ROWS:
            if k[0] == run:
                CHOSEN.append(k)
for key in CHOSEN:
    if key not in ROWS:
        print("MISSING", key)
        continue
    b = ROWS[key]
    T, rnd, bud = b["T"], b["rnd"], b["bud"]
    ratk = {K: b["aa"] * K * T / (K * b["m"] + b["c"]) for K in (2, 3, 4)}
    print("%s %s %s B%d T_SH %.3f | round %.3f y_med %.4f budget %.3f (%.4f SH) | ratio %.4f (budget/round %.4f)"
          % (key[0], key[1], b["gpu"], b["B"], T, rnd, b["y"], bud, bud / T, b["r_sh"], bud / rnd))
    print("   draft %.3f (%.2f SH, %.1f %%) joint %.3f (%.4f SH, %.1f %%) c %.3f (%.2f %%) sum-diff %.3f"
          % (b["draft"], b["draft"] / T, 100 * b["draft"] / rnd, b["joint"], b["joint"] / T, 100 * b["joint"] / rnd,
             b["c"], 100 * b["c"] / rnd, rnd - b["draft"] - b["joint"] - b["c"]))
    print("   per-token: d %.3f j1 %.3f J0 %.3f c %.3f | d+j1 %.3f ms = %.4f SH | a*T %.3f | K2 %.4f K3 %.4f K4 %.4f "
          "limit %.4f | needed cut for 1.0 x SH %.1f %%"
          % (b["d"], b["j1"], b["J0"], b["c"], b["m"], b["m"] / T, b["aa"] * T, ratk[2], ratk[3], ratk[4],
             b["aa"] * T / b["m"], 100 * (1 - b["aa"] * T / b["m"])))
    print("   joint/budget %.4f; drafts-free ratio %.4f; strongest factor SH/strong %.5f"
          % (b["joint"] / bud, bud / (b["joint"] + b["c"]), b["shmed"] / STRONG[b["gpu"]]))
    if b["g"]:
        print("   probe groups", {k: rd(v, 3) for k, v in b["g"].items()})
    if b["kv"]:
        kv = b["kv"]
        w = kv.get("wait_ms_median") or 0
        print("   kv new blocks %.1f MB %.1f wait %.3f -> %.2f GB/s" % (kv.get("new_blocks_per_round", 0),
                                                                     kv.get("new_MB_per_round", 0), w,
                                                                     kv.get("new_MB_per_round", 0) / w if w else -1))
