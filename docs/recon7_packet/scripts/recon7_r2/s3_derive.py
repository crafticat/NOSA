"""RECON7 R2 part 3 (pure arithmetic, no file read): the derivations of RECON7_R2_joint.md from MEASURED inputs copied
from the logs (source lines in the comments) and the CITED model terms. Every printed number is DERIVED from the
operands printed beside it. Run inside srun on amd."""
import statistics as S

# ---------------------------------------------------------------- CITED model terms (HUMAN_RECON_REQUEST_2026-10-30.md:35-40)
MODEL = {
    216: dict(td=54.8137, extra=5.0333, dense=63.4452, s=0.105766, budget=70.1555, C=128),
    318: dict(td=75.2948, extra=7.8025, dense=89.9657, s=0.078585, budget=97.0356, C=73),
}
U, I, G63 = 69.1629464, 60.1939174, 63.0                       # FC:22 (forecast-oct12-17pct.md:22)
W_GB, KV_GB_PER_REQ = 15.768289, 0.132637                     # FC:23
OLD_MSCAN = {128: 14.59, 256: 20.03, 320: 25.46, 384: 27.26, 512: 34.62, 640: 42.93, 672: 51.68, 768: 52.07}  # gm1 :40-49
GEMM861_MODEL = 58.3739739                                    # FC:26 ("M 861 extrapolated past 768")


def mscan(m):
    """The model's GEMM curve: linear interpolation of the old graph-timed scan; past 768 the slope that gives
    GEMM(861) = 58.3739739 (FC:26)."""
    ks = sorted(OLD_MSCAN)
    if m > ks[-1]:
        sl = (GEMM861_MODEL - OLD_MSCAN[768]) / (861 - 768)
        return OLD_MSCAN[768] + (m - 768) * sl
    for a, b in zip(ks[:-1], ks[1:]):
        if a <= m <= b:
            return OLD_MSCAN[a] + (m - a) * (OLD_MSCAN[b] - OLD_MSCAN[a]) / (b - a)
    raise ValueError(m)


# ---------------------------------------------------------------- MEASURED, profiled passes (cpu-logs/recon7_r2_s2.txt)
# joint pass labels, device ms (e4a g112 :50-61; c4b g112 :641-652); per label the GEMM / H2D parts
JOINT = {
    216: dict(attn=40.3121, chain=3.5039, embed=0.0414, fetch=12.1613, head=2.4337, post=58.4702, pre=8.2401,
              resid=21.0460, select=27.3100, twin=3.9888, union=12.0867, write=9.3286,
              gemm_pre=6.840, gemm_post=55.862, gemm_head=2.097, h2d_resid=10.206, h2d_fetch=11.096, fork=37.118,
              busy=198.9227, busy_g96=195.6904, h2d_g96=18.2480),
    318: dict(attn=56.1065, chain=4.5397, embed=0.0586, fetch=91.0844, head=3.4701, post=73.6410, pre=8.8052,
              resid=81.1582, select=38.4720, twin=4.4696, union=12.5069, write=9.6022,
              gemm_pre=6.904, gemm_post=69.996, gemm_head=2.979, h2d_resid=70.258, h2d_fetch=89.878, fork=52.165,
              busy=383.9143, busy_g96=379.2845, h2d_g96=156.1636),
}
# draft0 pass labels, device ms (e4a g112 :109-121; c4b g112 :700-712)
DRAFT = {
    216: dict(amask=0.9399, attnA=19.0631, attnB=1.4346, bmask=0.4907, embed=0.0135, head=0.7509, merge=1.7010,
              post=18.0661, pre=2.5982, resid=12.5763, select=9.4801, twin=0.5173, write=2.4757,
              gemm_pre=1.937, gemm_post=17.185, gemm_head=0.663, h2d=1.446, busy=70.1072, busy1=68.4373,
              busy_g96=69.4464, busy1_g96=68.2837),
    318: dict(amask=1.1277, attnA=27.7635, attnB=2.1280, bmask=0.5070, embed=0.0174, head=1.0284, merge=2.1273,
              post=23.9653, pre=2.7057, resid=14.1411, select=14.0384, twin=0.6352, write=3.3126,
              gemm_pre=1.954, gemm_post=22.855, gemm_head=0.896, h2d=2.124, busy=93.4977, busy1=91.5312,
              busy_g96=93.8438, busy1_g96=91.0762),
}
# steady phase medians (cpu-logs/recon7_r2_s1.txt): I3 L4PTF :26-27 / :89-90; I4C L4PTFD (same job as the profile)
# e4a :290-291, c4b :411-412; best arms d3a :636-637, d4b :818-819
I3 = {216: dict(reset=0.110, joint=207.377, commit=6.004, draft=151.363, round=365.423, sh=64.787, gpu="nova07"),
      318: dict(reset=0.115, joint=382.041, commit=8.351, draft=194.550, round=585.463, sh=123.489, gpu="nova05")}
REF = {216: dict(reset=0.115, joint=207.457, commit=5.977, draft=143.987, round=357.981, run="i4c_e4a (nova07)"),
       318: dict(reset=0.117, joint=383.231, commit=8.344, draft=189.278, round=581.517, run="i4c_c4b (nova07)")}
BEST = {216: dict(arm="L4PTFDGUDCD", reset=0.12, joint=186.011, commit=6.058, draft=132.287, round=324.883,
                  run="i4dc_d3a (nova05)", wait=20.247, new_blocks=16946.2),
        318: dict(arm="L4PTFDGUDKWCD", reset=0.125, joint=338.961, commit=8.421, draft=176.437, round=524.659,
                  run="i4dc_d4b (nova07)", wait=115.643, new_blocks=77960.2)}
# eager probe groups (joint_phases_fork.groups_median): I3 :35 / :98; best :646 / :828
PROBE_I3 = {216: dict(attention=40.9338, chain=3.8507, check=0.0952, dense_next=96.1311, fetch=13.0201,
                      residency=20.8722, union=14.9745, write=11.1457, total=202.0649, wait=20.205),
            318: dict(attention=56.446, chain=4.9439, check=0.0952, dense_next=121.2191, fetch=86.3887,
                      residency=81.4029, union=14.9064, write=10.6363, total=376.2524, wait=153.828)}
PROBE_BEST = {216: dict(attention=40.3282, chain=3.6326, check=0.0952, dense_next=95.7588, fetch=12.6443,
                        residency=20.1134, union=6.1383, write=3.0582, total=182.83),
              318: dict(attention=55.3114, chain=4.6643, check=0.0942, dense_next=121.2887, fetch=86.3846,
                        residency=64.6001, union=5.9085, write=2.4689, total=342.782)}
KV = {216: dict(new=16946.2, mb=557.462, sel=2619984.283, ready=2603038.1),
      318: dict(new=101764.2, mb=3347.635, sel=3841019.051, ready=3739254.8)}       # kv_at_verify :28 / :91
# check cell i3cb TFC (C 73 B 318, 32 teacher passes of 4 verify rows): summary.json :1085, :1133-1142
TFC = dict(fork_loaded=47046615, union_mean=71.17643, union_max=99, visible=166682880, vb_rows=2605056,
           win_host=1711345, extras_host=3021858, passes=32, B=318)
L, H, BLK = 32, 2, 32896
PARAMS = {"wqkv": 4608 * 4096, "wo": 4096 * 4096, "gate_up": 32768 * 4096, "down": 4096 * 16384}   # gm1 meta shapes
LM = 73448 * 4096
PEAK_TF = 312.0                                                # A100 bf16 dense tensor peak (vendor spec; CITED)


def r(x, n=3):
    return round(x, n)


def model_terms():
    print("== 1. MODEL terms (CITED) and checks (DERIVED)")
    for B, m in MODEL.items():
        f = W_GB / (W_GB + B * KV_GB_PER_REQ)
        ex = m["td"] * (1 - f) * (U - I) / G63
        hbm = m["td"] + m["extra"]
        bud = max(hbm, m["dense"]) * (1 + m["s"])
        gb, gbk = mscan(B), mscan(3 * B)
        kvshare = m["td"] * (1 - f)
        print("B %d: f = %.4f / (%.4f + %d x %.6f) = %.6f; extra = td(1-f)(U-I)/63 = %.4f (cited %.4f); HBM = td + extra"
              " = %.4f; dense %.4f; budget = max x (1 + s) = %.4f (cited %.4f); interference allowance = dense x s = %.4f"
              % (B, W_GB, W_GB, B, KV_GB_PER_REQ, f, ex, m["extra"], hbm, m["dense"], bud, m["budget"],
                 m["dense"] * m["s"]))
        print("   model GEMM curve (old scan, gm1 :40-49): GEMM(%d) = %.4f, GEMM(%d) = %.4f, sum %.4f (cited dense %.4f, "
              "diff %.4f)" % (B, gb, 3 * B, gbk, gb + gbk, m["dense"], gb + gbk - m["dense"]))
        print("   model KV-read share of td = td(1 - f) = %.4f; weight share td f = %.4f; KV share + extra = %.4f"
              % (kvshare, m["td"] * f, kvshare + m["extra"]))
        ser = m["td"] + m["extra"] + (m["dense"] - gb)
        print("   SERIAL FORM (MODEL, this file): td + extra + (dense - GEMM(B)) = %.4f + %.4f + (%.4f - %.4f) = %.4f; "
              "serial - budget = %.4f = (HBM - GEMM(B) = %.4f) - (interference %.4f)"
              % (m["td"], m["extra"], m["dense"], gb, ser, ser - m["budget"], hbm - gb, m["dense"] * m["s"]))
        m.update(f=f, gb=gb, ser=ser, kvshare=kvshare)


def work_counts():
    print("== 2. WORK COUNTS (DERIVED)")
    nlh = TFC["passes"] * L * H * TFC["B"]
    up = TFC["fork_loaded"] / nlh
    vis = TFC["visible"] / TFC["vb_rows"]
    print("i3cb TFC: per (layer, KV head, request): fork pages %d / %d = %.3f; union_mean %.3f (max %d); window host loads "
          "%.3f; listed store blocks %.3f; per (layer, KV head, row) visible pages %d / %d = %.3f"
          % (TFC["fork_loaded"], nlh, up, TFC["union_mean"], TFC["union_max"], TFC["win_host"] / nlh,
             TFC["extras_host"] / nlh, TFC["visible"], TFC["vb_rows"], vis))
    print("model groups per (layer, KV head, request) per joint: 63 + (U - I) = %.3f; U = %.3f; I = %.3f; I / U = %.4f"
          % (G63 + U - I, U, I, I / U))
    for B in (216, 318):
        n = L * H * B
        k = KV[B]
        fb = up * BLK * H * L * B / 1e9
        db = 64 * BLK * H * L * B / 1e9
        j, d = JOINT[B], DRAFT[B]
        gemm_j = j["gemm_pre"] + j["gemm_post"] + j["gemm_head"]
        gemm_d = d["gemm_pre"] + d["gemm_post"] + d["gemm_head"]
        p = sum(PARAMS.values()) * L + LM
        fl_j = 2 * 4 * B * p / 1e12
        fl_d = 2 * B * p / 1e12
        print("B %d: rows per joint pass 4B = %d; GEMM calls 129 at M = %d (draft: 129 at M = %d)" % (B, 4 * B, 4 * B, B))
        print("   GEMM device %.3f ms (joint) vs %.3f (draft): x%.3f for x4 rows; FLOP %.3f T vs %.3f T; at %.0f TF: %.2f / "
              "%.2f ms; achieved %.1f %% / %.1f %% of peak; weight bytes %.2f GB / %.3f ms = %.3f TB/s (draft)"
              % (gemm_j, gemm_d, gemm_j / gemm_d, fl_j, fl_d, PEAK_TF, fl_j / PEAK_TF * 1e3, fl_d / PEAK_TF * 1e3,
                 100 * fl_j / PEAK_TF * 1e3 / gemm_j, 100 * fl_d / PEAK_TF * 1e3 / gemm_d, 2 * p / 1e9, gemm_d,
                 2 * p / 1e9 / gemm_d))
        print("   fork: pages read %.3f x %d = %.0f per pass = %.3f GB; device %.3f ms -> %.3f TB/s; draft call A: 64 x %d = "
              "%d pages = %.3f GB in %.3f ms -> %.3f TB/s; fork at call A's rate: %.3f ms (excess %.3f ms)"
              % (up, n, up * n, fb, j["fork"], fb / j["fork"], n, 64 * n, db, d["attnA"], db / d["attnA"],
                 fb / (db / d["attnA"]), j["fork"] - fb / (db / d["attnA"])))
        print("   model KV groups per joint %.3f x %d = %.0f (measured fork pages / model = %.4f)"
              % (G63 + U - I, n, (G63 + U - I) * n, up / (G63 + U - I)))
        print("   H2D: NEW blocks per joint pass %.1f = %.3f per (layer, KV head, request); %.3f MB = %.3f MB per request; "
              "H2D kernels %.3f ms (resid %.3f + fetch %.3f) -> %.2f GB/s; selected verify slots %.1f = %.3f per (layer,"
              " head, request) = %.3f per verify row (3 rows); ready share %.4f"
              % (k["new"], k["new"] / n, k["mb"], k["mb"] / B, j["h2d_resid"] + j["h2d_fetch"], j["h2d_resid"],
                 j["h2d_fetch"], k["mb"] / (j["h2d_resid"] + j["h2d_fetch"]), k["sel"], k["sel"] / n, k["sel"] / n / 3,
                 k["ready"] / k["sel"]))


def items(B):
    """The joint pass's device ms by work item (labels), the draft step's, and the increment."""
    j, d = JOINT[B], DRAFT[B]
    gj = j["gemm_pre"] + j["gemm_post"] + j["gemm_head"]
    gd = d["gemm_pre"] + d["gemm_post"] + d["gemm_head"]
    small_j = (j["pre"] - j["gemm_pre"]) + (j["post"] - j["gemm_post"]) + (j["head"] - j["gemm_head"]) + j["embed"]
    small_d = (d["pre"] - d["gemm_pre"]) + (d["post"] - d["gemm_post"]) + (d["head"] - d["gemm_head"]) + d["embed"]
    att_d = d["attnA"] + d["attnB"] + d["amask"] + d["bmask"] + d["merge"]
    it = [
        ("dense GEMM (129 calls)", gj, gd),
        ("attention (fork + lists) / draft call A + B + masks + merge", j["attn"], att_d),
        ("selection (stage 1 + pooling + top-k)", j["select"], d["select"]),
        ("residency + fetch (incl. H2D)", j["resid"] + j["fetch"], d["resid"]),
        ("tentative state: union + write + chain + twin", j["union"] + j["write"] + j["chain"] + j["twin"],
         d["write"] + d["twin"]),
        ("norms / rope / act / embed / logits cast", small_j, small_d),
    ]
    return it, gj, gd


def gap_tables():
    print("== 3. GAP DECOMPOSITION (DERIVED)")
    for B in (216, 318):
        m, j, d, i3 = MODEL[B], JOINT[B], DRAFT[B], I3[B]
        it, gj, gd = items(B)
        tot = sum(x[1] for x in it)
        totd = sum(x[2] for x in it)
        print("-- B %d (C %d): joint busy %.4f (items sum %.4f); draft busy %.4f (items sum %.4f)"
              % (B, m["C"], j["busy"], tot, d["busy"], totd))
        nb = i3["joint"] - j["busy"]
        nb96 = i3["joint"] - j["busy_g96"]
        h2d = j["h2d_resid"] + j["h2d_fetch"]
        inter = m["dense"] * m["s"]
        print("   A. MAX FORM (the human's budget %.4f = dense %.4f + interference %.4f); I3 joint %.3f; gap %.4f"
              % (m["budget"], m["dense"], inter, i3["joint"], i3["joint"] - m["budget"]))
        rows = [("dense GEMM vs model dense", gj - m["dense"]),
                ("attention + lists (model: KV share %.3f hidden under max)" % (m["kvshare"] + m["extra"]), it[1][1]),
                ("selection over 4B rows (model: B rows hidden in td, 3B omitted)", it[2][1]),
                ("residency + fetch: H2D %.3f + bookkeeping %.3f, minus the interference allowance %.3f"
                 % (h2d, it[3][1] - h2d, inter), it[3][1] - inter),
                ("tentative state + union lists (not modelled)", it[4][1]),
                ("norms / rope / act / embed / cast (hidden in td)", it[5][1]),
                ("steady non-busy = I3 joint - profiled busy (g112; g96 %.3f)" % nb96, nb)]
        s = 0.0
        for nm, v in rows:
            s += v
            print("      %-95s %9.3f" % (nm, v))
        print("      sum %.3f vs gap %.3f: residual %.4f" % (s, i3["joint"] - m["budget"], i3["joint"] - m["budget"] - s))
        print("   B. SERIAL FORM chain: budget %.3f + overlap assumption %.3f = serial %.3f; measured = one-step device "
              "(draft0) %.3f + joint increment %.3f + non-busy %.3f = %.3f"
              % (m["budget"], m["ser"] - m["budget"], m["ser"], d["busy"], j["busy"] - d["busy"], nb,
                 d["busy"] + j["busy"] - d["busy"] + nb))
        print("      one step: measured draft0 busy %.3f - td %.3f = %.3f; model increment = extra + dense - GEMM(B) = %.3f"
              " + %.3f = %.3f; measured increment %.3f; increment excess %.3f"
              % (d["busy"], m["td"], d["busy"] - m["td"], m["extra"], m["dense"] - m["gb"], m["extra"] + m["dense"]
                 - m["gb"], j["busy"] - d["busy"], (j["busy"] - d["busy"]) - (m["extra"] + m["dense"] - m["gb"])))
        model_inc = {0: m["dense"] - m["gb"], 1: m["extra"]}
        s2 = 0.0
        for k, (nm, a, b) in enumerate(it):
            mi = model_inc.get(k, 0.0)
            s2 += (a - b) - mi
            print("      %-60s joint %9.3f draft %8.3f increment %9.3f model increment %8.3f excess %9.3f"
                  % (nm, a, b, a - b, mi, (a - b) - mi))
        print("      excess sum %.3f; check: serial gap %.3f = one-step %.3f + excess %.3f + non-busy %.3f -> %.3f"
              % (s2, i3["joint"] - m["ser"], d["busy"] - m["td"], s2, nb, d["busy"] - m["td"] + s2 + nb))


def rounds():
    print("== 4. ROUNDS (DERIVED)")
    for B in (216, 318):
        x, j, d = REF[B], JOINT[B], DRAFT[B]
        ph = x["reset"] + x["joint"] + x["commit"] + x["draft"]
        db = d["busy"] + d["busy1"]
        db96 = d["busy_g96"] + d["busy1_g96"]
        print("B %d %s: phases reset %.3f + joint %.3f + commit %.3f + draft %.3f = %.3f; round median %.3f; residual %.3f"
              % (B, x["run"], x["reset"], x["joint"], x["commit"], x["draft"], ph, x["round"], x["round"] - ph))
        print("   joint: busy g112 %.3f / g96 %.3f -> non-busy %.3f / %.3f; draft phase: busy g112 %.3f + %.3f = %.3f / "
              "g96 %.3f -> non-busy %.3f / %.3f; device-busy share of joint + draft (g112) %.4f"
              % (j["busy"], j["busy_g96"], x["joint"] - j["busy"], x["joint"] - j["busy_g96"], d["busy"], d["busy1"], db,
                 db96, x["draft"] - db, x["draft"] - db96, (j["busy"] + db) / (x["joint"] + x["draft"])))
        y = I3[B]
        p3 = y["reset"] + y["joint"] + y["commit"] + y["draft"]
        print("   I3 %s: %.3f + %.3f + %.3f + %.3f = %.3f; round %.3f; residual %.3f; I3 draft - REF draft = %.3f (lean draft)"
              % (y["gpu"], y["reset"], y["joint"], y["commit"], y["draft"], p3, y["round"], y["round"] - p3,
                 y["draft"] - x["draft"]))
        b = BEST[B]
        pb = b["reset"] + b["joint"] + b["commit"] + b["draft"]
        print("   BEST %s %s: %.3f + %.3f + %.3f + %.3f = %.3f; round %.3f; residual %.3f"
              % (b["arm"], b["run"], b["reset"], b["joint"], b["commit"], b["draft"], pb, b["round"], b["round"] - pb))
    # inter-round host time of I3 r4a (MODEL: probe rounds at 372 ms; wall_s :42, rates :27, prelude :43)
    wall, mean, tpr, st, rnd, pre = 79179.557, 1560.5, 570.878, 181, 194, 217.566
    ssum = st * tpr / (mean / 1e3)
    rest = wall - ssum - 12 * 372.0 - pre
    print("I3 r4a inter-round host time (MODEL): steady sum %d x %.3f / %.4f = %.1f ms; wall %.1f - steady - 12 probes x 372"
          " - prelude %.1f = %.1f ms over %d rounds = %.1f ms per round (upper bound: includes round 0 and the warm step)"
          % (st, tpr, mean / 1e3, ssum, wall, pre, rest, rnd, rest / rnd))


def best():
    print("== 5. BEST ARM vs I3 at the same boundaries (DERIVED; eager probe groups; cross-GPU noted)")
    for B in (216, 318):
        a, b = PROBE_I3[B], PROBE_BEST[B]
        m = MODEL[B]
        s = 0.0
        out = []
        for k in ("attention", "chain", "check", "dense_next", "fetch", "residency", "union", "write"):
            dd = b[k] - a[k]
            s += dd
            out.append("%s %.3f" % (k, dd))
        dj = BEST[B]["joint"] - I3[B]["joint"]
        rem_i3 = I3[B]["joint"] - a["total"]
        rem_b = BEST[B]["joint"] - b["total"]
        print("B %d: label deltas: %s; sum %.3f; probe total delta %.3f; steady - probe: I3 %.3f, best %.3f (delta %.3f);"
              " steady joint delta %.3f = probe total delta + remainder delta %.3f"
              % (B, ", ".join(out), s, b["total"] - a["total"], rem_i3, rem_b, rem_b - rem_i3, dj,
                 (b["total"] - a["total"]) + (rem_b - rem_i3)))
        print("   best gap vs budget %.3f - %.4f = %.3f; I3 gap %.3f; change %.3f; H2D gather spans (wait) I3 %.3f -> best "
              "%.3f; NEW blocks %.1f -> %.1f"
              % (BEST[B]["joint"], m["budget"], BEST[B]["joint"] - m["budget"], I3[B]["joint"] - m["budget"], dj, a["wait"],
                 BEST[B]["wait"], KV[B]["new"], BEST[B]["new_blocks"]))


def drain():
    print("== 6. L7 at C 73: the probe sample (ana_rev6_s1.txt :89 probe_joint, rounds 16..208)")
    pj = [369.5227, 374.5085, 374.5648, 398.1404, 393.4167, 391.5422, 390.4963, 397.8196, 369.3899, 248.3049, 233.3945]
    rd = [16, 32, 48, 64, 80, 128, 144, 160, 176, 192, 208]
    keep = [v for g, v in zip(rd, pj) if g < 170]
    print("median of 11 probes %.3f; without the drain rounds >= 170 (%d probes) %.3f; steady L4PTFD joint 383.231 ->"
          " probe overhead %.3f (recorded -8.666) vs %.3f" % (S.median(pj), len(keep), S.median(keep),
                                                              S.median(pj) - 383.231, S.median(keep) - 383.231))
    pj1 = [225.6343, 212.6520, 216.9283, 210.4914, 209.9599, 207.6047, 208.5376, 209.6202, 202.4411, 188.7637]
    rd1 = [16, 32, 48, 64, 80, 128, 144, 160, 176, 192]
    k1 = [v for g, v in zip(rd1, pj1) if g < 170]
    print("C 128 (c3a :46): median of 10 probes %.3f; without rounds >= 170 %.3f; steady 206.157"
          % (S.median(pj1), S.median(k1)))


if __name__ == "__main__":
    model_terms()
    work_counts()
    gap_tables()
    rounds()
    best()
    drain()
