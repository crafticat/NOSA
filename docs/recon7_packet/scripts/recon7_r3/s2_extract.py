"""RECON7 R3 part 2: extract the work / time / byte / memory / schedule fields of the ladder records (read-only).
Every number printed is a record value (MEASURED) or arithmetic on record values (DERIVED, the formula is printed)."""
import json
import statistics as st

R = "/mnt/beegfs/ojerbi/r2-runs/spec/"
RUNS = [
    "i3_r4a_2195576_i3a", "i3_r3a_2195575_i3a", "i3_r3b_2195575_i3b", "i3_r4b_2195576_i3b",
    "i4a_q3a_2195575_i4aa", "i4a_k4a_2195576_i4aa", "i4a_q4b_2195576_i4ab", "i4a_k3b_2195575_i4ab",
    "i4d_d3c_2195575_i4dc", "i4d_d4d_2195576_i4dd", "i4d_d3a_2195575_i4da", "i4d_d4b_2195576_i4db",
    "i4d_f3b_2195575_i4db",
    "i4dc_d3a_2195575_i4dca", "i4dc_d4b_2195576_i4dcb",
    "i4m_n3128_2195575_i4m128", "i4m_v4128_2195576_i4m128", "i4m_m3p73_2195575_i4mp73",
    "i4m_m3p128_2195575_i4mp128",
]
B_KEYS = ("start_GB", "prefill_peak_GB", "decode_peak_GB", "decode_peak_reserved_GB", "fit_batch_decode_MODEL",
          "fit_batch_prefill_MODEL")


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def r3(x):
    return round(x, 3) if isinstance(x, float) else x


for run in RUNS:
    try:
        with open(R + run + "/summary.json") as f:
            d = json.load(f)
    except Exception as e:                                           # a run without a record: say so
        print("=" * 20, run, "NO RECORD", repr(e))
        continue
    print("=" * 20, run)
    meta = d.get("meta") or {}
    print("META batch", meta.get("batch"), "C", meta.get("capacity_C"), "pool", meta.get("pool_blocks"),
          "dims", meta.get("dims"), "steps", meta.get("steps"), "gpu", (meta.get("gpu_uuid") or meta.get("device_uuid")))
    sh = d.get("SH") or {}
    print("SH steady_ms", g(sh, "rates", "steady_step_ms"), "tok_s", g(sh, "rates", "steady_tok_s"), "mean_ms",
          g(sh, "steady", "mean_ms"), "peak", sh.get("gpu_peak_alloc_GB"), "start", g(sh, "memory", "start_GB"))
    for arm, a in (d.get("SPEC") or {}).items():
        if not isinstance(a, dict):
            continue
        B = g(a, "meta", "B") or meta.get("batch")
        print("-- ARM", arm, "B", B, "rounds", a.get("rounds"), "runner", a.get("runner"))
        print("   rates", a.get("rates"))
        print("   phases", a.get("phases_median"), "draft_step_ms", a.get("draft_step_ms"), "draft_step_sh",
              a.get("draft_step_sh"), "joint_sh", a.get("joint_sh_steps"))
        jp = a.get("joint_phases_fork") or {}
        print("   joint_groups_median", jp.get("groups_median"), "n_probes", jp.get("n_probes"))
        print("   kv_at_verify", a.get("kv_at_verify"))
        sr = a.get("store_record")
        if sr:
            print("   store_record", {k: v for k, v in sr.items() if not isinstance(v, list)})
        acc = a.get("acceptance") or {}
        print("   acceptance", {k: r3(v) for k, v in acc.items()})
        pa = a.get("paper") or {}
        apr = pa.get("accepted_per_round") or []
        print("   paper verify_rows_hist", pa.get("verify_rows_hist"), "acc_steady_mean",
              pa.get("accepted_per_round_steady_mean"), "acc_per_req_round", r3(pa.get("accepted_per_request_round")))
        if apr and B:
            ex = set(a.get("steady_excluded_rounds") or [])
            pe = 16
            stl = [x for i, x in enumerate(apr) if i > 0 and i % pe != 0 and i not in ex]
            med = st.median(stl)
            print("   accepted_per_round: n", len(apr), "first5", apr[:5], "last12", apr[-12:],
                  "steady-like n", len(stl), "median", med, "median/B", r3(med / B), "mean/B", r3(st.fmean(stl) / B))
            lo = [i for i, x in enumerate(apr) if x < 0.9 * med]
            print("   rounds with accepted < 0.9 x median:", lo)
            print("   DERIVED sum(accepted_per_round)", sum(apr))
        ta = g(a, "schedule_record", "tokens_acct")
        print("   tokens_acct", ta)
        tr = a.get("tent") or {}
        print("   tent", {k: (sum(v) if isinstance(v, list) else v) for k, v in tr.items() if k not in ("calls",)},
              "calls", tr.get("calls"))
        print("   calls", a.get("calls"))
        print("   draft", a.get("draft"))
        mem = a.get("memory") or {}
        print("   memory", {k: mem.get(k) for k in B_KEYS}, "arena", mem.get("arena"))
        print("   peak_with_prefill", a.get("peak_with_prefill_GB"), "gpu_peak_alloc", a.get("gpu_peak_alloc_GB"),
              "reserved", a.get("gpu_peak_reserved_GB"), "graph_pool", g(a, "graph", "pool_bytes"),
              "dgraph_static", g(a, "dgraph", "static_bytes"), "store", a.get("store"))
        un = a.get("union") or {}
        print("   union calls_per_joint_pass", un.get("calls_per_joint_pass"), "counters",
              {k: v for k, v in (un.get("counters") or {}).items() if v})
        print("   wall_s", r3(a.get("wall_s")), "prelude_ms", r3(a.get("prelude_ms")), "count", a.get("count_min"),
              a.get("count_max"), "bytes_per_round", a.get("bytes_per_round"), "rollback", a.get("rollback_per_round"))
        # the drain: the round of each request's last trace entry (its final pass)
        tt = g(a, "schedule_record", "trace_tail") or []
        th = g(a, "schedule_record", "trace_head") or []
        if tt:
            fin = []
            for req in tt:
                rs = [e.get("round") for e in req if e.get("role") in ("final", "target")]
                fin.append(max(rs) if rs else None)
            ok = [x for x in fin if x is not None]
            G = a.get("rounds")
            if ok and G:
                done = sum(G - 1 - x for x in ok)
                print("   DRAIN finish-round per request: n", len(ok), "min", min(ok), "median", st.median(ok), "max",
                      max(ok), "done request-rounds sum(G - 1 - finish)", done, "of B x G", len(ok) * G,
                      "share", r3(done / (len(ok) * G)))
                hist = {}
                for x in ok:
                    hist[x] = hist.get(x, 0) + 1
                print("   DRAIN finish-round histogram", sorted(hist.items()))
            print("   TRACE_TAIL req0", json.dumps(tt[0])[:3000])
        if th:
            print("   TRACE_HEAD req0", json.dumps(th[0])[:3000])
