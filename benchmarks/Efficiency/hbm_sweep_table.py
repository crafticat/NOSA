"""CPU-only convenience summary of the weights-versus-KV batch sweep (hbm_batch_sweep.py). Stdlib + hbm_sweep_lib;
runs anywhere (python3 hbm_sweep_table.py <OUT>). Codex performs the real aggregation and plotting from the raw
per-cell files; these tables only make the job log readable. Every table says which stage its numbers come from.

Outputs under <OUT>/tables/:
  step_stats.csv        stage U: per (C, B, mode, timer, subset) n / median / p95 / mean step ms, tokens/s
  memory_breakdown.csv  stage U: steady occupied memory by category (sums to cudaMemGetInfo device-used)
  memory_peaks.csv      stage U: per window family peak allocated / reserved (never added across windows)
  analytic_hbm.csv      ANALYTIC bytes per step by category (from the measured per-step loads / pool actions)
  hbm_measured.csv      stage P: DRAM bytes per profiled step by (NVTX range, kernel family), attribution labelled
  transfer.csv          CPU-GPU traffic per step: analytic H2D from loads, measured sysmem / PCIe when collected
  effective_bw.csv      stage P bytes / stage U median gated step (blank where stage P did not run)
  gates.csv             exactness gates (repeats, resident, profiled windows) and the C=63 vs C=128 logits identity
  ladder.csv            the largest-batch search and every non-OK cell with its class
  summary.md            the above, abridged
Exit code = failed exactness gates of the driver + requested cells with neither a result nor a classification.
"""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hbm_sweep_lib as HL  # noqa: E402


def _load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _cells(stage_dir, prefix="c"):
    out = []
    if not os.path.isdir(stage_dir):
        return out
    for d in sorted(os.listdir(stage_dir)):
        full = os.path.join(stage_dir, d)
        if os.path.isdir(full) and d.startswith(prefix) and "_b" in d:
            out.append((d, full))
    return out


def _f(x, fmt="%.2f"):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return fmt % x


def step_stats_rows(cell_dir):
    res = _load_json(os.path.join(cell_dir, "result.json"), {})
    steps = HL.read_jsonl(os.path.join(cell_dir, "steps.jsonl"))
    C, B = res.get("C"), res.get("B")
    rows = []
    groups = OrderedDict()
    for r in steps:
        if not r.get("timed"):
            continue
        for subset in ("all", "first_exec", "non_burst"):
            if subset == "first_exec" and not r.get("first_exec"):
                continue
            if subset == "non_burst" and r.get("burst"):
                continue
            groups.setdefault((r["mode"], r["timer"], subset), []).append(r)
    for (mode, timer, subset), rs in groups.items():
        valid = [r for r in rs if timer != "gated" or r.get("gate_valid")]
        s = HL.step_stats([r["device_ms"] for r in valid])
        rows.append(dict(C=C, B=B, mode=mode, timer=timer, subset=subset, n=s["n"], n_gate_invalid=len(rs) - len(valid), median_ms=s["median"], p95_ms=s["p95"],
                         mean_ms=s["mean"], min_ms=s["min"], max_ms=s["max"], tokens_per_s=HL.tokens_per_s(B, s["median"]),
                         host_enqueue_median_ms=HL.step_stats([r["host_enqueue_ms"] for r in rs])["median"],
                         loads_mean=sum(r.get("loads", 0) for r in rs) / len(rs), stage="U (unprofiled)",
                         timer_meaning="gated: device time after a GPU sleep gate (host enqueue excluded)" if timer == "gated" else "eager: CUDA events around enqueue + run (host gaps included)"))
    return rows


def analytic_rows(cell_dir, weight_read_bytes=None):
    res = _load_json(os.path.join(cell_dir, "result.json"), {})
    C, B = res.get("C"), res.get("B")
    steps = [r for r in HL.read_jsonl(os.path.join(cell_dir, "steps.jsonl")) if r.get("timed")]
    out = []
    by = defaultdict(list)
    for r in steps:
        by[r["mode"]].append(r)
    for mode, rs in by.items():
        for subset in ("non_burst", "all"):
            sel = [r for r in rs if subset == "all" or not r.get("burst")]
            if not sel:
                continue
            acc = OrderedDict()
            for r in sel:
                rows = HL.analytic_step_bytes(B, C, loads=r.get("loads", 0), tail_rows=r.get("tail_rows", 0), compress_keys=r.get("compress_keys", 1007),
                                              burst=bool(r.get("burst")), pool_swaps=r.get("pool_swap", 0), pool_moves=r.get("pool_move_in", 0) + r.get("pool_move_out", 0),
                                              weight_read_bytes=weight_read_bytes)
                for a in rows:
                    k = (a["category"], a["subcategory"])
                    x = acc.setdefault(k, dict(read_bytes=0.0, write_bytes=0.0, basis=a["basis"]))
                    x["read_bytes"] += a["read_bytes"] / len(sel)
                    x["write_bytes"] += a["write_bytes"] / len(sel)
            flat = [dict(category=k[0], subcategory=k[1], **v) for k, v in acc.items()]
            tot = HL.analytic_totals(flat)
            for f in flat:
                cat = tot["categories"][f["category"]]
                out.append(dict(C=C, B=B, mode=mode, subset=subset, n_steps=len(sel), category=f["category"], subcategory=f["subcategory"],
                                read_GB=f["read_bytes"] / 1e9, write_GB=f["write_bytes"] / 1e9, category_pct_of_reads=cat["pct_of_reads"],
                                category_pct_of_writes=cat["pct_of_writes"], step_read_GB=tot["read_bytes"] / 1e9, step_write_GB=tot["write_bytes"] / 1e9,
                                analytic_weight_share_of_reads=tot["weight_share_of_reads"], basis=f["basis"], label="ANALYTIC (not a counter)"))
    return out


def measured_rows(cell_dir, tag):
    ws = _load_json(os.path.join(cell_dir, "window_summary.json"), {})
    settings = _load_json(os.path.join(cell_dir, "ncu_settings.json"), {})
    res = _load_json(os.path.join(cell_dir, "result.json"), {})
    C, B = res.get("C"), res.get("B")
    n = HL.num
    rows, totals = [], []
    for w in ws.get("windows", []):
        t = w["total"]
        rd = n(t.get("dram_read_bytes"))
        gemm_rd = sum(n(g.get("dram_read_bytes")) for g in w["by_range_family"] if g["family"] == "gemm" and n(g.get("dram_read_bytes")) == n(g.get("dram_read_bytes")))
        totals.append(dict(C=C, B=B, tag=tag, mode=w.get("mode"), step=w.get("step"), cache_control=settings.get("cache_control"), single_pass=settings.get("single_pass"),
                           n_kernels=t.get("n_kernels"), dram_read_GB=rd / 1e9, dram_write_GB=n(t.get("dram_write_bytes")) / 1e9,
                           l2_write_arrival_GB=n(t.get("l2_device_write_arrival_bytes")) / 1e9, sysmem_read_GB=n(t.get("sysmem_read_bytes")) / 1e9,
                           pcie_read_GB=n(t.get("pcie_read_bytes")) / 1e9, pcie_write_GB=n(t.get("pcie_write_bytes")) / 1e9, kernel_time_ms=n(t.get("gpu_time_s")) * 1e3,
                           gemm_family_read_share=gemm_rd / rd if rd == rd and rd else float("nan"),
                           loads=w.get("loads"), pool_swap=w.get("pool_swap"), compress_keys=w.get("compress_keys"), tail_rows=w.get("tail_rows"),
                           label="MEASURED (Nsight Compute, one profiled step; kernel-family / NVTX attribution)"))
        for g in w["by_range_family"]:
            grd = n(g.get("dram_read_bytes"))
            rows.append(dict(C=C, B=B, tag=tag, mode=w.get("mode"), cache_control=settings.get("cache_control"), range=g["range"], family=g["family"],
                             attribution_method=g["attribution_method"], n_kernels=g["n_kernels"], dram_read_GB=grd / 1e9,
                             dram_write_GB=n(g.get("dram_write_bytes")) / 1e9, pct_of_step_dram_read=100.0 * grd / rd if rd == rd and rd else float("nan"),
                             l2_write_arrival_GB=n(g.get("l2_device_write_sectors")) * 32 / 1e9, sysmem_read_GB=n(g.get("l2_sysmem_tex_read_sectors")) * 32 / 1e9,
                             kernel_time_ms=n(g.get("gpu_time_s")) * 1e3, content_hint=g.get("content_hint", "")))
    return rows, totals


def main(out_dir, L=16128):
    tdir = os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    nfail = 0
    ss, mb, mp, an, gates = [], [], [], [], []
    shas = defaultdict(dict)
    udir = os.path.join(out_dir, "U")
    for name, d in _cells(udir):
        res = _load_json(os.path.join(d, "result.json"))
        if not res:
            continue
        ss += step_stats_rows(d)
        inv = HL.read_csv(os.path.join(d, "inventory_steady.csv")) if os.path.exists(os.path.join(d, "inventory_steady.csv")) else []
        wbytes = sum(int(r["counted_bytes"]) for r in inv if r["category"] == "weights" and r["subcategory"] == "gemm_weight") or None
        an += analytic_rows(d, weight_read_bytes=wbytes)
        if os.path.exists(os.path.join(d, "breakdown_steady.csv")):
            for r in HL.read_csv(os.path.join(d, "breakdown_steady.csv")):
                mb.append(r)
        for fam, v in (res.get("window_family_peaks") or {}).items():
            mp.append(dict(C=res["C"], B=res["B"], window_family=fam, peak_allocated_GB=v / 1e9))
        pk = res.get("peaks") or {}
        if pk.get("max_peak_allocated"):
            mp.append(dict(C=res["C"], B=res["B"], window_family="MAX over windows (never a sum)", peak_allocated_GB=pk["max_peak_allocated"] / 1e9,
                           window=pk.get("max_window"), peak_reserved_GB=(pk.get("max_peak_reserved") or 0) / 1e9))
        gf = res.get("gate_failures") or []
        steps = HL.read_jsonl(os.path.join(d, "steps.jsonl"))
        checked = [r for r in steps if "logits_equal" in r]
        gates.append(dict(C=res["C"], B=res["B"], check="repeat and resident exactness (logits torch.equal the ordinary step, load counts)", n_checked=len(checked),
                          n_failed=len(gf), verdict="PASS" if not gf and res.get("status") == "OK" else ("FAIL" if gf else res.get("status"))))
        nfail += len(gf)
        for r in HL.read_jsonl(os.path.join(d, "logits_sha.jsonl")):
            shas[res["B"]][(res["C"], r["step"])] = r["sha"]
    for B, m in sorted(shas.items()):
        steps = sorted({s for (_, s) in m})
        both = [s for s in steps if (63, s) in m and (128, s) in m]
        eq = [s for s in both if m[(63, s)] == m[(128, s)]]
        if both:
            gates.append(dict(C="63 vs 128", B=B, check="pool invisibility: C=128 ordinary-step logits sha == C=63 (same documents, teacher-forced)",
                              n_checked=len(both), n_failed=len(both) - len(eq), verdict="PASS" if len(eq) == len(both) else "FINDING (not a harness failure)"))
    # stage P
    hm, ht, tr, eb = [], [], [], []
    med = {}
    for r in ss:
        if r["timer"] == "gated" and r["subset"] == "all":
            med[(r["C"], r["B"], r["mode"])] = r["median_ms"]
    for name, d in _cells(os.path.join(out_dir, "P")):
        rows, totals = measured_rows(d, name)
        hm += rows
        ht += totals
        wmeta = _load_json(os.path.join(d, "windows.json"), {})
        for w in wmeta.get("windows", []):
            if w.get("mode") == "resident":
                ok = bool(w.get("logits_equal_ordinary")) and bool(w.get("loads_ok"))
                gates.append(dict(C=wmeta.get("C"), B=wmeta.get("B"),
                                  check="profiled resident window: 0 loads and logits torch.equal the profiled ordinary step (%s)" % name, n_checked=1,
                                  n_failed=0 if ok else 1, verdict="PASS" if ok else "FAIL"))
                nfail += 0 if ok else 1
        for t in totals:
            loads = t.get("loads") or 0
            pc = HL.pcie_step_bytes(loads)
            tr.append(dict(C=t["C"], B=t["B"], mode=t["mode"], tag=name, analytic_h2d_GB=pc["h2d_bytes"] / 1e9, measured_sysmem_read_GB=t["sysmem_read_GB"],
                           measured_pcie_read_GB=t["pcie_read_GB"], measured_pcie_write_GB=t["pcie_write_GB"], basis=pc["basis"],
                           label="CPU-GPU traffic kept apart from HBM; sysmem = L2 sectors from the SM gathers to host memory x 32 B"))
            m = med.get((t["C"], t["B"], t["mode"]))
            byt = (t["dram_read_GB"] + t["dram_write_GB"]) if t["dram_read_GB"] == t["dram_read_GB"] else float("nan")
            eb.append(dict(C=t["C"], B=t["B"], mode=t["mode"], tag=name, cache_control=t["cache_control"], dram_GB_per_step=byt, dram_read_GB_per_step=t["dram_read_GB"],
                           unprofiled_median_ms=m, effective_TBps=(byt / (m * 1e-3) / 1e3) if (m and byt == byt) else None,
                           effective_read_TBps=(t["dram_read_GB"] / (m * 1e-3) / 1e3) if m else None,
                           label="stage P measured DRAM bytes of one profiled step / stage U unprofiled median gated step (two stages combined)"))
    lad = []
    for stage in ("U", "P"):
        for r in HL.read_jsonl(os.path.join(out_dir, stage, "orchestrator.jsonl")):
            lad.append(dict(stage=stage, **{k: r.get(k) for k in ("kind", "C", "B", "tag", "status", "cls", "detail", "wall_s", "planned_s")}))
    missing = [r for r in lad if r["status"] in (None, "", "FAIL") and r["kind"] in ("cell", "ncu_cell")]
    nfail += len(missing)
    HL.write_csv(os.path.join(tdir, "step_stats.csv"), ss)
    HL.write_csv(os.path.join(tdir, "memory_breakdown.csv"), mb)
    HL.write_csv(os.path.join(tdir, "memory_peaks.csv"), mp)
    HL.write_csv(os.path.join(tdir, "analytic_hbm.csv"), an)
    HL.write_csv(os.path.join(tdir, "hbm_measured.csv"), hm)
    HL.write_csv(os.path.join(tdir, "hbm_measured_totals.csv"), ht)
    HL.write_csv(os.path.join(tdir, "transfer.csv"), tr)
    HL.write_csv(os.path.join(tdir, "effective_bw.csv"), eb)
    HL.write_csv(os.path.join(tdir, "gates.csv"), gates)
    HL.write_csv(os.path.join(tdir, "ladder.csv"), lad)
    md = render(ss, mb, an, ht, eb, gates, lad)
    with open(os.path.join(tdir, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    return min(nfail, 100)


def render(ss, mb, an, ht, eb, gates, lad):
    L = ["# Weights-versus-KV batch sweep: convenience summary (Codex aggregates the raw files)", ""]
    L += ["## Stage U: unprofiled step time (gated = device time after a GPU sleep gate; subset all)", "",
          "| C | B | mode | timer | n | median ms | p95 ms | tokens/s | host enqueue ms | loads/step |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in ss if r["subset"] == "all"), key=lambda r: (r["C"], r["B"], r["mode"], r["timer"])):
        L.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s |" % (r["C"], r["B"], r["mode"], r["timer"], r["n"], _f(r["median_ms"]), _f(r["p95_ms"]),
                                                                          _f(r["tokens_per_s"], "%.0f"), _f(r["host_enqueue_median_ms"]), _f(r["loads_mean"], "%.0f")))
    L += ["", "## Stage U: steady occupied GPU memory (GB; sums to device used)", ""]
    cats = list(HL.MEM_CATEGORIES)
    L += ["| C | B | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    agg = OrderedDict()
    for r in mb:
        k = (r.get("C"), r.get("B"))
        agg.setdefault(k, defaultdict(float))[r["category"]] += float(r["gb"])
    for (c, b), v in sorted(agg.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        L.append("| %s | %s | " % (c, b) + " | ".join(_f(v.get(k, 0.0)) for k in cats) + " |")
    L += ["", "## ANALYTIC HBM per step (non-burst steps; GB read / write; not counters)", "",
          "| C | B | mode | read GB | write GB | weights % of reads |", "|---|---|---|---|---|---|"]
    seen = set()
    for r in an:
        k = (r["C"], r["B"], r["mode"], r["subset"])
        if r["subset"] != "non_burst" or k in seen:
            continue
        seen.add(k)
        L.append("| %s | %s | %s | %s | %s | %s |" % (r["C"], r["B"], r["mode"], _f(r["step_read_GB"]), _f(r["step_write_GB"]), _f(100 * r["analytic_weight_share_of_reads"], "%.1f")))
    L += ["", "## Stage P: measured DRAM per profiled step (kernel-family / NVTX attribution; GEMM share includes activations)", "",
          "| C | B | mode | cache | kernels | DRAM read GB | DRAM write GB | L2 write arrivals GB | sysmem read GB | GEMM-family % of DRAM reads |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in ht:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t["C"], t["B"], t["mode"], t["cache_control"], t["n_kernels"], _f(t["dram_read_GB"]), _f(t["dram_write_GB"]),
                                                                          _f(t["l2_write_arrival_GB"]), _f(t["sysmem_read_GB"], "%.3f"), _f(100 * t["gemm_family_read_share"], "%.1f")))
    L += ["", "## Effective HBM bandwidth (stage P bytes / stage U median)", "", "| C | B | mode | GB/step | median ms | TB/s |", "|---|---|---|---|---|---|"]
    for e in eb:
        L.append("| %s | %s | %s | %s | %s | %s |" % (e["C"], e["B"], e["mode"], _f(e["dram_GB_per_step"]), _f(e["unprofiled_median_ms"]), _f(e["effective_TBps"], "%.3f")))
    L += ["", "## Gates", "", "| C | B | check | n | failed | verdict |", "|---|---|---|---|---|---|"]
    for g in gates:
        L.append("| %s | %s | %s | %s | %s | %s |" % (g["C"], g["B"], g["check"], g["n_checked"], g["n_failed"], g["verdict"]))
    L += ["", "## Cells, largest batch, deferrals", "", "| stage | kind | C | B | status | class | detail |", "|---|---|---|---|---|---|---|"]
    for r in lad:
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["stage"], r["kind"], r["C"], r["B"], r["status"], r["cls"], (r["detail"] or "")[:200]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HBM_OUT", "hbm_sweep_out")))
