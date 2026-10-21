"""R2 shared ready-KV reads, cell S3 (docs/R2_SHAREDKV2_KEYS_S3.md; docs/R2_SHAREDKV_PLAN.md section 5): the HBM
(DRAM) bytes and the L2 hit rate of piece 3 v2's attention (per-row reads: _bi_split_kernel + _bi_combine_kernel) vs
the shared kernel (_bi_shared_kernel + the same combine) on the SAME inputs, inside the eager tick, measured by
Nsight Compute. SOURCE ONLY, NOT RUN. Holder step: benchmarks/Efficiency/r2_sharedkv_ncu_step.sh.

THE CELL (--mode cell, run under ncu by orchestrate): the model, B documents of pg19-test-c16k at S tokens,
build_ragged, then ONE engine:
  --backend resident  TickEngine.from_r2 (the ResidentStore, every block resident; B 32 fits, plan section 5)
  --backend b2 | sc   the SG5 path: r2_tick_sg5.make_engine (OwnResidency / ResidencySC at C / X / R_ver, the gather
                      transport: the attention bytes do not depend on the transport) with the Residency plan's page
                      keys ([sharedkv2]: plan.page_key)
whose ops.attn is SharedKVProbe(piece 3, mode, ncu_window=(skip, count)): every keyed call runs BOTH kernels on the
same inputs (piece 3's output drives the tick); among the calls with 4 rows of one request (rows_max 4), calls
skip .. skip + count - 1 are wrapped in one profiler window each (cudaProfilerStart / Stop) with NVTX ranges
skv|pr|c<j>, skv|bd|c<j> (the union build, part of the shared side) and skv|sh|c<j>. ncu runs with
--profile-from-start off, so ONLY those launches are profiled (the counting launch stays outside). A free-running round of --steps committed tokens per request
(draft_drop 0 on the ResidentStore: full draft selections, as the 4.5 harness and the forecast).
cell.json: the probe's report (diff_rows: the bitwise A/B of every keyed call) and ncu_report (per marked call: rows,
sum of num_pages, union pages).

ORCHESTRATE: ncu --query-metrics (the metric names are checked before any profiling; a missing L2 hit-rate metric is
dropped and reported, a missing DRAM / time metric stops), then per --variants cache-control (all: every kernel
starts with a flushed L2 = the kernel's OWN reuse only, the primary; none: the tick's L2 state, REPORT) one ncu child
(the cell), the raw CSV export, and the summary.

SUMMARY (summary.json; the last stdout line R2_SKV_NCU_DONE failed=<n>; exit = n):
MUST  O_ncu_<cc>_ran, O_metrics, O_cell_bitwise (diff_rows 0 over all keyed calls, and > 0 rows compared),
      O_windows_<cc> (every marked call has exactly piece 3's two kernels in skv|pr and the shared kernel + combine in
      skv|sh; no other profiled kernel), O_bounds_all (the physics of a cold-L2 read, per kernel kind summed over the
      window: union bytes x 0.9 <= DRAM read <= sum bytes x 1.15; else the attribution or the metric is wrong and no
      ratio below means anything).
REPORT (the registered bands S3_PRED, IN / OUT): x_pr = piece 3 DRAM / logical sum bytes; x_sh = shared DRAM / sum;
      ratio = shared / piece 3 DRAM (main kernel, and with the combine); sh_over_union; f = (piece 3 DRAM - union) /
      (sum - union): the part of the logical excess (the paper's -29 % row 6 gap) that piece 3 really reads from HBM;
      the L2 read hit rates; device kernel time under ncu (gpu__time_duration; serialized, cold L2: a REPORT, not the
      tick's time); the decision D1 / D2 / D3 against the plan's break-even 0.466 (P4).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAGE_BYTES = 64 * 128 * 2 * 2           # one (row, head) page: 64 token rows of K and V, D 128, bf16
BREAK_EVEN = 0.466                      # docs/R2_SHAREDKV_PLAN.md P4: tiles / (4 x 64) <= (63 + 56.4) / 256
REQUIRED = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
            "lts__t_sectors_srcunit_tex_op_read.sum")
OPTIONAL = ("lts__t_sector_op_read_hit_rate.pct", "lts__t_sector_hit_rate.pct")
TAG_RE = re.compile(r"skv\|(pr|sh|bd)\|c(\d+)")                      # bd: the union build (torch ops), shared side
KERNELS = dict(pr=("_bi_split_kernel", "_bi_combine_kernel"), sh=("_bi_shared_kernel", "_bi_combine_kernel"))
# REGISTERED before any run (docs/R2_SHAREDKV2_KEYS_S3.md section 3): (central, lo, hi) per mode, cache-control all
S3_PRED = {
    "x_pr": dict(split=(0.43, 0.30, 0.65), whole=(0.43, 0.30, 0.65)),
    "ratio": dict(split=(0.75, 0.55, 0.95), whole=(0.70, 0.50, 0.95)),
    "sh_over_union": dict(split=(1.15, 0.95, 1.60), whole=(1.05, 0.95, 1.30)),
    "f": dict(split=(0.20, 0.05, 0.50), whole=(0.20, 0.05, 0.50)),
    "hit_pr": dict(split=(55.0, 30.0, 80.0), whole=(55.0, 30.0, 80.0)),
    "hit_sh": dict(split=(15.0, 0.0, 40.0), whole=(10.0, 0.0, 35.0)),
    # [review MAJOR-2 / MAJOR-3] the PRIMARY time ratio: device kernel time (gpu__time_duration) of (union build +
    # shared + combine) / (piece 3 split + combine), on the cache-control none variant (the tick's L2 state; all =
    # cold, REPORT); and the kernel-only ratio (the plan's P3, unchanged)
    "time_with_build": dict(split=(1.30, 0.80, 2.20), whole=(2.10, 1.00, 4.00)),
    # [sharedkv3] --union fused (docs/R2_SHAREDKV3_UNION.md): the build is one kernel per call (+ the groups once per
    # tick); and the build alone per marked call, in microseconds of device time
    # [sharedkv4 MINOR-4] the split band's top IS F5' (1.10); no whole fused arm exists: no whole fused band
    "time_with_fused_build": dict(split=(0.90, 0.60, 1.10), whole=None),
    # [sharedkv4 MINOR-5] per marked call whose build range holds ONLY the union kernel (the groups cached: not the
    # tick's first keyed layer)
    "build_us_fused": dict(split=(15.0, 5.0, 60.0), whole=None),
    "time_kernel": dict(split=(0.80, 0.55, 1.10), whole=(1.60, 0.80, 3.50)),
}


def log(msg):
    print("[r2_sharedkv_ncu %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("orchestrate", "cell", "summarize"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", default="")
    ap.add_argument("--biattn", default="", help="piece 3's bi_paged_attention.py (the pinned copy)")
    ap.add_argument("--sharedkv", default="", help="bi_paged_attention_shared.py")
    ap.add_argument("--backend", default="resident", choices=("resident", "b2", "sc"))
    ap.add_argument("--attn-mode", default="split", choices=("split", "whole"))
    ap.add_argument("--union", default="torch", choices=("torch", "fused"), help="[sharedkv3] the union build")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--steps", type=int, default=16, help="committed tokens per request")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0, help="0 = (1 + K) B")
    ap.add_argument("--C", type=int, default=96)
    ap.add_argument("--X", type=int, default=32)
    ap.add_argument("--R-ver", type=int, default=16)
    ap.add_argument("--window", default="32,64", help="skip,count over the keyed calls with 4 rows of a request")
    ap.add_argument("--ncu", default="/opt/nvidia/nsight-compute/2025.1.0/ncu")
    ap.add_argument("--variants", default="all,none", help="ncu --cache-control values; the first is the primary")
    ap.add_argument("--cc", default="all", help=argparse.SUPPRESS)
    ap.add_argument("--child-timeout", type=int, default=2400)
    a = ap.parse_args(argv)
    a.window = tuple(int(x) for x in a.window.split(","))
    a.variants = [v for v in a.variants.split(",") if v]
    if len(a.window) != 2 or a.window[0] < 1 or a.window[1] < 1 or not a.variants \
            or any(v not in ("all", "none") for v in a.variants):
        ap.error("--window skip,count (skip >= 1: the first calls compile), --variants in all,none")
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    return a


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


# ------------------------------------------------------------------ the cell (under ncu)
def mode_cell(a):
    import torch
    import r2_g0_gate as G0
    import r2_g0bi_gate as GB
    import nosi.cache_engine as CE
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    from nosi.r2_bi_decode import BIOps
    from nosi.r2_decode_path import build_ragged
    from nosi.r2_tick import TickEngine, run_round
    if not NL.R2_DECODE or NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
        raise SystemExit("REFUSED: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS 0 are needed")
    os.makedirs(a.out)
    attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
    sk = GB.load_by_path(a.sharedkv, "sharedkv_bi_paged_attention")
    probe = sk.SharedKVProbe(attn, mode=a.attn_mode, timing=True, count=True, ncu_window=a.window, union=a.union)
    ops = BIOps(probe, a.m_fix, gemm="exact")
    docs, sha = G0.load_workload(a.workload, a.batch, a.S)
    model = Llama(model_name=a.model, device="cuda", offload=True)
    meta = dict(backend=a.backend, batch=a.batch, S=a.S, steps=a.steps, mode=a.attn_mode, window=a.window,
                cc=a.cc, workload_sha256=sha, nosi_commit=os.environ.get("TICK_COMMIT", "unknown"),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    tp = None
    with torch.inference_mode():
        prompts = [docs[b, :a.S] for b in range(a.batch)]
        cache, st, first_t = build_ragged(model, prompts, a.steps + a.LAMBDA + 64, f4_fix=True)
        for lay in cache.layers:                                       # the tick's own store replaces the windows
            e = lay.cache_engine
            e._k_gpu = e._v_gpu = e._kv_bias_gpu = None
        GB.free()
        first = [int(x) for x in first_t.view(-1).tolist()]
        if a.backend == "resident":
            eng = TickEngine.from_r2(model, cache, st, ops)
        else:
            import r2_tick_sg5 as SG5
            from types import SimpleNamespace
            sa = SimpleNamespace(twin="torch", rebuild="per_call", transport="gather", threads=8, K=a.K,
                                 b2_checked=False)
            eng, svc, tp = SG5.make_engine(sa, model, cache, st, ops, a.backend, a.C, a.X, a.R_ver, None,
                                           transport="gather")
            meta["want_keys"] = bool(getattr(svc, "want_keys", False))
        t0 = time.time()
        try:
            rnd = run_round(eng, first, [a.steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA)
            torch.cuda.synchronize()
        finally:
            if tp is not None:
                tp.close()
        meta.update(wall_s=round(time.time() - t0, 1), ticks=rnd.ticks, stats=dict(eng.stats))
    rep = probe.report()
    rep.pop("per_row_ms", None)                                        # per-call lists: kept short in cell.json
    rep.pop("shared_ms", None)
    rep.pop("build_ms", None)
    write_json(os.path.join(a.out, "cell.json"), dict(meta=meta, probe=rep, ncu=probe.ncu_report()))
    log("cell done: %d keyed calls, %d marked, diff_rows %d" % (rep["keyed_calls"], len(probe.marked),
                                                                rep["diff_rows"]))
    return 0


# ------------------------------------------------------------------ the CSV and the summary (host only)
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def ncu_rows(text, fields):
    """The kernel rows of an ncu raw CSV (--csv --page raw --print-units base; the r2-runs/jhbm parser's rule): the
    header row naming 'Kernel Name', rows with a numeric ID; tag = the first skv|kind|c<j> in ANY cell (the NVTX
    push/pop column)."""
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hi is None:
        raise ValueError("no header row with 'Kernel Name'")
    head = rows[hi]
    out = []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        if not d.get("ID", "").strip().isdigit():
            continue
        tag = None
        for cell in r:
            m = TAG_RE.search(cell)
            if m:
                tag = (m.group(1), int(m.group(2)))
                break
        row = dict(id=int(d["ID"]), name=d["Kernel Name"], tag=tag)
        for f in fields:
            row[f] = _num(d.get(f, ""))
        out.append(row)
    return out


def band(name, mode, value):
    if S3_PRED[name].get(mode) is None:                                 # [sharedkv4] no arm: not registered
        return "%s %s (no band registered for %s: no arm)" % (name, value, mode)
    c, lo, hi = S3_PRED[name][mode]
    if value is None:
        return "%s n/a (band [%s, %s], central %s)" % (name, lo, hi, c)
    return "%s %.4f %s [%s, %s] (central %s)" % (name, value, "IN" if lo <= value <= hi else "OUT", lo, hi, c)


def summarize_variant(cc, rows, cell, mode, fields):
    """Per variant: the window check, the sums per kind and kernel, the ratios. An optional metric counts only when
    the export holds a value for it (a dropped metric is not read as 0)."""
    fields = [f for f in fields if f in REQUIRED or any(x.get(f) is not None for x in rows)]
    marked = {m["call"]: m for m in cell["ncu"]["marked"]}
    by = {}
    untagged, wrong = 0, []
    for r in rows:
        if r["tag"] is None:
            untagged += 1
            continue
        kind, j = r["tag"]
        by.setdefault((kind, j), []).append(r)
    def known(name):                                                    # the Triton kernel by its function name
        hits = [k for k in ("_bi_split_kernel", "_bi_shared_kernel", "_bi_combine_kernel") if k in name]
        return hits[0] if len(hits) == 1 else "other:" + name[:60]
    for j in marked:
        for kind in ("pr", "sh"):
            names = sorted(known(x["name"]) for x in by.get((kind, j), []))
            want = sorted(KERNELS[kind])
            if names != want:
                wrong.append((j, kind, names))
        if not by.get(("bd", j)):                                       # the build ran inside the window
            wrong.append((j, "bd", []))
    extra = sorted({j for (_, j) in by} - set(marked))
    ok = not wrong and not untagged and not extra and bool(marked)
    win = gate("O_windows_%s" % cc, ok, True, "%d marked calls; wrong %s; untagged %d; unmarked tags %s"
               % (len(marked), wrong[:4], untagged, extra[:4]))

    def tot(kind, field, main_only):
        s = 0.0
        for (k, j), rs in by.items():
            if k != kind or j not in marked:
                continue
            for x in rs:
                if main_only and "_bi_combine_kernel" in x["name"]:
                    continue
                s += x.get(field) or 0.0
        return s
    SUM = sum(m["sum_pages"] for m in marked.values()) * PAGE_BYTES
    UNI = sum(m["union_pages"] for m in marked.values()) * PAGE_BYTES
    d = {}
    for kind in ("pr", "sh"):
        d[kind] = dict(dram_read_main=tot(kind, "dram__bytes_read.sum", True),
                       dram_read_all=tot(kind, "dram__bytes_read.sum", False),
                       dram_write_all=tot(kind, "dram__bytes_write.sum", False),
                       l2_read_sectors_main=tot(kind, "lts__t_sectors_srcunit_tex_op_read.sum", True),
                       time_ns_main=tot(kind, "gpu__time_duration.sum", True),
                       time_ns_all=tot(kind, "gpu__time_duration.sum", False))
        for f in OPTIONAL:
            if f in fields:                                              # a percentage: weighted by the L2 reads
                num = 0.0
                for (k, j), rs in by.items():
                    if k == kind and j in marked:
                        for x in rs:
                            if "_bi_combine_kernel" not in x["name"] and x.get(f) is not None:
                                num += x[f] * (x.get("lts__t_sectors_srcunit_tex_op_read.sum") or 0.0)
                den = d[kind]["l2_read_sectors_main"]
                d[kind][f] = num / den if den else None
        sec = d[kind]["l2_read_sectors_main"]
        d[kind]["hit_derived_pct"] = 100.0 * (1.0 - d[kind]["dram_read_main"] / (32.0 * sec)) if sec else None
    d["bd"] = dict(dram_read_all=tot("bd", "dram__bytes_read.sum", False),
                   time_ns_all=tot("bd", "gpu__time_duration.sum", False),
                   kernels=sum(len(rs) for (k, j), rs in by.items() if k == "bd" and j in marked))
    pr, sh = d["pr"]["dram_read_main"], d["sh"]["dram_read_main"]
    q = dict(sum_bytes=SUM, union_bytes=UNI, union_over_sum=UNI / SUM if SUM else None,
             x_pr=pr / SUM if SUM else None, x_sh=sh / SUM if SUM else None,
             ratio=sh / pr if pr else None,
             ratio_with_combine=d["sh"]["dram_read_all"] / d["pr"]["dram_read_all"] if d["pr"]["dram_read_all"] else None,
             sh_over_union=sh / UNI if UNI else None,
             f=(pr - UNI) / (SUM - UNI) if SUM > UNI else None,
             time_ratio_main=d["sh"]["time_ns_main"] / d["pr"]["time_ns_main"] if d["pr"]["time_ns_main"] else None,
             time_ratio_all=d["sh"]["time_ns_all"] / d["pr"]["time_ns_all"] if d["pr"]["time_ns_all"] else None,
             time_with_build=(d["sh"]["time_ns_all"] + d["bd"]["time_ns_all"]) / d["pr"]["time_ns_all"]
             if d["pr"]["time_ns_all"] else None)
    q["time_kernel"] = q["time_ratio_all"]
    q["build_us_per_call"] = d["bd"]["time_ns_all"] / 1000.0 / len(marked) if marked else None   # [sharedkv3]
    # [sharedkv4 MINOR-5] the build range per marked call: its _union_kernel count and its other kernels (the group
    # ops on a tick's first keyed layer, or a torch build); the fused build band uses the calls with ONLY the kernel
    per_call = {}
    for (k, j), rs in by.items():
        if k == "bd" and j in marked:
            nk = sum(1 for x in rs if "_union_kernel" in x["name"])
            t = sum(x.get("gpu__time_duration.sum") or 0.0 for x in rs)
            per_call[j] = (nk, len(rs) - nk, t)
    d["bd"]["union_kernels_per_call"] = sorted({v[0] for v in per_call.values()})
    only = [v[2] for v in per_call.values() if v[0] == 1 and v[1] == 0]
    d["bd"]["calls_only_union_kernel"] = len(only)
    q["build_us_union_only"] = sum(only) / 1000.0 / len(only) if only else None
    q["bd_calls"] = {str(j): dict(union_kernels=v[0], other=v[1], ns=v[2]) for j, v in sorted(per_call.items())}
    hk = OPTIONAL[0] if OPTIONAL[0] in fields else None
    q["hit_pr"] = d["pr"][hk] if hk else d["pr"]["hit_derived_pct"]
    q["hit_sh"] = d["sh"][hk] if hk else d["sh"]["hit_derived_pct"]
    q["hit_source"] = hk or "derived: 1 - DRAM read / (32 B x L2 read sectors from L1)"
    return win, d, q


def mode_summarize(a, og=None):
    gates = list(og or [])
    out = dict(meta=dict(args={k: v for k, v in vars(a).items()}), variants={})
    cp = os.path.join(a.out, "P_%s" % a.variants[0], "cell.json")
    fields = list(REQUIRED) + [f for f in OPTIONAL if f in getattr(a, "metrics_ok", OPTIONAL)]
    cell = json.load(open(cp)) if os.path.exists(cp) else None
    if cell is None:
        gates.append(gate("O_cell_json", False, True, "no %s" % cp))
    else:
        p = cell["probe"]
        gates.append(gate("O_cell_bitwise", p["diff_rows"] == 0 and p["rows_compared"] > 0
                          and p["stats"].get("shared", 0) > 0, True,
                          "diff_rows %d over %d rows, stats %s" % (p["diff_rows"], p["rows_compared"], p["stats"])))
        if getattr(a, "union", "torch") == "fused":                    # [sharedkv4, review MAJOR-1]
            ud = p.get("union_diff", {})
            gates.append(gate("O_cell_fused_union_equals_torch", p.get("union_checked", 0) > 0
                              and all(v == 0 for v in ud.values()), True,
                              "%d keyed calls compared with build_union; differing elements per field %s"
                              % (p.get("union_checked", 0), ud)))
        gates.append(gate("O_cell_marked", len(cell["ncu"]["marked"]) == a.window[1], True,
                          "%d marked of %d requested (calls with 4 rows seen: %d)"
                          % (len(cell["ncu"]["marked"]), a.window[1], cell["ncu"]["full_calls"])))
        out["cell"] = dict(meta=cell["meta"], tile_ratio=p.get("tile_ratio"), row_ratio=p.get("row_ratio"),
                           tiles=p.get("tiles"))
    mode = a.attn_mode
    for cc in a.variants:
        csvp = os.path.join(a.out, "%s_raw.csv" % cc)
        cpv = os.path.join(a.out, "P_%s" % cc, "cell.json")
        if not (os.path.exists(csvp) and os.path.exists(cpv)):
            gates.append(gate("O_windows_%s" % cc, False, cc == a.variants[0], "no %s or %s" % (csvp, cpv)))
            continue
        cv = json.load(open(cpv))
        rows = ncu_rows(open(csvp, errors="replace").read(), fields)
        win, d, q = summarize_variant(cc, rows, cv, mode, fields)
        primary = cc == a.variants[0]
        win["must"] = primary
        gates.append(win)
        out["variants"][cc] = dict(kinds=d, quantities=q)
        if cc == "all" and win["passed"]:
            lo_ok = all(d[k]["dram_read_main"] >= 0.9 * q["union_bytes"] for k in ("pr", "sh"))
            hi_ok = all(d[k]["dram_read_main"] <= 1.15 * q["sum_bytes"] for k in ("pr", "sh"))
            gates.append(gate("O_bounds_all", lo_ok and hi_ok, True,
                              "cold L2: 0.9 x union %.3g <= DRAM read (pr %.3g, sh %.3g) <= 1.15 x sum %.3g"
                              % (q["union_bytes"], d["pr"]["dram_read_main"], d["sh"]["dram_read_main"],
                                 q["sum_bytes"])))
        if primary and win["passed"]:
            txt = [band(n, mode, q[n]) for n in ("x_pr", "ratio", "sh_over_union", "f", "hit_pr", "hit_sh")]
            gates.append(gate("R_S3_bands_%s" % cc, True, False, "; ".join(txt)))
            x_pr, x_sh = q["x_pr"], q["x_sh"]
            if x_pr is not None and x_sh is not None:
                dec = ("D1: piece 3 already reads <= the break-even bytes from HBM (x_pr %.3f <= %.3f): the row-6 gap "
                       "is not an HBM-byte gap" % (x_pr, BREAK_EVEN) if x_pr <= BREAK_EVEN else
                       "D2: the gap is real in HBM bytes (x_pr %.3f > %.3f) and the shared kernel closes it (x_sh "
                       "%.3f)" % (x_pr, BREAK_EVEN, x_sh) if x_sh <= BREAK_EVEN else
                       "D3: real and NOT closed by the shared kernel (x_pr %.3f, x_sh %.3f > %.3f)"
                       % (x_pr, x_sh, BREAK_EVEN))
                gates.append(gate("R_S3_decision", True, False, dec + "; f (the paid part of the logical excess) "
                                  "%s" % (None if q["f"] is None else round(q["f"], 4))))
        if win["passed"] and getattr(a, "union", "torch") == "fused":  # [sharedkv4 MINOR-5] the build that ran
            nks = d["bd"]["union_kernels_per_call"]
            gates.append(gate("O_build_kernel_%s" % cc, nks == [1], primary,
                              "fused arm: _union_kernel launches per marked call %s (must be exactly 1 in every "
                              "call; 0 = a silent fallback to the torch build); calls with only the kernel %d"
                              % (nks, d["bd"]["calls_only_union_kernel"])))
        if win["passed"]:                                               # [review MAJOR-2 / MAJOR-3] device time
            gates.append(gate("R_S3_device_time_%s" % cc, True, False,
                              "%s device kernel time (ncu gpu__time_duration, serialized, %s L2): %s; %s; main "
                              "kernels only %s; build %d kernels %.0f ns"
                              % ("PRIMARY" if cc == "none" else "REPORT", "cold" if cc == "all" else "tick",
                                 (band("time_with_fused_build", mode, q["time_with_build"]) + "; " +
                                  band("build_us_fused", mode, q["build_us_union_only"])
                                  if getattr(a, "union", "torch") == "fused" else
                                  band("time_with_build", mode, q["time_with_build"])),
                                 band("time_kernel", mode, q["time_kernel"]), q["time_ratio_main"],
                                 d["bd"]["kernels"], d["bd"]["time_ns_all"])))
    if "all" in out["variants"] and "none" in out["variants"]:
        ra, rn = out["variants"]["all"]["quantities"]["x_pr"], out["variants"]["none"]["quantities"]["x_pr"]
        gates.append(gate("R_S3_cache_state", True, False, "x_pr all %s none %s (F-S3-4: > 20 %% apart = the cold "
                          "measurement does not represent the tick)" % (ra, rn)))
    out["gates"] = gates
    write_json(os.path.join(a.out, "summary.json"), out)
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                str(g["detail"])[:400]), flush=True)
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("R2_SKV_NCU_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 125)


# ------------------------------------------------------------------ orchestrate (not profiled itself)
def child_argv(a, out, cc):
    return [sys.executable, os.path.abspath(__file__), "--mode", "cell", "--out", out, "--model", a.model,
            "--workload", a.workload, "--biattn", a.biattn, "--sharedkv", a.sharedkv, "--backend", a.backend,
            "--attn-mode", a.attn_mode, "--union", a.union, "--batch", str(a.batch), "--S", str(a.S), "--steps", str(a.steps),
            "--K", str(a.K), "--LAMBDA", str(a.LAMBDA), "--m-fix", str(a.m_fix), "--C", str(a.C), "--X", str(a.X),
            "--R-ver", str(a.R_ver), "--window", "%d,%d" % a.window, "--cc", cc]


def run_logged(argv, path, timeout):
    """argv with stdout + stderr to path; its own process group, killed at the timeout. Returns the exit or
    'timeout'."""
    with open(path, "w") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            return "timeout"


def metric_check(a):
    """ncu --query-metrics --chip ga100 (saved): every REQUIRED base name present, else a named stop; OPTIONAL ones
    kept only when present. Returns (gate, kept optional metrics)."""
    path = os.path.join(a.out, "metrics_query.txt")
    rc = run_logged([a.ncu, "--query-metrics", "--chip", "ga100"], path, 300)
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    names = set(re.findall(r"^\s*([a-z0-9_]+__[a-z0-9_]+)", text, flags=re.M))
    base = lambda m: m.rsplit(".", 1)[0]                                # noqa: E731
    missing = [m for m in REQUIRED if base(m) not in names]
    kept = [m for m in OPTIONAL if base(m) in names]
    g = gate("O_metrics", rc == 0 and not missing, True, "query exit %s; %d names; missing required %s; optional kept "
             "%s, dropped %s" % (rc, len(names), missing, kept, [m for m in OPTIONAL if m not in kept]))
    return g, kept


def mode_orchestrate(a):
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not os.access(a.ncu, os.X_OK):
        print("REFUSED: no ncu at %s" % a.ncu, file=sys.stderr)
        return 2
    os.makedirs(a.out)
    write_json(os.path.join(a.out, "config.json"), dict(vars(a), S3_PRED=S3_PRED, env={
        k: v for k, v in os.environ.items() if k.startswith(("NOSI_", "TICK_", "PYTORCH_"))}))
    og = []
    mg, kept = metric_check(a)
    og.append(mg)
    a.metrics_ok = kept
    if not mg["passed"]:
        return mode_summarize(a, og)
    metrics = ",".join(list(REQUIRED) + kept)
    for cc in a.variants:
        rep = os.path.join(a.out, "rep_" + cc)
        ncu = [a.ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", cc,
               "--clock-control", "none", "--nvtx", "--metrics", metrics, "--target-processes", "application-only",
               "-f", "-o", rep]
        t1 = time.time()
        rc = run_logged(ncu + child_argv(a, os.path.join(a.out, "P_" + cc), cc), os.path.join(a.out, "P_%s.log" % cc),
                        a.child_timeout)
        log("P_%s exit %s after %.0f s" % (cc, rc, time.time() - t1))
        exp = None
        if os.path.exists(rep + ".ncu-rep"):
            exp = run_logged([a.ncu, "--import", rep + ".ncu-rep", "--csv", "--page", "raw", "--print-units", "base"],
                             os.path.join(a.out, "%s_raw.csv" % cc), 1800)
        og.append(gate("O_ncu_%s_ran" % cc, rc == 0 and exp == 0, cc == a.variants[0],
                       "child exit %s, export exit %s" % (rc, exp)))
    return mode_summarize(a, og)


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    a.out = os.path.abspath(a.out)
    try:
        if a.mode == "cell":
            return mode_cell(a)
        if a.mode == "orchestrate":
            if not (a.model and a.workload and a.biattn and a.sharedkv):
                print("REFUSED: --model, --workload, --biattn and --sharedkv are required", file=sys.stderr)
                return 2
            return mode_orchestrate(a)
        return mode_summarize(a)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
