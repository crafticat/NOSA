"""Rung J1 cell j1f (AREA 1; KS/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md section 4; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE FORK'S RATE, MATCHED. The ledger's 0.885 vs 1.53 TB/s compares the fork (4 rows,
union pages, per-row masks, no split) with the draft's flash call A (1 row, its own window, split-KV): UNMATCHED
(LEDGER_rev7.md 2.7). This cell times the shipped fork and the shipped one-row flash call on the SAME synthetic pages
of the MEASURED shapes, in matched pairs:
  FORK     the shipped fork (fia6's blob, verify_rows' FORK_LAUNCH: 4 warps, 3 stages, tile 64, the bf16 CIS bias), 4 rows
           per request, each row 64 pages shifted (0, 3, 5, 8): union 72 per (KV head, request) (MEASURED: 72.2 at C 128,
           r7pa; 72.24 at C 73, V1B); the union built in the kernel (production)
  FORKT    the same with the union tables built before (BUILD_UNION off): the in-kernel build's cost
  FORKA1   FORK through fork_vllm_unified_attention_ahead.py with ahead=1 (the table page id and the extents word read
           two tiles ahead, as scalars); FORKA2: ahead=2 (and the bias one tile ahead); FORKA3: ahead=3 (the fork's
           K / V page-id loads kept in the source, only ext_rows and the bias ahead; the CPU compile witness shows
           that every level, 3 included, loses Triton's async K / V copies). MUST gate <shape>-FORKAn-EQ:
           the output = FORK's, bit for bit (no arithmetic change is allowed)
  FORK3Dn  the same with the fork's EXISTING 3D split path (seq_threshold_3D, n segments per (request, KV head),
           reduce_segments), n in --segments: wave balance (never run before: its numeric gate is REPORT)
  FORK1R   the fork with ONE row per request (row 0's 64 pages; BLOCK_Q 1): the kernel at the baseline's work
  NS1R1    the shipped one-row flash call (flash_attn_nosa_with_kvcache, split-KV 4: the draft's call A) on row 0's 64
           pages (a contiguous window per request): EQUAL rows and bytes as FORK1R
  NS1x4    the same call on all 4 rows, each over its own 64-page window: what the baseline pays for the 4 rows
  FLASHU1  the same call on ONE row per request over the 72 UNION pages: the flash rate on the fork's own bytes
  NSRU     the same call on the 4 rows over ONE shared union window per request (cache_batch_idx, per-entry masking
           bias: verify_rows route 1): the shipped kernel's own read sharing (through L2)
OUTPUT WRITE (j1f source follow-up, 2026-11-02): the FORK variants write their preallocated output directly. The four
flash controls above (NS1R1, NS1x4, FLASHU1, NSRU) now do the same: they call the library's own binding
(flash_attn_2_cuda_nosa.fwd_kvcache, the op that flash_attn_with_kvcache calls; the wrapper passes None at its out_
argument) with out_ = a view of the preallocated output in the kernel's swapped (B, HQ/HKV, HKV, HD) layout
(swapped_out_view). Each control has two more variants:
  <X>_CP   the j1f form: the wrapper returns a new tensor and the timed region copy_s it into the output (the old
           numbers of X were this variant)
  <X>_CO   the copy_ alone (the same bytes, from <X>_CP's output into a separate buffer)
So direct = X (an END-TO-END library call into a preallocated output: NOT kernel-only, the pinned C++ still allocates
softmax_lse and the split-KV scratch), copy-only = X_CO, total (j1f form) = X_CP. residual = X_CP - X - X_CO: a
difference of separately timed forms, not an isolated measurement of the wrapper's allocations.
MATCHED PAIRS (the summary's ratios): FORK / FLASHU1 (equal logical bytes: the cost of 4 rows' tile at the same bytes);
FORK1R / NS1R1 (equal rows and logical bytes: the time ratio of the fork and the flash call at matched work); FORK /
NS1x4 (equal rows: the saving of the shared read); FORKT / FORK (the union build); FORK3Dn / FORK (the split); NSRU /
FORK. The same pairs against the <X>_CP variants reproduce the j1f form. j1f's ratios need this rerun to be re-scored.
LOGICAL BYTES per call: pages x BLOCK_BYTES (K + V + bias of one (KV head, block)) x KV heads x requests (or entries);
achieved logical TB/s = bytes / time. Logical, not DRAM counters: these ratios are time ratios at matched logical work;
they do not give a physical per-byte ceiling or optimality (that needs DRAM counters, e.g. an ncu pass of the lead).
GATES (MUST unless noted): <shape>-RUN; <shape>-UNION (FORK union_len 72, FORK1R 64 everywhere); <shape>-<v>-EXACT
(every variant's output after its last timed pass = its first output, bit for bit, AND both hold only finite values:
identical NaN bits fail); <shape>-FORK3Dn-PATH (REPORT: the 3D launch wrote its NaN-filled segment maxima; if not, the
variant is INVALID); <shape>-FORKT-EQ (FORKT = FORK bit for bit); <shape>-<X>-OUTEQ (the end-to-end direct output
form = the j1f form X_CP, bit for bit); <shape>-<v>-NUM (every checked value of the output, of the float64 reference and of their
difference is finite (torch.isfinite, before any accumulation; a FAIL names the count and the first index), and max
|output - float64 reference| <= TOL on --sample requests, every row and head; FORK3Dn: REPORT); <shape>-BUDGET;
RUN-WALL. A variant whose NUM, EXACT, OUTEQ or FORKAn-EQ fails is reported INVALID in the ratios, the 3D saving and
the flash timing split (the decision lines say INVALID; the 3D saving line is suppressed).
NO NEW KERNEL: the fork (fia6's blob), flash_attn_nosa's op, torch ops. Synthetic inputs (seeded, x 0.05 bf16).
Order-rotated passes (lesson L62) after a warm-up of each variant; one pass = --layers calls (one per layer).
GRANT: GRANT_j1f below (default deny; r2_j1_step.sh reads it at <sha>).
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside a shape.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GRANT_j1f = None
HQ, HKV, HD, BS, P_ROW, R = 32, 2, 128, 64, 64, 4
ROW_SHIFT = (0, 3, 5, 8)                                      # the rows' page offsets: union 64 + 8 = 72
UNION = P_ROW + ROW_SHIFT[-1]
LAST0 = 40                                                    # row r sees 40 + r tokens of its last page
BLOCK_BYTES = BS * HD * 2 * 2 + BS * 2                        # K + V + bias of one (KV head, block): 32,896 B
FORK_LAUNCH = dict(num_warps=4, num_stages=3, tile_size=64)   # = verify_rows.FORK_LAUNCH (the test checks)
SPLITS = 4                                                    # NOSI_ATTN_SPLITS of the cells (the shipped call)
MASK_BIAS = -3.0e4                                            # = verify_rows.MASK_BIAS
TOL = 1e-3                                                    # fia5's numeric tolerance (float64 reference)
FLASH = ("NS1R1", "NS1x4", "FLASHU1", "NSRU")                 # the flash controls (end-to-end, direct output)
AHEAD = ("FORKA1", "FORKA2", "FORKA3")                        # the fork with its loads ahead (ahead = 1, 2, 3)
BASE = ("FORK", "FORKT") + AHEAD + ("FORK1R",) + tuple(x + s for x in FLASH for s in ("", "_CP", "_CO"))  # _CP < _CO
# TRACK 2 (2026-11-02): the ROWS SWEEP over the SAME union tables as FORKT (built once by FORK's first call, read):
ROWS_M = (1, 2, 3, 4)            # RSMn: BLOCK_Q 4 (FORKT's kernel and launch); rows >= n off by the query length
ROWS_Q = (1, 2, 4)               # RSQn: BLOCK_Q n (tile rows 16 n); the query rows 0 .. n-1 of each request
ROWS = tuple("RSM%d" % n for n in ROWS_M) + tuple("RSQ%d" % n for n in ROWS_Q)
GRANT_j1n = None                 # the ncu cell (j1n): C 81 B 287 timing + the rows sweep + ncu per arm
NCU_ARMS = ("FORK", "FORKT", "FORK1R", "FLASHU1", "NS1R1", "NS1x4", "RSM1", "RSM4", "RSQ1", "RSQ2", "RSQ4")
NCU_REQUIRED = ("gpu__time_duration.sum", "dram__bytes_read.sum", "lts__t_sectors_srcunit_tex_op_read.sum")
NCU_OPTIONAL = ("dram__bytes_write.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__warps_active.avg.per_cycle_active", "lts__t_sector_hit_rate.pct",
                "launch__registers_per_thread", "launch__shared_mem_per_block_static",
                "launch__shared_mem_per_block_dynamic", "launch__occupancy_limit_registers",
                "launch__occupancy_limit_shared_mem", "launch__waves_per_multiprocessor", "launch__grid_size")


def log(msg):
    print("[r2_j1_forkbench %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=None, help="j1f (the grant)")
    ap.add_argument("--shapes", default="128:216,73:318", help="C:B pairs")
    ap.add_argument("--segments", default="2,4,8", help="the 3D path's segments per (request, KV head)")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=8, help="requests checked against the float64 reference")
    ap.add_argument("--max-shape-s", type=float, default=300.0)
    ap.add_argument("--max-run-s", type=float, default=900.0)
    ap.add_argument("--ncu", default=None, help="the ncu binary: one ncu child per arm and shape after the timing")
    ap.add_argument("--ncu-arms", default=",".join(NCU_ARMS))
    ap.add_argument("--ncu-cap", type=float, default=300.0, help="seconds per ncu child")
    ap.add_argument("--mode", default="main", choices=("main", "ncu-cell"))
    a = ap.parse_args(argv)
    try:
        a.shapes = {int(c): int(b) for c, b in (x.split(":") for x in a.shapes.split(",") if x)}
        a.segments = tuple(int(x) for x in a.segments.split(",") if x)
        a.ncu_arms = tuple(x for x in a.ncu_arms.split(",") if x)
    except ValueError:
        ap.error("--shapes C:B pairs, --segments integers")
    if [x for x in a.ncu_arms if x not in variants(a.segments)]:
        ap.error("--ncu-arms: unknown %s" % [x for x in a.ncu_arms if x not in variants(a.segments)])
    if a.mode == "ncu-cell" and (len(a.shapes) != 1 or len(a.ncu_arms) != 1):
        ap.error("--mode ncu-cell: one shape and one arm")
    if not a.shapes or any(b < 1 or b > 1024 for b in a.shapes.values()) or any(s < 2 or s > 64 for s in a.segments):
        ap.error("--shapes B in 1 .. 1024; --segments in 2 .. 64")
    if a.layers < 1 or a.reps < 3 or a.warmup < 1 or a.sample < 1:
        ap.error("--layers >= 1, --reps >= 3, --warmup >= 1, --sample >= 1")
    return a


def granted(cell) -> bool:
    g = {"j1f": GRANT_j1f, "j1n": GRANT_j1n}.get(cell)
    return isinstance(g, str) and len(g) == 20 and g.endswith("Z")


# ------------------------------------------------------------------ pure parts (CPU-tested)
def variants(segments):
    return BASE + ROWS + tuple("FORK3D%d" % s for s in segments)


def row_pages(b: int, r: int):
    """Arena blocks of row r of request b (ascending) and the tokens of its last page."""
    first = b * UNION + ROW_SHIFT[r]
    return list(range(first, first + P_ROW)), LAST0 + r


def work(name: str, B: int) -> dict:
    """Rows per call and LOGICAL pages per call (summed over (KV head, request or entry)) of a variant. <X>_CP: X's
    work; <X>_CO (the copy alone): no attention work (rows 0, pages 0)."""
    if name.endswith("_CO") and name[:-3] in FLASH:
        return dict(rows=0, pages=0)
    if name.endswith("_CP") and name[:-3] in FLASH:
        name = name[:-3]
    if name.startswith("FORK3D") or name in ("FORK", "FORKT") + AHEAD:
        return dict(rows=R * B, pages=HKV * B * UNION)
    if name.startswith(("RSM", "RSQ")) and name in ROWS:           # the same union pages; n active rows
        return dict(rows=int(name[3:]) * B, pages=HKV * B * UNION)
    if name in ("FORK1R", "NS1R1"):
        return dict(rows=B, pages=HKV * B * P_ROW)
    if name == "NS1x4":
        return dict(rows=R * B, pages=HKV * R * B * P_ROW)
    if name == "FLASHU1":
        return dict(rows=B, pages=HKV * B * UNION)
    if name == "NSRU":                                         # every entry reads its window prefix (shift_r + 64)
        return dict(rows=R * B, pages=HKV * B * sum(s + P_ROW for s in ROW_SHIFT))
    raise KeyError(name)


def rotation(names, r: int):
    n = len(names)
    k = r % n
    return list(names[k:]) + list(names[:k])


def stats(v):
    if not v:
        return None
    q = statistics.quantiles(v, n=4) if len(v) >= 2 else [v[0]] * 3
    mean = statistics.fmean(v)
    return dict(n=len(v), median=round(statistics.median(v), 4), min=round(min(v), 4), max=round(max(v), 4),
                q1=round(q[0], 4), q3=round(q[2], 4), cv=(round(statistics.pstdev(v) / mean, 4) if len(v) >= 2 else None))


PAIRS = (("FORK", "FLASHU1", "equal logical bytes (union 72): the 4-row tile's cost at the same bytes"),
         ("FORK1R", "NS1R1", "equal rows and logical bytes (1 row, 64 pages): fork time / flash end-to-end time "
          "(direct output) at matched logical work; not a physical per-byte ceiling"),
         ("FORK", "NS1x4", "equal rows (4 per request): the shared read's saving vs the baseline's 4 calls"),
         ("FORKT", "FORK", "the in-kernel union build's cost"),
         ("NSRU", "FORK", "the shipped kernel's own sharing (L2) vs the fork"),
         ("FORKA1", "FORK", "the fork with the table and extents read two tiles ahead (bit-equal by FORKA1-EQ)"),
         ("FORKA2", "FORK", "the same and the bias one tile ahead (bit-equal by FORKA2-EQ)"),
         ("FORKA3", "FORK", "the fork's K / V page-id loads kept in the source; ext_rows and the bias ahead "
          "(compiled: the async K / V copies are lost as at levels 1-2; FORKA3-EQ)"))
PAIRS_CP = (("FORK", "FLASHU1_CP", "j1f form (flash: wrapper + copy_): equal logical bytes"),
            ("FORK1R", "NS1R1_CP", "j1f form (flash: wrapper + copy_): equal rows and bytes"),
            ("FORK", "NS1x4_CP", "j1f form (flash: wrapper + copy_): equal rows"),
            ("NSRU_CP", "FORK", "j1f form (flash: wrapper + copy_): the shipped kernel's own sharing vs the fork"))


def summarize(times: dict, B: int, layers: int, valid: dict, segments) -> dict:
    """Per variant: median ms per pass and per call, logical bytes per call, logical TB/s; the matched ratios (INVALID
    where a member failed its NUM, EXACT, OUTEQ or FORKAn-EQ gate); the 3D split vs FORK (only when FORK and the 3D
    member are valid; else best_3d is None and best_3d_note says why); FORK - FLASHU1 (both valid); per flash
    control: direct (X: the end-to-end library call into a preallocated output, NOT kernel-only), copy-only (X_CO),
    total in the j1f form (X_CP) and the residual X_CP - X - X_CO (a difference of separately timed forms)."""
    st = {k: stats(v) for k, v in times.items()}
    per = {}
    for k, s in st.items():
        if s is None:
            continue
        w = work(k, B)
        ms = s["median"] / layers
        by = w["pages"] * BLOCK_BYTES
        per[k] = dict(ms_pass=s["median"], ms_call=round(ms, 5), rows=w["rows"], pages=w["pages"], bytes=by,
                      logical_tb_s=round(by / (ms * 1e-3) / 1e12, 4) if ms > 0 and by > 0 else None,
                      valid=bool(valid.get(k)))
    ratios = {}
    pairs = list(PAIRS) + list(PAIRS_CP) + [("FORK3D%d" % s, "FORK", "the 3D split (%d segments) vs the 2D launch" % s)
                                            for s in segments]
    for a_, b_, what in pairs:
        if a_ in per and b_ in per:
            ok = per[a_]["valid"] and per[b_]["valid"]
            ratios["%s/%s" % (a_, b_)] = dict(ratio=round(per[a_]["ms_call"] / per[b_]["ms_call"], 4), what=what,
                                              valid=ok, note=None if ok else
                                              "INVALID: a member failed NUM, EXACT, OUTEQ or FORKAn-EQ")
    out = dict(stats_ms=st, per=per, ratios=ratios)
    if "FORK" in per and "FLASHU1" in per and per["FORK"]["valid"] and per["FLASHU1"]["valid"]:
        f, u = per["FORK"]["ms_call"], per["FLASHU1"]["ms_call"]
        out["fork_minus_flashu1_ms_per_pass"] = round((f - u) * layers, 3)
    ft = {}
    for x in FLASH:
        if x in per and x + "_CP" in per and x + "_CO" in per:
            d_, cp, co = per[x]["ms_pass"], per[x + "_CP"]["ms_pass"], per[x + "_CO"]["ms_pass"]
            ft[x] = dict(direct_e2e_ms_pass=d_, copy_only_ms_pass=co, total_j1f_form_ms_pass=cp,
                         residual_ms_pass=round(cp - d_ - co, 4),
                         valid=per[x]["valid"] and per[x + "_CP"]["valid"] and per[x + "_CO"]["valid"])
    out["flash_timing"] = ft
    out["rows_sweep"] = rows_sweep(per, layers)
    d3 = [(per[k]["ms_call"], k) for k in per if k.startswith("FORK3D") and per[k]["valid"]]
    out["best_3d"], out["best_3d_note"] = None, None
    if "FORK" not in per or not per["FORK"]["valid"]:                    # the same rule as the ratios
        out["best_3d_note"] = "INVALID: FORK failed NUM or EXACT (or did not run): no 3D saving"
    elif not d3:
        out["best_3d_note"] = "INVALID: no valid 3D variant"
    else:
        best = min(d3)
        out["best_3d"] = dict(variant=best[1], ratio=round(best[0] / per["FORK"]["ms_call"], 4),
                              saving_ms_per_pass=round((per["FORK"]["ms_call"] - best[0]) * layers, 3))
    return out


def rows_sweep(per: dict, layers: int) -> dict:
    """TRACK 2: the rows sweep at fixed union bytes (ms per pass; a value only where every member is valid).
    masked: RSMn (n = 1..4 active rows by extents; the same kernel, launch and MMA work as FORKT): the marginal ms per
    extra active row = least squares slope over n (the masking / softmax / V-reach share). tile: RSQn (BLOCK_Q n, tile
    rows 16 n): ms per pass for n = 1, 2, 4 and the marginal ms per extra row of each step (the MMA rows, registers and
    occupancy share). e_fork = RSQ1 - FORK1R: one row over the 72-page union minus one row over its own 64 pages, in the
    fork (the E analogue inside the fork kernel). tile4_fork = FORKT - RSQ1: the 4-row tile at fixed bytes."""
    def ok(*names):
        return all(n in per and per[n]["valid"] for n in names)

    def ms(n):
        return per[n]["ms_pass"]
    out = dict(masked=None, tile=None, e_fork_ms_pass=None, tile4_fork_ms_pass=None)
    m = [n for n in ROWS_M if ok("RSM%d" % n)]
    if len(m) == len(ROWS_M):
        xs, ys = list(m), [ms("RSM%d" % n) for n in m]
        xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
        out["masked"] = dict(ms_pass={n: ms("RSM%d" % n) for n in m}, slope_ms_per_row=round(slope, 4),
                             rsm4_minus_forkt=round(ms("RSM4") - ms("FORKT"), 4) if ok("FORKT") else None)
    if ok(*("RSQ%d" % n for n in ROWS_Q)):
        q = {n: ms("RSQ%d" % n) for n in ROWS_Q}
        out["tile"] = dict(ms_pass=q, per_row_1_to_2=round(q[2] - q[1], 4), per_row_2_to_4=round((q[4] - q[2]) / 2, 4))
    if ok("RSQ1", "FORK1R"):
        out["e_fork_ms_pass"] = round(ms("RSQ1") - ms("FORK1R"), 4)
    if ok("FORKT", "RSQ1"):
        out["tile4_fork_ms_pass"] = round(ms("FORKT") - ms("RSQ1"), 4)
    return out


def decision_lines(name: str, s: dict):
    """The summary's decision lines of one shape. An invalid ratio says INVALID; an invalid FORK - FLASHU1 or 3D saving
    is SUPPRESSED (no number printed)."""
    out = []
    for k, v in s["ratios"].items():
        out.append("J1F %s RATIO %s = %s (%s)%s" % (name, k, v["ratio"], v["what"], "" if v["valid"] else " INVALID"))
    for x, t in s["flash_timing"].items():
        out.append("J1F %s FLASHT %-7s direct (end-to-end, preallocated output) %.3f, copy-only %.3f, total (j1f form) "
                   "%.3f, residual %.3f ms per pass%s" % (name, x, t["direct_e2e_ms_pass"], t["copy_only_ms_pass"],
                                                         t["total_j1f_form_ms_pass"], t["residual_ms_pass"],
                                                         "" if t["valid"] else " INVALID"))
    d = s.get("fork_minus_flashu1_ms_per_pass")
    out.append("J1F %s FORK - FLASHU1 = %s" % (name, "%s ms per pass" % d if d is not None else
                                               "SUPPRESSED (FORK or FLASHU1 invalid or missing)"))
    b = s.get("best_3d")
    out.append("J1F %s best 3D %s" % (name, b if b is not None else "SUPPRESSED (%s)" % s.get("best_3d_note")))
    rs = s.get("rows_sweep") or {}
    for k in ("masked", "tile", "e_fork_ms_pass", "tile4_fork_ms_pass"):
        if k in rs:
            out.append("J1F %s ROWS %s %s" % (name, k, rs[k] if rs[k] is not None else
                                               "SUPPRESSED (a member invalid or missing)"))
    return out


# ------------------------------------------------------------------ ncu (TRACK 2; pure parts CPU-tested)
def ncu_metric_names(query_text: str):
    """Metric base names in `ncu --query-metrics --chip ga100` output."""
    import re
    return set(re.findall(r"^\s*([a-z0-9_]+__[a-z0-9_]+)", query_text, flags=re.M))


def ncu_metric_list(names):
    """(missing required, the list to ask): the required metrics and the optional ones the query names."""
    base = lambda m: m.split(".", 1)[0]                                   # noqa: E731
    missing = [m for m in NCU_REQUIRED if base(m) not in names]
    return missing, list(NCU_REQUIRED) + [m for m in NCU_OPTIONAL if base(m) in names]


def ncu_argv(ncu: str, a, C: int, B: int, arm: str, child_out: str, rep: str, metrics):
    """One ncu child: the profiler window holds ONE call (one layer) of ONE arm, after its warm-up outside the window;
    every kernel replayed with the caches flushed (--cache-control all), so DRAM bytes are the arm's own reads."""
    return [ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", "all",
            "--clock-control", "none", "--metrics", ",".join(metrics), "--section", "LaunchStats", "--section",
            "Occupancy", "--target-processes", "application-only", "-f", "-o", rep,
            sys.executable, os.path.abspath(__file__), "--mode", "ncu-cell", "--out", child_out, "--cell", a.cell,
            "--shapes", "%d:%d" % (C, B), "--ncu-arms", arm, "--seed", str(a.seed), "--sample", str(a.sample),
            "--segments", ",".join(str(s) for s in a.segments)]


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def parse_ncu_csv(text: str, fields):
    """Kernel rows of `ncu --import <rep> --csv --page raw --print-units base`: the header row naming 'Kernel Name',
    then the rows with a numeric ID (the units row has none)."""
    import csv
    import io
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hi is None:
        raise ValueError("no header row with 'Kernel Name'")
    head, out = rows[hi], []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        if not d.get("ID", "").strip().isdigit():
            continue
        row = dict(id=int(d["ID"]), name=d["Kernel Name"])
        row.update({f: _num(d.get(f, "")) for f in fields})
        out.append(row)
    return out


def ncu_arm_summary(rows, logical_bytes: int) -> dict:
    """One arm's kernels (one call): time, DRAM bytes read, L2 -> SM bytes (sectors x 32), their ratios to the
    LOGICAL bytes of the call, achieved occupancy and registers of the longest kernel."""
    g = lambda r, k: r.get(k) or 0.0                                      # noqa: E731
    t = sum(g(r, "gpu__time_duration.sum") for r in rows)
    rd = sum(g(r, "dram__bytes_read.sum") for r in rows)
    l2 = sum(g(r, "lts__t_sectors_srcunit_tex_op_read.sum") for r in rows) * 32.0
    top = max(rows, key=lambda r: g(r, "gpu__time_duration.sum")) if rows else {}
    return dict(n_kernels=len(rows), kernels=sorted({r["name"] for r in rows}), time_us=round(t / 1e3, 3),
                dram_read=rd, l2sm_bytes=l2, logical_bytes=logical_bytes,
                dram_over_logical=round(rd / logical_bytes, 4) if logical_bytes else None,
                l2sm_over_logical=round(l2 / logical_bytes, 4) if logical_bytes else None,
                occupancy_pct=top.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
                warps_per_cycle=top.get("sm__warps_active.avg.per_cycle_active"),
                registers=top.get("launch__registers_per_thread"),
                limit_registers=top.get("launch__occupancy_limit_registers"),
                limit_shared=top.get("launch__occupancy_limit_shared_mem"))


def swapped_out_view(out, hkv: int):
    """The view of a preallocated (N, 1, HQ, HD) output that fwd_kvcache's out_ argument needs. With one query row,
    HQ > HKV, no window and HD % 8 == 0, the op swaps q to (N, HQ/HKV, HKV, HD) (q.reshape(N, HKV, G, HD).transpose(1,
    2)) and checks out_ against THAT shape; it then returns out.transpose(1, 2).reshape(N, 1, HQ, HD). This view puts
    element [n, g, k, d] of the swapped output at [n, 0, k * G + g, d] of `out`: the wrapper's own return layout."""
    n, s, hq, hd = out.shape
    assert s == 1 and hq % hkv == 0 and hq > hkv and hd % 8 == 0 and out.is_contiguous(), (tuple(out.shape), hkv)
    return out.view(n, hkv, hq // hkv, hd).transpose(1, 2)


def nonfinite(t):
    """(count of non-finite values in t, the index of the first one or None). torch.isfinite: NaN, +Inf, -Inf."""
    import torch
    bad = ~torch.isfinite(t)
    n = int(bad.sum())
    return n, (tuple(int(i) for i in bad.nonzero()[0]) if n else None)


def out_row(name: str, out, key, B: int):
    """The output row (HQ, HD) of variant `name` for the reference key (b, r), or None when the variant does not
    compute that row. <X>_CP and <X>_CO hold X's rows."""
    b, r = key
    base = name[:-3] if name.endswith(("_CP", "_CO")) else name
    if base in ROWS:                                                # the rows sweep: active rows 0 .. n-1 only
        n = int(base[3:])
        if r == "U" or r >= n:
            return None
        return out.view(B * n, HQ, HD)[b * n + r]
    if base in ("FORK1R", "NS1R1"):
        return out.view(B, HQ, HD)[b] if r == 0 else None
    if base == "FLASHU1":
        return out.view(B, HQ, HD)[b] if r == "U" else None
    return None if r == "U" else out.view(B * R, HQ, HD)[b * R + r]


def num_check(name: str, out, ref: dict, B: int) -> dict:
    """Over the sampled requests (every row the variant computes, every head): torch.isfinite on the output, on the
    float64 reference and on their difference FIRST; a row with any non-finite value is counted (the first index kept)
    and never enters the max. err = max |output - reference| over the finite rows (a NaN cannot vanish in the max)."""
    import torch
    err, bad, first, rows = 0.0, 0, None, 0
    for key, want in ref.items():
        got = out_row(name, out, key, B)
        if got is None:
            continue
        rows += 1
        g = got.double()
        d = g - want
        fin = torch.isfinite(g) & torch.isfinite(want) & torch.isfinite(d)
        if not bool(fin.all()):
            nf = ~fin
            bad += int(nf.sum())
            if first is None:
                h, x = (int(i) for i in nf.nonzero()[0])
                first = dict(request=key[0], row=key[1], head=h, dim=x, out=repr(float(g[h, x])),
                             ref=repr(float(want[h, x])))
            continue
        err = max(err, float(d.abs().max()))
    return dict(err=err, nonfinite=bad, first=first, rows=rows)


def num_verdict(chk: dict, tol: float = TOL):
    """(passed, detail) of a NUM gate: FAIL on any non-finite value (count and first index), on no checked row, or
    on err > tol."""
    if chk["nonfinite"]:
        f = chk["first"]
        return False, ("NON-FINITE: %d value(s) in the output, the float64 reference or their difference; first at "
                       "request %s row %s head %d dim %d (out %s, ref %s); max |out - float64| over the finite rows "
                       "%.3e" % (chk["nonfinite"], f["request"], f["row"], f["head"], f["dim"], f["out"], f["ref"],
                                 chk["err"]))
    if not chk["rows"]:
        return False, "no row checked (no reference row for this variant)"
    return chk["err"] <= tol, "max |out - float64| %.3e over %d rows (tol %.0e)" % (chk["err"], chk["rows"], tol)


def rows_eq(name: str, got, forkt, B: int):
    """(passed, detail): the rows sweep variant's rows 0 .. n-1 (its output holds n rows per request) = FORKT's same
    rows bit for bit, and finite."""
    n = int(name[3:])
    ref = forkt.view(B, R, HQ, HD)[:, :n].contiguous()
    act = got.view(B, n, HQ, HD)
    nd = int((bits(act) != bits(ref)).sum())
    nf = nonfinite(act)[0]
    return nd == 0 and nf == 0, "rows 0..%d = FORKT bit for bit: %d values differ; non-finite %d" % (n - 1, nd, nf)


def repeat_check(out, first):
    """(passed, detail) of an EXACT gate: the output after the last pass = the first output bit for bit AND both
    hold only finite values (identical NaN bits are equal bits; they FAIL here)."""
    eq = bool((bits(out) == bits(first)).all())
    n1, i1 = nonfinite(out)
    n0, i0 = nonfinite(first)
    ok = eq and n1 == 0 and n0 == 0
    return ok, "output after the last pass = the first output, bit for bit: %s; non-finite values: last %d (first at " \
               "%s), first output %d (first at %s)" % (eq, n1, i1, n0, i0)


def time_variants(parts: dict, names, layers: int, warmup: int, reps: int, timer, t0: float, max_s: float,
                  clock=time.perf_counter):
    """The order-rotated timing loop: `warmup` passes of each variant, then `reps` rounds in rotated order; one pass =
    timer(callable, layers) in ms. Stops after a round once clock() - t0 > max_s. Returns (times, warm, stop)."""
    times = {v: [] for v in names}
    warm = {v: [] for v in names}
    for v in names:
        for _ in range(warmup):
            warm[v].append(timer(parts[v], layers))
    stop = None
    for r in range(reps):
        for v in rotation(names, r):
            times[v].append(timer(parts[v], layers))
        if clock() - t0 > max_s:
            stop = "wall clock above %.0f s after pass %d" % (max_s, r)
            break
    return times, warm, stop


def cuda_timer(fn, layers: int) -> float:
    """One pass on the GPU: `layers` calls between two CUDA events; ms."""
    import torch
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(layers):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return float(e0.elapsed_time(e1))


# ------------------------------------------------------------------ the GPU parts
def build_gpu(B: int, seed: int, segments, sample: int):
    """The variants of one shape: {name: callable}, {name: output tensor}, meta (the union lengths, the references)."""
    import torch
    import fork_vllm_unified_attention as FK
    import fork_vllm_unified_attention_ahead as FKA                         # the fork + the [ahead] lines (sibling)
    from flash_attn_nosa import flash_attn_with_kvcache as FA
    from flash_attn_nosa import flash_attn_interface as FAI
    FAOP = FAI.flash_attn_cuda.fwd_kvcache                                   # the op the wrapper FA calls
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf = torch.bfloat16
    E = R * B
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(bf).to(dev)   # noqa: E731
    NB = B * UNION + 8
    ak, av, ab = rnd(NB, BS, HKV, HD), rnd(NB, BS, HKV, HD), rnd(NB, BS, HKV)
    q = rnd(E, HQ, HD)
    scale = HD ** -0.5
    i32 = torch.int32
    ar = torch.arange(B, dtype=torch.int64, device=dev)
    pg = (ar.view(B, 1, 1) * UNION + torch.tensor(ROW_SHIFT, device=dev).view(1, R, 1)
          + torch.arange(P_ROW, device=dev).view(1, 1, P_ROW))                     # (B, R, 64) arena blocks
    lastlen = (LAST0 + torch.arange(E, dtype=i32, device=dev) % R).contiguous()
    keys4 = pg.reshape(E, 1, P_ROW).expand(E, HKV, P_ROW).to(i32).contiguous()
    lists4 = dict(page_key=keys4, page_table=keys4.clone(), num_pages=torch.full((E, HKV), P_ROW, dtype=i32, device=dev),
                  last_len=lastlen, seq_idx=torch.zeros(E, dtype=i32, device=dev), slots_per_entry=1)
    keys1 = pg[:, 0].reshape(B, 1, P_ROW).expand(B, HKV, P_ROW).to(i32).contiguous()
    lists1 = dict(page_key=keys1, page_table=keys1.clone(), num_pages=torch.full((B, HKV), P_ROW, dtype=i32, device=dev),
                  last_len=torch.full((B,), LAST0, dtype=i32, device=dev), seq_idx=torch.zeros(B, dtype=i32, device=dev),
                  slots_per_entry=1)
    cu4 = torch.arange(0, E + 1, R, dtype=i32, device=dev)
    cu1 = torch.arange(0, B + 1, dtype=i32, device=dev)
    seq = torch.zeros(B, dtype=i32, device=dev)
    parts, outs, meta = {}, {}, dict(B=B, E=E, NB=NB, union=UNION, segments=list(segments))

    def tables(width):
        return (torch.zeros((HKV, B, width), dtype=i32, device=dev), torch.zeros((HKV, B, width), dtype=i32, device=dev),
                torch.zeros((HKV, B), dtype=i32, device=dev))

    def fork_call(qx, out, cu, rows, tab, ext, ln, lists, extra=None, mod=None, block_q=None):
        ua = (mod or FK).unified_attention

        def call():                                  # rows = the query rows per request; block_q = BLOCK_Q (default rows)
            ua(qx, ak, av, out, cu, rows, seq, BS, scale, True, (-1, -1), tab, 0, None, None, None,
               union_ext=ext, union_len=ln, cis_bias=ab, build_union=lists, rows_per_q_block=block_q or rows,
               **dict(FORK_LAUNCH, **(extra or {})))
        return call
    o = torch.zeros((E, HQ, HD), dtype=bf, device=dev)
    t4 = tables(R * P_ROW)
    parts["FORK"], outs["FORK"] = fork_call(q, o, cu4, R, *t4, lists4), o
    parts["FORK"]()                                                          # builds the tables once (FORKT's input)
    torch.cuda.synchronize()
    meta["ln_fork"] = t4[2]
    tT = tuple(t.clone() for t in t4)
    oT = torch.zeros_like(o)
    parts["FORKT"], outs["FORKT"] = fork_call(q, oT, cu4, R, *tT, None), oT
    # The rows sweep: FORKT's tables and extents unchanged (the V reach = the 4 rows' largest byte: the same bytes);
    # the requests carry n query rows (rows 0 .. n-1). RSMn keeps BLOCK_Q 4 (FORKT's kernel and launch): the rows
    # n .. 3 of each program are off by the query length (masked loads, -inf scores, masked stores). Masking them by
    # zero extents instead would store 0 / 0 = NaN rows (softmax_step: L becomes 0 when a row sees no token).
    for n in sorted(set(ROWS_M) | set(ROWS_Q)):
        qn = q.view(B, R, HQ, HD)[:, :n].contiguous().view(B * n, HQ, HD)
        cun = torch.arange(0, B * n + 1, n, dtype=i32, device=dev)
        if n in ROWS_M:
            oM = torch.zeros((B * n, HQ, HD), dtype=bf, device=dev)
            parts["RSM%d" % n], outs["RSM%d" % n] = fork_call(qn, oM, cun, n, *tT, None, block_q=R), oM
        if n in ROWS_Q:                               # BLOCK_Q n: the tile rows 16 n (rows 0 .. n-1 read bytes 0 .. n-1)
            oQ = torch.zeros((B * n, HQ, HD), dtype=bf, device=dev)
            parts["RSQ%d" % n], outs["RSQ%d" % n] = fork_call(qn, oQ, cun, n, *tT, None), oQ
    for lvl, v in enumerate(AHEAD, 1):                                       # production mode (build in the kernel)
        oA = torch.zeros_like(o)
        parts[v], outs[v] = fork_call(q, oA, cu4, R, *tables(R * P_ROW), lists4, dict(ahead=lvl), FKA), oA
    for s in segments:
        o3 = torch.zeros_like(o)
        t3 = tables(R * P_ROW)
        nan = float("nan")                                                   # the PATH witness: NaN until a 3D launch
        seg = dict(seq_threshold_3D=B, num_par_softmax_segments=s,
                   softmax_segm_output=torch.full((E, HQ, s, HD), nan, dtype=torch.float32, device=dev),
                   softmax_segm_max=torch.full((E, HQ, s), nan, dtype=torch.float32, device=dev),
                   softmax_segm_expsum=torch.full((E, HQ, s), nan, dtype=torch.float32, device=dev))
        parts["FORK3D%d" % s], outs["FORK3D%d" % s] = fork_call(q, o3, cu4, R, *t3, lists4, seg), o3
        meta.setdefault("segm_max", {})["FORK3D%d" % s] = seg["softmax_segm_max"]
    q1 = q.view(B, R, HQ, HD)[:, 0].contiguous()
    o1 = torch.zeros((B, HQ, HD), dtype=bf, device=dev)
    t1 = tables(P_ROW)
    parts["FORK1R"], outs["FORK1R"] = fork_call(q1, o1, cu1, 1, *t1, lists1), o1
    meta["ln_fork1"] = t1[2]
    # the shipped flash call: contiguous windows (made here, not timed)
    flat = pg.reshape(E, P_ROW)
    kw4 = ak[flat].reshape(E, P_ROW * BS, HKV, HD).contiguous()
    vw4 = av[flat].reshape(E, P_ROW * BS, HKV, HD).contiguous()
    bw4 = ab[flat].reshape(E, P_ROW * BS, HKV).contiguous()
    lens4 = ((P_ROW - 1) * BS + lastlen).to(i32).contiguous()
    qe = q.view(E, 1, HQ, HD)

    def flash_set(x, qx, kc, vc, bc, lens, cbi=None):
        """Control x three ways on the same inputs: x (the library op writes the preallocated output directly, as
        FORK does), x_CP (the j1f form: the wrapper's new tensor copy_d into the output), x_CO (that copy_ alone)."""
        o_d, o_cp, o_co = (torch.zeros((qx.shape[0], 1, HQ, HD), dtype=bf, device=dev) for _ in range(3))
        ov = swapped_out_view(o_d, HKV)

        def direct():                    # = flash_attn_with_kvcache's own op call with out_ = ov (it passes None)
            FAOP(qx, kc, vc, bc, None, None, lens, None, None, cbi, None, None, None, ov, scale, False, -1, -1, 0.0,
                 True, SPLITS)

        def wrapper_cp():
            o_cp.copy_(FA(qx, kc, vc, bc, cache_seqlens=lens, cache_batch_idx=cbi, num_splits=SPLITS))

        def copy_co():
            o_co.copy_(o_cp)
        parts[x], outs[x] = direct, o_d
        parts[x + "_CP"], outs[x + "_CP"] = wrapper_cp, o_cp
        parts[x + "_CO"], outs[x + "_CO"] = copy_co, o_co
    flash_set("NS1x4", qe, kw4, vw4, bw4, lens4)
    k1 = kw4.view(B, R, P_ROW * BS, HKV, HD)[:, 0].contiguous()
    v1 = vw4.view(B, R, P_ROW * BS, HKV, HD)[:, 0].contiguous()
    b1 = bw4.view(B, R, P_ROW * BS, HKV)[:, 0].contiguous()
    lens1 = torch.full((B,), (P_ROW - 1) * BS + LAST0, dtype=i32, device=dev)
    q1e = q1.view(B, 1, HQ, HD)
    flash_set("NS1R1", q1e, k1, v1, b1, lens1)
    ku = ak[:B * UNION].view(B, UNION * BS, HKV, HD)                        # the union windows: arena views
    vu = av[:B * UNION].view(B, UNION * BS, HKV, HD)
    bu = ab[:B * UNION].view(B, UNION * BS, HKV)
    lensu = torch.full((B,), UNION * BS, dtype=i32, device=dev)
    flash_set("FLASHU1", q1e, ku, vu, bu, lensu)
    j = torch.arange(UNION * BS, device=dev).view(1, UNION * BS) // BS           # the page of each union token
    sh = torch.tensor(ROW_SHIFT, device=dev).repeat(B).view(E, 1)
    vis = (j >= sh)                                                           # pages before the row's first: masked
    biasE = torch.where(vis.unsqueeze(-1), bu.repeat_interleave(R, 0), torch.full((), MASK_BIAS, dtype=bf,
                                                                                   device=dev)).contiguous()
    lensR = ((sh.view(E) + P_ROW - 1) * BS + lastlen).to(i32).contiguous()
    cbi = (torch.arange(E, device=dev) // R).to(i32).contiguous()
    flash_set("NSRU", qe, ku, vu, biasE[:B], lensR, cbi)
    # float64 references of the sampled requests
    idx = sorted({int(x) for x in torch.linspace(0, B - 1, min(sample, B)).round().tolist()})
    meta["sample"] = idx
    ref = {}
    for b in idx:
        for r in range(R):
            pages, last = row_pages(b, r)
            ref[(b, r)] = _ref(ak, av, ab, q[b * R + r], pages, last, scale)
        ref[(b, "U")] = _ref(ak, av, ab, q[b * R], list(range(b * UNION, (b + 1) * UNION)), BS, scale)
    meta["ref"] = ref
    return parts, outs, meta


def _ref(ak, av, ab, qrow, pages, last: int, scale: float):
    """float64 attention of one row (HQ heads) over `pages` (the last one `last` tokens): (q.k + bias) x scale."""
    import torch
    K = ak[pages].double().reshape(-1, HKV, HD)
    V = av[pages].double().reshape(-1, HKV, HD)
    Bi = ab[pages].double().reshape(-1, HKV)
    n = (len(pages) - 1) * BS + last
    K, V, Bi = K[:n], V[:n], Bi[:n]
    out = torch.empty((HQ, HD), dtype=torch.float64, device=qrow.device)
    g = HQ // HKV
    for h in range(HQ):
        kv = h // g
        s = (K[:, kv] @ qrow[h].double() + Bi[:, kv]) * scale
        p = torch.softmax(s, 0)
        out[h] = p @ V[:, kv]
    return out


def bits(t):
    import torch
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def run_shape(a, C: int, B: int, t0: float):
    import torch
    name = "C%d_B%d" % (C, B)
    gates = []
    parts, outs, meta = build_gpu(B, a.seed + C, a.segments, a.sample)
    names = [v for v in variants(a.segments) if v in parts]
    ln4, ln1 = meta.pop("ln_fork"), meta.pop("ln_fork1")
    ref = meta.pop("ref")
    for v in names:                                                           # the first outputs (EXACT's reference)
        try:
            parts[v]()
        except Exception as e:                                                 # a broken variant: reported, dropped
            gates.append(("%s-%s-RUN" % (name, v), not v.startswith("FORK3D"), False, "EXCEPTION %s: %s" % (
                type(e).__name__, str(e)[:200])))
            parts.pop(v)
    for x in FLASH:                                                           # x_CO copies x_CP's output: without
        if x + "_CO" in parts and x + "_CP" not in parts:                     # it, the copy has no meaning
            gates.append(("%s-%s_CO-RUN" % (name, x), True, False, "dropped: its source %s_CP did not run" % x))
            parts.pop(x + "_CO")
    names = [v for v in names if v in parts]
    torch.cuda.synchronize()
    gates.append(("%s-UNION" % name, True, bool((ln4 == UNION).all()) and bool((ln1 == P_ROW).all()),
                  "FORK union_len min %d max %d (want %d); FORK1R min %d max %d (want %d)" % (
                      int(ln4.min()), int(ln4.max()), UNION, int(ln1.min()), int(ln1.max()), P_ROW)))
    first = {v: outs[v].clone() for v in names}
    valid = {}
    path3d = {}
    for v, sm in meta.pop("segm_max", {}).items():                        # the 3D launch wrote its segment maxima
        path3d[v] = v in parts and not bool(torch.isnan(sm).all())
        gates.append(("%s-%s-PATH" % (name, v), False, path3d[v], "3D path taken (segment maxima written): %s" % (
            path3d[v])))
    for v in names:
        ok, detail = num_verdict(num_check(v, outs[v], ref, B))             # isfinite first, then the max
        ok = ok and path3d.get(v, True)                                      # a FORK3D without its 3D launch: INVALID
        valid[v] = ok
        gates.append(("%s-%s-NUM" % (name, v), not v.startswith("FORK3D"), ok, "%s; requests %s" % (
            detail, meta["sample"])))
    if "FORKT" in first:
        eq = bool((bits(first["FORKT"]) == bits(first["FORK"])).all())
        gates.append(("%s-FORKT-EQ" % name, True, eq, "FORKT = FORK bit for bit: %s" % eq))
    for v in AHEAD:                                                          # the hard constraint: the same bits
        if v in first and "FORK" in first:
            eq = bool((bits(first[v]) == bits(first["FORK"])).all())
            nd = int((bits(first[v]) != bits(first["FORK"])).sum())
            valid[v] = valid[v] and eq
            gates.append(("%s-%s-EQ" % (name, v), True, eq, "%s = FORK bit for bit: %s (%d values differ)" % (
                v, eq, nd)))
    for v in ROWS:                                                           # the rows sweep vs FORKT, row by row
        if v in first and "FORKT" in first:
            ok, d = rows_eq(v, first[v], first["FORKT"], B)
            must = v.startswith("RSM")                    # RSM: the same kernel and launch -> the same bits (MUST)
            valid[v] = valid[v] and (ok or not must)      # RSQ: another BLOCK_M (tile layout): REPORT only
            gates.append(("%s-%s-EQ" % (name, v), must, ok, d))
    for x in FLASH:                                                          # the direct write = the j1f form
        if x in first and x + "_CP" in first:
            eq = bool((bits(first[x]) == bits(first[x + "_CP"])).all())
            n_d, n_cp = nonfinite(first[x])[0], nonfinite(first[x + "_CP"])[0]
            ok = eq and n_d == 0 and n_cp == 0
            valid[x] = valid[x] and ok
            gates.append(("%s-%s-OUTEQ" % (name, x), True, ok, "end-to-end direct output = %s_CP (wrapper + copy_) bit for "
                          "bit: %s; non-finite values: direct %d, %s_CP %d" % (x, eq, n_d, x, n_cp)))
    times, warm, stop = time_variants(parts, names, a.layers, a.warmup, a.reps, cuda_timer, t0, a.max_shape_s)
    for v in names:
        eq, d = repeat_check(outs[v], first[v])
        valid[v] = valid[v] and eq
        gates.append(("%s-%s-EXACT" % (name, v), not v.startswith("FORK3D"), eq, d))
    summ = summarize(times, B, a.layers, valid, a.segments)
    wall = time.perf_counter() - t0
    gates.append(("%s-BUDGET" % name, True, wall <= a.max_shape_s and stop is None, "wall %.1f s (limit %.0f); stop "
                  "%s" % (wall, a.max_shape_s, stop)))
    del parts, outs, first
    torch.cuda.empty_cache()
    return gates, dict(C=C, B=B, meta=meta, times_ms=times, warmup_ms=warm, summary=summ, wall_s=round(wall, 1),
                       stop=stop)


def run_logged(argv, path, timeout):
    """argv with stdout + stderr to path, in its own process group, killed at the timeout: the exit or 'timeout'."""
    import signal
    import subprocess
    with open(path, "w") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            return "timeout"


def run_ncu(a, C: int, B: int, gates) -> dict:
    """The ncu children of one shape (one per arm of --ncu-arms); REPORT gates <shape>-NCU-<arm>."""
    name = "C%d_B%d" % (C, B)
    qpath = os.path.join(a.out, "ncu_metrics_query.txt")
    if not os.path.exists(qpath):
        run_logged([a.ncu, "--query-metrics", "--chip", "ga100"], qpath, 120)
    missing, metrics = ncu_metric_list(ncu_metric_names(open(qpath, errors="replace").read()))
    if missing:
        gates.append(("%s-NCU" % name, False, False, "the metric query misses %s" % missing))
        return {}
    res = {}
    for arm in a.ncu_arms:
        tag = "%s_%s" % (name, arm)
        child, rep = os.path.join(a.out, "ncu_" + tag), os.path.join(a.out, "ncu_%s_rep" % tag)
        t1 = time.time()
        rc = run_logged(ncu_argv(a.ncu, a, C, B, arm, child, rep, metrics), os.path.join(a.out, "ncu_%s.log" % tag),
                        a.ncu_cap)
        exp, summ = None, None
        if os.path.exists(rep + ".ncu-rep"):
            csv_path = os.path.join(a.out, "ncu_%s_raw.csv" % tag)
            exp = run_logged([a.ncu, "--import", rep + ".ncu-rep", "--csv", "--page", "raw", "--print-units", "base"],
                             csv_path, 300)
            if exp == 0:
                rows = parse_ncu_csv(open(csv_path, errors="replace").read(), metrics)
                summ = ncu_arm_summary(rows, work(arm, B)["pages"] * BLOCK_BYTES)
        res[arm] = dict(child_exit=rc, export_exit=exp, seconds=round(time.time() - t1, 1), summary=summ)
        gates.append(("%s-NCU-%s" % (name, arm), False, rc == 0 and exp == 0 and summ is not None,
                      "child exit %s, export %s; %s" % (rc, exp, summ)))
    return res


def mode_ncu_cell(a) -> int:
    """Under ncu: build the shape, warm the arm OUTSIDE the profiler window, then ONE call (one layer) inside it."""
    import torch
    (C, B), = a.shapes.items()
    arm = a.ncu_arms[0]
    with torch.inference_mode():
        parts, outs, meta = build_gpu(B, a.seed + C, a.segments, a.sample)
        for _ in range(2):
            parts[arm]()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        parts[arm]()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    with open(os.path.join(a.out, "cell.json"), "w") as f:
        json.dump(dict(C=C, B=B, arm=arm, logical_bytes=work(arm, B)["pages"] * BLOCK_BYTES, union=UNION,
                       finite=nonfinite(outs[arm])[0] == 0), f, indent=1)
    return 0


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if not granted(a.cell):
        print("REFUSED: cell %r is not granted (GRANT_j1f = %r, GRANT_j1n = %r; default deny)" % (
            a.cell, GRANT_j1f, GRANT_j1n), file=sys.stderr)
        return 20
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if a.ncu is not None and not os.access(a.ncu, os.X_OK):
        print("REFUSED: --ncu %s is not executable" % a.ncu, file=sys.stderr)
        return 20
    import torch
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device", file=sys.stderr)
        return 20
    os.makedirs(a.out)
    if a.mode == "ncu-cell":
        return mode_ncu_cell(a)
    gates, raw = [], {}
    t_run = time.perf_counter()
    with torch.inference_mode():
        for C, B in sorted(a.shapes.items(), key=lambda kv: -kv[0]):
            name = "C%d_B%d" % (C, B)
            t0 = time.perf_counter()
            try:
                g, r = run_shape(a, C, B, t0)
                gates.extend(g)
                raw[name] = r
                gates.append(("%s-RUN" % name, True, r["stop"] is None, "%d variants x %d passes of %d calls" % (
                    len(r["times_ms"]), a.reps, a.layers)))
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % name, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
                break
    rw = time.perf_counter() - t_run
    gates.append(("RUN-WALL", True, rw <= a.max_run_s, "run wall %.1f s (limit %.0f)" % (rw, a.max_run_s)))
    if a.ncu is not None:                                  # after the timing (its wall is not in RUN-WALL)
        for name, r in raw.items():
            r["ncu"] = run_ncu(a, r["C"], r["B"], gates)
    meta = dict(cell=a.cell, grant={"j1f": GRANT_j1f, "j1n": GRANT_j1n}.get(a.cell), shapes=a.shapes, segments=a.segments, layers=a.layers, reps=a.reps,
                warmup=a.warmup, seed=a.seed, sample=a.sample, fork_launch=FORK_LAUNCH, splits=SPLITS, union=UNION,
                row_shift=ROW_SHIFT, block_bytes=BLOCK_BYTES, tol=TOL, gpu=torch.cuda.get_device_properties(0).name,
                torch=torch.__version__, base_commit=os.environ.get("BASE_COMMIT"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, shapes=raw, gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3])
                                                     for g in gates]), f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-28s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for name, r in raw.items():
        s = r["summary"]
        for v, p in s["per"].items():
            print("J1F %s %-9s %.4f ms/call (%.3f ms/pass) rows %d logical %.3f GB/call %s TB/s valid %s" % (
                name, v, p["ms_call"], p["ms_pass"], p["rows"], p["bytes"] / 1e9, p["logical_tb_s"], p["valid"]))
        for line in decision_lines(name, s):
            print(line)
        for arm, n in (r.get("ncu") or {}).items():
            print("J1F %s NCU %-7s %s" % (name, arm, n.get("summary") or n))
    failed = [g[0] for g in gates if g[1] and not g[2]]
    print("r2_j1_forkbench: %d MUST gate(s) failed" % len(failed))
    return min(len(failed), 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
