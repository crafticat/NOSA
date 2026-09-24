"""Co-execution of the side transfer with the decode step, from an Nsight Systems trace (review B3: a DIRECT record
that the copy ran while the step's kernels ran, which interval coverage cannot show). Pure python (sqlite3), CPU-tested
(retroinfer-eval tests/test_hisparse_copy_plan.py).

INPUT: the SQLite export of an `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node` run of hisparse_repro.py with
HR_NVTX=1, so every concurrent rep sits inside an NVTX push/pop range 'gf|<arm>|<mode>|<regime>|<rep>' (the range spans
the whole bracket and its final synchronize, so every GPU activity of that rep lies inside it).
PER RANGE: GPU activities (kernels from CUPTI_ACTIVITY_KIND_KERNEL, copies from CUPTI_ACTIVITY_KIND_MEMCPY) that START
inside the range, classified
  side   kernels whose name contains a SIDE_PATTERNS entry (the HiSparse copy kernel, NOSI's persistent gather) and
         host-to-device memcpys (copyKind 1: the dma2d arm)
  gate   the bracket's GPU sleep (a name containing 'spin' or 'sleep'): excluded
  main   every other kernel (the decode step; in graph mode its nodes, traced with --cuda-graph-trace=node)
and reported as busy time of the UNION of each class's intervals:
  side_busy_ms, main_busy_ms, coexec_ms = |union(side) intersect union(main)|, main_span_ms = first main start -> last
  main end, coexec_frac_of_side = coexec / side_busy, side_busy_in_main_span_ms = side busy time inside the main span.
PER CELL (arm, mode, regime): medians over its reps. Output: <out>.json and <out>.md.
    python nsys_coexec.py trace.sqlite --out prefix
"""
import json
import sqlite3
import sys
from typing import Dict, List, Tuple

SIDE_PATTERNS = ("copy_cache_planned", "flash_h2d_persistent", "transfer_kernel_impl", "hicache_transfer_per_layer")
GATE_PATTERNS = ("spin", "sleep")
MEMCPY_HTOD = 1


def union(iv: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def length(iv: List[Tuple[int, int]]) -> int:
    return sum(e - s for s, e in iv)


def intersect(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Both inputs are sorted disjoint unions."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def classify(name: str) -> str:
    n = (name or "").lower()
    if any(p in n for p in SIDE_PATTERNS):
        return "side"
    if any(p in n for p in GATE_PATTERNS):
        return "gate"
    return "main"


def range_stats(side: List[Tuple[int, int]], main: List[Tuple[int, int]]) -> Dict:
    us, um = union(side), union(main)
    co = intersect(us, um)
    span = (um[0][0], um[-1][1]) if um else None
    in_span = intersect(us, [span]) if span else []
    ms = 1e-6
    return dict(side_busy_ms=length(us) * ms, main_busy_ms=length(um) * ms, coexec_ms=length(co) * ms,
                main_span_ms=((span[1] - span[0]) * ms if span else 0.0), side_busy_in_main_span_ms=length(in_span) * ms,
                coexec_frac_of_side=(length(co) / length(us) if us else float("nan")), n_side=len(side), n_main=len(main))


def _tables(con) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(con, table: str) -> set:
    return {r[1] for r in con.execute("PRAGMA table_info(%s)" % table)}


def load(path: str) -> Dict:
    """-> dict(ranges=[(start, end, text)], kernels=[(start, end, name)], memcpy=[(start, end, copyKind)])."""
    con = sqlite3.connect(path)
    tabs = _tables(con)
    strings = {}
    if "StringIds" in tabs:
        strings = dict(con.execute("SELECT id, value FROM StringIds"))
    ranges = []
    if "NVTX_EVENTS" in tabs:
        cols = _cols(con, "NVTX_EVENTS")
        tid = "textId" if "textId" in cols else "NULL"
        for s, e, text, text_id in con.execute("SELECT start, end, text, %s FROM NVTX_EVENTS WHERE end IS NOT NULL" % tid):
            t = text if text is not None else strings.get(text_id)
            if t and t.startswith("gf|"):
                ranges.append((int(s), int(e), t))
    kernels = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
        cols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_col = "demangledName" if "demangledName" in cols else ("shortName" if "shortName" in cols else None)
        for s, e, nid in con.execute("SELECT start, end, %s FROM CUPTI_ACTIVITY_KIND_KERNEL" % (name_col or "NULL")):
            kernels.append((int(s), int(e), strings.get(nid, str(nid)) if isinstance(nid, int) else str(nid)))
    memcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
        for s, e, kind in con.execute("SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            memcpy.append((int(s), int(e), int(kind)))
    con.close()
    return dict(ranges=sorted(ranges), kernels=sorted(kernels), memcpy=sorted(memcpy))


def analyse(data: Dict) -> Dict:
    per_range = []
    for rs, re_, text in data["ranges"]:
        side, main = [], []
        for s, e, name in data["kernels"]:
            if rs <= s < re_:
                c = classify(name)
                if c == "side":
                    side.append((s, e))
                elif c == "main":
                    main.append((s, e))
        for s, e, kind in data["memcpy"]:
            if rs <= s < re_ and kind == MEMCPY_HTOD:
                side.append((s, e))
        parts = text.split("|")
        rec = dict(range=text, arm=parts[1] if len(parts) > 1 else "?", mode=parts[2] if len(parts) > 2 else "?",
                   regime=parts[3] if len(parts) > 3 else "?", rep=parts[4] if len(parts) > 4 else "?")
        rec.update(range_stats(side, main))
        per_range.append(rec)
    cells: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in per_range:
        cells.setdefault((r["arm"], r["mode"], r["regime"]), []).append(r)

    def med(xs):
        xs = sorted(x for x in xs if x == x)
        if not xs:
            return float("nan")
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    summary = []
    for (arm, mode, regime), rs in sorted(cells.items()):
        summary.append(dict(arm=arm, mode=mode, regime=regime, reps=len(rs),
                            **{k: med([r[k] for r in rs]) for k in ("side_busy_ms", "main_busy_ms", "coexec_ms", "main_span_ms",
                                                                     "side_busy_in_main_span_ms", "coexec_frac_of_side")}))
    return dict(per_range=per_range, cells=summary, side_patterns=list(SIDE_PATTERNS), gate_patterns=list(GATE_PATTERNS),
                n_ranges=len(data["ranges"]), n_kernels=len(data["kernels"]), n_memcpy=len(data["memcpy"]))


def render(res: Dict) -> str:
    L = ["# Side-transfer / decode co-execution from the nsys trace (medians over reps)", "",
         "%d NVTX cell ranges, %d kernels, %d memcpys. side = %s kernels + host-to-device memcpys; the gate's sleep kernel is excluded." % (
             res["n_ranges"], res["n_kernels"], res["n_memcpy"], ", ".join(res["side_patterns"])), "",
         "| arm | mode | regime | reps | main span ms | main busy ms | side busy ms | side busy inside main span ms | co-executing ms | co-executing / side busy |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        L.append("| %s | %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            c["arm"], c["mode"], c["regime"], c["reps"], c["main_span_ms"], c["main_busy_ms"], c["side_busy_ms"], c["side_busy_in_main_span_ms"],
            c["coexec_ms"], c["coexec_frac_of_side"]))
    return "\n".join(L) + "\n"


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = argv[argv.index("--out") + 1] if "--out" in argv else argv[1].rsplit(".", 1)[0] + ".coexec"
    res = analyse(load(argv[1]))
    json.dump(res, open(out + ".json", "w"), indent=1)
    text = render(res)
    open(out + ".md", "w").write(text)
    print(text)
    return 0 if res["n_ranges"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
