"""Copy-engine confirmation and transport-side attribution for the CPU-packing timeline stage. Pure python (sqlite3), no
GPU; CPU-tested on a synthetic nsys-like schema (retroinfer-eval tests/test_cpupack_transport.py). Launch starvation of the
decode's main stream is launch_timeline.py (unchanged, dc818e4); this module adds what that one does not read.

PER TRACED REP (the NVTX range 'lt|b<B>|<arm>|step<it>|<phase>|rep<k>' of cpupack_transport.py's stage TIMELINE):
  memcpy rows whose launch API call lies inside the range, from CUPTI_ACTIVITY_KIND_MEMCPY with copyKind / srcKind /
  dstKind / bytes (names from ENUM_CUDA_MEMCPY_OPER / ENUM_CUDA_MEM_KIND when the export has them):
    h2d      HtoD rows: count, bytes, per-row GB/s, the streams they ran on, every row pinned -> device?
    d2h      DtoH rows (the plan-list delivery): count, bytes
  copy_streams_kernels  kernels that ran on the H2D streams (0 = the DMA is on the copy engine, no SM copy kernel)
  scatter   kernels whose name contains 'index' on streams other than the main stream (index_copy_ = the placement),
            count and busy ms; 'flash_h2d_persistent' kernels = the W8 gather.
  COPY-ENGINE PROOF (rep): >= 1 HtoD row, every HtoD row pinned -> device, no kernel on an HtoD stream.
    python cpupack_timeline.py OUT_PREFIX run1.sqlite [run2.sqlite ...]
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from typing import Dict, List, Optional

PINNED_NAMES = ("pinned",)


def _tables(con) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(con, t) -> set:
    return {r[1] for r in con.execute("PRAGMA table_info(%s)" % t)}


def _enum(con, tabs, name) -> Dict[int, str]:
    if name not in tabs:
        return {}
    cols = _cols(con, name)
    lab = "label" if "label" in cols else "name"
    return {int(i): str(n) for i, n in con.execute("SELECT id, %s FROM %s" % (lab, name))}


def load(path: str) -> Dict:
    con = sqlite3.connect(path)
    tabs = _tables(con)
    S = dict(con.execute("SELECT id, value FROM StringIds")) if "StringIds" in tabs else {}
    api = {}
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tabs:
            for s, e, c in con.execute("SELECT start, end, correlationId FROM %s" % t):
                if c is not None:
                    api[int(c)] = (int(s), int(e))
    kinds = _enum(con, tabs, "ENUM_CUDA_MEMCPY_OPER")
    mem = _enum(con, tabs, "ENUM_CUDA_MEM_KIND")
    memcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
        cols = _cols(con, "CUPTI_ACTIVITY_KIND_MEMCPY")
        sel = ["start", "end", "streamId", "correlationId"] + [c if c in cols else "NULL" for c in ("bytes", "copyKind", "srcKind", "dstKind")]
        for s, e, st, c, nb, ck, sk, dk in con.execute("SELECT %s FROM CUPTI_ACTIVITY_KIND_MEMCPY" % ", ".join(sel)):
            memcpy.append(dict(start=int(s), end=int(e), stream=st, corr=c, bytes=nb, copyKind=ck, srcKind=sk, dstKind=dk,
                               copy=kinds.get(ck, str(ck)), src=mem.get(sk, str(sk)), dst=mem.get(dk, str(dk))))
    kern = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
        cols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        nm = "demangledName" if "demangledName" in cols else "shortName"
        for s, e, st, c, n in con.execute("SELECT start, end, streamId, correlationId, %s FROM CUPTI_ACTIVITY_KIND_KERNEL" % nm):
            kern.append(dict(start=int(s), end=int(e), stream=st, corr=c, name=S.get(n, str(n))))
    ranges = []
    if "NVTX_EVENTS" in tabs:
        cols = _cols(con, "NVTX_EVENTS")
        tid = "textId" if "textId" in cols else "NULL"
        for s, e, text, text_id in con.execute("SELECT start, end, text, %s FROM NVTX_EVENTS WHERE end IS NOT NULL" % tid):
            t = text if text is not None else S.get(text_id)
            if t and t.startswith("lt|"):
                ranges.append((int(s), int(e), t))
    con.close()
    return dict(api=api, memcpy=memcpy, kernels=kern, ranges=sorted(ranges))


def _norm(x) -> str:
    return str(x or "").lower().replace("-", " ").replace("_", " ")


def is_h2d(r) -> bool:
    c = _norm(r.get("copy"))
    return "htod" in c or "host to device" in c or r.get("copyKind") == 1


def is_d2h(r) -> bool:
    c = _norm(r.get("copy"))
    return "dtoh" in c or "device to host" in c or r.get("copyKind") == 2


def is_pinned_to_device(r) -> bool:
    s, d = str(r.get("src", "")).lower(), str(r.get("dst", "")).lower()
    return any(p in s for p in PINNED_NAMES) and "device" in d


def analyse(data: Dict, main_stream_of: Optional[Dict[str, int]] = None) -> List[Dict]:
    out = []
    for s0, e0, text in data["ranges"]:
        def inside(corr):
            a = data["api"].get(corr)
            return a is not None and s0 <= a[0] <= e0
        mc = [r for r in data["memcpy"] if inside(r["corr"])]
        ks = [k for k in data["kernels"] if inside(k["corr"])]
        h2d = [r for r in mc if is_h2d(r)]
        d2h = [r for r in mc if is_d2h(r)]
        h2d_streams = sorted({r["stream"] for r in h2d})
        k_on_copy = [k for k in ks if k["stream"] in set(h2d_streams)]
        main = (main_stream_of or {}).get(text)
        if main is None:
            gates = [k for k in ks if "sleep" in k["name"].lower() or "spin" in k["name"].lower()]
            main = min(gates, key=lambda k: k["start"])["stream"] if gates else None
        scat = [k for k in ks if "index" in k["name"].lower() and k["stream"] != main]
        w8 = [k for k in ks if "flash_h2d_persistent" in k["name"]]
        gbps = [r["bytes"] / (r["end"] - r["start"]) for r in h2d if r.get("bytes") and r["end"] > r["start"]]
        rec = dict(label=text, n_h2d=len(h2d), h2d_bytes=sum(r.get("bytes") or 0 for r in h2d), h2d_streams=h2d_streams,
                   h2d_all_pinned_to_device=bool(h2d) and all(is_pinned_to_device(r) for r in h2d),
                   h2d_gbps_median=(sorted(gbps)[len(gbps) // 2] if gbps else None), n_d2h=len(d2h),
                   d2h_bytes=sum(r.get("bytes") or 0 for r in d2h), kernels_on_h2d_streams=len(k_on_copy),
                   n_scatter_kernels=len(scat), scatter_busy_ms=sum(k["end"] - k["start"] for k in scat) * 1e-6,
                   n_w8_kernels=len(w8), main_stream=main)
        rec["copy_engine_proof"] = bool(h2d) and rec["h2d_all_pinned_to_device"] and not k_on_copy
        out.append(rec)
    return out


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = argv[1]
    sqls = [p for pat in argv[2:] for p in sorted(glob.glob(pat))]
    reps = []
    for sp in sqls:
        for r in analyse(load(sp)):
            r["sqlite"] = sp
            reps.append(r)
    cpu = [r for r in reps if "|cpu" in r["label"]]
    summary = dict(n_reps=len(reps), cpu_reps=len(cpu), cpu_reps_with_proof=sum(1 for r in cpu if r["copy_engine_proof"]))
    json.dump(dict(summary=summary, reps=reps), open(out + ".json", "w"), indent=1)
    print(json.dumps(summary))
    return 0 if reps else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
