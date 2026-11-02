"""Rung J1 cell j1f (AREA 1; KS/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md section 4; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE FORK'S RATE, MATCHED. The ledger's 0.885 vs 1.53 TB/s compares the fork (4 rows,
union pages, per-row masks, no split) with the draft's flash call A (1 row, its own window, split-KV): UNMATCHED
(LEDGER_rev7.md 2.7). This cell times the shipped fork and the shipped one-row flash call on the SAME synthetic pages
of the MEASURED shapes, in matched pairs:
  FORK     the shipped fork (fia6's blob, verify_rows' FORK_LAUNCH: 4 warps, 3 stages, tile 64, the bf16 CIS bias), 4 rows
           per request, each row 64 pages shifted (0, 3, 5, 8): union 72 per (KV head, request) (MEASURED: 72.2 at C 128,
           r7pa; 72.24 at C 73, V1B); the union built in the kernel (production)
  FORKT    the same with the union tables built before (BUILD_UNION off): the in-kernel build's cost
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
So kernel-only = X, copy-only = X_CO, total (j1f form) = X_CP; the rest X_CP - X - X_CO = the wrapper's allocations.
MATCHED PAIRS (the summary's ratios): FORK / FLASHU1 (equal logical bytes: the cost of 4 rows' tile at the same bytes);
FORK1R / NS1R1 (equal rows and bytes: the fork kernel's per-byte efficiency); FORK / NS1x4 (equal rows: the saving of
the shared read); FORKT / FORK (the union build); FORK3Dn / FORK (the split); NSRU / FORK. The same pairs against the
<X>_CP variants reproduce the j1f form. j1f's per-byte ratios need this rerun to be re-scored.
LOGICAL BYTES per call: pages x BLOCK_BYTES (K + V + bias of one (KV head, block)) x KV heads x requests (or entries);
achieved logical TB/s = bytes / time. Logical, not DRAM counters: an ncu pass is a separate option of the lead.
GATES (MUST unless noted): <shape>-RUN; <shape>-UNION (FORK union_len 72, FORK1R 64 everywhere); <shape>-<v>-EXACT
(every variant's output after its last timed pass = its first output, bit for bit, AND both hold only finite values:
identical NaN bits fail); <shape>-FORK3Dn-PATH (REPORT: the 3D launch wrote its NaN-filled segment maxima; if not, the
variant is INVALID); <shape>-FORKT-EQ (FORKT = FORK bit for bit); <shape>-<X>-OUTEQ (the direct out_ write = the j1f
form X_CP, bit for bit); <shape>-<v>-NUM (every checked value of the output, of the float64 reference and of their
difference is finite (torch.isfinite, before any accumulation; a FAIL names the count and the first index), and max
|output - float64 reference| <= TOL on --sample requests, every row and head; FORK3Dn: REPORT); <shape>-BUDGET;
RUN-WALL. A variant whose NUM, EXACT or OUTEQ fails is reported INVALID in the ratios (the decision lines say INVALID).
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
FLASH = ("NS1R1", "NS1x4", "FLASHU1", "NSRU")                 # the flash controls (direct out_ write)
BASE = ("FORK", "FORKT", "FORK1R") + tuple(x + s for x in FLASH for s in ("", "_CP", "_CO"))   # _CP before _CO


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
    a = ap.parse_args(argv)
    try:
        a.shapes = {int(c): int(b) for c, b in (x.split(":") for x in a.shapes.split(",") if x)}
        a.segments = tuple(int(x) for x in a.segments.split(",") if x)
    except ValueError:
        ap.error("--shapes C:B pairs, --segments integers")
    if not a.shapes or any(b < 1 or b > 1024 for b in a.shapes.values()) or any(s < 2 or s > 64 for s in a.segments):
        ap.error("--shapes B in 1 .. 1024; --segments in 2 .. 64")
    if a.layers < 1 or a.reps < 3 or a.warmup < 1 or a.sample < 1:
        ap.error("--layers >= 1, --reps >= 3, --warmup >= 1, --sample >= 1")
    return a


def granted(cell) -> bool:
    return cell == "j1f" and isinstance(GRANT_j1f, str) and len(GRANT_j1f) == 20 and GRANT_j1f.endswith("Z")


# ------------------------------------------------------------------ pure parts (CPU-tested)
def variants(segments):
    return BASE + tuple("FORK3D%d" % s for s in segments)


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
    if name.startswith("FORK3D") or name in ("FORK", "FORKT"):
        return dict(rows=R * B, pages=HKV * B * UNION)
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
         ("FORK1R", "NS1R1", "equal rows and bytes (1 row, 64 pages): the fork kernel's per-byte efficiency"),
         ("FORK", "NS1x4", "equal rows (4 per request): the shared read's saving vs the baseline's 4 calls"),
         ("FORKT", "FORK", "the in-kernel union build's cost"),
         ("NSRU", "FORK", "the shipped kernel's own sharing (L2) vs the fork"))
PAIRS_CP = (("FORK", "FLASHU1_CP", "j1f form (flash: wrapper + copy_): equal logical bytes"),
            ("FORK1R", "NS1R1_CP", "j1f form (flash: wrapper + copy_): equal rows and bytes"),
            ("FORK", "NS1x4_CP", "j1f form (flash: wrapper + copy_): equal rows"),
            ("NSRU_CP", "FORK", "j1f form (flash: wrapper + copy_): the shipped kernel's own sharing vs the fork"))


def summarize(times: dict, B: int, layers: int, valid: dict, segments) -> dict:
    """Per variant: median ms per pass and per call, logical bytes per call, logical TB/s; the matched ratios (INVALID
    where a member failed its NUM, EXACT or OUTEQ gate); the 3D split vs FORK; the decomposition of FORK - FLASHU1;
    per flash control: kernel-only (X), copy-only (X_CO), total in the j1f form (X_CP) and the rest."""
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
                                              "INVALID: a member failed NUM, EXACT or OUTEQ")
    out = dict(stats_ms=st, per=per, ratios=ratios)
    if "FORK" in per and "FLASHU1" in per and per["FORK"]["valid"] and per["FLASHU1"]["valid"]:
        f, u = per["FORK"]["ms_call"], per["FLASHU1"]["ms_call"]
        out["fork_minus_flashu1_ms_per_pass"] = round((f - u) * layers, 3)
    ft = {}
    for x in FLASH:
        if x in per and x + "_CP" in per and x + "_CO" in per:
            k_, cp, co = per[x]["ms_pass"], per[x + "_CP"]["ms_pass"], per[x + "_CO"]["ms_pass"]
            ft[x] = dict(kernel_ms_pass=k_, copy_only_ms_pass=co, total_j1f_form_ms_pass=cp,
                         rest_ms_pass=round(cp - k_ - co, 4),
                         valid=per[x]["valid"] and per[x + "_CP"]["valid"] and per[x + "_CO"]["valid"])
    out["flash_timing"] = ft
    d3 = [(per[k]["ms_call"], k) for k in per if k.startswith("FORK3D") and per[k]["valid"]]
    if d3 and "FORK" in per:
        best = min(d3)
        out["best_3d"] = dict(variant=best[1], ratio=round(best[0] / per["FORK"]["ms_call"], 4),
                              saving_ms_per_pass=round((per["FORK"]["ms_call"] - best[0]) * layers, 3))
    return out


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

    def fork_call(qx, out, cu, rows, tab, ext, ln, lists, extra=None):
        def call():
            FK.unified_attention(qx, ak, av, out, cu, rows, seq, BS, scale, True, (-1, -1), tab, 0, None, None, None,
                                 union_ext=ext, union_len=ln, cis_bias=ab, build_union=lists, rows_per_q_block=rows,
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
    for x in FLASH:                                                          # the direct write = the j1f form
        if x in first and x + "_CP" in first:
            eq = bool((bits(first[x]) == bits(first[x + "_CP"])).all())
            n_d, n_cp = nonfinite(first[x])[0], nonfinite(first[x + "_CP"])[0]
            ok = eq and n_d == 0 and n_cp == 0
            valid[x] = valid[x] and ok
            gates.append(("%s-%s-OUTEQ" % (name, x), True, ok, "direct out_ write = %s_CP (wrapper + copy_) bit for "
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


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if not granted(a.cell):
        print("REFUSED: cell %r is not granted (GRANT_j1f = %r; default deny)" % (a.cell, GRANT_j1f), file=sys.stderr)
        return 20
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    import torch
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device", file=sys.stderr)
        return 20
    os.makedirs(a.out)
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
    meta = dict(cell=a.cell, grant=GRANT_j1f, shapes=a.shapes, segments=a.segments, layers=a.layers, reps=a.reps,
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
        for k, v in s["ratios"].items():
            print("J1F %s RATIO %s = %s (%s)%s" % (name, k, v["ratio"], v["what"], "" if v["valid"] else " INVALID"))
        for x, t in s["flash_timing"].items():
            print("J1F %s FLASHT %-7s kernel-only %.3f, copy-only %.3f, total (j1f form) %.3f, rest %.3f ms per pass%s"
                  % (name, x, t["kernel_ms_pass"], t["copy_only_ms_pass"], t["total_j1f_form_ms_pass"],
                     t["rest_ms_pass"], "" if t["valid"] else " INVALID"))
        print("J1F %s FORK - FLASHU1 = %s ms per pass; best 3D %s" % (name, s.get("fork_minus_flashu1_ms_per_pass"),
                                                                       s.get("best_3d")))
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
