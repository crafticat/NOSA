"""VERIFICATION ALONE: the ordinary NOSA decode step with the miss volume
CONTROLLED (retroinfer-eval fork; REPRODUCE.md ledger 'SCOPE CORRECTION,
2026-09-20 (author): MEASURE VERIFICATION ALONE FIRST' -- the contract).

One-position verification = the ordinary decode step itself (same model
computation, target selection, precision, attention kernel and split, batch,
context, restored state). What is measured is how its cost moves with the
number of blocks per stream that the step must fetch over PCIe before it can
attend. Teacher-forced NOSA-8B on PG-19 (the verify_pilot.py pattern): prefill
of an L-token prompt, N decode steps fed the document's own continuation; the
gated steps WARM .. N-1 are measured.

PER GATED STEP t, from ONE restored state R (state_snapshot.CacheSnapshot,
the full snapshot, taken once per gated step):
  shipped_nat  the plain shipped decode from R, timed (its natural load count,
               its natural slot layout). Its logits against the reference are
               an OBSERVATION of layout invariance (see below), never a control.
  reference    R with the block map FLUSHED (every non-tail slot empty, the
               engine's own post-prefill state), then the ordinary step: it
               fetches the whole selection (63 blocks per stream) and leaves the
               map in the layout that is a FIXED POINT of "evict any set of
               slots, then diff" (miss_control.py's module docstring). Its
               post-step map is T_l (miss_control.record_targets) and its
               logits are THE REFERENCE. Its selection must equal shipped_nat's
               (asserted per stream: the step's recomputed selection after a
               restore is the same set) -- and it is, as a set, exactly the
               selection the natural step attended; only the slots differ.
  A  (D = 0)   R -> flush -> prewarm(T_l) through the engine's own diff +
               gathers (untimed) -> the ordinary step, TIMED: 0 loads asserted,
               map == T_l after, logits torch.equal to the reference.
  B(D, pat)    R -> flush -> prewarm(T_l) -> evict D slots per stream with the
               pattern (map entries pointed outside T_l, rows NaN-poisoned) ->
               the ordinary step, TIMED: loads == D x streams asserted (from the
               engines' own load masks, cross-checked against the trace's
               archive), map == T_l after, logits torch.equal, no NaN.
  shipped      R -> flush -> prewarm(M*) where M* holds the natural pre-step
               resident set in T_l-compatible slots (miss_control.
               monotone_prestep_map) -> the ordinary step, TIMED: it fetches
               exactly the natural miss set (asserted), ends in T_l, logits
               torch.equal. This is the shipped decode's cost at the controlled
               layout, so that every gated comparison is within ONE layout.
  advance      the ordinary step from R (the natural trajectory continues);
               its logits torch.equal to shipped_nat's = the restore is exact
               (hygiene, counted as a failure when it is not).

TIMED RUNS ARE UNINSTRUMENTED apart from two CUDA events around
decode_inference (call_ms) and the transfer trace's own per-layer events (mode
"1": diff / gather / attention brackets, the load-mask archive). The trace's
per-step logits clone is disabled (AloneTrace). In mode `profile` a chosen D is
repeated under torch.profiler with record_function ranges 'diff', 'gather',
'attention', 'rest' at the trace's bracket positions (ProfileTrace) and the
call wrapped in 'decode_call'; the chrome trace + timeline_manifest.json are
what scripts/verify_timeline_report.py reads (--call-ranges decode_call
--brackets diff,gather,attention,rest). Mode `reference` is the plain decode in
a bare process (no snapshot, no arms): the harness's own overhead check, its
per-step logits hashes compared to the points process's advance steps by the
table. Mode `path1` (optional, off by default in the sbatch) times Path 1
verify_inference at U = 1 as the diagnostic third arm; it is NOT the definition
of verification here.

Modes (NOSI_ALONE_MODE): points | reference | profile | path1 | table.
Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_ALONE_OUT, NOSI_ALONE_TAG,
NOSI_ALONE_L (16128), NOSI_ALONE_N (12), NOSI_ALONE_WARM (4), NOSI_ALONE_B
(batch = requests per document batch, 128), NOSI_ALONE_DOCS (document batches,
2: batch j cycles the qualifying PG-19 documents from offset j x B),
NOSI_ALONE_D ("0 1 2 4 8 16 32 63", per stream), NOSI_ALONE_FRAG_D (8),
NOSI_ALONE_FRAG ("layers4 adjacent scattered"), NOSI_ALONE_SEED (0),
NOSI_ALONE_PROFILE_D ("4 32"), NOSI_ALONE_PROFILE_STEPS (2),
NOSI_ALONE_CEILING_GBPS (25.1: the device's matched pinned-H2D ceiling, the
floor probe's asymptote), NOSI_ALONE_BATCHES ("128 64", table: the requested
cells), NOSI_ATTN_SPLITS / NOSI_POOL_BLOCKS / NOSI_VERIFY_ROUND_SLOTS (read by
the engine at import: 4 / 0 / 0 in the sbatch; path1 needs ROUND_SLOTS > 0).
Outputs under NOSI_ALONE_OUT: points_<tag>.json, reference_<tag>.json,
profile_<tag>.json + timeline_*.json.gz + timeline_manifest.json,
path1_<tag>.json; the table writes points.json and points.md.
"""
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time

import torch

MODE = os.environ.get("NOSI_ALONE_MODE", "table")
OUT = os.environ.get("NOSI_ALONE_OUT", "nosi_verify_alone")
TAG = os.environ.get("NOSI_ALONE_TAG", MODE)
L = int(os.environ.get("NOSI_ALONE_L", "16128"))
N = int(os.environ.get("NOSI_ALONE_N", "12"))
WARM = int(os.environ.get("NOSI_ALONE_WARM", "4"))
BATCH = int(os.environ.get("NOSI_ALONE_B", "128"))
NDOCS = int(os.environ.get("NOSI_ALONE_DOCS", "2"))
D_LIST = tuple(int(x) for x in os.environ.get("NOSI_ALONE_D", "0 1 2 4 8 16 32 63").split())
FRAG_D = int(os.environ.get("NOSI_ALONE_FRAG_D", "8"))
FRAG = tuple(os.environ.get("NOSI_ALONE_FRAG", "layers4 adjacent scattered").split())
SEED = int(os.environ.get("NOSI_ALONE_SEED", "0"))
PROFILE_D = tuple(int(x) for x in os.environ.get("NOSI_ALONE_PROFILE_D", "4 32").split())
PROFILE_STEPS = int(os.environ.get("NOSI_ALONE_PROFILE_STEPS", "2"))
CEILING_GBPS = float(os.environ.get("NOSI_ALONE_CEILING_GBPS", "25.1"))
BATCHES = tuple(int(x) for x in os.environ.get("NOSI_ALONE_BATCHES", "128 64").split())
os.makedirs(OUT, exist_ok=True)

BLOCK_BYTES = 32768                 # K + V per loaded (layer, head, request, slot)
ROW_BYTES_ATTN = 65792              # K + V + bias of one 64-token block for BOTH KV heads: 64 x 2 x (256 + 256 + 2)
KV_HEADS, NUM_LAYERS, WINDOW = 2, 32, 64
BRACKETS = ("diff", "gather", "attention", "rest")
CALL_RANGE = "decode_call"
CHECKS = ("loads_ok", "logits_equal", "map_ok", "prewarm_ok", "evict_ok", "nan_free")

# REGISTERED PREDICTIONS (the ledger entry, stated before the run; 4 splits, fix A, 16128 x 128).
REGISTERED = dict(
    batch=128, L=16128,
    resident_ms=40.0,                    # cap-63 window, 0 loads: 80.3 - 41.7 + 1.7
    fixed_ms=(1.0, 2.0),                 # gather issue + early-exit scheduling (floor probe)
    bw_small_gbps=18.0,                  # ~1 MB per launch
    bw_large_gbps=(24.0, 25.0),          # >= 16 MB per launch; 25.1 asymptote, 25.2 bulk
    T_ms={1: (51.0, 52.0), 4: (83.0, 83.0), 16: (215.0, 215.0), 63: (715.0, 715.0)},
    bw_at_D_ge_4="within 10% of 25 GB/s",
    layers4_vs_spread="layer-concentrated faster than spread by the floor-probe difference (~5-10% of the transfer time)",
    adjacent_vs_scattered="within 5%",
    falsifiers=("residual after subtracting T_resident and bytes / 25.1 GB/s exceeding 3 ms/step at any D (the excess must be named by operation from the timeline)",
                "achieved BW below 20 GB/s at D >= 4",
                "any logits mismatch"),
)
RESIDUAL_LIMIT_MS = 3.0
BW_LIMIT_GBPS = 20.0
FIXED_MS_FORMULA = 1.5


def die(msg):
    print("[verify_alone] REFUSED: " + msg, flush=True)
    sys.exit(2)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def check_budget():
    if not (1 <= WARM < N):
        die("need 1 <= WARM=%d < N=%d (the warm-up decode step precedes every gated step)" % (WARM, N))
    if (L % 64) + N >= 64:
        die("L%%64 + N = %d >= 64: a step would fill the tail block (record_targets refuses a rollover); pick L, N with (L %% 64) + N < 64" % ((L % 64) + N))
    for D in D_LIST + (FRAG_D,):
        if not (0 <= D <= WINDOW - 1):
            die("D=%d outside 0..%d" % (D, WINDOW - 1))
    if "adjacent" in FRAG and FRAG_D > 16:
        die("adjacent needs FRAG_D <= 16 (the local window present in every stream)")


def load_corpus(path, batch: int):
    """The qualifying PG-19 test documents (>= L+N+1 tokens), tokenized ONCE per
    process: (ids list, doc rows). Stops once `batch` documents qualify; a batch of
    128 requests scans the whole test split (~100 qualifying books) and cycles."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("parquet", data_files=os.environ["NOSI_PG19_PARQUET"])["train"]["text"]
    ids, rows = [], []
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < L + N + 1:
            continue
        rows.append(i)
        ids.append(t[0, :L + N + 1])
        if len(rows) == batch:
            break
    assert rows, "no document with >= %d tokens" % (L + N + 1)
    return ids, rows


def pick_batch(corpus, batch: int, offset: int):
    """(ids (batch, L+N+1), doc rows, distinct): document batch j = the qualifying
    documents cycled from offset j x batch (CPU-tested)."""
    ids, rows = corpus
    distinct = len(rows)
    pick = [(offset + i) % distinct for i in range(batch)]
    return torch.stack([ids[p] for p in pick]), [rows[p] for p in pick], distinct


# ---------------------------------------------------------------------------
# pure bookkeeping (CPU-tested in retroinfer-eval tests/test_nosi_verify_alone.py)
# ---------------------------------------------------------------------------
def row_plan(d_list, frag_d, frag) -> list:
    """[(D, pattern)] of one gated step: A (D = 0, 'spread') first, the D sweep, the fragmentation rows."""
    plan = [(0, "spread")] + [(int(D), "spread") for D in d_list if int(D) > 0]
    plan += [(int(frag_d), p) for p in frag]
    seen, out = set(), []
    for p in plan:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def logical_unique_kv_bytes(B: int, layers: int = NUM_LAYERS, heads: int = KV_HEADS, window: int = WINDOW) -> int:
    """Distinct (layer, head, request, block) entries attended by one step x 32 KiB."""
    return window * heads * B * layers * BLOCK_BYTES


def hbm_traffic_bytes(wire_bytes: int, B: int, layers: int = NUM_LAYERS, heads: int = KV_HEADS, window: int = WINDOW) -> dict:
    """gather writes (= the wire bytes) + attention reads (64 blocks x B rows x 65,792 B x layers)
    + map copies (diff reads 2 maps and writes 2, the copy_ reads 1 and writes 1: 6 x H x B x 64 x 8 B per layer)."""
    attn = window * B * ROW_BYTES_ATTN * layers
    maps = 6 * heads * B * window * 8 * layers
    return dict(gather_write_bytes=int(wire_bytes), attention_read_bytes=int(attn), map_copy_bytes=int(maps),
                hbm_traffic_bytes=int(wire_bytes) + int(attn) + int(maps))


def predicted_ms(bytes_wire: float, resident_ms: float, ceiling_gbps: float = CEILING_GBPS, fixed_ms: float = FIXED_MS_FORMULA) -> float:
    """The hypothesis under test: T(D) = T_resident + fixed + bytes / ceiling."""
    return resident_ms + fixed_ms + bytes_wire / (ceiling_gbps * 1e9) * 1e3


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def aggregate(rows, ceiling_gbps: float = CEILING_GBPS) -> list:
    """Per (batch, L, arm, D, pattern): means over the gated steps with the FIRST gated
    step of every document batch excluded (JIT, first-touch), std of call_ms, bytes from
    the measured loads, achieved GB/s = bytes / gather bracket, resident_ms = the same
    (batch, L)'s arm A, residual_ms = call_ms - resident_ms - bytes / ceiling, the HBM
    columns, and the correctness verdict over ALL rows of the group (the first step
    included: correctness is gated on every row)."""
    first = {}
    for r in rows:
        k = (r["batch"], r["L"], r["doc"])
        first[k] = min(first.get(k, r["step"]), r["step"])
    groups = {}
    for r in rows:
        groups.setdefault((r["batch"], r["L"], r["arm"], r["D"], r["pattern"]), []).append(r)
    cells = []
    for (batch, Lv, arm, D, pattern), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        timed = [r for r in rs if r["step"] != first[(r["batch"], r["L"], r["doc"])]] if len(rs) > 1 else []
        fails = sum(int(r.get("fails", 0)) for r in rs)
        cell = dict(batch=batch, L=Lv, arm=arm, D=D, pattern=pattern, n=len(timed), n_rows=len(rs), fails=fails,
                    verdict="PASS" if fails == 0 else "FAIL(%d)" % fails, checks={c: sum(1 for r in rs if r.get(c) is False) for c in CHECKS},
                    layer_plan=rs[0].get("layer_plan"), layout=rs[0].get("layout"), profiled=bool(rs[0].get("profiled", False)))
        if timed:
            def mean(k):
                return _mean([float(r[k]) for r in timed])
            cell.update(call_ms=mean("call_ms"), call_ms_std=_std([float(r["call_ms"]) for r in timed]), call_ms_min=min(float(r["call_ms"]) for r in timed),
                        step_ms=mean("step_ms") if all("step_ms" in r for r in timed) else None,
                        gather_ms=mean("gather_ms"), diff_ms=mean("diff_ms"), attn_ms=mean("attn_ms"),
                        blocks_loaded=mean("blocks_loaded"), blocks_per_stream=mean("blocks_loaded") / (KV_HEADS * NUM_LAYERS * batch))
            cell["bytes"] = cell["blocks_loaded"] * BLOCK_BYTES
            cell["achieved_gbps"] = (cell["bytes"] / cell["gather_ms"] * 1e-6) if cell["gather_ms"] > 0 and cell["bytes"] > 0 else 0.0
            cell["logical_unique_kv_bytes"] = logical_unique_kv_bytes(batch)
            cell.update(hbm_traffic_bytes(cell["bytes"], batch))
            cell["max_abs_max"] = max(float(r.get("max_abs", 0.0)) for r in rs)
            cell["argmax_agree_min"] = min(float(r.get("argmax_agree", 1.0)) for r in rs)
        cells.append(cell)
    resident = {(c["batch"], c["L"]): c["call_ms"] for c in cells if c["arm"] == "A" and c.get("n", 0) > 0}
    for c in cells:
        rm = resident.get((c["batch"], c["L"]))
        c["resident_ms"] = rm
        if rm is not None and c.get("n", 0) > 0:
            c["residual_ms"] = c["call_ms"] - rm - c["bytes"] / (ceiling_gbps * 1e9) * 1e3
            c["predicted_formula_ms"] = predicted_ms(c["bytes"], rm, ceiling_gbps)
        else:
            c["residual_ms"] = None
            c["predicted_formula_ms"] = None
    return cells


def evaluate(cells) -> list:
    """The registered falsifiers and predictions against the aggregated cells: one dict per item."""
    out = []
    sweep = [c for c in cells if c["arm"] in ("A", "B") and c["pattern"] == "spread" and c.get("n", 0) > 0]
    for (batch, Lv) in sorted({(c["batch"], c["L"]) for c in cells}):
        mine = [c for c in sweep if c["batch"] == batch and c["L"] == Lv]
        if not mine:
            continue
        key = "[L=%d,B=%d]" % (Lv, batch)
        resid = [(c["D"], c["residual_ms"]) for c in mine if c["residual_ms"] is not None]
        bad = [(d, r) for d, r in resid if r > RESIDUAL_LIMIT_MS]
        out.append(dict(item="F1-residual" + key, verdict="REFUTED" if bad else ("PASS" if resid else "n/a"),
                        detail="residual_ms = call - resident - bytes/%.1f GB/s > %.0f ms at D in %s" % (CEILING_GBPS, RESIDUAL_LIMIT_MS, [d for d, _ in bad]) if bad
                        else ("max residual %.2f ms over D in %s" % (max(r for _, r in resid), [d for d, _ in resid]) if resid
                              else "no arm A row (D = 0) with a timed step: resident_ms and the residual are undefined")))
        lowbw = [(c["D"], c["achieved_gbps"]) for c in mine if c["D"] >= 4 and c["achieved_gbps"] < BW_LIMIT_GBPS]
        big = [c for c in mine if c["D"] >= 4]
        out.append(dict(item="F2-bandwidth" + key, verdict="REFUTED" if lowbw else ("PASS" if big else "n/a"),
                        detail=("achieved GB/s < %.0f at D in %s" % (BW_LIMIT_GBPS, [d for d, _ in lowbw])) if lowbw
                        else ("min achieved %.1f GB/s at D >= 4" % min(c["achieved_gbps"] for c in big) if big else "no D >= 4")))
        allc = [c for c in cells if c["batch"] == batch and c["L"] == Lv and c["arm"] in ("A", "B", "shipped")]
        nf = sum(c["fails"] for c in allc)
        out.append(dict(item="F3-logits" + key, verdict="PASS" if nf == 0 else "REFUTED",
                        detail="%d failed checks over %d controlled rows (torch.equal, load counts, map, NaN)" % (nf, sum(c["n_rows"] for c in allc))))
        if batch == REGISTERED["batch"] and Lv == REGISTERED["L"]:
            for D, (lo, hi) in sorted(REGISTERED["T_ms"].items()):
                c = next((c for c in mine if c["D"] == D), None)
                if c is not None:
                    out.append(dict(item="P-T(%d)%s" % (D, key), verdict="within 10%" if lo * 0.9 <= c["call_ms"] <= hi * 1.1 else "outside 10%",
                                    detail="registered %g-%g ms, measured %.1f +- %.1f ms" % (lo, hi, c["call_ms"], c["call_ms_std"])))
            a = next((c for c in mine if c["D"] == 0), None)
            if a is not None:
                out.append(dict(item="P-resident" + key, verdict="within 10%" if abs(a["call_ms"] - REGISTERED["resident_ms"]) <= 0.1 * REGISTERED["resident_ms"] else "outside 10%",
                                detail="registered ~%.0f ms, measured %.1f ms" % (REGISTERED["resident_ms"], a["call_ms"])))
        frag = {c["pattern"]: c for c in cells if c["batch"] == batch and c["L"] == Lv and c["arm"] == "B" and c["D"] == FRAG_D and c.get("n", 0) > 0}
        if "spread" in frag and "layers4" in frag:
            s, l4 = frag["spread"], frag["layers4"]
            out.append(dict(item="P-layers4-vs-spread" + key, verdict="faster" if l4["call_ms"] < s["call_ms"] else "not faster",
                            detail="spread %.1f ms (gather %.1f), layers4 %.1f ms (gather %.1f): %+.1f%% of call, %+.1f%% of the spread gather bracket; registered: layers4 faster by ~5-10%% of the transfer time"
                            % (s["call_ms"], s["gather_ms"], l4["call_ms"], l4["gather_ms"], 100 * (l4["call_ms"] - s["call_ms"]) / s["call_ms"],
                               100 * (l4["gather_ms"] - s["gather_ms"]) / s["gather_ms"] if s["gather_ms"] > 0 else float("nan"))))
        if "adjacent" in frag and "scattered" in frag:
            a, sc = frag["adjacent"], frag["scattered"]
            d = 100 * (a["call_ms"] - sc["call_ms"]) / sc["call_ms"]
            out.append(dict(item="P-adjacent-vs-scattered" + key, verdict="within 5%" if abs(d) <= 5 else "outside 5%",
                            detail="adjacent %.1f ms, scattered %.1f ms: %+.1f%% (gather %.1f vs %.1f ms)" % (a["call_ms"], sc["call_ms"], d, a["gather_ms"], sc["gather_ms"])))
    return out


def _f(x, fmt="%.1f"):
    return "-" if x is None else fmt % x


def render_points(cells, evaluation, extras=None) -> str:
    """points.md: the per-cell table, the fragmentation rows, the falsifiers, the registered
    predictions, the observations. An empty measurement renders '-' (never a number)."""
    extras = extras or {}
    out = ["# verification alone: the ordinary decode step at controlled miss volume", "",
           "call_ms = CUDA events around decode_inference (uninstrumented run; mean over the gated steps, first gated step of each document batch excluded; +- std). "
           "gather_ms / diff_ms / attn_ms = the transfer trace's per-layer brackets summed over the 32 layers. bytes = blocks_loaded x 32768 (measured from the engines' load masks). "
           "achieved GB/s = bytes / gather_ms. resident_ms = arm A (D = 0) of the same cell. residual_ms = call_ms - resident_ms - bytes / %.1f GB/s (the device's matched pinned-H2D ceiling). "
           "predicted = resident + %.1f + bytes / %.1f GB/s (the hypothesis). logical unique KV = 64 blocks x streams x 32 KiB; HBM traffic = gather writes + attention reads (64 x B x 65,792 B x 32) + map copies. "
           "verdict = every controlled row's checks: load count exact, map == T_l after the step, prewarm/evict invariants, logits torch.equal to the reference, no NaN." % (CEILING_GBPS, FIXED_MS_FORMULA, CEILING_GBPS), ""]
    hdr = ("| B | L | arm | D | pattern | n | call ms | std | gather ms | diff ms | attn ms | blocks/stream | bytes MB | GB/s | resident ms | residual ms | predicted ms | unique KV GB | HBM GB | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out += list(hdr)
    for c in cells:
        if c.get("n", 0) > 0:
            out.append("| %d | %d | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c["batch"], c["L"], c["arm"], c["D"] if c["D"] >= 0 else "nat", c["pattern"], c["n"], _f(c["call_ms"]), _f(c["call_ms_std"]), _f(c["gather_ms"]), _f(c["diff_ms"], "%.2f"), _f(c["attn_ms"]),
                _f(c["blocks_per_stream"], "%.2f"), _f(c["bytes"] / 1e6, "%.0f"), _f(c["achieved_gbps"]), _f(c["resident_ms"]), _f(c["residual_ms"], "%.2f"), _f(c["predicted_formula_ms"]),
                _f(c["logical_unique_kv_bytes"] / 1e9, "%.2f"), _f(c["hbm_traffic_bytes"] / 1e9, "%.2f"), c["verdict"]))
        else:
            out.append("| %d | %d | %s | %s | %s | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | %s (no timed step) |" % (c["batch"], c["L"], c["arm"], c["D"] if c["D"] >= 0 else "nat", c["pattern"], c["verdict"]))
    out += ["", "## registered predictions and falsifiers (ledger 2026-09-20; stated before the run)", "",
            "| item | verdict | detail |", "|---|---|---|"]
    for e in evaluation:
        out.append("| %s | %s | %s |" % (e["item"], e["verdict"], e["detail"]))
    out += ["", "registered at 16128 x 128, 4 splits: T_resident ~ %.0f ms; fixed %g-%g ms/step; BW ~%g GB/s at 1 MB/launch rising to %g-%g at >= 16 MB; T(1) ~ %g-%g, T(4) ~ %g, T(16) ~ %g, T(63) ~ %g ms; %s; %s; %s. FALSIFIERS: %s."
            % (REGISTERED["resident_ms"], REGISTERED["fixed_ms"][0], REGISTERED["fixed_ms"][1], REGISTERED["bw_small_gbps"], REGISTERED["bw_large_gbps"][0], REGISTERED["bw_large_gbps"][1],
               REGISTERED["T_ms"][1][0], REGISTERED["T_ms"][1][1], REGISTERED["T_ms"][4][0], REGISTERED["T_ms"][16][0], REGISTERED["T_ms"][63][0],
               "achieved BW at D >= 4 " + REGISTERED["bw_at_D_ge_4"], REGISTERED["layers4_vs_spread"], "adjacent vs scattered " + REGISTERED["adjacent_vs_scattered"], "; ".join(REGISTERED["falsifiers"]))]
    for name, lines in extras.items():
        out += ["", "## " + name, ""] + list(lines)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# the trace subclasses (bound to transfer_trace.TRACE by the drivers)
# ---------------------------------------------------------------------------
def make_trace_classes():
    from nosi import transfer_trace as _tt

    class AloneTrace(_tt.TransferTrace):
        """Mode "1" without the per-step logits clone (66 MB per step at batch 128;
        the driver keeps the logits it compares)."""

        def record_logits(self, logits):
            pass

    class ProfileTrace(AloneTrace):
        """AloneTrace plus torch.profiler record_function ranges at the trace's own
        bracket positions: 'diff' = [fetch_begin, fetch_mid] (diff_offload + the map
        copy), 'gather' = [fetch_mid, fetch_end] (the two Triton gathers), 'attention'
        = [attn_begin, attn_end], 'rest' = everything else inside the step (scoring,
        the pooling graph, the mask archive copies, GEMMs, norms). The ranges are
        sequential, never nested; the CUDA events are still recorded."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._rf = None

        def _enter(self, name):
            self._exit()
            self._rf = torch.profiler.record_function(name)
            self._rf.__enter__()

        def _exit(self):
            if self._rf is not None:
                self._rf.__exit__(None, None, None)
                self._rf = None

        def begin_step(self):
            super().begin_step()
            self._enter("rest")

        def fetch_begin(self):
            super().fetch_begin()
            self._enter("diff")

        def fetch_mid(self):
            super().fetch_mid()
            self._enter("gather")

        def fetch_end(self):
            super().fetch_end()
            self._enter("rest")

        def attn_begin(self):
            super().attn_begin()
            self._enter("attention")

        def attn_end(self):
            super().attn_end()
            self._enter("rest")

        def end_step(self):
            super().end_step()
            self._exit()

    return AloneTrace, ProfileTrace


def harvest_into(tr, pending: list):
    """Attach the trace's per-step brackets to the pending rows (decode-call order),
    cross-check the archive's load count against the engines' masks, reset the trace."""
    step_rows, layer_rows, _, _, _ = tr.harvest(timed_steps=())
    if len(step_rows) != len(pending) or tr.dropped_steps:
        raise RuntimeError("trace recorded %d steps for %d decode calls (dropped %d)" % (len(step_rows), len(pending), tr.dropped_steps))
    for s, row in zip(step_rows, pending):
        row.update(step_ms=s["step_ms"], gather_ms=s["transfer_ms"], diff_ms=s["diff_ms"], attn_ms=s["attn_ms"], trace_blocks_loaded=s["blocks_loaded"],
                   layers_recorded=s["layers_recorded"], fetch_in_step=s["fetch_in_step"], attn_after_fetch=s["attn_after_fetch"])
        if "blocks_loaded" in row and row["blocks_loaded"] != s["blocks_loaded"]:
            row["loads_ok"] = False
            row["fails"] = int(row.get("fails", 0)) + 1
            row["trace_mismatch"] = True
    tr.new_document()
    pending.clear()


# ---------------------------------------------------------------------------
# GPU drivers
# ---------------------------------------------------------------------------
def load_model(path, need_round_slots: bool = False):
    """The model, ONCE per process (the engine knobs are checked here, before 16 GB are read)."""
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    if _ce.POOL_BLOCKS != 0:
        die("NOSI_POOL_BLOCKS must be 0 (the experiment is the shipped 64-slot layout)")
    if need_round_slots and _ce.VERIFY_ROUND_SLOTS <= 0:
        die("path1 needs NOSI_VERIFY_ROUND_SLOTS > 0 (the engine read %d at import)" % _ce.VERIFY_ROUND_SLOTS)
    if not need_round_slots and _ce.VERIFY_ROUND_SLOTS != 0:
        die("NOSI_VERIFY_ROUND_SLOTS must be 0 outside path1 mode (the map/window helpers assume the shipped allocation)")
    splits = int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0)
    print("[alone] mode=%s tag=%s L=%d N=%d warm=%d batch=%d docs=%d D=%s frag D=%d %s ROUND_SLOTS=%d ATTN_SPLITS=%d ceiling=%.1f GB/s"
          % (MODE, TAG, L, N, WARM, BATCH, NDOCS, D_LIST, FRAG_D, FRAG, _ce.VERIFY_ROUND_SLOTS, splits, CEILING_GBPS), flush=True)
    return Llama(model_name=path, device="cuda", offload=True)


def _setup(model, ids):
    """A fresh cache + prefill for one document batch; the model re-warms (buffers and
    the pooling graph recaptured at the first decode step, as in a fresh process)."""
    from nosi import cache_engine as _ce
    from nosi.cache_engine import InfLLMv2Cache
    splits = int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0)
    x = ids.to("cuda")
    prompt, forced = x[:, :L], x[:, L:L + N]
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    model.has_buffers = False
    t0 = time.time()
    logits, position_ids = model.batch_prefill(prompt, cache)
    torch.cuda.synchronize()
    eng = cache.layers[0].cache_engine
    alloc_gb = sum(t.numel() * t.element_size() for t in (eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu)) * model.num_layers / 1e9
    print("[alone] prefill %d x %d: %.1fs; allocation %s, K+V+bias over %d layers %.2f GB" % (ids.shape[0], L, time.time() - t0, tuple(eng._k_gpu.shape), model.num_layers, alloc_gb), flush=True)
    meta = dict(mode=MODE, tag=TAG, L=L, N=N, warm=WARM, batch=int(ids.shape[0]), round_slots=_ce.VERIFY_ROUND_SLOTS, pool_blocks=_ce.POOL_BLOCKS,
                attn_splits=splits, kv_bias_scale=_ce.KV_BIAS_SCALE, capacity_gb=alloc_gb, ceiling_gbps=CEILING_GBPS, seed=SEED,
                D_list=list(D_LIST), frag_D=FRAG_D, frag=list(FRAG), nosi_commit=os.environ.get("NOSI_COMMIT"))
    return model, cache, logits, position_ids, forced, meta


def engine_kernels():
    """The engine's own (diff_offload, flash_h2d_from_mask, flash_h2d_from_mask_bias), resolved
    here (GPU process) and injected into miss_control: cache_engine compiles the diff extension
    at import, so the verify package never imports it."""
    from nosi import cache_engine as _ce
    from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
    from nosi.flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
    return _ce.diff.diff_offload, flash_h2d_from_mask, flash_h2d_from_mask_bias


def _timed_call(fn):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    out = fn()
    e1.record()
    torch.cuda.synchronize()
    return out, e0.elapsed_time(e1)


@torch.inference_mode()
def run_points(model, ids, doc_batch: int, distinct: int):
    from nosi import state_snapshot as ss
    from nosi import transfer_trace as _tt
    from nosi.verify import miss_control as mc
    AloneTrace, _ = make_trace_classes()
    model, cache, logits, position_ids, forced, meta = _setup(model, ids)
    B, nl = ids.shape[0], model.num_layers
    engines = [lay.cache_engine for lay in cache.layers]
    kernels = engine_kernels()
    tr = AloneTrace(nl, "1", max_steps=64)
    _tt.TRACE = tr
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    snap = ss.CacheSnapshot(cache)
    trans = None
    plan = row_plan(D_LIST, FRAG_D, FRAG)
    rows, obs, hyg, pending = [], [], [], []
    nfail = 0
    hashes = []

    def loaded_now():
        return int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum())

    def step(arm, D, pattern, **extra):
        (lg, ms) = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
        row = dict(batch=B, L=L, doc=doc_batch, step=it, arm=arm, D=D, pattern=pattern, call_ms=ms, blocks_loaded=loaded_now(), **extra)
        pending.append(row)
        rows.append(row)
        return lg, row

    def restore():
        snap.restore()
        ss.assert_transients_intact(model, trans)

    def prewarm_all(target):
        oks, n = [], []
        for l, (lay, e) in enumerate(zip(cache.layers, engines)):
            r = mc.prewarm(e, l, target[l], lay.total_cis, kernels)
            oks.append(r.ok)
            n.append(r.loaded)
        return bool(torch.stack(oks).all()), int(torch.stack(n).sum())

    def evict_all(D, pattern):
        per = mc.layer_plan(pattern, D, nl)
        pat = "spread" if pattern == "layers4" else pattern
        oks, n = [], []
        for l, e in enumerate(engines):
            if per[l] == 0:
                continue
            r = mc.evict(e, l, T[l], per[l], pat, SEED + 7919 * it + 104729 * doc_batch)
            oks.append(r.ok)
            n.append(r.evicted.sum())
        return (bool(torch.stack(oks).all()) if oks else True), (int(torch.stack(n).sum()) if n else 0), per

    def check(row, lg, expected, prewarm_ok, evict_ok):
        map_ok = bool(torch.stack([(e._block_map == T[l]).all() for l, e in enumerate(engines)]).all())
        eq = bool(torch.equal(lg, lg_ref))
        row.update(blocks_expected=expected, loads_ok=(row["blocks_loaded"] == expected), logits_equal=eq,
                   max_abs=float((lg - lg_ref).abs().max()), argmax_agree=float((lg.argmax(-1) == lg_ref.argmax(-1)).float().mean()),
                   map_ok=map_ok, prewarm_ok=bool(prewarm_ok), evict_ok=bool(evict_ok), nan_free=not bool(torch.isnan(lg).any()), layout="T_l")
        row["fails"] = sum(1 for c in CHECKS if not row[c])
        if row["fails"]:
            print("[alone] FAIL step %d %s D=%s %s: %s" % (it, row["arm"], row["D"], row["pattern"], {c: row[c] for c in CHECKS if not row[c]}), flush=True)
        return row["fails"]

    for it in range(N):
        tok = forced[:, it:it + 1]
        if it < WARM:
            lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
            torch.cuda.synchronize()
            pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="warm", D=-1, pattern="natural", blocks_loaded=loaded_now()))
            hashes.append(dict(step=it, arm="warm", sha=sha(lg)))
        else:
            if trans is None:
                trans = ss.transient_ids(model)
            t_wall = time.time()
            snap.take()
            prev = [e._block_map.clone() for e in engines]
            # 1. the plain shipped decode from R (natural layout): timed; its logits are an observation
            lg_nat, row_nat = step("shipped_nat", -1, "natural", layout="natural")
            T_nat = [mc.record_targets(e, l) for l, e in enumerate(engines)]
            hashes.append(dict(step=it, arm="shipped_nat", sha=sha(lg_nat)))
            restore()
            # 2. the reference: flushed map -> the ordinary step -> T_l and the reference logits
            for e in engines:
                mc.flush_map(e)
            lg_ref, row_ref = step("reference", WINDOW - 1, "flush", layout="T_l")
            T = [mc.record_targets(e, l) for l, e in enumerate(engines)]
            wf = bool(torch.stack([mc.targets_well_formed(t, WINDOW - 1) for t in T]).all())
            same = bool(torch.stack([mc.same_selection(T[l], T_nat[l]) for l in range(nl)]).all())
            ref_loads_ok = row_ref["blocks_loaded"] == (WINDOW - 1) * KV_HEADS * B * nl
            row_ref.update(targets_well_formed=wf, selection_equals_natural=same, loads_ok=ref_loads_ok, blocks_expected=(WINDOW - 1) * KV_HEADS * B * nl,
                           nan_free=not bool(torch.isnan(lg_ref).any()), logits_equal=True, map_ok=True, prewarm_ok=True, evict_ok=True)
            row_ref["fails"] = int(not wf) + int(not same) + int(not ref_loads_ok) + int(not row_ref["nan_free"])
            if row_ref["fails"]:
                print("[alone] FAIL step %d reference: well_formed=%s selection_equals_natural=%s loads_ok=%s nan_free=%s" % (it, wf, same, ref_loads_ok, row_ref["nan_free"]), flush=True)
            nfail += row_ref["fails"]
            row_nat.update(natural_equal_reference=bool(torch.equal(lg_nat, lg_ref)), max_abs=float((lg_nat - lg_ref).abs().max()),
                           argmax_agree=float((lg_nat.argmax(-1) == lg_ref.argmax(-1)).float().mean()), nan_free=not bool(torch.isnan(lg_nat).any()),
                           blocks_expected=None, note="natural layout: torch.equal here is an observation of layout invariance, not a control")
            obs.append(dict(step=it, doc=doc_batch, natural_equal_reference=row_nat["natural_equal_reference"], max_abs=row_nat["max_abs"], argmax_agree=row_nat["argmax_agree"],
                            natural_loads=row_nat["blocks_loaded"]))
            restore()
            # 3. arm A and the B(D, pattern) rows
            for D, pattern in plan:
                for e in engines:
                    mc.flush_map(e)
                pre_ok, pre_n = prewarm_all(T)
                if D > 0:
                    ev_ok, expected, per = evict_all(D, pattern)
                else:
                    ev_ok, expected, per = True, 0, [0] * nl
                lg, row = step("A" if D == 0 else "B", D, pattern, prewarm_loaded=pre_n, layer_plan=per if pattern == "layers4" else None)
                nfail += check(row, lg, expected, pre_ok, ev_ok)
                restore()
            # 4. the shipped decode at the controlled layout: the natural resident set in T_l-compatible slots
            ms_ok, miss = [], []
            Mstar = []
            for l in range(nl):
                m, ok, misses = mc.monotone_prestep_map(prev[l], T[l], WINDOW - 1)
                Mstar.append(m)
                ms_ok.append(ok)
                miss.append(misses.sum())
            for e in engines:
                mc.flush_map(e)
            pre_ok, pre_n = prewarm_all(Mstar)
            expected = int(torch.stack(miss).sum())
            lg, row = step("shipped", -1, "natural", prewarm_loaded=pre_n)
            nfail += check(row, lg, expected, pre_ok, bool(torch.stack(ms_ok).all()))
            row["natural_loads_equal"] = (row_nat["blocks_loaded"] == expected)
            if not row["natural_loads_equal"]:
                row["fails"] += 1
                nfail += 1
                print("[alone] FAIL step %d shipped: natural step loaded %d, M* predicts %d" % (it, row_nat["blocks_loaded"], expected), flush=True)
            restore()
            # 5. advance the natural trajectory; hygiene = the restore is exact
            lg_adv = model.decode_inference(tok, cu, position_ids, cache)
            torch.cuda.synchronize()
            pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="advance", D=-1, pattern="natural", blocks_loaded=loaded_now()))
            h = dict(step=it, doc=doc_batch, same=bool(torch.equal(lg_adv, lg_nat)), max_abs=float((lg_adv - lg_nat).abs().max()))
            hyg.append(h)
            hashes.append(dict(step=it, arm="advance", sha=sha(lg_adv)))
            if not h["same"]:
                nfail += 1
                print("[alone] FAIL step %d hygiene: the advance step differs from the natural step (max|d| %.4g)" % (it, h["max_abs"]), flush=True)
            print("[alone] step %d done in %.1fs: nat %.1f ms (%d loads) | ref %.1f | A %.1f | %s | shipped %.1f ms (%d loads) | fails so far %d" % (
                it, time.time() - t_wall, row_nat["call_ms"], row_nat["blocks_loaded"], row_ref["call_ms"],
                next(r["call_ms"] for r in rows if r["step"] == it and r["arm"] == "A"),
                " ".join("D%d/%s %.1f" % (r["D"], r["pattern"][:3], r["call_ms"]) for r in rows if r["step"] == it and r["arm"] == "B"),
                row["call_ms"], row["blocks_loaded"], nfail), flush=True)
        harvest_into(tr, pending)
        position_ids = position_ids + 1
    _tt.TRACE = None
    nfail += sum(int(r.get("trace_mismatch", False)) for r in rows)
    payload = dict(meta, doc_batch=doc_batch, distinct_docs=distinct, rows=rows, observations=obs, hygiene=hyg, hashes=hashes, fails=nfail,
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
    return payload


@torch.inference_mode()
def run_reference(model, ids, doc_batch: int, distinct: int):
    """The plain decode in a bare process: N steps timed, per-step brackets and logits hashes."""
    from nosi import transfer_trace as _tt
    AloneTrace, _ = make_trace_classes()
    model, cache, logits, position_ids, forced, meta = _setup(model, ids)
    B, nl = ids.shape[0], model.num_layers
    engines = [lay.cache_engine for lay in cache.layers]
    tr = AloneTrace(nl, "1", max_steps=64)
    _tt.TRACE = tr
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    rows, pending, hashes = [], [], []
    for it in range(N):
        tok = forced[:, it:it + 1]
        lg, ms = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0)))
        row = dict(batch=B, L=L, doc=doc_batch, step=it, arm="bare", D=-1, pattern="natural", call_ms=ms,
                   blocks_loaded=int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum()), layout="natural", gated=(it >= WARM))
        pending.append(row)
        rows.append(row)
        hashes.append(dict(step=it, arm="bare", sha=sha(lg)))
        harvest_into(tr, pending)
        position_ids = position_ids + 1
    _tt.TRACE = None
    gated = [r for r in rows if r["gated"]]
    print("[alone] bare decode: %s -> gated mean %.1f ms" % (" ".join("%.1f" % r["call_ms"] for r in rows), _mean([r["call_ms"] for r in gated])), flush=True)
    return dict(meta, doc_batch=doc_batch, distinct_docs=distinct, rows=rows, hashes=hashes, fails=0,
                peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)


@torch.inference_mode()
def run_profile(model, ids, doc_batch: int, distinct: int):
    """Chosen D values under torch.profiler: chrome traces + timeline_manifest.json for
    scripts/verify_timeline_report.py; the same rows unprofiled (CUDA events) beside them."""
    from nosi import state_snapshot as ss
    from nosi import transfer_trace as _tt
    from nosi.verify import miss_control as mc
    from torch.profiler import ProfilerActivity, profile, record_function
    _, ProfileTrace = make_trace_classes()
    model, cache, logits, position_ids, forced, meta = _setup(model, ids)
    B, nl = ids.shape[0], model.num_layers
    engines = [lay.cache_engine for lay in cache.layers]
    kernels = engine_kernels()
    tr = ProfileTrace(nl, "1", max_steps=64)
    _tt.TRACE = tr
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    snap = ss.CacheSnapshot(cache)
    trans = None
    rows, pending, sessions = [], [], []
    nfail, n_prof = 0, 0

    def loaded_now():
        return int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum())

    def prepare(D):
        for e in engines:
            mc.flush_map(e)
        oks = [mc.prewarm(e, l, T[l], lay.total_cis, kernels).ok for l, (lay, e) in enumerate(zip(cache.layers, engines))]
        ok = bool(torch.stack(oks).all())
        if D > 0:
            ev = [mc.evict(e, l, T[l], D, "spread", SEED + 7919 * it) for l, e in enumerate(engines)]
            ok = ok and bool(torch.stack([r.ok for r in ev]).all())
        return ok, D * KV_HEADS * B * nl

    def one(D, profiled, tag):
        nonlocal nfail
        ok, expected = prepare(D)
        if profiled:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(CALL_RANGE):
                    lg, ms = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
            fn = "timeline_%s.json.gz" % tag
            tmp = os.path.join(OUT, "timeline_%s.json" % tag)
            prof.export_chrome_trace(tmp)
            with open(tmp, "rb") as f_in, gzip.open(os.path.join(OUT, fn), "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(tmp)
        else:
            fn = None
            lg, ms = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
        row = dict(batch=B, L=L, doc=doc_batch, step=it, arm="A" if D == 0 else "B", D=D, pattern="spread", call_ms=ms, blocks_loaded=loaded_now(),
                   blocks_expected=expected, profiled=profiled, layout="T_l", prewarm_ok=ok, evict_ok=ok, file=fn, tag=tag)
        row.update(loads_ok=(row["blocks_loaded"] == expected), logits_equal=bool(torch.equal(lg, lg_ref)), max_abs=float((lg - lg_ref).abs().max()),
                   map_ok=bool(torch.stack([(e._block_map == T[l]).all() for l, e in enumerate(engines)]).all()), nan_free=not bool(torch.isnan(lg).any()))
        row["fails"] = sum(1 for c in CHECKS if not row[c])
        nfail += row["fails"]
        pending.append(row)
        rows.append(row)
        if fn is not None:
            sessions.append(dict(file=fn, tag=tag, call_range=CALL_RANGE, brackets=list(BRACKETS), D=D, pattern="spread", step=it, batch=B, L=L, profiled=True))
        return row

    for it in range(N):
        tok = forced[:, it:it + 1]
        if it < WARM or n_prof >= PROFILE_STEPS:
            lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
            torch.cuda.synchronize()
            pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="warm" if it < WARM else "advance", D=-1, pattern="natural", blocks_loaded=loaded_now()))
        else:
            if trans is None:
                trans = ss.transient_ids(model)
            snap.take()
            if n_prof == 0:
                # kineto's first session is a warm-up (CUPTI init): not exported
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
                    with record_function(CALL_RANGE):
                        model.decode_inference(tok, cu, position_ids, cache)
                torch.cuda.synchronize()
                pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="profiler_warmup", D=-1, pattern="natural", blocks_loaded=loaded_now()))
                snap.restore(); ss.assert_transients_intact(model, trans)
            # the natural shipped step, profiled, for the report's reference
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(CALL_RANGE):
                    lg_nat, ms = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
            tag = "b%d_shipped_nat_step%d" % (B, it)
            tmp = os.path.join(OUT, "timeline_%s.json" % tag)
            prof.export_chrome_trace(tmp)
            with open(tmp, "rb") as f_in, gzip.open(os.path.join(OUT, "timeline_%s.json.gz" % tag), "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(tmp)
            row = dict(batch=B, L=L, doc=doc_batch, step=it, arm="shipped_nat", D=-1, pattern="natural", call_ms=ms, blocks_loaded=loaded_now(), profiled=True, layout="natural", tag=tag)
            pending.append(row); rows.append(row)
            sessions.append(dict(file="timeline_%s.json.gz" % tag, tag=tag, call_range=CALL_RANGE, brackets=list(BRACKETS), D=-1, pattern="natural", step=it, batch=B, L=L, profiled=True))
            snap.restore(); ss.assert_transients_intact(model, trans)
            for e in engines:
                mc.flush_map(e)
            lg_ref, ms = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
            T = [mc.record_targets(e, l) for l, e in enumerate(engines)]
            pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="reference", D=WINDOW - 1, pattern="flush", call_ms=ms, blocks_loaded=loaded_now()))
            rows.append(pending[-1])
            snap.restore(); ss.assert_transients_intact(model, trans)
            for D in (0,) + tuple(PROFILE_D):
                one(D, False, "b%d_D%d_step%d_unprofiled" % (B, D, it))
                snap.restore(); ss.assert_transients_intact(model, trans)
                one(D, True, "b%d_D%d_step%d" % (B, D, it))
                snap.restore(); ss.assert_transients_intact(model, trans)
            lg_adv = model.decode_inference(tok, cu, position_ids, cache)
            torch.cuda.synchronize()
            pending.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="advance", D=-1, pattern="natural", blocks_loaded=loaded_now()))
            if not torch.equal(lg_adv, lg_nat):
                nfail += 1
                print("[alone] FAIL step %d hygiene (profile): advance differs from the natural step" % it, flush=True)
            n_prof += 1
            print("[alone] profile step %d: %s (fails %d)" % (it, " ".join("%s=%.1f" % (r["tag"], r["call_ms"]) for r in rows if r["step"] == it and "tag" in r), nfail), flush=True)
        harvest_into(tr, pending)
        position_ids = position_ids + 1
    _tt.TRACE = None
    brackets = [dict(U=r["D"], profiled=r.get("profiled", False), total_ms=r["call_ms"], score_ms=0.0, fetch_ms=r["diff_ms"] + r["gather_ms"], attn_ms=r["attn_ms"],
                     rest_ms=r["call_ms"] - r["diff_ms"] - r["gather_ms"] - r["attn_ms"], layers=r.get("layers_recorded"), tag=r.get("tag"), arm=r["arm"],
                     diff_ms=r["diff_ms"], gather_ms=r["gather_ms"], blocks_loaded=r["blocks_loaded"])
                for r in rows if "tag" in r]
    unprof = [r["call_ms"] for r in rows if r["arm"] == "shipped_nat"]
    manifest = dict(meta, doc_batch=doc_batch, sessions=sessions, brackets=brackets, call_range=CALL_RANGE, bracket_names=list(BRACKETS),
                    decode_step_ms_unprofiled=_mean(unprof) if unprof else float("nan"),
                    note="decode step traces: 'diff' = diff_offload + map copy, 'gather' = the two Triton gathers, 'attention' = flash_attn_nosa, 'rest' = scoring, pooling graph, GEMMs, norms, archive copies; profiled rows carry profiler overhead")
    with open(os.path.join(OUT, "timeline_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return dict(meta, doc_batch=doc_batch, distinct_docs=distinct, rows=rows, sessions=sessions, fails=nfail,
                peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)


@torch.inference_mode()
def run_path1(model, ids, doc_batch: int, distinct: int):
    """DIAGNOSTIC third arm: Path 1 verify_inference at U = 1 from the same restored
    state, timed with CUDA events, beside the shipped step. Needs ROUND_SLOTS > 0."""
    from nosi import state_snapshot as ss
    model, cache, logits, position_ids, forced, meta = _setup(model, ids)
    B = ids.shape[0]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    snap = ss.CacheSnapshot(cache)
    trans = None
    rows, nfail = [], 0
    for it in range(N):
        tok = forced[:, it:it + 1]
        if it >= WARM:
            if trans is None:
                trans = ss.transient_ids(model)
            snap.take()
            lv, ms_v = _timed_call(lambda: model.verify_inference(tok, cu, position_ids, cache))
            rows.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="path1_u1", D=-1, pattern="natural", call_ms=ms_v, layout="union"))
            snap.restore(); ss.assert_transients_intact(model, trans)
            lg, ms_d = _timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
            rows.append(dict(batch=B, L=L, doc=doc_batch, step=it, arm="shipped_nat", D=-1, pattern="natural", call_ms=ms_d, layout="natural",
                             path1_argmax_agree=float((lv[:, -1].argmax(-1) == lg[:, -1].argmax(-1)).float().mean()), path1_max_abs=float((lv[:, -1] - lg[:, -1]).abs().max())))
            snap.restore(); ss.assert_transients_intact(model, trans)
            print("[alone] path1 step %d: verify U=1 %.1f ms, shipped %.1f ms" % (it, ms_v, ms_d), flush=True)
        lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
        torch.cuda.synchronize()
        position_ids = position_ids + 1
    return dict(meta, doc_batch=doc_batch, distinct_docs=distinct, rows=rows, fails=nfail, peak_gb=torch.cuda.max_memory_allocated() / 1e9)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def _mc():
    """miss_control by file path: the table needs layer_plan only, and importing the
    `nosi` package would compile the diff extension (cache_engine at import)."""
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nosi", "nosi", "verify", "miss_control.py")
    spec = importlib.util.spec_from_file_location("nosi_miss_control_by_path", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_payloads(prefix):
    out = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith(prefix) and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                out.append((f[:-5], json.load(fh)))
    return out


def table() -> int:
    """points.json + points.md from every points_/reference_/profile_/path1_ payload of OUT.
    Exit code = requested cells (NOSI_ALONE_BATCHES x the D sweep x the fragmentation rows)
    with no timed row + hygiene/harness failures found here."""
    points = _load_payloads("points_")
    rows = [dict(r, source=name) for name, p in points for r in p["rows"] if r["arm"] in ("A", "B", "shipped", "shipped_nat", "reference")]
    cells = aggregate(rows)
    ev = evaluate(cells)
    fails_reported = sum(int(p.get("fails", 0)) for _, p in points)
    nfail = 0
    extras = {}
    # requested cells
    want = [(b, D, pat) for b in BATCHES for (D, pat) in row_plan(D_LIST, FRAG_D, FRAG)]
    have = {(c["batch"], c["D"], c["pattern"]) for c in cells if c["arm"] in ("A", "B") and c.get("n", 0) > 0}
    missing = [w for w in want if w not in have]
    nfail += len(missing)
    lines = ["MISSING cell batch=%d D=%d %s: no timed row (an empty measurement fails the point)" % w for w in missing] or ["every requested cell has timed rows"]
    lines.append("failed correctness checks reported by the points processes: %d (their exit codes carry them)" % fails_reported)
    extras["coverage"] = lines
    # hygiene, observations, provenance
    hl, ol, pl = [], [], []
    for name, p in points:
        for h in p.get("hygiene", []):
            hl.append("| %s | %d | %d | %s | %.3g |" % (name, h["doc"], h["step"], h["same"], h["max_abs"]))
        for o in p.get("observations", []):
            ol.append("| %s | %d | %d | %s | %.3g | %.4f | %d |" % (name, o["doc"], o["step"], o["natural_equal_reference"], o["max_abs"], o["argmax_agree"], o["natural_loads"]))
        pl.append("%s: L=%d N=%d warm=%d batch=%d doc batch %s (distinct docs %s) ROUND_SLOTS=%d POOL=%d ATTN_SPLITS=%d kv_bias_scale=%s capacity %.2f GB peak %.1f GB reserved %.1f GB fails=%d commit=%s"
                  % (name, p["L"], p["N"], p["warm"], p["batch"], p.get("doc_batch"), p.get("distinct_docs"), p["round_slots"], p["pool_blocks"], p["attn_splits"], p["kv_bias_scale"],
                     p["capacity_gb"], p["peak_gb"], p.get("reserved_gb", float("nan")), p.get("fails", 0), p.get("nosi_commit")))
    extras["hygiene (the advance step torch.equal to the natural step from the same restored state)"] = ["| source | doc | step | equal | max abs |", "|---|---|---|---|---|"] + (hl or ["(none)"])
    extras["observation: the natural-layout shipped step against the T_l-layout reference (NOT a control: the KV rows sit in other slots, the softmax reduction order may differ)"] = \
        ["| source | doc | step | torch.equal | max abs | argmax agree | natural loads |", "|---|---|---|---|---|---|---|"] + (ol or ["(none)"])
    # bare-process reference vs the points process's advance steps (same documents, same steps)
    refs = _load_payloads("reference_")
    rl = ["| B | doc | gated steps | bare call ms | points shipped_nat ms | points shipped (T_l) ms | logits hashes identical |", "|---|---|---|---|---|---|---|"]
    for name, p in refs:
        gated = [r for r in p["rows"] if r.get("gated")]
        pts = [(n, q) for n, q in points if q["batch"] == p["batch"] and q.get("doc_batch") == p.get("doc_batch")]
        for n, q in pts or [(None, None)]:
            nat = [r["call_ms"] for r in (q["rows"] if q else []) if r["arm"] == "shipped_nat"]
            shp = [r["call_ms"] for r in (q["rows"] if q else []) if r["arm"] == "shipped"]
            hb = {(h["step"]): h["sha"] for h in p["hashes"]}
            hp = {(h["step"]): h["sha"] for h in (q["hashes"] if q else []) if h["arm"] in ("warm", "advance")}
            common = sorted(set(hb) & set(hp))
            ident = all(hb[s] == hp[s] for s in common) if common else None
            if common and not ident:
                nfail += 1
            rl.append("| %d | %s | %d | %.1f | %s | %s | %s (%d steps compared) |" % (p["batch"], p.get("doc_batch"), len(gated), _mean([r["call_ms"] for r in gated]) if gated else float("nan"),
                                                                                    _f(_mean(nat) if nat else None), _f(_mean(shp) if shp else None), ident, len(common)))
    extras["bare-process reference (mode reference: no snapshot, no arms) vs the points process"] = rl if len(rl) > 2 else ["(no reference payload)"]
    # profile sessions and path1
    profs = _load_payloads("profile_")
    if profs:
        extras["profile sessions (torch.profiler; chrome traces for scripts/verify_timeline_report.py; profiled rows carry profiler overhead)"] = \
            ["| tag | arm | D | profiled | call ms | gather ms | diff ms | attn ms | loads | verdict |", "|---|---|---|---|---|---|---|---|---|---|"] + \
            ["| %s | %s | %s | %s | %.1f | %s | %s | %s | %d | %s |" % (r.get("tag"), r["arm"], r["D"], r.get("profiled", False), r["call_ms"], _f(r.get("gather_ms")), _f(r.get("diff_ms"), "%.2f"), _f(r.get("attn_ms")), r["blocks_loaded"],
                                                                      "PASS" if r.get("fails", 0) == 0 else "FAIL(%d)" % r["fails"]) for _, p in profs for r in p["rows"] if "tag" in r]
    p1 = _load_payloads("path1_")
    if p1:
        extras["diagnostic third arm: Path 1 verify_inference at U = 1 (NOT the definition of verification here)"] = \
            ["| B | doc | step | path1 U=1 ms | shipped ms | argmax agree | max abs |", "|---|---|---|---|---|---|---|"] + \
            ["| %d | %d | %d | %.1f | %.1f | %.4f | %.3g |" % (a["batch"], a["doc"], a["step"], a["call_ms"], b["call_ms"], b["path1_argmax_agree"], b["path1_max_abs"])
             for _, p in p1 for a, b in zip([r for r in p["rows"] if r["arm"] == "path1_u1"], [r for r in p["rows"] if r["arm"] == "shipped_nat"])]
    extras["provenance"] = pl or ["(no points payload)"]
    extras["columns"] = ["capacity_gb = the K+V+bias allocation of the process (provenance); logical unique KV bytes and HBM traffic are the two other columns of the table, reported separately as the ledger asks",
                         "layers4 = the same bytes as spread at D = %d concentrated at 63 blocks per launch (%s per layer; the ledger's '8 x D in 4 layers' exceeds the 63 usable slots by one, so the bytes are kept equal with a fifth launch)" % (FRAG_D, _mc().layer_plan("layers4", FRAG_D)),
                         "adjacent = the blocks T-1 .. T-%d (consecutive host ids inside the forced local window); scattered = per-stream random slots (seeded); spread = evenly spaced slots" % FRAG_D,
                         "reference row = the ordinary step from a flushed map (63 loads per stream): its logits and layout are the reference; 'shipped' = the natural resident set in the reference layout; 'shipped_nat' = the literal plain decode (natural layout)"]
    text = render_points(cells, ev, extras)
    print(text, flush=True)
    with open(os.path.join(OUT, "points.md"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, "points.json"), "w") as f:
        json.dump(dict(cells=cells, evaluation=ev, registered=REGISTERED, missing=missing, fails_reported=fails_reported, fails_table=nfail,
                       requested=dict(batches=list(BATCHES), D=list(D_LIST), frag_D=FRAG_D, frag=list(FRAG)), ceiling_gbps=CEILING_GBPS,
                       columns=dict(call_ms="CUDA events around decode_inference, mean over gated steps (first per document batch excluded)", call_ms_std="population std of the same",
                                    gather_ms="transfer trace [fetch_mid, fetch_end] summed over layers: the two Triton gathers", diff_ms="[fetch_begin, fetch_mid]: diff_offload + the map copy",
                                    attn_ms="[attn_begin, attn_end] summed over layers", bytes="blocks_loaded x 32768 (engines' load masks; cross-checked against the trace archive)",
                                    achieved_gbps="bytes / gather_ms", resident_ms="arm A of the same (batch, L)", residual_ms="call_ms - resident_ms - bytes / ceiling",
                                    predicted_formula_ms="resident_ms + %.1f + bytes / ceiling" % FIXED_MS_FORMULA, logical_unique_kv_bytes="64 x 2 heads x B x 32 layers x 32768",
                                    hbm_traffic_bytes="gather writes + attention reads (64 x B x 65792 x 32) + map copies (6 x 2 x B x 64 x 8 x 32)")),
                  f, indent=1)
    return nfail


if __name__ == "__main__":
    if MODE == "table":
        sys.exit(min(table(), 200))
    check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = load_corpus(path, BATCH)
    model = load_model(path, need_round_slots=(MODE == "path1"))
    total_fail = 0
    for j in range(NDOCS if MODE != "path1" else 1):
        ids, docs, distinct = pick_batch(corpus, BATCH, j * BATCH)
        print("[docs] batch %d: %s%s (%d distinct)  L=%d N=%d" % (j, docs[:8], "..." if len(docs) > 8 else "", distinct, L, N), flush=True)
        if MODE == "points":
            payload = run_points(model, ids, j, distinct)
        elif MODE == "reference":
            payload = run_reference(model, ids, j, distinct)
        elif MODE == "profile":
            payload = run_profile(model, ids, j, distinct)
        elif MODE == "path1":
            payload = run_path1(model, ids, j, distinct)
        else:
            raise SystemExit("unknown NOSI_ALONE_MODE %r" % MODE)
        payload["docs"] = docs
        fn = os.path.join(OUT, "%s_%s_doc%d.json" % (MODE, TAG, j))
        with open(fn, "w") as f:
            json.dump(payload, f, indent=1)
        total_fail += int(payload.get("fails", 0))
        print("[alone] saved %s (peak %.2f GB, fails %d)" % (fn, payload["peak_gb"], payload.get("fails", 0)), flush=True)
        del payload
        gc.collect()
        torch.cuda.empty_cache()
    sys.exit(min(total_fail, 200))
