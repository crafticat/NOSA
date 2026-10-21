"""R2 step 7, PIECE 4: the GPU gate of the FLATTENED JOINT TICK (nosi/nosi/r2_tick.py) against the BI SERIAL DECODE
(nosi/nosi/r2_bi_decode.py, C's G0-BI path, dd5c4b9): SG0 - SG4 of docs/R2_STEP7_SPEC_ROUND.md 7.2 (SG5, the R-PERF
budget, is designed in docs/R2_STEP7_TICK_SG5.md against piece 5's Residency). Holder step: benchmarks/Efficiency/
r2_tick_step.sh. CPU tests: benchmarks/Efficiency/tests_cpu/test_r2_tick.py. One model, the parts run one after
another in one process.

THE OPS (one BIOps object for the reference and the tick, C's contract): piece 3 v2's paged attention (--biattn, a
pinned copy), the GEMM mode --gemm (exact = G-b PaddedLinear(--m-fix) for wqkv / wo / gate_up / down / lm_head; fast =
natural cuBLAS, the weaker claim: SG0 / SG1 / SG2 become REPORT), m_fix = 4B by default (G-b's bits depend on m_fix),
the deterministic selection, the fixed-order twin means.

SETUP. B ragged documents of pg19-test-c16k (S - deltas[b] tokens). ONE build_ragged (body r2, f4_fix True) with the
horizon --steps + 80. BASE = TickEngine.from_r2 of that build, taken BEFORE its first decode step (a copy: the store
from the host K / V rows and the twin's cis rows, both banks from the twins). Then on the build itself:
  REF  the BI serial decode (attach_bi), greedy, free-running, N_REF = --steps + 16 steps: the tokens (the inputs, then
       the argmaxes) and every step's logits (the reference rows, (N_REF, B, V) float32 on the GPU).
Every part below runs on its own copy of BASE (BASE.clone()).
  SG0  (MUST) the engine's serial mode (run_serial: one exact row per request per call, committed at once),
       teacher-forced on REF's tokens, --sg0-steps steps: every row torch.equal to REF's row. The storage, the page
       lists and the ops of the engine agree with the BI serial decode before any flattening.
  SG1  (MUST) the teacher-forced tick: drafts and corrections are REF's tokens; the rejection index comes from
       r2_tick.forced_schedule (offsets 0 .. L - 1 and full acceptance in turn; a cohort that crosses a block boundary
       is rolled back from the new block's first row, and from the old block's last row, in turn); until every request
       committed --steps tokens. Every verify row, REJECTED ROWS INCLUDED, torch.equal to REF's row at the same
       (request, position). MUST preconditions (else the gate is blind): rejections at every offset 0 .. K - 1, a full
       acceptance, both boundary kinds. MUST: the committed sequences equal REF's. CONTROL (MUST): every compared row
       differs from REF's row at the NEXT position (the comparison can fail).
  SG2  (MUST) the free-running greedy tick: draft rows with --draft-drop of their middle blocks not resident;
       decisions by the verifier's argmax (r2_spec_core.Round). Every request's committed --steps tokens torch.equal
       to REF's; every verify row ON REF'S PATH (its input and its cohort's earlier inputs equal REF's tokens) torch.equal
       to REF's row. MUST preconditions: rejections > 0 and publications > 0.
  SG3  (MUST) the PHASE MIX: SG1's teacher-forced tick with request b's cohorts gated to the ticks t with
       (t + b) % --phase == 0 (r2_tick.phase_ready): requests verify at different ticks, a request drafts while others
       verify, a request whose cohort waits reaches its lead bound and verifies without its own draft row. Every verify
       row torch.equal to REF's row, counted per composition class (own draft row in the tick, another request's verify
       rows, another request's draft row). MUST precondition: verify rows with and without their own draft row, with
       and without another request's verify rows, and >= 3 tick sizes.
  SG4  the NEGATIVES, each SG1's teacher-forced tick over --sg4-steps tokens with ONE op replaced: (a) natural cuBLAS
       (gemm fast), (b) a batch-keyed split-KV attention (8 pages per split for <= B rows, else 4: r2_tick.
       BatchKeyedSplit), (c) torch.topk(sorted=False) selection (r2_tick.topk_select). MUST: each run completes;
       REPORT: DETECTED (some verify row differs from REF) or BLIND. SG1 and SG4c count the NATURAL ties across the
       two cuts of the selection per layer (r2_tick.tie_stats; REPORT: with none, the tie rule is unobservable).
       SG4c PLANTED (lead, 2026-10-21): ties planted across the top-k cut (r2_tick.plant_tie) at layers 0, L / 2,
       L - 1 for the rows of requests 0 and 1; MUST precondition (asserted first): the engine counts a top-k tie for
       every planted (row, head, layer); MUST control: the higher-id tie rule (r2_tick.high_select) changes some verify
       row (by construction, so the plants reach the selection); REPORT: torch.topk(sorted=False) against the stable
       rule under the same plants. Tick v3 (A100, torch 2.6; tick_v3_2190876): natural ties are COMMON (106506 of
       147136 selections tie across the top-k cut, 32 / 32 layers), every planted pair tied, the control differed in
       300 / 300 rows and torch.topk in 0 / 300: CUDA's topk breaks ties toward the lower block id exactly as the
       stable rule, so the shipped tie rule IS the stable rule there (a REPORT since then, lead 2026-10-21).
REPORT: ticks, rows per tick (draft / verify), decision reads, ticks with verify rows, rollbacks, publications,
write-backs, acceptance (NOT representative: the drafts are a stand-in), host ms per tick, ms per REF step, GPU peak.
OUTPUT: --out/summary.json (written after every part). The last stdout line is R2_TICK_GATE_DONE failed=<n>; exit = n
(2 = refused). ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0 (the BI serial decode's).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0  # noqa: E402  (load_workload)
import r2_g0bi_gate as GB  # noqa: E402  (decode_loop, load_by_path, free, gate)

PARTS = ("SG0", "SG1", "SG2", "SG3", "SG4")
ATTN_VARIANTS = ("per_row", "shared", "shared-whole", "probe", "probe-whole", "probe-plant")   # [sharedkv]
ROOT = Path(HERE).resolve().parents[1]                                                          # [sharedkv]
# [sharedkv] REGISTERED bands (docs/R2_SHAREDKV_PLAN.md section 5, revised after the review of f79d0df). Tiles: no
# fixed band; the expectation is DERIVED from the recorded group mix (bi_paged_attention_shared.RHO per group size,
# expected_tile_ratio) and the measured ratio must lie in RHO_BAND x that expectation. Time (REPORT; the primary time
# ratio is the S3 cell's device kernel time): over the keyed calls of rows_max 4, the order-pooled median of (union
# build + shared kernel + combine) / (piece 3's split + combine), CUDA events behind a pre-queued GPU.
SKV_PRED = {"split": dict(time_with_build=(1.30, 0.80, 2.20),        # (central, lo, hi), --union torch
                          time_with_fused_build=(0.90, 0.60, 1.10)),  # [sharedkv4 MINOR-4] hi = F5' (1.10)
            "whole": dict(time_with_build=(2.10, 1.00, 4.00),
                          time_with_fused_build=None)}                # [sharedkv4 MINOR-4] no whole fused arm: none


def select_attn(a, attn):
    """[sharedkv] the tick's attention object for --attn (docs/R2_SHAREDKV_PLAN.md). per_row: piece 3 itself, the
    default (SG5 / p6 comparability). Every other choice wraps the SAME loaded piece 3 module; calls without page_key
    (the BI serial REF) still run piece 3 (the wrapper's stats count them)."""
    if a.attn == "per_row":
        return attn
    sk = GB.load_by_path(a.sharedkv, "sharedkv_bi_paged_attention")
    mode = "whole" if a.attn.endswith("whole") else "split"
    if a.attn.startswith("probe"):
        return sk.SharedKVProbe(attn, mode=mode, plant=(a.attn == "probe-plant"), union=a.union)
    return sk.SharedKV(attn, mode=mode, union=a.union)


def sharedkv_gates(a, obj, gates, res):
    """[sharedkv] after the parts: the selector RAN (MUST, else every bitwise gate above is blind to it), and for the
    probes the in-tick bitwise A/B (MUST 0 differing rows; probe-plant MUST > 0 = DETECTED, 0 = a NO-OP plant, L34)
    and the per-call times (REPORT; the registered band lives in the plan, the summary keeps the raw times)."""
    if a.attn == "per_row":
        return
    st = dict(obj.stats)
    gates.append(GB.gate("SKV_shared_kernel_ran", st.get("shared", 0) > 0, True, "stats %s" % st))
    n4 = obj.by_rows_max.get(4, 0)                                         # [review MINOR-3] F3, machine-checked
    gates.append(GB.gate("SKV_rows_max4_calls", n4 > 0, True, "keyed calls per rows_max %s (F3: none with 4 rows of "
                         "one request = the cell did not test the 4-row groups)" % (dict(obj.by_rows_max),)))
    want_u = getattr(a, "union", "torch")                                  # [sharedkv4 MINOR-5] the build that ran
    ub = dict(getattr(obj, "union_builds", {}))
    other = "torch" if want_u == "fused" else "fused"
    gates.append(GB.gate("SKV_union_build_kind", ub.get(want_u, 0) > 0 and ub.get(other, 0) == 0, True,
                         "--union %s; union builds %s (a fallback to the other build would be silent otherwise)"
                         % (want_u, ub)))
    if not a.attn.startswith("probe"):
        res["SKV"] = dict(attn=a.attn, stats=st, by_rows_max=dict(obj.by_rows_max), union_builds=ub)
        return
    rep = obj.report()
    res["SKV"] = rep
    if want_u == "fused":                                                  # [sharedkv4, review MAJOR-1]
        diff = rep.get("union_diff", {})
        gates.append(GB.gate("SKV_fused_union_equals_torch", rep.get("union_checked", 0) > 0
                             and all(v == 0 for v in diff.values()), True,
                             "%d keyed calls: the fused union vs build_union on the same inputs, differing elements "
                             "per field %s (a too-wide window is invisible to the bitwise gate)"
                             % (rep.get("union_checked", 0), diff)))
    try:                                                                   # [review MAJOR-1] F4's own tables
        import torch
        path = os.path.join(a.out, "skv_tables.pt")
        torch.save(obj.report_tables(), path)
        gates.append(GB.gate("R_SKV_tables", True, False, "%d recorded calls -> %s" % (len(obj.tables), path)))
    except Exception as e:
        gates.append(GB.gate("R_SKV_tables", False, False, "not saved: %s: %s" % (type(e).__name__, str(e)[:200])))
    if a.attn == "probe-plant":
        gates.append(GB.gate("SKV_plant_detected", rep["diff_rows"] > 0, True,
                             "%d of %d compared rows differ (0 = the plant was a NO-OP on these inputs: BLIND)"
                             % (rep["diff_rows"], rep["rows_compared"])))
        return
    gates.append(GB.gate("SKV_bitwise_shared_equals_per_row", rep["diff_rows"] == 0 and rep["rows_compared"] > 0, True,
                         "%d of %d compared rows differ (int views: a sign-of-zero difference counts); differing "
                         "elements: sign-of-zero only %d, value %d (F1 triage); first samples %s"
                         % (rep["diff_rows"], rep["rows_compared"], rep["diff_elements_zero_sign"],
                            rep["diff_elements_value"], rep["diff_samples"][:4])))
    if rep["tile_identity_checked"] > 0:                                   # [review MAJOR-1] the counter's identities
        gates.append(GB.gate("SKV_tile_identity", not rep["tile_identity_bad"], True,
                             "%d calls checked (whole: tiles = union pages; split: union <= tiles <= sum); bad %s"
                             % (rep["tile_identity_checked"], rep["tile_identity_bad"][:4])))
    pred = SKV_PRED[rep["mode"]]
    bandv = pred["time_with_fused_build" if getattr(a, "union", "torch") == "fused" else "time_with_build"]
    c, lo, hi = bandv if bandv is not None else (None, float("-inf"), float("inf"))   # None: no arm, not registered
    pooled = {}
    for (nr, first, pr), (_, _, sh) in zip(rep["per_row_ms"], rep["shared_ms"]):
        pooled.setdefault(nr, []).append(sh / pr if pr > 0 else float("nan"))
    txt = []
    for nr in sorted(pooled):                                              # [review MINOR-10] order-pooled median
        xs = sorted(pooled[nr])
        med = xs[len(xs) // 2]
        txt.append("rows_max %d: n %d median (build + shared) / per_row %.3f%s" % (
            nr, len(xs), med, "" if nr != 4 else " band [%s, %s] central %s %s" % (lo, hi, c, "IN" if lo <= med <= hi
                                                                                   else "OUT")))
    gates.append(GB.gate("R_SKV_time_ratio", True, False, "CUDA events behind a pre-queued GPU, REPORT only (the "
                         "primary time ratio is the S3 cell's device kernel time): " + ("; ".join(txt) or "no timed call")))
    tr, ex = rep["tile_ratio"], rep["expected_tile_ratio"]
    if tr is None or ex is None:
        verdict = "n/a"
    else:
        verdict = "IN" if RHO_BAND(obj)[0] * ex <= tr <= RHO_BAND(obj)[1] * ex else "OUT"
    gates.append(GB.gate("R_SKV_tile_ratio", True, False,
                         "tiles shared / per-row %s; expected from the recorded group mix %s (RHO %s, band %s x "
                         "expected) %s; union / sum per group size %s; groups per size %s"
                         % (tr, ex, type(obj).report.__globals__.get("RHO", {}).get(rep["mode"]),
                            RHO_BAND(obj), verdict, rep["union_ratio_by_size"], rep["mix_all"][0])))


def RHO_BAND(obj):
    """[review MAJOR-1] the registered relative band of the tile ratio around its mix-derived expectation (the shared
    module's RHO_BAND, read from the probe class's own module globals: the module may be loaded by path)."""
    return type(obj).report.__globals__.get("RHO_BAND", (0.80, 1.25))


def log(msg):
    print("[r2_tick %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy; v2 7c826c1)")
    ap.add_argument("--gemm", default="exact", choices=("exact", "fast"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--deltas", default="0,1,17,63,64,65,777,4097")
    ap.add_argument("--steps", type=int, default=96, help="committed tokens per request (the first token included)")
    ap.add_argument("--sg0-steps", type=int, default=64)
    ap.add_argument("--sg4-steps", type=int, default=32, help="committed tokens per request in each SG4 negative")
    ap.add_argument("--phase", type=int, default=3, help="SG3: request b's cohort runs at ticks t, (t + b) % phase == 0")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--draft-drop", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0, help="G-b's row count; 0 = 4B")
    ap.add_argument("--no-check", action="store_true", help="skip the per-layer selection check (a host read)")
    ap.add_argument("--parts", default="SG0,SG1,SG2,SG3,SG4")
    ap.add_argument("--attn", default="per_row", choices=ATTN_VARIANTS,                         # [sharedkv]
                    help="the tick's attention: per_row = piece 3 (default, unchanged); shared[-whole] = the "
                         "shared-read kernel IN the tick (SG1-SG3 then gate it bitwise against the BI serial REF); "
                         "probe[-whole] = both kernels on every keyed call, piece 3's output returned, bitwise and "
                         "time A/B; probe-plant = probe with one union page dropped (the negative control)")
    ap.add_argument("--union", default="torch", choices=("torch", "fused"),                     # [sharedkv3]
                    help="the shared side's union build: torch (build_union, about 50 ops per call) or fused (the "
                         "groups once per tick + one kernel per call, build_union_fused)")
    ap.add_argument("--sharedkv", default=str(ROOT / "nosi/nosi/bi_paged_attention_shared.py"),   # [sharedkv]
                    help="the shared-read module (loaded by path; it imports nothing from nosi)")
    a = ap.parse_args(argv)
    a.deltas = [int(x) for x in a.deltas.split(",") if x != ""]
    a.parts = [p for p in a.parts.split(",") if p]
    if not a.parts or any(p not in PARTS for p in a.parts):
        ap.error("parts: a list of %s" % (PARTS,))
    if len(a.deltas) != a.batch or min(a.deltas) < 0:
        ap.error("--deltas needs %d non-negative values" % a.batch)
    if a.S - max(a.deltas) < 64 * 64:
        ap.error("every prompt needs >= 4096 tokens (64 blocks: the selection of 64 blocks must name real blocks)")
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    if a.m_fix < (1 + a.K) * a.batch:
        ap.error("m_fix %d < (1 + K) B = %d: a tick has up to (1 + K) B rows" % (a.m_fix, (1 + a.K) * a.batch))
    if a.steps < 2 or not 1 <= a.sg0_steps <= a.steps or not 0.0 <= a.draft_drop < 1.0:
        ap.error("steps >= 2, 1 <= sg0-steps <= steps, draft-drop in [0, 1)")
    if not 2 <= a.sg4_steps <= a.steps or a.phase < 2:
        ap.error("2 <= sg4-steps <= steps, phase >= 2")
    if a.attn != "per_row" and a.no_check:                               # [review MINOR-5; sharedkv4 MINOR-3]
        ap.error("--attn %s needs the per-layer selection check (it guarantees the keys' last two entries T - 1, T "
                 "and strictly ascending keys per row, which the fused union build assumes; without it the shared "
                 "side can break the law silently): drop --no-check" % a.attn)
    return a


def cmp_rows(logits, ref_logits, S_list, picks):
    """picks: [(row k, request b, position p)] with p - S_b < N_REF. Device bools (len(picks),): row == REF's row at p,
    and row == REF's row at p + 1 (the control; False where p + 1 is past REF)."""
    import torch
    n_ref = ref_logits.shape[0]
    if not picks:
        z = torch.zeros(0, dtype=torch.bool, device=logits.device)
        return z, z
    dev = logits.device
    kk = torch.tensor([k for k, _, _ in picks], dtype=torch.int64, device=dev)
    ss = torch.tensor([p - S_list[b] for _, b, p in picks], dtype=torch.int64, device=dev)
    bb = torch.tensor([b for _, b, _ in picks], dtype=torch.int64, device=dev)
    rows = logits.index_select(0, kk)
    eq = (rows == ref_logits[ss, bb]).all(-1)
    nxt_ok = ss + 1 < n_ref
    nxt = (rows == ref_logits[(ss + 1).clamp(max=n_ref - 1), bb]).all(-1) & nxt_ok
    return eq, nxt


def part_sg0(a, base, ref, gates, res):
    import torch
    eng = base.clone()
    S_list = eng.S_list
    toks = ref["tok_host"]
    eqs, nxts = [], []
    t0 = time.time()
    for s in range(a.sg0_steps):
        out = eng.run_serial([int(toks[s, b]) for b in range(eng.B)])
        e, n = cmp_rows(out.logits, ref["logits"], S_list, [(b, b, S_list[b] + s) for b in range(eng.B)])
        eqs.append(e)
        nxts.append(n)
    eq = torch.stack(eqs).cpu()                                            # (steps, B)
    nx = torch.stack(nxts).cpu()
    bad = (~eq).nonzero().tolist()
    gates.append(GB.gate("SG0_engine_serial_equals_BI_serial", not bad, a.gemm == "exact",
                         "%d of %d (step, request) rows differ%s" % (len(bad), eq.numel(),
                                                                     "; first (step, request) %s" % bad[:4] if bad else "")))
    gates.append(GB.gate("SG0_control_next_position_differs", not bool(nx.any()), True,
                         "%d of %d rows equal REF's row at the next position" % (int(nx.sum()), nx.numel())))
    res["SG0"] = dict(steps=a.sg0_steps, ms_per_step=round(1e3 * (time.time() - t0) / a.sg0_steps, 1),
                      writebacks=eng.stats["writebacks"])
    del eng


def tie_summary(eng):
    """The engine's tie counters (one read): per layer (pairs, ties at the QK_SELECT cut, ties at the top-k cut)."""
    rows = eng.tie_report()
    pairs = sum(r[0] for r in rows)
    tq, tc = sum(r[1] for r in rows), sum(r[2] for r in rows)
    text = ("%d (row, head, layer) selections: %d tie across the QK_SELECT cut, %d across the top-k cut (layers with a "
            "top-k tie: %d of %d)%s" % (pairs, tq, tc, sum(1 for r in rows if r[2]), len(rows),
                                        "" if tq or tc else "; NO natural tie at either cut: the tie rule (stable vs "
                                        "torch.topk) is unobservable in practice on these pooled scores"))
    return dict(per_layer=rows, pairs=pairs, ties_qk=tq, ties_topk=tc, text=text)


def planted_run(a, base, ref, select, layers, requests, steps):
    """SG4c PLANTED: SG1's teacher-forced tick over `steps` tokens with ties planted across the top-k cut (r2_tick.
    plant_tie) at `layers` for the rows of `requests`, the selection rule `select`. Returns (verify logits in tick and
    row order (GPU), the tie counters per layer, the planted (row, head, layer) pairs counted on the host)."""
    import torch
    from nosi.r2_tick import forced_schedule, run_round
    eng = base.clone(select=select)
    eng.count_ties()
    eng.plant_ties(requests, layers)
    B, S_list, toks = eng.B, eng.S_list, ref["tok_host"]
    logits, planted = [], [0]
    rq = set(requests)

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        vi = [k for k, r in enumerate(rows) if r.kind == "v"]
        if vi:
            logits.append(out.logits.index_select(0, torch.tensor(vi, device=out.logits.device)).clone())
        planted[0] += sum(1 for r in rows if r.b in rq) * eng.H * len(layers)
    reject_at, _ = forced_schedule(a.K)
    run_round(eng, [int(toks[0, b]) for b in range(B)], [steps] * B, K=a.K, LAMBDA=a.LAMBDA, teacher=teacher,
              reject_at=reject_at, on_tick=on_tick)
    out = (torch.cat(logits) if logits else None, eng.tie_report(), planted[0])
    del eng
    return out


def part_planted(a, base, ref, gates, res):
    """SG4c PLANTED (lead, 2026-10-21): the precondition (every planted (row, head, layer) has a tie across the top-k
    cut, counted by the engine) asserted FIRST; the higher-id tie rule is the control that the plants reach the
    selection (MUST: it differs by construction); torch.topk(sorted=False) against the stable rule under the same
    plants is a REPORT (tick v3: CUDA's topk breaks ties toward the lower id, as the stable rule; module docstring)."""
    from nosi.r2_tick import high_select, topk_select
    from nosi.r2_spec_core import select_blocks
    layers = sorted({0, len(base.twins) // 2, len(base.twins) - 1})
    requests = list(range(min(2, base.B)))
    st_logits, st_ties, _ = planted_run(a, base, ref, select_blocks, layers, requests, a.sg4_steps)
    planted = sum(st_ties[l][3] for l in layers)
    got = sum(st_ties[l][4] for l in layers)
    pre = st_logits is not None and planted > 0 and got == planted
    gates.append(GB.gate("SG4c_planted_precondition", pre, True,
                         "planted (row, head, layer) pairs %d (counted on the device); of them tie across the top-k cut "
                         "%d (must be all)" % (planted, got)))
    out = dict(layers=layers, requests=requests, planted=planted, ties_stable=st_ties)
    runs = {}
    for name, sel in (("SG4d_planted_high", high_select), ("SG4c_planted_topk", topk_select)):
        lg, ties, _ = planted_run(a, base, ref, sel, layers, requests, a.sg4_steps)
        same_shape = lg is not None and st_logits is not None and lg.shape == st_logits.shape
        differ = int((lg != st_logits).any(-1).sum()) if same_shape else -1
        runs[name] = (differ, 0 if lg is None else int(lg.shape[0]))
        out[name] = dict(differ=differ, rows=runs[name][1], ties=ties)
    d_hi, n_hi = runs["SG4d_planted_high"]
    gates.append(GB.gate("SG4d_planted_high_detected", pre and d_hi > 0, True,
                         "ties to the higher block id (the CONTROL: the plants reach the selection) under the planted "
                         "ties: %d of %d verify rows differ from the stable rule's%s"
                         % (d_hi, n_hi, "" if pre else " (the precondition FAILED: not a valid detection)")))
    d_tk, n_tk = runs["SG4c_planted_topk"]
    # REPORT (lead, 2026-10-21, after tick v3 on the A100 / torch 2.6): torch.topk(sorted=False) broke every planted
    # tie toward the lower block id, exactly as the stable rule (0 / 300 rows differ, the control 300 / 300)
    if pre and d_hi > 0 and d_tk == 0:
        why = ("SAME RULE: on this GPU / torch, torch.topk(sorted=False) breaks the planted ties toward the LOWER block id "
               "exactly as the stable rule (0 of %d rows differ while the higher-id control differs in %d): the shipped "
               "tie rule IS the stable rule here, so the stable selection costs no fidelity and this negative cannot be "
               "a negative on this kernel" % (n_tk, d_hi))
    else:
        why = "torch.topk(sorted=False) under the planted ties: %d of %d verify rows differ from the stable rule's" % (
            d_tk, n_tk)
    gates.append(GB.gate("R_SG4c_planted_topk", True, False, why))
    res["SG4c_planted"] = out


def part_tick(a, base, ref, gates, res, forced, name=None, ready=None, eng_kw=None, steps=None, negative=None):
    """SG1 (forced), SG2 (free running), SG3 (forced + ready = phase_ready: the phase mix) and the SG4 negatives
    (forced, eng_kw replaces the ops or the selection; negative = its label: detection is a REPORT)."""
    import torch
    from nosi.r2_tick import Ref, compositions, forced_schedule, run_round
    name = name or ("SG1" if forced else "SG2")
    steps = steps or a.steps
    eng = base.clone(**(eng_kw or {}))                                     # draft rows drop --draft-drop in all
    if name in ("SG1", "SG4c"):
        eng.count_ties()                                                   # natural ties at the two cuts (report)
    B, S_list = eng.B, eng.S_list
    toks = ref["tok_host"]
    n_ref = ref["logits"].shape[0]
    records, eqs, nxts, comps = [], [], [], []
    per_tick = []
    skipped = [0]

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        picks = []
        comp = compositions(rows)
        for k, row in enumerate(rows):
            if row.kind != "v":
                continue
            s = row.pos - S_list[row.b]
            if s >= n_ref:
                skipped[0] += 1
                continue
            picks.append((k, row.b, row.pos))
            records.append((row.b, row.pos, tokens[k], (row.b, row.epoch, row.m), row.i))
            comps.append(comp[k][1:])
        e, n = cmp_rows(out.logits, ref["logits"], S_list, picks)
        eqs.append(e)
        nxts.append(n)
        per_tick.append((sum(1 for r in rows if r.kind == "d"), sum(1 for r in rows if r.kind == "v")))

    reject_at, sched = (forced_schedule(a.K) if forced else (None, []))
    t0 = time.time()
    rnd = run_round(eng, [int(toks[0, b]) for b in range(B)], [steps] * B, K=a.K, LAMBDA=a.LAMBDA,
                    teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick, ready=ready)
    torch.cuda.synchronize()
    sec = time.time() - t0
    eq = torch.cat(eqs).cpu().tolist() if eqs else []
    nx = torch.cat(nxts).cpu().tolist() if nxts else []
    tok_all = eng.tok_log[:eng.n_tok].cpu().tolist()
    inp = [tok_all[t.slot] if isinstance(t, Ref) else int(t) for _, _, t, _, _ in records]
    ref_in = [int(toks[p - S_list[b], b]) for b, p, _, _, _ in records]
    # on REF's path: the row's input and its cohort's earlier inputs equal REF's
    on_path = [False] * len(records)
    by_cohort = {}
    for j, (b, p, _, key, i) in enumerate(records):
        by_cohort.setdefault(key, []).append((i, j))
    for key, lst in by_cohort.items():
        ok = True
        for i, j in sorted(lst):
            ok = ok and inp[j] == ref_in[j]
            on_path[j] = ok
    if forced:
        wrong_in = [j for j in range(len(records)) if inp[j] != ref_in[j]]
        cmp_idx = list(range(len(records)))
    else:
        wrong_in = []
        cmp_idx = [j for j in range(len(records)) if on_path[j]]
    bad = [records[j][:2] for j in cmp_idx if not eq[j]]
    ctrl = [records[j][:2] for j in cmp_idx if nx[j]]
    committed_ok = [rnd.reqs[b].out == [int(toks[s, b]) for s in range(steps)] for b in range(B)]
    ties = tie_summary(eng) if eng.tie_acc is not None else None
    if ties is not None:
        gates.append(GB.gate("R_%s_natural_ties" % name, True, False, ties["text"]))
    if negative is not None:                                               # SG4: detection is a REPORT
        gates.append(GB.gate("R_%s_detected" % name, True, False,
                             ("DETECTED (%s): %d of %d verify rows differ from REF, first (request, position) %s"
                              % (negative, len(bad), len(cmp_idx), bad[:3])) if bad else
                             ("BLIND (%s): every one of %d verify rows equals REF: the check could not see this "
                              "negative here" % (negative, len(cmp_idx)))))
        gates.append(GB.gate("%s_ran" % name, bool(cmp_idx) and all(committed_ok), True,
                             "the teacher-forced run completed: %d verify rows compared, committed == REF for %d of %d"
                             % (len(cmp_idx), sum(committed_ok), B)))
        res[name] = dict(negative=negative, compared=len(cmp_idx), differ=len(bad), ticks=rnd.ticks, ties=ties)
        del eng
        return
    must = a.gemm == "exact"
    gates.append(GB.gate("%s_verify_rows_equal_BI_serial" % name, not bad and not wrong_in and cmp_idx, must,
                         "%d of %d compared verify rows differ (first (request, position) %s); %d rows with a non-REF "
                         "input in the teacher-forced run; %d not compared (past REF), %d off REF's path"
                         % (len(bad), len(cmp_idx), bad[:4], len(wrong_in), skipped[0], len(records) - len(cmp_idx))))
    gates.append(GB.gate("%s_control_next_position_differs" % name, not ctrl and cmp_idx, True,
                         "%d of %d compared rows equal REF's row at the next position" % (len(ctrl), len(cmp_idx))))
    gates.append(GB.gate("%s_committed_equal_BI_serial" % name, all(committed_ok), must or forced,
                         "requests whose %d committed tokens equal REF's: %d of %d" % (steps, sum(committed_ok), B)))
    mirror = eng.mirror_failures()
    gates.append(GB.gate("%s_bookkeeping_twin_counters_match" % name, not mirror, True,
                         "%d (layer, bank) twin position counters differ from the host mirrors%s"
                         % (len(mirror), "; first %s" % mirror[:3] if mirror else "")))
    st = dict(rnd.stats)
    info = dict(ticks=rnd.ticks, sec=round(sec, 1), host_ms_per_tick=round(eng.stats["ms_rows"] / max(1, rnd.ticks), 1),
                rows=eng.stats["rows"], draft_rows=eng.stats["draft_rows"], verify_rows=eng.stats["verify_rows"],
                ticks_with_verify=sum(1 for d, v in per_tick if v), decision_reads=eng.stats["decision_reads"],
                round_syncs=rnd.syncs, rollbacks=eng.stats["rollbacks"], publications=eng.stats["publications"],
                writebacks=eng.stats["writebacks"], accepted=st["accepted"], rejected=st["rejected"],
                cohorts=st["cohorts"], stall_lead=st["stall_lead"], draft_drop=eng.draft_drop,
                rows_per_tick_max=max((d + v for d, v in per_tick), default=0))
    gates.append(GB.gate("%s_one_decision_read_per_verify_tick" % name,
                         info["decision_reads"] == info["ticks_with_verify"] == rnd.syncs, True,
                         "decision reads %d, ticks with verify rows %d, round syncs %d"
                         % (info["decision_reads"], info["ticks_with_verify"], rnd.syncs)))
    if ready is not None:                                                  # SG3: the phase mix
        classes = {}
        for j in cmp_idx:
            c = comps[j][:3]
            n_all, n_bad = classes.get(c, (0, 0))
            classes[c] = (n_all + 1, n_bad + (0 if eq[j] else 1))
        sizes = sorted({c[3] for c in comps})
        need = [any(c[0] for c in classes), any(not c[0] for c in classes), any(c[1] for c in classes),
                any(not c[1] for c in classes), len(sizes) >= 3]
        gates.append(GB.gate("SG3_precondition_compositions", all(need), True,
                             "verify rows WITH / WITHOUT their own draft row in the tick %s / %s, WITH / WITHOUT another "
                             "request's verify rows %s / %s, %d distinct tick sizes %s"
                             % (need[0], need[1], need[2], need[3], len(sizes), sizes[:12])))
        info_classes = {"own_draft=%s other_verify=%s other_draft=%s" % c: dict(rows=v[0], differ=v[1])
                        for c, v in sorted(classes.items())}
    if forced and name == "SG1":
        from nosi.r2_tick import schedule_coverage
        cov = schedule_coverage(sched, a.K)
        gates.append(GB.gate("SG1_precondition_schedule_coverage", cov["complete"], True,
                             "cycle rejection offsets %s (need 0 .. %d), full acceptances %d, boundary rollbacks keeping "
                             "the new block's first row %d / rolling back the whole new block %d, steered %d"
                             % (cov["offsets"], a.K - 1, cov["full"], cov["cross_new"], cov["cross_old"], cov["steer"])))
        gates.append(GB.gate("SG1_schedule_equals_its_simulation", sched == a.sim_log, True,
                             "the run's %d decisions %s the model-free simulation's %d (the bookkeeping is a function of "
                             "positions and counts only)" % (len(sched), "equal" if sched == a.sim_log else "DIFFER from",
                                                             len(a.sim_log))))
        info.update(schedule=cov)
    elif not forced:
        pre = info["rollbacks"] > 0 and info["publications"] > 0
        gates.append(GB.gate("SG2_precondition_rejects_and_publishes", pre, True,
                             "rollbacks %d, publications %d (a run that never rolls back is blind)"
                             % (info["rollbacks"], info["publications"])))
        info["acceptance_NOT_REPRESENTATIVE"] = round(st["accepted"] / max(1, st["accepted"] + st["rejected"]), 4)
    info["draft_stand_in"] = ("draft rows miss a hashed %g of their middle blocks (all-resident store): NO r, acceptance "
                              "or R-PERF number of this run is representative (C, Q2 2026-10-21)" % eng.draft_drop)
    if ready is not None:
        info["compositions"] = info_classes
    res[name] = info
    del eng


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        from nosi.r2_tick import schedule_coverage, simulate_schedule
        S_list = [a.S - d for d in a.deltas]
        a.sim_log, _ = simulate_schedule(S_list, [a.steps] * a.batch, K=a.K, LAMBDA=a.LAMBDA)
        cov = schedule_coverage(a.sim_log, a.K)
        if "SG1" in a.parts and not cov["complete"]:
            raise SystemExit("REFUSED: the SG1 schedule would be blind for these prompts and steps: %s" % cov)
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_bi_decode import BIOps, attach_bi
        from nosi.r2_decode_path import build_ragged
        from nosi.r2_tick import TickEngine
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1 (the BI serial decode is the R2 path's mode)")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be > 0 and NOSI_POOL_BLOCKS 0 (%d, %d)"
                             % (NL._ATTN_SPLITS, CE.POOL_BLOCKS))
        docs, sha = G0.load_workload(a.workload, a.batch, a.S)
        os.makedirs(a.out)
        attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
        attn_tick = select_attn(a, attn)                                   # [sharedkv] per_row: attn itself
        ops = BIOps(attn_tick, a.m_fix, gemm=a.gemm)
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s, allocated %.1f GB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
        gates, res = [], {}
        n_ref = a.steps + 16
        meta = dict(batch=a.batch, S=a.S, deltas=a.deltas, steps=a.steps, n_ref=n_ref, sg0_steps=a.sg0_steps, K=a.K,
                    LAMBDA=a.LAMBDA, draft_drop=a.draft_drop, seed=a.seed, m_fix=a.m_fix, gemm=a.gemm,
                    bi=ops.describe(), biattn=a.biattn, parts=a.parts, check=not a.no_check,
                    attn_splits=NL._ATTN_SPLITS, workload_sha256=sha, nosi_commit=os.environ.get("TICK_COMMIT", "unknown"),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    storage="ResidentStore: every committed block resident (the residency mover is SG5's dXFER); "
                            "valid for SG1-SG4 exactness, NOT for SG5 or any memory / transfer claim (C, Q1)")

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)

        prompts = [docs[b, :a.S - a.deltas[b]] for b in range(a.batch)]
        with torch.inference_mode():
            t0 = time.time()
            cache, st, first = build_ragged(model, prompts, n_ref + 80, f4_fix=True)
            base = TickEngine.from_r2(model, cache, st, ops, draft_drop=a.draft_drop, drop_seed=a.seed,
                                      check=not a.no_check)
            attach_bi(st, ops)
            log("build + BASE in %.0f s; REF: the BI serial decode, %d steps" % (time.time() - t0, n_ref))
            pos0 = st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)
            r = GB.decode_loop(model, cache, first, pos0, n_ref)
            ref = dict(logits=r["logits"], tokens=r["tokens"], tok_host=r["tokens"].cpu())
            res["REF"] = dict(steps=n_ref, ms_per_step=round(1e3 * r["sec"] / n_ref, 1), S_list=base.S_list)
            del cache, st, r
            GB.free()
            save()
            from types import SimpleNamespace
            from nosi.r2_tick import BatchKeyedSplit, phase_ready, topk_select
            negs = (("SG4a", "natural cuBLAS in the flattened call (gemm fast)",
                     dict(ops=BIOps(attn, a.m_fix, gemm="fast"))),
                    ("SG4b", "split-KV attention with a batch-keyed split size (8 pages for <= B rows, else 4)",
                     dict(ops=SimpleNamespace(linear=ops.linear, attn=BatchKeyedSplit(attn, a.batch), mean=ops.mean))),
                    ("SG4c", "torch.topk(sorted=False) selection (the shipped tie rule)", dict(select=topk_select)))
            runs = [("SG0", None), ("SG1", None), ("SG2", None), ("SG3", None)] + [("SG4", n) for n in negs]
            runs.append(("SG4", ("SG4c_planted", "planted ties", None)))
            for part, neg in runs:
                if part not in a.parts:
                    continue
                label = neg[0] if neg else part
                log(label)
                try:
                    if part == "SG0":
                        part_sg0(a, base, ref, gates, res)
                    elif part in ("SG1", "SG2"):
                        part_tick(a, base, ref, gates, res, forced=(part == "SG1"))
                    elif part == "SG3":
                        part_tick(a, base, ref, gates, res, forced=True, name="SG3", ready=phase_ready(a.phase))
                    elif neg[0] == "SG4c_planted":
                        part_planted(a, base, ref, gates, res)
                    else:
                        part_tick(a, base, ref, gates, res, forced=True, name=neg[0], eng_kw=neg[2],
                                  steps=a.sg4_steps, negative=neg[1])
                except Exception as e:
                    traceback.print_exc()
                    gates.append(GB.gate("PART_%s" % label, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])))
                GB.free()
                save()
        try:                                                               # [sharedkv]
            sharedkv_gates(a, attn_tick, gates, res)
        except Exception as e:
            traceback.print_exc()
            gates.append(GB.gate("PART_SKV", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])))
        meta["attn"] = a.attn
        meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_TICK_GATE_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
