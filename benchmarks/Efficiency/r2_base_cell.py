"""R2-M2 "r2base": the throughput cell of the CAPTURED R2 decode step against the SHIPPED decode, at the same B, in the
same process (benchmarks/Efficiency/r2_base_step.sh runs one B per step). Design: docs/R2_BASE.md.

WORKLOAD: the PG19 94-document 16128-token file (kstep-cpu-runs/workload/pg19-test-c16k, sha256 checked by
r2_ragged_gate.load_docs); request b reads row b % 94 (B > 94 repeats documents; the agreement is also reported over
the first min(B, 94) requests, every document once).

ARMS (in this order, one model):
  SH  the SHIPPED decode (nosa_llama.Llama.decode_inference, offload engine cache_engine.InfLLMv2Cache, the host rows
      capped at S + N + 64 by max_gen_len, as the sustained harness does), greedy, free-running, N steps, RE-WARMED
      where its warm-up-sized buffers require it (rewarm_need: a new 64-token block or a new 128-column stage-1 score
      width; the rule of the sustained harness, nosi-sustained a1c8a24 benchmarks/Efficiency/sustained_core.py:217-235).
      Its tokens are the reference sequence.
  R2  the R2 path (build_ragged: per-request prefill, widths preallocated from the horizon, no re-warm), TEACHER-FORCED
      on SH's input tokens: --warmup eager capture-safe steps (r2_capture.decode_step_fixed), then ONE captured step
      replayed for the rest. Inputs advance ON THE DEVICE (no host sync per step). Its argmax at every step is
      compared with SH's next token.

METRICS per arm (CUDA events; a step's period = start of step t to start of step t + 1):
  steady per-step ms   median period of the steady steps (SH: every step that is not a re-warm; R2: the replayed steps
                       after the capture step)
  steady decode tok/s  B / steady step; per-request rate = 1 / steady step (tok/s of one request)
  effective tok/s      B x N / (sum of all periods): includes SH's re-warms and R2's warm-up + capture
  GPU peak             torch.cuda.max_memory_allocated / reserved over the arm (prefill included); the allocated
                       memory at the arm's start (gpu_base_alloc_GB)
  prefill s            the arm's prefill wall time
  host memory          REPORT: VmRSS / VmHWM / VmLck / VmPin after the load, after SH, after the release of SH's
                       cached pinned blocks (free(host=True)), after R2 (meta.host_mem)
AGREEMENT (R2 vs SH): teacher-forced agreement over (request, step); the free-running FIRST DIVERGENCE per request =
its first teacher-forced disagreement (before it the two runs have equal inputs, so the teacher-forced run IS the
free-running run up to there); the prefill first tokens (REPORT: the logits GEMM runs at M = B in SH, M = 1 in R2's
per-request prefill).

MUST: SH-RUN and R2-RUN (N steps each), R2-ERR (no refusal bit), R2-PTR (no state data_ptr moves across the
replays). REPORT: everything else, and the registered predictions P1-P5 (docs/R2_BASE.md) with their verdicts.
EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside an arm.

--r2 fused (docs/R2_FUSED.md section 8): the R2 arm runs the FUSED twin step (r2_twin_fused.attach on step_fixed, a
DECLARED change: per-bank refusal through the overflow flag, the fixed-order mean) and the LEAN body
(r2_body_fused.FusedBody), attached before the warm-up, so the capture records them. R2-ERR then also requires every
twin's overflow flag clear. The predictions are P1F (agreement), P3F (steady ratio per B), P4F (effective ratio per B)
and P5. --gt (with --r2 fused): r2_fused_cell.part_gt first, before the model loads (its gates GT-EQ, GT-COVER,
GT-NEG, GT-NEG-ORDER are MUST gates of this cell).

--pool P (r2-mbase, the MATCHED BASELINE; paper-notes/recon-rows-2-3.md row 2): the cache capacity C = 63 + P, P the
victim pool of the offload engine (NOSI_POOL_BLOCKS, read once at import by cache_engine; the cell REFUSES a run whose
knob differs from --pool, so the record names its C). P > 0 runs the SH arm ONLY: the R2 decode body has no victim pool
(r2_decode_body.DecodeBody.allocate raises on pool_blocks), so a captured R2 / fused step at C > 63 does not exist yet.
The SH "steady" statistic excludes every re-warm (= re-capture of the shipped after-pooling graph) period, so the SH
steady step IS the shipped decode without the re-capture cost (the recon's definition; R2 fused / SH = 0.990 at C 63,
B 219, basef 2190876). --mbase selects the registered mbase predictions (verdicts_mbase, PRED_MBASE):
  MB-STEP  the SH steady step inside the band registered for (C, B) (REPORT at an unregistered pair);
  MB-FIT   the SH peak allocated within 5 % of the prefill-peak model mbase_peak_model(C, B) (2175791's output 2);
  MB-F     at C 63 with the R2 arm: R2 (fused) / SH steady inside (0.95, 1.00);
  MB-R2C   REPORT (a MODEL, not a measurement): SH steady x MBASE_FUSED_FACTOR, the fused step this C would have.
The cell also reports the device memory in use (cudaMemGetInfo) at the end of the SH arm, beside the allocator peak.

r2-mbstrong (docs/NOTES_mbstrong.md; every flag default off; the SH arm only, --arms SH):
  --recap-nogc  each SH re-warm runs ONE release (synchronize, gc.collect, empty_cache) and then captures its 32
                pooling + top-k graphs with r2_recap_nogc.GraphNoRelease (torch.cuda.graph without its release on
                entry). The engine files are not edited. MUST gate RN-GC: 1 gc.collect call and 32 GraphNoRelease
                contexts per re-warm.
  --ab-record   the A/B record ab_record.pt: every token, the logits digest of every re-warm step and of the last step.
                A plain run (no --recap-nogc) reports RN-GC (32 gc.collect calls per re-warm, the shipped contexts).
  --ab-ref DIR  DIR is the other half of the A/B pair (the same C, B, run and holder; the other mode). MUST gate RN-EQ.
  --fit-probe   an out-of-memory SH arm is a REPORT: SH-RUN is then not a MUST gate.
  With --recap-nogc or --ab-record the SH arm runs run_shipped_ab (the lines of run_shipped plus the instrument);
  without them it runs run_shipped. The batch arithmetic: fit_batch (the prefill-peak model mbase_peak_model and the
  int32 envelope of cache_engine.py:216-221). The registered predictions PRED_MBS (TBD by the lead) print UNREGISTERED
  until the lead stamps them.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_ragged_gate as RG        # noqa: E402

S0 = 16128
N_DOCS = 94
R_TOK, KERNEL, STRIDE, ROUND = 64, 32, 16, 128
PRED = dict(P3_ratio_b64=0.8, P3_ratio_b219=0.9, P4_eff_ratio_b64=1.5, P5_peak_ratio=1.15)
# --r2 fused (registered with the commit that adds them; basis: rb1 2190876 and M-F f1 2190876, docs/R2_FUSED.md 8)
PRED_FUSED = dict(P1F_agreement=0.999,
                  P3F_ratio={64: (0.95, 0.99), 128: (0.95, 1.00), 219: (0.95, 1.00)},
                  P4F_eff_ratio={64: (2.6, 3.2), 128: (2.05, 2.5), 219: (1.65, 1.95)})
# --mbase (registered with the commit that adds them; basis: job 2175791 = retroinfer-eval docs/evidence/
# hbm_sweep_2175791 output 2 (prefill peaks) and output 3 (eager medians), basef 2190876 (SH steady at C 63; P3F 0.990))
C_BASE = 63
POOL_MAX = 65
MBASE_PEAK = dict(intercept_GB=19.0, slope_c63_GB=0.155, slope_per_block_GB=0.002105, fit_GB=82.04, tol=0.05)
MBASE_FUSED_FACTOR = 0.990
MBASE_REF = dict(C=128, B=216, eager_median_ms=68.336639, tok_s=3160.822684, gain_q_tok_s=3703.54)
# SH steady step (ms): (point, low, high) per (C, B). C 128 / C 63: 2175791's eager median x [0.95, 1.03] / [0.97, 1.03]
# (our SH steady vs its eager at C 63: 0.9973 / 0.9956 / 1.0071 at B 64 / 128 / 219); C 79 / 96 / 113: the step model
# R(B) (2175791's C 63 resident step) + 1.26 us per load (64 x misses(C) per request, x 0.7-1.0) + the pool's copies.
PRED_MBASE = {(128, 216): (68.0, 64.92, 70.39), (113, 240): (74.5, 68.0, 81.0), (96, 280): (85.6, 78.0, 94.0),
              (79, 328): (106.0, 92.0, 118.0), (63, 344): (180.6, 174.5, 185.3)}
MBASE_F_BAND = (0.95, 1.00)
# r2-mbstrong: the batch arithmetic (fit_batch) and the registered predictions PRED_MBS.
INT32 = 2 ** 31
KV_HEADS, HEAD_DIM, BLOCK = 2, 128, 64              # NOSA-8B (config.json); block = topk block (nosa_llama.py:680)
MBS_BUDGET_GB = dict(matched=82.04, edge=84.0)      # matched: mb1's measured peak (2175791, dr1); edge: the 84 GB budget
# REGISTERED predictions: TBD. The lead writes every value and the stamp (`date -u` in the same command, L33) in a
# commit before the run. A None value prints UNREGISTERED. Forms: a band (low, high); a dict of bands keyed "C" or
# "C_B"; mx_fits a dict "C_B" -> True (fits) / False (out of memory).
PRED_MBS = dict(stamp=None,
                rn_steady_ratio=None,       # --recap-nogc / plain SH steady median, every A/B pair
                rn_eff_ratio=None,          # {"C": band}: --recap-nogc / plain effective tok/s
                rn_rewarm_later_ms=None,    # median of the later re-warm periods (ms) under --recap-nogc
                rn_rewarm_first_ms=None,    # the first re-warm period (ms) under --recap-nogc (the JIT included)
                mx_steady_tok_s=None,       # {"C_B": band}: SH steady tok/s of each sub-cell
                mx_peak_ratio=None,         # band: SH peak allocated / mbase_peak_model(C, B)
                mx_fits=None)               # {"C_B": bool}: the fit probes


def log(msg):
    print("[r2_base %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--arms", default="SH,R2")
    ap.add_argument("--r2", default="built", choices=("built", "fused"),
                    help="the R2 arm: as built (step_fixed, DeviceWritebackBody) or fused (docs/R2_FUSED.md 8)")
    ap.add_argument("--gt", action="store_true", help="with --r2 fused: r2_fused_cell.part_gt first (MUST gates)")
    ap.add_argument("--pool", type=int, default=0,
                    help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS; P > 0 needs --arms SH")
    ap.add_argument("--mbase", action="store_true", help="the registered mbase predictions (verdicts_mbase)")
    ap.add_argument("--recap-nogc", action="store_true",
                    help="r2-mbstrong: one release per SH re-warm, captures without the per-context release")
    ap.add_argument("--ab-record", action="store_true", help="r2-mbstrong: write the A/B record ab_record.pt")
    ap.add_argument("--ab-ref", default=None, help="r2-mbstrong: the other half of the A/B pair (MUST gate RN-EQ)")
    ap.add_argument("--fit-probe", action="store_true", help="r2-mbstrong: an out-of-memory SH arm is a REPORT")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if a.arms not in (["SH", "R2"], ["SH"]):
        ap.error("arms must be SH,R2 or SH (R2 is teacher-forced on SH's tokens)")
    if a.gt and a.r2 != "fused":
        ap.error("--gt goes with --r2 fused")
    if not 0 <= a.pool <= POOL_MAX:
        ap.error("--pool in 0 .. %d" % POOL_MAX)
    if a.pool > 0 and "R2" in a.arms:
        ap.error("--pool > 0 needs --arms SH (the R2 decode body has no victim pool)")
    if a.batch < 1 or not 1 <= a.warmup < a.steps or a.steps > 1000:
        ap.error("--batch >= 1, --warmup in 1 .. steps - 1, --steps <= 1000 (the shipped total_cis holds S + 1024)")
    if (a.recap_nogc or a.ab_record or a.fit_probe) and a.arms != ["SH"]:
        ap.error("--recap-nogc, --ab-record and --fit-probe need --arms SH")
    if a.ab_ref is not None and not a.ab_record:
        ap.error("--ab-ref needs --ab-record (RN-EQ compares two records)")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def doc_rows(B: int, n_docs: int = N_DOCS):
    return [b % n_docs for b in range(B)]


def round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def warm_state(seq: int, n_nocomp: int, m_comp: int):
    """What a warm-up at cache length seq fixes (sustained_core.py:217-221): out_len = ceil((seq + 1) / 64) and the
    stage-1 score width round128(m'), m' = m + (n >= KERNEL) (the step's own compression)."""
    mp = int(m_comp) + (1 if int(n_nocomp) >= KERNEL else 0)
    return dict(seq=int(seq), out_len=-(-(int(seq) + 1) // R_TOK), width=round_up(mp, ROUND), m_next=mp)


def rewarm_need(seq: int, n_nocomp: int, m_comp: int, warm):
    """(re-warm before this step?, why) (sustained_core.py:224-235): 'first', 'out_len', 'width', or '' (steady)."""
    if warm is None:
        return True, "first"
    now = warm_state(seq, n_nocomp, m_comp)
    if now["out_len"] != warm["out_len"]:
        return True, "out_len"
    if now["width"] != warm["width"]:
        return True, "width"
    return False, ""


def expected_rewarms(S: int, steps: int):
    """The shipped decode's re-warm steps for a prompt of S tokens over `steps` steps, from the cache lengths the
    shipped buffers follow (cache_engine.py:672-692: no_compress_k_len starts at 16 + S % 16, a compression when it
    reached 32 before the write; n_comp starts at S // 16 - 1)."""
    seq, n, m = S, STRIDE + S % STRIDE, S // STRIDE - 1
    warm, out = None, []
    for t in range(steps):
        need, why = rewarm_need(seq, n, m, warm)
        if need:
            warm = warm_state(seq, n, m)
            out.append((t, why))
        if n >= KERNEL:
            m, n = m + 1, STRIDE + 1
        else:
            n += 1
        seq += 1
    return out


def period_stats(periods, keep):
    xs = sorted(float(periods[i]) for i in keep)
    if not xs:
        return dict(n=0, median_ms=None, p10_ms=None, p90_ms=None, mean_ms=None)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
    return dict(n=len(xs), median_ms=round(statistics.median(xs), 3), p10_ms=round(q(0.1), 3), p90_ms=round(q(0.9), 3),
                mean_ms=round(statistics.fmean(xs), 3))


def rates(B: int, steps: int, periods, steady):
    total_ms = float(sum(periods))
    med = steady["median_ms"]
    return dict(steady_step_ms=med, steady_tok_s=(round(1e3 * B / med, 1) if med else None),
                per_request_tok_s=(round(1e3 / med, 3) if med else None), total_s=round(total_ms / 1e3, 3),
                effective_tok_s=round(1e3 * B * steps / total_ms, 1) if total_ms > 0 else None)


def agreement(pred, inputs, n_unique: int):
    """pred (B, N): R2's argmax at step t; inputs (B, N + 1): SH's tokens (inputs[:, t + 1] = SH's token after step t).
    The free-running first divergence of request b = its first teacher-forced disagreement (None: none in N steps)."""
    import torch
    ok = pred.detach().cpu() == inputs.detach().cpu()[:, 1:]
    Bn, Nn = ok.shape
    first = []
    for b in range(Bn):
        bad = (~ok[b]).nonzero()
        first.append(int(bad[0, 0]) if bad.numel() else None)
    div = sorted(f for f in first if f is not None)
    return dict(tf_agreement=float(ok.float().mean()), tf_agreement_unique_docs=float(ok[:n_unique].float().mean()),
                tf_disagreements=int((~ok).sum()), min_step_agreement=float(ok.float().mean(0).min()),
                n_requests=Bn, n_diverged=len(div), first_divergence_min=(div[0] if div else None),
                first_divergence_median=(div[len(div) // 2] if div else None), first_divergence=first,
                steps=Nn)


def verdicts(B: int, sh, r2, agree, expected):
    """The registered predictions P1-P5 (docs/R2_BASE.md) against this cell's numbers: [(name, verdict, detail)]."""
    out = []
    if agree is not None:
        ok = agree["n_diverged"] == 0
        out.append(("P1", "CONFIRMED" if ok else "REFUTED", "teacher-forced agreement %.6f, %d of %d requests diverged "
                    "(first at step %s)" % (agree["tf_agreement"], agree["n_diverged"], agree["n_requests"],
                                            agree["first_divergence_min"])))
    if sh is not None:
        got = [tuple(x) for x in sh["rewarms"]]
        ok = got == [tuple(x) for x in expected]
        out.append(("P2", "CONFIRMED" if ok else "REFUTED", "shipped re-warms %s (expected %s); R2 warm-up %s eager + 1 "
                    "capture" % (got, expected, r2["warmup"] if r2 else None)))
    if sh is not None and r2 is not None and sh["rates"]["steady_step_ms"] and r2["rates"]["steady_step_ms"]:
        ratio = r2["rates"]["steady_step_ms"] / sh["rates"]["steady_step_ms"]
        lim = PRED["P3_ratio_b64"] if B <= 64 else (PRED["P3_ratio_b219"] if B >= 219 else None)
        v = "REPORT" if lim is None else ("CONFIRMED" if ratio <= lim else "REFUTED")
        out.append(("P3", v, "R2 / SH steady step time %.3f (limit %s at B %d)" % (ratio, lim, B)))
        er = r2["rates"]["effective_tok_s"] / sh["rates"]["effective_tok_s"]
        v = ("CONFIRMED" if er >= PRED["P4_eff_ratio_b64"] else "REFUTED") if B == 64 else "REPORT"
        out.append(("P4", v, "R2 / SH effective tok/s %.3f (limit >= %s at B 64)" % (er, PRED["P4_eff_ratio_b64"])))
        pr = r2["gpu_peak_alloc_GB"] / sh["gpu_peak_alloc_GB"]
        out.append(("P5", "CONFIRMED" if pr <= PRED["P5_peak_ratio"] else "REFUTED",
                    "R2 / SH peak allocated %.3f (limit %s)" % (pr, PRED["P5_peak_ratio"])))
    return out


def verdicts_fused(B: int, sh, r2, agree, expected):
    """--r2 fused: P1F (teacher-forced agreement >= P1F_agreement), P2 (as verdicts), P3F / P4F (the steady and the
    effective R2 / SH ratios inside the band registered for B; REPORT at another B), P5 (as verdicts)."""
    out = []
    if agree is not None:
        lim = PRED_FUSED["P1F_agreement"]
        out.append(("P1F", "CONFIRMED" if agree["tf_agreement"] >= lim else "REFUTED",
                    "teacher-forced agreement %.6f (>= %s), %d of %d requests diverged (first at step %s)" % (
                        agree["tf_agreement"], lim, agree["n_diverged"], agree["n_requests"],
                        agree["first_divergence_min"])))
    base = {p[0]: p for p in verdicts(B, sh, r2, None, expected)}
    if "P2" in base:
        out.append(base["P2"])
    if sh is not None and r2 is not None and sh["rates"]["steady_step_ms"] and r2["rates"]["steady_step_ms"]:
        for name, key, val in (("P3F", "P3F_ratio", r2["rates"]["steady_step_ms"] / sh["rates"]["steady_step_ms"]),
                               ("P4F", "P4F_eff_ratio",
                                r2["rates"]["effective_tok_s"] / sh["rates"]["effective_tok_s"])):
            band = PRED_FUSED[key].get(B)
            v = "REPORT" if band is None else ("CONFIRMED" if band[0] <= val <= band[1] else "REFUTED")
            what = "steady step time" if name == "P3F" else "effective tok/s"
            out.append((name, v, "R2 (fused) / SH %s %.3f (band %s at B %d)" % (what, val, band, B)))
        if "P5" in base:
            out.append(base["P5"])
    return out


def mbase_peak_model(C: int, B: int) -> float:
    """The prefill-peak model (GB allocated) of the offload engine at capacity C and batch B, fitted on 2175791's
    output 2: C 63 B 64 28.93, C 128 B 64 37.69 / B 216 82.04 -> 19.0 + B x (0.155 + (C - 63) x 0.002105). Check
    (outside the fit): basef 2190876 SH C 63 B 219 peak 52.925 (model 52.945)."""
    p = MBASE_PEAK
    return p["intercept_GB"] + int(B) * (p["slope_c63_GB"] + (int(C) - C_BASE) * p["slope_per_block_GB"])


def mbase_batch(C: int) -> int:
    """The registered batch of capacity C: the largest multiple of 8 whose modeled peak is <= fit_GB (82.04, the
    largest peak that fitted in 2175791). C 128 -> 216 (2175791: 216 fits, 224 OOM); C 63 -> 400 (the sweep uses 344,
    2175791's measured point, instead: see the step script)."""
    B = 8
    while mbase_peak_model(C, B + 8) <= MBASE_PEAK["fit_GB"]:
        B += 8
    return B


def verdicts_mbase(C: int, B: int, sh, r2):
    """--mbase: MB-STEP, MB-FIT, MB-F (C 63 with R2 only), MB-R2C (REPORT, a model) and the tok/s REPORT against
    the forecast reference and the gain threshold Q (paper-notes/recon-rows-2-3.md)."""
    out = []
    if sh is None or not sh["rates"]["steady_step_ms"]:
        return [("MB-STEP", "NOT_RUN", "no SH steady step")]
    ms = sh["rates"]["steady_step_ms"]
    reg = PRED_MBASE.get((int(C), int(B)))
    if reg is None:
        out.append(("MB-STEP", "REPORT", "SH steady %.3f ms at C %d B %d (no registered band)" % (ms, C, B)))
    else:
        pt, lo, hi = reg
        out.append(("MB-STEP", "CONFIRMED" if lo <= ms <= hi else "REFUTED",
                    "SH steady %.3f ms at C %d B %d (point %s, band [%s, %s])" % (ms, C, B, pt, lo, hi)))
    model = mbase_peak_model(C, B)
    r = sh["gpu_peak_alloc_GB"] / model
    out.append(("MB-FIT", "CONFIRMED" if abs(r - 1) <= MBASE_PEAK["tol"] else "REFUTED",
                "SH peak allocated %.3f GB / model %.3f GB = %.4f (tolerance %s)" % (
                    sh["gpu_peak_alloc_GB"], model, r, MBASE_PEAK["tol"])))
    if r2 is not None and r2["rates"]["steady_step_ms"]:
        f = r2["rates"]["steady_step_ms"] / ms
        lo, hi = MBASE_F_BAND
        out.append(("MB-F", ("CONFIRMED" if lo <= f <= hi else "REFUTED") if int(C) == C_BASE else "REPORT",
                    "R2 / SH steady %.4f (band %s at C 63)" % (f, MBASE_F_BAND)))
    model_ms = ms * MBASE_FUSED_FACTOR
    out.append(("MB-R2C", "REPORT", "MODEL: SH steady x %.3f = %.3f ms, %.1f tok/s (a C 63 ratio moved to C %d)" % (
        MBASE_FUSED_FACTOR, model_ms, 1e3 * B / model_ms, C)))
    tok = 1e3 * B / ms
    out.append(("MB-TOK", "REPORT", "SH steady %.1f tok/s at C %d B %d: / reference %.2f = %.4f; Q %.2f / this = %.4f "
                "(gain %+.2f %% if this is the matched reference)" % (
                    tok, C, B, MBASE_REF["tok_s"], tok / MBASE_REF["tok_s"], MBASE_REF["gain_q_tok_s"],
                    MBASE_REF["gain_q_tok_s"] / tok, 100 * (MBASE_REF["gain_q_tok_s"] / tok - 1))))
    return out


# ------------------------------------------------------------------ r2-mbstrong pure parts (CPU-tested)
def int32_batch_limit(S: int, steps: int, C: int) -> int:
    """The largest B that the int32 envelope of the pooled engine accepts (cache_engine.py:216-221, asserted when
    P > 0): host side B x (S + max_gen_len) x H x D < 2^31, max_gen_len = steps + 64 (run_shipped); GPU side
    B x (topk + P) x block x H x D < 2^31, topk + P = C + 1 (C = 63 + P). At P = 0 the engine does not assert, and the
    gathers widen their program ids (flash_h2d_mask.py:23-27); the limit is applied there too (it does not bind below
    B 502 at S 16128, 512 steps)."""
    hd = KV_HEADS * HEAD_DIM
    host = (INT32 - 1) // ((int(S) + int(steps) + 64) * hd)
    gpu = (INT32 - 1) // ((int(C) + 1) * BLOCK * hd)
    return min(host, gpu)


def fit_batch(C: int, budget_GB: float, S: int = S0, steps: int = 512) -> int:
    """The largest B whose prefill-peak model mbase_peak_model(C, B) is <= budget_GB, capped by int32_batch_limit.
    The model is fitted at S 16128 only (2175791), so another S is refused."""
    if int(S) != S0:
        raise ValueError("fit_batch: the peak model holds at S %d only (got %d)" % (S0, S))
    p = MBASE_PEAK
    slope = p["slope_c63_GB"] + (int(C) - C_BASE) * p["slope_per_block_GB"]
    B = max(0, int((budget_GB - p["intercept_GB"]) / slope))
    while B > 0 and mbase_peak_model(C, B) > budget_GB:
        B -= 1
    while mbase_peak_model(C, B + 1) <= budget_GB:
        B += 1
    return min(B, int32_batch_limit(S, steps, C))


def rewarm_stats(periods, rewarms) -> dict:
    """The re-warm (re-capture) periods: the first (it holds the one-time JIT work), the later ones, the sum."""
    xs = [float(periods[t]) for t, _ in rewarms]
    later = xs[1:]
    return dict(n=len(xs), first_ms=round(xs[0], 1) if xs else None,
                later_median_ms=round(statistics.median(later), 1) if later else None,
                later_mean_ms=round(statistics.fmean(later), 1) if later else None, sum_ms=round(sum(xs), 1))


def effective_excl_first(B: int, periods, rewarms):
    """REPORT: effective tok/s without step 0 when step 0 is the first re-warm (its one-time JIT work)."""
    if not rewarms or rewarms[0][0] != 0 or len(periods) < 2:
        return None
    return round(1e3 * B * (len(periods) - 1) / float(sum(periods[1:])), 1)


def is_oom(e) -> bool:
    """An out-of-memory exception: the caching allocator's (torch.cuda.OutOfMemoryError) or a CUDA driver error."""
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _pred(name, val, band, what):
    if val is None:
        return (name, "NOT_RUN", what + ": no value")
    if band is None:
        return (name, "UNREGISTERED", "%s %.4f" % (what, val))
    lo, hi = band
    return (name, "CONFIRMED" if lo <= val <= hi else "REFUTED", "%s %.4f (band [%s, %s])" % (what, val, lo, hi))


def mbs_verdicts(C: int, B: int, sh, nogc: bool, ref_sh, fit, pred=None):
    """The PRED_MBS predictions of one sub-cell: RN-* (the A/B pair: ref_sh is the other half's SH summary; the
    ratios are --recap-nogc / plain), RN-RW-* (--recap-nogc re-warm periods), MX-* (steady tok/s, peak / model, fit)."""
    pred = PRED_MBS if pred is None else pred
    key = "%d_%d" % (C, B)
    out = []
    pick = lambda d, k: None if d is None else d.get(k)   # noqa: E731
    if fit is not None:
        want = pick(pred["mx_fits"], key)
        v = "UNREGISTERED" if want is None else ("CONFIRMED" if want == (fit == "fits") else "REFUTED")
        out.append(("MX-FIT", v, "C %d B %d: %s (registered %s)" % (C, B, fit, want)))
    if sh is None or not sh["rates"]["steady_step_ms"]:
        return out
    out.append(_pred("MX-TOK", sh["rates"]["steady_tok_s"], pick(pred["mx_steady_tok_s"], key),
                     "SH steady tok/s at C %d B %d" % (C, B)))
    out.append(_pred("MX-PEAK", sh["gpu_peak_alloc_GB"] / mbase_peak_model(C, B), pred["mx_peak_ratio"],
                     "SH peak allocated / model at C %d B %d" % (C, B)))
    rs = sh.get("rewarm_stats") or {}
    if nogc:
        out.append(_pred("RN-RW-LATER", rs.get("later_median_ms"), pred["rn_rewarm_later_ms"],
                         "--recap-nogc later re-warm median ms"))
        out.append(_pred("RN-RW-FIRST", rs.get("first_ms"), pred["rn_rewarm_first_ms"],
                         "--recap-nogc first re-warm ms"))
    if ref_sh is not None and ref_sh["rates"]["steady_step_ms"]:
        x, y = (sh, ref_sh) if nogc else (ref_sh, sh)            # x: --recap-nogc, y: plain
        out.append(_pred("RN-STEADY", x["rates"]["steady_step_ms"] / y["rates"]["steady_step_ms"],
                         pred["rn_steady_ratio"], "--recap-nogc / plain SH steady median"))
        out.append(_pred("RN-EFF", x["rates"]["effective_tok_s"] / y["rates"]["effective_tok_s"],
                         pick(pred["rn_eff_ratio"], str(C)), "--recap-nogc / plain effective tok/s at C %d" % C))
    return out


class CudaClock:
    def event(self):
        import torch
        return torch.cuda.Event(enable_timing=True)

    def record(self, e):
        e.record()

    def elapsed(self, a, b):
        return a.elapsed_time(b)

    def sync(self):
        import torch
        torch.cuda.synchronize()


class HostClock:
    def event(self):
        return [0.0]

    def record(self, e):
        e[0] = time.perf_counter()

    def elapsed(self, a, b):
        return (b[0] - a[0]) * 1e3

    def sync(self):
        pass


def r2_teacher_forced(RC, model, st, cb, cache, inputs, warmup: int, use_graph: bool, clock):
    """The R2 arm's decode loop (r2_capture module RC, all requests active): step t reads SH's input inputs[:, t] and
    the RoPE position S_b + t, both advanced ON THE DEVICE; steps < warmup run decode_step_fixed eagerly, then (with
    use_graph) ONE captured step is replayed. Returns the predictions (B, N), the periods (ms), the capture time and
    the refusal bits / pointer check."""
    import torch
    Bn, N1 = inputs.shape
    N = N1 - 1
    pred = torch.zeros((Bn, N), dtype=torch.int64, device=inputs.device)
    st.active_host.fill_(True)
    st.active.fill_(True)
    cb.pos_ids.copy_(st.pos_host.view(-1, 1))
    ev = [clock.event() for _ in range(N + 1)]
    graph, ptr0, capture_ms = None, None, None
    for t in range(N):
        clock.record(ev[t])
        cb.ids.copy_(inputs[:, t:t + 1])
        if use_graph and t >= warmup:
            if graph is None:
                clock.sync()
                t0 = time.perf_counter()
                ptr0 = RC.state_pointers(st, cb, cache)
                graph = RC.capture_step(model, st, cb, cache)
                clock.sync()
                capture_ms = (time.perf_counter() - t0) * 1e3
            graph.replay()
        else:
            RC.decode_step_fixed(model, st, cb, cache)
        pred[:, t].copy_(cb.logits[:, -1].argmax(-1))
        cb.pos_ids.add_(1)
        st.pos_host += 1
        st.steps += 1
    clock.record(ev[N])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(N)]
    body_err, twin_err = RC.errors(st, cb)
    moved = None
    if ptr0 is not None:
        after = RC.state_pointers(st, cb, cache)
        moved = sorted(k for k in set(ptr0) | set(after) if ptr0.get(k) != after.get(k))
    return dict(pred=pred, periods=periods, capture_ms=capture_ms, body_err=body_err, twin_err=twin_err,
                pointers_changed=moved, pointers_checked=(len(ptr0) if ptr0 is not None else None), graph=graph)


# ------------------------------------------------------------------ the arms (GPU)
def run_shipped(model, ids, steps: int, clock):
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device="cuda")
    inputs = torch.empty((Bn, steps + 1), dtype=torch.int64, device="cuda")
    inputs[:, 0:1].copy_(tok)
    model.has_buffers = False
    warm, rewarms = None, []
    ev = [clock.event() for _ in range(steps + 1)]
    for t in range(steps):
        l0 = cache.layers[0]
        need, why = rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
        if need:
            warm = warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            rewarms.append((t, why))
            model.has_buffers = False
        clock.record(ev[t])
        lg = model.decode_inference(tok, cu, pos, cache)
        tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        inputs[:, t + 1:t + 2].copy_(tok)
        pos = pos + 1
    clock.record(ev[steps])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(steps)]
    fr, tot = torch.cuda.mem_get_info()                  # REPORT: device memory in use (context + reserved) at the end
    del cache
    return dict(inputs=inputs, periods=periods, rewarms=rewarms, prefill_s=round(prefill_s, 1),
                device_used_end_GB=round((tot - fr) / 1e9, 3))


def run_shipped_ab(model, ids, steps: int, clock, ab):
    """r2-mbstrong (--recap-nogc / --ab-record): run_shipped with the instrument ab (r2_recap_nogc.ABRecorder). The
    prefill lines are the lines of run_shipped; the decode loop is shipped_loop_ab. Without the flags main calls
    run_shipped. The CPU test MBS1 checks both by AST against run_shipped."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    out = shipped_loop_ab(model, cache, tok, pos, Bn, steps, clock, ab, "cuda")
    fr, tot = torch.cuda.mem_get_info()                  # REPORT: device memory in use (context + reserved) at the end
    del cache
    out.update(prefill_s=round(prefill_s, 1), device_used_end_GB=round((tot - fr) / 1e9, 3))
    return out


def shipped_loop_ab(model, cache, tok, pos, Bn: int, steps: int, clock, ab, device):
    """The decode loop of run_shipped (the same statements in the same order) with three additions, all through ab:
    `with ab.step(t, need):` around the decode call, `ab.after(t, need, lg)` after it, `ab.finish(lg)` after the timed
    loop. device: "cuda" (run_shipped_ab) or "cpu" (the CPU test)."""
    import torch
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device=device)
    inputs = torch.empty((Bn, steps + 1), dtype=torch.int64, device=device)
    inputs[:, 0:1].copy_(tok)
    model.has_buffers = False
    warm, rewarms = None, []
    ev = [clock.event() for _ in range(steps + 1)]
    for t in range(steps):
        l0 = cache.layers[0]
        need, why = rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
        if need:
            warm = warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            rewarms.append((t, why))
            model.has_buffers = False
        clock.record(ev[t])
        with ab.step(t, need):
            lg = model.decode_inference(tok, cu, pos, cache)
        ab.after(t, need, lg)
        tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        inputs[:, t + 1:t + 2].copy_(tok)
        pos = pos + 1
    clock.record(ev[steps])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(steps)]
    ab.finish(lg)
    return dict(inputs=inputs, periods=periods, rewarms=rewarms, ab=ab.result())


def run_r2(model, ids, inputs, steps: int, warmup: int, clock, mode: str = "built"):
    import torch
    from nosi import r2_capture as RC
    from nosi.r2_decode_path import build_ragged
    Bn = ids.shape[0]
    prompts = [ids[b].cpu() for b in range(Bn)]
    clock.sync()
    t0 = time.time()
    cache, st, first = build_ragged(model, prompts, steps, f4_fix=False)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((first.view(-1) == inputs[:, 0]).sum())
    kernels, wb = RC.default_kernels()
    cb = RC.CaptureBuffers(model, st, False, kernels, wb)
    if mode == "fused":                                  # before the warm-up: the capture records the fused kernels
        from nosi import r2_twin_fused as FU
        from nosi.r2_body_fused import FusedBody
        for t in st.twins:
            FU.attach(t, methods=("step_fixed",), require_fixed_order=False)
        cb.body = FusedBody(False, wb)
    res = r2_teacher_forced(RC, model, st, cb, cache, inputs, warmup, True, clock)
    res.update(prefill_s=round(prefill_s, 1), first_token_agreement=first_agree, widths=st.meta.get("widths"),
               mode=mode, overflow=[int(t.overflow) for t in st.twins] if mode == "fused" else None)
    del res["graph"], cb, st, cache
    return res


def gpu_peak():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 3), round(torch.cuda.max_memory_reserved() / 1e9, 3)


def host_mem():
    """REPORT: this process's host memory (GB) from /proc/self/status. Never raises."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith(("VmRSS", "VmHWM", "VmLck", "VmPin")):
                    k, v = ln.split(":", 1)
                    out[k + "_GB"] = round(int(v.split()[0]) / 1048576, 1)
    except Exception as e:   # the instrument never stops the cell
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def free(host: bool = False):
    """Collect the arm's garbage (an InfLLMv2Cache can be cyclic), then release the CUDA caches. host=True also
    releases PyTorch's CACHED PINNED HOST blocks (torch._C._host_emptyCache): the two arms allocate the same host-row
    shape, and the caching host allocator may round each block up to a power of two (about 137 GB per arm at B 128
    and 219), so the shipped arm's blocks must not stay cached beside the R2 arm's. Returns whether it ran."""
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    ran = False
    if host and hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
        ran = True
    torch.cuda.reset_peak_memory_stats()
    return ran


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 20
    import torch
    import nosi.cache_engine as CEN
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != a.pool:
        print("REFUSED: needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS == --pool (%d; the knob is %d)" % (
            a.pool, CEN.POOL_BLOCKS), file=sys.stderr)
        return 20
    ab_on = a.recap_nogc or a.ab_record                   # r2-mbstrong (flag off: both False)
    mbs_on = ab_on or a.fit_probe
    contract, ab_ref, ab_rec, cmp, fit = None, None, None, None, None
    if ab_on:                                            # refuse before the model loads
        import r2_recap_nogc as RN
        try:
            contract = RN.check_contract()
            ab_ref = RN.load_ref(a.ab_ref) if a.ab_ref is not None else None
        except Exception as e:
            print("REFUSED: r2-mbstrong: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return 20
    try:
        docs, sha = RG.load_docs(a.workload, N_DOCS, S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, r2, agree, gt = [], None, None, None, {}
    if a.gt:                                             # the fused twin's GPU gate, before the model loads
        log("GT (r2_fused_cell.part_gt)")
        try:
            import r2_fused_cell as FC
            FC.part_gt(gates, gt)
        except Exception as e:
            traceback.print_exc()
            gates.append(("GT-RAN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        free()
    rows = doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    expected = expected_rewarms(S0, a.steps)
    mem = dict(after_load=host_mem())
    with torch.inference_mode():
        free()
        sh_base = round(torch.cuda.memory_allocated() / 1e9, 3)
        log("arm SH (B %d, %d steps); host %s" % (a.batch, a.steps, mem["after_load"]))
        try:
            if ab_on:
                ab = RN.ABRecorder(a.recap_nogc, a.ab_record, model.num_layers)
                out = run_shipped_ab(model, ids, a.steps, clock, ab)
            else:
                out = run_shipped(model, ids, a.steps, clock)
            pa, pr = gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(rewarms=out["rewarms"], prefill_s=out["prefill_s"], steady=steady,
                      rates=rates(a.batch, a.steps, out["periods"], steady), gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, gpu_base_alloc_GB=sh_base,
                      device_used_end_GB=out["device_used_end_GB"],
                      rewarm_ms=[round(out["periods"][t], 1) for t, _ in out["rewarms"]])
            if mbs_on:
                sh["rewarm_stats"] = rewarm_stats(out["periods"], out["rewarms"])
                sh["rates"]["effective_tok_s_excl_first"] = effective_excl_first(a.batch, out["periods"],
                                                                                 out["rewarms"])
                fit = "fits" if a.fit_probe else None
            if ab_on:
                sh["ab"] = out["ab"]
                if a.ab_record:
                    ab_rec = dict(tokens=out["inputs"].detach().cpu(), rewarms=[list(x) for x in out["rewarms"]],
                                  nogc=a.recap_nogc, **ab.record_tensors())
            inputs = out["inputs"]
            del out
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(sh["rewarms"]))))
        except Exception as e:
            traceback.print_exc()
            if a.fit_probe and is_oom(e):
                fit = "oom"
                gates.append(("SH-RUN", False, False, "OUT OF MEMORY at C %d B %d (--fit-probe: a REPORT, not a MUST): "
                              "%s: %s" % (C_BASE + a.pool, a.batch, type(e).__name__, str(e)[:300])))
            else:
                gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            inputs = None
        mem["after_SH"] = host_mem()
        mem["host_cache_released_after_SH"] = free(host=True)
        mem["after_SH_release"] = host_mem()
        log("host after SH %s, after the release %s" % (mem["after_SH"], mem["after_SH_release"]))
        if "R2" in a.arms:
            if inputs is None:
                gates.append(("R2-RUN", True, False, "NOT_RUN: SH failed (R2 is teacher-forced on SH's tokens)"))
            else:
                log("arm R2")
                r2_base = round(torch.cuda.memory_allocated() / 1e9, 3)
                try:
                    res = run_r2(model, ids, inputs, a.steps, a.warmup, clock, a.r2)
                    pa, pr = gpu_peak()
                    steady = period_stats(res["periods"], list(range(a.warmup + 1, a.steps)))
                    r2 = dict(prefill_s=res["prefill_s"], warmup=a.warmup, capture_ms=res["capture_ms"], steady=steady,
                              rates=rates(a.batch, a.steps, res["periods"], steady), gpu_peak_alloc_GB=pa,
                              gpu_peak_reserved_GB=pr, gpu_base_alloc_GB=r2_base,
                              first_token_agreement=res["first_token_agreement"],
                              widths=res["widths"], body_err=res["body_err"], twin_err=res["twin_err"],
                              mode=res["mode"], overflow=res["overflow"])
                    agree = agreement(res["pred"], inputs, min(a.batch, N_DOCS))
                    gates.append(("R2-RUN", True, True, "%d steps (R2 %s)" % (a.steps, res["mode"])))
                    errs = res["body_err"] or any(res["twin_err"]) or any(res["overflow"] or [])
                    gates.append(("R2-ERR", True, not errs, "refusal bits body %s twins %s; overflow flags %s" % (
                        res["body_err"], sorted(set(res["twin_err"])),
                        sorted(set(res["overflow"])) if res["overflow"] is not None else "n/a (built)")))
                    ch = res["pointers_changed"]
                    gates.append(("R2-PTR", True, ch == [], "%s of %s state pointers moved: %s" % (
                        len(ch) if ch is not None else "?", res["pointers_checked"], (ch or [])[:10])))
                except Exception as e:
                    traceback.print_exc()
                    gates.append(("R2-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_R2"] = host_mem()
            free()
    preds = (verdicts_fused if a.r2 == "fused" else verdicts)(a.batch, sh, r2, agree, expected)
    C = C_BASE + a.pool
    if a.mbase:
        preds = preds + verdicts_mbase(C, a.batch, sh, r2)
    if mbs_on:                                           # r2-mbstrong gates and predictions (flag off: none)
        rec_meta = dict(batch=a.batch, steps=a.steps, S0=S0, capacity_C=C, pool_blocks=CEN.POOL_BLOCKS,
                        attn_splits=NL._ATTN_SPLITS, workload_sha256=sha, doc_rows=rows)
        if ab_on:
            gates.append(("RN-CONTRACT", True, True, "torch %s: graph.__enter__ %s; GraphNoRelease.__enter__ %s" % (
                contract["torch"], contract["graph_enter"], contract["no_release_enter"])))
            if sh is not None:
                ok, det = RN.rn_gc_gate(sh["ab"], [t for t, _ in sh["rewarms"]])
                gates.append(("RN-GC", a.recap_nogc, ok, det))
        if ab_rec is not None:
            torch.save(dict(ab_rec, meta=rec_meta), os.path.join(a.out, "ab_record.pt"))
        if ab_ref is not None:
            if ab_rec is None:
                gates.append(("RN-EQ", True, False, "NOT_RUN: the SH arm did not finish"))
            else:
                cmp = RN.compare_ab(ab_ref["record"], dict(ab_rec, meta=rec_meta))
                gates.append(("RN-EQ", True, cmp["passed"], "vs %s: meta differs %s; modes (ref, this) %s; tokens differ "
                              "%s (first [b, t] %s); re-warm steps equal %s; digest keys ok %s; digests differ at %s; "
                              "last-step digest equal %s" % (
                                  a.ab_ref, cmp["meta_diff"], cmp["modes"], cmp["token_mismatches"],
                                  cmp["first_token_mismatch"], cmp["rewarms_equal"], cmp["digest_keys_ok"],
                                  cmp["digest_mismatch_steps"], cmp["final_equal"])))
        preds = preds + mbs_verdicts(C, a.batch, sh, a.recap_nogc,
                                     ab_ref["summary"]["SH"] if ab_ref is not None else None, fit)
    meta = dict(batch=a.batch, steps=a.steps, warmup=a.warmup, S0=S0, doc_rows=rows, n_unique_docs=min(a.batch, N_DOCS),
                workload_sha256=sha, expected_rewarms=expected, attn_splits=NL._ATTN_SPLITS,
                pool_blocks=CEN.POOL_BLOCKS, capacity_C=C, arms=a.arms, mbase=a.mbase,
                mbase_predictions=(dict(steady_ms={"%d_%d" % k: v for k, v in PRED_MBASE.items()}, peak=MBASE_PEAK,
                                        fused_factor=MBASE_FUSED_FACTOR, f_band=MBASE_F_BAND, reference=MBASE_REF,
                                        peak_model_GB=round(mbase_peak_model(C, a.batch), 3))
                                   if a.mbase else None),
                predictions=(PRED_FUSED if a.r2 == "fused" else PRED), r2=a.r2, gt=a.gt,
                nosi_commit=os.environ.get("BASE_COMMIT", "unknown"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), host_mem=mem)
    if mbs_on:
        meta.update(recap_nogc=a.recap_nogc, ab_record=a.ab_record, ab_ref=a.ab_ref, fit_probe=a.fit_probe, fit=fit,
                    graph_contract=contract, mbs_predictions=PRED_MBS,
                    fit_model=dict(peak_model_GB=round(mbase_peak_model(C, a.batch), 3), budgets_GB=MBS_BUDGET_GB,
                                   fit_batch={k: fit_batch(C, v, S0, a.steps) for k, v in MBS_BUDGET_GB.items()},
                                   int32_batch_limit=int32_batch_limit(S0, a.steps, C)))
    out = dict(meta=meta, SH=sh, R2=r2, agreement=agree, GT=gt or None,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    if mbs_on and cmp is not None:
        out["AB"] = cmp
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:300]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for name, arm in (("SH", sh), ("R2", r2)):
        if arm:
            print("REPORT %s B %d: steady %s ms/step, %s tok/s, %s tok/s per request, effective %s tok/s, peak %s GB, "
                  "prefill %s s" % (name, a.batch, arm["rates"]["steady_step_ms"], arm["rates"]["steady_tok_s"],
                                    arm["rates"]["per_request_tok_s"], arm["rates"]["effective_tok_s"],
                                    arm["gpu_peak_alloc_GB"], arm["prefill_s"]))
    if agree:
        print("REPORT agreement: teacher-forced %.6f, diverged %d of %d (first at %s)" % (
            agree["tf_agreement"], agree["n_diverged"], agree["n_requests"], agree["first_divergence_min"]))
    if mbs_on:
        rs = (sh or {}).get("rewarm_stats") or {}
        print("REPORT RN C %d B %d %s fit %s: steady median %s mean %s ms; effective %s tok/s (without step 0: %s); "
              "re-warm first %s ms, later median %s ms, sum %s ms (n %s); peak %s GB (model %.3f)" % (
                  C, a.batch, "recap-nogc" if a.recap_nogc else "plain", fit,
                  sh["steady"]["median_ms"] if sh else None, sh["steady"]["mean_ms"] if sh else None,
                  sh["rates"]["effective_tok_s"] if sh else None,
                  sh["rates"].get("effective_tok_s_excl_first") if sh else None, rs.get("first_ms"),
                  rs.get("later_median_ms"), rs.get("sum_ms"), rs.get("n"), sh["gpu_peak_alloc_GB"] if sh else None,
                  mbase_peak_model(C, a.batch)))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_base: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
