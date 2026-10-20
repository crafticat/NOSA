"""R2 step 7, G0-BI: the BATCH-INVARIANT SERIAL DECODE (nosi/nosi/r2_bi_decode.py, a mode of the R2 path) on the GPU:
its determinism and batch invariance (MUST) and its QUALITY against the shipped decode (REPORT: what BI costs in
fidelity). docs/R2_STEP7_SPEC_ROUND.md sections 3.5 and 7.2. One model, arms run one after another in one process.

BI OPS: the GEMM mode is a run parameter (--gemm): exact (default; G-b PaddedLinear(--m-fix) for wqkv / wo / gate_up
/ down / lm_head, the decision of note 3.3), fast (natural cuBLAS, the weaker claim: I1 / I2 become REPORT) or bigemm
(G-a: --bigemm PATH --bigemm-fn NAME, by path); piece 3's paged attention (--biattn PATH, a pinned copy; v2 7c826c1);
the deterministic selection; the fixed-order twin means.

PART C (continuous; --steps-c >= 256 decode steps; B documents, all S tokens; build_ragged: one prefill per request,
the R2 path's own widths from the horizon, so findings F1 / F3 do not stop it):
  REF  the R2 path, body r2, f4_fix False: the shipped ops in the shipped order (F4 reproduced), bitwise the shipped
       decode inside the F1 window (G0F PASS, holder 2188818). Greedy, free-running.
  E    BI, teacher-forced on REF's tokens (every step sees REF's prefix).
  E2   E again from a new build: MUST torch.equal to E at every step (determinism).
  I1   BI on subset_state(rows = [0]) of E's own build (request 0 ALONE, taken before E's first step), teacher-forced:
       MUST torch.equal to E's row 0 at every step (batch invariance: B = 1 vs B).
  I2   BI on subset_state(rows = B-1 .. 0) (the batch REVERSED), teacher-forced: MUST torch.equal to E's rows in that
       order at every step (invariance to the row index and the neighbours).
  FRC  BI, greedy, free-running.
  MUST P_C: the first tokens (the prefill argmax) of every build are equal.
PART NI (negative control of I1, --ni-steps steps): E and I1 with the GEMMs at the NATURAL row count (F.linear, cuBLAS
  at M = B and M = 1): request 0 must differ somewhere (detection REPORT; "blind" if it does not).
PART W (windows, against the REAL shipped decode): for w = 0 .. --windows - 1 the prefix S_w = S - 64 (windows - 1 - w)
  tokens of each document (0 mod 64: every window is a full F1 window of 64 valid shipped steps), batch prefill:
  SH_w (the shipped decode, NOSI_ATTN_SPLITS fixed, greedy), TF_w (BI attached to its own prefill, teacher-forced on
  SH_w's tokens), FR_w (BI, free-running). MUST P_w: the three prefill logits are torch.equal.

QUALITY (REPORT), for (REF, E) and every (SH_w, TF_w): token agreement (overall, per document, per window); the
distribution of the per-step logit max |d| (quantiles, histogram, exact zeros) and of the reference top-2 margin;
every argmax difference with its margin class, the BI margin, max |d| and the regret (reference logit of its own top
token minus that of BI's). FREE-RUNNING: the first divergence step per document (FRC vs REF, FR_w vs SH_w), and its
consistency with the teacher-forced first disagreement (equal when the decode is deterministic).
TIMING (REPORT): ms per decode step per arm.

OUTPUT: --out/summary.json (written after every arm), --out/differences.json (every argmax difference). Exit: the
number of failed MUST gates; 2 a refusal. ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0  # noqa: E402  (load_workload)

MARGIN_BINS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, math.inf]
D_BINS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, math.inf]
QUANTILES = [0.5, 0.9, 0.99, 1.0]


def log(msg):
    print("[r2_g0bi %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy)")
    ap.add_argument("--gemm", default="exact", choices=("exact", "fast", "bigemm"),
                    help="the GEMM mode: exact = G-b PaddedLinear(m_fix); fast = natural cuBLAS (the weaker claim: "
                         "I1 / I2 become REPORT); bigemm = G-a by path")
    ap.add_argument("--bigemm", default=None, help="G-a: the bigemm module (a pinned copy), with --gemm bigemm")
    ap.add_argument("--bigemm-fn", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--steps-c", type=int, default=256)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--steps-w", type=int, default=64)
    ap.add_argument("--ni-steps", type=int, default=8)
    ap.add_argument("--m-fix", type=int, default=32)
    ap.add_argument("--parts", default="C,NI,W")
    a = ap.parse_args(argv)
    a.parts = [p for p in a.parts.split(",") if p]
    if not a.parts or any(p not in ("C", "NI", "W") for p in a.parts):
        ap.error("parts: a list of C, NI, W")
    if (a.bigemm is None) != (a.bigemm_fn is None) or (a.gemm == "bigemm") != (a.bigemm is not None):
        ap.error("--bigemm and --bigemm-fn go together, with --gemm bigemm only")
    if a.S % 64 or a.S - 64 * (a.windows - 1) < 1024:
        ap.error("S must be 0 mod 64 and leave >= 1024 tokens in the first window")
    if not 1 <= a.steps_w <= 64:
        ap.error("steps-w must be in 1 .. 64 (the F1 window of a prefix 0 mod 64)")
    if a.batch < 1 or a.batch > a.m_fix:
        ap.error("batch must be in 1 .. m_fix")
    if a.steps_c < 1 or a.ni_steps < 1 or a.windows < 1:
        ap.error("steps-c, ni-steps and windows must be >= 1")
    return a


def load_by_path(path, name):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def free():
    """After the caller has deleted its references (del cache, state): collect them and return the cached blocks."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ decode driver
def decode_loop(model, cache, first, pos0, steps, forced=None, keep=True, ref=None, ref_rows=None):
    """Greedy decode of `steps` steps. first (B, 1) int64 device: the first input token; pos0 (B, 1) int32 device: its
    position. forced (steps + 1, B): the input of step s is forced[s] (teacher forcing). keep: return the logits
    (steps, B, V) float. ref (steps, B', V) with ref_rows (a row list): torch.equal of this run's logits with
    ref[s][ref_rows] every step. Returns dict(tokens (steps + 1, B) = the inputs then the argmaxes, logits, eq, sec)."""
    import torch
    B = first.shape[0]
    cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
    tok = first.view(B, 1)
    pos = pos0.view(B, 1).to(torch.int32)
    toks, kept, eq = [first.view(-1).clone()], [], []
    rows = None if ref_rows is None else torch.tensor(ref_rows, dtype=torch.int64, device="cuda")
    t0 = time.time()
    for s in range(steps):
        if forced is not None:
            tok = forced[s].view(B, 1)
        lg = model.decode_inference(tok, cu, pos, cache)[:, -1].float()
        nxt = lg.argmax(-1)
        toks.append(nxt.clone())
        if keep:
            kept.append(lg.clone())
        if ref is not None:
            r = ref[s] if rows is None else ref[s].index_select(0, rows)
            eq.append(bool(torch.equal(lg, r)))
        tok = nxt.view(B, 1)
        pos = pos + 1
    torch.cuda.synchronize()
    return dict(tokens=torch.stack(toks), logits=torch.stack(kept) if keep else None, eq=eq, sec=time.time() - t0)


def eq_verdict(eq):
    bad = [s for s, e in enumerate(eq) if not e]
    ok = bool(eq) and not bad
    return ok, ("torch.equal at all %d steps" % len(eq) if ok else
                "%d of %d steps differ; first at step %s" % (len(bad), len(eq), bad[0] if bad else None))


# ------------------------------------------------------------------ quality
def quality_records(ref_logits, ref_tokens, bi_logits):
    """Per (step, request): max |d|, the reference / BI top-2 margins, the argmaxes, agreement, the regret. Tensors
    (N, B) on the CPU. ref_tokens (N + 1, B): ref_tokens[s + 1] must be argmax ref_logits[s] (counted)."""
    import torch
    out = {k: [] for k in ("max_d", "ref_margin", "bi_margin", "ref_arg", "bi_arg", "regret")}
    for s in range(ref_logits.shape[0]):
        R, X = ref_logits[s], bi_logits[s]
        rv, xv = R.topk(2, dim=-1).values, X.topk(2, dim=-1).values
        ra, xa = R.argmax(-1), X.argmax(-1)
        out["max_d"].append((X - R).abs().amax(-1))
        out["ref_margin"].append(rv[:, 0] - rv[:, 1])
        out["bi_margin"].append(xv[:, 0] - xv[:, 1])
        out["ref_arg"].append(ra)
        out["bi_arg"].append(xa)
        out["regret"].append(R.gather(-1, ra[:, None])[:, 0] - R.gather(-1, xa[:, None])[:, 0])
    rec = {k: torch.stack(v).cpu() for k, v in out.items()}
    rec["agree"] = rec["ref_arg"] == rec["bi_arg"]
    rec["ref_token_mismatch"] = int((rec["ref_arg"] != ref_tokens[1:].cpu()).sum())
    return rec


def histogram(x, bins):
    import torch
    edges = torch.tensor(bins[1:-1], dtype=x.dtype)
    idx = torch.bucketize(x, edges, right=True)
    counts = torch.bincount(idx.flatten(), minlength=len(bins) - 1).tolist()
    return [dict(lo=bins[i], hi=bins[i + 1], n=int(counts[i])) for i in range(len(bins) - 1)]


def quantiles(x):
    import torch
    q = torch.quantile(x.flatten().double(), torch.tensor(QUANTILES, dtype=torch.float64))
    return {("p%g" % (100 * p)): float(v) for p, v in zip(QUANTILES, q)}


def margin_class(m):
    for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
        if lo <= m < hi:
            return "[%g, %g)" % (lo, hi)
    return "nan"


def summarize_quality(recs, tag):
    """recs: a list of (label, rec) with rec from quality_records. Returns (summary, differences)."""
    import torch
    cat = {k: torch.cat([r[k] for _, r in recs], 0) for k in ("max_d", "ref_margin", "bi_margin", "agree", "regret")}
    n = int(cat["agree"].numel())
    diffs = []
    for label, r in recs:
        for s, b in (~r["agree"]).nonzero().tolist():
            m = float(r["ref_margin"][s, b])
            diffs.append(dict(part=tag, label=label, step=s, doc=b, ref_token=int(r["ref_arg"][s, b]),
                              bi_token=int(r["bi_arg"][s, b]), ref_margin=m, margin_class=margin_class(m),
                              bi_margin=float(r["bi_margin"][s, b]), max_d=float(r["max_d"][s, b]),
                              regret=float(r["regret"][s, b])))
    classes = []
    for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
        sel = (cat["ref_margin"] >= lo) & (cat["ref_margin"] < hi)
        k = int(sel.sum())
        d = int((sel & ~cat["agree"]).sum())
        classes.append(dict(cls="[%g, %g)" % (lo, hi), steps=k, differences=d, rate=(d / k if k else None)))
    # every fraction is an exact count over an exact count, divided in float64 (Python floats)
    agree_n = int(cat["agree"].sum())
    per_doc_n = cat["agree"].sum(0).tolist() if cat["agree"].dim() == 2 else None
    steps_n = int(cat["agree"].shape[0])
    summ = dict(part=tag, comparisons=n, agree_count=agree_n, agreement=(agree_n / n if n else None),
                differences=len(diffs),
                agree_count_per_doc=per_doc_n, steps_per_doc=steps_n,
                agreement_per_doc=([c / steps_n for c in per_doc_n] if per_doc_n is not None and steps_n else None),
                agreement_per_label={label: int(r["agree"].sum()) / int(r["agree"].numel()) for label, r in recs},
                max_d=dict(quantiles=quantiles(cat["max_d"]), histogram=histogram(cat["max_d"], D_BINS),
                           exact_zero=int((cat["max_d"] == 0).sum())),
                ref_margin=dict(quantiles=quantiles(cat["ref_margin"]), histogram=histogram(cat["ref_margin"], MARGIN_BINS)),
                differences_by_ref_margin=classes,
                max_regret=float(cat["regret"].max()) if n else None,
                ref_token_mismatch=sum(r["ref_token_mismatch"] for _, r in recs))
    return summ, diffs


def first_divergence(fr_tokens, ref_tokens):
    """Per document: the first decode step s whose argmax (token s + 1) differs from the reference's; None if never."""
    out = []
    a, b = fr_tokens.cpu(), ref_tokens.cpu()
    for d in range(a.shape[1]):
        bad = (a[1:, d] != b[1:, d]).nonzero().flatten().tolist()
        out.append(bad[0] if bad else None)
    return out


def first_disagreement(rec):
    out = []
    for d in range(rec["agree"].shape[1]):
        bad = (~rec["agree"][:, d]).nonzero().flatten().tolist()
        out.append(bad[0] if bad else None)
    return out


# ------------------------------------------------------------------ the parts
def part_c(a, model, ops, docs, res, gates, diffs, save):
    import torch
    from nosi.r2_decode_path import build_ragged
    from nosi.r2_bi_decode import attach_bi, subset_state
    B, S, N = a.batch, a.S, a.steps_c
    prompts = [docs[b, :S] for b in range(B)]
    firsts = {}

    def build(f4_fix, bi):
        t0 = time.time()
        cache, st, first = build_ragged(model, prompts, N, f4_fix=f4_fix)
        if bi:
            attach_bi(st, ops)
        return cache, st, first, time.time() - t0

    def pos0(st):
        return st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)

    timing = {}
    with torch.inference_mode():
        log("C: REF (R2 shipped-order path, body r2, f4_fix False), %d steps" % N)
        cache, st, first, tb = build(False, False)
        firsts["REF"] = first.view(-1).cpu()
        ref = decode_loop(model, cache, first, pos0(st), N)
        timing["REF"] = dict(build_s=round(tb, 1), ms_per_step=round(1e3 * ref["sec"] / N, 2))
        del cache, st
        free()
        log("C: E (BI, teacher-forced) + subsets I1 (request 0 alone) and I2 (reversed), taken before E's first step")
        cache, st, first, tb = build(True, True)
        firsts["E"] = first.view(-1).cpu()
        c1, s1 = subset_state(model, cache, st, [0])
        attach_bi(s1, ops)
        rev = list(range(B - 1, -1, -1))
        c2, s2 = subset_state(model, cache, st, rev)
        attach_bi(s2, ops)
        e = decode_loop(model, cache, first, pos0(st), N, forced=ref["tokens"])
        timing["E"] = dict(build_s=round(tb, 1), ms_per_step=round(1e3 * e["sec"] / N, 2))
        del cache, st
        free()
        log("C: I1, I2")
        i1 = decode_loop(model, c1, ref["tokens"][0, :1].view(1, 1), pos0(s1), N, forced=ref["tokens"][:, :1],
                         keep=False, ref=e["logits"], ref_rows=[0])
        timing["I1"] = dict(ms_per_step=round(1e3 * i1["sec"] / N, 2))
        del c1, s1
        free()
        i2 = decode_loop(model, c2, ref["tokens"][0, rev].view(B, 1), pos0(s2), N, forced=ref["tokens"][:, rev],
                         keep=False, ref=e["logits"], ref_rows=rev)
        timing["I2"] = dict(ms_per_step=round(1e3 * i2["sec"] / N, 2))
        del c2, s2
        free()
        log("C: E2 (determinism)")
        cache, st, first, tb = build(True, True)
        firsts["E2"] = first.view(-1).cpu()
        e2 = decode_loop(model, cache, first, pos0(st), N, forced=ref["tokens"], keep=False, ref=e["logits"])
        timing["E2"] = dict(build_s=round(tb, 1), ms_per_step=round(1e3 * e2["sec"] / N, 2))
        del cache, st
        free()
        log("C: FRC (BI, free-running)")
        cache, st, first, tb = build(True, True)
        firsts["FRC"] = first.view(-1).cpu()
        frc = decode_loop(model, cache, first, pos0(st), N, keep=False)
        timing["FRC"] = dict(build_s=round(tb, 1), ms_per_step=round(1e3 * frc["sec"] / N, 2))
        del cache, st
        free()
        rec = quality_records(ref["logits"], ref["tokens"], e["logits"])
        del e
        free()
    p_ok = all(torch.equal(v, firsts["REF"]) for v in firsts.values())
    gates.append(gate("P_C_first_tokens", p_ok, True, "first tokens of the builds %s %s" % (sorted(firsts), "equal" if p_ok else "DIFFER")))
    for name, run in (("E2_C_determinism", e2), ("I1_C_request0_alone", i1), ("I2_C_batch_reversed", i2)):
        ok, why = eq_verdict(run["eq"])
        if ops.mode == "fast" and name != "E2_C_determinism":           # the weaker claim: GEMM rows depend on M
            gates.append(gate("R_" + name, True, False, "FAST (natural cuBLAS, the weaker claim): " + why))
        else:
            gates.append(gate(name, ok, True, why))
    summ, d = summarize_quality([("C", rec)], "C")
    diffs += d
    fd = first_divergence(frc["tokens"], ref["tokens"])
    td = first_disagreement(rec)
    res["C"] = dict(steps=N, batch=B, S=S, quality=summ, free_running=dict(first_divergence=fd, tf_first_disagreement=td,
                    consistent=(fd == td)), timing=timing,
                    ref_note="REF = the R2 path, body r2, f4_fix False (shipped ops, shipped order, F4 reproduced)")
    gates.append(gate("R_C_quality", True, False, "agreement %.4f over %d (step, doc); %d differences; max |d| p50 %.3g p99 %.3g"
                      % (summ["agreement"], summ["comparisons"], summ["differences"], summ["max_d"]["quantiles"]["p50"],
                         summ["max_d"]["quantiles"]["p99"])))
    save()
    return dict(tokens=ref["tokens"])                                     # part NI needs the tokens only


def part_ni(a, model, ops_nat, docs, ref, res, gates, save):
    import torch
    from nosi.r2_decode_path import build_ragged
    from nosi.r2_bi_decode import attach_bi, subset_state
    B, S, N = a.batch, a.S, a.ni_steps
    prompts = [docs[b, :S] for b in range(B)]
    with torch.inference_mode():
        cache, st, first = build_ragged(model, prompts, a.steps_c, f4_fix=True)
        attach_bi(st, ops_nat)
        c1, s1 = subset_state(model, cache, st, [0])
        attach_bi(s1, ops_nat)
        pos = st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)
        e = decode_loop(model, cache, first, pos, N, forced=ref["tokens"])
        del cache, st
        free()
        i1 = decode_loop(model, c1, ref["tokens"][0, :1].view(1, 1), pos[:1], N, forced=ref["tokens"][:, :1],
                         keep=False, ref=e["logits"], ref_rows=[0])
        del c1, s1, e
        free()
    ok, why = eq_verdict(i1["eq"])
    res["NI"] = dict(steps=N, eq=i1["eq"], note="the GEMMs at the natural row count (M = B vs M = 1)")
    gates.append(gate("R_NI_natural_gemm_detected", True, False,
                      ("DETECTED: " + why) if not ok else "BLIND: request 0 alone equals its batch row at every step "
                      "with natural cuBLAS (the I1 check could not see an M-dependent GEMM here)"))
    save()


def part_w(a, model, ops, docs, res, gates, diffs, save):
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi.r2_decode_path import R2State
    from nosi.r2_bi_decode import attach_bi
    B, n = a.batch, a.steps_w
    recs, fr, timing = [], {}, {}
    for w in range(a.windows):
        Sw = a.S - 64 * (a.windows - 1 - w)
        x = docs[:B, :Sw].to("cuda")
        pre = {}

        def prefill():
            cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
            lg, pos = model.batch_prefill(x, cache)
            model.has_buffers = False                                    # the shipped warm-up rebinds its graphs
            return cache, lg[:, -1].clone(), (pos[:, -1:] + 1).to(torch.int32)

        with torch.inference_mode():
            log("W%d: S_w %d: SH (shipped), TF (BI teacher-forced), FR (BI free-running)" % (w, Sw))
            cache, lg, p0 = prefill()
            pre["SH"] = lg
            first = lg.argmax(-1, keepdim=True)
            sh = decode_loop(model, cache, first, p0, n)
            del cache
            free()
            cache, lg, p0 = prefill()
            pre["TF"] = lg
            st = R2State.attach(model, cache, Sw, body="r2", f4_fix=True)
            attach_bi(st, ops)
            tf = decode_loop(model, cache, lg.argmax(-1, keepdim=True), p0, n, forced=sh["tokens"])
            del cache, st
            free()
            cache, lg, p0 = prefill()
            pre["FR"] = lg
            st = R2State.attach(model, cache, Sw, body="r2", f4_fix=True)
            attach_bi(st, ops)
            frw = decode_loop(model, cache, lg.argmax(-1, keepdim=True), p0, n, keep=False)
            del cache, st
            free()
            rec = quality_records(sh["logits"], sh["tokens"], tf["logits"])
        p_ok = torch.equal(pre["SH"], pre["TF"]) and torch.equal(pre["SH"], pre["FR"])
        gates.append(gate("P_W%d_prefill" % w, p_ok, True, "S_w %d: the three prefill logits %s" % (Sw, "equal" if p_ok else "DIFFER")))
        label = "W%d_S%d" % (w, Sw)
        recs.append((label, rec))
        fr[label] = dict(first_divergence=first_divergence(frw["tokens"], sh["tokens"]),
                         tf_first_disagreement=first_disagreement(rec))
        fr[label]["consistent"] = fr[label]["first_divergence"] == fr[label]["tf_first_disagreement"]
        timing[label] = dict(sh_ms_per_step=round(1e3 * sh["sec"] / n, 2), bi_ms_per_step=round(1e3 * tf["sec"] / n, 2))
        del sh, tf, frw
        free()
        log("W%d: agreement %d / %d" % (w, int(rec["agree"].sum()), int(rec["agree"].numel())))
    summ, d = summarize_quality(recs, "W")
    diffs += d
    res["W"] = dict(windows=a.windows, steps_per_window=n, batch=B, quality=summ, free_running=fr, timing=timing,
                    ref_note="the shipped decode itself, inside its F1 window (prefix 0 mod 64)")
    gates.append(gate("R_W_quality", True, False, "agreement %.4f over %d (step, doc); %d differences; max |d| p50 %.3g p99 %.3g"
                      % (summ["agreement"], summ["comparisons"], summ["differences"], summ["max_d"]["quantiles"]["p50"],
                         summ["max_d"]["quantiles"]["p99"])))
    save()


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_bi_decode import BIOps
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1 (the gate drives the switch)")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be > 0 and NOSI_POOL_BLOCKS 0 (%d, %d)" % (NL._ATTN_SPLITS, CE.POOL_BLOCKS))
        docs, sha = G0.load_workload(a.workload, a.batch, a.S)
        os.makedirs(a.out)
        attn = load_by_path(a.biattn, "piece3_bi_paged_attention")
        lin = getattr(load_by_path(a.bigemm, "r2_bigemm_by_path"), a.bigemm_fn) if a.bigemm else None
        ops = BIOps(attn, a.m_fix, gemm=a.gemm, linear=lin, linear_name=(a.bigemm_fn if a.bigemm else None))
        ops_nat = BIOps(attn, a.m_fix, gemm="fast")                         # the negative control NI
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s, allocated %.1f GB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
        res, gates, diffs = {}, [], []
        meta = dict(S=a.S, batch=a.batch, steps_c=a.steps_c, windows=a.windows, steps_w=a.steps_w, ni_steps=a.ni_steps,
                    m_fix=a.m_fix, parts=a.parts, bi=ops.describe(), biattn=a.biattn, bigemm=a.bigemm,
                    bigemm_fn=a.bigemm_fn, gemm=a.gemm, attn_splits=NL._ATTN_SPLITS, workload_sha256=sha,
                    nosi_commit=os.environ.get("G0BI_COMMIT", "unknown"), torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), margin_bins=MARGIN_BINS[:-1], d_bins=D_BINS[:-1])

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)
            json.dump(diffs, open(os.path.join(a.out, "differences.json"), "w"), indent=1, default=str)

        ref = None
        for part in ("C", "NI", "W"):
            if part not in a.parts:
                continue
            try:
                if part == "C":
                    ref = part_c(a, model, ops, docs, res, gates, diffs, save)
                elif part == "NI":
                    if ref is None:
                        raise RuntimeError("part NI needs part C's REF tokens")
                    part_ni(a, model, ops_nat, docs, ref, res, gates, save)
                else:
                    part_w(a, model, ops, docs, res, gates, diffs, save)
            except Exception as e:
                traceback.print_exc()
                gates.append(gate("PART_%s" % part, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])))
                save()
            gc.collect()
            torch.cuda.empty_cache()
        meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL", g["detail"][:400]))
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("r2_g0bi: %d MUST gate(s) failed" % nfail)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
