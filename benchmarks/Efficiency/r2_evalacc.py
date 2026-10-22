"""[evalsplit] EVAL CELL (b): ACCURACY EQUIVALENCE of FAST (natural cuBLAS) and EXACT (G-c strict) in the R2 path.

HUMAN DECISION part 3 (2026-10-22, binding): an element of normal serving that only adds non-determinism and does not
change accuracy runs EXACT for the bit-exactness demonstrations and FAST for the timing in the eval; an element that
changes accuracy may run only on draft rows. This cell is the evidence that FAST "does not affect accuracy". Cell (a)
(r2_tick_sg5.py --eval-role fast-time) refuses to time FAST unless this cell's summary.json passed (--accuracy-record).
Holder step: benchmarks/Efficiency/r2_evalacc_step.sh. CPU tests: tests_cpu/test_r2_evalsplit.py (the whole driver runs
on CPU through an injectable environment, controls included).

DATA (no new dataset): the project's PG-19 workloads. pg19-test-ctx65280 holds the first 65,280 tokens of the 55 test
documents with >= 65,280 tokens; each of them is also in pg19-test-c16k (the same tokenizer call, add_special_tokens
True, make_workload.py), so the c16k row (16,128 tokens) is the prompt and ctx65280's next --n-gold tokens are the GOLD
continuation. MUST EA_P0: the c16k row equals the first 16,128 tokens of the ctx65280 row for every document used. The
first --batch of the matched documents (document order) are the eval subset. TASK METRIC: teacher-forced next-token
prediction on the gold continuation (PG-19 language modelling): top-1 accuracy and mean NLL (nats) over
(n_gold - 1) x B positions. The prompt prefill is shared by every arm (one build_ragged, copies by subset_state).

ARMS (one process; every arm starts from a COPY of the same build, before its first decode step):
  TF  teacher-forced on the gold tokens (the BI serial decode, nosi/nosi/r2_bi_decode.py, attach_bi):
      E      EXACT (--exact-gemm: bi_c_strict = G-c strict, the human's EXACT; or exact = G-b), batch B (logits kept)
      E_rep  E again (MUST EA_E_determinism: torch.equal to E at every step)
      E_alt  E in chunks of --alt-batch rows (MUST EA_E_batch_invariance: every chunk row torch.equal to E's row)
      F      FAST (natural cuBLAS F.linear), batch B (logits kept)
      F_rep  F again (REPORT EA_R_F_replica: run-to-run noise of FAST at one M)
      F_alt  F in chunks of --alt-batch rows (REPORT EA_R_F_NI: natural cuBLAS M-dependence; the BATCH-SIZE noise of
             normal serving, agr1-style: analysis/eqmem/AGREEMENT_h43.md)
      P<d>   NEGATIVE CONTROLS: F with every GEMM output times (1 + d z), z ~ N(0, 1) from a generator seeded per arm
             (PlantLinear), d in --doses (default 2^-8, 2^-4, 2^-2): an accuracy-changing perturbation of the TARGET
  SPEC the R2 speculative round (nosi/nosi/r2_tick.py run_round, all-resident store; draft rows miss a hashed
      --draft-drop of their middle blocks, the SAME seed in every arm), greedy, free-running, --spec-steps committed
      tokens per request:
      FRC    the BI serial decode with EXACT, greedy, free-running (the reference of SG2)
      SE     EXACT round, batch B (MUST EA_SG2: committed tokens equal FRC's for every request)
      SE_alt EXACT round in chunks (MUST EA_SPEC_batch_invariance: committed tokens equal SE's)
      SF     FAST round, batch B;  SF_rep FAST again (the acceptance noise: same drafts);  SF_alt FAST in chunks
             (REPORT: agr1-style agreement; its acceptance is NOT a noise reference: the stand-in drop hashes the
             request index, so chunks draft differently)
METRICS per pair (X vs Y over the same positions): greedy argmax agreement, per-position max |d log p| over the
vocabulary (quantiles), |d log p(gold)|, top-1 accuracy difference (discordant n10 / n01), NLL difference, per-document
cluster standard errors; SPEC: acceptance accepted / (accepted + rejected), gold match (committed token i == gold[i]),
committed-sequence agreement (first divergence per request, full agreements).

PRE-REGISTERED BAND: "does not affect accuracy" (verdict EQUIVALENT) iff for each metric m in TF top-1 accuracy,
TF NLL, SPEC acceptance and SPEC gold match:
    |m(F) - m(E)| <= max(NOISE_MULT x noise_m, SE_MULT x se_m) + floor_m   and   |m(F) - m(E)| <= cap_m
noise_m = FAST's own spread for m (TF: max over F_alt and F_rep of |m(F_x) - m(F)|; SPEC: |m(SF_rep) - m(SF)|): the
non-determinism normal serving already has; se_m = the standard error of the difference (TF: per-document cluster SE;
SPEC: binomial); NOISE_MULT 2, SE_MULT 3; floor / cap in BANDS. EXACT is batch-invariant (E_alt = E bitwise, MUST), so
EXACT's own batch noise is 0 by construction: the noise reference is FAST's.
MUST EA_VERDICT_FAST: FAST is EQUIVALENT. MUST EA_NEG_precondition: the top dose changes F's logits somewhere (not a
no-op on real data, lessons ledger L34). MUST EA_NEG_detected: the top dose's verdict is AFFECTS (TF criteria).
REPORT: the verdict of every lower dose (the detection floor), agreement and max |d log p| bands below.

REGISTERED PREDICTIONS (2026-10-22T14:21:48Z, date -u, before any run; INFERENCE by the builder, scored by a scorer):
  EA-P0        PASS (55 matched documents; prefix equal)
  EA-E         E_rep and E_alt bitwise (G-c strict passed bigc-gate G2 / G2w at M 1..1148 on nova07)
  EA-TF-AGREE  F vs E teacher-forced argmax agreement in [0.970, 0.998]
  EA-TF-MAXD   F vs E max |d log p| p50 in [0.005, 0.30], p99 in [0.05, 2.0] nats
  EA-TF-ACC    E top-1 accuracy in [0.35, 0.65]; E NLL in [1.6, 3.0] nats (PG-19, 8B model, 16K context)
  EA-TF-D      |acc(F) - acc(E)| <= 0.003; |NLL(F) - NLL(E)| <= 0.003 nats
  EA-NI        F_alt differs from F (natural cuBLAS changes kernels with M; bigc G3b: 4-8 kernel names per shape)
  EA-FREP      F_rep bitwise equal to F (cuBLAS deterministic at one M) - if not, FAST has run-to-run noise (REPORT)
  EA-SPEC      SE = FRC and SE_alt = SE (SG2 under G-c); |acceptance(SF) - acceptance(SE)| <= 0.01;
               SF vs SE full-agreement fraction >= 0.5 (free-running divergence after a near tie; REPORT)
  EA-NEG       dose 2^-2: AFFECTS with |d NLL| > 0.05; dose 2^-8: EQUIVALENT (rounding-scale noise); 2^-4: no band
  EA-VERDICT   FAST EQUIVALENT
  FALSIFIER    FAST AFFECTS (the human's condition for FAST timing fails: cell (a) must not report a FAST headline)
OUTPUT: --out/summary.json (after every arm; results.verdict). The last stdout line is R2_EVALACC_DONE failed=<n>;
exit = n (2 = refused). ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PARTS = ("TF", "SPEC", "NEG")
EXACT_CHOICES = ("bi_c_strict", "exact")
DOSES = (2.0 ** -8, 2.0 ** -4, 2.0 ** -2)
PLANT_SEED = 1234
NOISE_MULT = 2.0
SE_MULT = 3.0
BANDS = dict(acc=dict(floor=0.001, cap=0.01),          # TF top-1 accuracy (fraction)
             nll=dict(floor=0.001, cap=0.01),          # TF mean NLL (nats)
             accept=dict(floor=0.002, cap=0.02),       # SPEC acceptance (fraction)
             gold=dict(floor=0.002, cap=0.02))         # SPEC gold match (fraction)
TF_CRITERIA = ("acc", "nll")
SPEC_CRITERIA = ("accept", "gold")
QUANTILES = (0.5, 0.9, 0.99, 1.0)
PRED = dict(tf_agree=(0.970, 0.998), maxd_p50=(0.005, 0.30), maxd_p99=(0.05, 2.0), acc_E=(0.35, 0.65),
            nll_E=(1.6, 3.0), d_acc=0.003, d_nll=0.003, spec_d_accept=0.01, spec_full_agree=0.5, neg_top_dnll=0.05)


def log(msg):
    print("[r2_evalacc %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True, help="pg19-test-c16k (the prompts)")
    ap.add_argument("--gold-workload", required=True, help="pg19-test-ctx65280 (the gold continuation)")
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy)")
    ap.add_argument("--exact-gemm", default="bi_c_strict", choices=EXACT_CHOICES,
                    help="EXACT: bi_c_strict (G-c strict, the human's EXACT for bit-exactness) or exact (G-b)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--alt-batch", type=int, default=8, help="the chunk size of the *_alt arms (divides --batch)")
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--n-gold", type=int, default=256, help="gold tokens per document (n_gold - 1 scored positions)")
    ap.add_argument("--spec-steps", type=int, default=64, help="committed tokens per request in SPEC (first included)")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--draft-drop", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7, help="the draft-drop seed (every SPEC arm)")
    ap.add_argument("--doses", default=",".join(repr(d) for d in DOSES), help="NEG plant doses; the last is the top")
    ap.add_argument("--plant-seed", type=int, default=PLANT_SEED)
    ap.add_argument("--parts", default="TF,SPEC,NEG")
    a = ap.parse_args(argv)
    a.parts = [p for p in a.parts.split(",") if p]
    if not a.parts or any(p not in PARTS for p in a.parts):
        ap.error("parts: a list of %s" % (PARTS,))
    if "NEG" in a.parts and "TF" not in a.parts:
        ap.error("NEG needs TF (its references E and F)")
    try:
        a.dose_list = [float(x) for x in a.doses.split(",") if x]
    except ValueError:
        a.dose_list = []
    if "NEG" in a.parts and (not a.dose_list or any(d <= 0 for d in a.dose_list)
                             or a.dose_list != sorted(a.dose_list)):
        ap.error("--doses: positive, ascending; the last is the top dose (MUST detected)")
    if a.batch < 2 or a.alt_batch < 1 or a.alt_batch >= a.batch or a.batch % a.alt_batch:
        ap.error("2 <= batch, 1 <= alt-batch < batch, alt-batch divides batch (equal chunk M)")
    if a.n_gold < 2 or a.spec_steps < 2 or a.S < 4096:
        ap.error("n-gold >= 2, spec-steps >= 2, S >= 4096")
    if not 0.0 <= a.draft_drop < 1.0:
        ap.error("draft-drop in [0, 1)")
    a.m_fix = (1 + a.K) * a.batch
    return a


# ------------------------------------------------------------------ data
def _manifest(d):
    names = [n for n in os.listdir(d) if n.endswith("_manifest.json")]
    if len(names) != 1:
        raise SystemExit("REFUSED: %s needs exactly one *_manifest.json (has %s)" % (d, names))
    man = json.load(open(os.path.join(d, names[0])))
    return man, names[0][:-len("_manifest.json")]


def _rows(d, stem, man):
    import numpy as np
    path = os.path.join(d, stem + "_tokens.bin")
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    sums = {}
    sf = os.path.join(d, "SHA256SUMS")
    if os.path.exists(sf):
        for line in open(sf):
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
    if sums.get(stem + "_tokens.bin") not in (None, sha):
        raise SystemExit("REFUSED: %s sha256 %s is not SHA256SUMS's" % (path, sha))
    n, ctx = man["shape"]
    return np.frombuffer(raw, dtype="<i4").reshape(n, ctx), sha


def load_gold(c16k_dir, gold_dir, S, n_gold, B):
    """The eval subset (module docstring, DATA): dict(prompts (B, S) int64, gold (B, n_gold) int64, doc_index,
    matched, prefix_mismatch (doc indexes whose c16k row differs from the gold row's prefix), sha_prompts, sha_gold)."""
    import torch
    mc, sc = _manifest(c16k_dir)
    mg, sg = _manifest(gold_dir)
    rc, sha_c = _rows(c16k_dir, sc, mc)
    rg, sha_g = _rows(gold_dir, sg, mg)
    ctx_c, ctx_g = rc.shape[1], rg.shape[1]
    if S > ctx_c or ctx_c + n_gold > ctx_g or S != ctx_c:
        raise SystemExit("REFUSED: S %d must equal the prompt rows' %d and %d + n_gold %d <= %d"
                         % (S, ctx_c, ctx_c, n_gold, ctx_g))
    pos_c = {int(d["index"]): i for i, d in enumerate(mc["docs"])}
    pairs = [(pos_c[int(d["index"])], j, int(d["index"])) for j, d in enumerate(mg["docs"]) if int(d["index"]) in pos_c]
    if len(pairs) < B:
        raise SystemExit("REFUSED: %d matched documents < batch %d" % (len(pairs), B))
    use = pairs[:B]
    mism = [k for i, j, k in use if not (rc[i] == rg[j, :ctx_c]).all()]
    prompts = torch.from_numpy(rc[[i for i, _, _ in use]].astype("int64"))
    gold = torch.from_numpy(rg[[j for _, j, _ in use], ctx_c:ctx_c + n_gold].astype("int64"))
    return dict(prompts=prompts, gold=gold, doc_index=[k for _, _, k in use], matched=len(pairs),
                prefix_mismatch=mism, sha_prompts=sha_c, sha_gold=sha_g)


# ------------------------------------------------------------------ the negative control
class PlantLinear:
    """NEGATIVE CONTROL (an accuracy-changing TARGET perturbation): base(x, w), then every output element times
    (1 + dose z), z ~ N(0, 1) from a generator seeded with `seed` at the arm's first call (deterministic in call
    order). dose 0 is base itself (bitwise)."""

    def __init__(self, dose: float, seed: int, base=None):
        import torch.nn.functional as F
        self.dose, self.seed = float(dose), int(seed)
        self.base = base or F.linear
        self.gen, self.calls = None, 0

    def __call__(self, x, w, *rest):
        import torch
        y = self.base(x, w, *rest)
        if self.dose == 0.0:
            return y
        if self.gen is None or self.gen.device != y.device:
            self.gen = torch.Generator(device=y.device)
            self.gen.manual_seed(self.seed)
        z = torch.randn(y.shape, generator=self.gen, device=y.device, dtype=torch.float32)
        self.calls += 1
        return (y.float() * (1.0 + self.dose * z)).to(y.dtype)


def plant_ops(fast_ops, dose, seed):
    """FAST's ops with PlantLinear(dose) as the linear (attention, mean and m_fix unchanged)."""
    lin = PlantLinear(dose, seed, base=fast_ops.linear)
    d = dict(fast_ops.describe()) if hasattr(fast_ops, "describe") else {}
    d.update(gemm="NEGATIVE CONTROL: FAST x (1 + %g z), seed %d" % (dose, seed))
    return SimpleNamespace(linear=lin, attn=fast_ops.attn, mean=fast_ops.mean, m_fix=fast_ops.m_fix,
                           mode="plant", gemm=d["gemm"], describe=lambda: d)


# ------------------------------------------------------------------ teacher-forced arms
def tf_arm(step, gold, refs=None, keep=False):
    """One teacher-forced arm. step(s, inputs (k,) int64) -> logits (k, V) float32 (one decode step of this arm's k
    rows). gold (k, n_gold) int64 on the step's device: the input of step s is gold[:, s], its target gold[:, s + 1].
    refs: {name: (logits (steps, B, V), rows (k,) int64 index into B)}. Returns dict(correct (steps, k) bool, gold_lp
    (steps, k) float64, arg (steps, k) int64 (CPU), cmp {name: dict(maxd (steps, k) float64, agree (steps, k) bool,
    equal_steps [bool])}, logits (steps, k, V) on the device when keep)."""
    import torch
    steps = gold.shape[1] - 1
    corr, glp, args, kept = [], [], [], []
    cmp = {n: dict(maxd=[], agree=[], equal_steps=[]) for n in (refs or {})}
    for s in range(steps):
        lg = step(s, gold[:, s]).float()
        tgt = gold[:, s + 1]
        lp = torch.log_softmax(lg, -1)
        arg = lg.argmax(-1)
        corr.append(arg == tgt)
        glp.append(lp.gather(-1, tgt[:, None])[:, 0].double())
        args.append(arg)
        if keep:
            kept.append(lg.clone())
        for n, (ref, rows) in (refs or {}).items():
            r = ref[s].index_select(0, rows)
            cmp[n]["equal_steps"].append(bool(torch.equal(lg, r)))
            cmp[n]["maxd"].append((lp - torch.log_softmax(r, -1)).abs().amax(-1).double())
            cmp[n]["agree"].append(arg == r.argmax(-1))
    out = dict(correct=torch.stack(corr).cpu(), gold_lp=torch.stack(glp).cpu(), arg=torch.stack(args).cpu(),
               cmp={n: dict(maxd=torch.stack(c["maxd"]).cpu(), agree=torch.stack(c["agree"]).cpu(),
                            equal_steps=c["equal_steps"]) for n, c in cmp.items()})
    out["logits"] = torch.stack(kept) if keep else None
    return out


def merge_chunks(parts):
    """Chunk results (each from tf_arm over consecutive rows) -> one result in the main row order (dim 1)."""
    import torch
    out = dict(correct=torch.cat([p["correct"] for p in parts], 1), gold_lp=torch.cat([p["gold_lp"] for p in parts], 1),
               arg=torch.cat([p["arg"] for p in parts], 1), logits=None, cmp={})
    for n in parts[0]["cmp"]:
        out["cmp"][n] = dict(maxd=torch.cat([p["cmp"][n]["maxd"] for p in parts], 1),
                             agree=torch.cat([p["cmp"][n]["agree"] for p in parts], 1),
                             equal_steps=[all(p["cmp"][n]["equal_steps"][s] for p in parts)
                                          for s in range(len(parts[0]["cmp"][n]["equal_steps"]))])
    return out


def quantiles(x):
    import torch
    q = torch.quantile(x.flatten().double(), torch.tensor(QUANTILES, dtype=torch.float64))
    return {("p%g" % (100 * p)): float(v) for p, v in zip(QUANTILES, q)}


def arm_stats(r):
    """Top-1 accuracy and mean NLL over every (step, document), and per document."""
    c = r["correct"].double()
    return dict(n=int(c.numel()), acc=float(c.mean()), nll=float(-r["gold_lp"].mean()),
                acc_doc=c.mean(0).tolist(), nll_doc=(-r["gold_lp"]).mean(0).tolist())


def _cluster_se(xs):
    """Standard error of the mean of per-document values (documents are the independent units)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def tf_pair(x, y, cmp=None):
    """X vs Y (tf_arm results over the same positions). cmp: X's comparison record against Y (tf_arm refs)."""
    sx, sy = arm_stats(x), arm_stats(y)
    n10 = int((x["correct"] & ~y["correct"]).sum())
    n01 = int((~x["correct"] & y["correct"]).sum())
    out = dict(d_acc=sx["acc"] - sy["acc"], d_nll=sx["nll"] - sy["nll"], n=sx["n"], n10=n10, n01=n01,
               se_acc=_cluster_se([a - b for a, b in zip(sx["acc_doc"], sy["acc_doc"])]),
               se_nll=_cluster_se([a - b for a, b in zip(sx["nll_doc"], sy["nll_doc"])]),
               d_gold_lp_abs_mean=float((x["gold_lp"] - y["gold_lp"]).abs().mean()))
    if cmp is not None:
        agree = cmp["agree"]
        out.update(agree=float(agree.double().mean()), disagree_n=int((~agree).sum()), maxd=quantiles(cmp["maxd"]),
                   maxd_zero=int((cmp["maxd"] == 0).sum()), bitwise_steps=sum(cmp["equal_steps"]),
                   steps=len(cmp["equal_steps"]), bitwise=bool(cmp["equal_steps"]) and all(cmp["equal_steps"]))
    return out


# ------------------------------------------------------------------ the band
def inside_noise(d, noise, se, band):
    """The pre-registered rule (module docstring): (ok, allowance)."""
    allow = max(NOISE_MULT * abs(noise or 0.0), SE_MULT * (se or 0.0)) + band["floor"]
    return abs(d) <= allow and abs(d) <= band["cap"], allow


def verdict(diffs, noises, ses, criteria):
    """diffs / noises / ses: {criterion: value}. Returns dict(verdict EQUIVALENT | AFFECTS, criteria {c: dict(d,
    noise, se, allow, cap, ok)}, failed [c])."""
    out, failed = {}, []
    for c in criteria:
        ok, allow = inside_noise(diffs[c], noises.get(c), ses.get(c), BANDS[c])
        out[c] = dict(d=diffs[c], noise=noises.get(c), se=ses.get(c), allow=allow, cap=BANDS[c]["cap"], ok=ok)
        if not ok:
            failed.append(c)
    return dict(verdict="EQUIVALENT" if not failed else "AFFECTS", criteria=out, failed=failed)


def tf_noise(f_main, f_others):
    """FAST's own TF spread: max over the other FAST arms of |metric(F_x) - metric(F)|."""
    sm = arm_stats(f_main)
    acc = max((abs(arm_stats(o)["acc"] - sm["acc"]) for o in f_others), default=0.0)
    nll = max((abs(arm_stats(o)["nll"] - sm["nll"]) for o in f_others), default=0.0)
    return dict(acc=acc, nll=nll)


# ------------------------------------------------------------------ speculative arms
def spec_metrics(outs, stats, gold):
    """outs: committed token lists per request (the first token included); stats: the round's stats (accepted,
    rejected); gold: (B, n_gold) list of lists. Acceptance and gold match (committed token i == gold[i])."""
    acc_n, rej_n = int(stats.get("accepted", 0)), int(stats.get("rejected", 0))
    match = total = 0
    for b, o in enumerate(outs):
        g = gold[b]
        for i in range(min(len(o), len(g))):
            total += 1
            match += int(o[i] == g[i])
    return dict(accepted=acc_n, rejected=rej_n, decisions=acc_n + rej_n,
                acceptance=(acc_n / (acc_n + rej_n)) if acc_n + rej_n else None,
                gold_match=(match / total) if total else None, gold_n=total)


def seq_agreement(xs, ys):
    """agr1-style: per request the first index where the committed sequences differ (None if equal), full agreements."""
    first = []
    for x, y in zip(xs, ys):
        n = min(len(x), len(y))
        d = next((i for i in range(n) if x[i] != y[i]), None)
        if d is None and len(x) != len(y):
            d = n
        first.append(d)
    full = sum(1 for d in first if d is None)
    div = sorted(d for d in first if d is not None)
    return dict(requests=len(first), full=full, full_fraction=full / len(first) if first else None,
                first_divergence=first, median_first_divergence=(div[len(div) // 2] if div else None))


def _binom_se(p1, n1, p2, n2):
    if p1 is None or p2 is None or not n1 or not n2:
        return 0.0
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    return math.sqrt(max(p * (1 - p), 0.0) * (1.0 / n1 + 1.0 / n2))


def spec_verdict_inputs(se_m, sf_m, sf_rep_m):
    diffs = dict(accept=sf_m["acceptance"] - se_m["acceptance"], gold=sf_m["gold_match"] - se_m["gold_match"])
    noises = dict(accept=abs(sf_rep_m["acceptance"] - sf_m["acceptance"]),
                  gold=abs(sf_rep_m["gold_match"] - sf_m["gold_match"]))
    ses = dict(accept=_binom_se(sf_m["acceptance"], sf_m["decisions"], se_m["acceptance"], se_m["decisions"]),
               gold=_binom_se(sf_m["gold_match"], sf_m["gold_n"], se_m["gold_match"], se_m["gold_n"]))
    return diffs, noises, ses


# ------------------------------------------------------------------ the environment (real: nosi on the GPU)
class RealEnv:
    """The production computation: build_ragged, subset_state, attach_bi, decode_inference, TickEngine.run_round."""

    def setup(self, a):
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS 0 are needed (%d, %d)"
                             % (NL._ATTN_SPLITS, CE.POOL_BLOCKS))
        import r2_g0bi_gate as GB
        from nosi import NOSALlama as Llama
        self.torch, self.GB = torch, GB
        self.attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
        t0 = time.time()
        self.model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        return dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), attn_splits=NL._ATTN_SPLITS)

    def ops(self, mode, a):
        from nosi.r2_bi_decode import BIOps
        return BIOps(self.attn, a.m_fix, gemm=mode)

    def device(self):
        return "cuda"

    def build(self, prompts, horizon):
        from nosi.r2_decode_path import build_ragged
        cache, st, first = build_ragged(self.model, [p for p in prompts], horizon, f4_fix=True)
        return cache, st, [int(x) for x in first.view(-1).tolist()]

    def subset(self, cache, st, rows):
        from nosi.r2_bi_decode import subset_state
        return subset_state(self.model, cache, st, rows)

    def stepper(self, cache, st, ops):
        from nosi.r2_bi_decode import attach_bi
        torch = self.torch
        attach_bi(st, ops)
        B = st.B
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        pos = [st.pos_host.view(B, 1).to(device="cuda", dtype=torch.int32)]

        def step(s, inp):
            lg = self.model.decode_inference(inp.reshape(B, 1).contiguous(), cu, pos[0], cache)[:, -1]
            pos[0] = pos[0] + 1
            return lg
        return step

    def frc(self, cache, st, ops, first, steps):
        """The BI serial decode, greedy, free-running: (steps, B) tokens (the first token, then the argmaxes)."""
        from nosi.r2_bi_decode import attach_bi
        torch = self.torch
        attach_bi(st, ops)
        f = torch.tensor(first, dtype=torch.int64, device="cuda").view(-1, 1)
        p0 = st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)
        r = self.GB.decode_loop(self.model, cache, f, p0, steps - 1, keep=False)
        return r["tokens"].cpu().t().tolist()

    def engine(self, cache, st, ops, a):
        from nosi.r2_tick import TickEngine
        return TickEngine.from_r2(self.model, cache, st, ops, draft_drop=a.draft_drop, drop_seed=a.seed, check=True)

    def spec(self, eng, first, a):
        from nosi.r2_tick import run_round
        req = [0]

        def on_tick(rows, tokens, out):
            req[0] += len({r.b for r in rows})
        rnd = run_round(eng, first, [a.spec_steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA, on_tick=on_tick)
        self.torch.cuda.synchronize()
        mirror = eng.mirror_failures()
        return dict(outs=[list(r.out) for r in rnd.reqs], stats=dict(rnd.stats), ticks=rnd.ticks,
                    request_ticks=req[0], bookkeeping_ok=(not mirror and not eng.void
                                                         and eng.stats["void_ticks"] == 0))

    def free(self):
        self.GB.free()

    def peak_gb(self):
        return round(self.torch.cuda.max_memory_allocated() / 1e9, 2)


# ------------------------------------------------------------------ the run
def run(a, env, gates, res, meta, save):
    import torch
    data = load_gold(a.workload, a.gold_workload, a.S, a.n_gold, a.batch)
    meta.update(doc_index=data["doc_index"], matched_docs=data["matched"], sha_prompts=data["sha_prompts"],
                sha_gold=data["sha_gold"])
    gates.append(gate("EA_P0_gold_prefix", not data["prefix_mismatch"], True,
                      "%d matched documents; the first %d used; c16k row != gold prefix for %s"
                      % (data["matched"], a.batch, data["prefix_mismatch"] or "none")))
    if data["prefix_mismatch"]:
        return
    dev = env.device()
    gold_dev = data["gold"].to(dev)
    gold_list = data["gold"].tolist()
    B, k = a.batch, a.alt_batch
    chunks = [list(range(c, c + k)) for c in range(0, B, k)]
    horizon = max(a.n_gold, a.spec_steps + 16) + 80
    ops_e, ops_f = env.ops(a.exact_gemm, a), env.ops("fast", a)
    meta.update(exact_ops=ops_e.describe(), fast_ops=ops_f.describe(), chunks=len(chunks))
    t0 = time.time()
    with torch.inference_mode():
        cache0, st0, first = env.build(list(data["prompts"]), horizon)
        meta["build_s"] = round(time.time() - t0, 1)

        def tf(ops, rows, refs=None, keep=False):
            c, s = env.subset(cache0, st0, rows)
            idx = torch.tensor(rows, dtype=torch.int64, device=dev)
            r = tf_arm(env.stepper(c, s, ops), gold_dev.index_select(0, idx),
                       {n: (ref, idx) for n, ref in (refs or {}).items()}, keep)
            del c, s
            env.free()
            return r

        def tf_chunks(ops, refs):
            return merge_chunks([tf(ops, rows, refs) for rows in chunks])
        arms = {}
        if "TF" in a.parts:
            log("TF: E (%s), E_rep, E_alt" % a.exact_gemm)
            arms["E"] = tf(ops_e, list(range(B)), keep=True)
            E_lg = arms["E"].pop("logits")
            arms["E_rep"] = tf(ops_e, list(range(B)), {"E": E_lg})
            arms["E_alt"] = tf_chunks(ops_e, {"E": E_lg})
            log("TF: F (FAST), F_rep, F_alt")
            arms["F"] = tf(ops_f, list(range(B)), {"E": E_lg}, keep=True)
            F_lg = arms["F"].pop("logits")
            arms["F_rep"] = tf(ops_f, list(range(B)), {"E": E_lg, "F": F_lg})
            arms["F_alt"] = tf_chunks(ops_f, {"E": E_lg, "F": F_lg})
            for name in ("E_rep", "E_alt"):
                p = tf_pair(arms[name], arms["E"], arms[name]["cmp"]["E"])
                must = name.replace("E_rep", "EA_E_determinism").replace("E_alt", "EA_E_batch_invariance")
                gates.append(gate(must, p["bitwise"], True, "%s vs E: logits torch.equal at %d of %d steps (%s)"
                                  % (name, p["bitwise_steps"], p["steps"], a.exact_gemm)))
            pairs = dict(F_vs_E=tf_pair(arms["F"], arms["E"], arms["F"]["cmp"]["E"]),
                         F_rep_vs_F=tf_pair(arms["F_rep"], arms["F"], arms["F_rep"]["cmp"]["F"]),
                         F_alt_vs_F=tf_pair(arms["F_alt"], arms["F"], arms["F_alt"]["cmp"]["F"]),
                         F_alt_vs_E=tf_pair(arms["F_alt"], arms["E"], arms["F_alt"]["cmp"]["E"]))
            res["TF"] = dict(arms={n: {kk: v for kk, v in arm_stats(r).items() if not kk.endswith("_doc")}
                                   for n, r in arms.items()}, pairs=pairs, positions=(a.n_gold - 1) * B)
            gates.append(gate("EA_R_F_NI", True, False, "F_alt vs F (natural cuBLAS at M %d vs %d): %s; agreement %.5f"
                              % (k, B, "bitwise EQUAL (BLIND: no M-dependence seen)" if pairs["F_alt_vs_F"]["bitwise"]
                                 else "DIFFER (natural M-dependence detected)", pairs["F_alt_vs_F"]["agree"])))
            gates.append(gate("EA_R_F_replica", True, False, "F_rep vs F at M %d: %s" % (
                B, "bitwise equal (no run-to-run noise)" if pairs["F_rep_vs_F"]["bitwise"]
                else "DIFFER (FAST has run-to-run noise at one M)")))
            fe = pairs["F_vs_E"]
            gates.append(gate("EA_R_TF_report", True, False,
                              "E acc %.4f nll %.4f; F acc %.4f nll %.4f; F vs E agreement %.5f, max|d log p| p50 %.3g "
                              "p99 %.3g max %.3g" % (res["TF"]["arms"]["E"]["acc"], res["TF"]["arms"]["E"]["nll"],
                                                     res["TF"]["arms"]["F"]["acc"], res["TF"]["arms"]["F"]["nll"],
                                                     fe["agree"], fe["maxd"]["p50"], fe["maxd"]["p99"],
                                                     fe["maxd"]["p100"])))
            res["TF"]["noise"] = tf_noise(arms["F"], [arms["F_rep"], arms["F_alt"]])
            save()
            if "NEG" in a.parts:
                res["NEG"] = {}
                for i, d in enumerate(a.dose_list):
                    top = i == len(a.dose_list) - 1
                    log("NEG: plant dose %g%s" % (d, " (top)" if top else ""))
                    r = tf(plant_ops(ops_f, d, a.plant_seed), list(range(B)), {"E": E_lg, "F": F_lg})
                    pe, pf = tf_pair(r, arms["E"], r["cmp"]["E"]), tf_pair(r, arms["F"], r["cmp"]["F"])
                    v = verdict(dict(acc=pe["d_acc"], nll=pe["d_nll"]), res["TF"]["noise"],
                                dict(acc=pe["se_acc"], nll=pe["se_nll"]), TF_CRITERIA)
                    res["NEG"]["%g" % d] = dict(vs_E=pe, vs_F=pf, verdict=v)
                    noop = pf["bitwise"]
                    if top:
                        gates.append(gate("EA_NEG_precondition", not noop, True,
                                          "top dose %g: the plant %s F's logits (%d of %d steps bitwise)"
                                          % (d, "does NOT change" if noop else "changes", pf["bitwise_steps"],
                                             pf["steps"])))
                        gates.append(gate("EA_NEG_detected", (not noop) and v["verdict"] == "AFFECTS", True,
                                          "top dose %g: verdict %s (failed %s); d acc %+.4f, d nll %+.4f nats"
                                          % (d, v["verdict"], v["failed"], pe["d_acc"], pe["d_nll"])))
                    else:
                        gates.append(gate("EA_R_NEG_%g" % d, True, False, "dose %g: verdict %s (failed %s); d acc "
                                          "%+.4f, d nll %+.4f; agreement vs E %.5f" % (d, v["verdict"], v["failed"],
                                                                                       pe["d_acc"], pe["d_nll"],
                                                                                       pe["agree"])))
                    del r
                    save()
            del E_lg, F_lg
            env.free()
        if "SPEC" in a.parts:
            log("SPEC: FRC (EXACT serial), SE, SE_alt, SF, SF_rep, SF_alt")
            c, s = env.subset(cache0, st0, list(range(B)))
            frc = env.frc(c, s, ops_e, first, a.spec_steps)
            del c, s
            env.free()
            runs = {}
            for name, ops in (("SE", ops_e), ("SF", ops_f), ("SF_rep", ops_f)):   # one engine at a time (an
                eng = env.engine(cache0, st0, ops, a)                               # all-resident store is about
                runs[name] = env.spec(eng, first, a)                                # 0.53 GB per request)
                del eng
                env.free()
            for name, ops in (("SE_alt", ops_e), ("SF_alt", ops_f)):
                outs, st_sum, ok = [], dict(accepted=0, rejected=0), True
                for rows in chunks:
                    c, s = env.subset(cache0, st0, rows)
                    eng = env.engine(c, s, ops, a)
                    r = env.spec(eng, [first[i] for i in rows], a)
                    outs += r["outs"]
                    for kk in st_sum:
                        st_sum[kk] += int(r["stats"].get(kk, 0))
                    ok = ok and r["bookkeeping_ok"]
                    del c, s, eng
                    env.free()
                runs[name] = dict(outs=outs, stats=st_sum, bookkeeping_ok=ok)
            m = {n: spec_metrics(r["outs"], r["stats"], gold_list) for n, r in runs.items()}
            agr = dict(SE_vs_FRC=seq_agreement(runs["SE"]["outs"], frc),
                       SE_alt_vs_SE=seq_agreement(runs["SE_alt"]["outs"], runs["SE"]["outs"]),
                       SF_vs_SE=seq_agreement(runs["SF"]["outs"], runs["SE"]["outs"]),
                       SF_rep_vs_SF=seq_agreement(runs["SF_rep"]["outs"], runs["SF"]["outs"]),
                       SF_alt_vs_SF=seq_agreement(runs["SF_alt"]["outs"], runs["SF"]["outs"]))
            gates.append(gate("EA_SG2_spec_equals_serial", agr["SE_vs_FRC"]["full"] == B, True,
                              "EXACT round vs the BI serial decode: %d of %d requests' %d committed tokens equal"
                              % (agr["SE_vs_FRC"]["full"], B, a.spec_steps)))
            gates.append(gate("EA_SPEC_batch_invariance", agr["SE_alt_vs_SE"]["full"] == B, True,
                              "EXACT round in chunks of %d vs batch %d: %d of %d requests equal"
                              % (k, B, agr["SE_alt_vs_SE"]["full"], B)))
            gates.append(gate("EA_SPEC_bookkeeping", all(r.get("bookkeeping_ok", True) for r in runs.values()), True,
                              "twin counters = host mirrors and no void tick in every SPEC arm"))
            gates.append(gate("EA_R_SPEC_report", True, False,
                              "acceptance SE %s SF %s SF_rep %s; gold match SE %s SF %s; SF vs SE full agreement %d / %d "
                              "(median first divergence %s); SF_alt vs SF %d / %d"
                              % (m["SE"]["acceptance"], m["SF"]["acceptance"], m["SF_rep"]["acceptance"],
                                 m["SE"]["gold_match"], m["SF"]["gold_match"], agr["SF_vs_SE"]["full"], B,
                                 agr["SF_vs_SE"]["median_first_divergence"], agr["SF_alt_vs_SF"]["full"], B)))
            res["SPEC"] = dict(metrics=m, agreement=agr, steps=a.spec_steps,
                               draft_note="stand-in drafts (draft_drop %g, seed %d): acceptance is a FAST vs EXACT "
                                          "comparison, NOT a representative rate" % (a.draft_drop, a.seed))
            save()
    # the verdict
    if "TF" in a.parts:
        fe = res["TF"]["pairs"]["F_vs_E"]
        diffs, noises, ses = dict(acc=fe["d_acc"], nll=fe["d_nll"]), dict(res["TF"]["noise"]), \
            dict(acc=fe["se_acc"], nll=fe["se_nll"])
        crit = list(TF_CRITERIA)
        if "SPEC" in a.parts:
            d2, n2, s2 = spec_verdict_inputs(res["SPEC"]["metrics"]["SE"], res["SPEC"]["metrics"]["SF"],
                                             res["SPEC"]["metrics"]["SF_rep"])
            diffs.update(d2)
            noises.update(n2)
            ses.update(s2)
            crit += list(SPEC_CRITERIA)
        v = verdict(diffs, noises, ses, crit)
        res["verdict"] = dict(fast_vs_exact=v["verdict"], detail=v, exact_gemm=a.exact_gemm,
                              scope="FAST (natural cuBLAS) vs EXACT (%s), batch %d and chunks of %d, PG-19 c16k "
                                    "prompts, %d gold tokens; not other batches or GPUs" % (a.exact_gemm, B, k,
                                                                                          a.n_gold))
        gates.append(gate("EA_VERDICT_FAST", v["verdict"] == "EQUIVALENT", True, "FAST vs EXACT: %s; %s" % (
            v["verdict"], "; ".join("%s d %+.5f allow %.5f cap %g %s" % (c, x["d"], x["allow"], x["cap"],
                                                                          "ok" if x["ok"] else "OUT")
                                    for c, x in v["criteria"].items()))))
        bands = []
        tfp = res["TF"]["pairs"]["F_vs_E"]
        bands.append(("EA-TF-AGREE", PRED["tf_agree"][0] <= tfp["agree"] <= PRED["tf_agree"][1]))
        bands.append(("EA-TF-MAXD", PRED["maxd_p50"][0] <= tfp["maxd"]["p50"] <= PRED["maxd_p50"][1]
                      and PRED["maxd_p99"][0] <= tfp["maxd"]["p99"] <= PRED["maxd_p99"][1]))
        ea = res["TF"]["arms"]["E"]
        bands.append(("EA-TF-ACC", PRED["acc_E"][0] <= ea["acc"] <= PRED["acc_E"][1]
                      and PRED["nll_E"][0] <= ea["nll"] <= PRED["nll_E"][1]))
        bands.append(("EA-TF-D", abs(tfp["d_acc"]) <= PRED["d_acc"] and abs(tfp["d_nll"]) <= PRED["d_nll"]))
        gates.append(gate("EA_R_registered_bands", True, False, "; ".join("%s %s" % (n, "IN" if ok else "OUT")
                                                                          for n, ok in bands)))
        res["bands"] = dict(bands)


def main(argv=None, env=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if env is None and not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    env = env or RealEnv()
    gates, res = [], {}
    meta = dict(args={kk: v for kk, v in vars(a).items()}, nosi_commit=os.environ.get("EVAL_COMMIT", "unknown"),
                bands=BANDS, noise_mult=NOISE_MULT, se_mult=SE_MULT, pred=PRED,
                rule="EQUIVALENT iff |d| <= max(NOISE_MULT x FAST's own spread, SE_MULT x se) + floor and |d| <= cap "
                     "for every criterion (module docstring)")
    try:
        meta.update(env.setup(a))
        os.makedirs(a.out)

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)
        try:
            run(a, env, gates, res, meta, save)
        except SystemExit:
            raise
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("EA_completed", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])))
        meta["peak_gb"] = env.peak_gb()
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_EVALACC_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


def check_record(path):
    """[evalsplit] Cell (a)'s evidence check: the accuracy record exists, every MUST gate in it passed, and its
    verdict is EQUIVALENT. Returns a short dict for the time cell's meta; SystemExit (a refusal) otherwise."""
    if not path or not os.path.exists(path):
        raise SystemExit("REFUSED: the accuracy record %r does not exist (run the evalacc cell first)" % path)
    try:
        d = json.load(open(path))
    except Exception as e:
        raise SystemExit("REFUSED: the accuracy record %s is not JSON: %s" % (path, e))
    gs = d.get("gates") or []
    musts = [g for g in gs if g.get("must")]
    failed = [g["name"] for g in musts if not g.get("passed")]
    v = (d.get("results") or {}).get("verdict") or {}
    if not musts or failed:
        raise SystemExit("REFUSED: the accuracy record %s has %s" % (
            path, "no MUST gate" if not musts else "failed MUST gates %s" % failed))
    names = {g["name"] for g in musts}
    for need in ("EA_VERDICT_FAST", "EA_NEG_detected", "EA_E_batch_invariance"):
        if need not in names:
            raise SystemExit("REFUSED: the accuracy record %s has no MUST gate %s" % (path, need))
    if v.get("fast_vs_exact") != "EQUIVALENT":
        raise SystemExit("REFUSED: the accuracy record %s says FAST %s" % (path, v.get("fast_vs_exact")))
    return dict(path=path, verdict=v.get("fast_vs_exact"), scope=v.get("scope"), musts=len(musts),
                nosi_commit=(d.get("meta") or {}).get("nosi_commit"))


if __name__ == "__main__":
    sys.exit(main())
