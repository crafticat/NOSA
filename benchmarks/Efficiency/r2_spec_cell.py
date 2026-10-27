"""Phase 2, rungs L3 and L4 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the
GPU cell of the SPECULATIVE ROUND RUNNER (nosi/nosi/spec_round.py; opt-in flag --spec K). A NEW script: r2_base_cell.py
is not edited (its functions are imported), so every older cell keeps its bytes. Step script:
benchmarks/Efficiency/r2_spec_step.sh (cells sp3m, sp3a, sp3b).

ONE PROCESS, ONE MODEL, the arms in this order, each with its own shipped prefill:
  SH   the shipped decode (r2_base_cell.run_shipped) at the cell's C and B, greedy, free-running, --steps steps; it
       records its tokens and the logits fingerprint (max, float64 sum) of every (request, step). The same-job baseline.
  L3   spec_round level L3: per round K resident-only draft steps (undone), then K serial verify steps (the ragged
       step: gemm natural; twin --spec-twin, default FUSED: SCORING_rg1.md, the fused twin's step is +0.6 / +1.7 % of
       cached NOSA and its logits were bitwise SH's in rg1f, while the torch twin costs +13-14 %), accept / correct /
       commit per request.
  L4X  spec_round level L4, EXACT joint pass (gemm rowgroup: the shipped GEMM shape): the token proof of L4.
  L4F  spec_round level L4, FAST joint pass (gemm natural, one residency + extras store): the timing of L4.
Every speculative arm starts from SH's first token (inputs[:, 0]) and runs FREE (greedy): its committed tokens must be
SH's tokens. It runs --spec-rounds rounds (default ceil(steps / K): no request passes steps + K - 1 tokens).

MEASURED per speculative arm (CUDA events; one host sync per round, after the round):
  committed tok/s      per round = the round's committed tokens / the round's span; MEDIAN over the steady rounds (round
                       0, the first launches and the graph captures, and the probe rounds excluded); MEAN = the steady
                       rounds' tokens / their time; EFFECTIVE = every committed token / the arm's wall time of the decode
                       (the L4 prelude, the syncs and round 0 included). Each against SH's steady / mean / effective
                       tok/s (B x 1000 / the SH step), in the same job.
  acceptance           accepted drafts / valid drafts; the histogram of n (accepted drafts per request and round); the
                       full-accept fraction; committed tokens per request and round.
  phases per round     L3: draft, verify, commit; L4: joint, truncate, draft, commit (ms, medians over steady rounds).
  KV at verify         NEW: the (head, request) blocks the verify fetched from the host (count and bytes per round);
                       READY: the verify rows' selected slots minus NEW (FAST: the union fetches, so READY is an upper
                       bound); IN FLIGHT: 0 (no background prefetch in L3 / L4: --spec-prefetch is rung L5); the WAIT:
                       the CUDA-event spans of the verify's host gathers on the probe rounds (ms per round).
  token check          the first min(committed, steps) tokens of every request against SH's, BITWISE; the logits
                       fingerprints of those committed rows against SH's (EXACT arms: equal bits expected).
  draft                skipped slots per (layer, KV head, request) per draft call; redirected slots; the rollback
                       journal counters (destroyed, saved, overflow, overflow_dead, redirected); truncations.
MUST GATES: SH-RUN; per arm <arm>-RUN, <arm>-ERR (no body refusal bit, no twin overflow flag), <arm>-COMMIT (every
request committed at least one token per round, and the per-round sums add up to the committed total); per EXACT arm
(L3, L4X) <arm>-TOK (every checked committed token bitwise SH's, at least one checked) and <arm>-FP (every checked
committed row's logits fingerprint bitwise SH's). sp3b at 6de1cfd had no token gate: its EXACT arms lost SH's tokens
and the cell reported 0 failed MUST gates (SCORING_sp3b.md).
REGISTERED PREDICTIONS PRED_SPEC: TBD by the lead (a None value gives REPORT).
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402

ARMS = ("L3", "L4X", "L4F")
ARM_OPS = dict(L3=dict(level="L3", mode="exact", gemm="rowgroup"), L4X=dict(level="L4", mode="exact", gemm="rowgroup"),
               L4F=dict(level="L4", mode="fast", gemm="natural"))
EXACT_ARMS = ("L3", "L4X")
PHASES = {"L3": ("draft", "verify", "commit"), "L4": ("joint", "truncate", "draft", "commit")}
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill every None and the stamp (`date -u`, L33) in a commit BEFORE the run.
#   <arm>_tok      True / False: the arm's committed tokens equal SH's, bitwise (L3, L4X)
#   L4F_agree      the lowest fraction of L4F's committed tokens equal to SH's (FAST is not bitwise)
#   <arm>_ratio    (low, high): the arm's median steady committed tok/s / SH's steady tok/s
#   accept         (low, high): accepted / valid drafts (L3; the same draft step in every arm)
#   L4F_vs_L3      (low, high): L4F's median steady committed tok/s / L3's (the plan's L4 gate: "faster than L3")
# Inputs for the bands (SCORING_dr1.md): at C 128 B 216 the draft step is 0.947 SH and the ceiling of speculation is
# about +4 %; at C 73 B 318 dr73 (registered 0.80 [0.60, 0.95]) is pending.
# Lead registration 2026-10-27T20:08:53Z. Basis: dr1 (draft 0.947 SH), vr1 MEASURED (VRC joint pass 4.144 SH steps,
# VRF FAST 6.300), the builder's round model (docs/NOTES_spec.md section 6) with vr1's measured joint pass: L4F round
# about 404 + 30 + 120 + 60 = 614 ms for at most 3 tokens vs SH 191 ms -> about 0.31. Both cells use these bands; the
# expectation is that NO arm beats SH (the plan's stop rule at C 128 B 216; sp3b is the small-cache test).
PRED_SPEC = dict(stamp="2026-10-27T20:08:53Z", L3_tok=True, L4X_tok=True, L4F_agree=0.99, L3_ratio=(0.3, 0.5),
                 L4X_ratio=(0.2, 0.45), L4F_ratio=(0.2, 0.5), accept=(0.97, 1.0), L4F_vs_L3=(0.6, 1.2))


def log(msg):
    print("[r2_spec %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=0, help="THE FLAG: K drafts per round (0 = off: the cell refuses)")
    ap.add_argument("--spec-arms", default="L3,L4X,L4F", help="a subset of %s, in run order" % ",".join(ARMS))
    ap.add_argument("--spec-rounds", type=int, default=0, help="rounds per arm (0: ceil(steps / K))")
    ap.add_argument("--spec-journal-blocks", type=int, default=256,
                    help="the rollback's saved slots per layer and step (a lost slot past it comes back from the host)")
    ap.add_argument("--spec-extra", type=int, default=16, help="L4F: X extras slots per request")
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"),
                    help="fused (default, rg1f: +0.6 / +1.7 %% vs cached NOSA, logits bitwise SH's at B 216) or torch "
                         "(the shipped summary bits; rg1: +13-14 %%, about 8 ms of small launches per step)")
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"),
                    help="a draft loads a block filled in its own group from the GPU staging")
    ap.add_argument("--spec-prefetch", default="off", choices=("off", "on"),
                    help="rung L5 (background prefetch of the draft misses): not wired to a transport here")
    ap.add_argument("--probe-every", type=int, default=16, help="a probe round every N rounds (round 0 excluded)")
    a = ap.parse_args(argv)
    a.spec_arms = [x for x in a.spec_arms.split(",") if x]
    if not a.spec_arms or any(x not in ARMS for x in a.spec_arms) or len(set(a.spec_arms)) != len(a.spec_arms):
        ap.error("--spec-arms: distinct names from %s" % (ARMS,))
    if a.spec != 0 and not 2 <= a.spec <= 18:
        ap.error("--spec K in 2 .. 18 (L4 needs K >= 2; the horizon steps + 3 K + 8 fits the host rows steps + 64)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000 (the shipped total_cis holds S + 1024)")
    if a.spec_rounds < 0 or a.spec_journal_blocks < 1 or a.spec_extra < 2 or a.probe_every < 2:
        ap.error("--spec-rounds >= 0, --spec-journal-blocks >= 1, --spec-extra >= 2, --probe-every >= 2")
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
WARM = 1                           # spec_round.SpecRunner.run: the plain first step after the prefill


def rounds_for(steps: int, K: int, rounds: int = 0) -> int:
    """The rounds of one arm: --spec-rounds, else ceil((steps - WARM) / K) (WARM plain steps commit first)."""
    return int(rounds) if rounds > 0 else -(-(int(steps) - WARM) // int(K))


def steady_rounds(G: int, every: int):
    """Round 0 (the first launches, the graph captures) and the probe rounds excluded."""
    return [g for g in range(1, G) if not (every > 0 and g % every == 0)]


def token_check(out, count, inputs, steps: int):
    """out (B, >= steps) committed tokens, count (B,) committed per request, inputs (B, steps + 1) SH's tokens
    (inputs[:, j + 1] = SH's token after step j). Compares the first min(count, steps) columns of each request.
    Returns dict(n_checked, n_unequal, equal_fraction, first = (request, column) of the earliest unequal column or None,
    min_checked)."""
    import torch
    out, inputs = out.detach().cpu(), inputs.detach().cpu()
    B = out.shape[0]
    n = torch.as_tensor(count).cpu().clamp(max=steps)
    col = torch.arange(steps).view(1, steps)
    chk = col < n.view(B, 1)
    bad = chk & (out[:, :steps] != inputs[:, 1:steps + 1])
    nc, nb = int(chk.sum()), int(bad.sum())
    first = None
    if nb:
        idx = bad.nonzero()
        c0 = int(idx[:, 1].min())
        first = (int(idx[idx[:, 1] == c0][0, 0]), c0)
    return dict(n_checked=nc, n_unequal=nb, equal_fraction=(1.0 if nc == 0 else (nc - nb) / nc), first=first,
                min_checked=int(n.min()) if B else 0)


def fingerprint_check(fp, count, fp_sh, steps: int):
    """The logits fingerprints (max float32, sum float64) of the committed rows (B, >= steps) against SH's (B, steps),
    BITWISE, over the first min(count, steps) columns. Returns dict(n_checked, n_unequal, equal_fraction, first)."""
    import torch
    fm, fs = fp[0].detach().cpu()[:, :steps].contiguous(), fp[1].detach().cpu()[:, :steps].contiguous()
    sm, ss = fp_sh[0].detach().cpu()[:, :steps].contiguous(), fp_sh[1].detach().cpu()[:, :steps].contiguous()
    B = fm.shape[0]
    n = torch.as_tensor(count).cpu().clamp(max=steps)
    chk = torch.arange(steps).view(1, steps) < n.view(B, 1)
    eq = (fm.view(torch.int32) == sm.view(torch.int32)) & (fs.view(torch.int64) == ss.view(torch.int64))
    bad = chk & ~eq
    nc, nb = int(chk.sum()), int(bad.sum())
    first = None
    if nb:
        idx = bad.nonzero()
        c0 = int(idx[:, 1].min())
        first = (int(idx[idx[:, 1] == c0][0, 0]), c0)
    return dict(n_checked=nc, n_unequal=nb, equal_fraction=(1.0 if nc == 0 else (nc - nb) / nc), first=first)


def round_rates(committed, round_ms, steady, wall_s, total_committed):
    """committed tok/s: median over the steady rounds of (round tokens / round ms), the mean (steady tokens / steady
    time) and the effective rate (every committed token / the wall time). Lists per round."""
    per = [1e3 * committed[g] / round_ms[g] for g in steady if round_ms[g] > 0]
    st_tok = sum(committed[g] for g in steady)
    st_ms = sum(round_ms[g] for g in steady)
    return dict(median_tok_s=(round(statistics.median(per), 1) if per else None),
                mean_tok_s=(round(1e3 * st_tok / st_ms, 1) if st_ms > 0 else None),
                effective_tok_s=(round(total_committed / wall_s, 1) if wall_s and wall_s > 0 else None),
                steady_rounds=len(per), round_ms_median=(round(statistics.median([round_ms[g] for g in steady]), 3)
                                                         if steady else None),
                tokens_per_round_mean=(round(st_tok / len(steady), 3) if steady else None))


def sh_rates(B: int, sh):
    """SH's steady, mean and effective tok/s (B x 1000 / the steady median / the steady mean; r2_base_cell.rates)."""
    st = sh["steady"]
    return dict(median_tok_s=sh["rates"]["steady_tok_s"],
                mean_tok_s=(round(1e3 * B / st["mean_ms"], 1) if st.get("mean_ms") else None),
                effective_tok_s=sh["rates"]["effective_tok_s"])


def phase_medians(level: str, phases, steady):
    names = PHASES[level]
    out = {}
    for i, nm in enumerate(names):
        xs = [phases[g][i] for g in steady if g < len(phases) and i < len(phases[g])]
        out[nm] = round(statistics.median(xs), 3) if xs else None
    return out


def kv_at_verify(rd, steady, rows_per_round: int, L: int, H: int, B: int, topk: int, block_bytes: int):
    """NEW (host fetches), READY (selected slots minus NEW), IN FLIGHT (0: no prefetch in L3 / L4) per round, and the
    gather WAIT on the probe rounds (ms per round)."""
    blocks = [rd["round_blocks"][g] for g in steady if g < len(rd["round_blocks"])]
    sel = rows_per_round * L * H * B * topk
    waits = [w for w in rd["probe_wait_ms"] if w is not None]
    nb = statistics.fmean(blocks) if blocks else None
    return dict(new_blocks_per_round=(round(nb, 1) if nb is not None else None),
                new_MB_per_round=(round(nb * block_bytes / 1e6, 3) if nb is not None else None),
                ready_slots_per_round=(round(sel - nb, 1) if nb is not None else None), selected_slots_per_round=sel,
                in_flight_blocks=0, in_flight_note="no background prefetch in L3 / L4 (--spec-prefetch is rung L5)",
                wait_ms_median=(round(statistics.median(waits), 3) if waits else None), wait_rounds=len(waits),
                block_bytes=block_bytes)


def verdicts_spec(sh, sp, B: int):
    """The registered predictions PRED_SPEC against the arms' numbers: [(name, verdict, detail)]; a registered None
    gives REPORT; an arm that did not run gives NOT_RUN."""
    P = PRED_SPEC
    out = []
    shr = sh_rates(B, sh) if sh else None

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))
    for arm in ARMS:
        r = (sp or {}).get(arm)
        if r is None:
            out.append(("%s-RUN" % arm, "NOT_RUN", "the arm did not run"))
            continue
        tc = r["tokens"]
        if arm in EXACT_ARMS:
            want = P["%s_tok" % arm]
            got = tc["n_checked"] > 0 and tc["n_unequal"] == 0
            v = "REPORT" if want is None else ("CONFIRMED" if got == bool(want) else "REFUTED")
            out.append(("%s-TOK" % arm, v, "committed tokens equal to SH's: %d of %d unequal (first %s); fingerprints "
                        "%d of %d unequal; registered %s" % (tc["n_unequal"], tc["n_checked"], tc["first"],
                                                            r["fingerprints"]["n_unequal"],
                                                            r["fingerprints"]["n_checked"], want)))
        else:
            want = P["L4F_agree"]
            v = "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else "REFUTED")
            out.append(("L4F-AGREE", v, "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s"
                        % (tc["equal_fraction"], tc["first"], want)))
        ratio = (r["rates"]["median_tok_s"] / shr["median_tok_s"]) if (shr and shr["median_tok_s"] and
                                                                         r["rates"]["median_tok_s"]) else None
        band("%s-RATIO" % arm, ratio, P["%s_ratio" % arm])
    l3 = (sp or {}).get("L3")
    if l3 is not None:
        band("SP-ACCEPT", l3["acceptance"]["rate"], P["accept"])
    l4f = (sp or {}).get("L4F")
    if l3 is not None and l4f is not None and l3["rates"]["median_tok_s"] and l4f["rates"]["median_tok_s"]:
        band("L4F-VS-L3", l4f["rates"]["median_tok_s"] / l3["rates"]["median_tok_s"], P["L4F_vs_L3"])
    else:
        out.append(("L4F-VS-L3", "NOT_RUN", "L3 or L4F did not run"))
    return out


def tok_gates(arm, rec):
    """The token MUST gates of an EXACT arm (EXACT_ARMS; SCORING_sp3b.md: sp3b's L3 / L4X lost SH's tokens from column 1
    and the cell still reported 0 failed MUST gates). <arm>-TOK: at least one token checked and every checked committed
    token bitwise SH's; <arm>-FP: every checked committed row's logits fingerprint bitwise SH's. rec: summarize()'s
    record, or None (the check did not run: both FAIL). A FAST arm has none: [(name, must, passed, detail)]."""
    if arm not in EXACT_ARMS:
        return []
    if rec is None:
        return [("%s-TOK" % arm, True, False, "no token check (no record)"),
                ("%s-FP" % arm, True, False, "no fingerprint check (no record)")]
    tc, fc = rec["tokens"], rec["fingerprints"]
    return [("%s-TOK" % arm, True, tc["n_checked"] > 0 and tc["n_unequal"] == 0,
             "committed tokens bitwise SH's: %d of %d unequal (first %s; min checked per request %d)" % (
                 tc["n_unequal"], tc["n_checked"], tc["first"], tc["min_checked"])),
            ("%s-FP" % arm, True, fc["n_checked"] > 0 and fc["n_unequal"] == 0,
             "logits fingerprints bitwise SH's: %d of %d unequal (first %s)" % (fc["n_unequal"], fc["n_checked"],
                                                                               fc["first"]))]


def arm_gates(arm, res, rec=None):
    """The MUST gates of one speculative arm's result: [(name, must, passed, detail)]; an EXACT arm's token gates
    (tok_gates) from summarize()'s record rec."""
    rd = res["read"]
    g = []
    body_err, overflow = rd["errors"]
    g.append(("%s-ERR" % arm, True, body_err == 0 and not any(overflow),
              "body refusal bits %s; twin overflow flags %s" % (body_err, sorted(set(overflow)))))
    cnt = rd["count"]
    warm = rd["warm"] * len(cnt)
    ok = min(cnt) >= rd["rounds"] + rd["warm"] and sum(rd["round_committed"]) + warm == sum(cnt)
    g.append(("%s-COMMIT" % arm, True, ok, "%d warm step(s) + %d rounds; committed per request min %d max %d; warm %d "
              "+ per-round sum %d, total %d" % (rd["warm"], rd["rounds"], min(cnt), max(cnt), warm,
                                                sum(rd["round_committed"]), sum(cnt))))
    return g + tok_gates(arm, rec)


def summarize(arm, res, inputs, fp_sh, sh, B: int, steps: int, dims):
    """The arm's record (no tensors) from run_spec's result. dims: L, H, topk, block_bytes."""
    rd = res["read"]
    level = ARM_OPS[arm]["level"]
    K = res["K"]
    G = rd["rounds"]
    steady = steady_rounds(G, res["probe_every"])
    rates = round_rates(rd["round_committed"], rd["round_ms"], steady, rd["wall_s"], sum(rd["count"]))
    shr = sh_rates(B, sh) if sh else None
    vs_sh = {k: (round(rates[k] / shr[k], 4) if (shr and shr.get(k) and rates.get(k)) else None)
             for k in ("median_tok_s", "mean_tok_s", "effective_tok_s")}
    hist = rd["acc_hist"]
    acc = dict(rate=(rd["accepted"] / rd["valid"] if rd["valid"] else None), accepted=rd["accepted"],
               valid=rd["valid"], hist=hist, full_fraction=(rd["full"] / (B * G) if G else None),
               committed_per_request_round=(sum(rd["round_committed"]) / (B * G) if G else None))
    calls = rd["draft_calls"]                            # draft decode_fixed calls: one per layer of a draft step / row
    skip = (sum(rd["draft_skipped"]) / (calls * dims["H"] * B)) if calls else None
    return dict(arm=arm, level=level, K=K, rounds=G, rates=rates, vs_sh=vs_sh, acceptance=acc,
                phases_median=phase_medians(level, rd["phases"], steady),
                kv_at_verify=kv_at_verify(rd, steady, K, dims["L"], dims["H"], B, dims["topk"], dims["block_bytes"]),
                tokens=token_check(res["out"], rd["count"], inputs, steps),
                fingerprints=fingerprint_check(res["fp"], rd["count"], fp_sh, steps),
                draft=dict(skip_per_stream_per_call=skip, calls=calls, steps=rd["draft_steps"],
                           redirected=sum(rd["draft_redirected"])),
                journal=rd["journal"], truncations=rd["truncations"], groups=rd["groups"],
                prelude_ms=rd["prelude_ms"], wall_s=rd["wall_s"], count_min=min(rd["count"]),
                count_max=max(rd["count"]), verify=rd["verify"], errors=rd["errors"], prefill_s=res["prefill_s"],
                first_token_agreement=res["first_token_agreement"], meta=res["meta"])


# ------------------------------------------------------------------ the speculative arm (GPU)
def run_spec(model, ids, inputs, steps: int, clock, opt):
    """ONE speculative arm: its own shipped prefill (SH's cache and max_gen_len), ragged_engine.attach (horizon steps +
    3 K + 8), spec_round.SpecRunner from SH's first token, opt["rounds"] rounds. opt: arm, K, rounds, journal, extra,
    twin, body, graph, redirect, probe_every."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    cfg = model.config
    model.has_buffers = False                            # the shipped warm-up state is not used: release it
    for name in BC.MODEL_WARM_BUFFERS:
        if hasattr(model, name):
            setattr(model, name, None)
    for layer in model.layers:
        for name in BC.LAYER_WARM_STATE:
            if hasattr(layer, name):
                setattr(layer, name, None)
    gc.collect()
    torch.cuda.empty_cache()
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"],
                       graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()                             # the released shipped tables
    a = ARM_OPS[opt["arm"]]
    sops = SR.SpecOps(level=a["level"], K=K, mode=a["mode"], gemm=a["gemm"], extra=opt["extra"],
                      journal=opt["journal"], redirect=opt["redirect"], probe_every=opt["probe_every"])
    runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
    G = rounds_for(steps, K, opt["rounds"])
    runner.run(steps, rounds=G)
    rd = runner.read()
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=round(prefill_s, 1), first_token_agreement=first_agree,
               meta=dict(rs.meta, spec=runner.meta))
    runner.detach()
    del runner, rs, cache
    return res


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.spec == 0:
        print("REFUSED: --spec K is the flag of this cell (0 = off)", file=sys.stderr)
        return 20
    if a.spec_prefetch == "on":
        print("REFUSED: --spec-prefetch on is rung L5: pool_prefetch.PoolPrefetch hooks the SHIPPED engine's "
              "decode_update (its reader event and commit wait) and reads seq_length; on the per-request engine and "
              "inside a rollback group it would write pool slots outside the journal. Not wired here.", file=sys.stderr)
        return 20
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
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs = [], None, {}, None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    # one (layer, KV head, request) block: 64 rows of K and V (bf16, head_dim D) and 64 bias values (bf16)
    dims = dict(L=len(model.layers), H=H, topk=64, block_bytes=64 * D * 2 * 2 + 64 * 2)
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps, C %d)" % (a.batch, a.steps, BC.C_BASE + a.pool))
        fp_sh = (torch.zeros((a.batch, a.steps), dtype=torch.float32, device="cuda"),
                 torch.zeros((a.batch, a.steps), dtype=torch.float64, device="cuda"))
        try:
            out = BC.run_shipped(model, ids, a.steps, clock, fp=fp_sh)
            pa, pr = BC.gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(steady=steady, rates=BC.rates(a.batch, a.steps, out["periods"], steady),
                      rewarms=out["rewarms"], prefill_s=out["prefill_s"], gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, device_used_end_GB=out["device_used_end_GB"])
            sh["tok_s"] = sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        mem["after_SH"] = BC.host_mem()
        BC.free()
        for arm in a.spec_arms:
            if inputs is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (the arm starts from SH's first token)"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, extra=a.spec_extra,
                       twin=a.spec_twin, body=a.spec_body, graph=a.spec_graph == "on",
                       redirect=a.spec_redirect == "on", probe_every=a.probe_every)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_spec(model, ids, inputs, a.steps, clock, opt)
                pa, pr = BC.gpu_peak()
                rec = summarize(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend(arm_gates(arm, res, rec))
                del res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts_spec(sh, sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, spec_arms=a.spec_arms, spec_rounds=a.spec_rounds,
                spec_journal_blocks=a.spec_journal_blocks, spec_extra=a.spec_extra, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                spec_prefetch=a.spec_prefetch, probe_every=a.probe_every, predictions=PRED_SPEC, dims=dims,
                host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-12s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB, prefill %s s" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"], sh["prefill_s"]))
    for name in a.spec_arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; acceptance %s "
              "(full %s); phases %s; KV at verify %s; tokens equal %.6f (first %s); fingerprints equal %.6f; draft skip "
              "per stream %s; journal overflow %s; peak %s GB" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], r["acceptance"]["rate"], r["acceptance"]["full_fraction"],
                  r["phases_median"], r["kv_at_verify"], r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r["fingerprints"]["equal_fraction"], r["draft"]["skip_per_stream_per_call"],
                  sum(r["journal"]["overflow"]), r["gpu_peak_alloc_GB"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_spec: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
