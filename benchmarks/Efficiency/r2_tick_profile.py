"""R2 step 7: M-T, the per-op DEVICE PROFILE of the captured tick and the fused twin step in it (docs/R2_FUSED.md
section 7). The setup is the TC gate's (r2_tick_capture_gate.py): B ragged documents of pg19-test-c16k, ONE
build_ragged (f4_fix True), BASE = TickEngine.from_r2, REF = the BI serial decode (its tokens are the teacher's),
EXACT GEMMs at m_fix = 4B, piece 3 v2, SG1's teacher-forced round (forced_schedule). Every arm runs that round on its own
copy of BASE with check off (as the gate's part T).

ARMS
  TE   the eager tick (record_te): the reference of TGF-EQ and the plan (rows and capacity of every tick).
  TF   FixedTickEngine, eager, with ranges on the instance (record_function): t.tick (the region), t.twin
       (twin.advance_unchecked), t.stage1, t.pool, t.select, t.pages (page_table), t.attn (piece 3's bi_paged_decode),
       t.gemm (ops.linear: the G-b PaddedLinear). In the tables, '-' is the rest of the region (the row cells, the
       store writes, rmsnorm, nosa_linear, rope, silu, the residuals, the embedding, lm_head, the token log) and 'out'
       is the device work between regions (the table upload, the decision read, the events).
  TG   CapturedTickEngine (one graph per capacity), no ranges: each window tick is cut at its graph launch and takes,
       kernel by kernel, the range of the TF tick of the same index (the same rows, capacity and op order; the name
       sequences must be equal).
  TGF  TG with the fused twin attached to every twin (r2_twin_fused.attach: advance_unchecked is ONE kernel per layer).
       First a round compared with TE tick by tick and at the end (the TC gate's compare_run): MUST TGF-EQ. Then a
       fresh TGF round, profiled; its ranges come from TG's tick with the twin ops removed on both sides (the rest must
       be equal), its fused kernels are t.twin.
THE WINDOW: --win-len ticks from the first tick >= --win-start whose window holds >= 2 ticks of each capacity (G_d =
(B, 1), G_v = (B (1 + K), K)), the same ticks in every arm. Per arm and capacity (the mean over its window ticks): GPU
ops and device ms per range, per name class (r2_p3_profile.CLASS_RULES) and per range x class; the per-tick period
(CUDA events at every tick, the whole round, the median per capacity).
MUST: TE-RAN, TGF-EQ, WIN (each arm's window cut into --win-len ticks), ALIGN (TG == TF and TGF == TG minus the twin, by
name, every window tick). REPORT: the tables; the registered predictions TK2-TK4 (commit message) get a verdict each.
OUTPUT: --out/summary.json, trace_{TF,TG,TGF}.json.gz. EXIT: the number of failed MUST gates; 2 refused.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0               # noqa: E402  (load_workload)
import r2_g0bi_gate as GB             # noqa: E402  (decode_loop, load_by_path, free, gate)
import r2_p3_profile as P             # noqa: E402  (parse_trace, aggregate, CLASS_RULES, ranged)
import r2_tick_capture_gate as TCG    # noqa: E402  (record_te, compare_run, run_forced, plan_caps)

RANGES = ("t.twin", "t.stage1", "t.pool", "t.select", "t.pages", "t.attn", "t.gemm")
PRED = dict(TG_twin_ops=dict(G_v=(16000, 21000), G_d=(5500, 7200)), TG_twin_ms=dict(G_v=(22.0, 45.0), G_d=(7.0, 16.0)),
            TGF_twin_ops=(32, 40), TGF_twin_ms=dict(G_v=1.0, G_d=0.5),
            TG_minus_TGF_ms=dict(G_v=(21.0, 44.5), G_d=(6.5, 15.5)))


def log(msg):
    print("[r2_tick_profile %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy; v2 7c826c1)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--deltas", default="0,1,17,63,64,65,777,4097")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--draft-drop", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--win-start", type=int, default=40)
    ap.add_argument("--win-len", type=int, default=6)
    a = ap.parse_args(argv)
    a.deltas = [int(x) for x in a.deltas.split(",") if x != ""]
    if len(a.deltas) != a.batch or min(a.deltas) < 0 or a.S - max(a.deltas) < 64 * 64:
        ap.error("--deltas needs %d non-negative values and every prompt >= 4096 tokens" % a.batch)
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    if a.m_fix < (1 + a.K) * a.batch or a.steps < 2 or a.warm < 1 or a.win_start < 4 or a.win_len < 4:
        ap.error("m_fix >= (1 + K) B, steps >= 2, warm >= 1, win-start >= 4, win-len >= 4")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested: tests_cpu/test_r2_fused.py F4)
def cap_label(cap, B):
    return "G_d" if tuple(cap) == (B, 1) else "G_v"


def pick_window(labels, start, n, need=2):
    """The first window [w, w + n), w >= start, holding >= need ticks of each of G_d and G_v; None if none."""
    for w in range(start, len(labels) - n + 1):
        seg = labels[w:w + n]
        if seg.count("G_d") >= need and seg.count("G_v") >= need:
            return w, w + n
    return None


def cut(ops, starts):
    """ops in device order cut at the sorted start times: [[op]] per start (ops before the first start dropped)."""
    seq = sorted(ops, key=lambda o: o["ts"])
    starts = sorted(starts)
    out = [[] for _ in starts]
    j = -1
    for o in seq:
        while j + 1 < len(starts) and o["ts"] >= starts[j + 1]:
            j += 1
        if j >= 0:
            out[j].append(o)
    return out


def tf_starts(ops):
    """The start of every region instance of an eager arm: the earliest op attributed to a t.tick range instance."""
    first = {}
    for o in ops:
        if o["range"] == "t.tick" and o.get("rid") is not None:
            first[o["rid"]] = min(first.get(o["rid"], o["ts"]), o["ts"])
    return sorted(first.values())


def graph_starts(ops, graph_corrs):
    """The start of every graph replay: the earliest op carrying a graph launch's correlation id."""
    first = {}
    for o in ops:
        if o.get("corr") in graph_corrs:
            first[o["corr"]] = min(first.get(o["corr"], o["ts"]), o["ts"])
    return sorted(first.values())


def eager_ranges(seg):
    """TF's own labels: the innermost range, 't.tick' -> '-' (the rest of the region), none -> 'out'."""
    return [("-" if o["range"] == "t.tick" else (o["range"] or "out")) for o in seg]


def transfer(seg, ref_names, ref_ranges):
    """Labels for a graph tick from its reference tick (names and labels in device order): equal name sequences map
    one to one; else, with the reference's t.twin ops and the tick's TWINF kernels removed, equal rests map one to one
    and the TWINF kernels are t.twin. Returns (labels or None, how)."""
    names = [o["name"] for o in seg]
    if names == list(ref_names):
        return list(ref_ranges), "equal"
    keep = [(n, r) for n, r in zip(ref_names, ref_ranges) if r != "t.twin"]
    idx = [i for i, o in enumerate(seg) if o["cls"] != "TWINF"]
    if [names[i] for i in idx] == [n for n, _ in keep]:
        out = ["t.twin"] * len(seg)
        for i, (_, r) in zip(idx, keep):
            out[i] = r
        return out, "equal without the twin"
    return None, "differs (%d vs %d ops)" % (len(names), len(ref_names))


def tick_table(seg, labels):
    """{'total': [ops, us], 'ranges': {r: [ops, us]}, 'classes': {c: [ops, us]}, 'cross': {r: {c: [ops, us]}}}."""
    t = dict(total=[0, 0.0], ranges={}, classes={}, cross={})
    for o, r in zip(seg, labels):
        for d in (t["total"], t["ranges"].setdefault(r, [0, 0.0]), t["classes"].setdefault(o["cls"], [0, 0.0]),
                  t["cross"].setdefault(r, {}).setdefault(o["cls"], [0, 0.0])):
            d[0] += 1
            d[1] += o["dur_us"]
    return t


def mean_tables(tables):
    """The mean of tick tables: ops and ms per tick."""
    n = len(tables)

    def acc(get):
        out = {}
        for t in tables:
            for k, v in get(t).items():
                x = out.setdefault(k, [0, 0.0])
                x[0] += v[0]
                x[1] += v[1]
        return {k: dict(ops=round(v[0] / n, 2), ms=round(v[1] / 1e3 / n, 4)) for k, v in sorted(out.items())}
    tot = [sum(t["total"][0] for t in tables), sum(t["total"][1] for t in tables)]
    cross = {}
    for r in sorted({r for t in tables for r in t["cross"]}):
        cross[r] = acc(lambda t, r=r: t["cross"].get(r, {}))
    return dict(ticks=n, total=dict(ops=round(tot[0] / n, 2), ms=round(tot[1] / 1e3 / n, 4)),
                ranges=acc(lambda t: t["ranges"]), classes=acc(lambda t: t["classes"]), cross=cross)


def verdicts(prof):
    """TK2-TK4 against the per-capacity means of TG and TGF: [(name, verdict, detail)]."""
    out = []
    tg, tgf = prof.get("TG"), prof.get("TGF")

    def v(ok):
        return "CONFIRMED" if ok else "REFUTED"
    for cap in ("G_v", "G_d"):
        if not tg or cap not in tg or not tgf or cap not in tgf:
            out.append(("TK %s" % cap, "UNDETERMINED", "no TG / TGF table for %s" % cap))
            continue
        tw = tg[cap]["ranges"].get("t.twin", dict(ops=0, ms=0.0))
        lo, hi = PRED["TG_twin_ops"][cap]
        mlo, mhi = PRED["TG_twin_ms"][cap]
        out.append(("TK2 %s" % cap, v(lo <= tw["ops"] <= hi and mlo <= tw["ms"] <= mhi),
                    "TG t.twin %.0f ops / %.3f ms per tick (predicted %d .. %d / %.1f .. %.1f)"
                    % (tw["ops"], tw["ms"], lo, hi, mlo, mhi)))
        fw = tgf[cap]["ranges"].get("t.twin", dict(ops=0, ms=0.0))
        lo, hi = PRED["TGF_twin_ops"]
        out.append(("TK3 %s" % cap, v(lo <= fw["ops"] <= hi and fw["ms"] <= PRED["TGF_twin_ms"][cap]),
                    "TGF t.twin %.0f ops / %.3f ms per tick (predicted %d .. %d / <= %.1f)"
                    % (fw["ops"], fw["ms"], lo, hi, PRED["TGF_twin_ms"][cap])))
        d = tg[cap]["total"]["ms"] - tgf[cap]["total"]["ms"]
        lo, hi = PRED["TG_minus_TGF_ms"][cap]
        out.append(("TK4 %s" % cap, v(lo <= d <= hi), "TG - TGF device ms per tick %.3f (TG %.3f, TGF %.3f; predicted "
                    "%.1f .. %.1f)" % (d, tg[cap]["total"]["ms"], tgf[cap]["total"]["ms"], lo, hi)))
    return out


# ------------------------------------------------------------------ GPU parts
class Probe:
    """on_tick: a CUDA event at every tick (the period); the profiler over the ticks [w0, w1) (started after tick
    w0 - 1's rows, stopped after tick w1 - 1's rows)."""

    def __init__(self, win, path):
        self.w0, self.w1, self.path = win[0], win[1], path
        self.t, self.prof, self.ev = 0, None, []

    def on_tick(self, rows, tokens, out):
        import torch
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev.append(e)
        if self.t == self.w0 - 1:
            torch.cuda.synchronize()
            self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
            self.prof.start()
        if self.t == self.w1 - 1 and self.prof is not None:
            torch.cuda.synchronize()
            self.prof.stop()
            self.prof.export_chrome_trace(self.path)
            self.prof = None
        self.t += 1

    def periods(self):
        import torch
        torch.cuda.synchronize()
        return [self.ev[i - 1].elapsed_time(self.ev[i]) for i in range(1, len(self.ev))]


def instrument(eng):
    """Ranges on the eager engine (its instance, its twins, and the SHARED ops / attention module); returns undo()."""
    undo = []
    for t in eng.twins:
        t.advance_unchecked = P.ranged("t.twin", t.advance_unchecked)
        undo.append(lambda t=t: delattr(t, "advance_unchecked"))
    for name, rng in (("stage1", "t.stage1"), ("pool", "t.pool"), ("select", "t.select"), ("page_table", "t.pages"),
                      ("forward", "t.tick")):
        had, old = name in vars(eng), getattr(eng, name)
        setattr(eng, name, P.ranged(rng, old))
        undo.append(lambda name=name, had=had, old=old: setattr(eng, name, old) if had else delattr(eng, name))
    attn, ops = eng.ops.attn, eng.ops
    old_attn, old_lin = attn.bi_paged_decode, ops.linear
    attn.bi_paged_decode = P.ranged("t.attn", old_attn)
    ops.linear = P.ranged("t.gemm", old_lin)
    undo.append(lambda: setattr(attn, "bi_paged_decode", old_attn))
    undo.append(lambda: setattr(ops, "linear", old_lin))

    def run():
        for f in reversed(undo):
            f()
    return run


def load_and_pack(path):
    """(ops, the correlation ids of the graph launches); the trace gzipped beside it (the .json removed)."""
    with open(path) as f:
        trace = json.load(f)
    ops = P.parse_trace(trace)
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    corrs = {(e.get("args") or {}).get("correlation") for e in ev
             if e.get("ph") == "X" and e.get("cat") in P.LAUNCH_CATS and "GraphLaunch" in str(e.get("name", ""))}
    del trace, ev
    with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    corrs.discard(None)
    return ops, corrs


def run_arm(eng, a, toks, win, path):
    probe = Probe(win, path)
    TCG.run_forced(eng, a, toks, probe.on_tick)
    return probe.periods()


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    import torch
    import nosi.cache_engine as CE
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    from nosi import r2_twin_fused as FU
    from nosi.r2_bi_decode import BIOps, attach_bi
    from nosi.r2_decode_path import build_ragged
    from nosi.r2_tick import TickEngine
    from nosi.r2_tick_capture import CapturedTickEngine
    from nosi.r2_tick_fixed import FixedTickEngine
    if not NL.R2_DECODE or NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
        print("REFUSED: needs NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS=0", file=sys.stderr)
        return 2
    docs, sha = G0.load_workload(a.workload, a.batch, a.S)
    os.makedirs(a.out)
    attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
    ops = BIOps(attn, a.m_fix, gemm="exact")
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    gates, res, prof, timing = [], {}, {}, {}
    meta = dict(batch=a.batch, S=a.S, deltas=a.deltas, steps=a.steps, K=a.K, LAMBDA=a.LAMBDA, draft_drop=a.draft_drop,
                seed=a.seed, m_fix=a.m_fix, warm=a.warm, bi=ops.describe(), biattn=a.biattn, class_rules=P.CLASS_RULES,
                predictions=PRED, workload_sha256=sha, nosi_commit=os.environ.get("TICKPROF_COMMIT", "unknown"),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))

    def save():
        json.dump(dict(meta=meta, gates=gates, results=res, profile=prof, timing=timing),
                  open(os.path.join(a.out, "summary.json"), "w"), indent=1, sort_keys=True, default=str)
    n_ref = a.steps + 16
    prompts = [docs[b, :a.S - a.deltas[b]] for b in range(a.batch)]
    with torch.inference_mode():
        cache, st, first = build_ragged(model, prompts, n_ref + 80, f4_fix=True)
        base = TickEngine.from_r2(model, cache, st, ops, draft_drop=a.draft_drop, drop_seed=a.seed, check=False)
        attach_bi(st, ops)
        pos0 = st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)
        r = GB.decode_loop(model, cache, first, pos0, n_ref, keep=False)
        toks = r["tokens"].cpu()
        del cache, st, r
        GB.free()
        te = TCG.record_te(base.clone(), a, toks)
        plan = [x["rows"] for x in te["ticks"]]
        probe_eng = FixedTickEngine.from_engine(base, v_cap=a.K)
        labels = [cap_label(c, a.batch) for c in TCG.plan_caps(plan, probe_eng)]
        del probe_eng
        win = pick_window(labels, a.win_start, a.win_len)
        gates.append(GB.gate("TE-RAN", win is not None, True, "%d ticks (G_v %d, G_d %d); window %s" % (
            len(plan), labels.count("G_v"), labels.count("G_d"), win)))
        res["plan"] = dict(ticks=len(plan), labels=labels, window=win)
        save()
        if win is None:
            print("R2_TICK_PROFILE_DONE failed=1", flush=True)
            return 1
        # TGF-EQ: the fused twin inside the captured tick, bitwise against TE
        try:
            eng = CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm, check=False)
            for t in eng.twins:
                FU.attach(t)
            rr = TCG.compare_run(eng, a, toks, te)
            gates.append(GB.gate("TGF-EQ", rr["first"] is None and rr["error"] is None, True,
                                 "fused-twin captured tick vs TE: first difference %s; error %s; %d ticks; captures %d"
                                 % (rr["first"], rr["error"], rr["ticks"], eng.stats.get("captures", -1))))
            del eng, rr
        except Exception as e:                                                # noqa: BLE001
            traceback.print_exc()
            gates.append(GB.gate("TGF-EQ", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        GB.free()
        save()
        segs, starts_of = {}, {}
        for arm in ("TF", "TG", "TGF"):
            log("arm %s, window %s" % (arm, win))
            path = os.path.join(a.out, "trace_%s.json" % arm)
            try:
                if arm == "TF":
                    eng = FixedTickEngine.from_engine(base, v_cap=a.K, check=False)
                    undo = instrument(eng)
                    try:
                        per = run_arm(eng, a, toks, win, path)
                    finally:
                        undo()
                else:
                    eng = CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm, check=False)
                    if arm == "TGF":
                        for t in eng.twins:
                            FU.attach(t)
                    per = run_arm(eng, a, toks, win, path)
                del eng
                GB.free()
                ops_, corrs = load_and_pack(path)
                st_ = tf_starts(ops_) if arm == "TF" else graph_starts(ops_, corrs)
                segs[arm] = cut(ops_, st_)
                starts_of[arm] = len(st_)
                timing[arm] = {lab: dict(n=len(v), median_ms=round(statistics.median(v), 3)) for lab, v in (
                    (lab, [p for i, p in enumerate(per) if labels[i + 1] == lab]) for lab in ("G_v", "G_d")) if v}
                gates.append(GB.gate("WIN-%s" % arm, len(st_) == win[1] - win[0], True,
                                     "%d tick starts in the window (expected %d)" % (len(st_), win[1] - win[0])))
            except Exception as e:                                            # noqa: BLE001
                traceback.print_exc()
                gates.append(GB.gate("ARM-%s" % arm, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            save()
        labs = {}
        if "TF" in segs:
            labs["TF"] = [eager_ranges(s) for s in segs["TF"]]
        for arm, ref in (("TG", "TF"), ("TGF", "TG")):
            if arm not in segs or ref not in labs or len(segs[arm]) != len(segs[ref]):
                gates.append(GB.gate("ALIGN-%s" % arm, False, True, "no tick-by-tick reference (%s)" % ref))
                continue
            out, hows = [], []
            for s, rs, rl in zip(segs[arm], segs[ref], labs[ref]):
                lab, how = transfer(s, [o["name"] for o in rs], rl)
                out.append(lab)
                hows.append(how)
            ok = all(x is not None for x in out)
            gates.append(GB.gate("ALIGN-%s" % arm, ok, True, "per tick: %s" % hows))
            if ok:
                labs[arm] = out
        for arm in ("TF", "TG", "TGF"):
            if arm not in labs:
                continue
            per_cap = {}
            for k, (s, lab) in enumerate(zip(segs[arm], labs[arm])):
                per_cap.setdefault(labels[win[0] + k], []).append(tick_table(s, lab))
            prof[arm] = {cap: mean_tables(v) for cap, v in per_cap.items()}
        preds = verdicts(prof)
        res["predictions"] = [dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds]
        save()
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                g["detail"][:400]), flush=True)
    for p in preds:
        print("PREDICTION %s %s: %s" % p, flush=True)
    for arm in ("TF", "TG", "TGF"):
        for cap, t in sorted(prof.get(arm, {}).items()):
            rng = ", ".join("%s %.0f/%.3f" % (k, v["ops"], v["ms"]) for k, v in t["ranges"].items())
            cls = ", ".join("%s %.0f/%.3f" % (k, v["ops"], v["ms"]) for k, v in t["classes"].items())
            print("REPORT %s %s (%d ticks): %.0f ops / %.3f ms per tick; ranges %s; classes %s; period %s" % (
                arm, cap, t["ticks"], t["total"]["ops"], t["total"]["ms"], rng, cls, timing.get(arm, {}).get(cap)),
                flush=True)
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("R2_TICK_PROFILE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 125)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:                                                         # noqa: BLE001
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
