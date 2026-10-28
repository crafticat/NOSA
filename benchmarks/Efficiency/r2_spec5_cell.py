"""Phase 2, rung L5 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU
cell of the BACKGROUND PREFETCH of the draft misses (nosi/nosi/spec_prefetch.py; opt-in --spec-prefetch on). A NEW
script: r2_spec_cell.py and r2_base_cell.py are not edited (their functions are imported), so every older cell keeps its
bytes. Step script: benchmarks/Efficiency/r2_spec5_step.sh (cells sp5m, sp5a, sp5b).

ONE PROCESS, ONE MODEL, the arms in this order, each with its own shipped prefill:
  SH    the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline, run FIRST
  L3    rung L3 (r2_spec_cell arm L3): K resident-only drafts (undone), K serial verify steps, accept / commit
  L3P   L3 + the prefetch: each draft step's skipped blocks go to the C++ packer in the background (lag 1); one commit
        per epoch, OUTSIDE every rollback group, puts them into soft-pinned pool slots before the verify group; the
        verify finds them as pool hits
  L4P   rung L4P (the paper's schedule, r2_spec_cell arm L4P), the requested-KV pin off
  L4PP  L4P + the prefetch + the requested-KV pin (paper 3.3): the draft rows' and drafts' misses of one round are
        committed at the next round's start (after RequestPin.apply), for the next joint pass
Every speculative arm starts from SH's first token and runs FREE (greedy): its committed tokens must be SH's.

MEASURED per prefetch arm, per round (CUDA events and device counters; spec_prefetch.SpecPrefetch.report):
  KV at verify (paper 4.3)  the verify-selected slots split into READY (resident: window or pool hits; READY_PF the
                       prefetched pool hits whose job had landed before the verify started), IN FLIGHT (prefetched pool
                       hits whose job had not landed: the verify waited its event) and NEW (the urgent GPU gather);
                       blocks and bytes (block_bytes = K + V + bias of one (layer, head, request) block) per round.
  waits                IN FLIGHT: the main-stream waits for the layers' commits (events A_l / B_l around each wait,
                       summed over the layers) and the boundary stall (events at the entry and the exit of the commit
                       point: the GPU idle time while the host waits the newest job's submission and enqueues the
                       commits); NEW: the gather spans of the probe rounds (r2_spec_cell's kv_at_verify); READY: 0.
  useful prefetch      committed blocks read by a verify row (ready / in flight), by a draft request only, evicted
                       (pin overflow / pin broken), unread in their slot at the release, moved; refetched (a verify host
                       miss of a block that the prefetch delivered in the same epoch).
  pin overflows        commit_overflow (no unpinned slot at the commit) + evict_overflow (a pinned block evicted while
                       every unclaimed slot of its row was pinned).
  packer               jobs in flight at the verify (backlog), delivery ms (issue -> arrival probes; "late" samples
                       are upper bounds), lane refusals, the worker's stats, the host ms of each seam.
MUST GATES: SH-RUN, PF-BUILD (when a prefetch arm runs); per arm <arm>-RUN, <arm>-ERR, <arm>-COMMIT, <arm>-TOK,
<arm>-FP (every arm is EXACT: tokens and logits fingerprints bitwise SH's); per prefetch arm <arm>-PF-ACT (jobs > 0
and prefetched verify hits > 0), <arm>-PF-CONS (pin conservation: committed == read + evicted + unread + moved; every
pin and tag ended; requested == committed + dup + overflow; READY_resident >= 0 in every round), <arm>-PF-PIN (no
prefetched block evicted while an unpinned unclaimed slot existed), <arm>-PF-NEW (the split's NEW against the gather
meter: equal for L3P, at most the meter for L4PP); with --require-stress (sp5m) <arm>-STRESS (journal overflow and
overflow_dead > 0, and a prefetch overflow path ran: lane refusal, commit overflow or pin overflow).
REGISTERED PREDICTIONS PRED_SPEC5: TBD by the lead (a None value gives REPORT); the step script STOPS sp5a / sp5b while
PRED_STAMP5 is None.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_spec_cell as SC          # noqa: E402

BC = SC.BC
ARMS5 = ("L3", "L3P", "L4P", "L4PP")
BASE_ARM = dict(L3="L3", L3P="L3", L4P="L4P", L4PP="L4P")
PREFETCH_ARMS = ("L3P", "L4PP")
PIN_ARMS = ("L4PP",)
EXACT_ARMS5 = ARMS5
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill every None and PRED_STAMP5 (`date -u`, L33) in a commit BEFORE sp5a /
# sp5b (the step script stops while PRED_STAMP5 is None).
#   <arm>_tok          True / False: the arm's committed tokens equal SH's, bitwise (L3P, L4PP)
#   L3P_vs_L3          (low, high): L3P's median steady committed tok/s / L3's (the prefetch against no prefetch)
#   L4PP_vs_L4P        (low, high): L4PP's / L4P's (the plan's L5 gate is "faster than L4": REPORT L5-VS-L4)
#   <arm>_served       (low, high): (READY_PF + IN_FLIGHT) / (READY_PF + IN_FLIGHT + NEW) over the steady rounds
#   <arm>_inflight_ms  (low, high): median per steady round of the IN FLIGHT wait (device waits + boundary stall), ms
# Inputs (MEASURED elsewhere): SCORING_pf1.md (64 / 192 MiB per step: +0.25 / +1.46 %, delivery 11.5 / 20.6 ms,
# backlog 0), SCORING_sp3b_fix.md (L3 at C 73 B 318: NEW 3,379 MB and the gather WAIT 136.5 ms per round),
# SCORING_sp3a.md (L4P at C 128 B 216: NEW 421 MB, WAIT 20.0 ms per round).
# Lead registration 2026-10-28T05:37:07Z (one set for sp5a and sp5b). A perfect prefetch hides at most the NEW gather
# wait (136.5 ms of a 967 ms L3 round at C 73; 20 ms at C 128), against packer and accounting costs: central +5 %.
PRED_STAMP5 = "2026-10-28T05:37:07Z"
PRED_SPEC5 = dict(stamp=PRED_STAMP5, L3P_tok=True, L4PP_tok=True, L3P_vs_L3=(0.9, 1.15), L4PP_vs_L4P=(0.9, 1.15),
                  L3P_served=(0.2, 0.9), L4PP_served=(0.2, 0.9), L3P_inflight_ms=(0.0, 150.0),
                  L4PP_inflight_ms=(0.0, 150.0))
SPLIT_KEYS = ("sel", "new", "ready_resident", "read_ready", "read_inflight", "refetch", "read_draft",
              "evict_overflow", "evict_free", "unread", "moved", "committed", "victim_named")


def log(msg):
    print("[r2_spec5 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=0, help="K drafts per round (0 = off: the cell refuses)")
    ap.add_argument("--spec-arms", default="L3,L3P,L4P,L4PP", help="a subset of %s, in run order" % ",".join(ARMS5))
    ap.add_argument("--spec-rounds", type=int, default=0, help="rounds per arm (0: ceil((steps - 1) / K))")
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--spec-prefetch", default="off", choices=("off", "on"),
                    help="THE FLAG of rung L5: the arms L3P / L4PP run only with it")
    ap.add_argument("--spec-pf-lanes", type=int, default=128, help="staging lanes per (layer, KV head) and epoch")
    ap.add_argument("--spec-pf-threads", type=int, default=8, help="the packer's OpenMP team")
    ap.add_argument("--spec-pf-cap-groups", type=int, default=256, help="the transport's chunk (groups per chunk)")
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1),
                    help="1: MUST gate <arm>-STRESS (the miniature must reach the overflow paths)")
    a = ap.parse_args(argv)
    a.spec_arms = [x for x in a.spec_arms.split(",") if x]
    if not a.spec_arms or any(x not in ARMS5 for x in a.spec_arms) or len(set(a.spec_arms)) != len(a.spec_arms):
        ap.error("--spec-arms: distinct names from %s" % (ARMS5,))
    if a.spec != 0 and not 3 <= a.spec <= 18:
        ap.error("--spec K in 3 .. 18 (L4P needs K >= 3)")
    pf = [x for x in a.spec_arms if x in PREFETCH_ARMS]
    if pf and a.spec_prefetch != "on":
        ap.error("the arms %s need --spec-prefetch on" % pf)
    if a.spec_prefetch == "on" and not pf:
        ap.error("--spec-prefetch on without a prefetch arm (%s)" % (PREFETCH_ARMS,))
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.spec_rounds < 0 or a.spec_journal_blocks < 1 or a.probe_every < 2:
        ap.error("--spec-rounds >= 0, --spec-journal-blocks >= 1, --probe-every >= 2")
    if a.spec_pf_lanes < 1 or not 1 <= a.spec_pf_threads <= 256 or a.spec_pf_cap_groups < 1:
        ap.error("--spec-pf-lanes >= 1, --spec-pf-threads in 1 .. 256, --spec-pf-cap-groups >= 1")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d (the prefetch needs the victim pool)" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def pick_cores(n, allowed):
    """The C++ worker's cores: the last n of the allowed set (None when the set has no more than n; r2_poolpf_cell)."""
    allowed = sorted(allowed)
    return allowed[-n:] if len(allowed) > n else None


def _q(xs, f):
    xs = [x for x in xs if x is not None]
    return round(f(xs), 4) if xs else None


def _p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def pf_summary(rep, steady, block_bytes: int, rd):
    """The prefetch record of one arm (no tensors) from SpecPrefetch.report() rep, the steady rounds and the runner's
    read() rd: the split per steady round (blocks, MB), the waits, useful prefetch, the pin overflows, the packer."""
    sp = rep["spec"]
    want = set(steady)
    rel = [r for r in sp["rounds"] if isinstance(r["round"], int) and "split" in r]
    st = [r for r in rel if r["round"] in want]
    col = {k: [r["split"][k] for r in st] for k in SPLIT_KEYS}
    cat = dict(READY_resident="ready_resident", READY_prefetched="read_ready", IN_FLIGHT="read_inflight", NEW="new")
    blocks = {c: _q(col[k], statistics.fmean) for c, k in cat.items()}
    mb = {c: (round(v * block_bytes / 1e6, 3) if v is not None else None) for c, v in blocks.items()}
    dev = [r.get("dev_wait_ms") for r in st]
    stall = [r.get("stall_ms") for r in st]
    both = [(d or 0.0) + (s or 0.0) for d, s in zip(dev, stall)]
    probe = [w for w in rd.get("probe_wait_ms", []) if w is not None]
    t = sp["totals"]
    c = rep["counters"]
    hits = sum(col["read_ready"]) + sum(col["read_inflight"])
    served = (hits / (hits + sum(col["new"]))) if (hits + sum(col["new"])) else None
    comm = t["read_ready"] + t["read_inflight"] + t["read_draft"] + t["evict_overflow"] + t["evict_free"] + \
        t["unread"] + t["moved"]
    useful = {k: t[k] for k in ("read_ready", "read_inflight", "read_draft", "evict_overflow", "evict_free",
                                "unread", "moved", "refetch")}
    useful.update(committed=sp["totals_cs"]["committed"], ended=comm,
                  read_by_verify_fraction=(round((t["read_ready"] + t["read_inflight"]) / comm, 4) if comm else None),
                  read_by_nothing_fraction=(round((t["unread"] + t["evict_overflow"] + t["evict_free"]) / comm, 4)
                                            if comm else None))
    dl = [x[0] for x in rep.get("delivery_ms", [])]
    late = sum(1 for x in rep.get("delivery_ms", []) if x[1])
    hms = {}
    for k in ("issue_s", "verify_s", "release_s", "account_s", "snap_wait_s", "submit_wait_s"):
        hms[k.replace("_s", "_ms_median")] = _q([r.get("host_ms", {}).get(k) for r in st], statistics.median)
    return dict(
        steady_rounds=len(st), released_rounds=len(rel),
        blocks_per_round=blocks, MB_per_round=mb, block_bytes=block_bytes,
        selected_per_round=_q(col["sel"], statistics.fmean),
        served_fraction=(round(served, 4) if served is not None else None),
        waits=dict(inflight_dev_ms_median=_q(dev, statistics.median), stall_ms_median=_q(stall, statistics.median),
                   inflight_total_ms_median=_q(both, statistics.median),
                   inflight_total_ms_mean=_q(both, statistics.fmean), new_gather_ms_median_probe=_q(
                       probe, statistics.median), ready_ms=0.0, probe_rounds=len(probe)),
        useful=useful, pin_overflow=dict(commit_overflow=c["commit_overflow"], evict_overflow=t["evict_overflow"],
                                         total=sp["pin_overflow"]),
        delivery_ms=dict(n=len(dl), median=(_q(dl, statistics.median)), p90=(round(_p90(dl), 4) if dl else None),
                         max=(round(max(dl), 4) if dl else None), late=late),
        backlog=dict(jobs_inflight_mean=_q([r["jobs_inflight"] for r in st], statistics.fmean),
                     jobs_inflight_max=(max([r["jobs_inflight"] for r in st]) if st else None),
                     jobs_ready_mean=_q([r["jobs_ready"] for r in st], statistics.fmean),
                     items_inflight_mean=_q([r["items_inflight"] for r in st], statistics.fmean)),
        issue=dict(sp["issue"], jobs=rep["stats"]["jobs"], items_issued=rep["stats"]["items_issued"],
                   bytes_issued=rep["stats"]["bytes_issued"], lane_refused=rep["stats"]["lane_refused"],
                   dup_items=rep["stats"]["dup_items"], requested=c["requested"], committed=c["committed"],
                   dup_dropped=c["dup_dropped"]),
        host_ms_per_round=hms, host_s_total=sp["host_s"], conservation=sp["conservation"],
        transport=rep.get("transport"), geometry=rep.get("geometry"))


def summarize(arm, res, inputs, fp_sh, sh, B: int, steps: int, dims):
    """r2_spec_cell.summarize of the base arm (L3 / L4P), then the prefetch record (prefetch arms)."""
    rec = SC.summarize(BASE_ARM[arm], res, inputs, fp_sh, sh, B, steps, dims)
    rec["arm"], rec["base_arm"] = arm, BASE_ARM[arm]
    if arm in PREFETCH_ARMS:
        steady = SC.steady_rounds(res["read"]["rounds"], res["probe_every"])
        rec["prefetch"] = (pf_summary(res["prefetch"], steady, dims["block_bytes"], res["read"])
                           if res.get("prefetch") is not None else None)
        if rec["prefetch"] is not None:                  # r2_spec_cell's kv_at_verify writes IN FLIGHT 0 (no prefetch)
            rec["kv_at_verify"]["in_flight_blocks"] = rec["prefetch"]["blocks_per_round"]["IN_FLIGHT"]
            rec["kv_at_verify"]["in_flight_note"] = "rung L5: the split of the verify-selected slots is in 'prefetch'"
    return rec


def arm_gates(arm, res, rec=None, require_stress: bool = False):
    """The MUST gates of one arm: r2_spec_cell.arm_gates of the base arm (ERR, COMMIT, TOK, FP; renamed to this arm),
    then the prefetch gates (module docstring). [(name, must, passed, detail)]."""
    base = BASE_ARM[arm]
    g = [(n.replace(base + "-", arm + "-", 1), m, ok, d) for n, m, ok, d in SC.arm_gates(base, res, rec)]
    if arm not in PREFETCH_ARMS:
        return g
    p = res.get("prefetch")
    names = ["PF-ACT", "PF-CONS", "PF-PIN", "PF-NEW"] + (["STRESS"] if require_stress else [])
    if p is None:
        return g + [("%s-%s" % (arm, n), True, False, "no prefetch report") for n in names]
    sp, t = p["spec"], p["spec"]["totals"]
    jobs, hits = p["stats"]["jobs"], t["read_ready"] + t["read_inflight"]
    g.append(("%s-PF-ACT" % arm, True, jobs > 0 and hits > 0, "jobs %d; prefetched verify hits %d (ready %d, in flight "
              "%d)" % (jobs, hits, t["read_ready"], t["read_inflight"])))
    cons = sp["conservation"]
    neg = [r["round"] for r in sp["rounds"] if "split" in r and r["split"]["ready_resident"] < 0]
    g.append(("%s-PF-CONS" % arm, True, bool(cons["conserved"]) and not neg,
              "committed %d == ended %d (read %d + %d + draft %d, evicted %d + %d, unread %d, moved %d); %s; rounds "
              "with READY_resident < 0: %s" % (cons["tags_committed"], cons["tags_ended"], t["read_ready"],
                                                t["read_inflight"], t["read_draft"], t["evict_overflow"],
                                                t["evict_free"], t["unread"], t["moved"],
                                                {k: v for k, v in cons.items() if k not in ("tags_committed",
                                                                                            "tags_ended")}, neg[:5])))
    g.append(("%s-PF-PIN" % arm, True, t["evict_free"] == 0,
              "prefetched blocks evicted while an unpinned unclaimed slot existed: %d (pin overflow evictions %d)"
              % (t["evict_free"], t["evict_overflow"])))
    meter = sum(res["read"]["round_blocks"])
    ok = (t["new"] == meter) if base == "L3" else (t["new"] <= meter)
    g.append(("%s-PF-NEW" % arm, True, ok, "the split's NEW %d %s the gather meter's host loads %d" % (
        t["new"], "==" if base == "L3" else "<=", meter)))
    if require_stress:
        j = res["read"]["journal"]
        ovf, dead = sum(j["overflow"]), sum(j["overflow_dead"])
        pfo = p["stats"]["lane_refused"] + p["counters"]["commit_overflow"] + t["evict_overflow"]
        g.append(("%s-STRESS" % arm, True, ovf > 0 and dead > 0 and pfo > 0,
                  "journal overflow %d, overflow_dead %d; prefetch overflow paths: lane refused %d, commit overflow "
                  "%d, pin overflow evictions %d" % (ovf, dead, p["stats"]["lane_refused"],
                                                      p["counters"]["commit_overflow"], t["evict_overflow"])))
    return g


def verdicts(sh, sp, B: int):
    """PRED_SPEC5 against the arms: [(name, verdict, detail)]; None registered gives REPORT; an arm that did not run
    gives NOT_RUN. Two plan gates are REPORT only: L5-VS-L4 (L4PP / L4P) and L5-VS-L0 (L4PP / SH)."""
    P = PRED_SPEC5
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))

    def med(r):
        return (r or {}).get("rates", {}).get("median_tok_s")
    for arm in PREFETCH_ARMS:
        r = (sp or {}).get(arm)
        if r is None:
            out.append(("%s-RUN" % arm, "NOT_RUN", "the arm did not run"))
            continue
        tc = r["tokens"]
        want = P["%s_tok" % arm]
        got = tc["n_checked"] > 0 and tc["n_unequal"] == 0
        out.append(("%s-TOK" % arm, "REPORT" if want is None else ("CONFIRMED" if got == bool(want) else "REFUTED"),
                    "%d of %d unequal (first %s); registered %s" % (tc["n_unequal"], tc["n_checked"], tc["first"],
                                                                     want)))
        b = (sp or {}).get(BASE_ARM[arm])
        band("%s-VS-%s" % (arm, BASE_ARM[arm]), (med(r) / med(b)) if (med(r) and med(b)) else None,
             P["L3P_vs_L3" if arm == "L3P" else "L4PP_vs_L4P"])
        pf = r.get("prefetch") or {}
        band("%s-SERVED" % arm, pf.get("served_fraction"), P["%s_served" % arm])
        band("%s-INFLIGHT-MS" % arm, (pf.get("waits") or {}).get("inflight_total_ms_median"), P["%s_inflight_ms" % arm])
    l4pp, l4p = (sp or {}).get("L4PP"), (sp or {}).get("L4P")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    out.append(("L5-VS-L4", "REPORT", "L4PP / L4P median steady tok/s: %s (the plan's gate: > 1)" % (
        round(med(l4pp) / med(l4p), 4) if (med(l4pp) and med(l4p)) else None)))
    out.append(("L5-VS-L0", "REPORT", "L4PP / SH median steady tok/s: %s (the plan's stop rule: L5 must beat L0)" % (
        round(med(l4pp) / shm, 4) if (med(l4pp) and shm) else None)))
    return out


# ------------------------------------------------------------------ the speculative arm (GPU)
def run_spec(model, ids, inputs, steps: int, clock, opt, ext=None, cores=None):
    """r2_spec_cell.run_spec with the prefetch (opt["prefetch"]): a SpecPrefetch on the per-request engine, passed as
    SpecOps(prefetch=...); its report goes into res["prefetch"]."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    cfg = model.config
    model.has_buffers = False
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
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()
    pf, pf_build_s = None, None
    if opt["prefetch"]:
        from nosi import spec_prefetch as SP
        tb = time.time()
        pf = SP.SpecPrefetch(rs, ext, lanes=opt["pf_lanes"], threads=opt["pf_threads"], cores=cores,
                             cap_groups=opt["pf_cap_groups"])
        pf_build_s = round(time.time() - tb, 3)
    a = SC.ARM_OPS[BASE_ARM[opt["arm"]]]
    sops = SR.SpecOps(level=a["level"], K=K, mode=a["mode"], gemm=a["gemm"], journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"], pin=bool(opt["pin"]), prefetch=pf)
    runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
    G = SC.rounds_for(steps, K, opt["rounds"])
    done = False
    try:
        runner.run(steps, rounds=G)
        rd = runner.read()
        rep = pf.report() if pf is not None else None
        res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
                   probe_every=opt["probe_every"], prefill_s=round(prefill_s, 1), first_token_agreement=first_agree,
                   meta=dict(rs.meta, spec=runner.meta, pf_build_s=pf_build_s), prefetch=rep)
        done = True
    finally:
        if done:
            runner.detach()
            if pf is not None:
                pf.close()
        elif pf is not None:                             # an exception inside a round: stop the worker thread only
            try:
                pf.tp.close()
            except Exception:
                traceback.print_exc()
    del runner, rs, cache, pf
    return res


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.spec == 0:
        print("REFUSED: --spec K is the flag of this cell (0 = off)", file=sys.stderr)
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
    ext, build_s, cores = None, None, None
    if any(x in PREFETCH_ARMS for x in a.spec_arms):
        import gilfree_pack as GP
        cores = pick_cores(a.spec_pf_threads, os.sched_getaffinity(0))
        try:
            tb = time.time()
            ext = GP.load(GP.build_dir_from_env())
            build_s = round(time.time() - tb, 1)
            gates.append(("PF-BUILD", True, int(ext.VERSION) == 3 and int(ext.DESC) >= 2 and int(ext.OPENMP) == 1,
                          "gilfree_pack v%s DESC %s OPENMP %s, built in %s s; worker cores %s" % (
                              ext.VERSION, ext.DESC, ext.OPENMP, build_s, cores)))
        except Exception as e:
            traceback.print_exc()
            gates.append(("PF-BUILD", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            ext = None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
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
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
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
            if arm in PREFETCH_ARMS and ext is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: the packer extension did not build"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, twin=a.spec_twin,
                       body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                       probe_every=a.probe_every, pin=arm in PIN_ARMS, prefetch=arm in PREFETCH_ARMS,
                       pf_lanes=a.spec_pf_lanes, pf_threads=a.spec_pf_threads, pf_cap_groups=a.spec_pf_cap_groups)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_spec(model, ids, inputs, a.steps, clock, opt, ext, cores)
                pa, pr = BC.gpu_peak()
                rec = summarize(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend(arm_gates(arm, res, rec, require_stress=bool(a.require_stress)))
                del res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, spec_arms=a.spec_arms, spec_rounds=a.spec_rounds,
                spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin, spec_body=a.spec_body,
                spec_graph=a.spec_graph, spec_redirect=a.spec_redirect, spec_prefetch=a.spec_prefetch,
                spec_pf_lanes=a.spec_pf_lanes, spec_pf_threads=a.spec_pf_threads,
                spec_pf_cap_groups=a.spec_pf_cap_groups, pf_cores=cores, pf_build_s=build_s,
                require_stress=a.require_stress, probe_every=a.probe_every, predictions=PRED_SPEC5, dims=dims,
                host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-14s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.spec_arms:
        r = sp.get(name)
        if r is None:
            continue
        j = r["journal"]
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s; vs SH %s; acceptance %s; phases %s; "
              "tokens equal %.6f; journal overflow %s dead %s; peak %s GB" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"], r["vs_sh"],
                  r["acceptance"]["rate"], r["phases_median"], r["tokens"]["equal_fraction"], sum(j["overflow"]),
                  sum(j["overflow_dead"]), r["gpu_peak_alloc_GB"]))
        pf = r.get("prefetch")
        if pf:
            print("REPORT %s KV at verify per steady round: blocks %s, MB %s, served %s; waits %s; useful %s; pin "
                  "overflow %s; delivery ms %s; backlog %s; host ms %s" % (
                      name, pf["blocks_per_round"], pf["MB_per_round"], pf["served_fraction"], pf["waits"],
                      pf["useful"], pf["pin_overflow"], pf["delivery_ms"], pf["backlog"], pf["host_ms_per_round"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_spec5: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
