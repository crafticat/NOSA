"""WS-B of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs; departure D2): the GPU cell of
the joint pass with ONE residency per layer for the UNION of the rows' selections (nosi/nosi/verify_rows.py mode
"union"; opt-in --joint-residency union). A NEW script: r2_spec_cell.py and r2_base_cell.py are not edited (their
functions are imported). Step script: benchmarks/Efficiency/r2_wsb_step.sh (cells wb1m, wb1a, wb1b, wb1ca, wb1cb).

ONE PROCESS, ONE MODEL, the arms in the order given, each with its own shipped prefill; SH runs FIRST:
  SH    the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline; its tokens and
        logits fingerprints (max, float64 sum) of every (request, step)
TIMING ARMS (no check-only work; accounting off; the same configuration as the L4X arm of sp3a / sp3b, so the extra
work is told apart from slower execution):
  L4X   spec_round level L4, EXACT joint pass (gemm rowgroup; the serial residency per row): the token proof
  L4XU  spec_round level L4, the UNION joint pass (gemm natural: one GEMM call over 4 B rows; rows 0 .. K - 1 verify
        rows, row K a resident draft row; joint_stats off). FAST: only the float order may differ.
CHECK ARMS (the coordinator's binding correction 2026-10-28: FAST is checked on FAST itself against SH; not timed):
  SHR   the SH REFERENCE on the per-request engine: the serial ragged step (gemm natural, the cell's twin / body /
        graph: the configuration whose logits were bitwise SH's in sp3a's L3 verify), TEACHER-FORCED on SH's tokens for
        --check-steps steps; it records every step's selection per layer (the step's own top-k ids), its logits
        fingerprint and its top-16 logits. MUST: its fingerprints and argmaxes equal SH's (so its selections are SH's).
  UC    the union joint pass (PRODUCTION: gemm natural), TEACHER-FORCED on SH's tokens (every row a verify row: its K / V
        stays committed state, as SH's), check on: per pass and layer every row's selection SET against SHR's at the
        same position (a changed selection is a MUST failure, with the symmetric-difference size), the PRESENCE recount
        (every selected block in the window / store / tail / next block: dropped == 0), the VISIBLE COUNT (union
        residency must not become union attention: each row sees exactly its selection size), the logits deltas per
        verified position against SHR's top-16 and SH's fingerprint, one PROFILED pass (torch.profiler: the kernels
        each call dispatches).
  UCR   UC with gemm rowgroup (the shipped GEMM shape: SH's GEMM bits), so a selection change there is the union
        attention's float order alone.
MEASURED (timing arms; CUDA events): the round phases (joint, truncate, draft, commit), the joint-pass ms, committed
tok/s against SH and against the strongest baseline 3,407.2 (SCORING_specp_fix.md section 4), and on the PROBE rounds the
joint pass's phases (verify_rows phase marks: pre, twin, select, resid, union, fetch, attn, spill, merge, check, end,
post, head; EXACT: resid / attn per row) and the COMPLETE time of each layer; the host invocation counts per pass
(verify_rows.VerifyState.calls: GEMM, attention, residency, gather calls, store fills); the joint pass's host loads per
round (the KVMeter: the residual misses).
MUST GATES: SH-RUN; per timing arm <arm>-RUN, -ERR, -COMMIT, L4X-TOK / -FP (EXACT), L4XU-RESID (one residency call per
layer and joint pass); per check arm SHR-RUN, SHR-FP, SHR-TOK, <arm>-RUN, -ERR, -SEL (no selection change), -PRESENT
(dropped 0 and unserved 0), -VIS (no visible-count error), -RESID, -IDX (sampled entries of the window call and of the
first store call bitwise a plain one-entry call), with --require-stress 1 <arm>-STRESS (the spill ran).
REGISTERED PREDICTIONS PRED_WSB: TBD by the lead (a None value gives REPORT); the step script STOPS wb1a / wb1b while
PRED_STAMP_WSB is None.
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
ARMS = ("L4X", "L4XU", "SHR", "UC", "UCR")
TIMING_ARMS = ("L4X", "L4XU")
CHECK_ARMS = ("SHR", "UC", "UCR")
UNION_ARMS = ("L4XU", "UC", "UCR")
CHECK_GEMM = dict(UC="natural", UCR="rowgroup")
STRONGEST = 3407.2                 # the whole-step graph, C 128 B 216, nova07 (SCORING_specp_fix.md section 4)
TOPV = 16                          # SHR keeps its top-16 logits per (request, step): the logits deltas of UC
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill every None and PRED_STAMP_WSB (`date -u`, L33) in a commit BEFORE wb1a /
# wb1b (the step script stops while PRED_STAMP_WSB is None). Keys per capacity (C128: pool 65; C73: pool 10):
#   joint_sh_steps   (low, high): L4XU's median steady joint-pass ms / SH's steady step ms
#   ratio_vs_sh      (low, high): L4XU's median steady committed tok/s / SH's steady tok/s
#   ratio_vs_l4x     (low, high): L4XU's / L4X's median steady committed tok/s
# Plan (MODEL, PLAN_L4P_faithful.md WS-B): joint pass of 4 rows about 2.85 SH steps at C 128 B 216 (184 ms), about 3.2
# at C 73 B 318 (398 ms); FALSIFIER > 3.5 / > 4.0 steps.
# The lead's REGISTRATION (2026-10-28T10:57:16Z, `date -u`, before any WS-B cell ran). MEASURED inputs
# (SCORING_specp_fix.md section 7, same configs, same-job SH): L4X joint 330.735 ms = 5.129 SH steps (C 128) and
# 539.924 ms = 4.354 (C 73); L4X round 749.440 / 1,122.537 ms; L4X DIAG vs SH 0.2115 / 0.8214 = 0.2575 and
# 0.2864 / 0.8668 = 0.3304. MODEL: the round keeps L4X's draft / truncate / commit and swaps the joint pass; with the
# joint bands below the round ratio is 1.16-1.32 (C 128) and 1.04-1.26 (C 73). The bands are wider upward because one
# residency per layer journals fewer window changes (truncate + commit can shrink, unknown) and downward because the
# drafts after a union pass find fewer window hits (NOTES_wsb.md risk 7, unknown).
PRED_STAMP_WSB = "2026-10-28T10:57:16Z"
PRED_WSB = dict(stamp=PRED_STAMP_WSB, C128=dict(joint_sh_steps=(2.3, 3.5), ratio_vs_sh=(0.27, 0.41),
                                                ratio_vs_l4x=(1.05, 1.60)),
                C73=dict(joint_sh_steps=(2.4, 4.0), ratio_vs_sh=(0.31, 0.50), ratio_vs_l4x=(0.95, 1.50)))
# The phase boundaries (verify_rows.py; each label runs from its mark to the next mark of the same pass): docs/NOTES_wsb.md
# section 4 states them with file:line and the dependencies between them.
DENSE = ("pre", "post", "head")
PHASE_GROUPS = dict(dense=DENSE, twin=("twin",), selection=("select",), residency=("resid", "union", "rows"),
                    fetch=("fetch",), attention=("attn", "merge"), spill=("spill",), check=("check",), end=("end",),
                    probe=("probe",))


def log(msg):
    print("[r2_wsb %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K drafts per round (L4: K verify rows + 1 draft row)")
    ap.add_argument("--arms", default="L4X,L4XU", help="a subset of %s, in run order (SH runs first)" % ",".join(ARMS))
    ap.add_argument("--joint-residency", default="serial", choices=("serial", "union"),
                    help="THE FLAG of WS-B: the arms %s run only with 'union'" % (UNION_ARMS,))
    ap.add_argument("--joint-extra", type=int, default=48, help="X: store slots per request (slot 0 the next block)")
    ap.add_argument("--joint-spill", type=int, default=0,
                    help="store fills per layer (0: the bound ceil((R - 1) topk / (X - 1)): nothing can be dropped)")
    ap.add_argument("--check-steps", type=int, default=128, help="SHR / UC / UCR: teacher-forced positions")
    ap.add_argument("--spec-rounds", type=int, default=0, help="rounds per timing arm (0: ceil((steps - 1) / K))")
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--profile", default="on", choices=("on", "off"), help="check arms: one profiled pass / step")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1),
                    help="1: MUST gate <arm>-STRESS for UC / UCR (the spill ran)")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    un = [x for x in a.arms if x in UNION_ARMS]
    if un and a.joint_residency != "union":
        ap.error("the arms %s need --joint-residency union" % un)
    if a.joint_residency == "union" and not un:
        ap.error("--joint-residency union without a union arm (%s)" % (UNION_ARMS,))
    ck = [x for x in a.arms if x in ("UC", "UCR")]
    if ck and ("SHR" not in a.arms or a.arms.index("SHR") > min(a.arms.index(x) for x in ck)):
        ap.error("UC / UCR compare against SHR: SHR must run before them")
    if not 3 <= a.spec <= 18:
        ap.error("--spec K in 3 .. 18")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    R = a.spec + 1
    if any(x in CHECK_ARMS for x in a.arms) and (a.check_steps < R or a.check_steps > a.steps or a.check_steps % R):
        ap.error("--check-steps: a multiple of K + 1 = %d, at most --steps (the check arms)" % R)
    if a.joint_extra < 2 or a.joint_spill < 0 or a.probe_every < 2 or a.spec_rounds < 0 or a.spec_journal_blocks < 1:
        ap.error("--joint-extra >= 2, --joint-spill >= 0, --probe-every >= 2, --spec-rounds >= 0, journal >= 1")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def cap_key(pool: int) -> str:
    return "C%d" % (BC.C_BASE + int(pool))


def group_phases(ph: dict) -> dict:
    """{label: ms} of one pass -> {group: ms} (PHASE_GROUPS) plus "total" (every label; the complete pass)."""
    out = {g: round(sum(ph.get(x, 0.0) for x in labs), 4) for g, labs in PHASE_GROUPS.items()}
    out["total"] = round(sum(ph.values()), 4)
    return out


def phase_summary(phases, layers):
    """The probe passes' phases: phases [{label: ms}], layers [[ms per layer]] -> medians per label and per group, the
    median complete layer time and the median pass total."""
    if not phases:
        return dict(n_probes=0)
    labels = sorted({k for p in phases for k in p})
    med = {k: round(statistics.median([p.get(k, 0.0) for p in phases]), 4) for k in labels}
    grp = [group_phases(p) for p in phases]
    gmed = {g: round(statistics.median([x[g] for x in grp]), 4) for g in grp[0]}
    lay = [x for ls in layers for x in ls]
    return dict(n_probes=len(phases), labels_median=med, groups_median=gmed,
                layer_ms_median=(round(statistics.median(lay), 4) if lay else None),
                layer_ms_max=(round(max(lay), 4) if lay else None),
                layers_per_pass=(len(layers[0]) if layers else None))


def per_pass(calls: dict, passes: int) -> dict:
    """Host invocation counts per pass."""
    return {k: (round(v / passes, 3) if passes else None) for k, v in calls.items()}


def quantiles(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return dict(n=0)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
    return dict(n=len(xs), median=round(statistics.median(xs), 6), p90=round(q(0.9), 6), p99=round(q(0.99), 6),
                max=round(xs[-1], 6))


def check_gates(arm, res, require_stress: bool = False, L: int = 32):
    """The MUST gates of a check arm (UC / UCR) from run_check_union's result: [(name, must, passed, detail)]."""
    st = res["stats"]
    g = []
    body_err, overflow = res["errors"]
    g.append(("%s-ERR" % arm, True, body_err == 0 and not any(overflow),
              "body refusal bits %s; twin overflow flags %s" % (body_err, sorted(set(overflow)))))
    sc = res["sel"]
    g.append(("%s-SEL" % arm, True, sc["compared"] > 0 and sc["changed"] == 0,
              "row selections against SHR's at the same position: %d changed of %d (head, request, row, layer) sets; "
              "by layer %s; symmetric-difference sizes %s; first %s" % (
                  sc["changed"], sc["compared"], sc["by_layer_nonzero"], sc["symdiff_hist"], sc["first"])))
    g.append(("%s-PRESENT" % arm, True, st["dropped"] == 0 and st["unserved"] == 0 and st["union_samples"] > 0,
              "selected blocks of verify rows not present: %d (recount); unserved list entries %d; spill %d of %d "
              "extras" % (st["dropped"], st["unserved"], st["spill"], st["extras"])))
    g.append(("%s-VIS" % arm, True, st["vis_bad"] == 0 and st["visible_blocks"] > 0,
              "rows whose visible block count is not their selection size: %d (per row %s); visible %s, selected %s "
              "per row index" % (st["vis_bad"], st["row_vis_bad"], st["row_visible"], st["row_selected"])))
    passes = st["passes"]
    g.append(("%s-RESID" % arm, True, passes > 0 and st["calls"]["resid"] == L * passes,
              "residency calls %d for %d passes of %d layers" % (st["calls"]["resid"], passes, L)))
    fa = res["fa_check"] or {}
    ok = all(k in fa and fa[k]["entries"] and all(fa[k]["equal"]) for k in ("A", "B"))
    g.append(("%s-IDX" % arm, True, ok, "window call %s; first store call %s" % (fa.get("A"), fa.get("B"))))
    if require_stress:
        g.append(("%s-STRESS" % arm, True, st["spill"] > 0 and st["spill_fills_used"] > 0,
                  "spill %d blocks; spill fills (c >= 1) that served an entry %d" % (st["spill"],
                                                                                  st["spill_fills_used"])))
    return g


def timing_union_gates(arm, rd, L: int = 32):
    """L4XU: one residency call per layer and joint pass (host count of verify_rows), [(name, must, passed, detail)]."""
    v = rd["verify"] or {}
    calls, passes = (v.get("calls") or {}), v.get("passes", 0)
    return [("%s-RESID" % arm, True, passes > 0 and calls.get("resid") == L * passes,
             "residency calls %s for %d joint passes of %d layers" % (calls.get("resid"), passes, L))]


def verdicts(sh, sp, pool: int):
    """PRED_WSB against the timing arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN when an arm is
    missing."""
    P = PRED_WSB.get(cap_key(pool)) or {}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))
    u, x = (sp or {}).get("L4XU"), (sp or {}).get("L4X")
    if u is None:
        out.append(("L4XU-RUN", "NOT_RUN", "the arm did not run"))
        return out
    shm = sh["rates"]["steady_step_ms"] if sh else None
    jm = (u.get("phases_median") or {}).get("joint")
    band("JOINT-SH-STEPS", (jm / shm) if (jm and shm) else None, P.get("joint_sh_steps"))
    band("RATIO-VS-SH", u["vs_sh"]["median_tok_s"], P.get("ratio_vs_sh"))
    um, xm = u["rates"]["median_tok_s"], (x or {}).get("rates", {}).get("median_tok_s")
    band("RATIO-VS-L4X", (um / xm) if (um and xm) else None, P.get("ratio_vs_l4x"))
    out.append(("VS-STRONGEST", "REPORT", "L4XU median committed tok/s %s / %.1f = %s" % (
        um, STRONGEST, round(um / STRONGEST, 4) if um else None)))
    return out


# ------------------------------------------------------------------ GPU helpers
class PhaseVR:
    """verify_rows with CUDA-event phase marks on the joint pass of the runner's PROBE rounds (already excluded from the
    steady statistics). It replaces runner.VR; every other attribute is the module's."""

    def __init__(self, VR, runner, clock):
        self._VR, self._runner, self._clock = VR, runner, clock
        self.marks = []

    def __getattr__(self, name):
        return getattr(self._VR, name)

    def joint_pass(self, model, vs, ids, pos):
        r = self._runner
        pe = r.ops.probe_every
        probe = pe > 0 and r.round > 0 and r.round % pe == 0
        if probe:
            vs.marks, vs.clock = [], self._clock
        try:
            return self._VR.joint_pass(model, vs, ids, pos)
        finally:
            if probe:
                self.marks.append(vs.marks)
                vs.marks = None


def _release_warm(model):
    import torch
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


def _prefill(model, ids, inputs, steps: int, clock):
    """Its own shipped prefill (SH's cache and max_gen_len): (cache, prefill s, first-token agreement)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    _release_warm(model)
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    first = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    return cache, round(time.time() - t0, 1), first


def _profile(fn, top: int = 40):
    """CHECK-ONLY, REPORT: one call of fn under torch.profiler. Returns (fn's result, dict(kernels = [(kernel name,
    calls, device ms)] the top `top` by device time, error)). fn runs exactly once: without the profiler when it cannot
    start; an exception of fn itself propagates (the arm fails, as without profiling)."""
    import torch
    try:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()
    except Exception as ex:                                      # the profiler is evidence, not a gate
        return fn(), dict(kernels=None, n_kernel_names=0, error="start: %s: %s" % (type(ex).__name__, str(ex)[:200]))
    try:
        out = fn()
        torch.cuda.synchronize()
    finally:
        prof.__exit__(None, None, None)
    try:
        rows = []
        for e in prof.key_averages():
            dev = getattr(e, "device_time_total", None)
            if dev is None:
                dev = getattr(e, "cuda_time_total", 0.0)
            if dev and dev > 0 and "CUDA" in str(getattr(e, "device_type", "")):
                rows.append((e.key, int(e.count), round(dev / 1e3, 4)))
        rows.sort(key=lambda x: -x[2])
        return out, dict(kernels=rows[:top], n_kernel_names=len(rows), error=None)
    except Exception as ex:
        return out, dict(kernels=None, n_kernel_names=0, error="read: %s: %s" % (type(ex).__name__, str(ex)[:200]))


# ------------------------------------------------------------------ the timing arms (GPU)
def run_spec(model, ids, inputs, steps: int, clock, opt):
    """ONE timing arm (L4X / L4XU): r2_spec_cell.run_spec's flow with the arm's SpecOps, the phase proxy on the probe
    rounds and the host invocation counts."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    cache, prefill_s, first = _prefill(model, ids, inputs, steps, clock)
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()
    if opt["arm"] == "L4X":
        sops = SR.SpecOps(level="L4", K=K, mode="exact", gemm="rowgroup", journal=opt["journal"],
                          redirect=opt["redirect"], probe_every=opt["probe_every"])
    else:                                       # L4XU: production union, no accounting, no check
        sops = SR.SpecOps(level="L4", K=K, mode="union", gemm="natural", extra=opt["extra"], journal=opt["journal"],
                          redirect=opt["redirect"], probe_every=opt["probe_every"], joint_stats=False)
    runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
    proxy = PhaseVR(runner.VR, runner, clock)
    runner.VR = proxy
    G = SC.rounds_for(steps, K, opt["rounds"])
    runner.run(steps, rounds=G)
    rd = runner.read()
    VR = proxy._VR
    phases = [VR.phase_ms(m, clock.elapsed) for m in proxy.marks]
    layers = [VR.layer_ms(m, clock.elapsed) for m in proxy.marks]
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=prefill_s, first_token_agreement=first,
               meta=dict(rs.meta, spec=runner.meta), joint_phases=phase_summary(phases, layers),
               calls_per_pass=per_pass((rd["verify"] or {}).get("calls", {}), (rd["verify"] or {}).get("passes", 0)),
               roles=dict(verify_rows_per_request=K, draft_rows_per_request=1, joint_rows=K + 1))
    runner.detach()
    del runner, rs, cache
    return res


# ------------------------------------------------------------------ the check arms (GPU)
def run_shr(model, ids, inputs, steps: int, n: int, clock, opt, fp_sh):
    """SHR (module docstring): the serial ragged step teacher-forced on SH's tokens for n steps. Returns its record:
    sel (n, L, H, B, T) int16 on the host, top-16 logits (values, ids), fingerprints, the gates' facts."""
    import torch
    from nosi import ragged_engine as RE
    cache, prefill_s, first = _prefill(model, ids, inputs, steps, clock)
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps, ops)
    gc.collect()
    torch.cuda.empty_cache()
    L, B = rs.L, rs.B
    eng0 = rs.cache.layers[0].cache_engine
    H, T = int(eng0.head_num), int(eng0.topk)
    lay = {lay_.cache_engine._k_gpu.data_ptr(): l for l, lay_ in enumerate(rs.cache.layers)}
    buf = torch.empty((L, H, B, T), dtype=torch.int16, device="cuda")
    sel = torch.empty((n, L, H, B, T), dtype=torch.int16)
    body = rs.body
    orig = body.decode_fixed

    def hook(eng, key_states, value_states, kv_bias, topk_idx, *args):    # CHECK-ONLY: the step's own top-k ids
        buf[lay[eng._k_gpu.data_ptr()]].copy_(topk_idx)
        return orig(eng, key_states, value_states, kv_bias, topk_idx, *args)
    body.decode_fixed = hook
    topv = torch.empty((n, B, TOPV), dtype=torch.float32)
    topi = torch.empty((n, B, TOPV), dtype=torch.int64)
    fmax = torch.empty((B, n), dtype=torch.float32, device="cuda")
    fsum = torch.empty((B, n), dtype=torch.float64, device="cuda")
    pred = torch.empty((B, n), dtype=torch.int64, device="cuda")
    prof = None
    try:
        for t in range(n):
            ids1 = inputs[:, t:t + 1].contiguous()
            pos1 = torch.full((B, 1), BC.S0 + t, dtype=torch.int32, device="cuda")

            def step():
                return RE.decode_step(model, rs, ids1, pos1)[:, -1]
            if opt["profile"] and t == n // 2:
                lg, prof = _profile(step)                            # CHECK-ONLY: the kernels one step dispatches
            else:
                lg = step()
            sel[t].copy_(buf.cpu())
            v, i = lg.topk(TOPV, dim=-1)
            topv[t].copy_(v.cpu())
            topi[t].copy_(i.cpu())
            fmax[:, t].copy_(lg.amax(dim=-1))
            fsum[:, t].copy_(lg.sum(dim=-1, dtype=torch.float64))
            pred[:, t].copy_(lg.argmax(dim=-1))
    finally:
        del body.decode_fixed
    fpc = BC.fingerprint_compare((fp_sh[0][:, :n], fp_sh[1][:, :n]), (fmax, fsum))
    tok_bad = int((pred.cpu() != inputs[:, 1:n + 1].cpu()).sum())
    body_err, overflow = RE.errors(rs)
    res = dict(sel=sel, topv=topv, topi=topi, fmax=fmax.cpu(), fp=fpc, tok_bad=tok_bad, n=n, prefill_s=prefill_s,
               first_token_agreement=first, errors=(body_err, overflow), profile=prof, L=L, H=H, T=T)
    del rs, cache
    return res


def run_check_union(model, ids, inputs, steps: int, clock, opt, shr):
    """UC / UCR (module docstring): union joint passes teacher-forced on SH's tokens, check on, against SHR."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import verify_rows as VR
    cache, prefill_s, first = _prefill(model, ids, inputs, steps, clock)
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=False)
    rs = RE.attach(model, cache, BC.S0, steps, ops)
    gc.collect()
    torch.cuda.empty_cache()
    K = int(opt["K"])
    R = K + 1
    B, L = rs.B, rs.L
    vs = VR.attach(model, rs, VR.VerifyOps(K=K, mode="union", gemm=CHECK_GEMM[opt["arm"]], extra=opt["extra"],
                                           spill=(opt["spill"] or None), check=True, stats=True))
    vs.clock = clock
    vs.fa_check = {}
    n = shr["n"]
    G = n // R
    every = int(opt["probe_every"])
    offs = torch.arange(R, dtype=torch.int64, device="cuda")
    changed_lr = torch.zeros((L, R), dtype=torch.int64, device="cuda")
    hist = torch.zeros(6, dtype=torch.int64, device="cuda")           # symmetric-difference sizes 2, 4, .., 10, 12+
    compared, first_bad = 0, []
    d_top, d_max, agree = [], [], []
    marks, prof = [], None
    is_probe = lambda g: every > 0 and g > 0 and g % every == 0         # noqa: E731
    g_prof = next((g for g in range(G // 2, G) if not is_probe(g)), None)   # the first non-probe pass from G // 2
    for g in range(G):
        probe = is_probe(g)
        vs.marks = [] if probe else None
        ids_g = inputs[:, g * R:(g + 1) * R].contiguous()
        pos_g = (BC.S0 + g * R + offs).to(torch.int32).view(1, R).expand(B, R).contiguous()
        if opt["profile"] and g == g_prof:                           # CHECK-ONLY: the kernels one pass dispatches
            lg, prof = _profile(lambda: VR.joint_pass(model, vs, ids_g, pos_g))
        else:
            lg = VR.joint_pass(model, vs, ids_g, pos_g)
        if probe:
            marks.append(vs.marks)
            vs.marks = None
        # CHECK-ONLY: the selections against SHR's (layer by layer), the logits against SHR's top-16 and SH's tokens
        ref = shr["sel"][g * R:(g + 1) * R].to("cuda").to(torch.int64)      # (R, L, H, B, T)
        for l in range(L):
            cur = vs.chk_sel[l]                                              # (H, B, R, T)
            rf = ref[:, l].permute(1, 2, 0, 3)                               # (H, B, R, T)
            eq = (cur.sort(-1).values == rf.sort(-1).values).all(-1)        # (H, B, R)
            nb = ~eq
            changed_lr[l] += nb.sum((0, 1))
            only = (~(cur.unsqueeze(-1) == rf.unsqueeze(-2)).any(-1)).sum(-1)    # (H, B, R): ids of cur not in ref
            sym = 2 * only
            hist += torch.bincount((sym // 2 - 1).clamp(min=0, max=5)[nb], minlength=6)[:6]
            if len(first_bad) < 5 and bool(nb.any()):
                for h, b, r in nb.nonzero().tolist()[:5 - len(first_bad)]:
                    first_bad.append((g, l, r, h, b, int(sym[h, b, r])))
        compared += L * vs.H * B * R
        tv = shr["topv"][g * R:(g + 1) * R].to("cuda")                     # (R, B, 16)
        ti = shr["topi"][g * R:(g + 1) * R].to("cuda")
        lgr = lg.permute(1, 0, 2)                                            # (R, B, V)
        d_top.append((lgr.gather(-1, ti) - tv).abs().amax(-1).cpu())        # (R, B)
        d_max.append((lgr.amax(-1) - shr["fmax"][:, g * R:(g + 1) * R].t().to("cuda")).abs().cpu())
        agree.append((lg.argmax(-1) == inputs[:, g * R + 1:(g + 1) * R + 1]).cpu())   # (B, R)
    clock.sync()
    phases = [VR.phase_ms(m, clock.elapsed) for m in marks]
    layers = [VR.layer_ms(m, clock.elapsed) for m in marks]
    stats = VR.read_stats(vs)
    dt = torch.stack(d_top)                                                  # (G, R, B)
    dm = torch.stack(d_max)
    ag = torch.stack(agree)                                                  # (G, B, R)
    by_row = {r: dict(top16_abs=quantiles(dt[:, r].reshape(-1).tolist()), max_abs=quantiles(dm[:, r].reshape(-1).tolist()),
                      argmax_agree=round(float(ag[:, :, r].float().mean()), 6)) for r in range(R)}
    cl = changed_lr.cpu()
    sel_rec = dict(compared=compared, changed=int(cl.sum()), by_layer_nonzero={l: cl[l].tolist() for l in range(L)
                                                                                if int(cl[l].sum())},
                   by_row=cl.sum(0).tolist(), symdiff_hist=dict(zip(("2", "4", "6", "8", "10", "12+"),
                                                                     hist.cpu().tolist())), first=first_bad)
    res = dict(stats=stats, sel=sel_rec, logits=dict(by_row=by_row, top16_abs=quantiles(dt.reshape(-1).tolist()),
                                                    argmax_agree=round(float(ag.float().mean()), 6)),
               errors=RE.errors(rs), fa_check=vs.fa_check, passes=G, R=R, joint_phases=phase_summary(phases, layers),
               calls_per_pass=per_pass(stats["calls"], stats["passes"]), profile=prof, prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, verify=vs.meta),
               ready=dict(win_sel=stats["win_sel"], win_host=stats["win_host"], extras_pool=stats["extras_pool"],
                          extras_host=stats["extras_host"]),
               roles=dict(rows_full=stats["rows_full"], rows_res=stats["rows_res"], passes=G))
    del vs, rs, cache
    return res


def ready_record(ready: dict, passes: int, block_bytes: int) -> dict:
    """The READY intersection (selected blocks already on the GPU: window hits, pool hits brought by the residency or
    copied into the store) and the RESIDUAL misses (host loads: the window's and the store's), blocks and MB per
    pass."""
    rdy = ready["win_sel"] - ready["win_host"] + ready["extras_pool"]
    miss = ready["win_host"] + ready["extras_host"]
    f = (lambda x: round(x / passes, 1)) if passes else (lambda x: None)
    return dict(ready_blocks_per_pass=f(rdy), residual_blocks_per_pass=f(miss),
                ready_MB_per_pass=(round(rdy * block_bytes / 1e6 / passes, 3) if passes else None),
                residual_MB_per_pass=(round(miss * block_bytes / 1e6 / passes, 3) if passes else None),
                block_bytes=block_bytes, raw=ready)


# ------------------------------------------------------------------ main
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
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, chk, inputs, shr = [], None, {}, {}, None, None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    L = len(model.layers)
    dims = dict(L=L, H=H, topk=64, block_bytes=64 * D * 2 * 2 + 64 * 2)
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
        for arm in a.arms:
            if inputs is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (every arm uses SH's tokens)"))
                continue
            if arm in ("UC", "UCR") and shr is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: no SHR record"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, twin=a.spec_twin,
                       body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                       probe_every=a.probe_every, extra=a.joint_extra, spill=(a.joint_spill if arm in ("UC", "UCR")
                                                                              else 0),
                       profile=a.profile == "on")
            log("arm %s (%s)" % (arm, opt))
            try:
                if arm in TIMING_ARMS:
                    res = run_spec(model, ids, inputs, a.steps, clock, opt)
                    pa, pr = BC.gpu_peak()
                    base = "L4X" if arm == "L4X" else "L4F"      # L4F: r2_spec_cell's FAST level-L4 arm (no TOK / FP)
                    rec = SC.summarize(base, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               joint_phases=res["joint_phases"], calls_per_pass=res["calls_per_pass"],
                               roles=res["roles"])
                    jm = rec["phases_median"].get("joint")
                    shm = sh["rates"]["steady_step_ms"] if sh else None
                    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                    rec["vs_strongest"] = (round(rec["rates"]["median_tok_s"] / STRONGEST, 4)
                                           if rec["rates"]["median_tok_s"] else None)
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend((n.replace(base + "-", arm + "-", 1), m, ok, d)
                                 for n, m, ok, d in SC.arm_gates(base, res, rec))
                    if arm == "L4XU":
                        gates.extend(timing_union_gates(arm, res["read"], L))
                elif arm == "SHR":
                    n = a.check_steps
                    shr = run_shr(model, ids, inputs, a.steps, n, clock, opt, fp_sh)
                    pa, pr = BC.gpu_peak()
                    fpc = shr["fp"]
                    chk[arm] = dict(n=n, fingerprints=fpc, tok_bad=shr["tok_bad"], errors=shr["errors"],
                                    profile=shr["profile"], prefill_s=shr["prefill_s"], gpu_peak_alloc_GB=pa,
                                    first_token_agreement=shr["first_token_agreement"])
                    gates.append(("SHR-RUN", True, True, "%d teacher-forced steps" % n))
                    gates.append(("SHR-FP", True, fpc["n_unequal"] == 0 if "n_unequal" in fpc else False,
                                  "SHR logits fingerprints against SH's: %s" % (fpc,)))
                    gates.append(("SHR-TOK", True, shr["tok_bad"] == 0,
                                  "SHR argmax against SH's tokens: %d of %d unequal" % (shr["tok_bad"], n * a.batch)))
                else:
                    res = run_check_union(model, ids, inputs, a.steps, clock, opt, shr)
                    pa, pr = BC.gpu_peak()
                    res.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               ready_record=ready_record(res["ready"], res["passes"], dims["block_bytes"]))
                    chk[arm] = res
                    gates.append(("%s-RUN" % arm, True, True, "%d teacher-forced passes of R %d" % (res["passes"],
                                                                                              res["R"])))
                    gates.extend(check_gates(arm, res, bool(a.require_stress), L))
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.pool)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, joint_residency=a.joint_residency,
                joint_extra=a.joint_extra, joint_spill=a.joint_spill, check_steps=a.check_steps,
                spec_rounds=a.spec_rounds, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, profile=a.profile, require_stress=a.require_stress, predictions=PRED_WSB,
                phase_groups=PHASE_GROUPS, dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"),
                strongest=STRONGEST)
    summary = dict(meta=meta, SH=sh, SPEC=sp, CHECK=chk,
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
    for name in a.arms:
        r = sp.get(name)
        if r is not None:
            print("REPORT %s K %d: %d rounds; committed tok/s median %s (vs SH %s, vs strongest %s); joint pass %s SH "
                  "steps; phases %s; joint-pass phases (probe) %s; calls per pass %s; KV at verify %s; tokens equal %.6f; "
                  "peak %s GB" % (name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["vs_sh"], r["vs_strongest"],
                                  r["joint_sh_steps"], r["phases_median"], r["joint_phases"], r["calls_per_pass"],
                                  r["kv_at_verify"], r["tokens"]["equal_fraction"], r["gpu_peak_alloc_GB"]))
        c = chk.get(name)
        if c is not None and name != "SHR":
            print("REPORT %s: selections %s; logits %s; stats %s; ready / residual %s; phases %s; calls per pass %s; "
                  "profile %s" % (name, c["sel"], c["logits"], c["stats"], c["ready_record"], c["joint_phases"],
                                  c["calls_per_pass"], c["profile"]))
        elif c is not None:
            print("REPORT SHR: fingerprints %s; argmax unequal %d; profile %s" % (c["fingerprints"], c["tok_bad"],
                                                                                  c["profile"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_wsb: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
