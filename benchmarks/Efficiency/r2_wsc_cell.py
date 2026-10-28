"""WS-C of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs; question Q2, after the fia6
A/B, SCORING_fia6.md): the union joint pass of WS-B with ONE call of the shared-KV fork per layer (verify_rows mode
"union", VerifyOps(joint_attn="fork"); opt-in --joint-attn fork, default "store" = WS-B unchanged). A NEW script:
r2_wsb_cell.py, r2_spec_cell.py and r2_base_cell.py are not edited (their functions are imported). Step script:
benchmarks/Efficiency/r2_wsc_step.sh (cells wc1m, wc1a, wc1b, wc1ca, wc1cb).

ONE PROCESS, ONE MODEL, the arms in the order given, each with its own shipped prefill; SH runs FIRST:
  SH    the shipped decode (r2_base_cell.run_shipped): the same-job baseline; tokens and logits fingerprints
TIMING ARMS (no check-only work, accounting off; the configuration of the L4X arm of sp3a / sp3b):
  L4X   L4, EXACT joint pass (the token proof)                                  (r2_wsb_cell.run_spec)
  L4XU  L4, the WS-B union joint pass: window call + store calls + merge         (r2_wsb_cell.run_spec)
  L4XF  L4, the union joint pass with ONE fork call per layer (FAST)             (run_spec_fork)
        Its K / V arena (benchmarks/Efficiency/r2_kv_arena.py) is made before the prefill and the engines are re-homed
        into it during the prefill (PrefillHook); its store holds X = 1 + (K - 1) x 64 slots (the bound, C = 1).
CHECK ARMS (FAST checked on FAST itself against SH; not timed):
  SHR   the SH reference, teacher-forced (r2_wsb_cell.run_shr)
  UC / UCR   WS-B's union check arms (r2_wsb_cell.run_check_union), for a same-job comparison
  UCF   the fork joint pass, teacher-forced on SH's tokens (every row a verify row, X = 1 + (R - 1) x 64), check on:
        SEL against SHR (MUST, unchanged), PRESENT, VIS and VB from the tables the KERNEL wrote, RESID, IDX (the "F"
        probe: sampled requests alone = their rows of the full launch, bitwise), RA (a row alone, bitwise), ATTN1 /
        FILL1 (one fork call and one store fill per layer and pass); one profiled pass.
  UCFR  UCF with gemm rowgroup (the shipped GEMM bits): a selection change there is the attention's float order.
MUST GATES: SH-RUN; per timing arm -RUN, -ERR, -COMMIT, L4X-TOK / -FP; L4XU-RESID; L4XF-RESID, -ATTN1, -FILL1;
SHR-RUN, -FP, -TOK; UC / UCR as WS-B; UCF / UCFR -RUN, -ERR, -SEL, -PRESENT, -VIS, -VB, -RESID, -IDX, -RA, -ATTN1,
-FILL1; with --require-stress 1 also -STRESS (past-edge rows, F4 rows, pool blocks read in place, a store occupancy
above WS-B's miniature fill of 3). C0_FORK (MUST) the fork module imports and is the byte copy of fia6's (blob).
REGISTERED PREDICTIONS PRED_WSC: every value None until the lead registers them with PRED_STAMP_WSC; the step script
STOPS the timing cells wc1a / wc1b while PRED_STAMP_WSC is None.
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
import r2_wsb_cell as WB           # noqa: E402
import r2_kv_arena as KA           # noqa: E402

SC = WB.SC
BC = WB.BC
ARMS = ("L4X", "L4XU", "L4XF", "SHR", "UC", "UCR", "UCF", "UCFR")
TIMING_ARMS = ("L4X", "L4XU", "L4XF")
CHECK_ARMS = ("SHR", "UC", "UCR", "UCF", "UCFR")
STORE_UNION_ARMS = ("L4XU", "UC", "UCR")
FORK_ARMS = ("L4XF", "UCF", "UCFR")
FORK_CHECK_GEMM = dict(UCF="natural", UCFR="rowgroup")
TOPK = 64
FORK_FILE = "fork_vllm_unified_attention.py"
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"   # nosi-r2-fia6 94d5bb0:nosi/nosi/fork_vllm_unified_attention.py
STRESS_STORE = 3                   # STRESS: a (head, request) listed more store blocks than one wb1m fill (X 4: 3)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill every None and PRED_STAMP_WSC (`date -u`, L33) in a
# commit BEFORE wc1a / wc1b (the step script stops while PRED_STAMP_WSC is None). Keys per capacity (C128: pool 65;
# C73: pool 10), each (low, high):
#   joint_sh_steps   L4XF's median steady joint-pass ms / SH's steady step ms
#   joint_saving_ms  L4XU's median steady joint-pass ms - L4XF's (same job): the plan's WS-C falsifier is < 15 ms
#   ratio_vs_l4xu    L4XF's / L4XU's median steady committed tok/s
#   ratio_vs_sh      L4XF's median steady committed tok/s / SH's steady tok/s
# The builder's INFERENCE (docs/NOTES_wsc.md section 7, with the arithmetic) is an input for the lead, not a registration.
# The lead's REGISTRATION (2026-10-28T12:59:25Z, `date -u`, before any WS-C cell ran): the builder's bands (NOTES_wsc.md
# section 7), checked against the lead's WS-B bands (r2_wsb_cell.PRED_WSB, 1a8ce68: L4XU / SH (0.27, 0.41) / (0.31,
# 0.50)): L4XU / SH x L4XF / L4XU gives about 0.28-0.47 / 0.32-0.56, which contains these ratio_vs_sh bands. MEASURED basis
# of the saving: SCORING_fia6.md (FV6 complete layer 1.26 ms at 216x4 against the shipped call's 0.61 ms x R; the union
# store calls, merge and empty spill fills of WS-B removed). RISK registered with it: at C 128 B 216 the arena + store
# leave about 0.5 GB of the card (NOTES_wsc.md risk 1); an OOM there is a RUN failure, not a band miss.
PRED_STAMP_WSC = "2026-10-28T12:59:25Z"
PRED_WSC = dict(stamp=PRED_STAMP_WSC,
                C128=dict(joint_sh_steps=(2.0, 3.1), joint_saving_ms=(20.0, 70.0), ratio_vs_l4xu=(1.03, 1.14),
                          ratio_vs_sh=(0.28, 0.40)),
                C73=dict(joint_sh_steps=(2.4, 3.8), joint_saving_ms=(30.0, 100.0), ratio_vs_l4xu=(1.02, 1.12),
                         ratio_vs_sh=(0.33, 0.46)))
PLAN_FALSIFIER_MS = 15.0           # PLAN_L4P_faithful.md WS-C: FALSIFIER saving < 15 ms (REPORT here; the lead scores)


def log(msg):
    print("[r2_wsc %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K drafts per round (L4: K verify rows + 1 draft row)")
    ap.add_argument("--arms", default="L4X,L4XU,L4XF", help="a subset of %s, in run order (SH runs first)" % ",".join(ARMS))
    ap.add_argument("--joint-residency", default="serial", choices=("serial", "union"),
                    help="WS-B's flag: the union arms %s run only with 'union'" % (STORE_UNION_ARMS + FORK_ARMS,))
    ap.add_argument("--joint-attn", default="store", choices=("store", "fork"),
                    help="THE FLAG of WS-C: the arms %s run only with 'fork' (the store arms run as WS-B's)" % (FORK_ARMS,))
    ap.add_argument("--joint-extra", type=int, default=48, help="the store arms' X (WS-B: slot 0 the next block)")
    ap.add_argument("--fork-store", type=int, default=0,
                    help="the fork arms' X (0: the bound, 1 + (full rows - 1) x 64; a smaller X is refused)")
    ap.add_argument("--check-steps", type=int, default=128, help="SHR / UC / UCR / UCF / UCFR: teacher-forced positions")
    ap.add_argument("--check-phase", type=int, default=0,
                    help="wsc2: UCF / UCFR run steps 0 .. phase - 1 serially, then passes from step phase (0: as "
                         "22c27ee; the miniature uses 2 so that a pass spans a 64-token block edge)")
    ap.add_argument("--spec-rounds", type=int, default=0, help="rounds per timing arm (0: ceil((steps - 1) / K))")
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--profile", default="on", choices=("on", "off"), help="check arms: one profiled pass / step")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1),
                    help="1: MUST gate <arm>-STRESS for the check arms (WS-B: the spill ran; fork: the stressed regime)")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    un = [x for x in a.arms if x in STORE_UNION_ARMS + FORK_ARMS]
    if un and a.joint_residency != "union":
        ap.error("the arms %s need --joint-residency union" % un)
    if a.joint_residency == "union" and not un:
        ap.error("--joint-residency union without a union arm")
    fk = [x for x in a.arms if x in FORK_ARMS]
    if fk and a.joint_attn != "fork":
        ap.error("the arms %s need --joint-attn fork" % fk)
    if a.joint_attn == "fork" and not fk:
        ap.error("--joint-attn fork without a fork arm (%s)" % (FORK_ARMS,))
    ck = [x for x in a.arms if x in ("UC", "UCR", "UCF", "UCFR")]
    if ck and ("SHR" not in a.arms or a.arms.index("SHR") > min(a.arms.index(x) for x in ck)):
        ap.error("the check arms compare against SHR: SHR must run before them")
    if a.spec != 3:
        ap.error("--spec: K 3 only (the WS-C cells: K + 1 = 4 rows = one fork tile of 4 rows per request)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    R = a.spec + 1
    if any(x in CHECK_ARMS for x in a.arms) and (a.check_steps < R or a.check_steps > a.steps or a.check_steps % R):
        ap.error("--check-steps: a multiple of K + 1 = %d, at most --steps (the check arms)" % R)
    if not 0 <= a.check_phase < 64 or a.check_phase + R > a.check_steps:
        ap.error("--check-phase in 0 .. 63 with phase + K + 1 <= --check-steps")
    if a.joint_extra < 2 or a.probe_every < 2 or a.spec_rounds < 0 or a.spec_journal_blocks < 1:
        ap.error("--joint-extra >= 2, --probe-every >= 2, --spec-rounds >= 0, journal >= 1")
    if a.fork_store != 0 and a.fork_store < KA.store_slots(TOPK, R):
        ap.error("--fork-store %d: below the bound of the check arms, 1 + (R - 1) x 64 = %d (0 = the bound per arm)"
                 % (a.fork_store, KA.store_slots(TOPK, R)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def cap_key(pool: int) -> str:
    return "C%d" % (BC.C_BASE + int(pool))


def check_schedule(n: int, R: int, phase: int = 0) -> list:
    """wsc2: the first step of each teacher-forced pass of the fork check arms (UCF / UCFR): steps 0 .. phase - 1 run
    as serial ragged steps (the same steps as SHR's), then passes of R rows start at phase, phase + R, ... while the
    pass ends inside SHR's n steps. phase 0 = the passes of 22c27ee (0, R, 2 R, ...)."""
    n, R, phase = int(n), int(R), int(phase)
    if R < 1 or phase < 0 or phase + R > n:
        raise ValueError("check schedule: R >= 1, 0 <= phase, phase + R <= n (n %d, R %d, phase %d)" % (n, R, phase))
    return list(range(phase, n - R + 1, R))


def schedule_facts(S0: int, n: int, R: int, phase: int = 0, bs: int = 64) -> dict:
    """wsc2, the precondition of UCF-STRESS from the schedule alone (every check request starts at S0): rows past a
    block edge (a row whose block is not the block of its pass's row 0) and F4 rows (an in-tail row at p % 64 = 63),
    per request, over the passes of check_schedule."""
    bases = check_schedule(n, R, phase)
    past = f4 = 0
    for a in bases:
        p0 = int(S0) + a
        for r in range(R):
            p = p0 + r
            if p // bs != p0 // bs:
                past += 1
            elif p % bs == bs - 1:
                f4 += 1
    return dict(passes=len(bases), past_edge_rows=past, f4_rows=f4, first=bases[:1], last=bases[-1:])


def fork_store_slots(arm: str, K: int, override: int = 0) -> int:
    """X of a fork arm: L4XF has K full rows (row K is the resident draft row); UCF / UCFR K + 1 (every row a verify
    row). 0 = the bound; an override must not be below it."""
    full = K if arm == "L4XF" else K + 1
    x = KA.store_slots(TOPK, full)
    return max(x, int(override)) if override else x


def store_gb(B: int, X: int, H: int = 2, D: int = 128, bytes_per: int = 2) -> float:
    """The arena's store: B x X x 64 tokens x H x (K D + V D + bias 1) x 2 B (bf16), in GB."""
    return round(B * X * 64 * H * (2 * D + 1) * bytes_per / 1e9, 4)


def check_gates_fork(arm, res, require_stress: bool = False, L: int = 32):
    """The MUST gates of a fork check arm (UCF / UCFR) from run_check_fork's result: [(name, must, passed, detail)].
    WS-B's ERR / SEL / PRESENT / VIS / RESID kept (the same facts); IDX = the "F" probe; new: VB, RA, ATTN1, FILL1;
    STRESS (require_stress): the stressed regime of the big cells."""
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
              "selected blocks of verify rows not present: %d (recount); unserved list entries %d (store X %s, bound "
              "%s); listed host blocks %d, pool blocks read in place %d" % (
                  st["dropped"], st["unserved"], st.get("store_slots"), st.get("bound"), st["extras_host"],
                  st["extras_pool"])))
    g.append(("%s-VIS" % arm, True, st["vis_bad"] == 0 and st["visible_blocks"] > 0,
              "rows whose visible block count (from the kernel's tables) is not their own: %d (per row %s); visible %s, "
              "selected %s per row index" % (st["vis_bad"], st["row_vis_bad"], st["row_visible"], st["row_selected"])))
    g.append(("%s-VB" % arm, True, st["vb_bad"] == 0 and st["vb_rows"] > 0,
              "rows whose (block id, tokens) set in the kernel's tables is not their own selection: %d of %d (head, row) "
              "checks; loaded union entries %d (per row %.3f) against visible %d" % (
                  st["vb_bad"], st["vb_rows"], st["fork_loaded"],
                  (st["fork_loaded"] / max(1, st["vb_rows"]) * 1.0), st["visible_blocks"])))
    passes = st["passes"]
    g.append(("%s-RESID" % arm, True, passes > 0 and st["calls"]["resid"] == L * passes,
              "residency calls %d for %d passes of %d layers" % (st["calls"]["resid"], passes, L)))
    g.append(("%s-ATTN1" % arm, True, passes > 0 and st["calls"]["attn"] == L * passes,
              "attention calls %d for %d passes of %d layers (one fork call per layer)" % (st["calls"]["attn"], passes,
                                                                                         L)))
    g.append(("%s-FILL1" % arm, True, passes > 0 and st["calls"]["fills"] == L * passes,
              "store fills %d for %d passes of %d layers (C = 1)" % (st["calls"]["fills"], passes, L)))
    fa = res["fa_check"] or {}
    f, ra = fa.get("F") or {}, fa.get("RA") or {}
    g.append(("%s-IDX" % arm, True, bool(f.get("entries")) and all(f.get("equal") or [False]),
              "sampled requests alone against the full launch, bitwise: %s" % (f,)))
    g.append(("%s-RA" % arm, True, bool(ra.get("entries")) and all(ra.get("equal") or [False]),
              "a row alone against the full launch, bitwise (G_block0 precondition: %d rows whose first visible block "
              "is not block 0): %s" % (st["block0_bad"], ra)))
    if require_stress:
        ok = (st["past_edge"] > 0 and st["f4_rows"] > 0 and st["extras_max"] > STRESS_STORE and st["store_reads"] > 0
              and st["pool_reads"] > 0)
        g.append(("%s-STRESS" % arm, True, ok,
                  "rows past a block edge %d; F4 rows %d; the most listed store blocks of one (head, request) %d (> %d); "
                  "store reads %d; pool blocks read in place %d" % (st["past_edge"], st["f4_rows"], st["extras_max"],
                                                                   STRESS_STORE, st["store_reads"], st["pool_reads"])))
    return g


def timing_fork_gates(arm, rd, L: int = 32):
    """L4XF: one residency call, ONE fork call and ONE store fill per layer and joint pass (verify_rows host counts)."""
    v = rd["verify"] or {}
    calls, passes = (v.get("calls") or {}), v.get("passes", 0)
    out = []
    for key, name in (("resid", "RESID"), ("attn", "ATTN1"), ("fills", "FILL1")):
        out.append(("%s-%s" % (arm, name), True, passes > 0 and calls.get(key) == L * passes,
                    "%s calls %s for %d joint passes of %d layers" % (key, calls.get(key), passes, L)))
    return out


def verdicts(sh, sp, pool: int):
    """PRED_WSC against the timing arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN when L4XF is
    missing. The plan's WS-C falsifier (saving < 15 ms) is printed as a REPORT."""
    P = PRED_WSC.get(cap_key(pool)) or {}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))
    f, u = (sp or {}).get("L4XF"), (sp or {}).get("L4XU")
    if f is None:
        out.append(("L4XF-RUN", "NOT_RUN", "the arm did not run"))
        return out
    shm = sh["rates"]["steady_step_ms"] if sh else None
    jf = (f.get("phases_median") or {}).get("joint")
    ju = ((u or {}).get("phases_median") or {}).get("joint")
    band("JOINT-SH-STEPS", (jf / shm) if (jf and shm) else None, P.get("joint_sh_steps"))
    saving = (ju - jf) if (ju and jf) else None
    band("JOINT-SAVING-MS", saving, P.get("joint_saving_ms"))
    fm, um = f["rates"]["median_tok_s"], ((u or {}).get("rates") or {}).get("median_tok_s")
    band("RATIO-VS-L4XU", (fm / um) if (fm and um) else None, P.get("ratio_vs_l4xu"))
    band("RATIO-VS-SH", f["vs_sh"]["median_tok_s"], P.get("ratio_vs_sh"))
    out.append(("PLAN-FALSIFIER", "REPORT", "joint saving %s ms against the plan's falsifier < %.0f ms: %s" % (
        None if saving is None else round(saving, 3), PLAN_FALSIFIER_MS,
        "n/a" if saving is None else ("FIRED" if saving < PLAN_FALSIFIER_MS else "not fired"))))
    out.append(("VS-STRONGEST", "REPORT", "L4XF median committed tok/s %s / %.1f = %s" % (
        fm, WB.STRONGEST, round(fm / WB.STRONGEST, 4) if fm else None)))
    return out


def attention_compare(u, f):
    """REPORT: the probe-pass phase groups of L4XU and L4XF (joint-pass ms per pass, medians) and the complete layer
    time: what the fork removed (WS-B: attn + spill + merge; fork: attn) and what it costs (locate, fill, lists)."""
    gu = ((u or {}).get("joint_phases") or {}).get("groups_median") or {}
    gf = ((f or {}).get("joint_phases") or {}).get("groups_median") or {}
    keys = ("residency", "fetch", "attention", "spill", "check", "end", "dense", "twin", "selection", "total")
    rows = {k: (gu.get(k), gf.get(k)) for k in keys}
    lu = ((u or {}).get("joint_phases") or {}).get("layer_ms_median")
    lf = ((f or {}).get("joint_phases") or {}).get("layer_ms_median")
    return dict(groups_L4XU_L4XF=rows, layer_ms_median=(lu, lf))


# ------------------------------------------------------------------ the fork module (GPU)
def import_fork():
    """(module, gate): the fork of THIS tree, benchmarks/Efficiency/fork_vllm_unified_attention.py (a byte copy of
    fia6's; the step script checks its blob); a failed import is the MUST gate C0_FORK."""
    try:
        import fork_vllm_unified_attention as FORK
        ok = callable(getattr(FORK, "unified_attention", None)) and hasattr(FORK, "kernel_unified_attention")
        return FORK, ("C0_FORK", True, ok, "%s (blob pinned by the step: %s)" % (FORK.__file__, FORK_BLOB))
    except Exception as e:                                       # the import is evidence: a failure is a MUST fail
        return None, ("C0_FORK", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300]))


def compile_report(FORK):
    """REPORT: the compiled kernel of the last fork launch (registers, local memory words, shared memory), as fia5."""
    ck = (getattr(FORK, "FORK_LAST_LAUNCH", None) or {}).get("kernel") if FORK is not None else None
    if ck is None:
        return None
    md = getattr(ck, "metadata", None)
    return dict(num_warps=getattr(md, "num_warps", None), num_stages=getattr(md, "num_stages", None),
                regs=getattr(ck, "n_regs", None), local_words=getattr(ck, "n_spills", None),
                shared=getattr(md, "shared", None))


def _prefill_arena(model, ids, inputs, steps: int, clock, X: int):
    """Its own shipped prefill (SH's cache and max_gen_len), with the engines re-homed into a new K / V arena of X store
    slots per request (r2_kv_arena.PrefillHook): (cache, arena, prefill s, first-token agreement)."""
    from nosi.cache_engine import InfLLMv2Cache
    WB._release_warm(model)
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    arena = KA.for_engine(cache.layers[0].cache_engine, len(cache.layers), int(ids.shape[0]), X)
    clock.sync()
    t0 = time.time()
    with KA.PrefillHook(cache, arena):
        logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    first = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    return cache, arena, round(time.time() - t0, 1), first


# ------------------------------------------------------------------ the fork timing arm (GPU)
def run_spec_fork(model, ids, inputs, steps: int, clock, opt, fork):
    """L4XF: r2_wsb_cell.run_spec's flow (L4XU's SpecOps) with joint_attn "fork", the arena and the phase proxy."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    K = int(opt["K"])
    X = fork_store_slots("L4XF", K, opt["fork_store"])
    cache, arena, prefill_s, first = _prefill_arena(model, ids, inputs, steps, clock, X)
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    sops = SR.SpecOps(level="L4", K=K, mode="union", gemm="natural", extra=opt["extra"], journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"], joint_stats=False, joint_attn="fork",
                      fork=fork.unified_attention)
    runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
    proxy = WB.PhaseVR(runner.VR, runner, clock)
    runner.VR = proxy
    G = SC.rounds_for(steps, K, opt["rounds"])
    runner.run(steps, rounds=G)
    rd = runner.read()
    VR = proxy._VR
    phases = [VR.phase_ms(m, clock.elapsed) for m in proxy.marks]
    layers = [VR.layer_ms(m, clock.elapsed) for m in proxy.marks]
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=prefill_s, first_token_agreement=first,
               meta=dict(rs.meta, spec=runner.meta, arena=arena.describe()), joint_phases=WB.phase_summary(phases, layers),
               calls_per_pass=WB.per_pass((rd["verify"] or {}).get("calls", {}), (rd["verify"] or {}).get("passes", 0)),
               roles=dict(verify_rows_per_request=K, draft_rows_per_request=1, joint_rows=K + 1),
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4)), compile=compile_report(fork))
    runner.detach()
    del runner, rs, cache, arena
    return res


# ------------------------------------------------------------------ the fork check arms (GPU)
def run_check_fork(model, ids, inputs, steps: int, clock, opt, shr, fork):
    """UCF / UCFR: fork joint passes teacher-forced on SH's tokens, check on, against SHR (the loop of
    r2_wsb_cell.run_check_union, with the arena, joint_attn "fork" and the "F" / "RA" probes)."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import verify_rows as VR
    K = int(opt["K"])
    R = K + 1
    X = fork_store_slots(opt["arm"], K, opt["fork_store"])
    cache, arena, prefill_s, first = _prefill_arena(model, ids, inputs, steps, clock, X)
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=False)
    rs = RE.attach(model, cache, BC.S0, steps, ops)
    KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    B, L = rs.B, rs.L
    vs = VR.attach(model, rs, VR.VerifyOps(K=K, mode="union", gemm=FORK_CHECK_GEMM[opt["arm"]], extra=opt["extra"],
                                           check=True, stats=True, joint_attn="fork", fork=fork.unified_attention))
    vs.clock = clock
    vs.fa_check = {}
    n = shr["n"]
    P = int(opt.get("check_phase", 0))
    bases = check_schedule(n, R, P)                                  # pass g starts at step bases[g] (S0 + bases[g])
    G = len(bases)
    for t in range(P):                                               # wsc2: the serial prefix (SHR's own steps 0..P-1)
        RE.decode_step(model, rs, inputs[:, t:t + 1].contiguous(),
                       torch.full((B, 1), BC.S0 + t, dtype=torch.int32, device="cuda"))
    every = int(opt["probe_every"])
    offs = torch.arange(R, dtype=torch.int64, device="cuda")
    changed_lr = torch.zeros((L, R), dtype=torch.int64, device="cuda")
    hist = torch.zeros(6, dtype=torch.int64, device="cuda")           # symmetric-difference sizes 2, 4, .., 10, 12+
    compared, first_bad = 0, []
    d_top, d_max, agree = [], [], []
    marks, prof = [], None
    is_probe = lambda g: every > 0 and g > 0 and g % every == 0         # noqa: E731
    g_prof = next((g for g in range(G // 2, G) if not is_probe(g)), None)
    for g in range(G):
        probe = is_probe(g)
        vs.marks = [] if probe else None
        a0, a1 = bases[g], bases[g] + R                                  # the pass's steps (SHR's indices)
        ids_g = inputs[:, a0:a1].contiguous()
        pos_g = (BC.S0 + a0 + offs).to(torch.int32).view(1, R).expand(B, R).contiguous()
        if opt["profile"] and g == g_prof:                           # CHECK-ONLY: the kernels one pass dispatches
            lg, prof = WB._profile(lambda: VR.joint_pass(model, vs, ids_g, pos_g))
        else:
            lg = VR.joint_pass(model, vs, ids_g, pos_g)
        if probe:
            marks.append(vs.marks)
            vs.marks = None
        ref = shr["sel"][a0:a1].to("cuda").to(torch.int64)                  # (R, L, H, B, T)
        for l in range(L):
            cur = vs.chk_sel[l]                                              # (H, B, R, T)
            rf = ref[:, l].permute(1, 2, 0, 3)
            eq = (cur.sort(-1).values == rf.sort(-1).values).all(-1)
            nb = ~eq
            changed_lr[l] += nb.sum((0, 1))
            only = (~(cur.unsqueeze(-1) == rf.unsqueeze(-2)).any(-1)).sum(-1)
            sym = 2 * only
            hist += torch.bincount((sym // 2 - 1).clamp(min=0, max=5)[nb], minlength=6)[:6]
            if len(first_bad) < 5 and bool(nb.any()):
                for h, b, r in nb.nonzero().tolist()[:5 - len(first_bad)]:
                    first_bad.append((g, l, r, h, b, int(sym[h, b, r])))
        compared += L * vs.H * B * R
        tv = shr["topv"][a0:a1].to("cuda")
        ti = shr["topi"][a0:a1].to("cuda")
        lgr = lg.permute(1, 0, 2)
        d_top.append((lgr.gather(-1, ti) - tv).abs().amax(-1).cpu())
        d_max.append((lgr.amax(-1) - shr["fmax"][:, a0:a1].t().to("cuda")).abs().cpu())
        agree.append((lg.argmax(-1) == inputs[:, a0 + 1:a1 + 1]).cpu())
    clock.sync()
    phases = [VR.phase_ms(m, clock.elapsed) for m in marks]
    layers = [VR.layer_ms(m, clock.elapsed) for m in marks]
    stats = VR.read_stats(vs)
    dt = torch.stack(d_top)
    dm = torch.stack(d_max)
    ag = torch.stack(agree)
    by_row = {r: dict(top16_abs=WB.quantiles(dt[:, r].reshape(-1).tolist()),
                      max_abs=WB.quantiles(dm[:, r].reshape(-1).tolist()),
                      argmax_agree=round(float(ag[:, :, r].float().mean()), 6)) for r in range(R)}
    cl = changed_lr.cpu()
    sel_rec = dict(compared=compared, changed=int(cl.sum()), by_layer_nonzero={l: cl[l].tolist() for l in range(L)
                                                                                if int(cl[l].sum())},
                   by_row=cl.sum(0).tolist(), symdiff_hist=dict(zip(("2", "4", "6", "8", "10", "12+"),
                                                                     hist.cpu().tolist())), first=first_bad)
    res = dict(stats=stats, sel=sel_rec, logits=dict(by_row=by_row, top16_abs=WB.quantiles(dt.reshape(-1).tolist()),
                                                    argmax_agree=round(float(ag.float().mean()), 6)),
               errors=RE.errors(rs), fa_check=vs.fa_check, passes=G, R=R, joint_phases=WB.phase_summary(phases, layers),
               calls_per_pass=WB.per_pass(stats["calls"], stats["passes"]), profile=prof, prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, verify=vs.meta),
               ready=dict(win_sel=stats["win_sel"], win_host=stats["win_host"], extras_pool=stats["extras_pool"],
                          extras_host=stats["extras_host"]),
               roles=dict(rows_full=stats["rows_full"], rows_res=stats["rows_res"], passes=G),
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4)), compile=compile_report(fork))
    del vs, rs, cache, arena
    return res


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
    FORK = None
    if any(x in FORK_ARMS for x in a.arms):
        FORK, g0 = import_fork()
        gates.append(g0)
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
    dims = dict(L=L, H=H, topk=TOPK, block_bytes=64 * D * 2 * 2 + 64 * 2)
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
            if arm in ("UC", "UCR", "UCF", "UCFR") and shr is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: no SHR record"))
                continue
            if arm in FORK_ARMS and FORK is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: the fork did not import (C0_FORK)"))
                continue
            BC.free(host=True)
            base_arm = {"UCF": "UC", "UCFR": "UCR"}.get(arm, arm)
            opt = dict(arm=arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, twin=a.spec_twin,
                       body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                       probe_every=a.probe_every, extra=a.joint_extra, spill=0, fork_store=a.fork_store,
                       check_phase=a.check_phase,
                       profile=a.profile == "on")
            log("arm %s (%s)" % (arm, opt))
            try:
                if arm in TIMING_ARMS:
                    if arm == "L4XF":
                        res = run_spec_fork(model, ids, inputs, a.steps, clock, opt, FORK)
                    else:
                        res = WB.run_spec(model, ids, inputs, a.steps, clock, opt)
                    pa, pr = BC.gpu_peak()
                    base = "L4X" if arm == "L4X" else "L4F"      # L4F: r2_spec_cell's FAST level-L4 arm (no TOK / FP)
                    rec = SC.summarize(base, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               joint_phases=res["joint_phases"], calls_per_pass=res["calls_per_pass"],
                               roles=res["roles"], store=res.get("store"), compile=res.get("compile"))
                    jm = rec["phases_median"].get("joint")
                    shm = sh["rates"]["steady_step_ms"] if sh else None
                    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                    rec["vs_strongest"] = (round(rec["rates"]["median_tok_s"] / WB.STRONGEST, 4)
                                           if rec["rates"]["median_tok_s"] else None)
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend((n.replace(base + "-", arm + "-", 1), m, ok, d)
                                 for n, m, ok, d in SC.arm_gates(base, res, rec))
                    if arm == "L4XU":
                        gates.extend(WB.timing_union_gates(arm, res["read"], L))
                    elif arm == "L4XF":
                        gates.extend(timing_fork_gates(arm, res["read"], L))
                elif arm == "SHR":
                    n = a.check_steps
                    shr = WB.run_shr(model, ids, inputs, a.steps, n, clock, opt, fp_sh)
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
                elif arm in ("UC", "UCR"):
                    res = WB.run_check_union(model, ids, inputs, a.steps, clock, dict(opt, arm=base_arm), shr)
                    pa, pr = BC.gpu_peak()
                    res.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               ready_record=WB.ready_record(res["ready"], res["passes"], dims["block_bytes"]))
                    chk[arm] = res
                    gates.append(("%s-RUN" % arm, True, True, "%d teacher-forced passes of R %d" % (res["passes"],
                                                                                              res["R"])))
                    gates.extend(WB.check_gates(arm, res, bool(a.require_stress), L))
                else:
                    res = run_check_fork(model, ids, inputs, a.steps, clock, opt, shr, FORK)
                    pa, pr = BC.gpu_peak()
                    res.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               ready_record=WB.ready_record(res["ready"], res["passes"], dims["block_bytes"]))
                    chk[arm] = res
                    gates.append(("%s-RUN" % arm, True, True, "%d teacher-forced passes of R %d" % (res["passes"],
                                                                                              res["R"])))
                    gates.extend(check_gates_fork(arm, res, bool(a.require_stress), L))
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.pool)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, joint_residency=a.joint_residency,
                joint_attn=a.joint_attn, joint_extra=a.joint_extra, fork_store=a.fork_store,
                fork_store_slots={x: fork_store_slots(x, a.spec, a.fork_store) for x in FORK_ARMS if x in a.arms},
                check_steps=a.check_steps, check_phase=a.check_phase,
                check_schedule=(schedule_facts(BC.S0, a.check_steps, a.spec + 1, a.check_phase)
                                if any(x in ("UCF", "UCFR") for x in a.arms) else None),
                spec_rounds=a.spec_rounds, spec_journal_blocks=a.spec_journal_blocks,
                spec_twin=a.spec_twin, spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, profile=a.profile, require_stress=a.require_stress, predictions=PRED_WSC,
                phase_groups=WB.PHASE_GROUPS, dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"),
                strongest=WB.STRONGEST, fork_file=FORK_FILE, fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp, CHECK=chk,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds],
                   attention_compare=attention_compare(sp.get("L4XU"), sp.get("L4XF")))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-16s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is not None:
            print("REPORT %s K %d: %d rounds; committed tok/s median %s (vs SH %s, vs strongest %s); joint pass %s SH "
                  "steps; phases %s; joint-pass phases (probe) %s; calls per pass %s; KV at verify %s; tokens equal %.6f; "
                  "store %s; compile %s; peak %s GB" % (
                      name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["vs_sh"], r["vs_strongest"],
                      r["joint_sh_steps"], r["phases_median"], r["joint_phases"], r["calls_per_pass"],
                      r["kv_at_verify"], r["tokens"]["equal_fraction"], r.get("store"), r.get("compile"),
                      r["gpu_peak_alloc_GB"]))
        c = chk.get(name)
        if c is not None and name != "SHR":
            print("REPORT %s: selections %s; logits %s; stats %s; ready / residual %s; phases %s; calls per pass %s; "
                  "store %s; profile %s" % (name, c["sel"], c["logits"], c["stats"], c["ready_record"],
                                           c["joint_phases"], c["calls_per_pass"], c.get("store"), c["profile"]))
        elif c is not None:
            print("REPORT SHR: fingerprints %s; argmax unequal %d; profile %s" % (c["fingerprints"], c["tok_bad"],
                                                                                  c["profile"]))
    if "L4XU" in sp and "L4XF" in sp:
        print("REPORT ATTENTION L4XU vs L4XF (probe-pass group medians, ms per pass; complete layer ms): %s"
              % (summary["attention_compare"],))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_wsc: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
