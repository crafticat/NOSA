"""BF1 cell (KS/analysis/plan/DESIGN_BGFETCH_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE A/B OF THE
DRAFT-ISSUED BACKGROUND FETCH on the best arm. The mechanism: benchmarks/Efficiency/r2_bf1_fetch.py (its docstring). Step
script: benchmarks/Efficiency/r2_bf1_step.sh (cells bf1m, bf1a, bf1b). NO NEW KERNEL.

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order of ARMS (the I4DC cell's, benchmarks/Efficiency/
r2_i4dc_cell.py, imported, not edited) and the BF1 arms last:
  L4PTFDGUDKWCDX  = L4PTFDGUDKWCD with BF1's classes, the fetch OFF, the store X of --bf-store (the matched control: the
                    same code path and memory, no prefetch)
  L4PTFDGUDKWCDB  = L4PTFDGUDKWCD + BF1 (mode "ce"), the store X of --bf-store
A BF1 arm needs L4PTFDGUDKWCD EARLIER in the job (its LISTED reference; it also compiled the helpers). Every arm runs
run_arm_bf = the I4DC cell's run_arm_d line for line plus the lines marked "# BF1" (tests_cpu/test_bf1.py checks it).
MUST GATES: the I4DC cell's for every arm; a BF1 arm takes L4PTFDGUDKWCD's gates under its own name (EQF = L4PTFD bit for
bit: tokens, check-round fingerprints, state checksums, acceptance record) except COUNTS (here without round_blocks, the
NEW blocks: the quantity that BF1 changes) and KEEP-EQ / KEEP-REF (the keep's fetched and window-hit counts change by
design), plus -LISTED (the keep's listed blocks per joint pass = L4PTFDGUDKWCD's: J's selections and residency did not
change) and -BF (B: the mechanism ran, issued, committed, conserved, no worker error; X: off, nothing issued).
REPORT: NEW blocks per round and the probe-round gather wait (kv_at_verify) against L4PTFDGUDKWCD, the barrier wait, the
H2D and pack rates, late / capped / kept-alive items, used / wasted bytes.
REGISTERED PREDICTIONS PRED_BF1_BY_B: every value None until the lead registers them with PRED_STAMP_BF1 (a None key is
REPORT); the step script STOPS the timing cells while PRED_STAMP_BF1 is None.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_i4dc_cell as DCC        # noqa: E402
import r2_bf1_fetch as BF         # noqa: E402

CCC, CD = DCC.CCC, DCC.CD
S4, CC, IA, GC, DG, LN, GR, HC, KC, UC, DC, KP, SK = (DCC.S4, DCC.CC, DCC.IA, DCC.GC, DCC.DG, DCC.LN, DCC.GR, DCC.HC,
                                                      DCC.KC, DCC.UC, DCC.DC, DCC.KP, DCC.SK)
I3C, WA, SC, BC = DCC.I3C, DCC.WA, DCC.SC, DCC.BC
REF = "L4PTFDGUDKWCD"
BF_ARMS = dict(L4PTFDGUDKWCDX="off", L4PTFDGUDKWCDB="ce")           # a BF1 arm -> its fetch mode
ARMS = DCC.ARMS + tuple(BF_ARMS)
GRAPH_ARMS = DCC.GRAPH_ARMS + tuple(BF_ARMS)
TOPK = DCC.TOPK
FORK_BLOB = DCC.FORK_BLOB
STRONGEST = DCC.STRONGEST
BUDGET_GB = DCC.BUDGET_GB
USABLE_GB = DCC.USABLE_GB
BASE_D = DCC.BASE_D
DROP = ("-COUNTS", "-KEEP-EQ", "-KEEP-REF")
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM (PRED_STAMP_BF1 from `date -u`, L33, in a commit before
# bf1a / bf1b). Keys (low, high) unless noted; B = L4PTFDGUDKWCDB, X = L4PTFDGUDKWCDX, R = L4PTFDGUDKWCD (same job):
#   BF1_vs_X        [x] B / X median steady committed tok/s
#   BF1_vs_R        [x] B / R
#   BF1_new_ratio   [x] B's NEW blocks per steady round / X's (kv_at_verify)
#   BF1_wait_saving [ms per round] X's probe-round gather wait median - B's
#   BF1_joint_saving[ms per round] X's joint phase median - B's
#   BF1_draft_ratio [x] B's draft phase median / X's (the interference of the copies and the side commits)
#   BF1_barrier_ms  [ms] B's barrier wait median
#   BF1_eq          [True] B-EQF passes
# THE WRITER'S MODEL (an input for the lead, not a registration). Inputs: the exposed gather 20.25 / 115.64 ms per round
# (RECON7_R3 table 2: C 128 all arms / C 73 best arm); flagA share 0.46-0.48 (r7pm, C 73 B 8, lead's message); X1:
# interference 0.8-3.4 %. Coverage c = used / NEW_X in 0.35 .. 0.55 (flagA minus late / no-room / evicted losses).
#   C 73 B 318 (pool 10): wait saving c x 115.6 = 40 .. 64 ms; draft phase + 0.8-3.4 % of 176 ms = 1.4 .. 6 ms; host GIL
#     contention unknown; round 524.7 -> 466 .. 491 ms: BF1_vs_X 1.07 .. 1.13 (central 1.10).
#   C 128 B 216 (pool 65): THE C 128 DECISION (MODEL on MEASURED inputs): the keep + window arm already cuts the exposed
#     gather 20.25 -> 13.65 ms (R3 table 2) but costs about 1.4 % (I4D eager: joint +5.5 ms, SI4:212-229); BF1 on it hides
#     c x 13.65 = 4.8 .. 7.5 ms, minus copy interference in the draft phase 0.8-3.4 % of 132 ms = 1.1 .. 4.5 ms: B / X
#     1.001 .. 1.020; B / L4PTFDGUDCD = (-4.6 + 0.3 .. 6.4 ms) / 324.9 = 0.987 .. 1.006 (inside the 1 % cross-run spread).
#     Ceiling (c = 1, no interference, keep at its eager cost): +9 ms = 1.028x. BF1 on the PLAIN arm L4PTFDGUDCD is not
#     buildable without the keep machinery (its store is a per-layer scratch fill with no map; a separate prefetch region
#     of 128 slots per request is 1.82 GB at B 216, and that arm's prefill peak is 83.90 GB of 84.0). DECISION: no BF1
#     lever at C 128; bf1a is a DIAGNOSTIC only (full-B coverage; the first graph-mode KW vs L4PTFDGUDCD at C 128); adopt
#     there only if B / L4PTFDGUDCD >= 1.02 in the same process.
PRED_STAMP_BF1 = "2026-10-31T13:43:47Z"
PRED_KEYS = ("BF1_vs_X", "BF1_vs_R", "BF1_new_ratio", "BF1_wait_saving", "BF1_joint_saving", "BF1_draft_ratio",
             "BF1_barrier_ms", "BF1_eq")
# THE LEAD'S REGISTRATION (2026-10-31T13:43:47Z, before bf1a / bf1b; the writer's bands; r7pb MEASURED flagA 0.469 at
# C 73 B 318 (R/recon7_pb1_2195630_r7pb) and r7pa 0.509 at C 128 B 216, inside the modelled 0.46-0.50).
PRED_BF1_BY_B = {
    216: dict(BF1_vs_X=(1.001, 1.020), BF1_vs_R=(0.995, 1.020), BF1_new_ratio=(0.45, 0.65), BF1_wait_saving=(3.0, 8.0),
              BF1_joint_saving=(3.0, 8.0), BF1_draft_ratio=(1.00, 1.04), BF1_barrier_ms=(0.0, 10.0), BF1_eq=True),
    318: dict(BF1_vs_X=(1.06, 1.14), BF1_vs_R=(1.05, 1.14), BF1_new_ratio=(0.45, 0.65), BF1_wait_saving=(35.0, 65.0),
              BF1_joint_saving=(35.0, 65.0), BF1_draft_ratio=(1.00, 1.04), BF1_barrier_ms=(0.0, 10.0), BF1_eq=True)}


def runner_choice(arm: str, check: bool):
    """The I4DC cell's runner choice; a BF1 arm: the kind SKCDB with L4PTFDGUDKWCD's keyword arguments."""
    if arm in BF_ARMS:
        return "SKCDB", dict(keep=True, kwin=True, kcheck=bool(check), cc_backend="inductor")
    return DCC.runner_choice(arm, check)


def runner_class(kind: str, STF):
    if kind == "SKCDB":
        return BF.bf_classes(STF)[1]
    return DCC.runner_class(kind, STF)


def log(msg):
    print("[r2_bf1 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=318)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=10, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDX,L4PTFDGUDKWCDB",
                    help="a subset of %s, in this order (SH runs first; L4PTFD first)" % ",".join(ARMS))
    ap.add_argument("--fork-store", type=int, default=0, help="X of the I4DC arms (0: the bound 1 + (K - 1) x 64)")
    ap.add_argument("--bf-store", type=int, default=0, help="X of the BF1 arms (0: --fork-store's)")
    ap.add_argument("--bf-ncap", type=int, default=2048, help="items per (draft pass, layer) job (the landing entry)")
    ap.add_argument("--bf-ring", type=int, default=8, help="landing / staging entries")
    ap.add_argument("--bf-worker", default="thread", choices=BF.WORKERS)
    ap.add_argument("--bf-acct", type=int, default=0, choices=(0, 1), help="1: count the window (kwin) uses too")
    ap.add_argument("--bf-close-wait-ms", type=float, default=0.0, help="host wait for the last jobs at J's barrier")
    ap.add_argument("--bf-packer", default="gilfree", choices=BF.PACKERS,
                    help="gilfree: r2_transport_gilfree_v3 (C++ OpenMP team, GIL released; the production path); "
                         "torch: index_select into pinned staging on the worker thread")
    ap.add_argument("--bf-threads", type=int, default=14, help="the gilfree packing team (16 cpus: the compute thread "
                                                                   "and the BF1 worker keep 2)")
    ap.add_argument("--bf-cap-groups", type=int, default=1024, help="gilfree: (head, block) groups per packed chunk")
    ap.add_argument("--bf-late", default="defer", choices=BF.LATES,
                    help="defer: a job not issued at the joint pass's start commits at ITS layer of the pass; drop: "
                         "abandoned")
    ap.add_argument("--bf-late-wait-ms", type=float, default=100.0,
                    help="defer: the host wait budget per joint pass for late jobs (the host runs ahead of the GPU)")
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"))
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"))
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0)
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms] or a.arms[0] != "L4PTFD":
        ap.error("--arms: L4PTFD first, then in the order %s (EQF compares with L4PTFD)" % ",".join(ARMS))
    for x, c in BASE_D.items():
        if x in a.arms and c not in a.arms:
            ap.error("--arms: %s needs its C arm %s (it compiles the joint helpers)" % (x, c))
    if any(x in a.arms for x in BF_ARMS) and REF not in a.arms:
        ap.error("--arms: a BF1 arm needs %s earlier (its LISTED reference)" % REF)
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.check_every == 1 and a.batch > 64:
        ap.error("--check-every 1: the keep's byte check needs a scratch store: --batch <= 64")
    if a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, --check-every >= 0, journal >= 1")
    bound = I3C.KA.store_slots(TOPK, a.spec)
    for name, x in (("--fork-store", a.fork_store), ("--bf-store", a.bf_store)):
        if x != 0 and x < bound:
            ap.error("%s %d: below the bound 1 + (K - 1) x 64 = %d (0 = the default)" % (name, x, bound))
    if (a.bf_ncap < 1 or a.bf_ring < 2 or a.bf_close_wait_ms < 0 or not 1 <= a.bf_threads <= 64
            or not 16 <= a.bf_cap_groups <= 4096 or a.bf_late_wait_ms < 0):
        ap.error("--bf-ncap >= 1, --bf-ring >= 2, --bf-close-wait-ms >= 0, --bf-threads in 1 .. 64, --bf-cap-groups in "
                 "16 .. 4096, --bf-late-wait-ms >= 0")
    if a.bf_packer == "gilfree" and any(x in a.arms for x in BF_ARMS) and not os.environ.get("TORCH_EXTENSIONS_DIR"):
        ap.error("--bf-packer gilfree: TORCH_EXTENSIONS_DIR must name the run's build directory (never HOME)")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


def bf_opt(a, arm: str) -> dict:
    """BgFetch's keyword arguments of a BF1 arm."""
    return dict(mode=BF_ARMS[arm], n_cap=a.bf_ncap, ring=a.bf_ring, worker=a.bf_worker, acct=bool(a.bf_acct),
                close_wait_ms=a.bf_close_wait_ms, packer=a.bf_packer, threads=a.bf_threads,
                gf=dict(cap_groups=a.bf_cap_groups), late=a.bf_late, late_wait_ms=a.bf_late_wait_ms)


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_bf(model, ids, inputs, steps: int, clock, opt, FORK, arm: str):
    """The I4DC cell's run_arm_d line for line plus the lines marked BF1: a BF1 arm's store X, its runner kind (SKCDB:
    the I4DC configure and graph module) and its BgFetch options."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, check)
    kw = dict(kw, bf=opt["bf"]) if arm in BF_ARMS else kw   # BF1
    Runner = runner_class(kind, STF)
    if kind in ("SC", "SKC", "SCD", "SKCD"):
        CC.configure()
    if kind == "SKCDB":   # BF1
        CC.configure()   # BF1
    gb = 1024 ** 3
    K = int(opt["K"])
    X = I3C.fork_store_slots("L4PTF", K, opt["fork_store"])
    X = I3C.fork_store_slots("L4PTF", K, opt["bf_store"]) if (arm in BF_ARMS and opt["bf_store"]) else X   # BF1
    cache, arena, prefill_s, first, mem = I3C._prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    I3C.KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()                              # the decode peak (the prefill peak is above)
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=opt["tu_gemm"], extra=X, journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    gk = dict(GR=GR) if kind in ("S", "SK", "SC", "SKC", "SCD", "SKCD") else {}
    gk = dict(GR=GR) if kind == "SKCDB" else gk   # BF1
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True, **kw, **gk)
    if check:
        runner.vs.fa_check = {}                                        # CHECK-ONLY: the "F" / "RA" probes
    runner.run(steps)
    rd = runner.read()
    rd.pop("trace_all", None)
    clock.sync()
    mem.update(decode_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               decode_peak_reserved_GB=round(torch.cuda.max_memory_reserved() / gb, 3))
    phases = [VR.phase_ms(m, clock.elapsed) for m in runner.marks if m]
    layers = [VR.layer_ms(m, clock.elapsed) for m in runner.marks if m]
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()),
               fp_ok=runner.fp_ok.clone(), K=K, probe_every=opt["probe_every"], prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, spec=runner.meta), undo=None, mem=mem,
               joint_phases=I3C.fork_phase_summary(phases, layers), fa_check=runner.vs.fa_check,
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4), cb0_GB=round(arena.chain_bytes() / 1e9, 4)),
               compile=I3C._compile_report(FORK), state=IA.state_checksum(runner), runner_kind=kind, runner_kw=kw)
    runner.detach()
    del runner, rs, cache, arena
    return res


# ------------------------------------------------------------------ pure parts (CPU-tested)
def listed_gate(arm, kp, kp_ref, ref: str):
    """<arm>-LISTED (MUST): the keep's listed host blocks per joint pass = ref's (J's selections and residency did not
    change; BF1 changes only where the blocks come from)."""
    if not kp_ref or not kp:
        return ("%s-LISTED" % arm, True, False, "no keep record of %s or of the arm" % ref)
    a, b = list(kp.get("listed") or []), list(kp_ref.get("listed") or [])
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    ok = len(a) == len(b) and not bad and sum(a) > 0
    return ("%s-LISTED" % arm, True, ok, "joint passes %d / %d; listed totals %s / %s; passes that differ %d %s" % (
        len(a), len(b), sum(a), sum(b), len(bad), bad[:5]))


def bf_gate(arm, bf, mode: str):
    """<arm>-BF (MUST): mode "ce": no worker error, jobs issued, items committed, the host items = the device items,
    committed + revived + no_room = items, tags conserved (used + unused + pending = committed + revived); mode "off":
    nothing issued."""
    bf = bf or {}
    if not bf:
        return ("%s-BF" % arm, True, False, "no bf record")
    if mode == "off":
        ok = bf.get("mode") == "off" and not bf.get("issued") and not bf.get("items")
        return ("%s-BF" % arm, True, bool(ok), "mode %s; issued %s; items %s" % (bf.get("mode"), bf.get("issued"),
                                                                                  bf.get("items")))
    items, sent = bf.get("items"), bf.get("items_sent")
    cons = (bf.get("committed", 0) + bf.get("revived", 0) + bf.get("no_room", 0)
            + bf.get("no_room_deferred", 0)) == items
    tags = (bf.get("used_store", 0) + bf.get("used_win", 0) + bf.get("unused", 0) + bf.get("pending", 0)
            == bf.get("committed", 0) + bf.get("revived", 0))
    room = bf.get("no_room", 0) == 0 and bf.get("revived", 0) == 0    # THE ROOM RULE: the host never sends what the
    ok = (bf.get("error") is None and bf.get("mode") == mode and (bf.get("issued") or 0) > 0 and items == sent   # commit
          and (bf.get("committed") or 0) > 0 and cons and tags and room)                                     # cannot place
    if bf.get("packer") == "gilfree":                                  # the C++ team ran at the requested size
        ok = ok and bool((bf.get("transport") or {}).get("threads_ok"))
    return ("%s-BF" % arm, True, bool(ok),
            "error %s; jobs %s, issued %s, late %s; items seen %s, kept or sent before %s, capped %s, sent %s = device "
            "%s; committed %s, revived %s, no room %s (sum ok %s); used store %s, window %s, unused %s, pending %s (tags "
            "ok %s)" % (bf.get("error"), bf.get("jobs"), bf.get("issued"), bf.get("late"), bf.get("items_seen"),
                        bf.get("items_have"), bf.get("items_cap"), sent, items, bf.get("committed"),
                        bf.get("revived"), bf.get("no_room"), cons, bf.get("used_store"), bf.get("used_win"),
                        bf.get("unused"), bf.get("pending"), tags))


def arm_gates_bf(arm, rd, rec, cpu, keep, check: bool, B: int, pool: int, steps: int, L: int, pa):
    """The I4DC cell's arm gates; a BF1 arm: REF's gates under its own name without DROP, COUNTS against REF without
    round_blocks, -LISTED against REF, -BF."""
    if arm not in BF_ARMS:
        return DCC.arm_gates(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
    out = []
    for n, m, ok, d in DCC.arm_gates(REF, rd, rec, cpu, keep, check, B, pool, steps, L, pa):
        n2 = arm + n[len(REF):] if n.startswith(REF + "-") else n
        if not any(n2 == arm + x for x in DROP):
            out.append((n2, m, ok, d))
    ref = keep.get(REF)
    out.append(KC.counts_gate(arm, cpu["counts"], ref["counts"] if ref else None, REF, skip=("round_blocks",)))
    out.append(listed_gate(arm, rd.get("keep"), (ref or {}).get("keep"), REF))
    out.append(bf_gate(arm, rd.get("bf"), BF_ARMS[arm]))
    return out


def new_record(rec, rd) -> dict:
    """NEW host blocks of the arm's steady rounds (kv_at_verify) and the per-round list (round_blocks)."""
    kv = rec.get("kv_at_verify") or {}
    rb = list(rd.get("round_blocks") or [])
    return dict(new_blocks_per_round=kv.get("new_blocks_per_round"), wait_ms_median=kv.get("wait_ms_median"),
                round_blocks_total=sum(rb), rounds=len(rb))


def verdicts(sp, B: int, eqf=None):
    """PRED_BF1_BY_B[B] against B / X / R of the job: [(name, verdict, detail)]; None registered gives REPORT."""
    P = PRED_BF1_BY_B.get(B) or {k: None for k in PRED_KEYS}
    out = []
    b, x, r = sp.get("L4PTFDGUDKWCDB"), sp.get("L4PTFDGUDKWCDX"), sp.get(REF)
    if b is None:
        return [("BF1-RUN", "NOT_RUN", "no L4PTFDGUDKWCDB in this job")]

    def band(name, v, bb):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (bb,)))
        elif bb is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if bb[0] <= v <= bb[1] else "REFUTED", "%.4f; registered %s" % (v, bb)))

    def med(z):
        return ((z or {}).get("rates") or {}).get("median_tok_s")

    def ph(z, n):
        return ((z or {}).get("phases_median") or {}).get(n)

    def kv(z, n):
        return ((z or {}).get("kv_at_verify") or {}).get(n)

    def ratio(p, q):
        return (p / q) if (p is not None and q) else None

    def diff(p, q):
        return (p - q) if (p is not None and q is not None) else None
    band("BF1-VS-X", ratio(med(b), med(x)), P.get("BF1_vs_X"))
    band("BF1-VS-R", ratio(med(b), med(r)), P.get("BF1_vs_R"))
    band("BF1-NEW-RATIO", ratio(kv(b, "new_blocks_per_round"), kv(x, "new_blocks_per_round")), P.get("BF1_new_ratio"))
    band("BF1-WAIT-SAVING", diff(kv(x, "wait_ms_median"), kv(b, "wait_ms_median")), P.get("BF1_wait_saving"))
    band("BF1-JOINT-SAVING", diff(ph(x, "joint"), ph(b, "joint")), P.get("BF1_joint_saving"))
    band("BF1-DRAFT-RATIO", ratio(ph(b, "draft"), ph(x, "draft")), P.get("BF1_draft_ratio"))
    band("BF1-BARRIER-MS", ((b.get("bf") or {}).get("barrier_ms_median")), P.get("BF1_barrier_ms"))
    want = P.get("BF1_eq")
    out.append(("BF1-EQ", "REPORT" if (want is None or eqf is None) else ("CONFIRMED" if eqf == bool(want) else "REFUTED"),
                "L4PTFDGUDKWCDB-EQF passed %s; registered %s" % (eqf, want)))
    return out


def bf_lines(name, r) -> list:
    """The REPORT lines of a BF1 arm (the formula terms of AREA 2)."""
    bf = r.get("bf") or {}
    nr = r.get("new") or {}
    used = (bf.get("used_store") or 0) + (bf.get("used_win") or 0)
    return ["BF1 %s: mode %s worker %s X %s n_cap %s ring %s; epochs %s, jobs %s, issued %s, late %s (items %s), empty "
            "%s; items seen %s, kept / sent before %s, capped %s, SENT %s (%s MB), committed %s, revived %s, no room %s; "
            "USED store %s + window %s = %s (%s MB), unused %s, pending %s; pack %s s (%s GB/s), H2D %s ms (%s GB/s); "
            "barrier waits %s, median %s ms; close wait %s s; snapshot stalls %s; captures locked %s; device %s MB; "
            "NEW blocks per steady round %s, probe gather wait median %s ms; error %s" % (
                name, bf.get("mode"), bf.get("worker"), bf.get("X"), bf.get("n_cap"), bf.get("ring"), bf.get("epochs"),
                bf.get("jobs"), bf.get("issued"), bf.get("late"), bf.get("items_late"), bf.get("empty_jobs"),
                bf.get("items_seen"), bf.get("items_have"), bf.get("items_cap"), bf.get("items_sent"),
                round((bf.get("bytes_sent") or 0) / 1e6, 1), bf.get("committed"), bf.get("revived"), bf.get("no_room"),
                bf.get("used_store"), bf.get("used_win"), used, round((bf.get("bytes_used") or 0) / 1e6, 1),
                bf.get("unused"), bf.get("pending"), round(bf.get("pack_s") or 0.0, 3), bf.get("pack_GBps"),
                bf.get("h2d_ms_total"), bf.get("h2d_GBps"), bf.get("barrier_waits"), bf.get("barrier_ms_median"),
                round(bf.get("close_wait_s") or 0.0, 3), bf.get("snap_stall"), bf.get("captures_locked"),
                round((bf.get("device_bytes") or 0) / 1e6, 1), nr.get("new_blocks_per_round"), nr.get("wait_ms_median"),
                bf.get("error")),
            "BF1 %s packer %s: transport %s" % (name, bf.get("packer"), bf.get("transport")),
            "BF1 %s ROOM: eligible slots per (head, request) at the epoch start mean %s min %s; demand per (head, request) "
            "per epoch %s; items dropped for room BEFORE packing %s. LATE: queued %s, packing %s, of the last draft pass "
            "%s; age at the close median %s ms, max %s ms; worker busy %s s" % (
                name, bf.get("room_free_mean"), bf.get("room_free_min"), bf.get("room_demand_per_hb_mean"),
                bf.get("items_room"), bf.get("late_queued"), bf.get("late_packing"), bf.get("late_last_pass"),
                bf.get("late_age_ms_median"), bf.get("late_age_ms_max"), round(bf.get("worker_busy_s") or 0.0, 3)),
            "BF1 %s LATE MODE %s (wait budget %s ms per joint pass): deferred jobs %s, committed at their layer %s, "
            "landed after their layer ran (passed) %s, host wait in the joint passes %s s, no room at the deferred commit "
            "%s items" % (name, bf.get("late_mode"), bf.get("late_wait_ms"), bf.get("deferred"),
                          bf.get("deferred_committed"), bf.get("late_passed"), round(bf.get("late_wait_s") or 0.0, 3),
                          bf.get("no_room_deferred"))]


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
    if any(x in a.arms for x in tuple(CCC.BASE) + tuple(BASE_D) + tuple(BF_ARMS)) and \
            not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        print("REFUSED: a C, D or BF1 arm needs TORCHINDUCTOR_CACHE_DIR (the step script sets it; never the home "
              "cache)", file=sys.stderr)
        return 20
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs, keep, rds, caps = [], None, {}, None, {}, {}, {}
    eqf = {}
    FORK, g0 = I3C.import_fork()
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
    dims = dict(L=L, H=H, D=D, topk=TOPK, block_bytes=64 * D * 2 * 2 + 64 * 2)
    bb = KP.block_bytes(D)
    check_t = a.check_every == 1
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps, C %d)" % (a.batch, a.steps, BC.C_BASE + a.pool))
        fp_sh = (torch.zeros((a.batch, a.steps), dtype=torch.float32, device="cuda"),
                 torch.zeros((a.batch, a.steps), dtype=torch.float64, device="cuda"))
        try:
            sh_start = torch.cuda.memory_allocated() / 1024 ** 3
            out = BC.run_shipped(model, ids, a.steps, clock, fp=fp_sh)
            pa, pr = BC.gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(steady=steady, rates=BC.rates(a.batch, a.steps, out["periods"], steady),
                      rewarms=out["rewarms"], prefill_s=out["prefill_s"], gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, device_used_end_GB=out["device_used_end_GB"],
                      memory=dict(start_GB=round(sh_start, 3), peak_GB=pa, budget_GB=BUDGET_GB))
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            gates.append(IA.mem_record("SH", "prefill + decode", pa))
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
            if FORK is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: the fork did not import (C0_FORK)"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, journal=a.spec_journal_blocks, tu_gemm=a.tu_gemm, twin=a.spec_twin,
                       body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                       probe_every=a.probe_every, check_every=a.check_every, fork_store=a.fork_store,
                       prefill_hook=a.prefill_hook, bf_store=a.bf_store,
                       bf=(bf_opt(a, arm) if arm in BF_ARMS else None))
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_bf(model, ids, inputs, a.steps, clock, opt, FORK, arm)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"), dgraph=rd.get("dgraph"),
                           hoist=rd.get("hoist"), capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw={k: v for k, v in res["runner_kw"].items() if k != "bf"}, cc=rd.get("compile"),
                           bf=rd.get("bf"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                rec["effective_wo_capture_tok_s"] = HC.effective_wo_capture(rec, rd)
                rec["effective_wo_setup_tok_s"] = CCC.effective_wo_setup(rec, rd)
                rec["store_record"] = DC.acct_summary(rd.get("keep"), bb)
                rec["new"] = new_record(rec, rd)
                sp[arm] = rec
                rds[arm] = rd
                if arm in GRAPH_ARMS:
                    caps[arm] = HC.capture_rounds_x(rd)
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(rd, K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(rd), state=res["state"], counts=GC.counts_record(rd), keep=rd.get("keep"))
                if arm == "L4PTFD":
                    gates.append(IA.mem_record(arm, res["mem"], pa))
                ag = arm_gates_bf(arm, rd, rec, cpu, keep, check_t, a.batch, a.pool, a.steps, L, pa)
                gates.extend(ag)
                if arm != "L4PTFD":
                    eqf[arm] = next((bool(x[2]) for x in ag if x[0] == "%s-EQF" % arm), None)
                keep[arm] = cpu
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    xr = HC.excluded_rounds(caps)                                      # THE L61 FIX: the same rounds for every arm
    for arm, rec in sp.items():
        HC.restrict(rec, rds[arm], sh, a.batch, a.spec, a.probe_every, xr)
    preds = verdicts(sp, a.batch, eqf.get("L4PTFDGUDKWCDB"))
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                bf_store=a.bf_store, bf=dict(ncap=a.bf_ncap, ring=a.bf_ring, worker=a.bf_worker, acct=a.bf_acct,
                                             close_wait_ms=a.bf_close_wait_ms),
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_BF1, by_batch=PRED_BF1_BY_B), capture_rounds=caps,
                steady_excluded_rounds=xr, strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB,
                compile=dict(torch=torch.__version__, inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                             kernel_cache=os.environ.get("TRITON_CACHE_DIR")))
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-24s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-22s %s: %s" % p)
    print("REPORT L61 capture rounds %s; excluded from every arm's steady set: %s" % (caps, xr))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        nr = r.get("new") or {}
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f %s; round "
              "median %s ms; phases %s; draft step %s ms; joint %s SH steps; steady rounds %s (excluded %s); NEW blocks "
              "per steady round %s, probe gather wait median %s ms; tokens equal %.6f (first %s); peak %s GB (with the "
              "prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["joint_sh_steps"],
                  r.get("steady_rounds_used"), r.get("steady_excluded_rounds"), nr.get("new_blocks_per_round"),
                  nr.get("wait_ms_median"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        if name in BF_ARMS:
            for line in bf_lines(name, r):
                print(line)
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_bf1: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
