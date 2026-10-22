"""CPU tests of the FUSED RESIDENCY RESOLVE (nosi/nosi/r2_fusedres.py; OwnResidency(resolve="fused" | "check"),
docs/R2_FUSEDRES_PLAN.md). The kernels run under the Triton interpreter (TRITON_INTERPRET=1) in a FRESH process
(triton reads the flag at its first import; this module's own process imports triton through S-C).

FR0  the modes: "torch" by default (report()["b2"]["resolve"]); an unknown mode, a plant without the fused path, and
     the fused path with an S-C / S-B / sparse negative control or a record_cap are refused.
FR1  DECISION EQUALITY, every tick and every layer: B2's lockstep schedules of test_r2_own2.py (f6: ready + in flight
     + new; f12b: capacity pressure, refusals, the R_ver reserve; f20b: Policy E, LRU kind; f21b: Policy E, in-flight
     kind) with resolve="check": at EVERY layer the fused outputs equal the torch path's (every decision field, the
     page tables, the load ids, the counters, the sparse record, the selection bitmaps, the bias rows); the B2 world
     on the fused decisions equals v1 (pages, content, state, counters, readers, mirror); err stays 0. Preconditions:
     the schedule reached its transitions (v1's counts and the fused stats).
FR2  NEGATIVE CONTROLS with an L34 precondition: plant 1 (Policy E takes the newest candidate first) and plant 2 (a
     draft grant takes the (q + 1)-th FREE slot). Per run the kernels count the entries where the plant CHANGES an
     output (stats pe_flip_effective / d_grant). A run where the plant is effective must differ from the torch path
     at EXACTLY the first layer where it became effective; a run where it never is effective must show no
     difference (reported as a no-op, never as a catch); each plant must be effective on at least one schedule.
FR3  LAUNCHES PER LAYER (f12b, its draft-only ticks and its verify ticks): fused: 3 Triton launches per draft-only
     layer (D1, SCAN, D2), 6 per layer with verify and draft rows; on a draft-only layer after the tick's first, NO
     torch op other than an
     allocation or a view (TorchDispatchMode, paused inside the launches). The torch path's compute ops per such
     layer are counted for comparison (precondition: the counter sees them).
FR4  STATIC: the per-layer path (FusedResolver._layer) has no host read (S-C's F12d list) and no host-to-device copy
     (a planted read is found); every @triton.jit function of the module resolves its names in its module (L36,
     as F23; a planted factory kernel is found).
FR5  ONE TICK ROUND (the interpreter): the free-running greedy round (r2_tick.run_round, K 3, LAMBDA 7) on b2 with
     resolve="check" == resolve="torch" (rows, argmax per tick, committed tokens, decision stats); no difference at
     any compared layer; err 0.
FR6  HAND CASES for the branches the schedules do not reach (review M5), each with its precondition asserted BEFORE
     the equality claim: (a) one verify tick at C 1 / X 1 whose first misses overflow (K5 refuses, no Policy E
     candidate), whose second verify row repeats an overflowed block (a dup behind an overflowed first miss) and
     selects one block twice in the row: v_over, v_dup_over, sel_dup > 0; the tick is void (CapacityRefusal) after
     the comparison; (b) draft-only ticks with TWO draft rows of one batch slot whose selections overlap: max rows
     per b == 2 and d_dup > 0 (the draft first-miss dedup across rows). Fused == torch at every layer, err 0.
FR7  C's SHARED FIXTURES (r2_lockstep_fixtures.py) on the B2 world with resolve="check": the B2 world equals the
     vendored reference on every fixture (B2I-1's rule) and fused == torch at every resolved layer; precondition: the
     void_overflow fixture reaches v_over > 0.
FR8  THE KERNEL-TIME PARSER (r2_fusedres_prof.py, review M3) on synthetic Kineto traces through its production entry:
     the device time of the kernels launched inside "t.res" per tick; a launch moved out of the range is not counted
     (negative control); an orphan device event is counted; a trace without "t.tick" ranges is refused.
FR9  THE SG5 FLAGS: --b2-fused-plant (hidden, 0 / 1 / 2, FR-g0p) and --b2-res-profile parse; make_engine passes the
     plant; one_point routes the draft-only phase through profile_draft_only.
FR12 [stack2] C + X ABOVE 512 (job 386 stack1a128: "C + X = 992 > 512"): (a) slot_bits gives 23b8a38's key (2^56,
     512, 511) up to NP 512 and an int64-safe key up to MAX_NPOS; (b) resolvers at C + X 992 and 1440 equal the torch
     path at every layer, with Policy E victims in slots > 511 (precondition); (c) NEGATIVE CONTROL: the 9-bit field
     (slot_bits forced to 9) differs at the first verify layer and nowhere before it.
FR13 [stack2] NO COMPILE INSIDE A DECISION (job 386 stack1a: 17 variants compiled in the timed round): with fused_warm
     'on', every launch's compile key (constexprs, num_warps, dtype / alignment / int class of the arguments not in
     do_not_specialize) was launched by warm_up first; decisions, stats and launch counts equal the warm 'off' run.
     NEGATIVE CONTROL: under 23b8a38's rule (ints-only do_not_specialize, LZ and SCAN BLOCK per tick) some real key
     is new.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_fusedres.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOSI = ROOT / "nosi/nosi"


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


OWNT = _load_by_path("test_r2_own2_fr", HERE / "test_r2_own2.py")          # B2's lockstep and its worlds
TRM, OWN2 = OWNT.TRM, OWNT.OWN2
TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")                # the tick's CPU world (FR5)
p3 = TT.p3                                                                   # pytest fixture, re-exported
SCHED = sorted(OWNT.SCHEDULES)


def _has_triton():
    return importlib.util.find_spec("triton") is not None


# =====================================================================================================================
# the child process (TRITON_INTERPRET=1 from its start)
# =====================================================================================================================
FR_CHILD = r'''
import importlib.util, json, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "FR child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "FR child: triton was imported before the interpreter flag was read"
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
HERE, PART = sys.argv[1], sys.argv[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


OUT = {}
if PART == "lockstep":
    OWN = load("test_r2_own2_frchild", os.path.join(HERE, "test_r2_own2.py"))
    TRM, OWN2 = OWN.TRM, OWN.OWN2
    FR = TRM._load("r2_fusedres")
    assert FR.triton is not None, "FR child: triton did not import"

    def run(name, mode="check", plant=0):
        kw = OWN.SCHEDULES[name]()
        svc_kw = dict(resolve=mode)
        if plant:
            svc_kw["fused_plant"] = plant
        bad, wv, wb = OWN._lockstep_b2(svc_kw=svc_kw, **kw)
        fr = wb.svc.fusedres
        rep = fr.report()
        m, c = wv.svc.metrics, wv.svc.metrics.counts
        return dict(sched=name, mode=mode, plant=plant, n_bad=len(bad), bad=[repr(x)[:300] for x in bad[:3]],
                    ticks=len(kw["sched"]), L=TRM.L, compared=fr.compared, layers=fr.layers, launches=fr.launches,
                    n_diffs=len(fr.diffs), diffs=[list(d) for d in fr.diffs[:8]], stats=rep["stats"], err=rep["err"],
                    bias_casts=rep["bias_casts"], sel_conversions=rep["sel_conversions"],
                    max_rows_per_b=rep["max_rows_per_b"], gate=rep["gate"], first_diff_at=rep["first_diff_at"],
                    first_effective=rep["first_effective"],
                    trace=[[t, l, st, e] for (t, l, st, e) in fr.trace],
                    v1=dict(evictions=c.get("verifier_evictions", 0),
                            inflight_evictions=c.get("verifier_inflight_evictions", 0), refused=m.refused,
                            new=m.k43["new"], in_flight=m.k43["in_flight"], ready=m.k43["ready"]),
                    b2_applied=wb.svc.b2_applied)

    OUT["equal"] = {name: run(name) for name in sorted(OWN.SCHEDULES)}
    OUT["plants"] = {"%d-%s" % (p, name): run(name, plant=p) for p, names in ((1, ("f20b", "f21b")),
                                                                              (2, ("f6", "f12b"))) for name in names}

    # FR3: launches and torch ops per resolve (the counter is paused inside the launches)
    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    LAUNCH = [0]
    orig_launch = FR._launch

    def counting_launch(kernel, grid, *a, **k):
        LAUNCH[0] += 1
        with _disable_current_modes():
            orig_launch(kernel, grid, *a, **k)
    FR._launch = counting_launch

    def counted(cls, attr, get_svc, recs):
        orig = getattr(cls, attr)

        def resolve(self, l, rows, sel):
            svc = get_svc(self)
            first = not svc._tick
            mixed = any(r.kind == "v" for r in rows)
            l0 = LAUNCH[0]
            cm = Count()
            with cm:
                out = orig(self, l, rows, sel)
            recs.append(dict(tick=svc.tick_no, l=l, first=first, mixed=mixed, launches=LAUNCH[0] - l0, ops=cm.ops))
            return out
        setattr(cls, attr, resolve)
        return orig

    OUT["count"] = {}
    for mode in ("fused", "torch"):
        recs = []
        if mode == "fused":
            cls, attr, get = FR.FusedResolver, "resolve", (lambda s: s.svc)
        else:
            cls, attr, get = OWN2.OwnResidency, "resolve", (lambda s: s)
        orig = counted(cls, attr, get, recs)
        try:
            kw = OWN.SCHEDULES["f12b"]()
            bad, wv, wb = OWN._lockstep_b2(svc_kw=dict(resolve=mode), **kw)
        finally:
            setattr(cls, attr, orig)
        OUT["count"][mode] = dict(n_bad=len(bad), bad=[repr(x)[:300] for x in bad[:3]], recs=recs)
    FR._launch = orig_launch

    # FR6: hand cases (review M5)
    Row, S_, Ev, L = TRM.Row, TRM.S, TRM.Ev, TRM.L

    def hand_overflow():
        w = TRM.world(C=1, X=1, R_ver=1, cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="check"))
        svc = w.svc
        svc.begin_request(0, "u0")
        svc.begin_request(1, "u1")
        hb, tail = 15, svc.tail0
        key = ("u0", 1, 1)
        rows = [Row(0, "v", key, 1000, hb, [(hb, tail)]), Row(0, "v", key, 1000, hb, [(hb, tail)])]
        # row 0: blocks 3 (a local take, C 1), 4 (the layer's one shared extra at head 0), 5 (refused, no candidate:
        # an OVERFLOW); row 1: block 5 twice (an in-row duplicate) behind its overflowed first miss, and block 6
        sel = S_([[3, 4, 5, hb], [3, 4, 5, hb]], [[5, 5, 6, hb], [5, 5, 6, hb]])
        raised = None
        try:
            svc.poll()
            for l in range(L):
                plan = svc.resolve(l, rows, sel)
                svc.read_done(l, plan, Ev(True))
            svc.end_tick()
        except Exception as e:                                        # noqa: BLE001 - reported by class name
            raised = type(e).__name__
        fr = svc.fusedres
        rep = fr.report()
        return dict(raised=raised, compared=fr.compared, layers=fr.layers, n_diffs=len(fr.diffs),
                    diffs=[list(d) for d in fr.diffs[:8]], stats=rep["stats"], err=rep["err"], L=L)

    def hand_two_drafts(mode):
        """Ticks with two draft rows of batch slot 0 whose selections overlap (the draft first-miss dedup across
        rows), up to B2's first refusal. Production never builds it (r2_tick.py RoundState.finish_tick refuses "two draft rows"), and B2's
        consumer refuses its draft merges into a take of the same tick (d_merge): so the claim is decision equality
        at both layers, and the SAME refusal at end_tick in torch and in check mode."""
        # job 2191261: tick 0 alone does not refuse (its dups are merged into same-tick takes without a d_merge
        # violation); 2191259's six-tick world (landing after each tick) did refuse with d_merge. So the same six
        # ticks run in both modes and stop at the FIRST raise: the claim is the same tick and the same text.
        w = TRM.world(C=2, X=4, R_ver=2, cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve=mode))
        svc = w.svc
        svc.begin_request(0, "u0")
        svc.begin_request(1, "u1")
        hb, tail = 15, svc.tail0
        raised = raised_tick = None
        for t in range(6):
            x = t % 5
            rows = [Row(0, "d", ("u0", 1, 1), 1000 + t, hb, [(hb, tail)]),
                    Row(0, "d", ("u0", 1, 2), 1000 + t, hb, [(hb, tail)]),
                    Row(1, "d", ("u1", 1, 1), 1000 + t, hb, [(hb, tail)])]
            sel = S_([[x, x + 1, x + 2, hb], [x, x + 1, x + 2, hb]],
                     [[x + 1, x + 2, x + 3, hb], [x + 1, x + 2, x + 3, hb]],
                     [[x + 2, x + 4, hb], [x + 2, x + 4, hb]])
            try:
                svc.poll()
                for l in range(L):
                    plan = svc.resolve(l, rows, sel)
                    svc.read_done(l, plan, Ev(True))
                svc.end_tick()
                TRM.land_all(w)
            except Exception as e:                                    # noqa: BLE001
                raised, raised_tick = "%s: %s" % (type(e).__name__, str(e)[:160]), t
                break
        fr = svc.fusedres
        if fr is None:
            return dict(raised=raised, raised_tick=raised_tick)
        rep = fr.report()
        return dict(raised=raised, raised_tick=raised_tick, L=L, compared=fr.compared, layers=fr.layers,
                    n_diffs=len(fr.diffs), diffs=[list(d) for d in fr.diffs[:8]], stats=rep["stats"], err=rep["err"],
                    max_rows_per_b=rep["max_rows_per_b"])

    OUT["hand"] = dict(overflow=hand_overflow(), two_drafts=hand_two_drafts("check"),
                       two_drafts_torch=hand_two_drafts("torch"))

    # FR7: C's shared fixtures on the B2 world in check mode
    FXM = TRM.FXM
    fx_out = {}
    for name in sorted(FXM.FIXTURES):
        fx = FXM.FIXTURES[name]
        ref = fx.worlds[0]
        worlds = {ref: TRM._fx_world(fx, ref)}
        worlds["b2"] = OWN._fx_b2(fx, svc_kw=dict(resolve="check"))
        try:
            res = FXM.run(fx, worlds, TRM._fx_snapshot, Row)
            rdiffs, rerr = [repr(d)[:300] for d in res.diffs[:2]], None
        except Exception as e:                                        # noqa: BLE001
            rdiffs, rerr = [], "%s: %s" % (type(e).__name__, str(e)[:200])
        fr = worlds["b2"].svc.fusedres
        rep = fr.report()
        fx_out[name] = dict(res_diffs=rdiffs, run_error=rerr, compared=fr.compared, layers=fr.layers,
                            n_diffs=len(fr.diffs), diffs=[list(d) for d in fr.diffs[:5]], stats=rep["stats"])
    OUT["fixtures"] = fx_out

elif PART == "guard":
    # FR10 (fusedres2b, job 2191246): every kernel-written buffer has a sentinel guard after its end, checked after
    # EVERY launch; a separate process so a crash elsewhere cannot hide its result
    OWN = load("test_r2_own2_frguard", os.path.join(HERE, "test_r2_own2.py"))
    TRM, OWN2 = OWN.TRM, OWN.OWN2
    FR = TRM._load("r2_fusedres")
    res = {}
    for name in ("f20b", "f21b", "f6", "f12b"):
        kw = OWN.SCHEDULES[name]()
        bad, wv, wb = OWN._lockstep_b2(svc_kw=dict(resolve="check", fused_guard=True), **kw)
        fr = wb.svc.fusedres
        res[name] = dict(n_bad=len(bad), bad=[repr(x)[:300] for x in bad[:2]], hits=[list(x) for x in fr.guard_hits],
                         checks=fr.guard_checks, n_diffs=len(fr.diffs), layers=fr.layers,
                         guarded=len(fr._g_persist) + len(fr._g_tick) + len(fr._g_layer))
    # NEGATIVE CONTROL: a changed guard element is found and named (the detector's own precondition)
    name, full, n = fr._g_persist[-1]
    full[n + 5] = 7
    before = len(fr.guard_hits)
    fr._check_guards((99, 0), "control")
    res["control"] = dict(buffer=name, new_hits=[list(x) for x in fr.guard_hits[before:]])
    OUT["guard"] = res

elif PART == "tick":
    P3PATH = sys.argv[3]
    TT = load("test_r2_tick", os.path.join(HERE, "test_r2_tick.py"))
    P3 = load("piece3_v2_fr5", P3PATH)
    TK = TT.TK
    FR = TT._load("r2_fusedres")
    assert FR.triton is not None, "FR5 child: triton did not import"
    ops = TT.make_ops(P3)

    def run(mode):
        eng = TT.make_res_engine(ops, kind="b2", C=12, X=24, R_ver=4, svc_kw=dict(resolve=mode))
        rec = []

        def on_tick(rows, tokens, out):
            rec.append(([(r.b, r.kind, r.pos) for r in rows], out.argmax.tolist()))
        rnd = TK.run_round(eng, TT.FIRST, [TT.STEPS] * eng.B, K=3, LAMBDA=7, on_tick=on_tick)
        st = {k: eng.stats[k] for k in ("ticks", "rows", "decision_reads", "rollbacks", "publications", "void_ticks")}
        svc = eng.res_world.svc
        fr = svc.fusedres.report() if svc.fusedres is not None else None
        diffs = [list(d) for d in svc.fusedres.diffs[:8]] if svc.fusedres is not None else []
        return dict(rec=rec, out=[list(r.out) for r in rnd.reqs], stats=st, fused=fr, diffs=diffs,
                    mirror=eng.mirror_failures())
    rt = run("torch")
    rc = run("check")
    OUT["tick"] = dict(torch_stats=rt["stats"], check_stats=rc["stats"], same_rec=rt["rec"] == rc["rec"],
                       same_out=rt["out"] == rc["out"], same_stats=rt["stats"] == rc["stats"], fused=rc["fused"],
                       diffs=rc["diffs"], mirror=[rt["mirror"], rc["mirror"]],
                       first_rec_diff=next((i for i, (a, b) in enumerate(zip(rt["rec"], rc["rec"])) if a != b), None))

elif PART == "wide":
    # FR12 [stack2]: C + X above 512 (stack1a128 992 = C 96 + X 896, stack1b 1440 = C 96 + X 1344). Phase A: draft rows
    # fill C slots per stream (slots 0 .. C - 1, LRU at C exactly: no K1 eviction). Phase B: draft rows read blocks
    # 0 .. 511 again (touch: they go to the LRU's recent end). The cohort is decided. Phase C: one verify tick with 3
    # new blocks per head at usage C: K5 grants X of them (the 2 FREE slots), the rest are refused, and Policy E evicts
    # the least recent LRU candidates, which are in slots 512 .. C - 1. Fused == torch at every layer (check mode).
    # Argument "old": slot_bits forced to 9 (23b8a38's key layout and mask) = the NEGATIVE CONTROL.
    OWN = load("test_r2_own2_frwide", os.path.join(HERE, "test_r2_own2.py"))
    TRM, OWN2 = OWN.TRM, OWN.OWN2
    FR = TRM._load("r2_fusedres")
    assert FR.triton is not None, "FR12 child: triton did not import"
    Row, S_, Ev, L = TRM.Row, TRM.S, TRM.Ev, TRM.L
    WIN = 127                                                   # + the tail entry: k 128 (KP 128)
    TOUCH = 512                                                 # blocks 0 .. 511: slots 0 .. 511 after phase A

    def wide(C, X, old=False):
        hb = C + 16                                             # history bound: the new blocks C, C + 1, C + 2 < hb
        TRM.NB = hb + 8                                         # world() sizes the host rows by the module's NB
        if old:
            FR.slot_bits = lambda np_: 9                        # NEGATIVE CONTROL: the 9-bit slot field at NP > 512
        w = TRM.world(C=C, X=X, R_ver=2, cls=OWN2.OwnResidency,       # warm_up on (FR13 at the wide NP too)
                      svc_kw=dict(nblk=TRM.NB, resolve="check", fused_warm="off" if old else "on"))
        svc = w.svc
        fr = svc.fusedres
        uids = ("u0", "u1")
        for b, u in enumerate(uids):
            svc.begin_request(b, u)
        tail = svc.tail0
        ticks = []                                              # (phase, the draft / verify block list)
        for x in range(0, C, WIN):
            ticks.append(("A", list(range(x, min(x + WIN, C)))))
        for x in range(0, TOUCH, WIN):
            ticks.append(("B", list(range(x, min(x + WIN, TOUCH)))))
        ticks.append(("C", [C, C + 1, C + 2]))
        raised = raised_tick = None
        ev_slots, take_slots, phase_c_tick = [], [], None
        for t, (ph, blocks) in enumerate(ticks):
            if ph == "C":
                phase_c_tick = t
                for u in uids:                                  # the draft cohort is decided before the verify tick
                    svc.decided((u, 1, 1))
            kind, key_m = ("v", 2) if ph == "C" else ("d", 1)
            rows = [Row(b, kind, (u, 1, key_m), 1000 + t, hb, [(hb, tail)]) for b, u in enumerate(uids)]
            sel = S_(*[[blocks + [hb], blocks + [hb]] for _ in uids])
            try:
                svc.poll()
                for l in range(L):
                    plan = svc.resolve(l, rows, sel)
                    if ph == "C":                               # the fused decision (== torch's when no diff)
                        vd = svc._tick[-1]["vdec"]
                        ev = vd.t["evict"] != 0
                        ev_slots += vd.t["take_slot"][ev].tolist()
                        take_slots += vd.t["take_slot"][(vd.t["granted"] != 0)].tolist()
                    svc.read_done(l, plan, Ev(True))
                svc.end_tick()
                TRM.land_all(w)
            except Exception as e:                              # noqa: BLE001 - reported
                raised, raised_tick = "%s: %s" % (type(e).__name__, str(e)[:200]), t
                break
        rep = fr.report()
        return dict(C=C, X=X, old=old, n_hist=svc.n_hist, NP=max(FR._pow2(svc.n_hist), 64),
                    sb=FR.slot_bits(max(FR._pow2(svc.n_hist), 64)), ticks=len(ticks), phase_c_tick=phase_c_tick,
                    raised=raised, raised_tick=raised_tick, L=L, compared=fr.compared, layers=fr.layers,
                    n_diffs=len(fr.diffs), diffs=[list(d) for d in fr.diffs[:8]], stats=rep["stats"],
                    err=rep["err"], ev_slots=ev_slots, take_slots=take_slots, b2_applied=svc.b2_applied,
                    warm_launches=rep["warm_launches"], warmed=rep["warmed"])
    which = sys.argv[3]
    if which == "old":
        OUT["wide"] = wide(990, 2, old=True)
    else:
        n = int(which)
        OUT["wide"] = wide(n - 2, 2)
elif PART == "warm":
    # FR13 [stack2]: every launch's compile key (Triton's: constexprs + num_warps, and for each argument NOT in
    # do_not_specialize: a pointer's dtype and 16-byte alignment, an int's == 1 and % 16) is one that warm_up
    # launched before it. The old rule (23b8a38: DNS_OLD, LZ = pow2(Rv * 4), BLOCK = pow2(max(R H, S))) is the
    # NEGATIVE CONTROL: some real key is new under it.
    import inspect
    OWN = load("test_r2_own2_frwarm", os.path.join(HERE, "test_r2_own2.py"))
    TRM, OWN2 = OWN.TRM, OWN.OWN2
    FR = TRM._load("r2_fusedres")
    assert FR.triton is not None, "FR13 child: triton did not import"

    def p2(n):
        return FR._pow2(int(n))

    def ckey(kernel, args, kw, dns, old):
        fn = getattr(kernel, "fn", kernel)
        name = fn.__name__
        params = list(inspect.signature(fn).parameters)
        skip = set(dns[name])
        spec = []
        for p, v in zip(params, args):
            if p in skip:
                continue
            if isinstance(v, torch.Tensor):
                spec.append((p, str(v.dtype), v.data_ptr() % 16 == 0))
            elif isinstance(v, bool):
                spec.append((p, v))
            elif isinstance(v, int):
                spec.append((p, v == 1, v % 16 == 0))
            else:
                spec.append((p, repr(v)))
        cx = dict(kw)
        if old and name == "_fr_v1":
            cx["LZ"] = min(256, max(p2(args[params.index("NLOOK")]), 2))
        if old and name == "_fr_scan":
            cx["BLOCK"] = min(FR.SCAN_BLOCK, max(p2(max(args[params.index("RH")], args[params.index("S")])), 2))
        return [name, sorted((k, repr(v)) for k, v in cx.items()), spec]

    LOG = []
    orig_launch = FR._launch

    def keyed(kernel, grid, *a, **k):
        LOG.append((FR._warm_depth[0] > 0, json.dumps(ckey(kernel, a, k, FR.DNS, False)),
                    json.dumps(ckey(kernel, a, k, FR.DNS_OLD, True))))
        return orig_launch(kernel, grid, *a, **k)
    FR._launch = keyed

    def run(name, warm):
        del LOG[:]
        kw = OWN.SCHEDULES[name]()
        bad, wv, wb = OWN._lockstep_b2(svc_kw=dict(resolve="check", fused_warm=warm), **kw)
        fr = wb.svc.fusedres
        rep = fr.report()
        warm_new = {k for (w, k, _) in LOG if w}
        warm_old = {o for (w, _, o) in LOG if w}
        real = [(k, o) for (w, k, o) in LOG if not w]
        miss_new = sorted({k for (k, _) in real if k not in warm_new})
        miss_old = sorted({o for (_, o) in real if o not in warm_old})
        return dict(n_bad=len(bad), bad=[repr(x)[:300] for x in bad[:3]], n_diffs=len(fr.diffs),
                    diffs=[list(d) for d in fr.diffs[:5]], err=rep["err"], compared=fr.compared, layers=fr.layers,
                    warm=rep["warm"], warmed=rep["warmed"], warm_launches=rep["warm_launches"],
                    launches=rep["launches"], real=len(real), n_warm_keys=len(warm_new),
                    miss_new=[json.loads(m)[0] for m in miss_new], miss_old=[json.loads(m)[0] for m in miss_old],
                    trace=[[t, l, st, e] for (t, l, st, e) in fr.trace], b2_applied=wb.svc.b2_applied)
    OUT["warm"] = {"%s-%s" % (n, wm): run(n, wm) for n in ("f12b", "f20b") for wm in ("on", "off")}
    FR._launch = orig_launch
print("FR_JSON " + json.dumps(OUT))
'''


def _child(part, *extra):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", FR_CHILD, str(HERE), part] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=5400)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("FR_JSON ")]
    assert r.returncode == 0 and last, ("FR child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("FR_JSON "):])


@pytest.fixture(scope="module")
def lockstep():
    if not _has_triton():
        pytest.skip("triton is not installed")
    return _child("lockstep")


# =====================================================================================================================
# FR0 the modes
# =====================================================================================================================
def test_FR0_the_default_is_torch_and_bad_modes_are_refused():
    w = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB))
    assert w.svc.resolve_mode == "torch" and w.svc.fusedres is None
    assert w.svc.report()["b2"]["resolve"] == "torch"
    with pytest.raises(ValueError, match="resolve 'bogus'"):
        TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="bogus"))
    with pytest.raises(ValueError, match="fused_plant needs"):
        TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, fused_plant=1))
    if not _has_triton():
        pytest.skip("triton is not installed (the fused constructor checks)")
    for bad in (dict(sc_mut={"no_overflow_flag": True}), dict(sb_mut={"no_dedup": True}),
                dict(sparse_mut={"rowsel_by_entry": True}), dict(record_cap=(8, 8)), dict(fused_plant=3)):
        with pytest.raises(ValueError):
            TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="fused", **bad))
    with pytest.raises(ValueError, match="_layer_record"):
        TRM.world(cls=OWN2.TrashWriterB2, svc_kw=dict(nblk=TRM.NB, resolve="fused"))
    f = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="check"))
    assert f.svc.resolve_mode == "check" and f.svc.fusedres.check and "fused_res_err" in f.svc._flags


# =====================================================================================================================
# FR1 decision equality, every tick and layer
# =====================================================================================================================
PRE = dict(f6=lambda r: r["v1"]["ready"] > 0 and r["v1"]["in_flight"] > 0 and r["v1"]["new"] > 0
           and r["stats"]["v_grant"] > 0 and r["stats"]["d_grant"] > 0,
           f12b=lambda r: r["v1"]["refused"] > 0 and r["stats"]["d_refused"] > 0,
           f20b=lambda r: r["v1"]["evictions"] > 0 and r["stats"]["v_evict"] > 0,
           f21b=lambda r: r["v1"]["inflight_evictions"] > 0 and r["stats"]["v_evict_infl"] > 0)


@pytest.mark.parametrize("sched", SCHED)
def test_FR1_fused_equals_torch_at_every_layer_and_b2_equals_v1(lockstep, sched):
    r = lockstep["equal"][sched]
    assert r["n_bad"] == 0, r["bad"]                                   # the B2 world on fused decisions == v1
    assert r["n_diffs"] == 0, r["diffs"]                               # fused == torch, every output
    assert r["compared"] == r["layers"] == r["ticks"] * r["L"], (r["compared"], r["layers"], r["ticks"], r["L"])
    assert r["err"] == 0, r["err"]
    assert r["bias_casts"] == 0 and r["sel_conversions"] == 0, (r["bias_casts"], r["sel_conversions"])
    assert r["b2_applied"] == r["ticks"]
    assert PRE[sched](r), (sched, r["v1"], r["stats"])                 # the schedule reached its transitions


def test_FR1_the_lockstep_schedules_reach_the_verify_dedup(lockstep):
    """Review M5: the verify first-miss dedup across rows (v_dup) ran on at least one lockstep schedule (two verify
    rows of one cohort per request every third tick). The overflow branches are FR6's (a lockstep run voids)."""
    dups = {s: lockstep["equal"][s]["stats"]["v_dup"] for s in SCHED}
    assert any(v > 0 for v in dups.values()), dups
    assert all(lockstep["equal"][s]["stats"]["v_over"] == 0 for s in SCHED)   # a lockstep tick never overflows


# =====================================================================================================================
# FR2 negative controls (L34: effective -> caught at that layer; not effective -> a no-op, not a catch)
# =====================================================================================================================
EFFECT = {1: "pe_flip_effective", 2: "d_grant"}


@pytest.mark.parametrize("plant", [1, 2])
def test_FR2_each_plant_is_caught_at_its_first_effective_layer(lockstep, plant):
    runs = {k: v for k, v in lockstep["plants"].items() if v["plant"] == plant}
    assert runs, lockstep["plants"].keys()
    effective = []
    for key, r in sorted(runs.items()):
        stat = EFFECT[plant]
        first_eff = next(((t, l) for (t, l, st, e) in r["trace"] if dict(zip(list(r["stats"]), st))[stat] > 0),
                         None)
        if first_eff is None:                                          # never effective: a NO-OP, must not differ
            assert r["n_diffs"] == 0, (key, "a plant without effect changed an output", r["diffs"])
            print("FR2 plant %d on %s: NO-OP (%s = 0), not counted" % (plant, r["sched"], stat))
            continue
        effective.append(key)
        assert r["n_diffs"] > 0, (key, "effective at %s but not caught" % (first_eff,))
        assert tuple(r["diffs"][0][:2]) == first_eff, (key, r["diffs"][0], first_eff)
        # the report's own scoring fields (FR-g0p reads exactly these on the GPU, M2)
        assert r["first_diff_at"] == list(first_eff) == r["first_effective"][stat], (key, r["first_diff_at"],
                                                                                    r["first_effective"])
        print("FR2 plant %d on %s: effective first at tick %d layer %d, caught there (%s)"
              % (plant, r["sched"], first_eff[0], first_eff[1], r["diffs"][0][2]))
    assert effective, ("precondition (L34): plant %d was effective on no schedule" % plant,
                       {k: v["stats"] for k, v in runs.items()})


def test_FR2_the_stats_trace_is_cumulative_and_matches_the_report(lockstep):
    """The precondition counters are the kernels' own: per run the trace is non-decreasing, its last entry is the
    report, and a clean run's err stays 0 at every layer."""
    for key, r in list(lockstep["equal"].items()) + list(lockstep["plants"].items()):
        tr = [st for (_, _, st, _) in r["trace"]]
        assert tr, key
        assert all(a <= b for x, y in zip(tr, tr[1:]) for a, b in zip(x, y)), key
        assert tr[-1] == [r["stats"][s] for s in r["stats"]], key
    for key, r in lockstep["equal"].items():
        assert all(e == 0 for (_, _, _, e) in r["trace"]), key


# =====================================================================================================================
# FR3 launches per layer
# =====================================================================================================================
VIEW_OR_ALLOC = {"empty", "empty_strided", "select", "slice", "view", "alias", "as_strided", "detach",
                 "_reshape_alias", "unsqueeze", "expand", "reshape", "t", "permute", "transpose", "lift_fresh",
                 "_unsafe_view"}


def test_FR3_three_launches_per_draft_only_layer_six_with_verify_rows_and_no_torch_compute(lockstep):
    c = lockstep["count"]
    assert c["fused"]["n_bad"] == 0 and c["torch"]["n_bad"] == 0, (c["fused"]["bad"], c["torch"]["bad"])
    fz, tz = c["fused"]["recs"], c["torch"]["recs"]
    assert len(fz) == len(tz) > 0
    d_only = [r for r in fz if not r["mixed"]]
    mixed = [r for r in fz if r["mixed"]]
    assert d_only and mixed, "precondition: f12b has draft-only and verify layers"
    assert all(r["launches"] == 3 for r in d_only), [(r["tick"], r["l"], r["launches"]) for r in d_only]
    assert all(r["launches"] == 6 for r in mixed), [(r["tick"], r["l"], r["launches"]) for r in mixed]  # f12b: every
    # layer with verify rows also has draft rows (_sched_pressure); a verify-only layer would launch 3
    later = [r for r in d_only if not r["first"]]
    assert later, "precondition: a draft-only layer after the tick's first"
    compute = sorted({op for r in later for op in r["ops"] if op not in VIEW_OR_ALLOC})
    assert compute == [], compute                                     # no torch kernel on such a layer
    t_later = [r for r in tz if not r["mixed"] and not r["first"]]
    t_compute = [sum(1 for op in r["ops"] if op not in VIEW_OR_ALLOC) for r in t_later]
    assert t_compute and min(t_compute) >= 20, t_compute               # precondition: the counter sees compute ops
    print("FR3 per draft-only layer (not the tick's first): fused %d Triton launches, %d torch compute ops; torch "
          "path %.1f compute ops (min %d, max %d) over %d layers; per verify layer fused %d launches"
          % (3, 0, sum(t_compute) / len(t_compute), min(t_compute), max(t_compute), len(t_compute), 6))


# =====================================================================================================================
# FR4 static: no host read in the per-layer path; the kernels' names resolve in their module (L36)
# =====================================================================================================================
def _method_source(path, cls, name):
    src = path.read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == name)
    return ast.get_source_segment(src, fn)


HOST_COPY = ("_h2d(", "torch.tensor(", ".pin_memory(", ".to(\"cpu\")", ".to('cpu')")


def _host_reads(seg):
    return [t for t in TRM.FORBIDDEN + HOST_COPY if t in seg]


def test_FR4_the_per_layer_path_has_no_host_read_and_a_planted_one_is_found():
    seg = _method_source(NOSI / "r2_fusedres.py", "FusedResolver", "_layer")
    for k in ("_fr_v1", "_fr_scan.v", "_fr_v2", "_fr_d1", "_fr_scan.d", "_fr_d2"):   # precondition: the right
        assert 'self._go(where, "%s"' % k in seg, k                              # function (launches via _go)
    assert _host_reads(seg) == [], _host_reads(seg)
    planted = seg.replace("        self.layers += 1", "        self.layers += 1\n        n = int(npg.sum())")
    assert planted != seg and "int(" in _host_reads(planted)            # NEGATIVE CONTROL


def test_FR4_every_triton_function_resolves_its_names_in_its_module():
    if not _has_triton():
        pytest.skip("triton is not installed")
    from triton.runtime.jit import JITFunction
    FR = TRM._load("r2_fusedres")
    fns = {n: v for n, v in vars(FR).items() if isinstance(v, JITFunction)}
    need = {"_fr_v1", "_fr_v2", "_fr_d1", "_fr_d2", "_fr_scan", "_row", "_sorted_unique", "_lookup", "_seen", "_pack", "_put1",
            "_slots", "_d_state", "_at_slot", "_selbits"}
    assert need <= set(fns), sorted(need - set(fns))
    bad = {n: TRM._kernel_name_problems(v) for n, v in fns.items() if TRM._kernel_name_problems(v)}
    assert bad == {}, bad
    import triton

    def factory():                                                          # NEGATIVE CONTROL
        import triton.language as tl_local

        @triton.jit
        def planted(x_ptr, BLOCK: tl_local.constexpr):
            offs = tl_local.arange(0, BLOCK)
            tl_local.store(x_ptr + offs, offs)
        return planted
    assert TRM._kernel_name_problems(factory()) == ["tl_local"]


# =====================================================================================================================
# FR5 one tick round
# =====================================================================================================================
def test_FR5_a_free_running_round_on_b2_check_equals_torch(p3):
    if not _has_triton():
        pytest.skip("triton is not installed")
    r = _child("tick", p3.__file__)["tick"]
    assert r["same_rec"] and r["same_out"] and r["same_stats"], (r["first_rec_diff"], r["torch_stats"],
                                                                  r["check_stats"])
    assert r["mirror"] == [[], []], r["mirror"]
    f = r["fused"]
    assert f is not None and f["mode"] == "check" and f["compared"] > 0 and f["compared"] == f["layers"], f
    assert f["diffs"] == 0 and r["diffs"] == [], r["diffs"]
    assert f["err"] == 0, f
    assert r["torch_stats"]["ticks"] > 20 and r["torch_stats"]["publications"] > 0, r["torch_stats"]
    print("FR5 OK: %d ticks, %d fused layers compared, launches per layer %.2f, stats %s"
          % (r["check_stats"]["ticks"], f["compared"], f["launches_per_layer"], f["stats"]))


# =====================================================================================================================
# FR6 hand cases (review M5): preconditions first, then the equality claim
# =====================================================================================================================
def test_FR6a_overflow_dup_behind_overflow_and_an_in_row_duplicate(lockstep):
    r = lockstep["hand"]["overflow"]
    s = r["stats"]
    assert s["v_over"] > 0 and s["v_dup_over"] > 0 and s["sel_dup"] > 0, ("precondition", s)
    assert r["compared"] == r["layers"] == r["L"], r                    # every layer compared before the void
    assert r["n_diffs"] == 0, r["diffs"]
    assert r["raised"] == "CapacityRefusal", r["raised"]               # the overflow voids the tick (S-C's rule)


def test_FR6b_two_draft_rows_of_one_batch_slot(lockstep):
    """Decision equality on a layer with two draft rows of one batch slot (the D-dup branch), then the same B2
    refusal as the torch path. PROOF that production never builds such a tick: RoundState.finish_tick refuses it
    (pinned below); B2's apply refuses its same-tick draft merges in both modes (d_merge)."""
    r, t = lockstep["hand"]["two_drafts"], lockstep["hand"]["two_drafts_torch"]
    assert r["max_rows_per_b"][1] == 2 and r["stats"]["d_dup"] > 0, ("precondition", r["max_rows_per_b"],
                                                                      r["stats"])
    assert r["raised"] is not None, "the six-tick world must reach B2's refusal (2191259 saw d_merge)"
    assert (r["raised_tick"], r["raised"]) == (t["raised_tick"], t["raised"]), (r, t)   # same tick, same text
    assert "d_merge" in r["raised"], r["raised"]
    assert r["compared"] == r["layers"] == (r["raised_tick"] + 1) * r["L"], r   # every resolved layer compared
    assert r["n_diffs"] == 0, r["diffs"]                               # fused == torch at every layer
    assert r["err"] == 0, r["err"]
    src = (NOSI / "r2_tick.py").read_text()
    assert 'raise RuntimeError("tick: request %d has two draft rows" % row.b)' in src   # the production refusal


# =====================================================================================================================
# FR7 C's shared fixtures in check mode
# =====================================================================================================================
def test_FR7_the_shared_fixtures_in_check_mode(lockstep):
    fx = lockstep["fixtures"]
    assert fx, "no fixture ran"
    bad = {n: (v["run_error"], v["res_diffs"][:1]) for n, v in fx.items() if v["run_error"] or v["res_diffs"]}
    assert bad == {}, bad                                               # the B2 world (fused) == the reference
    fdiff = {n: v["diffs"][:2] for n, v in fx.items() if v["n_diffs"]}
    assert fdiff == {}, fdiff                                           # fused == torch at every resolved layer
    assert all(v["compared"] == v["layers"] for v in fx.values()), {n: (v["compared"], v["layers"])
                                                                    for n, v in fx.items()}
    assert "void_overflow" in fx and fx["void_overflow"]["stats"]["v_over"] > 0, fx.get("void_overflow")
    assert sum(v["compared"] for v in fx.values()) > 0


# =====================================================================================================================
# FR10 the out-of-bounds detector (fusedres2b): sentinel guards after every kernel-written buffer, every launch
# =====================================================================================================================
def test_FR10_no_kernel_writes_past_a_buffer_and_the_guard_detector_works():
    if not _has_triton():
        pytest.skip("triton is not installed")
    g = _child("guard")["guard"]
    c = g.pop("control")
    assert c["new_hits"] == [[99, 0, "control", c["buffer"]]], c        # NEGATIVE CONTROL: a changed guard is named
    for name, r in g.items():
        assert r["checks"] > 0 and r["guarded"] > 0, (name, "precondition: guards were checked", r)
        assert r["hits"] == [], (name, "a kernel wrote past a buffer (tick, layer, kernel, buffer)", r["hits"][:5])
        assert r["n_bad"] == 0 and r["n_diffs"] == 0, (name, r["bad"])


# =====================================================================================================================
# FR8 the kernel-time parser (r2_fusedres_prof.py)
# =====================================================================================================================
FP = _load_by_path("r2_fusedres_prof_fr", ROOT / "benchmarks/Efficiency/r2_fusedres_prof.py")


def _ann(name, ts, dur, tid=1):
    return dict(ph="X", cat="user_annotation", name=name, ts=ts, dur=dur, pid=7, tid=tid)


def _launch_ev(ts, corr, tid=1):
    return dict(ph="X", cat="cuda_driver", name="cuLaunchKernel", ts=ts, dur=1, pid=7, tid=tid,
                args=dict(correlation=corr))


def _kern(name, corr, dur):
    return dict(ph="X", cat="kernel", name=name, ts=10000 + corr, dur=dur, pid=0, tid=7, args=dict(correlation=corr))


def _trace(move_out=False):
    ev = [_ann("t.tick", 0, 100), _ann("t.res", 10, 20), _ann("t.res", 50, 10),
          _ann("t.tick", 200, 100), _ann("t.res", 210, 20)]
    ev += [_launch_ev(15 if not move_out else 40, 1), _launch_ev(55, 2), _launch_ev(80, 3), _launch_ev(215, 4)]
    ev += [_kern("_fr_d1", 1, 30), _kern("_fr_d2", 2, 50), _kern("gemm", 3, 400), _kern("_fr_d1", 4, 70),
           _kern("orphan", 99, 5)]
    return dict(traceEvents=ev)


def test_FR8_the_kernel_time_parser_and_its_negative_control():
    p = FP.parse_res_trace(_trace())
    assert p["ticks"] == 2 and p["n"] == [2, 1], p
    assert [round(x, 6) for x in p["ms"]] == [0.08, 0.07], p          # 30 + 50 us; 70 us (gemm is outside t.res)
    d1 = p["by_name"]["_fr_d1"]
    assert d1["n"] == 2 and round(d1["ms"], 6) == 0.1 and "gemm" not in p["by_name"], p["by_name"]
    assert p["orphans"] == 1, p                                         # never silently dropped
    t = FP.parse_res_trace(_trace(), cat="tick")
    assert [round(x, 6) for x in t["ms"]] == [0.48, 0.07], t           # the whole tick includes gemm
    q = FP.parse_res_trace(_trace(move_out=True))                       # NEGATIVE CONTROL: the launch leaves t.res
    assert q["n"] == [1, 1] and round(q["ms"][0], 6) == 0.05, q
    s = FP.summarize(p, skip=1)
    assert s["ms_stats"]["n"] == 1 and round(s["ms_stats"]["median"], 6) == 0.07, s
    with pytest.raises(ValueError, match="t.tick"):
        FP.parse_res_trace(dict(traceEvents=[e for e in _trace()["traceEvents"] if e["name"] != "t.tick"]))


# =====================================================================================================================
# FR9 the SG5 flags
# =====================================================================================================================
def test_FR9_the_sg5_flags_for_the_plant_and_the_profile():
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x",
            "--backends", "b2"]
    a = G.parse_args(base)
    assert a.b2_resolve == "torch" and a.b2_fused_plant == 0 and a.b2_res_profile is False
    b = G.parse_args(base + ["--b2-resolve", "check", "--b2-fused-plant", "2", "--b2-res-profile"])
    assert b.b2_resolve == "check" and b.b2_fused_plant == 2 and b.b2_res_profile is True
    with pytest.raises(SystemExit):
        G.parse_args(base + ["--b2-fused-plant", "3"])
    src = (ROOT / "benchmarks/Efficiency/r2_tick_sg5.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert 'fused_plant=int(getattr(a, "b2_fused_plant", 0))' in fns["make_engine"]
    assert 'resolve=getattr(a, "b2_resolve", "torch")' in fns["make_engine"]
    assert "profile_draft_only(a, eng, first, label)" in fns["one_point"]
    assert 'record_function("t." + cat)' in fns["profile_draft_only"]
    assert "FP.parse_res_trace(" in fns["profile_draft_only"]


# =====================================================================================================================
# FR11 the FR-g step cells of r2_tick_sg5_step.sh (T19's DRYRUN pattern)
# =====================================================================================================================
FRNOSI = Path("/mnt/beegfs/ojerbi/nosi-r2-fusedres2")
P6_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-sg5-p6"
FR_CELLS = dict(              # resolve, plant, profile, (C, X, R_ver), steps, draft ticks, pool baseline, STEP_MIN
    frg0=("check", 0, False, (96, 64, 16), 48, 4, False, 120),
    frg0x=("check", 0, False, (24, 8, 4), 48, 4, False, 120),
    frg0p=("check", 2, False, (96, 64, 16), 8, 4, False, 60),
    frg1=("torch", 0, False, (96, 64, 16), 48, 12, True, 60),
    frg2=("fused", 0, False, (96, 64, 16), 48, 12, True, 60),
    frg3=("fused", 0, True, (96, 64, 16), 8, 12, False, 60),
    frg3t=("torch", 0, True, (96, 64, 16), 8, 12, False, 90))
FR_ORDER = ("frg0", "frg0x", "frg0p", "frg1", "frg2", "frg3", "frg3t")


def test_FR11_the_frg_step_cells(tmp_path):
    """The cells frg0 .. frg3t and frg of r2_tick_sg5_step.sh: static pins (FRNOSI is the root of the frg* cells only;
    the NOSI line is p6's), refusals (unknown / two cells: usage, exit 2; a sha that is not the root's HEAD: STOP,
    exit 1; p5b2 with the fusedres2 HEAD stops at p6's root: the root is not repointed), and on a checkout whose step
    script is the fusedres2 HEAD's blob, each cell's DRYRUN through the harness's own parse_args (its flags, point,
    steps, draft ticks, pool baseline, STEP_MIN, names, PYTHONPATH) and the combined cell's order. frg1 / frg2 are
    p5b2's arguments token for token plus the resolve flag."""
    script = ROOT / "benchmarks/Efficiency/r2_tick_sg5_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert "NOSI=%s " % P6_ROOT in text and "FRNOSI=%s " % FRNOSI in text
    assert text.count("NOSI=$FRNOSI") == 1 and text.index("frg*) BACKENDS=b2") < text.index("NOSI=$FRNOSI")
    assert lines[-1] == 'main "$@"; exit $?'
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "416.gpu_uuid").write_text(uid + "\n")
    e = dict(env, SG5_HOLDERS=str(tmp_path), DRYRUN="1")

    def dry(sha, *words):
        return subprocess.run(["bash", str(script), "--nosi", sha, "416", "t1", *words], env=e, capture_output=True,
                              text=True, timeout=600)
    for bad in (("frg9",), ("frg0", "frg1"), ("FRG0",), ("frg", "p5b2"), ("frg0x", "frg0x"), ("frg3", "x")):
        r = dry("0" * 40, *bad)
        assert r.returncode == 2 and "usage" in r.stderr, bad
    r = dry("0" * 40, "frg0")
    assert r.returncode == 1 and "%s HEAD is not" % FRNOSI in r.stderr, r.stderr[-300:]
    if not (FRNOSI / ".git").exists():
        pytest.skip("no %s worktree here" % FRNOSI)
    fr_head = subprocess.run(["git", "-C", str(FRNOSI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    r = dry(fr_head, "p5b2")                                            # the other cells keep p6's root
    assert r.returncode == 1 and "%s HEAD is not" % P6_ROOT in r.stderr, r.stderr[-300:]
    blob = subprocess.run(["git", "-C", str(FRNOSI), "rev-parse", "%s:benchmarks/Efficiency/r2_tick_sg5_step.sh" % fr_head],
                          capture_output=True, text=True).stdout.strip()
    mine = subprocess.run(["git", "hash-object", "--", str(script)], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(FRNOSI), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if blob != mine or dirty:
        pytest.skip("this step script is not the fusedres2 HEAD's blob, or that worktree has tracked changes")
    TTP = TT                                                            # p3's argument string (T19)
    p5b2 = (TTP.P3_SG5_ARGS.format(n="t1_416_{c}").replace(TTP.P3_SG5_SWEEP, "")
            .replace(TTP.P3_SG5_POINTS, "--modes exact,fast --backends b2").replace(" --X 32 ", " --X 64 "))
    G = TT._sg5_gate()
    for cell, (resolve, plant, prof, (C, X, R_ver), steps, dticks, pool, step_min) in FR_CELLS.items():
        r = dry(fr_head, cell)
        assert r.returncode == 0, (cell, r.stderr[-400:])
        got = r.stdout.strip().splitlines()[-1].split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
        pa = G.parse_args(got.split())
        assert pa.backends == ["b2"] and pa.modes == ["exact", "fast"] and pa.sweep_pts == [], cell
        assert (pa.C, pa.X, pa.R_ver, pa.steps, pa.draft_ticks, pa.pool_baseline) == (C, X, R_ver, steps, dticks, pool)
        assert (pa.b2_resolve, pa.b2_fused_plant, pa.b2_res_profile, pa.b2_checked) == (resolve, plant, prof, False)
        assert "nosi root %s; STEP_MIN %d; MIN_LEFT %d s" % (FRNOSI, step_min, step_min * 60 + 600) in r.stdout
        assert "--time=%d " % step_min in r.stdout and "--job-name=sg5-t1-416-%s " % cell in r.stdout
        assert "PYTHONPATH=%s/nosi" % FRNOSI in r.stdout and "%s/benchmarks/Efficiency/r2_tick_sg5.py" % FRNOSI in r.stdout
        assert "TRITON_CACHE_DIR=/mnt/beegfs/ojerbi/nosa-env/jit/sg5_t1_416_%s/triton" % cell in r.stdout
        if cell in ("frg1", "frg2"):                                    # p5b2's arguments + the resolve flag only
            assert got == p5b2.format(c=cell) + " --b2-resolve " + resolve, (cell, got)
    r = dry(fr_head, "frg")
    assert r.returncode == 0, r.stderr[-400:]
    order = [ln.split()[4] for ln in r.stdout.splitlines() if ln.startswith("R2 SG5 frg: cell ")]
    assert order == list(FR_ORDER), order
    assert "R2 SG5 frg: done; cells failed or refused 0; first MUST failure none" in r.stdout
    src = (ROOT / "benchmarks/Efficiency/r2_tick_sg5_step.sh").read_text()
    assert 'case $sub in frg1|frg3t) ctl=1 ;; *) ctl=0 ;; esac' in src            # the controls always run


# =====================================================================================================================
# FR12 [stack2] C + X above 512: Policy E's slot field (r2_fusedres.slot_bits), job 386 stack1a128's refusal
# =====================================================================================================================
def test_FR12a_the_slot_field_keeps_23b8a38s_key_up_to_512_and_never_overflows():
    """SB 9 for NP <= 512: the multipliers and the mask are 23b8a38's literals (2^56, 512, 511), so the compiled V2 is
    the same arithmetic there. Above 512 the field grows with NP and the largest legal key (class 2) stays below
    2^63 up to MAX_NPOS. A resolver above MAX_NPOS is refused."""
    if not _has_triton():
        pytest.skip("triton is not installed")
    FR = TRM._load("r2_fusedres")
    for np_ in (64, 128, 256, 512):
        sb = FR.slot_bits(np_)
        assert (1 << sb, 1 << (FR.STAMP_BITS + sb), (1 << sb) - 1) == (512, 72057594037927936, 511), np_
    assert [FR.slot_bits(n) for n in (1024, 2048, 4096)] == [10, 11, 12]
    assert FR.MAX_NPOS >= 2048 + 96                                    # bigb's probe X 2048 at C 96 fits too
    for n_hist in (992, 1440, 2048, FR.MAX_NPOS):
        np_ = max(FR._pow2(n_hist), 64)
        sb = FR.slot_bits(np_)
        assert np_ - 1 < (1 << sb), n_hist                             # every slot index fits the field
        top = 2 * (1 << (FR.STAMP_BITS + sb))                          # class 2, stamp 0, slot 0
        legal = 1 * (1 << (FR.STAMP_BITS + sb)) + ((1 << FR.STAMP_BITS) - 1) * (1 << sb) + np_ - 1
        assert legal < top < (1 << 63), n_hist                         # classes ordered, no int64 overflow
        assert np_ - n_hist <= FR.LANE_PAD, n_hist                     # a masked NP lane stays in the pad
    with pytest.raises(ValueError, match="C \\+ X = 4098 > 4096"):
        TRM.world(C=4096, X=2, cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="fused"))


@pytest.mark.parametrize("n_hist", [992, 1440])
def test_FR12b_fused_equals_torch_with_policy_e_victims_above_slot_511(n_hist):
    """A resolver at C + X = 992 (stack1a128) and 1440 (stack1b): every layer of every tick equals the torch path
    (check mode), err 0, no raise. Preconditions (L34): Policy E evicted, and every victim's slot is > 511 (the slots
    that the 9-bit field could not hold)."""
    if not _has_triton():
        pytest.skip("triton is not installed")
    r = _child("wide", str(n_hist))["wide"]
    assert r["n_hist"] == n_hist and r["NP"] == max(1024, 1 << (n_hist - 1).bit_length()) and r["sb"] > 9, r
    assert r["raised"] is None, (r["raised"], r["raised_tick"])
    assert r["compared"] == r["layers"] == r["ticks"] * r["L"], (r["compared"], r["layers"], r["ticks"])
    assert r["stats"]["v_evict"] > 0 and r["ev_slots"], ("precondition: Policy E evicted", r["stats"])
    assert min(r["ev_slots"]) > 511, ("precondition: victims above slot 511", r["ev_slots"])
    assert r["n_diffs"] == 0, r["diffs"]
    assert r["err"] == 0, r["err"]
    assert r["b2_applied"] == r["ticks"], r["b2_applied"]
    assert r["warmed"] >= 1 and r["warm_launches"] >= 8, (r["warmed"], r["warm_launches"])   # FR13 at this NP
    print("FR12b C + X %d (NP %d, SB %d): %d layers equal; victims %s; grants %s"
          % (n_hist, r["NP"], r["sb"], r["compared"], sorted(r["ev_slots"]), sorted(r["take_slots"])))


def test_FR12c_the_old_9_bit_slot_field_is_caught_at_the_first_eviction():
    """NEGATIVE CONTROL: slot_bits forced to 9 at C + X = 992 rebuilds 23b8a38's key and mask (slot & 511). Before the
    verify tick no V2 runs (draft rows only), so the plant is not effective there and must not differ; at the verify
    tick (victims above 511, FR12b's precondition) the check must differ at layer 0."""
    if not _has_triton():
        pytest.skip("triton is not installed")
    r = _child("wide", "old")["wide"]
    assert r["sb"] == 9 and r["NP"] == 1024, r
    assert r["n_diffs"] > 0, ("the 9-bit slot field was not caught", r["stats"], r["ev_slots"])
    assert r["diffs"][0][:2] == [r["phase_c_tick"], 0], (r["diffs"][0], r["phase_c_tick"])
    assert any(d[2].startswith("v.t.") for d in r["diffs"]), r["diffs"]


# =====================================================================================================================
# FR13 [stack2] no compile inside a timed tick: every launch's compile key was warmed first (job 386 stack1a)
# =====================================================================================================================
@pytest.fixture(scope="module")
def warmrun():
    if not _has_triton():
        pytest.skip("triton is not installed")
    return _child("warm")["warm"]


@pytest.mark.parametrize("sched", ["f12b", "f20b"])
def test_FR13a_warm_up_covers_every_compile_key_and_changes_no_output(warmrun, sched):
    """With fused_warm 'on', warm_up launches every variant of a selection width before the width's first decision:
    every real launch's compile key (constexprs, num_warps, and the dtype / alignment / int class of each argument not
    in do_not_specialize) is a key that warm_up launched. The run is unchanged: check mode equal at every layer, err 0,
    the B2 world equal to v1, and the same per-layer stats trace as the run with warm 'off'."""
    on, off = warmrun["%s-on" % sched], warmrun["%s-off" % sched]
    for r in (on, off):
        assert r["n_bad"] == 0 and r["n_diffs"] == 0 and r["err"] == 0, (r["bad"], r["diffs"], r["err"])
        assert r["compared"] == r["layers"] > 0
    assert on["warm"] is True and off["warm"] is False and off["warm_launches"] == 0
    assert on["warmed"] >= 1 and on["warm_launches"] >= 8 and on["n_warm_keys"] >= 7, on   # V1 SCAN V2x2 D1x2 D2x2
    assert on["real"] > 0 and on["miss_new"] == [], ("compiled inside a decision", on["miss_new"])
    assert on["trace"] == off["trace"] and on["b2_applied"] == off["b2_applied"], "warm_up changed a decision"
    assert on["launches"] == off["launches"]                          # warm_up's launches are not counted there


def test_FR13b_the_old_variant_rule_compiles_inside_decisions(warmrun):
    """NEGATIVE CONTROL: under 23b8a38's rule (ints only in do_not_specialize, so pointer alignment counts;
    LZ = pow2(Rv * 4); SCAN BLOCK = pow2(max(R H, S))) the same launches have keys that warm_up never launched: the
    detector sees the per-tick variants that compiled inside the stack1a round."""
    miss = {s: warmrun["%s-on" % s]["miss_old"] for s in ("f12b", "f20b")}
    assert any(miss.values()), miss
    print("FR13b old rule: kernels with a key first seen inside a decision: %s" % miss)
