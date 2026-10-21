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
FR3  LAUNCHES PER LAYER (f12b, its draft-only ticks and its verify ticks): fused: 2 Triton launches per draft-only
     layer, 4 per layer with verify rows; on a draft-only layer after the tick's first, NO torch op other than an
     allocation or a view (TorchDispatchMode, paused inside the launches). The torch path's compute ops per such
     layer are counted for comparison (precondition: the counter sees them).
FR4  STATIC: the per-layer path (FusedResolver._layer) has no host read (S-C's F12d list) and no host-to-device copy
     (a planted read is found); every @triton.jit function of the module resolves its names in its module (L36,
     as F23; a planted factory kernel is found).
FR5  ONE TICK ROUND (the interpreter): the free-running greedy round (r2_tick.run_round, K 3, LAMBDA 7) on b2 with
     resolve="check" == resolve="torch" (rows, argmax per tick, committed tokens, decision stats); no difference at
     any compared layer; err 0.

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
                    bias_fallbacks=rep["bias_fallbacks"],
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
    assert r["bias_fallbacks"] == 0, r["bias_fallbacks"]               # the bias rows took the kernel's copy
    assert r["b2_applied"] == r["ticks"]
    assert PRE[sched](r), (sched, r["v1"], r["stats"])                 # the schedule reached its transitions


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


def test_FR3_two_launches_per_draft_only_layer_four_with_verify_rows_and_no_torch_compute(lockstep):
    c = lockstep["count"]
    assert c["fused"]["n_bad"] == 0 and c["torch"]["n_bad"] == 0, (c["fused"]["bad"], c["torch"]["bad"])
    fz, tz = c["fused"]["recs"], c["torch"]["recs"]
    assert len(fz) == len(tz) > 0
    d_only = [r for r in fz if not r["mixed"]]
    mixed = [r for r in fz if r["mixed"]]
    assert d_only and mixed, "precondition: f12b has draft-only and verify layers"
    assert all(r["launches"] == 2 for r in d_only), [(r["tick"], r["l"], r["launches"]) for r in d_only]
    assert all(r["launches"] == 4 for r in mixed), [(r["tick"], r["l"], r["launches"]) for r in mixed]  # f12b: every
    # layer with verify rows also has draft rows (_sched_pressure); a verify-only layer would launch 2
    later = [r for r in d_only if not r["first"]]
    assert later, "precondition: a draft-only layer after the tick's first"
    compute = sorted({op for r in later for op in r["ops"] if op not in VIEW_OR_ALLOC})
    assert compute == [], compute                                     # no torch kernel on such a layer
    t_later = [r for r in tz if not r["mixed"] and not r["first"]]
    t_compute = [sum(1 for op in r["ops"] if op not in VIEW_OR_ALLOC) for r in t_later]
    assert t_compute and min(t_compute) >= 20, t_compute               # precondition: the counter sees compute ops
    print("FR3 per draft-only layer (not the tick's first): fused %d Triton launches, %d torch compute ops; torch "
          "path %.1f compute ops (min %d, max %d) over %d layers; per verify layer fused %d launches"
          % (2, 0, sum(t_compute) / len(t_compute), min(t_compute), max(t_compute), len(t_compute), 4))


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
    assert "_launch(_fr_d1" in seg and "_launch(_fr_d2" in seg            # precondition: the right function
    assert _host_reads(seg) == [], _host_reads(seg)
    planted = seg.replace("        self.layers += 1", "        self.layers += 1\n        n = int(npg.sum())")
    assert planted != seg and "int(" in _host_reads(planted)            # NEGATIVE CONTROL


def test_FR4_every_triton_function_resolves_its_names_in_its_module():
    if not _has_triton():
        pytest.skip("triton is not installed")
    from triton.runtime.jit import JITFunction
    FR = TRM._load("r2_fusedres")
    fns = {n: v for n, v in vars(FR).items() if isinstance(v, JITFunction)}
    need = {"_fr_v1", "_fr_v2", "_fr_d1", "_fr_d2", "_row", "_sorted_unique", "_lookup", "_seen", "_pack", "_put1",
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
