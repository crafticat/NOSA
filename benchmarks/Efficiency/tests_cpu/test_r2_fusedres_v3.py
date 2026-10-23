"""[fr3] CPU tests of the fused resolver's V3 kernels (nosi/nosi/r2_fusedres.py, module docstring section V3: bounded
register tiles; FusedResolver(kernel="v3")). The kernels run under the Triton interpreter (TRITON_INTERPRET=1) in a
FRESH process. The interpreter runs the kernels' own flat addresses on the real tensors (L28: no hand-made twin of the
addresses), and the guards of fusedres2b find a write past a buffer.

MIRRORS of test_r2_fusedres.py (the SAME child script, with every `FR = ..._load("r2_fusedres")` line followed by
`FR.DEFAULT_KERNEL = "v3"; FR.KERNELS = ("v3",)`, so a resolver with any other kernel raises in the child):
FR3-1   FR1 on the four lockstep schedules (v3 == torch at every layer; the B2 world == v1), FR1's verify dedup.
FR3-2   FR2's negative controls (plant 1: Policy E newest first; plant 2: the (q + 1)-th FREE slot) are caught at their
        first effective layer; the stats trace is cumulative.
FR3-3   launches per layer: 4 per draft-only layer (PREP, D1, SCAN, D2), 7 with verify rows; no torch compute op on
        a draft-only layer after the tick's first.
FR3-5   FR5's free-running tick round: check == torch, the report names kernel v3.
FR3-6   FR6's hand cases (overflow / dup behind an overflow / in-row duplicate; two draft rows of one batch slot).
FR3-7   FR7's shared fixtures.
FR3-10  FR10: no v3 kernel writes past a buffer (guards checked after EVERY launch); the detector's control.
FR3-12  FR12b / c: C + X 992 and 1440, Policy E victims above slot 511 (v3 == torch); the 9-bit slot field is caught
        at the first eviction (negative control).
FR3-13  FR13a / b: warm_up launches every v3 compile key before the first decision; the old rule's control.
NEW for v3:
FR3-0   the kernel option: "v2" by default (unchanged), "v3" selectable, another name refused; plant 3 exists in v3
        only; the report names the kernel.
FR3-R   RANDOMIZED rounds: the four lockstep schedule kinds with new seeds (world and schedule), v3 == torch at every
        compared layer, err 0; preconditions over the set: evictions (LRU and in flight), grants, refusals, dups.
FR3-P   plant 3 (v3 only: a verify dup ignores the slot an earlier row took; the took table read) is caught at its
        first effective layer (where v_dup - v_dup_over grows) and never before; effective on >= 1 schedule (L34).
FR3-G   the v3 scratch (fr3.tab, fr3.flist, fr3.pm) is guarded; a REAL v3 kernel write past a buffer is found: PREP
        launched with a block table one stream short writes into the guard (named), the right size does not.
FR3-S   STATIC: no v3 kernel source has tl.sort or an NP-wide arange (the spill sources of _fr_v2, which has both:
        the negative control); _layer launches the five v3 kernels through _go.
FR3-T   Policy E's extraction on a CPU twin of the kernel's flat key layout (post[2], (S, NP) per stream, CH chunks):
        the e-th extracted key == torch.sort's e-th key (EXACT) and plant 1's order == the reversed candidate order.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fusedres_v3.py
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


FRT = _load_by_path("test_r2_fusedres_fr3ref", HERE / "test_r2_fusedres.py")   # the v2 tests: child and claims
TRM, OWN2, SCHED = FRT.TRM, FRT.OWN2, FRT.SCHED
p3 = FRT.p3                                                                      # pytest fixture, re-exported
_has_triton = FRT._has_triton

# the mirrored child: FR_CHILD with the kernel switched at every place a part loads the module
_LOADS = ('FR = TRM._load("r2_fusedres")', 'FR = TT._load("r2_fusedres")')
_SWITCH = '; FR.DEFAULT_KERNEL = "v3"; FR.KERNELS = ("v3",)'


def _mirror_src():
    src = FRT.FR_CHILD
    counts = [src.count(t) for t in _LOADS]
    assert counts == [4, 1], ("precondition: FR_CHILD loads the module in 4 TRM parts and 1 TT part", counts)
    for t in _LOADS:
        src = src.replace(t, t + _SWITCH)
    assert src.count(_SWITCH) == 5
    return src


# the v3-only child (randomized rounds, plant 3, the guard names and the planted out-of-bounds write)
FR3_CHILD = r'''
import importlib.util, json, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "FR3 child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "FR3 child: triton was imported before the interpreter flag was read"
import torch
HERE, PART = sys.argv[1], sys.argv[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


OWN = load("test_r2_own2_fr3child", os.path.join(HERE, "test_r2_own2.py"))
TRM, OWN2 = OWN.TRM, OWN.OWN2
FR = TRM._load("r2_fusedres")
assert FR.triton is not None, "FR3 child: triton did not import"
FR.DEFAULT_KERNEL = "v3"
FR.KERNELS = ("v3",)                                    # a resolver with any other kernel raises
OUT = {}
# RANDOMIZED: the four schedule kinds of OWN.SCHEDULES (same shapes), new world and schedule seeds
RAND = dict(
    r6=lambda: dict(seed=21, C=8, X=48, R_ver=8, sched=TRM._sched(301, 24, True)),
    r12b=lambda: dict(seed=23, C=2, X=96, R_ver=88, sched=TRM._sched_pressure(303, 15), hb=8),
    r20b=lambda: dict(seed=25, C=24, X=2, R_ver=2, sched=TRM._sched_pressure(305, 18, hb=36, window=3, nver=3), hb=36),
    r21b=lambda: dict(seed=27, C=16, X=2, R_ver=2, sched=TRM._sched_split(309, 30, hb=30, d_hi=20, window=2, nver=1),
                      hb=30, mode="force_only"),
    # [fr3] r21b (seed 27 / 309) reached no in-flight eviction (v_evict_infl 0, cpu-logs/fr3_wip2.txt): three more split
    # seeds, so the set reaches that branch (the precondition below still decides)
    r21c=lambda: dict(seed=29, C=16, X=2, R_ver=2, sched=TRM._sched_split(311, 30, hb=30, d_hi=20, window=2, nver=1),
                      hb=30, mode="force_only"),
    r21d=lambda: dict(seed=31, C=16, X=2, R_ver=2, sched=TRM._sched_split(313, 30, hb=30, d_hi=20, window=2, nver=1),
                      hb=30, mode="force_only"),
    r21e=lambda: dict(seed=7, C=16, X=2, R_ver=2, sched=TRM._sched_split(317, 30, hb=30, d_hi=20, window=2, nver=1),
                      hb=30, mode="force_only"),
)


def run(kw, plant=0, guard=False):
    svc_kw = dict(resolve="check")
    if plant:
        svc_kw["fused_plant"] = plant
    if guard:
        svc_kw["fused_guard"] = True
    bad, wv, wb = OWN._lockstep_b2(svc_kw=svc_kw, **kw)
    fr = wb.svc.fusedres
    rep = fr.report()
    return dict(kernel=rep["kernel"], plant=plant, n_bad=len(bad), bad=[repr(x)[:300] for x in bad[:3]],
                ticks=len(kw["sched"]), L=TRM.L, compared=fr.compared, layers=fr.layers, launches=fr.launches,
                n_diffs=len(fr.diffs), diffs=[list(d) for d in fr.diffs[:8]], stats=rep["stats"], err=rep["err"],
                trace=[[t, l, st, e] for (t, l, st, e) in fr.trace], first_diff_at=rep["first_diff_at"],
                guard_hits=[list(x) for x in fr.guard_hits], guard_checks=fr.guard_checks,
                guarded=sorted({n for (n, _, _) in fr._g_persist + fr._g_tick + fr._g_layer})), wb


if PART == "rand":
    OUT["rand"] = {name: run(RAND[name]())[0] for name in sorted(RAND)}
elif PART == "plant3":
    kws = dict((n, OWN.SCHEDULES[n]) for n in OWN.SCHEDULES)
    kws.update(RAND)
    OUT["plant3"] = {name: run(kws[name](), plant=3)[0] for name in sorted(kws)}
elif PART == "oob":
    r, wb = run(OWN.SCHEDULES["f6"](), guard=True)
    svc = wb.svc
    B, H = svc.B, svc.H
    S = B * H
    NP, ZP = max(FR._pow2(svc.n_hist), 64), max(FR._pow2(svc.nblk), 64)
    K = KP = 4
    SB = FR.slot_bits(NP)
    dm = svc.dmirror

    def prep(tab_streams):
        """One PREP launch (HAS_V: every store of the kernel) on private guarded buffers; the names of the buffers
        whose guard changed. tab_streams < S: the block table is too short, so the last programs write past it."""
        grp = []

        def gbuf(n, name):
            full = torch.full((n + FR.LANE_PAD,), FR.GUARD_SENTINEL, dtype=torch.int64)
            grp.append((name, full, n))
            return full[:n]

        def z(n):
            return torch.zeros((n + FR.LANE_PAD,), dtype=torch.int64)[:n]
        tab, flist, pm = gbuf(tab_streams * 7 * ZP, "tab"), gbuf(S * 2 * NP, "flist"), gbuf(S * 4, "pm")
        post, usg, err = gbuf(4 * S * NP, "post"), gbuf(S, "usg"), gbuf(1, "err")
        err.zero_()
        sel = torch.full((H, 1, K), -1, dtype=torch.int64)
        one, cnt = z(1), z(B)
        FR._launch(FR._fr3_prep, (S,), dm.code[0], dm.block[0], dm.cur[0], dm.prot[0], dm.stamp[0], sel, one,
                   one, one, cnt, cnt, one, one, one, cnt, cnt, one, tab, flist, pm, post, usg, err,
                   H, 1, K, svc.nblk, svc.n_hist, 1, 1, 0, 0, S,
                   KP=KP, NP=NP, ZP=ZP, CH=FR.fr3_chunk(KP, NP), ZC=FR.FR3_ZC, TC=FR.FR3_TC, HAS_V=True,
                   PE_INFL=True, SMUL=1 << SB, CMUL=1 << (FR.STAMP_BITS + SB), num_warps=4)
        return [name for (name, full, n) in grp if not bool((full[n:] == FR.GUARD_SENTINEL).all())]
    OUT["oob"] = dict(run=r, clean=prep(S), short=prep(S - 1), pad_ok=7 * ZP <= FR.LANE_PAD, ZP=ZP, S=S)
print("FR_JSON " + json.dumps(OUT))
'''


def _run_child(src, part, *extra):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", src, str(HERE), part] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=5400)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("FR_JSON ")]
    assert r.returncode == 0 and last, ("FR3 child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("FR_JSON "):])


def _mirror(part, *extra):
    return _run_child(_mirror_src(), part, *extra)


def _v3(part):
    return _run_child(FR3_CHILD, part)


def _need_triton():
    if not _has_triton():
        pytest.skip("triton is not installed")


@pytest.fixture(scope="module")
def lockstep3():
    _need_triton()
    return _mirror("lockstep")


@pytest.fixture(scope="module")
def warmrun3():
    _need_triton()
    return _mirror("warm")["warm"]


@pytest.fixture(scope="module")
def rand3():
    _need_triton()
    return _v3("rand")["rand"]


@pytest.fixture(scope="module")
def plant3():
    _need_triton()
    return _v3("plant3")["plant3"]


# =====================================================================================================================
# FR3-0 the kernel option
# =====================================================================================================================
def test_FR3_0_the_kernel_option_default_v2_v3_selectable_others_refused(monkeypatch):
    _need_triton()
    FR = TRM._load("r2_fusedres")
    assert FR.DEFAULT_KERNEL == "v2" and FR.KERNELS == ("v2", "v3")
    w = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="fused"))
    assert w.svc.fusedres.kernel == "v2"                                 # the default is unchanged
    with pytest.raises(ValueError, match="fused plant 3"):
        TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="check", fused_plant=3))
    monkeypatch.setattr(FR, "DEFAULT_KERNEL", "v3")
    w3 = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="check", fused_plant=3))
    assert w3.svc.fusedres.kernel == "v3" and w3.svc.fusedres.plant == 3
    assert w3.svc.report()["b2"]["fused"]["kernel"] == "v3"
    monkeypatch.setattr(FR, "DEFAULT_KERNEL", "v9")
    with pytest.raises(ValueError, match="fused kernel 'v9'"):
        TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="fused"))


# =====================================================================================================================
# FR3-1 / 2 / 6 / 7: FR1, FR2, FR6, FR7 on v3 (the v2 file's own claims, called on the v3 child's output)
# =====================================================================================================================
@pytest.mark.parametrize("sched", SCHED)
def test_FR3_1_v3_equals_torch_at_every_layer_and_b2_equals_v1(lockstep3, sched):
    FRT.test_FR1_fused_equals_torch_at_every_layer_and_b2_equals_v1(lockstep3, sched)


def test_FR3_1_the_lockstep_schedules_reach_the_verify_dedup(lockstep3):
    FRT.test_FR1_the_lockstep_schedules_reach_the_verify_dedup(lockstep3)


@pytest.mark.parametrize("plant", [1, 2])
def test_FR3_2_each_plant_is_caught_at_its_first_effective_layer(lockstep3, plant):
    FRT.test_FR2_each_plant_is_caught_at_its_first_effective_layer(lockstep3, plant)


def test_FR3_2_the_stats_trace_is_cumulative_and_matches_the_report(lockstep3):
    FRT.test_FR2_the_stats_trace_is_cumulative_and_matches_the_report(lockstep3)


def test_FR3_6a_overflow_dup_behind_overflow_and_an_in_row_duplicate(lockstep3):
    FRT.test_FR6a_overflow_dup_behind_overflow_and_an_in_row_duplicate(lockstep3)


def test_FR3_6b_two_draft_rows_of_one_batch_slot(lockstep3):
    FRT.test_FR6b_two_draft_rows_of_one_batch_slot(lockstep3)


def test_FR3_7_the_shared_fixtures_in_check_mode(lockstep3):
    FRT.test_FR7_the_shared_fixtures_in_check_mode(lockstep3)


# =====================================================================================================================
# FR3-3 launches per layer (v3: PREP + 3 per row kind present)
# =====================================================================================================================
def test_FR3_3_four_launches_per_draft_only_layer_seven_with_verify_rows_and_no_torch_compute(lockstep3):
    c = lockstep3["count"]
    assert c["fused"]["n_bad"] == 0 and c["torch"]["n_bad"] == 0, (c["fused"]["bad"], c["torch"]["bad"])
    fz = c["fused"]["recs"]
    d_only = [r for r in fz if not r["mixed"]]
    mixed = [r for r in fz if r["mixed"]]
    assert d_only and mixed, "precondition: f12b has draft-only and verify layers"
    assert all(r["launches"] == 4 for r in d_only), [(r["tick"], r["l"], r["launches"]) for r in d_only]
    assert all(r["launches"] == 7 for r in mixed), [(r["tick"], r["l"], r["launches"]) for r in mixed]
    later = [r for r in d_only if not r["first"]]
    assert later, "precondition: a draft-only layer after the tick's first"
    compute = sorted({op for r in later for op in r["ops"] if op not in FRT.VIEW_OR_ALLOC})
    assert compute == [], compute


# =====================================================================================================================
# FR3-5 one tick round
# =====================================================================================================================
def test_FR3_5_a_free_running_round_on_b2_check_equals_torch(p3):
    _need_triton()
    r = _mirror("tick", p3.__file__)["tick"]
    assert r["same_rec"] and r["same_out"] and r["same_stats"], (r["first_rec_diff"], r["torch_stats"],
                                                                  r["check_stats"])
    assert r["mirror"] == [[], []], r["mirror"]
    f = r["fused"]
    assert f is not None and f["kernel"] == "v3" and f["mode"] == "check", f
    assert f["compared"] > 0 and f["compared"] == f["layers"], f
    assert f["diffs"] == 0 and r["diffs"] == [], r["diffs"]
    assert f["err"] == 0, f
    assert r["torch_stats"]["ticks"] > 20 and r["torch_stats"]["publications"] > 0, r["torch_stats"]


# =====================================================================================================================
# FR3-10 the out-of-bounds detector on every v3 launch
# =====================================================================================================================
def test_FR3_10_no_v3_kernel_writes_past_a_buffer_and_the_guard_detector_works():
    _need_triton()
    g = _mirror("guard")["guard"]
    c = g.pop("control")
    assert c["new_hits"] == [[99, 0, "control", c["buffer"]]], c        # NEGATIVE CONTROL: a changed guard is named
    for name, r in g.items():
        assert r["checks"] > 0 and r["guarded"] > 0, (name, "precondition: guards were checked", r)
        assert r["hits"] == [], (name, "a kernel wrote past a buffer (tick, layer, kernel, buffer)", r["hits"][:5])
        assert r["n_bad"] == 0 and r["n_diffs"] == 0, (name, r["bad"])


def test_FR3_G_the_v3_scratch_is_guarded_and_a_real_v3_overrun_is_found():
    _need_triton()
    o = _v3("oob")["oob"]
    r = o["run"]
    assert r["kernel"] == "v3" and r["n_diffs"] == 0 and r["n_bad"] == 0, (r["diffs"], r["bad"])
    assert r["guard_checks"] > 0 and r["guard_hits"] == [], r["guard_hits"][:5]
    assert {"fr3.tab", "fr3.flist", "fr3.pm"} <= set(r["guarded"]), r["guarded"]
    assert o["pad_ok"], ("precondition: the planted overrun stays in the pad", o["ZP"])
    assert o["clean"] == [], o["clean"]                                  # the right size writes nothing past it
    assert o["short"] == ["tab"], o["short"]                             # NEGATIVE CONTROL: found and named


# =====================================================================================================================
# FR3-12 C + X above 512 (victims above slot 511) and the 9-bit control
# =====================================================================================================================
@pytest.mark.parametrize("n_hist", [992, 1440])
def test_FR3_12b_v3_equals_torch_with_policy_e_victims_above_slot_511(n_hist):
    _need_triton()
    r = _mirror("wide", str(n_hist))["wide"]
    assert r["n_hist"] == n_hist and r["NP"] == max(1024, 1 << (n_hist - 1).bit_length()) and r["sb"] > 9, r
    assert r["raised"] is None, (r["raised"], r["raised_tick"])
    assert r["compared"] == r["layers"] == r["ticks"] * r["L"], (r["compared"], r["layers"], r["ticks"])
    assert r["stats"]["v_evict"] > 0 and r["ev_slots"], ("precondition: Policy E evicted", r["stats"])
    assert min(r["ev_slots"]) > 511, ("precondition: victims above slot 511", r["ev_slots"])
    assert r["n_diffs"] == 0, r["diffs"]
    assert r["err"] == 0, r["err"]
    assert r["b2_applied"] == r["ticks"], r["b2_applied"]
    assert r["warmed"] >= 1 and r["warm_launches"] == 10 * r["warmed"], (r["warmed"], r["warm_launches"])   # v3


def test_FR3_12c_the_old_9_bit_slot_field_is_caught_at_the_first_eviction():
    _need_triton()
    r = _mirror("wide", "old")["wide"]
    assert r["sb"] == 9 and r["NP"] == 1024, r
    assert r["n_diffs"] > 0, ("the 9-bit slot field was not caught", r["stats"], r["ev_slots"])
    assert r["diffs"][0][:2] == [r["phase_c_tick"], 0], (r["diffs"][0], r["phase_c_tick"])
    assert any(d[2].startswith("v.t.") for d in r["diffs"]), r["diffs"]


# =====================================================================================================================
# FR3-13 no compile inside a decision (v3's variants)
# =====================================================================================================================
@pytest.mark.parametrize("sched", ["f12b", "f20b"])
def test_FR3_13a_warm_up_covers_every_v3_compile_key_and_changes_no_output(warmrun3, sched):
    FRT.test_FR13a_warm_up_covers_every_compile_key_and_changes_no_output(warmrun3, sched)
    on = warmrun3["%s-on" % sched]
    assert on["warm_launches"] == 10 * on["warmed"] and on["n_warm_keys"] >= 10, on   # PREP x2 V1 SCAN V2x2 D1x2 D2x2


def test_FR3_13b_the_old_variant_rule_compiles_inside_decisions(warmrun3):
    FRT.test_FR13b_the_old_variant_rule_compiles_inside_decisions(warmrun3)


# =====================================================================================================================
# FR3-R randomized rounds
# =====================================================================================================================
def test_FR3_R_randomized_rounds_v3_equals_torch(rand3):
    for name, r in sorted(rand3.items()):
        assert r["kernel"] == "v3", (name, r["kernel"])
        assert r["compared"] == r["layers"] > 0, (name, r["compared"], r["layers"])
        assert r["n_diffs"] == 0, (name, r["diffs"])                    # v3 == torch, every compared layer
        assert r["err"] == 0, (name, r["err"])
        if r["n_bad"]:                                                   # B2 vs v1 is not this file's claim
            print("FR3-R %s: B2 vs v1 differences %d (first %s); v3 == torch still holds" % (name, r["n_bad"],
                                                                                            r["bad"][:1]))
    tot = {k: sum(r["stats"][k] for r in rand3.values()) for k in
           ("v_evict", "v_evict_infl", "v_grant", "d_grant", "d_refused", "v_dup")}
    assert all(v > 0 for v in tot.values()), ("precondition: the randomized set reaches every branch", tot)
    print("FR3-R: %d randomized runs, %d layers compared, stats %s"
          % (len(rand3), sum(r["compared"] for r in rand3.values()), tot))


# =====================================================================================================================
# FR3-P plant 3 (the took table read): caught at its first effective layer
# =====================================================================================================================
def test_FR3_P_plant_3_is_caught_at_its_first_effective_layer(plant3):
    effective = []
    for name, r in sorted(plant3.items()):
        assert r["kernel"] == "v3" and r["plant"] == 3, (name, r["kernel"], r["plant"])
        names = list(r["stats"])
        first_eff = next(((t, l) for (t, l, st, e) in r["trace"]
                          if dict(zip(names, st))["v_dup"] - dict(zip(names, st))["v_dup_over"] > 0), None)
        if first_eff is None:
            assert r["n_diffs"] == 0, (name, "a plant without effect changed an output", r["diffs"])
            print("FR3-P on %s: NO-OP (every verify dup is behind an overflow), not counted" % name)
            continue
        effective.append(name)
        assert r["n_diffs"] > 0, (name, "effective at %s but not caught" % (first_eff,))
        assert tuple(r["diffs"][0][:2]) == first_eff, (name, r["diffs"][0], first_eff)
        assert r["first_diff_at"] == list(first_eff), (name, r["first_diff_at"])
        print("FR3-P on %s: effective first at tick %d layer %d, caught there (%s)"
              % (name, first_eff[0], first_eff[1], r["diffs"][0][2]))
    assert effective, ("precondition (L34): plant 3 was effective on no schedule",
                       {k: (v["stats"]["v_dup"], v["stats"]["v_dup_over"]) for k, v in plant3.items()})


# =====================================================================================================================
# FR3-S static: the spill sources are gone from every v3 kernel
# =====================================================================================================================
V3_FNS = ("_fr3_prep", "_fr3_v1", "_fr3_v2", "_fr3_d1", "_fr3_d2", "_tab3", "_d_state3", "_selbits_at")
SPILL_TOKENS = ("tl.sort(", "tl.arange(0, NP)", "_lookup(", "_at_slot(", "_slots(", "_d_state(", "_selbits(")


def _fn_sources(path, names):
    src = path.read_text()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    return out


def test_FR3_S_no_v3_kernel_has_a_sort_or_an_np_wide_tile_and_layer_launches_them():
    srcs = _fn_sources(NOSI / "r2_fusedres.py", set(V3_FNS) | {"_fr_v2"})
    assert set(V3_FNS) <= set(srcs), sorted(set(V3_FNS) - set(srcs))
    bad = {n: [t for t in SPILL_TOKENS if t in srcs[n]] for n in V3_FNS}
    assert all(v == [] for v in bad.values()), bad
    ctl = [t for t in SPILL_TOKENS if t in srcs["_fr_v2"]]                 # NEGATIVE CONTROL: v2's V2 has them
    assert "tl.sort(" in ctl and "_lookup(" in ctl and "_at_slot(" in ctl, ctl
    seg = FRT._method_source(NOSI / "r2_fusedres.py", "FusedResolver", "_layer")
    for k in ("_fr3_prep", "_fr3_v1", "_fr_scan.v", "_fr3_v2", "_fr3_d1", "_fr_scan.d", "_fr3_d2"):
        assert 'self._go(where, "%s"' % k in seg, k
    assert FRT._host_reads(seg) == [], FRT._host_reads(seg)


# =====================================================================================================================
# FR3-T Policy E's extraction on a CPU twin of the kernel's flat key layout
# =====================================================================================================================
def _extract(flat, pid, S, NP, CH, n, smallest, cmul2=None):
    """The kernel's loop on the flat post buffer: keys of stream pid at (2 S + pid) NP + c0 + lane, CH lanes per
    pass; n extractions. smallest: the next key above the last; else (plant 1) the next candidate key below it."""
    base = (2 * S + pid) * NP
    last = -1 if smallest else cmul2
    out = []
    for _ in range(n):
        best = (1 << 63) - 1 if smallest else -1
        for c0 in range(0, NP, CH):
            kv = flat[base + c0: base + c0 + CH]
            if smallest:
                cand = kv[kv > last]
                if cand.numel():
                    best = min(best, int(cand.min()))
            else:
                cand = kv[(kv < last) & (kv < cmul2)]
                if cand.numel():
                    best = max(best, int(cand.max()))
        last = best
        out.append(best)
    return out


def test_FR3_T_policy_e_extraction_equals_the_sort_on_the_flat_layout():
    g = torch.Generator().manual_seed(7)
    S, NP, CH, SB = 6, 256, 64, 9
    smul, cmul = 1 << SB, 1 << (47 + SB)
    flat = torch.full((4 * S * NP,), -7, dtype=torch.int64)                 # other planes: not read
    for pid in range(S):
        cls = torch.randint(0, 3, (NP,), generator=g)
        stamp = torch.randint(0, 1 << 20, (NP,), generator=g)              # ties across slots: the slot breaks them
        stamp[: NP // 4] = 5
        key = cls * cmul + torch.where(cls < 2, stamp, torch.zeros_like(stamp)) * smul + torch.arange(NP)
        flat[(2 * S + pid) * NP:(2 * S + pid + 1) * NP] = key
    for pid in range(S):
        keys = flat[(2 * S + pid) * NP:(2 * S + pid + 1) * NP]
        srt = torch.sort(keys).values.tolist()
        ncand = int((keys < 2 * cmul).sum())
        assert ncand > 4, "precondition: candidates"
        n = min(ncand, 40)
        assert _extract(flat, pid, S, NP, CH, n, True) == srt[:n], pid           # EXACT: srtc[e]
        assert _extract(flat, pid, S, NP, CH, n, False, 2 * cmul) == [srt[ncand - 1 - e] for e in range(n)], pid
        slots = [k & (smul - 1) for k in srt[:n]]
        assert len(set(slots)) == n                                              # distinct victims
    other = flat.clone()                                                         # NEGATIVE CONTROL: a twin that
    pid = 1                                                                      # reads the wrong plane differs
    wrong = _extract(other, pid, S, NP, CH, 3, True)
    shifted = _extract(torch.roll(other, NP), pid, S, NP, CH, 3, True)
    assert wrong != shifted
