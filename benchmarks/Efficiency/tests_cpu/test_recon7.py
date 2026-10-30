"""CPU tests of RECON7 (KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the
measurement cells of the reconciliation's discriminators: (a) benchmarks/Efficiency/r2_recon7_cell.py with the probe
r2_recon7_probe.py, (b) r2_recon7_streams.py, (c) r2_recon7_mem.py; the protocol stamp r2_recon7_protocol.py; the step
script r2_recon7_step.sh; docs/NOTES_recon7.md. The D-arm world: tests_cpu/i4dc_world.py (FakeGraphsDCD: CUDA graphs
and Inductor's GPU backend do not run on the CPU). CPU stand-ins establish the plumbing, the bit equality of the probed
arm with the plain arm, the counters' arithmetic and the gates; only the GPU cells establish timing, CUPTI's kernel
attribution, memory and the concurrency answer.

GATE 0    test_recon7_gate0_files: the tree = I4DC 2e9e497 + the RECON7 files + ONE declaration block in each older Gate-0
          test (and ONE line of test_i4dc's name scan); nosi/ and every older file are 2e9e497's; the byte copies
          r2_i4c_cell.py (3aef060), r2_i4m_cell.py and r2_wsm_cell.py (f7fb2c5) are their blobs; no kernel source; the
          fork is fia6's blob; no hand-written kernel; nobody else names the RECON7 modules.
          test_recon7_copies: run_arm_r7 = r2_i4dc_cell.run_arm_d and probe_arm_r7 = r2_i4m_cell.probe_arm_x line for
          line, plus the lines marked "# R7" only.
PROBE     test_probe_equal (both D arms): the probed run = the plain run except TentState's check-only counters and the
          compile record's section count; L4PTFDGUDCDP-PEQ / -COUNTS and every inherited arm gate pass; windows joint,
          commit, draft0, draft1 per profiled round on the replay path; the labelled windows hold the graph-path labels;
          the counted pass's counters close (the tap's NEW blocks = the meter's); the recording round recorded.
          test_probe_gates: the probe gates pass on the CPU evidence and their plants fail.
          test_probe_plant: a probe that changes a value is CAUGHT by PEQ (and is not a no-op, L34).
FIX       test_probe_statistic_fix: R2's measured C 73 probe joints (section 3.5) planted with their drain rounds: the
          old median 374.6 is biased by -16.4 ms against the drain-free 391.0; the CPU run's drain rounds = an independent
          derivation from the runner's host counts.
COUNTERS  test_skip_flags_synthetic: SkipRecord + GatherTap on planted skips and gathers (flagA, flagB, store, range).
STREAMS   test_streams_pure, test_streams_measure_plumbing (CPU stand-ins; the EXACT plant).
MEM       test_mem_pure (the occupancy tap, the fit, the arguments).
CELL      test_recon7_cell_pure, test_protocol_state, test_step_script_recon7, test_notes_recon7.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_recon7.py
"""
from __future__ import annotations

import ast
import collections
import functools
import gzip
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "2e9e497de3dd2104341bada4769363dce820c11b"          # nosi-r2-i4dc: I4DC with the lead's registration
I4C_SHA = "3aef060e7e3eea94e029fd36a2a6c0b9ab96d40d"           # nosi-r2-i4c: the profiler window (r2_i4c_cell.py)
I4M_SHA = "f7fb2c59dd68fb9eb95add3adc8c7af3139cceea"           # nosi-r2-i4m: the memory probe (r2_i4m_cell, r2_wsm_cell)
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-recon7"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
COPIES = {E_ + "r2_i4c_cell.py": I4C_SHA, E_ + "r2_i4m_cell.py": I4M_SHA, E_ + "r2_wsm_cell.py": I4M_SHA}
MODULES = ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py", "r2_recon7_streams.py",
           "r2_recon7_mem.py")
RECON7_NEW = tuple(E_ + m for m in MODULES) + (E_ + "r2_recon7_step.sh", T_ + "test_recon7.py",
                                               "docs/NOTES_recon7.md") + tuple(COPIES)
RECON7_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                       "i4h", "i4s", "i4cc", "i4dc"))
RECON7_EDITED = {T_ + "test_i4dc.py": "test_i4dc_gate0_files"}
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
PE = 4                                 # the world's timing scenario: probe (and check) rounds every 4 rounds
N_RUN = 48                             # committed tokens per request: about 16 rounds + the drain


class Evidence(UserWarning):
    """An evidence line (the CPU template runs pytest without -s: the warnings summary carries these lines)."""


def _ev(msg: str) -> None:
    print(msg)
    warnings.warn(Evidence(msg), stacklevel=2)


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4dc_world")
TS = _load("test_i4s")
BC, L = X.BC, X.L


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _mod(name):
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    return importlib.import_module(name)


# ------------------------------------------------------------------ GATE 0
def test_recon7_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(RECON7_NEW) | set(RECON7_DECL), sorted(changed - set(RECON7_NEW) - set(RECON7_DECL))  # (a)
    assert all((ROOT / f).is_file() for f in RECON7_NEW), [f for f in RECON7_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""       # (b) nosi: 2e9e497's
    for f, sha in COPIES.items():                                         # (c) the byte copies
        if _have(sha):
            assert _git("hash-object", str(ROOT / f)).stdout.strip() == _git("rev-parse", "%s:%s" % (sha, f)).stdout.strip(), f
    for f in RECON7_DECL:                                                 # (d) one RECON7 block each (+ one line)
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        allowed = {RECON7_EDITED[f]} if f in RECON7_EDITED else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert len(minus) <= (1 if f in RECON7_EDITED else 0), (f, minus)
        assert any("r2_recon7_cell.py" in x for x in body) and any("RECON7" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    for name in MODULES:                                                   # (e) NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (f) nobody else names them
        if p.name.startswith("r2_recon7_"):
            continue
        assert "r2_recon7" not in p.read_text(), p.name


def _body_lines(fn):
    """The source lines of a function after its docstring."""
    src = inspect.getsource(fn)
    lines = src.splitlines()
    node = ast.parse(src).body[0]
    first = node.body[0]
    start = first.end_lineno if (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                                 and isinstance(first.value.value, str)) else first.lineno - 1
    return lines[start:]


def test_recon7_copies():
    C, M = _mod("r2_recon7_cell"), _mod("r2_recon7_mem")
    for new, old in ((C.run_arm_r7, C.DCC.run_arm_d), (M.probe_arm_r7, M.I4M.probe_arm_x)):
        nb, ob = _body_lines(new), _body_lines(old)
        core = [x for x in nb if not x.rstrip().endswith("# R7")]
        extra = [x for x in nb if x.rstrip().endswith("# R7")]
        assert core == ob, [(i, a, b) for i, (a, b) in enumerate(zip(core, ob)) if a != b][:3] or (len(core), len(ob))
        assert 2 <= len(extra) <= 5, extra
    _ev("RECON7 COPIES: run_arm_r7 = run_arm_d + %d R7 lines; probe_arm_r7 = probe_arm_x + %d R7 lines" % (
        sum(1 for x in _body_lines(C.run_arm_r7) if x.rstrip().endswith("# R7")),
        sum(1 for x in _body_lines(M.probe_arm_r7) if x.rstrip().endswith("# R7"))))


# ------------------------------------------------------------------ the probed runs (CPU world)
ARM_X = dict(GUDCD="L4PTFDGUDCD", GUDKCD="L4PTFDGUDKWCD")
ARM_P = dict(GUDCD="L4PTFDGUDCDP", GUDKCD="L4PTFDGUDKWCDP")
C_OF = dict(GUDCD="GUDC", GUDKCD="GUDKC")


@functools.lru_cache(maxsize=None)
def _plain(arm):
    return X.run_arm(arm, "timing", B=8, N=N_RUN)


def _rounds_plan(rounds: int):
    """profile [the 2nd probe round] (labelled), count [the 3rd]; the others marked."""
    probe = [g for g in range(PE, rounds, PE)]
    assert len(probe) >= 4, "precondition: four probe rounds (rounds %d)" % rounds
    return [probe[1]], [probe[1]], [probe[2]]


_PROBED = {}


def _probed(arm, tmp_factory, cls=None, tag="p"):
    key = (arm, tag)
    if key not in _PROBED:
        RP = _mod("r2_recon7_probe")
        cls = cls or RP.ProbeR7
        _plain(C_OF[arm])                                              # the C arm first: it compiles the joint helpers
        base = _plain(arm)
        prof, lab, cnt = _rounds_plan(base["rd"]["rounds"])
        tmp = str(tmp_factory.mktemp("r7_%s_%s" % (arm, tag)))
        box, hist = {}, {}

        def on_runner(run):
            box["p"] = cls(run, BC.HostClock(), tmp, ARM_P[arm], prof, lab, cnt, True,
                           X.W.BS * X.W.D * 2 * 2 + X.W.BS * 2)
            a0 = run._ps_after

            def ps_after(P, hc):                                        # the independent drain derivation
                out = a0(P, hc)
                hist[int(run.round)] = sum(1 for c in run.h_count if int(c) >= int(run.N))
                return out
            run._ps_after = ps_after
        ev = X.run_arm(arm, "timing", B=8, N=N_RUN, on_runner=on_runner)
        p = box["p"]
        p.detach()
        ev["probe"] = p.evidence(BC.HostClock().elapsed)
        ev["plan"] = (prof, lab, cnt)
        ev["after_hist"] = hist
        ev["tmp"] = tmp
        _PROBED[key] = ev
    return _PROBED[key]


CHECK_ONLY = ("forced", "injected", "edge_rows", "fill_rows")


def _cpu(ev, C):
    return X.cell_inputs(ev, C)


@pytest.mark.parametrize("arm", ["GUDCD", "GUDKCD"])
def test_probe_equal(arm, tmp_path_factory):
    C, RP = _mod("r2_recon7_cell"), _mod("r2_recon7_probe")
    base = _plain(arm)
    ev = _probed(arm, tmp_path_factory)
    keep = arm == "GUDKCD"
    bad = TS._diff(base, ev, skip=("tent", "compile"), tensors_skip=("forced", "injected", "c.edge_rows",
                                                                     "c.fill_rows"))
    assert bad == [], bad[:10]                                         # THE CLAIM: the probe changes no value
    ta, tb = dict(ev["rd"]["tent"]), dict(base["rd"]["tent"])
    diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
    assert set(diff) <= set(CHECK_ONLY), diff                          # only the check-only counters
    pv = ev["probe"]
    prof, lab, cnt = ev["plan"]
    slots = sorted((r["round"], r["slot"]) for r in pv["profile"]["passes"])
    assert slots == sorted((g, s) for g in prof for s in ("joint", "commit", "draft0", "draft1")), slots
    sk = pv["profile"]["skipped"]                                      # marks / records skip non-replay passes only
    assert not [s for s in sk if not s[2].startswith("marks: ")] and pv["profile"]["done"] == prof, sk
    for rec in pv["profile"]["passes"]:
        assert rec.get("error") is None, rec
        if rec["slot"] != "commit":
            assert rec["path_ok"] is True, (rec["slot"], rec["path"])  # the steady replay path inside every window
        with gzip.open(os.path.join(ev["tmp"], rec["trace"]), "rt") as f:
            tr = json.load(f)
        names = [e["name"][len(RP.LABEL_PREFIX):] for e in tr["traceEvents"] if e.get("ph") == "X"
                 and e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith(RP.LABEL_PREFIX)]
        kind = dict(joint="jointg", commit="commit").get(rec["slot"], "draftg")
        want, wl = RP.expected_labels_r7(kind, L, keep)
        if kind != "commit":
            want, wl = want[:-1], wl[:-1]                              # "end" opens no range
        if rec["labelled"]:
            got = sorted(names)
            assert got == sorted("%s@%d" % (a, b) for a, b in zip(want, wl)), (rec["slot"], got[:12])
        else:
            assert names == [], rec["slot"]
    c = pv["counted"]
    assert [x["round"] for x in c] == cnt and pv["recorded"] == [(cnt[0] - 1, "draft0"), (cnt[0] - 1, "draft1")]
    t, u = c[0]["tap"], c[0]["union"]
    assert c[0]["eager_layers"] == L and u["layers"] == L and t["calls"] >= L and t["captured"] == 0
    new = t["new_win"] + t["new_store"] + t["new_other"]
    assert new == c[0]["meter_new"], (t, c[0]["meter_new"])           # the tap's NEW blocks = the meter's count
    assert t["a_win"] + t["a_store"] <= new and t["b_win"] <= t["new_win"] and t["ab_win"] <= t["b_win"]
    assert t["oor"] == 0 and c[0]["skip_oor"] == 0 and c[0]["recorded_layers"] == 2 * L
    _ev("RECON7 PROBE %s: rounds %d, profiled %s, counted %s; windows %d; tap %s; union %s; skips recorded %s; "
        "drain %s; check-only differences %s" % (arm, ev["rd"]["rounds"], prof, cnt, len(pv["profile"]["passes"]),
                                                 {k: t[k] for k in ("calls", "kinds", "new_win", "new_store", "a_win",
                                                                    "a_store", "b_win", "st_max")},
                                                 {k: u[k] for k in ("u_max", "layers")}, c[0]["skips"], pv["drain"],
                                                 diff))
    # the arm gates on the CPU evidence: the plain arm's gates under the probed name, COUNTS and PEQ against it
    x = _plain(arm)
    C_ = _plain(C_OF[arm])
    keepd = {"L4PTFD": _cpu(_plain("D"), C), ARM_X[arm]: _cpu(x, C)}
    if keep:
        keepd["L4PTFDGUDKWC"] = _cpu(C_, C)
    rd = ev["rd"]
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = C.arm_gates_r7(ARM_P[arm], rd, rec, _cpu(ev, C), keepd, False, 8, 10, N_RUN, L, 1.0)
    names = [g[0] for g in gs]
    assert all(n.startswith(ARM_P[arm] + "-") for n in names) and ARM_P[arm] + "-PEQ" in names, names
    assert names.count(ARM_P[arm] + "-COUNTS") == 1
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]


def test_probe_gates(tmp_path_factory):
    C, RP = _mod("r2_recon7_cell"), _mod("r2_recon7_probe")
    ev = _probed("GUDCD", tmp_path_factory)
    pv, rd = ev["probe"], ev["rd"]
    prof, lab, cnt = ev["plan"]
    caps = {"x": C.HC.capture_rounds_x(rd)}
    xr = sorted(set(C.HC.excluded_rounds(caps)) | set(pv["recording_rounds"]))
    rec = dict(phases_median=dict(joint=C.WA._med([rd["phases"][g][RP.JOINT] for g in C.SC.steady_rounds(
        rd["rounds"], PE) if g not in set(xr)])))
    res = dict(probe=pv, read=rd)
    rec.update(C.probe_record_r7(res, rec, None, None, PE, pv["drain"], dict(L=L, H=2, D=X.W.D, block_bytes=64),
                                 8))
    gs = {g[0]: g for g in C.probe_gates_r7("P", res, rec, xr, PE, L, X.K0, True)}
    _ev("RECON7 GATES (CPU): %s" % {k: (v[2], v[3][:140]) for k, v in gs.items()})
    for n in ("P-PROF-PATH", "P-MARKS-NEUTRAL", "P-MARKS-SHAPE", "P-COUNT"):
        assert gs[n][2] is True, gs[n]
    assert gs["P-PROF"][1] is False                                    # REPORT
    big = RP.I4C.phase_sum_gate("P", rec["probe_joint"]["all"], tol=(1e9, 0.0))   # CPU: host time, not device time
    assert big[2], big[3]
    assert pv["joint"] and pv["draft"] and all(x["round"] not in prof + cnt for x in pv["joint"])
    # PLANTS
    passes = pv["profile"]["passes"]
    bad = [dict(p, path_ok=False) if p["slot"] == "joint" else p for p in passes]
    assert RP.prof_path_gate("P", prof, pv["profile"]["done"], bad, [], 2)[2] is False      # an eager window
    assert RP.prof_path_gate("P", prof, pv["profile"]["done"], [p for p in passes if p["slot"] != "commit"], [],
                             2)[2] is False                                                 # a missing window
    assert RP.prof_path_gate("P", prof, pv["profile"]["done"], passes, [(prof[0], "joint", "x")], 2)[2] is False
    marked = sorted({x["round"] for x in pv["joint"]})
    assert RP.marks_neutral_gate_r7("P", rd, PE, marked, prof, cnt, pv["recording_rounds"], [], rec)[2] is False
    steady = [g for g in C.SC.steady_rounds(rd["rounds"], PE) if g not in set(xr)]
    assert RP.marks_neutral_gate_r7("P", rd, PE, marked + [steady[0]], prof, cnt, pv["recording_rounds"], xr,
                                    rec)[2] is False                                        # a marked steady round
    assert RP.marks_neutral_gate_r7("P", rd, PE, marked, prof, prof, pv["recording_rounds"], xr, rec)[2] is False
    assert RP.count_gate("P", cnt, pv["counted"], [], L, 2)[2] is False                    # no recording
    c2 = [dict(c, eager_layers=L - 1) for c in pv["counted"]]
    assert RP.count_gate("P", cnt, c2, pv["recorded"], L, 2)[2] is False                   # a replayed layer
    seq = [dict(x, seq=list(x["seq"])[:-1]) for x in pv["draft"]]
    assert RP.marks_shape_r7("P", pv["joint"], seq, L, True)[2] is False                   # a missing draft mark


def test_probe_plant(tmp_path_factory):
    C, RP = _mod("r2_recon7_cell"), _mod("r2_recon7_probe")

    class Plant(RP.ProbeR7):                                           # NEGATIVE CONTROL: the probe computes
        def _profiled(self, model, plan, g, slot):
            out = super()._profiled(model, plan, g, slot)
            if slot == "draft0":                                       # request 0's draft logits: token 5 wins
                out = out.clone()
                out[0, 0, 5] += 1e9
            return out
    ev = _probed("GUDCD", tmp_path_factory, cls=Plant, tag="plant")
    base = _plain("GUDCD")
    bad = TS._diff(base, ev, skip=("tent", "compile"), tensors_skip=("forced", "injected", "c.edge_rows",
                                                                     "c.fill_rows"))
    g = RP.peq_gate_r7("P", _cpu(ev, C), _cpu(base, C), "L4PTFDGUDCD", N_RUN)
    _ev("RECON7 PLANT probe_computes: differs %s; PEQ %s" % (bad[:6], g[2]))
    assert bad != [] and g[2] is False                                 # not a no-op (L34); CAUGHT by PEQ


# ------------------------------------------------------------------ THE STATISTIC FIX
R2_C73 = [(16, 369.5), (32, 374.5), (48, 374.6), (64, 398.1), (80, 393.4), (128, 391.5), (144, 390.5), (160, 397.8),
          (176, 369.4), (192, 248.3), (208, 233.4)]                    # RECON7_R2_joint.md section 3.5 (A6:89)


def test_probe_statistic_fix(tmp_path_factory):
    RP = _mod("r2_recon7_probe")
    drain = RP.drain_rounds({g: (1 if g >= 170 else 0) for g, _ in R2_C73})
    s = RP.probe_stat(R2_C73, drain)
    _ev("RECON7 FIX (R2's C 73 probe joints): old median %s, drain-free %s, bias %s, excluded %s" % (
        s["all"], s["nodrain"], s["bias"], s["excluded"]))
    assert drain == [176, 192, 208] and s["all"] == 374.6 and s["nodrain"] == 391.0 and s["bias"] == -16.4
    assert s["n_all"] == 11 and s["n_nodrain"] == 8
    jst = [dict(round=g, labels=dict(a=v), total=v, lead=1.0, tail=1.0, span=v - 2.0) for g, v in R2_C73]
    rd = dict(rounds=209, phases=[[0.1, 0.0, 1.0, 1.0] for _ in range(209)])
    for g, v in R2_C73:
        rd["phases"][g][RP.JOINT] = v + 0.5
    js = RP.joint_probe_summary_r7(jst, rd, drain)
    assert js["nodrain"]["total_median"] == 391.0 and js["all"]["total_median"] == 374.6
    assert js["probe_joint"]["bias"] == -16.4 and js["drain_excluded"] == [176, 192, 208]
    ref = RP.ref_probe_total([(g, dict(resid=v, post=0.0)) for g, v in R2_C73], drain)
    assert ref["nodrain"] is not None and ref["all"] is not None
    # the CPU run: the probe's drain rounds = the rounds that begin with a finished request (independent: _ps_after)
    ev = _probed("GUDCD", tmp_path_factory)
    fin, hist = ev["probe"]["finished"], ev["after_hist"]
    want = sorted(g for g in fin if g - 1 in hist and hist[g - 1] > 0)
    _ev("RECON7 FIX (CPU run): drain rounds %s; finished at round start %s" % (ev["probe"]["drain"], fin))
    assert ev["probe"]["drain"] == want and want, (fin, hist)


# ------------------------------------------------------------------ the counters on planted tensors
class _Meter:
    def __init__(self, host):
        self.on, self.host = True, set(host)


def test_skip_flags_synthetic():
    RP = _mod("r2_recon7_probe")
    H, B, T, NB, Ls = 2, 3, 4, 10, 2
    rec = RP.SkipRecord(Ls, NB)
    lm = torch.tensor([[[3, -1, 5, 12], [1, 2, -1, -1], [-1, -1, -1, 7]],
                       [[0, 4, -1, -1], [9, -1, -1, 6], [2, -1, -1, -1]]], dtype=torch.int64)
    skip = torch.zeros((H, B, T), dtype=torch.bool)
    skip[0, 0, 0] = skip[0, 0, 3] = skip[1, 1, 3] = True                # ids 3, 12 (out of range), 6
    rec.reset(5, skip)
    rec.record(1, lm, skip)
    assert int(rec.n_skip) == 3 and int(rec.oor) == 1
    assert bool(rec.blk[1, 0, 0, 3]) and bool(rec.blk[1, 1, 1, 6]) and int(rec.blk[1, :, :, :NB].sum()) == 2
    assert rec.clr[1].sum() == 3 and bool(rec.clr[1, 0, 0, 0]) and not bool(rec.clr[1, 0, 0, 2])
    lm2 = torch.full((H, B, T), -1, dtype=torch.int64)
    lm2[0, 0, 0] = 8                                                   # a later load into the cleared slot (0, 0, 0)
    rec.record(1, lm2, torch.zeros((H, B, T), dtype=torch.bool))
    assert not bool(rec.clr[1, 0, 0, 0]) and rec.clr[1].sum() == 2 and int(rec.n_skip) == 3

    class Lay:
        def __init__(self):
            self.cache_engine = type("E", (), {})()
            self.cache_engine._k_cpu = torch.zeros(4)
            self.cache_engine._k_gpu = torch.zeros(4)

    lays = [Lay() for _ in range(Ls)]
    rs = type("RS", (), {})()
    rs.ops = type("O", (), {})()
    calls = []
    rs.ops.kernels = dict(gather_k=lambda g, c, i, bs: calls.append(1))
    rs.cache = type("Cc", (), {})()
    rs.cache.layers = lays
    vs = type("VS", (), {})()
    vs.ext_k = torch.zeros(4)
    runner = type("R", (), {})()
    runner.rs, runner.vs = rs, vs
    runner.meter = _Meter([lay.cache_engine._k_cpu.data_ptr() for lay in lays])
    tap = RP.GatherTap(runner, rec)
    tap.attach()
    e = lays[1].cache_engine
    win = torch.full((H, B, T), -1, dtype=torch.int64)
    win[0, 0, 0], win[0, 0, 2], win[1, 1, 3], win[1, 2, 1] = 3, 5, 6, 3   # 3 skipped at (0, 0); 6 skipped at (1, 1)
    win[0, 0, 3] = 8                                                   # a cleared slot, a block no draft skipped
    rs.ops.kernels["gather_k"](e._k_gpu, e._k_cpu, win, 64)            # window loads of layer 1
    st = torch.full((H, B, 5), -1, dtype=torch.int64)
    st[0, 0, 1], st[0, 0, 2], st[1, 2, 1] = 3, 9, 2
    rs.ops.kernels["gather_k"](vs.ext_k, e._k_cpu, st, 64)             # a store fill of layer 1
    runner.meter.on = False
    rs.ops.kernels["gather_k"](e._k_gpu, e._k_cpu, win, 64)            # meter off: not counted
    tap.detach()
    s = tap.state()
    assert len(calls) == 3 and s["calls"] == 2 and s["kinds"] == dict(win=1, store=1, other=0)
    assert (s["new_win"], s["new_store"]) == (5, 3)
    assert s["a_win"] == 2 and s["a_store"] == 1                       # 3 @ (0,0) and 6 @ (1,1); store 3 @ (0,0)
    assert s["b_win"] == 2 and s["ab_win"] == 1                        # cleared: (0,0,3), (1,1,3); (0,0,0) refilled
    assert s["st_sum"] == 3 and s["st_max"] == 2 and s["st_n"] == 2 and s["oor"] == 0
    with pytest.raises(RuntimeError):
        tap.attach()
        rs.ops.kernels["gather_k"] = lambda *a: None                   # another wrapper over the tap
        tap.detach()


# ------------------------------------------------------------------ STREAMS (b)
def test_streams_pure():
    S = _mod("r2_recon7_streams")
    vr = (ROOT / "nosi/nosi/verify_rows.py").read_text()
    assert "FORK_LAUNCH = dict(num_warps=4, num_stages=3, tile_size=64)" in vr and "FORK_ROWS = 4 " in vr
    lists = S.make_lists(5, 2)
    k = lists["page_key"]
    assert k.shape == (20, 2, 64) and k.dtype == torch.int32 and bool((k[..., 1:] > k[..., :-1]).all())
    assert torch.equal(lists["page_table"], k) and bool((S.union_count(lists, 5, 2) == 72).all())
    assert lists["last_len"].tolist()[:5] == [40, 41, 42, 43, 40]
    for B, ps, share in ((216, 1.226, 0.5), (318, 5.0, 0.5), (7, 0.3, 0.0)):
        w, s = S.counts_per_layer(B, 2, ps, share)
        assert w + s == round(ps * 2 * B)
        cw = S.spread(w, 2, B, 16, 1)
        cs = S.spread(s, 2, B, 16, 2)
        assert int(cw.sum()) == w and int(cs.sum()) == s and int(cw.max() - cw.min()) <= 1
        iw, is_ = S.make_ids(cw, 64, 16, 3), S.make_ids(cs, 129, 16, 4, 1)
        assert int((iw >= 0).sum()) == w and int((is_ >= 0).sum()) == s and bool((is_[..., 0] == -1).all())
        for ids in (iw, is_):
            for h in range(2):
                for b in range(B):
                    v = ids[h, b][ids[h, b] >= 0]
                    assert v.numel() == v.unique().numel() and bool((v < 16).all())
    sl = S.slice_lists(lists, 8, 16)                                   # request group [2, 4) of 5
    assert sl["page_key"].shape == (8, 2, 64) and sl["page_key"].is_contiguous() and sl["slots_per_entry"] == 1
    assert torch.equal(sl["page_key"], k[8:16]) and bool((S.union_count(sl, 2, 2) == 72).all())
    cfg = {c[0]: c for c in S.CONFIGS}
    assert S.BLOCK_BYTES == 32896 and cfg["SER"][2] == [["Aq", "G", "F", "Ar"]] and cfg["SER"][1] == "c"
    for c, ctl in S.CONTROL_OF.items():                                # every legal schedule has its matched control
        ops = sorted(p for seq in cfg[c][2] for p in seq)
        assert cfg[ctl][1] == "c" and len(cfg[ctl][2]) == 1 and sorted(cfg[ctl][2][0]) == ops, (c, ctl)
    assert cfg["GRPL"][3] is True and cfg["GRP"][3] is False and cfg["GRP"][2] == [S.CHAIN0, S.CHAIN1]
    for c in ("GRP", "GRPL", "PF"):                                    # no two parts of one layer's chain overlap
        lists_ = cfg[c][2]
        assert any(seq == S.CHAIN or seq in (S.CHAIN0, S.CHAIN1) for seq in lists_) and all(
            set(seq) & set(S.CHAIN) in (set(), set(S.CHAIN)) for seq in lists_), c
    assert {c[1] for c in S.CONFIGS} == {"a", "b", "c"} and set(S.PARTS) == {p for c in S.CONFIGS for seq in c[2]
                                                                             for p in seq}
    names = [c[0] for c in S.CONFIGS]
    assert [S.rotation(names, r)[0] for r in range(len(names))] == names and S.rotation(names, len(names)) == names
    t = {k: [10.0] * 3 for k in names}
    t.update(A=[60.0] * 3, F=[40.0] * 3, M=[20.0] * 3, G=[20.0] * 3, FG=[60.0] * 3, P=[20.0] * 3, SER=[120.0] * 3)
    t.update(**{"GRP-SER": [130.0] * 3, "GRP": [100.0] * 3, "GRPL": [118.0] * 3, "PF-SER": [140.0] * 3,
                "PF": [122.0] * 3})
    t["A|F|M"] = [64.0] * 3
    t["A|F"] = [100.0] * 3
    s = S.summarize(t)
    assert s["capability"] == "REACHABLE" and s["concurrent"]["A|F"]["eff"] == 0.0
    assert s["concurrent"]["A|F|M"]["eff"] == round((120 - 64) / 60.0, 4)
    at = s["attainable"]
    assert (at["GRP"]["split_cost_ms"], at["GRP"]["overlap_ms"], at["GRP"]["net_ms"], at["GRP"]["worth"]) == (
        10.0, 30.0, 20.0, True)
    assert at["GRPL"]["net_frac"] == round(2 / 120, 4) and at["GRPL"]["worth"] is False
    assert at["PF"]["hidden_frac"] == 0.9 and at["PF"]["chain_slowdown"] == round(122 / 120 - 1, 4) and at["PF"]["hides"]
    assert at["WINF"].startswith("NOT MEASURED")
    t["A|F|M"] = [110.0] * 3
    assert S.summarize(t)["capability"] == "SERIAL"
    t["A|F|M"] = [90.0] * 3
    t["PF"] = [135.0] * 3
    s = S.summarize(t)
    assert s["capability"] == "PARTIAL" and s["attainable"]["PF"]["hides"] is False
    assert S.need_bytes(318) > S.need_bytes(216) > 8e9
    a = S.parse_args(["--out", "x"])
    assert a.shapes == {128: 216, 73: 318} and a.new_blocks == {128: 1.226, 73: 5.0} and a.layers == 32
    assert (a.reps, a.warmup, a.seed, a.max_shape_s) == (10, 2, 7, 240.0)
    for bad in (["--shapes", "128:216", "--new-blocks", "73:5.0"], ["--reps", "2"], ["--win-share", "2"],
                ["--union-pages", "64"], ["--shapes", "x"], ["--shapes", "128:7", "--new-blocks", "128:1"],
                ["--max-shape-s", "0"]):
        with pytest.raises(SystemExit):
            S.parse_args(["--out", "x"] + bad)
    assert S.main(["--out", "x"]) == 20                                # PROTOCOL_STAMP_RECON7 is None: refused


class _HostDev:
    """The CPU stand-in of CudaDev: one host timeline, no concurrency."""
    import contextlib as _cl

    def __init__(self):
        self.main = "main"

    def stream(self):
        return object()

    def event(self):
        return [0.0]

    def ctx(self, s):
        return self._cl.nullcontext()

    def record(self, e, s):
        import time as _t
        e[0] = _t.perf_counter()

    def wait(self, s, e):
        pass

    def elapsed(self, a, b):
        return (b[0] - a[0]) * 1e3

    def sync(self):
        pass


class _RecDev(_HostDev):
    """_HostDev that logs every part call with its stream and the lock-step joins (the schedule's order)."""

    def __init__(self):
        super().__init__()
        self.cur, self.waits = "main", 0

    def ctx(self, s):
        dev = self

        class _C:
            def __enter__(self_):
                dev.cur = s

            def __exit__(self_, *a):
                dev.cur = "main"
        return _C()

    def wait(self, s, e):
        self.waits += 1


def _cpu_parts(plant=False, log=None, dev=None):
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(8, 16, generator=g), torch.randn(12, 16, generator=g)
    src = torch.randn(64, generator=g)
    parts, outs = {}, {}
    n = dict(F=0)
    for tag, rows in (("", slice(0, 8)), ("0", slice(0, 4)), ("1", slice(4, 8))):
        k = rows.stop - rows.start
        y1, y2, fo, go = torch.empty(k, 12), torch.empty(k, 12), torch.empty(4), torch.zeros(64)

        def F(fo=fo, tag=tag):
            n["F"] += 1
            fo.copy_(src[:4] * (2.0 if (plant and tag == "" and n["F"] > 1) else 1.0))
        parts.update({"Aq" + tag: (lambda y1=y1, rows=rows: torch.mm(x[rows], w.t(), out=y1)),
                      "Ar" + tag: (lambda y2=y2, rows=rows: torch.mm(x[rows], w.t(), out=y2)), "F" + tag: F,
                      "G" + tag: (lambda go=go: go.index_copy_(0, torch.tensor([3, 7]), src[:2]))})
        outs.update({"Aq" + tag: [y1], "Ar" + tag: [y2], "F" + tag: [fo], "G" + tag: [go]})
    for p in ("M", "P"):
        d = torch.empty(64)
        parts[p] = (lambda d=d: d.copy_(src))
        outs[p] = [d]
    if log is not None:
        for p, f in list(parts.items()):
            parts[p] = (lambda p=p, f=f: (log.append((dev.cur if dev else None, p)), f())[1])
    return parts, outs


def test_streams_measure_plumbing():
    S = _mod("r2_recon7_streams")
    names = [c[0] for c in S.CONFIGS]
    parts, outs = _cpu_parts()
    times, gates, stop = S.measure(_HostDev(), parts, outs, 2, 3, 1)
    assert stop is None and set(times) == set(names) and all(len(v) == 3 for v in times.values())
    assert [g[0] for g in gates] == ["%s-EXACT" % k for k in S.rotation(names, 2)] and all(g[2] for g in gates), gates
    s = S.summarize(times)
    assert set(s["concurrent"]) == set(S.CONCURRENT) and s["capability"] in ("REACHABLE", "SERIAL", "PARTIAL")
    assert set(s["attainable"]) == {"GRP", "GRPL", "PF", "WINF"}
    # the schedules: GRP's two group chains each in the joint pass's order on their own stream; GRPL joins per layer
    cfg = {c[0]: c for c in S.CONFIGS}
    for name, joins in (("GRP", 0), ("GRPL", 2 * 2 * (3 - 1))):       # 2 streams x 2 events x (layers - 1)
        log, dev = [], _RecDev()
        p2, o2 = _cpu_parts(log=log, dev=dev)
        side = [dev.stream() for _ in range(3)]
        S.run_config(dev, p2, cfg[name][2], 3, side, cfg[name][3])
        for st in {s_ for s_, _ in log}:
            seq = [p for s_, p in log if s_ is st]
            assert seq in (S.CHAIN0 * 3, S.CHAIN1 * 3), seq            # every layer's chain in order, per group
        assert dev.waits - 2 - 2 == joins, (name, dev.waits)           # start waits + end joins + lock-step joins
    # the order rotation: pass r starts at configuration r mod n
    order = []
    p3, o3 = _cpu_parts()
    real = S.run_config
    try:
        S.run_config = lambda dev, parts, spec, layers, side, lock=False: (order.append(spec), real(
            dev, parts, spec, layers, side, lock))[1]
        S.measure(_HostDev(), p3, o3, 1, 3, 1)
    finally:
        S.run_config = real
    n0 = len(S.PARTS) + len(names)                                     # the references, the warm-ups
    firsts = [order[n0 + r * len(names)] for r in range(3)]
    assert firsts == [cfg[names[r]][2] for r in range(3)]
    # STOP criteria: an EXACT failure ends the measurement (NEGATIVE CONTROL: F's later outputs differ)
    parts, outs = _cpu_parts(plant=True)
    times, gates, stop = S.measure(_HostDev(), parts, outs, 2, 3, 1)
    bad = [g[0] for g in gates if not g[2]]
    first_f = next(k for k in S.rotation(names, 2) if "F" in {p for seq in cfg[k][2] for p in seq})
    _ev("RECON7 X1 (CPU plumbing): %d configurations; planted F -> stop %r, failed %s" % (len(names), stop, bad))
    assert bad == ["%s-EXACT" % first_f] and stop == "EXACT failed in %s" % first_f
    tick = iter(range(0, 10 ** 6, 100))                                # a fake clock: 100 s per reading
    _, _, stop = S.measure(_HostDev(), *_cpu_parts(), 1, 3, 1, max_s=250, clock=lambda: next(tick))
    assert stop is not None and stop.startswith("wall clock")


# ------------------------------------------------------------------ MEM (c)
def test_mem_pure():
    M = _mod("r2_recon7_mem")
    a = M.parse_args(["--out", "x", "--workload", "w"])
    assert (a.batches, a.pool, a.steps, a.probe_rounds, a.margin_gb) == ([64, 192], 18, 512, 49, 0.10)
    for bad in (["--batches", "64"], ["--batches", "64,64"], ["--spec", "4"], ["--probe-rounds", "20"],
                ["--batches", "a"]):
        with pytest.raises(SystemExit):
            M.parse_args(["--out", "x", "--workload", "w"] + bad)
    assert M.main(["--out", "x", "--workload", "w", "--model", "m"]) == 20     # the protocol stamp: refused
    lines = M.lines_from([(64, 20.0 + 0.21 * 64), (192, 20.0 + 0.21 * 192)], [(64, 18.0 + 0.2 * 64),
                                                                              (192, 18.0 + 0.2 * 192)])
    assert abs(lines["prefill"][0] - 20.0) < 1e-6 and abs(lines["prefill"][1] - 0.21) < 1e-9
    f = M.fit_record(81, lines, 512, 0.10)
    want = int((84.0 - 0.10 - 20.0) / 0.21 + 1e-9)
    assert f["B_fit"] == min(want, f["int32_limit"]) and f["peak_at_anchor_GB"] == round(20.0 + 0.21 * 287, 4)
    assert f["anchor_fits"] is True and f["B_fit_usable"] == int((83.75 - 20.0) / 0.21) and set(f["margins"]) == {
        "0.00", "0.10", "0.25"}
    tap = M.OccTap(2, 5)

    class Lay:
        def __init__(self):
            self.cache_engine = type("E", (), {})()
            self.cache_engine._k_cpu = torch.zeros(4)
    runner = type("R", (), {})()
    runner.rs = type("RS", (), {})()
    runner.rs.ops = type("O", (), {})()
    runner.rs.ops.kernels = dict(gather_k=lambda *a: None)
    runner.rs.cache = type("C", (), {})()
    runner.rs.cache.layers = [Lay(), Lay()]
    runner.vs = type("V", (), {})()
    runner.vs.ext_k = torch.zeros(4)
    tap.attach(runner)
    ids = torch.full((2, 3, 5), -1, dtype=torch.int64)
    ids[0, 0, 1:4] = torch.tensor([3, 4, 5])
    ids[1, 2, 1] = 7
    ids[0, 1, 0] = 9                                                   # slot 0 (CB1) never counts
    lay1 = runner.rs.cache.layers[1].cache_engine
    runner.rs.ops.kernels["gather_k"](runner.vs.ext_k, lay1._k_cpu, ids, 64)
    runner.rs.ops.kernels["gather_k"](torch.zeros(4), lay1._k_cpu, ids, 64)    # not a store fill
    tap.detach()
    s = tap.state()
    assert s["fills"] == [0, 1] and s["max_by_layer"] == [0, 3] and s["max"] == 3 and s["hist"] == [4, 1, 0, 1]
    assert s["sum_by_layer"] == [0, 4] and s["streams_with_listed"] == [0, 2]
    rd = dict(tent=dict(calls=dict(joint_passes=1)))
    assert M.occ_gate("B64", s, rd, 2)[2] is False and M.occ_gate("B64", dict(s, fills=[1, 1]), rd, 2)[2] is True


# ------------------------------------------------------------------ CELL (a), the protocol, the step script, the notes
def test_recon7_cell_pure():
    C, RP = _mod("r2_recon7_cell"), _mod("r2_recon7_probe")
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.probe_every, a.profile_rounds, a.label_rounds, a.count_rounds,
            a.check_every) == (["L4PTFD", "L4PTFDGUDC", "L4PTFDGUDCD", "L4PTFDGUDCDP"], 216, 512, 65, 16, [96, 112],
                               [112], [48, 64, 144], 0)
    for arms in ("L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDP",
                 "L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDP,L4PTFDGUDKWCDP"):
        assert C.parse_args(base + ["--arms", arms]).arms == arms.split(",")
    for bad in (["--arms", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCDP"], ["--arms", "L4PTFD,L4PTFDGUDCDP"],
                ["--count-rounds", "96"], ["--profile-rounds", "17"], ["--label-rounds", "64"],
                ["--check-every", "1"], ["--profile-rounds", "16,32,48,64,80"], ["--count-rounds", "a"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    assert C.runner_choice("L4PTFDGUDKWCDP", True) == C.DCC.runner_choice("L4PTFDGUDKWCD", True)
    assert C.main(base + ["--model", "m"]) == 20                       # PROTOCOL_STAMP_RECON7 is None: refused
    lab, lay = RP.expected_labels_r7("jointg", 2, keep=True)
    assert lab == ["embed"] + list(RP.JOINT_GK) * 2 + ["head", "end"] and lay[-1] == 2
    assert RP.expected_labels_r7("draftg", 3)[0][:4] == ["static", "A", "resid", "B"]
    assert len(RP.expected_labels_r7("draftg", 32)[0]) == 3 * 32 + 3 and RP.expected_labels_r7("joint", 32)[0] == \
        RP.I4C.expected_labels("joint", 32)[0]
    ev = [dict(ph="X", cat="user_annotation", name="i4c.joint", ts=1000, dur=1000, pid=1, tid=1),
          dict(ph="X", cat="cuda_runtime", name="cudaGraphLaunch", ts=1100, dur=2, pid=1, tid=1,
               args=dict(correlation=1)),
          dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=1200, dur=2, pid=1, tid=1,
               args=dict(correlation=2)),
          dict(ph="X", cat="kernel", name="kernel_unified_attention_2d", ts=1300, dur=40, pid=0, tid=7,
               args=dict(correlation=1)),
          dict(ph="X", cat="kernel", name="flash_h2d_from_mask_kernel", ts=1350, dur=20, pid=0, tid=7,
               args=dict(correlation=1)),
          dict(ph="X", cat="kernel", name="ampere_bf16_s16816gemm", ts=1400, dur=100, pid=0, tid=7,
               args=dict(correlation=2)),
          dict(ph="X", cat="kernel", name="elementwise_kernel", ts=1450, dur=5, pid=0, tid=7)]   # no launch: unranged
    w = RP.work_counts(dict(traceEvents=ev), "i4c.joint")
    assert (w["gpu_ops"], w["fork_calls"], w["gemm_calls"], w["h2d_kernels"], w["h2d_ms"], w["graph_launches_traced"],
            w["unranged_ops"]) == (4, 1, 1, 1, 0.02, 1, 1), w
    assert RP.h2d_rate(1000, 32896, 2.0) == (32896000, 16.448) and RP.h2d_rate(None, 1, 1.0) == (None, None)
    c = RP.counted_record([dict(round=48, tap=dict(new_win=6, new_store=4, new_other=0, a_win=3, a_store=1, b_win=2,
                                                   ab_win=2, oor=0, calls=4, st_sum=4, st_max=2, st_n=3),
                                union=dict(u_sum=[144, 146], u_pos=[2, 2], npg_sum=[512, 512], npg_pos=[8, 8], u_max=74,
                                           layers=2), meter_new=10, skips=5, skip_oor=0, eager_layers=2,
                                recorded=True)], 2, 1, 2, 100)
    p = c["passes"][0]
    assert (p["new"], p["flagA"], p["flagA_share"], p["flagB_share_win"], p["union_pages_per_stream"],
            p["pages_per_row"], p["new_bytes"]) == (10, 4, 0.4, round(2 / 6, 4), 72.5, 64.0, 1000)
    passes = [dict(slot="commit", round=96, busy_ms=5.0, wall_ms=6.0, n_gpu_ops=100, same_round=True,
                   detail=dict(labels={k: dict(ms=1.0) for k in RP.COMMIT_LABELS}))]
    cr = RP.commit_record(passes)
    assert cr["phase_device_ms"] == 8.0 and cr["decision_device_ms"] == 2.0 and cr["commit_device_ms"] == 2.0


def test_protocol_state():
    P = _mod("r2_recon7_protocol")
    src = (EFF / "r2_recon7_protocol.py").read_text()
    assert ("\nPROTOCOL_STAMP_RECON7 = None\n" in src) == (P.PROTOCOL_STAMP_RECON7 is None)   # the step script's grep
    assert P.PROTOCOL_STAMP_RECON7 is None, "the builder never sets the stamp (the human approves the protocol)"
    assert P.stamp_ok() is False and P.refusal("x").startswith("REFUSED") and P.stamp_ok("2026-10-30T12:00:00Z")
    assert not P.stamp_ok("2026-10-30") and not P.stamp_ok(None)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_recon7_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("r7pm", 10, "r2_recon7_cell.py", 30, "100G", ["--batch 8", "--steps 72", "--probe-every 4",
                                                          "--count-rounds 16\\,20", "--profile-rounds 8\\,12"]),
         ("r7pa", 65, "r2_recon7_cell.py", 110, "240G", ["--batch 216", "--steps 512", "--count-rounds 48\\,64\\,144"]),
         ("r7pb", 10, "r2_recon7_cell.py", 160, "340G", ["--batch 318", "--arms L4PTFD\\,L4PTFDGUDKWC"]),
         ("r7s", 65, "r2_recon7_streams.py", 15, "64G", ["--shapes 128:216\\,73:318", "--reps 10", "--warmup 2",
                                                         "--seed 7", "--max-shape-s 240"]),
         ("r7mem", 18, "r2_recon7_mem.py", 70, "240G", ["--batches 64\\,192", "--probe-rounds 49", "--pool 18"]))


def test_step_script_recon7():
    src = (EFF / "r2_recon7_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src and "OUTROOT/sp_active_$JOB" in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_jit " in src
    assert "grep -q '^PROTOCOL_STAMP_RECON7 = None$'" in src
    for m in MODULES + ("r2_i4c_cell.py", "r2_i4m_cell.py", "r2_wsm_cell.py", "r2_i4dc_cell.py", "r2_p3_profile.py"):
        assert "benchmarks/Efficiency/%s" % m in src, m
    for bad in ("r7", "r7px", "i4dca", "r7mm"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    approved = "\nPROTOCOL_STAMP_RECON7 = None\n" not in _git("show", "%s:%s" % (sha, E_ + "r2_recon7_protocol.py")).stdout
    for cell, pool, py, smin, mem, wants in CELLS:
        r = _step(cell, sha)
        if r.returncode != 0 and "PROTOCOL_STAMP_RECON7 is None" not in r.stderr:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        if not approved:                                               # every cell STOPS until the human approves
            assert r.returncode == 1 and "PROTOCOL_STAMP_RECON7 is None" in r.stderr, (cell, r.stderr[-300:])
        else:
            assert r.returncode == 0, (cell, r.stderr[-300:])
        out = r.stdout.replace("\\", "")                               # printf %q may quote "," and ":"
        line = "cell %s: C %d (pool %d) " % (cell, 63 + pool, pool)
        assert line in out and "STEP_MIN %d mem %s" % (smin, mem) in out, (cell, out[-600:])
        for want in ["NOSI_POOL_BLOCKS=%d" % pool, STEP_ROOT + "/benchmarks/Efficiency/" + py, "--mem=%s" % mem,
                     "--time=%d" % smin, "recon7_t1_%s_%s" % (HOLDER, cell)] + [w.replace("\\", "") for w in wants]:
            assert want in out, (cell, want, out[-800:])
        assert ("protocol approved %d" % (1 if approved else 0)) in out
    _ev("RECON7 DRYRUN: %d cells printed their step command; protocol approved %s" % (len(CELLS), approved))


def test_notes_recon7():
    doc = (ROOT / "docs/NOTES_recon7.md").read_text()
    for word in ("test_recon7.py", "PROTOCOL_STAMP_RECON7", "STEP_MIN", "NO NEW KERNEL", "r7pm", "r7pa", "r7pb", "r7s",
                 "r7mem", "falsifier", "decides", "MEASURED", "MODEL", "drain", "PEQ", "D1", "D2", "D3", "D5", "D6",
                 "D7", "GPU only"):
        assert word in doc, word
