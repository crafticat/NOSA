"""[acc216] CPU tests of the accuracy / exactness proof at batch 216 (analysis/evalfast/STATUS_accuracy_B216.md section 5):
r2_tick_sg5.py --store-committed-tokens and its gates (acc_gates), the scope REPORT of the accuracy record
(R_EF_accuracy_scope) and the step cells evalacc216m / evalacc216 of r2_tick_sg5_step.sh. No CUDA.

Each MUST gate has a NEGATIVE CONTROL: one planted change makes it fail. Lesson L49: a plant on a shared object can
cancel itself, so every plant goes into a FRESH fixture (no list is shared between the FAST point, the EXACT point and
the serial reference) and only at the site under test; each control also shows that the OTHER gates still pass.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_acc216.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
STEP = EFF / "r2_tick_sg5_step.sh"
ACC_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3ldx"                         # [f3ldx] (was nosi-r2-f3ld / -f3lia / -acc216)
# [f3lia] the commit the acc216 delta applies to in this tree: r2-f3lia's merge of flean + admitfill (3463341), the
# parent of the acc216 merge (was 8ccf262, acc216's own base: flean and admitfill changed make_engine / preflight /
# one_point after it, so the line compare is made against the tree that acc216 was merged into)
BASE_SHA = "346334116e779add7d4e77aff02e7ffaab25f62c"
sys.path.insert(0, str(EFF))


def _mod(name, file):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(EFF / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


EA = _mod("r2_evalacc", "r2_evalacc.py")
G = _mod("r2_tick_sg5_acc216_cpu", "r2_tick_sg5.py")
SG5_BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
ACC_OK = ["--modes", "fast,bi_c_strict", "--backends", "b2", "--layout", "pool", "--X", "64",
          "--store-committed-tokens"]


# =====================================================================================================================
# AC1 the flag parsing
# =====================================================================================================================
def test_AC1_flag_parsing():
    a = G.parse_args(SG5_BASE)                                          # the default: off, nothing else set
    assert (a.store_committed_tokens, a.serial_rows, a.gold_workload, a.link_record) == (False, 8, "", "")
    assert "bi_c_strict" in G.MODES and G.ACC_EXACT == "bi_c_strict"
    a = G.parse_args(SG5_BASE + ACC_OK + ["--serial-rows", "4", "--gold-workload", "gw", "--link-record", "l.json"])
    assert a.store_committed_tokens and a.modes == ["fast", "bi_c_strict"] and a.serial_rows == 4
    assert a.gold_workload == "gw" and a.link_record == "l.json"
    assert G.parse_args(SG5_BASE + ACC_OK + ["--serial-rows", "0"]).serial_rows == 0
    for bad in (["--serial-rows", "4"], ["--gold-workload", "gw"], ["--link-record", "l.json"]):   # need the flag
        with pytest.raises(SystemExit):
            G.parse_args(SG5_BASE + bad)
    for bad in (["--serial-rows", "-1"], ["--serial-rows", "65"]):     # 0 <= N <= --batch (64)
        with pytest.raises(SystemExit):
            G.parse_args(SG5_BASE + ACC_OK + bad)
    # the arms that compare and drop the tokens themselves: each base parses, the flag makes it refuse
    both = ["--backends", "b2", "--layout", "both", "--arrival-lag", "2", "--X", "64"]
    stk = ["--backends", "b2", "--stack", "full", "--stack-compare", "--arrival-lag", "2", "--sel-check", "--X", "64"]
    for base in (both, stk):
        G.parse_args(SG5_BASE + base)
        with pytest.raises(SystemExit):
            G.parse_args(SG5_BASE + base + ["--store-committed-tokens"])


# =====================================================================================================================
# AC2 flag off: unchanged
# =====================================================================================================================
def _base_source():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:benchmarks/Efficiency/r2_tick_sg5.py" % BASE_SHA],
                           capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return r.stdout if r.returncode == 0 else None


def _func(src, name):
    i = src.index("\ndef %s(" % name) + 1
    j = src.index("\n\n\n", i)
    return src[i:j]


def test_AC2_flag_off_is_unchanged():
    op = inspect.getsource(G.one_point)
    new = 'if fp is not None or getattr(a, "store_committed_tokens", False):'
    assert op.count(new) == 1 and op.count('info["committed_tokens"] = [list(r.out) for r in rnd.reqs]') == 1
    assert op.index(new) < op.index('info["committed_tokens"] = [list(r.out) for r in rnd.reqs]')
    assert G.parse_args(SG5_BASE).store_committed_tokens is False      # off: the condition is "fp is not None"
    src = (EFF / "r2_tick_sg5.py").read_text()
    main = _func(src, "main")
    assert "        acc = None\n        if a.store_committed_tokens:" in main
    assert "if acc is not None and ACC_EXACT in a.modes and a.serial_rows > 0:" in main
    i = main.index("        if acc is not None:   ")
    assert main.index("acc_gates(a, meta, res, gates, acc[", i) - i < 160   # the gates: the next statement
    # the serial reference reads the build's GPU windows: it runs BEFORE main frees them
    assert main.index("serial_reference(") < main.index("e._k_gpu = e._v_gpu = e._kv_bias_gpu = None")
    old = _base_source()
    if old is None:
        pytest.skip("no git object %s here: the line compare against the base needs it" % BASE_SHA)
    old_l, new_l = _func(old, "one_point").splitlines(), _func(src, "one_point").splitlines()
    assert len(old_l) == len(new_l)
    diff = [(x, y) for x, y in zip(old_l, new_l) if x != y]
    assert len(diff) == 1, diff                                       # one_point: the ONE condition line, no other
    assert diff[0][0].strip().startswith("if fp is not None:") and diff[0][1].strip().startswith(new)
    for f in ("make_engine", "run_draft_only", "preflight", "load_docs", "one_point_both", "one_point_stacks"):
        assert _func(old, f) == _func(src, f), f                        # no other path of the timed round changed


# =====================================================================================================================
# the CPU stand-ins of a finished run (fresh objects per call: L49)
# =====================================================================================================================
B_T, N_T, STEPS_T, SER_T = 10, 4, 6, 3          # rows, documents (rows b and b + 4, b + 8 share one), tokens, serial rows


def _toks(b):
    d = b % N_T
    return [(d * 7 + i * 3) % 50 for i in range(STEPS_T)]


def _docs(plant_row=None):
    base = torch.arange(N_T * 8, dtype=torch.int64).view(N_T, 8)
    docs = torch.cat([base] * 3)[:B_T].clone()
    if plant_row is not None:
        docs[plant_row, 0] += 1
    return docs


def _res(exact_counts=(9732, 420), fast_counts=(9732, 420)):
    fast = [_toks(b) for b in range(B_T)]
    fast[5][-1] = (fast[5][-1] + 1) % 50                         # FAST row 5 (a copy of row 1) differs: a REPORT only
    exact = [_toks(b) for b in range(B_T)]
    ser = [_toks(b) for b in range(SER_T)]                       # its own lists: a plant in EXACT never reaches it
    pt = lambda g, t, c: dict(gemm=g, label="SG5_%s_b2_fused_batched" % g, committed_tokens=t,   # noqa: E731
                              round_stats=dict(accepted=c[0], rejected=c[1]), committed=B_T * (STEPS_T - 1), ticks=61)
    return dict(points=[pt("fast", fast, fast_counts), pt("bi_c_strict", exact, exact_counts)],
                serial_reference=dict(rows=list(range(SER_T)), tokens=ser, steps=STEPS_T, gemm="bi_c_strict"))


def _gold(prefix_mismatch=(), unmapped=()):
    rows = {b: [(x + (1 if i % 2 else 0)) % 50 for i, x in enumerate(_toks(b))] for b in (0, 2, 3)}   # half match
    return dict(rows=rows, matched=3, doc_index=[0, 2, 3], prefix_mismatch=list(prefix_mismatch),
                unmapped=list(unmapped), sha_gold="x")


def _run(res, docs=None, gold=None, link=None, peak=83.3):
    docs = _docs() if docs is None else docs
    pairs = G.dup_pairs(B_T, N_T)
    gates = []
    G.acc_gates(SimpleNamespace(), dict(peak_gb=peak), res, gates, pairs, G.prompts_differ(docs, pairs), gold, link)
    return {g["name"]: g for g in gates}


MUSTS = ("ACC_exact_batch_invariance", "ACC_exact_vs_serial", "ACC_AE_fast_vs_exact")


def _only_failed(g, name):
    for n in MUSTS:
        assert g[n]["must"], n
        assert g[n]["passed"] == (n != name), (n, g[n]["detail"])


def test_AC3_the_pass_case_and_the_reports():
    assert G.dup_pairs(216, 94) == [(b % 94, b) for b in range(94, 216)] and len(G.dup_pairs(216, 94)) == 122
    assert G.dup_pairs(94, 94) == [] and G.dup_pairs(48, 94) == [] and G.dup_pairs(10, 0) == []
    assert G.dup_pairs(B_T, N_T) == [(0, 4), (1, 5), (2, 6), (3, 7), (0, 8), (1, 9)]
    g = _run(_res(), gold=_gold())
    for n in MUSTS:
        assert g[n]["must"] and g[n]["passed"], (n, g[n]["detail"])
    for n in ("R_ACC_fast_dup_rows", "R_ACC_fast_vs_exact_tokens", "R_ACC_fast_link", "R_ACC_peak"):
        assert not g[n]["must"] and g[n]["passed"], n
    assert "1 of 6 pairs differ" in g["R_ACC_fast_dup_rows"]["detail"]      # FAST row 5 vs row 1
    assert "0 of 6 pairs differ" in g["ACC_exact_batch_invariance"]["detail"]
    assert "9 of 10 rows fully equal" in g["R_ACC_fast_vs_exact_tokens"]["detail"]
    assert "no --link-record" in g["R_ACC_fast_link"]["detail"] and "inside" in g["R_ACC_peak"]["detail"]
    assert "OUTSIDE" in _run(_res(), peak=84.2)["R_ACC_peak"]["detail"]
    # gates run only when their inputs exist: a FAST-only run has no MUST gate
    r = _res()
    r["points"] = r["points"][:1]
    del r["serial_reference"]
    g = _run(r)
    assert not any(x in g for x in MUSTS) and "R_ACC_fast_link" in g
    # no repeated document: batch invariance is NOT tested, and says so (REPORT, never a silent pass)
    gates = []
    G.acc_gates(SimpleNamespace(), dict(peak_gb=1.0), _res(), gates, [], [], None, None)
    g = {x["name"]: x for x in gates}
    assert "ACC_exact_batch_invariance" not in g and "NOT tested" in g["R_ACC_exact_batch_invariance"]["detail"]


# =====================================================================================================================
# AC4 batch invariance (rows that hold the same document) and its negative controls
# =====================================================================================================================
def test_AC4_batch_invariance_negative_controls():
    r = _res()
    r["points"][1]["committed_tokens"][9][3] += 1                # EXACT row 9 = a copy of row 1 (not a serial row)
    _only_failed(_run(r, gold=_gold()), "ACC_exact_batch_invariance")
    r = _res()
    r["points"][1]["committed_tokens"][8][0] += 1                # the third copy of document 0
    g = _run(r, gold=_gold())
    _only_failed(g, "ACC_exact_batch_invariance")
    assert "1 of 6 pairs differ (first [(0, 8)])" in g["ACC_exact_batch_invariance"]["detail"]
    # the PRECONDITION: two rows that do NOT share a prompt cannot prove batch invariance, even with equal tokens
    g = _run(_res(), docs=_docs(plant_row=8), gold=_gold())
    _only_failed(g, "ACC_exact_batch_invariance")
    assert "NO, [(0, 8)]" in g["ACC_exact_batch_invariance"]["detail"]
    # a FAST difference in the same place is a REPORT only (FAST may add non-determinism)
    r = _res()
    r["points"][0]["committed_tokens"][9][3] += 1
    g = _run(r, gold=_gold())
    assert all(g[n]["passed"] for n in MUSTS) and "2 of 6 pairs differ" in g["R_ACC_fast_dup_rows"]["detail"]


# =====================================================================================================================
# AC5 the exact target distribution (EXACT rows = the BI serial decode) and its negative controls
# =====================================================================================================================
class SerialEnv:
    """The RealEnv interface that serial_reference uses (subset, frc, free) on a toy greedy decoder."""

    def __init__(self, fail=False):
        self.calls, self.freed, self.fail = [], 0, fail

    def subset(self, cache, st, rows):
        self.calls.append(("subset", cache, list(rows)))
        return dict(kind="copy"), SimpleNamespace(rows=list(rows))

    def frc(self, c, s, ops, first, steps):
        self.calls.append(("frc", list(first), steps, ops.mode))
        if self.fail:
            raise RuntimeError("planted frc failure")
        return [[f] + [(f * 3 + i) % 50 for i in range(1, steps)] for f in first]

    def free(self):
        self.freed += 1


def test_AC5_serial_reference_path():
    src = inspect.getsource(G.serial_reference)
    assert "EA.RealEnv()" in src and "env.subset(cache, st, rows)" in src and "env.frc(c, s, ops," in src
    frc = inspect.getsource(EA.RealEnv.frc)                     # the reference of EA_SG2_spec_equals_serial
    assert "attach_bi(st, ops)" in frc and "decode_loop" in frc
    env = SerialEnv()
    a = SimpleNamespace(serial_rows=3, steps=STEPS_T)
    first = list(range(20, 30))
    out = G.serial_reference(a, None, "CACHE", "ST", first, SimpleNamespace(mode="bi_c_strict"), env=env)
    assert env.calls == [("subset", "CACHE", [0, 1, 2]), ("frc", [20, 21, 22], STEPS_T, "bi_c_strict")]
    assert env.freed == 1 and out["rows"] == [0, 1, 2] and out["gemm"] == "bi_c_strict" and out["steps"] == STEPS_T
    assert out["tokens"][1] == [21] + [(21 * 3 + i) % 50 for i in range(1, STEPS_T)]
    env = SerialEnv(fail=True)                                  # the copy is freed when the decode raises
    with pytest.raises(RuntimeError):
        G.serial_reference(a, None, "CACHE", "ST", first, SimpleNamespace(mode="bi_c_strict"), env=env)
    assert env.freed == 1


def test_AC5b_exact_vs_serial_negative_controls():
    r = _res()
    r["serial_reference"]["tokens"][1][-1] += 1                  # the serial reference only
    g = _run(r, gold=_gold())
    _only_failed(g, "ACC_exact_vs_serial")
    assert "2 of 3 rows" in g["ACC_exact_vs_serial"]["detail"]
    r = _res()
    r["points"][1]["committed_tokens"][2][4] += 1               # EXACT row 2: a serial row AND a gold row, not a copy
    g = _run(r)                                                   # (no gold: the AE gate judges acceptance only)
    assert not g["ACC_exact_vs_serial"]["passed"]
    assert not g["ACC_exact_batch_invariance"]["passed"]          # row 2 has copies (6): both gates see it
    assert g["ACC_AE_fast_vs_exact"]["passed"]
    r = _res()
    r["serial_reference"]["gemm"] = "fast"                       # a reference in the wrong mode is no reference
    _only_failed(_run(r, gold=_gold()), "ACC_exact_vs_serial")
    r = _res()
    r["serial_reference"]["rows"], r["serial_reference"]["tokens"] = [], []   # no row compared: never a pass
    _only_failed(_run(r, gold=_gold()), "ACC_exact_vs_serial")


# =====================================================================================================================
# AC6 accuracy equivalence: r2_evalacc's band rule, and its negative controls
# =====================================================================================================================
def test_AC6_ae_band_rule_and_negative_controls():
    g = _run(_res(), gold=_gold())
    r = _res()
    _run(r, gold=_gold())
    v = r["acc"]["ae"]["verdict"]
    p, n = 9732 / 10152, 10152
    se = EA._binom_se(p, n, p, n)
    ok, allow = EA.inside_noise(0.0, 0.0, se, EA.BANDS["accept"])
    assert ok and v["criteria"]["accept"]["allow"] == pytest.approx(allow) and v["criteria"]["accept"]["noise"] == 0
    assert set(v["criteria"]) == {"accept", "gold"} and r["acc"]["ae"]["gold_rows"] == [0, 2, 3]
    assert r["acc"]["ae"]["fast"]["gold_n"] == 3 * STEPS_T and r["acc"]["ae"]["fast"]["gold_match"] == pytest.approx(0.5)
    assert "EQUIVALENT" in g["ACC_AE_fast_vs_exact"]["detail"]
    # NEGATIVE CONTROL 1: EXACT's counts only (an acceptance change of about -0.06)
    g = _run(_res(exact_counts=(9152, 1000)), gold=_gold())
    _only_failed(g, "ACC_AE_fast_vs_exact")
    assert "AFFECTS" in g["ACC_AE_fast_vs_exact"]["detail"] and "accept d" in g["ACC_AE_fast_vs_exact"]["detail"]
    # NEGATIVE CONTROL 2: FAST's tokens in the gold rows only (rows 0, 2, 3 have copies, so FAST REPORTs move too).
    # The shift is 2: the gold rows are the tokens + 1 at odd positions, so a shift of 1 would still match half of them
    r = _res()
    for b in (0, 2, 3):
        r["points"][0]["committed_tokens"][b] = [(x + 2) % 50 for x in r["points"][0]["committed_tokens"][b]]
    g = _run(r, gold=_gold())
    _only_failed(g, "ACC_AE_fast_vs_exact")
    assert r["acc"]["ae"]["verdict"]["failed"] == ["gold"] and r["acc"]["ae"]["fast"]["gold_match"] == 0.0
    # the same FAST plant without --gold-workload: acceptance only, so it is not seen (the gold criterion is what sees it)
    r2 = _res()
    for b in (0, 2, 3):
        r2["points"][0]["committed_tokens"][b] = [(x + 2) % 50 for x in r2["points"][0]["committed_tokens"][b]]
    assert _run(r2)["ACC_AE_fast_vs_exact"]["passed"] and set(r2["acc"]["ae"]["verdict"]["criteria"]) == {"accept"}
    # NEGATIVE CONTROL 3: the gold PRECONDITION (a prefix mismatch, an unmapped document, no gold row at all)
    for gold in (_gold(prefix_mismatch=[2]), _gold(unmapped=[7]), dict(_gold(), rows={})):
        _only_failed(_run(_res(), gold=gold), "ACC_AE_fast_vs_exact")


def _write_wl(d, stem, ctx, docs, rows):
    d.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(rows, dtype="<i4").tobytes()
    (d / (stem + "_tokens.bin")).write_bytes(raw)
    (d / (stem + "_manifest.json")).write_text(json.dumps(dict(ctx=ctx, shape=[len(docs), ctx],
                                                               docs=[dict(index=i) for i in docs])))
    (d / "SHA256SUMS").write_text("%s  %s\n" % (hashlib.sha256(raw).hexdigest(), stem + "_tokens.bin"))
    return d


def test_AC6b_gold_rows_maps_first_copies(tmp_path):
    S, extra = 64, 10
    g = np.random.default_rng(3)
    full = {i: g.integers(0, 50, size=S + extra) for i in range(N_T)}
    c = _write_wl(tmp_path / "c", "c16k", S, list(range(N_T)), [full[i][:S] for i in range(N_T)])
    grow = [full[i].copy() for i in (0, 2, 3)]
    gd = _write_wl(tmp_path / "g", "ctx%d" % (S + extra), S + extra, [0, 2, 3], grow)
    a = SimpleNamespace(workload=str(c), gold_workload=str(gd), S=S, steps=STEPS_T, batch=B_T)
    assert G.doc_count(a) == N_T
    prompts = torch.tensor(np.stack([full[i][:S] for i in range(N_T)]), dtype=torch.int64)
    docs = torch.cat([prompts] * 3)[:B_T]
    out = G.gold_rows(a, docs, N_T)
    assert sorted(out["rows"]) == [0, 2, 3] and out["matched"] == 3 and not out["unmapped"]
    assert not out["prefix_mismatch"]
    assert out["rows"][2] == [int(x) for x in full[2][S:S + STEPS_T]]       # the NEXT --steps tokens
    # NEGATIVE CONTROL: a gold row whose prefix differs is named (the AE gate then fails, AC6)
    grow[1][5] = (grow[1][5] + 1) % 50
    gd2 = _write_wl(tmp_path / "g2", "ctx%d" % (S + extra), S + extra, [0, 2, 3], grow)
    out2 = G.gold_rows(SimpleNamespace(**dict(vars(a), gold_workload=str(gd2))), docs, N_T)
    assert out2["prefix_mismatch"] == [2]
    # a prompt row that is not in the batch's first copies is unmapped (here: the batch holds documents 0 .. 1 only)
    out3 = G.gold_rows(SimpleNamespace(**dict(vars(a), batch=2)), docs[:2], N_T)
    assert sorted(out3["rows"]) == [0] and out3["unmapped"] == [2, 3]


# =====================================================================================================================
# AC7 the link to the timed runs (REPORT) and its negative control
# =====================================================================================================================
def test_AC7_link_report(tmp_path):
    rec = dict(results=dict(points=[dict(gemm="fast", label="SG5_fast_b2_fused_batched", committed=B_T * (STEPS_T - 1),
                                         ticks=61, round_stats=dict(accepted=9732, rejected=420, cohorts=1))]))
    p = tmp_path / "timed.json"
    p.write_text(json.dumps(rec))
    link = G.load_link(str(p))
    assert link["counts"] == dict(accepted=9732, rejected=420, committed=B_T * (STEPS_T - 1), ticks=61)
    g = _run(_res(), link=link)
    assert not g["R_ACC_fast_link"]["must"] and "EQUAL (this proof ran the timed path)" in g["R_ACC_fast_link"]["detail"]
    # NEGATIVE CONTROL: one count of the proof's FAST point
    g = _run(_res(fast_counts=(9731, 421)), link=link)
    assert "DIFFER" in g["R_ACC_fast_link"]["detail"] and g["R_ACC_fast_link"]["passed"]   # a REPORT, never a MUST
    # the record's number is read from the record, not from a constant
    rec["results"]["points"][0]["round_stats"]["accepted"] = 1
    p.write_text(json.dumps(rec))
    assert "DIFFER" in _run(_res(), link=G.load_link(str(p)))["R_ACC_fast_link"]["detail"]
    bad = G.load_link(str(tmp_path / "missing.json"))
    assert "error" in bad and "unreadable" in _run(_res(), link=bad)["R_ACC_fast_link"]["detail"]


# =====================================================================================================================
# AC8 the accuracy record's scope (REPORT in every --eval-role fast-time cell)
# =====================================================================================================================
def test_AC8_accuracy_scope_report(tmp_path):
    need = [dict(name=n, must=True, passed=True) for n in ("EA_VERDICT_FAST", "EA_NEG_detected", "EA_E_batch_invariance")]
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(dict(meta=dict(nosi_commit="0cc3cc8", args=dict(batch=48, alt_batch=8)), gates=need,
                                 results=dict(verdict=dict(fast_vs_exact="EQUIVALENT", scope="s")))))
    rec = G.check_accuracy_record(str(p))
    assert rec["batch"] == 48 and rec["alt_batch"] == 8 and rec["nosi_commit"] == "0cc3cc8" and rec["musts"] == 3
    with pytest.raises(SystemExit):                                     # the refusals are unchanged
        G.check_accuracy_record(str(tmp_path / "missing.json"))
    ok, text = G.accuracy_scope(rec, 216, "8ccf262")
    assert not ok and "SCOPE MISMATCH (REPORT, not MUST)" in text and "batch 48" in text and "batch 216" in text
    ok, text = G.accuracy_scope(dict(rec, nosi_commit="8ccf262"), 48, "8ccf262")
    assert ok and "MISMATCH" not in text
    ok, _ = G.accuracy_scope(dict(rec, nosi_commit="8ccf262"), 216, "8ccf262")   # the batch alone: a mismatch
    assert not ok
    assert not G.accuracy_scope(None, 216, "c")[0] and not G.accuracy_scope(dict(batch="48"), 48, "c")[0]
    # eval_gates writes it as a REPORT; a mismatch never fails a MUST gate
    a = G.parse_args(SG5_BASE + ["--eval-role", "fast-time", "--modes", "fast", "--backends", "b2", "--layout", "pool",
                                 "--X", "64", "--bare", "--mem-cap-gb", "84", "--accuracy-record", "r.json",
                                 "--batch", "4", "--steps", "9"])
    pt = dict(label="L", gemm="fast", bare_round=dict(ticks=10, committed=32, wall_s=0.8),
              bare_T_tick=dict(mean=50.0, median=40.0), round=dict(device_ms=dict(tick=dict(mean=60.0, median=50.0))),
              committed=32, ticks=10, wall_s=1.0)
    gates = []
    G.eval_gates(a, dict(peak_gb=83.0, accuracy_record=rec, nosi_commit="8ccf262"), dict(points=[pt]), gates)
    g = {x["name"]: x for x in gates}
    assert not g["R_EF_accuracy_scope"]["must"] and g["R_EF_accuracy_scope"]["passed"]
    assert "SCOPE MISMATCH" in g["R_EF_accuracy_scope"]["detail"]
    assert all(x["passed"] for x in gates if x["must"])


# =====================================================================================================================
# AC9 the step cells evalacc216m / evalacc216 (static, then a DRYRUN on a clean checkout of this ROOT)
# =====================================================================================================================
def test_AC9a_the_cells_static():
    text = STEP.read_text()
    lines = text.splitlines()
    root = next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0][len("NOSI_EVAL="):]
    assert root == ACC_ROOT
    assert "|evalfast2ia|evalacc216m|evalacc216|" in text              # [f3lia] the merged cells around it
    assert "        evalacc216m|evalacc216) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    blk = text[text.index("    evalacc216m|evalacc216) MODES="):]
    blk = blk[:blk.index("\nesac\n")]
    for w in ("MODES=fast,bi_c_strict", "BACKENDS=b2", "PC=96 ; PX=1512 ; PRV=1464", "BATCH=216", "MEMG=240G",
              "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL", 'if [ "$CELL" = evalacc216m ]; then STEPS=8',
              'LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2rd/summary.json")',
              "--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped",
              "--b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads --res-lean",
              "--res-dev --store-committed-tokens --serial-rows 8 --gold-workload \"$GOLD\" \"${LINKARG[@]}\""):
        assert w in blk, w
    for w in ("--eval-role", "--accuracy-record", "--mem-cap-gb", "--eval-bands", "--bare", "STEP_MIN="):
        assert w not in blk, w                                   # STEP_MIN: the default 120
    assert "GOLD_SHA=f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3" in text
    assert 'if [ -n "$ACC" ]; then' in text and "ctx65280_tokens.bin sha256 is not $GOLD_SHA" in text
    reg = text[text.index("# REGISTERED for evalacc216m / evalacc216"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T18:04:12Z, before any run", "ACC-BI", "0 differing pairs of 122", "ACC-ETD", "8 of 8 rows", "ACC-AE",
                "|d acceptance| <= 0.01", "ACC-LINK", "accepted 9732, rejected 420, committed 10152", "ticks 61",
                "ACC-AGREE", "ACC-FDUP", "ACC-MEM", "83.1-83.5 GB", "79.25 GiB = 85.10 GB", "B 188", "ACC-MINI",
                "NOT a falsifier", "SCOPE", "SCOPE MISMATCH (REPORT, not MUST)", "TIME", "nosi-r2-acc216"):
        assert tag in reg, tag
    assert reg.count("FALSIFIER") >= 5
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def _clean_head():
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--",
                                    "nosi"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        return None
    return head.stdout.strip()


@pytest.mark.parametrize("cell", ["evalacc216m", "evalacc216"])
def test_AC9b_the_cells_dryrun(tmp_path, cell):
    """On a clean checkout: the DRYRUN prints the cell's arguments, which the harness parses. In this ROOT's own
    checkout a DRYRUN stop FAILS (never a skip); elsewhere it skips."""
    head = _clean_head()
    if head is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    (tmp_path / "418.gpu_uuid").write_text("33cfb929-315c-8d27-5622-1169affb7e04\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    r = subprocess.run(["bash", str(STEP), "--nosi", head, "418", "t1", cell], env=env, capture_output=True, text=True,
                       timeout=180)
    if r.returncode != 0:
        assert Path(ACC_ROOT).resolve() != ROOT.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the acc216 ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--mem=240G", "--time=120", "--variants fused/batched",
              "PYTHONPATH=%s/nosi" % ACC_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell,
              "--gold-workload /mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280"):
        assert w in got or w in line, (cell, w)
    assert "[acc216] modes fast,bi_c_strict" in r.stdout
    pa = G.parse_args(got.split())
    assert pa.modes == ["fast", "bi_c_strict"] and pa.backends == ["b2"] and pa.layout == "pool"
    assert pa.store_committed_tokens and pa.serial_rows == 8 and pa.repeat_docs and pa.X_pool == 1512
    assert pa.eval_role == "" and not pa.bare and pa.mem_cap_gb == 0 and not pa.accuracy_record
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert pa.wait_index and pa.b2_fewer_reads and pa.res_lean and pa.res_dev
    assert pa.variant_list == [("fused", "batched")] and pa.arrival_lag == 2 and pa.log_slots == -1
    if cell == "evalacc216m":
        assert pa.steps == 8 and pa.link_record == ""
    else:
        assert pa.steps == 48
        assert pa.link_record == "/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_418_evalfast2rd/summary.json"
