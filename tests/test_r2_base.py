"""CPU tests of the r2base throughput cell (benchmarks/Efficiency/r2_base_cell.py). Design: docs/R2_BASE.md.

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-base python -m pytest -q tests/test_r2_base.py

B1  the shipped re-warm schedule at S 16128, N 512: exactly steps 0 (first), 64, 128, 192, 256 (out_len), 288 (the
    128-column score width), 320, 384, 448 (out_len): the registered P2.
B2  warm_state / rewarm_need on hand cases (the rule of the sustained harness).
B3  period_stats / rates on a hand case.
B4  agreement: teacher-forced fraction, per-request first divergence, the unique-document subset.
B5  verdicts: P1-P5 limits at B 64 / 128 / 219.
B6  doc_rows (B > 94 repeats rows), parse_args refusals.
B7  the R2 arm's teacher-forced driver on the CPU fake world (tests/r2_fake_world.py, eager): fed the validated eager
    path's own free-running tokens, it predicts every next token exactly (agreement 1.0, no divergence, no refusal
    bit); NEGATIVE CONTROLS: random input tokens are NOT predicted (agreement < 0.5); a corrupted reference token is
    found at exactly its step.
GPU-only (the cell): the shipped arm, the capture and the replay, the real kernels and the timings.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
EFF = NOSI_REPO / "benchmarks/Efficiency"
if not (EFF / "r2_base_cell.py").is_file() or not (NOSI_REPO / "nosi/nosi/r2_capture.py").is_file():
    pytest.skip("no r2_base_cell.py / r2_capture.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)
sys.path.insert(0, str(EFF))
import r2_base_cell as BC   # noqa: E402


def _load_world():
    name = "r2_fake_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / "r2_fake_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


EXPECTED_16128 = [(0, "first"), (64, "out_len"), (128, "out_len"), (192, "out_len"), (256, "out_len"),
                  (288, "width"), (320, "out_len"), (384, "out_len"), (448, "out_len")]


def test_B1_the_shipped_rewarm_schedule_at_16128():
    assert BC.expected_rewarms(16128, 512) == EXPECTED_16128
    # every out_len change of the schedule is a 64-token boundary of the written position
    for t, why in BC.expected_rewarms(16100, 300):
        if why == "out_len":
            assert (16100 + t) % 64 == 0


def test_B2_rewarm_rule_hand_cases():
    w = BC.warm_state(16128, 16, 1007)
    assert w == dict(seq=16128, out_len=253, width=1024, m_next=1007)
    assert BC.rewarm_need(16128, 16, 1007, None) == (True, "first")
    assert BC.rewarm_need(16129, 17, 1007, w) == (False, "")
    assert BC.rewarm_need(16192, 17, 1011, w) == (True, "out_len")
    assert BC.warm_state(16400, 32, 1024)["width"] == 1152                 # the step's own compression counts
    assert BC.rewarm_need(16400, 32, 1024, dict(w, out_len=BC.warm_state(16400, 32, 1024)["out_len"])) == (True, "width")


def test_B3_period_stats_and_rates():
    periods = [100.0, 5.0, 6.0, 7.0, 4.0, 200.0]
    st = BC.period_stats(periods, [1, 2, 3, 4])
    assert st["n"] == 4 and st["median_ms"] == 5.5 and st["mean_ms"] == 5.5
    r = BC.rates(64, 6, periods, st)
    assert r["steady_step_ms"] == 5.5 and r["steady_tok_s"] == round(64e3 / 5.5, 1)
    assert r["per_request_tok_s"] == round(1e3 / 5.5, 3)
    assert r["total_s"] == 0.322 and r["effective_tok_s"] == round(1e3 * 64 * 6 / 322.0, 1)
    assert BC.period_stats(periods, [])["median_ms"] is None


def test_B4_agreement():
    inputs = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [7, 8, 9, 9, 9]])
    pred = torch.tensor([[2, 3, 4, 5], [2, 0, 4, 0], [8, 9, 9, 9]])
    ag = BC.agreement(pred, inputs, 2)
    assert ag["first_divergence"] == [None, 1, None]
    assert ag["n_diverged"] == 1 and ag["first_divergence_min"] == 1 and ag["tf_disagreements"] == 2
    assert ag["tf_agreement"] == pytest.approx(10 / 12) and ag["tf_agreement_unique_docs"] == pytest.approx(6 / 8)


def test_B5_verdicts():
    sh = dict(rewarms=[list(x) for x in EXPECTED_16128], rates=dict(steady_step_ms=100.0, effective_tok_s=100.0),
              gpu_peak_alloc_GB=50.0)
    good = dict(warmup=2, rates=dict(steady_step_ms=70.0, effective_tok_s=200.0), gpu_peak_alloc_GB=55.0)
    ag = dict(n_diverged=0, tf_agreement=1.0, n_requests=64, first_divergence_min=None)
    v = {n: verdict for n, verdict, _ in BC.verdicts(64, sh, good, ag, EXPECTED_16128)}
    assert v == dict(P1="CONFIRMED", P2="CONFIRMED", P3="CONFIRMED", P4="CONFIRMED", P5="CONFIRMED")
    bad = dict(warmup=2, rates=dict(steady_step_ms=85.0, effective_tok_s=140.0), gpu_peak_alloc_GB=60.0)
    v = {n: verdict for n, verdict, _ in BC.verdicts(64, dict(sh, rewarms=[[0, "first"]]), bad,
                                                     dict(ag, n_diverged=3, tf_agreement=0.99), EXPECTED_16128)}
    assert v == dict(P1="REFUTED", P2="REFUTED", P3="REFUTED", P4="REFUTED", P5="REFUTED")
    assert {n: x for n, x, _ in BC.verdicts(128, sh, good, ag, EXPECTED_16128)}["P3"] == "REPORT"
    v219 = {n: x for n, x, _ in BC.verdicts(219, sh, dict(good, rates=dict(steady_step_ms=89.0, effective_tok_s=1.0)),
                                            ag, EXPECTED_16128)}
    assert v219["P3"] == "CONFIRMED" and v219["P4"] == "REPORT"


def test_B6_doc_rows_and_arguments():
    rows = BC.doc_rows(219)
    assert len(rows) == 219 and rows[:3] == [0, 1, 2] and rows[94] == 0 and rows[218] == 218 - 188
    assert BC.doc_rows(64) == list(range(64))
    base = ["--out", "x", "--workload", "w", "--batch", "64"]
    a = BC.parse_args(base)
    assert (a.steps, a.warmup, a.arms) == (512, 2, ["SH", "R2"])
    for bad in (["--arms", "R2"], ["--arms", "R2,SH"], ["--warmup", "0"], ["--steps", "1001"]):
        with pytest.raises(SystemExit):
            BC.parse_args(base + bad)


def _free_running(FW, model, steps):
    """The validated eager path (r2_decode_path.decode_inference), all requests active: its token sequence (B, N+1)."""
    cache, st, tok = FW.build(model, horizon=steps)
    Bn = tok.shape[0]
    cu = torch.arange(Bn + 1, dtype=torch.int32)
    inputs = torch.zeros((Bn, steps + 1), dtype=torch.int64)
    inputs[:, 0:1] = tok
    for t in range(steps):
        st.set_active(torch.ones(Bn, dtype=torch.bool))
        pos = st.pos_host.view(Bn, 1).to(torch.int32)
        lg = FW.RDP.decode_inference(model, tok, cu, pos, cache)
        tok = lg[:, -1].argmax(-1, keepdim=True)
        inputs[:, t + 1:t + 2] = tok
    return inputs


def _teacher_forced(FW, model, inputs, steps):
    with torch.inference_mode():
        cache, st, _ = FW.build(model, horizon=steps)
        cb = FW.RC.CaptureBuffers(model, st, False, FW.STUBS, FW.WB.masked_writeback_torch)
        return BC.r2_teacher_forced(FW.RC, model, st, cb, cache, inputs, 2, False, BC.HostClock())


def test_B7_teacher_forced_driver_reproduces_the_eager_path():
    FW = _load_world()
    steps = 40
    model = FW.make_model()
    inputs = _free_running(FW, model, steps)
    res = _teacher_forced(FW, model, inputs, steps)
    ag = BC.agreement(res["pred"], inputs, FW.B)
    assert ag["tf_agreement"] == 1.0 and ag["n_diverged"] == 0, ag
    assert res["body_err"] == 0 and not any(res["twin_err"]) and len(res["periods"]) == steps
    assert res["pointers_changed"] is None and res["capture_ms"] is None
    # NEGATIVE CONTROL 1: random input tokens (seeded): the model does not predict a random next token, so the
    # agreement of the positive case is not vacuous (a random tiny model may cycle, so a shifted copy would not do)
    rnd = torch.randint(0, FW.V, tuple(inputs.shape), generator=torch.Generator().manual_seed(9))
    res2 = _teacher_forced(FW, model, rnd, steps)
    assert BC.agreement(res2["pred"], rnd, FW.B)["tf_agreement"] < 0.5
    # NEGATIVE CONTROL 2: a corrupted reference token is found at exactly its step
    bad = inputs.clone()
    bad[2, 17] = (bad[2, 17] + 1) % FW.V
    ag_bad = BC.agreement(res["pred"], bad, FW.B)
    assert ag_bad["first_divergence"][2] == 16 and ag_bad["n_diverged"] == 1
