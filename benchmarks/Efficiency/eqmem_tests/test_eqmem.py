"""CPU tests of the equal-memory headline harness (benchmarks/Efficiency/eqmem_core.py, eqmem_bench.py). No GPU, no
model, no engine import.

  EM1  cells: parsing, slots / usable capacity / knobs / lookahead, refusals
  EM2  the memory fit: the 2175389 readings, the least-squares refit (recovery and singular refusal), the per-stage
       accounting sums, the refit from measured rows
  EM3  the plan at 75 GB and N 512: each family's batch, maximality, the other bounds (host memory, int32 envelope),
       refusals (no batch fits, the total_cis headroom, N < 2)
  EM4  the workload: rows, reuse beyond 94 documents, the sha-checked loader and its refusal
  EM5  re-warm: lookahead 1 on 1000 tokens (every 64 tokens and the width); lookahead U keeps every round of up to U
       tokens inside one shape (the matched negative: lookahead 1 lets a round cross)
  EM6  metrics: arm_metrics, the block-bootstrap CI, repetitions, the memory verdict, prefix agreement
  EM7  the comparison: best per family, speedups, over-budget and failed cells excluded, mixed comparisons refused
  EM8  the driver on the CPU: plan mode, --out exists, the knob refusal, report mode over synthetic results

From the worktree root:
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \\
        "benchmarks/Efficiency/eqmem_tests/test_eqmem.py::<node>"
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch

EFF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EFF))
sys.dont_write_bytecode = True

import eqmem_bench as EB  # noqa: E402
import eqmem_core as EC  # noqa: E402


# ------------------------------------------------------------------------------------------------------------ EM1
def test_EM1_cells():
    o, c, s, s30 = (EC.parse_cell(x) for x in ("original", "cached:50", "spec:2", "spec:2:30"))
    assert (o.slots, o.usable, o.name, o.lookahead) == (64, 63, "original", 1)
    assert (c.slots, c.usable, c.name, c.env()) == (114, 113, "cached_p50", dict(NOSI_POOL_BLOCKS="50",
                                                                                   NOSI_VERIFY_ROUND_SLOTS="0"))
    assert (s.slots, s.R, s.K, s.lookahead, s.name) == (128, 62, 2, 3, "spec_k2_r62")
    assert s.env() == dict(NOSI_POOL_BLOCKS="0", NOSI_VERIFY_ROUND_SLOTS="62") and s30.slots == 96
    for bad in ("cached:0", "spec:0", "foo", "cached:x", "spec:1:0", "original:1", "cached", "spec:1:2:3"):
        with pytest.raises(ValueError):
            EC.parse_cell(bad)


# ------------------------------------------------------------------------------------------------------------ EM2
def test_EM2_memory_fit():
    fit = EC.MemFit()
    res = [abs(fit.alloc_gb(EC.Cell("cached", P=P) if P else EC.Cell("original"), B) - gb) for B, P, gb in EC.FRONTIER_2175389]
    assert max(res) <= 0.15, res
    assert res[0] <= 0.02 and res[1] <= 0.02 and res[4] <= 0.02                     # the three anchor readings
    A, b, c, r = EC.fit_abc([(B, 64 + P, gb) for B, P, gb in EC.FRONTIER_2175389])
    assert r <= 0.25 and abs(A - fit.A) < 1.0 and abs(c - fit.c) < 2e-4
    rows = [(B, s, 20.0 + 0.03 * B + 0.002 * B * s) for B, s in ((10, 64), (100, 64), (100, 128), (300, 97), (50, 200))]
    A2, b2, c2, r2 = EC.fit_abc(rows)
    assert (round(A2, 6), round(b2, 6), round(c2, 6)) == (20.0, 0.03, 0.002) and r2 < 1e-9
    with pytest.raises(ValueError):
        EC.fit_abc([(10, 64, 1.0), (20, 64, 2.0), (40, 64, 3.0)])                    # B * slots proportional to B
    for cell, B in ((EC.Cell("original"), 300), (EC.Cell("cached", P=50), 191), (EC.parse_cell("spec:2"), 161)):
        st = EC.stage_breakdown(fit, cell, B)
        parts = sum(v for k, v in st.items() if k not in ("alloc", "used"))
        assert parts == pytest.approx(st["used"]) and st["used"] == pytest.approx(fit.used_gb(cell, B))
        assert st["alloc"] == pytest.approx(fit.alloc_gb(cell, B))
        assert (st["stage2_pool_slots"] > 0) == (cell.family == "cached")
        assert (st["stage3_round_slots"] > 0) == (st["stage3_pilot_scratch"] > 0) == (cell.family == "spec")
    meas = [dict(family="original", B=B, slots=s, alloc_peak_gb=20.0 + 0.03 * B + 0.002 * B * s,
                 reserved_peak_gb=(20.0 + 0.03 * B + 0.002 * B * s) * 1.05, nonpt_gb=n)
            for (B, s), n in zip(((10, 64), (100, 64), (100, 128), (300, 97)), (0.8, 1.1, 0.9, 1.2))]
    meas.append(dict(family="spec", B=100, slots=128, alloc_peak_gb=20.0 + 3.0 + 0.012 * 100 + 0.03 * 100 + 0.002 * 12800,
                     reserved_peak_gb=(20.0 + 3.0 + 1.2 + 3.0 + 25.6) * 1.05, nonpt_gb=1.0))
    f2 = EC.fit_from_rows(meas)
    assert (round(f2.A, 6), round(f2.b, 6), round(f2.c, 6)) == (20.0, 0.03, 0.002)
    assert f2.reserve_frac == pytest.approx(0.05) and f2.nonpt_gb == 1.2 and f2.spec_fixed_gb == pytest.approx(3.0)
    assert "refit over 4 measured rows" in f2.source


# ------------------------------------------------------------------------------------------------------------ EM3
def test_EM3_plan():
    fit = EC.MemFit()
    want = {"original": 304, "cached:50": 191, "cached:33": 219, "spec:2": 161}
    for text, B in want.items():
        cell = EC.parse_cell(text)
        p = EC.plan_cell(cell, 75.0, 512, fit)
        assert (p["B"], p["bound_by"]) == (B, "memory"), (text, p["B"], p["bounds"])
        assert p["predicted_used_gb"] <= 75.0 - 1.0 < fit.used_gb(cell, B + 1)          # the largest that fits
        assert p["max_gen"] == EC.max_gen_for(512, cell.K) and p["stages"]["used"] == pytest.approx(p["predicted_used_gb"])
    assert EC.max_gen_for(512) == 576 and EC.max_gen_for(512, 2) == 640
    big = EC.plan_cell(EC.Cell("original"), 200.0, 512, fit, host_budget_gb=1000.0)
    assert big["bound_by"] == "int32_host" and big["B"] == (2 ** 31 - 1) // ((16128 + 576) * 256) == 502
    host = EC.plan_cell(EC.Cell("original"), 200.0, 512, fit)
    assert host["bound_by"] == "host_memory" and host["predicted_host_gb"] <= 220.0
    assert EC.plan_cell(EC.Cell("original"), 75.0, 512, fit, b_cap=64)["bound_by"] == "cap"
    with pytest.raises(ValueError):
        EC.plan_cell(EC.Cell("original"), 20.0, 512, fit)                               # no batch fits
    with pytest.raises(ValueError):
        EC.plan_cell(EC.parse_cell("spec:8"), 75.0, 1016, fit)                          # N + K + 1 > 1024
    with pytest.raises(ValueError):
        EC.plan_cell(EC.Cell("original"), 75.0, 1, fit)
    a, b = EC.plan_cell(EC.Cell("original"), 75.0, 512, fit), EC.plan_cell(EC.parse_cell("spec:3"), 75.0, 512, fit)
    assert EC.plan_key(a) == EC.plan_key(b)


# ------------------------------------------------------------------------------------------------------------ EM4
def _workload(tmp: Path, rows=3, corrupt=None):
    data, docs = bytearray(), []
    for i in range(rows):
        row = torch.arange(i, i + EC.CTX, dtype=torch.int32).numpy().astype("<i4").tobytes()
        data += row
        docs.append(dict(index=10 + i, n_tokens=20000, sha256=hashlib.sha256(row).hexdigest()))
    if corrupt is not None:
        docs[corrupt]["sha256"] = "0" * 64
    (tmp / "tok.bin").write_bytes(bytes(data))
    (tmp / "man.json").write_text(json.dumps(dict(ctx=EC.CTX, dtype="int32-le", shape=[rows, EC.CTX], docs=docs)))
    return tmp / "tok.bin", tmp / "man.json"


def test_EM4_workload(tmp_path):
    d = EC.doc_rows(100, 94)
    assert d["rows"][93] == 93 and d["rows"][94] == 0 and (d["distinct"], d["reused"], d["repeats"]) == (94, True, 6)
    d2 = EC.doc_rows(64, 94, offset=90)
    assert d2["rows"][:5] == [90, 91, 92, 93, 0] and (d2["distinct"], d2["reused"]) == (64, False)
    with pytest.raises(ValueError):
        EC.doc_rows(0, 94)
    tok, man = _workload(tmp_path)
    ids, wl = EB.load_workload(tok, man, [2, 0, 2, 1])
    assert ids.shape == (4, EC.CTX) and int(ids[0, 0]) == 2 and int(ids[2, -1]) == 2 + EC.CTX - 1 and wl["n_rows"] == 3
    assert [x["row"] for x in wl["docs"]] == [0, 1, 2]
    bad = tmp_path / "bad"
    bad.mkdir()
    tb, mb = _workload(bad, corrupt=1)
    with pytest.raises(EB.Refused, match=r"^row 1 sha256 "):
        EB.load_workload(tb, mb, [0, 1])
    EB.load_workload(tb, mb, [0, 2])                                                  # an unused bad row is not read


# ------------------------------------------------------------------------------------------------------------ EM5
def _engine_run(lookahead_rule, U_max, seed=0, n_tokens=1000, n0=32, m0=1007):
    """Rounds of 1 .. U_max tokens on the engine's length rules: before a round, re-warm (one shipped token each) while
    the rule asks; a round's tokens must all have the shapes of the last warm-up (else ValueError, as the engine's
    score_buf.copy_ / pooled-block count would break)."""
    rng = random.Random(seed)
    s, n, m = 16128, n0, m0
    warm = None
    tokens, rewarms = 0, []
    while tokens < n_tokens:
        need, why = lookahead_rule(s, n, m, warm)
        if need:
            warm = EC.warm_state(s, n, m)
            rewarms.append((tokens, why))
            m, n = EC._advance(n, m)
            s += 1
            tokens += 1
            continue
        c = rng.randint(1, U_max)
        for _ in range(c):
            if EC.shape_at(s, n, m) != (warm["out_len"], warm["width"]):
                raise ValueError("a round token at %d needs %s, warmed %s" % (s, EC.shape_at(s, n, m), warm))
            m, n = EC._advance(n, m)
            s += 1
            tokens += 1
    return rewarms


def test_EM5_rewarm_lookahead():
    rw = _engine_run(lambda s, n, m, w: EC.rewarm_need(s, n, m, w, 1), 1)
    whys = [w for _, w in rw]
    assert whys[0] == "first" and whys.count("width") >= 1
    assert all(t in [x for x, _ in rw] for t in range(64, 1000, 64))
    for U in (2, 3, 4):
        for seed in range(5):
            _engine_run(lambda s, n, m, w, U=U: EC.rewarm_need(s, n, m, w, U), U, seed=seed)
    with pytest.raises(ValueError, match="^a round token at "):
        for seed in range(5):
            _engine_run(lambda s, n, m, w: EC.rewarm_need(s, n, m, w, 1), 3, seed=seed)
    sim = EC.simulate(16128, 32, 1007, 20)
    assert sim[1] == (16129, 17, 1008) and sim[16] == (16144, 32, 1008) and sim[17] == (16145, 17, 1009)


# ------------------------------------------------------------------------------------------------------------ EM6
def test_EM6_metrics():
    m = EC.arm_metrics(4, 10, 100.0, 130.0, 160.0, [1000.0, 2000.0, 3000.0, 4000.0], 4100.0, [1.0] * 9, [50.0] * 9,
                       loads=1000, wait_ms=900.0, n_boot=200)
    assert m["e2e_tok_s"] == pytest.approx(40 / 60.0) and m["decode_tok_s"] == pytest.approx(36 / 30.0)
    assert m["decode_tok_s_gpu_units"] == pytest.approx(36 / 0.45) and m["decode_tok_s_ci"] == [pytest.approx(80.0)] * 2
    assert m["transferred"]["bytes"] == 1000 * 32768 and m["transferred"]["bytes_per_committed_token"] == 1000 * 32768 / 36
    assert m["data_wait"]["share_of_decode"] == pytest.approx(0.9 / 30.0)
    assert m["latency_s"]["first_token"] == pytest.approx(4.1) and m["latency_s"]["own_prefill_max"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        EC.arm_metrics(4, 10, 100.0, 90.0, 160.0, [], 0.0, [1.0], [1.0], 0, 0.0)
    rnd = random.Random(3)
    ms = [50.0 + rnd.uniform(-10, 10) for _ in range(200)]
    lo, hi = EC.ratio_ci([2.0] * 200, ms, n_boot=500, seed=1)
    est = 400.0 / sum(ms) * 1000.0
    assert lo <= est <= hi and hi - lo > 0 and EC.ratio_ci([2.0] * 200, ms, n_boot=500, seed=1) == (lo, hi)
    assert EC.reps_summary([5.0])["ci_lo"] is None
    s3 = EC.reps_summary([1.0, 2.0, 3.0])
    assert s3["mean"] == 2.0 and s3["sd"] == pytest.approx(1.0) and s3["ci_hi"] - 2.0 == pytest.approx(4.303 / 3 ** 0.5)
    samples = [dict(used_gb=10.0, reserved_gb=8.5), dict(used_gb=70.0, reserved_gb=68.0)]
    ok = EC.memory_record(60.0, 72.0, samples, 75.0)
    assert ok["nonpt_gb"] == pytest.approx(2.0) and ok["used_peak_est_gb"] == pytest.approx(74.0) and ok["verdict"] == "VALID"
    assert EC.memory_record(60.0, 73.5, samples, 75.0)["verdict"] == "OVER_BUDGET"
    pa = EC.prefix_agreement([[1, 2, 3], [1, 2, 9], [4]], [[1, 2, 3, 4], [1, 2, 3], [4], [7]])
    assert pa["requests"] == 3 and pa["full_agreement"] == 2 and pa["prefix_min"] == 2


# ------------------------------------------------------------------------------------------------------------ EM7
def _res(cell, family, e2e, dec, verdict="VALID", key=(75.0, 1.0, 512, 16128), batch=100, error=None):
    r = dict(cell=cell, family=family, plan_key=list(key), memory=dict(verdict=verdict, used_peak_est_gb=74.0),
             metrics=dict(batch=batch, e2e_tok_s=e2e, decode_tok_s=dec, decode_tok_s_gpu_units=dec,
                          transferred=dict(bytes_per_committed_token=1e6), data_wait=dict(share_of_decode=0.1),
                          latency_s=dict(end_to_end=600.0), e2e_s=600.0))
    if error:
        r["error"] = error
    return r


def test_EM7_comparison():
    res = [_res("original", "original", 100.0, 1700.0), _res("original", "original", 110.0, 1800.0),
           _res("cached_p50", "cached", 150.0, 2700.0), _res("cached_p33", "cached", 160.0, 2600.0),
           _res("cached_p65", "cached", 999.0, 9999.0, verdict="OVER_BUDGET"),
           _res("spec_k2_r62", "spec", 90.0, 1500.0), _res("spec_k3_r62", "spec", 0.0, 0.0, error=dict(type="X"))]
    cmp_ = EC.comparison(res)
    assert {f: b["cell"] for f, b in cmp_["best"].items()} == dict(original="original", cached="cached_p33",
                                                                    spec="spec_k2_r62")
    assert cmp_["best"]["original"]["mean"] == pytest.approx(105.0) and cmp_["best"]["original"]["reps"] == 2
    assert cmp_["speedup_e2e_tok_s"]["cached_vs_original"] == pytest.approx(160.0 / 105.0 - 1)
    assert cmp_["speedup_e2e_tok_s"]["spec_vs_cached"] == pytest.approx(90.0 / 160.0 - 1)
    assert cmp_["speedup_decode_tok_s"]["cached_vs_original"] == pytest.approx(2700.0 / 1750.0 - 1)
    assert cmp_["cells"]["cached_p65"]["over_budget"] == 1 and cmp_["cells"]["spec_k3_r62"]["errors"] == 1
    assert cmp_["cells"]["original"]["e2e_tok_s"]["n"] == 2
    with pytest.raises(ValueError, match="^results of different comparisons mixed"):
        EC.comparison(res + [_res("original", "original", 1.0, 1.0, key=(70.0, 1.0, 512, 16128))])


# ------------------------------------------------------------------------------------------------------------ EM8
def test_EM8_driver_cpu(tmp_path, monkeypatch):
    out = tmp_path / "plan.json"
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "original,cached:50,spec:2", "--out", str(out)]) == 0
    plans = {p["cell"]: p["B"] for p in json.loads(out.read_text())["plans"]}
    assert plans == {"original": 304, "cached_p50": 191, "spec_k2_r62": 161}
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--out", str(out)]) == 2
    fitj = tmp_path / "fit.json"
    fitj.write_text(json.dumps(dict(nonpt_gb=4.5)))
    out2 = tmp_path / "plan2.json"
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "original", "--fit", str(fitj), "--out", str(out2)]) == 0
    assert json.loads(out2.read_text())["plans"][0]["B"] < 304
    tok, man = _workload(tmp_path)
    arm = ["--mode", "arm", "--budget-gb", "75", "--cell", "cached:50", "--model", "m", "--tokens", str(tok),
           "--manifest", str(man), "--out", str(tmp_path / "arm")]
    monkeypatch.setenv("NOSI_POOL_BLOCKS", "0")
    monkeypatch.setenv("NOSI_VERIFY_ROUND_SLOTS", "0")
    assert EB.main(arm) == 4 and not (tmp_path / "arm").exists()                        # the knob refusal
    assert EB.main(["--mode", "arm", "--budget-gb", "75", "--cell", "original", "--out", str(tmp_path / "x")]) == 4
    assert EB.main(arm[:3] + ["20"] + arm[4:]) == 4                                       # no batch fits
    # report over synthetic results: two cells, the original's tokens as the agreement reference
    run = tmp_path / "run"
    for name, fam, e2e, B in (("original_rep0", "original", 100.0, 4), ("cached_p50_rep0", "cached", 150.0, 3)):
        d = run / name
        d.mkdir(parents=True)
        r = _res(name.rsplit("_rep", 1)[0], fam, e2e, e2e * 10, batch=B)
        r.update(status="done", rep=0, plan=dict(budget_gb=75.0, N=512))
        (d / "result.json").write_text(json.dumps(r))
        seq = torch.arange(B * 6).view(B, 6) % 5
        if fam == "cached":
            seq[1, 3] = 4 if int(seq[1, 3]) != 4 else 3
        torch.save(dict(seq=seq, rows=list(range(B))), d / "tokens.pt")
    rep_out = tmp_path / "rep"
    assert EB.main(["--mode", "report", "--results", str(run), "--out", str(rep_out)]) == 0
    rep = json.loads((rep_out / "report.json").read_text())
    assert rep["best"]["cached"]["cell"] == "cached_p50" and rep["speedup_e2e_tok_s"]["cached_vs_original"] == pytest.approx(0.5)
    assert rep["agreement_reference"] == "original_rep0"                                 # v2: the first completed original
    ag = rep["agreement_vs_original"]["cached_p50_rep0"]
    assert ag["requests"] == 3 and ag["full_agreement"] == 2 and ag["prefix_min"] == 3
    assert rep["derived_workload"]["original"]["batches"] == 1 and rep["derived_workload"]["cached"]["batches"] == 2
    assert "LOCKSTEP" in (rep_out / "report.md").read_text()
