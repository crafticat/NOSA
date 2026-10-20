"""CPU tests of the context-sweep harness (paper 4.2: benchmarks/Efficiency/ctxsweep_core.py, ctxsweep_bench.py,
ctxsweep_step.sh, ctxsweep_cells.txt, and the eqmem_bench additions). No GPU, no model, no engine import.

  CS1  contexts: the engine state after the prefill, the F1 window, re-warm at 65280 / 130816 (every 64 tokens, the
       width change at token 288; rounds of U tokens stay inside one shape; the matched negative: lookahead 1 with
       rounds of 3 / 5 tokens crosses)
  CS2  the S-aware GPU fit: the long-context peaks exact, the slot sweeps within 0.1 GB, the 16K frontier never
       under-read; the per-context refit (two batch sizes recover the truth, one batch size, the envelope over noisy
       rows, spec_fixed, the 4.1 rows)
  CS3  the host model: the 8 measured RSS rows are the power-of-two model (one base, ~17 GiB), not the nominal one;
       the 4.1 OOM; '2x' >= 'pow2'; the refit; the --mem need; the host bound
  CS4  the plan: each bound binds where it must and refuses a cell above it (int32_host, host_memory, memory,
       int32_gpu, cap) and each refusal (F3, rope, prefill, --mem, batch < 1); document reuse reported
  CS5  latency on one clock: shipped steps and spec rounds
  CS6  the cell list: parsed, every entry admitted, --mem covers the need, the minutes per tier, the time model
       against the 4.1 step; parse refusals
  CS7  the driver on the CPU: plan mode (exit 0 / 2 / 4, the refit from a run directory), arm refusals (another
       context's manifest, a batch above the bound, the knob, missing arguments), report mode (helps / hurts, equal
       memory, a mixed comparison refused per context, 4.1-style rows)
  CS8  host parsing: /proc/self/status, /proc/self/cgroup (v1, v2), pinned sizes, eqmem_bench.host_record
  CS9  the step script: the GPU uuid file (missing, unreadable, malformed: STOP 1; a good file passes to the next
       check; the DRYRUN-only override); the workload pins equal each SHA256SUMS

From the worktree root (inside the image; the 4.1 tests EM1-EM8 must pass too):
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/eqmem_tests
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

EFF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EFF))
sys.dont_write_bytecode = True

import ctxsweep_bench as CB  # noqa: E402
import ctxsweep_core as CS  # noqa: E402
import eqmem_bench as EB  # noqa: E402
import eqmem_core as EC  # noqa: E402

CTXS = (16128, 65280, 130816)


# ------------------------------------------------------------------------------------------------------------ CS1
def test_CS1_contexts_and_rewarm():
    assert {S: c.n_docs for S, c in CS.CONTEXTS.items()} == {16128: 94, 65280: 55, 130816: 24}
    for S in CTXS:
        assert S % 64 == 0 and CS.f1_window(S) == 64
        n0, m0 = CS.compressed_state(S)
        assert (n0, m0) == (16, S // 16 - 1) and m0 % 128 == 111
        rw = CS.rewarm_schedule(S, 1000, 1)
        pts = [t for t, _ in rw]
        assert rw[0] == (0, "first")
        assert all(t in pts for t in range(64, 1000, 64))                     # F1: every 64-token block
        assert (288, "width") in rw and [w for _, w in rw].count("width") == 1
        assert max(b - a for a, b in zip(pts, pts[1:])) <= CS.f1_window(S)
        assert len(CS.rewarm_schedule(S, 511, 1)) == 9
        for U in (2, 3, 4):
            CS.rewarm_schedule(S, 1000, U)                                     # rounds of U tokens: no crossing
        for c in (3, 5):
            with pytest.raises(ValueError, match="^token "):
                CS.rewarm_schedule(S, 1000, 1, commit=c)                       # lookahead 1 lets a round cross
    assert CS.f1_window(16100) == 64 - 16100 % 64


# ------------------------------------------------------------------------------------------------------------ CS2
def test_CS2_gpu_fit():
    f = CS.prior_fit()
    o = EC.Cell("original")
    for S, B, gb in CS.LONG_P0:
        assert f.at(S).alloc_gb(o, B) == pytest.approx(gb, abs=1e-6)
    for P, gb in CS.SLOT_SWEEP_64K:
        assert abs(f.at(65280).alloc_gb(EC.Cell("cached", P=P) if P else o, 64) - gb) <= 0.1
    for P, gb in CS.SLOT_PAIR_128K:
        assert abs(f.at(130816).alloc_gb(EC.Cell("cached", P=P) if P else o, 60) - gb) <= 0.1
    for B, P, gb in EC.FRONTIER_2175389:
        d = f.at(16128).alloc_gb(EC.Cell("cached", P=P) if P else o, B) - gb
        assert -1e-6 <= d <= 1.0, (B, P, d)                                    # the upper envelope at 16K
    assert f.A_at(16128) == f.A16 and f.b_at(16128) == f.b16 and f.calibrated(16128) == 0
    assert f.A16 < f.A_at(65280) < f.A_at(130816) and f.b16 < f.b_at(65280) < f.b_at(130816)
    assert f.a1 > 0 and f.a2 > 0
    assert CS.C_SLOT_BYTES == 2105344 and f.c == pytest.approx(0.0021133, abs=2e-6)
    assert CS.B1_BYTES == pytest.approx(1180.28125) and CS.SPEC_B1_BYTES == 1024
    assert f.spec_req_at(130816) - f.spec_req_at(16128) == pytest.approx(1024 * (130816 - 16128) / 1e9)
    assert f.spec_req_at(16128) == pytest.approx(0.012)

    def row(fam, S, B, slots, extra=0.0):
        a = 30.0 + 0.25 * B + f.c * B * slots + extra
        return dict(family=fam, S=S, B=B, slots=slots, alloc_peak_gb=a, reserved_peak_gb=a * 1.03, nonpt_gb=0.9)
    rows = [row("original", 130816, 4, 64), row("original", 130816, 16, 64), row("cached", 130816, 16, 114)]
    g = CS.fit_ctx_from_rows(rows, f)
    assert g.A_at(130816) == pytest.approx(30.0) and g.b_at(130816) == pytest.approx(0.25)
    assert g.calibrated(130816) == 2 and g.calibrated(65280) == 0
    assert g.A_at(65280) == pytest.approx(f.A_at(65280)) and g.b_at(65280) == pytest.approx(f.b_at(65280))
    assert g.reserve_frac == pytest.approx(0.03) and g.nonpt_gb == 0.9
    one = CS.fit_ctx_from_rows(rows[:1], f)
    assert one.b_at(130816) == pytest.approx(1.25 * f.b_at(130816)) and one.calibrated(130816) == 1
    assert one.A_at(130816) + 4 * one.b_at(130816) == pytest.approx(31.0)       # through the measured row
    noisy = []
    for B, e in ((8, 0.3), (16, -0.2), (32, 0.1), (32, -0.3)):
        noisy.append(row("original", 65280, B, 64, extra=e))
    h = CS.fit_ctx_from_rows(noisy, f)
    assert h.calibrated(65280) == 3
    for r in noisy:
        assert h.at(65280).alloc_gb(o, r["B"]) >= r["alloc_peak_gb"] - 1e-9    # never under a measured row
    sp = row("spec", 130816, 8, 128, extra=f.spec_req_at(130816) * 8 + 2.0)
    assert CS.fit_ctx_from_rows(rows + [sp], f).spec_fixed_gb == pytest.approx(2.0)
    h41 = [dict(family="cached", B=219, slots=97, alloc_peak_gb=68.155732992, reserved_peak_gb=68.620910592,
                nonpt_gb=0.8355840000000114),
           dict(family="cached", B=191, slots=114, alloc_peak_gb=68.712947712, reserved_peak_gb=69.275222016,
                nonpt_gb=0.8355840000000114)]                                    # job 2188818 h41 fit_row.json (no S)
    k = CS.fit_ctx_from_rows(h41, f)
    assert k.calibrated(16128) == 2
    for r in h41:
        assert k.at(16128).alloc_gb(EC.Cell("cached", P=r["slots"] - 64), r["B"]) >= r["alloc_peak_gb"] - 1e-9
        assert f.at(16128).alloc_gb(EC.Cell("cached", P=r["slots"] - 64), r["B"]) > r["alloc_peak_gb"]  # prior above


# ------------------------------------------------------------------------------------------------------------ CS3
def test_CS3_host_model():
    p2 = CS.HostFit(base_gib=0.0, mode="pow2")
    nom = CS.HostFit(base_gib=0.0, mode="nominal")
    two = CS.HostFit(base_gib=0.0, mode="2x")
    res = [gib - p2.host_gib(B, S, mg) for S, B, mg, gib, _ in CS.HOST_EVIDENCE]
    assert all(16.5 <= x <= 17.5 for x in res), res                            # one base explains all 8 rows
    unexplained = [gib - nom.host_gib(B, S, mg) - max(res) for S, B, mg, gib, _ in CS.HOST_EVIDENCE]
    assert max(unexplained) >= 100.0                                           # the nominal size does not
    for S, B, mg, gib, _ in CS.HOST_EVIDENCE:
        assert two.host_gib(B, S, mg) + max(res) >= gib                        # '2x' was conservative on every row
    S, B, mg, layers, limit = CS.HOST_OOM_4_1
    t = CS.kv_tensor_bytes(B, S, mg)
    assert t == 304 * 16704 * 512 and CS.pow2ceil(t) == 2 ** 32
    assert 2 * layers * CS.pow2ceil(t) / CS.GIB >= limit                       # pow2: 30 layers reach 240 GiB
    assert 2 * 32 * t / CS.GIB + max(res) < limit                              # nominal: never killed
    for S in CTXS:
        for B in list(range(1, 70)) + [127, 128, 300]:
            assert two.kv_gib(B, S, 576) >= p2.kv_gib(B, S, 576) >= nom.kv_gib(B, S, 576)
    assert [CS.pow2ceil(x) for x in (1, 2, 3, 4, 5, 2 ** 31, 2 ** 31 + 1)] == [1, 2, 4, 4, 8, 2 ** 31, 2 ** 32]
    m = dict(S=65280, B=8, max_gen=576, host_peak_gib=p2.host_gib(8, 65280, 576) + 15.0)
    h = CS.fit_host_from_rows([m])
    assert h.mode == "pow2" and h.base_gib == pytest.approx(15.0 + CS.HOST_REFIT_SLACK_GIB)
    assert CS.fit_host_from_rows([dict(S=65280, B=8, max_gen=576)]) == CS.HostFit()   # no measured host: the default
    assert CS.HostFit().mode == "2x" and CS.HostFit().base_gib == pytest.approx(24e9 / 2 ** 30)
    assert (CS.mem_need_gib(54.5), CS.mem_need_gib(10.0), CS.mem_need_gib(150.0)) == (70, 64, 170)
    hd = CS.HostFit()
    for S, mem in ((65280, 180), (130816, 440), (16128, 64)):
        b = CS.host_bound(hd, S, 576, mem, 4097)
        assert CS.mem_need_gib(hd.host_gib(b, S, 576)) <= mem < CS.mem_need_gib(hd.host_gib(b + 1, S, 576))


# ------------------------------------------------------------------------------------------------------------ CS4
def test_CS4_plan_bounds_and_refusals():
    f = CS.prior_fit()
    two, p2 = CS.HostFit(), CS.HostFit(base_gib=18.0, mode="pow2")
    o, spec2 = EC.Cell("original"), EC.parse_cell("spec:2")
    p = CS.plan_ctx_cell(o, 130816, 75.0, 512, f, p2, mem_gib=440)
    assert (p["B"], p["bound_by"]) == (63, "int32_host") == (p["B_max"], p["B_max_bound_by"])
    assert p["bounds"]["int32_host"] == (2 ** 31 - 1) // ((130816 + 576) * 256) == 63
    assert p["bounds"]["memory"] > 63 and p["host"]["need_gib"] <= 440 and p["host"]["kv_pow2_gib"] == 256.0
    assert p["f1_window"] == 64 and p["rewarms_shipped"] == 9 and p["S"] == 130816
    assert EC.plan_key(p) == (75.0, 1.0, 512, 130816)
    with pytest.raises(ValueError, match=r"^batch 64 > the largest admitted 63 \(bound_by int32_host"):
        CS.plan_ctx_cell(o, 130816, 75.0, 512, f, p2, mem_gib=440, batch=64)
    assert CS.plan_ctx_cell(o, 65280, 75.0, 512, f, p2, mem_gib=440)["bounds"]["int32_host"] == 127
    q = CS.plan_ctx_cell(o, 130816, 75.0, 512, f, two, mem_gib=440)               # '2x' until measured
    assert q["bound_by"] == "host_memory" and 40 <= q["B"] < 63
    with pytest.raises(ValueError, match="bound_by host_memory"):
        CS.plan_ctx_cell(o, 130816, 75.0, 512, f, two, mem_gib=180, batch=32)
    assert CS.plan_ctx_cell(o, 130816, 75.0, 512, f, two, mem_gib=180, batch=16)["host"]["need_gib"] <= 180
    s = CS.plan_ctx_cell(spec2, 65280, 75.0, 512, f, p2, mem_gib=440)
    assert s["bound_by"] == "memory" and s["B"] < s["bounds"]["int32_host"]
    assert s["predicted_used_gb"] <= 75.0 - 1.0 < f.at(65280).used_gb(spec2, s["B"] + 1)
    with pytest.raises(ValueError, match="bound_by memory"):
        CS.plan_ctx_cell(spec2, 65280, 75.0, 512, f, p2, mem_gib=440, batch=s["B"] + 1)
    g = CS.plan_ctx_cell(EC.parse_cell("cached:2000"), 16128, 1000.0, 512, f, p2, mem_gib=440)
    assert (g["bound_by"], g["B"]) == ("int32_gpu", (2 ** 31 - 1) // (2064 * 64 * 256))
    with pytest.raises(ValueError, match="bound_by int32_gpu"):
        CS.plan_ctx_cell(EC.parse_cell("cached:2000"), 16128, 1000.0, 512, f, p2, mem_gib=440, batch=g["B"] + 1)
    assert CS.plan_ctx_cell(o, 16128, 75.0, 512, f, p2, mem_gib=440, b_cap=16)["bound_by"] == "cap"
    with pytest.raises(ValueError, match="^F3: "):
        CS.plan_ctx_cell(EC.parse_cell("spec:8"), 65280, 75.0, 1016, f, p2)
    with pytest.raises(ValueError, match="^rope: "):
        CS.plan_ctx_cell(o, 131520, 75.0, 600, f, p2, batch=1)                   # F3 passes (601 <= 1024)
    assert CS.plan_ctx_cell(o, 130816, 75.0, 1023, f, p2, batch=1)["rope_positions_needed"] <= CS.ROPE_POSITIONS
    with pytest.raises(ValueError, match="^prefill: "):
        CS.plan_ctx_cell(o, 130816, 36.0, 512, f, p2)
    assert CS.plan_ctx_cell(o, 16128, 36.0, 512, f, p2)["B"] >= 1                 # the S terms make the difference
    with pytest.raises(ValueError, match="^--mem 500 GiB"):
        CS.plan_ctx_cell(o, 16128, 75.0, 512, f, p2, mem_gib=500)
    with pytest.raises(ValueError, match="^batch 0 < 1"):
        CS.plan_ctx_cell(o, 16128, 75.0, 512, f, p2, batch=0)
    pd = CS.plan_ctx_cell(o, 130816, 75.0, 512, f, p2, batch=32)
    assert pd["docs"] == dict(n_docs=24, distinct=24, reused=True, repeats=8) and pd["bound_by"] == "requested"
    assert CS.plan_ctx_cell(o, 65280, 75.0, 512, f, p2, batch=32)["docs"]["reused"] is False


# ------------------------------------------------------------------------------------------------------------ CS5
def test_CS5_latency():
    lm = CS.latency_metrics(4, 5, 3000.0, [1000.0, 2000.0, 2500.0, 2900.0], [1.0] * 4, [50.0, 60.0, 70.0, 80.0],
                            itl_ms=[55.0, 60.0, 70.0, 85.0], e2e_event_ms=3270.0)
    assert lm["ttft_s"] == 3.0 and lm["token_latency_basis"] == "steps"
    assert lm["token_latency_ms"]["mean"] == pytest.approx(67.5) and lm["token_latency_ms"]["p50"] == pytest.approx(65.0)
    assert lm["token_latency_ms"]["max"] == 85.0 and lm["own_prefill_s"]["max"] == pytest.approx(2.9)
    assert lm["tpot_ms"] == pytest.approx(67.5) and lm["e2e_tok_s"] == pytest.approx(20 / 3.27)
    ls = CS.latency_metrics(2, 4, 1000.0, [400.0, 900.0], [2.0, 1.0], [100.0, 40.0], None, 1200.0)
    assert ls["token_latency_basis"] == "rounds" and ls["token_latency_ms"]["mean"] == pytest.approx(140.0 / 3)
    assert ls["token_latency_ms"]["max"] == 50.0 and ls["unit_ms"]["max"] == 100.0
    assert CS.latency_metrics(2, 4, 1000.0, [400.0], [1.0], [10.0])["e2e_s"] is None


# ------------------------------------------------------------------------------------------------------------ CS6
def test_CS6_cell_list_and_time():
    entries = CS.parse_cell_list((EFF / "ctxsweep_cells.txt").read_text())
    tiers = [e["tier"] for e in entries]
    assert len(entries) == 21 and tiers.count("1") == 19 and tiers.count("2") == 2
    assert {e["S"] for e in entries} == set(CTXS) and (entries[0]["S"], entries[1]["S"]) == (65280, 130816)
    f = CS.prior_fit()
    dflt = CS.sweep_plan(entries, 75.0, 512, f, CS.HostFit())
    assert not [p for p in dflt if "error" in p], [p for p in dflt if "error" in p]
    for p in dflt:
        assert p["plan"]["host"]["need_gib"] <= p["entry"]["mem_gib"] <= 440
    pw = CS.sweep_plan(entries, 75.0, 512, f, CS.HostFit(base_gib=18.0, mode="pow2"))
    assert not [p for p in pw if "error" in p]
    tm = CS.tier_minutes(pw)
    assert 150.0 <= tm["1"] <= 195.0 and 40.0 <= tm["2"] <= 75.0, tm
    mx = {(p["entry"]["S"], p["entry"]["cell"]): p["plan"]["B"] for p in pw if p["entry"]["batch"] == "max"}
    assert mx[(130816, "cached:50")] == 63 and mx[(65280, "cached:50")] == 127 and 45 <= mx[(130816, "spec:2")] < 63
    e = CS.est_minutes(16128, 191, EC.Cell("cached", P=50), 512, 9)          # 4.1 cached p50: the step took 10:22
    assert e["prefill_s"] == pytest.approx(191 * 16128 * 0.160e-3) and 9.5 <= e["minutes"] <= 12.5
    assert CS.prefill_ms_per_token(40000) == pytest.approx(0.160 + 0.031 * (40000 - 16128) / (65280 - 16128))
    for bad in ("65280 original 8", "20000 original 8 80", "65280 original 0 80", "65280 original 8 500",
                "65280 foo 8 80"):
        with pytest.raises(ValueError):
            CS.parse_cell_list(bad)


# ------------------------------------------------------------------------------------------------------------ CS7
def _workload(tmp: Path, ctx: int, rows: int = 3):
    data, docs = bytearray(), []
    for i in range(rows):
        row = array.array("i", range(i, i + ctx)).tobytes()
        data += row
        docs.append(dict(index=10 + i, n_tokens=ctx + 5, sha256=hashlib.sha256(row).hexdigest()))
    (tmp / "tok.bin").write_bytes(bytes(data))
    (tmp / "man.json").write_text(json.dumps(dict(ctx=ctx, dtype="int32-le", shape=[rows, ctx], docs=docs)))
    return tmp / "tok.bin", tmp / "man.json"


def _res(root: Path, name, cell, fam, S, B, e2e, dec, ci, budget=75.0, verdict="VALID", lat=True):
    d = root / name
    d.mkdir(parents=True)
    r = dict(cell=cell, family=fam, rep=0, status="done", plan=dict(S=S, B=B, budget_gb=budget, N=512, margin_gb=1.0),
             plan_key=[budget, 1.0, 512, S], memory=dict(verdict=verdict, used_peak_est_gb=60.0),
             metrics=dict(batch=B, e2e_tok_s=e2e, decode_tok_s=dec, decode_tok_s_gpu_units=dec, decode_tok_s_ci=ci,
                          transferred=dict(bytes_per_committed_token=1e6), data_wait=dict(share_of_decode=0.1),
                          latency_s=dict(end_to_end=100.0), e2e_s=100.0))
    if lat:
        r["latency"] = dict(ttft_s=50.0, token_latency_ms=dict(p50=20.0, p95=30.0))
    (d / "result.json").write_text(json.dumps(r))


def test_CS7_driver_cpu(tmp_path, monkeypatch):
    cells = str(EFF / "ctxsweep_cells.txt")
    out = tmp_path / "plan.json"
    assert CB.main(["--mode", "plan", "--budget-gb", "75", "--cells-file", cells, "--out", str(out)]) == 0
    pj = json.loads(out.read_text())
    assert len(pj["cells"]) == 21 and pj["hfit"]["mode"] == "2x" and set(pj["tier_minutes"]) == {"1", "2"}
    assert CB.main(["--mode", "plan", "--budget-gb", "75", "--cells-file", cells, "--out", str(out)]) == 2
    bad = tmp_path / "bad.txt"
    bad.write_text("130816 original 64 440\n")
    assert CB.main(["--mode", "plan", "--budget-gb", "75", "--cells-file", str(bad), "--out", str(tmp_path / "p2.json")]) == 4
    assert not (tmp_path / "p2.json").exists()
    run = tmp_path / "run" / "s65280_original_b8_rep0"
    run.mkdir(parents=True)
    (run / "fit_row.json").write_text(json.dumps(dict(family="original", cell="original", B=8, slots=64, alloc_peak_gb=30.0,
                                                      reserved_peak_gb=30.5, nonpt_gb=0.9, S=65280, max_gen=576,
                                                      host_peak_gib=50.0)) + "\n")
    out3 = tmp_path / "plan3.json"
    assert CB.main(["--mode", "plan", "--budget-gb", "75", "--cells-file", cells, "--fit-rows-dir", str(tmp_path / "run"),
                    "--out", str(out3)]) == 0
    p3 = json.loads(out3.read_text())
    assert p3["hfit"]["mode"] == "pow2" and p3["fit_rows"] == 1 and [r[0] for r in p3["sfit"]["per_ctx"]] == [65280]
    # arm refusals: nothing is created
    tok, man = _workload(tmp_path, 16128)
    monkeypatch.setenv("NOSI_POOL_BLOCKS", "0")
    monkeypatch.setenv("NOSI_VERIFY_ROUND_SLOTS", "0")
    arm = ["--mode", "arm", "--budget-gb", "75", "--ctx", "65280", "--cell", "original", "--batch", "2", "--mem-gib", "80",
           "--model", "m", "--tokens", str(tok), "--manifest", str(man), "--out", str(tmp_path / "arm")]
    assert CB.main(arm) == 4 and not (tmp_path / "arm").exists()                 # the manifest is of 16128
    big = list(arm)
    big[big.index("65280")], big[big.index("2")], big[big.index("80")] = "130816", "64", "440"
    assert CB.main(big) == 4 and not (tmp_path / "arm").exists()                 # 64 > 63 (int32_host)
    knob = list(arm)
    knob[knob.index("original")] = "cached:50"
    assert CB.main(knob) == 4 and not (tmp_path / "arm").exists()                # NOSI_POOL_BLOCKS 0 != 50
    assert CB.main(["--mode", "arm", "--out", str(tmp_path / "x")]) == 4
    assert CB.main(arm[:arm.index("--batch")] + ["--batch", "lots"] + arm[arm.index("--batch") + 2:]) == 4
    # report
    rr = tmp_path / "res"
    _res(rr, "a", "original", "original", 65280, 8, 10.0, 100.0, [95.0, 105.0])
    _res(rr, "b", "cached_p50", "cached", 65280, 8, 12.0, 130.0, [125.0, 135.0])
    _res(rr, "c", "spec_k2_r62", "spec", 65280, 8, 13.0, 150.0, [140.0, 160.0])
    _res(rr, "d", "original", "original", 65280, 32, 30.0, 300.0, [290.0, 310.0])
    _res(rr, "e", "cached_p50", "cached", 65280, 32, 40.0, 400.0, [390.0, 410.0])
    _res(rr, "f", "spec_k2_r62", "spec", 65280, 32, 20.0, 200.0, [190.0, 210.0])
    _res(rr, "g", "cached_p97", "cached", 65280, 32, 99.0, 999.0, [1.0, 2.0], verdict="OVER_BUDGET")
    _res(rr, "h", "spec_k2_r62", "spec", 16128, 161, 116.3, 287.5, [280.0, 295.0], lat=False)
    _res(rr, "i", "cached_p33", "cached", 16128, 219, 169.2, 1084.5, [1070.0, 1100.0], lat=False)
    _res(rr, "j", "original", "original", 130816, 4, 1.0, 10.0, [9.0, 11.0], budget=70.0)
    _res(rr, "k", "original", "original", 130816, 16, 2.0, 20.0, [19.0, 21.0])
    rep_out = tmp_path / "rep"
    assert CB.main(["--mode", "report", "--results", str(rr), "--out", str(rep_out)]) == 0
    rep = json.loads((rep_out / "report.json").read_text())
    m64 = rep["contexts"]["65280"]["matched"]
    assert m64["8"]["baseline"] == "cached_p50" and m64["8"]["candidates"]["spec_k2_r62"]["verdict"] == "helps"
    assert m64["32"]["baseline"] == "cached_p50" and m64["32"]["candidates"]["spec_k2_r62"]["verdict"] == "hurts"
    assert m64["32"]["candidates"]["spec_k2_r62"]["speedup"]["e2e_tok_s"] == pytest.approx(20.0 / 40.0 - 1)
    assert "cached_p97" not in m64["32"]["cells"]                                 # OVER_BUDGET: never compared
    eq = rep["contexts"]["65280"]["equal_memory"]
    assert eq["best"]["cached"]["cell"] == "cached_p50@32" and eq["best"]["spec"]["cell"] == "spec_k2_r62@32"
    assert eq["speedup_e2e"]["spec_vs_best_baseline"] == pytest.approx(20.0 / 40.0 - 1)
    assert rep["contexts"]["65280"]["cells"]["cached_p97@32"]["over_budget"] == 1
    c16 = rep["contexts"]["16128"]
    assert c16["matched"]["161"]["candidates"]["spec_k2_r62"]["verdict"] == "no baseline at this batch"
    assert c16["equal_memory"]["speedup_e2e"]["spec_vs_best_baseline"] == pytest.approx(116.3 / 169.2 - 1)
    assert rep["contexts"]["130816"]["error"].startswith("results of different comparisons mixed")
    kinds = {(v["S"], v["kind"], v.get("B"), v["verdict"]) for v in rep["verdicts"]}
    assert (65280, "matched_batch", 8, "helps") in kinds and (65280, "matched_batch", 32, "hurts") in kinds
    md = (rep_out / "report.md").read_text()
    assert "## S = 65280" in md and "LOCKSTEP" in md and "ERROR: results of different" in md


# ------------------------------------------------------------------------------------------------------------ CS8
def test_CS8_host_parsing():
    st = "Name:\tpython\nVmHWM:\t 2097152 kB\nVmRSS:\t 1048576 kB\nVmLck:\t       0 kB\nVmPin:\t  524288 kB\nThreads:\t4\n"
    assert CS.parse_proc_status(st) == dict(hwm_gib=2.0, rss_gib=1.0, lck_gib=0.0, pin_gib=0.5)
    v1 = "12:memory:/slurm/uid_1/job_2/step_3/task_0\n11:cpuset:/slurm/uid_1/job_2/step_3\n"
    assert CS.cgroup_memory_files(v1) == dict(
        version=1, usage="/sys/fs/cgroup/memory/slurm/uid_1/job_2/step_3/task_0/memory.usage_in_bytes",
        peak="/sys/fs/cgroup/memory/slurm/uid_1/job_2/step_3/task_0/memory.max_usage_in_bytes")
    v2 = "0::/system.slice/slurmstepd.scope/job_2/step_3/user/task_0\n"
    assert CS.cgroup_memory_files(v2) == dict(
        version=2, usage="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_2/step_3/user/task_0/memory.current",
        peak="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_2/step_3/user/task_0/memory.peak")
    assert CS.cgroup_memory_files("") is None and CS.cgroup_memory_files("5:cpu,cpuacct:/x\n") is None
    assert CS.pinned_summary([2 ** 30, 3 * 2 ** 30]) == dict(tensors=2, nominal_gib=4.0, pow2_gib=5.0)
    now = EB.host_now()
    assert isinstance(now, dict)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace())
    rec = EB.host_record(fake, [dict(host=dict(hwm_gib=5.0)), dict(host=dict(hwm_gib=7.5, cgroup_peak_gib=9.0)), dict()],
                         [2 ** 30, 3 * 2 ** 30])
    assert rec["peak_gib"] >= 7.5 and rec["cgroup_peak_gib"] >= 9.0 and rec["pinned"]["pow2_gib"] == 5.0
    assert rec["torch_host_memory_stats"] is None


# ------------------------------------------------------------------------------------------------------------ CS9
UUID = "f31fc723-9dfe-407a-9947-28173556232a"


def _step(tmp: Path, holders: Path, dry: bool = True):
    env = {"PATH": "/usr/bin:/bin", "CTXSWEEP_DRYRUN_HOLDERS": str(holders)}
    if dry:
        env["DRYRUN"] = "1"
    args = ["bash", str(EFF / "ctxsweep_step.sh"), "--nosi", "0" * 40, "4242", "t1", "75", "65280", "original", "8", "80"]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp))


def test_CS9_step_gpu_uuid_file(tmp_path):
    hold = tmp_path / "holders"
    hold.mkdir()
    r = _step(tmp_path, hold)
    assert r.returncode == 1 and "STOP: no GPU uuid file %s/4242.gpu_uuid" % hold in r.stderr, r.stderr
    f = hold / "4242.gpu_uuid"
    for bad in ("GPU-%s\n" % UUID, "%s\nextra\n" % UUID, UUID.upper() + "\n", "not-a-uuid\n", "", UUID[:-1] + "\n"):
        f.write_text(bad)
        r = _step(tmp_path, hold)
        assert r.returncode == 1 and "does not hold exactly one uuid line" in r.stderr, (bad, r.stderr)
    if os.geteuid() != 0:
        f.write_text(UUID + "\n")
        f.chmod(0)
        r = _step(tmp_path, hold)
        assert r.returncode == 1 and "is not readable" in r.stderr, r.stderr
        f.chmod(0o644)
    f.write_text(UUID + "\n")
    r = _step(tmp_path, hold)
    assert r.returncode == 1 and "gpu_uuid" not in r.stderr and "HEAD is not " + "0" * 40 in r.stderr, r.stderr
    r = _step(tmp_path, hold, dry=False)                                          # the override is DRYRUN only
    assert r.returncode == 1 and "/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/4242.gpu_uuid" in r.stderr, r.stderr
    for args, msg in ((["65280", "original", "128", "80"], "int32 envelope 127"), (["65280", "original", "8", "500"], "mem 500G"),
                      (["20000", "original", "8", "80"], "ctx '20000'"), (["65280", "spec:9", "8", "80"], "K '9'")):
        rr = subprocess.run(["bash", str(EFF / "ctxsweep_step.sh"), "--nosi", "0" * 40, "4242", "t1", "75"] + args,
                            env={"PATH": "/usr/bin:/bin", "DRYRUN": "1"}, capture_output=True, text=True, timeout=60)
        assert rr.returncode == 1 and msg in rr.stderr, (args, rr.stderr)


def test_CS9_step_workload_pins():
    text = (EFF / "ctxsweep_step.sh").read_text()
    pins = dict(re.findall(r"^(TOK_SHA_\d+|MAN_SHA_\d+)=([0-9a-f]{64})", text, re.M))
    assert set(pins) == {"%s_SHA_%d" % (k, S) for k in ("TOK", "MAN") for S in CTXS}
    eq = (EFF / "eqmem_step.sh").read_text()
    assert "TOK_SHA=%s" % pins["TOK_SHA_16128"] in eq and "MAN_SHA=%s" % pins["MAN_SHA_16128"] in eq
    root = Path("/mnt/beegfs/ojerbi/kstep-cpu-runs/workload")
    if not root.is_dir():
        pytest.skip("the workload directory is not mounted here")
    for S in CTXS:
        c = CS.CONTEXTS[S]
        sums = dict(tuple(reversed(x.split())) for x in (root / c.dirname / "SHA256SUMS").read_text().splitlines()
                    if x.strip())
        assert sums["%s_tokens.bin" % c.stem] == pins["TOK_SHA_%d" % S]
        assert sums["%s_manifest.json" % c.stem] == pins["MAN_SHA_%d" % S]
