"""CPU tests of "p0sweep" (r2_base_profile_cell.py: VerifyProbe, the GEMM-rows and verify summaries, verify_ceiling,
PRED_SWEEP; r2_base_profile_step.sh: the cells p0c63b64, p0c63b128, p0c128b64, p0k64c63, p0k64c128 and the sequence
p0sweep). No CUDA, no Triton launch, no job. The engine modules are replaced by fakes with the same names.

PS1  parse_args: --context 65280; the verify defaults (34, 12, 1,2,3,4); --vfy-steps 0; refusals: the verify window
     over the selection window, over a re-warm, over the profiler window, too few / too many steps, past the last step;
     --gemm-mults without 1, without --sel-rows, not increasing, a duplicate, above 8, not integers. S 65280 has the
     re-warm steps of S 16128.
PS2  expected_sequence with vfy: the lengths (pool / no pool), 'union' just before tail_write, 'load_count' just before
     fetch_k, gemm_prep + gemm_x<m> after ffn_gemm and before inter / final; no probe label in GROUPS.
PS3  VerifyProbe.union / count with CPU torch: the ring, the union of R consecutive selections per (head, request)
     (hand count), the loads (mask >= 0) per step, the samples per step, the labels; the inputs unchanged; take()
     before any call; an error disables the probe and stops its labels.
PS4  VerifyProbe.gemm with CPU torch and tiny weights: the labels, the rows m x B (repeat_interleave layout), the op
     order and shapes of the shipped GEMM stack (+ final rmsnorm and lm_head at the last layer only), _stack equal to a
     direct computation; the input unchanged; an error disables the probe.
PS5  ls_line, gemm_rows_summary, vfy_summary by hand arithmetic (partial steps left out).
PS6  verify_ceiling and verify_case by hand arithmetic (measured / hidden fetch, two draft shares, two acceptances);
     the missing inputs.
PS7  Patcher + VerifyProbe on a fake engine (the shipped call order, real CPU tensors): a verify step's boundaries equal
     expected_sequence(vfy) with and without the pool, a main step's equal the plain sequence; the probe counts;
     restore. NEGATIVE CONTROL: a fake engine that calls the K gather through a private alias loses 'load_count' and
     'fetch_k', and the chain names 'load_count' first.
PS8  TimerHooks with a verify window: the kind of every step, the probe on and its step index in the window only (a
     re-warm inside the window is not a verify step), off after stop().
PS9  PRED_SWEEP and verdicts_sweep: the keys of the step script's sweep cells, every value TBD (or a stamped band);
     TBD -> UNREGISTERED; a stamped block -> CONFIRMED / REFUTED; the dr1 share by pool; NOT_RUN without values.
PS10 load_workload at S 65280: the real file (shape 55 x 65280, sha256 = the step script's pin; skipped when absent).
PS11 static: the probe's GEMM stack names the ops of the shipped decode (nosa_llama.py), the decode_update_kv and gather
     arguments that the probe reads (cache_engine.py, flash_h2d_mask.py); the int32 envelope gives B 127 at 64K and B 63
     at 128K, and the step script uses B 127 at 64K.
PS12 r2_base_profile_step.sh: malformed sweep cells STOP; at a clean checkout of the ROOT, DRYRUN prints each sweep
     cell's C, pool, B, S, steps, STEP_MIN, --mem and arguments, the 64K pin, and the p0sweep order for an even and an
     odd holder.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_p0sweep.py
"""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-p0sweep"
WLM = Path("/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280")
WLM_SHA = "f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3"
MODEL_CFG = Path("/mnt/beegfs/ojerbi/models/NOSA-8B/config.json")
UUID = "f31fc723-9dfe-407a-9947-28173556232a"
SWEEP = {"p0c63b64": (63, 0, 64, 16128, 512, 30, "240G"),
         "p0c63b128": (63, 0, 128, 16128, 512, 35, "240G"),
         "p0c128b64": (128, 65, 64, 16128, 512, 30, "240G"),
         "p0k64c63": (63, 0, 127, 65280, 512, 90, "440G"),
         "p0k64c128": (128, 65, 127, 65280, 512, 90, "440G")}


def _pc():
    sys.path.insert(0, str(EFF))
    import r2_base_profile_cell as PC
    return PC


def _bc():
    sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


class FakeClock:
    """Every record advances the time by 1 ms."""

    def __init__(self):
        self.now = 0.0

    def event(self):
        return [None]

    def record(self, e):
        self.now += 1.0
        e[0] = self.now

    def elapsed(self, a, b):
        return b[0] - a[0]

    def wait(self, e):
        assert e[0] is not None

    def sync(self):
        pass


def _labels(rec, t):
    """The labels of step t's stages (after finish)."""
    return set(rec.steps[t].get("stages", {}))


# ------------------------------------------------------------------ PS1
def test_PS1_parse_args_sweep():
    PC, BC = _pc(), _bc()
    base = ["--out", "x", "--workload", "w", "--batch", "127"]
    a = PC.parse_args(base + ["--stage-timers", "--context", "65280", "--pool", "65"])
    assert (a.context, a.vfy_start, a.vfy_steps, a.gemm_mults) == (65280, 34, 12, (1, 2, 3, 4))
    assert PC.parse_args(base + ["--stage-timers", "--context", "65280", "--pool", "0"]).pool == 0
    assert PC.parse_args(base + ["--stage-timers", "--vfy-steps", "0"]).vfy_steps == 0
    assert PC.parse_args(base + ["--stage-timers", "--vfy-start", "40"]).vfy_start == 40
    assert PC.parse_args(base + ["--stage-timers", "--gemm-mults", "1,4"]).gemm_mults == (1, 4)
    assert PC.parse_args(base + ["--stage-timers", "--sel-rows", "2", "--gemm-mults", "1,2"]).gemm_mults == (1, 2)
    assert PC.parse_args(base + ["--gemm-mults", "2,4"]).gemm_mults == (2, 4)      # not checked without the timers
    bad = (["--stage-timers", "--vfy-start", "28"],          # over the selection window 24 .. 31
           ["--stage-timers", "--vfy-start", "20"],
           ["--stage-timers", "--vfy-start", "60"],          # re-warm 64
           ["--stage-timers", "--vfy-start", "5"],           # the profiler window 8 .. 10 and its neighbours
           ["--stage-timers", "--vfy-steps", "5"],           # < --sel-rows + 2
           ["--stage-timers", "--vfy-steps", "65"],
           ["--stage-timers", "--vfy-start", "505"],         # past step 511
           ["--stage-timers", "--gemm-mults", "2,4"], ["--stage-timers", "--gemm-mults", "1,2,3"],
           ["--stage-timers", "--gemm-mults", "1,4,2"], ["--stage-timers", "--gemm-mults", "1,1,4"],
           ["--stage-timers", "--gemm-mults", "1,4,9"], ["--gemm-mults", "1,x"], ["--context", "65281"])
    for b in bad:
        with pytest.raises(SystemExit):
            PC.parse_args(base + b)
    r16 = [t for t, _ in BC.expected_rewarms(16128, 512)]
    assert [t for t, _ in BC.expected_rewarms(65280, 512)] == r16
    assert BC.doc_rows(127, 55)[54:57] == [54, 0, 1] and max(BC.doc_rows(127, 55)) == 54


# ------------------------------------------------------------------ PS2
def test_PS2_expected_sequence_vfy():
    PC = _pc()
    m = (1, 2, 3, 4)
    assert PC.vfy_labels(m) == ("union", "load_count", "gemm_prep", "gemm_x1", "gemm_x2", "gemm_x3", "gemm_x4")
    for pool, n in ((True, 706), (False, 610)):
        s = PC.expected_sequence(32, pool, vfy=m)
        assert len(s) == n, (pool, len(s))
        assert len(PC.expected_sequence(32, pool)) + 32 * 7 == n
        assert s.count("union") == s.count("load_count") == s.count("gemm_x4") == 32
        for i, x in enumerate(s):
            if x == "union":
                assert s[i + 1] == "tail_write" and s[i - 1] == "pool_topk"
            if x == "load_count":
                assert s[i + 1] == "fetch_k" and s[i - 1] == ("map_commit" if pool else "diff")
            if x == "gemm_prep":
                assert s[i - 1] == "ffn_gemm" and s[i + 1:i + 5] == ("gemm_x1", "gemm_x2", "gemm_x3", "gemm_x4")
                assert s[i + 5] in ("inter", "final")
    s = PC.expected_sequence(2, True, vfy=(1, 4))
    assert s[-3:] == ("gemm_x4", "final", "sample")
    grouped = {x for _, ls in PC.GROUPS for x in ls}
    assert not grouped & set(PC.vfy_labels(m))


# ------------------------------------------------------------------ PS3
def test_PS3_union_and_load_count():
    torch = pytest.importorskip("torch")
    PC = _pc()
    H, B, TOPK, R, N = 2, 3, 4, 4, 6
    rec = PC.StageRecorder(FakeClock(), 2, None, ring=2)
    vp = PC.VerifyProbe(rec, N, torch, None, None, None, mults=(1, 4), rows=R)
    assert vp.take() == [[0] * N, [0] * N, [0] * N]
    kept = []
    for i in range(N):
        rec.step_begin(i, True, "embed", "vfy")
        vp.active, vp.step = True, i
        for li in range(2):
            rec.layer(li)
            sel = (torch.arange(TOPK) + i).expand(H, B, TOPK).clone()
            sel[1, 2] = torch.arange(TOPK)                 # one (head, request) keeps its blocks: union TOPK
            mask = torch.full((H, B, TOPK), -1, dtype=torch.int64)
            mask.view(-1)[:i + li] = 7                     # i + li host loads
            kept.append((sel.clone(), mask.clone(), sel, mask))
            vp.union((None, None, None, li, sel), {})
            vp.count((None, None, mask, 64), {})
        rec.mark("sample")
    rec.finish()
    assert all(torch.equal(a, c) and torch.equal(b, d) for a, b, c, d in kept)     # the inputs are not written
    acc = vp.take()
    # ids i - 3 .. i + 3 over R = 4 consecutive steps: TOPK + R - 1 = 7 distinct; the fixed stream: 4
    assert acc[0] == [0, 0, 0, 78, 78, 78]                # 2 layers x (5 streams x 7 + 1 stream x 4)
    assert acc[1] == [0, 0, 0, 7, 7, 7]
    assert acc[2] == [2 * i + 1 for i in range(N)]
    assert vp.union_n == [0, 0, 0, 12, 12, 12] and vp.streams == [12] * N
    assert vp.calls == dict(union=12, count=12, gemm=0) and not vp.disabled
    assert all({"union", "load_count"} <= _labels(rec, i) for i in range(N))
    # an error disables the probe; its labels stop
    rec = PC.StageRecorder(FakeClock(), 1, None, ring=2)
    vp = PC.VerifyProbe(rec, 2, torch, None, None, None, rows=R)
    rec.step_begin(0, True, "embed", "vfy")
    vp.active, vp.step = True, 0
    rec.layer(0)
    vp.union((None, None, None), {})                       # no topk_idx
    assert vp.disabled and len(vp.errors) == 1 and vp.errors[0].startswith("union layer 0: KeyError")
    n0 = rec.n[rec.cur]
    vp.count((None, None, torch.zeros(2, 3, 4)), {})
    vp.union((None, None, None, 0, torch.zeros(2, 3, 4)), {})
    assert rec.n[rec.cur] == n0 and vp.calls == dict(union=0, count=0, gemm=0)


# ------------------------------------------------------------------ PS4
def _gemm_world(torch):
    import torch.nn.functional as TF
    D, Q, KV, I, V, B = 8, 8, 2, 6, 5, 3
    calls = []

    def linear(x, w):
        calls.append(("linear", tuple(x.shape), tuple(w.shape)))
        return TF.linear(x, w)

    def norm(x, eps, w):
        calls.append(("norm", tuple(x.shape), eps, x.clone()))
        return x * w

    def silu(g, out):
        calls.append(("silu", tuple(g.shape), tuple(out.shape)))
        d = g.shape[-1] // 2
        out.copy_(TF.silu(g[..., :d]) * g[..., d:])
    g = torch.Generator().manual_seed(0)

    def W(*s):
        return torch.randn(*s, generator=g)
    layer = types.SimpleNamespace(wqkv=W(Q + 2 * KV, D), wo=W(D, Q), gate_up_proj=W(2 * I, D), down_proj=W(D, I),
                                  input_layernorm_weight=W(D), input_layernorm_variance_epsilon=1e-5,
                                  post_attention_layernorm_weight=W(D), post_attention_layernorm_variance_epsilon=2e-5)
    final = (W(D), 3e-5, W(V, D))
    F = types.SimpleNamespace(linear=linear)
    return dict(D=D, Q=Q, KV=KV, I=I, V=V, B=B, calls=calls, norm=norm, silu=silu, layer=layer, final=final, F=F,
                x=W(B, 1, D), TF=TF)


def test_PS4_gemm_probe():
    torch = pytest.importorskip("torch")
    PC = _pc()
    w = _gemm_world(torch)
    D, Q, KV, I, V, B, calls, layer = w["D"], w["Q"], w["KV"], w["I"], w["V"], w["B"], w["calls"], w["layer"]
    rec = PC.StageRecorder(FakeClock(), 2, None, ring=2)
    vp = PC.VerifyProbe(rec, 4, torch, w["F"], w["norm"], w["silu"], final=w["final"], mults=(1, 2, 4), rows=4)
    x = w["x"]
    x0 = x.clone()
    rec.step_begin(0, True, "embed", "vfy")
    vp.active, vp.step = True, 0
    rec.layer(0)
    vp.gemm(layer, (x, "cache"), False)
    s = rec.cur
    assert rec.lab[s][:rec.n[s]] == ["embed", "gemm_prep", "gemm_x1", "gemm_x2", "gemm_x4"]
    rec.layer(1)
    vp.gemm(layer, (x, "cache"), True)
    rec.mark("sample")
    rec.finish()
    assert torch.equal(x, x0) and vp.calls["gemm"] == 2 and not vp.disabled
    lin = [c[1:] for c in calls if c[0] == "linear"]
    assert len(lin) == 3 * 4 + 3 * 5                         # 4 GEMMs per stack; + lm_head at the last layer
    for j, m in enumerate((1, 2, 4)):
        r = (m * B, 1)
        assert lin[4 * j:4 * j + 4] == [(r + (D,), (Q + 2 * KV, D)), (r + (Q,), (D, Q)), (r + (D,), (2 * I, D)),
                                        (r + (I,), (D, I))]
        assert lin[12 + 5 * j:12 + 5 * j + 5] == lin[4 * j:4 * j + 4] + [(r + (D,), (V, D))]
    norms = [c for c in calls if c[0] == "norm"]
    assert [c[2] for c in norms[:6]] == [1e-5, 2e-5] * 3 and [c[2] for c in norms[6:]] == [1e-5, 2e-5, 3e-5] * 3
    assert torch.equal(norms[2][3], x.repeat_interleave(2, dim=0))  # m = 2: the rows of one request side by side
    assert [c[1:] for c in calls if c[0] == "silu"][:1] == [((B, 1, 2 * I), (B, 1, I))]
    # _stack = the shipped op order, computed directly
    TF, (fw, feps, head) = w["TF"], w["final"]
    h = x * layer.input_layernorm_weight
    qkv = TF.linear(h, layer.wqkv)
    h = x + TF.linear(qkv[..., :Q], layer.wo)
    gu = TF.linear(h * layer.post_attention_layernorm_weight, layer.gate_up_proj)
    h = h + TF.linear(TF.silu(gu[..., :I]) * gu[..., I:], layer.down_proj)
    want = TF.linear(h * fw, head).float()
    got = vp._stack(layer, x, True)
    assert got.shape == (B, 1, V) and got.dtype == torch.float32 and torch.allclose(got, want, atol=1e-5)
    assert vp._stack(layer, x, False).shape == (B, 1, D)
    # an error disables the probe
    bad = types.SimpleNamespace(**{k: v for k, v in vars(layer).items() if k != "wo"})
    rec = PC.StageRecorder(FakeClock(), 1, None, ring=2)
    vp = PC.VerifyProbe(rec, 2, torch, w["F"], w["norm"], w["silu"], mults=(1, 4), rows=4)
    rec.step_begin(0, True, "embed", "vfy")
    vp.active, vp.step = True, 0
    rec.layer(0)
    vp.gemm(bad, (x,), False)
    assert vp.disabled and vp.errors[0].startswith("gemm layer 0: AttributeError") and vp.calls["gemm"] == 0


# ------------------------------------------------------------------ PS5
def test_PS5_summaries():
    PC = _pc()
    assert PC.ls_line([1, 2, 3], [2, 4, 6]) == (2.0, 0.0)
    assert PC.ls_line([1], [1]) == (None, None) and PC.ls_line([2, 2], [1, 3]) == (None, None)
    g = {"gemm_x1": 11.0, "gemm_x2": 12.0, "gemm_x4": 16.0, "gemm_prep": 0.5}
    steps = {t: dict(period_ms=50.0, inner=True, kind="vfy", stages=dict(g)) for t in (35, 36, 37)}
    steps[38] = dict(period_ms=50.0, inner=True, kind="main", stages=dict(g))                 # not a verify step
    steps[39] = dict(period_ms=50.0, inner=True, kind="vfy", stages={"gemm_x1": 99.0})          # partial: left out
    s = PC.gemm_rows_summary(steps, [35, 36, 37, 38, 39], (1, 2, 4), 10, shipped_ms=10.0)
    assert s["n"] == 3 and [p["rows"] for p in s["points"]] == [10, 20, 40]
    assert [p["median_ms"] for p in s["points"]] == [11.0, 12.0, 16.0]
    assert [p["ms_per_row"] for p in s["points"]] == [1.1, 0.6, 0.4]
    assert [p["over_floor_ms"] for p in s["points"]] == [0.4, 1.4, 5.4]
    assert [m["ms_per_row"] for m in s["marginal"]] == [0.1, 0.2] and s["marginal"][0]["rows"] == [10, 20]
    assert (s["slope_ms_per_row"], s["intercept_ms"]) == (0.17143, 9.0)              # 240 / 1400; 13 - 4
    assert (s["x1_over_shipped"], s["prep"]["median_ms"], s["floor_ms"]) == (1.1, 0.5, 10.6)
    assert PC.gemm_rows_summary(steps, [38], (1, 2, 4), 10)["n"] == 0
    # the verify window: 5 steps; index 0 = the warm-up; index 4 partial (a probe error): both left out of the loads
    ts = [34, 35, 36, 37, 38]
    vsteps = {t: dict(stages=dict(fetch_k=float(k), fetch_v=float(k))) for k, t in enumerate(ts)}
    vsteps[34]["stages"] = dict(fetch_k=100.0, fetch_v=100.0)
    acc = [[0, 0, 0, 256 * 72, 256 * 74], [0, 0, 0, 90, 95], [999, 100, 200, 300, 400]]
    v = PC.vfy_summary(vsteps, ts, acc, [0, 0, 0, 256, 256], [256, 256, 256, 256, 128], 64)
    assert (v["union_steps"], v["union_samples"], v["union_mean"], v["union_max"]) == (2, 512, 73.0, 95)
    assert v["extras_per_stream"] == 9.0
    assert (v["load_steps"], v["loads"], v["streams_per_step"], v["loads_per_step"]) == (3, 600, 256.0, 200.0)
    assert v["loads_per_stream"] == pytest.approx(600 / 768, abs=1e-5)
    assert (v["fetch_ms_per_step"], v["ms_per_load"]) == (4.0, 0.02)                   # (2 + 4 + 6) / 3; 12 / 600
    assert v["union_fetch_ms"] == pytest.approx(9.0 * 256 * 0.02) and v["extras_over_loads"] == 11.52
    # a union step with fewer samples (a probe error) is left out too
    v = PC.vfy_summary(vsteps, ts, acc, [0, 0, 0, 256, 100], [256] * 5, 64)
    assert (v["union_steps"], v["union_mean"]) == (1, 72.0)
    v = PC.vfy_summary(vsteps, ts, [[0] * 5, [0] * 5, [0] * 5], [0] * 5, [256] * 5, 64)
    assert v["union_mean"] is None and v["union_fetch_ms"] is None and v["ms_per_load"] is None


# ------------------------------------------------------------------ PS6
def test_PS6_verify_ceiling():
    PC = _pc()
    gemm = dict(points=[dict(mult=1, median_ms=12.0), dict(mult=2, median_ms=20.0), dict(mult=4, median_ms=30.0)])
    sel = dict(extra_joint_ms=5.0, r_rows=dict(median_ms=9.0))
    rec = (("floor", 1.0), ("half", 0.5))
    vc = PC.verify_ceiling(100.0, 100, 50.0, gemm, sel, dict(union_fetch_ms=20.0), rows=4, attention_ms=10.0,
                           accept=(1.0, 0.5), recovery=rec, ref_tok_s=2000.0, mb1_tok_s=1000.0)
    assert (vc["K"], vc["gemm_extra_ms"], vc["own_tok_s"]) == (3, 18.0, 1000.0)
    assert vc["joint_ms"] == {"hidden": 123.0, "measured": 143.0}
    assert vc["joint_steps"] == {"hidden": 1.23, "measured": 1.43}
    assert vc["components"] == dict(gemm_r_rows_ms=30.0, selection_r_rows_ms=9.0, one_attention_ms=10.0,
                                    union_fetch_ms=20.0, rest_of_step_ms=74.0)
    assert [(c["fetch"], c["draft"], c["accept"]) for c in vc["cases"]] == [
        (f, d, a) for f in ("measured", "hidden") for d in ("floor", "half") for a in (1.0, 0.5)]
    c = PC.verify_case(vc, "measured", "floor", 1.0)
    assert (c["draft_ms"], c["round_ms"], c["tokens_per_request"], c["tok_s"]) == (50.0, 293.0, 4.0, 1365.2)
    assert (c["vs_cached"], c["vs_own"], c["vs_mb1"]) == (0.6826, 1.3652, 1.3652)
    assert (c["breakeven_joint_own_ms"], c["margin_own_ms"]) == (250.0, 107.0)
    assert (c["breakeven_joint_cached_ms"], c["margin_cached_ms"]) == (50.0, -93.0)
    c = PC.verify_case(vc, "measured", "floor", 0.5)
    assert (c["tokens_per_request"], c["tok_s"], c["vs_own"]) == (1.875, 639.9, 0.6399)
    assert (c["breakeven_joint_own_ms"], c["margin_own_ms"], c["breakeven_joint_cached_ms"]) == (37.5, -105.5, -56.25)
    c = PC.verify_case(vc, "measured", "half", 1.0)
    assert (c["draft_ms"], c["round_ms"], c["tok_s"], c["vs_own"]) == (75.0, 368.0, 1087.0, 1.087)
    assert (c["breakeven_joint_own_ms"], c["margin_own_ms"], c["margin_cached_ms"]) == (175.0, 32.0, -168.0)
    c = PC.verify_case(vc, "hidden", "floor", 1.0)
    assert (c["round_ms"], c["tok_s"], c["vs_own"], c["margin_own_ms"]) == (273.0, 1465.2, 1.4652, 127.0)
    assert PC.verify_case(vc, "measured", "dr1_69", 1.0) is None and PC.verify_case(None, "x", "y", 1) is None
    # no union fetch: the hidden variant only, no components
    vc = PC.verify_ceiling(100.0, 100, 50.0, gemm, sel, dict(union_fetch_ms=None), rows=4, attention_ms=10.0,
                           accept=(1.0,), recovery=rec, ref_tok_s=2000.0)
    assert set(vc["joint_ms"]) == {"hidden"} and "components" not in vc
    assert {c["fetch"] for c in vc["cases"]} == {"hidden"} and vc["cases"][0]["vs_mb1"] is None
    # missing inputs
    assert PC.verify_ceiling(100.0, 100, None, gemm, sel, None)["missing"] == ["gather_ms"]
    assert PC.verify_ceiling(100.0, 100, 50.0, dict(points=gemm["points"][:2]), sel, None)["missing"] == ["gemm x4"]
    assert PC.verify_ceiling(100.0, 100, 50.0, gemm, None, None, rows=2)["missing"] == ["selection extra"]
    # the defaults: R 4, acceptance 0.995 and 1.0, three draft shares, 3,387.7
    vc = PC.verify_ceiling(100.0, 100, 50.0, gemm, sel, dict(union_fetch_ms=20.0))
    assert vc["ref_tok_s"] == 3387.7 and len(vc["cases"]) == 2 * 3 * 2
    assert PC.verify_case(vc, "measured", "dr1_69", 0.995)["tokens_per_request"] == pytest.approx(3.9701, abs=1e-5)
    assert PC.verify_case(vc, "hidden", "dr1_44", 1.0)["draft_ms"] == pytest.approx(100.0 - 0.444 * 50.0, abs=1e-3)


# ------------------------------------------------------------------ PS7
def _fake_engine(torch, pool=True, alias_bypass=False):
    nl = types.ModuleType("fake_nl")
    cen = types.ModuleType("fake_cen")
    calls = []
    H, B, TOPK = 2, 3, 4

    def nosa_linear(x):
        calls.append("nosa_linear")
        return x

    def stage1(q, k, v, **kw):
        calls.append("stage1")
        return q

    def attn(q, *a, **kw):
        calls.append("attn")
        return q
    nl.nosa_linear, nl.infllmv2_attn_stage1_fast, nl.flash_attn_nosa_with_kvcache = nosa_linear, stage1, attn

    class LlamaLayer:
        def __init__(self, i, weights):
            self.layer_idx = i
            for k, v in vars(weights).items():
                setattr(self, k, v)

        def decode_forward(self, x, cache):           # the shipped order (nosa_llama.py:556, :592, :618, :634)
            nl.nosa_linear(x)
            nl.infllmv2_attn_stage1_fast(x, "k", "k", cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=1,
                                         max_seqlen_k=8, causal=False)
            sel = (torch.arange(TOPK) + cache.t).expand(H, B, TOPK).clone()
            cache.decode_update_kv(x, x, None, self.layer_idx, sel)
            nl.flash_attn_nosa_with_kvcache(x)
            return x + 1
    nl.LlamaLayer = LlamaLayer

    def h2d(gpu, cpu, load_ids, block_size):
        calls.append("h2d_k")

    def h2d_bias(gpu, cpu, bias_gpu, kv_bias, load_ids, block_size):
        calls.append("h2d_v")
    cen.diff = types.SimpleNamespace(diff_offload=lambda *a: calls.append("diff"))
    cen._pool = types.SimpleNamespace(pool_update=lambda *a: calls.append("pool_update")) if pool else None
    cen._flash_pool_swap = (lambda *a: calls.append("swap")) if pool else None
    cen.flash_h2d_from_mask, cen.flash_h2d_from_mask_bias = h2d, h2d_bias
    private = h2d                                     # an alias bound before any patch

    class InfLLMv2Cache:
        t = 0

        def decode_update_kv(self, k, v, kv_bias, layer_idx, topk_idx):   # cache_engine.py:803-806, :480-519
            cen.diff.diff_offload()
            if cen._pool is not None:
                cen._pool.pool_update()
                cen._flash_pool_swap()
            mask = torch.full((H, B, TOPK), -1, dtype=torch.int64)
            mask.view(-1)[:layer_idx + 1] = 5
            (private if alias_bypass else cen.flash_h2d_from_mask)(None, None, mask, 64)
            cen.flash_h2d_from_mask_bias(None, None, None, None, mask, 64)
            return "kv%d" % layer_idx
    cen.InfLLMv2Cache = InfLLMv2Cache
    return nl, cen, types.SimpleNamespace(graph=object), calls


def _vstep(rec, vfy, layers, cache, x, t, i=None):
    """One timed step of the fake engine; i = the index in the verify window (None: a main step)."""
    rec.step_begin(t, True, "embed", "main" if i is None else "vfy")
    vfy.active, vfy.step = i is not None, i
    cache.t = t
    for layer in layers:
        x = layer.decode_forward(x, cache)
    rec.mark("sample")
    vfy.active = False


def test_PS7_patcher_with_verify_probe():
    torch = pytest.importorskip("torch")
    PC = _pc()
    w = _gemm_world(torch)
    mults, n = (1, 4), 3
    for pool in (True, False):
        nl, cen, cuda, calls = _fake_engine(torch, pool)
        exp = dict(main=PC.expected_sequence(n, pool), vfy=PC.expected_sequence(n, pool, vfy=mults))
        rec = PC.StageRecorder(FakeClock(), n, exp, ring=2)
        vfy = PC.VerifyProbe(rec, 4, torch, w["F"], w["norm"], w["silu"], final=w["final"], mults=mults, rows=2)
        cache = cen.InfLLMv2Cache()
        before = PC.snapshot_targets(nl, cen, cuda)
        p = PC.Patcher(rec, nl, cen, n, None, vfy)
        p.install(cache)
        layers = [nl.LlamaLayer(i, w["layer"]) for i in range(n)]
        _vstep(rec, vfy, layers, cache, w["x"], 0)                          # a main step: the probe stays off
        assert vfy.calls == dict(union=0, count=0, gemm=0)
        for i in range(4):
            _vstep(rec, vfy, layers, cache, w["x"], 1 + i, i)
        p.restore()
        rec.finish()
        assert PC.moved_targets(before, PC.snapshot_targets(nl, cen, cuda)) == []
        assert "decode_update_kv" not in vars(cache)
        assert rec.chain_of("main") == dict(ok=1, bad=0, examples=[]), (pool, rec.chain_of("main"))
        assert rec.chain_of("vfy") == dict(ok=4, bad=0, examples=[]), (pool, rec.chain_of("vfy"))
        assert vfy.calls == dict(union=12, count=12, gemm=12) and not vfy.errors
        acc = vfy.take()
        assert vfy.union_n == [0, 18, 18, 18] and acc[0] == [0, 90, 90, 90] and acc[1] == [0, 5, 5, 5]
        assert acc[2] == [6] * 4 and vfy.streams == [18] * 4                # loads 1 + 2 + 3 per step
        one = ["nosa_linear", "stage1", "diff"] + (["pool_update", "swap"] if pool else []) + ["h2d_k", "h2d_v", "attn"]
        assert calls == one * n * 5
    # NEGATIVE CONTROL: the K gather through a private alias never reaches its wrapper
    nl, cen, cuda, calls = _fake_engine(torch, True, alias_bypass=True)
    seq = PC.expected_sequence(2, True, vfy=mults)
    rec = PC.StageRecorder(FakeClock(), 2, dict(vfy=seq), ring=2)
    vfy = PC.VerifyProbe(rec, 2, torch, w["F"], w["norm"], w["silu"], final=w["final"], mults=mults, rows=2)
    cache = cen.InfLLMv2Cache()
    p = PC.Patcher(rec, nl, cen, 2, None, vfy)
    p.install(cache)
    _vstep(rec, vfy, [nl.LlamaLayer(0, w["layer"]), nl.LlamaLayer(1, w["layer"])], cache, w["x"], 0, 0)
    p.restore()
    rec.finish()
    c = rec.chain_of("vfy")
    assert c["bad"] == 1 and c["examples"][0]["expected"] == "load_count" and c["examples"][0]["got"] == "fetch_v"
    assert vfy.calls["count"] == 0 and vfy.calls["union"] == 2


# ------------------------------------------------------------------ PS8
class FakePatcher:
    def __init__(self, log):
        self.log = log

    def install(self, cache):
        self.log.append("install")

    def suspend(self):
        self.log.append("suspend")

    def resume(self):
        self.log.append("resume")

    def restore(self):
        self.log.append("restore")


class FakeProf:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append("prof.start")

    def stop(self):
        self.log.append("prof.stop")


def test_PS8_timer_hooks_verify_window():
    PC = _pc()
    log = []

    class Base:
        def __init__(self, g):
            pass
    clock = FakeClock()
    rec = PC.StageRecorder(clock, 1, None, ring=2)
    probe = types.SimpleNamespace(active=False)
    vfy = types.SimpleNamespace(active=False, step=None)
    hooks = PC.TimerHooks(rec, FakePatcher(log), PC.CaptureTimer(rec), PC.GcTimer(), (4, 6), clock,
                          lambda: FakeProf(log), types.SimpleNamespace(graph=Base), 24,
                          gc_mod=types.SimpleNamespace(callbacks=[]), sel_window=(8, 10), probe=probe,
                          vfy_window=(12, 18), vfy=vfy)
    st = dict(t=0)
    seen = {}

    def rw():
        return st["t"] in (0, 14), ("first" if st["t"] == 0 else ("out_len" if st["t"] == 14 else ""))

    def dec():
        seen[st["t"]] = (probe.active, vfy.active, vfy.step)

    def smp(t):
        st["t"] = t + 1
    hooks.start(types.SimpleNamespace())
    try:
        PC.drive(24, rw, dec, smp, hooks)
    finally:
        hooks.stop()
    kinds = {t: rec.steps[t]["kind"] for t in range(24)}
    assert [t for t in range(24) if kinds[t] == "vfy"] == [12, 13, 15, 16, 17]        # 14 is a re-warm
    assert [t for t in range(24) if kinds[t] == "sel"] == [8, 9]
    assert [t for t in range(24) if seen[t][1]] == [12, 13, 15, 16, 17]
    assert [seen[t][2] for t in (12, 13, 15, 16, 17)] == [0, 1, 3, 4, 5]
    assert all(seen[t][2] is None for t in range(24) if t not in (12, 13, 15, 16, 17))
    assert [t for t in range(24) if seen[t][0]] == [8, 9]
    assert not vfy.active and not probe.active
    assert log == ["install", "suspend", "prof.start", "prof.stop", "resume", "restore"]


# ------------------------------------------------------------------ PS9
def _sweep_res(cached=(0.8, 0.7, 0.6)):
    PC = _pc()
    cases = [dict(fetch="measured", draft=d, accept=0.995, vs_cached=v, vs_own=1.2) for d, v in
             zip(("floor", "dr1_69", "dr1_44"), cached)]
    return dict(stages=dict(groups={g: dict(median_ms=float(i + 1)) for i, (g, _) in enumerate(PC.GROUPS)}),
                gemm_rows=dict(points=[dict(mult=1, median_ms=12.0), dict(mult=4, median_ms=30.0)],
                               slope_ms_per_row=0.05),
                selection=dict(extra_joint_ms=14.0),
                verify_probe=dict(extras_per_stream=8.1, loads_per_stream=0.5, union_fetch_ms=40.0),
                verify_ceiling=dict(rows=4, joint_steps=dict(measured=1.5, hidden=1.2), cases=cases))


def test_PS9_pred_sweep_and_verdicts():
    PC = _pc()
    assert re.fullmatch(r"TBD|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", PC.REGISTERED_SWEEP_UTC)
    keys = {PC.cell_key(C, B, S) for C, _, B, S, _, _, _ in SWEEP.values()}
    assert keys == set(PC._SWEEP_KEYS) == {"63_64", "63_128", "128_64", "63_127_s65280", "128_127_s65280"}
    groups = {g for g, _ in PC.GROUPS}
    for name, v in PC.PRED_SWEEP.items():
        assert set(v) == keys, name
        for k, x in v.items():
            vals = x.values() if name == "stage_ms" else [x]
            if name == "stage_ms":
                assert set(x) == groups, (name, k)
            assert all(y == "TBD" or (isinstance(y, tuple) and len(y) == 2) for y in vals), (name, k)
    assert (PC.dr1_for_pool(0), PC.dr1_for_pool(65)) == ("dr1_69", "dr1_44")
    names = ["SW-STAGE-%s" % g for g, _ in PC.GROUPS] + [
        "SW-GEMM-X1", "SW-GEMM-XR", "SW-GEMM-SLOPE", "SW-SEL", "SW-UNION", "SW-LOADS", "SW-UFETCH", "SW-JOINT",
        "SW-CACHED-FLOOR", "SW-CACHED-DR1", "SW-OWN-FLOOR"]
    v = PC.verdicts_sweep(63, 64, _sweep_res(), utc="TBD")
    assert [n for n, _, _ in v] == names and {x for _, x, _ in v} == {"UNREGISTERED"}
    assert {x for _, x, _ in PC.verdicts_sweep(63, 64, {}, utc="TBD")} == {"NOT_RUN"}
    tbd = {n: {k: ({g: "TBD" for g in groups} if n == "stage_ms" else "TBD") for k in keys}   # builder's state, not
           for n in PC.PRED_SWEEP}                                                            # the lead's live stamp
    stamped = dict(tbd, stage_ms={"63_64": {"fetch": (3, 5)}}, gemm_x1_ms={"63_64": (10, 11)},
                   union_extras={"63_64": (7, 9)}, joint_steps={"63_64": (1.0, 1.4)},
                   vs_cached_dr1={"63_64": (0.65, 0.75), "128_64": (0.65, 0.75)}, vs_own_floor={"63_64": (1.1, 1.3)})
    v = {n: x for n, x, _ in PC.verdicts_sweep(63, 64, _sweep_res(), pred=stamped, utc="2026-10-28T08:00:00Z")}
    assert (v["SW-STAGE-fetch"], v["SW-STAGE-attention"], v["SW-GEMM-X1"], v["SW-UNION"]) == (
        "CONFIRMED", "UNREGISTERED", "REFUTED", "CONFIRMED")
    assert (v["SW-JOINT"], v["SW-CACHED-DR1"], v["SW-OWN-FLOOR"], v["SW-CACHED-FLOOR"]) == (
        "REFUTED", "CONFIRMED", "CONFIRMED", "UNREGISTERED")
    # the dr1 share by pool: C 63 reads dr1_69 (0.7), C 128 reads dr1_44 (0.6)
    v = {n: (x, d) for n, x, d in PC.verdicts_sweep(128, 64, _sweep_res(), pred=stamped, utc="2026-10-28T08:00:00Z")}
    assert v["SW-CACHED-DR1"][0] == "REFUTED" and "dr1_44" in v["SW-CACHED-DR1"][1] and " 0.6 " in v["SW-CACHED-DR1"][1]


# ------------------------------------------------------------------ PS10
def test_PS10_load_workload_64k():
    PC = _pc()
    if not (WLM / "ctx65280_tokens.bin").is_file():
        pytest.skip("no 64K workload at %s" % WLM)
    docs, sha, n = PC.load_workload(str(WLM), 65280)
    assert tuple(docs.shape) == (55, 65280) and n == 55 and sha == WLM_SHA and str(docs.dtype) == "torch.int64"
    sums = (WLM / "SHA256SUMS").read_text()
    assert "%s  ctx65280_tokens.bin" % WLM_SHA in sums


# ------------------------------------------------------------------ PS11
def _body(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def _in_order(body, needles):
    pos = [body.index(n) for n in needles]
    assert pos == sorted(pos), list(zip(needles, pos))
    for n in needles:
        assert body.count(n) == 1, (n, body.count(n))


def test_PS11_static_probe_and_envelope():
    PC = _pc()
    nl = (NOSI / "nosa_llama.py").read_text()
    for s in ("from flashinfer.activation import silu_and_mul", "import torch.nn.functional as F",
              "def layer_norm(\n    hidden_states: torch.Tensor,\n    eps: float,\n    w: torch.Tensor,",
              "self.input_layernorm_weight = ", "self.post_attention_layernorm_variance_epsilon = ",
              "self.lm_head = ", "self.norm_weight = ", "self.norm_variance_epsilon = "):
        assert s in nl, s
    body = _body(nl, "    def decode_forward(self, hidden_states", "\nclass Llama:")
    _in_order(body, [
        "hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)",
        "qkv = F.linear(hidden_states, self.wqkv)",
        "= cache_engine.decode_update_kv(key_states, value_states, ucis, self.layer_idx, topk_idx)",
        "hidden_states = F.linear(attn_output, self.wo)",
        "hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, "
        "self.post_attention_layernorm_weight)",
        "hidden_states = F.linear(hidden_states, self.gate_up_proj)", "silu_and_mul(hidden_states, out)",
        "hidden_states = F.linear(out, self.down_proj)"])
    dec = _body(nl, "    def decode_inference(self,", "    def batch_prefill(")
    _in_order(dec, ["hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)",
                    "logits = F.linear(hidden_states, self.lm_head).float()"])
    src = inspect.getsource(PC.VerifyProbe._stack)
    for s in ("layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight", "F.linear(h, layer.wqkv)",
              "layer.wo)", "layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight",
              "layer.gate_up_proj)", "self.silu(g, out)", "F.linear(out, layer.down_proj)", "head).float()"):
        assert s in src, s
    ce = (NOSI / "cache_engine.py").read_text()
    for start, end in (("    def decode_update_has_kv_bias(self", "    def decode_update_has_kv_bias_pool("),
                       ("    def decode_update_has_kv_bias_pool(", "    def decode_update_no_kv_bias(")):
        _in_order(_body(ce, start, end), ["flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, "
                                          "self.block_size)"])
    assert "def decode_update_kv(self, key_states, value_states, kv_bias, layer_idx, topk_idx):" in ce
    assert "assert total_bsz * (S + self.max_gen_len) * _hd < 2**31" in ce
    assert "load_ids, # (H,B,M)" in (NOSI / "flash_cache_engine/flash_h2d_mask.py").read_text()
    kv, hd = 2, 128
    if MODEL_CFG.is_file():
        cfg = json.loads(MODEL_CFG.read_text())
        kv, hd = cfg["num_key_value_heads"], cfg["head_dim"]
        assert (kv, hd) == (2, 128)
    gen = 512 + 64                                     # decode_loop: max_gen_len = steps + 64
    for S, bmax in ((65280, 127), (130816, 63)):
        assert bmax * (S + gen) * kv * hd < 2 ** 31 <= (bmax + 1) * (S + gen) * kv * hd, S
    sh = (EFF / "r2_base_profile_step.sh").read_text()
    assert "p0k64c63) POOL=0 BATCH=127 " in sh and "p0k64c128) POOL=65 BATCH=127 " in sh
    assert "WLM_SHA=%s" % WLM_SHA in sh


# ------------------------------------------------------------------ PS12
def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def _step(cell, holder="118", sha="0" * 40, holders=None):
    e = _env()
    e.pop("P0_HOLDERS", None)
    e["DRYRUN"] = "1"
    if holders is not None:
        e["P0_HOLDERS"] = str(holders)
    argv = ["bash", str(EFF / "r2_base_profile_step.sh"), "--nosi", sha, holder, "t1", cell]
    return subprocess.run(argv, env=e, capture_output=True, text=True, timeout=300)


def test_PS12_step_script_sweep(tmp_path):
    for bad in ("p0sweepx", "p0k64", "p0c63b65", "P0SWEEP", "p0k64c96", "p0c128b216"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    r = _step("p0sweep")                                 # passes the cell stage, stops on the sha pin
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    for h in ("117", "118"):
        (tmp_path / ("%s.gpu_uuid" % h)).write_text(UUID + "\n")
    out = "/mnt/beegfs/ojerbi/r2-runs/p0prof/p0prof_t1_118_"
    for cell, (C, P, B, S, steps, smin, mem) in SWEEP.items():
        r = _step(cell, sha=sha, holders=tmp_path)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        o = r.stdout
        line = "cell %s: C %d (pool %d) B %d S %d steps %d STEP_MIN %d mem %s python r2_base_profile_cell.py" % (
            cell, C, P, B, S, steps, smin, mem)
        assert line in o, (cell, o[-800:])
        assert "64K workload %s" % WLM_SHA in o
        assert "NOSI_POOL_BLOCKS=%d" % P in o and "NOSI_ATTN_SPLITS=4" in o and "--batch %d " % B in o
        assert "--mem=%s" % mem in o and "--time=%d" % smin in o and out + cell in o and "STEP_MIN total %d" % smin in o
        assert "--stage-timers" in o and "--baseline %smb1/summary.json" % out in o
        assert ("--mbase" in o) == (S == 16128), cell
        assert ("--context 65280" in o and "pg19-test-ctx65280" in o) == (S == 65280), cell
        assert "--pool %d " % P in o and "pg19-test-ctx130816 " not in o
    for h, order in (("118", ["mb1", "p0c63b64", "p0c63b128", "p0c128b64", "p0k64c63", "p0k64c128"]),
                     ("117", ["mb1", "p0k64c128", "p0k64c63", "p0c128b64", "p0c63b128", "p0c63b64"])):
        r = _step("p0sweep", holder=h, sha=sha, holders=tmp_path)
        assert r.returncode == 0, r.stderr[-300:]
        got = re.findall(r"R2 p0prof step: cell (\w+): C ", r.stdout)
        assert got == order, (h, got)
        assert "STEP_MIN total 310" in r.stdout and "cells %s;" % " ".join(order) in r.stdout
