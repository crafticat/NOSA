"""[evalsplit] CPU tests of the two eval cells of human decision part 3 (2026-10-22): cell (b) r2_evalacc.py (FAST vs
EXACT accuracy equivalence) and cell (a) r2_tick_sg5.py --eval-role fast-time (the FAST timing headline), and their
holder steps (r2_evalacc_step.sh; r2_tick_sg5_step.sh cell evalfast). No CUDA. Cell (b)'s whole driver runs through
an injectable environment (ToyEnv: a deterministic toy decoder whose GEMM mode is the ops object's linear), so every
code path, the controls included, executes here. Each MUST gate has a planted failure that the test sees.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_evalsplit.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
sys.path.insert(0, str(EFF))


def _mod(name, file):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(EFF / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


EA = _mod("r2_evalacc", "r2_evalacc.py")


def _sg5():
    return _mod("r2_tick_sg5_evalsplit_cpu", "r2_tick_sg5.py")


# =====================================================================================================================
# synthetic workloads (the c16k / ctx65280 layout: <stem>_manifest.json, <stem>_tokens.bin, SHA256SUMS)
# =====================================================================================================================
S_TOY, V = 4096, 32


def _write_wl(d, ctx, docs, rows, sums=True, bad_sum=False):
    d.mkdir(parents=True, exist_ok=True)
    stem = "c16k" if ctx == 16128 else "ctx%d" % ctx
    arr = np.asarray(rows, dtype="<i4")
    raw = arr.tobytes()
    (d / (stem + "_tokens.bin")).write_bytes(raw)
    man = dict(ctx=ctx, dtype="int32-le", shape=[len(docs), ctx],
               docs=[dict(index=i, n_tokens=ctx * 4, sha256="x") for i in docs])
    (d / (stem + "_manifest.json")).write_text(json.dumps(man))
    if sums:
        h = hashlib.sha256(raw).hexdigest() if not bad_sum else "0" * 64
        (d / "SHA256SUMS").write_text("%s  %s\n" % (h, stem + "_tokens.bin"))
    return d


def _workloads(tmp, n_docs=12, gold_docs=None, n_gold=17, mismatch=None):
    """Prompt rows: documents 0 .. n_docs - 1 (S_TOY tokens); gold rows: gold_docs (S_TOY + n_gold tokens, the
    prompt as their prefix unless the doc is in mismatch)."""
    g = np.random.default_rng(5)
    full = {i: g.integers(0, V, size=S_TOY + n_gold) for i in range(n_docs)}
    gold_docs = gold_docs if gold_docs is not None else [i for i in range(n_docs) if i != 3]
    pr = _write_wl(tmp / "c", S_TOY, list(range(n_docs)), [full[i][:S_TOY] for i in range(n_docs)])
    grows = []
    for i in gold_docs:
        r = full[i].copy()
        if mismatch and i in mismatch:
            r[7] = (r[7] + 1) % V
        grows.append(r)
    gd = _write_wl(tmp / "g", S_TOY + n_gold, gold_docs, grows)
    return pr, gd, full


# =====================================================================================================================
# ToyEnv: the RealEnv interface on a deterministic toy decoder
# =====================================================================================================================
DIM = 16
W_TOY = torch.Generator().manual_seed(11)
W_MAT = torch.randn(V, DIM, generator=W_TOY)


def _feat(doc, pos, tok):
    g = torch.Generator().manual_seed((doc * 1000003 + pos * 101 + int(tok) * 7 + 3) % (2 ** 31))
    return torch.randn(DIM, generator=g)


class ToyOps:
    def __init__(self, mode, m_fix, env):
        self.mode, self.m_fix, self.attn, self.mean = mode, m_fix, None, None
        self.gemm = {"fast": "FAST: toy natural", "bi_c_strict": "G-c STRICT toy", "exact": "EXACT: toy G-b"}[mode]
        self.linear = env.fast_linear if mode == "fast" else env.exact_linear

    def describe(self):
        return dict(gemm=self.gemm, mode=self.mode, m_fix=self.m_fix)


class ToyEnv:
    """exact_batch_dep: EXACT rows depend on M (the batch-invariance control); fast_shift: FAST's perturbation scale;
    spec_break: the speculative round commits a wrong token for request 0 (the SG2 control); plant_noop: the decoder
    ignores the plant's linear (a no-op control on 'real data', L34)."""

    def __init__(self, exact_batch_dep=False, fast_shift=1e-5, spec_break=False, plant_noop=False):
        self.exact_batch_dep, self.fast_shift = exact_batch_dep, fast_shift
        self.spec_break, self.plant_noop = spec_break, plant_noop
        self.prompts = None

    def setup(self, a):
        return dict(torch=torch.__version__, gpu="toy")

    def ops(self, mode, a):
        return ToyOps(mode, a.m_fix, self)

    def exact_linear(self, x, w):                                   # row by row: no M-dependence by construction
        y = torch.stack([(w.double() @ r.double()) for r in x]).float()
        if self.exact_batch_dep:
            y = y + 1e-4 * (x.shape[0] % 5)
        return y

    def fast_linear(self, x, w):
        y = torch.stack([(w.double() @ r.double()) for r in x]).float()
        return y + self.fast_shift * ((x.shape[0] % 7) + 1) * torch.cos(torch.arange(y.shape[1], dtype=torch.float32))

    def device(self):
        return "cpu"

    def build(self, prompts, horizon):
        self.prompts = prompts
        B = len(prompts)
        first = [int(p[-1]) % V for p in prompts]
        return dict(kind="toy"), SimpleNamespace(B=B, rows=list(range(B))), first

    def subset(self, cache, st, rows):
        return dict(kind="toy-copy"), SimpleNamespace(B=len(rows), rows=[st.rows[r] for r in rows])

    def _logits(self, ops, docs, pos, toks):
        H = torch.stack([_feat(d, pos, t) for d, t in zip(docs, toks)])
        lin = ops.linear
        if self.plant_noop and getattr(ops, "mode", "") == "plant":
            lin = self.fast_linear
        return lin(H, W_MAT)

    def stepper(self, cache, st, ops):
        def step(s, inp):
            return self._logits(ops, st.rows, s, inp.tolist())
        return step

    def _greedy(self, ops, docs, first, steps):
        toks = [list([f]) for f in first]
        for i in range(1, steps):
            lg = self._logits(ops, docs, 1000 + i, [t[-1] for t in toks])
            for b, x in enumerate(lg.argmax(-1).tolist()):
                toks[b].append(int(x))
        return toks

    def frc(self, cache, st, ops, first, steps):
        return self._greedy(ops, st.rows, first, steps)

    def engine(self, cache, st, ops, a):
        return SimpleNamespace(st=st, ops=ops, a=a)

    def spec(self, eng, first, a):
        outs = self._greedy(eng.ops, eng.st.rows, first, a.spec_steps)
        if self.spec_break and 0 in eng.st.rows:
            k = eng.st.rows.index(0)
            outs[k][-1] = (outs[k][-1] + 1) % V
        acc = rej = 0
        for k, d in enumerate(eng.st.rows):                       # the stand-in draft: the target plus a hashed drop
            for i in range(1, a.spec_steps):
                lg = self._logits(eng.ops, [d], 1000 + i, [outs[k][i - 1]])[0]
                g = torch.Generator().manual_seed(d * 31 + i * 17 + a.seed)
                dr = lg + 3.0 * torch.randn(V, generator=g) * a.draft_drop
                if int(dr.argmax()) == int(lg.argmax()):
                    acc += 1
                else:
                    rej += 1
        return dict(outs=outs, stats=dict(accepted=acc, rejected=rej, cohorts=0), ticks=a.spec_steps,
                    request_ticks=a.spec_steps * len(first), bookkeeping_ok=True)

    def free(self):
        pass

    def peak_gb(self):
        return 0.0


def _args(tmp, pr, gd, extra=()):
    return ["--out", str(tmp / "out"), "--workload", str(pr), "--gold-workload", str(gd), "--biattn", "b",
            "--S", str(S_TOY), "--n-gold", "17", "--spec-steps", "6", "--batch", "8", "--alt-batch", "4",
            *extra]


def _run(tmp, env, extra=(), **wk):
    pr, gd, _ = _workloads(tmp, **wk)
    rc = EA.main(_args(tmp, pr, gd, extra), env=env)
    s = json.loads((tmp / "out" / "summary.json").read_text())
    return rc, s, {g["name"]: g for g in s["gates"]}


# =====================================================================================================================
# EV1-EV6: the pieces
# =====================================================================================================================
def test_EV1_parse_defaults_and_refusals():
    base = ["--out", "o", "--workload", "w", "--gold-workload", "g", "--biattn", "b"]
    a = EA.parse_args(base)
    assert (a.batch, a.alt_batch, a.n_gold, a.spec_steps, a.exact_gemm) == (48, 8, 256, 64, "bi_c_strict")
    assert a.dose_list == list(EA.DOSES) and a.m_fix == 4 * 48 and a.parts == ["TF", "SPEC", "NEG"]
    for bad in (["--alt-batch", "7"], ["--alt-batch", "48"], ["--batch", "1"], ["--parts", "NEG"], ["--parts", "X"],
                ["--doses", "0.25,0.01"], ["--doses", "0"], ["--n-gold", "1"], ["--exact-gemm", "fast"],
                ["--draft-drop", "1.0"], ["--S", "1024"]):
        with pytest.raises(SystemExit):
            EA.parse_args(base + bad)


def test_EV2_gold_loader(tmp_path):
    pr, gd, full = _workloads(tmp_path)
    d = EA.load_gold(str(pr), str(gd), S_TOY, 17, 8)
    assert d["doc_index"] == [0, 1, 2, 4, 5, 6, 7, 8] and d["matched"] == 11 and not d["prefix_mismatch"]
    assert torch.equal(d["prompts"][3], torch.tensor(full[4][:S_TOY], dtype=torch.int64))
    assert torch.equal(d["gold"][3], torch.tensor(full[4][S_TOY:S_TOY + 17], dtype=torch.int64))
    t2 = tmp_path / "m"
    pr2, gd2, _ = _workloads(t2, mismatch={5})                      # a planted prefix difference is named
    assert EA.load_gold(str(pr2), str(gd2), S_TOY, 17, 8)["prefix_mismatch"] == [5]
    with pytest.raises(SystemExit):                                  # too few matched documents
        EA.load_gold(str(pr), str(gd), S_TOY, 17, 12)
    with pytest.raises(SystemExit):                                  # the gold rows are too short
        EA.load_gold(str(pr), str(gd), S_TOY, 18, 8)
    t3 = tmp_path / "s"
    _write_wl(t3 / "c", S_TOY, [0], [full[0][:S_TOY]], bad_sum=True)
    _write_wl(t3 / "g", S_TOY + 17, [0], [full[0]])
    with pytest.raises(SystemExit):                                  # SHA256SUMS disagrees
        EA.load_gold(str(t3 / "c"), str(t3 / "g"), S_TOY, 17, 1)


def test_EV3_plant_linear():
    x, w = torch.randn(6, 16), torch.randn(10, 16)
    assert torch.equal(EA.PlantLinear(0.0, 1)(x, w), F.linear(x, w))            # dose 0 = the base, bitwise
    p1, p2 = EA.PlantLinear(0.25, 9), EA.PlantLinear(0.25, 9)
    y1 = [p1(x, w) for _ in range(3)]
    y2 = [p2(x, w) for _ in range(3)]
    assert all(torch.equal(u, v) for u, v in zip(y1, y2))                         # deterministic in call order
    assert not torch.equal(y1[0], y1[1])                                         # a fresh draw per call
    assert not torch.equal(y1[0], F.linear(x, w))                                # not a no-op
    big = EA.PlantLinear(0.125, 3)(torch.ones(256, 64), torch.ones(512, 64))    # relative noise of the dose
    rel = (big / 64.0 - 1.0).std().item()
    assert abs(rel - 0.125) < 0.01, rel
    ops = SimpleNamespace(linear=F.linear, attn="A", mean="M", m_fix=12, describe=lambda: dict(gemm="FAST"))
    po = EA.plant_ops(ops, 0.5, 4)
    assert po.attn == "A" and po.mean == "M" and po.m_fix == 12 and "NEGATIVE CONTROL" in po.describe()["gemm"]


def test_EV4_the_band_arithmetic():
    b = EA.BANDS["acc"]
    ok, allow = EA.inside_noise(0.004, 0.001, 0.0005, b)            # max(2 x 0.001, 3 x 0.0005) + 0.001 = 0.003
    assert not ok and abs(allow - 0.003) < 1e-12
    ok, allow = EA.inside_noise(0.0024, 0.0, 0.0005, b)             # max(0, 0.0015) + 0.001 = 0.0025
    assert ok and abs(allow - 0.0025) < 1e-12
    ok, _ = EA.inside_noise(0.02, 0.05, 0.0, b)                      # inside the noise but above the cap: OUT
    assert not ok
    v = EA.verdict(dict(acc=0.0, nll=0.5), dict(acc=0.0, nll=0.0), dict(acc=0.0, nll=0.0), EA.TF_CRITERIA)
    assert v["verdict"] == "AFFECTS" and v["failed"] == ["nll"]
    v = EA.verdict(dict(acc=0.0005, nll=-0.0005), dict(), dict(), EA.TF_CRITERIA)
    assert v["verdict"] == "EQUIVALENT"


def test_EV5_tf_arm_chunks_and_pairs():
    g = torch.Generator().manual_seed(2)
    L = torch.randn(5, 4, V, generator=g)                            # (steps, B, V): a fixed reference
    gold = torch.randint(0, V, (4, 6), generator=g)

    def stepper(rows, delta=0.0):
        def step(s, inp):
            return L[s].index_select(0, torch.tensor(rows)) + delta
        return step
    ref = EA.tf_arm(stepper([0, 1, 2, 3]), gold, keep=True)
    assert ref["logits"].shape == (5, 4, V)
    exp = (L.argmax(-1) == gold[:, 1:].t())
    assert torch.equal(ref["correct"], exp)
    lp = torch.log_softmax(L, -1)
    assert torch.allclose(ref["gold_lp"], lp.gather(-1, gold[:, 1:].t()[..., None])[..., 0].double())
    parts = [EA.tf_arm(stepper(r), gold[r], {"R": (ref["logits"], torch.tensor(r))}) for r in ([0, 1], [2, 3])]
    m = EA.merge_chunks(parts)
    assert torch.equal(m["correct"], ref["correct"]) and all(m["cmp"]["R"]["equal_steps"])
    p = EA.tf_pair(m, ref, m["cmp"]["R"])
    assert p["bitwise"] and p["d_acc"] == 0 and p["agree"] == 1.0 and p["maxd"]["p100"] == 0.0
    sh = EA.tf_arm(stepper([0, 1, 2, 3], 0.5), gold, {"R": (ref["logits"], torch.arange(4))})
    q = EA.tf_pair(sh, ref, sh["cmp"]["R"])                          # a constant shift: log p unchanged, not bitwise
    assert not q["bitwise"] and q["agree"] == 1.0 and q["maxd"]["p100"] < 1e-5
    assert abs(q["d_nll"]) < 1e-5


def test_EV6_spec_metrics_and_agreement():
    m = EA.spec_metrics([[1, 2, 3], [4, 5, 6]], dict(accepted=6, rejected=2), [[1, 2, 0], [4, 0, 6]])
    assert m["acceptance"] == 0.75 and m["gold_n"] == 6 and abs(m["gold_match"] - 4 / 6) < 1e-12
    a = EA.seq_agreement([[1, 2, 3], [4, 5, 6], [7, 8]], [[1, 2, 3], [4, 9, 6], [7, 8, 9]])
    assert a["full"] == 1 and a["first_divergence"] == [None, 1, 2] and a["median_first_divergence"] == 2
    assert EA._binom_se(0.5, 100, 0.5, 100) == pytest.approx(math.sqrt(0.25 * 0.02))


# =====================================================================================================================
# EV7-EV8: the whole driver on the toy decoder, and every MUST gate's planted failure
# =====================================================================================================================
MUSTS = ("EA_P0_gold_prefix", "EA_E_determinism", "EA_E_batch_invariance", "EA_NEG_precondition", "EA_NEG_detected",
         "EA_SG2_spec_equals_serial", "EA_SPEC_batch_invariance", "EA_SPEC_bookkeeping", "EA_VERDICT_FAST")


def test_EV7_the_driver_passes_on_an_equivalent_fast(tmp_path, capsys):
    rc, s, g = _run(tmp_path, ToyEnv())
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "R2_EVALACC_DONE failed=0" and rc == 0, [x for x in out if " FAIL" in x]
    for n in MUSTS:
        assert g[n]["must"] and g[n]["passed"], (n, g[n]["detail"])
    r = s["results"]
    assert r["verdict"]["fast_vs_exact"] == "EQUIVALENT" and set(r["verdict"]["detail"]["criteria"]) == {
        "acc", "nll", "accept", "gold"}
    assert r["NEG"]["%g" % 0.25]["verdict"]["verdict"] == "AFFECTS"           # the top dose is detected
    assert r["NEG"]["%g" % 2 ** -8]["verdict"]["verdict"] == "EQUIVALENT"     # rounding-scale noise passes
    assert "DIFFER" in g["EA_R_F_NI"]["detail"] and "bitwise equal" in g["EA_R_F_replica"]["detail"]
    assert r["TF"]["positions"] == 16 * 8 and r["TF"]["pairs"]["F_alt_vs_F"]["bitwise"] is False
    assert s["meta"]["doc_index"] == [0, 1, 2, 4, 5, 6, 7, 8] and s["meta"]["chunks"] == 2
    rec = EA.check_record(str(tmp_path / "out" / "summary.json"))           # cell (a) accepts this record
    assert rec["verdict"] == "EQUIVALENT"
    assert EA.main(_args(tmp_path, tmp_path / "c", tmp_path / "g"), env=ToyEnv()) == 2   # --out exists: refused


@pytest.mark.parametrize("plant,gate_name", [
    (dict(env=dict(exact_batch_dep=True)), "EA_E_batch_invariance"),
    (dict(env=dict(spec_break=True)), "EA_SG2_spec_equals_serial"),
    (dict(env=dict(fast_shift=2.0)), "EA_VERDICT_FAST"),
    (dict(env=dict(plant_noop=True)), "EA_NEG_precondition"),
    (dict(wl=dict(mismatch={2})), "EA_P0_gold_prefix")])
def test_EV8_each_must_gate_sees_its_plant(tmp_path, plant, gate_name):
    rc, s, g = _run(tmp_path, ToyEnv(**plant.get("env", {})), **plant.get("wl", {}))
    assert g[gate_name]["must"] and not g[gate_name]["passed"], g[gate_name]
    assert rc >= 1
    if gate_name == "EA_NEG_precondition":
        assert not g["EA_NEG_detected"]["passed"]                              # a no-op plant is never "detected"
    if gate_name != "EA_VERDICT_FAST":
        with pytest.raises(SystemExit):
            EA.check_record(str(tmp_path / "out" / "summary.json"))
    else:
        assert s["results"]["verdict"]["fast_vs_exact"] == "AFFECTS"
        with pytest.raises(SystemExit):
            EA.check_record(str(tmp_path / "out" / "summary.json"))


def test_EV9_check_record_refusals(tmp_path):
    def w(name, gates, verdict="EQUIVALENT"):
        p = tmp_path / name
        p.write_text(json.dumps(dict(meta=dict(nosi_commit="c"), gates=gates,
                                     results=dict(verdict=dict(fast_vs_exact=verdict, scope="s")))))
        return str(p)
    need = [dict(name=n, must=True, passed=True) for n in ("EA_VERDICT_FAST", "EA_NEG_detected",
                                                            "EA_E_batch_invariance")]
    assert EA.check_record(w("good.json", need))["musts"] == 3
    for p in (w("fail.json", need + [dict(name="EA_x", must=True, passed=False)]),
              w("aff.json", need, "AFFECTS"), w("noneg.json", need[:1] + need[2:]), w("empty.json", []),
              str(tmp_path / "missing.json"), ""):
        with pytest.raises(SystemExit):
            EA.check_record(p)
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(SystemExit):
        EA.check_record(str(tmp_path / "bad.json"))


# =====================================================================================================================
# EV10-EV11: cell (a) in r2_tick_sg5.py
# =====================================================================================================================
SG5_BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
EF_OK = ["--eval-role", "fast-time", "--modes", "fast", "--backends", "b2", "--layout", "pool", "--X", "64", "--bare",
         "--mem-cap-gb", "84", "--accuracy-record", "r.json"]


def test_EV10_fast_time_parse_rules():
    G = _sg5()
    a = G.parse_args(SG5_BASE)
    assert a.eval_role == "" and a.accuracy_record == "" and a.mem_cap_gb == 0 and a.eval_bands_v is None
    a = G.parse_args(SG5_BASE + EF_OK + ["--eval-bands", "55:125:30:70:27:65", "--variants",
                                         "fused/batched,torch/per_call"])
    assert a.eval_bands_v == [55.0, 125.0, 30.0, 70.0, 27.0, 65.0] and a.modes == ["fast"]
    assert G.parse_args(SG5_BASE + EF_OK[:3] + ["fast,exact"] + EF_OK[4:]).modes == ["fast", "exact"]

    def without(flag, n=2):
        i = EF_OK.index(flag)
        return EF_OK[:i] + EF_OK[i + n:]
    for bad in (EF_OK[:3] + ["exact"] + EF_OK[4:],                     # the headline must be FAST
                EF_OK[:5] + ["sc"] + EF_OK[6:],                         # b2 only
                without("--bare", 1), without("--mem-cap-gb"), without("--accuracy-record"),
                EF_OK[:7] + ["stream"] + EF_OK[8:],                     # the pool layout
                EF_OK + ["--stack", "full", "--shared-unchecked"],
                EF_OK + ["--sweep", "96:64"],
                EF_OK + ["--eval-bands", "1:2:3"], EF_OK + ["--eval-bands", "2:1:3:4:5:6"],
                ["--mem-cap-gb", "84"], ["--accuracy-record", "r.json"], ["--eval-bands", "1:2:3:4:5:6"]):
        with pytest.raises(SystemExit):
            G.parse_args(SG5_BASE + bad)


def test_EV11_fast_time_metrics_and_gates(tmp_path):
    G = _sg5()
    a = G.parse_args(SG5_BASE + EF_OK + ["--batch", "4", "--steps", "9", "--eval-bands", "55:125:30:70:27:65"])
    pt = dict(label="SG5_fast_b2_fused_batched", gemm="fast", rebuild="batched",       # one_point: gemm=ops.mode
              bare_round=dict(ticks=10, committed=32, wall_s=0.8), bare_T_tick=dict(mean=50.0, median=40.0),
              round=dict(device_ms=dict(tick=dict(mean=60.0, median=50.0))), committed=32, ticks=10, wall_s=1.0)
    e = G.eval_metrics(pt)
    h = e["headline"]
    assert h["steady_tok_s_median"] == pytest.approx(32 / (10 * 0.040))      # 80
    assert h["steady_tok_s_mean"] == pytest.approx(32 / (10 * 0.050))        # 64
    assert h["effective_tok_s"] == pytest.approx(32 / 0.8)                   # 40
    assert e["instrumented"]["steady_tok_s_median"] == pytest.approx(64.0)
    meta = dict(peak_gb=83.5, accuracy_record=dict(path="r.json"))
    gates = []
    G.eval_gates(a, meta, dict(points=[dict(pt)]), gates)
    g = {x["name"]: x for x in gates}
    for n in ("EF_headline_fast", "EF_bare_SG5_fast_b2_fused_batched", "EF_mem_cap", "EF_accuracy_record"):
        assert g[n]["must"] and g[n]["passed"], (n, g[n]["detail"])
    assert "steady_median 80.00 in [55, 125]: IN" in g["R_EF_bands"]["detail"]
    assert "effective 40.00 in [27, 65]: IN" in g["R_EF_bands"]["detail"]
    assert "HEADLINE (bare): steady 80.00" in g["R_EF_SG5_fast_b2_fused_batched"]["detail"]
    gates = []                                                          # the planted failures
    G.eval_gates(a, dict(peak_gb=84.2, accuracy_record=None),
                 dict(points=[dict(pt, gemm="exact", bare_round=dict(ticks=10, committed=31, wall_s=0.8))]),
                 gates)
    g = {x["name"]: x for x in gates}
    assert not any(g[n]["passed"] for n in ("EF_headline_fast", "EF_bare_SG5_fast_b2_fused_batched", "EF_mem_cap",
                                            "EF_accuracy_record"))
    with pytest.raises(SystemExit):
        G.check_accuracy_record(str(tmp_path / "missing.json"))
    assert G.main(SG5_BASE[:1] + [str(tmp_path / "o2")] + SG5_BASE[2:] + ["--model", "m"] + EF_OK[:-1]
                  + [str(tmp_path / "m.json")]) == 2
    assert not (tmp_path / "o2").exists()                                # refused before any work


def test_EV12_the_default_bare_path_is_unchanged():
    """The bare round's new counts exist only under --eval-role (the default records bare_T_tick as before)."""
    src = (EFF / "r2_tick_sg5.py").read_text()
    seg = src[src.index("    if bare:\n        eng, svc, tp = make_engine("):src.index("XP_TICK_BAND = 0.15")]
    assert 'if getattr(a, "eval_role", ""):' in seg and 'info["bare_round"]' in seg
    assert seg.index('if getattr(a, "eval_role", ""):') < seg.index('info["bare_round"]')
    assert 'info["bare_T_tick"] = bs["device_ms"].get("tick")' in seg
    main_src = src[src.index("def main(argv=None)"):]
    assert 'if a.eval_role == "fast-time":' in main_src and "eval_gates(a, meta, res, gates)" in main_src
    assert main_src.index("check_accuracy_record(a.accuracy_record)") < main_src.index("os.makedirs(a.out)")


# =====================================================================================================================
# EV13: the holder steps (lint, DRYRUN on a clean checkout of this branch)
# =====================================================================================================================
def _clean_head():
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--",
                                    "nosi"], capture_output=True, text=True)
    except FileNotFoundError:                                        # no git in this environment: no DRYRUN
        return None
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        return None
    return head.stdout.strip()


def test_EV13_the_step_scripts(tmp_path):
    for name in ("r2_evalacc_step.sh", "r2_tick_sg5_step.sh"):
        text = (EFF / name).read_text()
        for word in ("awk", "perl", "python3", " bc ", "node "):
            for ln in text.splitlines():
                if not ln.lstrip().startswith("#"):
                    assert word not in ln, (name, word, ln)
    sg = (EFF / "r2_tick_sg5_step.sh").read_text()
    root = next(ln for ln in sg.splitlines() if ln.startswith("NOSI_EVAL=")).split()[0][len("NOSI_EVAL="):]
    assert root == "/mnt/beegfs/ojerbi/nosi-r2-f3ld"                    # [f3ld] (was nosi-r2-f3lia / -flean / -admitfill / -acc216)
    ea = (EFF / "r2_evalacc_step.sh").read_text()
    assert next(ln for ln in ea.splitlines() if ln.startswith("NOSI=")).split()[0] == "NOSI=/mnt/beegfs/ojerbi/nosi-r2-sg5par"
    # [hostfix] stack2 4ddb795 (cherry-picked) moved the stack1 cells' ROOT from integ6 to stack2
    assert next(ln for ln in sg.splitlines() if ln.startswith("NOSI_INTEG=")).split()[0].endswith("nosi-r2-stack2")
    head = _clean_head()
    if head is None:
        pytest.skip("not a clean git checkout: the DRYRUN needs one")
    uid = "33cfb929-315c-8d27-5622-1169affb7e04"
    (tmp_path / "418.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(EACC_HOLDERS=str(tmp_path), SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    want = {"evalacc": ["--batch 48", "--alt-batch 8", "--n-gold 256", "--spec-steps 64", "--time=120"],
            "evalacc-mini": ["--batch 8", "--alt-batch 4", "--n-gold 32", "--spec-steps 16", "--time=45"]}
    for cell, has in want.items():
        r = subprocess.run(["bash", str(EFF / "r2_evalacc_step.sh"), "--nosi", head, "418", "t1", cell], env=env,
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            assert Path("/mnt/beegfs/ojerbi/nosi-r2-sg5par").resolve() != ROOT.resolve(), r.stderr[-300:]
            pytest.skip("DRYRUN stopped outside the sg5par ROOT: %s" % r.stderr.strip()[-200:])
        line = r.stdout.strip().splitlines()[-1]
        got = line.split("r2_evalacc.py ", 1)[1].replace("\\,", ",")
        for w in has + ["--exact-gemm bi_c_strict", "--parts TF,SPEC,NEG", "--S 16128",
                        "--gold-workload /mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280"]:
            assert w in got or w in line, (cell, w)
        assert "claim /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_active_418 (one per holder)" in r.stdout
        sfx = cell.replace("-", "")
        assert "--job-name=eacc-t1-418-%s " % sfx in line and "/r2-runs/evalacc/eacc_t1_418_%s " % sfx in line
        pa = EA.parse_args(got.split())
        assert pa.exact_gemm == "bi_c_strict" and pa.parts == ["TF", "SPEC", "NEG"]
    for bad in ((), ("evalacc", "x"), ("evalaccx",)):
        rb = subprocess.run(["bash", str(EFF / "r2_evalacc_step.sh"), "--nosi", head, "418", "t1", *bad], env=env,
                            capture_output=True, text=True, timeout=60)
        assert rb.returncode == 2, (bad, rb.stderr)
    r = subprocess.run(["bash", str(EFF / "r2_tick_sg5_step.sh"), "--nosi", head, "418", "t1", "evalfast"], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-300:]
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ("--modes fast --backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--steps 48",
              "--variants fused/batched,torch/per_call", "--layout pool --X-pool 1512 --arrival-lag 2 --bare",
              "--skip-shipped --eval-role fast-time --mem-cap-gb 84",
              "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json",
              "--eval-bands 55:125:30:70:27:65", "--mem=240G", "PYTHONPATH=/mnt/beegfs/ojerbi/nosi-r2-f3ld/nosi"):
        assert w in got or w in line, w
    assert "claim /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_active_418 (one per holder)" in r.stdout
    for w in ("--pool-baseline", "--stack", "--sweep ", "--host-profile", "--gc-freeze", "--lean-time", "--b2-resolve",
              "--wait-index", "--b2-fewer-reads"):                     # [hostfix] evalfast keeps its arguments
        assert w not in got, w
    pa = _sg5().parse_args(got.split())
    assert pa.eval_role == "fast-time" and pa.batch == 216 and pa.X_pool == 1512 and pa.log_slots == -1
    assert pa.variant_list == [("fused", "batched"), ("torch", "per_call")]


# =====================================================================================================================
# EV14 [sg5par]: one claim per holder (the claim's mkdir through the DRYRUN-only probe; a global claim as the control)
# =====================================================================================================================
@pytest.mark.parametrize("name,var,prefix,cell", [("r2_tick_sg5_step.sh", "SG5", "sg5_active", "evalfast"),
                                                  ("r2_evalacc_step.sh", "EACC", "eacc_active", "evalacc")])
def test_EV14_one_claim_per_holder(tmp_path, name, var, prefix, cell):
    """EV14: two holders get two distinct claims (both taken at the same time); a second step in the same holder is
    refused; the real mkdir and rmdir use the same CLAIM (OUTROOT/<prefix>_<holder>). NEGATIVE CONTROL: the same
    script with the claim name of the old global claim (<prefix>, no holder) refuses the second holder, so this test
    sees a shared claim. The probe runs before every git pin, so no clean checkout is necessary."""
    text = (EFF / name).read_text()
    assert ("CLAIMNAME=%s_$JOB " % prefix) in text and "CLAIM=$OUTROOT/$CLAIMNAME\n" in text
    assert 'mkdir "$CLAIM" 2>/dev/null ||' in text and 'rmdir "$CLAIM" || ok=0' in text
    assert ('mkdir "$%s_CLAIM_PROBE/$CLAIMNAME"' % var) in text
    assert ('[ "$DRY" = 1 ] || stop "%s_CLAIM_PROBE is for DRYRUN (the CPU test) only"' % var) in text
    assert ('"$OUTROOT/%s"' % prefix) not in text                     # no global claim left
    assert text.index("CLAIM=$OUTROOT/$CLAIMNAME") < text.index('[ "$(git -C "$NOSI" rev-parse HEAD)"')
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index("CLAIMNAME=")
    holders = tmp_path / "holders"
    holders.mkdir()
    for job, uid in ((418, "33cfb929-315c-8d27-5622-1169affb7e04"), (419, "f31fc723-9dfe-407a-9947-28173556232a")):
        (holders / ("%d.gpu_uuid" % job)).write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def take(script, probe, job):
        e = dict(env, DRYRUN="1", **{var + "_HOLDERS": str(holders), var + "_CLAIM_PROBE": str(probe)})
        return subprocess.run(["bash", str(script), "--nosi", "0" * 40, str(job), "t1", cell], env=e,
                              capture_output=True, text=True, timeout=60)

    probe = tmp_path / "probe"
    probe.mkdir()
    r = take(EFF / name, probe, 418)
    assert r.returncode == 0 and ("took %s_418" % prefix) in r.stdout, r.stderr[-300:]
    r = take(EFF / name, probe, 418)                                   # the same holder: refused
    assert r.returncode == 1 and ("%s_418 is held" % prefix) in r.stderr, r.stderr[-300:]
    r = take(EFF / name, probe, 419)                                   # the other holder, while 418 holds its claim
    assert r.returncode == 0 and ("took %s_419" % prefix) in r.stdout, r.stderr[-300:]
    assert sorted(p.name for p in probe.iterdir()) == ["%s_418" % prefix, "%s_419" % prefix]
    r = take(EFF / name, probe, 419)
    assert r.returncode == 1 and ("%s_419 is held" % prefix) in r.stderr
    # NEGATIVE CONTROL: the old global claim (one name for every holder) refuses the second holder
    old = ("CLAIMNAME=%s_$JOB " % prefix)
    assert text.count(old) == 1
    glob_script = tmp_path / ("global_" + name)
    glob_script.write_text(text.replace(old, "CLAIMNAME=%s " % prefix))
    gprobe = tmp_path / "gprobe"
    gprobe.mkdir()
    r = take(glob_script, gprobe, 418)
    assert r.returncode == 0 and ("took %s\n" % prefix) in r.stdout, r.stderr[-300:]
    r = take(glob_script, gprobe, 419)
    assert r.returncode == 1 and ("%s is held" % prefix) in r.stderr, r.stderr[-300:]
    assert sorted(p.name for p in gprobe.iterdir()) == [prefix]
