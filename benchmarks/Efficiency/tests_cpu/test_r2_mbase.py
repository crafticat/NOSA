"""CPU tests of r2-mbase: the matched-baseline cells (r2_base_cell --pool / --mbase; r2_base_fused_step.sh mb1, mb2c*,
fit<C>b<B>) and the GEMM scan gm1 (r2_bi_microgate --gemm-scan; r2_bi_step.sh ... gm1). No CUDA, no Triton launch.

MB1  r2_base_cell.parse_args: --pool in 0 .. 65; --pool > 0 with the R2 arm refused (the R2 body has no victim pool).
MB2  the prefill-peak model against its inputs (2175791 output 2) and one point outside its fit (basef 2190876 SH C 63
     B 219 peak 52.925 GB); mbase_batch: C 128 -> 216 (2175791: 216 fits, 224 OOM), the registered B of every mb cell.
MB3  verdicts_mbase: inside / outside the registered bands, REPORT at an unregistered (C, B), MB-F only with R2,
     2175791's own eager medians fall inside the C 128 and C 63 bands (the band is not built to miss its basis).
MB4  r2_base_fused_step.sh: malformed cells STOP before any pin (fit C / B forms, bare "mb2"); at a clean checkout of the
     step's ROOT, DRYRUN prints each mb / fit cell's C, pool, B, arms, steps, STEP_MIN, --mem, NOSI_POOL_BLOCKS,
     --pool and --mbase (skipped elsewhere, as S13).
GM1  lm_head's shape from the model config; scan_shapes; scan_plan (EXACT pairs (u B, 4 B)); refusals.
GM2  per_step_totals: 32 x four linears + lm_head; a missing or a repeated shape is refused.
GM3  gm1_verdicts: the registered points CONFIRMED, moved values REFUTED, absent totals REPORT / omitted.
GM4  the PRODUCTION entry point main([... --gemm-scan --device cpu --shape-div 64 ...]): exit 0, summary.json with
     GM-RAN PASS, every timing, per-step totals for every M and pair, padded invariance 0 rows; refusals (--device
     without --gemm-scan, --gemm-scan with --biattn, --out exists).
GM5  r2_bi_step.sh: a wrong cell and 6 arguments STOP / usage; at a clean checkout of the ROOT, DRYRUN gm1 prints the
     gm1 python arguments, no piece 3 argument, the _gm1 output and --time=15 (skipped elsewhere).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_mbase.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
NOSA8B_CONFIG = Path("/mnt/beegfs/ojerbi/models/NOSA-8B/config.json")
HOLDER = "2190876"                       # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-mbase"


def _bc():
    sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


def _mg():
    if "nosi" not in sys.modules:                       # nosi/__init__ imports the model: load only the modules needed
        pkg = types.ModuleType("nosi")
        pkg.__path__ = [str(NOSI)]
        sys.modules["nosi"] = pkg
    spec = importlib.util.spec_from_file_location("r2_bi_microgate_mbase", str(EFF / "r2_bi_microgate.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    return mg


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def _clean_head():
    """The checkout's HEAD if this checkout IS the step scripts' ROOT and has no tracked change, else None."""
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


# ------------------------------------------------------------------ MB
def test_MB1_pool_arguments():
    BC = _bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--mbase"])
    assert (a.pool, a.mbase, a.arms) == (65, True, ["SH"])
    assert BC.parse_args(base).pool == 0 and not BC.parse_args(base).mbase
    assert BC.parse_args(base + ["--r2", "fused", "--pool", "0", "--mbase"]).arms == ["SH", "R2"]
    for bad in (["--pool", "65"], ["--arms", "SH", "--pool", "66"], ["--arms", "SH", "--pool", "-1"]):
        with pytest.raises(SystemExit):
            BC.parse_args(base + bad)


def test_MB2_peak_model_and_batches():
    BC = _bc()
    m = BC.mbase_peak_model
    for C, B, peak in ((63, 64, 28.93), (128, 64, 37.69), (128, 216, 82.04)):     # 2175791 output 2 (the fit)
        assert abs(m(C, B) / peak - 1) < 0.01, (C, B, m(C, B))
    assert abs(m(63, 219) - 52.925) / 52.925 < 0.005                              # basef 2190876 SH (outside the fit)
    assert m(128, 224) > BC.MBASE_PEAK["fit_GB"]                                  # 2175791: B 224 OOM at C 128
    got = {C: BC.mbase_batch(C) for C in (63, 79, 96, 113, 128)}
    assert got == {63: 400, 79: 328, 96: 280, 113: 240, 128: 216}, got
    for (C, B) in BC.PRED_MBASE:                          # every registered band is at the model's batch, except C 63
        assert B == (344 if C == 63 else got[C]), (C, B)
        pt, lo, hi = BC.PRED_MBASE[(C, B)]
        assert lo < pt < hi


def test_MB3_verdicts_mbase():
    BC = _bc()

    def sh(ms, peak):
        return dict(rates=dict(steady_step_ms=ms), gpu_peak_alloc_GB=peak)
    v = {n: x for n, x, _ in BC.verdicts_mbase(128, 216, sh(68.0, 82.0), None)}
    assert v == dict(**{"MB-STEP": "CONFIRMED", "MB-FIT": "CONFIRMED", "MB-R2C": "REPORT", "MB-TOK": "REPORT"}), v
    v = {n: x for n, x, _ in BC.verdicts_mbase(128, 216, sh(71.0, 70.0), None)}
    assert (v["MB-STEP"], v["MB-FIT"]) == ("REFUTED", "REFUTED"), v
    v = {n: x for n, x, _ in BC.verdicts_mbase(128, 208, sh(66.0, 80.0), None)}
    assert v["MB-STEP"] == "REPORT"
    for C, B, eager in ((128, 216, 68.336639), (63, 344, 179.87)):                 # 2175791 eager medians
        v = {n: x for n, x, _ in BC.verdicts_mbase(C, B, sh(eager, BC.mbase_peak_model(C, B)), None)}
        assert v["MB-STEP"] == "CONFIRMED", (C, B)
    r2 = dict(rates=dict(steady_step_ms=178.8))
    v = {n: x for n, x, _ in BC.verdicts_mbase(63, 344, sh(180.6, 72.3), r2)}
    assert v["MB-F"] == "CONFIRMED", v
    v = {n: x for n, x, _ in BC.verdicts_mbase(63, 344, sh(180.6, 72.3), dict(rates=dict(steady_step_ms=182.0)))}
    assert v["MB-F"] == "REFUTED", v
    assert BC.verdicts_mbase(128, 216, None, None)[0][1] == "NOT_RUN"
    d = dict((n, x) for n, _, x in BC.verdicts_mbase(128, 216, sh(68.0, 82.0), None))
    assert "3176.5 tok/s" in d["MB-TOK"] and "MODEL" in d["MB-R2C"]


def _fused_step(cell, sha="0" * 40, dry=True):
    e = _env()
    if dry:
        e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_base_fused_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def test_MB4_step_script_cells():
    for bad in ("mb2", "mb2c80", "fit64b8", "fit79b12", "fit79b408", "fit079b16", "fit79b016", "fit79b", "fitb8",
                "fit79b8x"):
        r = _fused_step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    want = {"mb1": (128, 65, 216, "SH", 512, 35, "240G", True), "mb2c113": (113, 50, 240, "SH", 512, 35, "240G", True),
            "mb2c96": (96, 33, 280, "SH", 512, 40, "340G", True), "mb2c79": (79, 16, 328, "SH", 512, 45, "340G", True),
            "mb2c63": (63, 0, 344, "SH,R2", 512, 70, "340G", True), "fit96b288": (96, 33, 288, "SH", 16, 30, "340G", False),
            "fit128b224": (128, 65, 224, "SH", 16, 30, "240G", False), "b219": (63, 0, 219, "SH,R2", 512, 50, "220G", False)}
    for cell, (C, P, B, arms, steps, smin, mem, mb) in want.items():
        r = _fused_step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = "cell %s: C %d (pool %d) B %d arms %s steps %d STEP_MIN %d mem %s" % (cell, C, P, B, arms, steps, smin, mem)
        assert line in r.stdout, (cell, r.stdout[-600:])
        assert "NOSI_POOL_BLOCKS=%d" % P in r.stdout and "--pool %d" % P in r.stdout
        assert "--batch %d" % B in r.stdout and "--mem=%s" % mem in r.stdout and "--time=%d" % smin in r.stdout
        assert ("--mbase" in r.stdout) == mb, cell
        assert "basef_t1_%s_%s" % (HOLDER, cell) in r.stdout


# ------------------------------------------------------------------ GM
def test_GM1_shapes_and_plan():
    mg = _mg()
    assert NOSA8B_CONFIG.is_file(), "the model config %s is needed" % NOSA8B_CONFIG
    cfg = json.loads(NOSA8B_CONFIG.read_text())
    assert mg.lm_head_from_config(cfg) == mg.LM_HEAD == (73448, 4096)
    assert mg.lm_head_from_config(dict(cfg, vocab_size=128256)) != mg.LM_HEAD                 # negative control
    s = mg.scan_shapes()
    assert set(s) == {"wqkv", "wo", "gate_up", "down", "lm_head"} and s["lm_head"] == mg.LM_HEAD
    assert mg.scan_shapes(64)["down"] == (64, 256)
    nat, pad = mg.scan_plan(mg.SCAN_MS, mg.SCAN_EXACT_B)
    assert set(mg.SCAN_MS) <= set(nat) and {287, 574, 861, 1148, 216, 432, 648, 864, 192} <= set(nat)
    for p in ((287, 1148), (574, 1148), (861, 1148), (1148, 1148), (216, 864), (64, 256), (256, 256), (861, 861)):
        assert p in pad, p
    assert all(live <= mfix for live, mfix in pad) and len(pad) == len(set(pad))
    assert mg.scan_plan([8], [4], U=1) == ([4, 8], [(4, 8), (8, 8)])
    for bad in (([], [4]), ([0], [4]), ([8], [0])):
        with pytest.raises(ValueError):
            mg.scan_plan(*bad)


def test_GM2_per_step_totals():
    mg = _mg()
    rows = [dict(shape=s, M=8, ms=v) for s, v in (("wqkv", 1.0), ("wo", 2.0), ("gate_up", 3.0), ("down", 4.0),
                                                   ("lm_head", 5.0))]
    assert mg.per_step_totals(rows, "M") == {8: 32 * 10.0 + 5.0}
    prow = [dict(r, pair=[8, 16]) for r in rows]
    assert mg.per_step_totals(prow, "pair") == {(8, 16): 325.0}
    with pytest.raises(ValueError):
        mg.per_step_totals(rows[:4], "M")
    with pytest.raises(ValueError):
        mg.per_step_totals(rows + [rows[0]], "M")


def test_GM3_verdicts():
    mg = _mg()
    nat = {M: pt for M, (pt, _, _) in mg.PRED_GM1.items()}
    nat.update({287: 25.1, 574: 45.0})
    pad = {(287, 1148): 96.1, (574, 1148): 96.2, (861, 1148): 96.3, (1148, 1148): 96.4}
    v = {n: x for n, x, _ in mg.gm1_verdicts(nat, pad)}
    assert all(v["GM-NAT-%d" % M] == "CONFIRMED" for M in mg.PRED_GM1), v
    assert v["GM-EXACT-287-1148"] == "CONFIRMED" and v["GM-EXACT-861-1148"] == "CONFIRMED"
    assert v["GM-EXACT-574-1148"] == "REPORT" and v["GM-DENSE-FAST"] == "CONFIRMED" and v["GM-DENSE-EXACT"] == "CONFIRMED"
    d = {n: x for n, _, x in mg.gm1_verdicts(nat, pad)}
    assert "not binding" in d["GM-DENSE-FAST"] and "BINDING" in d["GM-DENSE-EXACT"]
    moved = dict(nat, **{861: 90.0, 64: 20.0})
    v = {n: x for n, x, _ in mg.gm1_verdicts(moved, dict(pad, **{(287, 1148): 50.0}))}
    assert (v["GM-NAT-861"], v["GM-NAT-64"], v["GM-EXACT-287-1148"]) == ("REFUTED", "REFUTED", "REFUTED"), v
    v = {n: x for n, x, _ in mg.gm1_verdicts({}, {})}
    assert all(x == "REPORT" for x in v.values()) and "GM-DENSE-FAST" not in v and "GM-DENSE-EXACT" not in v


def test_GM4_gemm_scan_through_main(tmp_path):
    mg = _mg()
    out = tmp_path / "gm"
    args = ["--out", str(out), "--gemm-scan", "--device", "cpu", "--shape-div", "64", "--ms", "8,16", "--exact-b", "4",
            "--U", "3", "--warm", "1", "--reps", "2"]
    assert mg.main(args) == 0
    s = json.loads((out / "summary.json").read_text())
    g = {x["name"]: x for x in s["gates"]}
    assert g["GM-RAN"]["passed"] and g["GM-RAN"]["must"] and not g["R_GM_padded_invariance"]["must"]
    nat, pad = mg.scan_plan([8, 16], [4], 3)
    assert len(s["tables"]["natural"]) == 5 * len(nat) and len(s["tables"]["padded"]) == 5 * len(pad)
    assert sorted(int(k) for k in s["tables"]["per_step_natural"]) == nat
    assert sorted(tuple(int(x) for x in k.split("_")) for k in s["tables"]["per_step_padded"]) == pad
    assert all(v["rows_differing_garbage"] == 0 and v["rows_differing_zero"] == 0
               for v in s["tables"]["invariance"].values())
    assert s["meta"]["device"] == "cpu" and s["meta"]["shape_div"] == 64 and s["meta"]["cell"] == "gm1"
    assert mg.main(args) == 2                                                     # --out exists
    assert mg.main(["--out", str(tmp_path / "a"), "--device", "cpu"]) == 2
    assert mg.main(["--out", str(tmp_path / "b"), "--gemm-scan", "--biattn", "x.py"]) == 2
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_GM5_bi_step_gm1(tmp_path):
    script = EFF / "r2_bi_step.sh"
    (tmp_path / "117.gpu_uuid").write_text("f31fc723-9dfe-407a-9947-28173556232a\n")
    e = dict(_env(), BI_HOLDERS=str(tmp_path), DRYRUN="1")

    def run(*extra, sha="0" * 40):
        return subprocess.run(["bash", str(script), "--nosi", sha, "117", "t1", *extra], env=e, capture_output=True,
                              text=True, timeout=120)
    r = run("gm2")
    assert r.returncode == 1 and "cell 'gm2'" in r.stderr
    r = run("gm1", "x")
    assert r.returncode == 2 and "usage" in r.stderr
    r = run("gm1")                                       # passes the cell stage, stops on the sha pin
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    r = run("gm1", sha=sha)
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
    assert "cell gm1 STEP_MIN 15" in r.stdout and "--time=15" in r.stdout
    # printf %q escapes the commas (the bi1 record shows "--batches 64\,128")
    assert ("--gemm-scan --ms 64\\,128\\,256\\,320\\,512\\,768\\,861\\,1024\\,1148 --exact-b 287\\,216\\,64 --U 3"
            in r.stdout), r.stdout[-800:]
    assert "--biattn" not in r.stdout and "bi_t1_117_gm1" in r.stdout and "--job-name=bi-t1-117-gm1" in r.stdout
    r = run(sha=sha)                                     # the default cell is unchanged
    assert r.returncode == 0 and "--biattn /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_117.step/piece3_bi_paged_attention.py" in r.stdout
