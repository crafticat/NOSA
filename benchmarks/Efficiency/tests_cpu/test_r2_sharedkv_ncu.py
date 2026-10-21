"""CPU tests of cell S3 (benchmarks/Efficiency/r2_sharedkv_ncu.py, r2_sharedkv_ncu_step.sh; docs/R2_SHAREDKV2_KEYS_S3.md):
the summary on synthetic ncu raw CSV exports (the parser, the window check, the cold-L2 byte bounds, the ratios and
their negatives), the probe's call marking through the production tick on the CPU tiny model, and the step's DRYRUN.
SOURCE ONLY; the lead runs them on a compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_sharedkv_ncu.py
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
PIECE3_BLOB = "39ed5e1894ca1662f92e384a10e01a4eaab99042"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RN = _load(EFF / "r2_sharedkv_ncu.py", "r2_sharedkv_ncu_under_test")
PB = RN.PAGE_BYTES
NVTX_COL = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
COLS = ["ID", "Process ID", NVTX_COL, "Kernel Name"] + list(RN.REQUIRED) + [RN.OPTIONAL[0]]


def write_case(d, calls=2, sum_pages=100, union_pages=40, pr_pages=50.0, sh_pages=42.0, drop=None, untagged=False,
               diff_rows=0, hit=True):
    """A synthetic variant 'all': P_all/cell.json and all_raw.csv (a units row included, as ncu writes it). Per marked
    call: skv|pr = _bi_split_kernel (pr_pages pages of DRAM) + _bi_combine_kernel (1000 B); skv|sh =
    _bi_shared_kernel (sh_pages) + combine. drop = (kind, call) leaves that call's main kernel out."""
    os.makedirs(d / "P_all")
    marked = [dict(call=j, rows=128, first="per_row", sum_pages=sum_pages, union_pages=union_pages)
              for j in range(1, 1 + calls)]
    cell = dict(meta=dict(backend="resident"), ncu=dict(window=[1, calls], full_calls=calls + 3, marked=marked),
                probe=dict(diff_rows=diff_rows, rows_compared=999, stats=dict(shared=9), tile_ratio=0.4, row_ratio=0.4,
                           tiles={}))
    (d / "P_all" / "cell.json").write_text(json.dumps(cell))
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(COLS)
    w.writerow(["", "", "", ""] + ["byte"] * len(COLS[4:]))                 # the units row (no numeric ID)
    i = 0

    def row(tag, name, dram, hitpct):
        nonlocal i
        sec = dram / 32.0 * 2.0
        w.writerow([str(i), "7", '7 "<default domain>:%s:none"' % tag if tag else "", name, "1000", str(dram), "10",
                    str(sec), str(hitpct) if hit else ""])
        i += 1
    for m in marked:
        j = m["call"]
        for kind, main, pages, hp in (("pr", "_bi_split_kernel", pr_pages, 50.0), ("sh", "_bi_shared_kernel", sh_pages,
                                                                                    12.0)):
            if drop != (kind, j):
                row("skv|%s|c%d" % (kind, j), main, pages * PB, hp)
            row("skv|%s|c%d" % (kind, j), "_bi_combine_kernel", 1000.0, 90.0)
        if drop != ("bd", j):                                               # the union build (torch ops)
            row("skv|bd|c%d" % j, "void at::native::bitonicSortKVInPlace<long>", 100.0, 0.0)
    if untagged:
        row(None, "void at::native::vectorized_elementwise_kernel<4>", 5.0, 0.0)
    (d / "all_raw.csv").write_text(buf.getvalue())


def summarize(d, mode="split"):
    rc = RN.main(["--mode", "summarize", "--out", str(d), "--variants", "all", "--window", "1,2", "--attn-mode", mode])
    s = json.loads((d / "summary.json").read_text())
    return rc, {g["name"]: g for g in s["gates"]}, s


def test_N1_summary_on_a_synthetic_export(tmp_path, capsys):
    """N1: the parser keeps the tagged kernel rows (the units row skipped), the window check passes, the cold-L2
    bounds hold, and the quantities are the exact arithmetic of the synthetic numbers: x_pr = 50 / 100, x_sh = 0.42,
    ratio = 42 / 50, sh_over_union = 42 / 40, f = (50 - 40) / (100 - 40); hit rates from the metric, L2-read weighted;
    the decision D1 (x_pr 0.5 > 0.466, x_sh 0.42 <= 0.466: D2); exit 0, last line R2_SKV_NCU_DONE failed=0."""
    write_case(tmp_path)
    rc, g, s = summarize(tmp_path)
    assert rc == 0
    for name in ("O_cell_bitwise", "O_cell_marked", "O_windows_all", "O_bounds_all"):
        assert g[name]["passed"] and g[name]["must"], g[name]
    q = s["variants"]["all"]["quantities"]
    assert q["sum_bytes"] == 2 * 100 * PB and q["union_bytes"] == 2 * 40 * PB
    assert abs(q["x_pr"] - 0.5) < 1e-12 and abs(q["x_sh"] - 0.42) < 1e-12
    assert abs(q["ratio"] - 0.84) < 1e-12 and abs(q["sh_over_union"] - 42 / 40) < 1e-12
    assert abs(q["f"] - 10 / 60) < 1e-12
    assert abs(q["hit_pr"] - 50.0) < 1e-9 and abs(q["hit_sh"] - 12.0) < 1e-9
    assert g["R_S3_decision"]["detail"].startswith("D2")
    assert "x_pr 0.5000 IN" in g["R_S3_bands_all"]["detail"]
    # device time (every kernel 1000 ns): (build + shared + combine) / (split + combine) = (2 + 4) / 4; kernel only 1
    assert abs(q["time_with_build"] - 1.5) < 1e-12 and abs(q["time_kernel"] - 1.0) < 1e-12
    assert "time_with_build 1.5000 IN" in g["R_S3_device_time_all"]["detail"]
    assert s["variants"]["all"]["kinds"]["bd"]["kernels"] == 2
    assert capsys.readouterr().out.strip().splitlines()[-1] == "R2_SKV_NCU_DONE failed=0"


@pytest.mark.parametrize("case", ["missing_kernel", "missing_build", "untagged", "below_union", "above_sum", "bitwise",
                                  "no_hit"])
def test_N2_summary_negatives(tmp_path, case):
    """N2: each defect fails its MUST gate (exit >= 1), the others stay as in N1: a marked call without its shared
    kernel or without its union build in the window (O_windows_all), an untagged kernel in the export
    (O_windows_all), piece 3 reading less than 0.9 x the
    union from a cold L2 (O_bounds_all: physics), more than 1.15 x the sum (O_bounds_all), a differing row in the cell
    (O_cell_bitwise). no_hit: the L2 hit-rate metric dropped: NOT read as 0, the derived rate is used and named."""
    kw = dict(missing_kernel=dict(drop=("sh", 2)), missing_build=dict(drop=("bd", 1)), untagged=dict(untagged=True),
              below_union=dict(pr_pages=30.0), above_sum=dict(pr_pages=120.0), bitwise=dict(diff_rows=1),
              no_hit=dict(hit=False))[case]
    write_case(tmp_path, **kw)
    rc, g, s = summarize(tmp_path)
    want = dict(missing_kernel="O_windows_all", missing_build="O_windows_all", untagged="O_windows_all",
                below_union="O_bounds_all",
                above_sum="O_bounds_all", bitwise="O_cell_bitwise").get(case)
    if want is None:
        q = s["variants"]["all"]["quantities"]
        assert rc == 0 and q["hit_source"].startswith("derived") and abs(q["hit_pr"] - 50.0) < 1e-9, q
        return
    assert rc >= 1 and not g[want]["passed"], g[want]
    others = [n for n in ("O_cell_bitwise", "O_windows_all", "O_bounds_all") if n != want and n in g]
    assert all(g[n]["passed"] for n in others), [g[n] for n in others]


def test_N3_probe_marks_calls_with_four_rows_in_the_tick(tmp_path):
    """N3: SharedKVProbe(ncu_window=(1, 2)) through the production tick (CPU tiny model, ResidentStore, teacher-
    forced): exactly 2 marked calls, both with rows_max 4 (full_calls counts only those), marked indices 1 and 2,
    sum_pages >= union_pages > 0 on each; the probe stays exact (0 differing rows); on the CPU no profiler call is
    made (prof needs CUDA)."""
    sys.path.insert(0, str(HERE))
    import test_r2_tick as RT
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3_BLOB], capture_output=True)
    assert body.returncode == 0 and body.stdout
    (tmp_path / "p3.py").write_bytes(body.stdout)
    p3 = _load(tmp_path / "p3.py", "piece3_v2_ncu_cpu")
    sk = _load(ROOT / "nosi/nosi/bi_paged_attention_shared.py", "sharedkv_ncu_cpu")
    toks, logits = RT.serial_ref(RT.make_engine(RT.make_ops(p3)))
    probe = sk.SharedKVProbe(p3, mode="split", timing=False, impl="reference", ncu_window=(1, 2))
    ops = SimpleNamespace(linear=RT.exact_linear, attn=probe, mean=RT.BD.fixed_order_mean)
    res = RT.tick_check(RT.make_engine(ops), toks, logits, forced=True)
    assert not res["bad"] and all(res["committed_ok"])
    rep = probe.ncu_report()
    assert rep["full_calls"] >= 3, rep
    assert [m["call"] for m in rep["marked"]] == [1, 2], rep
    assert all(m["sum_pages"] >= m["union_pages"] > 0 for m in rep["marked"]), rep
    assert probe.report()["diff_rows"] == 0


def test_N4_parse_args_refusals():
    """N4: --window needs skip >= 1 (the first calls compile the kernels) and a count; --variants all / none only."""
    for argv in (["--window", "0,4"], ["--window", "3"], ["--variants", "warm"]):
        with pytest.raises(SystemExit):
            RN.parse_args(["--mode", "summarize", "--out", "o"] + argv)
    a = RN.parse_args(["--mode", "cell", "--out", "o", "--batch", "64"])
    assert a.m_fix == 256 and a.window == (32, 64) and a.variants == ["all", "none"]


def test_N5_step_dryrun_refusals_and_pins(tmp_path):
    """N5 (as test_r2_g0bi.py G8): r2_sharedkv_ncu_step.sh: a wrong arm is a usage error (2); a missing uuid file,
    a GPU- prefix and two lines STOP (1); TICK_HOLDERS outside DRYRUN is refused; a wrong --nosi stops at HEAD. On a
    clean checkout of THIS worktree at HEAD the DRYRUN prints both pins (piece 3's blob, the shared module's blob at
    HEAD) and the driver arguments of the arm."""
    script = EFF / "r2_sharedkv_ncu_step.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    uid = "f31fc723-9dfe-407a-9947-28173556232a"

    def run(job, content=None, dry=True, arm="b32-split", sha="0" * 40):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, TICK_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1", arm], env=e, capture_output=True,
                              text=True, timeout=300)
    r = run(310, arm="b16-split")
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(311)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(312, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(313, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(314, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(315, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    nosi = "/mnt/beegfs/ojerbi/nosi-r2-sharedkv4"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(nosi).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % nosi)
    sha = head.stdout.strip()
    skv = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:nosi/nosi/bi_paged_attention_shared.py"],
                         capture_output=True, text=True).stdout.strip()
    for arm, words in (("b32-split", ("--backend resident", "--batch 32", "--m-fix 128", "--attn-mode split")),
                       ("b64-whole", ("--backend b2", "--batch 64", "--m-fix 256", "--attn-mode whole",
                                      "--union torch")),
                       ("b64-split-fused", ("--backend b2", "--batch 64", "--attn-mode split", "--union fused"))):
        r = run(316, uid + "\n", arm=arm, sha=sha)
        assert r.returncode == 0, r.stderr[-400:]
        assert "piece 3 7c826c1bc7235ee9153e3af82c99d85836915416:nosi/nosi/bi_paged_attention.py blob %s" % PIECE3_BLOB \
            in r.stdout
        assert "shared %s" % skv in r.stdout
        for w in words + ("--mode orchestrate", "--ncu /opt/nvidia/nsight-compute/2025.1.0/ncu", "--window 32\\,64",
                          "--variants all\\,none"):
            assert w in r.stdout, w
