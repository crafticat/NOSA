"""CPU tests of the GFW gate's plumbing (benchmarks/Efficiency/r2_gfw_gate.py: the gilfree C++ worker on the A100) and
of its holder step (benchmarks/Efficiency/r2_gfw_step.sh, DRYRUN). No CUDA: the gate's GFW1 / GFW1n / GFW_T run on the
GPU only (GFW1 is tests_cpu/test_r2_transport_gilfree.py GTG1 as a gate). Each check has a negative control.

    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_gfw_gate.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
V2 = dict(sha="f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43", py="45c56f679b017436810bea8e16fa2719d642ba54",
          cpp="df3e3c8026fb0a48cc2fe2597013c3f20ab1338b")


def _gate():
    spec = importlib.util.spec_from_file_location("r2_gfw_gate_cpu", str(EFF / "r2_gfw_gate.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


def _registered(text, prefix="# "):
    lines = text.splitlines()
    i, j = lines.index(prefix + "REGISTERED GFW BEGIN"), lines.index(prefix + "REGISTERED GFW END")
    return "\n".join(ln[len(prefix):] for ln in lines[i + 1:j])


def test_GFWC1_args_plans_gates_and_the_registered_verdict():
    """GFWC1: the gate's defaults (8 threads, 20 timed releases after 3 warm-ups, 64 and 192 MiB, 1500 launches behind
    the spin, a 2e8-cycle GPU sleep) and refusals; a plan of 64 / 192 MiB is 2048 / 6144 groups of 32 KiB, 4 / 12
    distinct destinations per (b, h) stream, layers and blocks in range (negative control: a planted duplicate
    destination is seen; a size that does not spread evenly is refused); the registered verdict's bands (HELD within
    +-15 %, FALSIFIED at >= 1.30, GREY between), the same numbers as the step script's REGISTERED block; GFW_ready /
    GFW1 / GFW1n are MUST and the timing lines REPORT; the gate imports no nosi package (it loads by path)."""
    G = _gate()
    base = ["--out", "o", "--nosi-pkg", "n", "--v3", "a", "--v3-build", "/b", "--v2", "c", "--v2-build", "/d"]
    a = G.parse_args(base)
    assert (a.threads, a.releases, a.warmup, a.sizes, a.launches, a.spin_cycles) == (8, 20, 3, [64, 192], 1500,
                                                                                     200_000_000)
    for bad in (["--sizes", "65"], ["--sizes", "272"], ["--launches", "1000"], ["--releases", "2"], ["--threads", "0"],
                ["--spin-cycles", "10"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    for mib, groups, per in ((64, 2048, 4), (192, 6144, 12)):
        p = G.plan(mib, 7)
        assert len(p[0]) == groups and groups * G.GROUP_BYTES == mib << 20 and G.plan_ok(p), mib
        assert int(p[3].max()) == per - 1 and 0 <= int(p[2].min()) and int(p[2].max()) < G.LAYERS_T
        assert int(p[4].max()) < G.NBLK_T and G.plan(mib, 7)[2].tolist() == p[2].tolist()      # seeded
    bad = G.plan(64, 7)
    bad[3][1] = bad[3][0]                                       # NEGATIVE CONTROL: groups 0 and 1 share (b, h, m)
    assert not G.plan_ok(bad)
    with pytest.raises(ValueError, match="spread evenly"):
        G.plan(8, 1)
    assert [G.verdict(x) for x in (1.0, 0.86, 1.15, 1.2, 1.29, 1.3, 0.5, None)] == [
        "HELD", "HELD", "HELD", "GREY", "GREY", "FALSIFIED", "GREY", "MISSING"]
    block = _registered((EFF / "r2_gfw_step.sh").read_text())
    R = G.REGISTERED
    assert "|ratio - 1| <= %.2f" % R["pred_band"] in block and "ratio >= %.2f" % R["falsifier_ratio"] in block, block
    assert "at 64 and at 192 MiB" in block and a.sizes == [64, 192]
    src = (EFF / "r2_gfw_gate.py").read_text()
    for need in ('gate("GFW_ready", ready, True', "gate(name, ok, True, detail)", 'gate("R_GFW_T%d" % mib',
                 '("GFW1", False, 1 << 24), ("GFW1n", True, 1 << 16)', "set_skip_publish(True)",
                 'r["t4_pending_at_issue"]', 'r["spin_ms"] > 10.0'):
        assert need in src, need
    assert "from nosi" not in src and "import nosi" not in src


def test_GFWC2_step_script_uuid_file_pins_and_dryrun(tmp_path):
    """GFWC2: r2_gfw_step.sh: the uuid file rules (missing, "GPU-" prefix, two lines: STOP before any other check;
    GFW_HOLDERS outside DRYRUN refused); the pins (packer v2's two blobs of f2b981a; packer v3 = the blobs of this
    commit, refused when the .cpp is v2's or not VERSION 3); on a clean checkout at HEAD the DRYRUN prints the uuid,
    both packers' blobs and the gate's arguments. Static: the claim gfw_active, STEP_MIN 12, --export=NONE, no awk /
    python3 / perl outside comments."""
    script = EFF / "r2_gfw_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert 'mkdir "$OUTROOT/gfw_active"' in text and "STEP_MIN=12" in text and "--export=NONE" in text
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index('mkdir "$OUTROOT/gfw_active"')
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-gilfree-worker " in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def run(job, content=None, dry=True, sha="0" * 40):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, GFW_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True,
                              text=True, timeout=120)
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    r = run(511)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(512, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(513, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(514, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(515, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr and "uuid" not in r.stderr.lower()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean git checkout: the full DRYRUN needs one")
    r = run(516, uid + "\n", sha=head.stdout.strip())
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the uuid stage: %s" % r.stderr.strip()[-200:])
    v3 = {}
    for k, p in (("py", "benchmarks/Efficiency/gilfree_pack.py"), ("cpp", "benchmarks/Efficiency/gilfree_pack.cpp")):
        got = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:" + p], capture_output=True, text=True)
        assert got.returncode == 0
        v3[k] = got.stdout.strip()
    assert v3["cpp"] != V2["cpp"]
    assert "gpu_uuid %s (%s/516.gpu_uuid)" % (uid, tmp_path) in r.stdout
    assert ("packer v3 %s:benchmarks/Efficiency/gilfree_pack.py blob %s, benchmarks/Efficiency/gilfree_pack.cpp blob %s"
            % (head.stdout.strip(), v3["py"], v3["cpp"])) in r.stdout, r.stdout[-600:]
    assert ("packer v2 %s:benchmarks/Efficiency/gilfree_pack.py blob %s, benchmarks/Efficiency/gilfree_pack.cpp blob %s"
            % (V2["sha"], V2["py"], V2["cpp"])) in r.stdout
    rec = "/mnt/beegfs/ojerbi/r2-runs/gfw/gfw_t1_516.step"
    for arg in ("--threads 8", "--releases 20", "--warmup 3", "--launches 1500", "--spin-cycles 200000000",
                "--export=NONE", "--gres=gpu:a100:1", "--nosi-pkg /mnt/beegfs/ojerbi/nosi-r2-gilfree-worker/nosi/nosi",
                "--v3 %s/packer_v3" % rec, "--v2 %s/packer_v2" % rec, "r2_gfw_gate.py"):
        assert arg in r.stdout, arg
    assert "--sizes 64,192" in r.stdout or "--sizes 64\\,192" in r.stdout
