"""CPU tests of the GFW gate's plumbing (benchmarks/Efficiency/r2_gfw_gate.py: the gilfree C++ worker on the A100) and
of its holder step (benchmarks/Efficiency/r2_gfw_step.sh, DRYRUN). No CUDA: the gate's GFW1 / GFW1n / GFW_T run on the
GPU only (GFW1 is tests_cpu/test_r2_transport_gilfree.py GTG1 as a gate). Each check has a negative control.

    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_gfw_gate.py
"""
from __future__ import annotations

import builtins
import dis
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


def _registered(text, prefix="# ", tag="GFW"):
    lines = text.splitlines()
    i, j = lines.index(prefix + "REGISTERED %s BEGIN" % tag), lines.index(prefix + "REGISTERED %s END" % tag)
    return "\n".join(ln[len(prefix):] for ln in lines[i + 1:j])


def _takes_ok(desc, a):
    """gfw2's takes: (6, n), per layer distinct destinations (b, h, slot), every field in range, distinct gens."""
    rows = desc.t().tolist()
    ns, nblk = a.l2_c + a.l2_x - 1, a.l2_s_host // 64
    dst = [(l, b, h, s) for (l, b, h, _, s, _) in rows]
    return (len(set(dst)) == len(dst) and len({r[5] for r in rows}) == len(rows)
            and all(0 <= l < a.l2_layers and 0 <= b < a.l2_batch and 0 <= h < a.l2_heads and 0 <= k < nblk
                    and 0 <= s < ns for (l, b, h, k, s, _) in rows))


def test_GFWC3_the_gfw2_cell_args_takes_and_its_registered_verdict():
    """GFWC3 (cell gfw2): its defaults (6496 takes over 4 layers of the B 64 geometry: KV heads 2, host rows 16311,
    C 96 + X 32) and refusals; the takes are (6, 6496) with 1624 per layer, distinct destinations per layer, slots
    below C + X - 1 (room for the planted shift), blocks below the 254 complete host blocks, distinct generations, and
    seeded (negative control: a planted duplicate destination is seen); the verdict's bands (HELD <= 1.0 ms,
    FALSIFIED >= 5.0 ms), the same numbers as the step script's REGISTERED GFW2 block; GFW2 / GFW2n MUST and
    R_GFW2_host REPORT; cell gfw1 keeps its own gates."""
    G = _gate()
    base = ["--out", "o", "--nosi-pkg", "n", "--v3", "a", "--v3-build", "/b", "--v2", "c", "--v2-build", "/d"]
    assert G.parse_args(base).cell == "gfw1"
    a = G.parse_args(base + ["--cell", "gfw2"])
    assert (a.takes, a.l2_layers, a.l2_batch, a.l2_heads, a.l2_s_host, a.l2_c, a.l2_x, a.l2_sleep_cycles) == (
        6496, 4, 64, 2, 16311, 96, 32, 50_000_000)
    for bad in (["--cell", "gfw3"], ["--takes", "6497"], ["--takes", "200000"], ["--l2-s-host", "100"],
                ["--l2-sleep-cycles", "10"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + ["--cell", "gfw2"] + bad)
    d = G.l2_takes(a)
    assert tuple(d.shape) == (6, 6496) and d.dtype == G.l2_takes(a).dtype and d.is_contiguous()
    assert all(int((d[0] == l).sum()) == 1624 for l in range(4)) and _takes_ok(d, a)
    assert int(d[4].max()) < 127 and int(d[3].max()) < 254 and int(d[5].min()) == 2
    assert G.l2_takes(a).tolist() == d.tolist()                                       # seeded
    bad = d.clone()
    bad[1:3, 1] = bad[1:3, 0]
    bad[4, 1] = bad[4, 0]                                                             # NEGATIVE CONTROL: take 1 = take 0's
    assert not _takes_ok(bad, a)
    assert [G.verdict2(x) for x in (0.2, 1.0, 1.5, 4.99, 5.0, None)] == ["HELD", "HELD", "GREY", "GREY", "FALSIFIED",
                                                                         "MISSING"]
    block = _registered((EFF / "r2_gfw_step.sh").read_text(), tag="GFW2")
    R = G.REGISTERED2
    assert "PREDICTION <= %.1f ms" % R["desc_issue_pred_ms"] in block and \
        "FALSIFIER >= %.1f ms" % R["desc_issue_fals_ms"] in block and "6496 takes" in block, block
    src = (EFF / "r2_gfw_gate.py").read_text()
    for need in ('gates.append(gate("GFW2", ok2, True', 'gates.append(gate("GFW2n", not r["planted_equal"], True',
                 'gates.append(gate("R_GFW2_host", v == "HELD", False', 'if ready and a.cell == "gfw1":',
                 'if ready and a.cell == "gfw2":', "set_plant_slot_shift(1)", 'r["stores_equal"] and r["boards_equal"]'):
        assert need in src, need


def test_GFWC1_args_plans_gates_and_the_registered_verdict():
    """GFWC1: the gate's defaults (8 threads, 20 timed releases after 3 warm-ups, 64 and 192 MiB, 1500 launches behind
    the spin, a 2e8-cycle GPU sleep) and refusals; a plan of 64 / 192 MiB is 2048 / 6144 groups of 32 KiB, 4 / 12
    distinct destinations per (b, h) stream, layers and blocks in range (negative control: a planted duplicate
    destination is seen; a size that does not spread evenly is refused); the registered verdict's bands (HELD within
    +-15 %, FALSIFIED at >= 1.30, GREY between), the same numbers as the step script's REGISTERED block; GFW_ready /
    GFW1 (captured: 64 launches behind the spin, no submission wait) / GFW1e (eager: 1500 launches, the host's
    submission wait first, >= 50 ms) / GFW1n are MUST, R_GFW1x (the eager hazard: 1500 launches, no wait, a 2^20
    bound; expected err 1) and the timing lines REPORT; the gate imports no nosi package (it loads by path)."""
    G = _gate()
    base = ["--out", "o", "--nosi-pkg", "n", "--v3", "a", "--v3-build", "/b", "--v2", "c", "--v2-build", "/d"]
    a = G.parse_args(base)
    assert (a.threads, a.releases, a.warmup, a.sizes, a.launches, a.spin_cycles) == (8, 20, 3, [64, 192], 1500,
                                                                                     200_000_000)
    assert (a.launches_few, a.hazard_spin) == (64, 1 << 20)
    for bad in (["--sizes", "65"], ["--sizes", "272"], ["--launches", "1000"], ["--releases", "2"], ["--threads", "0"],
                ["--spin-cycles", "10"], ["--launches-few", "300"], ["--hazard-spin", "10"]):
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
    for need in ('gate("GFW_ready", ready, True', "gates.append(gate(name, ok, must, detail))", 'gate("R_GFW_T%d" % mib',
                 '("GFW1", True, False, 1 << 24, a.launches_few, False)',
                 '("GFW1e", True, False, 1 << 24, a.launches, True)',
                 '("GFW1n", True, True, 1 << 16, a.launches_few, False)',
                 '("R_GFW1x", False, False, a.hazard_spin, a.launches, False)', "gf.wait_submitted(trs[0])",
                 '(r["submission_wait_ms"] or 0.0) >= 50.0', "set_skip_publish(True)",
                 'r["t4_pending_at_issue"]', 'r["spin_ms"] > 10.0'):
        assert need in src, need
    assert "from nosi" not in src and "import nosi" not in src


def test_GFWC2_step_script_uuid_file_pins_and_dryrun(tmp_path):
    """GFWC2: r2_gfw_step.sh: the uuid file rules (missing, "GPU-" prefix, two lines: STOP before any other check;
    GFW_HOLDERS outside DRYRUN refused); the pins (packer v2's two blobs of f2b981a; packer v3 = the blobs of this
    commit, refused when the .cpp is v2's or not VERSION 3); an unknown cell is refused (usage); on a clean checkout at
    HEAD the DRYRUN prints the uuid, both packers' blobs and the gate's arguments of the default cell gfw1 (unchanged)
    and of the cell gfw2. Static: the claim gfw_active, STEP_MIN 12, --export=NONE, no awk / python3 / perl outside
    comments."""
    script = EFF / "r2_gfw_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert 'mkdir "$OUTROOT/gfw_active"' in text and "STEP_MIN=12" in text and "--export=NONE" in text
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index('mkdir "$OUTROOT/gfw_active"')
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-gfw1d " in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def run(job, content=None, dry=True, sha="0" * 40, cell=None):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, GFW_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"] + ([cell] if cell else []), env=e,
                              capture_output=True, text=True, timeout=120)
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
    r = run(515, uid + "\n", cell="gfw3")
    assert r.returncode == 2 and "usage" in r.stderr                    # an unknown cell is refused first
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
                "--launches-few 64", "--hazard-spin 1048576",
                "--export=NONE", "--gres=gpu:a100:1", "--nosi-pkg /mnt/beegfs/ojerbi/nosi-r2-gfw1d/nosi/nosi",
                "--v3 %s/packer_v3" % rec, "--v2 %s/packer_v2" % rec, "r2_gfw_gate.py", "--cell gfw1", "; cell gfw1"):
        assert arg in r.stdout, arg
    assert "--sizes 64,192" in r.stdout or "--sizes 64\\,192" in r.stdout
    assert "--takes" not in r.stdout                                     # gfw1 is the default cell, unchanged
    r2 = run(516, sha=head.stdout.strip(), cell="gfw2")
    assert r2.returncode == 0, r2.stderr[-300:]
    for arg in ("--cell gfw2", "--takes 6496", "--l2-layers 4", "--l2-batch 64", "--l2-heads 2", "--l2-s-host 16311",
                "--l2-c 96", "--l2-x 32", "--l2-sleep-cycles 50000000", "--threads 8", "--export=NONE", "; cell gfw2"):
        assert arg in r2.stdout, arg
    assert "--releases" not in r2.stdout and "--launches" not in r2.stdout
    r3 = run(516, sha=head.stdout.strip(), cell="gfw1d")
    assert r3.returncode == 0, r3.stderr[-300:]
    for arg in ("--cell gfw1d", "--env CUDA_DEVICE_MAX_CONNECTIONS=32", "--launches-few 64", "--spin-cycles 200000000",
                "--threads 8", "--export=NONE", "; cell gfw1d"):
        assert arg in r3.stdout, arg
    assert "--releases" not in r3.stdout and "--takes" not in r3.stdout
    # the variable is set in gfw1d's child only (NEGATIVE CONTROL: the other cells' commands do not carry it)
    assert r3.stdout.index("--env CUDA_DEVICE_MAX_CONNECTIONS=32") < r3.stdout.index("nosa-env.sif")
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in r.stdout and "CUDA_DEVICE_MAX_CONNECTIONS" not in r2.stdout


def _global_loads(code):
    out = [ins.argval for ins in dis.get_instructions(code) if ins.opname == "LOAD_GLOBAL"]
    for c in code.co_consts:
        if hasattr(c, "co_code"):
            out += _global_loads(c)
    return out


def _name_problems(jit):
    """F23's property: the kernel's code has no free variable and every global it loads resolves in its module or in
    builtins (Triton's compiler resolves a kernel's names in fn.__globals__)."""
    fn = getattr(jit, "fn", jit)
    code = fn.__code__
    return list(code.co_freevars) + [n for n in _global_loads(code)
                                     if n not in fn.__globals__ and not hasattr(builtins, n)]


def test_GFWC5_the_gfw1d_cell_is_report_only_and_registered():
    """GFWC5 (cell gfw1d, the shared-hardware-queue hypothesis after gfw1c): --cell gfw1d parses; its only line besides
    GFW_ready is R_GFW1d, REPORT (must False), GFW1's captured protocol (64 launches, no submission wait), and it
    requires the variable CUDA_DEVICE_MAX_CONNECTIONS = 32 in the process (else INVALID) and records it in the summary;
    the gate's REGISTERED1D and the step script's REGISTERED GFW1D block name the same variable and value, the
    prediction (err 0 if the shared queue is the cause) and the falsifier (err 1 at the bound with the variable set)."""
    G = _gate()
    base = ["--out", "o", "--nosi-pkg", "n", "--v3", "a", "--v3-build", "/b", "--v2", "c", "--v2-build", "/d"]
    assert G.parse_args(base + ["--cell", "gfw1d"]).cell == "gfw1d" and "gfw1d" in G.CELLS
    src = (EFF / "r2_gfw_gate.py").read_text()
    for need in ('if ready and a.cell == "gfw1d":', 'gates.append(gate("R_GFW1d", ok, False, detail))',
                 "gfw1(GF, SCG, ext3, False, 1 << 24, a.spin_cycles, a.launches_few, a.threads, wait_submission=False)",
                 'env == REGISTERED1D["value"] and r["err"] == 0', 'cuda_device_max_connections=os.environ.get('):
        assert need in src, need
    assert G.REGISTERED1D == dict(env="CUDA_DEVICE_MAX_CONNECTIONS", value="32")
    block = _registered((EFF / "r2_gfw_step.sh").read_text(), tag="GFW1D")
    for frag in ("CUDA_DEVICE_MAX_CONNECTIONS=32", "PREDICTION if the shared hardware queue is the cause: err 0",
                 "FALSIFIER of that hypothesis: err 1 at the spin's bound"):
        assert frag in block, frag


def test_GFWC4_the_gate_loads_the_fixed_exact_wait_kernel():
    """GFWC4 (gfw1 on holder 2190876: GFW1 / GFW1n failed with "CompilationError: at 1:69" = fg5's "NameError: tl is
    not defined": the branch predated C's fix 30b404b). Through the GATE's own loader (load_nosi of this tree's nosi
    package, as GFW1 loads it), r2_residency_sc._wait_kernel() returns the MODULE-LEVEL board_wait_kernel, a Triton
    JITFunction with no free variable whose every global resolves in its module or in builtins (C's F23 property; F23
    itself runs in this tree: test_r2_residency.py); GFW1 launches exactly that call; 30b404b is an ancestor of HEAD.
    Negative control: a kernel built in a factory with a local-only `import triton.language` fails the same check."""
    triton = pytest.importorskip("triton")
    from triton.runtime.jit import JITFunction
    G = _gate()
    _, SCG = G.load_nosi(str(ROOT / "nosi/nosi"), pkg="r2gfw_c4")
    k = SCG._wait_kernel()
    assert isinstance(k, JITFunction) and k is SCG.board_wait_kernel, type(k)
    assert _name_problems(k) == [], _name_problems(k)
    assert "kern = SCG._wait_kernel()" in (EFF / "r2_gfw_gate.py").read_text()
    anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "30b404b", "HEAD"])
    assert anc.returncode == 0

    def factory():
        import triton.language as tl_local

        @triton.jit
        def planted(x_ptr, BLOCK: tl_local.constexpr):
            offs = tl_local.arange(0, BLOCK)
            tl_local.store(x_ptr + offs, offs)
        return planted
    assert _name_problems(factory()) == ["tl_local"]           # NEGATIVE CONTROL
