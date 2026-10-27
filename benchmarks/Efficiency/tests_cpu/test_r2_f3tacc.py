"""[f3tacc] CPU tests of the cell evalacc216f3t of r2_tick_sg5_step.sh: the accuracy / exactness proof (acc216:
--modes fast,bi_c_strict, --store-committed-tokens --serial-rows 8 --gold-workload GOLD) on the evalfast2f3t flag set
(--b2-resolve fused --fused-kernel v3 --res-lean --res-dev --worker cpp --b2-issue desc --no-transfer-mirror, no
--inflight-array; packer v3 of --nosi, GF3). No source change: the cell only wires flags that test_r2_f3ld.py (TO*)
and test_r2_f3ldx.py (XD*) prove on this path, and the gates of test_r2_acc216.py (AC*).

  TA1  the cell (static): its case line, its block, the flag set equal to evalfast2f3t's F3FLAGS, the link to the
       same job's evalfast2f3t, no timing arguments; the control evalacc216f3lia keeps its arguments; the REGISTERED
       ACC3T block names the ACC3LIA gates and is not stamped by the builder (the lead stamps it).
  TA2  the DRYRUN argv on a clean checkout of this ROOT parses with the f3t flags and the acc216 flags.
  TA3  outside DRYRUN the cell stops at the SG5_HOLDERS rule, before any launch.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3tacc.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # the shared name (f3lia, f3ld use it too)
HF = RF.HF                                                            # the step helpers (_dry, _clean_head, STEP)
TACC_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3tacc"
CELL = "evalacc216f3t"
F3T_FLAGS = "--worker cpp --b2-issue desc --no-transfer-mirror"


def _block(text, head):
    i = text.index(head)
    return text[i:text.index(" ;;\n", i) + 3]


def test_TA1_the_cell_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + TACC_ROOT
    assert TACC_ROOT == HF.HF_ROOT
    assert "|evalfast2f3c|evalfast2c3c|evalacc216f3t] [gilfree|gather]" in text
    assert "        evalacc216f3t) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    ac = _block(text, "    evalacc216f3t) MODES=fast,bi_c_strict")
    for w in ("BACKENDS=b2", "PC=96 ; PX=1512 ; PRV=1464", "BATCH=216", "MEMG=240G", "VARIANTS=fused/batched",
              "NOSI=$NOSI_EVAL", "ACC=1", "GF3=1",
              'LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2f3t/summary.json")',
              "--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped",
              "--b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads",
              "--res-lean --res-dev " + F3T_FLAGS + " --store-committed-tokens",
              '--serial-rows 8 --gold-workload "$GOLD" "${LINKARG[@]}") ;;'):
        assert w in ac, w
    for w in ("--inflight-array", "--eval-role", "--accuracy-record", "--mem-cap-gb", "--eval-bands", "--bare",
              "STEP_MIN=", "STEPS="):
        assert w not in ac, w
    # the same flag set as the timing cell evalfast2f3t (test_r2_f3ld.py TO5)
    assert "F3FLAGS=(%s)" % F3T_FLAGS in text
    # the control evalacc216f3lia keeps its arguments (test_r2_f3lia.py F3G)
    ctl = _block(text, "    evalacc216f3lia) MODES=fast,bi_c_strict")
    assert "--res-lean --res-dev --inflight-array --store-committed-tokens" in ctl and "_evalfast2f3lia/" in ctl
    assert "--b2-issue" not in ctl and "GF3=1" not in ctl
    # the registration: the ACC3LIA gates by name; the stamp is the lead's
    reg = text[text.index("# REGISTERED for evalacc216f3t"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("ACC3T: same gates as ACC3LIA", "ACC3T-BI", "ACC3T-ETD", "ACC3T-AE", "ACC3T-LINK", "evalfast2f3t",
                "nosi-r2-f3tacc", "3769b64"):
        assert tag in reg, tag
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state
        assert "NOT STAMPED" in reg
    for tag in ("ACC3LIA-BI", "ACC3LIA-ETD", "ACC3LIA-AE", "ACC3LIA-LINK", "ACC3LIA-MEM"):
        assert tag in text, tag                                       # the gates ACC3T reuses are registered
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_TA2_the_dryrun_argv(tmp_path, capsys):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, CELL)
    if r.returncode != 0:
        assert Path(TACC_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the f3tacc ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    with capsys.disabled():
        print("TA2 DRYRUN %s: %s" % (CELL, got))
    for s in ("--backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2",
              "--repeat-docs --log-slots -1 --skip-shipped", "--variants fused/batched", "--mem=240G", "--time=120",
              "--b2-resolve fused --fused-kernel v3", "--res-lean --res-dev " + F3T_FLAGS,
              "PYTHONPATH=%s/nosi" % TACC_ROOT, "--job-name=sg5-t1-418-%s " % CELL, "/r2-runs/sg5/sg5_t1_418_%s " % CELL):
        assert s in got or s in line, s
    assert "--inflight-array" not in got and "torch/per_call" not in got
    assert "[f3ld] packer copied: %s (packer v3 of --nosi, DESC 2)" % HF._clean_head() in r.stdout
    assert "[acc216] modes fast,bi_c_strict" in r.stdout
    pa = HF._sg5().parse_args(got.split())
    assert (pa.batch, pa.X_pool, pa.b2_resolve, pa.fused_kernel, pa.layout) == (216, 1512, "fused", "v3", "pool")
    assert pa.b2_issue == "desc" and pa.worker == "cpp" and pa.no_transfer_mirror and not pa.inflight_array
    assert pa.res_lean and pa.res_dev and pa.arrival_lag == 2
    assert pa.modes == ["fast", "bi_c_strict"] and pa.store_committed_tokens and pa.serial_rows == 8
    assert pa.eval_role == "" and not pa.bare and pa.mem_cap_gb == 0 and not pa.accuracy_record and pa.steps == 48
    assert pa.host_profile and pa.gc_freeze and pa.lean_time and pa.wait_index and pa.b2_fewer_reads
    assert pa.gold_workload == "/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280"
    assert pa.link_record == "/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_418_evalfast2f3t/summary.json"


def test_TA3_the_step_stops_outside_dryrun(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", CELL], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 1 and "r2_tick_sg5.py" not in r.stdout, (r.returncode, r.stderr[-300:])
    assert "SG5_HOLDERS is for DRYRUN" in r.stderr, r.stderr[-300:]
