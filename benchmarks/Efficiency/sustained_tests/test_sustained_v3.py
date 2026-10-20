"""CPU tests of the v3 additions to the sustained harness (feature/nosi-sustained-v3): arm options (cpu mode L / LP, any
arm's GPU sleep), the per-step decode-thread CPU time and decode_ms, the per-sleep scoring, the mode-aware content and
staging checks, the host / NUMA record parsers, and the step script's holder-uuid guard. Every plumbing test has a
negative control that must FAIL. Reuses the fake decode and helpers of test_sustained.py.

  V1  parse_arms options: names, moves_data, the v2 names unchanged, every refusal
  V2  check_config: the registered v3 arm list at --steps 512; a sleep arm without its alone:sleep arm is refused
  V3  cpu_work_fn modes on the eager backend: full writes the scratch and packs the staging; LP packs, scratch untouched;
      L neither. NEGATIVES: full work checked as L (stray writes, packed staging); L work checked as LP (not packed)
  V4  parse_maps / parse_numa_maps / bytes_by_node (pro rata, uncounted driver mappings) / parse_node_meminfo; smoke of
      numa_record, node_memfree, host_topology on this host
  V5  finalize on synthetic records: each arm against the alone arm of the same sleep, decode_ms for sleep arms, the
      host off-CPU summary. NEGATIVES: a sleep missing from the period; a sleep arm without its reference
  V6  run_arm end to end (fake decode, host clock, host sleep) for the six registered arm kinds: PASS, schema, host_cpu_ms
      <= host_iter_ms, decode_ms <= period, the sleep gap. NEGATIVES: mode ignored (L runs full) -> content + staging FAIL;
      LP runs L -> staging FAIL; sleep not applied -> FAIL; no sleep in the context -> refused
  V7  sustained_step.sh: missing / malformed holder uuid file -> STOP before anything; a valid one passes on to the
      HEAD check; no uuid is hard-coded

From the worktree root:
    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/sustained_tests/test_sustained_v3.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
EFF = HERE.parent
sys.path.insert(0, str(EFF))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import cpupack_core as K  # noqa: E402
import sustained_core as SC  # noqa: E402
import sustained_feed as SF  # noqa: E402
import test_sustained as T  # noqa: E402  (helpers only: fake decode, sources, scratch, config)

GEO = T.GEO
REGISTERED = "alone,cpu:64:32,cpu:64:32:mode=L,cpu:64:32:mode=LP,alone:sleep=30,cpu:64:32:sleep=30"
V2_ARMS = "alone,cpu:64:32,gpu:64:32,cpu:192:96,gpu:192:96"


# ------------------------------------------------------------------------------------------------------------- V1
def test_V1_arm_options():
    arms = SC.parse_arms(REGISTERED)
    assert [x.name for x in arms] == ["alone", "cpu_r64_f32", "cpu_r64_f32_mL", "cpu_r64_f32_mLP", "alone_s30",
                                      "cpu_r64_f32_s30"]
    assert [x.moves_data for x in arms] == [False, True, False, False, False, True]
    assert (arms[1].mode, arms[1].sleep_ms, arms[2].mode, arms[4].sleep_ms) == ("full", 0.0, "L", 30.0)
    assert [x.name for x in SC.parse_arms(V2_ARMS)] == ["alone", "cpu_r64_f32", "gpu_r64_f32", "cpu_r192_f96",
                                                        "gpu_r192_f96"]
    assert SC.parse_arms("cpu:64:32:sleep=30:mode=LP")[0].name == "cpu_r64_f32_mLP_s30"
    assert SC.parse_arms("gpu:64:32:sleep=2.5")[0].name == "gpu_r64_f32_s2.5"
    for bad in ("gpu:64:32:mode=L", "alone:mode=L", "cpu:64:32:mode=X", "cpu:64:32:sleep=-1", "cpu:64:32:sleep=1001",
                "cpu:64:32:sleep=nan", "cpu:64:32:sleep=abc", "cpu:64:32:foo=1", "cpu:64:32:mode=L:mode=LP",
                "cpu:mode=L:64:32", "alone:5", "alone:sleep=30,alone:sleep=30", "cpu:64:32:mode=full,cpu:64:32"):
        with pytest.raises(ValueError):
            SC.parse_arms(bad)


# ------------------------------------------------------------------------------------------------------------- V2
def test_V2_check_config_registered_and_refusals():
    cpus = list(range(16))
    cfg = SF.check_config(T._args(extra=["--steps", "512", "--arms", REGISTERED]), 94, cpus)
    assert (cfg["w0"], cfg["w1"], cfg["max_gen"]) == (8, 509, 576)
    assert cfg["nreq"] == {n: 64 for n in ("cpu_r64_f32", "cpu_r64_f32_mL", "cpu_r64_f32_mLP", "cpu_r64_f32_s30")}
    with pytest.raises(SF.SystemExit4, match="alone:sleep=30"):
        SF.check_config(T._args(extra=["--arms", "alone,cpu:64:32:sleep=30"]), 94, cpus)
    with pytest.raises(SF.SystemExit4, match="mode applies to cpu arms only"):
        SF.check_config(T._args(extra=["--arms", "alone,gpu:64:32:mode=LP"]), 94, cpus)
    SF.check_config(T._args(extra=["--arms", "alone:sleep=30,cpu:64:32:sleep=30"]), 94, cpus)   # no sleep-0 alone needed


# ------------------------------------------------------------------------------------------------------------- V3
def _mode_transport(g, sk, sv, mode, cap=8, ring=2):
    be = SC.EagerBackend()
    sb = K.slot_bytes(cap)
    stage = torch.zeros((ring, sb), dtype=torch.uint8)
    pipe = K.Pipe(be, stage, torch.zeros_like(stage), cap)
    SC.poison_stage(stage, cap, g.R)
    kk, vv = T._scratch(g)
    return pipe, kk, vv, SC.cpu_work_fn(K, be, pipe, g, sk, sv, kk, vv, mode=mode)


def _run_mode(sk, sv, mode, n=12):
    sched = SC.Schedule(GEO, 3, 6, seed=2)
    rels = [sched.release(t) for t in range(n)]
    pipe, kk, vv, w = _mode_transport(GEO, sk, sv, mode)
    fd = T._feed(w, rels)
    assert all(r.done is not None for r in fd.done_list) and len(fd.done_list) == n
    return sched, pipe, kk, vv, w


def test_V3_modes_scratch_and_staging():
    sk, sv = T._sources()
    st = SC.poison_stage(torch.zeros((2, K.slot_bytes(4)), dtype=torch.uint8), 4)
    p0 = K.idx_region(4)
    assert int(st[:, :p0].sum()) == 0 and bool((st[:, p0:].contiguous().view(torch.int16) == SC.POISON_I16).all())
    with pytest.raises(ValueError):
        SC.cpu_work_fn(K, SC.EagerBackend(), None, GEO, sk, sv, None, None, mode="X")
    res = {}
    for mode in ("full", "LP", "L"):
        sched, pipe, kk, vv, w = _run_mode(sk, sv, mode)
        c = SC.content_check(kk, vv, sk, sv, sched, mode == "full")
        s = SC.staged_check(K, pipe, w.last, sk, sv)
        assert c["ok"], (mode, c)
        assert SC.stage_ok(mode, s), (mode, s)
        res[mode] = (c, s)
    assert res["full"][0]["expect"] == "last writes" and res["full"][0]["k"]["groups_checked"] > 0
    assert res["LP"][0]["expect"] == "untouched" and res["LP"][0]["k"]["untouched_checked"] == GEO.B * GEO.H * GEO.M
    assert res["full"][1]["packed"] and res["LP"][1]["packed"] and not res["LP"][1]["poisoned"]
    assert res["L"][1]["poisoned"] and not res["L"][1]["packed"]
    assert not SC.stage_ok("full", dict(chunks_seen=False, packed=False, poisoned=False))
    # NEGATIVE 1: a broken mode plumbing (the L arm runs the full work): the L checks see stray writes and packing
    sched, pipe, kk, vv, w = _run_mode(sk, sv, "full")
    c = SC.content_check(kk, vv, sk, sv, sched, False)
    assert not c["ok"] and c["k"]["untouched_bad"] > 0 and c["v"]["untouched_bad"] > 0
    assert not SC.stage_ok("L", SC.staged_check(K, pipe, w.last, sk, sv))
    # NEGATIVE 2: an LP arm that does not pack (runs L): its staging is still the poison
    sched, pipe, kk, vv, w = _run_mode(sk, sv, "L")
    assert not SC.stage_ok("LP", SC.staged_check(K, pipe, w.last, sk, sv))


# ------------------------------------------------------------------------------------------------------------- V4
def test_V4_numa_parsers_and_record():
    maps_txt = ("7f0000000000-7f0000400000 rw-s 00000000 00:05 123 /dev/nvidiactl\n"
                "7f1000000000-7f1000200000 rw-p 00000000 00:00 0\n"
                "garbage\n")
    numa_txt = ("7f0000000000 default file=/dev/nvidiactl\n"
                "7f1000000000 bind:1 anon=512 dirty=512 N0=256 N1=256 kernelpagesize_kB=4\n"
                "zz\n")
    maps = SC.parse_maps(maps_txt)
    assert maps == [(0x7f0000000000, 0x7f0000400000), (0x7f1000000000, 0x7f1000200000)]
    numa = SC.parse_numa_maps(numa_txt)
    assert numa[0x7f1000000000]["nodes"] == {0: 256, 1: 256} and numa[0x7f1000000000]["kpage_kB"] == 4
    assert numa[0x7f0000000000]["nodes"] == {} and numa[0x7f1000000000]["policy"] == "bind:1" and len(numa) == 2
    half = SC.bytes_by_node([(0x7f1000000000, 0x7f1000100000)], maps, numa)                # half of the anon mapping
    assert half["bytes_by_node"] == {"0": 256 * 4096 * 0.5, "1": 256 * 4096 * 0.5}
    assert half["covered_bytes"] == 0x100000 and half["uncounted_bytes"] == 0 and half["n_mappings"] == 1
    drv = SC.bytes_by_node([(0x7f0000000000, 0x7f0000400000)], maps, numa)                  # a driver mapping
    assert drv["bytes_by_node"] == {} and drv["uncounted_bytes"] == drv["covered_bytes"] == 0x400000
    assert SC.bytes_by_node([(1, 2)], maps, numa)["covered_bytes"] == 0
    mi = ("Node 0 MemTotal:       527622520 kB\nNode 0 MemFree:        9471888 kB\n"
          "Node 0 HugePages_Total:     0\nnonsense line\n")
    got = SC.parse_node_meminfo(mi)
    assert got["MemTotal"] == 527622520 and got["MemFree"] == 9471888 and got["HugePages_Total"] == 0
    rec = SC.numa_record([torch.zeros(1 << 18), torch.zeros(1 << 16)])
    assert ("error" in rec) or (rec["range_bytes"] == 4 * ((1 << 18) + (1 << 16)) and rec["covered_bytes"] >= 0)
    assert isinstance(SC.node_memfree(), dict) and "nodes" in SC.host_topology()


# ------------------------------------------------------------------------------------------------------------- V5
def _rec(name, method, sleep, per, dec, cpu_frac=0.5):
    return dict(arm=name, method=method, sleep_ms=sleep, mode="full", periods=per, decode_ms=dec, host_iter_ms=list(per),
                host_cpu_ms=[p * cpu_frac for p in per], tokens_sha256="s", rewarm_steps=[dict(t=0, why="first")])


def test_V5_finalize_scoring_per_sleep():
    a, cfg = T._cfg("alone,cpu:0.25:4,alone:sleep=5,cpu:0.25:4:sleep=5", steps=60)
    N = 60
    recs = [_rec("alone", "alone", 0.0, [10.0] * N, [10.0] * N), _rec("cpu_r0.25_f4", "cpu", 0.0, [16.0] * N, [16.0] * N),
            _rec("alone_s5", "alone", 5.0, [15.0] * N, [10.0] * N), _rec("cpu_r0.25_f4_s5", "cpu", 5.0, [15.5] * N, [10.5] * N)]
    res = SF.finalize(recs, a, cfg)
    assert res == dict(verdict="PASS", failures=[]), res
    s0, s5 = recs[1]["slowdown"], recs[3]["slowdown"]
    assert (s0["metric"], s0["reference"]) == ("period", "alone") and s0["slowdown_pct"] == pytest.approx(60.0)
    assert (s5["metric"], s5["reference"]) == ("decode_ms", "alone_s5") and s5["slowdown_pct"] == pytest.approx(5.0)
    assert recs[3]["slowdown_period"]["slowdown_pct"] == pytest.approx((15.5 / 15.0 - 1) * 100)
    assert "slowdown" not in recs[0] and "slowdown" not in recs[2] and recs[2]["sleep_gap_ms"] == pytest.approx(5.0)
    assert recs[1]["host"]["off_cpu_mean_ms"] == pytest.approx(8.0) and recs[1]["host"]["cpu_mean_ms"] == pytest.approx(8.0)
    assert recs[3]["decode"]["decode_mean_ms"] == pytest.approx(10.5)
    # NEGATIVE 1: the sleep is not in the period (decode_ms == period) -> FAIL naming the arm
    bad = [_rec("alone", "alone", 0.0, [10.0] * N, [10.0] * N), _rec("alone_s5", "alone", 5.0, [15.0] * N, [10.0] * N),
           _rec("cpu_r0.25_f4_s5", "cpu", 5.0, [10.5] * N, [10.5] * N)]
    r1 = SF.finalize(bad, a, cfg)
    assert r1["verdict"] == "FAIL" and any(f.startswith("cpu_r0.25_f4_s5: the 5 ms sleep is not in the period")
                                           for f in r1["failures"]), r1
    # NEGATIVE 2: a sleep arm without the alone arm of its sleep
    r2 = SF.finalize([_rec("alone", "alone", 0.0, [10.0] * N, [10.0] * N),
                      _rec("cpu_r0.25_f4_s5", "cpu", 5.0, [15.5] * N, [10.5] * N)], a, cfg)
    assert r2["verdict"] == "FAIL" and any("no alone arm with sleep 5" in f for f in r2["failures"]), r2


# ------------------------------------------------------------------------------------------------------------- V6
def _ctx_v3(model, g, sleep="host", force_mode=None):
    sk, sv = T._sources(g)

    def transport(arm):
        kk, vv = T._scratch(g)
        be = SC.EagerBackend()
        pipe = None
        if arm.method == "cpu":
            sb = K.slot_bytes(8)
            stage = torch.zeros((2, sb), dtype=torch.uint8)
            pipe = K.Pipe(be, stage, torch.zeros_like(stage), 8)
            SC.poison_stage(stage, 8, g.R)
            w = SC.cpu_work_fn(K, be, pipe, g, sk, sv, kk, vv, mode=force_mode or arm.mode)
        else:
            ids_h = [torch.empty((g.H, g.B, g.M), dtype=torch.int32) for _ in range(2)]
            ids_d = [torch.empty((g.H, g.B, g.M), dtype=torch.int32) for _ in range(2)]
            w = SC.gpu_work_fn(be, SC.gather_reference, g, sk, sv, kk, vv, ids_h, ids_d)

        def finish(sched):
            out = SC.content_check(kk, vv, sk, sv, sched, arm.moves_data, g.R)
            if arm.method == "cpu":
                out["stage"] = SC.staged_check(K, pipe, w.last, sk, sv, g.R)
            return out
        return w, (lambda: None), finish
    sl = {"host": (lambda ms: time.sleep(ms / 1000.0)), "noop": (lambda ms: None), None: None}[sleep]
    ids = torch.arange(g.B * 8).view(g.B, 8)
    return SF.ArmContext(model, lambda: T.FakeCache(32, T.M0), SC.HostClock(), transport, "cpu", ids, lambda: None,
                         sleep=sl)


def _run(arms_txt, steps, **kw):
    a, cfg = T._cfg(arms_txt, steps=steps)
    ctx = _ctx_v3(T.FakeModel(step_s=0.003), GEO, **kw)
    recs = [SF.run_arm(ctx, arm, cfg, a) for arm in cfg["arms"]]
    return recs, SF.finalize(recs, a, cfg)


def test_V6_run_arm_v3_arms_and_negatives():
    arms_txt = "alone,cpu:0.25:4,cpu:0.25:4:mode=L,cpu:0.25:4:mode=LP,alone:sleep=4,cpu:0.25:4:sleep=4"
    recs, res = _run(arms_txt, 100)
    assert res == dict(verdict="PASS", failures=[]), res
    by = {r["arm"]: r for r in recs}
    assert set(by) == {"alone", "cpu_r0.25_f4", "cpu_r0.25_f4_mL", "cpu_r0.25_f4_mLP", "alone_s4", "cpu_r0.25_f4_s4"}
    for r in recs:
        assert r["trajectory_equal"] and len(r["steps_rows"]) == 100
        for row in r["steps_rows"]:
            SC.jsonl_line(SC.STEP_FIELDS, row)
            assert 0.0 <= row["host_cpu_ms"] <= row["host_iter_ms"] + 0.5, row
            assert 0.0 <= row["decode_ms"] <= row["period_ms"] + 1e-6, row
    assert by["cpu_r0.25_f4_mL"]["content"]["expect"] == "untouched" and by["cpu_r0.25_f4_mL"]["content"]["stage"]["poisoned"]
    assert by["cpu_r0.25_f4_mLP"]["content"]["stage"]["packed"] and by["cpu_r0.25_f4"]["content"]["stage"]["packed"]
    assert by["alone_s4"]["sleep_gap_ms"] >= 3.5 and by["cpu_r0.25_f4_s4"]["sleep_gap_ms"] >= 3.5
    s = by["cpu_r0.25_f4_s4"]["slowdown"]
    assert (s["metric"], s["reference"]) == ("decode_ms", "alone_s4") and s["n_steps"] > 0
    assert by["cpu_r0.25_f4_mL"]["slowdown"]["reference"] == "alone"
    # the host sleep is off-CPU time of the decode thread: the off-CPU summary must see it
    assert by["alone_s4"]["host"]["off_cpu_mean_ms"] - by["alone"]["host"]["off_cpu_mean_ms"] >= 3.0
    # NEGATIVE 1: mode ignored (the L arm runs the full work) -> content and staging FAIL for that arm
    _, r1 = _run("alone,cpu:0.25:4:mode=L", 60, force_mode="full")
    assert r1["verdict"] == "FAIL", r1
    assert any(f.startswith("cpu_r0.25_f4_mL content k") for f in r1["failures"]), r1
    assert any(f.startswith("cpu_r0.25_f4_mL staging") for f in r1["failures"]), r1
    # NEGATIVE 2: the LP arm does not pack (runs L) -> staging FAIL, content still clean
    _, r2 = _run("alone,cpu:0.25:4:mode=LP", 60, force_mode="L")
    assert r2["verdict"] == "FAIL" and [f.split(" (")[0] for f in r2["failures"]] == ["cpu_r0.25_f4_mLP staging"], r2
    # NEGATIVE 3: the sleep is not applied -> FAIL for both sleep arms
    _, r3 = _run("alone,alone:sleep=4,cpu:0.25:4:sleep=4", 60, sleep="noop")
    assert r3["verdict"] == "FAIL" and sum("sleep is not in the period" in f for f in r3["failures"]) == 2, r3
    # NEGATIVE 4: no sleep in the context -> a sleep arm is refused before it runs
    a4, cfg4 = T._cfg("alone:sleep=4", steps=60)
    with pytest.raises(ValueError, match="asks for a 4 ms sleep"):
        SF.run_arm(_ctx_v3(T.FakeModel(), GEO, sleep=None), cfg4["arms"][0], cfg4, a4)


# ------------------------------------------------------------------------------------------------------------- V7
STEP = EFF / "sustained_step.sh"


def _step(tmp, job, holders):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "SUSTAINED_HOLDERS_DIR": str(holders)}
    return subprocess.run(["bash", str(STEP), "0" * 40, str(job), "v3test_tag"], env=env, capture_output=True, text=True,
                          timeout=60)


def test_V7_step_script_reads_the_holder_uuid(tmp_path):
    h = tmp_path / "holders"
    h.mkdir()
    rec = EFF.parent.parent / "results" / "sustained" / "v3test_tag.step"
    r = _step(tmp_path, 4242, h)
    assert r.returncode == 1 and "gpu_uuid file" in r.stderr and "is missing" in r.stderr, r.stderr
    (h / "4242.gpu_uuid").write_text("not-a-uuid\n")
    r = _step(tmp_path, 4242, h)
    assert r.returncode == 1 and "malformed gpu uuid" in r.stderr, r.stderr
    (h / "4242.gpu_uuid").write_text("f31fc723-9dfe-407a-9947-28173556232a\n")
    r = _step(tmp_path, 4242, h)
    assert r.returncode == 1 and "HEAD is not" in r.stderr, r.stderr         # the uuid passed; the next guard stops it
    assert not rec.exists()                                                   # nothing was created
    assert "f31fc723" not in STEP.read_text()                                 # no uuid pinned in the script
