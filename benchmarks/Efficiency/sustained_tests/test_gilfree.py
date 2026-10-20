"""CPU tests of the GIL-free packer (benchmarks/Efficiency/gilfree_pack.cpp + gilfree_pack.py) and its cpu_native arm.
The extension is built JIT ONCE per session under $TORCH_EXTENSIONS_DIR (else a pytest tmp dir; never under HOME); the
build needs the image's C++ compiler, ninja and the CUDA headers (nosa-env.sif), not a GPU.

  GF1  bytes: for 24 random releases (random sizes, layers beyond the source count, random blocks and slots) the native
       packing equals the Python packer's bytes BIT FOR BIT (destination rows, K rows, V rows: the concatenation of
       cpupack_core.build_desc + index_select per layer). NEGATIVE: a one-block change of the reference is detected.
       Refusals: duplicate destination, out-of-range block, wrong output size
  GF2  the NativePipe (CPU rings) driven by the feeder leaves the SAME scratch bits as the Python Pipe (cpu_work_fn) over
       40 releases; chunk counts; the last staging slot holds the last chunk's [indices | K | V]. NEGATIVE: sources bound
       in the wrong layer order -> the content check fails
  GF3  the GIL: while pack_into runs, a Python thread keeps running (largest stall << call); NEGATIVE CONTROL: the same
       work through pack_into_holding_gil stalls that thread for the whole call. A single large NativePipe.release also
       lets the thread run
  GF4  harness plumbing: parse_arms / check_config for cpu_native (TORCH_EXTENSIONS_DIR required, refused under HOME),
       the build-dir guard, run_arm end to end (fake decode) for alone / cpu / cpu_native: PASS and equal content.
       NEGATIVE: a cpu_native transport bound to the wrong sources -> FAIL

From the worktree root (inside nosa-env.sif):
    TORCH_EXTENSIONS_DIR=<run dir>/torch_ext python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/sustained_tests/test_gilfree.py
"""
from __future__ import annotations

import os
import sys
import threading
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
import gilfree_pack as GF  # noqa: E402
import sustained_core as SC  # noqa: E402
import sustained_feed as SF  # noqa: E402
import test_sustained as T  # noqa: E402  (helpers only)

GEO2 = SC.Geometry(B=6, S=768, L=8, M=12)          # 12 blocks per request, 144 destination slots, plans over 8 layers


@pytest.fixture(scope="module")
def ext(tmp_path_factory):
    base = os.environ.get("TORCH_EXTENSIONS_DIR") or str(tmp_path_factory.mktemp("torch_ext"))
    t0 = time.perf_counter()
    e = GF.load(os.path.join(GF.check_build_dir(base), GF.EXT_NAME), verbose=True)
    print("gilfree_pack built / loaded in %.1f s from %s" % (time.perf_counter() - t0, getattr(e, "__file__", "?")))
    return e


def _spec(b, h, l, m, blk):
    t = [torch.as_tensor(x, dtype=torch.int64).reshape(-1).contiguous() for x in (b, h, l, m, blk)]
    return SC.ReleaseSpec(t=0, b=t[0], h=t[1], l=t[2], m=t[3], blk=t[4], n_req=0)


def _random_spec(rng, g, n, n_layers):
    slots = torch.randperm(g.B * g.H * g.M, generator=rng)[:n]
    return _spec(slots // (g.H * g.M), (slots // g.M) % g.H, torch.randint(0, n_layers, (n,), generator=rng), slots % g.M,
                 torch.randint(0, g.nblk, (n,), generator=rng))


def _eq(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


# ------------------------------------------------------------------------------------------------------------ GF1
def test_GF1_native_bytes_equal_python_bytes(ext):
    sk, sv = T._sources(GEO2, layers=3)                                    # plan layers 0..7 map to sources l % 3
    rng = torch.Generator().manual_seed(7)
    sizes = [1, 2, 5, 17, 64, 100, 144] + [int(x) for x in torch.randint(1, 145, (17,), generator=rng)]
    for n in sizes:
        spec = _random_spec(rng, GEO2, n, 8)
        ri, rk, rv = GF.reference_stream(K, spec, GEO2, sk, sv)
        ni, nk, nv, st = GF.native_pack(ext, spec, GEO2, sk, sv)
        assert list(st[:2]) == [n, n * GEO2.R]
        assert torch.equal(ni, ri), n
        assert _eq(nk, rk) and _eq(nv, rv), n
    # NEGATIVE: the comparison sees a one-block change of the reference
    spec = _random_spec(rng, GEO2, 40, 8)
    bad = _spec(spec.b, spec.h, spec.l, spec.m, (spec.blk + 1) % GEO2.nblk)
    _, rk2, _ = GF.reference_stream(K, bad, GEO2, sk, sv)
    _, nk2, _, _ = GF.native_pack(ext, spec, GEO2, sk, sv)
    assert not _eq(nk2, rk2)
    with pytest.raises(RuntimeError, match="duplicate destination"):
        GF.native_pack(ext, _spec([0, 0], [1, 1], [0, 3], [2, 2], [0, 1]), GEO2, sk, sv)
    with pytest.raises(RuntimeError, match="out of range"):
        GF.native_pack(ext, _spec([0], [0], [0], [0], [GEO2.nblk]), GEO2, sk, sv)
    with pytest.raises(RuntimeError, match="out_k / out_v"):
        ext.pack_into(spec.b, spec.h, spec.l, spec.m, spec.blk, sk, sv, GEO2.R, GEO2.M * GEO2.R,
                      torch.empty(40 * GEO2.R, dtype=torch.int64), torch.empty((1, GEO2.D), dtype=torch.bfloat16),
                      torch.empty((1, GEO2.D), dtype=torch.bfloat16), 1)


# ------------------------------------------------------------------------------------------------------------ GF2
def _native_cpu(ext, g, sk, sv, cap, ring):
    sb = K.slot_bytes(cap)
    stage = torch.zeros((ring, sb), dtype=torch.uint8)
    land = torch.zeros_like(stage)
    npipe = ext.NativePipe(stage, land, cap, g.R)
    npipe.set_sources(sk, sv)
    return npipe, stage


def test_GF2_native_pipe_matches_python_pipe(ext):
    g = T.GEO
    sk, sv = T._sources(g)
    s1, s2 = SC.Schedule(g, 3, 6, seed=11), SC.Schedule(g, 3, 6, seed=11)
    r1 = [s1.release(t) for t in range(40)]
    r2 = [s2.release(t) for t in range(40)]
    _, _, kk_py, vv_py, w_py = T._cpu_transport(g, sk, sv, cap=8, ring=2)
    T._feed(w_py, r1)
    cap, ring = 4, 3
    npipe, stage = _native_cpu(ext, g, sk, sv, cap, ring)
    assert npipe.cuda is False and npipe.p0 == K.idx_region(cap)
    kk, vv = T._scratch(g)
    w = GF.native_work_fn(npipe, SC.EagerBackend(), g, kk, vv)
    fd = T._feed(w, r2)
    assert _eq(kk, kk_py) and _eq(vv, vv_py)                               # the SAME scratch bits as the Python Pipe
    assert SC.content_check(kk, vv, sk, sv, s2, True)["ok"]
    exp_chunks = sum(-(-r.n_groups // cap) for r in r2)
    assert w.stats["chunks"] == exp_chunks == npipe.chunks_issued and w.stats["releases"] == 40
    assert w.stats["groups"] == sum(r.n_groups for r in r2)
    assert all(r.done is not None and r.layers == len(r.spec.layers()) for r in fd.done_list)
    spec = r2[-1]
    ri, rk, rv = GF.reference_stream(K, spec, g, sk, sv)
    n_last = spec.n_groups - cap * ((spec.n_groups - 1) // cap)
    rows = n_last * g.R
    kst, vst, ist, lo, hi = K.stage_views(stage[(npipe.chunks_issued - 1) % ring], n_last, "row", "row", cap)
    assert torch.equal(ist, ri[-rows:]) and _eq(kst, rk[-rows:]) and _eq(vst, rv[-rows:])
    # NEGATIVE: sources bound in the wrong layer order -> the content check fails
    np2, _ = _native_cpu(ext, g, sk[::-1], sv[::-1], cap, ring)
    k2, v2 = T._scratch(g)
    s3 = SC.Schedule(g, 3, 6, seed=11)
    T._feed(GF.native_work_fn(np2, SC.EagerBackend(), g, k2, v2), [s3.release(t) for t in range(40)])
    assert not SC.content_check(k2, v2, sk, sv, s3, True)["ok"]


# ------------------------------------------------------------------------------------------------------------ GF3
def _stall_during(fn):
    """(call seconds, the longest pause of a Python spinner thread while fn runs)."""
    stop = threading.Event()
    out = {}

    def spin():
        last = time.perf_counter()
        gap = 0.0
        while not stop.is_set():
            now = time.perf_counter()
            if now - last > gap:
                gap = now - last
            last = now
        out["gap"] = gap
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    stop.set()
    th.join(10)
    return dt, out["gap"]


def test_GF3_native_calls_release_the_gil(ext):
    sk, sv = T._sources(GEO2, layers=3)
    spec = _random_spec(torch.Generator().manual_seed(3), GEO2, 144, 8)
    old = torch.get_num_threads()
    torch.set_num_threads(1)                                                # the packer on this thread only
    try:
        st = GF.native_pack(ext, spec, GEO2, sk, sv, repeat=20)[3]
        per_s = max(int(st[2]), 1) / 20 / 1e9
        rep = int(min(200000, max(20, 0.4 / per_s)))
        dt_rel, gap_rel = _stall_during(lambda: GF.native_pack(ext, spec, GEO2, sk, sv, repeat=rep))
        dt_hold, gap_hold = _stall_during(lambda: GF.native_pack(ext, spec, GEO2, sk, sv, repeat=rep, hold_gil=True))
        g3 = SC.Geometry(B=16, S=4096, L=2, M=128)                          # one release of 4096 groups (134 MB)
        sk3, sv3 = T._sources(g3)
        sched = SC.Schedule(g3, 16, 256, seed=1)
        big = sched.release(0)
        npipe, _ = _native_cpu(ext, g3, sk3, sv3, 256, 3)
        kk, vv = T._scratch(g3)
        dt_pipe, gap_pipe = _stall_during(lambda: npipe.release(big.b, big.h, big.l, big.m, big.blk,
                                                                kk.view(-1, g3.D), vv.view(-1, g3.D)))
    finally:
        torch.set_num_threads(old)
    assert dt_rel >= 0.15 and dt_hold >= 0.15, (dt_rel, dt_hold)
    assert gap_rel <= 0.3 * dt_rel, ("the spinner stalled during a GIL-released call", dt_rel, gap_rel)
    assert gap_hold >= 0.8 * dt_hold, ("NEGATIVE CONTROL: the GIL-holding call must stall the spinner", dt_hold, gap_hold)
    assert dt_pipe >= 0.03 and gap_pipe <= 0.5 * dt_pipe, ("NativePipe.release", dt_pipe, gap_pipe)
    assert SC.content_check(kk, vv, sk3, sv3, sched, True)["ok"]


# ------------------------------------------------------------------------------------------------------------ GF4
def test_GF4_build_dir_guard_and_config(monkeypatch, tmp_path):
    home = str(tmp_path / "home")
    with pytest.raises(ValueError, match="absolute"):
        GF.check_build_dir("relative/dir", home=home)
    with pytest.raises(ValueError, match="under HOME"):
        GF.check_build_dir(home + "/x/torch_ext", home=home)
    assert GF.check_build_dir(str(tmp_path / "run" / "torch_ext"), home=home).endswith("torch_ext")
    with pytest.raises(ValueError, match="TORCH_EXTENSIONS_DIR is not set"):
        GF.build_dir_from_env({}, home=home)
    assert GF.build_dir_from_env({"TORCH_EXTENSIONS_DIR": str(tmp_path / "r")}, home=home).endswith(GF.EXT_NAME)
    arms = SC.parse_arms("alone,cpu_native:64:32,cpu_native:192:96:sleep=30,alone:sleep=30")
    assert [x.name for x in arms] == ["alone", "cpu_native_r64_f32", "cpu_native_r192_f96_s30", "alone_s30"]
    assert arms[1].moves_data and arms[1].mode == "full"
    with pytest.raises(ValueError, match="mode applies to cpu arms only"):
        SC.parse_arms("cpu_native:64:32:mode=L")
    cpus = list(range(16))
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    with pytest.raises(SF.SystemExit4, match="TORCH_EXTENSIONS_DIR is not set"):
        SF.check_config(T._args(extra=["--arms", "alone,cpu_native:64:32"]), 94, cpus)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", os.path.join(os.path.expanduser("~"), "torch_ext_test"))
    with pytest.raises(SF.SystemExit4, match="under HOME"):
        SF.check_config(T._args(extra=["--arms", "alone,cpu_native:64:32"]), 94, cpus)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "run" / "torch_ext"))
    cfg = SF.check_config(T._args(extra=["--steps", "512", "--arms", "alone,cpu:64:32,cpu_native:64:32,cpu_native:192:96"]),
                          94, cpus)
    assert cfg["nreq"] == {"cpu_r64_f32": 64, "cpu_native_r64_f32": 64, "cpu_native_r192_f96": 64}


def _ctx_native(ext, model, g, reverse=False):
    sk, sv = T._sources(g)

    def transport(arm):
        kk, vv = T._scratch(g)
        be = SC.EagerBackend()
        if arm.method == "cpu_native":
            npipe, _ = _native_cpu(ext, g, (sk[::-1] if reverse else sk), (sv[::-1] if reverse else sv), 8, 3)
            w = GF.native_work_fn(npipe, be, g, kk, vv)
        else:
            sb = K.slot_bytes(8)
            stage = torch.zeros((2, sb), dtype=torch.uint8)
            pipe = K.Pipe(be, stage, torch.zeros_like(stage), 8)
            w = SC.cpu_work_fn(K, be, pipe, g, sk, sv, kk, vv)

        def finish(sched):
            out = SC.content_check(kk, vv, sk, sv, sched, arm.moves_data, g.R)
            if arm.method == "cpu_native":
                out["native"] = dict(w.stats)
            return out
        return w, (lambda: None), finish
    ids = torch.arange(g.B * 8).view(g.B, 8)
    return SF.ArmContext(model, lambda: T.FakeCache(32, T.M0), SC.HostClock(), transport, "cpu", ids, lambda: None,
                         sleep=lambda ms: time.sleep(ms / 1000.0))


def test_GF4_run_arm_cpu_native_end_to_end(ext, monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch_ext"))
    a, cfg = T._cfg("alone,cpu:0.25:4,cpu_native:0.25:4", steps=100)
    ctx = _ctx_native(ext, T.FakeModel(step_s=0.003), T.GEO)
    recs = [SF.run_arm(ctx, arm, cfg, a) for arm in cfg["arms"]]
    res = SF.finalize(recs, a, cfg)
    assert res == dict(verdict="PASS", failures=[]), res
    by = {r["arm"]: r for r in recs}
    nat = by["cpu_native_r0.25_f4"]
    assert nat["content"]["ok"] and nat["content"]["native"]["releases"] == cfg["w1"] - cfg["w0"]
    assert nat["content"]["native"]["chunks"] >= nat["content"]["native"]["releases"]
    assert len(nat["release_rows"]) == cfg["w1"] - cfg["w0"] and nat["slowdown"]["reference"] == "alone"
    for row in nat["release_rows"]:
        SC.jsonl_line(SC.RELEASE_FIELDS, row)
        assert row["layers"] >= 1
    # NEGATIVE: the native transport bound to the wrong sources -> content FAIL for that arm only
    a2, cfg2 = T._cfg("alone,cpu_native:0.25:4", steps=60)
    ctx2 = _ctx_native(ext, T.FakeModel(step_s=0.003), T.GEO, reverse=True)
    r2 = SF.finalize([SF.run_arm(ctx2, arm, cfg2, a2) for arm in cfg2["arms"]], a2, cfg2)
    assert r2["verdict"] == "FAIL" and all(f.startswith("cpu_native_r0.25_f4 content") for f in r2["failures"]), r2
