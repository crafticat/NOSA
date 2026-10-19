"""CPU tests of the sustained decode + feeder harness (benchmarks/Efficiency/sustained_core.py, sustained_feed.py). No
GPU, no model: the transports run on CPU tensors (cpupack_core.Pipe through sustained_core.EagerBackend; the W8
kernel's copy by sustained_core.gather_reference), the decode is a fake that keeps NOSA's cache-length rules and its
warm-up-sized shapes (it raises where the engine would), and the clock is the host's.

  ST1  configuration: arms, served requests, every refusal of check_config (exit 4 cases), the CIS headroom cap
  ST2  the schedule: determinism, bytes, request rotation, (layer, head) spread, unique destinations, the last-writer map
       against a brute-force replay; a refused frag
  ST3  the re-warm rule against an engine-shaped fake on 960 steps (no stale-shape step); the matched negatives: without
       the width rule, and without the out_len rule, the fake raises at the predicted step
  ST4  the CPU-packing work (live descriptors + Pipe) delivers every last write bit for bit; the content check catches a
       wrong group (exact first_bad) and a stray write into an untouched slot
  ST5  the matched GPU-gather work (reference copy) leaves the SAME scratch bits as the CPU-packing work
  ST6  the feeder: FIFO, backlog, pause only at safe points (a layer never starts while paused), error propagation, drain
  ST7  metrics: delivery / on-time / lateness / table / drain / outstanding, queue_trend, paired_slowdown and its CI,
       /proc stat parsing, cores_used
  ST8  run_arm end to end on the fake decode: alone / cpu / gpu arms PASS (equal trajectories, re-warm steps as
       predicted, every warm-up with the feeder paused or idle, content clean, rows match the schema); negatives: a slow
       feeder -> GROWING queue and late releases; pause disabled -> the capture probe sees a busy feeder; the re-warm rule
       disabled -> the fake engine raises
  ST9  main(): --dry on a synthetic workload writes the plan; a sha mismatch and steps > 1024 exit 4; an existing --out
       exits 2

From the worktree root:
    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/sustained_tests/test_sustained.py::<node>
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

EFF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EFF))
sys.dont_write_bytecode = True

import cpupack_core as K  # noqa: E402
import sustained_core as SC  # noqa: E402
import sustained_feed as SF  # noqa: E402

GEO = SC.Geometry(B=4, S=512, L=4, M=8)            # small source: 8 blocks per request, 4 layers


def _args(tmp=None, extra=()):
    base = ["--model", "m", "--tokens", "t", "--manifest", "x", "--out", str(tmp or "o")]
    return SF.build_parser().parse_args(base + list(extra))


# ------------------------------------------------------------------------------------------------------------ ST1
def test_ST1_configuration_and_refusals():
    arms = SC.parse_arms("alone,cpu:64:32,gpu:0.25:4")
    assert [x.name for x in arms] == ["alone", "cpu_r64_f32", "gpu_r0.25_f4"]
    for bad in ("cpu:64", "foo", "cpu:0:4", "alone,alone"):
        with pytest.raises(ValueError):
            SC.parse_arms(bad)
    assert SC.served_requests(64, 32, 64) == 64 and SC.served_requests(0.25, 4, 4) == 2
    with pytest.raises(ValueError):
        SC.served_requests(65, 32, 64)                  # 65 MiB / 1 MiB = 65 > B
    with pytest.raises(ValueError):
        SC.served_requests(1.1, 32, 64)                 # not a whole number of requests
    cpus = list(range(16))
    cfg = SF.check_config(_args(), 94, cpus)
    assert (cfg["w0"], cfg["w1"], cfg["max_gen"]) == (8, 957, 1024)
    assert cfg["nreq"] == {"cpu_r64_f32": 64, "gpu_r64_f32": 64, "cpu_r192_f96": 64, "gpu_r192_f96": 64}
    assert cfg["main_core"] == 0 and cfg["feeder_cores"] == list(range(1, 9))
    refusals = [["--steps", "1025"], ["--batch", "95"], ["--deadline", "0"], ["--feed-start", "955", "--steps", "960"],
                ["--settle", "949"], ["--max-gen", "100"], ["--arms", "cpu:200:32"], ["--pack-threads", "16"],
                ["--slots", "0"], ["--ring", "1"]]
    for extra in refusals:
        with pytest.raises(SF.SystemExit4):
            SF.check_config(_args(extra=extra), 94, cpus)
    SF.check_config(_args(extra=["--steps", "1024"]), 94, cpus)                      # the cap itself is allowed


# ------------------------------------------------------------------------------------------------------------ ST2
def test_ST2_schedule():
    a, b = SC.Schedule(GEO, 2, 5, seed=3), SC.Schedule(GEO, 2, 5, seed=3)
    for t in range(10, 40):
        ra, rb = a.release(t), b.release(t)
        for f in ("b", "h", "l", "m", "blk"):
            assert torch.equal(getattr(ra, f), getattr(rb, f)), (t, f)
        assert ra.n_groups == 10 and ra.useful == 10 * SC.GROUP_USEFUL
        assert sorted(set(ra.b.tolist())) == sorted({(t * 2 + i) % GEO.B for i in range(2)})
        for bb in set(ra.b.tolist()):
            s = (ra.l * GEO.H + ra.h)[ra.b == bb]
            assert len(set(s.tolist())) == 5                                         # consecutive distinct streams
        trip = list(zip(ra.b.tolist(), ra.h.tolist(), ra.m.tolist()))
        assert len(set(trip)) == len(trip)                                          # unique destinations
        assert int(ra.blk.min()) >= 0 and int(ra.blk.max()) < GEO.nblk
        for l in ra.layers():
            p = ra.plan(l, GEO)
            assert p.dtype == torch.int32 and int((p >= 0).sum()) == int((ra.l == l).sum())
    # last-writer map == a brute-force replay in processing order
    s = SC.Schedule(GEO, 3, 6, seed=1)
    last = {}
    for t in range(50):
        r = s.release(t)
        for l in r.layers():
            sel = (r.l == l).nonzero().view(-1).tolist()
            for i in sel:
                last[(int(r.b[i]), int(r.h[i]), int(r.m[i]))] = (l, int(r.blk[i]))
    for (bb, h, m), (l, blk) in last.items():
        assert (int(s.last_l[bb, h, m]), int(s.last_blk[bb, h, m])) == (l, blk)
    assert int((s.last_l >= 0).sum()) == len(last)
    with pytest.raises(ValueError):
        SC.Schedule(GEO, 1, 2 * GEO.M + 1)                                          # > M groups per (b, h)
    with pytest.raises(ValueError):
        s.release(10)                                                               # out of order


# ------------------------------------------------------------------------------------------------------------ ST3
class ShapeEngine:
    """The engine's shape constraints (module docstring REWARM): a warm-up fixes out_len and the score width; a steady
    step with another out_len or width raises as nosa_llama would (score_buf.copy_ / an unselectable block)."""

    def __init__(self, seq, n, m):
        self.seq, self.n, self.m, self.warm = seq, n, m, None
        self.warmups = []

    def step(self, warmup: bool, t: int):
        mp = self.m + (1 if self.n >= SC.KERNEL else 0)
        out_len, width = -(-(self.seq + 1) // 64), SC.round_up(mp, SC.ROUND)
        if warmup:
            self.warm = (out_len, width)
            self.warmups.append(t)
        elif self.warm != (out_len, width):
            raise RuntimeError("stale shape at step %d: warmed %s, needs %s" % (t, self.warm, (out_len, width)))
        if self.n >= SC.KERNEL:
            self.m, self.n = self.m + 1, SC.STRIDE + 1
        else:
            self.n += 1
        self.seq += 1


def _drive(rule, seq0=16128, n0=32, m0=1007, steps=960):
    eng = ShapeEngine(seq0, n0, m0)
    warm = None
    for t in range(steps):
        need, why = rule(eng.seq, eng.n, eng.m, warm)
        if need:
            warm = SC.warm_state(eng.seq, eng.n, eng.m)
        eng.step(need, t)
    return eng


def test_ST3_rewarm_rule():
    for n0 in (17, 20, 32):
        eng = _drive(SC.rewarm_need, n0=n0)
        assert eng.warmups[0] == 0
        assert all(t in eng.warmups for t in range(64, 960, 64))                     # every new 64-token block
        assert len(eng.warmups) >= 15 + 1
    sim = SC.simulate_cache_lengths(16128, 32, 1007, 960)
    assert sim[0] == (16128, 32, 1007) and sim[1] == (16129, 17, 1008) and sim[16] == (16144, 32, 1008)
    assert sim[17] == (16145, 17, 1009)

    def no_width(seq, n, m, warm):
        need, why = SC.rewarm_need(seq, n, m, warm)
        return (need, why) if why != "width" else (False, "")

    def no_outlen(seq, n, m, warm):
        need, why = SC.rewarm_need(seq, n, m, warm)
        return (need, why) if why != "out_len" else (False, "")
    with pytest.raises(RuntimeError, match=r"^stale shape at step \d+"):
        _drive(no_width)
    with pytest.raises(RuntimeError, match=r"^stale shape at step 64: "):
        _drive(no_outlen)


# ------------------------------------------------------------------------------------------------------------ ST4
def _sources(g=GEO, layers=None):
    sk = [SC.fill_source(torch.empty((g.B, g.S, g.H, g.D), dtype=torch.bfloat16), l, 0) for l in range(layers or g.L)]
    sv = [SC.fill_source(torch.empty((g.B, g.S, g.H, g.D), dtype=torch.bfloat16), l, 1) for l in range(layers or g.L)]
    return sk, sv


def _scratch(g=GEO):
    k = SC.poison(torch.empty((g.B, g.M * g.R, g.H, g.D), dtype=torch.bfloat16))
    return k, SC.poison(torch.empty_like(k))


def _cpu_transport(g, sk, sv, cap=8, ring=2):
    be = SC.EagerBackend()
    sb = K.slot_bytes(cap)
    pipe = K.Pipe(be, torch.zeros((ring, sb), dtype=torch.uint8), torch.zeros((ring, sb), dtype=torch.uint8), cap)
    kk, vv = _scratch(g)
    return be, pipe, kk, vv, SC.cpu_work_fn(K, be, pipe, g, sk, sv, kk, vv)


def _feed(work, releases):
    fd = SC.Feeder(work)
    fd.start()
    for r in releases:
        fd.release(SC.Release(spec=r, deadline_t=r.t + 1))
    fd.stop(drain=True)
    fd.raise_if_failed()
    return fd


def test_ST4_cpu_work_and_content_check():
    v = SC.source_value(1, 70, 1, 5, GEO.S, GEO.H, 2, 1)
    sk, sv = _sources()
    assert int(sv[2].view(torch.int16)[1, 70, 1, 5]) == v
    sched = SC.Schedule(GEO, 3, 6, seed=2)
    rels = [sched.release(t) for t in range(30)]
    be, pipe, kk, vv, work = _cpu_transport(GEO, sk, sv)
    fd = _feed(work, rels)
    assert [r.spec.t for r in fd.done_list] == list(range(30)) and all(r.done is not None for r in fd.done_list)
    ck = SC.check_scratch(kk, sk, sched.last_l, sched.last_blk)
    cv = SC.check_scratch(vv, sv, sched.last_l, sched.last_blk)
    n_written = int((sched.last_l >= 0).sum())
    assert ck["groups_checked"] == cv["groups_checked"] == n_written and ck["groups_bad"] == cv["groups_bad"] == 0
    assert ck["untouched_checked"] == GEO.B * GEO.H * GEO.M - n_written and ck["untouched_bad"] == 0
    # negative 1: one wrong element in the last write of (b, h, m)
    b, h, m = (sched.last_l >= 0).nonzero()[0].tolist()
    kk.view(torch.int16)[b, m * 64 + 3, h, 7] += 1
    bad = SC.check_scratch(kk, sk, sched.last_l, sched.last_blk)
    assert bad["groups_bad"] == 1 and bad["first_bad"][:3] == [b, h, m]
    # negative 2: one release only (most slots untouched), then a stray write into an untouched slot
    one = SC.Schedule(GEO, 1, 2, seed=5)
    rel = one.release(0)
    _, _, k1, v1, w1 = _cpu_transport(GEO, sk, sv)
    _feed(w1, [rel])
    clean = SC.check_scratch(v1, sv, one.last_l, one.last_blk)
    assert clean["groups_bad"] == 0 and clean["untouched_bad"] == 0 and clean["untouched_checked"] == GEO.B * GEO.H * GEO.M - 2
    b2, h2, m2 = (one.last_l == -1).nonzero()[0].tolist()
    v1.view(torch.int16)[b2, m2 * 64, h2, 0] = 0
    assert SC.check_scratch(v1, sv, one.last_l, one.last_blk)["untouched_bad"] == 1


# ------------------------------------------------------------------------------------------------------------ ST5
def test_ST5_gpu_work_matches_cpu_work():
    sk, sv = _sources()
    s1, s2 = SC.Schedule(GEO, 2, 7, seed=4), SC.Schedule(GEO, 2, 7, seed=4)
    r1 = [s1.release(t) for t in range(25)]
    r2 = [s2.release(t) for t in range(25)]
    _, _, ck, cv, w_cpu = _cpu_transport(GEO, sk, sv)
    _feed(w_cpu, r1)
    gk, gv = _scratch()
    be = SC.EagerBackend()
    ids_h = [torch.empty((GEO.H, GEO.B, GEO.M), dtype=torch.int32) for _ in range(2)]
    ids_d = [torch.empty((GEO.H, GEO.B, GEO.M), dtype=torch.int32) for _ in range(2)]
    w_gpu = SC.gpu_work_fn(be, SC.gather_reference, GEO, sk, sv, gk, gv, ids_h, ids_d)
    _feed(w_gpu, r2)
    assert torch.equal(ck.view(torch.int16), gk.view(torch.int16)) and torch.equal(cv.view(torch.int16), gv.view(torch.int16))
    assert SC.check_scratch(gk, sk, s2.last_l, s2.last_blk)["groups_bad"] == 0


# ------------------------------------------------------------------------------------------------------------ ST6
def test_ST6_feeder_fifo_pause_error_drain():
    log = []
    in_layer = threading.Event()
    gate = threading.Event()

    def work(rel, fd):
        for i in range(3):
            fd.safe_point()
            in_layer.set()
            log.append(("layer", rel.spec.t, i, fd.pause_req))
            gate.wait(5)
            time.sleep(0.002)
            in_layer.clear()
        rel.done = SC._HostEv()

    fd = SC.Feeder(work)
    fd.start()
    spec = types.SimpleNamespace
    for t in range(4):
        fd.release(SC.Release(spec=spec(t=t, useful=1, n_groups=1), deadline_t=t + 1))
    while not in_layer.is_set():
        time.sleep(0.001)
    assert fd.backlog() == 4                                                         # 3 queued + 1 in progress
    th = threading.Thread(target=fd.pause)
    th.start()
    time.sleep(0.02)
    assert th.is_alive()                                                             # still inside a layer
    gate.set()
    th.join(5)
    assert not th.is_alive() and fd.paused and not in_layer.is_set()
    n_at_pause = len(log)
    time.sleep(0.05)
    assert len(log) == n_at_pause                                                    # no layer starts while paused
    fd.resume()
    fd.stop(drain=True)
    assert [r.spec.t for r in fd.done_list] == [0, 1, 2, 3]                          # FIFO, drained
    assert all(not x[3] for x in log)                                                # no layer ran with a pause request
    assert len(fd.pauses) == 1 and fd.paused_ns_total > 0

    def boom(rel, fd2):
        fd2.safe_point()
        raise ValueError("injected")
    fe = SC.Feeder(boom)
    fe.start()
    fe.release(SC.Release(spec=spec(t=0), deadline_t=1))
    fe.stop(drain=True)
    with pytest.raises(SC.FeederError, match="^feeder failed: ValueError: injected$"):
        fe.raise_if_failed()


# ------------------------------------------------------------------------------------------------------------ ST7
def test_ST7_metrics():
    ts = [10.0 * t for t in range(21)]                                               # 20 steps of 10 ms
    rel_t = list(range(2, 17))
    useful = [100] * len(rel_t)
    done = [ts[t] + (15.0 if t % 5 else 35.0) for t in rel_t]                        # d = 2 -> late when 35 ms
    gate = [ts[t] for t in rel_t]
    dl = SC.delivery(rel_t, useful, done, gate, ts, 2, 2, 17)
    late = sum(1 for t in rel_t if t % 5 == 0)
    assert dl["on_time_count"] == pytest.approx(1 - late / len(rel_t)) and dl["on_time_bytes"] == dl["on_time_count"]
    assert dl["on_time_table"]["4"]["count"] == 1.0 and dl["on_time_table"]["1"]["count"] == 0.0
    assert dl["lateness_ms"]["max"] == pytest.approx(15.0)
    assert dl["window_ms"] == pytest.approx(150.0) and dl["drain_ms"] == pytest.approx(max(done) - ts[17])
    assert dl["outstanding_gpu"][0] == 1 and max(dl["outstanding_gpu"]) <= 4
    assert dl["delivered_cum_bytes"][-1] == 100 * sum(1 for x in done if x <= ts[17])
    flat = [3.0 + (i % 3) for i in range(60)]
    grow = [0.1 * i for i in range(60)]
    assert SC.queue_trend(flat)["verdict"] == "BOUNDED" and SC.queue_trend(grow)["verdict"] == "GROWING"
    assert SC.queue_trend([1.0] * 5)["verdict"] == "TOO_SHORT"
    alone = {t: 20.0 + (t % 7) * 0.1 for t in range(100)}
    arm = {t: alone[t] * 1.1 for t in range(100)}
    sd = SC.paired_slowdown(alone, arm, range(100), n_boot=200)
    assert sd["slowdown_pct"] == pytest.approx(10.0) and sd["ci_lo"] == pytest.approx(10.0)
    assert sd["median_ratio_pct"] == pytest.approx(10.0)
    rnd = __import__("random").Random(5)
    noisy = {t: alone[t] * (1.1 + rnd.uniform(-0.05, 0.05)) for t in range(100)}
    sd2 = SC.paired_slowdown(alone, noisy, range(100), n_boot=300, seed=1)
    assert sd2["ci_lo"] <= sd2["slowdown_pct"] <= sd2["ci_hi"] and sd2["ci_hi"] - sd2["ci_lo"] > 0
    assert 5.0 < sd2["slowdown_pct"] < 15.0
    assert SC.paired_slowdown(alone, noisy, range(100), n_boot=300, seed=1) == sd2      # deterministic
    line = ("4242 (pt worker) (x) S 1 2 3 4 5 6 7 8 9 10 111 22 " + " ".join(str(i) for i in range(13, 36))
            + " 7 37 38")
    assert SC.parse_stat(line) == ("pt worker) (x", 111, 22, 7)
    a = {1: ("main", 100, 0, 0), 2: ("feeder", 0, 0, 1), 3: ("omp", 0, 0, 2)}
    b = {1: ("main", 200, 0, 0), 2: ("feeder", 50, 50, 1), 3: ("omp", 300, 0, 2), 4: ("new", 100, 0, 3)}
    cu = SC.cores_used(a, b, 1.0, 1, 2, clk=100)
    assert (cu["main"], cu["feeder"], cu["other"], cu["total"]) == (1.0, 1.0, 4.0, 6.0)
    snap = SC.cpu_snapshot()
    assert threading.get_native_id() in snap


# ------------------------------------------------------------------------------------------------------------ ST8
M0 = 1019        # fake compressed length after prefill: width 1024 until step 80 (m' = 1025), out_len at 64 and 128


class FakeLayer:
    def __init__(self, n, m):
        self.no_compress_k_len, self.cached_compressed_max_seqlen = n, m
        self.cache_engine = types.SimpleNamespace(max_gen_len=0)


class FakeCache:
    def __init__(self, n0, m0):
        self.layers = [FakeLayer(n0, m0) for _ in range(2)]
        self.seq = 0

    def get_seq_length(self, layer_idx=0):
        return self.seq


class FakeModel:
    """NOSA's decode interface on CPU: prefill sets the cache length; decode advances it by the engine's rules, keeps
    the warm-up-sized shapes (ShapeEngine's check: a stale step raises) and calls probe() inside every warm-up (where
    the real engine captures CUDA graphs)."""

    def __init__(self, V=32, seq0=16128, step_s=0.001, probe=None):
        self.V, self.seq0, self.step_s, self.probe = V, seq0, step_s, probe
        self.has_buffers = False
        self.warm = None
        self.warmups = []
        self.t = 0

    def batch_prefill(self, ids, cache):
        cache.seq = self.seq0
        self.t = 0
        B = ids.shape[0]
        logits = torch.zeros((B, 1, self.V))
        logits[:, 0, :].scatter_(1, (ids[:, -1:] % self.V), 1.0)
        return logits, torch.arange(self.seq0).repeat(B, 1)

    def decode_inference(self, ids, cu, pos, cache, warmup=False):
        l0 = cache.layers[0]
        n, m = l0.no_compress_k_len, l0.cached_compressed_max_seqlen
        mp = m + (1 if n >= SC.KERNEL else 0)
        shape = (-(-(cache.seq + 1) // 64), SC.round_up(mp, SC.ROUND))
        if not self.has_buffers:
            if self.probe is not None:
                self.probe(self.t)
            self.warm = shape
            self.warmups.append(self.t)
            self.has_buffers = True
        elif shape != self.warm:
            raise RuntimeError("stale shape at step %d" % self.t)
        for ly in cache.layers:
            if ly.no_compress_k_len >= SC.KERNEL:
                ly.cached_compressed_max_seqlen += 1
                ly.no_compress_k_len = SC.STRIDE + 1
            else:
                ly.no_compress_k_len += 1
        cache.seq += 1
        time.sleep(self.step_s)
        self.t += 1
        B = ids.shape[0]
        logits = torch.zeros((B, 1, self.V))
        logits[:, 0, :].scatter_(1, ((ids * 7 + cache.seq) % self.V), 1.0)
        return logits


def _cpu_ctx(model, g, extra_work=None):
    sk, sv = _sources(g)
    holder = {}

    def transport(arm):
        kk, vv = _scratch(g)
        be = SC.EagerBackend()
        if arm.method == "cpu":
            sb = K.slot_bytes(8)
            pipe = K.Pipe(be, torch.zeros((2, sb), dtype=torch.uint8), torch.zeros((2, sb), dtype=torch.uint8), 8)
            w = SC.cpu_work_fn(K, be, pipe, g, sk, sv, kk, vv)
        else:
            ids_h = [torch.empty((g.H, g.B, g.M), dtype=torch.int32) for _ in range(2)]
            ids_d = [torch.empty((g.H, g.B, g.M), dtype=torch.int32) for _ in range(2)]
            w = SC.gpu_work_fn(be, SC.gather_reference, g, sk, sv, kk, vv, ids_h, ids_d)
        holder[arm.name] = (kk, vv)
        work = w if extra_work is None else extra_work(w)

        def finish(sched):
            return dict(k=SC.check_scratch(kk, sk, sched.last_l, sched.last_blk),
                        v=SC.check_scratch(vv, sv, sched.last_l, sched.last_blk))
        return work, (lambda: None), finish
    ids = torch.arange(g.B * 8).view(g.B, 8)
    return SF.ArmContext(model, lambda: FakeCache(32, M0), SC.HostClock(), transport, "cpu", ids, lambda: None), holder


def _cfg(arms, steps=140, extra=()):
    a = _args(extra=["--batch", str(GEO.B), "--steps", str(steps), "--feed-start", "4", "--settle", "4",
                     "--deadline", "3", "--pin", "off", "--arms", arms, "--boot", "200"] + list(extra))
    cfg = SF.check_config(a, 94, list(range(16)))
    cfg["geom"] = GEO
    return a, cfg


def test_ST8_run_arm_end_to_end_and_negatives(monkeypatch):
    feeders = []
    violations = []

    class RecFeeder(SC.Feeder):
        def __init__(self, *x, **kw):
            super().__init__(*x, **kw)
            feeders.append(self)

    monkeypatch.setattr(SC, "Feeder", RecFeeder)

    def probe(t):
        fd = feeders[-1] if feeders and feeders[-1].thread is not None and feeders[-1].thread.is_alive() else None
        if fd is not None and fd.busy and not fd.paused:
            violations.append(t)
    a, cfg = _cfg("alone,cpu:0.25:4,gpu:0.25:4")
    model = FakeModel(step_s=0.003, probe=probe)            # a decode step well above a release's CPU work
    ctx, _ = _cpu_ctx(model, GEO)
    arms = []
    for arm in cfg["arms"]:
        feeders.clear()
        arms.append(SF.run_arm(ctx, arm, cfg, a))
    res = SF.finalize(arms, a, cfg)
    assert res == dict(verdict="PASS", failures=[]), res
    pred = []
    warm = None
    for t, (seq, n, m) in enumerate(SC.simulate_cache_lengths(16128, 32, M0, 140)):
        need, why = SC.rewarm_need(seq, n, m, warm)
        if need:
            warm = SC.warm_state(seq, n, m)
            pred.append(dict(t=t, why=why))
    whys = {x["why"] for x in pred}
    assert {"first", "out_len", "width"} <= whys
    for r in arms:
        assert r["rewarm_steps"] == pred and r["trajectory_equal"]
        for row in r["steps_rows"]:
            SC.jsonl_line(SC.STEP_FIELDS, row)
    assert violations == []
    for r in arms[1:]:
        assert r["content"]["k"]["groups_bad"] == 0 and r["content"]["v"]["groups_bad"] == 0
        assert len(r["release_rows"]) == cfg["w1"] - cfg["w0"]
        for row in r["release_rows"]:
            SC.jsonl_line(SC.RELEASE_FIELDS, row)
        assert r["queue_outstanding_gpu"]["verdict"] == "BOUNDED", r["queue_outstanding_gpu"]
        assert r["pauses"]["count"] == len(pred) and "slowdown" in r and r["slowdown"]["n_steps"] > 0
        assert r["delivery"]["releases"] == cfg["w1"] - cfg["w0"]

    # NEGATIVE 1: a feeder slower than the decode -> GROWING queue, late releases
    def slow(w):
        def work(rel, fd):
            time.sleep(0.004)
            w(rel, fd)
        return work
    a2, cfg2 = _cfg("cpu:0.25:4", steps=100)
    ctx2, _ = _cpu_ctx(FakeModel(step_s=0.001), GEO, extra_work=slow)
    r2 = SF.run_arm(ctx2, cfg2["arms"][0], cfg2, a2)
    assert r2["queue_outstanding_gpu"]["verdict"] == "GROWING" and r2["delivery"]["on_time_count"] < 0.5

    # NEGATIVE 2: no pause around the warm-ups -> the capture probe sees a busy feeder
    def busy(w):
        def work(rel, fd):
            fd.safe_point()
            time.sleep(0.003)
            w(rel, fd)
        return work
    violations.clear()
    monkeypatch.setattr(RecFeeder, "pause", lambda self, timeout_s=600.0: 0)
    a3, cfg3 = _cfg("cpu:0.25:4", steps=100)

    def probe_wait(t):
        fd = feeders[-1] if feeders else None
        if fd is None or t == 0:
            return
        t0 = time.time()
        while time.time() - t0 < 2.0:                     # the backlog keeps the feeder busy: a violation shows at once
            if fd.busy and not fd.paused:
                violations.append(t)
                return
            time.sleep(0.0002)
    ctx3, _ = _cpu_ctx(FakeModel(step_s=0.0002, probe=probe_wait), GEO, extra_work=busy)
    SF.run_arm(ctx3, cfg3["arms"][0], cfg3, a3)
    assert violations and violations[0] == 64
    monkeypatch.undo()

    # NEGATIVE 3: the re-warm rule disabled (first warm-up only) -> the engine-shaped fake raises
    monkeypatch.setattr(SC, "rewarm_need", lambda seq, n, m, warm: (warm is None, "first" if warm is None else ""))
    a4, cfg4 = _cfg("alone", steps=100)
    ctx4, _ = _cpu_ctx(FakeModel(), GEO)
    with pytest.raises(RuntimeError, match=r"^stale shape at step \d+$"):
        SF.run_arm(ctx4, cfg4["arms"][0], cfg4, a4)


# ------------------------------------------------------------------------------------------------------------ ST9
def _workload(tmp: Path, rows=3, corrupt=False):
    data = bytearray()
    docs = []
    for i in range(rows):
        row = torch.arange(i, i + SF.CTX, dtype=torch.int32).numpy().astype("<i4").tobytes()
        data += row
        docs.append(dict(index=i * 2, n_tokens=20000, sha256=hashlib.sha256(row).hexdigest()))
    if corrupt:
        docs[1]["sha256"] = "0" * 64
    (tmp / "tok.bin").write_bytes(bytes(data))
    (tmp / "man.json").write_text(json.dumps(dict(ctx=SF.CTX, dtype="int32-le", shape=[rows, SF.CTX], docs=docs)))
    return tmp / "tok.bin", tmp / "man.json"


def test_ST9_main_dry_and_exits(tmp_path):
    tok, man = _workload(tmp_path)
    base = ["--model", "m", "--tokens", str(tok), "--manifest", str(man), "--batch", "2", "--pin", "off",
            "--arms", "alone,cpu:2:32", "--steps", "200"]
    out = tmp_path / "o1"
    assert SF.main(base + ["--out", str(out), "--dry"]) == 0
    rec = json.loads((out / "summary.json").read_text())
    assert rec["status"] == "dry" and rec["derived"]["nreq"] == {"cpu_r2_f32": 2}
    assert [d["row"] for d in rec["workload"]["rows"]] == [0, 1] and (out / "SHA256SUMS").exists()
    assert SF.main(base + ["--out", str(out), "--dry"]) == 2                          # --out exists
    assert SF.main(base + ["--out", str(tmp_path / "o2"), "--dry", "--steps", "1025"]) == 4
    assert not (tmp_path / "o2").exists()
    bad = tmp_path / "bad"
    bad.mkdir()
    tb, mb = _workload(bad, corrupt=True)
    args = ["--model", "m", "--tokens", str(tb), "--manifest", str(mb), "--batch", "2", "--pin", "off",
            "--arms", "alone", "--steps", "200", "--out", str(tmp_path / "o3"), "--dry"]
    assert SF.main(args) == 4 and not (tmp_path / "o3").exists()
    ids = SF.load_workload(tok, man, 1, 2)["ids"]
    assert ids.shape == (2, SF.CTX) and int(ids[0, 0]) == 1 and int(ids[1, -1]) == 2 + SF.CTX - 1
