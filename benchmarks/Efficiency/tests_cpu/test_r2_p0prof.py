"""CPU tests of Task 0.1 "p0prof" (benchmarks/Efficiency/r2_base_profile_cell.py, r2_base_profile_step.sh). No CUDA,
no Triton launch, no job. The engine modules are replaced by fakes with the same names (nosi/ imports CUDA builds).

PP1  parse_args: the defaults (no timers, no nsys); the opt-in flag; refusals: --nsys-window with --stage-timers, an
     nsys / profiler / selection window over a re-warm, a selection window over the profiler window, too few rows,
     --mbase at S 130816, an unknown --context, --pool 66, --ring 1, --steps 1001. S 130816 has the re-warm steps of
     S 16128.
PP2  expected_sequence: the lengths (pool / no pool / selection step), 'final' ends the last layer, the probe labels sit
     between stage1 and pool_topk; GROUPS hold every label of a steady step exactly once and no probe label; SAVES and
     the step labels name known labels only.
PP3  StageRecorder with a fake clock: rings 2, 3 and 4 give the same steps; periods, stages, closure 0, per-layer sums,
     the set grows past its cap; a re-warm step keeps its period only; capturing blocks the marks; the chain per kind.
     NEGATIVE CONTROL: a step that skips one boundary is counted bad, with its index and labels.
PP4  Patcher on fake modules (the shipped call order; the names read at call time): one step's boundaries equal
     expected_sequence with and without the pool; arguments and results pass through; restore gives back the original
     objects (snapshot_targets / moved_targets); suspend / resume; a second install is refused; the proxy keeps the
     other attributes. NEGATIVE CONTROL: a fake engine that calls the V gather through a private alias loses its
     boundaries, and the chain check names the first missing one.
PP5  SelectionProbe with CPU torch and fake stage-1 / pooling: the labels; the 1-row call is the shipped call; the
     R-row call has cu_seqlens_q 0, R, 2R, ..., max_seqlen_q R and the rows repeated; the pooling sees B and R x B rows;
     the top-k result; the shipped inputs unchanged; the buffers reused; an error disables the probe.
PP6  CaptureTimer (enter / exit times, the capturing flag, restore, an enter that raises, a second install refused);
     GcTimer (a fake clock; a real gc.collect() counted in generation 2).
PP7  TimerHooks + drive with fakes: the label and kind of every step (re-warm, edge, window, selection, steady); the
     profiler on at p0 and off after p1 - 1 with the wrappers suspended between; the probe on in the selection steps
     only; the re-capture rows; after an exception inside the window, stop() stops the profiler and restores the
     wrappers, the graph class and gc.callbacks. NsysHooks: the profiler range and the NVTX ranges.
PP8  stage_summary, selection_summary, ceiling and pays, by hand arithmetic.
PP9  launch_kind, launch_counts, busy_union_ms on a synthetic Kineto trace; pack_trace keeps a .gz and removes the .json.
PP10 recapture_summary; load_baseline on the real mb1 record (skipped when absent) and on a broken file;
     baseline_ratios (same cell / another cell / no baseline).
PP11 verdicts_p0 and cell_key: the committed block is TBD (or a stamped UTC); TBD -> UNREGISTERED, P0-HEAD from the
     plan's rule; a stamped block gives CONFIRMED / REFUTED; NOT_RUN without values; every per-cell dict has the keys of
     the step script's cells.
PP12 main() refusals before torch: --out exists, no model.
PP13 static: the shipped call order and the module-level names that the boundaries assume (nosa_llama.py,
     cache_engine.py of this checkout); decode_loop repeats run_shipped's device lines inside try / finally;
     FLAG OFF = BASELINE: nosi/, r2_base_cell.py, r2_ragged_gate.py and r2_g0_compare.py equal ded9d41.
PP14 load_workload: the 128K workload (shape, sha256, 24 documents; skipped when absent); a wrong shape and a wrong
     sha256 are REFUSED.
PP15 r2_base_profile_step.sh: malformed cells, holder ids, usage and P0_HOLDERS outside DRYRUN STOP; a fake sha passes
     the cell stage; at a clean checkout of the ROOT, DRYRUN prints each cell's C, pool, B, S, steps, STEP_MIN, --mem,
     python file and arguments (mb1 = the python arguments of the basef mb1 record), the baseline argument, the nsys
     wrapper, and the p0all order for an even and an odd holder.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_p0prof.py
"""
from __future__ import annotations

import gc
import gzip
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
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-p0sweep"    # the step script's ROOT (p0sweep moved it from nosi-r2-p0prof)
BASE_SHA = "ded9d412d81aef76e2c1b6983673e6eaa7010adc"
MB1 = Path("/mnt/beegfs/ojerbi/r2-runs/basef/basef_mba_2190876_mb1")
WLK = Path("/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx130816")
WLK_SHA = "ff5df555025599d59d0c5c586c39ec7ed112cede78175b565387c0dde185a5d2"
UUID = "f31fc723-9dfe-407a-9947-28173556232a"


def _pc():
    sys.path.insert(0, str(EFF))
    import r2_base_profile_cell as PC
    return PC


def _bc():
    sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


class FakeClock:
    """Every record advances the time by 1 ms; bump() adds more (a long stage)."""

    def __init__(self):
        self.now = 0.0
        self.waits = 0

    def event(self):
        return [None]

    def record(self, e):
        self.now += 1.0
        e[0] = self.now

    def bump(self, ms):
        self.now += ms

    def elapsed(self, a, b):
        assert a[0] is not None and b[0] is not None
        return b[0] - a[0]

    def wait(self, e):
        assert e[0] is not None
        self.waits += 1

    def sync(self):
        pass


def _play(rec, seq, bump=None, skip=None):
    """Mark the labels of one timed step after its first boundary, as the wrappers do (rec.layer at qkv_gemm)."""
    layer = 0
    skipped = False
    for lab in seq[1:]:
        if lab == "qkv_gemm":
            rec.layer(layer)
            layer += 1
        if lab == skip and not skipped:
            skipped = True
            continue
        rec.mark(lab)
        if bump and (lab, layer - 1) in bump:
            rec.clock.bump(bump[(lab, layer - 1)])


# ------------------------------------------------------------------ PP1
def test_PP1_parse_args():
    PC, BC = _pc(), _bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = PC.parse_args(base)
    assert (a.stage_timers, a.nsys_window, a.prof_start, a.prof_steps, a.ring) == (False, None, 8, 3, 4)
    assert (a.sel_start, a.sel_steps, a.sel_rows, a.context, a.pool, a.mbase) == (24, 8, 4, 16128, 0, False)
    assert PC.parse_args(base + ["--stage-timers", "--pool", "65", "--mbase"]).stage_timers
    assert PC.parse_args(base + ["--nsys-window", "8", "--steps", "40"]).nsys_window == 8
    assert PC.parse_args(base + ["--stage-timers", "--sel-steps", "0"]).sel_steps == 0
    assert PC.parse_args(base + ["--stage-timers", "--context", "130816", "--pool", "65"]).context == 130816
    bad = (["--stage-timers", "--nsys-window", "8"], ["--nsys-window", "50"], ["--nsys-window", "25", "--steps", "40"],
           ["--nsys-window", "1"], ["--stage-timers", "--prof-start", "62"], ["--stage-timers", "--prof-start", "1"],
           ["--stage-timers", "--prof-steps", "0"], ["--stage-timers", "--prof-steps", "17"],
           ["--stage-timers", "--sel-start", "60"], ["--stage-timers", "--sel-start", "6"],
           ["--stage-timers", "--sel-rows", "1"], ["--stage-timers", "--sel-steps", "1"],
           ["--stage-timers", "--sel-start", "508"], ["--mbase", "--context", "130816"], ["--context", "32768"],
           ["--pool", "66"], ["--ring", "1"], ["--steps", "1001"])
    for b in bad:
        with pytest.raises(SystemExit):
            PC.parse_args(base + b)
    r16 = [t for t, _ in BC.expected_rewarms(16128, 512)]
    assert r16 == [0, 64, 128, 192, 256, 288, 320, 384, 448]
    assert [t for t, _ in BC.expected_rewarms(130816, 512)] == r16


# ------------------------------------------------------------------ PP2
def test_PP2_expected_sequence_and_groups():
    PC = _pc()
    for pool, sel, n in ((True, False, 482), (False, False, 386), (True, True, 642), (False, True, 546)):
        s = PC.expected_sequence(32, pool, sel)
        assert len(s) == n, (pool, sel, len(s))
        assert s[0] == "embed" and s[-1] == "sample" and s[-2] == "final"
        assert s.count("final") == 1 and s.count("inter") == 31 and s.count("qkv_gemm") == 32
        assert ("pool_swap" in s) == pool and ("sel_prep" in s) == sel
        if sel:
            i = s.index("stage1")
            assert s[i:i + 7] == ("stage1",) + PC.SEL_LABELS + ("pool_topk",)
    grouped = [x for _, ls in PC.GROUPS for x in ls]
    assert len(grouped) == len(set(grouped))
    steady = set(PC.expected_sequence(2, True)) | set(PC.expected_sequence(2, False))
    assert steady == set(grouped)
    assert not set(PC.SEL_LABELS) & set(grouped) and not PC.STEP_LABELS & set(grouped) - {"embed", "sample"}
    assert all(set(ls) <= set(grouped) for _, ls in PC.SAVES)
    assert [g for g, _ in PC.GROUPS] == ["selection", "resolve", "pool_swap", "fetch", "attention", "gemm_mlp", "rest"]


# ------------------------------------------------------------------ PP3
def _run_recorder(ring, n_layers=2):
    PC = _pc()
    seq = PC.expected_sequence(n_layers, True)
    sseq = PC.expected_sequence(n_layers, True, sel=True)
    clock = FakeClock()
    rec = PC.StageRecorder(clock, n_layers, dict(main=seq, sel=sseq), ring=ring, cap=4)
    rec.step_begin(0, False, "rewarm")
    _play(rec, seq)                                   # not inner: no mark is recorded
    for t in range(1, 6):
        rec.step_begin(t, True, "embed")
        _play(rec, seq, bump={("fetch_k", 0): 5.0})
    rec.step_begin(6, True, "embed", "sel")
    _play(rec, sseq)
    rec.finish()
    return rec, seq, sseq, clock


def test_PP3_stage_recorder():
    PC = _pc()
    rec, seq, sseq, clock = _run_recorder(4)
    for ring in (2, 3):
        other = _run_recorder(ring)[0]
        assert other.steps == rec.steps and other.layer_sum == rec.layer_sum and other.chain == rec.chain
    assert set(rec.steps) == set(range(7)) and clock.waits == 7
    r0 = rec.steps[0]
    assert r0 == dict(period_ms=1.0, inner=False, label="rewarm", kind="main")
    for t in range(1, 6):
        r = rec.steps[t]
        assert r["period_ms"] == len(seq) + 5.0 and r["kind"] == "main"
        assert r["stages"]["fetch_k"] == 2 + 5.0 and r["stages"]["qkv_gemm"] == 2.0 and r["stages"]["final"] == 1.0
        assert r["stages"]["embed"] == 1.0 and r["stages"]["sample"] == 1.0
        assert sum(r["stages"].values()) == r["period_ms"]
    assert rec.steps[6]["period_ms"] == len(sseq) and rec.steps[6]["kind"] == "sel"
    assert rec.closure_max_ms == 0.0
    assert rec.n_inner == dict(main=5, sel=1)
    assert rec.chain_of("main") == dict(ok=5, bad=0, examples=[]) and rec.chain_of("sel")["ok"] == 1
    assert rec.layer_sum["fetch_k"] == [30.0, 5.0] and rec.per_layer_mean()["fetch_k"] == [6.0, 1.0]
    assert "embed" not in rec.layer_sum and "sel_prep" not in rec.layer_sum       # step labels; sel steps not in it
    assert len(rec.ev[0]) > 4                                                      # the set grew past cap 4
    # NEGATIVE CONTROL: one boundary skipped
    clock = FakeClock()
    rec = PC.StageRecorder(clock, 2, dict(main=seq), ring=2)
    rec.step_begin(1, True, "embed")
    _play(rec, seq, skip="fetch_v")
    rec.step_begin(2, True, "embed")
    _play(rec, seq)
    rec.finish()
    c = rec.chain_of("main")
    j = seq.index("fetch_v")
    assert (c["ok"], c["bad"]) == (1, 1)
    assert c["examples"][0] == dict(t=1, index=j, got="writeback", expected="fetch_v", n_got=len(seq) - 1,
                                    n_expected=len(seq))
    assert sum(rec.steps[1]["stages"].values()) == rec.steps[1]["period_ms"]
    # capturing blocks the marks; a step without an inner flag keeps its period only
    rec = PC.StageRecorder(FakeClock(), 1, None, ring=2)
    rec.step_begin(0, True, "embed")
    rec.capturing = True
    rec.mark("stage1")
    rec.capturing = False
    rec.mark("sample")
    rec.finish()
    assert set(rec.steps[0]["stages"]) == {"embed", "sample"}
    with pytest.raises(ValueError):
        PC.StageRecorder(FakeClock(), 1, None, ring=1)


# ------------------------------------------------------------------ PP4
def _fake_world(pool=True, alias_bypass=False):
    nl = types.ModuleType("fake_nl")
    cen = types.ModuleType("fake_cen")
    calls = []

    def nosa_linear(x):
        calls.append("nosa_linear")
        return ("lin", x)

    def stage1(q, k, v, **kw):
        calls.append("stage1")
        return ("s1", q, kw["max_seqlen_k"])

    def attn(q, *a, **kw):
        calls.append("attn")
        return ("attn", q)
    nl.nosa_linear, nl.infllmv2_attn_stage1_fast, nl.flash_attn_nosa_with_kvcache = nosa_linear, stage1, attn

    class LlamaLayer:
        def __init__(self, i):
            self.layer_idx = i

        def decode_forward(self, x, cache):           # the shipped order (nosa_llama.py:558, :592, :618, :634)
            nl.nosa_linear(x)
            nl.infllmv2_attn_stage1_fast(x, "k", "k", cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=1,
                                         max_seqlen_k=8, causal=False)
            cache.decode_update_kv(x, self.layer_idx)
            nl.flash_attn_nosa_with_kvcache(x)
            return x + 1
    nl.LlamaLayer = LlamaLayer

    def h2d(*a):
        calls.append("h2d_k")

    def h2d_bias(*a):
        calls.append("h2d_v")
    cen.diff = types.SimpleNamespace(diff_offload=lambda *a: calls.append("diff"), other="kept")
    cen._pool = types.SimpleNamespace(pool_update=lambda *a: calls.append("pool_update")) if pool else None
    cen._flash_pool_swap = (lambda *a: calls.append("swap")) if pool else None
    cen.flash_h2d_from_mask, cen.flash_h2d_from_mask_bias = h2d, h2d_bias
    private = h2d_bias                                # an alias bound before any patch

    class InfLLMv2Cache:
        def decode_update_kv(self, x, layer_idx):     # the shipped order (cache_engine.py:480-519)
            cen.diff.diff_offload()
            if cen._pool is not None:
                cen._pool.pool_update()
                cen._flash_pool_swap()
            cen.flash_h2d_from_mask()
            (private if alias_bypass else cen.flash_h2d_from_mask_bias)()
            return "kv%d" % layer_idx
    cen.InfLLMv2Cache = InfLLMv2Cache
    cuda = types.SimpleNamespace(graph=object)
    return nl, cen, cuda, calls


def _decode_step(rec, layers, cache, t):
    rec.step_begin(t, True, "embed")
    x = 0
    for layer in layers:
        x = layer.decode_forward(x, cache)
    rec.mark("sample")


def test_PP4_patcher_on_fakes():
    PC = _pc()
    for pool in (True, False):
        nl, cen, cuda, calls = _fake_world(pool)
        n = 3
        rec = PC.StageRecorder(FakeClock(), n, dict(main=PC.expected_sequence(n, pool)), ring=2)
        cache = cen.InfLLMv2Cache()
        before = PC.snapshot_targets(nl, cen, cuda)
        orig_stage1 = nl.infllmv2_attn_stage1_fast
        p = PC.Patcher(rec, nl, cen, n)
        p.install(cache)
        assert PC.moved_targets(before, PC.snapshot_targets(nl, cen, cuda)) != []
        assert nl.infllmv2_attn_stage1_fast.__wrapped__ is orig_stage1 and cen.diff.other == "kept"
        assert nl.nosa_linear(5) == ("lin", 5) and cache.decode_update_kv(0, 1) == "kv1"
        with pytest.raises(RuntimeError):
            p.install(cache)
        calls.clear()
        layers = [nl.LlamaLayer(i) for i in range(n)]
        for t in range(3):
            _decode_step(rec, layers, cache, t)
        one = ["nosa_linear", "stage1", "diff"] + (["pool_update", "swap"] if pool else []) + ["h2d_k", "h2d_v", "attn"]
        assert calls == one * n * 3
        p.suspend()
        assert PC.moved_targets(before, PC.snapshot_targets(nl, cen, cuda)) == []
        assert "decode_update_kv" not in vars(cache)
        p.resume()
        assert nl.nosa_linear is not before["NL.nosa_linear"] and "decode_update_kv" in vars(cache)
        p.restore()
        rec.finish()
        assert PC.moved_targets(before, PC.snapshot_targets(nl, cen, cuda)) == []
        assert "decode_update_kv" not in vars(cache) and p.installs == 2
        assert rec.chain_of("main") == dict(ok=3, bad=0, examples=[]), (pool, rec.chain_of("main"))
    # NEGATIVE CONTROL: the V gather through a private alias never reaches its wrapper
    nl, cen, cuda, calls = _fake_world(True, alias_bypass=True)
    seq = PC.expected_sequence(2, True)
    rec = PC.StageRecorder(FakeClock(), 2, dict(main=seq), ring=2)
    cache = cen.InfLLMv2Cache()
    p = PC.Patcher(rec, nl, cen, 2)
    p.install(cache)
    _decode_step(rec, [nl.LlamaLayer(0), nl.LlamaLayer(1)], cache, 0)
    p.restore()
    rec.finish()
    c = rec.chain_of("main")
    assert c["bad"] == 1 and c["examples"][0]["expected"] == "fetch_v" and c["examples"][0]["got"] == "attention"
    # moved_targets compares by identity, gc.callbacks element by element
    a = dict(x=1.5, y=None)
    assert PC.moved_targets(a, dict(a)) == [] and PC.moved_targets(a, dict(x=float("1.5"), y=None)) == ["x"]
    f = lambda *x: None                                                            # noqa: E731
    assert PC.moved_targets({"gc.callbacks": (f,)}, {"gc.callbacks": ()}) == ["gc.callbacks"]


# ------------------------------------------------------------------ PP5
def test_PP5_selection_probe():
    torch = pytest.importorskip("torch")
    PC = _pc()
    B, R, H, W, OUT = 3, 4, 2, 16, 12
    calls = []

    def stage1(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        calls.append(dict(kind="stage1", rows=q.shape[0], cu=cu_seqlens_q.tolist(), mq=max_seqlen_q, mk=max_seqlen_k,
                          causal=causal, q=q.clone()))
        return q[:, :H, 0].t().unsqueeze(-1).expand(H, q.shape[0], W).contiguous().float()

    def pooling(score, cis, max_k, pq, pc, stride, block_size, local_blocks, init_blocks):
        calls.append(dict(kind="pool", rows=score.shape[1], cis_rows=cis.shape[1], mk=max_k,
                          args=(stride, block_size, local_blocks, init_blocks)))
        pq.copy_(torch.arange(pq.shape[-1], dtype=pq.dtype).expand_as(pq))
        pc.copy_(torch.arange(pc.shape[-1], dtype=pc.dtype).flip(0).expand_as(pc))
    layer = types.SimpleNamespace(compressed_cis_buf=torch.randn(H, B, W), topk_blocks=4, pooling_stride=16,
                                  block_size=64, window_blocks=16, init_blocks=1)
    rec = PC.StageRecorder(FakeClock(), 1, None, ring=2)
    probe = PC.SelectionProbe(rec, [layer], lambda: (OUT, torch.float32), stage1, pooling, torch, rows=R, qk_select=3)
    q = torch.randn(B, 32, 8)
    cu = torch.arange(0, B + 1, dtype=torch.int32)
    q0, cis0 = q.clone(), layer.compressed_cis_buf.clone()
    kw = dict(cu_seqlens_q=cu, cu_seqlens_k="cuk", max_seqlen_q=1, max_seqlen_k=8, causal=False)
    rec.step_begin(0, True, "embed")
    rec.layer(0)
    probe.run((q, "k", "k"), kw)
    s = rec.cur
    assert rec.lab[s][:rec.n[s]] == ["embed"] + list(PC.SEL_LABELS)
    st = [c for c in calls if c["kind"] == "stage1"]
    po = [c for c in calls if c["kind"] == "pool"]
    assert (st[0]["rows"], st[0]["cu"], st[0]["mq"], st[0]["mk"], st[0]["causal"]) == (B, list(range(B + 1)), 1, 8, False)
    assert (st[1]["rows"], st[1]["cu"], st[1]["mq"]) == (B * R, list(range(0, B * R + 1, R)), R)
    assert all(torch.equal(st[1]["q"][R * b + j], q[b]) for b in range(B) for j in range(R))
    assert [(c["rows"], c["cis_rows"], c["mk"], c["args"]) for c in po] == [(B, B, 8, (16, 64, 16, 1)),
                                                                            (B * R, B * R, 8, (16, 64, 16, 1))]
    assert torch.equal(q, q0) and torch.equal(layer.compressed_cis_buf, cis0)
    assert probe.calls == 1 and not probe.disabled and len(probe.bufs) == 2
    b = probe.bufs[("rows", H, B * R, OUT, 4)]
    assert b["i"].shape == (H, B * R, 4)
    assert all(set(row.tolist()) == {0, OUT - 3, OUT - 2, OUT - 1} for row in b["i"].reshape(-1, 4))
    ids = {k: id(v["pq"]) for k, v in probe.bufs.items()}
    probe.run((q, "k", "k"), kw)
    assert {k: id(v["pq"]) for k, v in probe.bufs.items()} == ids and probe.calls == 2
    bad = dict(kw)
    del bad["max_seqlen_k"]
    n0 = rec.n[rec.cur]
    probe.run((q, "k", "k"), bad)
    assert probe.disabled and len(probe.errors) == 1 and "KeyError" in probe.errors[0]
    probe.run((q, "k", "k"), kw)
    assert rec.n[rec.cur] == n0 and probe.calls == 2


# ------------------------------------------------------------------ PP6
def test_PP6_capture_and_gc_timers():
    PC = _pc()
    now = [0.0]

    class Base:
        def __init__(self, g, fail=False):
            self.g, self.fail = g, fail

        def __enter__(self):
            if self.fail:
                raise ValueError("enter")
            now[0] += 0.007
            return None

        def __exit__(self, *exc):
            now[0] += 0.002
            return False
    rec = types.SimpleNamespace(capturing=False)
    ct = PC.CaptureTimer(rec, now=lambda: now[0])
    owner = types.SimpleNamespace(graph=Base)
    ct.install(owner)
    assert owner.graph is not Base and issubclass(owner.graph, Base)
    with pytest.raises(RuntimeError):
        ct.install(owner)
    with owner.graph("g"):
        assert rec.capturing
    assert not rec.capturing
    with owner.graph("g"):
        pass
    got = ct.take()
    assert got["n_captures"] == 2 and got["graph_enter_ms"] == pytest.approx(14.0) and \
        got["graph_exit_ms"] == pytest.approx(4.0) and got["graph_enter_first_ms"] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        with owner.graph("g", fail=True):
            pass
    assert not rec.capturing and ct.take()["n_captures"] == 0
    ct.restore()
    assert owner.graph is Base
    t = [0.0]
    g = PC.GcTimer(now=lambda: t[0])
    g("start", {})
    t[0] += 0.003
    g("stop", dict(generation=1))
    g("stop", dict(generation=2))                     # a stop without a start is ignored
    assert g.take() == (3.0, [0, 1, 0]) and g.take() == (0.0, [0, 0, 0])
    real = PC.GcTimer()
    gc.callbacks.append(real)
    try:
        gc.collect()
    finally:
        gc.callbacks.remove(real)
    ms, count = real.take()
    assert count[2] >= 1 and ms >= 0.0


# ------------------------------------------------------------------ PP7
class FakePatcher:
    def __init__(self, log):
        self.log, self.on = log, False

    def install(self, cache):
        self.log.append("install")
        self.on = True

    def suspend(self):
        self.log.append("suspend")
        self.on = False

    def resume(self):
        self.log.append("resume")
        self.on = True

    def restore(self):
        self.log.append("restore")
        self.on = False


class FakeProf:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append("prof.start")

    def stop(self):
        self.log.append("prof.stop")


def _hooks_world(PC, fail_at=None):
    log = []

    class Base:
        def __init__(self, g):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False
    clock = FakeClock()
    rec = PC.StageRecorder(clock, 1, None, ring=2)
    patcher = FakePatcher(log)
    cuda = types.SimpleNamespace(graph=Base)
    gc_mod = types.SimpleNamespace(callbacks=[])
    probe = types.SimpleNamespace(active=False)
    hooks = PC.TimerHooks(rec, patcher, PC.CaptureTimer(rec), PC.GcTimer(), (4, 6), clock, lambda: FakeProf(log),
                          cuda, 20, gc_mod=gc_mod, sel_window=(8, 11), probe=probe)
    st = dict(t=0)
    seen = {}

    def rewarm_check():
        t = st["t"]
        return t in (0, 12), ("first" if t == 0 else ("out_len" if t == 12 else ""))

    def decode():
        t = st["t"]
        seen[t] = (rec.inner, probe.active, patcher.on)
        if t in (0, 12):
            with cuda.graph("g"):
                pass
        if t == 12:
            for cb in gc_mod.callbacks:
                cb("start", {})
                cb("stop", dict(generation=2))
        if t == fail_at:
            raise RuntimeError("decode failed")

    def sample(t):
        st["t"] = t + 1
    return hooks, rec, log, cuda, Base, gc_mod, seen, rewarm_check, decode, sample


def test_PP7_timer_and_nsys_hooks():
    PC = _pc()
    hooks, rec, log, cuda, Base, gc_mod, seen, rw, dec, smp = _hooks_world(PC)
    hooks.start(types.SimpleNamespace())
    assert cuda.graph is not Base and gc_mod.callbacks == [hooks.gct]
    try:
        rewarms = PC.drive(20, rw, dec, smp, hooks)
    finally:
        hooks.stop()
    assert rewarms == [(0, "first"), (12, "out_len")]
    lab = {t: (rec.steps[t]["label"], rec.steps[t]["kind"], rec.steps[t]["inner"]) for t in range(20)}
    assert lab[0] == ("rewarm", "main", False) and lab[12] == ("rewarm", "main", False)
    assert [lab[t][0] for t in (3, 4, 5, 6)] == ["edge", "window", "window", "edge"]
    assert all(lab[t] == ("embed", "sel", True) for t in (8, 9, 10))
    assert all(lab[t] == ("embed", "main", True) for t in (1, 2, 7, 11, 13, 19))
    assert [t for t in range(20) if seen[t][1]] == [8, 9, 10]                      # the probe: sel steps only
    assert [t for t in range(20) if not seen[t][2]] == [4, 5]                      # the wrappers: off in the window
    assert log == ["install", "suspend", "prof.start", "prof.stop", "resume", "restore"]
    assert [(r["t"], r["why"], r["n_captures"]) for r in hooks.recap] == [(0, "first", 1), (12, "out_len", 1)]
    assert hooks.recap[1]["gc_count"] == [0, 0, 1] and hooks.recap[0]["gc_count"] == [0, 0, 0]
    assert all(x is not None for x in hooks.host_ms) and cuda.graph is Base and gc_mod.callbacks == []
    assert hooks.instance_left is False and not hooks.probe.active
    # an exception inside the profiler window: stop() stops the profiler and restores everything
    hooks, rec, log, cuda, Base, gc_mod, seen, rw, dec, smp = _hooks_world(PC, fail_at=5)
    hooks.start(types.SimpleNamespace())
    with pytest.raises(RuntimeError):
        try:
            PC.drive(20, rw, dec, smp, hooks)
        finally:
            hooks.stop()
    assert log == ["install", "suspend", "prof.start", "prof.stop", "restore"]
    assert cuda.graph is Base and gc_mod.callbacks == [] and not hooks.prof_running
    # NsysHooks
    log = []
    clock = FakeClock()
    rec = PC.StageRecorder(clock, 1, None, ring=2, cap=4)
    nvtx = types.SimpleNamespace(range_push=lambda s: log.append("push " + s), range_pop=lambda: log.append("pop"))
    nh = PC.NsysHooks(rec, (2, 4), clock, FakeProf(log), nvtx)
    st = dict(t=0)
    PC.drive(6, lambda: (st["t"] == 0, "first" if st["t"] == 0 else ""), lambda: None,
             lambda t: st.update(t=t + 1), nh)
    nh.stop()
    assert log == ["prof.start", "push p0.step2", "pop", "push p0.step3", "pop", "prof.stop"]
    assert [rec.steps[t]["label"] for t in range(6)] == ["rewarm", "embed", "window", "window", "embed", "embed"]
    assert all(not rec.steps[t]["inner"] for t in range(6))


# ------------------------------------------------------------------ PP8
def test_PP8_summaries_ceiling_pays():
    PC = _pc()
    s1 = {"stage1": 1.0, "pool_topk": 1.0, "fetch_k": 2.0, "fetch_v": 1.0, "attention": 3.0, "qkv_gemm": 1.0,
          "ffn_gemm": 1.0, "mystery": 0.0}
    steps = {1: dict(period_ms=10.0, inner=True, kind="main", stages=s1),
             2: dict(period_ms=20.0, inner=True, kind="main", stages={k: 2 * v for k, v in s1.items()}),
             3: dict(period_ms=5.0, inner=False, kind="main", label="rewarm")}
    st = PC.stage_summary(steps, [1, 2, 3])
    assert st["n"] == 2 and st["groups"]["selection"]["median_ms"] == 3.0 and st["groups"]["fetch"]["median_ms"] == 4.5
    assert st["group_frac"]["fetch"] == 0.3 and st["group_frac"]["attention"] == 0.3 and st["group_frac"]["rest"] == 0.0
    assert st["save_frac"] == dict(gather_only=0.3, fetch_path=0.3) and st["closure_max_ms"] == 0.0
    assert st["ungrouped"] == ["mystery"] and st["period"]["median_ms"] == 15.0
    assert PC.stage_summary(steps, [3]) == dict(n=0)
    ss = {"stage1": 1.0, "pool_topk": 1.0, "sel_prep": 0.5, "sel1_stage1": 2.0, "sel1_pool": 1.0, "selr_stage1": 3.0,
          "selr_pool": 3.0}
    sel_steps = {t: dict(period_ms=40.0, inner=True, kind="sel", stages=dict(ss)) for t in (24, 25, 26, 27)}
    sel_steps[28] = dict(period_ms=40.0, inner=True, kind="main", stages=dict(ss))  # not a probe step: ignored
    s = PC.selection_summary(sel_steps, [25, 26, 27, 28], 4)
    assert (s["n"], s["rows"], s["K"]) == (3, 4, 3)
    assert (s["shipped"]["median_ms"], s["one_row"]["median_ms"], s["r_rows"]["median_ms"]) == (2.0, 3.0, 6.0)
    assert (s["extra_joint_ms"], s["extra_serial_ms"], s["extra_serial_shipped_ms"]) == (3.0, 9.0, 6.0)
    assert (s["per_position_joint_ms"], s["rows_over_one"], s["prep"]["median_ms"]) == (1.0, 2.0, 0.5)
    assert PC.selection_summary(sel_steps, [], 4) == dict(n=0, rows=4, K=3)
    c = PC.ceiling(60.0, 216, dict(gather_only=0.25, fetch_path=0.5),
                   sel=dict(K=3, extra_joint_ms=20.0, extra_serial_ms=40.0), ref_tok_s=3387.7)
    g = c["gather_only"]
    assert (g["save_ms"], g["draft_floor_ms"], g["max_speedup"], g["ceiling_tok_s"]) == (15.0, 45.0, 1.3333, 4800.0)
    assert (g["save_over_rest"], g["ceiling_vs_ref"]) == (0.3333, 1.4169)
    assert g["verify_budget_ms"] == {"1": 15.0, "2": 30.0, "3": 45.0, "4": 60.0}
    assert g["verify_budget_after_sel_joint_ms"] == {"1": -5.0, "2": 10.0, "3": 25.0, "4": 40.0}
    assert g["verify_budget_after_sel_serial_ms"] == {"1": -25.0, "2": -10.0, "3": 5.0, "4": 20.0}
    assert (g["min_accepted_joint"], g["min_accepted_serial"]) == (2, 3)
    f = c["fetch_path"]
    assert (f["draft_floor_ms"], f["max_speedup"], f["ceiling_tok_s"], f["save_over_rest"]) == (30.0, 2.0, 7200.0, 1.0)
    assert (f["min_accepted_joint"], f["min_accepted_serial"]) == (1, 2)
    assert c["baseline_tok_s"] == 3600.0 and c["K"] == 3 and "K extra selections" in c["formula"]
    c = PC.ceiling(60.0, 216, dict(gather_only=1.0))
    assert c["gather_only"]["draft_floor_ms"] is None and c["extra_sel_joint_ms"] is None
    c = PC.ceiling(60.0, 216, dict(gather_only=0.25))
    assert "verify_budget_after_sel_joint_ms" not in c["gather_only"]
    assert PC.pays(45.0, 20.0, 2, 60.0) and not PC.pays(45.0, 20.0, 2, 60.0, extra_sel_ms=20.0)
    assert not PC.pays(45.0, 30.0, 2, 60.0)                                        # equality does not pay


# ------------------------------------------------------------------ PP9
def _x(cat, name, ts=0.0, dur=1.0, corr=None):
    e = dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, pid=1, tid=1)
    if corr is not None:
        e["args"] = dict(correlation=corr)
    return e


def test_PP9_launches_busy_pack(tmp_path):
    PC = _pc()
    assert [PC.launch_kind(n) for n in ("cudaLaunchKernelExC", "cuLaunchKernel", "cudaGraphLaunch",
                                        "cuMemcpyHtoDAsync_v2", "cudaMemsetAsync", "cudaEventRecord",
                                        "cudaStreamSynchronize", "cudaFuncGetAttributes")] == [
        "kernel", "kernel", "graph", "memcpy", "memset", "event", "sync", "other"]
    ev = ([_x("cuda_runtime", "cudaLaunchKernel")] * 4 + [_x("cuda_driver", "cuLaunchKernel")] * 2
          + [_x("cuda_runtime", "cudaLaunchKernelExC"), _x("cuda_runtime", "cudaGraphLaunch")]
          + [_x("cuda_runtime", "cudaMemcpyAsync")] * 2 + [_x("cuda_runtime", "cudaMemsetAsync")]
          + [_x("cuda_runtime", "cudaEventRecord")] * 3 + [_x("cuda_runtime", "cudaStreamSynchronize")]
          + [_x("cuda_runtime", "cudaGetDevice")] * 5 + [_x("cpu_op", "aten::mm"), dict(ph="i", name="x")])
    lc = PC.launch_counts(dict(traceEvents=ev), 2)
    assert lc["launches_per_step"] == 5.5
    assert lc["kinds_per_step"] == dict(event=1.5, graph=0.5, kernel=3.5, memcpy=1.0, memset=0.5, other=2.5, sync=0.5)
    assert list(lc["names_per_step"])[:2] == ["cudaGetDevice", "cudaLaunchKernel"] and "aten::mm" not in lc["names_per_step"]
    assert PC.launch_counts(ev, 1)["launches_per_step"] == 11.0
    ops = [dict(ts=0.0, dur_us=10.0), dict(ts=5.0, dur_us=10.0), dict(ts=20.0, dur_us=5.0), dict(ts=30.0, dur_us=0.0)]
    assert PC.busy_union_ms(ops, 1) == 0.02 and PC.busy_union_ms(ops, 2) == 0.01 and PC.busy_union_ms([], 1) == 0.0
    tr = dict(traceEvents=[_x("cuda_runtime", "cudaLaunchKernel", 0, 2, corr=1),
                           _x("kernel", "flash_h2d_from_mask_kernel", 10, 7, corr=1)])
    path = tmp_path / "trace_T.json"
    path.write_text(json.dumps(tr))
    ops, lc = PC.pack_trace(str(path), 1)
    assert [(o["name"], o["cls"], o["dur_us"], o["range"]) for o in ops] == [("flash_h2d_from_mask_kernel", "H2D", 7.0, None)]
    assert lc["launches_per_step"] == 1.0 and not path.exists()
    with gzip.open(str(path) + ".gz", "rt") as f:
        assert json.load(f) == tr


# ------------------------------------------------------------------ PP10
def test_PP10_recapture_baseline(tmp_path):
    PC = _pc()

    def row(t, why, host, gc_ms, enter, exit_):
        return dict(t=t, why=why, host_ms=host, gc_ms=gc_ms, gc_count=[0, 0, 1], n_captures=32, graph_enter_ms=enter,
                    graph_enter_first_ms=1.0, graph_exit_ms=exit_)
    recap = [row(0, "first", 100.0, 10.0, 50.0, 5.0), row(64, "out_len", 60.0, 30.0, 40.0, 4.0),
             row(128, "out_len", 80.0, 50.0, 50.0, 6.0)]
    steps = {0: dict(period_ms=120.0), 64: dict(period_ms=100.0), 128: dict(period_ms=50.0)}
    r = PC.recapture_summary(recap, steps, [11000.0, 6000.0, 5900.0])
    assert r["n"] == 3 and r["each"][1]["host_other_ms"] == 16.0 and r["each"][1]["gc_share"] == 0.3
    assert r["each"][1]["headline_period_ms"] == 6000.0 and r["each"][0]["period_ms"] == 120.0
    lm = r["later_median"]
    assert (lm["period_ms"], lm["gc_ms"], lm["gc_share"], lm["headline_period_ms"]) == (75.0, 40.0, 0.65, 5950.0)
    assert PC.recapture_summary([], {})["later_median"]["period_ms"] is None
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert "error" in PC.load_baseline(str(bad)) and "error" in PC.load_baseline(str(tmp_path / "none.json"))
    if (MB1 / "summary.json").is_file():
        b = PC.load_baseline(str(MB1 / "summary.json"))
        assert (b["C"], b["B"], b["S"], b["median_ms"], b["mean_ms"]) == (128, 216, 16128, 63.761, 67.141), b
        assert (b["effective_tok_s"], b["steady_tok_s"]) == (1201.1, 3387.7)
    base = dict(C=128, B=216, S=16128, median_ms=63.761, mean_ms=67.141, effective_tok_s=1201.1, steady_tok_s=3387.7)
    h = dict(steady=dict(median_ms=64.398, mean_ms=67.141), rates=dict(effective_tok_s=1201.1, steady_tok_s=3354.1))
    x = PC.baseline_ratios(128, 216, 16128, h, base)
    assert x["same_cell"] and (x["median_ratio"], x["mean_ratio"], x["effective_ratio"]) == (1.01, 1.0, 1.0)
    assert x["steady_tok_ratio"] == round(3354.1 / 3387.7, 4)
    assert not PC.baseline_ratios(63, 344, 16128, h, base)["same_cell"]
    assert not PC.baseline_ratios(128, 216, 130816, h, base)["same_cell"]
    assert PC.baseline_ratios(128, 216, 16128, h, dict(error="x")) is None and PC.baseline_ratios(1, 1, 1, None, base) is None


# ------------------------------------------------------------------ PP11
def _res(median_ratio=1.01, same=True):
    return dict(baseline_ratios=dict(same_cell=same, median_ratio=median_ratio, mean_ratio=1.0, effective_ratio=1.0,
                                     steady_tok_ratio=1.0),
                overhead_ratio=1.05,
                stages=dict(groups={g: dict(median_ms=v) for g, v in (("selection", 3.0), ("resolve", 1.0),
                                                                      ("pool_swap", 5.0), ("fetch", 7.5),
                                                                      ("attention", 19.0), ("gemm_mlp", 20.0),
                                                                      ("rest", 2.0))}),
                launches=dict(launches_per_step=1600.0), gap_ms=2.0,
                recapture=dict(later_median=dict(period_ms=6000.0, gc_share=0.95)),
                selection=dict(extra_joint_ms=15.0, K=3), ceiling=dict(gather_only=dict(max_speedup=1.15)))


def test_PP11_registered_block():
    PC = _pc()
    assert re.fullmatch(r"TBD|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", PC.REGISTERED_UTC)
    assert PC.PRED_P0["head_band"] == (0.98, 1.02) and "K extra selections" in PC.CEILING_FORMULA
    keys = {"128_216", "63_344", "63_216", "128_63_s130816"}
    for name in ("stage_ms", "launches", "gap_ms", "sel_extra_ms", "ceil_speedup"):
        assert set(PC.PRED_P0[name]) == keys, name
    assert (PC.cell_key(128, 216), PC.cell_key(63, 344, 16128), PC.cell_key(128, 63, 130816)) == (
        "128_216", "63_344", "128_63_s130816")
    v = {n: x for n, x, _ in PC.verdicts_p0(128, 216, _res(), utc="TBD")}
    assert v.pop("P0-HEAD") == "CONFIRMED" and set(v.values()) == {"UNREGISTERED"}, v
    assert {n: x for n, x, _ in PC.verdicts_p0(128, 216, _res(1.03), utc="TBD")}["P0-HEAD"] == "REFUTED"
    assert {n: x for n, x, _ in PC.verdicts_p0(63, 344, _res(same=False), utc="TBD")}["P0-HEAD"] == "REPORT"
    names = [n for n, _, _ in PC.verdicts_p0(128, 216, _res(), utc="TBD")]
    assert names == ["P0-HEAD", "P0-OVH"] + ["P0-STAGE-%s" % g for g, _ in PC.GROUPS] + [
        "P0-LAUNCH", "P0-GAP", "P0-RECAP", "P0-RECAP-GC", "P0-SEL", "P0-CEIL"]
    stamped = dict(PC.PRED_P0, ovh_band=(1.0, 1.1), stage_ms=dict(PC.PRED_P0["stage_ms"], **{"128_216": {"fetch": (7, 8)}}),
                   launches={"128_216": (1000, 1500)}, gap_ms={"128_216": (0, 5)}, recap_ms=(5000, 7000),
                   recap_gc_share=(0.5, 0.9), sel_extra_ms={"128_216": (10, 20)}, ceil_speedup={"128_216": (1.1, 1.2)})
    v = {n: x for n, x, _ in PC.verdicts_p0(128, 216, _res(), pred=stamped, utc="2026-10-27T20:00:00Z")}
    assert (v["P0-OVH"], v["P0-STAGE-fetch"], v["P0-STAGE-selection"], v["P0-LAUNCH"]) == (
        "CONFIRMED", "CONFIRMED", "UNREGISTERED", "REFUTED")
    assert (v["P0-GAP"], v["P0-RECAP"], v["P0-RECAP-GC"], v["P0-SEL"], v["P0-CEIL"]) == (
        "CONFIRMED", "CONFIRMED", "REFUTED", "CONFIRMED", "CONFIRMED")
    v = {n: x for n, x, _ in PC.verdicts_p0(128, 216, {}, pred=stamped, utc="2026-10-27T20:00:00Z")}
    assert v.pop("P0-HEAD") == "NOT_RUN" and set(v.values()) == {"NOT_RUN"}, v


# ------------------------------------------------------------------ PP12
def test_PP12_main_refusals(tmp_path, monkeypatch, capsys):
    PC = _pc()
    monkeypatch.delenv("NOSI_MODEL_PATH", raising=False)
    out = tmp_path / "exists"
    out.mkdir()
    assert PC.main(["--out", str(out), "--workload", "w", "--batch", "8", "--model", "m"]) == 20
    assert PC.main(["--out", str(tmp_path / "new"), "--workload", "w", "--batch", "8"]) == 20
    assert not (tmp_path / "new").exists() and "REFUSED" in capsys.readouterr().err


# ------------------------------------------------------------------ PP13
def _body(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def _in_order(body, needles):
    pos = [body.index(n) for n in needles]
    assert pos == sorted(pos), list(zip(needles, pos))
    for n in needles:
        assert body.count(n) == 1, (n, body.count(n))


def test_PP13_static_call_order_and_flag_off():
    PC, BC = _pc(), _bc()
    nl = (NOSI / "nosa_llama.py").read_text()
    for s in ("from .max_pooling_fused import nosa_pooling", "from .nosa_linear import nosa_linear",
              "    infllmv2_attn_stage1_fast,\n",
              "from flash_attn_nosa import flash_attn_with_kvcache as flash_attn_nosa_with_kvcache", "\nQK_SELECT = 33",
              "hidden_states = self.layers[idx].decode_forward("):
        assert s in nl, s
    body = _body(nl, "    def decode_forward(self, hidden_states", "\nclass Llama:")
    _in_order(body, ["= nosa_linear(", "score = infllmv2_attn_stage1_fast(", "self.after_pooling_graph.replay()",
                     "= cache_engine.decode_update_kv(", "attn_output = flash_attn_nosa_with_kvcache("])
    assert "cu_seqlens_q=cu_seqlens," in body and "max_seqlen_k=max_seqlen_comp," in body
    ce = (NOSI / "cache_engine.py").read_text()
    for s in ("\ndiff = load(", "from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask\n",
              "from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias\n", "\n_pool = None\n",
              "\n_flash_pool_swap = None\n", "self.decode_update = self.decode_update_has_kv_bias_pool",
              "return self.cache_engine.decode_update(", "return self.layers[layer_idx].decode_update_kv("):
        assert s in ce, s
    pool = _body(ce, "    def decode_update_has_kv_bias_pool(", "    def decode_update_no_kv_bias(")
    _in_order(pool, ["diff.diff_offload(", "_pool.pool_update(", "_flash_pool_swap(", "flash_h2d_from_mask(",
                     "flash_h2d_from_mask_bias(", "if tail_full:"])
    nopool = _body(ce, "    def decode_update_has_kv_bias(self", "    def decode_update_has_kv_bias_pool(")
    _in_order(nopool, ["diff.diff_offload(", "flash_h2d_from_mask(", "flash_h2d_from_mask_bias(", "if tail_full:"])
    rs, dl = inspect.getsource(BC.run_shipped), inspect.getsource(PC.decode_loop)
    for s in ("layer.cache_engine.max_gen_len = steps + 64", "model.batch_prefill(ids, cache)",
              "inputs = torch.empty((Bn, steps + 1), dtype=torch.int64, device=\"cuda\")", "inputs[:, 0:1].copy_(tok)",
              "cu = torch.arange(0, Bn + 1, dtype=torch.int, device=\"cuda\")", "model.has_buffers = False",
              ".argmax(dim=-1, keepdim=True)", "rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len"):
        assert s in rs and s in dl, s
    _in_order(dl, ["    try:", "hooks.start(cache)", "rewarms = drive(", "    finally:\n", "hooks.stop()"])
    assert "model.decode_inference(st[\"tok\"], cu, st[\"pos\"], cache)" in dl
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", BASE_SHA + "^{commit}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("base commit %s not in this repository" % BASE_SHA)
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", BASE_SHA, "--", "nosi",
                        "benchmarks/Efficiency/r2_base_cell.py", "benchmarks/Efficiency/r2_ragged_gate.py",
                        "benchmarks/Efficiency/r2_g0_compare.py"], capture_output=True)
    assert r.returncode == 0, "FLAG OFF = BASELINE broken: the engine or the baseline cell differs from %s" % BASE_SHA


# ------------------------------------------------------------------ PP14
def test_PP14_load_workload(tmp_path):
    PC = _pc()
    if (WLK / "ctx130816_tokens.bin").is_file():
        docs, sha, n = PC.load_workload(str(WLK), 130816)
        assert tuple(docs.shape) == (24, 130816) and n == 24 and sha == WLK_SHA and str(docs.dtype) == "torch.int64"
    d = tmp_path / "w"
    d.mkdir()
    (d / "ctx130816_manifest.json").write_text(json.dumps(dict(ctx=100, dtype="int32-le", shape=[1, 100])))
    with pytest.raises(SystemExit) as e:
        PC.load_workload(str(d), 130816)
    assert "REFUSED" in str(e.value)
    (d / "ctx130816_manifest.json").write_text(json.dumps(dict(ctx=130816, dtype="int32-le", shape=[1, 130816])))
    (d / "ctx130816_tokens.bin").write_bytes(bytes(4 * 130816))
    (d / "SHA256SUMS").write_text("%s  ctx130816_tokens.bin\n" % ("0" * 64))
    with pytest.raises(SystemExit) as e:
        PC.load_workload(str(d), 130816)
    assert "sha256" in str(e.value)


# ------------------------------------------------------------------ PP15
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


def _step(cell, holder="118", sha="0" * 40, dry=True, holders=None, n_args=None):
    e = _env()
    e.pop("P0_HOLDERS", None)
    if dry:
        e["DRYRUN"] = "1"
    else:
        e.pop("DRYRUN", None)
    if holders is not None:
        e["P0_HOLDERS"] = str(holders)
    argv = ["bash", str(EFF / "r2_base_profile_step.sh"), "--nosi", sha, holder, "t1", cell]
    if n_args:
        argv.append("extra")
    return subprocess.run(argv, env=e, capture_output=True, text=True, timeout=300)


def test_PP15_step_script(tmp_path):
    for bad in ("p0", "p0c64", "mb2c63", "p0allx", "fit128b216", "P0ALL", "p0k128"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    for h in ("0118", "abc", "1x"):
        r = _step("mb1", holder=h)
        assert r.returncode == 1 and "holder job id" in r.stderr, (h, r.stderr[-300:])
    r = _step("mb1", n_args=True)
    assert r.returncode == 2 and "usage" in r.stderr
    r = _step("mb1", dry=False, holders=tmp_path)
    assert r.returncode == 1 and "P0_HOLDERS is for DRYRUN" in r.stderr
    r = _step("p0all")                                   # passes the cell stage, stops on the sha pin
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    for h in ("117", "118"):
        (tmp_path / ("%s.gpu_uuid" % h)).write_text(UUID + "\n")
    out = "/mnt/beegfs/ojerbi/r2-runs/p0prof/p0prof_t1_118_"
    want = {"mb1": (128, 65, 216, 16128, 512, 35, "240G", "r2_base_cell.py"),
            "p0c128": (128, 65, 216, 16128, 512, 45, "240G", "r2_base_profile_cell.py"),
            "p0c63b344": (63, 0, 344, 16128, 512, 60, "340G", "r2_base_profile_cell.py"),
            "p0c63b216": (63, 0, 216, 16128, 512, 45, "240G", "r2_base_profile_cell.py"),
            "p0k128b63": (128, 65, 63, 130816, 512, 100, "440G", "r2_base_profile_cell.py"),
            "p0nsys": (128, 65, 216, 16128, 40, 30, "240G", "r2_base_profile_cell.py")}
    for cell, (C, P, B, S, steps, smin, mem, py) in want.items():
        r = _step(cell, sha=sha, holders=tmp_path)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = "cell %s: C %d (pool %d) B %d S %d steps %d STEP_MIN %d mem %s python %s" % (
            cell, C, P, B, S, steps, smin, mem, py)
        o = r.stdout
        assert line in o, (cell, o[-800:])
        assert "NOSI_POOL_BLOCKS=%d" % P in o and "NOSI_ATTN_SPLITS=4" in o and "--batch %d" % B in o
        assert "--mem=%s" % mem in o and "--time=%d" % smin in o and out + cell in o and "STEP_MIN total %d" % smin in o
        assert ("--stage-timers" in o) == (cell not in ("mb1", "p0nsys")), cell
        assert ("--baseline %smb1/summary.json" % out in o) == (cell != "mb1"), cell
        assert ("--mbase" in o) == (cell in ("mb1", "p0c128", "p0c63b344", "p0c63b216")), cell
        assert ("--context 130816" in o and "pg19-test-ctx130816" in o) == (cell == "p0k128b63"), cell
        assert ("NsightSystems" in o) == (cell == "p0nsys"), cell
    r = _step("mb1", sha=sha, holders=tmp_path)
    tail = ("--model /mnt/beegfs/ojerbi/models/NOSA-8B --workload /mnt/beegfs/ojerbi/kstep-cpu-runs/workload/"
            "pg19-test-c16k --batch 216 --steps 512 --warmup 2 --r2 fused --arms SH --pool 65 --mbase")
    assert tail in r.stdout and "r2_base_cell.py --out %smb1 " % out in r.stdout
    rec = MB1.parent / (MB1.name + ".step") / "command.txt"
    if rec.is_file():
        assert tail in rec.read_text(), "the mb1 python arguments are not those of the basef mb1 record"
    r = _step("p0nsys", sha=sha, holders=tmp_path)
    o = r.stdout
    assert "/usr/local/cuda-12.8/NsightSystems-cli-2024.6.2/target-linux-x64/nsys profile -t cuda\\,nvtx" in o
    assert "--capture-range=cudaProfilerApi --capture-range-end=stop -o %sp0nsys.step/nsys_p0nsys" % out in o
    assert "--steps 40 --pool 65 --nsys-window 8" in o
    for h, order in (("118", ["mb1", "p0c128", "p0c63b344", "p0c63b216", "p0k128b63"]),
                     ("117", ["mb1", "p0k128b63", "p0c63b216", "p0c63b344", "p0c128"])):
        r = _step("p0all", holder=h, sha=sha, holders=tmp_path)
        assert r.returncode == 0, r.stderr[-300:]
        got = re.findall(r"R2 p0prof step: cell (\w+): C ", r.stdout)
        assert got == order, (h, got)
        assert "STEP_MIN total 285" in r.stdout and "cells %s;" % " ".join(order) in r.stdout
