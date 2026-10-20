"""CPU tests of W4, the CAPTURED fixed-capacity tick (nosi/nosi/r2_tick_capture.py) and the pure parts of its TC gate
(benchmarks/Efficiency/r2_tick_capture_gate.py). The CPU has no CUDA graph: FakeGraph stands in for one. Its replay()
re-runs the region with the engine's host values in FREEZE held at their CAPTURE-TIME values, as a captured graph
keeps them; its outputs are fixed tensors (the graph's static outputs). The world is the tick's own (test_r2_tick.py).

  G1  the captured engine (fake graphs) == the eager tick, bitwise every tick (forced schedule, free-running, phase
      mix, cap_mode max): the capture order (one eager tick per capacity, then one capture, then replays) and the
      counts (captures == graphs, replays + eager ticks == ticks).
  G2  PTR: no state data_ptr moves after the last capture (state_pointers / pointer_diff); NEGATIVE CONTROL: a
      rebound token log and a rebound twin counter are named.
  G3  a replay returns the SAME static output tensors at every tick of its capacity; warm and clone.
  G4  the gate's pure parts: counts_of, plan_caps, predicted_pad_leak, predicted_frozen_capture on hand plans; its
      arguments and refusals.
  G5  the gate's planted faults (its own fault_classes on the CPU engine) and its own compare_run: NG1 / NG2 / NG3
      first differ from the eager tick at EXACTLY the predicted tick; the fixed and the captured engines do not differ.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_tick_graph.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EFF = HERE.parent


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")
TC = _load_by_path("test_r2_tick_capture", HERE / "test_r2_tick_capture.py")
FX = TT._load("r2_tick_fixed")
CAPM = TT._load("r2_tick_capture")
TK = TT.TK
sys.path.insert(0, str(EFF))
GATE = _load_by_path("r2_tick_capture_gate_cpu", EFF / "r2_tick_capture_gate.py")
p3 = TT.p3                                                             # pytest fixtures, re-exported
ref = TT.ref
te_forced = TC.te_forced

FREEZE = ("_lo", "_R")


class FakeGraph:
    """CPU stand-in of a captured region (module docstring)."""

    def __init__(self, eng, cb):
        self.eng, self.cb = eng, cb
        self.frozen = {k: getattr(eng, k) for k in FREEZE if hasattr(eng, k)}
        V = eng.model.lm_head.shape[0]
        self.logits = torch.zeros((cb.R_cap, V), dtype=torch.float32)
        self.arg = torch.zeros(cb.R_cap, dtype=torch.int64)
        self.replays = 0

    def replay(self):
        saved = {k: getattr(self.eng, k) for k in self.frozen}
        for k, v in self.frozen.items():
            setattr(self.eng, k, v)
        try:
            lg, a = self.eng._forward_fixed(self.cb)
        finally:
            for k, v in saved.items():
                setattr(self.eng, k, v)
        self.logits.copy_(lg)
        self.arg.copy_(a)
        self.replays += 1


def fake_factory(eng, cb):
    g = FakeGraph(eng, cb)
    return g, g.logits, g.arg


def captured(p3, cls=None, cap_mode="family", warm=1, draft_drop=0.5):
    base = TT.make_engine(TT.make_ops(p3), draft_drop=draft_drop)
    return (cls or CAPM.CapturedTickEngine).from_engine(base, v_cap=3, cap_mode=cap_mode, warm=warm,
                                                        graph_factory=fake_factory)


def _uses(eng):
    return {k: v for k, v in eng.stats["caps"].items()}


# =====================================================================================================================
# G1 - G3
# =====================================================================================================================
@pytest.mark.parametrize("mode", ["forced", "free", "phase", "max"])
def test_G1_captured_tick_equals_the_eager_tick(p3, ref, te_forced, mode):
    toks = ref[0]
    ready = TK.phase_ready(3) if mode == "phase" else None
    forced = mode != "free"
    te = te_forced if mode in ("forced", "max") else TC.record(TC.eager(p3), toks, forced, ready)
    rec = TC.record(captured(p3, cap_mode="max" if mode == "max" else "family"), toks, forced, ready)
    assert TC.first_diff(te, rec) is None, TC.first_diff(te, rec)
    eng = rec["eng"]
    eng.check_flags()
    st, uses = eng.stats, _uses(eng)
    assert st["captures"] == len(eng.graphs) == sum(1 for n in uses.values() if n >= 2), (st, uses)
    assert st["eager_region"] == sum(min(n, 1) for n in uses.values()), (st, uses)
    assert st["replays"] + st["eager_region"] == st["ticks"] and st["replays"] > 0
    assert all(g["graph"].replays == uses[k] - 1 for k, g in eng.graphs.items())


def test_G2_no_state_pointer_moves_after_the_last_capture(p3, ref):
    eng = captured(p3)
    snap = {}

    def post(e, cb, rows):
        if e.stats["captures"] != snap.get("n"):
            snap["n"] = e.stats["captures"]
            snap["ptr"] = CAPM.state_pointers(e)
    eng.post_region = post
    TC.record(eng, ref[0], True)
    now = CAPM.state_pointers(eng)
    assert snap["n"] == 2 and [k for k, v in snap["ptr"].items() if now.get(k) != v] == []
    assert len(snap["ptr"]) > 40
    eng.tok_log = eng.tok_log.clone()                                  # NEGATIVE CONTROLS: planted rebindings
    eng.twins[1].pos = eng.twins[1].pos.clone()
    assert CAPM.pointer_diff(now, CAPM.state_pointers(eng)) == ["tok_log", "twin1.pos"]


def test_G3_static_outputs_warm_and_clone(p3, ref):
    eng = captured(p3)
    S0 = eng.S_list
    outs = [eng.run_serial([int(x) for x in ref[0][s]]) for s in range(4)]
    g = eng.graphs[(eng.B, 1)]
    assert eng.stats["eager_region"] == 1 and eng.stats["captures"] == 1 and g["graph"].replays == 3
    for o in outs[1:]:                                                 # views of the same static tensors
        assert o.logits.data_ptr() == g["logits"].data_ptr() and o.argmax.data_ptr() == g["arg"].data_ptr()
    assert outs[0].logits.data_ptr() != g["logits"].data_ptr()        # the warm tick was eager
    assert torch.equal(outs[3].logits, ref[1][3])
    with pytest.raises(TK.TickError):
        CAPM.CapturedTickEngine.from_engine(TT.make_engine(TT.make_ops(p3)), warm=0)
    fresh = captured(p3, warm=2)
    c = fresh.clone()
    assert c.warm == 2 and c.graph_factory is fake_factory and c.v_cap == 3 and type(c) is CAPM.CapturedTickEngine
    assert S0 == fresh.S_list


# =====================================================================================================================
# G4: the gate's pure parts
# =====================================================================================================================
def test_G4_gate_pure_parts(p3):
    B = 4
    assert GATE.counts_of([(0, "v", 9), (0, "v", 10), (2, "d", 5)], B) == [2, 0, 0, 0, 0, 0, 1, 0]
    probe = FX.FixedTickEngine.from_engine(TT.make_engine(TT.make_ops(p3)), v_cap=3)
    d4 = [(b, "d", 500) for b in range(4)]
    plan = [d4, d4[:3], [(0, "v", 1), (0, "v", 2)] + d4, d4, d4, [(1, "v", 3)] + d4[:2], d4]
    caps = GATE.plan_caps(plan, probe)
    # tick 5: one verify row, 3 rows <= B, one row per bank: it fits G_d
    assert caps == [(4, 1), (4, 1), (16, 3), (4, 1), (4, 1), (4, 1), (4, 1)]
    assert GATE.predicted_pad_leak(plan, caps, B) == 1                 # tick 1: 3 rows < 4, bank 0 empty < 1
    two = [(0, "v", 1), (0, "v", 2)] + d4[1:]                          # bank 0 holds 2 < T_cap 3: the leak shows
    assert GATE.predicted_pad_leak([d4, two], [(4, 1), (16, 3)], B) == 1
    full = [(0, "v", 1), (0, "v", 2), (0, "v", 3)] + d4[1:]            # bank 0 holds T_cap rows: a leak changes nothing
    assert GATE.predicted_pad_leak([d4, full], [(4, 1), (16, 3)], B) is None
    assert GATE.predicted_pad_leak([d4], [(4, 1)], B) is None          # no padding: never
    assert GATE.predicted_frozen_capture(caps, 1) == 3                 # (4, 1): eager 0, capture 1, frozen replay 3
    assert GATE.predicted_frozen_capture(caps, 2) == 4
    assert GATE.predicted_frozen_capture([(4, 1), (16, 3)], 1) is None
    base = ["--out", "x", "--workload", "w", "--biattn", "p"]
    a = GATE.parse_args(base)
    assert a.m_fix == 32 and a.warm == 1 and a.stale_at == 10 and a.parts == list(GATE.PARTS)
    for bad in (["--m-fix", "16"], ["--parts", "XX"], ["--warm", "0"], ["--deltas", "0,1"], ["--ts-steps", "200"]):
        with pytest.raises(SystemExit):
            GATE.parse_args(base + bad)


# =====================================================================================================================
# G5: the gate's planted faults and its comparison, on the CPU
# =====================================================================================================================
@pytest.fixture()
def nosi_alias(monkeypatch):
    """The gate imports nosi.r2_tick inside its functions: point those names at the CPU-loaded modules."""
    monkeypatch.setitem(sys.modules, "nosi", types.ModuleType("nosi"))
    monkeypatch.setitem(sys.modules, "nosi.r2_tick", TK)
    monkeypatch.setitem(sys.modules, "nosi.r2_tick_fixed", FX)
    monkeypatch.setitem(sys.modules, "nosi.r2_tick_capture", CAPM)


def _args():
    return GATE.parse_args(["--out", "x", "--workload", "w", "--biattn", "p", "--batch", "4", "--deltas", "0,0,0,0",
                            "--steps", str(TT.STEPS), "--S", "16128"])


@pytest.mark.parametrize("ng", ["NG1", "NG2", "NG3"])
def test_G5_planted_faults_first_differ_at_the_predicted_tick(p3, ref, nosi_alias, ng):
    a = _args()
    toks = ref[0]
    te = GATE.record_te(TC.eager(p3), a, toks)
    assert len(te["ticks"]) > 20
    plan = [x["rows"] for x in te["ticks"]]
    probe = FX.FixedTickEngine.from_engine(TT.make_engine(TT.make_ops(p3)), v_cap=3)
    caps = GATE.plan_caps(plan, probe)
    pred = dict(NG1=GATE.predicted_pad_leak(plan, caps, probe.B), NG2=a.stale_at,
                NG3=GATE.predicted_frozen_capture(caps, a.warm))[ng]
    assert pred is not None and pred < len(plan)
    cls = GATE.fault_classes(CAPM.CapturedTickEngine, FX.XFIELDS)[ng]
    eng = captured(p3, cls=cls)
    if ng == "NG2":
        eng.stale_at = a.stale_at
    r = GATE.compare_run(eng, a, toks, te)
    assert r["first"] is not None and r["first"][0] == pred, (ng, r["first"], pred, r["error"])


def test_G5b_the_gate_comparison_passes_the_fixed_and_the_captured_tick(p3, ref, nosi_alias):
    a = _args()
    toks = ref[0]
    te = GATE.record_te(TC.eager(p3), a, toks)
    for eng in (TC.fixed(p3), captured(p3), captured(p3, cap_mode="max")):
        r = GATE.compare_run(eng, a, toks, te)
        assert r["first"] is None and r["error"] is None, (type(eng).__name__, r["first"], r["error"])
        assert GATE.final_diff(eng, te["eng"]) == []
