"""CPU tests of the tick's DEVICE TOKEN LOG size (nosi/nosi/r2_tick.py: TickEngine(log_slots=...), default LOG = 1 << 16).
The tiny NOSA-shaped model, the torch stand-ins and piece 3's fixture of test_r2_tick.py; no CUDA.

A run's log slots are never reused, so a long run passes slot 65536 after 65536 rows. The tests do not run 65536 rows
on the CPU: the engine's write position starts at `start` (the state of a run that already wrote `start` rows; no row
of the round references a slot below it), and the round's own ticks cross the boundary.

  LOG1  the default: the class constant is 1 << 16 and an engine built without log_slots has 65536 slots
  LOG2  the default REFUSES at the same row: with start = 65536 - cum[k] + 1 the k-th row tick is refused ("token
        log full") before it writes (n_tok stays at start + cum[k - 1]); every earlier tick ran
  LOG3  a SIZED log passes the same round past row 65536 and commits exactly the tokens of the reference round
        (start 0, default log): the log's offset changes no token
  LOG4  clone() keeps the sized log, clone(log_slots=...) replaces it; from_r2 forwards log_slots (its **kw); a size
        below 1 is refused
  LOG5  NEGATIVE CONTROL: an engine that IGNORES the argument (its LOG put back to the class value: the defect this
        change removes) does not refuse at the planted size, and the detector of LOG2 says so

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_tick_log.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r2_tick as T  # noqa: E402  (the tiny model, the CPU stand-ins)
from test_r2_tick import p3  # noqa: E402,F401  (piece 3's fixture)

TK = T.TK
STEPS = 24                      # committed tokens per request in each round (<= T.STEPS, the engine's widths)


def _engine(p3, **kw):
    return T.make_engine(T.make_ops(p3), **kw)


def _round(eng, start=0):
    """The free-running round of STEPS tokens per request from log position `start`. Returns (round, per-tick rows)."""
    eng.n_tok = int(start)
    rows = []

    def on_tick(rs, toks, out):
        rows.append(len(rs))
    rnd = TK.run_round(eng, T.FIRST, [STEPS] * eng.B, K=3, LAMBDA=7, on_tick=on_tick)
    return rnd, rows


def _refuses_at(eng, start, k, cum):
    """True when the round from `start` is refused at its k-th row tick, before it writes, and not before."""
    seen = []
    eng.n_tok = int(start)

    def on_tick(rs, toks, out):
        seen.append(len(rs))
    try:
        TK.run_round(eng, T.FIRST, [STEPS] * eng.B, K=3, LAMBDA=7, on_tick=on_tick)
    except Exception as e:                       # only the clean refusal counts; a crash past the log's end does not
        return (isinstance(e, TK.TickError) and "token log full" in str(e) and len(seen) == k - 1
                and eng.n_tok == start + cum[k - 1])
    return False


@pytest.fixture(scope="module")
def reference(p3):
    rnd, rows = _round(_engine(p3))
    cum = [0]
    for r in rows:
        cum.append(cum[-1] + r)
    return dict(out=[list(r.out) for r in rnd.reqs], rows=rows, cum=cum)


def test_LOG1_default_is_unchanged(p3):
    assert TK.TickEngine.LOG == 1 << 16
    eng = _engine(p3)
    assert eng.LOG == 1 << 16 and eng.tok_log.numel() == 1 << 16


def test_LOG2_default_refuses_at_the_same_row(p3, reference):
    cum, k = reference["cum"], 5
    assert len(reference["rows"]) > k and reference["rows"][k - 1] >= 1
    start = (1 << 16) - cum[k] + 1                       # the k-th tick's last row would be slot 65536
    assert _refuses_at(_engine(p3), start, k, cum)


def test_LOG3_a_sized_log_passes_the_same_round(p3, reference):
    cum, k = reference["cum"], 5
    start = (1 << 16) - cum[k] + 1
    eng = _engine(p3, log_slots=(1 << 16) + 4096)
    assert eng.LOG == (1 << 16) + 4096 and eng.tok_log.numel() == eng.LOG
    rnd, rows = _round(eng, start)
    assert rows == reference["rows"] and eng.n_tok == start + reference["cum"][-1] > 1 << 16
    assert [list(r.out) for r in rnd.reqs] == reference["out"]


def test_LOG4_clone_and_from_r2_carry_the_size(p3):
    eng = _engine(p3, log_slots=4096)
    assert eng.clone().LOG == 4096 and eng.clone(log_slots=8192).LOG == 8192
    src = inspect.getsource(TK.TickEngine.from_r2)
    assert "**kw):" in src and "return cls(model, ops, twins, sk, sv, sb, S_list, st.M_alloc, **kw)" in src
    for bad in (0, -1):
        with pytest.raises(TK.TickError, match="log_slots must be >= 1"):
            _engine(p3, log_slots=bad)


def test_LOG5_negative_control_an_ignored_size_is_detected(p3, reference):
    cum, k = reference["cum"], 3
    small = cum[k] - 1                                    # the k-th tick from start 0 would pass this size
    assert _refuses_at(_engine(p3, log_slots=small), 0, k, cum)        # the argument is honoured
    planted = _engine(p3, log_slots=small)
    planted.LOG = TK.TickEngine.LOG                       # the defect: the size argument ignored
    assert not _refuses_at(planted, 0, k, cum)
