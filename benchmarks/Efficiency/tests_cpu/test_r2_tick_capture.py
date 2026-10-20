"""CPU tests C1-C7 of the FIXED-CAPACITY TICK and the sync-free events (nosi/nosi/r2_tick_fixed.py; docs/
R2_TICK_CAPTURE_PLAN.md, W0-W3). The world is the tick's own (test_r2_tick.py: the tiny NOSA-shaped model, the CPU
stand-ins, piece 3 v2's torch twin); the reference is the EAGER tick, r2_tick.TickEngine (SG0-SG4 on the A100).

  C1  TF == TE bitwise: the fixed tick and the eager tick, the same round (teacher-forced with the forced schedule;
      free-running; the phase mix; the serial mode): every tick's rows, live logits and argmax, and the digest of every
      twin table, every store cell outside the trash slot, the token log and the host mirrors after every tick.
  C2  padding invariance: cap_mode exact (no padding), family, max: equal. NEGATIVE CONTROL: with a GEMM whose
      rounding depends on the row count, exact and max differ (the comparison sees the row count).
  C3  the padded rows write only the trash cells: request 0's trash row 0, nothing else of the trash slot.
  C4  NEGATIVE CONTROLS with exact predicted first ticks (from the eager run's plan): NG1 the padded rows counted into
      bank 0; NG2 a stale row table at tick 5; NG3 the token-log destinations frozen per capacity.
  C5  the refusals (a tick above the capacity, C's twin capacity condition, the token log) before any write; a tick
      at the capacity is accepted.
  C6  E1: commit_unchecked == PositionalTwin.commit and rebuild_from_own_rows_unchecked == rebuild_from_own_rows,
      bitwise, with negative controls; commit_banks == rollback_bank in an engine; its host-mirror checks refuse a
      stale snapshot, a keep outside the snapshot, a snapshot ahead of the state, a rollback beyond the ring; the
      driver and E1 contain no host read, read_tokens exactly one.
  C7  the capture lint (r2_capture_lint.py, e5fc694's): the region (_forward_fixed, _layer_fixed) is clean; its
      callees show only documented findings; the validated eager functions show their classes; planted regressions in
      the region and the planted function are found.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py benchmarks/Efficiency/tests_cpu/test_r2_tick.py \
        benchmarks/Efficiency/tests_cpu/test_r2_twin_unchecked.py
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")          # the tick's CPU world
LINT = _load_by_path("r2_capture_lint", HERE / "r2_capture_lint.py")
FX = TT._load("r2_tick_fixed")
TK, SC, TW, BD = TT.TK, TT.SC, TT.TW, TT.BD
BS = TT.BS
p3 = TT.p3                                                             # pytest fixtures, re-exported
ref = TT.ref


# =====================================================================================================================
# recording and comparing runs
# =====================================================================================================================
TWIN_KEYS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis")


def digest(t):
    t = t.detach().cpu().contiguous().reshape(-1)
    return "%s %s %s" % (t.dtype, tuple(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest()[:32])


def state(eng):
    """Digests of every twin table and counter, every store cell outside the trash slot (slots 0 .. NB + 3), the
    written token log, and the host mirrors."""
    s = {}
    n = (eng.NB + 4) * BS
    for l, t in enumerate(eng.twins):
        for k in TWIN_KEYS:
            s["L%d.%s" % (l, k)] = digest(getattr(t, k))
        for name, st in (("k", eng.store_k[l]), ("v", eng.store_v[l]), ("bias", eng.store_bias[l])):
            s["L%d.store_%s" % (l, name)] = digest(st[:, :n])
    s["tok_log"] = digest(eng.tok_log[:eng.n_tok])
    s["host"] = repr((list(eng.bank_pos), list(eng.committed), list(eng.wb), list(eng.synced), eng.n_tok))
    return s


def record(eng, toks, forced=True, ready=None, steps=TT.STEPS, catch=False):
    """A round on eng (teacher-forced on toks with the forced schedule, or free-running greedy). Per tick (each
    run_rows): the rows, their input tokens, the live logits and argmax, and state(eng) after the tick's rows. catch:
    stop at the first exception (the negative controls may void their own run later; the first difference is
    earlier)."""
    ticks = []
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos) for r in rows],
                          tokens=[("ref", t.slot) if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=out.logits.clone(), argmax=out.argmax.clone(), state=state(eng)))
    reject_at = TK.forced_schedule(3)[0] if forced else None
    rec = dict(ticks=ticks, eng=eng, error=None)
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [steps] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick, ready=ready)
    except Exception as e:                                             # noqa: BLE001 (catch=True only)
        if not catch:
            raise
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        return rec
    rec.update(final=state(eng), out=[r.out for r in rnd.reqs], rnd=rnd)
    return rec


def first_diff(a, b):
    """(tick, what) of the first difference between two records, or None."""
    for t, (x, y) in enumerate(zip(a["ticks"], b["ticks"])):
        if x["rows"] != y["rows"]:
            return t, "rows"
        if x["tokens"] != y["tokens"]:
            return t, "tokens"
        if not torch.equal(x["logits"], y["logits"]):
            return t, "logits"
        if not torch.equal(x["argmax"], y["argmax"]):
            return t, "argmax"
        bad = sorted(k for k in x["state"] if x["state"][k] != y["state"].get(k))
        if bad:
            return t, "state %s" % bad[:6]
    n = min(len(a["ticks"]), len(b["ticks"]))
    if len(a["ticks"]) != len(b["ticks"]) or a.get("error") or b.get("error"):
        return n, "tick count %d / %d, errors %s / %s" % (len(a["ticks"]), len(b["ticks"]), a.get("error"),
                                                          b.get("error"))
    bad = sorted(k for k in a["final"] if a["final"][k] != b["final"].get(k))
    if bad:
        return n, "final state %s" % bad[:6]
    if a["out"] != b["out"]:
        return n, "committed tokens"
    return None


def eager(p3, ops=None, draft_drop=0.5):
    return TT.make_engine(ops or TT.make_ops(p3), draft_drop=draft_drop)


def fixed(p3, cap_mode="family", ops=None, cls=None, draft_drop=0.5, v_cap=3):
    base = TT.make_engine(ops or TT.make_ops(p3), draft_drop=draft_drop)
    return (cls or FX.FixedTickEngine).from_engine(base, v_cap=v_cap, cap_mode=cap_mode)


def counts(rows, B):
    c = [0] * (2 * B)
    for (b, kind, _) in rows:
        c[b if kind == "v" else B + b] += 1
    return c


@pytest.fixture(scope="module")
def te_forced(p3, ref):
    """The eager tick, teacher-forced with the forced schedule (the reference of C1, C3, C4)."""
    return record(eager(p3), ref[0], True)


# =====================================================================================================================
# C1 - C3
# =====================================================================================================================
@pytest.mark.parametrize("mode", ["forced", "free", "phase"])
def test_C1_fixed_tick_equals_the_eager_tick(p3, ref, te_forced, mode):
    """C1: TF == TE bitwise, every tick (module docstring). Preconditions: rollbacks, publications and write-backs
    happen; both capacities are used; some rows are padded; the decision reads are the eager tick's; no flag set."""
    toks = ref[0]
    ready = TK.phase_ready(3) if mode == "phase" else None
    te = te_forced if mode == "forced" else record(eager(p3), toks, mode != "free", ready)
    tf = record(fixed(p3), toks, mode != "free", ready)
    assert first_diff(te, tf) is None, first_diff(te, tf)
    eng = tf["eng"]
    eng.check_flags()
    st = eng.stats
    assert st["rollbacks"] > 0 and st["publications"] > 0 and st["writebacks"] >= 1, st
    assert (eng.B, 1) in st["caps"] and (eng.R_max, eng.T_max) in st["caps"] and st["padded_rows"] > 0, st["caps"]
    assert st["decision_reads"] == te["eng"].stats["decision_reads"] > 0
    assert len(tf["ticks"]) > 20


@pytest.mark.parametrize("cap_mode", ["family", "max"])
def test_C1s_serial_mode(p3, ref, cap_mode):
    """C1 (serial mode, SG0's path): run_serial on the fixed tick equals the eager serial mode and the reference rows,
    step by step, with the state; with cap_mode max every call pads 4 rows to 16."""
    toks, logits = ref
    te, tf = eager(p3), fixed(p3, cap_mode=cap_mode)
    for s in range(14):                                                # positions 500 .. 513: a write-back is read
        tok = [int(x) for x in toks[s]]
        a, b = te.run_serial(tok), tf.run_serial(tok)
        assert torch.equal(a.logits, b.logits) and torch.equal(b.logits, logits[s]), s
        assert state(te) == state(tf), s
    tf.check_flags()
    assert tf.stats["writebacks"] >= 1
    assert (tf.stats["padded_rows"] > 0) == (cap_mode == "max")


def test_C2_padding_invariance(p3, ref, te_forced):
    """C2: the same teacher-forced round with cap_mode exact (no padding), family and max is bitwise equal (and equal
    to the eager tick). NEGATIVE CONTROL: with the M-dependent GEMM stand-in, exact and max differ."""
    toks = ref[0]
    recs = {m: record(fixed(p3, cap_mode=m), toks, True) for m in ("exact", "family", "max")}
    assert first_diff(recs["exact"], recs["family"]) is None
    assert first_diff(recs["exact"], recs["max"]) is None
    assert first_diff(te_forced, recs["exact"]) is None
    pad = {m: r["eng"].stats["padded_rows"] for m, r in recs.items()}
    assert pad["exact"] == 0 < pad["family"] < pad["max"], pad
    e = record(fixed(p3, cap_mode="exact", ops=TT.make_ops(p3, TT.mdep_linear)), toks, True, catch=True)
    m = record(fixed(p3, cap_mode="max", ops=TT.make_ops(p3, TT.mdep_linear)), toks, True, catch=True)
    assert first_diff(e, m) is not None


def test_C3_padded_rows_write_only_the_trash_cells(p3, ref, te_forced):
    """C3: after a round with padding (cap_mode max), the trash slot NB + 4 holds the padded rows' K / V / bias in
    request 0's row 0 and nothing else; every cell outside it equals the eager tick's (C1's digests)."""
    rec = record(fixed(p3, cap_mode="max"), ref[0], True)
    assert first_diff(te_forced, rec) is None
    eng = rec["eng"]
    assert eng.stats["padded_rows"] > 0 and eng.trash == eng.NB + 4
    r0 = eng.trash * BS
    for l in range(eng.L):
        for st in (eng.store_k[l], eng.store_v[l], eng.store_bias[l]):
            assert st.shape[1] == (eng.NB + 5) * BS
            slot = st[:, r0:r0 + BS]
            assert bool((slot[0, 0] != 0).any()), "the padded rows wrote request 0's trash row 0"
            assert not bool((slot[0, 1:] != 0).any()) and not bool((slot[1:] != 0).any()), "only that row"


# =====================================================================================================================
# C4: the negative controls
# =====================================================================================================================
class PadLeak(FX.FixedTickEngine):
    """NG1: the padded rows are counted into bank 0 (request 0's exact bank advances by them)."""

    def padded_table(self, tab, count, R, cb, lo):
        host = super().padded_table(tab, count, R, cb, lo)
        host[len(FX.XFIELDS) * cb.R_cap] += cb.R_cap - R
        return host


class StaleTable(FX.FixedTickEngine):
    """NG2: tick STALE_AT runs on the table its capacity's buffer already holds (the copy is skipped)."""
    STALE_AT = 5

    def upload(self, cb, host):
        if self.stats["ticks"] == self.STALE_AT:
            return
        super().upload(cb, host)


class FrozenCursor(FX.FixedTickEngine):
    """NG3: the token-log destinations of a capacity stay those of its first tick (a host value frozen at capture)."""

    def padded_table(self, tab, count, R, cb, lo):
        host = super().padded_table(tab, count, R, cb, lo)
        NF = len(FX.XFIELDS)
        t = host[:NF * cb.R_cap].view(NF, cb.R_cap)
        row = FX.XFIELDS.index("logslot")
        frozen = self.__dict__.setdefault("_frozen", {})
        key = (cb.R_cap, cb.T_cap)
        if key not in frozen:
            frozen[key] = t[row].clone()
        t[row] = frozen[key]
        return host


def predicted_pad_leak(te, eng):
    """The first tick with padding (R < R_cap) whose bank 0 has room left (count[0] < T_cap)."""
    for t, tick in enumerate(te["ticks"]):
        c = counts(tick["rows"], eng.B)
        R_cap, T_cap = eng.choose_cap(len(tick["rows"]), max(c))
        if len(tick["rows"]) < R_cap and c[0] < T_cap:
            return t
    return None


def predicted_frozen_cursor(te, eng):
    """The first tick after which the token log under frozen destinations differs from the true one."""
    sim, true, frozen, lo = {}, {}, {}, 0
    for t, tick in enumerate(te["ticks"]):
        R = len(tick["rows"])
        cap = eng.choose_cap(R, max(counts(tick["rows"], eng.B)))
        if cap not in frozen:
            frozen[cap] = [lo + k for k in range(R)] + [eng.LOG - 1] * (cap[0] - R)
        args = tick["argmax"].tolist()
        for k in range(R):
            sim[frozen[cap][k]] = args[k]
            true[lo + k] = args[k]
        lo += R
        if any(sim.get(s, 0) != true[s] for s in range(lo)):
            return t
    return None


@pytest.mark.parametrize("ng", ["pad_leak", "stale_table", "frozen_cursor"])
def test_C4_negative_controls_first_differ_at_the_predicted_tick(p3, ref, te_forced, ng):
    """C4: each planted fault of a captured tick, simulated in the eager fixed tick, first differs from the eager tick
    at EXACTLY the tick its prediction names (computed from the eager run's plan before the faulty run)."""
    probe = fixed(p3)
    pred = dict(pad_leak=predicted_pad_leak, frozen_cursor=predicted_frozen_cursor,
                stale_table=lambda te, e: StaleTable.STALE_AT)[ng](te_forced, probe)
    assert pred is not None and pred < len(te_forced["ticks"])
    cls = dict(pad_leak=PadLeak, stale_table=StaleTable, frozen_cursor=FrozenCursor)[ng]
    rec = record(fixed(p3, cls=cls), ref[0], True, catch=True)
    fd = first_diff(te_forced, rec)
    assert fd is not None and fd[0] == pred, (ng, fd, pred)


# =====================================================================================================================
# C5: the refusals
# =====================================================================================================================
def test_C5_refusals_before_any_write(p3):
    """C5: a tick above the capacity (four verify rows in one bank, v_cap 3), C's twin capacity condition, and a tick
    that would reach the token log's trash slot are refused BEFORE any write (no buffer made, the state unchanged). A
    tick at the capacity (three verify rows) is accepted."""
    eng = fixed(p3)
    S0 = eng.S_list[0]
    snap = state(eng)
    four = [SimpleNamespace(b=0, kind="v", pos=S0 + i) for i in range(4)]
    with pytest.raises(TK.TickError, match="capacity"):
        eng.run_rows(four, [1, 2, 3, 4])
    with pytest.raises(TK.TickError, match="capacity"):
        eng.choose_cap(eng.R_max + 1, 1)
    big = [0] * (2 * eng.B)
    big[0] = eng.twins[0].n_max - eng.bank_pos[0] + 1
    with pytest.raises(TK.TickError, match="N_max"):
        eng.capacity_check(big)
    eng.n_tok = eng.LOG - 2
    with pytest.raises(TK.TickError, match="token log"):
        eng.run_rows([SimpleNamespace(b=0, kind="v", pos=S0)], [1])
    eng.n_tok = 0
    assert state(eng) == snap and eng.bufs == {} and eng.stats["ticks"] == 0
    out = eng.run_rows(four[:3], [1, 2, 3])                            # at the capacity: accepted
    assert out.cap == (eng.R_max, eng.T_max) and eng.stats["padded_rows"] == eng.R_max - 3
    assert state(eng) != snap


# =====================================================================================================================
# C6: E1
# =====================================================================================================================
def _twin_pair(B=4, H=2, Dh=8, S=200, seed=61):
    g = torch.Generator().manual_seed(seed)
    pk, pc = torch.randn(S, H, Dh, generator=g), torch.randn(S, H, generator=g)
    keys, cis = torch.randn(B, 12, H, Dh, generator=g), torch.randn(B, 12, H, generator=g)   # positions 200 .. 211

    def one():
        t = TW.PositionalTwin(B, H, Dh, 40, 400)
        t.mean_fn = BD.fixed_order_mean
        for b in range(B):
            t.prefill(b, pk, pc)
        return t
    return one, keys, cis


def _same_twin(a, f):
    return all(torch.equal(getattr(a, k), getattr(f, k)) for k in TWIN_KEYS)


def test_C6a_commit_unchecked_equals_commit():
    """C6a: commit_unchecked == PositionalTwin.commit (every table and counter) after a ragged advance from position
    200 (n = 12, 2, 0, 9: the cis compression due at 207 and the key compression due at 208 inside; keep 9 rolls back
    after both, keep 8 between them). NEGATIVE CONTROL: another keep differs."""
    one, keys, cis = _twin_pair()
    n, keep = [12, 2, 0, 9], [9, 1, 0, 8]
    a, f, g = one(), one(), one()
    ca, cf, cg = a.advance(keys, cis, n), f.advance(keys, cis, n), g.advance(keys, cis, n)
    assert int(a.n_comp[0]) == 12 and int(a.n_ccis[0]) == 12                # both compressions happened in bank 0
    a.commit(ca, keep)
    FX.commit_unchecked(f, cf, torch.tensor(keep))
    assert _same_twin(a, f)
    FX.commit_unchecked(g, cg, torch.tensor([9, 1, 0, 7]))
    assert not _same_twin(a, g)


def _t12_twin():
    """test_r2_tick.py T12's state: the draft bank ran ahead to 212, the verifier published 200 .. 204."""
    B, H, Dh, S = 2, 2, 8, 200
    g = torch.Generator().manual_seed(12)
    pk, pc = torch.randn(S, H, Dh, generator=g), torch.randn(S, H, generator=g)
    ek, ec = torch.randn(5, H, Dh, generator=g), torch.randn(5, H, generator=g)
    dk, dc = torch.randn(12, H, Dh, generator=g), torch.randn(12, H, generator=g)
    t = TW.PositionalTwin(2 * B, H, Dh, 40, 400)
    t.mean_fn = BD.fixed_order_mean
    for bank in range(2 * B):
        t.prefill(bank, pk, pc)
    SC.flatten_advance(t, {B + 0: (dk, dc)})
    SC.flatten_advance(t, {0: (ek, ec)})
    return t


def test_C6b_rebuild_unchecked_equals_rebuild():
    """C6b: rebuild_from_own_rows_unchecked == r2_tick.rebuild_from_own_rows (the captured rows and every table and
    counter; the replayed span holds a key and a cis compression). NEGATIVE CONTROL: a publication point one position
    later differs."""
    a, f, g = _t12_twin(), _t12_twin(), _t12_twin()
    ka, ca = TK.rebuild_from_own_rows(a, 0, 2, 205, 212)
    kf, cf = FX.rebuild_from_own_rows_unchecked(f, 0, 2, 205, 212)
    assert torch.equal(ka, kf) and torch.equal(ca, cf) and _same_twin(a, f)
    FX.rebuild_from_own_rows_unchecked(g, 0, 2, 206, 212)
    assert not _same_twin(a, g)
    with pytest.raises(TK.TickError):
        FX.rebuild_from_own_rows_unchecked(_t12_twin(), 0, 2, 100, 212)


def test_C6c_commit_banks_equals_rollback_and_refuses_on_its_host_mirrors(p3):
    """C6c: in an engine, commit_banks == SC.rollback_bank per layer (the eager tick's rollback), and its host-mirror
    checks refuse a stale snapshot (a second commit with the same counters), a keep outside the snapshot, a snapshot
    ahead of the state and a rollback beyond the key ring, each before any change of the twins."""
    S0 = TT.S_TINY[0]
    rows = [SimpleNamespace(b=0, kind="v", pos=S0), SimpleNamespace(b=0, kind="v", pos=S0 + 1)]
    te, tf = eager(p3), fixed(p3)
    te.run_rows(rows, [1, 2])
    tf.run_rows(rows, [1, 2])
    for l in range(te.L):
        SC.rollback_bank(te.twins[l], te.last_ctr[l], {0: 1})
    tf.commit_banks({0: 1})
    assert all(_same_twin(a, f) for a, f in zip(te.twins, tf.twins))

    def refused(eng, keep, match):
        before = [digest(getattr(t, k)) for t in eng.twins for k in TWIN_KEYS]
        with pytest.raises(TK.TickError, match=match):
            eng.commit_banks(keep)
        assert [digest(getattr(t, k)) for t in eng.twins for k in TWIN_KEYS] == before
    refused(tf, {0: 1}, "stale")
    for keep, tweak, match in (({0: 9}, 0, "outside"), ({0: 1}, -5, "ahead"), ({0: 0}, 40, "rollback over")):
        e = fixed(p3)
        e.run_rows(rows, [1, 2])
        e.bank_pos[0] += tweak
        refused(e, keep, match)


def _method_src(qual):
    src = (TT.NOSI / "r2_tick_fixed.py").read_text()
    code = ast.get_source_segment(src, LINT.fn_from_source(src, qual))
    return "\n".join(l.split("#")[0] for l in code.splitlines())


def test_C6d_driver_and_E1_have_no_host_read():
    """C6d: the driver, the region and E1 contain no host read (.item, .tolist, .cpu, .numpy, synchronize, bool);
    read_tokens is the one decision read (exactly one .tolist()). Negative control: the scan finds it there."""
    for qual in ("FixedTickEngine.run_rows", "FixedTickEngine._forward_fixed", "FixedTickEngine._layer_fixed",
                 "FixedTickEngine.apply_events", "FixedTickEngine.commit_banks", "FixedTickEngine.padded_table",
                 "FixedTickEngine.choose_cap", "FixedTickEngine.capacity_check", "FixedTickEngine.upload",
                 "FixedTickEngine.cap_buffers", "FixedTickEngine.forward", "commit_unchecked",
                 "rebuild_from_own_rows_unchecked"):
        code = _method_src(qual)
        for bad in (".item(", ".tolist(", ".cpu(", ".numpy(", "synchronize", "bool("):
            assert bad not in code, (qual, bad)
    assert _method_src("FixedTickEngine.read_tokens").count(".tolist()") == 1


# =====================================================================================================================
# C7: the capture lint
# =====================================================================================================================
def _sources(p3):
    return dict(fixed=(TT.NOSI / "r2_tick_fixed.py").read_text(), twin=(TT.NOSI / "r2_positional_twin.py").read_text(),
                core=(TT.NOSI / "r2_spec_core.py").read_text(), tick=(TT.NOSI / "r2_tick.py").read_text(),
                bid=(TT.NOSI / "r2_bi_decode.py").read_text(), p3=Path(p3.__file__).read_text())


REGION = [("fixed", "FixedTickEngine._forward_fixed"), ("fixed", "FixedTickEngine._layer_fixed")]

# the region's callees: their findings must all be DOCUMENTED (r2_capture_lint.py: transient allocations under the
# allocation rule; in-place augmented assignments on tensor attributes; Python-scalar conversions of non-tensor
# arguments). docs/R2_TICK_CAPTURE_PLAN.md Tables T-B / T-C.
CALLEES = {
    ("twin", "PositionalTwin.step_unchecked"): [("ALLOC", r"^torch\.arange\(self\.B"), ("ALLOC", r"\.clone\(\)$"),
                                                ("HOSTSTATE", r"^self\.(overflow \|=|n_comp \+=|n_ccis \+=|pos \+=)")],
    ("twin", "PositionalTwin.advance_unchecked"): [],
    ("twin", "PositionalTwin.key_window"): [],
    ("twin", "PositionalTwin._key_due"): [],
    ("twin", "PositionalTwin._cis_due"): [],
    ("twin", "PositionalTwin._mean"): [],
    ("twin", "PositionalTwin.counters"): [],
    ("twin", "PositionalTwin.comp_k_flat"): [],
    ("bid", "fixed_order_mean"): [],
    ("core", "select_blocks"): [("ALLOC", r"^torch\.zeros_like\(pc")],
    ("core", "stable_topk_desc"): [],
    ("core", "PaddedLinear.__call__"): [],
    ("tick", "TickEngine.page_table"): [("ALLOC", r"^torch\.(full|arange|full_like)\("),
                                        ("SYNC", r"^int\(round\(self\.draft_drop \* 1000\)\)$")],
    ("p3", "bi_paged_decode"): [("SYNC", r"^int\(pages_per_split\)$"), ("SYNC", r"^float\(softmax_scale\)$"),
                                ("ALLOC", r"^torch\.empty(_like)?\(")],
}

# the validated eager functions the region replaces: the lint must find AT LEAST these classes (Tables T-A / T-B)
EAGER = {
    ("tick", "TickEngine.run_rows"): {"ALLOC", "HOSTSTATE"},
    ("tick", "TickEngine._layer"): {"ALLOC", "SYNC", "BRANCH", "CALLS_EAGER"},
    ("twin", "PositionalTwin.advance"): {"ALLOC", "SYNC", "BRANCH", "CALLS_EAGER"},
    ("twin", "PositionalTwin.step"): {"ALLOC", "SYNC", "SHAPE", "BRANCH"},
    ("twin", "PositionalTwin.commit"): {"ALLOC", "SYNC", "BRANCH", "HOSTSTATE"},
}


@pytest.mark.parametrize("mod,qual", REGION, ids=[q for _, q in REGION])
def test_C7a_the_region_is_clean(p3, mod, qual):
    found = LINT.lint_function(LINT.fn_from_source(_sources(p3)[mod], qual))
    assert found == [], (qual, found)


@pytest.mark.parametrize("mod,qual", sorted(CALLEES), ids=[q for _, q in sorted(CALLEES)])
def test_C7b_callees_show_only_documented_findings(p3, mod, qual):
    found = LINT.lint_function(LINT.fn_from_source(_sources(p3)[mod], qual))
    assert LINT.unexplained(found, CALLEES[(mod, qual)]) == [], (qual, found)


@pytest.mark.parametrize("mod,qual", sorted(EAGER), ids=[q for _, q in sorted(EAGER)])
def test_C7c_eager_functions_show_their_classes(p3, mod, qual):
    classes = {c for c, _, _ in LINT.lint_function(LINT.fn_from_source(_sources(p3)[mod], qual))}
    assert EAGER[(mod, qual)] <= classes, (qual, EAGER[(mod, qual)], classes)


@pytest.mark.parametrize("line,cls", [("x = bool(live.any())", "SYNC"), ("b2 = live.nonzero()", "SHAPE"),
                                      ("w = torch.empty(3)", "ALLOC"), ("self.n_tok += 1", "HOSTSTATE"),
                                      ("y = pos[pos.max()]", "SCALARIDX"),
                                      ("ctr = twin.advance(cb.keys, cb.cis, cb.cnt)", "CALLS_EAGER"),
                                      ("assert live.any()", "BRANCH")])
def test_C7d_planted_regressions_in_the_region_are_found(p3, line, cls):
    """C7d: a planted violation inside _layer_fixed is found (the clean result of C7a is not vacuous); the planted
    function of e5fc694 shows every class."""
    fn = LINT.fn_from_source(_sources(p3)["fixed"], "FixedTickEngine._layer_fixed")
    fn.body.insert(1, ast.parse(line).body[0])
    assert cls in {c for c, _, _ in LINT.lint_function(fn)}
    assert {c for c, _, _ in LINT.lint_function(ast.parse(LINT.PLANTED).body[0])} == set(LINT.CLASSES)
