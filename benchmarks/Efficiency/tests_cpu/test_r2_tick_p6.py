"""CPU tests of p6 (docs/R2_SG5_P6_PLAN.md): the BATCHED REBUILD on the eager SG5 tick (TickEngine(rebuild="batched"),
r2_tick.rebuild_banks_unchecked / _publish_batched / rows_to_copy / copy_rows_many) and the harness's --variants.
The world is the SG5 tick's (test_r2_tick.py: the tiny model, piece 3 v2, make_engine / make_res_engine).

  P6a  rebuild_banks_unchecked == SG5's CHECKED per-call rebuild_from_own_rows, per bank, bitwise: every table and
       counter of every bank and the captured rows (a replay across a key and a cis compression; a bank with nothing
       to replay). NEGATIVE CONTROL: a planted bank-order swap differs. Refusals: two items of one bank, a replay longer
       than the ring.
  P6b  the SG5-shaped rounds with rebuild="batched" == rebuild="per_call", bitwise every tick (rows, live logits,
       argmax, twin counters) and at the end (every twin table, the stores, the token log, the committed tokens):
       teacher-forced on the ResidentStore and on v1 / sb / sc / b2, free-running on b2. Precondition: a tick published
       two or more requests. NEGATIVE CONTROL: ONE swap planted in the engine's batched call, at the first tick whose
       two publications' replay rows differ (precondition, L34), changes the twin tables (planted != clean on copies)
       and is caught, with the twin tables recorded every tick, first at or after the tick after the plant.
  P6c  (the Triton interpreter, a child process) the free-running round on b2 with twin "fused" and rebuild "batched" ==
       twin "torch" and rebuild "per_call", bitwise (rows, argmax, twin counters every tick, committed tokens, decision
       counts, tables); PLANT 2 in the fused path effective and caught.
  P6d  the harness: parse_variants (the default, the lists, the refusals), the labels, make_engine's rebuild, the main
       loop over the variants; the engine's option at construction, clone and refusal.
  P6e  the lint (r2_capture_lint.py, unchanged from the capture line): rebuild_banks_unchecked, _publish_batched,
       rows_to_copy and copy_rows_many have no host read, no data-dependent shape (nonzero / mask indexing) and no host
       branch on a device value; the per-call rebuild shows its host reads. A planted read is found.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_tick_p6.py
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

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


TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")
LINT = _load_by_path("r2_capture_lint", HERE / "r2_capture_lint.py")
TK, SC, TW, BD = TT.TK, TT.SC, TT.TW, TT.BD
p3 = TT.p3
ref = TT.ref
TWIN_KEYS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis")


def _same_twin(a, f):
    return all(torch.equal(getattr(a, k), getattr(f, k)) for k in TWIN_KEYS)


# =====================================================================================================================
# P6a
# =====================================================================================================================
def _multi_twin(B=3, H=2, Dh=8, S=200, seed=21):
    """Three request pairs from position 200: bank 0's draft ran ahead to 212 and its verifier published 200 .. 204
    (rows 205 .. 211 replayed: the cis compression due at 207 and the key compression due at 208 inside); bank 1's
    draft and verifier are both at 207 (nothing to replay); bank 2's draft ran ahead to 203, nothing published."""
    g = torch.Generator().manual_seed(seed)
    pk, pc = torch.randn(S, H, Dh, generator=g), torch.randn(S, H, generator=g)
    t = TW.PositionalTwin(2 * B, H, Dh, 40, 400)
    t.mean_fn = BD.fixed_order_mean
    for bank in range(2 * B):
        t.prefill(bank, pk, pc)
    for bank, n in ((B + 0, 12), (0, 5), (B + 1, 7), (1, 7), (B + 2, 3)):
        SC.flatten_advance(t, {bank: (torch.randn(n, H, Dh, generator=g), torch.randn(n, H, generator=g))})
    return t, [(0, 205, 212), (1, 207, 207), (2, 200, 203)], B


def test_P6a_batched_rebuild_equals_the_checked_per_call_path():
    a, items, B = _multi_twin()
    f, _, _ = _multi_twin()
    s, _, _ = _multi_twin()
    got_a = {b: TK.rebuild_from_own_rows(a, b, B, p, d) for (b, p, d) in items}       # SG5's CHECKED per-call path
    got_f = TK.rebuild_banks_unchecked(f, items, B)
    assert int(a.n_comp[B]) > int(a.n_comp[0]) and int(a.n_ccis[B]) > int(a.n_ccis[0])   # the replay compressed
    assert _same_twin(a, f)
    assert sorted(got_a) == sorted(got_f) and all(torch.equal(got_a[b][0], got_f[b][0])
                                                  and torch.equal(got_a[b][1], got_f[b][1]) for b in got_a)
    TK.rebuild_banks_unchecked(s, items, B, swap=True)                                     # NEGATIVE CONTROL
    assert not _same_twin(a, s)
    with pytest.raises(TK.TickError, match="one publication per bank"):
        TK.rebuild_banks_unchecked(_multi_twin()[0], items + [(0, 205, 212)], B)
    with pytest.raises(TK.TickError, match="unverified rows"):
        TK.rebuild_banks_unchecked(_multi_twin()[0], [(0, 100, 212)], B)
    assert TK.rebuild_banks_unchecked(_multi_twin()[0], [], B) == {}


# =====================================================================================================================
# P6b
# =====================================================================================================================
def _copy_twin(t):
    """An independent copy of a twin's tables and counters (TickEngine.clone's copy), same mean_fn."""
    c = TW.PositionalTwin(t.B, t.H, t.D, t.lk_max, t.n_max, dtype=t.ring.dtype, device=t.pos.device)
    for k in TWIN_KEYS + ("overflow",):
        if hasattr(t, k) and hasattr(c, k):
            getattr(c, k).copy_(getattr(t, k))
    c.mean_fn = t.mean_fn
    return c


def _replay_rows(twin, B, b, p, d):
    """Draft bank B + b's own unverified rows [p, d) (rebuild_banks_unchecked's gather, before the sync)."""
    idx = torch.arange(int(p), int(d), dtype=torch.int64, device=twin.pos.device)
    return twin.ring[B + b, idx % TK.BLOCK], twin.total_cis[B + b, idx]


def _first_rows(x, n):
    """The first n rows of x, zero-padded (rebuild_banks_unchecked pads every item's rows to T with zeros)."""
    if x.shape[0] >= n:
        return x[:n]
    return torch.cat([x, torch.zeros((n - x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)])


def _swap_changes(twin, B, it0, it1):
    """PRECONDITION (L34): the swap of items it0 = (b0, p0, d0) and it1 feeds bank B + b0 the first n0 rows of it1's
    (zero-padded) rows and bank B + b1 the first n1 of it0's (the counts stay each bank's own: cnt is the unswapped hn).
    True iff either bank then gets a key or cis row that is not its own."""
    (b0, p0, d0), (b1, p1, d1) = it0, it1
    n0, n1 = int(d0) - int(p0), int(d1) - int(p1)
    k0, c0 = _replay_rows(twin, B, b0, p0, d0)
    k1, c1 = _replay_rows(twin, B, b1, p1, d1)
    same = (torch.equal(_first_rows(k1, n0), k0) and torch.equal(_first_rows(c1, n0), c0)
            and torch.equal(_first_rows(k0, n1), k1) and torch.equal(_first_rows(c0, n1), c1))
    return not same


class SwapEngine(TK.TickEngine):
    """NEGATIVE CONTROL: ONE batched call with two banks' replay rows exchanged (rebuild_banks_unchecked swap=True), at
    the first call with two publications whose exchange changes a bank's rows (precondition, L34; the pair goes first
    in pub, since the swap exchanges the first two items; the banks are independent, so the order alone changes
    nothing: P6a). The plant is checked against a clean rebuild of copies of the twins (planted != clean).
    seen: (tick, [(bank, replay rows)], the exchange of pub's FIRST two items changes rows) of every call with two or
    more publications up to the plant (the diagnosis of the old always-on control's no-op)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.plant = None
        self.seen = []

    def _publish_batched(self, pub, swap=False):
        if self.plant is not None or len(pub) < 2:
            return super()._publish_batched(pub)
        B, tick = self.B, self.stats["ticks"] - 1
        items = [(b, p, self.bank_pos[B + b]) for b, p in pub.items()]

        def changes(i, j):
            return any(_swap_changes(t, B, items[i], items[j]) for t in self.twins)
        self.seen.append((tick, [(b, d - p) for (b, p, d) in items], changes(0, 1)))
        pair = next(((i, j) for i in range(len(items)) for j in range(i + 1, len(items)) if changes(i, j)), None)
        if pair is None:
            return super()._publish_batched(pub)
        order = [pair[0], pair[1]] + [k for k in range(len(items)) if k not in pair]
        clean = [_copy_twin(t) for t in self.twins]                 # BEFORE the plant
        super()._publish_batched({items[k][0]: items[k][1] for k in order}, swap=True)
        for c in clean:
            TK.rebuild_banks_unchecked(c, [items[k] for k in order], B)
        self.plant = dict(tick=tick, items=[items[k] for k in order[:2]],
                          effective=not all(_same_twin(c, t) for c, t in zip(clean, self.twins)))


def _probe(eng):
    """Wrap the engine's batched call: (tick, number of publications) of every call."""
    calls = []
    orig = eng._publish_batched

    def wrapped(pub, swap=False):
        calls.append((eng.stats["ticks"] - 1, len(pub)))
        return orig(pub, swap=swap) if swap else orig(pub)
    eng._publish_batched = wrapped
    return calls


def _record(eng, toks, forced=True, catch=False, tables=False):
    S_list = eng.S_list
    ticks = []

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos) for r in rows], logits=out.logits.clone(), arg=out.argmax.clone(),
                          ctr=torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])))
        if tables:                                                     # the negative control: every twin table
            ticks[-1]["tab"] = [getattr(t, k).clone() for t in eng.twins for k in TWIN_KEYS]
    reject_at = TK.forced_schedule(3)[0] if forced else None
    rec = dict(ticks=ticks, error=None, eng=eng)
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                             # noqa: BLE001 (catch=True only)
        if not catch:
            raise
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        return rec
    n = (eng.NB + 4) * TT.BS
    rec["out"] = [list(r.out) for r in rnd.reqs]
    rec["final"] = ([getattr(t, k).clone() for t in eng.twins for k in TWIN_KEYS]
                    + [x[:, :n].clone() for x in eng.store_k + eng.store_v + eng.store_bias]
                    + [eng.tok_log[:eng.n_tok].clone()])
    return rec


def _first_diff(a, b):
    for t, (x, y) in enumerate(zip(a["ticks"], b["ticks"])):
        if x["rows"] != y["rows"]:
            return t, "rows"
        for k in ("logits", "arg", "ctr"):
            if not torch.equal(x[k], y[k]):
                return t, k
        if "tab" in x and "tab" in y and not all(torch.equal(u, v) for u, v in zip(x["tab"], y["tab"])):
            return t, "twin tables"
    n = min(len(a["ticks"]), len(b["ticks"]))
    if len(a["ticks"]) != len(b["ticks"]) or a["error"] or b["error"]:
        return n, "tick count %d / %d, errors %s / %s" % (len(a["ticks"]), len(b["ticks"]), a["error"], b["error"])
    if a["out"] != b["out"]:
        return n, "committed tokens"
    for i, (x, y) in enumerate(zip(a["final"], b["final"])):
        if not torch.equal(x, y):
            return n, "final %d" % i
    return None


def _engine(p3, kind, **kw):
    ops = TT.make_ops(p3)
    if kind == "resident":
        return TT.make_engine(ops, draft_drop=0.5, **kw)
    return TT.make_res_engine(ops, kind=kind, **kw)


@pytest.mark.parametrize("kind,forced", [("resident", True), ("v1", True), ("sb", True), ("sc", True), ("b2", True),
                                         ("b2", False)])
def test_P6b_batched_rounds_equal_per_call_rounds(p3, ref, kind, forced):
    toks = ref[0]
    per = _record(_engine(p3, kind), toks, forced)
    eng = _engine(p3, kind, rebuild="batched")
    calls = _probe(eng)
    bat = _record(eng, toks, forced)
    assert _first_diff(per, bat) is None, _first_diff(per, bat)
    assert per["eng"].rebuild == "per_call" and eng.rebuild == "batched"
    assert calls and max(n for _, n in calls) >= 2, calls                      # a tick published two or more
    assert eng.stats["publications"] == per["eng"].stats["publications"] > 0
    assert eng.stats["rollbacks"] == per["eng"].stats["rollbacks"] and not eng.mirror_failures()
    if kind != "resident" or not forced:
        return
    # NEGATIVE CONTROL (L34): the old always-on swap of pub's first two items was a no-op on this data (job 2191107)
    per_t = _record(_engine(p3, kind), toks, forced, tables=True)                 # the clean run, tables every tick
    assert _first_diff(per, per_t) is None, _first_diff(per, per_t)
    bad = TT.make_engine(TT.make_ops(p3), draft_drop=0.5, cls=SwapEngine, rebuild="batched")
    rb = _record(bad, toks, forced, catch=True, tables=True)
    print("P6b control: calls with >= 2 publications (tick, [(bank, replay rows)], pub's first two exchange rows): "
          "%s; plant %s" % (bad.seen, bad.plant))
    assert bad.plant is not None, ("precondition: no tick published two banks with different replay rows", bad.seen)
    assert bad.plant["effective"], ("precondition: the plant left every twin table as the clean rebuild", bad.plant)
    t0 = bad.plant["tick"]
    fd = _first_diff(per_t, rb)
    assert fd is not None and fd[0] >= t0 + 1, (fd, t0, bad.plant)              # caught, never before the plant


# =====================================================================================================================
# P6c: the fused twin with the batched rebuild (the Triton interpreter, a child process)
# =====================================================================================================================
P6C_CHILD = r'''
import functools, importlib.util, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton was imported before the interpreter flag was read"
import torch
HERE, P3PATH = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("test_r2_tick", os.path.join(HERE, "test_r2_tick.py"))
TT = importlib.util.module_from_spec(spec)
sys.modules["test_r2_tick"] = TT
spec.loader.exec_module(TT)
spec = importlib.util.spec_from_file_location("piece3_v2_p6c", P3PATH)
P3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P3)
TK = TT.TK
FU = TT._load("r2_twin_fused")
assert FU.triton is not None, "P6c child: triton did not import"
ops = TT.make_ops(P3)
FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis", "overflow")


def bits(t):
    t = t.contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def run(twin, rebuild, plant=0):
    """The free-running greedy round on b2 (C 12 / X 24 / R_ver 4, the adversarial transport), B 4."""
    eng = TT.make_res_engine(ops, kind="b2", C=12, X=24, R_ver=4, twin=twin, rebuild=rebuild)
    assert eng.twin_variant == twin and eng.rebuild == rebuild
    if plant:
        for t in eng.twins:
            t.advance_unchecked = functools.partial(FU.advance_fused, t, plant=plant)
    rec = []

    def on_tick(rows, tokens, out):
        ctr = torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])
        rec.append(([(r.b, r.kind, r.pos) for r in rows], out.argmax.clone(), ctr.clone()))
    rnd = TK.run_round(eng, TT.FIRST, [TT.STEPS] * eng.B, K=3, LAMBDA=7, on_tick=on_tick)
    eng.check_overflow()
    tables = [{k: bits(getattr(t, k)).clone() for k in FIELDS} for t in eng.twins]
    st = {k: eng.stats[k] for k in ("ticks", "rows", "decision_reads", "rollbacks", "publications", "void_ticks")}
    return dict(rec=rec, out=[list(r.out) for r in rnd.reqs], tables=tables, stats=st, mirror=eng.mirror_failures())


def first_diff(x, y):
    if len(x["rec"]) != len(y["rec"]):
        return ("ticks", len(x["rec"]), len(y["rec"]))
    for i, (a, b) in enumerate(zip(x["rec"], y["rec"])):
        if a[0] != b[0]:
            return (i, "rows")
        if not torch.equal(a[1], b[1]):
            return (i, "argmax")
        if not torch.equal(a[2], b[2]):
            return (i, "twin counters")
    if x["out"] != y["out"]:
        return ("end", "committed tokens")
    if x["stats"] != y["stats"]:
        return ("end", "decisions %s / %s" % (x["stats"], y["stats"]))
    for l, (a, b) in enumerate(zip(x["tables"], y["tables"])):
        for k in FIELDS:
            if not torch.equal(a[k], b[k]):
                return ("end", "layer %d %s" % (l, k))
    return None


rt = run("torch", "per_call")
rf = run("fused", "batched")
assert rt["mirror"] == [] and rf["mirror"] == [], (rt["mirror"], rf["mirror"])
assert rt["stats"]["ticks"] > 20 and rt["stats"]["publications"] > 0, rt["stats"]
d = first_diff(rt, rf)
assert d is None, "P6c child: fused / batched differs from torch / per_call: %s" % (d,)
try:
    rp = run("fused", "batched", plant=2)
    eff, got = first_diff(rf, rp), first_diff(rt, rp)
except Exception as e:                                                # the fault may void its own round: also a catch
    eff = got = ("raised", "%s: %s" % (type(e).__name__, str(e)[:120]))
assert eff is not None and got is not None, ("P6c child: PLANT 2 not effective / caught", eff, got)
print("P6c OK b2, free-running (%d ticks, %d publications, %d rollbacks): fused / batched == torch / per_call bitwise; "
      "PLANT 2 effective at %s, caught at %s" % (rt["stats"]["ticks"], rt["stats"]["publications"],
                                                 rt["stats"]["rollbacks"], eff, got))
'''


def test_P6c_fused_batched_equals_torch_per_call_on_b2(p3):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", P6C_CHILD, str(HERE), p3.__file__], env=env, cwd=str(HERE.parents[2]),
                       capture_output=True, text=True, timeout=3000)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith("P6c OK"), (
        "P6c child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:],
                                                                      r.stderr[-6000:]))


# =====================================================================================================================
# P6d: the harness and the engine's option
# =====================================================================================================================
def test_P6d_the_harness_variants_and_the_engine_option(p3):
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base)
    assert a.variants == "" and a.variant_list == [("torch", "per_call")] and a.rebuild == "per_call"
    assert G.parse_args(base + ["--twin", "fused"]).variant_list == [("fused", "per_call")]
    full = "torch/per_call,fused/per_call,torch/batched,fused/batched"
    b = G.parse_args(base + ["--variants", full])
    assert b.variant_list == [("torch", "per_call"), ("fused", "per_call"), ("torch", "batched"), ("fused", "batched")]
    assert b.rebuild == "per_call"
    for bad in ("torch", "torch/lazy", "triton/per_call", "torch/per_call,torch/per_call", "torch/per_call,",
                "torch/per_call/x"):
        with pytest.raises(SystemExit):
            G.parse_args(base + ["--variants", bad])
    assert G.variant_label(a, "exact", "b2", ("torch", "per_call")) == "SG5_exact_b2"          # p3's labels
    assert G.variant_label(b, "exact", "b2", ("fused", "batched")) == "SG5_exact_b2_fused_batched"
    src = (TT.EFF / "r2_tick_sg5.py").read_text()
    assert "TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin, **ekw)" in src   # [twin-variant]
    #                                                      pin ([integ2]: bigb 91e2fef added **ekw, the token-log size)
    assert 'eng.rebuild = getattr(a, "rebuild", "per_call")' in src
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body
                                            if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert main.index("for kind in backends:") < main.index("for variant in a.variant_list:")
    assert "a.twin, a.rebuild = variant" in main and 'pt["twin"], pt["rebuild"] = variant' in main
    ops = TT.make_ops(p3)                                                  # the engine's option
    e = TT.make_engine(ops)
    assert e.rebuild == "per_call" and e.clone().rebuild == "per_call"
    f = TT.make_engine(ops, rebuild="batched")
    assert f.rebuild == "batched" and f.clone().rebuild == "batched" and f.clone(rebuild="per_call").rebuild == "per_call"
    with pytest.raises(TK.TickError, match="rebuild must be one of"):
        TT.make_engine(ops, rebuild="lazy")
    assert TK.REBUILD_MODES == ("per_call", "batched")
    tick_src = (TT.NOSI / "r2_tick.py").read_text()
    assert tick_src.count("[batched-rebuild]") >= 12


# =====================================================================================================================
# P6e: the lint
# =====================================================================================================================
BATCHED = [(None, "rebuild_banks_unchecked"), ("TickEngine", "_publish_batched"), ("TickEngine", "rows_to_copy"),
           ("TickEngine", "copy_rows_many")]


def _fn(cls, name):
    src = (TT.NOSI / "r2_tick.py").read_text()
    return LINT.fn_from_source(src, name if cls is None else "%s.%s" % (cls, name))


@pytest.mark.parametrize("cls,name", BATCHED, ids=[n for _, n in BATCHED])
def test_P6e_the_batched_path_has_no_host_read(cls, name):
    found = LINT.lint_function(_fn(cls, name))
    # documented: transient allocations; the host check of the host list of items; Python ints of host values
    documented = [("ALLOC", r"."), ("BRANCH", r"^len\(set\(bs\)\) != len\(bs\)$"), ("SYNC", r"^int\((b|p|d|dpos)\)$")]
    bad = [f for f in LINT.unexplained(found, documented) if f[0] in ("SYNC", "SHAPE", "BRANCH", "SCALARIDX")]
    assert bad == [], (name, bad)
    assert not [f for f in found if f[0] in ("SHAPE", "SCALARIDX")], (name, found)


def test_P6e_the_per_call_path_shows_its_host_reads_and_a_planted_read_is_found():
    checked = {c for c, _, _ in LINT.lint_function(LINT.fn_from_source(
        (TT.NOSI / "r2_positional_twin.py").read_text(), "PositionalTwin.advance"))}
    assert "SYNC" in checked                                               # the per-call path's checked advance
    fn = _fn(None, "rebuild_banks_unchecked")
    fn.body.insert(3, ast.parse("x = int(hn.max())").body[0])              # NEGATIVE CONTROL: a planted host read
    assert "SYNC" in {c for c, _, _ in LINT.lint_function(fn)}
    fn = _fn("TickEngine", "copy_rows_many")
    fn.body.insert(0, ast.parse("m = cp[0] > 0").body[0])
    fn.body.insert(1, ast.parse("y = cp[0][m]").body[0])                   # a boolean-mask index: a nonzero (a sync)
    assert "SHAPE" in {c for c, _, _ in LINT.lint_function(fn)}
