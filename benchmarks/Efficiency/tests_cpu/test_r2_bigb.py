"""[bigb] CPU tests of the LARGE-B limits and the memory model (benchmarks/Efficiency/bigb_core.py; docs/R2_BIGB_PLAN.md).
Production geometry (L 32, H 2, D 128, bf16, S 16128) through "meta" tensors (shapes, no memory); the tiny model of
test_r2_tick.py where a round must run. No CUDA.

  BG1  THE TOKEN LOG up to B 320: a TickEngine built at the production geometry for B in 64 .. 320 on BOTH layouts
       (per-stream store; XPool mode pool) holds log_slots_for(B, 24) slots, at least the round's minimum rows;
       the default 1 << 16 is below log_slots_for(B, 512) at every B >= 64 (the old cap: B x N). The harness's
       --log-slots -1 gives log_slots_for(B, max(steps, sweep-steps, preflight-steps)); 0 keeps the default (None).
  BG2  THE SIZE IS ENOUGH on a real round (the tiny model, free running): n_tok <= log_slots_for(B, N); a log of the
       round's MINIMUM rows minus one is refused ("token log full"), so the check is live and the bound is real.
  BG3  EVERY BUFFER'S BYTES equal the model's terms at B in 64 .. 320 (meta): the 2B-bank twins, the store (both
       layouts; r2_xpool's two formulas), the token log, B2's Table, S-C's DeviceMirror + GenBoard + the sparse pins,
       the XPool map, PaddedLinear's buffers + lm_head's padded output (the workspace).
  BG4  THE FIT: F0 and R_FIX from p5b2 are small and positive (the code's terms explain 67.36 of 67.49 GB with TWO
       engines); the same fit predicts p4sc64 (67.41 GB, S-C: no B2 table) within 0.1 GB. NEGATIVE: one engine
       predicts 42.7 GB and the build phase 28.8 GB: neither explains the measured peak.
  BG5  THE PREFLIGHT RELEASES ITS ROUND: a real TickRound holds its engine (precondition, on the tiny engine);
       r2_tick_sg5.preflight's next engine is built only after the previous one is collectable. NEGATIVE CONTROLS:
       the same preflight with feaa7d6's release line (rnd not released) keeps the first engine alive; a round_fn
       that stashes its round keeps it alive (the detector detects). one_point releases rnd too (static).
  BG6  int32 extents: every int32 quantity of the tick path admits B > 320 at 16K (host rows: 515, the smallest).
  BG7  THE CELL: B_max at 76 GB is 244 (C 81) and 216 (C 96) on the pool layout; the step's bigb cells name exactly
       those B and B 128; their host memory fits --mem 200G; the shipped child would not fit 120G at B >= 128.
  BG8  THE TIME BANDS: at B 64 every band is p5b2's record; low <= center <= high, increasing in B; the step's
       registered numbers are tick_bands' (rounded).
  BG9  the harness's --skip-shipped / --log-slots refusals and defaults; the step's bigb cells (static + DRYRUN in
       the bigb ROOT checkout).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_bigb.py
"""
from __future__ import annotations

import gc
import importlib.util
import inspect
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _shared(name, file):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


TT = _shared("test_r2_tick_shared_bigb", "test_r2_tick.py")
TK, TW, RSC, OWN2 = TT.TK, TT.TW, TT.RSC, TT.OWN2
O = sys.modules[TT.PKG + ".r2_own2_ops"]
SC = TT.SC
XPM = TT._load("r2_xpool")
SPARSE = sys.modules[TT.PKG + ".r2_residency_sparse"]
p3 = TT.p3
EFF, ROOT = TT.EFF, TT.ROOT
sys.path.insert(0, str(EFF))
import bigb_core as BC  # noqa: E402

BF = torch.bfloat16
BS_ALL = (64, 128, 192, 244, 256, 287, 320)
S16 = 16128


def _nb(ts):
    return sum(t.numel() * t.element_size() for t in ts)


def _tensors(obj):
    return [v for v in vars(obj).values() if isinstance(v, torch.Tensor)]


def meta_engine(B, layout, C=81, X=64, X_pool=64, steps=24, log_slots=None):
    """A TickEngine at the production geometry on "meta" (no memory): 32 layers of 2B-bank twins and the store of the
    layout (the from_r2 shapes), stand-ins for the model and the kernels (never called)."""
    w = BC.widths(S16, steps)
    twins = [TW.PositionalTwin(2 * B, BC.H, BC.D, w["lk_max"], w["n_max"], dtype=BF, device="meta")
             for _ in range(BC.L)]
    xp = None
    if layout == "stream":
        rows = (C + X + 4) * BC.BLOCK
        sk = [torch.zeros((B, rows, BC.H, BC.D), dtype=BF, device="meta") for _ in range(BC.L)]
        sv = [torch.zeros_like(k) for k in sk]
        sb = [torch.zeros((B, rows, BC.H), dtype=BF, device="meta") for _ in range(BC.L)]
    else:
        xp = XPM.XPool(BC.L, B, BC.H, C + X, mode="pool", C=C, X_pool=X_pool, device="meta")
        sk, sv, sb = [], [], []
        for _ in range(BC.L):
            k, v, b = xp.new_store(BC.D, dtype=BF, device="meta")
            sk.append(k)
            sv.append(v)
            sb.append(b)
    model = SimpleNamespace(layers=[None] * BC.L)
    eng = TK.TickEngine(model, SimpleNamespace(mean=None), twins, sk, sv, sb, [S16] * B, w["M_alloc"], nl=object(),
                        stage1=object(), pool=object(), xpool=xp, log_slots=log_slots)
    return eng, xp, w


# =====================================================================================================================
# BG1 / BG2 the token log
# =====================================================================================================================
@pytest.mark.parametrize("layout", ["stream", "pool"])
def test_BG1_the_token_log_is_sized_for_B_up_to_320(layout):
    for B in BS_ALL:
        n = BC.log_slots_for(B, 24)
        eng, _, _ = meta_engine(B, layout, log_slots=n)
        assert eng.B == B and eng.LOG == n == eng.tok_log.numel() and eng.tok_log.dtype == torch.int64
        assert n >= BC.min_round_rows(B, 24) and n == 10 * B * 34
        assert BC.log_slots_for(B, 512) > TK.TickEngine.LOG                     # the old cap: B x N at N 512
        assert eng.clone is not None and eng.NB == 81 + 64
    d, _, _ = meta_engine(64, layout)                                           # the default is unchanged
    assert d.LOG == 1 << 16 == d.tok_log.numel()
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--batch", "287", "--steps", "24", "--log-slots", "-1"])
    assert a.log_slots_n == BC.log_slots_for(287, 24, 3, 7) == 97580 > 1 << 16
    a = G.parse_args(base + ["--batch", "287", "--steps", "40", "--sweep-steps", "12", "--log-slots", "-1"])
    assert a.log_slots_n == BC.log_slots_for(287, 40, 3, 7)
    assert G.parse_args(base).log_slots_n is None and G.parse_args(base + ["--log-slots", "5000"]).log_slots_n == 5000


def test_BG2_the_size_holds_a_real_round_and_the_refusal_is_live(p3):
    B, N = len(TT.S_TINY), 24
    eng = TT.make_engine(TT.make_ops(p3), log_slots=BC.log_slots_for(B, N))
    rnd = TK.run_round(eng, TT.FIRST, [N] * B, K=3, LAMBDA=7)
    assert all(r.done for r in rnd.reqs)
    assert BC.min_round_rows(B, N) <= eng.n_tok <= BC.log_slots_for(B, N), (eng.n_tok, BC.log_slots_for(B, N))
    small = TT.make_engine(TT.make_ops(p3), log_slots=BC.min_round_rows(B, N) - 1)
    with pytest.raises(TK.TickError, match="token log full"):
        TK.run_round(small, TT.FIRST, [N] * B, K=3, LAMBDA=7)
    assert small.n_tok <= small.LOG


# =====================================================================================================================
# BG3 every buffer's bytes
# =====================================================================================================================
@pytest.mark.parametrize("layout", ["stream", "pool"])
def test_BG3_every_buffer_equals_the_model(layout):
    C, X, Xp = 81, 64, 64
    for B in BS_ALL:
        n = BC.log_slots_for(B, 24)
        eng, xp, w = meta_engine(B, layout, C, X, Xp, log_slots=n)
        model = BC.engine_bytes(layout, B, C, X, Xp, w, "b2", n)
        twins = sum(_nb(_tensors(t)) for t in eng.twins)
        assert twins == model["twins_2B"], (B, twins, model["twins_2B"])
        assert _nb(eng.store_k + eng.store_v + eng.store_bias) == model["store"]
        if layout == "pool":
            assert model["store"] == BC.L * XPM.store_bytes_formula(B, BC.H, BC.D, C, Xp, 2)
            assert _nb([xp.pos_of, xp.occ, xp.err, xp.peak]) == model["xpool_map"]
        else:
            assert model["store"] == BC.L * XPM.per_stream_bytes_formula(B, BC.H, BC.D, C + X, 2)
        assert _nb([eng.tok_log]) == model["tok_log"] == 8 * n
        geo = O.Geo(L=BC.L, B=B, H=BC.H, C=C, X=X, R_ver=16, NBLK=w["nblk"], KEYS=BC.B2_KEYS)
        tab = O.Table(geo, device="meta")
        assert _nb(list(tab.tensors().values())) == model["b2_table"]
        dm = RSC.DeviceMirror(BC.L, B, BC.H, w["nblk"], C + X, "meta")
        gb = RSC.GenBoard(BC.L, B, BC.H, C + X, "meta")
        pins = torch.zeros((BC.L, B, BC.H, w["nblk"] + 1, 2), dtype=torch.int64, device="meta")
        assert _nb(_tensors(dm) + _tensors(gb) + [pins]) == model["sc"]
        assert w["nblk"] == w["n_host"] // 64 == 254
    # the pins' shape is the code's (r2_residency_sparse.ResidencySparse.__init__)
    src = inspect.getsource(SPARSE.ResidencySparse.__init__)
    assert "self.pins = torch.zeros((self.L, self.B, self.H, self.nblk + 1, 2), dtype=torch.int64" in src
    # S-C allocates both device tables in __init__; b2 is an S-C (OwnResidency inherits it)
    sc_src = inspect.getsource(RSC.ResidencySC.__init__)
    assert "GenBoard(self.L, self.B, self.H, self.n_hist" in sc_src and "DeviceMirror(self.L, self.B, self.H" in sc_src
    assert issubclass(OWN2.OwnResidency, RSC.ResidencySC)


def test_BG3b_the_workspace_equals_the_padded_gemm_buffers():
    for B in (64, 244, 320):
        m = 4 * B
        pl = SC.PaddedLinear(m)
        bufs = [pl.buffer(K, BF, "meta") for K in (BC.HIDDEN, BC.INTER)]
        x = torch.zeros((m, BC.HIDDEN), dtype=BF, device="meta")
        y = torch.nn.functional.linear(x, torch.zeros((BC.VOCAB, BC.HIDDEN), dtype=BF, device="meta"))
        assert _nb(bufs + [y, y.float()]) == BC.workspace_bytes(B, 3)


# =====================================================================================================================
# BG4 the fit
# =====================================================================================================================
def test_BG4_the_fit_explains_p5b2_and_p4sc64_with_two_engines():
    m = BC.MemModel()
    assert 0 < m.F0 - BC.WEIGHTS_BYTES < 0.3e9                    # the rope cache and the allocator's own part
    assert 0 < m.R_FIX < 0.4e9                                    # the code's terms explain all but < 0.4 GB
    kw = dict(steps=48, sweep_steps=24, log_slots=1 << 16)
    p5 = m.peak_gb("stream", 64, 96, 64, None, backend="b2", engines=2, **kw)
    assert abs(p5 - 67.49) < 1e-6
    p4 = m.peak_gb("stream", 64, 96, 64, None, backend="sc", engines=2, **kw)
    assert abs(p4 - BC.P4SC64["peak_gb"]) < 0.1, p4             # an independent point: S-C without B2's table
    one = m.peak_gb("stream", 64, 96, 64, None, backend="b2", engines=1, **kw)
    assert 42.6 < one < 42.9 and 67.49 - one > 20               # NEGATIVE: one engine cannot be the measured peak
    assert m.build_peak_gb(64, steps=48) < 30                     # NEGATIVE: nor the build phase
    eng = sum(BC.engine_bytes("stream", 64, 96, 64, None, BC.widths(S16, 48), "b2").values()) / 1e9
    assert 24.7 < eng < 24.8                                      # the second engine: 24.75 GB


# =====================================================================================================================
# BG5 the preflight releases its round
# =====================================================================================================================
def test_BG5_a_tick_round_holds_its_engine(p3):
    """PRECONDITION on the real code: run_round's TickRound keeps the engine alive after `del eng`."""
    eng = TT.make_engine(TT.make_ops(p3))
    rnd = TK.run_round(eng, TT.FIRST, [6] * eng.B, K=3, LAMBDA=7)
    ref = weakref.ref(eng)
    del eng
    gc.collect()
    assert ref() is not None                                      # held by rnd.read_tokens / on_publish
    del rnd
    gc.collect()
    assert ref() is None


def _preflight_run(G, preflight, stash=None):
    """preflight with stand-ins over two transports x one backend (gilfree + gather: two engines in sequence).
    Returns, per engine build after the first, whether the PREVIOUS engine was still alive."""
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--backends", "b2"])
    refs, alive = [], []

    class Eng:
        B = 2

        def read_tokens(self, lo, hi):
            return []

        def mirror_failures(self):
            return []

    class Tp:
        def close(self):
            pass

    def engine_fn(a_, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=None, transport=None):
        gc.collect()
        if refs:
            alive.append(refs[-1]() is not None)
        e = Eng()
        refs.append(weakref.ref(e))
        return e, None, Tp()

    def round_fn(eng, first, max_new, K, LAMBDA):
        r = SimpleNamespace(ticks=3, syncs=1, read_tokens=eng.read_tokens)    # as TickRound: a bound method
        if stash is not None:
            stash.append(r)
        return r
    gates, res = [], {}
    passed = preflight(a, None, None, None, [0, 0], None, None, gates, res, engine_fn=engine_fn, round_fn=round_fn)
    assert passed == ["b2"] and len(refs) == 2
    return alive


def test_BG5b_the_preflight_builds_the_next_engine_after_the_last_one_is_gone():
    G = TT._sg5_gate()
    assert _preflight_run(G, G.preflight) == [False]
    # NEGATIVE CONTROL 1: feaa7d6's release line (rnd not released) keeps the first engine alive
    src = inspect.getsource(G.preflight)
    fixed = "del eng, svc, tp, rnd"
    assert src.count(fixed) == 1
    old = src.replace(fixed, "del eng, svc, tp")
    ns = dict(G.__dict__)
    exec(compile(old, "preflight_feaa7d6", "exec"), ns)
    assert _preflight_run(G, ns["preflight"]) == [True]
    # NEGATIVE CONTROL 2: a caller that keeps its round keeps the engine (the detector sees a live reference)
    assert _preflight_run(G, G.preflight, stash=[]) == [True]
    # one_point releases its round with the engine (static: it needs the CUDA harness to run)
    op = inspect.getsource(G.one_point)
    assert "rnd = None" in op and op.count("del eng, svc, tp, rnd") == 1
    assert op.index("del eng, svc, tp, rnd") < op.index("if bare:")


# =====================================================================================================================
# BG6 int32
# =====================================================================================================================
def test_BG6_no_int32_extent_binds_below_B_320():
    for C in (81, 96):
        b = BC.int32_bounds(C=C)
        assert min(b.values()) > 320, b
        assert b["host_rows"] == 515 == min(b.values())          # the smallest (a margin: the kernels use int64)
    eng, xp, w = meta_engine(320, "pool", 96, 64, 64)
    assert xp.M * BC.BLOCK < 2 ** 31 and 2 * 320 * w["lk_max"] < 2 ** 31


# =====================================================================================================================
# BG7 the cell
# =====================================================================================================================
def test_BG7_the_cell_plan_and_the_step_cells():
    m = BC.MemModel()
    p81, p96 = BC.cell_plan(81, model=m), BC.cell_plan(96, model=m)
    assert (p81["B_max"], p96["B_max"]) == (244, 216)
    for p in (p81, p96):
        for pt in p["points"]:
            assert pt["peak_gb"] <= BC.BUDGET_GB and pt["build_peak_gb"] < pt["peak_gb"]
            assert pt["host_gib"] <= 200 and pt["stream_peak_gb"] > pt["peak_gb"]
            assert pt["log_slots"] == BC.log_slots_for(pt["B"], 24)
    nxt = m.peak_gb("pool", 245, 81, 64, 64, steps=24)
    assert nxt > BC.BUDGET_GB                                     # B_max + 1 is over the budget
    got = {(p["C"], pt["B"]): round(pt["peak_gb"], 2) for p in (p81, p96) for pt in p["points"]}
    assert got == {(81, 128): 47.72, (81, 244): 75.82, (96, 128): 51.78, (96, 216): 75.89}
    assert [round(m.build_peak_gb(B), 2) for B in (128, 216, 244)] == [38.71, 52.33, 56.66]
    for B in (128, 216, 244):                                     # limit 3: the shipped child's host rows
        assert BC.host_gib(B, shipped=True) > 120 and BC.host_gib(B) <= 200
    text = (EFF / "r2_tick_sg5_step.sh").read_text()
    for c in ("bigb81-128", "bigb81-244", "bigb96-128", "bigb96-216"):
        assert c in text
    for line in ("bigb81-128 47.72   bigb81-244 75.82   bigb96-128 51.78   bigb96-216 75.89",
                 "38.71 at B 128, 52.33 at B 216, 56.66 at B 244"):
        assert line in text


# =====================================================================================================================
# BG8 the time bands
# =====================================================================================================================
def test_BG8_the_time_bands():
    t64 = BC.tick_bands(64)
    for band in ("low", "center", "high"):
        assert abs(t64[band]["T_tick"] - sum(BC.P5B2_ROUND.values())) < 1e-9
        assert abs(t64[band]["td_dev"] - 60.647) < 1e-3
    prev = t64
    for B in (128, 216, 244, 287):
        t = BC.tick_bands(B)
        for k in ("T_tick", "td_dev", "draft_tick"):
            assert t["low"][k] <= t["center"][k] <= t["high"][k]
            assert t["center"][k] > prev["center"][k]
        prev = t
    reg = {128: ((1.97, 2.36, 2.62), (77, 92, 107), (0.91, 1.09, 1.22)),
           216: ((2.82, 3.75, 4.37), (99, 135, 171), (1.28, 1.71, 2.03)),
           244: ((3.09, 4.19, 4.93), (106, 148, 191), (1.40, 1.91, 2.28))}
    text = (EFF / "r2_tick_sg5_step.sh").read_text()
    for B, (tt, td, dw) in reg.items():
        t = BC.tick_bands(B)
        assert tuple(round(t[b]["T_tick"] / 1e3, 2) for b in ("low", "center", "high")) == tt, B
        assert tuple(int(round(t[b]["td_dev"])) for b in ("low", "center", "high")) == td, B
        assert tuple(round(t[b]["draft_tick"] / 1e3, 2) for b in ("low", "center", "high")) == dw, B
        assert "B %d: %.2f / %.2f / %.2f s" % ((B,) + tt) in text
        assert "B %d: %d / %d / %d ms" % ((B,) + td) in text


# =====================================================================================================================
# BG9 the harness flags and the step cells
# =====================================================================================================================
def test_BG9_harness_flags_and_the_step_cells(tmp_path):
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base)
    assert (a.skip_shipped, a.log_slots, a.log_slots_n) == (False, 0, None)
    for bad in (["--skip-shipped", "--pool-baseline"], ["--log-slots", "-2"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    msrc = inspect.getsource(G.main)
    assert "if a.skip_shipped:" in msrc and '"R_shipped_skipped", True, False' in msrc
    assert 'meta["preflight_peak_gb"]' in msrc and 'meta["build_peak_gb"]' in msrc
    script = EFF / "r2_tick_sg5_step.sh"
    text = script.read_text()
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in text.splitlines():
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    assert "NOSI_BIGB=/mnt/beegfs/ojerbi/nosi-r2-bigb " in text
    assert "--batch 64 --S 16128 --steps 48" not in text and '--batch "$BATCH"' in text and '--mem="$MEMG"' in text
    assert "BATCH=64 STEPS=48 MEMG=120G" in text and "POOLB=(--pool-baseline)" in text   # every other cell: as before
    bigb_root = next(ln for ln in text.splitlines() if ln.startswith("NOSI_BIGB=")).split()[0][len("NOSI_BIGB="):]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", "nosi"],
                               capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        pytest.skip("not a clean git checkout: the DRYRUN needs one")
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "417.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    for cell, C, B in (("bigb81-128", 81, 128), ("bigb81-244", 81, 244), ("bigb96-128", 96, 128),
                       ("bigb96-216", 96, 216)):
        r = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "417", "t1", cell], env=env,
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            assert Path(bigb_root).resolve() != ROOT.resolve(), r.stderr.strip()[-300:]   # in the ROOT: a FAILURE
            pytest.skip("DRYRUN stopped outside the bigb ROOT: %s" % r.stderr.strip()[-200:])
        line = r.stdout.strip().splitlines()[-1]
        got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
        for w in ("--batch %d " % B, "--C %d --X 64 --R-ver 16" % C, "--steps 24 ", "--modes exact --backends b2",
                  "--layout pool --X-pool 64 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped"):
            assert w in got, (cell, w)
        assert "--pool-baseline" not in got and "--sweep " not in got and "--variants" not in got
        assert "--mem=200G " in line and "PYTHONPATH=/mnt/beegfs/ojerbi/nosi-r2-bigb/nosi" in line
        assert "--job-name=sg5-t1-417-%s " % cell.replace("_", "-") in line
        pa = G.parse_args(got.split())
        assert (pa.batch, pa.C, pa.layout, pa.X_pool, pa.arrival_lag, pa.skip_shipped, pa.repeat_docs) == \
            (B, C, "pool", 64, 2, True, True)
        assert pa.log_slots_n == BC.log_slots_for(B, 24, 3, 7)
    for bad in (("bigb81-245",), ("bigb81",), ("bigb81-128", "x1")):
        rb = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "417", "t1", *bad], env=env,
                            capture_output=True, text=True, timeout=120)
        assert rb.returncode == 2 and "usage" in rb.stderr, bad
