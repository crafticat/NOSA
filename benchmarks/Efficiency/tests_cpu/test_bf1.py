"""CPU tests of BF1 (KS/analysis/plan/DESIGN_BGFETCH_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the
draft-issued background fetch into the kept store, benchmarks/Efficiency/r2_bf1_fetch.py, its cell r2_bf1_cell.py and its
step script r2_bf1_step.sh. The world: tests_cpu/i4dc_world.py (FakeGraphsDCD: CUDA graphs and Inductor's GPU backend do
not run on the CPU). The CPU establishes the plumbing, the bit equality with the plain arms, the conservation of the
counters, that prefetched blocks REPLACE host gathers of the joint pass, and that a wrong byte is caught; only the GPU
cells establish streams, events, the copy engine, timing and memory.

GATE 0    test_bf1_gate0_files: the tree = RECON7 cd0f0c5 + the BF1 files + ONE declaration block in each older Gate-0
          test; nosi/ and every older module are cd0f0c5's; no kernel source; no hand-written kernel; nobody else names
          the BF1 modules.
          test_bf1_copy: run_arm_bf = r2_i4dc_cell.run_arm_d line for line plus the lines marked "# BF1" only.
PURE      test_pure_rows_rank_alive: rows_of, group_rank, kept_alive against plain Python.
WORLD     test_bf1_equal (timing and check scenarios, inline worker): L4PTFDGUDKWCDB = L4PTFDGUDKWCD in every value but
          the keep's fetched / window-hit counts, the NEW blocks, the map and the bf record; EQF against L4PTFD; the
          cell's gates (LISTED, BF, COUNTS without round_blocks, the inherited ones incl. the keep's fresh-fetch byte
          checks in the check scenario) pass; prefetched blocks were USED and the joint pass's NEW blocks FELL.
          test_bf1_off: L4PTFDGUDKWCDX = L4PTFDGUDKWCD in EVERY value (the hooks are inert with the fetch off).
          test_bf1_thread: the thread worker (CPU stand-in events) keeps EQF and conserves the counters.
          test_bf1_plants: "wrong_bytes" and "no_copy" break EQF and the byte check (L34: the plant was used).
          test_bf1_gilfree (inline, thread) / test_bf1_gilfree_plant: the packer switch (r2_transport_gilfree_v3, packer
          v3 built here with OpenMP, as test_pool_prefetch.py builds it): the same claim and gates; the plant caught.
CELL      test_cell_pure (arguments, gates on planted records, verdicts), test_step_script_bf1 (the cells' arguments
          parse; the pins name the BF1 files).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_bf1.py
"""
from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import re
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "cd0f0c59c1f1cac8a3ef05f6c3abcc0a2079d926"          # nosi-r2-recon7: RECON7 (I4DC 2e9e497 + its cells)
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
MODULES = ("r2_bf1_fetch.py", "r2_bf1_cell.py")
BF1_NEW = tuple(E_ + m for m in MODULES) + (E_ + "r2_bf1_step.sh", T_ + "test_bf1.py")
BF1_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                    "i4h", "i4s", "i4cc", "i4dc", "recon7"))
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
N_RUN = 48                             # committed tokens per request: about 16 rounds + the drain


class Evidence(UserWarning):
    """An evidence line (the CPU template runs pytest without -s: the warnings summary carries these lines)."""


def _ev(msg: str) -> None:
    print(msg)
    warnings.warn(Evidence(msg), stacklevel=2)


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4dc_world")
TS = _load("test_i4s")
BC, L = X.BC, X.L


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _mod(name):
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    return importlib.import_module(name)


# ------------------------------------------------------------------ GATE 0
def test_bf1_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(BF1_NEW) | set(BF1_DECL), sorted(changed - set(BF1_NEW) - set(BF1_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in BF1_NEW), [f for f in BF1_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: cd0f0c5's
    for f in BF1_DECL:                                                    # (c) one BF1 block each, added lines only
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        assert set(old) == set(new) and all(old[k] == new[k] for k in old), (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert body and all(x.startswith("+") for x in body), (f, body[:5])
        assert any("r2_bf1_fetch.py" in x for x in body) and any("BF1" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    for name in MODULES:                                                  # (d) NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    tree = ast.parse((EFF / "r2_bf1_fetch.py").read_text())             # the mechanism: torch, numpy, the I4DC module
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module != "__future__"}
    assert mods <= {"contextlib", "importlib", "queue", "threading", "time", "traceback", "numpy", "torch",
                    "r2_i4dc_compile"}, mods
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names them
        if p.name.startswith("r2_bf1_"):
            continue
        assert "r2_bf1" not in p.read_text(), p.name


def _body_lines(fn):
    """The source lines of a function after its docstring."""
    src = inspect.getsource(fn)
    lines = src.splitlines()
    node = ast.parse(src).body[0]
    first = node.body[0]
    start = first.end_lineno if (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                                 and isinstance(first.value.value, str)) else first.lineno - 1
    return lines[start:]


def test_bf1_copy():
    C = _mod("r2_bf1_cell")
    nb, ob = _body_lines(C.run_arm_bf), _body_lines(C.DCC.run_arm_d)
    core = [x for x in nb if not x.rstrip().endswith("# BF1")]
    extra = [x for x in nb if x.rstrip().endswith("# BF1")]
    assert core == ob, [(i, a, b) for i, (a, b) in enumerate(zip(core, ob)) if a != b][:3] or (len(core), len(ob))
    assert 3 <= len(extra) <= 6, extra
    _ev("BF1 COPY: run_arm_bf = run_arm_d + %d BF1 lines" % len(extra))


# ------------------------------------------------------------------ pure parts
def test_pure_rows_rank_alive():
    BF = _mod("r2_bf1_fetch")
    rng = np.random.default_rng(3)
    B, T, H, bs = 5, 300, 2, 64
    h = rng.integers(0, H, 40)
    b = rng.integers(0, B, 40)
    blk = rng.integers(0, T // bs, 40)
    rows = BF.rows_of(h, b, blk, T, H, bs)
    host = torch.arange(B * T * H * 3, dtype=torch.float32).view(B, T, H, 3)
    got = host.view(-1, 3)[torch.from_numpy(rows)].view(40, bs, 3)
    for i in range(40):                                                 # the 64 rows of the (head, block) of item i
        assert torch.equal(got[i], host[b[i], blk[i] * bs:(blk[i] + 1) * bs, h[i]])
    o = np.lexsort((b, h))
    hs, bsrt = h[o], b[o]
    rk = BF.group_rank(hs, bsrt, B)
    want, last, c = [], None, 0
    for x, y in zip(hs, bsrt):
        c = c + 1 if (x, y) == last else 0
        last = (x, y)
        want.append(c)
    assert rk.tolist() == want
    kid = np.full((H, B, 7), -1, dtype=np.int64)
    kage = np.full((H, B, 7), -(1 << 40), dtype=np.int64)
    kid[0, 1, 2], kage[0, 1, 2] = (1 << 32) * 1 + 3, 2 * 4 + 1              # layer 1 block 3, used at g 2 (alive at g 2)
    kid[1, 4, 5], kage[1, 4, 5] = 7, 1 * 4 + 3                              # layer 0 block 7, used at g 1 (dead at g 2)
    kid[1, 0, 0], kage[1, 0, 0] = 9, 2 * 4                                  # column 0: never a key
    l_, h_, b_, k_ = BF.kept_alive(kid, kage, 2, 4, 50)
    assert list(zip(l_.tolist(), h_.tolist(), b_.tolist(), k_.tolist())) == [(1, 0, 1, 3)]


# ------------------------------------------------------------------ the world runs
def _builder(mode="ce", worker="inline", Xs=None, acct=True, bneg=None, check=False, stats=False, kcheck=False,
             phase_marks=False, probe_every=0, n_cap=4096, ring=4, packer="torch", gf=None, threads=2):
    """i4dc_world.d_builder's L4PTFDGUDKWCD with the BF1 runner and its options (and the store X)."""
    BF = _mod("r2_bf1_fetch")
    W, RE, SR, I3, KA, RW, WB, Z = X.W, X.RE, X.SR, X.I3, X.KA, X.RW, X.WB, X.Z
    Runner = BF.bf_classes(X.STF)[1]

    def build(model, cache, S, horizon, tok0, K=X.K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, Xs or KA.store_slots(X.TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm="natural", extra=arena.X, attn=WB.fake_attention_u,
                          journal=RW.CAP, redirect=True, record=record, probe_every=probe_every)
        kw = dict(fork=Z.fake_fork, t0=tok0, hooks=None, check_every=check_every, union_check=check, union_stats=stats,
                  phase_marks=phase_marks, dneg=None, hneg=None, gneg=None, sneg=None, dgneg=None,
                  cc_backend="inductor", ccneg=None, cdneg=None, keep=True, kwin=True, kcheck=kcheck, kneg=None,
                  kslots=None, cneg=None, bf=dict(mode=mode, worker=worker, acct=acct, bneg=bneg, n_cap=n_cap,
                                                  ring=ring, packer=packer, gf=gf, threads=threads))
        be = X.FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def _run_bf(scen="timing", B=8, N=N_RUN, seed=0, **kw):
    X.CC.configure()
    s = X.SCEN[scen]
    model, c, S, hz, tok0 = X.world(B, N, seed)
    run = _builder(check=s["check"], stats=s["stats"], kcheck=bool(s["check"]), phase_marks=s["phase_marks"],
                   probe_every=s["probe_every"], **kw)(model, c, S, hz, tok0, check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm="L4PTFDGUDKWCDB")


@functools.lru_cache(maxsize=None)
def _plain(arm, scen="timing"):
    return X.run_arm(arm, scen, B=8, N=N_RUN)


@functools.lru_cache(maxsize=None)
def _bf(scen="timing", mode="ce", worker="inline", bneg=None):
    _plain("GUDKC", scen)                                              # the C arm first: it compiles the joint helpers
    _plain("GUDKCD", scen)
    return _run_bf(scen, mode=mode, worker=worker, bneg=bneg)


def _cpu(ev, C):
    return X.cell_inputs(ev, C)


def _gates(ev, scen, arm="L4PTFDGUDKWCDB"):
    C = _mod("r2_bf1_cell")
    keepd = {"L4PTFD": _cpu(_plain("D", scen), C), "L4PTFDGUDKWC": _cpu(_plain("GUDKC", scen), C),
             C.REF: _cpu(_plain("GUDKCD", scen), C)}
    rd = ev["rd"]
    rec = dict(graph=rd["graph"], dgraph=rd["dgraph"], gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    return C.arm_gates_bf(arm, rd, rec, _cpu(ev, C), keepd, scen == "check", 8, 10, N_RUN, L, 1.0)


DIFF_RD = ("keep", "round_blocks", "bf", "compile")                    # the values BF1 changes (and compile timing)
DIFF_T = ("kid", "kage", "blocks")


@pytest.mark.parametrize("scen", ["timing", "check"])
def test_bf1_equal(scen):
    ref, ev = _plain("GUDKCD", scen), _bf(scen)
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]                                         # THE CLAIM: only where the bytes come from
    bf = ev["rd"]["bf"]
    gs = _gates(ev, scen)
    names = [g[0] for g in gs]
    for n in ("-EQF", "-LISTED", "-BF", "-COUNTS"):
        assert "L4PTFDGUDKWCDB" + n in names, (n, names)
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    used = bf["used_store"] + bf["used_win"]
    nb, nr = sum(ev["rd"]["round_blocks"]), sum(ref["rd"]["round_blocks"])
    assert bf["committed"] > 0 and used > 0, bf                        # the mechanism is not a no-op (L34)
    assert nb < nr, (nb, nr)                                           # prefetched blocks REPLACED host gathers
    kr, kb = ref["rd"]["keep"], ev["rd"]["keep"]
    assert kb["listed"] == kr["listed"] and kb["fetched_total"] <= kr["fetched_total"], (kb["fetched_total"],
                                                                                        kr["fetched_total"])
    if scen == "check":
        ck = kb["check"]
        assert ck["bytes_bad"] == 0 and ck["bytes_checked"] > 0 and ck["win_checked"] == kb["win_hits_total"], ck
    _ev("BF1 EQUAL %s: rounds %d; NEW blocks %d -> %d (%.3f); keep fetched %d -> %d, window hits %d -> %d; bf jobs %d, "
        "issued %d, items seen %d, have %d, sent %d, committed %d, revived %d, no room %d, used store %d + window %d, "
        "unused %d, pending %d; check %s" % (
            scen, ev["rd"]["rounds"], nr, nb, nb / nr, kr["fetched_total"], kb["fetched_total"], kr["win_hits_total"],
            kb["win_hits_total"], bf["jobs"], bf["issued"], bf["items_seen"], bf["items_have"], bf["items_sent"],
            bf["committed"], bf["revived"], bf["no_room"], bf["used_store"], bf["used_win"], bf["unused"],
            bf["pending"], kb.get("check")))


def test_bf1_off():
    ref, ev = _plain("GUDKCD"), _bf("timing", mode="off")
    bad = TS._diff(ref, ev, skip=("bf", "compile"))
    assert bad == [], bad[:10]                                         # the hooks are inert with the fetch off
    gs = _gates(ev, "timing", arm="L4PTFDGUDKWCDX")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert ev["rd"]["bf"]["items"] == 0 and ev["rd"]["bf"]["mode"] == "off"


def test_bf1_thread():
    ref, ev = _plain("GUDKCD"), _bf("timing", worker="thread")
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    bf = ev["rd"]["bf"]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["error"] is None and bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    _ev("BF1 THREAD: jobs %d, issued %d, late %d, empty %d; sent %d, committed %d, used %d" % (
        bf["jobs"], bf["issued"], bf["late"], bf["empty_jobs"], bf["items_sent"], bf["committed"],
        bf["used_store"] + bf["used_win"]))


@pytest.mark.parametrize("bneg", ["wrong_bytes", "no_copy"])
def test_bf1_plants(bneg):
    ev = _bf("check", bneg=bneg)
    bf = ev["rd"]["bf"]
    assert bf["used_store"] + bf["used_win"] > 0, bf                   # L34: the plant reached a reader
    gs = {g[0]: g for g in _gates(ev, "check")}
    assert not gs["L4PTFDGUDKWCDB-EQF"][2], gs["L4PTFDGUDKWCDB-EQF"]   # caught by EQF
    assert ev["rd"]["keep"]["check"]["bytes_bad"] > 0                  # and by the keep's fresh-fetch byte check
    _ev("BF1 PLANT %s: caught (EQF %s; bytes_bad %d)" % (bneg, gs["L4PTFDGUDKWCDB-EQF"][3][:120],
                                                        ev["rd"]["keep"]["check"]["bytes_bad"]))


@pytest.fixture(scope="module")
def gfext(tmp_path_factory):
    """Packer v3 of THIS tree, built with OpenMP (as tests_cpu/test_pool_prefetch.py's fixture)."""
    gp = _load("gilfree_pack_v3_bf1", EFF / "gilfree_pack.py")
    e = gp.load(str(tmp_path_factory.mktemp("gilfree3_bf1")))
    assert int(e.VERSION) == 3 and int(e.DESC) >= 2 and int(e.OPENMP) == 1
    return e


@pytest.mark.parametrize("worker", ["inline", "thread"])
def test_bf1_gilfree(worker, gfext):
    """THE PACKER SWITCH: r2_transport_gilfree_v3 (worker "cpp", descriptor jobs into the flat landing) in place of
    index_select: the same claim (= L4PTFDGUDKWCD but where the bytes come from), the same gates, used blocks, the NEW
    blocks fall; the transport's own stats (a team of the requested size)."""
    _plain("GUDKC")
    ref = _plain("GUDKCD")
    cls = X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport
    ev = _run_bf("timing", worker=worker, packer="gilfree", gf=dict(ext=gfext, cls=cls), threads=2)
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    bf = ev["rd"]["bf"]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    nb, nr = sum(ev["rd"]["round_blocks"]), sum(ref["rd"]["round_blocks"])
    tp = bf["transport"]
    assert bf["packer"] == "gilfree" and tp is not None and tp["threads_ok"] and tp["releases"] > 0, tp
    assert bf["used_store"] + bf["used_win"] > 0 and nb <= nr, (bf, nb, nr)
    if worker == "inline":
        assert nb < nr and bf["late"] == 0, (nb, nr, bf["late"])
    _ev("BF1 GILFREE %s: NEW %d -> %d; sent %d, committed %d, used %d, late %d; transport %s" % (
        worker, nr, nb, bf["items_sent"], bf["committed"], bf["used_store"] + bf["used_win"], bf["late"],
        {k: tp.get(k) for k in ("releases", "groups", "chunks", "pack_ns", "threads_min", "threads_max")}))


@pytest.mark.parametrize("bneg", ["wrong_bytes"])
def test_bf1_gilfree_plant(bneg, gfext):
    """The gilfree path's bytes are checked by the same gates: a packed wrong block is caught (EQF, bytes_bad)."""
    _plain("GUDKC", "check")
    _plain("GUDKCD", "check")
    cls = X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport
    ev = _run_bf("check", worker="inline", packer="gilfree", gf=dict(ext=gfext, cls=cls), threads=2, bneg=bneg)
    bf = ev["rd"]["bf"]
    assert bf["used_store"] + bf["used_win"] > 0, bf
    gs = {g[0]: g for g in _gates(ev, "check")}
    assert not gs["L4PTFDGUDKWCDB-EQF"][2] and ev["rd"]["keep"]["check"]["bytes_bad"] > 0


# ------------------------------------------------------------------ the cell
def test_cell_pure(monkeypatch):
    C = _mod("r2_bf1_cell")
    base = ["--out", "/x", "--workload", "/w", "--bf-packer", "torch"]
    a = C.parse_args(base)
    assert a.arms == ["L4PTFD", "L4PTFDGUDKWC", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDX", "L4PTFDGUDKWCDB"]
    for bad in (["--arms", "L4PTFD,L4PTFDGUDKWCDB"], ["--arms", "L4PTFD,L4PTFDGUDKWCDB,L4PTFDGUDKWCD"],
                ["--bf-store", "100"], ["--bf-ring", "1"], ["--check-every", "1"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    assert C.bf_opt(a, "L4PTFDGUDKWCDB")["mode"] == "ce" and C.bf_opt(a, "L4PTFDGUDKWCDX")["mode"] == "off"
    assert C.bf_opt(a, "L4PTFDGUDKWCDB")["packer"] == "torch"
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    with pytest.raises(SystemExit):                                    # gilfree without a build directory
        C.parse_args(["--out", "/x", "--workload", "/w", "--bf-packer", "gilfree"])
    with pytest.raises(SystemExit):
        C.parse_args(base + ["--bf-threads", "0"])
    good = dict(mode="ce", error=None, issued=3, items=10, items_sent=10, committed=7, revived=1, no_room=2,
                used_store=4, used_win=1, unused=2, pending=1)
    assert C.bf_gate("B", good, "ce")[2]
    for k, v in (("error", "x"), ("items", 9), ("no_room", 3), ("unused", 3), ("committed", 0), ("issued", 0)):
        assert not C.bf_gate("B", dict(good, **{k: v}), "ce")[2], k
    assert C.bf_gate("X", dict(mode="off", issued=0, items=0), "off")[2]
    assert not C.bf_gate("X", dict(mode="off", issued=1, items=4), "off")[2]
    kp = dict(listed=[3, 4, 5])
    assert C.listed_gate("B", kp, dict(listed=[3, 4, 5]), "R")[2] and not C.listed_gate("B", kp, dict(listed=[3, 4, 6]),
                                                                                         "R")[2]
    sp = {n: dict(rates=dict(median_tok_s=t), phases_median=dict(joint=j, draft=d), kv_at_verify=dict(
        new_blocks_per_round=nb, wait_ms_median=w), bf=dict(barrier_ms_median=0.5))
          for n, t, j, d, nb, w in (("L4PTFDGUDKWCDB", 110.0, 300.0, 180.0, 50.0, 60.0),
                                    ("L4PTFDGUDKWCDX", 100.0, 340.0, 176.0, 100.0, 115.0),
                                    ("L4PTFDGUDKWCD", 101.0, 339.0, 176.0, 100.0, 116.0))}
    v = {p[0]: p for p in C.verdicts(sp, 318, True)}
    assert v["BF1-VS-X"][1] == "REPORT" and "1.1000" in v["BF1-VS-X"][2] and v["BF1-NEW-RATIO"][2].startswith("0.5000")
    assert C.PRED_STAMP_BF1 is None                                    # the lead registers (step script stops before)


CELLS = {"bf1m": (10, 8, 72), "bf1a": (65, 216, 512), "bf1b": (10, 318, 512)}


def test_step_script_bf1(monkeypatch, tmp_path):
    C = _mod("r2_bf1_cell")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))         # the step script sets it ($JIT/torch_ext)
    text = (EFF / "r2_bf1_step.sh").read_text()
    assert "--bf-packer gilfree" in text and "nosi/nosi/r2_transport_gilfree_v3.py" in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-bf1" in text and "CELLPY=benchmarks/Efficiency/r2_bf1_cell.py" in text
    assert "benchmarks/Efficiency/r2_bf1_fetch.py" in text and "^PRED_STAMP_BF1 = None$" in text
    for cell, (pool, batch, steps) in CELLS.items():
        m = re.search(r"^\s+%s\) (.*?) ;;$" % cell, text, re.M | re.S)
        assert m, cell
        kv = dict(x.split("=", 1) for x in shlex.split(m.group(1).replace("\n", " ")))
        assert (int(kv["POOL"]), int(kv["BATCH"]), int(kv["STEPS"])) == (pool, batch, steps), (cell, kv)
        argv = ["--out", "/x", "--workload", "/w", "--batch", kv["BATCH"], "--steps", kv["STEPS"], "--pool",
                kv["POOL"], "--arms", kv["ARMS"], "--bf-store", kv["BFX"], "--bf-ncap", kv["NCAP"],
                "--check-every", kv.get("CHECK", "0"), "--require-stress", kv.get("STRESS", "0"),
                "--bf-acct", kv.get("ACCT", "0"), "--bf-packer", "gilfree", "--bf-threads", "8"]
        a = C.parse_args(argv)
        assert a.bf_packer == "gilfree"
        assert "L4PTFDGUDKWCDB" in a.arms and "L4PTFDGUDKWCDX" in a.arms, (cell, a.arms)
    _ev("BF1 STEP: cells %s parse" % sorted(CELLS))
