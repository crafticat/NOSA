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
          test_bf1_gilfree_batch / test_bf1_carry: the worker's batches (one release of all layers, one commit; a job
          that does not fit is carried, not dropped). test_bf1_pipeline: batch k's select runs while batch k - 1's
          C++ job is pending (overlapped > 0; a stand-in pipe holds each submission 50 ms). test_bf1_store_x: a store X other than REF's (bf1b): COUNTS
          declares union.cap / store_slots, the warm compiles loc2 before the first capture; warm off -> COMPILE fails.
SNAP / ARRIVAL (bf1r, INTEGRATION_NOTES_2026-10-31.md section 13; fake events that complete after a delay):
          test_bf1_ready_only (a: snap "ready" takes no job with an incomplete snapshot event; "newest" does: the
          control), test_bf1_lifetimes (b: snapshot and staging lifetimes under the pipeline; two planted defects
          caught), test_bf1_arrival_modes (c: the arrival block in every mode, its closures), test_bf1_arrival_late (d:
          a planted late job is counted late, an independent count agrees; the inline control has none).
          REVIEW 5: test_bf1_worker_failure (query / sync: a raising readiness event outside _prep / _finish; the
          original exception reaches the compute thread, the worker stops, no slot stays owned),
          test_bf1_arrival_store_only (acct=False / arrival=True with window-only consumers: the metric is STORE-ONLY).
CELL      test_cell_pure (arguments, gates on planted records, verdicts), test_counts_gate_bf (bf1b's COUNTS defect and
          its fix), test_step_script_bf1 (the cells' arguments parse; the pins name the BF1 files).

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
import threading
import time
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
# INT (nosi-r2-int: J1 + ST1 + BF1 in one arm, r2_int_cell.py): the J1 and ST1 rungs' files and its own (its own Gate
# 0: tests_cpu/test_int.py); BF1_LATER: its modules import the BF1 modules (scan (e) skips them)
BF1_NEW = BF1_NEW + (E_ + "r2_j1_select.py", E_ + "r2_j1_cell.py", E_ + "r2_j1_step.sh", E_ + "r2_j1_forkbench.py",
                     T_ + "test_j1.py", "docs/NOTES_j1.md",
                     E_ + "r2_st1_commit.py", E_ + "r2_st1_cell.py", E_ + "r2_st1_step.sh", T_ + "test_st1.py",
                     E_ + "r2_int_arms.py", E_ + "r2_int_cell.py", E_ + "r2_int_step.sh", T_ + "test_int.py",
                     "docs/NOTES_int.md")                                                         # INT
BF1_LATER = ("r2_int_arms.py", "r2_int_cell.py")                                                  # INT
# R5 (review 5, ST: the yield / window population repair, r2_i4h_cell.population_repair; its tests in test_int.py)
BF1_NEW = BF1_NEW + (E_ + "r2_i4h_cell.py",)                                                       # R5
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
        if p.name.startswith("r2_bf1_") or p.name in BF1_LATER:                                     # INT
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
    free, (l2, h2, b2, k2) = BF.store_room(kid, kage, 2, 4, 4, 50)     # THE ROOM RULE: empty or dead slots 1 .. X - 1
    want = np.zeros((H, B), dtype=np.int64)
    for hh in range(H):
        for bb in range(B):
            want[hh, bb] = sum(1 for x in range(1, 7) if kid[hh, bb, x] < 0 or kage[hh, bb, x] < 2 * 4)
    assert free.tolist() == want.tolist() and free[0, 1] == 5 and free[1, 4] == 6     # the dead slot counts as free
    assert sorted(zip(l2.tolist(), h2.tolist(), b2.tolist(), k2.tolist())) == [(0, 1, 4, 7), (1, 0, 1, 3)]
    blk_ = np.zeros((1, 1, 6), dtype=bool)
    blk_[..., 3:] = True
    assert BF.store_room(kid, kage, 2, 4, 4, 50, blk_)[0][0, 0] == 3


# ------------------------------------------------------------------ the world runs
def _builder(mode="ce", worker="inline", Xs=None, acct=True, bneg=None, check=False, stats=False, kcheck=False,
             phase_marks=False, probe_every=0, n_cap=4096, ring=4, packer="torch", gf=None, threads=2, late="drop",
             late_wait_ms=100.0, stall_ms=0.0, batch=8, hold=0, snap="newest", arrival=True, poll_ms=1.0,
             ledger=True, retain=False, lneg=None, kslots=None):
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
                  kslots=kslots, cneg=None, bf=dict(mode=mode, worker=worker, acct=acct, bneg=bneg, n_cap=n_cap,
                                                    ring=ring, packer=packer, gf=gf, threads=threads, late=late,
                                                    late_wait_ms=late_wait_ms, stall_ms=stall_ms, batch=batch,
                                                    hold=hold, snap=snap, arrival=arrival, poll_ms=poll_ms,
                                                    ledger=ledger, retain=retain, lneg=lneg))
        be = X.FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def _run_bf(scen="timing", B=8, N=N_RUN, seed=0, setup=None, **kw):
    X.CC.configure()
    s = X.SCEN[scen]
    model, c, S, hz, tok0 = X.world(B, N, seed)
    run = _builder(check=s["check"], stats=s["stats"], kcheck=bool(s["check"]), phase_marks=s["phase_marks"],
                   probe_every=s["probe_every"], **kw)(model, c, S, hz, tok0, check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if setup is not None:                                              # tests: fake events, oracles, plants
        setup(run)
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


REPORT_KEYS = ("mode", "packer", "late_mode", "late_wait_ms", "lanes_per_entry", "items", "committed", "no_room",
               "no_room_deferred", "used_store", "unused", "pending", "bytes_sent", "bytes_used", "barrier_ms_median",
               "late_age_ms_median", "room_free_mean", "transport", "device_bytes", "item_bytes", "error", "snap_mode",
               "snap_unready_taken", "polls", "arrival")
ARRIVAL_KEYS = ("enabled", "useful_scope", "window_uses_covered", "window_uses_acct", "gil_hold_measured", "batches",
                "items_sent_early", "items_sent_late", "items_sent_nodeadline", "items_useful_store_early",
                "items_useful_store_late", "items_useful_store_nodeadline", "items_useful_store_unattributed",
                "useful_store_counted", "items_sent_counted", "useful_early_share_store", "margin_ms_useful_store",
                "margin_ms_sent", "items_sent_arrived_after_jstart", "items_useful_store_arrived_after_jstart",
                "deferred_wait_and_commit", "host")


@pytest.mark.parametrize("mode,packer,late", [("off", "torch", "drop"), ("off", "gilfree", "defer"),
                                              ("ce", "torch", "drop"), ("ce", "torch", "defer"),
                                              ("ce", "gilfree", "drop"), ("ce", "gilfree", "defer")])
def test_bf1_report_every_mode(mode, packer, late, request):
    """THE ATTRIBUTE CLASS (twice: tp in 9ca5796, lanes_e in 808bf2d): report(), stop() and bytes_device() work on a
    fresh BgFetch in EVERY mode (off / ce x torch / gilfree x drop / defer), twice in a row, before any run; every key
    the cell reads is present."""
    gf = None
    if packer == "gilfree" and mode != "off":
        gf = dict(ext=request.getfixturevalue("gfext"), cls=X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport)
    X.CC.configure()
    model, c, S, hz, tok0 = X.world(8, N_RUN, 0)
    run = _builder(mode=mode, packer=packer, late=late, worker="thread", gf=gf)(model, c, S, hz, tok0)
    bf = run.bf
    assert bf.bytes_device() > 0
    for i in range(2):
        rep = bf.report()
        bf.stop()
        assert [k for k in REPORT_KEYS if k not in rep] == [], [k for k in REPORT_KEYS if k not in rep]
        assert (rep["mode"], rep["packer"], rep["late_mode"]) == (mode, packer, late) and rep["error"] is None
        assert rep["items"] == 0 and rep["items_sent"] == 0, rep
        ar = rep["arrival"]
        assert [k for k in ARRIVAL_KEYS if k not in ar] == [] and ar["batches"] == 0, ar
        assert ar["enabled"] == (mode != "off") and rep["snap_mode"] == "newest", (ar["enabled"], rep["snap_mode"])
        live_tp = packer == "gilfree" and mode != "off" and i == 0     # the transport reports once, then is closed
        assert (rep["transport"] is not None) == live_tp, (i, rep["transport"])
    _ev("BF1 REPORT %s / %s / %s: device %d B, lanes per entry %s" % (mode, packer, late, rep["device_bytes"],
                                                                     rep["lanes_per_entry"]))


@pytest.mark.parametrize("late", ["drop", "defer"])
def test_bf1_late(late):
    """A SLOW worker (stall 200 ms per job, tests only) makes jobs late at the joint pass's start. drop: abandoned
    (late > 0, the claim holds); defer: committed at THEIR layer of the joint pass on the compute stream (deferred
    committed > 0 with a large host budget, none passed), the claim and the gates hold, the counters close."""
    ref = _plain("GUDKCD")
    ev = _run_bf("timing", worker="thread", late=late, late_wait_ms=60000.0, stall_ms=200.0, ring=2)
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    bf = ev["rd"]["bf"]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    if late == "drop":
        assert bf["late"] > 0 and not bf.get("deferred"), bf
        assert not bf.get("pipelined"), bf.get("pipelined")             # drop mode runs batch by batch (a329dc6)
    else:
        assert bf.get("deferred_committed", 0) > 0 and not bf.get("late_passed"), bf
        assert bf["deferred_committed"] == bf["deferred"], bf
    _ev("BF1 LATE %s: jobs %d, issued %d, late %d (queued %s, packing %s, passed %s), deferred %s / committed %s, "
        "no room deferred %s, host wait %s s, used %d" % (
            late, bf["jobs"], bf["issued"], bf["late"], bf.get("late_queued"), bf.get("late_packing"),
            bf.get("late_passed"), bf.get("deferred"), bf.get("deferred_committed"), bf.get("no_room_deferred"),
            bf.get("late_wait_s"), bf["used_store"] + bf["used_win"]))


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


def test_bf1_gilfree_batch(gfext):
    """THE BATCH (bf1b: the worker's cost per CALL bound it): a CERTAIN backlog (hold = the epoch's 2 L jobs: the
    worker waits for them, or for the joint pass's start; 4981010's stall 20 ms gave none: the CPU world makes 4 jobs
    per epoch and a CPU draft layer outlasts 20 ms); the worker takes the queued jobs of one epoch as one batch and
    makes ONE release of all their layers (NativePipe.submit; cap_groups 1: one chunk per item, so chunks = groups) and
    ONE commit. The claim, the gates and the counters hold; preconditions (L34): batches of several jobs, several
    non-empty jobs of several layers in one batch."""
    _plain("GUDKC")
    ref = _plain("GUDKCD")
    cls = X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport
    ev = _run_bf("timing", worker="thread", packer="gilfree", gf=dict(ext=gfext, cls=cls, cap_groups=1), threads=2,
                 late="defer", late_wait_ms=60000.0, batch=8, hold=2 * L)
    bf = ev["rd"]["bf"]
    tp = bf["transport"]
    _ev("BF1 BATCH: jobs %d in %d batches (offered max %s; done per batch -> batches %s; max %s non-empty jobs, %s "
        "layers); hold %s s; releases %d, chunks %d, groups %d; sent %d, used %d; worker ms per epoch %s" % (
            bf["jobs"], bf["batches"], bf.get("batch_in_max"), bf.get("batch_sizes"), bf.get("batch_jobs_max"),
            bf.get("batch_layers_max"),
            bf.get("hold_s"), tp["releases"], tp["chunks"], tp["groups"], bf["items_sent"],
            bf["used_store"] + bf["used_win"], bf["worker_ms_per_epoch"]))
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["error"] is None and bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    assert bf["batch_in_max"] > 1 and max(int(k) for k in bf["batch_sizes"]) > 1, (bf["batch_in_max"],
                                                                                   bf["batch_sizes"])
    assert bf["batches"] < bf["jobs"], (bf["batches"], bf["jobs"])
    assert bf["batch_jobs_max"] > 1 and bf["batch_layers_max"] > 1, (bf["batch_jobs_max"], bf["batch_layers_max"])
    assert tp["releases"] <= bf["batches"] and tp["chunks"] == tp["groups"] >= bf["items_sent"] > 0, (tp, bf)
    assert bf["used_store"] + bf["used_win"] > 0, bf


class _SlowPipe:
    """A NativePipe stand-in for the PIPELINE test: every call goes to the real pipe (the real job runs, its bytes are
    the real ones), but a job's SUBMISSION is reported delay_s after its submit (submitted, wait_submission, wait_job,
    job_done, stream_wait_job), as if the C++ worker packed and copied for that long (C 73 B 318: about 19 ms per
    batch)."""

    def __init__(self, real, delay_s):
        self._real, self._delay, self._t = real, float(delay_s), {}

    def submit(self, *args):
        seq = int(self._real.submit(*args))
        self._t[seq] = time.perf_counter() + self._delay
        return seq

    def _hold(self, seq):
        d = self._t.get(int(seq), 0.0) - time.perf_counter()
        if d > 0:
            time.sleep(d)

    @property
    def submitted(self):
        real, now = int(self._real.submitted), time.perf_counter()
        held = [s for s, t in self._t.items() if t > now and s <= real]
        return min(held) - 1 if held else real

    def wait_submission(self, seq):
        self._hold(seq)
        return self._real.wait_submission(seq)

    def wait_job(self, seq):
        self._hold(seq)
        return self._real.wait_job(seq)

    def job_done(self, seq):
        return self._t.get(int(seq), 0.0) <= time.perf_counter() and self._real.job_done(seq)

    def stream_wait_job(self, seq, stream):
        self._hold(seq)
        return self._real.stream_wait_job(seq, stream)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _slow_transport(delay_s):
    base = X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport

    class SlowTransport(base):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.npipe = _SlowPipe(self.npipe, delay_s)
    return SlowTransport


@pytest.mark.parametrize("snap", ["newest", "ready"])
def test_bf1_pipeline(snap, gfext):
    """THE PIPELINE (bf1b b2: the Python and C++ stages ran in series, 375 ms per epoch): every C++ job's submission is
    held 50 ms (_SlowPipe), batch 1 and a certain backlog (hold = the epoch's 2 L jobs, once per epoch), so batch k's
    select starts while batch k - 1's job is still pending: OVERLAPPED > 0 (the counter of batches whose select ran
    while the previous submission was pending). The claim (= L4PTFDGUDKWCD but where the bytes come from), the gates
    and the counters hold with one batch in flight. Both worker loops (snap newest: b3; snap ready: bf1r)."""
    _plain("GUDKC")
    ref = _plain("GUDKCD")
    ev = _run_bf("timing", worker="thread", packer="gilfree", gf=dict(ext=gfext, cls=_slow_transport(0.05)),
                 threads=2, late="defer", late_wait_ms=60000.0, batch=1, hold=2 * L, snap=snap)
    bf = ev["rd"]["bf"]
    _ev("BF1 PIPELINE %s: jobs %d, batches %d, pipelined %s, overlapped %s; issued %d, deferred %s, late %d; worker ms "
        "per epoch %s" % (snap, bf["jobs"], bf["batches"], bf.get("pipelined"), bf.get("overlapped"), bf["issued"],
                          bf.get("deferred"), bf["late"], bf["worker_ms_per_epoch"]))
    assert bf["snap_mode"] == snap, bf["snap_mode"]
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["error"] is None and bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    assert bf["overlapped"] > 0 and bf["pipelined"] >= bf["overlapped"], (bf["pipelined"], bf["overlapped"])
    assert bf["worker_ms_per_epoch"]["cxx_wait_s"] < bf["worker_ms_per_epoch"]["cxx_s"], bf["worker_ms_per_epoch"]
    assert bf["used_store"] + bf["used_win"] > 0, bf


class _DelayEv:
    """A fake event that completes `delay` s after its record (the GPU behind the host): query, synchronize (sleeps)
    and elapsed_time on the completion clock. t: the completion time of the last record (None: never recorded)."""

    def __init__(self, delay):
        self.delay, self.t = float(delay), None

    def record(self, stream=None):
        self.t = time.perf_counter() + self.delay

    def query(self):
        return self.t is None or time.perf_counter() >= self.t

    def synchronize(self):
        if self.t is not None:
            d = self.t - time.perf_counter()
            if d > 0:
                time.sleep(d)

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3


def _ready_oracle(viol, seen):
    """setup: the snapshot events complete 0.3 s after their record; an ORACLE on _prep (the fake event's own clock):
    every job that a batch takes without its snapshot read must have a COMPLETE snapshot event."""
    def setup(run):
        bf = run.bf
        bf.snap_ev = [_DelayEv(0.3) for _ in range(bf.nsnap)]
        orig = bf._prep

        def prep(jobs, idss=None, pseq=0):
            for i, j in enumerate(jobs):
                if idss is None or idss[i] is None:
                    seen.append(j[3])
                    if not bf.snap_ev[j[2]].query():
                        viol.append(j[3])
            return orig(jobs, idss, pseq)
        bf._prep = prep
    return setup


@pytest.mark.parametrize("snap", ["ready", "newest"])
def test_bf1_ready_only(snap, gfext):
    """(a) SNAP = READY: a batch takes only jobs whose snapshot event already completed. Snapshots complete 0.3 s after
    their record (_DelayEv: the GPU behind the host), a certain backlog (hold = the epoch's 2 L jobs), C++ submissions
    held 50 ms (_SlowPipe: the poll path). The oracle on _prep reads the fake events' own clock. ready: 0 unready jobs
    taken, polls > 0 (the worker waited for the first of the oldest snapshot and the previous C++ job), the claim,
    the gates and the counters hold. NEGATIVE CONTROL newest (the b3 loop): the oracle finds unready jobs taken (the
    plant of the test is effective, L34), and the record's snap_unready_taken counts them."""
    _plain("GUDKC")
    ref = _plain("GUDKCD")
    viol, seen = [], []
    ev = _run_bf("timing", worker="thread", packer="gilfree", gf=dict(ext=gfext, cls=_slow_transport(0.05)),
                 threads=2, late="defer", late_wait_ms=60000.0, batch=8, hold=2 * L, snap=snap,
                 setup=_ready_oracle(viol, seen))
    bf = ev["rd"]["bf"]
    _ev("BF1 READY-ONLY %s: jobs %d, batches %d (sizes %s), taken %d, unready taken %d (record %s), polls %s; worker ms "
        "per epoch %s" % (snap, bf["jobs"], bf["batches"], bf.get("batch_sizes"), len(seen), len(viol),
                          bf["snap_unready_taken"], bf["polls"], bf["worker_ms_per_epoch"]))
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["error"] is None and bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    assert len(seen) == bf["jobs"], (len(seen), bf["jobs"])                # every job passed the oracle once
    if snap == "ready":
        assert viol == [] and bf["snap_unready_taken"] == 0, (viol, bf["snap_unready_taken"])
        assert bf["polls"] > 0 and bf["worker_ms_per_epoch"]["snap_rest_s"] < 1.0, (bf["polls"],
                                                                                   bf["worker_ms_per_epoch"])
    else:                                                              # the record counts at the read (<= the oracle)
        assert len(viol) > 0 and 0 < bf["snap_unready_taken"] <= len(viol), (viol, bf["snap_unready_taken"])


class _Logged:
    """A ring of tensors (snap_h, stage_it, stage_k, stage_v) whose item access calls check(i) first (the ORACLE)."""

    def __init__(self, real, check):
        self._real, self._check = real, check

    def __getitem__(self, i):
        self._check(i)
        return self._real[i]

    def __getattr__(self, name):
        return getattr(self._real, name)


def _lifetime_oracle(viol, n, plant=None):
    """setup: snapshot events and timing events (the H2D event of a staging entry) complete 0.05 s after their record.
    ORACLES (their own clocks): SNAPSHOT: the worker reads slot s only when its event completed, the main thread
    rewrites slot s only after the worker read it; STAGING: the worker writes entry e only when its last H2D event
    completed and no deferred job holds the entry. plant "no_stage_wait" (_stage_acquire a no-op) and "no_snap_wait"
    (_read_snap without its wait): the NEGATIVE CONTROLS."""
    def setup(run):
        bf = run.bf
        bf.snap_ev = [_DelayEv(0.05) for _ in range(bf.nsnap)]
        bf._tev = lambda: _DelayEv(0.05)
        state = {}

        def snap_check(s):
            n["snap"] += 1
            if threading.current_thread().name == "bf1-worker":
                if not bf.snap_ev[s].query():
                    viol.append(("snap read before its event", s))
                state[s] = "read"
            else:
                if state.get(s) == "written":
                    viol.append(("snap rewritten before the worker read it", s))
                state[s] = "written"

        def stage_check(e):
            if threading.current_thread().name != "bf1-worker":
                return
            n["stage"] += 1
            h = bf.h2d_done[e]
            if h is not None and not h.query():
                viol.append(("staging reused before its H2D", e))
            if bf.land_busy[e]:
                viol.append(("staging reused while a deferred job holds it", e))
        bf.snap_h = _Logged(bf.snap_h, snap_check)
        for name in ("stage_it", "stage_k", "stage_v"):
            setattr(bf, name, _Logged(getattr(bf, name), stage_check))
        if plant == "no_stage_wait":
            bf._stage_acquire = lambda e, ep: None
        elif plant == "no_snap_wait":
            def read_nowait(s, key):
                ids = bf.snap_h[s].numpy().copy()
                bf._release_snap(s)
                return ids
            bf._read_snap = read_nowait
    return setup


@pytest.mark.parametrize("snap,plant", [("ready", None), ("newest", None), ("ready", "no_stage_wait"),
                                        ("newest", "no_snap_wait")])
def test_bf1_lifetimes(snap, plant):
    """(b) THE LIFETIMES under the PIPELINE (thread worker, torch packer, defer, ring 2, batch 2, a certain backlog):
    a snapshot slot is read only after its event and rewritten only after the read; a staging entry is rewritten only
    after its last H2D event and its deferred commit. The events complete 50 ms after their record (_DelayEv). No
    violation in either loop; the claim and the gates hold. NEGATIVE CONTROLS (L34): without the staging wait or the
    snapshot wait the oracles find violations."""
    ref = _plain("GUDKCD")
    viol, n = [], dict(snap=0, stage=0)
    ev = _run_bf("timing", worker="thread", late="defer", late_wait_ms=60000.0, ring=2, batch=2, hold=2 * L, snap=snap,
                 setup=_lifetime_oracle(viol, n, plant))
    bf = ev["rd"]["bf"]
    _ev("BF1 LIFETIMES %s plant %s: snapshot accesses %d, staging writes %d, violations %d %s; batches %d, deferred %s" % (
        snap, plant, n["snap"], n["stage"], len(viol), sorted(set(v[0] for v in viol)), bf["batches"],
        bf.get("deferred")))
    assert n["snap"] > 0 and n["stage"] > 0 and bf["batches"] > 2, (n, bf["batches"])
    if plant is None:
        assert viol == [], viol[:5]
        bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
        assert bad == [], bad[:10]
        gs = _gates(ev, "timing")
        assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    else:
        want = "staging reused before its H2D" if plant == "no_stage_wait" else "snap read before its event"
        assert any(v[0] == want for v in viol), viol[:5]


@pytest.mark.parametrize("mode,late,snap,packer", [("off", "drop", "newest", "torch"), ("ce", "drop", "newest", "torch"),
                                                   ("ce", "drop", "ready", "torch"), ("ce", "defer", "newest", "torch"),
                                                   ("ce", "defer", "ready", "torch"),
                                                   ("ce", "defer", "ready", "gilfree")])
def test_bf1_arrival_modes(mode, late, snap, packer, request):
    """(c) THE ARRIVAL COUNTERS in every mode (off / drop / defer x newest / ready, the gilfree packer once), thread
    worker: the block is present; off: disabled, nothing counted; ce: batches > 0, the USEFUL STORE items per (batch,
    layer) (device counts) close to used_store (review 5: the window uses are on neither side; with acct=True their
    total is window_uses_acct = used_win), the sent items of the records close to items_sent, early + late + no
    deadline = the records' items, margins and host overlap present; the claim and the gates (-BF checks the closures
    too) hold."""
    gf = None
    if packer == "gilfree":
        gf = dict(ext=request.getfixturevalue("gfext"), cls=X.W.pkg_load("r2_transport_gilfree_v3").GilfreeTransport)
    ref = _plain("GUDKCD")
    ev = (_bf("timing", mode="off") if mode == "off" else
          _run_bf("timing", worker="thread", late=late, late_wait_ms=60000.0, snap=snap, packer=packer, gf=gf,
                  threads=2))
    bf = ev["rd"]["bf"]
    ar = bf["arrival"]
    _ev("BF1 ARRIVAL %s / %s / %s / %s: %s" % (mode, late, snap, packer, {k: ar.get(k) for k in (
        "batches", "batches_deferred", "items_sent_early", "items_sent_late", "items_sent_nodeadline",
        "items_useful_store_early", "items_useful_store_late", "useful_store_counted", "window_uses_acct",
        "useful_early_share_store", "margin_ms_useful_store", "deferred_wait_and_commit", "host", "memory")}))
    assert [k for k in ARRIVAL_KEYS if k not in ar] == [], [k for k in ARRIVAL_KEYS if k not in ar]
    if mode == "off":
        assert not ar["enabled"] and ar["batches"] == 0 and ar["useful_store_counted"] == 0, ar
        return
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    recs = ev["runner"].bf.arr_b
    assert ar["enabled"] and ar["batches"] == len(recs) > 0, (ar["batches"], len(recs))
    assert ar["useful_scope"] == "store_only" and not ar["window_uses_covered"] and not ar["gil_hold_measured"], ar
    assert ar["useful_store_counted"] == bf["used_store"] > 0, (ar["useful_store_counted"], bf["used_store"])
    assert ar["window_uses_acct"] == bf["used_win"], (ar["window_uses_acct"], bf["used_win"])   # acct=True here
    assert ar["useful_store_counted"] == sum(ar["items_useful_store_" + k] for k in (
        "early", "late", "nodeadline", "unattributed"))
    assert ar["items_sent_counted"] == bf["items_sent"], (ar["items_sent_counted"], bf["items_sent"])
    assert ar["items_sent_early"] + ar["items_sent_late"] + ar["items_sent_nodeadline"] == sum(r["n"] for r in recs)
    assert ar["joint_passes"] > 0 and ar["margin_ms_sent"] is not None and ar["host"]["draft"]["windows"] > 0, ar
    assert ar["host"]["draft"]["busy_ms"] + ar["host"]["joint"]["busy_ms"] > 0, ar["host"]
    mem = ar["memory"]                                                 # the per-epoch counters are in bytes_device
    run_bf = ev["runner"].bf
    assert mem["counter_bytes_per_epoch"] == (run_bf.bl_max * run_bf.L + 1) * 8 and mem["epochs_kept"] == len(
        run_bf.arr_ep) and mem["counter_bytes_total"] == mem["epochs_kept"] * mem["counter_bytes_per_epoch"], mem
    assert bf["device_bytes"] >= mem["pbid_bytes"] + mem["counter_bytes_total"], (bf["device_bytes"], mem)
    if late == "defer":                                                # one span per committed deferred batch
        dw = ar["deferred_wait_and_commit"]
        assert dw["jobs"] == sum(1 for r in recs if r["deferred"] and r["consumed_at"] is not None), (
            dw, bf.get("deferred"))


def test_bf1_arrival_late():
    """(d) A PLANTED LATE JOB is counted late: a slow worker (stall 200 ms per job), defer, ring 2: jobs land while the
    joint pass waits at their layer, after the layer's deadline event. The report's early / late split equals an
    independent count from the raw records (the host clocks of the arrival and the deadline events). CONTROL: the
    inline worker (every job commits at its draft layer) has no late item and early items > 0."""
    ev = _run_bf("timing", worker="thread", late="defer", late_wait_ms=60000.0, stall_ms=200.0, ring=2)
    bf, run = ev["rd"]["bf"], ev["runner"]
    ar = bf["arrival"]
    late = early = 0
    for r in run.bf.arr_b:                                             # the independent count (raw host clocks)
        ta = r["arr"].t
        for l, k in r["lcnt"]:
            d = r["ep"]["jl"].get(int(l))
            if d is None:
                continue
            late += k if d.t < ta else 0
            early += k if d.t >= ta else 0
    ctl = _bf("timing")["rd"]["bf"]["arrival"]                         # inline worker, drop
    _ev("BF1 ARRIVAL LATE: stall: sent early %d / late %d (independent %d / %d), useful early %d / late %d, late ms "
        "mean %s, margins %s; control (inline): early %d, late %d" % (
            ar["items_sent_early"], ar["items_sent_late"], early, late, ar["items_useful_store_early"],
            ar["items_useful_store_late"], ar["late_ms_useful_store_mean"], ar["margin_ms_sent"],
            ctl["items_sent_early"], ctl["items_sent_late"]))
    assert ar["items_sent_late"] > 0 and (ar["items_sent_early"], ar["items_sent_late"]) == (early, late), (ar, early,
                                                                                                         late)
    assert ar["margin_ms_sent"]["min"] < 0, ar["margin_ms_sent"]
    assert ctl["items_sent_late"] == 0 and ctl["items_sent_early"] > 0, ctl
    dw = ar["deferred_wait_and_commit"]                                # deferred jobs ran: the span is recorded
    assert dw["jobs"] > 0 and dw["ms_total"] >= 0, dw


def test_bf1_arrival_store_only():
    """REVIEW 5 (2): THE TIMING CELLS' REAL COMBINATION acct=False / arrival=True, with WINDOW-ONLY consumers. The same
    inline world twice (deterministic): acct=True finds window-only uses (used_win > 0: prefetched blocks that the joint
    pass read only through its window loads); acct=False counts none (on_window returns early) and those tags become
    "unused" (unused_F = unused_T + used_win_T). The arrival metric is STORE-ONLY in both: useful_store_counted =
    used_store (the same in both runs), window_uses_covered False, window_uses_acct = used_win_T / None. So the window
    population is NOT in useful_early_share_store; the label says so."""
    _plain("GUDKC")
    _plain("GUDKCD")
    evT = _run_bf("timing", worker="inline", acct=True)
    evF = _run_bf("timing", worker="inline", acct=False)
    bT, bF = evT["rd"]["bf"], evF["rd"]["bf"]
    aT, aF = bT["arrival"], bF["arrival"]
    _ev("BF1 ARRIVAL STORE-ONLY: acct True: used store %d, window-only %d, unused %d, arrival useful store %d (share %s), "
        "window_uses_acct %s; acct False: used store %d, window %d, unused %d, arrival useful store %d (share %s), "
        "window_uses_acct %s" % (bT["used_store"], bT["used_win"], bT["unused"], aT["useful_store_counted"],
                                 aT["useful_early_share_store"], aT["window_uses_acct"], bF["used_store"],
                                 bF["used_win"], bF["unused"], aF["useful_store_counted"],
                                 aF["useful_early_share_store"], aF["window_uses_acct"]))
    assert bT["used_win"] > 0, bT["used_win"]                          # L34: window-only consumers exist here
    assert bF["used_win"] == 0 and bF["acct"] is False, (bF["used_win"], bF["acct"])
    assert bF["used_store"] == bT["used_store"] and bF["unused"] == bT["unused"] + bT["used_win"], (bT, bF)
    for a, b in ((aT, bT), (aF, bF)):
        assert a["useful_scope"] == "store_only" and not a["window_uses_covered"], a
        assert a["useful_store_counted"] == b["used_store"], (a["useful_store_counted"], b["used_store"])
    assert aT["window_uses_acct"] == bT["used_win"] and aF["window_uses_acct"] is None, (aT, aF)
    assert aT["useful_early_share_store"] == aF["useful_early_share_store"], (aT, aF)   # the window is not in it
    for ev in (evT, evF):
        gs = _gates(ev, "timing")
        assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]


class _RaiseEv:
    """A snapshot event that is never complete (query False) and RAISES the planted exception in query() ("query") or
    in synchronize() ("sync"): both are outside _prep / _finish in the ready loop (_ready_count, _wait_oldest)."""

    def __init__(self, where, exc):
        self.where, self.exc = where, exc

    def record(self, stream=None):
        return None

    def query(self):
        if self.where == "query":
            raise self.exc
        return False

    def synchronize(self):
        if self.where == "sync":
            raise self.exc

    def elapsed_time(self, other):
        return 0.0


@pytest.mark.parametrize("where", ["query", "sync"])
def test_bf1_worker_failure(where):
    """REVIEW 5 (1): FAILURE PROPAGATION from the readiness operations OUTSIDE the worker's _prep / _finish (snap
    ready, thread worker, defer): a snapshot event whose query() (or, separately, synchronize()) raises a planted
    exception. The worker records THAT exception (error_exc is the same object, error names it), stops (the thread
    ends, no retry), notifies the compute thread, which raises BFError FROM it at its next hook; no snapshot slot stays
    owned (snap_busy all False) and the flight counters close (inflight 0)."""
    planted = RuntimeError("planted %s failure" % where)
    box = {}

    def setup(run):
        bf = run.bf
        bf.snap_ev = [_RaiseEv(where, planted) for _ in range(bf.nsnap)]
        box["bf"] = bf
    with pytest.raises(_mod("r2_bf1_fetch").BFError) as ei:
        _run_bf("timing", worker="thread", late="defer", late_wait_ms=60000.0, snap="ready", setup=setup)
    bf = box["bf"]
    th = bf.thread
    if th is not None:
        th.join(10.0)
    _ev("BF1 WORKER FAILURE %s: raised %r (cause %r); error %r; worker alive %s; slots owned %d; inflight %d" % (
        where, ei.value, ei.value.__cause__, (bf.error or "")[:80], th is not None and th.is_alive(),
        sum(bf.snap_busy), bf.inflight))
    assert ei.value.__cause__ is planted, ei.value.__cause__        # the ORIGINAL exception
    assert bf.error_exc is planted and "planted %s failure" % where in bf.error, bf.error
    assert th is not None and not th.is_alive()                     # the worker stopped (no retry)
    assert not any(bf.snap_busy), bf.snap_busy                      # no slot stays owned
    assert bf.inflight == 0 and bf.q.qsize() == 0, (bf.inflight, bf.q.qsize())
    bf.stop()


# ------------------------------------------------------------------ section 16: the completed schedule and THE LEDGER
LEDGER_SUM = ("hit_cache", "hit_retained", "miss_verifier_only", "miss_req_incomplete", "miss_lost")


def _ledger_checks(ev, ref, scen="timing", claim=True):
    """The ledger's closures against INDEPENDENT counters: the keep's own per-pass record (r2_i4d_keep keep_read:
    listed = the demand, fetched = the misses, hits_total); the claim and the gates (the -BF gate checks closure_ok and
    addr_bad == 0). Returns (bf record, ledger)."""
    bf = ev["rd"]["bf"]
    lg = bf["arrival"]["ledger"]
    kp = ev["rd"]["keep"]
    tot = lg["totals"]
    assert lg["enabled"] and lg["closure_ok"] and lg["passes"] > 0, lg
    assert tot["demand"] == sum(tot[k] for k in LEDGER_SUM), tot
    assert tot["demand"] == kp["listed_total"], (tot["demand"], kp["listed_total"])          # independent: the keep
    assert tot["hit_cache"] + tot["hit_retained"] == kp["hits_total"], (tot, kp["hits_total"])
    assert tot["miss_verifier_only"] + tot["miss_req_incomplete"] + tot["miss_lost"] == kp["fetched_total"], (
        tot, kp["fetched_total"])
    assert tot["addr_bad"] == 0 and tot["addr_ok"] == tot["hit_retained"], tot      # every retained hit read at its slot
    assert tot["requested"] >= tot["completed"], tot
    if claim:
        bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
        assert bad == [], bad[:10]
        gs = _gates(ev, scen)
        assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    return bf, lg


@pytest.mark.parametrize("worker,late,snap,retain", [("inline", "drop", "newest", False),
                                                     ("thread", "defer", "newest", False),
                                                     ("thread", "complete", "ready", True)])
def test_bf1_ledger(worker, late, snap, retain):
    """SECTION 16, THE LEDGER in the arms B (inline / thread, drop / defer) and S (ready + complete + retain): the demand
    partition closes and equals the keep's own listed / fetched / hit counters (independent); every retained hit is
    read at its slot by the fork's page ids (addr_ok = hit_retained, addr_bad 0); the fetch side (requested >=
    completed > 0) and the coverage are reported; the claim and the gates hold. complete: no job passed."""
    ref = _plain("GUDKCD")
    ev = _run_bf("timing", worker=worker, late=late, late_wait_ms=60000.0, snap=snap, retain=retain)
    bf, lg = _ledger_checks(ev, ref)
    tot = lg["totals"]
    _ev("BF1 LEDGER %s / %s / %s / retain %s: %s; covered %s, retained %s, residual %s items per pass; completed share "
        "%s" % (worker, late, snap, retain, tot, lg["covered_share"], lg["retained_share"],
                lg["residual_items_per_pass"], lg["completed_share"]))
    assert tot["hit_retained"] > 0 and tot["completed"] > 0, tot    # L34: the retained path was read
    if late == "complete":
        assert not bf["late_passed"] and bf["late_mode"] == "complete", bf["late_passed"]


def test_bf1_ledger_check_rounds():
    """The ledger with RETENTION in the CHECK scenario (every round a check round): the keep's capacity proof still
    holds with the retention mask (kchk overflow 0, missing 0, dup 0) and the fresh-fetch byte check passes (bytes_bad
    0): the protected slots never displace this layer's hits, and every read byte is the host block's."""
    _plain("GUDKC", "check")
    ref = _plain("GUDKCD", "check")
    ev = _run_bf("check", worker="thread", late="complete", late_wait_ms=60000.0, snap="ready", retain=True)
    bf, lg = _ledger_checks(ev, ref, scen="check")
    ck = ev["rd"]["keep"]["check"]
    _ev("BF1 LEDGER CHECK: %s; keep check %s" % (lg["totals"], ck))
    assert ck["overflow"] == 0 and ck["missing"] == 0 and ck["dup"] == 0 and ck["bytes_bad"] == 0, ck
    assert ck["bytes_checked"] > 0 and lg["totals"]["hit_retained"] > 0, (ck, lg["totals"])


@pytest.mark.parametrize("retain", [False, True])
def test_bf1_retention_pressure(retain):
    """RETENTION under capacity pressure (kslots = the most listed blocks of one (head, request, layer) of the plain arm,
    as test_i4d.py's test_keep_pressure: the keep fits, little is spare), check scenario: without retention the keep's
    eviction may take unread retained slots of later layers; with retention they are protected up to the spare
    capacity (retain_protected) and the overflow fallback is counted (retain_overflow); in both, the capacity proof and
    the byte check hold (no stale read) and the claim holds."""
    _plain("GUDKC", "check")
    ref = _plain("GUDKCD", "check")
    em = ((_plain("D", "check")["rd"].get("union") or {}).get("extras_max"))
    assert em is not None, "no union.extras_max in the plain arm's check record"
    ks = max(1, int(em))
    ev = _run_bf("check", worker="inline", late="drop", retain=retain, kslots=ks)
    bf, lg = _ledger_checks(ev, ref, scen="check")
    tot, ck = lg["totals"], ev["rd"]["keep"]["check"]
    _ev("BF1 RETENTION kslots %d retain %s: lost_tagged %d, miss_lost %d, protected %d, overflow %d; covered %s; keep "
        "check %s" % (ks, retain, tot["lost_tagged"], tot["miss_lost"], tot["retain_protected"],
                      tot["retain_overflow"], lg["covered_share"], ck))
    assert ck["overflow"] == 0 and ck["bytes_bad"] == 0 and ck["missing"] == 0, ck
    if retain:
        assert tot["retain_protected"] + tot["retain_overflow"] > 0, tot           # L34: there was something to protect
    else:
        assert tot["retain_protected"] == 0 and tot["retain_overflow"] == 0, tot


def test_bf1_lost_retention_plant():
    """PLANT: EVICTION BEFORE THE READ (lneg "evict_retained": before every joint layer's keep, every unread retained
    slot of a LATER layer leaves the map). Lost retention is counted (lost_tagged > 0, miss_lost > 0: requested,
    completed, missing at the read); the keep refetches those blocks (a synchronous gather), so NO stale byte is read:
    the claim, the gates and the byte check (check scenario) hold."""
    _plain("GUDKC", "check")
    ref = _plain("GUDKCD", "check")
    ev = _run_bf("check", worker="inline", late="drop", lneg="evict_retained")
    bf, lg = _ledger_checks(ev, ref, scen="check")
    tot, ck = lg["totals"], ev["rd"]["keep"]["check"]
    _ev("BF1 LOST-RETENTION PLANT: %s; keep check %s" % (tot, ck))
    assert tot["lost_tagged"] > 0 and tot["miss_lost"] > 0, tot
    assert ck["bytes_bad"] == 0 and ck["bytes_checked"] > 0, ck


@pytest.mark.parametrize("late", ["complete", "defer0"])
def test_bf1_late_fetch_plant(late):
    """PLANT: A LATE FETCH (a slow worker, stall 200 ms per job). complete: the joint pass WAITS for it (no job passed;
    deferred jobs commit at their layer; their items are counted late by the arrival block, and their reads are
    retained hits); defer with a 0 ms budget: jobs that land after their layer ran are PASSED and their demand is
    counted (c) miss_req_incomplete. In both, no stale byte (the claim holds)."""
    ref = _plain("GUDKCD")
    lm, lw = ("complete", 60000.0) if late == "complete" else ("defer", 0.0)
    ev = _run_bf("timing", worker="thread", late=lm, late_wait_ms=lw, stall_ms=200.0, ring=2)
    bf, lg = _ledger_checks(ev, ref)
    tot, ar = lg["totals"], bf["arrival"]
    _ev("BF1 LATE-FETCH %s: passed %s, deferred committed %s, sent late %d; ledger %s" % (
        late, bf.get("late_passed"), bf.get("deferred_committed"), ar["items_sent_late"], tot))
    if late == "complete":
        assert not bf.get("late_passed") and bf.get("deferred_committed", 0) > 0, bf
        assert ar["items_sent_late"] > 0, ar["items_sent_late"]       # late, but completed and read retained
    else:
        assert bf.get("late_passed", 0) > 0 and tot["miss_req_incomplete"] > 0, (bf.get("late_passed"), tot)


def test_bf1_verifier_only_plant():
    """PLANT: VERIFIER-ONLY DEMAND (no draft requests anything: the draft hook is a no-op). Every missed block is
    verifier-only (nothing requested, nothing completed, nothing retained): miss_verifier_only = fetched, hit_retained
    = miss_req_incomplete = miss_lost = requested = 0."""
    ref = _plain("GUDKCD")

    def setup(run):
        run.bf.on_draft = lambda *a, **k: None
    ev = _run_bf("timing", worker="inline", late="drop", setup=setup)
    bf, lg = _ledger_checks(ev, ref, claim=False)                    # -BF needs issued > 0: not this plant's claim
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    tot = lg["totals"]
    _ev("BF1 VERIFIER-ONLY PLANT: %s" % tot)
    assert tot["requested"] == tot["completed"] == tot["hit_retained"] == 0, tot
    assert tot["miss_req_incomplete"] == tot["miss_lost"] == 0, tot
    assert tot["miss_verifier_only"] == ev["rd"]["keep"]["fetched_total"] > 0, tot


def test_bf1_consumer_address_plant():
    """PLANT: THE CONSUMER-ADDRESS CHECK (lneg "addr_redirect": the fork's page id of every retained hit points at slot
    0 instead of its slot). The check counts every redirected read (addr_bad = hit_retained > 0, addr_ok 0) and the
    wrong read changes the output (EQF fails): the check sees the address that the attention reads. CONTROL: the
    ledger tests (addr_bad 0, addr_ok = hit_retained)."""
    _plain("GUDKC", "check")
    _plain("GUDKCD", "check")
    ev = _run_bf("check", worker="inline", late="drop", lneg="addr_redirect")
    tot = ev["rd"]["bf"]["arrival"]["ledger"]["totals"]
    gs = {g[0]: g for g in _gates(ev, "check")}
    _ev("BF1 ADDRESS PLANT: addr_ok %d, addr_bad %d, hit_retained %d; EQF %s; BF %s" % (
        tot["addr_ok"], tot["addr_bad"], tot["hit_retained"], gs["L4PTFDGUDKWCDB-EQF"][2], gs["L4PTFDGUDKWCDB-BF"][2]))
    assert tot["hit_retained"] > 0 and tot["addr_bad"] == tot["hit_retained"] and tot["addr_ok"] == 0, tot
    assert not gs["L4PTFDGUDKWCDB-BF"][2], gs["L4PTFDGUDKWCDB-BF"]       # the -BF gate fails on addr_bad
    assert not gs["L4PTFDGUDKWCDB-EQF"][2], gs["L4PTFDGUDKWCDB-EQF"]     # the wrong read is caught downstream too


def test_bf1_carry():
    """A job that does not fit the batch's landing entry (n_cap 1: one item per batch) is CARRIED into the next batch,
    not dropped: carried > 0, the claim, the gates and the counters hold (torch packer, thread worker, a certain
    backlog: hold = the epoch's 2 L jobs)."""
    ref = _plain("GUDKCD")
    ev = _run_bf("timing", worker="thread", late="defer", late_wait_ms=60000.0, n_cap=1, batch=8, hold=2 * L)
    bf = ev["rd"]["bf"]
    _ev("BF1 CARRY: jobs %d, batches %d (jobs offered max %s; done per batch -> batches %s), carried %s, capped items "
        "%d, sent %d" % (bf["jobs"], bf["batches"], bf.get("batch_in_max"), bf.get("batch_sizes"), bf.get("carried"),
                         bf["items_cap"], bf["items_sent"]))
    bad = TS._diff(ref, ev, skip=DIFF_RD, tensors_skip=DIFF_T)
    assert bad == [], bad[:10]
    gs = _gates(ev, "timing")
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert bf["error"] is None and bf["jobs"] == bf["issued"] + bf["late"] + bf["empty_jobs"], bf
    assert bf["batch_in_max"] > 1, bf["batch_in_max"]                  # the backlog formed (jobs offered to a batch)
    assert bf.get("carried", 0) > 0 and bf["batch_jobs_max"] == 1, (bf.get("carried"), bf["batch_jobs_max"])


@pytest.mark.parametrize("warm", [False, True])
def test_bf1_store_x(warm, monkeypatch):
    """bf1b's two gate defects at a store X other than REF's (--bf-store 257 vs 129). COUNTS: union.cap / store_slots
    DECLARED and checked against the arm's X. COMPILE: bf_warm_joint compiles loc2 at the new X before the arm's first
    capture. NEGATIVE CONTROL (warm off, another fresh X): loc2 compiles at the first joint pass, after the draft
    captures, and COMPILE fails (the plant is effective: L34)."""
    _plain("GUDKC")
    _plain("GUDKCD")
    x0 = X.KA.store_slots(X.TOPK, X.K0)
    xs = x0 + (64 if warm else 128)                                    # a shape that no earlier run compiled
    if not warm:
        monkeypatch.setattr(_mod("r2_bf1_fetch").bf_classes(X.STF)[0], "bf_warm_joint",
                            lambda self: dict(ran=False, why="plant"))
    ev = _run_bf("timing", Xs=xs)
    gs = {g[0]: g for g in _gates(ev, "timing")}
    cg, kg = gs["L4PTFDGUDKWCDB-COMPILE"], gs["L4PTFDGUDKWCDB-COUNTS"]
    events = ev["rd"]["compile"]["events"]
    assert ev["rd"]["bf"]["X"] == xs and kg[2] and "DECLARED" in kg[3], kg
    late = [e for e in events if e.get("capture") or e.get("after_captures", 1) != 0]
    if warm:
        w = ev["runner"].meta["bf"]["warm"]
        assert w["ran"] and w["X"] == xs and "loc2" in w["compiled"], w
        assert late == [] and all(g[2] for g in gs.values() if g[1]), [g for g in gs.values() if g[1] and not g[2]]
    else:
        assert any(e["name"] == "loc2" for e in late) and not cg[2], events
    _ev("BF1 STORE X %d (warm %s): events %s; COMPILE %s; COUNTS %s" % (
        xs, warm, [(e["name"], e.get("after_captures"), e.get("warm")) for e in events], cg[2], kg[2]))


# ------------------------------------------------------------------ the cell
def test_counts_gate_bf():
    """bf1b's COUNTS defect reproduced (union.cap 256 vs 128, union.store_slots 257 vs 129 fail KC.counts_gate) and
    fixed: declared when the store differs, checked against the arm's X; every other key still compared."""
    C = _mod("r2_bf1_cell")
    ref = dict(union=dict(cap=128, store_slots=129, calls=dict(attn=3)), tent=dict(calls=dict(chain=2)))
    arm = dict(union=dict(cap=256, store_slots=257, calls=dict(attn=3)), tent=dict(calls=dict(chain=2)))
    assert not C.KC.counts_gate("B", arm, ref, C.REF, skip=("round_blocks",))[2]       # the defect
    g = C.counts_gate_bf("B", arm, ref, 257)
    assert g[2] and "DECLARED" in g[3], g
    assert not C.counts_gate_bf("B", arm, ref, 129)[2]                 # store_slots must be the arm's X
    assert not C.counts_gate_bf("B", dict(arm, union=dict(arm["union"], cap=255)), ref, 257)[2]
    assert not C.counts_gate_bf("B", dict(arm, union=dict(arm["union"], calls=dict(attn=4))), ref, 257)[2]
    assert not C.counts_gate_bf("B", dict(arm, tent=dict(calls=dict(chain=3))), ref, 257)[2]
    same = C.counts_gate_bf("B", ref, ref, 129)
    assert same[2] and "DECLARED" not in same[3], same
    assert not C.counts_gate_bf("B", arm, None, 257)[2]
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
    good = dict(mode="ce", error=None, issued=3, items=10, items_sent=10, committed=10, revived=0, no_room=0,
                used_store=6, used_win=1, unused=2, pending=1)
    assert C.bf_gate("B", good, "ce")[2]
    for k, v in (("error", "x"), ("items", 9), ("no_room", 3), ("unused", 3), ("committed", 0), ("issued", 0),
                 ("revived", 1)):
        assert not C.bf_gate("B", dict(good, **{k: v}), "ce")[2], k
    assert not C.bf_gate("B", dict(good, snap_mode="ready", snap_unready_taken=1), "ce")[2]   # SNAP ready
    assert C.bf_gate("B", dict(good, snap_mode="newest", snap_unready_taken=5), "ce")[2]
    arr = dict(enabled=True, useful_scope="store_only", useful_store_counted=6, items_sent_counted=10,
               items_useful_store_early=4, items_useful_store_late=1, items_useful_store_nodeadline=1,
               items_useful_store_unattributed=0)                      # used_store 6 (good): STORE-ONLY closures
    assert C.bf_gate("B", dict(good, arrival=arr), "ce")[2]           # ARRIVAL closures
    for k, v in (("useful_store_counted", 7), ("items_sent_counted", 9), ("items_useful_store_late", 2),
                 ("useful_scope", "store+window")):
        assert not C.bf_gate("B", dict(good, arrival=dict(arr, **{k: v})), "ce")[2], k
    assert C.bf_gate("B", dict(good, arrival=dict(arr, enabled=False, useful_store_counted=0)), "ce")[2]
    ar_ = C.parse_args(base + ["--arms", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDX,L4PTFDGUDKWCDB,"
                                         "L4PTFDGUDKWCDR"])
    assert C.bf_opt(ar_, "L4PTFDGUDKWCDR")["snap"] == "ready" and C.bf_opt(ar_, "L4PTFDGUDKWCDB")["snap"] == "newest"
    assert C.bf_opt(ar_, "L4PTFDGUDKWCDR")["mode"] == "ce" and C.bf_opt(ar_, "L4PTFDGUDKWCDR")["arrival"] is True
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
    P = C.PRED_BF1_BY_B[318]

    def want(key, val):                                                # STATE-AWARE: REPORT before the registration
        bb = P.get(key)
        return "REPORT" if bb is None else ("CONFIRMED" if bb[0] <= val <= bb[1] else "REFUTED")
    assert v["BF1-VS-X"][1] == want("BF1_vs_X", 1.1) and "1.1000" in v["BF1-VS-X"][2], v["BF1-VS-X"]
    assert v["BF1-NEW-RATIO"][1] == want("BF1_new_ratio", 0.5) and v["BF1-NEW-RATIO"][2].startswith("0.5000")
    st = C.PRED_STAMP_BF1                                              # None, or a `date -u` stamp with bands
    assert st is None or (re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", st) and P.get("BF1_vs_X") is not None), st
    sp["L4PTFDGUDKWCDR"] = dict(sp["L4PTFDGUDKWCDB"], rates=dict(median_tok_s=111.0),
                                bf=dict(arrival=dict(useful_early_share_store=0.9)))
    sp["L4PTFDGUDKWCDB"]["bf"]["worker_ms_per_epoch"] = dict(snap_rest_s=80.0)
    v = {p[0]: p for p in C.verdicts(sp, 318, True)}
    PR = C.PRED_BF1R_BY_B.get(318) or {}
    for name, key, val in (("BF1R-VS-X", "BF1R_vs_X", 1.11), ("BF1R-VS-B", "BF1R_vs_B", 111.0 / 110.0),
                           ("BF1R-USEFUL-EARLY-STORE", "BF1R_useful_early_store", 0.9), ("BF1R-SNAP-REST-MS", "BF1R_snap_rest_ms",
                                                                              80.0)):
        bb = PR.get(key)
        assert v[name][1] == ("REPORT" if bb is None else ("CONFIRMED" if bb[0] <= val <= bb[1] else "REFUTED")), v[name]
    sr = C.PRED_STAMP_BF1R
    assert sr is None or (re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", sr) and PR.get("BF1R_vs_X") is not None), sr


CELLS = {"bf1m": (10, 8, 72), "bf1a": (65, 216, 512), "bf1b": (10, 318, 512), "bf1r": (10, 318, 512),
         "bf1c": (18, 287, 512)}


def test_step_script_bf1(monkeypatch, tmp_path):
    C = _mod("r2_bf1_cell")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))         # the step script sets it ($JIT/torch_ext)
    text = (EFF / "r2_bf1_step.sh").read_text()
    assert "--bf-packer gilfree" in text and "nosi/nosi/r2_transport_gilfree_v3.py" in text
    assert "--bf-threads 14" in text and "--bf-late defer" in text and "--bf-cap-groups 2048 --bf-batch 8" in text
    assert '--bf-ring "$RING"' in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-bf1" in text and "CELLPY=benchmarks/Efficiency/r2_bf1_cell.py" in text
    assert "benchmarks/Efficiency/r2_bf1_fetch.py" in text and "^PRED_STAMP_BF1 = None$" in text   # the STOP rule
    assert "^PRED_STAMP_BF1R = None$" in text and '[ "$NEEDPRED" = 2 ] && [ "$NOPREDR" = 1 ]' in text     # bf1r's
    assert "--bf-snap newest --bf-poll-ms 1" in text and "--bf-arrival 1 --bf-ledger 1" in text
    assert "^PRED_STAMP_BF1S = None$" in text and '[ "$NEEDPRED" = 3 ] && [ "$NOPREDS" = 1 ]' in text     # bf1c's
    ls_ = "PRED_STAMP_BF1S = None" if C.PRED_STAMP_BF1S is None else 'PRED_STAMP_BF1S = "%s"' % C.PRED_STAMP_BF1S
    assert re.search("^%s$" % re.escape(ls_), (EFF / "r2_bf1_cell.py").read_text(), re.M), ls_
    lr = "PRED_STAMP_BF1R = None" if C.PRED_STAMP_BF1R is None else 'PRED_STAMP_BF1R = "%s"' % C.PRED_STAMP_BF1R
    assert re.search("^%s$" % re.escape(lr), (EFF / "r2_bf1_cell.py").read_text(), re.M), lr
    cell_src = (EFF / "r2_bf1_cell.py").read_text()                   # STATE-AWARE: the cell's line the rule reads
    line = "PRED_STAMP_BF1 = None" if C.PRED_STAMP_BF1 is None else 'PRED_STAMP_BF1 = "%s"' % C.PRED_STAMP_BF1
    assert re.search("^%s$" % re.escape(line), cell_src, re.M), line
    nostop = re.search(r"^PRED_STAMP_BF1 = None$", cell_src, re.M) is None
    assert nostop == (C.PRED_STAMP_BF1 is not None)                   # the timing cells stop exactly while None
    for cell, (pool, batch, steps) in CELLS.items():
        m = re.search(r"^\s+%s\) (.*?) ;;$" % cell, text, re.M | re.S)
        assert m, cell
        kv = dict(x.split("=", 1) for x in shlex.split(m.group(1).replace("\n", " ")))
        assert (int(kv["POOL"]), int(kv["BATCH"]), int(kv["STEPS"])) == (pool, batch, steps), (cell, kv)
        argv = ["--out", "/x", "--workload", "/w", "--batch", kv["BATCH"], "--steps", kv["STEPS"], "--pool",
                kv["POOL"], "--arms", kv["ARMS"], "--bf-store", kv["BFX"], "--bf-ncap", kv["NCAP"],
                "--check-every", kv.get("CHECK", "0"), "--require-stress", kv.get("STRESS", "0"),
                "--bf-acct", kv.get("ACCT", "0"), "--bf-packer", "gilfree", "--bf-threads", "14", "--bf-ring",
                kv["RING"], "--bf-cap-groups", "2048", "--bf-batch", "8", "--bf-late", "defer", "--bf-late-wait-ms",
                "250", "--bf-snap", "newest", "--bf-poll-ms", "1", "--bf-arrival", "1", "--bf-ledger", "1"]
        a = C.parse_args(argv)
        o = C.bf_opt(a, "L4PTFDGUDKWCDB")
        assert a.bf_packer == "gilfree" and o["late"] == "defer" and o["batch"] == 8 and o["ring"] == int(kv["RING"])
        assert o["gf"] == dict(cap_groups=2048) and o["snap"] == "newest" and o["arrival"] is True
        assert o["ledger"] is True and o["retain"] is False
        assert "L4PTFDGUDKWCDB" in a.arms and "L4PTFDGUDKWCDX" in a.arms, (cell, a.arms)
        assert ("L4PTFDGUDKWCDR" in a.arms) == (cell in ("bf1r", "bf1m", "bf1c")), (cell, a.arms)
        assert ("L4PTFDGUDKWCDS" in a.arms) == (cell in ("bf1m", "bf1c")), (cell, a.arms)
        if cell in ("bf1r", "bf1m", "bf1c"):
            assert C.bf_opt(a, "L4PTFDGUDKWCDR")["snap"] == "ready"
        if cell in ("bf1m", "bf1c"):
            s = C.bf_opt(a, "L4PTFDGUDKWCDS")
            assert (s["snap"], s["late"], s["retain"], s["ledger"]) == ("ready", "complete", True, True), s
        if cell == "bf1c":
            assert int(kv["BFX"]) == 0, kv["BFX"]                       # 129 slots at the anchor (memory)
    _ev("BF1 STEP: cells %s parse" % sorted(CELLS))
