"""CPU tests of rung I4D (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 4 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (the store refetch): the ACCOUNTING option and the arm L4PTFK =
L4PTFD with the store kept across rounds (benchmarks/Efficiency/r2_i4d_keep.py; the cell r2_i4d_cell.py; the step
script r2_i4d_step.sh). The world: tests_cpu/i4a_world.py over i3_world (the SHIPPED engine from its own source; every
CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). CPU stand-ins do not establish GPU dispatch,
ordering, timing or memory (the cells i4dm / i4dc / i4dd / i4da / i4db do).

GATE 0    test_i4d_gate0_files: the tree = I4A's acbe433 except the I4D files, ONE added declaration block in each older
          Gate-0 test, and in test_i4a.py the declaration block plus ONE changed line of test_i4a_gate0_files (its name
          scan skips the two I4D modules); nosi/ and every I4A / I3 file are acbe433's bytes; the fork is fia6's blob; the
          keep module imports torch and r2_i4a_lean only, no "triton"; no nosi module and no older cell names it.
MAP       test_keep_assign_matches_model: keep_assign = a pure Python model of the rule (the slot of every list entry,
          hits, keys, ages), step by step on random sequences (layers, passes, slot counts, id collisions across layers).
CAPACITY  test_keep_capacity_and_eviction: the eviction ORDER on a hand-made map (empty, dead oldest first, alive most
          recent last, protected never); under full pressure (every list = the slot count) no overflow, no duplicate
          key, every listed key present at its slot, the map never larger than its slots; the plant evict_used breaks it.
BYTES     test_keep_same_bytes_unit: the kept store returns the bytes of a fresh fetch (the CPU gather stand-ins, random
          host rows per layer) for every listed block; the plant layer_blind returns another layer's bytes.
ACCT      test_acct_counts_hand_made: acct_repeat on a hand-made sequence (expected counts by hand).
EQUAL     test_l4ptfk_equals_l4ptfd (B 1, B 8 rowgroup, B 8 natural; L4PTFK and L4PTFKW): the keep arm's run = L4PTFD's
          BIT FOR BIT (i2_world.same_run without read()'s "keep" and the meter's "round_blocks"); the window's contents
          after every round; the meter: L4PTFD's NEW blocks - the arm's = the keep's hits (+ the window's); the
          accounting's window loads = the union's win_host; the CHECK counters 0 (fresh-fetch byte checks on every
          layer, store and window); preconditions: hits > 0 (L4PTFKW B 8: window hits > 0).
          test_keep_chain_bytes: the fork_t records of a recorded keep run: every entry the kernel's tables show holds
          the right host bytes (i3_world.chain_bytes, slot-ordered ids).
PRESSURE  test_keep_pressure: kslots = the most listed blocks of one (head, request, layer): evictions of alive slots
          happen (precondition), and the run = L4PTFD's.
DEFAULT   test_default_path_unchanged: the cell's L4PTFD without the accounting is I4A's TentRunnerFD; TentStateFK with keep
          and acct off = L4PTFD bit for bit with the same aten op count per joint pass; with acct on the same values, and
          the repeats = an independent count from the recorded lists.
PLANTS    test_keep_plants_caught: layer_blind, skip_miss, evict_used (with pressure), win_skip (L4PTFKW) each change the
          run (not a no-op) and are CAUGHT by the cell's EQF.
COUNT     test_keep_op_count: an independent aten op counter: L4PTFK's joint pass - L4PTFD's = L x KEEP_EXTRA_OPS (+ 1 in
          the first pass), L4PTFKW's - L4PTFK's = L x KEEP_WIN_EXTRA_OPS; no host read in the keep path.
CELL      test_i4d_cell_arguments, test_i4d_cell_pure_parts, test_i4d_registration_state, test_step_script_i4d.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i4d.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import random
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
I4A_TIP = "acbe433da7f07db7f7b08791c036372d6d8db885"         # nosi-r2-i4a: rung I4A with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4d"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I4D_NEW = ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
           "benchmarks/Efficiency/r2_i4d_step.sh", T_ + "test_i4d.py")
I4D_DECL = (T_ + "test_wsa.py", T_ + "test_wsb.py", T_ + "test_wsc.py", T_ + "test_i2.py", T_ + "test_i3.py")
I4D_EDITED = {T_ + "test_i4a.py": {"test_i4a_gate0_files"}}


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4a_world")
WT = _load("wsa_tests_lib", HERE / "test_wsa.py")           # WS-A's test helpers (filters)
I3, I, W, RW, SW, SR, BC, VR, KA, LN, STF = X.I3, X.I, X.W, X.RW, X.SW, X.SR, X.BC, X.VR, X.KA, X.LN, X.STF
import r2_i4d_keep as KP          # noqa: E402  (i4a_world put benchmarks/Efficiency on sys.path)

TentStateFK, TentRunnerFK = KP.keep_classes(STF)
K, N, L, BS, TOL, TOPK = X.K0, X.N0, X.L, X.BS, X.TOL, X.TOPK
R = K + 1
BIG = VR.FORK_BIG


class Evidence(UserWarning):
    """An evidence line (the CPU template runs pytest without -s: the warnings summary carries these lines)."""


def _ev(msg: str) -> None:
    print(msg)
    warnings.warn(Evidence(msg), stacklevel=2)


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4d_cell
    return r2_i4d_cell


def _units(src: str) -> dict:
    """{name: ast.dump}: top-level functions, class methods (Class.method), the other statements (#i)."""
    out, k = {}, 0
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                out["%s.%s" % (n.name, getattr(m, "name", "#"))] = ast.dump(m)
        else:
            out["#%d" % k] = ast.dump(n)
            k += 1
    return out


# ------------------------------------------------------------------ GATE 0
def test_i4d_gate0_files():
    if not _have(I4A_TIP):
        pytest.skip("no git or no commit %s in %s" % (I4A_TIP, ROOT))
    changed = {x for x in _git("diff", "--name-only", I4A_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    allowed = set(I4D_NEW) | set(I4D_DECL) | set(I4D_EDITED)
    assert changed <= allowed, sorted(changed - allowed)                                  # (a)
    assert all((ROOT / f).is_file() for f in I4D_NEW), [f for f in I4D_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", I4A_TIP, "--", "nosi").stdout.strip() == ""        # (b) nosi: acbe433's
    base = [f for f in _git("ls-tree", "-r", "--name-only", I4A_TIP, "--", "benchmarks/Efficiency").stdout.split()
            if re.search(r"r2_i4a_|r2_i3_|fork_vllm|r2_kv_arena|r2_wsa_cell|r2_i2_cell|i3_world|i4a_world", f)]
    assert len(base) >= 9 and _git("diff", "--name-only", I4A_TIP, "--", *base).stdout.strip() == ""
    for f in I4D_DECL:                                                    # (c) one added declaration block each
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I4A_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        assert old == new, (f, sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I4A_TIP, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        assert body and all(x.startswith("+") for x in body), (f, body[:5])
        assert any("r2_i4d_keep.py" in x for x in body) and any("I4D" in x for x in body), (f, body[:5])
    for f, fns in I4D_EDITED.items():                                     # (d) test_i4a: the block + one line
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (I4A_TIP, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} == fns, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", I4A_TIP, "--", f).stdout.splitlines()
        minus = [x for x in d if x.startswith("-") and not x.startswith("---")]
        assert minus == ['-        if p.name.startswith("r2_i4a_"):'], minus
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    src = (EFF / "r2_i4d_keep.py").read_text()                          # (e) the keep module: torch + the lean module
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"torch", "r2_i4a_lean"}, mods
    assert "triton" not in src and "triton" not in (EFF / "r2_i4d_cell.py").read_text()     # NO NEW KERNEL
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (f) nobody else names it
        if p.name.startswith("r2_i4d_"):
            continue
        assert "r2_i4d" not in p.read_text(), p.name


# ------------------------------------------------------------------ a pure Python model of the keep rule
class Model:
    """THE RULE (r2_i4d_keep docstring 2) per (head, request), plain Python: slots 0 .. cap - 1 (store slot 1 + s)."""

    def __init__(self, cap, usable=None):
        self.cap = cap
        self.usable = cap if usable is None else usable
        self.key = [-1] * cap
        self.age = [KP.EMPTY_AGE] * cap

    def step(self, ids, l, t, L_, layer_blind=False):
        need = [(0 if layer_blind else l * KP.KM) + x for x in ids]
        hit_slot = {}
        for i, k in enumerate(need):
            if k in self.key:
                hit_slot[i] = self.key.index(k)
        used = set(hit_slot.values())

        def score(s):
            a = self.age[s]
            return KP.ALIVE_BASE - a if a > t - L_ else a
        free = sorted((s for s in range(self.usable) if s not in used), key=lambda s: (score(s), s))
        slots, j = [], 0
        for i, k in enumerate(need):
            if i in hit_slot:
                slots.append(hit_slot[i])
            else:
                assert j < len(free), "overflow"
                slots.append(free[j])
                j += 1
                self.key[free[j - 1]] = k
        for s in slots:
            self.age[s] = t
        return [s + 1 for s in slots], [i in hit_slot for i in range(len(need))]


def _lst(rows, cap):
    """(H, B, cap) int64 from a nested list of id lists (ascending, BIG past the end)."""
    H, B = len(rows), len(rows[0])
    out = torch.full((H, B, cap), BIG, dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            v = sorted(set(rows[h][b]))
            out[h, b, :len(v)] = torch.tensor(v, dtype=torch.int64)
    return out


def _chk():
    return {k: torch.zeros((), dtype=torch.int64) for k in KP.CHECK_KEYS}


@pytest.mark.parametrize("seed,cap,usable,L_,idmax", [(0, 8, None, 3, 14), (1, 6, 4, 4, 9), (2, 5, None, 2, 40),
                                                     (3, 12, 7, 3, 20)])
def test_keep_assign_matches_model(seed, cap, usable, L_, idmax):
    rnd = random.Random(seed)
    H, B = 2, 3
    lim = cap if usable is None else usable
    kid, kage = KP.new_map(H, B, cap + 1, "cpu")
    block = None if usable is None else (torch.arange(cap) >= usable).view(1, 1, cap)
    models = [[Model(cap, usable) for _ in range(B)] for _ in range(H)]
    chk = _chk()
    hits = misses = 0
    for g in range(7):
        for l in range(L_):
            t = g * L_ + l
            rows = [[rnd.sample(range(idmax), rnd.randint(0, lim)) for _ in range(B)] for _ in range(H)]
            lst = _lst(rows, cap)
            m = KP.keep_assign(kid, kage, lst, l, t, L_, block=block, chk=chk, big=BIG)
            for h in range(H):
                for b in range(B):
                    ids = sorted(set(rows[h][b]))
                    want_s, want_h = models[h][b].step(ids, l, t, L_)
                    n = len(ids)
                    assert m["slot"][h, b, :n].tolist() == want_s, (g, l, h, b)
                    assert m["hit"][h, b, :n].tolist() == want_h and not bool(m["hit"][h, b, n:].any())
                    assert kid[h, b, 1:].tolist() == models[h][b].key, (g, l, h, b)
                    got_age = kage[h, b, 1:].tolist()
                    assert got_age == models[h][b].age, (g, l, h, b)
                    hits += sum(want_h)
                    misses += n - sum(want_h)
    got = {k: int(v) for k, v in chk.items() if k not in ("bytes_checked", "win_checked")}
    assert got == dict(overflow=0, dup=0, missing=0, bytes_bad=0), got
    assert hits > 0 and misses > 0                                                 # preconditions (L34)


def test_keep_capacity_and_eviction():
    cap, L_ = 4, 3
    kid, kage = KP.new_map(1, 1, cap + 1, "cpu")
    # hand-made map: slot 1 empty; slot 2 dead (age 1); slot 3 alive (age 8); slot 4 alive (age 9); t = 10, L 3 ->
    # alive iff age > 7. Layer 2's list holds the key of slot 4 (a hit) and 3 new ids.
    keys = [-1, 0 * KP.KM + 5, 1 * KP.KM + 6, 2 * KP.KM + 7]
    ages = [KP.EMPTY_AGE, 1, 8, 9]
    kid[0, 0, 1:] = torch.tensor(keys)
    kage[0, 0, 1:] = torch.tensor(ages)
    lst = _lst([[[7, 20, 21]]], cap)
    m = KP.keep_assign(kid, kage, lst, 2, 10, L_, big=BIG, stats=True)
    assert m["hit"][0, 0, :3].tolist() == [True, False, False]
    assert m["slot"][0, 0, :3].tolist() == [4, 1, 2]           # the hit stays; misses: the empty, then the dead slot
    assert kid[0, 0, 1:].tolist() == [2 * KP.KM + 20, 2 * KP.KM + 21, 1 * KP.KM + 6, 2 * KP.KM + 7]
    assert int(m["victims_alive"]) == 0
    assert kage[0, 0, 1:].tolist() == [10, 10, 8, 10]                  # the hit and the misses: last use t
    # a full map, ages 5, 9, 7, 8 at t 10 (alive iff age > 7): dead oldest first (slot 1, slot 3), then alive MOST RECENT
    # first (slot 2 (9), then slot 4 (8))
    kid[0, 0, 1:] = torch.tensor([1 * KP.KM + 1, 1 * KP.KM + 2, 2 * KP.KM + 3, 2 * KP.KM + 4])
    kage[0, 0, 1:] = torch.tensor([5, 9, 7, 8])
    m = KP.keep_assign(kid, kage, _lst([[[30, 31, 32]]], cap), 0, 10, L_, big=BIG, stats=True)
    assert m["slot"][0, 0, :3].tolist() == [1, 3, 2] and int(m["victims_alive"]) == 1, m["slot"]
    # full pressure: every list = cap entries, random ids, 3 layers, 6 passes: no overflow, every key present, distinct
    rnd = random.Random(5)
    H, B, cap = 2, 4, 6
    kid, kage = KP.new_map(H, B, cap + 1, "cpu")
    chk = _chk()
    for g in range(6):
        for l in range(3):
            rows = [[rnd.sample(range(12), cap) for _ in range(B)] for _ in range(H)]
            lst = _lst(rows, cap)
            m = KP.keep_assign(kid, kage, lst, l, g * 3 + l, 3, big=BIG, chk=chk)
            assert tuple(kid.shape) == (H, B, cap + 1)                              # (i) nothing grows
            for h in range(H):
                for b in range(B):
                    occ = [k for k in kid[h, b, 1:].tolist() if k >= 0]
                    assert len(occ) == len(set(occ)) <= cap                          # (ii) distinct, at most cap
                    s = m["slot"][h, b].tolist()
                    assert sorted(s) == list(range(1, cap + 1))                      # (iii) distinct slots
                    for j, x in enumerate(sorted(set(rows[h][b]))):                  # (iv) present where read
                        assert int(kid[h, b, s[j]]) == l * KP.KM + x
    assert {k: int(v) for k, v in chk.items()} == dict(overflow=0, dup=0, missing=0, bytes_bad=0, bytes_checked=0,
                                                       win_checked=0)
    # the plant evict_used (no protection) under pressure: a hit's slot is taken by a miss (missing > 0)
    kid, kage = KP.new_map(1, 1, 3, "cpu")
    chk = _chk()
    KP.keep_assign(kid, kage, _lst([[[1, 2]]], 2), 0, 0, 2, big=BIG, chk=chk)
    KP.keep_assign(kid, kage, _lst([[[1, 3]]], 2), 0, 2, 2, big=BIG, chk=chk, kneg="evict_used")
    assert int(chk["missing"]) > 0 or int(chk["dup"]) > 0, chk


def _host(Lh, B, nblk, H, D, bs, seed):
    g = torch.Generator().manual_seed(seed)
    k = [torch.randn((B, nblk * bs, H, D), generator=g) for _ in range(Lh)]
    v = [torch.randn((B, nblk * bs, H, D), generator=g) for _ in range(Lh)]
    c = [torch.randn((B, nblk * bs, H), generator=g) for _ in range(Lh)]
    return k, v, c


@pytest.mark.parametrize("kneg", [None, "layer_blind"])
def test_keep_same_bytes_unit(kneg):
    """The kept store against a fresh fetch (the CPU gather stand-ins W.KERNELS): every listed block's K, V, bias rows."""
    H, B, D, bs, cap, Lh, nblk = 2, 3, 4, 4, 6, 3, 10
    Xs = cap + 1
    hk, hv, hc = _host(Lh, B, nblk, H, D, bs, 7)
    gk, gv = W.KERNELS["gather_k"], W.KERNELS["gather_v_bias"]
    sk, sv, sb = torch.zeros((B, Xs * bs, H, D)), torch.zeros((B, Xs * bs, H, D)), torch.zeros((B, Xs * bs, H))
    kid, kage = KP.new_map(H, B, Xs, "cpu")
    rnd = random.Random(11)
    bad = checked = hits = 0
    for g in range(6):
        for l in range(Lh):
            rows = [[rnd.sample(range(nblk), rnd.randint(1, 4)) for _ in range(B)] for _ in range(H)]
            lst = _lst(rows, cap)
            m = KP.keep_assign(kid, kage, lst, l, g * Lh + l, Lh, kneg=kneg, big=BIG)
            hx = torch.full((H, B, Xs), -1, dtype=torch.int64)
            hx.scatter_(-1, m["widx"], torch.where(m["miss"], lst, torch.tensor(-1)))
            gk(sk, hk[l], hx, bs)
            gv(sv, hv[l], sb, hc[l], hx, bs)
            fk, fv, fb = torch.zeros_like(sk), torch.zeros_like(sv), torch.zeros_like(sb)
            fx = torch.full((H, B, Xs), -1, dtype=torch.int64)
            fx[..., 1:] = torch.where(m["valid"], lst, torch.tensor(-1))
            gk(fk, hk[l], fx, bs)
            gv(fv, hv[l], fb, hc[l], fx, bs)
            hits += int((m["hit"] & m["valid"]).sum())
            for h in range(H):
                for b in range(B):
                    for j in range(int(m["valid"][h, b].sum())):
                        s = int(m["slot"][h, b, j])
                        a = slice(s * bs, (s + 1) * bs)
                        f = slice((j + 1) * bs, (j + 2) * bs)
                        ok = (torch.equal(sk[b, a, h], fk[b, f, h]) and torch.equal(sv[b, a, h], fv[b, f, h])
                              and torch.equal(sb[b, a, h], fb[b, f, h]))
                        assert torch.equal(fk[b, f, h], hk[l][b, int(lst[h, b, j]) * bs:(int(lst[h, b, j]) + 1) * bs, h])
                        bad += not ok
                        checked += 1
    _ev("I4D BYTES kneg %s: %d listed blocks checked, %d hits, %d unequal to a fresh fetch" % (kneg, checked, hits,
                                                                                               bad))
    assert checked > 0 and hits > 0                                                 # preconditions (L34)
    if kneg is None:
        assert bad == 0
    else:
        assert bad > 0                                                              # the plant is caught


def test_acct_counts_hand_made():
    cap = 4
    prev = torch.full((1, 2, cap), BIG, dtype=torch.int64)
    seq = [([[3, 5, 7], [1]], 0), ([[5, 7, 9], [1, 2]], 3), ([[9], []], 1), ([[9, 10, 11, 12], [2, 3]], 1),
           ([[], [3]], 1)]
    for rows, want in seq:
        lst = _lst([rows], cap)
        rep = KP.acct_repeat(prev, lst[:, :, :], lst < BIG)
        assert int(rep.sum()) == want, (rows, int(rep.sum()), want)
        assert torch.equal(prev, lst)
    assert KP.block_bytes(128) == 32896 and KP.block_bytes(128, H=2) == 65792     # (head, block); one slot of H 2


# ------------------------------------------------------------------ the CPU runs
def tentfk_builder(gemm="rowgroup", check=True, stats=True, vrecord=False, **kk):
    """i4a_world.tentfd_builder's builder with r2_i4d_keep's TentRunnerFK (kk: keep, acct, kneg, kslots, kcheck)."""
    fa, fk = X.WB.fake_attention_u, X.Z.fake_fork

    def build(model, cache, S, horizon, tok0, level="L4P", K=K, draft_filter=None, neg=None, tneg=None, poison=False,
              hooks=None, check_every=1, after_round=None, record=True):
        assert level == "L4P"
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(TOPK, K))
        ops = X.RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = X.RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level=level, K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, draft_filter=draft_filter, neg=neg, record=record, probe_every=0,
                          after_round=after_round)
        return TentRunnerFK(model, rs, sops, BC.HostClock(), fork=fk, t0=tok0, hooks=hooks, check_every=check_every,
                            poison=poison, tneg=tneg, union_check=check, union_stats=stats, vneg=None, fneg=None,
                            vrecord=vrecord, **kk)
    return build


def run_k(B, gemm="rowgroup", filt="periodic", vrecord=False, **kk):
    return I3.run_tentf(B, build=tentfk_builder(gemm=gemm, vrecord=vrecord, **kk), draft_filter=WT._filter(filt),
                        with_reference=True)


@functools.lru_cache(maxsize=None)
def _d(B, gemm="rowgroup", vrecord=False):
    """L4PTFD's evidence (i4a_world.tentfd_builder; check and stats on)."""
    return I3.run_tentf(B, build=X.tentfd_builder(gemm=gemm, vrecord=vrecord), draft_filter=WT._filter("periodic"),
                        with_reference=True)


@functools.lru_cache(maxsize=None)
def _k(B, gemm="rowgroup", vrecord=False, kslots=None, acct=True, kwin=False):
    return run_k(B, gemm=gemm, vrecord=vrecord, keep=True, acct=acct, kslots=kslots, kcheck=True, kwin=kwin)


def strip(ev: dict, keys=("keep", "round_blocks")) -> dict:
    out = dict(ev)
    out["rd"] = {k: v for k, v in ev["rd"].items() if k not in keys}
    return out


def _cpu(ev):
    """The cell's EQF input (r2_i4a_cell.eq_gate) from a CPU run."""
    C = _cell()
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=C.A4.acct_record(ev["rd"]),
                state=C.A4.state_checksum(r))


CASES = dict(B1=(1, "rowgroup", False), B8=(8, "rowgroup", False), B8nat=(8, "natural", False),
             B1w=(1, "rowgroup", True), B8w=(8, "rowgroup", True), B8natw=(8, "natural", True))


@pytest.mark.parametrize("case", list(CASES))
def test_l4ptfk_equals_l4ptfd(case):
    B, gemm, kwin = CASES[case]
    arm = "L4PTFKW" if kwin else "L4PTFK"
    d, k = _d(B, gemm), _k(B, gemm, kwin=kwin)
    kp, u = k["rd"]["keep"], k["rd"]["union"]
    wh = kp.get("win_hits_total", 0)
    _ev("I4D %s %s: joint passes %d; listed %d, hits %d (%.3f), fetched %d; repeat (A) %d; window loads %d, of them "
          "in the previous store (B) %d; window loads from the kept store %s; victims alive %s; check %s; NEW blocks "
          "L4PTFD %d, %s %d" % (case, arm, kp["passes"], kp["listed_total"], kp["hits_total"],
                                 kp["hits_total"] / max(kp["listed_total"], 1), kp["fetched_total"], kp["repeat_total"],
                                 kp["win_loads_total"], kp["win_prev_total"], kp.get("win_hits_total"),
                                 kp.get("victims_alive_total"), kp["check"], sum(d["rd"]["round_blocks"]), arm,
                                 sum(k["rd"]["round_blocks"])))
    assert kp["hits_total"] > 0 and kp["fetched_total"] > 0                        # preconditions (L34)
    if kwin:
        assert kp["kwin"] is True and (wh > 0 or B == 1)                           # precondition (B 8): window loads served
    bad = I.same_run(strip(d), strip(k))
    assert bad == [], bad                                                          # THE CLAIM: L4PTFD's run, bit for bit
    assert k["rd"]["count"] == [N] * B and k["bad_tokens"] == [] and k["max_logit_delta"] <= TOL
    assert k["resid"] == [] and k["host"] == []                                   # the window's contents after rounds
    # the meter: the NEW host blocks of L4PTFD minus the arm's = the keep's hits (+ the window's) (per round: >= 0)
    rb_d, rb_k = d["rd"]["round_blocks"], k["rd"]["round_blocks"]
    assert len(rb_d) == len(rb_k) and all(x >= y for x, y in zip(rb_d, rb_k))
    assert sum(rb_d) - sum(rb_k) == kp["hits_total"] + wh, (sum(rb_d), sum(rb_k), kp["hits_total"], wh)
    assert kp["passes"] == u["passes"] and kp["listed_total"] == u["extras_host"], (kp["passes"], u)
    assert kp["win_loads_total"] == u["win_host"] and kp["win_prev_total"] <= kp["win_loads_total"], u
    c = kp["check"]
    assert c["overflow"] == 0 and c["dup"] == 0 and c["missing"] == 0 and c["bytes_bad"] == 0, c
    assert c["bytes_checked"] == kp["listed_total"] > 0, c                          # every layer of every pass checked
    assert c["win_checked"] == wh, c
    assert u["dropped"] == 0 and u["vb_bad"] == 0 and u["vis_bad"] == 0 and u["chain_bad"] == 0, u
    assert kp["repeat_total"] <= kp["listed_total"]
    C = _cell()
    gs = [C.cap_gate(arm, k["rd"]), C.acct_gate(arm, k["rd"])] + ([C.keep_gate(arm, k["rd"], True)] if wh > 0 or not
                                                                 kwin else [])
    for g in gs:
        assert g[2] is True, g
    assert C.eq_gate(arm, _cpu(k), _cpu(d), N, True)[2] is True


@pytest.mark.parametrize("kwin", [False, True])
def test_keep_chain_bytes(kwin):
    k = _k(8, vrecord=True, kwin=kwin)
    log = k["runner"].vs.log
    cb = I3.chain_bytes(log, R)
    n_store = sum(1 for x in log if x["kind"] == "fork_t")
    _ev("I4D CHAIN BYTES: %d fork_t records; checked %d (chain %d); bad %s foreign %s cross %s tail %s" % (
        n_store, cb["checked"], cb["chain_checked"], cb["bad_contents"][:3], cb["foreign"][:3], cb["cross_request"][:3],
        cb["tail_slot"][:3]))
    assert cb["checked"] > 0 and cb["bad_contents"] == [] and cb["foreign"] == [] and cb["cross_request"] == []
    assert cb["tail_slot"] == []
    # precondition: a store slot that the kernel read held a block fetched in an EARLIER pass (a hit was read)
    assert k["rd"]["keep"]["hits_total"] > 0


def _most_listed(ev) -> int:
    return int(ev["rd"]["union"]["extras_max"])


def test_keep_pressure():
    d = _d(8)
    m = max(1, _most_listed(d))
    assert m < KA.store_slots(TOPK, K) - 1, m                                       # precondition: fewer map slots
    k = run_k(8, keep=True, acct=True, kslots=m, kcheck=True)
    kp = k["rd"]["keep"]
    _ev("I4D PRESSURE: map slots %d (store %d); listed %d, hits %d, victims alive %d; check %s" % (
        m, kp["X"] - 1, kp["listed_total"], kp["hits_total"], kp["victims_alive_total"], kp["check"]))
    assert kp["slots"] == m and kp["victims_alive_total"] > 0 and kp["hits_total"] > 0     # preconditions (L34)
    assert I.same_run(strip(d), strip(k)) == []
    assert kp["check"]["overflow"] == 0 and kp["check"]["bytes_bad"] == 0 and kp["check"]["missing"] == 0


def test_default_path_unchanged():
    C = _cell()
    assert C.runner_choice("L4PTFD", False, False) == ("FD", {}) and C.runner_choice("L4PTFD", False, True) == ("FD", {})
    assert C.runner_choice("L4PTFD", True, False) == ("FK", dict(keep=False, acct=True))
    assert C.runner_choice("L4PTFK", False, True) == ("FK", dict(keep=True, acct=False, kcheck=True))
    assert C.runner_choice("L4PTFKW", True, False) == ("FK", dict(keep=True, acct=True, kcheck=False, kwin=True))
    assert LN.lean_classes(STF)[1] is X.TentRunnerFD                                 # the cell's FD = I4A's class
    d = _d(8, vrecord=True)
    for acct in (False, True):
        e = I3.run_tentf(8, build=tentfk_builder(keep=False, acct=acct, vrecord=True),
                         draft_filter=WT._filter("periodic"), with_reference=True)
        assert I.same_run(strip(d), strip(e)) == [], acct                         # same values, bit for bit
        assert e["rd"]["round_blocks"] == d["rd"]["round_blocks"]                 # the same fetches
        kp = e["rd"]["keep"]
        assert kp["keep"] is False and kp["fetched_total"] == kp["listed_total"] == (
            e["rd"]["union"]["extras_host"] if acct else 0)                       # counted only with acct
        if acct:                                                                  # the repeats: an independent count
            recs = [x for x in e["runner"].vs.log if x["kind"] == "fork_t"]
            prev, per = {}, {}
            for i, x in enumerate(recs):
                g = i // L
                lst = x["lst"]
                cur = {(h, b): {int(y) for y in lst[h, b].tolist() if y < BIG}
                       for h in range(lst.shape[0]) for b in range(lst.shape[1])}
                n = sum(len(cur[hb] & prev.get(x["layer"], {}).get(hb, set())) for hb in cur)
                per[g] = per.get(g, 0) + n
                prev[x["layer"]] = cur
            want = [per[g] for g in sorted(per)]
            assert kp["repeat"] == want and sum(want) > 0, (kp["repeat"], want)      # precondition: repeats exist
            _ev("I4D ACCT: per pass listed %s, repeat %s (independent %s)" % (kp["listed"], kp["repeat"], want))
    # the op count per joint pass: TentStateFK with keep and acct off = L4PTFD's, op for op
    fd, _ = _count(None)
    fo, _ = _count(dict(keep=False, acct=False))
    assert fd["joint"] == fo["joint"] and len(fd["joint"]) > 0, (fd["joint"], fo["joint"])


@pytest.mark.parametrize("kneg,kwin", [("layer_blind", False), ("skip_miss", False), ("evict_used", False),
                                       ("win_skip", True)])
def test_keep_plants_caught(kneg, kwin):
    d = _d(8)
    kslots = max(1, _most_listed(d)) if kneg == "evict_used" else None
    p = run_k(8, keep=True, acct=True, kneg=kneg, kslots=kslots, kcheck=False, kwin=kwin)
    bad = I.same_run(strip(d), strip(p))
    n, m, ok, det = _cell().eq_gate("L4PTFK", _cpu(p), _cpu(d), N, True)
    _ev("I4D PLANT %s: differs from L4PTFD in %s; bad tokens vs SH %d; EQF: %s" % (
        kneg, [b if isinstance(b, str) else b[0] for b in bad][:8], len(p["bad_tokens"]), det))
    assert bad != [], kneg                                                         # not a no-op (L34)
    assert ok is False, (kneg, det)                                                # CAUGHT by L4PTFK-EQF


# ------------------------------------------------------------------ the op count, no host read
class PCount(X.OpCount):
    """i4a_world.OpCount that pauses inside a kernel stand-in (the stand-ins' own ops depend on the data: a GPU kernel
    is ONE launch, counted as 1 by count_joint)."""

    def __init__(self):
        super().__init__()
        self.paused = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.paused:
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


def count_joint(runner) -> dict:
    """Per JOINT pass: the aten compute ops outside the kernels + 1 per kernel call (the engine kernels of
    rs.ops.kernels and the fork). The one-row passes: i4a_world.count_one_row_ops's whole-pass count."""
    T = runner.T
    box = dict(joint=[], passes=[], cur=None)
    kern = runner.rs.ops.kernels

    def wrap(f):
        def w(*a, **k):
            c = box["cur"]
            if c is None:
                return f(*a, **k)
            c.paused += 1
            c.n += 1
            try:
                return f(*a, **k)
            finally:
                c.paused -= 1
        return w
    for name in ("diff", "gather_k", "gather_v_bias", "writeback", "pool_update", "pool_swap"):   # the block kernels
        if name in kern:
            kern[name] = wrap(kern[name])
    runner.vs.ops.fork = wrap(runner.vs.ops.fork)
    run0 = T.run_pass

    def run_pass(model, plan):
        c = PCount()
        if plan.R > 1:
            box["cur"] = c
        try:
            with c:
                return run0(model, plan)
        finally:
            (box["joint"] if plan.R > 1 else box["passes"]).append(c.n)
            box["cur"] = None
    T.run_pass = run_pass
    return box


def _count(kk, B=4):
    model, c, S, hz, tok0 = X.small_world(B=B)
    build = (X.tentfd_builder(check=False, stats=False) if kk is None else
             tentfk_builder(check=False, stats=False, **kk))
    run = build(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)
    box = count_joint(run)
    run.run(8, rounds=3)
    return box, run


def test_keep_op_count():
    bd, rd_ = _count(None)
    bk, rk = _count(dict(keep=True, acct=False))
    assert torch.equal(rd_.out, rk.out) and torch.equal(W.bits(rd_.fmax), W.bits(rk.fmax))   # the same run
    dj = [a - b for a, b in zip(bk["joint"], bd["joint"])]
    _ev("I4D COUNT: joint passes %d of L %d; aten compute ops per joint pass L4PTFD %s, L4PTFK %s; L4PTFK - L4PTFD %s "
          "(stated %d per layer: %d per pass, + 1 in the first)" % (
              len(bd["joint"]), L, bd["joint"], bk["joint"], dj, KP.KEEP_EXTRA_OPS, L * KP.KEEP_EXTRA_OPS))
    assert len(dj) > 1 and len(bk["joint"]) == len(bd["joint"])
    assert dj[0] == L * KP.KEEP_EXTRA_OPS + 1 and all(x == L * KP.KEEP_EXTRA_OPS for x in dj[1:]), dj
    assert bd["passes"] == bk["passes"]                                             # the one-row passes: unchanged
    bw, rw = _count(dict(keep=True, acct=False, kwin=True))
    assert torch.equal(rd_.out, rw.out)
    dw = [a - b for a, b in zip(bw["joint"], bk["joint"])]
    _ev("I4D COUNT kwin: L4PTFKW - L4PTFK per joint pass %s (stated %d per layer)" % (dw, KP.KEEP_WIN_EXTRA_OPS))
    assert all(x == L * KP.KEEP_WIN_EXTRA_OPS for x in dw), dw
    model, c, S, hz, tok0 = X.small_world(B=2)
    run = tentfk_builder(check=False, stats=False, keep=True, acct=True, kwin=True)(model, c, S, hz, tok0, level="L4P",
                                                                                    hooks=None, check_every=0,
                                                                                    record=False)
    with RW.SyncTrap(Path(KP.__file__)) as trap:
        run.run(8, rounds=3)
    assert trap.hits == [], trap.hits[:10]                                          # no host read in the keep path
    assert run.T.kpass >= 1 and run.read()["keep"]["listed_total"] >= 0
    with pytest.raises(SR.SpecError):
        tentfk_builder(keep=True, kneg="nope")(*X.small_world(B=2), level="L4P", hooks=None, check_every=0,
                                                record=False)
    with pytest.raises(SR.SpecError):
        tentfk_builder(keep=False, kcheck=True)(*X.small_world(B=2), level="L4P", hooks=None, check_every=0,
                                                 record=False)


# ------------------------------------------------------------------ CELL
def test_i4d_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.store_acct, a.fork_store, a.check_every, a.eq_must) == (
        ["L4PTFD", "L4PTFK", "L4PTFKW"], 216, 512, 65, 3, 0, 0, 0, 0)
    assert C.parse_args(base + ["--arms", "L4PTFD,L4PTFKW", "--eq-must", "1"]).arms == ["L4PTFD", "L4PTFKW"]
    for bad in (["--arms", "L4PTFK,L4PTFD"], ["--arms", "L4PTFKW,L4PTFK"], ["--arms", "L4PTF"],
                ["--arms", "L4PTFD", "--eq-must", "1"], ["--arms", "L4PTFK", "--eq-must", "1"],
                ["--spec", "4"], ["--fork-store", "100"], ["--arms", "L4PTFK,L4PTFK"], ["--pool", "0"],
                ["--steps", "4"], ["--check-every", "1", "--batch", "216"], ["--store-acct", "2"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + ["--batch", "8", "--steps", "72", "--pool", "10", "--check-every", "1", "--store-acct",
                             "1", "--require-stress", "1", "--eq-must", "1"])
    assert (a.check_every, a.require_stress, a.eq_must, a.store_acct) == (1, 1, 1, 1)
    a = C.parse_args(base + ["--arms", "L4PTFD", "--store-acct", "1", "--steps", "128"])
    assert a.arms == ["L4PTFD"]
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused


def test_i4d_cell_pure_parts():
    C = _cell()
    d, k = _d(8), _k(8)
    n, m, ok, det = C.eq_gate("L4PTFK", _cpu(k), _cpu(d), N, True)
    assert (n, m, ok) == ("L4PTFK-EQF", True, True) and "L4PTFD's" in det, det
    planted = _cpu(k)
    planted["out"] = planted["out"].clone()
    planted["out"][0, 3] += 1                                          # CONTROL: one token changed
    assert C.eq_gate("L4PTFK", planted, _cpu(d), N, False)[2] is False
    planted = _cpu(k)
    planted["state"] = planted["state"][:-1] + [planted["state"][-1] + 1]
    assert C.eq_gate("L4PTFK", planted, _cpu(d), N, False)[2] is False   # CONTROL: one state checksum changed
    assert C.eq_gate("L4PTFK", _cpu(k), None, N, True)[2] is False
    rd = k["rd"]
    assert C.keep_gate("L4PTFK", rd, True)[2] is True and C.cap_gate("L4PTFK", rd)[2] is True
    assert C.keep_gate("L4PTFK", d["rd"], False)[2] is False           # CONTROL: L4PTFD has no keep record
    no_hit = dict(rd, keep=dict(rd["keep"], fetched_total=rd["keep"]["listed_total"]))
    assert C.keep_gate("L4PTFK", no_hit, False)[2] is False            # CONTROL: no hit
    bad_cap = dict(rd, keep=dict(rd["keep"], check=dict(rd["keep"]["check"], overflow=1)))
    assert C.cap_gate("L4PTFK", bad_cap)[2] is False                   # CONTROL: an overflow
    assert C.acct_gate("L4PTFK", rd)[2] is True and C.acct_gate("L4PTFD", d["rd"])[2] is False
    kd = I3.run_tentf(8, build=tentfk_builder(keep=False, acct=True), draft_filter=WT._filter("periodic"),
                      with_reference=False)["rd"]["keep"]
    g = C.acct_eq_gate(kd, rd["keep"])
    assert g[0] == "ACCT-EQ-L4PTFK" and g[2] is True, g                 # both arms list the same blocks
    assert C.acct_eq_gate(kd, _k(8, kwin=True)["rd"]["keep"], "L4PTFKW")[2] is True     # the same window loads too
    assert C.acct_eq_gate(kd, dict(rd["keep"], repeat=[0] * len(rd["keep"]["repeat"])))[2] is (
        sum(rd["keep"]["repeat"]) == 0)                                 # CONTROL: other repeats
    assert C.acct_eq_gate(kd, dict(rd["keep"], win_loads=[0] * len(rd["keep"]["win_loads"])))[2] is False
    s = C.acct_summary(rd["keep"], KP.block_bytes(128))
    l1 = rd["keep"]["listed"][1:]
    assert s["passes"] == rd["keep"]["passes"] and s["listed_pass_after_first"] == round(sum(l1) / len(l1), 2)
    assert s["repeat_share_after_first"] == round(sum(rd["keep"]["repeat"][1:]) / sum(l1), 6)
    assert 0 < s["hit_share_after_first"] <= 1 and C.acct_summary(None, 1) is None
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    rec = lambda tok, jt, nb, wt: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=16.0, joint=jt),  # noqa
                                       kv_at_verify=dict(new_blocks_per_round=nb, wait_ms_median=wt),
                                       tokens=dict(equal_fraction=0.4, first=(1, 1)), vs_strongest=0.5,
                                       gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0))
    vv = C.verdicts(sh, dict(L4PTFD=rec(400.0, 30.0, 100.0, 20.0)), 216)
    assert vv[-2:] == [("L4PTFK-RUN", "NOT_RUN", "the arm did not run"), ("L4PTFKW-RUN", "NOT_RUN",
                                                                          "the arm did not run")], vv
    saved = {b: dict(C.PRED_I4D_BY_B[b]) for b in C.PRED_I4D_BY_B}        # the REPORT check runs on an all-None block
    try:
        for b in C.PRED_I4D_BY_B:
            C.PRED_I4D_BY_B[b].update({x: None for x in C.PRED_KEYS})
        v = {x[0]: x for x in C.verdicts(sh, dict(L4PTFD=rec(400.0, 30.0, 100.0, 20.0),
                                                  L4PTFK=rec(420.0, 27.0, 80.0, 17.0),
                                                  L4PTFKW=rec(440.0, 24.0, 60.0, 14.0)), 216,
                                         eqf=dict(L4PTFK=True, L4PTFKW=True),
                                         acct=dict(L4PTFD=dict(acct=True, repeat_share_after_first=0.5,
                                                               listed_pass_after_first=10.0,
                                                               win_prev_share_after_first=0.25,
                                                               win_prev_pass_after_first=7.0),
                                                   L4PTFK=dict(keep=True, hit_share_after_first=0.4),
                                                   L4PTFKW=dict(keep=True, win_hits_pass_after_first=6.0)))}
    finally:
        for b in saved:
            C.PRED_I4D_BY_B[b].clear()
            C.PRED_I4D_BY_B[b].update(saved[b])
    assert all(x[1] == "REPORT" for x in v.values()), v                # registration None: every key REPORT
    assert "1.0500" in v["I4D-VS-I4A"][2] and "3.0000" in v["I4D-JOINT-SAVING"][2] and "0.2000" in v[
        "I4D-NEW-SAVING"][2] and "3.0000" in v["I4D-WAIT-SAVING"][2] and "0.5000" in v["I4D-REPEAT-SHARE"][2], v
    assert "1.1000" in v["I4DW-VS-I4A"][2] and "6.0000" in v["I4DW-JOINT-SAVING"][2] and "0.4000" in v[
        "I4DW-NEW-SAVING"][2] and "0.2500" in v["I4D-WIN-PREV-SHARE"][2] and "6.0000" in v[
        "I4DW-WIN-HITS-PASS"][2], v
    per = ("RATIO", "VS-I4A", "JOINT-SAVING", "WAIT-SAVING", "NEW-SAVING", "PEAK-GB", "AGREE", "EQ-I4A", "ABOVE-I4A",
           "VS-STRONGEST")
    assert set(v) == ({"I4A-RATIO", "I4D-REPEAT-SHARE", "I4D-LISTED-PASS", "I4D-WIN-PREV-SHARE", "I4D-WIN-PREV-PASS",
                       "I4D-HIT-SHARE", "I4DW-WIN-HITS-PASS"} | {"I4D-" + x for x in per}
                      | {"I4DW-" + x for x in per}), sorted(v)


def test_i4d_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4D is None:                                        # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4D_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4D)
    assert set(C.PRED_I4D_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4D_BY_B.values())
    src = (EFF / "r2_i4d_cell.py").read_text()
    assert ("\nPRED_STAMP_I4D = None\n" in src) == (C.PRED_STAMP_I4D is None)  # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4d_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("i4dm", 10, 8, 72, "L4PTFD,L4PTFK,L4PTFKW", 1, 1, 1, 1, 40, "100G"),
         ("i4dc", 65, 216, 128, "L4PTFD", 1, 0, 0, 0, 35, "240G"),
         ("i4dd", 10, 318, 128, "L4PTFD", 1, 0, 0, 0, 45, "340G"),
         ("i4da", 65, 216, 512, "L4PTFD,L4PTFK,L4PTFKW", 0, 0, 0, 0, 80, "240G"),
         ("i4db", 10, 318, 512, "L4PTFD,L4PTFK,L4PTFKW", 0, 0, 0, 0, 110, "340G"))


def test_step_script_i4d():
    src = (EFF / "r2_i4d_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "benchmarks/Efficiency/r2_i4d_keep.py" in src and "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4am", "i4aa", "i4d", "i4dx"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4D = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4d_cell.py" % sha).stdout
    for cell, pool, batch, steps, arms, acct, check, stress, eq, smin, mem in CELLS:
        r = _step(cell, sha)
        if cell in ("i4da", "i4db") and not registered:               # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4D is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.returncode, r.stderr[-400:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 fork-store 0 acct %d check %d stress %d eq %d "
                "STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, arms, steps, acct, check, stress, eq, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--store-acct %d" % acct, "--fork-store 0", "--prefill-hook nocopy", "--tu-gemm natural",
                     "--check-every %d" % check, "--require-stress %d" % stress, "--eq-must %d" % eq,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4d_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i4d_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms %s " % arms.replace(",", r"\\?,"), r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--store-acct", str(acct), "--check-every",
                                str(check), "--require-stress", str(stress), "--eq-must", str(eq)])
        assert a.arms == arms.split(",")                                # every cell's arguments parse
