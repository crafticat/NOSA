"""CPU tests of the SHARED ready-KV read attention (nosi/nosi/bi_paged_attention_shared.py; docs/R2_SHAREDKV_PLAN.md)
against piece 3 v2 (nosi/nosi/bi_paged_attention.py, the pinned blob 39ed5e18 of 7c826c1, brought into this tree
unchanged). The torch twins run here; the Triton kernels run under TRITON_INTERPRET=1 in a FRESH process (Triton reads
the flag at its first import), skipped when triton is not installed. L36: the interpreter is not the GPU compiler, so
these tests show the ORDER and the INDEXING, not the tensor-core bits; the GPU cell (the plan) decides those.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_bi_paged_attention_shared.py

(compute node only.)
"""
from __future__ import annotations

import ast
import builtins
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = ROOT / "nosi/nosi/bi_paged_attention.py"
SK_PATH = ROOT / "nosi/nosi/bi_paged_attention_shared.py"
PIECE3_BLOB = "39ed5e1894ca1662f92e384a10e01a4eaab99042"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bi = _load(BASE_PATH, "piece3_base_under_test")
sk = _load(SK_PATH, "sharedkv_under_test")
PAGE = bi.PAGE
PPS = bi.PAGES_PER_SPLIT


# ---------------------------------------------------------------------------
# cases: requests with 1 draft + 0..3 verify rows, each row a sorted selection of its request's blocks
# ---------------------------------------------------------------------------
def make_group_case(seed, overlap="half", n_req=4, rows_per_req=None, Hkv=2, G=16, D=32, P=20, ring=False,
                    dtype=torch.float32, shuffle=True):
    """overlap: 'same' (every row of a request lists the same blocks; 100%), 'half' (a common core of about half the
    pages plus own pages), 'disjoint' (no block listed by two rows of a request; 0%), 'random' (independent picks).
    ring: each row's LAST block is read from a per-row ring slot (S_hist + row % 2) instead of its block's slot (the
    tick's tail pages: the same slot may hold another block for another row: (key, slot) pairs, not slots, are
    shared). Rows of all requests are shuffled (the grouping is by seq_idx, not adjacency). Returns the case dict with
    page_key and the request of each row."""
    g = torch.Generator().manual_seed(5000 + seed)
    if rows_per_req is None:
        rows_per_req = [1 + int(torch.randint(0, 4, (1,), generator=g)) for _ in range(n_req)]
    n_req = len(rows_per_req)
    R = sum(rows_per_req)
    pool = 4 * P                                                    # block ids 0 .. pool - 1 per request
    S_hist = pool
    S = S_hist + 2
    q = torch.randn(R, Hkv * G, D, generator=g).to(dtype)
    k = torch.randn(n_req, S * PAGE, Hkv, D, generator=g).to(dtype)
    v = torch.randn(n_req, S * PAGE, Hkv, D, generator=g).to(dtype)
    bias = (torch.rand(n_req, S * PAGE, Hkv, generator=g) - 0.5).to(dtype)
    req_of = []
    for b, c in enumerate(rows_per_req):
        req_of += [b] * c
    order = torch.randperm(R, generator=g).tolist() if shuffle else list(range(R))
    req_of = [req_of[i] for i in order]
    slot_of = [torch.randperm(S_hist, generator=g).tolist() for _ in range(n_req)]   # block -> slot per request
    table = torch.full((R, Hkv, P), -1, dtype=torch.int32)
    key = torch.full((R, Hkv, P), -1, dtype=torch.int32)
    npg = torch.zeros(R, Hkv, dtype=torch.int32)
    rows_of = {}
    for r, b in enumerate(req_of):
        rows_of.setdefault(b, []).append(r)
    for b, rows in rows_of.items():
        for h in range(Hkv):
            perm = torch.randperm(pool, generator=g).tolist()
            core = perm[:P]
            for j, r in enumerate(rows):
                if overlap == "same":
                    blocks = core
                elif overlap == "half":
                    half = set(core[:P // 2])
                    own = [x for x in torch.randperm(pool, generator=g).tolist() if x not in half]
                    blocks = core[:P // 2] + own[:P - P // 2]
                elif overlap == "disjoint":
                    blocks = perm[j * (pool // 4):(j + 1) * (pool // 4)][:P]
                    assert len(rows) <= 4, "disjoint: at most 4 rows per request"
                else:
                    blocks = torch.randperm(pool, generator=g)[:P].tolist()
                if overlap == "same":
                    n = P - (h % 2) * 3                     # the same list for every row of the request
                else:
                    n = int(torch.randint(P // 2, P + 1, (1,), generator=g))
                n = max(1, min(n, len(blocks)))
                blocks = sorted(blocks)[:n]
                slots = [slot_of[b][x] for x in blocks]
                if ring:
                    slots[-1] = S_hist + (r % 2)
                table[r, h, :n] = torch.tensor(slots, dtype=torch.int32)
                key[r, h, :n] = torch.tensor(blocks, dtype=torch.int32)
                npg[r, h] = n
    last_len = torch.randint(1, PAGE + 1, (R,), generator=g).to(torch.int32)
    return dict(q=q, k=k, v=v, bias=bias, seq_idx=torch.tensor(req_of, dtype=torch.int32), table=table.contiguous(),
                num_pages=npg.contiguous(), last_len=last_len, key=key.contiguous(), rows_of=rows_of)


def args_of(c):
    return (c["q"], c["k"], c["v"], c["bias"], c["seq_idx"], c["table"], c["num_pages"], c["last_len"])


def bits_equal(a, b):
    iv = torch.int16 if a.element_size() == 2 else torch.int32
    return a.shape == b.shape and torch.equal(a.view(iv), b.view(iv))


def split_tiles_oracle(c, rows_max, pps=PPS):
    """INDEPENDENT of build_union: groups by seq_idx (ascending row index, chunks of rows_max); per (group, head,
    split k) the distinct (key, slot) pairs at the rows' list positions [k pps, (k + 1) pps). Returns (split-mode
    tiles = the sum over k of those set sizes, whole-mode tiles = the union per (group, head), the per-row sum)."""
    tab, key, npg = c["table"].tolist(), c["key"].tolist(), c["num_pages"].tolist()
    groups = []
    for b in sorted(set(c["seq_idx"].tolist())):
        rows = [r for r in range(len(tab)) if int(c["seq_idx"][r]) == b]
        groups += [rows[i:i + rows_max] for i in range(0, len(rows), rows_max)]
    split = whole = total = 0
    for rows in groups:
        for h in range(len(tab[0])):
            uni = set()
            kmax = max(-(-npg[r][h] // pps) for r in rows)
            for kk in range(kmax):
                s = set()
                for r in rows:
                    for i in range(kk * pps, min((kk + 1) * pps, npg[r][h])):
                        s.add((key[r][h][i], tab[r][h][i]))
                split += len(s)
                uni |= s
            whole += len(uni)
            total += sum(npg[r][h] for r in rows)
    return split, whole, total


CASES = [  # (seed, overlap, rows per request, ring)
    (1, "same", [4, 4, 2], False),
    (2, "half", [4, 3, 1, 2], False),
    (3, "disjoint", [4, 2, 3], False),
    (4, "random", [1, 2, 3, 4], False),
    (5, "half", [4, 4, 4, 1], True),
    (6, "half", [6, 1], False),                     # 6 rows of one request: two groups (rows_max 4)
]


# ---------------------------------------------------------------------------
# (1) the builder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,overlap,rpr,ring", CASES)
@pytest.mark.parametrize("whole", [False, True])
def test_build_union_invariants(seed, overlap, rpr, ring, whole):
    """Groups = rows of one seq_idx in ascending row index, chunks of rows_max; the union = the distinct (key, slot)
    pairs in ascending order; rowpos and uidx inverse to each other; every row's list is a subsequence of the union
    (uidx strictly ascending); the split windows hold every row's split-k pages; whole mode = (0, ucnt)."""
    c = make_group_case(seed, overlap, rows_per_req=rpr, ring=ring)
    sk.validate_keys(c["table"], c["num_pages"], c["key"])
    NR = 4
    un = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], NR, PPS, whole)
    grow, ut, ucnt, rowpos, win, uidx = (un[x].tolist() for x in ("grow", "ut", "ucnt", "rowpos", "win", "uidx"))
    tab, key, npg = c["table"].tolist(), c["key"].tolist(), c["num_pages"].tolist()
    R, Hkv = len(tab), len(tab[0])
    exp_groups = []
    for b in sorted(set(c["seq_idx"].tolist())):
        rows = [r for r in range(R) if int(c["seq_idx"][r]) == b]
        exp_groups += [rows[i:i + NR] for i in range(0, len(rows), NR)]
    got = [[r for r in grow[g] if r >= 0] for g in range(R)]
    assert sorted(x for x in got if x) == sorted(exp_groups)
    assert all(not x for x in got[len(exp_groups):])
    for g, rows in enumerate(got):
        for h in range(Hkv):
            pairs = sorted({(key[r][h][i], tab[r][h][i]) for r in rows for i in range(npg[r][h])})
            assert ucnt[g][h] == len(pairs)
            assert ut[g][h][:len(pairs)] == [s for _, s in pairs]
            for j, r in enumerate(rows):
                ux = uidx[g][h][j][:npg[r][h]]
                assert all(b > a for a, b in zip(ux, ux[1:])), "a row's list is not a subsequence of the union"
                for i, u in enumerate(ux):
                    assert pairs[u] == (key[r][h][i], tab[r][h][i]) and rowpos[g][h][j][u] == i
                assert sum(1 for x in rowpos[g][h][j] if x >= 0) == npg[r][h]
            if whole:
                assert win[g][h] == [[0, len(pairs)]]
            else:
                for kk, (u0, u1) in enumerate(win[g][h]):
                    need = [uidx[g][h][j][i] for j, r in enumerate(rows)
                            for i in range(kk * PPS, min((kk + 1) * PPS, npg[r][h]))]
                    if need:
                        assert u0 == min(need) and u1 == max(need) + 1
                    else:
                        assert u0 == u1 == 0


# ---------------------------------------------------------------------------
# (2) the twin: bitwise equal to piece 3's twin, per row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,overlap,rpr,ring", CASES)
@pytest.mark.parametrize("mode", ["split", "whole"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_twin_bitwise_equals_piece3_twin(seed, overlap, rpr, ring, mode, dtype):
    """(a) The shared twin runs EVERY row on EVERY union page of its window (non-member pages = the identity step)
    and is bitwise equal to piece 3's twin for every row. Preconditions (else vacuous): some group has >= 2 rows,
    some row has >= 2 splits, and (overlap != disjoint) some group shares a page; for 'half' / 'random' some page
    sits in different splits of two rows (misaligned split boundaries)."""
    c = make_group_case(seed, overlap, rows_per_req=rpr, ring=ring, dtype=dtype)
    assert max(rpr) >= 2 and int(c["num_pages"].max()) > PPS
    split, whole, total = split_tiles_oracle(c, 4)
    if overlap != "disjoint":
        assert whole < total, "precondition: some page is listed by two rows"
    else:
        assert whole == total
    if overlap in ("half", "random"):
        assert split > whole, "precondition: some shared page is in different splits of two rows"
    ref = bi.bi_paged_decode_reference(*args_of(c))
    for nr in (4, 2):
        out = sk.bi_paged_decode_shared_reference(bi, *args_of(c), c["key"], rows_max=nr, mode=mode)
        bad = [r for r in range(ref.shape[0]) if not bits_equal(out[r], ref[r])]
        assert not bad, (nr, bad)


def test_twin_rows_max_one_is_piece3_twin():
    c = make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    out = sk.bi_paged_decode_shared_reference(bi, *args_of(c), c["key"], rows_max=1)
    assert bits_equal(out, bi.bi_paged_decode_reference(*args_of(c)))


# ---------------------------------------------------------------------------
# (3) negative controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["split", "whole"])
def test_negative_drop_one_union_page_is_caught(mode):
    """(b) plant_drop_page: no row processes one union page. PRECONDITION (L34): the dropped page is listed by some
    row (its rowpos column has an entry >= 0) and the plant CHANGES the twin's output on these inputs (else it is a
    no-op, reported, not counted). Then: exactly the rows that list the page differ from piece 3's twin; every other
    row stays bitwise equal (the plant is local)."""
    c = make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    whole = mode == "whole"
    un = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, PPS, whole)
    g, h = 0, 0
    u = int(un["ucnt"][g, h]) // 2                                       # a middle page: shared or not
    listers = [int(un["grow"][g, j]) for j in range(4) if int(un["rowpos"][g, h, j, u]) >= 0]
    assert listers, "precondition: the dropped page is selected by some row"
    planted = sk.plant_drop_page(un, g, h, u)
    ref = bi.bi_paged_decode_reference(*args_of(c))
    out = sk.bi_paged_decode_shared_reference(bi, *args_of(c), c["key"], rows_max=4, mode=mode, union=planted)
    changed = [r for r in range(ref.shape[0]) if not bits_equal(out[r], ref[r])]
    assert changed, "the plant was a NO-OP on these inputs (not caught, not counted)"
    assert set(changed) == set(listers), (changed, listers)
    # the device default (u None = the last union page) also plants a listed page
    last = sk.plant_drop_page(un, g, h)
    assert bool((last["rowpos"][g, h] != un["rowpos"][g, h]).any())
    # [review MINOR-12] L34 on the EXACT variant the GPU uses (u None): it changes the output on these inputs, only on
    # the rows that list the last union page
    ul = int(un["ucnt"][g, h]) - 1
    listers_l = [int(un["grow"][g, j]) for j in range(4) if int(un["rowpos"][g, h, j, ul]) >= 0]
    out_l = sk.bi_paged_decode_shared_reference(bi, *args_of(c), c["key"], rows_max=4, mode=mode, union=last)
    changed_l = [r for r in range(ref.shape[0]) if not bits_equal(out_l[r], ref[r])]
    assert listers_l and changed_l, "the default plant was a NO-OP on these inputs"
    assert set(changed_l) == set(listers_l), (changed_l, listers_l)


def test_negative_union_order_must_follow_the_keys():
    """Ordering the union by SLOT instead of (key, slot) changes some row's reduction order: caught (bitwise), with
    the precondition that slot order and key order differ for some listed pair."""
    c = make_group_case(4, "random", rows_per_req=[1, 2, 3, 4])
    bad_key = c["table"].clone()                                         # key := slot (the slot order)
    tab, key, npg = c["table"], c["key"], c["num_pages"]
    differs = any(sorted(tab[r, h, :int(npg[r, h])].tolist()) != tab[r, h, :int(npg[r, h])].tolist()
                  for r in range(tab.shape[0]) for h in range(tab.shape[1]))
    assert differs, "precondition: some row's slots are not ascending in its key order"
    un = sk.build_union(c["seq_idx"], tab, npg, bad_key, 4, PPS, False)
    ref = bi.bi_paged_decode_reference(*args_of(c))
    try:
        out = sk.bi_paged_decode_shared_reference(bi, *args_of(c), bad_key, rows_max=4, union=un)
    except RuntimeError:
        return                                                           # a missing partial: also caught
    assert any(not bits_equal(out[r], ref[r]) for r in range(ref.shape[0]))


# ---------------------------------------------------------------------------
# (4) the byte model: tiles loaded = the union (whole), sum over splits of the split unions (split), never the sum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,overlap,rpr,ring", CASES)
def test_twin_loads_union_not_sum(seed, overlap, rpr, ring):
    """(c) The twin counts its tile loads exactly as the kernel's COUNT counter does. whole: tiles == the union per
    (group, head) (independent oracle), < the per-row sum whenever rows share a page; split: tiles == the sum over k
    of the split-k unions (between the union and the sum; == the union when the lists are the same)."""
    c = make_group_case(seed, overlap, rows_per_req=rpr, ring=ring)
    split, whole, total = split_tiles_oracle(c, 4)
    for mode, want in (("whole", whole), ("split", split)):
        loads = [0, 0]
        sk.bi_paged_decode_shared_reference(bi, *args_of(c), c["key"], rows_max=4, mode=mode, loads=loads)
        assert loads[0] == want, (mode, loads, want, whole, total)
    assert whole <= split <= total
    if overlap == "same":
        assert split == whole < total
    if overlap == "disjoint":
        assert split == whole == total


# ---------------------------------------------------------------------------
# (5) the kernel under the interpreter (fresh process)
# ---------------------------------------------------------------------------
INTERP_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton imported before the flag was read"
sys.dont_write_bytecode = True
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bi = load(sys.argv[1], "bi_interp")
sk = load(sys.argv[2], "sk_interp")
t = load(sys.argv[3], "sk_tests")
fails = []
ran = 0
for seed, overlap, rpr, ring in t.CASES:
    c = t.make_group_case(seed, overlap, rows_per_req=rpr, ring=ring)
    ref = bi.bi_paged_decode(*t.args_of(c))
    split, whole, total = t.split_tiles_oracle(c, 4)
    for mode in ("split", "whole"):
        cnt = torch.zeros(2, dtype=torch.int32)
        out = sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=4, mode=mode, count_loads=cnt)
        ran += 1
        bad = [r for r in range(ref.shape[0]) if not t.bits_equal(out[r], ref[r])]
        if bad:
            fails.append(("bitwise", seed, mode, bad[:6]))
        want = whole if mode == "whole" else split
        loads = [0, 0]
        sk.bi_paged_decode_shared_reference(bi, *t.args_of(c), c["key"], rows_max=4, mode=mode, loads=loads)
        if int(cnt[0]) != want or [int(cnt[0]), int(cnt[1])] != loads:
            fails.append(("loads", seed, mode, cnt.tolist(), loads, want, total))
    out2 = sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=2)
    if not t.bits_equal(out2, ref):
        fails.append(("bitwise-rows_max-2", seed))
# the SharedKV selector: keyed calls use the shared kernel (stats), unkeyed / single-row calls use piece 3
c = t.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
ref = bi.bi_paged_decode(*t.args_of(c))
sel = sk.SharedKV(bi)
o1 = sel.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4)
o2 = sel.bi_paged_decode(*t.args_of(c))
o3 = sel.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=1)
if not (t.bits_equal(o1, ref) and t.bits_equal(o2, ref) and t.bits_equal(o3, ref)):
    fails.append(("selector-bits",))
if sel.stats != dict(shared=1, base_no_key=1, base_single_row=1):
    fails.append(("selector-stats", sel.stats))
# NEGATIVE CONTROL on the KERNEL: one union page dropped; precondition: it is listed by some row; caught = the
# listing rows differ, the others do not
for mode in ("split", "whole"):
    un = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, bi.PAGES_PER_SPLIT, mode == "whole")
    u = int(un["ucnt"][0, 0]) // 2
    listers = {int(un["grow"][0, j]) for j in range(4) if int(un["rowpos"][0, 0, j, u]) >= 0}
    if not listers:
        fails.append(("plant-precondition-unmet", mode))
        continue
    out = sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=4, mode=mode,
                                    union=sk.plant_drop_page(un, 0, 0, u))
    changed = {r for r in range(ref.shape[0]) if not t.bits_equal(out[r], ref[r])}
    if not changed:
        fails.append(("plant-no-op (BLIND)", mode))
    elif changed != listers:
        fails.append(("plant-not-local", mode, sorted(changed), sorted(listers)))
# the probe (CPU: no timing): returns piece 3's output, counts no differing row; with the plant it counts some
pr = sk.SharedKVProbe(bi, timing=False)
po = pr.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4)
rep = pr.report()
if not (t.bits_equal(po, ref) and rep["diff_rows"] == 0 and rep["rows_compared"] == ref.shape[0]):
    fails.append(("probe", rep))
split_t, whole_t, total_t = t.split_tiles_oracle(c, 4)
if rep["tiles"]["shared"] != split_t or rep["tiles"]["per_row"] != total_t or not rep["tile_ratio"] < 1:
    fails.append(("probe-tiles", rep["tiles"], split_t, total_t))
pp = sk.SharedKVProbe(bi, timing=False, plant=True)
pp.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4)
if pp.report()["diff_rows"] == 0:
    fails.append(("probe-plant-blind",))
print("RAN", ran)
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("SHAREDKV-INTERP OK")
'''


def test_kernel_under_the_triton_interpreter():
    """The Triton kernel on CPU float32 (TRITON_INTERPRET=1, a fresh process), both schedules, every case:
    (a) bitwise equal (int views) to PIECE 3'S KERNEL on the same inputs for every row, rows_max 4 and 2;
    (c) the kernel's own load counter equals the independent oracle (whole: the union; split: the split unions) and
        the twin's count (tiles and rows);
    the selector's stats (shared / no key / single row); (b) the drop-one-page plant caught on the kernel and local,
    after its precondition; the probe returns piece 3's bits and detects the plant."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(BASE_PATH), str(SK_PATH), str(Path(__file__).resolve())],
                       env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["SHAREDKV-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# ---------------------------------------------------------------------------
# (6) the tick's selector (CPU tiny model, torch twins): the production entry point (L36: not hand-built inputs)
# ---------------------------------------------------------------------------
def _tick_tests():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_r2_tick as RT                                             # its tiny model, engine and SG1 check
    return RT


def _p3_pinned(tmp_path):
    """Piece 3 v2 from the object store by its blob (as test_r2_tick's p3 fixture)."""
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3_BLOB], capture_output=True)
    assert body.returncode == 0 and body.stdout
    path = tmp_path / "piece3_v2.py"
    path.write_bytes(body.stdout)
    return _load(path, "piece3_v2_sharedkv_tick")


def test_tick_page_keys_follow_the_table(tmp_path):
    """TickEngine.page_keys: the block id of every page_table entry, compacted like the table (the draft drop): the
    historical entries (slot = block id in the ResidentStore) equal the table, the last two keys are T - 1 and T,
    keys strictly ascending. Negative: the Residency backend refuses (never guessed)."""
    RT = _tick_tests()
    p3 = _p3_pinned(tmp_path)
    eng = RT.make_engine(RT.make_ops(p3), draft_drop=0.5)
    H = RT.HKV
    T = torch.tensor([9, 9, 10])
    sel = torch.stack([torch.stack([torch.tensor(sorted([0, 2, 3, 4, 5] + [int(t) - 2, int(t) - 1, int(t)]))
                                    for t in T]) for _ in range(H)])
    pos = T * RT.BS + 5
    rb = torch.tensor([0, 1, 2])
    prev = torch.tensor([8, 8, 9], dtype=torch.int32)
    tail = torch.tensor([eng.NB + 2, eng.NB + 2, eng.NB + 0], dtype=torch.int32)
    draft = torch.tensor([True, True, False])
    lay = eng.model.layers[0]
    tab, npg = eng.page_table(0, sel, pos, rb, prev, tail, draft, True, lay)
    key = eng.page_keys(0, sel, pos, rb, draft, True, lay)
    assert key.dtype == torch.int32 and key.shape == tab.shape and key.is_contiguous()
    assert int(npg[:2].sum()) < 2 * H * RT.TOPK, "precondition: the draft drop removed some page"
    for r in range(3):
        for h in range(H):
            n = int(npg[r, h])
            ks, ts = key[r, h, :n].tolist(), tab[r, h, :n].tolist()
            assert all(b > a for a, b in zip(ks, ks[1:]))
            assert ks[-2:] == [int(T[r]) - 1, int(T[r])]
            assert ks[:-2] == ts[:-2]
    sk.validate_keys(tab, npg, key)
    with pytest.raises(RT.TK.TickError):
        RT.TK.ResidencyBackend.page_keys(object(), 0, None, None, None)


@pytest.mark.parametrize("mode", ["split", "whole"])
def test_tick_selector_teacher_forced_equals_serial(tmp_path, mode):
    """SG1 on the CPU with the selector: (i) the PROBE (piece 3's twin output returned, the shared twin computed on
    every keyed call of every layer): 0 differing rows, keyed calls with >= 2 rows of a request happened (else the
    test is blind), and the tick's checks pass; (ii) the shared twin IN the tick (its output used): every verify row
    torch.equal to the serial row, committed sequences equal; (iii) the probe with the plant (one union page dropped)
    counts differing rows (DETECTED; 0 would be a NO-OP plant, L34)."""
    RT = _tick_tests()
    p3 = _p3_pinned(tmp_path)
    toks, logits = RT.serial_ref(RT.make_engine(RT.make_ops(p3)))

    def ops_with(attn):
        return RT.SimpleNamespace(linear=RT.exact_linear, attn=attn, mean=RT.BD.fixed_order_mean)

    probe = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference")
    res = RT.tick_check(RT.make_engine(ops_with(probe), draft_drop=0.5), toks, logits, forced=True)
    rep = probe.report()
    assert probe.stats["shared"] > 0 and rep["rows_compared"] > 0, probe.stats
    assert rep["diff_rows"] == 0, rep["diff_rows"]
    assert res["compared"] > 50 and not res["bad"] and not res["ctrl"] and all(res["committed_ok"])
    used = sk.SharedKV(p3, mode=mode, impl="reference")
    res2 = RT.tick_check(RT.make_engine(ops_with(used), draft_drop=0.5), toks, logits, forced=True)
    assert used.stats["shared"] > 0, used.stats
    assert res2["compared"] > 50 and not res2["bad"] and not res2["ctrl"] and all(res2["committed_ok"])
    plant = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference", plant=True)
    RT.tick_check(RT.make_engine(ops_with(plant), draft_drop=0.5), toks, logits, forced=True)
    assert plant.report()["diff_rows"] > 0, "the plant was a NO-OP in the tick (BLIND)"


def test_gate_selector_default_is_piece3_and_choices():
    """The GPU gate's --attn: default per_row returns the loaded piece 3 module ITSELF (SG5 / p6 comparability); the
    other choices wrap it (probe / plant / whole); sharedkv_gates on per_row adds no gate; on a probe it adds the
    MUST bitwise gate (a differing row fails it) and on the plant the MUST detection gate (0 = BLIND fails it)."""
    eff = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(eff))
    G = _load(eff / "r2_tick_gate.py", "r2_tick_gate_sharedkv_cpu")
    base_args = ["--out", "x", "--workload", "w", "--biattn", "p"]
    a = G.parse_args(base_args)
    assert a.attn == "per_row" and Path(a.sharedkv) == SK_PATH
    assert G.select_attn(a, bi) is bi
    gates = []
    G.sharedkv_gates(a, bi, gates, {})
    assert gates == []
    for choice, cls, mode, plant in (("shared", "SharedKV", "split", None), ("shared-whole", "SharedKV", "whole", None),
                                     ("probe", "SharedKVProbe", "split", False), ("probe-plant", "SharedKVProbe", "split", True),
                                     ("probe-whole", "SharedKVProbe", "whole", False)):
        a = G.parse_args(base_args + ["--attn", choice])
        obj = G.select_attn(a, bi)
        assert type(obj).__name__ == cls and obj.mode == mode and obj.base is bi
        if plant is not None:
            assert obj.plant is plant
    with pytest.raises(SystemExit):
        G.parse_args(base_args + ["--attn", "fast"])
    fake = sk.SharedKVProbe(bi, timing=False)
    fake.stats["shared"] = 1
    fake.diff_rows, fake.rows_cmp = torch.tensor(3), 10
    a = G.parse_args(base_args + ["--attn", "probe"])
    gates = []
    G.sharedkv_gates(a, fake, gates, {})
    got = {g["name"]: g for g in gates}
    assert got["SKV_shared_kernel_ran"]["passed"] and not got["SKV_bitwise_shared_equals_per_row"]["passed"]
    fake.diff_rows = torch.tensor(0)
    a = G.parse_args(base_args + ["--attn", "probe-plant"])
    gates = []
    G.sharedkv_gates(a, fake, gates, {})
    assert not {g["name"]: g for g in gates}["SKV_plant_detected"]["passed"]
    # [review MINOR-3] F3 machine-checked: a MUST gate on the keyed calls with 4 rows of one request
    a = G.parse_args(base_args + ["--attn", "probe"])
    for counts, ok in (({}, False), ({2: 5, 3: 1}, False), ({4: 2}, True)):
        fake.by_rows_max = dict(counts)
        gates = []
        G.sharedkv_gates(a, fake, gates, {})
        g4 = {g["name"]: g for g in gates}["SKV_rows_max4_calls"]
        assert g4["must"] and g4["passed"] is ok, (counts, g4)
    # [review MINOR-5; sharedkv4 MINOR-3] every selector arm refuses --no-check (the fused build needs strictly
    # ascending keys per row); per_row keeps it
    for choice in ("shared", "shared-whole", "probe", "probe-whole", "probe-plant"):
        with pytest.raises(SystemExit):
            G.parse_args(base_args + ["--attn", choice, "--no-check"])
    assert G.parse_args(base_args + ["--no-check"]).no_check
    # [sharedkv4 MINOR-5 / MAJOR-1] the build kind and the fused-union equality are MUST gates
    a = G.parse_args(base_args + ["--attn", "probe", "--union", "fused"])
    for ub, diff, checked, ok_kind, ok_eq in (({"torch": 0, "fused": 3}, {"win": 0}, 3, True, True),
                                              ({"torch": 2, "fused": 3}, {"win": 0}, 3, False, True),
                                              ({"torch": 0, "fused": 3}, {"win": 2}, 3, True, False),
                                              ({"torch": 0, "fused": 3}, {"win": 0}, 0, True, False)):
        fake.union_builds, fake.union_diff, fake.union_checked = dict(ub), dict(diff), checked
        gates = []
        G.sharedkv_gates(a, fake, gates, {})
        got = {g["name"]: g for g in gates}
        assert got["SKV_union_build_kind"]["must"] and got["SKV_union_build_kind"]["passed"] is ok_kind, (ub, got)
        assert got["SKV_fused_union_equals_torch"]["must"]
        assert got["SKV_fused_union_equals_torch"]["passed"] is ok_eq, (diff, checked)


def test_step_script_shape_and_its_sg1_schedule_is_not_blind():
    """The proposed step (SOURCE ONLY): pins, arms, B 32 with 32 deltas through the GATE'S OWN parse_args (a wrong
    count is refused there), the SG1 schedule of those prompts covers every kind (r2_tick.simulate_schedule, the
    gate's refusal rule), no here-doc / ulimit / interpreter words outside comments."""
    step = Path(__file__).resolve().parents[1] / "r2_sharedkv_step.sh"
    text = step.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-sharedkv" in text and "BIATTN_BLOB=%s" % PIECE3_BLOB in text
    assert 'UFILE=$HOLDERS/$JOB.gpu_uuid' in text and text.index("UFILE=$HOLDERS") < text.index('mkdir "$OUTROOT/skv_active"')
    assert "--export=NONE" in text and '--attn "$ATTN" --union "$UNION"' in text
    assert "<<" not in text and "ulimit" not in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    val = {}
    for ln in lines:
        for name in ("DELTAS", "BATCH", "M_FIX", "STEPS"):
            if ln.startswith(name + "="):
                val[name] = ln.split("=", 1)[1].split()[0]
    eff = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(eff))
    G = _load(eff / "r2_tick_gate.py", "r2_tick_gate_sharedkv_step_cpu")
    a = G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p", "--batch", val["BATCH"], "--deltas",
                      val["DELTAS"], "--m-fix", val["M_FIX"], "--steps", val["STEPS"], "--parts", "SG1",
                      "--attn", "probe"])
    assert a.batch == 32 and len(a.deltas) == 32 and a.m_fix == 128
    RT = _tick_tests()
    sim, _ = RT.TK.simulate_schedule([a.S - d for d in a.deltas], [a.steps] * a.batch, K=a.K, LAMBDA=a.LAMBDA)
    assert RT.TK.schedule_coverage(sim, a.K)["complete"], "the step's SG1 schedule would be blind"


# ---------------------------------------------------------------------------
# (7) source checks: the pinned base, L36 globals lint, the law in the text
# ---------------------------------------------------------------------------
def test_base_is_the_pinned_piece3_blob():
    r = subprocess.run(["git", "-C", str(ROOT), "hash-object", "--", str(BASE_PATH)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == PIECE3_BLOB, (r.stdout, r.stderr[-300:])


def _jit_functions(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and f.attr == "jit":
                    out.append(node)
    return out


def _module_names(tree):
    """The names bound at MODULE level: top-level statements and the bodies of top-level if / try blocks (the
    kernels live under `if triton is not None:`, tl under `try:`), never inside a function or a class."""
    names = set()

    def visit(stmts):
        for s in stmts:
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in s.names)
            elif isinstance(s, (ast.FunctionDef, ast.ClassDef)):
                names.add(s.name)
            elif isinstance(s, (ast.Assign, ast.AnnAssign)):
                for t in (s.targets if isinstance(s, ast.Assign) else [s.target]):
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(s, ast.If):
                visit(s.body)
                visit(s.orelse)
            elif isinstance(s, ast.Try):
                visit(s.body)
                for hd in s.handlers:
                    visit(hd.body)
                visit(s.orelse)
                visit(s.finalbody)
    visit(tree.body)
    return names


@pytest.mark.parametrize("path", [SK_PATH, BASE_PATH])
def test_jit_bodies_resolve_their_globals_in_the_module(path):
    """L36: the GPU compiler resolves a @triton.jit body's names from fn.__globals__ (the interpreter does not care).
    Every name a jit body LOADS is a parameter, a local, a module-level name or a builtin. Negative control: a body
    with a factory-local name (the fg5 defect) is flagged."""
    src = path.read_text()
    tree = ast.parse(src)
    mod_names = _module_names(tree)

    def unresolved(fn):
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        local = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        return sorted(x for x in loads if x not in params | local | mod_names and not hasattr(builtins, x))

    fns = _jit_functions(tree)
    assert len(fns) >= (6 if path == SK_PATH else 2)
    for fn in fns:
        assert not unresolved(fn), (fn.name, unresolved(fn))
    bad = ast.parse("def factory():\n    import triton.language as tl2\n    @triton.jit\n    def k(X):\n        tl2.store(X, 1)\n    return k\n")
    assert unresolved(_jit_functions(bad)[0]) == ["tl2"]


def test_shared_source_keeps_piece3_constants_and_no_host_read():
    src = SK_PATH.read_text()
    tree = ast.parse(src)
    assert "triton.autotune" not in src and "triton.heuristics" not in src
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bi_paged_decode_shared")
    seg = ast.get_source_segment(src, fn)
    code = "\n".join(l.split("#")[0] for l in seg.split('"""')[2].splitlines())
    assert "num_warps=base.NUM_WARPS, num_stages=base.NUM_STAGES" in code
    assert "base._bi_combine_kernel[(R, Hkv)](" in code and "S = base.splits_for(page_table.shape[2], pps)" in code
    fn2 = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_union")
    code2 = "\n".join(l.split("#")[0] for l in ast.get_source_segment(src, fn2).split('"""')[2].splitlines())
    for bad in (".item(", ".tolist(", ".cpu(", "bool(", "synchronize", "int(t"):
        assert bad not in code and bad not in code2, bad
    step = next(n for n in tree.body if isinstance(n, ast.If)).body
    names = {n.name for n in step if isinstance(n, ast.FunctionDef)}
    assert {"_sk_row_step", "_bi_shared_kernel", "_sk_row_ok", "_sk_row_init", "_sk_row_store", "_sk_row_flush"} <= names
    rs = ast.get_source_segment(src, next(n for n in step if isinstance(n, ast.FunctionDef) and n.name == "_sk_row_step"))
    for line in ("sc = tl.dot(q, tl.trans(kz)) + bz[None, :]",
                 'sc = tl.where(ok[None, :], sc * scale_log2, float("-inf"))',
                 "m_new = tl.maximum(m_i, tl.max(sc, 1))",
                 'alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))',
                 "pr = tl.where(ok[None, :], tl.exp2(sc - m_new[:, None]), 0.0)",
                 "l_i = l_i * alpha + tl.sum(pr, 1)",
                 "acc = acc * alpha[:, None] + tl.dot(pr.to(vz.dtype), vz)"):
        assert line in rs, line
    base_src = BASE_PATH.read_text()                                    # the same formulas as piece 3's split kernel
    for line in ('sc = tl.where(ok[None, :], sc * scale_log2, float("-inf"))',
                 "m_new = tl.maximum(m_i, tl.max(sc, 1))",
                 'alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))',
                 "pr = tl.where(ok[None, :], tl.exp2(sc - m_new[:, None]), 0.0)",
                 "l_i = l_i * alpha + tl.sum(pr, 1)",
                 "acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)"):
        assert line in base_src, line


# ---------------------------------------------------------------------------
# (8) [sharedkv2] the review fixes of f79d0df (kstep-cpu-runs/paper-notes/review-sharedkv-f79d0df.md)
# ---------------------------------------------------------------------------
def test_probe_records_the_group_mix_and_the_expectation():
    """MAJOR-1: the probe records, per keyed call, the group mix on the device: per group size s the groups, their
    per-row pages (both heads) and their union pages. Oracle: the case's own requests (rows_of; one group per
    request with <= 4 rows) counted on the host. expected_tile_ratio is the registered formula on that mix. The tile
    identity check flags a call whose counter breaks union <= tiles <= sum (a planted bad count; the CPU twin has no
    counter of its own). The recorded tables are the call's inputs."""
    c = make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    pr = sk.SharedKVProbe(bi, timing=False, impl="reference", record_tables=2)
    pr.bi_paged_decode(*args_of(c), page_key=c["key"], rows_max=4)
    rep = pr.report()
    want = [[0] * 5 for _ in range(3)]
    want[0][0] = c["seq_idx"].shape[0] - len(c["rows_of"])                # empty group slots (grid rows past the last)
    for b, rows in c["rows_of"].items():
        s = len(rows)
        want[0][s] += 1
        want[1][s] += int(sum(int(c["num_pages"][r].sum()) for r in rows))
        for h in range(c["num_pages"].shape[1]):
            pairs = {(int(c["key"][r, h, i]), int(c["table"][r, h, i])) for r in rows
                     for i in range(int(c["num_pages"][r, h]))}
            want[2][s] += len(pairs)
    assert rep["mix_all"] == want and rep["mix_full"] == want, (rep["mix_all"], want)
    rho = sk.RHO["split"]
    exp = sum(rho[s] * want[1][s] for s in range(1, 5)) / sum(want[1][s] for s in range(1, 5))
    assert abs(rep["expected_tile_ratio"] - exp) < 1e-12
    assert rep["by_rows_max"] == {4: 1} and rep["tile_identity_checked"] == 0
    tabs = pr.report_tables()
    assert len(tabs) == 1 and torch.equal(tabs[0]["page_key"], c["key"]) and torch.equal(tabs[0]["page_table"], c["table"])
    call, nr, mix, s, _ = pr.mixes[0]
    pr.mixes.append((1, nr, mix, s, s + 1))                               # NEGATIVE: more tiles than the sum
    pr.mixes.append((2, nr, mix, s, torch.tensor(sum(want[2][1:]))))     # the union exactly: fine in split
    rep2 = pr.report()
    assert rep2["tile_identity_checked"] == 2 and [x["call"] for x in rep2["tile_identity_bad"]] == [1]


def test_probe_triages_sign_of_zero_and_value_differences():
    """MINOR-2: the probe splits differing elements into sign-of-zero only (the floats compare equal) and value
    differences, and records the first differing element of each call (flat index, both bit patterns). Planted on
    the CPU: the shared side returns piece 3's output with one +0.0 turned into -0.0 (call 0), then with one value
    changed (call 1); a clean call (call 2) adds nothing."""
    c = make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    plants = []

    class Planted(sk.SharedKVProbe):
        def _base(self, args, softmax_scale, pps, out=None, workspace=None):
            o = super()._base(args, softmax_scale, pps)
            o[1, 2, 3] = 0.0
            return o

        def _shared(self, args, page_key, nr, softmax_scale, pps, out=None, workspace=None, union=None):
            o = self._base(args, softmax_scale, pps)
            kind = plants[self.calls - 1]
            if kind == "zero":
                o[1, 2, 3] = -0.0
            elif kind == "value":
                o[0, 0, 0] = o[0, 0, 0] + 1.0
            return o
    pr = Planted(bi, timing=False, impl="reference")
    for kind in ("zero", "value", "clean"):
        plants.append(kind)
        pr.bi_paged_decode(*args_of(c), page_key=c["key"], rows_max=4)
    rep = pr.report()
    assert rep["diff_rows"] == 2 and rep["diff_elements_zero_sign"] == 1 and rep["diff_elements_value"] == 1, rep
    Hq, D = c["q"].shape[1], c["q"].shape[2]
    s0, s1 = rep["diff_samples"]
    assert (s0["call"], s0["flat_index"]) == (0, 1 * Hq * D + 2 * D + 3) and s0["bits_per_row"] != s0["bits_shared"]
    assert (s1["call"], s1["flat_index"]) == (1, 0)


def test_probe_times_the_build_with_the_shared_side_and_launches_no_extra_zeros():
    """MAJOR-2 / MAJOR-3 (source): the probe's shared side builds the union inside its own timed span (NVTX range
    bd); a torch.cuda._sleep pre-queues the GPU before the timed pair; the uncounted launch passes the persistent
    dummy counter (no torch.zeros launch per call)."""
    src = SK_PATH.read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedKVProbe")
    body = ast.get_source_segment(src, next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                             and n.name == "bi_paged_decode"))
    assert 'box["un"] = ranged("bd", build)' in body
    assert body.index("torch.cuda._sleep(self.sleep_cycles)") < body.index("ev[0].record()")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bi_paged_decode_shared")
    code = "\n".join(l.split("#")[0] for l in ast.get_source_segment(src, fn).split('"""')[2].splitlines())
    assert "torch.zeros(" not in code and "_dummy_counter(q.device)" in code


def test_step_dryrun_refusals_and_pins(tmp_path):
    """MAJOR-4 (as test_r2_g0bi.py G8): r2_sharedkv_step.sh EXECUTED in DRYRUN. A wrong arm is a usage error (2); a
    missing uuid file, a GPU- prefix and two lines STOP (1); TICK_HOLDERS and TICK_PIN_* outside DRYRUN are refused; a
    wrong --nosi stops at HEAD. On a clean checkout of the step's own worktree at HEAD: each arm prints its PARTS /
    DROP / --attn (per_row and the probes SG1 0.0, the shared arms SG0-SG3 0.5) and both blobs; piece 3's FIRST pin
    (at 7c826c1) refuses a wrong blob, the SECOND (in this tree) refuses an older piece 3 blob that the first pin
    accepts (its commit from git log)."""
    script = Path(__file__).resolve().parents[1] / "r2_sharedkv_step.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "TICK_"))}
    uid = "f31fc723-9dfe-407a-9947-28173556232a"

    def run(job, content=None, dry=True, arm="probe", sha="0" * 40, holders=True, extra=None):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env)
        if holders:
            e["TICK_HOLDERS"] = str(tmp_path)
        e.update(extra or {})
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1", arm], env=e, capture_output=True,
                              text=True, timeout=300)
    r = run(410, arm="probe-fast")
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(411)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(412, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(413, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(414, uid + "\n", dry=False)
    assert r.returncode == 1 and "TICK_HOLDERS is for DRYRUN" in r.stderr
    r = run(415, dry=False, holders=False, extra=dict(TICK_PIN_BIATTN_BLOB="0" * 40))
    assert r.returncode == 1 and "for DRYRUN" in r.stderr and "TICK_PIN" in r.stderr
    r = run(416, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    nosi = "/mnt/beegfs/ojerbi/nosi-r2-sharedkv4"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(nosi).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % nosi)
    sha = head.stdout.strip()
    skv = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:nosi/nosi/bi_paged_attention_shared.py"],
                         capture_output=True, text=True).stdout.strip()
    for arm, parts, drop, attn, union in (
            ("per_row", "SG1", "0.0", "per_row", "torch"), ("probe-plant", "SG1", "0.0", "probe-plant", "torch"),
            ("probe", "SG1", "0.0", "probe", "torch"), ("probe-whole", "SG1", "0.0", "probe-whole", "torch"),
            ("shared", "SG0,SG1,SG2,SG3", "0.5", "shared", "torch"),
            ("shared-whole", "SG0,SG1,SG2,SG3", "0.5", "shared-whole", "torch"),
            ("probe-plant-fused", "SG1", "0.0", "probe-plant", "fused"),                  # [sharedkv3]
            ("probe-fused", "SG1", "0.0", "probe", "fused"),
            ("shared-fused", "SG0,SG1,SG2,SG3", "0.5", "shared", "fused")):
        r = run(417, uid + "\n", arm=arm, sha=sha)
        assert r.returncode == 0, (arm, r.stderr[-400:])
        assert "arm %s parts %s drop %s attn %s union %s" % (arm, parts, drop, attn, union) in r.stdout, arm
        assert "piece 3 7c826c1bc7235ee9153e3af82c99d85836915416:nosi/nosi/bi_paged_attention.py blob %s" % PIECE3_BLOB \
            in r.stdout and "shared %s" % skv in r.stdout
        for w in ("--attn %s --union %s" % (attn, union), "--parts %s" % parts.replace(",", "\\,"),
                  "--draft-drop %s" % drop,
                  "--sharedkv /mnt/beegfs/ojerbi/r2-runs/sharedkv/skv_%s_t1_417.step/bi_paged_attention_shared.py" % arm):
            assert w in r.stdout, (arm, w)
    r = run(418, uid + "\n", sha=sha, extra=dict(TICK_PIN_BIATTN_BLOB="0" * 40))
    assert r.returncode == 1 and "piece 3: nosi/nosi/bi_paged_attention.py at" in r.stderr, r.stderr[-300:]
    hist = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "7c826c1bc7235ee9153e3af82c99d85836915416",
                           "--", "nosi/nosi/bi_paged_attention.py"], capture_output=True, text=True).stdout.split()
    older = None
    for h in hist:
        b = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", h + ":nosi/nosi/bi_paged_attention.py"],
                           capture_output=True, text=True).stdout.strip()
        if b and b != PIECE3_BLOB:
            older = (h, b)
            break
    assert older is not None, "precondition: an older piece 3 blob exists in the history (c866d7c at the time of writing)"
    r = run(419, uid + "\n", sha=sha, extra=dict(TICK_PIN_BIATTN_SHA=older[0], TICK_PIN_BIATTN_BLOB=older[1]))
    assert r.returncode == 1 and "piece 3 in this tree" in r.stderr, r.stderr[-300:]
