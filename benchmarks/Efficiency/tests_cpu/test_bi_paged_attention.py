"""CPU tests of the BATCH-INVARIANT PAGED decode attention (nosi/nosi/bi_paged_attention.py; R2-M2 step 7, piece 3;
design note docs/R2_STEP7_SPEC_ROUND.md section 3.4). The module is loaded by path (nosi/nosi/__init__.py imports CUDA
kernels). The torch twin runs here; the Triton kernel runs under TRITON_INTERPRET=1 in a FRESH process (Triton reads
the flag at its first import; job 2189872), skipped when triton is not installed.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_bi_paged_attention.py

(compute node only; PYTHONDONTWRITEBYTECODE=1 keeps the checkout free of untracked __pycache__.)
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
ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "nosi/nosi/bi_paged_attention.py"
GATE = ROOT / "benchmarks/Efficiency/biattn_gate.py"
STEP = ROOT / "benchmarks/Efficiency/biattn_step.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bi = _load(MOD, "bi_paged_attention_under_test")
PAGE = bi.PAGE


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def make_case(seed, R=10, N=4, S=12, Hkv=2, G=16, D=32, P=8, dtype=torch.float32):
    """R rows over N storage entries (several rows share an entry: a request's rows at other positions); per (row,
    head) a random number of pages 1..P with distinct random block ids and distinct random slots, given to
    build_page_table in RANDOM order; random last_len 1..64."""
    g = torch.Generator().manual_seed(1000 + seed)
    q = torch.randn(R, Hkv * G, D, generator=g).to(dtype)
    k = torch.randn(N, S * PAGE, Hkv, D, generator=g).to(dtype)
    v = torch.randn(N, S * PAGE, Hkv, D, generator=g).to(dtype)
    bias = (torch.rand(N, S * PAGE, Hkv, generator=g) - 0.5).to(dtype)
    seq_idx = torch.randint(0, N, (R,), generator=g).to(torch.int32)
    ids = torch.full((R, Hkv, P), -1, dtype=torch.int64)
    slots = torch.full((R, Hkv, P), -1, dtype=torch.int64)
    for r in range(R):
        for h in range(Hkv):
            npg = int(torch.randint(1, P + 1, (1,), generator=g))
            blk = torch.randperm(200, generator=g)[:npg]
            sl = torch.randperm(S, generator=g)[:npg]
            order = torch.randperm(P, generator=g)[:npg]          # random positions in the input list
            ids[r, h, order] = blk
            slots[r, h, order] = sl
    table, num_pages = bi.build_page_table(ids, slots)
    last_len = torch.randint(1, PAGE + 1, (R,), generator=g).to(torch.int32)
    return dict(q=q, k=k, v=v, bias=bias, seq_idx=seq_idx, table=table, num_pages=num_pages, last_len=last_len,
                ids=ids, slots=slots)


def run(fn, c, rows=None, table=None, **kw):
    """fn over the case's rows (all, or the given list of row indices, in that order)."""
    idx = torch.arange(c["q"].shape[0]) if rows is None else torch.tensor(rows, dtype=torch.int64)
    t = c["table"] if table is None else table
    return fn(c["q"][idx].contiguous(), c["k"], c["v"], c["bias"], c["seq_idx"][idx].contiguous(),
              t[idx].contiguous(), c["num_pages"][idx].contiguous(), c["last_len"][idx].contiguous(), **kw)


def poisoned(c, value=float("nan")):
    """A copy of the case whose caches hold `value` in every row that no row of the case lists: unlisted slots of
    each (entry, head), and in a listed page that is the LAST page of every row listing it the rows at or beyond the
    largest extent among those rows."""
    k, v, b = c["k"].clone(), c["v"].clone(), c["bias"].clone()
    N, SR, Hkv, _ = k.shape
    used = torch.zeros(N, SR, Hkv, dtype=torch.bool)
    tab, npg, ll, si = c["table"].tolist(), c["num_pages"].tolist(), c["last_len"].tolist(), c["seq_idx"].tolist()
    for r in range(len(si)):
        for h in range(Hkv):
            for i in range(npg[r][h]):
                s = tab[r][h][i]
                cnt = PAGE if i < npg[r][h] - 1 else ll[r]
                used[si[r], s * PAGE:s * PAGE + cnt, h] = True
    k[~used] = value
    v[~used] = value
    b[~used] = value
    return dict(c, k=k, v=v, bias=b)


def differs(a, b):
    return not torch.equal(a, b)


# ---------------------------------------------------------------------------
# (1) page lists and small helpers
# ---------------------------------------------------------------------------
def test_build_page_table_is_canonical_and_independent_of_input_order_and_slots():
    c = make_case(0)
    ids, slots, table, npg = c["ids"], c["slots"], c["table"], c["num_pages"]
    R, Hkv, P = ids.shape
    for r in range(R):
        for h in range(Hkv):
            pairs = sorted((int(i), int(s)) for i, s in zip(ids[r, h].tolist(), slots[r, h].tolist()) if i >= 0)
            assert int(npg[r, h]) == len(pairs)
            assert table[r, h, :len(pairs)].tolist() == [s for _, s in pairs]
            assert all(x == -1 for x in table[r, h, len(pairs):].tolist())
    # the same (block, slot) pairs in another input order: the same table
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(9))
    t2, n2 = bi.build_page_table(ids[..., perm], slots[..., perm])
    assert torch.equal(t2, table) and torch.equal(n2, npg)
    assert table.dtype == torch.int32 and table.is_contiguous() and npg.dtype == torch.int32


def test_last_len_and_query_head_tile():
    assert bi.last_len_of(torch.tensor([16128, 16191, 16192, 63, 64])).tolist() == [1, 64, 1, 64, 1]
    assert [bi.block_h_for(g) for g in (1, 4, 16, 17, 32)] == [16, 16, 16, 32, 32]
    assert bi.PAGES_PER_TILE == 1 and bi.PAGE == 64


def test_validate_inputs_refuses_bad_tables():
    c = make_case(1)
    bi.validate_inputs(c["q"], c["k"], c["v"], c["bias"], c["seq_idx"], c["table"], c["num_pages"], c["last_len"])
    bad = c["table"].clone()
    bad[0, 0, 0] = 99
    with pytest.raises(ValueError):
        bi.validate_inputs(c["q"], c["k"], c["v"], c["bias"], c["seq_idx"], bad, c["num_pages"], c["last_len"])
    ll = c["last_len"].clone()
    ll[0] = 0
    with pytest.raises(ValueError):
        bi.validate_inputs(c["q"], c["k"], c["v"], c["bias"], c["seq_idx"], c["table"], c["num_pages"], ll)
    with pytest.raises(ValueError):                                      # int64 indices are refused
        bi.bi_paged_decode_reference(c["q"], c["k"], c["v"], c["bias"], c["seq_idx"].long(), c["table"],
                                     c["num_pages"], c["last_len"])


# ---------------------------------------------------------------------------
# (2) the twin: semantics, bias, extents, NaN poison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-5), (torch.bfloat16, 3e-2)])
def test_twin_equals_the_float64_semantics(dtype, atol):
    c = make_case(2, dtype=dtype)
    out = run(bi.bi_paged_decode_reference, c)
    ref = run(bi.attention_oracle, c)
    assert out.dtype == dtype
    assert float((out.double() - ref).abs().max()) <= atol


def test_bias_is_added_to_the_raw_dot_before_the_scale_and_only_listed_rows_count():
    c = make_case(3)
    scale = 0.3
    out = run(bi.bi_paged_decode_reference, c, softmax_scale=scale)
    ref = run(bi.attention_oracle, c, softmax_scale=scale)
    assert float((out.double() - ref).abs().max()) <= 2e-5
    # a listed row's bias moves the output; a bias change in rows nobody lists does not (bitwise)
    r, h = 0, 0
    s0 = int(c["table"][r, h, 0])
    b2 = c["bias"].clone()
    b2[int(c["seq_idx"][r]), s0 * PAGE, h] += 3.0
    assert differs(run(bi.bi_paged_decode_reference, dict(c, bias=b2))[r], out[r])
    p = poisoned(c, value=0.0)
    b3 = c["bias"].clone()
    unlisted = p["bias"] != c["bias"]
    b3[unlisted] += 5.0
    assert bool(unlisted.any())
    assert torch.equal(run(bi.bi_paged_decode_reference, dict(c, bias=b3), softmax_scale=scale), out)


def _read_rows(c, r):
    """The exact (entry, cache row, head) set that row r attends: every listed page in full except the last, which
    is cut at its last_len."""
    n, last = int(c["seq_idx"][r]), int(c["last_len"][r])
    out = set()
    for h in range(c["table"].shape[1]):
        npg = int(c["num_pages"][r, h])
        for i in range(npg):
            s = int(c["table"][r, h, i])
            out.update((n, s * PAGE + j, h) for j in range(PAGE if i < npg - 1 else last))
    return out


def _extent_cut_failures(fn, c):
    """The cut of the last page, checked two ways (an empty list = pass):
    (a) each row ALONE: NaN in K, V and bias at every row of its last page at or beyond its own last_len (every head)
        leaves its output bitwise unchanged; +7 on the key of its LAST READ row (last_len - 1) changes it;
    (b) the BATCH: NaN at the rows beyond EVERY row's extent (the union M); a row whose exact read set (_read_rows)
        misses M is bitwise unchanged, a row that reads part of M (a page that is another row's last page, read in
        full, or the same last page with a longer extent) is changed; at least one row is unchanged.
    The first version (c866d7c) excluded only rows whose OWN last page another row read in full, so a row that read
    another row's last page in full was wrongly expected unchanged (CPU job 2190749, row 7)."""
    bad = []
    R, Hkv = c["table"].shape[:2]
    nan = float("nan")
    tab, npg, si, ll = c["table"].tolist(), c["num_pages"].tolist(), c["seq_idx"].tolist(), c["last_len"].tolist()
    out = run(fn, c)
    for r in range(R):
        alone = run(fn, c, rows=[r])[0]
        beyond = {key: c[key].clone() for key in ("k", "v", "bias")}
        lastrow = c["k"].clone()
        n, L = si[r], ll[r]
        for h in range(Hkv):
            s = tab[r][h][npg[r][h] - 1]
            for key in ("k", "v", "bias"):
                beyond[key][n, s * PAGE + L:(s + 1) * PAGE, h] = nan
            lastrow[n, s * PAGE + L - 1, h] += 7.0
        if not torch.equal(run(fn, dict(c, **beyond), rows=[r])[0], alone):
            bad.append(("alone: a row beyond the extent was read", r))
        if torch.equal(run(fn, dict(c, k=lastrow), rows=[r])[0], alone):
            bad.append(("alone: the last row of the extent was not read", r))
    pois = {key: c[key].clone() for key in ("k", "v", "bias")}
    M = set()
    for r in range(R):
        n, L = si[r], ll[r]
        for h in range(Hkv):
            s = tab[r][h][npg[r][h] - 1]
            M.update((n, s * PAGE + j, h) for j in range(L, PAGE))
            for key in ("k", "v", "bias"):
                pois[key][n, s * PAGE + L:(s + 1) * PAGE, h] = nan
    out2 = run(fn, dict(c, **pois))
    untouched = 0
    for r in range(R):
        if _read_rows(c, r) & M:
            if torch.equal(out2[r], out[r]):
                bad.append(("batch: reads a changed row but did not change", r))
        else:
            untouched += 1
            if not torch.equal(out2[r], out[r]):
                bad.append(("batch: reads no changed row but changed", r))
    if untouched == 0:
        bad.append(("batch: no untouched row (vacuous)",))
    return bad


def _cut_shifted(delta):
    """A WRONG attention for the negative controls: the twin with every last_len moved by delta (clamped to 1..64),
    i.e. a cut one row too late (+1) or too early (-1)."""
    def fn(q, k, v, b, si, t, npg, ll, **kw):
        moved = (ll.to(torch.int64) + delta).clamp(1, PAGE).to(torch.int32).contiguous()
        return bi.bi_paged_decode_reference(q, k, v, b, si, t, npg, moved, **kw)
    return fn


@pytest.mark.parametrize("last", [1, 2, 37, 63, 64, None])
def test_the_last_page_is_cut_at_the_rows_extent(last):
    """None = the case's random per-row last_len (rows of one entry with different extents on one page). S = 16
    slots keep rows that read no changed row likely (the check refuses a vacuous batch)."""
    c = make_case(4, S=16)
    if last is not None:
        c["last_len"] = torch.full_like(c["last_len"], last)
    out = run(bi.bi_paged_decode_reference, c)
    assert float((out.double() - run(bi.attention_oracle, c)).abs().max()) <= 2e-5
    bad = _extent_cut_failures(bi.bi_paged_decode_reference, c)
    assert not bad, bad[:6]


def test_negative_a_cut_one_row_late_or_early_is_caught():
    """The extent check cannot pass wrongly: a cut one row too late reads the poisoned row L (caught in (a) and
    (b)); a cut one row too early skips row L-1 (caught by (a)'s positive control)."""
    c = make_case(4, S=16)
    assert bool((c["last_len"] < PAGE).any()) and bool((c["last_len"] > 1).any()), "precondition: room on both sides"
    late = _extent_cut_failures(_cut_shifted(+1), c)
    early = _extent_cut_failures(_cut_shifted(-1), c)
    assert any(k[0].startswith("alone: a row beyond") for k in late), late[:4]
    assert any(k[0].startswith("alone: the last row") for k in early), early[:4]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nan_poison_unlisted_pages_and_rows_beyond_extents_are_never_read(dtype):
    c = make_case(5, dtype=dtype)
    clean = run(bi.bi_paged_decode_reference, c)
    p = poisoned(c)
    assert bool(torch.isnan(p["k"]).any())
    dirty = run(bi.bi_paged_decode_reference, p)
    assert bool(torch.isfinite(dirty).all()) and torch.equal(dirty, clean)
    # positive control: poison ONE listed row and the poison reaches the output
    k3 = c["k"].clone()
    k3[int(c["seq_idx"][0]), int(c["table"][0, 0, 0]) * PAGE, 0] = float("nan")
    assert bool(torch.isnan(run(bi.bi_paged_decode_reference, dict(c, k=k3))[0]).any())


# ---------------------------------------------------------------------------
# (3) the law: row invariance, and the two negative controls
# ---------------------------------------------------------------------------
def _invariance_failures(fn, c, **kw):
    """Rows whose output is not torch.equal across: alone, the full batch, a permuted batch, new neighbours, the
    rows of a single entry together, and a wider page table."""
    R = c["q"].shape[0]
    full = run(fn, c, **kw)
    bad = set()
    for r in range(R):
        if differs(run(fn, c, rows=[r], **kw)[0], full[r]):
            bad.add(("alone", r))
    perm = torch.randperm(R, generator=torch.Generator().manual_seed(7)).tolist()
    pout = run(fn, c, rows=perm, **kw)
    for i, r in enumerate(perm):
        if differs(pout[i], full[r]):
            bad.add(("permuted", r))
    other = make_case(99, R=R, N=c["k"].shape[0], S=c["k"].shape[1] // PAGE, P=c["table"].shape[2],
                      dtype=c["q"].dtype)
    for r in range(R):                                   # every other row replaced by an unrelated row (same caches)
        mixed = dict(c)
        for key in ("q", "seq_idx", "table", "num_pages", "last_len"):
            t = other[key].clone()
            t[r] = c[key][r]
            mixed[key] = t
        if differs(run(fn, mixed, **kw)[r], full[r]):
            bad.add(("neighbours", r))
    for n in set(c["seq_idx"].tolist()):                 # a request's own rows at other positions only
        rows = [r for r in range(R) if int(c["seq_idx"][r]) == n]
        o = run(fn, c, rows=rows, **kw)
        for i, r in enumerate(rows):
            if differs(o[i], full[r]):
                bad.add(("same-request", r))
    wide = torch.cat([c["table"], torch.full_like(c["table"], -1)], dim=-1)
    wout = run(fn, c, table=wide, **kw)
    for r in range(R):
        if differs(wout[r], full[r]):
            bad.add(("wider-table", r))
    return bad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_twin_row_is_invariant_under_any_batch_composition(dtype):
    for seed in (6, 7):
        c = make_case(seed, dtype=dtype)
        assert not _invariance_failures(bi.bi_paged_decode_reference, c)


def test_negative_split_kv_by_the_batchs_longest_row_is_caught():
    c = make_case(8, R=10, P=12)
    assert int(c["num_pages"].max()) >= 8, "precondition: a long row exists"
    bad = _invariance_failures(bi.bi_paged_decode_reference, c, split_by_batch=True)
    assert any(kind == "alone" for kind, _ in bad), bad


def test_negative_batch_keyed_tile_is_caught():
    c = make_case(9, R=10, P=8)
    bad = _invariance_failures(bi.bi_paged_decode_reference, c, tile_by_batch=True)
    assert any(kind == "alone" for kind, _ in bad), bad


def test_page_order_reaches_the_bits_so_the_list_must_be_canonical():
    """Negative control for canonicalisation: the same pages listed in SLOT order instead of block-id order give a
    different result for some row (the reduction order follows the list), while any input order through
    build_page_table gives the canonical result bitwise."""
    c = make_case(10, P=8)
    canon = run(bi.bi_paged_decode_reference, c)
    R, Hkv, P = c["table"].shape
    slot_order = c["table"].clone()
    for r in range(R):
        for h in range(Hkv):
            n = int(c["num_pages"][r, h])
            keep_last = slot_order[r, h, n - 1].item()            # the live tail stays last (its extent is cut)
            head = sorted(slot_order[r, h, :n - 1].tolist())
            slot_order[r, h, :n] = torch.tensor(head + [keep_last], dtype=torch.int32)
    moved = [r for r in range(R) if differs(run(bi.bi_paged_decode_reference, c, table=slot_order)[r], canon[r])]
    assert moved, "precondition: some row's slot order differs from its block order"
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(11))
    t2, _ = bi.build_page_table(c["ids"][..., perm], c["slots"][..., perm])
    assert torch.equal(run(bi.bi_paged_decode_reference, c, table=t2), canon)


# ---------------------------------------------------------------------------
# (4) the Triton kernel under the interpreter (fresh process)
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
t = load(sys.argv[2], "bi_tests")                # the test module: its case builders (it imports no triton itself
                                                 # beyond the module under test, already loaded under the flag)
fails = []
for seed in (20, 21):
    c = t.make_case(seed, R=8, N=3, S=10, P=6)
    k_out = t.run(bi.bi_paged_decode, c)
    ref = t.run(bi.attention_oracle, c)
    twin = t.run(bi.bi_paged_decode_reference, c)
    # 3e-3: an interpreter may emulate TF32 for an fp32 dot; an indexing error is O(1)
    if float((k_out.double() - ref).abs().max()) > 3e-3:
        fails.append(("semantics", seed, float((k_out.double() - ref).abs().max())))
    if float((k_out - twin).abs().max()) > 3e-3:
        fails.append(("twin", seed, float((k_out - twin).abs().max())))
    bad = t._invariance_failures(bi.bi_paged_decode, c)
    if bad:
        fails.append(("invariance", seed, sorted(bad)[:5]))
    p = t.poisoned(c)
    dirty = t.run(bi.bi_paged_decode, p)
    if not (bool(torch.isfinite(dirty).all()) and torch.equal(dirty, k_out)):
        fails.append(("nan-poison", seed))
# the cut of the last page on the KERNEL: per row alone and in the batch (exact read sets), random extents
c = t.make_case(23, R=8, N=4, S=16, P=6)
bad = t._extent_cut_failures(bi.bi_paged_decode, c)
if bad:
    fails.append(("extent-cut", bad[:4]))
# negative control on the KERNEL: a batch-keyed tile (2 pages per tile for the batch, 1 alone) must be caught
c = t.make_case(22, R=8, N=3, S=10, P=6)
full2 = t.run(bi.bi_paged_decode, c, pages_per_tile=2)
caught = any(not torch.equal(t.run(bi.bi_paged_decode, c, rows=[r])[0], full2[r]) for r in range(8))
if not caught:
    fails.append(("negative-tile-not-caught",))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("BIATTN-INTERP OK")
'''


def test_kernel_under_the_triton_interpreter():
    """The Triton kernel itself on CPU float32 (TRITON_INTERPRET=1, a fresh process): equal to the float64 semantics
    and to the twin within rounding; row-invariant (alone, permuted, new neighbours, a request's own rows, a wider
    table: torch.equal); NaN poison never read; the batch-keyed tile (pages_per_tile=2 for the batch) is caught."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(MOD), str(Path(__file__).resolve())], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["BIATTN-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# ---------------------------------------------------------------------------
# (5) source pins: the law in the text, the gate and the step script
# ---------------------------------------------------------------------------
def test_kernel_source_keeps_the_law():
    src = MOD.read_text()
    tree = ast.parse(src)
    assert "triton.autotune" not in src and "triton.heuristics" not in src
    launch = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bi_paged_decode"))
    assert "_bi_paged_decode_kernel[(R, Hkv)](" in launch
    assert "num_warps=NUM_WARPS, num_stages=NUM_STAGES" in launch
    code = "\n".join(l.split("#")[0] for l in launch.split('"""')[2].splitlines())
    for bad in (".item(", ".tolist(", ".cpu(", "bool(", "synchronize", "num_splits"):
        assert bad not in code, bad
    assert "PAGES_PER_TILE = 1 " in src and "NUM_WARPS = 4 " in src and "NUM_STAGES = 2" in src
    assert '"n_tiles"])' in src and '"s_ptp"' in src


def test_gate_driver_shape():
    src = GATE.read_text()
    assert src.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')
    assert 'print("BIATTN_GATE_DONE failed=%d" % nfail, flush=True)' in src
    assert "pages_per_tile=2" in src and "flash_attn_with_kvcache" in src


def test_step_script_reads_the_holder_uuid_and_keeps_its_pins():
    text = STEP.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert "HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders" in text
    assert 'UFILE=$HOLDERS/$JOB.gpu_uuid' in text and "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in text
    assert 'mkdir "$OUTROOT/biattn_active"' in text and "STEP_MIN=15" in text and "--export=NONE" in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-biattn" in text and "biattn_gate.py" in text
    # the uuid is read before the claim
    assert text.index('UFILE=$HOLDERS/$JOB.gpu_uuid') < text.index('mkdir "$OUTROOT/biattn_active"')
    assert "<<" not in text and "ulimit" not in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
