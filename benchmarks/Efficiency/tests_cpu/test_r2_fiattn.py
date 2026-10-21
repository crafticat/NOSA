"""CPU tests of the FIATTN comparison (benchmarks/Efficiency/r2_fiattn_bench.py, r2_fiattn_step.sh;
docs/R2_FIATTN_PLAN.md): everything that needs no GPU and no FlashInfer: the geometry, the seeded cases, the
FlashInfer index packing against piece 3's page tables, the flat-address twin of FlashInfer's paged_kv_t and of the
variant's bias read (lessons L28), the fp64 reference against piece 3's oracle, the poison plant, every gate's logic
with fake arms (positives and the negatives each gate must catch), the ncu CSV summary, the variant source, the
predictions table against the plan doc, the refusals and the step's DRYRUN. SOURCE ONLY; the lead runs them on a
compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
PIECE3_PATH = "nosi/nosi/bi_paged_attention.py"
PIECE3_BLOB = "39ed5e1894ca1662f92e384a10e01a4eaab99042"
SKV_BLOB = "5c2e7b25b12020e32dc56e0f6b5189b4884c2289"
NOSI = "/mnt/beegfs/ojerbi/nosi-r2-fiattn"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load(EFF / "r2_fiattn_bench.py", "r2_fiattn_bench_under_test")
GEOM = dict(Hq=4, Hkv=2, D=16, G=2)                     # a small geometry with NOSA's structure (GQA, 2 KV heads)


def piece3():
    """Piece 3 v2 from THIS tree, checked to be the pinned blob (the step pins the same blob)."""
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:" + PIECE3_PATH], capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip() != PIECE3_BLOB:
        pytest.skip("piece 3 at HEAD is not the pinned blob %s (%s)" % (PIECE3_BLOB, r.stdout.strip()))
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", PIECE3_PATH], capture_output=True,
                           text=True)
    assert dirty.stdout.strip() == "", "piece 3 has local changes"
    return _load(ROOT / PIECE3_PATH, "piece3_for_fiattn_tests")


@pytest.fixture(scope="module")
def case():
    return M.make_case(GEOM, 2, 4, 7)


# ------------------------------------------------------------------ FA1 geometry, cells, arms
def test_FA1_geometry_cells_arms():
    """FA1: the geometry from a HF config (and from NOSA-8B's own config when present: 32 / 2 / 128); bad geometry,
    cells and arms are refused; spread keeps the first and last rows."""
    assert M.geometry_from_config(dict(num_attention_heads=32, num_key_value_heads=2, hidden_size=4096,
                                       head_dim=128)) == dict(Hq=32, Hkv=2, D=128, G=16)
    assert M.geometry_from_config(dict(num_attention_heads=8, num_key_value_heads=2, hidden_size=512))["D"] == 64
    for bad in (dict(num_attention_heads=6, num_key_value_heads=4, hidden_size=384),
                dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=4, head_dim=24)):
        with pytest.raises(ValueError):
            M.geometry_from_config(bad)
    cfg = Path("/mnt/beegfs/ojerbi/models/NOSA-8B/config.json")
    if cfg.is_file():
        assert M.geometry_from_config(json.loads(cfg.read_text())) == dict(Hq=32, Hkv=2, D=128, G=16)
    assert M.parse_cells("8x1,8x4,32x1,32x4") == [(8, 1), (8, 4), (32, 1), (32, 4)]
    for bad in ("8x5", "0x1", "8x1,8x1", "8-1", ""):
        with pytest.raises(ValueError):
            M.parse_cells(bad)
    assert M.parse_arms("P,F8") == ["P", "F8"]
    for bad in ("P,XX", "P,P", ""):
        with pytest.raises(ValueError):
            M.parse_arms(bad)
    assert M.spread(5, 16) == [0, 1, 2, 3, 4]
    s = M.spread(128, 16)
    assert len(s) == 16 and s[0] == 0 and s[-1] == 127
    assert M.TAIL_BLOCK == 251 and M.S_SLOTS == 256 and M.FIXED_SPLIT == 8


# ------------------------------------------------------------------ FA2 the seeded case
def test_FA2_case_lists_locality_and_determinism(case):
    """FA2: per (row, head) 64 distinct blocks in ascending order: block 0 first, the tail block 251 last, the 15
    recent blocks present; page_table = the entry's block -> slot map; a request's rows share >= 60 pages with its
    first row and sit at consecutive positions; the alt batch changes exactly the odd rows; the hazard rows have 5, 8,
    9 pages on the extra entry; the same seed gives the same case, with_data=False the same lists, another seed
    another case."""
    f = case["full"]
    r_, hkv, p_ = f["page_key"].shape
    assert (r_, hkv, p_) == (8, 2, 64) and case["N"] == 3
    recent = set(range(M.TAIL_BLOCK - M.N_RECENT, M.TAIL_BLOCK))
    for r in range(r_):
        n = int(f["seq_idx"][r])
        assert n == r // 4
        for h in range(hkv):
            blk = f["page_key"][r, h].tolist()
            assert int(f["num_pages"][r, h]) == 64 and len(set(blk)) == 64 and blk == sorted(blk)
            assert blk[0] == 0 and blk[-1] == M.TAIL_BLOCK and recent <= set(blk)
            assert f["page_table"][r, h].tolist() == case["slot_of_block"][n, torch.tensor(blk)].tolist()
            base = set(f["page_key"][(r // 4) * 4, h].tolist())
            assert len(base & set(blk)) >= 64 - M.N_SWAP
    ll = f["last_len"].view(2, 4)
    assert bool(((ll[:, 1:] - ll[:, :-1]) == 1).all()) and int(ll.min()) >= 1 and int(ll.max()) <= 64
    assert M.changed_rows(M.batch_of(case, "full"), M.batch_of(case, "alt")) == [1, 3, 5, 7]
    s = case["short"]
    assert s["num_pages"][:, 0].tolist() == list(M.SHORT_COUNTS) and set(s["seq_idx"].tolist()) == {2}
    assert set(s["last_len"].tolist()) == {M.SHORT_LAST}
    again = M.make_case(GEOM, 2, 4, 7)
    lists = M.make_case(GEOM, 2, 4, 7, with_data=False)
    other = M.make_case(GEOM, 2, 4, 8)
    for key in ("page_table", "page_key", "num_pages", "last_len", "seq_idx"):
        assert torch.equal(again["full"][key], f[key]) and torch.equal(lists["full"][key], f[key])
        assert torch.equal(lists["alt"][key], case["alt"][key])
    assert torch.equal(again["k"], case["k"]) and torch.equal(again["full"]["q"], f["q"]) and lists["k"] is None
    assert not torch.equal(other["full"]["page_key"], f["page_key"])
    assert float(case["bias"].float().abs().max()) > 4.0                   # the bias has a large range (gate CW)
    h = M.batch_of(case, "hazard")
    assert h["q"].shape[0] == r_ + 3 and torch.equal(h["page_table"][r_:], s["page_table"])


# ------------------------------------------------------------------ FA3 / FA4 packing
def test_FA3_option_a_packing_equals_piece3_tables(case):
    """FA3: piece 3's build_page_table on the case's (block id, slot) pairs, given in a shuffled order, gives the
    case's page_table / num_pages; pack_option_a lists, per KV head and row, exactly entry * S + piece 3's slots in
    piece 3's order (short rows included: ragged counts), indptr is their prefix sum, last_page_len = last_len."""
    p3 = piece3()
    b = M.batch_of(case, "hazard")
    pk = b["page_key"].to(torch.int64)
    sl = b["page_table"].to(torch.int64)
    perm = torch.randperm(pk.shape[-1], generator=torch.Generator().manual_seed(3))
    table, npg = p3.build_page_table(pk[..., perm], sl[..., perm])
    assert torch.equal(table, b["page_table"]) and torch.equal(npg, b["num_pages"])
    S = case["S"]
    for h in range(GEOM["Hkv"]):
        indptr, indices, last = M.pack_option_a(b, S, h)
        assert indptr.dtype == indices.dtype == last.dtype == torch.int32
        assert indptr.tolist() == [0] + torch.cumsum(b["num_pages"][:, h].to(torch.int64), 0).tolist()
        assert torch.equal(last, b["last_len"]) and indices.numel() == b["q"].shape[0] * case["P"]
        for r in range(b["q"].shape[0]):
            c = int(b["num_pages"][r, h])
            want = (int(b["seq_idx"][r]) * S + b["page_table"][r, h, :c].to(torch.int64)).tolist()
            assert indices[int(indptr[r]):int(indptr[r + 1])].tolist() == want


def test_FA4_option_b_packing(case):
    """FA4: one call over (row, head) requests, row-major: page id = (entry * S + slot) * 64 * Hkv + h; the last
    page length repeats per head."""
    b = M.batch_of(case, "hazard")
    S, hkv = case["S"], GEOM["Hkv"]
    indptr, indices, last = M.pack_option_b(b, S)
    r_ = b["q"].shape[0]
    assert indptr.numel() == r_ * hkv + 1 and last.tolist() == b["last_len"].repeat_interleave(hkv).tolist()
    for r in range(r_):
        for h in range(hkv):
            i = r * hkv + h
            c = int(b["num_pages"][r, h])
            want = ((int(b["seq_idx"][r]) * S + b["page_table"][r, h, :c].to(torch.int64)) * (64 * hkv) + h).tolist()
            assert indices[int(indptr[i]):int(indptr[i + 1])].tolist() == want


# ------------------------------------------------------------------ FA5 the address twin (L28)
def test_FA5_flat_address_twin_of_flashinfer_views_and_bias(case):
    """FA5 (lessons L28: model the kernel's flat addresses): for every listed page of every (row, head), FlashInfer's
    element offset (page.cuh get_elem_offset: page * stride_page + entry * stride_n + feat) through the option-A view
    and the option-B overlapping view equals the store offset piece 3 reads, and indexing the views gives the store
    element; the variant's bias offset equals the store's bias offset in both options; option B's largest page id
    stays inside its view; non-contiguous stores are refused."""
    S, hkv, d = case["S"], GEOM["Hkv"], GEOM["D"]
    k = torch.arange(case["k"].numel(), dtype=torch.float64).view(case["k"].shape)      # distinct values
    bias = torch.arange(case["bias"].numel(), dtype=torch.float64).view(case["bias"].shape)
    b = M.batch_of(case, "hazard")
    vb = M.kv_view_b(k)
    ip_b, ix_b, _ = M.pack_option_b(b, S)
    assert int(ix_b[:int(ip_b[-1])].max()) < vb.shape[0]
    flat = k.reshape(-1)
    bflat = bias.reshape(-1)
    for h in range(hkv):
        va = M.kv_view_a(k, S, h)
        ip, ix, _ = M.pack_option_a(b, S, h)
        for r in range(b["q"].shape[0]):
            n = int(b["seq_idx"][r])
            c = int(b["num_pages"][r, h])
            for p in (0, c // 2, c - 1):
                slot = int(b["page_table"][r, h, p])
                ga = int(ix[int(ip[r]) + p])
                gb = int(ix_b[int(ip_b[r * hkv + h]) + p])
                for e in (0, 1, 63):
                    for f in (0, d - 1):
                        want = M.store_offset(k, n, slot, e, h, f)
                        assert M.fi_offset(va, ga, e, f) == want and M.fi_offset(vb, gb, e, f) == want
                        assert va[ga, e, 0, f] == flat[want] and vb[gb, e, 0, f] == flat[want]
                    bw = n * bias.stride(0) + (slot * 64 + e) * bias.stride(1) + h * bias.stride(2)
                    assert M.bias_offset(ga, e, *M.bias_args("a", hkv, h)) == bw
                    assert M.bias_offset(gb, e, *M.bias_args("b", hkv)) == bw
                    assert bflat[bw] == bias[n, slot * 64 + e, h]
    with pytest.raises(ValueError):
        M.kv_view_a(k.transpose(2, 3).contiguous().transpose(2, 3), S, 0)
    with pytest.raises(ValueError):
        M.bias_args("c", hkv)


# ------------------------------------------------------------------ FA6 the reference and the poison plant
def test_FA6_fp64_reference_matches_piece3_oracle_and_poison_is_unread(case):
    """FA6: this module's float64 reference equals piece 3's attention_oracle (an independent implementation of the
    same semantics) on full and hazard rows; with zero_bias it equals the oracle on a zero bias; the poisoned caches
    leave the reference bit-identical (only unread rows are NaN) while the plant is NOT a no-op (lessons L34: NaN
    rows exist, among them unlisted pages and tail rows past the extent, and a reader of the whole store sees them)."""
    p3 = piece3()
    b = M.batch_of(case, "hazard")
    rows = list(range(b["q"].shape[0]))
    ref = M.reference_fp64(b, rows, GEOM)
    orc = p3.attention_oracle(b["q"], b["k"], b["v"], b["bias"], b["seq_idx"], b["page_table"], b["num_pages"],
                              b["last_len"])
    assert torch.allclose(ref, orc, rtol=0, atol=1e-12)
    ref0 = M.reference_fp64(b, rows[:4], GEOM, zero_bias=True)
    orc0 = p3.attention_oracle(b["q"][:4], b["k"], b["v"], torch.zeros_like(b["bias"]), b["seq_idx"][:4],
                               b["page_table"][:4], b["num_pages"][:4], b["last_len"][:4])
    assert torch.allclose(ref0, orc0, rtol=0, atol=1e-12)
    assert float((ref[:4] - ref0).abs().max()) > 10 * M.TOL_C              # the bias moves the output
    bf = M.batch_of(case, "full")
    pois, st = M.poison_caches(case, bf)
    bp = dict(bf)
    bp.update(pois)
    full_rows = list(range(bf["q"].shape[0]))
    assert torch.equal(M.reference_fp64(bp, full_rows, GEOM), M.reference_fp64(bf, full_rows, GEOM))
    assert st["poisoned_rows"] > 0 and st["unlisted_pages"] > 0
    assert bool(torch.isnan(pois["k"].float()).any()) and bool(torch.isnan(pois["bias"].float()).any())
    reach = M.reach_map(bf, case["N"], case["S"], GEOM["Hkv"])
    for n in range(2):
        tail = int(case["slot_of_block"][n, M.TAIL_BLOCK])
        mx = int(bf["last_len"][n * 4:(n + 1) * 4].max())
        assert reach[n, tail].tolist() == [mx, mx]
        if mx < 64:
            assert bool(torch.isnan(pois["v"][n, tail * 64 + mx, 0].float()).all())
            assert not bool(torch.isnan(pois["v"][n, tail * 64 + mx - 1, 0].float()).any())
    assert bool(torch.isnan(pois["k"][2].float()).all())                  # the hazard entry: listed by no full row


# ------------------------------------------------------------------ FA7 gate logic with fakes
class Fake:
    """A fake arm: the float64 reference per row (batch-invariant by construction), cast to bf16, plus a planted
    dependence of the given kind; .info mimics FlashInfer's plan per KV head."""
    def __init__(self, kind="exact", info="split", chunk_dep=True):
        self.kind, self.info_kind, self.chunk_dep, self.info = kind, info, chunk_dep, None

    def __call__(self, b):
        r_ = b["q"].shape[0]
        out = M.reference_fp64(b, list(range(r_)), GEOM)
        if self.kind == "batch":
            out = out + 0.05 * r_
        elif self.kind == "order":
            out = out + 0.05 * torch.arange(r_, dtype=out.dtype).view(-1, 1, 1)
        elif self.kind == "siblings":
            out = out + b["q"].double()[:, 0, 0].sum()
        elif self.kind == "poison":
            out = out + 0.0 * b["k"].double().sum()
        split = bool((b["num_pages"] > M.FIXED_SPLIT).any()) if self.info_kind == "split" else False
        chunk = 64 * r_ if self.chunk_dep else 128
        self.info = [dict(split_kv=split, chunk_tokens=chunk, padded_batch_size=r_, cta_tile_q=16)] * GEOM["Hkv"]
        return out.to(torch.bfloat16)


def _suite(case, arm, hazard_pre=True):
    bf, ba, bh = M.batch_of(case, "full"), M.batch_of(case, "alt"), M.batch_of(case, "hazard")
    r_ = case["R"]
    rows = M.spread(r_, 4)
    perms = [("reversed_requests", [n * 4 + u for n in reversed(range(2)) for u in range(4)]),
             ("random", [5, 2, 7, 0, 3, 6, 1, 4])]
    pois, _ = M.poison_caches(case, bf)
    g, full = M.invariance_suite("F8", "2x4", arm, bf, ba, bh, rows, perms, pois, list(range(r_, r_ + 3)),
                                 hazard_pre, True)
    return {x["name"].split("_")[0]: x for x in g}, full


def test_FA7a_invariance_gates_pass_on_an_invariant_arm_and_catch_each_plant(case):
    """FA7a: the exact fake passes I1-I5 (I4 with its precondition met: short rows plan no split alone and split
    batched); each planted dependence fails its gate: batch size -> I1 (and I4), row position -> I2, siblings' q ->
    I3, a read of the whole store -> I5; a plan whose split flag never switches makes I4 VOID (= fail)."""
    g, _ = _suite(case, Fake("exact"))
    assert all(x["passed"] for x in g.values()), [x for x in g.values() if not x["passed"]]
    assert "precondition met" in g["I4"]["detail"]
    for kind, gid in (("batch", "I1"), ("order", "I2"), ("siblings", "I3"), ("poison", "I5")):
        g, _ = _suite(case, Fake(kind))
        assert not g[gid]["passed"], (kind, g[gid])
    g, _ = _suite(case, Fake("batch"))
    assert not g["I4"]["passed"]
    g, _ = _suite(case, Fake("exact", info="never"))
    assert not g["I4"]["passed"] and g["I4"].get("status") == "VOID"
    g, _ = _suite(case, Fake("exact", info="never"), hazard_pre=False)
    assert g["I4"]["passed"]


def test_FA7b_negative_control_states(case):
    """FA7b: the heuristic control is DETECTED only with a batch-dependent plan AND a differing row; a plan that does
    not depend on the batch is VOID (never counted as caught); a dependent plan with identical outputs is BLIND."""
    bf = M.batch_of(case, "full")
    rows = M.spread(case["R"], 4)
    g, _ = M.negative_control("2x4", Fake("batch", chunk_dep=True), bf, rows, True)
    assert g["passed"] and g["status"] == "DETECTED"
    g, _ = M.negative_control("2x4", Fake("batch", chunk_dep=False), bf, rows, True)
    assert not g["passed"] and g["status"] == "VOID"
    g, _ = M.negative_control("2x4", Fake("exact", chunk_dep=True), bf, rows, True)
    assert not g["passed"] and g["status"] == "BLIND"


def test_FA7c_correctness_and_bias_witness(case):
    """FA7c: C passes within TOL_C and fails beyond it or on a NaN; the bias witness is DETECTED for an arm that
    honours the batch's bias (run with bias 0 against the biased reference), VOID when the bias does not move the
    reference, and BLIND for an arm that ignores the bias field."""
    bf = M.batch_of(case, "full")
    rows = [0, 3, 5]
    ref = M.reference_fp64(bf, rows, GEOM)
    ref0 = M.reference_fp64(bf, rows, GEOM, zero_bias=True)
    assert M.correctness_gate("F8", "2x4", ref + 5e-4, ref, M.TOL_C, True)["passed"]
    assert not M.correctness_gate("F8", "2x4", ref + 2e-3, ref, M.TOL_C, True)["passed"]
    bad = ref.clone()
    bad[0, 0, 0] = float("nan")
    assert not M.correctness_gate("F8", "2x4", bad, ref, M.TOL_C, True)["passed"]
    g = M.bias_witness("2x4", Fake("exact"), bf, rows, ref, ref0, M.TOL_C, True)
    assert g["passed"] and g["status"] == "DETECTED"
    g = M.bias_witness("2x4", Fake("exact"), bf, rows, ref0, ref0, M.TOL_C, True)
    assert not g["passed"] and g["status"] == "VOID"

    class IgnoresBias(Fake):
        def __call__(self, b):
            bb = dict(b)
            bb["bias"] = bf["bias"]
            return super().__call__(bb)
    g = M.bias_witness("2x4", IgnoresBias("exact"), bf, rows, ref, ref0, M.TOL_C, True)
    assert not g["passed"] and g["status"] == "BLIND"


def test_FA7d_bits_equal_and_bits_vs():
    """FA7d: bitwise equality counts a sign-of-zero and a NaN payload; D reports rows that differ and never fails."""
    a = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    b = a.clone()
    assert M.bits_equal(a, b)
    b[1, 0, 0] = -0.0
    assert not M.bits_equal(a, b) and torch.equal(a, b)
    assert not M.bits_equal(a, a.float())
    g = M.bits_vs("F8", "t", b, a)
    assert g["passed"] and not g["must"] and g["detail"].startswith("1 of 2 rows")


# ------------------------------------------------------------------ FA8 predictions = the plan doc
def test_FA8_predictions_registered_in_the_plan_doc():
    """FA8: docs/R2_FIATTN_PLAN.md carries the REGISTERED marker and a json block equal to PRED; every band is
    [central, low, high] with low <= central <= high; every cell and must arm has its time band."""
    doc = (ROOT / "docs/R2_FIATTN_PLAN.md").read_text()
    assert "REGISTERED <stamp-by-lead>" in doc
    m = re.search(r"<!-- PRED-BEGIN -->\s*```json\s*(.*?)```\s*<!-- PRED-END -->", doc, flags=re.S)
    assert m, "no PRED block in the plan doc"
    assert json.loads(m.group(1)) == json.loads(json.dumps(M.PRED))

    def bands(x):
        if isinstance(x, list) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from bands(v)
    for c, lo, hi in bands(M.PRED):
        assert lo <= c <= hi
    for cell in ("8x1", "8x4", "32x1", "32x4"):
        assert cell in M.PRED["p3_ms"]
        for arm in ("F8", "F0"):
            assert cell in M.PRED["ratio_vs_p3"][arm]
    assert M.PRED["tol_c_abs"] == M.TOL_C


# ------------------------------------------------------------------ FA9 ncu summary and argv
def test_FA9_ncu_summary_filter_and_argv(tmp_path):
    """FA9: the ncu raw CSV summary sums DRAM bytes per NVTX-tagged arm (a units row and an untagged torch kernel
    are skipped and listed as stray), counts kernels against KERNELS_PER_CALL and divides by the ideal / union
    bytes; the kernel filter matches every attention kernel and no torch kernel; ncu_cap skips when the budget is
    short; the child argv carries the cold-L2 cache control, the filter, the arms, --sharedkv only with S5 and
    --bias-free only when asked."""
    nv = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    cols = ["ID", "Process ID", nv, "Kernel Name"] + list(M.NCU_REQUIRED)
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(cols)
    w.writerow(["", "", "", "", "nsecond", "byte", "byte"])
    i = 0
    for tag, names, rd in (("P", ("_bi_split_kernel", "_bi_combine_kernel"), 100.0),
                           ("F8", ("BatchPrefillWithPagedKVCacheKernel", "PersistentVariableLengthMergeStatesKernel") * 2,
                            50.0)):
        for nm in names:
            w.writerow([str(i), "9", '9 "<default domain>:fia|%s:none"' % tag, "void flashinfer::%s<...>(...)" % nm,
                        "1000", str(rd), "10"])
            i += 1
    w.writerow([str(i), "9", "", "void at::native::vectorized_elementwise_kernel<4>", "5", "7", "1"])
    rows = M.ncu_rows(buf.getvalue(), M.NCU_REQUIRED)
    s = M.ncu_summary(rows, ["P", "F8"], ideal=400.0, union=200.0, rpr=4)
    assert s["arms"]["P"]["read"] == 200.0 and s["arms"]["P"]["n_kernels"] == 2 == s["arms"]["P"]["expected_kernels"]
    assert s["arms"]["F8"]["read"] == 200.0 and s["arms"]["F8"]["n_kernels"] == 4
    assert s["arms"]["P"]["read_over_ideal"] == 0.5 and s["arms"]["F8"]["read_over_union"] == 1.0
    assert "IN" in s["arms"]["P"]["band"] and len(s["stray"]) == 1
    rx = re.compile(M.KERNEL_FILTER[len("regex:"):])
    for nm in M.ATTN_KERNELS:
        assert rx.search(nm)
    for nm in ("vectorized_elementwise_kernel", "CatArrayBatchedCopy", "bitonicSortKVInPlace"):
        assert not rx.search(nm)
    assert M.ncu_cap(3000, 600) == 600 and M.ncu_cap(300, 600) == 180 and M.ncu_cap(150, 600) is None

    class A:
        ncu, config, biattn, sharedkv, seed = "/x/ncu", "/c.json", "/p3.py", "/s5.py", 5
    argv = M.ncu_argv(A, 32, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "F8", "F0", "S5"], False)
    j = argv.index("--cache-control")
    assert argv[j + 1] == "all" and M.KERNEL_FILTER in argv and "--sharedkv" in argv and "--bias-free" not in argv
    assert argv[argv.index("--mode") + 1] == "ncu-cell" and argv[argv.index("--cells") + 1] == "32x4"
    argv = M.ncu_argv(A, 8, 1, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "F8"], True)
    assert "--sharedkv" not in argv and "--bias-free" in argv


# ------------------------------------------------------------------ FA10 refusals
def test_FA10_refusals(tmp_path, monkeypatch):
    """FA10: main refuses (2) an existing --out, a missing config / module file, bad cells or arms and bad numbers,
    before it creates anything; on a CPU node it refuses for the missing CUDA device; with CUDA it refuses an unset
    FLASHINFER_WORKSPACE_BASE. The FlashInfer version check accepts 0.5.3 (with a local suffix) only."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(num_attention_heads=32, num_key_value_heads=2, hidden_size=4096, head_dim=128)))
    mod = tmp_path / "p3.py"
    mod.write_text("")
    out = tmp_path / "out"
    base = ["--out", str(out), "--config", str(cfg), "--biattn", str(mod)]
    assert M.main(base + ["--cells", "8x5"]) == 2
    assert M.main(base + ["--arms", "P,XX"]) == 2
    assert M.main(base + ["--budget-s", "10"]) == 2
    assert M.main(["--out", str(out), "--config", str(tmp_path / "none.json"), "--biattn", str(mod)]) == 2
    assert M.main(base + ["--sharedkv", str(tmp_path / "none.py")]) == 2
    assert M.main(["--config", str(cfg)]) == 2
    assert not out.exists()
    out.mkdir()
    assert M.main(base) == 2
    out.rmdir()
    if not torch.cuda.is_available():
        assert M.main(base) == 2 and not out.exists()
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)
    a = M.parse_args(base)
    assert "FLASHINFER_WORKSPACE_BASE" in M.refuse_common(a)
    assert M.version_ok("0.5.3") and M.version_ok("0.5.3+cu128torch2.8")
    assert not M.version_ok("0.5.4") and not M.version_ok("0.5.30") and not M.version_ok("unknown")


# ------------------------------------------------------------------ FA11 the variant source and jit_args
def test_FA11_variant_source_and_jit_args():
    """FA11: jit_args follow gen_customize_batch_prefill_module('fa2', uri, dtype_q, dtype_kv, dtype_o, idtype,
    head_dim_qk, head_dim_vo, tensor names, tensor dtypes, scalar names, scalar dtypes, variant name, variant decl);
    the variant guards the paged-only read (a trait: the ragged kernels compile with the same variant), reads the
    bias only inside the extent, scales (logit + bias) by the scalar scale and sets sm_scale_log2 = log2(e) (the
    ALiBi pattern); every declared parameter is used; the braces balance."""
    ja = M.jit_args_for(128)
    assert len(ja) == 13 and ja[0] == "r2_fiattn_cisbias_bf16_d128" and ja[5] == ja[6] == 128
    assert ja[1] is ja[2] is ja[3] is torch.bfloat16 and ja[4] is torch.int32
    assert ja[7] == ["cis_bias"] and ja[8] == ["nv_bfloat16"] and len(ja[9]) == len(ja[10]) == 4
    assert ja[11] == M.VARIANT_NAME and ja[12] == M.VARIANT_DECL
    src = M.VARIANT_DECL
    for needle in ("r2fi_has_paged_kv<Params>::value", "kv_idx < kv_len", "sm_scale_log2 = math::log2e",
                   "(logits + b) * static_cast<float>(params.cis_sm_scale)", "window_left = kv_len",
                   "static constexpr bool use_softmax = true", "struct %s : AttentionVariantBase" % M.VARIANT_NAME,
                   "params.paged_kv.indices[params.paged_kv.indptr[batch_idx] + pg]", "__bfloat162float"):
        assert needle in src, needle
    for name in ja[7] + ja[9]:
        assert "params.%s" % name in src, name
    assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    assert not any(n.startswith("maybe") for n in ja[7])            # not optional tensors


# ------------------------------------------------------------------ FA12 / FA13 the step script
def test_FA12_step_dryrun_refusals_and_pins(tmp_path):
    """FA12 (as test_r2_sharedkv_ncu.py N5): a wrong argument count is a usage error (2); a missing uuid file, a GPU-
    prefix and two lines STOP (1); FIATTN_HOLDERS outside DRYRUN is refused; a wrong --nosi stops at HEAD. On a clean
    checkout of THIS worktree at HEAD the DRYRUN prints both pins and the bench arguments."""
    script = EFF / "r2_fiattn_step.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    uid = "f31fc723-9dfe-407a-9947-28173556232a"

    def run(job, content=None, dry=True, sha="0" * 40, extra=()):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, FIATTN_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"] + list(extra), env=e,
                              capture_output=True, text=True, timeout=300)
    r = run(320, extra=["x"])
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(321)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(322, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(323, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(324, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(325, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(NOSI).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % NOSI)
    r = run(326, uid + "\n", sha=head.stdout.strip())
    assert r.returncode == 0, r.stderr[-400:]
    assert "piece 3 7c826c1bc7235ee9153e3af82c99d85836915416:%s blob %s" % (PIECE3_PATH, PIECE3_BLOB) in r.stdout
    assert "sharedkv5 5eb75144644d547e5c601f28606b91136ab9df3a:nosi/nosi/bi_paged_attention_shared.py blob %s" \
        % SKV_BLOB in r.stdout
    for wd in ("--mode main", "--cells 8x1\\,8x4\\,32x1\\,32x4", "--arms P\\,F8\\,F0\\,FH\\,S5",
               "--ncu /opt/nvidia/nsight-compute/2025.1.0/ncu", "--ncu-cap 600", "--budget-s 2400",
               "--config /mnt/beegfs/ojerbi/models/NOSA-8B/config.json", "--time=45",
               "FLASHINFER_WORKSPACE_BASE=/mnt/beegfs/ojerbi/nosa-env/jit/fiattn_t1_326/flashinfer",
               "TVM_FFI_CACHE_DIR=/mnt/beegfs/ojerbi/nosa-env/jit/fiattn_t1_326/tvm_ffi"):
        assert wd in r.stdout, wd


def test_FA13_step_static_shape():
    """FA13: the step's body is one main() (lessons L26: an edit during a run cannot corrupt it), its claim is
    fiattn_active with an owner file taken after the existence check, the budget fits the step, the pins are the
    blobs this file names, and no interpreter name appears in it."""
    src = (EFF / "r2_fiattn_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    assert src.index('[ ! -e "$OUT" ]') < src.index('mkdir "$OUTROOT/fiattn_active"') < src.index("fiattn_active/owner")
    step_min = int(re.search(r"^STEP_MIN=(\d+)$", src, flags=re.M).group(1))
    budget = int(re.search(r"^BUDGET_S=(\d+)$", src, flags=re.M).group(1))
    assert step_min <= 45 and budget + 300 <= step_min * 60
    assert "BIATTN_BLOB=%s" % PIECE3_BLOB in src and "SKV_BLOB=%s" % SKV_BLOB in src
    for word in ("awk", "perl", "python3 -", "sed -i", "<<"):
        assert word not in src, word
