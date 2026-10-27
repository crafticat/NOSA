"""CPU tests of the fia3 arms (benchmarks/Efficiency/r2_fiattn_bench.py, the [fia3] parts; r2_fia3_step.sh;
docs/NOTES_fia3.md): everything that needs no GPU and no FlashInfer. The FlashInfer wrapper is replaced by a CPU fake
that holds the plan's tensors and reads them at run time through the bench's flat-address twin (lessons L28), as the
existing tests replace the arms by fakes. SOURCE ONLY; the lead runs them on a compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia3.py
"""
from __future__ import annotations

import csv
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
NOSI3 = "/mnt/beegfs/ojerbi/nosi-r2-fia3"
OLD_SHA = "3948fab"                                     # the fia2 bench (the data must stay bitwise the same)
S1_RECORDED = Path("/mnt/beegfs/ojerbi/r2-runs/sharedkv/skv_probe_s1a_2191262/skv_tables.pt")
FI068 = Path("/mnt/beegfs/ojerbi/venvs/vllm-port/lib/python3.10/site-packages/flashinfer")
VSTEP_PIN = ("dde91d4cfe345810c57450047e81e0be1524f38d", "nosi/nosi/verify/verify_step.py",
             "769047f602e81c4f2792888210ac4a65f8607792")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load(EFF / "r2_fiattn_bench.py", "r2_fiattn_bench_fia3_under_test")
GEOM = dict(Hq=4, Hkv=2, D=16, G=2)                     # NOSA's structure (GQA, 2 KV heads), small


@pytest.fixture(scope="module")
def case():
    return M.make_case(GEOM, 2, 4, 7)


def as_f64(b):
    out = dict(b)
    for f in ("q", "k", "v", "bias"):
        out[f] = b[f].double().contiguous()
    return out


def crafted(case):
    """Request 0: extents 10 .. 13 (odd bit phases of the mask rows). Request 1 CROSSES a block edge: rows 4, 5 end
    in block 250 (63, 64 tokens; the tail 251 dropped from their lists), rows 6, 7 start block 251 (1, 2 tokens)."""
    b = M.select(M.batch_of(case, "full"), range(8))
    b["last_len"] = torch.tensor([10, 11, 12, 13, 63, 64, 1, 2], dtype=torch.int32)
    npg = b["num_pages"].clone()
    npg[4:6] = 63
    b["num_pages"] = npg
    return b


# ------------------------------------------------------------------ the fake FlashInfer prefill wrapper
class FakePrefillWrapper:
    """CPU fake of BatchPrefillWithPagedKVCacheWrapper (non-graph mode, 0.6.8 prefill.py:1670-2069, :2136-2453):
    plan() HOLDS the tensors it gets (a device tensor's .to(device) is the tensor itself) and records its keywords;
    run() reads the held tensors at RUN time through the bench's twin; a plan without packed_custom_mask runs with no
    mask (mode NON_CAUSAL: our variant's LogitsMask is then called on the last tiles only, which the fake drops)."""
    def __init__(self, float_workspace_buffer, kv_layout="NHD", use_cuda_graph=False, backend="auto", jit_args=None,
                 jit_kwargs=None):
        self.jit_args, self.backend = jit_args, backend
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
        self._pin_memory_int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
        self._custom_mask_buf = None
        self.plans, self.kw = 0, None

    def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
        self._float_workspace_buffer, self._int_workspace_buffer = float_workspace_buffer, int_workspace_buffer

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_qo_heads, num_kv_heads,
             head_dim_qk, page_size, head_dim_vo=None, custom_mask=None, packed_custom_mask=None, causal=False,
             sm_scale=None, q_data_type="float16", kv_data_type=None, fixed_split_size=None, disable_split_kv=False):
        self._qo, self._ip, self._ix, self._last = qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len
        self._custom_mask_buf = packed_custom_mask
        self._page = page_size
        self.kw = dict(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim_qk,
                       fixed_split_size=fixed_split_size, disable_split_kv=disable_split_kv, sm_scale=sm_scale)
        split = (not disable_split_kv) and fixed_split_size is not None
        self._plan_info = [int(qo_indptr.numel()) - 1, int(qo_indptr[-1]), 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           int(split)]
        self.plans += 1

    def run(self, q, paged_kv_cache, *args, out=None):
        bias, mask, mi, page_mul, entry_stride, head, scale = args
        k, v = paged_kv_cache
        res = M.fi_prefill_twin(q, k, v, bias, self._qo, self._ip, self._ix, self._last, self._page, mask, mi,
                                page_mul, entry_stride, head, scale, custom=self._custom_mask_buf is not None)
        out.copy_(res.to(out.dtype))
        return out


FAKE_FI = types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=FakePrefillWrapper, __version__="0.5.3")


def fp_arm(name, split, mask, S):
    return M.FPArm(FAKE_FI, name, split, mask, GEOM, S, 4, ["fake-jit-args"], "cpu")


def fp_inputs(b, rpr, S, ones=False):
    u = M.fp_union(b, rpr, S, ones=ones)
    ip, ix, last = M.fp_csr(u)
    m, mi, _ = M.fp_pack(u)
    return u, ip, ix, last, m, mi


def twin_all(b, rpr, S, inputs, custom=True):
    """Both KV heads through the twin, interleaved as FPArm.launch does."""
    _, ip, ix, last, m, mi = inputs
    g, hkv = GEOM["G"], GEOM["Hkv"]
    qo = torch.arange(b["q"].shape[0] // rpr + 1) * rpr
    outs = [M.fi_prefill_twin(b["q"][:, h * g:(h + 1) * g], M.kv_view_a(b["k"], S, h), M.kv_view_a(b["v"], S, h),
                              b["bias"], qo, ip[h], ix[h], last[h], 64, m, mi[h], *M.bias_args("a", hkv, h),
                              GEOM["D"] ** -0.5, custom=custom) for h in range(hkv)]
    return torch.cat(outs, dim=1)


# ------------------------------------------------------------------ FC1 cells, arguments, versions
def test_FC1_cells_arguments_versions(tmp_path):
    """FC1: cells up to B 256 and the s1 suffix; repeats and bad sizes refused; the ncu order (fia2's for the old
    cells); FlashInfer 0.5.3 and 0.6.8.post1 accepted; V1 without --verify-step and an s1 cell without --s1-tables
    are refusals (2) that create nothing."""
    assert M.parse_cells("32x4,216x4,32x4s1,216x4s1") == [(32, 4), (216, 4), (32, 4, "s1"), (216, 4, "s1")]
    for bad in ("257x4", "216x5", "32x4s2", "32x4,32x4", "32x4s1,32x4s1", "0x4s1"):
        with pytest.raises(ValueError):
            M.parse_cells(bad)
    assert [M.cell_tag(*c) for c in M.parse_cells("216x4,216x4s1")] == ["216x4", "216x4s1"]
    assert M.cell_parts((8, 1)) == (8, 1, "gen") and M.cell_parts((8, 4, "s1")) == (8, 4, "s1")
    assert sorted(M.parse_cells("8x1,32x4,8x4,32x1"), key=M.ncu_key) == [(32, 4), (32, 1), (8, 4), (8, 1)]
    assert sorted(M.parse_cells("32x4s1,32x4,216x4s1,216x4"), key=M.ncu_key) == \
        [(216, 4), (32, 4), (216, 4, "s1"), (32, 4, "s1")]
    assert M.version_ok("0.5.3") and M.version_ok("0.6.8.post1") and M.version_ok("0.6.8.post1+cu128")
    assert not M.version_ok("0.6.8") and not M.version_ok("0.6.13")
    for a in ("FP0", "FP8", "FPU", "V1"):
        assert a in M.ARMS
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(num_attention_heads=32, num_key_value_heads=2, hidden_size=4096, head_dim=128)))
    mod = tmp_path / "p3.py"
    mod.write_text("")
    out = tmp_path / "out"
    base = ["--out", str(out), "--config", str(cfg), "--biattn", str(mod)]
    assert M.main(base + ["--arms", "P,V1"]) == 2
    assert M.main(base + ["--cells", "32x4s1"]) == 2
    assert M.main(base + ["--arms", "P,V1", "--verify-step", str(tmp_path / "none.py")]) == 2
    assert M.main(base + ["--cells", "32x4s1", "--s1-tables", str(tmp_path / "none.pt")]) == 2
    assert not out.exists()
    a = M.parse_args(base + ["--arms", "P,FP0,V1", "--verify-step", str(mod), "--cells", "216x4s1",
                             "--s1-tables", str(mod)])
    assert a.cells == [(216, 4, "s1")] and a.arms == ["P", "FP0", "V1"]


# ------------------------------------------------------------------ FC2 altreq; the fia2 data unchanged
def test_FC2_altreq_and_the_fia2_data_bitwise_unchanged(case, tmp_path):
    """FC2: altreq changes exactly the odd requests (q and lists; canonical lists; the same extents), with_data=False
    gives the same altreq lists; and make_case's full / alt / short batches and K / V / bias equal the 3948fab bench's
    (git show), bitwise, for a 4-row and a 1-row cell: the fia2 cells keep their data."""
    f, q = case["full"], case["altreq"]
    assert M.changed_rows(M.batch_of(case, "full"), M.batch_of(case, "altreq")) == [4, 5, 6, 7]
    for r in range(8):
        same = torch.equal(f["page_key"][r], q["page_key"][r]) and torch.equal(f["q"][r], q["q"][r])
        assert same == (r < 4)
        for h in range(2):
            blk = q["page_key"][r, h].tolist()
            assert blk[0] == 0 and blk[-1] == M.TAIL_BLOCK and blk == sorted(blk) and len(set(blk)) == 64
    assert torch.equal(q["last_len"], f["last_len"]) and torch.equal(q["seq_idx"], f["seq_idx"])
    assert torch.equal(M.make_case(GEOM, 2, 4, 7, with_data=False)["altreq"]["page_key"], q["page_key"])
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:benchmarks/Efficiency/r2_fiattn_bench.py" % OLD_SHA],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("the 3948fab bench is not readable from git here")
    p = tmp_path / "old_bench_3948fab.py"
    p.write_text(r.stdout)
    old = _load(p, "r2_fiattn_bench_3948fab_for_fc2")
    for bb, rr in ((2, 4), (2, 1)):
        new_c, old_c = M.make_case(GEOM, bb, rr, 7), old.make_case(GEOM, bb, rr, 7)
        for k in ("full", "alt", "short"):
            assert set(new_c[k]) == set(old_c[k])
            for fld in old_c[k]:
                assert torch.equal(new_c[k][fld], old_c[k][fld]), (bb, rr, k, fld)
        for fld in ("k", "v", "bias", "slot_of_block"):
            assert torch.equal(new_c[fld], old_c[fld]), (bb, rr, fld)


# ------------------------------------------------------------------ FC3 the union
def test_FC3_union_equals_a_set_model(case):
    """FC3: per (KV head, request) the union is the sorted set of the rows' (block id, slot) pairs, upage = entry x S
    + slot; ext = 64 / last_len (the row's LAST LISTED page) / 0; uidx maps every list position back; klast = the
    largest extent in the last union page; ones = the reach for every row; rows that are not whole requests refused."""
    S = case["S"]
    for b in (M.batch_of(case, "full"), crafted(case)):
        u = M.fp_union(b, 4, S)
        for h in range(2):
            for n in range(2):
                rows = list(range(n * 4, n * 4 + 4))
                lists = {r: [(int(b["page_key"][r, h, i]), int(b["page_table"][r, h, i]))
                             for i in range(int(b["num_pages"][r, h]))] for r in rows}
                pairs = sorted({kv for r in rows for kv in lists[r]})
                c = int(u["ucnt"][h, n])
                assert c == len(pairs)
                assert u["ukey"][h, n, :c].tolist() == [k for k, _ in pairs]
                assert u["upage"][h, n, :c].tolist() == [n * S + s for _, s in pairs]
                for ui, r in enumerate(rows):
                    own = {kv: i for i, kv in enumerate(lists[r])}
                    last = len(lists[r]) - 1
                    want = [0 if kv not in own else (int(b["last_len"][r]) if own[kv] == last else 64) for kv in pairs]
                    assert u["ext"][h, n, ui, :c].tolist() == want
                    assert int(u["ext"][h, n, ui, c:].abs().sum()) == 0
                    for i, kv in enumerate(lists[r]):
                        assert pairs[int(u["uidx"][h, n, ui, i])] == kv
                assert int(u["klast"][h, n]) == max(u["ext"][h, n, :, c - 1].tolist())
    b = M.batch_of(case, "full")
    u, uo = M.fp_union(b, 4, S), M.fp_union(b, 4, S, ones=True)
    assert torch.equal(uo["ext"], u["ext"].amax(dim=2, keepdim=True).expand_as(u["ext"]))
    assert int(M.fp_union(crafted(case), 4, S)["klast"][0, 1]) == 2       # the crossing request's last page: 251
    with pytest.raises(ValueError):
        M.fp_union(M.select(b, [0, 1, 2]), 4, S)


# ------------------------------------------------------------------ FC4 the packed mask
def slow_pack(u):
    """Per-bit reference of FlashInfer's packed layout (variants.cuh DefaultAttention): segments in (head, request)
    order, byte aligned; bit u x kv_len + t is set iff row u sees at least t % 64 + 1 tokens of union page t // 64."""
    ext = u["ext"].tolist()
    hkv, nb, rpr = len(ext), len(ext[0]), len(ext[0][0])
    out, starts = [], []
    for h in range(hkv):
        for n in range(nb):
            kv = (int(u["ucnt"][h, n]) - 1) * 64 + int(u["klast"][h, n])
            bits = [1 if t % 64 < ext[h][n][r][t // 64] else 0 for r in range(rpr) for t in range(kv)]
            starts.append(len(out))
            for k in range((len(bits) + 7) // 8):
                out.append(sum(bits[8 * k + i] << i for i in range(8) if 8 * k + i < len(bits)))
    starts.append(len(out))
    return out, starts


def test_FC4_packed_mask_equals_the_per_bit_reference(case):
    """FC4: fp_pack's bytes, absolute byte offsets and kv_len equal the per-bit reference on the full, altreq and
    crafted batches (precondition, L34: some mask rows start in the middle of a byte, so the shifted path runs); the
    dump byte holds no bit."""
    S = case["S"]
    phases = 0
    for b in (M.batch_of(case, "full"), M.batch_of(case, "altreq"), crafted(case)):
        u = M.fp_union(b, 4, S)
        m, mi, kvl = M.fp_pack(u)
        want, starts = slow_pack(u)
        hkv, nb = u["ucnt"].shape
        assert mi.dtype == torch.int32 and m.dtype == torch.uint8
        assert mi.tolist() == [starts[h * nb:h * nb + nb + 1] for h in range(hkv)]
        assert m[:starts[-1]].tolist() == want
        assert int(m[-1]) == 0 and int(m[starts[-1]:].to(torch.int64).sum()) == 0
        assert kvl.tolist() == [[(int(u["ucnt"][h, n]) - 1) * 64 + int(u["klast"][h, n]) for n in range(nb)]
                                for h in range(hkv)]
        phases += sum(1 for x in kvl.reshape(-1).tolist() if x % 8)
        assert m.numel() == M.fp_mask_cap(hkv, nb, 4, u["ext"].shape[-1]) + 1
    assert phases > 0


# ------------------------------------------------------------------ FC5 the twin through FlashInfer's addresses
def test_FC5_flat_address_twin_equals_the_reference(case):
    """FC5 (L28): the FP arm's inputs (union CSR, last_page_len, packed mask, byte offsets, option-A views, the
    variant's bias offsets) read through FlashInfer's flat addresses give the bench's fp64 reference on every row, for
    the full, altreq and crafted (block-edge) batches; the ones mask does not (MW's premise)."""
    S = case["S"]
    ref = None
    for bb in (M.batch_of(case, "full"), M.batch_of(case, "altreq"), crafted(case)):
        b = as_f64(bb)
        out = twin_all(b, 4, S, fp_inputs(b, 4, S))
        ref = M.reference_fp64(b, list(range(8)), GEOM)
        assert torch.allclose(out, ref, rtol=0, atol=1e-10)
    b = as_f64(M.batch_of(case, "full"))
    ref = M.reference_fp64(b, list(range(8)), GEOM)
    ones = twin_all(b, 4, S, fp_inputs(b, 4, S, ones=True))
    assert float((ones - ref).abs().max()) > M.TOL_C
    nomask = twin_all(b, 4, S, fp_inputs(b, 4, S), custom=False)
    assert float((nomask - ref).abs().max()) > M.TOL_C


# ------------------------------------------------------------------ FC6 FPArm with the fake wrapper
def test_FC6_fp_arm_end_to_end_plan_reuse_and_its_negative_control(case):
    """FC6: FPArm drives the fake wrapper like FlashInfer: per KV head, num_qo_heads = the group, num_kv_heads 1,
    head_dim D, page 64; FP0 plans disable_split_kv with our mask buffer as packed_custom_mask (the SAME tensor), FP8
    fixed_split_size 8 (and its float need), FPU no mask; the output equals the reference up to bf16 rounding. Plan
    reuse: relayer writes the next layer in place (no new plan, the same held tensors) and equals a fresh plan bitwise
    with other union sizes (precondition, made certain: request 1 of the next layer is four copies of one row, so its
    union is 64 pages against >= 68); building into NEW tensors (the planted defect) is caught by R_REUSE. MW detects
    the ones arm; relayer refuses a split arm."""
    S = case["S"]
    bf, bq = M.batch_of(case, "full"), M.batch_of(case, "altreq")
    layer2 = M.select(bf, [0, 1, 2, 3, 4, 4, 4, 4])
    ref = M.reference_fp64(bf, list(range(8)), GEOM)
    arm = fp_arm("FP0", "none", "rows", S)
    out = arm(bf)
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (8, 4, 16)
    assert torch.allclose(out.double(), ref, rtol=1e-2, atol=1e-3)
    for h, w in enumerate(arm.wr):
        assert w.jit_args == ["fake-jit-args"] and w.backend == "fa2" and w.plans == 1
        assert w.kw["num_qo_heads"] == 2 and w.kw["num_kv_heads"] == 1 and w.kw["head_dim"] == 16
        assert w.kw["disable_split_kv"] is True and w.kw["fixed_split_size"] is None
        assert w._custom_mask_buf is arm.bufs["mask"]
        assert w._ip.data_ptr() == arm.bufs["indptr"][h].data_ptr()
        assert w._ix.data_ptr() == arm.bufs["ix"][h].data_ptr()
    held = [(w._ip.data_ptr(), w._ix.data_ptr(), w._last.data_ptr()) for w in arm.wr]
    uc_a = arm.u["ucnt"].clone()
    arm.relayer(layer2)
    assert [w.plans for w in arm.wr] == [1, 1]
    assert [(w._ip.data_ptr(), w._ix.data_ptr(), w._last.data_ptr()) for w in arm.wr] == held
    reused = arm.launch().clone()
    assert arm.u["ucnt"][:, 1].tolist() == [64, 64] and int(uc_a[:, 1].min()) >= 68
    fresh = fp_arm("FP0", "none", "rows", S)(layer2)
    assert M.bits_equal(reused, fresh)
    g = M.fp_reuse_check("FP0", "2x4", fp_arm("FP0", "none", "rows", S), bf, layer2)
    assert g["passed"] and "precondition met" in g["detail"]
    bad = fp_arm("FP0", "none", "rows", S)
    bad.fresh_buffers = True
    g = M.fp_reuse_check("FP0", "2x4", bad, bf, layer2)
    assert not g["passed"] and "precondition met" in g["detail"]
    assert M.fp_reuse_check("FP0", "2x4", fp_arm("FP0", "none", "rows", S), bf, bf)["status"] == "VOID"
    f8 = fp_arm("FP8", "fixed", "rows", S)
    o8 = f8(bf)
    assert all(w.kw["fixed_split_size"] == M.FP_SPLIT and w.kw["disable_split_kv"] is False for w in f8.wr)
    uc = f8.u["ucnt"]
    assert f8.float_need == max(M.fp_float_need(uc[h].tolist(), "fixed", 2, 16, 4) for h in range(2)) > 0
    assert M.bits_equal(o8, out)                                          # the fake does not split
    with pytest.raises(ValueError):
        f8.relayer(bq)
    fu = fp_arm("FPU", "none", "off", S)
    ou = fu(bf)
    assert all(w._custom_mask_buf is None for w in fu.wr) and not M.bits_equal(ou, out)
    ones = fp_arm("FP0ones", "none", "ones", S)
    g = M.mask_witness("2x4", ones, bf, [0, 3, 5, 7], ref[[0, 3, 5, 7]], M.fp_union(bf, 4, S), M.TOL_C, True)
    assert g["passed"] and g["status"] == "DETECTED"
    g = M.row_alone_probe("FP0", "2x4", arm, bf, [0, 5], 4)
    assert g["passed"] and not g["must"] and "sampled rows differ alone" in g["detail"]
    with pytest.raises(ValueError):
        M.FPArm(FAKE_FI, "X", "heuristic", "rows", GEOM, S, 4, None, "cpu")
    with pytest.raises(ValueError):
        arm(M.select(bf, [0, 1, 2]))                                      # not whole requests


# ------------------------------------------------------------------ FC7 the request suite
class ReqFake:
    """A request-unit fake arm: the fp64 reference per row, cast to bf16, plus a planted dependence."""
    def __init__(self, kind="exact"):
        self.kind = kind

    def __call__(self, b):
        r_ = b["q"].shape[0]
        out = M.reference_fp64(b, list(range(r_)), GEOM)
        if self.kind == "batch":
            out = out + 0.05 * r_
        elif self.kind == "order":
            out = out + 0.05 * torch.arange(r_, dtype=out.dtype).view(-1, 1, 1)
        elif self.kind == "others":
            out = out + b["q"].double()[:, 0, 0].sum()
        elif self.kind == "poison":
            out = out + 0.0 * b["v"].double().sum()
        return out.to(torch.bfloat16)


def _rsuite(case, arm, pre=(True, "ok")):
    bf, bq = M.batch_of(case, "full"), M.batch_of(case, "altreq")
    pois, _ = M.poison_caches(case, bf)
    g, _ = M.request_suite("FP0", "2x4", arm, bf, bq, 4, [0, 1], [("reversed_requests", [1, 0]), ("random", [1, 0])],
                           pois, pre, True)
    return {x["name"].split("_")[0]: x for x in g}


def test_FC7_request_suite_passes_an_invariant_arm_and_catches_each_plant(case):
    """FC7: the exact fake passes FI1, FI2, FI3, FI5 (FI3's precondition: altreq changes exactly the odd requests);
    batch size -> FI1, row position -> FI2, other requests' q -> FI3, a read of the whole store -> FI5; FI5 without its
    precondition is a VOID REPORT (not a MUST)."""
    g = _rsuite(case, ReqFake("exact"))
    assert set(g) == {"FI1", "FI2", "FI3", "FI5"} and all(x["passed"] and x["must"] for x in g.values())
    assert "precondition met" in g["FI3"]["detail"]
    for kind, gid in (("batch", "FI1"), ("order", "FI2"), ("others", "FI3"), ("poison", "FI5")):
        assert not _rsuite(case, ReqFake(kind))[gid]["passed"], kind
    g = _rsuite(case, ReqFake("poison"), pre=(False, "a middle page is not reached"))
    assert not g["FI5"]["must"] and g["FI5"]["status"] == "VOID"


# ------------------------------------------------------------------ FC8 G_block0
def test_FC8_block0_gate_and_its_negative(case):
    """FC8: G_block0 passes on the generator's batches (every row lists block 0); a row of request 0, KV head 1 whose
    block 0 is replaced by an unlisted block fails it, and the union page 0 then has a row with no visible token."""
    S = case["S"]
    bf, bq = M.batch_of(case, "full"), M.batch_of(case, "altreq")
    g = M.block0_gate("2x4", [bf, bq], 4, S, True)
    assert g["passed"] and g["must"]
    b2 = M.select(bf, range(8))
    pk, pt = b2["page_key"].clone(), b2["page_table"].clone()
    free = sorted(set(range(1, M.TAIL_BLOCK - M.N_RECENT)) - set(pk[2, 1].tolist()))[0]
    blk = pk[2, 1].to(torch.int64).clone()
    blk[0] = free
    order = torch.argsort(blk)
    pk[2, 1] = blk[order].to(torch.int32)
    pt[2, 1] = case["slot_of_block"][0][blk][order].to(torch.int32)
    b2["page_key"], b2["page_table"] = pk, pt
    g = M.block0_gate("2x4", [b2], 4, S, True)
    assert not g["passed"] and "without block 0: 1;" in g["detail"]
    assert int(M.fp_union(b2, 4, S)["ext"][1, 0, 2, 0]) == 0


# ------------------------------------------------------------------ FC9 tiles, chunks, workspace
def test_FC9_kv_tile_cta_tile_chunk_hazard_workspace():
    """FC9: the KV tile model (A100: 128 tokens at CTA_TILE_Q 16 and 64, 64 at 128), the CTA tile rule, the FP8
    chunk hazard on a hand case (a row missing both pages of a chunk's first tile), FlashInfer's prefill workspace
    formula by hand."""
    assert M.kv_tile_tokens(64, 128) == 128 and M.KV_TILE_PAGES == 2
    assert M.kv_tile_tokens(16, 128) == 128 and M.kv_tile_tokens(128, 128) == 64
    assert M.cta_tile_prefill(4, 16) == 64 and M.cta_tile_prefill(1, 16) == 16 and M.cta_tile_prefill(8, 16) == 128
    ext = torch.full((1, 1, 2, 16), 64, dtype=torch.int64)
    ext[0, 0, 1, 8:10] = 0
    u = dict(ext=ext, ucnt=torch.tensor([[16]]))
    assert M.chunk_tile_hazard(u) == (1, 1, 2)
    ext[0, 0, 1, 9] = 64
    assert M.chunk_tile_hazard(u) == (0, 0, 2)
    ext[0, 0, 1, 0:2] = 0                                                 # chunk 0 too (FP0 is protected by G_block0)
    assert M.chunk_tile_hazard(u) == (1, 1, 2)
    assert M.fp_float_need([76, 80], "fixed", 16, 128, 4) == 10485760 + 81920
    assert M.fp_float_need([76, 80], "none", 16, 128, 4) == 0
    with pytest.raises(ValueError):
        M.fp_float_need([1], "heuristic", 16, 128, 4)


# ------------------------------------------------------------------ FC10 V1
def test_FC10_v1_store_twin_and_precondition(case):
    """FC10: V1's contiguous union store holds the paged store's rows at the slots topk_idx names (the tail at W - 1,
    W = the largest union), its twin (the varlen layout, the causal cut, the two exp(cis) values and the quotient) is
    the fp64 reference on every row; V1_pre holds on the generator batch and fails on non-consecutive extents and on
    the block-edge request."""
    S = case["S"]
    b = as_f64(M.batch_of(case, "full"))
    u = M.fp_union(b, 4, S)
    ok, txt = M.v1_precondition(u, b, 4)
    assert ok, txt
    a = M.v1_args(b, u, 4, GEOM)
    w = a["W"]
    assert w == int(u["ucnt"].max()) and a["seqused_k"].dtype == torch.int32
    tk = a["topk_idx"]
    assert tk.dtype == torch.int32 and tuple(tk.shape) == (2, 8, 64) and tk.is_contiguous()
    assert int(tk.max()) == w - 1 and int(tk.min()) >= -1
    for h in range(2):
        for r in range(8):
            row = tk[h, r].tolist()
            assert row.count(w - 1) == 1 and len(set(row)) == len(row)
    kf = b["k"].reshape(-1, 2, 16)
    for h, n in ((0, 0), (1, 1)):
        c = int(u["ucnt"][h, n])
        for j in (0, 5, c - 1):
            slot = j if j < c - 1 else w - 1
            page = int(u["upage"][h, n, j])
            assert torch.equal(a["ku"][n, slot * 64:(slot + 1) * 64, h], kf[page * 64:(page + 1) * 64, h])
    vs, vfk = M.v1_prep(a)
    out = M.v1_twin(b["q"], a, vs, vfk, 4)
    assert torch.allclose(out, M.reference_fp64(b, list(range(8)), GEOM), rtol=0, atol=1e-10)
    b2 = dict(b)
    ll = b["last_len"].clone()
    ll[[1, 2]] = ll[[2, 1]]
    b2["last_len"] = ll
    assert not M.v1_precondition(M.fp_union(b2, 4, S), b2, 4)[0]
    c3 = crafted(case)
    ok, txt = M.v1_precondition(M.fp_union(c3, 4, S), c3, 4)
    assert not ok and "1 have a row whose last page" in txt


# ------------------------------------------------------------------ FC11 the variant
def test_FC11_fp_variant_source_and_jit_args():
    """FC11: jit_args_fp is the 13-item list with its own URI; the LogitsTransform is R2FiCisBias's text; the
    LogitsMask is DefaultAttention's formula over our mask and absolute byte offsets (and the readable 0.6.8
    variants.cuh has the same formula); every declared tensor / scalar is read; no name starts with "maybe" (both
    FlashInfer versions then pass the user arguments in order); the braces balance."""
    ja = M.jit_args_fp(128)
    assert len(ja) == 13 and ja[0] == "r2_fia3_cisbias_mask_bf16_d128" != M.jit_args_for(128)[0]
    assert ja[1] is ja[2] is ja[3] is torch.bfloat16 and ja[4] is torch.int32 and ja[5] == ja[6] == 128
    assert ja[7] == ["cis_bias", "r2_mask", "r2_mask_indptr"] and ja[8] == ["nv_bfloat16", "uint8_t", "int32_t"]
    assert ja[9] == list(M.VARIANT_SCALARS[0]) and ja[10] == list(M.VARIANT_SCALARS[1])
    assert ja[11] == M.VARIANT_NAME_FP == "R2FiCisBiasMask" and ja[12] == M.VARIANT_DECL_FP
    src, old = M.VARIANT_DECL_FP, M.VARIANT_DECL

    def transform(s):
        t = s[s.index("REGISTER_LOGITS_TRANSFORM"):]
        return t[:t.index("})") + 2]
    assert transform(src) == transform(old).replace("r2fi_", "r2fm_")
    for needle in ("REGISTER_LOGITS_MASK", "static_cast<uint64_t>(qo_idx) * kv_len + kv_idx",
                   "(r2_mask_ptr[offset / 8] >> (offset % 8)) & 1", "params.r2_mask + params.r2_mask_indptr[batch_idx]",
                   "qo_idx >= qo_len || kv_idx >= kv_len", "struct R2FiCisBiasMask : AttentionVariantBase",
                   "sm_scale_log2 = math::log2e"):
        assert needle in src, needle
    for name in ja[7] + ja[9]:
        assert "params.%s" % name in src, name
    assert not any(n.startswith("maybe") for n in ja[7])
    assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    vp = FI068 / "data/include/flashinfer/attention/variants.cuh"
    if vp.is_file():
        txt = vp.read_text()
        assert "static_cast<uint64_t>(qo_idx) * kv_len + kv_idx" in txt
        assert "(custom_mask_ptr[offset / 8] >> (offset % 8)) & 1" in txt


# ------------------------------------------------------------------ FC12 the API gate
def test_FC12_fp_api_check():
    """FC12: FP_API passes on a wrapper with every parameter the FP arms pass, fails on one without
    packed_custom_mask and on a module without the wrapper; the readable 0.6.8.post1 source has them all (static
    read, no import)."""
    ok, why = M.check_fp_api(FAKE_FI)
    assert ok, why

    class NoMask:
        def __init__(self, ws, layout="NHD", backend="auto", jit_args=None):
            pass

        def plan(self, qo, kv, ix, last, hq, hkv, d, page, sm_scale=None, q_data_type=None, kv_data_type=None,
                 disable_split_kv=False, fixed_split_size=None):
            pass
    ok, why = M.check_fp_api(types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=NoMask))
    assert not ok and "packed_custom_mask" in why
    assert not M.check_fp_api(types.SimpleNamespace())[0]
    pp = FI068 / "prefill.py"
    if pp.is_file():
        txt = pp.read_text()
        cls = txt[txt.index("class BatchPrefillWithPagedKVCacheWrapper"):]
        cls = cls[:cls.index("\nclass ")]
        for p in M.FP_INIT_PARAMS + M.FP_PLAN_PARAMS:
            assert re.search(r"\b%s\s*:" % p, cls), p


# ------------------------------------------------------------------ FC13 completeness
def test_FC13_completeness_with_fp_and_v1():
    """FC13: COMPLETE expects FP0's gates (with CW_FP0 only when the bias variant built) and V1's two gates, per cell
    tag (the s1 suffix included); a missing one fails; without fp / v1 they are not expected."""
    fp_names = ["G_block0", "C_FP0", "MW_FP0", "FI1_request_alone_FP0", "FI2_order_FP0", "FI3_other_requests_FP0",
                "FI5_poison_FP0", "CW_FP0"]
    for tag, cell in (("2x4", (2, 4)), ("2x4s1", (2, 4, "s1"))):
        names = ["%s_%s" % (n, tag) for n in fp_names + ["V1_pre", "C_V1"]]
        gates = [M.gate(n, False, True, "") for n in names]
        out = M.completeness_gates(gates, [cell], ["FP0", "V1"], True, fp=True, v1=True)[0]
        assert out["name"] == "COMPLETE_%s" % tag and out["passed"]
        out = M.completeness_gates(gates[1:], [cell], ["FP0", "V1"], True, fp=True, v1=True)[0]
        assert not out["passed"]
        out = M.completeness_gates([g for g in gates if not g["name"].startswith("CW_FP0")], [cell], ["FP0", "V1"],
                                   False, fp=True, v1=True)[0]
        assert out["passed"]
        assert M.completeness_gates([], [cell], ["FP0", "V1"], True)[0]["passed"]


# ------------------------------------------------------------------ FC14 ncu
def test_FC14_ncu_summary_argv_and_metric_query(tmp_path, monkeypatch):
    """FC14: the ncu summary adds L2 -> SM bytes (sectors x 32), registers, local-memory bytes (a spill) and PRED3's
    bands; V1's InfLLM-v2 kernels match the filter; the child argv carries --section LaunchStats, --verify-step only
    with V1 and --s1-tables only for an s1 cell; the metric query keeps a ".avg.pct_of_peak..." metric that the chip
    lists and drops one it does not (the query's output replaced: run_logged writes a fixed listing)."""
    nv = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    extra = ["lts__t_sectors_srcunit_tex_op_read.sum", "launch__registers_per_thread",
             "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"]
    cols = ["ID", "Process ID", nv, "Kernel Name"] + list(M.NCU_REQUIRED) + extra
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(cols)
    i = 0
    for tag, nm, rd, sec, reg, st in (("P", "_bi_split_kernel", 80, 15, 128, 0), ("P", "_bi_combine_kernel", 20, 5, 40, 0),
                                      ("FP0", "BatchPrefillWithPagedKVCacheKernel", 25, 1, 96, 0),
                                      ("FP0", "BatchPrefillWithPagedKVCacheKernel", 25, 3, 96, 8),
                                      ("V1", "flash_fwd_kernel", 30, 4, 255, 0), ("V1", "kernel_topk_to_uint64", 1, 1, 32, 0)):
        w.writerow([str(i), "9", '9 "<default domain>:fia|%s:none"' % tag, "void flashinfer::%s<...>(...)" % nm,
                    "1000", str(rd), "10", str(sec), str(reg), "0", str(st)])
        i += 1
    rows = M.ncu_rows(buf.getvalue(), list(M.NCU_REQUIRED) + list(M.NCU_OPTIONAL) + list(M.NCU_LAUNCH))
    s = M.ncu_summary(rows, ["P", "FP0", "V1"], ideal=400.0, union=100.0, rpr=4)
    p, f, v = s["arms"]["P"], s["arms"]["FP0"], s["arms"]["V1"]
    assert p["l2sm_bytes"] == 20 * 32 and f["l2sm_bytes"] == 4 * 32 and f["l2sm_over_union"] == 1.28
    assert p["regs"] == [128, 40] and f["local_bytes"] == 8.0 and p["local_bytes"] == 0.0
    assert "DRAM/union" in f["band3"] and "L2->SM/sum" in p["band3"] and v["n_kernels"] == 2 and not s["stray"]
    assert "(SPILLS)" in M.ncu_arm_text("FP0", f) and "registers [96]" in M.ncu_arm_text("FP0", f)

    class A:
        ncu, config, biattn, sharedkv, seed = "/x/ncu", "/c.json", "/p3.py", None, 5
        verify_step, s1_tables = "/r/eq_verify_step.py", "/r/skv_tables.pt"
    argv = M.ncu_argv(A, 216, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "FP0", "V1"], False, src="s1")
    assert argv[argv.index("--section") + 1] == "LaunchStats" and argv[argv.index("--cells") + 1] == "216x4s1"
    assert argv[argv.index("--verify-step") + 1] == A.verify_step and argv[argv.index("--s1-tables") + 1] == A.s1_tables
    argv = M.ncu_argv(A, 216, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "FP0"], False)
    assert "--verify-step" not in argv and "--s1-tables" not in argv and "216x4" in argv
    rx = re.compile(M.KERNEL_FILTER[len("regex:"):])
    for nm in ("flash_fwd_kernel", "flash_fwd_splitkv_kernel", "kernel_topk_to_uint64"):
        assert rx.search(nm)
    seen = []

    def fake_run(argv, path, timeout):
        seen.append(list(argv))
        Path(path).write_text("  gpu__time_duration   Counter  ns\n  dram__bytes_read   Counter  byte\n"
                              "  dram__bytes_write   Counter  byte\n  lts__t_sectors   Counter  sector\n"
                              "  lts__t_sectors_srcunit_tex_op_read   Counter  sector\n")
        return 0
    monkeypatch.setattr(M, "run_logged", fake_run)

    class B:
        ncu = "/x/ncu"
    rc, missing, metrics = M.ncu_metrics(B, str(tmp_path))
    assert seen == [["/x/ncu", "--query-metrics", "--chip", "ga100"]]
    assert rc == 0 and not missing
    assert "lts__t_sectors.avg.pct_of_peak_sustained_elapsed" in metrics
    assert "lts__t_sectors_srcunit_tex_op_read.sum" in metrics
    assert "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum" not in metrics


# ------------------------------------------------------------------ FC15 the registered block
def test_FC15_pred3_is_registered_in_the_notes():
    """FC15: docs/NOTES_fia3.md carries the TBD stamp marker and a json block equal to PRED3; every band is [central,
    low, high] with low <= central <= high; the FP0 / P band is the plan's (0.75 in [0.45, 1.3]); the thresholds are
    the plan's rule (0.8, 1 ms, 32 layers, 3x)."""
    doc = (ROOT / "docs/NOTES_fia3.md").read_text()
    assert "REGISTERED <TBD-for-lead-stamp>" in doc
    m = re.search(r"<!-- PRED3-BEGIN -->\s*```json\s*(.*?)```\s*<!-- PRED3-END -->", doc, flags=re.S)
    assert m, "no PRED3 block in the notes"
    assert json.loads(m.group(1)) == json.loads(json.dumps(M.PRED3))
    assert M.PRED3["fp0_over_p_device"]["216x4"] == [0.75, 0.45, 1.3]
    assert M.PRED3["thresholds"] == dict(fp0_over_p_max=0.8, tick_overhead_ms_max=1.0, layers_per_tick=32,
                                         hbm_bound_l2sm_drop=3.0)
    for x in list(M.PRED3["fp0_over_p_device"].values()) + [M.PRED3[k] for k in (
            "fp0_dram_read_over_union", "fp0_l2sm_read_over_union", "p_l2sm_read_over_sum")]:
        assert len(x) == 3 and x[1] <= x[0] <= x[2]
    for word in ("0.8 x P", "1 ms per tick", "15 % byte bound", "rung-4"):
        assert word in M.PRED3["decision"]
    assert M.LAYERS_PER_TICK == M.PRED3["thresholds"]["layers_per_tick"]


# ------------------------------------------------------------------ FC16 the decision report
def test_FC16_decision_report_branches():
    """FC16: R_DECISION says ADOPT only with FP0 / P <= 0.8, a tick overhead <= 1 ms (eager or graph) and all FP0 MUST
    gates of the cell passing; it flags HBM-bound (L2 -> SM fell >= 3x, time did not), plan / build bound, and the
    mask share; another cell's failing gate does not count; an unregistered cell gets no line."""
    def rep(tp, tf, tfu, l2p, l2f, ove, ovg, tag="216x4"):
        return dict(ncu={tag: dict(summary=dict(arms=dict(P=dict(time_ns=tp, l2sm_bytes=l2p),
                                                         FP0=dict(time_ns=tf, l2sm_bytes=l2f),
                                                         FPU=dict(time_ns=tfu))))},
                    cells={tag: dict(arms=dict(FP0=dict(tick_overhead_eager_ms=ove, tick_overhead_graph_ms=ovg)))})
    ok = [M.gate("C_FP0_216x4", True, True, ""), M.gate("G_block0_216x4", True, True, ""),
          M.gate("C_FP0_216x4s1", False, True, "")]
    g = M.decision_report(rep(1000, 700, 600, 3e9, 1e9, 2.0, 0.4), ok, [(216, 4)])[0]
    assert g["name"] == "R_DECISION_216x4" and g["detail"].startswith("ADOPT") and not g["must"]
    assert "mask share of FP0's device time 0.143" in g["detail"]
    g = M.decision_report(rep(1000, 950, 900, 3.4e9, 1e9, 0.5, 0.3), ok, [(216, 4)])[0]
    assert g["detail"].startswith("NOT MET") and "HBM-bound" in g["detail"]
    g = M.decision_report(rep(1000, 700, 600, 3e9, 2e9, 5.0, 2.0), ok, [(216, 4)])[0]
    assert g["detail"].startswith("NOT MET") and "plan / build bound" in g["detail"]
    g = M.decision_report(rep(1000, 700, 600, 3e9, 1e9, 0.5, 0.3), ok[:1] + [M.gate("G_block0_216x4", False, True, "")],
                          [(216, 4)])[0]
    assert g["detail"].startswith("NOT MET") and "NOT ALL PASS" in g["detail"]
    assert M.decision_report(rep(1000, 700, 600, 3e9, 1e9, 0.5, 0.3), ok, [(32, 4)]) == []


# ------------------------------------------------------------------ FC17 the S1 tables
def _write_tables(case, path):
    """Two recorded calls: call 0 has requests seq 3 (rows 0 .. 3, in order), seq 9 (rows 4 .. 7 stored in REVERSED
    position order) and seq 5 (two rows: not a 4-row request); call 1 has one 4-row request with an extent of 0 (not
    valid: skipped)."""
    f = case["full"]
    idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 1]
    t0 = {k: f[k][idx].clone() for k in ("page_table", "num_pages", "last_len", "page_key")}
    t0["seq_idx"] = torch.tensor([3, 3, 3, 3, 9, 9, 9, 9, 5, 5], dtype=torch.int32)
    t1 = {k: f[k][:4].clone() for k in ("page_table", "num_pages", "last_len", "page_key")}
    t1["seq_idx"] = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    t1["last_len"][2] = 0
    tables = [dict(call=0, rows_max=4, mode="split", tiles=None, **t0), dict(call=7, rows_max=4, mode="split", tiles=None,
                                                                             **t1)]
    torch.save(tables, str(path))


def test_FC17_s1_tables_groups_and_case(case, tmp_path):
    """FC17: load_s1_tables reads a saved list as weights only; s1_groups keeps the requests with exactly 4 valid
    rows, in POSITION order; make_case_s1 cycles them over B requests on their own entries (lists and extents as
    recorded), alt / altreq change odd rows / odd requests, S >= 256, the hazard rows exist; with_data=False keeps
    the lists; G_block0 holds; the FP twin on the s1 batch is the reference."""
    p = tmp_path / "skv_tables.pt"
    _write_tables(case, p)
    tables = M.load_s1_tables(str(p))
    groups = M.s1_groups(tables, 4, 2)
    assert [(gr["call"], gr["seq"]) for gr in groups] == [(0, 3), (0, 9)]
    assert groups[1]["rows"] == [7, 6, 5, 4] and groups[1]["last_len"].tolist() == case["full"]["last_len"][4:8].tolist()
    assert M.s1_groups(tables, 2, 2)[0]["seq"] == 5
    with pytest.raises(ValueError):
        M.s1_groups(tables, 4, 4)
    c = M.make_case_s1(GEOM, 3, 4, 7, groups)
    f = c["full"]
    assert c["S"] == 256 and c["N"] == 4 and c["src"] == "s1" and c["s1_groups"] == 2
    assert torch.equal(f["page_key"][8:12], groups[0]["page_key"].to(torch.int32))
    assert torch.equal(f["page_table"][4:8], groups[1]["page_table"].to(torch.int32))
    assert f["seq_idx"].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert M.changed_rows(M.batch_of(c, "full"), M.batch_of(c, "altreq")) == [4, 5, 6, 7]
    assert M.changed_rows(M.batch_of(c, "full"), M.batch_of(c, "alt")) == [1, 3, 5, 7, 9, 11]
    assert c["short"]["num_pages"][:, 0].tolist() == list(M.SHORT_COUNTS)
    lists = M.make_case_s1(GEOM, 3, 4, 7, groups, with_data=False)
    assert torch.equal(lists["full"]["page_table"], f["page_table"]) and lists["k"] is None
    assert M.block0_gate("3x4s1", [M.batch_of(c, "full")], 4, c["S"], True)["passed"]
    b = as_f64(M.batch_of(c, "full"))
    out = twin_all(b, 4, c["S"], fp_inputs(b, 4, c["S"]))
    assert torch.allclose(out, M.reference_fp64(b, list(range(12)), GEOM), rtol=0, atol=1e-10)
    with pytest.raises(ValueError):
        M.make_case_s1(GEOM, 3, 4, 7, [])


def test_FC18_the_recorded_s1_tables(capsys):
    """FC18 (facts for the run; prints them): the recorded S1 file loads as weights only; it has 4-row requests with
    64 pages per (row, head) on 2 KV heads; for a 32-request s1 case the test prints G_block0, the union sizes, the
    poison precondition, V1_pre and the FP8 chunk hazard. Asserted: the load, the shapes, the counts are computed."""
    if not S1_RECORDED.is_file():
        pytest.skip("no recorded tables at %s" % S1_RECORDED)
    tables = M.load_s1_tables(str(S1_RECORDED))
    groups = M.s1_groups(tables, 4, 2)
    assert groups, "no 4-row request in the recorded calls"
    assert all(tuple(gr["page_table"].shape) == (4, 2, 64) for gr in groups)
    c = M.make_case_s1(GEOM, 32, 4, 20261021, groups, with_data=False)
    bf = c["full"]
    g = M.block0_gate("32x4s1", [bf, c["altreq"]], 4, c["S"], True)
    u = M.fp_union(bf, 4, c["S"])
    pre_p = M.poison_precondition(u)
    pre_v = M.v1_precondition(u, bf, 4)
    hz = M.chunk_tile_hazard(u)
    print("S1 recorded: %d calls, %d four-row requests (calls %s), S %d" % (len(tables), len(groups),
                                                                           sorted({gr["call"] for gr in groups}), c["S"]))
    print("G_block0 %s: %s" % ("PASS" if g["passed"] else "FAIL", g["detail"]))
    print("union %s; poison precondition %s; V1_pre %s; FP8 chunk hazard cells / rows / chunks %s"
          % (M.union_stats(u, bf), pre_p, pre_v, hz))
    assert isinstance(g["passed"], bool) and isinstance(pre_v[0], bool)


# ------------------------------------------------------------------ FC19 the step
def test_FC19_step_static_shape_and_pins():
    """FC19: one main(); the claim fiattn_active after the existence check, with an owner file; the budget fits the
    step; the pins are piece 3's and EQ verify_step's blobs (checked against git when the objects exist) and the S1
    tables' size (checked against the file); the bench gets the fia3 cells, arms, --verify-step, --s1-tables and
    --no-strided; no interpreter word."""
    src = (EFF / "r2_fia3_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    assert src.index('[ ! -e "$OUT" ]') < src.index('mkdir "$OUTROOT/fiattn_active"') < src.index("fiattn_active/owner")
    step_min = int(re.search(r"^STEP_MIN=(\d+)$", src, flags=re.M).group(1))
    budget = int(re.search(r"^BUDGET_S=(\d+)$", src, flags=re.M).group(1))
    assert step_min <= 45 and budget + 300 <= step_min * 60
    assert "BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042" in src
    sha, path, blob = VSTEP_PIN
    assert "VSTEP_SHA=%s" % sha in src and "VSTEP_PATH=%s" % path in src and "VSTEP_BLOB=%s" % blob in src
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (sha, path)], capture_output=True, text=True)
    if r.returncode == 0:
        assert r.stdout.strip() == blob
    m = re.search(r"^S1_BYTES=(\d+)", src, flags=re.M)
    assert m and "S1_TABLES=%s" % S1_RECORDED in src
    if S1_RECORDED.is_file():
        assert S1_RECORDED.stat().st_size == int(m.group(1))
    assert re.search(r"^CELLS=32x4,216x4,32x4s1,216x4s1(\s|$)", src, flags=re.M)
    assert re.search(r"^ARMS=P,F8,FP0,FP8,FPU,V1(\s|$)", src, flags=re.M)
    for wd in ('--verify-step "$VSTEP_MOD"', '--s1-tables "$S1_COPY"', "--no-strided", '--arms "$ARMS"',
               "NOSI=%s" % NOSI3):
        assert wd in src, wd
    for word in ("awk", "perl", "python3 -", "sed -i", "<<"):
        assert word not in src, word
    for c in M.parse_cells("32x4,216x4,32x4s1,216x4s1"):
        assert M.cell_tag(*c) in src


def test_FC20_step_dryrun_refusals_and_pins(tmp_path):
    """FC20 (as FA12): usage (2); a missing uuid file, a GPU- prefix, two lines STOP (1); FIATTN_HOLDERS outside
    DRYRUN refused; a wrong --nosi stops at HEAD. On a clean checkout of THIS worktree at HEAD the DRYRUN prints the
    pins (piece 3, V1, the S1 tables) and the bench arguments."""
    script = EFF / "r2_fia3_step.sh"
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
    r = run(420, extra=["x"])
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(421)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(422, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(423, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(424, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(425, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(NOSI3).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % NOSI3)
    r = run(426, uid + "\n", sha=head.stdout.strip())
    assert r.returncode == 0, r.stderr[-400:]
    sha, path, blob = VSTEP_PIN
    assert "V1 %s:%s blob %s" % (sha, path, blob) in r.stdout
    assert "S1 tables %s (439458 B)" % S1_RECORDED in r.stdout
    for wd in ("--mode main", "--cells 32x4\\,216x4\\,32x4s1\\,216x4s1", "--arms P\\,F8\\,FP0\\,FP8\\,FPU\\,V1",
               "--no-strided", "--ncu /opt/nvidia/nsight-compute/2025.1.0/ncu", "--budget-s 2400", "--time=45",
               "FLASHINFER_WORKSPACE_BASE=/mnt/beegfs/ojerbi/nosa-env/jit/fia3_t1_426/flashinfer",
               "--verify-step /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_t1_426.step/eq_verify_step.py",
               "--s1-tables /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_t1_426.step/skv_tables.pt"):
        assert wd in r.stdout, wd


# ------------------------------------------------------------------ FC21 no host read in the build
def test_FC21_the_build_has_no_host_read():
    """FC21: fp_union, fp_csr and fp_pack (the per-layer build, captured in a CUDA graph by time_fp_extras) contain no
    host read of a tensor (.item / .cpu / .tolist / int / float / bool of a tensor / .numpy)."""
    for fn in (M.fp_union, M.fp_csr, M.fp_pack):
        src = inspect.getsource(fn)
        body = src.split('"""', 2)[-1]
        for bad in (".item(", ".cpu(", ".tolist(", "int(", "float(", "bool(", ".numpy("):
            assert bad not in body, (fn.__name__, bad)
