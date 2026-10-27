"""CPU tests of the fia4 fork and its bench arms: nosi/nosi/fork_vllm_unified_attention.py (a fork of vLLM 0.20.2's
Triton unified attention), the [fia4] parts of benchmarks/Efficiency/r2_fiattn_bench.py (arms FV / FVT),
benchmarks/Efficiency/r2_fia4_step.sh and docs/NOTES_fia4.md. Everything that needs no GPU. The fork's Triton kernel
runs under TRITON_INTERPRET=1 in a FRESH child process (Triton reads the flag at its first import), on CPU float32
tensors, against the bench's float64 reference; skipped when triton is not installed. L36: the interpreter is not the
GPU compiler (it does not check registers, layouts or the GPU's memory order). SOURCE ONLY; the lead runs them on a
compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia4.py

FV1 license header and revision; FV2 every changed line carries [fork] (difflib against the pristine upstream copies)
with two planted negative controls; FV3 one fork, no other kernel; FV4 the torch tables = a set model, the width, the
sort precondition (and the recorded S1 lists, when present); FV5 the kernel under the interpreter (13 checks, a planted
mask error among them); FV6 completeness and the PRED4 block; FV7 the decision report; FV8 ncu summary, argv and the
arguments; FV9 / FV10 the step (static shape, DRYRUN).
"""
from __future__ import annotations

import ast
import csv
import difflib
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
FORK = ROOT / "nosi/nosi/fork_vllm_unified_attention.py"
FIX = HERE / "fixtures_vllm_0_20_2"
UP_UNIFIED = FIX / "triton_unified_attention.py.upstream"
UP_HELPERS = FIX / "triton_attention_helpers.py.upstream"
VENV_OPS = Path("/mnt/beegfs/ojerbi/venvs/vllm-port/lib/python3.10/site-packages/vllm/v1/attention/ops")
NOSI4 = "/mnt/beegfs/ojerbi/nosi-r2-fia4"
S1_RECORDED = Path("/mnt/beegfs/ojerbi/r2-runs/sharedkv/skv_probe_s1a_2191262/skv_tables.pt")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load(EFF / "r2_fiattn_bench.py", "r2_fiattn_bench_fia4_under_test")
GEOM4 = dict(Hq=8, Hkv=2, D=16, G=4)                    # BLOCK_M = 4 rows x 4 query heads = 16 (tl.dot's minimum)
GEOM_REAL = dict(Hq=32, Hkv=2, D=128, G=16)             # NOSA-8B: BLOCK_M = 4 x 16 = 64


# ------------------------------------------------------------------ helpers (also used by the interpreter child)
def reference_lines():
    """The upstream text the fork starts from (docs/NOTES_fia4.md section 3): the unified file's lines 1-31 (header,
    imports), the helpers file from line 14 (its tl / triton import, then the helpers), the unified file from line
    32 on."""
    u = UP_UNIFIED.read_text().splitlines(keepends=True)
    h = UP_HELPERS.read_text().splitlines(keepends=True)
    return u[:31] + h[13:] + u[31:]


def untagged_changes(ref, fork):
    """(fork lines inside a replace / insert hunk that are not blank and carry no "[fork]"; delete hunks with no
    "[fork]" line within 3 fork lines of the deletion point). difflib without autojunk (blank and bracket lines are
    frequent, so the junk heuristic would misalign them)."""
    bad_lines, bad_deletes = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, fork, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            bad_lines += [(j + 1, fork[j].rstrip()) for j in range(j1, j2)
                          if fork[j].strip() and "[fork]" not in fork[j]]
        elif op == "delete" and not any("[fork]" in x for x in fork[max(0, j1 - 3):j1 + 3]):
            bad_deletes.append((i1 + 1, i2, j1 + 1))
    return bad_lines, bad_deletes


def groups_of(nb, rpr):
    return [[n * rpr + u for u in range(rpr)] for n in range(nb)]


def set_model(b, groups, s_slots):
    """Per (KV head h, request n = group index): the sorted distinct (block id, slot) pairs of the group's listed
    pages, as (page id = entry x S + slot, packed bytes: 64 / last_len for the row's last listed position / 0, row u
    at bits 8u ..). A pure-Python model (no torch op)."""
    pk, pt = b["page_key"].tolist(), b["page_table"].tolist()
    npg, ll, si = b["num_pages"].tolist(), b["last_len"].tolist(), b["seq_idx"].tolist()
    out = {}
    for h in range(len(pk[0])):
        for n, rows in enumerate(groups):
            vals = {}
            for u, r in enumerate(rows):
                c = npg[r][h]
                for i in range(c):
                    vals.setdefault((pk[r][h][i], pt[r][h][i]), [0] * 4)[u] = ll[r] if i == c - 1 else 64
            out[(h, n)] = [(si[rows[0]] * s_slots + s, sum(vals[(k, s)][u] << (8 * u) for u in range(4)))
                           for k, s in sorted(vals)]
    return out


def tables_match(model, table, ext, ulen):
    """The tables (Hkv, B, W) / lengths (Hkv, B) against set_model, on each union's valid prefix."""
    for (h, n), want in model.items():
        c = len(want)
        if int(ulen[h, n]) != c:
            return False, "length (%d, %d): %d != %d" % (h, n, int(ulen[h, n]), c)
        if table[h, n, :c].tolist() != [p for p, _ in want] or ext[h, n, :c].tolist() != [w for _, w in want]:
            return False, "entries of (%d, %d) differ" % (h, n)
    return True, "%d (head, request) tables equal" % len(model)


def crafted4(case):
    """test_r2_fia3.crafted on a 2 x 4 case: request 0 at extents 10 .. 13; request 1 CROSSES a block edge (rows 4, 5
    end in block 250 with 63 / 64 tokens, the tail 251 dropped from their lists; rows 6, 7 start block 251)."""
    b = M.select(M.batch_of(case, "full"), range(8))
    b["last_len"] = torch.tensor([10, 11, 12, 13, 63, 64, 1, 2], dtype=torch.int32)
    npg = b["num_pages"].clone()
    npg[4:6] = 63
    b["num_pages"] = npg
    return b


def f32(b):
    """The batch with float32 q / k / v / bias (the interpreter's dtype; the lists unchanged)."""
    out = dict(b)
    for f in ("q", "k", "v", "bias"):
        out[f] = b[f].float().contiguous()
    return out


# ------------------------------------------------------------------ FV1 header and revision
def test_FV1_license_header_and_revision():
    """FV1: the fork keeps vLLM's SPDX / copyright / authors header (its first 9 lines are upstream's), names the
    revision and both upstream files, imports no vllm module; the fixtures are the venv's files byte for byte when the
    venv is readable."""
    fork = FORK.read_text().splitlines()
    up = UP_UNIFIED.read_text().splitlines()
    assert fork[:9] == up[:9] and up[0] == "# SPDX-License-Identifier: Apache-2.0"
    txt = FORK.read_text()
    for word in ("vLLM 0.20.2", "vllm/v1/attention/ops/triton_unified_attention.py",
                 "vllm/v1/attention/ops/triton_attention_helpers.py", "fia3_f1a_2195573", "rung 4"):
        assert word in txt, word
    code = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("#")]
    assert not any(re.match(r"\s*(import vllm|from vllm)", ln) for ln in code)
    for name, fx in (("triton_unified_attention.py", UP_UNIFIED), ("triton_attention_helpers.py", UP_HELPERS)):
        if (VENV_OPS / name).is_file():
            assert (VENV_OPS / name).read_bytes() == fx.read_bytes(), name


# ------------------------------------------------------------------ FV2 every changed line is tagged
def test_FV2_every_changed_line_is_tagged_and_two_plants_are_caught():
    """FV2: against the composed upstream text, every fork line in a changed hunk carries "[fork]" (blank lines
    excepted) and every pure deletion has a "[fork]" marker within 3 lines. Negative controls (precondition first,
    L34): an upstream line of softmax_step changed without a tag, and the same line deleted without a marker, are
    both caught."""
    ref = reference_lines()
    fork = FORK.read_text().splitlines(keepends=True)
    bad_lines, bad_deletes = untagged_changes(ref, fork)
    assert not bad_lines, bad_lines[:10]
    assert not bad_deletes, bad_deletes[:10]
    target = "    l_j = tl.sum(P, axis=1)\n"
    assert ref.count(target) == 1 and fork.count(target) == 1, "precondition: one upstream line to plant on"
    j = fork.index(target)
    assert not any("[fork]" in x for x in fork[j - 4:j + 5]), "precondition: no tag near the plant"
    changed = fork[:j] + ["    l_j = tl.sum(P, axis=1) * 1.0\n"] + fork[j + 1:]
    bl, _ = untagged_changes(ref, changed)
    assert any("* 1.0" in x for _, x in bl)
    deleted = fork[:j] + fork[j + 1:]
    _, bd = untagged_changes(ref, deleted)
    assert bd


# ------------------------------------------------------------------ FV3 one fork, no other kernel
def test_FV3_one_fork_and_no_other_kernel():
    """FV3: the fork's @triton.jit functions are upstream's (kernel, reduce, the copied helpers) plus two helpers
    called only inside the kernel (fork_build_union from kernel_unified_attention, fork_row_lower_bound from
    fork_build_union); only upstream's two kernels are launched; the bench defines and launches no kernel."""
    tree = ast.parse(FORK.read_text())
    jit = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
           and any(ast.unparse(d).startswith("triton.jit") for d in n.decorator_list)}
    assert set(jit) == {"cdiv_fn", "apply_softcap", "resolve_seq_and_query_len", "find_seq_idx", "init_softmax_M",
                        "store_segm_reduce_scalars", "softmax_step", "_cast_kv_tile", "fork_build_union",
                        "fork_row_lower_bound", "kernel_unified_attention", "reduce_segments"}
    launched = {n.func.value.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)}
    assert launched == {"kernel_unified_attention", "reduce_segments"}
    calls = {name: {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
             for name, fn in jit.items()}
    assert [n for n, c in calls.items() if "fork_build_union" in c] == ["kernel_unified_attention"]
    assert [n for n, c in calls.items() if "fork_row_lower_bound" in c] == ["fork_build_union"]
    bench = (EFF / "r2_fiattn_bench.py").read_text()
    assert "@triton.jit" not in bench and "import triton" not in bench


# ------------------------------------------------------------------ FV4 the torch tables
def test_FV4_torch_tables_equal_a_set_model_width_and_sort_precondition():
    """FV4: fv_tables_from_union(fp_union) = the set model on the generator batch, the altreq batch and the
    block-edge batch; the width rule; fv_sorted_rows passes on the generator lists and fails on a swapped list; the
    recorded S1 lists (when the file exists) are sorted: the in-kernel build's precondition for the s1 cells."""
    case = M.make_case(GEOM4, 2, 4, 7)
    S = case["S"]
    for b in (M.batch_of(case, "full"), M.batch_of(case, "altreq"), crafted4(case)):
        t, x, n = M.fv_tables_from_union(M.fp_union(b, 4, S), M.fv_width(64))
        assert t.dtype == x.dtype == n.dtype == torch.int32 and tuple(t.shape) == (2, 2, 256)
        ok, why = tables_match(set_model(b, groups_of(2, 4), S), t, x, n)
        assert ok, why
        assert M.fv_sorted_rows(b)[0]
    assert M.fv_width(64) == 256 and M.fv_width(63) == 256 and M.fv_width(8) == 32 and M.fv_width(65) == 512
    b = M.batch_of(case, "full")
    pk = b["page_key"].clone()
    pk[0, 0, [1, 2]] = pk[0, 0, [2, 1]]
    b["page_key"] = pk
    ok, why = M.fv_sorted_rows(b)
    assert not ok and why.startswith("1 ")
    with pytest.raises(ValueError):
        M.fv_tables_from_union(M.fp_union(M.batch_of(case, "full"), 4, S), 128)
    if S1_RECORDED.is_file():
        groups = M.s1_groups(M.load_s1_tables(str(S1_RECORDED)), 4, 2)
        c = M.make_case_s1(GEOM4, 32, 4, 20261021, groups, with_data=False)
        for which in ("full", "altreq"):
            ok, why = M.fv_sorted_rows(c[which])
            print("S1 recorded lists (%s): %s" % (which, why))
            assert ok, why


# ------------------------------------------------------------------ FV5 the kernel under the interpreter
INTERP_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton imported before the flag was read"
sys.dont_write_bytecode = True
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load(sys.argv[3], "tests_fia4_interp")          # the helpers of this test file (its bench copy is T.M)
M = T.M
F, why = M.import_fv(sys.argv[2])
assert F is not None, why
TOL = 1e-4
fails, ran = [], []


def check(name, ok, detail=""):
    ran.append(name)
    if not ok:
        fails.append((name, str(detail)[:300]))


def err(out, b, rows, geom):
    ref = M.reference_fp64(b, rows, geom)
    return float((out[torch.as_tensor(rows)].double() - ref).abs().max())


def views(b, s_slots):
    n_, sr, hkv, d = b["k"].shape
    nblk = n_ * s_slots
    return (b["k"].view(nblk, 64, hkv, d), b["v"].view(nblk, 64, hkv, d), b["bias"].view(nblk, 64, hkv), sr)


def build_args(b, s_slots):
    return dict(page_key=b["page_key"], page_table=b["page_table"], num_pages=b["num_pages"],
                last_len=b["last_len"], seq_idx=b["seq_idx"], slots_per_entry=s_slots)


G4 = T.GEOM4
case = M.make_case(G4, 2, 4, 11)
S = case["S"]
bf = T.f32(M.batch_of(case, "full"))
baq = T.f32(M.batch_of(case, "altreq"))
rows = list(range(bf["q"].shape[0]))
ref = M.reference_fp64(bf, rows, G4)
fv = M.FVArm(F, "FV", G4, S, 4, build=True, device="cpu")
ft = M.FVArm(F, "FVT", G4, S, 4, build=False, device="cpu")
o_fv = fv(bf)
o_ft = ft(bf)
check("I1_fv_vs_fp64", err(o_fv, bf, rows, G4) <= TOL, err(o_fv, bf, rows, G4))
check("I1_fvt_vs_fp64", err(o_ft, bf, rows, G4) <= TOL, err(o_ft, bf, rows, G4))
check("I3_fvt_equals_fv", M.bits_equal(o_fv, o_ft))
g = M.fv_table_gate("2x4", fv, [("full", bf), ("altreq", baq)], 4, S, True)
check("I2_kernel_tables_equal_torch_tables", g["passed"], g["detail"])
fv(bf)
check("I2_kernel_tables_equal_the_set_model", *T.tables_match(T.set_model(bf, T.groups_of(2, 4), S), *fv.tables()))

# I4 the NEGATIVE CONTROL: a planted mask error (row u of request n loses 8 full pages of KV head h)
ft.prepare(bf)
base = ft.launch().clone()
h, n, u = 1, 1, 2
w = ft.bufs["ext"][h, n].clone()
cand = [j for j in range(1, int(ft.bufs["ulen"][h, n])) if ((int(w[j]) >> (8 * u)) & 0xFF) == 64][:8]
check("I4_plant_precondition", len(cand) == 8, cand)
for j in cand:
    ft.bufs["ext"][h, n, j] = int(w[j]) & ~(0xFF << (8 * u))
o = ft.launch().clone()
r = n * 4 + u
e_r = err(o, bf, [r], G4)
check("I4_plant_detected", e_r > M.TOL_C, "%.3e" % e_r)
check("I4_plant_local", all(M.bits_equal(o[x], base[x]) for x in rows if x != r))
check("I4_plant_kv_head_local", M.bits_equal(o[r, :G4["G"]], base[r, :G4["G"]]))

ones = M.FVArm(F, "FVones", G4, S, 4, build=False, ones=True, device="cpu")
gw = M.mask_witness("2x4", ones, bf, rows, ref, M.fp_union(bf, 4, S), M.TOL_C, True, gname="MW_FV_2x4")
check("I5_mask_witness", gw["passed"] and gw["name"] == "MW_FV_2x4", gw["detail"])
ref0 = M.reference_fp64(bf, rows, G4, zero_bias=True)
gb = M.bias_witness("2x4", fv, bf, rows, ref, ref0, M.TOL_C, True, gname="CW_FV_2x4")
check("I6_bias_witness", gb["passed"], gb["detail"])

pois, _ = M.poison_caches(case, bf)
gs, full = M.request_suite("FV", "2x4", fv, bf, baq, 4, [0, 1], [("reversed_requests", [1, 0]), ("random", [1, 0])],
                           pois, (True, "loads stop at the tile reach"), True)
check("I7_request_suite", len(gs) == 4 and all(x["passed"] for x in gs), [x["detail"] for x in gs if not x["passed"]])
check("I7_full_is_the_first_output", M.bits_equal(full, o_fv))
gx = M.fv_row_alone_gate("2x4", fv, bf, rows, 4, o_fv)
check("I8_row_alone_exact", gx["passed"], gx["detail"])

c3 = T.f32(T.crafted4(case))
o3 = fv(c3)
check("I9_block_edge_vs_fp64", err(o3, c3, rows, G4) <= TOL, err(o3, c3, rows, G4))
g3 = M.fv_table_gate("crafted", fv, [("crafted", c3)], 4, S, True)
check("I9_block_edge_tables", g3["passed"], g3["detail"])

# I10 ragged requests (1, 3 and 4 rows) through the launcher (the in-kernel build; seq_idx 0, 1, 1, 1, 2, 2, 2, 2)
cr = M.make_case(G4, 3, 4, 13)
br = M.select(T.f32(M.batch_of(cr, "full")), [0, 4, 5, 6, 8, 9, 10, 11])
kv, vv, bv, sr = views(br, cr["S"])
tab = torch.zeros((2, 3, M.fv_width(64)), dtype=torch.int32)
ext = torch.zeros_like(tab)
ul = torch.zeros((2, 3), dtype=torch.int32)
o10 = torch.empty_like(br["q"])
F.unified_attention(br["q"], kv, vv, o10, torch.tensor([0, 1, 4, 8], dtype=torch.int32), 4,
                    torch.zeros(3, dtype=torch.int32), sr, G4["D"] ** -0.5, True, (-1, -1), tab, 0, None, None, None,
                    union_ext=ext, union_len=ul, cis_bias=bv, build_union=build_args(br, cr["S"]))
check("I10_ragged_vs_fp64", err(o10, br, list(range(8)), G4) <= TOL, err(o10, br, list(range(8)), G4))
check("I10_ragged_tables", *T.tables_match(T.set_model(br, [[0], [1, 2, 3], [4, 5, 6, 7]], cr["S"]), tab, ext, ul))

# I11 the 3D (split) path behind its flag: 2 segments, both the in-kernel and the torch table
fv.prepare(bf)
kv, vv, bv, sr = views(bf, S)
for name, arm, bu in (("FV", fv, build_args(bf, S)), ("FVT", ft, None)):
    arm.prepare(bf)
    so = torch.zeros((8, G4["Hq"], 2, G4["D"]), dtype=torch.float32)
    sm = torch.zeros((8, G4["Hq"], 2), dtype=torch.float32)
    se = torch.zeros((8, G4["Hq"], 2), dtype=torch.float32)
    o11 = torch.empty_like(bf["q"])
    F.unified_attention(bf["q"], kv, vv, o11, arm.bufs["cu"], 4, arm.bufs["seqused"], sr, G4["D"] ** -0.5, True,
                        (-1, -1), arm.bufs["table"], 0, None, None, None, seq_threshold_3D=1000,
                        num_par_softmax_segments=2, softmax_segm_output=so, softmax_segm_max=sm,
                        softmax_segm_expsum=se, union_ext=arm.bufs["ext"], union_len=arm.bufs["ulen"], cis_bias=bv,
                        build_union=bu)
    d = float((o11.double() - o_fv.double()).abs().max())
    check("I11_3d_%s_vs_2d" % name, d <= 1e-5, d)
    check("I11_3d_%s_vs_fp64" % name, err(o11, bf, rows, G4) <= TOL, err(o11, bf, rows, G4))
    check("I11_3d_%s_wrote_partials" % name, float(se.abs().sum()) > 0)

# I12 the sort precondition: a swapped list is caught by the gate (VOID), whatever the kernel computes
bu_ = dict(bf)
pk, pt = bf["page_key"].clone(), bf["page_table"].clone()
pk[0, 0, [1, 2]] = pk[0, 0, [2, 1]]
pt[0, 0, [1, 2]] = pt[0, 0, [2, 1]]
bu_["page_key"], bu_["page_table"] = pk, pt
g12 = M.fv_table_gate("u", fv, [("unsorted", bu_)], 4, S, True)
check("I12_unsorted_is_void", (not g12["passed"]) and g12.get("status") == "VOID", g12["detail"])

# I13 the NOSA-8B geometry: BLOCK_M 64, D 128, 64 pages per row (one request)
GR = T.GEOM_REAL
cg = M.make_case(GR, 1, 4, 17)
bg = T.f32(M.batch_of(cg, "full"))
fg = M.FVArm(F, "FV", GR, cg["S"], 4, build=True, device="cpu")
og = fg(bg)
check("I13_real_geometry_vs_fp64", err(og, bg, [0, 1, 2, 3], GR) <= 2e-4, err(og, bg, [0, 1, 2, 3], GR))
g13 = M.fv_table_gate("1x4", fg, [("full", bg)], 4, cg["S"], True)
check("I13_real_geometry_tables", g13["passed"], g13["detail"])

# I14 the launcher's refusals (before any launch)
fv.prepare(bf)
kv, vv, bv, sr = views(bf, S)
base_kw = dict(q=bf["q"], k=kv, v=vv, out=torch.empty_like(bf["q"]), cu_seqlens_q=fv.bufs["cu"], max_seqlen_q=4,
               seqused_k=fv.bufs["seqused"], max_seqlen_k=sr, softmax_scale=0.25, causal=True, window_size=(-1, -1),
               block_table=fv.bufs["table"], softcap=0, q_descale=None, k_descale=None, v_descale=None,
               union_ext=fv.bufs["ext"], union_len=fv.bufs["ulen"], cis_bias=bv, build_union=build_args(bf, S))
for name, change in (("window", dict(window_size=(7, -1))), ("alibi", dict(alibi_slopes=torch.zeros(8))),
                     ("quant", dict(kv_quant_mode=F.KVQuantMode.FP8_PER_TENSOR)), ("rows3", dict(rows_per_q_block=3)),
                     ("qlen5", dict(max_seqlen_q=5)), ("no_ext", dict(union_ext=None)),
                     ("narrow_table", dict(block_table=fv.bufs["table"][:, :, :64].contiguous(),
                                           union_ext=fv.bufs["ext"][:, :, :64].contiguous()))):
    kw = dict(base_kw)
    kw.update(change)
    try:
        F.unified_attention(**kw)
        check("I14_refuses_" + name, False, "no refusal")
    except AssertionError:
        check("I14_refuses_" + name, True)

print("RAN %d: %s" % (len(ran), " ".join(ran)))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("FIA4-INTERP OK")
'''


def test_FV5_the_fork_under_the_triton_interpreter():
    """FV5 (needs triton; skipped without it): the fork on CPU float32 under TRITON_INTERPRET=1 (a fresh process):
    I1 FV and FVT = the fp64 reference (1e-4); I2 the kernel-built tables = the torch tables (the gate) = the set
    model; I3 FVT == FV bitwise; I4 the NEGATIVE CONTROL: 8 full pages dropped from one row's mask on one KV head are
    DETECTED (> TOL_C) and local (every other row and the other KV head bitwise unchanged), after its precondition;
    I5 / I6 the mask and bias witnesses DETECTED; I7 FI1-FI3 and FI5 (poison, no precondition) pass; I8 a row alone
    (4 copies) == in its request, bitwise (EXACT); I9 the block-edge batch; I10 requests of 1, 3 and 4 rows; I11 the
    3D split path (2 segments) = the 2D output (1e-5); I12 an unsorted list makes the table gate VOID; I13 the NOSA-8B
    geometry (BLOCK_M 64, D 128); I14 the launcher refuses window, ALiBi, quantized KV, 3 rows per block, 5 query
    rows, a missing ext table and a table narrower than BLOCK_Q x P_PAD."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(EFF / "r2_fiattn_bench.py"), str(FORK),
                        str(Path(__file__).resolve())], env=env, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=3000)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["FIA4-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# ------------------------------------------------------------------ FV6 completeness and the registered block
def test_FV6_completeness_and_pred4_in_the_notes():
    """FV6: COMPLETE expects FV's gates (CW_FV only with the bias variant) per cell tag; a missing one fails; without
    fv none is expected. docs/NOTES_fia4.md carries a REGISTERED line (the TBD marker or the lead's stamp) and a json
    block equal to PRED4; the band is the brief's (0.7 in [0.4, 1.1]); the thresholds are the rule (0.8, 1 ms, 32)."""
    names = ["G_block0", "C_FV", "MW_FV", "U_FV_table", "FI1_request_alone_FV", "FI2_order_FV",
             "FI3_other_requests_FV", "FI5_poison_FV", "CW_FV"]
    for tag, cell in (("2x4", (2, 4)), ("2x4s1", (2, 4, "s1"))):
        gates = [M.gate("%s_%s" % (n, tag), False, True, "") for n in names]
        assert M.completeness_gates(gates, [cell], ["FV"], True, fv=True)[0]["passed"]
        assert not M.completeness_gates(gates[1:], [cell], ["FV"], True, fv=True)[0]["passed"]
        assert M.completeness_gates(gates[:-1], [cell], ["FV"], False, fv=True)[0]["passed"]
        assert not M.completeness_gates(gates[:-1], [cell], ["FV"], True, fv=True)[0]["passed"]
        assert M.completeness_gates([], [cell], ["FV"], True)[0]["passed"]
    doc = (ROOT / "docs/NOTES_fia4.md").read_text()
    stamp = re.search(r"^REGISTERED (<TBD-for-lead-stamp>|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \(lead stamp)", doc, re.M)
    assert stamp, "no REGISTERED line (TBD marker or the lead's stamp) in the notes"
    assert M.PRED4["registered"].startswith("REGISTERED " + stamp.group(1)[:20])
    m = re.search(r"<!-- PRED4-BEGIN -->\s*```json\s*(.*?)```\s*<!-- PRED4-END -->", doc, flags=re.S)
    assert m, "no PRED4 block in the notes"
    assert json.loads(m.group(1)) == json.loads(json.dumps(M.PRED4))
    assert M.PRED4["fv_over_p_device"] == {"216x4": [0.7, 0.4, 1.1], "216x4s1": [0.7, 0.4, 1.1]}
    assert M.PRED4["thresholds"] == dict(fv_over_p_max=0.8, union_build_ms_per_tick_max=1.0, layers_per_tick=32)
    assert M.LAYERS_PER_TICK == M.PRED4["thresholds"]["layers_per_tick"]
    for word in ("0.8 x P", "1 ms", "MUST gates"):
        assert word in M.PRED4["decision"]


# ------------------------------------------------------------------ FV7 the decision report
def test_FV7_decision_report4_branches():
    """FV7: R_DECISION4 says ADOPT only with FV / P <= 0.8 (ncu), a union build <= 1 ms per tick (32 x (FV - FVT)
    ncu, else the event difference) and all FV MUST gates of the cell passing; each failing condition gives NOT MET;
    a failing gate of another cell or a FVT gate does not count; an unregistered cell gets no line."""
    def rep(tp, tf, tft, ev=None, tag="216x4"):
        arms = dict(P=dict(time_ns=tp), FV=dict(time_ns=tf, regs=[168], local_bytes=0.0, occ=[12.0], smem=[65536],
                                                read_over_union=1.01, l2sm_over_union=1.05))
        if tft is not None:
            arms["FVT"] = dict(time_ns=tft)
        return dict(ncu={tag: dict(summary=dict(arms=arms))},
                    cells={tag: dict(arms=dict(FV=dict(build_in_kernel_ms_per_tick_events=ev)))})
    ok = [M.gate("C_FV_216x4", True, True, ""), M.gate("G_block0_216x4", True, True, ""),
          M.gate("U_FV_table_216x4", True, True, ""), M.gate("C_FV_216x4s1", False, True, ""),
          M.gate("C_FVT_216x4", False, False, "")]
    g = M.decision_report4(rep(1000e3, 700e3, 690e3), ok, [(216, 4)])[0]
    assert g["name"] == "R_DECISION4_216x4" and g["detail"].startswith("ADOPT") and not g["must"]
    assert "union build per tick 0.3200 ms" in g["detail"] and "registers [168]" in g["detail"]
    assert M.decision_report4(rep(1000e3, 850e3, 840e3), ok, [(216, 4)])[0]["detail"].startswith("NOT MET")
    assert M.decision_report4(rep(1000e3, 700e3, 650e3), ok, [(216, 4)])[0]["detail"].startswith("NOT MET")
    g = M.decision_report4(rep(1000e3, 700e3, None, ev=0.5), ok, [(216, 4)])[0]
    assert g["detail"].startswith("ADOPT") and "ncu n/a, events 0.5000" in g["detail"]
    assert M.decision_report4(rep(1000e3, 700e3, None, ev=None), ok, [(216, 4)])[0]["detail"].startswith("NOT MET")
    bad = ok + [M.gate("U_FV_table_216x4", False, True, "")]
    g = M.decision_report4(rep(1000e3, 700e3, 690e3), bad, [(216, 4)])[0]
    assert g["detail"].startswith("NOT MET") and "NOT ALL PASS" in g["detail"]
    assert M.decision_report4(rep(1000e3, 700e3, 690e3), ok, [(32, 4)]) == []


# ------------------------------------------------------------------ FV8 ncu, argv, arguments
def test_FV8_ncu_summary_occupancy_argv_and_arguments(tmp_path):
    """FV8: the ncu summary of an FV row keeps the achieved occupancy, the shared memory (dynamic + static) and the
    occupancy limits, and ncu_arm_text prints them; the kernel filter matches the fork's kernel and still no torch
    kernel; KERNELS_PER_CALL FV = FVT = 1; the child argv carries --fv-kernel only with an FV arm; FV / FVT need
    --fv-kernel (a refusal that creates nothing) and a missing file is refused."""
    nv = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    extra = ["lts__t_sectors_srcunit_tex_op_read.sum", "launch__registers_per_thread",
             "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
             "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
             "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
             "launch__waves_per_multiprocessor"]
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(["ID", "Process ID", nv, "Kernel Name"] + list(M.NCU_REQUIRED) + extra)
    for i, (tag, nm, rd) in enumerate((("P", "_bi_split_kernel", 80), ("P", "_bi_combine_kernel", 20),
                                       ("FV", "kernel_unified_attention", 50), ("FVT", "kernel_unified_attention", 49))):
        w.writerow([str(i), "9", '9 "<default domain>:fia|%s:none"' % tag, nm, "1000", str(rd), "10", "3", "168", "0",
                    "12.5", "65536", "1024", "2", "2", "2.1"])
    rows = M.ncu_rows(buf.getvalue(), list(M.NCU_REQUIRED) + list(M.NCU_OPTIONAL) + list(M.NCU_LAUNCH))
    s = M.ncu_summary(rows, ["P", "FV", "FVT"], ideal=400.0, union=100.0, rpr=4)
    f = s["arms"]["FV"]
    assert f["occ"] == [12.5] and f["smem"] == [66560] and f["limits"] == [(2.0, 2.0, 2.1)] and f["n_kernels"] == 1
    assert f["expected_kernels"] == 1 == M.KERNELS_PER_CALL["FVT"] and not s["stray"]
    txt = M.ncu_arm_text("FV", f)
    assert "achieved occupancy [12.5] %" in txt and "shared memory per block [66560] B" in txt
    rx = re.compile(M.KERNEL_FILTER[len("regex:"):])
    assert rx.search("kernel_unified_attention")
    for nm in ("vectorized_elementwise_kernel", "CatArrayBatchedCopy", "bitonicSortKVInPlace"):
        assert not rx.search(nm)

    class A:
        ncu, config, biattn, sharedkv, seed = "/x/ncu", "/c.json", "/p3.py", None, 5
        fv_kernel = "/t/nosi/nosi/fork_vllm_unified_attention.py"
    argv = M.ncu_argv(A, 216, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "FV", "FVT"], False)
    assert argv[argv.index("--fv-kernel") + 1] == A.fv_kernel and argv[argv.index("--arms") + 1] == "P,FV,FVT"
    assert "--fv-kernel" not in M.ncu_argv(A, 216, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "FP0"], False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(num_attention_heads=32, num_key_value_heads=2, hidden_size=4096, head_dim=128)))
    mod = tmp_path / "p3.py"
    mod.write_text("")
    out = tmp_path / "out"
    base = ["--out", str(out), "--config", str(cfg), "--biattn", str(mod)]
    assert M.main(base + ["--arms", "P,FV"]) == 2
    assert M.main(base + ["--arms", "P,FVT", "--fv-kernel", str(tmp_path / "none.py")]) == 2
    assert not out.exists()
    a = M.parse_args(base + ["--arms", "P,FV,FVT", "--fv-kernel", str(mod), "--cells", "216x4"])
    assert a.arms == ["P", "FV", "FVT"] and a.fv_kernel == str(mod)


# ------------------------------------------------------------------ FV9 / FV10 the step
def test_FV9_step_static_shape_and_pins():
    """FV9: one main(); the claim fiattn_active after the existence check, with an owner file; the budget fits the
    step; piece 3's blob pin; the fork is THIS tree's file (its blob recorded and checked before the launch); the S1
    tables' size; the fia3 cells; the arms P, FP0, FPU, FV, FVT; --fv-kernel and --s1-tables; no interpreter word."""
    src = (EFF / "r2_fia4_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    assert src.index('[ ! -e "$OUT" ]') < src.index('mkdir "$OUTROOT/fiattn_active"') < src.index("fiattn_active/owner")
    step_min = int(re.search(r"^STEP_MIN=(\d+)$", src, flags=re.M).group(1))
    budget = int(re.search(r"^BUDGET_S=(\d+)$", src, flags=re.M).group(1))
    assert step_min <= 45 and budget + 300 <= step_min * 60
    assert "BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042" in src
    assert re.search(r"^FV_PATH=nosi/nosi/fork_vllm_unified_attention.py(\s|$)", src, flags=re.M) and FORK.is_file()
    assert src.index('[ "$(git hash-object -- "$FV_MOD")" = "$FV_BLOB" ]') < src.index('"${ARGV[@]}" < /dev/null')
    m = re.search(r"^S1_BYTES=(\d+)", src, flags=re.M)
    assert m and "S1_TABLES=%s" % S1_RECORDED in src
    if S1_RECORDED.is_file():
        assert S1_RECORDED.stat().st_size == int(m.group(1))
    assert re.search(r"^CELLS=32x4,216x4,32x4s1,216x4s1(\s|$)", src, flags=re.M)
    assert re.search(r"^ARMS=P,FP0,FPU,FV,FVT(\s|$)", src, flags=re.M)
    for wd in ('--fv-kernel "$FV_MOD"', '--s1-tables "$S1_COPY"', "--no-strided", '--arms "$ARMS"', "NOSI=%s" % NOSI4,
               "fia4_${RUN}_$JOB"):
        assert wd in src, wd
    for word in ("awk", "perl", "python3 -", "sed ", "<<", "--verify-step"):
        assert word not in src, word


def test_FV10_step_dryrun_refusals_and_pins(tmp_path):
    """FV10 (as FC20): usage (2); a missing uuid file, a GPU- prefix, two lines STOP (1); FIATTN_HOLDERS outside
    DRYRUN refused; a wrong --nosi stops at HEAD. On a clean checkout of THIS worktree at HEAD the DRYRUN prints the
    pins (piece 3, the fork's blob, the S1 tables) and the bench arguments."""
    script = EFF / "r2_fia4_step.sh"
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
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(NOSI4).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % NOSI4)
    r = run(426, uid + "\n", sha=head.stdout.strip())
    assert r.returncode == 0, r.stderr[-400:]
    blob = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:nosi/nosi/fork_vllm_unified_attention.py"],
                          capture_output=True, text=True).stdout.strip()
    assert "fork nosi/nosi/fork_vllm_unified_attention.py blob %s" % blob in r.stdout
    assert "S1 tables %s (439458 B)" % S1_RECORDED in r.stdout
    for wd in ("--mode main", "--cells 32x4\\,216x4\\,32x4s1\\,216x4s1", "--arms P\\,FP0\\,FPU\\,FV\\,FVT",
               "--no-strided", "--ncu /opt/nvidia/nsight-compute/2025.1.0/ncu", "--budget-s 2400", "--time=45",
               "--fv-kernel %s/nosi/nosi/fork_vllm_unified_attention.py" % NOSI4,
               "FLASHINFER_WORKSPACE_BASE=/mnt/beegfs/ojerbi/nosa-env/jit/fia4_t1_426/flashinfer",
               "--s1-tables /mnt/beegfs/ojerbi/r2-runs/fiattn/fia4_t1_426.step/skv_tables.pt"):
        assert wd in r.stdout, wd
