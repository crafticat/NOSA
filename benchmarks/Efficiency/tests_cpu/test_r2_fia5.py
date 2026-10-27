"""CPU tests of the fia5 edits: the fork nosi/nosi/fork_vllm_unified_attention.py after fia5 (the rows' extents as
vectors from the table, no extents mask on K, V read up to the tile's reach, a float bias, a token tile of a part of a
block, use_extents=False), the [fia5] parts of benchmarks/Efficiency/r2_fiattn_bench.py (FVN, --fv-sweep, the swept
arms' gates, the compile witness, R_FV_sweep, R_DECISION5), benchmarks/Efficiency/r2_fia5_step.sh and
docs/NOTES_fia5.md. Everything that needs no GPU. SOURCE ONLY; the lead runs them on a compute node (with the fia4,
fia3 and fiattn tests, which run on THIS tree's fork and bench):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia5.py benchmarks/Efficiency/tests_cpu/test_r2_fia4.py \
        benchmarks/Efficiency/tests_cpu/test_r2_fia3.py benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py

F5_1 static: the fia5 tile loop (vector extents loads with the page-id idiom, no scalar extents load, no extents mask
     on K, upstream's V mask kept, no mask on the bias, the new keywords reach the kernel, the compile handle kept);
     the [fork] tag check of the whole file is FV2 in test_r2_fia4.py.
F5_2 the sweep flag: canonical names, defaults, refusals, parse_args, the ncu child's arms and argv.
F5_3 every swept configuration runs the same gates (run_fv with a pure-torch FAKE of the fork: fast, and a swept arm
     whose first call fails becomes a REPORT FINDING, not a MUST failure); completeness with a planted missing gate.
F5_4 R_FV_sweep (the best swept arm) and R_DECISION5 branches.
F5_5 the compile witness (the fia4 record's TTGIR lines; the programs-per-SM model gives the measured 2).
F5_6 the fork under TRITON_INTERPRET=1 (a fresh process): tile 32 and 16, the bf16 bias, FVN, the 3D path at tile 32,
     ragged requests, two MUTATION controls (without the V reach mask, and without the score mask, the poisoned batch
     is not finite: FI5 rests on those two lines), launcher refusals, and run_fv on swept arms with the real kernel.
F5_7 PRED5 and the notes.
F5_8 / F5_9 the step: static shape, DRYRUN refusals and pins.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
FORK = ROOT / "nosi/nosi/fork_vllm_unified_attention.py"
NOSI5 = "/mnt/beegfs/ojerbi/nosi-r2-fia5"
S1_RECORDED = Path("/mnt/beegfs/ojerbi/r2-runs/sharedkv/skv_probe_s1a_2191262/skv_tables.pt")
STEP5 = EFF / "r2_fia5_step.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load(EFF / "r2_fiattn_bench.py", "r2_fiattn_bench_fia5_under_test")
T4 = _load(HERE / "test_r2_fia4.py", "test_r2_fia4_helpers_for_fia5")      # set_model, tables_match, f32, GEOM4
GEOM4 = T4.GEOM4


# ------------------------------------------------------------------ helpers (also used by the interpreter child)
def f32_keep_bias(b):
    """The batch with float32 q / k / v and the store's bf16 bias (the lists unchanged)."""
    out = dict(b)
    for f in ("q", "k", "v"):
        out[f] = b[f].float().contiguous()
    return out


def case_f32(case):
    """The case with float32 q (every batch) and K / V; the bias stays bf16 (B16 and B32 then differ in dtype)."""
    c = dict(case)
    for which in ("full", "alt", "short", "altreq"):
        c[which] = dict(case[which])
        c[which]["q"] = case[which]["q"].float().contiguous()
    c["k"], c["v"] = case["k"].float().contiguous(), case["v"].float().contiguous()
    return c


def gate_set(gates, x):
    """The gate names of fork arm x with x replaced by <A> (the cross-arm REPORTs R_D_bits_vs_FV and R_FVT_equals_FV
    left out), sorted: equal sets = the same gates."""
    key = "_%s_" % x
    return sorted(g["name"].replace(key, "_<A>_") for g in gates if key in g["name"]
                  and not g["name"].startswith(("R_D_bits_vs_FV_", "R_FVT_equals_FV_")))


def kernel_body():
    """The source of kernel_unified_attention and of unified_attention (ast.unparse)."""
    tree = ast.parse(FORK.read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return fns["kernel_unified_attention"], fns["unified_attention"], tree


class FakeFork:
    """A pure-torch model of the fork's launcher for the bench's plumbing tests: float64 math over the union tables
    of the launch (the bench's torch builder writes them when build_union is given), each (request, KV head, row)
    attends to the tokens its byte of each union entry gives (all PAGE with use_extents=False). It records the launch
    keywords of every call; fail_tile: a call with that tile_size raises (a compile or resource failure)."""
    def __init__(self, fail_tile=None):
        self.calls, self.fail_tile = [], fail_tile
        self.FORK_LAST_LAUNCH = {"kernel": None}

    def unified_attention(self, q, k, v, out, cu, max_seqlen_q, seqused_k, max_seqlen_k, scale, causal, window,
                          table, softcap, q_descale, k_descale, v_descale, union_ext=None, union_len=None,
                          cis_bias=None, build_union=None, rows_per_q_block=4, num_warps=None, num_stages=None,
                          tile_size=None, use_extents=True):
        self.calls.append(dict(num_warps=num_warps, num_stages=num_stages, tile_size=tile_size,
                               use_extents=use_extents, bias_dtype=cis_bias.dtype))
        if self.fail_tile is not None and tile_size == self.fail_tile:
            raise RuntimeError("fake: out of resources for tile %d" % tile_size)
        nb = cu.numel() - 1
        if build_union is not None:
            b = {f: build_union[f] for f in ("page_key", "page_table", "num_pages", "last_len", "seq_idx")}
            u = M.fp_union(b, rows_per_q_block, int(build_union["slots_per_entry"]))
            t, x, n = M.fv_tables_from_union(u, table.shape[-1])
            table.copy_(t)
            union_ext.copy_(x)
            union_len.copy_(n)
        nblk, page, hkv, d = k.shape
        g = q.shape[1] // hkv
        kf, vf, bb = k.reshape(nblk * page, hkv, d), v.reshape(nblk * page, hkv, d), cis_bias.reshape(nblk * page, hkv)
        ar = torch.arange(page).view(1, -1)
        for n in range(nb):
            for u_, r in enumerate(range(int(cu[n]), int(cu[n + 1]))):
                for h in range(hkv):
                    c = int(union_len[h, n])
                    pages = table[h, n, :c].long().view(-1, 1)
                    cnt = ((union_ext[h, n, :c].long() >> (8 * u_)) & 0xFF) if use_extents else torch.full((c,), page)
                    ix = (pages * page + ar)[ar < cnt.view(-1, 1)]
                    w = torch.softmax((q[r, h * g:(h + 1) * g].double() @ kf[ix, h].double().t()
                                       + bb[ix, h].double().unsqueeze(0)) * scale, dim=1)
                    out[r, h * g:(h + 1) * g] = (w @ vf[ix, h].double()).to(out.dtype)


def fake_cell(names, fail_tile=None, seed=7):
    """run_fv on a 2 x 4 generator cell (GEOM4, float32 q / K / V, bf16 bias) with the fake fork and arms names.
    Returns (gates, arms, outs, rep, fake)."""
    case = case_f32(M.make_case(GEOM4, 2, 4, seed))
    S = case["S"]
    fake = FakeFork(fail_tile)
    bf = M.batch_of(case, "full")
    rows = list(range(8))
    ref = M.reference_fp64(bf, rows, GEOM4)
    ref0 = M.reference_fp64(bf, rows, GEOM4, zero_bias=True)
    pois, _ = M.poison_caches(case, bf)
    arms = {n: M.make_fv_arm(fake, n, GEOM4, S, 4, "cpu") for n in names}
    gates, outs, rep = [], {}, dict(arms={})
    M.run_fv(GEOM4, case, arms, outs, bf, rows, ref, ref0, pois, 4, "2x4", True, fake, gates, rep)
    return gates, arms, outs, rep, fake


# ------------------------------------------------------------------ F5_1 the fia5 tile loop (static)
def test_F5_1_the_fia5_tile_loop_static():
    """F5_1: in kernel_unified_attention: no scalar extents load (fia4's tl.load(union_ext_ptr + block_table_offset +
    j)); the two extents loads are vector loads (the page-id index seq_offset // BLOCK_SIZE, and offs_m % TILE_SIZE for
    the rows); the K copy's mask is dim_mask only; the V copy keeps upstream's mask (dim_mask & tile_mask); the bias
    load has no mask; USE_EXTENTS is a kernel constexpr and the launcher passes it, with tile_size and use_extents as
    new keywords; the launch keeps the CompiledKernel in FORK_LAST_LAUNCH; every line with a fia5 name carries
    [fork]."""
    kern, launcher, tree = kernel_body()
    src = ast.unparse(kern)
    assert "tl.load(union_ext_ptr + block_table_offset + j)" not in src
    loads = [ast.unparse(c) for c in ast.walk(kern) if isinstance(c, ast.Call) and ast.unparse(c.func) == "tl.load"]
    ext_loads = sorted(x for x in loads if "union_ext_ptr" in x)                # ast.walk has no specified order
    assert ext_loads == sorted(["tl.load(union_ext_ptr + block_table_offset + seq_offset // BLOCK_SIZE)",
                                "tl.load(union_ext_ptr + block_table_offset + (j * TILE_SIZE + offs_m % TILE_SIZE) "
                                "// BLOCK_SIZE)"]), ext_loads

    def mask_of(target):
        for n in ast.walk(kern):
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target:
                kws = {k.arg: ast.unparse(k.value) for k in n.value.keywords}
                return kws.get("mask")
        raise AssertionError("no assignment to %s" % target)
    assert mask_of("K_load") == "dim_mask[:, None]"
    assert mask_of("V_load") == "dim_mask[None, :] & tile_mask[:, None]"
    cis = [n for n in ast.walk(kern) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "cis"
           and "tl.load" in ast.unparse(n.value)]
    assert len(cis) == 1 and "mask" not in ast.unparse(cis[0].value)
    params = [a.arg for a in kern.args.args]
    assert "USE_EXTENTS" in params and kern.args.args[params.index("USE_EXTENTS")].annotation is not None
    largs = [a.arg for a in launcher.args.args]
    assert largs[-2:] == ["tile_size", "use_extents"]
    lsrc = ast.unparse(launcher)
    assert "USE_EXTENTS=bool(use_extents)" in lsrc
    assert "FORK_LAST_LAUNCH['kernel'] = kernel_unified_attention[grid]" in lsrc
    assert any(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "FORK_LAST_LAUNCH" for n in tree.body)
    for ln in FORK.read_text().splitlines():
        if any(w in ln for w in ("USE_EXTENTS", "TILES_PER_BLOCK", "ext_rows", "ext_t", "FORK_LAST_LAUNCH",
                                 "use_extents", "tile_size=None")):
            assert "[fork]" in ln, ln


# ------------------------------------------------------------------ F5_2 the sweep flag
def test_F5_2_the_sweep_flag_parses(tmp_path):
    """F5_2: parse_fv_sweep gives canonical names (tile default PAGE, bias default 32), the brief's grid of 6, refuses a
    bad item, an out-of-range value and a repeat; fv_sweep_launch inverts fv_sweep_name and refuses non-canonical
    names; parse_arms accepts a canonical swept name (the ncu child's --arms) and still refuses XX; parse_args appends
    the swept arms after --arms and refuses --fv-sweep without FV (a refusal that creates nothing); the ncu argv of a
    swept arm carries --fv-kernel; make_fv_arm builds each kind with its launch keywords."""
    grid = M.parse_fv_sweep("w4s2,w4s3,w4s4,w8s2,w8s3,w8s4")
    assert grid == ["FVW4S2T64B32", "FVW4S3T64B32", "FVW4S4T64B32", "FVW8S2T64B32", "FVW8S3T64B32", "FVW8S4T64B32"]
    assert M.parse_fv_sweep("w8s4t32,w4s3b16, w4s3t16b16") == ["FVW8S4T32B32", "FVW4S3T64B16", "FVW4S3T16B16"]
    assert M.parse_fv_sweep("") == [] and M.parse_fv_sweep(None) == []
    for bad in ("w4", "s3", "w3s3", "w4s0", "w4s7", "w4s3t128", "w4s3t48", "w4s3t8", "w4s3b8", "W4S3", "w4s3,w4s3",
                "w4s3t64,w4s3", "x"):
        with pytest.raises(ValueError):
            M.parse_fv_sweep(bad)
    for name in grid + ["FVW8S4T32B32", "FVW4S3T16B16"]:
        cfg = M.fv_sweep_launch(name)
        assert M.fv_sweep_name(cfg["num_warps"], cfg["num_stages"], cfg["tile_size"], cfg["bias_bits"]) == name
    for bad in ("FV", "FVT", "FVW04S3T64B32", "FVW4S3T64", "FVW4S3T48B32", "FVW3S3T64B32", "fvw4s3t64b32"):
        assert M.fv_sweep_launch(bad) is None
    assert M.parse_arms("P,FV,FVW8S4T32B32,FVN") == ["P", "FV", "FVW8S4T32B32", "FVN"]
    for bad in ("P,XX", "P,FVW3S3T64B32", "P,FVW4S3T64B32,FVW4S3T64B32"):
        with pytest.raises(ValueError):
            M.parse_arms(bad)
    assert M.is_fv_arm("FVN") and M.is_fv_arm("FVW4S3T64B32") and not M.is_fv_arm("FP0")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(num_attention_heads=32, num_key_value_heads=2, hidden_size=4096, head_dim=128)))
    mod = tmp_path / "p3.py"
    mod.write_text("")
    out = tmp_path / "out"
    base = ["--out", str(out), "--config", str(cfg), "--biattn", str(mod), "--fv-kernel", str(mod)]
    a = M.parse_args(base + ["--arms", "P,FV,FVT,FVN", "--fv-sweep", "w4s3,w8s4t32,w4s3b16", "--cells", "216x4"])
    assert a.arms == ["P", "FV", "FVT", "FVN", "FVW4S3T64B32", "FVW8S4T32B32", "FVW4S3T64B16"]
    assert a.fv_sweep == ["FVW4S3T64B32", "FVW8S4T32B32", "FVW4S3T64B16"]
    assert M.parse_args(base + ["--arms", "P,FV"]).fv_sweep == []
    assert M.main(base + ["--arms", "P,FVT", "--fv-sweep", "w4s3"]) == 2
    assert M.main(base + ["--arms", "P,FV", "--fv-sweep", "w4s9"]) == 2
    assert not out.exists()

    class A:
        ncu, config, biattn, sharedkv, seed = "/x/ncu", "/c.json", "/p3.py", None, 5
        fv_kernel = "/t/nosi/nosi/fork_vllm_unified_attention.py"
    argv = M.ncu_argv(A, 216, 4, "/o/c", "/o/rep", list(M.NCU_REQUIRED), ["P", "FVW8S4T32B32"], False)
    assert argv[argv.index("--fv-kernel") + 1] == A.fv_kernel and argv[argv.index("--arms") + 1] == "P,FVW8S4T32B32"
    fake = FakeFork()
    a1 = M.make_fv_arm(fake, "FVW8S4T32B16", GEOM4, 256, 4, "cpu")
    assert a1.launch_kw == dict(num_warps=8, num_stages=4, tile_size=32) and a1.bias_bits == 16 and a1.build_in_kernel
    a2 = M.make_fv_arm(fake, "FVN", GEOM4, 256, 4, "cpu")
    assert a2.launch_kw == dict(use_extents=False) and a2.bias_bits == 32 and a2.build_in_kernel
    a3 = M.make_fv_arm(fake, "FVT", GEOM4, 256, 4, "cpu")
    assert a3.launch_kw == {} and not a3.build_in_kernel
    a4 = M.make_fv_arm(fake, "FVW8S4T32B16", GEOM4, 256, 4, "cpu", ones=True)
    assert a4.ones and not a4.build_in_kernel and a4.launch_kw == a1.launch_kw and a4.name == "FVW8S4T32B16ones"
    assert M.make_fv_arm(fake, "FV", GEOM4, 256, 4, "cpu").launch_kw == {}
    with pytest.raises(ValueError):
        M.make_fv_arm(fake, "FP0", GEOM4, 256, 4, "cpu")
    assert M.KERNELS_PER_CALL["FVN"] == 1 and "FVN" in M.ARMS and "FVN" in M.FV_ARMS


# ------------------------------------------------------------------ F5_3 every swept config runs the same gates
def test_F5_3_every_swept_configuration_runs_the_same_gates():
    """F5_3 (the fake fork, fast): run_fv on FV, three swept arms, FVN and FVT. Each swept arm gets FV's gate set
    under its own name (gate_set equal), every MUST gate passes, each swept arm's calls carry its own num_warps /
    num_stages / tile_size and its bias dtype (B32 float32, B16 the store's bf16), FVN calls with use_extents=False and
    is DETECTED as not attention, COMPLETE passes, and a planted missing gate of one swept arm fails COMPLETE. A swept
    arm whose first call raises gives a REPORT FINDING (no MUST failure), leaves arms, and COMPLETE still passes."""
    sw = ["FVW4S3T64B32", "FVW8S4T32B32", "FVW4S3T64B16"]
    names = ["FV"] + sw + ["FVN", "FVT"]
    gates, arms, outs, rep, fake = fake_cell(names)
    ref_set = gate_set(gates, "FV")
    assert len(ref_set) >= 12, ref_set
    for x in sw:
        assert gate_set(gates, x) == ref_set, (x, gate_set(gates, x), ref_set)
    bad = [g for g in gates if g["must"] and not g["passed"]]
    assert not bad, [(g["name"], g["detail"][:200]) for g in bad]
    seen = {(c["num_warps"], c["num_stages"], c["tile_size"], c["use_extents"], c["bias_dtype"]) for c in fake.calls}
    assert (4, 3, 64, True, torch.float32) in seen and (8, 4, 32, True, torch.float32) in seen
    assert (4, 3, 64, True, torch.bfloat16) in seen and (None, None, None, False, torch.float32) in seen
    assert (None, None, None, True, torch.float32) in seen                       # FV, FVT: Triton's defaults
    by = {g["name"]: g for g in gates}
    assert by["R_FVN_not_attention_2x4"]["status"] == "DETECTED"
    assert by["X_row_alone_FVW8S4T32B32_2x4"]["passed"] and by["R_D_bits_vs_FV_FVW4S3T64B16_2x4"]["detail"].startswith(
        "8 of 8")
    assert set(rep["fv_compile"]) >= set(sw) | {"FV", "FVN", "FVT"}
    comp = M.completeness_gates(gates, [(2, 4)], names, True, fv=True)
    assert comp[0]["passed"], comp[0]["detail"]
    drop = [g for g in gates if g["name"] != "U_FVW8S4T32B32_table_2x4"]
    assert not M.completeness_gates(drop, [(2, 4)], names, True, fv=True)[0]["passed"]
    drop = [g for g in gates if g["name"] != "R_FVW8S4T32B32_first_call_2x4"]
    out = M.completeness_gates(drop, [(2, 4)], names, True, fv=True)[0]
    assert not out["passed"] and "R_FVW8S4T32B32_first_call_2x4" in out["detail"]
    gates, arms, outs, rep, fake = fake_cell(["FV", "FVW4S3T64B32", "FVW8S4T32B32"], fail_tile=32)
    by = {g["name"]: g for g in gates}
    g1 = by["R_FVW8S4T32B32_first_call_2x4"]
    assert not g1["passed"] and not g1["must"] and g1["detail"].startswith("FINDING")
    assert "FVW8S4T32B32" not in arms and "FVW4S3T64B32" in arms
    assert not [g for g in gates if "_FVW8S4T32B32_" in g["name"] and g["name"] != "R_FVW8S4T32B32_first_call_2x4"]
    assert not [g for g in gates if g["must"] and not g["passed"]]
    assert M.completeness_gates(gates, [(2, 4)], ["FV", "FVW4S3T64B32", "FVW8S4T32B32"], True, fv=True)[0]["passed"]


# ------------------------------------------------------------------ F5_4 the best swept arm and R_DECISION5
def test_F5_4_best_swept_arm_and_decision_report5():
    """F5_4: fv_sweep_report picks the lowest event median among the swept arms whose MUST gates all pass (a faster
    arm with a failing gate, or with no gate, is never the best); R_DECISION5 says ADOPT <config> only with X / P <=
    0.8 (ncu), X's MUST gates passing and the build <= 1 ms per tick (32 x (FV - FVT), else the events), the lower one
    when both qualify; it prints the extents' share (FV - FVN) / FV; an unregistered cell gets no line."""
    tag = "216x4"
    rep = dict(arms={x: dict(call_ms=dict(median=t)) for x, t in (
        ("P", 1.0), ("FV", 0.9), ("FVN", 0.8), ("FVW4S3T64B32", 0.92), ("FVW8S4T32B32", 0.7), ("FVW4S4T32B32", 0.6),
        ("FVW8S3T64B32", 0.5))}, fv_compile={"FVW8S4T32B32": dict(regs=168, kv_depth=2)})
    gates = [M.gate("G_block0_%s" % tag, True, True, "")]
    for x in ("FV", "FVW4S3T64B32", "FVW8S4T32B32", "FVW4S4T32B32"):
        gates.append(M.gate("C_%s_%s" % (x, tag), x != "FVW4S4T32B32", True, ""))
        gates.append(M.gate("U_%s_table_%s" % (x, tag), True, True, ""))
    gates.append(M.gate("C_FVW8S4T32B32_216x4s1", False, True, ""))                     # another cell: no effect
    best = M.fv_sweep_report(rep, tag, gates, ["FVW4S3T64B32", "FVW8S4T32B32", "FVW4S4T32B32", "FVW8S3T64B32"])
    assert best == "FVW8S4T32B32" == rep["fv_best"]
    g = gates[-1]
    assert g["name"] == "R_FV_sweep_216x4" and not g["must"] and "best swept arm FVW8S4T32B32" in g["detail"]
    assert "FVW4S4T32B32 0.6000 ms (x0.600 of P; MUST NOT ALL PASS" in g["detail"]

    def report(tp, tf, tft, tfn, tb, best="FVW8S4T32B32", ev=None):
        arms = dict(P=dict(time_ns=tp), FV=dict(time_ns=tf, regs=[255], local_bytes=4e5, occ=[12.4], smem=[60416],
                                                 read_over_union=1.01))
        if tft is not None:
            arms["FVT"] = dict(time_ns=tft)
        if tfn is not None:
            arms["FVN"] = dict(time_ns=tfn)
        if tb is not None:
            arms[best] = dict(time_ns=tb, regs=[168], local_bytes=0.0, occ=[24.0], smem=[53248], read_over_union=1.01)
        cell = dict(fv_best=best, fv_compile={best: dict(kv_depth=2, programs_per_sm=dict(model=2))},
                    arms=dict(FV=dict(build_in_kernel_ms_per_tick_events=ev)))
        return dict(ncu={tag: dict(summary=dict(arms=arms))}, cells={tag: cell})
    ok = [M.gate("G_block0_216x4", True, True, ""), M.gate("C_FV_216x4", True, True, ""),
          M.gate("C_FVW8S4T32B32_216x4", True, True, "")]
    d = M.decision_report5(report(1000e3, 900e3, 890e3, 850e3, 700e3), ok, [(216, 4)])[0]
    assert d["name"] == "R_DECISION5_216x4" and not d["must"] and d["detail"].startswith("ADOPT FVW8S4T32B32")
    assert "union build per tick 0.3200 ms" in d["detail"] and "(FV - FVN) / FV 0.0556" in d["detail"]
    assert "FVW8S4T32B32 / P device 0.7000" in d["detail"] and "depth 2" in d["detail"]
    assert M.decision_report5(report(1000e3, 750e3, 740e3, 700e3, 790e3), ok, [(216, 4)])[0]["detail"].startswith(
        "ADOPT FV:")
    assert M.decision_report5(report(1000e3, 900e3, 890e3, None, 850e3), ok, [(216, 4)])[0]["detail"].startswith(
        "NOT MET")
    bad = ok[:2] + [M.gate("C_FVW8S4T32B32_216x4", False, True, "")]
    assert M.decision_report5(report(1000e3, 900e3, 890e3, 850e3, 700e3), bad, [(216, 4)])[0]["detail"].startswith(
        "NOT MET")
    assert M.decision_report5(report(1000e3, 900e3, 860e3, 850e3, 700e3), ok, [(216, 4)])[0]["detail"].startswith(
        "NOT MET")                                                          # build 32 x 40 us = 1.28 ms
    d = M.decision_report5(report(1000e3, 900e3, None, None, 700e3, ev=0.5), ok, [(216, 4)])[0]
    assert d["detail"].startswith("ADOPT FVW8S4T32B32") and "ncu n/a, events 0.5000" in d["detail"]
    d = M.decision_report5(report(1000e3, 900e3, 890e3, 850e3, None, best=None), ok, [(216, 4)])[0]
    assert d["detail"].startswith("NOT MET") and "best swept: none" in d["detail"]
    assert M.decision_report5(report(1000e3, 700e3, 690e3, 650e3, 600e3), ok, [(32, 4)]) == []


# ------------------------------------------------------------------ F5_5 the compile witness
FIA4_TTGIR = """    %1399 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc209)
    %1400 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64xi32, #shared2, #triton_gpu.shared_memory, mutable> loc(#loc210)
    %1401 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc211)
    %1402 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64xi32, #shared2, #triton_gpu.shared_memory, mutable> loc(#loc210)
      %1555 = triton_gpu.local_alloc %1554 : (tensor<64x64xbf16, #mma>) -> !tt.memdesc<64x64xbf16, #shared, #triton_gpu.shared_memory> loc(#loc267)
"""


def test_F5_5_compile_witness():
    """F5_5: fv_compile_info on a handle shaped like Triton 3.2's CompiledKernel, with the fia4 record's TTGIR lines
    (kernel_unified_attention.ttgir:1524-1527 of fia4_k1a_2195573; the in-loop P alloc has an operand and is not a
    loop buffer): buffers 1x128x64xbf16 / 1x64x128xbf16 / 1x64xi32, largest depth 1; the programs-per-SM MODEL at 255
    registers, 4 warps, 59392 B gives 2 (the record's measured limit); a 2-deep copy gives depth 2; no handle -> None
    and a REPORT that says so. The ncu latency witness: the long-scoreboard stall share is an optional metric, the
    WarpStateStats section follows LaunchStats, and the summary and its text keep the value per arm."""
    MD = namedtuple("MD", "num_warps num_stages shared")

    class CK:
        def __init__(self, text, regs, spills, md):
            self.asm, self.n_regs, self.n_spills, self.metadata = {"ttgir": text}, regs, spills, md

    class Mod:
        FORK_LAST_LAUNCH = {"kernel": CK(FIA4_TTGIR, 255, 1, MD(4, 3, 59392))}
    info = M.fv_compile_info(Mod, dict(num_warps=4))
    assert info["buffers"] == ["1x128x64xbf16", "1x64x128xbf16", "1x64xi32"] and info["kv_depth"] == 1
    assert info["regs"] == 255 and info["local_words_per_thread"] == 1 and info["shared"] == 59392
    assert info["programs_per_sm"] == dict(registers=2, shared=2, warps=16, model=2)
    Mod.FORK_LAST_LAUNCH = {"kernel": CK(FIA4_TTGIR.replace("<1x128x64x", "<2x128x64x"), 168, 0, MD(8, 4, 92160))}
    info = M.fv_compile_info(Mod)
    assert info["kv_depth"] == 2 and info["programs_per_sm"]["model"] == 1
    g, info = M.fv_compile_gate("FVW8S4T64B32", "216x4", Mod, None)
    assert g["passed"] and "largest depth 2" in g["detail"] and g["name"] == "R_FVW8S4T64B32_compile_216x4"

    class Bare:
        FORK_LAST_LAUNCH = {"kernel": None}
    assert M.fv_compile_info(Bare) is None and M.fv_compile_info(object()) is None
    g, info = M.fv_compile_gate("FV", "2x4", Bare, None)
    assert info is None and not g["passed"] and not g["must"]
    lsb = "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"     # the ncu latency witness (optional)
    assert lsb in M.NCU_OPTIONAL and "WarpStateStats" in M.NCU_SECTIONS and M.NCU_SECTIONS[0] == "LaunchStats"
    row = {"id": 0, "name": "kernel_unified_attention", "tag": "FVW8S4T32B32", "gpu__time_duration.sum": 1000.0,
           lsb: 41.234}
    s = M.ncu_summary([row], ["FV", "FVW8S4T32B32"], ideal=1.0, union=1.0, rpr=4)
    p = s["arms"]["FVW8S4T32B32"]
    assert p["stall_lsb"] == [41.23] and p["expected_kernels"] == 1 == p["n_kernels"] and not s["stray"]
    assert "long-scoreboard stall [41.23] % of active warps" in M.ncu_arm_text("FVW8S4T32B32", p)
    assert s["arms"]["FV"]["stall_lsb"] == [] and "long-scoreboard" not in M.ncu_arm_text("FV", s["arms"]["FV"])


# ------------------------------------------------------------------ F5_6 the fork under the interpreter
INTERP_CHILD5 = r'''
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


T = load(sys.argv[3], "tests_fia5_interp")          # this test file: its bench copy T.M, the fia4 helpers T.T4
M, T4 = T.M, T.T4
F, why = M.import_fv(sys.argv[2])
assert F is not None, why
TMP = sys.argv[4]
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


G4 = T4.GEOM4
case = M.make_case(G4, 2, 4, 11)
S = case["S"]
bf = T4.f32(M.batch_of(case, "full"))
baq = T4.f32(M.batch_of(case, "altreq"))
rows = list(range(8))
ref = M.reference_fp64(bf, rows, G4)
ref0 = M.reference_fp64(bf, rows, G4, zero_bias=True)
u_full = M.fp_union(bf, 4, S)
pois, _ = M.poison_caches(case, bf)
bp = dict(bf)
bp.update(pois)

# J1 the default (tile 64) and a 32-token tile against fp64; the swept arm's launch keywords
fv = M.FVArm(F, "FV", G4, S, 4, build=True, device="cpu")
a32 = M.make_fv_arm(F, "FVW4S3T32B32", G4, S, 4, "cpu")
o64 = fv(bf)
o32 = a32(bf)
check("J1_t64_vs_fp64", err(o64, bf, rows, G4) <= TOL, err(o64, bf, rows, G4))
check("J1_t32_vs_fp64", err(o32, bf, rows, G4) <= TOL, err(o32, bf, rows, G4))
check("J1_launch_kw", a32.launch_kw == dict(num_warps=4, num_stages=3, tile_size=32), a32.launch_kw)
check("J1_t32_tables", *T4.tables_match(T4.set_model(bf, T4.groups_of(2, 4), S), *a32.tables()))
check("J13_no_handle_under_the_interpreter", F.FORK_LAST_LAUNCH["kernel"] is None and M.fv_compile_info(F) is None)

# J2 FI1-FI3 and FI5 at tile 32; the poison plant has teeth (entries reached in part: poison INSIDE loaded tiles)
npart = M.fv_partial_reach(u_full)
check("J2_poison_has_teeth", npart > 0, npart)
gs, full32 = M.request_suite("FVW4S3T32B32", "2x4", a32, bf, baq, 4, [0, 1],
                             [("reversed_requests", [1, 0]), ("random", [1, 0])], pois, (True, "none"), True)
check("J2_request_suite_t32", len(gs) == 4 and all(x["passed"] for x in gs),
      [x["detail"] for x in gs if not x["passed"]])
op64 = fv(bp)
check("J2_poison_t64_finite_and_equal", bool(torch.isfinite(op64).all()) and M.bits_equal(op64, o64))

# J3 a row alone at tile 32 (a tile the row does not see is an identity step): bitwise
gx = M.fv_row_alone_gate("2x4", a32, bf, rows, 4, full32, gname="X_row_alone_FVW4S3T32B32_2x4")
check("J3_row_alone_exact_t32", gx["passed"], gx["detail"])

# J4 the mask and bias witnesses at tile 32
ones = M.make_fv_arm(F, "FVW4S3T32B32", G4, S, 4, "cpu", ones=True)
gw = M.mask_witness("2x4", ones, bf, rows, ref, u_full, M.TOL_C, True, gname="MW_x")
check("J4_mask_witness_t32", gw["passed"], gw["detail"])
gb = M.bias_witness("2x4", a32, bf, rows, ref, ref0, M.TOL_C, True, gname="CW_x")
check("J4_bias_witness_t32", gb["passed"], gb["detail"])

# J5 the store's bf16 bias (B16) == its float32 copy (B32), bitwise: bf16 -> fp32 is exact
bh = T.f32_keep_bias(M.batch_of(case, "full"))
check("J5_precondition_bf16", bh["bias"].dtype == torch.bfloat16 and bh["k"].dtype == torch.float32)
a16 = M.make_fv_arm(F, "FVW4S3T64B16", G4, S, 4, "cpu")
b32 = M.make_fv_arm(F, "FVW4S3T64B32", G4, S, 4, "cpu")
o16 = a16(bh)
ob = b32(bh)
check("J5_dtypes", a16._bias.dtype == torch.bfloat16 and b32._bias.dtype == torch.float32)
check("J5_b16_equals_b32", M.bits_equal(o16, ob))
check("J5_b16_vs_fp64", err(o16, bh, rows, G4) <= TOL, err(o16, bh, rows, G4))

# J6 FVN: finite on the clean store, and NOT attention (every row sees every union token)
fvn = M.make_fv_arm(F, "FVN", G4, S, 4, "cpu")
on = fvn(bf)
check("J6_fvn_kw", fvn.launch_kw == dict(use_extents=False), fvn.launch_kw)
check("J6_fvn_finite", bool(torch.isfinite(on).all()))
check("J6_fvn_not_attention", err(on, bf, rows, G4) > M.TOL_C, err(on, bf, rows, G4))

# J7 the 3D path at tile 32: 4 segments over the union ENTRIES == the 2D tile-32 output (1e-5); vs fp64
kv, vv, bv, sr = views(bf, S)
a32.prepare(bf)
so = torch.zeros((8, G4["Hq"], 4, G4["D"]), dtype=torch.float32)
sm = torch.zeros((8, G4["Hq"], 4), dtype=torch.float32)
se = torch.zeros((8, G4["Hq"], 4), dtype=torch.float32)
o7 = torch.empty_like(bf["q"])
F.unified_attention(bf["q"], kv, vv, o7, a32.bufs["cu"], 4, a32.bufs["seqused"], sr, G4["D"] ** -0.5, True,
                    (-1, -1), a32.bufs["table"], 0, None, None, None, seq_threshold_3D=1000,
                    num_par_softmax_segments=4, softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
                    union_ext=a32.bufs["ext"], union_len=a32.bufs["ulen"], cis_bias=bv, build_union=build_args(bf, S),
                    tile_size=32)
d7 = float((o7.double() - o32.double()).abs().max())
check("J7_3d_t32_vs_2d", d7 <= 1e-5, d7)
check("J7_3d_t32_vs_fp64", err(o7, bf, rows, G4) <= TOL, err(o7, bf, rows, G4))
check("J7_3d_wrote_four_segments", bool((se.abs().sum(dim=(0, 1)) > 0).all()), se.abs().sum(dim=(0, 1)).tolist())

# J8 ragged requests (1, 3 and 4 rows) at tile 32 through the launcher (the in-kernel build)
cr = M.make_case(G4, 3, 4, 13)
br = M.select(T4.f32(M.batch_of(cr, "full")), [0, 4, 5, 6, 8, 9, 10, 11])
kv8, vv8, bv8, sr8 = views(br, cr["S"])
tab = torch.zeros((2, 3, M.fv_width(64)), dtype=torch.int32)
ext = torch.zeros_like(tab)
ul = torch.zeros((2, 3), dtype=torch.int32)
o8 = torch.empty_like(br["q"])
F.unified_attention(br["q"], kv8, vv8, o8, torch.tensor([0, 1, 4, 8], dtype=torch.int32), 4,
                    torch.zeros(3, dtype=torch.int32), sr8, G4["D"] ** -0.5, True, (-1, -1), tab, 0, None, None, None,
                    union_ext=ext, union_len=ul, cis_bias=bv8, build_union=build_args(br, cr["S"]), tile_size=32)
check("J8_ragged_t32_vs_fp64", err(o8, br, list(range(8)), G4) <= TOL, err(o8, br, list(range(8)), G4))

# J9 a 16-token tile
a16t = M.make_fv_arm(F, "FVW4S3T16B32", G4, S, 4, "cpu")
o9 = a16t(bf)
check("J9_t16_vs_fp64", err(o9, bf, rows, G4) <= TOL, err(o9, bf, rows, G4))
op9 = a16t(bp)
check("J9_t16_poison_finite_and_equal", bool(torch.isfinite(op9).all()) and M.bits_equal(op9, o9))

# J10 / J11 MUTATION controls (lessons L34): the fork without the V reach mask, and without the score mask, is
# finite on the clean store and NOT finite on the poisoned one, so FI5 rests on those two lines (K and the bias load
# whole tiles, so the poisoned tokens past the reach ARE loaded)
src = open(sys.argv[2]).read()
v_line = "            mask=dim_mask[None, :] & tile_mask[:, None],\n"
s_line = '            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")\n'
check("J10_precondition_one_v_mask", src.count(v_line) == 1, src.count(v_line))
check("J11_precondition_one_s_mask", src.count(s_line) == 1, src.count(s_line))
for jn, old, new in (("J10_no_v_reach_mask", v_line, "            mask=dim_mask[None, :],\n"),
                     ("J11_no_score_mask", s_line,
                      '            query_mask_1[:, None] & query_mask_0[:, None], S, float("-inf")\n')):
    path = os.path.join(TMP, jn + ".py")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    Fm = M.load_by_path(path, "fork_mutant_" + jn)
    am = M.FVArm(Fm, "FV", G4, S, 4, build=True, device="cpu")
    check(jn + "_clean_finite", bool(torch.isfinite(am(bf)).all()))
    check(jn + "_poison_DETECTED", not bool(torch.isfinite(am(bp)).all()), "the mutant stayed finite")

# J12 the launcher refuses a tile below 16, above the block or not dividing it, and a non-float bias
fv.prepare(bf)
kv, vv, bv, sr = views(bf, S)
base_kw = dict(q=bf["q"], k=kv, v=vv, out=torch.empty_like(bf["q"]), cu_seqlens_q=fv.bufs["cu"], max_seqlen_q=4,
               seqused_k=fv.bufs["seqused"], max_seqlen_k=sr, softmax_scale=0.25, causal=True, window_size=(-1, -1),
               block_table=fv.bufs["table"], softcap=0, q_descale=None, k_descale=None, v_descale=None,
               union_ext=fv.bufs["ext"], union_len=fv.bufs["ulen"], cis_bias=bv, build_union=build_args(bf, S))
for name, change in (("tile8", dict(tile_size=8)), ("tile48", dict(tile_size=48)), ("tile128", dict(tile_size=128)),
                     ("int_bias", dict(cis_bias=bv.to(torch.int32)))):
    kw = dict(base_kw)
    kw.update(change)
    try:
        F.unified_attention(**kw)
        check("J12_refuses_" + name, False, "no refusal")
    except AssertionError:
        check("J12_refuses_" + name, True)

# J14 every swept configuration runs the same gates, with the REAL kernel: run_fv on FV, two swept arms, FVN, FVT
names = ["FV", "FVW4S3T32B32", "FVW8S4T64B16", "FVN", "FVT"]
cf = T.case_f32(M.make_case(G4, 2, 4, 11))
bf2 = M.batch_of(cf, "full")
ref2 = M.reference_fp64(bf2, rows, G4)
ref02 = M.reference_fp64(bf2, rows, G4, zero_bias=True)
pois2, _ = M.poison_caches(cf, bf2)
arms = {n: M.make_fv_arm(F, n, G4, S, 4, "cpu") for n in names}
gates, outs, rep = [], {}, dict(arms={})
M.run_fv(G4, cf, arms, outs, bf2, rows, ref2, ref02, pois2, 4, "2x4", True, F, gates, rep)
base_set = T.gate_set(gates, "FV")
for x in ("FVW4S3T32B32", "FVW8S4T64B16"):
    check("J14_same_gates_" + x, T.gate_set(gates, x) == base_set, (T.gate_set(gates, x), base_set))
badm = [(g["name"], g["detail"][:160]) for g in gates if g["must"] and not g["passed"]]
check("J14_all_must_pass", not badm, badm)
check("J14_complete", M.completeness_gates(gates, [(2, 4)], names, True, fv=True)[0]["passed"])
xr = [g for g in gates if g["name"].startswith("X_row_alone_")]
check("J14_row_alone_exact_each", len(xr) == 3 and all(g["passed"] for g in xr), [g["detail"] for g in xr])
gn = [g for g in gates if g["name"] == "R_FVN_not_attention_2x4"]
check("J14_fvn_detected", len(gn) == 1 and gn[0]["status"] == "DETECTED", gn)

print("RAN %d: %s" % (len(ran), " ".join(ran)))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("FIA5-INTERP OK")
'''


def test_F5_6_the_fork_under_the_triton_interpreter(tmp_path):
    """F5_6 (needs triton; skipped without it): the fork on CPU float32 under TRITON_INTERPRET=1 (a fresh process):
    J1 tile 64 and tile 32 = fp64 (1e-4), the tables; J2 FI1-FI3 and FI5 at tile 32, the poison plant has teeth (union
    entries reached in part), FI5 at tile 64; J3 a row alone at tile 32, bitwise; J4 the mask and bias witnesses at
    tile 32; J5 the bf16 bias == its float32 copy, bitwise; J6 FVN finite and not attention; J7 the 3D path at tile 32
    (4 segments) = the 2D output; J8 ragged requests at tile 32; J9 a 16-token tile (and FI5); J10 / J11 the two
    MUTATION controls (no V reach mask / no score mask: the poisoned batch is NOT finite); J12 launcher refusals; J13
    no compile handle under the interpreter; J14 run_fv on FV, two swept arms, FVN and FVT with the real kernel: the
    same gates per swept arm, every MUST gate passing, COMPLETE."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD5, str(EFF / "r2_fiattn_bench.py"), str(FORK),
                        str(Path(__file__).resolve()), str(tmp_path)], env=env, cwd=str(ROOT), capture_output=True,
                       text=True, timeout=3000)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["FIA5-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# ------------------------------------------------------------------ F5_7 the registered block
def test_F5_7_pred5_in_the_notes():
    """F5_7: docs/NOTES_fia5.md carries a REGISTERED line (the TBD marker or the lead's stamp) and a json block equal
    to PRED5; the bands are the brief's suggestion (0.85 in [0.6, 1.05]) for FV and the best swept arm; the
    thresholds are PRED4's rule; the docs name the changed regions and the test files."""
    doc = (ROOT / "docs/NOTES_fia5.md").read_text()
    stamp = re.search(r"^REGISTERED (<TBD-for-lead-stamp>|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \(lead stamp)", doc, re.M)
    assert stamp, "no REGISTERED line (TBD marker or the lead's stamp) in the notes"
    assert M.PRED5["registered"].startswith("REGISTERED " + stamp.group(1)[:20])
    m = re.search(r"<!-- PRED5-BEGIN -->\s*```json\s*(.*?)```\s*<!-- PRED5-END -->", doc, flags=re.S)
    assert m, "no PRED5 block in the notes"
    assert json.loads(m.group(1)) == json.loads(json.dumps(M.PRED5))
    band = {"216x4": [0.85, 0.6, 1.05], "216x4s1": [0.85, 0.6, 1.05]}
    assert M.PRED5["fv_over_p_device"] == band == M.PRED5["best_swept_over_p_device"]
    assert M.PRED5["thresholds"] == M.PRED4["thresholds"]
    for word in ("0.8 x P", "1 ms", "MUST gates", "best swept"):
        assert word in M.PRED5["decision"]
    for word in ("test_r2_fia5.py", "test_r2_fia4.py", "r2_fia5_step.sh", "FV_SWEEP"):
        assert word in doc, word


# ------------------------------------------------------------------ F5_8 / F5_9 the step
def test_F5_8_step_static_shape_and_pins():
    """F5_8: the fia4 step's shape (one main(); the claim after the existence check with an owner file; the budget fits
    the step; piece 3's blob pin; the fork of THIS tree checked before the launch; the S1 tables' size), with the fia5
    cells (216x4, 216x4s1, 32x4), arms (P, FV, FVT, FVN), the sweep (the brief's num_warps 4 / 8 x num_stages 2 / 3 / 4
    grid and more; it parses), --fv-sweep in the bench argv, and no interpreter word."""
    src = STEP5.read_text()
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
    assert re.search(r"^CELLS=216x4,216x4s1,32x4(\s|$)", src, flags=re.M)
    assert re.search(r"^ARMS=P,FV,FVT,FVN(\s|$)", src, flags=re.M)
    sweep = re.search(r"^FV_SWEEP=(\S+)$", src, flags=re.M).group(1)
    names = M.parse_fv_sweep(sweep)
    for w in (4, 8):
        for s in (2, 3, 4):
            assert M.fv_sweep_name(w, s) in names, (w, s)
    assert "FVW4S3T64B16" in names and any(M.fv_sweep_launch(x)["tile_size"] == 32 for x in names)
    for wd in ('--fv-kernel "$FV_MOD"', '--s1-tables "$S1_COPY"', "--no-strided", '--arms "$ARMS"',
               '--fv-sweep "$FV_SWEEP"', "NOSI=%s" % NOSI5, "fia5_${RUN}_$JOB",
               "SELF=benchmarks/Efficiency/r2_fia5_step.sh"):
        assert wd in src, wd
    for word in ("awk", "perl", "python3 -", "sed ", "<<", "--verify-step", "fia4_${RUN}", "fia4-${RUN}", "R2 fia4"):
        assert word not in src, word


def test_F5_9_step_dryrun_refusals_and_pins(tmp_path):
    """F5_9 (as FV10): usage (2); a missing uuid file, a GPU- prefix, two lines STOP (1); FIATTN_HOLDERS outside
    DRYRUN refused; a wrong --nosi stops at HEAD. On a clean checkout of THIS worktree at HEAD the DRYRUN prints the
    pins and the bench arguments (cells, arms, the sweep)."""
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
        return subprocess.run(["bash", str(STEP5), "--nosi", sha, str(job), "t1"] + list(extra), env=e,
                              capture_output=True, text=True, timeout=300)
    r = run(520, extra=["x"])
    assert r.returncode == 2 and "usage" in r.stderr
    r = run(521)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(522, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(523, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(524, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(525, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip() or ROOT.resolve() != Path(NOSI5).resolve():
        pytest.skip("the full DRYRUN needs a clean checkout of %s itself" % NOSI5)
    r = run(526, uid + "\n", sha=head.stdout.strip())
    assert r.returncode == 0, r.stderr[-400:]
    blob = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:nosi/nosi/fork_vllm_unified_attention.py"],
                          capture_output=True, text=True).stdout.strip()
    assert "fork nosi/nosi/fork_vllm_unified_attention.py blob %s" % blob in r.stdout
    assert "S1 tables %s (439458 B)" % S1_RECORDED in r.stdout
    for wd in ("--mode main", "--cells 216x4\\,216x4s1\\,32x4", "--arms P\\,FV\\,FVT\\,FVN", "--fv-sweep w4s2\\,",
               "--no-strided", "--ncu /opt/nvidia/nsight-compute/2025.1.0/ncu", "--budget-s 2400", "--time=45",
               "--fv-kernel %s/nosi/nosi/fork_vllm_unified_attention.py" % NOSI5,
               "FLASHINFER_WORKSPACE_BASE=/mnt/beegfs/ojerbi/nosa-env/jit/fia5_t1_526/flashinfer",
               "--s1-tables /mnt/beegfs/ojerbi/r2-runs/fiattn/fia5_t1_526.step/skv_tables.pt"):
        assert wd in r.stdout, wd
