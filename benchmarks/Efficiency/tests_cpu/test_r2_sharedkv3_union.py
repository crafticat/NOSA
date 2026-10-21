"""[sharedkv3] CPU tests of the FUSED union build (nosi/nosi/bi_paged_attention_shared.py: build_groups,
build_union_fused_reference, _union_kernel / build_union_fused, SharedKV(union="fused"); docs/R2_SHAREDKV3_UNION.md):
the same output as build_union, bit for bit (int tensors), on the random cases of test_bi_paged_attention_shared; the
dedup plant caught after its precondition; the groups built once per tick in the production tick. The kernel runs
under TRITON_INTERPRET=1 in a fresh process (skipped without triton; L36: the interpreter is not the GPU compiler).
SOURCE ONLY; the lead runs them on a compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_sharedkv3_union.py
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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import test_bi_paged_attention_shared as T  # noqa: E402  (its cases, its loaded modules)

sk, bi = T.sk, T.bi
FIELDS = ("grow", "ut", "ucnt", "rowpos", "win", "uidx")


def same_union(a, b):
    """The fields that differ (dtype, shape or a value) between two union dicts; [] = bit for bit equal."""
    return [f for f in FIELDS if not (a[f].dtype == b[f].dtype and a[f].shape == b[f].shape and torch.equal(a[f], b[f]))]


def ref_union(c, nr, whole):
    return sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], nr, T.PPS, whole)


@pytest.mark.parametrize("seed,overlap,rpr,ring", T.CASES)
@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.parametrize("nr", [4, 2])
def test_U1_twin_equals_build_union(seed, overlap, rpr, ring, whole, nr):
    """U1: build_groups equals build_union's grow; the fused build's torch twin (the kernel's algorithm: first
    occurrences, uid = count of first occurrences below) equals build_union in every field, with and without the
    groups passed in (the selector's per-tick cache)."""
    c = T.make_group_case(seed, overlap, rows_per_req=rpr, ring=ring)
    ref = ref_union(c, nr, whole)
    grow = sk.build_groups(c["seq_idx"], nr)
    assert grow.dtype == torch.int32 and grow.is_contiguous() and torch.equal(grow, ref["grow"])
    for groups in (None, grow):
        tw = sk.build_union_fused_reference(c["seq_idx"], c["table"], c["num_pages"], c["key"], nr, T.PPS, whole,
                                            groups=groups)
        assert same_union(tw, ref) == [], (seed, whole, nr, same_union(tw, ref))
        assert (tw["rows_max"], tw["whole"], tw["pps"]) == (ref["rows_max"], ref["whole"], ref["pps"])


@pytest.mark.parametrize("seed,overlap,rpr", [(1, "same", [4, 4, 2]), (2, "half", [4, 3, 1, 2]),
                                              (3, "disjoint", [4, 2, 3])])
def test_U2_dedup_plant_is_caught_after_its_precondition(seed, overlap, rpr):
    """U2 (negative control, L34): dedup=False counts every listed entry as a first occurrence. PRECONDITION on the
    case's own data: some group lists a (key, slot) value in two rows (build_union's ucnt < the group's listed
    entries). With it, the plant changes ucnt (caught); without it ('disjoint': no shared value) the plant is a
    NO-OP and the test says so instead of counting it."""
    c = T.make_group_case(seed, overlap, rows_per_req=rpr)
    ref = ref_union(c, 4, False)
    listed = (ref["uidx"] >= 0).sum(dim=(2, 3))                          # (R, Hkv) listed entries per group
    shared = bool((ref["ucnt"] < listed).any())
    planted = sk.build_union_fused_reference(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, T.PPS, False,
                                             dedup=False)
    diff = same_union(planted, ref)
    if shared:
        assert "ucnt" in diff, "the dedup plant was NOT caught although values are shared (BLIND)"
    else:
        assert overlap == "disjoint" and diff == [], "precondition unmet: the plant is a NO-OP here (not counted)"


def test_U3_selector_builds_the_groups_once_per_tick_object():
    """U3: SharedKV(union='fused', impl='reference') reuses build_groups for the same seq_idx OBJECT (the tick's per-
    layer calls), rebuilds for another object or after an in-place change; its output is bitwise piece 3's twin and
    the torch-union selector's. Negative control: a fresh object with the SAME values also rebuilds (identity, not
    content, is the key: no stale cache can survive an address reuse)."""
    c = T.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    ref = bi.bi_paged_decode_reference(*T.args_of(c))
    fu = sk.SharedKV(bi, impl="reference", union="fused")
    to = sk.SharedKV(bi, impl="reference")
    o1 = fu.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    o2 = fu.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    assert fu.group_builds == 1
    assert T.bits_equal(o1, ref) and T.bits_equal(o2, ref)
    assert T.bits_equal(o1, to.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4))
    args = list(T.args_of(c))
    args[4] = c["seq_idx"].clone()                                       # same values, another object
    fu.bi_paged_decode(*args, page_key=c["key"], rows_max=4)
    assert fu.group_builds == 2
    args[4].add_(0)                                                      # in place: the version counter moves
    fu.bi_paged_decode(*args, page_key=c["key"], rows_max=4)
    assert fu.group_builds == 3
    fu.bi_paged_decode(*args, page_key=c["key"], rows_max=2)             # another rows_max
    assert fu.group_builds == 4
    with pytest.raises(ValueError):
        sk.SharedKV(bi, union="hash")


@pytest.mark.parametrize("mode", ["split", "whole"])
def test_U4_fused_union_in_the_tick(tmp_path, mode):
    """U4 (production entry point): SG1 on the CPU tiny model through the tick with SharedKV(union='fused') driving
    the tick: every verify row torch.equal to the serial row; the groups are built at most once per tick (fewer
    group builds than shared calls: not once per layer); the probe with the fused union reports 0 differing rows."""
    RT = T._tick_tests()
    p3 = T._p3_pinned(tmp_path)
    toks, logits = RT.serial_ref(RT.make_engine(RT.make_ops(p3)))

    def ops_with(attn):
        return RT.SimpleNamespace(linear=RT.exact_linear, attn=attn, mean=RT.BD.fixed_order_mean)
    used = sk.SharedKV(p3, mode=mode, impl="reference", union="fused")
    res = RT.tick_check(RT.make_engine(ops_with(used), draft_drop=0.5), toks, logits, forced=True)
    assert used.stats["shared"] > 0 and not res["bad"] and all(res["committed_ok"]) and res["compared"] > 50
    ticks = res["eng"].stats["ticks"]
    assert 0 < used.group_builds <= ticks and used.group_builds < used.stats["shared"], (used.group_builds, ticks,
                                                                                         used.stats)
    probe = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference", union="fused")
    RT.tick_check(RT.make_engine(ops_with(probe), draft_drop=0.5), toks, logits, forced=True)
    rep = probe.report()
    assert probe.stats["shared"] > 0 and rep["diff_rows"] == 0


UNION_CHILD = r'''
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

bi = load(sys.argv[1], "bi_interp_u")
sk = load(sys.argv[2], "sk_interp_u")
t = load(sys.argv[3], "sk_tests_u")
FIELDS = ("grow", "ut", "ucnt", "rowpos", "win", "uidx")
fails = []
ran = 0
for seed, overlap, rpr, ring in t.CASES:
    c = t.make_group_case(seed, overlap, rows_per_req=rpr, ring=ring)
    for whole in (False, True):
        for nr in (4, 2):
            ref = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], nr, t.PPS, whole)
            for te in (16, 64):                                  # 16: several tiles per pass (the loops are real)
                got = sk.build_union_fused(c["seq_idx"], c["table"], c["num_pages"], c["key"], nr, t.PPS, whole, te=te)
                ran += 1
                bad = [f for f in FIELDS if not (got[f].dtype == ref[f].dtype and got[f].shape == ref[f].shape
                                                 and torch.equal(got[f], ref[f]))]
                if bad:
                    fails.append(("union", seed, whole, nr, te, bad))
# NEGATIVE CONTROL on the KERNEL: DEDUP off; precondition: a value shared by two rows of a group
c = t.make_group_case(1, "same", rows_per_req=[4, 4, 2])
ref = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, t.PPS, False)
listed = (ref["uidx"] >= 0).sum(dim=(2, 3))
if not bool((ref["ucnt"] < listed).any()):
    fails.append(("plant-precondition-unmet",))
else:
    pl = sk.build_union_fused(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, t.PPS, False, dedup=False)
    if torch.equal(pl["ucnt"], ref["ucnt"]):
        fails.append(("dedup-plant-blind",))
# the selector with the fused union: kernel output = piece 3's kernel, groups once for the same seq_idx object
c = t.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
ref = bi.bi_paged_decode(*t.args_of(c))
fu = sk.SharedKV(bi, union="fused")
o1 = fu.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4)
o2 = fu.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4)
if not (t.bits_equal(o1, ref) and t.bits_equal(o2, ref)) or fu.group_builds != 1:
    fails.append(("selector-fused", fu.group_builds))
print("RAN", ran)
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("UNION-INTERP OK")
'''


def test_U5_kernel_under_the_triton_interpreter():
    """U5: _union_kernel (build_union_fused) equals build_union in every field on every case, both modes, rows_max 4
    and 2, tile sizes 16 and 64; the DEDUP plant is caught after its precondition; SharedKV(union='fused') with the
    kernels returns piece 3's kernel bits and builds the groups once for one seq_idx object."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", UNION_CHILD, str(T.BASE_PATH), str(T.SK_PATH),
                        str(HERE / "test_bi_paged_attention_shared.py")],
                       env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["UNION-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


def test_U6_fused_path_source():
    """U6 (source): the fused launcher makes ONE kernel launch and allocates with torch.empty only (no torch op that
    launches a kernel: zeros / full / sort / cumsum / scatter); the selector's fused build never calls build_union;
    the kernel uses barriers between its passes."""
    src = T.SK_PATH.read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_union_fused")
    code = "\n".join(l.split("#")[0] for l in ast.get_source_segment(src, fn).split('"""')[2].splitlines())
    assert code.count("_union_kernel[") == 1
    for bad in ("torch.zeros(", "torch.full(", "torch.sort(", ".cumsum(", ".scatter_(", ".item(", ".tolist(", ".cpu("):
        assert bad not in code, bad
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedKV")
    b = ast.get_source_segment(src, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build"))
    fused_part = b.split('if self.union == "torch":')[1].split("return build_union(", 1)[1]
    assert "build_union(" not in fused_part
    kern = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_union_kernel")
    assert ast.get_source_segment(src, kern).count("tl.debug_barrier()") == 2
