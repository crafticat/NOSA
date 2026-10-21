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


def test_U3_selector_builds_the_groups_once_per_tick_token():
    """U3 (under torch.inference_mode, as the production tick; CPU job 2191241: inference tensors have no version
    counter): SharedKV(union='fused', impl='reference') reuses build_groups for the same TICK TOKEN, seq_idx object
    and rows_max (the tick's per-layer calls); it rebuilds for another token, another seq_idx object (same values),
    another rows_max, and on every call without a token. Its output is bitwise piece 3's twin and the torch-union
    selector's."""
    with torch.inference_mode():
        c = T.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
        assert c["seq_idx"].is_inference()                               # the production condition of the cache
        ref = bi.bi_paged_decode_reference(*T.args_of(c))
        fu = sk.SharedKV(bi, impl="reference", union="fused")
        to = sk.SharedKV(bi, impl="reference")
        t1, t2 = object(), object()
        steps = []

        def call(seq, nr, tok):
            args = list(T.args_of(c))
            args[4] = seq
            o = fu.bi_paged_decode(*args, page_key=c["key"], rows_max=nr, group_token=tok)
            steps.append(fu.group_builds)
            return o
        s1, s2 = c["seq_idx"], c["seq_idx"].clone()                        # s2: same values, another object
        o1 = call(s1, 4, t1)
        o2 = call(s1, 4, t1)                                               # the same tick: cached
        call(s1, 4, t2)                                                    # a new tick
        call(s2, 4, t2)                                                    # another seq_idx object
        call(s2, 2, t2)                                                    # another rows_max
        call(s2, 4, None)                                                  # no token: always rebuilt
        call(s2, 4, None)
        assert steps == [1, 1, 2, 3, 4, 5, 6], steps
        assert T.bits_equal(o1, ref) and T.bits_equal(o2, ref)
        assert T.bits_equal(o1, to.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4, group_token=t1))
    with pytest.raises(ValueError):
        sk.SharedKV(bi, union="hash")


@pytest.mark.parametrize("mode", ["split", "whole"])
def test_U4_fused_union_in_the_tick(tmp_path, mode):
    """U4 (production entry point, under torch.inference_mode as the production tick): SG1 on the CPU tiny model
    through the tick with SharedKV(union='fused') driving the tick: every verify row torch.equal to the serial row;
    the groups are built at most once per tick (the tick's ctx is the token: fewer group builds than shared calls,
    not once per layer); the probe with the fused union reports 0 differing rows."""
    RT = T._tick_tests()
    p3 = T._p3_pinned(tmp_path)

    def ops_with(attn):
        return RT.SimpleNamespace(linear=RT.exact_linear, attn=attn, mean=RT.BD.fixed_order_mean)
    with torch.inference_mode():
        toks, logits = RT.serial_ref(RT.make_engine(RT.make_ops(p3)))
        used = sk.SharedKV(p3, mode=mode, impl="reference", union="fused")
        res = RT.tick_check(RT.make_engine(ops_with(used), draft_drop=0.5), toks, logits, forced=True)
        assert used.stats["shared"] > 0 and not res["bad"] and all(res["committed_ok"]) and res["compared"] > 50
        ticks = res["eng"].stats["ticks"]
        assert 0 < used.group_builds <= ticks and used.group_builds < used.stats["shared"], (
            used.group_builds, ticks, used.stats)
        probe = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference", union="fused")
        RT.tick_check(RT.make_engine(ops_with(probe), draft_drop=0.5), toks, logits, forced=True)
        rep = probe.report()
        assert probe.stats["shared"] > 0 and rep["diff_rows"] == 0


def hand_case():
    """A hand case (U7, U5): one request, 2 rows, 1 KV head, P 4, PPS 2. Row 0 lists keys 1, 3, 5, 7 (slots 11, 13,
    15, 17), row 1 keys 3, 4, 7 (slots 13, 14, 17). Distinct values ascending: 1, 3, 4, 5, 7 -> uid 0 .. 4. uidx
    row 0 = 0, 1, 3, 4; row 1 = 1, 2, 4, -1. Split 0 (positions 0, 1): uids {0, 1, 1, 2} -> [0, 3); split 1
    (positions 2, 3): {3, 4, 4} -> [3, 5). Group 1 (no rows): empty windows (0, 0)."""
    i32 = torch.int32
    table = torch.tensor([[[11, 13, 15, 17]], [[13, 14, 17, -1]]], dtype=i32)
    key = torch.tensor([[[1, 3, 5, 7]], [[3, 4, 7, -1]]], dtype=i32)
    npg = torch.tensor([[4], [3]], dtype=i32)
    seq = torch.tensor([0, 0], dtype=i32)
    want = dict(
        grow=torch.tensor([[0, 1], [-1, -1]], dtype=i32),
        ucnt=torch.tensor([[5], [0]], dtype=i32),
        ut=torch.tensor([[[11, 13, 14, 15, 17, -1, -1, -1]], [[-1] * 8]], dtype=i32),
        uidx=torch.tensor([[[[0, 1, 3, 4], [1, 2, 4, -1]]], [[[-1] * 4, [-1] * 4]]], dtype=i32),
        rowpos=torch.tensor([[[[0, 1, -1, 2, 3, -1, -1, -1], [-1, 0, 1, -1, 2, -1, -1, -1]]],
                             [[[-1] * 8, [-1] * 8]]], dtype=i32),
        win_split=torch.tensor([[[[0, 3], [3, 5]]], [[[0, 0], [0, 0]]]], dtype=i32),
        win_whole=torch.tensor([[[[0, 5]]], [[[0, 0]]]], dtype=i32))
    return seq, table, npg, key, want


def test_U7_hand_case_pins_win():
    """U7 (CPU job 2191241: the kernel's 'win' differed in every split case): on the hand case, build_union AND the
    fused twin give exactly the hand-derived grow / ucnt / ut / uidx / rowpos and the split windows [[0, 3], [3, 5]]
    (whole: [[0, 5]]), and empty windows for the group without rows. U5 runs the KERNEL on the same case."""
    seq, table, npg, key, want = hand_case()
    for whole in (False, True):
        for build in (sk.build_union, sk.build_union_fused_reference):
            got = build(seq, table, npg, key, 2, 2, whole)
            for f in ("grow", "ucnt", "ut", "uidx", "rowpos"):
                assert torch.equal(got[f], want[f]), (build.__name__, whole, f, got[f].tolist())
            w = want["win_whole" if whole else "win_split"]
            assert torch.equal(got["win"], w), (build.__name__, whole, got["win"].tolist())


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
if not (t.bits_equal(o1, ref) and t.bits_equal(o2, ref)) or fu.group_builds != 2:   # no token: one build per call
    fails.append(("selector-fused", fu.group_builds))
tok = object()
fu2 = sk.SharedKV(bi, union="fused")
fu2.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4, group_token=tok)
o3 = fu2.bi_paged_decode(*t.args_of(c), page_key=c["key"], rows_max=4, group_token=tok)
if not t.bits_equal(o3, ref) or fu2.group_builds != 1:
    fails.append(("selector-fused-token", fu2.group_builds))
# the hand case (U7): the KERNEL's windows pinned
i32 = torch.int32
table = torch.tensor([[[11, 13, 15, 17]], [[13, 14, 17, -1]]], dtype=i32)
key = torch.tensor([[[1, 3, 5, 7]], [[3, 4, 7, -1]]], dtype=i32)
npg = torch.tensor([[4], [3]], dtype=i32)
seq = torch.tensor([0, 0], dtype=i32)
for whole, want in ((False, [[[[0, 3], [3, 5]]], [[[0, 0], [0, 0]]]]), (True, [[[[0, 5]]], [[[0, 0]]]])):
    for te in (2, 16):
        got = sk.build_union_fused(seq, table, npg, key, 2, 2, whole, te=te)
        if got["win"].tolist() != want or got["ucnt"].tolist() != [[5], [0]] \
                or got["uidx"].tolist() != [[[[0, 1, 3, 4], [1, 2, 4, -1]]], [[[-1] * 4, [-1] * 4]]]:
            fails.append(("hand", whole, te, got["win"].tolist(), got["ucnt"].tolist(), got["uidx"].tolist()))
print("RAN", ran)
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("UNION-INTERP OK")
'''


def test_U5_kernel_under_the_triton_interpreter():
    """U5: _union_kernel (build_union_fused) equals build_union in every field on every case, both modes, rows_max 4
    and 2, tile sizes 16 and 64; the DEDUP plant is caught after its precondition; SharedKV(union='fused') with the
    kernels returns piece 3's kernel bits, builds the groups on every call without a token and once per token; the
    hand case of U7 (tile sizes 2 and 16) gives exactly the hand windows, ucnt and uidx."""
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
    assert ast.get_source_segment(src, kern).count("tl.debug_barrier()") == 3      # pass 0 | A | B | C (split)
    assert "_version" not in src.split("class SharedKV:")[1].split("class SharedKVProbe")[0]   # job 2191241
