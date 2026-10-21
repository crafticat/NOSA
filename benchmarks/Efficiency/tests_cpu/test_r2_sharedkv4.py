"""[sharedkv4] CPU tests of the fixes after the review of 9a501d8 (kstep-cpu-runs/paper-notes/review-sharedkv3-9a501d8.md;
docs/R2_SHAREDKV4_FIXES.md): the probe's fused-union equality check (MAJOR-1) and its negative control, the build
counters (MINOR-5), the interpreter repro of a360d86's window reduction (MINOR-1, a DIAGNOSTIC node), one bf16 case
of the shared kernel under the interpreter (MINOR-11), the supplied-union check (MINOR-9), the early exit of empty
groups (MINOR-2, source), the claim hygiene of both step scripts (MINOR-7, source). SOURCE ONLY; the lead runs them:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_sharedkv4.py
"""
from __future__ import annotations

import ast
import importlib.util
import os
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
sys.path.insert(0, str(HERE))
import test_bi_paged_attention_shared as T  # noqa: E402

sk, bi = T.sk, T.bi


class WideWindows(sk.SharedKVProbe):
    """NEGATIVE CONTROL (MAJOR-1): the selector's fused union with every split window widened to the whole union
    (0, ucnt). Every row still runs only its own split's pages (the others are identity steps), so the OUTPUT BITS do
    not change: the bitwise gate is blind to it, the union comparison is not."""

    def _build(self, args, page_key, nr, pps):
        un = super()._build(args, page_key, nr, pps)
        w = un["win"].clone()
        w[..., 0] = 0
        w[..., 1] = un["ucnt"].unsqueeze(-1).expand_as(w[..., 1])
        return dict(un, win=w.contiguous())


def test_F1_probe_checks_the_fused_union_against_build_union():
    """MAJOR-1 / MINOR-5 (CPU twin path): the fused probe compares its union with build_union on every keyed call
    (union_checked = shared calls, every field 0 differing elements) and counts its builds as fused only; the torch
    probe compares nothing and counts torch builds. NEGATIVE CONTROL: widened windows. PRECONDITION (L34): the
    original split windows are not already the whole union somewhere (else the plant is a no-op). Then: the output
    bits stay equal (diff_rows 0: the bitwise gate is BLIND) and the union check counts differing 'win' elements."""
    c = T.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2])
    fu = sk.SharedKVProbe(bi, timing=False, impl="reference", union="fused")
    for _ in range(2):
        fu.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    rep = fu.report()
    assert rep["union_checked"] == 2 and all(v == 0 for v in rep["union_diff"].values()), rep["union_diff"]
    assert rep["union_builds"] == {"torch": 0, "fused": 2} and rep["diff_rows"] == 0
    to = sk.SharedKVProbe(bi, timing=False, impl="reference")
    to.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    rt = to.report()
    assert rt["union_checked"] == 0 and rt["union_builds"] == {"torch": 1, "fused": 0}
    ref = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, T.PPS, False)
    whole_like = (ref["win"][..., 0] == 0) & (ref["win"][..., 1] == ref["ucnt"].unsqueeze(-1))
    assert not bool(whole_like.all()), "PRECONDITION: the windows are already the whole union (a NO-OP plant)"
    wide = WideWindows(bi, timing=False, impl="reference", union="fused")
    wide.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    rw = wide.report()
    assert rw["diff_rows"] == 0, "the widened windows changed the bits: the plant is not the blind case it models"
    assert rw["union_diff"]["win"] > 0 and all(v == 0 for f, v in rw["union_diff"].items() if f != "win"), rw


def test_F2_tables_keep_the_fused_union():
    """MAJOR-1: a recorded call keeps its fused union (host copies in report_tables), next to its inputs."""
    c = T.make_group_case(1, "same", rows_per_req=[4, 4, 2])
    fu = sk.SharedKVProbe(bi, timing=False, impl="reference", union="fused", record_tables=1)
    fu.bi_paged_decode(*T.args_of(c), page_key=c["key"], rows_max=4)
    tabs = fu.report_tables()
    ref = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, T.PPS, False)
    assert len(tabs) == 1 and set(tabs[0]["union"]) == set(sk.UNION_FIELDS)
    assert all(torch.equal(tabs[0]["union"][f], ref[f]) for f in sk.UNION_FIELDS)


REPRO_CHILD = r'''
import os, sys, json
assert os.environ.get("TRITON_INTERPRET") == "1"
import torch
import triton
import triton.language as tl


@triton.jit
def v1(UID, OK, IE, LO, HI, N: tl.constexpr, TE: tl.constexpr, SP: tl.constexpr, PPS: tl.constexpr,
       WHOLE: tl.constexpr):
    """a360d86's construct: an [TE, SP] masked min / max carried across a runtime tile loop, inside a constexpr if."""
    ar = tl.arange(0, TE)
    kk = tl.arange(0, SP)
    lo = tl.full([SP], 2147483647, tl.int32)
    hi = tl.full([SP], -1, tl.int32)
    for e0 in range(0, N, TE):
        e = e0 + ar
        m = e < N
        uid = tl.load(UID + e, mask=m, other=0)
        ok = (tl.load(OK + e, mask=m, other=0) != 0) & m
        ie = tl.load(IE + e, mask=m, other=0)
        if not WHOLE:
            sel = ok[:, None] & ((ie // PPS)[:, None] == kk[None, :])
            lo = tl.minimum(lo, tl.min(tl.where(sel, uid[:, None], 2147483647), 0))
            hi = tl.maximum(hi, tl.max(tl.where(sel, uid[:, None], -1), 0))
    tl.store(LO + kk, lo)
    tl.store(HI + kk, hi)


@triton.jit
def v2(UID, OK, IE, LO, HI, N: tl.constexpr, TE: tl.constexpr, SP: tl.constexpr, PPS: tl.constexpr):
    """v1 without the constexpr if."""
    ar = tl.arange(0, TE)
    kk = tl.arange(0, SP)
    lo = tl.full([SP], 2147483647, tl.int32)
    hi = tl.full([SP], -1, tl.int32)
    for e0 in range(0, N, TE):
        e = e0 + ar
        m = e < N
        uid = tl.load(UID + e, mask=m, other=0)
        ok = (tl.load(OK + e, mask=m, other=0) != 0) & m
        ie = tl.load(IE + e, mask=m, other=0)
        sel = ok[:, None] & ((ie // PPS)[:, None] == kk[None, :])
        lo = tl.minimum(lo, tl.min(tl.where(sel, uid[:, None], 2147483647), 0))
        hi = tl.maximum(hi, tl.max(tl.where(sel, uid[:, None], -1), 0))
    tl.store(LO + kk, lo)
    tl.store(HI + kk, hi)


@triton.jit
def v4(X, LO, HI, R: tl.constexpr, C: tl.constexpr):
    """The reduction alone: tl.min / tl.max over axis 0 of an int32 [R, C] tile."""
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    x = tl.load(X + r * C + c)
    tl.store(LO + tl.arange(0, C), tl.min(x, 0))
    tl.store(HI + tl.arange(0, C), tl.max(x, 0))


i32 = torch.int32
uid = torch.tensor([0, 1, 3, 4, 1, 2, 4, 0], dtype=i32)           # the U7 hand case, entries j * P + i (P 4)
ok = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=i32)
ie = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=i32)
want_lo, want_hi = [0, 3], [2, 4]                                  # PPS 2: split 0 uids {0, 1, 1, 2}, split 1 {3, 4, 4}
res = {}
for name, fn, te, kw in (("V1_te2", v1, 2, dict(WHOLE=False)), ("V1_te8", v1, 8, dict(WHOLE=False)),
                         ("V2_te2", v2, 2, {}), ("V3_te8", v2, 8, {})):
    lo = torch.zeros(2, dtype=i32)
    hi = torch.zeros(2, dtype=i32)
    try:
        fn[(1,)](uid, ok, ie, lo, hi, N=8, TE=te, SP=2, PPS=2, **kw)
        res[name] = dict(lo=lo.tolist(), hi=hi.tolist(), ok=lo.tolist() == want_lo and hi.tolist() == want_hi)
    except Exception as e:
        res[name] = dict(error="%s: %s" % (type(e).__name__, str(e)[:200]), ok=False)
x = torch.tensor([[5, 9], [2, 7], [8, 1], [3, 4]], dtype=i32)
lo = torch.zeros(2, dtype=i32)
hi = torch.zeros(2, dtype=i32)
try:
    v4[(1,)](x, lo, hi, R=4, C=2)
    res["V4_reduce"] = dict(lo=lo.tolist(), hi=hi.tolist(), ok=lo.tolist() == [2, 1] and hi.tolist() == [8, 9])
except Exception as e:
    res["V4_reduce"] = dict(error="%s: %s" % (type(e).__name__, str(e)[:200]), ok=False)
if not res["V4_reduce"]["ok"]:
    cause = "the interpreter's int32 tl.min / tl.max reduction itself"
elif res["V3_te8"]["ok"] and not res["V2_te2"]["ok"]:
    cause = "an [SP] min / max carried across a runtime tile loop (one tile is right)"
elif res["V2_te2"]["ok"] and not res["V1_te2"]["ok"]:
    cause = "the constexpr if inside the runtime loop"
elif all(v["ok"] for v in res.values()):
    cause = "NOT reproduced on the hand case: the a360d86 window fault lies elsewhere in its code"
else:
    cause = "mixed: see the variants"
print("REPRO " + json.dumps(dict(cause=cause, variants=res), sort_keys=True))
'''


def test_D1_interpreter_repro_of_the_a360d86_window_reduction():
    """MINOR-1 (a DIAGNOSTIC node: it passes whenever the child runs, and NAMES the cause in its output and a
    warning): a360d86's masked [TE, SP] tl.min / tl.max carried across the tile loop, on the U7 hand case (expected
    lo [0, 3], hi [2, 4]), in variants: V1 the exact construct (TE 2 and 8), V2 without the constexpr if, V3 = V2 in
    one tile, V4 the int32 reduction alone. The REPRO line goes into the run's log (-rA shows passed output)."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", REPRO_CHILD], env=env, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("REPRO ")]
    assert r.returncode == 0 and lines, "child exit %d\n%s\n%s" % (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    print(lines[-1])
    warnings.warn(lines[-1])


BF16_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.dont_write_bytecode = True
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bi = load(sys.argv[1], "bi_interp_b")
sk = load(sys.argv[2], "sk_interp_b")
t = load(sys.argv[3], "sk_tests_b")
fails = []
c = t.make_group_case(2, "half", rows_per_req=[4, 3, 1, 2], dtype=torch.bfloat16)
try:
    ref = bi.bi_paged_decode(*t.args_of(c))
except Exception as e:
    print("BF16-UNSUPPORTED piece 3 kernel: %s: %s" % (type(e).__name__, str(e)[:300]))
    sys.exit(0)
for mode in ("split", "whole"):
    out = sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=4, mode=mode)
    bad = [r for r in range(ref.shape[0]) if not t.bits_equal(out[r], ref[r])]
    if bad:
        fails.append(("bf16-bitwise", mode, bad[:6]))
twin = bi.bi_paged_decode_reference(*t.args_of(c))
if not t.bits_equal(twin, ref):
    print("NOTE bf16: piece 3's kernel and its twin differ under the interpreter (the GPU decides)")
# MINOR-9: a supplied union that is not contiguous is refused
un = sk.build_union(c["seq_idx"], c["table"], c["num_pages"], c["key"], 4, bi.PAGES_PER_SPLIT, False)
U = un["rowpos"].shape[-1]
wide = torch.full(un["rowpos"].shape[:-1] + (U + 3,), -1, dtype=torch.int32)
wide[..., :U] = un["rowpos"]
try:
    sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=4, union=dict(un, rowpos=wide[..., :U]))
    fails.append(("non-contiguous union accepted",))
except ValueError as e:
    if "contiguous int32" not in str(e):
        fails.append(("wrong refusal", str(e)[:200]))
try:
    sk.bi_paged_decode_shared(bi, *t.args_of(c), c["key"], rows_max=4, union=dict(un, win=un["win"].to(torch.int64)))
    fails.append(("int64 union accepted",))
except ValueError:
    pass
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("BF16-INTERP OK")
'''


def test_B1_bf16_shared_kernel_under_the_interpreter():
    """MINOR-11: one bf16 case (rows per request 4, 3, 1, 2; overlap half): the shared KERNEL bitwise equal to piece
    3's KERNEL in both schedules under TRITON_INTERPRET. If the interpreter cannot run piece 3's bf16 kernel, the node
    SKIPS with that reason (the GPU decides, L36). MINOR-9: a non-contiguous or int64 supplied union is refused."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", BF16_CHILD, str(T.BASE_PATH), str(T.SK_PATH),
                        str(HERE / "test_bi_paged_attention_shared.py")],
                       env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=3000)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if last.startswith("BF16-UNSUPPORTED"):
        pytest.skip(last)
    assert r.returncode == 0 and last == "BF16-INTERP OK", (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


def test_S1_union_kernel_exits_early_for_empty_groups():
    """MINOR-2 (source): the kernel's pass 0 writes the empty group's outputs (ut / rowpos / uidx -1, ucnt and win
    0), reads GROW[g, 0], and runs passes A, B, C only under `if r0 >= 0:` (the 3 barriers inside that branch).
    U5 / U7 (test_r2_sharedkv3_union.py) check the empty groups' values under the interpreter."""
    src = T.SK_PATH.read_text()
    tree = ast.parse(src)
    kern = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_union_kernel")
    seg = ast.get_source_segment(src, kern)
    body = seg.split('"""')[2]
    assert "r0 = tl.load(GROW + g * s_gr)" in body and "if r0 >= 0:" in body
    head, live = body.split("if r0 >= 0:", 1)
    assert "tl.debug_barrier()" not in head and live.count("tl.debug_barrier()") == 3
    for w in ("tl.store(UIDX", "tl.store(UCNT", "tl.store(WIN", "tl.store(ROWPOS", "tl.store(UT"):
        assert w in head, w


@pytest.mark.parametrize("script,claim", [("r2_sharedkv_step.sh", "skv_active"),
                                          ("r2_sharedkv_ncu_step.sh", "skv2_active")])
def test_S2_step_claim_hygiene(script, claim):
    """MINOR-7 (source): the OUT / REC / JIT existence check comes BEFORE the claim's mkdir; the claim gets an owner
    file before any launch; on a client timeout (124 / 137) the wrapper waits (bounded) for the holder's steps,
    never freezes an output whose writer may run, and keeps the claim (ok=0); a release removes the owner file."""
    text = (EFF / script).read_text()
    i_exist = text.index('[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ]')
    i_claim = text.index('mkdir "$OUTROOT/%s"' % claim)
    i_owner = text.index('> "$OUTROOT/%s/owner"' % claim)
    i_launch = text.index('"${ARGV[@]}" < /dev/null')
    assert i_exist < i_claim < i_owner < i_launch
    t_block = text[text.index("124|137)"):text.index("esac", text.index("124|137)"))]
    assert "ok=0" in t_block and "nosteps" in t_block and "sleep 10" in t_block and "writer=running" in t_block
    assert '[ "$t" = "$OUT" ] && [ "$writer" = running ] && continue' in text
    assert 'rm -f "$OUTROOT/%s/owner" && rmdir "$OUTROOT/%s"' % (claim, claim) in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-sharedkv4 " in text
