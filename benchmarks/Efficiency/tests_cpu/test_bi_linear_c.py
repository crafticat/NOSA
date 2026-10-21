"""CPU tests of G-c (nosi/nosi/bi_linear_c.py; design note /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/
bi-gemm-design.md section 5.1), its wiring (r2_bi_decode.BIOps --gemm bi_c_strict / bi_c_bucket, the tick's linear
path), its cells (bigc_gate.py, bigc_tune.py, r2_bi_microgate.py --gemm-scan --bigemm-c) and the step script cells
(r2_bi_step.sh bigc-tune / bigc-gate / bigc-time).

HUMAN DIRECTION (binding, 2026-10-21): the relaxed law is NOT approved. bi_c_strict (one launch per weight shape) is THE
CANDIDATE; bi_c_bucket (launch by M bucket) is a RESEARCH ARM only. G-c is a NEW target reference (its bits are not
G-b's); no test here lowers or replaces an old reference test.

LIMIT (lesson L36, design note 5.1): the Triton CPU interpreter runs tl.dot through numpy, whose float order is not the
tensor-core order, so these tests check the logic (contract, tables, masks, addressing, controls, wiring). Across
DIFFERENT launch configs they use integer-valued inputs (exact in fp32 in any order): they CANNOT test Claim R across
configs. Only the GPU gate can (bigc_gate.py G2 / G2w / G2g, bigc_tune.py cross-launch bits).

Interpreter cases run in a FRESH process with TRITON_INTERPRET=1 (Triton reads the flag at its first import); they are
skipped when triton is not installed.

  BC-C1  contract and table: contract_for / strict_launch_for take no M; CONTRACT = BLOCK_K 64, split 1, ascending;
         every entry's tile dims >= 16; check_table refuses a changed contract, a split, a tile of 8, a short bucket list.
  BC-C2  bucket map: launch_for total for M 1 .. 4 x 287; 16/17, 64/65, 256/257, 512/513 change bucket; strict never.
  BC-P   the pinned table: source TABLE == the in-tree copy == bigc-tune 2191262's table.json (bytes), one digest =
         TABLE_SHA256 = the step's BIGC_TABLE_SHA; a changed entry changes the digest and is refused.
  BC-C3  semantics (interpreter): both variants vs float64, the five NOSA-8B shapes / 64, a 3-D input.
  BC-C4  M invariance (interpreter): integer inputs, M in {1, 7, 16, 17, 64, 65, 256, 257, 287} (every bucket), strict ==
         bucket; random fp32 strict for EVERY M 1 .. 80 (one config: valid on the interpreter), its precondition first
         (split 2 changes the bits of these random rows, else the random check would be blind).
  BC-C5  masks with NaN plants (interpreter): x a strided view of a NaN-padded buffer (rows >= M and a K tail), out a
         view of a sentinel buffer: no NaN, the sentinel rows untouched, equal to the contiguous call.
  BC-C6  tile position (interpreter): offsets 1, 63, 129 and a permutation, integer inputs, both variants.
  BC-C7  NEGATIVE CONTROL split-K 2 (interpreter): the fp32 precondition by hand; the kernel-to-kernel precondition on the
         planted rows (contract 0, split 2 gives 1); the M-dependent split is DETECTED by the row checker; the twin agrees.
  BC-C8  NEGATIVE CONTROL K reversed: as BC-C7.
  BC-C9  twin == interpreter on integer inputs and on the planted rows (contract, split 2, reversed).
  BC-C10 graph safety (static): no host sync in the launch path; allocations only under `if out is None` / a
         non-unit last stride.
  BC-L36 every @triton.jit body's global names resolve in the module's globals; negative control: the fg5 factory form.
  BC-W1  BIOps: default exact = G-b PaddedLinear (unchanged); bi_c_strict / bi_c_bucket route to G-c, describe() carries
         the variant label and the reference-change note; EXACT_GEMM_MODES has no research arm; refusals.
  BC-W2  the tick's linear path (TickEngine._lin with BIOps; interpreter): G-c rows == PaddedLinear's rows on integer
         inputs at M 1 .. 32 (both exact there), G-c's kernel launched once per call, the exact mode never.
  BC-W3  the tiny tick (test_r2_tick's model and harness; interpreter) with G-c strict as the linear: the teacher-forced
         tick's verify rows equal its own G-c serial reference; negative control: an M-dependent perturbation of the same
         linear is caught; the order controls are NO-OPs at the tiny K (<= 64: one chunk; L34: asserted and reported).
  BC-W4  defaults unchanged: r2_tick_gate / r2_g0bi_gate default --gemm exact; r2_tick_sg5 --modes default exact,fast;
         the gate's EXACT_GEMMS == r2_bi_decode.EXACT_GEMM_MODES; gm1's scan without --bigemm-c has no G-c table.
  BC-M   r2_bi_microgate --gemm-scan --bigemm-c both through main on the CPU (interpreter); bigc_verdicts / gm1_repeat;
         refusals.
  BC-T   bigc_tune helpers: candidates, pick, build_table (tuned, check_table), BUCKET_REPS vs the buckets; load_table
         refuses a wrong digest.
  BC-G   bigc_gate driver: shape, M set, refusals through main.
  BC-S   r2_bi_step.sh cells: refusals, the table pins, own claim, DRYRUN commands (full DRYRUN at a clean checkout of
         the step's ROOT only).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_bi_linear_c.py
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
MOD = NOSI / "bi_linear_c.py"
STEP = EFF / "r2_bi_step.sh"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-bigemmc"
PKG = "r2bigc"


def _load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = _load_path(MOD, "bi_linear_c_under_test")


def _pkg(mod):
    """r2_bi_decode / r2_tick through a stand-in package (as test_r2_tick: nosa_llama is never imported on CPU)."""
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
        stub = types.ModuleType(PKG + ".nosa_llama")
        stub.QK_SELECT = 33
        sys.modules[PKG + ".nosa_llama"] = stub
        pkg.nosa_llama = stub
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        setattr(sys.modules[PKG], mod, m)
    return sys.modules[full]


def ints(seed, M, K, N, lo=-3, hi=4):
    """Integer-valued fp32 inputs: every product and partial sum is exact in fp32, in any order."""
    g = torch.Generator().manual_seed(900 + seed)
    return (torch.randint(lo, hi, (M, K), generator=g).float(), torch.randint(lo, hi, (N, K), generator=g).float())


def rnd(seed, M, K, N):
    g = torch.Generator().manual_seed(1100 + seed)
    return torch.randn(M, K, generator=g), torch.randn(N, K, generator=g) * 0.1


def planted(seed=3, M=300, K=256, N=70):
    x, w = rnd(seed, M, K, N)
    bl.plant_order_rows(x, w, (0, 1, 2, 3), range(8))
    return x, w


def attn_stub():
    return SimpleNamespace(bi_paged_decode=lambda *a, **k: None, build_page_table=lambda *a, **k: None)


# =====================================================================================================================
# the interpreter child
# =====================================================================================================================
INTERP_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton imported before the flag was read"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("bigc_tests_child", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
getattr(t, "case_" + sys.argv[2])()
print("BIGC-INTERP OK " + sys.argv[2])
'''


def _interp(name, timeout=2400):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(Path(__file__).resolve()), name], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["BIGC-INTERP OK " + name], (
        "child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (name, r.returncode, r.stdout[-4000:],
                                                                      r.stderr[-4000:]))


VARIANTS = (("strict", lambda: bl.bi_c_linear_strict), ("bucket", lambda: bl.bi_c_linear_bucket))


def case_C3():
    for name, (N, K) in bl.NOSA8B_SHAPES.items():
        N, K = N // 64, K // 64
        x, w = rnd(1, 40, K, N)
        ref = x.double() @ w.double().t()
        for v, f in VARIANTS:
            y = f()(x, w)
            d = float((y.double() - ref).abs().max())
            assert y.shape == (40, N) and d <= 1e-3 * (1 + float(ref.abs().max())), (name, v, d)
            assert torch.equal(f()(x.view(40, 1, K), w).view(40, N), y), (name, v, "3-D input")


def case_C4():
    N, K = 70, 256
    x, w = ints(4, 300, K, N)
    fulls = {v: f()(x, w) for v, f in VARIANTS}
    assert torch.equal(fulls["strict"], fulls["bucket"])
    assert torch.equal(fulls["strict"], (x.double() @ w.double().t()).float())
    for v, f in VARIANTS:
        for M in (1, 7, 16, 17, 64, 65, 256, 257, 287):
            assert torch.equal(f()(x[:M], w), fulls[v][:M]), (v, M)
    xr, wr = rnd(4, 80, K, N)
    L = bl.strict_launch_for(N, K)
    assert not torch.equal(bl.control_split_k(xr, wr, L), bl.bi_c_linear_strict(xr, wr)), \
        "precondition: split 2 changes the bits of the random rows (else the random check is blind)"
    full = bl.bi_c_linear_strict(xr, wr)
    for M in range(1, 81):
        assert torch.equal(bl.bi_c_linear_strict(xr[:M], wr), full[:M]), M


def case_C5():
    N, K, M, TAIL = 70, 256, 37, 64
    x, w = rnd(5, M, K, N)
    buf = torch.full((M + 5, K + TAIL), float("nan"))
    buf[:M, :K] = x
    xv = buf[:M, :K]
    assert torch.isnan(buf[M:]).all() and torch.isnan(buf[:M, K:]).all() and xv.stride() == (K + TAIL, 1)
    for v, f in VARIANTS:
        obuf = torch.full((M + 3, N), 12345.0)
        out = obuf[:M]
        r = f()(xv, w, out=out)
        assert r.data_ptr() == out.data_ptr(), v
        assert not torch.isnan(out).any(), v
        assert (obuf[M:] == 12345.0).all(), v
        assert torch.equal(out, f()(x.contiguous(), w)), v


def case_C6():
    N, K = 70, 256
    x, w = ints(6, 300, K, N)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(6))
    for v, f in VARIANTS:
        full = f()(x, w)
        for a in (1, 63, 129):
            for M in (7, 65, 100):
                assert torch.equal(f()(x[a:a + M], w), full[a:a + M]), (v, a, M)
        assert torch.equal(f()(x[perm], w), full[perm]), v


def _hand_precondition():
    t = torch.tensor(2.0 ** 24, dtype=torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    assert torch.equal(t + one, t), "2**24 + 1 must round to 2**24 in fp32"
    assert float((t + one) + (-t)) == 0.0                    # the contract's fold
    assert float(t + (one + (-t))) == 1.0                    # split 2's fold
    assert float(((-t) + one) + t) == 1.0                    # K reversed


def _control_case(control, twin_kw):
    _hand_precondition()
    x, w = planted()
    N, K = w.shape
    L = bl.strict_launch_for(N, K)
    clean = bl.bi_c_linear_strict(x[:7], w)
    ctrl = control(x[:7], w, L)
    assert (clean[:4, :8] == 0).all() and (ctrl[:4, :8] == 1).all(), "kernel-to-kernel precondition on the plant"
    tw = bl.bi_c_reference(x[:7], w, **twin_kw)
    assert torch.equal(tw[:4, :8], ctrl[:4, :8]) and (bl.bi_c_reference(x[:7], w)[:4, :8] == 0).all()

    def pick(xx, ww):                                         # the M-dependent control (as a heuristic GEMM does)
        return control(xx, ww, L) if xx.reshape(-1, K).shape[0] > 64 else bl.bi_c_linear_strict(xx, ww)
    seen = [M for M in (1, 7, 33) if not torch.equal(pick(x[:M], w), pick(x, w)[:M])]
    assert seen, "the row checker is BLIND to the M-dependent control"
    assert all(torch.equal(bl.bi_c_linear_strict(x[:M], w), bl.bi_c_linear_strict(x, w)[:M]) for M in (1, 7, 33))


def case_C7():
    _control_case(lambda x, w, L: bl.control_split_k(x, w, L, 2), dict(split_k=2))


def case_C8():
    _control_case(bl.control_k_reversed, dict(reverse_k=True))


def case_C9():
    x, w = ints(9, 50, 256, 70)
    assert torch.equal(bl.bi_c_reference(x, w), bl.bi_c_linear_strict(x, w))
    xp, wp = planted(9, M=20)
    L = bl.strict_launch_for(*wp.shape)
    assert torch.equal(bl.bi_c_reference(xp, wp)[:4], bl.bi_c_linear_strict(xp, wp)[:4])
    assert torch.equal(bl.bi_c_reference(xp, wp, split_k=2)[:4], bl.control_split_k(xp, wp, L, 2)[:4])
    assert torch.equal(bl.bi_c_reference(xp, wp, reverse_k=True)[:4], bl.control_k_reversed(xp, wp, L)[:4])


def case_W2():
    BD = _pkg("r2_bi_decode")
    TK = _pkg("r2_tick")
    ops = {m: BD.BIOps(attn_stub(), 32, gemm=m) for m in ("exact", "bi_c_strict", "bi_c_bucket")}
    BLC = sys.modules[PKG + ".bi_linear_c"]
    real, calls = BLC._launch, []

    def counting(*a, **k):
        calls.append(a[0].shape[0])
        return real(*a, **k)
    BLC._launch = counting
    try:
        N, K = 40, 96
        for M in (1, 4, 7, 16, 17, 32):
            x, w = ints(20 + M, M, K, N)
            outs = {}
            for m, o in ops.items():
                n0 = len(calls)
                eng = SimpleNamespace(ops=o, _span=lambda cat: contextlib.nullcontext())
                outs[m] = TK.TickEngine._lin(eng, x.view(M, 1, K), w)
                assert len(calls) - n0 == (0 if m == "exact" else 1), (m, M, calls[n0:])
            assert torch.equal(outs["bi_c_strict"], outs["exact"]) and torch.equal(outs["bi_c_bucket"], outs["exact"]), M
    finally:
        BLC._launch = real


def case_W3():
    TT = _load_path(EFF / "tests_cpu/test_r2_tick.py", "test_r2_tick_for_bigc")
    p = TT.PIECE3_V2
    body = subprocess.run(["git", "-C", str(ROOT), "show", p["blob"]], capture_output=True)
    assert body.returncode == 0 and body.stdout
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "piece3_v2.py"
        path.write_bytes(body.stdout)
        p3 = _load_path(path, "piece3_v2_bigc")
    lin = TT.BD.BIOps(attn_stub(), 32, gemm="bi_c_strict").linear
    BLC = sys.modules["r2tick.bi_linear_c"]
    assert lin is BLC.bi_c_linear_strict
    ops = TT.make_ops(p3, lin)
    toks, logits = TT.serial_ref(TT.make_engine(ops))
    res = TT.tick_check(TT.make_engine(ops, draft_drop=0.5), toks, logits, forced=True)
    assert res["compared"] > 50 and not res["wrong_in"] and not res["ctrl"], res["compared"]
    assert not res["bad"], res["bad"][:5]
    assert all(res["committed_ok"]) and not res["mirror_bad"]
    # L34: at the tiny K (<= 64 = one chunk) the order controls change nothing: REPORTED, never counted as caught
    x, w = rnd(30, 9, 32, 128)
    L = BLC.strict_launch_for(128, 32)
    assert torch.equal(BLC.control_split_k(x, w, L, 2), lin(x, w)) and torch.equal(BLC.control_k_reversed(x, w, L),
                                                                                    lin(x, w))
    print("BC-W3 REPORT: split-K 2 and K-reversed controls are NO-OPs at the tiny model's K (one chunk)")
    seen_m = []

    def mdep(xx, ww):                                        # NEGATIVE CONTROL: G-c with an M-dependent last bit
        M = xx.reshape(-1, xx.shape[-1]).shape[0]
        seen_m.append(M)
        y = lin(xx, ww)
        return y + 1e-3 if M > len(TT.S_TINY) else y
    ops_n = TT.make_ops(p3, mdep)
    caught = False
    try:
        r = TT.tick_check(TT.make_engine(ops_n, draft_drop=0.5), toks, logits, forced=True)
        caught = bool(r["bad"]) or not all(r["committed_ok"]) or bool(r["mirror_bad"])
    except Exception:                                        # noqa: BLE001 (a refusal is a detection)
        caught = True
    assert any(m > len(TT.S_TINY) for m in seen_m), "precondition: the tick called the linear with more rows than B"
    assert caught, "the M-dependent G-c perturbation was not caught"


def case_M():
    sys.path.insert(0, str(EFF))
    if "nosi" not in sys.modules:
        pkg = types.ModuleType("nosi")
        pkg.__path__ = [str(NOSI)]
        sys.modules["nosi"] = pkg
    mg = _load_path(EFF / "r2_bi_microgate.py", "r2_bi_microgate_bigc")
    d = tempfile.mkdtemp()
    out = os.path.join(d, "gc")
    args = ["--out", out, "--gemm-scan", "--device", "cpu", "--shape-div", "64", "--ms", "8,16", "--exact-b", "4",
            "--U", "3", "--warm", "0", "--reps", "1", "--bigemm-c", "both"]
    assert mg.main(args) == 0
    s = json.loads(Path(out, "summary.json").read_text())
    g = {x["name"]: x for x in s["gates"]}
    assert g["GM-RAN"]["passed"] and g["GC-RAN"]["passed"] and g["GC-RAN"]["must"], g
    assert "GC-NAT-REPEATS-GM1" not in g and not g["R_GC-NAT-REPEATS-GM1"]["must"]
    nat, _ = mg.scan_plan([8, 16], [4], 3)
    assert len(s["tables"]["bigc"]) == 5 * len(nat) * 2
    assert sorted(s["tables"]["per_step_bigc"]) == ["bucket", "strict"]
    assert s["meta"]["cell"] == "bigc-time" and s["meta"]["bigc"]["cpu_fp32_upcast"]
    assert "CANDIDATE" in s["meta"]["bigc"]["describe"]["strict"]["label"]
    assert "RESEARCH" in s["meta"]["bigc"]["describe"]["bucket"]["label"]
    names = [p["name"] for p in s["predictions"]]
    assert "GC-STRICT-RATIO-8" in names and "GC-BUCKET-RESEARCH-RATIO-8" in names


# =====================================================================================================================
# BC-C1 / C2: the contract and the tables
# =====================================================================================================================
def test_BC_C1_contract_and_table():
    assert list(inspect.signature(bl.contract_for).parameters) == ["N", "K"]
    assert list(inspect.signature(bl.strict_launch_for).parameters) == ["N", "K", "table"]
    assert bl.CONTRACT.block_k == 64 and bl.CONTRACT.split_k == 1 and bl.CONTRACT.k_order == "ascending"
    for N, K in bl.NOSA8B_SHAPES.values():
        assert bl.contract_for(N, K) is bl.CONTRACT
    with pytest.raises(ValueError):
        bl.contract_for(0, 5)
    for t in (bl.check_table(bl.TABLE), bl.check_table(bl.UNTUNED_PRIOR)):
        assert t["contract"] == list(bl.CONTRACT)
        entries = [e for lst in list(t["shapes"].values()) + [t["default"]] for e in lst] + list(t["strict"].values())
        assert entries and all(min(e[0], e[1]) >= 16 for e in entries)
    assert bl.TABLE["tuned"] is True and bl.UNTUNED_PRIOR["tuned"] is False
    assert bl.NOSA8B_SHAPES == {"wqkv": (4608, 4096), "wo": (4096, 4096), "gate_up": (32768, 4096),
                                "down": (4096, 16384), "lm_head": (73448, 4096)}
    bad = []
    for mut in (lambda x: x.__setitem__("contract", [32] + x["contract"][1:]),
                lambda x: x.__setitem__("contract", [64, 2] + x["contract"][2:]),
                lambda x: x["strict"].__setitem__("4096x4096", [8, 128, 4, 4, 8]),
                lambda x: x["default"].__setitem__(0, [16, 32, 3, 4, 8]),
                lambda x: x.__setitem__("buckets", [16, 64, 256]),
                lambda x: x.__setitem__("tuned", "yes")):
        c = json.loads(json.dumps(bl.TABLE))
        mut(c)
        try:
            bl.check_table(c)
        except ValueError:
            continue
        bad.append(c)
    assert not bad, "check_table accepted a broken table"
    assert len(bl.table_digest(bl.TABLE)) == 64 and bl.table_digest() == bl.table_digest(bl.TABLE)


def test_BC_C2_bucket_map():
    b = bl.TABLE["buckets"]
    assert b == [16, 64, 256, 512]
    for N, K in list(bl.NOSA8B_SHAPES.values()) + [(70, 256)]:
        s = bl.strict_launch_for(N, K)
        for M in range(1, 4 * 287 + 1):
            assert isinstance(bl.launch_for(N, K, M), bl.Launch)
        for edge in b:
            assert bl.bucket_index(edge, b) + 1 == bl.bucket_index(edge + 1, b)
        assert [bl.bucket_index(M, b) for M in (1, 16, 17, 64, 65, 256, 257, 512, 513, 1148)] == \
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        assert all(bl.strict_launch_for(N, K) == s for _ in range(3))
    assert bl.launch_for(4608, 4096, 287) == bl.Launch(*bl.TABLE["shapes"]["4608x4096"][3])
    assert bl.launch_for(7, 9, 1) == bl.Launch(*bl.TABLE["default"][0])


RUN_TABLE = Path("/mnt/beegfs/ojerbi/r2-runs/bi/bi_bgca_2191262_bigc-tune/table.json")


def test_BC_P_pinned_table():
    """BC-P: the frozen table of bigc-tune 2191262 is the source TABLE, its in-tree copy and the step's pins, with one
    digest (computed here, on the compute node). Negative control: one changed entry changes the digest and
    load_table refuses it."""
    f = ROOT / bl.TABLE_FILE
    assert json.loads(f.read_text()) == bl.TABLE
    assert bl.table_digest(bl.TABLE) == bl.TABLE_SHA256 == "cd8684baba2165e380728cfc619d1ecdfca5723f0982c470199162f55022b814"
    assert bl.load_table(str(f), bl.TABLE_SHA256) == bl.TABLE
    if RUN_TABLE.is_file():                                   # the run's own output (frozen 0444)
        assert RUN_TABLE.read_bytes() == f.read_bytes()
    assert bl.TABLE["tuned"] is True and bl.TABLE["bucket_invariant"] is True
    assert bl.strict_launch_for(4608, 4096) == bl.Launch(64, 128, 4, 3, 8)
    assert bl.strict_launch_for(32768, 4096) == bl.Launch(64, 256, 8, 3, 8)
    assert bl.strict_launch_for(73448, 4096) == bl.Launch(64, 256, 4, 3, 8)
    text = STEP.read_text()
    assert "\nBIGC_TABLE=%s/%s " % (STEP_ROOT, bl.TABLE_FILE) in text
    assert "\nBIGC_TABLE_SHA=%s " % bl.TABLE_SHA256 in text
    assert '\n    "tuned": True,\n' in MOD.read_text()            # the step's grep for a frozen source table
    moved = json.loads(json.dumps(bl.TABLE))
    moved["strict"]["4096x4096"] = [64, 64, 4, 3, 8]
    assert bl.table_digest(moved) != bl.TABLE_SHA256
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        p.write_text(json.dumps(moved, sort_keys=True, separators=(",", ":")))
        with pytest.raises(ValueError):
            bl.load_table(str(p), bl.TABLE_SHA256)


# =====================================================================================================================
# BC-C3 .. C9: the kernel under the interpreter (fresh processes)
# =====================================================================================================================
def test_BC_C3_semantics():
    _interp("C3")


def test_BC_C4_m_invariance():
    _interp("C4")


def test_BC_C5_masks_nan_plants():
    _interp("C5")


def test_BC_C6_tile_position():
    _interp("C6")


def test_BC_C7_negative_split_k():
    _interp("C7")


def test_BC_C8_negative_k_reversed():
    _interp("C8")


def test_BC_C9_twin_equals_kernel():
    _interp("C9")


def test_BC_C9b_twin_controls_on_cpu():
    """The twin alone (no Triton): the planted rows give 0 / 1 / 1 for contract / split 2 / reversed; integer rows are
    equal in every order (so the integer checks cannot see an order change: the reason C4 also runs random rows)."""
    x, w = planted(11, M=6)
    assert (bl.bi_c_reference(x, w)[:4, :8] == 0).all()
    assert (bl.bi_c_reference(x, w, split_k=2)[:4, :8] == 1).all()
    assert (bl.bi_c_reference(x, w, reverse_k=True)[:4, :8] == 1).all()
    xi, wi = ints(11, 5, 256, 30)
    assert torch.equal(bl.bi_c_reference(xi, wi), bl.bi_c_reference(xi, wi, split_k=2))
    assert bl.split_bounds(256, 2) == [(0, 128), (128, 256)] and bl.split_bounds(16384, 2)[1][0] == 8192
    with pytest.raises(ValueError):
        bl.plant_order_rows(*rnd(1, 2, 192, 4), [0], [0])           # three chunks: refused (G-a's blind case)


# =====================================================================================================================
# BC-C10 / L36: static
# =====================================================================================================================
def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_BC_C10_graph_safety_static():
    src = MOD.read_text()
    tree = ast.parse(src)
    alloc = {"empty", "zeros", "full", "ones", "empty_like", "zeros_like", "new_empty", "contiguous", "clone"}
    for name in ("_run", "_prepare", "_launch", "bi_c_linear_strict", "bi_c_linear_bucket"):
        fn = _fn(tree, name)
        code = "\n".join(l.split("#")[0] for l in ast.get_source_segment(src, fn).splitlines())
        for bad in (".item(", ".tolist(", ".cpu(", "synchronize", "bool(", "print("):
            assert bad not in code, (name, bad)
        allowed = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.If) and ast.unparse(n.test) in ("out is None", "x2.stride(-1) != 1"):
                allowed |= {id(c) for c in ast.walk(n)}
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in alloc]
        assert all(id(c) in allowed for c in calls), (name, [ast.unparse(c) for c in calls if id(c) not in allowed])
    assert "triton.autotune" not in src and "triton.heuristics" not in src
    assert src.count('@triton.jit(do_not_specialize=["M"])') == 1
    assert src.splitlines()[0].startswith("# Adapted from SGLang") and "Apache License, Version 2.0" in src


def jit_global_misses(src):
    """{jit function: names it loads that are neither its own (arguments, assignments, loop variables) nor module
    globals (top-level imports / defs / assignments, also inside top-level if / try) nor builtins} (lesson L36)."""
    tree = ast.parse(src)
    names = set()

    def collect(body):
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    names.update(x.id for x in ast.walk(tg) if isinstance(x, ast.Name))
            elif isinstance(n, (ast.If, ast.Try)):
                collect(n.body)
                collect(n.orelse)
                for h in getattr(n, "handlers", []):
                    collect(h.body)
                collect(getattr(n, "finalbody", []))
    collect(tree.body)
    misses = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in fn.decorator_list):
            own = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            own |= {x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
            loads = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
            bad = sorted(loads - own - names - set(dir(builtins)))
            if bad:
                misses[fn.name] = bad
    return misses


def test_BC_L36_jit_globals_resolve():
    assert jit_global_misses(MOD.read_text()) == {}
    neg = ("import triton\n\ndef factory():\n    import triton.language as tl\n\n    @triton.jit\n"
           "    def k(X):\n        tl.store(X, 1)\n    return k\n")
    assert jit_global_misses(neg) == {"k": ["tl"]}, "negative control (the fg5 factory form) must be flagged"


# =====================================================================================================================
# BC-W: the wiring
# =====================================================================================================================
def test_BC_W1_biops_modes():
    BD = _pkg("r2_bi_decode")
    SC = sys.modules[PKG + ".r2_spec_core"]
    d = BD.BIOps(attn_stub(), 8)
    assert d.mode == "exact" and d.linear is d.padded and isinstance(d.padded, SC.PaddedLinear) and "G-b" in d.gemm
    assert "gemm_c" not in d.describe()
    s = BD.BIOps(attn_stub(), 8, gemm="bi_c_strict")
    BLC = sys.modules[PKG + ".bi_linear_c"]
    assert s.linear is BLC.bi_c_linear_strict and "STRICT" in s.gemm
    ds = s.describe()["gemm_c"]
    assert ds["variant"] == "strict" and "CANDIDATE" in ds["label"] and "NEW target reference" in ds["reference"]
    assert ds["table_digest"] == BLC.table_digest() and ds["buckets"] is None
    b = BD.BIOps(attn_stub(), 8, gemm="bi_c_bucket")
    assert b.linear is BLC.bi_c_linear_bucket and "RESEARCH" in b.gemm and "RESEARCH" in b.describe()["gemm_c"]["label"]
    assert BD.EXACT_GEMM_MODES == ("exact", "bi_c_strict") and "bi_c_bucket" in BD.GEMM_MODES
    for kw in (dict(gemm="bi_c"), dict(gemm="bi_c_strict", linear=lambda x, w: x), dict(gemm="bi_c_bucket", linear=len)):
        with pytest.raises(Exception):
            BD.BIOps(attn_stub(), 8, **kw)
    with pytest.raises(ValueError):
        BLC.describe("relaxed")


def test_BC_W2_tick_linear_path_equals_padded_rows():
    _interp("W2")


def test_BC_W3_tiny_tick_with_gc_strict():
    _interp("W3")


def test_BC_W4_defaults_unchanged():
    sys.path.insert(0, str(EFF))
    G = _load_path(EFF / "r2_tick_gate.py", "r2_tick_gate_bigc")
    a = G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p"])
    assert a.gemm == "exact"
    assert G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p", "--gemm", "bi_c_strict"]).gemm == "bi_c_strict"
    with pytest.raises(SystemExit):
        G.parse_args(["--out", "x", "--workload", "w", "--biattn", "p", "--gemm", "bi_c"])
    BD = _pkg("r2_bi_decode")
    assert G.EXACT_GEMMS == BD.EXACT_GEMM_MODES and set(G.GEMMS) <= set(BD.GEMM_MODES)
    GB = _load_path(EFF / "r2_g0bi_gate.py", "r2_g0bi_gate_bigc")
    assert GB.parse_args(["--out", "x", "--workload", "w", "--biattn", "p"]).gemm == "exact"
    sg5 = (EFF / "r2_tick_sg5.py").read_text()
    assert 'ap.add_argument("--modes", default="exact,fast")' in sg5
    assert 'MODES = ("exact", "fast", "bi_c_strict", "bi_c_bucket")' in sg5
    mg = _mg()
    d = tempfile.mkdtemp()
    out = os.path.join(d, "gm")
    assert mg.main(["--out", out, "--gemm-scan", "--device", "cpu", "--shape-div", "64", "--ms", "8", "--exact-b", "2",
                    "--U", "1", "--warm", "0", "--reps", "1"]) == 0
    s = json.loads(Path(out, "summary.json").read_text())
    assert s["meta"]["cell"] == "gm1" and s["meta"]["bigc"] is None and "bigc" not in s["tables"]
    assert not any(g["name"].startswith(("GC-", "R_GC-")) for g in s["gates"])


# =====================================================================================================================
# BC-M: the timing column
# =====================================================================================================================
def _mg():
    sys.path.insert(0, str(EFF))
    if "nosi" not in sys.modules:
        pkg = types.ModuleType("nosi")
        pkg.__path__ = [str(NOSI)]
        sys.modules["nosi"] = pkg
    return _load_path(EFF / "r2_bi_microgate.py", "r2_bi_microgate_bigc_parent")


def test_BC_M1_scan_with_gc_columns():
    _interp("M")


def test_BC_M2_verdicts_and_refusals(tmp_path):
    mg = _mg()
    nat = {64: 14.225, 287: 27.689, 861: 72.006}
    ok = {"strict": {64: 18.0, 287: 31.8, 861: 82.8}, "bucket": {64: 30.0, 287: 31.8, 861: 82.8}}
    v = {n: x for n, x, _ in mg.bigc_verdicts(nat, ok)}
    assert v["GC-STRICT-RATIO-287"] == v["GC-STRICT-RATIO-861"] == v["GC-STRICT-RATIO-64"] == "CONFIRMED", v
    assert v["GC-STRICT-DENSE"] == "CONFIRMED" and v["GC-STRICT-FALSIFIER"] == "NOT-FALSIFIED"
    assert v["GC-BUCKET-RESEARCH-RATIO-64"] == "REFUTED" and v["GC-BUCKET-RESEARCH-DENSE"] == "CONFIRMED"
    assert not any(n.startswith("GC-BUCKET-") and "RESEARCH" not in n for n in v), "the research arm is labelled"
    bad = {"strict": {64: 18.0, 287: 50.0, 861: 90.0}}
    v = {n: x for n, x, _ in mg.bigc_verdicts(nat, bad)}
    assert v["GC-STRICT-RATIO-287"] == "REFUTED" and v["GC-STRICT-DENSE"] == "REFUTED"
    assert v["GC-STRICT-FALSIFIER"] == "FALSIFIED"                       # 3 x 50 + 90 = 240 >= 213.904
    assert mg.bigc_verdicts({}, {"strict": {}}) == []
    rep = mg.gm1_repeat({287: 27.689 * 1.04, 861: 72.006 * 1.06, 999: 1.0})
    assert sorted(rep) == [287, 861] and abs(rep[287][2] - 0.04) < 1e-9 and rep[861][2] > mg.GM1_REPEAT_TOL
    assert mg.GM1_NATURAL[287] == 27.689 and mg.GM1_NATURAL[861] == 72.006 and mg.PRED_BIGC_DENSE["falsifier_ms"] == 213.904
    assert mg.main(["--out", str(tmp_path / "a"), "--bigemm-c", "strict"]) == 2
    assert mg.main(["--out", str(tmp_path / "b"), "--gemm-scan", "--bigemm-c", "both", "--bigc-table", "t.json"]) == 2
    assert mg.main(["--out", str(tmp_path / "c"), "--gemm-scan", "--bigc-table", "t", "--bigc-table-sha", "0"]) == 2
    assert not any((tmp_path / n).exists() for n in ("a", "b", "c"))


# =====================================================================================================================
# BC-T / BC-G: the tuning and gate drivers
# =====================================================================================================================
def test_BC_T_tune_helpers(tmp_path):
    T = _load_path(EFF / "bigc_tune.py", "bigc_tune_cpu")
    c = T.candidates(bl)
    assert len(c) == 64 and len(set(c)) == 64
    for e in c:
        bl._check_entry(list(e), "candidate")
    assert len(T.BUCKET_REPS) == len(bl.TABLE["buckets"]) + 1
    assert all(bl.bucket_index(M, bl.TABLE["buckets"]) == i for i, reps in enumerate(T.BUCKET_REPS) for M in reps)
    assert T.STRICT_WEIGHTS == {287: 3.0, 861: 1.0} and {287, 861} <= set(T.TUNE_MS)
    L1, L2, L3 = c[0], c[1], c[2]
    times = {L1: {287: 1.0, 861: 5.0}, L2: {287: 2.0, 861: 1.0}, L3: {287: 0.1}}
    assert T.pick(times, T.STRICT_WEIGHTS) == (L2, 7.0)                  # 3 x 2 + 1 < 3 x 1 + 5; L3 lacks 861
    assert T.pick({}, T.STRICT_WEIGHTS) == (None, None)
    sp = {k: L2 for k in bl.UNTUNED_PRIOR["strict"]}
    bp = {k: [L1] * 5 for k in bl.UNTUNED_PRIOR["shapes"]}
    before = json.dumps(bl.UNTUNED_PRIOR, sort_keys=True)
    t = T.build_table(bl, bl.UNTUNED_PRIOR, sp, bp, True, "test")
    assert t["tuned"] is True and t["bucket_invariant"] is True and t["strict"]["4096x4096"] == list(L2)
    assert json.dumps(bl.UNTUNED_PRIOR, sort_keys=True) == before, "build_table must not change its prior"
    p = tmp_path / "table.json"
    p.write_text(json.dumps(t, sort_keys=True, separators=(",", ":")))
    dg = bl.table_digest(t)
    assert bl.load_table(str(p), dg) == t
    with pytest.raises(ValueError):
        bl.load_table(str(p), "0" * 64)
    assert bl.launch_for(4096, 4096, 300, table=t) == L1 and bl.strict_launch_for(4096, 4096, table=t) == L2


def test_BC_G_gate_driver(tmp_path):
    G = _load_path(EFF / "bigc_gate.py", "bigc_gate_cpu")
    assert G.MS == (1, 7, 16, 17, 64, 65, 256, 257, 287, 861, 1148) and G.ROWS == 1148 and G.GRAPH_MS == (287, 861)
    assert all(a + M <= G.ROWS for a, M in G.WINDOWS) and {a for a, _ in G.WINDOWS} == {1, 63, 129}
    src = (EFF / "bigc_gate.py").read_text()
    assert src.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')
    assert 'print("BIGC_GATE_DONE failed=%d" % nfail, flush=True)' in src
    for s in ("G3a", "G3b", "G3c", "split-K-2", "K-reversed", "natural-cuBLAS", "BLIND", "R5-reference-change"):
        assert s in src, s
    assert G.main(["--out", str(tmp_path / "a"), "--variants", "relaxed"]) == 2
    assert G.main(["--out", str(tmp_path / "b"), "--table", "t.json"]) == 2
    (tmp_path / "c").mkdir()
    assert G.main(["--out", str(tmp_path / "c")]) == 2
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


# =====================================================================================================================
# BC-S: the step script
# =====================================================================================================================
def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


STAND_IN_TABLE = "/mnt/beegfs/ojerbi/models/NOSA-8B/config.json"    # an existing file under the bound root (DRYRUN only)


def test_BC_S_step_cells(tmp_path):
    text = STEP.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?'
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-bigemmc" in text and "<<" not in text
    for word in ("awk", "perl", "python3", " bc ", "sed -i"):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    assert 'mkdir "$OUTROOT/$CLAIM"' in text and "CLAIM=bigc_active" in text and "CLAIM=bi_active" in text
    (tmp_path / "118.gpu_uuid").write_text("f31fc723-9dfe-407a-9947-28173556232a\n")
    e = dict(_env(), BI_HOLDERS=str(tmp_path), DRYRUN="1")

    def run(*extra, sha="0" * 40, env=None):
        return subprocess.run(["bash", str(STEP), "--nosi", sha, "118", "t1", *extra], env=env or e,
                              capture_output=True, text=True, timeout=120)
    r = run("bigc-relaxed")
    assert r.returncode == 1 and "cell 'bigc-relaxed'" in r.stderr
    r = run("bigc-gate", "x")
    assert r.returncode == 2 and "usage" in r.stderr
    r = run("bigc-gate", env=dict(_env(), BIGC_TEST_TABLE=STAND_IN_TABLE))
    assert r.returncode == 1 and "BIGC_TEST_TABLE is for DRYRUN" in r.stderr
    r = run("bigc-gate")                                     # passes the cell stage, stops on the sha pin
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    tbl = dict(e, BIGC_TEST_TABLE=STAND_IN_TABLE, BIGC_TEST_TABLE_SHA="a" * 64)
    unset = dict(e, BIGC_TEST_TABLE="-")                     # DRYRUN: the committed pins unset
    pin_args = "%s/%s --%s %s" % (STEP_ROOT, bl.TABLE_FILE, "%s", bl.TABLE_SHA256)
    # the committed pins (the frozen table of bigc-tune 2191262)
    r = run("bigc-gate", sha=sha)
    assert r.returncode == 0 and "G-c table %s/%s sha %s" % (STEP_ROOT, bl.TABLE_FILE, bl.TABLE_SHA256) in r.stdout, \
        r.stderr[-300:]
    assert ("bigc_gate.py --out /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_118_bigc-gate --variants strict\\,bucket --seed 1234 "
            "--table " + pin_args % "table-sha") in r.stdout, r.stdout[-800:]
    r = run("bigc-time", sha=sha)
    assert r.returncode == 0 and ("--bigemm-c both --bigc-table " + pin_args % "bigc-table-sha") in r.stdout, \
        r.stdout[-800:]
    r = run("bigc-tune", sha=sha)                            # pins set: a new tuning run is refused
    assert r.returncode == 1 and "bigc-tune makes the table" in r.stderr
    r = run("bigc-tune", sha=sha, env=tbl)
    assert r.returncode == 1 and "bigc-tune makes the table" in r.stderr
    # pins unset: the source TABLE is frozen ("tuned": True), so gate / time run on it without --table; the STOP for an
    # untuned source cannot be reached at this commit (its text is pinned instead)
    r = run("bigc-gate", sha=sha, env=unset)
    assert r.returncode == 0 and "G-c table source" in r.stdout and "--table" not in r.stdout, r.stderr[-300:]
    assert "needs the frozen table: run bigc-tune" in STEP.read_text()
    r = run("bigc-gate", sha=sha, env=dict(e, BIGC_TEST_TABLE=STAND_IN_TABLE))
    assert r.returncode == 1 and "partly set" in r.stderr
    r = run("bigc-gate", sha=sha, env=dict(e, BIGC_TEST_TABLE=STAND_IN_TABLE, BIGC_TEST_TABLE_SHA="xyz"))
    assert r.returncode == 1 and "not a sha256" in r.stderr
    # a path outside the bound root (pytest's tmp_path may itself lie under /mnt/beegfs/ojerbi: job 2191402); the
    # root check comes before the existence check, so the file need not exist
    r = run("bigc-gate", sha=sha, env=dict(e, BIGC_TEST_TABLE="/tmp/bigc-not-bound/t.json", BIGC_TEST_TABLE_SHA="a" * 64))
    assert r.returncode == 1 and "not under /mnt/beegfs/ojerbi" in r.stderr, r.stderr[-300:]
    missing = "/mnt/beegfs/ojerbi/models/NOSA-8B/no-such-bigc-table.json"
    assert not os.path.exists(missing)
    r = run("bigc-gate", sha=sha, env=dict(e, BIGC_TEST_TABLE=missing, BIGC_TEST_TABLE_SHA="a" * 64))
    assert r.returncode == 1 and "no table file %s" % missing in r.stderr, r.stderr[-300:]
    r = run("bigc-tune", sha=sha, env=unset)
    assert r.returncode == 0, r.stderr[-300:]
    assert "cell bigc-tune STEP_MIN 60 claim bigc_active" in r.stdout and "--time=60" in r.stdout
    assert "Efficiency/bigc_tune.py --out /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_118_bigc-tune --warm 3 --reps 10" in r.stdout
    assert "--biattn" not in r.stdout and "--job-name=bi-t1-118-bigc-tune" in r.stdout
    r = run("bigc-gate", sha=sha, env=tbl)
    assert r.returncode == 0 and "cell bigc-gate STEP_MIN 30 claim bigc_active" in r.stdout, r.stderr[-300:]
    assert ("bigc_gate.py --out /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_118_bigc-gate --variants strict\\,bucket --seed "
            "1234 --table %s --table-sha %s" % (STAND_IN_TABLE, "a" * 64)) in r.stdout, r.stdout[-800:]
    r = run("bigc-time", sha=sha, env=tbl)
    assert r.returncode == 0 and "--time=20" in r.stdout
    assert ("--gemm-scan --ms 64\\,128\\,256\\,320\\,512\\,768\\,861\\,1024\\,1148 --exact-b 287\\,216\\,64 --U 3 --warm 5 "
            "--reps 30 --seed 7 --bigemm-c both --bigc-table %s --bigc-table-sha %s" % (STAND_IN_TABLE, "a" * 64)
            ) in r.stdout, r.stdout[-800:]
    r = run("gm1", sha=sha)                                  # gm1 and the default cell are unchanged
    assert r.returncode == 0 and "claim bi_active" in r.stdout and "--bigemm-c" not in r.stdout
    r = run(sha=sha)
    assert r.returncode == 0 and "--biattn /mnt/beegfs/ojerbi/r2-runs/bi/bi_t1_118.step/piece3_bi_paged_attention.py" \
        in r.stdout and "claim bi_active" in r.stdout
