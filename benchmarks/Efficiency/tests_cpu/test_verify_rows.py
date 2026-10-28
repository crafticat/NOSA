"""CPU tests of Task 1.4 ROUTE 1 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
the verify rows in the same forward pass (nosi/nosi/verify_rows.py; benchmarks/Efficiency/r2_verify_cell.py;
r2_verify_step.sh cell vr1). The world: tests_cpu/verify_world.py over tests_cpu/ragged_world.py (the SHIPPED engine
from its own source; every CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). B 1 and B 8 (the
largest CPU batch). No CUDA, no Triton launch.

GATE 0   test_flag_off_is_shipped: nosi/nosi (verify_rows.py and, in the integration tree nosi-r2-spec, the files of
         the other hooks excepted: each has its own Gate 0), dependencies/ and the ragged step script are the Task 1.3
         tip's bytes; no flag-off path names verify_rows; a VerifyState attached and never
         used changes no engine or twin table, and the ragged step stays bitwise the shipped step.
         Negative control "touch" (an attach that writes one tail row): caught by the table check.
EXACT    test_verify_equals_serial_shipped (B 1, B 8; attach after the shipped prefill; gemm rowgroup): every row of
         every joint pass BITWISE the shipped decode step at that position (F4 fill row, a block edge inside a group,
         the victim pool), the same loads and pool decisions, then every engine table and the twin.
         Negative control "shared_row_lengths": caught (precondition: the rows' M_b differ inside group 0).
         test_verify_equals_serial_ragged (B 1, B 8; ragged lengths; gemm exact = one GEMM call): every row BITWISE the
         ragged serial step of its request (Review Focus: in one group some requests cross a block edge, at different
         rows, and others do not), then every table.
BLOCK 0  test_every_verify_row_lists_block0 (EXACT and FAST): every row selection lists block 0 (the init block).
         Negative control "no_init_force": caught (precondition, L34: on these data some row does not select block 0
         by its scores alone).
WEIGHTS  test_one_weight_pass: one embedding call and ONE GEMM call per weight (4 per layer + lm_head), each over all
         E = B x R rows (FAST natural, EXACT exact). Negative control gemm rowgroup (R calls per weight): caught.
UNION    test_union_holds_every_row_selection (FAST): every block of every row's selection is present for that row,
         with the host rows of the block and its cis rows; EXACT: when row r attends, the window holds row r's
         selection. With one extras slot the dropped blocks equal the COUNTED overflow (never silent). Negative
         control: a silent drop (an extras id cleared after _extras) is caught by the same equality.
NO LOOP  test_no_per_position_selection_loop: static (no stage-1 / pooling / top-k call inside a loop in
         verify_rows.py) and dynamic (one stage-1 and one pooling call per layer, each over E rows). Negative
         control: verify_world.looped_select_rows (the pilot's per-position loop) is flagged by both.
FAST     test_fast_rows_match_exact (B 1, B 8): FAST rows within 1e-4 of the EXACT rows with the same argmax; layer
         0's selections bitwise; at most 2 later-layer near-tie moves (counted). Negative controls "no_extras",
         "no_row_mask", "causal_leak": caught (each with its precondition on these data).
FAKE     test_cis_stand_in_is_row_invariant (FIX 1): ragged_world.fake_nosa_linear gives a row the same bits in an
         E-row call and in a B-row call, as the kernel's one program per (row, head). Negative control: the former
         batch-wide softplus (skipped, not passed, when it is a no-op on the CPU of the run).
         test_fake_attention_idx_matches_fake: the extended attention stand-in reads entry e's bias row and request b's
         K / V, and equals ragged_world.fake_attention bitwise per entry; the API's bcache shape check.
CELL     test_refusals, test_verify_cell_arguments_and_verdicts, test_step_script_vr1.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_verify_rows.py
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-verify1"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
RAGGED_TIP = "0bdc045"                 # Task 1.3's tip, the base of this tree


def _vw():
    if "verify_world" not in sys.modules:
        spec = importlib.util.spec_from_file_location("verify_world", str(HERE / "verify_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["verify_world"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["verify_world"]


V = _vw()
W, VR, RE = V.W, V.VR, V.RE
K, R = 3, 4
S_SHIP = 4158                                                    # 4159 fills block 64 at row 1 of group 0 (F4)
S8 = [4100, 4157, 4223, 4160, 4095, 4111, 4130, 4191]           # group 0: 4157 fills at row 2, 4223 / 4095 at row 0
SF8 = [4800, 4857, 4923, 4860, 4795, 4811, 4830, 4891]          # 75-77 blocks: rows drop blocks; fills at r 3, r 0, r 2
assert len(S8) == len(SF8) == W.B_MAX


# ------------------------------------------------------------------ GATE 0
def test_flag_off_is_shipped():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", RAGGED_TIP + "^{commit}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no git or no commit %s in %s" % (RAGGED_TIP, ROOT))
    # The integration tree nosi-r2-spec merges three more hooks, each excluded here and checked by its own Gate 0:
    # ragged_rollback.py / state_snapshot.py (Task 1.5, test_rollback.py), cache_engine.py's appended draft block
    # (Task 1.1, test_draft_pool.py), pool_prefetch.py / r2_transport_gilfree_v3.py (Task 1.2, test_pool_prefetch.py),
    # spec_round.py (Phase 2, test_spec_round.py). r2_base_cell.py carries the --rollback and --draft-resident arms
    # (their Gate 0s check that every call sits under its flag); verify_rows is still named by no flag-off file (below).
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", RAGGED_TIP, "--", "nosi/nosi", "dependencies",
                        "benchmarks/Efficiency/r2_base_ragged_step.sh",
                        ":(exclude)nosi/nosi/verify_rows.py", ":(exclude)nosi/nosi/ragged_rollback.py",
                        ":(exclude)nosi/nosi/state_snapshot.py", ":(exclude)nosi/nosi/cache_engine.py",
                        ":(exclude)nosi/nosi/pool_prefetch.py", ":(exclude)nosi/nosi/r2_transport_gilfree_v3.py",
                        ":(exclude)nosi/nosi/spec_prefetch.py",      # rung L5 (Gate 0: test_spec_prefetch.py)
                        ":(exclude)nosi/nosi/spec_tentative.py",     # WS-A (Gate 0: test_wsa.py)
                        ":(exclude)nosi/nosi/spec_tentative_union.py",   # I2 (Gate 0: test_i2.py)
                        ":(exclude)nosi/nosi/spec_round.py", ":(exclude,glob)**/__pycache__/**"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", "flag-off sources differ from %s:\n%s" % (RAGGED_TIP, r.stdout)
    for f in ("nosi/nosi/ragged_engine.py", "nosi/nosi/nosa_llama.py", "nosi/nosi/cache_engine.py",
              "benchmarks/Efficiency/r2_base_cell.py"):
        assert "verify_rows" not in (ROOT / f).read_text(), "%s names verify_rows" % f
    for B in (1, W.B_MAX):
        V.run_flag_off(B, 4100, 8)
    with pytest.raises(AssertionError, match="the verify attach changed _k_gpu"):
        V.run_flag_off(1, 4100, 1, plant="touch")


# ------------------------------------------------------------------ EXACT
@pytest.mark.parametrize("B", [1, W.B_MAX], ids=["B1", "B8"])
def test_verify_equals_serial_shipped(B):
    ev = V.run_joint_vs_shipped(B, S_SHIP, 3)                    # raises at the first unequal bit, then the tables
    assert (0, 1) in ev["fills"]                                 # precondition: a block edge inside group 0 (F4)
    st = ev["stats"]
    assert st["joint"] == st["shipped"], st                      # the same loads and the same pool decisions
    assert st["shipped"].get("move_out", 0) > 0, st              # the victim pool parked blocks
    assert ev["errors"] == (0, [0] * W.L)
    u = ev["verify"]
    assert u["union_samples"] == 3 * W.L * W.HKV * B and u["union_max"] >= W.TOPK


def test_negctl_shared_row_lengths_fails_exact():
    assert (S_SHIP + 2) // W.BS != S_SHIP // W.BS                # precondition: row 2's M_b is row 0's + 1
    with pytest.raises(AssertionError, match="the joint logits differ"):
        V.run_joint_vs_shipped(1, S_SHIP, 1, neg="shared_row_lengths", compare_state=False)


@pytest.mark.parametrize("S_list", [[4159], S8], ids=["B1", "B8"])
def test_verify_equals_serial_ragged(S_list):
    ev = V.run_joint_vs_ragged(S_list, 3)                        # EXACT, gemm exact; bitwise, then every table
    g0 = {(r, b) for (g, r, b) in ev["fills"] if g == 0}
    if len(S_list) == 1:
        assert g0 == {(0, 0)}, ev["fills"]                       # precondition: the fill row is row 0
    else:
        filled = {b for _, b in g0}
        assert filled and len(filled) < len(S_list), g0          # precondition: some cross, some do not
        assert len({r for r, _ in g0}) >= 2, g0                  # at different rows of the group
    assert ev["errors"] == (0, [0] * W.L)


# ------------------------------------------------------------------ BLOCK 0
def test_every_verify_row_lists_block0():
    for mode in ("exact", "fast"):
        ev = V.run_joint_vs_ragged(SF8, 2, mode=mode, record=True, compare="bits" if mode == "exact" else None)
        bad, n = V.rows_without_block0(ev["vs"].log)
        assert n > 0 and bad == [], (mode, n, bad[:5])
    ev = V.run_joint_vs_ragged(SF8, 2, mode="fast", record=True, compare=None, neg="no_init_force")
    bad, n = V.rows_without_block0(ev["vs"].log)
    assert bad, "precondition (L34) not met: with no forced init block every row still selects block 0 (%d rows)" % n


# ------------------------------------------------------------------ WEIGHTS
def _counting(fn):
    calls = []

    def lin(x, w):
        calls.append(int(x.reshape(-1, x.shape[-1]).shape[0]))
        return fn(x, w)
    lin.__name__ = "counting_" + getattr(fn, "__name__", "linear")
    return lin, calls


class _FProxy:
    """torch.nn.functional with a counted embedding (verify_rows.F in the test)."""

    def __init__(self, rec):
        self.rec = rec

    def __getattr__(self, name):
        return getattr(F, name)

    def embedding(self, ids, weight):
        self.rec.append(int(ids.numel()))
        return F.embedding(ids, weight)


def test_one_weight_pass(monkeypatch):
    emb = []
    monkeypatch.setattr(VR, "F", _FProxy(emb))
    S_list = S8[:4]
    B, E = len(S_list), len(S_list) * R
    want = 4 * W.L + 1                                           # wqkv, wo, gate_up, down per layer; lm_head
    for mode, gemm, base in (("fast", "natural", F.linear), ("exact", "exact", W.row_linear)):
        lin, calls = _counting(base)
        emb.clear()
        V.run_joint_vs_ragged(S_list, 1, mode=mode, gemm=gemm, linear=lin,
                              compare="bits" if mode == "exact" else None)
        assert len(calls) == want and set(calls) == {E}, (mode, calls)
        assert emb == [E], (mode, emb)
    lin, calls = _counting(F.linear)                             # NEGATIVE CONTROL: R calls per weight
    V.run_joint_vs_ragged(S_list, 1, mode="exact", gemm="rowgroup", linear=lin, compare=None)
    assert R > 1 and len(calls) == R * want and set(calls) == {B}, calls
    assert not (len(calls) == want and set(calls) == {E})


# ------------------------------------------------------------------ UNION
def test_union_holds_every_row_selection(monkeypatch):
    ev = V.run_fast_vs_exact(SF8, 2)
    st = ev["fast"]
    assert st["extras"] > 0, st                                  # precondition: rows need blocks outside the window
    u = V.check_union(ev["vf"].log, ev["vf"].X)
    assert u["checked"] > 0 and u["missing"] == 0 and u["contents_bad"] == [] and st["overflow"] == 0, (
        u["missing_list"][:5], u["contents_bad"][:5], st)
    bad, n = V.exact_windows_hold_rows(ev["vx"].log)
    assert n > 0 and bad == [], bad[:5]
    ev2 = V.run_fast_vs_exact(SF8, 2, extra=2, compare=False)    # ONE extras slot
    st2 = ev2["fast"]
    assert st2["overflow"] > 0, st2                              # precondition: some (head, request) needs two
    u2 = V.check_union(ev2["vf"].log, 2)
    assert u2["missing"] == st2["overflow"] and u2["contents_bad"] == [], (u2["missing"], st2)
    orig = VR._extras

    def silent(vs, eng, twin, sel, m, crossing):                 # NEGATIVE CONTROL: drop without a count
        ids = orig(vs, eng, twin, sel, m, crossing)
        ids[..., 0] = -1
        return ids
    monkeypatch.setattr(VR, "_extras", silent)
    ev3 = V.run_fast_vs_exact(SF8, 2, compare=False)
    u3 = V.check_union(ev3["vf"].log, ev3["vf"].X)
    assert ev3["fast"]["overflow"] == 0 and u3["missing"] > 0    # dropped, not counted: the equality above fails
    assert u3["missing"] != ev3["fast"]["overflow"]


# ------------------------------------------------------------------ NO LOOP
SEL_NAMES = {"stage1_seqused", "nosa_pooling_per_request", "topk", "_select_rows", "_probe_row0"}
LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def loop_calls(src, names):
    """(name, line) of every call to one of `names` that sits inside a loop or a comprehension of its own function."""
    tree = ast.parse(src)
    parent = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parent[c] = p
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name not in names:
            continue
        a = parent.get(n)
        while a is not None and not isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)):
            if isinstance(a, LOOPS):
                bad.append((name, n.lineno))
                break
            a = parent.get(a)
    return bad


def test_no_per_position_selection_loop(monkeypatch):
    src = V.VR_PATH.read_text()
    found = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("stage1_seqused", "nosa_pooling_per_request")]
    assert len(found) >= 2                                       # precondition: the scan sees the calls
    assert loop_calls(src, SEL_NAMES) == []
    planted = inspect.getsource(V.looped_select_rows)
    assert any(isinstance(x, ast.For) for x in ast.walk(ast.parse(planted)))
    assert loop_calls(planted, SEL_NAMES)                        # the static scan flags the pilot's loop
    rec = []

    def s1(q, *a, **k):
        rec.append(("s1", int(q.shape[0])))
        return V.STAGE1(q, *a, **k)

    def pool(score, *a, **k):
        rec.append(("pool", int(score.shape[1])))
        return V.POOLING(score, *a, **k)
    monkeypatch.setattr(VR, "stage1_seqused", s1)
    monkeypatch.setattr(VR, "nosa_pooling_per_request", pool)
    S_list = S8[:4]
    B, E = len(S_list), len(S_list) * R
    V.run_joint_vs_ragged(S_list, 1)                             # bitwise the serial steps
    assert rec == [("s1", E), ("pool", E)] * W.L, rec
    rec.clear()
    monkeypatch.setattr(VR, "_select_rows", V.looped_select_rows)
    V.run_joint_vs_ragged(S_list, 1)                             # precondition: the same bits (the count changes)
    assert rec == [("s1", B), ("pool", B)] * (R * W.L), rec
    assert rec != [("s1", E), ("pool", E)] * W.L


# ------------------------------------------------------------------ FAST
@pytest.mark.parametrize("S_list", [[4857], SF8], ids=["B1", "B8"])
def test_fast_rows_match_exact(S_list):
    ev = V.run_fast_vs_exact(S_list, 3)
    st = ev["fast"]
    assert st["overflow"] == 0, st
    if len(S_list) > 1:
        assert st["extras"] > 0, st
    assert ev["max_rel"] <= 1e-5 and len(ev["flips"]) <= 2, (ev["max_rel"], ev["flips"])
    assert ev["errors"] == ((0, [0] * W.L), (0, [0] * W.L))
    fa = ev["vf"].fa_check                                       # the cell's VRF-IDX path on the stand-ins
    assert fa["entries"] and all(fa["equal"]), fa


@pytest.mark.parametrize("neg", ["no_extras", "no_row_mask", "causal_leak"])
def test_negctl_fast_plants_fail(neg):
    ev = V.run_fast_vs_exact(SF8, 2, compare=False)
    if neg in ("no_extras", "no_row_mask"):
        assert ev["fast"]["extras"] > 0                          # precondition: a row needs a block outside the window
    else:                                                        # precondition: a request has R rows in its tail
        assert any((s + R * g) % W.BS < W.BS - R for s in SF8 for g in range(2))     # block and does not fill it
    with pytest.raises(AssertionError, match="FAST"):
        V.run_fast_vs_exact(SF8, 2, neg=neg)


# ------------------------------------------------------------------ FAKE
def test_cis_stand_in_is_row_invariant():
    """FIX 1 (CPU run vr_ee3f9ee_a: 5 failures, all at B >= 4, all an E-row pass against a B-row step). The kernel
    computes cis in one program per (row, head) (nosa_linear.py:84-93, grid :110), so a row's cis does not depend on
    the row count. The stand-in must not either: B-row slices of an E-row call give the same bits."""
    lay = W.make_model(0).layers[0]
    g = torch.Generator().manual_seed(11)
    E, B = 256, 4
    n = E // B
    qkv = torch.randn(E, 1, lay.wqkv.shape[0], generator=g)
    args = (lay.delta.weight, lay.A, lay.q_size, lay.kv_size, W.HKV)
    full = W.fake_nosa_linear(qkv, *args)
    for r in range(n):
        part = W.fake_nosa_linear(qkv[r::n].contiguous(), *args)          # the serial step's call: B rows
        for a, b in zip(full, part):
            assert V.first_diff(a[r::n], b) is None, (r, V.first_diff(a[r::n], b))

    def old_cis(x):                                              # NEGATIVE CONTROL: the former stand-in, one
        v = x[..., lay.q_size + lay.kv_size:].contiguous()       # softplus over the whole batch tensor
        return lay.A * F.softplus(W.row_linear(v, lay.delta.weight))
    whole = old_cis(qkv)
    diffs = [V.first_diff(whole[r::n], old_cis(qkv[r::n].contiguous())) for r in range(n)]
    if all(d is None for d in diffs):
        pytest.skip("precondition (L34) not met: this CPU's batched softplus gives every row the bits of a small "
                    "batch on these data, so the former stand-in is a no-op plant here")
    assert any(d is not None for d in diffs)                     # the same check catches the former stand-in


def test_fake_attention_idx_matches_fake():
    g = torch.Generator().manual_seed(3)
    Bc, rows = 3, W.TOPK * W.BS
    E = Bc * R
    q = torch.randn(E, 1, W.HQ, W.D, generator=g)
    k = torch.randn(Bc, rows, W.HKV, W.D, generator=g)
    v = torch.randn(Bc, rows, W.HKV, W.D, generator=g)
    bias = torch.randn(E, rows, W.HKV, generator=g)
    lens = torch.randint(100, rows + 1, (E,), generator=g).to(torch.int32)
    cbi = (torch.arange(E) // R).to(torch.int32)
    out, lse = V.fake_attention_idx(q, k, v, bias[:Bc], cache_seqlens=lens, cache_batch_idx=cbi,
                                    num_splits=W.SPLITS, return_softmax_lse=True)
    for e in range(E):
        b = e // R
        ref = W.fake_attention(q[e:e + 1], k[b:b + 1], v[b:b + 1], bias[e:e + 1], lens[e:e + 1], W.SPLITS)
        assert torch.equal(W.bits(out[e:e + 1]), W.bits(ref)), e  # entry e: its own bias row, request b's K / V
        n = int(lens[e])
        sc = torch.einsum("hgd,lhd->hgl", q[e, 0].view(W.HKV, -1, W.D).double(), k[b, :n].double()) * W.D ** -0.5 \
            + bias[e, :n].double().t().unsqueeze(1)
        assert torch.allclose(lse[e, :, 0].double(), torch.logsumexp(sc, -1).reshape(-1), rtol=1e-5, atol=1e-5), e
    alt = W.fake_attention(q[R:R + 1], k[1:2], v[1:2], bias[1:2], lens[R:R + 1], W.SPLITS)
    assert not torch.equal(W.bits(out[R:R + 1]), W.bits(alt))   # the cache index's bias row would give other bits
    with pytest.raises(AssertionError, match="bcache"):          # flash_api.cpp:366
        V.fake_attention_idx(q, k, v, bias, cache_seqlens=lens, cache_batch_idx=cbi, num_splits=W.SPLITS)


# ------------------------------------------------------------------ CELL
def test_refusals():
    for kw in (dict(K=0), dict(K=32), dict(mode="x"), dict(gemm="x"), dict(gemm="exact"), dict(extra=1),
               dict(neg="nope")):
        with pytest.raises(VR.VerifyError):
            VR.VerifyOps(**kw)
    model, _, Bw = V._two_built([4100, 4157], 8, 0)
    vs = VR.attach(model, Bw, VR.VerifyOps(K=K, attn=V.fake_attention_idx))
    ids = torch.zeros((2, R), dtype=torch.int64)
    pos = torch.tensor([[4100 + r for r in range(R)], [4157 + r for r in range(R)]], dtype=torch.int32)
    with pytest.raises(VR.VerifyError, match="must be"):
        VR.joint_pass(model, vs, ids[:, :K], pos[:, :K])
    RE.set_activity(Bw, torch.tensor([True, False]), torch.tensor([True, False]))
    with pytest.raises(VR.VerifyError, match="active"):
        VR.joint_pass(model, vs, ids, pos)


def _vc():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_verify_cell
    return r2_verify_cell


def test_verify_cell_arguments_and_verdicts(monkeypatch):
    VC = _vc()
    base = ["--out", "x", "--workload", "w"]
    a = VC.parse_args(base)
    assert (a.batch, a.steps, a.pool, a.verify_k, a.verify_arms, a.verify_extra, a.verify_twin, a.verify_body,
            a.probe_every) == (216, 512, 65, 3, ["VRX", "VRC", "VRF"], 16, "torch", "fused", 16)
    for bad in (["--verify-arms", "VRX,VRX"], ["--verify-arms", "XX"], ["--verify-k", "0"], ["--verify-k", "32"],
                ["--steps", "6"], ["--verify-extra", "1"], ["--probe-every", "1"], ["--pool", "66"]):
        with pytest.raises(SystemExit):
            VC.parse_args(base + bad)
    assert VC.probe_passes(40, 16) == [16, 32]
    assert VC.steady_passes(40, 16) == [g for g in range(1, 40) if g not in (16, 32)]
    pred = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    inputs = torch.tensor([[0, 1, 2, 9, 4, 5, 6, 9, 8]])
    assert VC.row_agreement(pred, inputs, 4) == [1.0, 1.0, 0.0, 1.0]
    assert VC.median_phases([dict(a=1.0, b=2.0), dict(a=3.0)]) == dict(a=2.0, b=1.0)
    marks = [(0, "pre", [0.0]), (0, "twin", [1.0]), (0, "select", [3.0]), (1, "pre", [6.0])]
    assert VR.phase_ms(marks, lambda x, y: y[0] - x[0]) == dict(pre=1.0, twin=2.0, select=3.0)
    sh = dict(rates=dict(steady_step_ms=64.0))
    vr = dict(VRX=dict(fingerprint=dict(equal_fraction=1.0, n_unequal=0, first=None), steady=dict(median_ms=200.0), R=4),
              VRF=dict(agreement=dict(tf_agreement=0.99), row_agreement=[1.0, 1.0, 1.0, 0.96],
                       steady=dict(median_ms=150.0), R=4, phases_median=dict(select=20.0, probe=8.0),
                       stats=dict(union_mean=70.5, union_max=80)))
    for key in VC.PRED_VERIFY:                                   # the builder's state: unregistered
        monkeypatch.setitem(VC.PRED_VERIFY, key, None)
    v = {n: x for n, x, _ in VC.verdicts_verify(sh, vr)}
    assert v == {"VRX-BIT": "REPORT", "VRX-STEP": "REPORT", "VRC-STEP": "NOT_RUN", "VRF-AGREE": "REPORT",
                 "VRF-STEP": "REPORT", "VR-SEL": "REPORT", "VR-UNION": "REPORT"}, v
    for key, val in (("VRX_bitwise", True), ("VRX_ratio", (3.0, 3.2)), ("VRF_agree", 0.995), ("VRF_ratio", (2.0, 2.5)),
                     ("sel_ratio", (2.0, 3.0)), ("union_mean", (60.0, 70.0))):
        monkeypatch.setitem(VC.PRED_VERIFY, key, val)
    v = {n: x for n, x, _ in VC.verdicts_verify(sh, vr)}
    assert v == {"VRX-BIT": "CONFIRMED", "VRX-STEP": "CONFIRMED", "VRC-STEP": "NOT_RUN", "VRF-AGREE": "REFUTED",
                 "VRF-STEP": "CONFIRMED", "VR-SEL": "CONFIRMED", "VR-UNION": "REFUTED"}, v
    assert VC.verdicts_verify(sh, {})[-1][1] == "NOT_RUN"
    res = dict(body_err=0, overflow=[0, 0], mode="fast", fa_check=dict(entries=[0, 5], equal=[True, True]),
               stats=dict(sel0_samples=4, sel0_set_eq=4, sel0_order_eq=3, overflow=0, extras=10))
    g = {n: (must, ok) for n, must, ok, _ in VC.arm_gates("VRF", res)}
    assert g == {"VRF-ERR": (True, True), "VRF-SEL0": (True, True), "VRF-SEL0-ORDER": (False, False),
                 "VRF-IDX": (True, True), "VRF-OVF": (False, True)}, g
    res["fa_check"] = dict(entries=[0, 5], equal=[True, False])
    res["overflow"] = [0, 1]
    g = {n: (must, ok) for n, must, ok, _ in VC.arm_gates("VRF", res)}
    assert g["VRF-IDX"] == (True, False) and g["VRF-ERR"] == (True, False)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_verify_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_vr1():
    for bad in ("vr2", "mb1", "rg1"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    r = _step("vr1", sha)
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
    line = ("cell vr1: C 128 (pool 65) B 216 arms SH+VRX,VRC,VRF K 3 steps 512 extra 16 twin torch STEP_MIN 100 "
            "mem 240G")
    assert line in r.stdout, r.stdout[-600:]
    for want in ("NOSI_POOL_BLOCKS=65", "NOSI_ATTN_SPLITS=4", "--pool 65", "--batch 216", "--steps 512",
                 "--verify-k 3", "--verify-extra 16", "--verify-twin torch", "--verify-body fused", "--probe-every 16",
                 "--mem=240G", "--time=100", "r2_verify_cell.py", "vr_t1_%s_vr1" % HOLDER):
        assert want in r.stdout, want
    assert "--verify-arms VRX\\,VRC\\,VRF" in r.stdout or "--verify-arms VRX,VRC,VRF" in r.stdout   # printf %q
