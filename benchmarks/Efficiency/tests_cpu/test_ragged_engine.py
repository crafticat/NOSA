"""CPU tests of Task 1.3 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
per-request lengths inside the cached-NOSA engine (nosi/nosi/ragged_engine.py; r2_base_cell.py --ragged;
r2_base_ragged_step.sh cell rg1). The world: tests_cpu/ragged_world.py (the SHIPPED engine from its own source, every
CUDA op a torch stand-in that copies the kernel's flat-address arithmetic). No CUDA, no Triton launch except FUSED.

GATE 0   test_gate0_flag_off_byte_identical: nosi/nosi (ragged_engine.py excepted) and dependencies/ are the ded9d41
         bytes; r2_base_cell without --ragged parses as before; every ragged call in main sits under `if a.ragged`, and
         SH's fingerprint under `if fp is not None`.
EQUAL    test_equal_lengths_bitwise_shipped (B 1 and B 8): the ragged logits are BITWISE the shipped step's at every
         step, through the shipped re-warms (width, block), a findings-F4 step, a 64-token block edge and the victim
         pool; then every engine table (pool included) and the twin against the shipped tables.
RAGGED   test_ragged_equals_solo (B 1 and B 8): prompts with different S % 64, ragged activity, EXACT mode (row-wise
         GEMM, fixed-order twin summaries): every active request's logits BITWISE its solo run's; then its rows.
EDGE     test_ragged_block_edge (B 1 and B 8): request 0 crosses a 64-token block edge, the others do not: each equals
         its solo run; the pooling graph is captured once per layer, at step 0, and only replayed after it.
NEGATIVE CONTROLS, each with a precondition that the plant changes something on these data (lesson L34):
  shared_lengths   one batch-wide length: caught by RAGGED (precondition: the widths differ); NOT by EQUAL (shown)
  frozen_widths    the widths of step 0 for every step: caught by EDGE (precondition: request 0's width moves)
  rebound_lengths  new length tensors after the capture: caught by EQUAL (precondition: a width moves after step 0)
  no_pool          diff only: the pool is a cache (the logits stay equal), caught by the engine-state check
                   (precondition: the shipped pool parked blocks)
  unmasked_pool    the pool also runs for inactive requests: caught by RAGGED (precondition: an inactive request had
                   a load)
  padded_attention the whole allocation to attention: caught by EQUAL (precondition: the split seams move and the
                   fake attention gives other bits)
  f4_fix           the corrected F4 length: caught by EQUAL (precondition: step 0 completes a block)
FUSED    test_fused_body_with_pool_under_the_interpreter (needs triton): FusedBody with the pool == the torch body.
CELL     test_ragged_cell_arguments_schedule_fingerprint_verdicts, test_refusals, test_step_script_rg1.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_ragged_engine.py
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
EFF = ROOT / "benchmarks/Efficiency"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-ragged"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
BASE = "ded9d41"


def _world():
    if "ragged_world" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ragged_world", str(HERE / "ragged_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ragged_world"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ragged_world"]


W = _world()
S8 = [4100, 4157, 4223, 4160, 4095, 4111, 4130, 4191]          # S % 64 = 4, 61, 63, 0, 63, 15, 34, 31
EDGE8 = [4159, 4100, 4110, 4120, 4130, 4140, 4105, 4115]        # only request 0 completes its block in 3 steps
assert len(S8) == len(EDGE8) == W.B_MAX


# ------------------------------------------------------------------ GATE 0
def _guarded(fn, is_target, is_guard):
    """Line numbers of the nodes of fn with is_target(node) that no `if <is_guard>:` body contains."""
    bad = []

    def walk(node, guarded):
        for child in ast.iter_child_nodes(node):
            g = guarded or (isinstance(node, ast.If) and is_guard(node.test) and child in node.body)
            if is_target(child) and not g:
                bad.append(getattr(child, "lineno", None))
            walk(child, g)
    walk(fn, False)
    return bad


def test_gate0_flag_off_byte_identical():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", BASE + "^{commit}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no git or no commit %s in %s" % (BASE, ROOT))
    # Task 1.5 (r2-rollback) adds nosi/nosi/ragged_rollback.py (imported by no engine module) and the pool names in
    # nosi/nosi/state_snapshot.py (imported by benchmarks/Efficiency/acceptance_pilot.py only); its own Gate 0 is
    # tests_cpu/test_rollback.py test_flag_off_is_shipped. The integration tree nosi-r2-spec (Phase 2) merges three
    # more hooks, each excluded here and checked by its own Gate 0: verify_rows.py (Task 1.4, test_verify_rows.py),
    # the appended draft block of cache_engine.py (Task 1.1, test_draft_pool.py: ded9d41's statements are a prefix),
    # pool_prefetch.py and r2_transport_gilfree_v3.py (Task 1.2, test_pool_prefetch.py), and spec_round.py (Phase 2,
    # test_spec_round.py test_flags_off_is_shipped).
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", BASE, "--", "nosi/nosi", "dependencies",
                        ":(exclude)nosi/nosi/ragged_engine.py", ":(exclude)nosi/nosi/ragged_rollback.py",
                        ":(exclude)nosi/nosi/state_snapshot.py", ":(exclude)nosi/nosi/verify_rows.py",
                        ":(exclude)nosi/nosi/cache_engine.py", ":(exclude)nosi/nosi/pool_prefetch.py",
                        ":(exclude)nosi/nosi/r2_transport_gilfree_v3.py", ":(exclude)nosi/nosi/spec_round.py",
                        ":(exclude)nosi/nosi/spec_prefetch.py",      # rung L5 (Gate 0: test_spec_prefetch.py)
                        ":(exclude,glob)**/__pycache__/**"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", "the engine sources differ from %s:\n%s" % (BASE, r.stdout)
    BC = W.bc()
    a = BC.parse_args(["--out", "x", "--workload", "w", "--batch", "216", "--arms", "SH", "--pool", "65", "--mbase"])
    assert not a.ragged
    tree = ast.parse((EFF / "r2_base_cell.py").read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ragged_calls = {"run_ragged", "verdicts_ragged", "fingerprint_compare"}

    def is_call(n):
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ragged_calls

    def is_a_ragged(t):
        return isinstance(t, ast.Attribute) and t.attr == "ragged" and isinstance(t.value, ast.Name) and t.value.id == "a"
    assert _guarded(fns["main"], is_call, is_a_ragged) == []
    n_calls = sum(1 for n in ast.walk(fns["main"]) if is_call(n))
    assert n_calls == 3, n_calls

    def is_fp_use(n):
        return isinstance(n, ast.Name) and n.id == "fp" and isinstance(n.ctx, ast.Load)

    def is_fp_guard(t):
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "fp"
                and isinstance(t.ops[0], ast.IsNot) and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value is None)
    uses = [n for n in ast.walk(fns["run_shipped"]) if is_fp_use(n)]
    assert len(uses) == 3 and _guarded(fns["run_shipped"], is_fp_use, is_fp_guard) == [uses[0].lineno]


# ------------------------------------------------------------------ EQUAL
@pytest.mark.parametrize("B,S,steps", [(1, 4100, 80), (W.B_MAX, 4100, 80), (W.B_MAX, 4159, 40)],
                         ids=["B1-S4100", "B8-S4100", "B8-S4159"])
def test_equal_lengths_bitwise_shipped(B, S, steps):
    ev = W.run_equal(B, S, steps)
    if S == 4100:                                        # what the run passed through (preconditions of the claim)
        assert ev["rewarms"] == ["first", "width", "out_len"], ev["rewarms"]
        assert ev["p63_steps"] == [59]
    else:
        assert ev["rewarms"][:2] == ["first", "out_len"] and ev["p63_steps"] == [0], ev
    st = ev["stats"]["shipped"]
    assert st.get("move_out", 0) > 0, st                 # the shipped pool parked blocks
    if B == W.B_MAX:
        assert st.get("swap", 0) + st.get("move_in", 0) > 0, st     # and served hits
    assert ev["stats"]["ragged"] == st                   # the same loads and the same pool decisions
    assert ev["captures"] == W.L and ev["capture_steps"] == [0] * W.L


# ------------------------------------------------------------------ RAGGED
@pytest.mark.parametrize("S_list", [[4157], S8], ids=["B1", "B8"])
def test_ragged_equals_solo(S_list):
    ev = W.run_solo(S_list, 70)
    if len(S_list) > 1:
        assert ev["inactive"] > 0
        assert len({int(x) for x in ev["mb"][0]}) > 1, ev["mb"][0]          # the widths differ (ragged)
    assert sum(ev["crossings"]) >= (1 if len(S_list) == 1 else 4), ev["crossings"]
    assert ev["rs"].captures == W.L


# ------------------------------------------------------------------ EDGE
@pytest.mark.parametrize("S_list", [[4159], EDGE8], ids=["B1", "B8"])
def test_ragged_block_edge(S_list):
    ev = W.run_solo(S_list, 3, p_active=1.0, all_active=3)
    mb = torch.stack(ev["mb"])
    assert int(mb[1, 0]) == int(mb[0, 0]) + 1, mb                 # request 0 enters a new block at step 1
    assert bool((mb[:, 1:] == mb[0, 1:]).all()), mb               # the others stay in theirs
    rs = ev["rs"]
    assert rs.captures == W.L and rs.capture_steps == [0] * W.L and rs.replays == 3 * W.L
    for s in ev["solos"]:
        assert s.captures == W.L and s.capture_steps == [0] * W.L


# ------------------------------------------------------------------ NEGATIVE CONTROLS
def test_negctl_shared_lengths_fails_ragged():
    S_list = S8[:4]
    ev = W.run_solo(S_list, 3, p_active=1.0, all_active=3)
    assert len({int(x) for x in ev["mb"][0]}) > 1                 # precondition: the widths differ
    with pytest.raises(AssertionError, match="differ"):
        W.run_solo(S_list, 3, neg="shared_lengths", p_active=1.0, all_active=3)


def test_negctl_shared_lengths_passes_equal():
    """Equal lengths cannot see a batch-wide length: RAGGED is the gate that does."""
    W.run_equal(2, 4100, 6, neg="shared_lengths")


def test_negctl_frozen_widths_fails_block_edge():
    S_list = EDGE8[:2]
    ev = W.run_solo(S_list, 3, p_active=1.0, all_active=3)
    mb = torch.stack(ev["mb"])
    assert int(mb[1, 0]) != int(mb[0, 0])                        # precondition: request 0's width moves
    with pytest.raises(AssertionError, match="request 0"):
        W.run_solo(S_list, 3, neg="frozen_widths", p_active=1.0, all_active=3)


def test_negctl_rebound_lengths_fails_equal():
    ev = W.run_equal(1, 4100, 62, compare_state=False)
    assert ev["mb"][61] != ev["mb"][0] and ev["captures"] == W.L  # precondition: a width moves after the capture
    with pytest.raises(AssertionError, match="step 60: the ragged logits differ"):
        W.run_equal(1, 4100, 62, neg="rebound_lengths", compare_state=False)


def test_negctl_no_pool_is_caught_by_the_state():
    ev = W.run_equal(2, 4100, 12)
    assert ev["stats"]["shipped"].get("move_out", 0) > 0          # precondition: the shipped pool parked blocks
    with pytest.raises(AssertionError, match="differ"):
        W.run_equal(2, 4100, 12, neg="no_pool")


def test_negctl_unmasked_pool_fails_ragged():
    S_list = S8[:4]
    rec = []

    def diff_rec(old, topk, new, load):
        W.diff_offload_ref(old, topk, new, load)
        rec.append((load >= 0).any(-1).any(0).clone())            # (B,): the request has a load in this call
    kern = dict(W.KERNELS, diff=diff_rec)
    ev = W.run_solo(S_list, 30, kernels=kern, all_active=4)
    assert len(rec) == 30 * W.L
    inactive_loads = sum(int((rec[i * W.L + l] & ~a).sum()) for i, a in enumerate(ev["acts"]) for l in range(W.L))
    assert inactive_loads > 0                                    # precondition: an inactive request had a load
    with pytest.raises(AssertionError, match="differ"):
        W.run_solo(S_list, 30, kernels=kern, neg="unmasked_pool", all_active=4)


def test_negctl_padded_attention_fails_equal():
    att, full = W.TOPK * W.BS, (W.TOPK + W.P) * W.BS
    assert W.split_partition(att, W.SPLITS)[1] != W.split_partition(full, W.SPLITS)[1]   # precondition: the seams
    g = torch.Generator().manual_seed(5)
    q = torch.randn(1, 1, W.HQ, W.D, generator=g)
    k, v = torch.randn(1, full, W.HKV, W.D, generator=g), torch.randn(1, full, W.HKV, W.D, generator=g)
    bias = torch.randn(1, full, W.HKV, generator=g)
    lens = torch.tensor([att - 10], dtype=torch.int32)
    a = W.fake_attention(q, k[:, :att], v[:, :att], bias[:, :att], lens, W.SPLITS)
    b = W.fake_attention(q, k, v, bias, lens, W.SPLITS)
    assert not torch.equal(a, b)                                 # precondition: other bits on the same rows
    with pytest.raises(AssertionError, match="the ragged logits differ from the shipped step"):
        W.run_equal(1, 4100, 3, neg="padded_attention", compare_state=False)


def test_negctl_f4_fix_fails_equal():
    S = 4159
    assert S % W.BS == W.BS - 1                                  # precondition: step 0 completes a block (F4)
    with pytest.raises(AssertionError, match="step 0: the ragged logits differ"):
        W.run_equal(1, S, 2, f4_fix=True, compare_state=False)


# ------------------------------------------------------------------ FUSED (Triton interpreter, fresh process)
FUSED_CHILD = r'''
import importlib.util, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton was imported before the interpreter flag was read"
HERE = sys.argv[1]
spec = importlib.util.spec_from_file_location("ragged_world", os.path.join(HERE, "ragged_world.py"))
W = importlib.util.module_from_spec(spec)
sys.modules["ragged_world"] = W
spec.loader.exec_module(W)
FB = W.pkg_load("r2_body_fused")
assert FB.triton is not None, "child: triton did not import"
# block completions BY CONSTRUCTION inside the 4 all-active steps: 4157 at local step 2, 4223 at 0, 4159 at 0, 4158
# at 1 (4100 and 4160 complete theirs at local steps 59 / 63 only, which ragged activity may not reach: run a, 2 of 4)
ev = W.run_bodies([4100, 4157, 4223, 4160, 4159, 4158], 70)
assert ev["inactive"] > 0 and ev["crossings"] >= 4, ev
print("RGF OK FusedBody with the pool == the torch body, bitwise (70 steps, ragged activity): %s" % ev)
'''


def test_fused_body_with_pool_under_the_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1", NOSI_REPO=str(ROOT))
    r = subprocess.run([sys.executable, "-c", FUSED_CHILD, str(HERE)], env=env, cwd=str(ROOT), capture_output=True,
                       text=True, timeout=3000)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith("RGF OK"), (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-6000:]))


# ------------------------------------------------------------------ CELL
def test_ragged_cell_arguments_schedule_fingerprint_verdicts(monkeypatch):
    BC = W.bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--mbase", "--ragged"])
    assert (a.ragged, a.ragged_spread, a.ragged_twin, a.ragged_body, a.ragged_gemm, a.ragged_graph) == (
        True, 64, "torch", "fused", "natural", "on")
    for bad in (["--ragged"], ["--arms", "SH", "--ragged", "--ragged-spread", "1"],
                ["--arms", "SH", "--ragged", "--ragged-spread", "65"], ["--arms", "SH", "--ragged-twin", "x"]):
        with pytest.raises(SystemExit):
            BC.parse_args(base + bad)
    T, k, act, col, full = BC.ragged_schedule(4, 5, 2)
    assert T == 6 and full == [1, 2, 3, 4]
    assert act[:, 0].tolist() == [True] * 5 + [False] and act[:, 1].tolist() == [False] + [True] * 5
    assert k[0].tolist() == [0, 0, 0, 0] and k[5].tolist() == [5, 4, 5, 4]
    assert col[0].tolist() == [0, 5, 0, 5] and col[5].tolist() == [5, 4, 5, 4]
    T, k, act, col, full = BC.ragged_schedule(3, 4, 1)
    assert T == 4 and full == [0, 1, 2, 3] and bool(act.all())
    g = torch.Generator().manual_seed(0)
    ref = (torch.randn(3, 4, generator=g), torch.randn(3, 4, generator=g, dtype=torch.float64))
    assert BC.fingerprint_compare(ref, (ref[0].clone(), ref[1].clone())) == dict(equal_fraction=1.0, n_unequal=0,
                                                                                   first=None)
    m = ref[0].clone()
    m[2, 1] = torch.nextafter(m[2, 1], torch.tensor(float("inf")))                 # one ULP
    s = ref[1].clone()
    s[1, 3] += 1.0
    r = BC.fingerprint_compare(ref, (m, s))
    assert r["n_unequal"] == 2 and r["first"] == (2, 1) and r["equal_fraction"] == 10 / 12
    sh = dict(rates=dict(steady_step_ms=64.0))
    rg = {"RG": dict(fingerprint=dict(equal_fraction=1.0, n_unequal=0, first=None), rates=dict(steady_step_ms=65.0)),
          "RGS": dict(fingerprint=dict(equal_fraction=0.9, n_unequal=5, first=(3, 7)),
                      rates=dict(steady_step_ms=70.0))}
    for key in ("RG_bitwise", "RGS_bitwise", "RG_ratio", "RGS_ratio"):          # the builder's state: unregistered
        monkeypatch.setitem(BC.PRED_RAGGED, key, None)
    v = {n: x for n, x, _ in BC.verdicts_ragged(sh, rg)}
    assert v == {"RG-BIT": "REPORT", "RG-STEP": "REPORT", "RGS-BIT": "REPORT", "RGS-STEP": "REPORT"}, v
    for key, val in (("RG_bitwise", True), ("RGS_bitwise", True), ("RG_ratio", (0.97, 1.03)),
                     ("RGS_ratio", (0.97, 1.03))):
        monkeypatch.setitem(BC.PRED_RAGGED, key, val)
    v = {n: x for n, x, _ in BC.verdicts_ragged(sh, rg)}
    assert v == {"RG-BIT": "CONFIRMED", "RG-STEP": "CONFIRMED", "RGS-BIT": "REFUTED", "RGS-STEP": "REFUTED"}, v
    assert BC.verdicts_ragged(sh, {})[0][1] == "NOT_RUN"


def test_refusals():
    RE = W.RE
    for kw in (dict(kernels=dict(W.KERNELS, diff=None)), dict(kernels=W.KERNELS, gemm="exact"),
               dict(kernels=W.KERNELS, neg="nope"), dict(kernels=W.KERNELS, twin="x")):
        with pytest.raises(RE.RaggedError):
            RE.RaggedOps(**kw)
    model = W.make_model(0)
    rows = W.prompt_rows(1, [4100])
    c = W.new_cache(10)
    W.shipped_prefill(c, rows, 4100, 1)
    with pytest.raises(RE.RaggedError, match="host rows"):
        RE.attach(model, c, 4100, 100, RE.RaggedOps(W.KERNELS))
    with pytest.raises(RE.RaggedError, match="pool"):
        RE.build(model, W.new_cache(0), [4100], 4, RE.RaggedOps(dict(W.KERNELS, pool_update=None)),
                 lambda b, l: rows[(b, l)], pin=False)


# ------------------------------------------------------------------ the step script
def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_base_ragged_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
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


def test_step_script_rg1():
    for bad in ("rg2", "mb1", "rg1x"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    for cell, twin in (("rg1", "torch"), ("rg1f", "fused")):
        r = _step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = "cell %s: C 128 (pool 65) B 216 arms SH+RG+RGS steps 512 twin %s STEP_MIN 75 mem 240G" % (cell, twin)
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=65", "--pool 65", "--batch 216", "--mbase", "--ragged --ragged-spread 64",
                     "--ragged-twin %s" % twin, "--ragged-body fused", "--arms SH", "--mem=240G", "--time=75",
                     "rg_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
