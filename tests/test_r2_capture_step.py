"""CPU tests of the capture-safe decode step (nosi/nosi/r2_capture.py) and of the capture gate's plumbing
(benchmarks/Efficiency/r2_capture_gate.py). Design: docs/R2_CAPTURE_AUDIT.md, "X4 and the capture gate".

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-base python -m pytest -q tests/test_r2_capture_step.py

THE FAKE WORLD (tests/r2_fake_world.py): a tiny fake NOSA model on the CPU (B 4, 2 layers, 2 KV heads of 8, 4 query
heads, 50 tokens; the shipped block 64 and topk 64, prompts of 4100 .. 4223 tokens); the CUDA ops (rmsnorm,
nosa_linear, rope, stage 1, pooling, attention, silu_and_mul, the diff kernel and the gathers) are torch stand-ins, THE
SAME for both paths. So a difference is a difference of the R2 code: the validated eager r2_decode_path.
decode_inference (CE) against r2_capture.decode_step_fixed driven by prepare_step / finish_step (CF), on every
request, every step (the gate's own record_all, tables_digest, r2_g0_compare.compare_views and verdicts).

S1  CF == CE bitwise on every request at every step and in the final tables, ragged activity, 80 steps (several block
    boundaries); refusal bits 0; no state pointer moves after the first step.
S2  NG1 (cb.freeze_widths at step W, the eager twin of the captured control): the first difference is exactly
    r2_capture_gate.ng1_first_step.
S3  NG2 (the device mask not written after step W): the first difference is exactly ng2_first_step.
S4  a driver that skips finish_step: the first difference is step 1.
S5  state_pointers / pointer_diff: a rebound buffer is named.
S6  the gate's pure parts: parse_args refusals, positions_before, ng1 / ng2 on hand cases, digest.
S7  THE FIRST GATE RUN'S FAILURE, REPRODUCED (2188818 step 94: CF and CG differed from CE only in the logits and the
    selection of requests 2 and 7, first at step 0, every state view and table equal). The window rows that no gather
    has written are poisoned with different seeds in CE and CF (on the GPU they are torch.empty memory) and requests 1
    and 3 are inactive at step 0 (late_activity): the raw comparison shows exactly that signature; the gate's
    compare_step (the defined result) is bitwise equal, and every raw difference is before the request's first active
    step (undefined_ok).
S8  negative controls: the same poison in both arms gives no difference even raw; a window corrupted after the
    request's first active step fails undefined_ok at that (step, request); NG2 still differs first exactly at
    ng2_first_step under the blanking.
S9  first_active, mask_inactive, compare_step, undefined_ok on hand cases (with negative controls).
GPU-only (the gate): the capture itself, the replay, the Triton write-back, the real kernels.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
EFF = NOSI_REPO / "benchmarks/Efficiency"
if not (NOSI_REPO / "nosi/nosi/r2_capture.py").is_file() or not (EFF / "r2_capture_gate.py").is_file():
    pytest.skip("no r2_capture.py / r2_capture_gate.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)


def _load_world():
    name = "r2_fake_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / "r2_fake_world.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


FW = _load_world()
RC, RDP, RB, TW, WB = FW.RC, FW.RDP, FW.RB, FW.TW, FW.WB
B, V, S_LIST, STEPS, W = FW.B, FW.V, FW.S_LIST, FW.STEPS, FW.W
STUBS, make_model, build = FW.STUBS, FW.make_model, FW.build
sys.path.insert(0, str(EFF))
import r2_capture_gate as G   # noqa: E402
import r2_g0_compare as C     # noqa: E402


def activity(seed=6):
    return G.RG.activity(STEPS, B, 0.7, seed, True)


def poison_window(cache, seed, request=None):
    """Seeded noise in the attention-window rows 0 .. (topk - 1) * 64 - 1 (slots 0 .. 62) of every engine: on the
    GPU, build_ragged leaves them as torch.empty memory until a request's first active step loads them (the CPU build
    above zeroes them). request None: every request; else only that one."""
    g = torch.Generator().manual_seed(seed)
    for lay in cache.layers:
        e = lay.cache_engine
        rows = (e.topk - 1) * e.block_size
        sel = slice(None) if request is None else slice(request, request + 1)
        for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu):
            t[sel, :rows] = torch.randn(tuple(t[sel, :rows].shape), generator=g) * 3


def late_activity():
    """A hand-made table: request 0 always active; request 1 inactive at steps 0-2 (first active at 3) and 40-44;
    request 3 inactive at steps 0-4 (first active at 5) and 10; request 2 inactive at 7 and 20-22 (after its first
    active step 0)."""
    act = torch.ones((STEPS, B), dtype=torch.bool)
    act[0:3, 1] = False
    act[40:45, 1] = False
    act[0:5, 3] = False
    act[10, 3] = False
    act[7, 2] = False
    act[20:23, 2] = False
    return act


def run_ce(model, act, poison=None):
    cache, st, tok = build(model)
    if poison is not None:
        poison_window(cache, poison)
    recs = []
    cu = torch.arange(B + 1, dtype=torch.int32)
    for s in range(STEPS):
        st.set_active(act[s])
        pos = st.pos_host.view(B, 1).to(torch.int32)
        lg = RDP.decode_inference(model, tok, cu, pos, cache)
        tok = torch.where(st.active.view(B, 1), lg[:, -1].argmax(-1, keepdim=True), tok)
        recs.append(G.record_all(cache, st.twins, lg, torch.stack(st.topk_log)))
    return dict(steps=recs, tables=G.tables_digest(cache, st.twins))


def run_cf(model, act, ref, mode=None, poison=None, plant=None, raw=False):
    """The fixed step with the driver; mode None | 'frozen' (NG1) | 'stale' (NG2) | 'nofinish'. poison: the seed of
    the never-written window rows (poison_window); plant (step, request): overwrite that request's window rows right
    before that step; raw: compare the raw rows (the first gate run's comparison) instead of the gate's
    compare_step (the defined result). Returns (tracker, errors seen, pointer differences after the first step, the
    (step, request) pairs whose raw rows differed while inactive; [] when raw)."""
    tr = C.Tracker()
    undef = []
    with torch.inference_mode():
        cache, st, tok = build(model)
        if poison is not None:
            poison_window(cache, poison)
        cb = RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch)
        errs, ptr0 = [], None
        for s in range(STEPS):
            if plant is not None and s == plant[0]:
                poison_window(cache, 99, plant[1])
            RC.prepare_step(st, cb, tok, act[s], write_active=not (mode == "stale" and s > W))
            if mode == "frozen" and s == W:
                cb.freeze_widths(st)
            RC.decode_step_fixed(model, st, cb, cache)
            if mode != "nofinish":
                RC.finish_step(st)
            body_err, twin_err = RC.errors(st, cb)
            if body_err or any(twin_err):
                errs.append((s, body_err, twin_err))
            if s == 0:
                ptr0 = RC.state_pointers(st, cb, cache)
            tok = torch.where(st.active.view(B, 1), cb.logits[:, -1].argmax(-1, keepdim=True), tok)
            rec = G.record_all(cache, st.twins, cb.logits, cb.topk_log)
            if raw:
                tr.update(s, None, C.compare_views(ref["steps"][s], rec))
            else:
                diffs, u = G.compare_step(ref["steps"][s], rec, act[s])
                tr.update(s, None, diffs)
                undef.extend((s, b) for b in u)
        tr.update(STEPS - 1, None, C.compare_views(ref["tables"], G.tables_digest(cache, st.twins)))
        moved = G.pointer_diff(ptr0, RC.state_pointers(st, cb, cache))
    return tr, errs, moved, undef


@pytest.fixture(scope="module")
def world():
    model = make_model()
    act = activity()
    return model, act, run_ce(model, act)


def test_S1_fixed_step_equals_the_validated_eager_step(world):
    model, act, ref = world
    tr, errs, moved, undef = run_cf(model, act, ref)
    ok, why = C.verdict_equal(tr)
    assert ok, why
    assert errs == [] and moved == [] and undef == []                    # both builds zero the window: no undefined row
    pb = G.positions_before(S_LIST, act)
    crossed = sum(int(((pb[-1] // 64) > (pb[0] // 64))[b]) for b in range(B))
    assert crossed >= 3, "the run must cross block boundaries (crossed %d)" % crossed
    assert not bool(act.all()) and bool(act[:, 0].all())


def test_S2_ng1_frozen_widths_first_differs_where_predicted(world):
    model, act, ref = world
    s0 = G.ng1_first_step(S_LIST, act, W)
    assert s0 is not None
    tr, _, _, _ = run_cf(model, act, ref, mode="frozen")
    ok, why = C.verdict_neg(tr, s0, None)
    assert ok, why


def test_S3_ng2_stale_mask_first_differs_where_predicted(world):
    model, act, ref = world
    s0 = G.ng2_first_step(act, W)
    assert s0 is not None
    tr, _, _, _ = run_cf(model, act, ref, mode="stale")
    ok, why = C.verdict_neg(tr, s0, None)
    assert ok, why


def test_S4_a_driver_without_finish_step_differs_at_step_1(world):
    model, act, ref = world
    tr, _, _, _ = run_cf(model, act, ref, mode="nofinish")
    ok, why = C.verdict_neg(tr, 1, None)
    assert ok, why


@pytest.fixture(scope="module")
def late_world():
    """late_activity() and the CE reference with the never-written window rows poisoned (seed 11), as on the GPU."""
    model = make_model()
    act = late_activity()
    return model, act, run_ce(model, act, poison=11)


def test_S7_the_first_gate_failure_reproduced_inactive_rows_before_the_first_active_step(late_world):
    """The first capture gate run (2188818, step 94): CF and CG differed from CE ONLY in the logits / selection of
    requests 2 and 7, first at step 0, every state view and table equal. Cause: those requests were inactive at step 0
    and their attention read window rows that no gather had written (torch.empty memory, different in each arm)."""
    model, act, ref = late_world
    fa = G.first_active(act)
    assert fa == [0, 3, 0, 5] and act[0].tolist() == [True, False, True, False]
    # THE REPRODUCTION: the first run's raw comparison, another garbage (seed 12) in CF's window
    tr, errs, _, _ = run_cf(model, act, ref, poison=12, raw=True)
    assert errs == []
    assert tr.first_step() == 0 and tr.first["logits"][0] == 0 and tr.first["logits"][2] == [1, 3], tr.first
    assert set(tr.first) <= {"logits", "topk"}, tr.first                 # no state view, no table differs
    assert tr.ever["logits"] <= {1, 3} and tr.ever.get("topk", set()) <= {1, 3}
    # THE FIX: the gate's compare_step: the DEFINED result is bitwise equal; the raw differences are only on inactive
    # rows before the request's first active step (CF-UNDEF)
    tr, errs, moved, undef = run_cf(model, act, ref, poison=12)
    ok, why = C.verdict_equal(tr)
    assert ok, why
    assert errs == [] and moved == []
    assert {b for _, b in undef} == {1, 3} and all(s < fa[b] for s, b in undef), undef
    assert G.undefined_ok(undef, act)[0]
    # the inactive steps AFTER a first active step (request 1 at 40-44, 2 at 7 and 20-22, 3 at 10) gave equal raw rows
    assert not [p for p in undef if p[0] >= 5]


def test_S8_negative_controls_of_the_undefined_rows(late_world):
    model, act, ref = late_world
    # NC1: the SAME garbage in both arms: no difference at all, even raw (the code paths agree on every row)
    tr, _, _, _ = run_cf(model, act, ref, poison=11, raw=True)
    ok, why = C.verdict_equal(tr)
    assert ok, why
    # NC2: request 3's window overwritten at step 10 (inactive, AFTER its first active step 5): CF-UNDEF fails and
    # names (10, 3)
    _, _, _, undef = run_cf(model, act, ref, poison=12, plant=(10, 3))
    ok, why = G.undefined_ok(undef, act)
    assert not ok and (10, 3) in undef, (why, undef)
    # NC3: blanking does not hide an ACTIVE row: NG2 (the stale device mask) still differs first exactly at
    # ng2_first_step (request 1 becomes active at step 3; the device keeps it inactive and its row is compared)
    s0 = G.ng2_first_step(act, W)
    assert s0 == 3
    tr, _, _, _ = run_cf(model, act, ref, mode="stale", poison=12)
    ok, why = C.verdict_neg(tr, s0, None)
    assert ok, why
    assert 1 in tr.ever["logits"]


def test_S5_pointer_snapshot_names_a_rebound_buffer():
    model = make_model()
    with torch.inference_mode():
        cache, st, tok = build(model)
        cb = RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch)
        RC.prepare_step(st, cb, tok, torch.ones(B, dtype=torch.bool))
        RC.decode_step_fixed(model, st, cb, cache)
        before = RC.state_pointers(st, cb, cache)
        assert G.pointer_diff(before, RC.state_pointers(st, cb, cache)) == []
        st.pq = st.pq.clone()                                           # a planted rebinding
        cache.layers[1].cache_engine._cache_lens = cache.layers[1].cache_engine._cache_lens.clone()
        assert G.pointer_diff(before, RC.state_pointers(st, cb, cache)) == ["eng1._cache_lens", "st.pq"]
    with pytest.raises(RDP.R2Error):
        RC.CaptureBuffers(model, st, False, STUBS, WB.masked_writeback_torch, ng="unknown")


def test_S6_gate_pure_parts():
    base = ["--out", "x", "--workload", "w"]
    for bad in (["--arms", "CF,CE"], ["--arms", "CE,ZZ"], ["--deltas", "0,1"], ["--deltas", "1,0,0,0,0,0,0,0"],
                ["--warmup", "0"], ["--warmup", "192"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    a = G.parse_args(base)
    assert (a.batch, a.steps, a.warmup, a.arms) == (8, 192, 2, list(G.ARMS))
    assert G.prompt_lengths(a) == [16128 - d for d in a.deltas]
    act = torch.tensor([[True, True], [True, False], [True, True], [False, True]])
    assert G.positions_before([10, 20], act).tolist() == [[10, 20], [11, 21], [12, 21], [13, 22]]
    act2 = torch.ones((70, 2), dtype=torch.bool)
    assert G.ng1_first_step([63 * 64 + 60, 100 * 64], act2, 2) == 4          # request 0 reaches 64 * 64 at step 4
    act4 = act2.clone()
    act4[4, 0] = False                                   # its new width at step 4 is not observable: inactive there
    assert G.ng1_first_step([63 * 64 + 60, 100 * 64], act4, 2) == 5
    assert G.ng2_first_step(act, 1) == 2 and G.ng2_first_step(act2, 2) is None
    x = torch.arange(12, dtype=torch.float32)
    assert G.digest(x) == G.digest(x.clone()) and G.digest(x) != G.digest(x + 1)
    assert G.pointer_diff({"a": 1, "b": 2}, {"a": 1, "c": 3}) == ["b", "c"]


def test_S9_undefined_rows_pure_parts():
    act3 = torch.tensor([[True, False, False], [True, False, True], [True, True, False]])
    assert G.first_active(act3) == [0, 2, 1]
    assert G.first_active(torch.zeros((2, 1), dtype=torch.bool)) == [None]
    assert G.undefined_ok([(0, 1), (1, 1), (0, 2)], act3)[0]
    ok, why = G.undefined_ok([(0, 1), (2, 1)], act3)
    assert not ok and "(2, 1)" in why
    assert G.undefined_ok([(1, 0)], torch.zeros((2, 1), dtype=torch.bool))[0]   # never active: always undefined
    lg = torch.zeros((3, 5))
    tk = torch.zeros((1, 1, 3, 2), dtype=torch.int64)
    ref = dict(logits=(lg, 0), topk=(tk, 2), counters=(torch.zeros(3), 0))
    # an INACTIVE row differs (even by a NaN): not a difference of the defined result, reported as undefined
    lg2 = lg.clone()
    lg2[1, 0] = float("nan")
    tk2 = tk.clone()
    tk2[0, 0, 2, 1] = 5
    assert G.compare_step(ref, dict(ref, logits=(lg2, 0), topk=(tk2, 2)), act3[0]) == ({}, [1, 2])
    # NEGATIVE CONTROLS: an ACTIVE row's logits, an active row's selection, a state view of an inactive request: each
    # is a difference, and none is reported as undefined
    lg3 = lg.clone()
    lg3[0, 4] = 1.0
    assert G.compare_step(ref, dict(ref, logits=(lg3, 0)), act3[0]) == ({"logits": [0]}, [])
    tk3 = tk.clone()
    tk3[0, 0, 0, 0] = 7
    assert G.compare_step(ref, dict(ref, topk=(tk3, 2)), act3[0]) == ({"topk": [0]}, [])
    assert G.compare_step(ref, dict(ref, counters=(torch.tensor([0., 1., 0.]), 0)), act3[0]) == ({"counters": [1]}, [])
    # the blanking copies: the recorded views are not modified
    m = G.mask_inactive(dict(ref, logits=(lg2, 0)), act3[0])
    assert torch.isnan(lg2[1, 0]) and bool((m["logits"][0][1] == 0).all()) and m["counters"][0] is ref["counters"][0]
