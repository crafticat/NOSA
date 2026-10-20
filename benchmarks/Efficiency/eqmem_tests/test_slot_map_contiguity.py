"""The slot-map stride defect of the speculative loop, fixed 2026-10-20 (the spec arm of the equal-memory harness runs
cache_engine.spec_draft_update / spec_round_update through the pilot).

spec_loop._assign returned ext[..., :W], a view with rows of W + 1, as plan_draft's issue map; spec_draft_update hands
it to flash_h2d_from_mask / flash_h2d_from_mask_bias, whose kernels read entry (h, b, m) at h * B * M + b * M + m from
the data pointer (flash_h2d_mask.py:34, flash_h2d_mask_bias.py:34). Every (h, b) row but (0, 0) was read shifted: the
draft's prefetches landed in slots the store did not name (window, mirror and tail slots included).

  EMX1  spec_draft_update (the source, executed against a fake engine) with gathers that ADDRESS THE MAP AS THE KERNELS
        DO: every prefetch holds the host rows of the block the plan named, slot 63 keeps its rows; NEGATIVE CONTROL:
        the pre-fix map (the same values as a narrowed view) misplaces blocks.
  EMX2  the maps handed to the gathers are contiguous: plan_draft's issue (the fix) and plan_round's fetch (contiguous
        already before the fix: torch.where rebuilt it, so the verify's late fetch was not affected).
  EMX3  both gather wrappers refuse a non-contiguous map (the first statement; checked by AST and by a call that stops
        before any launch).

From the worktree root, on a CPU node:
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \\
        "benchmarks/Efficiency/eqmem_tests/test_slot_map_contiguity.py::<node>"
"""
from __future__ import annotations

import ast
import importlib.util
import textwrap
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
FCE = ROOT / "nosi/nosi/flash_cache_engine"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sl = _load(ROOT / "nosi/nosi/spec_loop.py", "nosi_spec_loop_contiguity_test")
SRC_ENGINE = (ROOT / "nosi/nosi/cache_engine.py").read_text()
BS, TOPK, R, H, D, KSEL = 64, 8, 5, 2, 4, 6
LAY = sl.store_layout(TOPK, R)


def _flat_ids(ids):
    """What the kernels read: entry (h, b, m) at h * B * M + b * M + m from the data pointer."""
    Hn, Bn, Mn = ids.shape
    return torch.as_strided(ids, (Hn * Bn * Mn,), (1,), ids.storage_offset())


def _kernel_gather_k(gpu, cpu, ids, bs):
    Hn, Bn, Mn = ids.shape
    flat = _flat_ids(ids)
    for h in range(Hn):
        for b in range(Bn):
            for m in range(Mn):
                blk = int(flat[h * Bn * Mn + b * Mn + m])
                if blk >= 0:
                    gpu[b, m * bs:(m + 1) * bs, h] = cpu[b, blk * bs:(blk + 1) * bs, h]


def _kernel_gather_v(gpu, cpu, bias_buf, bias, ids, bs):
    _kernel_gather_k(gpu, cpu, ids, bs)
    Hn, Bn, Mn = ids.shape
    flat = _flat_ids(ids)
    for h in range(Hn):
        for b in range(Bn):
            for m in range(Mn):
                blk = int(flat[h * Bn * Mn + b * Mn + m])
                if blk >= 0:
                    bias_buf[b, m * bs:(m + 1) * bs, h] = bias[b, blk * bs:(blk + 1) * bs, h].to(bias_buf.dtype)


def _old_assign(orig):
    """The pre-fix _assign: the same values as a narrowed (H, B, W + 1) view."""
    def f(want, ids, cand, key):
        issue, fits, n_cand = orig(want, ids, cand, key)
        Hn, Bn, Wn = issue.shape
        ext = torch.full((Hn, Bn, Wn + 1), -1, dtype=issue.dtype)
        ext[..., :Wn] = issue
        return ext[..., :Wn], fits, n_cand
    return f


def _method(name):
    tree = ast.parse(SRC_ENGINE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CacheEngine")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC_ENGINE, fn))


class _Event:
    def record(self, stream=None):
        pass

    def query(self):
        return False


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ints(shape, g):
    return torch.randint(-8, 9, tuple(shape), generator=g).to(torch.float32)


def _run_draft(seed, steps=4, B=4):
    """steps draft steps of the LOCKSTEP spec_draft_update (the equal-memory spec arm's path) with kernel-addressed
    gathers. Returns (misplaced prefetches, requests whose slot-63 rows below the tail length changed, issued)."""
    g = torch.Generator().manual_seed(seed)
    pos = 9 * BS + 10
    T, tl0 = pos // BS, pos % BS
    n_host = pos + 2 * BS
    eng = types.SimpleNamespace(block_size=BS, topk=TOPK, head_num=H, head_dim=D, _tail_block_idx_on_gpu=TOPK - 1,
                                verify_round_slots=R, pool_blocks=0, device=None, seq_length=pos, _tail_block_len_on_gpu=tl0)
    eng._k_gpu, eng._v_gpu = _ints((B, LAY.W * BS, H, D), g).bfloat16(), _ints((B, LAY.W * BS, H, D), g).bfloat16()
    eng._kv_bias_gpu = _ints((B, LAY.W * BS, H), g).bfloat16()
    eng._k_cpu, eng._v_cpu = _ints((B, n_host, H, D), g).bfloat16(), _ints((B, n_host, H, D), g).bfloat16()
    eng._block_map = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    for b in range(B):
        for h in range(H):
            eng._block_map[h, b, :TOPK - 1] = torch.randperm(T, generator=g)[:TOPK - 1]
    eng._block_map[..., TOPK - 1] = T
    eng._cache_lens = torch.full((B,), (TOPK - 1) * BS + tl0, dtype=torch.int32)
    ns = {"torch": torch, "_sl": sl, "_bias_rows": lambda x: x,
          "flash_h2d_from_mask": _kernel_gather_k, "flash_h2d_from_mask_bias": _kernel_gather_v}
    exec(_method("spec_draft_update"), ns)
    eng.spec_draft_update = types.MethodType(ns["spec_draft_update"], eng)
    st = sl.Store(1, H, B, LAY)
    side = types.SimpleNamespace(wait_event=lambda e: None)
    ctx = sl.SpecContext(st, side=side, make_event=_Event, current_stream=lambda: "main", stream_ctx=lambda s: _Null(),
                         num_layers=1, block_size=BS, kernel_size=32, kernel_stride=16)
    st.sync_window(0, eng._block_map, ctx.next_tick())
    ctx.tail_id, ctx.rolled, ctx.gen_round0 = T, [False], ctx.gen + 1
    ts = LAY.tail_slot * BS
    rows0 = eng._k_gpu[:, ts:ts + tl0].clone()
    kvb = _ints((B, n_host, H), g)
    bad = issued = 0
    for j in range(steps):
        ctx.next_tick()
        sel = torch.stack([torch.stack([torch.randperm(T + 1, generator=g)[:KSEL] for _ in range(B)]) for _ in range(H)])
        eng.spec_draft_update(_ints((B, 1, H, D), g), _ints((B, 1, H, D), g), kvb, sel, ctx, 0)
        issue = eng._spec_last_draft.issue
        for h, b, w in (issue >= 0).nonzero().tolist():
            issued += 1
            blk = int(issue[h, b, w])
            if not torch.equal(eng._k_gpu[b, w * BS:(w + 1) * BS, h], eng._k_cpu[b, blk * BS:(blk + 1) * BS, h]):
                bad += 1
    tail_hit = sum(1 for b in range(B) if not torch.equal(eng._k_gpu[b, ts:ts + tl0], rows0[b]))
    return bad, tail_hit, issued


def test_EMX1_draft_prefetch_lands_in_the_slots_the_plan_names_with_the_kernels_addressing(monkeypatch):
    bad, tail_hit, issued = _run_draft(7)
    assert issued > 0 and bad == 0 and tail_hit == 0, (bad, tail_hit, issued)
    monkeypatch.setattr(sl, "_assign", _old_assign(sl._assign))      # NEGATIVE CONTROL: the pre-fix map
    bad2, tail_hit2, issued2 = _run_draft(7)
    assert bad2 > 0, (bad2, tail_hit2, issued2)


def test_EMX2_the_maps_handed_to_the_gathers_are_contiguous_and_the_verify_fetch_was_before_the_fix(monkeypatch):
    g = torch.Generator().manual_seed(8)
    B = 3
    st = sl.Store(1, H, B, LAY)
    bm = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    for b in range(B):
        for h in range(H):
            bm[h, b, :TOPK - 1] = torch.randperm(10, generator=g)[:TOPK - 1]
    bm[..., TOPK - 1] = 10
    st.sync_window(0, bm, 1)
    sel = torch.stack([torch.stack([torch.randperm(11, generator=g)[:KSEL] for _ in range(B)]) for _ in range(H)])
    for old in (False, True):
        if old:
            monkeypatch.setattr(sl, "_assign", _old_assign(sl._assign))
        s1 = sl.Store(1, H, B, LAY)
        for n in ("ids", "state", "gen", "stamp", "used") + st.COUNTERS:
            getattr(s1, n).copy_(getattr(st, n))
        d = sl.plan_draft(s1, 0, sel, 10, False, 1, 2, 10)
        s1.mark_arrived(10)
        r = sl.plan_round(s1, 0, torch.stack([sel, sel]), 10, 2, 1, 4, 5, torch.zeros((H, B, LAY.W), dtype=torch.bool))
        assert r.fetch.is_contiguous()                                    # the verify's late fetch: safe before and after
        assert d.issue.is_contiguous() == (not old)                       # the draft's prefetch: the defect and the fix
        flat = _flat_ids(d.issue).view(H, B, LAY.W)
        assert torch.equal(flat, d.issue) == (not old)                    # what the kernels read vs what the plan wrote


def test_EMX3_the_gather_wrappers_refuse_a_non_contiguous_map():
    for fname, fn in (("flash_h2d_mask.py", "flash_h2d_from_mask"), ("flash_h2d_mask_bias.py", "flash_h2d_from_mask_bias")):
        src = (FCE / fname).read_text()
        f = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        assert isinstance(f.body[0], ast.Assert) and "load_ids.is_contiguous()" in ast.get_source_segment(src, f.body[0]), fn
    m1 = _load(FCE / "flash_h2d_mask.py", "nosi_flash_h2d_mask_guard_eqmem")         # imports triton; nothing is launched
    m2 = _load(FCE / "flash_h2d_mask_bias.py", "nosi_flash_h2d_mask_bias_guard_eqmem")
    view = torch.full((H, 2, LAY.W + 1), -1, dtype=torch.int64)[..., :LAY.W]
    gpu, cpu = torch.zeros((2, LAY.W * BS, H, D)), torch.zeros((2, 4 * BS, H, D))
    with pytest.raises(AssertionError):
        m1.flash_h2d_from_mask(gpu, cpu, view, BS)
    with pytest.raises(AssertionError):
        m2.flash_h2d_from_mask_bias(gpu, cpu, torch.zeros((2, LAY.W * BS, H)), torch.zeros((2, 4 * BS, H)), view, BS)
