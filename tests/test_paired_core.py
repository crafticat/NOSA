"""CPU gate of the PAIRED TICK's core (nosi/nosi/paired/core.py) and of the
importability / text of its GPU body (tick.py, twin.py). Spec: retroinfer-eval
docs/superpowers/specs/2026-09-20-paired-tick.md sections 1-3, gates G1-G8.

    srun -p amd -n1 -c4 --mem=16G --time=00:20:00 apptainer exec \\
        --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi /mnt/central/users/ojerbi/manar28.sif \\
        python -m pytest -q /mnt/beegfs/ojerbi/nosi-paired/tests/test_paired_core.py

No CUDA, no extension: the package is imported under a STUB ``nosi`` package
(the real nosi/__init__.py imports the model module, which imports the CUDA
extensions at module level), so ``nosi.paired.core``'s relative imports of
verify/rows_attention.py, verify/tail_write.py and spec_loop.py resolve to
the fork's pure modules. AGENTS.md: torch.equal is the standard, positive
controls, poison what must not be read, prove the code path executed.

Small dims: block_size 64 (so the tail rollover is tested at the literal
tail lengths 62 / 63 / 64), topk 4 (tail slot 3), R = 3 round slots -> W = 9:
provisional slot 4, ring 5..6, mirrors 7, 8; H = 2 KV heads, Hq = 4, D = 8.
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOSI_PKG = Path(os.environ.get("NOSI_PAIRED_PKG", ROOT / "nosi" / "nosi"))
if not (NOSI_PKG / "paired" / "core.py").is_file():
    pytest.skip("no %s/paired/core.py" % NOSI_PKG, allow_module_level=True)


def _bootstrap():
    if getattr(sys.modules.get("nosi"), "__paired_stub__", False):
        return
    for name in [m for m in list(sys.modules) if m == "nosi" or m.startswith("nosi.")]:
        del sys.modules[name]
    pkg = types.ModuleType("nosi")
    pkg.__path__ = [str(NOSI_PKG)]
    pkg.__paired_stub__ = True
    sys.modules["nosi"] = pkg


_bootstrap()
core = importlib.import_module("nosi.paired.core")
tick = importlib.import_module("nosi.paired.tick")
twin = importlib.import_module("nosi.paired.twin")
ra = importlib.import_module("nosi.verify.rows_attention")
tw = importlib.import_module("nosi.verify.tail_write")
sl = importlib.import_module("nosi.spec_loop")

BS, TOPK, R = 64, 4, 3
H, HQ, D = 2, 4, 8
LAY = core.paired_layout(TOPK, R, BS)
W = LAY.W


# ---------------------------------------------------------------------------
# a fake engine with the CacheEngine attributes the core touches
# ---------------------------------------------------------------------------
class FakeEngine:
    def __init__(self, B=3, tail_len=5, T=40, seed=0, host_len=6000):
        g = torch.Generator().manual_seed(seed)
        self.block_size, self.topk, self.head_num, self.head_dim = BS, TOPK, H, D
        self.verify_round_slots, self.pool_blocks = R, 0
        self._k_gpu = torch.randn((B, W * BS, H, D), generator=g)
        self._v_gpu = torch.randn((B, W * BS, H, D), generator=g)
        self._kv_bias_gpu = torch.randn((B, W * BS, H), generator=g)
        self._k_cpu = torch.randn((B, host_len, H, D), generator=g)
        self._v_cpu = torch.randn((B, host_len, H, D), generator=g)
        self._block_map = torch.full((H, B, TOPK), -1, dtype=torch.int64)
        for h in range(H):
            for b in range(B):
                self._block_map[h, b, :TOPK - 1] = torch.tensor([3 + h, 10 + b, 20 + h + b])   # distinct window ids
        self._block_map[..., TOPK - 1] = T
        self.seq_length = T * BS + tail_len
        self._tail_block_len_on_gpu = tail_len
        self._tail_block_idx_on_gpu = TOPK - 1
        self._cache_lens = torch.full((B,), (TOPK - 1) * BS + tail_len, dtype=torch.int32)
        self.B, self.T = B, T


def _sel(H_, n, ids):
    """(H, n, K) int64 selection rows, the same ids on every (h, b)."""
    return torch.tensor(ids, dtype=torch.int64).view(1, 1, -1).expand(H_, n, -1).contiguous()


# ---------------------------------------------------------------------------
# layout, row plan
# ---------------------------------------------------------------------------
def test_layout_and_engine_check():
    assert (LAY.tail_slot, LAY.prov_slot, LAY.ring_lo, LAY.ring_hi, LAY.mirror_lo, LAY.mirror_hi, LAY.W) == (3, 4, 5, 7, 7, 8, 9)
    with pytest.raises(ValueError):
        core.paired_layout(TOPK, 0, BS)          # no provisional slot without a round slot
    eng = FakeEngine()
    core.check_layout_against_engine(LAY, eng)
    eng.pool_blocks = 1
    with pytest.raises(ValueError):
        core.check_layout_against_engine(LAY, eng)


def test_row_plan_adjacent_and_subset():
    p = core.row_plan(torch.arange(3), True, True)
    assert p.U == 2 and p.req.tolist() == [0, 0, 1, 1, 2, 2] and p.role.tolist() == [0, 1, 0, 1, 0, 1]
    t = core.row_plan(torch.arange(3), True, False)
    assert t.U == 1 and t.role.tolist() == [0, 0, 0]
    s = core.row_plan(torch.tensor([2, 0]), False, True)
    assert s.U == 1 and s.req.tolist() == [2, 0] and s.role.tolist() == [1, 1]
    with pytest.raises(ValueError):
        core.row_plan(torch.arange(2), False, False)


# ---------------------------------------------------------------------------
# the state machine (G8): accept / reject / restart sequences, lockstep
# ---------------------------------------------------------------------------
def test_state_machine_sequence():
    B = 4
    st = core.DraftState(B)
    fallback = torch.tensor([7, 7, 7, 7])
    assert st.s_inputs(fallback).tolist() == [7, 7, 7, 7]              # no draft: the S row runs on the fallback, rejected at tick end
    out = st.tick_end(committed=torch.tensor([1, 2, 3, 4]), s_argmax=torch.tensor([11, 12, 13, 14]))
    assert out.accepted.tolist() == [False] * 4 and out.had_draft.tolist() == [False] * 4
    assert out.restart_idx.tolist() == [0, 1, 2, 3] and out.next_draft.tolist() == [-1] * 4
    st.apply_restart(out.restart_idx, torch.tensor([21, 22, 23, 24]))  # the restart's outputs become the drafts
    assert st.draft.tolist() == [21, 22, 23, 24]
    # tick 2: requests 0 and 2 committed what they drafted -> accept, their S outputs become the drafts
    out = st.tick_end(committed=torch.tensor([21, 99, 23, 98]), s_argmax=torch.tensor([31, 32, 33, 34]))
    assert out.accepted.tolist() == [True, False, True, False] and out.had_draft.tolist() == [True] * 4
    assert out.next_draft.tolist() == [31, -1, 33, -1] and out.restart_idx.tolist() == [1, 3]
    assert out.committed.tolist() == [21, 99, 23, 98]                  # G8: one committed token per request, whatever S did
    with pytest.raises(AssertionError):
        st.apply_restart(torch.tensor([0]), torch.tensor([5]))         # request 0 still has a draft
    st.apply_restart(out.restart_idx, torch.tensor([41, 43]))
    assert st.draft.tolist() == [31, 41, 33, 43]
    with pytest.raises(ValueError):
        st.tick_end(committed=torch.tensor([1, -1, 3, 4]), s_argmax=torch.zeros(4, dtype=torch.int64))   # not a token
    assert core.lockstep_check([torch.zeros(B, dtype=torch.int64)] * 5, B) == 5
    with pytest.raises(AssertionError):
        core.lockstep_check([torch.zeros(B - 1, dtype=torch.int64)], B)


# ---------------------------------------------------------------------------
# accounting identities
# ---------------------------------------------------------------------------
def test_set_counts_and_bytes():
    a = torch.tensor([[[1, 2, 3, -1]]])
    b = torch.tensor([[[3, 4, -1, -1]]])
    a_not_b, b_not_a, both = core.set_counts(a, b)
    assert (int(a_not_b), int(b_not_a), int(both)) == (2, 1, 1)
    assert int(a_not_b + both) == int((a >= 0).sum()) and int(b_not_a + both) == int((b >= 0).sum())   # identities
    assert core.blocks_to_bytes(3) == 3 * 32768 and core.BYTES_PER_BLOCK == 32768 == sl.BYTES_WIRE_PER_BLOCK
    # isolation: stream (h, b) never reads stream (h', b')
    a2 = torch.tensor([[[1, 2], [5, 6]]])
    b2 = torch.tensor([[[9, 8], [5, 6]]])
    x, y, z = core.set_counts(a2, b2)
    assert x.tolist() == [[2, 0]] and y.tolist() == [[2, 0]] and z.tolist() == [[0, 2]]


def test_ledger_summary_sums_and_rates():
    L = core.Ledger()
    n = 3
    acct = core.LayerAccount(v_miss=torch.tensor([[1, 2, 0], [0, 1, 1]]), s_sel=torch.full((H, n), 3), s_hit=torch.full((H, n), 2),
                             div_s_not_v=torch.ones(H, n, dtype=torch.int64), div_v_not_s=torch.zeros(H, n, dtype=torch.int64),
                             div_both=torch.full((H, n), 2), tail_len_after=6, filled=False)
    L.add_tick(0, [acct, acct], accepted=torch.tensor([True, False, True]), greedy_agree=torch.tensor([True, True, True]), n_restart=1, ms=10.0)
    L.add_tick(1, [acct, acct], accepted=torch.tensor([True, True, True]), greedy_agree=None, n_restart=0, ms=11.0)
    L.add_restart(0, rows=1, ms=5.0, layers=[acct, acct])
    s = L.summary()
    assert s["ticks"] == 2 and s["restarts"] == 1 and s["restart_rows"] == 1
    assert s["v_miss_total"] == 2 * 2 * 5 and s["v_miss_bytes"] == 20 * 32768
    assert s["accepted"] == 5 and s["rows"] == 6 and abs(s["acceptance_rate"] - 5 / 6) < 1e-12
    assert s["greedy_agree"] is None                    # one tick had no record: no number is invented
    both, snv = s["div_both_total"], s["div_s_not_v_total"]
    assert abs(s["prefetch_precision"] - both / (both + snv)) < 1e-12 and s["prefetch_recall"] == 1.0
    assert len(s["v_miss_per_layer_mean"]) == 2


# ---------------------------------------------------------------------------
# slot ids, readiness (G4), visible rows, canonical order (G2)
# ---------------------------------------------------------------------------
def _ids_ready(eng, ring_ids=None, ring_ready=None):
    content = eng._block_map[..., LAY.tail_slot].clone()
    ids = core.slot_ids(eng._block_map, content, LAY, ring_ids=ring_ids)
    ready = core.ready_mask(H, eng.B, LAY, ring_ready=ring_ready)
    return ids, ready


def test_v_visible_rows_is_the_decode_window():
    vis = core.v_visible_rows(H, 2, LAY, tail_len_after=6)
    assert vis.shape == (2, W, H)
    assert (vis[:, :LAY.tail_slot] == BS).all() and (vis[:, LAY.tail_slot] == 6).all() and (vis[:, LAY.tail_slot + 1:] == 0).all()
    vis0 = core.v_visible_rows(H, 2, LAY, tail_len_after=0)
    assert (vis0[:, LAY.tail_slot] == 0).all()                       # the fill step attends (topk-1)*bs rows, as the S == 1 decode does
    with pytest.raises(ValueError):
        core.v_visible_rows(H, 2, LAY, tail_len_after=BS)


def test_s_visible_rows_selection_tail_and_provisional():
    eng = FakeEngine(B=2, tail_len=6)
    ids, ready = _ids_ready(eng)
    # every (h, b) selects its window ids of slots 0 and 1 (3+h, 10+b) and the tail T; slot 2 (20+h+b) is not selected
    sel = torch.stack([torch.stack([torch.tensor([3 + h, 10 + b, eng.T, -1]) for b in range(2)], 0) for h in range(H)], 0)   # (H, B, K)
    vis = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=6)
    assert vis.shape == (2, W, H)
    for h in range(H):
        assert vis[0, 0, h] == BS and vis[0, 1, h] == BS and vis[0, 2, h] == 0          # slot 2 holds 20+h+0, not selected
        assert vis[0, LAY.tail_slot, h] == 6 and vis[0, LAY.prov_slot, h] == 1
        assert (vis[0, LAY.ring_lo:LAY.ring_hi, h] == 0).all() and vis[0, LAY.mirror_lo, h] == 0 and vis[0, LAY.mirror_hi, h] == 0
    # the tail slot is named by id: a selection without T sees 0 tail rows; the provisional row is S's own, always
    sel2 = sel.clone(); sel2[..., 2] = -1
    vis2 = core.s_visible_rows(ids, ready, sel2, LAY, tail_rows=6)
    assert (vis2[:, LAY.tail_slot] == 0).all() and (vis2[:, LAY.prov_slot] == 1).all()
    # after a fill the slot physically holds T complete: tail_rows = bs
    vis3 = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=BS)
    assert (vis3[:, LAY.tail_slot] == BS).all()


def test_g4_not_ready_slot_is_excluded():
    eng = FakeEngine(B=1)
    ring_ids = torch.full((H, 1, LAY.ring_hi - LAY.ring_lo), -1, dtype=torch.int64)
    ring_ids[..., 0] = 77                                                  # block 77 sits in ring slot 5
    sel = _sel(H, 1, [77, 3, -1, -1])
    not_ready = torch.zeros((H, 1, LAY.ring_hi - LAY.ring_lo), dtype=torch.bool)
    ids, ready = _ids_ready(eng, ring_ids, not_ready)
    vis = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=5)
    assert (vis[0, LAY.ring_lo] == 0).all(), "a slot whose event is not complete must be excluded"
    is_ready = not_ready.clone(); is_ready[..., 0] = True                  # positive control: the same slot, event complete
    ids, ready = _ids_ready(eng, ring_ids, is_ready)
    vis = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=5)
    assert (vis[0, LAY.ring_lo] == BS).all()
    assert (ready[..., LAY.mirror_lo] == False).all() and (ready[..., LAY.mirror_hi] == False).all()   # mirrors never


def test_g2_canonical_order_deterministic_and_batch_independent():
    eng = FakeEngine(B=3, seed=1)
    ids, ready = _ids_ready(eng)
    sel = _sel(H, 3, [20, 3, 10, eng.T])
    vis_a = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=5)
    vis_b = core.s_visible_rows(ids.clone(), ready.clone(), sel.clone(), LAY, tail_rows=5)
    assert torch.equal(vis_a, vis_b)
    order = core.canonical_order(ids[0, 0], vis_a[0, :, 0])
    ids_in_order = ids[0, 0][order].tolist()
    named = [i for i in ids_in_order if i >= 0]
    assert named == sorted(named), "ascending block id"
    assert order.tolist()[-1] == LAY.prov_slot, "the id-less provisional slot sorts last"
    # a neighbour's state does not enter request 0's rows (isolation / float-order law)
    eng2 = FakeEngine(B=3, seed=1)
    eng2._block_map[:, 1, :LAY.tail_slot] = torch.tensor([90, 91, 92])
    ids2, ready2 = _ids_ready(eng2)
    vis_c = core.s_visible_rows(ids2, ready2, sel, LAY, tail_rows=5)
    assert torch.equal(vis_c[0], vis_a[0]) and torch.equal(vis_c[2], vis_a[2]) and not torch.equal(vis_c[1], vis_a[1])


def test_assemble_rows_interleaves_v_then_s():
    plan = core.row_plan(torch.arange(2), True, True)
    vis_v = core.v_visible_rows(H, 2, LAY, 5)
    vis_s = torch.zeros((2, W, H), dtype=torch.int64); vis_s[:, LAY.prov_slot] = 1
    vis, cbi = core.assemble_rows(plan, vis_v, vis_s, torch.arange(2))
    assert vis.shape == (4, W, H) and cbi.tolist() == [0, 0, 1, 1] and cbi.dtype == torch.int32
    assert torch.equal(vis[0], vis_v[0]) and torch.equal(vis[1], vis_s[0]) and torch.equal(vis[2], vis_v[1]) and torch.equal(vis[3], vis_s[1])
    sub = core.row_plan(torch.tensor([2, 0]), False, True)
    vis2, cbi2 = core.assemble_rows(sub, None, vis_s, torch.tensor([2, 0]))
    assert cbi2.tolist() == [2, 0]


# ---------------------------------------------------------------------------
# the rows bias (G7: finite mask; exact extent; in-place storage)
# ---------------------------------------------------------------------------
def test_paired_bias_values_extent_and_inplace():
    eng = FakeEngine(B=2, tail_len=6)
    plan = core.row_plan(torch.arange(2), True, True)
    ids, ready = _ids_ready(eng)
    sel = _sel(H, 2, [3, 4, 10, eng.T])
    vis_v = core.v_visible_rows(H, 2, LAY, 6)
    vis_s = core.s_visible_rows(ids, ready, sel, LAY, tail_rows=6)
    vis, cbi = core.assemble_rows(plan, vis_v, vis_s, torch.arange(2))
    rb = core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2)
    assert isinstance(rb, ra.RowsBias) and rb.bias.shape == (4, W * BS, H) and rb.bias.is_contiguous()
    assert core.MASKED == -3.0e4 and torch.isfinite(rb.bias).all(), "G7: the masked value is finite"
    # V row 0: cis of request 0 on window rows and tail rows < 6, MASKED above; nothing past the tail slot
    b0 = rb.bias[0].view(W, BS, H)
    assert torch.equal(b0[:LAY.tail_slot], eng._kv_bias_gpu[0].view(W, BS, H)[:LAY.tail_slot])
    assert torch.equal(b0[LAY.tail_slot, :6], eng._kv_bias_gpu[0].view(W, BS, H)[LAY.tail_slot, :6]) and (b0[LAY.tail_slot, 6:] == core.MASKED).all()
    assert (b0[LAY.prov_slot:] == core.MASKED).all()
    assert int(rb.cache_seqlens[0]) == LAY.tail_slot * BS + 6, "the exact extent = the decode's _cache_lens"
    # S row 1 (request 0): its provisional row visible, extent = prov_slot*bs + 1
    b1 = rb.bias[1].view(W, BS, H)
    assert torch.equal(b1[LAY.prov_slot, 0], eng._kv_bias_gpu[0].view(W, BS, H)[LAY.prov_slot, 0]) and (b1[LAY.prov_slot, 1:] == core.MASKED).all()
    assert int(rb.cache_seqlens[1]) == LAY.prov_slot * BS + 1
    assert rb.cache_batch_idx.tolist() == [0, 0, 1, 1]
    # in-place into a storage of >= max(R, B) rows: same bytes as the allocating path
    out = torch.empty((6, W * BS, H))
    rb2 = core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2, out=out)
    assert rb2.bias.data_ptr() == out.data_ptr() and torch.equal(out[:4], rb.bias) and torch.equal(rb2.cache_seqlens, rb.cache_seqlens)
    with pytest.raises(ValueError):
        core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2, out=torch.empty((1, W * BS, H)))
    # a flipped mask entry moves the bias (the mask is read)
    vis_f = vis.clone(); vis_f[0, 2, 0] = 0
    rb3 = core.paired_bias(eng._kv_bias_gpu, vis_f, cbi, 2)
    assert not torch.equal(rb3.bias, rb.bias)


def test_reference_twin_gathered_equals_dense_membership():
    """The pure-torch twin over the paired bias: gathered (visible columns
    only) vs dense (masked value in the logits) agree on membership (p == 0
    exactly on masked columns) and are allclose (summation order)."""
    eng = FakeEngine(B=2, tail_len=6, seed=3)
    plan = core.row_plan(torch.arange(2), True, True)
    ids, ready = _ids_ready(eng)
    sel = _sel(H, 2, [3, 4, 10, eng.T])
    vis, cbi = core.assemble_rows(plan, core.v_visible_rows(H, 2, LAY, 6), core.s_visible_rows(ids, ready, sel, LAY, 6), torch.arange(2))
    rb = core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2)
    q = torch.randn((4, HQ, D), generator=torch.Generator().manual_seed(5))
    scale = D ** -0.5
    og = ra.reference_rows_attention(q, eng._k_gpu, eng._v_gpu, rb, scale, torch.float32, gather=True)
    od, P = ra.reference_rows_attention(q, eng._k_gpu, eng._v_gpu, rb, scale, torch.float32, gather=False, return_p=True)
    assert torch.isfinite(og).all() and torch.isfinite(od).all()
    assert torch.allclose(og, od, atol=1e-5, rtol=1e-5)
    r_in = torch.arange(BS).view(1, BS)
    for r in range(4):
        for h in range(H):
            visible = (r_in < rb.visible_rows[r, :, h].view(-1, 1)).reshape(-1)
            for j in range(h * (HQ // H), (h + 1) * (HQ // H)):
                assert (P[r, j][~visible] == 0).all(), "a masked column contributes exactly 0"
                assert (P[r, j][visible] > 0).all()


# ---------------------------------------------------------------------------
# G1 / G5 / G7: the provisional write, the poison, the exact store untouched
# ---------------------------------------------------------------------------
def test_g5_provisional_write_and_poison_leave_exact_store_unchanged():
    eng = FakeEngine(B=3, tail_len=6)
    before = core.exact_store_digest(eng, LAY)
    k = torch.randn((3, H, D)); v = torch.randn((3, H, D)); b = torch.randn((3, H))
    row = core.s_provisional_write(eng, k, v, b, LAY)
    assert row == LAY.prov_slot * BS
    assert torch.equal(eng._k_gpu[:, row], k) and torch.equal(eng._v_gpu[:, row], v) and torch.equal(eng._kv_bias_gpu[:, row], b)
    assert core.digests_equal(before, core.exact_store_digest(eng, LAY)) == []
    core.poison_provisional(eng, LAY)
    assert torch.isnan(eng._k_gpu[:, row]).all() and torch.isnan(eng._kv_bias_gpu[:, row]).all()
    assert core.digests_equal(before, core.exact_store_digest(eng, LAY)) == []
    # a subset write touches only its requests
    idx = torch.tensor([2, 0])
    core.s_provisional_write(eng, k[idx], v[idx], b[idx], LAY, idx)
    assert torch.equal(eng._k_gpu[2, row], k[2]) and torch.isnan(eng._k_gpu[1, row]).all()
    assert core.digests_equal(before, core.exact_store_digest(eng, LAY)) == []
    # positive control: a V write into the tail DOES change the exact store (the digest sees it)
    kv_bias = torch.randn((3, eng.seq_length + 4, H))
    core.v_tail_write(eng, k.unsqueeze(1), v.unsqueeze(1), kv_bias)
    assert "k" in core.digests_equal(before, core.exact_store_digest(eng, LAY))


def test_g7_poison_control_surfaces_only_when_read():
    """The dense reference twin loads every column below a row's extent
    (as the kernel does): a NaN-poisoned provisional row is harmless for the
    V row (its extent stops at the tail slot) and for the S row that wrote
    it, and POISONS a row whose extent reaches it without writing it."""
    eng = FakeEngine(B=1, tail_len=6, seed=9)
    ids, ready = _ids_ready(eng)
    sel = _sel(H, 1, [3, 10, eng.T, -1])
    plan = core.row_plan(torch.arange(1), True, True)
    core.poison_provisional(eng, LAY)                                      # the previous tick's poison
    q = torch.randn((2, HQ, D), generator=torch.Generator().manual_seed(1))
    # V row alone: extent 3*64 + 6 < the provisional row -> finite
    vis_v = core.v_visible_rows(H, 1, LAY, 6)
    rb_v = core.paired_bias(eng._kv_bias_gpu, vis_v, torch.zeros(1, dtype=torch.int32), 1)
    assert int(rb_v.cache_seqlens[0]) < LAY.prov_slot * BS
    ov = ra.reference_rows_attention(q[:1], eng._k_gpu, eng._v_gpu, rb_v, D ** -0.5, torch.float32, gather=False)
    assert torch.isfinite(ov).all()
    # S row that wrote its row this tick -> finite
    core.s_provisional_write(eng, torch.randn(1, H, D), torch.randn(1, H, D), torch.randn(1, H), LAY)
    vis_s = core.s_visible_rows(ids, ready, sel, LAY, 6)
    vis, cbi = core.assemble_rows(plan, vis_v, vis_s, torch.arange(1))
    rb = core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2)
    o = ra.reference_rows_attention(q, eng._k_gpu, eng._v_gpu, rb, D ** -0.5, torch.float32, gather=False)
    assert torch.isfinite(o).all()
    # positive control: poison again and let a row's extent reach the row without naming it -> NaN surfaces
    core.poison_provisional(eng, LAY)
    bad = vis_v.clone(); bad[0, LAY.prov_slot + 1, :] = 1                     # a wrong extent past the poisoned row
    rb_bad = core.paired_bias(eng._kv_bias_gpu, bad, torch.zeros(1, dtype=torch.int32), 1)
    ob = ra.reference_rows_attention(q[:1], eng._k_gpu, eng._v_gpu, rb_bad, D ** -0.5, torch.float32, gather=False)
    assert torch.isnan(ob).any(), "the NaN-poison control must surface a read of the provisional row"


# ---------------------------------------------------------------------------
# G6: the tail rollover through the tail_write helper at tail_len 62 / 63 / 64
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tail_len0", [61, 62, 63])
def test_g6_tail_rollover_follows_the_s1_body(tail_len0):
    eng = FakeEngine(B=2, tail_len=tail_len0, T=40)
    B = 2
    seq0, map0 = eng.seq_length, eng._block_map.clone()
    k = torch.randn((B, 1, H, D)); v = torch.randn((B, 1, H, D))
    kv_bias = torch.randn((B, seq0 + 2, H))
    row, filled = core.v_tail_write(eng, k, v, kv_bias)
    tvw = core.tail_view(tail_len0, True, False, LAY)
    assert row == LAY.tail_slot * BS + tail_len0
    assert eng.seq_length == seq0 + 1
    assert torch.equal(eng._k_gpu[:, row], k[:, 0]) and torch.equal(eng._kv_bias_gpu[:, row], kv_bias[:, seq0])
    if tail_len0 + 1 < BS:                                                  # 62, 63 rows after the write: no fill
        assert not filled and eng._tail_block_len_on_gpu == tail_len0 + 1 and torch.equal(eng._block_map, map0)
        assert (eng._cache_lens == (TOPK - 1) * BS + tail_len0 + 1).all()
        assert tvw == core.TailView(tail_len0 + 1, False, False, tail_len0 + 1, 0)
    else:                                                                   # 64 rows: the fill = write-back, rename, len 0
        assert filled and eng._tail_block_len_on_gpu == 0
        assert torch.equal(eng._block_map[..., LAY.tail_slot], map0[..., LAY.tail_slot] + 1) and torch.equal(eng._block_map[..., :LAY.tail_slot], map0[..., :LAY.tail_slot])
        assert (eng._cache_lens == (TOPK - 1) * BS).all()
        base = LAY.tail_slot * BS
        assert torch.equal(eng._k_cpu[:, seq0 + 1 - BS:seq0 + 1], eng._k_gpu[:, base:base + BS])   # host rows = the completed block
        assert tvw == core.TailView(0, True, True, BS, 1)
        # what the two rows see at the fill step: V nothing of the tail (the decode's rule), S the complete block T by id
        assert (core.v_visible_rows(H, B, LAY, tvw.tail_len_after)[:, LAY.tail_slot] == 0).all()
        content = eng._block_map[..., LAY.tail_slot] - tvw.content_offset
        assert (content == 40).all()
        ids = core.slot_ids(eng._block_map, content, LAY)
        vis_s = core.s_visible_rows(ids, core.ready_mask(H, B, LAY), _sel(H, B, [40, 3, -1, -1]), LAY, tvw.tail_rows_for_s)
        assert (vis_s[:, LAY.tail_slot] == BS).all()
        # a restart before the next V write still sees the complete block; the next V write ends that
        tvr = core.tail_view(0, False, True, LAY)
        assert tvr.slot_complete and tvr.tail_rows_for_s == BS and tvr.content_offset == 1
        tvn = core.tail_view(0, True, True, LAY)
        assert tvn == core.TailView(1, False, False, 1, 0)
    with pytest.raises(ValueError):
        core.tail_view(BS, True, False, LAY)


def test_tail_view_after_prefill_without_a_partial_block():
    """tail_len 0 right after a block-aligned prefill: nothing sits in the
    tail slot (no fill happened), so an S row sees 0 tail rows."""
    tv = core.tail_view(0, False, False, LAY)
    assert tv == core.TailView(0, False, False, 0, 0)


# ---------------------------------------------------------------------------
# the attention entry on CPU with a fake kernel: the narrowed view, explicit splits
# ---------------------------------------------------------------------------
def test_attend_rows_arguments_and_split_refusal():
    eng = FakeEngine(B=2, tail_len=6)
    plan = core.row_plan(torch.arange(2), True, True)
    ids, ready = _ids_ready(eng)
    vis, cbi = core.assemble_rows(plan, core.v_visible_rows(H, 2, LAY, 6), core.s_visible_rows(ids, ready, _sel(H, 2, [3, 10, eng.T, -1]), LAY, 6), torch.arange(2))
    storage = torch.empty((4, W * BS, H))
    rb = core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2, out=storage)
    seen = {}

    def fake_fa(q4, k, v, bias_view, cache_seqlens, cache_batch_idx, num_splits):
        seen.update(rows=q4.shape[0], bias_rows=bias_view.shape[0], stride0=bias_view.stride(0), ptr=bias_view.data_ptr(), splits=num_splits,
                    csl=cache_seqlens.clone(), cbi=cache_batch_idx.clone())
        return torch.zeros((q4.shape[0], 1, q4.shape[2], q4.shape[3]))

    G = SimpleNamespace(fa=fake_fa)
    q = torch.randn((4, HQ, D))
    out = tick.attend_rows(G, q, eng._k_gpu, eng._v_gpu, rb, 4)
    assert out.shape == (4, HQ, D)
    assert seen["rows"] == 4 and seen["bias_rows"] == 2, "bias[:B] for flash_api.cpp:366, q carries the 2B rows"
    assert seen["stride0"] == W * BS * H and seen["ptr"] == storage.data_ptr() and seen["splits"] == 4
    assert torch.equal(seen["csl"], rb.cache_seqlens) and torch.equal(seen["cbi"], rb.cache_batch_idx)
    for bad in (0, 129):
        with pytest.raises(ValueError):
            tick.attend_rows(G, q, eng._k_gpu, eng._v_gpu, rb, bad)
    # a subset call: 1 row, storage still B rows
    rb1 = core.paired_bias(eng._kv_bias_gpu, vis[1:2], cbi[1:2], 1, out=storage)
    out1 = tick.attend_rows(G, q[:1], eng._k_gpu, eng._v_gpu, rb1, 8)
    assert out1.shape == (1, HQ, D) and seen["rows"] == 1 and seen["bias_rows"] == 2
    with pytest.raises(ValueError):
        tick.attend_rows(G, q, eng._k_gpu, eng._v_gpu, core.paired_bias(eng._kv_bias_gpu, vis, cbi, 2, out=torch.empty((4, W * BS, H)))._replace(bias=torch.empty((4, W * BS, H))[:, :10]), 4)


def test_config_from_env_reads_at_dispatch_and_refuses_heuristic(monkeypatch):
    monkeypatch.delenv("NOSI_ATTN_SPLITS", raising=False)
    monkeypatch.delenv("NOSI_PAIRED_POISON", raising=False)
    cfg = tick.config_from_env()
    assert cfg.num_splits == 4 and cfg.poison is False and cfg.masked == core.MASKED
    monkeypatch.setenv("NOSI_ATTN_SPLITS", "8"); monkeypatch.setenv("NOSI_PAIRED_POISON", "1")
    cfg = tick.config_from_env()
    assert cfg.num_splits == 8 and cfg.poison is True
    monkeypatch.setenv("NOSI_ATTN_SPLITS", "0")
    with pytest.raises(SystemExit):
        tick.config_from_env()


def test_twin_diagnose_names_first_term_and_g3_verdict():
    recs = [dict(layer=0, bias_equal=True, qkv_equal=True, attn_equal=True, ffn_equal=True),
            dict(layer=1, bias_equal=True, qkv_equal=False, q_maxabs=0.5, attn_equal=True, ffn_equal=True),
            dict(layer="head", norm_equal=True, lm_head_equal=False, lm_head_maxabs=1.0)]
    msg = twin.diagnose(recs)
    assert msg.startswith("layer 1: qkv GEMM")
    assert twin.diagnose(recs[:1]) is None
    a = [torch.zeros(2, 5), torch.ones(2, 5)]
    v = twin.g3_verdict(a, [torch.zeros(2, 5), torch.ones(2, 5)])
    assert v["all_equal"] and v["first_differing_step"] is None
    v2 = twin.g3_verdict(a, [torch.zeros(2, 5), torch.ones(2, 5) * 2])
    assert not v2["all_equal"] and v2["first_differing_step"] == 1 and v2["per_step_maxabs"][1] == 1.0


# ---------------------------------------------------------------------------
# source pins: importable on CPU, no CUDA import at module level, the V update is the shipped body
# ---------------------------------------------------------------------------
ALLOWED_TOP_IMPORTS = {"torch", "torch.nn.functional", "os", "typing", "__future__"}


def _top_imports(path: Path):
    tree = ast.parse(path.read_text())
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
    return out


@pytest.mark.parametrize("name", ["core.py", "tick.py", "twin.py", "__init__.py"])
def test_module_level_imports_are_pure(name):
    imps = _top_imports(NOSI_PKG / "paired" / name)
    for i in imps:
        assert i in ALLOWED_TOP_IMPORTS or i.startswith("."), "%s imports %s at module level" % (name, i)
    for banned in ("flash_attn", "infllm_v2", "flashinfer", "triton", "cpp_extension", "cache_engine", "nosa_llama"):
        assert not any(banned in i for i in imps), "%s must not import %s at module level" % (name, banned)


def test_tick_text_pins():
    src = (NOSI_PKG / "paired" / "tick.py").read_text()
    assert "cache.decode_update_kv(k[:, 0], v[:, 0], ucis, l, sel_v)" in src, "the V row's engine update is the shipped decode body, called"
    assert "flash_attn_with_kvcache" in src and "num_splits=ns" in src
    tree = ast.parse(src)
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    assert not any("environ" in ast.dump(n) for n in top), "no import-time knob read: the driver resolves NOSI_ATTN_SPLITS at dispatch (config_from_env)"
    assert "NL._ATTN_SPLITS" not in src and "nosa_llama._ATTN_SPLITS" not in src, "the paired tick never inherits the model module's import-time split default"
    assert "j.take(clayer)" in src and "j.restore(clayer)" in src, "the S position's tables are journaled"
    assert "core.poison_provisional" in src and "cfg.poison" in src
    assert "refuse_compress" in src
    twin_src = (NOSI_PKG / "paired" / "twin.py").read_text()
    assert "core.row_plan(sc.req_all, True, False)" in twin_src, "the twin is V rows only through the same body"
    for f in ("core.py", "tick.py", "twin.py"):
        assert "def " in (NOSI_PKG / "paired" / f).read_text()
    # the shipped decode path is untouched by this package: it never writes into nosa_llama / cache_engine
    assert "nosa_llama" not in (NOSI_PKG / "paired" / "core.py").read_text()


# ---------------------------------------------------------------------------
# the pilot's pure bookkeeping (benchmarks/Efficiency/paired_pilot.py), loaded by file path
# ---------------------------------------------------------------------------
def _load_pilot(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setenv("NOSI_PAIRED_OUT", str(tmp_path))
    monkeypatch.setenv("NOSI_PAIRED_MODE", "compare")
    spec = importlib.util.spec_from_file_location("nosi_paired_pilot_under_test", ROOT / "benchmarks" / "Efficiency" / "paired_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pilot_rows_and_gates(tmp_path, monkeypatch):
    pilot = _load_pilot(tmp_path, monkeypatch)
    assert pilot.compared_rows(4, 15) == list(range(5, 16)), "tick t -> row t+1 for t = WARM .. N-1"
    a = torch.zeros(2, 4, 6); b = a.clone(); b[:, 3, 0] = 0.5
    g = pilot.row_gate(a, b, [1, 2, 3])
    assert g["equal"] == [True, True, False] and g["first_bad"] == 3 and g["worst"] == 0.5 and g["argmax_agree"][1] == 2
    assert pilot.dlogit_tol(0.4844) == 4 * 0.4844
    with pytest.raises(ValueError):
        pilot.dlogit_tol(0.0)
    row = pilot.timing_row(64, "resident", "paired tick", [dict(total_ms=100.0), dict(total_ms=30.0, score_ms=5.0, fetch_ms=6.0, attn_ms=7.0, rest_ms=12.0),
                                                              dict(total_ms=32.0, score_ms=5.0, fetch_ms=6.0, attn_ms=7.0, rest_ms=14.0)], 40.0)
    assert row["n"] == 2 and row["ms_mean"] == 31.0 and row["ms_min"] == 30.0 and row["rest"] == 13.0, "the first call is excluded"
    empty = pilot.timing_row(64, "x", "y", [dict(total_ms=1.0)], 1.0)
    assert empty["n"] == 0 and "| 64 | x | y | 0 |" in pilot.render_timing([empty])


def test_pilot_refuses_without_tau_ctrl(tmp_path, monkeypatch):
    pilot = _load_pilot(tmp_path, monkeypatch)
    monkeypatch.delenv("NOSI_VERIFY_TAU_CTRL", raising=False)
    with pytest.raises(SystemExit):
        pilot.read_tau_ctrl()
    monkeypatch.setenv("NOSI_VERIFY_TAU_CTRL", "0.4844")
    assert pilot.read_tau_ctrl() == 0.4844
    monkeypatch.setenv("NOSI_VERIFY_TAU_CTRL", "inf")
    with pytest.raises(SystemExit):
        pilot.read_tau_ctrl()


# ---------------------------------------------------------------------------
# twin2b (GEMMs padded to M = 2B) and the registered decomposition
# ---------------------------------------------------------------------------
def test_linear_padded_discards_the_pad_rows():
    g = torch.Generator().manual_seed(2)
    x = torch.randn((3, 2, 8), generator=g)
    w = torch.randn((5, 8), generator=g)
    plain = torch.nn.functional.linear(x, w)
    assert torch.equal(tick.linear_padded(x, w, 0), plain), "pad 0 is the plain call"
    padded = tick.linear_padded(x, w, 6)
    assert padded.shape == plain.shape and torch.allclose(padded, plain, atol=1e-6)
    # the padded rows are zeros and never reach the output: a poisoned weight row still gives finite outputs where x is finite
    assert torch.isfinite(padded).all()


def test_config_gemm_pad_rows_and_twin2b_text(monkeypatch):
    monkeypatch.delenv("NOSI_ATTN_SPLITS", raising=False)
    assert tick.config_from_env().gemm_pad_rows == 0
    assert tick.config_from_env(gemm_pad_rows=64).gemm_pad_rows == 64
    with pytest.raises(SystemExit):
        tick.config_from_env(gemm_pad_rows=-1)
    src = (NOSI_PKG / "paired" / "twin.py").read_text()
    assert "cfg._replace(gemm_pad_rows=int(sc.B))" in src, "twin2b pads every GEMM by B rows: M = 2B"
    tsrc = (NOSI_PKG / "paired" / "tick.py").read_text()
    for w in ("layer.wqkv", "layer.wo", "layer.gate_up_proj", "layer.down_proj", "model.lm_head"):
        assert "linear_padded(" in tsrc and w in tsrc
    assert "F.linear(hs, layer.wqkv)" not in tsrc and "F.linear(attn, layer.wo)" not in tsrc, "every GEMM of the body goes through linear_padded"


def test_pilot_predictions_and_decomposition(tmp_path, monkeypatch):
    pilot = _load_pilot(tmp_path, monkeypatch)
    p = pilot.predicted_deltas(128)
    assert abs(p["rest_delta"] - (20.03 - 14.59)) < 1e-9 and p["attn_abs"] == 13.7 and abs(p["score_delta"] - (3.56 + 0.0281 * 128)) < 1e-9
    q = pilot.predicted_deltas(64)
    assert q["rest_delta"] is None and q["attn_abs"] is None and abs(q["score_delta"] - (3.56 + 0.0281 * 64)) < 1e-9, "no number is invented at a B without a registered point"
    twin_row = dict(n=3, ms_mean=40.0, score=7.0, fetch=5.0, attn=12.0, rest=16.0)
    tick_row = dict(n=3, ms_mean=60.0, score=14.0, fetch=6.0, attn=14.0, rest=26.0)
    txt = pilot.render_decomposition(128, twin_row, tick_row)
    assert "| rest | 16.00 | 26.00 | +10.00 | +5.44 | +4.56 |" in txt, "residual = measured - predicted, named per term"
    assert "| attn | 12.00 | 14.00 | +2.00 | +1.70 | +0.30 |" in txt
    txt64 = pilot.render_decomposition(64, twin_row, tick_row)
    assert "| rest | 16.00 | 26.00 | +10.00 | - | - |" in txt64
    assert "twin2b" in (ROOT / "benchmarks" / "Efficiency" / "paired_pilot.py").read_text()
    assert "G3-greedy" in (ROOT / "benchmarks" / "Efficiency" / "paired_pilot.py").read_text() and "G3-commit" in (ROOT / "benchmarks" / "Efficiency" / "paired_pilot.py").read_text()
