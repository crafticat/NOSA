"""Gates for the R2 per-request decode body (nosi/nosi/r2_decode_body.py).
Plan and table (1): docs/R2_STEP3_DECODE_BODY.md.

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2 python -m pytest -q tests/test_r2_decode_body.py

NOSI_REPO defaults to the tree that holds this file. Nothing imports
cache_engine.py (it compiles CUDA at import, cache_engine.py:19-23): the
SHIPPED decode body is the source of CacheEngine.decode_update_has_kv_bias,
extracted by AST and executed with diff.diff_offload and the two gathers
replaced by torch stand-ins (the pattern of the main repo's
tests/test_nosi_verify_engine.py). Both bodies get the same stand-ins, so a
difference is a difference of the bodies.

G-EQ   equal advance: the R2 body against the shipped body, torch.equal on
       every table and host row after prefill and after every step.
G-RAG  ragged activity: B = 4 against 4 independent B = 1 engines, per
       request, every step; the engine agrees with the positional twin.
G-POISON  NaN host rows around every request's write window: only the full
       ACTIVE request's block is written, exactly its rows.
NEGATIVE CONTROLS: an unmasked write-back and one that ignores activity fail
G-POISON; an unmasked write-back fails G-EQ; a shared tail position fails
G-RAG and passes G-EQ.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import textwrap
import types
from pathlib import Path

import pytest
import torch

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
N = NOSI_REPO / "nosi/nosi"
BODY = N / "r2_decode_body.py"
TWIN = N / "r2_positional_twin.py"
CE = N / "cache_engine.py"

if not BODY.is_file() or not TWIN.is_file():
    pytest.skip("no r2_decode_body.py / r2_positional_twin.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load(BODY, "nosi_r2_decode_body_under_test")
tw = _load(TWIN, "nosi_r2_positional_twin_for_body_test")

BS, TOPK = 64, 64                    # nosa_llama.py:680-681
TAIL0 = (TOPK - 1) * BS              # 4032
H_, D_ = 2, 4
DT = torch.float32
MAPS = ("_block_map", "_new_block_map_buf", "_load_mask")
TABLES = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_new_block_map_buf", "_load_mask",
          "_cache_lens", "_k_cpu", "_v_cpu")
PERSIST = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_cache_lens", "_k_cpu", "_v_cpu")


# ---------------------------------------------------------------------------
# torch stand-ins for the kernels (the same for both bodies)
# ---------------------------------------------------------------------------
def diff_offload_ref(old_map, new_act, new_map, load_map):
    """diff_kernel (flash_cache_engine/diff_offload_kernel.cu:10-99) per (h, b)
    row: an old id that is still selected keeps its slot (load -1); the k-th
    selected id that is not resident goes to the k-th slot whose old id is not
    selected (load = that id). Entries the kernel does not write keep their value."""
    H, B, M = old_map.shape
    s = old_map.sort(-1).values
    assert not bool(((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).any()), \
        "duplicate block id in a map row (diff_kernel needs distinct ids)"
    t = new_act.sort(-1).values
    assert not bool((t[..., 1:] == t[..., :-1]).any()), "duplicate id in the selection"
    old_hit = (old_map.unsqueeze(-1) == new_act.unsqueeze(-2)).any(-1)
    new_hit = (new_act.unsqueeze(-1) == old_map.unsqueeze(-2)).any(-1)
    new_map[old_hit] = old_map[old_hit]
    load_map[old_hit] = -1
    miss, free = ~new_hit, ~old_hit
    dummy = torch.full_like(new_act, M)
    by_rank = torch.full((H, B, M + 1), -2, dtype=torch.int64)
    by_rank.scatter_(-1, torch.where(miss, torch.cumsum(miss.long(), -1) - 1, dummy),
                     torch.where(miss, new_act, torch.full_like(new_act, -2)))
    val = by_rank.gather(-1, torch.where(free, torch.cumsum(free.long(), -1) - 1, dummy))
    write = free & (val != -2)
    new_map[write] = val[write]
    load_map[write] = val[write]


def _gather(gpu, cpu, load_ids, bs, bias_buf=None, bias=None):
    """flash_h2d_from_mask(_bias) addressing: slot m of the GPU tensor <- host
    block id, ids < 0 skipped."""
    B, S_GPU, H, D = gpu.shape
    assert tuple(load_ids.shape) == (H, B, S_GPU // bs) and load_ids.dtype == torch.int64
    h, b, m = (load_ids >= 0).nonzero(as_tuple=True)
    if h.numel() == 0:
        return
    ids = load_ids[h, b, m]
    r = torch.arange(bs)
    dst = m[:, None] * bs + r
    src = ids[:, None] * bs + r
    assert int(src.max()) < cpu.shape[1], "a gather reads past the host rows"
    gpu[b[:, None], dst, h[:, None]] = cpu[b[:, None], src, h[:, None]]
    if bias_buf is not None:
        assert int(src.max()) < bias.shape[1], "a bias gather reads past the cis rows"
        bias_buf[b[:, None], dst, h[:, None]] = bias[b[:, None], src, h[:, None]]


def gather_k_ref(gpu, cpu, load_ids, block_size):
    _gather(gpu, cpu, load_ids, block_size)


def gather_v_bias_ref(gpu, cpu, bias_buf, bias, load_ids, block_size):
    _gather(gpu, cpu, load_ids, block_size, bias_buf, bias)


STUBS = (diff_offload_ref, gather_k_ref, gather_v_bias_ref)


def _shipped_decode():
    """CacheEngine.decode_update_has_kv_bias (cache_engine.py:335-412), its own
    source, with the kernels replaced and the trace / availability hooks off."""
    src = CE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CacheEngine")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode_update_has_kv_bias")
    ns = dict(torch=torch,
              diff=types.SimpleNamespace(diff_offload=diff_offload_ref),
              _tt=types.SimpleNamespace(TRACE=None),
              _avail=types.SimpleNamespace(POLICY=None),
              flash_h2d_from_mask=gather_k_ref,
              flash_h2d_from_mask_bias=gather_v_bias_ref)
    code = textwrap.dedent(ast.get_source_segment(src, fn, padded=True))
    exec(compile(code, str(CE), "exec"), ns)
    return ns["decode_update_has_kv_bias"], code


SHIPPED_DECODE, SHIPPED_SRC = _shipped_decode()


# ---------------------------------------------------------------------------
# engines, streams, selections
# ---------------------------------------------------------------------------
def _engine(B, n_host, seed):
    """A CacheEngine look-alike: the attributes both bodies touch. Every buffer
    starts as finite random bytes (torch.empty in the shipped allocation)."""
    g = torch.Generator().manual_seed(seed)
    e = types.SimpleNamespace(head_num=H_, head_dim=D_, block_size=BS, topk=TOPK, dtype=DT,
                              device=torch.device("cpu"), pool_blocks=0)
    rows = TOPK * BS
    e._k_gpu = torch.randn(B, rows, H_, D_, generator=g)
    e._v_gpu = torch.randn(B, rows, H_, D_, generator=g)
    e._kv_bias_gpu = torch.randn(B, rows, H_, generator=g)
    e._k_cpu = torch.randn(B, n_host, H_, D_, generator=g)
    e._v_cpu = torch.randn(B, n_host, H_, D_, generator=g)
    e._block_map = torch.full((H_, B, TOPK), -1, dtype=torch.int64)
    e._new_block_map_buf = torch.randint(-5, 5, (H_, B, TOPK), generator=g)
    e._load_mask = torch.randint(-5, 5, (H_, B, TOPK), generator=g)
    e._cache_lens = torch.zeros(B, dtype=torch.int32)
    e._tail_block_idx_on_gpu = TOPK - 1
    return e


def _clone(e):
    c = types.SimpleNamespace(**vars(e))
    for k in TABLES:
        setattr(c, k, getattr(e, k).clone())
    return c


def _single(e, b):
    """Request b of a batched engine as an independent B = 1 engine, same bytes."""
    s = types.SimpleNamespace(**vars(e))
    for k in TABLES:
        t = getattr(e, k)
        setattr(s, k, (t[:, b:b + 1] if k in MAPS else t[b:b + 1]).clone())
    return s


def _req(e, k, b):
    t = getattr(e, k)
    return t[:, b] if k in MAPS else t[b]


def _streams(B, total, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, total, H_, D_, generator=g), torch.randn(B, total, H_, D_, generator=g),
            torch.randn(B, total, H_, generator=g))


def _selection(seed, b, p):
    """(H, 64) distinct block ids for request b writing position p: the tail id
    p // 64 (the forced local window) and 63 complete blocks from [0, p // 64)."""
    T = p // BS
    assert T >= TOPK - 1
    g = torch.Generator().manual_seed(seed * 1_000_003 + b * 100_003 + p)
    rows = []
    for _ in range(H_):
        row = torch.cat([torch.randperm(T, generator=g)[:TOPK - 1], torch.tensor([T])])
        rows.append(row[torch.randperm(TOPK, generator=g)])
    return torch.stack(rows)


def _topk(seed, p):
    return torch.stack([_selection(seed, b, int(p[b])) for b in range(p.shape[0])], dim=1).contiguous()


def _selection_local(seed, b, p):
    """Like _selection, but the previous block p // 64 - 1 is always selected
    too (a forced local window), so a just-completed block enters the window."""
    T = p // BS
    g = torch.Generator().manual_seed(seed * 1_000_003 + b * 100_003 + p)
    rows = []
    for _ in range(H_):
        row = torch.cat([torch.randperm(T - 1, generator=g)[:TOPK - 2], torch.tensor([T - 1, T])])
        rows.append(row[torch.randperm(TOPK, generator=g)])
    return torch.stack(rows)


def _assert_distinct_ids(eng, where):
    s = eng._block_map.sort(-1).values
    dup = (s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)
    assert not bool(dup.any()), "%s: duplicate block id in a map row" % where


def _shipped_prefill(e, keys, values, bias, S):
    """cache_engine.py:187-189 and :309-331 for current_batch_pos = 0, total_bsz = B, pool off."""
    B = e._k_gpu.shape[0]
    e.seq_length = S                                                  # :189
    e._tail_block_len_on_gpu = S % BS                                 # :309
    tail = e._tail_block_len_on_gpu
    if tail > 0:                                                      # :311-319
        e._k_gpu[:, TAIL0:TAIL0 + tail].copy_(keys[:, S - tail:S])
        e._v_gpu[:, TAIL0:TAIL0 + tail].copy_(values[:, S - tail:S])
        e._kv_bias_gpu[:, TAIL0:TAIL0 + tail].copy_(bias[:, S - tail:S])
    e._block_map[:, :, TOPK - 1] = S // BS                            # :321 / :324
    e._cache_lens = torch.full((B,), TAIL0 + tail, dtype=torch.int32)  # :327
    e._k_cpu[:, :S].copy_(keys[:, :S])                                # :330
    e._v_cpu[:, :S].copy_(values[:, :S])                              # :331


def _assert_engines_equal(x, y, names, where):
    for k in names:
        a, c = getattr(x, k), getattr(y, k)
        assert a.shape == c.shape and torch.equal(a, c), "%s: %s differs" % (where, k)


def _assert_request_equal(x, bx, y, by, where):
    for k in PERSIST:
        assert torch.equal(_req(x, k, bx), _req(y, k, by)), "%s: %s differs" % (where, k)


def _assert_twin_agrees(eng, twin, where):
    assert torch.equal(eng._cache_lens, twin.cache_lens()), "%s: cache_lens vs twin" % where
    assert torch.equal(eng._block_map[..., TOPK - 1], twin.tail_block().expand(H_, -1)), "%s: tail id vs twin" % where


def _assert_coherent(eng, b, keys, values, total_cis, pos, where):
    """Slot s < 63 holds the host rows (K, V) and the cis rows (bias) of the block
    its map names; slot 63 rows [0, pos % 64) hold the request's last tokens."""
    for h in range(H_):
        for s in range(TOPK - 1):
            i = int(eng._block_map[h, b, s])
            if i < 0:
                continue
            sl, hl = slice(s * BS, (s + 1) * BS), slice(i * BS, (i + 1) * BS)
            assert torch.equal(eng._k_gpu[b, sl, h], eng._k_cpu[b, hl, h]), "%s: slot %d K" % (where, s)
            assert torch.equal(eng._v_gpu[b, sl, h], eng._v_cpu[b, hl, h]), "%s: slot %d V" % (where, s)
            assert torch.equal(eng._kv_bias_gpu[b, sl, h], total_cis[b, hl, h]), "%s: slot %d bias" % (where, s)
    T, n = pos // BS, pos % BS
    assert torch.equal(eng._k_gpu[b, TAIL0:TAIL0 + n], keys[b, T * BS:T * BS + n]), "%s: tail K" % where
    assert torch.equal(eng._v_gpu[b, TAIL0:TAIL0 + n], values[b, T * BS:T * BS + n]), "%s: tail V" % where
    assert torch.equal(eng._kv_bias_gpu[b, TAIL0:TAIL0 + n], total_cis[b, T * BS:T * BS + n]), "%s: tail bias" % where


REF = rb.DecodeBody(False)
ONE = torch.ones(1, dtype=torch.bool)


# ---------------------------------------------------------------------------
# negative-control mutants
# ---------------------------------------------------------------------------
class _Unmasked(rb.DecodeBody):
    def full_mask(self, eng, p, active):
        return active.clone()                        # every active request writes back


class _IgnoresActivity(rb.DecodeBody):
    def full_mask(self, eng, p, active):
        return (p + 1) % eng.block_size == 0         # full positions write back, active or not


class _SharedTail(rb.DecodeBody):
    def positions(self, pos):
        return pos.max().expand_as(pos).clone()      # one tail position for the batch


class _UnconditionalTail(rb.DecodeBody):
    def write_tail(self, eng, key_states, value_states, kv_bias, p, active):
        super().write_tail(eng, key_states, value_states, kv_bias, p, torch.ones_like(active))


class _NoInactiveMask(rb.DecodeBody):
    def select(self, eng, topk_idx, active, diff_offload):
        diff_offload(eng._block_map, topk_idx, eng._new_block_map_buf, eng._load_mask)
        eng._block_map.copy_(eng._new_block_map_buf)


class _ResyncNoClear(rb.DecodeBody):
    def resync(self, eng, pos, mask):                # the first version: the tail id and length only
        B = pos.shape[0]
        col = eng._tail_block_idx_on_gpu
        eng._block_map[..., col] = torch.where(mask.view(1, B), (pos // eng.block_size).view(1, B),
                                               eng._block_map[..., col])
        eng._cache_lens.copy_(torch.where(mask, self.cache_lens_value(eng, pos - 1),
                                          eng._cache_lens.to(torch.int64)).to(torch.int32))


# ---------------------------------------------------------------------------
# gate runners
# ---------------------------------------------------------------------------
def run_eq(body, S, B, steps=150, seed=0):
    total = S + steps + BS
    keys, values, cis = _streams(B, total, seed)
    twin = tw.PositionalTwin(B, H_, D_, total // 16, total)
    base = _engine(B, total, seed + 1)
    ship, r2 = _clone(base), _clone(base)
    for b in range(B):
        twin.prefill(b, keys[b, :S], cis[b, :S])
    _shipped_prefill(ship, keys, values, twin.total_cis, S)
    for b in range(B):
        body.prefill_request(r2, b, keys[b, :S], values[b, :S], twin.total_cis[b, :S])
    _assert_engines_equal(ship, r2, TABLES, "prefill")
    act = torch.ones(B, dtype=torch.bool)
    for i in range(steps):
        p = twin.pos.clone()
        assert torch.equal(p, torch.full((B,), S + i, dtype=torch.int64))
        twin.step(keys[:, S + i], cis[:, S + i], active=act)          # the cis row first (nosa_llama.py:584)
        topk = _topk(seed, p)
        k1, v1 = keys[:, S + i:S + i + 1], values[:, S + i:S + i + 1]
        SHIPPED_DECODE(ship, k1, v1, twin.total_cis, topk)
        body.decode(r2, k1, v1, twin.total_cis, topk, p, act, p.clone(), act.clone(), *STUBS)
        _assert_engines_equal(ship, r2, TABLES, "step %d" % (i + 1))
        _assert_twin_agrees(r2, twin, "step %d" % (i + 1))


def run_rag(body, S_list, steps=200, seed=3, p_active=0.6):
    B = len(S_list)
    total = max(S_list) + steps + BS
    keys, values, cis = _streams(B, total, seed)
    twin = tw.PositionalTwin(B, H_, D_, total // 16, total)
    twins1 = [tw.PositionalTwin(1, H_, D_, total // 16, total) for _ in range(B)]
    base = _engine(B, total, seed + 1)
    eng = _clone(base)
    singles = [_single(base, b) for b in range(B)]
    for b, S in enumerate(S_list):
        twin.prefill(b, keys[b, :S], cis[b, :S])
        twins1[b].prefill(0, keys[b, :S], cis[b, :S])
        body.prefill_request(eng, b, keys[b, :S], values[b, :S], twin.total_cis[b, :S])
        REF.prefill_request(singles[b], 0, keys[b, :S], values[b, :S], twins1[b].total_cis[0, :S])
    for b in range(B):
        _assert_request_equal(eng, b, singles[b], 0, "prefill, request %d" % b)
    g = torch.Generator().manual_seed(seed + 2)
    bi = torch.arange(B)
    stats = {"ragged_steps": 0, "writebacks": 0}
    for i in range(steps):
        act = (bi % 2 == 0) if i == 0 else (torch.rand(B, generator=g) < p_active)
        stats["ragged_steps"] += int(bool(act.any()) and not bool(act.all()))
        p = twin.pos.clone()
        stats["writebacks"] += int((act & ((p + 1) % BS == 0)).sum())
        tok_k, tok_v, tok_c = keys[bi, p], values[bi, p], cis[bi, p]
        twin.step(tok_k, tok_c, active=act)
        topk = _topk(seed, p)
        body.decode(eng, tok_k.unsqueeze(1), tok_v.unsqueeze(1), twin.total_cis, topk, p, act,
                    p.clone(), act.clone(), *STUBS)
        for b in range(B):
            if bool(act[b]):
                q = twins1[b].pos.clone()
                assert int(q[0]) == int(p[b])
                twins1[b].step(tok_k[b:b + 1], tok_c[b:b + 1])
                REF.decode(singles[b], tok_k[b:b + 1].unsqueeze(1), tok_v[b:b + 1].unsqueeze(1), twins1[b].total_cis,
                           topk[:, b:b + 1].contiguous(), q, ONE, q.clone(), ONE.clone(), *STUBS)
            _assert_request_equal(eng, b, singles[b], 0, "step %d, request %d" % (i + 1, b))
        _assert_twin_agrees(eng, twin, "step %d" % (i + 1))
    for b in range(B):
        _assert_coherent(eng, b, keys, values, twin.total_cis, int(twin.pos[b]), "end, request %d" % b)
    return stats


POISON_S = [6079, 6016, 6143, 6050]          # (S + 1) % 64 == 0 for requests 0 and 2
POISON_ACTIVE = [True, True, False, True]    # request 2 is full but inactive


def run_poison(body):
    B = len(POISON_S)
    total = max(POISON_S) + 2 * BS
    keys, values, cis = _streams(B, total, 11)
    twin = tw.PositionalTwin(B, H_, D_, total // 16, total)
    eng = _engine(B, total, 12)
    for b, S in enumerate(POISON_S):
        twin.prefill(b, keys[b, :S], cis[b, :S])
        body.prefill_request(eng, b, keys[b, :S], values[b, :S], twin.total_cis[b, :S])
    p = twin.pos.clone()
    for b in range(B):
        lo, hi = int(p[b]) - BS, int(p[b]) + 2
        eng._k_cpu[b, lo:hi] = float("nan")
        eng._v_cpu[b, lo:hi] = float("nan")
    act = torch.tensor(POISON_ACTIVE)
    bi = torch.arange(B)
    tok_k, tok_v, tok_c = keys[bi, p], values[bi, p], cis[bi, p]
    twin.step(tok_k, tok_c, active=act)
    body.decode(eng, tok_k.unsqueeze(1), tok_v.unsqueeze(1), twin.total_cis, _topk(5, p), p, act,
                p.clone(), act.clone(), *STUBS)
    for b in range(B):
        pb = int(p[b])
        full_active = bool(act[b]) and (pb + 1) % BS == 0
        for name, host, stream in (("K", eng._k_cpu, keys), ("V", eng._v_cpu, values)):
            if full_active:
                assert torch.equal(host[b, pb - BS + 1:pb + 1], stream[b, pb - BS + 1:pb + 1]), \
                    "full request %d: %s block rows are not its tokens" % (b, name)
                assert bool(torch.isnan(host[b, pb - BS]).all()) and bool(torch.isnan(host[b, pb + 1]).all()), \
                    "full request %d: a %s neighbour row was written" % (b, name)
            else:
                assert bool(torch.isnan(host[b, pb - BS:pb + 2]).all()), \
                    "request %d (not full, or inactive): %s host rows written" % (b, name)
    return [b for b in range(B) if bool(act[b]) and (int(p[b]) + 1) % BS == 0]


# ---------------------------------------------------------------------------
# G-EQ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [6016, 6017, 6079])     # S % 64 = 0, 1, 63
def test_geq_r2_body_equals_shipped_body(S, B):
    run_eq(rb.DecodeBody(False), S, B)


def test_shipped_body_is_the_shipped_one():
    for text in ("tail_full = self._tail_block_len_on_gpu == self.block_size",
                 "diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)",
                 "self._block_map[..., self._tail_block_idx_on_gpu] += 1",
                 "cpu_block_base_pos = self.seq_length - self.block_size"):
        assert text in SHIPPED_SRC


# ---------------------------------------------------------------------------
# G-RAG
# ---------------------------------------------------------------------------
def test_grag_ragged_equals_independent_runs():
    stats = run_rag(rb.DecodeBody(False), [6016, 6017, 6079, 6100])
    assert stats["ragged_steps"] > 0
    assert stats["writebacks"] >= 4, "the run crossed too few block completions"


# ---------------------------------------------------------------------------
# G-POISON
# ---------------------------------------------------------------------------
def test_gpoison_writeback_writes_only_full_active_requests():
    assert run_poison(rb.DecodeBody(False)) == [0]


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
def test_negctl_unmasked_writeback_fails_gpoison():
    with pytest.raises(AssertionError, match="host rows written"):
        run_poison(_Unmasked(False))


def test_negctl_writeback_ignoring_activity_fails_gpoison():
    with pytest.raises(AssertionError, match="request 2 .*host rows written"):
        run_poison(_IgnoresActivity(False))


def test_negctl_unmasked_writeback_fails_geq():
    with pytest.raises(AssertionError, match="_k_cpu differs"):
        run_eq(_Unmasked(False), 6016, 1, steps=3)


def test_negctl_shared_tail_fails_grag():
    with pytest.raises(AssertionError, match="request 0: _k_gpu differs"):
        run_rag(_SharedTail(False), [6016, 6017, 6079, 6100], steps=5)


def test_negctl_shared_tail_passes_geq():
    """Equal advance cannot see a shared tail position: G-RAG is the gate that does."""
    run_eq(_SharedTail(False), 6017, 4, steps=70)


def test_negctl_unmasked_selection_fails_grag():
    with pytest.raises(AssertionError, match="request 1: _k_gpu differs"):
        run_rag(_NoInactiveMask(False), [6016, 6017, 6079, 6100], steps=5)


def test_negctl_f4_fix_fails_geq():
    """G-EQ sees a different attention length: the corrected F4 is not the shipped one."""
    with pytest.raises(AssertionError, match="_cache_lens differs"):
        run_eq(rb.DecodeBody(True), 6079, 1, steps=2)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def _setup(S_list, extra=80, seed=21):
    B = len(S_list)
    total = max(S_list) + extra + BS
    keys, values, cis = _streams(B, total, seed)
    twin = tw.PositionalTwin(B, H_, D_, total // 16, total)
    eng = _engine(B, total, seed + 1)
    for b, S in enumerate(S_list):
        twin.prefill(b, keys[b, :S], cis[b, :S])
        REF.prefill_request(eng, b, keys[b, :S], values[b, :S], twin.total_cis[b, :S])
    return eng, twin, keys, values, cis


def _step(body, eng, twin, keys, values, cis, act, seed=21):
    bi = torch.arange(eng._k_gpu.shape[0])
    p = twin.pos.clone()
    tok_k, tok_v, tok_c = keys[bi, p], values[bi, p], cis[bi, p]
    twin.step(tok_k, tok_c, active=act)
    body.decode(eng, tok_k.unsqueeze(1), tok_v.unsqueeze(1), twin.total_cis, _topk(seed, p), p, act,
                p.clone(), act.clone(), *STUBS)
    return p


def run_inactive(body):
    eng, twin, keys, values, cis = _setup([6016, 6040, 6079])
    for _ in range(3):
        _step(body, eng, twin, keys, values, cis, torch.ones(3, dtype=torch.bool))
    before = {k: _req(eng, k, 1).clone() for k in PERSIST}
    _step(body, eng, twin, keys, values, cis, torch.tensor([True, False, True]))
    for k in PERSIST:
        assert torch.equal(_req(eng, k, 1), before[k]), "inactive request: %s changed" % k
    assert bool((eng._load_mask[:, 1] == -1).all())
    assert torch.equal(eng._new_block_map_buf[:, 1], eng._block_map[:, 1])


def test_inactive_request_changes_nothing():
    run_inactive(REF)


def test_negctl_unconditional_tail_write_fails_inactive():
    with pytest.raises(AssertionError, match="inactive request: _k_gpu changed"):
        run_inactive(_UnconditionalTail(False))


def test_f4_switch():
    S = 6078                                          # p = S: row 62; p = S + 1: row 63 (full); p = S + 2: next block
    eng_s, twin_s, keys, values, cis = _setup([S])
    eng_f = _clone(eng_s)
    twin_f = tw.PositionalTwin(1, H_, D_, twin_s.lk_max, twin_s.n_max)
    twin_f.prefill(0, keys[0, :S], cis[0, :S])
    ship, fix = rb.DecodeBody(False), rb.DecodeBody(True)
    expect_s, expect_f = [TAIL0 + 63, TAIL0, TAIL0 + 1], [TAIL0 + 63, TAIL0 + 64, TAIL0 + 1]
    one = torch.ones(1, dtype=torch.bool)
    for i in range(3):
        _step(ship, eng_s, twin_s, keys, values, cis, one)
        _step(fix, eng_f, twin_f, keys, values, cis, one)
        assert int(eng_s._cache_lens[0]) == expect_s[i]
        assert int(eng_f._cache_lens[0]) == expect_f[i]
        _assert_twin_agrees(eng_s, twin_s, "shipped F4, step %d" % (i + 1))
        _assert_engines_equal(eng_s, eng_f, [k for k in TABLES if k != "_cache_lens"], "F4 switch, step %d" % (i + 1))
    with pytest.raises(TypeError):
        rb.DecodeBody(1)


def test_tail_in_selection():
    eng, twin, keys, values, cis = _setup([6016, 6050])
    p = twin.pos.clone()
    topk = _topk(21, p)
    assert REF.tail_in_selection(eng, topk, p).tolist() == [True, True]
    bad = topk.clone()
    bad[1, 0][bad[1, 0] == int(p[0]) // BS] = 0      # head 1 of request 0 loses the tail id
    assert REF.tail_in_selection(eng, bad, p).tolist() == [False, True]


def test_resync_after_a_rollback_inside_a_block():
    S_list = [6026, 6030]                             # S % 64 = 10, 14: five more tokens stay in the block
    eng, twin, keys, values, cis = _setup(S_list)
    B = len(S_list)
    # the committed-only runs start from the same bytes, right after prefill
    singles = [_single(eng, b) for b in range(B)]
    twins1 = []
    for b, S in enumerate(S_list):
        t1 = tw.PositionalTwin(1, H_, D_, twin.lk_max, twin.n_max)
        t1.prefill(0, keys[b, :S], cis[b, :S])
        twins1.append(t1)
    ctr = [twin.counters()]
    for _ in range(5):
        _step(REF, eng, twin, keys, values, cis, torch.ones(B, dtype=torch.bool))
        ctr.append(twin.counters())
    keep = torch.tensor([2, 5])
    twin.commit(torch.stack(ctr), keep)
    REF.resync(eng, twin.pos, torch.ones(B, dtype=torch.bool))
    _assert_twin_agrees(eng, twin, "after resync")
    for b in range(B):
        # the committed-only run of request b: the same tokens and selections, keep[b] steps
        e1, t1 = singles[b], twins1[b]
        for _ in range(int(keep[b])):
            p1 = t1.pos.clone()
            t1.step(keys[b, p1], cis[b, p1], active=ONE)
            REF.decode(e1, keys[b, p1].unsqueeze(1), values[b, p1].unsqueeze(1), t1.total_cis,
                       _selection(21, b, int(p1[0])).unsqueeze(1).contiguous(), p1, ONE, p1.clone(), ONE.clone(), *STUBS)
        pos = int(twin.pos[b])
        assert pos == int(t1.pos[0])
        n = pos % BS
        assert torch.equal(eng._block_map[:, b, TOPK - 1], e1._block_map[:, 0, TOPK - 1])
        assert int(eng._cache_lens[b]) == int(e1._cache_lens[0])
        for k in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            assert torch.equal(getattr(eng, k)[b, TAIL0:TAIL0 + n], getattr(e1, k)[0, TAIL0:TAIL0 + n]), k
        hb = (pos // BS) * BS
        assert torch.equal(eng._k_cpu[b, :hb], e1._k_cpu[0, :hb]) and torch.equal(eng._v_cpu[b, :hb], e1._v_cpu[0, :hb])
        _assert_coherent(eng, b, keys, values, twin.total_cis, pos, "after resync, request %d" % b)


def run_cross(body, keep=2, drafts=10, readvance=70):
    """Drafts run from S = 6076 over the end of block 94 (p = 6079) into block
    95, then a rollback keeps 2 tokens: the tail is block 94 again. The
    rejected drafts carry junk K / V / cis. After resync and B's masked tail
    copy (simulated here), the engine and a committed-only engine decode the
    true tokens; block 94 completes again and is written back."""
    S = 6076
    eng, twin, keys, values, cis = _setup([S], extra=150)
    single = _single(eng, 0)
    t1 = tw.PositionalTwin(1, H_, D_, twin.lk_max, twin.n_max)
    t1.prefill(0, keys[0, :S], cis[0, :S])
    g = torch.Generator().manual_seed(41)

    def junk(*shape):
        return torch.randint(100, 121, shape, generator=g).to(DT)

    def go(e, t, k, v, c):
        p = t.pos.clone()
        t.step(k, c, active=ONE)
        body.decode(e, k.unsqueeze(1), v.unsqueeze(1), t.total_cis,
                    _selection_local(31, 0, int(p[0])).unsqueeze(1).contiguous(), p, ONE, p.clone(), ONE.clone(), *STUBS)

    ctr = [twin.counters()]
    for u in range(drafts):                                  # positions 6076 .. 6085
        q = S + u
        if u < keep:
            go(eng, twin, keys[0:1, q], values[0:1, q], cis[0:1, q])
        else:
            go(eng, twin, junk(1, H_, D_), junk(1, H_, D_), junk(1, H_))
        ctr.append(twin.counters())
    assert int(eng._block_map[0, 0, TOPK - 1]) == 95
    assert bool((eng._block_map[:, 0, :TOPK - 1] == 94).any()), "the drafts loaded block 94 into the window"
    twin.commit(torch.stack(ctr), [keep])
    pos = int(twin.pos[0])
    assert pos == S + keep
    body.resync(eng, twin.pos, ONE)
    _assert_distinct_ids(eng, "after resync")
    assert int(eng._block_map[0, 0, TOPK - 1]) == pos // BS
    assert not bool((eng._block_map[:, 0, :TOPK - 1] >= pos // BS).any())
    _assert_twin_agrees(eng, twin, "after resync")
    T, n = pos // BS, pos % BS                               # B's masked tail copy, simulated
    eng._k_gpu[0, TAIL0:TAIL0 + n] = eng._k_cpu[0, T * BS:T * BS + n]
    eng._v_gpu[0, TAIL0:TAIL0 + n] = eng._v_cpu[0, T * BS:T * BS + n]
    eng._kv_bias_gpu[0, TAIL0:TAIL0 + n] = twin.total_cis[0, T * BS:T * BS + n]
    _assert_coherent(eng, 0, keys, values, twin.total_cis, pos, "after resync and tail copy")
    for u in range(keep):                                    # the committed-only run catches up
        go(single, t1, keys[0:1, S + u], values[0:1, S + u], cis[0:1, S + u])
    for i in range(readvance):
        q = pos + i
        for e, t in ((eng, twin), (single, t1)):
            go(e, t, keys[0:1, q], values[0:1, q], cis[0:1, q])
        w = "re-advance step %d" % (i + 1)
        P = int(twin.pos[0])
        assert P == int(t1.pos[0])
        Tn, nn = P // BS, P % BS
        assert torch.equal(eng._block_map[:, 0, TOPK - 1], single._block_map[:, 0, TOPK - 1]), "%s: tail id" % w
        assert torch.equal(eng._cache_lens, single._cache_lens), "%s: cache_lens" % w
        for k in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            assert torch.equal(getattr(eng, k)[0, TAIL0:TAIL0 + nn], getattr(single, k)[0, TAIL0:TAIL0 + nn]), \
                "%s: tail %s" % (w, k)
        assert torch.equal(eng._k_cpu[0, :Tn * BS], single._k_cpu[0, :Tn * BS]), "%s: host K" % w
        assert torch.equal(eng._v_cpu[0, :Tn * BS], single._v_cpu[0, :Tn * BS]), "%s: host V" % w
        _assert_distinct_ids(eng, w)
        _assert_coherent(eng, 0, keys, values, twin.total_cis, P, w)


def test_resync_after_a_rollback_across_a_block_boundary():
    run_cross(REF)


def test_negctl_resync_without_the_clear_duplicates_a_block():
    with pytest.raises(AssertionError, match="duplicate block id"):
        run_cross(_ResyncNoClear(False))


def test_allocate_shapes_and_refusals():
    eng = types.SimpleNamespace(head_num=H_, head_dim=D_, block_size=BS, topk=TOPK, dtype=DT,
                                device=torch.device("cpu"), pool_blocks=0)
    REF.allocate(eng, 3, 5000, False)
    assert tuple(eng._k_gpu.shape) == (3, TOPK * BS, H_, D_) and tuple(eng._kv_bias_gpu.shape) == (3, TOPK * BS, H_)
    assert tuple(eng._k_cpu.shape) == (3, 5000, H_, D_) and eng._k_cpu.device.type == "cpu"
    assert tuple(eng._block_map.shape) == (H_, 3, TOPK) and bool((eng._block_map == -1).all())
    assert eng._cache_lens.dtype == torch.int32 and bool((eng._cache_lens == TAIL0).all())
    assert eng._tail_block_idx_on_gpu == TOPK - 1
    eng.pool_blocks = 2
    with pytest.raises(RuntimeError, match="pool"):
        REF.allocate(eng, 3, 5000, False)
    cuda_eng = types.SimpleNamespace(head_num=H_, head_dim=D_, block_size=BS, topk=TOPK, dtype=DT,
                                     device=torch.device("cuda"), pool_blocks=0)
    with pytest.raises(ValueError, match="pinned"):
        REF.allocate(cuda_eng, 1, 5000, False)          # refused before any allocation: no GPU needed


def test_decode_refuses_out_of_range_and_bad_inputs():
    eng, twin, keys, values, cis = _setup([6016, 6050], extra=0)
    total = eng._k_cpu.shape[1]
    snap = {k: getattr(eng, k).clone() for k in TABLES}
    k1, v1 = keys[:, :1], values[:, :1]
    topk = _topk(21, twin.pos)
    bad = torch.tensor([total, 6050])                  # request 0 writes past the host rows
    act = torch.ones(2, dtype=torch.bool)
    with pytest.raises(IndexError):
        REF.decode(eng, k1, v1, twin.total_cis, topk, bad, act, bad.clone(), act.clone(), *STUBS)
    _assert_engines_equal(eng, types.SimpleNamespace(**snap), TABLES, "refused decode")
    with pytest.raises(ValueError):
        REF.decode(eng, keys[:, :2], v1, twin.total_cis, topk, twin.pos, act, twin.pos.clone(), act.clone(), *STUBS)
    with pytest.raises(ValueError):
        REF.decode(eng, k1, v1, twin.total_cis, topk, twin.pos, act, twin.pos.clone(), act.int(), *STUBS)
    # an inactive request out of range is not an error and changes nothing
    REF.decode(eng, k1, v1, twin.total_cis, topk, bad, torch.tensor([False, False]), bad.clone(),
               torch.tensor([False, False]), *STUBS)
    for k in PERSIST:
        assert torch.equal(getattr(eng, k), snap[k]), k


# ---------------------------------------------------------------------------
# the device-driven write-back backend (r2_writeback.py) inside the body
# ---------------------------------------------------------------------------
WBMOD = N / "r2_writeback.py"
wbm = _load(WBMOD, "nosi_r2_writeback_for_body_test") if WBMOD.is_file() else None


class _CountingKernel:
    """masked_writeback_torch that records what decode() hands it: the DEVICE mask and positions."""

    def __init__(self):
        self.calls = []

    def __call__(self, k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size):
        self.calls.append((full.clone(), pos.clone(), tail0, block_size))
        wbm.masked_writeback_torch(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size)


def _dev_body(f4_fix=False):
    if wbm is None:
        pytest.skip("no nosi/nosi/r2_writeback.py")
    return rb.DeviceWritebackBody(f4_fix, wbm.masked_writeback_torch)


def test_device_writeback_body_geq():
    run_eq(_dev_body(), 6079, 4, steps=70)


def test_device_writeback_body_grag():
    run_rag(_dev_body(), [6016, 6017, 6079, 6100], steps=120)


def test_device_writeback_body_gpoison():
    assert run_poison(_dev_body()) == [0]


def test_device_writeback_body_reads_the_device_mask():
    if wbm is None:
        pytest.skip("no nosi/nosi/r2_writeback.py")
    k = _CountingKernel()
    body = rb.DeviceWritebackBody(False, k)
    eng, twin, keys, values, cis = _setup([6079, 6016, 6143])
    act = torch.tensor([True, True, False])
    p = _step(body, eng, twin, keys, values, cis, act)
    assert len(k.calls) == 1, "the kernel runs once per decode step"
    full, pos, tail0, bs = k.calls[0]
    assert full.tolist() == [True, False, False], "full = active & (p + 1) % 64 == 0, computed on the device"
    assert torch.equal(pos, p) and (tail0, bs) == (TAIL0, BS)
    with pytest.raises(TypeError):
        rb.DeviceWritebackBody(False, None)


# ---------------------------------------------------------------------------
# the capture-safe decode (decode_fixed, docs/R2_CAPTURE_AUDIT.md fix X3) through the same gates
# ---------------------------------------------------------------------------
class _FixedBody(rb.DeviceWritebackBody):
    """decode() routed through decode_fixed (no host mirror read), raising IndexError as decode() does when the
    device refusal bit is set."""

    def __init__(self, f4_fix, kernel):
        super().__init__(f4_fix, kernel)
        self.err = torch.zeros((), dtype=torch.int32)

    def decode(self, eng, key_states, value_states, kv_bias, topk_idx, pos, active, pos_host, active_host,
               diff_offload, gather_k, gather_v_bias):
        self.err.zero_()
        out = self.decode_fixed(eng, key_states, value_states, kv_bias, topk_idx, pos, active, self.err,
                                diff_offload, gather_k, gather_v_bias)
        if int(self.err):
            raise IndexError("decode_fixed refused the step (err %d)" % int(self.err))
        return out


def _fixed_body(f4_fix=False):
    if wbm is None:
        pytest.skip("no nosi/nosi/r2_writeback.py")
    return _FixedBody(f4_fix, wbm.masked_writeback_torch)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [6016, 6017, 6079])
def test_geq_decode_fixed_equals_shipped_body(S, B):
    run_eq(_fixed_body(), S, B)


def test_grag_decode_fixed_equals_independent_runs():
    stats = run_rag(_fixed_body(), [6016, 6017, 6079, 6100])
    assert stats["ragged_steps"] > 0 and stats["writebacks"] >= 4


def test_gpoison_decode_fixed():
    assert run_poison(_fixed_body()) == [0]


def test_decode_fixed_inactive_and_cross_boundary_rollback():
    run_inactive(_fixed_body())
    run_cross(_fixed_body())


def test_decode_fixed_refuses_on_the_device():
    eng, twin, keys, values, cis = _setup([6016, 6050], extra=0)
    total = eng._k_cpu.shape[1]
    snap = {k: getattr(eng, k).clone() for k in PERSIST}
    body = _fixed_body()
    err = torch.zeros((), dtype=torch.int32)
    bad = torch.tensor([total, 6050])
    act = torch.ones(2, dtype=torch.bool)
    body.decode_fixed(eng, keys[:, :1], values[:, :1], twin.total_cis, _topk(21, twin.pos), bad, act, err, *STUBS)
    assert int(err) == rb.DecodeBody.ERR_POS
    for k in PERSIST:
        assert torch.equal(getattr(eng, k), snap[k]), "refused decode_fixed changed %s" % k
    with pytest.raises(TypeError):                       # the torch backend reads host mirrors: not capture-safe
        rb.DecodeBody(False).decode_fixed(eng, keys[:, :1], values[:, :1], twin.total_cis, _topk(21, twin.pos),
                                          twin.pos, act, err, *STUBS)


def test_body_imports_only_torch():
    tree = ast.parse(BODY.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert mods <= {"__future__", "torch"}, mods
