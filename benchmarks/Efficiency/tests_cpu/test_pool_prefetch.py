"""CPU tests of nosi/nosi/pool_prefetch.py (Task 1.2 of kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md):
the background fetch into pool slots with CPU packing and a soft pin, and the --fetch cpupack A/B. No CUDA.

WHAT RUNS FOR REAL
- cache_engine.CacheEngine: prefill_update and the SHIPPED decode_update_has_kv_bias_pool. The module is loaded in a
  synthetic package with NOSI_POOL_BLOCKS = P and its two CUDA extension builds stubbed; every CUDA / Triton op it calls
  is replaced, as a module attribute, by its CPU twin (pool_prefetch_twins.py). The h2d twin models the kernel's FLAT
  ADDRESSES (L28): narrowed views and the staging's batch stride 0 are read as the kernel reads them.
- The gilfree transport (r2_transport_gilfree_v3.py, worker "cpp", issue_desc, the flat-store option) in CPU mode, with
  the packer v3 of this tree (benchmarks/Efficiency/gilfree_pack.{py,cpp}) BUILT here with OpenMP into a pytest
  temporary directory (gilfree_pack.load refuses a directory under HOME).
- torch.Tensor.pin_memory is a no-op inside each test (prefill_update pins the host rows; there is no CUDA here).

  test_flag_off_is_shipped             Gate 0: the shipped files are unchanged against ded9d41; the binding before attach
                                        and after detach is the shipped bound method; attached with no request, 8 steps
                                        (one block edge) leave every engine tensor bitwise equal to a never-attached
                                        twin; the cpupack method differs from the shipped one in the fetch line only;
                                        the refusals of the configuration.
  test_prefetched_block_is_a_pool_hit  X lands in a pinned pool slot with the host bytes (K, V, the bias from
                                        total_cis); the next step finds it as an ordinary pool hit (SWAP or MOVE_IN at
                                        X's window slot), the host gather does not fetch it, the window holds its host
                                        bytes; control: the same step without the prefetch gathers X; the attended
                                        windows of both runs are bitwise equal.
  test_soft_pin_overflow_is_counted    P + 1 prefetches into one row: P pinned, 1 commit overflow, no pinned block
                                        overwritten; then 1 hit and 3 misses: pool_update evicts the 3 other pinned
                                        slots; the counters equal an independent count from the action plan and
                                        committed == pin_hits + pin_evicted + pins_live. NEGATIVE CONTROL: count=False
                                        breaks the conservation on the same plan (precondition: the evictions happen).
  test_scatter_waits_last_reader       a main-stream reader of pool slot q (a MOVE_IN of Y into the window) that has
                                        not run yet: the commit of X into q waits it, so the window gets Y. PLANTED
                                        NEGATIVE CONTROL skip_reader_wait: the window gets X. Preconditions (L34): the
                                        commit chose q in both runs, the reader ran after the scatter in the plant run,
                                        X's and Y's host rows differ.
  test_cpupack_equals_gpugather_bytes  8 steps (one block edge, a write-back), --fetch cpupack with 4 lanes (chunks)
                                        against the shipped gather: every engine tensor bitwise equal after every step;
                                        the shipped gathers were never called in the cpupack run; the packer took
                                        exactly the misses. PLANTED NEGATIVE CONTROL: the packer's lane shift
                                        (NativePipe.set_plant_slot_shift) changes the window. Preconditions: two
                                        misses in one (layer, head) with different host bytes; the plant job ran
                                        without an error.
  test_h2d_twin_matches_triton_interpreter  (needs triton; skipped without it) the REAL flash_h2d_from_mask and
                                        flash_h2d_from_mask_bias under TRITON_INTERPRET=1 in a fresh process, with the
                                        production argument shapes (destination: the pool rows as a narrowed view;
                                        source: a staging slice with batch stride 0), equal the twin.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import random
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
PKG = "pfcpu"
BASE = "ded9d41"
L, B, H, D, BS, TOPK, P = 2, 2, 2, 8, 64, 8, 4
S = 20 * BS + 60          # tail block 20 holds 60 rows: it fills at the 4th decode step (a block edge, a write-back)
GEN = 2 * BS              # host rows per request: S + GEN = 1468, so 22 complete host blocks fit the packer's S_floor
NONE, SWAP, MOVE_IN, MOVE_OUT = 0, 1, 2, 3
STATE = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_pool_map", "_pool_age", "_pool_target", "_pool_action",
         "_load_mask", "_k_cpu", "_v_cpu", "_cache_lens")


def _by_path(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


TW = _by_path(EFF / "tests_cpu/pool_prefetch_twins.py", "pool_prefetch_twins")
TWIN_OPS = types.SimpleNamespace(h2d=lambda g, c, ids, bs=64: TW.h2d_twin(g, c, ids, bs),
                                 h2d_bias=lambda g, c, bb, bias, ids, bs=64: TW.h2d_twin(g, c, ids, bs, bb, bias))


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
    return _by_path(NOSI / (mod + ".py"), PKG + "." + mod)


def _load_cache_engine():
    """cache_engine.py with NOSI_POOL_BLOCKS = P, its extension builds stubbed, every CUDA / Triton op a CPU twin."""
    full = PKG + ".cache_engine"
    if full in sys.modules:
        return sys.modules[full]
    import torch.utils.cpp_extension as cx
    real = cx.load
    keys = ("NOSI_POOL_BLOCKS", "NOSI_AVAIL", "NOSI_TRANSFER_TRACE")
    old = {k: os.environ.get(k) for k in keys}
    cx.load = lambda name, **kw: types.SimpleNamespace(diff_offload=TW.diff_twin, pool_update=TW.pool_update_twin)
    os.environ["NOSI_POOL_BLOCKS"] = str(P)
    os.environ.pop("NOSI_AVAIL", None)
    os.environ.pop("NOSI_TRANSFER_TRACE", None)
    try:
        CE = _load("cache_engine")
    finally:
        cx.load = real
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert CE.POOL_BLOCKS == P and CE._pool is not None
    CE._flash_pool_swap = TW.pool_swap_twin
    CE.flash_h2d_from_mask = TWIN_OPS.h2d
    CE.flash_h2d_from_mask_bias = TWIN_OPS.h2d_bias
    return CE


@pytest.fixture(scope="module")
def CE():
    return _load_cache_engine()


@pytest.fixture(scope="module")
def PF(CE):
    return _load("pool_prefetch")


@pytest.fixture(scope="module")
def ext(tmp_path_factory):
    """Packer v3 of THIS tree, built (OpenMP)."""
    gp = _by_path(EFF / "gilfree_pack.py", "gilfree_pack_v3_poolpf")
    e = gp.load(str(tmp_path_factory.mktemp("gilfree3_build")))
    assert int(e.VERSION) == 3 and int(e.DESC) >= 2 and int(e.OPENMP) == 1
    return e


@pytest.fixture
def nopin(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)


def _logged(monkeypatch, CE, log):
    """The shipped gathers of the engine (module attributes) as twins that log (gpu pointer, [(h, b, m, block)])."""
    monkeypatch.setattr(CE, "flash_h2d_from_mask", lambda g, c, ids, bs=64: TW.h2d_twin(g, c, ids, bs, log=log))
    monkeypatch.setattr(CE, "flash_h2d_from_mask_bias",
                        lambda g, c, bb, bias, ids, bs=64: TW.h2d_twin(g, c, ids, bs, bb, bias, log=log))


def _world(CE, seed):
    """L CacheEngines after a real prefill_update (CPU, float32), with their total_cis tables (the bias source).
    The memory that prefill_update allocates with torch.empty and does not write gets one fixed value
    (r2_poolpf_cell.define_undefined, the helper of the GPU gate PF-HIT too): two worlds of one seed are then bitwise
    equal before step 0 (CPU job pf_bccd3b1_a: without it they held different leftover bytes)."""
    define_undefined = _cell().define_undefined
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(L):
        e = CE.CacheEngine(gpu_mem_usage=1, num_layers=L, block_size=BS, head_num=H, head_dim=D, dtype=torch.float32,
                           device=torch.device("cpu"), topk=TOPK, has_kv_bias=True)
        e.max_gen_len = GEN
        k = torch.randn((B, S, H, D), generator=g)
        v = torch.randn((B, S, H, D), generator=g)
        tc = torch.randn((B, S + 1024, H), generator=g)
        e.prefill_update(k, v, tc[:, :S], 0, B)
        define_undefined(e)
        layers.append(types.SimpleNamespace(cache_engine=e, total_cis=tc))
    return types.SimpleNamespace(layers=layers)


def _pf(PF, w, ext, **kw):
    kw.setdefault("lanes", 8)
    return PF.PoolPrefetch(w, ext, threads=1, ops=TWIN_OPS, cap_groups=4, ring=2, **kw)


def _random_sel(w, rnd):
    """Per layer (H, B, TOPK): TOPK - 1 distinct complete blocks, then the tail block (seq_length // 64, kept)."""
    out = []
    for layer in w.layers:
        tail = int(layer.cache_engine.seq_length) // BS
        t = torch.empty((H, B, TOPK), dtype=torch.int64)
        for h in range(H):
            for b in range(B):
                t[h, b] = torch.tensor(rnd.sample(range(tail), TOPK - 1) + [tail])
        out.append(t)
    return out


def _step(w, sels, gen, pf=None):
    """One decode step over every layer (pf: its begin_step first); the new token's K / V from `gen`."""
    if pf is not None:
        pf.begin_step()
    outs = []
    for l, layer in enumerate(w.layers):
        k = torch.randn((B, 1, H, D), generator=gen)
        v = torch.randn((B, 1, H, D), generator=gen)
        outs.append([x.clone() for x in layer.cache_engine.decode_update(k, v, layer.total_cis, sels[l])])
    return outs


def _state(w):
    out = []
    for layer in w.layers:
        e = layer.cache_engine
        d = {k: getattr(e, k).clone() for k in STATE}
        d.update(seq=int(e.seq_length), tail=int(e._tail_block_len_on_gpu), stamp=int(e._pool_stamp_base))
        out.append(d)
    return out


# PER-STEP SCRATCH in STATE: undefined before the first step, fully written by every step before any read. Only the
# precondition "one seed gives one world" (before step 0) skips it; every comparison AFTER a step keeps it.
#   _load_mask  torch.empty_like (cache_engine.py:230). diff_offload writes EVERY slot of it in a well-formed step: an
#               old hit gets -1 (diff_offload_kernel.cu:44-48), the k-th new block goes to the k-th free slot (:77-98),
#               and #free == #new because old_map holds distinct ids (or -1 before the first fill, never selected) and
#               new_act holds distinct ids. Its readers in the step come after diff (pool_update cache_engine.py:504,
#               the gathers :518-519, the cpupack fetch); nothing reads it between steps (CPU job pf_f832a3c_a: the
#               only difference of two worlds of one seed before step 0).
# (_new_block_map_buf is scratch of the same kind; STATE does not hold it. We do not fill _load_mask with a value:
# that would hide a diff that does not write every slot at step 0.)
SCRATCH = ("_load_mask",)


def _diff(a, b, skip=()):
    """The names that differ between two _state lists ([] = bitwise equal); `skip`: names not compared."""
    bad = []
    for l, (x, y) in enumerate(zip(a, b)):
        for k in x:
            if k in skip:
                continue
            same =torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k]
            if not same:
                bad.append("layer %d %s" % (l, k))
    return bad


def _rows(t, b, blk, h):
    return t[b, blk * BS:(blk + 1) * BS, h]


# ------------------------------------------------------------------------------------------------ Gate 0
def test_flag_off_is_shipped(CE, PF, ext, nopin):
    # (a) the flag lives only in new files: the shipped engine, model and baseline cell are unchanged
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", BASE, "--", "nosi/nosi/cache_engine.py",
                        "nosi/nosi/nosa_llama.py", "nosi/nosi/flash_cache_engine", "nosi/nosi/r2_transport_gilfree.py",
                        "benchmarks/Efficiency/r2_base_cell.py", "benchmarks/Efficiency/r2_base_fused_step.sh"],
                       capture_output=True, text=True)
    assert r.returncode == 0, "a shipped file differs from %s: %s" % (BASE, r.stdout + r.stderr)
    for f in ("cache_engine.py", "nosa_llama.py"):
        assert "pool_prefetch" not in (NOSI / f).read_text()
    # (b) + (c): a never-attached world A and an attached world B (gpugather, no request), 8 steps, same inputs
    wa, wb = _world(CE, 1), _world(CE, 1)
    assert _diff(_state(wa), _state(wb), SCRATCH) == []      # precondition: one seed gives one world, bitwise
    shipped = [layer.cache_engine.__dict__["decode_update"] for layer in wb.layers]
    assert all(getattr(s, "__func__", None) is CE.CacheEngine.decode_update_has_kv_bias_pool for s in shipped)
    with pytest.raises(ValueError):
        PF.PoolPrefetch(wb, ext, fetch="other", ops=TWIN_OPS)
    with pytest.raises(ValueError):
        PF.PoolPrefetch(wb, ext, fetch="cpupack", bg_mib=1.0, ops=TWIN_OPS)
    with pytest.raises(ValueError):
        PF.PoolPrefetch(wb, None, ops=TWIN_OPS)
    pf = _pf(PF, wb, ext)
    pf.attach()
    assert all(getattr(layer.cache_engine.decode_update, "pool_prefetch", None) is pf for layer in wb.layers)
    rnd_a, rnd_b = random.Random(3), random.Random(3)
    ga, gb = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    for t in range(8):
        oa = _step(wa, _random_sel(wa, rnd_a), ga)
        ob = _step(wb, _random_sel(wb, rnd_b), gb, pf)
        assert all(torch.equal(x, y) for la, lb in zip(oa, ob) for x, y in zip(la, lb)), t
        assert _diff(_state(wa), _state(wb)) == [], t
    assert int(wa.layers[0].cache_engine.seq_length) > (S // BS + 1) * BS        # precondition: a block edge passed
    pf.close()
    assert all(layer.cache_engine.__dict__["decode_update"] is s for layer, s in zip(wb.layers, shipped))
    # (d) the cpupack method = the shipped source with the two gather lines replaced by one call (and its name)
    _, src = PF.derive_cpupack(CE)
    old = textwrap.dedent(inspect.getsource(CE.CacheEngine.__dict__[PF.SHIPPED_NAME])).split("\n")
    new = src.split("\n")
    i = next(k for k, s in enumerate(old) if s.strip() == PF.FETCH_K)
    assert old[i + 1].strip() == PF.FETCH_V and new[i].strip().startswith(PF.FETCH_CALL)
    assert new[i + 1:] == old[i + 2:]
    assert [s.replace(PF.DERIVED_NAME, PF.SHIPPED_NAME) for s in new[:i]] == old[:i]
    assert not any(PF.FETCH_K in s or PF.FETCH_V in s for s in new)
    real = PF.FETCH_K
    try:                                                     # the derivation refuses a source without the anchor
        PF.FETCH_K = real + " # absent"
        with pytest.raises(RuntimeError, match="occur 0 times"):
            PF.derive_cpupack(CE)
    finally:
        PF.FETCH_K = real


# ------------------------------------------------------------------------------------------------ the pool hit
def test_prefetched_block_is_a_pool_hit(CE, PF, ext, nopin, monkeypatch):
    log = []
    _logged(monkeypatch, CE, log)
    w1, w0 = _world(CE, 2), _world(CE, 2)
    assert _diff(_state(w1), _state(w0), SCRATCH) == []      # precondition: one seed gives one world, bitwise
    pf1, pf0 = _pf(PF, w1, ext), _pf(PF, w0, ext)
    pf1.attach()
    pf0.attach()
    s0 = _random_sel(w1, random.Random(7))
    g1, g0 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    _step(w1, s0, g1, pf1)
    _step(w0, s0, g0, pf0)
    l, h, b = 0, 0, 1
    e1, e0 = w1.layers[l].cache_engine, w0.layers[l].cache_engine
    tc = w1.layers[l].total_cis
    tail = int(e1.seq_length) // BS
    resident = set(e1._block_map[h, b].tolist()) | set(e1._pool_map[h, b].tolist())
    X = next(x for x in range(tail) if x not in resident)
    assert pf1.prefetch(torch.tensor([[l, h, b, X]])) == 1
    pf1.tp.flush()
    pf1.begin_step()                                         # the commit
    q = (e1._pool_map[h, b] == X).nonzero().view(-1)
    assert q.numel() == 1
    q = int(q[0])
    assert int(e1._pool_age[h, b, q]) >= PF.PIN_BASE
    for got, want in ((e1._k_gpu, e1._k_cpu), (e1._v_gpu, e1._v_cpu), (e1._kv_bias_gpu, tc)):
        assert torch.equal(_rows(got, b, TOPK + q, h), _rows(want, b, X, h))
    s1 = [t.clone() for t in s0]                             # step 1: one non-tail block of (l, h, b) becomes X
    s1[l][h, b, 0] = X
    log.clear()
    _step(w1, s1, g1)
    fetched1 = [x for p, lst in log if p == e1._k_gpu.data_ptr() for x in lst]
    log.clear()
    pf0.begin_step()
    _step(w0, s1, g0)
    fetched0 = [x for p, lst in log if p == e0._k_gpu.data_ptr() for x in lst]
    assert (h, b) in [(x[0], x[1]) for x in fetched0 if x[3] == X]          # control: without the prefetch, a miss
    assert (h, b) not in [(x[0], x[1]) for x in fetched1 if x[3] == X]      # with it: no host fetch of X
    s = (e1._block_map[h, b] == X).nonzero().view(-1)
    assert s.numel() == 1
    s = int(s[0])
    assert int(e1._pool_action[h, b, s]) in (SWAP, MOVE_IN) and int(e1._pool_target[h, b, s]) == q
    for got, want in ((e1._k_gpu, e1._k_cpu), (e1._v_gpu, e1._v_cpu), (e1._kv_bias_gpu, tc)):
        assert torch.equal(_rows(got, b, s, h), _rows(want, b, X, h))
    c = pf1.report()["counters"]
    assert (c["committed"], c["pin_hits"], c["pin_evicted"], c["pins_live"]) == (1, 1, 0, 0) and c["conserved"]
    for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):        # the pool is a cache of exact bytes
        for l2 in range(L):
            a, z = getattr(w0.layers[l2].cache_engine, name), getattr(w1.layers[l2].cache_engine, name)
            assert torch.equal(a[:, :TOPK * BS], z[:, :TOPK * BS]), (name, l2)
    assert all(torch.equal(x.cache_engine._block_map, y.cache_engine._block_map) for x, y in zip(w0.layers, w1.layers))
    pf1.close()
    pf0.close()


# ------------------------------------------------------------------------------------------------ the soft pin
def _overflow_run(CE, PF, ext, count):
    w = _world(CE, 3)
    pf = _pf(PF, w, ext, count=count)
    pf.attach()
    s0 = _random_sel(w, random.Random(11))
    gen = torch.Generator().manual_seed(6)
    _step(w, s0, gen, pf)
    l, h, b = 0, 0, 0
    e = w.layers[l].cache_engine
    assert bool((e._pool_map[h, b] == -1).all())             # precondition: step 0 parks nothing (every Y is -1)
    tail = int(e.seq_length) // BS
    win = set(e._block_map[h, b].tolist())
    cand = [x for x in range(tail) if x not in win]
    pins, extra, news = cand[:P], cand[P], cand[P + 1:P + 4]
    assert len(news) == 3
    assert pf.prefetch(torch.tensor([[l, h, b, x] for x in pins + [extra]])) == P + 1
    pf.tp.flush()
    pf.begin_step()                                          # the commit: P pins, then no unpinned slot is left
    c0 = pf.report()["counters"]
    pinned_map = sorted(e._pool_map[h, b].tolist())
    pinned_all = bool((e._pool_age[h, b] >= PF.PIN_BASE).all())
    s1 = [t.clone() for t in s0]                             # 1 pinned block enters (a hit), 3 new blocks (misses)
    s1[l][h, b, 0:4] = torch.tensor([pins[0]] + news)
    before = e._pool_map[h, b].clone()
    _step(w, s1, gen)
    act, tgt = e._pool_action[h, b].tolist(), e._pool_target[h, b].tolist()
    evicted = [m for m in range(TOPK) if act[m] == MOVE_OUT and int(before[tgt[m]]) in pins]
    hits = [m for m in range(TOPK) if act[m] in (SWAP, MOVE_IN) and int(before[tgt[m]]) in pins]
    c1 = pf.report()["counters"]
    pf.close()
    return dict(c0=c0, c1=c1, pins=sorted(pins), pinned_map=pinned_map, pinned_all=pinned_all, evicted=evicted,
                hits=hits)


def test_soft_pin_overflow_is_counted(CE, PF, ext, nopin):
    r = _overflow_run(CE, PF, ext, True)
    assert r["c0"]["committed"] == P and r["c0"]["commit_overflow"] == 1 and r["c0"]["requested"] == P + 1
    assert r["pinned_map"] == r["pins"] and r["pinned_all"]  # the overflow dropped the NEW item, no pinned block
    assert len(r["evicted"]) == 3 and len(r["hits"]) == 1    # precondition: pool_update evicted 3 pinned slots
    c = r["c1"]
    assert (c["pin_evicted"], c["pin_hits"], c["pins_live"]) == (3, 1, 0) and c["conserved"]
    n = _overflow_run(CE, PF, ext, False)                    # NEGATIVE CONTROL: the same plan, no pin accounting
    assert len(n["evicted"]) == 3 and len(n["hits"]) == 1    # precondition: the plant run evicts the same way
    assert (n["c1"]["pin_evicted"], n["c1"]["pin_hits"]) == (0, 0) and not n["c1"]["conserved"]


# ------------------------------------------------------------------------------------------------ the reader wait
class Pending:
    """A main-stream event that has not completed: synchronize() runs the deferred main-stream work first."""

    def __init__(self, run):
        self.run, self.done, self.ran_by = run, False, None

    def query(self):
        return self.done

    def synchronize(self, by="wait"):
        if not self.done:
            self.run()
            self.done, self.ran_by = True, by


def _reader_run(CE, PF, ext, plant):
    w = _world(CE, 4)
    pf = _pf(PF, w, ext)
    pf.attach()
    _step(w, _random_sel(w, random.Random(13)), torch.Generator().manual_seed(8), pf)
    l, h, b, q0, s = 0, 1, 0, 0, 2
    e, tc = w.layers[l].cache_engine, w.layers[l].total_cis
    tail = int(e.seq_length) // BS
    win = set(e._block_map[h, b].tolist())
    Y, X = [x for x in range(tail) if x not in win][:2]
    for dst, src in ((e._k_gpu, e._k_cpu), (e._v_gpu, e._v_cpu), (e._kv_bias_gpu, tc)):   # pool slot q0 holds Y
        _rows(dst, b, TOPK + q0, h).copy_(_rows(src, b, Y, h))
    e._pool_map[h, b, q0] = Y
    e._pool_age[h, b] = torch.tensor([0, 10, 11, 12])        # q0 is the least recently used slot
    tgt = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    act = torch.zeros((H, B, TOPK), dtype=torch.int8)
    tgt[h, b, s], act[h, b, s] = q0, MOVE_IN                  # the reader: a MOVE_IN of pool slot q0 into window slot s
    reader = Pending(lambda: TW.pool_swap_twin(e._k_gpu, e._v_gpu, e._kv_bias_gpu, tgt, act, TOPK, BS))
    pf.reader_ev[l] = reader
    pf.skip_reader_wait = plant
    assert pf.prefetch(torch.tensor([[l, h, b, X]])) == 1
    pf.tp.flush()
    pf.begin_step()                                          # the commit of X
    q = int((e._pool_map[h, b] == X).nonzero().view(-1)[0])
    reader.synchronize(by="late")                            # the main stream catches up
    got = [_rows(t, b, s, h).clone() for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu)]
    ys = [_rows(t, b, Y, h).clone() for t in (e._k_cpu, e._v_cpu, tc)]
    xs = [_rows(t, b, X, h).clone() for t in (e._k_cpu, e._v_cpu, tc)]
    pf.close()
    return dict(q=q, q0=q0, ran_by=reader.ran_by, got=got, ys=ys, xs=xs)


def test_scatter_waits_last_reader(CE, PF, ext, nopin):
    r = _reader_run(CE, PF, ext, False)
    assert r["q"] == r["q0"] and r["ran_by"] == "wait"       # the commit chose the reader's slot and waited the reader
    assert all(torch.equal(g, y) for g, y in zip(r["got"], r["ys"]))
    n = _reader_run(CE, PF, ext, True)                       # PLANT: the commit does not wait the reader
    assert n["q"] == n["q0"] and n["ran_by"] == "late"       # preconditions: same slot; the reader ran after the scatter
    assert all(not torch.equal(x, y) for x, y in zip(n["xs"], n["ys"]))   # precondition: X and Y differ
    assert all(torch.equal(g, x) for g, x in zip(n["got"], n["xs"]))      # the difference: the reader read X
    assert not all(torch.equal(g, y) for g, y in zip(n["got"], n["ys"]))


# ------------------------------------------------------------------------------------------------ cpupack A/B
def test_cpupack_equals_gpugather_bytes(CE, PF, ext, nopin, monkeypatch):
    log = []
    _logged(monkeypatch, CE, log)
    wg, wc, wp = _world(CE, 5), _world(CE, 5), _world(CE, 5)
    assert _diff(_state(wg), _state(wc), SCRATCH) == [] and _diff(_state(wg), _state(wp), SCRATCH) == []   # one world
    pg = _pf(PF, wg, ext)
    pc = _pf(PF, wc, ext, fetch="cpupack", lanes=4)
    pp = _pf(PF, wp, ext, fetch="cpupack", lanes=16)
    pp.tp.npipe.set_plant_slot_shift(1)                      # NEGATIVE CONTROL: every take lands one lane up
    for p in (pg, pc, pp):
        p.attach()
    ptr_g = {layer.cache_engine._k_gpu.data_ptr(): l for l, layer in enumerate(wg.layers)}
    ptr_c = {t.data_ptr() for layer in wc.layers for t in (layer.cache_engine._k_gpu, layer.cache_engine._v_gpu)}
    rnd = random.Random(17)
    gg, gc, gp = (torch.Generator().manual_seed(9) for _ in range(3))
    misses, ref0, multi, chunks0, want0 = 0, None, False, None, None
    for t in range(8):
        sel = _random_sel(wg, rnd)
        log.clear()
        _step(wg, sel, gg, pg)
        got = [(ptr_g[p], x) for p, lst in log if p in ptr_g for x in lst]
        misses += len(got)
        if t == 0:
            ref0 = _state(wg)
            per = {}
            for l, (h, b, m, blk) in got:
                per.setdefault((l, h), []).append((b, blk))
            multi = any(len(v) >= 2 and not torch.equal(_rows(wg.layers[l].cache_engine._k_cpu, v[0][0], v[0][1], h),
                                                         _rows(wg.layers[l].cache_engine._k_cpu, v[1][0], v[1][1], h))
                        for (l, h), v in per.items())
            # the chunks of step 0: per layer, the largest number of misses of one KV head, cut by 4 lanes
            want0 = sum(max(-(-len(per.get((l, h), [])) // 4) for h in range(H)) for l in range(L))
        log.clear()
        _step(wc, sel, gc, pc)
        assert not [p for p, _ in log if p in ptr_c], t      # the cpupack run never calls the shipped gathers
        if t == 0:
            chunks0 = pc.stats["cp_chunks"]
            _step(wp, sel, gp, pp)
        assert _diff(_state(wg), _state(wc)) == [], (t, _diff(_state(wg), _state(wc)))
    assert misses > 0 and int(wg.layers[0].cache_engine.seq_length) > (S // BS + 1) * BS   # a block edge passed
    assert want0 >= 2 * L and chunks0 == want0               # 4 lanes: step 0's misses were cut into chunks
    rep = pc.report()
    assert rep["stats"]["cp_items"] == misses and int(rep["transport"]["groups"]) == misses
    assert multi                                             # precondition: two misses of one (layer, head) differ
    pp.tp.poll()                                             # precondition: the plant job ran without an error
    assert any(x.endswith("_k_gpu") for x in _diff(_state(wp), ref0))   # the plant changes the window's K
    for p in (pg, pc, pp):
        p.close()


# ------------------------------------------------------------------------------------------------ the pf1 cell
def _cell():
    sys.path.insert(0, str(EFF))
    import r2_poolpf_cell as PC
    return PC


def test_cell_arms_args_and_verdicts():
    """r2_poolpf_cell's pure parts: the arms, the opt-in flag rules, the cores, the per-arm summary and the verdicts
    (TBD bands give REPORT)."""
    PC = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = PC.parse_args(base)
    assert a.schedule == PC.SCHEDULE.split(",") and not a.pool_prefetch and a.fetch == "gpugather"
    assert PC.arm_config("SH", a) is None
    assert PC.arm_config("BG192", a) == dict(fetch="gpugather", bg_mib=192.0, bg_commit="none")
    assert PC.arm_config("BGP64", a) == dict(fetch="gpugather", bg_mib=64.0, bg_commit="pool")
    assert PC.arm_config("CP", a)["fetch"] == "cpupack"
    for bad in (["--fetch", "cpupack"], ["--pf-bg-mib", "8"], ["--schedule", "SH,PF"], ["--schedule", "PF0,CP"],
                ["--schedule", "SH,XX"], ["--pool", "0"], ["--pool-prefetch", "--fetch", "cpupack", "--pf-bg-mib", "8"],
                ["--seg", "200"]):
        with pytest.raises(SystemExit):
            PC.parse_args(base + bad)
    a = PC.parse_args(base + ["--pool-prefetch", "--fetch", "cpupack", "--schedule", "SH,PF"])
    assert PC.arm_config("PF", a) == dict(fetch="cpupack", bg_mib=0.0, bg_commit="none")
    assert PC.pick_cores(8, range(16)) == list(range(8, 16)) and PC.pick_cores(8, range(8)) is None
    periods = [100.0, 60.0, 62.0, 100.0, 66.0, 68.0, 100.0, 61.0, 63.0]
    segs = [dict(arm="SH", t0=0, t1=3, cpu_s=3.0, wall_s=3.0, pf=None),
            dict(arm="BG64", t0=3, t1=6, cpu_s=8.0, wall_s=4.0,
                 pf=dict(delivery_ms=[(5.0, False), (7.0, True)], cp_fetch_ms_per_layer=[], backlog=[0, 1, 2],
                         stats=dict(steps=3, bytes_issued=3 << 26), counters={}, transport={}, geometry={})),
            dict(arm="SH", t0=6, t1=9, cpu_s=3.0, wall_s=3.0, pf=None)]
    arms = PC.summarize(216, ["SH", "BG64", "SH"], periods, [(0, "first"), (3, "out_len"), (6, "out_len")], segs)
    assert arms["SH"]["steady"]["n"] == 4 and arms["SH"]["steady"]["median_ms"] == 61.5
    assert arms["BG64"]["ratio_median"] == round(67.0 / 61.5, 4) and arms["BG64"]["cores_minus_SH"] == 1.0
    assert arms["BG64"]["delivery"]["n"] == 2 and arms["BG64"]["delivery"]["late"] == 1
    assert arms["BG64"]["backlog"]["growth_per_segment"] == [2] and arms["BG64"]["bytes_per_step"] == 1 << 26
    v = {n: x for n, x, _ in PC.verdicts_pf(arms)}
    assert v["PF-BG64"] == "REPORT" and v["PF-CP"] == "NOT_RUN" and v["PF-DLV64"] == "REPORT"
    old = dict(PC.PRED_PF)
    try:
        PC.PRED_PF["PF-BG64"] = (1.0, 1.05)
        assert {n: x for n, x, _ in PC.verdicts_pf(arms)}["PF-BG64"] == "REFUTED"
        PC.PRED_PF["PF-BG64"] = (1.05, 1.15)
        assert {n: x for n, x, _ in PC.verdicts_pf(arms)}["PF-BG64"] == "CONFIRMED"
    finally:
        PC.PRED_PF.clear()
        PC.PRED_PF.update(old)


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def test_step_script_stops():
    """r2_poolpf_step.sh: a wrong cell and a wrong argument count STOP / usage; at a clean checkout of the step's ROOT,
    DRYRUN pf1 at HEAD STOPS while the registered predictions are TBD (PRED_STAMP None), or prints the pf1 command."""
    script = EFF / "r2_poolpf_step.sh"

    def run(*args):
        return subprocess.run(["bash", str(script), *args], env=dict(_env(), DRYRUN="1"), capture_output=True,
                              text=True, timeout=300)
    r = run("--nosi", "0" * 40, "2190876", "t1", "pf2")
    assert r.returncode == 1 and "cell 'pf2'" in r.stderr, r.stderr[-300:]
    r = run("--nosi", "0" * 40, "2190876", "t1")
    assert r.returncode == 2 and "usage" in r.stderr
    root = "/mnt/beegfs/ojerbi/nosi-r2-poolpf"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if str(ROOT) != root or head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean checkout of %s: the DRYRUN at HEAD needs one" % root)
    r = run("--nosi", head.stdout.strip(), "2190876", "t1", "pf1")
    if _cell().PRED_STAMP is None:
        assert r.returncode == 1 and "registered predictions are TBD" in r.stderr, r.stderr[-300:]
    elif r.returncode == 0:
        assert "--schedule SH\\,PF0\\,BG64\\,BG192\\,CP\\,CP\\,BG192\\,BG64\\,PF0\\,SH" in r.stdout
        assert "NOSI_POOL_BLOCKS=65" in r.stdout and "--batch 216" in r.stdout and "--time=40" in r.stdout
    else:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])


# ------------------------------------------------------------------------------------------------ the real kernel
PF_CHILD = r'''
import importlib.util, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "PF child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "PF child: triton was imported before the interpreter flag was read"
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K = load(sys.argv[1], "flash_h2d_mask_interpreted")
V = load(sys.argv[2], "flash_h2d_mask_bias_interpreted")
TW = load(sys.argv[3], "pool_prefetch_twins_interpreted")
assert K.triton is not None, "PF child: triton did not import"
g = torch.Generator().manual_seed(21)
B, H, D, BS, TOPK, P, LANES = 3, 2, 8, 64, 4, 3, 5
kg = torch.randn((B, (TOPK + P) * BS, H, D), generator=g)
vg = torch.randn((B, (TOPK + P) * BS, H, D), generator=g)
bg = torch.randn((B, (TOPK + P) * BS, H), generator=g)
st_k = torch.randn((2 * LANES * BS, H, D), generator=g)       # two layers of staging; layer 1 is the source
st_v = torch.randn((2 * LANES * BS, H, D), generator=g)
st_b = torch.randn((2 * LANES * BS, H), generator=g)
R = LANES * BS
sk = st_k[R:].unsqueeze(0).expand(B, -1, -1, -1)
sv = st_v[R:].unsqueeze(0).expand(B, -1, -1, -1)
sb = st_b[R:].unsqueeze(0).expand(B, -1, -1)
assert sk.stride(0) == 0 and sb.stride(0) == 0
ids = torch.full((H, B, P), -1, dtype=torch.int64)
lane = [0] * H
for h, b, q in ((0, 0, 1), (0, 2, 0), (1, 1, 2), (1, 0, 0), (0, 1, 2)):
    ids[h, b, q] = lane[h]
    lane[h] += 1
for name, run, twin in (
        ("K", lambda a: K.flash_h2d_from_mask(a[0][:, TOPK * BS:], sk, ids, BS),
         lambda a: TW.h2d_twin(a[0][:, TOPK * BS:], sk, ids, BS)),
        ("V+bias", lambda a: V.flash_h2d_from_mask_bias(a[0][:, TOPK * BS:], sv, a[1][:, TOPK * BS:], sb, ids, BS),
         lambda a: TW.h2d_twin(a[0][:, TOPK * BS:], sv, ids, BS, a[1][:, TOPK * BS:], sb))):
    base = (kg if name == "K" else vg, bg)
    a = [x.clone() for x in base]
    z = [x.clone() for x in base]
    run(a)
    twin(z)
    assert torch.equal(a[0], z[0]) and torch.equal(a[1], z[1]), "PF child: %s kernel differs from the twin" % name
    assert not torch.equal(a[0], base[0]), "PF child: %s moved no row (control)" % name
    assert torch.equal(a[0][:, :TOPK * BS], base[0][:, :TOPK * BS]), "PF child: %s wrote the window" % name
print("PF OK")
'''


def test_h2d_twin_matches_triton_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", PF_CHILD, str(NOSI / "flash_cache_engine/flash_h2d_mask.py"),
                        str(NOSI / "flash_cache_engine/flash_h2d_mask_bias.py"),
                        str(EFF / "tests_cpu/pool_prefetch_twins.py")], env=env, cwd=str(ROOT), capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["PF OK"], (
        "PF child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
