"""r2-draftpool (Task 1.1 of kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md): CPU tests of the DRAFT STEP
on the resident KV (nosi/nosi/cache_engine.py, the r2-draftpool block at the end of the file), of the DR arm's pure
parts (benchmarks/Efficiency/r2_base_cell.py --draft-resident) and of the step script r2_draftpool_step.sh.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_draft_pool.py

(on a compute node, never the login node; PYTHONDONTWRITEBYTECODE=1 because this repository tracks __pycache__.)

HOW THE ENGINE RUNS HERE. cache_engine.py builds two CUDA extensions at import and imports three Triton wrappers. The
loader below imports the REAL module file (this tree's, or ded9d41's from `git show` for Gate 0) as a synthetic package,
with torch.utils.cpp_extension.load and the three wrapper modules replaced by CPU TWINS. So every Python line of the
engine that runs on the GPU runs here. The twins model the kernels (lesson L28): diff_offload, pool_update, the pool
swap and the two gathers read their index tensors (maps, masks, targets, actions, load ids) as FLAT arrays from the
base pointer, as the kernels do, and assert the contiguity that this needs. They address K, V and the bias through
the strides, because the kernels read the strides off the tensors. prefill_update pins host memory: the tests make
pin_memory a no-op. Uninitialised memory is filled (torch.use_deterministic_algorithms with
fill_uninitialized_memory: NaN for floats, the maximum for integers), so two worlds compare bit for bit, and a slot that
no step loaded holds NaN, which is the hazard that it is on the GPU.

THE FAKE WORLD. One engine (one layer); 2 KV heads, head_dim 4, 4 query heads, block 64, topk 64 (the NOSA values); a
pool of P slots; B 1 or B 5 (the two ends of the CPU batch). A selection generator replaces up to `churn` blocks per
(head, request) per step and picks a recently dropped block again with probability `reuse` (pool hits). The tail block
is always in the selection. An attention twin reads the rows below cache_seqlens of the views that it gets and adds
the bias by KV head (flash_attn_nosa on the decode path), in float64 with no matmul (no library kernel whose result can
depend on the address).

Every equality has a planted negative control, and every plant has a precondition that shows that the plant acts (L34).
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import random
import re
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.utils.cpp_extension
import torch.utils.deterministic

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi" / "nosi"
EFF = ROOT / "benchmarks" / "Efficiency"
CE_PATH = NOSI / "cache_engine.py"
STEP = EFF / "r2_draftpool_step.sh"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-draftpool"
BASE = "ded9d41"
HOLDER = "2190876"                       # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)

H, D, HQ, BS, TOPK = 2, 4, 4, 64, 64
BATCHES = (1, 5)
NONE, SWAP, MOVE_IN, MOVE_OUT = 0, 1, 2, 3        # pool_update_kernel.cu / flash_pool_swap.py action codes
_MISSING = object()


@pytest.fixture(autouse=True)
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    prev_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)
        torch.utils.deterministic.fill_uninitialized_memory = prev_fill


@contextlib.contextmanager
def _patched(env=None, attrs=()):
    """Set environment variables (None = unset) and attributes for the block; restore them after it."""
    env = env or {}
    saved_env = {k: os.environ.get(k) for k in env}
    saved = []
    for obj, name, _ in attrs:
        old = obj.__dict__.get(name, _MISSING) if isinstance(obj, type) else getattr(obj, name, _MISSING)
        saved.append((obj, name, old))
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for obj, name, val in attrs:
            setattr(obj, name, val)
        yield
    finally:
        for obj, name, old in reversed(saved):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ===========================================================================
# the kernel twins (flat addresses, L28)
# ===========================================================================
class Log:
    def __init__(self):
        self.reset()

    def reset(self):
        self.k, self.v = [], []
        self.k_calls = self.v_calls = self.diff_calls = self.pool_calls = self.swap_calls = 0


LOG = Log()


def _flat(t, dtype, what):
    assert t.dtype == dtype and t.is_contiguous(), (
        "%s: the kernel reads a flat contiguous %s array from the base pointer (got %s, strides %s)"
        % (what, dtype, t.dtype, t.stride()))
    return t.view(-1)


def twin_diff(old_map, new_act, new_map, load_map):
    """diff_offload_kernel.cu diff_kernel: per (h, b) row at base (h*B + b)*M. An old id that is in new_act keeps its
    slot (load -1); the k-th new id that is not in old goes to the k-th slot whose old id is not in new_act (map and load
    = that id). Entries that the kernel does not decide keep their values (the buffers are reused)."""
    LOG.diff_calls += 1
    om, na = _flat(old_map, torch.int64, "old_map"), _flat(new_act, torch.int64, "new_act")
    nm, lm = _flat(new_map, torch.int64, "new_map"), _flat(load_map, torch.int64, "load_map")
    Hn, Bn, M = old_map.shape
    assert tuple(new_act.shape) == tuple(new_map.shape) == tuple(load_map.shape) == (Hn, Bn, M)
    for h in range(Hn):
        for b in range(Bn):
            base = (h * Bn + b) * M
            old, new = om[base:base + M].tolist(), na[base:base + M].tolist()
            out_map, out_load = nm[base:base + M].tolist(), lm[base:base + M].tolist()
            old_hit = [o in new for o in old]
            new_hit = [x in old for x in new]
            for m in range(M):
                if old_hit[m]:
                    out_map[m], out_load[m] = old[m], -1
            free = [j for j in range(M) if not old_hit[j]]
            k = 0
            for m in range(M):
                if not new_hit[m]:
                    if k < len(free):
                        out_map[free[k]], out_load[free[k]] = new[m], new[m]
                    k += 1
            nm[base:base + M] = torch.tensor(out_map, dtype=torch.int64)
            lm[base:base + M] = torch.tensor(out_load, dtype=torch.int64)


def twin_pool_update(block_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base, tail_slot):
    """pool_update_kernel.cu: one CUDA block per (h, b), one thread per slot m, every tensor read flat at
    (h*B + b)*M or (h*B + b)*P. Phase 1 claims a pool hit with an atomicCAS (the lowest m wins here; the loser is
    demoted to a miss); phase 2 ranks the slots that want a victim; phase 3 orders the unclaimed pool slots by (age,
    slot); phase 4 commits: SWAP / MOVE_IN on a hit (load -1), MOVE_OUT of Y into the rank-th unclaimed slot on a miss,
    nothing on an overflow (Y is dropped). The 48 KB shared-memory check is the TORCH_CHECK of pool_update_cuda."""
    LOG.pool_calls += 1
    bm, lm = _flat(block_map, torch.int64, "block_map"), _flat(load_mask, torch.int64, "load_mask")
    pm, pa = _flat(pool_map, torch.int64, "pool_map"), _flat(pool_age, torch.int64, "pool_age")
    pt, pc = _flat(pool_target, torch.int64, "pool_target"), _flat(pool_action, torch.int8, "pool_action")
    Hn, Bn, M = block_map.shape
    P = pool_map.shape[2]
    assert tuple(load_mask.shape) == tuple(pool_target.shape) == tuple(pool_action.shape) == (Hn, Bn, M)
    assert tuple(pool_map.shape) == tuple(pool_age.shape) == (Hn, Bn, P)
    smem = 8 * 2 * P + 4 * (3 * P + M)
    if smem > 48 * 1024:
        raise RuntimeError("victim pool needs %d B of shared memory" % smem)
    for h in range(Hn):
        for b in range(Bn):
            base, pbase = (h * Bn + b) * M, (h * Bn + b) * P
            s_pmap, s_page = pm[pbase:pbase + P].tolist(), pa[pbase:pbase + P].tolist()
            X, Y = lm[base:base + M].tolist(), bm[base:base + M].tolist()
            claim = [-1] * P
            target, action = [-1] * M, [NONE] * M
            s_hit, hit_q = [-2] * M, [-1] * M
            active = [m != tail_slot and X[m] >= 0 for m in range(M)]
            for m in range(M):                                         # phase 1
                if not active[m]:
                    continue
                q = next((j for j in range(P) if s_pmap[j] == X[m]), -1)
                if q >= 0:
                    if claim[q] == -1:
                        claim[q], s_hit[m], hit_q[m] = m, q, q
                    else:
                        s_hit[m] = -1 if Y[m] >= 0 else -2
                elif Y[m] >= 0:
                    s_hit[m] = -1
            rank, c = [-1] * M, 0                                      # phase 2
            for m in range(M):
                if s_hit[m] == -1:
                    rank[m], c = c, c + 1
            free = sorted((j for j in range(P) if claim[j] < 0), key=lambda j: (s_page[j], j))   # phase 3
            new_pm, new_pa, new_lm = list(s_pmap), list(s_page), list(X)
            for m in range(M):                                         # phase 4
                if not active[m]:
                    continue
                if hit_q[m] >= 0:
                    q = hit_q[m]
                    new_lm[m], target[m] = -1, q
                    if Y[m] >= 0:
                        action[m], new_pm[q], new_pa[q] = SWAP, Y[m], stamp_base + m
                    else:
                        action[m], new_pm[q], new_pa[q] = MOVE_IN, -1, -P + q
                elif 0 <= rank[m] < len(free):
                    q = free[rank[m]]
                    target[m], action[m], new_pm[q], new_pa[q] = q, MOVE_OUT, Y[m], stamp_base + m
            lm[base:base + M] = torch.tensor(new_lm, dtype=torch.int64)
            pm[pbase:pbase + P] = torch.tensor(new_pm, dtype=torch.int64)
            pa[pbase:pbase + P] = torch.tensor(new_pa, dtype=torch.int64)
            pt[base:base + M] = torch.tensor(target, dtype=torch.int64)
            pc[base:base + M] = torch.tensor(action, dtype=torch.int8)


def twin_pool_swap(k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action, topk, block_size=64):
    """flash_pool_swap.py: program (b, h, m) reads target and action flat at h*B*M + b*M + m; P from S_GPU; both sides
    are loaded before either is stored; K, V and bias through their strides."""
    LOG.swap_calls += 1
    Bn, S_GPU, Hn, _ = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert Hn == H2 and Bn == B2 and M == topk and tuple(pool_action.shape) == tuple(pool_target.shape)
    P = S_GPU // block_size - topk
    assert P >= 1
    tg = _flat(pool_target, torch.int64, "pool_target").tolist()
    ac = _flat(pool_action, torch.int8, "pool_action").tolist()
    for b in range(Bn):
        for h in range(Hn):
            for m in range(M):
                idx = h * Bn * M + b * M + m
                act = ac[idx]
                if act == NONE:
                    continue
                q = tg[idx]
                if q < 0 or q >= P:
                    continue
                read_att, read_pool = act in (SWAP, MOVE_OUT), act in (SWAP, MOVE_IN)
                a0, p0 = m * block_size, (topk + q) * block_size
                assert a0 + block_size <= S_GPU and p0 + block_size <= S_GPU
                for t in (k_gpu, v_gpu, kv_bias_buf):
                    att, pool = t[b, a0:a0 + block_size, h].clone(), t[b, p0:p0 + block_size, h].clone()
                    if read_att:
                        t[b, p0:p0 + block_size, h] = att
                    if read_pool:
                        t[b, a0:a0 + block_size, h] = pool


def _programs(gpu_data, cpu_data, load_ids, bs):
    Bn, S_GPU, Hn, _ = gpu_data.shape
    S_CPU = cpu_data.shape[1]
    H2, B2, M = load_ids.shape
    assert (Hn, Bn) == (H2, B2), (gpu_data.shape, load_ids.shape)
    ids = _flat(load_ids, torch.int64, "load_ids").tolist()
    for b in range(Bn):
        for h in range(Hn):
            for m in range(M):
                i = ids[h * Bn * M + b * M + m]
                if i < 0:
                    continue
                assert (i + 1) * bs <= S_CPU, "source block %d beyond the host rows %d" % (i, S_CPU)
                assert (m + 1) * bs <= S_GPU
                yield b, h, m, i


def twin_gather_k(gpu_data, cpu_data, load_ids, block_size):
    """flash_h2d_from_mask: program (b, h, m) copies host rows id*bs .. id*bs+bs-1 of head h into slot m."""
    LOG.k_calls += 1
    for b, h, m, i in list(_programs(gpu_data, cpu_data, load_ids, block_size)):
        gpu_data[b, m * block_size:(m + 1) * block_size, h] = cpu_data[b, i * block_size:(i + 1) * block_size, h]
        LOG.k.append((b, h, m, i))


def twin_gather_v(gpu_data, cpu_data, kv_bias_buf, kv_bias, load_ids, block_size):
    """flash_h2d_from_mask_bias: the same program copies the V rows and the bias rows (kv_bias at the block's rows)."""
    LOG.v_calls += 1
    assert tuple(kv_bias_buf.shape) == tuple(gpu_data.shape[:3])
    for b, h, m, i in list(_programs(gpu_data, cpu_data, load_ids, block_size)):
        gpu_data[b, m * block_size:(m + 1) * block_size, h] = cpu_data[b, i * block_size:(i + 1) * block_size, h]
        kv_bias_buf[b, m * block_size:(m + 1) * block_size, h] = kv_bias[b, i * block_size:(i + 1) * block_size, h]
        LOG.v.append((b, h, m, i))


def fake_attention(q, k_cache, v_cache, bias, cache_seqlens):
    """flash_attn_nosa_with_kvcache on the decode path: rows 0 .. cache_seqlens[b]-1 of the views it gets, the bias
    of KV head h added to the scores of its query group, softmax, the weighted V. float64, broadcast + sum only."""
    Bq, _, Hq, Dq = q.shape
    Hk = k_cache.shape[2]
    out = torch.empty((Bq, 1, Hq, Dq), dtype=torch.float64)
    for b in range(Bq):
        n = int(cache_seqlens[b])
        k = k_cache[b, :n].double().permute(1, 0, 2)                  # (Hk, n, D)
        v = v_cache[b, :n].double().permute(1, 0, 2)
        qq = q[b, 0].double().reshape(Hk, Hq // Hk, Dq)              # (Hk, g, D)
        s = (qq.unsqueeze(2) * k.unsqueeze(1)).sum(-1) / Dq ** 0.5    # (Hk, g, n)
        s = s + bias[b, :n].double().t().unsqueeze(1)
        p = torch.softmax(s, dim=-1)
        out[b, 0] = (p.unsqueeze(-1) * v.unsqueeze(1)).sum(2).reshape(Hq, Dq)
    return out


# ===========================================================================
# the loader: the real module file, the twins in place of CUDA and Triton
# ===========================================================================
def _transformers_stub():
    try:
        import transformers.cache_utils  # noqa: F401
        return {}
    except Exception:
        tr, cu = types.ModuleType("transformers"), types.ModuleType("transformers.cache_utils")
        for n in ("CacheLayerMixin", "DynamicLayer", "Cache", "DynamicCache", "StaticCache"):
            setattr(cu, n, type(n, (), {"__init__": lambda self, *a, **k: None}))
        tr.cache_utils = cu
        return {"transformers": tr, "transformers.cache_utils": cu}


def load_ce(tag, src_dir, P):
    """Import <src_dir>/cache_engine.py (with its transfer_trace.py and avail_policy.py) as package drp_<tag>_p<P>, with
    NOSI_POOL_BLOCKS = P at import (the knob is read once, there), NOSI_AVAIL and NOSI_TRANSFER_TRACE unset, every CUDA
    extension and Triton wrapper replaced by a twin. Cached per (tag, P)."""
    pkg = "drp_%s_p%d" % (tag, P)
    full = pkg + ".cache_engine"
    if full in sys.modules:
        return sys.modules[full]
    p = types.ModuleType(pkg)
    p.__path__ = [str(src_dir)]
    sys.modules[pkg] = p
    f = types.ModuleType(pkg + ".flash_cache_engine")
    f.__path__ = []
    sys.modules[f.__name__] = f
    for sub, attr, fn in (("flash_h2d_mask", "flash_h2d_from_mask", twin_gather_k),
                          ("flash_h2d_mask_bias", "flash_h2d_from_mask_bias", twin_gather_v),
                          ("flash_pool_swap", "flash_pool_swap", twin_pool_swap)):
        m = types.ModuleType("%s.flash_cache_engine.%s" % (pkg, sub))
        setattr(m, attr, fn)
        sys.modules[m.__name__] = m
    built = []

    def fake_load(name, sources, verbose=False, **kw):
        built.append(name)
        if name == "diff_offload":
            return types.SimpleNamespace(diff_offload=twin_diff)
        if name == "nosi_pool":
            return types.SimpleNamespace(pool_update=twin_pool_update)
        raise AssertionError("cache_engine asked for an unknown extension %r" % name)

    stubs = _transformers_stub()
    with _patched({"NOSI_POOL_BLOCKS": str(P), "NOSI_AVAIL": None, "NOSI_TRANSFER_TRACE": None},
                  [(torch.utils.cpp_extension, "load", fake_load)]):
        sys.modules.update(stubs)
        try:
            for sub in ("transfer_trace", "avail_policy", "cache_engine"):
                spec = importlib.util.spec_from_file_location(pkg + "." + sub, str(Path(src_dir) / (sub + ".py")))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
        finally:
            for k in stubs:
                sys.modules.pop(k, None)
    ce = sys.modules[full]
    assert ce.POOL_BLOCKS == P and built == (["diff_offload", "nosi_pool"] if P > 0 else ["diff_offload"]), built
    assert ce._tt.TRACE is None and ce._avail.POLICY is None and ce._avail.SPEC == ""
    return ce


def _git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=check)


@pytest.fixture(scope="session")
def base_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("nosi_ded9d41")
    for f in ("cache_engine.py", "transfer_trace.py", "avail_policy.py"):
        (d / f).write_text(_git("show", "%s:nosi/nosi/%s" % (BASE, f)).stdout)
    return d


def head_ce(P):
    return load_ce("head", NOSI, P)


# ===========================================================================
# the fake world
# ===========================================================================
STATE = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_block_map", "_new_block_map_buf", "_load_mask",
         "_pool_map", "_pool_age", "_pool_target", "_pool_action", "_cache_lens")
SCALARS = ("seq_length", "_tail_block_len_on_gpu", "_tail_block_idx_on_gpu", "_pool_stamp_base")


def bits_equal(a, b):
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return False
    iv = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16,
          torch.float16: torch.int16}.get(a.dtype)
    if iv is None:
        return torch.equal(a, b)
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def snapshot(e):
    return types.SimpleNamespace(**{n: getattr(e, n).clone() for n in STATE}, **{n: getattr(e, n) for n in SCALARS})


def state_diff(e1, e2):
    out = [n for n in STATE if not bits_equal(getattr(e1, n), getattr(e2, n))]
    return out + [n for n in SCALARS if getattr(e1, n) != getattr(e2, n)]


class World:
    """One engine after prefill_update (the real method), a selection generator and a step driver."""

    def __init__(self, ce, B, P, S0, seed, churn=8, reuse=0.5, max_gen=192):
        self.ce, self.B, self.P, self.seed, self.churn, self.reuse = ce, B, P, seed, churn, reuse
        self.rng = random.Random(seed)
        g = torch.Generator().manual_seed(seed)
        e = ce.CacheEngine(gpu_mem_usage=1, num_layers=1, block_size=BS, head_num=H, head_dim=D,
                           dtype=torch.float32, device=torch.device("cpu"), topk=TOPK, has_kv_bias=True)
        e.max_gen_len = max_gen
        k, v = torch.randn(B, S0, H, D, generator=g), torch.randn(B, S0, H, D, generator=g)
        self.total_cis = torch.zeros(B, S0 + 1024, H)
        self.total_cis[:, :S0] = torch.randn(B, S0, H, generator=g) * 0.5
        with _patched(attrs=[(torch.Tensor, "pin_memory", lambda self, *a, **kw: self)]):
            e.prefill_update(k, v, self.total_cis[:, :S0], 0, B)
        self.e = e
        self.cache = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e)])
        T = S0 // BS
        self.cur = [[self.rng.sample(range(T), TOPK - 1) for _ in range(B)] for _ in range(H)]
        self.dropped = [[[] for _ in range(B)] for _ in range(H)]
        self.t = 0

    def enable(self):
        return self.ce.InfLLMv2Cache.enable_draft_resident(self.cache)

    def allowed(self):
        return self.ce.InfLLMv2Cache.draft_resident_allowed(self.cache)

    def selection(self):
        tail = self.e.seq_length // BS
        rows = []
        for h in range(H):
            for b in range(self.B):
                cur, drop = self.cur[h][b], self.dropped[h][b]
                for _ in range(self.rng.randint(0, self.churn)):
                    drop.append(cur.pop(self.rng.randrange(len(cur))))
                while len(cur) < TOPK - 1:
                    if drop and self.rng.random() < self.reuse:
                        c = drop.pop(self.rng.randrange(len(drop)))
                    else:
                        c = self.rng.randrange(tail)
                    if c not in cur:
                        cur.append(c)
                del drop[:-32]
                row = cur + [tail]
                self.rng.shuffle(row)
                rows.append(row)
        return torch.tensor(rows, dtype=torch.int64).view(H, self.B, TOPK)

    def step(self, draft=None, sel=None):
        """One decode step. draft None: call decode_update as it is bound (ded9d41 has no set_draft_resident).
        draft True / False: InfLLMv2Cache.set_draft_resident(draft) first (the cache-level guard)."""
        g = torch.Generator().manual_seed(self.seed * 100003 + self.t)
        key, val = torch.randn(self.B, 1, H, D, generator=g), torch.randn(self.B, 1, H, D, generator=g)
        q = torch.randn(self.B, 1, HQ, D, generator=g)
        cis = torch.randn(self.B, H, generator=g) * 0.5
        sel = self.selection() if sel is None else sel
        self.total_cis[:, self.e.seq_length] = cis          # update_uncompressed_cis writes row cis_len = seq_length
        if draft is not None:
            self.ce.InfLLMv2Cache.set_draft_resident(self.cache, draft)
        LOG.reset()
        out = self.e.decode_update(key, val, self.total_cis, sel)
        rec = dict(sel=sel, out=out, att=fake_attention(q, *out), k=list(LOG.k), v=list(LOG.v), k_calls=LOG.k_calls,
                   v_calls=LOG.v_calls, pool_calls=LOG.pool_calls, swap_calls=LOG.swap_calls)
        self.t += 1
        return rec


def _rows_equal(e, b, h, slot, blk, total_cis):
    g, c = slice(slot * BS, (slot + 1) * BS), slice(blk * BS, (blk + 1) * BS)
    return (bits_equal(e._k_gpu[b, g, h], e._k_cpu[b, c, h]) and bits_equal(e._v_gpu[b, g, h], e._v_cpu[b, c, h])
            and bits_equal(e._kv_bias_gpu[b, g, h], total_cis[b, c, h]))


def residency_errors(w):
    """The invariants after a step, per (head, request): (dup) the window map and the pool map hold no id twice; (win)
    a window slot s != tail that names X holds X's K, V and bias rows; (pool) a pool slot that names Y holds Y's rows;
    (tail) the tail slot names the tail block. Returns the violations (empty = sound)."""
    e, errs = w.e, []
    ts = e._tail_block_idx_on_gpu
    for h in range(H):
        for b in range(w.B):
            win, pool = e._block_map[h, b].tolist(), e._pool_map[h, b].tolist()
            ids = [x for x in win if x >= 0] + [y for y in pool if y >= 0]
            if len(ids) != len(set(ids)):
                errs.append(("dup", h, b))
            if win[ts] != e.seq_length // BS:
                errs.append(("tail", h, b, win[ts]))
            for s, x in enumerate(win):
                if s != ts and x >= 0 and not _rows_equal(e, b, h, s, x, w.total_cis):
                    errs.append(("win", h, b, s, x))
            for qq, y in enumerate(pool):
                if y >= 0 and not _rows_equal(e, b, h, TOPK + qq, y, w.total_cis):
                    errs.append(("pool", h, b, qq, y))
    return errs


def _window_rows(w):
    """Per (h, b): the non-tail window ids, the pool ids, the tail id."""
    e = w.e
    ts = e._tail_block_idx_on_gpu
    out = {}
    for h in range(H):
        for b in range(w.B):
            win = e._block_map[h, b].tolist()
            out[h, b] = ([x for s, x in enumerate(win) if s != ts and x >= 0],
                         [y for y in e._pool_map[h, b].tolist() if y >= 0], win[ts])
    return out


def sel_all_hit(w, seed, n=3):
    """A selection whose new blocks are all in the pool: the window with up to n ids replaced by pool ids."""
    rng = random.Random(seed)
    rows = []
    for h in range(H):
        for b in range(w.B):
            win, pool, tail = _window_rows(w)[h, b]
            assert len(win) == TOPK - 1 and pool, "precondition: a full window and a non-empty pool at (%d, %d)" % (h, b)
            new = list(win)
            for i, y in zip(rng.sample(range(len(new)), min(n, len(pool))), rng.sample(pool, min(n, len(pool)))):
                new[i] = y
            rows.append(new + [tail])
    return torch.tensor(rows, dtype=torch.int64).view(H, w.B, TOPK)


def sel_one_host_miss(w, seed):
    """The window as it is, except one block at (0, 0) that is neither in the window nor in the pool."""
    rng = random.Random(seed)
    rows = []
    for h in range(H):
        for b in range(w.B):
            win, pool, tail = _window_rows(w)[h, b]
            assert len(win) == TOPK - 1
            new = list(win)
            if (h, b) == (0, 0):
                cand = [x for x in range(tail) if x not in win and x not in pool]
                new[rng.randrange(len(new))] = rng.choice(cand)
            rows.append(new + [tail])
    return torch.tensor(rows, dtype=torch.int64).view(H, w.B, TOPK)


# ===========================================================================
# Gate 0: flag off = the shipped step
# ===========================================================================
@pytest.mark.parametrize("B", BATCHES)
def test_flag_off_is_shipped(B, base_dir):
    # (a) STATIC. ded9d41's module statements are a prefix of this tree's (comments aside); the rest only adds names.
    base_src, head_src = _git("show", "%s:nosi/nosi/cache_engine.py" % BASE).stdout, CE_PATH.read_text()
    bt, ht = ast.parse(base_src), ast.parse(head_src)
    n = len(bt.body)
    assert len(ht.body) > n
    for i, (x, y) in enumerate(zip(bt.body, ht.body)):
        assert ast.dump(x) == ast.dump(y), "module statement %d of %s changed" % (i, BASE)
    base_names = {s.name for s in bt.body if isinstance(s, (ast.FunctionDef, ast.ClassDef))}
    base_names |= {t.id for s in bt.body if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name)}
    members = {c.name: {m.name for m in c.body if isinstance(m, ast.FunctionDef)}
               for c in bt.body if isinstance(c, ast.ClassDef)}
    for s in ht.body[n:]:
        if isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            assert s.name not in base_names, "the appended block redefines %s" % s.name
            continue
        assert isinstance(s, ast.Assign) and len(s.targets) == 1, ast.dump(s)
        t = s.targets[0]
        assert isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name), ast.dump(s)
        assert t.value.id in ("CacheEngine", "InfLLMv2Cache") and t.attr not in members[t.value.id], (t.value.id, t.attr)
    assert head_src.count("decode_update_has_kv_bias_pool") == 2     # the guarded bind and the def (no new bind site)
    # (b) STATIC. The files the flag does not need are the files of ded9d41.
    fce = ["nosi/nosi/flash_cache_engine/" + f for f in (
        "__init__.py", "diff_offload.cpp", "diff_offload_kernel.cu", "flash_h2d_mask.py", "flash_h2d_mask_bias.py",
        "flash_pool_swap.py", "pool_update.cpp", "pool_update_kernel.cu")]      # sources only (__pycache__ is tracked)
    r = _git("diff", "--quiet", BASE, "--", "nosi/nosi/nosa_llama.py", "nosi/nosi/avail_policy.py",
             "nosi/nosi/transfer_trace.py", *fce, check=False)
    assert r.returncode == 0, "a file outside cache_engine.py changed: %s" % r.stderr
    # (c) BEHAVIOUR. ded9d41, this tree with the flag off, and this tree enabled with every step shipped: bit-equal
    #     state, returned tensors, gather programs and attention at every step (40 steps; the tail block fills at t 13).
    P, S0 = 6, 64 * 100 + 50
    ce = head_ce(P)
    wb = World(load_ce("base", base_dir, P), B, P, S0, seed=11)
    wh, we = World(ce, B, P, S0, seed=11), World(ce, B, P, S0, seed=11)
    shipped = ce.CacheEngine.__dict__["decode_update_has_kv_bias_pool"]
    assert wh.e.decode_update.__func__ is shipped and getattr(wh.e, "_draft_skip", None) is None
    we.enable()
    bound0 = we.e._draft_shipped_update
    assert bound0.__func__ is shipped and we.e.decode_update is bound0
    for t in range(40):
        rb, rh, rx = wb.step(), wh.step(), we.step(draft=False)
        assert we.e.decode_update is bound0                         # the very object that __init__ made
        assert rb["k_calls"] == rb["v_calls"] == 1
        for r, w in ((rh, wh), (rx, we)):
            assert state_diff(wb.e, w.e) == [], (t, state_diff(wb.e, w.e))
            assert all(bits_equal(x, y) for x, y in zip(rb["out"], r["out"])), t
            assert bits_equal(rb["att"], r["att"]) and r["k"] == rb["k"] and r["v"] == rb["v"], t
    assert wb.e.seq_length // BS > S0 // BS                             # a write-back ran
    assert not torch.isnan(rb["att"]).any()
    # (d) NEGATIVE CONTROL. One draft step in the enabled world. Precondition: it skips a slot (the plant acts).
    sel = sel_one_host_miss(wb, seed=12)
    rd, rb = we.step(draft=True, sel=sel), wb.step(sel=sel)
    assert int(we.e.draft_misses()[0].sum()) == 1
    assert state_diff(wb.e, we.e) != [] and not bits_equal(rb["att"], rd["att"])


def test_draft_body_is_shipped_prefix():
    """KEEP THE TWO BODIES IN SYNC: the draft step's statements before its own section C are the pooled decode's
    statements up to the map commit, without the trace hooks; after that the draft has no gather and no write-back."""
    tree = ast.parse(CE_PATH.read_text())
    cls = next(s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == "CacheEngine")
    pooled = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "decode_update_has_kv_bias_pool")
    draft = next(s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == "_decode_update_draft_resident")

    def is_trace(s):
        if isinstance(s, ast.Assign) and ast.unparse(s.value) == "_tt.TRACE":
            return True
        return isinstance(s, ast.If) and ast.unparse(s.test) == "_tr is not None"

    sh = [s for s in pooled.body[1:] if not is_trace(s)]
    commit = ast.dump(ast.parse("self._block_map.copy_(self._new_block_map_buf, non_blocking=True)").body[0])
    i = next(j for j, s in enumerate(sh) if ast.dump(s) == commit)
    prefix = sh[:i + 1]
    dr = draft.body[1:]
    assert ast.unparse(dr[0]) == "_why = _draft_refusal(self)" and ast.unparse(dr[1]).startswith("if _why is not None:")
    got = dr[2:2 + len(prefix)]
    assert [ast.dump(s) for s in got] == [ast.dump(s) for s in prefix]
    rest = ast.unparse(ast.Module(body=dr[2 + len(prefix):], type_ignores=[]))
    for word in ("flash_h2d_from_mask", "_k_cpu", "_v_cpu", "_kv_bias_gpu[", "if tail_full"):
        assert word not in rest, word
    # NEGATIVE CONTROL: one changed statement is seen. Precondition: the plant changes the dump.
    plant = list(got)
    k = next(j for j, s in enumerate(plant) if ast.unparse(s) == "self._pool_stamp_base += self.topk")
    plant[k] = ast.parse("self._pool_stamp_base += 1").body[0]
    assert ast.dump(plant[k]) != ast.dump(got[k])
    assert [ast.dump(s) for s in plant] != [ast.dump(s) for s in prefix]


# ===========================================================================
# the draft step
# ===========================================================================
@pytest.mark.parametrize("B", BATCHES)
def test_all_hit_draft_equals_shipped(B):
    P, S0 = 6, 64 * 100 + 20
    ce = head_ce(P)
    wa, wd = World(ce, B, P, S0, seed=21), World(ce, B, P, S0, seed=21)
    wa.enable()
    wd.enable()
    for _ in range(12):                                                 # fill the pool
        wa.step(draft=False)
        wd.step(draft=False)
    assert state_diff(wa.e, wd.e) == []
    sel = sel_all_hit(wa, seed=22)
    ra, rd = wa.step(draft=False, sel=sel), wd.step(draft=True, sel=sel)
    # preconditions: the pool served blocks and nothing needs the host
    act = wa.e._pool_action
    assert int(((act == SWAP) | (act == MOVE_IN)).sum()) >= B * H
    assert int((wa.e._load_mask >= 0).sum()) == 0
    assert ra["k_calls"] == ra["v_calls"] == 1 and ra["k"] == []        # the shipped step launched empty gathers
    assert rd["k_calls"] == rd["v_calls"] == 0 and rd["pool_calls"] == rd["swap_calls"] == 1   # the draft path ran
    assert wd.e.decode_update.__func__ is ce.CacheEngine.decode_update_draft_resident
    # the equality: what attention gets, what it gives, and the whole engine state
    assert rd["out"][2].untyped_storage().data_ptr() == wd.e._draft_bias.untyped_storage().data_ptr()
    assert all(bits_equal(x, y) for x, y in zip(ra["out"], rd["out"]))
    assert bits_equal(ra["att"], rd["att"]) and not torch.isnan(ra["att"]).any()
    assert state_diff(wa.e, wd.e) == []
    # NEGATIVE CONTROL: one host miss. Precondition: the draft skips exactly that slot (the plant acts).
    sel = sel_one_host_miss(wa, seed=23)
    ra, rd = wa.step(draft=False, sel=sel), wd.step(draft=True, sel=sel)
    assert int(wd.e._draft_skip.sum()) == 1 and len(ra["k"]) == 1
    assert not bits_equal(ra["att"], rd["att"]) and state_diff(wa.e, wd.e) != []


@pytest.mark.parametrize("B,P,T0", [(1, 6, 100), (5, 6, 100), (1, 65, 160), (5, 65, 160)])
def test_pool_has_no_duplicate_ids(B, P, T0):
    S0 = 64 * T0 + 10                                                  # the tail block fills at t 53
    ce = head_ce(P)
    ws = {name: World(ce, B, P, S0, seed=31) for name in ("main", "stale", "noedit")}
    for w in ws.values():
        w.enable()
    sched = random.Random(32)
    cov = Counter()
    planted = {}
    for t in range(80):
        want = t > 0 and (sched.random() < 0.6 or not ws["main"].allowed())     # always ask at the block edge
        for name in list(ws):
            w = ws[name]
            draft = want and w.allowed()
            pre = w.e._block_map.clone()
            w.step(draft=draft)
            e = w.e
            if name == "main":
                assert residency_errors(w) == [], (t, residency_errors(w)[:5])
                cov["drafts"] += draft
                cov["edges"] += want and not draft
                if draft:
                    cov["skips"] += int(e._draft_skip.sum())
                    cov["parked_skips"] += int((e._draft_skip & (e._pool_action == MOVE_OUT)).sum())
                cov["pool_hits"] += int(((e._pool_action == SWAP) | (e._pool_action == MOVE_IN)).sum())
                cov["overflows"] += int(((e._load_mask >= 0) & (e._pool_action == NONE) & (pre >= 0)).sum())
                continue
            # the plants: the two other ways to leave the map at a skipped slot (precondition: the case occurs)
            if not draft:
                continue
            if name == "stale" and bool((e._draft_skip & (e._pool_action == MOVE_OUT)).any()):
                e._block_map.copy_(torch.where(e._draft_skip, pre, e._block_map))        # keep Y (avail mech=stale)
                errs = residency_errors(w)
                assert any(x[0] == "dup" for x in errs), errs[:5]
                planted[name] = t
                del ws[name]
            elif name == "noedit" and bool(e._draft_skip.any()):
                e._block_map.copy_(torch.where(e._draft_skip, e._draft_miss_ids, e._block_map))   # keep X
                errs = residency_errors(w)
                assert any(x[0] == "win" for x in errs), errs[:5]
                planted[name] = t
                del ws[name]
    assert set(planted) == {"stale", "noedit"}, planted
    # coverage: the test reached the cases that the map edit is for
    assert cov["drafts"] >= 20 and cov["skips"] >= 1 and cov["parked_skips"] >= 1 and cov["edges"] >= 1, cov
    assert cov["pool_hits"] >= 1 and (P > 8 or cov["overflows"] >= 1), cov


@pytest.mark.parametrize("B", BATCHES)
def test_skipped_blocks_listed(B):
    P, S0 = 6, 64 * 100 + 5
    ce = head_ce(P)
    wa, wd = World(ce, B, P, S0, seed=41), World(ce, B, P, S0, seed=41)
    wa.enable()
    wd.enable()
    for t in range(60):
        sel = wa.selection()
        assert torch.equal(sel, wd.selection())
        pre = wa.e._block_map.clone()
        draft = t > 0 and wd.allowed()
        ra = wa.step(draft=False, sel=sel)
        act = wa.e._pool_action
        hits = int(((act == SWAP) | (act == MOVE_IN)).sum())
        if not draft or hits == 0 or not ra["k"]:
            wd.step(draft=False, sel=sel)
            continue
        # precondition met: the shipped step had a pool hit and a host fetch
        rd = wd.step(draft=True, sel=sel)
        slots, ids = wd.e.draft_misses()
        assert ids.dtype == torch.int64 and ids.is_contiguous() and tuple(ids.shape) == (H, B, TOPK)
        assert slots.dtype == torch.bool and tuple(slots.shape) == (H, B, TOPK)
        assert torch.equal(wd.e._load_mask, wa.e._load_mask)
        assert torch.equal(slots, wa.e._load_mask >= 0)
        assert torch.equal(ids, torch.where(wa.e._load_mask >= 0, wa.e._load_mask, torch.full_like(ids, -1)))
        fetched = set(ra["k"])
        assert fetched == set(ra["v"])
        listed = {(b, h, m, int(ids[h, b, m])) for h, b, m in slots.nonzero().tolist()}
        assert listed == fetched                                           # the miss list IS what the shipped step fetched
        assert rd["k_calls"] == rd["v_calls"] == 0 and rd["k"] == []
        assert torch.equal(wd.e._block_map, torch.where(slots, torch.full_like(ids, -1), wa.e._block_map))
        # NEGATIVE CONTROL: diff's raw load mask (before pool_update) is not the fetched set. Precondition: a pool hit
        # happened, so the raw list is a strict superset.
        raw_map, raw = torch.empty_like(pre), torch.empty_like(pre)
        twin_diff(pre, sel, raw_map, raw)
        raw_set = {(b, h, m, int(raw[h, b, m])) for h, b, m in (raw >= 0).nonzero().tolist()}
        assert fetched < raw_set and listed != raw_set
        return
    pytest.fail("no step with a pool hit and a host fetch in 60 steps")


@pytest.mark.parametrize("B", BATCHES)
def test_bias_mask_is_scratch(B):
    P, S0 = 6, 64 * 100 + 30
    ce = head_ce(P)
    MB = ce._avail.MASK_BIAS
    w, wp = World(ce, B, P, S0, seed=51), World(ce, B, P, S0, seed=51)
    w.enable()
    wp.enable()
    for _ in range(6):
        w.step(draft=False)
        wp.step(draft=False)
    sel = sel_one_host_miss(w, seed=52)
    pre_live = w.e._kv_bias_gpu.clone()
    r = w.step(draft=True, sel=sel)
    wp.step(draft=True, sel=sel)
    e = w.e
    skip = e._draft_skip
    assert int(skip.sum()) == 1                                          # precondition: a slot is masked
    bias = r["out"][2]
    # 1. the returned bias is the scratch, with the live view's shape and strides
    assert bias.untyped_storage().data_ptr() != e._kv_bias_gpu.untyped_storage().data_ptr()
    assert tuple(bias.shape) == tuple(e._kv_bias_gpu_att.shape) and bias.stride() == e._kv_bias_gpu_att.stride()
    # 2. scratch = the live view, except MASK_BIAS on the 64 rows of each skipped (slot, head)
    m4 = skip.permute(1, 2, 0).unsqueeze(2).expand(B, TOPK, BS, H).reshape(B, TOPK * BS, H)
    assert int(m4.sum()) == BS and bool((bias[m4] == MB).all())
    assert bits_equal(bias[~m4], e._kv_bias_gpu_att[~m4])
    # 3. the live bias holds no MASK_BIAS, and the skipped rows are the rows before the step (no gather ran)
    assert not bool((e._kv_bias_gpu == MB).any())
    assert bits_equal(e._kv_bias_gpu_att[m4], pre_live[:, :TOPK * BS][m4])
    # 4. the next shipped step: still no MASK_BIAS in the live bias, and the map is sound
    w.step(draft=False)
    assert not bool((w.e._kv_bias_gpu == MB).any()) and residency_errors(w) == []
    # NEGATIVE CONTROL: mask the LIVE view in place (the defect avail_policy.py warns about). Precondition: the plant
    # writes an element. Check 3 sees it.
    ep = wp.e
    before = ep._kv_bias_gpu.clone()
    ep._kv_bias_gpu_att.view(B, TOPK, BS, H).masked_fill_(ep._draft_bias_skip, MB)
    assert not bits_equal(before, ep._kv_bias_gpu)
    assert bool((ep._kv_bias_gpu == MB).any())


@pytest.mark.parametrize("B", BATCHES)
def test_refuses_block_edge(B):
    P, S0 = 6, 64 * 100 + 61                                           # tail_len 61 after the prefill
    ce = head_ce(P)
    w = World(ce, B, P, S0, seed=61)
    w.enable()
    # the first-step rule: no draft before the first shipped step after the prefill; nothing moves
    snap = snapshot(w.e)
    with pytest.raises(RuntimeError, match="first shipped decode step"):
        ce.InfLLMv2Cache.set_draft_resident(w.cache, True)
    assert w.e.decode_update is w.e._draft_shipped_update and state_diff(snap, w.e) == []
    # PLANT (the rule guards a real hazard): bypass the refusal before the first shipped step. Precondition: every
    # non-tail slot is skipped. Then the never-loaded slots (NaN) reach attention under the mask.
    wp = World(ce, B, P, S0, seed=61)
    wp.enable()
    with _patched(attrs=[(ce, "_draft_refusal", lambda eng: None)]):
        wp.e.decode_update = wp.e.decode_update_draft_resident
        rp = wp.step()
    assert int(wp.e._draft_skip.sum()) == (TOPK - 1) * H * B
    assert bool(torch.isnan(rp["att"]).all())
    # the block edge
    w.step(draft=False)                                                # tail_len 61 -> 62
    assert w.allowed()
    w.step(draft=True)                                                 # 62 -> 63: one row is left, so allowed
    assert w.e._tail_block_len_on_gpu == BS - 1 and not w.allowed()
    snap = snapshot(w.e)
    with pytest.raises(ce.DraftBlockEdge, match="tail block fills"):
        ce.InfLLMv2Cache.set_draft_resident(w.cache, True)
    assert w.e.decode_update is w.e._draft_shipped_update              # the refusal bound the shipped step
    assert state_diff(snap, w.e) == []                                 # and moved nothing
    # the engine-level guard: the draft step bound by hand raises before this engine writes anything
    w.e.decode_update = w.e.decode_update_draft_resident
    with pytest.raises(ce.DraftBlockEdge):
        w.step()
    assert state_diff(snap, w.e) == []
    # precondition of the guard (it guards a real write): the shipped step at the edge copies the tail block to the host
    w.e.decode_update = w.e._draft_shipped_update
    host_before = w.e._k_cpu.clone()
    w.step(draft=False)
    assert w.e._tail_block_len_on_gpu == 0 and not bits_equal(host_before, w.e._k_cpu)
    assert residency_errors(w) == [] and w.allowed()


def test_enable_refusals():
    ce0 = head_ce(0)
    w0 = World(ce0, 1, 0, 64 * 70 + 3, seed=71)
    assert w0.e.decode_update.__func__ is ce0.CacheEngine.__dict__["decode_update_has_kv_bias"]
    with pytest.raises(RuntimeError, match="needs the victim pool"):
        w0.enable()
    ce = head_ce(6)
    e = ce.CacheEngine(gpu_mem_usage=1, num_layers=1, block_size=BS, head_num=H, head_dim=D, dtype=torch.float32,
                       device=torch.device("cpu"), topk=TOPK, has_kv_bias=True)
    with pytest.raises(RuntimeError, match="after the prefill"):
        ce.InfLLMv2Cache.enable_draft_resident(types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=e)]))
    w = World(ce, 2, 6, 64 * 70 + 3, seed=72)
    live = w.e._kv_bias_gpu
    skip = torch.zeros((2, TOPK, 1, H), dtype=torch.bool)
    with pytest.raises(ValueError, match="is the live bias"):
        w.e.enable_draft_resident(live, skip)
    with pytest.raises(ValueError, match="shape, strides"):
        w.e.enable_draft_resident(torch.zeros(2, 5, H), skip)
    with pytest.raises(ValueError, match="skip buffer"):
        w.e.enable_draft_resident(torch.zeros_like(live), torch.zeros((2, TOPK, H), dtype=torch.bool))
    with pytest.raises(RuntimeError, match="not draft-capable"):
        ce.InfLLMv2Cache.set_draft_resident(w.cache, True)
    w.enable()
    with pytest.raises(RuntimeError, match="already enabled"):
        w.enable()
    w2 = World(ce, 1, 6, 64 * 70 + 3, seed=73)
    with _patched(attrs=[(ce._avail, "POLICY", object())]):
        with pytest.raises(RuntimeError, match="availability"):
            w2.enable()


# ===========================================================================
# the DR arm of r2_base_cell.py (pure parts) and the step script
# ===========================================================================
def _bc():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


def test_dr_kind_schedule():
    BC = _bc()
    S0 = BC.S0

    def kinds(steps):
        rw = {t for t, _ in BC.expected_rewarms(S0, steps)}
        return [BC.dr_kind(t, t in rw, (S0 % BS + t) % BS + 1 < BS, 3, 4) for t in range(steps)]

    k = kinds(512)
    assert k[:8] == ["rewarm", "g0", "g0", "g0", "draft", "draft", "draft", "ship"]
    assert [t for t, x in enumerate(k) if x == "edge"] == list(range(63, 512, 64))
    assert [t for t, x in enumerate(k) if x == "rewarm"] == [t for t, _ in BC.expected_rewarms(S0, 512)]
    n = Counter(k)
    assert 340 <= n["draft"] <= 390 and 100 <= n["ship"] <= 130 and "draft" not in k[:4], n
    m = kinds(80)                                                        # dr1m: one edge and one re-warm after t 0
    assert m[63] == "edge" and m[64] == "rewarm" and Counter(m)["draft"] >= 40


def test_parse_args_draft_resident():
    BC = _bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216"]
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--draft-resident"])
    assert a.arms == ["SH", "DR"] and (a.dr_k, a.dr_gate0, a.dr_bands_parsed) == (3, 4, None)
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-bands", "52:57:0.8:0.9"])
    assert a.dr_bands_parsed == (52.0, 57.0, 0.8, 0.9)
    assert BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-bands", "TBD"]).dr_bands_parsed is None
    a = BC.parse_args(base + ["--arms", "SH", "--pool", "65", "--mbase"])           # mb1: unchanged
    assert a.arms == ["SH"] and not a.draft_resident
    for bad in (["--arms", "SH", "--draft-resident"], ["--pool", "65", "--draft-resident"],
                ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-k", "0"],
                ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-gate0", "0"],
                ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-bands", "57:52:0.8:0.9"],
                ["--arms", "SH", "--pool", "65", "--draft-resident", "--dr-bands", "1:2:3"],
                ["--arms", "SH", "--pool", "65", "--dr-bands", "1:2:0.1:0.2"]):
        with pytest.raises(SystemExit):
            BC.parse_args(base + bad)


def test_dr_summary_and_verdicts():
    BC = _bc()
    kinds = ["rewarm", "g0", "g0", "draft", "draft", "ship", "edge"]
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
    pred = inputs[:, 1:].clone()
    pred[1, 3] = 99                                                      # one draft step disagrees
    res = dict(kinds=kinds, periods=[100.0, 60.0, 61.0, 50.0, 52.0, 62.0, 63.0], pred=pred, skipped=[0, 0, 0, 10, 6, 0, 0],
               streams_per_step=4, rewarms=[(0, "first")], prefill_s=1.0, first_token_agreement=2)
    dr = BC.dr_summary(2, res, inputs, 3)
    assert dr["stats"]["draft"]["median_ms"] == 51.0 and dr["stats"]["ship"]["n"] == 1 and dr["n_kind"]["edge"] == 1
    g = dr["agreement"]
    assert g["g0_equal"] == g["g0_total"] == 6 and g["per_kind"]["draft"] == 0.75
    assert dr["skip_per_stream"] == (10 + 6) / (2 * 4)
    # NEGATIVE CONTROL: a difference inside the prefix is counted (DR-G0 fails on it)
    pred2 = pred.clone()
    pred2[0, 1] = 99
    g2 = BC.dr_summary(2, dict(res, pred=pred2), inputs, 3)["agreement"]
    assert g2["g0_equal"] == 5 and g2["g0_first_diff_request_step"] == [0, 1]
    sh = dict(rates=dict(steady_step_ms=60.0), gpu_peak_alloc_GB=82.0)
    dr.update(gpu_peak_alloc_GB=82.02)
    v = {n: x for n, x, _ in BC.verdicts_draft(sh, dr, (50.0, 52.0, 0.8, 0.9))}
    assert (v["DR-STEP"], v["DR-RATIO"], v["DR-SHIP"], v["DR-SKIP"], v["DR-AGREE"], v["DR-MEM"]) == (
        "CONFIRMED", "CONFIRMED", "REPORT", "REPORT", "REPORT", "REPORT"), v
    v = {n: x for n, x, _ in BC.verdicts_draft(sh, dr, (52.5, 55.0, 0.9, 0.95))}
    assert (v["DR-STEP"], v["DR-RATIO"]) == ("REFUTED", "REFUTED")
    v = {n: x for n, x, _ in BC.verdicts_draft(sh, dr, None)}
    assert (v["DR-STEP"], v["DR-RATIO"]) == ("REPORT", "REPORT")
    assert BC.verdicts_draft(sh, None, None)[0][1] == "NOT_RUN"


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def _clean_head():
    """The checkout's HEAD if this checkout IS the step script's ROOT and has no tracked change, else None."""
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD", check=False)
    dirty = _git("status", "--porcelain", "--untracked-files=no", check=False)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def _step(cell, sha="0" * 40, dry=True):
    e = _env()
    if dry:
        e["DRYRUN"] = "1"
    else:
        e.pop("DRYRUN", None)
    return subprocess.run(["bash", str(STEP), "--nosi", sha, HOLDER, "t1", cell], env=e, capture_output=True,
                          text=True, timeout=300)


def test_step_script_dr1():
    text = STEP.read_text()
    preds = [ln for ln in text.splitlines() if "PREDICTION:" in ln]
    assert len(preds) >= 10 and all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
    assert "NOT STAMPED" in text
    for bad in ("dr2", "mb1", "dr1x", ""):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    if re.search(r"^DR1_BANDS=TBD\s", text, re.M):                     # outside DRYRUN: STOP before any pin or claim
        r = _step("dr1", dry=False)
        assert r.returncode == 1 and "--dr-bands TBD" in r.stderr, r.stderr[-300:]
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    want = {"dr1": (216, 512, 60, "240G", True), "dr1m": (8, 80, 20, "100G", False)}
    for cell, (B, steps, smin, mem, mb) in want.items():
        r = _step(cell, sha)
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        bands = re.search(r"^DR1_BANDS=(\S+)", text, re.M).group(1) if cell == "dr1" else "none"
        line = "cell %s: C 128 (pool 65) B %d arms SH,DR steps %d STEP_MIN %d mem %s bands %s" % (
            cell, B, steps, smin, mem, bands)
        assert line in r.stdout, (cell, r.stdout[-600:])
        assert "NOSI_POOL_BLOCKS=65" in r.stdout and "--pool 65 --draft-resident --dr-k 3 --dr-gate0 4" in r.stdout
        assert ("--mbase" in r.stdout) == mb and ("--dr-bands " in r.stdout) == (cell == "dr1")
        assert "--batch %d" % B in r.stdout and "--steps %d" % steps in r.stdout
        assert "--mem=%s" % mem in r.stdout and "--time=%d" % smin in r.stdout
        assert "drp_t1_%s_%s" % (HOLDER, cell) in r.stdout and "--job-name=drp-%s-t1-%s" % (cell, HOLDER) in r.stdout
    BC = _bc()                                                           # dr1's python arguments parse
    a = BC.parse_args(["--out", "x", "--workload", "w", "--batch", "216", "--steps", "512", "--arms", "SH", "--pool",
                       "65", "--draft-resident", "--dr-k", "3", "--dr-gate0", "4", "--dr-bands", "TBD", "--mbase"])
    assert a.arms == ["SH", "DR"] and a.mbase
