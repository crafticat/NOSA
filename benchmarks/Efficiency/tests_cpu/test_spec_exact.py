"""R2 verify branch (2026-10-20): CPU tests of the EXACT verify (nosi/nosi/spec_loop.py, section EXACT VERIFY) and of
its GPU gate (benchmarks/Efficiency/spec_exact_gate.py, spec_exact_r2_step.sh).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_spec_exact.py

(on a compute node, never the login node; PYTHONDONTWRITEBYTECODE=1 because this repository does not ignore
__pycache__, and spec_exact_r2_step.sh refuses a checkout with untracked files under nosi/ or benchmarks/.)

HOW THE SHIPPED CODE RUNS HERE. cache_engine.py compiles a CUDA extension at import, so it cannot be imported. The
SOURCE of the shipped methods (CacheEngine.decode_update_has_kv_bias, decode_update_has_kv_bias_verify_slots,
spec_draft_update; InfLLMv2CacheLayer's four decode table updates, decode_update_kv, spec_draft_update_kv;
InfLLMv2Cache.decode_update_kv, spec_draft_update_kv, get_seq_length; the module function _bias_rows) is extracted
by AST and executed on fake engines, with diff_offload and the two Triton gathers replaced by pure-torch twins that
address exactly as the kernels do (diff_offload_kernel.cu, flash_h2d_mask.py, flash_h2d_mask_bias.py), and
avail_policy replaced by a stub module whose POLICY the store hook binds. So the engine body that runs on the GPU is
the body that runs here, line for line.

THE FAKE MODEL. Per layer it does what LlamaLayer.decode_forward does, in its order: k / v / q / cis from the hidden
state, the four table updates, a selection (a deterministic function of the hidden state and the tables, the live
tail forced, only ids the pooling grid can name), the engine update through the cache, an attention that reads
exactly the rows below cache_seqlens of the returned views (softmax plus a row-order-sensitive term), the residual.
Its draft step runs the shipped draft engine path (spec_draft_update: tail write, plan, side-stream prefetch twin,
masked bias); its drafted TOKENS come from an oracle (the reference sequence with scheduled corruptions), so the
acceptance pattern is controlled. A REFERENCE world (the same seed) runs the shipped decode only.

Node list (registered for the lead's CPU run): see the test function names; every test is independent.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import re
import textwrap
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi" / "nosi"
SPEC_LOOP = NOSI / "spec_loop.py"
CACHE_ENGINE = NOSI / "cache_engine.py"
NOSA_LLAMA = NOSI / "nosa_llama.py"
GATE = ROOT / "benchmarks" / "Efficiency" / "spec_exact_gate.py"
STEP = ROOT / "benchmarks" / "Efficiency" / "spec_exact_r2_step.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sl = _load(SPEC_LOOP, "nosi_spec_loop_exact_cpu_test")
CE_SRC = CACHE_ENGINE.read_text()
CE_TREE = ast.parse(CE_SRC)


# ===========================================================================
# the sync counter (a stub that counts host syncs at the Python level)
# ===========================================================================
class SyncCounter:
    """Counts every Python-level conversion of a tensor to host data (bool / int / float / index / item / tolist /
    numpy / cpu), every tensor built from host data (torch.tensor / as_tensor: a pageable H2D copy on the device,
    i.e. a stream sync), every op whose output size depends on the data (nonzero, masked_select, unique, a boolean
    index). The twins of the GPU kernels and the harness's own bookkeeping run under pause()."""

    METHODS = ("__bool__", "__int__", "__float__", "__index__", "item", "tolist", "numpy", "cpu", "nonzero", "masked_select", "unique")
    FUNCS = ("tensor", "as_tensor", "nonzero", "masked_select", "unique")

    def __init__(self):
        self.n = 0
        self.paused = 0
        self.what = []

    @contextlib.contextmanager
    def pause(self):
        self.paused += 1
        try:
            yield
        finally:
            self.paused -= 1

    def hit(self, what):
        if not self.paused:
            self.n += 1
            self.what.append(what)


SYNC = SyncCounter()


def _has_bool_index(idx):
    if torch.is_tensor(idx):
        return idx.dtype == torch.bool
    if isinstance(idx, tuple):
        return any(torch.is_tensor(i) and i.dtype == torch.bool for i in idx)
    return False


@contextlib.contextmanager
def counting_syncs(counter):
    missing = object()
    saved_m, saved_f = {}, {}

    def wrap(name, orig):
        def f(*a, **k):
            counter.hit(name)
            with counter.pause():
                return orig(*a, **k)
        return f

    for name in SyncCounter.METHODS:
        saved_m[name] = torch.Tensor.__dict__.get(name, missing)
        setattr(torch.Tensor, name, wrap(name, getattr(torch.Tensor, name)))
    orig_gi = torch.Tensor.__getitem__
    saved_m["__getitem__"] = torch.Tensor.__dict__.get("__getitem__", missing)

    def gi(self, idx):
        if _has_bool_index(idx):
            counter.hit("bool-index")
        with counter.pause():
            return orig_gi(self, idx)
    torch.Tensor.__getitem__ = gi
    for name in SyncCounter.FUNCS:
        saved_f[name] = getattr(torch, name)
        setattr(torch, name, wrap("torch." + name, saved_f[name]))
    try:
        yield counter
    finally:
        for name, old in saved_m.items():
            if old is missing:
                delattr(torch.Tensor, name)
            else:
                setattr(torch.Tensor, name, old)
        for name, old in saved_f.items():
            setattr(torch, name, old)


# ===========================================================================
# the kernel twins
# ===========================================================================
def twin_diff(old_map, new_act, new_map, load_map):
    """diff_offload_kernel.cu diff_kernel, per (h, b) row: an old id present in new_act keeps its slot (load -1);
    the k-th new id absent from old goes to the k-th slot whose old id is absent from new_act (map and load =
    that id). Entries the kernel does not write keep their previous values."""
    with SYNC.pause():
        H, B, M = old_map.shape
        for h in range(H):
            for b in range(B):
                old = old_map[h, b].tolist()
                new = new_act[h, b].tolist()
                old_hit = [o in new for o in old]
                new_hit = [x in old for x in new]
                for m in range(M):
                    if old_hit[m]:
                        new_map[h, b, m] = old[m]
                        load_map[h, b, m] = -1
                free = [j for j in range(M) if not old_hit[j]]
                k = 0
                for m in range(M):
                    if not new_hit[m]:
                        if k < len(free):
                            new_map[h, b, free[k]] = new[m]
                            load_map[h, b, free[k]] = new[m]
                        k += 1


class GatherLog:
    def __init__(self):
        self.k_entries = 0
        self.v_entries = 0


GLOG = GatherLog()


def _programs(gpu_data, cpu_data, load_ids, bs):
    B, S_GPU, H, D = gpu_data.shape
    S_CPU = cpu_data.shape[1]
    H2, B2, M = load_ids.shape
    assert (H, B) == (H2, B2), (gpu_data.shape, load_ids.shape)
    assert load_ids.dtype == torch.int64 and load_ids.is_contiguous(), "the kernels read load_ids as a flat int64 (H, B, M) array"
    same = gpu_data.data_ptr() == cpu_data.data_ptr()
    for b in range(B):
        for h in range(H):
            for m in range(M):
                i = int(load_ids[h, b, m])
                if i < 0:
                    continue
                assert (i + 1) * bs <= S_CPU, "source block %d beyond the source rows %d" % (i, S_CPU)
                assert (m + 1) * bs <= S_GPU
                if same:
                    assert i >= M, "a D2D launch would read slot %d of its own destination range [0, %d)" % (i, M)
                yield b, h, m, i


def twin_gather_k(gpu_data, cpu_data, load_ids, block_size):
    """flash_h2d_from_mask: program (b, h, m) copies source rows id*bs .. id*bs+bs-1 of head h into slot m."""
    with SYNC.pause():
        for b, h, m, i in list(_programs(gpu_data, cpu_data, load_ids, block_size)):
            gpu_data[b, m * block_size:(m + 1) * block_size, h] = cpu_data[b, i * block_size:(i + 1) * block_size, h]
            GLOG.k_entries += 1


def twin_gather_v(gpu_data, cpu_data, kv_bias_buf, kv_bias, load_ids, block_size):
    """flash_h2d_from_mask_bias: the same program copies the V rows and the bias rows (from kv_bias at the block's
    rows)."""
    with SYNC.pause():
        assert tuple(kv_bias_buf.shape) == tuple(gpu_data.shape[:3])
        for b, h, m, i in list(_programs(gpu_data, cpu_data, load_ids, block_size)):
            gpu_data[b, m * block_size:(m + 1) * block_size, h] = cpu_data[b, i * block_size:(i + 1) * block_size, h]
            kv_bias_buf[b, m * block_size:(m + 1) * block_size, h] = kv_bias[b, i * block_size:(i + 1) * block_size, h]
            GLOG.v_entries += 1


TWIN_OPS = sl.GatherOps(gather_k=twin_gather_k, gather_v=twin_gather_v)
AVAIL = types.SimpleNamespace(POLICY=None)
NS = dict(torch=torch, F=F, diff=types.SimpleNamespace(diff_offload=twin_diff), flash_h2d_from_mask=twin_gather_k,
          flash_h2d_from_mask_bias=twin_gather_v, _avail=AVAIL, _tt=types.SimpleNamespace(TRACE=None), _sl=sl,
          KV_BIAS_SCALE=1.0)


def _cls(name):
    return next(n for n in CE_TREE.body if isinstance(n, ast.ClassDef) and n.name == name)


def _fn_from(node):
    ns = dict(NS)
    exec(compile(textwrap.dedent(ast.get_source_segment(CE_SRC, node)), str(CACHE_ENGINE), "exec"), ns)
    return ns[node.name]


def _method(cls, name):
    return _fn_from(next(n for n in _cls(cls).body if isinstance(n, ast.FunctionDef) and n.name == name))


NS["_bias_rows"] = _fn_from(next(n for n in CE_TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_bias_rows"))

# ===========================================================================
# the fake world
# ===========================================================================
B, H, D, BS, TOPK, R, LYR = 3, 2, 4, 4, 8, 6, 2
S0 = BS * 24 + 1            # the prompt: 24 complete blocks and one tail row
HOST_ROWS = S0 + 160
V, DH = 23, 12
KS, KST = 32, 16            # the compression window and stride (pooling_block_size, pooling_stride)


class FakeEngine:
    decode_update_has_kv_bias = _method("CacheEngine", "decode_update_has_kv_bias")
    decode_update_has_kv_bias_verify_slots = _method("CacheEngine", "decode_update_has_kv_bias_verify_slots")
    spec_draft_update = _method("CacheEngine", "spec_draft_update")

    def __init__(self, g, total_cis):
        self.head_num, self.head_dim, self.block_size, self.topk = H, D, BS, TOPK
        self.pool_blocks, self.verify_round_slots, self._verify_slots = 0, R, R + 2
        W = TOPK + R + 2
        self._k_cpu = torch.randn(B, HOST_ROWS, H, D, generator=g)
        self._v_cpu = torch.randn(B, HOST_ROWS, H, D, generator=g)
        self._k_gpu = torch.zeros(B, W * BS, H, D)
        self._v_gpu = torch.zeros(B, W * BS, H, D)
        self._kv_bias_gpu = torch.zeros(B, W * BS, H)
        self._att_rows = TOPK * BS
        self._k_gpu_att = self._k_gpu[:, :self._att_rows]
        self._v_gpu_att = self._v_gpu[:, :self._att_rows]
        self._kv_bias_gpu_att = self._kv_bias_gpu[:, :self._att_rows]
        self._block_map = torch.full((H, B, TOPK), -1, dtype=torch.int64)
        self._new_block_map_buf = torch.full((H, B, TOPK), -5, dtype=torch.int64)
        self._load_mask = torch.full((H, B, TOPK), -5, dtype=torch.int64)
        self.seq_length = S0
        tl = S0 % BS
        self._tail_block_len_on_gpu = tl
        self._tail_block_idx_on_gpu = TOPK - 1
        ts = (TOPK - 1) * BS
        self._k_gpu[:, ts:ts + tl] = self._k_cpu[:, S0 - tl:S0]
        self._v_gpu[:, ts:ts + tl] = self._v_cpu[:, S0 - tl:S0]
        self._kv_bias_gpu[:, ts:ts + tl] = total_cis[:, S0 - tl:S0]
        self._block_map[..., TOPK - 1] = S0 // BS
        self._cache_lens = torch.full((B,), (TOPK - 1) * BS + tl, dtype=torch.int32)
        self.decode_update = self.decode_update_has_kv_bias_verify_slots


class FakeLayer:
    update_no_compress_k_decode = _method("InfLLMv2CacheLayer", "update_no_compress_k_decode")
    update_compress_k_decode = _method("InfLLMv2CacheLayer", "update_compress_k_decode")
    update_cis = _method("InfLLMv2CacheLayer", "update_cis")
    update_uncompressed_cis = _method("InfLLMv2CacheLayer", "update_uncompressed_cis")
    decode_update_kv = _method("InfLLMv2CacheLayer", "decode_update_kv")
    spec_draft_update_kv = _method("InfLLMv2CacheLayer", "spec_draft_update_kv")
    get_seq_length = _method("InfLLMv2CacheLayer", "get_seq_length")

    def __init__(self, g):
        self.total_cis = torch.zeros(B, S0 + 1024, H)
        self.total_cis[:, :S0] = torch.rand(B, S0, H, generator=g) - 0.5
        self.cis_len, self.cis_prefilled_batch = S0, B
        self.cache_engine = FakeEngine(g, self.total_cis)
        self.seq_length = S0
        self.no_compress_k_cache = torch.randn(B, KS, H, D, generator=g)
        self.no_compress_k_len = KST + S0 % KST
        m = 9
        self.compress_k_cache_varlen = torch.randn(B, m, H, D, generator=g)
        self.cached_compressed_cu_seqlens = torch.arange(B + 1, dtype=torch.int32) * m
        self.cached_compressed_cu_seqlens_adder = torch.arange(B + 1, dtype=torch.int32)
        self.cached_compressed_max_seqlen = m
        self.tail_cis = torch.randn(H, B, KS, generator=g)
        self.tail_cis_len = KST + S0 % KST
        self.comp_cis_len = 7
        self.compressed_cis = torch.randn(H, B, self.comp_cis_len, generator=g)


class FakeCache:
    decode_update_kv = _method("InfLLMv2Cache", "decode_update_kv")
    spec_draft_update_kv = _method("InfLLMv2Cache", "spec_draft_update_kv")
    get_seq_length = _method("InfLLMv2Cache", "get_seq_length")

    def __init__(self, layers):
        self.layers = layers
        self._seen_tokens = 1000


class Weights:
    def __init__(self, g):
        self.embed = torch.randn(V, DH, generator=g)
        self.out = torch.randn(DH, V, generator=g)
        self.Wk = [torch.randn(DH, H * D, generator=g) for _ in range(LYR)]
        self.Wv = [torch.randn(DH, H * D, generator=g) for _ in range(LYR)]
        self.Wq = [torch.randn(DH, H * D, generator=g) for _ in range(LYR)]
        self.Wc = [torch.randn(DH, H, generator=g) for _ in range(LYR)]
        self.Wo = [torch.randn(H * D + H, DH, generator=g) * 0.3 for _ in range(LYR)]


def attend(q, kk, vv, bb, cl):
    """Reads exactly rows [0, cache_seqlens) of the views it is handed; softmax over them plus a row-order-sensitive
    term, so a block in another slot changes the output."""
    S = kk.shape[1]
    rows = torch.arange(S)
    valid = (rows.unsqueeze(0) < cl.to(torch.int64).unsqueeze(1)).unsqueeze(-1)       # (B, S, 1)
    s = (kk * q.unsqueeze(1)).sum(-1) * 0.5 + bb                                       # (B, S, H)
    s = torch.where(valid, s, torch.full_like(s, float("-inf")))
    w = torch.softmax(s, dim=1)
    o = (w.unsqueeze(-1) * vv).sum(1)                                                  # (B, H, D)
    order = (torch.where(valid, s, torch.zeros_like(s)) * (rows + 1).view(1, S, 1).to(s.dtype)).sum(1)   # (B, H)
    return torch.cat([o.reshape(o.shape[0], H * D), 1e-4 * order], dim=-1)


class FakeModel:
    def __init__(self, w, slack, oracle=None, record=True):
        self.w = w
        self.slack = int(slack)
        self.oracle = oracle
        self.record = record
        self.has_buffers = False
        self.pooling_buf_all = None
        self.block_size = BS
        self.num_layers = LYR
        self.layers = [types.SimpleNamespace(pooling_block_size=KS, pooling_stride=KST) for _ in range(LYR)]
        self.records = {}
        self.draft_calls = 0

    def _select(self, l, h, layer, p):
        T = p // BS
        lim = min(T + 1, int(self.pooling_buf_all.shape[-1]))
        assert lim >= TOPK, "the fake world needs >= topk nameable blocks"
        ids = torch.arange(lim, dtype=torch.float32)
        feat = (h.sum(-1) + 0.001 * layer.compress_k_cache_varlen.sum(dim=(1, 2, 3))
                + 0.01 * layer.total_cis[:, :layer.cis_len].sum(dim=(1, 2)))                # (B,)
        hh = torch.arange(H, dtype=torch.float32)
        sc = torch.sin(ids.view(1, 1, lim) * 1.37 + feat.view(1, B, 1) * 2.9 + hh.view(H, 1, 1) * 0.61 + 0.23 * l)
        if T < lim:
            sc[..., T] = 10.0                     # the live tail is always selected (the forced local window)
        return torch.topk(sc, TOPK, dim=-1).indices.contiguous()                           # (H, B, TOPK) int64

    def _layer(self, l, h, cache, p, ctx=None):
        layer = cache.layers[l]
        eng = layer.cache_engine
        pol = AVAIL.POLICY
        if ctx is None and pol is not None:
            pol.begin_layer(l)
        w = self.w
        k = torch.tanh(h @ w.Wk[l]).view(B, H, D)
        v = torch.tanh(h @ w.Wv[l]).view(B, H, D)
        q = torch.tanh(h @ w.Wq[l]).view(B, H, D)
        cis = (torch.sigmoid(h @ w.Wc[l]) - 0.5).view(B, 1, H)
        nck = layer.update_no_compress_k_decode(k.unsqueeze(1), KS, KST)
        new_ck = nck.mean(dim=1, keepdim=True) if nck is not None else None
        layer.update_compress_k_decode(new_ck, None)
        ucis = layer.update_uncompressed_cis(cis, 0, B)
        layer.update_cis(cis.permute(2, 0, 1), 0, B)
        sel = self._select(l, h, layer, p)
        if ctx is None:
            kk, vv, bb, cl = cache.decode_update_kv(k, v, ucis, l, sel)
            if pol is not None:
                bb = pol.mask_bias(bb)
            if self.record:
                with SYNC.pause():
                    n = int(cl[0])
                    assert bool((cl == n).all())
                    self.records[(p, l)] = (eng._block_map.clone(), cl.clone(), kk[:, :n].clone(), vv[:, :n].clone(), bb[:, :n].clone())
        else:
            kk, vv, bb, cl = cache.spec_draft_update_kv(k, v, ucis, l, sel, ctx)
        return h + 0.5 * torch.tanh(attend(q, kk, vv, bb, cl) @ w.Wo[l])

    def _hidden(self, tok, pos):
        return self.w.embed[tok[:, 0]] + 0.01 * pos.to(torch.float32)

    def decode_inference(self, tok, cu, pos, cache, warmup=False):
        pol = AVAIL.POLICY
        if pol is not None:
            pol.begin_step()
        if not self.has_buffers:
            out_len = (cache.get_seq_length(0) + 1 + BS - 1) // BS + self.slack
            self.pooling_buf_all = torch.empty(1, 1, out_len)
            self.has_buffers = True
        p = int(cache.layers[0].cache_engine.seq_length)
        h = self._hidden(tok, pos)
        for l in range(LYR):
            h = self._layer(l, h, cache, p)
        return (h @ self.w.out).unsqueeze(1)

    def draft_inference(self, tok, cu, pos, cache, ctx):
        assert self.has_buffers
        p = int(cache.layers[0].cache_engine.seq_length)
        h = self._hidden(tok, pos)
        for l in range(LYR):
            h = self._layer(l, h, cache, p, ctx=ctx)
        self.draft_calls += 1
        if self.oracle is None:
            return (h @ self.w.out).unsqueeze(1)
        t = self.oracle(p)                                                                 # (B,) int64
        return torch.zeros(B, V).scatter_(1, t.view(B, 1), 10.0).unsqueeze(1)


class FakeEvent:
    def record(self, stream=None):
        return None

    def query(self):
        return True


class FakeStream:
    def wait_event(self, ev):
        return None


def needs_rewarm(model, cache):
    seq = cache.get_seq_length(0)
    return (not model.has_buffers) or (seq + 1 + BS - 1) // BS > model.pooling_buf_all.shape[-1]


def build_world(seed, slack, oracle=None, record=True):
    g = torch.Generator().manual_seed(1000 + seed)
    w = Weights(g)
    cache = FakeCache([FakeLayer(g) for _ in range(LYR)])
    model = FakeModel(w, slack, oracle=oracle, record=record)
    t0 = torch.tensor([[1], [5], [9]], dtype=torch.int64)
    pos0 = torch.full((B, 1), S0, dtype=torch.int32)
    return model, cache, t0, pos0


def run_reference(seed, slack, total):
    """The shipped decode only: tokens (B, total + 1), the logits of the step whose input is token n, the records."""
    model, cache, t0, pos0 = build_world(seed, slack)
    tok, pos = t0, pos0.clone()
    seq, logits = [t0], []
    for it in range(total):
        if needs_rewarm(model, cache):
            model.has_buffers = False
        lg = model.decode_inference(tok, None, pos, cache)
        logits.append(lg[:, -1, :].clone())
        tok = lg[:, -1, :].argmax(-1, keepdim=True)
        seq.append(tok)
        pos = pos + 1
    return dict(seq=torch.cat(seq, 1), logits=logits, records=model.records, cache=cache, model=model)


def corrupt_schedule(kind):
    """(b, i) -> True when the oracle drafts a WRONG token for token index i + 1 of request b."""
    if kind == "none":
        return lambda b, i: False
    if kind == "mixed":
        return lambda b, i: (i * 7 + b * 3) % 5 == 0
    if kind == "first":
        return lambda b, i: b == 0 and i % 3 == 1
    raise ValueError(kind)


def make_oracle(ref_seq, kind):
    """The draft's tokens: the reference token of the next index, corrupted per the schedule, as a precomputed table
    (no host data is built inside the timed code)."""
    bad = corrupt_schedule(kind)
    n = ref_seq.shape[1]
    table = ref_seq.clone()
    for i in range(n - 1):
        for b in range(B):
            if bad(b, i):
                table[b, i + 1] = (table[b, i + 1] + 1) % V

    def oracle(p):
        i = p - S0
        j = min(i + 1, n - 1)
        return table[:, j].contiguous()
    return oracle


def run_spec(seed, slack, total, K, arm, ref, corrupt="mixed", pin=True, hook_ops=None, loop_ops=None, record=True,
             count=None):
    model, cache, t0, pos0 = build_world(seed, slack, oracle=make_oracle(ref["seq"], corrupt), record=record)
    store = sl.Store(LYR, H, B, sl.store_layout(TOPK, R))
    if pin:
        sl.pin_window(store)
    ctx = sl.SpecContext(store, side=FakeStream(), make_event=FakeEvent, current_stream=FakeStream,
                         stream_ctx=lambda s: contextlib.nullcontext(), num_layers=LYR, block_size=BS,
                         kernel_size=KS, kernel_stride=KST)
    hook = sl.StoreServeHook([lay.cache_engine for lay in cache.layers], store, hook_ops or TWIN_OPS)
    marks = sl.RoundMarks(LYR, K) if arm != "early" else None
    got = {}

    def on_logits(row, n, u, kind):
        got.setdefault(n, []).append(row.clone())
    cm = counting_syncs(count) if count is not None else contextlib.nullcontext()
    with cm:
        out = sl.exact_loop(model, cache, ctx, hook, AVAIL, marks, loop_ops or TWIN_OPS, t0, pos0, None, K, total, arm,
                            ref_seq=ref["seq"] if arm == "tf" else None, on_logits=on_logits)
    return dict(out=out, logits=got, store=store, ctx=ctx, cache=cache, model=model, hook=hook)


def state_equal(ca, cb, why):
    """The committed state of two caches: engine maps, counters, the window rows, the tail rows below tail_len,
    the host rows of complete blocks, the layer tables up to their lengths, the token counter."""
    assert ca._seen_tokens == cb._seen_tokens, (why, "seen", ca._seen_tokens, cb._seen_tokens)
    for l, (la, lb) in enumerate(zip(ca.layers, cb.layers)):
        ea, eb = la.cache_engine, lb.cache_engine
        assert (ea.seq_length, ea._tail_block_len_on_gpu) == (eb.seq_length, eb._tail_block_len_on_gpu), (why, l)
        assert torch.equal(ea._block_map, eb._block_map), (why, l, "block map")
        assert torch.equal(ea._cache_lens, eb._cache_lens), (why, l, "cache lens")
        wr = (TOPK - 1) * BS
        for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            assert torch.equal(getattr(ea, name)[:, :wr], getattr(eb, name)[:, :wr]), (why, l, name, "window")
            tl = ea._tail_block_len_on_gpu
            assert torch.equal(getattr(ea, name)[:, wr:wr + tl], getattr(eb, name)[:, wr:wr + tl]), (why, l, name, "tail")
        hc = (ea.seq_length // BS) * BS
        assert torch.equal(ea._k_cpu[:, :hc], eb._k_cpu[:, :hc]) and torch.equal(ea._v_cpu[:, :hc], eb._v_cpu[:, :hc]), (why, l, "host")
        for n in sl.LayerJournal.SCALARS:
            assert getattr(la, n) == getattr(lb, n), (why, l, n)
        assert torch.equal(la.no_compress_k_cache[:, :la.no_compress_k_len], lb.no_compress_k_cache[:, :lb.no_compress_k_len]), (why, l)
        assert torch.equal(la.tail_cis[..., :la.tail_cis_len], lb.tail_cis[..., :lb.tail_cis_len]), (why, l)
        assert torch.equal(la.compress_k_cache_varlen, lb.compress_k_cache_varlen), (why, l)
        assert torch.equal(la.cached_compressed_cu_seqlens, lb.cached_compressed_cu_seqlens), (why, l)
        assert torch.equal(la.total_cis[:, :la.cis_len], lb.total_cis[:, :lb.cis_len]), (why, l)
        assert torch.equal(la.compressed_cis[..., :la.comp_cis_len], lb.compressed_cis[..., :lb.comp_cis_len]), (why, l)


def mismatches(ref, spec):
    """Every compared position's logits against the reference step with the same input index, and the attention
    inputs of every committed step against the reference's at the same position. Returns a list of reasons."""
    bad = []
    for n, rows in spec["logits"].items():
        for row in rows:
            if not torch.equal(row, ref["logits"][n]):
                bad.append(("logits", n))
    n_fin = spec["out"]["n"]
    for (p, l), rec in spec["model"].records.items():
        if p >= S0 + n_fin:
            continue
        r0 = ref["records"][(p, l)]
        for i, name in enumerate(("block_map", "cache_lens", "k rows", "v rows", "bias rows")):
            if not torch.equal(rec[i], r0[i]):
                bad.append((name, p, l))
    return bad


def coverage(spec):
    ex = [r for r in spec["out"]["rounds"] if r["kind"] == "round"]
    st = spec["store"]
    return dict(rounds=len(ex), rollbacks=sum(1 for r in ex if r["rollback"] is not None),
                undo_fill=sum(1 for r in ex if (r["rollback"] or {}).get("undo_fill")),
                rows_restored=sum((r["rollback"] or {}).get("rows_restored", 0) for r in ex),
                fills=sum(1 for r in ex if r["u_fill"] is not None),
                early_stops=sum(1 for r in ex if r["computed"] < r["U"]),
                served=int(st.verify_served_blocks.sum()), host=int(st.verify_host_blocks.sum()),
                rb_loads=int(st.rollback_served_blocks.sum() + st.rollback_host_blocks.sum()),
                plain=sum(1 for r in spec["out"]["rounds"] if r["kind"] == "plain"))


TOTAL = 30


@pytest.fixture(scope="module")
def refs():
    return {(seed, slack): run_reference(seed, slack, TOTAL) for seed in (0, 1) for slack in (0, 1000)}


# ===========================================================================
# (1) the twins and the counter themselves
# ===========================================================================
def test_twin_diff_keeps_hits_in_place_and_fills_free_slots_in_order():
    old = torch.tensor([[[5, 6, 7, 8]]])
    new = torch.tensor([[[9, 6, 4, 8]]])
    nm, lm = torch.full_like(old, -9), torch.full_like(old, -9)
    twin_diff(old, new, nm, lm)
    assert nm.tolist() == [[[9, 6, 4, 8]]] and lm.tolist() == [[[9, -1, 4, -1]]]
    # an empty slot (-1) is free; the one miss goes there
    old = torch.tensor([[[-1, 6, 7, 8]]])
    new = torch.tensor([[[6, 8, 3, 7]]])
    nm, lm = torch.full_like(old, -9), torch.full_like(old, -9)
    twin_diff(old, new, nm, lm)
    assert nm.tolist() == [[[3, 6, 7, 8]]] and lm.tolist() == [[[3, -1, -1, -1]]]
    # more free slots than misses (a duplicate in the selection): the extra free slot is NOT written (stale -9)
    old = torch.tensor([[[5, 6, 7, 8]]])
    new = torch.tensor([[[6, 7, 8, 6]]])
    nm, lm = torch.full_like(old, -9), torch.full_like(old, -9)
    twin_diff(old, new, nm, lm)
    assert nm.tolist() == [[[-9, 6, 7, 8]]] and lm.tolist() == [[[-9, -1, -1, -1]]]


def test_twin_gathers_address_like_the_kernels_and_refuse_an_overlapping_d2d_launch():
    gpu = torch.zeros(1, 6 * 2, 1, 2)
    host = torch.arange(1 * 10 * 2 * 1 * 2, dtype=torch.float32).view(1, 20, 1, 2)
    ids = torch.tensor([[[-1, 3, -1]]])
    twin_gather_k(gpu, host, ids, 2)
    assert torch.equal(gpu[0, 2:4], host[0, 6:8]) and float(gpu[0, :2].abs().sum()) == 0.0
    alloc = torch.randn(1, 6 * 2, 1, 2)
    good = torch.tensor([[[4, -1, 5]]])                  # sources 4, 5 >= M = 3
    twin_gather_k(alloc, alloc, good, 2)
    assert torch.equal(alloc[0, 0:2], alloc[0, 8:10]) and torch.equal(alloc[0, 4:6], alloc[0, 10:12])
    with pytest.raises(AssertionError):
        twin_gather_k(alloc, alloc, torch.tensor([[[1, -1, -1]]]), 2)


def test_sync_counter_sees_every_conversion_kind_positive_controls():
    c = SyncCounter()
    t = torch.arange(5)
    with counting_syncs(c):
        _ = bool(t.any())
        _ = t.sum().item()
        _ = int(t.sum())
        _ = float(t.float().mean())
        _ = t.tolist()
        _ = torch.tensor([1, 2])
        _ = t[t > 2]
        _ = t.nonzero()
        with c.pause():
            _ = bool(t.any())
        _ = t + 1                                      # no sync
        _ = t[1:3]
    assert c.n == 8, c.what
    assert not isinstance(torch.Tensor.__dict__.get("item"), type(lambda: 0)), "the patch was not undone"


def test_sync_counter_sees_the_old_verify_plan_round_syncs_positive_control():
    """The Path-1 round's plan (spec_loop.plan_round) reads the device twice per layer (the INFLIGHT guard and a
    pageable torch.tensor of the live-tail ids): the counter must see them."""
    st = sl.Store(1, H, B, sl.store_layout(TOPK, R))
    st.sync_window(0, torch.arange(H * B * TOPK).view(H, B, TOPK) % 20, 1)
    sel = (torch.arange(2 * H * B * TOPK).view(2, H, B, TOPK) % 20).to(torch.int64)
    c = SyncCounter()
    with counting_syncs(c):
        sl.plan_round(st, 0, sel, 20, 2, 1, 2, 2, torch.zeros(H, B, st.lay.W, dtype=torch.bool))
    assert c.n >= 2 and "__bool__" in c.what and "torch.tensor" in c.what, c.what


# ===========================================================================
# (2) arithmetic: round span, fill position, the tf schedule, the stacked window sync, the pinned window
# ===========================================================================
@pytest.mark.parametrize("seq,out_len,K,rem,want", [
    (100, 26, 3, 50, 4),        # 26*4 - 100 = 4 positions left before the grid boundary
    (101, 26, 3, 50, 3),
    (103, 26, 3, 50, 1),        # one position: a plain step
    (104, 26, 3, 50, 0),        # the next step needs a re-warm
    (90, 26, 3, 2, 2),          # remaining tokens bound it
    (10, 100, 9, 99, 4),        # at most block_size = 4 (one fill per round)
])
def test_round_span(seq, out_len, K, rem, want):
    assert sl.round_span(seq, out_len, K, rem, 4) == want


def test_round_span_agrees_with_the_pilot_rewarm_rule_and_refuses_tensors():
    for seq in range(60, 140):
        for out_len in range(16, 36):
            needs = (seq + 1 + 3) // 4 > out_len           # spec_loop_pilot.needs_rewarm_len, bs = 4
            u = sl.round_span(seq, out_len, 3, 99, 4)
            assert (u == 0) == needs
            for j in range(u):                              # every position of the round is served by the grid
                assert not ((seq + j + 1 + 3) // 4 > out_len)
    with pytest.raises(TypeError):
        sl.round_span(torch.tensor(5), 3, 1, 1, 4)


def test_fill_position_and_tf_schedule():
    # position u writes row tail_len0 + u; the fill is the write of row block_size - 1
    assert sl.fill_position(61, 4, 64) == 2 and sl.fill_position(61, 2, 64) is None
    assert sl.fill_position(60, 4, 64) == 3 and sl.fill_position(63, 2, 64) == 0 and sl.fill_position(0, 64, 64) == 63
    with pytest.raises(ValueError):
        sl.fill_position(64, 2, 64)
    assert [sl.tf_commit(r, 3, None) for r in range(4)] == [1, 2, 3, 1]
    assert sl.tf_commit(5, 4, 2) == 2 and sl.tf_commit(6, 4, 0) == 3 and sl.tf_commit(4, 4, 3) == 3
    with pytest.raises(ValueError):
        sl.tf_commit(0, 1, None)


def test_sync_window_stacked_equals_the_per_layer_sync():
    g = torch.Generator().manual_seed(3)
    lay = sl.store_layout(TOPK, R)
    a, b = sl.Store(3, H, B, lay), sl.Store(3, H, B, lay)
    for st in (a, b):
        st.ids[..., lay.round_base:lay.mirror_lo] = torch.randint(-1, 30, (3, H, B, R), generator=torch.Generator().manual_seed(4))
        st.state[..., lay.round_base:lay.mirror_lo] = sl.RESIDENT
    maps = torch.randint(0, 30, (3, H, B, TOPK), generator=g)
    for l in range(3):
        a.sync_window(l, maps[l], 7)
    b.sync_window_stacked(maps, 7)
    for n in ("ids", "state", "stamp", "used", "wasted_blocks"):
        assert torch.equal(getattr(a, n), getattr(b, n)), n


def test_pinned_window_is_never_a_prefetch_target_and_the_unpinned_one_is():
    """plan_draft with pin_window never issues into slots 0 .. topk-1; the negative control (not pinned) does, once
    the round region is full."""
    lay = sl.store_layout(TOPK, R)
    for pinned in (True, False):
        st = sl.Store(1, H, B, lay)
        if pinned:
            sl.pin_window(st)
        st.sync_window(0, torch.arange(TOPK).view(1, 1, TOPK).expand(H, B, TOPK).contiguous() + 0, 1)
        hit_window = False
        for step in range(6):
            base = 100 + 9 * step
            sel = torch.cat([torch.full((H, B, 1), 50), torch.arange(base, base + TOPK - 1).view(1, 1, -1).expand(H, B, -1)], -1).contiguous()
            plan = sl.plan_draft(st, 0, sel, 50, False, step + 1, step + 2, 10 ** 6)
            st.mark_arrived(step + 1)
            hit_window = hit_window or bool((plan.issue[..., :TOPK] >= 0).any())
        assert hit_window is (not pinned), "pinned=%s" % pinned


def _kernel_read(t):
    """What the gathers load (flash_h2d_mask.py:31, flash_h2d_mask_bias.py:34): numel entries of the storage at the
    tensor's data pointer, read flat as a contiguous (H, B, M) array."""
    return torch.as_strided(t, (t.numel(),), (1,), t.storage_offset())


def test_draft_issue_map_is_contiguous_as_the_gather_kernels_read_it():
    """CPU run 2190738 found it: _assign returned ext[..., :W] (row stride W + 1) and spec_draft_update handed it to
    the gathers, which read it flat. Now the issue map (and plan_round's fetch) is contiguous, so the kernel's flat
    read equals the map; negative control: the old view's flat read is a different map."""
    lay = sl.store_layout(TOPK, R)
    st = sl.pin_window(sl.Store(1, H, B, lay))
    st.sync_window(0, torch.arange(TOPK).view(1, 1, TOPK).expand(H, B, TOPK).contiguous(), 1)
    sel = torch.cat([torch.full((H, B, 1), 50),
                     torch.arange(H * B * (TOPK - 1)).view(H, B, TOPK - 1) % 40 + 100], -1).contiguous()
    plan = sl.plan_draft(st, 0, sel, 50, False, 1, 2, 10 ** 6)
    assert plan.issue.is_contiguous()
    assert bool(((plan.issue >= 0).sum(-1) > 0).all()), "precondition: every (h, b) row issues a prefetch"
    assert torch.equal(_kernel_read(plan.issue), plan.issue.reshape(-1))
    ext = torch.full((H, B, lay.W + 1), -1, dtype=torch.int64)
    ext[..., :lay.W] = plan.issue
    old = ext[..., :lay.W]
    assert not old.is_contiguous() and not torch.equal(_kernel_read(old), old.reshape(-1))
    st2 = sl.Store(1, H, B, lay)
    st2.sync_window(0, torch.arange(H * B * TOPK).view(H, B, TOPK) % 20, 1)
    sel2 = (torch.arange(2 * H * B * TOPK).view(2, H, B, TOPK) % 20).to(torch.int64)
    rp = sl.plan_round(st2, 0, sel2, 20, 2, 1, 2, 2, torch.zeros(H, B, lay.W, dtype=torch.bool))
    assert rp.fetch.is_contiguous()


# ===========================================================================
# (3) one shipped step served from the store is byte-identical to the plain step
# ===========================================================================
def _one_world_step(seed, serve, corrupt_store=False):
    """Warm both engines with one plain step, prefetch the NEXT step's selection into the round region (the twin
    gather from the host), then run the next shipped step with or without the store hook. Returns the cache, what
    attention got (views, cache_lens) and the hook."""
    model, cache, t0, pos0 = build_world(seed, slack=0)
    model.decode_inference(t0, None, pos0, cache)
    st = sl.pin_window(sl.Store(LYR, H, B, sl.store_layout(TOPK, R)))
    st.sync_window_stacked(torch.stack([lay.cache_engine._block_map for lay in cache.layers]), 1)
    lay0 = st.lay
    # prefetch: blocks 0 .. R-1 of every stream into the round region (some of them the next step will load)
    for l, lay in enumerate(cache.layers):
        eng = lay.cache_engine
        issue = torch.full((H, B, lay0.W), -1, dtype=torch.int64)
        issue[..., lay0.round_base:lay0.mirror_lo] = torch.arange(R).view(1, 1, R)
        twin_gather_k(eng._k_gpu, eng._k_cpu, issue.contiguous(), BS)
        twin_gather_v(eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, lay.total_cis, issue.contiguous(), BS)
        st.ids[l] = torch.where(issue >= 0, issue, st.ids[l])
        st.state[l] = torch.where(issue >= 0, torch.full_like(st.state[l], sl.RESIDENT), st.state[l])
        if corrupt_store:
            eng._k_gpu[:, lay0.round_base * BS:lay0.mirror_lo * BS] += 1.0
    hook = sl.StoreServeHook([lay.cache_engine for lay in cache.layers], st, TWIN_OPS)
    tok = torch.tensor([[2], [3], [4]])
    model.records = {}
    with (sl.hook_installed(AVAIL, hook) if serve else contextlib.nullcontext()):
        lg = model.decode_inference(tok, None, pos0 + 1, cache)
    return cache, model, lg, hook, st


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_served_step_is_byte_identical_to_the_shipped_step(seed):
    ca, ma, la, _, _ = _one_world_step(seed, serve=False)
    cb, mb, lb, hook, st = _one_world_step(seed, serve=True)
    served = int(st.verify_served_blocks.sum())
    assert hook.calls == LYR and served > 0, "precondition: the store served at least one load (served=%d)" % served
    assert torch.equal(la, lb)
    for key, r in ma.records.items():
        for x, y in zip(r, mb.records[key]):
            assert torch.equal(x, y), key
    for l in range(LYR):
        ea, eb = ca.layers[l].cache_engine, cb.layers[l].cache_engine
        assert torch.equal(ea._block_map, eb._block_map)
        for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            assert torch.equal(getattr(ea, n)[:, :TOPK * BS], getattr(eb, n)[:, :TOPK * BS]), n
    assert AVAIL.POLICY is None


def test_a_corrupted_store_slot_is_seen_negative_control():
    ca, ma, la, _, _ = _one_world_step(0, serve=False)
    cb, mb, lb, hook, st = _one_world_step(0, serve=True, corrupt_store=True)
    assert int(st.verify_served_blocks.sum()) > 0
    assert not torch.equal(la, lb) or any(not torch.equal(ma.records[k][2], mb.records[k][2]) for k in ma.records)


def test_hook_installed_refuses_an_installed_policy_and_always_restores_none():
    hook = sl.StoreServeHook([], None, TWIN_OPS)
    with pytest.raises(RuntimeError):
        with sl.hook_installed(AVAIL, hook):
            raise RuntimeError("boom")
    assert AVAIL.POLICY is None
    AVAIL.POLICY = object()
    try:
        with pytest.raises(RuntimeError):
            with sl.hook_installed(AVAIL, hook):
                pass
    finally:
        AVAIL.POLICY = None
    with pytest.raises(RuntimeError):
        hook.on_diff(None, None, None, None, 7)            # before begin_layer


# ===========================================================================
# (4) END TO END: every verify position equals the shipped step, every arm, both grid regimes
# ===========================================================================
@pytest.mark.parametrize("slack", [0, 1000])
@pytest.mark.parametrize("arm,K", [("tf", 3), ("tf", 1), ("round", 3), ("round", 2), ("early", 3)])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_loop_equals_the_shipped_decode(refs, seed, slack, arm, K):
    ref = refs[(seed, slack)]
    spec = run_spec(seed, slack, TOTAL, K, arm, ref)
    bad = mismatches(ref, spec)
    assert not bad, bad[:5]
    seq = torch.cat(spec["out"]["committed"], 1)
    assert spec["out"]["n"] == TOTAL and torch.equal(seq, ref["seq"][:, :seq.shape[1]])
    # the final state of the two engines (the reference ran exactly TOTAL steps too)
    state_equal(ref["cache"], spec["cache"], (seed, slack, arm, K))
    cov = coverage(spec)
    assert cov["rounds"] > 0 and cov["served"] > 0, cov
    if arm in ("tf", "round"):
        assert cov["rollbacks"] > 0, cov
    if arm == "early":
        assert cov["early_stops"] > 0, cov
    assert AVAIL.POLICY is None


@pytest.mark.parametrize("seed", [0, 1])
def test_tf_rolls_back_a_rejected_fill_with_tail_rows_restored_when_rounds_cross_the_fill(refs, seed):
    """slack 1000: the pooling grid never forces a re-warm, so a round can run past the fill; the tf schedule rejects
    the first fill at u >= 1, and the positions after it had overwritten tail rows that the mirror restores."""
    ref = refs[(seed, 1000)]
    spec = run_spec(seed, 1000, TOTAL, 3, "tf", ref)
    cov = coverage(spec)
    assert cov["undo_fill"] >= 1 and cov["rows_restored"] >= 1, cov
    assert not mismatches(ref, spec)


@pytest.mark.parametrize("seed", [0, 1])
def test_with_the_shipped_regrid_a_fill_is_the_last_position_of_its_round(refs, seed):
    """slack 0 (the GPU's re-warm at every block boundary): round_span stops every round at the grid boundary, which
    is right after the fill, so a fill inside a round is always its LAST position (no tail rows to restore; the
    rename is undone)."""
    ref = refs[(seed, 0)]
    spec = run_spec(seed, 0, TOTAL, 3, "tf", ref)
    ex = [r for r in spec["out"]["rounds"] if r["kind"] == "round"]
    assert any(r["u_fill"] is not None for r in ex)
    assert all(r["u_fill"] == r["U"] - 1 for r in ex if r["u_fill"] is not None)
    assert coverage(spec)["rows_restored"] == 0


# ===========================================================================
# (5) negative controls: the end-to-end witness sees each defect
# ===========================================================================
def _noop_k(gpu_data, cpu_data, load_ids, block_size):
    return None


def _noop_v(gpu_data, cpu_data, kv_bias_buf, kv_bias, load_ids, block_size):
    return None


def test_negative_rollback_without_the_window_restore_is_caught(refs):
    hit = False
    for seed in (0, 1):
        ref = refs[(seed, 0)]
        spec = run_spec(seed, 0, TOTAL, 3, "round", ref, loop_ops=sl.GatherOps(_noop_k, _noop_v))
        cov = coverage(spec)
        if cov["rb_loads"] > 0:
            assert mismatches(ref, spec), "a rollback that skipped %d window reloads went unseen" % cov["rb_loads"]
            hit = True
    assert hit, "precondition: no rollback changed a window slot"


def test_negative_rollback_without_the_mirror_is_caught(refs, monkeypatch):
    monkeypatch.setattr(sl, "mirror_tail", lambda engine, mirror_lo: None)
    hit = False
    for seed in (0, 1):
        ref = refs[(seed, 1000)]
        spec = run_spec(seed, 1000, TOTAL, 3, "tf", ref)
        if coverage(spec)["rows_restored"] > 0:
            assert mismatches(ref, spec)
            hit = True
    assert hit, "precondition: no rejected fill had later positions"


def test_negative_unpinned_window_written_by_the_draft_is_caught(refs, monkeypatch):
    orig = sl.plan_draft
    seen = {"window_issue": 0}

    def spy(store, l, sel, *a, **k):
        plan = orig(store, l, sel, *a, **k)
        with SYNC.pause():
            seen["window_issue"] += int((plan.issue[..., :TOPK] >= 0).sum())
        return plan
    monkeypatch.setattr(sl, "plan_draft", spy)
    caught = False
    for seed in (0, 1):
        for arm in ("round", "tf"):
            ref = refs[(seed, 0)]
            seen["window_issue"] = 0
            spec = run_spec(seed, 0, TOTAL, 3, arm, ref, pin=False)
            if seen["window_issue"] > 0 and mismatches(ref, spec):
                caught = True
    assert caught, "an unpinned draft never wrote a window slot that a verify step then read"


def test_negative_wrong_bytes_served_by_the_hook_are_caught(refs):
    def corrupt_k(gpu_data, cpu_data, load_ids, block_size):
        twin_gather_k(gpu_data, cpu_data, load_ids, block_size)
        with SYNC.pause():
            for b, h, m, i in list(_programs(gpu_data, cpu_data, load_ids, block_size)):
                gpu_data[b, m * block_size:(m + 1) * block_size, h] += 0.5
    ref = refs[(0, 0)]
    spec = run_spec(0, 0, TOTAL, 3, "round", ref, hook_ops=sl.GatherOps(corrupt_k, twin_gather_v))
    assert coverage(spec)["served"] > 0
    assert mismatches(ref, spec)


# ===========================================================================
# (6) host syncs: the stub counts them through a whole loop
# ===========================================================================
@pytest.mark.parametrize("arm", ["tf", "round", "early"])
def test_host_syncs_are_the_decisions_only(refs, arm):
    """No host sync in the draft, the hook, the marks, the rollback or the store sync: the counter sees exactly the
    decision syncs (round: 1 per round; tf with its forced commit: 0; early: min(computed, U - 1))."""
    ref = refs[(0, 0)]
    c = SYNC                         # the twins pause THIS counter (they stand for GPU kernels)
    c.n, c.what, c.paused = 0, [], 0
    spec = run_spec(0, 0, TOTAL, 3, arm, ref, record=False, count=c)
    ex = [r for r in spec["out"]["rounds"] if r["kind"] == "round"]
    if arm == "round":
        want = [1] * len(ex)
    elif arm == "tf":
        want = [0] * len(ex)
    else:
        want = [min(r["computed"], r["U"] - 1) for r in ex]
    assert [r["host_syncs"] for r in ex] == want
    assert c.n == sum(want), (c.n, sum(want), c.what[:20])
    assert not mismatches(ref, dict(spec, model=types.SimpleNamespace(records={})))


# ===========================================================================
# (7) misuse is refused
# ===========================================================================
def test_exact_round_refuses_misuse(refs):
    model, cache, t0, pos0 = build_world(0, 0)
    st = sl.Store(LYR, H, B, sl.store_layout(TOPK, R))
    ctx = sl.SpecContext(st, side=FakeStream(), make_event=FakeEvent, current_stream=FakeStream,
                         stream_ctx=lambda s: contextlib.nullcontext(), num_layers=LYR, block_size=BS,
                         kernel_size=KS, kernel_stride=KST)
    hook = sl.StoreServeHook([lay.cache_engine for lay in cache.layers], st, TWIN_OPS)
    marks = sl.RoundMarks(LYR, 2)
    model.has_buffers = True
    model.pooling_buf_all = torch.empty(1, 1, 100)
    for kw in (dict(U=1, mode="round"), dict(U=3, mode="other"), dict(U=4, mode="round"),
               dict(U=3, mode="early", forced_c=2), dict(U=5, mode="early")):
        with pytest.raises(ValueError):
            sl.exact_round(model, cache, ctx, hook, AVAIL, marks, TWIN_OPS, t0, pos0, None, **kw)
    model.has_buffers = False
    with pytest.raises(RuntimeError):
        sl.exact_round(model, cache, ctx, hook, AVAIL, marks, TWIN_OPS, t0, pos0, None, 2, "round")
    with pytest.raises(ValueError):
        sl.rollback_exact(cache, TWIN_OPS, st, marks, 2, 2, None, st.lay.mirror_lo, 1)
    with pytest.raises(RuntimeError):
        sl.rollback_exact(cache, TWIN_OPS, st, sl.RoundMarks(LYR, 2), 1, 2, None, st.lay.mirror_lo, 1)


# ===========================================================================
# (8) source-level pins: the conditions E1-E3 live in the shipped text the verify calls
# ===========================================================================
def _func(tree, cls, name):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_decode_attention_call_is_the_64_slot_views_with_the_engine_cache_lens_and_the_split_knob():
    src = NOSA_LLAMA.read_text()
    tree = ast.parse(src)
    fn = _func(tree, "LlamaLayer", "decode_forward")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "flash_attn_nosa_with_kvcache"]
    assert len(calls) == 1
    call = calls[0]
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw == {"cache_seqlens": "_cache_lens", "num_splits": "_ATTN_SPLITS"}, kw
    assert [ast.unparse(a) for a in call.args] == ["query_states.unsqueeze(1)", "key_states", "value_states", "kv_bias"]
    text = ast.get_source_segment(src, fn)
    assert "key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(" in text
    assert re.search(r'_ATTN_SPLITS = int\(os\.environ\.get\("NOSI_ATTN_SPLITS", "0"\) or 0\)', src)
    ce = CE_SRC
    assert "self.decode_update = self.decode_update_has_kv_bias_verify_slots" in ce
    v = ast.get_source_segment(ce, _func(CE_TREE, "CacheEngine", "decode_update_has_kv_bias_verify_slots"))
    assert "return self._k_gpu_att, self._v_gpu_att, self._kv_bias_gpu_att, self._cache_lens" in v
    assert "self._att_rows = self.topk * self.block_size" in ce


def test_shipped_body_has_the_hook_between_diff_and_the_gathers_and_the_f4_reset():
    body = ast.get_source_segment(CE_SRC, _func(CE_TREE, "CacheEngine", "decode_update_has_kv_bias"))
    i_diff = body.index("diff.diff_offload(self._block_map, topk_idx")
    i_hook = body.index("_ap.on_diff(self._block_map, self._new_block_map_buf, self._load_mask,")
    i_copy = body.index("self._block_map.copy_(self._new_block_map_buf")
    i_g1 = body.index("flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask")
    i_fill = body.index("if tail_full:")
    i_f4 = body.index("self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu", i_fill)
    assert i_diff < i_hook < i_copy < i_g1 < i_fill < i_f4
    assert body.index("self._tail_block_len_on_gpu = 0", i_fill) < i_f4
    dec = ast.get_source_segment(NOSA_LLAMA.read_text(), _func(ast.parse(NOSA_LLAMA.read_text()), "LlamaLayer", "decode_forward"))
    assert dec.index("_ap.begin_layer(self.layer_idx)") < dec.index("cache_engine.decode_update_kv(")
    assert "if _ap is not None: kv_bias = _ap.mask_bias(kv_bias)" in dec
    inf = ast.get_source_segment(NOSA_LLAMA.read_text(), _func(ast.parse(NOSA_LLAMA.read_text()), "Llama", "decode_inference"))
    assert "if _ap is not None: _ap.begin_step()" in inf
    assert "logits = F.linear(hidden_states, self.lm_head).float()" in inf


def test_exact_section_has_no_host_sync_constructs_outside_the_decisions():
    src = SPEC_LOOP.read_text()
    tree = ast.parse(src)
    names = ("store_sources", "serve_from_store", "StoreServeHook", "hook_installed", "mirror_tail", "RoundMarks",
             "rollback_exact", "exact_round", "plain_step", "exact_loop", "pin_window", "round_span", "fill_position")
    seen = set()
    for node in tree.body:
        if getattr(node, "name", None) in names:
            seen.add(node.name)
            seg = ast.get_source_segment(src, node)
            for sub in ast.walk(node):                     # drop the docstrings, then the comments
                if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Constant) and isinstance(sub.value.value, str):
                    seg = seg.replace(ast.get_source_segment(src, sub), "")
            code = "\n".join(line.split("#")[0] for line in seg.splitlines())
            for bad in (".item(", ".tolist(", ".cpu(", "synchronize(", "torch.tensor(", "nonzero(", ".numpy(", "bool("):
                assert bad not in code, (node.name, bad)
    assert seen == set(names), set(names) - seen
    stop = ast.get_source_segment(src, next(n for n in tree.body if getattr(n, "name", None) == "stop_after"))
    assert "bool((argmax_u.reshape(-1) != next_input.reshape(-1)).any())" in stop
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in ("torch", "typing", "collections", "__future__"), node.module
        elif isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in ("torch", "typing", "collections") for a in node.names)


def test_gate_driver_shape():
    src = GATE.read_text()
    tree = ast.parse(src)
    assert src.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')
    assert 'print("SPEC_EXACT_GATE_DONE failed=%d" % nfail, flush=True)' in src
    run = ast.get_source_segment(src, next(n for n in tree.body if getattr(n, "name", None) == "run_exact_arm"))
    assert "_sl.exact_loop(" in run and "_sl.pin_window(" in run
    assert [ast.unparse(d) for d in next(n for n in tree.body if getattr(n, "name", None) == "run_exact_arm").decorator_list] == ["torch.inference_mode()"]
    assert [ast.unparse(d) for d in next(n for n in tree.body if getattr(n, "name", None) == "run_shipped").decorator_list] == ["torch.inference_mode()"]


def test_gate_driver_pure_helpers():
    src = GATE.read_text()
    tree = ast.parse(src)
    ns = dict(torch=torch)
    for name in ("identity_gate", "summarize_rounds"):
        node = next(n for n in tree.body if getattr(n, "name", None) == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(GATE), "exec"), ns)
    g = ns["identity_gate"](torch.tensor([[1, 2, 3], [1, 2, 4]]), torch.tensor([[1, 2, 3, 9], [1, 2, 3, 9]]))
    assert not g["passed"] and g["compared_tokens"] == 3 and g["mismatches"] == [dict(request=1, position=2, spec_token=4, ref_token=3)]
    recs = [dict(kind="plain"), dict(kind="round", U=3, computed=3, c=2, draft_ms=[1.0, 1.0], verify_ms=[5.0, 5.0, 5.0], rollback_ms=0.5, host_syncs=1),
            dict(kind="round", U=3, computed=3, c=3, draft_ms=[2.0, 2.0], verify_ms=[6.0, 6.0, 6.0], rollback_ms=None, host_syncs=1)]
    s = ns["summarize_rounds"](recs)
    assert (s["exact_rounds"], s["plain_steps"], s["rollbacks"], s["host_syncs_per_round_max"]) == (2, 1, 1, 1)
    assert s["round_ms"] == 22.0 and s["verify_ms_per_position"] == 6.0 and abs(s["ms_per_committed_token"] - 22.0 / 3) < 1e-12


def test_step_script_pins_and_guards():
    text = STEP.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-verify" in text
    assert "--export=NONE" in text and "NOSI_ATTN_SPLITS=$SPLITS" in text and "SPLITS=4" in text
    assert "spec_exact_gate.py" in text and 'mkdir "$OUTROOT/r2v_active"' in text
    assert "<<" not in text and "ulimit" not in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if ln.lstrip().startswith("#"):
                continue
            assert word not in ln, (word, ln)
