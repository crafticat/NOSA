"""Task 1.3 of KS/analysis/plan/PLAN_speckv_on_cached_nosa.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): PER-REQUEST
LENGTHS inside the cached-NOSA engine (the victim pool on, cell mb1). Opt-in: benchmarks/Efficiency/r2_base_cell.py
--ragged. Without that flag no code imports this module, and cache_engine.py / nosa_llama.py are not edited (Gate 0).

WHY. The shipped decode keeps ONE length for the whole batch, so every request must advance by the same amount. A
speculative decode (plan 3.4) needs requests that advance by different amounts. This module replaces each batch-wide
scalar with per-request (B,) state. It reuses ported code only (NO NEW KERNEL):
  r2_positional_twin.PositionalTwin (step_unchecked: no host read), r2_twin_fused (optional fused step),
  r2_decode_path (bridge_layer, ragged_widths, R2State buffers), stage1_seqused (the shipped stage-1 op + seqused_k),
  max_pooling_per_request (the shipped pooling kernel with per-request lengths), r2_decode_body.DeviceWritebackBody and
  r2_body_fused.FusedBody (per-request tail, map, lengths, write-back), and the shipped kernels diff_offload,
  pool_update, flash_pool_swap, flash_h2d_from_mask(_bias), flash_attn_nosa_with_kvcache.

THE BATCH-WIDE FIELDS AND WHAT REPLACES THEM (cache_engine.py and nosa_llama.py at ded9d41):
  CacheEngine.seq_length (:189, :345; pool path :450)       -> twin.pos (B,) int64 (pos_before = the write position)
  CacheEngine._tail_block_len_on_gpu (:309, :357; :462, :530) -> pos % 64, per request (DecodeBody.write_tail)
  tail_full (:359; :464) and the write-back (:401-405; :523-527) -> full = active & (p + 1) % 64 == 0, per request,
                                                                 masked device write-back (r2_writeback / FusedBody)
  _block_map[..., 63] += 1 (:407; :529)                      -> (p + 1) // 64 for active requests (DecodeBody)
  _cache_lens[...] (:346, :409; :451, :531)                  -> per request; F4 kept (f4_fix False, the parity target)
  InfLLMv2CacheLayer.no_compress_k_len (:678, :687-691)       -> positions [16 n_comp, pos) in the 64-row key ring
  compress_k_cache_varlen, torch.cat (:667)                   -> comp_k (B, Lk_max, H, D), row n_comp[b]
  cached_compressed_cu_seqlens += adder (:668)                -> cu = arange(B + 1) * Lk_max (constant), seqused_k
  cached_compressed_max_seqlen += 1 (:669)                    -> n_comp (B,) -> n_used (B,) int32 on the device
  tail_cis_len (:731, :753, :762)                             -> positions [16 n_ccis, pos) of total_cis
  compressed_cis / comp_cis_len (:727, :759; findings F2)     -> ccis (H, B, Lk_max), row n_ccis[b] (it grows)
  cis_len (:770, :781) / total_cis (B, S + 1024, H) (:768)    -> pos / twin.total_cis (B, N_host, H)
  the pooling width out_len (nosa_llama.py:817-818, frozen
  in the captured graph, re-captured by the harness)          -> M_b = pos_before // 64 + 1 (B,) int32 on the device
  the stage-1 score width round128(n) (re-warm "width")       -> round128(Lk_max), fixed for the run
KEPT BATCH-WIDE, ON PURPOSE: _pool_stamp_base (a step clock, not a length: pool_update_kernel.cu compares the ages of
ONE (head, request) row only, so a request that skips steps keeps the same LRU order); _tail_block_idx_on_gpu (the
constant 63); block_size, topk.

ONE STEP, per layer, in the shipped order (nosa_llama.py:539-662):
  1. rmsnorm, wqkv, nosa_linear, rope: the shipped ops. The GEMMs go through ops.linear (natural = F.linear, the
     shipped call; exact = a batch-invariant linear, r2_spec_core.PaddedLinear on the GPU).
  2. twin.step_unchecked(k, cis, active): the four table updates of :575-585, per request, without a host read.
     mean_fn None = torch's .mean (the shipped summaries, bit for bit); "bi" / "fused" = the fixed-order mean.
  3. n_used <- n_comp, M_b <- pos_before // 64 + 1, written IN PLACE into static buffers.
  4. stage1_seqused on the padded table (seqused_k = n_used): the shipped C++ op.
  5. pooling + top-k (nosa_llama.py:462-490 with nosa_pooling_per_request): a CUDA graph per layer, captured ONCE at
     the first step (the widths are preallocated from the horizon, so nothing re-captures), or eager (ops.graph False).
  6. the body (decode_fixed). Its diff_offload argument is a composed call: diff_offload, then load = -1 for inactive
     requests, then pool_update + flash_pool_swap + stamp += topk (cache_engine.py:480-511, the shipped order: the
     pool reads the OLD map before the map commit). Both bodies call diff_offload first, so the pool sits exactly where
     the shipped engine puts it, with no change to either body.
  7. attention on the attended VIEWS _k_gpu_att / _v_gpu_att / _kv_bias_gpu_att (cache_engine.py:298-306, :538): the
     split-KV partition depends on the allocated length (job 2173299), so attention never sees the pool rows.
  8. wo, FFN through ops.linear. Final rmsnorm, lm_head through ops.linear, then .float().

EXACTNESS (plan, HUMAN DECISION part 3). With gemm natural and twin torch, at equal lengths, every op gets the
shipped inputs, so the step is bitwise the shipped step (benchmarks/Efficiency/tests_cpu/test_ragged_engine.py
test_equal_lengths_bitwise_shipped). The ragged widths differ from the shipped ones (M_alloc, round128(Lk_max)); the
per-request pooling writes -inf past M_b, and torch.topk(sorted=False) on CUDA returns the indices in index order, so
the widths do not change the selection order (an assumption of the CPU fakes; the GPU cell rg1 measures it). With gemm
exact and twin bi / fused, a request's arithmetic does not depend on its neighbours (EXACT mode, test_ragged_equals_solo).

NEGATIVE CONTROLS (tests only; ops.neg): "shared_lengths" (one batch-wide n_used / M_b: the batch maximum),
"frozen_widths" (M_b of the first step for every step), "rebound_lengths" (new length tensors after the capture: the
graph reads stale ones), "no_pool" (diff only), "unmasked_pool" (the pool also runs for inactive requests),
"padded_attention" (attention gets the whole allocation, pool rows included).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import r2_decode_body as RB
from . import r2_decode_path as RDP
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK, PositionalTwin
from .stage1_seqused import stage1_seqused

GEMM_MODES = ("natural", "exact")
TWIN_MODES = ("torch", "bi", "fused")
BODY_MODES = ("torch", "fused")
NEG_KINDS = (None, "shared_lengths", "frozen_widths", "rebound_lengths", "no_pool", "unmasked_pool",
             "padded_attention")
KERNEL_NAMES = ("diff", "gather_k", "gather_v_bias", "writeback")
SHIPPED_TABLES = ("compress_k_cache_varlen", "no_compress_k_cache", "total_cis", "compressed_cis", "tail_cis")


class RaggedError(RuntimeError):
    pass


def default_kernels():
    """The GPU kernels: the shipped engine's (cache_engine.py:19-23, :78-83: diff_offload, pool_update,
    flash_pool_swap, flash_h2d_from_mask, flash_h2d_from_mask_bias) and the device write-back
    (r2_writeback.masked_writeback). stage1_ext None = the installed infllm_v2.C."""
    from . import cache_engine as CE
    from . import r2_writeback as WB
    return dict(diff=CE.diff.diff_offload, gather_k=CE.flash_h2d_from_mask, gather_v_bias=CE.flash_h2d_from_mask_bias,
                pool_update=None if CE._pool is None else CE._pool.pool_update, pool_swap=CE._flash_pool_swap,
                writeback=WB.masked_writeback, stage1_ext=None)


def exact_linear(m_fix: int):
    """The GPU's batch-invariant GEMM: r2_spec_core.PaddedLinear (cuBLAS at one row count m_fix >= every batch)."""
    from .r2_spec_core import PaddedLinear
    return PaddedLinear(m_fix)


class RaggedOps:
    """The run parameters of one ragged engine (all recorded in RaggedState.meta).
      kernels  dict: diff, gather_k, gather_v_bias, writeback (required); pool_update, pool_swap (required when the
               engine has a pool); stage1_ext (None = the installed op)
      gemm     "natural" (F.linear, the shipped call) or "exact" (linear = a batch-invariant callable)
      twin     "torch" (torch's .mean: the shipped bits), "bi" (fixed_order_mean), "fused" (r2_twin_fused, one kernel)
      body     "torch" (r2_decode_body.DeviceWritebackBody) or "fused" (r2_body_fused.FusedBody)
      graph    True: pooling + top-k in one CUDA graph per layer, captured once; False: eager
      f4_fix   False (the shipped attention length, findings F4) or True
      neg      a negative control (NEG_KINDS), tests only"""

    def __init__(self, kernels, gemm="natural", linear=None, twin="torch", body="torch", graph=True, f4_fix=False,
                 neg=None):
        if gemm not in GEMM_MODES or twin not in TWIN_MODES or body not in BODY_MODES:
            raise RaggedError("gemm %r / twin %r / body %r: one of %s / %s / %s"
                              % (gemm, twin, body, GEMM_MODES, TWIN_MODES, BODY_MODES))
        if (gemm == "exact") != (linear is not None):
            raise RaggedError("gemm 'exact' takes a batch-invariant linear, and only it does")
        if neg not in NEG_KINDS:
            raise RaggedError("unknown negative control %r" % (neg,))
        missing = [k for k in KERNEL_NAMES if not callable(kernels.get(k))]
        if missing:
            raise RaggedError("kernels %s are missing" % missing)
        self.kernels = dict(kernels)
        self.gemm, self.twin, self.body = gemm, twin, body
        self.linear = F.linear if gemm == "natural" else linear
        self.graph = bool(graph)
        self.f4_fix = bool(f4_fix)
        self.neg = neg

    def make_body(self):
        wb = self.kernels["writeback"]
        if self.body == "torch":
            return RB.DeviceWritebackBody(self.f4_fix, wb)
        from . import r2_body_fused as FB
        return FB.fused_body_class(RB.DeviceWritebackBody, RB.DecodeBody)(self.f4_fix, wb)

    def setup_twin(self, twin: PositionalTwin) -> None:
        """The twin's summary mean (and, for "fused", the fused step bound on the instance)."""
        if self.twin in ("bi", "fused"):
            from .r2_bi_decode import fixed_order_mean
            twin.mean_fn = fixed_order_mean
        if self.twin == "fused":
            from . import r2_twin_fused as FU
            FU.attach(twin, methods=("step_unchecked",))

    def describe(self) -> dict:
        return dict(gemm=self.gemm, linear=getattr(self.linear, "__name__", type(self.linear).__name__),
                    m_fix=getattr(self.linear, "m_fix", None), twin=self.twin, body=self.body, graph=self.graph,
                    f4_fix=self.f4_fix, neg=self.neg)


class RaggedState:
    """The per-request decode state of one batch: the R2State (twins, pooling / top-k buffers, activity, host
    positions), the body, the static graph inputs (pos_before, n_used, M_b, score_buf), the per-layer graphs."""

    def __init__(self, model, st, cache, ops: RaggedOps):
        self.st, self.cache, self.ops = st, cache, ops
        self.B, self.L = int(st.B), len(model.layers)
        dev = st.cu_q.device
        eng0 = cache.layers[0].cache_engine
        self.pool = int(getattr(eng0, "pool_blocks", 0) or 0) > 0
        if self.pool and not (callable(ops.kernels.get("pool_update")) and callable(ops.kernels.get("pool_swap"))):
            raise RaggedError("the engine has a victim pool (pool_blocks %d) but no pool kernels were given"
                              % eng0.pool_blocks)
        if self.pool and getattr(eng0, "_k_gpu_att", None) is None:
            raise RaggedError("the pooled engine has no attended view (_k_gpu_att)")
        self.body = ops.make_body()
        self.nl = RDP._nl()
        self.pos_before = torch.zeros(self.B, dtype=torch.int64, device=dev)
        self.n_used = torch.zeros(self.B, dtype=torch.int32, device=dev)
        self.M_b = torch.zeros(self.B, dtype=torch.int32, device=dev)
        self.M0 = None
        self.err = torch.zeros((), dtype=torch.int32, device=dev)
        self.score_buf = None
        self.graphs = [None] * self.L
        self.captures = 0
        self.capture_steps = []
        self.replays = 0
        self.all_active = True
        self.meta = dict(ops=ops.describe(), B=self.B, pool_blocks=int(getattr(eng0, "pool_blocks", 0) or 0),
                         lk_max=st.lk_max, n_max=st.n_max, M_alloc=st.M_alloc, score_width=None)


# ------------------------------------------------------------------ builders
def _state(model, B, H, dev, S, widths, twins, ops, pos_host):
    """An R2State (r2_decode_path.py:192-219, the shared buffers) around the given twins."""
    st = RDP.R2State.__new__(RDP.R2State)
    st.B, st.H, st.S = int(B), int(H), int(S)
    st.lk_max, st.n_max, st.M_alloc = int(widths["lk_max"]), int(twins[0].n_max), int(widths["M_alloc"])
    st.twins = twins
    st._common(model, B, H, dev, "r2", None, ops.f4_fix, None, pos_host)
    return st


def attach(model, cache, S: int, horizon: int, ops: RaggedOps, free_shipped: bool = True) -> RaggedState:
    """The ragged state of a batch after the SHIPPED prefill of S tokens per request (pool on or off): each layer's
    positional tables are bridged into a PositionalTwin (r2_decode_path.bridge_layer), then (free_shipped) the shipped
    batch-wide tables of that layer are released, so the bridge adds no memory. The engine's slot cache, host rows and
    pool state are used as they are. Widths from the horizon (r2_decode_path.ragged_widths); the twin's N_max is the
    host row count, so the bias gather (masked by the host rows, flash_h2d_mask_bias.py:43) never reads past it."""
    lay0, l0 = model.layers[0], cache.layers[0]
    eng0 = l0.cache_engine
    B = int(eng0._k_gpu.shape[0])
    H, D = lay0.num_key_value_heads, lay0.head_dim
    dev, dt = eng0._k_gpu.device, l0.total_cis.dtype
    w = RDP.ragged_widths([S], horizon)
    n_host = int(eng0._k_cpu.shape[1])
    if n_host < int(S) + int(horizon):
        raise RaggedError("host rows %d < S + horizon = %d (max_gen_len too small)" % (n_host, int(S) + int(horizon)))
    twins = []
    for lay in cache.layers:
        t = PositionalTwin(B, H, D, w["lk_max"], n_host, dtype=dt, device=dev)
        RDP.bridge_layer(t, lay, int(S))
        ops.setup_twin(t)
        twins.append(t)
        if free_shipped:
            for name in SHIPPED_TABLES:
                setattr(lay, name, None)
    st = _state(model, B, H, dev, S, w, twins, ops, torch.full((B,), int(S), dtype=torch.int64))
    rs = RaggedState(model, st, cache, ops)
    rs.meta.update(builder="attach", S=int(S), horizon=int(horizon), free_shipped=bool(free_shipped))
    cache.ragged_state = rs
    return rs


def allocate(eng, batch: int, n_host: int, pin: bool) -> None:
    """The tables of one CacheEngine for `batch` requests of n_host host rows: r2_decode_body.DecodeBody.allocate,
    plus, when eng.pool_blocks > 0, the victim pool of cache_engine.prefill_update (:202-306): topk + P slots, the
    pool state (empty slot q has age q - P), the pool rows zeroed, the attended views of topk * block_size rows, and
    the int32 envelope of the two gathers (:215-221)."""
    P = int(getattr(eng, "pool_blocks", 0) or 0)
    if P == 0:
        RB.DecodeBody(False).allocate(eng, batch, n_host, pin)
        return
    if batch < 1 or n_host < eng.block_size:
        raise ValueError("batch must be >= 1 and n_host >= block_size")
    dev = torch.device(eng.device)
    if dev.type == "cuda" and not pin:
        raise ValueError("a CUDA engine needs pinned host rows: the Triton gathers read them directly")
    H, D, bs, topk = eng.head_num, eng.head_dim, eng.block_size, eng.topk
    hd = H * D
    if batch * n_host * hd >= 2 ** 31 or batch * (topk + P) * bs * hd >= 2 ** 31:
        raise ValueError("int32 gather envelope: B %d, host rows %d, slots %d" % (batch, n_host, topk + P))
    dt, rows = eng.dtype, (topk + P) * bs
    eng._k_cpu = torch.empty((batch, n_host, H, D), dtype=dt, device="cpu", pin_memory=pin)
    eng._v_cpu = torch.empty((batch, n_host, H, D), dtype=dt, device="cpu", pin_memory=pin)
    eng._k_gpu = torch.empty((batch, rows, H, D), dtype=dt, device=dev)
    eng._v_gpu = torch.empty((batch, rows, H, D), dtype=dt, device=dev)
    eng._kv_bias_gpu = torch.empty((batch, rows, H), dtype=dt, device=dev)
    eng._block_map = torch.full((H, batch, topk), -1, dtype=torch.int64, device=dev)
    eng._new_block_map_buf = torch.empty_like(eng._block_map)
    eng._load_mask = torch.empty_like(eng._block_map)
    eng._cache_lens = torch.full((batch,), (topk - 1) * bs, dtype=torch.int32, device=dev)
    eng._tail_block_idx_on_gpu = topk - 1
    eng._pool_map = torch.full((H, batch, P), -1, dtype=torch.int64, device=dev)
    eng._pool_age = (torch.arange(P, dtype=torch.int64, device=dev) - P).expand(H, batch, P).contiguous()
    eng._pool_target = torch.full((H, batch, topk), -1, dtype=torch.int64, device=dev)
    eng._pool_action = torch.zeros((H, batch, topk), dtype=torch.int8, device=dev)
    eng._pool_stamp_base = 0
    eng._k_gpu[:, topk * bs:].zero_()
    eng._v_gpu[:, topk * bs:].zero_()
    eng._kv_bias_gpu[:, topk * bs:].zero_()
    eng._att_rows = topk * bs
    eng._k_gpu_att = eng._k_gpu[:, :eng._att_rows]
    eng._v_gpu_att = eng._v_gpu[:, :eng._att_rows]
    eng._kv_bias_gpu_att = eng._kv_bias_gpu[:, :eng._att_rows]


def build(model, cache, S_list, horizon: int, ops: RaggedOps, rows_of, pin: bool = True, widths=None) -> RaggedState:
    """A batch with a prompt length per request, without the shipped prefill: allocate() on every layer's engine,
    then for request b and layer l, rows_of(b, l) -> (keys (S_b, H, D), values (S_b, H, D), cis (S_b, H)):
    twin.prefill (the shipped compressions) and DecodeBody.prefill_request (cache_engine.py:309-331 for one request).
    widths: None = ragged_widths(S_list, horizon); a solo run passes its batch's widths."""
    S_list = [int(s) for s in S_list]
    B = len(S_list)
    w = dict(widths) if widths is not None else RDP.ragged_widths(S_list, horizon)
    lay0 = model.layers[0]
    H, D = lay0.num_key_value_heads, lay0.head_dim
    eng0 = cache.layers[0].cache_engine
    dev, dt = torch.device(eng0.device), eng0.dtype
    twins = []
    for lay in cache.layers:
        allocate(lay.cache_engine, B, w["n_host"], pin)
        twins.append(PositionalTwin(B, H, D, w["lk_max"], w["n_host"], dtype=dt, device=dev))
    loader = RB.DecodeBody(False)
    for b, S in enumerate(S_list):
        for l, lay in enumerate(cache.layers):
            k, v, c = rows_of(b, l)
            twins[l].prefill(b, k, c)
            loader.prefill_request(lay.cache_engine, b, k, v, twins[l].total_cis[b, :S])
    for t in twins:
        ops.setup_twin(t)
    st = _state(model, B, H, dev, max(S_list), w, twins, ops, torch.tensor(S_list, dtype=torch.int64))
    rs = RaggedState(model, st, cache, ops)
    rs.meta.update(builder="build", S_list=S_list, horizon=int(horizon), widths=w)
    cache.ragged_state = rs
    return rs


# ------------------------------------------------------------------ the step
def set_activity(rs: RaggedState, active, active_host) -> None:
    """The activity of the next step: active (B,) bool on the device (a device-to-device copy, no sync) and its host
    mirror active_host (B,) bool on the CPU."""
    a = torch.as_tensor(active_host, dtype=torch.bool)
    if tuple(a.shape) != (rs.B,) or tuple(active.shape) != (rs.B,):
        raise RaggedError("active must be (%d,)" % rs.B)
    rs.st.active_host.copy_(a)
    rs.st.active.copy_(active)


def _lengths(rs: RaggedState, twin: PositionalTwin) -> None:
    """n_used <- n_comp and M_b <- pos_before // 64 + 1, in place (the static graph inputs); the negative controls."""
    neg = rs.ops.neg
    if neg == "rebound_lengths" and rs.graphs[0] is not None:
        rs.n_used = torch.empty_like(rs.n_used)                        # the captured graphs keep the old tensors
        rs.M_b = torch.empty_like(rs.M_b)
    rs.n_used.copy_(twin.n_comp)
    rs.M_b.copy_(rs.pos_before // BLOCK + 1)
    if neg == "shared_lengths":                                        # one batch-wide length (the batch maximum)
        rs.n_used.copy_(rs.n_used.amax().expand(rs.B))
        rs.M_b.copy_(rs.M_b.amax().expand(rs.B))
    elif neg == "frozen_widths":                                       # the widths of the first step, for every step
        if rs.M0 is None:
            rs.M0 = rs.M_b.clone()
        rs.M_b.copy_(rs.M0)


def _pool_topk(nl, layer, score, ccis, n_used, M_b, pq, pc, tv_q, ti_q, mask, tv, ti) -> None:
    """The shipped after-pooling region (nosa_llama.py:462-490) with the per-request pooling
    (r2_decode_path.py:323-329). Every tensor is an argument, so a graph keeps exactly these tensors."""
    nosa_pooling_per_request(score, ccis, n_used, M_b, pq, pc, stride=layer.pooling_stride,
                             block_size=layer.block_size, local_blocks=layer.window_blocks,
                             init_blocks=layer.init_blocks)
    torch.topk(pq, nl.QK_SELECT, dim=-1, sorted=False, out=(tv_q, ti_q))
    mask.zero_()
    mask.scatter_(2, ti_q, True)
    pc.masked_fill_(mask, float("inf"))
    torch.topk(pc, layer.topk_blocks, dim=-1, sorted=False, out=(tv, ti))


def _select(rs: RaggedState, l: int, layer, score, twin) -> None:
    st = rs.st
    if rs.score_buf is None:
        rs.score_buf = torch.empty_like(score)
        rs.meta["score_width"] = int(score.shape[-1])
    elif tuple(rs.score_buf.shape) != tuple(score.shape):
        raise RaggedError("the stage-1 score moved from %s to %s: the widths are fixed for the run"
                          % (tuple(rs.score_buf.shape), tuple(score.shape)))
    rs.score_buf.copy_(score)                                          # the graph input (nosa_llama.py:607)
    args = (rs.nl, layer, rs.score_buf, twin.ccis, rs.n_used, rs.M_b, st.pq, st.pc, st.tv_q, st.ti_q, st.mask,
            st.tv, st.ti)
    if not rs.ops.graph:
        _pool_topk(*args)
        return
    g = rs.graphs[l]
    if g is None:
        _pool_topk(*args)                    # eager first: the Triton kernel compiles outside the capture (:447)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _pool_topk(*args)
        rs.graphs[l] = g
        rs.captures += 1
        rs.capture_steps.append(int(st.steps))
    g.replay()
    rs.replays += 1


def _diff_pool(rs: RaggedState, eng):
    """The body's diff_offload argument when the engine has a pool: the shipped section B / B2 (cache_engine.py:480,
    :504-511), with the loads of inactive requests cleared before the pool reads them."""
    kern, st, neg = rs.ops.kernels, rs.st, rs.ops.neg
    diff, pool_update, pool_swap = kern["diff"], kern["pool_update"], kern["pool_swap"]

    def run(old_map, topk_idx, new_map, load_mask):
        diff(old_map, topk_idx, new_map, load_mask)
        if not rs.all_active and neg != "unmasked_pool":
            load_mask.masked_fill_(~st.active.view(1, -1, 1), -1)
        if neg == "no_pool":
            return
        pool_update(old_map, load_mask, eng._pool_map, eng._pool_age, eng._pool_target, eng._pool_action,
                    eng._pool_stamp_base, eng._tail_block_idx_on_gpu)
        pool_swap(eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action, eng.topk,
                  eng.block_size)
        eng._pool_stamp_base += eng.topk
    return run


def _layer(rs: RaggedState, l: int, layer, hidden, pos_flat, cos_sin_cache):
    NL, ops, st, lin = rs.nl, rs.ops, rs.st, rs.ops.linear
    B = hidden.shape[0]
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    twin = st.twins[l]
    eng = rs.cache.layers[l].cache_engine
    residual = hidden
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = lin(h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
    q = q.view(B, -1)
    k = k.view(B, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
    q = q.reshape(B, layer.num_heads, D)
    k = k.reshape(B, Hkv, D)
    v = v.reshape(B, Hkv, D)
    rs.pos_before.copy_(twin.pos)
    twin.step_unchecked(k, cis.view(B, Hkv), st.active)
    _lengths(rs, twin)
    ck = twin.comp_k_flat()
    score = stage1_seqused(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=rs.n_used,
                           ext=ops.kernels.get("stage1_ext"))
    _select(rs, l, layer, score, twin)
    diff = _diff_pool(rs, eng) if rs.pool else ops.kernels["diff"]
    key_states, value_states, kv_bias, cache_lens = rs.body.decode_fixed(
        eng, k.unsqueeze(1), v.unsqueeze(1), twin.total_cis, st.ti, rs.pos_before, st.active, rs.err,
        diff, ops.kernels["gather_k"], ops.kernels["gather_v_bias"])
    if rs.pool and ops.neg != "padded_attention":
        key_states, value_states, kv_bias = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
    attn = NL.flash_attn_nosa_with_kvcache(q.unsqueeze(1), key_states, value_states, kv_bias, cache_seqlens=cache_lens,
                                           num_splits=NL._ATTN_SPLITS)
    attn = attn.view(B, 1, -1)
    hidden = residual + lin(attn, layer.wo)
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = lin(h, layer.gate_up_proj)
    out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
    NL.silu_and_mul(h, out)
    return residual + lin(out, layer.down_proj)


@torch.inference_mode()
def decode_step(model, rs: RaggedState, input_ids, position_ids):
    """One decode step of every request (inactive ones are computed and discarded): input_ids (B, 1) int64,
    position_ids (B, 1) (each request's own position). Returns logits (B, 1, V) float. Host bookkeeping: steps += 1,
    pos_host += active_host. No host read of a device value."""
    st = rs.st
    if tuple(input_ids.shape) != (rs.B, 1):
        raise RaggedError("input_ids must be (%d, 1), got %s" % (rs.B, tuple(input_ids.shape)))
    rs.all_active = bool(st.active_host.all())
    hidden = F.embedding(input_ids, model.embed_tokens)
    pos_flat = position_ids.flatten()
    for l, layer in enumerate(model.layers):
        hidden = _layer(rs, l, layer, hidden, pos_flat, model.cos_sin_cache)
    hidden = rs.nl.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    logits = rs.ops.linear(hidden, model.lm_head).float()
    st.steps += 1
    st.pos_host += st.active_host.to(torch.int64)
    return logits


def errors(rs: RaggedState):
    """ONE sync: (the body's refusal bits, [every twin's overflow flag as 0 / 1]); all 0 when nothing was refused."""
    vals = torch.stack([rs.err.to(torch.int64)] + [t.overflow.to(torch.int64) for t in rs.st.twins]).cpu().tolist()
    return int(vals[0]), [int(x) for x in vals[1:]]
