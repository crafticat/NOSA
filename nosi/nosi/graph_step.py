"""Task 1.6, rung 1 (--graph-step, default off): the cached-NOSA decode step, ALL 32 layers, in ONE CUDA graph per
length bucket. Plan: kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md Task 1.6. Notes: NOTES_graphstep.md.

WHAT IT IS. A graph-safe twin of the shipped step (nosa_llama.LlamaLayer.decode_forward, nosa_llama.py:539-662, and
cache_engine.CacheEngine.decode_update_has_kv_bias_pool, cache_engine.py:414-538, with the cache-layer updates
cache_engine.py:663-692, :717-782). It calls the SAME kernels in the SAME order on the SAME state tensors. It adds no
kernel. The shipped files are not edited: with the flag off, nothing here is imported.

WHAT CHANGES AGAINST THE SHIPPED STEP (each change keeps the values bitwise equal; NOTES_graphstep.md section 2):
  1. Every per-step position (the raw-key column, the cis columns, the tail row, the bias column) comes from a device
     table `sched` (one row per step), read with index_select at t_dev. The shipped step uses host ints in slices.
     index_copy_ / index_select replace the slice copies (pure copies).
  2. The compressed keys live in ONE padded table per layer (B, m_cap, H, D), written in place. Stage 1 reads it with
     cu_seqlens_k = arange(B + 1) * m_cap and seqused_k = m (stage1_seqused.py, the installed op's own seqused_k
     argument; R2 gate r2_kernels_gate G2 / G3 passed bitwise, job 2188818). The shipped step grows the table with
     torch.cat every 16 steps (a new allocation, so a captured graph would read a stale pointer).
  3. The pool LRU stamp is a constant STAMP_K. Before each step, every REAL age (>= 0) goes down by topk. This keeps
     every age comparison of pool_update_kernel.cu equal to the shipped one (the kernel only compares ages and writes
     stamp + m or -P + q). The shipped step passes a host int that grows by topk per step: a capture would freeze it.
     All 32 layers' ages are views of one tensor, so the shift is 3 kernels per step.
  4. The tail write-back (every 64 steps) is one contiguous cudaMemcpyAsync per request into the pinned host rows.
     The shipped step copies a strided view; torch then stages it through PAGEABLE memory and a host copy, which is a
     host sync and cannot be captured.
  5. The pooling reads the stage-1 score directly. The shipped step first copies it into the warm-up's score buffer
     (nosa_llama.py:607) because its pooling graph froze that buffer's address.
  6. The pooling buffers are sized per out_len bucket, and the pooling scalar max_seqlen is the value the shipped
     re-warm froze (the compressed length at the bucket's first step). So the pooled values equal the shipped ones.

BUCKETS. A graph is fixed by a StepKey: (kc, cc, tf, m, out_len, m_frozen, wb_row). kc = this step compresses a key
(every 16 steps), cc = this step compresses cis, tf = the tail block fills (every 64 steps), m = the compressed length
after this step (the stage-1 width), out_len / m_frozen = the shipped re-warm state. plan_steps() computes every key
from the host counters after the prefill, before step 0. capture_all() captures every key of steps >= warmup before
the timed loop. A step whose key has no graph is REFUSED (no capture inside the timed loop).

HOST STATE. The driver does not touch the shipped host counters during the run. export(done) writes them (and the
contiguous compressed keys, the unshifted ages, the stamp) so that the cache equals the shipped cache after `done`
steps. CPU mode (capture=False) runs the same body eagerly: the CPU tests compare it with the shipped step, with fakes.
"""
from __future__ import annotations

import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from . import avail_policy as _avail
from . import cache_engine as CE
from . import nosa_llama as NL
from . import stage1_seqused as S1
from . import transfer_trace as _tt

BLOCK = 64          # cache_engine block_size (nosa_llama.py:680 asserts 64)
TOPK = 64           # attended slots (nosa_llama.py:681 asserts 64)
KERNEL = 32         # key compression window (pooling_block_size)
STRIDE = 16         # key compression stride (pooling_stride)
ROUND = 128         # stage-1 output width rounding (flash_api.cpp: seqlen_k_rounded)
STAMP_K = 64 << 31  # the constant LRU stamp of the shifted age frame (module docstring, change 3)
SCHED_COLS = ("n_col", "cis_col", "tl_col", "tail_pos", "bias_col")
CAPTURED_REGION = ("GraphStep._body", "GraphStep._layer", "GraphStep._engine")   # linted by the CPU tests
_STAGE1_EXT = None  # None = the installed infllm_v2.C; a CPU test sets a fake


class StepKey(NamedTuple):
    kc: bool
    cc: bool
    tf: bool
    m: int
    out_len: int
    m_frozen: int
    wb_row: Optional[int]


def round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def prefill_counters(S: int) -> dict:
    """The host counters of a fresh shipped prefill of S tokens per request (cache_engine.py:189, :309, :678, :731,
    :770; nosa_llama.py:211-234 compress_k): seq S, raw keys 16 + S % 16, compressed keys S // 16 - 1, tail S % 64,
    tail cis 16 + S % 16, cis S, stamp 0."""
    return dict(S=int(S), n0=STRIDE + S % STRIDE, m0=S // STRIDE - 1, tail0=S % BLOCK, tl0=STRIDE + S % STRIDE,
                cis0=int(S), stamp0=0)


def plan_steps(S, n0, m0, tail0, tl0, cis0, steps, stamp0=0):
    """The host values the SHIPPED step would use at each of `steps` decode steps after the state (S, n0, m0, tail0,
    tl0, cis0, stamp0). Pure. Returns (rows, end): rows[t] a dict (t, seq, kc, n_col, m, cis_col, tl_col, cc, tail_pos,
    bias_col, tf, wb_row, stamp, out_len, width, m_frozen, rewarm, key); end the counters after the last step.
    `stamp` is the shipped LRU stamp of the POOL path (cache_engine.py:507, :511); at P = 0 the shipped stamp never
    moves, and export() leaves it alone.
    Sources: update_no_compress_k_decode cache_engine.py:681-692, update_compress_k_decode :663-670,
    update_uncompressed_cis :780-781, update_cis :752-762, decode_update_has_kv_bias_pool :450-464 and :523-531, the
    shipped re-warm rule r2_base_cell.warm_state / rewarm_need (sustained_core.py:217-235)."""
    rows = []
    seq, n, m, tail, tl, cis = int(S), int(n0), int(m0), int(tail0), int(tl0), int(cis0)
    warm = None
    for t in range(int(steps)):
        m_next = m + (1 if n >= KERNEL else 0)
        now = dict(out_len=-(-(seq + 1) // BLOCK), width=round_up(m_next, ROUND), m_frozen=m_next)
        rewarm = warm is None or now["out_len"] != warm["out_len"] or now["width"] != warm["width"]
        if rewarm:
            warm = now
        kc = n >= KERNEL
        if kc:
            n_col, n, m = STRIDE, STRIDE + 1, m + 1
        else:
            n_col, n = n, n + 1
        cis_col, cis = cis, cis + 1
        tl_col, tl = tl, tl + 1
        cc = tl == KERNEL
        if cc:
            tl = STRIDE
        seq_after = seq + 1
        tail_pos = (TOPK - 1) * BLOCK + tail
        tail += 1
        if tail > BLOCK:                          # the shipped assert, cache_engine.py:474-476
            raise ValueError("step %d: the tail ran past its block (%d > %d)" % (t, tail, BLOCK))
        tf = tail == BLOCK
        wb_row = seq_after - BLOCK if tf else None
        if tf:
            tail = 0
        if round_up(m, ROUND) != warm["width"]:   # the re-warm rule keeps the stage-1 width fixed in a bucket
            raise ValueError("step %d: width %d is not the bucket's %d" % (t, round_up(m, ROUND), warm["width"]))
        key = StepKey(kc, cc, tf, m, warm["out_len"], warm["m_frozen"], wb_row)
        rows.append(dict(t=t, seq=seq, kc=kc, n_col=n_col, m=m, cis_col=cis_col, tl_col=tl_col, cc=cc,
                         tail_pos=tail_pos, bias_col=seq_after - 1, tf=tf, wb_row=wb_row,
                         stamp=int(stamp0) + TOPK * t, out_len=warm["out_len"], width=warm["width"],
                         m_frozen=warm["m_frozen"], rewarm=rewarm, key=key))
        seq = seq_after
    end = dict(seq=seq, n=n, m=m, tail=tail, tl=tl, cis=cis, stamp=int(stamp0) + TOPK * int(steps))
    return rows, end


def capture_plan(rows, warmup: int) -> List[StepKey]:
    """The distinct keys of the steps >= warmup, in first-use order: the graphs capture_all() records."""
    out, seen = [], set()
    for r in rows[int(warmup):]:
        if r["key"] not in seen:
            seen.add(r["key"])
            out.append(r["key"])
    return out


def shift_in(age: torch.Tensor, stamp_first: int) -> torch.Tensor:
    """Shipped ages before the first graph step (whose shipped stamp is stamp_first) -> the shifted frame BEFORE that
    step's shift: reals + STAMP_K - stamp_first + TOPK; empties (< 0) unchanged."""
    return torch.where(age >= 0, age + (STAMP_K - int(stamp_first) + TOPK), age)


def shift_out(age: torch.Tensor, stamp_last: int) -> torch.Tensor:
    """Shifted ages after the step whose shipped stamp is stamp_last -> shipped ages: reals - STAMP_K + stamp_last."""
    return torch.where(age >= 0, age - STAMP_K + int(stamp_last), age)


class GraphStep:
    """The --graph-step driver for one cache after a fresh shipped prefill (module docstring)."""

    def __init__(self, model, cache, cu_seqlens, steps: int, warmup: int = 2, capture: bool = True,
                 pos_dtype=torch.int32, shift_ages: bool = True):
        if _tt.TRACE is not None or _avail.POLICY is not None:
            raise RuntimeError("graph_step: NOSI_TRANSFER_TRACE / NOSI_AVAIL hooks are host code; unset them")
        if not 1 <= int(warmup) <= int(steps):
            raise ValueError("graph_step: warmup %r must be in 1 .. steps (%r)" % (warmup, steps))
        self.model, self.cache, self.cu_q = model, cache, cu_seqlens
        self.steps, self.warmup, self.capture = int(steps), int(warmup), bool(capture)
        self.shift_ages = bool(shift_ages)       # False only in a CPU negative control
        self.layers = list(cache.layers)
        self.engines = [L.cache_engine for L in self.layers]
        L0, e0 = self.layers[0], self.engines[0]
        self.host0 = dict(S=e0.seq_length, n0=L0.no_compress_k_len, m0=L0.cached_compressed_max_seqlen,
                          tail0=e0._tail_block_len_on_gpu, tl0=L0.tail_cis_len, cis0=L0.cis_len,
                          stamp0=e0._pool_stamp_base)
        for l, (L, e) in enumerate(zip(self.layers, self.engines)):
            got = dict(S=e.seq_length, n0=L.no_compress_k_len, m0=L.cached_compressed_max_seqlen,
                       tail0=e._tail_block_len_on_gpu, tl0=L.tail_cis_len, cis0=L.cis_len, stamp0=e._pool_stamp_base)
            if got != self.host0 or L.seq_length != e.seq_length:
                raise ValueError("graph_step: layer %d counters %s differ from layer 0 %s" % (l, got, self.host0))
            if e._tail_block_idx_on_gpu != e.topk - 1 or e.topk != TOPK or e.block_size != BLOCK:
                raise ValueError("graph_step: layer %d is not the shipped topk %d / block %d layout" % (l, TOPK, BLOCK))
            if L.compress_k_cache_varlen is None or L.total_cis is None or L.cis_prefilled_batch != e._k_gpu.shape[0]:
                raise ValueError("graph_step: layer %d is not a complete prefill" % l)
        self.rows, self.end = plan_steps(steps=self.steps, **self.host0)
        self.keys = [r["key"] for r in self.rows]
        self.pool = e0.pool_blocks > 0
        dev = e0._k_gpu.device
        self.device = dev
        B, _, H, D = e0._k_gpu.shape
        self.B, self.H, self.D = B, H, D
        self.m_cap = self.end["m"] + 1
        V = model.lm_head.shape[0]
        with torch.inference_mode():
            self.ck = []
            for L in self.layers:
                old = L.compress_k_cache_varlen
                pad = torch.zeros((B, self.m_cap, H, D), dtype=old.dtype, device=dev)
                pad[:, :old.shape[1]].copy_(old)
                self.ck.append(pad)
                L.compress_k_cache_varlen = None        # export() rebuilds it; frees the shipped table now
            self.cu_k = torch.arange(B + 1, dtype=torch.int32, device=dev) * self.m_cap
            self.seqused = torch.zeros((B,), dtype=torch.int32, device=dev)
            self.sched = torch.tensor([[r[c] for c in SCHED_COLS] for r in self.rows], dtype=torch.int64, device=dev)
            self.t_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
            self.ids = torch.zeros((B, 1), dtype=torch.int64, device=dev)
            self.pos = torch.zeros((B, 1), dtype=pos_dtype, device=dev)
            self.logits = torch.zeros((B, 1, V), dtype=torch.float32, device=dev)
            dt = model.layers[0].wqkv.dtype
            self.tv_q = torch.empty((H, B, NL.QK_SELECT), dtype=dt, device=dev)
            self.ti_q = torch.empty((H, B, NL.QK_SELECT), dtype=torch.int64, device=dev)
            self.tv = torch.empty((H, B, TOPK), dtype=dt, device=dev)
            self.ti = torch.empty((H, B, TOPK), dtype=torch.int64, device=dev)
            self.pool_bufs = {}
            for ol in sorted({k.out_len for k in self.keys}):
                self.pool_bufs[ol] = (torch.empty((2 * H, B, ol), dtype=dt, device=dev),
                                      torch.empty((H, B, ol), dtype=torch.bool, device=dev))
            self.ages = None
            if self.pool:
                self.ages = torch.stack([e._pool_age for e in self.engines]).contiguous()
                self.ages.copy_(shift_in(self.ages, self.host0["stamp0"]))
                for l, e in enumerate(self.engines):
                    e._pool_age = self.ages[l]
        self.graphs: Dict[StepKey, object] = {}
        self.capture_ms: Dict[StepKey, float] = {}
        self.stream = torch.cuda.Stream() if self.capture else None
        self.graph_pool = None
        self.done = 0
        self.replays = 0
        self.eager_steps = 0

    # ------------------------------------------------------------------ the captured region
    def _body(self, key: StepKey):
        """One whole decode step for `key` (nosa_llama.py:796-843). No host sync, no host state, no data-dependent
        shape: every per-step value is read from the device (sched at t_dev) or is static in key."""
        model = self.model
        row = torch.index_select(self.sched, 0, self.t_dev)
        idx = {c: row[0, j:j + 1] for j, c in enumerate(SCHED_COLS)}
        self.seqused.fill_(key.m)
        if self.pool and self.shift_ages:
            self.ages.sub_(self.ages.ge(0).long().mul_(TOPK))
        hs = F.embedding(self.ids, model.embed_tokens)
        for l, layer in enumerate(model.layers):
            hs = self._layer(l, layer, hs, key, idx)
        hs = NL.layer_norm(hs, w=model.norm_weight, eps=model.norm_variance_epsilon)
        self.logits.copy_(F.linear(hs, model.lm_head))

    def _layer(self, l: int, layer, hs, key: StepKey, idx):
        """LlamaLayer.decode_forward (nosa_llama.py:539-662) with the cache updates of cache_engine.py:663-692,
        :717-782 inlined; the engine update is _engine."""
        Lc = self.layers[l]
        B, H, D = self.B, self.H, self.D
        residual = hs
        hs = NL.layer_norm(hs, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = F.linear(hs, layer.wqkv)
        q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size,
                                      layer.num_key_value_heads)
        q = q.view(B, -1)
        k = k.view(B, -1)
        NL.apply_rope_with_cos_sin_cache_inplace(self.pos.flatten(), q, k, layer.head_dim, self.model.cos_sin_cache,
                                                 True)
        q = q.reshape(B, layer.num_heads, layer.head_dim)
        k = k.reshape(B, H, D)
        v = v.reshape(B, H, D)
        # compress k: update_no_compress_k_decode (cache_engine.py:681-692) + update_compress_k_decode (:663-670)
        k4 = k.unsqueeze(1)
        nck = Lc.no_compress_k_cache
        if key.kc:
            to_compress = nck.clone()
            nck[:, :STRIDE].copy_(nck[:, STRIDE:])
            nck[:, STRIDE:STRIDE + 1].copy_(k4)
            new_ck = to_compress.mean(dim=1, keepdim=True)
            self.ck[l][:, key.m - 1:key.m].copy_(new_ck)
            Lc.cached_compressed_cu_seqlens += Lc.cached_compressed_cu_seqlens_adder
        else:
            nck.index_copy_(1, idx["n_col"], k4)
        ckf = self.ck[l].view(B * self.m_cap, H, D)
        # compress cis: update_uncompressed_cis (:780-782) + update_cis (:752-763)
        ucis = Lc.total_cis
        ucis.index_copy_(1, idx["cis_col"], cis)
        tcis = Lc.tail_cis
        tcis.index_copy_(2, idx["tl_col"], cis.permute(2, 0, 1))
        if key.cc:
            tail_comp = tcis.mean(dim=-1, keepdim=True)
            # the shipped write (:759); comp_cis_len is the width of compressed_cis, so the slice is empty
            Lc.compressed_cis[:, :, Lc.comp_cis_len:Lc.comp_cis_len + 1].copy_(tail_comp)
            tcis[:, :, :STRIDE].copy_(tcis[:, :, STRIDE:])
        compressed_cis = Lc.compressed_cis
        # stage 1 (nosa_llama.py:592-601): the padded table, per-request length m (static in key)
        score = S1.stage1_seqused(q, ckf, ckf, self.cu_q, self.cu_k, 1, key.m, seqused_k=self.seqused,
                                  ext=_STAGE1_EXT)
        # pooling + top-k: the shipped after_pooling_graph region (nosa_llama.py:462-490)
        pb, mask = self.pool_bufs[key.out_len]
        mp = pb[:H]
        mpc = pb[H:]
        NL.nosa_pooling(score, compressed_cis, key.m_frozen, mp, mpc, stride=layer.pooling_stride,
                        block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
        torch.topk(mp, NL.QK_SELECT, dim=-1, sorted=False, out=(self.tv_q, self.ti_q))
        mask.zero_()
        mask.scatter_(2, self.ti_q, True)
        mpc.masked_fill_(mask, float("inf"))
        torch.topk(mpc, layer.topk_blocks, dim=-1, sorted=False, out=(self.tv, self.ti))
        k_att, v_att, bias_att, cache_lens = self._engine(l, k, v, ucis, self.ti, key, idx)
        attn = NL.flash_attn_nosa_with_kvcache(q.unsqueeze(1), k_att, v_att, bias_att, cache_seqlens=cache_lens,
                                               num_splits=NL._ATTN_SPLITS)
        attn = attn.view(B, 1, -1)
        hs = F.linear(attn, layer.wo)
        hs = residual + hs
        residual = hs
        hs = NL.layer_norm(hs, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hs = F.linear(hs, layer.gate_up_proj)
        dd = hs.shape[-1] // 2
        out = torch.empty(hs.shape[:-1] + (dd,), dtype=hs.dtype, device=hs.device)
        NL.silu_and_mul(hs, out)
        hs = F.linear(out, layer.down_proj)
        hs = residual + hs
        return hs

    def _engine(self, l: int, k, v, ucis, topk_idx, key: StepKey, idx):
        """decode_update_has_kv_bias_pool (cache_engine.py:447-538); decode_update_has_kv_bias (:342-412) at P = 0."""
        eng = self.engines[l]
        bs = eng.block_size
        tail = eng._tail_block_idx_on_gpu
        eng._cache_lens[...] += 1
        eng._k_gpu.index_copy_(1, idx["tail_pos"], k.unsqueeze(1))
        eng._v_gpu.index_copy_(1, idx["tail_pos"], v.unsqueeze(1))
        eng._kv_bias_gpu.index_copy_(1, idx["tail_pos"], ucis.index_select(1, idx["bias_col"]))
        CE.diff.diff_offload(eng._block_map, topk_idx, eng._new_block_map_buf, eng._load_mask)
        if self.pool:
            CE._pool.pool_update(eng._block_map, eng._load_mask, eng._pool_map, eng._pool_age, eng._pool_target,
                                 eng._pool_action, STAMP_K, tail)
            CE._flash_pool_swap(eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action,
                                eng.topk, bs)
        eng._block_map.copy_(eng._new_block_map_buf, non_blocking=True)
        CE.flash_h2d_from_mask(eng._k_gpu, eng._k_cpu, eng._load_mask, bs)
        CE.flash_h2d_from_mask_bias(eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, ucis, eng._load_mask, bs)
        if key.tf:
            tb = tail * bs
            r0 = key.wb_row
            for b in range(self.B):
                eng._k_cpu[b, r0:r0 + bs].copy_(eng._k_gpu[b, tb:tb + bs], non_blocking=True)
            for b in range(self.B):
                eng._v_cpu[b, r0:r0 + bs].copy_(eng._v_gpu[b, tb:tb + bs], non_blocking=True)
            eng._block_map[..., tail] += 1
            eng._cache_lens[...] = (eng.topk - 1) * bs
        if self.pool:
            return eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att, eng._cache_lens
        return eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._cache_lens

    # ------------------------------------------------------------------ the driver (host, outside the graphs)
    def _eager(self, key: StepKey):
        if self.stream is None:
            self._body(key)
            return
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            self._body(key)
        cur.wait_stream(self.stream)

    def warm_pooling(self):
        """Compile nosa_pooling for every bucket's arguments on SCRATCH tensors (Triton specialises integer arguments
        by value; a compile inside a capture is not allowed). Writes no state."""
        H, B = self.H, self.B
        L0 = self.layers[0]
        layer = self.model.layers[0]
        seen = set()
        with torch.inference_mode():
            for k in self.keys:
                sig = (k.out_len, k.m_frozen, round_up(k.m, ROUND))
                if sig in seen:
                    continue
                seen.add(sig)
                score = torch.zeros((H, B, sig[2]), dtype=self.tv.dtype, device=self.device)
                ccis = torch.zeros_like(L0.compressed_cis)
                pb = torch.empty((2 * H, B, k.out_len), dtype=self.tv.dtype, device=self.device)
                NL.nosa_pooling(score, ccis, k.m_frozen, pb[:H], pb[H:], stride=layer.pooling_stride,
                                block_size=layer.block_size, local_blocks=layer.window_blocks,
                                init_blocks=layer.init_blocks)

    def capture_all(self):
        """Capture one graph per key of the steps >= warmup (capture_plan), all before the timed loop. Uses the raw
        capture API on one side stream and one shared pool: no gc.collect() / empty_cache() per graph (the
        torch.cuda.graph context runs both on entry; NOTES_graphstep.md section 3). Returns the total ms."""
        if not self.capture:
            raise RuntimeError("graph_step: capture_all() with capture=False")
        if self.done < self.warmup:
            raise RuntimeError("graph_step: run the %d eager warm-up steps first (done %d)" % (self.warmup, self.done))
        t_all = time.perf_counter()
        self.warm_pooling()
        torch.cuda.synchronize()
        if self.graph_pool is None:
            self.graph_pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for key in capture_plan(self.rows, self.warmup):
                if key in self.graphs:
                    continue
                t0 = time.perf_counter()
                g = torch.cuda.CUDAGraph()
                cur = torch.cuda.current_stream()
                self.stream.wait_stream(cur)
                with torch.cuda.stream(self.stream):
                    g.capture_begin(pool=self.graph_pool)
                    ok = False
                    try:
                        self._body(key)
                        ok = True
                    finally:
                        try:
                            g.capture_end()
                        except Exception:
                            if ok:
                                raise
                cur.wait_stream(self.stream)
                torch.cuda.synchronize()
                self.graphs[key] = g
                self.capture_ms[key] = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        return (time.perf_counter() - t_all) * 1e3

    def step(self, t: int, tok, pos):
        """Decode step t (0-based, in order). tok (B, 1) int64, pos (B, 1) the RoPE positions. Returns the static
        logits (B, 1, V) float32: read them before the next step."""
        if t != self.done:
            raise RuntimeError("graph_step: step %d out of order (next is %d)" % (t, self.done))
        if t >= self.steps:
            raise RuntimeError("graph_step: step %d beyond the planned %d steps" % (t, self.steps))
        key = self.keys[t]
        with torch.inference_mode():
            self.ids.copy_(tok)
            self.pos.copy_(pos)
            self.t_dev.fill_(t)
            if t < self.warmup or not self.capture:
                self._eager(key)
                self.eager_steps += 1
            else:
                g = self.graphs.get(key)
                if g is None:
                    raise RuntimeError("graph_step: no graph for step %d (key %s); a capture inside the timed loop "
                                       "is refused" % (t, key))
                g.replay()
                self.replays += 1
        self.done = t + 1
        return self.logits

    def export(self, done: Optional[int] = None):
        """Write the shipped host state after `done` steps (default: the steps run) into the cache: counters, the
        contiguous compressed keys, the unshifted pool ages and the stamp. Afterwards the shipped eager step can
        continue on this cache."""
        done = self.done if done is None else int(done)
        _, end = plan_steps(steps=done, **self.host0)
        with torch.inference_mode():
            if self.pool:
                stamp_last = self.host0["stamp0"] + TOPK * (done - 1)
                self.ages.copy_(shift_out(self.ages, stamp_last))
            for l, (L, e) in enumerate(zip(self.layers, self.engines)):
                L.seq_length = e.seq_length = end["seq"]
                L.no_compress_k_len = end["n"]
                L.cached_compressed_max_seqlen = end["m"]
                L.cis_len = end["cis"]
                L.tail_cis_len = end["tl"]
                e._tail_block_len_on_gpu = end["tail"]
                if self.pool:
                    # FIX 2: the shipped step moves the stamp on the POOL path only (cache_engine.py:511, inside
                    # decode_update_has_kv_bias_pool, bound at :173 when P > 0). At P = 0 the bound method is
                    # decode_update_has_kv_bias (:175), which never touches it: it stays at its value (:140).
                    e._pool_stamp_base = end["stamp"]
                L.compress_k_cache_varlen = self.ck[l][:, :end["m"]].contiguous()
        if hasattr(self.cache, "_seen_tokens"):
            self.cache._seen_tokens += self.H * done   # InfLLMv2Cache.decode_update_kv adds key_states.shape[-2]
        return end

    def state_pointers(self) -> Dict[str, int]:
        """{name: data_ptr} of every state tensor the step reads or writes (the GS-PTR gate)."""
        out = {}
        for l, (L, e) in enumerate(zip(self.layers, self.engines)):
            for k in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_block_map", "_new_block_map_buf",
                      "_load_mask", "_cache_lens"):
                out["eng%d.%s" % (l, k)] = getattr(e, k).data_ptr()
            if self.pool:
                for k in ("_pool_map", "_pool_age", "_pool_target", "_pool_action", "_k_gpu_att", "_v_gpu_att",
                          "_kv_bias_gpu_att"):
                    out["eng%d.%s" % (l, k)] = getattr(e, k).data_ptr()
            for k in ("no_compress_k_cache", "total_cis", "tail_cis", "compressed_cis", "cached_compressed_cu_seqlens"):
                out["layer%d.%s" % (l, k)] = getattr(L, k).data_ptr()
            out["ck%d" % l] = self.ck[l].data_ptr()
        for k in ("cu_k", "seqused", "sched", "t_dev", "ids", "pos", "logits", "tv_q", "ti_q", "tv", "ti"):
            out["gs.%s" % k] = getattr(self, k).data_ptr()
        for ol, (pb, mask) in sorted(self.pool_bufs.items()):
            out["gs.pb%d" % ol] = pb.data_ptr()
            out["gs.mask%d" % ol] = mask.data_ptr()
        out["cu_q"] = self.cu_q.data_ptr()
        return out

    def report(self) -> dict:
        """The capture plan and costs: per graph its key, first step, kind and capture ms."""
        first = {}
        for r in self.rows:
            first.setdefault(r["key"], r["t"])
        graphs = []
        for key, ms in self.capture_ms.items():
            kind = "+".join(n for n, on in (("kc", key.kc), ("cc", key.cc), ("tf", key.tf)) if on) or "plain"
            graphs.append(dict(first_step=first[key], kind=kind, m=key.m, out_len=key.out_len, m_frozen=key.m_frozen,
                               capture_ms=round(ms, 2)))
        graphs.sort(key=lambda g: g["first_step"])
        return dict(steps=self.steps, warmup=self.warmup, eager_steps=self.eager_steps, replays=self.replays,
                    n_graphs=len(self.graphs), planned_graphs=len(capture_plan(self.rows, self.warmup)),
                    capture_ms_sum=round(sum(self.capture_ms.values()), 1), graphs=graphs, m_cap=self.m_cap,
                    host0=self.host0, pool=self.pool, shift_ages=self.shift_ages)
