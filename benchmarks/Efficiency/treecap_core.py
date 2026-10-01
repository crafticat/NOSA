"""TREE-VECTOR PIGGYBACK CAPTURE: the pure extraction (no CUDA; CPU-tested by retroinfer-eval tests/test_cache_curve.py on a
fake engine). The capture driver is treecap_capture.py, a SEPARATE small process outside every timed region. Spec:
retroinfer-eval docs/evidence/treecap_scores_L16128/PIGGYBACK_SPEC.md (2026-10-01).

WHAT IS CAPTURED (per decode call t, layer l in the set, request b, KV head h; NOSA-8B: 32 q heads, 2 KV heads, head_dim 128,
q head j -> KV head j // 16, the GQA layout of infllmv2_sparse_attention.py view(total_q, nheads_k, nheads_per_group, D)):
  q_rope      post-RoPE queries (B, 32, 128) bf16, right after apply_rope_with_cos_sin_cache_inplace in decode_forward(_warmup)
  k_rope_new  the call's post-RoPE key (B, 2, 128) bf16, same instant; position_ids (B,)
  n_comp      the compressed-key count the stage-1 kernel sees at the call (update_compress_k_decode's max_seqlen) and whether a
              16-token chunk completed at this call (no_compress_k was not None)
  stage1      the UNPOOLED stage-1 score (2, B, N_comp) as returned by infllmv2_attn_stage1_fast (before nosa_pooling and before
              any forcing) and the compressed CIS (2, B, N_comp) update_cis returned
  pooled      the pooled QK / CIS buffers AFTER forcing (transfer_trace.record_scores) + topk_idx_buf_q (QK top-33) and
              topk_idx_buf (the final 64 ids)
  maps/masks  block_map after the diff and the load mask (C = 63 cache-before = the previous call's map; misses = the mask)
ONCE, after the last call (host reads only): group means G[l, b, h, m] = fp32 mean of post-RoPE keys over tokens [64m, 64m+64)
for the COMPLETE groups (host cache _k_cpu holds post-RoPE keys: RoPE runs before prefill_update_kv, nosa_llama.py:208), raw
per-token keys of request 0 (layers TC_RAW_LAYERS), the final compressed keys (compress_k_cache_varlen, append-only: the prefix
at call t is its first n_comp[t] rows) and the raw CIS total_cis[:, :L+T].

CAUSAL ACCOUNTING: a 64-token group m is usable from the first call whose position is >= 64(m+1) (never earlier); at L = 16128
(= 252 x 64) and 64 calls every position 16128..16191 lies in tail block 252, so groups 0..251 are frozen at prefill and NO group
completes during decode (newly_completed == 0 at every call, recorded, not assumed). Compressed chunks DO complete during decode
(about every 16 tokens); n_comp is recorded per call and compared with the decode rule floor((pos - 32) / 16) + 1 (n_comp_rule).

STORAGE: one NPZ (bf16 stored as uint16 bit patterns + a dtype table), one JSON (shapes, dtypes, sha256 per array, pins,
checkpoint identity, config, causal accounting, checks, explicit MISSING), SHA256SUMS; the total must stay < 200 MB (assemble
drops the optional raw keys of request 0 to fewer layers before it would exceed, and says so).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

NOSA_CONFIG = dict(block_size=64, topk_blocks=64, tail_slot=63, qk_select=33, init_blocks=1, window_blocks=16, local_forced_blocks_in_kernel=17,
                   compress_kernel_tokens=32, compress_stride_tokens=16, pooling_set_size=5, pooling_padding=1, q_heads=32, kv_heads=2, head_dim=128,
                   q_head_to_kv_head="kv = q // 16",
                   pool="block m = max over compressed chunks n = 4m-1 .. 4m+3 (set_size 5, padding 1) of the stage-1 score, chunks clipped to "
                        "[0, n_comp); chunk n = mean of post-RoPE keys of tokens [16n, 16n+32) (max_pooling_fused.py nosa_pool_kernel)",
                   forced="the pooling kernel writes +inf for m < init_blocks and M - m <= local_blocks + 1 = 17 into BOTH pooled buffers "
                          "(max_pooling_fused.py boundary_mask; nosa_pooling passes local_blocks+1)",
                   selection="QK top-33 (sorted=False) -> those blocks +inf in the CIS buffer -> final = top-64 of the CIS buffer "
                             "(nosa_llama.py after_pooling_graph)")
SIZE_LIMIT_BYTES = 200 * 10 ** 6
MISSING_ALWAYS = (
    "per-q-head stage-1 scores: infllmv2_attn_stage1_fast returns one score per KV head; its reduction over the 16 q heads of a group "
    "happens inside the CUDA kernel and is not exposed (the per-q-head QK can be RECOMPUTED on CPU from q_rope and the compressed keys)",
    "values and kv_bias rows (not part of the tree-vector question; host V exists in _v_cpu but is not exported)",
    "full logits (not exported; the selection ground truth is the block map)",
    "cache-before / misses at capacities other than C = 63 (derive them by LRU replay of block_map; the cache-size curve's capture at "
    "each C records the true natural misses with the pool, in a different process)",
    "NOSA-1B data (its trace arm aborted in prefill)")


# --------------------------------------------------------------------------------------------- bf16 / checksums
def bf16_bits(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


def as_np(t) -> np.ndarray:
    """bf16 tensors as their uint16 bit patterns, other tensors as numpy, arrays unchanged."""
    if torch.is_tensor(t):
        return bf16_bits(t) if t.dtype == torch.bfloat16 else t.detach().cpu().contiguous().numpy()
    return np.asarray(t)


def bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


# --------------------------------------------------------------------------------------------- the offline model
def pool_offline(score: torch.Tensor, n_valid: int, M: int, init: int = 1, local: int = 17, set_size: int = 5, pad: int = 1,
                 forced_value: float = float("inf")) -> torch.Tensor:
    """nosa_pool_kernel on the CPU: pooled[..., m] = max(score[..., n] for n in [m(set_size-1) - pad, + set_size) clipped to
    [0, n_valid)), -inf when empty; forced blocks (m < init or M - m <= local) = forced_value. score (..., N) any float dtype;
    returns float32 (the max is exact in any dtype)."""
    x = score.float()
    out = torch.full(x.shape[:-1] + (M,), float("-inf"), dtype=torch.float32)
    for m in range(M):
        lo = max(0, m * (set_size - 1) - pad)
        hi = min(int(n_valid), m * (set_size - 1) - pad + set_size, x.shape[-1])
        if hi > lo:
            out[..., m] = x[..., lo:hi].max(dim=-1).values
    forced = torch.zeros(M, dtype=torch.bool)
    forced[:init] = True
    forced[M - local:] = True
    out[..., forced] = forced_value
    return out


def forced_blocks(M: int, init: int = 1, local: int = 17) -> torch.Tensor:
    f = torch.zeros(M, dtype=torch.bool)
    f[:init] = True
    f[max(0, M - local):] = True
    return f


def select_offline(pooled_qk: torch.Tensor, pooled_cis: torch.Tensor, qk_select: int = 33, topk: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """(qk top-33 as a bool mask, final top-64 as a bool mask) over the block axis (sets: the GPU's topk is sorted=False)."""
    q = pooled_qk.float()
    c = pooled_cis.float().clone()
    i33 = torch.topk(q, qk_select, dim=-1).indices
    m33 = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, i33, True)
    c[m33] = float("inf")
    i64 = torch.topk(c, topk, dim=-1).indices
    m64 = torch.zeros_like(c, dtype=torch.bool).scatter_(-1, i64, True)
    return m33, m64


def ids_to_mask(ids: torch.Tensor, M: int) -> torch.Tensor:
    m = torch.zeros(ids.shape[:-1] + (M,), dtype=torch.bool)
    v = ids.to(torch.int64)
    ok = (v >= 0) & (v < M)
    return m.scatter_(-1, torch.where(ok, v, torch.zeros_like(v)), ok)


def group_means(k_host: torch.Tensor, n_complete: int, block: int = 64) -> torch.Tensor:
    """k_host (B, S, H, D) post-RoPE keys -> fp32 (B, H, n_complete, D): mean over tokens [64m, 64m+64)."""
    B, S, H, D = k_host.shape
    assert n_complete * block <= S, (n_complete, block, S)
    x = k_host[:, :n_complete * block].float().view(B, n_complete, block, H, D).mean(dim=2)
    return x.permute(0, 2, 1, 3).contiguous()


def compress_k_ref(k: torch.Tensor, n_comp: int, kernel: int = 32, stride: int = 16) -> torch.Tensor:
    """(B, n_comp, H, D) fp32: chunk n = mean of k over tokens [stride*n, stride*n + kernel) (the prefill compress_k and the
    decode update_no_compress_k_decode + mean(dim=1) rule)."""
    B, S, H, D = k.shape
    out = torch.empty((B, n_comp, H, D), dtype=torch.float32)
    for n in range(n_comp):
        out[:, n] = k[:, stride * n: stride * n + kernel].float().mean(dim=1)
    return out


def n_comp_rule(pos: int, kernel: int = 32, stride: int = 16) -> int:
    """The compressed-key count the stage-1 kernel sees at the decode call of position pos: floor((pos - 32) / 16) + 1. Chunk n
    covers tokens [16n, 16n + 32) and is compressed at the first call AFTER its last token was written (update_no_compress_k_decode
    compresses when the buffer already holds 32 tokens, BEFORE appending the call's key; cache_engine.py:984-995), e.g. 1007 at
    positions 16128..16143 and 1008 from 16144. PIGGYBACK_SPEC wrote floor((pos - 16) / 16) + 1, one chunk early; the recorded value
    decides, this rule is only compared with it."""
    return (int(pos) - kernel) // stride + 1


def causal_groups(positions: Sequence[int], block: int = 64) -> Dict:
    """Per call: groups usable at that call (m with 64(m+1) <= pos) and the groups that became usable at it."""
    avail = [int(p) // block for p in positions]
    newly = [0] + [max(0, a - b) for a, b in zip(avail[1:], avail[:-1])]
    return dict(rule="group m usable from the first call with position >= 64(m+1)", groups_available=avail, newly_completed=newly,
                any_completed_during_decode=any(x > 0 for x in newly[1:]), first_call_groups=avail[0] if avail else None)


# --------------------------------------------------------------------------------------------- the capture recorder
class Recorder:
    """Per-call device clones keyed by (call, layer); everything on the device until harvest (no synchronize inside a call).
    The driver's wrappers call note_* from inside the model's decode; the recorder knows the current call and layer from the
    trace (begin_step / begin_layer)."""

    def __init__(self, layers: Sequence[int], n_calls: int):
        self.layers = tuple(int(x) for x in layers)
        self.n_calls = int(n_calls)
        self.slot = {l: i for i, l in enumerate(self.layers)}
        self.q, self.k, self.pos = {}, {}, {}
        self.ncomp, self.chunk_done = {}, {}
        self.stage1, self.ccis = {}, {}
        self.top33, self.top64 = {}, {}
        self.calls_seen = set()

    def want(self, call, layer) -> bool:
        return call is not None and layer is not None and int(layer) in self.slot and 0 <= int(call) < self.n_calls

    def note_rope(self, call, layer, positions, query, key, n_heads, n_kv, head_dim):
        if not self.want(call, layer):
            return
        B = query.shape[0]
        self.q[(call, layer)] = query.detach().reshape(B, n_heads, head_dim).clone()
        self.k[(call, layer)] = key.detach().reshape(B, n_kv, head_dim).clone()
        self.pos[(call, layer)] = positions.detach().reshape(-1).clone()
        self.calls_seen.add(int(call))

    def note_compress(self, call, layer, max_seqlen, completed):
        if self.want(call, layer):
            self.ncomp[(call, layer)] = int(max_seqlen)
            self.chunk_done[(call, layer)] = bool(completed)

    def note_stage1(self, call, layer, score):
        if self.want(call, layer):
            self.stage1[(call, layer)] = score.detach().clone()

    def note_cis(self, call, layer, ccis, n_valid):
        if self.want(call, layer):
            self.ccis[(call, layer)] = ccis.detach()[..., :int(n_valid)].clone()

    def note_select(self, call, layer, top33, top64):
        if self.want(call, layer):
            self.top33[(call, layer)] = top33.detach().clone()
            self.top64[(call, layer)] = top64.detach().clone()

    def complete(self) -> Dict:
        need = [(c, l) for c in range(self.n_calls) for l in self.layers]
        miss = {n: [k for k in need if k not in d] for n, d in (("q", self.q), ("k", self.k), ("ncomp", self.ncomp), ("stage1", self.stage1),
                                                                 ("ccis", self.ccis), ("top33", self.top33), ("top64", self.top64))}
        return {n: len(v) for n, v in miss.items()}

    def stack(self, name, pad_value=float("nan")) -> Optional[torch.Tensor]:
        """(calls, layers_in_set, ...) with the last axis padded to the max width (stage-1 / CIS grow with n_comp); float
        tensors pad with pad_value, integer tensors with -1."""
        d = getattr(self, name)
        if not d:
            return None
        ex = next(iter(d.values()))
        width = max(v.shape[-1] for v in d.values())
        fill = pad_value if ex.is_floating_point() else -1
        out = torch.full((self.n_calls, len(self.layers)) + tuple(ex.shape[:-1]) + (width,), fill, dtype=ex.dtype)
        for (c, l), v in d.items():
            out[c, self.slot[l], ..., :v.shape[-1]] = v.cpu()
        return out

    def scalars(self, name, fill=-1) -> np.ndarray:
        d = getattr(self, name)
        out = np.full((self.n_calls, len(self.layers)), fill, dtype=np.int64)
        for (c, l), v in d.items():
            out[c, self.slot[l]] = int(v)
        return out


# --------------------------------------------------------------------------------------------- the hooks (no model edit)
def install_hooks(nl, cache_cls, rec: "Recorder", get_trace, n_heads: int = 32, n_kv: int = 2):
    """Wrap, in THIS process only, the module globals NOSA's decode looks up at call time (nl = nosi.nosa_llama:
    apply_rope_with_cos_sin_cache_inplace, infllmv2_attn_stage1_fast) and two InfLLMv2Cache methods (update_compress_k_decode,
    update_cis). nosa_llama.py / cache_engine.py are not edited; the wrappers record only while the trace has a current decode
    call (prefill has none). Returns the uninstall function."""
    saved = dict(rope=nl.apply_rope_with_cos_sin_cache_inplace, stage1=nl.infllmv2_attn_stage1_fast,
                 ckd=cache_cls.update_compress_k_decode, cis=cache_cls.update_cis)

    def where():
        tr = get_trace()
        if tr is None or getattr(tr, "cur", None) is None:
            return None, None
        return tr.cur.index, tr.layer

    def rope(positions, query, key, head_size, cos_sin_cache, is_neox):
        saved["rope"](positions, query, key, head_size, cos_sin_cache, is_neox)
        c, l = where()
        if c is not None:
            rec.note_rope(c, l, positions, query, key, n_heads, n_kv, head_size)

    def stage1(*a, **k):
        s = saved["stage1"](*a, **k)
        c, l = where()
        if c is not None:
            rec.note_stage1(c, l, s)
        return s

    def ckd(self, key_states, layer_idx, *a, **k):
        out = saved["ckd"](self, key_states, layer_idx, *a, **k)
        c, _ = where()
        if c is not None:
            rec.note_compress(c, layer_idx, out[2], key_states is not None)
        return out

    def cis(self, cis_t, layer_idx, *a, **k):
        out = saved["cis"](self, cis_t, layer_idx, *a, **k)
        c, _ = where()
        if c is not None and cis_t.shape[-1] == 1:                       # the decode branch (N == 1, cache_engine.update_cis)
            rec.note_cis(c, layer_idx, out, out.shape[-1])
        return out

    nl.apply_rope_with_cos_sin_cache_inplace, nl.infllmv2_attn_stage1_fast = rope, stage1
    cache_cls.update_compress_k_decode, cache_cls.update_cis = ckd, cis

    def uninstall():
        nl.apply_rope_with_cos_sin_cache_inplace, nl.infllmv2_attn_stage1_fast = saved["rope"], saved["stage1"]
        cache_cls.update_compress_k_decode, cache_cls.update_cis = saved["ckd"], saved["cis"]
    return uninstall


def make_treecap_trace(Base, rec: "Recorder", model):
    """transfer_trace.TransferTrace in mode 'scores' + the QK top-33 and final top-64 id buffers, copied at record_scores
    (right after after_pooling_graph.replay(), nosa_llama.py decode_forward)."""

    class TreecapTrace(Base):
        def record_scores(self, qk_scores, cis_scores):
            super().record_scores(qk_scores, cis_scores)
            if self.cur is not None and self.layer is not None:
                rec.note_select(self.cur.index, self.layer, model.topk_idx_buf_q, model.topk_idx_buf)

    return TreecapTrace


# --------------------------------------------------------------------------------------------- checks
def check_pooling(stage1: torch.Tensor, ccis: torch.Tensor, ncomp: np.ndarray, pooled_qk: torch.Tensor, pooled_cis: torch.Tensor, top33: torch.Tensor,
                  cfg: Dict = NOSA_CONFIG) -> Dict:
    """Spec tests 1 and 2 on the capture: pool_offline(stage-1) == the recorded pooled QK at every NON-forced block, bit for bit;
    pool_offline(CIS) == the recorded pooled CIS at every non-forced, non-QK-top-33 block. Inputs per (call, layer slot):
    stage1 / ccis (C, Ls, 2, B, W), ncomp (C, Ls), pooled (C, Ls, 2, B, M), top33 ids (C, Ls, 2, B, 33)."""
    C, Ls = stage1.shape[:2]
    M = pooled_qk.shape[-1]
    forced = forced_blocks(M, cfg["init_blocks"], cfg["local_forced_blocks_in_kernel"])
    rows_qk = rows_cis = ok_qk = ok_cis = 0
    for c in range(C):
        for s in range(Ls):
            n = int(ncomp[c, s])
            pq = pool_offline(stage1[c, s], n, M, cfg["init_blocks"], cfg["local_forced_blocks_in_kernel"], cfg["pooling_set_size"], cfg["pooling_padding"])
            pc = pool_offline(ccis[c, s], n, M, cfg["init_blocks"], cfg["local_forced_blocks_in_kernel"], cfg["pooling_set_size"], cfg["pooling_padding"])
            rq = pooled_qk[c, s].float()
            rc = pooled_cis[c, s].float()
            in33 = ids_to_mask(top33[c, s], M)
            keep_q = ~forced.expand_as(rq)
            keep_c = keep_q & ~in33
            eq_q = ((pq.to(pooled_qk.dtype).float() == rq) | ~keep_q).all(dim=-1)
            eq_c = ((pc.to(pooled_cis.dtype).float() == rc) | ~keep_c).all(dim=-1)
            rows_qk += eq_q.numel()
            rows_cis += eq_c.numel()
            ok_qk += int(eq_q.sum())
            ok_cis += int(eq_c.sum())
    return dict(pool_qk_rows_equal=ok_qk, pool_qk_rows=rows_qk, pool_cis_rows_equal=ok_cis, pool_cis_rows=rows_cis,
                ok=bool(ok_qk == rows_qk and ok_cis == rows_cis))


def check_selection(pooled_qk: torch.Tensor, pooled_cis_forced: torch.Tensor, block_map_ids: torch.Tensor, cfg: Dict = NOSA_CONFIG) -> Dict:
    """select_offline on the recorded pooled buffers == the recorded final selection (block_map as a set), per row."""
    M = pooled_qk.shape[-1]
    _, m64 = select_offline(pooled_qk, pooled_cis_forced, cfg["qk_select"], cfg["topk_blocks"])
    sel = ids_to_mask(block_map_ids, M)
    eq = (m64 == sel).all(dim=-1)
    return dict(rows_equal=int(eq.sum()), rows=int(eq.numel()), ok=bool(eq.all()))


def check_compressed(k_host: torch.Tensor, comp_final: torch.Tensor, n_check: int, kernel: int = 32, stride: int = 16) -> Dict:
    """The compressed-key PREFIX equals compress_k(host post-RoPE keys) to bf16 tolerance (the kernel's mean is fp32-ish)."""
    n = min(int(n_check), int(comp_final.shape[1]))
    ref = compress_k_ref(k_host, n, kernel, stride)
    got = comp_final[:, :n].float()
    err = (got - ref).abs()
    tol = 2 ** -7 * ref.abs().clamp(min=1e-2)                     # bf16 has 8 significant bits
    return dict(n=n, max_abs_err=float(err.max()) if err.numel() else 0.0, frac_within_bf16=float((err <= tol).float().mean()) if err.numel() else 1.0,
                ok=bool(err.numel() == 0 or float((err <= tol).float().mean()) >= 0.999))


# --------------------------------------------------------------------------------------------- assemble
def assemble(out_dir: str, arrays: Dict[str, object], meta: Dict, missing: Sequence[str], optional: Sequence[str] = (),
             size_limit: int = SIZE_LIMIT_BYTES, stem: str = "treecap_vectors_L16128") -> Dict:
    """NPZ (bf16 tensors as uint16 bit patterns) + JSON (shapes, dtypes, per-array sha256, meta, MISSING) + SHA256SUMS. Arrays in
    `optional` are dropped (largest first) until the uncompressed total fits size_limit; every drop is recorded as MISSING."""
    np_arrays, dtypes = {}, {}
    for k, v in arrays.items():
        if v is None:
            continue
        if torch.is_tensor(v):
            if v.dtype == torch.bfloat16:
                np_arrays[k], dtypes[k] = bf16_bits(v), "bfloat16 stored as uint16 bits; f32 = (u16.astype(uint32) << 16).view(float32)"
            else:
                np_arrays[k], dtypes[k] = v.detach().cpu().contiguous().numpy(), str(v.dtype).replace("torch.", "")
        else:
            a = np.asarray(v)
            np_arrays[k], dtypes[k] = a, str(a.dtype)
    dropped = []
    total = lambda: sum(a.nbytes for a in np_arrays.values())
    for k in sorted([k for k in optional if k in np_arrays], key=lambda k: -np_arrays[k].nbytes):
        if total() <= size_limit:
            break
        dropped.append(dict(name=k, bytes=int(np_arrays[k].nbytes)))
        del np_arrays[k], dtypes[k]
    if total() > size_limit:
        raise ValueError("treecap export %d bytes > the %d byte limit after dropping the optional arrays" % (total(), size_limit))
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, stem + ".npz")
    if os.path.exists(npz):
        raise FileExistsError(npz)
    np.savez(npz, **np_arrays)
    info = {k: dict(shape=list(a.shape), dtype=dtypes[k], sha256=array_sha(a), bytes=int(a.nbytes)) for k, a in np_arrays.items()}
    miss = list(missing) + ["%s (dropped to stay under %d MB: %d bytes)" % (d["name"], size_limit // 10 ** 6, d["bytes"]) for d in dropped]
    doc = dict(meta, arrays=info, missing_fields=miss, dropped_for_size=dropped, total_uncompressed_bytes=int(total()),
               npz=os.path.basename(npz), npz_sha256=sha256_file(npz), npz_bytes=os.path.getsize(npz), size_limit_bytes=size_limit)
    js = os.path.join(out_dir, stem + ".json")
    with open(js, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    sums = os.path.join(out_dir, "SHA256SUMS")
    with open(sums, "a") as f:
        for p in (npz, js):
            f.write("%s  %s\n" % (sha256_file(p), os.path.basename(p)))
    return dict(npz=npz, json=js, sums=sums, bytes=os.path.getsize(npz), dropped=dropped, arrays=sorted(np_arrays))


def fixture(arrays: Dict[str, np.ndarray], call_slice=slice(0, 2), layer_slot: int = 0, request: int = 0) -> Dict[str, np.ndarray]:
    """A small correctness fixture: the first calls of one layer slot / request of the per-call arrays (axes [call, layer slot,
    ...]); the test re-runs pool_offline / select_offline on it."""
    out = {}
    for k in ("stage1_score_f32", "compressed_cis_f32", "n_comp", "qk_pooled_bf16", "cis_pooled_bf16", "top33_ids", "top64_ids", "block_map"):
        if k in arrays:
            out[k] = np.ascontiguousarray(np.asarray(arrays[k])[call_slice, layer_slot])
    return out
