"""Minimal real-vector export for the frozen-tree layout test (retroinfer-eval fork, 2026-10-01).

Per 64-token group and KV head: the fp32 mean of the POST-RoPE prefill keys. They are read ONCE from the
host KV, cache_engine.layers[l].cache_engine._k_cpu (pinned bf16 [B, S + max_gen_len, H, D], written by
CacheEngine.prefill_update after apply_rope_with_cos_sin_cache_inplace), right after the synchronized
prefill and before any decode call. The export only reads: it writes no GPU or cache state and calls no
kernel, so the selection trace of the same run stays comparable bit for bit with earlier runs.

The group mean is a LAYOUT HEURISTIC. It is not NOSA's selector: NOSA scores 32-token compressed keys
(stride 16) pooled to 64-token blocks, plus CIS.

Two checks run in the job, on the real tensors:
  - spot: float64 re-computation of chosen (layer, request, head, group) cells straight from the host
    rows [64g, 64g + 64); negative control = the rows of group g + 1;
  - pair: NOSA's own compressed keys (compress_k_cache_varlen, chunk c = bf16 mean of tokens
    [16c, 16c + 32)) give group g as the mean of chunks 4g and 4g + 2, to bf16 rounding; negative
    control = the chunks of group g + 1.

Opt-in: NOSI_TREECAP_LAYERS="0,4,8,12,16,20,24,28". Unset or empty: nothing runs and nothing is written.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

ENV = "NOSI_TREECAP_LAYERS"
BLOCK = 64                  # NOSA block = group size
KERNEL, STRIDE = 32, 16     # compress_k (nosa_llama.py compress_k defaults)
PAIR_GATE = 2.0             # |pair - G| <= PAIR_GATE x the bf16 half-ulp bound of the two chunks
SPOT_RTOL = 1e-5


def layers_from_env(n_layers, env=None):
    s = (os.environ if env is None else env).get(ENV, "").strip()
    if not s:
        return None
    ls = [int(x) for x in s.split(",") if x.strip()]
    assert ls and len(set(ls)) == len(ls) and all(0 <= l < n_layers for l in ls), \
        f"{ENV}={s!r}: need distinct layer ids in [0, {n_layers})"
    return ls


def group_means(k_host, n_groups, block=BLOCK):
    """k_host [B, S, H, D] -> fp32 [B, H, n_groups, D]; group g = mean of tokens [block*g, block*g + block)."""
    B, S, H, D = k_host.shape
    assert n_groups * block <= S, (n_groups, block, S)
    x = k_host[:, :n_groups * block].to(torch.float32).reshape(B, n_groups, block, H, D)
    return x.mean(dim=2).permute(0, 2, 1, 3).contiguous()


def pair_chunks(n_groups, shift=0, block=BLOCK, kernel=KERNEL, stride=STRIDE):
    """Compressed-chunk ids whose two disjoint 32-token spans tile group g + shift: 4(g+shift), 4(g+shift)+2."""
    a = (torch.arange(n_groups) + shift) * (block // stride)
    return a, a + kernel // stride


def pair_check(ck, G, shift=0):
    """ck [B, Nc, H, D] bf16 compressed keys, G [B, H, n, D] fp32. Returns (max ratio to the bf16 bound,
    relative Frobenius error). Only groups whose shifted chunks exist are compared."""
    n = G.shape[2]
    a, b = pair_chunks(n, shift)
    ok = b < ck.shape[1]
    a, b, Gs = a[ok], b[ok], G[:, :, ok.nonzero().flatten()]
    ca, cb = ck[:, a].to(torch.float32), ck[:, b].to(torch.float32)
    pair = ((ca + cb) / 2).permute(0, 2, 1, 3)
    bound = (2.0 ** -8) * (ca.abs() + cb.abs()).permute(0, 2, 1, 3) / 2 + 1e-6
    err = (pair - Gs).abs()
    return float((err / bound).max()), float(err.norm() / Gs.norm())


def spot_cells(n_layers, B, H, n_groups):
    """Edge and interior groups for every (request, head), spread over the layers."""
    gs = sorted({0, 1, n_groups // 2, n_groups - 2, n_groups - 1})
    return [(i % n_layers, b, h, g) for i, (b, h, g) in
            enumerate((b, h, g) for b in range(B) for h in range(H) for g in gs)]


def spot_check(k_rows, G, cells, block=BLOCK):
    """k_rows[i] = host keys [B, S, H, D] of exported layer i. Returns (max rel err, neg-control min rel err)."""
    worst, neg = 0.0, float("inf")
    for i, b, h, g in cells:
        ref = k_rows[i][b, block * g:block * g + block, h].to(torch.float64).mean(0)
        got = G[i, b, h, g].to(torch.float64)
        worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
        g2 = g + 1 if g + 1 < G.shape[3] else g - 1
        other = k_rows[i][b, block * g2:block * g2 + block, h].to(torch.float64).mean(0)
        neg = min(neg, float((got - other).abs().max() / ref.abs().max()))
    return worst, neg


def _sha_tensor(t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def export(cache_engine, layers, L, docs, out, provenance=None, n_q_heads=32):
    """Write treecap_groupmean_L<L>.{npz,json} under out. Returns the check verdict dict. Read-only on cache_engine."""
    t0 = time.time()
    n_groups = L // BLOCK                       # groups complete at prefill; positions >= 64*n_groups are later
    rows, cks = [], []
    for l in layers:
        lay = cache_engine.layers[l]
        k = lay.cache_engine._k_cpu
        assert k.device.type == "cpu" and k.dtype == torch.bfloat16 and k.dim() == 4 and k.shape[1] >= L, \
            (l, k.device, k.dtype, tuple(k.shape))
        rows.append(k[:, :L])
        cks.append(lay.compress_k_cache_varlen.detach().to("cpu"))
    B, _, H, D = rows[0].shape
    G = torch.stack([group_means(r, n_groups) for r in rows])          # [n_layers, B, H, n_groups, D] fp32
    t_mean = time.time() - t0

    spot_err, spot_neg = spot_check(rows, G, spot_cells(len(layers), B, H, n_groups))
    pair = [pair_check(ck, G[i]) for i, ck in enumerate(cks)]
    pair_neg = [pair_check(ck, G[i], shift=1) for i, ck in enumerate(cks)]
    checks = dict(
        spot_max_rel_err=spot_err, spot_negative_min_rel_diff=spot_neg,
        pair_max_bound_ratio=max(p[0] for p in pair), pair_rel_err=[p[1] for p in pair],
        pair_negative_min_bound_ratio=min(p[0] for p in pair_neg), pair_negative_rel_err=[p[1] for p in pair_neg],
        compressed_chunks=[int(ck.shape[1]) for ck in cks],
    )
    checks["spot_ok"] = spot_err <= SPOT_RTOL and spot_neg > 100 * SPOT_RTOL
    checks["pair_ok"] = checks["pair_max_bound_ratio"] <= PAIR_GATE and checks["pair_negative_min_bound_ratio"] > 10 * PAIR_GATE
    checks["ok"] = bool(checks["spot_ok"] and checks["pair_ok"])

    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, f"treecap_groupmean_L{L}.npz")
    arrays = dict(
        group_mean_key=G.numpy(),
        layer_ids=np.asarray(layers, np.int16),
        request_ids=np.arange(B, dtype=np.int16),
        doc_index=np.asarray(docs, np.int32),
        kv_head_ids=np.arange(H, dtype=np.int16),
        group_ids=np.arange(n_groups, dtype=np.int16),
        group_start_token=(np.arange(n_groups, dtype=np.int32) * BLOCK),
        group_end_token=(np.arange(n_groups, dtype=np.int32) * BLOCK + BLOCK),
        q_head_to_kv_head=(np.arange(n_q_heads, dtype=np.int16) // (n_q_heads // H)),
    )
    np.savez(npz, **arrays)
    meta = dict(
        what="per-64-token-group, per-KV-head fp32 mean of the POST-RoPE prefill keys (NOSA-8B host KV)",
        label="LAYOUT HEURISTIC, not NOSA's selector (NOSA scores 32-token compressed keys, stride 16, pooled to 64-token blocks, plus CIS)",
        construction="group_mean_key[i, b, h, g, :] = mean over tokens t in [64g, 64g+64) of float32(K_bf16[layer_ids[i]][b, t, h, :]), "
                     "torch.float32 mean on the CPU; K = cache_engine.layers[l].cache_engine._k_cpu, read once after the synchronized "
                     "prefill and before decode call 0; RoPE positions = token index (prefill position_ids 0..L-1)",
        axes=["layer", "request", "kv_head", "group", "dim"],
        shape=list(G.shape), dtype="float32", nbytes=int(G.numpy().nbytes),
        context=L, groups_complete_at_prefill=n_groups, block=BLOCK,
        causal_note=f"positions >= {BLOCK * n_groups} arrive during decode; with L={L} and the archive's 64 calls they all fall in "
                    f"block {n_groups}, the tail block, so groups 0..{n_groups - 1} are frozen before any decode call",
        layer_ids=list(layers), doc_index=list(docs), request_ids=list(range(B)), kv_head_ids=list(range(H)),
        q_head_to_kv_head="q // %d" % (n_q_heads // H),
        raw_key_sha256={str(l): _sha_tensor(r) for l, r in zip(layers, rows)},
        raw_key_layout="bf16 [B, L, H, D] (the host slab _k_cpu[:, :L], made contiguous; int16 view of the bf16 bits)",
        checks=checks,
        seconds=dict(group_mean=t_mean, total=time.time() - t0),
        missing=["raw per-token keys (only their sha256 per layer)", "values / kv_bias", "per-call queries",
                 "unpooled stage-1 scores", "compressed keys and CIS (read for the check only, not exported)",
                 "layers other than layer_ids", "groups completed during decode (none for this scope)"],
        provenance=dict(provenance or {}),
    )
    meta["npz_sha256"] = _sha_file(npz)
    meta["npz_bytes"] = os.path.getsize(npz)
    with open(os.path.join(out, f"treecap_groupmean_L{L}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[treecap] layers {layers} G {tuple(G.shape)} fp32 {meta['nbytes']} B  npz {meta['npz_bytes']} B  "
          f"spot {spot_err:.2e} (neg {spot_neg:.2e})  pair ratio {checks['pair_max_bound_ratio']:.3f} "
          f"(neg {checks['pair_negative_min_bound_ratio']:.1f})  ok={checks['ok']}  {meta['seconds']['total']:.1f}s")
    return checks
