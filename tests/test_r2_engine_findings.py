"""CPU reproducers for docs/R2_ENGINE_FINDINGS.md (F1-F4), NOSI pin 6e6f80a.

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2 python -m pytest -q tests/test_r2_engine_findings.py

NOSI_REPO defaults to the tree that holds this file. No engine module is
imported: nosi/nosi/__init__.py imports nosa_llama (CUDA), and cache_engine.py
compiles a CUDA extension at import (cache_engine.py:19-23). The tests read
the source as text (test_citations_hold) and mirror the cited lines in Python.
A mirror is evidence about the LOGIC of the cited lines, not about the compiled
kernels; the document says which parts only a GPU run can confirm.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
N = NOSI_REPO / "nosi/nosi"
FCE = N / "flash_cache_engine"
IV2 = NOSI_REPO / "dependencies/infllmv2_cuda_impl"

if not (N / "cache_engine.py").is_file():
    pytest.skip("no nosi/nosi/cache_engine.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)

BLOCK = 64   # nosa_llama.py:680
TOPK = 64    # nosa_llama.py:681; the tail slot is TOPK - 1 (cache_engine.py:312)


# ---------------------------------------------------------------------------
# every file:line the document relies on
# ---------------------------------------------------------------------------
_CE = N / "cache_engine.py"
_CEG = N / "cache_engine_gpu.py"
_NL = N / "nosa_llama.py"
_MP = N / "max_pooling_fused.py"
_MB = FCE / "flash_h2d_mask_bias.py"
_MK = FCE / "flash_h2d_mask.py"
_DK = FCE / "diff_offload_kernel.cu"

CITATIONS = [
    (_CE, 132, "self.max_gen_len = 8192"),
    (_CE, 195, "S+self.max_gen_len"),
    (_CE, 196, "S+self.max_gen_len"),
    (_CE, 309, "self._tail_block_len_on_gpu = S % self.block_size"),
    (_CE, 312, "self._tail_block_idx_on_gpu = self.topk - 1"),
    (_CE, 321, "= S // self.block_size"),
    (_CE, 324, "= S // self.block_size"),
    (_CE, 327, "(self.topk - 1) * self.block_size + self._tail_block_len_on_gpu"),
    (_CE, 346, "self._cache_lens[...] += 1"),
    (_CE, 349, "_tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu"),
    (_CE, 354, "kv_bias[:, self.seq_length-1:self.seq_length, :]"),
    (_CE, 357, "self._tail_block_len_on_gpu += 1"),
    (_CE, 359, "tail_full = self._tail_block_len_on_gpu == self.block_size"),
    (_CE, 363, "diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)"),
    (_CE, 377, "self._block_map.copy_(self._new_block_map_buf"),
    (_CE, 381, "flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask"),
    (_CE, 382, "flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask"),
    (_CE, 401, "if tail_full:"),
    (_CE, 404, "self._k_cpu[:, cpu_block_base_pos"),
    (_CE, 405, "self._v_cpu[:, cpu_block_base_pos"),
    (_CE, 407, "self._block_map[..., self._tail_block_idx_on_gpu] += 1"),
    (_CE, 409, "self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu"),
    (_CE, 412, "return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens"),
    (_CE, 702, "return self.cache_engine.decode_update("),
    (_CE, 725, "F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride)"),
    (_CE, 727, "self.comp_cis_len = M"),
    (_CE, 728, "self.compressed_cis =  torch.empty((H, total_bsz, M)"),
    (_CE, 731, "self.tail_cis_len = kernel_stride + tail_block_len"),
    (_CE, 752, "self.tail_cis[:, :, self.tail_cis_len:self.tail_cis_len+1].copy_(cis"),
    (_CE, 754, "if self.tail_cis_len == kernel_size:"),
    (_CE, 757, "tail_comp = tail.mean(dim=-1, keepdim=True)"),
    (_CE, 758, "# self.compressed_cis = torch.cat((self.compressed_cis, tail_comp), dim=-1)"),
    (_CE, 759, "self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len+1].copy_(tail_comp"),
    (_CE, 763, "return self.compressed_cis"),
    (_CE, 768, "self.total_cis = torch.empty((total_bsz, S+1024, H)"),
    (_CE, 780, "self.total_cis[:, self.cis_len:self.cis_len+1, :].copy_(cis"),
    (_CE, 781, "self.cis_len += 1"),
    (_CE, 782, "return self.total_cis"),
    (_CEG, 183, "self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu"),
    (_CEG, 233, "self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu"),
    (_CEG, 370, "self.comp_cis_len = M"),
    (_CEG, 402, "self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len+1].copy_(tail_comp"),
    (_CEG, 411, "self.total_cis = torch.empty((total_bsz, S+1024, H)"),
    (_NL, 459, "self.after_pooling_graph = torch.cuda.CUDAGraph()"),
    (_NL, 465, "max_seqlen_comp,"),
    (_NL, 489, "out=(topk_val_buf, topk_idx_buf)"),
    (_NL, 577, "new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)"),
    (_NL, 584, "ucis = cache_engine.update_uncompressed_cis(cis, self.layer_idx, 0, bsz)"),
    (_NL, 607, "self.score_buf.copy_(score)"),
    (_NL, 612, "self.after_pooling_graph.replay()"),
    (_NL, 618, "decode_update_kv(key_states, value_states, ucis, self.layer_idx, topk_idx)"),
    (_NL, 639, "cache_seqlens=_cache_lens,"),
    (_NL, 674, "init_blocks = 1,"),
    (_NL, 675, "window_blocks = 16,"),
    (_NL, 677, "pooling_stride = 16,"),
    (_NL, 678, "pooling_block_size = 32"),
    (_NL, 680, "assert block_size == 64"),
    (_NL, 681, "assert topk_blocks == 64"),
    (_NL, 800, "warmup = not self.has_buffers"),
    (_NL, 817, "total_len = cache_engine.get_seq_length(0) + 1"),
    (_NL, 818, "out_len = (total_len + self.block_size - 1) // self.block_size"),
    (_NL, 819, "self.pooling_buf_all = torch.empty(2 * self.num_key_value_heads, bsz, out_len"),
    (_NL, 828, "self.mask_buf = torch.empty(self.num_key_value_heads, bsz, out_len"),
    (N / "nosa_linear.py", 105, "cis = torch.empty((B, S, kv_heads)"),
    (N / "nosa_linear.py", 128, "cis = cis.view(B, S, kv_heads)"),
    (_MP, 47, "mask = (block_idx + n_offset < max_seq_len)"),
    (_MP, 61, "boundary_mask = (tidx_m < init_blocks) | (M - tidx_m <= local_blocks)"),
    (_MP, 83, "M = q_score_pooled.shape[-1]"),
    (_MP, 86, "grid = (H, B, M)"),
    (_MP, 105, "init_blocks, local_blocks+1,"),
    (_DK, 38, "if (old_val == new_act[base + j])"),
    (_DK, 53, "if (new_val == old_map[base + j])"),
    (_DK, 82, "if (sm_old_hit[j] == 0)"),
    (_DK, 96, "new_map[write_idx] = new_val_m;"),
    (_DK, 97, "load_map[write_idx] = new_val_m;"),
    (_MK, 61, "vals = tl.load(src_ptrs, mask=mask_src)"),
    (_MK, 62, "tl.store(dst_ptrs, vals, mask=mask_dst)"),
    (_MB, 43, "mask_src = (cpu_block_id * block_size + offset_s) < S_CPU"),
    (_MB, 58, "bias_vals = tl.load(bias_src_ptrs, mask=mask_src)"),
    (_MB, 93, "_, S_CPU, _, _ = cpu_data.shape"),
    (_MB, 100, "stride_b_bias, stride_m_bias, stride_h_bias = kv_bias.stride()"),
    (NOSI_REPO / "benchmarks/Efficiency/test_nosa_pg19.py", 21, "model = Llama("),
    (NOSI_REPO / "benchmarks/Efficiency/test_nosa_pg19.py", 32, '"NOSI_BENCH_NEW_TOKENS", 4'),
    (IV2 / "csrc/flash_attn/flash_api.cpp", 785, "mha_varlen_fwd_stage1("),
    (IV2 / "csrc/flash_attn/flash_api.cpp", 929, "const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);"),
    (IV2 / "csrc/flash_attn/flash_api.cpp", 936, "seqlen_k_rounded }"),
    (IV2 / "csrc/flash_attn/flash_api.cpp", 1043, "mha_bwd("),
    (IV2 / "infllm_v2/infllmv2_sparse_attention.py", 625, "def infllmv2_attn_stage1_fast("),
    (IV2 / "infllm_v2/infllmv2_sparse_attention.py", 729, "return result[0]"),
]


@pytest.mark.parametrize("path,lineno,text", CITATIONS,
                         ids=["%s:%d" % (p.name, n) for p, n, _ in CITATIONS])
def test_citations_hold(path, lineno, text):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lineno <= len(lines), "%s has %d lines, cited %d" % (path, len(lines), lineno)
    assert text in lines[lineno - 1], "%s:%d is %r, expected to contain %r" % (path, lineno, lines[lineno - 1], text)


# ---------------------------------------------------------------------------
# mirrors of the shipped tail bookkeeping and of the warm-up grid
# ---------------------------------------------------------------------------
def _out_len(S):
    total_len = S + 1                                   # nosa_llama.py:817
    return (total_len + BLOCK - 1) // BLOCK             # nosa_llama.py:818


def _tail_by_step(S, steps):
    """Per decode step i = 1..steps: (tail id at the START of step i, the
    _cache_lens returned to attention at step i, the row of slot 63 that step i
    writes). Mirrors cache_engine.py:309-327 (prefill) and :345-359, :401-409."""
    tail_len = S % BLOCK                                # :309
    tail_id = S // BLOCK                                # :321 / :324
    cache_lens = (TOPK - 1) * BLOCK + tail_len          # :327
    out = []
    for _ in range(steps):
        start_id = tail_id
        cache_lens += 1                                 # :346
        row = tail_len                                  # :349
        tail_len += 1                                   # :357
        if tail_len == BLOCK:                           # :359, :401
            tail_id += 1                                # :407
            tail_len = 0
            cache_lens = (TOPK - 1) * BLOCK + tail_len  # :409
        out.append((start_id, cache_lens, row))         # :412 returns this _cache_lens
    return out


def _diff_mirror(old_map, new_act):
    """Python mirror of diff_kernel (diff_offload_kernel.cu:10-99) for ONE
    (h, b) row of M slots. Entries the kernel does not write keep -2."""
    M = len(old_map)
    new_map = [-2] * M
    load_map = [-2] * M
    sm_old_hit = [0] * M                                # :29
    sm_new_hit = [0] * M                                # :30
    for m in range(M):                                  # phase 1, one thread per slot
        old_val, new_val = old_map[m], new_act[m]       # :32-33
        if any(old_val == new_act[j] for j in range(M)):   # :36-42
            sm_old_hit[m] = 1                           # :45
            new_map[m] = old_val                        # :46
            load_map[m] = -1                            # :47
        if any(new_val == old_map[j] for j in range(M)):   # :51-57
            sm_new_hit[m] = 1                           # :60
    # __syncthreads() :63
    for m in range(M):                                  # phase 2
        if sm_new_hit[m] != 0:
            continue
        k = sum(1 for j in range(m) if sm_new_hit[j] == 0)   # :66-75
        cnt, target = 0, -1
        for j in range(M):                              # :78-90
            if sm_old_hit[j] == 0:
                if cnt == k:
                    target = j
                    break
                cnt += 1
        if target >= 0:                                 # :92-98
            new_map[target] = new_act[m]
            load_map[target] = new_act[m]
    return new_map, load_map


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [4032, 4033, 4047, 4095, 4096, 16128, 16129, 16191, 65280])
def test_f1_window_arithmetic(S):
    out_len = _out_len(S)
    assert out_len - 1 == S // BLOCK, "the largest id the warm-up grid can emit"
    rows = _tail_by_step(S, 200)
    first_bad = next(i for i, (tid, _, _) in enumerate(rows, 1) if tid >= out_len)
    assert first_bad == 65 - S % 64
    assert all(tid < out_len for tid, _, _ in rows[:first_bad - 1])
    assert rows[first_bad - 1][0] == out_len


def _maps_around_rename(S):
    out_len = _out_len(S)
    tail_prev = out_len - 1                 # the block the rename step completed
    resident = list(range(tail_prev - 63, tail_prev))   # 63 resident ids (S >= 4032 so these are >= 0)
    old_pre = resident + [tail_prev]        # a valid step: slot 63 holds the tail id
    old_post = resident + [out_len]         # after :407 renamed slot 63
    new_act = [tail_prev] + resident        # topk is unsorted (:489, sorted=False); order is free
    return out_len, old_pre, old_post, new_act


@pytest.mark.parametrize("S", [4032, 16128, 16129, 65280])
def test_f1_diff_mirror_frees_the_live_tail(S):
    out_len, old_pre, old_post, new_act = _maps_around_rename(S)
    # control: a valid step -- every selected block is resident, nothing loads
    nm, lm = _diff_mirror(old_pre, new_act)
    assert lm == [-1] * TOPK
    assert nm == old_pre
    # the first bad step: slot 63 counts as free and receives a load
    nm, lm = _diff_mirror(old_post, new_act)
    assert lm[TOPK - 1] == out_len - 1, "the gather would overwrite the live tail slot"
    assert nm[TOPK - 1] == out_len - 1, "the map no longer names the tail block"
    assert lm[:TOPK - 1] == [-1] * (TOPK - 1)


def test_f1_diff_mirror_random_selections():
    g = torch.Generator().manual_seed(0)
    S = 16128
    out_len = _out_len(S)
    tail_prev = out_len - 1
    forced = set(range(out_len - 17, out_len)) | {0}    # max_pooling_fused.py:61, local_blocks = 16 + 1 (:105), init 1
    for _ in range(200):
        old = torch.randperm(tail_prev, generator=g)[:TOPK - 1].tolist() + [out_len]
        rest = [x for x in torch.randperm(out_len, generator=g).tolist() if x not in forced]
        sel = sorted(forced) + rest[:TOPK - len(forced)]
        order = torch.randperm(TOPK, generator=g).tolist()
        new_act = [sel[o] for o in order]
        assert len(set(new_act)) == TOPK and max(new_act) < out_len
        nm, lm = _diff_mirror(old, new_act)
        assert -2 not in nm and -2 not in lm, "every slot is written"
        assert sorted(nm) == sorted(new_act)
        assert lm[TOPK - 1] >= 0 and nm[TOPK - 1] != out_len


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------
def test_f2_empty_slice_copy_is_a_silent_noop():
    H, B, S = 2, 4, 16128
    full = (S // 16) * 16                                       # cache_engine.py:722, :724
    M = F.avg_pool1d(torch.zeros(H * B, 1, full), kernel_size=32, stride=16).shape[-1]   # :725-726
    assert M == 1007
    comp_cis_len = M                                            # :727
    g = torch.Generator().manual_seed(2)
    compressed_cis = torch.randn(H, B, M, generator=g)          # :728-729 after prefill
    before = compressed_cis.clone()
    tail_comp = torch.randn(H, B, 32, generator=g).mean(dim=-1, keepdim=True)   # :757
    dst = compressed_cis[:, :, comp_cis_len:comp_cis_len + 1]
    assert dst.shape == (H, B, 0)
    dst.copy_(tail_comp, non_blocking=True)                     # :759 -- no error
    assert compressed_cis.shape == (H, B, M)
    assert torch.equal(compressed_cis, before)
    grown = torch.cat((compressed_cis, tail_comp), dim=-1)      # the intent, :758
    assert grown.shape == (H, B, M + 1) and torch.equal(grown[..., M:], tail_comp)


def test_f2_mirror_decode_never_grows():
    H, B, S, ks, st = 2, 3, 16128, 32, 16
    g = torch.Generator().manual_seed(3)
    cis = torch.randn(H, B, S, generator=g)                     # (H, B, N) as update_cis receives it
    full_block_len, tail_block_len = S // st, S % st            # :722-723
    full_cis = cis[:, :, :full_block_len * st].reshape(H * B, 1, full_block_len * st)   # :724
    comp_cis = F.avg_pool1d(full_cis, kernel_size=ks, stride=st).reshape(H, B, -1)       # :725
    M = comp_cis.shape[-1]                                      # :726
    comp_cis_len = M                                            # :727
    compressed_cis = torch.empty((H, B, M))                     # :728
    compressed_cis[:, :B, :].copy_(comp_cis)                    # :729
    tail_cis = torch.empty((H, B, ks))                          # :730
    tail_cis_len = st + tail_block_len                          # :731
    tail_cis[:, :B, :tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * st:])          # :732
    snapshot = compressed_cis.clone()
    computed = 0
    for _ in range(64):
        c = torch.randn(H, B, 1, generator=g)
        tail_cis[:, :, tail_cis_len:tail_cis_len + 1].copy_(c)  # :752
        tail_cis_len += 1                                       # :753
        if tail_cis_len == ks:                                  # :754
            tail_comp = tail_cis.mean(dim=-1, keepdim=True)     # :757
            compressed_cis[:, :, comp_cis_len:comp_cis_len + 1].copy_(tail_comp)       # :759
            tail_cis[:, :, :st].copy_(tail_cis[:, :, st:])      # :761
            tail_cis_len = st                                   # :762
            computed += 1
    assert computed == 4, "steps 16, 32, 48, 64 for S % 16 == 0"
    assert compressed_cis.shape[-1] == M
    assert torch.equal(compressed_cis, snapshot)


def test_f2_comp_cis_len_is_assigned_once():
    tree = ast.parse(_CE.read_text(encoding="utf-8"))
    assigns, augs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "comp_cis_len":
                    assigns.append(node.lineno)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            t = node.target
            if isinstance(t, ast.Attribute) and t.attr == "comp_cis_len":
                augs.append(node.lineno)
    assert assigns == [727]
    assert augs == []


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------
def test_f3_total_cis_write_is_a_silent_noop():
    B, S, H = 2, 64, 2
    total_cis = torch.zeros((B, S + 1024, H))                   # cache_engine.py:768
    before = total_cis.clone()
    cis_len = S + 1024                                          # after 1024 decode writes (:780-781)
    cis = torch.ones(B, 1, H)                                   # decode cis (nosa_linear.py:105, :128)
    dst = total_cis[:, cis_len:cis_len + 1, :]
    assert dst.shape == (B, 0, H)
    dst.copy_(cis, non_blocking=True)                           # :780 -- no error
    assert torch.equal(total_cis, before)


def test_f3_tail_bias_write_raises():
    B, S, H = 2, 64, 2
    total_cis = torch.zeros((B, S + 1024, H))                   # :768
    kv_bias_gpu = torch.zeros((B, TOPK * BLOCK, H))             # :224 with the pool off
    p = S + 1024                                                # step 1025
    seq_length = p + 1                                          # :345
    tail_write_pos = (TOPK - 1) * BLOCK + p % BLOCK             # :349
    src = total_cis[:, seq_length - 1:seq_length, :]            # :354
    assert src.shape == (B, 0, H)
    with pytest.raises(RuntimeError):
        kv_bias_gpu[:, tail_write_pos:tail_write_pos + 1, :].copy_(src, non_blocking=True)
    # control: one step earlier the same write works
    ok = total_cis[:, p - 1:p, :]
    assert ok.shape == (B, 1, H)
    kv_bias_gpu[:, tail_write_pos:tail_write_pos + 1, :].copy_(ok, non_blocking=True)


def test_f3_bias_rows_alias_the_next_request():
    B, S, H = 3, 64, 2
    rows_cis = S + 1024                                         # :768
    S_CPU = S + 8192                                            # flash_h2d_mask_bias.py:93 with cache_engine.py:132, :195-196
    total_cis = torch.arange(B * rows_cis * H, dtype=torch.float32).view(B, rows_cis, H)  # value == flat offset
    sb, sm, sh = total_cis.stride()                             # flash_h2d_mask_bias.py:100
    c = rows_cis // BLOCK + 1                                   # a block with every row in [S + 1024, S_CPU)
    rows = [c * BLOCK + s for s in range(BLOCK)]
    assert all(rows_cis <= r < S_CPU for r in rows), "mask_src (:43) lets every row through"
    for b in range(B):
        for r in rows:
            for h in range(H):
                off = b * sb + r * sm + h * sh                  # :46-51
                if b + 1 < B:
                    rr = r - rows_cis
                    assert 0 <= rr < rows_cis
                    assert off == (b + 1) * sb + rr * sm + h * sh
                    assert total_cis[b + 1, rr, h].item() == off, "reads request b+1"
                else:
                    assert off >= total_cis.numel(), "reads past the allocation"


# ---------------------------------------------------------------------------
# F4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [4032, 4033, 4047, 4095, 16128])
def test_f4_cache_lens_drop_on_block_completion(S):
    for i, (_, cache_lens, row) in enumerate(_tail_by_step(S, 200), 1):
        p = S + i - 1
        assert row == p % BLOCK
        if p % BLOCK == BLOCK - 1:
            assert cache_lens == (TOPK - 1) * BLOCK, "slot 63 (rows 4032..4095) is not attended"
        else:
            assert cache_lens == (TOPK - 1) * BLOCK + p % BLOCK + 1, "the tail through the current token"
        assert cache_lens == (TOPK - 1) * BLOCK + (p + 1) % BLOCK, "the twin's formula"
