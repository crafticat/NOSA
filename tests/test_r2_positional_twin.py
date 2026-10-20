"""Gates for the R2 CPU twin of the positional tables (nosi/nosi/r2_positional_twin.py).

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2 python -m pytest -q tests/test_r2_positional_twin.py

NOSI_REPO defaults to the tree that holds this file. The twin is loaded by file
path (nosi/nosi/__init__.py imports CUDA modules).

GATE 1 (test_gate1_*): the twin against ShippedRef, a direct re-implementation
of the SHIPPED tables in the shipped update order (each line cites
cache_engine.py / nosa_llama.py), prefill residues S % 16 in {0, 1, 15},
B = 1 and B = 4, 300 decode tokens with equal advance. torch.equal on the live
prefix of every table after prefill and after every step. ShippedRef deviates
from the shipped code in ONE place, F2 (the cis summary is appended as the
commented-out cat at cache_engine.py:758 intends); test_gate1_detects_shipped_f2
shows that the shipped line :759 fails the gate.

GATE 2 (test_gate2_*): a ragged run (requests advance different amounts, with
rollbacks) against B independent B = 1 twins. One set follows the same
schedule and is compared after every step; one set gets only the committed
tokens, never rolls back, and is compared after every commit. torch.equal per
request.

NEGATIVE CONTROLS: an off-by-one key or cis compression event fails gate 1; a
shared counter fails gate 2 and passes gate 1 (so gate 2 is the gate that sees
it).

DATA. The gates use integer-valued float32 data in [-8, 8] (rejected draft
tokens in [100, 120]). Every window sum is exact, so torch.equal does not
depend on the reduction order: a mismatch is a bookkeeping error.
test_gate1_bf16 repeats one gate-1 cell with bf16 normals. There the twin and
ShippedRef reduce tensors of the same shape and layout, so they must agree bit
for bit as well.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
TWIN = NOSI_REPO / "nosi/nosi/r2_positional_twin.py"

if not TWIN.is_file():
    pytest.skip("no nosi/nosi/r2_positional_twin.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tw = _load(TWIN, "nosi_r2_positional_twin_under_test")
STRIDE, KERNEL, BLOCK = tw.STRIDE, tw.KERNEL, tw.BLOCK
H_, D_ = 2, 8


def _ints(shape, g, lo=-8, hi=8, dtype=torch.float32):
    return torch.randint(lo, hi + 1, tuple(shape), generator=g).to(dtype)


def _normals(shape, g, dtype=torch.bfloat16):
    return torch.randn(tuple(shape), generator=g).to(dtype)


def _eq(a, b, where, what):
    assert a.shape == b.shape, "%s: %s shape %s != %s" % (where, what, tuple(a.shape), tuple(b.shape))
    assert torch.equal(a, b), "%s: %s differs (max |diff| %s)" % (
        where, what, (a.float() - b.float()).abs().max().item())


# ---------------------------------------------------------------------------
# ShippedRef: the shipped tables of ONE layer, in the shipped update order
# ---------------------------------------------------------------------------
def _calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """nosa_llama.py:51-95 (the arange end is taken as an int; same values)."""
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, int(max_num_chunks_per_seq) * kernel_stride, kernel_stride)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    chunk_indices = torch.arange(0, chunk_size)[None, :]
    filtered_indices = (valid_chunk_starts[:, None] + chunk_indices).view(-1)
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32)
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    return filtered_indices, cu_seqlens_compressed


def _compress_k(k, cu_seqlens, kernel_size=32, kernel_stride=16):
    """nosa_llama.py:97-109."""
    idx, cu_c = _calc_chunks_with_stride(cu_seqlens, kernel_size, kernel_stride)
    filtered_k = k.index_select(0, idx.view(-1))
    filtered_k = filtered_k.view(filtered_k.shape[0] // kernel_size, kernel_size, k.shape[1], k.shape[2])
    return filtered_k.mean(dim=1), cu_c


class ShippedRef:
    """The shipped tables of one layer for a LOCKSTEP batch (every request has
    the same S and advances together; max_pooling_fused.py:78). Attribute
    names follow the shipped ones: ck = compress_k_cache_varlen, ck_cu =
    cached_compressed_cu_seqlens, ck_max = cached_compressed_max_seqlen,
    nck = no_compress_k_cache. One deliberate change, F2: unless
    f2_as_shipped, the decode cis summary is appended (cache_engine.py:758,
    the commented-out intent) instead of written into an empty slice (:759)."""

    TOPK, BLOCK, KS, ST = 64, 64, 32, 16

    def __init__(self, keys, cis, f2_as_shipped=False):
        B, S, H, D = keys.shape
        self.B, self.S, self.H, self.D = B, S, H, D
        self.f2_as_shipped = f2_as_shipped
        self.total_cis = None
        self.compressed_cis = None
        # CacheEngine.prefill_update, tail bookkeeping (cache_engine.py:178-333)
        self.seq_length = S                                                  # :189
        self.tail_len = S % self.BLOCK                                       # :309
        self.tail_block = S // self.BLOCK                                    # :321 / :324
        self.cache_lens = torch.full((B,), (self.TOPK - 1) * self.BLOCK + self.tail_len, dtype=torch.int32)   # :327
        # LlamaLayer.prefill_forward, one request at a time (nosa_llama.py:771-776, T = 1)
        for t in range(B):
            cu = torch.tensor([0, S], dtype=torch.int32)                     # :774
            ck, cu_comp = _compress_k(keys[t], cu)                           # :211 (key_states (q_len, H, D), :207)
            max_seqlen_comp = int((cu_comp[1:] - cu_comp[:-1]).max().item())  # :213
            self._update_uncompressed_cis(cis[t:t + 1], t)                   # :216
            self._update_cis_prefill(cis[t:t + 1].permute(2, 0, 1), t)       # :217
            self._update_compress_k_prefill(ck, cu_comp, max_seqlen_comp, t)  # :229
            self._update_no_compress_k_prefill(keys[t:t + 1, max_seqlen_comp * self.ST:], t)   # :230-234

    def _update_uncompressed_cis(self, cis, t):                    # cache_engine.py:765-778
        Bt, S, H = cis.shape
        if self.total_cis is None:
            self.total_cis = torch.empty((self.B, S + 1024, H), dtype=cis.dtype)    # :768
            self.total_cis[:Bt, :S, :].copy_(cis)                                   # :769
            self.cis_len = S                                                        # :770
            self.cis_prefilled_batch = Bt                                           # :771
        else:
            assert self.cis_prefilled_batch < self.B and self.cis_len == S          # :773, :775
            self.total_cis[t:t + Bt, :S, :].copy_(cis)                              # :776
            self.cis_prefilled_batch += Bt                                          # :777

    def _update_cis_prefill(self, cis, t):                         # cache_engine.py:717-742 (N > 1)
        H, Bt, N = cis.shape
        full_block_len, tail_block_len = N // self.ST, N % self.ST                  # :722-723, :735-736
        full_cis = cis[:, :, :full_block_len * self.ST].reshape(H * Bt, 1, full_block_len * self.ST)   # :724
        comp_cis = F.avg_pool1d(full_cis, kernel_size=self.KS, stride=self.ST).reshape(H, Bt, -1)        # :725
        if self.compressed_cis is None:                                             # :721
            M = comp_cis.shape[-1]                                                  # :726
            self.comp_cis_len = M                                                   # :727
            self.compressed_cis = torch.empty((H, self.B, M), dtype=cis.dtype)      # :728
            self.compressed_cis[:, :Bt, :].copy_(comp_cis)                          # :729
            self.tail_cis = torch.empty((H, self.B, self.KS), dtype=cis.dtype)      # :730
            self.tail_cis_len = self.ST + tail_block_len                            # :731
            self.tail_cis[:, :Bt, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * self.ST:])   # :732
        else:
            self.compressed_cis[:, t:t + Bt, :].copy_(comp_cis)                     # :740
            self.tail_cis[:, t:t + Bt, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * self.ST:])   # :741

    def _update_compress_k_prefill(self, ck, cu_comp, max_seqlen, t):   # cache_engine.py:644-661
        n, H, D = ck.shape
        key_pad = ck.view(-1, max_seqlen, H, D).contiguous()                       # :645-646
        if t == 0:
            self.ck = torch.empty((self.B, max_seqlen, H, D), dtype=ck.dtype)      # :648
        self.ck[t:t + key_pad.shape[0]].copy_(key_pad)                             # :649
        if t == 0:
            self.ck_cu = torch.empty((self.B + 1,), dtype=cu_comp.dtype)           # :651
            self.ck_cu[:cu_comp.shape[-1]].copy_(cu_comp)                          # :652
            self.ck_max = max_seqlen                                               # :653
            self.ck_adder = torch.arange(self.B + 1, dtype=cu_comp.dtype)          # :658
        else:
            self.ck_cu[t:t + cu_comp.shape[-1]].copy_(cu_comp + self.ck_cu[t])     # :660

    def _update_no_compress_k_prefill(self, nck, t):               # cache_engine.py:672-679
        Bt, N, H, D = nck.shape
        if t == 0:
            self.nck = torch.empty((self.B, self.KS, H, D), dtype=nck.dtype)       # :676
        self.nck[t:t + Bt, :N].copy_(nck)                                          # :677
        self.nck_len = N                                                           # :678

    def decode_step(self, keys, cis):
        """One lockstep decode step: keys (B, H, D), cis (B, H)."""
        k1 = keys.unsqueeze(1)                                   # nosa_llama.py:575
        # update_no_compress_k_decode (cache_engine.py:681-692)
        if self.nck_len >= self.KS:                              # :683
            to_compress = self.nck.clone()                       # :684
            self.nck[:, :self.ST].copy_(self.nck[:, self.ST:])   # :685
            self.nck[:, self.ST:self.ST + 1].copy_(k1)           # :686
            self.nck_len = self.ST + 1                           # :687
        else:
            self.nck[:, self.nck_len:self.nck_len + 1].copy_(k1)  # :690
            self.nck_len += 1                                    # :691
            to_compress = None
        # nosa_llama.py:576-580 and update_compress_k_decode (cache_engine.py:663-670)
        if to_compress is not None:
            new = to_compress.mean(dim=1, keepdim=True)          # nosa_llama.py:577
            self.ck = torch.cat((self.ck, new), dim=1)           # cache_engine.py:667
            self.ck_cu += self.ck_adder                          # :668
            self.ck_max += 1                                     # :669
        # update_uncompressed_cis, decode branch (cache_engine.py:779-782; nosa_llama.py:584)
        c3 = cis.unsqueeze(1)                                    # (B, 1, H), nosa_linear.py:128
        self.total_cis[:, self.cis_len:self.cis_len + 1, :].copy_(c3)   # :780
        self.cis_len += 1                                        # :781
        # update_cis, decode branch (cache_engine.py:750-763; nosa_llama.py:585)
        ch = c3.permute(2, 0, 1)                                 # (H, B, 1)
        self.tail_cis[:, :, self.tail_cis_len:self.tail_cis_len + 1].copy_(ch)   # :752
        self.tail_cis_len += 1                                   # :753
        if self.tail_cis_len == self.KS:                         # :754
            tail_comp = self.tail_cis.mean(dim=-1, keepdim=True)  # :756-757
            if self.f2_as_shipped:
                self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len + 1].copy_(tail_comp)   # :759
            else:
                self.compressed_cis = torch.cat((self.compressed_cis, tail_comp), dim=-1)            # :758
            self.tail_cis[:, :, :self.ST].copy_(self.tail_cis[:, :, self.ST:])   # :761
            self.tail_cis_len = self.ST                          # :762
        # decode_update_has_kv_bias, tail bookkeeping (nosa_llama.py:618; cache_engine.py:700-702)
        self.seq_length += 1                                     # :345
        self.cache_lens += 1                                     # :346
        self.tail_len += 1                                       # :357
        if self.tail_len == self.BLOCK:                          # :359, :401
            self.tail_block += 1                                 # :407
            self.tail_len = 0                                    # :408
            self.cache_lens[...] = (self.TOPK - 1) * self.BLOCK + self.tail_len   # :409


def _assert_matches_ref(twin, ref, where):
    B = ref.B
    assert torch.equal(twin.cu, torch.arange(B + 1, dtype=torch.int32) * twin.lk_max), "%s: cu moved" % where
    rf = ref.ck.flatten(0, 1)                                    # nosa_llama.py:581
    tf = twin.comp_k_flat()
    for b in range(B):
        w = "%s, request %d" % (where, b)
        pos, n, m = int(twin.pos[b]), int(twin.n_comp[b]), int(twin.n_ccis[b])
        assert pos == ref.seq_length == ref.cis_len, "%s: pos %d, shipped seq_length %d, cis_len %d" % (
            w, pos, ref.seq_length, ref.cis_len)
        assert n == ref.ck_max == ref.ck.shape[1], "%s: n_comp %d, shipped %d / %d" % (w, n, ref.ck_max, ref.ck.shape[1])
        r0, r1 = int(ref.ck_cu[b]), int(ref.ck_cu[b + 1])
        assert r1 - r0 == n, "%s: shipped cu segment %d != n_comp %d" % (w, r1 - r0, n)
        _eq(twin.comp_k[b, :n], ref.ck[b], w, "compressed keys")
        c0 = int(twin.cu[b])
        _eq(tf[c0:c0 + n], rf[r0:r1], w, "compressed keys through cu")
        assert pos - STRIDE * n == ref.nck_len, "%s: live keys %d, shipped no_compress_k_len %d" % (
            w, pos - STRIDE * n, ref.nck_len)
        _eq(twin.live_keys(b), ref.nck[b, :ref.nck_len], w, "uncompressed keys")
        _eq(twin.total_cis[b, :pos], ref.total_cis[b, :pos], w, "total_cis")
        assert m == ref.compressed_cis.shape[-1], "%s: n_ccis %d, shipped width %d" % (w, m, ref.compressed_cis.shape[-1])
        _eq(twin.ccis[:, b, :m], ref.compressed_cis[:, b, :], w, "compressed cis")
        assert pos - STRIDE * m == ref.tail_cis_len, "%s: live cis %d, shipped tail_cis_len %d" % (
            w, pos - STRIDE * m, ref.tail_cis_len)
        _eq(twin.total_cis[b, STRIDE * m:pos].t(), ref.tail_cis[:, b, :ref.tail_cis_len], w, "tail cis")
        assert int(twin.tail_row()[b]) == ref.tail_len, "%s: tail row" % w
        assert int(twin.tail_block()[b]) == ref.tail_block, "%s: tail block" % w
        assert int(twin.cache_lens()[b]) == int(ref.cache_lens[b]), "%s: cache_lens" % w


def _assert_request_equal(x, ia, y, ib, where):
    """Request ia of twin x against request ib of twin y: counters, live prefixes, derived values."""
    for name in ("pos", "n_comp", "n_ccis"):
        a, c = int(getattr(x, name)[ia]), int(getattr(y, name)[ib])
        assert a == c, "%s: %s %d != %d" % (where, name, a, c)
    pos, n, m = int(x.pos[ia]), int(x.n_comp[ia]), int(x.n_ccis[ia])
    _eq(x.comp_k[ia, :n], y.comp_k[ib, :n], where, "compressed keys")
    _eq(x.live_keys(ia), y.live_keys(ib), where, "uncompressed keys")
    _eq(x.total_cis[ia, :pos], y.total_cis[ib, :pos], where, "total_cis")
    _eq(x.ccis[:, ia, :m], y.ccis[:, ib, :m], where, "compressed cis")
    for name in ("tail_row", "tail_block", "cache_lens", "grid_width", "seqused_k"):
        a, c = int(getattr(x, name)()[ia]), int(getattr(y, name)()[ib])
        assert a == c, "%s: %s %d != %d" % (where, name, a, c)


# ---------------------------------------------------------------------------
# negative-control mutants
# ---------------------------------------------------------------------------
class _OffByOneKey(tw.PositionalTwin):
    def _key_due(self, p, n):
        return STRIDE * n + KERNEL <= p + 1          # not deferred: one step early


class _OffByOneCis(tw.PositionalTwin):
    def _cis_due(self, p, n):
        return STRIDE * n + KERNEL <= p              # deferred: one step late


class _SharedCounter(tw.PositionalTwin):
    """One counter set for the whole batch, as the shipped scalars."""

    def _share(self):
        for c in (self.pos, self.n_comp, self.n_ccis):
            c.fill_(int(c.max()))

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        self._share()
        return out

    def commit(self, *args, **kwargs):
        super().commit(*args, **kwargs)
        self._share()


# ---------------------------------------------------------------------------
# gate runners
# ---------------------------------------------------------------------------
def run_gate1(cls, S, B, steps=300, seed=0, data="ints", invariants=True, ref_kwargs=None):
    g = torch.Generator().manual_seed(seed)
    total = S + steps
    if data == "ints":
        keys, cis, dtype = _ints((B, total, H_, D_), g), _ints((B, total, H_), g), torch.float32
    else:
        keys, cis, dtype = _normals((B, total, H_, D_), g), _normals((B, total, H_), g), torch.bfloat16
    ref = ShippedRef(keys[:, :S], cis[:, :S], **(ref_kwargs or {}))
    twin = cls(B, H_, D_, lk_max=total // STRIDE, n_max=total, dtype=dtype)
    for b in range(B):
        twin.prefill(b, keys[b, :S], cis[b, :S])
    _assert_matches_ref(twin, ref, "prefill")
    if invariants:
        twin.check_invariants()
    for i in range(steps):
        ref.decode_step(keys[:, S + i], cis[:, S + i])
        written = twin.step(keys[:, S + i], cis[:, S + i])
        assert written.tolist() == [S + i] * B
        _assert_matches_ref(twin, ref, "step %d" % (i + 1))
        if invariants:
            twin.check_invariants()


def run_gate2(cls, S_list, rounds=40, T=12, seed=0, invariants=True):
    g = torch.Generator().manual_seed(seed)
    B = len(S_list)
    total = max(S_list) + rounds * T
    lk, nmax = total // STRIDE, total
    keys_c = [_ints((S + rounds * T, H_, D_), g) for S in S_list]
    cis_c = [_ints((S + rounds * T, H_), g) for S in S_list]
    batch = cls(B, H_, D_, lk, nmax)
    same = [tw.PositionalTwin(1, H_, D_, lk, nmax) for _ in S_list]   # same schedule, B = 1
    only = [tw.PositionalTwin(1, H_, D_, lk, nmax) for _ in S_list]   # committed tokens only, B = 1
    for b, S in enumerate(S_list):
        for t, i in ((batch, b), (same[b], 0), (only[b], 0)):
            t.prefill(i, keys_c[b][:S], cis_c[b][:S])

    def compare(where, against):
        for b in range(B):
            _assert_request_equal(batch, b, against[b], 0, "%s, request %d" % (where, b))

    compare("prefill", same)
    compare("prefill", only)
    stats = {"rollbacks": 0, "undone_compressions": 0, "ragged_steps": 0}
    for r in range(rounds):
        if r == 0:
            n = (torch.arange(B) * 3) % (T + 1)               # ragged from the first step
        else:
            n = torch.randint(0, T + 1, (B,), generator=g)
        keep = torch.tensor([int(torch.randint(0, int(n[b]) + 1, (1,), generator=g)) for b in range(B)])
        P = [int(only[b].pos[0]) for b in range(B)]          # committed positions, from the ground truth
        ks = _ints((B, T, H_, D_), g, 100, 120)              # rejected draft tokens
        cs = _ints((B, T, H_), g, 100, 120)
        for b in range(B):
            k = int(keep[b])
            ks[b, :k] = keys_c[b][P[b]:P[b] + k]
            cs[b, :k] = cis_c[b][P[b]:P[b] + k]
        ctr_b = [batch.counters()]
        ctr_s = [[same[b].counters()] for b in range(B)]
        for u in range(T):
            act = n > u
            stats["ragged_steps"] += int(bool(act.any()) and not bool(act.all()))
            batch.step(ks[:, u], cs[:, u], active=act)
            ctr_b.append(batch.counters())
            for b in range(B):
                same[b].step(ks[b:b + 1, u], cs[b:b + 1, u], active=act[b:b + 1])
                ctr_s[b].append(same[b].counters())
            if invariants:
                batch.check_invariants()
            compare("round %d, token %d" % (r, u), same)
        top = ctr_b[-1]
        batch.commit(torch.stack(ctr_b), keep)
        for b in range(B):
            same[b].commit(torch.stack(ctr_s[b]), keep[b:b + 1])
            for u in range(int(keep[b])):
                only[b].step(keys_c[b][P[b] + u].unsqueeze(0), cis_c[b][P[b] + u].unsqueeze(0))
            if int(keep[b]) < int(n[b]):
                stats["rollbacks"] += 1
                stats["undone_compressions"] += int(int(top[1, b]) > int(batch.n_comp[b])
                                                    or int(top[2, b]) > int(batch.n_ccis[b]))
        if invariants:
            batch.check_invariants()
        compare("round %d, after commit" % r, same)
        compare("round %d, after commit" % r, only)
    return stats


# ---------------------------------------------------------------------------
# gate 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [4032, 4033, 4047])     # S % 16 = 0, 1, 15 (and S % 64 = 0, 1, 15)
def test_gate1_twin_equals_shipped_tables(S, B):
    run_gate1(tw.PositionalTwin, S, B)


def test_gate1_bf16():
    run_gate1(tw.PositionalTwin, 4033, 4, data="normals")


def test_gate1_detects_shipped_f2():
    """The shipped line cache_engine.py:759 (an empty-slice copy_) does not grow
    the compressed-cis table; the gate sees it at the first cis summary."""
    with pytest.raises(AssertionError, match="n_ccis"):
        run_gate1(tw.PositionalTwin, 4033, 1, ref_kwargs={"f2_as_shipped": True})


# ---------------------------------------------------------------------------
# gate 2
# ---------------------------------------------------------------------------
def test_gate2_ragged_equals_independent_runs():
    stats = run_gate2(tw.PositionalTwin, [4032, 4033, 4047, 4100])
    assert stats["ragged_steps"] > 0
    assert stats["rollbacks"] > 0
    assert stats["undone_compressions"] > 0, "no rollback crossed a compression event"


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
def test_negctl_key_event_off_by_one_fails_gate1():
    with pytest.raises(AssertionError):
        run_gate1(_OffByOneKey, 4033, 1, invariants=False)


def test_negctl_cis_event_off_by_one_fails_gate1():
    with pytest.raises(AssertionError):
        run_gate1(_OffByOneCis, 4033, 1, invariants=False)


def test_negctl_shared_counter_fails_gate2():
    with pytest.raises(AssertionError):
        run_gate2(_SharedCounter, [4033] * 4, invariants=False)


def test_negctl_shared_counter_passes_gate1():
    """Lockstep cannot see a shared counter: gate 2 is the gate that does."""
    run_gate1(_SharedCounter, 4033, 4)


# ---------------------------------------------------------------------------
# units: rollback bound, refusals, layout, derived values
# ---------------------------------------------------------------------------
def _twin_from(keys, cis, S, lk=None, nmax=None):
    B, total = keys.shape[0], keys.shape[1]
    t = tw.PositionalTwin(B, keys.shape[2], keys.shape[3], lk or total // STRIDE, nmax or total)
    for b in range(B):
        t.prefill(b, keys[b, :S], cis[b, :S])
    return t


def _state(t):
    return {k: getattr(t, k).clone() for k in ("pos", "n_comp", "n_ccis", "epoch", "comp_k", "ring", "total_cis", "ccis")}


def _assert_state_equal(t, snap, where):
    for k, v in snap.items():
        assert torch.equal(getattr(t, k), v), "%s: %s changed" % (where, k)


def _worst_case_pair():
    """S % 16 == 15 and one committed step: the live keys are 32 positions,
    the most the invariant allows."""
    g = torch.Generator().manual_seed(7)
    S = 4047
    keys, cis = _ints((1, S + 100, H_, D_), g), _ints((1, S + 100, H_), g)
    twin, ref = _twin_from(keys, cis, S), _twin_from(keys, cis, S)
    for t in (twin, ref):
        t.step(keys[:, S], cis[:, S])
    assert int(twin.pos[0]) - STRIDE * int(twin.n_comp[0]) == 32
    return twin, ref, g


def test_rollback_over_32_positions_is_exact():
    twin, ref, g = _worst_case_pair()
    ctr = twin.advance(_ints((1, 32, H_, D_), g, 100, 120), _ints((1, 32, H_), g, 100, 120), [32])
    assert int(twin.n_comp[0]) > int(ctr[0, 1, 0]), "the rejected tokens crossed a compression event"
    twin.commit(ctr, [0])
    _assert_request_equal(twin, 0, ref, 0, "after a 32-position rollback")


def test_rollback_over_33_positions_is_refused_and_would_corrupt():
    twin, ref, g = _worst_case_pair()
    ctr = twin.advance(_ints((1, 33, H_, D_), g, 100, 120), _ints((1, 33, H_), g, 100, 120), [33])
    snap = _state(twin)
    with pytest.raises(tw.TwinRollbackError):
        twin.commit(ctr, [0])
    _assert_state_equal(twin, snap, "refused commit")
    # what the guard prevents: force the reset, and a live key row is gone
    twin.pos[0], twin.n_comp[0], twin.n_ccis[0] = ctr[0, 0, 0], ctr[0, 1, 0], ctr[0, 2, 0]
    assert not torch.equal(twin.live_keys(0), ref.live_keys(0))


def test_stale_snapshot_is_refused():
    g = torch.Generator().manual_seed(8)
    S = 4040
    keys, cis = _ints((2, S + 50, H_, D_), g), _ints((2, S + 50, H_), g)
    twin = _twin_from(keys, cis, S)
    ctr = twin.advance(keys[:, S:S + 5], cis[:, S:S + 5], [5, 3])
    twin.commit(ctr, [5, 3])
    snap = _state(twin)
    with pytest.raises(tw.TwinRollbackError, match="stale"):
        twin.commit(ctr, [2, 1])
    _assert_state_equal(twin, snap, "stale commit")


def test_advance_equals_manual_steps():
    g = torch.Generator().manual_seed(9)
    S, T = 4040, 20
    keys, cis = _ints((3, S + T, H_, D_), g), _ints((3, S + T, H_), g)
    a, m = _twin_from(keys, cis, S), _twin_from(keys, cis, S)
    n = torch.tensor([20, 7, 0])
    ctr = a.advance(keys[:, S:], cis[:, S:], n)
    man = [m.counters()]
    for u in range(T):
        m.step(keys[:, S + u], cis[:, S + u], active=n > u)
        man.append(m.counters())
    assert torch.equal(ctr, torch.stack(man))
    for b in range(3):
        _assert_request_equal(a, b, m, b, "advance vs steps")
        for u in range(int(n[b]), T + 1):
            assert torch.equal(ctr[u, :, b], ctr[int(n[b]), :, b]), "counters move after n[b]"


def test_capacity_refusal_changes_nothing():
    g = torch.Generator().manual_seed(10)
    S = 64
    keys, cis = _ints((1, 200, H_, D_), g), _ints((1, 200, H_), g)
    # Lk_max = the prompt's 3 windows: the cis summary due at p = 79 has no row
    twin = _twin_from(keys, cis, S, lk=S // STRIDE - 1, nmax=200)
    for p in range(S, 79):
        twin.step(keys[:, p], cis[:, p])
    snap = _state(twin)
    with pytest.raises(tw.TwinCapacityError):
        twin.step(keys[:, 79], cis[:, 79])
    _assert_state_equal(twin, snap, "Lk_max refusal")
    # N_max = S + 2: positions 64 and 65 fit, 66 does not
    twin = _twin_from(keys, cis, S, lk=50, nmax=S + 2)
    twin.step(keys[:, 64], cis[:, 64])
    twin.step(keys[:, 65], cis[:, 65])
    snap = _state(twin)
    with pytest.raises(tw.TwinCapacityError):
        twin.step(keys[:, 66], cis[:, 66])
    _assert_state_equal(twin, snap, "N_max refusal")


def test_prefill_and_step_refusals():
    g = torch.Generator().manual_seed(11)
    keys, cis = _ints((100, H_, D_), g), _ints((100, H_), g)
    twin = tw.PositionalTwin(2, H_, D_, lk_max=4, n_max=80)
    with pytest.raises(ValueError):
        twin.prefill(0, keys[:31], cis[:31])                 # no complete window
    with pytest.raises(tw.TwinCapacityError):
        twin.prefill(0, keys[:96], cis[:96])                 # 5 windows > Lk_max 4
    with pytest.raises(tw.TwinCapacityError):
        tw.PositionalTwin(1, H_, D_, lk_max=10, n_max=40).prefill(0, keys[:48], cis[:48])   # S > N_max
    twin.prefill(0, keys[:64], cis[:64])
    with pytest.raises(RuntimeError, match="prefill"):
        twin.step(torch.zeros(2, H_, D_), torch.zeros(2, H_))   # request 1 has no prefill


def test_key_window_is_in_position_order():
    S = 40
    pk = torch.arange(400, dtype=torch.float32).view(400, 1, 1)   # key value = its position
    twin = tw.PositionalTwin(1, 1, 1, lk_max=20, n_max=400)
    twin.prefill(0, pk[:S], torch.zeros(S, 1))
    for start in (0, 8):
        w = twin.key_window(torch.tensor([0]), torch.tensor([start]))
        assert torch.equal(w.flatten(), torch.arange(start, start + 32, dtype=torch.float32))
    for p in range(S, 201):
        twin.step(pk[p].view(1, 1, 1), torch.zeros(1, 1))
    for start in (137, 150, 169):                                 # the ring holds positions 137 .. 200
        w = twin.key_window(torch.tensor([0]), torch.tensor([start]))
        assert torch.equal(w.flatten(), torch.arange(start, start + 32, dtype=torch.float32))
    n = int(twin.n_comp[0])
    assert torch.equal(twin.comp_k[0, :n].flatten(), STRIDE * torch.arange(n, dtype=torch.float32) + 15.5)


def test_inactive_requests_are_untouched():
    g = torch.Generator().manual_seed(12)
    S = 4040
    keys, cis = _ints((3, S + 10, H_, D_), g), _ints((3, S + 10, H_), g)
    twin = _twin_from(keys, cis, S)
    snap = _state(twin)
    written = twin.step(keys[:, S], cis[:, S], active=torch.tensor([True, False, True]))
    assert written.tolist() == [S, -1, S]
    for k in ("pos", "n_comp", "n_ccis", "comp_k", "ring", "total_cis"):
        assert torch.equal(getattr(twin, k)[1], snap[k][1]), "request 1: %s changed" % k
    assert torch.equal(twin.ccis[:, 1], snap["ccis"][:, 1])
    snap = _state(twin)
    written = twin.step(keys[:, S + 1], cis[:, S + 1], active=torch.zeros(3, dtype=torch.bool))
    assert written.tolist() == [-1, -1, -1]
    _assert_state_equal(twin, snap, "no active request")


def test_constant_cu_flat_view_and_seqused():
    g = torch.Generator().manual_seed(13)
    S = 4033
    keys, cis = _ints((3, S + 60, H_, D_), g), _ints((3, S + 60, H_), g)
    twin = _twin_from(keys, cis, S)
    cu0 = twin.cu.clone()
    twin.advance(keys[:, S:], cis[:, S:], [60, 17, 33])
    assert torch.equal(twin.cu, cu0)
    assert torch.equal(cu0, torch.arange(4, dtype=torch.int32) * twin.lk_max)
    s = twin.seqused_k()
    assert s.dtype == torch.int32 and s.is_contiguous() and torch.equal(s.long(), twin.n_comp)
    flat = twin.comp_k_flat()
    for b in range(3):
        n = int(twin.n_comp[b])
        _eq(flat[int(twin.cu[b]):int(twin.cu[b]) + n], twin.comp_k[b, :n], "request %d" % b, "flat view")


@pytest.mark.parametrize("S", [4032, 4033, 4095, 16128])
def test_derived_values_match_the_shipped_prefill(S):
    g = torch.Generator().manual_seed(14)
    keys, cis = _ints((1, S, 1, 1), g), _ints((1, S, 1), g)
    twin = _twin_from(keys, cis, S)
    assert int(twin.grid_width()[0]) == (S + 1 + BLOCK - 1) // BLOCK      # nosa_llama.py:817-818
    assert int(twin.cache_lens()[0]) == 4032 + S % BLOCK                 # cache_engine.py:327
    assert int(twin.tail_row()[0]) == S % BLOCK                          # cache_engine.py:309
    assert int(twin.tail_block()[0]) == S // BLOCK                       # cache_engine.py:321
    assert int(twin.n_comp[0]) == int(twin.n_ccis[0]) == S // STRIDE - 1


def test_twin_imports_only_torch():
    tree = ast.parse(TWIN.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert mods <= {"__future__", "torch", "torch.nn.functional"}, mods
