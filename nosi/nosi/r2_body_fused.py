"""R2: the LEAN BODY WRITE PATH. DecodeBody.decode_fixed (the capture step's body, docs/R2_CAPTURE_AUDIT.md fix X3)
runs 48 GPU ops per layer in the M-P3 run 2190876: the diff kernel, the two host gathers, the write-back kernel and 44
small torch ops (the refusal check, write_tail's gathers / where / index_put, the selection's where / masked_fill /
copies, the full mask, the tail id and the attention lengths). In the captured step they take 3.96 ms per step at B 64
against 1.12 ms for the shipped decode_update_kv's non-gather ops (the gathers themselves: 19.28 vs 19.36 ms).
FusedBody.decode_fixed keeps the diff kernel, the gathers and the write-back kernel and replaces the 44 ops by ONE
Triton kernel (_body_kernel, one program per request): 5 GPU ops per layer. The results are BITWISE decode_fixed's:
_k_gpu, _v_gpu, _kv_bias_gpu, _block_map, _new_block_map_buf, _load_mask, _cache_lens, err and the host rows.

ORDER. decode_fixed: write_tail, diff, select (where / masked_fill / copy), gathers, full, write-back, tail id, lengths.
FusedBody: diff, _body_kernel (write_tail, select, tail id, lengths, full), gathers, write-back. The two moves are
exact: diff reads _block_map and topk_idx (write_tail writes neither), and the gathers read _load_mask and write the
loaded slots (the tail id and the lengths are read by neither). The tail rows are written before the gathers in both
orders (if a gather ever wrote slot 63, findings F1, it would overwrite them in both).

_body_kernel, program b (all B programs read every request's pos / active for the whole-batch refusal):
  bad   = any over requests of active & (pos < 0 | pos >= min(N_host, N_bias)); program 0: err |= ERR_POS if bad
  act   = active[b] & not bad;  p = pos[b]
  tail  rows 4032 + p % 64 of _k_gpu, _v_gpu <- key / value; of _kv_bias_gpu <- kv_bias[b, clamp(p, 0, N - 1)]; where act
  map   per (h, j): m = act ? new[h, b, j] : old[h, b, j]; new <- m; load <- -1 where not act;
        _block_map <- (j == 63 and act) ? (p + 1) // 64 : m
  full  [b] <- act and (p + 1) % 64 == 0 (a (B,) bool buffer the write-back kernel reads)
  lens  [b] <- 4032 + (p + 1) % 64 (f4_fix: 4032 + p % 64 + 1) where act
FusedBody refuses a subclass that overrides one of DecodeBody's pieces (positions, full_mask, cache_lens_value,
write_tail, select, writeback): those are the negative controls of the torch body, and the kernel would ignore them.

Imports torch (and triton when it is installed). Load it by file path.
"""
import torch            # no `from __future__ import annotations`: the kernels' constexpr annotations must be objects

try:
    import triton
    import triton.language as tl
except Exception:                                      # pragma: no cover
    triton = tl = None

PLANTS = {0: "none", 1: "the tail rows written for an inactive request too", 2: "the tail id (slot 63's block) not set",
          3: "the load mask of an inactive request kept"}


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


if triton is not None:
    @triton.jit
    def _body_kernel(pos_ptr, act_ptr, err_ptr, ks_ptr, vs_ptr, bias_ptr, kg_ptr, vg_ptr, bg_ptr,
                     map_ptr, new_ptr, load_ptr, lens_ptr, full_ptr,
                     B, limit, n_bias, rows_gpu, tail0, base_len,
                     s_kb, s_kh, s_kd, s_vb, s_vh, s_vd, s_bb, s_bn, s_bh,
                     H: tl.constexpr, D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr, BP: tl.constexpr,
                     TOPK: tl.constexpr, BS: tl.constexpr, F4FIX: tl.constexpr, ERR_POS: tl.constexpr,
                     PLANT: tl.constexpr):
        b = tl.program_id(0).to(tl.int64)
        # ln: a 2-lane vector for the masked stores of ONE value. A scalar mask must not be used: the Triton
        # interpreter's store reads `if not mask` on its scalar mask, so a False scalar mask stores anyway (CPU job
        # 2190958, F3: an inactive request's _cache_lens overwritten); the compiled kernel is not affected.
        ln = tl.arange(0, 2)
        lanes = tl.arange(0, BP)
        ml = lanes < B
        pa = tl.load(pos_ptr + lanes, mask=ml, other=0)
        aa = tl.load(act_ptr + lanes, mask=ml, other=0) != 0
        badv = aa & ((pa < 0) | (pa >= limit))
        any_bad = tl.max(badv.to(tl.int32), axis=0)                       # 0 or 1
        e = tl.load(err_ptr)
        tl.store(err_ptr + ln * 0, (e | (any_bad * ERR_POS)) + ln * 0, mask=(ln == 0) & (b == 0))
        p = tl.load(pos_ptr + b)
        act = (tl.load(act_ptr + b) != 0) & (any_bad == 0)
        hh = tl.arange(0, HP).to(tl.int64)
        dd = tl.arange(0, DP).to(tl.int64)
        mh = hh < H
        mhd = mh[:, None] & (dd < D)[None, :]
        wmask = act
        if PLANT == 1:
            wmask = act | (act == 0)
        # the tail rows
        row = tail0 + p % BS
        kv = tl.load(ks_ptr + b * s_kb + hh[:, None] * s_kh + dd[None, :] * s_kd, mask=mhd, other=0)
        vv = tl.load(vs_ptr + b * s_vb + hh[:, None] * s_vh + dd[None, :] * s_vd, mask=mhd, other=0)
        dst = (b * rows_gpu + row) * (H * D) + hh[:, None] * D + dd[None, :]
        tl.store(kg_ptr + dst, kv, mask=mhd & wmask)
        tl.store(vg_ptr + dst, vv, mask=mhd & wmask)
        prow = tl.minimum(tl.maximum(p, 0), n_bias - 1)
        bv = tl.load(bias_ptr + b * s_bb + prow * s_bn + hh * s_bh, mask=mh, other=0)
        tl.store(bg_ptr + (b * rows_gpu + row) * H + hh, bv, mask=mh & wmask)
        # the selection, then slot 63's block id
        jj = tl.arange(0, TOPK).to(tl.int64)
        moff = hh[:, None] * (B * TOPK) + b * TOPK + jj[None, :]
        mm = mh[:, None] & (jj < TOPK)[None, :]
        old = tl.load(map_ptr + moff, mask=mm, other=0)
        new = tl.load(new_ptr + moff, mask=mm, other=0)
        m = tl.where(act, new, old)
        tl.store(new_ptr + moff, m, mask=mm)
        if PLANT != 3:
            tl.store(load_ptr + moff, tl.full([HP, TOPK], -1, tl.int64), mask=mm & (act == 0))
        if PLANT == 2:
            fin = m
        else:
            fin = tl.where((jj[None, :] == TOPK - 1) & act, (p + 1) // BS, m)
        tl.store(map_ptr + moff, fin, mask=mm)
        # the write-back mask and the attention length
        tl.store(full_ptr + b, (act & ((p + 1) % BS == 0)).to(tl.uint8))
        if F4FIX:
            val = base_len + p % BS + 1
        else:
            val = base_len + (p + 1) % BS
        tl.store(lens_ptr + b + ln * 0, val.to(tl.int32) + ln * 0, mask=(ln == 0) & act)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


_CLASSES = {}


def fused_body_class(DeviceWritebackBody, DecodeBody):
    """FusedBody over the given classes (the tests pass the CPU-loaded r2_decode_body's); one class per pair."""
    key = (DeviceWritebackBody, DecodeBody)
    if key not in _CLASSES:
        _CLASSES[key] = _make_class(DeviceWritebackBody, DecodeBody)
    return _CLASSES[key]


def _make_class(DeviceWritebackBody, DecodeBody):

    class FusedBody(DeviceWritebackBody):
        """DeviceWritebackBody whose decode_fixed runs the lean path (module docstring). plant: a NEGATIVE CONTROL
        (PLANTS), tests only."""

        PIECES = ("positions", "full_mask", "cache_lens_value", "write_tail", "select", "writeback")

        def __init__(self, f4_fix: bool, writeback_kernel, plant: int = 0, num_warps: int = 4):
            super().__init__(f4_fix, writeback_kernel)
            for name in self.PIECES:
                if getattr(type(self), name) is not getattr(DeviceWritebackBody if name == "writeback" else DecodeBody,
                                                           name):
                    raise TypeError("FusedBody runs its own kernel: the override of %s would be ignored" % name)
            self.plant = int(plant)
            self.num_warps = int(num_warps)
            self._full = {}                                   # (B, device) -> (B,) bool, made once (eager warm-up)

        def _full_buf(self, B: int, device) -> torch.Tensor:
            key = (int(B), str(device))
            if key not in self._full:
                self._full[key] = torch.zeros(B, dtype=torch.bool, device=device)
            return self._full[key]

        def decode_fixed(self, eng, key_states, value_states, kv_bias, topk_idx, pos, active, err,
                         diff_offload, gather_k, gather_v_bias):
            """DecodeBody.decode_fixed, bitwise, in 5 GPU ops (module docstring); the same static checks."""
            if triton is None:
                raise RuntimeError("triton is not installed")
            B, S, H, D = key_states.shape
            if S != 1 or H != eng.head_num or D != eng.head_dim or tuple(value_states.shape) != (B, 1, H, D):
                raise ValueError("key_states / value_states must be (B, 1, %d, %d)" % (eng.head_num, eng.head_dim))
            if kv_bias.dim() != 3 or kv_bias.shape[0] != B or kv_bias.shape[2] != H:
                raise ValueError("kv_bias must be (%d, N, %d), got %s" % (B, H, tuple(kv_bias.shape)))
            if tuple(topk_idx.shape) != (H, B, eng.topk) or topk_idx.dtype != torch.int64:
                raise ValueError("topk_idx must be (%d, %d, %d) int64" % (H, B, eng.topk))
            if tuple(pos.shape) != (B,) or pos.dtype != torch.int64 or tuple(active.shape) != (B,) \
                    or active.dtype != torch.bool:
                raise ValueError("pos must be (%d,) int64 and active (%d,) bool" % (B, B))
            if tuple(err.shape) != () or err.dtype != torch.int32 or err.device != pos.device:
                raise ValueError("err must be an int32 0-d tensor on %s" % pos.device)
            bs, topk = int(eng.block_size), int(eng.topk)
            if bs & (bs - 1) or topk & (topk - 1):
                raise ValueError("block_size and topk must be powers of two")
            for name in ("_block_map", "_new_block_map_buf", "_load_mask"):
                t = getattr(eng, name)
                if tuple(t.shape) != (H, B, topk) or t.dtype != torch.int64 or not t.is_contiguous():
                    raise ValueError("eng.%s must be a contiguous (%d, %d, %d) int64 tensor" % (name, H, B, topk))
            for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
                if not getattr(eng, name).is_contiguous():
                    raise ValueError("eng.%s must be contiguous" % name)
            if eng._cache_lens.dtype != torch.int32 or tuple(eng._cache_lens.shape) != (B,):
                raise ValueError("eng._cache_lens must be (%d,) int32" % B)
            ks, vs = key_states[:, 0], value_states[:, 0]
            if ks.dtype != eng._k_gpu.dtype:                    # write_tail's .to(dtype) (no-op when equal)
                ks = ks.to(eng._k_gpu.dtype)
            if vs.dtype != eng._v_gpu.dtype:
                vs = vs.to(eng._v_gpu.dtype)
            kb = kv_bias if kv_bias.dtype == eng._kv_bias_gpu.dtype else kv_bias.to(eng._kv_bias_gpu.dtype)
            limit = min(int(eng._k_cpu.shape[1]), int(kv_bias.shape[1]))
            full = self._full_buf(B, pos.device)
            rows_gpu = int(eng._k_gpu.shape[1])
            tail0 = int(eng._tail_block_idx_on_gpu) * bs
            if int(eng._tail_block_idx_on_gpu) != topk - 1:
                raise ValueError("the tail slot must be topk - 1")
            diff_offload(eng._block_map, topk_idx, eng._new_block_map_buf, eng._load_mask)
            _body_kernel[(B,)](
                pos, active.view(torch.uint8), err, _bits(ks), _bits(vs), _bits(kb),
                _bits(eng._k_gpu), _bits(eng._v_gpu), _bits(eng._kv_bias_gpu),
                eng._block_map, eng._new_block_map_buf, eng._load_mask, eng._cache_lens, full.view(torch.uint8),
                B, limit, int(kv_bias.shape[1]), rows_gpu, tail0, (topk - 1) * bs,
                ks.stride(0), ks.stride(1), ks.stride(2), vs.stride(0), vs.stride(1), vs.stride(2),
                kb.stride(0), kb.stride(1), kb.stride(2),
                H=H, D=D, HP=_pow2(H), DP=_pow2(D), BP=_pow2(B), TOPK=topk, BS=bs, F4FIX=bool(self.f4_fix),
                ERR_POS=int(self.ERR_POS), PLANT=self.plant, num_warps=self.num_warps)
            gather_k(eng._k_gpu, eng._k_cpu, eng._load_mask, bs)
            gather_v_bias(eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, kv_bias, eng._load_mask, bs)
            self.writeback_kernel(eng._k_gpu, eng._v_gpu, eng._k_cpu, eng._v_cpu, full, pos, tail0, bs)
            return eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._cache_lens

    return FusedBody


def FusedBody(f4_fix: bool, writeback_kernel, **kw):
    """FusedBody over nosi's r2_decode_body (the GPU path; package import)."""
    from .r2_decode_body import DecodeBody, DeviceWritebackBody
    return fused_body_class(DeviceWritebackBody, DecodeBody)(f4_fix, writeback_kernel, **kw)
