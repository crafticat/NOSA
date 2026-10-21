"""R2: the FUSED TWIN STEP. One Triton kernel replaces PositionalTwin.step_unchecked (72 small kernels per layer in
the M-P3 run 2190876: 2304 GPU ops and 5.36 ms per step in the captured R2 step at B 64) and advance_unchecked (T steps
in ONE launch). The results are BITWISE those of the torch ops when twin.mean_fn is r2_bi_decode.fixed_order_mean (the
BI mode's and the tick's summaries): every table (comp_k, ring, total_cis, ccis), every counter (pos, n_comp, n_ccis),
the sticky overflow flag and advance's ctr. Docs: docs/R2_FUSED.md.

ONE PROGRAM PER BANK b (grid (B,)), T steps in program order. Per step u (active = n[b] > u, or the active mask):
  due_k = active & 16 n_comp + 32 <= pos;  due_c = active & 16 n_ccis + 32 <= pos + 1
  bad   = active & (pos >= N_max | not prefilled) | due_k & n_comp >= Lk_max | due_c & n_ccis >= Lk_max
          (bad -> the overflow flag is set and the bank writes nothing this step), a = active & not bad
  1. key summary: ring rows (16 n_comp + w) % 64, w = 0 .. 31, IN THAT ORDER: a float32 chain acc = x0; acc = acc + x_w;
     then acc * 2^-5 (bitwise acc / 32: both are the correctly rounded acc * 2^-5), rounded to the table dtype; written
     to comp_k[b, min(n_comp, Lk_max - 1)] where due_k & a; n_comp += due_k & a
  2. ring[b, pos % 64] = key where a          3. total_cis[b, min(pos, N_max - 1)] = cis where a
  4. cis summary over total_cis rows min(16 n_ccis + w, N_max - 1) AS AFTER step 3 (the row written in 3 takes the new
     value in registers: no store-then-load inside the program), the same chain per head; written to
     ccis[:, b, min(n_ccis, Lk_max - 1)] where due_c & a; n_ccis += due_c & a
  5. pos += a
The window loads are MASKED by due & a: a bank that does not write a summary reads nothing for it (torch computes the
mean of every bank and discards it; the written bits are the same, the per-element order does not depend on B). The
loads of a step come before a barrier, its stores after it, and a second barrier ends the step (the next step reads the
rows this one wrote).

ROUNDING TO bfloat16 is done with integer ops (round to nearest even on the float32 bits; NaN -> 0x7FC0, torch's CPU
value), so the Triton interpreter (whose float32 -> bfloat16 cast truncates) and the GPU give the same bits; torch's
.to(torch.bfloat16) rounds to nearest even on both devices. Tables of float32 are stored as float32.

NOT the semantics of step_fixed (the capture step's twin): step_fixed refuses the WHOLE batch and ORs ERR_* bits into
twin.err, and it uses torch's .mean. attach(twin, ("step_fixed",)) is therefore a DECLARED change (per-bank refusal
through the overflow flag, the fixed-order mean); it is what the R2F arm of the M-F cell measures.

Imports torch (and triton when it is installed). Load it by file path (nosi/nosi/__init__.py imports CUDA modules).
"""
import functools

import torch            # no `from __future__ import annotations`: the kernels' constexpr annotations must be objects

try:                                                   # triton is needed only to launch the kernel; the checks work
    import triton                                      # without it
    import triton.language as tl
except Exception:                                      # pragma: no cover - not installed, or a broken install
    triton = tl = None

WIN = 32             # r2_positional_twin.KERNEL (the summary window); the kernel's * 2^-5 is / 32
STRIDE = 16          # r2_positional_twin.STRIDE
RING = 64            # r2_positional_twin.RING
TABLE_DTYPES = (torch.bfloat16, torch.float32)
PLANTS = {0: "none", 1: "the key summary summed in REVERSE order", 2: "the cis summary reads total_cis BEFORE step 3",
          3: "the ring row written for an inactive bank too"}


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


if triton is not None:
    @triton.jit
    def _dec(x, BF16: tl.constexpr):
        """The table bits (int16 for bfloat16, int32 for float32) as float32 (exact)."""
        if BF16:
            return (x.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
        else:
            return x.to(tl.float32, bitcast=True)

    @triton.jit
    def _enc(f, BF16: tl.constexpr):
        """float32 -> the table bits: bfloat16 rounded to nearest even (NaN -> 0x7FC0), or the float32 bits."""
        if BF16:
            u = f.to(tl.uint32, bitcast=True)
            r = (u + (((u >> 16) & 1) + 0x7FFF)) >> 16
            nan = ((u & 0x7FFFFFFF) > 0x7F800000).to(tl.uint32)
            r = r + nan * (0x7FC0 - r)
            return r.to(tl.uint16).to(tl.int16, bitcast=True)
        else:
            return f.to(tl.int32, bitcast=True)

    @triton.jit
    def _twin_kernel(pos_ptr, ncomp_ptr, nccis_ptr, epoch_ptr, pre_ptr, ovf_ptr,
                     compk_ptr, ring_ptr, tcis_ptr, ccis_ptr,
                     keys_ptr, cis_ptr, act_ptr, n_ptr, ctr_ptr,
                     T, B, n_max, lk_max,
                     s_kb, s_kt, s_kh, s_kd, s_cb, s_ct, s_ch,
                     H: tl.constexpr, D: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr,
                     WINC: tl.constexpr, STRC: tl.constexpr, RINGC: tl.constexpr,
                     FROM_N: tl.constexpr, WRITE_CTR: tl.constexpr, BF16: tl.constexpr, PLANT: tl.constexpr):
        b = tl.program_id(0).to(tl.int64)
        hh = tl.arange(0, HP).to(tl.int64)
        dd = tl.arange(0, DP).to(tl.int64)
        mh = hh < H
        mhd = mh[:, None] & (dd < D)[None, :]
        hd = hh[:, None] * D + dd[None, :]                          # offset inside a contiguous (H, D) row
        HD = H * D
        rb = ring_ptr + b * (RINGC * HD)
        tb = tcis_ptr + b * n_max * H
        p = tl.load(pos_ptr + b)
        nk = tl.load(ncomp_ptr + b)
        nc = tl.load(nccis_ptr + b)
        ep = tl.load(epoch_ptr + b)
        pre = tl.load(pre_ptr + b)
        if WRITE_CTR:
            tl.store(ctr_ptr + b, p)
            tl.store(ctr_ptr + B + b, nk)
            tl.store(ctr_ptr + 2 * B + b, nc)
            tl.store(ctr_ptr + 3 * B + b, ep)
        for u in range(T):
            if FROM_N:
                act = tl.load(n_ptr + b) > u
            else:
                act = tl.load(act_ptr + b) != 0
            due_k = act & (STRC * nk + WINC <= p)
            due_c = act & (STRC * nc + WINC <= p + 1)
            bad = (act & ((p >= n_max) | (pre == 0))) | (due_k & (nk >= lk_max)) | (due_c & (nc >= lk_max))
            a = act & (bad == 0)
            wk = due_k & a
            wc = due_c & a
            tl.store(ovf_ptr + hh * 0, tl.full([HP], 1, tl.uint8), mask=(hh == 0) & bad)
            # ---- loads (state BEFORE this step)
            k0 = STRC * nk
            if PLANT == 1:
                acc = _dec(tl.load(rb + ((k0 + WINC - 1) % RINGC) * HD + hd, mask=mhd & wk, other=0), BF16)
                for w in tl.static_range(1, WINC):
                    acc = acc + _dec(tl.load(rb + ((k0 + WINC - 1 - w) % RINGC) * HD + hd, mask=mhd & wk, other=0),
                                     BF16)
            else:
                acc = _dec(tl.load(rb + (k0 % RINGC) * HD + hd, mask=mhd & wk, other=0), BF16)
                for w in tl.static_range(1, WINC):
                    acc = acc + _dec(tl.load(rb + ((k0 + w) % RINGC) * HD + hd, mask=mhd & wk, other=0), BF16)
            mean_k = _enc(acc * 0.03125, BF16)
            kv = tl.load(keys_ptr + b * s_kb + u * s_kt + hh[:, None] * s_kh + dd[None, :] * s_kd, mask=mhd, other=0)
            cv = tl.load(cis_ptr + b * s_cb + u * s_ct + hh * s_ch, mask=mh, other=0)
            cvf = _dec(cv, BF16)
            ip = tl.minimum(p, n_max - 1)
            c0 = STRC * nc
            row = tl.minimum(c0, n_max - 1)
            x = _dec(tl.load(tb + row * H + hh, mask=mh & wc, other=0), BF16)
            if PLANT != 2:
                x = tl.where(a & (row == ip), cvf, x)
            accc = x
            for w in tl.static_range(1, WINC):
                row = tl.minimum(c0 + w, n_max - 1)
                x = _dec(tl.load(tb + row * H + hh, mask=mh & wc, other=0), BF16)
                if PLANT != 2:
                    x = tl.where(a & (row == ip), cvf, x)
                accc = accc + x
            mean_c = _enc(accc * 0.03125, BF16)
            tl.debug_barrier()
            # ---- stores
            ik = tl.minimum(nk, lk_max - 1)
            tl.store(compk_ptr + (b * lk_max + ik) * HD + hd, mean_k, mask=mhd & wk)
            if PLANT == 3:
                tl.store(rb + (p % RINGC) * HD + hd, kv, mask=mhd)
            else:
                tl.store(rb + (p % RINGC) * HD + hd, kv, mask=mhd & a)
            tl.store(tb + ip * H + hh, cv, mask=mh & a)
            ic = tl.minimum(nc, lk_max - 1)
            tl.store(ccis_ptr + hh * (B * lk_max) + b * lk_max + ic, mean_c, mask=mh & wc)
            nk = nk + wk.to(tl.int64)
            nc = nc + wc.to(tl.int64)
            p = p + a.to(tl.int64)
            if WRITE_CTR:
                base = ctr_ptr + (u + 1) * 4 * B + b
                tl.store(base, p)
                tl.store(base + B, nk)
                tl.store(base + 2 * B, nc)
                tl.store(base + 3 * B, ep)
            tl.debug_barrier()
        tl.store(pos_ptr + b, p)
        tl.store(ncomp_ptr + b, nk)
        tl.store(nccis_ptr + b, nc)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def check_twin(twin) -> None:
    """Metadata checks of the twin's tables (no device read). Raises ValueError."""
    dt = twin.ring.dtype
    if dt not in TABLE_DTYPES:
        raise ValueError("the fused twin step takes tables of %s, not %s" % (TABLE_DTYPES, dt))
    B, H, D, lk, nmax = twin.B, twin.H, twin.D, twin.lk_max, twin.n_max
    shapes = dict(comp_k=(B, lk, H, D), ring=(B, RING, H, D), total_cis=(B, nmax, H), ccis=(H, B, lk))
    for name, shape in shapes.items():
        t = getattr(twin, name)
        if tuple(t.shape) != shape or t.dtype != dt or not t.is_contiguous() or t.device != twin.pos.device:
            raise ValueError("twin.%s must be a contiguous %s %s tensor on %s" % (name, shape, dt, twin.pos.device))
    for name in ("pos", "n_comp", "n_ccis", "epoch"):
        t = getattr(twin, name)
        if tuple(t.shape) != (B,) or t.dtype != torch.int64 or not t.is_contiguous() or t.device != twin.pos.device:
            raise ValueError("twin.%s must be a contiguous (%d,) int64 tensor" % (name, B))
    if tuple(twin.prefilled.shape) != (B,) or twin.prefilled.dtype != torch.bool or not twin.prefilled.is_contiguous():
        raise ValueError("twin.prefilled must be a contiguous (%d,) bool tensor" % B)
    ovf = getattr(twin, "overflow", None)
    if ovf is None or tuple(ovf.shape) != () or ovf.dtype != torch.bool or ovf.device != twin.pos.device:
        raise ValueError("twin.overflow must be a 0-d bool tensor on the twin's device (PositionalTwin of 4fe5fc9 on)")


def _launch(twin, keys4, cis3, T: int, active=None, n=None, ctr=None, plant: int = 0, num_warps: int = 4) -> None:
    if triton is None:
        raise RuntimeError("triton is not installed")
    dt = twin.ring.dtype
    if keys4.dtype != dt:
        keys4 = keys4.to(dt)                                  # step_unchecked's keys.to(dtype) (no-op when equal)
    if cis3.dtype != dt:
        cis3 = cis3.to(dt)
    dummy = twin.pos
    _twin_kernel[(twin.B,)](
        twin.pos, twin.n_comp, twin.n_ccis, twin.epoch, twin.prefilled.view(torch.uint8),
        twin.overflow.reshape(1).view(torch.uint8),
        _bits(twin.comp_k), _bits(twin.ring), _bits(twin.total_cis), _bits(twin.ccis),
        _bits(keys4), _bits(cis3),
        active.view(torch.uint8) if active is not None else dummy, n if n is not None else dummy,
        ctr if ctr is not None else dummy,
        int(T), twin.B, twin.n_max, twin.lk_max,
        keys4.stride(0), keys4.stride(1), keys4.stride(2), keys4.stride(3), cis3.stride(0), cis3.stride(1),
        cis3.stride(2),
        H=twin.H, D=twin.D, HP=_pow2(twin.H), DP=_pow2(twin.D), WINC=WIN, STRC=STRIDE, RINGC=RING,
        FROM_N=n is not None, WRITE_CTR=ctr is not None, BF16=dt == torch.bfloat16, PLANT=int(plant),
        num_warps=num_warps)


def step_fused(twin, keys: torch.Tensor, cis: torch.Tensor, active: torch.Tensor, plant: int = 0) -> None:
    """PositionalTwin.step_unchecked in ONE kernel (module docstring). keys (B, H, D), cis (B, H), active (B,) bool on
    the twin's device. plant: a NEGATIVE CONTROL (PLANTS), tests only."""
    check_twin(twin)
    dev = twin.pos.device
    if tuple(keys.shape) != (twin.B, twin.H, twin.D) or tuple(cis.shape) != (twin.B, twin.H):
        raise ValueError("keys (%d, %d, %d) and cis (%d, %d) expected" % (twin.B, twin.H, twin.D, twin.B, twin.H))
    if not torch.is_tensor(active) or active.dtype != torch.bool or tuple(active.shape) != (twin.B,) \
            or active.device != dev:
        raise ValueError("active must be a (%d,) bool tensor on %s" % (twin.B, dev))
    if keys.device != dev or cis.device != dev:
        raise ValueError("keys and cis must be on %s" % dev)
    _launch(twin, keys.unsqueeze(1), cis.unsqueeze(1), 1, active=active, plant=plant)


def advance_fused(twin, keys: torch.Tensor, cis: torch.Tensor, n: torch.Tensor, out: torch.Tensor = None,
                  plant: int = 0) -> torch.Tensor:
    """PositionalTwin.advance_unchecked in ONE kernel: keys (B, T, H, D), cis (B, T, H), n (B,) int64 on the twin's
    device. Returns ctr (T + 1, 4, B) int64 (into out when given: a preallocated (T + 1, 4, B) int64 tensor)."""
    check_twin(twin)
    dev = twin.pos.device
    if keys.dim() != 4 or keys.shape[0] != twin.B or tuple(keys.shape[2:]) != (twin.H, twin.D):
        raise ValueError("keys (B, T, H, D) expected")
    T = int(keys.shape[1])
    if tuple(cis.shape) != (twin.B, T, twin.H):
        raise ValueError("cis (B, T, H) expected")
    if not torch.is_tensor(n) or n.dtype != torch.int64 or tuple(n.shape) != (twin.B,) or n.device != dev:
        raise ValueError("n must be a (%d,) int64 tensor on the twin's device" % twin.B)
    if keys.device != dev or cis.device != dev:
        raise ValueError("keys and cis must be on %s" % dev)
    if out is None:
        out = torch.empty((T + 1, 4, twin.B), dtype=torch.int64, device=dev)
    elif tuple(out.shape) != (T + 1, 4, twin.B) or out.dtype != torch.int64 or not out.is_contiguous() \
            or out.device != dev:
        raise ValueError("out must be a contiguous (%d, 4, %d) int64 tensor on %s" % (T + 1, twin.B, dev))
    _launch(twin, keys, cis, T, n=n.contiguous(), ctr=out, plant=plant)
    return out


def attach(twin, methods=("step_unchecked", "advance_unchecked"), require_fixed_order=True):
    """Bind the fused versions on the INSTANCE (the class is unchanged): step_unchecked -> step_fused, advance_unchecked
    -> advance_fused, and, when named, step_fixed -> step_fused (a DECLARED change: module docstring). With
    require_fixed_order, the twin's mean_fn must be fixed_order_mean (the bits are then those of the torch ops)."""
    check_twin(twin)
    if require_fixed_order and getattr(getattr(twin, "mean_fn", None), "__name__", None) != "fixed_order_mean":
        raise ValueError("the fused step is bitwise the torch ops only with mean_fn = fixed_order_mean (got %r); pass "
                         "require_fixed_order=False to declare the change" % (getattr(twin, "mean_fn", None),))
    table = dict(step_unchecked=step_fused, advance_unchecked=advance_fused, step_fixed=step_fused)
    for m in methods:
        if m not in table:
            raise ValueError("cannot attach %r (one of %s)" % (m, sorted(table)))
        setattr(twin, m, functools.partial(table[m], twin))
    return twin
