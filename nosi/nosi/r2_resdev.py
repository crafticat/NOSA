"""[resdev] --res-dev (OPT-IN, needs --res-lean): the torch resolve's DEVICE hot spots, rewritten with equal integer
results (analysis/evalfast/SCORING_resfix.md 2.3 and 3.4). The --res-lean ops stay the REFERENCE: each lean function
runs the op below in place of its lean op only when its LeanConsts has dev True (ResidencySC.enable_res_dev).

Every function here gives the SAME values as the lean op it replaces (integer ops only; tests_cpu/test_r2_resdev.py:
RD1 compares them on random inputs, edge shapes included, RD2 the device functions, RD3 whole rounds, RD4 the negative
controls):
  excl_scan_rows(x)          x.cumsum(-1) - x for a 2-D int64 x. The lean scan of a (2, 347,328) tensor runs one
                             block per row (scan_innermost_dim, 0.81 ms per call at B 216); here ONE 1-D scan over the
                             flat tensor (a device-wide cub scan) minus each row's start total.
  group_start(first, pos)    torch.cummax(torch.where(first, pos, 0), 0).values for a 1-D bool `first` with first[0]
                             True and pos = arange(n): the start of each entry's group. The lean 1-D cummax runs on one
                             block; here a 1-D cub scan (the group ids), one scatter with DISTINCT indices and a gather.
  sorted_free(mask)          torch.sort(torch.where(mask, ar, NP), dim=-1).values for a 2-D bool mask (S, NP) and
                             ar = arange(NP): per row the True columns ascending, then NP. Here a compaction (the
                             exclusive scan above, one scatter with distinct indices): no sort.
  sort_key32(key, bound)     torch.sort(key, stable=True) with the key cast to int32 when every key value lies in
                             [0, bound] and bound < 2**31 - 1 (the caller proves the bound from host ints): a stable sort
                             gives the same permutation for any order-preserving key map; int32 keys need 4 radix
                             passes, not 8. The sorted keys come back as int32 (callers only compare them).
  spread(idx, ok, base, n)   where(ok, idx, base + arange(n)): every entry that does not write a real element writes
                             its OWN dummy element (base + its index), not ONE shared dummy. The lean ops send every
                             such entry to one address (hundreds of thousands of writes to one element per call). Only
                             the dummy region changes (it is never read); the real elements get the same writes.

DETERMINISM: every scatter / index_put_ here has distinct target indices, or only equal values per target, so the
result does not depend on the write order (the lean dummy element held an order-dependent value; nothing reads it).
"""
from __future__ import annotations

import torch

INT32_MAX = (1 << 31) - 1


def excl_scan_rows(x: torch.Tensor) -> torch.Tensor:
    """[resdev] x.cumsum(-1) - x for a 2-D integer x (R, N), with ONE flat 1-D scan."""
    R, N = x.shape
    if R == 0 or N == 0:
        return x.cumsum(-1) - x
    xf = x.reshape(-1)
    c = xf.cumsum(0).view(R, N)                                     # inclusive scan over the rows in order
    base = torch.cat([c.new_zeros(1), c[:-1, -1]]).view(R, 1)        # the total of the rows before each row
    return c - base - x


def group_start(first: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """[resdev] torch.cummax(torch.where(first, pos, 0), 0).values for first (n,) bool with first[0] True and pos =
    arange(n) (int64): the index of the latest True at or before each entry. gid = (Trues so far) - 1; the Trues write
    their position to starts[gid] (distinct: one True per group), the others to their own dummy n + i; gstart =
    starts[gid]."""
    n = first.shape[0]
    if n == 0:
        return pos[:0].clone()
    gid = first.to(torch.int64).cumsum(0) - 1
    starts = torch.empty(2 * n, dtype=pos.dtype, device=pos.device)
    starts.index_put_((torch.where(first, gid, pos + n),), pos)
    return starts[gid]


def sorted_free(mask: torch.Tensor, flat: torch.Tensor = None) -> torch.Tensor:
    """[resdev] torch.sort(torch.where(mask, ar, NP), dim=-1).values for mask (S, NP) bool, ar = arange(NP) (int64): per
    row the True columns ascending, then NP. The q-th True column of row s goes to column q (q = its exclusive count);
    every False entry writes its own dummy element S * NP + (its flat index). flat: arange(S * NP).view(S, NP) if the
    caller caches it (a tick constant), else built here."""
    S, NP = mask.shape
    dev = mask.device
    if S == 0 or NP == 0:
        return torch.empty((S, NP), dtype=torch.int64, device=dev)
    m64 = mask.to(torch.int64)
    q = excl_scan_rows(m64)
    if flat is None:
        flat = torch.arange(S * NP, dtype=torch.int64, device=dev).view(S, NP)
    row0 = flat[:, :1]                                               # s * NP
    col = flat - row0                                                # the column index (ar)
    out = torch.full((2 * S * NP,), NP, dtype=torch.int64, device=dev)
    out.index_put_((torch.where(mask, row0 + q, flat + S * NP).reshape(-1),), col.reshape(-1))
    return out[:S * NP].view(S, NP)


def sort_key32(key: torch.Tensor, bound: int):
    """[resdev] torch.sort(key, stable=True) with an int32 key when 0 <= key <= bound < 2**31 - 1 is PROVEN by the
    caller (a host bound). Returns (sorted keys, perm); the keys are int32 then (else as the lean sort gives them)."""
    if 0 <= bound < INT32_MAX:                                       # a Python int (the caller's host bound)
        return torch.sort(key.to(torch.int32), stable=True)
    return torch.sort(key, stable=True)


def spread(idx: torch.Tensor, ok: torch.Tensor, base: int, ar: torch.Tensor) -> torch.Tensor:
    """[resdev] torch.where(ok, idx, base + ar): the entries that do not write a real element get their own dummy
    element (ar = arange(idx.numel()) shaped like idx). The buffer must have base + idx.numel() elements."""
    return torch.where(ok, idx, ar + base)
