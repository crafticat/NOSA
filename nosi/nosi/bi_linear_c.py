# Adapted from SGLang python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (matmul_kernel_persistent; SGLang
# commit 87db743, read in the local checkout /mnt/beegfs/ojerbi/scratch_research/hisparse_sglang_kyFDRo/sglang), which
# is itself adapted from Thinking Machines Lab's batch_invariant_ops (https://github.com/thinking-machines-lab/
# batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py), through G-a (nosi-r2-m2-bigemm 26c45bd,
# nosi/nosi/bi_linear.py). All are licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0); this file is distributed under the same license.
# Modifications (G-c, 2026-10-21): no split-K in production; the reduction CONTRACT (BLOCK_K 64, split 1, ascending K,
# one rounding) separated from the LAUNCH parameters (BLOCK_M, BLOCK_N, warps, stages, GROUP_M); a launch table keyed
# on (weight shape, M bucket) and a STRICT one-config-per-shape variant; a GROUP_M swizzle; a non-persistent grid; the
# weight read in place as F.linear's (N, K); negative-control launches (K range, K reversed) for the gates only.
"""G-c: the BATCH-INVARIANT linear with a fixed REDUCTION CONTRACT (design note
/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/bi-gemm-design.md, option (a)). y = x @ W.T, F.linear's meaning, bf16.

CLAIM R (design note section 2). The bits of y[i, j] are a function of x[i, :], W[j, :] and the contract only:
  1. split-K = 1: one fp32 accumulator per output element, no partial sums from other programs;
  2. the K loop walks K in ascending BLOCK_K = 64 chunks; each chunk is tl.dot's sequence of MMA k-steps;
  3. one MMA instruction (sm80 MMAv2 m16n8k16 bf16 x bf16 -> fp32): every tile dim >= 16, so Triton never takes its
     FMA path;
  4. one fp32 -> bf16 rounding in the epilogue (no bias, nothing that reads other rows);
  5. no split of K across warps inside a program (Triton places warps on M and N only).
The design note argues that BLOCK_M, BLOCK_N, num_warps, num_stages and GROUP_M choose only WHICH program and warp
computes (i, j) and WHEN. THAT IS AN ARGUMENT, NOT AN ESTABLISHED FACT: a fixed K loop alone does not establish the
same per-output accumulation, precision, rounding and epilogue under other tile / warp choices (the Triton lowering may
differ per launch). Only a GPU test of bitwise row equality across M, for the variant under test, is evidence
(bigc_gate.py G2 / G2w / G2g), and it holds only for the shapes, M values, inputs, GPU and Triton version it ran.

TWO VARIANTS, both behind --gemm; the default (EXACT = G-b PaddedLinear) is unchanged. Human direction 2026-10-21
(binding): the relaxation of the batch-invariance law is NOT approved.
  bi_c_linear_strict  THE CANDIDATE. launch = strict_launch_for(N, K): ONE launch per weight shape for every M (no
                      config chosen by M; the law as it stands). --gemm bi_c_strict.
  bi_c_linear_bucket  RESEARCH ARM ONLY (opt-in, never the default, never cited as satisfying the contract): launch =
                      launch_for(N, K, M), a table keyed on (weight shape, M bucket). It exists to measure what the
                      relaxed law would buy if the human approves it later. --gemm bi_c_bucket.
REFERENCE CHANGE. G-c's arithmetic differs from the current BI target reference (G-b PaddedLinear / cuBLAS bits).
Adopting G-c is an explicit change of the declared target reference: G0-BI agreement and SG0-SG3 must re-run against
the new reference; the old reference and its tests stay as they are.

INTERFACE
    CONTRACT; contract_for(N, K) -> Contract (no M input)
    strict_launch_for(N, K, table=None) -> Launch; launch_for(N, K, M, table=None) -> Launch (research arm)
    bi_c_linear_strict(x, weight, out=None); bi_c_linear_bucket(x, weight, out=None) (research arm)
        x (..., K) bf16 on CUDA (float32 only on CPU: the interpreter; on CUDA an fp32 tl.dot would run as TF32,
        outside the contract), weight (N, K) contiguous, out optional (..., N) in x's dtype. No host sync; with `out`
        and a unit last stride of x, no allocation.
    bi_c_reference(x, weight, split_k=1, reverse_k=False) -> the CPU TWIN: per row, BLOCK_K chunks in order, each chunk
        as k16 steps in order, fp32 accumulate, one rounding (split_k / reverse_k: the negative controls' orders).
    control_split_k(x, weight, launch, splits=2), control_k_reversed(x, weight, launch): NEGATIVE CONTROLS (gates and
        tests only; a production caller never uses them).
    TABLE (the source table: "strict" per shape, and the research arm's buckets), table_digest(table),
        check_table(table), load_table(path, sha256), use_table(table). TABLE is the UNTUNED prior until
        bigc_tune.py's frozen output replaces it (TABLE["tuned"] and table_digest(TABLE) say which one runs).
"""
from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Optional

import torch

try:                                                   # triton is needed only to launch; the twin, the tables and the
    import triton                                      # tests import this module without it
    import triton.language as tl
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None


class Contract(NamedTuple):
    block_k: int
    split_k: int
    k_order: str
    mma: str
    rounding: str


class Launch(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    group_m: int


BLOCK_K = 64
CONTRACT = Contract(block_k=BLOCK_K, split_k=1, k_order="ascending",
                    mma="sm80 MMAv2 mma.sync m16n8k16 bf16 x bf16 -> fp32 (every tile dim >= 16)",
                    rounding="one fp32 -> output-dtype rounding, no bias")

# NOSA-8B's decode linears (nosa_llama.py: F.linear on the fused weights; no bias); bi_linear.NOSA8B_SHAPES of G-a
NOSA8B_SHAPES = {
    "wqkv": (4608, 4096),
    "wo": (4096, 4096),
    "gate_up": (32768, 4096),
    "down": (4096, 16384),
    "lm_head": (73448, 4096),
}
TILE_M = (16, 32, 64, 128)
TILE_N = (16, 32, 64, 128, 256)
WARPS = (4, 8)
STAGES = (2, 3, 4, 5)


def shape_key(N: int, K: int) -> str:
    return "%dx%d" % (int(N), int(K))


# THE SOURCE TABLE: the UNTUNED PRIOR (design note 3 (a): small BLOCK_N at small M for parallelism without split-K;
# vLLM / TML's 128 x 128 x 64, 8 warps, 3 stages at large M; BLOCK_M 64 above 512 rows for the wave count on 108 SMs).
# Entry = [BLOCK_M, BLOCK_N, num_warps, num_stages, GROUP_M]. "strict" / "strict_default": THE CANDIDATE's one launch
# per weight shape. "buckets" / "shapes" / "default": the RESEARCH ARM only ("buckets" are the inclusive upper row
# counts of the first len(buckets) buckets; the last bucket is open). bigc_tune.py replaces "strict" and "shapes" for
# the five NOSA-8B shapes, adds "bucket_invariant" and sets "tuned": true; the digest of the frozen table is recorded
# with every run. To freeze it into the source, paste table.json's content here (the digest must not change).
_NARROW = [[16, 32, 4, 4, 8], [32, 64, 4, 4, 8], [64, 64, 4, 4, 8], [128, 128, 8, 3, 8], [64, 128, 4, 4, 8]]
_WIDE = [[16, 64, 4, 4, 8], [32, 128, 4, 4, 8], [64, 128, 4, 4, 8], [128, 128, 8, 3, 8], [128, 128, 8, 3, 8]]
TABLE = {
    "version": 1,
    "tuned": False,
    "source": "untuned prior (bi-gemm-design.md 3 (a)); replace with bigc_tune.py's frozen table",
    "contract": list(CONTRACT),
    "buckets": [16, 64, 256, 512],
    "shapes": {shape_key(*NOSA8B_SHAPES["wqkv"]): _NARROW, shape_key(*NOSA8B_SHAPES["wo"]): _NARROW,
               shape_key(*NOSA8B_SHAPES["down"]): _NARROW, shape_key(*NOSA8B_SHAPES["gate_up"]): _WIDE,
               shape_key(*NOSA8B_SHAPES["lm_head"]): _WIDE},
    "default": _NARROW,
    "strict": {shape_key(*s): [64, 128, 4, 4, 8] for s in NOSA8B_SHAPES.values()},
    "strict_default": [64, 128, 4, 4, 8],
}
_ACTIVE = {"table": TABLE}


def contract_for(N: int, K: int) -> Contract:
    """The reduction contract of a weight shape: the SAME for every shape (and never a function of M)."""
    if int(N) <= 0 or int(K) <= 0:
        raise ValueError("contract_for(N=%d, K=%d)" % (int(N), int(K)))
    return CONTRACT


def table_digest(table: Optional[dict] = None) -> str:
    """sha256 of the table's canonical JSON (sorted keys, no spaces). Computed where it runs (a compute node)."""
    t = _ACTIVE["table"] if table is None else table
    return hashlib.sha256(json.dumps(t, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_entry(e, where):
    if not (isinstance(e, (list, tuple)) and len(e) == 5 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                  for v in e)):
        raise ValueError("%s: an entry is [BLOCK_M, BLOCK_N, warps, stages, GROUP_M] ints, not %r" % (where, e))
    bm, bn, w, s, g = e
    if bm not in TILE_M or bn not in TILE_N or w not in WARPS or s not in STAGES or g < 1:
        raise ValueError("%s: %r outside BLOCK_M %s, BLOCK_N %s, warps %s, stages %s, GROUP_M >= 1"
                         % (where, e, TILE_M, TILE_N, WARPS, STAGES))


def check_table(table: dict) -> dict:
    """Refuse a table that breaks the contract or the bucket shape. Returns the table."""
    if list(table.get("contract", [])) != list(CONTRACT):
        raise ValueError("table contract %r is not CONTRACT %r" % (table.get("contract"), list(CONTRACT)))
    b = table.get("buckets")
    if not (isinstance(b, list) and b and all(isinstance(x, int) and x >= 1 for x in b)
            and all(b[i] < b[i + 1] for i in range(len(b) - 1))):
        raise ValueError("buckets must be increasing positive ints: %r" % (b,))
    rows = dict(table.get("shapes", {}), default=table.get("default"))
    for k, lst in rows.items():
        if not isinstance(lst, list) or len(lst) != len(b) + 1:
            raise ValueError("shape %s: %d buckets need %d entries" % (k, len(b), len(b) + 1))
        for e in lst:
            _check_entry(e, "shape %s" % k)
    for k, e in dict(table.get("strict", {}), default=table.get("strict_default")).items():
        _check_entry(e, "strict %s" % k)
    if not isinstance(table.get("tuned"), bool):
        raise ValueError("table['tuned'] must be a bool")
    return table


def load_table(path: str, sha256: str) -> dict:
    """A frozen table file (bigc_tune.py's output 'table.json'), refused unless its canonical digest is sha256."""
    with open(path) as f:
        t = json.load(f)
    d = table_digest(t)
    if d != sha256:
        raise ValueError("table %s has digest %s, not the pinned %s" % (path, d, sha256))
    return check_table(t)


def use_table(table: dict) -> str:
    """Make `table` the active one (gates and the timing cell: a frozen file before it is committed). Returns its
    digest."""
    _ACTIVE["table"] = check_table(table)
    return table_digest(table)


def active_table() -> dict:
    return _ACTIVE["table"]


def bucket_index(M: int, buckets) -> int:
    """The bucket of M: the first i with M <= buckets[i], else len(buckets)."""
    M = int(M)
    for i, edge in enumerate(buckets):
        if M <= edge:
            return i
    return len(buckets)


def launch_for(N: int, K: int, M: int, table: Optional[dict] = None) -> Launch:
    """G-c's launch: (weight shape, M bucket) -> Launch. The contract does not depend on it."""
    t = _ACTIVE["table"] if table is None else table
    lst = t["shapes"].get(shape_key(N, K), t["default"])
    return Launch(*lst[bucket_index(M, t["buckets"])])


def strict_launch_for(N: int, K: int, table: Optional[dict] = None) -> Launch:
    """The strict variant's launch: a function of the weight shape only."""
    t = _ACTIVE["table"] if table is None else table
    return Launch(*t["strict"].get(shape_key(N, K), t["strict_default"]))


if triton is not None:
    @triton.jit(do_not_specialize=["M"])
    def _bic_kernel(X, W, OUT, M, N, s_xm, s_xk, s_wn, s_wk, s_om, s_on, K_LO, K_HI, K_SPAN,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                    REVERSE_K: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_in_group = GROUP_M * num_pid_n                 # GROUP_M swizzle: WHICH program computes a tile, never the
        group_id = pid // num_in_group                     # order inside it
        first_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % num_in_group) % group_size_m
        pid_n = (pid % num_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_ok = offs_m < M
        n_ok = offs_n < N
        rows = tl.where(m_ok, offs_m, 0).to(tl.int64)
        cols = tl.where(n_ok, offs_n, 0).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, K_SPAN, BLOCK_K):               # K_SPAN: whole BLOCK_K chunks from K_LO (the last masked)
            if REVERSE_K:                                  # NEGATIVE CONTROL only (control_k_reversed)
                k0 = K_LO + K_SPAN - BLOCK_K - kk
            else:                                          # THE CONTRACT: ascending chunks
                k0 = K_LO + kk
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_ok = offs_k < K_HI
            x = tl.load(X + rows[:, None] * s_xm + offs_k[None, :] * s_xk, mask=m_ok[:, None] & k_ok[None, :],
                        other=0.0)
            w = tl.load(W + cols[None, :] * s_wn + offs_k[:, None] * s_wk, mask=n_ok[None, :] & k_ok[:, None],
                        other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(OUT + rows[:, None] * s_om + cols[None, :] * s_on, acc.to(OUT.dtype.element_ty),
                 mask=m_ok[:, None] & n_ok[None, :])
else:                                                  # pragma: no cover
    _bic_kernel = None


def _check(x2, weight):
    if weight.dim() != 2 or x2.dim() != 2 or x2.shape[1] != weight.shape[1]:
        raise ValueError("x (..., K) and weight (N, K): got %s and %s" % (tuple(x2.shape), tuple(weight.shape)))
    if x2.dtype != weight.dtype:
        raise TypeError("x %s and weight %s dtypes differ" % (x2.dtype, weight.dtype))
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (N, K): F.linear's layout")
    if x2.device != weight.device:
        raise ValueError("x and weight on different devices")
    if x2.dtype == torch.float32 and x2.device.type == "cuda":
        raise TypeError("float32 on CUDA would run tl.dot as TF32: outside the contract (bf16 / fp16 only)")


def _launch(x2, weight, o2, launch: Launch, k_lo: int, k_hi: int, block_k: int = BLOCK_K, reverse_k: bool = False):
    """One kernel launch over the K range [k_lo, k_hi) (production: [0, K)). Returns Triton's launch handle."""
    M, N = x2.shape[0], weight.shape[0]
    span = -(-(k_hi - k_lo) // block_k) * block_k
    grid = (triton.cdiv(M, launch.block_m) * triton.cdiv(N, launch.block_n),)
    return _bic_kernel[grid](x2, weight, o2, M, N, x2.stride(0), x2.stride(1), weight.stride(0), weight.stride(1),
                             o2.stride(0), o2.stride(1), k_lo, k_hi, span,
                             BLOCK_M=launch.block_m, BLOCK_N=launch.block_n, BLOCK_K=block_k, GROUP_M=launch.group_m,
                             REVERSE_K=reverse_k, num_warps=launch.num_warps, num_stages=launch.num_stages)


def _prepare(x, weight, out):
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    _check(x2, weight)
    N = weight.shape[0]
    if out is None:
        out = torch.empty((*lead, N), dtype=x.dtype, device=x.device)
    elif out.shape != (*lead, N) or out.dtype != x.dtype:
        raise ValueError("out must be %s in %s" % (((*lead, N),), x.dtype))
    return x2, out, out.view(-1, N)


def _run(x, weight, out, launch_of):
    if _bic_kernel is None:
        raise RuntimeError("triton is not installed: use bi_c_reference")
    x2, out, o2 = _prepare(x, weight, out)
    M, K = x2.shape
    if M == 0:
        return out
    _launch(x2, weight, o2, launch_of(weight.shape[0], K, M), 0, K)
    return out


def bi_c_linear_strict(x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """G-c STRICT, THE CANDIDATE: y = x @ weight.T under CONTRACT with ONE launch per weight shape for every M."""
    return _run(x, weight, out, lambda N, K, M: strict_launch_for(N, K))


def bi_c_linear_bucket(x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RESEARCH ARM ONLY (the relaxed law is NOT approved): CONTRACT order, launch = launch_for(N, K, M) chosen by M.
    Never the default; never cited as satisfying the batch-invariance contract."""
    return _run(x, weight, out, launch_for)


def bi_c_with_launch(x, weight, launch: Launch, out=None, block_k: int = BLOCK_K):
    """Tests, the gate's reports and the tuning cell only: CONTRACT order under a GIVEN launch (block_k other than
    BLOCK_K: the gate's report R1 only)."""
    if _bic_kernel is None:
        raise RuntimeError("triton is not installed: use bi_c_reference")
    x2, out, o2 = _prepare(x, weight, out)
    if x2.shape[0]:
        _launch(x2, weight, o2, launch, 0, x2.shape[1], block_k=block_k)
    return out


def control_k_reversed(x, weight, launch: Launch):
    """NEGATIVE CONTROL: the K chunks in DESCENDING order (breaks contract item 2)."""
    if _bic_kernel is None:
        raise RuntimeError("triton is not installed")
    x2, out, o2 = _prepare(x, weight, None)
    if x2.shape[0]:
        _launch(x2, weight, o2, launch, 0, x2.shape[1], reverse_k=True)
    return out


def split_bounds(K: int, splits: int, block_k: int = BLOCK_K):
    """[(k_lo, k_hi)]: K cut into `splits` ranges of whole BLOCK_K chunks (the last range may be shorter)."""
    chunks = -(-K // block_k)
    per = -(-chunks // splits)
    out = []
    for s in range(splits):
        lo, hi = s * per * block_k, min((s + 1) * per * block_k, K)
        if lo < hi:
            out.append((lo, hi))
    return out


def control_split_k(x, weight, launch: Launch, splits: int = 2):
    """NEGATIVE CONTROL: split-K (breaks contract item 1): each K range into an fp32 partial, the partials added in
    range order in fp32, one rounding to x's dtype."""
    if _bic_kernel is None:
        raise RuntimeError("triton is not installed")
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).contiguous()
    _check(x2, weight)
    M, N = x2.shape[0], weight.shape[0]
    total = None
    for lo, hi in split_bounds(K, splits):
        part = torch.empty((M, N), dtype=torch.float32, device=x.device)
        if M:
            _launch(x2, weight, part, launch, lo, hi)
        total = part if total is None else total + part
    return total.to(x.dtype).view(*lead, N)


def bi_c_reference(x: torch.Tensor, weight: torch.Tensor, split_k: int = 1, reverse_k: bool = False,
                   block_k: int = BLOCK_K) -> torch.Tensor:
    """The CPU TWIN (tests only; host loops): per row, per K range (split order), its BLOCK_K chunks in ascending
    order (descending with reverse_k), each chunk as k16 steps in ascending order, accumulated in fp32; the range
    partials added in order; one rounding to x's dtype. The same call shape for every row and every M."""
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    if x2.device.type != "cpu":
        raise ValueError("bi_c_reference is the CPU twin: CPU tensors only")
    _check(x2, weight)
    M, N = x2.shape[0], weight.shape[0]
    f32 = torch.float32
    wf = weight.to(f32)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    ranges = split_bounds(K, split_k, block_k)
    for i in range(M):
        xi = x2[i].to(f32)
        total = None
        for lo, hi in ranges:
            starts = list(range(lo, hi, block_k))
            if reverse_k:
                starts = starts[::-1]
            acc = torch.zeros(N, dtype=f32, device=x.device)
            for c0 in starts:
                for k in range(c0, min(c0 + block_k, hi), 16):
                    k1 = min(k + 16, c0 + block_k, hi)
                    acc = acc + xi[k:k1].unsqueeze(0).matmul(wf[:, k:k1].t()).squeeze(0)
            total = acc if total is None else total + acc
        out[i] = total.to(x.dtype)
    return out.view(*lead, N)


def plant_order_rows(x: torch.Tensor, weight: torch.Tensor, rows, cols, block_k: int = BLOCK_K) -> None:
    """IN PLACE: make rows `rows` x cols `cols` of the product ORDER-SENSITIVE (the negative controls' input set,
    lesson L34). With n = K / block_k >= 4 chunks and h = the first chunk of the second range of split_bounds(K, 2)
    (h + 1 < n): x[r] is 1 at the first column of chunks 0, h and h + 1 and 0 elsewhere; w[c] holds 2**24, 1, -2**24
    there and 0 elsewhere.
    The contract (ascending, one accumulator) gives (2**24 + 1) - 2**24 = 0 (2**24 + 1 rounds to 2**24 in fp32);
    split-K 2 gives 2**24 + (1 - 2**24) = 1; K reversed gives (-2**24 + 1) + 2**24 = 1. Every value and partial is
    exact in bf16 / fp32, so the 0 / 1 difference survives the bf16 output rounding."""
    K = x.shape[-1]
    n = -(-K // block_k)
    if n < 4:
        raise ValueError("the plant needs >= 4 BLOCK_K chunks (K %d)" % K)
    h = split_bounds(K, 2, block_k)[1][0] // block_k
    ks = (0, h * block_k, (h + 1) * block_k)
    vals = (2.0 ** 24, 1.0, -(2.0 ** 24))
    for r in rows:
        x[r].zero_()
        for k in ks:
            x[r, k] = 1.0
    for c in cols:
        weight[c].zero_()
        for k, v in zip(ks, vals):
            weight[c, k] = v


VARIANTS = {"strict": "G-c STRICT (the candidate: one launch per weight shape for every M)",
            "bucket": "G-c BUCKET RESEARCH ARM (launch chosen by M; relaxed law NOT approved; not a contract claim)"}
REFERENCE_NOTE = ("G-c is a NEW target reference: its bits differ from G-b PaddedLinear's; G0-BI agreement and "
                  "SG0-SG3 must re-run against it; the old reference and its tests are unchanged")


def describe(variant: str) -> dict:
    """Run records: the variant's label, the contract, the active table's digest and whether it is tuned, and the
    reference-change note."""
    if variant not in VARIANTS:
        raise ValueError("variant %r not in %s" % (variant, sorted(VARIANTS)))
    t = _ACTIVE["table"]
    return dict(variant=variant, label=VARIANTS[variant], contract=CONTRACT._asdict(), table_digest=table_digest(t),
                table_tuned=t["tuned"], buckets=list(t["buckets"]) if variant == "bucket" else None,
                reference=REFERENCE_NOTE, triton=getattr(triton, "__version__", None))
