"""R2 FIATTN: a BOUNDED comparison of an EXISTING kernel against piece 3 v2, BEFORE any switch.

Existing kernel: FlashInfer 0.5.3 (installed in nosa-env.sif; NOT upgraded), BatchDecodeWithPagedKVCacheWrapper with
use_tensor_cores=True and backend "fa2" (it runs the FA2 paged prefill kernel with one query row per request), with a
JIT custom attention variant for the CIS bias. Our kernel: piece 3 v2 (nosi-r2-m2-biattn 7c826c1,
nosi/nosi/bi_paged_attention.py, blob 39ed5e18), loaded BY PATH from a pinned copy. Design of the comparison:
/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/attn-reuse-assessment.md (section 5, item 10, open items). Plan and the
REGISTERED predictions: docs/R2_FIATTN_PLAN.md (PRED below is the same table; test FA8 keeps them equal).
Standalone: production, the reference and the tick are NOT touched; nothing here is wired into the tick.
SOURCE ONLY, NOT RUN.

ARMS
  P   piece 3 v2: bi_paged_decode (fixed split, 8 pages per split; its own fp32 partials and combine).
  F8  FlashInfer, OPTION A (two calls per layer: one per KV head, num_kv_heads 1, the K / V view k[..., h:h+1, :]),
      plan(fixed_split_size=8) (= piece 3's PAGES_PER_SPLIT).
  F0  the same with plan(disable_split_kv=True) (one chunk per request: the CUDA-graph-safe form).
  FH  the same with FlashInfer's default split heuristic: a NEGATIVE CONTROL for the invariance gates only.
  S5  optional, rows-per-request 4 cells: sharedkv5 (nosi-r2-sharedkv5 5eb7514, blob 5c2e7b25) tile "reload", mode
      "split" (REPORT only).
  STRIDED (REPORT probe): OPTION B, one FlashInfer call per layer through an OVERLAPPING as_strided view of the store
      (page id = (entry * S + slot) * 64 * Hkv + h, stride_page = D, stride_n = Hkv * D).
  A FlashInfer "request" is one (row, KV head). Page ids of option A: entry * S + slot (the store viewed as
  (N * S, 64, Hkv, D) pages). Packing is device ops only (cumsum + scatter into an R * P buffer; no host read);
  plan() itself reads indptr / last_page_len on the HOST (FlashInfer's design).

THE CIS BIAS IN FLASHINFER: the variant R2FiCisBias (VARIANT_DECL) follows the ALiBi branch of FlashInfer's default
variant (logits arrive unscaled; the variant scales them and sets sm_scale_log2 = log2(e)):
    logit = (q.k + bias[page * page_mul + entry * entry_stride + head]) * sm_scale,
with page = indices[indptr[request] + kv_idx / 64], entry = kv_idx % 64, read only for kv_idx < kv_len (positions past
the request's extent are masked by the kernel; the guard keeps the read in bounds). Option A: page_mul = 64 * Hkv,
entry_stride = Hkv, head = h; option B: 1, Hkv, 0. If the JIT build fails, that is a FINDING (gate C0 fails) and the
run continues BIAS-FREE (bias = 0 for every arm, FlashInfer's default module), every other gate a REPORT.

CASES (fixed, seeded; make_case): NOSA-8B geometry read from --config (q heads, KV heads, head_dim), 16K context:
every row's tail block is 251 (positions 16064 .. 16127), S = 256 slots per storage entry with a seeded block -> slot
map, 64 pages of 64 per (row, KV head): block 0, the 15 recent blocks 236 .. 250, 47 scattered blocks drawn with
weight 1 / sqrt(251 - b) (locality), and the tail. Rows of one request (rows per request 1 or 4) sit at consecutive
positions; row u > 0 swaps 4 scattered blocks of the request's base list. bf16 q / K / V (N(0, 1)), bias uniform in
[-8, 8] (so that the bias moves the logits by about +-0.7: gate C has teeth, witness CW). B in {8, 32} x rows per
request in {1, 4}. Extra batches: "alt" (every odd row has a new q and new lists: siblings changed), "hazard" (the
full batch + 3 short rows of 5, 8 and 9 pages on an extra storage entry: the split_kv switch of the assessment, 2.3
item 5).

GATES (MUST unless noted; P, F8 and F0 each; "bitwise" = equal int16 views of the bf16 outputs)
  C0   the variant JIT build (time charged separately).
  C    max |arm - fp64 reference| <= TOL_C on N_SAMPLE rows (the reference is THIS module's float64 attention, not
       piece 3's code).
  CW   (F8) the bias witness: F8 with bias = 0 is FAR from the biased reference (> TOL_C). Precondition: the biased and
       bias-free references differ by >= WITNESS_FACTOR x TOL_C on the sampled rows, else VOID (= fail).
  I1   alone: each sampled row as a batch of one (its own plan) == the same row in the full batch.
  I2   order: the batch in reversed request order and in a random row order == unpermuted, per row.
  I3   siblings: the even rows of the "alt" batch == the even rows of the full batch.
  I4   the split_kv switch: each short row alone == the same row in the hazard batch, and every full row of the hazard
       batch == its full-batch output. F8 precondition (read from the plans): the rows of <= 8 pages plan split_kv
       false alone and true in the hazard batch, on every KV head; else VOID (= fail).
  I5   poison: every (slot, head) that no row of the batch lists, and every tail row past the largest extent that
       reads it, is NaN in K, V and bias; the full batch stays finite and bitwise equal.
  N    (FH) the negative control: some sampled row alone != batched. Precondition: the heuristic's chunk size (or
       split flag) for the batch of one differs from the full batch's (read from the plans); else VOID. Pass only
       when DETECTED with the precondition met.
  D    REPORT: bits vs piece 3 (rows differing, max |d|); rounding differences never reject.
  T    REPORT: CUDA-event time of the attention call (FlashInfer: both head calls + the output interleave; and the
       runs alone), plan() host time with host indptr and with device indptr (a D2H sync inside plan), packing time,
       JIT build / load time, workspace bytes and the peak allocated delta of one call. Bands from PRED.
  NCU  REPORT (--ncu): one ncu process per cell (all arms of the cell, NVTX range fia|<arm> each), kernel filter
       KERNEL_FILTER (the attention kernels only), --cache-control all (cold L2 per kernel), hard cap min(600 s, the
       budget left); DRAM bytes read / write per arm vs the per-row ideal and the request-union ideal. Label: every
       arm reuses pages through the HARDWARE cache only (L2; S5: a forced temporal locality inside one program), none
       has explicit shared loads (REUSE_LABEL).

BUDGET: --budget-s for the whole process (the step gives 2400 s inside a 45 min step); a cell that would start too
late is skipped and fails gate BUDGET; ncu cells get min(--ncu-cap, the time left - NCU_MARGIN_S) or are skipped
(REPORT).

OUTPUT: --out/summary.json (gates, reports, plans, timings, predictions, environment), one GATE / REPORT line per gate
on stdout. EXIT: the number of failed MUST gates (<= 125); 2 = refused (--out exists, bad arguments, no CUDA, no
FLASHINFER_WORKSPACE_BASE, a FlashInfer version other than 0.5.3).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback

import torch

PAGE = 64                                   # NOSA's block = one page
CTX = 16384                                 # 16K context
TAIL_BLOCK = CTX // PAGE - 5                # 251: the rows' tail block (positions 16064 .. 16127)
S_SLOTS = CTX // PAGE                       # 256 slots per storage entry (blocks 252 .. 255 are never listed)
N_SEL = 64                                  # pages per (row, KV head), the tail included
N_RECENT = 15                               # the recent blocks before the tail
N_SWAP = 4                                  # blocks a verify row swaps against the request's base list
BIAS_RANGE = 8.0                            # bias uniform in [-BIAS_RANGE, BIAS_RANGE]
SHORT_COUNTS = (5, 8, 9)                    # the hazard rows' page counts (5, 8 <= FIXED_SPLIT; 9: two chunks)
SHORT_LAST = 33                             # their extent in the tail page
FIXED_SPLIT = 8                             # = piece 3's PAGES_PER_SPLIT
CELLS = ((8, 1), (8, 4), (32, 1), (32, 4))  # (B, rows per request); 8x1 first: the inspected miniature
ARMS = ("P", "F8", "F0", "FH", "S5")
MUST_ARMS = ("P", "F8", "F0")
FI_VERSION = "0.5.3"
TOL_C = 1.0e-3                              # REGISTERED: max |arm - fp64| (absolute), every arm
WITNESS_FACTOR = 10.0                       # CW precondition: the bias moves the reference by >= 10 x TOL_C
FLOAT_WS_BYTES = 256 << 20                  # FlashInfer float workspace per FlashInfer arm (shared by its 2 wrappers)
N_SAMPLE = 16                               # rows per cell for C, I1 and N
CELL_RESERVE_S = 300                        # a cell starts only with this much budget left
NCU_ORDER = ((32, 4), (32, 1), (8, 4), (8, 1))
NCU_MARGIN_S = 120
NCU_CAP_S = 600
ATTN_KERNELS = ("BatchPrefillWithPagedKVCacheKernel", "PersistentVariableLengthMergeStatesKernel",
                "_bi_split_kernel", "_bi_combine_kernel", "_bi_shared_kernel")
KERNEL_FILTER = "regex:" + "|".join(ATTN_KERNELS)
NCU_REQUIRED = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum")
NCU_OPTIONAL = ("lts__t_sector_hit_rate.pct",)
TAG_RE = re.compile(r"fia\|([A-Z0-9]+)")
KERNELS_PER_CALL = dict(P=2, F8=4, F0=2, FH=None, S5=2)   # F8: 2 heads x (prefill + merge); F0: 2 x prefill
REUSE_LABEL = dict(
    P="hardware cache only: per-row loads; the rows of a request are adjacent programs of one grid (L2 reuse)",
    F8="hardware cache only: one CTA per (row, chunk); requests in request-major order (L2 reuse)",
    F0="hardware cache only: one CTA per row walking all its pages (L2 reuse)",
    FH="hardware cache only (negative control)",
    S5="hardware cache only, forced temporal locality: one program per (request, head, split) reloads each tile per "
       "row (tile 'reload'; L1 / L2), no register or shared-memory sharing",
)
VARIANT_NAME = "R2FiCisBias"
VARIANT_URI = "r2_fiattn_cisbias_bf16_d%d"
ROW_FIELDS = ("q", "seq_idx", "page_table", "page_key", "num_pages", "last_len")
CACHE_FIELDS = ("k", "v", "bias")

# The CIS bias variant (C++ in a Python string, compiled by FlashInfer's JIT inside the GPU step; our text). The
# ragged kernels are instantiated with the same variant and have no paged_kv: the trait keeps them compiling.
VARIANT_DECL = r"""
template <typename T, typename = void>
struct r2fi_has_paged_kv : std::false_type {};
template <typename T>
struct r2fi_has_paged_kv<T, std::void_t<decltype(std::declval<T>().paged_kv)>> : std::true_type {};

struct R2FiCisBias : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ R2FiCisBias(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float b = 0.f;
    if constexpr (r2fi_has_paged_kv<Params>::value) {
      if (kv_idx < kv_len) {
        uint32_t pg, e;
        params.paged_kv.page_size.divmod(kv_idx, pg, e);
        const int64_t page = static_cast<int64_t>(params.paged_kv.indices[params.paged_kv.indptr[batch_idx] + pg]);
        b = __bfloat162float(params.cis_bias[page * params.cis_page_mul +
                                             static_cast<int64_t>(e) * params.cis_entry_stride + params.cis_head]);
      }
    }
    return (logits + b) * static_cast<float>(params.cis_sm_scale);
  })
};
"""
VARIANT_TENSORS = (["cis_bias"], ["nv_bfloat16"])
VARIANT_SCALARS = (["cis_page_mul", "cis_entry_stride", "cis_head", "cis_sm_scale"],
                   ["int64_t", "int64_t", "int64_t", "double"])

# REGISTERED before any run (docs/R2_FIATTN_PLAN.md section 6 carries the same table; the lead stamps it).
# Bands are [central, low, high]. Times in ms (CUDA events, median), per attention call of one layer.
PRED = {
    "registered": "REGISTERED <stamp-by-lead>",
    "tol_c_abs": 1.0e-3,
    "err_vs_fp64": {"P": [1.2e-4, 5e-5, 4e-4], "F0": [1.5e-4, 5e-5, 5e-4], "F8": [3e-4, 1e-4, 1e-3]},
    "p3_ms": {"8x1": [0.06, 0.03, 0.15], "8x4": [0.15, 0.07, 0.3], "32x1": [0.15, 0.07, 0.3],
              "32x4": [0.4, 0.2, 0.8]},
    "ratio_vs_p3": {"F8": {"8x1": [1.3, 0.7, 2.5], "8x4": [1.1, 0.6, 2.0], "32x1": [1.1, 0.6, 2.0],
                           "32x4": [0.95, 0.6, 1.6]},
                    "F0": {"8x1": [3.0, 1.5, 8.0], "8x4": [2.5, 1.2, 6.0], "32x1": [2.5, 1.2, 6.0],
                           "32x4": [1.8, 1.0, 4.0]},
                    "S5": {"8x4": [0.9, 0.6, 1.3], "32x4": [0.85, 0.5, 1.3]}},
    "plan_ms_per_tick_host_indptr": [0.2, 0.05, 1.0],
    "plan_ms_per_tick_device_indptr": [0.3, 0.08, 1.5],
    "pack_ms_per_layer": [0.05, 0.01, 0.3],
    "jit_build_s": [240, 60, 900],
    "invariance": {"P": "PASS", "F8": "PASS (p 0.7; the split_kv switch I4 is the main risk, p 0.25)",
                   "F0": "PASS (p 0.85)", "FH": "DETECTED with the precondition met (p 0.9)"},
    "bits_vs_p3": {"F8": "most rows differ (> 0.9 of rows), max |d| in [1e-4, 1e-3]",
                   "F0": "most rows differ (> 0.5 of rows), max |d| <= 5e-4"},
    "dram_read_over_ideal": {"1": {"P": [1.0, 0.9, 1.2], "F8": [1.0, 0.9, 1.25], "F0": [1.0, 0.9, 1.25]},
                             "4": {"P": [0.45, 0.28, 0.9], "F8": [0.45, 0.28, 0.9], "F0": [0.45, 0.28, 0.9],
                                   "S5": [0.33, 0.28, 0.7]}},
}


def log(msg):
    print("[r2_fiattn %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail, status=None):
    g = dict(name=name, passed=bool(passed), must=bool(must), detail=detail)
    if status is not None:
        g["status"] = status
    return g


def load_by_path(path, name):
    """A module of another branch, copied into the step record by the step script (pinned by blob id)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def band_text(value, b):
    c, lo, hi = b
    if value is None:
        return "n/a (band [%s, %s], central %s)" % (lo, hi, c)
    return "%.4g %s [%s, %s] (central %s)" % (value, "IN" if lo <= value <= hi else "OUT", lo, hi, c)


# ------------------------------------------------------------------ geometry, cells, cases
def geometry_from_config(cfg: dict) -> dict:
    """q heads, KV heads, head_dim from a HF config dict (NOSA-8B: 32, 2, 128)."""
    hq, hkv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
    d = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // hq)
    if hq % hkv != 0 or d < 16 or d & (d - 1):
        raise ValueError("geometry Hq %d, Hkv %d, D %d" % (hq, hkv, d))
    return dict(Hq=hq, Hkv=hkv, D=d, G=hq // hkv)


def parse_cells(text):
    """'8x1,8x4' -> [(8, 1), (8, 4)]; B in 1 .. 64, rows per request in 1 .. 4; no repeats."""
    out = []
    for part in [p for p in str(text).split(",") if p]:
        m = re.fullmatch(r"(\d+)x(\d+)", part.strip())
        if m is None:
            raise ValueError("cell %r is not BxR" % part)
        b, r = int(m.group(1)), int(m.group(2))
        if not (1 <= b <= 64 and 1 <= r <= 4) or (b, r) in out:
            raise ValueError("cell %r (B 1 .. 64, rows per request 1 .. 4, no repeats)" % part)
        out.append((b, r))
    if not out:
        raise ValueError("no cells")
    return out


def parse_arms(text):
    arms = [a for a in str(text).split(",") if a]
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError("arms %r (subset of %s)" % (text, ARMS))
    return arms


def cell_tag(b, r):
    return "%dx%d" % (b, r)


def cell_seed(seed, b, r):
    return int(seed) * 1000 + int(b) * 10 + int(r)


def spread(n, k):
    """k row indices spread over 0 .. n - 1 (the first and the last included)."""
    if n <= k:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def _draw_scattered(g):
    pool = torch.arange(1, TAIL_BLOCK - N_RECENT)
    w = (TAIL_BLOCK - pool).double().rsqrt()                          # more recent -> more likely (locality)
    n_sc = N_SEL - 2 - N_RECENT
    return pool[torch.multinomial(w, n_sc, replacement=False, generator=g)]


def _swap(g, sc):
    pool = torch.arange(1, TAIL_BLOCK - N_RECENT)
    free = pool[~torch.isin(pool, sc)]
    out = sc.clone()
    pos = torch.randperm(sc.numel(), generator=g)[:N_SWAP]
    out[pos] = free[torch.randperm(free.numel(), generator=g)[:N_SWAP]]
    return out


def _full_list(sc):
    lst = torch.cat([torch.tensor([0]), torch.arange(TAIL_BLOCK - N_RECENT, TAIL_BLOCK), sc, torch.tensor([TAIL_BLOCK])])
    return torch.sort(lst).values


def _short_list(c):
    return torch.cat([torch.tensor([0]), torch.arange(TAIL_BLOCK - (c - 2), TAIL_BLOCK), torch.tensor([TAIL_BLOCK])])


def make_case(geom, b, rpr, seed, with_data=True):
    """The cell's fixed, seeded case on the CPU (module docstring). The lists are drawn BEFORE the data, so
    with_data=False (the ncu summary) gives the same lists without the tensors. Storage entries 0 .. B - 1 are the
    requests, entry B holds the hazard rows."""
    hq, hkv, d = geom["Hq"], geom["Hkv"], geom["D"]
    if not (1 <= rpr <= 4 and b >= 1):
        raise ValueError("B %r, rows per request %r" % (b, rpr))
    cs = cell_seed(seed, b, rpr)
    g = torch.Generator().manual_seed(cs)
    r_, n_, s_, p_ = b * rpr, b + 1, S_SLOTS, N_SEL
    slot_of_block = torch.stack([torch.randperm(s_, generator=g) for _ in range(n_)])        # (N, S)
    t0 = torch.randint(0, PAGE - rpr + 1, (b,), generator=g)
    seq = torch.arange(b).repeat_interleave(rpr)
    last = t0.repeat_interleave(rpr) + torch.arange(rpr).repeat(b) + 1                       # 1 .. 64
    blocks = torch.full((r_, hkv, p_), -1, dtype=torch.int64)
    for n in range(b):
        for h in range(hkv):
            sc0 = _draw_scattered(g)
            for u in range(rpr):
                blocks[n * rpr + u, h] = _full_list(sc0 if u == 0 else _swap(g, sc0))
    alt_blocks = blocks.clone()
    for r in range(1, r_, 2):
        for h in range(hkv):
            alt_blocks[r, h] = _full_list(_draw_scattered(g))
    sblocks = torch.full((len(SHORT_COUNTS), hkv, p_), -1, dtype=torch.int64)
    for i, c in enumerate(SHORT_COUNTS):
        sblocks[i, :, :c] = _short_list(c)

    def rows(blk, seq_idx, last_len):
        n_idx = seq_idx.view(-1, 1, 1).expand_as(blk)
        slots = torch.where(blk >= 0, slot_of_block[n_idx, blk.clamp(min=0)], torch.full_like(blk, -1))
        return dict(seq_idx=seq_idx.to(torch.int32).contiguous(), page_key=blk.to(torch.int32).contiguous(),
                    page_table=slots.to(torch.int32).contiguous(),
                    num_pages=(blk >= 0).sum(-1).to(torch.int32).contiguous(),
                    last_len=last_len.to(torch.int32).contiguous())
    full = rows(blocks, seq, last)
    alt = rows(alt_blocks, seq, last)
    short = rows(sblocks, torch.full((len(SHORT_COUNTS),), n_ - 1), torch.full((len(SHORT_COUNTS),), SHORT_LAST))
    case = dict(geom=dict(geom), B=b, rpr=rpr, R=r_, N=n_, S=s_, P=p_, seed=cs, slot_of_block=slot_of_block,
                full=full, alt=alt, short=short, k=None, v=None, bias=None)
    if with_data:
        q = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        qa = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        full["q"] = q
        odd = (torch.arange(r_) % 2 == 1).view(-1, 1, 1)
        alt["q"] = torch.where(odd, qa, q).contiguous()
        short["q"] = torch.randn(len(SHORT_COUNTS), hq, d, generator=g).to(torch.bfloat16)
        case["k"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=g).to(torch.bfloat16)
        case["v"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=g).to(torch.bfloat16)
        case["bias"] = ((torch.rand(n_, s_ * PAGE, hkv, generator=g) * 2 - 1) * BIAS_RANGE).to(torch.bfloat16)
    return case


def case_to(case, device):
    out = dict(case)
    for k in ("full", "alt", "short"):
        out[k] = {f: t.to(device) for f, t in case[k].items()}
    for f in CACHE_FIELDS:
        out[f] = None if case[f] is None else case[f].to(device)
    return out


def batch_of(case, which, caches=None):
    """A batch: the row fields + the caches (k, v, bias). which: 'full', 'alt', 'short' or 'hazard' (full + short)."""
    if which == "hazard":
        rows = {f: torch.cat([case["full"][f], case["short"][f]], 0).contiguous() for f in case["full"]}
    else:
        rows = dict(case[which])
    c = caches if caches is not None else dict(k=case["k"], v=case["v"], bias=case["bias"])
    rows.update({f: c[f] for f in CACHE_FIELDS})
    return rows


def select(b, idx):
    """Rows idx (a list) of batch b, in that order (contiguous copies); the caches are shared."""
    t = torch.as_tensor(list(idx), dtype=torch.long, device=b["page_table"].device)
    out = {f: b[f].index_select(0, t).contiguous() for f in ROW_FIELDS if f in b}
    out.update({f: b[f] for f in CACHE_FIELDS})
    return out


def check_store(*caches):
    """The flat addressing (option A / B views, the bias offsets) assumes contiguous caches."""
    for c in caches:
        if c is None or not c.is_contiguous():
            raise ValueError("the K / V / bias caches must be contiguous")


def reach_map(b, n_entries, s_slots, hkv):
    """(N, S, Hkv) int64: the number of leading rows of each (slot, head) page that some row of b reads (64 for a
    page listed before a row's last page, last_len for its last page; 0 = not listed)."""
    pt = b["page_table"].to("cpu", torch.int64)
    npg = b["num_pages"].to("cpu", torch.int64)
    si = b["seq_idx"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    r_, h_, p_ = pt.shape
    j = torch.arange(p_).view(1, 1, -1)
    valid = j < npg.unsqueeze(-1)
    is_last = j == (npg - 1).unsqueeze(-1)
    val = torch.where(is_last, ll.view(-1, 1, 1).expand_as(pt), torch.full_like(pt, PAGE))
    n_idx = si.view(-1, 1, 1).expand_as(pt)
    h_idx = torch.arange(h_).view(1, -1, 1).expand_as(pt)
    flat = (n_idx * s_slots + pt.clamp(min=0)) * hkv + h_idx
    reach = torch.zeros(n_entries * s_slots * hkv, dtype=torch.int64)
    reach.scatter_reduce_(0, flat[valid], val[valid], reduce="amax")
    return reach.view(n_entries, s_slots, hkv)


def poison_caches(case, b):
    """K, V, bias copies where every row no row of b reads is NaN, per (slot, head). Returns (caches dict, stats)."""
    n_, s_, hkv, d = case["N"], case["S"], case["geom"]["Hkv"], case["geom"]["D"]
    reach = reach_map(b, n_, s_, hkv)
    mask = torch.arange(PAGE).view(1, 1, -1, 1) >= reach.unsqueeze(2)                     # (N, S, 64, Hkv)
    dev = b["k"].device
    m = mask.to(dev)
    out = {}
    for f in ("k", "v"):
        t = b[f].clone()
        t.view(n_, s_, PAGE, hkv, d)[m] = float("nan")
        out[f] = t
    t = b["bias"].clone()
    t.view(n_, s_, PAGE, hkv)[m] = float("nan")
    out["bias"] = t
    stats = dict(poisoned_rows=int(mask.sum()), total_rows=int(mask.numel()),
                 unlisted_pages=int((reach == 0).sum()), pages=int(reach.numel()))
    return out, stats


def reference_fp64(b, rows, geom, zero_bias=False, scale=None):
    """The float64 attention of rows (this module's own code): per (row, KV head) the listed pages in list order,
    the last one cut at last_len; softmax((q.k + bias) * scale) . v with one softmax over all listed rows."""
    g, d, hkv = geom["G"], geom["D"], geom["Hkv"]
    sc = d ** -0.5 if scale is None else float(scale)
    k, v, bias = b["k"], b["v"], b["bias"]
    dev = k.device
    pt = b["page_table"].to("cpu", torch.int64)
    npg = b["num_pages"].to("cpu", torch.int64)
    si = b["seq_idx"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    out = torch.empty((len(rows), geom["Hq"], d), dtype=torch.float64, device=dev)
    for i, r in enumerate(rows):
        n, last = int(si[r]), int(ll[r])
        for h in range(hkv):
            c = int(npg[r, h])
            slots = pt[r, h, :c]
            tok = (slots.view(-1, 1) * PAGE + torch.arange(PAGE)).reshape(-1)[:(c - 1) * PAGE + last].to(dev)
            kk = k[n, tok, h].double()
            vv = v[n, tok, h].double()
            bb = torch.zeros(tok.numel(), dtype=torch.float64, device=dev) if zero_bias else bias[n, tok, h].double()
            qq = b["q"][r, h * g:(h + 1) * g].double()
            w = torch.softmax((qq @ kk.t() + bb.unsqueeze(0)) * sc, dim=1)
            out[i, h * g:(h + 1) * g] = w @ vv
    return out


def ideal_bytes(b, geom):
    """Bytes a row-private kernel must read: per (row, head) its listed token rows x (K + V + bias), bf16."""
    npg = b["num_pages"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    tok = ((npg - 1) * PAGE + ll.view(-1, 1)).sum()
    return int(tok) * (4 * geom["D"] + 2)


def union_bytes(b, geom, n_entries, s_slots):
    """Bytes of the request union: every (slot, head) page row that some row reads, once."""
    return int(reach_map(b, n_entries, s_slots, geom["Hkv"]).sum()) * (4 * geom["D"] + 2)


# ------------------------------------------------------------------ FlashInfer packing and addressing
def pack_option_a(b, s_slots, h):
    """FlashInfer CSR for KV head h, requests = rows: indptr (R + 1), indices (an R * P buffer; the first indptr[R]
    entries are the pages: entry * S + slot in the row's canonical order), last_page_len (R). Device ops only."""
    pt = b["page_table"]
    dev = pt.device
    r_, _, p_ = pt.shape
    npg = b["num_pages"][:, h].to(torch.int64)
    indptr = torch.zeros(r_ + 1, dtype=torch.int64, device=dev)
    indptr[1:] = torch.cumsum(npg, 0)
    j = torch.arange(p_, device=dev)
    valid = j.view(1, -1) < npg.view(-1, 1)
    gid = b["seq_idx"].to(torch.int64).view(-1, 1) * s_slots + pt[:, h, :].to(torch.int64)
    pos = torch.where(valid, indptr[:-1].view(-1, 1) + j.view(1, -1), torch.full_like(gid, r_ * p_))
    buf = torch.zeros(r_ * p_ + 1, dtype=torch.int32, device=dev)
    buf.scatter_(0, pos.reshape(-1), gid.to(torch.int32).reshape(-1))             # invalid entries -> the dump slot
    return indptr.to(torch.int32), buf[:r_ * p_], b["last_len"].to(torch.int32).contiguous()


def pack_option_b(b, s_slots):
    """FlashInfer CSR for ONE call over all KV heads, requests = (row, head) row-major (request r * Hkv + h); page id
    = (entry * S + slot) * 64 * Hkv + h (the overlapping as_strided view kv_view_b)."""
    pt = b["page_table"]
    dev = pt.device
    r_, hkv, p_ = pt.shape
    nq = r_ * hkv
    npg = b["num_pages"].to(torch.int64).reshape(-1)
    indptr = torch.zeros(nq + 1, dtype=torch.int64, device=dev)
    indptr[1:] = torch.cumsum(npg, 0)
    j = torch.arange(p_, device=dev)
    valid = j.view(1, -1) < npg.view(-1, 1)
    hh = torch.arange(hkv, device=dev).view(1, -1, 1)
    gid = ((b["seq_idx"].to(torch.int64).view(-1, 1, 1) * s_slots + pt.to(torch.int64)) * (PAGE * hkv) + hh)
    gid = gid.reshape(nq, p_)
    pos = torch.where(valid, indptr[:-1].view(-1, 1) + j.view(1, -1), torch.full_like(gid, nq * p_))
    buf = torch.zeros(nq * p_ + 1, dtype=torch.int32, device=dev)
    buf.scatter_(0, pos.reshape(-1), gid.to(torch.int32).reshape(-1))
    last = b["last_len"].to(torch.int32).repeat_interleave(hkv).contiguous()
    return indptr.to(torch.int32), buf[:nq * p_], last


def kv_view_a(cache, s_slots, h):
    """Option A: the store (N, S*64, Hkv, D) as NHD pages (N*S, 64, Hkv, D), KV head h only: (N*S, 64, 1, D)."""
    check_store(cache)
    n_, sr, hkv, d = cache.shape
    return cache.view(n_ * s_slots, PAGE, hkv, d)[:, :, h:h + 1, :]


def kv_view_b(cache):
    """Option B: an OVERLAPPING view (M, 64, 1, D), strides (D, Hkv*D, D, 1): page p = g * 64 * Hkv + h (g = entry *
    S + slot) starts at the store row (g * 64, h). M is the largest count that stays inside the storage."""
    check_store(cache)
    n_, sr, hkv, d = cache.shape
    m = n_ * sr * hkv - (PAGE - 1) * hkv
    return torch.as_strided(cache, (m, PAGE, 1, d), (d, hkv * d, d, 1), cache.storage_offset())


def fi_offset(view, page, entry, feat):
    """The element offset FlashInfer's paged_kv_t computes (page.cuh get_elem_offset, NHD, head 0)."""
    return view.storage_offset() + page * view.stride(0) + entry * view.stride(1) + feat * view.stride(3)


def store_offset(cache, n, slot, entry, h, feat):
    """The element offset piece 3 reads (k_base + rows * s_ks + d: bi_paged_attention.py:229-239)."""
    return (cache.storage_offset() + n * cache.stride(0) + (slot * PAGE + entry) * cache.stride(1)
            + h * cache.stride(2) + feat * cache.stride(3))


def bias_args(option, hkv, h=0):
    """The variant's scalars (page_mul, entry_stride, head) for option 'a' (KV head h) or 'b'."""
    if option == "a":
        return PAGE * hkv, hkv, h
    if option == "b":
        return 1, hkv, 0
    raise ValueError(option)


def bias_offset(page, entry, page_mul, entry_stride, head):
    """The variant's bias offset (VARIANT_DECL), relative to the bias tensor's data pointer."""
    return page * page_mul + entry * entry_stride + head


def jit_args_for(d):
    """FlashInfer's jit_args for gen_customize_batch_prefill_module('fa2', *jit_args) (decode.py:709-716)."""
    return [VARIANT_URI % d, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, d, d,
            list(VARIANT_TENSORS[0]), list(VARIANT_TENSORS[1]), list(VARIANT_SCALARS[0]), list(VARIANT_SCALARS[1]),
            VARIANT_NAME, VARIANT_DECL]


def version_ok(v):
    return str(v).split("+")[0] == FI_VERSION


def plan_info_of(w):
    """split_kv, chunk size (tokens), padded batch, CTA tile q of a planned FlashInfer wrapper: the plan vector
    (PrefillPlanInfo::ToVector, scheduler.cuh:650-666) and the kv_chunk_size int32 the plan wrote into its pinned
    host buffer (scheduler.cuh:747, offset vec[9])."""
    pi = [int(x) for x in w._plan_info]
    off = pi[9]
    raw = bytes(w._pin_memory_int_workspace_buffer[off:off + 4].tolist())
    return dict(padded_batch_size=pi[0], cta_tile_q=pi[3], split_kv=bool(pi[14]),
                chunk_tokens=int.from_bytes(raw, "little", signed=True))


# ------------------------------------------------------------------ arms (GPU)
class P3Arm:
    """Piece 3 v2 with a fixed workspace (R_max rows)."""
    def __init__(self, mod, geom, rmax, p_width, device):
        self.mod, self.geom = mod, geom
        nsp = mod.splits_for(p_width)
        bh = mod.block_h_for(geom["G"])
        self.ws = (torch.empty((rmax, geom["Hkv"], nsp, bh, geom["D"]), dtype=torch.float32, device=device),
                   torch.empty((rmax, geom["Hkv"], nsp, 2, bh), dtype=torch.float32, device=device))
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in self.ws)
        self.info = None
        self._b = self._out = None

    def prepare(self, b):
        self._b = b
        self._out = torch.empty_like(b["q"])

    def launch(self):
        b = self._b
        return self.mod.bi_paged_decode(b["q"], b["k"], b["v"], b["bias"], b["seq_idx"], b["page_table"],
                                        b["num_pages"], b["last_len"], out=self._out, workspace=self.ws)

    def __call__(self, b):
        self.prepare(b)
        return self.launch().clone()


class S5Arm(P3Arm):
    """sharedkv5: bi_paged_decode_shared(base = piece 3, ..., page_key, rows_max 4, mode 'split', tile 'reload')."""
    def __init__(self, skv, base, geom, rmax, p_width, device):
        super().__init__(base, geom, rmax, p_width, device)
        self.skv = skv

    def launch(self):
        b = self._b
        return self.skv.bi_paged_decode_shared(self.mod, b["q"], b["k"], b["v"], b["bias"], b["seq_idx"],
                                               b["page_table"], b["num_pages"], b["last_len"], b["page_key"],
                                               rows_max=4, mode="split", out=self._out, workspace=self.ws,
                                               tile="reload")


class FIArm:
    """FlashInfer option A: one wrapper per KV head (sharing one float workspace: the head calls run in stream
    order), plan per batch, run per head into (R, G, D) buffers, then the interleave into (R, Hq, D)."""
    def __init__(self, fi, name, split, geom, s_slots, jit_args, device):
        if split not in ("fixed", "none", "heuristic"):
            raise ValueError(split)
        self.fi, self.name, self.split, self.geom, self.S = fi, name, split, geom, s_slots
        self.custom = jit_args is not None
        self.scale = geom["D"] ** -0.5
        self.fws = torch.zeros(FLOAT_WS_BYTES, dtype=torch.uint8, device=device)
        kw = dict(use_tensor_cores=True, backend="fa2")
        if jit_args is not None:
            kw["jit_args"] = jit_args
        self.wr = [fi.BatchDecodeWithPagedKVCacheWrapper(self.fws, "NHD", **kw) for _ in range(geom["Hkv"])]
        self.workspace_bytes = FLOAT_WS_BYTES + sum(w._int_workspace_buffer.numel() for w in self.wr)
        self.pinned_bytes = sum(w._pin_memory_int_workspace_buffer.numel() for w in self.wr)
        self.info = None
        self._b = self._oh = None

    def plan_kwargs(self):
        if self.split == "fixed":
            return dict(fixed_split_size=FIXED_SPLIT)
        if self.split == "none":
            return dict(disable_split_kv=True)
        return {}

    def pack(self, b):
        return [pack_option_a(b, self.S, h) for h in range(self.geom["Hkv"])]

    def plan_packed(self, packed):
        """plan() per head. indptr / last_page_len on the host or on the device (then plan's own .to('cpu') is the
        D2H sync). The caller synchronizes first: plan rewrites its pinned buffer while the previous plan's H2D copy
        may still be queued (a FlashInfer usage hazard, recorded in the plan doc)."""
        g, d = self.geom["G"], self.geom["D"]
        for h, (indptr, indices, last) in enumerate(packed):
            self.wr[h].plan(indptr, indices, last, g, 1, d, PAGE, q_data_type=torch.bfloat16,
                            kv_data_type=torch.bfloat16, sm_scale=self.scale, **self.plan_kwargs())
        self.info = [plan_info_of(w) for w in self.wr]

    def prepare(self, b):
        check_store(b["k"], b["v"], b["bias"])
        torch.cuda.synchronize()
        packed = [(ip.cpu(), ix, ll.cpu()) for ip, ix, ll in self.pack(b)]
        self.plan_packed(packed)
        r_ = b["q"].shape[0]
        self._oh = [torch.empty((r_, self.geom["G"], self.geom["D"]), dtype=torch.bfloat16, device=b["q"].device)
                    for _ in range(self.geom["Hkv"])]
        self._b = b

    def launch_runs(self):
        b = self._b
        g, hkv = self.geom["G"], self.geom["Hkv"]
        for h in range(hkv):
            kv = (kv_view_a(b["k"], self.S, h), kv_view_a(b["v"], self.S, h))
            extra = [b["bias"], *bias_args("a", hkv, h), self.scale] if self.custom else []
            self.wr[h].run(b["q"][:, h * g:(h + 1) * g], kv, *extra, out=self._oh[h])
        return self._oh

    def launch(self):
        return torch.cat(self.launch_runs(), dim=1)

    def __call__(self, b):
        self.prepare(b)
        return self.launch()


class FIStridedArm(FIArm):
    """Option B (REPORT probe): one wrapper, one call over (row, head) requests through kv_view_b."""
    def __init__(self, fi, geom, s_slots, jit_args, device):
        super().__init__(fi, "STRIDED", "fixed", geom, s_slots, jit_args, device)
        self.wr = self.wr[:1]
        self.workspace_bytes = FLOAT_WS_BYTES + self.wr[0]._int_workspace_buffer.numel()
        self.pinned_bytes = self.wr[0]._pin_memory_int_workspace_buffer.numel()

    def prepare(self, b):
        check_store(b["k"], b["v"], b["bias"])
        torch.cuda.synchronize()
        ip, ix, ll = pack_option_b(b, self.S)
        self.wr[0].plan(ip.cpu(), ix, ll.cpu(), self.geom["G"], 1, self.geom["D"], PAGE, q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16, sm_scale=self.scale, fixed_split_size=FIXED_SPLIT)
        self.info = [plan_info_of(self.wr[0])]
        r_ = b["q"].shape[0]
        self._oh = [torch.empty((r_ * self.geom["Hkv"], self.geom["G"], self.geom["D"]), dtype=torch.bfloat16,
                                device=b["q"].device)]
        self._b = b

    def launch(self):
        b = self._b
        r_ = b["q"].shape[0]
        q2 = b["q"].view(r_ * self.geom["Hkv"], self.geom["G"], self.geom["D"])
        extra = [b["bias"], *bias_args("b", self.geom["Hkv"]), self.scale]
        self.wr[0].run(q2, (kv_view_b(b["k"]), kv_view_b(b["v"])), *extra, out=self._oh[0])
        return self._oh[0].view(r_, self.geom["Hq"], self.geom["D"])


# ------------------------------------------------------------------ gates (arm-agnostic; the CPU tests use fakes)
def bits_equal(a, b):
    """Bitwise equality (a -0.0 / +0.0 or a NaN payload difference counts)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def changed_rows(b1, b2):
    """Rows whose q or page lists differ between two batches of equal size (the I3 precondition)."""
    out = []
    for r in range(b1["q"].shape[0]):
        if not (torch.equal(b1["q"][r], b2["q"][r]) and torch.equal(b1["page_table"][r], b2["page_table"][r])):
            out.append(r)
    return out


def invariance_suite(name, tag, arm, bf, ba, bh, alone_rows, perms, poisoned, short_idx, hazard_pre, must):
    """I1 - I5 for one arm and cell. arm(batch) -> (R, Hq, D) output; arm.info after a call = the plan info per KV
    head (None for piece 3). Returns (gates, the full-batch output)."""
    gates = []
    full = arm(bf)
    r_ = full.shape[0]
    bad = [r for r in alone_rows if not bits_equal(arm(select(bf, [r]))[0], full[r])]
    gates.append(gate("I1_alone_%s_%s" % (name, tag), not bad, must,
                      "%d rows alone vs in the batch of %d; differing %s" % (len(alone_rows), r_, bad[:10])))
    bad, det = [], []
    for pname, p in perms:
        o = arm(select(bf, p))
        d = [r for j, r in enumerate(p) if not bits_equal(o[j], full[r])]
        det.append("%s %d" % (pname, len(d)))
        bad += d
    gates.append(gate("I2_order_%s_%s" % (name, tag), not bad, must,
                      "rows differing per permutation: %s; first %s" % (", ".join(det), sorted(set(bad))[:10])))
    oa = arm(ba)
    keep = list(range(0, r_, 2))
    changed = changed_rows(bf, ba)
    pre = bool(changed) and all(r % 2 == 1 for r in changed)
    bad = [r for r in keep if not bits_equal(oa[r], full[r])]
    gates.append(gate("I3_siblings_%s_%s" % (name, tag), pre and not bad, must,
                      "%s; %d odd rows changed (q and lists), %d even rows compared; differing %s"
                      % ("precondition met" if pre else "VOID: the alt batch did not change exactly odd rows",
                         len(changed), len(keep), bad[:10]), status=None if pre else "VOID"))
    oh = arm(bh)
    info_b = arm.info
    bad_rest = [r for r in range(r_) if not bits_equal(oh[r], full[r])]
    bad_short, pre_rows = [], []
    for i, s in enumerate(short_idx):
        o = arm(select(bh, [s]))
        info_a = arm.info
        if not bits_equal(o[0], oh[s]):
            bad_short.append(s)
        if SHORT_COUNTS[i] <= FIXED_SPLIT and info_a is not None and info_b is not None:
            pre_rows.append(all(not x["split_kv"] for x in info_a) and all(x["split_kv"] for x in info_b))
    pre4 = (not hazard_pre) or (bool(pre_rows) and all(pre_rows))
    gates.append(gate("I4_split_switch_%s_%s" % (name, tag), pre4 and not bad_short and not bad_rest, must,
                      "%s; short rows (%s pages) alone vs in the hazard batch: differing %s; full rows in the hazard "
                      "batch vs the full batch: differing %s"
                      % ("precondition met (split_kv false alone, true batched, every head)" if hazard_pre and pre4
                         else ("VOID: split_kv precondition not met %s" % pre_rows if hazard_pre else
                               "no split_kv precondition for this arm"),
                         ",".join(str(c) for c in SHORT_COUNTS), bad_short, bad_rest[:10]),
                      status=None if pre4 else "VOID"))
    bp = dict(bf)
    bp.update(poisoned)
    op = arm(bp)
    finite = bool(torch.isfinite(op.float()).all())
    bad = [r for r in range(r_) if not bits_equal(op[r], full[r])]
    gates.append(gate("I5_poison_%s_%s" % (name, tag), finite and not bad, must,
                      "unread rows NaN in K / V / bias: output finite %s; differing %s" % (finite, bad[:10])))
    return gates, full


def correctness_gate(name, tag, out_rows, ref, tol, must):
    d = (out_rows.double() - ref.to(out_rows.device)).abs()
    finite = bool(torch.isfinite(d).all())
    err = float(d.max()) if finite else float("inf")
    return gate("C_%s_%s" % (name, tag), finite and err <= tol, must,
                "max |arm - fp64| %.3e over %d rows (tol %.1e); %s"
                % (err, out_rows.shape[0], tol, band_text(err, PRED["err_vs_fp64"][name])
                   if name in PRED["err_vs_fp64"] else "no band"))


def bias_witness(tag, arm, bf, rows, ref_bias, ref_nobias, tol, must):
    """CW: the arm with bias = 0 must FAIL C against the biased reference; VOID when the bias barely moves the
    reference (a no-op plant, lessons L34)."""
    move = float((ref_bias - ref_nobias).abs().max())
    if not move >= WITNESS_FACTOR * tol:
        return gate("CW_bias_witness_%s" % tag, False, must, "VOID: the bias moves the reference by %.3e < %.0f x tol"
                    % (move, WITNESS_FACTOR), status="VOID")
    b0 = dict(bf)
    b0["bias"] = torch.zeros_like(bf["bias"])
    o = arm(b0)[torch.as_tensor(rows, device=bf["q"].device)]
    err = float((o.double() - ref_bias.to(o.device)).abs().max())
    st = "DETECTED" if err > tol else "BLIND"
    return gate("CW_bias_witness_%s" % tag, st == "DETECTED", must,
                "%s: bias-free arm vs biased reference %.3e (tol %.1e); the bias moves the reference by %.3e"
                % (st, err, tol, move), status=st)


def negative_control(tag, arm, bf, rows, must):
    """N: the heuristic split (FH) alone vs batched must differ on a row whose plan differs (precondition)."""
    full = arm(bf)
    info_b = arm.info
    pre, det, chunks = [], [], set()
    for r in rows:
        o = arm(select(bf, [r]))
        info_a = arm.info
        differs = any(a["chunk_tokens"] != c["chunk_tokens"] or a["split_kv"] != c["split_kv"]
                      for a, c in zip(info_a, info_b))
        chunks.add((tuple(x["chunk_tokens"] for x in info_a), tuple(x["chunk_tokens"] for x in info_b)))
        if differs:
            pre.append(r)
            if not bits_equal(o[0], full[r]):
                det.append(r)
    st = "VOID" if not pre else ("DETECTED" if det else "BLIND")
    return gate("N_heuristic_negative_%s" % tag, st == "DETECTED", must,
                "%s: %d of %d rows with a batch-dependent plan (chunk tokens alone / batched per head: %s); "
                "differing %d" % (st, len(pre), len(rows), sorted(chunks)[:4], len(det)), status=st), full


def bits_vs(name, tag, out, ref_out):
    """D (REPORT): rows / elements of out that differ bitwise from piece 3's, and the max |difference|."""
    it = torch.int16
    neq = out.contiguous().view(it) != ref_out.contiguous().view(it)
    rows = int(neq.flatten(1).any(1).sum())
    diff = float((out.float() - ref_out.float()).abs().max())
    return gate("R_D_bits_vs_P_%s_%s" % (name, tag), True, False,
                "%d of %d rows differ from piece 3 (%.4f of elements); max |d| %.3e; %s"
                % (rows, out.shape[0], float(neq.float().mean()), diff, PRED["bits_vs_p3"].get(name, "no prediction")))


# ------------------------------------------------------------------ timing (GPU)
def summarize_ms(xs):
    s = sorted(xs)
    n = len(s)
    return dict(median=round(statistics.median(s), 5), q1=round(s[n // 4], 5), q3=round(s[(3 * n) // 4], 5), n=n)


def cuda_time(fn, warm, reps):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        xs.append(e0.elapsed_time(e1))
    return summarize_ms(xs)


def host_time(fn, reps):
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1e3)
    torch.cuda.synchronize()
    return summarize_ms(xs)


def peak_delta(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - base)


def fi_float_used(info, g, d):
    """Float workspace bytes a plan reserves (scheduler.cuh:750-755: the allocator sizes tmp_v with sizeof(float),
    although the kernel stores DTypeO partials into it, plus tmp_s); 0 without split_kv."""
    if not info["split_kv"]:
        return 0
    return g * info["padded_batch_size"] * info["cta_tile_q"] * d * 4 + g * info["padded_batch_size"] * info["cta_tile_q"] * 4


def time_arm(name, arm, b, warm, reps, geom):
    rep = dict(arm=name)
    arm.prepare(b)
    torch.cuda.synchronize()
    rep["call_ms"] = cuda_time(arm.launch, warm, reps)
    if isinstance(arm, FIArm) and not isinstance(arm, FIStridedArm):
        rep["runs_only_ms"] = cuda_time(arm.launch_runs, warm, reps)
        packed = arm.pack(b)
        host = [(ip.cpu(), ix, ll.cpu()) for ip, ix, ll in packed]
        rep["plan_host_indptr_ms_per_tick"] = host_time(lambda: arm.plan_packed(host), 50)
        rep["plan_device_indptr_ms_per_tick"] = host_time(lambda: arm.plan_packed(packed), 50)
        rep["pack_ms_per_layer"] = cuda_time(lambda: arm.pack(b), 5, 50)
        rep["plan_info"] = arm.info
        rep["float_ws_used_bytes"] = max(fi_float_used(i, geom["G"], geom["D"]) for i in arm.info)
        rep["pinned_host_bytes"] = arm.pinned_bytes
        arm.prepare(b)
    rep["workspace_bytes"] = arm.workspace_bytes
    rep["peak_alloc_delta_bytes"] = peak_delta(arm.launch)
    return rep


# ------------------------------------------------------------------ ncu (REPORT)
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def ncu_rows(text, fields):
    """Kernel rows of an ncu raw CSV (--csv --page raw --print-units base): the header row naming 'Kernel Name', rows
    with a numeric ID; tag = the first fia|<arm> in any cell (the NVTX column)."""
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hi is None:
        raise ValueError("no header row with 'Kernel Name'")
    head = rows[hi]
    out = []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        if not d.get("ID", "").strip().isdigit():
            continue
        tag = None
        for c in r:
            m = TAG_RE.search(c)
            if m:
                tag = m.group(1)
                break
        row = dict(id=int(d["ID"]), name=d["Kernel Name"], tag=tag)
        for f in fields:
            row[f] = _num(d.get(f, ""))
        out.append(row)
    return out


def ncu_summary(rows, arms, ideal, union, rpr):
    """Per arm: DRAM bytes read / written, kernel time, kernel count (vs KERNELS_PER_CALL), ratios to the ideal
    and the union bytes, the band (PRED dram_read_over_ideal), the reuse label."""
    per = {a: dict(read=0.0, write=0.0, time_ns=0.0, kernels=[], label=REUSE_LABEL.get(a, "")) for a in arms}
    stray = []
    for r in rows:
        if r["tag"] not in per or not any(k in r["name"] for k in ATTN_KERNELS):
            stray.append((r["tag"], r["name"][:60]))
            continue
        p = per[r["tag"]]
        p["read"] += r.get("dram__bytes_read.sum") or 0.0
        p["write"] += r.get("dram__bytes_write.sum") or 0.0
        p["time_ns"] += r.get("gpu__time_duration.sum") or 0.0
        p["kernels"].append(r["name"][:60])
    for a, p in per.items():
        p["n_kernels"] = len(p["kernels"])
        p["expected_kernels"] = KERNELS_PER_CALL.get(a)
        p["read_over_ideal"] = p["read"] / ideal if ideal else None
        p["read_over_union"] = p["read"] / union if union else None
        bands = PRED["dram_read_over_ideal"].get("4" if rpr > 1 else "1", {})
        p["band"] = band_text(p["read_over_ideal"], bands[a]) if a in bands else "no band"
    return dict(arms=per, stray=stray[:20], ideal_bytes=ideal, union_bytes=union)


def ncu_cap(left_s, cap_s):
    """Seconds for one ncu child, or None (skip) when the budget left cannot cover a minimal child + the margin."""
    room = left_s - NCU_MARGIN_S
    if room < 60:
        return None
    return int(min(cap_s, room))


def ncu_argv(a, b, rpr, child_out, rep, metrics, arms, bias_free):
    argv = [a.ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", "all",
            "--clock-control", "none", "--nvtx", "--kernel-name", KERNEL_FILTER, "--metrics", ",".join(metrics),
            "--target-processes", "application-only", "-f", "-o", rep,
            sys.executable, os.path.abspath(__file__), "--mode", "ncu-cell", "--out", child_out, "--config", a.config,
            "--biattn", a.biattn, "--cells", cell_tag(b, rpr), "--seed", str(a.seed), "--arms", ",".join(arms)]
    if "S5" in arms:
        argv += ["--sharedkv", a.sharedkv]
    if bias_free:
        argv += ["--bias-free"]
    return argv


def run_logged(argv, path, timeout):
    """argv with stdout + stderr to path, in its own process group, killed at the timeout. The exit or 'timeout'."""
    with open(path, "w") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            return "timeout"


def ncu_metrics(a, out):
    """ncu --query-metrics --chip ga100 (saved): the required names present, the optional kept when present."""
    path = os.path.join(out, "ncu_metrics_query.txt")
    rc = run_logged([a.ncu, "--query-metrics", "--chip", "ga100"], path, 120)
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    names = set(re.findall(r"^\s*([a-z0-9_]+__[a-z0-9_]+)", text, flags=re.M))
    base = lambda m: m.rsplit(".", 1)[0]                                # noqa: E731
    missing = [m for m in NCU_REQUIRED if base(m) not in names]
    kept = [m for m in NCU_OPTIONAL if base(m) in names]
    return rc, missing, list(NCU_REQUIRED) + kept


# ------------------------------------------------------------------ the cell (main mode)
def build_arms(fi, a, geom, case, p3, skv, jit, arms, device):
    out = {}
    rmax = case["R"] + len(SHORT_COUNTS)
    for name in arms:
        if name == "P":
            out[name] = P3Arm(p3, geom, rmax, case["P"], device)
        elif name == "S5":
            if skv is not None and case["rpr"] > 1:
                out[name] = S5Arm(skv, p3, geom, rmax, case["P"], device)
        else:
            split = dict(F8="fixed", F0="none", FH="heuristic")[name]
            out[name] = FIArm(fi, name, split, geom, case["S"], jit, device)
    return out


def run_cell(fi, a, geom, p3, skv, jit, b, rpr, gates):
    """All gates and reports of one cell; appends to gates, returns the cell's report dict."""
    tag = cell_tag(b, rpr)
    custom = jit is not None
    t0 = time.time()
    case = case_to(make_case(geom, b, rpr, a.seed), "cuda")
    if not custom:                                                     # the bias-free REPORT (C0 failed)
        case["bias"] = torch.zeros_like(case["bias"])
    bf, ba, bh = batch_of(case, "full"), batch_of(case, "alt"), batch_of(case, "hazard")
    r_ = case["R"]
    short_idx = list(range(r_, r_ + len(SHORT_COUNTS)))
    rows = spread(r_, a.sample)
    rev = [n * rpr + u for n in reversed(range(b)) for u in range(rpr)]
    rnd = torch.randperm(r_, generator=torch.Generator().manual_seed(case["seed"] + 1)).tolist()
    perms = [("reversed_requests", rev), ("random", rnd)]
    poisoned, pstats = poison_caches(case, bf)
    ref = reference_fp64(bf, rows, geom)
    ref0 = reference_fp64(bf, rows, geom, zero_bias=True) if custom else None
    rep = dict(cell=tag, R=r_, rows_sampled=rows, poison=pstats, ideal_bytes=ideal_bytes(bf, geom),
               union_bytes=union_bytes(bf, geom, case["N"], case["S"]), arms={})
    arms = build_arms(fi, a, geom, case, p3, skv, jit, a.arms, "cuda")
    outs = {}
    for name in ("P", "F8", "F0", "S5"):
        if name not in arms:
            continue
        must = custom and name in MUST_ARMS
        try:
            g, full = invariance_suite(name, tag, arms[name], bf, ba, bh, rows, perms, poisoned, short_idx,
                                       hazard_pre=(name == "F8"), must=must)
            gates.extend(g)
            outs[name] = full
            gates.append(correctness_gate(name, tag, full[torch.as_tensor(rows, device=full.device)], ref, TOL_C,
                                          must))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("ARM_%s_%s" % (name, tag), False, must, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                            str(e)[:300])))
    if "FH" in arms:
        try:
            g, _ = negative_control(tag, arms["FH"], bf, rows, custom)
            gates.append(g)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("N_heuristic_negative_%s" % tag, False, custom, "EXCEPTION %s: %s"
                              % (type(e).__name__, str(e)[:300])))
    if custom and "F8" in arms:
        try:
            gates.append(bias_witness(tag, arms["F8"], bf, rows, ref, ref0, TOL_C, True))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("CW_bias_witness_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                             str(e)[:300])))
    if "P" in outs:
        for name in ("F8", "F0", "S5"):
            if name in outs:
                gates.append(bits_vs(name, tag, outs[name], outs["P"]))
    tp = None
    for name in ("P", "F8", "F0", "FH", "S5"):
        if name not in arms:
            continue
        try:
            t = time_arm(name, arms[name], bf, a.warm, a.reps, geom)
            rep["arms"][name] = t
            med = t["call_ms"]["median"]
            if name == "P":
                tp = med
                txt = "P %.4f ms; %s" % (med, band_text(med, PRED["p3_ms"][tag]) if tag in PRED["p3_ms"] else "no band")
            else:
                ratio = med / tp if tp else None
                bands = PRED["ratio_vs_p3"].get(name, {})
                txt = "%s %.4f ms, x%s of P; %s" % (name, med, "%.3f" % ratio if ratio else "n/a",
                                                    band_text(ratio, bands[tag]) if tag in bands else "no band")
                if "plan_host_indptr_ms_per_tick" in t:
                    txt += "; plan/tick host %.4f (%s) device %.4f (%s); pack/layer %.4f (%s); runs only %.4f" % (
                        t["plan_host_indptr_ms_per_tick"]["median"],
                        band_text(t["plan_host_indptr_ms_per_tick"]["median"], PRED["plan_ms_per_tick_host_indptr"]),
                        t["plan_device_indptr_ms_per_tick"]["median"],
                        band_text(t["plan_device_indptr_ms_per_tick"]["median"],
                                  PRED["plan_ms_per_tick_device_indptr"]),
                        t["pack_ms_per_layer"]["median"], band_text(t["pack_ms_per_layer"]["median"],
                                                                    PRED["pack_ms_per_layer"]),
                        t["runs_only_ms"]["median"])
            txt += "; workspace %d B, peak delta %d B" % (t["workspace_bytes"], t["peak_alloc_delta_bytes"])
            gates.append(gate("R_T_%s_%s" % (name, tag), True, False, txt))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_T_%s_%s" % (name, tag), False, False, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                             str(e)[:300])))
    if custom and a.strided and "F8" in outs:
        try:
            sa = FIStridedArm(fi, geom, case["S"], jit, "cuda")
            o = sa(bf)
            same = sum(1 for r in range(r_) if bits_equal(o[r], outs["F8"][r]))
            err = float((o[torch.as_tensor(rows, device=o.device)].double() - ref).abs().max())
            t = time_arm("STRIDED", sa, bf, a.warm, a.reps, geom)
            rep["arms"]["STRIDED"] = t
            gates.append(gate("R_STRIDED_probe_%s" % tag, True, False,
                              "the overlapping as_strided view RAN: %d of %d rows bitwise equal to F8 (option A); max "
                              "|err| %.3e; %.4f ms per layer (one call)" % (same, r_, err, t["call_ms"]["median"])))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_STRIDED_probe_%s" % tag, False, False, "REFUSED or failed: %s: %s"
                              % (type(e).__name__, str(e)[:300])))
    rep["seconds"] = round(time.time() - t0, 1)
    del arms, outs, case, bf, ba, bh, poisoned
    torch.cuda.empty_cache()
    return rep


# ------------------------------------------------------------------ modes
def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("main", "ncu-cell"), default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, help="the model's config.json (NOSA-8B)")
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy)")
    ap.add_argument("--sharedkv", default=None, help="sharedkv5's bi_paged_attention_shared.py (a pinned copy; S5)")
    ap.add_argument("--cells", default=",".join(cell_tag(b, r) for b, r in CELLS))
    ap.add_argument("--arms", default="P,F8,F0,FH,S5")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ncu", default="none", help="the ncu binary, or 'none'")
    ap.add_argument("--ncu-cap", type=int, default=NCU_CAP_S)
    ap.add_argument("--budget-s", type=int, default=2400)
    ap.add_argument("--no-strided", dest="strided", action="store_false")
    ap.add_argument("--bias-free", action="store_true", help="ncu-cell only: the default module, bias = 0")
    a = ap.parse_args(argv)
    a.cells = parse_cells(a.cells)
    a.arms = parse_arms(a.arms)
    if a.sample < 2 or a.warm < 0 or a.reps < 1 or a.budget_s < 60 or not 1 <= a.ncu_cap <= NCU_CAP_S:
        raise ValueError("--sample >= 2, --warm >= 0, --reps >= 1, --budget-s >= 60, --ncu-cap 1 .. %d" % NCU_CAP_S)
    if "S5" in a.arms and a.mode == "ncu-cell" and not a.sharedkv:
        raise ValueError("ncu-cell with S5 needs --sharedkv")
    return a


def refuse_common(a):
    """Refusals that need no GPU (before anything is created). None = go."""
    if os.path.exists(a.out):
        return "--out %s exists" % a.out
    for f, what in ((a.config, "--config"), (a.biattn, "--biattn")):
        if not os.path.isfile(f):
            return "%s %s is not a file" % (what, f)
    if a.sharedkv and not os.path.isfile(a.sharedkv):
        return "--sharedkv %s is not a file" % a.sharedkv
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not os.environ.get("FLASHINFER_WORKSPACE_BASE"):
        return "FLASHINFER_WORKSPACE_BASE is unset (FlashInfer would use HOME: lessons C1)"
    return None


def import_flashinfer():
    import flashinfer
    v = getattr(flashinfer, "__version__", "unknown")
    if not version_ok(v):
        raise RuntimeError("FlashInfer %s is not %s" % (v, FI_VERSION))
    return flashinfer, v


def mode_main(a):
    why = refuse_common(a)
    if why:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 2
    try:
        fi, fiv = import_flashinfer()
    except Exception as e:
        print("REFUSED: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    t_start = time.time()
    deadline = t_start + a.budget_s
    os.makedirs(a.out)
    geom = geometry_from_config(json.load(open(a.config)))
    p3 = load_by_path(a.biattn, "piece3_bi_paged_attention")
    skv = load_by_path(a.sharedkv, "sharedkv5_bi_paged_attention_shared") if a.sharedkv else None
    gates, report = [], dict(cells={}, ncu={}, findings=[])
    jit = jit_args_for(geom["D"])
    ws0 = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    try:
        t0 = time.time()
        fi.BatchDecodeWithPagedKVCacheWrapper(ws0, "NHD", use_tensor_cores=True, backend="fa2", jit_args=jit)
        build_s = time.time() - t0
        t1 = time.time()
        fi.BatchDecodeWithPagedKVCacheWrapper(ws0, "NHD", use_tensor_cores=True, backend="fa2", jit_args=jit)
        load_s = time.time() - t1
        report["jit"] = dict(build_s=round(build_s, 1), load_s=round(load_s, 2))
        gates.append(gate("C0_variant_jit_build", True, True, "R2FiCisBias built and loaded in %.1f s (%s); a second "
                          "wrapper loads in %.2f s" % (build_s, band_text(build_s, PRED["jit_build_s"]), load_s)))
    except Exception as e:
        tb = traceback.format_exc()
        with open(os.path.join(a.out, "jit_build_error.txt"), "w") as f:
            f.write(tb)
        report["findings"].append("the CIS bias variant did not build: %s: %s" % (type(e).__name__, str(e)[:500]))
        gates.append(gate("C0_variant_jit_build", False, True, "FINDING: the variant JIT build FAILED (%s: %s; "
                          "jit_build_error.txt); the bias-free comparison follows as a REPORT"
                          % (type(e).__name__, str(e)[:200])))
        jit = None
    del ws0
    for b, rpr in a.cells:
        tag = cell_tag(b, rpr)
        left = deadline - time.time()
        if left < CELL_RESERVE_S:
            gates.append(gate("BUDGET_cell_%s" % tag, False, True, "skipped: %d s left < %d s" % (left, CELL_RESERVE_S)))
            continue
        log("cell %s (%d s left)" % (tag, left))
        try:
            report["cells"][tag] = run_cell(fi, a, geom, p3, skv, jit, b, rpr, gates)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("CELL_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    if a.ncu != "none":
        try:
            run_ncu(a, geom, jit is None, deadline, gates, report)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_ncu", False, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    meta = dict(mode="main", cells=[cell_tag(*c) for c in a.cells], arms=a.arms, seed=a.seed, sample=a.sample,
                warm=a.warm, reps=a.reps, budget_s=a.budget_s, seconds=round(time.time() - t_start, 1), geom=geom,
                torch=torch.__version__, flashinfer=fiv, gpu=torch.cuda.get_device_name(0),
                commit=os.environ.get("FIATTN_COMMIT", "unknown"), biattn=a.biattn, sharedkv=a.sharedkv,
                bias_free=jit is None, tol_c=TOL_C, env={k: v for k, v in os.environ.items()
                                                         if k.startswith(("FLASHINFER_", "TVM_FFI", "TRITON_",
                                                                          "TORCH_", "MAX_JOBS", "TMPDIR"))})
    write_json(os.path.join(a.out, "summary.json"), dict(meta=meta, gates=gates, report=report, pred=PRED))
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                g["detail"][:500]))
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("r2_fiattn_bench: %d MUST gate(s) failed" % nfail)
    return min(nfail, 125)


def run_ncu(a, geom, bias_free, deadline, gates, report):
    if not os.access(a.ncu, os.X_OK):
        gates.append(gate("R_ncu", False, False, "no executable ncu at %s" % a.ncu))
        return
    rc, missing, metrics = ncu_metrics(a, a.out)
    if rc != 0 or missing:
        gates.append(gate("R_ncu_metrics", False, False, "query exit %s; missing %s" % (rc, missing)))
        return
    for b, rpr in [c for c in NCU_ORDER if c in a.cells]:
        tag = cell_tag(b, rpr)
        cap = ncu_cap(deadline - time.time(), a.ncu_cap)
        if cap is None:
            gates.append(gate("R_ncu_%s" % tag, False, False, "skipped: budget"))
            continue
        arms = [x for x in ("P", "F8", "F0") if x in a.arms]
        if rpr > 1 and a.sharedkv and "S5" in a.arms:
            arms.append("S5")
        child = os.path.join(a.out, "ncu_%s" % tag)
        rep = os.path.join(a.out, "ncu_%s_rep" % tag)
        t1 = time.time()
        rc = run_logged(ncu_argv(a, b, rpr, child, rep, metrics, arms, bias_free), os.path.join(a.out,
                        "ncu_%s.log" % tag), cap)
        exp = None
        summ = None
        if os.path.exists(rep + ".ncu-rep"):
            csv_path = os.path.join(a.out, "ncu_%s_raw.csv" % tag)
            exp = run_logged([a.ncu, "--import", rep + ".ncu-rep", "--csv", "--page", "raw", "--print-units", "base"],
                             csv_path, 300)
            if exp == 0:
                case = make_case(geom, b, rpr, a.seed, with_data=False)
                bf = case["full"]
                summ = ncu_summary(ncu_rows(open(csv_path).read(), metrics), arms, ideal_bytes(bf, geom),
                                   union_bytes(bf, geom, case["N"], case["S"]), rpr)
        report["ncu"][tag] = dict(child_exit=rc, export_exit=exp, cap_s=cap, seconds=round(time.time() - t1, 1),
                                  summary=summ)
        txt = "child exit %s (cap %d s), export %s" % (rc, cap, exp)
        if summ:
            txt += "; " + "; ".join("%s read %.3g B = %s of ideal, %s of union, %d kernels (expected %s); %s"
                                    % (x, p["read"], "%.3f" % p["read_over_ideal"], "%.3f" % p["read_over_union"],
                                       p["n_kernels"], p["expected_kernels"], p["band"])
                                    for x, p in summ["arms"].items())
        gates.append(gate("R_ncu_%s" % tag, rc == 0 and exp == 0, False, txt))


def mode_ncu_cell(a):
    """Run under ncu: one cell, every arm planned and warmed OUTSIDE the profiler window, then one call per arm
    inside it, each in an NVTX range fia|<arm>. Writes --out/cell.json."""
    why = refuse_common(a)
    if why:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 2
    fi, fiv = import_flashinfer()
    os.makedirs(a.out)
    geom = geometry_from_config(json.load(open(a.config)))
    p3 = load_by_path(a.biattn, "piece3_bi_paged_attention")
    skv = load_by_path(a.sharedkv, "sharedkv5_bi_paged_attention_shared") if a.sharedkv else None
    b, rpr = a.cells[0]
    case = case_to(make_case(geom, b, rpr, a.seed), "cuda")
    jit = None if a.bias_free else jit_args_for(geom["D"])
    if a.bias_free:
        case["bias"] = torch.zeros_like(case["bias"])
    bf = batch_of(case, "full")
    arms = build_arms(fi, a, geom, case, p3, skv, jit, a.arms, "cuda")
    for arm in arms.values():
        arm.prepare(bf)
        arm.launch()
    for arm in arms.values():
        arm.prepare(bf)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, arm in arms.items():
        torch.cuda.nvtx.range_push("fia|%s" % name)
        arm.launch()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    write_json(os.path.join(a.out, "cell.json"), dict(cell=cell_tag(b, rpr), arms=list(arms), flashinfer=fiv,
                                                      ideal_bytes=ideal_bytes(bf, geom),
                                                      union_bytes=union_bytes(bf, geom, case["N"], case["S"])))
    return 0


def main(argv=None):
    try:
        a = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return 2 if e.code else 0
    except ValueError as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return 2
    a.out = os.path.abspath(a.out)
    try:
        return mode_ncu_cell(a) if a.mode == "ncu-cell" else mode_main(a)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
