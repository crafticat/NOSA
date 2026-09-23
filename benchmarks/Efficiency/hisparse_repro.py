"""HiSparse LOW-INTERFERENCE REPRODUCTION, separate from NOSI (retroinfer-eval focused gather experiment,
authorized 2026-09-23). Question: does SGLang HiSparse's small-grid GPU copy (copy_cache_planned_kernel,
sglang 87db743, vendored verbatim in nosi/flash_cache_engine/hisparse_copy/) interfere little with a decode
step on OUR A100 PCIe when the step is a CUDA graph, the host buffer is GPU-local, and the payload is known?
Near-capacity NOSI integration is a different experiment (worker_sweep.py); nothing here touches NOSI's engine.

THE DECODE STEP (proxy of NOSA-8B, random weights of the real shapes, config.json of the model): per layer
rmsnorm, wqkv GEMM (M = B, 4096 -> 4608), no rope, attention of 32 query heads over a RESIDENT KV cache of
HR_ATTN_TOKENS (4096) tokens x 2 KV heads with flash_attn_nosa.flash_attn_with_kvcache (the kernel NOSI's decode
calls, imported as nosi/nosi/nosa_llama.py:42 imports it; kv_bias = 0, num_splits = HR_NUM_SPLITS = 4) or torch
SDPA when that import fails (the JSON says which), wo, residual, rmsnorm, gate_up (4096 -> 32768), silu-and-mul
(flashinfer, as NOSI; torch fallback recorded), down, residual; after 32 layers rmsnorm + lm_head (73448) in fp32.
NOT modelled: NOSA's block scoring / pooling / top-k and the engine bookkeeping (the proxy step is shorter than
NOSI's). Batch HR_B (default 16 = Strata Fig. 5's decode pass of 16 requests x 4k). Modes (HR_MODES): graph = the
whole step captured in ONE CUDA graph and replayed; eager = the same Python step, launched while the gate's GPU
sleep runs (host launch time never enters the bracket); eager_live = the host enqueues the step only after the
gate fires, as a real eager server. The side is launched eagerly on a side stream beside the replay (SGLang
captures its fork / join inside the graph: a recorded difference).

TWO PAYLOAD FAMILIES (HR_PAYLOAD), one decode step's miss volume over all layers = ONE PASS:
  nosi    NOSI's layout: pinned (L, B, S_cpu, H = 2, Dh = 128) K and V; per (layer, [head,] request)
          HR_MISS_PER_HEAD distinct host blocks of HR_BLOCK_ROWS rows (64 = NOSI's blocks; 1 = token rows),
          uniform over the context (seeded), into distinct destination slots of a HR_WINDOW_ROWS window (tail slot
          excluded); every layer carries the same count. HR_HEADS = coupled (both KV heads miss the same blocks
          into the same slots: every item kind is exact) | independent (NOSI's per-head slots; only i256 is exact).
          Sides:
            hisparse<W>_<item>_t<T>  the vendored kernel, IsMLA = false (K then V per item), W blocks x T threads,
                                    one launch per layer, plan built by hisparse_copy/plan.py from the SAME ids:
                                    i256 = one head's token row (THE NOSI ADAPTER: exact for any NOSI descriptor),
                                    i512 = one token row of both heads (HiSparse's token item for NOSI's KV),
                                    i32k = one whole (request, block) with both heads
            triton<W>               NOSI's throttled persistent Triton gather (flash_h2d_persistent.py), W CTAs
            dma2d                   cudaMemcpy2DAsync, per layer K and V: B rows of the SAME byte count (a contiguous
                                    prefix per request: a copy engine cannot take scattered records in one call;
                                    a link-ceiling reference, NOT a matched-id arm)
  native  HiSparse's OWN layout, no adapter (the native control): a pinned linear pool of HR_NATIVE_ITEM (1152 =
          576 x bf16, HiSparse's BF16 MLA token record) bytes per token and layer, B x HR_NATIVE_CTX (32768) tokens
          per layer, and a linear device buffer of B x HR_NATIVE_SLOTS (4096) records per layer; per (layer,
          request) HR_NATIVE_MISSES (274 = 13.4% of k = 2048, the published LRU miss rate) distinct scattered
          tokens (seeded) into distinct slots. Sides: hisparse<W>_native_t<T> = the vendored kernel with IsMLA =
          true (K only: SGLang's shipped copy_cache_planned_mla), one launch per layer; dma2d (the same bytes as
          one contiguous prefix per request and layer). The Triton gather does not apply (NOSI's per-head tile
          layout): recorded as skipped.
TWO REGIMES (HR_REGIMES), reported in SEPARATE tables:
  saturating  side sized to cover the CONCURRENT step (side_sizing.py): the Strata Fig. 5 analog (a continuous
              transfer stream co-running with a fixed decode pass)
  burst       passes fixed at ONE, no sizing: one step's misses per step, HiSparse's actual regime; reports whether
              the side finished inside the step. When sizing chose 1 pass the saturating row IS the burst row
              (recorded as same_as_saturating, not re-measured).
HOST BUFFERS: fresh pinned tensors; /proc/self/numa_maps pages per node over EVERY mapping each buffer touches
(numa_maps.py) and the process's Cpus_allowed / Mems_allowed are recorded; the sbatch binds numactl
--cpunodebind=0 --membind=0 (the GPU's node) and HR_REQUIRE_NUMA_NODE makes any page elsewhere a failure.
TIMER: the worker sweep's GPU-sleep-gated bracket (identical lines, test-enforced) plus live_main; host_lag = gate ->
step start must be <= 0.1 ms. Saturating VALID = every concurrent rep overlaps >= 0.95 of the step and host_lag
<= 0.1 ms; burst VALID = host_lag <= 0.1 ms. DURING-STEP (side_sizing.py, review B3): one event per launch unit
(layer) on the side stream -> bytes whose unit ended inside the step / step ms = a LOWER BOUND of the bandwidth
delivered during the step, next to the whole-interval GB/s (which includes the tail the side runs alone).
CORRECTNESS: before every transfer rep the destination records are zeroed; after it they must equal the source
records (torch.equal, all layers, K and V or K only); every step's logits must be torch.equal to that mode's
reference. CONTROL: the kernel with SkipIO = true must FAIL the transfer check. HR_NVTX = 1 wraps every concurrent
rep in an NVTX range 'gf|<arm>|<mode>|<regime>|<rep>' (the nsys stage attributes GPU kernels to cells by it).
Outputs HR_OUT/<HR_TAG>.json and .md; exit code = failures. `python hisparse_repro.py --compare DIR` renders the
cross-configuration adapter-overhead / workload comparison from DIR/*/*.json (DIR/compare.md, .json).
"""
import ctypes
import glob
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numa_maps as NM  # noqa: E402
import side_sizing as SZ  # noqa: E402

NOSI_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FCE = os.path.join(NOSI_ROOT, "nosi", "nosi", "flash_cache_engine")
CUDART = "/venv/nosa/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
# NOSA-8B, /mnt/beegfs/ojerbi/models/NOSA-8B/config.json
MODEL = dict(hidden=4096, n_heads=32, n_kv=2, head_dim=128, inter=16384, vocab=73448, layers=32, eps=1e-6)
GPU_CLOCK_HZ = 1.41e9          # the worker sweep's sleep calibration (A100 max SM clock)
PAYLOADS = ("nosi", "native")
REGIMES = ("saturating", "burst")


def _ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split())


@dataclass
class Config:
    B: int = 16
    layers: int = 32
    attn_tokens: int = 4096
    payload: str = "nosi"
    s_cpu: int = 16384
    block_rows: int = 64
    window_rows: int = 4096
    miss_per_head: float = 3.5
    heads: str = "coupled"
    native_item: int = 1152
    native_misses: int = 274
    native_ctx: int = 32768
    native_slots: int = 4096
    seed: int = 0
    hs_blocks: Tuple[int, ...] = (1, 2, 4, 8, 16)
    hs_threads: Tuple[int, ...] = (1024, 256)
    hs_blocks_t256: Tuple[int, ...] = (1, 4)
    hs_items: Tuple[str, ...] = ("i256", "i512", "i32k")
    triton_w: Tuple[int, ...] = (1, 2, 4, 8, 16)
    dma: bool = True
    modes: Tuple[str, ...] = ("graph", "eager", "eager_live")
    regimes: Tuple[str, ...] = REGIMES
    reps: int = 5
    sleep_ms: float = 15.0
    size_trials: int = 3
    num_splits: int = 4
    require_numa_node: Optional[int] = None
    nvtx: bool = False
    out: str = "."
    tag: str = "repro"


def config_from_env(env=None) -> Config:
    env = os.environ if env is None else env
    g = env.get
    rn = g("HR_REQUIRE_NUMA_NODE", "")
    cfg = Config(
        B=int(g("HR_B", "16")), layers=int(g("HR_LAYERS", "32")), attn_tokens=int(g("HR_ATTN_TOKENS", "4096")),
        payload=g("HR_PAYLOAD", "nosi"),
        s_cpu=int(g("HR_S_CPU", "16384")), block_rows=int(g("HR_BLOCK_ROWS", "64")), window_rows=int(g("HR_WINDOW_ROWS", "4096")),
        miss_per_head=float(g("HR_MISS_PER_HEAD", "3.5")), heads=g("HR_HEADS", "coupled"),
        native_item=int(g("HR_NATIVE_ITEM", "1152")), native_misses=int(g("HR_NATIVE_MISSES", "274")),
        native_ctx=int(g("HR_NATIVE_CTX", "32768")), native_slots=int(g("HR_NATIVE_SLOTS", "4096")), seed=int(g("HR_SEED", "0")),
        hs_blocks=_ints(g("HR_HS_BLOCKS", "1 2 4 8 16")), hs_threads=_ints(g("HR_HS_THREADS", "1024 256")),
        hs_blocks_t256=_ints(g("HR_HS_BLOCKS_T256", "1 4")),
        hs_items=tuple(g("HR_HS_ITEMS", "i256 i512 i32k").split()), triton_w=_ints(g("HR_TRITON_W", "1 2 4 8 16")),
        dma=g("HR_DMA", "1") == "1", modes=tuple(g("HR_MODES", "graph eager eager_live").split()),
        regimes=tuple(g("HR_REGIMES", "saturating burst").split()),
        reps=int(g("HR_REPS", "5")), sleep_ms=float(g("HR_SLEEP_MS", "15")), size_trials=int(g("HR_SIZE_TRIALS", "3")),
        num_splits=int(g("HR_NUM_SPLITS", "4")), require_numa_node=(int(rn) if rn.strip() else None), nvtx=g("HR_NVTX", "0") == "1",
        out=g("HR_OUT", "."), tag=g("HR_TAG", "repro"))
    if cfg.payload not in PAYLOADS:
        raise ValueError("HR_PAYLOAD must be one of %s, got %r" % (PAYLOADS, cfg.payload))
    bad = [r for r in cfg.regimes if r not in REGIMES]
    if bad or not cfg.regimes:
        raise ValueError("HR_REGIMES must be a non-empty subset of %s, got %r" % (REGIMES, cfg.regimes))
    return cfg


def blocks_for(cfg: Config, threads: int) -> Tuple[int, ...]:
    """The W list of the hisparse arms at `threads`: 256-thread blocks only at HR_HS_BLOCKS_T256 (their SM footprint
    is not W SMs, review NB2: a trimmed diagnostic), every other block size at HR_HS_BLOCKS."""
    return cfg.hs_blocks_t256 if threads == 256 else cfg.hs_blocks


def footprint(family: str, W: Optional[int], threads: Optional[int]) -> str:
    """What W means for the SMs (review NB2). copy_cache_planned_kernel at 1024 threads uses 48 registers x 1024 =
    49152 of an SM's 65536, so at most one copy CTA per SM: W CTAs occupy W SMs. At 256 threads (64 registers) up
    to 4 copy CTAs share an SM: W is a block count, not an SM count. The Triton gather's 128-thread CTAs likewise."""
    if family == "hisparse":
        return ("%d SMs" % W) if threads == 1024 else ("%d blocks (not W SMs)" % W)
    if family == "triton":
        return "%d CTAs x 128 thr (not W SMs)" % W
    if family == "dma2d":
        return "copy engine"
    return "-"


# --------------------------------------------------------------------------- payload (pure, CPU-tested)
def make_payload(L: int, B: int, H: int, n_src_blocks: int, n_slots: int, miss_per_head: float, heads: str, seed: int,
                 exclude_tail: bool = True) -> Dict:
    """Seeded NOSI-style miss descriptors for one decode step: ids (L, H, B, n_slots) int32, -1 = no load,
    else the host block that destination slot receives. Per layer, exactly round(frac x cells) cells carry
    floor(miss_per_head) + 1 misses and the rest floor(miss_per_head) (cell = (request) when heads =
    'coupled', (head, request) when 'independent'); host blocks distinct within a cell, uniform over the
    context; destination slots distinct within a cell, the tail slot (n_slots - 1) excluded."""
    if heads not in ("coupled", "independent"):
        raise ValueError("heads must be 'coupled' or 'independent', got %r" % heads)
    g = torch.Generator().manual_seed(int(seed))
    base = int(math.floor(miss_per_head))
    frac = miss_per_head - base
    per_layer = B if heads == "coupled" else H * B
    n_extra = int(round(frac * per_layer))
    counts = torch.full((L, per_layer), base, dtype=torch.int64)
    for l in range(L):
        counts[l, torch.randperm(per_layer, generator=g)[:n_extra]] += 1
    counts = counts.reshape(-1)
    n_cells = counts.numel()
    usable = n_slots - 1 if exclude_tail else n_slots
    cmax = int(counts.max()) if n_cells else 0
    if cmax > usable or cmax > n_src_blocks:
        raise ValueError("%d misses per cell do not fit %d slots / %d host blocks" % (cmax, usable, n_src_blocks))
    ids_cells = torch.full((n_cells, n_slots), -1, dtype=torch.int32)
    if cmax:
        src = torch.rand(n_cells, n_src_blocks, generator=g).topk(cmax, dim=1).indices
        dst = torch.rand(n_cells, usable, generator=g).topk(cmax, dim=1).indices
        keep = torch.arange(cmax)[None, :] < counts[:, None]
        cell = torch.arange(n_cells)[:, None].expand(-1, cmax)
        ids_cells[cell[keep], dst[keep]] = src[keep].to(torch.int32)
    if heads == "coupled":
        ids = ids_cells.view(L, 1, B, n_slots).expand(L, H, B, n_slots).contiguous()
    else:
        ids = ids_cells.view(L, H, B, n_slots).contiguous()
    head_blocks = int((ids >= 0).sum())
    return dict(ids=ids, heads=heads, seed=int(seed), base=base, extra_cells_per_layer=n_extra,
                miss_per_head_mean=head_blocks / float(L * H * B), head_blocks=head_blocks,
                head_blocks_per_layer=[int((ids[l] >= 0).sum()) for l in range(L)])


def payload_bytes(head_blocks: int, block_rows: int, head_dim: int, elem: int) -> int:
    """K and V bytes of one pass (NOSI's per-head tiles: rows x Dh x elem each)."""
    return 2 * head_blocks * block_rows * head_dim * elem


def native_pass_bytes(L: int, B: int, misses: int, item: int) -> int:
    """K-only bytes of one native pass: every layer, every request, `misses` records of `item` bytes."""
    return L * B * misses * item


def native_index(plans, B: int, ctx: int, slots: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure: the per-layer native plans -> global record indices into the (L * B * ctx) host pool and the
    (L * B * slots) device buffer, every planned item of every layer (the correctness check's gather indices)."""
    gsrc, gdst = [], []
    for l, p in enumerate(plans):
        R, P = p.src.shape
        used = torch.arange(P)[None, :] < p.counts.cpu().to(torch.int64)[:R, None]
        gsrc.append(l * B * ctx + p.src.cpu()[used])
        gdst.append(l * B * slots + p.dst.cpu().to(torch.int64)[used])
    return torch.cat(gsrc), torch.cat(gdst)


def dma_width_bytes(head_blocks_per_layer: List[int], B: int, block_rows: int, head_dim: int, elem: int) -> int:
    """Per-request row width of the matched cudaMemcpy2DAsync (one call per layer and tensor, height B): the
    layer's per-tensor payload / B. Every layer must carry the same payload and it must split evenly."""
    per_layer = {n * block_rows * head_dim * elem for n in head_blocks_per_layer}
    if len(per_layer) != 1:
        raise ValueError("layers carry different payloads: %s" % sorted(per_layer))
    total = per_layer.pop()
    if total % B or (total // B) % elem:
        raise ValueError("per-layer payload %d B does not split into %d equal rows of whole elements" % (total, B))
    return total // B


# --------------------------------------------------------------------------- placement records (pure)
def numa_pages(ptr: int):
    """pages per NUMA node of the mapping that contains `ptr` alone (kept for comparison with numa_range)."""
    return NM.pages_at(ptr)


def numa_range(ptr: int, nbytes: int, vmas=None):
    """pages per NUMA node over EVERY mapping the buffer [ptr, ptr + nbytes) touches (review NB11)."""
    try:
        return NM.range_pages(NM.parse() if vmas is None else vmas, ptr, nbytes)
    except OSError as e:
        return dict(error=str(e))


def proc_numa() -> Dict:
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
                k, v = line.split(":", 1)
                out[k] = v.strip()
    except OSError as e:
        out["error"] = str(e)
    return out


def numa_local(pages: Optional[Dict], node: int) -> bool:
    """Every counted page on `node` (and at least one page counted)."""
    return NM.all_on(pages, node)


# --------------------------------------------------------------------------- summary / tables (pure)
def _med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if len(xs) else float("nan")


def _p95(xs):
    return float(np.percentile(xs, 95)) if len(xs) else float("nan")


def summarize(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        regime = r.get("regime", "saturating")
        if r.get("error"):
            out.append(dict(arm=r["arm"], mode=r["mode"], regime=regime, error=r["error"]))
            continue
        ra = [r["bytes"] / (x * 1e6) for x in r["side_alone_ms"]]
        rc = [r["bytes"] / (x * 1e6) for x in r["side_conc_ms"]]
        da, dc = r["decode_alone_ms"], r["decode_conc_ms"]
        ends = r.get("last_end_ms") or list(r["side_conc_ms"])
        inside = sum(1 for e, m in zip(ends, dc) if e is not None and e == e and e <= m)
        slow = _med(dc) / _med(da) - 1 if da and dc else float("nan")
        sizing = r.get("sizing") or {}
        valid = SZ.valid(r["overlap_frac"], r["host_lag_ms"]) if regime == "saturating" else SZ.lag_ok(r["host_lag_ms"])
        out.append(dict(arm=r["arm"], mode=r["mode"], regime=regime, family=r["family"], W=r.get("W"), threads=r.get("threads"),
                        item=r.get("item"), footprint=r.get("footprint") or footprint(r["family"], r.get("W"), r.get("threads")),
                        passes=r["passes"], covered=sizing.get("covered") if regime == "saturating" else None,
                        same_as_saturating=bool(r.get("same_as_saturating")),
                        gbps_alone=_med(ra), gbps_conc=_med(rc), gbps_during=_med(r.get("during_gbps") or []),
                        inside_frac=_med(r.get("inside_frac") or []), side_alone_ms=_med(r["side_alone_ms"]), side_end_ms=_med(ends),
                        finished_inside=inside, n_conc=len(dc),
                        decode_alone_med=_med(da), decode_alone_p95=_p95(da), decode_conc_med=_med(dc), decode_conc_p95=_p95(dc),
                        slowdown=slow, norm_tput=(_med(da) / _med(dc) if da and dc else float("nan")),
                        min_overlap=min(r["overlap_frac"]) if r["overlap_frac"] else float("nan"),
                        max_host_lag=max(r["host_lag_ms"]) if r["host_lag_ms"] else float("nan"),
                        valid=valid, correct=r["fails"] == 0, fails=r["fails"]))
    return out


def _f(x, fmt="%.1f"):
    return "-" if x is None or x != x else fmt % x


def _cov(c):
    return "-" if c is None else ("yes" if c else "NO")


def render_saturating(summary: List[Dict]) -> List[str]:
    L = ["| arm | footprint | mode | passes | covered | transfer alone GB/s | transfer concurrent GB/s (whole interval) "
         "| during-step GB/s (lower bound) | side bytes inside step | decode alone median / p95 ms | decode concurrent median / p95 ms "
         "| slowdown | normalized decode throughput | min overlap | max host lag ms | valid | correct |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if s.get("regime", "saturating") != "saturating":
            continue
        if "error" in s:
            L.append("| %s | - | %s | - | - | - | - | - | - | - | - | - | - | - | - | - | ERROR: %s |" % (s["arm"], s["mode"], str(s["error"])[:120].replace("|", "/")))
            continue
        L.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %.2f / %.2f | %.2f / %.2f | %+.1f%% | %.1f%% | %.2f | %.3f | %s | %s |" % (
            s["arm"], s["footprint"], s["mode"], s["passes"], _cov(s["covered"]), _f(s["gbps_alone"]), _f(s["gbps_conc"]), _f(s["gbps_during"]),
            _f(100 * s["inside_frac"] if s["inside_frac"] == s["inside_frac"] else float("nan"), "%.0f%%"),
            s["decode_alone_med"], s["decode_alone_p95"], s["decode_conc_med"], s["decode_conc_p95"], 100 * s["slowdown"], 100 * s["norm_tput"],
            s["min_overlap"], s["max_host_lag"], "yes" if s["valid"] else "INVALID", "yes" if s["correct"] else "NO (%d)" % s["fails"]))
    return L


def render_burst(summary: List[Dict]) -> List[str]:
    L = ["| arm | footprint | mode | side alone ms | side end ms after step start (median) | finished inside step | during-step GB/s (lower bound) "
         "| side bytes inside step | decode alone median / p95 ms | decode concurrent median / p95 ms | slowdown | normalized decode throughput "
         "| max host lag ms | valid | correct | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if s.get("regime") != "burst":
            continue
        if "error" in s:
            L.append("| %s | - | %s | - | - | - | - | - | - | - | - | - | - | - | ERROR: %s | |" % (s["arm"], s["mode"], str(s["error"])[:120].replace("|", "/")))
            continue
        L.append("| %s | %s | %s | %.2f | %.2f | %d/%d | %s | %s | %.2f / %.2f | %.2f / %.2f | %+.1f%% | %.1f%% | %.3f | %s | %s | %s |" % (
            s["arm"], s["footprint"], s["mode"], s["side_alone_ms"], s["side_end_ms"], s["finished_inside"], s["n_conc"], _f(s["gbps_during"]),
            _f(100 * s["inside_frac"] if s["inside_frac"] == s["inside_frac"] else float("nan"), "%.0f%%"),
            s["decode_alone_med"], s["decode_alone_p95"], s["decode_conc_med"], s["decode_conc_p95"], 100 * s["slowdown"], 100 * s["norm_tput"],
            s["max_host_lag"], "yes" if s["valid"] else "INVALID", "yes" if s["correct"] else "NO (%d)" % s["fails"],
            "same as saturating (sizing chose 1 pass)" if s["same_as_saturating"] else ""))
    return L


def render_modes(summary: List[Dict], modes: List[str], regime: str) -> List[str]:
    sel = [s for s in summary if s.get("regime", "saturating") == regime]
    modes = [m for m in modes if any(s.get("mode") == m for s in sel)]
    arms = []
    for s in sel:
        if s["arm"] not in arms:
            arms.append(s["arm"])
    L = ["| arm | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    by = {(s["arm"], s["mode"]): s for s in sel}
    for a in arms:
        cells = []
        for m in modes:
            s = by.get((a, m))
            if s is None or "error" in s:
                cells.append("-" if s is None else "ERROR")
            else:
                cells.append("%+.1f%% (%s)%s" % (100 * s["slowdown"], _f(s["gbps_during"] if regime == "burst" else s["gbps_conc"]),
                                                  "" if s["valid"] else " INVALID"))
        L.append("| %s | %s |" % (a, " | ".join(cells)))
    return L


def adapter_pairs(summary: List[Dict], a_item: str = "i256", b_item: str = "i512") -> List[Dict]:
    """Paired hisparse rows that differ ONLY in the item (same W, threads, mode, regime, and by construction the same
    bytes into the same destinations): delta = a - b."""
    idx = {(s["W"], s["threads"], s["mode"], s.get("regime", "saturating"), s["item"]): s
           for s in summary if "error" not in s and s.get("family") == "hisparse"}
    out = []
    for (W, T, mode, regime, item), a in sorted(idx.items(), key=lambda kv: (kv[0][3], kv[0][2], kv[0][1], kv[0][0])):
        if item != a_item:
            continue
        b = idx.get((W, T, mode, regime, b_item))
        if b is None:
            continue
        out.append(dict(W=W, threads=T, mode=mode, regime=regime, a=a_item, b=b_item,
                        d_gbps_alone=a["gbps_alone"] - b["gbps_alone"], d_gbps_conc=a["gbps_conc"] - b["gbps_conc"],
                        d_gbps_during=a["gbps_during"] - b["gbps_during"], d_slowdown_pp=100 * (a["slowdown"] - b["slowdown"]),
                        a_gbps_alone=a["gbps_alone"], b_gbps_alone=b["gbps_alone"], a_slowdown=a["slowdown"], b_slowdown=b["slowdown"]))
    return out


def render_adapter(summary: List[Dict], plans: Dict) -> List[str]:
    pairs = adapter_pairs(summary)
    if not pairs:
        return []
    pa, pb = plans.get("i256") or {}, plans.get("i512") or {}
    L = ["", "## Adapter overhead: i256 (NOSI's per-head adapter) minus i512 (one token row of both heads) on IDENTICAL bytes and destinations", "",
         "Plan build per pass (all layers): i256 host %s ms, upload %s ms, device %s ms (%s items); i512 host %s ms, upload %s ms, device %s ms (%s items)." % (
             _f(pa.get("build_host_ms"), "%.2f"), _f(pa.get("upload_ms"), "%.2f"), _f(pa.get("build_device_ms"), "%.2f"), pa.get("n_items", "-"),
             _f(pb.get("build_host_ms"), "%.2f"), _f(pb.get("upload_ms"), "%.2f"), _f(pb.get("build_device_ms"), "%.2f"), pb.get("n_items", "-")), "",
         "| regime | mode | W | threads | i256 alone GB/s | i512 alone GB/s | delta alone GB/s | delta concurrent GB/s | delta during-step GB/s "
         "| i256 slowdown | i512 slowdown | delta slowdown (pp) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %+.1f%% | %+.1f%% | %+.1f |" % (
            p["regime"], p["mode"], p["W"], p["threads"], _f(p["a_gbps_alone"]), _f(p["b_gbps_alone"]), _f(p["d_gbps_alone"], "%+.1f"),
            _f(p["d_gbps_conc"], "%+.1f"), _f(p["d_gbps_during"], "%+.1f"), 100 * p["a_slowdown"], 100 * p["b_slowdown"], p["d_slowdown_pp"]))
    return L


def render(summary: List[Dict], meta: Dict) -> str:
    cfg = meta.get("config", {})
    if cfg.get("payload") == "native":
        pay = ("NATIVE HiSparse layout (no adapter): %s-byte K-only records (IsMLA), %s scattered tokens per (layer, request) of a %s-token "
               "context into a %s-slot buffer, seed %s; %.4f GB per pass" % (cfg.get("native_item"), cfg.get("native_misses"), cfg.get("native_ctx"),
                                                                           cfg.get("native_slots"), cfg.get("seed"), meta.get("pass_bytes", 0) / 1e9))
    else:
        pay = ("NOSI layout: %s misses per (layer, head, request) of %s-row blocks, heads %s, seed %s; %.4f GB per pass (K+V)"
               % (cfg.get("miss_per_head"), cfg.get("block_rows"), cfg.get("heads"), cfg.get("seed"), meta.get("pass_bytes", 0) / 1e9))
    numa = meta.get("numa") or {}
    L = ["# HiSparse low-interference reproduction [%s]: decode-step proxy (NOSA-8B shapes, random weights) + side-stream transfer" % cfg.get("tag", ""), "",
         "B=%s, %s layers, attention over %s resident tokens (%s, num_splits %s). Payload: %s. Host buffer pages per node (every mapping touched): %s; "
         "process %s." % (cfg.get("B"), cfg.get("layers"), cfg.get("attn_tokens"), meta.get("attn_backend"), cfg.get("num_splits"), pay,
                          json.dumps({k: (v or {}).get("pages_per_node") for k, v in numa.get("buffers", {}).items()}), json.dumps(numa.get("proc"))), "",
         "Medians (p95) over reps. Transfer GB/s = bytes / side interval (per rep, then median; includes the tail after the step). During-step "
         "GB/s = bytes of the per-layer launches that ENDED inside the step / step ms (a lower bound). slowdown = median concurrent / median "
         "alone - 1; normalized decode throughput = alone / concurrent (Strata Fig. 5's y axis). Saturating VALID = min overlap >= %.2f and "
         "max host lag <= %.1f ms; burst VALID = host lag only. CORRECT = every transfer rep's destination equals the source and every "
         "step's logits equal the mode's reference." % (SZ.OVERLAP_MIN, SZ.HOST_LAG_MAX_MS), "",
         "## SATURATING regime (the Strata Fig. 5 analog: the side is sized to cover the concurrent step)", ""]
    L += render_saturating(summary)
    L += ["", "## BURST regime (HiSparse's regime: ONE pass = one step's misses, no sizing)", ""]
    L += render_burst(summary)
    for regime in ("saturating", "burst"):
        if any(s.get("regime", "saturating") == regime for s in summary):
            L += ["", "## Decode modes side by side, %s: slowdown (%s)" % (regime, "during-step GB/s" if regime == "burst" else "concurrent GB/s"), ""]
            L += render_modes(summary, list(cfg.get("modes", [])), regime)
    L += render_adapter(summary, meta.get("plans") or {})
    L += ["", "## Plan build per pass (all layers; outside every bracket)", "",
          "| item | items per pass | item bytes | K only | host build ms | upload ms | device build ms | device == host |", "|---|---|---|---|---|---|---|---|"]
    for it, t in (meta.get("plans") or {}).items():
        if "refused" in t:
            L.append("| %s | REFUSED: %s | | | | | | |" % (it, t["refused"][:160]))
        else:
            L.append("| %s | %d | %d | %s | %s | %s | %s | %s |" % (it, t["n_items"], t["item_size_bytes"], bool(t.get("k_only")),
                                                                  _f(t.get("build_host_ms"), "%.2f"), _f(t.get("upload_ms"), "%.2f"),
                                                                  _f(t.get("build_device_ms"), "%.2f"), "n/a" if t.get("device_build_equal") is None else t["device_build_equal"]))
    L += ["", "## Controls, skipped arms and failures", ""]
    for k, v in (meta.get("controls") or {}).items():
        L.append("- %s: %s" % (k, json.dumps(v)))
    for sk in meta.get("skipped") or []:
        L.append("- skipped: %s" % sk)
    L.append("- failures: %d" % meta.get("failures", -1))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- cross-configuration comparison (pure)
def compare(results: List[Dict]) -> Dict:
    """results = the harness JSONs of several configurations. Returns the adapter pairs (i256 - i512) of every
    configuration that has both, and the native-vs-i512 workload comparison between a native configuration and a
    token-granular NOSI configuration at the same batch (matched W, threads, mode, regime)."""
    adapter, workload = [], []
    by_tag = {}
    for res in results:
        cfg = res["meta"]["config"]
        by_tag[cfg["tag"]] = res
        for p in adapter_pairs(res["summary"]):
            p.update(config=cfg["tag"], payload=cfg["payload"], B=cfg["B"], block_rows=cfg.get("block_rows"),
                     i256_plan=(res["meta"].get("plans") or {}).get("i256"), i512_plan=(res["meta"].get("plans") or {}).get("i512"))
            adapter.append(p)
    natives = [r for r in results if r["meta"]["config"]["payload"] == "native"]
    tokens = [r for r in results if r["meta"]["config"]["payload"] == "nosi" and r["meta"]["config"].get("block_rows") == 1]
    for nr in natives:
        for tr in tokens:
            if tr["meta"]["config"]["B"] != nr["meta"]["config"]["B"]:
                continue
            ti = {(s["W"], s["threads"], s["mode"], s.get("regime", "saturating")): s for s in tr["summary"]
                  if "error" not in s and s.get("item") == "i512"}
            for s in nr["summary"]:
                if "error" in s or s.get("family") != "hisparse":
                    continue
                t = ti.get((s["W"], s["threads"], s["mode"], s.get("regime", "saturating")))
                if t is None:
                    continue
                workload.append(dict(native_config=nr["meta"]["config"]["tag"], token_config=tr["meta"]["config"]["tag"], W=s["W"], threads=s["threads"],
                                     mode=s["mode"], regime=s.get("regime", "saturating"),
                                     native_item_bytes=nr["meta"]["config"]["native_item"], native_k_only=True, native_pass_bytes=nr["meta"]["pass_bytes"],
                                     i512_item_bytes=512, i512_k_and_v=True, i512_pass_bytes=tr["meta"]["pass_bytes"],
                                     native_gbps_alone=s["gbps_alone"], i512_gbps_alone=t["gbps_alone"], native_gbps_conc=s["gbps_conc"],
                                     i512_gbps_conc=t["gbps_conc"], native_gbps_during=s["gbps_during"], i512_gbps_during=t["gbps_during"],
                                     native_slowdown=s["slowdown"], i512_slowdown=t["slowdown"]))
    return dict(adapter=adapter, workload=workload, configs=sorted(by_tag))


def render_compare(c: Dict) -> str:
    L = ["# Gather focus, stage R across configurations", "",
         "## Adapter overhead (i256 = NOSI's per-head adapter, minus i512 = one token row of both heads; identical bytes and destinations)", "",
         "| config | B | block rows | regime | mode | W | threads | i256 alone GB/s | i512 alone GB/s | delta alone | delta concurrent | delta during-step "
         "| i256 slowdown | i512 slowdown | delta slowdown (pp) | plan build host ms i256 / i512 | device build ms i256 / i512 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in c["adapter"]:
        pa, pb = p.get("i256_plan") or {}, p.get("i512_plan") or {}
        L.append("| %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %s | %s | %+.1f%% | %+.1f%% | %+.1f | %s / %s | %s / %s |" % (
            p["config"], p["B"], p["block_rows"], p["regime"], p["mode"], p["W"], p["threads"], _f(p["a_gbps_alone"]), _f(p["b_gbps_alone"]),
            _f(p["d_gbps_alone"], "%+.1f"), _f(p["d_gbps_conc"], "%+.1f"), _f(p["d_gbps_during"], "%+.1f"), 100 * p["a_slowdown"], 100 * p["b_slowdown"],
            p["d_slowdown_pp"], _f(pa.get("build_host_ms"), "%.2f"), _f(pb.get("build_host_ms"), "%.2f"), _f(pa.get("build_device_ms"), "%.2f"),
            _f(pb.get("build_device_ms"), "%.2f")))
    L += ["", "## Workload difference, NOT an adapter cost: HiSparse's native record (1152 B, K only, IsMLA) vs i512 token rows of NOSI's KV "
          "(512 B, K then V per item), same batch, same misses per request and layer", "",
          "| native config | token config | regime | mode | W | threads | native pass MB | i512 pass MB | native alone GB/s | i512 alone GB/s "
          "| native concurrent GB/s | i512 concurrent GB/s | native during-step GB/s | i512 during-step GB/s | native slowdown | i512 slowdown |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in c["workload"]:
        L.append("| %s | %s | %s | %s | %d | %d | %.1f | %.1f | %s | %s | %s | %s | %s | %s | %+.1f%% | %+.1f%% |" % (
            w["native_config"], w["token_config"], w["regime"], w["mode"], w["W"], w["threads"], w["native_pass_bytes"] / 1e6, w["i512_pass_bytes"] / 1e6,
            _f(w["native_gbps_alone"]), _f(w["i512_gbps_alone"]), _f(w["native_gbps_conc"]), _f(w["i512_gbps_conc"]), _f(w["native_gbps_during"]),
            _f(w["i512_gbps_during"]), 100 * w["native_slowdown"], 100 * w["i512_slowdown"]))
    L.append("")
    L.append("Configurations read: %s" % ", ".join(c["configs"]))
    return "\n".join(L) + "\n"


def compare_dir(d: str) -> int:
    paths = sorted(glob.glob(os.path.join(d, "*", "*.json")))
    results = []
    for p in paths:
        try:
            res = json.load(open(p))
        except (OSError, ValueError) as e:
            print("[compare] skip %s: %r" % (p, e), flush=True)
            continue
        if isinstance(res, dict) and "summary" in res and "meta" in res:
            results.append(res)
    c = compare(results)
    json.dump(c, open(os.path.join(d, "compare.json"), "w"), indent=1)
    text = render_compare(c)
    open(os.path.join(d, "compare.md"), "w").write(text)
    print(text, flush=True)
    return 0


# --------------------------------------------------------------------------- GPU side
def _load_by_path(name: str, path: str, package_dir: Optional[str] = None):
    kw = dict(submodule_search_locations=[package_dir]) if package_dir else {}
    spec = importlib.util.spec_from_file_location(name, path, **kw)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_hisparse_copy():
    d = os.path.join(FCE, "hisparse_copy")
    return _load_by_path("hisparse_copy", os.path.join(d, "__init__.py"), d)


def load_persistent_gather():
    return _load_by_path("flash_h2d_persistent", os.path.join(FCE, "flash_h2d_persistent.py")).flash_h2d_persistent


def _cudart():
    cudart = ctypes.CDLL(CUDART)
    cudart.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cudart.cudaMemcpy2DAsync.restype = ctypes.c_int
    return cudart


class Proxy:
    """The decode-step proxy. step() writes self.logits in place; capture() records step() in one CUDA graph.
    device / model / force_torch exist for the CPU smoke test (tiny shapes, torch ops); the run uses the defaults."""

    def __init__(self, cfg: Config, device: str = "cuda", model: Optional[Dict] = None, force_torch: bool = False):
        import torch.nn.functional as F
        self.F = F
        m = MODEL if model is None else model
        self.B, self.L, self.T = cfg.B, cfg.layers, cfg.attn_tokens
        self.nh, self.nkv, self.hd, self.hid, self.inter = m["n_heads"], m["n_kv"], m["head_dim"], m["hidden"], m["inter"]
        self.eps, self.splits = m["eps"], cfg.num_splits
        dev, dt = device, torch.bfloat16
        g = torch.Generator(device=dev).manual_seed(cfg.seed + 1)

        def w(out_f, in_f):
            return torch.randn(out_f, in_f, generator=g, device=dev, dtype=dt).mul_(in_f ** -0.5)
        qkv_out = (self.nh + 2 * self.nkv) * self.hd
        self.wqkv = [w(qkv_out, self.hid) for _ in range(self.L)]
        self.wo = [w(self.hid, self.nh * self.hd) for _ in range(self.L)]
        self.wgu = [w(2 * self.inter, self.hid) for _ in range(self.L)]
        self.wd = [w(self.hid, self.inter) for _ in range(self.L)]
        self.lm_head = w(m["vocab"], self.hid)
        self.ln1 = [torch.ones(self.hid, device=dev, dtype=dt) for _ in range(self.L)]
        self.ln2 = [torch.ones(self.hid, device=dev, dtype=dt) for _ in range(self.L)]
        self.lnf = torch.ones(self.hid, device=dev, dtype=dt)
        self.kc = [torch.randn(self.B, self.T, self.nkv, self.hd, generator=g, device=dev, dtype=dt) for _ in range(self.L)]
        self.vc = [torch.randn(self.B, self.T, self.nkv, self.hd, generator=g, device=dev, dtype=dt) for _ in range(self.L)]
        self.kvb = [torch.zeros(self.B, self.T, self.nkv, device=dev, dtype=dt) for _ in range(self.L)]
        self.lens = torch.full((self.B,), self.T, dtype=torch.int32, device=dev)
        self.x_in = torch.randn(self.B, self.hid, generator=g, device=dev, dtype=dt)
        self.logits = torch.empty(self.B, m["vocab"], device=dev, dtype=torch.float32)
        self.graph = None
        try:
            if force_torch:
                raise ImportError("force_torch")
            from flashinfer.norm import rmsnorm as _rms
            from flashinfer.activation import silu_and_mul as _sm
            self._rms, self._sm, self.norm_backend = _rms, _sm, "flashinfer (rmsnorm, silu_and_mul; as nosa_llama.py:4-5)"
        except Exception as e:  # noqa: BLE001
            self._rms, self._sm, self.norm_backend = None, None, "torch (flashinfer import failed: %s)" % e
        try:
            if force_torch:
                raise ImportError("force_torch")
            from flash_attn_nosa import flash_attn_with_kvcache as _fa
            self._fa, self.attn_backend = _fa, "flash_attn_nosa.flash_attn_with_kvcache (as nosa_llama.py:42)"
        except Exception as e:  # noqa: BLE001
            self._fa, self.attn_backend = None, "torch SDPA enable_gqa (flash_attn_nosa import failed: %s)" % e

    def weight_bytes(self) -> int:
        ts = self.wqkv + self.wo + self.wgu + self.wd + [self.lm_head]
        return sum(t.numel() * t.element_size() for t in ts)

    def rms(self, x, w):
        if self._rms is not None:
            return self._rms(x, w, self.eps)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def silu_mul(self, gu, out):
        if self._sm is not None:
            self._sm(gu, out)
        else:
            out.copy_(self.F.silu(gu[:, : self.inter]) * gu[:, self.inter:])

    def attn(self, q, l):
        if self._fa is not None:
            return self._fa(q, self.kc[l], self.vc[l], self.kvb[l], cache_seqlens=self.lens, num_splits=self.splits)
        o = self.F.scaled_dot_product_attention(q.transpose(1, 2), self.kc[l].transpose(1, 2), self.vc[l].transpose(1, 2), enable_gqa=True)
        return o.transpose(1, 2)

    def step(self):
        F, B = self.F, self.B
        h = self.x_in
        for l in range(self.L):
            r = h
            x = self.rms(h, self.ln1[l])
            qkv = F.linear(x, self.wqkv[l])
            q = qkv[:, : self.nh * self.hd].contiguous().view(B, 1, self.nh, self.hd)   # one small copy: stands in for NOSI's rope write of q
            o = self.attn(q, l)
            h = r + F.linear(o.reshape(B, self.nh * self.hd), self.wo[l])
            r = h
            x = self.rms(h, self.ln2[l])
            gu = F.linear(x, self.wgu[l])
            a = torch.empty((B, self.inter), dtype=gu.dtype, device=gu.device)
            self.silu_mul(gu, a)
            h = r + F.linear(a, self.wd[l])
        x = self.rms(h, self.lnf)
        self.logits.copy_(F.linear(x, self.lm_head).float())
        return self.logits

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step()
        torch.cuda.synchronize()
        self.graph = g


def make_bracket(main, side, sleep_cycles):
    """The worker sweep's GPU-sleep-gated bracket (worker_sweep.py bracket(), identical lines) plus live_main: GPU sleep
    on main -> gate -> side enqueued behind the gate -> [live_main: host waits for the gate] -> t0 -> main work. The
    side functions mark SZ.LAUNCH_LOG after every launch unit; a concurrent bracket adds the during-step record."""
    def bracket(fn_main, fn_side, live_main=False):
        torch.cuda.synchronize()
        SZ.LAUNCH_LOG.reset()
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("pre", "gate", "t0", "tm", "ts")}
        ev["pre"].record(main)
        torch.cuda._sleep(sleep_cycles)
        ev["gate"].record(main)
        h0 = time.perf_counter()
        if fn_side is not None:
            side.wait_event(ev["gate"])
            with torch.cuda.stream(side):
                fn_side()
                ev["ts"].record(side)
        host_enqueue_ms = 1000 * (time.perf_counter() - h0)
        if live_main:
            ev["gate"].synchronize()
        ev["t0"].record(main)
        out = fn_main() if fn_main is not None else None
        ev["tm"].record(main)
        torch.cuda.synchronize()
        r = dict(host_lag_ms=ev["gate"].elapsed_time(ev["t0"]), sleep_ms=ev["pre"].elapsed_time(ev["gate"]), host_enqueue_ms=host_enqueue_ms)
        if fn_main is not None:
            r["main_ms"] = ev["t0"].elapsed_time(ev["tm"])
        if fn_side is not None:
            r["side_ms"] = ev["gate"].elapsed_time(ev["ts"])
        if fn_main is not None and fn_side is not None:
            r["overlap_frac"] = SZ.overlap_frac(r["side_ms"], r["host_lag_ms"], r["main_ms"])
        if fn_main is not None and fn_side is not None and SZ.LAUNCH_LOG.n:
            r.update(SZ.during_step([ev["gate"].elapsed_time(e) for e, _ in SZ.LAUNCH_LOG.marks()], [b for _, b in SZ.LAUNCH_LOG.marks()], r["host_lag_ms"], r["main_ms"]))
        return out, r
    return bracket


def _fill_pinned(t: torch.Tensor, seed: int) -> None:
    """Fill a pinned host tensor layer by layer from the GPU (dim 0 = layers)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    for l in range(t.shape[0]):
        t[l].copy_(torch.randn(t.shape[1:], generator=g, device="cuda", dtype=t.dtype))
    torch.cuda.synchronize()


def _plan_totals(per, tms) -> Dict:
    tot = dict(build_host_ms=0.0, upload_ms=0.0, build_device_ms=0.0, n_items=0, device_build_equal=True)
    for tm in tms:
        for k in ("build_host_ms", "upload_ms", "n_items"):
            tot[k] += tm[k]
        if tm.get("build_device_ms") is None:
            tot["build_device_ms"], tot["device_build_equal"] = None, None
        elif tot["build_device_ms"] is not None:
            tot["build_device_ms"] += tm["build_device_ms"]
            tot["device_build_equal"] &= tm["device_build_equal"]
    tot.update(item_size_bytes=per[0].item_size_bytes, kind=per[0].kind, k_only=bool(per[0].k_only), plan_stride_max=max(p.plan_stride for p in per))
    return tot


def _setup_nosi(cfg: Config, hs, side, cudart) -> Dict:
    """NOSI-layout payload: buffers, plans (the adapter), correctness hooks and side arms."""
    m = MODEL
    H, Dh, L, B, rows_ = m["n_kv"], m["head_dim"], cfg.layers, cfg.B, cfg.block_rows
    elem = 2
    fails, controls, skipped = 0, {}, []
    n_src_blocks, n_slots = cfg.s_cpu // rows_, cfg.window_rows // rows_
    pay = make_payload(L, B, H, n_src_blocks, n_slots, cfg.miss_per_head, cfg.heads, cfg.seed)
    ids = pay.pop("ids")
    pass_bytes = payload_bytes(pay["head_blocks"], rows_, Dh, elem)
    print("[repro] payload nosi: %.4f misses per (layer, head, request), %d head-blocks of %d rows, %.4f GB per pass (K+V)"
          % (pay["miss_per_head_mean"], pay["head_blocks"], rows_, pass_bytes / 1e9), flush=True)

    t0 = time.time()
    host_k = torch.empty((L, B, cfg.s_cpu, H, Dh), dtype=torch.bfloat16, pin_memory=True)
    host_v = torch.empty(host_k.shape, dtype=host_k.dtype, pin_memory=True)
    _fill_pinned(host_k, cfg.seed + 2)
    _fill_pinned(host_v, cfg.seed + 3)
    vmas = NM.parse()
    nb = host_k.numel() * elem
    numa = dict(buffers=dict(host_k=numa_range(host_k.data_ptr(), nb, vmas), host_v=numa_range(host_v.data_ptr(), nb, vmas)),
                base_mapping_only=dict(host_k=NM.pages_at(host_k.data_ptr(), vmas)), proc=proc_numa(),
                bind=os.environ.get("HR_NUMA_BIND", ""), gpu_numa_node=os.environ.get("HR_GPU_NUMA_NODE", ""), bytes_each=nb)
    print("[repro] host buffers %.2f GB each, pinned+filled in %.0fs; NUMA %s" % (nb / 1e9, time.time() - t0, json.dumps(numa)), flush=True)

    dst_k = torch.zeros((L, B, cfg.window_rows, H, Dh), dtype=torch.bfloat16, device="cuda")
    dst_v = torch.zeros_like(dst_k)
    l_i, h_i, b_i, m_i = (ids >= 0).nonzero(as_tuple=True)
    blk = ids[l_i, h_i, b_i, m_i].to(torch.int64)
    r = torch.arange(rows_)
    li, bi, hi = (x[:, None].expand(-1, rows_).reshape(-1) for x in (l_i, b_i, h_i))
    row_src = (blk[:, None] * rows_ + r).reshape(-1)
    row_dst = (m_i[:, None] * rows_ + r).reshape(-1)
    ref_k = host_k[li, bi, row_src, hi].cuda()
    ref_v = host_v[li, bi, row_src, hi].cuda()
    gidx = tuple(x.cuda() for x in (li, bi, row_dst, hi))
    ids_gpu = ids.cuda()
    w_dma = dma_width_bytes(pay["head_blocks_per_layer"], B, rows_, Dh, elem)
    w_el = w_dma // elem
    dma_ref_k = host_k.view(L, B, -1)[:, :, :w_el].cuda()
    dma_ref_v = host_v.view(L, B, -1)[:, :, :w_el].cuda()
    dma_bytes = 2 * L * B * w_dma
    print("[repro] dma2d: %d B per request row per layer tensor; %.4f GB per pass (payload %.4f GB)" % (w_dma, dma_bytes / 1e9, pass_bytes / 1e9), flush=True)

    def poison(kind):
        if kind == "dma2d":
            dst_k.view(L, B, -1)[:, :, :w_el].zero_(); dst_v.view(L, B, -1)[:, :, :w_el].zero_()
        else:
            dst_k[gidx] = 0; dst_v[gidx] = 0

    def check(kind) -> bool:
        if kind == "dma2d":
            return torch.equal(dst_k.view(L, B, -1)[:, :, :w_el], dma_ref_k) and torch.equal(dst_v.view(L, B, -1)[:, :, :w_el], dma_ref_v)
        return torch.equal(dst_k[gidx], ref_k) and torch.equal(dst_v[gidx], ref_v)

    plans, plan_meta = {}, {}
    for item in cfg.hs_items:
        try:
            per, tms = [], []
            for l in range(L):
                pl, tm = hs.timed_build(ids[l], device="cuda", s_cpu=cfg.s_cpu, s_dst=cfg.window_rows, n_heads=H, head_dim=Dh,
                                        elem_size=elem, block_rows=rows_, item=item)
                hs.validate_plan(pl)
                per.append(pl); tms.append(tm)
            tot = _plan_totals(per, tms)
            if sum(p.bytes_per_launch for p in per) != pass_bytes:
                raise AssertionError("plan bytes %d != payload %d" % (sum(p.bytes_per_launch for p in per), pass_bytes))
            fails += int(not tot["device_build_equal"])
            plans[item], plan_meta[item] = per, tot
        except hs.PlanRefused as e:
            plan_meta[item] = dict(refused=str(e))
        print("[repro] plan %s: %s" % (item, json.dumps(plan_meta[item])), flush=True)

    fhp = load_persistent_gather()
    unit_bytes = pass_bytes // L

    def side_hisparse(item, W, T, skip_io=False):
        per = plans[item]

        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        hs.copy_plan(per[l], host_k[l], host_v[l], dst_k[l], dst_v[l], W, T, skip_io)
                        SZ.LAUNCH_LOG.mark(per[l].bytes_per_launch)
            return f
        return mk

    def side_triton(W):
        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        fhp(dst_k[l], host_k[l], ids_gpu[l], rows_, n_ctas=W)
                        fhp(dst_v[l], host_v[l], ids_gpu[l], rows_, n_ctas=W)
                        SZ.LAUNCH_LOG.mark(unit_bytes)
            return f
        return mk

    def side_dma():
        spitch, dpitch = host_k[0].stride(0) * elem, dst_k[0].stride(0) * elem

        def mk(passes):
            def f():
                st = ctypes.c_void_p(side.cuda_stream)
                for _ in range(passes):
                    for l in range(L):
                        for src, dst in ((host_k[l], dst_k[l]), (host_v[l], dst_v[l])):
                            rc = cudart.cudaMemcpy2DAsync(ctypes.c_void_p(dst.data_ptr()), dpitch, ctypes.c_void_p(src.data_ptr()), spitch, w_dma, B, 1, st)
                            if rc != 0:
                                raise RuntimeError("cudaMemcpy2DAsync rc=%d" % rc)
                        SZ.LAUNCH_LOG.mark(2 * B * w_dma)
            return f
        return mk

    arms = []
    for item in cfg.hs_items:
        if item in plans:
            for T in cfg.hs_threads:
                for W in blocks_for(cfg, T):
                    arms.append(dict(arm="hisparse%d_%s_t%d" % (W, item, T), family="hisparse", W=W, threads=T, item=item, kind="scatter",
                                     mk=side_hisparse(item, W, T), bytes_pass=pass_bytes))
    if cfg.dma:
        arms.append(dict(arm="dma2d", family="dma2d", W=None, threads=None, item="prefix", kind="dma2d", mk=side_dma(), bytes_pass=dma_bytes))
    for W in cfg.triton_w:
        arms.append(dict(arm="triton%d" % W, family="triton", W=W, threads=128, item="tile", kind="scatter", mk=side_triton(W), bytes_pass=pass_bytes))
    ctrl_item = next((i for i in cfg.hs_items if i in plans), None)
    skip_io_mk = side_hisparse(ctrl_item, max(cfg.hs_blocks or (4,)), 1024, skip_io=True) if ctrl_item else None
    return dict(arms=arms, poison=poison, check=check, fails=fails, controls=controls, skipped=skipped, skip_io=(ctrl_item, skip_io_mk),
                meta=dict(payload=pay, pass_bytes=pass_bytes, dma_bytes=dma_bytes, dma_width_bytes=w_dma, numa=numa, plans=plan_meta))


def _setup_native(cfg: Config, hs, side, cudart) -> Dict:
    """HiSparse-native payload: linear 1152-B K-only records, no adapter (module docstring)."""
    L, B, item = cfg.layers, cfg.B, cfg.native_item
    E = item // 2
    fails, controls, skipped = 0, {}, []
    if item % 16:
        raise ValueError("HR_NATIVE_ITEM must be a multiple of 16, got %d" % item)
    pass_bytes = native_pass_bytes(L, B, cfg.native_misses, item)
    print("[repro] payload native: %d-byte K-only records, %d misses per (layer, request), B = %d, ctx %d, slots %d: %.4f GB per pass"
          % (item, cfg.native_misses, B, cfg.native_ctx, cfg.native_slots, pass_bytes / 1e9), flush=True)
    t0 = time.time()
    host = torch.empty((L, B * cfg.native_ctx, E), dtype=torch.bfloat16, pin_memory=True)
    _fill_pinned(host, cfg.seed + 2)
    vmas = NM.parse()
    nb = host.numel() * 2
    numa = dict(buffers=dict(host_pool=numa_range(host.data_ptr(), nb, vmas)), base_mapping_only=dict(host_pool=NM.pages_at(host.data_ptr(), vmas)),
                proc=proc_numa(), bind=os.environ.get("HR_NUMA_BIND", ""), gpu_numa_node=os.environ.get("HR_GPU_NUMA_NODE", ""), bytes_each=nb)
    print("[repro] native host pool %.2f GB pinned+filled in %.0fs; NUMA %s" % (nb / 1e9, time.time() - t0, json.dumps(numa)), flush=True)
    dst = torch.zeros((L, B * cfg.native_slots, E), dtype=torch.bfloat16, device="cuda")

    per, tms = [], []
    for l in range(L):
        pl, tm = hs.timed_native_build(device="cuda", B=B, misses=cfg.native_misses, ctx=cfg.native_ctx, slots=cfg.native_slots,
                                       item_size_bytes=item, seed=cfg.seed * 7919 + l)
        hs.validate_plan(pl)
        per.append(pl); tms.append(tm)
    tot = _plan_totals(per, tms)
    if sum(p.bytes_per_launch for p in per) != pass_bytes:
        raise AssertionError("native plan bytes %d != payload %d" % (sum(p.bytes_per_launch for p in per), pass_bytes))
    plan_meta = {"native": tot}
    print("[repro] plan native: %s" % json.dumps(tot), flush=True)
    gsrc, gdst = native_index(per, B, cfg.native_ctx, cfg.native_slots)
    ref = host.view(-1, E)[gsrc].cuda()
    gdst_gpu = gdst.cuda()
    w_dma = cfg.native_misses * item
    w_el = w_dma // 2
    dma_ref = host.view(L, B, -1)[:, :, :w_el].cuda()
    dma_bytes = L * B * w_dma
    flat = dst.view(-1, E)

    def poison(kind):
        if kind == "dma2d":
            dst.view(L, B, -1)[:, :, :w_el].zero_()
        else:
            flat[gdst_gpu] = 0

    def check(kind) -> bool:
        if kind == "dma2d":
            return torch.equal(dst.view(L, B, -1)[:, :, :w_el], dma_ref)
        return torch.equal(flat[gdst_gpu], ref)

    pv_h, pv_d = host.new_empty(0), dst.new_empty(0)          # V placeholders: IsMLA never reads them (nullptr, as upstream)

    def side_native(W, T, skip_io=False):
        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        hs.copy_plan(per[l], host[l], pv_h, dst[l], pv_d, W, T, skip_io)
                        SZ.LAUNCH_LOG.mark(per[l].bytes_per_launch)
            return f
        return mk

    def side_dma():
        spitch, dpitch = cfg.native_ctx * item, cfg.native_slots * item       # one request's host span / device span

        def mk(passes):
            def f():
                st = ctypes.c_void_p(side.cuda_stream)
                for _ in range(passes):
                    for l in range(L):
                        rc = cudart.cudaMemcpy2DAsync(ctypes.c_void_p(dst[l].data_ptr()), dpitch, ctypes.c_void_p(host[l].data_ptr()), spitch, w_dma, B, 1, st)
                        if rc != 0:
                            raise RuntimeError("cudaMemcpy2DAsync rc=%d" % rc)
                        SZ.LAUNCH_LOG.mark(B * w_dma)
            return f
        return mk

    arms = []
    for T in cfg.hs_threads:
        for W in blocks_for(cfg, T):
            arms.append(dict(arm="hisparse%d_native_t%d" % (W, T), family="hisparse", W=W, threads=T, item="native", kind="scatter",
                             mk=side_native(W, T), bytes_pass=pass_bytes))
    if cfg.dma:
        arms.append(dict(arm="dma2d", family="dma2d", W=None, threads=None, item="prefix", kind="dma2d", mk=side_dma(), bytes_pass=dma_bytes))
    if cfg.triton_w:
        skipped.append("triton%s: NOSI's per-head tile gather does not apply to HiSparse's linear %d-B K-only records" % (list(cfg.triton_w), item))
    if [i for i in cfg.hs_items if i != "native"]:
        skipped.append("items %s: the native payload has one item kind (native, %d B)" % ([i for i in cfg.hs_items if i != "native"], item))
    skip_io_mk = side_native(max(cfg.hs_blocks or (4,)), 1024, skip_io=True)
    return dict(arms=arms, poison=poison, check=check, fails=fails, controls=controls, skipped=skipped, skip_io=("native", skip_io_mk),
                meta=dict(payload=dict(native_item=item, native_misses=cfg.native_misses, native_ctx=cfg.native_ctx, native_slots=cfg.native_slots,
                                       k_only=True, order="draw order (HiSparse orders misses by top-k index, not by address)"),
                          pass_bytes=pass_bytes, dma_bytes=dma_bytes, dma_width_bytes=w_dma, numa=numa, plans=plan_meta))


def _row(a: Dict, mode: str, regime: str, passes: int, sizing, fails: int, alone, s_alone, conc) -> Dict:
    return dict(arm=a["arm"], mode=mode, regime=regime, family=a["family"], W=a["W"], threads=a["threads"], item=a["item"],
                footprint=footprint(a["family"], a["W"], a["threads"]), passes=passes, bytes=a["bytes_pass"] * passes, sizing=sizing, fails=fails,
                decode_alone_ms=[x["main_ms"] for x in alone], decode_conc_ms=[x["main_ms"] for x in conc],
                side_alone_ms=[x["side_ms"] - x["host_lag_ms"] for x in s_alone], side_conc_ms=[x["side_ms"] - x["host_lag_ms"] for x in conc],
                overlap_frac=[x["overlap_frac"] for x in conc], host_lag_ms=[x["host_lag_ms"] for x in alone + s_alone + conc],
                host_enqueue_ms=[x["host_enqueue_ms"] for x in conc], sleep_ms=[x["sleep_ms"] for x in conc],
                during_gbps=[x.get("during_gbps") for x in conc], inside_frac=[x.get("inside_frac") for x in conc],
                during_bytes=[x.get("during_bytes") for x in conc], last_end_ms=[x.get("last_end_ms") for x in conc],
                n_launches=[x.get("n_launches") for x in conc])


@torch.inference_mode()
def run(cfg: Config) -> Dict:
    t_start = time.time()
    fails, controls, rows = 0, {}, []
    hs = load_hisparse_copy()
    t0 = time.time()
    hs.load()
    print("[repro] hisparse_copy extension ready in %.0fs (%s)" % (time.time() - t0, hs.UPSTREAM_COMMIT), flush=True)
    cudart = _cudart() if cfg.dma else None
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    bracket = make_bracket(main, side, int(cfg.sleep_ms * 1e-3 * GPU_CLOCK_HZ))
    env = (_setup_native if cfg.payload == "native" else _setup_nosi)(cfg, hs, side, cudart)
    fails += env["fails"]
    controls.update(env["controls"])
    arms, poison, check = env["arms"], env["poison"], env["check"]
    if cfg.require_numa_node is not None:
        bufs = env["meta"]["numa"]["buffers"]
        ok = all(numa_local(v, cfg.require_numa_node) for v in bufs.values())
        controls["numa_local"] = dict(required_node=cfg.require_numa_node, ok=ok, checked="every mapping each buffer touches")
        fails += int(not ok)

    # ---- the step, modes, references
    t0 = time.time()
    proxy = Proxy(cfg)
    print("[repro] proxy: %.1f GB weights, attention %s, norm %s, built in %.0fs" % (proxy.weight_bytes() / 1e9, proxy.attn_backend, proxy.norm_backend, time.time() - t0), flush=True)
    refs = {}
    proxy.step(); torch.cuda.synchronize()
    ref_eager = proxy.logits.clone()
    if not bool(torch.isfinite(ref_eager).all()):
        controls["reference_finite"] = False
        fails += 1
    mains = {}
    for mode in cfg.modes:
        if mode == "graph":
            try:
                proxy.capture()
                proxy.graph.replay(); torch.cuda.synchronize()
                refs[mode] = proxy.logits.clone()
                mains[mode] = (proxy.graph.replay, False)
                controls["graph_vs_eager_logits_equal"] = bool(torch.equal(refs[mode], ref_eager))
            except Exception as e:  # noqa: BLE001
                controls["graph_capture_error"] = repr(e)[:300]
                fails += 1
        elif mode in ("eager", "eager_live"):
            refs[mode] = ref_eager
            mains[mode] = (proxy.step, mode == "eager_live")
        else:
            raise ValueError("unknown mode %r" % mode)
    print("[repro] modes %s ready; regimes %s; controls %s" % (list(mains), list(cfg.regimes), json.dumps(controls)), flush=True)

    def log_row(row):
        s = summarize([row])[0]
        print("[repro] %-26s %-10s %-10s passes=%d decode %.2f -> %.2f ms (%+.1f%%)  side %.1f -> %.1f GB/s (during %s)  inside %s  overlap %.2f  lag %.3f  %s%s"
              % (row["arm"], row["mode"], row["regime"], row["passes"], s["decode_alone_med"], s["decode_conc_med"], 100 * s["slowdown"], s["gbps_alone"],
                 s["gbps_conc"], _f(s["gbps_during"]), _f(s["inside_frac"], "%.2f"), s["min_overlap"], s["max_host_lag"], "valid" if s["valid"] else "INVALID",
                 "" if row["fails"] == 0 else " FAILS %d" % row["fails"]), flush=True)

    # ---- arms x modes x regimes
    for a in arms:
        try:
            poison(a["kind"])
            bracket(None, a["mk"](1))                                                   # first touch (JIT), untimed
            first_ok = check(a["kind"])
            fails += int(not first_ok)
        except Exception as e:  # noqa: BLE001
            for mode in mains:
                for regime in cfg.regimes:
                    rows.append(dict(arm=a["arm"], mode=mode, regime=regime, error=repr(e)[:300]))
            fails += 1
            print("[repro] %s FAILED at first touch: %r" % (a["arm"], e), flush=True)
            torch.cuda.synchronize()
            continue
        for mode, (fn_main, live) in mains.items():
            ref = refs[mode]
            try:
                n_common = 0
                alone = []
                for _ in range(cfg.reps):
                    _, rr = bracket(fn_main, None, live)
                    n_common += int(not torch.equal(proxy.logits, ref))
                    alone.append(rr)
                t_alone = _med([x["main_ms"] for x in alone])
                poison(a["kind"])
                _, r1 = bracket(None, a["mk"](1))
                n_common += int(not check(a["kind"]))
                fails += n_common

                def side_alone(passes):
                    out, nf = [], 0
                    for _ in range(cfg.reps):
                        poison(a["kind"])
                        _, rr = bracket(None, a["mk"](passes))
                        nf += int(not check(a["kind"]))
                        out.append(rr)
                    return out, nf

                def concurrent(passes, regime):
                    out, nf = [], 0
                    for i in range(cfg.reps):
                        poison(a["kind"])
                        if cfg.nvtx:
                            torch.cuda.nvtx.range_push("gf|%s|%s|%s|%d" % (a["arm"], mode, regime, i))
                        _, rr = bracket(fn_main, a["mk"](passes), live)
                        if cfg.nvtx:
                            torch.cuda.nvtx.range_pop()
                        lg_ok, tr_ok = torch.equal(proxy.logits, ref), check(a["kind"])
                        nf += int(not lg_ok) + int(not tr_ok)
                        rr.update(logits_equal=bool(lg_ok), transfer_equal=bool(tr_ok))
                        out.append(rr)
                    return out, nf

                sat = None
                if "saturating" in cfg.regimes:
                    n_trial = [0]

                    def trial(p):
                        poison(a["kind"])
                        _, rt = bracket(fn_main, a["mk"](p), live)
                        n_trial[0] += int(not torch.equal(proxy.logits, ref)) + int(not check(a["kind"]))
                        return rt["side_ms"] - rt["host_lag_ms"], rt["main_ms"]
                    sizing = SZ.size_side(trial, t_alone, r1["side_ms"] - r1["host_lag_ms"], max_trials=cfg.size_trials)
                    passes = sizing["passes"]
                    s_al, f1 = side_alone(passes)
                    conc, f2 = concurrent(passes, "saturating")
                    nf = n_common + n_trial[0] + f1 + f2
                    fails += n_trial[0] + f1 + f2
                    sat = _row(a, mode, "saturating", passes, sizing, nf, alone, s_al, conc)
                    rows.append(sat)
                    log_row(sat)
                if "burst" in cfg.regimes:
                    if sat is not None and sat["passes"] == 1:
                        row = dict(sat, regime="burst", sizing=None, same_as_saturating=True)
                    else:
                        s_al, f1 = side_alone(1)
                        conc, f2 = concurrent(1, "burst")
                        fails += f1 + f2
                        row = _row(a, mode, "burst", 1, None, (n_common if sat is None else 0) + f1 + f2, alone, s_al, conc)
                    rows.append(row)
                    log_row(row)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(arm=a["arm"], mode=mode, regime="saturating", error=repr(e)[:300]))
                fails += 1
                print("[repro] %s %s FAILED: %r" % (a["arm"], mode, e), flush=True)

    # ---- control: the checker must see missing bytes (SkipIO walks the plan and moves nothing)
    ctrl_item, ctrl_mk = env["skip_io"]
    if ctrl_mk is not None:
        try:
            poison("scatter")
            bracket(None, ctrl_mk(1))
            detected = not check("scatter")
            controls["skip_io_detected"] = dict(item=ctrl_item, detected=detected)
            fails += int(not detected)
        except Exception as e:  # noqa: BLE001
            controls["skip_io_error"] = repr(e)[:300]
            fails += 1

    meta = dict(config=asdict(cfg), torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                attn_backend=proxy.attn_backend, norm_backend=proxy.norm_backend, weight_bytes=proxy.weight_bytes(),
                controls=controls, skipped=env["skipped"], failures=fails, upstream=hs.UPSTREAM_COMMIT, wall_s=time.time() - t_start,
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                sizing_rule=dict(alone_margin=SZ.ALONE_MARGIN, conc_margin=SZ.CONC_MARGIN, overlap_min=SZ.OVERLAP_MIN, host_lag_max_ms=SZ.HOST_LAG_MAX_MS),
                **env["meta"])
    return dict(meta=meta, rows=rows, summary=summarize(rows), fails=fails)


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        return compare_dir(sys.argv[2])
    cfg = config_from_env()
    os.makedirs(cfg.out, exist_ok=True)
    print("[repro] config %s" % json.dumps(asdict(cfg)), flush=True)
    res = run(cfg)
    base = os.path.join(cfg.out, cfg.tag)
    json.dump(res, open(base + ".json", "w"), indent=1)
    text = render(res["summary"], res["meta"])
    open(base + ".md", "w").write(text)
    print(text, flush=True)
    print("[repro] saved %s.json / .md (failures %d)" % (base, res["fails"]), flush=True)
    return min(int(res["fails"]), 200)


if __name__ == "__main__":
    sys.exit(main())
