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
NOSI's). Modes (HR_MODES): graph = the whole step captured in ONE CUDA graph and replayed; eager = the same
Python step, launched while the gate's GPU sleep runs (the worker sweep's bracket: host launch time never
enters the bracket); eager_live = the host enqueues the step only after the gate fires, as a real eager server.

THE SIDE (one pass = one decode step's miss volume over all layers, K and V), on a side stream:
  hisparse<W>_<item>_t<T>  the vendored kernel, W blocks x T threads, one launch per layer (K and V), fed a plan
                           built by hisparse_copy/plan.py from the SAME ids (i256 per-head rows = exact general
                           adapter; i512 token rows = HiSparse's native token item at H = 2; i32k whole blocks)
  triton<W>                NOSI's throttled persistent Triton gather (flash_h2d_persistent.py), W CTAs x 128 thr
  dma2d                    cudaMemcpy2DAsync, per layer K and V: B rows of the SAME byte count (a contiguous prefix
                           of every request: a copy engine cannot take scattered blocks in one call)
THE PAYLOAD (seeded, HR_SEED): per (layer, [head,] request) HR_MISS_PER_HEAD (3.5 = NOSI's measured per-step
misses per (layer, head, request) at 63 slots) distinct host blocks drawn uniformly from the HR_S_CPU / rows
blocks of the context, into distinct destination slots of a HR_WINDOW_ROWS window (the tail slot excluded).
Every layer carries exactly the same number of misses (so dma2d's per-layer width is exact). HR_HEADS =
coupled (both KV heads miss the same blocks into the same slots: every item kind is exact) | independent
(per-head ids, as NOSI's engine; i512 / i32k are REFUSED and recorded). HiSparse's published payload: set
HR_B, HR_MISS_PER_HEAD, HR_BLOCK_ROWS (1 = token-granular), HR_S_CPU, HR_WINDOW_ROWS once it is stated.
HOST BUFFER: fresh pinned tensors (L, B, S_cpu, H, Dh) for K and V; /proc/self/numa_maps pages per node and the
process's Cpus_allowed / Mems_allowed are recorded; the sbatch binds numactl --cpunodebind=0 --membind=0 (the
GPU's node) and HR_REQUIRE_NUMA_NODE makes any page elsewhere a failure.

TIMER: the worker sweep's GPU-sleep-gated bracket (copied): main stream sleeps, the host enqueues the side behind
the gate, the step's start event follows the enqueue; host_lag = gate -> step start must be <= 0.1 ms.
SIZING (side_sizing.py): side >= 1.2 x step alone, then trial concurrent brackets until it lasts >= 1.15 x the
CONCURRENT step; VALID = every concurrent rep overlaps >= 0.95 of the step and host_lag <= 0.1 ms.
CORRECTNESS: before every transfer rep the destination rows are zeroed; after it they must equal the source
rows (torch.equal, all layers, K and V); every step's logits must be torch.equal to that mode's reference.
CONTROL: the kernel with SkipIO = true must FAIL the transfer check (the checker sees missing bytes).
Outputs HR_OUT/<HR_TAG>.json and .md; exit code = failures (correctness, crashes, controls, placement).
"""
import ctypes
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
import side_sizing as SZ  # noqa: E402

NOSI_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FCE = os.path.join(NOSI_ROOT, "nosi", "nosi", "flash_cache_engine")
CUDART = "/venv/nosa/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
# NOSA-8B, /mnt/beegfs/ojerbi/models/NOSA-8B/config.json
MODEL = dict(hidden=4096, n_heads=32, n_kv=2, head_dim=128, inter=16384, vocab=73448, layers=32, eps=1e-6)
GPU_CLOCK_HZ = 1.41e9          # the worker sweep's sleep calibration (A100 max SM clock)


def _ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split())


@dataclass
class Config:
    B: int = 64
    layers: int = 32
    attn_tokens: int = 4096
    s_cpu: int = 16384
    block_rows: int = 64
    window_rows: int = 4096
    miss_per_head: float = 3.5
    heads: str = "coupled"
    seed: int = 0
    hs_blocks: Tuple[int, ...] = (1, 2, 4, 8, 16)
    hs_threads: Tuple[int, ...] = (256, 1024)
    hs_items: Tuple[str, ...] = ("i256", "i512", "i32k")
    triton_w: Tuple[int, ...] = (1, 2, 4, 8, 16)
    dma: bool = True
    modes: Tuple[str, ...] = ("graph", "eager", "eager_live")
    reps: int = 5
    sleep_ms: float = 15.0
    size_trials: int = 3
    num_splits: int = 4
    require_numa_node: Optional[int] = None
    out: str = "."
    tag: str = "repro"


def config_from_env(env=None) -> Config:
    env = os.environ if env is None else env
    g = env.get
    rn = g("HR_REQUIRE_NUMA_NODE", "")
    return Config(
        B=int(g("HR_B", "64")), layers=int(g("HR_LAYERS", "32")), attn_tokens=int(g("HR_ATTN_TOKENS", "4096")),
        s_cpu=int(g("HR_S_CPU", "16384")), block_rows=int(g("HR_BLOCK_ROWS", "64")), window_rows=int(g("HR_WINDOW_ROWS", "4096")),
        miss_per_head=float(g("HR_MISS_PER_HEAD", "3.5")), heads=g("HR_HEADS", "coupled"), seed=int(g("HR_SEED", "0")),
        hs_blocks=_ints(g("HR_HS_BLOCKS", "1 2 4 8 16")), hs_threads=_ints(g("HR_HS_THREADS", "256 1024")),
        hs_items=tuple(g("HR_HS_ITEMS", "i256 i512 i32k").split()), triton_w=_ints(g("HR_TRITON_W", "1 2 4 8 16")),
        dma=g("HR_DMA", "1") == "1", modes=tuple(g("HR_MODES", "graph eager eager_live").split()),
        reps=int(g("HR_REPS", "5")), sleep_ms=float(g("HR_SLEEP_MS", "15")), size_trials=int(g("HR_SIZE_TRIALS", "3")),
        num_splits=int(g("HR_NUM_SPLITS", "4")), require_numa_node=(int(rn) if rn.strip() else None),
        out=g("HR_OUT", "."), tag=g("HR_TAG", "repro"))


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
    """pages per NUMA node of the mapping that contains `ptr`, from /proc/self/numa_maps (None if not found)."""
    best = None
    try:
        for line in open("/proc/self/numa_maps"):
            parts = line.split()
            start = int(parts[0], 16)
            if start <= ptr and (best is None or start > best[0]):
                best = (start, parts)
    except OSError as e:
        return dict(error=str(e))
    if best is None:
        return None
    nodes = {p.split("=")[0]: int(p.split("=")[1]) for p in best[1] if p.startswith("N") and "=" in p}
    return dict(mapping_start=hex(best[0]), policy=best[1][1], pages_per_node=nodes, flags=[p for p in best[1][2:] if not p.startswith("N")][:6])


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
    """Every page of the mapping on `node` (and at least one page counted)."""
    if not pages or "pages_per_node" not in pages:
        return False
    ppn = pages["pages_per_node"]
    return bool(ppn) and set(ppn) == {"N%d" % node}


# --------------------------------------------------------------------------- summary / table (pure)
def _med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def _p95(xs):
    return float(np.percentile(xs, 95)) if len(xs) else float("nan")


def summarize(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        if r.get("error"):
            out.append(dict(arm=r["arm"], mode=r["mode"], error=r["error"]))
            continue
        ra = [r["bytes"] / (x * 1e6) for x in r["side_alone_ms"]]
        rc = [r["bytes"] / (x * 1e6) for x in r["side_conc_ms"]]
        da, dc = r["decode_alone_ms"], r["decode_conc_ms"]
        out.append(dict(arm=r["arm"], mode=r["mode"], family=r["family"], W=r.get("W"), threads=r.get("threads"), item=r.get("item"),
                        passes=r["passes"], gbps_alone=_med(ra), gbps_conc=_med(rc), decode_alone_med=_med(da), decode_alone_p95=_p95(da),
                        decode_conc_med=_med(dc), decode_conc_p95=_p95(dc), slowdown=_med(dc) / _med(da) - 1 if da and dc else float("nan"),
                        min_overlap=min(r["overlap_frac"]) if r["overlap_frac"] else float("nan"),
                        max_host_lag=max(r["host_lag_ms"]) if r["host_lag_ms"] else float("nan"),
                        valid=SZ.valid(r["overlap_frac"], r["host_lag_ms"]), correct=r["fails"] == 0, fails=r["fails"]))
    return out


def render(summary: List[Dict], meta: Dict) -> str:
    cfg = meta.get("config", {})
    L = ["# HiSparse low-interference reproduction: decode-step proxy (NOSA-8B shapes, random weights) + side-stream transfer", "",
         "B=%s, %s layers, attention over %s resident tokens (%s, num_splits %s), payload %.3f GB per pass (%s misses per (layer, head, request), "
         "%s-row blocks, heads %s, seed %s); host buffer pages %s; process %s." % (
             cfg.get("B"), cfg.get("layers"), cfg.get("attn_tokens"), meta.get("attn_backend"), cfg.get("num_splits"),
             meta.get("pass_bytes", 0) / 1e9, cfg.get("miss_per_head"), cfg.get("block_rows"), cfg.get("heads"), cfg.get("seed"),
             json.dumps({k: (v or {}).get("pages_per_node") for k, v in (meta.get("numa") or {}).get("buffers", {}).items()}),
             json.dumps((meta.get("numa") or {}).get("proc"))), "",
         "Medians (p95) over reps. GB/s = bytes / side interval (per rep, then median). slowdown = median concurrent / median alone - 1. "
         "VALID = min overlap >= %.2f and max host lag <= %.1f ms. CORRECT = every transfer rep's destination equals the source and every "
         "step's logits equal the mode's reference." % (SZ.OVERLAP_MIN, SZ.HOST_LAG_MAX_MS), "",
         "| arm | mode | passes | transfer alone GB/s | transfer concurrent GB/s | decode alone median / p95 ms | decode concurrent median / p95 ms "
         "| slowdown | min overlap | max host lag ms | valid | correct |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if "error" in s:
            L.append("| %s | %s | - | - | - | - | - | - | - | - | - | ERROR: %s |" % (s["arm"], s["mode"], str(s["error"])[:120].replace("|", "/")))
            continue
        L.append("| %s | %s | %d | %.1f | %.1f | %.2f / %.2f | %.2f / %.2f | %+.1f%% | %.2f | %.3f | %s | %s |" % (
            s["arm"], s["mode"], s["passes"], s["gbps_alone"], s["gbps_conc"], s["decode_alone_med"], s["decode_alone_p95"],
            s["decode_conc_med"], s["decode_conc_p95"], 100 * s["slowdown"], s["min_overlap"], s["max_host_lag"],
            "yes" if s["valid"] else "INVALID", "yes" if s["correct"] else "NO (%d)" % s["fails"]))
    modes = [m for m in cfg.get("modes", []) if any(s.get("mode") == m for s in summary)]
    arms = []
    for s in summary:
        if s["arm"] not in arms:
            arms.append(s["arm"])
    L += ["", "## Decode modes side by side: slowdown (concurrent GB/s)", "",
          "| arm | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    by = {(s["arm"], s["mode"]): s for s in summary}
    for a in arms:
        cells = []
        for m in modes:
            s = by.get((a, m))
            if s is None or "error" in s:
                cells.append("-" if s is None else "ERROR")
            else:
                cells.append("%+.1f%% (%.1f)%s" % (100 * s["slowdown"], s["gbps_conc"], "" if s["valid"] else " INVALID"))
        L.append("| %s | %s |" % (a, " | ".join(cells)))
    L += ["", "## Adapter cost (plan build per pass: all layers)", "",
          "| item | items per pass | item bytes | host build ms | upload ms | device build ms | device == host |", "|---|---|---|---|---|---|---|"]
    for it, t in (meta.get("plans") or {}).items():
        if "refused" in t:
            L.append("| %s | REFUSED: %s | | | | | |" % (it, t["refused"][:160]))
        else:
            L.append("| %s | %d | %d | %.2f | %.2f | %.2f | %s |" % (it, t["n_items"], t["item_size_bytes"], t["build_host_ms"], t["upload_ms"],
                                                                 t["build_device_ms"], t["device_build_equal"]))
    L += ["", "## Controls and failures", ""]
    for k, v in (meta.get("controls") or {}).items():
        L.append("- %s: %s" % (k, json.dumps(v)))
    L.append("- failures: %d" % meta.get("failures", -1))
    return "\n".join(L) + "\n"


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
    """The worker sweep's GPU-sleep-gated bracket (worker_sweep.py bracket(), copied) plus live_main: GPU sleep on
    main -> gate -> side enqueued behind the gate -> [live_main: host waits for the gate] -> t0 -> main work."""
    def bracket(fn_main, fn_side, live_main=False):
        torch.cuda.synchronize()
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
        return out, r
    return bracket


@torch.inference_mode()
def run(cfg: Config) -> Dict:
    t_start = time.time()
    fails, controls, rows = 0, {}, []
    m = MODEL
    H, Dh, L, B, rows_ = m["n_kv"], m["head_dim"], cfg.layers, cfg.B, cfg.block_rows
    elem = 2
    n_src_blocks, n_slots = cfg.s_cpu // rows_, cfg.window_rows // rows_
    pay = make_payload(L, B, H, n_src_blocks, n_slots, cfg.miss_per_head, cfg.heads, cfg.seed)
    ids = pay.pop("ids")
    pass_bytes = payload_bytes(pay["head_blocks"], rows_, Dh, elem)
    print("[repro] payload: %.4f misses per (layer, head, request), %d head-blocks, %.3f GB per pass (K+V)"
          % (pay["miss_per_head_mean"], pay["head_blocks"], pass_bytes / 1e9), flush=True)

    # ---- host buffer: fresh pinned K and V, placement recorded
    t0 = time.time()
    host_k = torch.empty((L, B, cfg.s_cpu, H, Dh), dtype=torch.bfloat16, pin_memory=True)
    host_v = torch.empty(host_k.shape, dtype=host_k.dtype, pin_memory=True)
    gfill = torch.Generator(device="cuda").manual_seed(cfg.seed + 2)
    for l in range(L):
        for t in (host_k, host_v):
            t[l].copy_(torch.randn(t.shape[1:], generator=gfill, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    numa = dict(buffers=dict(host_k=numa_pages(host_k.data_ptr()), host_v=numa_pages(host_v.data_ptr())), proc=proc_numa(),
                bind=os.environ.get("HR_NUMA_BIND", ""), gpu_numa_node=os.environ.get("HR_GPU_NUMA_NODE", ""),
                bytes_each=host_k.numel() * elem)
    print("[repro] host buffers %.1f GB each, pinned+filled in %.0fs; NUMA %s" % (host_k.numel() * elem / 1e9, time.time() - t0, json.dumps(numa)), flush=True)
    if cfg.require_numa_node is not None:
        ok = all(numa_local(numa["buffers"][k], cfg.require_numa_node) for k in ("host_k", "host_v"))
        controls["numa_local"] = dict(required_node=cfg.require_numa_node, ok=ok)
        fails += int(not ok)

    # ---- destination windows, index tensors, reference rows
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
    w_dma = dma_width_bytes(pay["head_blocks_per_layer"], B, rows_, Dh, elem)          # bytes of one request's contiguous prefix
    w_el = w_dma // elem
    dma_ref_k = host_k.view(L, B, -1)[:, :, :w_el].cuda()
    dma_ref_v = host_v.view(L, B, -1)[:, :, :w_el].cuda()
    dma_bytes = 2 * L * B * w_dma
    print("[repro] dma2d: %d B per request row per layer tensor; %.3f GB per pass (payload %.3f GB)" % (w_dma, dma_bytes / 1e9, pass_bytes / 1e9), flush=True)

    def poison(kind):
        if kind == "dma2d":
            dst_k.view(L, B, -1)[:, :, :w_el].zero_(); dst_v.view(L, B, -1)[:, :, :w_el].zero_()
        else:
            dst_k[gidx] = 0; dst_v[gidx] = 0

    def check(kind) -> bool:
        if kind == "dma2d":
            return torch.equal(dst_k.view(L, B, -1)[:, :, :w_el], dma_ref_k) and torch.equal(dst_v.view(L, B, -1)[:, :, :w_el], dma_ref_v)
        return torch.equal(dst_k[gidx], ref_k) and torch.equal(dst_v[gidx], ref_v)

    # ---- plans (the adapter), timed, outside every bracket
    hs = load_hisparse_copy()
    t0 = time.time()
    hs.load()
    print("[repro] hisparse_copy extension ready in %.0fs (%s)" % (time.time() - t0, hs.UPSTREAM_COMMIT), flush=True)
    plans, plan_meta = {}, {}
    for item in cfg.hs_items:
        try:
            per, tot = [], dict(build_host_ms=0.0, upload_ms=0.0, build_device_ms=0.0, n_items=0, device_build_equal=True)
            for l in range(L):
                pl, tm = hs.timed_build(ids[l], device="cuda", s_cpu=cfg.s_cpu, s_dst=cfg.window_rows, n_heads=H, head_dim=Dh,
                                        elem_size=elem, block_rows=rows_, item=item)
                hs.validate_plan(pl)
                per.append(pl)
                for k in ("build_host_ms", "upload_ms", "build_device_ms", "n_items"):
                    tot[k] += tm[k]
                tot["device_build_equal"] &= tm["device_build_equal"]
            tot.update(item_size_bytes=per[0].item_size_bytes, kind=per[0].kind, plan_stride_max=max(p.plan_stride for p in per))
            if sum(p.bytes_per_launch for p in per) != pass_bytes:
                raise AssertionError("plan bytes %d != payload %d" % (sum(p.bytes_per_launch for p in per), pass_bytes))
            fails += int(not tot["device_build_equal"])
            plans[item], plan_meta[item] = per, tot
        except hs.PlanRefused as e:
            plan_meta[item] = dict(refused=str(e))
        print("[repro] plan %s: %s" % (item, json.dumps(plan_meta[item])), flush=True)

    # ---- sides
    fhp = load_persistent_gather()
    cudart = None
    if cfg.dma:
        cudart = ctypes.CDLL(CUDART)
        cudart.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        cudart.cudaMemcpy2DAsync.restype = ctypes.c_int
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    bracket = make_bracket(main, side, int(cfg.sleep_ms * 1e-3 * GPU_CLOCK_HZ))

    def side_hisparse(item, W, T, skip_io=False):
        per = plans[item]

        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        hs.copy_plan(per[l], host_k[l], host_v[l], dst_k[l], dst_v[l], W, T, skip_io)
            return f
        return mk

    def side_triton(W):
        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        fhp(dst_k[l], host_k[l], ids_gpu[l], rows_, n_ctas=W)
                        fhp(dst_v[l], host_v[l], ids_gpu[l], rows_, n_ctas=W)
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
            return f
        return mk

    arms = []
    for item in cfg.hs_items:
        if item in plans:
            for T in cfg.hs_threads:
                for W in cfg.hs_blocks:
                    arms.append(dict(arm="hisparse%d_%s_t%d" % (W, item, T), family="hisparse", W=W, threads=T, item=item, kind="scatter",
                                     mk=side_hisparse(item, W, T), bytes_pass=pass_bytes))
    for W in cfg.triton_w:
        arms.append(dict(arm="triton%d" % W, family="triton", W=W, threads=128, item="tile", kind="scatter", mk=side_triton(W), bytes_pass=pass_bytes))
    if cfg.dma:
        arms.append(dict(arm="dma2d", family="dma2d", W=None, threads=None, item="prefix", kind="dma2d", mk=side_dma(), bytes_pass=dma_bytes))

    # ---- the step, both modes, references
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
    print("[repro] modes %s ready; controls %s" % (list(mains), json.dumps(controls)), flush=True)

    # ---- arms x modes
    for a in arms:
        try:
            poison(a["kind"])
            bracket(None, a["mk"](1))                                                   # first touch (JIT), untimed
            first_ok = check(a["kind"])
            fails += int(not first_ok)
        except Exception as e:  # noqa: BLE001
            for mode in mains:
                rows.append(dict(arm=a["arm"], mode=mode, error=repr(e)[:300]))
            fails += 1
            print("[repro] %s FAILED at first touch: %r" % (a["arm"], e), flush=True)
            torch.cuda.synchronize()
            continue
        for mode, (fn_main, live) in mains.items():
            ref = refs[mode]
            try:
                n_fail = 0
                alone = []
                for _ in range(cfg.reps):
                    _, rr = bracket(fn_main, None, live)
                    n_fail += int(not torch.equal(proxy.logits, ref))
                    alone.append(rr)
                t_alone = _med([x["main_ms"] for x in alone])
                poison(a["kind"])
                _, r1 = bracket(None, a["mk"](1))
                n_fail += int(not check(a["kind"]))

                def trial(p, a=a, fn_main=fn_main, live=live, ref=ref):
                    nonlocal n_fail
                    poison(a["kind"])
                    _, rt = bracket(fn_main, a["mk"](p), live)
                    n_fail += int(not torch.equal(proxy.logits, ref)) + int(not check(a["kind"]))
                    return rt["side_ms"] - rt["host_lag_ms"], rt["main_ms"]
                sizing = SZ.size_side(trial, t_alone, r1["side_ms"] - r1["host_lag_ms"], max_trials=cfg.size_trials)
                passes = sizing["passes"]
                s_alone, conc = [], []
                for _ in range(cfg.reps):
                    poison(a["kind"])
                    _, rr = bracket(None, a["mk"](passes))
                    n_fail += int(not check(a["kind"]))
                    s_alone.append(rr)
                for _ in range(cfg.reps):
                    poison(a["kind"])
                    _, rr = bracket(fn_main, a["mk"](passes), live)
                    lg_ok, tr_ok = torch.equal(proxy.logits, ref), check(a["kind"])
                    n_fail += int(not lg_ok) + int(not tr_ok)
                    rr.update(logits_equal=bool(lg_ok), transfer_equal=bool(tr_ok))
                    conc.append(rr)
                row = dict(arm=a["arm"], mode=mode, family=a["family"], W=a["W"], threads=a["threads"], item=a["item"], passes=passes,
                           bytes=a["bytes_pass"] * passes, sizing=sizing, fails=n_fail,
                           decode_alone_ms=[x["main_ms"] for x in alone], decode_conc_ms=[x["main_ms"] for x in conc],
                           side_alone_ms=[x["side_ms"] - x["host_lag_ms"] for x in s_alone], side_conc_ms=[x["side_ms"] - x["host_lag_ms"] for x in conc],
                           overlap_frac=[x["overlap_frac"] for x in conc], host_lag_ms=[x["host_lag_ms"] for x in alone + s_alone + conc],
                           host_enqueue_ms=[x["host_enqueue_ms"] for x in conc], sleep_ms=[x["sleep_ms"] for x in conc])
                rows.append(row)
                fails += n_fail
                s = summarize([row])[0]
                print("[repro] %-22s %-10s passes=%d decode %.2f -> %.2f ms (%+.1f%%)  side %.1f -> %.1f GB/s  overlap %.2f  lag %.3f  %s%s"
                      % (a["arm"], mode, passes, s["decode_alone_med"], s["decode_conc_med"], 100 * s["slowdown"], s["gbps_alone"], s["gbps_conc"],
                         s["min_overlap"], s["max_host_lag"], "valid" if s["valid"] else "INVALID", "" if n_fail == 0 else " FAILS %d" % n_fail), flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(arm=a["arm"], mode=mode, error=repr(e)[:300]))
                fails += 1
                print("[repro] %s %s FAILED: %r" % (a["arm"], mode, e), flush=True)

    # ---- control: the checker must see missing bytes (SkipIO walks the plan and moves nothing)
    ctrl_item = next((i for i in cfg.hs_items if i in plans), None)
    if ctrl_item is not None:
        try:
            poison("scatter")
            bracket(None, side_hisparse(ctrl_item, max(cfg.hs_blocks or (4,)), 1024, skip_io=True)(1))
            detected = not check("scatter")
            controls["skip_io_detected"] = dict(item=ctrl_item, detected=detected)
            fails += int(not detected)
        except Exception as e:  # noqa: BLE001
            controls["skip_io_error"] = repr(e)[:300]
            fails += 1

    meta = dict(config=asdict(cfg), torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                attn_backend=proxy.attn_backend, norm_backend=proxy.norm_backend, weight_bytes=proxy.weight_bytes(),
                payload=pay, pass_bytes=pass_bytes, dma_bytes=dma_bytes, dma_width_bytes=w_dma, numa=numa, plans=plan_meta,
                controls=controls, failures=fails, upstream=hs.UPSTREAM_COMMIT, wall_s=time.time() - t_start,
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                sizing_rule=dict(alone_margin=SZ.ALONE_MARGIN, conc_margin=SZ.CONC_MARGIN, overlap_min=SZ.OVERLAP_MIN, host_lag_max_ms=SZ.HOST_LAG_MAX_MS))
    return dict(meta=meta, rows=rows, summary=summarize(rows), fails=fails)


def main() -> int:
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
