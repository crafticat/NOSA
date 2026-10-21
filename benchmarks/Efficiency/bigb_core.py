"""[bigb] The tick at a LARGE BATCH (B 128 .. 320) at 16K on one A100 80GB: the token-log size, the GPU memory model of
the SG5 harness per B for both store layouts, the int32 extents, the cell plan and the registered time bands
(docs/R2_BIGB_PLAN.md). Pure Python, no torch: the CPU tests (tests_cpu/test_r2_bigb.py) compare every byte term with
the code's own tensors ("meta" device) and the fit with the measured peaks of p5b2 and p4sc64.

WHAT LIMITED B (static reads at feaa7d6, docs/R2_BIGB_PLAN.md section 1):
  1. the DEVICE TOKEN LOG: TickEngine.LOG = 1 << 16 slots per run, never reused (r2_tick.py run_rows: "token log
     full"). A round takes about 2 to 2K + 4 rows per committed token and request, so B x N must stay below about
     32k / 6.5k. Lifted by D's 7c0f998 (TickEngine(log_slots=...), cherry-picked) and the harness's --log-slots
     (-1 = log_slots_for below).
  2. the HARNESS's PREFLIGHT kept TWO ENGINES alive: r2_tick.run_round returns a TickRound that holds the engine's
     bound methods (read_tokens, on_publish, on_rollback), and r2_tick_sg5.preflight deleted eng / svc / tp but not
     `rnd`, so the first preflight engine (store, twins, service) lived while the second one was built. The measured
     peaks of p5b2 (67.49 GB) and p4sc64 (67.41 GB) are explained by exactly two engines (fit below; the residual is
     0.13 GB). One engine: 42.7 GB at the same point. Fixed in r2_tick_sg5 (rnd released before the next engine).
  3. the shipped child's pinned host rows (S + 8192 rows per request, rounded UP to a power of two by the CUDA caching
     host allocator) and the step's --mem=120G: B 128 already needs about 145 GiB of host memory there. Lifted by the
     harness's --skip-shipped (opt-in) and the bigb cells' --mem.
  4. not limits up to B 320 (checked): int32 extents (int32_bounds), the gilfree ring (chunks of 256 groups), B2's
     reader window (ids per layer read, not per request), PaddedLinear (m_fix = (1 + K) B, the tick's row bound).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# ------------------------------------------------------------------ NOSA-8B and the R2 path (config.json; code refs)
L, H, D = 32, 2, 128                  # layers, KV heads, head dim
HQ = 32                               # query heads
HIDDEN, INTER, VOCAB = 4096, 16384, 73448
BLOCK, STRIDE, TOPK = 64, 16, 64      # r2_positional_twin.py: BLOCK, STRIDE, TOPK
N_TAIL = 4                            # r2_tick.py: 2 exact + 2 draft tail pages per stream
TWIN_RING = 64                        # r2_positional_twin.RING
ELEM = 2                              # bf16
INT32_MAX = 2 ** 31 - 1
SLOT_BYTES = L * BLOCK * H * (2 * D + 1) * ELEM          # K + V + bias of one slot, every layer: 2,105,344
WEIGHTS_BYTES = 16370581512                              # model.safetensors (r2_tick_sg5_step.sh MODEL_SIZE pin)
B2_KEYS = 7                                              # r2_own2.OwnResidency(keys=7): key slots per request
B2_SLOT_COL_BYTES = 70                                   # r2_own2_ops.SLOT_COLS: 6 x 1 + 6 x 4 + 5 x 8 bytes per slot
SC_SLOT_BYTES = 6 * 8 + 8                                # DeviceMirror (6 int64) + GenBoard (1 int64) per slot
BUILD_TABLES_REQ = L * (3 * H * TOPK * 8 + 4)            # r2_decode_body.allocate: 3 block maps int64 + cache_lens
GIB = 2 ** 30


# ------------------------------------------------------------------ the run's widths (r2_tick_sg5.main, r2_decode_path)
def horizon(steps: int, sweep_steps: int = 24, LAMBDA: int = 7) -> int:
    """build_ragged's horizon in r2_tick_sg5.main: max(steps, sweep_steps) + LAMBDA + 64."""
    return max(int(steps), int(sweep_steps)) + int(LAMBDA) + 64


def widths(S: int, steps: int, sweep_steps: int = 24, LAMBDA: int = 7) -> Dict[str, int]:
    """r2_decode_path.ragged_widths at the harness's horizon, plus the service's nblk = n_host // 64."""
    end = int(S) + horizon(steps, sweep_steps, LAMBDA)
    lk_max = end // STRIDE + 1
    return dict(end=end, lk_max=lk_max, n_max=end + BLOCK, n_host=end + BLOCK, M_alloc=end // BLOCK + 1,
                nblk=(end + BLOCK) // BLOCK)


# ------------------------------------------------------------------ the token log
def log_factor(K: int) -> int:
    """Token-log slots per committed token and request (D's ctxsweep_core.log_factor, aef58e3): the worst case is
    every cohort rejected at its first position, 2K rows for one committed token; max(8, 2K + 4) covers it."""
    return max(8, 2 * int(K) + 4)


def log_slots_for(B: int, N: int, K: int = 3, LAMBDA: int = 7) -> int:
    """The token log's size for a round of B requests and N committed tokens each: log_factor(K) x B x (N + K +
    LAMBDA). The engine's refusal 'token log full' stays the guard."""
    return log_factor(K) * int(B) * (int(N) + int(K) + int(LAMBDA))


def min_round_rows(B: int, N: int) -> int:
    """A lower bound of a round's rows: every committed token after the first takes a draft row and a verify row."""
    return 2 * int(B) * (int(N) - 1)


# ------------------------------------------------------------------ byte terms (each one is checked by a CPU test)
def twin_bank_bytes(lk_max: int, n_max: int) -> int:
    """One bank (one request) of the 32 layers' PositionalTwin (r2_positional_twin.py __init__, bf16 tables): comp_k
    lk x H x D, ring 64 x H x D, total_cis n_max x H, ccis H x lk (2 B each); pos / n_comp / n_ccis / epoch int64,
    prefilled bool, cu int32: 37 B. Per twin object also _win (256 B), the last cu entry (4 B), overflow (1 B)."""
    per_layer = (int(lk_max) * H * D * ELEM + TWIN_RING * H * D * ELEM + int(n_max) * H * ELEM
                 + H * int(lk_max) * ELEM + 4 * 8 + 1 + 4)
    return L * per_layer


TWIN_OBJECT_BYTES = L * (256 + 4 + 1)


def store_bytes(layout: str, B: int, C: int, X: int, X_pool: Optional[int] = None) -> int:
    """The tick's store, every layer (K, V, bias): "stream" = r2_xpool.per_stream_bytes_formula(n_hist = C + X) (D4:
    B (C + X + 4) slots); "pool" = r2_xpool.store_bytes_formula (B (C + 4) + X_pool positions)."""
    if layout == "stream":
        return int(B) * (int(C) + int(X) + N_TAIL) * SLOT_BYTES
    if layout == "pool":
        xp = int(X if X_pool is None else X_pool)
        return (int(B) * (int(C) + N_TAIL) + xp) * SLOT_BYTES
    raise ValueError("layout stream | pool, not %r" % (layout,))


def b2_table_bytes(B: int, C: int, X: int, nblk: int, keys: int = B2_KEYS) -> int:
    """r2_own2_ops.Table (B2's state) on the device: SLOT_COLS over (L, B, H, C + X), pin_w (2 keys, L, B, H, WB)
    int64, owner, key_rd, rel_q, kcancel, hist_from, hist_id (B, nblk) int64, xfer (B, L, H, nblk) int32."""
    npos, wb = int(C) + int(X), (int(nblk) + 31) // 32
    B = int(B)
    return (B2_SLOT_COL_BYTES * L * B * H * npos + 2 * keys * L * B * H * wb * 8 + 4 * B + L * B * keys * 8
            + L * B * keys + B * keys + 8 * B + B * int(nblk) * 8 + B * L * H * int(nblk) * 4)


def sc_device_bytes(B: int, C: int, X: int, nblk: int) -> int:
    """S-C's device state that b2 inherits (r2_residency_sc.ResidencySC.__init__: DeviceMirror 6 x int64 and GenBoard
    1 x int64 per (L, B, H, C + X) slot; r2_residency_sparse.ResidencySparse.__init__: pins (L, B, H, nblk + 1, 2)
    int64)."""
    return SC_SLOT_BYTES * L * int(B) * H * (int(C) + int(X)) + L * int(B) * H * (int(nblk) + 1) * 2 * 8


def xpool_map_bytes(B: int, C: int, X: int, X_pool: int) -> int:
    """r2_xpool.XPool (mode pool) on the device: pos_of (L, B, H, C + X + 4) int64, occ (L, H, B C + X_pool) bool, err
    int64, peak (L, H) int64."""
    return L * int(B) * H * (int(C) + int(X) + N_TAIL) * 8 + L * H * (int(B) * int(C) + int(X_pool)) + 8 + L * H * 8


def workspace_bytes(B: int, K: int = 3) -> int:
    """The tick's largest per-row transient in EXACT mode (m_fix = (1 + K) B rows): lm_head's padded output (m_fix, V)
    bf16 + its float copy of up to m_fix rows + PaddedLinear's persistent input buffers (m_fix, 4096) and (m_fix,
    16384) bf16 (r2_spec_core.PaddedLinear; r2_tick.run_rows)."""
    m = (1 + int(K)) * int(B)
    return m * (VOCAB * (2 + 4) + (HIDDEN + INTER) * ELEM)


# ------------------------------------------------------------------ the measured anchors
P5B2 = dict(job="2190876 p5b2 (c675779)", record="/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p5_2190876_p5b2/summary.json",
            B=64, C=96, X=64, S=16128, steps=48, sweep_steps=24, K=3, layout="stream", backend="b2",
            build_gb=17.73, peak_gb=67.49, preflight_engines=2)
P4SC64 = dict(job="2190876 p4sc64 (3ae7d3b)", B=64, C=96, X=64, S=16128, steps=48, sweep_steps=24, K=3,
              layout="stream", backend="sc", peak_gb=67.41, preflight_engines=2)
BUILD_TRANSIENT_GB = 18.83 - 16.37   # eqmem_core.MemFit A16 minus the weights: one 16K request's prefill transient


def engine_bytes(layout: str, B: int, C: int, X: int, X_pool: Optional[int], w: Dict[str, int], backend: str = "b2",
                 log_slots: int = 1 << 16) -> Dict[str, int]:
    """The bytes ONE engine of the harness holds (TickEngine.from_r2 + its service): the 2B-bank twins, the store, the
    token log, S-C's device state, B2's table (b2), the xpool map (pool)."""
    out = dict(twins_2B=2 * int(B) * twin_bank_bytes(w["lk_max"], w["n_max"]) + TWIN_OBJECT_BYTES,
               store=store_bytes(layout, B, C, X, X_pool), tok_log=8 * int(log_slots),
               sc=sc_device_bytes(B, C, X, w["nblk"]))
    if backend == "b2":
        out["b2_table"] = b2_table_bytes(B, C, X, w["nblk"])
    if layout == "pool":
        out["xpool_map"] = xpool_map_bytes(B, C, X, int(X if X_pool is None else X_pool))
    return out


def build_resident_req(w: Dict[str, int]) -> int:
    """Per request, what the build leaves on the device once its windows are freed: one twin bank (the build twins
    the engines copy from) and the cache engines' block maps."""
    return twin_bank_bytes(w["lk_max"], w["n_max"]) + BUILD_TABLES_REQ


class MemModel:
    """The GPU memory model of the SG5 harness (PyTorch max_memory_allocated, GB = 1e9 B) per B:
      build_resident(B) = F0 + B x build_resident_req           (memory_allocated after the build: meta.build_gb)
      peak(B)           = build_resident(B) + engines x engine(B) + workspace(B) + R_FIX
    engines = 2 in the harness BEFORE the preflight fix (p5b2, p4sc64), 1 after it. F0 (the weights, the rope cache,
    the allocator's fixed part) is fitted to p5b2's build_gb; R_FIX (cuBLAS / Triton workspaces, the gilfree landing
    ring, the transients this model does not list) to p5b2's peak. Every other term comes from the code."""

    def __init__(self, anchor: Dict = P5B2):
        self.anchor = dict(anchor)
        a = self.anchor
        w = widths(a["S"], a["steps"], a["sweep_steps"])
        self.F0 = int(round(a["build_gb"] * 1e9)) - a["B"] * build_resident_req(w)
        eng = sum(engine_bytes(a["layout"], a["B"], a["C"], a["X"], None, w, a["backend"]).values())
        self.R_FIX = (int(round(a["peak_gb"] * 1e9)) - self.F0 - a["B"] * build_resident_req(w)
                      - a["preflight_engines"] * eng - workspace_bytes(a["B"], a["K"]))

    def terms(self, layout: str, B: int, C: int, X: int, X_pool: Optional[int] = None, S: int = 16128,
              steps: int = 24, sweep_steps: int = 24, K: int = 3, backend: str = "b2", engines: int = 1,
              log_slots: Optional[int] = None) -> Dict[str, int]:
        w = widths(S, steps, sweep_steps)
        ls = log_slots_for(B, max(steps, sweep_steps), K) if log_slots is None else int(log_slots)
        eng = engine_bytes(layout, B, C, X, X_pool, w, backend, ls)
        out = dict(F0=self.F0, build_req=int(B) * build_resident_req(w), workspace=workspace_bytes(B, K),
                   R_FIX=self.R_FIX)
        for k, v in eng.items():
            out["engine_" + k] = engines * v
        return out

    def peak_gb(self, *args, **kw) -> float:
        return sum(self.terms(*args, **kw).values()) / 1e9

    def build_peak_gb(self, B: int, S: int = 16128, steps: int = 24, sweep_steps: int = 24) -> float:
        """The BUILD phase (before the windows are freed): build_resident + the R2 body's 64-slot windows per request
        + one request's prefill transient (BUILD_TRANSIENT_GB, the 4.1 fit's A16 minus the weights)."""
        w = widths(S, steps, sweep_steps)
        return (self.F0 + int(B) * (build_resident_req(w) + TOPK * SLOT_BYTES)) / 1e9 + BUILD_TRANSIENT_GB

    def b_max(self, budget_gb: float, layout: str, C: int, X: int, X_pool: Optional[int] = None, hi: int = 4096,
              **kw) -> int:
        """The largest B whose peak (this model) is <= budget_gb (and whose build peak is too)."""
        best = 0
        for B in range(1, hi + 1):
            if self.peak_gb(layout, B, C, X, X_pool, **kw) <= budget_gb and self.build_peak_gb(B) <= budget_gb:
                best = B
            else:
                break
        return best


# ------------------------------------------------------------------ int32 extents
def int32_bounds(S: int = 16128, steps: int = 24, C: int = 96, X: int = 64, X_pool: int = 64,
                 K: int = 3) -> Dict[str, int]:
    """The largest B for each int32 quantity of the tick path (an index the code holds in int32, or an element
    offset a kernel may form; piece 3, flash_h2d_mask, h2d_entries and the packer widen to int64, so the last two are
    margins, not limits):
      cu_k          r2_tick.run_rows: bank x lk_max in int32, bank < 2B
      pool_pos      r2_xpool.translate: absolute positions in int32, < M x 64
      stream_store  one per-stream store tensor's elements, B (C + X + 4) 64 H D
      pool_store    one pool store tensor's elements, (B (C + 4) + X_pool) 64 H D
      host_rows     one pinned host-row tensor's elements, B n_host H D
      tok_log       token-log slot ids (int64 tensor; a host int): log_slots_for(B, N) < 2^31"""
    w = widths(S, steps)
    N = max(int(steps), 24)
    return dict(cu_k=INT32_MAX // (2 * w["lk_max"]),
                pool_pos=(INT32_MAX // BLOCK - int(X_pool)) // (int(C) + N_TAIL),
                stream_store=INT32_MAX // ((int(C) + int(X) + N_TAIL) * BLOCK * H * D),
                pool_store=(INT32_MAX // (BLOCK * H * D) - int(X_pool)) // (int(C) + N_TAIL),
                host_rows=INT32_MAX // (w["n_host"] * H * D),
                tok_log=INT32_MAX // (log_factor(K) * (N + K + 7)))


# ------------------------------------------------------------------ host memory (the step's --mem)
def pow2ceil(x: int) -> int:
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


HOST_BASE_GIB = 17.0      # D's HOST_EVIDENCE (ctxsweep_core): the process's host memory beside the pinned KV


def host_gib(B: int, S: int = 16128, steps: int = 24, sweep_steps: int = 24, shipped: bool = False) -> float:
    """The parent's host memory (GiB): 64 pinned host-row tensors (32 layers x K, V) of B x n_host x H x D bf16,
    each rounded UP to a power of two (the CUDA caching host allocator; D's HOST_EVIDENCE), the build's temporary
    prefill caches (64 x pow2((S + 8192) x H x D x 2), reused), the packer's staging ring, HOST_BASE_GIB. shipped:
    the shipped child's own (S + 8192 rows per request) instead, when it is larger (it runs before, alone)."""
    w = widths(S, steps, sweep_steps)
    rows = 2 * L * pow2ceil(int(B) * w["n_host"] * H * D * ELEM)
    tmp = 2 * L * pow2ceil((int(S) + 8192) * H * D * ELEM)
    stage = pow2ceil(2 * (256 * 64 * 8 + 2 * 256 * 64 * D * ELEM))
    par = (rows + tmp + stage) / GIB + HOST_BASE_GIB
    if shipped:
        ch = 2 * L * pow2ceil(int(B) * (int(S) + 8192) * H * D * ELEM) / GIB + HOST_BASE_GIB
        return max(par, ch)
    return par


# ------------------------------------------------------------------ the registered cell (docs/R2_BIGB_PLAN.md 4)
BUDGET_GB = 76.0
CELL_POINT = dict(X=64, X_pool=64, R_ver=16, steps=24, K=3, LAMBDA=7, S=16128, layout="pool", backend="b2",
                  mode="exact", arrival_lag=2)


def cell_plan(C: int, budget_gb: float = BUDGET_GB, model: Optional[MemModel] = None) -> Dict:
    """The cell bigb at C: B 128 (the stepping stone) and B_max (the largest B whose MODEL peak <= budget_gb) on the
    pool layout at CELL_POINT; per B the model's peak, build peak, host GiB and token-log size."""
    m = model or MemModel()
    p = CELL_POINT
    bmax = m.b_max(budget_gb, p["layout"], C, p["X"], p["X_pool"], hi=1024, steps=p["steps"])
    out = dict(C=C, B_max=bmax, points=[])
    for B in (128, bmax):
        out["points"].append(dict(B=B, peak_gb=m.peak_gb(p["layout"], B, C, p["X"], p["X_pool"], steps=p["steps"]),
                                  build_peak_gb=m.build_peak_gb(B), host_gib=host_gib(B),
                                  log_slots=log_slots_for(B, p["steps"], p["K"], p["LAMBDA"]),
                                  stream_peak_gb=m.peak_gb("stream", B, C, p["X"], None, steps=p["steps"])))
    return out


# ------------------------------------------------------------------ the registered time bands (MODEL)
# p5b2 EXACT b2, B 64, per-stream C 96 X 64 (record P5B2): per-category device ms (TickTimer means) and the residual
P5B2_ROUND = dict(gemm=24.620, att=12.856, sel=18.520, store=1.654, twin=119.411, res=251.713, residual=918.190)
P5B2_DRAFT = dict(gemm=25.059, att=13.469, sel=20.281, store=1.838, twin=57.977, res=153.431, residual=358.698)
# the B-independent fraction of each category at B 64 (the rest grows linearly with B); the band moves every f by
# -0.3 (high) / +0.3 (low), clipped to [0, 1]. gemm is separate: EXACT pads to m_fix = 4B rows.
FIXED_FRACTION = dict(att=0.57, sel=0.60, store=0.30, twin=0.80, res=0.30, residual=0.15)
GEMM_SLOPE_MS_PER_ROW = dict(center=0.0622, low=0.045, high=0.080)   # the forecast's GEMM(287) -> GEMM(861) slope
GEMM_ROWS_AT_ANCHOR = 256


def _gemm_ms(base: float, B: int, band: str) -> float:
    m = 4 * int(B)
    return base + GEMM_SLOPE_MS_PER_ROW[band] * max(0, m - GEMM_ROWS_AT_ANCHOR) * base / P5B2_DRAFT["gemm"]


def tick_bands(B: int) -> Dict[str, Dict[str, float]]:
    """Per band (low, center, high): T_tick (round, ms), td_dev (the draft-only tick's gemm + att + sel + store, the
    reconciliation's td proxy) and the draft-only tick's wall (ms). MODEL: scaled from p5b2's B 64 categories."""
    out = {}
    for band, d in (("low", 0.3), ("center", 0.0), ("high", -0.3)):
        def sc(x, cat):
            f = min(1.0, max(0.0, FIXED_FRACTION[cat] + d))
            return x * (f + (1 - f) * int(B) / 64.0)
        r = {c: (sc(v, c) if c != "gemm" else _gemm_ms(v, B, band)) for c, v in P5B2_ROUND.items()}
        q = {c: (sc(v, c) if c != "gemm" else _gemm_ms(v, B, band)) for c, v in P5B2_DRAFT.items()}
        out[band] = dict(T_tick=sum(r.values()), td_dev=q["gemm"] + q["att"] + q["sel"] + q["store"],
                         draft_tick=sum(q.values()), gemm=q["gemm"])
    return out
