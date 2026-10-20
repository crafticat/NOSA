"""EQUAL-MEMORY HEADLINE (paper 4.1): original NOSA vs cached NOSA vs the speculative design (stage 3, the R2 pilot's
round:K-draft-then-verify loop, LOCKSTEP) at the SAME workload and the SAME total GPU-memory budget, each family free
to pick its cache capacity and batch inside the budget. The pure, device-free pieces (CPU tests:
benchmarks/Efficiency/eqmem_tests/test_eqmem.py); the GPU driver is eqmem_bench.py, the holder step eqmem_step.sh.

CELLS. A cell is one configuration of one family; one cell = one process = one fresh prefill:
  original        NOSA's shipped offloading engine, P = 0: 64 GPU slots per (layer, KV head, request) (63 usable + tail)
  cached:<P>      the victim pool, NOSI_POOL_BLOCKS = P: 64 + P slots (usable capacity 63 + P)
  spec:<K>[:<R>]  the R2 pilot's spec mode, NOSI_VERIFY_ROUND_SLOTS = R (default 62: W = 64 + R + 2 = 128 slots), pool
                  off (the engine refuses round slots with the pool), K draft steps per round, lockstep commit
The family's best MEASURED cell (committed tokens / s, end to end, inside the budget) represents it ("free to pick").

MEMORY FIT (MemFit; every cell's batch is the largest the fit admits). Predicted PyTorch peak allocation, GB:
    alloc(B, slots) = A + b * B + c * B * slots  (+ spec_fixed + spec_per_req * B for the spec family)
  default A 18.83, b 0.04115, c 0.002035: the fit of the 6 [proof] peak readings (torch max_memory_allocated) of the
  16K equal-memory frontier job 2175389 (docs/evidence/ctrl_frontier_2175389/peak_gb.txt: B 16 / 344 at P 0, 318 / 290
  / 249 / 222 at P 10 / 18 / 33 / 50): exact through B 16 and 344 at P 0 and B 249 at P 33, the other three within
  0.14 GB; default attention splits, expandable segments); spec_fixed 1.5 GB and
  spec_per_req 0.012 GB are the R2 pilot's stated scratch (spec_loop_pilot.py SCRATCH_GB_*). The budget is compared with
  the DEVICE-LEVEL use: used(B, slots) = alloc * (1 + reserve_frac) + nonpt_gb (reserve_frac 0.02, nonpt_gb 1.5:
  MODELLED until refit from measured rows: every cell writes its measured row, fit_from_rows refits). Per-stage
  accounting (stage_breakdown): stage 1 = A (weights, engine buffers, prefill activations) + b * B (per-request tables)
  + c * B * 64 (the 64-slot window); stage 2 adds c * B * P (pool slots); stage 3 adds c * B * (R + 2) (round region
  and tail mirror) and the pilot scratch; then the allocator reserve and the non-PyTorch use (context, cuBLAS,
  Triton). A cell is VALID only when its MEASURED device-level peak (reserved peak + the non-PyTorch use sampled in
  the process) is <= the budget; an over-budget cell is reported and never selected.
OTHER BOUNDS on B (plan_cell; the smallest wins, named in bound_by): the int32 envelope of the gathers (host side
  B * (S + max_gen) * H * D < 2^31, GPU side B * slots * 64 * H * D < 2^31; cache_engine.py prefill_update); the
  host memory of the pinned KV (B * (S + max_gen) * H * D * 2 B * 2 * 32 layers + host_fixed <= host_budget); b_cap.
  max_gen (the host KV headroom, an attribute of the engine set before prefill) = round64(N + K + 64).

WORKLOAD. The B requests of a cell are the rows (offset + i) mod n_rows of the 16128-token PG19 workload (94 distinct
documents): B > 94 REUSES documents (distinct and reused are reported); every family's cells use the same rows in the
same order and the same N generated tokens per request (the first from the prefill's logits).

RE-WARM (rewarm_need, as the sustained harness): before a step (lookahead 1) or a round of U = K + 1 tokens (lookahead
U), a fresh warm-up when any of the next `lookahead` tokens needs another pooled-block count (every 64 tokens) or
another stage-1 score width (round128 of the compressed length: about every 2048 tokens, the first at about +272 at
16128). A trajectory holds at most 1024 tokens past the prompt (total_cis, S + 1024 columns).

METRICS (arm_metrics; all from one wall clock; prefill INSIDE): e2e committed tokens / s = B * N / (end - start);
decode tokens / s = B * (N - 1) / (end - prefill end); per-request latency (static batch: every request ends at the
end; its own prefill completion and the first-token time are reported); transferred bytes (blocks x 32 KiB, K + V: the
offloading engine's load mask after every decode_update, the pilot's wire counters for the rounds); data wait (GPU
time of the decode_update windows: diff, pool swap, the synchronous H2D gathers, tail write; the rounds' late-fetch
gathers); memory (allocated / reserved peaks, the sampled device use, non-PyTorch). Uncertainty: a moving-block
bootstrap of decode tokens / s over the steps (or rounds) of one run; across repetitions (fresh processes) a t
interval.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

FAMILIES = ("original", "cached", "spec")
TOPK, BLOCK, H_KV, HEAD_D, N_LAYERS, CTX = 64, 64, 2, 128, 32, 16128
KERNEL, STRIDE, ROUND = 32, 16, 128
CIS_HEADROOM = 1024                       # total_cis columns past the prompt (InfLLMv2CacheLayer.update_uncompressed_cis)
BYTES_PER_BLOCK = 2 * BLOCK * HEAD_D * 2  # K + V of one (layer, head, request) block (spec_loop.BYTES_WIRE_PER_BLOCK)
INT32 = 2 ** 31
DEFAULT_R = 62


# ------------------------------------------------------------------------------------------------------------ cells
@dataclass(frozen=True)
class Cell:
    family: str
    P: int = 0
    K: int = 0
    R: int = 0

    @property
    def slots(self) -> int:
        """GPU slots per (layer, KV head, request): 64 + P + (R + 2 when R > 0)."""
        return TOPK + self.P + (self.R + 2 if self.R > 0 else 0)

    @property
    def usable(self) -> int:
        return TOPK - 1 + self.P

    @property
    def lookahead(self) -> int:
        return self.K + 1 if self.family == "spec" else 1

    @property
    def name(self) -> str:
        if self.family == "original":
            return "original"
        if self.family == "cached":
            return "cached_p%d" % self.P
        return "spec_k%d_r%d" % (self.K, self.R)

    def env(self) -> Dict[str, str]:
        """The engine knobs the process must be started with (read once at import by cache_engine.py)."""
        return dict(NOSI_POOL_BLOCKS=str(self.P), NOSI_VERIFY_ROUND_SLOTS=str(self.R))


def parse_cell(text: str) -> Cell:
    """'original' | 'cached:<P>' (P >= 1) | 'spec:<K>' | 'spec:<K>:<R>' (K >= 1, R >= 1)."""
    p = text.strip().split(":")
    try:
        if p == ["original"]:
            return Cell("original")
        if p[0] == "cached" and len(p) == 2 and int(p[1]) >= 1:
            return Cell("cached", P=int(p[1]))
        if p[0] == "spec" and len(p) in (2, 3) and int(p[1]) >= 1:
            R = int(p[2]) if len(p) == 3 else DEFAULT_R
            if R >= 1:
                return Cell("spec", K=int(p[1]), R=R)
    except ValueError:
        pass
    raise ValueError("cell %r: expected original | cached:<P >= 1> | spec:<K >= 1>[:<R >= 1>]" % text)


# -------------------------------------------------------------------------------------------------------- memory fit
@dataclass(frozen=True)
class MemFit:
    A: float = 18.83
    b: float = 0.04115
    c: float = 0.002035
    reserve_frac: float = 0.02
    nonpt_gb: float = 1.5
    spec_fixed_gb: float = 1.5
    spec_per_req_gb: float = 0.012
    host_fixed_gb: float = 24.0             # the model's host copy at load, python, the tokenizer, pinned staging
    source: str = ("A, b, c: the 6 [proof] peaks of job 2175389 (16128, P 0/10/18/33/50; exact at three, <= 0.14 GB); "
                   "reserve_frac, nonpt_gb, host_fixed_gb MODELLED; spec terms: the R2 pilot's stated scratch")

    def alloc_gb(self, cell: Cell, B: int) -> float:
        a = self.A + self.b * B + self.c * B * cell.slots
        if cell.family == "spec":
            a += self.spec_fixed_gb + self.spec_per_req_gb * B
        return a

    def used_gb(self, cell: Cell, B: int) -> float:
        return self.alloc_gb(cell, B) * (1.0 + self.reserve_frac) + self.nonpt_gb

    def as_dict(self) -> Dict[str, Any]:
        return dict(A=self.A, b=self.b, c=self.c, reserve_frac=self.reserve_frac, nonpt_gb=self.nonpt_gb,
                    spec_fixed_gb=self.spec_fixed_gb, spec_per_req_gb=self.spec_per_req_gb,
                    host_fixed_gb=self.host_fixed_gb, source=self.source)


FRONTIER_2175389 = (                       # (B, P, peak GB): docs/evidence/ctrl_frontier_2175389/peak_gb.txt + manifest
    (16, 0, 21.57), (344, 0, 77.79), (318, 10, 79.94), (290, 18, 79.24), (249, 33, 78.23), (222, 50, 79.48))


def stage_breakdown(fit: MemFit, cell: Cell, B: int) -> Dict[str, float]:
    """The predicted device use split by stage (module docstring MEMORY FIT); the parts sum to used_gb."""
    st1 = fit.A + fit.b * B + fit.c * B * TOPK
    pool = fit.c * B * cell.P
    rnd = fit.c * B * (cell.R + 2 if cell.R > 0 else 0)
    scratch = (fit.spec_fixed_gb + fit.spec_per_req_gb * B) if cell.family == "spec" else 0.0
    alloc = st1 + pool + rnd + scratch
    reserve = alloc * fit.reserve_frac
    return dict(stage1_weights_engine_window=st1, stage2_pool_slots=pool, stage3_round_slots=rnd,
                stage3_pilot_scratch=scratch, alloc=alloc, allocator_reserve=reserve, non_pytorch=fit.nonpt_gb,
                used=alloc + reserve + fit.nonpt_gb)


def _solve3(m: List[List[float]], v: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting for a 3 x 3 system."""
    a = [row[:] + [vv] for row, vv in zip(m, v)]
    n = 3
    tol = 1e-10 * max(abs(x) for row in m for x in row)
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) <= tol:
            raise ValueError("singular fit: the rows do not vary B and B * slots independently")
        a[i], a[p] = a[p], a[i]
        for r in range(i + 1, n):
            f = a[r][i] / a[i][i]
            for k in range(i, n + 1):
                a[r][k] -= f * a[i][k]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][k] * x[k] for k in range(i + 1, n))) / a[i][i]
    return x


def fit_abc(rows: Sequence[Tuple[int, int, float]]) -> Tuple[float, float, float, float]:
    """Least squares alloc = A + b * B + c * B * slots over rows (B, slots, alloc GB); returns (A, b, c, max |residual|)."""
    if len(rows) < 3:
        raise ValueError("need >= 3 rows")
    X = [(1.0, float(B), float(B) * float(s)) for B, s, _ in rows]
    y = [float(g) for _, _, g in rows]
    m = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
    v = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(3)]
    A, b, c = _solve3(m, v)
    res = max(abs(A + b * x[1] + c * x[2] - yy) for x, yy in zip(X, y))
    return A, b, c, res


def fit_from_rows(rows: Sequence[Dict[str, Any]], base: MemFit = MemFit()) -> MemFit:
    """Refit from MEASURED cell rows (eqmem_bench writes one per cell: family, B, slots, alloc_peak_gb,
    reserved_peak_gb, nonpt_gb). A, b, c from the original / cached rows (>= 3, else kept); reserve_frac = the mean
    (reserved - alloc) / alloc and nonpt_gb = the max measured non-PyTorch use over all rows; the spec terms = the mean
    excess of the spec rows over the refit A + b B + c B slots, split as fixed (kept per-request)."""
    base_rows = [r for r in rows if r["family"] in ("original", "cached")]
    A, b, c = base.A, base.b, base.c
    src = []
    if len(base_rows) >= 3:
        A, b, c, res = fit_abc([(int(r["B"]), int(r["slots"]), float(r["alloc_peak_gb"])) for r in base_rows])
        src.append("A, b, c refit over %d measured rows (max residual %.2f GB)" % (len(base_rows), res))
    rf = base.reserve_frac
    if rows:
        rf = sum((float(r["reserved_peak_gb"]) - float(r["alloc_peak_gb"])) / float(r["alloc_peak_gb"]) for r in rows) / len(rows)
        src.append("reserve_frac from %d rows" % len(rows))
    npt = max((float(r["nonpt_gb"]) for r in rows), default=base.nonpt_gb)
    spec_rows = [r for r in rows if r["family"] == "spec"]
    sf = base.spec_fixed_gb
    if spec_rows:
        ex = [float(r["alloc_peak_gb"]) - (A + b * int(r["B"]) + c * int(r["B"]) * int(r["slots"]))
              - base.spec_per_req_gb * int(r["B"]) for r in spec_rows]
        sf = max(0.0, sum(ex) / len(ex))
        src.append("spec_fixed from %d spec rows" % len(spec_rows))
    return replace(base, A=A, b=b, c=c, reserve_frac=max(0.0, rf), nonpt_gb=npt, spec_fixed_gb=sf,
                   source="; ".join(src) or base.source)


# -------------------------------------------------------------------------------------------------------------- plan
def max_gen_for(N: int, K: int = 0) -> int:
    """The host KV headroom: round64(N + K + 64) tokens (the tail write-back and the drafted rows stay inside)."""
    x = int(N) + int(K) + 64
    return (x + BLOCK - 1) // BLOCK * BLOCK


def host_gb(B: int, S: int, max_gen: int, fixed: float) -> float:
    return B * (S + max_gen) * H_KV * HEAD_D * 2 * 2 * N_LAYERS / 1e9 + fixed


def plan_cell(cell: Cell, budget_gb: float, N: int, fit: MemFit = MemFit(), margin_gb: float = 1.0, S: int = CTX,
              host_budget_gb: float = 220.0, b_cap: int = 4096) -> Dict[str, Any]:
    """The largest batch of a cell inside every bound (module docstring); refuses (ValueError) when none fits or the
    trajectory would leave the total_cis headroom."""
    if N < 2:
        raise ValueError("N = %d: need >= 2 generated tokens" % N)
    if N + cell.K + 1 > CIS_HEADROOM:
        raise ValueError("N %d + K %d leaves the %d-column total_cis headroom" % (N, cell.K, CIS_HEADROOM))
    mg = max_gen_for(N, cell.K)
    per_req = fit.b + fit.c * cell.slots + (fit.spec_per_req_gb if cell.family == "spec" else 0.0)
    fixed = fit.A + (fit.spec_fixed_gb if cell.family == "spec" else 0.0)
    room = (budget_gb - margin_gb - fit.nonpt_gb) / (1.0 + fit.reserve_frac) - fixed
    bounds = dict(
        memory=int(math.floor(room / per_req)) if room > 0 else 0,
        int32_host=(INT32 - 1) // ((S + mg) * H_KV * HEAD_D),
        int32_gpu=(INT32 - 1) // (cell.slots * BLOCK * H_KV * HEAD_D),
        host_memory=int(math.floor((host_budget_gb - fit.host_fixed_gb) / (host_gb(1, S, mg, 0.0)))),
        cap=int(b_cap))
    bound_by = min(bounds, key=lambda k: (bounds[k], list(bounds).index(k)))
    B = bounds[bound_by]
    if B < 1:
        raise ValueError("cell %s: no batch fits (bounds %s)" % (cell.name, bounds))
    return dict(cell=cell.name, family=cell.family, P=cell.P, K=cell.K, R=cell.R, slots=cell.slots, usable=cell.usable,
                B=B, bound_by=bound_by, bounds=bounds, max_gen=mg, N=N, S=S, budget_gb=budget_gb, margin_gb=margin_gb,
                predicted_alloc_gb=fit.alloc_gb(cell, B), predicted_used_gb=fit.used_gb(cell, B),
                predicted_host_gb=host_gb(B, S, mg, fit.host_fixed_gb), stages=stage_breakdown(fit, cell, B),
                fit=fit.as_dict())


def plan_key(plan: Dict[str, Any]) -> Tuple:
    """What every cell of one comparison must share."""
    return (plan["budget_gb"], plan["margin_gb"], plan["N"], plan["S"])


# ---------------------------------------------------------------------------------------------------------- workload
def doc_rows(B: int, n_rows: int, offset: int = 0) -> Dict[str, Any]:
    """Request i -> workload row (offset + i) mod n_rows; distinct rows and whether documents repeat."""
    if B < 1 or n_rows < 1 or offset < 0:
        raise ValueError("B %d, n_rows %d, offset %d" % (B, n_rows, offset))
    rows = [(offset + i) % n_rows for i in range(B)]
    distinct = len(set(rows))
    return dict(rows=rows, distinct=distinct, reused=distinct < B, repeats=B - distinct)


# ------------------------------------------------------------------------------------------------------------ rewarm
def round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def _advance(n: int, m: int) -> Tuple[int, int]:
    return (m + 1, STRIDE + 1) if n >= KERNEL else (m, n + 1)


def shape_at(seq: int, n: int, m: int) -> Tuple[int, int]:
    """(pooled blocks, stage-1 score width) a step starting at cache length seq needs."""
    mp = m + (1 if n >= KERNEL else 0)
    return -(-(seq + 1) // BLOCK), round_up(mp, ROUND)


def warm_state(seq: int, n: int, m: int) -> Dict[str, int]:
    ol, w = shape_at(seq, n, m)
    return dict(seq=int(seq), out_len=ol, width=w)


def rewarm_need(seq: int, n: int, m: int, warm: Optional[Dict[str, int]], lookahead: int = 1) -> Tuple[bool, str]:
    """(re-warm now?, reason): 'first', 'out_len' or 'width' when one of the next `lookahead` tokens (from the live
    cache length seq, no_compress_k_len n, compressed length m) needs other shapes than the last warm-up's."""
    if warm is None:
        return True, "first"
    s, nn, mm = int(seq), int(n), int(m)
    for _ in range(max(1, int(lookahead))):
        ol, w = shape_at(s, nn, mm)
        if ol != warm["out_len"]:
            return True, "out_len"
        if w != warm["width"]:
            return True, "width"
        mm, nn = _advance(nn, mm)
        s += 1
    return False, ""


def simulate(seq0: int, n0: int, m0: int, steps: int) -> List[Tuple[int, int, int]]:
    out, s, n, m = [], int(seq0), int(n0), int(m0)
    for _ in range(int(steps)):
        out.append((s, n, m))
        m, n = _advance(n, m)
        s += 1
    return out


# ----------------------------------------------------------------------------------------------------------- metrics
def pctl(xs: Sequence[float], q: float) -> Optional[float]:
    v = sorted(float(x) for x in xs)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    f, c = int(math.floor(k)), int(math.ceil(k))
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def mean(xs: Sequence[float]) -> Optional[float]:
    v = [float(x) for x in xs]
    return sum(v) / len(v) if v else None


def ratio_ci(tokens: Sequence[float], ms: Sequence[float], block: int = 16, n_boot: int = 2000, seed: int = 0,
             alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """Moving-block bootstrap CI of sum(tokens) / sum(ms) * 1000 (tokens / s) over consecutive units (steps or
    rounds), keeping each unit's (tokens, ms) pair."""
    n = len(tokens)
    if n == 0 or n != len(ms):
        return None, None
    rng = random.Random(seed)
    blk = max(1, min(block, n))
    starts = list(range(0, n - blk + 1))
    vals = []
    for _ in range(n_boot):
        ix: List[int] = []
        while len(ix) < n:
            s0 = rng.choice(starts)
            ix.extend(range(s0, s0 + blk))
        ix = ix[:n]
        tt, mm = sum(tokens[i] for i in ix), sum(ms[i] for i in ix)
        vals.append(tt / mm * 1000.0 if mm > 0 else float("inf"))
    return pctl(vals, 100 * alpha / 2), pctl(vals, 100 * (1 - alpha / 2))


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def reps_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Across repetitions (fresh processes): n, mean, sd, the 95% t interval (n >= 2), min, max."""
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return dict(n=0)
    m = sum(v) / n
    if n == 1:
        return dict(n=1, mean=m, sd=None, ci_lo=None, ci_hi=None, min=m, max=m)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    t = T975.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, ci_lo=m - h, ci_hi=m + h, min=min(v), max=max(v))


def arm_metrics(B: int, gen_tokens: int, t_start: float, t_prefill_end: float, t_end: float,
                prefill_done_ms: Sequence[float], first_token_ms: float, unit_tokens: Sequence[float],
                unit_ms: Sequence[float], loads: int, wait_ms: float, n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """One cell's paper metrics (module docstring METRICS). gen_tokens = tokens generated per request (the first from
    the prefill); unit_tokens / unit_ms = the decode units (steps or rounds: tokens committed per request, GPU ms) for
    the bootstrap; loads = blocks fetched host -> GPU; wait_ms = GPU ms of the data windows."""
    if not (t_start <= t_prefill_end <= t_end) or B < 1 or gen_tokens < 2:
        raise ValueError("arm_metrics: times %s, B %d, tokens %d" % ((t_start, t_prefill_end, t_end), B, gen_tokens))
    e2e_s, pre_s, dec_s = t_end - t_start, t_prefill_end - t_start, t_end - t_prefill_end
    committed = B * gen_tokens
    dec_tokens = B * (gen_tokens - 1)
    lo, hi = ratio_ci([x * B for x in unit_tokens], unit_ms, n_boot=n_boot, seed=seed)
    unit_rate = (sum(unit_tokens) * B / sum(unit_ms) * 1000.0) if unit_ms and sum(unit_ms) > 0 else None
    nbytes = int(loads) * BYTES_PER_BLOCK
    return dict(
        batch=B, gen_tokens=gen_tokens, committed_tokens=committed,
        e2e_s=e2e_s, prefill_s=pre_s, decode_s=dec_s,
        e2e_tok_s=committed / e2e_s if e2e_s > 0 else None,
        decode_tok_s=dec_tokens / dec_s if dec_s > 0 else None,
        decode_tok_s_gpu_units=unit_rate, decode_tok_s_ci=[lo, hi],
        latency_s=dict(end_to_end=e2e_s, first_token=first_token_ms / 1000.0,
                       own_prefill_p50=(pctl(prefill_done_ms, 50) or 0.0) / 1000.0,
                       own_prefill_max=(max(prefill_done_ms) if prefill_done_ms else 0.0) / 1000.0),
        transferred=dict(blocks=int(loads), bytes=nbytes, bytes_per_committed_token=nbytes / dec_tokens if dec_tokens else None),
        data_wait=dict(ms=wait_ms, ms_per_decode_token=wait_ms / (gen_tokens - 1),
                       share_of_decode=(wait_ms / 1000.0) / dec_s if dec_s > 0 else None))


def memory_record(alloc_peak: float, reserved_peak: float, samples: Sequence[Dict[str, float]], budget_gb: float) -> Dict[str, Any]:
    """Peaks (GB) and the budget verdict: used_peak_est = reserved peak + the largest sampled non-PyTorch use (device
    use - reserved at the same instant); also the largest sampled device use. VALID iff used_peak_est <= budget."""
    nonpt = max((float(s["used_gb"]) - float(s["reserved_gb"]) for s in samples), default=0.0)
    used_sampled = max((float(s["used_gb"]) for s in samples), default=0.0)
    est = float(reserved_peak) + max(0.0, nonpt)
    return dict(alloc_peak_gb=float(alloc_peak), reserved_peak_gb=float(reserved_peak), nonpt_gb=max(0.0, nonpt),
                used_sampled_max_gb=used_sampled, used_peak_est_gb=est, budget_gb=budget_gb,
                verdict="VALID" if est <= budget_gb else "OVER_BUDGET", samples=list(samples))


def prefix_agreement(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Per common request: the length of the common prefix of two committed sequences (greedy decodes of the same
    document in different processes / batches may part at a near tie: an observation, not a gate)."""
    n = min(len(a), len(b))
    lens = []
    for i in range(n):
        x, y = a[i], b[i]
        k = 0
        while k < min(len(x), len(y)) and int(x[k]) == int(y[k]):
            k += 1
        lens.append(k)
    agree = [k >= min(len(a[i]), len(b[i])) for i, k in enumerate(lens)]
    parted = [k for k, ok in zip(lens, agree) if not ok]
    # prefix_min: the EARLIEST divergence among the requests that part (None when every request agrees); a short
    # request that agrees fully is not a divergence (the first version took the minimum over all requests)
    return dict(requests=n, full_agreement=sum(agree), prefix_p50=pctl(lens, 50),
                prefix_min=min(parted) if parted else None)


# ------------------------------------------------------------------------------------------------------------ report
def select_best(results: Sequence[Dict[str, Any]], family: str, key: str = "e2e_tok_s") -> Optional[Dict[str, Any]]:
    """The family's best VALID cell (no error, inside the budget) by the mean of `key` over its repetitions."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if r.get("family") != family or r.get("error") or r.get("memory", {}).get("verdict") != "VALID":
            continue
        by.setdefault(r["cell"], []).append(r)
    best = None
    for cell, rs in by.items():
        v = mean([x["metrics"][key] for x in rs if x["metrics"].get(key) is not None])
        if v is not None and (best is None or v > best[1]):
            best = (cell, v, rs)
    if best is None:
        return None
    return dict(cell=best[0], mean=best[1], reps=len(best[2]))


def comparison(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The headline table: every cell (reps summarized), the best per family, speedups of each family's best over
    the original's and the cached best (e2e and decode tokens / s). Refuses results of different budgets / N / S."""
    keys = {tuple(r["plan_key"]) for r in results}
    if len(keys) > 1:
        raise ValueError("results of different comparisons mixed: %s" % sorted(keys))
    cells: Dict[str, Dict[str, Any]] = {}
    for r in results:
        c = cells.setdefault(r["cell"], dict(cell=r["cell"], family=r["family"], reps=[], errors=0, over_budget=0))
        if r.get("error"):
            c["errors"] += 1
            continue
        if r["memory"]["verdict"] != "VALID":
            c["over_budget"] += 1
        c["reps"].append(r)
    table = {}
    for name, c in cells.items():
        rs = c["reps"]
        row = dict(family=c["family"], runs=len(rs), errors=c["errors"], over_budget=c["over_budget"])
        if rs:
            row["batch"] = rs[0]["metrics"]["batch"]
            for k in ("e2e_tok_s", "decode_tok_s", "decode_tok_s_gpu_units"):
                row[k] = reps_summary([x["metrics"][k] for x in rs if x["metrics"].get(k) is not None])
            row["bytes_per_committed_token"] = mean([x["metrics"]["transferred"]["bytes_per_committed_token"] for x in rs])
            row["data_wait_share"] = mean([x["metrics"]["data_wait"]["share_of_decode"] for x in rs])
            row["latency_e2e_s"] = mean([x["metrics"]["latency_s"]["end_to_end"] for x in rs])
            row["used_peak_est_gb"] = max(x["memory"]["used_peak_est_gb"] for x in rs)
        table[name] = row
    best = {f: select_best(results, f) for f in FAMILIES}
    out = dict(cells=table, best=best)
    for key in ("e2e_tok_s", "decode_tok_s"):
        bv = {f: (select_best(results, f, key) or {}).get("mean") for f in FAMILIES}
        sp = {}
        for f in FAMILIES:
            for ref in ("original", "cached"):
                if bv.get(f) and bv.get(ref) and f != ref:
                    sp["%s_vs_%s" % (f, ref)] = bv[f] / bv[ref] - 1.0
        out["speedup_" + key] = sp
    return out
