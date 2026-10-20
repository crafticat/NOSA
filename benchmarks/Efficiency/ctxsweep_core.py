"""CONTEXT SWEEP (paper 4.2): committed throughput and latency across context lengths, batch sizes and cache
capacities, every configuration inside the same total GPU-memory budget (the 4.1 rules of eqmem_core.py, made
S-aware). The pure, device-free pieces (CPU tests: benchmarks/Efficiency/eqmem_tests/test_ctxsweep.py); the GPU driver
is ctxsweep_bench.py (one arm = eqmem_bench.run_arm with the S-aware plan below), the holder step ctxsweep_step.sh, the
chain's cell list ctxsweep_cells.txt.

CELLS. An eqmem_core cell (original | cached:<P> | spec:<K>[:<R>]) at a context S and a batch B (explicit, or 'max':
the largest B every bound admits). original and cached are the BASELINE families. Every other family (spec now, the
per-request R2 engine later) is a CANDIDATE: it is compared with the best baseline cell at the same (S, B), and at
equal memory with the best baseline cell of the same S (each family free to pick its batch and its capacity).

CONTEXTS. S in {16128, 65280, 130816}: 94 / 55 / 24 distinct PG19 documents (make_workload.py --ctx). A batch above
that REUSES documents (reported, never refused). S mod 64 = 0 at all three, so the shipped decode is valid for 64 steps
after a warm-up (R2_ENGINE_FINDINGS F1). eqmem_core.rewarm_need re-warms at every 64-token block and at every round128
width change of the compressed length; at all three contexts the first width change is at decode token 288
(compressed_state: m0 = S / 16 - 1 = 111 mod 128); rewarm_schedule replays the rule. F3: total_cis has S + 1024 rows
(N + K + 1 <= 1024). F4 is parity (kept). The findings document has no F5 (F1-F4 only).

GPU MEMORY (SMemFit; PyTorch allocated peak, GB; the budget is compared with the DEVICE-LEVEL use, as in 4.1):
    alloc(S, B, slots) = A(S) + b(S) * B + c(S) * B * slots   (+ spec_fixed + spec_req(S) * B for the spec family)
    A(S) = A16 + a1 * (S - S0) + a2 * (S^2 - S0^2), S0 = 16128: the weights, the cos/sin cache and the PREFILL
        transient (one request at a time: the MLP activations grow with S, the stage-1 scores with S^2 / 16)
    b(S) = b16 + b1 * (S - S0): per-request tables. B1_BYTES = 1180.3 B per prompt token per request: the compressed
        keys 1024 (32 layers x 2 heads x 128 x 2 B / 16), total_cis 128, compressed_cis 8, the decode score buffer 8,
        the pooling buffers 0.28, the prompt ids and positions on the GPU 12
    spec_req(S) = spec_req16 + spec_b1 * (S - S0): the round journal keeps the round-start compressed-key tensor while
        a draft re-binds it (spec_loop.LayerJournal.compress_k_ref): 1024 B per prompt token per request
  prior_fit (MODELLED until refit): the 16K ANCHOR is B's 4.1 fit (eqmem_core.MemFit: A16 18.83, b16 0.04115,
  c16 0.002035, the spec terms, the reserve and non-PyTorch terms), so at S = 16128 every 4.1 cell plans exactly as
  4.1 (CPU test CS10). c(S): c16 at 16128, c_long = max(the code's bytes per slot per request, 0.0021053; the measured
  slopes at long context: 64K x 64 over P 0..97, job 2174639; 128K x 60 over P 0 / 50, job 2174878) from 65280 on,
  linear in between (B's three-term fit puts part of the per-request bytes into b16). b1 from the code; a1, a2 solved
  so that the two long-context P = 0 peaks (64K x 64: 40.82 GB, 128K x 60: 55.11 GB) are exact. The old harness
  (test_nosa_pg19) read 2.5-3 GB ABOVE the 4.1 harness at 16K (job 2188818 h41 rows), so the prior is conservative.
  fit_ctx_from_rows: every measured cell writes a row (S, B, slots, peaks). Per context, the baseline rows refit
  (A_S, b_S): two or more batch sizes give the least-squares slope and the upper-envelope intercept; one batch size
  gives b_S = 1.25 x the prior's b(S) and the envelope intercept. A cell is VALID only when its MEASURED device-level
  peak is <= the budget (eqmem_core.memory_record).
  PHASES (run cs1, CS1_ROWS): the PREFILL phase sets every measured peak (128K x 4: 38.5 GB in prefill, 17.6 GB in
  decode). The decode-phase peaks lie on their own line: A_dec 16.45 GB at 64K and 128K, b_dec 0.081 / 0.162 GB per
  request, the same slot term. With two batch sizes of decode rows at a context, the peak model there is the larger of
  the prefill line (A_S + b_S B + c B slots) and the decode line (A_dec + b_dec B + c B slots + the spec terms: the
  pilot's scratch and the journal exist only in the decode). Without them, the 4.1 rule (spec terms on the prefill
  line) stays, which is conservative for spec.

HOST MEMORY (HostFit; GiB, the unit of srun --mem). The pinned host KV is 64 tensors (32 layers x K, V) of
  B * (S + max_gen) * 2 heads * 128 * 2 B. MEASURED (HOST_EVIDENCE, 8 rows, VmHWM): the CUDA caching host allocator
  ROUNDS EVERY PINNED TENSOR UP TO A POWER OF TWO. 16K x 222..344, 64K x 64 and 128K x 60 all read 273.0-273.1 GiB =
  64 x 4 GiB + 17 GiB; 16K x 16 reads 32.9 = 64 x 0.25 + 17. The nominal size explains none of the large rows (64K x 64
  is 143.5 GiB nominal). The 4.1 original cell (16K x 304, max_gen 576) was OOM-killed at 240G with 30 of 32 layers
  pinned: 60 x 4 GiB = 240 GiB (step 2188818.38). Its nominal KV (155 GiB) could not have been killed. This harness
  (run cs1, 11 cells, CS1_ROWS): 64K x 8 and 128K x 4 read 48.86 GiB = 64 x 0.5 + 16.86; 64K x 32 and 128K x 16 read
  144.8-144.9 GiB = 64 x 2 + 16.85-16.91; 16K x 16 reads 32.86-32.89 = 64 x 0.25 + 16.86. Refit base 17.9 GiB.
  Modes: '2x' is the DEFAULT until this harness measures a row (twice the nominal KV; always >= the power-of-two size);
  'pow2' comes from fit_host_from_rows (the base from the measured rows + 1 GiB); 'nominal' is for the evidence test.
  host(B) = base + KV + the prompt ids on the host (2 x B x S x 8 B). The step's --mem: need = ceil10(1.08 x host + 6),
  at least 64, at most 440 (the holder has 480G). The host bound of a cell = the largest B whose need fits its --mem.

OTHER BOUNDS (plan_ctx_cell; the smallest wins and is named in bound_by; a refusal raises ValueError naming the rule):
  int32_host  B * (S + max_gen) * 2 * 128 < 2^31 (the gathers' int32 offsets: 63 at 128K, 127 at 64K)
  int32_gpu   B * slots * 64 * 2 * 128 < 2^31
  memory      the GPU fit (budget - margin - the non-PyTorch use, the allocator reserve)
  host_memory the HostFit at the cell's --mem
  cap         b_cap
  Refusals: F3 (N + K + 1 > 1024); rope (S + N + 2K + 1 > 132096, the cos/sin cache of max_length 128 x 1024 + 1024);
  prefill (A(S) alone leaves no room for one request); an explicit batch above the largest admitted; --mem outside
  [64, 440]. Documents: reuse is reported (distinct, repeats).

LATENCY (latency_metrics; ONE clock, the arm's CUDA events from the prefill start): TTFT (static batch: every request
  gets its first token when the batch prefill ends; each request's own prefill end is reported too); the per-token
  decode latency (shipped: the interval between consecutive step-end events, one token per request per step; spec:
  each round's GPU time / its committed tokens, and the round time = the commit interval) as mean, p50, p95 and max;
  TPOT = (end - first token) / (N - 1); the committed tokens / s on the same clock (the wall-clock e2e stays in
  eqmem_core.arm_metrics).

  steady_tok_s (the report): B x 1000 / the p50 per-token latency, a decode rate without the re-warm steps.

TIME (est_minutes, per cell; MEASURED inputs, run cs1): prefill 0.168 / 0.1894 / 0.2140 ms per prompt token at
  16K / 64K / 128K (cs1: 2.71 s per 16K request, 12.36 per 64K, 28.0 per 128K; 4.1 h41: 2.55-2.63 per 16K); the first
  decode step about 9 s; a RE-WARM step about 6 s at every context and batch (the warm-up path re-captures the CUDA
  graphs; 8 re-warms in N = 512 take 75-80 % of the decode time at B <= 32); a steady shipped step (step_ms)
  15 + 0.11 S / 1000 + B (0.45 + 0.0012 S / 1000) ms, the per-request part x 0.8 with a pool; the spec family x 2.2 /
  2.8 / 4.3 for K = 1 / 2 / 3 (4.1 h41, the pre-fix pilot: spec k2 287 s and k3 436 s of decode at B 161 against 103 s
  estimated shipped); 0.7 min of process overhead; the FIRST cell of a run also compiles its kernels (+100 s of prefill:
  cs1 64K x 8). Check: 4.1 cached p50 at B 191 is estimated 11.0 min; the step took 10:22.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import eqmem_core as EC

S0 = EC.CTX                                   # 16128: the 4.1 context, the anchor of the S terms
ROPE_POSITIONS = 128 * 1024 + 1024            # nosa_llama.Llama(max_length=128 * 1024): arange(max_length + 1024)
GIB = 2 ** 30
MEM_FLOOR_GIB, MEM_MAX_GIB = 64, 440          # the step's --mem range (the holder has 480G)
MEM_SLACK_FRAC, MEM_SLACK_GIB = 0.08, 6.0
HOST_REFIT_SLACK_GIB = 1.0
BASELINE_FAMILIES = ("original", "cached")


# --------------------------------------------------------------------------------------------------------- contexts
@dataclass(frozen=True)
class Context:
    S: int
    stem: str                                 # <stem>_tokens.bin, <stem>_manifest.json
    dirname: str                              # under kstep-cpu-runs/workload/
    n_docs: int


CONTEXTS = {16128: Context(16128, "c16k", "pg19-test-c16k", 94),
            65280: Context(65280, "ctx65280", "pg19-test-ctx65280", 55),
            130816: Context(130816, "ctx130816", "pg19-test-ctx130816", 24)}


def f1_window(S: int) -> int:
    """Valid shipped decode steps after a warm-up (R2_ENGINE_FINDINGS F1): 64 - S mod 64."""
    return EC.BLOCK - int(S) % EC.BLOCK


def compressed_state(S: int) -> Tuple[int, int]:
    """(no_compress_k_len n0, compressed length m0) after the prefill of S tokens: m0 = (S - 32) // 16 + 1 windows
    (nosa_llama.calc_chunks_with_stride), n0 = S - 16 * m0 uncompressed keys (update_no_compress_k_prefill)."""
    if int(S) < EC.KERNEL:
        raise ValueError("S %d < the compression window %d" % (S, EC.KERNEL))
    m0 = (int(S) - EC.KERNEL) // EC.STRIDE + 1
    return int(S) - EC.STRIDE * m0, m0


def rewarm_schedule(S: int, tokens: int, lookahead: int = 1, commit: Optional[int] = None) -> List[Tuple[int, str]]:
    """Replays eqmem_core.rewarm_need over `tokens` decode tokens after a prefill of S: before every step (lookahead 1)
    or round (lookahead U), a re-warm (one shipped token) when one of the next `lookahead` tokens needs other shapes;
    otherwise a round of `commit` tokens (default = lookahead). Every token must have the shapes of the last warm-up,
    else ValueError (as the engine's score_buf.copy_ / pooled-block count would break). Returns (token, reason)."""
    n, m = compressed_state(S)
    s, warm, t, out = int(S), None, 0, []
    c_round = int(lookahead if commit is None else commit)
    while t < int(tokens):
        need, why = EC.rewarm_need(s, n, m, warm, lookahead)
        if need:
            warm = EC.warm_state(s, n, m)
            out.append((t, why))
            c = 1
        else:
            c = max(1, min(c_round, int(tokens) - t))
        for _ in range(c):
            if EC.shape_at(s, n, m) != (warm["out_len"], warm["width"]):
                raise ValueError("token %d at cache length %d needs %s, warmed %s" % (t, s, EC.shape_at(s, n, m), warm))
            m, n = EC._advance(n, m)
            s += 1
            t += 1
    return out


# ------------------------------------------------------------------------------------------------------- GPU memory
C_SLOT_BYTES = (2 * EC.BLOCK * EC.H_KV * EC.HEAD_D * 2 + EC.BLOCK * EC.H_KV * 2) * EC.N_LAYERS   # K + V + bias: 2105344
B1_BYTES = (EC.N_LAYERS * EC.H_KV * EC.HEAD_D * 2 / EC.STRIDE        # compressed keys (compress_k_cache_varlen): 1024
            + EC.N_LAYERS * EC.H_KV * 2                               # total_cis (S + 1024 rows, bf16): 128
            + EC.N_LAYERS * EC.H_KV * 2 / EC.STRIDE                   # compressed_cis: 8
            + EC.N_LAYERS * EC.H_KV * 2 / EC.STRIDE                   # the decode stage-1 score buffer (score_buf): 8
            + (4 * EC.H_KV * 2 + EC.H_KV) / EC.BLOCK                  # pooling_buf_all, max_pooling_buf(_cis), mask_buf
            + 8 + 4)                                                  # the prompt ids (int64) and positions (int32)
SPEC_B1_BYTES = EC.N_LAYERS * EC.H_KV * EC.HEAD_D * 2 / EC.STRIDE    # the journal's round-start compressed keys: 1024

LONG_P0 = ((65280, 64, 40.82),               # job 2174639 p0_L65280_B64: [doc] gpu_peak_gb max of 40.48 / 40.82
           (130816, 60, 55.11))              # job 2174878 p0_L130816_B60: max of 54.73 / 55.11
SLOT_SWEEP_64K = ((0, 40.82), (10, 42.19), (18, 43.27), (33, 45.29), (50, 47.58), (65, 49.61), (97, 53.92))  # B 64, 2174639
SLOT_PAIR_128K = ((0, 55.11), (50, 61.45))   # B 60, job 2174878


def _slope(points: Sequence[Tuple[float, float]]) -> float:
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx <= 0:
        raise ValueError("the points do not vary x")
    return sum((x - mx) * (y - my) for x, y in points) / sxx


def envelope_line(points: Sequence[Tuple[float, float]], slope: Optional[float] = None) -> Tuple[float, float]:
    """(intercept, slope): the least-squares slope (or the given one) and the intercept of the UPPER envelope (no point
    above the line)."""
    sl = _slope(points) if slope is None else float(slope)
    return max(y - sl * x for x, y in points), sl


C_LONG_FROM = 65280                           # c(S) = c_long from this context on (linear from c16 at S0)


@dataclass(frozen=True)
class SMemFit:
    A16: float
    b16: float
    c16: float
    c_long: float
    a1: float
    a2: float
    b1: float = B1_BYTES / 1e9
    S0: int = S0
    reserve_frac: float = 0.02
    nonpt_gb: float = 1.5
    spec_fixed_gb: float = 1.5
    spec_req16_gb: float = 0.012
    spec_b1: float = SPEC_B1_BYTES / 1e9
    # measured, per context: (S, A_S, b_S, distinct batch sizes, A_dec, b_dec); A_dec / b_dec = the DECODE-phase line
    # (decode_alloc_peak_gb rows; None when no row has it): with it the peak is the larger of the two phases
    per_ctx: Tuple[Tuple[Any, ...], ...] = ()
    source: str = "prior_fit"

    def c_at(self, S: int) -> float:
        w = min(1.0, max(0.0, (int(S) - self.S0) / float(C_LONG_FROM - self.S0)))
        return self.c16 + (self.c_long - self.c16) * w

    def _cal(self, S: int):
        for row in self.per_ctx:
            if int(row[0]) == int(S):
                return row
        return None

    def A_at(self, S: int) -> float:
        cal = self._cal(S)
        if cal is not None:
            return float(cal[1])
        d1, d2 = int(S) - self.S0, int(S) ** 2 - self.S0 ** 2
        return self.A16 + self.a1 * d1 + self.a2 * d2

    def b_at(self, S: int) -> float:
        cal = self._cal(S)
        if cal is not None:
            return float(cal[2])
        return self.b16 + self.b1 * (int(S) - self.S0)

    def spec_req_at(self, S: int) -> float:
        return self.spec_req16_gb + self.spec_b1 * (int(S) - self.S0)

    def calibrated(self, S: int) -> int:
        """How many distinct batch sizes calibrate S (0: the prior)."""
        cal = self._cal(S)
        return int(cal[3]) if cal is not None else 0

    def dec_at(self, S: int) -> Optional[Tuple[float, float]]:
        """(A_dec, b_dec) of the measured decode phase at S, or None (then the prefill-phase model holds alone)."""
        cal = self._cal(S)
        if cal is None or len(cal) < 6 or cal[4] is None or cal[5] is None:
            return None
        return float(cal[4]), float(cal[5])

    def phases(self, cell: EC.Cell, S: int) -> Dict[str, Any]:
        """The peak model of a cell at S as two lines in B (alloc = max over the phases of fixed + per_req * B):
        prefill = A(S) + b(S) B + c(S) B slots (the spec terms too while the decode phase is not measured: the 4.1
        rule); decode (measured contexts only) = A_dec + b_dec B + c(S) B slots + the spec terms."""
        spec = cell.family == "spec"
        sf, sr = (self.spec_fixed_gb, self.spec_req_at(S)) if spec else (0.0, 0.0)
        c = self.c_at(S)
        dec = self.dec_at(S)
        if dec is None:
            return dict(prefill=(self.A_at(S) + sf, self.b_at(S) + c * cell.slots + sr), decode=None)
        return dict(prefill=(self.A_at(S), self.b_at(S) + c * cell.slots),
                    decode=(dec[0] + sf, dec[1] + c * cell.slots + sr))

    def alloc_gb(self, cell: EC.Cell, S: int, B: int) -> float:
        ph = self.phases(cell, S)
        lines = [ph["prefill"]] + ([ph["decode"]] if ph["decode"] is not None else [])
        return max(f + p * B for f, p in lines)

    def used_gb(self, cell: EC.Cell, S: int, B: int) -> float:
        return self.alloc_gb(cell, S, B) * (1.0 + self.reserve_frac) + self.nonpt_gb

    def at(self, S: int) -> EC.MemFit:
        """The 4.1 fit (eqmem_core.MemFit) evaluated at S: eqmem_core.plan_cell / stage_breakdown use it unchanged."""
        return EC.MemFit(A=self.A_at(S), b=self.b_at(S), c=self.c_at(S), reserve_frac=self.reserve_frac,
                         nonpt_gb=self.nonpt_gb, spec_fixed_gb=self.spec_fixed_gb,
                         spec_per_req_gb=self.spec_req_at(S),
                         source="ctxsweep_core.SMemFit at S %d (%s)" % (int(S), self.source))

    def as_dict(self) -> Dict[str, Any]:
        return dict(A16=self.A16, b16=self.b16, c16=self.c16, c_long=self.c_long, a1=self.a1, a2=self.a2, b1=self.b1,
                    S0=self.S0,
                    reserve_frac=self.reserve_frac, nonpt_gb=self.nonpt_gb, spec_fixed_gb=self.spec_fixed_gb,
                    spec_req16_gb=self.spec_req16_gb, spec_b1=self.spec_b1, per_ctx=[list(r) for r in self.per_ctx],
                    source=self.source)


def prior_fit(anchor: Optional[EC.MemFit] = None) -> SMemFit:
    """The MODELLED S-aware fit (module docstring GPU MEMORY), computed from the evidence constants. anchor: the 16K
    fit (default eqmem_core.MemFit(), the 4.1 plan's): A16, b16, c16 and the reserve / non-PyTorch / spec terms."""
    m = anchor or EC.MemFit()
    s64 = _slope([(64.0 * P, gb) for P, gb in SLOT_SWEEP_64K])
    (p0, g0), (p1, g1) = SLOT_PAIR_128K
    s128 = (g1 - g0) / (60.0 * (p1 - p0))
    c_long = max(C_SLOT_BYTES / 1e9, s64, s128)
    b1 = B1_BYTES / 1e9
    rows = []
    for S, B, gb in LONG_P0:
        r = gb - (m.A + m.b * B + c_long * B * EC.TOPK) - b1 * B * (S - S0)
        rows.append((float(S - S0), float(S * S - S0 * S0), r))
    (d1, q1, r1), (d2, q2, r2) = rows
    det = d1 * q2 - d2 * q1
    a1 = (r1 * q2 - r2 * q1) / det
    a2 = (d1 * r2 - d2 * r1) / det
    return SMemFit(A16=m.A, b16=m.b, c16=m.c, c_long=c_long, a1=a1, a2=a2, b1=b1, reserve_frac=m.reserve_frac,
                   nonpt_gb=m.nonpt_gb, spec_fixed_gb=m.spec_fixed_gb, spec_req16_gb=m.spec_per_req_gb,
                   source=("prior: the 16K anchor = the 4.1 fit (%s); c_long = max(code %.7f, 64K slope %.7f, 128K slope "
                           "%.7f); b1 code bytes; a1, a2 exact through 64K x 64 and 128K x 60 at P 0"
                           % (m.source[:60], C_SLOT_BYTES / 1e9, s64, s128)))


def fit_ctx_from_rows(rows: Sequence[Dict[str, Any]], prior: Optional[SMemFit] = None) -> SMemFit:
    """Refit from MEASURED cell rows (fit_row.json: family, S (default 16128: the 4.1 rows), B, slots, alloc_peak_gb,
    reserved_peak_gb, nonpt_gb). Per context, the baseline rows give (A_S, b_S) (module docstring); reserve_frac = the
    mean (reserved - alloc) / alloc and nonpt_gb = the max non-PyTorch use over all rows; spec_fixed = the LARGEST
    excess of a spec row over the refit baseline terms and spec_req(S) (never below 0), over the DECODE phase when that
    phase is measured at the row's context (the spec scratch and the journal live in the decode only).
    The decode-phase line (A_dec, b_dec) comes the same way from decode_alloc_peak_gb (rows of this harness only); a
    context with fewer than two batch sizes keeps no decode line (the prefill-phase model alone, conservative)."""
    prior = prior or prior_fit()
    cal = {int(r[0]): tuple(r) for r in prior.per_ctx}
    src = []
    by_s: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("family") in BASELINE_FAMILIES:
            by_s.setdefault(int(r.get("S") or S0), []).append(r)

    def line(rs, key, S, b_one):
        pts = [(float(r["B"]), float(r[key]) - prior.c_at(S) * int(r["B"]) * int(r["slots"])) for r in rs]
        nb = len({int(x) for x, _ in pts})
        if nb >= 2:
            A, b = envelope_line(pts)
            if b < 0:
                A, b = envelope_line(pts, 0.0)
            return A, b, nb
        if b_one is None:
            return None, None, nb
        A, b = envelope_line(pts, b_one)
        return A, b, nb

    for S, rs in sorted(by_s.items()):
        A_S, b_S, nb = line(rs, "alloc_peak_gb", S, 1.25 * prior.b_at(S))
        drs = [r for r in rs if r.get("decode_alloc_peak_gb") is not None]
        A_d, b_d, nbd = line(drs, "decode_alloc_peak_gb", S, None) if drs else (None, None, 0)
        cal[S] = (S, A_S, b_S, nb, A_d, b_d)
        src.append("S %d: A %.2f b %.4f from %d rows (%s)%s" % (
            S, A_S, b_S, len(rs), "slope over %d batch sizes" % nb if nb >= 2 else "one batch size: b = 1.25 x the prior",
            "; decode phase A %.2f b %.4f" % (A_d, b_d) if A_d is not None else ""))
    fit = replace(prior, per_ctx=tuple(cal[k] for k in sorted(cal)))
    rf, npt, sf = prior.reserve_frac, prior.nonpt_gb, prior.spec_fixed_gb
    if rows:
        rf = max(0.0, sum((float(r["reserved_peak_gb"]) - float(r["alloc_peak_gb"])) / float(r["alloc_peak_gb"])
                          for r in rows) / len(rows))
        npt = max(float(r["nonpt_gb"]) for r in rows)
        src.append("reserve_frac, nonpt from %d rows" % len(rows))
    spec_rows = [r for r in rows if r.get("family") == "spec"]
    if spec_rows:
        ex = []
        for r in spec_rows:
            S, B = int(r.get("S") or S0), int(r["B"])
            dec = fit.dec_at(S)
            if dec is not None and r.get("decode_alloc_peak_gb") is not None:
                base, peak = dec[0] + dec[1] * B, float(r["decode_alloc_peak_gb"])
            else:
                base, peak = fit.A_at(S) + fit.b_at(S) * B, float(r["alloc_peak_gb"])
            ex.append(peak - (base + fit.c_at(S) * B * int(r["slots"]) + fit.spec_req_at(S) * B))
        sf = max(0.0, max(ex))
        src.append("spec_fixed from %d spec rows" % len(spec_rows))
    return replace(fit, reserve_frac=rf, nonpt_gb=npt, spec_fixed_gb=sf,
                   source="; ".join(src) if src else prior.source)


# ------------------------------------------------------------------------------------------------------ host memory
HOST_EVIDENCE = (        # (S, B, max_gen, VmHWM GiB, source): one row per cell (the larger of its two documents)
    (16128, 16, 8192, 32.9, "2175389 / 2174878 / 2174639 p0_L16128_B16"),
    (16128, 344, 8192, 273.1, "2175389 p0_L16128_B344"),
    (16128, 318, 8192, 273.1, "2175389 p10_L16128_B318"),
    (16128, 290, 8192, 273.1, "2175389 p18_L16128_B290"),
    (16128, 249, 8192, 273.0, "2175389 p33_L16128_B249"),
    (16128, 222, 8192, 273.1, "2175389 p50_L16128_B222"),
    (65280, 64, 8192, 273.1, "2174639 p0_L65280_B64"),
    (130816, 60, 8192, 273.1, "2174878 p0_L130816_B60"))
HOST_OOM_4_1 = (16128, 304, 576, 30, 240)   # (S, B, max_gen, layers pinned, --mem GiB): step 2188818.38 OUT_OF_MEMORY

# run cs1 (60b5b03, holder 2188818, 2026-10-20; /mnt/beegfs/ojerbi/ctxsweep-runs/ctxsweep_cs1_2188818_b75/*/fit_row.json):
# (family, S, B, slots, alloc_peak_gb, reserved_peak_gb, nonpt_gb, decode_alloc_peak_gb, host_peak_gib); max_gen 576
CS1_ROWS = (
    ("original", 16128, 16, 64, 21.497161728, 21.755854848, 0.600702976, 18.95066624, 32.863956451416016),
    ("cached", 16128, 16, 97, 22.60993024, 22.810722304, 0.600702976, 20.063434752, 32.8900260925293),
    ("original", 65280, 8, 64, 28.556770304, 29.746003968, 0.634257408, 18.181823488, 48.862632751464844),
    ("cached", 65280, 8, 114, 29.3996288, 30.50307584, 0.634257408, 19.024681984, 48.851234436035156),
    ("original", 65280, 32, 64, 33.672712704, 35.160850432, 0.835584, 23.369504256, 144.8959083557129),
    ("cached", 65280, 32, 114, 37.044081152, 38.155583488, 0.835584, 26.740872704, 144.84499740600586),
    ("cached", 65280, 32, 161, 40.212058624, 41.357934592, 0.835584, 29.908850176, 144.83872604370117),
    ("original", 130816, 4, 64, 38.454758912, 41.605398528, 0.634257408, 17.636417024, 48.866207122802734),
    ("cached", 130816, 4, 114, 38.876204544, 41.67041024, 0.634257408, 18.057862656, 48.833824157714844),
    ("original", 130816, 16, 64, 41.961658368, 43.541069824, 0.835584, 21.20457216, 144.89682388305664),
    ("cached", 130816, 16, 114, 43.647342592, 46.0849152, 0.835584, 22.890256384, 144.82362365722656),
    ("cached", 130816, 16, 161, 45.231347712, 47.666167808, 0.835584, 24.474261504, 144.94398498535156))


def cs1_fit_rows() -> List[Dict[str, Any]]:
    """CS1_ROWS in the fit_row.json format."""
    return [dict(family=f, S=S, B=B, slots=sl, alloc_peak_gb=a, reserved_peak_gb=r, nonpt_gb=n, decode_alloc_peak_gb=d,
                 host_peak_gib=h, max_gen=576) for f, S, B, sl, a, r, n, d, h in CS1_ROWS]


def pow2ceil(x: int) -> int:
    x = int(x)
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def kv_tensor_bytes(B: int, S: int, max_gen: int) -> int:
    """One pinned host tensor (one layer's K or V): B * (S + max_gen) * 2 heads * 128 * 2 B (cache_engine.py
    prefill_update)."""
    return int(B) * (int(S) + int(max_gen)) * EC.H_KV * EC.HEAD_D * 2


@dataclass(frozen=True)
class HostFit:
    base_gib: float = 24e9 / GIB             # B's host_fixed_gb (24 GB): the model's host copy at load, python, staging
    mode: str = "2x"
    source: str = "default: 2 x the nominal pinned KV until a row of this harness is measured (the lead's rule)"

    def kv_gib(self, B: int, S: int, max_gen: int) -> float:
        t = kv_tensor_bytes(B, S, max_gen)
        n = 2 * EC.N_LAYERS
        if self.mode == "pow2":
            return n * pow2ceil(t) / GIB
        if self.mode == "2x":
            return 2.0 * n * t / GIB
        if self.mode == "nominal":
            return n * t / GIB
        raise ValueError("host mode %r" % self.mode)

    def ids_gib(self, B: int, S: int) -> float:
        return 2.0 * int(B) * int(S) * 8 / GIB

    def host_gib(self, B: int, S: int, max_gen: int) -> float:
        return self.base_gib + self.kv_gib(B, S, max_gen) + self.ids_gib(B, S)

    def as_dict(self) -> Dict[str, Any]:
        return dict(base_gib=self.base_gib, mode=self.mode, source=self.source)


def fit_host_from_rows(rows: Sequence[Dict[str, Any]], prior: Optional[HostFit] = None) -> HostFit:
    """The 'pow2' model with its base from the MEASURED host peaks (rows with host_peak_gib, S, B, max_gen): base = the
    largest (peak - pow2 KV - ids) + 1 GiB. No such row: the prior (default '2x')."""
    prior = prior or HostFit()
    meas = [r for r in rows if r.get("host_peak_gib") is not None and r.get("max_gen") is not None]
    if not meas:
        return prior
    p2 = HostFit(base_gib=0.0, mode="pow2")
    two = HostFit(base_gib=0.0, mode="2x")
    res = [float(r["host_peak_gib"]) - p2.host_gib(int(r["B"]), int(r.get("S") or S0), int(r["max_gen"])) for r in meas]
    base = max(0.0, max(res)) + HOST_REFIT_SLACK_GIB
    worst2x = max(float(r["host_peak_gib"]) - two.host_gib(int(r["B"]), int(r.get("S") or S0), int(r["max_gen"])) - base
                  for r in meas)
    return HostFit(base_gib=base, mode="pow2",
                   source="pow2 from %d measured rows: base %.1f GiB (residual spread %.1f GiB); the 2x model's largest "
                          "under-read %.1f GiB (<= 0: 2x was conservative)" % (len(meas), base, max(res) - min(res), worst2x))


def mem_need_gib(host_gib: float) -> int:
    """The srun --mem (GiB) a predicted host peak needs: ceil10(1.08 x host + 6), at least 64."""
    return max(MEM_FLOOR_GIB, int(math.ceil((float(host_gib) * (1.0 + MEM_SLACK_FRAC) + MEM_SLACK_GIB) / 10.0)) * 10)


def host_bound(hfit: HostFit, S: int, max_gen: int, mem_gib: int, hi: int) -> int:
    """The largest B in [0, hi] whose --mem need fits mem_gib (host_gib is non-decreasing in B)."""
    lo, top = 0, int(hi)
    while lo < top:
        mid = (lo + top + 1) // 2
        if mem_need_gib(hfit.host_gib(mid, S, max_gen)) <= int(mem_gib):
            lo = mid
        else:
            top = mid - 1
    return lo


def pinned_summary(nbytes: Sequence[int]) -> Dict[str, Any]:
    """The pinned tensors' nominal size and their power-of-two size (what the caching host allocator reserves), GiB."""
    return dict(tensors=len(nbytes), nominal_gib=sum(int(x) for x in nbytes) / GIB,
                pow2_gib=sum(pow2ceil(int(x)) for x in nbytes) / GIB)


def cgroup_memory_files(text: str) -> Optional[Dict[str, Any]]:
    """The memory cgroup's usage and peak files from /proc/self/cgroup: v1 '<n>:memory:<path>' ->
    /sys/fs/cgroup/memory<path>/memory.usage_in_bytes, memory.max_usage_in_bytes; v2 '0::<path>' ->
    /sys/fs/cgroup<path>/memory.current, memory.peak. None when neither is listed."""
    v2 = None
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrls, path = parts
        if "memory" in ctrls.split(","):
            root = "/sys/fs/cgroup/memory" + path.rstrip("/")
            return dict(version=1, usage=root + "/memory.usage_in_bytes", peak=root + "/memory.max_usage_in_bytes")
        if hid == "0" and ctrls == "":
            v2 = path.rstrip("/")
    if v2 is not None:
        root = "/sys/fs/cgroup" + v2
        return dict(version=2, usage=root + "/memory.current", peak=root + "/memory.peak")
    return None


# ------------------------------------------------------------------------------------------------------------- time
PREFILL_MS_PER_TOKEN = ((16128, 0.168), (65280, 0.1894), (130816, 0.2140))   # MEASURED, run cs1 (module docstring)
FIRST_STEP_S, REWARM_S, OVERHEAD_MIN = 9.0, 6.0, 0.7
SPEC_DECODE_FACTOR = {1: 2.2, 2: 2.8, 3: 4.3}


def prefill_ms_per_token(S: int) -> float:
    pts = PREFILL_MS_PER_TOKEN
    if S <= pts[0][0]:
        return pts[0][1]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if S <= s1:
            return v0 + (v1 - v0) * (S - s0) / (s1 - s0)
    return pts[-1][1]


def step_ms(S: int, B: int, P: int = 0) -> float:
    """A steady shipped decode step (ms): 15 + 0.11 S / 1000 + B (0.45 + 0.0012 S / 1000), the per-request part x 0.8
    with a pool. cs1 p50: 16K x 16 24.7 (24.3), 64K x 8 26.7 (26.4), 64K x 32 39.7 (39.1), 128K x 4 31.0 (31.8),
    128K x 16 37.3 (39.1); old evidence 128K x 60 65 (65.8), 64K x 64 57.7 (56.0)."""
    return 15.0 + 0.11 * S / 1000.0 + B * (0.45 + 0.0012 * S / 1000.0) * (0.8 if P > 0 else 1.0)


def rewarm_s(S: int, B: int) -> float:
    """A re-warm step (warm-up path, CUDA-graph capture): about 6 s at every measured context and batch (cs1: decode
    time - 502 x the p50 step = 8 re-warms x 5.6-6.1 s + a first step of 8.4-9.8 s)."""
    return REWARM_S


def spec_factor(K: int) -> float:
    return SPEC_DECODE_FACTOR.get(int(K), 1.3 * int(K))


def est_minutes(S: int, B: int, cell: EC.Cell, N: int, n_rewarms: int) -> Dict[str, float]:
    """The planned GPU minutes of one cell (module docstring TIME): prefill, decode, the process overhead."""
    pre = B * S * prefill_ms_per_token(S) / 1000.0
    dec = FIRST_STEP_S + max(0, n_rewarms - 1) * rewarm_s(S, B) \
        + max(0, N - 1 - n_rewarms) * step_ms(S, B, cell.P) / 1000.0
    if cell.family not in BASELINE_FAMILIES:
        dec *= spec_factor(cell.K) if cell.family == "spec" else 4.0
    return dict(prefill_s=pre, decode_s=dec, minutes=OVERHEAD_MIN + (pre + dec) / 60.0)


# ------------------------------------------------------------------------------------------------------------- plan
def plan_ctx_cell(cell: EC.Cell, S: int, budget_gb: float, N: int, sfit: Optional[SMemFit] = None,
                  hfit: Optional[HostFit] = None, margin_gb: float = 1.0, mem_gib: int = MEM_MAX_GIB,
                  batch: Any = "max", b_cap: int = 4096) -> Dict[str, Any]:
    """One cell at context S: the bounds (module docstring), the batch (explicit or 'max'), the predictions (GPU, host,
    --mem need, minutes), the documents. Refuses (ValueError naming the rule) when the cell cannot run."""
    sfit = sfit or prior_fit()
    hfit = hfit or HostFit()
    S, N, mem_gib = int(S), int(N), int(mem_gib)
    if not MEM_FLOOR_GIB <= mem_gib <= MEM_MAX_GIB:
        raise ValueError("--mem %d GiB outside [%d, %d] (the holder has 480G)" % (mem_gib, MEM_FLOOR_GIB, MEM_MAX_GIB))
    if N + cell.K + 1 > EC.CIS_HEADROOM:
        raise ValueError("F3: N %d + K %d + 1 > %d (total_cis has S + %d rows)" % (N, cell.K, EC.CIS_HEADROOM,
                                                                                  EC.CIS_HEADROOM))
    pos = S + N + 2 * cell.K + 1
    if pos > ROPE_POSITIONS:
        raise ValueError("rope: S %d + N %d + 2K %d + 1 = %d positions > the cos/sin cache's %d (max_length 128 x 1024 "
                         "+ 1024, nosa_llama.py _set_cos_sin_cache)" % (S, N, 2 * cell.K, pos, ROPE_POSITIONS))
    fit = sfit.at(S)
    spec = cell.family == "spec"
    ph = sfit.phases(cell, S)                    # without a measured decode phase: the 4.1 model (= EC.plan_cell's)
    fixed, per_req = ph["prefill"]
    cap_alloc = (budget_gb - margin_gb - fit.nonpt_gb) / (1.0 + fit.reserve_frac)
    room = cap_alloc - fixed
    if room < per_req:
        raise ValueError("prefill: at S %d the fixed part A(S) %.2f GB (A16 %.2f + the S terms %.2f: the weights and the "
                         "one-request prefill transient)%s leaves %.2f GB for the requests, < one request's %.3f GB "
                         "(budget %.1f, margin %.1f)" % (S, fit.A, sfit.A16, fit.A - sfit.A16,
                                                         " + spec %.2f" % fit.spec_fixed_gb
                                                         if spec and ph["decode"] is None else "",
                                                         room, per_req, budget_gb, margin_gb))
    base = EC.plan_cell(cell, budget_gb, N, fit, margin_gb, S, host_budget_gb=1e12, b_cap=b_cap)
    mg = int(base["max_gen"])
    bounds = dict(base["bounds"])
    lines = [ph["prefill"]] + ([ph["decode"]] if ph["decode"] is not None else [])
    if ph["decode"] is not None:                 # measured context: the larger of the two phases must fit
        bounds["memory"] = min(int(math.floor((cap_alloc - f) / p)) if cap_alloc > f else 0 for f, p in lines)
    bounds["host_memory"] = host_bound(hfit, S, mg, mem_gib, int(b_cap) + 1)    # > b_cap when the host never binds
    order = list(bounds)
    top = min(bounds, key=lambda k: (bounds[k], order.index(k)))
    B_max = int(bounds[top])
    if B_max < 1:
        raise ValueError("cell %s at S %d: no batch fits (bound_by %s, bounds %s)" % (cell.name, S, top, bounds))
    if batch is None or batch == "max":
        B, why = B_max, top
    else:
        B = int(batch)
        if B < 1:
            raise ValueError("batch %d < 1" % B)
        if B > B_max:
            raise ValueError("batch %d > the largest admitted %d (bound_by %s, bounds %s)" % (B, B_max, top, bounds))
        why = "requested"
    ctx = CONTEXTS.get(S)
    docs = None
    if ctx is not None:
        d = EC.doc_rows(B, ctx.n_docs)
        docs = dict(n_docs=ctx.n_docs, distinct=d["distinct"], reused=d["reused"], repeats=d["repeats"])
    host_pred = hfit.host_gib(B, S, mg)
    rw = rewarm_schedule(S, N - 1, 1)
    rw_round = rewarm_schedule(S, N - 1, cell.lookahead) if cell.lookahead > 1 else rw
    stages = EC.stage_breakdown(fit, cell, B)
    stages["of_which_S_terms_in_A"] = fit.A - sfit.A16
    peaks = [f + p * B for f, p in lines]
    plan = dict(base)
    plan.pop("predicted_host_gb", None)
    plan.update(
        B=B, bound_by=why, B_max=B_max, B_max_bound_by=top, bounds=bounds, batch_request=str(batch),
        predicted_alloc_gb=fit.alloc_gb(cell, B) if ph["decode"] is None else sfit.alloc_gb(cell, S, B),
        predicted_used_gb=fit.used_gb(cell, B) if ph["decode"] is None else sfit.used_gb(cell, S, B), stages=stages,
        phases=dict(prefill=list(ph["prefill"]), decode=list(ph["decode"]) if ph["decode"] is not None else None,
                    binding=("prefill", "decode")[peaks.index(max(peaks))]),
        fit=fit.as_dict(), sfit=sfit.as_dict(), calibrated_batch_sizes=sfit.calibrated(S),
        host=dict(mode=hfit.mode, base_gib=hfit.base_gib, predicted_gib=host_pred, need_gib=mem_need_gib(host_pred),
                  mem_gib=mem_gib, kv_nominal_gib=HostFit(base_gib=0.0, mode="nominal").kv_gib(B, S, mg),
                  kv_pow2_gib=HostFit(base_gib=0.0, mode="pow2").kv_gib(B, S, mg), source=hfit.source),
        docs=docs, rope_positions_needed=pos, f1_window=f1_window(S),
        rewarms_shipped=len(rw), rewarms_round=len(rw_round),
        est=est_minutes(S, B, cell, N, len(rw)))
    return plan


def parse_cell_list(text: str) -> List[Dict[str, Any]]:
    """The chain's cell list: '<ctx> <cell> <batch|max> <mem GiB>' per line, '#' comments, '# tier <name>' starts a
    tier (the entries below it, until the next tier line)."""
    out, tier = [], "1"
    for no, raw in enumerate(text.splitlines(), 1):
        st = raw.strip()
        if st.lower().startswith("# tier"):
            tier = st[6:].strip().split()[0] if st[6:].strip() else tier
            continue
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        f = s.split()
        if len(f) != 4:
            raise ValueError("line %d: expected '<ctx> <cell> <batch|max> <mem GiB>', got %r" % (no, s))
        S = int(f[0])
        if S not in CONTEXTS:
            raise ValueError("line %d: context %d is not one of %s" % (no, S, sorted(CONTEXTS)))
        EC.parse_cell(f[1])
        if f[2] == "max":
            batch: Any = "max"
        else:
            batch = int(f[2])
            if batch < 1:
                raise ValueError("line %d: batch %d" % (no, batch))
        mem = int(f[3])
        if not MEM_FLOOR_GIB <= mem <= MEM_MAX_GIB:
            raise ValueError("line %d: --mem %d GiB outside [%d, %d]" % (no, mem, MEM_FLOOR_GIB, MEM_MAX_GIB))
        out.append(dict(line=no, tier=tier, S=S, cell=f[1], batch=batch, mem_gib=mem))
    return out


def sweep_plan(entries: Sequence[Dict[str, Any]], budget_gb: float, N: int, sfit: Optional[SMemFit] = None,
               hfit: Optional[HostFit] = None, margin_gb: float = 1.0, b_cap: int = 4096) -> List[Dict[str, Any]]:
    """Every entry of a cell list planned: dict(entry, plan) or dict(entry, error)."""
    sfit = sfit or prior_fit()
    out = []
    for e in entries:
        try:
            p = plan_ctx_cell(EC.parse_cell(e["cell"]), e["S"], budget_gb, N, sfit, hfit, margin_gb, e["mem_gib"],
                              e["batch"], b_cap)
            out.append(dict(entry=dict(e), plan=p))
        except ValueError as err:
            out.append(dict(entry=dict(e), error=str(err)))
    return out


def tier_minutes(plans: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for p in plans:
        if "plan" in p:
            t = p["entry"].get("tier", "1")
            tot[t] = tot.get(t, 0.0) + p["plan"]["est"]["minutes"]
    return tot


# ---------------------------------------------------------------------------------------------------------- latency
def _dist(xs: Sequence[float]) -> Dict[str, Any]:
    v = [float(x) for x in xs]
    return dict(n=len(v), mean=EC.mean(v), p50=EC.pctl(v, 50), p95=EC.pctl(v, 95), max=max(v) if v else None)


def latency_metrics(B: int, gen_tokens: int, first_token_ms: float, prefill_done_ms: Sequence[float],
                    unit_tokens: Sequence[float], unit_ms: Sequence[float], itl_ms: Optional[Sequence[float]] = None,
                    e2e_event_ms: Optional[float] = None) -> Dict[str, Any]:
    """One cell's latency on the CUDA-event clock (module docstring LATENCY). itl_ms: the shipped decode's intervals
    between consecutive step-end events (the first from the prefill end); None for rounds (spec)."""
    if itl_ms:
        tok = _dist(itl_ms)
        basis = "steps: consecutive step-end events, one token per request per step"
    else:
        per = [m / t for t, m in zip(unit_tokens, unit_ms) if t > 0]
        tok = _dist(per)
        tt = sum(float(t) for t in unit_tokens)
        tok["mean"] = sum(float(m) for m in unit_ms) / tt if tt > 0 else None
        basis = "rounds: each round's GPU time / its committed tokens (host gaps excluded)"
    out = dict(clock="CUDA events from the prefill start event",
               ttft_s=first_token_ms / 1000.0,
               ttft_note="static batch: every request gets its first token when the batch prefill ends",
               own_prefill_s=_dist([x / 1000.0 for x in prefill_done_ms]),
               token_latency_ms=tok, token_latency_basis=basis.split(":")[0], token_latency_detail=basis,
               unit_ms=_dist(unit_ms), e2e_s=None, tpot_ms=None, e2e_tok_s=None)
    if e2e_event_ms is not None and e2e_event_ms > 0:
        out["e2e_s"] = e2e_event_ms / 1000.0
        out["e2e_tok_s"] = B * gen_tokens / (e2e_event_ms / 1000.0)
        if gen_tokens > 1:
            out["tpot_ms"] = (e2e_event_ms - first_token_ms) / (gen_tokens - 1)
    return out


# ----------------------------------------------------------------------------------------------------------- report
def _batch_of(r: Dict[str, Any]) -> int:
    m = r.get("metrics") or {}
    return int(m.get("batch") or (r.get("plan") or {}).get("B") or 0)


def _cell_stats(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    e2e = [x["metrics"]["e2e_tok_s"] for x in rs if x["metrics"].get("e2e_tok_s") is not None]
    dec = [x["metrics"]["decode_tok_s_gpu_units"] for x in rs if x["metrics"].get("decode_tok_s_gpu_units") is not None]
    if len(dec) >= 2:
        s = EC.reps_summary(dec)
        ci = [s["ci_lo"], s["ci_hi"]]
    elif len(rs) == 1:
        ci = list(rs[0]["metrics"].get("decode_tok_s_ci") or [None, None])
    else:
        ci = [None, None]
    lat = [x.get("latency") or {} for x in rs]
    B = _batch_of(rs[0])
    p50 = [x["token_latency_ms"]["p50"] for x in lat if (x.get("token_latency_ms") or {}).get("p50")]
    return dict(cell=rs[0]["cell"], family=rs[0]["family"], B=B, reps=len(rs), e2e_tok_s=EC.mean(e2e),
                decode_tok_s_gpu_units=EC.mean(dec), decode_ci=ci,
                steady_tok_s=EC.mean([B * 1000.0 / v for v in p50]),
                ttft_s=EC.mean([x["ttft_s"] for x in lat if x.get("ttft_s") is not None]),
                token_p50_ms=EC.mean(p50),
                token_p95_ms=EC.mean([x["token_latency_ms"]["p95"] for x in lat if x.get("token_latency_ms")]),
                used_peak_est_gb=max(x["memory"]["used_peak_est_gb"] for x in rs))


STEADY_THRESHOLD = 0.02


def verdict_steady(cand: Dict[str, Any], base: Optional[Dict[str, Any]]) -> str:
    """The steady decode rate (B x 1000 / the p50 per-token latency: the re-warm steps excluded; a point estimate):
    'helps' / 'hurts' beyond +/- 2 %, else 'within 2 %'."""
    if base is None:
        return "no baseline at this batch"
    if not cand.get("steady_tok_s") or not base.get("steady_tok_s"):
        return "no latency record"
    r = cand["steady_tok_s"] / base["steady_tok_s"] - 1.0
    return "helps" if r > STEADY_THRESHOLD else ("hurts" if r < -STEADY_THRESHOLD else "within 2 %")


def verdict(cand: Dict[str, Any], base: Optional[Dict[str, Any]]) -> str:
    """'helps' / 'hurts': the e2e committed tokens / s moves that way AND the decode intervals (GPU units: the
    within-run bootstrap for one repetition, the t interval over repetitions) do not overlap; else 'inconclusive'."""
    if base is None:
        return "no baseline at this batch"
    if cand["e2e_tok_s"] is None or base["e2e_tok_s"] is None:
        return "inconclusive"
    (lc, hc), (lb, hb) = cand["decode_ci"], base["decode_ci"]
    if None in (lc, hc, lb, hb):
        return "inconclusive"
    if cand["e2e_tok_s"] > base["e2e_tok_s"] and lc > hb:
        return "helps"
    if cand["e2e_tok_s"] < base["e2e_tok_s"] and hc < lb:
        return "hurts"
    return "inconclusive"


def matched_batches(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per batch size: every VALID cell, the best baseline cell (by the e2e mean), each candidate's speedups and
    verdict against it."""
    valid = [r for r in results if not r.get("error") and (r.get("memory") or {}).get("verdict") == "VALID"]
    by_b: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for r in valid:
        by_b.setdefault(_batch_of(r), {}).setdefault(r["cell"], []).append(r)
    out = {}
    for B, cells in sorted(by_b.items()):
        st = {name: _cell_stats(rs) for name, rs in cells.items()}
        bases = [s for s in st.values() if s["family"] in BASELINE_FAMILIES and s["e2e_tok_s"] is not None]
        base = max(bases, key=lambda s: s["e2e_tok_s"]) if bases else None
        cands = {}
        for name, s in st.items():
            if s["family"] in BASELINE_FAMILIES:
                continue
            sp = {}
            if base is not None:
                for k in ("e2e_tok_s", "decode_tok_s_gpu_units", "steady_tok_s"):
                    if s[k] and base[k]:
                        sp[k] = s[k] / base[k] - 1.0
            cands[name] = dict(speedup=sp, verdict=verdict(s, base), verdict_steady=verdict_steady(s, base))
        out[str(B)] = dict(cells=st, baseline=base["cell"] if base else None, candidates=cands)
    return out


def sweep_report(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per context: the cells (a cell at each batch is its own row, '<cell>@<B>'), equal memory (the best VALID cell
    of each family over every batch and capacity; candidates vs the best baseline), the matched-batch verdicts."""
    by_s: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_s.setdefault(int((r.get("plan") or {}).get("S") or S0), []).append(r)
    out: Dict[str, Any] = dict(contexts={}, verdicts=[])
    for S, rs in sorted(by_s.items()):
        lab = [dict(r, cell="%s@%d" % (r["cell"], _batch_of(r))) for r in rs]
        blk: Dict[str, Any] = {}
        try:
            cmp_ = EC.comparison(lab)
        except ValueError as e:
            blk["error"] = str(e)
            out["contexts"][str(S)] = blk
            continue
        fams = sorted({r["family"] for r in lab})
        best = {f: EC.select_best(lab, f) for f in fams}
        bb = [b for f, b in best.items() if f in BASELINE_FAMILIES and b]
        top = max(bb, key=lambda b: b["mean"]) if bb else None
        eq = dict(best=best, best_baseline=top["cell"] if top else None, speedup_e2e={})
        for f, b in best.items():
            if f not in BASELINE_FAMILIES and b and top:
                eq["speedup_e2e"]["%s_vs_best_baseline" % f] = b["mean"] / top["mean"] - 1.0
                out["verdicts"].append(dict(S=S, kind="equal_memory", cell=b["cell"], vs=top["cell"],
                                            speedup_e2e=b["mean"] / top["mean"] - 1.0,
                                            verdict="helps" if b["mean"] > top["mean"] else "hurts (point estimate)"))
        mb = matched_batches(rs)
        for B, blkb in mb.items():
            for name, c in blkb["candidates"].items():
                out["verdicts"].append(dict(S=S, kind="matched_batch", B=int(B), cell=name, vs=blkb["baseline"],
                                            speedup=c["speedup"], verdict=c["verdict"],
                                            verdict_steady=c["verdict_steady"]))
        blk.update(cells=cmp_["cells"], equal_memory=eq, matched=mb)
        out["contexts"][str(S)] = blk
    return out
