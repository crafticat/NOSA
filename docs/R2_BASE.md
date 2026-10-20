# r2base: the captured R2 decode step against the shipped decode (throughput cell)

Branch `feature/nosi-r2-base` (from `feature/nosi-r2-m2-capture` 09e683a, the commit the
capture gate pins). Status: SOURCE ONLY. Nothing here comes from a GPU run. It runs AFTER
the capture gate passes.

Code: `benchmarks/Efficiency/r2_base_cell.py`; step `benchmarks/Efficiency/r2_base_step.sh`;
CPU tests `tests/test_r2_base.py` (the fake world moved to `tests/r2_fake_world.py`, shared
with `tests/test_r2_capture_step.py`).

## The cell

- Workload: the PG19 94-document file at 16128 tokens (`pg19-test-c16k`, sha256 pinned). Request
  b reads row b % 94, so B 128 and 219 repeat documents. The agreement is also reported over the
  first min(B, 94) requests, which see each document once.
- B 64, 128, 219 (one step each); N 512 decode steps; one model, one process per B.
- SH, the shipped decode:
  - `Llama.decode_inference` on the offload engine (`cache_engine.InfLLMv2Cache`); greedy,
    free-running.
  - The host rows are capped at S + N + 64 by `max_gen_len`, as the sustained harness does. The
    shipped default S + 8192 would need about 174 GB of pinned memory at B 219.
  - It is RE-WARMED where its warm-up-sized buffers require it: a new 64-token block (findings
    F1), or a new 128-column stage-1 score width. The rule is that of the sustained harness
    (`nosi-sustained` a1c8a24, `benchmarks/Efficiency/sustained_core.py:217-235`, ported as
    `rewarm_need`).
  - Its tokens are the reference.
- R2, the captured R2 step:
  - `build_ragged` (per-request prefill; widths preallocated from the horizon, so no re-warm) and
    `r2_capture.CaptureBuffers`.
  - TEACHER-FORCED on SH's input tokens: 2 eager steps of `decode_step_fixed`, then one captured
    step replayed 510 times.
  - Inputs and positions advance on the device; no host sync per step.

## Metrics

| Metric | Definition |
|---|---|
| steady per-step ms | median CUDA-event period of the steady steps (SH: not a re-warm step; R2: replays after the capture step) |
| steady decode tok/s | B / steady step |
| per-request rate | 1 / steady step (tok/s of one request) |
| effective tok/s | B x N / sum of all periods (SH's re-warms and R2's warm-up and capture included) |
| GPU peak | `max_memory_allocated` / `max_memory_reserved` over the arm, prefill included |
| teacher-forced agreement | R2's argmax at step t == SH's token after step t, over (request, step) |
| free-running first divergence | per request, the first teacher-forced disagreement |
| prefill first tokens | REPORT: SH's prefill logits GEMM runs at M = B, R2's per-request prefill at M = 1 |

Why the first teacher-forced disagreement is the free-running first divergence: until it, both
runs have the same inputs, so the teacher-forced run is the free-running run.

MUST:

- SH-RUN and R2-RUN: N steps each;
- R2-ERR: no refusal bit;
- R2-PTR: no state `data_ptr` moves across the replays.

Exit: the number of failed MUST gates; 20 refused; 30 an exception outside an arm.

## Registered predictions (made before any run; the cell prints a verdict for each)

- P1 EXACTNESS: teacher-forced agreement 1.0 at every B; no request diverges in 512 steps.
  - Basis:
    - G0: the R2 path is bitwise the shipped decode at equal advance;
    - k1 G7-order: `topk(sorted=False)` keeps the same raw order on rows widened with `-inf`
      (128 of 128 rows at k 33 and 64), so the wider preallocated pooling rows do not reorder the
      selection;
    - k1 G2 / G3: stage 1 with `seqused_k` and a padded table is bitwise the shipped call;
    - F4 is reproduced;
    - F2's stale compressed cis affects only blocks inside the forced local window while
      N < 1024.
  - Falsifier: any disagreement.
- P2 RE-WARMS: the shipped arm re-warms exactly 9 times. The steps are 0 (first); 64, 128, 192,
  256, 320, 384, 448 (a new block); and 288 (the score width 1024 -> 1152 at the 1025th compressed
  key). R2 warms up once (2 eager steps + 1 capture).
  - Falsifier: any other list (CPU test B1 computes it).
- P3 STEADY STEP TIME: R2 steady per-step / SH steady per-step <= 0.8 at B 64 and <= 0.9 at
  B 219 (REPORT at B 128).
  - Basis: the shipped decode is eager, about 1300 launches per step. Its steady step time at 16K
    has a large fixed part: about 167 ms at B 64 against about 288 ms at B 304 (eqmem original,
    1054.9 tok/s) puts the intercept near 135 ms. The capture removes most of that fixed part.
  - The R2 step adds work per step: the twin over all B requests, wider pooling rows, the device
    write-back. So the gain shrinks as B grows.
  - Falsifier: a ratio above the limit.
- P4 EFFECTIVE THROUGHPUT at B 64: R2 effective tok/s >= 1.5 x SH effective tok/s.
  - Basis: 9 re-warms at about 6 s each (D's measurement) add about 54 s to about 512 x 0.167 =
    85 s of steady decode. R2 pays about 2 eager steps + 1 capture (seconds).
  - Falsifier: a ratio below 1.5.
- P5 GPU PEAK: R2 peak allocated <= 1.15 x SH peak allocated at every B (the twin tables plus the
  graph pool).
  - Falsifier: a ratio above 1.15.

## Commands

CPU (inside a CPU job):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-base python -m pytest -q tests/test_r2_base.py tests/test_r2_capture_step.py tests/test_r2_capture.py

GPU (after the capture gate passes; DRYRUN on the login node first; one cell per step):

    DRYRUN=1 bash benchmarks/Efficiency/r2_base_step.sh --nosi <sha> <holder job> <run id> b64
    bash benchmarks/Efficiency/r2_base_step.sh --nosi <sha> <holder job> <run id> b64      # then b128, b219

| Cell | STEP_MIN | --mem | Pinned host KV per arm |
|---|---|---|---|
| b64 | 25 | 100G | about 35 GB |
| b128 | 35 | 160G | about 70 GB |
| b219 | 50 | 220G | about 120 GB |

The `--mem` must fit the holder's allocation. Prefill is about 2.5 s per 16K request and per arm,
so b219 spends about 18 minutes in its two prefills.

The GPU uuid comes from `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<job>.gpu_uuid`; claim
directory `/mnt/beegfs/ojerbi/r2-runs/base/base_active`.

## Risks

- B 219 on the shipped offload engine at 16K is not measured yet: GPU memory (about 30 GB slot
  cache + 16 GB weights) and host memory.
- The shipped arm reads its compressed cis past the table after a re-warm (F2). That is the
  shipped behaviour; the sustained runs survived it.
- The capture-safe step is proven bitwise on the CPU and by the capture gate at B 8; B 64 .. 219
  are new shapes.
