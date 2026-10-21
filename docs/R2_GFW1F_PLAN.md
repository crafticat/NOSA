# R2 GFW cell gfw1f: the production transport's built-in warm-up (plan, REPORT only)

Status: SOURCE ONLY, not run. Registered 2026-10-21 (written 2026-10-21T08:50:55Z, `date -u`), before any gfw1f run.
Branch `r2-gfw-warm`, worktree `/mnt/beegfs/ojerbi/nosi-r2-gfw-warm`.

## Basis (gfw1e)

- Record: `/mnt/beegfs/ojerbi/r2-runs/gfw/gfw_gfw1e_2190876/summary.json` (holder 2190876, source 372fe55).
- Protocol: GFW1's captured protocol. There is no submission wait. 64 launches go behind the spin. Each arm runs in a
  fresh child process.
- Arm eager (`CUDA_MODULE_LOADING=EAGER`): it stalled. err 1, spin 7255.1 ms, enqueue 9.182 ms.
- Arm warm (lazy loading). Before the spin, the gate ran one `add_` on the main stream, then one float32 `index_copy_`
  and one int64 `index_put_` on the scatter stream. This arm passed: err 0, spin 142.7 ms, enqueue 0.894 ms.
- Conclusion of gfw1e: the cause is the first call of those operations behind the spin. Module loading is not the
  cause.

## Change under test (this branch)

- `nosi/nosi/r2_transport_gilfree.py`: `GilfreeTransport(..., warm_up=True)` is the default. The constructor calls
  `warm_up()` once, before the feeder thread starts and before any issue. So it runs before the first tick, in the
  eager tick and in the captured tick.
- The operations are the gfw1e list, on tiny tensors, on the production device:
  - `add_` on the constructing thread's current stream.
  - `index_copy_` on the scatter stream. It writes into a tensor of the store's dtype, with an int64 index. This is
    the scatter of the C++ worker and of packer v2.
  - `index_put_` on the scatter stream. It writes int64 values into an int64 tensor, with an int64 index. This is the
    board publication.
  - Then a device synchronize.
- On the CPU, the same operations run on CPU tensors (no stream, no synchronize).
- A second call is refused, and so is a call after the first issue.
- `warm_up=False` is a NEGATIVE CONTROL flag. Only tests and gfw1f's control arm use it.
- The rule of 9508109 does not change: the host waits for the worker's submission in both tick modes
  (`CudaDeviceOps._submitted`).
- CPU test: `test_r2_transport_gilfree.py::test_GT12_the_warm_up_runs_once_before_the_first_submission`. It runs for
  two workers (cpp, python) and two tick modes (eager, captured). Its negative control (`warm_up=False`) must be
  detected.

## Cell gfw1f

Step: `benchmarks/Efficiency/r2_gfw_step.sh --nosi <sha> <holder> <run id> gfw1f`. ROOT is this worktree.
GFW_ready stays MUST. Every other line is REPORT.

| arm | child process | transport warm-up | gate warm-up | submission wait | expected |
| --- | --- | --- | --- | --- | --- |
| A `builtin` | fresh, lazy loading | on (production default) | off | none | PASS |
| B `nowarm` | fresh, lazy loading | off (`warm_up=False`) | off | none | STALL (the control) |

Both arms run GFW1's captured protocol: 64 launches behind the spin, `--spin-cycles 200000000`, max_spin 2^24.

## REGISTERED GFW1F (frozen before the run; the step script holds the same block)

- PREDICTION, arm A: it passes. err 0, 10 ms < spin < 1000 ms, every launch ran, the rows are the host truth, the T4
  event was pending at issue, and the transport's `warm_calls` is 1.
- PREDICTION, arm B: it stalls as the control. err 1 (the spin ends at its bound), and `warm_calls` is 0.
- FALSIFIER: arm A ends with err 1, or arm A's spin is 1000 ms or more.
- CONTROL_BLIND: arm A passes and arm B also passes. The stall did not reproduce without the warm-up, so arm A's pass
  does not show that the warm-up is what prevents the stall.
- INVALID: any other result. Examples: an arm is missing, a child failed, a child did not run in lazy loading, or
  `warm_calls` is not as built.
- Verdict line: `R_GFW1f` (HELD / FALSIFIED / CONTROL_BLIND / INVALID). The arm lines are `R_GFW1f_builtin` and
  `R_GFW1f_nowarm`.

## Other changes in the gate (needed to run on this branch)

- This branch merges C's 9508109 line. That line carries 74c7d9b, which gives `board_wait_kernel` a 7th argument
  (`spins_ptr`, the spin-round counter). The gate's `gfw1()` and GTG1 called the kernel with 6 arguments, so they
  now pass a one-cell int64 counter. `gfw1()` also reports it as `spin_rounds`.
- `gfw1(..., tp_warm_up=False)` is the default. So the cells gfw1, gfw1d and gfw1e keep the protocol they were
  registered with (no transport warm-up). Only gfw1f sets it per arm.
- gfw2 and GFW_T construct the transport with its production default, so they now include the warm-up. It runs once
  at construction, before any measured release.

## Open doubts (not claims)

- gfw1e warmed `add_` on the main stream and the scatter operations together, so it does not show which of them
  matters. The transport can warm only its own operations and the stand-in `add_`. In production, the main stream's
  kernels behind the spin are the model's kernels, and the transport cannot warm them. The hypothesis is that the
  worker's cold scatter kernels cause the deadlock, and that a cold main-stream kernel only extends the visible block.
  gfw1f does not test this split.
- The warm-up uses a small 2-D destination. If the large production store selects a different kernel variant of
  `index_copy_` (for example, a different index width or vector width), that variant can still be cold. This is not
  checked; gfw1f uses the gate's small store (float32), not the SG5 store (bfloat16).
