"""The PAIRED TICK: verify(t) + speculative draft(t+1) in ONE forward per layer
(retroinfer-eval fork, 2026-09-20; spec
docs/superpowers/specs/2026-09-20-paired-tick.md sections 1-3, stage E2 of
its ladder: the RESIDENT tick, no per-layer prefetch yet).

Modules:
  core.py   pure torch, CPU-runnable: the per-request state machine
            (draft / accept / reject / restart), the lockstep tick outcome,
            the per-(layer, head, request) byte and divergence accounting,
            the [V | S] row layout, the per-row visible-row rules, the rows
            bias, the provisional write + NaN poison, the S == 1 tail-write
            twin (G6) and the E4 readiness hook (G4).
  tick.py   the model-level per-layer body and the driver methods (GPU):
            one qkv GEMM over 2B rows, per-position scoring through the
            shipped kernels, the V row's engine update = the shipped decode
            body, the S row's provisional write, ONE rows-attention call,
            wo / FFN over 2B rows; the restart / seed forward over S rows.
  twin.py   the reference twin for gate G3 (U = 1, V rows only) and the
            in-situ per-layer comparator that names the differing term.
  DESIGN.md every assumption about the engine, with file:line pointers, and
            what stage E4 needs from the engine.

This package imports nothing GPU-specific at module level: ``core`` and the
module bodies of ``tick`` / ``twin`` import only torch and the fork's pure
modules (verify/rows_attention.py, verify/tail_write.py, spec_loop.py,
verify_trace.py); the CUDA-extension names are bound lazily inside the GPU
functions (tick._gpu). tests/test_paired_core.py loads it on CPU.
"""
