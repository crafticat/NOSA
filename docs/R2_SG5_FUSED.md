# SG5 with the fused twin step (`--twin fused`)

Branch `feature/nosi-r2-m2-sg5-fused`, worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-sg5-fused`.

**Base.** The base is 2e77d13 (SG5 p3) plus a merge of C's 674ab1f. `nosi/nosi/r2_twin_fused.py`
is the file of 586f49a, unchanged. Its design, proofs and GPU gates are in
`/mnt/beegfs/ojerbi/nosi-r2-fused2/docs/R2_FUSED.md`.

## 1. What changes (every touched line is marked `[twin-variant]`)

**`nosi/nosi/r2_tick.py`** (14 marked lines):
- `TWIN_VARIANTS = ("torch", "fused")`.
- The class docstring: 3 lines.
- `TickEngine.__init__(..., twin: str = "torch")`. The default is unchanged behaviour.
- After the `mean_fn` loop: a variant check, then `self.twin_variant`. With "fused", every
  twin's `step_unchecked` and `advance_unchecked` are bound on the instance to `r2_twin_fused`
  (`attach`). A mean other than `fixed_order_mean` is refused (TickError).
- `clone()`: `twin=self.twin_variant` in the arguments.

The call site `ctr = twin.advance_unchecked(keys, cis_b, n_dev)` in `_layer` is NOT changed: the
binding is on the twin instances. Test T16 pins this line. `from_r2` passes the argument through
`**kw`.

**`benchmarks/Efficiency/r2_tick_sg5.py`** (3 marked lines):
- `--twin {torch,fused}`, default torch.
- `make_engine` calls `TickEngine.from_r2(..., twin=a.twin)`. Every SG5 arm builds its engine
  there.
- `meta["twin"] = a.twin`. The value is also in `meta["args"]`.

**Not changed:** `r2_tick_sg5_step.sh` (its ROOT and its T19 test belong to the lead's merge). A
fused run needs one more argument in its ARGV, `--twin fused`, for example through an `SG5_TWIN`
variable in the merged script.

**Tests** (`benchmarks/Efficiency/tests_cpu/test_r2_twin_fused.py`):
- F1, F2 and F5 of 586f49a: the kernel against the torch twin, the planted faults, the order
  probe, and the bfloat16 proof.
- S1: the variant at construction, at clone and on refusal; SG5's `--twin` reaches `make_engine`
  and the meta.
- S2: ONE SG5-shaped round with the Triton interpreter.
  - The round: free-running greedy, K 3, LAMBDA 7, draft_drop 0.5, ResidencySC C 12 / X 24 /
    R_ver 4, the adversarial transport, B 4.
  - The fused twin must equal the torch twin bitwise: the rows of every tick, the argmax of every
    tick, the twin counters of every tick, the committed tokens, the decision counts and every
    twin table at the end.
  - PLANT 2 in the fused path must be effective (fused differs from fused + plant) and caught
    (torch differs from fused + plant).

## 2. Registered predictions: SG5 with `--twin fused`, B 64, C 96 / X 32 / R_ver 16, deltas to p3

**Basis.**

*Op counts.* The tick's twins use `fixed_order_mean`. With it, one `step_unchecked` is about 190
small kernels:
- 3 clones, 9 due / refusal ops, 2 overflow ops;
- the key window gather and the 65-kernel fixed-order mean;
- 4 kernels for each of the three masked writes;
- the cis window and its 65-kernel mean;
- the counters.

Per layer, `advance_unchecked` runs T steps plus the counter stacks: about 2 + 192 T kernels. The
tick spends about 6,200 twin kernels (T 1, draft ticks) or 18,500 (T 3, verify ticks) over 32
layers.

*Cost per kernel.* At 2B = 128 banks, each kernel takes 2.0-2.6 us of device time. An eager torch
op costs 6-8 us of host time (the TC gate's eager tick: about 106 ms of host per tick at B 8).

*The fused tick.* Each layer keeps 6 ops of the `twin` span (the zeros, the two scatters, the
`n_used` gather) plus ONE kernel launch:
- device: 0.2-0.6 ms per tick;
- host: about 32 x 8 ops at 7 us, plus 32 launches at 20-40 us = 2-3 ms per tick.

| prediction (per tick, p3's TickTimer categories) | draft tick (T 1) | verify tick (T 3) |
|---|---|---|
| PF1: p3's `twin` span (the torch twin) | 12-50 ms | 37-150 ms |
| PF2: the fused `twin` span | <= 4 ms | <= 4 ms |
| PF3: T_tick(fused) - T_tick(p3) | -(p3 twin span - fused twin span), within 15 % | same |

For PF1, the low end of each range is the device-bound case and the high end is the host-bound
case (the eager tick).

- PF4: the twin's share of p3's T_tick is 20-35 % on the round's mix of ticks. With p3's record,
  the expected drop is then 0.20-0.35 x T_tick.
- PF5: the decisions do not change: r (committed tokens per request-tick), the decision reads,
  the rollbacks and the publications equal those of a `--twin torch` run on the same documents.
  The CPU claim is S2; on the GPU, M-T's TGF-EQ. So r x T_shipped / T_tick improves by the factor
  T_tick(p3) / T_tick(fused).
- **Falsifiers:**
  - the fused `twin` span is above 4 ms;
  - |measured drop - (p3 twin span - fused twin span)| is above 15 % of the predicted drop;
  - any change in r or in the decision counts;
  - an SG5 MUST gate failing (the twins' counters against the host mirrors).

When p3's record lands, PF1 and PF4 get p3's measured `twin` span and T_tick, and PF3 becomes a
number in ms per tick.
