# R2 step S6: the shared X pool layout (opt-in)

Branch `r2-xpool`, from `nosi-r2-sg5-p6` `cf35536` (B2 `r2_own2.py` is in it: `git merge-base --is-ancestor c675779
cf35536` is true, so no merge). Status: SOURCE ONLY, NOT RUN. No CPU test and no GPU step ran on this branch.

Sources:
- the audit `paper-notes/recon-row-1b-alloc.md` (Q1-Q4);
- `kstep/ownership.py:47-50`: "the reduced B * H * C + X is step S6, not approved".

## 1. The problem

The store allocates `(B, (C + X + 4) * 64, H, D)` per layer (`r2_tick.py` `from_r2`; D4, `r2_residency.py:11-14`).
K5 lets only X extras per layer be in use, over all B x H streams (`ownership.py:409-427`). So 127/128 of the X
bytes stay zero at B 64, H 2. At C 96, X 64 the store is 22.10 GB at B 64 (345.28 MB per request).

## 2. The design

### 2.1 The layout

`nosi/nosi/r2_xpool.py`, class `XPool`, mode `"pool"`. Per layer there is ONE flat store:
- K and V: `(M * 64, H, D)`; bias: `(M * 64, H)`;
- `M = B * (C + 4) + X_pool` POSITIONS of 64 rows.

Every row holds the H KV heads. So a position has H LANES, and lane h holds KV head h only. Two pinned components
fix this:
- the gilfree packer writes destination row `(m * 64 + r) * H + h` (`gilfree_pack.cpp:152`, E's `f2b981a`);
- piece 3's kernel adds `h * stride(2)` to every address.

The regions:
- positions `[0, A)`, `A = B * C + X_pool`: the HISTORY ARENA. Each (layer, head) has its own free list of A lanes;
- positions `[A, M)`: the TAIL RINGS. They are static: ring page j of request b is position `A + 4 b + j`
  (j 0, 1 exact; 2, 3 draft).

### 2.2 Binding and capacity

The residency service does not change its decisions. Its slots stay LOGICAL `(b, h, s)`, `s < C + X`.
- BIND: `ResidencySC.resolve` binds a slot when the device takes it. This happens after `device_verify` and after
  `device_draft`, and before the urgent gather and the draft bias fill. Per head, the new slots in `(b, s)` order get
  the lowest free lanes. Device ops only; no host read.
- UNBIND: at the first resolve of the next tick, after `sync_mirror`, every FREE slot gives its lane back
  (`release_free(dmirror.code)`).
- THE LANE-LEVEL ORDER (review M4). A transfer waits the LOGICAL slot's last reader (`Transfer.reader_ev`,
  `r2_residency.py` `_take`), not the lane's previous occupant's (the M2 arena contract, `kstep/transfer.py:26`, says
  "the previous PHYSICAL occupant"). Two facts make the lane safe for a new owner, which in the pool can be ANOTHER
  request:
  1. a lane is released only at FREE, and RELEASING -> FREE waits the slot's last reader (`ownership.py:435-450`);
  2. two drop paths set `last_reader = None` without a reader gate (`ownership.py:338-339`, `:357-358`). There the
     lane is safe only because S-C's decision read is BLOCKING (`ResidencySB._d2h`: `t.to("cpu")`, after every
     layer's attention), so every reader of a tick has finished before the next tick's release.
  The pool therefore DEPENDS on the blocking read. `ResidencySB.DECISION_READ_BLOCKING = True` states it, and
  `ResidencySC.resolve` refuses the pool when a class sets it False. XP10 pins `_d2h` (no `non_blocking`) and checks the
  refusal. A captured or asynchronous tick must either keep a blocking read or add a per-lane last-reader event (the
  M2 arena's rule) before it uses the pool. XP13 checks the order on the CPU with readers that complete two ticks late.
- POLICY E reuses a slot of the same stream. Its binding stays, so no lane changes hands there.
- CAPACITY: K5 counts a slot from its take until it is FREE (RELEASING included, `ownership.py:135-136`). Per layer,
  at most `B * H * C + X` slots are in use, so per head at most `B * C + X`. With `X_pool >= X` the arena never runs
  out. `XPool` refuses `X_pool < X`. The binder still checks (err bit 1).
- The map `pos_of (L, B, H, C + X + 4)` is the only translation. Its users: the page tables (`translate`), the urgent
  gather (`gather_entries`, `h2d_entries`), the draft bias fill (`draft_bias_fill_flat`), the host take's bias fill
  (`Residency._fill_bias`), the transports (`Transfer.phys`, set by `assign_phys` at end_tick), and the engine's ring
  writes (`flat_ring_row`).

### 2.3 The error flag

`XPool.err` is a 0-dim int64 on the device. S-C watches it (`watch_flag("xpool_err")`), so the tick's ONE decision read
refuses the tick when it is set.
- bit 1: an arena ran out at a bind;
- bit 2: after a bind, two bound slots share a lane, or the free list disagrees with the bindings (`audit`, every
  bind);
- bit 4: a page table or a gather names an unbound slot.

### 2.4 What changed, file by file

| file | change |
|---|---|
| `r2_xpool.py` (new) | `XPool`, `equal_memory_pool`, the byte formulas, `draft_bias_fill_flat`, `h2d_entries` (a Triton kernel and its CPU twin) |
| `r2_tick.py` | `TickEngine(xpool=)`, `from_r2(xpool=(C, X_pool) or "resident")`; flat ring rows; the backends translate their tables; the attention gets `attn_view` |
| `r2_residency.py` | `Residency(xpool=)` (S-C and subclasses only; no trash slot); `_fill_bias`; `end_tick` sets `Transfer.phys`; `memory()` |
| `r2_residency_sc.py` | bind after each device decision; unbind after `sync_mirror`; `CpuDeviceSim.gather` and `CudaDeviceOps.gather` write lanes |
| `r2_residency_sparse.py` | the draft bias fill on lanes |
| `r2_own2.py` | `memory()["slots_physical"]` counts lanes |
| `r2_transport_gilfree.py` | `xpool=`: the group's `m' = Transfer.phys`; the flat destination |
| `r2_tick_sg5.py` | `--layout stream/pool/both`, `--X-pool`, `--pool-rule stream/k5/both`, `--arrival-lag`; `GatherTransport(xpool=)`; `one_point_both` (stream, pool, stream2) |
| `r2_tick_sg5_step.sh` | cell `x1` with its own root `NOSI_X1` (every other cell keeps `NOSI`, review m8); `r2_xpool.py` checked for x1 |

The default stays the per-stream layout. Every new branch is behind `xpool is not None` or `--layout != stream`.
Backends v1 and sb refuse the pool (their takes do not go through the device binder).

## 3. Kernel interface

Piece 3 v2 (`bi_paged_attention.bi_paged_decode`, `7c826c1`, blob `39ed5e18`) is UNCHANGED. This branch does not edit
the kernel. The FLAT ADDRESS MODE is a convention of the arguments, not a new argument:

```
bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, ...)   # unchanged
  k_cache, v_cache  (N, M*64, Hkv, D): views of ONE flat store (M*64, Hkv, D) with stride(0) == 0
                    (stride(1) = Hkv*D, stride(2) = D, stride(3) = 1); made by XPool.attn_view (expand, no copy)
  bias_cache        (N, M*64, Hkv): the flat bias (M*64, Hkv) with stride(0) == 0 (stride(1) = Hkv, stride(2) = 1)
  page_table[r,h,i] int32, an ABSOLUTE POSITION m in [0, M): rows m*64 .. m*64+63 of lane h; -1 after num_pages
  seq_idx[r]        int32 in [0, N): the engine passes the request b; with stride(0) == 0 it adds 0 to every address
  num_pages, last_len: unchanged
The split kernel's address  K + n*s_kn + h*s_kh + (slot*64 + r)*s_ks + d
becomes                     K + h*s_kh + (m*64 + r)*s_ks + d          (s_kn = 0; the same float ops, the same order)
```

For the shared-KV branch (`nosi-r2-sharedkv`): if it adds an explicit flag, use
`FLAT_KV: tl.constexpr` with the meaning "drop the `n * s_kn` term" (`k_base = K + h * s_kh`, the same for V and bias).
It is bitwise the same as stride 0. The launcher must then still accept `stride(0) == 0` views, and the page-table
entries and lanes stay as above. The two branches then merge without a conflict in the kernel body: this branch
touches no line of it.

The new gather kernel `r2_xpool._h2d_entries_kernel` (grid `(N,)`) takes an entry table `ent (5, N)` int64,
CONTIGUOUS (lesson L28): `b, h, block, position, valid`. It copies host rows `block*64 .. +64` of `(b, h)` into lane h
of the position. Its CPU twin computes the same flat offsets from the base pointers.

## 4. Memory

Units (NOSA-8B: L 32, H 2, D 128, bf16): one slot of one head is 32,896 B (audit). One lane in every layer, per
request at B 64: 32 x 32,896 / 64 = 16,448 B. One stream slot in every layer and head, per request: 2,105,344 B.

Per request (bytes from `store_bytes_formula`, CPU test XP2 checks each value):

| point | today, X 64 | audit, head-agnostic pool 407-422 slots | this layout, X_pool 407-422 positions (814-844 lanes) | this layout, X_pool 64 (cell x1) |
|---|---|---|---|---|
| C 81 | 313,696,256 B = 313.70 MB | 185.65-185.90 MB | 192,342,912-192,836,352 B = 192.34-192.84 MB | 181,059,584 B = 181.06 MB |
| C 96 | 345,276,416 B = 345.28 MB | 217.23-217.48 MB | 223,923,072-224,416,512 B = 223.92-224.42 MB | 212,639,744 B = 212.64 MB |

Why this layout costs more than the audit's pool: a position has H lanes, and K5's X bounds the SUM over heads. So to
keep K5's meaning exact, each head needs X lanes, and the pool holds H x X lanes, not X. The extra is
(H - 1) x X_pool x 16,448 B per request: 6.69-6.94 MB at X_pool 407-422. A head-agnostic pool needs a packer whose
destination row does not add h (E's `gilfree_pack.cpp:152`), so it is out of scope here (call it S6b). S6b also
touches `_h2d_entries_kernel` and `draft_bias_fill_flat` (both add h). Piece 3's kernel needs no edit for it: pass
stride(2) = 0 views and per-(row, head) positions (review m8).

At B 64, cell x1: store 13,608,943,616 B = 13.61 GB vs 22,097,690,624 B = 22.10 GB. Difference 8,488,747,008 B =
8.49 GB.

### 4.1 B at 84 GB (MODEL)

The audit's model: B = (84.0 - F) / marginal, F = 13.12 GB (the forecast's implied fixed part) or 16.37 GB (the
weights); the forecast's marginal 0.24616 GB per request already holds "C 81 groups + overflow 2.06 MB"
(2,056,192 B). The new term replaces the 2.06 MB by the pool share. Case A: the forecast does not count the 4 tail
pages, so they are added (8,421,376 B). Case B: the forecast counts them. The audit did not check which case is true.

| pool share per request | case | marginal (B) | B at F 13.12 | B at F 16.37 |
|---|---|---|---|---|
| X_pool 64 (2,105,344 B) | A (+ tails) | 254,630,528 | 278.3 | 265.6 |
| X_pool 64 | B | 246,209,152 | 287.8 | 274.6 |
| 814 lanes (13,388,672 B) | A | 265,913,856 | 266.5 | 254.3 |
| 814 lanes | B | 257,492,480 | 275.2 | 262.6 |
| 844 lanes (13,882,112 B) | A | 266,407,296 | 266.0 | 253.8 |
| 844 lanes | B | 257,985,920 | 274.7 | 262.1 |

Today's per-stream store: B 178.5-187.1 (audit), or 174.6-183.0 with the tails. Arithmetic: shell integer division
(`$((708800000000 / m))`, then / 10). These are MODEL values: no tick ran at B >= 128. X_pool 64 is the K5 bound of
p5b2's point, where the demand counter peaked at 407-422 per layer (K5 refused or Policy E covered the rest). So the
X_pool 64 rows are valid only for runs that keep K5's X at 64.

## 5. The equal-memory baseline

`r2_xpool.equal_memory_pool` (harness: `--pool-rule`, default `stream`, so p3-p6 stay comparable):
- `stream` (today): P = C + X + 4 - 64. At C 96, X 64: 100.
- `k5`: P = C + 4 - 64 + floor(pool_lanes / (B x H)), with pool_lanes = H x X_pool in this layout, so H cancels:
  P = C + 4 - 64 + floor(X_pool / B) (the harness's `pool_rule_value`; the `--kv-heads` argument of 44d5e95 is gone,
  review m5). At C 96, X_pool 64, B 64: 36 + 1 = 37. With the audit's head-agnostic 64 slots: 36; with 422 slots: 39
  (the audit's 36 + 3.30). This rule rounds DOWN; the audit's Q4 item 4 wrote ceil. The two agree at x1 (64 / 64 = 1)
  and differ by 1 whenever X_pool / B is not whole.

## 6. The registered GPU cell x1 (SOURCE ONLY, not run; registered before any run)

Command: `bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder> <run id> x1` (run from the
`nosi-r2-xpool` checkout; x1's root is `NOSI_X1`, every other cell keeps the script's `NOSI`).
Arguments: p5b2's point, EXACT only: `--modes exact --backends b2 --C 96 --X 64 --R-ver 16`, no sweep, 48 steps,
gilfree 8 threads, the superset wait, unchecked b2, `--layout both --X-pool 64 --pool-rule both --arrival-lag 2`.

Revision after the review of 44d5e95 (B1, m6). Under free timing two runs of the same documents differ in their draft
reads (note 12k, `docs/R2_STEP7_FETCH_RETENTION.md:739-747`), so a row-by-row comparison could fail for timing alone.
Now:
- every engine of the run (preflight, draft-only, rounds, both layouts) uses the fixed arrival lag D = 2 (12k's choice
  for gilfree); each arm reports `arrival` (lag, forced, held);
- per point three arms run in the order stream, pool, stream2. stream2 is the REPLICA: the same layout again, from a
  fresh engine. It must pass the same comparison first (X-R); without that the bitwise gate is VOID;
- the time and peak reports compare the pool with the MEAN of the two stream arms, so the arm order is not a confound.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| X-A | the b2 gilfree preflight on the POOL layout | PASS (MUST) | a refusal or an exception |
| X-R | `SG5_exact_b2_XP_replica` (MUST): stream2 = stream, every tick's rows and fingerprints and the committed tokens | PASS | any difference (then X-B is VOID: the comparison cannot pass under this timing) |
| X-C | `SG5_exact_b2_XP_committed` (MUST): the committed tokens, stream vs pool | PASS | any difference |
| X-B | `SG5_exact_b2_XP_bitwise` (MUST): every tick's rows and every row's logit fingerprint, stream vs pool | PASS | any difference with X-R passed |
| X-S | `store_gb` per arm | 22.098 (stream), 13.609 (pool) | another value (it is tensor bytes) |
| X-M | the round's peak GB, mean(stream) minus pool | 7.9-8.9 GB (MODEL 8.49) | below 7.5 or above 9.5 |
| X-T | T_tick pool / mean(stream, stream2) | 0.97-1.05 | outside 0.85-1.15 (the registered 15 %) |
| X-E | REPORT: the shipped baselines at P 100 and P 37; `xpool.report()` peak lanes per (layer, head); arrival forced / held; stream2 / stream T_tick | - | - |

Scoring of X-T between the prediction and the falsifier (review m6c): a ratio in 1.05-1.15 or 0.85-0.97 is "outside
the prediction, not falsified". It is a REPORT that asks for the per-category profile (`res`, `att`) before any claim.

Peak values (review m6b): the fingerprint allocates about R x V x 8 B of int64 per tick (about 0.26 GB at R 256, V
128,256) plus a float copy, outside the tick spans but inside `peak_gb_round`. So x1's absolute peaks are NOT
comparable with p5b2's or p6's; only the drop X-M is scored.

Time budget (review m6d; MODEL from p5b2's record `sg5_p5_2190876_p5b2.step`: the step ran 05:44:42Z-06:02:19Z,
17.6 min; its two shipped children took 10.5 min, model load and build 3 min, the two preflights 27 s, one EXACT point
1.75 min). x1: three shipped children (P 0, 100, 37) about 16 min; build 3.5 min; preflights 0.5 min; three EXACT arms
5.3 min, doubled for the arrival lag's forced waits and the fingerprints: 10.5 min. Total about 30 min, under a quarter
of STEP_MIN 120.

Derivation of X-T: the pool adds per layer one page-table gather, two binds (two sorts of H x 6,208 lanes and ops over
B x H x (C + X) = 20,480 slots, then the audit) and per tick one unbind over 655,360 slots and one device read
(`assign_phys`); about 25 small launches per layer, 800 per tick, at 6-15 us each: 5-12 ms. The urgent gather becomes
one launch over the decision's entries instead of two mask kernels over B x H x (C + X) slots. p5b2's EXACT round tick
was 1,347 ms, so the predicted change is +0.4 % to +0.9 %; the band allows the run-to-run spread.

Derivation of X-M: the round's peak is measured after `torch.cuda.reset_peak_memory_stats()`, from the round engine's
construction. The model, the build state and the twins are the same in both arms; the store differs by 8.49 GB; the
draft bias fill's buffer is (rows + 1) x H x 2 B per call, 2.7 MB (stream) vs 1.7 MB (pool).

What a result decides:
- X-R fails: the run is not reproducible even with the fixed lag; X-B says nothing. Find the remaining timing source
  first (X-C still holds or fails on its own).
- X-B fails with X-R passed: the pool changes numbers. The cell's other values are void; the first difference names
  tick and row.
- X-B and X-M hold: the shared pool removes the row 1b term (8.49 GB at B 64) with no numerical change.
- X-T fails high: a binder or translator op is slower than modelled. The profile categories `res` and `att` locate it.

## 7. CPU tests (`benchmarks/Efficiency/tests_cpu/test_r2_xpool.py`, NOT RUN)

| node | what |
|---|---|
| `test_XP1_pool_layout_equals_the_per_stream_layout_bitwise[default-b2]`, `[default-sc]`, `[pressure-b2]`, `[pressure-sc]` | (a) bitwise: per tick rows, tokens, logits, logical pages, memory; committed tokens; journal. Pressure preconditions: evictions > 0 and a lane changed hands between two requests |
| `test_XP1n_the_comparison_sees_a_wrong_page` | negative control of XP1's comparison (precondition: the planted table differs) |
| `test_XP1r_the_flat_resident_layout_equals_the_resident_store` | (a) resident: serial logits equal; T4's checks pass |
| `test_XP2_the_pool_store_bytes_equal_the_formula` | (b) bytes: tiny engine and the production geometry (meta tensors); the doc's numbers |
| `test_XP3_two_requests_on_one_lane_are_caught` | (c) negative control "alias", L34 precondition from the plant's record and the map; caught by `xpool_err` |
| `test_XP3b_the_flags_on_direct_inputs` | ERR_UNBOUND, ERR_FULL, release and rebind to another request, X_pool < X refused |
| `test_XP4_the_equal_memory_baseline_rules` | (d) the rule values; the harness's defaults, refusals and `pool_rule_value` |
| `test_XP5_the_entry_gather_twin_models_the_kernel_addresses` | the twin's flat offsets (L28), refusals, the kernel's names (L36) |
| `test_XP6_gilfree_lands_each_transfer_in_its_lane` | the pinned packer on the flat store |
| `test_XP7_the_harness_compares_the_two_arms` | `one_point_both`'s gates with a stand-in; the fingerprint sees one bit |
| `test_XP8_the_step_cell_x1` | the step cell (DRYRUN on a clean checkout) |
| `test_XP9_the_pool_per_layer_path_has_no_host_transfer` | F12d's tokens on bind / audit / release_free / translate / gather_entries / draft_bias_fill_flat; negative control: assign_phys reads |
| `test_XP10_the_pool_needs_a_blocking_decision_read` | (M4) `_d2h` blocks; `DECISION_READ_BLOCKING`; the state codes; a non-blocking class is refused |
| `test_XP11_the_entry_kernel_under_the_triton_interpreter` | (M2) the kernel = its twin under TRITON_INTERPRET=1 (fp32 / fp16 bias, no bias); the same out-of-bounds entries skipped; negative control (b / h swapped) |
| `test_XP12_gather_entries_lands_a_real_decision_as_the_device_stand_in[f20b]`, `[f21b]` | (M1) real S-C decisions with Policy E: gather_entries + twin = CpuDeviceSim.gather; negative control: the mask without evict |
| `test_XP13_the_pool_service_equals_the_per_stream_service_and_early_unbinds_are_caught` | (M3, M4) S-C pool = S-C per stream on F20b with a lane-level reader check; plants release_releasing (hazards) and release_inflight (the comparison differs), each with its precondition |

Regression files to run with them: `test_r2_tick.py` (T9, T18, T19), `test_r2_tick_b2.py`, `test_r2_tick_p6.py`,
`test_r2_residency.py` (F12d scans `ResidencySC.resolve` and `CudaDeviceOps.gather`; F23 now also lints
`r2_xpool.py`), `test_r2_transport_gilfree.py`, `test_r2_own2.py`, `test_r2_own2_b2t.py`.

## 8. Open points and risks

- The kernel `_h2d_entries_kernel` has not run on a GPU. XP11 runs it under the Triton interpreter; cell x1 is its
  first GPU run (X-A, X-B).
- The pool depends on S-C's blocking decision read (2.2, the lane-level order).
- An xpool_err set in a tick that also overflows is reported AFTER the overflow refusal (S-C and B2 apply_tick), so
  an overflowed page that names slot 0 is not blamed on the pool (review m7).
- `Residency(xpool=)` refuses a transport whose `.xpool` is not the pool (review m2); TorchGather is built but only v1
  calls it, and v1 refuses the pool.
- The pool layout adds ONE device-to-host read per tick that issues draft transfers (`assign_phys`). S-C's "one read
  per tick" becomes two in the pool layout only.
- The fingerprint is not a proof of bitwise equality: one differing element always changes it; several could cancel.
- `GatherTransport` with the pool is source only; x1 uses gilfree.
- v1 and sb refuse the pool. The captured tick (trash slot) refuses it too.
- The head-agnostic pool (S6b, the audit's exact numbers) needs a packer change; not started.
