# R2 step 7: capturing the joint tick (audit against law 3, and the design)

Branch `feature/nosi-r2-tick-capture`, from `feature/nosi-r2-m2-tick` a35037b (the tick that
passed SG0-SG4 on the A100: `/mnt/beegfs/ojerbi/r2-runs/tick/tick_v2_2190876`). Status:
DESIGN ONLY. No code changes. Nothing here comes from a GPU run.

Inputs:

- the tick: `nosi/nosi/r2_tick.py`, `nosi/nosi/r2_spec_core.py`, `nosi/nosi/r2_bi_decode.py`,
  `nosi/nosi/r2_positional_twin.py` (all at a35037b);
- C's sync-free twin step: 4fe5fc9 (`step_unchecked`, `advance_unchecked`; it is on the SG5
  line, NOT in a35037b);
- C's residency, piece 5: `feature/nosi-r2-m2-sg5` 39187a6 (`r2_residency.py` v1,
  `r2_residency_sb.py` S-B, `r2_residency_sc.py` S-C);
- piece 3's attention: `nosi-r2-m2-biattn` 7c826c1 `nosi/nosi/bi_paged_attention.py`;
- the capture of the plain decode step: `feature/nosi-r2-m2-capture` e5fc694
  (`docs/R2_CAPTURE_AUDIT.md`: the lint classes, the allocation rule, the undefined-rows
  lesson). Its gate PASSED on holder 2190876 (run cap2): CE 67.53, CF 67.42, CG 25.08 ms per
  step at B 8.

Line numbers below are those of a35037b unless a commit is named.

## 0. The answer in short

1. The tick cannot be captured as it is. There are four blockers:
   - **B1, ragged shapes.** The row count R (1 .. 4B per tick) and the rows per bank T
     (1 .. K) set the shape of every tensor of the forward.
   - **B2, the checked twin.** `PositionalTwin.advance` / `step` read device values on the
     host. That is about L x (1 + 7T) syncs per tick: 256 at T = 1 and 704 at T = 3, for
     L = 32.
   - **B3, host state inside the forward.** The token-log cursor, a host branch on
     `any_draft`, the `check` reads, and the per-layer `ctr` list.
   - **B4, the residency's host side (SG5 only).** v1 resolves on the host at every layer.
     S-C runs host work per layer: the mirror refresh, the row inputs and the event waits.
2. The per-tick decision read is NOT a blocker. It is between two ticks, outside the region.
   It stays: one sync per tick with verify rows. The host plans the next tick from it.
3. The design is a FIXED-CAPACITY TICK:
   - R_cap = B x (1 + K) rows (= G-b's `m_fix` = 4B at K = 3) and T_cap = K twin steps;
   - padded rows only read valid memory and write only to trash cells;
   - C's `advance_unchecked`, device error flags, static input / output buffers;
   - two graphs: G_d (R_cap = B, T_cap = 1) for draft-only ticks and G_v (R_cap = 4B,
     T_cap = K) for the others, with one shared memory pool.
   - The host side (plan, row table, decision read, `apply_events`) stays at the tick
     boundary. `apply_events` becomes sync-free first (E1). It is captured only if SG5's
     dHOST asks for it (E2).
4. Why B x (1 + K) and not B x (LAMBDA + 1): see 8.1. The design takes the cap as a
   parameter.
5. The gate TC (section 9):
   - TE (the eager tick, the reference);
   - TF (fixed capacity, eager) and TG (captured): MUST bitwise TE on every live row and in
     every state digest;
   - PTR, ERR, and the refusals;
   - three negative controls with exact predicted first ticks (padding leak, stale row
     table, frozen log cursor), and one REPORT control (a batch-keyed split across the graph
     family);
   - T (ms per tick).

## 1. The law, the region, the method

THE LAW (retroinfer-eval `AGENTS.md`, law 3, as in `docs/R2_CAPTURE_AUDIT.md`). Inside a
captured region:

- fixed shapes;
- fixed device addresses, with in-place writes only;
- no allocation;
- no host branch on a device value;
- no `tensor[scalar_tensor]` indexing;
- the region cannot raise.

The ALLOCATION RULE (lead, 2026-10-20) applies here too. Transient intermediates may come
from the graph's private pool. STATE (anything read after the replay or by the next tick) is
written in place into preallocated buffers.

THE REGION: `TickEngine.run_rows` (`r2_tick.py:436-485`), `_layer` (`:487-539`),
`page_table` (`:541-561`), and their callees:

- `PositionalTwin.advance` / `step`;
- `r2_spec_core.select_blocks`, `PaddedLinear.__call__`;
- `r2_bi_decode.fixed_order_mean`;
- piece 3's `bi_paged_decode`;
- stage 1, the pooling, rope, `nosa_linear`, `layer_norm`, `silu_and_mul`.

OUTSIDE THE REGION BY DESIGN (the driver, at the tick boundary): `Round.plan_tick`,
`TickEngine.row_table`, `TickRound.finish_tick` / `_decide` / `_resolve`, `read_tokens`,
`apply_events`, and the residency's host calls.

THE METHOD: a static read of every line of the region. The lint of e5fc694
(`tests/test_r2_capture.py`) is NOT in this tree. Order of work W1 (section 11) brings it
here and makes the tables below its expected findings, as in the first capture audit.

## 2. What one tick does today, and what it costs

Per tick, on the host: `plan_tick` -> `row_table` (a Python loop, one `torch.tensor` per row)
-> `_to_device` (pinned H2D) -> the forward (per layer: `twin.advance` over T steps, stage
1, the pooling, `select_blocks`, the store writes, the page table, attention, the FFN) ->
`tok_log[lo:lo + R]` -> `finish_tick`. On a tick with verify rows: `read_tokens` (the
decision read) -> `_decide` -> `apply_events`.

Host reads per tick (a static count; a35037b):

| Where | Reads | Per tick |
|---|---|---|
| `advance` `r2_positional_twin.py:287` | `bool(((n < 0) \| (n > T)).any())` | L |
| `step` `:223` | `active.nonzero()` | L x T |
| `step` `:226` | `bool(prefilled[b].all())` | L x T |
| `step` `:234`, `:236` | `bool(...)` capacity checks (2 or 3) | about 3 L x T |
| `step` `:248`, `:259` | `b[due_k]`, `b[due_c]` (boolean-mask indexing) | 2 L x T |
| `_layer` `r2_tick.py:520-523` | `check`: two `bool(...)` (gates only) | 2 L |
| `read_tokens` `:567` | the decision read | 1 (verify ticks) |

That is L x (1 + 7T) without `check`: 256 at T = 1 and 704 at T = 3 (L = 32). `apply_events`
adds more:

- `commit` (`:302`, `:305` `torch.equal`, `:308`, `:310`): 4 per layer on a rollback;
- `rebuild_from_own_rows` -> `advance` over the draft's own n rows: L x (1 + 7n) per
  publication (n <= LAMBDA = 7: up to 1600).

Measured (tick_v2_2190876, B 8, K 3, LAMBDA 7, m_fix 32):

| Part | Ticks | With verify rows | Max rows per tick | Host ms per tick |
|---|---|---|---|---|
| SG1 | 177 | 124 | 25 | 121.1 |
| SG2 | 121 | 102 | 32 | 134.1 |
| SG3 | 238 | 190 | (phase mix) | 114.8 |

For comparison: REF (the BI serial decode) took 136.3 ms per step, and the engine's serial
mode (SG0) 75.6 ms. SG2 reached 32 rows = B x (1 + K), the cap of 8.1. The eager tick is
sync- and launch-bound. One example: `fixed_order_mean` (`r2_bi_decode.py:46-55`) is a chain
of 31 elementwise adds, and the twin calls it twice per step, per layer.

## 3. Table T-A: the region (`r2_tick.py`)

Classes as in `docs/R2_CAPTURE_AUDIT.md` (SYNC, SHAPE, BRANCH, ALLOC, SCALARIDX, HOSTSTATE,
CALLS_EAGER). "T" = transient under the allocation rule (allowed). The fixes (F1 .. F14;
the numbers are labels, not a complete range) are the design of section 8.

| Line | What | Class | Fix |
|---|---|---|---|
| `:441`, `:484` | `time.perf_counter()` around the forward | HOSTSTATE (timing) | the driver times with CUDA events (F14) |
| `:443` | `row_table` returns R and T: every shape below depends on them | SHAPE (B1) | R_cap, T_cap (F1); the driver pads the table |
| `:445`, `:430-433` | `tab.pin_memory().to(dev, non_blocking=True)`: a new device tensor per tick | ALLOC (state input) | F2: copy into a STATIC device table (pinned staging made by the driver, before the replay) |
| `:447-452` | `.to(int32)`, `pos // 64 + 1`, `pos % 64 + 1` | T | F2: compute into static buffers (they are inputs of every layer) |
| `:454` | `cu_k = torch.cat([bank * lk, torch.full(...)])` | ALLOC, SHAPE (R + 1) | F2: a static (R_cap + 1) buffer, written in place in the region |
| `:455-457` | `cu_q_cache` keyed by R (a host dict) | BRANCH, ALLOC on a host value | F2: one static `arange(R_cap + 1)` |
| `:458` | `ids = where(lit >= 0, lit, tok_log[slot])` | clean (device indices) | none (reads the static table) |
| `:459` | `n_host = torch.tensor(count)` (CPU) -> `advance` | ALLOC, then SYNC inside `advance` | F4: per-bank counts in the static table (device) |
| `:460` | `any_draft` (host) -> `page_table` `:549` | BRANCH on per-tick host data | F9: no branch: the drop path always runs when `draft_drop > 0` (equal bits when no row is a draft row: keep = all, dest = arange) |
| `:461`, `:476-484` | `pos0`, `bank_pos +=`, stats | HOSTSTATE | the driver, after the replay (F13) |
| `:469-470` | lm_head through `ops.linear`, `argmax` | T | none |
| `:471-475` | `lo = self.n_tok`; `raise` if full; `tok_log[lo:lo + R].copy_(arg)` | HOSTSTATE (a slice at a host offset is frozen at capture), BRANCH + raise | F6: a `logslot` column in the table; `tok_log.index_put_((logslot,), arg)`; padded rows -> trash slot `LOG - 1`; the driver checks `lo + R < LOG - 1` |
| `:478` | `self.last_ctr = ctrs` (per-layer new tensors, read by `apply_events` after the tick) | ALLOC of STATE | F5: `ctr_buf[l]` (T_cap + 1, 4, 2B), static, written by `copy_` |
| `_layer` `:504-505` | `keys`, `cis_b` = `torch.zeros((2B, T, ...))` | T, but SHAPE (T) | F3: (2B, T_cap + 1, ...); column T_cap is the padded rows' trash column |
| `:506-507` | `keys[bank, u] = k` | clean (device indices); padded rows need a target | F3: padded rows write column T_cap of bank 0 |
| `:508` | `twin.advance(keys, cis_b, n_host)` | CALLS_EAGER (B2: L x (1 + 7T) syncs, a host loop over T, raises) | F4: C's `advance_unchecked` (4fe5fc9) over T_cap steps with the device counts; its sticky `overflow` flag |
| `:509` | `n_used = ctr[u + 1, 1, bank]` | clean | F3: padded rows read `ctr[min(u + 1, T_cap)]` (u = T_cap is out of range) |
| `:511` | `comp_k_flat()` (a view) | clean | none |
| `:512` | stage 1 | clean (captured by X4's CG) | F7: padded rows `seqused_k = 1` (minimal work, valid) |
| `:514-516` | `index_select`, `torch.empty` (pq, pc) | T | none |
| `:517` | pooling | clean (captured by X4's CG) | none |
| `:519` | `select_blocks(pq.float(), pc.float(), ...)` | T; `torch.sort(stable=True)` x 2 | OP-1: the first capture of a stable `torch.sort` here; the gate shows it |
| `:520-523` | `if self.check: bool(...)`, `raise` | SYNC, BRANCH, raise | F8: a device flag `err \|= (sel[..., -1] != T) \| (sel[..., -2] != T - 1)` over the LIVE rows, read with the decision read |
| `:525-527` | `store[l].index_put_((rb, ring_row), ...)` | clean (device indices); padded rows need a target | F3: padded rows -> a TRASH SLOT `NB + 4` of request 0 (Q-T2) |
| `:529`, `:546-561` | the page table, the drop path | T; BRANCH `:549` | F9 (above); F7: padded rows `num_pages = 1`, page 0 = the trash slot, `last_len = 1` |
| `:530` | `bi_paged_decode` | clean: shape checks only, grid (R, Hkv, ceil(P / 8)) from shapes, `out` / workspace transient or passed (`bi_paged_attention.py:297-353`) | F10: a static workspace for R_cap (optional; T otherwise) |
| `:537` | `out = torch.empty(...)` (FFN) | T | none |

## 4. Table T-B: the twin

| Function | Status | Fix |
|---|---|---|
| `advance` (`r2_positional_twin.py:277-293`) | a SYNC at `:287`; a host loop over T; `step` per u | replaced in the region by `advance_unchecked` (4fe5fc9 `:328-341`): static checks only, T from the shape, a device `n` |
| `step` (`:208-271`) | SYNC x about 7, SHAPE (`nonzero`, boolean masks), ALLOC (`torch.ones` / `full` / `as_tensor`), raises | replaced by `step_unchecked` (4fe5fc9 `:276-326`): masked ops over all banks, the same shape-stable means, `overflow \|= bad.any()` (a device flag), in-place counters. Its `.clone()` calls are T. Clean for capture by reading (W1's lint confirms it) |
| `commit` (`:295-317`) | SYNC x 4, raises | outside the region (in `apply_events`); E1: `commit_unchecked`, the checks on the HOST MIRRORS (bank positions, epochs, MAX_ROLLBACK), not on device values |

C's condition on `step_unchecked` (4fe5fc9 docstring): "A TRUE FLAG VOIDS THE TICK". The
driver reads `overflow` with the decision read. A true flag stops the run before any decision
of that tick is applied. On draft-only ticks nothing is read, so a flag raised there is seen
at the next decision read. That read then refuses every decision since the last read (Q-T5).

## 5. Table T-C: the callees

| Callee | Status | Note |
|---|---|---|
| `select_blocks` (`r2_spec_core.py:321-329`) | T only: `torch.sort(stable=True)`, `zeros_like`, `scatter_`, `masked_fill` | OP-1: capture of the stable sort (segment length M_alloc, about 260 at 16K) is not yet proven on this stack |
| `PaddedLinear.__call__` (`:440-449`) | `buffer()` allocates on the FIRST call per (K, dtype) | a warm-up tick before the capture creates every buffer; then clean (`buf[:M].copy_` with M = R_cap <= m_fix) |
| `fixed_order_mean` (`r2_bi_decode.py:46-55`) | clean (31 elementwise adds) | the graph removes its launch cost |
| `bi_paged_decode` (7c826c1 `:297-353`) | clean: no device read (`validate_inputs` is separate, tests only); `if R == 0` is static | a padded row needs `num_pages >= 1` (the kernel's contract, `:154`) |
| stage 1, pooling, rope, `nosa_linear`, `layer_norm`, `silu_and_mul` | captured by X4 (CG PASS, cap2) | same kernels, R_cap rows |

## 6. Table T-D: the driver (outside by design)

| Call | Host reads | In the captured design |
|---|---|---|
| `plan_tick`, `row_table` | none (Python) | stay; `row_table` builds the padded table from one Python list (no `torch.tensor` per row) for dHOST |
| `read_tokens` (`r2_tick.py:564-567`) | the decision read | stays: ONE read of `tok_log[lo:hi]`, the err word and `overflow` together (one D2H of a small staging tensor) |
| `finish_tick`, `_decide`, `_resolve` | none (host) | stay |
| `apply_events` (`:575-622`) | `commit` x 4 per layer; `rebuild_from_own_rows` -> checked `advance` | E1 (8.7): sync-free, eager; E2: an optional captured events graph |
| `writeback`, `copy_ring` (`:624-638`) | none (device copies, Python-int slices) | stay (eager) |
| `copy_rows` (`:640-649`) | none; `torch.tensor` + pinned H2D | stays (eager) |

## 7. Table T-E: the residency's host side (SG5)

| Version | Per layer | Capturable? |
|---|---|---|
| v1 (`r2_residency.py` `resolve`) | `sel` to the host (one sync per layer), a host plan, H2D of the page table | NO: a host round trip between the layers |
| S-B (`r2_residency_sb.py`) | the verify rows are v1 (host); the draft rows are on the device | NO (the verify rows) |
| S-C (`r2_residency_sc.py` at 39187a6) | `resolve` `:369-406`: `_check_rows` (host); `m.refresh(self, l)` (host loops over the streams, then H2D); `_row_inputs` (host tensors, H2D); `device_verify` / `device_draft` (device, no host read: F12d); `dev.wait` `:297-312`: `main.wait_event(tr.ev)` for EVERY transfer in flight at the layer, and a host wait for a gilfree pack's submission; `dev.gather` `:314-321`: `wait_event` on the host write-backs, then the device-driven gathers; `plan.last_len` / `seq_idx` built on the host | YES after the moves below |

The moves that make S-C capturable (they need C's confirmation, Q-T3):

1. **Mirror refresh.** The refresh of ALL L mirrors moves to the tick start, into static
   per-layer mirror buffers. Under S-C's schedule (`poll_in_resolve=False`,
   `defer_issue=True`, `defer_admit=True`) the host ownership state does not change inside a
   tick. The device updates of a layer (the urgent takes) are device ops and stay in the
   graph.
2. **Row inputs.** `_check_rows`, `_row_inputs`, `last_len` and `seq_idx` are the same at
   every layer of a tick. They are computed once per tick, before the replay. The verify and
   draft rows go to static (Rv_cap = B x K, Rd_cap = B) row buffers with a `valid` mask.
   `sel_d[:, ix]` with a host index list of varying length is replaced by fixed-size device
   index buffers.
3. **Waits.** The set of events varies per tick, and a graph cannot wait on a varying set.
   - (a) One superset wait before the replay. The transfers in flight are the same set at
     every layer, because draft transfers are issued at `end_tick` and urgent gathers run on
     the main stream. The price is overlap: layer 0 waits for the blocks that only layer 31
     reads.
   - (b) Per-layer event-wait nodes on a FIXED set of per-layer join events that the
     transport records again each tick (external events: check on torch 2.6).
   - The gilfree submission host-waits and the T4 write-back waits move before the replay in
     both cases.
4. **Reader events.** `read_done(l, plan, ev)` (`r2_residency.py:588`) records one event per
   layer. In a graph: either one event recorded after the replay (every slot read in the tick
   gets it), or per-layer external record nodes. The first is simpler. It is equivalent if no
   slot is released and taken again inside one tick (S-C applies releases at `end_tick`:
   Q-T3c).
5. **`end_tick`.** S-C reads the device ONCE PER TICK (`:408-417`), also on draft-only ticks.
   On verify ticks it merges with the decision read. On draft-only ticks it is an extra sync
   that the ResidentStore tick does not have (Q-T3e).

## 8. The design: a fixed-capacity tick

### 8.1 Capacities

- Draft rows: at most one per request per tick (`r2_spec_core.py:142-146`): <= B.
- Verify rows: `plan_tick` adds the rows of the OLDEST ready cohort only and stops at the
  first cohort that is not DONE (`:147-153`). A cohort has K - 1 rows (m = 1), K rows, or a
  final short cohort <= K rows (`cohort_range` `:59-62`, the flush `:184-188`). So there are
  at most B x K verify rows.
- So R <= B x (1 + K) = 4B at K = 3. SG2 reached 32 = 4B at B 8. Rows per bank: T <= K
  (exact bank) and 1 (draft bank).
- R_cap = B x (1 + K) is also G-b's `m_fix` (4B), so the EXACT GEMMs do not change.

The brief says "verify rows up to B x (LAMBDA + 1)". That bound applies only if a request
could verify its whole lead (up to LAMBDA drafted tokens, plus the correction position) in
ONE tick, which today's `plan_tick` never does. The design takes V_cap (verify rows per
request) as a parameter:

- V_cap = K: R_cap = 4B, m_fix = 4B (today's bits).
- V_cap = LAMBDA + 1: R_cap = 9B, and m_fix >= 9B, which changes G-b's bits and the
  reference's.
- Every per-row op (stage 1, pooling, selection, attention, the store writes) scales with
  R_cap.

Recommendation: V_cap = K now; LAMBDA + 1 only with a plan that verifies several cohorts per
tick (Q-T1). The driver REFUSES R > R_cap or T > T_cap before any replay, as a host check.

### 8.2 Static buffers (TickBuffers, allocated once per engine)

| Buffer | Shape | Written by |
|---|---|---|
| row table | (len(FIELDS) + 3, R_cap) int64 | the driver: one non-blocking copy from pinned staging before the replay (`t.pin_memory()` per tick keeps the caching host allocator's event protection when draft-only ticks run ahead) |
| extra fields | `live` (bool), `logslot` (the token-log slot), `ucol` (the twin column: u, or T_cap for a padded row) | the driver |
| per-bank counts `n` | (2B,) int64 | the driver (with the table) |
| `cu_q`, `cu_k`, `pos32`, `rb32`, `M_b`, `last`, `seqk` | (R_cap [+ 1],) int32 | the region, in place, from the table |
| `ctr_buf[l]` | (T_cap + 1, 4, 2B) int64 per layer | the region (`copy_` of `advance_unchecked`'s result) |
| `err` | () int32 | the region (`check` bits, F8); the driver zeroes it after each read |
| attention workspace | (R_cap, Hkv, S, bh, D) fp32 and (R_cap, Hkv, S, 2, bh) | the region (F10) |
| PaddedLinear buffers | (m_fix, K) per (K, dtype) | created by the warm-up tick |
| trash cells | store slot `NB + 4` (Q-T2), twin column T_cap (transient), token-log slot `LOG - 1` | the padded rows only |

### 8.3 Padded rows

A padded row k in [R, R_cap) is a fixed DUMMY row: request 0, bank 0, `pos` = bank 0's
current position (the driver's mirror), `live` = 0.

- It reads only valid memory: bank 0's tables; `seqused_k` = 1; `num_pages` = 1; page 0 =
  the trash slot; `last_len` = 1.
- It writes only trash cells: twin column T_cap (`n` counts live rows only), store slot
  `NB + 4`, token-log slot `LOG - 1`.
- Its outputs are UNDEFINED and never compared (the e5fc694 lesson: compare the defined
  result, and prove separately that the undefined rows reach no state). The state digests
  in TF / TG and the negative control NG1 (9.3) prove that it reaches no state.

### 8.4 The layer in fixed form (the changes from `_layer`)

- F3: keys / cis scratch (2B, T_cap + 1, ...); `keys[bank, ucol] = k`;
  `ctr = advance_unchecked(keys[:, :T_cap], cis[:, :T_cap], n)`;
  `ctr_buf[l].copy_(ctr)`; `n_used = ctr[(u + 1).clamp(max=T_cap), 1, bank]`;
  `seqk = where(live, n_used, 1)`.
- F7: padded rows: `num_pages`, page 0 and `last` overridden with `torch.where` on `live`.
- F8: the `check` flag on the device, live rows only.
- F9: the drop path without the host branch.
- F3 (store): the `index_put_` targets of padded rows point to the trash slot.
- F6 (after the last layer): `tok_log.index_put_((logslot,), argmax)`.

Everything else is `_layer` unchanged. So a LIVE row computes exactly the eager tick's ops
on the same data:

- the GEMMs are EXACT at m_fix in both;
- every other op is row-wise;
- `advance_unchecked` is bitwise `advance` (C's U1).

TF == TE bitwise is the claim of the gate.

### 8.5 The graph family

- G_d: (R_cap = B, T_cap = 1), for ticks with no verify row (R <= B, every bank T <= 1).
- G_v: (R_cap = 4B, T_cap = K), for every other tick.
- The driver replays the smallest graph that fits.

Both graphs are captured after at least one EAGER fixed tick of their own size. That tick
compiles the Triton kernels and creates the PaddedLinear buffers and the workspace. They use
one shared memory pool (`torch.cuda.graph(g, pool=...)`; the replays never overlap). The two
graphs must give equal bits on a tick that both can run (EXACT GEMMs at m_fix in both). The
gate checks this, and it is where a batch-keyed op shows up (NG4).

FAST mode (natural cuBLAS at R rows) changes under capture: the GEMM's M is R_cap, not R. So
FAST stays REPORT, as in SG0-SG4.

### 8.6 The driver, per tick

1. `plan_tick` -> rows.
2. The padded table (one Python list -> one tensor); the refusals:
   - R <= R_cap and T <= T_cap of the chosen graph;
   - the capacity checks on the host mirrors (C's condition for `step_unchecked`);
   - `lo + R < LOG - 1`.
3. [SG5: the residency's pre-replay part, 7.1-7.3.]
4. Pinned H2D of the table into the static buffer; `graph.replay()`.
5. If the tick has verify rows: ONE read (the token slice, `err`, `overflow`, [S-C's
   decisions]); `finish_tick` -> `_decide`; `apply_events` (E1).
6. Host bookkeeping: `bank_pos`, `n_tok`, stats.

Draft-only ticks make no read, so the host can enqueue several draft ticks ahead. This is as
today, but without the twin's syncs.

### 8.7 `apply_events`

- **E1 (eager, sync-free), first.**
  - `commit_unchecked(ctr, keep)`: the keep index goes to the device from the host dict; the
    counters are set by device ops, and `epoch += 1`. The checks (stale snapshot, ahead,
    MAX_ROLLBACK) run on the host mirrors.
  - `rebuild_from_own_rows` uses `advance_unchecked` with a device count.
  - `writeback`, `copy_ring` and `copy_rows` stay as they are (device copies).
  - Expected: no sync left in `apply_events`.
- **E2 (captured events graph), only if SG5's dHOST asks for it.**
  - A rollback / publish mask per request and a keep vector go in through a static table.
  - The masked counter reset, the draft-bank sync as masked whole-table copies, and the
    write-back with device indices. The copies are large: `comp_k` alone is about 34 MB per
    layer at B 64 (64 draft banks x about 1040 rows x 256 x 2 bytes), about 1.1 GB per tick
    over 32 layers, or about 1 ms of copy bandwidth.
  - This is more work than E1 on ticks with few events. Measure first (Q-T4).

### 8.8 The residency on the captured tick (S-C in the graph)

- Before the replay (host):
  - `poll`;
  - refresh all L mirrors;
  - the row inputs once;
  - the gilfree submission waits;
  - the main-stream waits: one superset (7.3a), or the per-layer joins (7.3b);
  - the T4 write-back waits.
- In the graph: `device_verify` / `device_draft` at fixed Rv_cap / Rd_cap with the `valid`
  mask, the urgent gathers (device-driven Triton H2D, as X4's gathers), and the mirror's
  device updates.
- After the replay: one reader event for the tick (7.4); `end_tick` (merged with the
  decision read on verify ticks).
- SG5's `commit_block` (exact ring page -> pinned host rows + an event) stays in
  `apply_events` (eager).
- v1 and S-B are NOT capturable. SG5 on the captured tick needs S-C, and S-C's own GPU gate
  FG1 first.

### 8.9 Expected effects (NOT registered; the gate's commit registers them)

- **Syncs per tick:** from L x (1 + 7T) (+ `apply_events`) to 1 on a verify tick and 0 on a
  draft-only tick (ResidentStore). With S-C, 1 per tick.
- **Launches:** one graph launch per tick instead of thousands. For example, the twin's
  means: 2 x 31 adds per step, per layer.
- **Anchor:** the plain decode step went from 67.5 to 25.1 ms at B 8 (cap2, 2.7x). The eager
  tick is more sync-bound (115-134 ms per tick at B 8), so I expect a larger factor. I will
  not give a number before W2's CPU tick and W4's warm-up timings exist.
- **Costs of the fixed capacity:**
  - every per-row op runs R_cap rows. A padded row's attention and stage 1 are one page and
    one key; the pooling and the selection run at full width M_alloc;
  - G_v runs K twin steps also on verify ticks whose banks have fewer rows;
  - two graphs' pools and the static buffers use memory.

## 9. The gate (TC; `benchmarks/Efficiency/r2_tick_capture_gate.py`, to build)

Setup as the tick gate (`r2_tick_gate.py`): B 8 ragged documents of pg19-test-c16k, one
`build_ragged`, BASE = `TickEngine.from_r2`, every arm on `BASE.clone()`, EXACT GEMMs at
m_fix = 4B, the same piece 3 pin. Teacher-forced on REF's tokens with SG1's
`forced_schedule`, and once with SG3's `phase_ready`. Both schedules are model-free
(`simulate_schedule`), so the gate knows every tick's rows, R and T before it loads the
model. The predictions of the negative controls come from that plan.

### 9.1 Arms

| Arm | What | Gate |
|---|---|---|
| TE | the eager tick (a35037b's `run_round`): the reference. Records per tick every live row's logits (float32), argmax and per-layer selection; per layer `ctr`; after every tick a digest of every twin (both banks, every table and counter) and of every store's historical slots and rings | the reference |
| TF | the FIXED-CAPACITY tick, EAGER (the fixes of section 8, G_d / G_v sizes, no capture) | MUST bitwise TE: every live row, every tick; every state digest, every tick; `err` 0, `overflow` False |
| TG | TF captured: G_d and G_v after one warm-up tick each, then replays | MUST bitwise TE as TF; PTR: no state `data_ptr` moves across the replays (twins incl. `overflow`, stores, `tok_log`, TickBuffers, PaddedLinear buffers, workspace); ERR |
| TX | TG with the graph CHOICE forced to G_v on every tick | MUST bitwise TG (the two graphs agree on every tick that both can run) |
| TS | the serial mode (`run_serial`) through the fixed path, R = B on G_d, against REF (the BI serial decode) | MUST bitwise (SG0 under capture) |
| TR | the refusals: a tick with R > R_cap, one with T > T_cap, one with `lo + R >= LOG - 1` | MUST: each refused BEFORE any replay, the state unchanged (digests) |
| T | ms per tick: TE, TF, TG (CUDA events from the first launch to the end of the replay; and the host gap from the replay's end to the next replay) | REPORT |

### 9.2 Defined rows

The compared rows are the LIVE rows only. Padded rows have no reference. That they reach no
state is shown by the per-tick state digests (TF, TG) and by NG1.

### 9.3 Negative controls (each on TG; the predicted first tick comes from the model-free plan)

| Control | Plant | Prediction |
|---|---|---|
| NG1 PAD-LEAK | padded rows count into bank 0's `n` (they advance bank 0's twin) | first difference exactly at the first tick with R < R_cap of its graph, in the twin digest / `ctr` of bank 0 |
| NG2 STALE-TABLE | the driver skips the table's H2D at tick t0 (the graph replays tick t0 - 1's rows) | first difference exactly at t0 |
| NG3 FROZEN-CURSOR | `logslot` fixed at its capture-tick values | first difference exactly at the first tick whose input is a `Ref` to a slot that only a later replay wrote |
| NG4 BATCH-KEYED SPLIT | SG4 (b)'s `BatchKeyedSplit` keyed on the graph's R_cap, with TX | REPORT: DETECTED (TX differs from TG) or BLIND |

### 9.4 Step script

The holder pattern of `r2_tick_step.sh`:

- pins: the image, the model size, the workload sha256, the piece 3 blob;
- the GPU uuid read from `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<job>.gpu_uuid`: STOP
  (exit 1) before any claim if it is missing, unreadable or not a uuid; record it in
  `command.txt`; the child keeps the `nvidia-smi` check (exit 3);
- a claim directory; exit = the number of failed MUST gates; 20 refused; 30 an exception
  outside an arm.

## 10. CPU proof (`benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py`, to build)

On the tick's CPU stand-ins (`test_r2_tick.py`):

- **C1:** TF == TE bitwise on every live row and every state digest. SG1's schedule over a
  run with rejections at every offset and both boundary kinds; SG3's phase mix; the serial
  mode.
- **C2:** padding invariance: the same tick at R_cap = R, B and 4B gives equal live rows and
  equal state.
- **C3:** padded rows write only trash cells. A digest of everything outside the trash cells
  is equal with and without padding.
- **C4:** NG1, NG2 and NG3 first differ at exactly the predicted ticks (the same functions as
  the gate).
- **C5:** the refusals of TR.
- **C6:** E1: `commit_unchecked` and `rebuild_from_own_rows` on `advance_unchecked` are
  bitwise the checked versions over the SG1 schedule. Negative control: a stale snapshot is
  refused by the host-mirror check.
- **C7:** the lint of e5fc694, moved to a shared helper, on the fixed functions,
  `step_unchecked`, `advance_unchecked`, `select_blocks`, `PaddedLinear.__call__` and the
  attention launcher:
  - the captured functions are clean;
  - today's `run_rows` / `_layer` / `advance` / `step` show at least the classes of
    Tables T-A and T-B (so this note cannot go stale silently);
  - planted violations are found.

The capture itself, the replay, PTR and the timings need the GPU.

## 11. Order of work (one commit each; SG0-SG4 stay bitwise at every step)

- **W0:** cherry-pick C's 4fe5fc9 (`step_unchecked` / `advance_unchecked`, twin only) onto
  this branch. Its tests U1 come with it.
- **W1:** the shared lint helper and C7 (the expected findings = Tables T-A / T-B).
- **W2:** TickBuffers, the fixed eager tick (TF) behind a switch, C1-C5. `run_rows` stays the
  default and the reference.
- **W3:** E1 and C6.
- **W4:** the capture (G_d, G_v, the shared pool), the gate TC and its step script. GPU run on
  the lead's GO.
- **W5:** S-C in the graph (8.8), after C's FG1 passes and Q-T3 is answered. Then SG5 on the
  captured tick.

## 12. Questions

- **Q-T1 (lead):** V_cap = K (R_cap = 4B, today's bits) or LAMBDA + 1 (R_cap = 9B, m_fix
  >= 9B, new bits for the reference too)? This note assumes K.
- **Q-T2 (C):** the padded rows' store writes need a trash target that no live row writes.
  Is one more tail slot acceptable (store `(C + X + 5) x 64` rows instead of `C + X + 4`), or
  does C prefer another target?
- **Q-T3 (C), S-C in a graph:**
  - (a) is the host ownership state constant inside a tick under `poll_in_resolve=False`,
    `defer_issue`, `defer_admit`, so that all L mirrors can be refreshed at the tick start?
  - (b) waits: one superset wait before the replay, or per-layer external join events?
  - (c) can a slot be released and taken again inside one tick? If not, one reader event per
    tick is equivalent to one per layer.
  - (d) `device_verify` / `device_draft` at fixed row capacities with the `valid` mask:
    acceptable?
  - (e) the per-tick `end_tick` read on draft-only ticks: can it be deferred to the next
    decision read, or must the host state follow every tick?
- **Q-T4 (lead):** E2 now, or after SG5 measures dHOST?
- **Q-T5 (C):** with `overflow` read only at decision reads, a flag raised on a draft-only
  tick voids every decision since the last read, and the run stops. Is that the intended use
  of "A TRUE FLAG VOIDS THE TICK"?

## 12a. Answers and status (2026-10-21)

The lead:

- **Q-T1:** the verify cap stays K (R_cap = B(1 + K) = 4B = m_fix). It is a parameter
  (`v_cap`), and a tick above it is refused.
- **Q-T4:** `apply_events` is not captured now. E1 (sync-free, eager) first; E2 is decided from
  SG5's dHOST.

C, at `nosi-r2-m2-fetch` 41bc238:

- **Q-T2:** YES. The trash slot is the slot after the four tail-ring slots, and the store has
  (NB + 5) x 64 rows (C + X + 5 in the Residency). Nothing else touches it.
- **Q-T3 (a):** YES, the host state is constant inside a tick. C splits S-C's `resolve` into a
  host part before the replay and a device part per layer. `sync_mirror` runs before the replay;
  `DeviceMirror.layer(l)` runs in the graph; the mirror stays on the device with one delta per
  tick (ca38648).
- **Q-T3 (b):** the superset wait before the replay is correct but removes all prefetch overlap.
  C is designing an exact device wait.
- **Q-T3 (c):** NO slot is released and taken again inside a tick. One reader event after the
  replay is equivalent and safer.
- **Q-T3 (d):** YES, with four conditions:
  - padded rows have `hist_bound` = 0 and `n_tail` = 0;
  - the live rows come first;
  - the draft pair count comes from the host's live count;
  - the index buffers have fixed length.
  The stable sorts carry OP-1.
- **Q-T3 (e):** NO deferral of the per-tick read. It can be made asynchronous: a pinned copy
  and an event, waited at the next tick start.
- **Q-T5:** YES. A sticky flag voids every tick since the last clean read. C adds a hook so that
  the twin flag rides in S-C's per-tick read.

BUILT (W0-W3, source only, this branch):

- **W0:** C's 4fe5fc9 cherry-picked (313baa2).
- **W1:** the lint, shared: `benchmarks/Efficiency/tests_cpu/r2_capture_lint.py` (e5fc694's,
  unchanged, plus `unexplained()` for documented findings).
- **W2:** `nosi/nosi/r2_tick_fixed.py` `FixedTickEngine` (a subclass; `r2_tick.TickEngine` is
  unchanged and stays the reference):
  - `choose_cap` (G_d / G_v, `cap_mode` family / max / exact) and the refusals;
  - `capacity_check` (C's condition);
  - `padded_table`, `CapBuffers`, one `upload` per tick;
  - the region `_forward_fixed` / `_layer_fixed`;
  - `read_tokens` with the flags (one transfer).
  - The trash slot follows C's Q-T2 (`from_engine` / `from_r2` add it).
- **W3 (E1):** `commit_banks` (the commit's checks on the host mirrors, including an epoch
  mirror), `commit_unchecked`, `rebuild_from_own_rows_unchecked`, `apply_events`.
- **CPU tests C1-C7:** `benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py`.

W4 (the capture) and W5 (S-C in the graph, on C's answers above) are next.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py benchmarks/Efficiency/tests_cpu/test_r2_tick.py \
        benchmarks/Efficiency/tests_cpu/test_r2_twin_unchecked.py

## 12b. W4 and W5 built (source only, 2026-10-21)

BASE: C's fetch line 41078e8 is MERGED into this branch (6a7271b). It does not contain the
tick (a35037b), so a rebase onto it would drop `r2_tick.py`. The merge brings:

- the trash slot (f321c03);
- the watched flags (0d3a94a);
- the exact device wait (41078e8);
- the device mirror (ca38648).

C's twin commit 4fe5fc9 is on both sides (my 313baa2 is its cherry-pick). The merge was clean.

W4, the captured tick:

- `nosi/nosi/r2_tick_capture.py`, `CapturedTickEngine(FixedTickEngine)`:
  - one CUDA graph per capacity, one shared pool (`torch.cuda.graph_pool_handle`);
  - per capacity: `warm` eager ticks (Triton compiles, PaddedLinear buffers), then one capture
    that is replayed, then replays;
  - the static outputs are returned as views;
  - `state_pointers` / `pointer_diff` for PTR.
- `r2_tick_fixed.py` gains `pre_region` / `post_region` (the host seams around the region) and
  `watch_flags_with(svc)`. The latter is C's `watch_flag`: S-C reads `tick.err` and every
  twin's overflow at every end_tick, and the decision read then reads the tokens only.
- `benchmarks/Efficiency/r2_tick_capture_gate.py`, the TC gate:
  - the arms TE / TF / TG / TX / TS / TR;
  - PTR and ERR;
  - NG1 / NG2 / NG3 with exact predicted first ticks (`predicted_pad_leak`, `--stale-at`,
    `predicted_frozen_capture`);
  - T.
  NG3 is a real capture fault: the region writes the token log at a host offset, which the graph
  freezes.
- `benchmarks/Efficiency/r2_tick_capture_step.sh`: the holder step (uuid file, the piece 3 pins,
  STEP_MIN 30, claim `tickcap_active`).
- `benchmarks/Efficiency/tests_cpu/test_r2_tick_graph.py`, G1-G5. A CPU `FakeGraph` replays the
  region with the host values held at their capture-time values. G5 runs the gate's own
  `fault_classes` and `compare_run` on the CPU.

W5, S-C at fixed capacity for the capture:

- `nosi/nosi/r2_residency_capture.py`, `CaptureSC`, on C's Q-T3 answers:
  - `host_part` before the replay: the row checks of every layer, `sync_mirror`, the T4 waits
    on the GPU, one copy of the fixed row buffers;
  - `device_part` per layer, inside the region: C's `device_verify` / `device_draft` / the
    mirror / wait / gather on Rv_cap / Rd_cap entries, padded entries on an INERT row;
  - `after_replay`: `read_done` with one reader event, then `end_tick`, one read.
  - On the GPU it requires `CudaDeviceOps(wait_mode="exact")`.
- `benchmarks/Efficiency/tests_cpu/test_r2_residency_capture.py`, S1-S5:
  - S1 / S2: lockstep against `ResidencySC.resolve` on F6's schedule (probe and exact wait):
    pages, state, counters, the device mirror, the stores and the copy counts are equal every
    tick;
  - S3: negative controls: a live "inert" row, no `sync_mirror`, a watched flag;
  - S4: the lint;
  - S5: `watch_flags_with`.

NOT BUILT: the tick on the Residency, i.e. SG5's seam steps 1-2. This covers the pages from the
plan instead of the ResidentStore's renaming, the history write-back to the host rows,
`decided` / `cancel_epoch`, `begin_request` / `end_request`, and the store layout of C + X + 5
slots. Until it exists, the GPU gate runs the captured tick on the ResidentStore (the trash slot
NB + 4 included), and `CaptureSC` is CPU-proven against S-C's `resolve`. `pre_region` /
`post_region` are the seams it plugs into.

CPU (inside a CPU job, from this worktree):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_tick_graph.py benchmarks/Efficiency/tests_cpu/test_r2_residency_capture.py benchmarks/Efficiency/tests_cpu/test_r2_tick_capture.py benchmarks/Efficiency/tests_cpu/test_r2_tick.py benchmarks/Efficiency/tests_cpu/test_r2_residency.py benchmarks/Efficiency/tests_cpu/test_r2_twin_unchecked.py

GPU (after the CPU tests; DRYRUN on the login node first):

    DRYRUN=1 bash benchmarks/Efficiency/r2_tick_capture_step.sh --nosi <sha> <holder job> <run id>
    bash benchmarks/Efficiency/r2_tick_capture_step.sh --nosi <sha> <holder job> <run id>

## 13. Risks

- **OP-1:** the stable `torch.sort` of `select_blocks` under capture is not yet proven here.
- **External events** for per-layer waits / reader events (7.3b, 7.4): not checked on torch
  2.6. The superset wait and the tick-end event do not need them.
- **Padding cost:** the pooling and the selection run R_cap rows at full width; G_v runs K
  twin steps on every verify tick.
- **G-b's bits depend on `m_fix`.** TE, TF, TG and REF must use the same `m_fix` (4B).
- **FAST mode** is not comparable between the eager tick (M = R) and the captured tick
  (M = R_cap). It stays REPORT.
- **The draft stand-in:** `draft_drop` is valid for exactness only (C's Q2). TC claims
  exactness and time per tick. It makes no acceptance or R-PERF claim.
- **Undefined rows:** a new comparison must never include padded rows (the capture gate's
  first failure, e5fc694).
