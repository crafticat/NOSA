# Adversarial review of the TC-R gate's use of S-C (C, 2026-10-21)

Reviewed: `/mnt/beegfs/ojerbi/nosi-r2-sg5-tcr` at `72100e3` (branch feature/nosi-r2-sg5-tcr), static reads only:
- `gate.py` = `benchmarks/Efficiency/r2_tick_capture_res_gate.py`
- `cap.py` = `nosi/nosi/r2_residency_capture.py`
- `fixed.py` = `nosi/nosi/r2_tick_fixed.py`
- `tcap.py` = `nosi/nosi/r2_tick_capture.py`
- `tick.py` = `nosi/nosi/r2_tick.py`
- `sg5.py` = `benchmarks/Efficiency/r2_tick_sg5.py`
- `res.py`, `sc.py`, `sp.py` = my `r2_residency.py`, `r2_residency_sc.py` and `r2_residency_sparse.py` in that tree
  (merged from c6ef37d)

At review time the TC-R worktree had an UNCOMMITTED change to `cap.py` (`fixed_decision` and `live_view`). Section 3
reviews that change too.

## Verdict

- The default gate (`--service sc`, S-C's dense record, which carries the registered TR0-TR3) has NO blocking finding.
  - PublishingGather's (index, gen) mapping and stream order are right.
  - The exact wait cannot see a stale or an early generation in any path the gate takes.
  - The arrival rule (12k) holds in both arms.
  - Policy E's in-flight kind is passed and waited correctly.
- `--service sparse` (B1) is BLOCKED at `72100e3`. The record hook reads `vdec.bh` at full capacity and the
  decision fields at the live size. The engine's CPU job 2191020 found this (128 live vs 192 entries). The
  uncommitted fix is right, but it must be committed and CPU-tested (R6) before any sparse GPU arm.
- Four non-blocking notes (section 5): one latent gap that is mine (T4's main-stream wait list has no producer), a
  copy-count undercount, and two dependencies that the gate relies on but does not state.

## 1. PublishingGather (gate.py:380-408): the (index, gen) mapping, the stream order, stale generations

MAPPING. Both sides use the same formula over the same board shape, so they agree:
- Publication: `board.flat(tr.layer, tr.b, tr.h, tr.slot)` = `((l B + b) H + h) npos + slot` (sc.py:591-592).
- Wait list: `((l B + bb) H + hh) npos + slot_w` (sc.py:601-615).
- `npos` is the service's `n_hist` = C + X (sc.py:827, GenBoard). GatherTransport's `n_slots` is C + X too
  (gate.py make_arm).
- The value is `int(tr.gen)`, the generation of the take. Each layer has its own index, so generations never mix
  across layers.

STREAM ORDER (`issue`):
1. `super().issue` enqueues on `self.side`: the waits on `hist_ev` and `reader_ev` (T4, T3), the load-map copy, the
   two h2d kernels, and an event.
2. Under `torch.cuda.stream(self.side)`: the pinned copy of (index, gen) and the `index_put_` into `board.t`.
3. A NEW event, recorded on `self.side` after the board write, replaces `tr.ev` for every transfer of the call.

So:
- The board write is stream-ordered after the rows of EVERY transfer of the call. That is later than needed for a
  single transfer, but never earlier.
- `done()` / `force()` (an event query / synchronize) now mean "rows landed AND published". The arrival rule's
  force at t + D waits for the publication too. The state does not change; only the forced / held counts can move.
- The volatile spin (sc.py:717) sees the board value only after the side stream ran the h2d kernels to completion.
  The rows are then in device memory before the main stream's next kernel (the gather / attention) reads them.
  This is the gilfree design (note 12f), same pattern.
- The pinned staging blocks and the device index tensor `d` are allocated and used on the side stream only, so
  their reuse is stream-ordered. `board.t` is never freed.
- `index_put_` on the inference-mode board runs inside `run_rows` (inference mode) in both arms. There is no
  feeder thread here, so the SG5 p2 mode problem cannot occur.

STALE GENERATION. The kernel waits until `board >= want`, then sets `err` if `board != want`. Generations of a slot
only grow (`Stream.take`: `gen += 1`), and a slot in flight is never re-taken (T3). I checked every path in which
the board value of a slot differs from the slot's generation:

| path | board | wait? | safe |
|---|---|---|---|
| draft prefetch in flight (gen g), not landed | < g | yes, spins until the publication | yes |
| landed and published, HELD by the arrival rule (not admitted) | = g | yes (still INFLIGHT, `cur`): passes at once | yes |
| urgent verify take (gen g, main-stream gather, NEVER published) | < g | never: a landed urgent transfer is admitted at the NEXT poll regardless of the lag (`res.py:258`: `_landed` skips `_arrived`), so it is resident before any later tick's mirror | yes, by that coupling |
| prefetch superseded by Policy E (gen g), then the urgent take (gen g + 1, unpublished) | = g | the urgent gather waits for g (wait_list's `evict_infl` / `ev_gen`); nothing ever waits for g + 1 | yes |
| a later draft prefetch into that slot (gen g + 2) | g until published | yes | yes (the board lags by design) |
| drop_unowned (old request's prefetch) | its gen when landed | no (`cur` 0) | yes |

- A board AHEAD of the wanted value would need a later transfer to land in a slot that is still INFLIGHT, which T3
  forbids.
- Each arm (and each NG arm) builds its own service, so its own board. No board is shared.
- The one residual risk is the coupling in row 3. If `_landed` ever went through the arrival lag, a verify could
  wait for an unpublished urgent generation. The bounded spin then sets `err`, the watched `devops_err` voids the
  tick, and INV fails. It would fail loudly, never hang or pass silently.

Also checked:
- Deadlock. A transfer's `reader_ev` is the slot's last reader at take time (end of tick t, recorded before the
  take). Its `hist_ev` was recorded in an earlier `apply_events`. Neither can be recorded after the replay that waits
  for the transfer.
- Liveness (gilfree). `host_part` calls `_submitted` for every layer before the replay (cap.py:120-123), because the
  capture-time call is not replayed. For PublishingGather it is a no-op: `tr.ev` is set at issue.

## 2. The arrival rule with arrival_lag 2 (12k) in CaptureSC

The rule: a transfer issued in tick t is admitted exactly at `poll(t + D)`. It is forced there if it is late, and
held (counted) if it landed early.
- `issue_tick` is stamped at the take (`res.py:482`) with `tick_no`, inside `apply_tick`, so at t.
- `tick_no` advances only in `Residency.end_tick` (`res.py:574-580`), which S-C's `end_tick` calls after
  `apply_tick` (sc.py:1058). In TG-R that is `after_replay` -> `svc.end_tick()` (cap.py:230). In TE-R it is
  `ResidencyBackend.end_tick` (tick.py). One advance per tick in both arms.
- The ONLY poll is `backend.begin_tick()` at the start of `run_rows`, in both arms (fixed.py run_rows,
  tick.py:887-888). S-C's resolve does not poll (`poll_in_resolve` False), and `host_part` / `after_replay` call no
  poll and change no `tick_no` (cap.py:98-156, 211-230).
- So a tick-t transfer is held at the poll of t + 1 and admitted (forced if late) at the poll of t + 2, in both arms.
- Urgent verify transfers bypass the lag (row 3 above), in both arms alike.
- A void tick (a set flag) raises in `apply_tick` before `Residency.end_tick`. Then `tick_no` does not advance and
  nothing is issued, in both arms. INV requires no void tick anyway.
- The forced / held counts depend on timing, so they are correctly NOT in `svc_digest`. The gate reports them per
  arm (gate.py:545-547). Use them when reading TR1: an arm that runs ahead forces more host syncs at t + D.

Two other timing dependencies of the state are closed in both arms:
- `poll_releases` (RELEASING -> FREE when `ev_done(last_reader)`) and the queued drops (`ev_done(key_reader)`)
  query CUDA events. Every reader event of tick t is recorded before tick t's `end_tick`. `apply_tick`'s one D2H
  read synchronizes the main stream after it (sc.py:1076, a blocking copy). So every reader event is complete at the
  next poll, and FREE timing is a function of the schedule.
- This needs a D2H read at EVERY tick. That holds because the watched flags (devops_err, the twin overflows, and in
  TG tick_err) are never empty, so `parts` is never empty. State this dependency in the gate's docstring: a service
  with no watched flag and a tick with no rows would skip the read.
- TE-R records one reader event per layer, TG-R one per tick (Q-T3 (c)). Readers are not in `svc_digest`, and both
  complete by the next poll, so this does not change the compared state.

## 3. The B1 hook: `_layer_record` run eagerly after the replay (cap.py:223-226)

The question: can anything in the replay, after layer l, change what layer l's record reads?

What `_layer_record` (sp.py:251-268) and `sparse_layer_record` (sp.py:114-183) read:
- (a) layer l's decisions;
- (b) the pin table `pins[l]`;
- (c) the mirror table `dmirror.code[l]`;
- (d) the per-tick key ids and epochs (`extra`);
- (e) `cis[l]`, read by `draft_bias_fill`, which also WRITES `store_b[l]`.

(a) The decisions:
- `fixed_decision` copies `.t` and `.counters` into PER-LAYER buffers (`vbufs[l]`, `dbufs[l]`). No other layer and
  no later op of the replay writes them.
- BUT at 72100e3 the verify decision's `bh` (the (b, h) of each entry) is NOT copied. `live_view` slices `.t` to the
  live N = nv H k entries and leaves `bh` at the full Rv_cap H k (cap.py:250-259).
- `sparse_layer_record` mixes the two: `bb, hh = vdec.bh` (sp.py:129), then `pk = kid * TWO32 + epoch_l[bb]`
  (sp.py:137, kid at live N, bb at full length). That is a shape mismatch whenever nv < Rv_cap, which is nearly every
  verify tick.
- The engine's CPU job 2191020 hit it (128 live vs 192 entries).
- BLOCKING for `--service sparse` only. The dense S-C consumer reads only `.t` and `.counters` (sc.py:1061-1115), so
  the default gate is unaffected.
- THE UNCOMMITTED FIX IS RIGHT:
  - `fixed_decision` now keeps `bh` in fixed buffers, made at the first (eager) call and copied at every later call,
    the capture included.
  - `live_view` slices `bh` to the same live N.
  - `page` / `num_pages` are sliced to the live rows; no consumer reads them.
  - The bh buffers sit outside the graph pool, like `.t`.
  - Also add them to `state_pointers` (tcap.py:171-173 lists `.t` and `.counters` only), so PTR covers them.
- A latent point at 72100e3 that the fix also removes: the capture-time `bh` tensors lived in the graph pool, kept
  alive only through the decision object in `self.vdec[l]`.

(b) The pin table:
- It is outside the graph and written only by `_layer_record` of its own layer.
- Its epochs change only outside the tick: `_drop` at the poll, `end_request`, and `_void_pins` in `apply_tick`
  after the record.
- The eager path (inside `ResidencySC.resolve`) and the captured path (after the replay) apply the same updates in
  the same order: layer by layer, verify pins (kind 1) and then draft pins (kind 0).

(c) The mirror table:
- It is written only by `dmirror.apply` in `sync_mirror`, which runs in `host_part` before the replay.
- In the region, `device_verify` / `device_draft` change only the derived per-layer copy from `dmirror.layer(l)`
  (fresh tensors).
- `after_replay`'s `read_done` calls make no journal entry and no dirty mark (plans with no reads and no looks), so
  nothing syncs between the replay and the record.

(d) The key ids and epochs:
- They are taken at the same point in both paths: after the mirror delta, before any decision. TE: `_tick_rows`;
  TG: `host_part`, cap.py:150-154.
- TG uploads them in a SECOND host-to-device copy that `row_uploads` does not count (note 5.2).

(e) The bias fill:
- It moves from before layer l's attention (TE) to after the replay (TG). This is safe: a granted draft-take slot
  was FREE at the tick start, and no read of this tick touches it.
  - Draft rows read READY slots only.
  - Verify pages are ready, in-flight or urgent slots, and `device_verify` removes its urgent slots from `m.free`
    before `device_draft`.
  - Tails are ring slots outside the pools; other layers are other pools.
- `cis[l]` (`twins[l].total_cis[:B]`) at historical blocks (< hist_bound <= T) is not written by the tick's advance.
  That advance writes the positions of block T and later. So the fill copies the same values in both paths.

So: with the bh fix, nothing in the replay can change what layer l's record reads after layer l. The eager record
equals the in-resolve record.

## 4. `device_part(..., inflight=svc.policy_e_inflight)` (cap.py:175-177)

It is correct and matches `ResidencySC.resolve`:
- Same flag, same `others`: the draft ENTRIES. A padded entry points at the inert row with hist_bound 0, so it
  selects nothing in `_selected_mask`.
- The Python bool is a constructor constant, baked into the graph at capture.

Under capture:
- A superseded prefetch is in `wait_list` (`evict_infl`, `ev_gen`). The captured exact wait spins until that
  prefetch is published before the captured urgent gather writes the slot (T3 for writers). Both run on the main
  stream, in the graph's order.
- A supersede of a prefetch that has landed but is HELD (arrival rule) is a candidate in both arms. Its board value
  is already g, so there is no spin.
- The host's `_evict_inflight` at `apply_tick` forces the transfer only if its side-stream event has not completed.
  After the D2H read that is at most the retire time of the board write: a host sync, not a state difference.
- Padded VERIFY entries are inert. They point at row R_cap: rb 0, hb 0, no tails. All their entries are invalid, so
  they make no take, no protect, no wait, no gather, and no K5 / Policy E rank.

## 5. Non-blocking notes

1. MY LATENT GAP: `_hist_unwaited` has no producer.
   - `ResidencySC` declares it (sc.py:855), and `CudaDeviceOps.gather` (sc.py:805-807) and `cap.py:125-129` wait on
     it. But `history_written` (res.py:191) never appends to it.
   - So T4 for the URGENT gather (zero-copy reads of the pinned host rows on the main stream) holds today only by
     stream order. `commit_block` (tick.py:923-934) issues its device-to-host copies on the current stream in
     `apply_events`, before the next tick's region or replay on that stream.
   - That is correct for TC-R, but the guard is empty: a write-back moved to another stream would not be waited.
   - The fix is on my side: S-C's `history_written` appends the event on CUDA. I will send it as a separate commit
     on GO; it does not change TC-R's result.
2. COPY COUNT. For `--service sparse`, TG makes TWO host-to-device row copies per tick (the row buffers and `extra`)
   and counts one. TE makes one. The `copies` digest still compares equal. Count `extra` in `row_uploads`, or report
   it.
3. UNSTATED DEPENDENCY, the D2H read per tick (section 2). EQ's FREE timing relies on `apply_tick`'s read at every
   tick, which the watched flags guarantee. Say so in the gate's docstring.
4. UNSTATED DEPENDENCY, the urgent bypass (section 1, row 3). The exact wait never waits for an urgent
   generation, because landed urgent transfers skip the arrival lag. A change there would void ticks (err).

## 6. What blocks the GPU run

- `--service sc` (the registered TR0-TR3): nothing.
- `--service sparse`: commit the bh fix, add the bh buffers to `state_pointers`, and pass the CPU R6 run first.
