# Adversarial review of B's B2 design (C, 2026-10-21)

Reviewed: `/mnt/beegfs/ojerbi/nosi-r2-m2-own2/docs/R2_OWN2_B2_DESIGN.md` at `1676028`
(sha256 cc8ad9e097f5ad33e9111f906e7aada94b3c0477adcbdc6b85dc59b4d6db4237).

Code cited: this branch at `c6ef37d` (feature/nosi-r2-m2-sparse).
- `ownership.py` = `nosi/nosi/kstep/ownership.py`, vendored byte-identical (91e6e47).
- `res.py` = `nosi/nosi/r2_residency.py`.
- `sc.py` = `nosi/nosi/r2_residency_sc.py`.

The fetch head `30b404b` has the same ownership code; its `res.py` lines differ only by the arrival rule.

The verdict in short:
- The layout (S-C's mirror made authoritative) is sound.
- Five points are WRONG or MISSING and must be fixed before code:
  (1) the reader-as-tick watermark is not the vendored reader model;
  (2) I8's recency is not equivalent when a stream has two draft rows in one layer;
  (3) I1 misses the in-flight dedup;
  (4) the admit / poll branches that reset readers are not stated;
  (5) six vendored rules have no counterpart: T4 write-back, cancelled keys, the late-protection guard, the key-slot
      reuse, end_request, cancel_epoch.
- The equality contract never reaches five transition families (end_request, cancel, two draft rows per stream,
  void ticks, non-monotone readers).
- The 10-25 ms cost is likely low by 2-4 x on one CPU core (the segment sorts and the object paths).

## 1. The invariants I1-I9 against the vendored code

| # | vendored enforcement (file:line) | B's predicate | verdict |
|---|---|---|---|
| I1 | `Stream.check` ownership.py:174-195 (no block in two maps, one slot per resident block). The DEDUP: `res.py:453` refuses a take of a block that is resident OR in flight for the current owner (`_in_flight` res.py:509-517 via `Stream.pending`, ownership.py:266-267). Admit refuses an arrival of a resident block, ownership.py:343-345. | resident slots per block <= 1 | WEAKER. Count resident and current-owner INFLIGHT (`cur`) slots together: <= 1 per (stream, block). A previous owner's INFLIGHT slot (cur 0) may coexist; exclude it. |
| I2 | K1 after EVERY insertion: `admit` -> `_evict_over_c` (ownership.py:347-351, :280-296), `acquire(evict=False)` + `evict_to_c` per (row, head) (res.py:686-706, ownership.py:297-303); `Stream.check` :187-188 | `(code == LRU).sum(-1) <= C` at the tick's end | WEAKER but sufficient for the end state. Also check it after `poll` (admissions evict one entry each). |
| I3 | `Stream.check` ownership.py:192-195; `unprotect` releases PO / VRES at their last reference, ownership.py:213-237, res.py:298-305 | pin != 0 for PO / VRES | EQUIVALENT. |
| I4 | `LayerPool.can_take` ownership.py:419-427 with `borrowed` :409-410 (usage counts EVERY non-FREE slot: INFLIGHT, LRU, PO, VRES, RELEASING, :135-136); `first_free` :429-433; `_take` res.py:438-489 (the K5 and first_free check of a device decision :456-465) | borrow by rank, borrowed_at_take by exclusive cumsum | EQUIVALENT for K5. Inside a tick, borrowed changes only by borrowing takes: a Policy E reuse, an evict_to_c release (LRU -> RELEASING) and an acquisition keep usage. State this proof in the note. MISSING: `first_free` (the q-th take of a stream gets its q-th FREE slot); B leaves it to the lockstep. It is one rank compare: make it a predicate. |
| I5 | T3 / K4: `poll_releases` ownership.py:435-452 (RELEASING -> FREE only when `last_reader` is done); `take` keeps `last_reader` and resets `first_reader` (:259-261); `_take` forwards the slot's previous last reader to the transfer (res.py:478); `_evict_now` host-waits a pending reader (res.py:399-418); `_evict_inflight` forces the superseded write (res.py:363-388) | FREE only in poll ...; Policy E with a wait / landed | WEAKER as written, and its text is WRONG. Policy E makes a slot FREE in the MIDDLE of a tick (`_evict_now`, `_evict_inflight`), not only in poll; the predicate must allow exactly those two. Add: a take keeps `rd_last` (T3 forward) and sets `rd_first = -1`. |
| I6 | `admit` ownership.py:322-361: generation (StaleGeneration :328-330); state, block and slot owner (:331-333); `pending` removal (:334-335); `seq = None` (:336); a stream owner != uid -> RELEASING with `last_reader = None` (:337-342); resident already -> violation (:343-345); `first_reader = None` (:346); draft in demand -> LRU + ONE `_evict_over_c` (:347-351); protected -> VRES (:352-356); else RELEASING with `last_reader = None` (:357-360). Order: res.py:245-264 (by (layer, b, h, seq)), with the arrival rule res.py:265-282. | INFLIGHT, gen, block, cur | EQUIVALENT for the checks, INCOMPLETE for the outcomes. B must state the two `rd_last := -1` resets (else a dropped slot frees one poll late: a lockstep difference), the `rd_first := -1` at every admission, and the arrival rule (admit only at `issue_tick + D`, force if late, hold if early). |
| I7 | `_verifier_candidates` res.py:320-344, `_unprotected_prefetches` :353-362 (`_in_flight` = the CURRENT owner's transfer, `kind == DRAFT`, not `_landed`, unprotected, not selected), `_evict_now` / `_evict_inflight`; S-C's host check `_host_candidates` sc.py:1135-1162 | as listed | WEAKER in one point: an in-flight victim must be the CURRENT owner's transfer (`cur`). COMMON-MODE: "not in the layer's selection bitmap" uses the device's own bitmap; a wrong bitmap is caught only by the lockstep (v1 uses the host's selections). |
| I8 | `_draft_recency` res.py:686-706: per draft row IN ORDER, per head, the row's READY blocks in ascending block id (touch / acquire); then `evict_to_c` for THAT stream (:705); a block read but no longer resident at the recency raises | stamps increase; victims = the `n_lru - C` least stamped after the stamps | NOT EQUIVALENT in general. With two draft rows of ONE stream in one layer, row 1's `evict_to_c` can move a block that row 2 read to PO; row 2 then acquires it back to MRU. A single eviction after all rows gives a different LRU. I8 must run per row-index round (at most K rounds), or prove that the flattened round never has two draft rows per stream and layer, and refuse such a record. |
| I9 | `take` ownership.py:259 (`gen += 1`); admit's generation check :328-330 | gen changes only at a take, +1 | EQUIVALENT. |

### 1.1 Vendored rules with NO counterpart in I1-I9

1. THE READER MODEL. B stores readers as TICKS against one watermark (`rd_last <= done_tick`).
   - The vendored code keeps one event OBJECT per reader and asks each one (`ev_done`, ownership.py:443-444).
   - The two agree only when every reader completes in tick order AND every host wait completes all earlier
     readers. On one main stream this holds on the GPU.
   - It does NOT hold in the CPU worlds. `Be.host_wait` completes one `Ev` only; scripted tests complete readers
     out of order (F12e: a pending reader of an earlier read while later ones are done).
   - Two consequences:
     (a) the lockstep must use monotone harnesses only, and say so;
     (b) B2 needs a "host wait of tick t" operation, which the force paths use: `_evict_now` res.py:410-415, the
         transport force, the arrival rule's force res.py:276-281.
2. T4, THE HOST WRITE-BACK. `_hist_ev_for` res.py:197-201 refuses a take of a block >= `hist_from` that has no
   `history_written` event, before any change (res.py:449). B2 has no write-back state.
3. CANCELLED KEYS. `cancel` res.py:768-785 (the key is refused from then on, its references queued); `cancel_epoch`
   :786-798 (needs `keys_seen`, filled by `_check_rows` :540-541); `_check_rows` :534-535 refuses a row of a cancelled
   key. B2 has `cancel(b, key_slot)`, but no refusal and no `keys_seen`.
4. THE LATE-PROTECTION GUARD. `_check_rows` res.py:538-539 refuses a row whose (key, l) release is already queued.
   No counterpart.
5. KEY-SLOT REUSE. B maps keys to `KEYS` bit slots. A slot may be reused only when no pin bit of it remains at ANY
   layer, its `rel_q` is empty and its `key_rd` readers are done. Otherwise a new key aliases the old key's pins.
   Needed: an invariant, and a refusal past KEYS (B states the refusal).
6. END_REQUEST. `end_request` res.py:220-234 + `end_owner` ownership.py:363-384:
   - LRU / PO / VRES -> RELEASING ("end"); a PO entry only while its slot is still PO (:370-372);
   - the maps, `prot`, `pin_kind` and `pending` are cleared; the owner becomes None;
   - in-flight transfers stay in flight and are dropped at their arrival (`cur` flips);
   - `hist_ev` / `hist_from` are removed;
   - queued keys of the uid become cancelled, and the queue is filtered.
   B's `end_request(b)` names none of these.
7. `begin_request` res.py:203-218: it refuses a slot that is still owned. No counterpart.
8. THE TRASH SLOT AND THE TAIL RINGS. Slots `C + X .. C + X + 3` (tails) and `C + X + 4` (trash, `trash_slots=1`) are
   outside every pool (res.py docstring, the store layout). B2 must refuse a record naming a slot >= NPOS. That is
   trivial, but state it.
9. THE POLL ORDER. res.py:283-297: arrivals, then the queued drops, then `poll_releases`. The order matters: an
   urgent verify block admitted VRES and then unprotected keeps its `last_reader` (the verify read). An admission
   after the drops would be `drop_vres_unprotected` with `last_reader = None` (FREE one poll earlier). B lists this
   order; make it a contract, with a planted swap (section 3).
10. THE CLASS "v" / "ad" (ownership.py:43-47, `take(cls=...)` res.py:480). In mode m1 nothing checks it, but
    `_state` compares it. An acquired VRES slot keeps "v". B2 must keep it, with no re-tag.
11. THE METRICS that `_counters` compares: k43 (first looks per (key, l), Metrics.look / look_one res.py:882-927),
    k46, `transferred` per (uid, l, h, block) (res.py:863-868, the refetch flag), `refused`, `masked`, the counts.
    B puts them in "lockstep only". B2 must still PRODUCE them, as arrays.

## 2. The equality contract: transitions that no listed schedule reaches

Listed: F6, F12b, F20b, F21b, F22, as harnesses that only `decided()` (no `end_request`, no `cancel`) and complete
readers two ticks late in order.

| transition / path | reached? | smallest schedule that reaches it |
|---|---|---|
| FREE -> INFLIGHT (draft, verify), INFLIGHT -> LRU / VRES, LRU -> PO / RELEASING, PO / VRES -> RELEASING (unprotect), RELEASING -> FREE, PO / VRES -> LRU (acquire) | yes (F6, F12b, F20b): add counters as preconditions (evict_po, acquire, admit_vres > 0) | - |
| LRU -> FREE (Policy E), INFLIGHT -> FREE (superseded) | yes (F20b, F21b) | - |
| forced and held arrivals | yes (F22) | - |
| INFLIGHT -> RELEASING at arrival for an ENDED request (drop_unowned, `rd_last := -1`), the `cur` flip, end_owner's releases | NO | the scripts F13 (`test_F13_a_transfer_of_an_ended_request_is_never_read_by_the_next`) and F14b, plus a final land: a draft take that stays in flight (force_only), `end_request`, `begin_request` of the next uid, the arrival dropped. 3 ticks. |
| `drop_vres_unprotected` | NO, and UNREACHABLE in R2's schedules: an urgent block is landed in place and admitted at the next poll BEFORE that poll's drops | a scripted admit (`Stream.admit` with demand {verify} and no protection), or v1 with a cancel between a verify take and its arrival |
| cancel / cancel_epoch / a refused cancelled key / the late-protection guard | NO in the lockstep schedules; only in scripted v1 tests (F2, F3: `svc.cancel`; F11: `cancel_epoch`) | F3's script (a late transfer of a cancelled cohort lands in its own slot) and F11's (the end of an epoch), run through B2's lockstep |
| two draft rows of ONE stream in one layer (I8's interleaving, a PO acquisition between rows) | NO (`_sched`, `_sched_pressure`, `_sched_split` have one draft row per request and tick) | one stream, C 2, X 0: tick 1, k1 drafts [0, 1, 2] (block 0 -> PO, protected); tick 2, two draft rows of the request read [0] and [1] |
| a void tick (overflow; a watched flag) | NO | F12f (overflow, 1 tick), F16 (the twin flag) |
| T4 refusal (no write-back registered) | NO in the lockstep schedules (`world()` sets no `hist_blocks`); F11 scripts it | F11's script (history events), or `begin_request(b, u, hist_blocks=4)`, then a take of block 5 |
| non-monotone reader completion | NO by design (see 1.1-1) | F12e (a pending earlier reader). B2 must REFUSE such a harness, not diverge silently. |
| the trash slot | NO | F15 |

## 3. The 14 planted defects

| B's defect | caught by (B) | my check |
|---|---|---|
| draft take into a non-FREE slot | I5 | yes. "at a slot other than first_free": only the lockstep. Make it P with the rank check (I4 row). |
| a grant past the extras; a draft into the reserve | I4 | yes |
| Policy E LRU: pinned; selected; not the least recent | I7 | yes for pinned / not least recent. "Selected" is caught only if the bitmap is right (common-mode, 1-I7); the lockstep catches a wrong bitmap. |
| Policy E in-flight: protected / landed; not the oldest; while an LRU candidate is unused | I7 | yes. ADD: a victim of a previous owner (cur 0) must be refused. |
| refusal dropped (the take happens) | I4 when over, else lockstep | yes |
| demand merge dropped | lockstep | yes: B projects `demand` per slot, so it differs at the same tick, not only at the arrival |
| acquisition dropped / wrong source | lockstep; I2 / I3 | yes |
| overflow ignored | "the void-tick rule" | the rule IS the thing the defect breaks; what catches it is the LOCKSTEP (the vendored tick refuses, B2 applies). Say so. |
| new pin dropped | I3; lockstep | yes (pins of non-resident blocks only in the lockstep) |
| draft stamps permuted in a row | I8 | yes |
| evict_to_c victims | I8 | yes for one row per stream; not decidable for two rows without the per-round I8 (section 1, I8) |
| read bitmap slot missing | lockstep | yes at the same tick (`rd_last` compare) |
| poll frees a slot with a pending reader | I5 | yes, under the monotone-reader model only |
| arrival stale / old owner | I6 | yes |

MISSING KINDS (each with its check):
1. A take of a block that is already resident or in flight for the owner. I1 extended (resident + cur INFLIGHT).
2. A take without a registered host write-back (T4). A new predicate from `hist_from` / `hist_ev`.
3. A row of a cancelled key, or a row after its (key, l) release was queued. New predicates (1.1-3, 1.1-4).
4. A key slot reused while its pins remain at some layer. The key-slot invariant (1.1-5).
5. The poll order swapped (drops before arrivals). The lockstep: VRES -> RELEASING timing and `rd_last`.
6. A `drop_unowned` / `drop_vres_unprotected` that keeps `rd_last`. The lockstep: it frees one poll late.
7. A take that drops `rd_last` (the T3 forward lost). The lockstep (`rd_last`); the transport's reader wait.
8. Two draft rows of one stream with one batch eviction. The per-round I8.

## 4. The cost estimate (10-25 ms per round tick on one CPU core)

Where I expect it to break:
- THE SEGMENT SORTS. Policy E and evict_to_c "over the touched streams" are 3.5 k + 3.3 k streams x 128 slots, about
  0.87 M int64 keys. torch.sort on one CPU core costs about 40-100 ns per element: 35-90 ms, not 3-8 ms.
  - Sort only the streams that need it: Policy E's victims (about 470 per tick) and the streams whose LRU exceeds C
    after the stamps.
  - Use a top-k of the few victims per stream.
- THE FULL-TABLE PASSES. 8 passes over 22.5 MB of mixed int8 / int32 / int64 columns are about 180 MB of traffic,
  plus the dtype casts that torch inserts. Expect 15-40 ms, not 5-12 ms. Fuse the passes, or keep one int64 slot word.
- THE SEQUENTIAL PARTS. I8 per row-index round (K rounds), and poll's admissions with one K1 eviction each per stream
  in seq order. They are segment scans. Each round costs the dispatch of a few dozen ops (about 0.5-1 ms).
- THE OBJECT PATHS OUTSIDE B2 dominate until Q-B3:
  - the transport still takes Python `Transfer` objects (B's 0.15-0.2 s per tick);
  - `poll_arrivals` iterates them;
  - the metrics' `lifetimes` / `transferred` are dicts per take (about 6.5 k x 1 us).
- My estimate for Phase 1 on one amd core: 40-80 ms for B2 itself, plus the issue path. So B's falsifier (> 25 ms)
  is likely to trigger. Measure the components separately in B2-T (sorts, passes, rounds, records), so the falsifier
  names what to move first.
- A measured reference: my B1 microbench (`test_R5_microbench_host_apply_dense_vs_sparse`; two runs on the shared
  amd node: x16 dense verify 4037 / 3410 ms, draft 1804 / 1737 ms; sparse verify 4269 / 3300 ms, draft 1755 /
  896 ms). The vendored per-record apply costs several microseconds. Any per-record Python in B2 multiplies about
  9.3 k records by that.

## 5. What B2 should change before code

1. Make the reader model explicit: tick watermarks with monotone completion, a "host wait of tick t" operation, and
   a refusal of a non-monotone harness. Or keep per-slot event ids with a completion set.
2. I1 over resident + current-owner INFLIGHT. I5 with the take's first_free rank, the Policy E mid-tick FREE, and the
   `rd_last` forward. I6 with the reader resets and the arrival rule. I7 with `cur`. I8 per row-index round.
3. Add the T4, cancelled-key, late-protection and key-slot invariants, and the end_request / begin_request contract.
4. Extend the lockstep with the end_request, cancel_epoch, two-draft-rows, void-tick, T4 and F15 scripts (section 2),
   and with the missing planted kinds (section 3).
5. Split the cost benchmark by component; sort only the streams that need it.
