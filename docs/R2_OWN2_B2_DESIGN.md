# B2: an array-backed ownership state beside the vendored one (design rev 2)

Branch `feature/nosi-r2-m2-own2`. Status: DESIGN rev 2. The lead gave the GO with the build order B2-T (the component
microbench) first, then `apply_tick` with check-then-write, then the CPU lockstep.

- Rev 1: `1676028` (sha256 cc8ad9e0...4237, job 2191001).
- Rev 2 answers C's adversarial review (`nosi-r2-m2-sparse` `32e88c2`, `docs/R2_OWN2_B2_REVIEW_C.md`) point by point
  (section 0). It also folds in:
  - C's answers in note 12l (Q-B1 record format, Q-B2 recency, Q-B4 key bound);
  - E's `issue_desc` (Q-B3, `feature/nosi-r2-gilfree-worker` `6397247`).

Code cited at own2 `2eb02c2`, which holds C's `7392ac6` code:
- `res.py` = `nosi/nosi/r2_residency.py`;
- `own.py` = `nosi/nosi/kstep/ownership.py`, vendored byte-identical at `91e6e47` (`res.py:4`, test F9);
- `sc.py` = `nosi/nosi/r2_residency_sc.py`;
- `sparse.py` = `nosi/nosi/r2_residency_sparse.py`;
- `core.py` = `nosi/nosi/r2_spec_core.py`;
- `test.py` = `benchmarks/Efficiency/tests_cpu/test_r2_residency.py`.

**Why B2.** C's B1 microbench shows no host speedup with the sparse record on the vendored objects. At B 64, 32 layers,
`apply_tick` takes, per tick (verify / draft):
- dense: 4037 / 1804 ms;
- sparse: 4269 / 1755 ms.

The cost is the per-record apply into the Python `Slot` / `Stream` objects. B2 keeps the same rules (K1-K5, T3, T4,
Policy E, O1) on arrays. The vendored objects stay the reference of the lockstep.

**Geometry** (p3, `R2_SG5_CAPTURE_PLAN.md:177`, `:187`): L 32, B 64, H 2, C 96, X 32, R_ver 16, 16K context.
- NPOS = C + X = 128 slots per stream (`res.py:160`); NBLK = n_host // 64, about 256 (`r2_tick_sg5.py:302`).
- That is 524,288 slots, and the verify read bitmap is 0.52 M bits = 64 KB. The capture plan's "2.2 M bits"
  (`:169`) fits H 8.

**The rows of a tick** (`core.py:136-155`):
- One draft row per request and tick. `finish_tick` refuses two (`core.py:157-164`).
- A cohort contributes one verify row per position `c.a .. c.b`, all with the same b and key (`core.py:146-150`).
- So at p3's mean (51 verify rows), the verify rows come from about 13 requests, each with up to K + 1 rows.

## 0. Rev 2: the answers to C's review

### 0.1 The five points that must be fixed

**(1) The reader model. ACCEPTED, and solved without a monotone assumption.**

B2 does not store readers as ticks. It stores READER EVENT IDS:
- Every `read_done` call (`res.py:743-761`) gets the next id e: one per layer per tick in v1, one per tick in the
  captured path.
- A host registry maps a live id to its event object and to E's mark (tick x L + layer).
- `rd_first`, `rd_last`, `key_rd` and `hist_id` hold ids; -1 = None, which counts as complete (`ev_done(None)`,
  `kstep/backend.py:27-34`).
- **Completion.** `lo` = the oldest incomplete id, and every id below `lo` is complete. `done_win` (bool) holds the
  ids in [lo, next_id). At each poll, the host refreshes `done_win` with `ev_done` on each live event object. That is
  about 32-100 queries per tick.
- Then `done(e) = (e < lo) | done_win[e - lo]`: one gather per full-table check.
- Completion in ANY order is exact: F12e's pending earlier reader, and `Be.host_wait` completing one `Ev`
  (`test.py:56-71`). So B2 does not refuse a non-monotone harness.
- **Host wait.** `wait_reader(e)` calls the service's `host_wait(ev)` on that one object, then re-queries only that
  event. The force paths use it:
  - Policy E's pending last reader (`res.py:410-415`);
  - the urgent gather's gates (`res.py:616-622`);
  - the arrival rule's force (`res.py:276-281`) goes to the transport, as today.

**(2) I8 with two draft rows of one stream in one layer. ACCEPTED: per-row rounds.**
- `_draft_recency` (`res.py:686-706`) handles each draft row IN ORDER: its READY blocks in ascending block id
  (touch / acquire), then `evict_to_c` for that row's streams (`res.py:705`, `own.py:297-303`).
- B2 runs the recency in ROUNDS. Round r is the r-th draft row of each request in row order (`rank_in_group` over b,
  `sb.py:75`).
  - Inside a round, each READY record is a touch if its slot is LRU, an acquisition if PO / VRES, and an
    `OwnershipViolation` otherwise (`res.py:696-704`).
  - Each round ends with `evict_to_c` on the round's streams.
- The number of rounds = the largest number of draft rows of one b (a host scalar).
- R2's planner gives 1 round (`core.py:157-164`). C's two-row script tests round 2.
- **A finding (B2-T test R3):** one eviction after all rows reaches the SAME end state, under two conditions. The pins
  do not change during the recency, and phase A pins every block a draft row reads (`res.py:668`). The reasons:
  - the final LRU is the C entries with the largest final stamps either way;
  - an entry that a round evicts and a later row reads is pinned, so it becomes PO and is acquired back;
  - every other victim leaves in both orders, with the same pin test.

  So the "one batch" defect (C's missing kind 8) changes only the transition counts (`evict_po` / `acquire`), not the
  state. The test asserts this on 300 random one-to-three-row cases. C reached the same result independently
  (correction 1 to its review; fixture `o1_two_rows`: rounds acquire 2 + evict_po 2, batch acquire 1 + touch 1 +
  evict_po 1; the stamps and mirror marks are equal).
- **Decision: B2 REPORTS the journal counts** (touch, acquire, evict_po, evict_rel per tick: `recency()` returns them).
  So B2 keeps the per-row rounds: with them, the counts equal the vendored journal's. They are REPORT counts: `_counters`
  (`test.py:607-611`) does not compare them. R2 runs one round per tick, so the rounds cost nothing there.

**(3) I1 without the dedup. ACCEPTED.**
- I1 counts the resident slots (LRU, PO, VRES) and the current owner's INFLIGHT slots (`uid == owner`) together:
  at most 1 per (stream, block).
- A previous owner's INFLIGHT slot may coexist (`res.py:509-517`; `own.py:266-267`, `:343-345`).
- Before any write, a take's block must be neither (`res.py:453`).

**(4) The resets, and the mid-tick FREE. ACCEPTED, stated as the transition table of section 3.1:**
- take: `gen += 1`, `rd_first := -1`, and `rd_last` is KEPT (`own.py:259-261`). It is the transfer's reader gate,
  the T3 forward (`res.py:478`), and so E's reader mark.
- admission (`own.py:322-361`):
  - `seq := none` (`:336`);
  - drop_unowned: RELEASING, `rd_last := -1`, and `rd_first` is kept (`:337-342`);
  - every other branch: `rd_first := -1` (`:346`);
  - LRU, plus ONE `_evict_over_c` (`:347-351`);
  - VRES if protected (`:352-356`);
  - else drop_vres_unprotected: RELEASING, `rd_last := -1` (`:357-360`).
- Policy E frees a slot in the MIDDLE of a tick:
  - `_evict_now` (`res.py:399-418`): LRU -> RELEASING -> a host wait if the last reader is pending (counted as
    `verifier_force_waits`) -> FREE;
  - `_evict_inflight` (`res.py:363-388`): the superseded write is forced, the transfer is forgotten -> FREE.

  I5 allows a FREE transition exactly there and in `poll`.
- I5 gets the `first_free` rank: the q-th FREE-slot take of a stream in the tick takes the stream's q-th FREE slot
  of the tick start (`res.py:456-465`, `own.py:429-433`). The proof is in I4 below.
- `release` (`own.py:271-278`) and `end_owner`'s releases keep both readers.

**(5) The rules with no counterpart. ACCEPTED: invariants I10-I15 and the lifecycle contract of section 3.2:**
- **I10, T4.** `hist_from` (B,) and `hist_id` (B, NBLK). A take of block >= `hist_from[b]` with no registered
  write-back is refused before any change (`res.py:197-201`, `:449`). The transfer's write-back gate = `hist_id`,
  which gives E's `hist_mark`.
- **I11, cancelled keys.** `kcancel` (B, KEYS):
  - a row of a cancelled key is refused (`res.py:534-535`);
  - `cancel` sets `kcancel` and queues (key, l) for every l (`res.py:768-785`);
  - `cancel_epoch` (`res.py:786-798`) reads `keys_seen` (`res.py:540-541`). That stays a host dict per (uid, epoch):
    one entry per row, not per entry.
- **I12, the late-protection guard.** A row of key slot k at layer l is refused while `rel_q[l, b, k]` is set
  (`res.py:538-539`).
- **I13, key-slot reuse.** A key slot is free only when:
  - no pin bit of it is set at any (l, h, block) of b;
  - `rel_q[:, b, k]` is all false;
  - its `key_rd` ids are complete.

  A new key takes the lowest free slot. With none free, the new key's first row raises `CapacityRefusal` (C's Q-B4
  answer: refused, never aliased). Post-tick invariant: no pin bit of a free key slot is set.
- **I14, the request lifecycle** (section 3.2):
  - `begin_request` refuses an owned slot (`res.py:203-218`);
  - `end_request` / `end_owner` (`res.py:220-234`, `own.py:363-384`) as listed there.
- **I15, the slot range.** A record that names a slot >= NPOS is refused: the tail rings (`res.py:161`) and the trash
  slot (`res.py:165`) are outside every pool.
- **The poll order is a contract:** arrivals, then the queued drops, then `poll_releases` (`res.py:283-297`). The
  lockstep has a planted swap.
- **Class.** A take is "v" for VERIFY, "ad" for DRAFT (`res.py:480`). `acquire` changes the state only
  (`own.py:305-321`). B2 never re-tags.
- **The metrics `_counters` compares** (`test.py:607-611`): masked, draft_pairs, refused, the named counts, k43
  (ready / in_flight / new / layer_calls), k46 and `transferred`. B2 PRODUCES them as arrays:
  - `xfer` (B, L, H, NBLK) counts the transfers per (request, l, h, block). It resets at `begin_request`, and
    refetch = count > 1 at the issue (`res.py:863-868`). The reports map `xfer` back to uid keys.
  - k43 / k46 come from `vlook` (the device's ready first looks) plus the in_flight / new records. The open-call
    dedup is per (key, l, h, block) (`res.py:882-927`).
  - The lifetimes' rd / rv counts (4.8) are two int32 slot columns that reset at a take. They are not compared.
- **I7 and the current owner.** An in-flight victim must be the current owner's transfer (`uid == owner`,
  `res.py:509-517`, `:353-362`). The "not selected" check uses the device's `sel` bitmap, so it is COMMON-MODE with
  the device. Only the lockstep (v1 uses its host selections) catches a wrong bitmap.

### 0.2 The other review findings

- **Section 2 (transitions that no schedule reaches): ACCEPTED.** The lockstep adds the scripts that C names and now
  writes as shared CPU schedules:
  - end_request / drop_unowned (F13, F14b plus a final land);
  - drop_vres_unprotected (a scripted demand-{verify} admission without protection, or a cancel between a verify take
    and its arrival);
  - cancel / cancel_epoch / T4 (F3, F11);
  - two draft rows of one stream (C 2, X 0, two ticks);
  - void ticks (F12f, F16);
  - non-monotone readers (F12e: B2 must MATCH, section 0.1 (1));
  - the trash slot (F15).

  Each schedule asserts its own precondition: the transition count > 0.
- **Section 3 (planted defects): ACCEPTED.**
  - "Overflow ignored" is caught by the LOCKSTEP: the vendored tick refuses, B2 applies.
  - "Selected" is common-mode.
  - The 8 missing kinds are in section 4.
- **Section 4 (cost): ACCEPTED as a measurement task.**
  - Sorts run only on the streams that need them: the Policy E streams and the streams over C. A few victims per
    stream take a top-k.
  - B2-T times every component (section 5). The falsifier stands: more than 25 ms per round tick on one amd core
    means the device version is designed before the host one.
- **The K5 proof that C asked for.** Inside a tick, usage changes only at a take into a FREE slot (+1):
  - a Policy E reuse keeps usage (FREE then taken at once: `res.py:420-437`);
  - an `evict_to_c` release keeps it (LRU -> RELEASING; RELEASING counts, `own.py:135-136`);
  - an acquisition keeps it (PO / VRES -> LRU);
  - no other FREE transition happens before `poll` (`poll_in_resolve=False` for S-B / S-C, `sb.py:188`).

  So usage_at = usage0 + the stream's earlier FREE-slot takes, and borrowed_at = borrowed0 + the layer's earlier
  borrowing takes. The FREE set of a stream only shrinks during the tick, so its q-th FREE-slot take gets its q-th
  FREE slot.

## 1. Prior art (the gate; unchanged from rev 1)

| what | where | use in B2 |
|---|---|---|
| S-C's `DeviceMirror`: per-slot tensors (L, B, H, C + X) of code, block, gen, cur, prot, stamp; `layer()` derives slot_of, the LRU rank and Policy E's in-flight order with device ops | `sc.py:161-227` | the core layout, made authoritative; the same codes (`C_FREE` .. `C_RELEASING`, `sc.py:92`) and stamp rules (`STAMP_KINDS`, `sc.py:99-101`) |
| S-B's `rank_in_group` | `sb.py:75-89` | bulk takes, the K5 prefix, the rounds |
| C's B1 device parts (sparse `c6ef37d`): `_compact`, `_pack_bits` / `unpack_bits`, `draft_bias_fill`, `_first_in_group`, `vlook`, the 12l record | `sparse.py:75-112` (at own2; c6ef37d may shift them), `:185-202` | used as they are. NOT the pin table: B2 holds protection itself. |
| my `spec_loop.Store` (`nosi-r2-perreq` `e58011c`, `spec_loop.py:111-284`, `:309-381`, `:496-593`) | | idioms only (watermark arrival, sort-key choice) |
| HiSparse `@87db743` `hisparse.cuh:640-672`: a device LRU per request | (`own.py:19-20`) | the only array-based residency outside; no pins, reader gates or extras |
| vLLM `BlockPool`, SGLang radix / HiCache (host Python), TensorRT-LLM KV manager (host C++) | | not tensor-native; nothing reusable as code |

## 2. The state (arrays)

Shapes: per slot (L, B, H, NPOS); per block (L, B, H, NBLK); per request (B,).

| array | dtype | vendored source |
|---|---|---|
| `code` | int8: FREE, INFLIGHT, LRU, PO, VRES, RELEASING (`sc.py:92`) | `Slot.state` and the stream maps (one code = one map) |
| `block`, `gen` | int32 | `Slot.block`, `Slot.gen` |
| `uid` | int32: the slot's owner request (-1 none) | `Slot.owner` (`cur` = `uid == owner[b]`) |
| `cls` | int8: none, "ad", "v" | `Slot.cls` |
| `stamp` | int64: the LRU recency of LRU slots; the take order of INFLIGHT slots | the `OrderedDict` order; S-C's `stamp` |
| `rd_first`, `rd_last` | int64 reader event ids (-1 none) | `Slot.first_reader`, `Slot.last_reader` |
| `seq`, `issue_tick` | int64 / int32: the slot's transfer (E's job seq) | `Slot.seq`, `Transfer.seq`, `Transfer.issue_tick` |
| `demand`, `issuer`, `req_key`, `landed` | int8 bits / int8 / int8 key slot / bool | `Transfer.demand`, `.kind`, `.requester`, `Residency._landed` |
| `hist_gate` | int64 write-back id of the slot's transfer | `Transfer.hist_ev` |
| `rd_n`, `rv_n` | int32 (4.8 lifetimes; reset at a take) | `Metrics.lifetimes` |
| `pin_w` | BIT PLANES of C's 32-bit words: (2 KEYS, L, B, H, ceil(NBLK / 32)) int64; plane k = key slot k for verify, KEYS + k for draft (KEYS <= 7). The rowsel OR is a word OR (B2-T: about 54 k words per tick), and a key drop clears two planes at (l, b). | `Stream.prot`, `Stream.pin_kind` |
| `owner` | (B,) int32 | `Stream.owner` (one per batch slot) |
| `key_of`, `kcancel` | (B, KEYS) the (epoch, m) of a live key slot; cancelled | the cohort keys; `Residency.cancelled` |
| `key_rd`, `rel_q` | (L, B, KEYS) reader id; queued for release | `Residency.key_reader`, `queue` / `queued` |
| `hist_from`, `hist_id` | (B,), (B, NBLK) int64 | `hist_from`, `hist_ev` |
| `xfer` | (B, L, H, NBLK) int32 | `Metrics.transferred` |
| event registry (host) | id -> (event, E's mark); `lo`, `done_win` | the event objects |

Derived: `resident` = code in {LRU, PO, VRES}; `usage` = (code != FREE).sum(-1); `slot_of` (one scatter per tick of the
resident and current-owner INFLIGHT slots); the layer's borrowed extras.

## 3. The rules

### 3.1 Transitions (B2 implements exactly these; each cites the vendored line)

| transition | where in B2 | vendored |
|---|---|---|
| FREE -> INFLIGHT (take, verify "v" / draft "ad"): `gen += 1`, `rd_first := -1`, `rd_last` kept, `uid := owner`, `stamp := take clock`, `hist_gate`, `xfer += 1` | apply | `res.py:438-489`, `own.py:238-269`, `res.py:863-868` |
| LRU -> FREE (Policy E, LRU kind; wait if `rd_last` pending) then FREE -> INFLIGHT | apply | `res.py:399-418`, `:420-437` |
| INFLIGHT -> FREE (Policy E, in-flight kind; the write forced) then taken | apply | `res.py:363-388` |
| demand merge (verify / draft bit OR) | apply | `res.py:605-611`, `:672-674` |
| PO / VRES -> LRU (acquisition), LRU touch | apply, recency round | `own.py:202-206`, `:305-321` |
| LRU -> PO (pinned) / RELEASING (`evict_to_c` per round; `_evict_over_c` per admission) | apply, poll | `own.py:280-303` |
| INFLIGHT -> LRU / VRES / RELEASING (admission branches, section 0.1 (4)) | poll | `own.py:322-361`, `res.py:236-282` |
| PO / VRES -> RELEASING (the last pin of a key dropped) | poll (queued drops), `end_request` | `own.py:213-237`, `res.py:298-305` |
| RELEASING -> FREE when `done(rd_last)` | poll | `own.py:435-452` |
| LRU / PO / VRES -> RELEASING ("end") | `end_request` | `own.py:363-384` |
| readers: `rd_first` if -1, `rd_last := e` for every slot read; `key_rd[l, b, k] := e` for the rows' keys; the verify keys' (key, l) queued | `read_done(e)` | `res.py:743-761` |

### 3.2 Request and key lifecycle

- `begin_request(b, uid, hist_blocks)` refuses an owned b. It sets `owner`, `hist_from` and `xfer[b] := 0`
  (`res.py:203-218`).
- `end_request(uid)` (`res.py:220-234`, `own.py:363-384`):
  - b's LRU / PO / VRES slots -> RELEASING ("end");
  - `pin[:, b] := 0`;
  - `owner[b] := -1`, so the in-flight slots of the old uid are dropped at their arrival (T3);
  - `hist_id[b] := -1` and `hist_from` cleared;
  - every queued (key, l) of the uid -> `kcancel`, and `rel_q[:, b] := false`.
- `decided(key)` queues (key, l) at every l (`res.py:763-766`). `cancel(key)` also sets `kcancel`
  (`res.py:768-785`). `cancel_epoch(uid, epoch)` cancels the seen keys (`res.py:786-798`).
- Key slots per I13.

### 3.3 The invariants (predicates in the check-then-write apply)

PRE-WRITE (record against the state; a failure voids the tick before any change):
- I4 K5 (with the proof in 0.2);
- I5 (a take into a FREE slot, the first_free rank, no reuse under a pending reader except Policy E's wait);
- I6 (an admission names an INFLIGHT slot: gen, block, uid);
- I7 (Policy E candidates and order; the current owner; `sel` common-mode);
- I10 (T4), I11 (cancelled), I12 (late protection), I13 (key slots), I15 (slot range);
- the record checks of note 12l: a READY draft record names the resident slot; a merge names the current owner's
  in-flight slot and generation; a refusal is one K5 refuses; a HOLE raises.

POST-WRITE (a failure rolls the undo log back, so the tick is void): I1 (resident + current-owner INFLIGHT, at most 1
per block), I2 (LRU <= C, also after poll), I3 (PO / VRES pinned), I9 (gen changes only at a take, by +1), and the
global K5 sum <= X per layer.

LOCKSTEP ONLY: the full LRU order, the pins of the blocks that are not resident, the release queue, the metrics, and a
wrong `sel` bitmap.

## 4. The equality contract (CPU lockstep with planted defects)

- **The worlds.**
  - F6, F12b, F20b, F21b, F22 (`test.py`);
  - the scripts of 0.2: F13 / F14b + land, drop_vres_unprotected, F3, F11, two draft rows, F12f, F16, F12e, F15.

  For each tick, B2 is fed C's sparse record from S-C's device decisions (`ResidencySparse._layer_record`). The
  vendored side runs as today (v1 and S-C).
- **C's shared fixtures** (`nosi-r2-m2-sparse` `827bbae`,
  `benchmarks/Efficiency/tests_cpu/r2_lockstep_fixtures.py`, note 12m) drive the scripts:
  - `run(fixture, worlds, snapshot, Row)` applies each op to every world and compares the exception class, the
    value and the snapshot; the first difference names its op and field;
  - a VENDORED world goes first (the counted preconditions read world 0), B2's world after it, with its own
    `land()`;
  - the fixtures: end_request (dedup merge, release "end", drop_unowned), cancel (cancel_epoch, the late guard),
    o1_two_rows, void_overflow, void_flag, t4, trash, nonmono, vres_drop (the poll order), key_alias.

  B2 imports the module by path (read only).
- **The comparison.** After every `end_tick` and every `poll`, `project(svc)` maps the vendored state onto B2's arrays,
  and they must be equal in every column:
  - LRU order as stamp ranks; pins through the key slots; readers as event ids; `pending` / `inflight` as slot
    columns;
  - the metrics of `_counters`.
- **Planted defects.** Each one must differ at the first tick it changes. A "P" also fails an in-apply predicate.
  - Rev 1's 14 (rev 1 section 4), with C's corrections: "overflow ignored" -> lockstep; "selected" -> lockstep.
  - The 8 that C adds (review section 3):
    1. a take of a block resident or in flight for the owner (P, I1 / dedup);
    2. a take without a registered write-back (P, I10);
    3. a row of a cancelled key, or after its (key, l) release was queued (P, I11 / I12);
    4. a key slot reused while its pins remain (P, I13);
    5. the poll order swapped (lockstep);
    6. a drop_unowned / drop_vres_unprotected that keeps `rd_last`. It shows in the READERS only (C's correction 2:
       the kept reader is the take's forwarded reader, which the transport waited for before the land, so the slot
       frees at the same poll). B2-T's control `keep_rd_last_on_drop` is caught by the reader compare;
    7. a take that drops `rd_last` (lockstep, and E's reader mark);
    8. two draft rows of one stream with one batch eviction. It is NOT observable in the state (section 0.1 (2)), only
       in the transition counts. The stamp-order defect (`reverse_stamps`) replaces it as the recency's negative
       control.
  - Plus: an in-flight Policy E victim of a previous owner (P, I7).

## 5. B2-T: the component microbench (built first; host CPU tensors at p3's geometry)

- `benchmarks/Efficiency/r2_own2_b2t.py` times each component separately per round tick. The ops are in
  `nosi/nosi/r2_own2_ops.py` (vectorized, no per-entry Python), which `apply_tick` will use.
- Inputs: synthetic, at p3's sizes and with C's record shapes:
  - per tick 1,054 urgent (467 Policy E), 64 verify merges, 6,493 draft takes, 1,269 refusals;
  - about 0.2 M READY draft records;
  - `vread` / `sel` / `rowsel` as 32-bit words;
  - 51 verify rows from 13 cohorts, 55 draft rows.
- Each component is a REPORT:
  - poll: the reader window, the arrivals, `_evict_over_c` on the streams over C only, the key drops, the frees;
  - apply:
    - the parse and decode;
    - the row checks;
    - the pins (rowsel OR, grouped by (b, key));
    - K5 and first_free;
    - slot_of and the dedup;
    - the READY / merge checks;
    - Policy E on its streams only (sort or top-k);
    - the writes with the undo log;
    - the recency rounds with `evict_to_c` on the streams over C only;
    - the post-write predicates;
    - the readers;
    - the metrics arrays;
    - E's `issue_desc` arguments.
- Each component's result is checked against a brute-force reference on small random inputs (CPU tests). The
  recency is checked against the vendored `Stream`, including the two-row case. Negative controls are included.
- REGISTERED FALSIFIER: total > 25 ms per round tick (median, one thread) refutes the host estimate. The device
  version is then designed before the host one, and the component table names what moves first.

### 5.1 B2-T on the device (B2TG; registered before the first GPU run, 2026-10-21)

**The host result.** Job 2191023, one thread: the median is 164.364 ms per round tick, so the host falsifier
(> 25 ms) FIRED. With 4 threads it is 72.778 ms. By the registered rule, the device version comes first.
- The ops are torch tensor ops only. So the SAME bench runs on CUDA tensors (`--device cuda`).
- The step script is `benchmarks/Efficiency/r2_own2_b2tg_step.sh`. It holds the same registration block as the
  bench's `REGISTERED_DEVICE`, and a CPU test checks that the two are equal.

**Two modes** (C's request, as K3X's checked mode):
- CHECKED: every predicate, read in TWO host reads per apply. The pre-write flags void the tick; the post-write flags
  roll the undo log back.
- UNCHECKED: the verification-only components (`CHECK_ONLY`) are skipped, and the writes go without the undo log. The
  device's decisions are trusted.
- Both modes must end in the SAME table (a coherence failure otherwise).

**The model behind the predictions:**
- The device ops are LAUNCH-BOUND: about 800 kernel launches and about 60 host synchronizations per round tick, at
  6-15 us per launch and 15-40 us per synchronization.
- The GPU's own work (about 30 full-table passes of 0.5 M slots plus the record ops) is 1.5-4 ms of kernel time.
- The unchecked mode drops about 330 launches and 10 synchronizations.

| id | quantity | prediction | band | falsifier |
|---|---|---|---|---|
| D1 | checked, pipelined median per round tick (whole ticks, no component synchronization) | 9.0 ms | 5.0-18.0 ms | > 18.0 ms |
| D1u | unchecked, pipelined median | 5.5 ms | 3.0-11.0 ms | > 11.0 ms |
| D2 | checked, the sum of the synchronized component medians | 10.0 ms | 5.0-20.0 ms | > 20.0 ms |
| D3 | profiled device busy time (the kernels' self time) | 3.0 ms | 1.5-6.0 ms | > 6.0 ms |
| D4 | kernel launches per tick | 800 | 500-1200 | REPORT |
| D5 | host synchronizations per tick | 60 | 35-120 | REPORT |
| D6 | every component with wall >= 0.2 ms has profiled device time <= 0.5 x wall (host-bound) | holds | - | one that fails |

**Expected HOST-BOUND components (`HOST_BOUND`):**
- the ones with a Python loop:
  - `poll.refresh` (the live events);
  - `apply.pins` (at most K + 1 rounds);
  - `apply.recency` (the rounds);
  - `apply.readers` (L event registrations);
- the ones with a host read:
  - nonzero / a boolean mask: `poll.arrivals`, `poll.key_drops`, `poll.releases`, `apply.k5_events`, `apply.takes`,
    `apply.merge_check`, `apply.post_checks`;
  - unique: `apply.policy_e`, `apply.merges`;
  - int(max) / bool(any): `poll.force_seq`, `apply.issue_desc`;
  - the designed reads: `apply.pre_flags`, `apply.post_flags`, `apply.metrics`.
- Tensor passes only: `apply.slot_of`, `apply.pin_words`, `apply.k5_first_free`, `apply.take_checks`,
  `apply.ready_check`, `apply.gen_snapshot`.

**What the result decides:**
- **D1 holds:** B2 on the device costs about 7-10 % of a 90-140 ms round tick. The next lever is removing the
  synchronizations (fixed-width top-k, masked writes) and the capture, not more ops.
- **D1 fires:** the per-op host cost on that node is above the model. Then CUDA-graph capture or fused kernels come
  before anything else.

**C's record formats (confirmed for the lead):**
- `rowsel` uses int64 words of 32 bits (`_pack_bits`), the same words as B2's pin planes.
- It is indexed by TICK row. B2 maps each tick row to (b, plane) (plane = key slot for verify, KEYS + key slot for
  draft) and ORs ALL rows in ONE `pins_or` call. B2-T already uses that layout.
- `pin_records=False` is right for B2.

## 6. The interface

- **The host consumer.** `OwnResidency.apply_tick(steps)` has C's signature (note 12l: `steps[i] = dict(l, vrows,
  vidx, vdec, drows, ddec, rec, caps)`). B2 consumes `vrec`, `drec`, `vlook`, `vread`, `sel` after the one read.
- **Requests to C** (through the lead):
  - (a) add `rowsel` (R, H, ceil(nblk / 32)) per layer. B2's protection is the rowsel OR, grouped by (b, key, kind);
  - (b) make the PIN-only records optional (a flag) when B2 consumes the record. They are about 0.26 M per verify tick
    (note 12l estimate), and B2 does not use them.

  B2 derives each draft INFL record's in-flight slot from its own `slot_of`, so the record needs no new field.
- **E's issue path** (Q-B3, as E built it):
  - `seq = tp.issue_desc(desc, reader_marks, hist_mark)`;
  - `desc` = B2's draft take list (6, n) CONTIGUOUS int64 (l, b, h, block, slot, gen). Lesson L28: a non-contiguous
    view into a raw-pointer gather reads shifted entries;
  - `reader_marks` (L,) = the newest E mark of the destination slots' kept `rd_last` per layer;
  - `hist_mark` = the newest write-back mark over the job's (b, block), or -1;
  - B2 stores `seq` per slot. A poll admits the slots with `seq <= tp.landed_upto()` and `issue_tick + D <= tick` in
    one vector op. A slot due by the arrival rule but not landed is forced through the transport first.
- **Lifecycle calls:** `poll(tick)`, `read_done(l, plan, ev)` (registers the id), `decided`, `cancel`,
  `cancel_epoch`, `begin_request`, `end_request`, `snapshot()` / `project()`.
