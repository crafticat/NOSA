# B2: an array-backed ownership state beside the vendored one (design; no code before the lead's GO)

Branch `feature/nosi-r2-m2-own2` from C's `c2fccb2` (feature/nosi-r2-m2-sparse). Status: DESIGN ONLY.

Why (the engine agent's split, `nosi-r2-sg5-capture` `1d1b956`, `docs/R2_SG5_CAPTURE_PLAN.md` sections 0 and 3.3):
after the captured tick (A) and C's sparse decision record (B1), a round tick at B 64 still costs 0.5-0.9 s. The bound
is the host's ~6.5 k draft takes per tick plus the per-slot bookkeeping of the vendored K3X ownership objects
(`nosi/nosi/kstep/ownership.py`, vendored byte-identical at `91e6e47`: `r2_residency.py:4`, test F9
`test_r2_residency.py:545`): one Python `Slot` / `Stream` object per slot, dicts and an `OrderedDict` per stream, one
method call per entry. The plan's Q-C4 (`R2_SG5_CAPTURE_PLAN.md:280-284`) asks for bulk takes: the tick then
approaches its device time (draft-only 45-60 ms, round 90-140 ms) plus the host part. B2 is that host part. Its target
is a host cost small against 90 ms. B2 keeps the SAME rules (K1-K5, T3, Policy E, O1) on arrays; the vendored objects
stay the reference.

Geometry used below (p3's point, `R2_SG5_CAPTURE_PLAN.md:177`, `:187`): L 32, B 64, H 2, C 96, X 32, R_ver 16, 16K
context. NPOS = C + X = 128 slots per stream (`n_hist`, `r2_residency.py:160`); NBLK = n_host // 64, about 256
(`r2_tick_sg5.py:302`). That is 524,288 slots.

## 1. Prior art (the gate)

In our trees (array-based, reusable):

| what | where | fits B2? |
|---|---|---|
| S-C's `DeviceMirror`: the ownership state of every layer as per-slot tensors (L, B, H, C + X): state code, block, generation, current-owner flag, protected flag, recency stamp; `layer()` derives slot_of (scatter), the LRU rank (sort of stamps), Policy E's in-flight order (sort of take stamps, `HALF_STAMP` offset) with device ops only | `nosi/nosi/r2_residency_sc.py:90-227` (C, `c2fccb2`) | YES: B2's core layout. B2 makes this table AUTHORITATIVE and adds the columns the mirror does not hold (owner identity, class, readers, pins, in-flight records). The same codes (`C_FREE` .. `C_RELEASING`) and stamp rules (`STAMP_KINDS`). |
| S-B's `rank_in_group`: the rank of an entry within its group (the q-th grant of a stream takes the q-th FREE slot) | `nosi/nosi/r2_residency_sb.py:75-89` | YES: the bulk takes (Q-C4). |
| my `spec_loop.Store`: per (layer, head, request, slot) `ids / state / gen / stamp / used`; `PrefetchQueue` (arrival = a generation watermark); `_assign` (a slot / victim choice by one sort key per stream); `plan_draft` / `plan_round` as `torch.where` updates | `nosi-r2-perreq` (`e58011c`) `nosi/nosi/spec_loop.py:111-284` (Store, `mark_arrived` :171, PrefetchQueue :208, `_assign` :243), `plan_draft` :309-381, `plan_round` :496-593 | IDIOMS ONLY. It has no owners, pins, PO / VRES, reader gates, K5 extras or Policy E. Its watermark arrival and its sort-key victim choice are what B2 uses for arrivals and evictions. |
| K3X M2 `Arena.pgen_t`, `RingPool.rgen_t`: int64 generation mirrors | `nosi/nosi/kstep/ownership.py:503-641`, `:642-828` | generation mirrors only. |

Outside (checked by name and code where we have a snapshot):
- HiSparse (SGLang, `@87db743`, `hisparse.cuh:640-672`, the upstream of K1's recency rule cited in `ownership.py:19-20`):
  a device-side LRU per request (hits to MRU) in CUDA. It is the only array-based residency we know of, and it has no
  pins, reader gates, shared extras or verifier reserve.
- vLLM's block manager (`BlockPool`, a doubly linked free queue) and SGLang's radix cache / HiCache are host Python
  objects; TensorRT-LLM's KV cache manager is host C++. None is tensor-native, and none has reader-gated reuse or
  per-cohort pins.
- So nothing outside is reusable as code. B2 = S-C's mirror layout made authoritative, plus the missing columns, with
  the vendored objects kept as the lockstep reference.

## 2. The state (arrays; one `OwnTable` per run)

NPOS = C + X slots per stream, NBLK = the run's block horizon, KEYS = the live cohort keys per request. KEYS is a
constructor argument from (K, LAMBDA): ceil(LAMBDA / K) + 1 undecided cohorts plus 1 decided key that waits for its
reader, so 5 at K 3, LAMBDA 7 (`r2_tick_sg5.py:77-78`). The lead said "at most about 4" (Q-B4). Shapes per layer,
stream (b, h), slot or block:

| array | shape, dtype | vendored source |
|---|---|---|
| `code` | (L, B, H, NPOS) int8: FREE, INFLIGHT, LRU, PO, VRES, RELEASING | `Slot.state` |
| `block`, `gen` | (L, B, H, NPOS) int32 | `Slot.block`, `Slot.gen` |
| `cls` | (L, B, H, NPOS) int8: none, "ad", "v" | `Slot.cls` (rev6) |
| `stamp` | (L, B, H, NPOS) int64: LRU recency (touch / admit / acquire), take order for in-flight slots | the LRU's `OrderedDict` order; S-C's `stamp` |
| `rd_first`, `rd_last` | (L, B, H, NPOS) int32: the TICK of the slot's first / last reader event (-1 none) | `Slot.first_reader`, `Slot.last_reader` |
| `seq`, `issue_tick` | (L, B, H, NPOS) int64 / int32: the transfer in flight into the slot | `Slot.seq`, `Transfer.seq`, `Transfer.issue_tick` |
| `demand` | (L, B, H, NPOS) int8 bits: draft, verify | `Transfer.demand` |
| `issuer`, `req_key` | (L, B, H, NPOS) int8: draft / verify; the requester's key slot | `Transfer.kind`, `Transfer.requester` |
| `landed`, `cur` | (L, B, H, NPOS) bool: landed in place (urgent, defer_admit); the transfer is the CURRENT owner's | `Residency._landed`; S-C's `cur` |
| `pin` | (L, B, H, NBLK) uint16: bit k = key slot k (k < KEYS <= 8) protects the block for verify, bit 8 + k for draft | `Stream.prot`, `Stream.pin_kind` (L9) |
| `owner` | (B,) int32 request id; `key_of` (B, KEYS) the (epoch, m) of each live key slot | `Stream.owner`; the cohort keys |
| `key_rd` | (L, B, KEYS) int32: the tick of the key's latest reader at layer l | `Residency.key_reader` |
| `rel_q` | (L, B, KEYS) bool: (key, l) queued for release behind its reader | `Residency.queue` / `queued` |

Derived, never stored: `resident` = code in {LRU, PO, VRES}; `usage` = (code != FREE).sum(-1); the layer's borrowed
extras = sum over streams of max(0, usage - C); `slot_of[block]` by one scatter (as `DeviceMirror.layer`). Readers are
TICK numbers: the capture records ONE reader event per tick for every layer (C's Q-T3 (c)), and the host polls it once
per tick, so "the reader event completed" is `rd_last <= done_tick` against one watermark. The slot columns are 43 B
per slot: 524,288 x 43 B = 22.5 MB. `pin` is 32 x 64 x 2 x 256 x 2 B = 2.1 MB.

## 3. The invariants, each as an array predicate (cheap, every tick) or a lockstep check

The apply is TWO-PHASE: it computes the new columns into scratch, evaluates the predicates on (old, new, record), and
only then commits. A failed predicate or the record's overflow flag refuses the whole tick before any write (as S-C's
void tick does).

| # | rule (source) | array predicate | cost |
|---|---|---|---|
| I1 | K3: one slot per resident block per stream | `scatter_add` of resident slots into (L, B, H, NBLK) counts: max <= 1 | one pass |
| I2 | K1: the LRU holds at most C (at the tick's end, after `evict_to_c`) | `(code == LRU).sum(-1) <= C` | one pass |
| I3 | K2: a PO / VRES slot is protected | `pin[.., block[slot]] != 0` where code in {PO, VRES} | one gather |
| I4 | K5 (`LayerPool.can_take`, `ownership.py:419-427`): a take with the stream's usage < C needs no extra; else a verify take needs a free extra, a draft take more than R_ver | the take records in their order key, per layer: borrow = (the stream's usage before the take >= C), by `rank_in_group` per stream; `borrowed_at_take = borrowed_before + exclusive cumsum(borrow)`; for a borrowing verify take `X - borrowed_at_take > 0`, for a borrowing draft take `X - borrowed_at_take > R_ver`. A Policy E grant reuses its victim's slot at once (`_evict_now`, `r2_residency.py:399-418`; `_free_slot_for_verifier` :420-437): usage is unchanged, so it does not borrow | O(record) |
| I5 | paper 3.4 / K4 / T3: no reuse before owners and readers finish | a take's slot is FREE before the tick; a slot becomes FREE only in `poll` where code == RELEASING and `rd_last <= done_tick`; a Policy E LRU eviction carries `rd_last <= done_tick` or a recorded wait; an in-flight eviction carries "landed or forced" | O(record) + one pass |
| I6 | T3: an arrival names its slot | code == INFLIGHT, gen and block match, `cur` set (else it is dropped: RELEASING) | O(arrivals) |
| I7 | Policy E (notes 12h, 12j): candidates and order | an LRU-kind victim is LRU, unpinned at the layer's start, not in the layer's selection bitmap, and the least-stamped such entry still unused; an in-flight victim is INFLIGHT, issuer draft, unpinned, not selected, not landed, the oldest take stamp; no in-flight victim while an LRU candidate of the stream is unused | O(record) + a segment sort of the touched streams |
| I8 | O1 / K1 recency: draft stamps in (row, head, ascending block) order; `evict_to_c` after the row's batch | the record's draft stamps strictly increase in its order key; the victims are the `n_lru - C` least-stamped LRU entries of the stream after the stamps; pinned -> PO, else RELEASING | a segment top-k over the touched streams |
| I9 | the generation | `gen` changes only at a take, by +1 | one compare |

Not cheap as predicates, therefore in the CPU lockstep only (section 4): the whole LRU ORDER (I2 and I8 check counts
and victims; the lockstep checks every rank), the pins of the blocks that are not resident, the release queue against
the reader ticks, and the metrics (k43 / k46 / k48, refusals, verifier evictions).

## 4. The equality contract (CPU lockstep with planted defects)

- **The lockstep.** The existing worlds and schedules of `benchmarks/Efficiency/tests_cpu/test_r2_residency.py`:
  - F6 (`_f6`, seeded delays, reordering, bursts, late readers);
  - F12b (capacity pressure: refusals, the verifier reserve);
  - F20b (Policy E, LRU kind);
  - F21b (asynchronous transport: Policy E, in-flight kind);
  - F22 (arrival lag D = 2: forced and held arrivals).

  The vendored service (v1 and S-C) runs as today. A B2 `OwnTable` is fed the same tick's record: C's sparse record
  from S-C's device decisions, or, for v1, a record synthesized from v1's plan and the stream journals that S-C
  already reports (`_Journal`, `_ProtDict`).
- **The comparison.** After every `end_tick` and after every `poll`, `project(svc)` maps the vendored state onto B2's
  arrays, and they must be equal in every column:
  - an LRU's `OrderedDict` order becomes ranks, compared with the rank of B2's stamps;
  - `Slot.first_reader` / `last_reader` become the tick of the test's event;
  - the protection map (block -> keys -> kinds) becomes `pin` bits through the request's key slots;
  - `inflight` / `pending` become the slot columns;
  - `key_reader` and `queue` become `key_rd` / `rel_q`;
  - the counters must be equal too.
- **Planted defects** (one per record kind). Each must fail the lockstep at the first tick it changes, and each marked
  P must also be refused by the in-apply predicate:

| record kind | planted defect | caught by |
|---|---|---|
| draft take | into a non-FREE slot (P, I5); at a slot other than K3X `first_free` | I5; lockstep |
| verifier grant | a grant past the extras (P, I4); a draft borrowing into the reserve (P, I4) | I4 |
| Policy E, LRU kind | a pinned victim (P, I7); a victim the layer selects (P, I7); not the least recent | I7 |
| Policy E, in-flight kind | a protected or landed victim (P, I7); not the oldest; an in-flight victim while an LRU candidate is unused (P, I7) | I7 |
| refusal | dropped (the take happens) | I4 (when over), else lockstep (counter, state) |
| demand merge | dropped: the arrival goes VRES instead of LRU | lockstep |
| PO / VRES acquisition | dropped; the wrong source | lockstep; I2 / I3 |
| overflow | ignored: the tick applies | the void-tick rule (no column changes) |
| new pin | dropped: a PO entry unprotected (P, I3) or an entry evicted that v1 keeps PO | I3; lockstep |
| draft stamps | permuted inside a row's batch (P, I8) | I8; lockstep (ranks) |
| evict_to_c victims | not the least recent (P, I8); a pinned victim released (P) | I8 |
| read bitmap | a slot missing: its `rd_last` too old, so it frees one tick early | lockstep (I5 cannot see a missing reader) |
| poll | frees a RELEASING slot whose reader is pending (P, I5) | I5 |
| arrival | stale generation, or admitted for an old owner (P, I6) | I6 |

## 5. The cost per tick at B 64

Inputs: 0.21 M verify and 0.23 M draft entries per round tick, about 9.3 k sparse records (1.05 k urgent, 0.06 k
merges, 0.47 k Policy E, 1.27 k refusals, 6.49 k draft takes; `R2_SG5_CAPTURE_PLAN.md:264-271`), the read bitmap of
L x B x H x NPOS = 32 x 64 x 2 x 128 = 0.52 M bits = 64 KB, NBLK about 256.

- **A number to correct in the plan:** `R2_SG5_CAPTURE_PLAN.md:169` says the bitmap is 2.2 M bits = 270 KB. That
  matches H 8 (32 x 64 x 8 x 128 = 2.1 M). At H 2 (the plan's own correction, `1d1b956`) it is 0.52 M bits = 64 KB.

**The pin volume.** The plan says "the protects are not counted in the record" (`:267`), so I counted them from the
code. v1 calls `st.protect(t, r.key, kind)` for EVERY selected entry: `r2_residency.py:597` (VERIFY) and `:668`
(DRAFT). That is 0.21 M + 0.23 M protections per round tick. Most of them repeat a (block, key, kind) that is already
pinned. The NEW pairs come mostly from the first row of a new key: 55 draft rows / K 3 x 32 x 2 x 64 = 75 k per tick.
Either count is larger than all the other records together.

- **PROPOSAL to C** (through the lead): the record does NOT carry pins one by one. It carries, per row and per
  (layer, head), the row's SELECTION BITMAP over NBLK, the row's key slot and its kind. B2 then ORs that bitmap into
  the key's bit of `pin`. That is 106 rows x 32 x 2 x 256 bits = 217 KB per tick, and one vector op. It is the same
  information as the selection that the device already holds.

| work per tick | host torch on CPU tensors | device (the same code on CUDA) |
|---|---|---|
| dispatch: about 60-120 ops | 0.5-1.5 ms (5-15 us each) | 0.5-1.2 ms of launches; capturable |
| full-table passes (poll, the reader bitmap into `rd_first` / `rd_last`, I1-I3, I9): about 8 passes over 22.5 MB (0.52 M slots each) | 5-12 ms | < 1 ms |
| the records (9.3 k) and the pin bitmaps (217 KB, OR into 2.1 MB): scatter / index_put by kind | 1-2 ms | < 0.2 ms |
| K5 prefix sums, Policy E and `evict_to_c` segment sorts over the touched streams (55 x 32 x 2 = 3.5 k draft and 51 x 32 x 2 = 3.3 k verify streams, x NPOS 128) | 3-8 ms | < 0.5 ms |
| the device mirror (S-C's `DeviceMirror`) | a delta of the dirty slots (about 20 k x 7 int64 = 1.1 MB pinned H2D): < 1 ms | none: the table IS the mirror |
| the record's read to the host | already in B1's one read | only the take list (6.5 k x 8 int64 = 0.4 MB) and the flags |
| **B2 total** | **about 10-25 ms** | **about 2-4 ms** |

These are estimates, not measurements (the per-op rates are typical torch CPU rates for 0.5 M-element int tensors). A
CPU micro-benchmark of `apply_tick` on a synthetic record of p3's counts measures them before any GPU run (section 7).

**Outside B2: the transfer issue.** `GilfreeTransport.issue` (`r2_transport_gilfree.py:265`) takes a list of Python
`Transfer` objects. The plan estimates the draft issue at 0.15-0.2 s per tick (`R2_SG5_CAPTURE_PLAN.md:270-271`). B2
alone does not remove it. B2 hands the takes out as TENSORS (l, b, h, block, slot, gen, seq, kind, key, the previous
reader's tick), CONTIGUOUS (lessons L28: a non-contiguous view into the raw-pointer Triton gathers read shifted
entries). A descriptor-tensor issue path into E's packer would remove the per-object cost. That is E's and C's
interface (Q-B3).

**Placement: build device-agnostic, place it on the host first.**
- Every B2 function is torch ops with no per-entry Python loop and no host read inside the apply (the predicates
  reduce to one flag tensor read once).
- PHASE 1 runs it on CPU tensors: the CPU lockstep is the same code, and C's host consumer already holds the record
  after B1's one read. The expected cost is 10-25 ms of B2, plus the issue path until Q-B3 is solved.
- PHASE 2 moves the table to the device after (A) and B1 are measured. The record then never leaves the device except
  for the take list. B2 shrinks to about 2-4 ms and becomes capturable. What Phase 2 needs: the arrivals as a device
  watermark (the generation board already exists: `GenBoard`, `r2_residency_sc.py:574`, written by the transport
  that publishes, `:611`) and the reader watermark as one scalar per tick.

## 6. The interface (one function for C's host consumer; the rest are lifecycle calls)

```python
class OwnTable:                                   # nosi/nosi/r2_own2.py (after the GO)
    def __init__(self, L, B, H, C, X, R_ver, nblk, K, LAMBDA, device="cpu", check=True): ...   # KEYS from (K, LAMBDA)
    # THE per-tick call: C's ResidencySparse.apply_tick, after its one device-to-host read
    def apply_tick(self, rec: "SparseRecord", tick: int) -> "TickResult": ...
    # lifecycle (small, per request or per tick)
    def poll(self, done_tick: int, arrivals: "Arrivals") -> "PollResult": ...   # admissions, releases -> FREE
    def decided(self, b: int, key_slot: int) -> None: ...                       # queue (key, l) behind its readers
    def cancel(self, b: int, key_slot: int) -> None: ...
    def begin_request(self, b: int, uid: int) -> None: ...
    def end_request(self, b: int) -> None: ...
    def snapshot(self) -> Dict[str, torch.Tensor]: ...                         # the lockstep's view
```

- **`apply_tick`** applies one tick in v1's semantics, per layer: the verify part (pins, merges, urgent grants with
  Policy E), the draft part (pins, merges, takes / refusals), the draft recency (stamps, then `evict_to_c`), then the
  read bitmap as readers of `tick`.
  - It returns the takes as tensors for the issue, the metric deltas, and (Phase 1) the dirty slots for the device
    mirror.
  - A void tick raises before any write: `CapacityRefusal` (overflow; `kstep/errors.py:36`) or `OwnershipViolation`
    (a predicate, with its first offending record; `kstep/errors.py:13`). The vendored exceptions are reused.
- **`poll`** at the tick start:
  - admits the arrivals (demand with draft -> LRU at MRU, then the over-C eviction; verify only -> VRES if pinned,
    else RELEASING; an old owner -> RELEASING);
  - drops the due (key, l) pins (`rel_q` and `key_rd <= done_tick`), releasing the unpinned PO / VRES;
  - frees the RELEASING slots with `rd_last <= done_tick`.

**THE RECORD FORMAT is C's (B1). What B2 needs from it, to agree through the lead** (Q-B1):
- one struct-of-arrays int64 tensor, (F, n);
- per record: kind, l, b, h, block, slot, the generation before, an ORDER key (the layer's resolve order: verify
  part, then draft part, then recency; inside each, (row, head, ascending block)), the row's key slot, the demand
  kind, and an aux field (the victim's take stamp or seq);
- kinds: draft take, verifier grant (K5), Policy E LRU, Policy E in-flight, refusal, merge, PO acquisition, VRES
  acquisition, draft stamp (touch), `evict_to_c` victim, overflow;
- side planes: the read bitmap (L, B, H, NPOS) and, per row, the selection bitmap (pins, section 5);
- **Q-B2 (to C):** B2 can recompute the stamps and the `evict_to_c` victims itself from the read set (they are
  determined by the order key). Should the record carry them, so that I8 checks the device's choice, or should B2
  derive them, so that the record shrinks? I prefer that the record carries the victims (a check of the device) and
  B2 derives the stamps.

## 7. Plan after the GO

1. `nosi/nosi/r2_own2.py` (`OwnTable`) and `project(svc)`; CPU tests B2-L1 .. L5 (the lockstep on F6, F12b, F20b,
   F21b, F22, for v1 and S-C), then B2-P (one planted defect per record kind, section 4), then B2-C (no per-entry
   Python loop and no host read in `apply_tick`: a source check, and a host-read check as F10d does for S-B,
   `test_r2_residency.py:757`).
2. B2-T: a CPU micro-benchmark of `apply_tick` on a synthetic record with p3's counts per tick (section 5). It is a
   REPORT that tests section 5's estimate before any GPU run. Registered falsifier: B2's host time > 25 ms per round
   tick on one amd core refutes the estimate, and Phase 2 then comes first.
3. Behind a flag in C's `ResidencySparse` (`own="vendored" | "b2"`); the vendored path stays the default and the
   reference.
4. Measured on the GPU only after (A) and B1: B2's host time per tick (REPORT), then Phase 2 (device) if it is still
   on the critical path.

Questions (to the lead, for C and E):
- **Q-B1:** the record format (section 6).
- **Q-B2:** recompute or carry the stamps and victims.
- **Q-B3:** a tensor-descriptor issue path from the take list into E's packer.
- **Q-B4:** the key slots. Is "at most 4 live cohort keys per request" a hard bound in the planner? My count at K 3,
  LAMBDA 7 (`r2_tick_sg5.py:77-78`) is ceil(7 / 3) + 1 = 4 undecided cohorts, plus 1 decided key while its pins wait
  for the reader (`rel_q`): 5. The lead sweep raises LAMBDA, so KEYS comes from (K, LAMBDA). A key beyond KEYS is an
  `OwnershipViolation` (never a silent drop). `pin` is uint16, so KEYS <= 8.
