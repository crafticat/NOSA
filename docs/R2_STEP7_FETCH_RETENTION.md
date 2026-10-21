# R2 step 7, piece 5: speculative fetch and retention on the R2 path (DESIGN NOTE)

Branch `feature/nosi-r2-m2-fetch` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-fetch`, from cf58fbc).
Status: design e58975d; decisions D1-D5 accepted (section 10). The CPU stand-ins (`nosi/nosi/r2_residency.py`,
`benchmarks/Efficiency/tests_cpu/test_r2_residency.py`, F1-F9) are in the commit after e58975d. No GPU result.

Sources (read, not copied):

- Paper sections 3.1, 3.2, 3.4 and 4.3, 4.6, 4.8, 4.9 (`/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-draft.txt`, lines
  240-381 and 508-686).
- The step-7 note `docs/R2_STEP7_SPEC_ROUND.md`: sections 4 (flattened rows), 5 (slot store, tail pages, background
  requests, urgent misses), 6 (rollback), 7.3 (the tick budget).
- K3X's own retention design (`/mnt/beegfs/ojerbi/nosi-kstep-bc`, 91e6e47, `nosi/nosi/kstep/`): `ownership.py`
  (states, K1-K5, LayerPool, Arena), `transfer.py` (Transfer, S0Transport, AdversarialTransport), `paper_metrics.py`
  (definitions :9-37), `dispatcher.py` (the call sites, listed in section 9), `mutations.py`. Its tests are in
  `/mnt/beegfs/ojerbi/research-stage-s4/tests/` (pinned to 91e6e47).
- Prior-art rows (`/mnt/central/users/ojerbi/retroinfer-eval/docs/superpowers/specs/2026-09-02-prior-art-table.md`,
  cited as TABLE row n = its line; and `/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/r2-prior-art-rows.md`, cited
  as R2 row n): TABLE rows 15, 27, 28, 53-55; R2 rows 1, 2, 4.
- The transports: NOSI's GPU gather (`cache_engine.py:363-382`; `flash_cache_engine/flash_h2d_mask.py`,
  `flash_h2d_mask_bias.py`, `diff_offload_kernel.cu`), agent E's GIL-free packer (`nosi-sustained-gilfree` 30f9035,
  `gilfree_pack.cpp` / `gilfree_pack.py`: NativePipe), the in-house victim pool (`cache_engine.py:26-70`).

## 0. Scope and the boundary with piece 4

- Piece 5 is the RESIDENCY SERVICE of the spec round: for every (layer, KV head, request) it decides which historical
  blocks are readable on the GPU, which are in transfer, and which slot each one occupies. It owns the requests, the
  transports, the retention (ownership), the memory accounting and the paper metrics 4.3 / 4.6 / 4.8 / 4.9.
- Piece 4 (the flattened tick, the verify agent) calls it once per layer of a tick (section 8 gives the interface).
  The tick keeps the arithmetic: selection, attention (piece 3 v2), GEMMs (EXACT / FAST), the decision.
- The service never changes WHICH blocks a verify row attends: the verifier reads exactly its selection, always
  (exactness). It changes only WHERE a block is read from and WHEN it arrives. A draft row reads only the selected
  blocks that are ready (paper 3.1): the draft is approximate by design and the verification corrects it.
- Units: a block key is (request uid, layer l, KV head h, block id t) with t < the request's tail block. HISTORICAL
  CONTENT IS IMMUTABLE: a block reaches the host only when all its 64 positions are committed (note 5.2), and a
  committed position is never rolled back. So a copy of a block is valid for every reader whatever its epoch.
  The hazards are about STORAGE (a slot overwritten while read, a late transfer into a reassigned slot), not about
  content versions.

## 1. Requirements (the lead's (a)-(e), paper 3.1 / 3.2 / 3.4)

| Id | Requirement | Section |
|---|---|---|
| (a) | A draft row selects over all blocks (the shipped two-step selection) but attends only the selected blocks that are READY on the GPU, plus its tail pages; it emits the miss list | 3 |
| (b) | The misses are transferred in the background. The transport is a run parameter: CPU pack + copy-engine DMA (agent E's GIL-free packer, or K3X's S0), or NOSI's GPU gather on a side stream | 4 |
| (c) | RETENTION: every block requested by a live speculative block, cache hits included, stays readable until the verifier's read of it finished: one reference per requester, storage reusable only after its owners and readers ended; the peak retained bytes are in the memory budget | 5, 7 |
| (d) | The verifier waits for in-flight blocks and GPU-gathers only urgent misses (selected by the verifier, not requested before), counting ready / in-flight / new | 6 |
| (e) | The paper metrics 4.3 / 4.6 / 4.8 (and 4.9 for the packer) on the R2 path | 11 |

## 2. Prior art and what is reused (and said so)

- K3X ownership (`kstep/ownership.py` at 91e6e47) is the model, REUSED AS IS (decision D1, section 10):
  - states FREE / INFLIGHT / LRU / PO / VRES / RELEASING (`ownership.py:3-11`, `:94`);
  - K1 a DRAFT-ONLY logical LRU of capacity C per stream (`:13-21`): verifier reads never touch it;
  - K2 protection = one reference per (block, cohort key (uid, epoch, m)) (`:22-23`; the key at
    `dispatcher.py:2605`), demand sets {draft, verify} merged on a second requester (`transfer.py:84-99`,
    `dispatcher.py:2955-2970`);
  - K4 references dropped per (cohort, layer) after that layer's last reader event; a slot FREE only after its own
    last reader event (`:24-27`, `LayerPool.poll_releases :435-452`);
  - K5 C slots per stream plus X shared extra slots per layer, a draft may borrow only while more than R_ver extras
    are free (the verifier reserve), a refused draft fetch is counted and the draft continues masked (`:28-33`,
    `LayerPool.can_take :419-427`);
  - generations (`Slot.gen`, `StaleGeneration`), the reader forwarding of a re-taken slot (`Transfer.reader_ev`,
    `dispatcher.py:3143-3152`), late arrivals after a cancellation (`dispatcher.py:1515-1539`, `admit :322-361`).
  The R2 row 4 / TABLE row 53 record the slot + generation ownership as ours (benchmark-only before K3X).
- The DISPATCHER's residency call sites are RE-IMPLEMENTED for the R2 tick (its tick shape differs) and cited line by
  line (section 9). Its 3991 lines are not imported.
- The metric DEFINITIONS are K3X's, verbatim (`paper_metrics.py:9-37`). The recorder is re-implemented, because
  `PaperMetrics(d)` reads K3X dispatcher internals (`:111-141`).
- HiSparse multi-position resolver (TABLE row 27; R2 row 4: no serving caller, prefetch off under speculation): the
  IDEA of deduplicating the misses of several positions and replaying a recorded plan is used for the draft and
  verify rows of one request in one tick (section 3). Cite the code; copy nothing.
- CPU packing (TABLE rows 53-55, 2026-10-04): a pinned staging ring, a bulk H2D copy, a device scatter, with slot +
  generation ownership. Ours: agent E's GIL-free NativePipe (30f9035, source only, its GPU step queued) and K3X's
  S0Transport. PQCache / InfiniGen are the cited pattern.
- Prefetch (TABLE row 15): prediction-driven prefetch exists only as papers (OasisKV, HiSparse anti-hint, SpeCache,
  DualDecoder). Ours: prefetch driven by the DRAFT's own selection on an EXACT engine. Cite only.
- The in-house victim pool (`cache_engine.py:26-70`, NOSI_POOL_BLOCKS) is the logical-cache extension of the shipped
  exact-fit window (misses per (layer, head, request, step) 3.58 at 63 slots, 1.20 at 80, 0.86 at 96, 0.71 at 128).
  In the BI path the canonical page list removes its view trick: a store larger than 64 slots does not change the
  arithmetic, because pages are listed by block id, never by slot.

## 3. The draft row (requirement (a))

- Selection: the shipped two-step rule with the deterministic top-k (`r2_spec_core.select_blocks`), 64 blocks per
  (row, head), the tail block included.
- Classification per (row, head) against the stream (l, h, b): READY = resident (LRU, PO or VRES), IN_FLIGHT =
  INFLIGHT, MISS = neither.
- The draft attends: its tail pages (exact and draft rings, note 5.2) + the READY selected blocks, in ascending block
  id. Fewer than 64 pages when some are missing (the draft is approximate; paper 3.1).
- Protection: the draft row's cohort key (uid, epoch, m) protects EVERY selected historical block, hits included
  (paper 3.4; K2). m = the cohort that will verify this draft position (cadence: the draft index s belongs to cohort
  m with s in I_m; d_{mK} is deferred to cohort m + 1).
- Requests: the MISS blocks, deduplicated over the request's rows and over the heads' streams of the tick (HiSparse
  resolver idea), are requested with demand {draft}; a block IN_FLIGHT for another requester gets the demand merged,
  no second transfer (`dispatcher.py:2592-2629`, `:2955-2970`).
- Recency (K1): the draft's reads of LRU entries move them to MRU; a draft read of a PO / VRES entry ACQUIRES it into
  the LRU (`Stream.touch`, `Stream.acquire`; `dispatcher.py:2899-2932`).
- Capacity (K5): a draft fetch takes a free stream slot, else a shared extra while more than R_ver extras are free;
  otherwise it is REFUSED (counted) and the block stays a miss. The paper's "drafting pauses when it cannot reserve"
  is implemented as K3X's "the draft continues masked" (decision D3).

## 4. Background transfer, the transport parameter (requirement (b))

One interface, K3X's (`transfer.py:8-10`): `issue(layer, transfers)`, `poll()`, `done(tr)` (non-blocking),
`force(tr)` (host wait). A `Transfer` carries (seq, uid, layer, b, h, block, slot, gen, kind, requester, reader_ev,
ev, demand) (`transfer.py:74-102`). The destination slot is INFLIGHT and never reassigned while in flight (T3).

| transport | what moves the bytes | notes |
|---|---|---|
| `gilfree` (default for runs) | agent E's NativePipe: GIL-free C++ pack of host rows into a pinned staging ring (OpenMP), `copy_(non_blocking)` on a COPY stream (copy engine), `index_copy_` scatter on a SCATTER stream into the destination slots (`gilfree_pack.cpp:150-281`) | ring lifetime waits ev_h2d / ev_sc (`:16-18`); completion = the done event on the scatter stream. ADAPTER `nosi/nosi/r2_transport_gilfree.py` (packer v2, f2b981a): one `issue(layer, transfers)` = one `NativePipe.release` (groups b, h, l = layer, m = slot, blk) into the layer's store; T4 host-waits each `hist_ev` before the pack; T3 `scatter.wait_event(reader_ev)` (CPU: host wait); `set_threads(N)`, more than 1 thread refused from a build without OpenMP, every release's team checked (`threads_min == threads_max == N`, agent E's `threads_ok`); `background=True` runs the releases in one feeder thread pinned with `os.sched_setaffinity(cores)` (the OpenMP team inherits it). CPU tests GT1-GT3 (`test_r2_transport_gilfree.py`): bytes equal to `s0` on the same plans and to the host truth; T4; the team rule with agent E's no-OpenMP build as the negative control |
| `s0` | K3X S0Transport: `index_select` pack into pinned staging, H2D on the copy stream, `index_copy_` scatter (`transfer.py:105-415`) | the K3X reference transport; runs on the CPU backend for tests |
| `gather` | NOSI's zero-copy Triton gather `flash_h2d_from_mask` reading the pinned host rows, launched on a SIDE stream with a load mask over the store's slots (`flash_h2d_mask.py:65-88`) | uses SMs and HBM bandwidth: the GPU-gather baseline of paper 4.4 |
| `adversarial` | K3X AdversarialTransport: seeded delays, reordering, bursts; delivery waits for reader_ev (`transfer.py:418-509`) | CPU tests |

- The destination's previous reader (`reader_ev`, forwarded at take: `dispatcher.py:3143-3152`) gates the write:
  the scatter / gather waits on it on the device (S0 `transfer.py:194-196`); a host-side pack never writes a GPU slot.
- BIAS (the token's CIS) is NOT transferred: the cis table `twin.total_cis` is on the GPU (B, N_max, H), so the
  arrival of block t writes the slot's bias rows by a device copy of rows [64 t, 64 t + 64) of the request's cis on
  the same stream as the scatter (one small kernel).
- Completion: the host polls the transfer events at the start of every tick and at every layer resolve
  (`event.query()`, no sync); an observed arrival calls `Stream.admit(idx, uid, block, gen, demand)`
  (`dispatcher.py:1499-1566`).

## 5. Retention and versioned ownership (requirement (c))

- Live speculative block = a cohort key (uid, epoch, m) between its first draft row and its decision.
- References: the draft rows of cohort m protect their selections (section 3); the verifier rows of cohort m protect
  their selection with kind "verify" (`dispatcher.py:2568-2589`).
- Release (K4, amended R1): after the joint call in which cohort m executed layer l, its (key, l) references are
  QUEUED; `poll` drops them only after the (key, l) last reader event completed (`dispatcher.py:1553-1588`); at the
  decision, the layers it did not execute are queued the same way. A PO / VRES slot that loses its last reference
  goes RELEASING; RELEASING becomes FREE only after the slot's own last reader event (`poll_releases`).
- The logical cache (the draft LRU, capacity C) decides membership; retention never changes it. An LRU eviction of a
  protected block makes it PO and keeps its slot (K3): "removing an entry from the logical cache does not free its
  storage while an owner or reader remains" (paper 3.4).
- Cancellation (rejection at index i*, cancel of later cohorts, request end):
  - the cancelled keys' references are dropped through the same reader gate (`_apply`, `_cancel_requester`,
    `dispatcher.py:3408-3480`);
  - an IN-FLIGHT transfer of a cancelled requester keeps its slot until it lands (never freed on cancel: the
    `free_on_cancel` mutation is the negative control); at arrival: draft demand -> LRU (history is immutable),
    verify-only and unprotected -> RELEASING, request ended -> dropped (`admit :322-361`);
  - a cancelled key can no longer read: the tick never plans its rows (the round's epoch check,
    `r2_spec_core.Round`), and the service REFUSES a resolve for a key in `cancelled_keys` (OwnershipViolation).
- Reader events: every resolve returns the slots the rows will read; after the layer's attention the tick records ONE
  event on the main stream and installs it as the last reader of every slot read at that layer and of every (key, l)
  that read (`dispatcher.py:1671-1694`).

## 6. The verifier (requirement (d))

Per verify row (cohort key k, layer l, head h), the exact selection is classified at its FIRST LOOK (K3X 4.3):

- READY: resident (LRU / PO / VRES). Read in place. A verifier read does not touch the LRU (K1).
- IN_FLIGHT: INFLIGHT for any requester. The verifier merges its demand ("verify") and WAITS: before the layer's
  attention, the main stream waits on the transfer's completion event (`cudaStreamWaitEvent`, no host sync). A CPU
  pack that has not been submitted yet has no event: the host waits for its SUBMISSION (not its completion) first.
  Never fetched twice.
- NEW (an urgent miss, paper 3.2): neither. The verifier takes a slot (stream slot, else any free extra, the R_ver
  reserve included; under pressure an unprotected LRU entry is evicted, counted: `dispatcher.py:3128-3129, 3187-3207`)
  and the block is GPU-GATHERED ON THE MAIN STREAM before the attention of that layer (`flash_h2d_from_mask` with a
  load mask over the store's slots), demand {verify}. At its observed arrival it becomes VRES (verify-only) or LRU.
- If no slot can be taken even after evictions (all protected), the layer cannot run: the service host-waits the
  needed in-flight transfers and pending reader events and retries (K3X `_force_for_layer`, `dispatcher.py:2416-2443`);
  a second failure is a CapacityRefusal (a configuration error: C + X too small for the cohort's union).
  SUPERSEDED for v1 and S-C by POLICY E (note 12h, 2026-10-21): the eviction candidate is the least recent LRU entry
  that is unprotected and selected by no row of the layer; it is evicted and its slot reused at once; there is no
  force-and-retry path; no candidate is a CapacityRefusal.

v1 RESOLUTION IS HOST-SIDE PER LAYER (decision D2), for draft and verify rows alike: at every layer the tick copies
the rows' selection to the host (one D2H, then one sync), the service classifies, protects, issues the background
requests, the urgent gathers and the stream waits, and returns the page tables (one H2D). This mirrors K3X's
per-layer host loop and is the correctness reference. Its cost (one sync per layer per tick) is dHOST in SG5. Later
stages move the draft rows (S-B: a device mirror of the stream maps, the selection sent to the host asynchronously)
and then the verifier (S-C: a pre-reserved slot pool and an overflow flag) to the device; section 13.

## 7. Memory budget (paper 3.4)

Per layer, physical (decision D4: K3X's M1 per-stream backing, `ownership.py:31-33`):

- the store: B x H x (C + X + 4) slots of 64 rows; a slot = K + V + bias = 64 x (2 x 128 x 2 + 2) B = 32.1 KiB per
  (request, head); the 4 extra slots are the request's tail rings (2 exact + 2 draft pages, note 5.2) at fixed
  indices C + X .. C + X + 3, so ONE tensor serves piece 3's page tables (draft and verify rows alike);
- the transport: the pinned staging ring (host) and the landing ring (GPU) of the packer;
- metadata: the stream maps (host), the page tables (GPU, per tick).

The logical limit is K3X's: |LRU| <= C per stream, extras borrowed <= X per layer, the verifier reserve R_ver.
Retained bytes = (PO + VRES) x block bytes; peak_retained per K3X 4.6. Example at B 64, 32 layers, H 2:
C + X + 4 = 100 slots per (request, head) = 32,896 B x 100 x 2 x 32 x 64 = 13.5 GB, against the shipped window's 64
slots = 8.6 GB (and the 16.4 GB model). The sizes are run parameters; SG5 reports the peak.
The captured tick (Q-T2) adds ONE TRASH SLOT per (request, head, layer) after the tail rings, at C + X + 4
(`Residency(trash_slots=1)`; counted in `memory()["slots_physical"]`; CPU test F15): C + X + 5 = 101 slots, i.e.
32,896 B x 101 x 2 x 32 x 64 = 13.61 GB (+0.13 GB), about +5.0 GB against the 64-slot window. Only the padded rows of
the captured tick write and read it; it is outside every LayerPool (no take, no transfer, no gather) and outside the
tail range (no Row can name it).

## 8. The interface piece 4 calls (one service per run)

```
svc = Residency(L, B, H, C, X, R_ver, transport, urgent_gather, store_b=, cis=, block_bytes=, host_wait=)   # v1
svc = ResidencySB(..., device=, nblk=)              # S-B (r2_residency_sb.py): the draft rows on the device
svc = ResidencySC(..., device=, nblk=, devops=)     # S-C (r2_residency_sc.py): verify and draft rows on the device;
                                                    # devops = CudaDeviceOps(svc, host_k, host_v, store_k, store_v,
                                                    # store_b, cis) on the GPU (the CPU stand-in by default); the
                                                    # mirror lives on the device (12d); sel must be on the device
svc.begin_request(b, uid, hist_blocks=S // 64)      # the prefill wrote the host rows of blocks < hist_blocks
# a tick
svc.poll()                                          # arrivals (admit), queued releases (reader gates), FREE
for l in layers:
    sel = selection of all rows ((H, R, k) int64; on the device for S-B; tail included)
    plan = svc.resolve(l, rows, sel)                # SAME signature in v1, S-B and S-C; rows: Row(b, kind, key,
                                                    # pos, hist_bound, tail); v1: a host read per layer; S-B: only
                                                    # for the verify rows; S-C: none (one resolve per layer per tick,
                                                    # the same rows at every layer; the first layer sends the tick's
                                                    # mirror delta and row inputs)
    # plan.page_table (R, H, P) int32, plan.num_pages (R, H) int32, plan.last_len / seq_idx (R,) int32 over the store;
    # the urgent gathers and the stream waits are already enqueued on the main stream. S-B: plan.reads / plan.blocks
    # hold the verify rows only (the draft reads are applied at end_tick)
    out = bi_paged_decode(q, store_k[l], store_v[l], store_b[l], plan.seq_idx, plan.page_table, plan.num_pages,
                          plan.last_len)
    svc.read_done(l, plan, torch.cuda.Event recorded on the main stream)
svc.end_tick()                                      # S-B: ONE device read, the draft decisions applied, the draft
                                                    # transfers issued; v1 (defer_issue): the draft transfers issued;
                                                    # S-C: ONE device read for verify and draft decisions; an
                                                    # overflow flag raises CapacityRefusal (the tick is invalid)
svc.history_written(uid, block, ev)                 # the tick wrote block back to the host rows; ev = its D2H copy
svc.decided(key)                                    # the verified cohort: its references queued behind the readers
svc.cancel(key) / svc.cancel_epoch(uid, epoch)      # a rejection: every later key of the old epoch (draft-only keys too)
svc.end_request(uid)                                # K3X end_owner: LRU / PO / VRES released, in-flight stays in flight
svc.report()                                        # 4.3 / 4.6 / 4.8 counters, refusals, masked fraction, memory
```

## 9. K3X call sites mirrored (re-implemented for the R2 tick, cited)

| R2 service | K3X dispatcher (91e6e47) |
|---|---|
| draft rows: protect + requests + demand merge | `:2592-2629`, `:2955-2970` |
| verify rows: look, protect, merge, ready / wait / new | `:2568-2589` (m1), `:1325-1382` (rev6), `_need :2492-2503` |
| issue: take, reader forwarding, Transfer | `_issue :3098-3179` |
| poll: arrivals, admit, releases, reference drops | `_poll :1499-1566`, `_release_refs :1568-1588` |
| reader events | `_install_reader :1671-1694` |
| draft recency | `_draft_recency :2899-2932` |
| force path | `_force_for_layer :2416-2443` |
| rollback, cancel | `_apply :3367-3430`, `_cancel_requester :3469-3480` |
| request end | `_end_request :831-875` |

## 10. Decisions (ACCEPTED by the lead, 2026-10-20 / 21)

- D1 REUSE BY VENDORING: K3X's `kstep/{__init__, errors, backend, executor, ownership, transfer, mutations}.py` are in
  this branch BYTE-IDENTICAL at 91e6e47 (`git checkout 91e6e47 -- <paths>`; CPU test F9 checks each blob id). They
  are never edited; the R2 specifics live in `nosi/nosi/r2_residency.py`. CREDIT: the ownership model, its states,
  rules K1-K5, the transfer record and the transports are K3X's (tracks B / C, `nosi-kstep-bc`); this piece adds the
  R2 call sites, the store layout, the bias rule and the R2 metric recorder.
- D2 v1 resolution is host-side per layer (section 6): the CORRECTNESS REFERENCE. SG5 reports its cost (dHOST)
  SEPARATELY. S-B (device-side draft rows) and S-C (device-side verifier) are REQUIRED before any throughput claim
  that includes this service.
- D3 a refused draft fetch continues MASKED (K3X). The service records the refused-fetch count and the masked
  fraction (draft (row, head) pairs that did not read every selected block) per resolve; the tick sums them per tick.
  The paper's 3.1 "pauses until verification catches up" (the lead bound) stays; its 3.4 "drafting pauses when it
  cannot safely reserve the space" must become the masked behaviour (on the author's decision list).
- D4 per-stream physical backing (M1: B x H x (C + X + 4) slots per layer). The extra memory against the shipped
  64-slot window (+4.9 GB at B 64 for 100 slots per stream) is charged in the equal-memory budget and reported. The
  shared arena (M2) comes later if memory needs it.
- D5 transport default for GPU runs: `gilfree` (its GPU step PASSED its registered prediction, 2026-10-21:
  cpu_native:64:32 +0.96 % decode slowdown with a bounded backlog; cpu_native:192:96 +1.82 % with a GROWING backlog,
  `/mnt/beegfs/ojerbi/nosi-sustained-gilfree/results/sustained/b64_s512_gilfree`); `s0` as the fallback; `gather` as
  the 4.4 baseline.

## 11. Metrics on the R2 path (requirement (e)); K3X definitions verbatim (`paper_metrics.py:9-37`)

- 4.3: per layer_call (one verifier cohort at one layer): ready / in_flight / new (blocks and bytes),
  in_flight_missing_at_look, wait_ns per block, layer_wait_ns, abandoned.
- 4.6: served by lru / po / vres (retained) / fetched_first / fetched_refetch / not_resident; peak_retained.
- 4.8: every transfer = one slot lifetime, classified draft_only / verify_only / both / neither, still_live,
  never_arrived, by issuer; repeated = transfers minus distinct (request, layer, head, block).
- 4.9 (the packer): packed bytes and rate, queue length over time, the fraction of verify-demanded blocks that
  arrived before their first look, CPU time per packed MB (from NativePipe's release counters).
- Bytes: block bytes = 64 x (K + V) per (request, head); the bias is a device copy and is reported separately.

## 12. CPU stand-ins and tests (next commit), each with a negative control

Module `nosi/nosi/r2_residency.py` (torch only) over the vendored K3X ownership and transports; tests
`benchmarks/Efficiency/tests_cpu/test_r2_residency.py`. The store tensors are CPU tensors; events are CPU objects
with a done flag (the shape of K3X's CPU backend events); the adversarial transport delays, reorders and bursts
deliveries ("force_only" for the scripted tests). The negative controls are K3X mutation switches where one exists
(prot_release_no_reader_gate, release_drops_all_keys, free_on_cancel, no_reader_wait, release_first_reader,
vres_counted_as_lru) and defect subclasses of the service otherwise (InflightReadable, NoDedup, DraftTakesReserve).
F5b (the verifier reserve) and G10 in `test_r2_g0bi.py` (the serial path refuses rows below position 4032, the verify
agent's Q4) are added. F10-F10d (S-B, 12b), F11 (history_written and the host write-back order: K3X T4; cancel_epoch)
and F12-F13 (S-C and the next request of a batch slot, 12c) follow.

| Id | Test | Negative control (must be caught) |
|---|---|---|
| F1 | draft resolve: pages = tail + READY selected blocks only, ascending; misses issued once per (stream, block) across rows and heads; in-flight blocks merge the demand | a page list with an INFLIGHT block; a duplicate transfer |
| F2 | retention: every block selected by a live cohort (hits included) is never FREE / re-taken until the cohort's (key, l) reader event of its verify read completed | `release_first_reader`, `release_drops_all_keys` (K3X mutations) |
| F3 | a late transfer of a cancelled cohort keeps its INFLIGHT slot until it lands, and the cancelled cohort never reads it (resolve refuses the key; the round never plans it) | `free_on_cancel` (the slot re-taken under the transfer: caught by the content check); a resolve by a cancelled key |
| F4 | a slot is reused only after its last reader event: the transport's write waits on reader_ev; RELEASING -> FREE only after the event | `no_reader_wait` (the reader sees overwritten rows: caught) |
| F5 | verify resolve: ready / in-flight / new counts; the stream waits only on the transfers of in-flight selected blocks; an urgent gather never targets an in-flight block; the R_ver reserve serves the verifier | an urgent duplicate of an in-flight block; a draft fetch that takes the reserve |
| F6 | END-TO-END CONTENT under the adversarial transport: every verify row's pages hold, row by row, the host truth of its blocks (K, V and bias), so the verify output equals the all-resident reference (the BI twin) at every step; draft rows read only arrived content | a stale-generation arrival accepted (StaleGeneration expected) |
| F7 | memory: physical slots <= B x H x (C + X + 4) per layer at every step; peak retained = max (PO + VRES); the refused draft fetches counted | an accounting that misses a VRES slot |
| F8 | metrics: a scripted trace gives exact 4.3 / 4.6 / 4.8 counts and bytes | a transfer counted twice (repeated off by one) |
| F9 | the vendored K3X files are byte-identical to 91e6e47 (blob ids) | a modified copy is refused |

GPU gates (after the CPU tests and piece 4's tick): FG1 content and reader safety with real streams and the gilfree /
s0 / gather transports (the F6 property with CUDA events); FG2 the 4.3 / 4.6 / 4.8 metrics on the R2 path at B 8 / 64
with real selections; FG3 the packer's sustained rate (4.9) against the tick's demand.

## 12b. S-B: the draft rows on the device (`nosi/nosi/r2_residency_sb.py`, 2026-10-21)

- Per layer, no host read for the draft rows: after the host-side verify rows (v1), the host refreshes the layer's
  DEVICE MIRROR (block -> resident slot, in-flight flags, FREE slots, usage per stream; a host-to-device copy) and
  `device_draft` decides with tensor ops: READY blocks and the page tables, the protections (hits included), the
  dedup per (stream, block) (in flight, or a later duplicate of the layer), K5 grants and refusals in K3X order (a
  stream's first C - usage takes are free; extras go to the first X - borrowed - R_ver extra-needing misses), the
  slot of each take (the stream's q-th FREE slot = K3X first_free), the masked and refusal counters.
- At end_tick ONE batched device read; the host APPLIES the decisions to the K3X state in v1's order and CHECKS each
  (a read of a block the host does not hold there, a merge of a block not in flight, a take K5 or first_free would
  not make, a refusal K5 would grant: OwnershipViolation), then the O1 recency, then the draft transfers.
- v1 now follows K3X's ordering contract O1 exactly (every draft row's READY set from the state before the recency
  update; the recency after all draft rows), and admits arrivals per stream in issue order (layer, b, h, seq).
- The S-B schedule: arrivals polled at the tick's start; draft transfers issued at end_tick. Under it S-B equals v1
  (Residency(poll_in_resolve=False, defer_issue=True)): F10 (F6's 24 ticks, verify rows every third tick, seeded
  delays / reordering / bursts) and F10b (C 2, X 4, R_ver 2, refusals): page tables, ownership state (LRU order,
  PO, VRES, protections with kinds, slots, in-flight transfers, the transport's pending list) and counters, every tick.
  Negative controls (F10c): no dedup, K5 without the reserve, no hit protection. F10d: device_draft's source holds no
  host read. Device reads counted: one per tick (+ one per layer of a verify tick, v1's verify rows).

## 12c. S-C: the verify rows on the device (`nosi/nosi/r2_residency_sc.py`, 2026-10-21)

- Per layer, NO host read: the host refreshes the layer's mirror (S-B's, plus the slot and generation of the owner's
  transfer in flight per block), then `device_verify` decides the verify rows in K3X order (verify row, head,
  ascending block): READY (resident), IN FLIGHT (the mirror's transfer: the read waits for it on the device), NEW (the
  first occurrence per (stream, block) takes a slot; a later occurrence, another verify row of the cohort, reads the
  same slot: one transfer per stream and block). K5 for the verifier: a stream's first C - usage takes are local,
  later ones take a shared extra while any is free, the reserve R_ver included (the PRE-RESERVED VERIFIER POOL: draft
  fetches never borrow its last R_ver extras); the slot is the stream's q-th FREE slot (K3X first_free). A first miss
  that K5 cannot grant sets the OVERFLOW FLAG (its page points at slot 0, in bounds) and end_tick raises
  CapacityRefusal before any change of the host state: the tick is invalid and the run stops (C + X or R_ver too
  small). v1 instead evicts an unprotected LRU entry or host-waits releases there (`dispatcher.py:2416-2443`); S-C
  does neither, so the equality tests assert that v1 never did either.
- The device then waits for the in-flight blocks, gathers the urgent blocks (K, V, bias) on the main stream before the
  layer's attention, and updates the mirror (urgent blocks in flight, their slots taken, usage); `device_draft` (S-B)
  then decides the layer's draft rows on it.
- At end_tick ONE batched device read of every layer's verify and draft decisions; per layer the host applies the
  verify decisions in v1's order and checks each (a ready slot the host does not hold, an in-flight slot or generation
  the host does not have, a take K5 or first_free would not make, a valid block without a slot and without the flag:
  OwnershipViolation), then the draft decisions (S-B).
- DEVICE OPERATIONS. `CpuDeviceSim` (the CPU tests): the needed transfers land (the transport's force, exactly v1's
  waits); the gather copies K, V and the cis rows, host-waits the block's write-back (T4) and refuses a slot whose last
  reader is pending (T3). `CudaDeviceOps` (the GPU; SOURCE ONLY, NOT RUN: GPU gate FG1): the main stream waits
  (`cudaStreamWaitEvent`) on the completion event of EVERY transfer in flight at the layer, a SUPERSET of the needed
  ones because the host does not read the selection (a gilfree pack not submitted yet is host-waited for its
  submission, Q2); a wait changes no ownership state, only when the main stream runs. The gather is NOSI's
  `flash_h2d_from_mask_bias` (K and bias) and `flash_h2d_from_mask` (V) with the decision's device load ids (H, B,
  C + X), after a main-stream wait on every host write-back registered since the last gather (T4); its slots are FREE,
  so their last reader completed (K4, T3). The exact alternative to the superset wait (a device spin on the slot
  generation that the scatter stream publishes) is left for a measured need.
- The S-C schedule (v1's reference): arrivals at the tick's start, draft transfers at end_tick, the waited and the
  urgent blocks read IN PLACE (their INFLIGHT slot) and admitted at the next poll: `Residency(poll_in_resolve=False,
  defer_issue=True, defer_admit=True)`.
- Tests (`test_r2_residency.py`): F12 S-C == v1 on F6's 24-tick schedule (pages, verify page content at the read
  point in both worlds, state, counters every tick; ready, in-flight and new blocks met; one device read per tick);
  F12b under capacity pressure (C 2, X 96, R_ver 88, 8 blocks, two overlapping verify rows per cohort: refusals, the
  verifier borrows beyond the drafts' limit, one transfer per block); F12c negative controls (a read before the
  in-flight block landed: caught by the content check; no verify protection; a second transfer of one block for two
  verify rows); F12d the source check (device_verify, device_draft, rank_in_group, resolve, the mirror refresh, the GPU
  operations: no host read, no `_d2h`; S-B's resolve is caught); F12e the urgent gather never writes a RELEASING slot
  (negative: RELEASING counted as FREE, caught by the device's T3 check and, with it off, by the host's first_free
  check); F12f the overflow flag (CapacityRefusal at end_tick; v1 refuses at resolve; negative: no flag, the host
  refuses the hole); F13 a transfer of an ended request is never merged with or read by the next request of the batch
  slot (negative: an owner-blind lookup reads the old rows).
- FIXES IN v1 found while building S-C (each covered above): (1) a block that an EARLIER verify row of the same layer
  fetched urgently was also put on the wait list (a force of a transfer the transport never had); (2) the 4.3 / 4.6
  looks counted a block once per verify row: now once per (cohort, layer, head, block), the first look classifies it;
  (3) the in-flight lookup ignored the stream's owner, so the next request of a batch slot could merge with (and,
  under defer_admit, read) an ended request's transfer (K3X keeps only the current owner's transfers in
  Stream.pending). device_draft's (N, B x H) one-hot ranks are replaced by a sort-based rank (O(N) memory; same
  values; F10 re-checks).

## 12d. S-C's mirror on the device, one delta per tick (Q5, 2026-10-21)

- `DeviceMirror`: the ownership state the device decides on, for EVERY layer, as a per-slot table on the device
  (L, B, H, C + X): state code, block, generation, and the flag "the current owner's transfer in flight". Each resolve
  derives its layer's mirror (S-B's fields plus the in-flight slot and generation per block) from the table with
  device ops only (`DeviceMirror.layer`).
- The delta: the vendored K3X ownership code logs every stream transition (`Stream.log`: take, release, evict_po,
  acquire, admit, admit_vres, drop_unowned, drop_vres_unprotected, free; `ownership.py:206-450`). S-C replaces each
  stream's log list by a list that also reports every append, so the slot becomes dirty AS IT CHANGES (no scan; the
  vendored files are unchanged). An owner change (begin_request / end_request) marks the batch slot's transfers in
  flight (their "current owner" flag changes). At the first resolve of a tick, `sync_mirror` sends the records of the
  dirty slots in ONE host-to-device copy (pinned, asynchronous on the GPU) and the table takes them. The tick's row
  inputs go in one more copy (every layer of a tick resolves the same rows; a later layer with other rows is refused).
  A host change in the middle of a tick is refused (the mirror would be stale). Per tick: at most two host-to-device
  copies and one device-to-host read; per layer: none.
- Tests: F14 (the delta-maintained mirror equals the full scan, `VerifyMirror.refresh`, at every layer and in every
  field at the start of every tick of F6's schedule and of F12b's pressure schedule; deltas smaller than the table);
  F14c negative controls, one per transition kind (take, admit, release, admit_vres, free), each with its PRECONDITION
  asserted first on the run without the drop: in some closed sync interval a slot is changed by that kind ONLY (a
  test-side trace of the journal and the owner marks per sync interval); then the run with that kind dropped from the
  delta must leave the device mirror unequal to the full scan; F14b (an owner change with a transfer in flight that
  the device already holds; precondition: the owner change is the slot's only change; negative: the owner change
  dropped); F12d extended (no host-to-device copy in the per-layer path; S-B's resolve and the full-scan refresh are
  caught). F12 / F12b now run on the delta mirror and count one row copy and at most one mirror copy per tick.
- WHY admit_vres AND free NEED THEIR OWN SCENARIOS (job 2190925: their F6 controls were not caught). A dropped kind can
  leave a stale record only when it is the slot's only change between two syncs; any other change of the slot in the
  same interval sends the slot's CURRENT record anyway. (1) admit_vres: only a transfer whose demand is {verify} alone
  is admitted VRES, and that is only a verifier urgent transfer (a draft transfer always carries "draft"). S-C takes it
  at end_tick and admits it at the very next poll (it landed in place), and S-C's own sync is at the next tick's first
  resolve, after that poll: so in S-C's own schedule its take and its arrival are ALWAYS in one interval, and a dropped
  admit_vres cannot leave a stale record. The control (_scn_split) therefore syncs between end_tick and the poll (an
  urgent block that no draft row merges, its key still protecting it at the arrival). (2) free: a slot frees in the
  poll that sees its last reader complete. In F6's 24 ticks no free was a slot's only change in an interval (job
  2190925); the likely reason: the released slots' last readers had already completed, so the release and the free
  fell into one interval (not traced). The control (_scn_late_reader) releases a slot whose last reader (a
  verify read) is still pending, syncs while it is RELEASING, then completes the reader so that the next poll frees it.

## 12e. Watched device flags in S-C's one read per tick (Q-T5, 2026-10-21)

- `ResidencySC.watch_flag(name, flag)`: a 0-dim device tensor (the twin's sticky `overflow`,
  `r2_positional_twin.py:152`) is read with S-C's ONE device read at every end_tick (also on a tick without rows). A
  nonzero value refuses THAT tick (CapacityRefusal) before any of its decisions is applied; `svc.flag_values` holds
  the values of the last read. A twin overflow so voids one tick, not every tick since the last decision read, at no
  extra sync.
- Test F16: a real twin (prompts of 38 tokens, N_max 41) overflows in its step of tick 3, a draft-only tick; S-C
  watching it refuses tick 3 at its own end_tick (one device read, the ownership state and the applied counters
  unchanged), ticks 0-2 ran normally. Negative control (no watch): no end_tick refuses, tick 3 and later apply their
  decisions, and the flag is seen only after them.

## 12f. The EXACT device wait (Q-T3 (b), Q6; design and CPU stand-in, 2026-10-21)

The superset wait (`CudaDeviceOps(wait_mode="superset")`) makes the main stream wait on the completion event of EVERY
transfer in flight at the layer, and host-waits a gilfree pack that is not submitted yet. Before a captured replay it
would wait for every in-flight transfer of every layer: no overlap. The exact wait waits for exactly the slots that the
layer's verify rows read in flight, on the device, with no host step:

- THE BOARD (`GenBoard`, owned by S-C, attached to the transport): (L, B, H, C + X) int64 on the device; per slot, the
  generation of the last transfer whose rows LANDED there (0: none). A take makes gen >= 1; a slot's generations only
  grow; a slot in flight is never taken again (T3). So "board >= gen" means "the awaited transfer landed".
- PUBLICATION: the transport writes (slot, gen) into the board on the stream that landed the rows, AFTER them.
  gilfree (`GilfreeTransport.attach_board` / `_publish`): after each `NativePipe.release`, a pinned copy and an
  `index_put_` enqueued on the SCATTER stream (CPU mode: at once), before the release's completion event. The tests'
  AdversarialTransport cannot publish (vendored): `PublishingTransport` wraps it and publishes every transfer it sees
  landed after a poll / force, with the same deliveries and pending list (the stand-in of the scatter stream).
- THE WAIT LIST (`wait_list`): from `device_verify`'s decision, per entry the board index of its in-flight slot (-1 if
  none) and the generation; fixed length, on the device: capturable.
- THE KERNEL (`_wait_kernel`, Triton, SOURCE ONLY, NOT RUN): ONE program on the main stream (one SM; the scatter
  stream keeps the others), the list in blocks of 1024, VOLATILE loads of the board, spinning until every entry
  reached its generation or `max_spin` rounds passed; then an entry short or past raises `err` (int32, atomic max).
  The board write that ends the spin is enqueued after the scatter, so the rows are complete when the main stream's
  next kernel (the attention) reads them. In the eager tick a CPU pack not yet submitted is host-waited for its
  SUBMISSION first (the liveness rule of 12g); its completion is waited on the device.
- NO HANG: `err` is watched by S-C (`watch_flag`, 12e): a spin that ran out refuses its tick at that tick's end_tick.
- `CudaDeviceOps(wait_mode="exact")` launches the kernel per layer; `"superset"` keeps the event waits. FG1 runs both
  and reports per layer the 4.3 layer_wait_ns, per tick the ms, the submission host-waits (superset only) and `err`.
- CPU stand-in: `ExactWaitSim` reads only device data (the wait list and the board; never the host's transfer
  records); a spin round is the stand-in of time (`PublishingTransport.progress_slot`: the transfer into that slot
  lands now); at most `max_spin` rounds; then `err`. `ResidencySC(wait_mode="exact")` selects it on the CPU.
- Tests: F17 (S-C with the exact wait equals v1 over F6's 24 ticks: pages, verify content, state, counters and the
  device mirror every tick; the wait spun, `err` False; the board is never ahead of a slot's generation); F17b (a
  REUSED slot: generation 1 published, generation 2 in flight: the wait spins until 2, the verifier reads the right
  block; negative controls: a wait that accepts the previous generation reads the old block (content check); a
  transport that never publishes runs the spin out and the watched `err` refuses the tick); GT4 (gilfree publishes
  exactly its transfers' (slot, generation) after the release; negative: no board attached, no publication).
- FIRST GPU COMPILE (fg5 on holder 2190876, 2026-10-21): the kernel failed with "NameError: tl is not defined".
  - Cause: it was built inside the factory `_wait_kernel`, whose local imports of triton / triton.language were
    invisible to Triton's GPU compiler. That compiler resolves a kernel's names in its module globals
    (fn.__globals__); in the nested kernel, `tl` was a closure variable. No CPU test compiled the kernel.
  - Fix: `board_wait_kernel` is now defined at MODULE level, with guarded module imports of triton and
    triton.language (as max_pooling_per_request.py does). `_wait_kernel()` returns it, or refuses without triton.
  - Test F23: for every @triton.jit kernel of the R2 line (r2_*.py, max_pooling_per_request.py, the h2d kernels),
    the kernel has no free variable, and every global it loads resolves in its module or in builtins.
    Precondition: board_wait_kernel is found. Negative control: a planted kernel with a local-only import fails.

## 12g. The GPU gates FG1 - FG3 on the real tick (`benchmarks/Efficiency/r2_fetch_fg.py`, 2026-10-21)

The gates live on branch `feature/nosi-r2-m2-fg` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-fg`, 0781b1d, accepted with
CPU job 2190942): the verify agent's SG5 795dbc1 (the tick with its ResidencyBackend) with this branch's 41078e8
merged. Step `benchmarks/Efficiency/r2_fetch_fg_step.sh`; CPU tests of the plumbing
`benchmarks/Efficiency/tests_cpu/test_r2_fetch_fg.py` (FGC1-FGC6); the gather transport
`nosi/nosi/r2_transport_gather.py`. THIS branch carries the gates' library changes (ported, with tests F18 / F19 and
GT4): the liveness rule below, `ResidencySC.journal_counts`, and gilfree's `stats["published"]`.

- SETUP: B 8 distinct ragged documents (S 16128 minus the tick gate's deltas), one build_ragged; every engine a
  from_r2 copy; TEACHER = the serial mode's greedy tokens (SG0's BI serial decode); RES = the ALL-RESIDENT tick
  (ResidentStore) teacher-forced with forced_schedule, 64 committed tokens per request: its verify rows (float32, with
  their (request, position)) are the reference of every S-C arm, which MUST give the same row sequence and BITWISE equal
  rows. The host rows past the prompts are zeroed before every S-C arm.
- FG1 THE WAIT (gilfree, C 80 / X 32 / R_ver 16): superset vs exact; per layer the device time of the wait (CUDA events
  around it on the main stream; in S-C both the look and the read are on the device, so this is the 4.3 layer wait)
  and its host time; ms per tick; the submission host-waits; exact: err 0, published == packed transfers, no board entry
  ahead of its slot's generation. NEGATIVE: exact without publication (the board never attached), max_spin 65536: a
  tick is refused (CapacityRefusal naming wait_err) within 900 s.
- LIVENESS RULE (found while writing FG1): in the EAGER tick the exact wait first host-waits the SUBMISSION of every
  transfer in flight at the layer (`CudaDeviceOps._submitted`: a threading.Event, the GIL released), then spins on the
  device. Without it the main stream could spin on a pack that the feeder cannot submit, because the host thread holds
  the GIL while it blocks in a kernel launch behind the spinning stream (the launch queue is full). The completion is
  still waited on the device and only for the needed slots. In a captured tick the submission waits move before the
  replay (the tick-capture plan 8.8).
- FG2 THE TRANSPORTS (superset wait): gather (NOSI's Triton flash_h2d_from_mask on a side stream; T4 / T3 as side-stream
  event waits; the board published on the side stream), s0 (K3X S0Transport over cpupack_core 91e6e47, its CUDA
  backend; n_pos = C + X + 4 so the destination rows match the store with its tail rings) and gilfree (8 threads):
  bitwise RES, and every 8 ticks every resident slot holds the host truth (K, V, bias); the 4.3 / 4.6 / 4.8 report per
  transport.
- FG3 FETCH PRESSURE (exact wait, gilfree): C:X:R_ver 72:128:64 and 64:256:128 (C below the primary 80: blocks are
  evicted and fetched again; X large enough that the verifier's urgent misses find extras at B 8 x H 8 = 64 streams per
  layer). Every completed point bitwise RES; a CapacityRefusal is reported as a configuration result. Precondition: a
  completed point re-fetched (4.8 repeated > 0) and evicted LRU entries (`ResidencySC.journal_counts`: evict_po,
  release:evict). NEGATIVE: NoHostCopy (a write-back that does not copy the rows to the host) differs or refuses.
- STEP_MIN 90 (estimate: model and build about 6 min, the extension build about 2 min, TEACHER and RES about 3 min,
  8.5 S-C arms at about 2 - 4 min each: S-C's host applies are Python loops over B x K x H x 64 entries per layer).

## 12h. The first GPU run (FG fg1 and SG5 p1, holder 2190876): two defects and their fixes (2026-10-21)

Both runs failed before any tick with verify rows finished
(`/mnt/beegfs/ojerbi/r2-runs/fg/fg_fg1_2190876/summary.json`, `/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p1_2190876`).

DEFECT 1, every gilfree arm: "S and S_dst must be multiples of R" (gilfree_pack.cpp order_and_check).
- E's R contract (FIX_DESIGN.md at f2b981a): R = 64. The sources are contiguous TOKEN-MAJOR (B, S, H, D) CPU
  tensors with S a multiple of 64. The destination is a (rows, D) view of a token-major store of the same B, H, D and
  dtype, with whole 64-row slots. The error does not say which side failed.
- Which side failed: the SOURCE side, by its row count, not by its layout.
  - The host rows are token-major, (B, n_host, H, D) (R2Body.allocate, r2_decode_body.py).
  - n_host = S_max + horizon + 64 per request (r2_decode_path.ragged_widths), not 16128 + 64:
    SG5 p1: 16128 + (48 + LAMBDA 7 + 64) + 64 = 16311; FG: 16128 + (64 + 16 + 80) + 64 = 16352.
    Neither is a multiple of 64.
  - The destination holds: the store is (B, (C + X + 4 [+ 1 trash]) x 64, H, D) (TickEngine), whole slots.
- The CPU tests used S_host = 12 x 64, so the defect stayed hidden.
- FIX (no packer change; E's R contract stands): the adapter hands the packer one zero-copy source per
  (layer, request), `host[l][b:b+1, :S_floor]`, with S_floor = (S_host // 64) x 64. It addresses the request in
  the destination row: m' = b x n_slots + slot.
- The adapter checks at construction, and each message names its side:
  - the host rows and the store are 4-D token-major tensors of the same B, H, D and dtype (a head-major host is
    refused, not read as S = H);
  - the host holds at least one complete block (SOURCE);
  - the store's rows per request are a multiple of 64 (DESTINATION).
- At release, the adapter refuses a block past S_floor. Such a block is never complete, so the service never
  requests it (the drivers pass nblk = n_host // 64).
- CPU test GT5: the production geometries (S_host 16311 / 16352 / 16192; C + X + 4 = 116 slots, with the trash slot
  117; C 96: 132). The bytes equal the host truth: the last complete block, and the first and last slot of a
  request. A block past the floor is refused.
- GT5 negative control: the former batched call is refused by the packer itself, for every S_host that is not a
  multiple of 64.
- CPU test GT5b: the construction refusals (a head-major host, a store of partial slots, a host without a complete
  block, a store of another B); positive control: a token-major host of 5 x 64 + 55 rows is accepted.

DEFECT 2, FG2 gather and s0 at C 80 / X 32 / R_ver 16: CapacityRefusal at layers 18 and 23 (10 and 6 blocks).
What the data shows (FG2_gather):
- NOSA-8B has H = 2 KV heads (slots_physical 59392 = 32 x 8 x 2 x 116), so a layer has 16 streams at B 8.
- 3 draft-only ticks ran. The 4th tick, the first with verify rows, refused.
- The drafts took 36311 blocks: 70.9 per stream. Journal: take 36311, admit 36311, evict_po 47, no release:evict.
  No cohort had been decided (decision_reads 0), so every resident block was protected by a live draft cohort.
- So no policy could free a slot there. v1 too would refuse: its eviction needs an unprotected LRU entry.
- The verifier had only C - usage, about 9 local slots per stream on average, plus at least 16 shared extras
  (X 32 minus the drafts' borrowing; drafts may borrow X - R_ver = 16). It needed about 10 more at layer 18.
- Its first-verify demand is therefore about 16 x 9 + 16 + 10 = 170 urgent blocks per layer, about 11 per stream.
CONCLUSION:
- (a) This refusal is a CONFIGURATION result. The first verify tick needs C above the drafts' protected union plus
  the verifier's urgent misses, per stream: about 71 + 11.
- (b) The data also shows the POLICY GAP. Once the LRU is full (usage >= C), every urgent miss of S-C needs a shared
  extra, because S-C could not evict. Those extras do not scale with B: per-stream backing gives every stream X
  physical extra slots, but the whole layer has only X in total.
- v1 (and K3X) instead evict an unprotected LRU entry. So both changes are made.

POLICY E (v1 `_verifier_candidates` / `_evict_now`; S-C `device_verify`, `_apply_verify`):
- Trigger: K5 grants the verifier's urgent miss no slot.
- Choice: the stream's least recent LRU entry that is UNPROTECTED at the start of the layer and selected by NO row
  of the layer (draft or verify), so no read of the layer uses it.
- It is evicted (release: verifier_evict) and its slot reused at once. Its last reader read it in an earlier tick,
  so on the GPU that read is ordered before the main-stream gather. On the host the reader is host-waited if
  pending (counted: verifier_force_waits). Then poll_releases' FREE transition runs for that one slot.
- With no candidate: CapacityRefusal, the configuration case.
- K3X's former force path is not used. It host-waits every RELEASING gate of the stream and polls the whole pool
  mid-tick, which a device decision cannot do.
- S-C's device mirror now also carries, per slot, the protection flag and the LRU recency stamp:
  - touch / admit / acquire give the slot a new stamp;
  - the stream's protection map reports protection changes (_ProtDict);
  - both go with the one delta per tick (F14 compares the lru_rank and prot fields too).
- The host checks every device eviction against v1's candidate list.
- S-B keeps v1's verify rows but sees no draft selection there. Its candidates spare only the verify rows' blocks:
  S-B is superseded and is not equal to v1 once evictions happen. It stays safe: its draft decision reads the mirror
  refreshed after the verify part, so an evicted block is a miss there, never a read of the reused slot. The S-B
  lockstep tests (F10) use C 8 / X 48 / R_ver 8 on F6's schedule (F12 asserts that v1 makes no verifier eviction
  there) or no verify rows (F10b); if F10 met an eviction, its lockstep would report the difference.

Tests:
- F20 (scripted, one stream, C 2, X 0):
  - all protected: both refuse;
  - the oldest protected: evict the next;
  - the oldest read by the tick's draft row: evict the next;
  - none excluded: evict the oldest.
  - v1 and S-C reach the same state. Negative controls: a device that evicts a protected entry, a read entry, or
    the newest one is refused by the host's check.
- F20b: the lockstep under eviction pressure (C 24 of 36 blocks, X 2). Precondition: v1 evicts. S-C equals v1 in
  every tick.
- F14c gains the kinds touch and prot.

RECOMMENDED POINTS (arithmetic above). The FG and SG5 step scripts change on their branches, not here:
- Primary point: C 96 / X 32 / R_ver 16.
  - It gives about 25 local slots per stream at the first verify tick, about 400 per layer at B 8, against a demand
    of about 170.
  - The cost: 132 slots per (request, head, layer) instead of 116. At 32,896 B per slot (the sizes above), B 64,
    H 2, 32 layers: 17.79 GB instead of 15.63 GB (+2.16 GB); at B 8: 2.22 GB instead of 1.95 GB.
  - The per-stream numbers do not depend on B, so the same C holds at B 64.
- FG3 pressure points: 72:384:256 and 64:512:384.
  - With C below the drafts' union, the verifier needs about 10 to 16 extras per stream at the first verify tick.
    R_ver >= 256 covers 16 streams.
  - The cost at B 8 (460 and 580 slots): 7.75 GB and 9.77 GB of store.
- SG5's sweep keeps C in {64, 80, 96}. Points with C <= 80 may still refuse at the first verify tick; that is a
  configuration result.

## 12i. SG5 p2 (holder 2190876): the gilfree feeder thread and torch's thread-local modes (2026-10-21)

The geometry fix of 12h worked in SG5 p2 (`/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p2_2190876`): there was no "multiples
of R". But every gilfree arm failed: "Inplace update to inference tensor outside InferenceMode is not allowed".
- Where: the packer's in-place writes (the landing ring copy_, the store's index_copy_), in the adapter's background
  feeder thread (`_feeder` -> `_release` -> NativePipe.release), re-raised by `poll`.
- Why: torch.inference_mode is THREAD-LOCAL. The tick makes the host rows and the store under inference mode on the
  main thread. A new thread starts with inference mode off, so it may not update those tensors in place.
- FIX (`r2_transport_gilfree.py`):
  - At construction the adapter records the modes the feeder needs. Inference mode is on when the constructing
    thread is in inference mode, or when any store, host or ring tensor is an inference tensor. Grad mode is the
    constructing thread's (off under inference mode). On the GPU it records the store's device.
  - The feeder enters them: `torch.cuda.set_device(...)` once at its start; `torch.inference_mode(...)` and
    `torch.set_grad_enabled(...)` around each release (since the fg2 follow-up below; first around its whole loop).
  - The current CUDA stream needs nothing. Every CUDA operation of a release names its stream: the extension's copy
    and scatter streams, the publication under `torch.cuda.stream(scatter)`, and `scatter.wait_event`.
  - The OpenMP team's threads copy raw bytes only (pack_range).
- The other transports and device operations start no thread: gather (a side stream of the tick's thread), s0 (it
  packs on the tick's thread at end_tick) and CudaDeviceOps (the tick's thread). grep finds no other Thread in
  nosi/nosi, and none in the pinned cpupack_core.
- CPU test GT6 (two cases): the host rows and the store are made under inference mode; the adapter is built either
  inside inference mode or outside it from those tensors. The releases run in the background feeder: it ran with
  inference on and grad off, and a two-request plan lands the host truth.
- GT6 negative control: thread_modes=False (the feeder keeps a new thread's defaults) gives the run's exact error,
  re-raised at force().
- FG fg2 (2026-10-21) failed the same way in every gilfree arm (FG1 superset / exact / negative, FG2 gilfree, both
  FG3 points: all transport "gilfree", 1 tick each); gather and s0 (no thread) passed. The feeder has a SECOND
  in-place write: the board publication (`_publish`, FG1's exact wait). 7ab1ca2 covered it only when the adapter or
  its tensors were in inference mode. Now attach_board also turns the feeder's inference mode on for an
  inference-mode board, and the feeder enters its modes around EACH release (it reads them per release).
- GT6b: the host rows, the store and the adapter are normal tensors; only the board is an inference tensor. The
  background feeder lands the host truth and publishes every transfer. Negative control: the board's mode ignored
  (as in 7ab1ca2): the rows land, then the publication raises the run's error; its traceback names _feeder,
  _release and _publish.

## 12j. SG5 p3 preflight (holder 2190876): Policy E under an ASYNCHRONOUS transport (2026-10-21)

THE RUN (`/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p3_2190876`, gates SG5_preflight_*; B 64, C 96 / X 32 / R_ver 16,
4 committed tokens per request, the same tokens in every arm):
- S-C on gilfree: CapacityRefusal at layers 16 / 22 / 23 (19, 4, 5 blocks without a slot).
- S-C on gather, S-B and v1 on both transports: passed (9 ticks, 5 decision ticks).
- The record has no per-stream state, so the cause below is the mechanism that explains all six results; the next
  refusal carries the evidence (the capacity report, below).

THE MECHANISM (file:line at this commit):
- A draft prefetch holds its slot while it is in flight: INFLIGHT counts in K5's usage (`ownership.py:135`), and an
  in-flight transfer is never freed before it lands (paper 3.4; note 5, the `free_on_cancel` control).
- Policy E (12h) took only LRU entries (`r2_residency.py` `_verifier_candidates`).
- When a cohort is decided, its references are dropped at the next poll (`r2_residency.py` `poll` / `_drop`).
  - With a (near) SYNCHRONOUS transport (SG5's gather, `r2_tick_sg5.py:185`: one Triton gather per issue on a GPU
    side stream, issued at end_tick, so it has usually completed by the next poll), the prefetch has landed by that
    poll: it is an LRU entry (admit with draft demand), now unprotected: a Policy E candidate.
  - With an ASYNCHRONOUS transport (gilfree: the feeder packs the 32 layers' releases one by one on the CPU and lags
    behind at B 64), the same prefetch is still in flight: it holds its slot and it is NO candidate. When every LRU
    entry is protected or read by the layer, the verifier refuses although the stream holds slots that no live
    cohort needs.
- S-B passed on gilfree at the same point: its candidates spare only the verify rows' blocks (12h), so it had more of
  them. v1 passed: its schedule polls at every resolve and issues at once, so its prefetches land earlier.
- CPU reproduction: F21 with policy_e_inflight=False (12h's rule). The asynchronous round (a force_only transport)
  refuses in v1 and S-C, with the report "LRU 0, unprotected 0; in flight 2, unprotected draft prefetches 2"; the
  synchronous round (land_all, the gather stand-in) passes.

THE POLICY (Policy E, second candidate kind):
- When K5 grants the verifier's urgent miss no slot, the candidates are, in this order:
  1. the stream's unprotected LRU entries that no row of the layer selects, least recent first (12h);
  2. the stream's in-flight DRAFT prefetches of the current owner that no live cohort protects (their requesters were
     decided or cancelled) and that no row of the layer selects, oldest issue first.
- A chosen in-flight prefetch is SUPERSEDED. Its write completes BEFORE the slot is reused; then the service forgets
  it (it is never admitted), the slot goes FREE and the urgent block takes it.
  - GPU, S-C: the layer's wait covers it. The superset wait already waits for every transfer in flight at the layer;
    the exact wait adds the prefetch's (slot, generation) to its board list (`wait_list`). The liveness rule (12g)
    already host-waits the SUBMISSION of every pack in flight at the layer.
  - v1 / S-B (host): the transport's force. CPU stand-ins: the force in CpuDeviceSim.wait, the board spin in
    ExactWaitSim.
- Paper rule: 3.4, "Storage becomes reusable only after the relevant owners and readers have finished". The
  prefetch has no live requester and no reader; its only owner is the transfer, and the reuse waits for it. It is
  Policy E's own rule applied to what a synchronous transport would have made an unprotected LRU entry by the next
  poll.
- COST:
  - the superseded prefetch's bytes are wasted (k48 counts it as never_arrived); a synchronous transport would have
    landed it and Policy E would have evicted it, so no reuse is lost against the synchronous case;
  - the urgent gather waits for that transfer on the device (no new host wait: the submission wait exists);
  - v1 / S-B host-wait it (force).
- No MUST is weakened. Every verify read still reads its full selection; every slot write follows the previous
  writer and reader; protected or selected blocks are never candidates.
- No candidate of either kind: CapacityRefusal, as before (a configuration result).

THE EVIDENCE FOR THE NEXT REFUSAL: v1's and S-C's CapacityRefusal messages now carry a capacity report per refused
stream (S-C: the first 3 per layer, from the host state at the tick's start; nothing is applied):
- usage of C; LRU and unprotected LRU entries; in flight and unprotected draft prefetches; PO; VRES; RELEASING;
- the layer's free extras of X, and R_ver.

S-C device (`device_verify`):
- The mirror carries the protection flag of in-flight slots too (`sync_mirror`; `_prot_changed` marks the slot in
  flight).
- A take stamps the slot (STAMP_KINDS): the issue order.
- LayerMirror gains fly, fly_stamp, infl_unprot, infl_rank and slot_gen. F14's full-scan reference compares
  infl_unprot and infl_rank (the in-flight order by issue sequence).
- The candidate key is the LRU stamp, or HALF_STAMP + the take stamp. The decision gains evict_infl and ev_gen.
- The layer mirror for the draft rows forgets the superseded block's flight.
- The host checks every in-flight eviction against v1's list (`_host_candidates`), then `_evict_inflight`.

TESTS:
- F21 (scenarios plain and protected_oldest):
  - the reproduction above;
  - with the fix: v1, S-C probe and S-C exact supersede the right prefetch; the urgent block holds the host truth;
    the superseded transfer never lands later; the kept prefetch stays in flight;
  - v1 and S-C reach the same state.
- F21c negative controls:
  - a device that supersedes a protected prefetch is refused by the host's check;
  - a wait that omits the superseded write (no_inflight_wait) is refused by the device's T3 check; with that check
    off, the slot's content is wrong.
- F21b: the lockstep with a force_only transport (probe and exact). Precondition: v1 supersedes prefetches.

AFTER THE FIRST CPU RUN (2051f9b: 5 failed, 138 passed, 2 skipped;
`/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/fetch_2051f9b_fail.txt`):
- F14b (the owner control's precondition): at a request's end, end_owner clears the protection map while the
  request's transfers are still in Stream.pending, so `_prot_changed` marked their slots. The owner change was no
  longer the only change of its slot, and a delta without the owner changes would no longer have been caught.
  - Fix: ResidencySC.end_request sets `_ending`; `_prot_changed` then marks resident slots only. A transfer of an
    ended request stops being the owner's, and the owner change marks its slot.
  - The test instrumentation (`_trace_journal`) follows the same rule.
- F21: the test's expectation was wrong. Stream.pending also holds the urgent block (landed in place, admitted at the
  next poll): {kept: kept, 5: gone}. The service behaved as designed.
- F21b: the precondition was not met (17 LRU evictions, 0 in flight). In `_sched_pressure` the verify rows read
  prefetched blocks, the waits landed them, and the LRU always had a candidate. The new schedule `_sched_split`
  puts the drafts in blocks 0 .. 19 and the verify reads in 20 .. 29 (C 16, 30 ticks). No prefetch lands, so only
  in-flight candidates can serve the verifier.

## 12k. The arrival rule: a fixed lag for comparable runs (Q-C3, branch feature/nosi-r2-m2-sparse, 2026-10-21)

THE PROBLEM:
- Draft rows read only READY blocks (paper 3.1). READY depends on which transfers the tick's poll admitted, and
  with free timing that depends on when the transport landed them.
- So two runs of the same documents differ: in the draft reads, the LRU order and the evictions. The exact wait keeps
  every verify read correct, but not the drafts' choices.
- "Landed at the first tick boundary after its event completes" still depends on timing, so it does not fix this.

THE RULE: `Residency(arrival_lag=D)` (None = free timing, the default and unchanged; D >= 1):
- A transport transfer issued in tick t (`tr.issue_tick`, taken from `tick_no`, which end_tick advances) is admitted
  exactly at the poll of tick t + D.
- At that poll the service FORCES a transfer that has not landed (`tp.force`). It counts the force in
  `arrival_forced` and times it in `arrival_forced_ns`.
- A transfer that landed earlier is held back until t + D; each held poll is counted in `arrival_held`.
- The report gives `report()["arrival"]` = {lag, forced, forced_ns, held}.
- Unchanged: an urgent block (landed in place) is admitted at the next poll; v1's own schedule still admits a waited
  block at once (a force at a fixed point).
- Then admissions, READY sets, draft reads, the LRU order, the evictions and Policy E's in-flight status are functions
  of the schedule alone.
- Reader events stay alike between runs:
  - a transport lands a transfer only after its slot's reader completed;
  - a force host-waits that reader.
- Cost: a host wait whenever the transport is late by more than D ticks. It is reported, not hidden. Suggested
  values: D = 2 for gilfree; D = 1 makes the gather transport deterministic too.

TEST F22 (v1 and S-C):
- The setup: two worlds on F6's schedule, with transports seeded differently ("delay", max_delay 4, bursts).
- Precondition 1: their pending lists differ at some tick.
- Precondition 2: both branches of the rule ran (a force at the admission tick, a landed transfer held back).
- Claim: at D = 2 the ownership states are equal after every tick.
- Negative control: free timing with the same seeds differs.
- Also: arrival_lag 0 is refused.

Q-C2, the submission wait (E's C++ feeder worker, feature/nosi-r2-gilfree-worker cdce147):
- `GilfreeTransport(..., worker="cpp")` sets `submits_without_gil = True`. With such a transport,
  `CudaDeviceOps._submitted` returns at once. The worker never needs the GIL, so the main thread cannot block it,
  and the exact spin with its bounded `err` stays the guard.
- Superset mode keeps its host wait (tr.ev exists only after the release).
- E's mode-order rule: attach_board must come before the worker's first issue. ResidencySC attaches it in its
  constructor; a late inference-mode board is refused by E's worker.
- CPU test F19b: an unsubmitted pack does not block `_submitted` with the attribute. Negative control: without the
  attribute (or with it False) the host wait remains.

## 12l. (B1) the sparse decision record on the vendored ownership objects (`nosi/nosi/r2_residency_sparse.py`)

THE BOUND. S-C's end_tick reads every decision field of every (row, head, selected block) entry and applies every
entry in Python. At B 64, H 2, k 64 that is about 0.8 M verify and 0.26 M draft entries per tick. SG5 p3: S-B's
draft-only ticks spent 7.06 s of backend host time per tick, against 27 ms of GEMM.

THE HOST CONSUMER (the capture seam's one call; a stable signature):
- `ResidencySC.apply_tick(steps)`. steps: the tick's layers in resolve order, each
  `dict(l, vrows, vidx, vdec, drows, ddec[, rec, caps])` with the decision tensors (fixed buffers in a captured tick).
- `end_tick` = `apply_tick(steps)`, then the deferred draft issue (`Residency.end_tick`) and the memory sample.
- S-C's `apply_tick` is the DENSE record. `ResidencySparse.apply_tick` replaces it with the sparse one. B2's
  array-backed layer implements the same function on the same records.
- Hooks for the record (device ops only): `_layer_record(l, step, tk)` (called by resolve) and `_tick_extra(rows)`
  (more integers for the one row copy per tick).

THE RECORD (per layer, int64, device; fixed shapes; built by `sparse_layer_record`):
- `vrec` (9, cap_v), its fields VREC = (o, blk, flags, slot, infl_slot, infl_gen, rslot, take_slot, evict_blk), and
  `vcnt` (1,).
  - The entries: a new pin, an in-flight merge (infl / dup), an urgent take (a grant or a Policy E eviction of
    either kind), a hole.
  - o = the entry index (row j = o // (H k), head h = (o // k) % H).
- `vlook` (Rv, 4): per verify row, the ready FIRST looks of its (key, layer) call, and their 4.6 classes
  (LRU, PO, VRES).
- `vread` (B, H, ceil((C + X) / 32)): the slots a verify row reads, as packed bits.
- `drec` (5, cap_d), its fields DREC = (o, blk, flags, slot, take_slot), and `dcnt` (1,).
  - The entries: every ready entry, a new pin, a merge, a take, a refusal, a hole.
- `sel` (B, H, ceil(nblk / 32)): the blocks that any row of the layer selects, as packed bits (for Policy E's host
  check).
- The flag bits: PIN 1, READY 2, INFL 4, DUP 8, GRANT 16, EVICT 32, EVICT_INFL 64, FIRST 128, HOLE 256, REFUSED 512.
- Compaction is a prefix sum (stable). The entry order (row, head, ascending block) is K3X's, so v1's apply order
  survives.
- The capacity: `record_cap` (cap_v, cap_d). None = every entry (no overflow). A count over the capacity voids the
  tick (CapacityRefusal, before any change).

WHAT THE HOST DOES PER RECORD (in v1's order; with an INVARIANT CHECK per record):
- PIN: `st.protect`.
  - New pins only: the device's PIN TABLE (L, B, H, nblk + 1, 2 kinds) holds (key id << 32 | epoch).
  - The host bumps the epoch of (layer, batch slot) when it drops references there (`_drop`, `end_request`) and on
    a void tick.
  - So a redundant pin record can happen (an idempotent protect), but a missing one cannot.
- READY (draft): the check "the host's resident slot is this one", the read, then the recency.
  - The touches use OrderedDict.move_to_end, with S-C's journal effects (count, stamp, dirty mark) in bulk.
  - Acquisitions and K1's evict_to_c run the vendored code.
- INFL / DUP: the owner's transfer in flight (slot and generation checked); the demand merge.
- GRANT / EVICT: v1's `_take` with its checks (K5, first_free, the evicted slot FREE). A Policy E eviction must be
  the host's next candidate (selections from `sel`).
- REFUSED: K5 must refuse it; the count must equal the device's.
- HOLE: raises.
- Ready VERIFY entries are not records. Their reads come from `vread`, their looks from `vlook`, and they get no
  host check (R5a's lockstep proves sparse == dense == v1 on the CPU).

DIFFERENCES FROM THE DENSE RECORD, by design:
- Stream.log does not receive the draft touch events (their effects are applied directly).
- The lifetimes' verify read counts (4.8 rv) count a slot once per layer. 4.8 uses them only as > 0.

DEVIATION FROM THE GO MESSAGE: the draft recency is not "bulk stamps plus device victims".
- With the vendored OrderedDict LRU, the host must reorder every touched entry anyway.
- So the record carries the touched entries in O1 order, and the host replays v1's recency at C speed. K1's victims
  come out of the vendored evict_to_c and are exact by construction.
- Bulk stamps with device victims belong to B2 (an array-backed LRU).

Q-C4, BULK TAKES:
- The device already chose take_slot. The part that bulks is on the device: the bias rows of the granted draft
  takes, one batched write per layer (`draft_bias_fill`), so the host take no longer copies them (`_take(bias=False)`).
- The host part stays per record: vendored `st.take` (the generation, the journal), the Transfer, the in-flight
  maps, metrics.issued. That is about 2-4 us per take; at p3's about 9.3 k records per tick, about 20-40 ms.
- A fully bulk registration (one vectorized call per layer) needs the array-backed slots of B2.
- The record format above is what B2 consumes: GRANT records in order give (row, head, block, take_slot) per take.

EXPECTED HOST TIME PER TICK AT B 64 (B1), as estimates; the REPORT microbench measures the CPU part:
- verify ticks:
  - one read and tolist of the records: about 10-30 ms;
  - new pins (each cohort pins its selection once, about 0.26 M per verify tick at about 0.4 us): about 100 ms;
  - the read bitmap (about 0.26 M slots at about 0.15 us): about 40 ms;
  - takes and merges: about 30 ms;
- draft ticks: ready records and touches (about 0.26 M at about 0.3-0.5 us), about 80-130 ms, plus pins at a
  cohort's first draft tick.
- So B1 should cut p3's 7 s to about 0.2-0.4 s per tick. B2 (array-backed) should bring it to tens of ms.
- REFUTED by the REPORT microbench (sparse 7392ac6, CPU, B 64, H 2, 2 layers): apply_tick per tick, dense verify
  252.3 ms / draft 112.8 ms; sparse verify 266.8 ms / draft 109.7 ms. Times 16 for 32 layers: dense 4037 / 1804 ms,
  sparse 4269 / 1755 ms. 114,904 sparse records applied.
  - So on the vendored objects, B1 gives NO host speedup. The cost is the per-record vendored apply (take, protect,
    move_to_end, the journal, the reads), not the reading of the dense record.
  - My estimate assumed sub-microsecond records; the vendored dict-of-set and OrderedDict objects cost several
    microseconds each. B1 stays as the dense == sparse reference and as the record producer; B2 is the speed path.

WHAT B2 CAN KEEP FROM B1's DEVICE PART, AS IS:
- `_compact` (the stable prefix-sum compaction into fixed buffers, with a count) and `_pack_bits` / `unpack_bits`.
  They are independent of the ownership representation.
- `draft_bias_fill` (the bias rows of the granted draft takes in one write per layer). It needs only the decision
  and the store.
- `_first_in_group` (the stable first-occurrence mask), and the look counts (vlook) if B2 keeps K3X's 4.3 / 4.6
  metrics.
- The record itself (note 12l's format), as the lockstep interface between B1 and B2.
- NOT the pin table. It exists only to tell the vendored host which (block, key, kind) pins are new. A device-
  authoritative B2 table keeps protection itself (per-row selection bitmaps, B's design).

AFTER THE CPU RUN of 7392ac6 (3 failed, 163 passed, 2 skipped;
`/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/sparse_7392ac6.txt`), both preconditions were mine:
- R5a f21b / f21b_exact: "touches > 0" cannot hold there. The split schedule never lands a prefetch, so no draft row
  reads a READY block. The touch precondition now applies to f6 / f12b / f20b only.
- R5b d_merge: the negative control's choice ran BEFORE the tick's applies. A draft merge into this layer's urgent
  verify take (demand {verify}) has no host transfer yet at that point, so it was skipped as "no transfer". It now
  counts as state-changing (the take creates the transfer at apply). A v_merge without a transfer (a dup of the
  layer's own urgent take) is still skipped: its demand holds verify already.

TESTS (`test_r2_residency.py`, R5):
- R5a: sparse == dense on F6 (probe and exact), F12b, F20b and F21b (probe and exact). Every tick it compares the
  pages, the verify content, both mirrors against the full scan, the ownership state, the counters and each slot's
  first / last reader. One device read per tick.
- R5b: one planted dropped record per kind (v_pin, v_merge, v_take, v_read, d_pin, d_ready, d_merge, d_take,
  d_refused). Each is caught at its own tick; the precondition is that a state-changing record of that kind
  occurred.
- R5c: a record over its capacity voids exactly the tick that the uncapped run predicts. Negative control: a
  truncated record (no_cap_check) differs at that tick.
- R5d: two draft take records of one stream swapped are refused by v1's take check.
- R5e: the lint: the per-layer device path has no host read or copy; the consumer has the one read.
- REPORT R5 microbench: apply_tick dense vs sparse at the B 64 geometry (2 layers, extrapolated to 32).

## 12m. Shared lockstep fixtures: the transitions the schedules never reach (branch feature/nosi-r2-m2-sparse)

WHY. The lockstep schedules (F6, F12b, F20b, F21b, F22) only call `decided()` and complete readers two ticks late,
in order. So they never reach end_request, cancel, two draft rows of one stream, void ticks, T4, the trash slot or
non-monotone readers (the B2 review, `docs/R2_OWN2_B2_REVIEW_C.md` section 2).

WHAT. `benchmarks/Efficiency/tests_cpu/r2_lockstep_fixtures.py`, a module that needs only torch:
- A FIXTURE is a short script of operations (begin, end, tick, poll, land, done, decided, cancel, cancel_epoch,
  history, watch, flag, raise), with its capacities, its worlds, a counted precondition and its plants.
- `run(fixture, worlds, snapshot, Row)` applies each operation to every world in lockstep. After each operation it
  compares the raised exception (by class name), the returned value (a tick returns its page tables) and the
  snapshot. The first difference ends the run and names its operation and its field ("raise", "value", or the first
  differing snapshot field).
- A PLANT is a defect in ONE world, with the operation and the field where the lockstep must see it first.
- B2's lockstep imports the same module: it puts a vendored world first (the preconditions read it) and its own
  world after it. The world may give its own `land()`.

THE FIXTURES (all on v1 under the S-C schedule, S-C dense and S-C sparse, force_only transports; void_flag on S-C
dense and sparse only, v1 has no device flags):

| fixture | review item | transition reached (precondition, counted) | plants (operation, field) |
|---|---|---|---|
| end_request | (a) F13 / F14b + a land | the dedup merge (no take); release:end; u1's verifier takes its own slot (new, no in-flight look); drop_unowned at the final land | dup_take_device (4, raise), dup_take_host (4, state), owner_not_mirrored (5, mirror), owner_blind (7, value), no_uid_check (8, raise) |
| cancel | (b) F3 / F11 | a cancelled key refused (op 4); arrivals after the cancel; the late-protection guard (op 7); cancel_epoch returns [K2] | free_on_cancel (2, state), cancel_refusal_off (4, raise), late_guard_off (7, raise) |
| o1_two_rows | (c) I8's O1 rounds, C 2 | two draft rows of one stream: block 2 goes PO (row 1's evict_to_c) and is acquired back (row 2) | batch_recency (3, journal) |
| void_overflow | (d) F12f | CapacityRefusal (terminal) | no_overflow_flag (1, raise) |
| void_flag | (d) F16 | a set flag voids that tick (state and counters unchanged); the next tick applies | flag_ignored (4, raise), void_keeps_pins (6, state) |
| t4 | (d) T4 | the take of block 5 carries its write-back; block 6 is refused (terminal) | no_t4_refusal (3, raise) |
| trash | (d) F15 | takes, arrivals, an urgent gather: the trash sentinel stays; a tail naming the trash slot is refused (terminal) | trash_written (1, trash) |
| nonmono | (d) F12e | a pending earlier reader while a later one is done; the urgent take avoids its slot; the T3 forward of e1; drop_unowned resets the reader | no_reader_wait (6, value), lost_forward (9, readers), drop_keeps_reader (12, readers) |
| vres_drop | the poll order | admit_vres then release:unprotect in one poll (no drop_vres_unprotected) | poll_order_swapped (2, readers) |
| key_alias | key ids | two live keys pin one block (prot = {K1, K2}) | key_alias (2, state) |

THE REVIEW'S 8 MISSING KINDS, on the vendored path (R6c checks the coverage):
1. A take of a block that is in flight: dup_take_device (the device; the host's take check refuses it),
   dup_take_host (the host; a second transfer).
2. T4: no_t4_refusal.
3. A cancelled key / a row after its release was queued: late_guard_off; cancel_refusal_off. The vendored code
   refuses a cancelled key TWICE: `cancel()` also queues the key's release at every layer, so the late-protection
   guard refuses it when the cancelled check is off. Only both off accept it.
4. Key aliasing: B1's form (`key_alias`): one key id per request in B1's pin table. B1 never reuses an id, so the
   plant aliases two LIVE keys. B2's key-slot reuse is the same class: a new key that inherits a stale pin bit.
5. The poll order swapped: poll_order_swapped.
6. A drop that keeps the reader: drop_keeps_reader. CORRECTION of the review (section 3, kind 6): in the vendored
   model the dropped slot does NOT free one poll late. The transfer carried the slot's previous reader (T3), and the
   transport waits on it before it lands, so that reader is done at the drop. The defect shows only in the reader
   projection (the slot's last reader is kept instead of reset).
7. The T3 forward lost: lost_forward.
8. One batch eviction for two draft rows: batch_recency.

CORRECTION OF THE REVIEW, I8 (section 1, row I8). The review said that one eviction after all draft rows "gives a
different LRU". That is WRONG for the end state. Proof sketch:
- Phase A protects every block that a draft row selects (K2), so no block that a later row reads is released by an
  earlier row's eviction: it goes PO and the later row acquires it back.
- Let order1 be the recency order after row 1's moves and order2 after row 2's moves. A block x that row 2 does not
  read and that is in the last C of order2 has at most as many blocks after it in order1 as in order2. So it is in
  the last C of order1: row 1's eviction did not remove it.
- So per-row and batch evictions keep the same LRU (the last C of order2, in that order), and the same PO / released
  sets (a protected victim goes PO, an unprotected one is released, in both).
- Only the TRANSITIONS differ: o1_two_rows has acquire 2 + evict_po 2 per row; the batch has acquire 1 + touch 1 +
  evict_po 1. The stamps (same order) and the mirror marks (the same slots) are equal too.
- So the state, the mirror and the readers cannot see the batch defect. The journal (S-C's journal_counts) can, and
  the planted batch_recency is caught there. For B2: a batch eviction is state-equivalent; if B2 reports the journal
  counts, it must count the per-row transitions.
- The review's own example (C 2, X 0: draft [0, 1, 2]) cannot reach the case: with X 0 the third block is refused.
  The fixture uses X 2, R_ver 0.

TESTS (`test_r2_residency.py`, R6):
- R6a: per fixture, the worlds agree after every operation; every operation runs; the precondition holds.
- R6b: per plant, the first difference is at the predicted operation and field.
- R6c: the plants cover the 8 review kinds; every plant has its defective world; every fixture's first world is a
  vendored one.
- The snapshot (`_fx_snapshot`): `_state`, `_counters`, the readers by name, the transfers in flight with their
  forwarded reader and write-back flag, the transitions per kind (`_journal_kinds`: S-C's journal_counts, or v1's
  Stream.log counted the same way), the RELEASING slots with a pending reader, S-C's mirror against the full scan,
  and the trash sentinel.

NOT RUN (static work only, no interpreter). The predicted operations and fields come from reading the code; the CPU
run decides.

## 12n. T4 on the GPU path: the write-back wait list gets its producer (the TC-R review, note 5.1)

THE GAP. S-C's `_hist_unwaited` ("host write-backs the main stream did not wait on yet") had consumers but no
producer:
- Consumers: `CudaDeviceOps.gather` (a main-stream `wait_event` per entry before its copies), and the captured
  tick's host part (the same waits before the replay, outside the graph).
- Producer: none. `history_written` registered the event only for the takes (`hist_ev`, which the transports wait).
- So the GPU urgent gather (zero-copy reads of the pinned host rows) was ordered after a write-back ONLY when both
  ran on one stream. That is true today (`commit_block` uses the current stream), but nothing guarded it.

THE FIX (`r2_residency_sc.py`):
- `ResidencySC.history_written` calls v1's and appends the event to `_hist_unwaited`.
- The CPU stand-in's gather (`CpuDeviceSim.gather`) host-waits every pending entry first and empties the list. This
  is the stand-in of the main stream's wait, and it keeps the list bounded on the CPU.
- On the GPU the list is now non-empty after every write-back. The captured tick's host part must keep emptying it
  before the replay: a capture cannot wait on an event recorded outside it. It does (r2_residency_capture.py
  host_part).

TEST F24:
- Positive: on the GPU path (CudaDeviceOps with CPU copies and a logging stand-in of the main stream), a write-back
  on another stream (a pending event) is waited ONCE, before the first copy of the urgent gather. The list is empty
  after it, and the take still carries the event.
- The CPU stand-in host-waits a pending write-back at its gather.
- Negative control: without the producer, the gather copies block 5 with no wait.

## 12o. SG5 p4a's K5 refusal: the policy is right, the report misled (2026-10-21)

THE RECORD (`/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p4_2190876_p4a.step/stderr.txt:7083`; gilfree, B 64, C 96, X 32,
R_ver 16):
- Layer 16 had 16 first misses with no slot. No other layer overflowed.
- Refused streams: (b 45 h 1), (b 47 h 0), (b 51 h 1), with usage 86 / 86 / 89 of C 96.
- In each: LRU 77, all protected; in flight 9 / 9 / 12, all protected; PO, VRES and RELEASING 0.
- The layer had all 32 extras free at the tick start.

THE K5 RULE:
- Stream.usage = every non-FREE slot (INFLIGHT + LRU + PO + VRES + RELEASING), ownership.py:135-136.
- LayerPool.borrowed = sum over the layer's streams of max(0, usage - C), :409-410. free_extras = X - borrowed,
  :412-413.
- can_take, :419-427:
  - usage < C: yes (a base slot);
  - else, for the verifier: free_extras > 0 (the R_ver reserve included);
  - for a draft: free_extras > R_ver.
- device_verify (r2_residency_sc.py, K5 block) implements the same rule in K3X order (row, head, block): a stream's
  first C - usage first misses take its FREE base slots, and every later one needs a shared extra. Extras go to the
  first X - borrowed such misses of the LAYER. Policy E serves the rest if it can.

THE ANSWERS:
1. Usage counts every non-FREE slot (86 = 77 LRU + 9 in flight). The free base slots ARE granted: 10 / 10 / 7 first
   misses per stream.
2. The verifier DOES take from the R_ver reserve: R_ver limits only drafts.
3. The 32 extras are ONE pool per LAYER, shared by its B x H = 128 streams (0.25 per stream on average). The report
   says "free 32" because it shows the host state at the tick START: a void tick applies nothing.
   - The decision gave all 32 to earlier first misses of layer 16 in K3X order. A first miss is refused by K5 only
     after the budget is spent, so all 32 were granted.
   - At least 16 more first misses needed one, with no Policy E candidate left. The refused streams have high batch
     slots, late in K3X order.
   - So layer 16's verify demand beyond the base slots was at least 48 first misses (32 granted + 16 with no slot,
     plus any Policy E evictions), and the allowance was 32. The "16" is the LAYER's total, not per stream.

WHY THE DEMAND IS THIS LARGE:
- NOSA-8B selects topk 64 blocks per row and head (nosa_llama.py:672-681); about 62 are historical.
- Every draft row protects its whole selection. With LAMBDA 7 and K 3, a stream holds the union of about 7-10 draft
  positions' selections, here 77-89 blocks. Q3's measured union over 4 positions was about 71.
- A verify cohort's own selection then misses about 10-30 blocks per stream, while 10 base slots and 0.25 extras per
  stream are free.

S-B AND v1 PASSED THE SAME PREFLIGHT (no traceback after the S-C one). That is NOT evidence of an S-C defect:
- With free timing their trajectories differ from S-C's. v1 polls arrivals inside resolve. S-C's draft rows see the
  verifier's urgent blocks of the same layer as in flight, not ready.
- The draft READY sets, so the draft tokens, the selections and the protected unions, all differ.
- On the SAME state v1 refuses at the same point (F25).

THE FIX (code):
- The refusal now reports the DECISION. `_k5_report`, shared by the dense and the sparse consumer, gives:
  - per layer: the streams with first misses, how many needed an extra, the free extras at the tick start, how many
    were granted, and how many found no slot;
  - per refused stream: first misses = base slots + shared extras + Policy E + no slot;
  - then v1's capacity report.
- `device_verify` gets two test-only negative controls: `verify_no_local` and `verify_reserve`.

TEST F25 (dense and sparse):
- p4a's state: usage 86 of C 96 in all 4 streams, LRU 77 and 9 in flight (all protected), the 32 extras free.
- First misses 38 / 20 / 15 / 15: S-C refuses with exactly this decision:
  - the base slots 10 each;
  - the shared extras 28, then 4;
  - no slot 6 + 5 + 5 = 16 over 3 streams;
  - nothing applied.
- v1 refuses the same state at block 100, the 15th first miss of stream (0, 1).
- At exactly the allowance (38 / 14 / 10 / 10), S-C and v1 pass with equal states and all 32 extras borrowed.
- Negative controls: a device K5 that keeps the reserve from the verifier, or grants no base slot, refuses the
  allowance case.

WHAT IT MEANS FOR MEASURING S-C:
- At C 96 / X 32 the first verify ticks at B 64 exceed the allowance in at least one layer.
- That tick needed X >= 48 at layer 16. The run's maximum is unknown; the new report gives it.
- X is per stream in the store (C + X slots per (layer, request, head)). At D 128, bf16 K and V, one slot per stream
  over L 32 x B 64 x H 2 is about 128 MiB, so X 64 costs about 4 GiB more than X 32.
- The alternative is fewer protected blocks per stream, for example a smaller LAMBDA (fewer draft positions ahead).
- DECIDED by the lead (2026-10-21): S-C is measured (SG5 and TC-R) at X 64, with C 96 and R_ver 16.

## 12p. The sparse record for a device consumer: rowsel and pin_records=False (B2's requests, rev 2 note 6)

ROWSEL:
- Each layer's record gets `step["rec"]["rowsel"]`, (R, H, ceil(nblk / 32)) int64 words in `_pack_bits`' format
  (32 bits per word; bit i of word w = block 32 w + i).
- It is indexed by TICK row: verify and draft rows in the tick's Row order. The decision's entry rows map to tick
  rows by the tick's verify / draft index lists (S-C's `_tick_rows`; the captured tick's `live_rows`).
- A bit is set for every valid historical block of the row's selection at that head (0 <= block < hist_bound,
  < nblk). Tails are excluded and duplicates collapse.
- It stays ON THE DEVICE. B1's host consumer does not read it, and it is not part of the one device-to-host read.

PIN_RECORDS=FALSE (`ResidencySparse(..., pin_records=False)`):
- The record carries no protection pins: no PIN flag and no pin table update.
- A verify entry that would be a record only for its new pin (a ready block) is no record. Draft records do not
  change, because every valid draft entry is a record for another reason too.
- B1's own consumer protects only from pins, so `ResidencySparse.apply_tick` refuses to run (ValueError). Only a
  consumer that keeps the protection itself (B2, from rowsel) may use it.
- `pin_only_records` (verify, draft) counts the PIN-only records that the default consumer applies. This is the
  saving.

TESTS:
- R7a: rowsel equals every tick row's valid selection, on every layer of F6's first 6 ticks; apply_tick does not
  name it. Negative control: rowsel indexed by the entry row differs at tick 2.
- R7b: the pins-off record equals the default record minus its PIN-only entries, with no PIN flag; the looks, read
  bitmap, selection and rowsel are equal. The dropped entries equal `pin_only_records` (> 0). A pins-off service
  refuses its own consumer, applies nothing and leaves the pin table untouched. Negative control: clearing only the
  PIN flag is not enough.

## 13. Open questions

- Q0 The per-layer sync of v1 drains the GPU 32 times per tick. If dHOST is too large, S-B (device-side draft rows,
  no sync in draft-only ticks) is the first optimization; S-C removes the sync of verify ticks.
- Q1 The draft's protection is known only when its selection reaches the host (v1: in the same layer, by the sync;
  S-B: later, asynchronously). In S-B a block read by a draft row could be released before the host learns of the
  read; the global reader gate (a slot becomes FREE only after the main-stream event recorded after the latest issued
  tick) covers it. To be proven by a CPU test in S-B.
- Q2 The verifier's wait on a not-yet-submitted CPU pack is a host wait. 4.9 measures how often it happens.
- Q3 C + X must hold the union of a cohort's selections plus the LRU (measured union over 4 positions about 71 blocks);
  a CapacityRefusal names the shortfall. SG5 sweeps C and X.
- Q4 Joint reads (paper 3.3: queries that use the same KV share its reads) are out of scope here.
- Q5 (DONE for S-C, 12d) The mirror refresh of S-B is a full host scan of the layer's streams and slots (Python, per
  layer, per tick): too slow for SG5 at B 64 x H 8 x (C + X) slots x 32 layers. S-C now keeps the mirror on the device
  and updates it by one batched delta per tick; the full scan stays as the reference (F14). S-B keeps the full scan
  (S-C supersedes it for runs).
- Q6 CudaDeviceOps' superset wait may stall the main stream on draft transfers that no verify row reads. The 4.3
  layer_wait_ns measures it in FG1; the exact device wait (12f) is built as a design and a CPU stand-in, and FG1 runs
  both.
