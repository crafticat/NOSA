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
  layer_wait_ns measures it in FG1; the exact device spin on published generations is the fallback.
