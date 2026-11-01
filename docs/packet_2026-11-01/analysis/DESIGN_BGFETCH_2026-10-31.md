# DESIGN BGFETCH (2026-10-31): useful background fetch and retention for the best specKV arm

Written 2026-10-31T11:44:21Z (from `date -u`). Writer: the background-fetch design agent (AREA 2). Read-only for this
file; no job ran. Style: ASD-STE100. Labels: MEASURED (a run record), DERIVED (arithmetic on MEASURED values, operands
shown), MODEL (needs an assumption, stated), CODE (read in source, file:line), CITED (another plan file, file:line).

Abbreviations. PL = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan. R = /mnt/beegfs/ojerbi/r2-runs/spec.
T7 = /mnt/beegfs/ojerbi/nosi-r2-recon7 (branch r2-recon7, the best arm's code = I4DC 2e9e497 + RECON7 cells).
ST = T7/nosi/nosi/spec_tentative.py. STF = T7/nosi/nosi/spec_tentative_fork.py. VR = T7/nosi/nosi/verify_rows.py.
E = T7/benchmarks/Efficiency. LN = E/r2_i4a_lean.py. KP = E/r2_i4d_keep.py. KC = E/r2_i4k_combo.py.
DG = E/r2_i4h_graph.py. GR = E/r2_i4g_graph.py. AR = E/r2_i3_arena.py. PP = T7/nosi/nosi/pool_prefetch.py.
SP = T7/nosi/nosi/spec_prefetch.py. PRB = E/r2_recon7_probe.py. LG = PL/LEDGER_rev7.md. R3 = PL/RECON7_R3_draft_transfer_capacity_schedule.md.
R4 = PL/RECON7_R4_prior_art.md. WSE = PL/SCORING_wse.md. SP5 = PL/SCORING_sp5.md. PF1 = PL/SCORING_pf1.md. SI4 = PL/SCORING_i4.md.
BEST = L4PTFDGUDCD (C128 B216) and L4PTFDGUDKWCD ("KW", C73 B318).

## 0. Answer in ten lines

1. Today the joint pass J loads every NEW host block URGENTLY: zero-copy Triton gathers (SM loads from pinned host
   memory) on the one compute stream, at the point of use, layer by layer. Nothing is fetched in the background.
2. The drafts KNOW which blocks they skipped: in each one-row pass the residency of layer l names every miss
   (eng._load_mask) and then skips it (ST:471-476; LN:213-220). This list is free on the device, once per layer.
3. Whether that list covers the NEXT J's NEW blocks (flagA) is NOT MEASURED on this ladder. The probe exists (PRB
   GatherTap / SkipRecord, PRB:372-495) but never ran on a GPU. The only coverage number is WS-E on the LEGACY schedule:
   at least 83.8 % of NEW was never requested in time (WSE:205-218). Its causes (a stale K-th snapshot; no pool room) are
   absent here.
4. Three earlier prefetch builds ran (poolpf pf1, spec5 L5, WS-E). All were exact. None gave a net gain: L5 -6.5 % ..
   -14.5 % (commit stall 38-50 ms); WS-E +1.8 % at C73, -4.4 % at C128 (host-bound commit, low coverage). The copy path
   itself worked: copy-engine wait about 0.1 ms per round, background interference +0.25 % .. +1.46 % (PF1).
5. X1 (MEASURED today, R/recon7_x1r3_2195630_r7s, lead's message): a copy-engine H2D from pinned staging overlaps the
   GEMMs and the fork (interference 0.8-3.4 %); a copy BESIDE the urgent zero-copy gathers shares the link and slows
   them by 12.3 %. So a prefetch must REPLACE urgent bytes, not add bytes beside them.
6. Recommended build (BF1, section 4): during D0 / D1, take each layer's skipped host blocks, pack them on the host,
   move them with the copy engine on a side stream, and COMMIT them into the KW arm's kept store (map kid / kage,
   KP:33-62) on a side stream, before the next J starts. J then finds them as ordinary keep hits (store list) or kwin
   hits (row 0's window loads, KP:381-425). No kernel changes; the bytes are the host bytes, so tokens stay bit-equal.
7. One barrier per round: J waits the side stream's last commit. No prefetch copy runs while J's urgent gathers run.
8. Expected effect (MODEL, coverage c unknown): saving about c x 115.6 ms per round at C73 B318 (round 524.7 ms) and
   c x 20.3 ms at C128 B216 (round 324.9 ms). c = 0.8: C73 about 1.21x (1,699 -> about 2,060 tok/s); C128 about 1.03x.
9. Smallest first experiment: the RECON7 probe's counted rounds (flagA, already built, minutes); then the BF1 miniature
   (B 8, check every round) and one 128-step accounting cell at C73 B318.
10. Risks: low coverage (WS-E's history); host pack rate (torch CPU gather of 256-byte rows, unmeasured); store room at
   C73 (needs X 257, +2.68 GB); late D1 layers at the J barrier; worker CUDA calls during a graph capture.

## 1. The current data flow per round (BEST; CODE)

Order of one steady round: reset -> J (K + 1 = 4 rows per request) -> commit -> D0 -> D1 (ST:940-1053). The round's ONE
host sync is the `.cpu()` of the plan values after D1 (ST:712); it syncs the compute stream only.

| step | what | where (file:line) | stream / kernel | buffer |
|---|---|---|---|---|
| selection, one-row pass | stage 1 + pooling + top-k of the one row | ST:353-358; graph A(l) of DG:11-30 | compute stream, captured | st.ti (H, B, topk) |
| selection, joint pass | batched selection of 4B rows | ST:360 -> VR:413 (_select_rows) | compute, captured (GR) | sel (H, B, R, topk) |
| residency, one-row pass (D0, D1) | diff -> pool_update -> pool_swap -> map; a DRAFT row skips EVERY miss: ids -1, map entry -1 | LN:196-224 (= ST:450-480) | EAGER between graphs A(l) and B(l) (DG:382-391) | window slots of eng (engine rows in the arena) |
| residency, J (row 0 only) | the same for row 0's selection (a verify row: misses LOADED) | STF:145-149; KW: KP:381-425 (_resid_k) | EAGER (KC header, "EAGER") | window |
| row 0's fetch (NEW, window) | flash_h2d_from_mask / _bias on lm | ST:473-474; KW: KP:411-416 (store first, host for the rest) | compute stream; zero-copy SM loads from pinned host | eng._k_gpu / _v_gpu / _kv_bias_gpu |
| rows 1-3 (NEW, store) | _fork_locate lists the host-only blocks of FULL rows, ascending, <= X - 1 per (head, request) | VR:1105-1177 | compute, captured | lst (H, B, X - 1) |
| store fill | non-keep: every listed block, every layer (scratch) STF:172-174 / KP:302-304; KW: keep_assign + miss fill only KP:465-492, KP:116-157 | compute, captured (KC) | arena store (AR:59-61) |
| attention | ONE fork call per layer over the union, pages in ASCENDING BLOCK ID | VR:1180-1209, :1095-1102 | compute, captured | arena (engine rows, store, CB0) |

Host buffers. _k_cpu / _v_cpu = (B, S + gen, H, D) bf16, PINNED (T7/nosi/nosi/cache_engine.py:195-196). One (head,
block) = 64 rows of 256 B with a row stride of H x D x 2 = 512 B: NOT contiguous per head; contiguous for both heads
(32 KB K + 32 KB V). The bias rows come from the device-resident twin.total_cis (flash_h2d_mask_bias.py:46-59).

Capacities (CODE / CITED). Window: topk = 64 slots per (layer, head, request). Victim pool P = C - 63 slots per
(layer, head, request): 65 at C128, 10 at C73. Store: ONE store of X = 1 + (K - 1) x 64 = 129 slots per (head,
request), SHARED by the 32 layers (AR:17-21, :59-61), 8.487 MB per request; slot 0 = CB1 (rewritten every layer).
Non-keep arm: the store is scratch, refilled every layer of every J. KW: the store is a cache keyed (layer, block)
(KP:33-50); its listed use is 1.03 / 2.28 of 128 slots per (layer, head, request) on average (CITED R3:59: "> 98 %
idle").

Retention between rounds. Window and pool: kept (ST:70-75, D1). Store: nothing (BEST C128); KW keeps the store map
across rounds and serves row 0's window loads from it (KP:72-78). MEASURED: avoidable refetch 23.0 % / 24.1 % of NEW
(CITED R3:58); the keep cut NEW by 15.2 % / 8.8 % (store) and 33.6 % / 23.4 % with the window part (SI4:215, :225).

Eviction. Window: the diff assigns slots to new ids; the old block moves to the pool (pool_update / pool_swap, LRU by
age). A draft skip EVICTS a window block for a block it does not load (ST:475: "the next row loads a skipped block
again"). Store (KW): protected = this layer's hits; dead (not used at its layer's last visit) oldest first; then alive,
most recent first (KP:44-50).

## 2. What the drafts know before J

Positions (CODE ST:27-45; trace R3:169-175). In steady state J(r + 1) has verify rows at positions p0, p1, p2 and a draft
row at p4. p0 was fed by D1(r - 1), p1 by J(r)'s draft row, p2 by D0(r); D1(r) fed p3 (the deferred row, replayed by
the twin, no attention). So D0(r) computed a selection AT THE SAME POSITION as J(r + 1)'s row 2, and D1(r) at the
position between rows 2 and 4. Layer 0's selection is identical (same embedding, same compressed keys); deeper layers
differ only by the drafts' skipped reads.

What is on the device after D0 / D1 layer l: eng._load_mask (H, B, topk) = every block the row selected that was in
neither the window nor the pool; skip = (lm >= 0) & (draft | tentative) (LN:213-216). For a draft row EVERY miss is
skipped. The candidates for a prefetch are skip & (lm >= 0) & (lm < qc // 64) (complete host blocks only; LN:214).
D1's residency of layer l overwrites D0's lm of layer l, so a copy is needed after each residency.

flagA (R3 D2; PRB:37-39): "a draft row skipped this block id of this (layer, KV head, request) since the previous
joint pass". MEASURED values: NONE on this ladder (the RECON7 cells never ran: PL/RECON7_RECEIPTS_RECOVERED.md lists CPU
receipts only). Indirect evidence:
- MEASURED magnitudes match: draft skips per round 18,229 (C128) / 101,864 (C73) vs NEW 16,946 / 101,764 (CITED
  R3:134-136). No per-block link.
- WS-E (legacy schedule, journal undo, L3E / L4PE; R/we_e1bm_2195573_we1bm): >= 83.8 % of NEW never requested in time,
  <= 16.2 % requested then lost; 0 % late (WSE:205-218). Causes (WSE:220-230): the K-th draft's snapshot joined the next
  epoch (SP:15-37 lag-1 design); at C73 L4PE dropped 28.2 % of offered items for lack of POOL room.
- RetroInfer working-set study (memory note, project_retroinfer_eval.md line 35): lag-1 coverage 0.65-0.69 at 131k;
  demand = stable core + 1-step flicker.
Conclusion: coverage is the first unknown. Measure it before any timing claim (section 5, E0).

## 3. Earlier prefetch / background-transfer attempts

| attempt (tree @ commit) | what it did | ran? (record) | MEASURED result | why dropped |
|---|---|---|---|---|
| poolpf, Task 1.2 (nosi-r2-poolpf @ ea05c6c; PP) | C++ OpenMP packer (gilfree v3) -> pinned staging ring -> copy engine on a copy stream -> landing -> index_copy_ -> shipped scatter into POOL slots with a soft pin; and --fetch cpupack on the critical path | yes, pf1 on SH C128 B216 (R/../poolpf/poolpf_r2p_2195574_pf1, PF1:7) | background 64 MiB/step +0.25 %, 192 MiB/step +1.46 % (drift-cancelled); packer backlog 0; 64 MiB lands in 11.2-11.6 ms; cpupack ON the critical path 1.62x slower (per-layer host round trip) (PF1:15-27) | component, not a speed lever by itself; proved the copy-engine path is cheap in the background |
| spec5, rung L5 (nosi-r2-spec5 @ 40ac348; SP) | draft misses -> lag-1 snapshot -> packer -> copy engine -> one merged commit per layer into pool slots at the verify start; soft pin | yes, sp5m / sp5b / sp5a (SP5:7-11) on L3P / L4PP, legacy schedule | exact; verify waits 0.09 ms for delivered blocks; BUT commit-point stall 38-50 ms per round, lane cap refused 48-53 % of misses, accounting 27-88 ms; net 6.5-14.5 % SLOWER (SP5:23-41) | stop rule (L5 not faster than L4) |
| WS-E (nosi-r2-wse @ daef13d) | spec5 + lazy per-layer commit (no stall), shared lanes, link budget, requested-KV pin | yes, we1m / we1b / we1a / we1bm / we1am (WSE:9-16) | stall 0.003-0.74 ms; sent 97.5 % of fresh misses; NEW only -3.2 % vs spec5; >= 83.8 % of NEW never requested in time; L3E / L3 1.018 (C73), 0.957 (C128); host issue 65.8 ms/round at C73 (+24.3 ms in the drafts); lazy commit host 37.8 ms not hidden at C128 (WSE:25-44, :164-193) | "exact, stall-free component, no speed credit"; lever = coverage and room, not delivery |
| I4D keep / keep + window (nosi-r2-i4d @ 99c231e; KP) | retention: the store kept across rounds; row 0's window loads from the kept store | yes, d3a / d4b / f3b (SI4:186-193) | NEW -15.2 % / -8.8 % (keep), -33.6 % / -23.4 % (keep + window); C73 +3.6 %, C128 -1.4 % (bookkeeping +4.4-5.5 ms) (SI4:209-229, :253) | kept at C73 (it is the KW best arm); lost at C128 |
| kappa (retroinfer-eval, 2026-09-20; memory note line 158) | side-stream Triton gather vs copy engine beside a decode step | yes (jobs 2175525..2175552) | Triton side-stream gather kappa 0.91-0.95 (no overlap); throttled 8-CTA 0.21-0.45; copy engine from contiguous pinned staging 0.006-0.05 | chose "host-packed DMA prefetch path" |
| X1 (T7 r2_recon7_streams.py, cell r7s) | copy engine beside GEMM / fork / urgent gathers | yes, R/recon7_x1r3_2195630_r7s (lead's message 2026-10-31) | C128 A 66.7, M 22.3, A\|M 68.9 ms (interference 3.4 %); C73 A 84.1, M 133.2 (3.35 GB at about 25 GB/s), A\|M 134.3 (0.8 %); F\|M 37.8 vs F 37.5; copy beside the zero-copy gathers: ordered calls +12.3 %, only 76 % absorbed | design input: replace urgent bytes; no copies beside J's gathers |
| spec_prefetch.py on the tentative ladder | - | NEVER RAN on L4PT / I-ladder arms (LG:441; HANDOFF_RECON7.md:85-86) | - | - |

## 4. Build plan BF1 (no new kernel)

### 4.1 Arm and base
New arm L4PTFDGUDKWCDB = KW best arm + BF1, at BOTH configs. Reason: the kept store (KW) is the only device structure
that (a) is invisible to the drafts, (b) is keyed by (layer, block), and (c) is already looked up by both NEW paths of J
(keep_assign for rows 1-3, kwin for row 0). At C128 the reference stays L4PTFDGUDCD (best); KW at C128 costs about 5 ms
of bookkeeping (SI4:213), so BF1 at C128 must save more than that.

### 4.2 Data path (per round r, for J(r + 1))
1. SNAPSHOT (main stream, eager, in the draft residency): override LN._resid_lean in a state subclass (DG:388 and
   DG:345 call it eagerly in replay and in capture). After super(): ids = where(skip & (lm >= 0) & (lm < qb) & draft,
   lm, -1) as int32 into a device buffer, then a non_blocking copy into a pinned host buffer, then an event. About 3 small
   ops per layer; no host read.
2. HOST WORKER (Python thread; torch CPU ops release the GIL): per (pass, layer) job: wait the event; nonzero; drop
   items already issued this round (D0 vs D1) and items that the kept store holds ALIVE (a D2H snapshot of kid / kage
   taken after J(r)); cap the count; pack K and V rows (64 rows of 256 B per item) with torch.index_select into pinned
   staging; then, on a COPY stream, non_blocking copies into a device landing entry (ring) + the item table.
3. COMMIT (BF side stream, eager torch ops, during D0 / D1): wait the landing event; per item: key = l x 2^32 + block;
   drop if the map holds the key; choose a store slot that is EMPTY or DEAD (not used in J(r): kage < g x L), oldest
   first, by (head, request) rank; index_copy_ the 64 K / V rows into the slot; bias rows from twin.total_cis on the
   device; write kid = key, kage = g x L + l; tag the slot as prefetched. Items without room write to slot 0 (CB1:
   rewritten by J's chain image before any read) and count as no_room.
4. BARRIER (J(r + 1) start): the main thread CLOSES the round (jobs not yet issued are abandoned and counted as late),
   then the compute stream waits the side stream's last commit event. J runs unchanged: keep_assign finds hits, kwin
   copies row 0's loads from the store (device to device), the urgent gathers fetch only the rest.
5. ACCOUNTING (J(r + 1) end, eager): used = tag & (kid == the committed key) & (kage >= (g + 1) x L) (keep_assign
   refreshed it); unused = tag & ~used; clear the tags. kwin uses do not touch kage (KP:76-77): count them through the
   arm's own kc row 6 (window loads served from the store) against the KW reference in the same job.

### 4.3 Correctness (what must stay bit-equal and why)
- The window, pool, map of the engine, the draft rows' reads: UNCHANGED (BF1 writes only the arena store, kid, kage).
  So drafts, acceptance and the round plan are those of KW.
- J's fork reads pages in ascending block id wherever they live (VR:1203-1205); a store slot holds the host block's bytes
  (host rows of a complete block do not change after the write-back; total_cis rows below qc do not change; KP:63-69).
  So J's outputs are bit-equal: EQF (tokens = L4PTFD's, every logit fingerprint) and KEEP-EQ are MUST gates.
- Hazards: (a) RAW store: J waits the last commit (barrier). (b) WAR store / map: the side stream starts a commit only
  after a draft snapshot event, which the compute stream records after J(r) ended. (c) landing reuse: the copy stream
  waits the commit event of the entry's previous use. (d) pinned staging reuse: the worker waits the copy event of the
  entry's previous use. (e) a stale block: only blocks below qc // 64 at snapshot time; qc never decreases.
- Fallback: an unused prefetched slot is an ordinary dead slot; the next miss evicts it. A late job is abandoned.
- Check-only gate: KW's kcheck byte check (KP:504-530, :438-463) compares every store-served block with a fresh host
  gather; a negative control (commit without the landing wait / wrong bytes) must fail it.

### 4.4 CUDA graphs
The copies, the commit and the barrier are EAGER ops on side streams between graph launches (draft residency and J's
residency are eager already: DG:382-391, KC header). Nothing enters a captured graph. A capture in global mode fails if
another thread calls a CUDA API: the runner pauses the worker during every capture (one lock around the backend's
begin / capture).

### 4.5 Memory (MODEL; the 84 GB cap)
- Landing ring: entries x n_cap items x 32 KB (K + V of one head-block, 64 x 128 x 2 B x 2). Default 8 x 2,048 = 0.54 GB
  device + 0.54 GB pinned host (C73); 8 x 512 = 0.13 GB (C128). Allocated after the prefill (the prefill peak binds).
- Store room at C73: per round about 77,960 / (318 x 2) = 122.6 NEW head-blocks per (head, request) (R3:120) against 128
  shared slots: X = 257 adds 128 x 65,792 B x 318 = 2.68 GB; prefill peak 77.69 + 2.68 = 80.37 GB. At C128: 16,946 /
  432 = 39.2 per (head, request): X = 129 suffices.
- Maps: kid / kage exist (KW); tags + keys 2 x H x B x X x 8 B = 1.3 MB.

## 5. Measurement

- E0 (coverage, minutes, built): RECON7 probe counted rounds on L4PTFDGUDKWCD at C73 B318 and L4PTFDGUDCD at C128 B216
  (T7/docs/NOTES_recon7.md). Output: flagA_share = NEW blocks a draft skipped since the previous J / NEW. Decision: if
  flagA_share < 0.3 at C73, BF1's MODEL gain is below 35 ms and the build should change its hint source.
- E1 (BF1 miniature, B 8, check every round): MUST EQF / KEEP-EQ / kcheck bytes_bad 0 / conservation committed = used +
  unused + evicted; negative control fails.
- E2 (accounting, 128 steps, meter on, C73 B318 and C128 B216): NEW_bf / NEW_KW, used / committed (useful bytes), late,
  no_room, pack GB/s, H2D GB/s.
- E3 (timing, 512 steps, same process, order SH -> L4PTFD -> KW (or L4PTFDGUDCD) -> BF1, then a second process in the
  reverse arm order): steady median tok/s, round ms, joint ms, the barrier wait ms, exposed H2D (probe-round gather
  spans), draft ms (interference). Decision: BF1 / KW >= 1.05 at C73 with EQF PASS; falsifier: NEW_bf / NEW_KW > 0.7
  or BF1 / KW < 1.0.
- Expected (MODEL): saving c x (115.64 ms C73, 20.25 ms C128) - barrier wait - interference (X1: 0.8-3.4 % of the
  overlapped D passes, 1.4-6.0 ms) - host GIL contention (unknown).

## 6. Prior art (scoped to R4 and code read locally)

| system | async KV fetch pattern | source |
|---|---|---|
| vLLM v0.20.2 kv_offload worker | host-built src / dst arrays, pooled stream, wait_event ordering, event.query polling; cuMemcpyBatchAsync (CUDA >= 12.8) or one cudaMemcpyAsync per block; step granularity, outside graphs | R4:496-503 |
| LMCache (1f3d4f0) | layer-wise: load_stream copies pinned objects into contiguous GPU staging (copy engine), scatter kernel into paged KV; current stream waits per layer; static token set | R4:486-494 |
| ShadowKV (e51904c) | V gathered from host by an SM kernel on copy_stream while K is rebuilt on the main stream | R4:465-474 |
| InfiniGen (3754f52) | speculative next-layer selection on a speculation stream, prefetch_cache_stream; misses not corrected | R4:454-463 |
| RetroInfer (03f912c) | host LRU on a CPU thread pool with a host sync per layer; 4 graphs per layer; copy stream for prefill offload | R4:476-484, :347-359 |
| InfLLM (12b7079) | side-stream selection + load beside local-window attention; host sync per layer | R4:299-311 |
| FlexGen | one-layer-ahead loads on 3 streams; full sync per layer | R4:446-452 |
| SGLang HiCache | NOT READ | - |
| HiSparse | NOT READ (memory note: serving blocker only) | - |

Scoped conclusion (R4:519-531): exact static-index systems overlap one layer or one step ahead with copy-engine DMA from
pinned staging + a device scatter; no source read overlaps an exact, per-layer, data-dependent selection inside a graph.
BF1 uses the draft's exact miss list of the SAME layer one round ahead as a hint and the exact J selection as the check.
