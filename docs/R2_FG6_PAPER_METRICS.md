# Paper metrics 4.3 / 4.6 / 4.8 of the residency service on the GPU: run fg6 (C, 2026-10-21)

Source: `/mnt/beegfs/ojerbi/r2-runs/fg/fg_fg6_2190876/summary.json` and `../fg_fg6_2190876.step/stdout.txt`.
- nosi 2b569f1, holder 2190876 (one A100), 2026-10-21 04:50:45Z - 05:09:53Z, exit 0, every gate passed.
- All numbers are copied from the record. Percentages and per-unit values are computed from those numbers.

## 1. The regime (state it with every number)

| item | value |
|---|---|
| model | NOSA-8B: 32 layers, 2 KV heads, head dim 128, bf16; sparse attention selects topk 64 blocks of 64 tokens per row and KV head |
| workload | pg19-test-c16k, B 8 distinct documents, prompts of S = 16,128 tokens (ragged) |
| streams | (layer, request, KV head): 16 per layer, 512 in all |
| schedule | the tick (r2_tick) TEACHER-FORCED by `forced_schedule` (rejections at every offset, both block-boundary kinds); speculative cadence K 3, lead bound LAMBDA 7; every request commits 64 tokens |
| per arm | 118 ticks (83 with verify rows, 35 draft-only); 1,510 rows (899 draft, 611 verify); 105 publications, 154 rollbacks, 6 host write-backs |
| service | S-C: verify and draft decisions on the device, one device read per tick |
| primary capacity | C 96 / X 32 / R_ver 16: 128 historical slots per stream plus 4 tail slots; 67,584 physical slots |
| pressure points (FG3) | C 72 / X 384 / R_ver 256 and C 64 / X 512 / R_ver 384: C below the drafts' protected union, so blocks are evicted and fetched again |
| unit | one BLOCK = (request, layer, KV head, 64 positions): K and V 32 KiB, plus 128 B bias |
| correctness | every arm: all 611 verify rows BITWISE equal to the all-resident reference; FG2: every resident slot equals the host truth (617,006 / 621,055 slots checked, 0 wrong) |

The arms:
- FG1: gilfree transport, wait mode superset vs exact.
- FG2: gather / s0 / gilfree transports, superset wait.
- FG3: gilfree, exact wait, at the two pressure points.

All arms have the same teacher-forced rows and so the SAME verify selections: 1,112,423 first looks in every arm.

## 2. 4.3: the verifier's selected blocks at their first look

Definitions (verbatim, `r2_residency.py` DEFINITIONS):
- READY: resident at the first look (LRU, PO or VRES), not fetched by this cohort.
- IN FLIGHT: not resident at the first look and not fetched by this cohort: another requester's transfer.
- NEW: fetched by this cohort at this layer: an urgent miss.
- The unit is one selected historical block per (cohort, layer) call: 8,288 calls, about 134 blocks per call.

| arm | ready | in flight | new (urgent) | ready % | in flight % | new % |
|---|---|---|---|---|---|---|
| FG2 gather (superset) | 1,110,857 | 0 | 1,566 | 99.86 | 0 | 0.141 |
| FG2 s0 (superset) | 1,110,857 | 0 | 1,566 | 99.86 | 0 | 0.141 |
| FG2 gilfree (superset) | 1,108,913 | 1,121 | 2,389 | 99.68 | 0.101 | 0.215 |
| FG1 gilfree superset | 1,108,902 | 1,110 | 2,411 | 99.68 | 0.100 | 0.217 |
| FG1 gilfree exact | 1,109,024 | 1,020 | 2,379 | 99.69 | 0.092 | 0.214 |
| FG3 C 72 (exact) | 1,107,752 | 1,174 | 3,497 | 99.58 | 0.106 | 0.314 |
| FG3 C 64 (exact) | 1,102,751 | 1,739 | 7,933 | 99.13 | 0.156 | 0.713 |

Notes:
- With the gather and s0 transports, every prefetch lands before the next verify reads it (in flight 0). Their state is identical: the schedule alone determines it.
- With the asynchronous gilfree transport, about 0.1% of the verifier's blocks are still in flight; the verifier waits for them on the device.

THE 4.3 WAIT. The device time of each layer's wait, from CUDA events around it on the main stream, over the run: 83 verify ticks x 32 layers = 2,656 layer waits.

| arm | device ms (run) | host ms (run) | device ms per layer wait | device ms per verify tick | share of the mean tick |
|---|---|---|---|---|---|
| FG1 superset (gilfree) | 115.2 | 305.7 | 0.043 | 1.39 | 0.18% of 785.9 ms |
| FG1 exact (gilfree) | 1,002.1 | 1,195.7 | 0.377 | 12.07 | 1.52% of 793.5 ms |
| FG2 gather (superset) | 92.0 | 282.4 | 0.035 | 1.11 | 0.14% of 782.9 ms |
| FG2 s0 (superset) | 92.3 | 287.1 | 0.035 | 1.11 | 0.14% of 798.6 ms |
| FG2 gilfree (superset) | 111.7 | 305.6 | 0.042 | 1.35 | 0.17% of 791.6 ms |
| FG3 C 72 (exact) | 700.9 | 892.5 | 0.264 | 8.44 | 0.91% of 930.2 ms |
| FG3 C 64 (exact) | 756.9 | 954.3 | 0.285 | 9.12 | 0.76% of 1,201.6 ms |

- SUPERSET: the main stream waits on the completion event of every transfer in flight at the layer. EXACT: one Triton program spins on the generation board for exactly the layer's in-flight slots.
- Both are correct here: 55,947 of 55,947 packed transfers published, err never set, and a transport without publication refuses its tick.
- Superset is the cheaper wait in this regime: one tick in about 800 ms carries 1.1-1.4 ms of wait.
- EXACT's cost is concentrated in layer 0: 331.8 ms of the 1,002.1 ms, 4.0 ms per verify tick. Layers 1-31 cost about 21.6 ms each over the run (0.26 ms per verify tick): the fixed cost of the spin program.
- The record does not explain layer 0's excess: the program records no spin count, and superset's layer 0 is 0.05 ms per verify tick. Do not cite exact's per-wait cost as a property of the wait until this is resolved.
- NOT THE 4.3 WAIT: S-C's own counters `wait_ns` (0 by construction: the arrival is not observed on the host at the look) and `layer_wait_ns` (29-46 s: the host time of applying the layer's decisions). In S-C the look and the read are both on the device, so cite only the device times above.

## 3. 4.6: what served each verifier block read

Definition: the state of the slot that served the read.
- lru / po / vres: retained.
- fetched_first / fetched_refetch: this cohort's own transfer; refetch when the block was transferred before.
- inflight_waited: another requester's transfer, waited on the device.

| arm | LRU | PO | VRES | fetched first | fetched refetch | in flight waited | peak retained (PO + VRES) |
|---|---|---|---|---|---|---|---|
| FG2 gather / s0 | 1,110,849 | 0 | 8 | 1,272 | 294 | 0 | 5 slots (160 KiB) |
| FG2 gilfree | 1,108,898 | 0 | 15 | 1,986 | 403 | 1,121 | 12 slots (384 KiB) |
| FG1 superset | 1,108,887 | 0 | 15 | 2,000 | 411 | 1,110 | 12 slots (384 KiB) |
| FG1 exact | 1,109,009 | 0 | 15 | 1,969 | 410 | 1,020 | 11 slots (352 KiB) |
| FG3 C 72 | 1,100,200 | 7,552 | 0 | 1,867 | 1,630 | 1,174 | 605 slots (18.9 MiB) |
| FG3 C 64 | 1,037,607 | 65,130 | 14 | 2,554 | 5,379 | 1,739 | 2,792 slots (87.3 MiB) |

- At the primary point the LRU serves 99.7% of the verifier's reads. PO (protected-only, evicted from the LRU but kept for a live cohort) serves none.
- Under pressure PO matters: at C 64 it serves 65,130 reads (5.9%), with a retained peak of 87 MiB.
- Of the verifier's own fetches, 17% (primary) to 68% (C 64) are refetches of blocks transferred before.

## 4. 4.8: the use of every transfer

Definition: every transfer is one slot lifetime, classified by its reads when it ended.
- draft only / verify only / both / neither;
- still live at the report;
- never arrived = ended before its arrival was observed;
- repeated = transfers minus distinct (request, layer, KV head, block).

| arm | transfers | distinct | repeated | both | verify only | draft only | neither | still live | never arrived |
|---|---|---|---|---|---|---|---|---|---|
| FG2 gather / s0 | 57,518 | 52,235 | 5,283 (9.2%) | 52,244 (90.8%) | 3,803 (6.6%) | 130 (0.23%) | 1,340 (2.3%) | 1 | 0 |
| FG2 gilfree | 58,521 | 52,642 | 5,879 (10.0%) | 51,868 (88.6%) | 4,620 (7.9%) | 122 (0.21%) | 1,910 (3.3%) | 1 | 0 |
| FG1 superset | 58,492 | 52,649 | 5,843 (10.0%) | 51,826 (88.6%) | 4,624 (7.9%) | 125 (0.21%) | 1,916 (3.3%) | 1 | 0 |
| FG1 exact | 58,326 | 52,485 | 5,841 (10.0%) | 51,846 (88.9%) | 4,611 (7.9%) | 132 (0.23%) | 1,736 (3.0%) | 1 | 0 |
| FG3 C 72 | 91,865 | 52,612 | 39,253 (42.7%) | 68,885 (75.0%) | 15,924 (17.3%) | 754 (0.82%) | 6,284 (6.8%) | 18 | 0 |
| FG3 C 64 | 158,527 | 52,741 | 105,786 (66.7%) | 93,384 (58.9%) | 43,529 (27.5%) | 2,417 (1.5%) | 19,163 (12.1%) | 34 | 0 |

By issuer, FG1 exact:
- Draft prefetches: 55,947. Both 51,455 (92.0%), verify only 2,624, draft only 132, neither 1,736 (3.1%).
- The verifier's urgent fetches: 2,379. Both 391, verify only 1,987, still live 1.

So, at the primary point:
- About 89% of all transfers serve both a draft read and a verify read.
- Only about 3% are never read.
- About 10% of the transfers repeat a block transferred before.

Under pressure the waste grows: neither 6.8% / 12.1%, repeated 43% / 67%.

Volume at 32 KiB per block: 58,326 transfers = 1.78 GiB per arm at the primary point; 158,527 = 4.84 GiB at C 64.

## 5. The draft side, for context

| arm | draft (row, head) pairs | masked fraction (D3) | K5 draft refusals | LRU evictions (evict_po + evict) | Policy E (verifier evictions) | mean tick |
|---|---|---|---|---|---|---|
| FG2 gather / s0 | 57,536 | 22.4% | 507 | 9,218 | 0 | 782.9 / 798.6 ms |
| FG2 gilfree | 57,536 | 24.0% | 1,311 | 9,413 | 15 | 791.6 ms |
| FG1 superset | 57,536 | 24.1% | 1,382 | 9,366 | 15 | 785.9 ms |
| FG1 exact | 57,536 | 24.0% | 1,304 | 9,292 | 17 | 793.5 ms |
| FG3 C 72 | 57,536 | 42.7% | 130 | 54,051 | 0 | 930.2 ms |
| FG3 C 64 | 57,536 | 82.6% | 6,471 | 154,818 | 0 | 1,201.6 ms |

- Masked fraction (D3): the draft (row, head) pairs that did not read every selected block, out of all draft pairs.
- A draft row attends only to its READY blocks. Its missing blocks are prefetched for the verifier, and drafting continues.

## 6. Caveats for the writer

1. B 8 (16 streams per layer) and a teacher-forced schedule. The production regime (SG5, B 64, free-running greedy) has different pressure: there C 96 needs X 64 for the shared extras, and Policy E serves about 13 urgent misses per layer per verify tick (record p4sc64). Do not transfer the FG percentages to B 64.
2. The waits are device times measured by events on the main stream. The superset / exact comparison holds for this transport (gilfree, Python feeder, 8 threads) and this regime only.
3. Exact's layer-0 excess is unexplained (section 2).
4. The identical gather / s0 rows are expected: both transports are complete before the next verify in this regime, and the state depends on the schedule only.
