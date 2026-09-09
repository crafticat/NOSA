"""CUDA-event timing of the host->device fetch and of the attention inside
NOSI's decode step, per layer, without changing the step's numerics.

Added by the retroinfer-eval fork (branch `nosi-transfer-test`, 2026-09-02)
for the transfer test of spec 2026-09-02-nosi-base-program.md, P1.3. The
upstream files carry only `if TRACE is not None:` guards around calls into
this module; with NOSI_TRANSFER_TRACE unset, TRACE is None and the decode
path executes the upstream statements and nothing else.

Modes (environment variable NOSI_TRANSFER_TRACE):
  unset / "0" -> TRACE is None: upstream-exact, zero hooks execute.
  "logits"    -> only the per-step logits copy (the numerics reference).
  "1"         -> per layer: events before diff_offload, between diff_offload
                 and the two flash_h2d calls, after the two flash_h2d calls,
                 and around flash_attn_nosa_with_kvcache; per step: events
                 around the whole decode_inference; an archive of `_load_mask`
                 and `_block_map` per (step, layer); the per-step logits copy.
  "scores"    -> mode "1" plus, per (step, layer), the two pooled score
                 buffers the selection is drawn from (`max_pooling_buf`: the
                 QK score; `max_pooling_buf_cis`: the cis score with the
                 QK-selected blocks forced to +inf), read after the captured
                 pooling->top-k graph replays. Phase P2 (offline predictor
                 study) ranks the blocks below the top-64 from these.

Nothing here synchronizes inside a step: events are recorded on the current
stream, masks are copied device-to-device into a preallocated archive, logits
are cloned on device. `harvest()` synchronizes once, after the benchmark's
own final synchronize, and turns the events into milliseconds.

Bytes. NOSA-8B: 2 KV heads, head_dim 128, block 64 tokens, bf16. One loaded
(head, request, slot) entry moves K (16 KiB) + V (16 KiB) = 32 KiB over
PCIe. `flash_h2d_from_mask_bias` also gathers 64 bf16 kv_bias values per
entry, but its source (`total_cis`) lives on the GPU (cache_engine.py
`update_uncompressed_cis`: `device=cis.device`), so those 128 B are a
device-to-device copy and are NOT counted as wire bytes. With the
GPU-resident cache (cache_engine_gpu.py, offload=False) the same gather runs
device-to-device; rows from that engine carry offload=0 and their "transfer"
is a D2D time, labelled as such by the harness.
"""
from __future__ import annotations

import hashlib
import os

import torch

MODE = os.environ.get("NOSI_TRANSFER_TRACE", "0")

# retroinfer-eval fork: the victim pool (cache_engine.py, knob
# NOSI_POOL_BLOCKS). Read the same env var so this module never has to import
# cache_engine. With the pool off, begin_layer builds exactly the event dict it
# built before, so a mode-"1" P=0 run records what it records today.
POOL_BLOCKS = int(os.environ.get("NOSI_POOL_BLOCKS", "0") or 0)
POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3
POOL_SERVED = (POOL_SWAP, POOL_MOVE_IN)   # actions that served a block from HBM

BLOCK_TOKENS = 64
HEAD_DIM = 128
BYTES_K = BLOCK_TOKENS * HEAD_DIM * 2          # 16384, bf16
BYTES_V = BYTES_K
BLOCK_BYTES_WIRE = BYTES_K + BYTES_V           # 32768 per loaded (head, request, slot)
BYTES_BIAS_D2D = BLOCK_TOKENS * 2              # 128, device-to-device, not wire


_LAYER_EVENTS = ("fetch_begin", "fetch_mid", "fetch_end", "attn_begin", "attn_end")
_LAYER_EVENTS_POOL = _LAYER_EVENTS + ("pool_begin", "pool_end")


def _ev():
    return torch.cuda.Event(enable_timing=True)


class _Step:
    __slots__ = ("index", "start", "end", "layers", "logits", "full")

    def __init__(self, index: int, num_layers: int, full: bool):
        self.index = index
        self.full = full
        self.start = _ev() if full else None
        self.end = _ev() if full else None
        self.layers = [None] * num_layers
        self.logits = None


class TransferTrace:
    """One per process. Call order inside a decode step (all from upstream
    code, each behind an `if TRACE is not None` guard):
      begin_step -> [begin_layer -> fetch_begin -> fetch_mid -> fetch_end ->
      record_mask -> attn_begin -> attn_end] x layers -> record_logits -> end_step
    """

    def __init__(self, num_layers: int, mode: str, max_steps: int = 16):
        self.num_layers = num_layers
        self.mode = mode
        self.full = mode in ("1", "scores")
        self.scores = mode == "scores"
        self.score_archive = None     # (max_steps, L, 2, H, B, out_len) bf16, device
        self.max_steps = max_steps
        self.steps: list[_Step] = []
        self.cur: _Step | None = None
        self.layer: int | None = None
        self.mask_archive = None      # (max_steps, L, H, B, M) int64, device
        self.map_archive = None
        self.pool_archive = None      # (max_steps, L, H, B, M) int8, device
        self.avail_archive = None     # (max_steps, L, H, B, M) int8, device: denials
        self.pool_blocks = POOL_BLOCKS
        self.dropped_steps = 0        # steps beyond max_steps: recorded nowhere
        # BOUND AT CONSTRUCTION, the way cache_engine.py binds `decode_update`,
        # so that a P=0 mode-"1" run executes the SAME instrument it executed
        # before the pool existed -- no per-event ternary and no per-event dict
        # lookup, 160 of which would otherwise land inside every timed step.
        # This is what keeps "P=0 is the upstream path" true of the instrument
        # as well as of the engine.
        self._event_names = _LAYER_EVENTS_POOL if POOL_BLOCKS > 0 else _LAYER_EVENTS
        self._rec = self._rec_pool if POOL_BLOCKS > 0 else self._rec_plain
        # WHICH ENGINE IS RUNNING decides whether the pool events exist, not the
        # environment variable. cache_engine_gpu.py (NOSI_BENCH_OFFLOAD=0) has
        # no pool and never calls pool_begin/pool_end, so with the knob set and
        # offload off the events would be CREATED and never RECORDED, and
        # elapsed_time() would raise inside harvest and lose the whole document.
        self.pool_recorded = False

    # -- document / step boundaries ------------------------------------------
    def new_document(self):
        self.steps = []
        self.cur = None
        self.layer = None
        self.dropped_steps = 0

    def begin_step(self):
        if len(self.steps) >= self.max_steps:
            self.dropped_steps += 1
            self.cur = None
            return
        s = _Step(len(self.steps), self.num_layers, self.full)
        self.steps.append(s)
        self.cur = s
        if self.full:
            s.start.record()

    def end_step(self):
        if self.cur is not None and self.full:
            self.cur.end.record()
        self.cur = None
        self.layer = None

    def record_logits(self, logits: torch.Tensor):
        if self.cur is not None:
            self.cur.logits = logits.detach().clone()

    # -- per layer -------------------------------------------------------------
    def begin_layer(self, layer_idx: int):
        self.layer = layer_idx
        if self.cur is not None and self.full:
            self.cur.layers[layer_idx] = {k: _ev() for k in self._event_names}

    def _rec_plain(self, key: str):
        """The pre-pool recorder, bound when POOL_BLOCKS == 0: every key in
        _LAYER_EVENTS exists, so there is nothing to look up defensively."""
        if self.cur is None or not self.full or self.layer is None:
            return
        d = self.cur.layers[self.layer]
        if d is not None:
            d[key].record()

    def _rec_pool(self, key: str):
        """Bound when POOL_BLOCKS > 0. A layer dict built before the knob was
        read (or by another engine) may lack the pool keys, so miss quietly."""
        if self.cur is None or not self.full or self.layer is None:
            return
        d = self.cur.layers[self.layer]
        if d is not None:
            e = d.get(key)
            if e is not None:
                e.record()

    def fetch_begin(self):
        self._rec("fetch_begin")

    def fetch_mid(self):
        self._rec("fetch_mid")

    def fetch_end(self):
        self._rec("fetch_end")

    def attn_begin(self):
        self._rec("attn_begin")

    def attn_end(self):
        self._rec("attn_end")

    def pool_begin(self):
        # the flag is what harvest trusts: only the offload CacheEngine calls
        # this, so it is proof the events were really recorded
        self.pool_recorded = True
        self._rec("pool_begin")

    def pool_end(self):
        self._rec("pool_end")

    def record_mask(self, load_mask: torch.Tensor, block_map: torch.Tensor):
        if self.cur is None or not self.full or self.layer is None:
            return
        if self.mask_archive is None or self.mask_archive.shape[2:] != tuple(load_mask.shape):
            shape = (self.max_steps, self.num_layers) + tuple(load_mask.shape)
            self.mask_archive = torch.full(shape, -2, dtype=load_mask.dtype, device=load_mask.device)
            self.map_archive = torch.full(shape, -2, dtype=block_map.dtype, device=block_map.device)
        self.mask_archive[self.cur.index, self.layer].copy_(load_mask, non_blocking=True)
        self.map_archive[self.cur.index, self.layer].copy_(block_map, non_blocking=True)

    def record_pool(self, pool_action: torch.Tensor):
        """Victim pool only: archive this layer's per-attended-slot action code
        (device-to-device, no sync), so `harvest` can count how many blocks the
        pool served instead of the wire. Same shape as the load mask."""
        if self.cur is None or not self.full or self.layer is None:
            return
        if self.pool_archive is None or self.pool_archive.shape[2:] != tuple(pool_action.shape):
            shape = (self.max_steps, self.num_layers) + tuple(pool_action.shape)
            self.pool_archive = torch.full(shape, -1, dtype=pool_action.dtype, device=pool_action.device)
        self.pool_archive[self.cur.index, self.layer].copy_(pool_action, non_blocking=True)

    def record_avail(self, denied: torch.Tensor):
        """RESTRICTED AVAILABILITY only (avail_policy.py, knob NOSI_AVAIL):
        archive this layer's per-attended-slot DENIAL mask, device-to-device and
        with no synchronisation, modelled line for line on record_pool above.

        THREE QUANTITIES, KEPT APART, and this archive holds the third.
          (a) CAPACITY is a configuration number, not a measurement: 63 usable
              attended slots (slot topk-1 is the tail, written locally and never
              fetched) plus the pool, which is 0 for this pilot.
          (b) THE LRU HIT FRACTION at that capacity is
              1 - (slots with _load_mask >= 0) / 63, and `record_mask` above
              already archives the mask it is read from.
          (c) READY AT THE INSTANT THE DRAFT ATTENDS is 1 - denials / 63, and it
              is the one that drives acceptance, because a block that is still in
              flight is not usable however large the cache is.
        A `denied` of None (a target step, or a draft step that denied nothing)
        leaves the archive at 0 for that (step, layer), which is the correct
        reading: nothing was withheld."""
        if self.cur is None or not self.full or self.layer is None or denied is None:
            return
        if self.avail_archive is None or self.avail_archive.shape[2:] != tuple(denied.shape):
            shape = (self.max_steps, self.num_layers) + tuple(denied.shape)
            self.avail_archive = torch.zeros(shape, dtype=denied.dtype, device=denied.device)
        self.avail_archive[self.cur.index, self.layer].copy_(denied, non_blocking=True)

    def record_scores(self, qk_scores: torch.Tensor, cis_scores: torch.Tensor):
        """Mode "scores" only: copy the (H, B, out_len) pooled score buffers of
        the current layer into the archive (device-to-device, no sync)."""
        if self.cur is None or not self.scores or self.layer is None:
            return
        if self.score_archive is None or self.score_archive.shape[3:] != tuple(qk_scores.shape):
            shape = (self.max_steps, self.num_layers, 2) + tuple(qk_scores.shape)
            self.score_archive = torch.full(shape, float("nan"), dtype=qk_scores.dtype, device=qk_scores.device)
        self.score_archive[self.cur.index, self.layer, 0].copy_(qk_scores, non_blocking=True)
        self.score_archive[self.cur.index, self.layer, 1].copy_(cis_scores, non_blocking=True)

    # -- harvest (one synchronize, after the timed region) ----------------------
    def harvest(self, timed_steps=(1, 2, 3, 4)):
        """Returns (step_rows, layer_rows, logits_rows, masks, maps).

        step_rows: one dict per recorded step (see `aggregate_step`);
        layer_rows: one dict per (step, layer); logits_rows: per step, the
        sha256 of the full fp32 logits tensor plus row 0; masks/maps: the
        archives on CPU (steps, L, H, B, M) int64, or None outside mode "1".
        """
        torch.cuda.synchronize()
        step_rows, layer_rows, logits_rows = [], [], []
        n_steps = len(self.steps)
        masks = maps = pools = None
        avails = None
        H = B = M = 0
        if self.full and self.mask_archive is not None:
            masks = self.mask_archive[:n_steps].cpu()
            maps = self.map_archive[:n_steps].cpu()
            _, _, H, B, M = masks.shape
        if self.full and self.pool_archive is not None:
            pools = self.pool_archive[:n_steps].cpu()
        if self.full and self.avail_archive is not None:
            avails = self.avail_archive[:n_steps].cpu()
        for s in self.steps:
            if s.logits is not None:
                lg = s.logits.float().cpu().contiguous()
                logits_rows.append({"step": s.index,
                                    "logits_sha256": hashlib.sha256(lg.numpy().tobytes()).hexdigest(),
                                    "logits_row0": lg[0].clone(),
                                    "shape": tuple(lg.shape)})
            if not self.full:
                continue
            layer_stats = []
            for l, d in enumerate(s.layers):
                if d is None:
                    continue
                st = {
                    "layer": l,
                    "diff_ms": d["fetch_begin"].elapsed_time(d["fetch_mid"]),
                    "transfer_ms": d["fetch_mid"].elapsed_time(d["fetch_end"]),
                    "attn_ms": d["attn_begin"].elapsed_time(d["attn_end"]),
                    "blocks_loaded": int((masks[s.index, l] >= 0).sum()) if masks is not None else -1,
                    "pool_hits": (int(((pools[s.index, l] == POOL_SWAP) | (pools[s.index, l] == POOL_MOVE_IN)).sum())
                                  if pools is not None else -1),
                    # `"pool_begin" in d` alone is NOT enough: the key is created
                    # from the env var but recorded only by cache_engine.py's
                    # pooled decode, so an offload=0 run with the knob set would
                    # raise "Both events must be recorded" here and lose the
                    # document. self.pool_recorded is set by pool_begin itself.
                    "pool_ms": (d["pool_begin"].elapsed_time(d["pool_end"])
                                if (self.pool_recorded and "pool_begin" in d) else 0.0),
                    "fetch_in": bool(s.start.elapsed_time(d["fetch_begin"]) >= 0 and d["fetch_end"].elapsed_time(s.end) >= 0),
                    "attn_in": bool(s.start.elapsed_time(d["attn_begin"]) >= 0 and d["attn_end"].elapsed_time(s.end) >= 0),
                    "attn_after": bool(d["fetch_end"].elapsed_time(d["attn_begin"]) >= 0),
                    # AVAILABILITY, per (step, layer). blocks_requested_miss is
                    # (b) the LRU miss count at this capacity, read BEFORE any
                    # denial -- cache_engine.py hands record_mask the policy's
                    # pre-denial clone in mech=stale, where the live mask has
                    # already had the denied slots set to -1, and the live mask
                    # everywhere else, where nothing rewrites it;
                    # blocks_denied is what the policy withheld;
                    # ready_slots is (c) what attention actually had. All three
                    # count the 63 non-tail slots per (KV head, request), never
                    # 64: slot topk-1 is the tail, written locally and never
                    # fetched, so counting it would move every rate by ~1.6%.
                    "blocks_requested_miss": (int((masks[s.index, l][..., :M - 1] >= 0).sum())
                                              if masks is not None and M > 1 else -1),
                    "blocks_denied": (int((avails[s.index, l][..., :M - 1] > 0).sum())
                                      if avails is not None and M > 1 else -1),
                    "ready_slots": (int(H * B * (M - 1) - (avails[s.index, l][..., :M - 1] > 0).sum())
                                    if avails is not None and M > 1 else -1),
                }
                layer_stats.append(st)
                layer_rows.append({"step": s.index, "layer": l, "diff_ms": st["diff_ms"],
                                   "transfer_ms": st["transfer_ms"], "attn_ms": st["attn_ms"],
                                   "blocks_loaded": st["blocks_loaded"],
                                   "pool_hits": st["pool_hits"], "pool_ms": st["pool_ms"],
                                   "blocks_requested_miss": st["blocks_requested_miss"],
                                   "blocks_denied": st["blocks_denied"],
                                   "ready_slots": st["ready_slots"]})
            same = -1
            if masks is not None and B > 1:
                same = int(bool((maps[s.index, :, :, 0, :] == maps[s.index, :, :, B - 1, :]).all()))
            step_rows.append(aggregate_step(s.index, s.index in timed_steps, s.start.elapsed_time(s.end),
                                            layer_stats, H, B, M, self.dropped_steps, same))
        return step_rows, layer_rows, logits_rows, masks, maps


def aggregate_step(index: int, timed: bool, step_ms: float, layer_stats: list, H: int, B: int, M: int,
                   dropped_steps: int = 0, batch_identical: int = -1) -> dict:
    """The per-step transfer numbers from per-layer measurements. Pure: no
    CUDA, so the arithmetic is CPU-tested (retroinfer-eval
    tests/test_nosi_transfer_trace.py). `layer_stats` holds one dict per
    recorded layer: diff_ms, transfer_ms, attn_ms, blocks_loaded (-1 when the
    mask archive is absent), fetch_in, attn_in, attn_after (window booleans).
      transfer_ms  = sum over layers of the [fetch_mid, fetch_end] window (the
                     two flash_h2d launches);
      bytes        = blocks_loaded x BLOCK_BYTES_WIRE (K+V per loaded entry);
      eff_gbps     = bytes / transfer_ms (GB = 1e9 B);
      gamma        = 1 - blocks_loaded / (layers x H x B x M): the fraction of
                     the selected set already resident;
      new_blocks_per_head_layer = blocks_loaded / (layers x H x B).

    VICTIM POOL (NOSI_POOL_BLOCKS > 0). Three more keys, ALWAYS present so the
    steps.csv schema is one schema across pool sizes:
      pool_hits    = attended slots served from the GPU victim pool instead of
                     the wire (-1 when the pool archive is absent);
      pool_ms      = sum over layers of the [pool_begin, pool_end] window. NOTE
                     that window is NESTED INSIDE [fetch_begin, fetch_mid], so
                     `diff_ms` INCLUDES it; `diff_only_ms` = diff_ms - pool_ms
                     is the comparable-to-baseline diff_offload cost;
      blocks_entering = blocks_loaded + pool_hits: how many blocks entered the
                     selection. THIS IS THE SELF-CHECK THAT MATTERS. It is a
                     property of the selection alone, so at a fixed cell it must
                     be the SAME at every pool size while blocks_loaded falls.
                     If it moves with the pool size, the pool has changed which
                     blocks the model attends to and the cell is VOID.
    `layer_stats` entries without the pool keys default to -1 / 0.0, so a
    baseline row aggregates exactly as before.
    """
    n_rec = len(layer_stats)
    tr_ms = sum(s["transfer_ms"] for s in layer_stats)
    diff_ms = sum(s["diff_ms"] for s in layer_stats)
    attn_ms = sum(s["attn_ms"] for s in layer_stats)
    pool_ms = sum(s.get("pool_ms", 0.0) for s in layer_stats)
    known = [s["blocks_loaded"] for s in layer_stats if s["blocks_loaded"] >= 0]
    loaded_total = sum(known) if known else -1
    known_hits = [s.get("pool_hits", -1) for s in layer_stats if s.get("pool_hits", -1) >= 0]
    pool_hits = sum(known_hits) if known_hits else -1
    entering = (loaded_total + max(pool_hits, 0)) if loaded_total >= 0 else -1
    nbytes = loaded_total * BLOCK_BYTES_WIRE if loaded_total >= 0 else -1
    denom = n_rec * H * B * M
    return {
        "step": index,
        "timed": int(bool(timed)),
        "step_ms": step_ms,
        "transfer_ms": tr_ms,
        "diff_ms": diff_ms,
        "pool_ms": pool_ms,
        "diff_only_ms": diff_ms - pool_ms,
        "attn_ms": attn_ms,
        "blocks_loaded": loaded_total,
        "pool_hits": pool_hits,
        "blocks_entering": entering,
        "pool_blocks": POOL_BLOCKS,
        "block_bytes": BLOCK_BYTES_WIRE,
        "bytes": nbytes,
        "eff_gbps": (nbytes / tr_ms / 1e6) if (tr_ms > 0 and nbytes >= 0) else 0.0,
        "transfer_share": (tr_ms / step_ms) if step_ms > 0 else 0.0,
        "gamma": (1.0 - loaded_total / denom) if (loaded_total >= 0 and denom > 0) else -1.0,
        "new_blocks_per_head_layer": (loaded_total / (n_rec * H * B)) if (loaded_total >= 0 and n_rec * H * B > 0) else -1.0,
        "layers_recorded": n_rec,
        "fetch_in_step": int(all(s["fetch_in"] for s in layer_stats)) if n_rec else 0,
        "attn_in_step": int(all(s["attn_in"] for s in layer_stats)) if n_rec else 0,
        "attn_after_fetch": int(all(s["attn_after"] for s in layer_stats)) if n_rec else 0,
        "batch_identical_selection": batch_identical,
        "dropped_steps": dropped_steps,
    }

TRACE: TransferTrace | None = None


def install(num_layers: int, max_steps: int = 16) -> TransferTrace | None:
    """Create the process-wide trace for the mode in NOSI_TRANSFER_TRACE.
    Returns None (and installs nothing) when the mode is unset/"0"."""
    global TRACE
    if MODE in ("0", "", "off"):
        TRACE = None
    else:
        TRACE = TransferTrace(num_layers, MODE, max_steps)
    return TRACE


def dump(out_dir: str, doc_idx: int, meta: dict, timed_steps=(1, 2, 3, 4)) -> dict:
    """Harvest the current document into CSV rows (appended) and .pt files.
    Returns the summary over the timed steps. `meta` (context, batch,
    offload, mode, ...) is copied into every row."""
    import csv
    assert TRACE is not None
    os.makedirs(out_dir, exist_ok=True)
    step_rows, layer_rows, logits_rows, masks, maps = TRACE.harvest(timed_steps)
    def _append(name, rows):
        if not rows:
            return
        rows = [{**meta, "doc_idx": doc_idx, **r} for r in rows]
        p = os.path.join(out_dir, name)
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
    _append("steps.csv", step_rows)
    _append("layers.csv", layer_rows)
    _append("logits.csv", [{k: v for k, v in r.items() if k != "logits_row0"} for r in logits_rows])
    tag = f"ctx{meta.get('context')}_b{meta.get('batch')}_off{meta.get('offload')}_doc{doc_idx}"
    if logits_rows:
        torch.save({"rows0": torch.stack([r["logits_row0"] for r in logits_rows]),
                    "sha256": [r["logits_sha256"] for r in logits_rows], "meta": meta},
                   os.path.join(out_dir, f"logits_{tag}.pt"))
    if masks is not None:
        # every request in the benchmark batch is the same document; row 0 is
        # the selection trace P2 needs, the batch-identity flag is in steps.csv
        torch.save({"masks_b0": masks[:, :, :, 0, :].to(torch.int32).clone(),
                    "maps_b0": maps[:, :, :, 0, :].to(torch.int32).clone(), "meta": meta},
                   os.path.join(out_dir, f"selection_{tag}.pt"))
    if TRACE.scores and TRACE.score_archive is not None and masks is not None:
        n_steps = masks.shape[0]
        torch.save({"masks": masks.to(torch.int32).clone(), "maps": maps.to(torch.int32).clone(),
                    "scores": TRACE.score_archive[:n_steps].cpu().clone(), "meta": meta},
                   os.path.join(out_dir, f"scores_{tag}.pt"))
    timed = [r for r in step_rows if r["timed"]]
    summ = {}
    if timed:
        for k in ("step_ms", "transfer_ms", "diff_ms", "pool_ms", "attn_ms", "blocks_loaded",
                  "pool_hits", "blocks_entering", "bytes", "transfer_share", "gamma"):
            summ[k] = sum(r[k] for r in timed) / len(timed)
        summ["eff_gbps"] = (summ["bytes"] / summ["transfer_ms"] / 1e6) if summ["transfer_ms"] > 0 else 0.0
        summ["n_timed"] = len(timed)
    return summ
