"""Pure-python helpers of the TWO-PASS RE-DRAFT experiment (two_pass.py). No torch, no CUDA: everything here is
CPU-testable (retroinfer-eval tests/test_two_pass_plan.py) and operates on the acceptance pilot's own VirtualCache
objects (avail_policy.VirtualCache, duck-typed: .res OrderedDict in recency order, .inflight dict, .t, .capacity,
.delay, .instep, .clone()).

Experiment label, used verbatim on every table, row family, manifest and caption:
    "same-position K=1 accuracy (single-token overlap, 1-TV)"
It is NOT an accepted-prefix / K-round measurement and must never be reported as one.

Sets. For one (layer, KV head, request) stream and one pass n:
    sel_n   the non-tail block ids the pass selected (the tail slot, the in-progress block, is excluded everywhere and
            is never denied -- the pilot's rule, avail_policy.py:82-87);
    req_n   the ids of sel_n the draft could NOT use (not resident in the draft LRU at attention) = the pass's misses,
            which the pass REQUESTS;
    before_n  the draft LRU's resident set at the instant the pass attended (post-arrival, avail_policy.py:227).
Gather after pass n admits exactly req_n into the LRU at capacity C with the pilot's arrival rule (res[b] = None;
move_to_end; then evict least-recent while over capacity, avail_policy.py:220-226) and records every eviction.

Categories (codes are stored in the raw export):
    1 NEW_NEVER_RESIDENT   (a) a new miss of pass n >= 2 that was never resident at S0 and never gathered;
    2 S0_EVICTED_BY_GATHER (b) resident at S0 (before_1) but evicted by a gather, then needed again;
    3 GATHERED_THEN_EVICTED (c) requested (and gathered) by an earlier pass but evicted before this pass;
    4 EVICTED_UNUSED        an evicted id that the next pass did not need.
New-miss ids carry 1/2/3; evicted ids carry 2/3/4.
"""
from __future__ import annotations

import json
import os

import numpy as np

LABEL = "same-position K=1 accuracy (single-token overlap, 1-TV)"

NEW_NEVER_RESIDENT = 1
S0_EVICTED_BY_GATHER = 2
GATHERED_THEN_EVICTED = 3
EVICTED_UNUSED = 4
CATEGORY_NAMES = {
    NEW_NEVER_RESIDENT: "(a) new miss, never resident at S0 and never gathered",
    S0_EVICTED_BY_GATHER: "(b) resident at S0, evicted by a gather",
    GATHERED_THEN_EVICTED: "(c) requested by an earlier pass (gathered) but evicted before this pass",
    EVICTED_UNUSED: "evicted and not needed by the next pass",
}

# Transfer units. NOSI's host cache per layer is (B, S, H=2, Dh=128) bf16: one token row of one head = 256 B, so one
# PER-HEAD block (64 rows) = 16 KiB of K and 16 KiB of V; one BOTH-HEADS block = 32 KiB of K and 32 KiB of V.
PER_HEAD_BLOCK_K_BYTES = 64 * 128 * 2
PER_HEAD_BLOCK_KV_BYTES = 2 * PER_HEAD_BLOCK_K_BYTES
BOTH_HEADS_BLOCK_K_BYTES = 2 * PER_HEAD_BLOCK_K_BYTES
BOTH_HEADS_BLOCK_KV_BYTES = 2 * BOTH_HEADS_BLOCK_K_BYTES


# ---------------------------------------------------------------------------------------------------------------
# LRU state
def lru_fingerprint(streams) -> tuple:
    """Content AND recency order of every stream, plus the in-flight queue, the step counter and the policy knobs.
    Two working sets with equal fingerprints drive every later step identically."""
    out = []
    for key in sorted(streams):
        vc = streams[key]
        out.append((tuple(key), tuple(vc.res.keys()), tuple(sorted(vc.inflight.items())), int(vc.t),
                    int(vc.capacity), float(vc.delay), float(vc.instep)))
    return tuple(out)


def clone_streams(streams) -> dict:
    return {k: v.clone() for k, v in streams.items()}


def admit(vc, blocks) -> tuple:
    """GATHER: admit exactly `blocks` into `vc` with the pilot's arrival rule, in sorted order (the order the pilot
    issues misses, avail_policy.py:229/242, so a natural d = 1 arrival would land them in the same order).
    Returns (evicted ids in eviction order, stray in-flight ids). A stray in-flight id is an in-flight request that is
    NOT in `blocks`: it would land at the next step and break "admit exactly the requested blocks", so it is removed
    and reported (the driver counts it as a consistency failure)."""
    want = sorted(set(int(b) for b in blocks))
    wset = set(want)
    stray = sorted(b for b in vc.inflight if b not in wset)
    for b in stray:
        del vc.inflight[b]
    for b in want:
        if b in vc.inflight:
            del vc.inflight[b]
        vc.res[b] = None
        vc.res.move_to_end(b)
    evicted = []
    while len(vc.res) > vc.capacity:
        evicted.append(vc.res.popitem(last=False)[0])
    return evicted, stray


# ---------------------------------------------------------------------------------------------------------------
# capture: one on_diff call -> per-key selected ids and denied (requested) ids
def selection_and_requests(new_map, deny, tail: int, layer: int) -> dict:
    """`new_map` (H, B, M) nested lists = the slot -> requested block id map the policy saw; `deny` (H, B, M) nested
    bools or None = the policy's denial for this layer. Returns {(layer, h, b): (sel ids in slot order, requested ids
    in slot order)}. The tail slot and empty slots (id < 0) are excluded, exactly as avail_policy.on_diff does; a
    denied tail is a hard error (the pilot asserts the same)."""
    out = {}
    H = len(new_map)
    for h in range(H):
        for b in range(len(new_map[h])):
            row = new_map[h][b]
            sel, req = [], []
            for m, blk in enumerate(row):
                if m == tail:
                    if deny is not None and deny[h][b][m]:
                        raise AssertionError("tail slot denied at layer %d head %d req %d" % (layer, h, b))
                    continue
                if blk < 0:
                    continue
                sel.append(int(blk))
                if deny is not None and deny[h][b][m]:
                    req.append(int(blk))
            out[(layer, h, b)] = (sel, req)
    return out


def check_requests(sel, req, before) -> bool:
    """For a delayed / fixed-residency point with f = 0, the pilot denies exactly the selected ids that are not
    resident (avail_policy.py:228-235, 550-562). True when the capture agrees."""
    return sorted(req) == sorted(b for b in sel if b not in before)


# ---------------------------------------------------------------------------------------------------------------
# categories
def categorize_new_misses(req_n, s0_res, gathered_before, evicted_before) -> tuple:
    """Categories of the misses of a pass n >= 2. `gathered_before` / `evicted_before` = unions over the gathers that
    ran before pass n. Returns ([(id, code)], n_inconsistent). An id that was gathered or resident at S0 but NOT
    evicted should have been resident, so the pass could not have missed it: that is an inconsistency."""
    out, bad = [], 0
    for b in req_n:
        if b in gathered_before:
            code = GATHERED_THEN_EVICTED
            bad += int(b not in evicted_before)
        elif b in s0_res:
            code = S0_EVICTED_BY_GATHER
            bad += int(b not in evicted_before)
        else:
            code = NEW_NEVER_RESIDENT
        out.append((int(b), code))
    return out, bad


def categorize_evictions(evicted_g, req_next, gathered_upto_g) -> list:
    """Categories of the ids evicted by gather g, given the misses of the NEXT pass."""
    need = set(req_next)
    out = []
    for e in evicted_g:
        if e in need:
            out.append((int(e), GATHERED_THEN_EVICTED if e in gathered_upto_g else S0_EVICTED_BY_GATHER))
        else:
            out.append((int(e), EVICTED_UNUSED))
    return out


def transfer_units(gathered_by_key) -> dict:
    """`gathered_by_key` {(layer, h, b): ids}. Per-head unit = one (layer, head, request, block); both-heads unit =
    one (layer, request, block) moved for both heads at once (NOSI's contiguous 32 KiB host block), counted once even
    when only one head needed it."""
    per_head = sum(len(v) for v in gathered_by_key.values())
    both = set()
    for (layer, h, b), ids in gathered_by_key.items():
        for i in ids:
            both.add((layer, b, int(i)))
    return dict(per_head_blocks=int(per_head), both_heads_blocks=int(len(both)),
                bytes_k_per_head_unit=int(per_head * PER_HEAD_BLOCK_K_BYTES),
                bytes_kv_per_head_unit=int(per_head * PER_HEAD_BLOCK_KV_BYTES),
                bytes_k_both_heads_unit=int(len(both) * BOTH_HEADS_BLOCK_K_BYTES),
                bytes_kv_both_heads_unit=int(len(both) * BOTH_HEADS_BLOCK_KV_BYTES))


# ---------------------------------------------------------------------------------------------------------------
# statistics: the document is the independent unit
def doc_bootstrap(per_doc_values, n_boot: int = 10000, seed: int = 20260923, lo: float = 2.5, hi: float = 97.5) -> dict:
    """`per_doc_values` = list (one entry per document) of lists of per-round values. The statistic is the mean over
    all rounds of the resampled documents (= the mean of per-document means when every document has the same number
    of rounds). Resamples documents with replacement, `n_boot` >= 10,000, fixed seed, percentile interval."""
    docs = [np.asarray(v, dtype=np.float64) for v in per_doc_values if len(v)]
    n = len(docs)
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_docs=0, n_boot=int(n_boot), seed=int(seed),
                    per_doc_mean=[])
    sums = np.array([d.sum() for d in docs])
    cnts = np.array([d.size for d in docs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    draws = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return dict(mean=float(sums.sum() / cnts.sum()), lo=float(np.percentile(draws, lo)), hi=float(np.percentile(draws, hi)),
                n_docs=n, n_boot=int(n_boot), seed=int(seed), per_doc_mean=[float(d.mean()) for d in docs])


def sign_counts(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return dict(positive=int((v > 0).sum()), zero=int((v == 0).sum()), negative=int((v < 0).sum()))


# ---------------------------------------------------------------------------------------------------------------
# raw export
def pad_ids(ids, width: int, fill: int = -1, dtype=np.int16) -> np.ndarray:
    a = np.full(width, fill, dtype=dtype)
    ids = list(ids)
    if len(ids) > width:
        raise ValueError("%d ids do not fit width %d" % (len(ids), width))
    if ids:
        a[:len(ids)] = np.asarray(ids, dtype=dtype)
    return a


def key_grid(by_key, n_layers: int, n_heads: int, n_req: int, width: int, fill: int = -1, dtype=np.int16) -> np.ndarray:
    """{(layer, h, b): list} -> (n_layers, n_heads, n_req, width), `fill` where absent."""
    out = np.full((n_layers, n_heads, n_req, width), fill, dtype=dtype)
    for (l, h, b), v in by_key.items():
        out[l, h, b] = pad_ids(v, width, fill, dtype)
    return out


def write_export(path: str, arrays: dict) -> dict:
    """Compressed NPZ; returns {name: (dtype, shape)} for the manifest."""
    np.savez_compressed(path, **arrays)
    return {k: dict(dtype=str(np.asarray(v).dtype), shape=list(np.asarray(v).shape)) for k, v in arrays.items()}


def verify_roundtrip(path: str, arrays: dict) -> list:
    """Reload and compare exactly (dtype, shape, values). Returns the list of mismatching names (empty = exact)."""
    bad = []
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(arrays):
            return sorted(set(z.files) ^ set(arrays))
        for k, v in arrays.items():
            v = np.asarray(v)
            r = z[k]
            # NaN is the float padding ('arm not run'), so NaN == NaN counts as equal for float arrays only
            if r.dtype != v.dtype or r.shape != v.shape or not np.array_equal(r, v, equal_nan=(v.dtype.kind == "f")):
                bad.append(k)
    return bad


def write_manifest(path: str, meta: dict):
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def export_size_mb(paths) -> float:
    return float(sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 1e6)
