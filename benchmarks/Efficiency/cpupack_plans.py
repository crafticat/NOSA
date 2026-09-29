"""Natural-plan capture helpers for the CPU-packing transport: the int16 AloneTrace, integrity invariants, canonical hashes,
export / import, cells and selection agreement. Pure torch / numpy, CPU-tested (retroinfer-eval
tests/test_cpupack_transport.py).

THE PLAN (verified at dc818e4): per decode step and layer, NOSI's load mask (H, B, 64) int64 written by diff_offload
(cache_engine.py:666), consumed by the shipped gathers (:684-685) and archived by transfer_trace.record_mask (:700 ->
transfer_trace.py:201-209) AFTER the gathers and BEFORE the rollover section. The capture is a separate NATURAL pass from the
post-prefill state (steps 0 .. N_cap-1, no flush / reference / resident step in between), with ONE harvest at the end, so
nothing resets the archive (verify_alone.harvest_into :428-442 would call new_document after every call; worker_sweep's
gated steps are flushed-reference and resident steps, not natural ones).

INVARIANTS (count violations; a capture is accepted only when all are 0):
  I1 no -2 (unwritten) entry in the archive          I2 the tail slot is never loaded
  I3 step 0 (the cold fill after prefill) loads every non-tail slot of every stream
  I4 maps[s][mask >= 0] == mask                      I5 maps[s][mask < 0] == maps[s-1][mask < 0]
  I6 a loaded block is not in the previous map row   I7 every map row holds distinct ids and the tail id is L // 64
  I8 per-step load count == the engines' own count   I9 0 <= block < L // 64 (history only; also int16-safe)
HASHES: sha256 of the C-order little-endian int16 bytes of masks[s, l] and maps[s, l] (separately); a step digest over its
layer hashes; a run digest over the step digests plus the canonical meta.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def make_int16_trace_class(AloneTrace):
    """AloneTrace (verify_alone.py:368-373) with the archives in int16 (block ids <= 379 at S_cpu = 24320): record_mask is
    transfer_trace.py:201-209 with only the archive dtype changed (device-to-device copy with conversion, no sync). A value
    that would not fit is caught by the per-step cross-check against the engines' int64 masks."""

    class Int16AloneTrace(AloneTrace):
        def record_mask(self, load_mask, block_map):
            if self.cur is None or not self.full or self.layer is None:
                return
            if self.mask_archive is None or self.mask_archive.shape[2:] != tuple(load_mask.shape):
                shape = (self.max_steps, self.num_layers) + tuple(load_mask.shape)
                self.mask_archive = torch.full(shape, -2, dtype=torch.int16, device=load_mask.device)
                self.map_archive = torch.full(shape, -2, dtype=torch.int16, device=block_map.device)
            self.mask_archive[self.cur.index, self.layer].copy_(load_mask, non_blocking=True)
            self.map_archive[self.cur.index, self.layer].copy_(block_map, non_blocking=True)

    return Int16AloneTrace


def counts_of(masks: torch.Tensor, tail_slot: int = 63) -> torch.Tensor:
    """(steps, L, H, B) loads per stream (non-tail slots)."""
    return (masks[..., :tail_slot] >= 0).sum(-1).to(torch.int16)


def invariants(masks: torch.Tensor, maps: torch.Tensor, *, L_prompt: int, block_rows: int = 64, tail_slot: int = 63,
               step0_cold: bool = True, step_loaded: Optional[Sequence[int]] = None) -> Dict[str, int]:
    """Violation counts of I1..I9 (0 everywhere = accepted). masks / maps: (steps, L, H, B, M) CPU int tensors."""
    tail_id = L_prompt // block_rows
    out = {}
    out["I1_unwritten"] = int((masks == -2).sum() + (maps == -2).sum())
    out["I2_tail_loaded"] = int((masks[..., tail_slot] >= 0).sum())
    cnt = counts_of(masks, tail_slot)
    out["I3_step0_not_full"] = int((cnt[0] != tail_slot).sum()) if step0_cold else 0
    loaded = masks >= 0
    out["I4_map_ne_loaded"] = int((loaded & (maps != masks)).sum())
    i5 = i6 = 0
    for s in range(1, masks.shape[0]):
        keep = ~loaded[s]
        i5 += int((keep & (maps[s] != maps[s - 1])).sum())
        for l in range(masks.shape[1]):
            ms, prev = masks[s, l], maps[s - 1, l]
            hit = (ms[..., :, None] == prev[..., None, :]).any(-1) & (ms >= 0)
            i6 += int(hit.sum())
    out["I5_kept_slot_changed"] = i5
    out["I6_loaded_was_resident"] = i6
    srt = torch.sort(maps.to(torch.int64), dim=-1).values
    out["I7_duplicate_ids"] = int((srt[..., 1:] == srt[..., :-1]).sum())
    out["I7_tail_id"] = int((maps[..., tail_slot] != tail_id).sum())
    if step_loaded is not None:
        per = cnt.to(torch.int64).sum(dim=(1, 2, 3))
        out["I8_count_mismatch"] = int(sum(int(a) != int(b) for a, b in zip(per.tolist(), step_loaded)))
    else:
        out["I8_count_mismatch"] = -1
    out["I9_block_range"] = int((loaded & ((masks < 0) | (masks >= tail_id))).sum())
    return out


def accepted(inv: Dict[str, int]) -> bool:
    return all(v == 0 for k, v in inv.items() if not (k == "I8_count_mismatch" and v == -1))


def _le16(t: torch.Tensor) -> bytes:
    return np.ascontiguousarray(t.to(torch.int16).cpu().numpy()).astype("<i2", copy=False).tobytes()


def plan_hashes(masks: torch.Tensor, maps: torch.Tensor, meta: Optional[Dict] = None) -> Dict:
    S, L = masks.shape[:2]
    per = []
    steps = []
    for s in range(S):
        row = []
        h = hashlib.sha256()
        for l in range(L):
            a = hashlib.sha256(_le16(masks[s, l])).hexdigest()
            b = hashlib.sha256(_le16(maps[s, l])).hexdigest()
            row.append([a, b])
            h.update((a + b).encode())
        per.append(row)
        steps.append(h.hexdigest())
    r = hashlib.sha256()
    for d in steps:
        r.update(d.encode())
    r.update(json.dumps(meta or {}, sort_keys=True).encode())
    return dict(per_step_layer=per, step_digest=steps, run_digest=r.hexdigest(), dtype="int16 little-endian C-order")


def export_npz(path: str, masks: torch.Tensor, maps: torch.Tensor, meta: Dict, step_loaded: Sequence[int],
               logits_sha: Sequence[str]) -> Dict:
    """Write the plans (int16, [step, layer, kv_head, request, slot], the selection_ids convention) and a JSON sidecar with
    the hashes; returns the hash record."""
    hs = plan_hashes(masks, maps, meta)
    np.savez(path, masks=masks.to(torch.int16).numpy(), maps=maps.to(torch.int16).numpy(),
             counts=counts_of(masks).numpy(), step_loaded=np.asarray(list(step_loaded), dtype=np.int64),
             logits_sha=np.asarray(list(logits_sha)), meta_json=np.asarray(json.dumps(meta, sort_keys=True)))
    side = path[:-4] + ".hashes.json" if path.endswith(".npz") else path + ".hashes.json"
    with open(side, "w") as f:
        json.dump(dict(meta=meta, run_digest=hs["run_digest"], step_digest=hs["step_digest"], per_step_layer=hs["per_step_layer"]), f)
    return hs


def load_npz(path: str) -> Dict:
    """Load an export and re-verify its hashes against the sidecar (a mismatch raises)."""
    z = np.load(path, allow_pickle=False)
    masks = torch.from_numpy(z["masks"].astype(np.int16))
    maps = torch.from_numpy(z["maps"].astype(np.int16))
    meta = json.loads(str(z["meta_json"]))
    hs = plan_hashes(masks, maps, meta)
    side = path[:-4] + ".hashes.json" if path.endswith(".npz") else path + ".hashes.json"
    ref = json.load(open(side))
    if hs["run_digest"] != ref["run_digest"]:
        raise ValueError("plan export %s: run digest %s != sidecar %s" % (path, hs["run_digest"], ref["run_digest"]))
    return dict(masks=masks, maps=maps, meta=meta, step_loaded=z["step_loaded"].tolist(), logits_sha=[str(x) for x in z["logits_sha"]],
                hashes=hs)


def layer_totals(masks: torch.Tensor, steps: Sequence[int], tail_slot: int = 63) -> Dict:
    """Per (step, layer) loads (both heads, all requests)."""
    c = counts_of(masks, tail_slot).to(torch.int64).sum(dim=(2, 3))
    return {(s, l): int(c[s, l]) for s in steps for l in range(c.shape[1])}


def tercile_cells(totals: Dict, steady_steps: Sequence[int]) -> Dict:
    """Cut points of the per-(step, layer) load totals over the steady steps (registered cells: low / mid / high)."""
    xs = sorted(v for (s, l), v in totals.items() if s in set(steady_steps))
    if not xs:
        return dict(lo=None, hi=None)
    q = lambda p: float(np.percentile(np.asarray(xs, dtype=np.float64), p))
    return dict(lo=q(100 / 3), hi=q(200 / 3), n=len(xs), min=xs[0], max=xs[-1])


def cell_of(total: int, cuts: Dict) -> str:
    if cuts.get("lo") is None:
        return "?"
    return "low" if total <= cuts["lo"] else ("high" if total > cuts["hi"] else "mid")


def selection_agreement(map_a: torch.Tensor, map_b: torch.Tensor) -> Dict:
    """Per stream (last dim = slots), set agreement of two maps regardless of slot order: same fraction, mean / min Jaccard."""
    a = map_a.to(torch.int64)
    b = map_b.to(torch.int64)
    inter = (a[..., :, None] == b[..., None, :]).any(-1).sum(-1).to(torch.float64)
    n = a.shape[-1]
    jac = inter / (2 * n - inter)
    return dict(same_frac=float((inter == n).to(torch.float64).mean()), jaccard_mean=float(jac.mean()), jaccard_min=float(jac.min()))
