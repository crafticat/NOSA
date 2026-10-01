"""GOLDEN-STATE GATE, DECODE GUARD and the post-prefill RESTART of the CPU-packing CONFIRMATION run (Codex's review of job
2179683, relayed by the user 2026-10-01). Pure torch and device-agnostic: the same code runs on the GPU engines and on the
fake CPU engines of retroinfer-eval tests/test_cpupack_confirm.py.

WHY. In job 2179683 the CORRECT stage's logits negative control decoded a second time WITHOUT a restore
(feature/nosi-cpupack-b @ 0051162, cpupack_transport.py:961). That decode's gathers overwrote live window KV and wrote the
tail; the CounterSnapshot restores only _block_map / _new_block_map_buf / _load_mask / _cache_lens, the engine scalars and
the layer tables (nosi/nosi/state_snapshot.py:248-249), never _k_gpu / _v_gpu / _kv_bias_gpu. gated() then timed the step
and ADVANCED from that state (cpupack_transport.py:783-784), so every later step descended from it, and each later step's
checks compared against a reference regenerated from the descendant state (not the intended clean trajectory).

THE REPAIR.
 1. NO DESTRUCTIVE OPERATION IN A MEASUREMENT PROCESS. DecodeGuard: inside a gated step a decode is legal only as the
    step's flushed reference (begin_step arms it, after take + flush) or right after a restore (on_restore arms it). Every
    decode consumes the arm, so a second decode without a restore raises UnsanctionedDecode BEFORE the model runs (state
    untouched). destructive() is refused outside the CONTROLS role: the destructive controls run in a separate process,
    first, and their failure stops the job before any measurement.
 2. GOLDEN TRAJECTORY. The same gated sequence -- flushed reference decode, post-step maps, the pre-timing resident check,
    the advance decode -- run with no transports and no controls, from the same start. Per gated step it records sha256 of
    the reference logits and of the advance logits and the STATE DIGEST after the reference decode (FAMILIES_PRE) and after
    the advance (FAMILIES_POST). The measured pass must match it at every gated step BEFORE any timing (and pass the
    pre-timing resident check: restore -> decode, 0 loads, logits torch.equal the reference), and must match the advance
    after it. A mismatch is GATE_FAIL: the step's timing is skipped (pre) or its rows are excluded (post), and it is counted.
 3. RESTART PROOF (option (i): golden and measured pass in ONE process). PostPrefillSnapshot puts back the counters, the
    layer tables, the host window and the 64 tail-slot rows (state_snapshot.py:166-212 with :248-249, :267-299). The
    non-tail window is NOT restored, but every non-tail map entry is -1 after the restore, so the first decode diffs a
    flushed map and cold-fills EVERY non-tail slot with full 64-row block gathers (cache_engine.py:666 diff_offload,
    :684-685 flash_h2d_from_mask / _bias; the capture invariant I3 = 0 shows step 0 loads all 63 non-tail slots of every
    stream). start digest (FAMILIES_START) after every restore must equal the digest taken right after prefill, and the
    warm natural steps must reproduce the first pass's logits and FULL state digests (window included) bit for bit.
    Fail-closed: any mismatch is exit 20 = the registered fallback (ii): a separate golden process, then a measurement
    process gated against its export.

STATE DIGEST (read-only; per layer; nothing in it depends on the HOST cache layout, so it is layout-independent across the
LAYOUT stage's in-place host conversion; the GPU window layout is never changed):
  map      sha256 of _block_map (all 64 slots incl. the tail id)
  newmap   sha256 of _new_block_map_buf                    (start digest only)
  mask     sha256 of _load_mask (after the advance: all -1 = the advance loaded nothing)
  cnt      sha256 of _cache_lens and (seq_length, _tail_block_len_on_gpu, _tail_block_idx_on_gpu)
  tail     sha256 of the VALID tail rows [t0, t0 + tail_len) of _k_gpu, _v_gpu, _kv_bias_gpu (the rows the kernel reads)
  tailslot sha256 of all 64 tail-slot rows                  (start digest only: the rows PostPrefillSnapshot restores)
  comp     sha256 of the small layer tables (cached_compressed_cu_seqlens, no_compress_k_cache, compressed_cis, tail_cis)
           and the layer scalars; fp64 of compress_k_cache_varlen and total_cis (whole tensors: both snapshots restore whole
           tensors, so even never-written tails are bit-identical across passes)
  win      fp64 of the attended window rows [0, (tail_idx + 1) * 64) of _k_gpu, _v_gpu, _kv_bias_gpu (every non-tail slot
           AND the whole tail slot: a write past the valid tail rows is flagged too; flash_attn_nosa reads rows
           0.._cache_lens-1 as one flat sequence, cache_engine.py:37-40)
  host     sha256 of the host window rows [seq - 128, seq + 128) through the engine's logical views (start digest only)
  top      the cache's _seen_tokens
fp64 = sum_i w_i * (2 (i + off) + 1) mod 2^64 over the tensor's 64-bit words (16-bit words when the byte count is not a
multiple of 8), plus the plain word sum and the word count, computed ON THE DEVICE in chunks: sha256 of 5.5 GB of
compressed keys per step on the host is not affordable at B336. Any single-word change changes it with certainty (an odd
weight is invertible mod 2^64); integer sums are exact and order-independent, so it is deterministic. It is NOT a
cryptographic hash; the per-step logits, maps, counters and tail rows are sha256 as registered.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import torch

MASK64 = (1 << 64) - 1
FAMILIES_PRE = ("map", "mask", "cnt", "tail", "comp", "win", "top")
FAMILIES_POST = ("map", "mask", "cnt", "tail", "comp", "top")
FAMILIES_START = ("map", "newmap", "mask", "cnt", "tailslot", "comp", "host", "top")
LAYER_SMALL = ("cached_compressed_cu_seqlens", "no_compress_k_cache", "compressed_cis", "tail_cis")
LAYER_BIG = ("compress_k_cache_varlen", "total_cis")
LAYER_SCALARS = ("cached_compressed_max_seqlen", "no_compress_k_len", "comp_cis_len", "tail_cis_len", "cis_len", "seq_length")
HOST_MARGIN_ROWS = 128                 # PostPrefillSnapshot's host window: [seq - 2 blocks, seq + 2 blocks)
FP_CHUNK_WORDS = 1 << 22               # 32 MiB of int64 per device temporary


# --------------------------------------------------------------------------------------------------------- the guard
class UnsanctionedDecode(RuntimeError):
    """A decode inside a gated step of a MEASUREMENT process that is neither the step's flushed reference nor preceded by
    a restore, or a destructive control outside the CONTROLS process. Raised before the model runs."""


class DecodeGuard:
    """See the module docstring (repair 1). Outside a gated step natural decodes are the trajectory itself (capture, warm
    steps, the natural steps between gated steps) and are not restricted."""

    ROLES = ("measure", "controls")

    def __init__(self, role: str = "measure"):
        if role not in self.ROLES:
            raise ValueError("guard role %r" % role)
        self.role = role
        self.in_step = None
        self.armed = None
        self.destructive_name = None
        self.decodes = 0
        self.log: List[Dict] = []

    def begin_step(self, it) -> None:
        if self.in_step is not None:
            raise UnsanctionedDecode("gated step %s begun inside gated step %s" % (it, self.in_step))
        self.in_step, self.armed = it, "reference"

    def on_restore(self) -> None:
        if self.in_step is not None:
            self.armed = "restored"

    def end_step(self) -> None:
        self.in_step, self.armed = None, None

    def check(self, what: str = "decode") -> None:
        """Called BEFORE every decode; consumes the arm."""
        if self.in_step is not None:
            if self.armed is None:
                if self.destructive_name is None:
                    raise UnsanctionedDecode(
                        "%s at gated step %s without a preceding restore: a decode that is neither the flushed reference nor a "
                        "restored step overwrites live window KV / tail state that the CounterSnapshot does not restore "
                        "(the job-2179683 defect); destructive controls run only in the CONTROLS process" % (what, self.in_step))
                self.log.append(dict(step=self.in_step, op=what, control=self.destructive_name, sanctioned="controls role"))
            self.armed = None
        self.decodes += 1

    @contextmanager
    def destructive(self, name: str):
        """The only way to run a destructive operation; refused (raises) in a measurement process."""
        if self.role != "controls":
            raise UnsanctionedDecode("destructive control %r refused in a %s process: destructive controls run only in the separate "
                                     "CONTROLS process" % (name, self.role))
        prev = self.destructive_name
        self.destructive_name = name
        self.log.append(dict(step=self.in_step, op="destructive", control=name))
        try:
            yield
        finally:
            self.destructive_name = prev


# ------------------------------------------------------------------------------------------------------- hashing
def _host_bytes(t: torch.Tensor):
    x = t.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    x = x.contiguous().reshape(-1)
    return x.view(torch.uint8).numpy()


def sha_parts(parts: Iterable) -> str:
    """sha256 over tensors (dtype, shape and raw bytes) and python values (repr), in order."""
    h = hashlib.sha256()
    for p in parts:
        if torch.is_tensor(p):
            h.update(("T%s%s|" % (str(p.dtype), tuple(p.shape))).encode())
            if p.numel():
                h.update(memoryview(_host_bytes(p)))
        else:
            h.update(("V%r|" % (p,)).encode())
    return h.hexdigest()


_ARANGE: Dict[str, torch.Tensor] = {}


def _arange(n: int, device) -> torch.Tensor:
    key = str(device)
    a = _ARANGE.get(key)
    if a is None or a.numel() < n:
        a = torch.arange(max(int(n), 1), dtype=torch.int64, device=device)
        _ARANGE[key] = a
    return a[:n]


def _words(t: torch.Tensor) -> torch.Tensor:
    x = t.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    x = x.reshape(-1)
    if x.numel() == 0:
        return x.view(torch.uint8)
    nbytes = x.numel() * x.element_size()
    u8 = x.view(torch.uint8)
    for size, dt in ((8, torch.int64), (2, torch.int16)):
        if nbytes % size == 0:
            try:
                return u8.view(dt)
            except RuntimeError:                         # a storage offset not aligned to the wider word
                continue
    return u8


def fp_stream(parts: Iterable[torch.Tensor], chunk_words: int = FP_CHUNK_WORDS) -> str:
    """fp64 (module docstring) of the concatenated words of `parts`, on their device."""
    acc, off, dev = None, 0, None
    for p in parts:
        w = _words(p)
        n = w.numel()
        if acc is None or w.device != dev:
            if acc is not None:
                raise ValueError("fp_stream parts on different devices")
            dev = w.device
            acc = torch.zeros(2, dtype=torch.int64, device=dev)
        for a in range(0, n, chunk_words):
            k = min(chunk_words, n - a)
            c = w[a:a + k]
            if c.dtype != torch.int64:
                c = c.to(torch.int64)
            s1 = c.sum()
            s2 = (c * _arange(k, dev)).sum()
            acc[0] += s1 * (2 * (off + a) + 1) + 2 * s2
            acc[1] += s1
        off += n
    v = [0, 0] if acc is None else [int(x) & MASK64 for x in acc.tolist()]
    return "%x:%016x%016x" % (off, v[0], v[1])


def fp_t(t: torch.Tensor) -> str:
    return fp_stream([t])


def fp_rows(t: torch.Tensor, rows: int, batch_chunk: int = 8) -> str:
    """fp64 of t[:, :rows] (dim 1 = rows) without a full contiguous copy."""
    if rows >= t.shape[1]:
        return fp_t(t)
    return fp_stream(t[b0:b0 + batch_chunk, :rows] for b0 in range(0, t.shape[0], batch_chunk))


# --------------------------------------------------------------------------------------------------- state digests
def comp_digest(lay) -> str:
    small = sha_parts([getattr(lay, n, None) for n in LAYER_SMALL] + [getattr(lay, n, None) for n in LAYER_SCALARS])
    big = ",".join(fp_t(v) if torch.is_tensor(v) else "V%r" % (v,) for v in (getattr(lay, n, None) for n in LAYER_BIG))
    return small + "|" + big


def host_window(e, margin: int = HOST_MARGIN_ROWS):
    n = int(e._k_cpu.shape[1])
    return max(0, int(e.seq_length) - margin), min(n, int(e.seq_length) + margin)


def layer_digest(lay, families: Sequence[str], R: int = 64) -> Dict[str, str]:
    e = lay.cache_engine
    t_idx, tl = int(e._tail_block_idx_on_gpu), int(e._tail_block_len_on_gpu)
    t0 = t_idx * R
    out = {}
    for f in families:
        if f == "map":
            out[f] = sha_parts([e._block_map])
        elif f == "newmap":
            out[f] = sha_parts([e._new_block_map_buf])
        elif f == "mask":
            out[f] = sha_parts([e._load_mask])
        elif f == "cnt":
            out[f] = sha_parts([e._cache_lens, int(e.seq_length), tl, t_idx])
        elif f == "tail":
            out[f] = sha_parts([e._k_gpu[:, t0:t0 + tl], e._v_gpu[:, t0:t0 + tl], e._kv_bias_gpu[:, t0:t0 + tl]])
        elif f == "tailslot":
            out[f] = sha_parts([e._k_gpu[:, t0:t0 + R], e._v_gpu[:, t0:t0 + R], e._kv_bias_gpu[:, t0:t0 + R]])
        elif f == "comp":
            out[f] = comp_digest(lay)
        elif f == "win":
            rows = (t_idx + 1) * R
            out[f] = ",".join(fp_rows(x, rows) for x in (e._k_gpu, e._v_gpu, e._kv_bias_gpu))
        elif f == "host":
            lo, hi = host_window(e)
            out[f] = sha_parts([e._k_cpu[:, lo:hi], e._v_cpu[:, lo:hi], lo, hi])
        elif f != "top":
            raise ValueError("digest family %r" % f)
    return out


def state_digest(cache, families: Sequence[str], R: int = 64) -> Dict:
    """{family: [per-layer hex]} plus 'top' (module docstring). Read-only."""
    per = [layer_digest(lay, families, R) for lay in cache.layers]
    out = {f: [p[f] for p in per] for f in families if f != "top"}
    if "top" in families:
        out["top"] = "V%r" % (getattr(cache, "_seen_tokens", None),)
    return out


def compare(golden: Optional[Dict], got: Optional[Dict], families: Optional[Sequence[str]] = None) -> List[str]:
    """'family[layer]' (or 'family') entries that differ. A family or layer missing on either side is a mismatch, never a
    pass; an absent golden record is a mismatch."""
    if not golden:
        return ["<no golden record>"]
    got = got or {}
    fams = list(families) if families is not None else sorted(set(golden) | set(got))
    bad = []
    for f in fams:
        a, b = golden.get(f), got.get(f)
        if a is None or b is None:
            bad.append("%s<missing>" % f)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                bad.append("%s<len %d != %d>" % (f, len(a), len(b)))
            bad += ["%s[%d]" % (f, i) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        elif a != b:
            bad.append(f)
    return bad


# ------------------------------------------------------------------------------------------- the hosted restart snapshot
def snapshot_offload(snap) -> int:
    """Move every device tensor a TAKEN snapshot holds to pageable host memory (the post-prefill snapshot must not hold
    ~7.5 GB of HBM at B336 while the golden / measured passes run). Returns the bytes moved."""
    moved, nbytes = [], 0
    for i, slot in enumerate(snap.layers):
        for part in ("engine", "layer"):
            d = slot[part]
            for k, v in list(d.items()):
                if torch.is_tensor(v) and v.device.type != "cpu":
                    d[k] = v.to("cpu")
                    moved.append((i, part, k, str(v.device)))
                    nbytes += v.numel() * v.element_size()
    snap._hosted = getattr(snap, "_hosted", []) + moved
    return nbytes


def snapshot_restore_hosted(snap):
    """Upload the hosted tensors to their devices, run the snapshot's OWN restore (state_snapshot.py), then drop the
    uploads (restore copies into the live tensors, or rebinds to a CLONE when a shape changed: nothing aliases them)."""
    hosted = list(getattr(snap, "_hosted", []))
    keep = {}
    for i, part, k, dev in hosted:
        d = snap.layers[i][part]
        keep[(i, part, k)] = d[k]
        d[k] = d[k].to(dev)
    try:
        snap.restore()
    finally:
        for (i, part, k), v in keep.items():
            snap.layers[i][part][k] = v
    return snap


# ------------------------------------------------------------------------------------------------ golden records
def golden_export(path: str, golden: Dict) -> str:
    g = dict(golden, steps={str(k): v for k, v in golden.get("steps", {}).items()})
    with open(path + ".tmp", "w") as f:
        json.dump(g, f)
    os.replace(path + ".tmp", path)
    return path


def golden_load(path: str) -> Dict:
    with open(path) as f:
        g = json.load(f)
    g["steps"] = {int(k): v for k, v in (g.get("steps") or {}).items()}
    return g


def golden_cross(a: Dict, b: Dict, steps: Optional[Sequence[int]] = None) -> Dict:
    """Two golden records of the SAME inputs from different processes, compared on their common gated steps (a
    determinism record; not a gate)."""
    common = sorted(set(a.get("steps", {})) & set(b.get("steps", {})))
    if steps is not None:
        common = [s for s in common if s in set(steps)]
    out = dict(steps=common, mismatches={})
    for s in common:
        x, y = a["steps"][s], b["steps"][s]
        bad = []
        for k in ("ref_sha", "adv_sha"):
            if x.get(k) != y.get(k):
                bad.append(k)
        bad += ["pre:" + m for m in compare(x.get("pre"), y.get("pre"))]
        bad += ["post:" + m for m in compare(x.get("post"), y.get("post"))]
        if bad:
            out["mismatches"][s] = bad[:32]
    out["equal"] = bool(common) and not out["mismatches"]
    return out
