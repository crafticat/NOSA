"""Paper section 4.5, "Does joint execution reduce GPU-memory traffic?": the PURE part of the harness (no CUDA, no
engine import; every function takes torch tensors on any device). The GPU driver is joint_hbm.py; the CPU tests are
tests_cpu/test_joint_hbm.py.

THE WORK (one K-step block, per request b; K = 3):
  verifier rows o = 0 .. K-1 at positions L + W + o (inputs: the committed tokens), and ONE draft row o = K at position
  L + W + K. A row reads, per KV head h:
    - its historical blocks (63 ids < T, the shipped selection at step W + o minus the tail block T),
    - the committed rows [0, t0) of the tail block T (t0 = the tail length at the block start),
    - its provisional rows: a verifier row o reads the EXACT rows t0 .. t0 + o (version x: written in this forward by
      the verifier rows 0 .. o); the draft row reads the DRAFT rows t0 .. t0 + K - 1 (version d: written by earlier
      draft steps) and its own row t0 + K (version d, written in this forward).
  A read unit is (request, head, block, version). Versions: c (committed: the historical blocks and the committed tail
  rows; immutable), x (exact provisional rows), d (draft provisional rows). x and d rows of the same position are
  different units: they are never shared.

THE FORMS (the same rows, the same selections, the same arithmetic per row):
  sep_private    two calls: draft (B rows), verify (K B rows). Every row has its OWN 64-slot window (the engine's
                 window at its step: slots 0 .. 62 in the engine's slot order, slot 63 = the tail). KV read = SUM.
  joint_private  one call of (K + 1) B rows over the same private windows: the weights are read once; KV = SUM.
  sep_shared     two calls; in each call the rows of a request share ONE window (the union of their selections,
                 ascending block id, then the tail slot) and a per-row bias masks the slots a row did not select.
  joint_shared   one call; one window per request: the union of all K + 1 rows, then an x-tail slot (committed rows
                 + x rows; extent-limited per verifier row) and a d-tail slot (committed rows + d rows + the draft's
                 own row; the x-tail slot is masked for the draft row).
  Masked slots carry the bias MASK (finite, -3e4: the paired tick's G7 value) and finite K / V (real blocks).

ACCOUNTING (exact, per layer; account_layer): tokens per version for three levels:
  sum            every row reads its own units (no sharing);
  union_sep      the union within each call of the separate forms (draft call, verify call);
  union_request  the union over all K + 1 rows of a request (the joint ideal).
LAYOUT METRICS (layout_metrics): per call, the attended tokens (sum over rows of the extent: the kernel loads every
row below a row's extent; worst case without reuse; the masked part is the extra arithmetic), the window tokens (sum
over entries of the largest extent: the best case when every byte is read once) and the useful tokens.

PARSERS: ncu raw CSV (ncu_rows, ncu_windows: windows are runs of spin_kernel markers; the NVTX tag jh|form|call|part
of each kernel), the ncu pass lines (ncu_passes), a chrome trace of torch.profiler (trace_busy: kernel count, busy,
span, idle). summarize joins the unprofiled result and the profiled windows.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Dict, List, Sequence, Tuple

import torch

BS = 64                                   # tokens per block (nosa_llama.Llama asserts 64)
SLOTS = 64                                # the engine's window: 63 historical slots + the tail slot (topk = 64)
MASK = -30000.0                           # finite mask bias (bf16 rounds it to -29952; exp underflows to 0 after the scale)
HEAD_DIM = 128
KV_BYTES = 2 * HEAD_DIM * 2               # K + V of one token row of one KV head, bf16: 512 B
BIAS_BYTES = 2                            # one bf16 bias value per token row per KV head
VERSIONS = ("c", "x", "d")
FORMS = ("sep_private", "joint_private", "sep_shared", "joint_shared")
FORM_CALLS = {                            # (call name, row kinds, layout)
    "sep_private": (("draft", "d", "private"), ("verify", "v", "private")),
    "joint_private": (("joint", "vd", "private"),),
    "sep_shared": (("draft", "d", "shared"), ("verify", "v", "shared")),
    "joint_shared": (("joint", "vd", "shared"),),
}
GROUPS = (("private", ("sep_private", "joint_private")), ("shared", ("sep_shared", "joint_shared")))
PARTS = ("embed", "norm_qkv", "rope", "score", "kvwrite", "attn", "wo_ffn", "head")
LIMITS = (
    "draft = target stand-in: the draft row is the target model at position L + W + K with the SHIPPED selection of "
    "that step and full acceptance; its d-version rows hold the shipped decode's own K / V of those positions "
    "(computed at batch B); draft-specific selection drift and draft-only masking are NOT modelled",
    "selections are CAPTURED from the shipped greedy decode at steps W .. W + K (W = 20 by default: no 16-token "
    "compress event and no 64-token tail fill inside the block); every form attends exactly these selections; the "
    "scoring is recomputed in every form for its traffic and compared with the capture (report only)",
    "all selected blocks are resident in HBM in every form: the question is HBM traffic, not PCIe",
    "the committed tail rows are stored twice in a joint_shared window (x-tail and d-tail slots): a layout cost, "
    "counted in the layout metrics, not in the logical union",
)


def kind(o: int, K: int) -> str:
    return "d" if o == K else "v"


def call_rows(B: int, K: int, kinds: str) -> List[Tuple[int, int]]:
    """The rows of a call, request-major, offset ascending (a request's rows are contiguous: cu_seqlens_q)."""
    return [(b, o) for b in range(B) for o in range(K + 1) if kind(o, K) in kinds]


def rid(b: int, o: int, K: int) -> int:
    """The private window entry of row (b, o)."""
    return b * (K + 1) + o


def check_block(L: int, W: int, K: int) -> Tuple[int, int]:
    """(T, t0) of a block starting W decode steps after a prefill of L tokens. Refuses a block that a tail fill reaches
    (the historical blocks must all be the prefill's, on the host, and the tail slot must hold rows t0 .. t0 + K)."""
    if L % BS + W + K + 1 > BS:
        raise ValueError("block (L %d, W %d, K %d) crosses a tail fill: L %% %d + W + K + 1 = %d > %d"
                         % (L, W, K, BS, L % BS + W + K + 1, BS))
    return (L + W) // BS, (L + W) % BS


# ------------------------------------------------------------------ read units and accounting
def row_units(ids: Sequence[int], o: int, K: int, t0: int, T: int) -> set:
    u = {("h", int(x)) for x in ids}
    if t0 > 0:
        u.add(("c", T))
    if o < K:
        u.update(("x", j) for j in range(o + 1))
    else:
        u.update(("d", j) for j in range(K + 1))
    return u


def unit_tokens(u, t0: int) -> int:
    return BS if u[0] == "h" else (t0 if u[0] == "c" else 1)


def tally(units, t0: int) -> Dict[str, int]:
    out = dict(c=0, x=0, d=0, hist_blocks=0)
    for u in units:
        if u[0] == "h":
            out["c"] += BS
            out["hist_blocks"] += 1
        elif u[0] == "c":
            out["c"] += t0
        else:
            out[u[0]] += 1
    return out


def _add(a: Dict[str, int], b: Dict[str, int]) -> None:
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def account_layer(hist: torch.Tensor, K: int, t0: int, T: int) -> Dict[str, Dict[str, int]]:
    """hist (B, K + 1, H, n) int: the historical block ids of row (b, o), head h. Returns, per level (sum, union_sep,
    union_request), the tokens per version (c, x, d) and the historical block count, summed over (b, h)."""
    h = hist.detach().to("cpu")
    B, K1, H, _ = h.shape
    assert K1 == K + 1
    out = {lvl: dict(c=0, x=0, d=0, hist_blocks=0) for lvl in ("sum", "union_sep", "union_request")}
    for b in range(B):
        for hh in range(H):
            units = [row_units(h[b, o, hh].tolist(), o, K, t0, T) for o in range(K + 1)]
            for u in units:
                _add(out["sum"], tally(u, t0))
            _add(out["union_sep"], tally(set().union(*units[:K]), t0))
            _add(out["union_sep"], tally(units[K], t0))
            _add(out["union_request"], tally(set().union(*units), t0))
    return out


def total_tokens(t: Dict[str, int]) -> int:
    return sum(t[v] for v in VERSIONS)


# ------------------------------------------------------------------ layouts
def private_tok(block_maps: torch.Tensor) -> torch.Tensor:
    """block_maps (K + 1, H, B, 64) int64: the engine's _block_map after the update of step W + o (slot 63 = T).
    Returns tok (E, 64 BS, H), E = B (K + 1), entry rid(b, o): the token index (into the per-request token store) of
    every window row: slot m holds block block_maps[o, h, b, m]."""
    K1, H, B, S = block_maps.shape
    assert S == SLOTS, S
    blk = block_maps.permute(2, 0, 3, 1).reshape(B * K1, S, H)
    i = torch.arange(BS, device=block_maps.device, dtype=block_maps.dtype)
    return (blk.unsqueeze(2) * BS + i.view(1, 1, BS, 1)).reshape(B * K1, S * BS, H)


def private_call(rows: Sequence[Tuple[int, int]], K: int, t0: int) -> Dict:
    """The private layout of a call: entry (cache_batch_idx), extent (cache_seqlens) and the write plan (layer
    independent). Window rows per entry: S_e = 64 BS; the tail slot is 63. Bias storage: (E, S_e, H), the call's rows
    at [0, R) in call order."""
    S_e = SLOTS * BS
    tail = (SLOTS - 1) * BS
    pos = {r: p for p, r in enumerate(rows)}
    entry = [rid(b, o, K) for b, o in rows]
    extent = [tail + t0 + o + 1 for _, o in rows]
    kv_src, kv_dst, b_src, b_dst = [], [], [], []
    for p, (b, o) in enumerate(rows):
        readers = [(b, K)] if o == K else [(b, oo) for oo in range(o, K) if (b, oo) in pos]
        for r in readers:
            kv_src.append(p)
            kv_dst.append(rid(r[0], r[1], K) * S_e + tail + t0 + o)
            b_src.append(p)
            b_dst.append(pos[r] * S_e + tail + t0 + o)
    return dict(layout="private", rows=list(rows), S_e=S_e, entry=entry, extent=extent, kv_src=kv_src, kv_dst=kv_dst,
                b_src=b_src, b_dst=b_dst)


def shared_call(hist: torch.Tensor, rows: Sequence[Tuple[int, int]], K: int, t0: int, T: int) -> Dict:
    """The shared layout of a call at one layer. hist (B, K + 1, H, n) int64 (the rows' historical ids). rows: the
    call's rows, request-major, the same offsets for every request. Returns the window token index tok (B, S_e, H),
    the visibility (R, S_e, H) bool (False -> MASK), entry, extent, the write plan, n_hist (the window's historical
    slots: the largest union over (b, h); a shorter union is padded with unselected real blocks, masked for every row)
    and n_union (B, H)."""
    dev = hist.device
    B, K1, H, _ = hist.shape
    offs = sorted({o for _, o in rows})
    U = len(offs)
    if list(rows) != [(b, o) for b in range(B) for o in offs]:
        raise ValueError("shared_call: rows must be request-major with the same offsets for every request")
    has_v = int(any(o < K for o in offs))
    has_d = int(K in offs)
    R = B * U
    rb = torch.arange(B, device=dev).repeat_interleave(U)
    ro = torch.tensor(offs, device=dev).repeat(B)
    rh = hist[rb, ro]                                                            # (R, H, n)
    if rh.numel() and (int(rh.min()) < 0 or int(rh.max()) >= T):
        raise ValueError("shared_call: a historical id outside [0, T = %d)" % T)
    member_row = torch.zeros((R, H, T), dtype=torch.bool, device=dev)
    member_row.scatter_(2, rh, True)
    member_req = member_row.view(B, U, H, T).any(1)                              # (B, H, T)
    n_union = member_req.sum(-1)                                                 # (B, H)
    n_hist = int(n_union.max())
    key = (~member_req).to(torch.int32)
    order = torch.sort(key, dim=-1, stable=True).indices[..., :n_hist]          # members first, ascending id
    tails = (["x"] if has_v else []) + (["d"] if has_d else [])
    n_slots = n_hist + len(tails)
    pad = n_slots % 2                     # the allocation is a multiple of 128 rows: the kernel's kBlockN (head dim 128)
    S_e = (n_slots + pad) * BS            # reads bias columns up to the 128-row tile end; the pad slot is past every
    i = torch.arange(BS, device=dev)      # extent (masked, finite tail rows)
    tok_hist = (order.unsqueeze(-1) * BS + i).permute(0, 2, 3, 1).reshape(B, n_hist * BS, H)
    tok_tail = (T * BS + i).view(1, BS, 1).expand(B, BS, H)
    tok = torch.cat([tok_hist] + [tok_tail] * (len(tails) + pad), dim=1)
    valid = torch.arange(n_hist, device=dev).view(1, 1, n_hist) < n_union.unsqueeze(-1)       # (B, H, n_hist)
    sel = torch.gather(member_row, 2, order[rb]) & valid[rb]                     # (R, H, n_hist)
    vis = [sel.permute(0, 2, 1).unsqueeze(2).expand(R, n_hist, BS, H).reshape(R, n_hist * BS, H)]
    lim_x = torch.where(ro < K, t0 + ro + 1, torch.zeros_like(ro))               # x-tail rows a row sees
    lim_d = torch.where(ro == K, torch.full_like(ro, t0 + K + 1), torch.zeros_like(ro))
    for t in tails:
        lim = lim_x if t == "x" else lim_d
        vis.append((i.view(1, BS) < lim.view(R, 1)).unsqueeze(-1).expand(R, BS, H))
    if pad:
        vis.append(torch.zeros((R, BS, H), dtype=torch.bool, device=dev))
    visible = torch.cat(vis, dim=1)
    pos = {r: p for p, r in enumerate(rows)}
    entry = [b for b, _ in rows]
    extent, kv_src, kv_dst, b_src, b_dst = [], [], [], [], []
    for p, (b, o) in enumerate(rows):
        if o < K:
            base = n_hist * BS + t0 + o
            extent.append(base + 1)
            kv_src.append(p)
            kv_dst.append(b * S_e + base)
            for oo in range(o, K):
                if (b, oo) in pos:
                    b_src.append(p)
                    b_dst.append(pos[(b, oo)] * S_e + base)
        else:
            base = (n_hist + has_v) * BS + t0 + K
            extent.append(base + 1)
            kv_src.append(p)
            kv_dst.append(b * S_e + base)
            b_src.append(p)
            b_dst.append(p * S_e + base)
    return dict(layout="shared", rows=list(rows), S_e=S_e, n_hist=n_hist, n_slots=n_slots, pad=pad, tails=tails, tok=tok,
                visible=visible, entry=entry, extent=extent, kv_src=kv_src, kv_dst=kv_dst, b_src=b_src, b_dst=b_dst,
                n_union=n_union, order=order)


def gather_window(store: torch.Tensor, tok: torch.Tensor, req: Sequence[int] = None) -> torch.Tensor:
    """store (B, S_store, H[, D]); tok (E, S_e, H) token indices; req (E,) the request of each entry (default e).
    Returns store[req[e], tok[e, s, h], h] -> (E, S_e, H[, D])."""
    E, S_e, H = tok.shape
    dev = store.device
    r = torch.arange(E, device=dev) if req is None else torch.as_tensor(list(req), device=dev)
    return store[r.view(E, 1, 1), tok.to(dev), torch.arange(H, device=dev).view(1, 1, H)]


def layout_metrics(call: Dict, units_tokens: int, H: int) -> Dict[str, int]:
    """attended = H x sum of extents (every row loads every row below its extent); window = H x sum over entries of
    the largest extent of the rows using it (each window byte read once); useful = the rows' own read tokens."""
    ext = call["extent"]
    per_entry: Dict[int, int] = {}
    for e, n in zip(call["entry"], ext):
        per_entry[e] = max(per_entry.get(e, 0), n)
    att = H * sum(ext)
    return dict(attended_tokens=att, window_tokens=H * sum(per_entry.values()), useful_tokens=units_tokens,
                masked_tokens=att - units_tokens, rows=len(ext), entries=len(per_entry))


def call_useful_tokens(hist: torch.Tensor, rows: Sequence[Tuple[int, int]], K: int, t0: int, T: int) -> int:
    h = hist.detach().to("cpu")
    H = h.shape[2]
    return sum(total_tokens(tally(row_units(h[b, o, hh].tolist(), o, K, t0, T), t0)) for b, o in rows for hh in range(H))


# ------------------------------------------------------------------ the stage-1 call without NVTX ranges
def stage1_direct(ext, q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    """infllmv2_attn_stage1_fast (dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py:625-728) as
    the decode calls it (v = k, causal False, no window, no softcap, no dropout, no block table, no alibi), WITHOUT the
    wrapper's NVTX ranges: Nsight Compute names a kernel by its INNERMOST push/pop range (job 2175791's raw.csv shows
    only 'stage 2' inside the step range), so the wrapper's 'kernel: stage 1' ranges would hide the harness tag. The
    same q reshape, the same 21 positional arguments of ext.varlen_fwd_stage1 (the CPU test J12 runs the SHIPPED
    function body with a recording stub and compares every argument), the first result."""
    total_q, nheads, head_dim = q.shape
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k
    softmax_scale = q.shape[-1] ** (-0.5)
    q = q.view(total_q, nheads_k, nheads_per_group, head_dim).permute(0, 2, 1, 3).reshape(total_q * nheads_per_group,
                                                                                         nheads_k, head_dim)
    result = ext.varlen_fwd_stage1(q, k, k, None, cu_seqlens_q * nheads_per_group, cu_seqlens_k, None, None, None, None,
                                   max_seqlen_q * nheads_per_group, max_seqlen_k, 0.0, softmax_scale, True, False, -1, -1,
                                   0.0, True, None)
    return result[0]


# ------------------------------------------------------------------ parsers (ncu, torch.profiler)
TAG_RE = re.compile(r"jh\|([a-z_]+)\|([a-z_]+)\|([a-z_]+)")
PASS_RE = re.compile(r'==PROF== Profiling "([^"]*)".*?(\d+) pass(?:es)?\b')
NUM_FIELDS = ("dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
              "lts__t_sectors_aperture_device_op_read.sum", "lts__t_sectors_aperture_device_op_write.sum",
              "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum")
MARKER = "spin_kernel"


def _num(x: str) -> float:
    x = (x or "").strip().replace(",", "")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def ncu_rows(text: str) -> List[Dict]:
    """The kernel rows of an ncu raw CSV export (--csv --page raw --print-units base): the header row (the one naming
    'Kernel Name'), an optional units row (no numeric ID) skipped. Each row: id, name, tag (form, call, part) or None
    (the first jh|form|call|part found in ANY cell: the NVTX push/pop column or a renamed kernel name), the numbers."""
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hi is None:
        raise ValueError("ncu_rows: no header row with 'Kernel Name'")
    head = rows[hi]
    out = []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        if not d.get("ID", "").strip().isdigit():
            continue
        tag = None
        for cell in r:
            m = TAG_RE.search(cell)
            if m:
                tag = m.groups()
                break
        row = dict(id=int(d["ID"]), name=d["Kernel Name"], tag=tag)
        for f in NUM_FIELDS:
            row[f] = _num(d.get(f, ""))
        out.append(row)
    out.sort(key=lambda x: x["id"])
    return out


def ncu_windows(rows: List[Dict]) -> List[Dict]:
    """Split kernel rows into windows: a run of k marker kernels opens window k; the window holds the rows up to the
    next marker run."""
    wins: List[Dict] = []
    cur = None
    run = 0
    for r in rows:
        if MARKER in r["name"]:
            if run == 0:
                cur = dict(markers=0, rows=[])
                wins.append(cur)
            run += 1
            cur["markers"] = run
            continue
        run = 0
        if cur is not None:
            cur["rows"].append(r)
    return wins


def window_summary(win: Dict) -> Dict:
    """Per part (and in total): kernel count, DRAM read / write bytes, kernel time (ns), L2 sectors; the forms and
    calls named by the tags; untagged kernels counted."""
    parts: Dict[str, Dict[str, float]] = {}
    forms, calls, untagged = set(), set(), 0
    tot = dict(kernels=0, **{f: 0.0 for f in NUM_FIELDS})
    for r in win["rows"]:
        key = r["tag"][2] if r["tag"] else "untagged"
        if r["tag"]:
            forms.add(r["tag"][0])
            calls.add(r["tag"][1])
        else:
            untagged += 1
        p = parts.setdefault(key, dict(kernels=0, **{f: 0.0 for f in NUM_FIELDS}))
        p["kernels"] += 1
        tot["kernels"] += 1
        for f in NUM_FIELDS:
            p[f] += r[f]
            tot[f] += r[f]
    return dict(markers=win["markers"], forms=sorted(forms), calls=sorted(calls), untagged=untagged, total=tot,
                parts=parts)


def ncu_passes(log_text: str) -> List[int]:
    return [int(m.group(2)) for m in PASS_RE.finditer(log_text)]


def trace_busy(trace: Dict) -> Dict[str, float]:
    """A torch.profiler chrome trace (dict): the device activities (cat kernel / gpu_memcpy / gpu_memset, ph X) ->
    count, busy (the union of their intervals), span (first start to last end), idle = span - busy, the largest gap
    between consecutive activities; all in ms."""
    ev = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)))
                for e in trace.get("traceEvents", [])
                if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    if not ev:
        return dict(n=0, busy_ms=0.0, span_ms=0.0, idle_ms=0.0, max_gap_ms=0.0)
    busy, gap = 0.0, 0.0
    s0, e0 = ev[0]
    end = e0
    for s, e in ev[1:]:
        if s > end:
            gap = max(gap, s - end)
            busy += end - s0
            s0, end = s, e
        else:
            end = max(end, e)
    busy += end - s0
    span = max(e for _, e in ev) - ev[0][0]
    return dict(n=len(ev), busy_ms=busy / 1e3, span_ms=span / 1e3, idle_ms=(span - busy) / 1e3, max_gap_ms=gap / 1e3)


# ------------------------------------------------------------------ the summary (CPU)
def _gb(x) -> float:
    return round(float(x) / 1e9, 6)


def summarize(u: Dict, p_variants: Dict[str, Dict]) -> Dict:
    """u: the unprofiled cell's result (joint_hbm.py mode cell); p_variants: {cache_control: {"windows": [window
    meta], "summaries": [window_summary]}}. Returns per form: the bytes by class (KV logical by version and level, KV
    layout bounds, weights, compressed keys, activations; analytic), the measured DRAM per part and in total per
    profiled variant, the latency and the launch / idle overhead; and the pairwise comparisons."""
    forms = {}
    for f in u["forms"]:
        a = u["analytic"][f]
        t = u["timing"].get(f, {})
        kt = u["kineto"].get(f, {})
        row = dict(calls=a["calls"], rows=a["rows"], forwards=a["forwards"],
                   analytic_gb=dict(
                       weights=_gb(a["weight_bytes"]), compressed_keys=_gb(a["ck_bytes"]),
                       activations=_gb(a["act_bytes"]),
                       kv_logical_sum={v: _gb(a["kv_sum"][v] * KV_BYTES) for v in VERSIONS},
                       kv_logical_form={v: _gb(a["kv_form"][v] * KV_BYTES) for v in VERSIONS},
                       kv_logical_request={v: _gb(a["kv_request"][v] * KV_BYTES) for v in VERSIONS},
                       kv_layout_window=_gb(a["layout"]["window_tokens"] * KV_BYTES),
                       kv_layout_attended=_gb(a["layout"]["attended_tokens"] * KV_BYTES),
                       bias_attended=_gb(a["layout"]["attended_tokens"] * BIAS_BYTES)),
                   masked_extra_tokens=a["layout"]["masked_tokens"],
                   latency_ms=dict(median=t.get("median_ms"), p95=t.get("p95_ms"), host_enqueue_median=t.get("host_median_ms"),
                                   per_call_median=t.get("per_call_median_ms")),
                   launch=dict(kernels=kt.get("n"), busy_ms=kt.get("busy_ms"), idle_ms=kt.get("idle_ms"),
                               span_ms=kt.get("span_ms"), max_gap_ms=kt.get("max_gap_ms")),
                   measured={})
        for cc, pv in p_variants.items():
            for meta, s in zip(pv["windows"], pv["summaries"]):
                if meta["form"] != f:
                    continue
                row["measured"][cc] = dict(
                    dram_read_bytes=s["total"]["dram__bytes_read.sum"], dram_write_bytes=s["total"]["dram__bytes_write.sum"],
                    dram_read_gb=_gb(s["total"]["dram__bytes_read.sum"]), dram_write_gb=_gb(s["total"]["dram__bytes_write.sum"]),
                    kernels=s["total"]["kernels"], kernel_ms=round(s["total"]["gpu__time_duration.sum"] / 1e6, 3),
                    parts={k: dict(kernels=v["kernels"], dram_read_bytes=v["dram__bytes_read.sum"],
                                   dram_read_gb=_gb(v["dram__bytes_read.sum"]), dram_write_gb=_gb(v["dram__bytes_write.sum"]),
                                   kernel_ms=round(v["gpu__time_duration.sum"] / 1e6, 3))
                           for k, v in sorted(s["parts"].items())},
                    untagged=s["untagged"], window=meta["markers"], order=meta["order"])
        forms[f] = row
    comps = []
    for a, b, what in (("sep_private", "joint_private", "weight sharing only (KV = sum in both)"),
                       ("sep_shared", "joint_shared", "joint execution with KV union vs separate with in-call union"),
                       ("sep_private", "sep_shared", "in-call KV union of the verifier rows"),
                       ("joint_private", "joint_shared", "KV union at equal weight reads"),
                       ("sep_private", "joint_shared", "everything")):
        if a not in forms or b not in forms:
            continue
        c = dict(base=a, other=b, what=what)
        la, lb = forms[a]["latency_ms"]["median"], forms[b]["latency_ms"]["median"]
        if la and lb:
            c["latency_ratio"] = round(lb / la, 4)
        for cc in p_variants:
            ma, mb = forms[a]["measured"].get(cc), forms[b]["measured"].get(cc)
            if ma and mb and ma["dram_read_bytes"]:
                c["dram_read_ratio_" + cc] = round(mb["dram_read_bytes"] / ma["dram_read_bytes"], 4)
                for part in ("attn", "norm_qkv", "wo_ffn", "score"):
                    pa, pb = ma["parts"].get(part), mb["parts"].get(part)
                    if pa and pb and pa["dram_read_bytes"]:
                        c["%s_dram_read_ratio_%s" % (part, cc)] = round(pb["dram_read_bytes"] / pa["dram_read_bytes"], 4)
        comps.append(c)
    return dict(meta=u["meta"], limits=list(LIMITS), gates=u["gates"], forms=forms, comparisons=comps,
                p_variants={cc: dict(windows=pv["windows"], order=[w["form"] for w in pv["windows"]],
                                     passes=pv.get("passes"), found=len(pv["summaries"]))
                            for cc, pv in p_variants.items()})


def summary_md(s: Dict) -> str:
    lines = ["# Joint vs separate execution: HBM traffic (paper section 4.5)", "",
             "meta: " + json.dumps(s["meta"], sort_keys=True), "", "## Stated limits", ""]
    lines += ["- " + x for x in s["limits"]]
    lines += ["", "## Gates", ""]
    for g in s["gates"]:
        lines.append("- %s %s (%s): %s" % (g["name"], "PASS" if g["passed"] else "FAIL", "MUST" if g["must"] else "report",
                                           g["detail"]))
    lines += ["", "## Forms", "",
              "| form | forwards | rows | median / p95 ms | host enqueue ms | kernels | busy / idle ms | weights GB | "
              "KV sum GB (c/x/d) | KV form-union GB | KV request-union GB | KV layout window / attended GB | "
              "compressed keys GB | activations GB | measured DRAM read GB (none / all) | attention DRAM read GB |",
              "|" + "---|" * 16]
    for f, r in s["forms"].items():
        a = r["analytic_gb"]
        kv = lambda d: "%s (%s)" % (round(sum(d.values()), 3), "/".join(str(d[v]) for v in VERSIONS))
        m = r["measured"]
        lines.append("| %s | %d | %d | %s / %s | %s | %s | %s / %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
            f, r["forwards"], r["rows"], r["latency_ms"]["median"], r["latency_ms"]["p95"],
            r["latency_ms"]["host_enqueue_median"], r["launch"]["kernels"], r["launch"]["busy_ms"], r["launch"]["idle_ms"],
            a["weights"], kv(a["kv_logical_sum"]), kv(a["kv_logical_form"]), kv(a["kv_logical_request"]),
            a["kv_layout_window"], a["kv_layout_attended"], a["compressed_keys"], a["activations"],
            " / ".join(str(m[c]["dram_read_gb"]) if c in m else "-" for c in ("none", "all")),
            " / ".join(str(m[c]["parts"].get("attn", {}).get("dram_read_gb")) if c in m else "-" for c in ("none", "all"))))
    lines += ["", "## Comparisons (other / base)", ""]
    for c in s["comparisons"]:
        lines.append("- %s vs %s (%s): %s" % (c["other"], c["base"], c["what"],
                                              ", ".join("%s %s" % (k, v) for k, v in c.items() if k not in ("base", "other", "what"))))
    return "\n".join(lines) + "\n"
