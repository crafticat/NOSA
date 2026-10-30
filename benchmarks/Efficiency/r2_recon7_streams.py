"""RECON7 cell X1 (= step cell r7s), R2 discriminator D1 (KS/analysis/plan/RECON7_R2_joint.md section 4; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE CONCURRENCY HARNESS. It separates TWO questions (docs/NOTES_recon7.md, section
"X1 protocol (about 5 min)"):
  (a) CAPABILITY: can INDEPENDENT dense GEMM work, the fork attention / gather kernels and a copy-engine H2D run at the
      same time on this A100 at our shapes? (no dependency between the streams: an upper bound)
  (b) ATTAINABLE: how much overlap do schedules that keep the joint pass's REAL dependencies get? Inside a layer QKV ->
      (selection) -> fetch -> attention -> MLP are causally dependent; the next layer of the same rows depends on this
      layer's output. Only LEGAL PAIRS overlap:
        GRP   request groups: the rows of request group 0 and of group 1 never read each other's values (GEMMs are
              row-wise, the fork attends per request), so group 0's chain and group 1's chain run on two streams.
              Variant GRPL: lock-step, both groups join after every layer. COST counted, not hidden: each group's GEMMs
              read the full weights, so the dense weight traffic is paid twice (the full-batch SER counts it once).
        PF    a NEXT-round prefetch (the draft rows' misses: their bytes, copy engine) beside the current pass's
              dependent chain: no read of it in this pass.
        WINF  window attention beside the selected-block fetch: NOT BUILDABLE with the shipped fork (its 2D launch writes
              the normalised output only: no log-sum-exp to merge a window part with a fetched part, fork :1060-1063)
              without a kernel change or a different (not bit-equal) computation: NOT MEASURED (stated in the record).
NO NEW KERNEL, NO MODEL: existing ops on SYNTHETIC inputs of the MEASURED shapes (one layer's work, replayed --layers
times = one pass): Aq / Ar the joint pass's dense GEMMs at M = 4 B rows (cuBLAS through torch.mm: wqkv 4608 x 4096; wo
4096 x 4096, gate_up 32768 x 4096, down 4096 x 16384; NOSA-8B config.json); F ONE call of the shipped fork (fia6's blob,
verify_rows._fork_call's arguments, FORK_LAUNCH) whose rows list 64 pages shifted (0, 3, 5, 8): union 72 per (KV head,
request); G the shipped zero-copy gathers (window K, V + bias; store K, V + bias) of --new-blocks NEW host blocks per
(layer, KV head, request); M / P the same bytes as ONE copy-engine copy from pinned staging (M: mode (a); P: mode (b)'s
prefetch, its own buffers); Aq0 .. F1 the same parts on request group 0 / 1 (half the requests each, own outputs).
SELECTION, the residency, the twin step and the norms are NOT replayed (stated: (b) bounds the overlap of the replayed
parts only).
MATCHED CONTROLS: every concurrent configuration has a SERIAL configuration of the same ops on one stream with the same
inputs (CONTROL); all configurations run in the same process, ORDER-ROTATED (lesson L62: the first arm pays more; pass
r runs the configurations starting at index r mod n), after a warm-up of every configuration.
MUST GATES: <shape>-RUN; <shape>-<config>-EXACT (every part's output bits after the configuration's last pass = the part's
reference run alone, so concurrent = serial control = alone); <shape>-UNION (fork union_len = 72 everywhere);
<shape>-IDS (the NEW blocks per layer as requested); <shape>-BUDGET (the shape finished inside --max-shape-s).
STOP CRITERIA (the run aborts the rest of a shape): an EXACT failure; the shape's wall clock above --max-shape-s; a UNION
or IDS failure (before any timing); free device memory below the shape's need (before any allocation).
EXIT: the number of failed MUST gates (0-19); 20 refused (PROTOCOL_STAMP_RECON7 is None: r2_recon7_protocol.py); 30 an
exception outside a shape.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_recon7_protocol as PROT  # noqa: E402

HIDDEN, INTER, HQ, HKV, HD, BS, TOPK, X_STORE, R = 4096, 16384, 32, 2, 128, 64, 64, 129, 4
GEMMS = (("wqkv", 4608, HIDDEN), ("wo", HIDDEN, HIDDEN), ("gate_up", 2 * INTER, HIDDEN), ("down", HIDDEN, INTER))
FORK_LAUNCH = dict(num_warps=4, num_stages=3, tile_size=64)   # = nosi/nosi/verify_rows.py FORK_LAUNCH (test checks)
FORK_ROWS = 4                                                 # = verify_rows.FORK_ROWS
ROW_SHIFT = (0, 3, 5, 8)                                      # union per (KV head, request) = 64 + 8 = 72
BLOCK_BYTES = BS * HD * 2 * 2 + BS * 2                        # K + V + bias of one (KV head, block): 32,896 B
NBH = 16                                                      # host blocks per request (the ids' range)
CHAIN = ["Aq", "G", "F", "Ar"]                                # the joint pass's per-layer order (selection not replayed)
CHAIN0, CHAIN1 = [p + "0" for p in CHAIN], [p + "1" for p in CHAIN]
PARTS = ("Aq", "Ar", "F", "G", "M", "P") + tuple(CHAIN0) + tuple(CHAIN1)
# (name, mode, stream lists, lock-step). Mode "a": CAPABILITY (independent work); "b": ATTAINABLE (legal pairs);
# "c": a CONTROL (the same ops serial on one stream).
CONFIGS = (
    ("A", "a", [["Aq", "Ar"]], False), ("F", "a", [["F"]], False), ("G", "a", [["G"]], False),
    ("M", "a", [["M"]], False), ("FG", "a", [["G", "F"]], False), ("P", "a", [["P"]], False),
    ("SER", "c", [CHAIN], False),
    ("A|F", "a", [["Aq", "Ar"], ["F"]], False), ("A|G", "a", [["Aq", "Ar"], ["G"]], False),
    ("A|M", "a", [["Aq", "Ar"], ["M"]], False), ("F|M", "a", [["F"], ["M"]], False),
    ("A|FG", "a", [["Aq", "Ar"], ["G", "F"]], False), ("A|F|M", "a", [["Aq", "Ar"], ["F"], ["M"]], False),
    ("GRP-SER", "c", [CHAIN0 + CHAIN1], False), ("GRP", "b", [CHAIN0, CHAIN1], False),
    ("GRPL", "b", [CHAIN0, CHAIN1], True),
    ("PF-SER", "c", [CHAIN + ["P"]], False), ("PF", "b", [CHAIN, ["P"]], False))
CONCURRENT = {"A|F": ("A", "F"), "A|G": ("A", "G"), "A|M": ("A", "M"), "F|M": ("F", "M"), "A|FG": ("A", "FG"),
              "A|F|M": ("A", "F", "M")}
CONTROL_OF = {"GRP": "GRP-SER", "GRPL": "GRP-SER", "PF": "PF-SER"}
# the legal pairs: lifetimes of the buffers each one shares (docs/NOTES_recon7.md, X1 protocol, section "pairs")
LEGAL = dict(
    GRP="group g's x / q / k / v rows, its fork lists and tables, its gather destinations: written and read by group g's "
        "stream only, from the layer's QKV GEMM to its MLP output; shared read-only: the weights, the arena (disjoint "
        "pages per request), the pinned host KV (disjoint requests)",
    PF="the prefetch staging (pinned host) and its device buffer: written during layer l of round t, first read by the "
       "joint pass of round t + 1; never read or written by this pass's chain (own allocation: no RAW / WAR hazard)",
    WINF="NOT MEASURED: needs the window part's log-sum-exp (the fork's 2D launch has none) or a non-bit-equal split")
MODEL_S = 0.0868458712                                        # the model's transport interference (REQ: 8.68458712 %)
REACH, SERIAL = 1.10, 0.90                                    # mode (a) rule
WORTH, HIDDEN_OK, SLOW_OK = 0.05, 0.80, 0.05                  # mode (b) rules


def log(msg):
    print("[r2_recon7_streams %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _pairs(ap, s, cast, what):
    out = {}
    try:
        for x in s.split(","):
            if x:
                k, v = x.split(":")
                out[int(k)] = cast(v)
    except ValueError:
        ap.error("%s: C:value pairs" % what)
    return out


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="128:216,73:318", help="C:B pairs (the measured configurations)")
    ap.add_argument("--new-blocks", default="128:1.226,73:5.0", help="C:NEW host blocks per (layer, KV head, request)")
    ap.add_argument("--union-pages", type=int, default=72, choices=(72,), help="the fork's union per (KV head, request)")
    ap.add_argument("--win-share", type=float, default=0.5, help="the share of the NEW blocks that are window loads")
    ap.add_argument("--layers", type=int, default=32, help="layers replayed per pass")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-shape-s", type=float, default=240.0, help="STOP: a shape's wall clock above this (s)")
    a = ap.parse_args(argv)
    a.shapes = _pairs(ap, a.shapes, int, "--shapes")
    a.new_blocks = _pairs(ap, a.new_blocks, float, "--new-blocks")
    if not a.shapes or set(a.shapes) != set(a.new_blocks):
        ap.error("--shapes and --new-blocks: the same C values")
    if not (0.0 <= a.win_share <= 1.0) or a.layers < 1 or a.reps < 3 or a.warmup < 1 or a.max_shape_s <= 0:
        ap.error("--win-share in [0, 1], --layers >= 1, --reps >= 3, --warmup >= 1, --max-shape-s > 0")
    if any(b < 2 or b > 1024 or b % 2 for b in a.shapes.values()):
        ap.error("--shapes: B even, in 2 .. 1024 (two request groups)")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def counts_per_layer(B: int, H: int, per_stream: float, win_share: float):
    """(window loads, store loads) per layer: round(per_stream x H x B) NEW blocks, the window share rounded."""
    tot = int(round(float(per_stream) * H * B))
    w = int(round(tot * float(win_share)))
    return w, tot - w


def spread(total: int, H: int, B: int, cap: int, seed: int):
    """(H, B) counts that sum to `total`, each <= cap: total // (H B) everywhere, the remainder on a seeded random set
    of (head, request) pairs."""
    import torch
    n = H * B
    if total > n * cap:
        raise ValueError("%d blocks do not fit %d streams of %d slots" % (total, n, cap))
    base, rem = divmod(int(total), n)
    out = torch.full((n,), base, dtype=torch.int64)
    if rem:
        g = torch.Generator().manual_seed(seed)
        out[torch.randperm(n, generator=g)[:rem]] += 1
    return out.view(H, B)


def make_ids(counts, M: int, nbh: int, seed: int, first_free: int = 0):
    """ids (H, B, M) int64: per (h, b) counts[h, b] DISTINCT host block ids in [0, nbh) at seeded random slots >=
    first_free (the store's slot 0 stays -1); every other slot -1."""
    import torch
    H, B = counts.shape
    g = torch.Generator().manual_seed(seed)
    ids = torch.full((H, B, M), -1, dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            c = int(counts[h, b])
            if c == 0:
                continue
            if c > min(nbh, M - first_free):
                raise ValueError("%d ids do not fit (nbh %d, slots %d)" % (c, nbh, M - first_free))
            slots = first_free + torch.randperm(M - first_free, generator=g)[:c]
            ids[h, b, slots] = torch.randperm(nbh, generator=g)[:c]
    return ids


def make_lists(B: int, H: int, U: int = 72):
    """The fork's row lists (verify_rows._fork_lists' form, entry 0, slots_per_entry 1): row r of request b, KV head h
    lists the arena blocks b x U + ROW_SHIFT[r] + [0, 64) in ascending order (the precondition of build_union);
    last_len 40 + r. Returns dict(page_key, page_table, num_pages, last_len, seq_idx) (int32, contiguous)."""
    import torch
    E = B * R
    base = (torch.arange(B, dtype=torch.int64) * U).view(B, 1, 1, 1)
    sh = torch.tensor(ROW_SHIFT, dtype=torch.int64).view(1, R, 1, 1)
    keys = (base + sh + torch.arange(TOPK, dtype=torch.int64).view(1, 1, 1, TOPK)).expand(B, R, H, TOPK)
    keys = keys.reshape(E, H, TOPK).to(torch.int32).contiguous()
    return dict(page_key=keys, page_table=keys.clone(), num_pages=torch.full((E, H), TOPK, dtype=torch.int32),
                last_len=(40 + torch.arange(E, dtype=torch.int32) % R).contiguous(),
                seq_idx=torch.zeros(E, dtype=torch.int32), slots_per_entry=1)


def slice_lists(lists, e0: int, e1: int):
    """The rows [e0, e1) of the row lists (a request group: contiguous int32 copies)."""
    return {k: (v[e0:e1].contiguous() if hasattr(v, "shape") else v) for k, v in lists.items()}


def union_count(lists, B: int, H: int):
    """The union size per (KV head, request) of the rows' lists (torch, as fork_build_union counts FIRST entries)."""
    import torch
    k = lists["page_key"].view(B, R, H, -1).permute(2, 0, 1, 3).reshape(H, B, -1)
    out = torch.zeros((H, B), dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            out[h, b] = int(torch.unique(k[h, b]).numel())
    return out


def rotation(names, r: int):
    """Pass r's order: the configurations from index r mod n (lesson L62: no configuration is always first)."""
    n = len(names)
    k = r % n
    return list(names[k:]) + list(names[:k])


def summarize(times: dict) -> dict:
    """times {config: [ms per pass]} -> medians; mode (a): the concurrent sets against their parts alone (max, sum,
    efficiency eff = (sum - T) / (sum - max)), the interference A|M, the decision; mode (b): each legal schedule against
    its matched serial control and against the full-batch serial chain SER (shared work counted once), the decisions."""
    med = {k: (round(statistics.median(v), 4) if v else None) for k, v in times.items()}
    out = dict(median_ms=med, spread_ms={k: ([round(min(v), 4), round(max(v), 4)] if v else None)
                                         for k, v in times.items()})
    conc = {}
    for c, parts in CONCURRENT.items():
        t, al = med.get(c), [med.get(p) for p in parts]
        if t is None or any(x is None for x in al):
            conc[c] = None
            continue
        mx, sm = max(al), sum(al)
        conc[c] = dict(ms=t, parts=list(parts), alone=al, max=round(mx, 4), sum=round(sm, 4),
                       over_max=round(t / mx, 4), over_sum=round(t / sm, 4),
                       eff=(round((sm - t) / (sm - mx), 4) if sm > mx else None))
    out["concurrent"] = conc
    ser = med.get("SER")
    parts = [med.get(p) for p in ("A", "F", "G")]
    out["serial"] = (dict(ms=ser, sum_alone=round(sum(parts), 4), over_sum=round(ser / sum(parts), 4))
                     if ser is not None and None not in parts else None)
    am = conc.get("A|M")
    out["interference_AM"] = round(am["over_max"] - 1.0, 4) if am else None
    out["model_s"] = MODEL_S
    fm = conc.get("A|F|M")
    if fm is None:
        out["capability"] = None
    elif fm["over_max"] <= REACH:
        out["capability"] = "REACHABLE"
    elif fm["over_sum"] >= SERIAL:
        out["capability"] = "SERIAL"
    else:
        out["capability"] = "PARTIAL"
    att = {}
    for c in ("GRP", "GRPL"):
        t, ctl = med.get(c), med.get(CONTROL_OF[c])
        if None in (t, ctl, ser):
            att[c] = None
            continue
        net = ser - t
        att[c] = dict(ms=t, control_ms=ctl, ser_ms=ser, split_cost_ms=round(ctl - ser, 4),
                      overlap_ms=round(ctl - t, 4), net_ms=round(net, 4), net_frac=round(net / ser, 4),
                      worth=(net / ser) >= WORTH)
    t, ctl, p = med.get("PF"), med.get("PF-SER"), med.get("P")
    if None in (t, ctl, ser, p) or not p:
        att["PF"] = None
    else:
        hid = (ser + p - t) / p
        slow = t / ser - 1.0
        att["PF"] = dict(ms=t, control_ms=ctl, ser_ms=ser, P_ms=p, hidden_frac=round(hid, 4), chain_slowdown=round(slow, 4),
                         control_over_sum=round(ctl / (ser + p), 4), hides=(hid >= HIDDEN_OK and slow <= SLOW_OK))
    att["WINF"] = "NOT MEASURED: " + LEGAL["WINF"]
    out["attainable"] = att
    return out


def exact_gate(name: str, ref: dict, got: dict, bits):
    """<name>-EXACT (MUST): every part output of the configuration = its reference output, bit for bit."""
    bad = [k for k in got if k not in ref or not (got[k].shape == ref[k].shape and bool((bits(got[k]) ==
                                                                                       bits(ref[k])).all()))]
    return ("%s-EXACT" % name, True, bool(got) and not bad, "%d outputs compared; unequal %s" % (len(got), bad[:4]))


# ------------------------------------------------------------------ the measurement (GPU; CPU stand-ins in the tests)
class CudaDev:
    """The CUDA streams / events backend."""

    def __init__(self):
        import torch
        self.torch = torch
        self.main = torch.cuda.current_stream()

    def stream(self):
        return self.torch.cuda.Stream()

    def event(self):
        return self.torch.cuda.Event(enable_timing=True)

    def ctx(self, s):
        return self.torch.cuda.stream(s)

    def record(self, e, s):
        e.record(s)

    def wait(self, s, e):
        s.wait_event(e)

    def elapsed(self, e0, e1) -> float:
        return float(e0.elapsed_time(e1))

    def sync(self):
        self.torch.cuda.synchronize()


def bits(t):
    import torch
    if t.is_floating_point():
        return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    return t


def run_config(dev, parts, streams_spec, layers: int, side, lockstep: bool = False):
    """ONE pass of a configuration: the main stream records a start event; each inner list runs on its own stream
    (one list: the main stream) after the start, layer by layer; lockstep: after every layer every stream waits for
    every other stream's layer end (a per-layer join); the main stream joins every stream and records the end. Returns
    the elapsed ms (after a sync)."""
    e0, e1 = dev.event(), dev.event()
    dev.record(e0, dev.main)
    ss = [dev.main] if len(streams_spec) == 1 else side[:len(streams_spec)]
    for s in ss:
        if s is not dev.main:
            dev.wait(s, e0)
    for l in range(layers):
        for s, seq in zip(ss, streams_spec):
            with dev.ctx(s):
                for p in seq:
                    parts[p]()
        if lockstep and len(ss) > 1 and l + 1 < layers:
            evs = []
            for s in ss:
                e = dev.event()
                dev.record(e, s)
                evs.append(e)
            for s in ss:
                for e in evs:
                    dev.wait(s, e)
    ends = []
    for s in ss:
        if s is not dev.main:
            e = dev.event()
            dev.record(e, s)
            ends.append(e)
    for e in ends:
        dev.wait(dev.main, e)
    dev.record(e1, dev.main)
    dev.sync()
    return dev.elapsed(e0, e1)


def measure(dev, parts, outputs, layers: int, reps: int, warmup: int, configs=CONFIGS, max_s=None, clock=None):
    """The reference outputs (each part alone, one layer); a warm-up of every configuration; then `reps` ORDER-ROTATED
    passes (pass r starts at configuration r mod n); after a configuration's last pass its EXACT gate. STOP: an EXACT
    failure or the wall clock above max_s ends the measurement (the record says so). Returns (times, gates, stop)."""
    clock = clock or time.perf_counter
    t0 = clock()
    side = [dev.stream() for _ in range(3)]
    spec = {c[0]: c for c in configs}
    used = [p for p in PARTS if p in parts]
    ref = {}
    for p in used:
        run_config(dev, parts, [[p]], 1, side)
        for i, t in enumerate(outputs[p]):
            ref["%s.%d" % (p, i)] = t.clone()
    names = [c[0] for c in configs]
    for name in names:
        for _ in range(warmup):
            run_config(dev, parts, spec[name][2], layers, side, spec[name][3])
    times, gates, stop = {n: [] for n in names}, [], None
    for r in range(reps):
        for name in rotation(names, r):
            _, _, lists, lock = spec[name]
            times[name].append(run_config(dev, parts, lists, layers, side, lock))
            if r == reps - 1:
                ps = sorted({p for seq in lists for p in seq})
                got = {"%s.%d" % (p, i): t for p in ps for i, t in enumerate(outputs[p])}
                g = exact_gate(name, ref, got, bits)
                gates.append(g)
                if not g[2]:
                    stop = "EXACT failed in %s" % name
                    break
            if max_s is not None and clock() - t0 > max_s:
                stop = "wall clock %.0f s above %.0f s (pass %d)" % (clock() - t0, max_s, r)
                break
        if stop:
            break
    return times, gates, stop


def need_bytes(B: int, union: int = 72) -> int:
    """MODEL: the device bytes of one shape (weights, activations, the outputs and the gather destinations four times:
    full and groups, each with its reference clone; the arena; the copy buffers; 1 GiB slack): the free-memory STOP."""
    E = R * B
    w = sum(N * K for _, N, K in GEMMS) * 2
    act = E * (3 * HIDDEN + INTER) * 2
    outs = 4 * E * sum(N for _, N, _ in GEMMS) * 2
    arena = (B * union + 8) * BS * HKV * (2 * HD + 1) * 2
    dst = 4 * B * (TOPK + X_STORE) * BS * HKV * (2 * HD + 1) * 2
    return w + act + outs + arena + dst + 2 * E * HQ * HD * 2 + (1 << 30)


def build_gpu(B: int, per_stream: float, win_share: float, seed: int, union: int = 72):
    """The parts of one shape on the GPU (synthetic inputs of the measured shapes; module docstring): the full-batch
    parts and the two request groups' parts (own outputs, own tables, slices of the shared read-only inputs). Returns
    (parts {name: callable}, outputs {name: [tensors]}, meta)."""
    import torch
    import fork_vllm_unified_attention as FK
    from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
    from nosi.flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf = torch.bfloat16
    E = R * B
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(bf).to(dev)   # noqa: E731
    W = {n: rnd(N, K) for n, N, K in GEMMS}
    x_in, x_att, x_mlp, x_dn = rnd(E, HIDDEN), rnd(E, HIDDEN), rnd(E, HIDDEN), rnd(E, INTER)
    NB = B * union + 8
    ak, av = rnd(NB, BS, HKV, HD), rnd(NB, BS, HKV, HD)
    ab = rnd(NB, BS, HKV)
    q = rnd(E, HQ, HD)
    lists = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in make_lists(B, HKV, union).items()}
    nw, ns = counts_per_layer(B, HKV, per_stream, win_share)
    ids_w = make_ids(spread(nw, HKV, B, min(NBH, TOPK), seed + 1), TOPK, NBH, seed + 2).to(dev)
    ids_s = make_ids(spread(ns, HKV, B, min(NBH, X_STORE - 1), seed + 3), X_STORE, NBH, seed + 4, 1).to(dev)
    hk = (torch.randn(B, NBH * BS, HKV, HD, generator=g) * 0.05).to(bf).pin_memory()
    hv = (torch.randn(B, NBH * BS, HKV, HD, generator=g) * 0.05).to(bf).pin_memory()
    cis = rnd(B, NBH * BS, HKV)
    nbytes = max((nw + ns) * BLOCK_BYTES, 1)
    parts, outputs = {}, {}

    def unit(tag, b0, b1):
        """The chain parts of the requests [b0, b1) (rows [R b0, R b1)), suffix tag."""
        e0, e1 = R * b0, R * b1
        Bg, Eg = b1 - b0, e1 - e0
        y = {n: torch.empty((Eg, N), dtype=bf, device=dev) for n, N, _ in GEMMS}
        xi, xa, xm, xd = x_in[e0:e1], x_att[e0:e1], x_mlp[e0:e1], x_dn[e0:e1]

        def Aq():
            torch.mm(xi, W["wqkv"].t(), out=y["wqkv"])

        def Ar():
            torch.mm(xa, W["wo"].t(), out=y["wo"])
            torch.mm(xm, W["gate_up"].t(), out=y["gate_up"])
            torch.mm(xd, W["down"].t(), out=y["down"])
        qg = q[e0:e1]
        fout = torch.zeros((Eg, HQ, HD), dtype=bf, device=dev)
        lg = slice_lists(lists, e0, e1)
        cu = torch.arange(0, Eg + 1, R, dtype=torch.int32, device=dev)
        seq = torch.zeros(Bg, dtype=torch.int32, device=dev)
        Wt = FORK_ROWS * TOPK
        tab = torch.zeros((HKV, Bg, Wt), dtype=torch.int32, device=dev)
        ext = torch.zeros((HKV, Bg, Wt), dtype=torch.int32, device=dev)
        ln = torch.zeros((HKV, Bg), dtype=torch.int32, device=dev)

        def F():
            FK.unified_attention(qg, ak, av, fout, cu, FORK_ROWS, seq, BS, HD ** -0.5, True, (-1, -1), tab, 0, None,
                                 None, None, union_ext=ext, union_len=ln, cis_bias=ab, build_union=lg,
                                 rows_per_q_block=FORK_ROWS, **FORK_LAUNCH)
        iw, is_ = ids_w[:, b0:b1].contiguous(), ids_s[:, b0:b1].contiguous()
        hkg, hvg, cg = hk[b0:b1], hv[b0:b1], cis[b0:b1]
        wk = torch.zeros((Bg, TOPK * BS, HKV, HD), dtype=bf, device=dev)
        wv, wb = torch.zeros_like(wk), torch.zeros((Bg, TOPK * BS, HKV), dtype=bf, device=dev)
        sk = torch.zeros((Bg, X_STORE * BS, HKV, HD), dtype=bf, device=dev)
        sv, sb = torch.zeros_like(sk), torch.zeros((Bg, X_STORE * BS, HKV), dtype=bf, device=dev)

        def G():
            flash_h2d_from_mask(wk, hkg, iw, BS)
            flash_h2d_from_mask_bias(wv, hvg, wb, cg, iw, BS)
            flash_h2d_from_mask(sk, hkg, is_, BS)
            flash_h2d_from_mask_bias(sv, hvg, sb, cg, is_, BS)
        parts.update({"Aq" + tag: Aq, "Ar" + tag: Ar, "F" + tag: F, "G" + tag: G})
        outputs.update({"Aq" + tag: [y["wqkv"]], "Ar" + tag: [y["wo"], y["gate_up"], y["down"]], "F" + tag: [fout],
                        "G" + tag: [wk, wv, wb, sk, sv, sb]})
        return ln, int((iw >= 0).sum()), int((is_ >= 0).sum())
    ln, cw, cs = unit("", 0, B)
    ln0, _, _ = unit("0", 0, B // 2)
    ln1, _, _ = unit("1", B // 2, B)

    def copier():
        stage = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, generator=g).pin_memory()
        dst = torch.zeros((nbytes,), dtype=torch.uint8, device=dev)
        return (lambda: dst.copy_(stage, non_blocking=True)), dst
    parts["M"], m = copier()
    parts["P"], p = copier()
    outputs.update(M=[m], P=[p])
    meta = dict(B=B, E=E, groups=[[0, B // 2], [B // 2, B]], NB_arena=NB, union=union, new_per_stream=per_stream,
                new_win=nw, new_store=ns, bytes_per_layer=nbytes, ids_win=cw, ids_store=cs, ln=[ln, ln0, ln1],
                gemm_flop_per_layer=sum(2 * E * N * K for _, N, K in GEMMS), seed=seed, legal=LEGAL)
    return parts, outputs, meta


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    why = PROT.refusal("r2_recon7_streams")
    if why is not None:
        print(why, file=sys.stderr)
        return 20
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    import torch
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device", file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, res = [], {}
    dev = CudaDev()
    props = torch.cuda.get_device_properties(0)
    for C, B in sorted(a.shapes.items(), key=lambda kv: -kv[0]):
        name = "C%d_B%d" % (C, B)
        t0 = time.perf_counter()
        free = torch.cuda.mem_get_info()[0]
        need = need_bytes(B, a.union_pages)
        if free < need:
            gates.append(("%s-RUN" % name, True, False, "STOP: free %.2f GB below the need %.2f GB (MODEL)" % (
                free / 1e9, need / 1e9)))
            continue
        log("shape %s: NEW %s per (layer, KV head, request); need %.2f GB of %.2f GB free" % (
            name, a.new_blocks[C], need / 1e9, free / 1e9))
        try:
            with torch.inference_mode():
                parts, outputs, meta = build_gpu(B, a.new_blocks[C], a.win_share, a.seed + C, a.union_pages)
                for p in ("F", "F0", "F1"):                            # build the union tables once
                    parts[p]()
                torch.cuda.synchronize()
                lns = meta.pop("ln")
                u_ok = all(bool((x == a.union_pages).all()) for x in lns)
                gates.append(("%s-UNION" % name, True, u_ok, "union_len min %s max %s (want %d at every (KV head, "
                              "request), full and both groups)" % ([int(x.min()) for x in lns],
                                                                  [int(x.max()) for x in lns], a.union_pages)))
                want = counts_per_layer(B, HKV, a.new_blocks[C], a.win_share)
                i_ok = (meta["ids_win"], meta["ids_store"]) == want
                gates.append(("%s-IDS" % name, True, i_ok, "NEW blocks per layer window %d store %d (want %s)" % (
                    meta["ids_win"], meta["ids_store"], want)))
                if not (u_ok and i_ok):
                    gates.append(("%s-RUN" % name, True, False, "STOP: UNION or IDS failed before the timing"))
                    continue
                times, eg, stop = measure(dev, parts, outputs, a.layers, a.reps, a.warmup,
                                          max_s=a.max_shape_s - (time.perf_counter() - t0))
                same = {}
                for p in ("Aq", "Ar", "F", "G"):                       # REPORT: the groups' bits = the full batch's
                    same[p] = all(bool((bits(torch.cat((x0, x1), 0)) == bits(x)).all()) for x, x0, x1 in
                                  zip(outputs[p], outputs[p + "0"], outputs[p + "1"]))
            gates.extend(("%s-%s" % (name, n), m, ok, d) for n, m, ok, d in eg)
            wall = time.perf_counter() - t0
            gates.append(("%s-BUDGET" % name, True, stop is None or not stop.startswith("wall"),
                          "wall %.1f s (limit %.0f s); stop %s" % (wall, a.max_shape_s, stop)))
            s = summarize(times)
            res[name] = dict(C=C, B=B, meta=meta, times_ms=times, summary=s, stop=stop, wall_s=round(wall, 1),
                             groups_equal_full=same)
            gates.append(("%s-RUN" % name, True, stop is None, "%d configurations x %d order-rotated passes of %d "
                          "layers; stop %s" % (len(CONFIGS), a.reps, a.layers, stop)))
            del parts, outputs
            torch.cuda.empty_cache()
            if stop is not None:
                break
        except Exception as e:
            traceback.print_exc()
            gates.append(("%s-RUN" % name, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            break
    meta = dict(shapes=a.shapes, new_blocks=a.new_blocks, union_pages=a.union_pages, win_share=a.win_share,
                layers=a.layers, reps=a.reps, warmup=a.warmup, seed=a.seed, max_shape_s=a.max_shape_s, gemms=GEMMS,
                fork_launch=FORK_LAUNCH, row_shift=ROW_SHIFT, block_bytes=BLOCK_BYTES, nbh=NBH, configs=CONFIGS,
                legal=LEGAL, rules=dict(reach=REACH, serial=SERIAL, worth=WORTH, hidden=HIDDEN_OK, slow=SLOW_OK),
                gpu=props.name, torch=torch.__version__, protocol_stamp=PROT.PROTOCOL_STAMP_RECON7,
                base_commit=os.environ.get("BASE_COMMIT"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, shapes=res, gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3])
                                                     for g in gates]), f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-22s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for name, r in res.items():
        s = r["summary"]
        print("X1 %s medians ms per pass of %d layers: %s" % (name, a.layers, s["median_ms"]))
        for c, v in s["concurrent"].items():
            print("X1 (a) CAPABILITY %s %s: %s" % (name, c, v))
        print("X1 (a) CAPABILITY %s: serial %s; interference A|M %s (model s %s); DECISION %s" % (
            name, s["serial"], s["interference_AM"], MODEL_S, s["capability"]))
        for c, v in s["attainable"].items():
            print("X1 (b) ATTAINABLE %s %s: %s" % (name, c, v))
        print("X1 %s groups' outputs = the full batch's bits (REPORT) %s; wall %s s; stop %s" % (
            name, r["groups_equal_full"], r["wall_s"], r["stop"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_recon7_streams: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
