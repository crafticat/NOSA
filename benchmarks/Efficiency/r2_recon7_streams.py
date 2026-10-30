"""RECON7 cell (b), R2 discriminator D1 (KS/analysis/plan/RECON7_R2_joint.md section 4; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE TWO-STREAM CONCURRENCY MICRO-HARNESS. Question: can the model's max(HBM, dense)
overlap exist on THIS A100 for THESE kernels at the measured shapes, or is the serial cost a hardware limit? The joint
pass runs ONE dependent stream (R2: 9,011 GPU ops, 0.000 ms concurrency), so its traces cannot answer it.

NO NEW KERNEL, NO MODEL: existing ops only, on SYNTHETIC inputs of the MEASURED shapes (one layer's work, replayed for
--layers layers = one pass):
  A  the joint pass's dense GEMMs of one layer at M = 4 B rows (cuBLAS through torch.mm, the op of F.linear):
     wqkv (4608 x 4096), wo (4096 x 4096), gate_up (32768 x 4096), down (4096 x 16384) (NOSA-8B config.json; the LM head
     runs once per pass and is not replayed per layer)
  F  ONE call of the shipped fork (benchmarks/Efficiency/fork_vllm_unified_attention.py, fia6's blob, the arguments of
     nosi/nosi/verify_rows.py _fork_call with FORK_LAUNCH) over 4 rows per request: each row lists 64 pages, the rows
     shifted by ROW_SHIFT, so the kernel's union is 72 pages per (KV head, request) (R2: 72.24 measured)
  G  the shipped zero-copy H2D gathers (nosi/nosi/flash_cache_engine/flash_h2d_mask(_bias).py) of one layer: the
     window load (K, V + bias, M = 64 slots) and the store fill (K, V + bias, M = 129 slots), --new-blocks NEW host
     blocks per (layer, KV head, request) (R2 / R3 kv_at_verify: 1.226 at C 128, 5.000 at C 73), --win-share of them
     window loads, from pinned host memory
  M  the SAME bytes as G as ONE copy-engine cudaMemcpyAsync per layer (torch copy_ from a pinned staging buffer)
CONFIGURATIONS (CONFIGS; one CUDA stream per inner list; each list runs its parts layer by layer; streams start on one
event and join on the main stream): alone A, F, G, M, FG (F + G on one stream: the HBM side with the in-stream H2D);
SER (Aq, G, F, Ar per layer on ONE stream: the joint pass's order); then concurrent A|F, A|G, A|M, F|M, A|FG, A|F|M.
REPORTED per shape: the median ms of each configuration over --reps passes (after --warmup), the overlap efficiency
eff = (sum of the parts alone - concurrent) / (sum - max) (1 = the max() is reached, 0 = serial), the interference
T(A|M) / max(A, M) - 1 against the model's s = 8.685 %. THE DECISION RULE (docs/NOTES_recon7.md section 3.2, REPORT):
REACHABLE when T(A|F|M) <= 1.10 x max(A, F, M); SERIAL when T >= 0.90 x the sum; PARTIAL otherwise.
MUST GATES: RUN per shape; EXACT per configuration (every part's output bits after the configuration = the same part's
output bits from its reference run, alone: concurrency changes no value); UNION (the fork's union_len = 72 at every (KV
head, request)); IDS (the NEW blocks per layer = the requested count). EXIT: the number of failed MUST gates (0-19); 20
refused (PROTOCOL_STAMP_RECON7 is None: r2_recon7_protocol.py); 30 an exception outside a shape.
"""
from __future__ import annotations

import argparse
import contextlib
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
PARTS = ("Aq", "Ar", "F", "G", "M")
CONFIGS = (("A", [["Aq", "Ar"]]), ("F", [["F"]]), ("G", [["G"]]), ("M", [["M"]]), ("FG", [["G", "F"]]),
           ("SER", [["Aq", "G", "F", "Ar"]]), ("A|F", [["Aq", "Ar"], ["F"]]), ("A|G", [["Aq", "Ar"], ["G"]]),
           ("A|M", [["Aq", "Ar"], ["M"]]), ("F|M", [["F"], ["M"]]), ("A|FG", [["Aq", "Ar"], ["G", "F"]]),
           ("A|F|M", [["Aq", "Ar"], ["F"], ["M"]]))
CONCURRENT = dict(**{"A|F": ("A", "F"), "A|G": ("A", "G"), "A|M": ("A", "M"), "F|M": ("F", "M"),
                     "A|FG": ("A", "FG"), "A|F|M": ("A", "F", "M")})
MODEL_S = 0.0868458712                                        # the model's transport interference (REQ: 8.68458712 %)
REACH, SERIAL = 1.10, 0.90


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
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    a.shapes = _pairs(ap, a.shapes, int, "--shapes")
    a.new_blocks = _pairs(ap, a.new_blocks, float, "--new-blocks")
    if not a.shapes or set(a.shapes) != set(a.new_blocks):
        ap.error("--shapes and --new-blocks: the same C values")
    if not (0.0 <= a.win_share <= 1.0) or a.layers < 1 or a.reps < 3 or a.warmup < 1:
        ap.error("--win-share in [0, 1], --layers >= 1, --reps >= 3, --warmup >= 1")
    if any(b < 1 or b > 1024 for b in a.shapes.values()):
        ap.error("--shapes: B in 1 .. 1024")
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


def union_count(lists, B: int, H: int):
    """The union size per (KV head, request) of the rows' lists (torch, as fork_build_union counts FIRST entries)."""
    import torch
    k = lists["page_key"].view(B, R, H, -1).permute(2, 0, 1, 3).reshape(H, B, -1)
    out = torch.zeros((H, B), dtype=torch.int64)
    for h in range(H):
        for b in range(B):
            out[h, b] = int(torch.unique(k[h, b]).numel())
    return out


def summarize(times: dict) -> dict:
    """times {config: [ms per pass]} -> the medians, the concurrent configurations' max / sum of their parts alone,
    the overlap efficiency, the ratios, the interference A|M, the decision of the rule."""
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
        out["decision"] = None
    elif fm["over_max"] <= REACH:
        out["decision"] = "REACHABLE"
    elif fm["over_sum"] >= SERIAL:
        out["decision"] = "SERIAL"
    else:
        out["decision"] = "PARTIAL"
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


def run_config(dev, parts, streams_spec, layers: int, side):
    """ONE pass of a configuration: the main stream records a start event; each inner list runs on its own stream
    (the first list on the main stream when there is one list) after the start, layer by layer; the main stream waits
    for every stream's end event and records the end. Returns the elapsed ms (after a sync)."""
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


def measure(dev, parts, outputs, layers: int, reps: int, warmup: int, configs=CONFIGS):
    """Every configuration: warmup + reps passes; the reference outputs from the parts alone (run once each before
    the configurations); the EXACT gate per configuration. outputs: {part: [tensors it writes]}."""
    side = [dev.stream() for _ in range(3)]
    ref = {}
    for p in PARTS:                                                    # the reference: each part alone, one pass
        run_config(dev, parts, [[p]], 1, side)
        for i, t in enumerate(outputs[p]):
            ref["%s.%d" % (p, i)] = t.clone()
    times, gates = {}, []
    for name, spec in configs:
        for _ in range(warmup):
            run_config(dev, parts, spec, layers, side)
        times[name] = [run_config(dev, parts, spec, layers, side) for _ in range(reps)]
        used = sorted({p for seq in spec for p in seq})
        got = {"%s.%d" % (p, i): t for p in used for i, t in enumerate(outputs[p])}
        gates.append(exact_gate(name, ref, got, bits))
    return times, gates


def build_gpu(B: int, per_stream: float, win_share: float, seed: int, union: int = 72):
    """The parts of one shape on the GPU (synthetic inputs of the measured shapes; module docstring) and the check
    values. Returns (parts {name: callable}, outputs {name: [tensors]}, meta)."""
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
    y = {n: torch.empty((E, N), dtype=bf, device=dev) for n, N, _ in GEMMS}

    def Aq():
        torch.mm(x_in, W["wqkv"].t(), out=y["wqkv"])

    def Ar():
        torch.mm(x_att, W["wo"].t(), out=y["wo"])
        torch.mm(x_mlp, W["gate_up"].t(), out=y["gate_up"])
        torch.mm(x_dn, W["down"].t(), out=y["down"])
    NB = B * union + 8
    ak, av = rnd(NB, BS, HKV, HD), rnd(NB, BS, HKV, HD)
    ab = rnd(NB, BS, HKV)
    q = rnd(E, HQ, HD)
    fout = torch.zeros((E, HQ, HD), dtype=bf, device=dev)
    lists = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in make_lists(B, HKV, union).items()}
    NS = B
    Wt = FORK_ROWS * TOPK
    cu = torch.arange(0, E + 1, R, dtype=torch.int32, device=dev)
    seq = torch.zeros(NS, dtype=torch.int32, device=dev)
    tab = torch.zeros((HKV, NS, Wt), dtype=torch.int32, device=dev)
    ext = torch.zeros((HKV, NS, Wt), dtype=torch.int32, device=dev)
    ln = torch.zeros((HKV, NS), dtype=torch.int32, device=dev)

    def F():
        FK.unified_attention(q, ak, av, fout, cu, FORK_ROWS, seq, BS, HD ** -0.5, True, (-1, -1), tab, 0, None, None,
                             None, union_ext=ext, union_len=ln, cis_bias=ab, build_union=lists,
                             rows_per_q_block=FORK_ROWS, **FORK_LAUNCH)
    nw, ns = counts_per_layer(B, HKV, per_stream, win_share)
    ids_w = make_ids(spread(nw, HKV, B, min(NBH, TOPK), seed + 1), TOPK, NBH, seed + 2).to(dev)
    ids_s = make_ids(spread(ns, HKV, B, min(NBH, X_STORE - 1), seed + 3), X_STORE, NBH, seed + 4, 1).to(dev)
    hk = (torch.randn(B, NBH * BS, HKV, HD, generator=g) * 0.05).to(bf).pin_memory()
    hv = (torch.randn(B, NBH * BS, HKV, HD, generator=g) * 0.05).to(bf).pin_memory()
    cis = rnd(B, NBH * BS, HKV)
    wk = torch.zeros((B, TOPK * BS, HKV, HD), dtype=bf, device=dev)
    wv, wb = torch.zeros_like(wk), torch.zeros((B, TOPK * BS, HKV), dtype=bf, device=dev)
    sk = torch.zeros((B, X_STORE * BS, HKV, HD), dtype=bf, device=dev)
    sv, sb = torch.zeros_like(sk), torch.zeros((B, X_STORE * BS, HKV), dtype=bf, device=dev)

    def G():
        flash_h2d_from_mask(wk, hk, ids_w, BS)
        flash_h2d_from_mask_bias(wv, hv, wb, cis, ids_w, BS)
        flash_h2d_from_mask(sk, hk, ids_s, BS)
        flash_h2d_from_mask_bias(sv, hv, sb, cis, ids_s, BS)
    nbytes = (nw + ns) * BLOCK_BYTES
    stage = torch.randint(0, 255, (max(nbytes, 1),), dtype=torch.uint8, generator=g).pin_memory()
    mdst = torch.zeros((max(nbytes, 1),), dtype=torch.uint8, device=dev)

    def M():
        mdst.copy_(stage, non_blocking=True)
    parts = dict(Aq=Aq, Ar=Ar, F=F, G=G, M=M)
    outputs = dict(Aq=[y["wqkv"]], Ar=[y["wo"], y["gate_up"], y["down"]], F=[fout], G=[wk, wv, wb, sk, sv, sb], M=[mdst])
    meta = dict(B=B, E=E, NB_arena=NB, union=union, new_per_stream=per_stream, new_win=nw, new_store=ns,
                bytes_per_layer=nbytes, ids_win=int((ids_w >= 0).sum()), ids_store=int((ids_s >= 0).sum()),
                ln=ln, gemm_flop_per_layer=sum(2 * E * N * K for _, N, K in GEMMS))
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
        log("shape %s: NEW %s per (layer, KV head, request)" % (name, a.new_blocks[C]))
        try:
            with torch.inference_mode():
                parts, outputs, meta = build_gpu(B, a.new_blocks[C], a.win_share, a.seed + C, a.union_pages)
                times, eg = measure(dev, parts, outputs, a.layers, a.reps, a.warmup)
                ln = meta.pop("ln")
                u_ok = bool((ln == a.union_pages).all())
                gates.append(("%s-UNION" % name, True, u_ok, "union_len min %d max %d (want %d at every (KV head, "
                              "request))" % (int(ln.min()), int(ln.max()), a.union_pages)))
            want = counts_per_layer(B, HKV, a.new_blocks[C], a.win_share)
            gates.append(("%s-IDS" % name, True, (meta["ids_win"], meta["ids_store"]) == want,
                          "NEW blocks per layer window %d store %d (want %s)" % (meta["ids_win"], meta["ids_store"],
                                                                                  want)))
            gates.extend(("%s-%s" % (name, n), m, ok, d) for n, m, ok, d in eg)
            s = summarize(times)
            res[name] = dict(C=C, B=B, meta=meta, times_ms=times, summary=s)
            gates.append(("%s-RUN" % name, True, True, "%d configurations x %d passes of %d layers" % (
                len(CONFIGS), a.reps, a.layers)))
            del parts, outputs
            torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()
            gates.append(("%s-RUN" % name, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    meta = dict(shapes=a.shapes, new_blocks=a.new_blocks, union_pages=a.union_pages, win_share=a.win_share,
                layers=a.layers, reps=a.reps, warmup=a.warmup, seed=a.seed, gemms=GEMMS, fork_launch=FORK_LAUNCH,
                row_shift=ROW_SHIFT, block_bytes=BLOCK_BYTES, nbh=NBH, configs=CONFIGS, rule=dict(reach=REACH,
                                                                                               serial=SERIAL),
                gpu=props.name, torch=torch.__version__, protocol_stamp=PROT.PROTOCOL_STAMP_RECON7,
                base_commit=os.environ.get("BASE_COMMIT"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, shapes=res, gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3])
                                                     for g in gates]), f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-22s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for name, r in res.items():
        s = r["summary"]
        print("R7 D1 %s: medians ms per pass of %d layers %s" % (name, a.layers, s["median_ms"]))
        for c, v in s["concurrent"].items():
            print("R7 D1 %s %s: %s" % (name, c, v))
        print("R7 D1 %s serial %s; interference A|M %s (model s %s); DECISION %s (rule: REACHABLE <= %.2f x max, "
              "SERIAL >= %.2f x sum); bytes per layer %s" % (name, s["serial"], s["interference_AM"], MODEL_S,
                                                             s["decision"], REACH, SERIAL, r["meta"]["bytes_per_layer"]))
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
