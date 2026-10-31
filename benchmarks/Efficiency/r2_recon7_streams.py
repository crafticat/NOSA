"""RECON7 cell X1 (= step cell r7s), R2 discriminator D1 (KS/analysis/plan/RECON7_R2_joint.md section 4; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE CONCURRENCY SCREEN. Two modes (docs/NOTES_recon7.md, section "X1 protocol"):
  (a) CAPABILITY: do INDEPENDENT dense GEMM work (A = Aq + Ar), the fork attention (F), the zero-copy H2D gathers (G)
      and a copy-engine H2D (M) run at the same time on this A100 at our shapes? Each concurrent set is classified
      against its MATCHED SERIAL CONTROL (the same ops, one stream, the same pass); the isolated maxima are an
      IDEALIZED reference only. No dependency between the streams.
  (b) ORDERED SYNTHETIC SCHEDULING SCREEN: the joint pass's per-layer CALL ORDER Aq, G, F, Ar replayed on synthetic,
      INDEPENDENT inputs: F does not read G's destinations or Aq's output, Ar does not read F's output. So (b) is NOT a
      producer / consumer dependency test, NOT an end-to-end bound of the production joint pass, NOT a useful-prefetch
      result. Schedules:
        GRP   request-group split: group 0's ordered calls on one stream, group 1's on another; GRPL: the same, both
              join after every layer. ELIGIBLE for a saving claim only when the groups' outputs equal the full batch's,
              bit for bit (the request-neighbour invariant, gate <shape>-GRP-EQ). The groups' GEMMs read the weights
              twice: nominal 474 MB per layer for the full batch, 948 MB for the two halves (473,956,352 B per layer
              in bf16); the split cost is MEASURED as GRP-SER - SER.
        CPY   a COPY-INTERFERENCE test: P = a copy-engine copy of the current pass's NEW-block byte count from pinned
              staging into its OWN device buffer, overwritten every layer and NEVER read, beside the ordered calls.
        WINF  window attention beside the selected-block fetch: NOT BUILT (the shipped fork's 2D launch writes no
              log-sum-exp, fork :1060-1063): NOT MEASURED.
      Producer / consumer and reader / DMA lifetime validation is a separate, not-built test.
NO NEW KERNEL, NO MODEL: existing ops on SYNTHETIC inputs of the MEASURED shapes (one layer's work, replayed --layers
times = one pass): Aq / Ar the dense GEMMs at M = 4 B rows (cuBLAS through torch.mm: wqkv 4608 x 4096; wo 4096 x 4096,
gate_up 32768 x 4096, down 4096 x 16384; NOSA-8B config.json); F ONE call of the shipped fork (fia6's blob,
verify_rows._fork_call's arguments, FORK_LAUNCH) whose rows list 64 pages shifted (0, 3, 5, 8): union 72 per (KV head,
request); G the shipped zero-copy gathers (window K, V + bias; store K, V + bias) of --new-blocks NEW host blocks per
(layer, KV head, request); M / P the same bytes as ONE copy-engine copy from pinned staging (own buffers each); Aq0 ..
Ar1 the same calls on request group 0 / 1 (half the requests each, own outputs). Selection, residency, the twin step
and the norms are NOT replayed.
CONTROLS: every concurrent set X has X-SER (the same ops serial, one stream); GRP / GRPL have GRP-SER; CPY has CPY-SER.
A control is CONSISTENT when it is within CTL_TOL (5 %) of its independent sum (X-SER: the sum of the parts alone;
GRP-SER: GRP0 + GRP1; CPY-SER: SER + P). An inconsistent control INVALIDATES its conclusion. All configurations run in
one process, ORDER-ROTATED (lesson L62: pass r starts at configuration r mod n) after a warm-up of each; every timed
pass and every warm-up pass is kept in the record.
MUST GATES: <shape>-RUN; <shape>-<config>-EXACT (every part's output bits after the configuration's last pass = the
part's reference run alone); <shape>-UNION (fork union_len = 72 everywhere); <shape>-IDS (the NEW blocks per layer as
requested); <shape>-BUDGET (the shape's TRUE wall clock, setup to summary, <= --max-shape-s); RUN-WALL (the run's wall
clock <= --max-run-s). DECISION GATE: <shape>-GRP-EQ (governs the GRP / GRPL claim only). Any failed MUST gate
SUPPRESSES every decision of the run (the timings stay in the record).
STOP THE RUN (the record says why; no later shape runs): free device memory below the need; a UNION or IDS failure;
an EXACT failure; the shape's deadline passed in setup, references, warm-up or timing; an exception.
EXIT: the number of failed MUST gates (0-19); 20 refused (cell r7s not granted: r2_recon7_protocol.py); 30 an exception
outside a shape.
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
WEIGHT_BYTES = sum(N * K for _, N, K in GEMMS) * 2              # 473,956,352 B per layer (bf16)
FORK_LAUNCH = dict(num_warps=4, num_stages=3, tile_size=64)   # = nosi/nosi/verify_rows.py FORK_LAUNCH (test checks)
FORK_ROWS = 4                                                 # = verify_rows.FORK_ROWS
ROW_SHIFT = (0, 3, 5, 8)                                      # union per (KV head, request) = 64 + 8 = 72
BLOCK_BYTES = BS * HD * 2 * 2 + BS * 2                        # K + V + bias of one (KV head, block): 32,896 B
NBH = 16                                                      # host blocks per request (the ids' range)
CHAIN = ["Aq", "G", "F", "Ar"]                                # the joint pass's per-layer CALL order (no data flow)
CHAIN0, CHAIN1 = [p + "0" for p in CHAIN], [p + "1" for p in CHAIN]
PARTS = ("Aq", "Ar", "F", "G", "M", "P") + tuple(CHAIN0) + tuple(CHAIN1)
ALONE = dict(A=["Aq", "Ar"], F=["F"], G=["G"], M=["M"], FG=["G", "F"], P=["P"])
CONCURRENT = {"A|F": ("A", "F"), "A|G": ("A", "G"), "A|M": ("A", "M"), "F|M": ("F", "M"), "A|FG": ("A", "FG"),
              "A|F|M": ("A", "F", "M")}
# (name, mode, stream lists, lock-step). Mode "a": CAPABILITY (independent streams); "b": the ORDERED SCREEN; "c": a
# serial CONTROL or reference (one stream).
CONFIGS = (tuple((k, "a", [v], False) for k, v in ALONE.items())
           + tuple(("%s-SER" % c, "c", [[p for x in xs for p in ALONE[x]]], False) for c, xs in CONCURRENT.items())
           + tuple((c, "a", [ALONE[x] for x in xs], False) for c, xs in CONCURRENT.items())
           + (("SER", "c", [CHAIN], False), ("GRP0", "c", [CHAIN0], False), ("GRP1", "c", [CHAIN1], False),
              ("GRP-SER", "c", [CHAIN0 + CHAIN1], False), ("GRP", "b", [CHAIN0, CHAIN1], False),
              ("GRPL", "b", [CHAIN0, CHAIN1], True),
              ("CPY-SER", "c", [CHAIN + ["P"]], False), ("CPY", "b", [CHAIN, ["P"]], False)))
CONTROL_OF = dict({c: "%s-SER" % c for c in CONCURRENT}, GRP="GRP-SER", GRPL="GRP-SER", CPY="CPY-SER")
# a control's independent sum (the consistency check): {control: [the configurations whose medians add up]}
CONTROL_SUM = dict({"%s-SER" % c: list(xs) for c, xs in CONCURRENT.items()}, **{"GRP-SER": ["GRP0", "GRP1"],
                                                                              "CPY-SER": ["SER", "P"]})
SCOPE = dict(
    GRP="request groups: group g's rows, fork lists, union tables and gather destinations are written and read by "
        "group g's stream only; shared read-only: the weights, the arena (disjoint pages per request), the pinned host "
        "KV (disjoint requests). Inputs are synthetic and independent between calls: an ORDERED-CALL screen, not a "
        "dependency test",
    CPY="a copy of the current pass's NEW-block byte count into an own device buffer, overwritten every layer, never "
        "read: COPY INTERFERENCE beside the ordered calls, not a useful prefetch",
    WINF="NOT MEASURED: needs the window part's log-sum-exp (the fork's 2D launch has none) or a non-bit-equal split")
MODEL_S = 0.0868458712                                        # the model's transport interference (REQ: 8.68458712 %)
REACH, SERIAL = 1.10, 0.90                                    # mode (a) rule
SAVING, ABSORB_OK, SLOW_OK = 0.05, 0.80, 0.05                 # mode (b) rules
CTL_TOL = 0.05                                                # a control within 5 % of its independent sum


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
    ap.add_argument("--cell", default=None, help="r7s: the granted cell (r2_recon7_protocol.py)")
    ap.add_argument("--shapes", default="128:216,73:318", help="C:B pairs (the measured configurations)")
    ap.add_argument("--new-blocks", default="128:1.226,73:5.0", help="C:NEW host blocks per (layer, KV head, request)")
    ap.add_argument("--union-pages", type=int, default=72, choices=(72,), help="the fork's union per (KV head, request)")
    ap.add_argument("--win-share", type=float, default=0.5, help="the share of the NEW blocks that are window loads")
    ap.add_argument("--layers", type=int, default=32, help="layers replayed per pass")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-shape-s", type=float, default=240.0, help="STOP: a shape's wall clock above this (s)")
    ap.add_argument("--max-run-s", type=float, default=600.0, help="RUN-WALL: the run's wall clock limit (s)")
    a = ap.parse_args(argv)
    a.shapes = _pairs(ap, a.shapes, int, "--shapes")
    a.new_blocks = _pairs(ap, a.new_blocks, float, "--new-blocks")
    if not a.shapes or set(a.shapes) != set(a.new_blocks):
        ap.error("--shapes and --new-blocks: the same C values")
    if not (0.0 <= a.win_share <= 1.0) or a.layers < 1 or a.reps < 3 or a.warmup < 1 or a.max_shape_s <= 0 or \
            a.max_run_s <= 0:
        ap.error("--win-share in [0, 1], --layers >= 1, --reps >= 3, --warmup >= 1, --max-shape-s > 0, --max-run-s > 0")
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


def stats(v):
    """n, median, min, max, quartiles and the coefficient of variation of one configuration's passes (ms)."""
    if not v:
        return None
    q = statistics.quantiles(v, n=4) if len(v) >= 2 else [v[0]] * 3
    mean = statistics.fmean(v)
    return dict(n=len(v), median=round(statistics.median(v), 4), min=round(min(v), 4), max=round(max(v), 4),
                q1=round(q[0], 4), q3=round(q[2], 4),
                cv=(round(statistics.pstdev(v) / mean, 4) if mean > 0 and len(v) >= 2 else None))


def control_check(med: dict, ctl: str):
    """(consistent?, the control's deviation from its independent sum, the sum) for control `ctl`."""
    parts = [med.get(x) for x in CONTROL_SUM[ctl]]
    c = med.get(ctl)
    if c is None or any(x is None for x in parts) or not sum(parts):
        return False, None, None
    s = sum(parts)
    dev = c / s - 1.0
    return abs(dev) <= CTL_TOL, round(dev, 4), round(s, 4)


def summarize(times: dict, eligible=None, suppressed=None) -> dict:
    """times {config: [ms per pass]} -> the per-configuration statistics and the decisions.
    (a) every concurrent set against its MATCHED control X-SER (eff = (control - T) / (control - max)) and against the
    isolated maxima (IDEALIZED reference); the A|F|M decision; the interference A|M. (b) GRP / GRPL against GRP-SER and
    the full-batch SER (eligible only with the request-neighbour invariant: eligible["GRP"]); CPY against CPY-SER.
    An inconsistent control (CTL_TOL) INVALIDATES its conclusion; `suppressed` (the failed MUST gates) replaces every
    decision by SUPPRESSED."""
    eligible = eligible or {}
    st = {k: stats(v) for k, v in times.items()}
    med = {k: (s["median"] if s else None) for k, s in st.items()}
    out = dict(stats_ms=st, median_ms=med)
    ctl = {c: dict(zip(("consistent", "dev_from_sum", "sum_ms"), control_check(med, c))) for c in CONTROL_SUM}
    out["controls"] = ctl
    conc = {}
    for c, xs in CONCURRENT.items():
        t, al, cn = med.get(c), [med.get(x) for x in xs], CONTROL_OF[c]
        cm = med.get(cn)
        if t is None or cm is None or any(x is None for x in al):
            conc[c] = None
            continue
        mx, sm = max(al), sum(al)
        conc[c] = dict(ms=t, parts=list(xs), alone=al, max=round(mx, 4), sum=round(sm, 4), control=cn, control_ms=cm,
                       control_consistent=ctl[cn]["consistent"], over_control=round(t / cm, 4),
                       eff_matched=(round((cm - t) / (cm - mx), 4) if cm > mx else None),
                       over_max_ideal=round(t / mx, 4))
    out["concurrent"] = conc
    fm = conc.get("A|F|M")
    if fm is None:
        cap = None
    elif not fm["control_consistent"]:
        cap = "INVALID: the matched control A|F|M-SER is %s from the sum of the parts alone (limit %.0f %%)" % (
            ctl["A|F|M-SER"]["dev_from_sum"], CTL_TOL * 100)
    elif fm["over_max_ideal"] <= REACH:
        cap = "REACHABLE"
    elif fm["over_control"] >= SERIAL:
        cap = "SERIAL"
    else:
        cap = "PARTIAL"
    out["capability"] = cap
    am = conc.get("A|M")
    out["interference_AM"] = (round(am["over_max_ideal"] - 1.0, 4) if am and am["control_consistent"] else None)
    out["model_s"] = MODEL_S
    scr, ser = {}, med.get("SER")
    for c in ("GRP", "GRPL"):
        t, cm = med.get(c), med.get("GRP-SER")
        if None in (t, cm, ser):
            scr[c] = None
            continue
        net = ser - t
        d = dict(ms=t, control_ms=cm, ser_ms=ser, split_cost_ms=round(cm - ser, 4), overlap_ms=round(cm - t, 4),
                 net_ms=round(net, 4), net_frac=round(net / ser, 4), control_consistent=ctl["GRP-SER"]["consistent"],
                 eligible=bool(eligible.get("GRP")))
        if not d["eligible"]:
            d["decision"] = "INELIGIBLE: the groups' outputs differ from the full batch's (request-neighbour invariant)"
        elif not d["control_consistent"]:
            d["decision"] = "INVALID: GRP-SER is %s from GRP0 + GRP1" % ctl["GRP-SER"]["dev_from_sum"]
        else:
            d["decision"] = ("SCREEN SAVING >= %.0f %% (ordered synthetic calls only)" % (SAVING * 100)
                             if net / ser >= SAVING else "NO SCREEN SAVING")
        scr[c] = d
    t, cm, p = med.get("CPY"), med.get("CPY-SER"), med.get("P")
    if None in (t, cm, ser, p) or not p:
        scr["CPY"] = None
    else:
        ab, slow = (ser + p - t) / p, t / ser - 1.0
        d = dict(ms=t, control_ms=cm, ser_ms=ser, P_ms=p, absorbed_frac=round(ab, 4), chain_slowdown=round(slow, 4),
                 control_consistent=ctl["CPY-SER"]["consistent"])
        if not d["control_consistent"]:
            d["decision"] = "INVALID: CPY-SER is %s from SER + P" % ctl["CPY-SER"]["dev_from_sum"]
        else:
            d["decision"] = ("COPY ABSORBED (copy interference only, not a prefetch)"
                             if ab >= ABSORB_OK and slow <= SLOW_OK else "COPY NOT ABSORBED")
        scr["CPY"] = d
    scr["WINF"] = SCOPE["WINF"]
    out["ordered_screen"] = scr
    if suppressed:
        why = "SUPPRESSED: MUST gate(s) failed: %s" % list(suppressed)[:6]
        out["capability"] = why if out["capability"] is not None else None
        out["interference_AM"] = None
        for c in ("GRP", "GRPL", "CPY"):
            if scr.get(c):
                scr[c]["decision"] = why
    return out


def exact_gate(name: str, ref: dict, got: dict, bits):
    """<name>-EXACT (MUST): every part output of the configuration = its reference output, bit for bit."""
    bad = [k for k in got if k not in ref or not (got[k].shape == ref[k].shape and bool((bits(got[k]) ==
                                                                                       bits(ref[k])).all()))]
    return ("%s-EXACT" % name, True, bool(got) and not bad, "%d outputs compared; unequal %s" % (len(got), bad[:4]))


def group_equal(outputs, bits):
    """{part: the groups' outputs, concatenated by request, = the full batch's, bit for bit} for Aq, Ar, F, G."""
    import torch
    out = {}
    for p in ("Aq", "Ar", "F", "G"):
        ok = True
        for x, x0, x1 in zip(outputs[p], outputs[p + "0"], outputs[p + "1"]):
            y = torch.cat((x0, x1), 0)
            ok = ok and y.shape == x.shape and bool((bits(y) == bits(x)).all())
        out[p] = ok
    return out


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


def measure(dev, parts, outputs, layers: int, reps: int, warmup: int, configs=CONFIGS, max_s=None, clock=None, t0=None):
    """The reference outputs (each part alone, one layer); a warm-up of every configuration; then `reps` ORDER-ROTATED
    passes (pass r starts at configuration r mod n); after a configuration's last pass its EXACT gate. The DEADLINE
    (t0 + max_s on `clock`; t0 = the shape's start) is checked after every reference, every warm-up and every timed
    pass. STOP: an EXACT failure or the deadline. Returns (times, gates, stop, warm)."""
    clock = clock or time.perf_counter
    t0 = clock() if t0 is None else t0

    def late(where):
        dt = clock() - t0
        return ("wall clock %.0f s above %.0f s (%s)" % (dt, max_s, where)) if max_s is not None and dt > max_s else None
    side = [dev.stream() for _ in range(3)]
    spec = {c[0]: c for c in configs}
    names = [c[0] for c in configs]
    times, warm, gates = {n: [] for n in names}, {n: [] for n in names}, []
    ref = {}
    for p in [p for p in PARTS if p in parts]:
        run_config(dev, parts, [[p]], 1, side)
        for i, t in enumerate(outputs[p]):
            ref["%s.%d" % (p, i)] = t.clone()
        stop = late("references")
        if stop:
            return times, gates, stop, warm
    for name in names:
        for _ in range(warmup):
            warm[name].append(run_config(dev, parts, spec[name][2], layers, side, spec[name][3]))
        stop = late("warm-up")
        if stop:
            return times, gates, stop, warm
    stop = None
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
            stop = late("pass %d" % r)
            if stop:
                break
        if stop:
            break
    return times, gates, stop, warm


def need_bytes(B: int, union: int = 72) -> int:
    """MODEL: the device bytes of one shape (weights, activations, the outputs and the gather destinations four times:
    full and groups, each with its reference clone; the arena; the copy buffers; 1 GiB slack): the free-memory STOP."""
    E = R * B
    w = WEIGHT_BYTES
    act = E * (3 * HIDDEN + INTER) * 2
    outs = 4 * E * sum(N for _, N, _ in GEMMS) * 2
    arena = (B * union + 8) * BS * HKV * (2 * HD + 1) * 2
    dst = 4 * B * (TOPK + X_STORE) * BS * HKV * (2 * HD + 1) * 2
    return w + act + outs + arena + dst + 2 * E * HQ * HD * 2 + (1 << 30)


def run_shapes(a, dev, build, free_bytes, sync, clock=time.perf_counter, ctx=contextlib.nullcontext, cleanup=None):
    """The shapes in order (largest C first). Every failure STOPS THE RUN (no later shape). Returns (gates, raw):
    gates (name, must, ok, detail); raw {shape: the meta, every pass, every warm-up, the stop, the wall, the
    group-equality}."""
    gates, raw = [], {}
    t_run = clock()
    for C, B in sorted(a.shapes.items(), key=lambda kv: -kv[0]):
        name = "C%d_B%d" % (C, B)
        t0 = clock()
        free, need = free_bytes(), need_bytes(B, a.union_pages)
        if free < need:
            gates.append(("%s-RUN" % name, True, False, "STOP THE RUN: free %.2f GB below the need %.2f GB (MODEL)" % (
                free / 1e9, need / 1e9)))
            raw[name] = dict(C=C, B=B, stop="free memory", free_bytes=free, need_bytes=need)
            break
        log("shape %s: NEW %s per (layer, KV head, request); need %.2f GB of %.2f GB free" % (
            name, a.new_blocks[C], need / 1e9, free / 1e9))
        stop = None
        try:
            with ctx():
                parts, outputs, meta = build(B, a.new_blocks[C], a.win_share, a.seed + C, a.union_pages)
                for p in ("F", "F0", "F1"):                            # build the union tables once
                    parts[p]()
                sync()
                lns = meta.pop("ln")
                u_ok = all(bool((x == a.union_pages).all()) for x in lns)
                gates.append(("%s-UNION" % name, True, u_ok, "union_len min %s max %s (want %d at every (KV head, "
                              "request), full and both groups)" % ([int(x.min()) for x in lns],
                                                                  [int(x.max()) for x in lns], a.union_pages)))
                want = counts_per_layer(B, HKV, a.new_blocks[C], a.win_share)
                i_ok = (meta["ids_win"], meta["ids_store"]) == want
                gates.append(("%s-IDS" % name, True, i_ok, "NEW blocks per layer window %d store %d (want %s)" % (
                    meta["ids_win"], meta["ids_store"], want)))
                setup_s = clock() - t0
                if not (u_ok and i_ok):
                    stop = "UNION or IDS failed before the timing"
                elif setup_s > a.max_shape_s:
                    stop = "wall clock %.0f s above %.0f s (setup)" % (setup_s, a.max_shape_s)
                times, eg, warm, same = {}, [], {}, {}
                if stop is None:
                    times, eg, stop, warm = measure(dev, parts, outputs, a.layers, a.reps, a.warmup, max_s=a.max_shape_s,
                                                    clock=clock, t0=t0)
                    same = group_equal(outputs, bits)
            gates.extend(("%s-%s" % (name, n), m, ok, d) for n, m, ok, d in eg)   # the shape names every gate
            if same:
                gates.append(("%s-GRP-EQ" % name, False, all(same.values()),
                              "DECISION GATE (GRP / GRPL only): the groups' outputs = the full batch's bits: %s" % same))
            wall = clock() - t0
            gates.append(("%s-BUDGET" % name, True, wall <= a.max_shape_s,
                          "wall %.1f s from the shape's start to its last gate (limit %.0f s); setup %.1f s; stop %s" % (
                              wall, a.max_shape_s, setup_s, stop)))
            raw[name] = dict(C=C, B=B, meta=meta, times_ms=times, warmup_ms=warm, stop=stop, wall_s=round(wall, 1),
                             setup_s=round(setup_s, 1), groups_equal_full=same)
            gates.append(("%s-RUN" % name, True, stop is None and wall <= a.max_shape_s,
                          "%d configurations x %d order-rotated passes of %d layers; stop %s" % (
                              len(CONFIGS), a.reps, a.layers, stop)))
            del parts, outputs
            if cleanup is not None:
                cleanup()
            if stop is not None or wall > a.max_shape_s:
                break
        except Exception as e:
            traceback.print_exc()
            gates.append(("%s-RUN" % name, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            raw[name] = dict(C=C, B=B, stop="exception %s" % type(e).__name__)
            break
    run_wall = clock() - t_run
    gates.append(("RUN-WALL", True, run_wall <= a.max_run_s, "run wall %.1f s (limit %.0f s)" % (run_wall, a.max_run_s)))
    return gates, raw


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
        """The ordered calls of the requests [b0, b1) (rows [R b0, R b1)), suffix tag."""
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
                gemm_flop_per_layer=sum(2 * E * N * K for _, N, K in GEMMS), weight_bytes_per_layer=WEIGHT_BYTES,
                seed=seed, scope=SCOPE)
    return parts, outputs, meta


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    why = PROT.refusal(a.cell, "r2_recon7_streams", a)
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
    props = torch.cuda.get_device_properties(0)
    gates, raw = run_shapes(a, CudaDev(), build_gpu, lambda: torch.cuda.mem_get_info()[0], torch.cuda.synchronize,
                            ctx=torch.inference_mode, cleanup=torch.cuda.empty_cache)
    failed = [g[0] for g in gates if g[1] and not g[2]]
    for name, r in raw.items():
        if "times_ms" in r:
            r["summary"] = summarize(r["times_ms"], eligible=dict(GRP=bool(r["groups_equal_full"]) and all(
                r["groups_equal_full"].values())), suppressed=failed)
    meta = dict(cell=a.cell, grant=PROT.grant(a.cell), shapes=a.shapes, new_blocks=a.new_blocks,
                union_pages=a.union_pages, win_share=a.win_share, layers=a.layers, reps=a.reps, warmup=a.warmup,
                seed=a.seed, max_shape_s=a.max_shape_s, max_run_s=a.max_run_s, gemms=GEMMS, weight_bytes=WEIGHT_BYTES,
                fork_launch=FORK_LAUNCH, row_shift=ROW_SHIFT, block_bytes=BLOCK_BYTES, nbh=NBH, configs=CONFIGS,
                control_sum=CONTROL_SUM, scope=SCOPE, rules=dict(reach=REACH, serial=SERIAL, saving=SAVING,
                                                                 absorb=ABSORB_OK, slow=SLOW_OK, ctl_tol=CTL_TOL),
                gpu=props.name, torch=torch.__version__, base_commit=os.environ.get("BASE_COMMIT"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, shapes=raw, gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3])
                                                     for g in gates]), f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-24s %s %s: %s" % (g[0], "MUST" if g[1] else "DECISION", "PASS" if g[2] else "FAIL", g[3]))
    for name, r in raw.items():
        s = r.get("summary")
        if s is None:
            print("X1 %s: no timing (stop %s)" % (name, r.get("stop")))
            continue
        print("X1 %s medians ms per pass of %d layers: %s" % (name, a.layers, s["median_ms"]))
        for c, v in s["controls"].items():
            print("X1 CONTROL %s %s: %s" % (name, c, v))
        for c, v in s["concurrent"].items():
            print("X1 (a) CAPABILITY %s %s: %s" % (name, c, v))
        print("X1 (a) CAPABILITY %s: interference A|M %s (model s %s); DECISION %s" % (
            name, s["interference_AM"], MODEL_S, s["capability"]))
        for c, v in s["ordered_screen"].items():
            print("X1 (b) ORDERED SCREEN %s %s: %s" % (name, c, v))
        print("X1 %s groups' outputs = the full batch's bits %s; wall %s s; stop %s" % (
            name, r["groups_equal_full"], r["wall_s"], r["stop"]))
    print("r2_recon7_streams: %d MUST gate(s) failed%s" % (len(failed), (": decisions SUPPRESSED %s" % failed[:6])
                                                          if failed else ""))
    return min(len(failed), 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
