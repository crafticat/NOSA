"""R2 step 7, (B2) B2-T: the COMPONENT MICROBENCH of the array-backed ownership state (docs/R2_OWN2_B2_DESIGN.md rev 2,
section 5). REPORT only. No model; the records have the shapes of C's note 12l; the same ops on CPU or CUDA tensors.

    python benchmarks/Efficiency/r2_own2_b2t.py --out <json> [--device cpu|cuda] [--threads 1] [--iters 20]
        [--warmup 3] [--pipelined-iters N] [--profile] [--cpu-reference] [--smoke]

--device cuda (refused, exit 2, when CUDA is absent): the world and the records are built on the host, then moved to the
device; every component stop SYNCHRONIZES the device (its own wall time) and a CUDA event pair records the stream's
span; a PIPELINED pass then times whole ticks with no component synchronization (only the ops' own host reads: the
real cost); --profile adds one tick under torch.profiler with a record_function range per component (device busy time,
kernel launches, synchronizations); --cpu-reference repeats the timing on the host of the same node. The REGISTERED
DEVICE PREDICTIONS (REGISTERED_DEVICE, rev 2 section 5.1, frozen before the first GPU run) are scored in the JSON.

THE WORLD (synthetic, seeded) at p3's geometry (L 32, B 64, H 2, C 96, X 32, R_ver 16, NBLK 256, k 64) and p3's
counts per round tick (R2_SG5_CAPTURE_PLAN.md:264-271: 1,054 urgent of which 467 Policy E, 64 verify merges, 6,493
draft takes, 1,269 refusals) plus two ASSUMED counts that p3 did not record (--d-infl 3000 draft in-flight entries,
--acquire 1024 PO / VRES acquisitions):
  - rows: 51 verify rows from 13 cohorts (one b and key per cohort, up to --cohort-rows rows, core.py:146-150) and 55
    draft rows (one per request, core.py:157-164);
  - streams: a NORMAL stream holds C - 13 LRU entries, 2 PO, 1 VRES, 4 in flight (two due at this poll: one draft
    demand, one urgent verify landed in place; two issued at t - 1), 2 RELEASING (one reader done, one pending);
    per layer X / 2 BORROWING streams hold C LRU + 2 PO (the layer's extras all borrowed: K5 sends their urgent misses
    to Policy E and refuses their draft misses);
  - pins: odd-recency LRU entries, PO, VRES and the urgent block by key slot 0; one PO per normal stream only by key
    slot KEYS - 1 (released at this poll at half the (l, b));
  - readers: event ids t x L + l; ticks <= t - 2 complete, t - 1 pending.
The records are generated against the POLLED state (the bench's own poll), so the apply sees what a device would
decide.

EACH ITERATION restores the world (untimed) and times every component in apply_tick's order:
  poll: refresh, force_seq, arrivals (+ K1 over C), post-poll I1 / I2, key drops, releases;
  apply: parse, row checks, slot_of, pin words, K5 events, K5 / first_free, take checks, READY, merges, Policy E,
  gen snapshot, verify pins, draft pins, takes, merges, recency rounds (+ evict_to_c), post checks, readers, metrics,
  E's issue_desc arguments.
The predicates are read in TWO host reads per apply (rev 2 section 3.3): the pre-write flags (a set flag voids the tick
before any write) and the post-write flags (a set flag rolls the undo log back).
JSON: per component median / p10 / p90 / mean ms (and CUDA event ms); the poll / apply / synced and pipelined totals;
the generated counts; the predicate violations (a coherence diagnostic of the synthetic world: 0 expected); the
REGISTERED FALSIFIERS: on the host (rev 2 section 5) the median total > 25 ms with one thread refutes the host estimate
(job 2191023: 164.364 ms, REFUTED); on the device REGISTERED_DEVICE (D1 the pipelined median, D2 the synced total, D3
the profiled device busy time). Last stdout line:
  R2_OWN2_B2T_DONE failed=<n> verdict=<...> total_ms=<x> device=<cpu|cuda>
exit = n (a generated count that differs from the requested one, or a predicate that fired, each counts 1); 2 refused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import types
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NOSI = os.path.join(ROOT, "nosi", "nosi")
PKG = "b2tnosi"
FALSIFIER_MS = 25.0
# Frozen before the first GPU run (rev 2 section 5.1). The model: the device ops are LAUNCH-BOUND: about 800 kernel
# launches (500-1200) and about 60 host synchronizations (35-120) per round tick (poll + apply), at 6-15 us per launch
# and 15-40 us per synchronization; the GPU's own work (about 30 full-table passes of 0.5 M slots plus the record ops)
# is 1.5-4 ms of kernel time.
# The UNCHECKED mode (C's request: the verification-only components gated as K3X's checked mode) drops about 330
# launches and 10 synchronizations: about 470 launches and 48 synchronizations.
REGISTERED_DEVICE = dict(
    D1_pipelined_ms=dict(point=9.0, band=(5.0, 18.0), falsified_above=18.0),    # THE registered falsifier (checked)
    D1u_pipelined_unchecked_ms=dict(point=5.5, band=(3.0, 11.0), falsified_above=11.0),
    D2_synced_ms=dict(point=10.0, band=(5.0, 20.0), falsified_above=20.0),
    D3_busy_ms=dict(point=3.0, band=(1.5, 6.0), falsified_above=6.0),
    D4_launches=dict(point=800, band=(500, 1200)),
    D5_syncs=dict(point=60, band=(35, 120)),
)
# components with a Python loop (rounds, live events, layer registrations) or a host read (nonzero, a boolean mask,
# unique, int(max), bool(any), the two flag reads): expected HOST-BOUND on the device (D6: every one with wall >= 0.2 ms
# has profiled device time <= 0.5 x its wall)
HOST_BOUND = ("poll.refresh", "poll.force_seq", "poll.arrivals", "poll.checks", "poll.key_drops", "poll.releases",
              "apply.parse", "apply.k5_events", "apply.merge_check", "apply.policy_e", "apply.pre_flags",
              "apply.pins", "apply.takes", "apply.merges", "apply.recency", "apply.post_checks", "apply.post_flags",
              "apply.readers", "apply.metrics", "apply.issue_desc")
TENSOR_ONLY = ("apply.row_checks", "apply.slot_of", "apply.pin_words", "apply.k5_first_free", "apply.take_checks",
               "apply.ready_check", "apply.gen_snapshot")
# verification only: skipped in the UNCHECKED mode (the next state does not depend on them)
CHECK_ONLY = ("poll.checks", "apply.row_checks", "apply.pin_words", "apply.k5_first_free", "apply.take_checks",
              "apply.ready_check", "apply.policy_e", "apply.pre_flags", "apply.gen_snapshot", "apply.post_checks",
              "apply.post_flags")


def _load(mod: str):
    """nosi/nosi/<mod>.py under a stand-in package (the package __init__ imports the models; not needed here)."""
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [NOSI]
        sys.modules[PKG] = pkg
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, os.path.join(NOSI, mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


OPS = _load("r2_own2_ops")
SP = sys.modules[PKG + ".r2_residency_sparse"]
C_FREE, C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING = (OPS.C_FREE, OPS.C_INFLIGHT, OPS.C_LRU, OPS.C_PO, OPS.C_VRES,
                                                        OPS.C_RELEASING)
BIG = OPS.BIG


def wh(c, x, y):
    """torch.where with plain ints as 0-dim int64 tensors (no reliance on the torch version's scalar support)."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32 if isinstance(x, float) else torch.int64)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32 if isinstance(y, float) else torch.int64)
    return torch.where(c, x, y)


# ===================================================================================================================
# arguments
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="a small world (CPU tests): L 2, B 8, C 24, X 8, k 8")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--mode", choices=("checked", "unchecked", "both"), default="both",
                    help="checked: every predicate (two flag reads per apply); unchecked: the verification-only "
                         "components skipped (CHECK_ONLY), the device's decisions trusted")
    ap.add_argument("--pipelined-iters", type=int, default=-1, help="whole ticks without component syncs (-1: --iters)")
    ap.add_argument("--profile", action="store_true", help="one tick under torch.profiler (record_function ranges)")
    ap.add_argument("--cpu-reference", action="store_true", help="with --device cuda: also time on this node's host")
    for n, v in (("L", 32), ("B", 64), ("H", 2), ("C", 96), ("X", 32), ("R-ver", 16), ("nblk", 256), ("keys", 4),
                 ("k", 64), ("verify-rows", 51), ("cohort-rows", 4), ("draft-rows", 55), ("lru-gap", 13),
                 ("urgent", 1054), ("policy-e", 467), ("v-merge", 64), ("d-take", 6493), ("refused", 1269),
                 ("d-infl", 3000), ("acquire", 1024), ("tick", 100), ("lag", 2)):
        ap.add_argument("--" + n, type=int, default=v)
    a = ap.parse_args(argv)
    if a.smoke:
        for n, v in (("L", 2), ("B", 8), ("C", 24), ("X", 8), ("R_ver", 2), ("nblk", 64), ("k", 8),
                     ("verify_rows", 5), ("cohort_rows", 2), ("draft_rows", 6), ("urgent", 6), ("policy_e", 3),
                     ("v_merge", 2), ("d_take", 8), ("refused", 4), ("d_infl", 3), ("acquire", 4)):
            setattr(a, n, v)
    return a


def geo_of(a) -> "OPS.Geo":
    return OPS.Geo(L=a.L, B=a.B, H=a.H, C=a.C, X=a.X, R_ver=a.R_ver, NBLK=a.nblk, KEYS=a.keys)


# ===================================================================================================================
# the world
class Ev:
    def __init__(self, done: bool):
        self.done = done


def ev_done(ev) -> bool:
    return True if ev is None else bool(ev.done)


def distribute(total: int, eligible: torch.Tensor, cap: torch.Tensor, g: torch.Generator, what: str) -> torch.Tensor:
    """Integer counts (one per stream) that sum to `total` over the eligible streams, each <= cap (round robin in a
    seeded random order). Raises when the world cannot hold them."""
    cap = wh(eligible, cap.clamp(min=0), torch.zeros_like(cap))
    if int(cap.sum()) < total:
        raise ValueError("world too small for %s: %d requested, room for %d" % (what, total, int(cap.sum())))
    out = torch.zeros_like(cap)
    left = int(total)
    while left > 0:
        idx = (out < cap).nonzero().squeeze(1)
        idx = idx[torch.randperm(idx.numel(), generator=g)][:left]
        out[idx] += 1
        left -= int(idx.numel())
    return out


def build_world(a) -> Dict:
    """The pre-poll world (module docstring): the Table, the reader window's events, the rows, the stream roles."""
    G = geo_of(a)
    g = torch.Generator().manual_seed(a.seed)
    L, B, H, C, X, NBLK, KEYS = G.L, G.B, G.H, G.C, G.X, G.NBLK, G.KEYS
    NPOS, S, t = G.NPOS, G.S, a.tick
    T = OPS.Table(G)
    T.owner.copy_(torch.arange(B, dtype=torch.int32) + 1000)
    # rows: cohorts of verify rows, one draft row per drafting request
    perm = torch.randperm(B, generator=g)
    n_coh = (a.verify_rows + a.cohort_rows - 1) // a.cohort_rows
    vreq, dreq = perm[:n_coh], perm[B - a.draft_rows:]
    sizes = [a.cohort_rows] * n_coh
    sizes[-1] = a.verify_rows - a.cohort_rows * (n_coh - 1)
    vb = torch.repeat_interleave(vreq, torch.tensor(sizes))
    rows = OPS.Rows(vb=vb, vk=torch.full_like(vb, 1), db=dreq.clone(), dk=torch.full_like(dreq, 2), k=a.k)
    s_all = torch.arange(S)
    b_of = torch.div(s_all, H, rounding_mode="floor").remainder(B)
    l_of = torch.div(s_all, B * H, rounding_mode="floor")
    inV, inD = torch.isin(b_of, vreq), torch.isin(b_of, dreq)
    # borrowing streams: per layer X / 2 (each borrows 2 extras): half among the verify streams, the rest among the
    # draft-only streams
    bor = torch.zeros(S, dtype=torch.bool)
    for l in range(L):
        vs = ((l_of == l) & inV).nonzero().squeeze(1)
        ds = ((l_of == l) & inD & ~inV).nonzero().squeeze(1)
        nbv = min(X // 4, vs.numel())
        nbd = min(X // 2 - nbv, ds.numel())
        bor[vs[torch.randperm(vs.numel(), generator=g)[:nbv]]] = True
        bor[ds[torch.randperm(ds.numel(), generator=g)[:nbd]]] = True
        if nbv + nbd != X // 2:
            raise ValueError("layer %d: %d borrowing streams, want %d" % (l, nbv + nbd, X // 2))
    # categories per stream in slot_perm order: LRU, PO, VRES, INFLIGHT, RELEASING, FREE
    n_lru = wh(bor, torch.full((S,), C), torch.full((S,), C - a.lru_gap))
    n_cat = torch.stack([n_lru, torch.full((S,), 2), wh(bor, 0, 1), wh(bor, 0, 4),
                         wh(bor, 0, 2)], 1)
    if bool((n_lru < a.k).any()) or C - a.lru_gap < 0:
        raise ValueError("lru-gap too large for k")
    cum = torch.cumsum(n_cat, 1)
    jj = torch.arange(NPOS).view(1, -1)
    cat = (jj >= cum[:, 0:1]).long() + (jj >= cum[:, 1:2]).long() + (jj >= cum[:, 2:3]).long() \
        + (jj >= cum[:, 3:4]).long() + (jj >= cum[:, 4:5]).long()
    start = torch.cat([torch.zeros(S, 1, dtype=torch.long), cum], 1)
    pin_cat = jj - start.gather(1, cat)                                  # the position inside its category
    codes = torch.tensor([C_LRU, C_PO, C_VRES, C_INFLIGHT, C_RELEASING, C_FREE])
    slot_perm = torch.argsort(torch.rand(S, NPOS, generator=g), 1)
    blk_perm = torch.argsort(torch.rand(S, NBLK, generator=g), 1)
    used = cat < 5
    blk_pos = wh(used, blk_perm[:, :NPOS], torch.full_like(jj.expand(S, -1), -1))
    ids = lambda tt: tt * L + l_of.view(-1, 1)                          # noqa: E731  reader id of tick tt, layer l
    own = T.owner[b_of].view(-1, 1).to(torch.int64)
    old_owner = (torch.rand(S, 1, generator=g) < 0.05) & ~bor.view(-1, 1)
    is_lru, is_po, is_vres, is_if, is_rel = (cat == 0), (cat == 1), (cat == 2), (cat == 3), (cat == 4)
    col = {}
    col["code"] = codes[cat]
    col["block"] = blk_pos
    col["gen"] = torch.randint(1, 50, (S, NPOS), generator=g)
    col["uid"] = wh(used, own, torch.full_like(own, -1))
    col["uid"] = wh(is_if & (pin_cat == 0) & old_owner, own + 5000, col["uid"])
    col["cls"] = wh(used, wh(is_vres, OPS.CLS_V, OPS.CLS_AD), 0)
    col["stamp"] = wh(is_lru, 1000 + pin_cat, wh(is_if, 100000 + pin_cat, 0))
    rd = wh(is_rel, wh(pin_cat == 0, ids(t - 3), ids(t - 1)),
                     wh(is_if, ids(t - 5), wh(pin_cat % 8 == 0, ids(t - 1), ids(t - 2 - pin_cat % 8))))
    col["rd_last"] = wh(used, rd, ids(t - 6))
    col["rd_first"] = wh(used, col["rd_last"], torch.full_like(rd, -1))
    itk = wh(pin_cat <= 1, wh(pin_cat == 1, t - 1, t - 2), t - 1)
    col["issue_tick"] = wh(is_if, itk, -1)
    col["seq"] = wh(is_if, itk, -1)
    col["demand"] = wh(is_if, wh(pin_cat == 1, OPS.DEM_V, OPS.DEM_D), 0)
    col["issuer"] = wh(is_if, wh(pin_cat == 1, OPS.ISS_V, OPS.ISS_D), 0)
    col["landed"] = is_if & (pin_cat == 1)                              # the urgent gather landed in place at t - 1
    col["req_key"] = wh(is_if, wh(pin_cat == 1, 1, 2), -1)
    for n, v in col.items():
        dst = T.c[n].view(S, NPOS)
        dst.scatter_(1, slot_perm, v.to(dst.dtype))
    # pins (bit planes): key slot 0 verify for odd-recency LRU, PO (borrowing: both), VRES, the urgent block; the
    # normal streams' second PO only by key slot KEYS - 1 (draft plane)
    planes = torch.zeros((2 * KEYS, S, NBLK + 1), dtype=torch.bool)
    p0 = (is_lru & (pin_cat % 2 == 1)) | (is_po & ((pin_cat == 0) | bor.view(-1, 1))) | is_vres | (is_if & (pin_cat == 1))
    pk = is_po & (pin_cat == 1) & ~bor.view(-1, 1)
    bp = blk_pos.clamp(min=0)
    planes[0].scatter_(1, wh(p0, bp, NBLK), True)
    planes[2 * KEYS - 1].scatter_(1, wh(pk, bp, NBLK), True)
    T.pin_w.view(2 * KEYS, S, G.WB).copy_(SP._pack_bits(planes[..., :NBLK]))
    # keys: every key's last reader at t - 2; key slot KEYS - 1 queued for release at half the (l, b)
    lb = torch.arange(L).view(-1, 1) + torch.arange(B).view(1, -1)
    T.key_rd.copy_((t - 2) * L + torch.arange(L).view(-1, 1, 1).expand(L, B, KEYS))
    T.rel_q[:, :, KEYS - 1] = lb % 2 == 0
    T.hist_from.zero_()                                                    # T4 active: every block needs a write-back
    T.hist_id.fill_((t - 3) * L)
    T.clock = 1 << 20
    # the tick rows in C's rowsel order (verify rows, then draft rows): batch slot and pin plane per row
    row_b = torch.cat([rows.vb, rows.db])
    row_plane = torch.cat([rows.vk, rows.dk + KEYS])
    n_vkeys = len(set(zip(rows.vb.tolist(), rows.vk.tolist())))
    return dict(G=G, T=T, rows=rows, inV=inV, inD=inD, bor=bor, b_of=b_of, l_of=l_of, vreq=vreq, dreq=dreq,
                row_b=row_b, row_plane=row_plane, n_vkeys=n_vkeys)


def fresh_window(a, G, dev="cpu") -> "OPS.ReaderWindow":
    """Reader events of ticks 0 .. t - 1 (one per layer, id = tick x L + l): t - 1 pending, the rest complete."""
    w = OPS.ReaderWindow(cap=1 << 14, device=dev)
    for tt in range(a.tick):
        for l in range(G.L):
            w.register(Ev(done=tt <= a.tick - 2), tt * G.L + l)
    w.refresh(ev_done)                       # the steady state: only the pending tick t - 1 is live (untimed)
    return w


class Spans:
    """Component timers. keep {name: [ns]} (None: untimed, the pipelined pass); on CUDA every stop synchronizes the
    device (the component's own wall time) and a CUDA event pair records the stream's span into evkeep at flush();
    profile: every component is also a torch.profiler.record_function range."""

    def __init__(self, dev, keep=None, evkeep=None, profile=False):
        self.cuda = torch.device(dev).type == "cuda"
        self.keep, self.evkeep, self.profile = keep, evkeep, profile
        self.pending: List = []

    def start(self, name: str):
        rf = None
        if self.profile:
            rf = torch.profiler.record_function(name)
            rf.__enter__()
        e0 = None
        if self.cuda and self.keep is not None:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
        return (name, time.perf_counter_ns(), e0, rf)

    def stop(self, tok) -> None:
        name, t0, e0, rf = tok
        if self.keep is not None:
            if self.cuda:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                torch.cuda.synchronize()
                self.pending.append((name, e0, e1))
            self.keep.setdefault(name, []).append(time.perf_counter_ns() - t0)
        if rf is not None:
            rf.__exit__(None, None, None)

    def flush(self) -> None:
        for name, e0, e1 in self.pending:
            if self.evkeep is not None:
                self.evkeep.setdefault(name, []).append(int(e0.elapsed_time(e1) * 1e6))
        self.pending = []


def _host(x):
    """Tensors -> Python numbers / lists (for the JSON; after the timing)."""
    if torch.is_tensor(x):
        return x.item() if x.dim() == 0 else x.tolist()
    if isinstance(x, dict):
        return {k: _host(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_host(v) for v in x]
    return x


def run_poll(T, win, a, sp=None, checked: bool = True) -> Dict:
    sp = sp or Spans(T.dev)
    out = {}
    k = sp.start("poll.refresh")
    win.refresh(ev_done)
    sp.stop(k)
    k = sp.start("poll.force_seq")
    out["force_seq"] = OPS.force_seq(T, a.tick, a.tick - 2, a.lag)
    sp.stop(k)
    k = sp.start("poll.arrivals")
    out.update(OPS.poll_arrivals(T, a.tick, a.tick - 2, a.lag))
    sp.stop(k)
    out["poll_I1"] = out["poll_I2"] = False
    if checked:
        k = sp.start("poll.checks")
        pc = OPS.post_checks(T, None, 0, which=("I1", "I2"))
        i1, i2 = torch.stack([pc["I1"], pc["I2"]]).tolist()           # one host read
        sp.stop(k)
        out["poll_I1"], out["poll_I2"] = bool(i1), bool(i2)
    k = sp.start("poll.key_drops")
    out.update(OPS.drop_due_keys(T, win))
    sp.stop(k)
    k = sp.start("poll.releases")
    out["freed"] = OPS.poll_releases(T, win)
    sp.stop(k)
    return out


# ===================================================================================================================
# the records (C's note-12l shapes) against the polled state
def gen_records(W: Dict, T1, a) -> Dict:
    G, rows = W["G"], W["rows"]
    g = torch.Generator().manual_seed(a.seed + 1)
    L, B, H, C, NBLK, NPOS, S, k = G.L, G.B, G.H, G.C, G.NBLK, G.NPOS, G.S, a.k
    inV, inD, bor = W["inV"], W["inD"], W["bor"]
    code = T1.c["code"].view(S, NPOS)
    stamp = T1.c["stamp"].view(S, NPOS)
    block = T1.c["block"].view(S, NPOS).to(torch.int64)
    uid = T1.c["uid"].view(S, NPOS)
    pos = torch.arange(NPOS).view(1, -1).expand(S, -1)
    big = torch.full((S, NPOS), BIG, dtype=torch.int64)
    lru_sorted = torch.argsort(wh(code == C_LRU, stamp, big), 1)
    n_lru = (code == C_LRU).sum(1)
    free_sorted = torch.argsort(wh(code == C_FREE, pos, big), 1)
    n_free = (code == C_FREE).sum(1)
    pvm = (code == C_PO) | (code == C_VRES)
    pv_sorted = torch.argsort(wh(pvm, pos, big), 1)
    n_pv = pvm.sum(1)
    own = T1.owner[W["b_of"]].view(-1, 1)
    ifl = (code == C_INFLIGHT) & (uid == own) & (torch.bitwise_and(T1.c["demand"].view(S, NPOS), OPS.DEM_D) != 0) \
        & ~T1.c["landed"].view(S, NPOS)
    ifl_sorted = torch.argsort(wh(ifl, pos, big), 1)
    n_if = ifl.sum(1)
    occ = torch.zeros((S, NBLK + 1), dtype=torch.bool)
    occ.scatter_(1, wh(code != C_FREE, block.clamp(min=0), torch.full_like(block, NBLK)), True)
    occ = occ[:, :NBLK]
    nr_sorted = torch.argsort(wh(~occ, torch.rand(S, NBLK, generator=g), torch.full((S, NBLK), 2.0)), 1)
    n_nr = (~occ).sum(1)
    room = (C - (NPOS - n_free)).clamp(min=0)
    z = torch.zeros(S, dtype=torch.int64)
    n_pe = distribute(a.policy_e, inV & bor, torch.full((S,), 4), g, "policy-e")
    n_vf = distribute(a.urgent - a.policy_e, inV & ~bor, torch.minimum(room, torch.full((S,), 2)), g, "urgent")
    n_vm = distribute(a.v_merge, inV & ~bor, n_if, g, "v-merge")
    n_dt = distribute(a.d_take, inD & ~bor, torch.minimum(room - n_vf, n_free - n_vf), g, "d-take")
    n_ref = distribute(a.refused, inD & bor, torch.full((S,), max(1, k // 4)), g, "refused")
    n_di = distribute(a.d_infl, inD & ~bor, n_if, g, "d-infl")
    poolA = inD & bor & (n_pe == 0)
    a1 = distribute(min(a.acquire, int(poolA.sum())), poolA, torch.ones(S, dtype=torch.int64), g, "acquire")
    n_acq = a1 + distribute(a.acquire - int(a1.sum()), inD & ~bor, n_pv, g, "acquire")
    n_vmiss = n_pe + n_vf
    n_vh = wh(inV, k - n_vmiss - n_vm, z)
    n_dmiss = n_dt + n_ref
    n_dh = wh(inD, k - n_dmiss - n_di - n_acq, z)
    for nm, nh in (("verify", n_vh), ("draft", n_dh)):
        if bool((nh < 0).any()) or bool((nh > n_lru).any()):
            raise ValueError("%s hits out of range" % nm)
    if bool((n_vmiss + n_dmiss > n_nr).any()):
        raise ValueError("not enough non-resident blocks")

    def entries(ss, segs):
        """(n, k) entries of streams ss: segs = [(count (S,), sorted list (S, N), offset (S,), is_slot)] in order;
        returns blk, slot (-1 for a block entry), cat."""
        e = torch.arange(k).view(1, -1)
        blk = torch.full((ss.numel(), k), -1, dtype=torch.int64)
        slot = torch.full_like(blk, -1)
        catv = torch.full_like(blk, -1)
        lo = torch.zeros(ss.numel(), 1, dtype=torch.int64)
        for ci, (cnt, srt, off, is_slot) in enumerate(segs):
            c = cnt[ss].view(-1, 1)
            m = (e >= lo) & (e < lo + c)
            ix = (off[ss].view(-1, 1) + e - lo).clamp(min=0, max=srt.shape[1] - 1)
            val = srt[ss].gather(1, ix)
            if is_slot:
                bl = block[ss].gather(1, val)
                slot = wh(m, val, slot)
                blk = wh(m, bl, blk)
            else:
                blk = wh(m, val, blk)
            catv = wh(m, torch.full_like(catv, ci), catv)
            lo = lo + c
        if bool((catv < 0).any()):
            raise ValueError("entries do not fill k")
        o = torch.argsort(blk, 1)
        return blk.gather(1, o), slot.gather(1, o), catv.gather(1, o)

    # verify streams: misses (PE or FREE grant), in-flight merges, hits (the most recent LRU entries)
    sv = inV.nonzero().squeeze(1)
    vblk, vslot, vcat = entries(sv, [(n_vmiss, nr_sorted, z, False), (n_vm, ifl_sorted, z, True),
                                     (n_vh, lru_sorted, n_lru - n_vh, True)])
    # draft streams: misses (take or refusal) after the verify misses, in flight, acquisitions, hits
    sd = inD.nonzero().squeeze(1)
    dblk, dslot, dcat = entries(sd, [(n_dmiss, nr_sorted, n_vmiss, False), (n_di, ifl_sorted, z, True),
                                     (n_acq, pv_sorted, z, True), (n_dh, lru_sorted, n_lru - n_dh, True)])
    # the layer selection (union of the stream's rows) and Policy E's candidates
    selb = torch.zeros((S, NBLK + 1), dtype=torch.bool)
    selb[sv.view(-1, 1).expand_as(vblk), vblk] = True
    selb[sd.view(-1, 1).expand_as(dblk), dblk] = True
    sel_w = SP._pack_bits(selb[:, :NBLK])                                 # (S, WB)
    anyw = OPS.any_pin_words(T1)
    bc = block.clamp(min=0)
    pinned = OPS.bit_test(anyw, torch.arange(S).view(-1, 1).expand(S, NPOS), bc)
    selected = OPS.bit_test(sel_w, torch.arange(S).view(-1, 1).expand(S, NPOS), bc)
    cand = torch.argsort(wh((code == C_LRU) & ~pinned & ~selected, stamp, big), 1)
    n_cand = ((code == C_LRU) & ~pinned & ~selected).sum(1)
    if bool((n_cand < n_pe).any()):
        raise ValueError("not enough Policy E candidates")
    # take slots: per verify stream the q-th miss (block order) -> FREE slot q, or Policy E candidate q
    vmiss = vcat == 0
    vq = torch.cumsum(vmiss.long(), 1) - 1
    vbor = bor[sv].view(-1, 1)
    vtake = wh(vbor, cand[sv].gather(1, vq.clamp(min=0)), free_sorted[sv].gather(1, vq.clamp(min=0)))
    vtake = wh(vmiss, vtake, torch.full_like(vtake, -1))
    vevb = wh(vmiss & vbor, block[sv].gather(1, vtake.clamp(min=0)), torch.full_like(vtake, -1))
    # draft takes: after the stream's verify FREE grants
    dmiss = dcat == 0
    dq = torch.cumsum(dmiss.long(), 1) - 1 + n_vf[sd].view(-1, 1)
    dbor = bor[sd].view(-1, 1)
    dtake = wh(dmiss & ~dbor, free_sorted[sd].gather(1, dq.clamp(min=0, max=NPOS - 1)),
                        torch.full_like(dq, -1))

    # rows -> records: stream s = (l, b, h); verify row j of cohort b; o = (j H + h) k + e
    Rv, Rd = rows.vb.numel(), rows.db.numel()
    map_v = torch.full((S,), -1, dtype=torch.int64)
    map_v[sv] = torch.arange(sv.numel())
    map_d = torch.full((S,), -1, dtype=torch.int64)
    map_d[sd] = torch.arange(sd.numel())
    lj = torch.arange(L).view(-1, 1, 1)
    hh = torch.arange(H).view(1, 1, -1)
    s_v = (lj * B + rows.vb.view(1, -1, 1)) * H + hh                       # (L, Rv, H)
    s_d = (lj * B + rows.db.view(1, -1, 1)) * H + hh
    iv, idd = map_v[s_v], map_d[s_d]
    first = torch.ones(Rv, dtype=torch.bool)
    first[1:] = rows.vb[1:] != rows.vb[:-1]                               # the first row of each cohort
    e = torch.arange(k).view(1, 1, 1, -1)
    jv = torch.arange(Rv).view(1, -1, 1, 1)
    ov = (jv * H + hh.view(1, 1, -1, 1)) * k + e
    VB, VS, VC, VT, VE = vblk[iv], vslot[iv], vcat[iv], vtake[iv], vevb[iv]   # (L, Rv, H, k)
    fr = first.view(1, -1, 1, 1).expand_as(VB)
    vbor4 = bor[s_v].unsqueeze(-1).expand_as(VB)
    miss, infl = VC == 0, VC == 1
    flags = torch.zeros_like(VB)
    flags += wh(miss & fr & ~vbor4, OPS.GRANT, 0) + wh(miss & fr & vbor4, OPS.EVICT, 0)
    flags += wh(miss & ~fr, OPS.DUP, 0) + wh(infl, OPS.INFL, 0) + wh(fr & (miss | infl), OPS.FIRST, 0)
    vrec_m = miss | infl
    sv4 = s_v.unsqueeze(-1).expand_as(VB)
    igen = T1.c["gen"].view(S, NPOS)[sv4, VS.clamp(min=0)].to(torch.int64)
    vfields = torch.stack([ov.expand_as(VB), VB, flags, torch.full_like(VB, -1), wh(infl, VS, -1),
                           wh(infl, igen, -1), wh(miss, VT, -1), wh(miss & fr, VT, -1),
                           wh(miss & fr & vbor4, VE, -1)])           # (9, L, Rv, H, k)
    jd = torch.arange(Rd).view(1, -1, 1, 1)
    od = (jd * H + hh.view(1, 1, -1, 1)) * k + e
    DB_, DS, DC, DT = dblk[idd], dslot[idd], dcat[idd], dtake[idd]
    dbor4 = bor[s_d].unsqueeze(-1).expand_as(DB_)
    dmiss4 = DC == 0
    dflags = wh(dmiss4 & ~dbor4, OPS.GRANT, 0) + wh(dmiss4 & dbor4, OPS.REFUSED, 0) \
        + wh(DC == 1, OPS.INFL, 0) + wh(DC >= 2, OPS.READY, 0)
    dfields = torch.stack([od.expand_as(DB_), DB_, dflags, wh(DC >= 2, DS, -1), wh(dmiss4, DT, -1)])
    layers = []
    for l in range(L):
        vm = vrec_m[l].reshape(-1)
        vr = vfields[:, l].reshape(9, -1)[:, vm]
        vr = vr[:, torch.argsort(vr[0])]
        dr = dfields[:, l].reshape(5, -1)
        dr = dr[:, torch.argsort(dr[0])]
        layers.append(dict(l=l, vrec=vr.contiguous(), drec=dr.contiguous()))
    # vread: the slots each verify stream reads (hits, merges, takes); vlook: the first rows' ready looks
    rbits = torch.zeros((S, NPOS + 1), dtype=torch.bool)
    rsl = wh(vcat == 0, vtake, vslot)
    rbits[sv.view(-1, 1).expand_as(rsl), wh(rsl >= 0, rsl, NPOS)] = True
    vread = SP._pack_bits(rbits[:, :NPOS]).view(L, B, H, G.WP)
    vlook = torch.zeros((L, Rv, 4), dtype=torch.int64)
    hits = (VC == 2).sum((2, 3))                                           # (L, Rv) hits over the heads
    vlook[:, :, 0] = wh(first.view(1, -1), hits, 0)
    vlook[:, :, 1] = vlook[:, :, 0]
    # per-row selections (rowsel, the pins) and the layer selection words
    rsel_v = torch.zeros((L, Rv, H, NBLK + 1), dtype=torch.bool)
    rsel_v.scatter_(3, VB.clamp(min=0), True)
    rsel_d = torch.zeros((L, Rd, H, NBLK + 1), dtype=torch.bool)
    rsel_d.scatter_(3, DB_.clamp(min=0), True)
    want = dict(urgent=a.urgent, policy_e=a.policy_e, v_merge=a.v_merge, d_take=a.d_take, refused=a.refused,
                d_infl=a.d_infl, acquire=a.acquire)
    got = dict(urgent=int((torch.bitwise_and(flags, OPS.GRANT | OPS.EVICT) != 0).sum()),
               policy_e=int((torch.bitwise_and(flags, OPS.EVICT) != 0).sum()), v_merge=int(n_vm.sum()),
               d_take=int((dflags == OPS.GRANT).sum()), refused=int((dflags == OPS.REFUSED).sum()),
               d_infl=int((dflags == OPS.INFL).sum()), acquire=int(n_acq.sum()),
               v_infl_records=int((torch.bitwise_and(flags, OPS.INFL) != 0).sum()),
               v_dup_records=int((torch.bitwise_and(flags, OPS.DUP) != 0).sum()),
               v_records=int(vrec_m.sum()), d_records=int(dflags.numel()), d_ready=int((dflags == OPS.READY).sum()))
    rowsel_v, rowsel_d = SP._pack_bits(rsel_v[..., :NBLK]), SP._pack_bits(rsel_d[..., :NBLK])
    return dict(layers=layers, vread=vread, vlook=vlook, sel=sel_w.view(L, B, H, G.WB), rowsel_v=rowsel_v,
                rowsel_d=rowsel_d, rowsel=torch.cat([rowsel_v, rowsel_d], 1).contiguous(),   # C's tick-row rowsel
                want=want, got=got)


# ===================================================================================================================
# one apply, component by component (the order OwnResidency.apply_tick will use)
def run_apply(T, win, W, R, a, sp=None, mut=None, checked: bool = True) -> Dict:
    """One apply in OwnResidency.apply_tick's order. checked: every predicate, read in TWO host reads (pre-write: a set
    flag voids the tick before any write; post-write: a set flag rolls the undo log back). Unchecked: the CHECK_ONLY
    components are skipped and the writes go without the undo log (the device's decisions are trusted)."""
    G, rows = W["G"], W["rows"]
    sp = sp or Spans(T.dev)
    pre: Dict[str, torch.Tensor] = {}

    def note(d):
        for n, v in d.items():
            pre[n] = v.sum() if v.dim() else v.to(torch.int64)

    k = sp.start("apply.parse")
    P = OPS.parse(T, R["layers"], rows)
    sp.stop(k)
    if checked:
        k = sp.start("apply.row_checks")
        note(OPS.row_checks(T, rows))
        for part in ("V", "D"):
            if P[part] is not None:
                pre["hole_" + part] = OPS.has(P[part]["flags"], OPS.HOLE).sum()
        sp.stop(k)
    k = sp.start("apply.slot_of")
    so = OPS.slot_of(T)
    sp.stop(k)
    if checked:
        k = sp.start("apply.pin_words")
        anyw = OPS.any_pin_words(T)
        sp.stop(k)
    k = sp.start("apply.k5_events")
    ev = OPS.k5_events(P, T)
    sp.stop(k)
    if checked:
        k = sp.start("apply.k5_first_free")
        note(OPS.k5_check(T, ev, mut))
        sp.stop(k)
        k = sp.start("apply.take_checks")
        note(OPS.take_checks(T, ev, so))
        sp.stop(k)
        k = sp.start("apply.ready_check")
        note(OPS.ready_check(T, P["D"]))
        sp.stop(k)
    k = sp.start("apply.merge_check")
    mc, fv, fd = OPS.merge_check(T, P, so, ev, check=checked)
    note(mc)
    sp.stop(k)
    viol: Dict[str, int] = {}
    undo = None
    gen_before = None
    if checked:
        k = sp.start("apply.policy_e")
        note(OPS.policy_e_check(T, ev, P, R["sel"], anyw, True, mut))
        sp.stop(k)
        k = sp.start("apply.pre_flags")
        names = list(pre)
        vals = torch.stack([pre[n] for n in names]).tolist()                # ONE host read before any write
        sp.stop(k)
        viol = {n: int(v) for n, v in zip(names, vals)}
        if any(viol.values()):
            return dict(void=True, stage="pre", viol=viol)
        k = sp.start("apply.gen_snapshot")
        gen_before = T.c["gen"].clone()
        undo = OPS.Undo()
        clock0 = T.clock
        undo.call(lambda: setattr(T, "clock", clock0))                 # a void tick also restores the stamp clock
        sp.stop(k)
    k = sp.start("apply.pins")
    npin = OPS.pins_or(T, R["rowsel"], W["row_b"], W["row_plane"], undo, mut)   # C's tick-row rowsel, one call
    sp.stop(k)
    k = sp.start("apply.takes")
    takes = OPS.write_takes(T, ev, a.tick, a.tick, win, undo, mut)
    sp.stop(k)
    k = sp.start("apply.merges")
    OPS.write_merges(T, fv, fd, undo)
    sp.stop(k)
    k = sp.start("apply.recency")
    rc = OPS.recency(T, P["D"], rows, undo, mut)
    sp.stop(k)
    if checked:
        k = sp.start("apply.post_checks")
        post = dict(OPS.post_checks(T, gen_before, int(takes["f"].numel())))
        post["recency_bad"] = rc["bad"]
        sp.stop(k)
        k = sp.start("apply.post_flags")
        pn = list(post)
        pv = torch.stack([post[n].to(torch.int64) for n in pn]).tolist()    # ONE host read after the writes
        sp.stop(k)
        viol.update({n: int(v) for n, v in zip(pn, pv)})
        if any(pv):
            k = sp.start("apply.rollback")
            undo.rollback()
            sp.stop(k)
            return dict(void=True, stage="post", viol=viol, takes=int(takes["f"].numel()), pin_rounds=npin,
                        recency=rc)
    k = sp.start("apply.readers")
    e_l = torch.tensor([win.register(Ev(done=False), a.tick * G.L + l) for l in range(G.L)], dtype=torch.int64,
                       device=T.dev)
    rd = OPS.read_done(T, R["vread"], P["D"], rows, e_l)
    sp.stop(k)
    k = sp.start("apply.metrics")
    met = OPS.metrics(P, R["vlook"], takes, rows, G, W["n_vkeys"])
    mt = [n for n, v in met.items() if torch.is_tensor(v)]
    mv = torch.stack([met[n].to(torch.int64) for n in mt]).tolist() if mt else []   # ONE host read
    met = dict(met, **dict(zip(mt, mv)))
    sp.stop(k)
    k = sp.start("apply.issue_desc")
    desc, rmarks, hmark = OPS.issue_desc_args(T, takes, win)
    sp.stop(k)
    return dict(void=False, viol=viol, takes=int(takes["f"].numel()), force_waits=takes["force_waits"],
                superseded=takes["superseded"], pin_rounds=npin, recency=rc, readers=rd, metrics=met,
                desc_shape=list(desc.shape), desc_contiguous=bool(desc.is_contiguous()), reader_marks=rmarks,
                hist_mark=hmark)


# ===================================================================================================================
def stats_ms(v: List[int]) -> Dict[str, float]:
    t = torch.tensor(v, dtype=torch.float64) / 1e6
    q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
    return dict(p10=round(q[0], 4), median=round(q[1], 4), p90=round(q[2], 4), mean=round(float(t.mean()), 4), n=len(v))


def rows_to(rows, dev):
    return OPS.Rows(vb=rows.vb.to(dev), vk=rows.vk.to(dev), db=rows.db.to(dev), dk=rows.dk.to(dev), k=rows.k)


def move_world(W: Dict, R: Dict, dev) -> tuple:
    Wd = dict(W, rows=rows_to(W["rows"], dev), row_b=W["row_b"].to(dev), row_plane=W["row_plane"].to(dev))
    Rd = dict(R, layers=[{n: (v.to(dev) if torch.is_tensor(v) else v) for n, v in x.items()} for x in R["layers"]])
    for n in ("vread", "vlook", "sel", "rowsel"):
        Rd[n] = R[n].to(dev)
    return Wd, Rd


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_on(dev, a, W: Dict, R: Dict, T0, checked: bool) -> Dict:
    """The component timing (each component synchronized) and the PIPELINED timing (whole ticks) on one device."""
    dev = torch.device(dev)
    Wd, Rd = move_world(W, R, dev)
    Td0 = T0.to(dev)
    T = Td0.clone()
    tm: Dict[str, List[int]] = {}
    evt: Dict[str, List[int]] = {}
    res = pol = None
    for it in range(a.warmup + a.iters):
        T.copy_from(Td0)
        win = fresh_window(a, W["G"], dev)
        _sync(dev)
        sp = Spans(dev, tm if it >= a.warmup else None, evt)
        pol = run_poll(T, win, a, sp, checked)
        res = run_apply(T, win, Wd, Rd, a, sp, None, checked)
        sp.flush()
    final = T.clone()
    pipe, pipe_ev = [], []
    for it in range(a.iters if a.pipelined_iters < 0 else a.pipelined_iters):
        T.copy_from(Td0)
        win = fresh_window(a, W["G"], dev)
        _sync(dev)
        e0 = e1 = None
        if dev.type == "cuda":
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        t0 = time.perf_counter_ns()
        run_poll(T, win, a, None, checked)
        run_apply(T, win, Wd, Rd, a, None, None, checked)
        if e1 is not None:
            e1.record()
        _sync(dev)
        pipe.append(time.perf_counter_ns() - t0)
        if e0 is not None:
            pipe_ev.append(int(e0.elapsed_time(e1) * 1e6))
    comp = {n: stats_ms(v) for n, v in tm.items()}
    poll_ms = sum(c["median"] for n, c in comp.items() if n.startswith("poll."))
    apply_ms = sum(c["median"] for n, c in comp.items() if n.startswith("apply."))
    return dict(device=str(dev), checked=checked, poll=_host(pol), apply=_host(res),
                components_ms=dict(sorted(comp.items(), key=lambda kv: -kv[1]["median"])),
                components_event_ms={n: stats_ms(v) for n, v in evt.items()} if evt else None,
                poll_ms=round(poll_ms, 4), apply_ms=round(apply_ms, 4), synced_total_ms=round(poll_ms + apply_ms, 4),
                pipelined_ms=stats_ms(pipe) if pipe else None, pipelined_event_ms=stats_ms(pipe_ev) if pipe_ev else None,
                final=final.to("cpu"))


def profile_on(dev, a, W: Dict, R: Dict, T0, checked: bool) -> Dict:
    """One tick under torch.profiler, every component a record_function range: the device busy time (the kernels'
    self time), the kernel launches, the host synchronizations and per component its host and device time."""
    try:
        from torch.profiler import ProfilerActivity, profile
        dev = torch.device(dev)
        Wd, Rd = move_world(W, R, dev)
        Td0 = T0.to(dev)
        T = Td0.clone()
        win = fresh_window(a, W["G"], dev)
        run_poll(T, win, a, None, checked)                          # warm
        run_apply(T, win, Wd, Rd, a, None, None, checked)
        T.copy_from(Td0)
        win = fresh_window(a, W["G"], dev)
        _sync(dev)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with profile(activities=acts) as prof:
            sp = Spans(dev, None, None, profile=True)
            run_poll(T, win, a, sp, checked)
            run_apply(T, win, Wd, Rd, a, sp, None, checked)
            _sync(dev)
        busy = 0.0
        launches = syncs = memcpy = 0
        comps = {}
        for e in prof.key_averages():
            selfd = float(getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0) or 0.0)
            totd = float(getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0.0) or 0.0)
            busy += selfd
            if e.key in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"):
                launches += e.count
            if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"):
                syncs += e.count
            if e.key in ("cudaMemcpyAsync", "cudaMemcpy"):
                memcpy += e.count
            if e.key.startswith(("poll.", "apply.")):
                comps[e.key] = dict(cpu_ms=round(e.cpu_time_total / 1e3, 4), device_ms=round(totd / 1e3, 4))
        return dict(device_busy_ms=round(busy / 1e3, 4), launches=launches, syncs=syncs, memcpy=memcpy,
                    components=comps)
    except Exception as ex:                                       # a REPORT: the run does not depend on it
        return dict(error=repr(ex))


def verdict_band(x, reg) -> str:
    if x is None:
        return "MISSING"
    lo, hi = reg["band"]
    if x > reg.get("falsified_above", float("inf")):
        return "FALSIFIED"
    return "HELD" if lo <= x <= hi else "GREY"


def main(argv=None) -> int:
    a = parse_args(argv)
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("R2_OWN2_B2T: --device cuda needs a CUDA device (torch.cuda.is_available() is False): refused",
              file=sys.stderr, flush=True)
        return 2
    torch.set_num_threads(max(1, a.threads))
    W = build_world(a)
    G, T0 = W["G"], W["T"]
    T1 = T0.clone()                                                # the records, against the polled state
    poll0 = run_poll(T1, fresh_window(a, G), a)
    R = gen_records(W, T1, a)
    modes = {"both": (True, False), "checked": (True,), "unchecked": (False,)}[a.mode]
    runs = {("checked" if c else "unchecked"): time_on(dev, a, W, R, T0, c) for c in modes}
    cpu_ref = None
    if a.cpu_reference and dev.type == "cuda":
        cpu_ref = {("checked" if c else "unchecked"): time_on(torch.device("cpu"), a, W, R, T0, c) for c in modes}
    prof = {("checked" if c else "unchecked"): profile_on(dev, a, W, R, T0, c) for c in modes} if a.profile else None
    failed = []
    for n, w in R["want"].items():
        if R["got"][n] != w:
            failed.append("count %s: got %d want %d" % (n, R["got"][n], w))
    for m, r in runs.items():
        for n, v in r["apply"]["viol"].items():
            if v:
                failed.append("%s: predicate %s fired %d" % (m, n, v))
        if r["apply"]["void"]:
            failed.append("%s: the tick was void" % m)
        for n in ("poll_I1", "poll_I2"):
            if r["poll"][n]:
                failed.append("%s: poll predicate %s fired" % (m, n))
    finals = [r.pop("final") for r in runs.values()]
    if cpu_ref:
        finals += [r.pop("final") for r in cpu_ref.values()]
    same = all(not finals[0].diff(f) for f in finals[1:])
    if not same:
        failed.append("the final tables differ between the modes / devices")
    head = runs.get("checked") or runs["unchecked"]
    total = head["synced_total_ms"]
    verdicts = {}
    if dev.type == "cpu":
        verdicts["host_25ms"] = dict(verdict="REFUTED" if total > FALSIFIER_MS else "SUPPORTED",
                                     applies=torch.get_num_threads() == 1 and not a.smoke and "checked" in runs)
    else:
        RD = REGISTERED_DEVICE
        pc = runs.get("checked")
        pu = runs.get("unchecked")
        verdicts["D1_pipelined_ms"] = verdict_band(pc["pipelined_ms"]["median"] if pc else None, RD["D1_pipelined_ms"])
        verdicts["D1u_pipelined_unchecked_ms"] = verdict_band(pu["pipelined_ms"]["median"] if pu else None,
                                                              RD["D1u_pipelined_unchecked_ms"])
        verdicts["D2_synced_ms"] = verdict_band(pc["synced_total_ms"] if pc else None, RD["D2_synced_ms"])
        pk = (prof or {}).get("checked") or {}
        verdicts["D3_busy_ms"] = verdict_band(pk.get("device_busy_ms"), RD["D3_busy_ms"])
        verdicts["D4_launches"] = verdict_band(pk.get("launches"), RD["D4_launches"])
        verdicts["D5_syncs"] = verdict_band(pk.get("syncs"), RD["D5_syncs"])
        d6 = None
        if pk.get("components") and pc:
            d6 = all(pk["components"].get(n, {}).get("device_ms", 0.0) <= 0.5 * c["median"]
                     for n, c in pc["components_ms"].items() if c["median"] >= 0.2)
        verdicts["D6_host_bound"] = "MISSING" if d6 is None else ("HELD" if d6 else "FALSIFIED")
    gpu = None
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        gpu = dict(name=p.name, uuid=str(getattr(p, "uuid", "")), cuda=torch.version.cuda)
    out = dict(geometry=G._asdict(), npos=G.NPOS, slots=G.S * G.NPOS, device=str(dev), gpu=gpu,
               threads=torch.get_num_threads(), torch=torch.__version__, host=platform.node(), args=vars(a),
               counts_requested=R["want"], counts_generated=R["got"], poll_first=_host(poll0), modes=runs,
               cpu_reference=cpu_ref, profile=prof, modes_state_equal=same, total_ms=total,
               registered_device=REGISTERED_DEVICE, host_bound=HOST_BOUND, check_only=CHECK_ONLY,
               verdicts=verdicts, failed=failed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for m, r in runs.items():
        for n, c in r["components_ms"].items():
            print("%-9s %-22s median %8.3f ms  p90 %8.3f ms" % (m, n, c["median"], c["p90"]))
        pm = r["pipelined_ms"]["median"] if r["pipelined_ms"] else float("nan")
        print("%-9s synced total %.3f ms; pipelined median %.3f ms" % (m, r["synced_total_ms"], pm))
    for n, v in verdicts.items():
        print("VERDICT %s %s" % (n, v))
    for f in failed:
        print("FAILED: " + f)
    vtxt = verdicts.get("D1_pipelined_ms") or verdicts.get("host_25ms", {}).get("verdict", "-")
    print("R2_OWN2_B2T_DONE failed=%d verdict=%s total_ms=%.3f device=%s" % (len(failed), vtxt, total, dev.type),
          flush=True)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
