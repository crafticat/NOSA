"""The CPU WORLD of tests_cpu/test_spec_prefetch.py (Phase 2, rung L5: nosi/nosi/spec_prefetch.py; not a test file).
Load it by file path, as test_spec_round.py loads spec_world.py. It builds on spec_world.py (read its docstring: the
SHIPPED engine from its own source, every CUDA op a torch stand-in with the kernel's flat addresses, L28).

THIS FILE ADDS
  FakeTransport          a CPU stand-in of GilfreeTransport's DESCRIPTOR interface, the calls that PoolPrefetch makes:
                         issue_desc, job_ev (query / synchronize / wait / wait_submitted), landed_upto,
                         npipe.submitted, mark_reader, mark_hist, flush, poll, stats, close. A job copies the host rows
                         of its takes into the staging lanes (row lane * 64 + r, head h: the flat-store layout of the
                         real packer with xpool=_Lanes) when it LANDS. hold=False: at the issue. hold=True: only when a
                         wait asks for it (the job is IN FLIGHT until the verify's commit waits it). FIFO: landing job s
                         lands every earlier job (the C++ worker is FIFO).
  make_pf(...)           a factory for spec_world.run_vs_shipped(make_prefetch=...): a SpecPrefetch on the attached
                         RaggedState, the scatter twins of ragged_world (h2d_ref / h2d_bias_ref), the FakeTransport or
                         (ext given) the REAL packer v3 in CPU mode
  VerifyRecount          a test-only hook OUTSIDE the prefetch hook: for every verify residency it classifies each
                         selected slot of each verify request FROM THE RESIDENCY ITSELF (the window map, the pool map and
                         the tags before the call): W window, P pool (not prefetched), F0 / F1 a prefetched block of
                         class ready / in flight, N new (a host load); it also recounts the host loads from the load mask
                         after the call, and checks that every tagged slot is pinned before the call
  run_pf(...)            run_vs_shipped with the prefetch and the recount; the evidence gets pf, rep (the prefetch
                         report) and recount
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SW = _load("spec_world")
W, RW, VW, RE, RR, SR, BC = SW.W, SW.RW, SW.VW, SW.RE, SW.RR, SW.SR, SW.BC
SP = W.pkg_load("spec_prefetch")
PFM = sys.modules[W.PKG + ".pool_prefetch"]
L, BS, TOPK, HKV, V = W.L, W.BS, W.TOPK, W.HKV, W.V
TWIN_OPS = types.SimpleNamespace(h2d=W.h2d_ref, h2d_bias=W.h2d_bias_ref)


# ---------------------------------------------------------------------------
# the transport stand-in
# ---------------------------------------------------------------------------
class FakeJobEv:
    def __init__(self, tp, seq: int):
        self.tp, self.seq = tp, int(seq)

    def query(self) -> bool:
        return self.tp.landed >= self.seq

    def synchronize(self) -> None:
        self.tp.land_upto(self.seq, "synchronize")

    def wait_submitted(self) -> None:
        self.tp.land_upto(self.seq, "wait_submitted")

    def wait(self, stream=None) -> None:
        self.tp.land_upto(self.seq, "wait")


class FakeTransport:
    """See the module docstring."""

    def __init__(self, hold: bool = False):
        self.hold = bool(hold)
        self.jobs, self.landed, self.by = [], 0, []
        self.npipe = types.SimpleNamespace(submitted=0)       # PoolPrefetch._submitted reads it (submitted = landed)
        self.marks = dict(reader=0, hist=0)
        self.closed = False
        self.bound = False

    @property
    def stats(self):
        return dict(jobs=len(self.jobs), landed=self.landed, takes=sum(int(d.shape[1]) for d in self.jobs),
                    landed_by=list(self.by))

    def bind(self, host_k, host_v, slots_k, slots_v, bs: int) -> None:
        self.host_k, self.host_v, self.slots_k, self.slots_v, self.bs = host_k, host_v, slots_k, slots_v, int(bs)
        self.bound = True

    def issue_desc(self, desc, reader_marks=None, hist_mark: int = -1) -> int:
        assert self.bound, "FakeTransport.bind first"
        assert torch.is_tensor(desc) and desc.dtype == torch.int64 and desc.dim() == 2 and desc.shape[0] == 6
        if desc.shape[1] == 0:
            return 0
        self.jobs.append(desc.clone())
        seq = len(self.jobs)
        self.npipe.submitted = self.landed
        if not self.hold:
            self.land_upto(seq, "issue")
        return seq

    def land_upto(self, seq: int, by: str) -> None:
        while self.landed < int(seq):
            self._land(self.jobs[self.landed])
            self.landed += 1
            self.by.append(by)
        self.npipe.submitted = self.landed

    def _land(self, desc) -> None:
        l, b, h, X, lane = desc[0], desc[1], desc[2], desc[3], desc[4]
        r = torch.arange(self.bs, dtype=torch.int64)
        for l0 in torch.unique(l).tolist():
            m = l == l0
            dst = lane[m].view(-1, 1) * self.bs + r.view(1, -1)
            src = X[m].view(-1, 1) * self.bs + r.view(1, -1)
            bb, hh = b[m].view(-1, 1), h[m].view(-1, 1)
            self.slots_k[l0][dst, hh] = self.host_k[l0][bb, src, hh]
            self.slots_v[l0][dst, hh] = self.host_v[l0][bb, src, hh]

    def job_ev(self, seq: int) -> FakeJobEv:
        assert int(seq) >= 1
        return FakeJobEv(self, seq)

    def landed_upto(self) -> int:
        return self.landed

    def mark_reader(self, mark, ev) -> None:
        self.marks["reader"] += 1

    def mark_hist(self, mark, ev) -> None:
        self.marks["hist"] += 1

    def flush(self) -> None:
        self.land_upto(len(self.jobs), "flush")

    def poll(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def make_pf(hold: bool = False, lanes: int = 16, neg=None, ext=None, box=None):
    """A factory for run_vs_shipped(make_prefetch=...). box (dict): gets the SpecPrefetch as box["pf"]."""
    def f(rs):
        if ext is None:
            tp = FakeTransport(hold=hold)
            pf = SP.SpecPrefetch(rs, transport=tp, lanes=lanes, ops=TWIN_OPS, neg=neg, probes=False)
            R = lanes * BS
            tp.bind([e._k_cpu for e in pf.engines], [e._v_cpu for e in pf.engines],
                    [pf.stage_k[l * R:(l + 1) * R] for l in range(pf.L)],
                    [pf.stage_v[l * R:(l + 1) * R] for l in range(pf.L)], BS)
        else:
            pf = SP.SpecPrefetch(rs, ext, lanes=lanes, threads=1, ops=TWIN_OPS, neg=neg, cap_groups=4, ring=2,
                                 probes=False)
        if box is not None:
            box["pf"] = pf
        return pf
    return f


# ---------------------------------------------------------------------------
# the independent recount of the verify (test-only hook)
# ---------------------------------------------------------------------------
class VerifyRecount:
    """See the module docstring. cats[(round, layer, call, request, head, block)] = "W" / "P" / "F0" / "F1" / "N"; tot:
    the sums; tagged_seen / tagged_unpinned: tagged slots before a verify call and those without a prefetch pin age."""

    def __init__(self, runner, pf=None):
        self.rn, self.pf = runner, pf
        body = runner.rs.body
        self.own = "decode_fixed" in body.__dict__
        self.inner = body.decode_fixed
        body.decode_fixed = self.hook
        self.lay = {lay.cache_engine._k_gpu.data_ptr(): l for l, lay in enumerate(runner.rs.cache.layers)}
        self.calls = {}
        self.cats = {}
        self.tot = dict(W=0, P=0, F0=0, F1=0, N=0, sel=0, new_lm=0, tagged_seen=0, tagged_unpinned=0)

    def detach(self) -> None:
        body = self.rn.rs.body
        assert body.__dict__.get("decode_fixed") == self.hook
        if self.own:
            body.decode_fixed = self.inner
        else:
            del body.decode_fixed

    def hook(self, eng, *args, **kwargs):
        rn = self.rn
        if not rn.meter.on:
            return self.inner(eng, *args, **kwargs)
        l = self.lay[eng._k_gpu.data_ptr()]
        g = int(rn.round)
        i = self.calls.get((g, l), 0)
        self.calls[(g, l)] = i + 1
        sel = args[3].clone()
        bm, pm, age = eng._block_map.clone(), eng._pool_map.clone(), eng._pool_age.clone()
        P = int(pm.shape[2])
        tags = (self.pf.tags[l][:, :, :P].clone() if self.pf is not None
                else torch.full_like(pm, -1))
        vm = None
        if rn.ops.level == "L4P" and rn.draft.rows is not None:
            vm = (~rn.draft.rows[i]).clone()
        out = self.inner(eng, *args, **kwargs)
        lm = eng._load_mask.clone()
        H, B, M = sel.shape
        for b in range(B):
            if vm is not None and not bool(vm[b]):
                continue
            for h in range(H):
                win = set(bm[h, b].tolist())
                pool = pm[h, b].tolist()
                for m in range(M):
                    X = int(sel[h, b, m])
                    self.tot["sel"] += 1
                    if X in win:
                        c = "W"
                    elif X in pool:
                        t = int(tags[h, b, pool.index(X)])
                        c = "P" if t < 0 else "F%d" % t
                    else:
                        c = "N"
                    self.tot[c] += 1
                    self.cats[(g, l, i, b, h, X)] = c
                self.tot["new_lm"] += int((lm[h, b] >= 0).sum())
        tg = tags >= 0
        self.tot["tagged_seen"] += int(tg.sum())
        self.tot["tagged_unpinned"] += int((tg & (age < SP.PIN_BASE + SP.PF_SEQ0)).sum())
        return out

    def undone_pf_hits(self, records) -> int:
        """Prefetched hits (F0 / F1) in a verify step / joint-pass row that the round undid (row i >= m of its
        request): the undo moved a prefetched block back into its pinned slot."""
        n = 0
        for (g, l, i, b, h, X), c in self.cats.items():
            if c in ("F0", "F1") and 0 <= g < len(records) and i >= int(records[g]["m"][b]):
                n += 1
        return n


def run_pf(B: int, level="L3", prefetch=True, hold=False, lanes=16, neg=None, journal=None, pin=False, filt=None,
           seed=0, S=None, N=None, max_rounds=0, ext=None, oracle_rounds=(-1, 0), recount=True):
    """spec_world.run_vs_shipped with a SpecPrefetch (prefetch True) and the VerifyRecount (recount True)."""
    box = {}

    def on_runner(runner):
        if recount:
            box["recount"] = VerifyRecount(runner, box.get("pf"))
    kw = dict(level=level, mode="exact", gemm="rowgroup", draft_filter=filt, neg=None, max_rounds=max_rounds,
              journal=(RW.CAP if journal is None else journal), on_runner=on_runner, pin=pin, seed=seed,
              oracle_rounds=oracle_rounds,
              make_prefetch=(make_pf(hold=hold, lanes=lanes, neg=neg, ext=ext, box=box) if prefetch else None))
    if S is not None:
        kw["S"] = S
    if N is not None:
        kw["N"] = N
    ev = SW.run_vs_shipped(B, **kw)
    pf = box.get("pf")
    ev["pf"] = pf
    ev["rep"] = pf.report() if pf is not None else None
    ev["recount"] = box.get("recount")
    return ev
