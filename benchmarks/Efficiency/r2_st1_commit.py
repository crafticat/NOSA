"""ST1 (AREA 3 of the RECON7 follow-up: state, schedule and memory efficiency; KS/analysis/plan/LEDGER_rev7.md rows 5, 8
and 9, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE COMMIT OF THE TENTATIVE ROWS, CAPTURED AND MOVED BY ROW. A hook on the
best arms (L4PTFDGUDCD at C 128, L4PTFDGUDKWCD at C 73); the new arms L4PTFDGUDCDM and L4PTFDGUDKWCDM run in
benchmarks/Efficiency/r2_st1_cell.py. NO NEW KERNEL: torch ops, the shipped device-driven masked write-back kernel
(rs.body.writeback_kernel, documented as capturable) and torch.cuda.CUDAGraph. nosi is unchanged. This file imports
torch and time only.

THE EVIDENCE (MEASURED; record /mnt/beegfs/ojerbi/r2-runs/spec/recon7_pm1_2195631_r7pm, stdout "R7 D6" lines: the C 73
B 8 miniature, profiled rounds 8 and 12, both best arms): the commit window's device busy is 1.73 ms; the labelled
TentState.commit is 1.469-1.475 ms of it (85 %); decide + accept 0.068 ms; the eager commit phase median is 3.26-3.43 ms
(round median 118-125 ms). At the timing configurations the commit phase median is 6.058 ms (C 128 B 216, i4dc_d3a) and
8.416 ms (C 73 B 318, i4dc_d4b). The eager commit (nosi/nosi/spec_tentative.py:247-286) issues per layer 6 where() over
the WHOLE tail slot (K, V, bias; before and after the write-back), the write-back kernel and 2 copies: about 9 launches x
32 layers, 4 of them reading and writing B x 64 x H x D x 2 B (7.1 MB at B 216, 10.4 MB at B 318) although at most K + 2
rows per request change.

THE CHANGE (TentState.commit's contract is kept: the same committed bytes, maps, lengths, counters and qc / ctr_c):
  1. ONE CUDA GRAPH for the whole commit (every layer), in its OWN memory pool (CudaGraphsPrivate; never the shared pool
     of the joint and draft graphs: a later capture there could reuse a block that this graph frees). Static inputs: m,
     ctr_m, frontier, dropped (copy_ before every call). ORDER: the first call runs eagerly on the static inputs (WARM:
     Triton and the caching allocator settle outside a capture), the second CAPTURES and replays, every later call
     replays. The original has no host read; so has this one (a capture would fail on one).
  2. MODE "rows" (default): only the J = K + 2 positions qc .. qc + J - 1 of each request move (the chain spans at most
     K + 2 positions past qc: spec_tentative.py:61; m <= the chain; a device counter counts m > J: the OVF gate), by
     index: ONE gather per tensor from the buffer (both phases' rows) and an index_put_ per tensor and phase into the
     tail slot, in the original's two phases around the write-back (phase 1: the rows of block qc // 64; then the
     write-back of a filled block; phase 2: after a fill also the next block's rows). A row that a phase must not write
     is REDIRECTED to the row of position qc with the value of that row (an identical second write: deterministic).
     Rows of UNCOMMITTED positions of the tail slot get the buffer's bytes where the original kept older bytes; no
     reader reads them (attention stops at the committed rows; the write-back moves full committed blocks only). The
     gates EQF / PEQ (tokens, fingerprints, acceptance, state checksums) and TAIL-EQ (the committed rows of every tail
     slot) check the claim on the GPU.
     MODE "rowsx": the same rows without the redirect, each value where()'d against the current tail row: every byte
     equals the original's (the control of the uncommitted-row claim; 4 more ops per layer).
     MODE "where": the original code (the inherited commit) inside the graph: the LAUNCH-ONLY control (what the graph
     alone removes).
HOST COUNTS: TentState.calls["commit"] moves once per call in every mode (mode "where": the eager and the captured
execution run the inherited code; a replay adds the count).
NEGATIVE CONTROLS (tests only; sneg): "stale" (m is not copied before a replay), "no_phase2" (the next block's rows are
not written after a fill), "narrow" (J = 1).
THE CPU TESTS: benchmarks/Efficiency/tests_cpu/test_st1.py (the stand-in backend makes the write-back one kernel node).
"""
import time

import torch

MODES = ("rows", "rowsx", "where")
SNEG = (None, "stale", "no_phase2", "narrow")
_INT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class CudaGraphsPrivate:
    """The commit's capture backend: torch.cuda.CUDAGraph in a PRIVATE pool. capture(fn, T, vs) -> (graph with
    .replay(), fn's result). The CPU tests pass a stand-in with the same interface."""

    def __init__(self):
        self.pool = None
        self.capture_s = 0.0

    def capture(self, fn, T, vs):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn()
        torch.cuda.synchronize()
        self.capture_s += time.perf_counter() - t0
        return g, out

    def pool_bytes(self):
        """The bytes of the allocator segments of the private pool (torch.cuda.memory_snapshot's segment_pool_id);
        None when the snapshot does not name pools."""
        if self.pool is None:
            return 0
        try:
            segs = torch.cuda.memory_snapshot()
        except Exception:                                              # the report is evidence, not a gate input
            return None
        tot, named = 0, False
        want = tuple(self.pool) if isinstance(self.pool, (tuple, list)) else (self.pool,)
        for s in segs:
            pid = s.get("segment_pool_id")
            if pid is None:
                continue
            named = True
            if (tuple(pid) if isinstance(pid, (tuple, list)) else (pid,)) == want:
                tot += int(s.get("total_size", 0))
        return tot if named else None

    def release(self):
        self.pool = None


class CommitST1:
    """The mixin of the state (module docstring): commit() in a graph, the row moves. Set up by _st1_setup."""

    def _st1_setup(self, mode: str, sneg, backend, vs, K: int) -> None:
        if mode not in MODES or sneg not in SNEG:
            raise ValueError("ST1: mode %r (one of %s), control %r (one of %s)" % (mode, MODES, sneg, SNEG))
        J = 1 if sneg == "narrow" else int(K) + 2
        if not 1 <= J <= self.bs // 2:
            raise ValueError("ST1: J %d outside 1 .. %d" % (J, self.bs // 2))
        dev = self.dev
        self._st1 = dict(mode=mode, sneg=sneg, backend=backend, vs=vs, J=J, static=None, graph=None, warm=0,
                         captures=0, replays=0, commits=0, capture_round=None, static_bytes=0)
        self._st1_arJ = torch.arange(J, dtype=torch.int64, device=dev)
        self._st1_ovf = torch.zeros((), dtype=torch.int64, device=dev)
        self._st1_bi = self.all_b.view(self.B, 1)

    def _st1_inherited(self):
        """The commit that this mixin overrides (the next class of the MRO that defines one)."""
        mro = type(self).__mro__
        for c in mro[mro.index(CommitST1) + 1:]:
            if "commit" in c.__dict__:
                return c.__dict__["commit"]
        raise RuntimeError("ST1: no inherited commit")

    def commit(self, m, ctr_m, frontier, dropped) -> None:
        st = self._st1
        st["commits"] += 1
        s = st["static"]
        if s is None:
            s = st["static"] = dict(m=torch.empty_like(m), ctr=torch.empty_like(ctr_m), fr=torch.empty_like(frontier),
                                    dr=torch.empty_like(dropped))
            st["static_bytes"] = sum(t.numel() * t.element_size() for t in s.values())
        if not (st["sneg"] == "stale" and st["graph"] is not None):    # NEGATIVE CONTROL (tests only): stale m
            s["m"].copy_(m)
        s["ctr"].copy_(ctr_m)
        s["fr"].copy_(frontier)
        s["dr"].copy_(dropped)
        where = st["mode"] == "where"
        if where:
            base = self._st1_inherited()

            def body():
                base(self, s["m"], s["ctr"], s["fr"], s["dr"])         # counts itself (eager and capture)
        else:
            self.calls["commit"] += 1

            def body():
                self._st1_rows(s["m"], s["ctr"], s["fr"], s["dr"])
        if st["graph"] is None:
            if st["warm"] == 0:                                        # WARM: eager on the static inputs
                st["warm"] = 1
                body()
                return
            g, _ = st["backend"].capture(body, self, st["vs"])         # CAPTURE (executes nothing), then replay
            st["graph"] = g
            st["captures"] += 1
            st["capture_round"] = int(getattr(self, "round", -1))
            g.replay()
            st["replays"] += 1
            return
        if where:
            self.calls["commit"] += 1                                  # the host count that a replay does not run
        st["graph"].replay()
        st["replays"] += 1

    def _st1_rows(self, m, ctr_m, frontier, dropped) -> None:
        """TentState.commit's effect by row (module docstring, modes "rows" and "rowsx")."""
        rs, B, bs = self.rs, self.B, self.bs
        st = self._st1
        J = st["J"]
        exact = st["mode"] == "rowsx"
        q0 = self.qc
        q1 = q0 + m
        b0 = q0 // bs
        x = q0.view(B, 1) + self._st1_arJ.view(1, J)                  # (B, J): the positions qc .. qc + J - 1
        idx = x % bs                                                   # their buffer row = their tail row
        in0 = (x // bs) == b0.view(B, 1)                               # in the committed block b0
        fill = (q1 // bs) > b0
        full = ((frontier // bs) > b0) if self.tneg == "writeback_tentative" else fill   # NEGATIVE CONTROL (tneg)
        if exact:
            ok = x < q1.view(B, 1)
            v1 = in0 & ok                                              # the original's m1 rows
            v2 = (~in0) & fill.view(B, 1) & ok                         # the original's m2 rows
            i1 = i2 = idx
        else:
            first = idx[:, :1].expand(B, J)
            i1 = torch.where(in0, idx, first)                          # phase 1: the rows of block b0
            i2 = torch.where(in0 | fill.view(B, 1), idx, first)        # phase 2: after a fill also the next block's
        ii = torch.cat((i1, i2), dim=1)                                # (B, 2 J): one gather per tensor
        bi = self._st1_bi
        r1, r2 = i1 + self.base, i2 + self.base
        wpos = b0 * bs + (bs - 1)
        tid = (q1 // bs).view(1, B).expand(self.H, B)
        eng0 = rs.cache.layers[0].cache_engine
        cl = rs.body.cache_lens_value(eng0, q1 - 1).to(torch.int32)    # the shipped length after the step at q1 - 1
        self._st1_ovf += (m > J).sum()
        phase2 = st["sneg"] != "no_phase2"                             # NEGATIVE CONTROL (tests only)
        for l, lay in enumerate(rs.cache.layers):
            e = lay.cache_engine
            pairs = ((e._k_gpu, self.k[l][bi, ii]), (e._v_gpu, self.v[l][bi, ii]),
                     (e._kv_bias_gpu, self.bias[l][bi, ii]))
            for dst, src in pairs:
                val = src[:, :J]
                if exact:
                    val = torch.where(v1.view((B, J) + (1,) * (src.dim() - 2)), val, dst[bi, r1])
                dst[bi, r1] = val
            rs.body.writeback_kernel(e._k_gpu, e._v_gpu, e._k_cpu, e._v_cpu, full, wpos, self.base, bs)
            if phase2:
                for dst, src in pairs:
                    val = src[:, J:]
                    if exact:
                        val = torch.where(v2.view((B, J) + (1,) * (src.dim() - 2)), val, dst[bi, r2])
                    dst[bi, r2] = val
            e._block_map[:, :, self.tail].copy_(tid)
            e._cache_lens.copy_(cl)
        c = self.c
        c["appended"] += m.sum()
        c["crossed"] += fill.sum()
        c["dropped"] += dropped.sum()
        self.qc.copy_(q1)
        self.ctr_c.copy_(ctr_m)

    def st1_report(self) -> dict:
        """ONE host read (after a run)."""
        st = self._st1
        b = st["backend"]
        return dict(mode=st["mode"], sneg=st["sneg"], J=st["J"], warm=st["warm"], captures=st["captures"],
                    replays=st["replays"], commits=st["commits"], capture_round=st["capture_round"],
                    overflow=int(self._st1_ovf.item()), static_bytes=st["static_bytes"],
                    capture_s=round(float(getattr(b, "capture_s", 0.0)), 3),
                    pool_bytes=(b.pool_bytes() if hasattr(b, "pool_bytes") else None), backend=type(b).__name__)

    def _st1_release(self) -> None:
        st = self._st1
        st["graph"] = None
        st["static"] = None
        if hasattr(st["backend"], "release"):
            st["backend"].release()


_STATES = {}
_RUNNERS = {}


def state_class(S):
    """The state class S with the mixin (cached)."""
    if S not in _STATES:
        _STATES[S] = type(S.__name__ + "ST1", (CommitST1, S), {})
    return _STATES[S]


def st1_runner(R, mode: str = "rows", sneg=None):
    """The runner class R (a best arm's) whose state commits by ST1 in `mode` (cached per (R, mode, sneg)). Its
    __init__ takes st1_backend (None: CudaGraphsPrivate) besides R's arguments."""
    if mode not in MODES or sneg not in SNEG:
        raise ValueError("ST1: mode %r (one of %s), control %r (one of %s)" % (mode, MODES, sneg, SNEG))
    key = (R, mode, sneg)
    if key not in _RUNNERS:
        class RunnerST1(R):
            _st1_mode, _st1_sneg = mode, sneg

            def __init__(self, *args, st1_backend=None, **kw):
                super().__init__(*args, **kw)
                T = self.T
                T.__class__ = state_class(type(T))
                T._st1_setup(self._st1_mode, self._st1_sneg,
                             st1_backend if st1_backend is not None else CudaGraphsPrivate(), self.vs, int(self.K))
                self.meta.update(st1=dict(mode=self._st1_mode, sneg=self._st1_sneg, J=T._st1["J"],
                                          backend=type(T._st1["backend"]).__name__))

            def read(self) -> dict:
                out = super().read()
                out.update(st1=self.T.st1_report())
                return out

            def detach(self) -> None:
                self.T._st1_release()
                super().detach()

        RunnerST1.__name__ = RunnerST1.__qualname__ = R.__name__ + "ST1"
        _RUNNERS[key] = RunnerST1
    return _RUNNERS[key]


def tail_checksum(T) -> list:
    """After a run: per layer and tensor (K, V, bias) the integer bit sum of the tail slot's COMMITTED rows (rows below
    qc % 64 of each request; the other rows hold no committed position). One small copy per tensor (the slot)."""
    B, bs = T.B, T.bs
    drop = ~(T.ar64.view(1, bs) < (T.qc % bs).view(B, 1))           # (B, 64): the rows that hold no committed position
    out = []
    for lay in T.rs.cache.layers:
        e = lay.cache_engine
        for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu):
            x = t[:, T.base:T.base + bs].contiguous()
            if x.is_floating_point():
                x = x.view(_INT[x.element_size()])
            x.masked_fill_(drop.view((B, bs) + (1,) * (x.dim() - 2)), 0)
            out.append(int(x.sum(dtype=torch.int64)))
    return out
