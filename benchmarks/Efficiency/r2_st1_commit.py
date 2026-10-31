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
ST2 (TwinST2; the arms L4PTFDGUDCDMT / L4PTFDGUDKWCDMT = the M arm + ST2). EVIDENCE (MEASURED, r7pm summary.json,
L4PTFDGUDCDP round 12, labelled joint window, C 73 B 8): the "twin" label of the joint pass is 1,194 GPU ops (192
_twin_kernel, 906 small, 96 copies), 3.85 ms device and 45.5 ms host per pass, EAGER (outside the joint graphs: the twin
loop of spec_tentative.TentState._layer:338-352 runs per row: replay, pos.clone, 2 .contiguous, step_unchecked, the
n_used / M_b copies). THE CHANGE: per joint pass, the step sequence of every request in position order (its active rows
0 .. a - 1, a prefix; a group request's deferred replay before its draft row v) is gathered once per layer from
(the rows' K / cis, the buffer's deferred row) and the twin ADVANCES it in ONE call: the shipped fused kernel's
advance (r2_twin_fused.advance_fused, T = R + 1 steps, FROM_N, the per-step counters ctr) when the twin's step is the
fused one, else the twin's advance_unchecked (step_unchecked per step); the rows' positions, n_used, M_b and the layer-0
snapshot are gathered from ctr. The banks are independent (one program per bank; per-bank sums), so a bank's step
sequence fixes its bits whatever the order of the banks' steps. Pad rows step nothing in both. NEGATIVE CONTROLS (tests
only; tneg2): "no_replay_seq" (the replay step left out of the sequence), "no_shift" (the rows after the replay read the
counters one step early).
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


TNEG2 = (None, "no_replay_seq", "no_shift")


class TwinST2:
    """ST2: THE TWIN SECTION OF A JOINT PASS IN ONE ADVANCE (module docstring, ST2). Overrides TentState._layer for a
    pass of R > 1 rows (no test record): the same code, with the per-row twin loop (spec_tentative.py:338-352: per row
    the deferred replay, pos.clone, two .contiguous copies, step_unchecked, the n_used / M_b copies, the layer-0
    counter snapshot) replaced by ONE advance of every request's step sequence in position order (its active rows,
    with the deferred replay before the draft row of a group request) and gathers of the per-step counters."""

    def _st2_setup(self, tneg2) -> None:
        if tneg2 not in TNEG2:
            raise ValueError("ST2: control %r (one of %s)" % (tneg2, TNEG2))
        mro = type(self).__mro__
        base = None
        for c in mro[mro.index(TwinST2) + 1:]:
            if "_layer" in c.__dict__:
                base = c.__dict__["_layer"]
                break
        if base is None:
            raise RuntimeError("ST2: no inherited _layer")
        vr = None                                                      # the shipped verify_rows module: the first
        for c in mro[mro.index(TwinST2) + 1:]:                         # inherited _layer whose module binds VR (INT:
            f = c.__dict__.get("_layer")                               # a J1 base's _layer lives in a module without
            if f is not None and "VR" in getattr(f, "__globals__", {}):   # it; spec_tentative.py:123 binds it)
                vr = f.__globals__["VR"]
                break
        if vr is None:
            raise RuntimeError("ST2: no inherited _layer whose module binds VR (verify_rows)")
        self._st2_vr = vr
        f = getattr(self.rs.st.twins[0].step_unchecked, "func", None)
        adv = f.__globals__.get("advance_fused") if (f is not None and getattr(f, "__name__", "") == "step_fused") \
            else None
        self._st2_base = base
        self._st2 = dict(neg=tneg2, fused=adv is not None, passes=0, layers=0, fallback=0)
        self._st2_adv = adv
        self._st2_cur = None
        self._st2_ctr = None
        self._st2_bi = self.all_b.view(self.B, 1)

    def _st2_pass(self, plan) -> dict:
        """Per pass (cached on the plan object): the step sequence of every request (module docstring, ST2)."""
        cur = self._st2_cur
        if cur is not None and cur[0] is plan:
            return cur[1]
        B, R = self.B, plan.R
        T = R + 1
        dev = self.dev
        act = plan.act                                                 # (R, B) bool: a prefix of rows per request
        rr = torch.arange(R, dtype=torch.int64, device=dev).view(R, 1)
        rows = tuple(plan.replay_rows)
        if plan.replay is not None and rows:
            keep = torch.zeros((R, 1), dtype=torch.bool, device=dev)
            keep[list(rows)] = True
            rmask = plan.replay & keep                                 # the replays of spec_tentative.py:339-340
        else:
            rmask = torch.zeros((R, B), dtype=torch.bool, device=dev)
        rep = rmask.any(0)
        if self.tneg == "no_replay" or self._st2["neg"] == "no_replay_seq":   # NEGATIVE CONTROLS (tests only)
            seq_rep = torch.zeros_like(rep)
        else:
            seq_rep = rep
        vb = (rmask.to(torch.int64) * rr).sum(0)                       # (B,): the row the replay precedes
        jj = torch.arange(T, dtype=torch.int64, device=dev).view(1, T)
        v1, s1 = vb.view(B, 1), seq_rep.view(B, 1)
        idx = torch.where(s1 & (jj == v1), torch.full_like(jj, R), torch.where(s1 & (jj > v1), jj - 1, jj))
        idx = idx.clamp(max=R)                                         # (B, T): 0 .. R - 1 a row's key, R the replay's
        n = act.to(torch.int64).sum(0) + seq_rep.to(torch.int64)       # (B,): the steps of each request
        shift = seq_rep.view(1, B) & (rr >= vb.view(1, B))
        if self._st2["neg"] == "no_shift":                             # NEGATIVE CONTROL (tests only)
            shift = torch.zeros_like(shift)
        S = rr + shift.to(torch.int64)                                 # (R, B): the step index of row r
        out = dict(idx=idx, n=n.contiguous(), S=S, S1=S + 1, i=(plan.deferred % self.bs) if plan.deferred is not None
                   else torch.zeros((B,), dtype=torch.int64, device=dev), nrep=rmask.sum(), T=T)
        self._st2_cur = (plan, out)
        self._st2["passes"] += 1
        return out

    def _st2_twin(self, l, twin, k4, c3, plan):
        """The twin section of layer l (module docstring, ST2). Returns pbs (the R rows' positions before their step)."""
        B, R = self.B, plan.R
        pp = self._st2_pass(plan)
        T = pp["T"]
        if l == 0:                                                     # TentState._replay's host count and counter
            self.calls["replay_calls"] += len(tuple(plan.replay_rows))
            if self.tneg != "no_replay" and plan.replay_rows:
                self.c["replays"] += pp["nrep"]
        i = pp["i"]
        dk, dc = twin.ring.dtype, twin.total_cis.dtype
        kr = self.k[l][self.all_b, i]                                  # spec_tentative.py:386-387 (the deferred row)
        cr = self.bias[l][self.all_b, i]
        ek = torch.cat((k4.to(dk), kr.to(dk).unsqueeze(1)), 1)         # (B, R + 1, H, D)
        ec = torch.cat((c3.to(dc), cr.to(dc).unsqueeze(1)), 1)         # (B, R + 1, H)
        bi = self._st2_bi
        sk = ek[bi, pp["idx"]]                                         # (B, T, H, D): each request's steps in order
        sc = ec[bi, pp["idx"]]
        if self._st2_adv is not None:                                  # the fused twin: ONE kernel for T steps
            if self._st2_ctr is None or self._st2_ctr.shape[0] != T + 1:
                self._st2_ctr = torch.empty((T + 1, 4, B), dtype=torch.int64, device=self.dev)
            ctr = self._st2_adv(twin, sk, sc, pp["n"], out=self._st2_ctr)
        else:                                                          # the torch twin: step_unchecked per step
            ctr = twin.advance_unchecked(sk, sc, pp["n"])
        pb = ctr[:, 0, :].gather(0, pp["S"])                           # (R, B): the position before row r's step
        plan.vs.n_used.copy_(ctr[:, 1, :].gather(0, pp["S1"]).t())     # spec_tentative.py:348
        plan.vs.M_b.copy_((pb // self.bs + 1).t())                     # spec_tentative.py:349
        if l == 0 and plan.snap:                                       # spec_tentative.py:350-351
            cs = ctr[:, :3, :].gather(0, pp["S1"].view(R, 1, B).expand(R, 3, B))
            for r in range(R):
                plan.ctr[r] = cs[r]
        self._st2["layers"] += 1
        return list(pb.unbind(0))

    def _layer(self, l, layer, hidden, pos_flat, cos_sin_cache, plan):
        """TentState._layer (spec_tentative.py:318-369) with the twin loop of a joint pass in one advance (ST2)."""
        base = self._st2_base
        if plan.R == 1 or self.rec is not None:
            if plan.R > 1:
                self._st2["fallback"] += 1
            return base(self, l, layer, hidden, pos_flat, cos_sin_cache, plan)
        VR = self._st2_vr
        rs = self.rs
        NL, st, lin = rs.nl, rs.st, plan.lin
        B, R = self.B, plan.R
        E = B * R
        Hkv, D = layer.num_key_value_heads, layer.head_dim
        twin = st.twins[l]
        eng = rs.cache.layers[l].cache_engine
        residual = hidden
        h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = lin(h, layer.wqkv)
        q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
        q = q.view(E, -1)
        k = k.view(E, -1)
        NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
        q = q.reshape(E, layer.num_heads, D)
        k4 = k.reshape(B, R, Hkv, D)
        v4 = v.reshape(B, R, Hkv, D)
        c3 = cis.reshape(B, R, Hkv)
        pbs = self._st2_twin(l, twin, k4, c3, plan)                    # ST2: 2. the twin (and the deferred replay)
        sel = VR._select_rows(plan.vs, layer, q, twin)                 # 3. the selection
        q4 = q.view(B, R, q.shape[1], q.shape[2])
        attn = self._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan).reshape(E, 1, -1)
        hidden = residual + lin(attn, layer.wo)
        residual = hidden
        h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = lin(h, layer.gate_up_proj)
        out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        NL.silu_and_mul(h, out)
        return residual + lin(out, layer.down_proj)

    def st2_report(self) -> dict:
        return dict(self._st2)


_STATES = {}
_RUNNERS = {}


def state_class(S, twin: bool = False):
    """The state class S with the mixin(s) (cached): CommitST1, and with twin also TwinST2."""
    key = (S, bool(twin))
    if key not in _STATES:
        bases = ((TwinST2,) if twin else ()) + (CommitST1, S)
        _STATES[key] = type(S.__name__ + ("ST12" if twin else "ST1"), bases, {})
    return _STATES[key]


def st1_runner(R, mode: str = "rows", sneg=None, twin: bool = False, tneg2=None):
    """The runner class R (a best arm's) whose state commits by ST1 in `mode` and, with twin, runs the joint pass's
    twin section by ST2 (cached per (R, mode, sneg, twin, tneg2)). Its __init__ takes st1_backend (None:
    CudaGraphsPrivate) besides R's arguments."""
    if mode not in MODES or sneg not in SNEG or tneg2 not in TNEG2:
        raise ValueError("ST1: mode %r (one of %s), control %r (one of %s), twin control %r (one of %s)" % (
            mode, MODES, sneg, SNEG, tneg2, TNEG2))
    key = (R, mode, sneg, bool(twin), tneg2)
    if key not in _RUNNERS:
        class RunnerST1(R):
            _st1_mode, _st1_sneg, _st2_on, _st2_neg = mode, sneg, bool(twin), tneg2

            def __init__(self, *args, st1_backend=None, **kw):
                super().__init__(*args, **kw)
                T = self.T
                T.__class__ = state_class(type(T), self._st2_on)
                T._st1_setup(self._st1_mode, self._st1_sneg,
                             st1_backend if st1_backend is not None else CudaGraphsPrivate(), self.vs, int(self.K))
                if self._st2_on:
                    T._st2_setup(self._st2_neg)
                self.meta.update(st1=dict(mode=self._st1_mode, sneg=self._st1_sneg, J=T._st1["J"],
                                          backend=type(T._st1["backend"]).__name__, twin=self._st2_on,
                                          tneg2=self._st2_neg))

            def read(self) -> dict:
                out = super().read()
                out.update(st1=self.T.st1_report())
                if self._st2_on:
                    out.update(st2=self.T.st2_report())
                return out

            def detach(self) -> None:
                self.T._st1_release()
                super().detach()

        RunnerST1.__name__ = RunnerST1.__qualname__ = R.__name__ + ("ST12" if twin else "ST1")
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
