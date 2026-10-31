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
ST3 (KeepFillST3; the keep arms: L4PTFDGUDKWCDMTK = the MT keep arm + ST3, and the INT keep arm through st3_runner).
EVIDENCE (MEASURED, /mnt/beegfs/ojerbi/r2-runs/spec/recon7_pb1_2195630_r7pb/summary.json, L4PTFDGUDKWCDP round 112,
labelled joint window, C 73 B 318, label "section"): SMALL 2,785 ops / 16.32 ms per joint pass = 87 per layer, against
1,473 / 6.94 ms = 46 per layer in the C 128 arm without the keep (r7pa); SELECT 224 vs 192 ops (7 vs 6 per layer: one
more radix sort). The section's named small kernels: reduce MaxOps<unsigned char> 32 ops 2.34 ms (THE KEEP's map
lookup, E.view(uint8).max(-1), one per layer, 73 us each), where 108 ops 2.49 ms (96 = the chain images' 3 per layer),
radixSortKVInPlace 96 ops 3.26 ms (3 per layer: host key, row-list key, the keep's score); the window's CompareEq<long>
96 ops 2.41 ms, of which 64 / 0.95 ms in "resid" (the kwin lookup), so 32 / 1.46 ms in the section (the keep's E).
ATTRIBUTION (CODE, per layer): the keep fill (r2_i4k_combo.TentStateFKg._keep_fill + keep_assign_d; KP.KEEP_EXTRA_OPS
= 30 counted there) issues about 38 SMALL launches + 1 sort: need, E, max, any, td - L, alive, ALIVE_BASE - a1, where,
masked_fill, valid, ~hit, miss, cumsum, - 1, clamp, gather, where, + 1, where, kid scatter, where, kage scatter, the
step (2), hx fill, where, hx scatter, stack, sum, add, the remap (5). 87 - 46 = 41 per layer: the keep fill IS the
difference (about 1,300 of the 2,785; the rest = the C 128 section's chain images 6, buffer writes, the compiled
helpers' kernels, the sorts' index fills, the counters). It is the largest avoidable part: the E compare and its two
reductions read 2 x 10.4 M bools per layer (E = (H, B, cap, cap) = 2 x 318 x 128 x 128) and every other op is a
separate launch. THE CHANGE: the same ops in TWO compiled helpers (kfa_fn: the hits, the protection, the scores, the
misses and their ranks; Inductor fuses the compare into the two reductions, E is never written; kfb_fn: the slots, the
map and age scatters, hx, the counters, the remap), the stable sort between them EAGER (the I4CC rule: aten.sort's
ties), the two shipped gathers after kfb_fn (they read hx). Integer / bool ops only (exact); the layer index and the
pass step are 0-d device tensors and the FORK_BIG bound is vs.big (one compile per helper for every layer and pass).
Column 0 of kid (CB1's slot, a write target never read) takes duplicate scatter writes in both versions. A CHECK PASS
runs the helpers too: keep_assign_d's check counters (overflow from kfa's used / rank / miss; dup and missing from kid
after kfb) and the byte check (_keep_bytes) run eagerly on the helpers' outputs (the GPU miniature checks every round:
its keep gates then check ST3's map). FALLBACK (the inherited eager fill, counted per reason): "acct" (kacct), "control"
(a keep or combo control: kneg, cneg), "kslots" (a blocked slot set), "nocc" (no compiled C / D state). The helper
calls are counted in the state's compile record (cc["calls"]); the compile EVENTS go into the ST3 record only (the D
root's -COMPILE gate wants cc's events before the first capture of any kind; ST3 compiles in the first joint pass,
after round 0's draft captures, outside any capture, as J1's F6 does): the cell's -KFILL gate checks them.
NEGATIVE CONTROLS (tests only; tneg3): "no_age" (kage not updated: the map's eviction order changes), "remap_off" (the
page-id remap reads the slot of the next list rank: a fork row reads another block).
ST4 (ResidST4; the arms L4PTFDGUDCDMTR = the MT arm + ST4 at C 128, L4PTFDGUDKWCDMTKR = the MTK arm + ST4 at C 73, and
both INT arms through st4_runner): THE DRAFT RESIDENCY'S INTEGER OPS IN ONE COMPILED HELPER. EVIDENCE (MEASURED, r7pa /
r7pb summary.json, the draft passes of L4PTFDGUDCDP / L4PTFDGUDKWCDP round 112, labels A / resid / B): SMALL 1,616
kernels per one-row pass (5.40 / 6.22 ms) = A 465 (1.12 / 1.24 ms) + resid 448 (1.65 / 1.79 ms; + COPY 64, 0.15 /
0.16 ms) + B 702 (2.54 / 3.06 ms) + head 1. The I4C draft trace (KS/cpu-logs/i4cc_ana_s1.txt:442-638, the eager
L4PTFDP pass at C 128, per layer) names them: B = the float32 log-sum-exp merge (13 ops, 55 us), the 2 residual adds
(float), the compiled dwrite / dmask kernels (I4DC; the eager write + masks were 11 ops, 124 us); A = the selection's
small ops (score copy, fills, mask zero / scatter / masked_fill: J1 F2's region), the twin's pos clone / pos_before
copy / lengths (6 ops, TentState._layer and ragged_engine._lengths) and the compiled dpre; resid = the eager residency
r2_i4a_lean.TentStateFD._resid_lean: ~am, masked_fill, where, 2 copies, gt, ge, or, and, where, masked_fill, ~tent,
and, sum, sum copy, add_ = 14 SMALL + 2 COPY per layer (about 52 / 61 us), between the shipped diff, pool_update,
pool_swap and the two gathers. It is EAGER in the I4H draft graphs (pool_update's host stamp) and in the C 73 INT arm
(BF1's on_draft hook); inside the graph in the C 128 INT arm (J1 F3, the stamp frame). Every op is integer / bool /
copy: the bit-exact part of the draft pass that no earlier rung compiled. THE CHANGE: ResidST4._resid_lean = the same
code with the ops after pool_swap (the block-map update, tent, skip, ids, the block-map mask of the skipped loads and
the skipped / check counters) in ONE compiled helper (rpost_fn) before the two shipped gathers; ~am and the load-mask
fill stay eager (pool_update reads them). The block-map mask moves before the gathers: they read ids only (the same
final bytes). The counters add into the layer's entry by a one-hot where over (L,) (no atomics, no view of a row).
PLACEMENT: ResidST4 is inserted DIRECTLY ABOVE the class that defines the original _resid_lean (r2_i4a_lean's
TentStateFD) by a C3 merge (st4_state: type(name, (S, Bottom), {}) with Bottom = (ResidST4, TentStateFD)), so every
override above it (BF1's TentStateBF._resid_lean: super() then on_draft; I4H's _dstub; J1 F3's captured call) still
runs and reaches it. FALLBACK (counted): "nocc" (no compiled state). Compile: the first one-row pass of a run (eager or
I4H's warm pass); events in the ST4 record only. NEGATIVE CONTROLS (tests only; tneg4): "no_block_mask" (the skipped
loads stay in the block map: a later row reads a block that was never loaded), "count_off" (the skipped counter is not
added).
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
TNEG3 = (None, "no_age", "remap_off")
HELPERS3 = ("kfa", "kfb")
KM = 1 << 32                       # = r2_i4d_keep.KM (key = layer x KM + block id; tests_cpu/test_st1.py checks)
ALIVE_BASE = 1 << 50               # = r2_i4d_keep.ALIVE_BASE
PROTECT = 1 << 62                  # = r2_i4d_keep.PROTECT
TNEG4 = (None, "no_block_mask", "count_off")
HELPERS4 = ("rpost",)


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


# ===================================================================================================================
# ST3: the keep fill in two compiled helpers (module docstring, ST3). Pure functions: tensors in, tensors out or
# in-place writes to tensors passed in; no state object, no host read; Python ints / bools are constants of a graph.
# ===================================================================================================================
def kfa_fn(lst, kid, kage, kpass_d, l_t, big, L: int):
    """r2_i4k_combo.keep_assign_d up to the sort (kneg None, block None, no stats, no checks): the key, the hits (at most
    one per need: the keys are distinct), the slots that hold a needed key (protected), the alive test of the device
    step, the scores, the valid entries, the misses and their ranks. l_t, kpass_d, big: 0-d int64 tensors."""
    need = lst + l_t * KM                                              # KP: lst + int(l) x KM
    k1, a1 = kid[..., 1:], kage[..., 1:]
    E = need.unsqueeze(-1) == k1.unsqueeze(-2)                         # (H, B, needs, slots)
    hv, hs = E.to(torch.uint8).max(-1)
    hit = hv != 0
    used = E.any(-2)                                                   # (H, B, slots): holds a needed key
    td = kpass_d * L + l_t                                             # = KP's int(t), on the device
    alive = a1 > (td - L)
    score = torch.where(alive, ALIVE_BASE - a1, a1).masked_fill(used, PROTECT)
    valid = lst < big
    miss = valid & ~hit
    rank = miss.cumsum(-1) - 1
    return score, hit, hs, miss, valid, rank, need, td, used


def kfb_fn(order, hit, hs, miss, valid, rank, need, td, kid, kage, hx, lst, minus1, ph, in_lst, sb, sb1, row,
           kcap: int, age: bool, shift: int):
    """keep_assign_d after the sort (order = the stable sort's indices, eager) and TentStateFKg._keep_fill without the
    shipped gathers: the slots, the map and age scatters (in place: kid, kage), hx (in place), the counter row (in place:
    row[0:2]), the page-id remap. Returns (the remapped phys, the slots). age False / shift 1: the controls no_age /
    remap_off."""
    ms = order.gather(-1, rank.clamp(min=0))
    slot = torch.where(hit, hs, ms) + 1
    widx = torch.where(miss, slot, 0)
    kid.scatter_(-1, widx, need)
    if age:
        ws = torch.where(valid, slot, 0)
        kage.scatter_(-1, ws, td.expand(ws.shape))
    hx.fill_(-1)                                                       # slot 0 (CB1) and every hit: no load
    hx.scatter_(-1, widx, torch.where(miss, lst, minus1))
    row[0:2].add_(torch.stack((valid, miss)).sum((1, 2, 3)))
    at = (ph - sb1 + shift).clamp(0, kcap - 1)                         # the list rank of a listed block
    sq = slot.gather(-1, at.view(at.shape[0], at.shape[1], -1)).view_as(at)
    return torch.where(in_lst, sb + sq, ph), slot


FNS3 = dict(kfa=kfa_fn, kfb=kfb_fn)
_COMPILED3 = {}


def compiled3(name: str, backend: str):
    """The compiled ST3 helper (one per (name, backend) per process; fullgraph, static shapes, as the I4CC helpers)."""
    key = (name, backend)
    f = _COMPILED3.get(key)
    if f is None:
        f = _COMPILED3[key] = torch.compile(FNS3[name], fullgraph=True, dynamic=False, backend=backend)
    return f


def _graphs_now() -> int:
    """Dynamo's process-wide count of compiled graphs (the compile-event record; read on the host only)."""
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


class KeepFillST3:
    """ST3 (module docstring): the keep fill of the compiled keep section through kfa_fn / kfb_fn. A mixin before a
    keep state of the I4DC line (L4PTFDGUDKWCD and the states built on it: ST1 / ST2, J1, BF1, INT). Set up by
    _st3_setup (st3_attach)."""

    def _st3_setup(self, tneg3) -> None:
        if tneg3 not in TNEG3:
            raise ValueError("ST3: control %r (one of %s)" % (tneg3, TNEG3))
        if getattr(self, "kid", None) is None or not getattr(self, "kkeep", False):
            raise ValueError("ST3: a keep state only (kid is None)")
        self._st3_l = [torch.full((), l, dtype=torch.int64, device=self.dev) for l in range(int(self.L))]
        self._st3 = dict(neg=tneg3, calls=0, checked=0, fallback={}, events=[], compile_s=0.0,
                         backend=getattr(self, "_cc_backend", None))

    def _st3_why(self):
        """None: the fill runs through the helpers; else the reason for the inherited eager fill."""
        if getattr(self, "_cc_backend", None) is None or getattr(self, "cc", None) is None:
            return "nocc"
        if self.kacct:
            return "acct"
        if self.kneg is not None or getattr(self, "cneg", None) is not None:
            return "control"
        if self.kblock is not None:
            return "kslots"
        return None

    def _st3_call(self, name, l, *args):
        """One call of the compiled helper `name` with its compile event (_CCMix._cc's fields). The call count goes into
        cc["calls"]; the EVENT goes into the ST3 record only: the D root's -COMPILE gate (r2_i4dc_compile.compile_gate_cd)
        wants every event of cc before the arm's first capture of ANY kind, and ST3 compiles in the first joint pass,
        after round 0's draft captures (as J1's F6 locate; -KFILL checks: before the first JOINT capture, none inside a
        capture of either kind)."""
        f = compiled3(name, self._cc_backend)
        g0 = _graphs_now()
        t0 = time.perf_counter()
        wrap = getattr(self, "_cc_wrap", None)                         # tests: the CPU capture stand-in's node maker
        out = wrap(name, f, args) if wrap is not None else f(*args)
        dt = time.perf_counter() - t0
        g1 = _graphs_now()
        cc = self.cc
        cc["calls"][name] = cc["calls"].get(name, 0) + 1
        if g1 != g0:
            cap = bool(getattr(self, "_cc_cap", False)) or bool(getattr(self, "_cd_cap", False))
            self._st3["events"].append(dict(name=name, graphs=g1 - g0, s=round(dt, 3), layer=int(l),
                                            joint_pass=int(self.calls["joint_passes"]), capture=cap))
            self._st3["compile_s"] += dt
        return out

    def _keep_fill(self, l, vs, eng, twin, lst, loc, kern):
        """TentStateFKg._keep_fill (r2_i4k_combo) through kfa_fn, the eager stable sort and kfb_fn; the same tensors
        written with the same values (kid's column 0 aside). A check pass adds keep_assign_d's check counters (eager, on
        the helpers' outputs, at the same points of the map's update) and the byte check (_keep_bytes)."""
        why = self._st3_why()
        if why is not None:
            self._st3["fallback"][why] = self._st3["fallback"].get(why, 0) + 1
            return super()._keep_fill(l, vs, eng, twin, lst, loc, kern)
        neg = self._st3["neg"]
        chk = self.kchk if self.checking else None
        sb, sb1 = self._ksb1                                           # made in _kpre (eager)
        score, hit, hs, miss, valid, rank, need, td, used = self._st3_call(
            "kfa", l, lst, self.kid, self.kage, self.kpass_d, self._st3_l[l], vs.big, int(self.L))
        if chk is not None:                                            # CHECK-ONLY (keep_assign_d): (iii) no overflow
            nfree = (~used).sum(-1, keepdim=True)
            chk["overflow"] += (miss & (rank >= nfree)).sum()
        order = torch.sort(score, dim=-1, stable=True).indices         # EAGER (KP: the same call)
        lphys, slot = self._st3_call("kfb", l, order, hit, hs, miss, valid, rank, need, td, self.kid, self.kage,
                                     vs.host_x, lst, vs.minus1, loc["phys"], loc["in_lst"], sb, sb1, self._crow(),
                                     int(self.kcap), neg != "no_age", 1 if neg == "remap_off" else 0)
        if chk is not None:                                            # CHECK-ONLY: (ii) distinct keys, (iv) present
            s = torch.sort(self.kid[..., 1:], dim=-1).values
            chk["dup"] += ((s[..., 1:] == s[..., :-1]) & (s[..., 1:] >= 0)).sum()
            chk["missing"] += (valid & (self.kid.gather(-1, slot) != need)).sum()
        bs = self.bs
        kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
        kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
        if chk is not None and self.kcheck:                            # CHECK-ONLY: the store bytes (KP _keep_bytes)
            self._keep_bytes(vs, eng, twin, lst, valid, slot)
        self._st3["calls"] += 1
        self._st3["checked"] += int(chk is not None)
        return lphys, valid

    def st3_report(self) -> dict:
        s = dict(self._st3)
        s["fallback"] = dict(s["fallback"])
        s["events"] = list(s["events"])
        s["compile_s"] = round(s["compile_s"], 3)
        return s


# kfb_fn's written arguments (positions; the CPU capture stand-in saves them before the first write)
KFB_WRITES = (8, 9, 10, 17)


# ===================================================================================================================
# ST4: the draft residency's integer ops in one compiled helper (module docstring, ST4)
# ===================================================================================================================
def rpost_fn(nb, bm, lm, am, qb3, drf3, minus1, skipped, forced, injected, inj, arL, l_t, chk, neg: int):
    """r2_i4a_lean.TentStateFD._resid_lean after pool_swap (r2_i4a_lean.py:211-223), the gathers aside: the block-map
    update (nb = eng._new_block_map_buf, bm = eng._block_map, in place), tent, skip, ids, the block map's skipped loads
    (-1), the counters (in place: the layer's entry of skipped; where chk forced / injected). Returns (skip, ids).
    chk: a 0-d bool tensor in production (ONE graph for check and timing passes: a Python bool would compile a second
    graph at the first check pass, INT CPU run of 00bfb51, -DRESID), a Python bool in the planted test. Off: the
    counters get + 0 (the same values). neg 1: no block-map mask; neg 2: no skipped count (the controls)."""
    nbv = torch.where(am, nb, bm)
    nb.copy_(nbv)
    tent = lm > qb3                                                    # = (lm >= 0) & (lm > qb3): qb3 >= 0
    skip = (lm >= 0) & (drf3 | tent)
    ids = torch.where(skip, minus1, lm)
    bm.copy_(nbv if neg == 1 else nbv.masked_fill(skip, -1))           # LN: copy_, the gathers, masked_fill_
    sel = arL == l_t                                                   # (L,): the layer's entry
    if neg != 2:
        skipped.add_(torch.where(sel, (skip & ~tent).sum(), 0))
    on = sel & chk                                                     # CHECK-ONLY counters (LN:221-223)
    forced.add_(torch.where(on, tent.sum(), 0))
    injected.add_(torch.where(on, inj.sum(), 0))
    return skip, ids


FNS4 = dict(rpost=rpost_fn)
_COMPILED4 = {}
RPOST_WRITES = (0, 1, 7, 8, 9)           # rpost_fn's written arguments (the CPU capture stand-in)


def compiled4(name: str, backend: str):
    key = (name, backend)
    f = _COMPILED4.get(key)
    if f is None:
        f = _COMPILED4[key] = torch.compile(FNS4[name], fullgraph=True, dynamic=False, backend=backend)
    return f


class ResidST4:
    """ST4 (module docstring): the draft residency with the integer ops after pool_swap in rpost_fn. Placed DIRECTLY
    ABOVE the class that defines the original _resid_lean (st4_state); set up by _st4_setup (st4_attach)."""

    def _st4_setup(self, tneg4) -> None:
        if tneg4 not in TNEG4:
            raise ValueError("ST4: control %r (one of %s)" % (tneg4, TNEG4))
        L, dev = int(self.L), self.dev
        engs = [lay.cache_engine for lay in self.rs.cache.layers]
        for e in engs:                                                 # host attribute reads: no aliased inputs
            p = {t.untyped_storage().data_ptr() for t in (e._block_map, e._new_block_map_buf, e._load_mask)}
            if len(p) != 3:
                raise ValueError("ST4: the block map, its buffer and the load mask share a storage")
        self._st4_ar = torch.arange(L, dtype=torch.int64, device=dev)
        self._st4_l = [torch.full((), l, dtype=torch.int64, device=dev) for l in range(L)]
        self._st4_chk = [torch.full((), c, dtype=torch.bool, device=dev) for c in (False, True)]   # rpost_fn's chk
        self._st4 = dict(neg=tneg4, calls=0, entered=0, fallback={}, events=[], compile_s=0.0,
                         backend=getattr(self, "_cc_backend", None))

    def _st4_why(self):
        if getattr(self, "_st4", None) is None:
            return "setup"
        if getattr(self, "_cc_backend", None) is None:
            return "nocc"
        return None

    def _resid_lean(self, l, eng, twin, ti, inj, act, drf, qb3):
        """r2_i4a_lean.TentStateFD._resid_lean (its host count, the shipped kernels, the host stamp, the gathers) with
        the ops after pool_swap in rpost_fn (module docstring, ST4)."""
        why = self._st4_why()
        st = getattr(self, "_st4", None)
        if st is not None:
            st["entered"] += 1
        if why is not None:
            if st is not None:
                st["fallback"][why] = st["fallback"].get(why, 0) + 1
            return super()._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)
        self.calls["resid"] += 1
        rs, B, bs = self.rs, self.B, self.bs
        kern = rs.ops.kernels
        kern["diff"](eng._block_map, ti, eng._new_block_map_buf, eng._load_mask)
        am = act.view(1, B, 1)
        eng._load_mask.masked_fill_(~am, -1)                           # an inactive row loads nothing, moves nothing
        if self.pool:
            kern["pool_update"](eng._block_map, eng._load_mask, eng._pool_map, eng._pool_age, eng._pool_target,
                                eng._pool_action, eng._pool_stamp_base, eng._tail_block_idx_on_gpu)
            kern["pool_swap"](eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action,
                              eng.topk, eng.block_size)
            eng._pool_stamp_base += eng.topk
        neg = {"no_block_mask": 1, "count_off": 2}.get(st["neg"], 0)
        skip, ids = self._st4_call("rpost", l, eng._new_block_map_buf, eng._block_map, eng._load_mask, am, qb3,
                                   drf.view(1, B, 1), self.minus1, self.skipped, self.forced, self.injected, inj,
                                   self._st4_ar, self._st4_l[l], self._st4_chk[int(bool(self.checking))], neg)
        kern["gather_k"](eng._k_gpu, eng._k_cpu, ids, bs)
        kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, ids, bs)
        st["calls"] += 1
        return skip

    def _st4_call(self, name, l, *args):
        """One call of the compiled helper with its compile event (the ST4 record only, as ST3's)."""
        f = compiled4(name, self._cc_backend)
        g0 = _graphs_now()
        t0 = time.perf_counter()
        wrap = getattr(self, "_cc_wrap", None)                         # tests: the CPU capture stand-in's node maker
        out = wrap(name, f, args) if wrap is not None else f(*args)
        dt = time.perf_counter() - t0
        g1 = _graphs_now()
        if g1 != g0:
            cap = bool(getattr(self, "_cc_cap", False)) or bool(getattr(self, "_cd_cap", False))
            self._st4["events"].append(dict(name=name, graphs=g1 - g0, s=round(dt, 3), layer=int(l),
                                            joint_pass=int(self.calls["joint_passes"]), capture=cap,
                                            dpass=int(getattr(self, "lean", {}).get("passes", -1))))
            self._st4["compile_s"] += dt
        return out

    def st4_report(self) -> dict:
        s = dict(self._st4)
        s["fallback"] = dict(s["fallback"])
        s["events"] = list(s["events"])
        s["compile_s"] = round(s["compile_s"], 3)
        return s


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


_STATES3 = {}
_RUNNERS3 = {}


def st3_state(S):
    """(KeepFillST3, S) (cached)."""
    if S not in _STATES3:
        _STATES3[S] = type(S.__name__ + "ST3", (KeepFillST3, S), {})
    return _STATES3[S]


def st3_attach(runner, tneg3=None) -> None:
    """Re-class the runner's state to (KeepFillST3, its class) and set ST3 up (st3_runner's __init__ calls it after
    the base runner's __init__: after ST1's, INT's and BF1's own re-classes)."""
    T = runner.T
    T.__class__ = st3_state(type(T))
    T._st3_setup(tneg3)
    runner.meta.update(st3=dict(tneg3=tneg3, helpers=HELPERS3))


def st3_runner(R, tneg3=None):
    """The runner class R (a keep arm's: the MT runner of L4PTFDGUDKWCD, or the INT keep runner r2_int_arms.int_runner)
    whose state fills the keep by ST3 (cached per (R, tneg3)). read() adds the ST3 record ("st3")."""
    if tneg3 not in TNEG3:
        raise ValueError("ST3: control %r (one of %s)" % (tneg3, TNEG3))
    key = (R, tneg3)
    if key not in _RUNNERS3:
        class RunnerST3(R):
            _st3_neg = tneg3

            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                st3_attach(self, self._st3_neg)

            def read(self) -> dict:
                out = super().read()
                out.update(st3=self.T.st3_report())
                return out

        RunnerST3.__name__ = RunnerST3.__qualname__ = R.__name__ + "ST3"
        _RUNNERS3[key] = RunnerST3
    return _RUNNERS3[key]


_STATES4 = {}
_RUNNERS4 = {}


def resid_owner(S):
    """The DEEPEST class of S's MRO that defines _resid_lean (r2_i4a_lean's TentStateFD)."""
    owners = [c for c in S.__mro__ if "_resid_lean" in c.__dict__]
    if not owners:
        raise ValueError("ST4: %s has no _resid_lean (not a lean one-row state)" % S.__name__)
    return owners[-1]


def st4_state(S):
    """S with ResidST4 directly above the original _resid_lean (module docstring, PLACEMENT; cached). The new class's
    MRO: S's classes down to the original's subclasses, then Bottom, ResidST4, the original, the rest."""
    if S not in _STATES4:
        if ResidST4 in S.__mro__:
            raise ValueError("ST4: %s has ResidST4 already" % S.__name__)
        base = resid_owner(S)
        bottom = type(base.__name__ + "ST4", (ResidST4, base), {})
        new = type(S.__name__ + "ST4", (S, bottom), {})
        mro = new.__mro__
        i = mro.index(ResidST4)
        if mro[i + 1] is not base or [c for c in mro[i + 1:] if "_resid_lean" in c.__dict__][:1] != [base]:
            raise ValueError("ST4: ResidST4 is not directly above %s in %s" % (base.__name__, new.__name__))
        _STATES4[S] = new
    return _STATES4[S]


def st4_attach(runner, tneg4=None) -> None:
    """Re-class the runner's state with ResidST4 (st4_state) and set ST4 up (after the base runner's __init__)."""
    T = runner.T
    T.__class__ = st4_state(type(T))
    T._st4_setup(tneg4)
    runner.meta.update(st4=dict(tneg4=tneg4, helpers=HELPERS4))


def st4_runner(R, tneg4=None):
    """The runner class R (an MT / MTK runner, or an INT runner r2_int_arms.int_runner) whose draft residency runs by
    ST4 (cached per (R, tneg4)). read() adds the ST4 record ("st4")."""
    if tneg4 not in TNEG4:
        raise ValueError("ST4: control %r (one of %s)" % (tneg4, TNEG4))
    key = (R, tneg4)
    if key not in _RUNNERS4:
        class RunnerST4(R):
            _st4_neg = tneg4

            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                st4_attach(self, self._st4_neg)

            def read(self) -> dict:
                out = super().read()
                out.update(st4=self.T.st4_report())
                return out

        RunnerST4.__name__ = RunnerST4.__qualname__ = R.__name__ + "ST4"
        _RUNNERS4[key] = RunnerST4
    return _RUNNERS4[key]


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
