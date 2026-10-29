"""Rung I4H of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), the "draft-step graph"
of the specKV-on-cached-NOSA ladder (option (b) of KS/analysis/plan/HANDOFF_2026-10-29.md; LEDGER_rev6.md R6.6 E1): ARM
L4PTFDGD = L4PTFDG (rung I4G: L4PTFD with the joint pass's union + fork section from CUDA graphs, r2_i4g_graph.py) whose
DRAFT (one-row) passes run from per-layer CUDA graphs. Docs: docs/NOTES_i4h.md. OPT-IN: benchmarks/Efficiency/
r2_i4h_cell.py (arm L4PTFDGD); no nosi module and no older cell imports this file. NO NEW KERNEL and no kernel edit:
torch.cuda.CUDAGraph (capture_begin / capture_end on a side stream, one shared pool), the shipped kernels and torch ops.
This file imports torch and time only.

WHY A CLASS FACTORY (as r2_i4g_graph.graph_classes): draft_classes(STF, base) builds the classes on the GIVEN
spec_tentative_fork module and on the GIVEN graph pair base = (TentStateFDG, TentRunnerFDG) of
r2_i4g_graph.graph_classes(STF, r2_i4a_lean.lean_classes(STF)).

WHAT IS CAPTURED. A one-row pass (R = 1: the prelude's K drafts, the D-slot passes, the one-row J slots) runs, per layer
l, TentState._layer (spec_tentative.py:318-369) with TentStateFD._row_lean (r2_i4a_lean.py:145-194). Its ONE part that
cannot be replayed is the residency (TentStateFD._resid_lean, r2_i4a_lean.py:196-224): pool_update takes the host int
eng._pool_stamp_base by value and the host then adds topk (spec_tentative.py:461-466); a device stamp needs an edit of
pool_update_kernel.cu (not allowed). So each layer is split at the residency into TWO graphs:
  A(l)  layer 0 only: the embedding (spec_tentative.py:308); then the input norm, the QKV GEMM, nosa_linear, rope, the
        twin step and the lengths, stage 1, the selection (pooling + top-k), and _row_lean up to the residency (layer 0:
        the hoisted tensors of the pass; hasq, the injection, ti).
  EAGER the residency (TentStateFD._resid_lean: its host count, the shipped kernels, the host stamp, the gathers, the
        meter), on A(l)'s outputs; its skip mask is copied into ONE static mask.
  B(l)  the buffer write (layer 0: tpos and the device counters; then call B's hoisted masks), call A, call B, the
        log-sum-exp merge, the output projection, the post norm, the MLP and the residual adds.
The final norm and the LM head stay eager (one large GEMM: no gap to remove). THE SPLIT IS MADE INSIDE ONE CALL of the
layer's own code: the capture of A(l) begins before TentState._layer(l) runs, and a stub in place of _resid_lean ends
A(l), replays it, runs the REAL residency eagerly, copies the skip mask and begins B(l); B(l) ends when the layer
returns. So the captured Python is the shipped L4PTFD code, line for line (no copy of it).
THE SELECTION GRAPH. ragged_engine._select (ragged_engine.py:346-370) replays the shipped per-layer pooling / top-k
graph. A graph launch inside a stream capture is not used here: during A(l)'s capture rs.graphs[l] is replaced by
_Inline, whose replay() runs ragged_engine._pool_topk on the SAME tensors (the arguments of ragged_engine.py:355-356),
so A(l) holds those kernels. The same kernels on the same inputs (the same bits).
THE ORDER per KEY (the Python branches inside the captured code: T.checking, the layer-0 check-only counters of
_write_lean; plan.snap, the layer-0 counter snapshot of spec_tentative.py:350-351): the first one-row pass of a key runs
EAGERLY on the static inputs (on the capture side stream: Triton compiles outside the capture, cuBLAS gets its stream
workspace); the second CAPTURES 2 L graphs (capture executes nothing; each graph is replayed right after its capture);
every later pass of the key replays 2 L graphs with L eager residencies between them.
STATIC INPUTS (one set, copy_ before every pass): plan.ids (B, 1), plan.pos (B, 1), plan.act / verify / draft rows.
Everything else that the graphs read is persistent (weights, the engine, the twins, the buffers, the selection buffers)
or is made by an earlier graph of the same pass.
EAGER (not captured) when: T.hooks (WS-G checks), T.rec (the test record), plan.replay_rows (a deferred replay),
plan.deferred is not None, plan.lin is not rs.ops.linear, a lean control (dneg), no capable backend (DEAGER_OK lists
the reasons that a cell accepts: none).
HOST COUNTS that a replay does not run (CODE: TentStateFD._rows lean["layers"] += 1, r2_i4a_lean.py:100; ragged_engine.
_select rs.replays += 1, ragged_engine.py:370, with rs.ops.graph): the capture measures them per graph (the counter
dicts before and after the captured Python) and every replay adds that delta; expected_delta states the expected one
and a capture whose delta differs is listed (delta_bad: the cell's DGRAPH gate fails). The residency's count runs
eagerly (TentStateFD._resid_lean: calls["resid"]).
HELD TENSORS (alive for the arm, in the graph pool): per layer ti and inj (the residency's inputs at a replay), the pass's
hoisted tensors (layer 0), the last layer's output and the counter snapshot (snap keys). Every other tensor of a graph
dies after the capture of the graph that last reads it; the pool reuses its memory for later captures, which is safe
because the replays run in the capture order and nothing outside the pass reads it (the held tensors aside).
L61 (capture rounds): the state records the round of every draft capture and of every joint capture (T.round) for the
cell, which reports them and excludes them from the steady set.
NEGATIVE CONTROLS (tests only; dgneg): "stale_ids" (plan.ids copied into the static input once only), "no_counts" (the
skipped host counts not added), "no_key" (a pass replays the graphs of the key with the other check flag),
"live_skip" (B(l) reads the capture pass's own skip mask, not the static one).
HOST SYNCS: none in a replayed pass (the capture synchronizes at each begin and end).
"""
import time

import torch

DGNEG = (None, "stale_ids", "no_counts", "no_key", "live_skip")
DEAGER_OK = ()                        # the reasons for an eager one-row pass that the cells accept (none)
GROUPS = ("T", "lean", "rs", "vs", "n")


def expected_delta(graph_select: bool):
    """(A, B): the host counts per layer that a replay of A(l) / B(l) must add (module docstring, HOST COUNTS)."""
    a = dict(T={}, lean=dict(layers=1), rs=(dict(replays=1) if graph_select else {}), vs={}, n={})
    b = {g: {} for g in GROUPS}
    return a, b


def _counts(T):
    rs, vs = T.rs, T._vs
    return dict(T=dict(T.calls), lean=dict(T.lean), rs=dict(replays=int(rs.replays), captures=int(rs.captures)),
                vs=(dict(vs.calls) if vs is not None else {}), n=(dict(vs.stat_n) if vs is not None else {}))


def _delta(a, b):
    return {g: {k: b[g][k] - a[g].get(k, 0) for k in b[g] if b[g][k] != a[g].get(k, 0)} for g in GROUPS}


def make_backend(GR):
    """CudaGraphsD: the split-capture backend over GR.CudaGraphs (GR = r2_i4g_graph; the joint pass's capture() and the
    SAME pool). begin() / end() capture one graph on a side stream with torch.cuda.CUDAGraph.capture_begin /
    capture_end (the steps of torch.cuda.graph without its gc.collect and empty_cache); side(fn) runs fn on that stream
    (the warm pass). A factory, so that this file imports torch and time only."""

    class CudaGraphsD(GR.CudaGraphs):
        def __init__(self):
            super().__init__()
            self.dcapture_s = 0.0
            self.dcaptures = 0
            self._stream = None
            self._open = None

        def _side(self):
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            return self._stream

        def begin(self):
            if self._open is not None:
                raise RuntimeError("CudaGraphsD: a capture is open")
            torch.cuda.synchronize()
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            ctx = torch.cuda.stream(self._side())
            ctx.__enter__()
            t0 = time.perf_counter()
            try:
                g.capture_begin(pool=self.pool)
            except BaseException:
                ctx.__exit__(None, None, None)
                raise
            self._open = (g, ctx, t0)

        def end(self):
            g, ctx, t0 = self._open
            self._open = None
            try:
                g.capture_end()
            finally:
                ctx.__exit__(None, None, None)
            torch.cuda.synchronize()
            self.dcapture_s += time.perf_counter() - t0
            self.dcaptures += 1
            return g

        def abort(self):
            if self._open is not None:
                try:
                    self.end()
                except Exception:                                      # the arm raises anyway
                    pass

        def side(self, fn):
            torch.cuda.synchronize()
            with torch.cuda.stream(self._side()):
                out = fn()
            torch.cuda.synchronize()
            return out

        def release(self):
            super().release()
            self._stream = None

    return CudaGraphsD


class _Inline:
    """In place of rs.graphs[l] during A(l)'s capture: replay() runs ragged_engine._pool_topk on the tensors of
    ragged_engine.py:355-356 (module docstring, THE SELECTION GRAPH)."""

    def __init__(self, RE, rs, layer, twin):
        self.RE, self.rs, self.layer, self.twin = RE, rs, layer, twin

    def replay(self):
        rs, st = self.rs, self.rs.st
        self.RE._pool_topk(rs.nl, self.layer, rs.score_buf, self.twin.ccis, rs.n_used, rs.M_b, st.pq, st.pc, st.tv_q,
                           st.ti_q, st.mask, st.tv, st.ti)


_CLASSES = {}


def draft_classes(STF, base):
    """(TentStateFDGD, TentRunnerFDGD) built on the module STF and the graph pair base = (TentStateFDG,
    TentRunnerFDG)."""
    key = (id(STF), id(base[0]), id(base[1]))
    if key not in _CLASSES:
        _CLASSES[key] = (STF, base) + _make(STF, base)
    return _CLASSES[key][2:]


def _make(STF, base):
    SpecError = STF.SpecError
    ST = STF.ST
    RE = ST.RE
    Plan = ST._Plan
    SG, RG = base

    class TentStateFDGD(SG):
        """L4PTFDG's state whose one-row passes run from per-layer CUDA graphs (module docstring)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None, dgneg=None):
            if dgneg not in DGNEG:
                raise SpecError("unknown draft-graph control %r (one of %s)" % (dgneg, DGNEG))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg, gneg=gneg)
            self.dgneg = dgneg
            self._vs = None                                            # set by the runner (its VerifyState)
            self._dg = {}                                              # key -> the graph set of one-row passes
            self._dwarm = set()                                        # the keys whose warm pass ran
            self._dsb = None                                           # the static inputs
            self._dskip = None                                         # the static skip mask
            self._dcur = None                                          # the open capture (layer, phase, counts)
            self.dgraph = dict(captures=0, warm=0, capture_passes=0, replays=0, replay_passes=0, eager=0,
                               eager_why={}, delta_bad=[], capture_rounds=[], joint_capture_rounds=[], static_bytes=0)

        # ---------------------------------------------------------- the policy
        def _dwhy(self, plan):
            """None: the one-row pass may run from graphs; else the reason it runs eagerly."""
            if self.hooks is not None:
                return "hooks"
            if self.rec is not None:
                return "record"
            if plan.replay_rows:
                return "replay"
            if plan.deferred is not None:
                return "deferred"
            if plan.lin is not self.rs.ops.linear:
                return "lin"
            if self.dneg is not None:
                return "dneg"
            if self.backend is None or not hasattr(self.backend, "begin"):
                return "backend"
            return None

        def _dkey(self, plan):
            return (bool(self.checking), bool(plan.snap))

        def _dstatic(self, plan):
            """The static inputs (module docstring); returns the pass's plan on them."""
            B = self.B
            sb = self._dsb
            if sb is None:
                dev = plan.ids.device
                sb = self._dsb = dict(ids=torch.zeros(tuple(plan.ids.shape), dtype=plan.ids.dtype, device=dev),
                                      pos=torch.zeros(tuple(plan.pos.shape), dtype=plan.pos.dtype, device=dev),
                                      act=torch.zeros((B,), dtype=torch.bool, device=dev),
                                      ver=torch.zeros((B,), dtype=torch.bool, device=dev),
                                      drf=torch.zeros((B,), dtype=torch.bool, device=dev))
                self.dgraph["static_bytes"] = sum(t.numel() * t.element_size() for t in sb.values())
                sb["ids"].copy_(plan.ids)                              # (the control below copies the ids once only)
            elif self.dgneg != "stale_ids":                            # NEGATIVE CONTROL (tests only): ids not copied
                sb["ids"].copy_(plan.ids)
            sb["pos"].copy_(plan.pos)
            sb["act"].copy_(plan.act[0])
            sb["ver"].copy_(plan.verify[0])
            sb["drf"].copy_(plan.draft[0])
            return Plan(R=1, kind=plan.kind, ids=sb["ids"], pos=sb["pos"], act=sb["act"].view(1, B),
                        verify=sb["ver"].view(1, B), draft=sb["drf"].view(1, B), replay=None, replay_rows=(),
                        deferred=None, vs=None, lin=plan.lin, snap=plan.snap)

        def _dapply(self, d):
            if self.dgneg == "no_counts":                              # NEGATIVE CONTROL (tests only)
                return
            vs = self._vs
            for k, v in d["T"].items():
                self.calls[k] = self.calls.get(k, 0) + v
            for k, v in d["lean"].items():
                self.lean[k] = self.lean.get(k, 0) + v
            for k, v in d["rs"].items():
                setattr(self.rs, k, getattr(self.rs, k) + v)
            for k, v in d["vs"].items():
                vs.calls[k] = vs.calls.get(k, 0) + v
            for k, v in d["n"].items():
                vs.stat_n[k] = vs.stat_n.get(k, 0) + v

        # ---------------------------------------------------------- one pass
        @torch.inference_mode()
        def run_pass(self, model, plan):
            if plan.R != 1:                                            # the joint pass: L4PTFDG's
                return super().run_pass(model, plan)
            why = self._dwhy(plan)
            if why is not None:                                        # EAGER: L4PTFD's pass on the real tensors
                self.dgraph["eager"] += 1
                self.dgraph["eager_why"][why] = self.dgraph["eager_why"].get(why, 0) + 1
                return super().run_pass(model, plan)
            key = self._dkey(plan)
            if self.dgneg == "no_key" and key not in self._dg:        # NEGATIVE CONTROL: the graphs of the key with
                key = next((k for k in self._dg if k[1] == key[1]), key)    # the same snap, the other check flag
            sp = self._dstatic(plan)
            gs = self._dg.get(key)
            if gs is None and key not in self._dwarm:                 # WARM: eager on the static inputs, side stream
                self._dwarm.add(key)
                self.dgraph["warm"] += 1
                out = self.backend.side(lambda: super(TentStateFDGD, self).run_pass(model, sp))
                plan.ctr = sp.ctr
                return out
            rs, B = self.rs, self.B
            self.lean["passes"] += 1                                   # TentStateFD.run_pass's count (R = 1)
            sp.ctr = [None]
            if gs is None:                                             # CAPTURE (2 L graphs), each replayed once
                gs = self._dcapture(model, sp, key)
                self.dgraph["capture_passes"] += 1
                self.dgraph["capture_rounds"].append((int(self.round), list(key)))
            else:                                                      # REPLAY
                self._dreplay(gs)
                self.dgraph["replay_passes"] += 1
            if sp.snap:
                sp.ctr[0] = gs["ctr"]
            hidden = rs.nl.layer_norm(gs["out"], w=model.norm_weight, eps=model.norm_variance_epsilon)   # as
            logits = sp.lin(hidden, model.lm_head).float()             # spec_tentative.py:312-316
            rs.st.steps += 1
            self.calls["passes"] += 1
            plan.ctr = sp.ctr
            return logits.reshape(B, 1, -1)

        def _dcapture(self, model, sp, key):
            gs = dict(key=key, A=[], B=[], dA=[], dB=[], args=[], out=None, lp=None, ctr=None)
            self._lp = None                                            # TentStateFD.run_pass (r2_i4a_lean.py:93)
            E = self.B
            try:
                for l, layer in enumerate(model.layers):
                    self._dbegin_a(gs, l, layer)
                    if l == 0:                                         # spec_tentative.py:308-309, inside A(0)
                        hidden = ST.F.embedding(sp.ids.reshape(E, 1), model.embed_tokens)
                        pos_flat = sp.pos.reshape(E)
                    hidden = self._layer(l, layer, hidden, pos_flat, model.cos_sin_cache, sp)
                    self._dend_b(gs, l)
            except BaseException:
                self._dabort()
                raise
            gs["out"], gs["lp"], gs["ctr"] = hidden, self._lp, sp.ctr[0]
            self._dg[key] = gs
            return gs

        def _dbegin_a(self, gs, l, layer):
            rs = self.rs
            cur = self._dcur = dict(l=l, gs=gs, phase="A", c0=_counts(self), sel=None, swapped=False)
            if rs.ops.graph:                                           # THE SELECTION GRAPH: inline in A(l)
                cur["sel"], cur["swapped"] = rs.graphs[l], True
                rs.graphs[l] = _Inline(RE, rs, layer, rs.st.twins[l])
            self._resid_lean = self._dstub                             # the split point (module docstring)
            self.backend.begin()

        def _dstub(self, l, eng, twin, ti, inj, act, drf, qb3):
            """In place of TentStateFD._resid_lean during A(l)'s capture: end A(l), replay it, the real residency
            (eager), the static skip mask, begin B(l)."""
            cur = self._dcur
            if cur is None or cur["phase"] != "A" or cur["l"] != l:
                raise SpecError("draft graph: the residency of layer %d outside its capture" % l)
            gA = self.backend.end()
            del self._resid_lean
            if cur["swapped"]:
                self.rs.graphs[l] = cur["sel"]
                cur["swapped"] = False
            dA = _delta(cur["c0"], _counts(self))
            gA.replay()
            skip = self._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)
            if self._dskip is None:
                self._dskip = torch.zeros_like(skip)
            self._dskip.copy_(skip)
            gs = cur["gs"]
            gs["A"].append(gA)
            gs["dA"].append(dA)
            gs["args"].append((l, eng, twin, ti, inj, act, drf, qb3))
            cur["c1"], cur["phase"] = _counts(self), "B"
            self.backend.begin()
            return skip if self.dgneg == "live_skip" else self._dskip  # NEGATIVE CONTROL: the live mask

        def _dend_b(self, gs, l):
            cur = self._dcur
            if cur is None or cur["phase"] != "B" or cur["l"] != l:
                raise SpecError("draft graph: layer %d returned without its residency" % l)
            gB = self.backend.end()
            dB = _delta(cur["c1"], _counts(self))
            gB.replay()
            gs["B"].append(gB)
            gs["dB"].append(dB)
            wa, wb = expected_delta(bool(self.rs.ops.graph))
            if gs["dA"][-1] != wa or dB != wb:
                self.dgraph["delta_bad"].append((l, gs["dA"][-1], dB))
            self._dcur = None
            self.dgraph["captures"] += 2
            self.dgraph["replays"] += 2

        def _dabort(self):
            cur = self._dcur
            self._dcur = None
            if hasattr(self.backend, "abort"):
                self.backend.abort()
            self.__dict__.pop("_resid_lean", None)
            if cur is not None and cur.get("swapped"):
                self.rs.graphs[cur["l"]] = cur["sel"]

        def _dreplay(self, gs):
            self._lp = gs["lp"]
            sk = self._dskip
            for l in range(self.L):
                gs["A"][l].replay()
                self._dapply(gs["dA"][l])
                sk.copy_(self._resid_lean(*gs["args"][l]))             # EAGER: the host stamp
                gs["B"][l].replay()
                self._dapply(gs["dB"][l])
            self.dgraph["replays"] += 2 * self.L

        # ---------------------------------------------------------- L61: the joint capture's round
        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            n0 = self.graph["captures"]
            out = super()._rows_union(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            if self.graph["captures"] > n0 and int(self.round) not in self.dgraph["joint_capture_rounds"]:
                self.dgraph["joint_capture_rounds"].append(int(self.round))
            return out

        def release(self):
            """Drops every graph and the static inputs (the arm's end)."""
            super().release()
            self._dg.clear()
            self._dwarm.clear()
            self._dsb = self._dskip = self._dcur = None

    class TentRunnerFDGD(RG):
        """L4PTFDG (TentRunnerFDG) with the one-row passes replayed from per-layer CUDA graphs (arm L4PTFDGD). dgneg: a
        draft-graph control (tests only); graph_backend: the capture backend (None: make_backend(GR)(); the CPU tests:
        a stand-in with capture(), begin(), end(), side())."""

        def __init__(self, *args, dgneg=None, graph_backend=None, GR=None, **kw):
            if dgneg not in DGNEG:                                     # before the meter attaches
                raise SpecError("unknown draft-graph control %r (one of %s)" % (dgneg, DGNEG))
            if graph_backend is None:
                if GR is None:
                    raise SpecError("L4PTFDGD: pass GR (r2_i4g_graph) or a graph_backend")
                graph_backend = make_backend(GR)()
            self._dgneg = dgneg
            super().__init__(*args, graph_backend=graph_backend, **kw)
            self.T._vs = self.vs
            self.meta.update(draft_graph=dict(dgneg=dgneg, backend=type(self.T.backend).__name__,
                                              deager_ok=DEAGER_OK))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDGD(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                 gneg=self._gneg, dgneg=self._dgneg)

        def dgraph_report(self) -> dict:
            T = self.T
            b = T.backend
            return dict(T.dgraph, keys=sorted([list(k) for k in T._dg]), warm_keys=sorted([list(k) for k in T._dwarm]),
                        eager_why=dict(T.dgraph["eager_why"]), dcapture_s=round(getattr(b, "dcapture_s", 0.0), 3),
                        L=T.L)

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(dgraph=self.dgraph_report())
            return out

    return TentStateFDGD, TentRunnerFDGD
