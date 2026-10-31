"""Rung J1 (AREA 1 of the RECON7 follow-up: joint execution, attention and memory reads; KS/analysis/plan/
DESIGN_JOINT_EFF_2026-10-31.md fix F1; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the BLOCK POOLING of the joint pass's
4B selection rows as ONE torch.compile'd gather + max, in place of NOSA's per-request pooling kernel. The arms:
  ARM L4PTFDGUDCDJ1   = L4PTFDGUDCD   (rung I4DC, the best arm at C 128) with the joint pooling compiled
  ARM L4PTFDGUDKWCDJ1 = L4PTFDGUDKWCD (rung I4DC, the best arm at C 73) with the joint pooling compiled
OPT-IN: benchmarks/Efficiency/r2_j1_cell.py; no nosi module and no older cell imports this file. NO NEW KERNEL by hand
and no kernel edit: Inductor generates the kernel from OUR Python (rung 1 of the skill reusing-before-building); the
shipped stage-1 call, both top-k calls and every other op of the selection stay as they are. This file imports torch,
time and the I4DC compile module only. File:line citations: docs/NOTES_j1.md.

WHY (MEASURED, the I3 joint path's Kineto tables, KS/cpu-logs/recon7_r2_s2.txt lines 21, 87, 615, 678): the shipped
pooling kernel launches ONE 128-thread program PER OUTPUT ELEMENT (grid (H, E, M) = (2, 864, 261) at B 216): 12.28 ms per
joint pass at C 128 B 216 and 18.10 ms at C 73 B 318 (384 / 566 us per layer), for about 7 MB of reads and writes per
layer. Its arithmetic is a MAX over at most 5 loads per output (no float arithmetic): any correct implementation has
the same bits.

WHAT CHANGES (one call per joint layer; TentState._layer's joint branch, COPIED here with the lines marked "# J1"):
  shipped  verify_rows._select_rows: stage 1; the compressed cis copied to the R rows of each request ((H, B, R, Lk));
           the per-request pooling kernel; top-k; mask zero / scatter / masked_fill; top-k
  J1       stage 1 (the same call); pool_fn compiled (reads the compressed cis by REQUEST index: no copy); the same
           top-k / mask / top-k ops (verify_rows' torch binding, so the CPU world's top-k stand-in applies)
The one-row (draft) passes keep the shipped selection (their captured graphs): fix F2 is a later rung.

pool_fn = the kernel's arithmetic (max_pooling_per_request: the per-request kernel; its torch twin pool_reference):
  for entry e, pooled block m < M (M = the buffer width):
    n_i = m (S - 1) - P + i, i = 0 .. S - 1 (S 5, P 1); a load is live when 0 <= n_i < n_used[e]
    acc = max over the live loads (-inf when none); +inf when m < init_blocks or M_b[e] - m <= window_blocks + 1;
    -inf when m >= M_b[e]
  q scores: score[h, e, n]; cis scores: ccis[h, e // R, n] (the shipped call copies ccis[h, b, :] to each of b's R rows
  and reads row e: the same values).
EXACTNESS (MUST, the cell): J1-PEQ: in every CHECK round (the probe rounds of the timing cells; every round of the
  miniature), every joint layer, the shipped path runs on the SAME score and ccis into scratch buffers and the two
  outputs must be equal BIT FOR BIT (integer views); the arm's EQF (committed tokens, fingerprints, acceptance record,
  state checksums = L4PTFD bit for bit, the I4 cells' gate). A NaN score or a signed zero could in principle make a
  max differ in sign or NaN handling: J1-PEQ on the live inputs is the gate of that assumption.
FALLBACK (counted, the gate needs none): the test record, WS-G hooks, a negative control of another lever (tneg, fneg,
  hneg, ops.neg) or a dtype the shipped path would treat otherwise: the shipped _select_rows.
COMPILE: fullgraph=True, dynamic=False, backend of the arm (inductor), one compile per process for one shape set; the
  call is EAGER (never inside a CUDA-graph capture: the joint selection runs before the captured section). Events are
  recorded (dynamo's unique_graphs before / after, seconds, capture flag); gate J1-COMPILE (r2_j1_cell).
NEGATIVE CONTROL (tests only; j1neg): "wrong_req" (the cis of request (b + 1) % B: PEQ and EQF must fail).
HOST SYNCS: none added (the PEQ counters stay on the device until read()).
"""
import time

import torch

import r2_i4dc_compile as CD

CC = CD.CC
J1NEG = (None, "wrong_req")
ARMS_J1 = dict(L4PTFDGUDCDJ1="L4PTFDGUDCD", L4PTFDGUDKWCDJ1="L4PTFDGUDKWCD")
SET_SIZE, PADDING = 5, 1                                   # max_pooling_per_request's SET_SIZE, PADDING (the launcher's)
bits = CC.bits


# ===================================================================================================================
# the compiled pooling (pure: tensors in, in-place writes to the two output buffers)
# ===================================================================================================================
def pool_fn(score, ccis, er, br, nidx, mar, n_used, M_b, pq, pc, init_blocks: int, local1: int):
    """The per-request pooling of E selection rows (module docstring). score (H, E, Nq); ccis (H, B, Nc); er (E, 1, 1)
    int64 = arange(E); br (E, 1, 1) int64 = the request of each entry; nidx (M, S) int64 = m (S - 1) - P + i; mar (1, M)
    int64 = arange(M); n_used, M_b (E,) int32; pq, pc (H, E, M) written in place; local1 = window_blocks + 1."""
    Nq, Nc = score.shape[-1], ccis.shape[-1]
    E = er.shape[0]
    live = (nidx >= 0) & (nidx < n_used.view(E, 1, 1))                 # (E, M, S)
    gq = score[:, er, nidx.clamp(0, Nq - 1)]                           # (H, E, M, S)
    gc = ccis[:, br, nidx.clamp(0, Nc - 1)]
    aq = torch.where(live, gq, float("-inf")).amax(-1)                 # (H, E, M)
    ac = torch.where(live, gc, float("-inf")).amax(-1)
    mb = M_b.view(E, 1)
    edge = (mar < init_blocks) | (mb - mar <= local1)                  # (E, M)
    valid = mar < mb
    pq.copy_(torch.where(valid, torch.where(edge, float("inf"), aq), float("-inf")))
    pc.copy_(torch.where(valid, torch.where(edge, float("inf"), ac), float("-inf")))


FNS = dict(pool=pool_fn)
_COMPILED = {}


def compiled(name: str, backend: str):
    """The compiled helper (one per (name, backend) per process, as the I4CC helpers)."""
    key = (name, backend)
    f = _COMPILED.get(key)
    if f is None:
        if backend not in CC.BACKENDS:
            raise ValueError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
        f = _COMPILED[key] = torch.compile(FNS[name], fullgraph=True, dynamic=False, backend=backend)
    return f


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:                                                  # evidence only
        return False


_CLASSES = {}


def j1_classes(STF) -> dict:
    """{arm: (state class, runner class)} for ARMS_J1, built on the spec_tentative_fork module STF and the I4DC classes."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF, _make(STF))
    return _CLASSES[key][1]


def _make(STF):
    SpecError = STF.SpecError
    VR = STF.VR
    RE = STF.ST.RE
    BLOCK = STF.ST.BLOCK
    DCL = CD.dc_classes(STF)

    class _J1Mix:
        """The compiled joint pooling on an I4DC state (a mixin before the I4DC state class in the MRO; the runner
        re-classes the state that the I4DC runner built)."""

        def _j1_setup(self, backend: str, j1neg):
            if backend not in CC.BACKENDS:
                raise SpecError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
            if j1neg not in J1NEG:
                raise SpecError("unknown J1 control %r (one of %s)" % (j1neg, J1NEG))
            self._j1_backend, self.j1neg = backend, j1neg
            z = lambda: torch.zeros((), dtype=torch.int64, device=self.dev)    # noqa: E731
            self.j1 = dict(calls=0, fallback={}, events=[], compile_s=0.0, checks=0, peq_bad=z(), peq_vals=z(),
                           peq_layers=0)
            self._j1_t = None                                          # the constant index tensors (first call)
            self._j1_ref = None                                        # PEQ scratch (first check)

        def _j1_why(self, vs):
            """None: the joint pooling runs compiled; else the reason for the shipped _select_rows."""
            ops = vs.ops
            if ops.record or self.rec is not None:
                return "record"
            if self.hooks is not None and hasattr(self.hooks, "on_select"):
                return "hooks"
            if self.tneg is not None or self.fneg is not None or getattr(self, "hneg", None) is not None \
                    or ops.neg is not None:
                return "control"
            if vs.pq.dtype != vs.pc.dtype:
                return "dtype"
            return None

        def _j1_tensors(self, vs, M: int):
            t = self._j1_t
            if t is None:
                dev = vs.pq.device
                E, R, B = vs.E, vs.R, vs.B
                ar = torch.arange(E, dtype=torch.int64, device=dev)
                req = ar // R
                if self.j1neg == "wrong_req":                          # NEGATIVE CONTROL (tests only)
                    req = (req + 1) % B
                m = torch.arange(M, dtype=torch.int64, device=dev)
                nidx = m.view(M, 1) * (SET_SIZE - 1) - PADDING + torch.arange(SET_SIZE, dtype=torch.int64,
                                                                              device=dev).view(1, SET_SIZE)
                t = self._j1_t = dict(er=ar.view(E, 1, 1), br=req.view(E, 1, 1).contiguous(), nidx=nidx.contiguous(),
                                      mar=m.view(1, M))
            return t

        def _j1_pool(self, l, score, twin, vs, layer):
            """One compiled pooling call with its compile event (module docstring, COMPILE)."""
            M = int(vs.pq.shape[-1])
            t = self._j1_tensors(vs, M)
            f = compiled("pool", self._j1_backend)
            from torch._dynamo.utils import counters as C
            g0 = int(C["stats"]["unique_graphs"])
            cap = _capturing()
            t0 = time.perf_counter()
            f(score, twin.ccis, t["er"], t["br"], t["nidx"], t["mar"], vs.n_used.view(vs.E), vs.M_b.view(vs.E), vs.pq,
              vs.pc, int(layer.init_blocks), int(layer.window_blocks) + 1)
            dt = time.perf_counter() - t0
            g1 = int(C["stats"]["unique_graphs"])
            self.j1["calls"] += 1
            if g1 != g0:
                self.j1["events"].append(dict(graphs=g1 - g0, s=round(dt, 3), layer=int(l), capture=cap,
                                              joint_pass=int(self.calls["joint_passes"])))
                self.j1["compile_s"] += dt

        def _j1_peq(self, score, twin, vs, layer):
            """CHECK-ONLY (J1-PEQ): the shipped copy + pooling on the same inputs into scratch; bit comparison."""
            H, B, R, E = vs.H, vs.B, vs.R, vs.E
            if self._j1_ref is None:
                self._j1_ref = (torch.empty_like(vs.pq), torch.empty_like(vs.pc))
            rq, rc = self._j1_ref
            vs.ccis_rows.copy_(twin.ccis.unsqueeze(2).expand(H, B, R, twin.lk_max))
            VR.nosa_pooling_per_request(score, vs.ccis_rows.view(H, E, twin.lk_max), vs.n_used.view(E),
                                        vs.M_b.view(E), rq, rc, stride=layer.pooling_stride,
                                        block_size=layer.block_size, local_blocks=layer.window_blocks,
                                        init_blocks=layer.init_blocks)
            return rq, rc

        def _j1_select(self, vs, layer, q, twin, l):
            """verify_rows._select_rows with the compiled pooling (module docstring, WHAT CHANGES). Returns the
            selections (H, B, R, topk) int64, a view of vs.ti."""
            why = self._j1_why(vs)
            if why is not None:
                self.j1["fallback"][why] = self.j1["fallback"].get(why, 0) + 1
                return VR._select_rows(vs, layer, q, twin)
            rs = vs.rs
            H, B, R, E = vs.H, vs.B, vs.R, vs.E
            ck = twin.comp_k_flat()
            score = VR.stage1_seqused(q, ck, ck, vs.cu_q, vs.cu_k, 1, twin.lk_max, seqused_k=vs.n_used.view(E),
                                      ext=rs.ops.kernels.get("stage1_ext"))
            ref = self._j1_peq(score, twin, vs, layer) if self.checking else None
            self._j1_pool(l, score, twin, vs, layer)
            if ref is not None:
                bad = (bits(vs.pq) != bits(ref[0])).sum() + (bits(vs.pc) != bits(ref[1])).sum()
                self.j1["peq_bad"] += bad
                self.j1["peq_vals"] += vs.pq.numel() + vs.pc.numel()
                self.j1["peq_layers"] += 1
            TV = VR.torch                                              # the module's binding (CPU: the top-k stand-in)
            TV.topk(vs.pq, rs.nl.QK_SELECT, dim=-1, sorted=False, out=(vs.tv_q, vs.ti_q))
            vs.mask.zero_()
            vs.mask.scatter_(2, vs.ti_q, True)
            vs.pc.masked_fill_(vs.mask, float("inf"))
            TV.topk(vs.pc, layer.topk_blocks, dim=-1, sorted=False, out=(vs.tv, vs.ti))
            return vs.ti.view(H, B, R, layer.topk_blocks)

        # ---------------------------------------------------------- spec_tentative.TentState._layer, COPIED (J1)
        def _layer(self, l, layer, hidden, pos_flat, cos_sin_cache, plan):
            if plan.R == 1:                                                                            # J1
                return super()._layer(l, layer, hidden, pos_flat, cos_sin_cache, plan)                # J1
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
            pbs = []
            for r in range(R):                                             # 2. the twin (and the deferred replay)
                if r in plan.replay_rows:
                    self._replay(l, twin, plan.replay[r], plan.deferred)
                pb = twin.pos.clone()
                if R == 1:
                    rs.pos_before.copy_(pb)
                twin.step_unchecked(k4[:, r].contiguous(), c3[:, r].contiguous(), plan.act[r])
                if R == 1:
                    RE._lengths(rs, twin)
                else:
                    plan.vs.n_used[:, r].copy_(twin.n_comp)
                    plan.vs.M_b[:, r].copy_(pb // BLOCK + 1)
                if l == 0 and plan.snap:
                    plan.ctr[r] = torch.stack((twin.pos, twin.n_comp, twin.n_ccis))
                pbs.append(pb)
            if R == 1:                                                     # 3. the selection
                ck = twin.comp_k_flat()
                score = RE.stage1_seqused(q, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=rs.n_used,
                                          ext=rs.ops.kernels.get("stage1_ext"))
                RE._select(rs, l, layer, score, twin)
                sel = st.ti.view(self.H, B, 1, layer.topk_blocks)
            else:
                sel = self._j1_select(plan.vs, layer, q, twin, l)                                     # J1
            q4 = q.view(B, R, q.shape[1], q.shape[2])
            attn = self._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan).reshape(E, 1, -1)
            hidden = residual + lin(attn, layer.wo)
            residual = hidden
            h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
            h = lin(h, layer.gate_up_proj)
            out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
            NL.silu_and_mul(h, out)
            return residual + lin(out, layer.down_proj)

        def j1_report(self) -> dict:
            """ONE host read (after a run): the J1 record (calls, fallback, compile events, PEQ counters)."""
            j = self.j1
            return dict(calls=int(j["calls"]), fallback=dict(j["fallback"]), events=list(j["events"]),
                        compile_s=round(float(j["compile_s"]), 3), peq_bad=int(j["peq_bad"].item()),
                        peq_vals=int(j["peq_vals"].item()), peq_layers=int(j["peq_layers"]),
                        backend=self._j1_backend, j1neg=self.j1neg)

    out = {}
    for arm, base in ARMS_J1.items():
        SB, RB = DCL[base]
        S = type("TentState_" + arm, (_J1Mix, SB), dict(__doc__="%s's state with the compiled joint pooling." % base))

        def _mk(RB=RB, S=S, arm=arm):
            class R_(RB):
                """The J1 arm: the I4DC runner whose state runs the compiled joint pooling. Keyword arguments: the
                I4DC runner's, j1neg (J1NEG, tests only)."""

                def __init__(self, *args, j1neg=None, **kw):
                    if j1neg not in J1NEG:
                        raise SpecError("unknown J1 control %r (one of %s)" % (j1neg, J1NEG))
                    self._j1neg = j1neg
                    super().__init__(*args, **kw)
                    self.meta.setdefault("compile", {}).update(arm=arm, j1neg=j1neg)

                def _arm_name(self):
                    return arm

                def _make_state(self, rs, attn, tneg, poison):
                    T = super()._make_state(rs, attn, tneg, poison)
                    T.__class__ = S                                    # the I4DC state, re-classed (same attributes)
                    T._j1_setup(getattr(T, "_cc_backend", "inductor"), self._j1neg)
                    return T

                @torch.inference_mode()
                def read(self) -> dict:
                    out = super().read()
                    out.update(j1=self.T.j1_report())
                    return out
            R_.__name__ = R_.__qualname__ = "TentRunner_" + arm
            return R_
        out[arm] = (S, _mk())
    return out


def j1_gates(arm: str, rd, L: int, check_rounds: bool = True) -> list:
    """The J1 MUST gates of a J1 arm (module docstring): [(name, must, passed, detail)].
      J1-PEQ      PEQ ran (layers > 0) and 0 differing values
      J1-COMPILE  the pooling helper compiled at most once, never inside a capture, no fallback, calls = the joint layers
                  of the run (joint passes x L)"""
    j = (rd or {}).get("j1") or {}
    calls = ((rd or {}).get("tent") or {}).get("calls") or {}            # TentState.report()'s host counts
    jp = int(calls.get("joint_passes", 0))
    ev = j.get("events") or []
    peq_ok = j.get("peq_layers", 0) > 0 and j.get("peq_bad") == 0 and j.get("peq_vals", 0) > 0
    comp_ok = (len(ev) <= 1 and not any(e.get("capture") for e in ev) and not j.get("fallback")
               and j.get("calls") == jp * int(L) and jp > 0)
    return [("%s-J1-PEQ" % arm, True, bool(peq_ok),
             "pooling bits vs the shipped kernel: %s differing of %s values in %s checked layers" % (
                 j.get("peq_bad"), j.get("peq_vals"), j.get("peq_layers"))),
            ("%s-J1-COMPILE" % arm, True, bool(comp_ok),
             "calls %s (joint passes %s x L %s); fallback %s; compile events %s; compile %s s; backend %s" % (
                 j.get("calls"), jp, L, j.get("fallback"), ev, j.get("compile_s"), j.get("backend")))]
