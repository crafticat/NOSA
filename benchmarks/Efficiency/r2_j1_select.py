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
F2 (the second J1 commit; runner keyword j1draft, default True): the ONE-ROW (draft) passes' pooling. The I4H draft
graphs capture, per layer, A(l) with the selection inline (its _Inline object in place of the layer's shipped pooling
graph); the J1 state puts _InlineJ1 there instead (in _dbegin_a, after the capture begins; the I4H code restores the
shipped object after A(l)): the compiled pool_fn (E = B, one entry per request) and ragged_engine._pool_topk's other
ops on the same tensors, so the draft graphs replay the compiled kernel. The pooling compiles BEFORE the first draft
capture (one eager warm call on scratch outputs). Check graphs (the check rounds' key) also run the shipped kernel into
scratch and count differing values on the device (J1-DRAFT). Eager one-row passes (the I4H module's eager reasons,
its warm pass) keep the shipped selection graph: the same bits by J1-DRAFT's comparison.
ShPool (F2 on the baseline): the same compiled arithmetic behind the shipped nosa_pooling signature, for the cell's
matched SHJ1 run.
F3 (the third J1 commit; arm L4PTFDGUDCDJ3 = L4PTFDGUDCDJ1 + F3; the non-keep arm only: the keep arm's residency
reads its kept store, not checked for capture here): the RESIDENCY inside the captured graphs. StampFrame (the
strongest baseline's shifted-age LRU frame, reused) replaces the host stamp of pool_update, so the residency has no
host value; the joint section graph (the I4S _gsection, COPIED with the lines marked "# J3") captures the section
WITH its residency (no eager residency, no stub); the draft graphs capture A(l), the residency and B(l) as ONE graph
per layer (_dstub / _dend_b / _dreplay; the draft key adds the meter's `on`). EXPECTED: small. The passes are
device-bound (docs/NOTES_j1.md section 2c); F3 removes the residency's eager launches from the host path only.

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
import types

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


def _copy_fn(fn, suffix: str):
    """A copy of fn with its own code object (dynamo keeps its cache on the code object: the copy's compiles do not
    count against fn's cache)."""
    co = fn.__code__.replace(co_name=fn.__code__.co_name + suffix)
    return types.FunctionType(co, fn.__globals__, co.co_name, fn.__defaults__, fn.__closure__)


pool_sh_fn = _copy_fn(pool_fn, "_sh")                  # the matched baseline's copy (ShPool: dynamic shapes)


def loc_fn(v, bm, pm, fullv, big, arw, arp, topk: int, nb: int):
    """F6: the I4CC loc1 helper's outputs (valid, in_win, wslot, in_pool, pslot, keyh) WITHOUT the two sorts of the
    window / pool maps: an INVERSE TABLE per (head, request), block id -> slot, made by one scatter of the slot index
    (ids outside [0, nb) and -1 go to a dump column nb), then one gather at the row ids. v (H, B, R x T) int64 (ids <
    nb: the pooled width, the top-k's range); bm (H, B, T), pm (H, B, P) the maps; arw / arp (T,) / (P,) int64 aranges.
    Valid ids are unique in a map, so the slot equals the sorted search's slot; where a row id is not in the map, the
    slot value differs from the sorted search's but no consumer reads it there (loc2 reads wslot under in_win and pslot
    under in_pool only)."""
    valid = v >= 0
    at = v.clamp(0, nb - 1)
    bmi = bm.long()
    wi = torch.where((bmi >= 0) & (bmi < nb), bmi, nb)
    invw = torch.full(bm.shape[:-1] + (nb + 1,), -1, dtype=torch.int64, device=bm.device)
    invw = invw.scatter(-1, wi, arw.expand_as(wi))
    wslot = invw.gather(-1, at)
    in_win = valid & (wslot >= 0)
    if pm is not None:
        pmi = pm.long()
        pi = torch.where((pmi >= 0) & (pmi < nb), pmi, nb)
        invp = torch.full(pm.shape[:-1] + (nb + 1,), -1, dtype=torch.int64, device=pm.device)
        invp = invp.scatter(-1, pi, arp.expand_as(pi))
        ps = invp.gather(-1, at)
        in_pool = valid & (ps >= 0) & ~in_win
        pslot = ps + topk
    else:
        in_pool = torch.zeros_like(valid)
        pslot = torch.zeros_like(v)
    host = valid & fullv & ~in_win & ~in_pool
    keyh = torch.where(host, v, big)
    return valid, in_win, wslot, in_pool, pslot, keyh


FNS = dict(pool=pool_fn, loc=loc_fn)
_COMPILED = {}


class ShPool:
    """F2 on the SAME-JOB BASELINE (the matched comparison): a callable with the signature of the shipped
    max_pooling_fused.nosa_pooling (scalar max_seqlen; the pooled width M = the buffer width; no per-request validity),
    computed by pool_fn compiled with dynamic shapes (the shipped step re-warms its pooling graphs when the widths
    change: one compile serves them). The cell puts it in place of nosa_llama.nosa_pooling for ONE extra shipped run
    (SHJ1) and restores it. EAGER calls (the warm-up call before each capture of the after-pooling graph) are also
    compared with the shipped kernel bit for bit (peq; a call inside a capture is not compared)."""

    def __init__(self, shipped, backend: str = "inductor"):
        if backend not in CC.BACKENDS:
            raise ValueError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
        self.shipped = shipped
        self.f = torch.compile(pool_sh_fn, fullgraph=True, dynamic=True, backend=backend)
        self._t = {}
        self.calls = self.captured = self.peq_calls = 0
        self.peq_bad = self.peq_vals = 0
        self.compile_s = 0.0
        self.events = []

    def __call__(self, q_score, cis_score, max_seqlen, q_score_pooled, cis_score_pooled, block_size=32, stride=16,
                 padding=1, set_size=5, init_blocks=1, local_blocks=16):
        if int(padding) != PADDING or int(set_size) != SET_SIZE:
            raise ValueError("ShPool: padding %d / set_size %d only" % (PADDING, SET_SIZE))
        H, B = int(q_score.shape[0]), int(q_score.shape[1])
        M = int(q_score_pooled.shape[-1])
        dev = q_score.device
        key = (B, M, str(dev))
        t = self._t.get(key)
        if t is None:
            ar = torch.arange(B, dtype=torch.int64, device=dev).view(B, 1, 1)
            m = torch.arange(M, dtype=torch.int64, device=dev)
            nidx = (m.view(M, 1) * (SET_SIZE - 1) - PADDING
                    + torch.arange(SET_SIZE, dtype=torch.int64, device=dev).view(1, SET_SIZE)).contiguous()
            t = self._t[key] = dict(ar=ar, nidx=nidx, mar=m.view(1, M), n=torch.zeros((B,), dtype=torch.int32,
                                                                                       device=dev),
                                    mb=torch.full((B,), M, dtype=torch.int32, device=dev))
        cap = _capturing()
        t["n"].fill_(int(max_seqlen))                                   # the shipped scalar (frozen in a capture)
        from torch._dynamo.utils import counters as C
        g0 = int(C["stats"]["unique_graphs"])
        t0 = time.perf_counter()
        self.f(q_score, cis_score, t["ar"], t["ar"], t["nidx"], t["mar"], t["n"], t["mb"], q_score_pooled,
               cis_score_pooled, int(init_blocks), int(local_blocks) + 1)
        dt = time.perf_counter() - t0
        g1 = int(C["stats"]["unique_graphs"])
        self.calls += 1
        self.captured += int(cap)
        if g1 != g0:
            self.events.append(dict(graphs=g1 - g0, s=round(dt, 3), B=B, M=M, capture=cap))
            self.compile_s += dt
        if not cap:                                                     # PEQ on the eager calls
            rq, rc = torch.empty_like(q_score_pooled), torch.empty_like(cis_score_pooled)
            self.shipped(q_score, cis_score, max_seqlen, rq, rc, block_size=block_size, stride=stride,
                         padding=padding, set_size=set_size, init_blocks=init_blocks, local_blocks=local_blocks)
            self.peq_bad += int(((bits(rq) != bits(q_score_pooled)).sum() + (bits(rc) != bits(cis_score_pooled)).sum())
                                .item())
            self.peq_vals += rq.numel() + rc.numel()
            self.peq_calls += 1

    def report(self) -> dict:
        return dict(calls=self.calls, captured=self.captured, peq_calls=self.peq_calls, peq_bad=self.peq_bad,
                    peq_vals=self.peq_vals, compile_s=round(self.compile_s, 3), events=list(self.events))


def compiled(name: str, backend: str):
    """The compiled helper (one per (name, backend) per process, as the I4CC helpers)."""
    key = (name, backend)
    f = _COMPILED.get(key)
    if f is None:
        if backend not in CC.BACKENDS:
            raise ValueError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
        f = _COMPILED[key] = torch.compile(FNS[name], fullgraph=True, dynamic=False, backend=backend)
    return f


class _InlineJ1:
    """In place of the I4H module's _Inline during a draft graph A(l)'s capture (F2): replay() runs the J1 state's
    _j1_dpool (the compiled pooling + ragged_engine._pool_topk's other ops on the same tensors)."""

    def __init__(self, T, l, layer, twin):
        self.T, self.l, self.layer, self.twin = T, int(l), layer, twin

    def replay(self):
        self.T._j1_dpool(self.l, self.layer, self.twin)


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

        def _j1_setup(self, backend: str, j1neg, draft: bool = True, f6: bool = True):
            if backend not in CC.BACKENDS:
                raise SpecError("compile backend %r (one of %s)" % (backend, CC.BACKENDS))
            if j1neg not in J1NEG:
                raise SpecError("unknown J1 control %r (one of %s)" % (j1neg, J1NEG))
            self._j1_backend, self.j1neg, self._j1_draft = backend, j1neg, bool(draft)
            z = lambda: torch.zeros((), dtype=torch.int64, device=self.dev)    # noqa: E731
            self.j1 = dict(calls=0, fallback={}, events=[], compile_s=0.0, checks=0, peq_bad=z(), peq_vals=z(),
                           peq_layers=0, draft=bool(draft), dlayers=0, dwarm=0, dpeq_bad=z(), dpeq_vals=z(),
                           dpeq_layers=z())
            self._j1_t = None                                          # the constant index tensors (first call)
            self._j1_ref = None                                        # PEQ scratch (first check)
            self._j1_dt = None                                         # the draft's index tensors and scratch
            self._j1_f6 = bool(f6)                                     # F6: the inverse-table locate
            self._j1_lt = None                                         # F6's aranges (first call, eager)
            self.j1.update(f6=bool(f6), f6calls=0)

        # ---------------------------------------------------------- F6: the joint section's locate without sorts
        def _j1_loc(self, l, v, eng, fullv, vs):
            """The I4CC loc1 outputs by loc_fn (compiled; module docstring F6). Called inside the joint section (eager,
            warm or under the capture: the compiled call is then a kernel node of the CPU stand-in, as I4CC's)."""
            t = self._j1_lt
            if t is None:
                dev = v.device
                T = int(eng._block_map.shape[-1])
                P = int(eng._pool_map.shape[-1]) if self.rs.pool else 1
                t = self._j1_lt = dict(arw=torch.arange(T, dtype=torch.int64, device=dev),
                                       arp=torch.arange(P, dtype=torch.int64, device=dev), nb=int(vs.pq.shape[-1]))
            f = compiled("loc", self._j1_backend)
            args = (v, eng._block_map, eng._pool_map if self.rs.pool else None, fullv, vs.big, t["arw"], t["arp"],
                    int(vs.topk), t["nb"])
            from torch._dynamo.utils import counters as C
            g0 = int(C["stats"]["unique_graphs"])
            cap = _capturing() or bool(getattr(self, "_cc_cap", False)) or bool(getattr(self, "_cd_cap", False))
            t0 = time.perf_counter()
            wrap = getattr(self, "_cc_wrap", None)
            out = wrap("j1loc", f, args) if wrap is not None else f(*args)
            dt = time.perf_counter() - t0
            g1 = int(C["stats"]["unique_graphs"])
            self.j1["f6calls"] += 1
            if g1 != g0:
                self.j1["events"].append(dict(graphs=g1 - g0, s=round(dt, 3), layer=int(l), capture=cap, kind="f6",
                                              joint_pass=int(self.calls["joint_passes"])))
                self.j1["compile_s"] += dt
            return out

        # ---------------------------------------------------------- the I4CC _cc_locate, COPIED (F6: no map sorts)
        def _cc_locate(self, l, eng, sel, sel_c, is_q, is_n, hp, vs, fetch: bool):
            """The I4CC _cc_locate with the window / pool sorts and loc1 replaced by _j1_loc (lines marked "# J1")."""
            if not self._j1_f6:                                                                       # J1
                return super()._cc_locate(l, eng, sel, sel_c, is_q, is_n, hp, vs, fetch)             # J1
            rs = self.rs
            H, B, R, T = sel.shape
            st = vs.ops.stats
            v = sel_c.view(H, B, R * T)
            if vs.ops.check:                                           # CHECK-ONLY: the dropped test, recounted
                self.hoist_next += ((v == hp.qb1.view(1, B, 1)) & (v >= 0)).sum()
            valid, in_win, wslot, in_pool, pslot, keyh = self._j1_loc(l, v, eng, hp.fullv, vs)                 # J1
            s = torch.sort(keyh, dim=-1).values
            out = self._cc("loc2", l, s, v, valid, in_win, wslot, in_pool, pslot, sel, is_q, is_n, self._eb[l],
                           self._sb1, self.cb0_id, vs.store_b.view(1, B, 1, 1), vs.lst_buf, vs.host_x, vs.minus1,
                           vs.big, self._cap_t, int(vs.cap), int(VR.FORK_BIG), bool(fetch), bool(st))
            lst = vs.lst_buf[..., :vs.cap]
            if st:                                                     # accounting (_locate_h's, on the outputs)
                first, keep, in_lst = out[3:]
                full, res = hp.full, hp.res
                FORK_BIG = VR.FORK_BIG
                cap = vs.cap
                sx = vs.stat
                rolev = (full | res).t().reshape(1, B, R, 1).expand(H, B, R, T).reshape(H, B, R * T)
                n_host = keep.to(torch.int64).sum(-1)
                n_all = first.to(torch.int64).sum(-1)
                sp = torch.sort(torch.where(in_pool & rolev, v, vs.big), dim=-1).values
                fp = torch.ones_like(sp, dtype=torch.bool)
                fp[..., 1:] = sp[..., 1:] != sp[..., :-1]
                fp &= sp < FORK_BIG
                n_pool = fp.to(torch.int64).sum(-1)
                sx["extras"] += n_all.sum() + n_pool.sum()
                sx["extras_host"] += n_host.sum()
                sx["extras_pool"] += n_pool.sum()
                sx["extras_max"] = torch.maximum(sx["extras_max"], n_host.max())
                sx["unserved"] += (n_all - cap).clamp(min=0).sum()
                sx["fills_used"] += (n_host.sum() > 0).to(torch.int64)
                sx["pool_reads"] += (in_pool & rolev).sum()
                sx["store_reads"] += (in_lst & rolev).sum()
                su = torch.sort(torch.where(valid & rolev, v, vs.big), dim=-1).values
                fu = torch.ones_like(su, dtype=torch.bool)
                fu[..., 1:] = su[..., 1:] != su[..., :-1]
                fu &= su < FORK_BIG
                n_u = fu.to(torch.int64).sum(-1)
                sx["union_sum"] += n_u.sum()
                sx["union_max"] = torch.maximum(sx["union_max"], n_u.max())
                vs.stat_n["union"] += H * B
            return out[:3], lst

        def _j1_call(self, l, kind: str, args):
            """One call of the compiled pooling with its compile event (module docstring, COMPILE). Inside a draft
            capture of the CPU stand-in, the node maker of the capture (_cc_wrap, as the I4DC helpers) runs it."""
            f = compiled("pool", self._j1_backend)
            from torch._dynamo.utils import counters as C
            g0 = int(C["stats"]["unique_graphs"])
            cap = _capturing() or bool(getattr(self, "_cd_cap", False)) or bool(getattr(self, "_cc_cap", False))
            wrap = getattr(self, "_cc_wrap", None)
            t0 = time.perf_counter()
            if wrap is not None:
                wrap("j1pool", f, args)
            else:
                f(*args)
            dt = time.perf_counter() - t0
            g1 = int(C["stats"]["unique_graphs"])
            if g1 != g0:
                self.j1["events"].append(dict(graphs=g1 - g0, s=round(dt, 3), layer=int(l), capture=cap, kind=kind,
                                              joint_pass=int(self.calls["joint_passes"])))
                self.j1["compile_s"] += dt

        # ---------------------------------------------------------- F2: the one-row (draft) passes' pooling
        def _j1_dtensors(self):
            """The draft's constant tensors and the PEQ scratch (made OUTSIDE any capture: the first _dbegin_a)."""
            t = self._j1_dt
            if t is None:
                st = self.rs.st
                dev, B, M = st.pq.device, self.B, int(st.pq.shape[-1])
                ar = torch.arange(B, dtype=torch.int64, device=dev)
                req = (ar + 1) % B if self.j1neg == "wrong_req" else ar  # NEGATIVE CONTROL (tests only)
                m = torch.arange(M, dtype=torch.int64, device=dev)
                nidx = (m.view(M, 1) * (SET_SIZE - 1) - PADDING
                        + torch.arange(SET_SIZE, dtype=torch.int64, device=dev).view(1, SET_SIZE)).contiguous()
                t = self._j1_dt = dict(er=ar.view(B, 1, 1), br=req.view(B, 1, 1).contiguous(), nidx=nidx,
                                       mar=m.view(1, M), rq=torch.empty_like(st.pq), rc=torch.empty_like(st.pc))
            return t

        def _dbegin_a(self, gs, l, layer):
            """The draft graph A(l)'s start (the I4H module): before it (outside the capture) the draft pooling's
            compile on scratch outputs (first layer only); after it, the selection's inline replay object is OURS
            (the compiled pooling + the shipped top-k ops on the shipped tensors)."""
            rs = self.rs
            use = self._j1_draft and rs.ops.graph and self._j1_why_d() is None
            if use and self.j1["dwarm"] == 0 and rs.score_buf is not None:
                t = self._j1_dtensors()
                # the warm call writes the REAL output buffers (scratch intermediates: the selection of every pass
                # rewrites them before its top-k reads them): a scratch made here would be an inference tensor, whose
                # dispatch keys differ from st.pq's, and dynamo would compile again inside the capture (CPU run of
                # 3e5dd00: a "draft" compile event with capture True)
                self._j1_call(l, "draft_warm", (rs.score_buf, rs.st.twins[l].ccis, t["er"], t["br"], t["nidx"],
                                                t["mar"], rs.n_used, rs.M_b, rs.st.pq, rs.st.pc,
                                                int(layer.init_blocks), int(layer.window_blocks) + 1))
                self.j1["dwarm"] += 1
            super()._dbegin_a(gs, l, layer)
            cur = self._dcur
            if use and self.j1["dwarm"] > 0 and cur is not None and cur.get("swapped"):
                rs.graphs[l] = _InlineJ1(self, l, layer, rs.st.twins[l])
                self.j1["dlayers"] += 1

        def _j1_why_d(self):
            if self.tneg is not None or self.fneg is not None or getattr(self, "hneg", None) is not None \
                    or getattr(self, "dneg", None) is not None:
                return "control"
            return None

        def _j1_dpool(self, l, layer, twin):
            """_Inline's replay with the compiled pooling: ragged_engine._pool_topk's ops on its tensors (the shipped
            pooling replaced; check graphs also run the shipped pooling into scratch and count differing values)."""
            rs = self.rs
            st, nl = rs.st, rs.nl
            t = self._j1_dt
            if self.checking:
                RE.nosa_pooling_per_request(rs.score_buf, twin.ccis, rs.n_used, rs.M_b, t["rq"], t["rc"],
                                            stride=layer.pooling_stride, block_size=layer.block_size,
                                            local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
            self._j1_call(l, "draft", (rs.score_buf, twin.ccis, t["er"], t["br"], t["nidx"], t["mar"], rs.n_used,
                                       rs.M_b, st.pq, st.pc, int(layer.init_blocks), int(layer.window_blocks) + 1))
            if self.checking:
                self.j1["dpeq_bad"] += (bits(st.pq) != bits(t["rq"])).sum() + (bits(st.pc) != bits(t["rc"])).sum()
                self.j1["dpeq_vals"] += st.pq.numel() + st.pc.numel()
                self.j1["dpeq_layers"] += 1
            TR = RE.torch                                              # the module's binding (CPU: the stand-ins)
            TR.topk(st.pq, nl.QK_SELECT, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
            st.mask.zero_()
            st.mask.scatter_(2, st.ti_q, True)
            st.pc.masked_fill_(st.mask, float("inf"))
            TR.topk(st.pc, layer.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))

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
            """One compiled joint pooling call (module docstring, COMPILE)."""
            M = int(vs.pq.shape[-1])
            t = self._j1_tensors(vs, M)
            self._j1_call(l, "joint", (score, twin.ccis, t["er"], t["br"], t["nidx"], t["mar"], vs.n_used.view(vs.E),
                                       vs.M_b.view(vs.E), vs.pq, vs.pc, int(layer.init_blocks),
                                       int(layer.window_blocks) + 1))
            self.j1["calls"] += 1

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
                        draft=bool(j["draft"]), dlayers=int(j["dlayers"]), dwarm=int(j["dwarm"]),
                        dpeq_bad=int(j["dpeq_bad"].item()), dpeq_vals=int(j["dpeq_vals"].item()),
                        dpeq_layers=int(j["dpeq_layers"].item()), f6=bool(j.get("f6")), f6calls=int(j.get("f6calls", 0)),
                        backend=self._j1_backend, j1neg=self.j1neg)

    out = {}
    for arm, base in ARMS_J1.items():
        SB, RB = DCL[base]
        S = type("TentState_" + arm, (_J1Mix, SB), dict(__doc__="%s's state with the compiled joint pooling." % base))

        def _mk(RB=RB, S=S, arm=arm):
            class R_(RB):
                """The J1 arm: the I4DC runner whose state runs the compiled joint pooling (F1) and, with j1draft, the
                compiled draft pooling inside the draft graphs (F2). Keyword arguments: the I4DC runner's, j1neg
                (J1NEG, tests only), j1draft (default True)."""

                def __init__(self, *args, j1neg=None, j1draft=True, j1f6=True, **kw):
                    if j1neg not in J1NEG:
                        raise SpecError("unknown J1 control %r (one of %s)" % (j1neg, J1NEG))
                    self._j1neg, self._j1draft, self._j1f6 = j1neg, bool(j1draft), bool(j1f6)
                    super().__init__(*args, **kw)
                    self.meta.setdefault("compile", {}).update(arm=arm, j1neg=j1neg, j1draft=bool(j1draft),
                                                               j1f6=bool(j1f6))

                def _arm_name(self):
                    return arm

                def _make_state(self, rs, attn, tneg, poison):
                    T = super()._make_state(rs, attn, tneg, poison)
                    T.__class__ = S                                    # the I4DC state, re-classed (same attributes)
                    T._j1_setup(getattr(T, "_cc_backend", "inductor"), self._j1neg, self._j1draft, self._j1f6)
                    return T

                @torch.inference_mode()
                def read(self) -> dict:
                    out = super().read()
                    out.update(j1=self.T.j1_report())
                    return out
            R_.__name__ = R_.__qualname__ = "TentRunner_" + arm
            return R_
        out[arm] = (S, _mk())
    out.update(_make_j3(STF, out))
    return out


# ===================================================================================================================
# F3: the residency inside the captured graphs (the strongest baseline's shifted-age LRU frame)
# ===================================================================================================================
STAMP_K = 64 << 31          # the constant LRU stamp of the shifted age frame (nosi-r2-graphstep 0159e82 graph_step.py:59)
ARMS_J3 = dict(L4PTFDGUDCDJ3="L4PTFDGUDCDJ1")          # the non-keep arm only (module docstring of the F3 part)


class StampFrame:
    """In place of the engine's pool_update kernel (rs.ops.kernels["pool_update"], one arm's dict): the SHIFTED AGE
    FRAME of the strongest baseline's whole-step graph (graph_step.py:17-19 change 3, :146-154 shift_in / shift_out,
    :243-244 the per-step shift; gs1 bitwise vs the eager step). The shipped callers pass a host int stamp that grows
    by topk per call and per layer (the reason the I4G / I4H rungs left the residency eager). Here every call of layer
    l first moves every REAL age (>= 0) down by topk, then runs the shipped kernel with the CONSTANT stamp STAMP_K: the
    kernel only compares ages and writes stamp + m (or a negative empty mark), so every decision is the shipped one. The
    first call of a layer (eager: the warm step) moves the layer's ages into the frame (shift_in with the stamp the
    caller passed); export() (after the run) moves them back (shift_out with the last call's shipped stamp) and sets
    the host stamp the shipped callers would have reached: first + topk x calls (calls: a device counter, so graph
    replays count). Capturable: no host value enters the kernel."""

    def __init__(self, orig, engines, topk: int):
        self.orig, self.engines, self.topk = orig, list(engines), int(topk)
        self.idx = {e._pool_age.data_ptr(): l for l, e in enumerate(self.engines)}
        self.first = [None] * len(self.engines)
        dev = self.engines[0]._pool_age.device
        self.n = torch.zeros((len(self.engines),), dtype=torch.int64, device=dev)
        self.exports = []

    def __call__(self, block_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base, tail):
        l = self.idx[pool_age.data_ptr()]
        if self.first[l] is None:
            if _capturing():
                raise RuntimeError("StampFrame: the first call of layer %d inside a capture" % l)
            self.first[l] = int(stamp_base)
            pool_age.copy_(torch.where(pool_age >= 0, pool_age + (STAMP_K - self.first[l] + self.topk), pool_age))
        pool_age.sub_(pool_age.ge(0).long().mul_(self.topk))
        self.orig(block_map, load_mask, pool_map, pool_age, pool_target, pool_action, STAMP_K, tail)
        self.n[l] += 1

    def export(self) -> dict:
        """After a run (one host read): the ages back in the shipped frame and the host stamps the shipped callers
        would hold; the frame re-arms (a later call shifts in again)."""
        ns = [int(x) for x in self.n.tolist()]
        rec = dict(first=list(self.first), calls=ns, stamps_ok=True)
        for l, e in enumerate(self.engines):
            if self.first[l] is None:
                continue
            last = self.first[l] + self.topk * (ns[l] - 1)
            e._pool_age.copy_(torch.where(e._pool_age >= 0, e._pool_age - STAMP_K + last, e._pool_age))
            want = self.first[l] + self.topk * ns[l]
            rec["stamps_ok"] = rec["stamps_ok"] and int(e._pool_stamp_base) >= int(self.first[l])
            e._pool_stamp_base = want
        self.first = [None] * len(self.engines)
        self.n.zero_()
        self.exports.append(rec)
        return rec


def _make_j3(STF, j1):
    """{J3 arm: (state, runner)}: the J1 arm + F3 (module docstring of the F3 part)."""
    SpecError = STF.SpecError
    SK = CC.SK                                                         # the I4S stack module (its GR / DG modules)
    GR, DG = SK.GR, SK.DG
    _counts_s = SK._counts_s
    expected_delta_s = SK.expected_delta_s

    def expected_delta_j3(stats: bool, H: int, B: int, l: int) -> dict:
        """The joint section's per-layer host counts with the residency INSIDE the graph: the I4S expectation + the
        residency's own count (calls["resid"] 1)."""
        d = expected_delta_s(stats, H, B, l)
        d["T"] = dict(d["T"], resid=d["T"].get("resid", 0) + 1)
        return d

    def expected_draft_a(graph_select: bool) -> dict:
        """The draft layer's host counts up to the residency's end with the residency inside: I4H's A + resid 1."""
        wa, _ = DG.expected_delta(graph_select)
        out = {g: dict(v) for g, v in wa.items()}
        out["T"] = dict(out["T"], resid=out["T"].get("resid", 0) + 1)
        return out

    class _J3Mix:
        """F3 on a J1 state (a mixin before the J1 state class in the MRO; the runner re-classes the J1 state)."""

        def _j3_setup(self):
            self._j3_on = True
            self.j3 = dict(joint_captures=0, draft_layers=0, delta_fixed=0)

        # ---------------------------------------------------------- the joint section: I4S _gsection, COPIED (J3)
        def _gsection(self, l, eng, twin, sel, q4, k4, v4, pbs, plan, section, stub: str, copy_skip: bool):
            if not getattr(self, "_j3_on", False):                                                    # J3
                return super()._gsection(l, eng, twin, sel, q4, k4, v4, pbs, plan, section, stub, copy_skip)   # J3
            vs = plan.vs
            why = self._why(l, vs, plan)
            key = None if why is not None else self._key(plan)
            g = self._g.get(l)
            if why is None and g is not None and g["key"] != key and self.gneg != "no_key":
                why = "key"
            if why is not None:                                        # EAGER: the section on the real tensors
                self.graph["eager"] += 1
                self.graph["eager_why"][why] = self.graph["eager_why"].get(why, 0) + 1
                return section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            sq, sk, sv, spb, splan = self._static(q4, k4, v4, pbs, plan)
            if g is None and self._warm.get(l) != key:                 # WARM: eager on the static inputs
                self._warm[l] = key
                self.graph["warm"] += 1
                return section(l, eng, twin, sel, sq, sk, sv, spb, splan)
            if g is None:                                              # CAPTURE (executes nothing), then replay
                before = _counts_s(self, vs)
                try:
                    graph, out = self.backend.capture(
                        lambda: section(l, eng, twin, sel, sq, sk, sv, spb, splan), self, vs)
                finally:
                    self.__dict__.pop(stub, None)
                d = GR._delta(before, _counts_s(self, vs))
                want = expected_delta_j3(bool(vs.ops.stats), self.H, self.B, l)                       # J3
                if d != want:
                    self.graph["delta_bad"].append((l, d, want))
                self.graph["deltas"].append((l, d))
                g = self._g[l] = dict(key=key, graph=graph, out=out, delta=d)
                self.graph["captures"] += 1
                self.graph["capture_passes"].append(int(self.calls["joint_passes"]))
                graph.replay()                                         # the counts of this pass ran at the capture
                self.graph["replays"] += 1
                self.j3["joint_captures"] += 1                                                      # J3
                return out
            g["graph"].replay()
            self._apply(g["delta"], vs)
            self.graph["replays"] += 1
            return g["out"]

        # ---------------------------------------------------------- the draft graphs: one graph per layer
        def _dkey(self, plan):
            """I4H's key + the meter's `on` (the residency's host gathers count blocks while it is on; the drain's
            one-row passes with target rows)."""
            k = super()._dkey(plan)
            return k + (bool(self.meter.on),) if getattr(self, "_j3_on", False) else k

        def _dbegin_a(self, gs, l, layer):
            if getattr(self, "_j3_on", False) and self._dskip is None:        # the static skip mask: before a capture
                self._dskip = torch.zeros((self.H, self.B, self.topk), dtype=torch.bool, device=self.dev)
            super()._dbegin_a(gs, l, layer)

        def _dstub(self, l, eng, twin, ti, inj, act, drf, qb3):
            """I4H's split point, F3: the REAL residency inside the open capture (A(l) and B(l) become one graph); the
            skip mask into the static mask; the counts up to here are A's."""
            if not getattr(self, "_j3_on", False):
                return super()._dstub(l, eng, twin, ti, inj, act, drf, qb3)
            cur = self._dcur
            if cur is None or cur["phase"] != "A" or cur["l"] != l:
                raise SpecError("draft graph: the residency of layer %d outside its capture" % l)
            del self._resid_lean
            if cur["swapped"]:
                self.rs.graphs[l] = cur["sel"]
                cur["swapped"] = False
            skip = self._resid_lean(l, eng, twin, ti, inj, act, drf, qb3)      # captured (the stamp frame)
            self._dskip.copy_(skip)
            gs = cur["gs"]
            c1 = DG._counts(self)
            gs["A"].append(None)
            gs["dA"].append(DG._delta(cur["c0"], c1))
            gs["args"].append((l, eng, twin, ti, inj, act, drf, qb3))
            gs["f3"] = True
            cur["c1"], cur["phase"] = c1, "B"
            self.j3["draft_layers"] += 1
            return self._dskip

        def _dend_b(self, gs, l):
            """I4H's end of B(l) (= the whole layer under F3); its count check expects A without the residency: an F3
            layer whose A delta is A + resid 1 and whose B delta is B's is not a mismatch."""
            n0 = len(self.dgraph["delta_bad"])
            super()._dend_b(gs, l)
            if gs.get("f3") and len(self.dgraph["delta_bad"]) == n0 + 1:
                lb, dA, dB = self.dgraph["delta_bad"][-1]
                _, wb = DG.expected_delta(bool(self.rs.ops.graph))
                if lb == l and dA == expected_draft_a(bool(self.rs.ops.graph)) and dB == wb:
                    self.dgraph["delta_bad"].pop()
                    self.j3["delta_fixed"] += 1

        def _dreplay(self, gs):
            if not gs.get("f3"):
                return super()._dreplay(gs)
            self._lp = gs["lp"]
            for l in range(self.L):
                gs["B"][l].replay()                                    # the whole layer, the residency inside
                self._dapply(gs["dA"][l])
                self._dapply(gs["dB"][l])
            self.dgraph["replays"] += 2 * self.L                       # I4H's count: A(l) and B(l) per layer

        def j3_report(self) -> dict:
            return dict(self.j3, frame=list(getattr(self, "_j3_frame_exports", [])))

    out = {}
    for arm, base in ARMS_J3.items():
        S1, R1 = j1[base]
        S = type("TentState_" + arm, (_J3Mix, S1), dict(__doc__="%s's state with the residency captured (F3)." % base))

        def _mk(R1=R1, S=S, arm=arm):
            class R3(R1):
                """The J3 arm: the J1 arm (F1 + F2) whose residency runs in the shifted-age frame (StampFrame) and inside
                the captured joint sections and draft graphs (F3)."""

                def _arm_name(self):
                    return arm

                def _make_state(self, rs, attn, tneg, poison):
                    T = super()._make_state(rs, attn, tneg, poison)
                    T.__class__ = S                                    # the J1 state, re-classed (same attributes)
                    T._j3_setup()
                    kern = rs.ops.kernels
                    if rs.pool:
                        self._frame = StampFrame(kern["pool_update"], [lay.cache_engine for lay in rs.cache.layers],
                                                 int(rs.cache.layers[0].cache_engine.topk))
                        kern["pool_update"] = self._frame
                    else:
                        self._frame = None
                    T._j3_frame_exports = []
                    return T

                def run(self, N, rounds: int = 0, max_rounds: int = 0, warm: int = 1):
                    out = super().run(N, rounds=rounds, max_rounds=max_rounds, warm=warm)
                    if self._frame is not None:                        # the ages back in the shipped frame
                        self.T._j3_frame_exports.append(self._frame.export())
                    return out

                @torch.inference_mode()
                def read(self) -> dict:
                    out = super().read()
                    out.update(j3=self.T.j3_report())
                    return out
            R3.__name__ = R3.__qualname__ = "TentRunner_" + arm
            return R3
        out[arm] = (S, _mk())
    return out


def j3_gates(arm: str, rd, L: int) -> list:
    """J1-RESID (MUST) of a J3 arm: the frame was exported once with equal call counts in every layer (> 0) and the
    host stamps restored; every joint layer graph captured with the residency inside (L captures); every draft capture
    pass ran F3 (draft layers = capture passes x L, each F3 count delta accepted)."""
    j = (rd or {}).get("j3") or {}
    fr = j.get("frame") or []
    dg = (rd or {}).get("dgraph") or {}
    ok_f = len(fr) == 1 and len(set(fr[0].get("calls") or [])) == 1 and (fr[0].get("calls") or [0])[0] > 0 \
        and fr[0].get("stamps_ok") is True
    want_d = int(dg.get("capture_passes", 0)) * int(L)
    ok = ok_f and j.get("joint_captures") == int(L) and want_d > 0 and j.get("draft_layers") == want_d \
        and j.get("delta_fixed") == want_d
    return [("%s-J1-RESID" % arm, True, bool(ok),
             "frame exports %s; joint captures with the residency %s (L %s); draft F3 layers %s (capture passes x L = "
             "%s), count deltas accepted %s" % (fr, j.get("joint_captures"), L, j.get("draft_layers"), want_d,
                                                j.get("delta_fixed")))]


def j1_gates(arm: str, rd, L: int, check_rounds: bool = True) -> list:
    """The J1 MUST gates of a J1 arm (module docstring): [(name, must, passed, detail)].
      J1-PEQ      joint pooling: PEQ ran (layers > 0) and 0 differing values
      J1-COMPILE  at most one compile event per kind (joint, draft_warm), none inside a capture, no draft compile
                  outside its warm call, no fallback, calls = the joint layers of the run (joint passes x L)
      J1-DRAFT    (j1draft) every captured one-row layer runs the compiled pooling (dlayers = draft capture passes
                  x L, one warm call) and the draft PEQ ran (layers > 0) with 0 differing values"""
    j = (rd or {}).get("j1") or {}
    calls = ((rd or {}).get("tent") or {}).get("calls") or {}            # TentState.report()'s host counts
    jp = int(calls.get("joint_passes", 0))
    ev = j.get("events") or []
    kinds = [e.get("kind", "joint") for e in ev]
    peq_ok = j.get("peq_layers", 0) > 0 and j.get("peq_bad") == 0 and j.get("peq_vals", 0) > 0
    sections = ((rd or {}).get("compile") or {}).get("sections")         # the I4CC compiled sections (python calls)
    f6_ok = (not j.get("f6")) or (j.get("f6calls", 0) == sections and (sections or 0) > 0)
    comp_ok = (all(kinds.count(k) <= 1 for k in set(kinds)) and set(kinds) <= {"joint", "draft_warm", "f6"}
               and not any(e.get("capture") for e in ev) and not j.get("fallback")
               and j.get("calls") == jp * int(L) and jp > 0 and f6_ok)
    out = [("%s-J1-PEQ" % arm, True, bool(peq_ok),
            "joint pooling bits vs the shipped kernel: %s differing of %s values in %s checked layers" % (
                j.get("peq_bad"), j.get("peq_vals"), j.get("peq_layers"))),
           ("%s-J1-COMPILE" % arm, True, bool(comp_ok),
            "calls %s (joint passes %s x L %s); F6 %s calls %s (compiled sections %s); fallback %s; compile events %s; "
            "compile %s s; backend %s" % (j.get("calls"), jp, L, j.get("f6"), j.get("f6calls"), sections,
                                         j.get("fallback"), ev, j.get("compile_s"), j.get("backend")))]
    if j.get("draft"):
        dg = (rd or {}).get("dgraph") or {}
        want = int(dg.get("capture_passes", 0)) * int(L)
        d_ok = (want > 0 and j.get("dlayers") == want and j.get("dwarm") == 1 and j.get("dpeq_layers", 0) > 0
                and j.get("dpeq_bad") == 0 and j.get("dpeq_vals", 0) > 0)
        out.append(("%s-J1-DRAFT" % arm, True, bool(d_ok),
                    "draft layers with the compiled pooling %s (draft capture passes %s x L %s); warm calls %s; draft "
                    "pooling bits vs the shipped kernel: %s differing of %s values in %s checked layer replays" % (
                        j.get("dlayers"), dg.get("capture_passes"), L, j.get("dwarm"), j.get("dpeq_bad"),
                        j.get("dpeq_vals"), j.get("dpeq_layers"))))
    return out
