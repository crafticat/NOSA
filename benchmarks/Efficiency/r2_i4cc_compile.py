"""Rung I4CC of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the OFF-THE-SHELF
COMPILER trial (LEDGER_rev6.md R6.6 E4) on the best stacked arm. Torch.compile (Inductor, mode "default") fuses the
small integer / mask / index / bit-copy ops of the joint pass's captured section; our own per-layer CUDA graphs then
capture the compiled kernels:
  ARM L4PTFDGUDC   = L4PTFDGUD   (r2_i4s_stack.py) with its hoisted joint section (STK_S below) through the helpers
  ARM L4PTFDGUDKWC = L4PTFDGUDKW (r2_i4s_stack.py) with its combined keep section (STK_K below) through the helpers
OPT-IN: benchmarks/Efficiency/r2_i4cc_cell.py; no nosi module and no older cell imports this file. NO NEW KERNEL by hand
and no kernel edit: Inductor generates the fused kernels from OUR Python (the rung-1 tool of the skill
reusing-before-building); the shipped kernels, the fork and the sorts stay as they are. This file imports torch, time and
r2_i4s_stack only.

ABBREVIATIONS. STK = benchmarks/Efficiency/r2_i4s_stack.py; STK_S = the hoisted joint section of L4PTFDGUD (the I4U hoist
module's TentStateFDU._rows_union, reached as STK's TentStateFDGU._usection); STK_K = STK's TentStateFKU._section (the
same section with the keep's store fill). The I4C traces: /mnt/beegfs/ojerbi/r2-runs/spec/i4c_c3a_2195575_i4ca and
i4c_c4b_2195576_i4cb; their fusion MODEL: KS/cpu-logs/i4cc_ana_s2.txt (KS/tmp_sc/i4cc_ana/s2_groups.py).

WHAT IS COMPILED (module-level pure functions: tensors in, tensors out or in-place writes to tensors passed in; a layer
index is never a Python constant of a graph, so 32 layers give ONE compile per helper; fullgraph=True, dynamic=False):
  pre    the head's block-qb test, the tail-id injection and ti (the residency's inputs), sel_c, is_q, is_n
  write  the buffer write of the R rows (layers >= 1; STK_S's _write_h, rows in order) on INTEGER VIEWS of the bf16
         buffers: a where / index / index_put is a bit copy, so the view gives the same bits by construction (no NaN
         canonicalisation through fp32)
  loc1   the window and pool searches (searchsorted, gather, the masks) and the host-only key
  loc2   the host list (first, rank = cumsum, keep, fill + scatter into vs.lst_buf), the list search, present / phys
         (L4PTFDGUDC: the final present / phys, the host ids into vs.host_x and the sort key of the row lists;
         L4PTFDGUDKWC: present and the keep's inputs phys / in_lst)
  kphys  L4PTFDGUDKWC only: the final phys from the keep's slots and the sort key of the row lists
  lists  the row lists: the gather by the sort order, the counts, the int32 copies into vs.pkey / pslot / npg
WHAT STAYS EAGER IN THE SECTION (captured by our graphs as before): the residency (or its stub), layer 0's buffer write
  (tpos and the counters), the chain images (their tensors are views of ONE arena storage: a compiled function with two
  aliased inputs of which one is written makes AOTAutograd build a synthetic base of the whole arena, so they stay
  eager), the four sorts (window map, pool map, host key, row-list key: Inductor 2.6 lowers aten.sort and may order ties
  differently; eager keeps the bits by construction), the shipped host gathers and the keep's fill (STK_K), the fork call,
  the counters, and the check-only / accounting code (the same code as STK_S / STK_K on the compiled outputs).
EXACTNESS (the EQF MUST: tokens, fingerprints, acceptance record, state checksums = L4PTFD bit for bit). Every compiled op
  is an integer, boolean or bit-copy op: comparisons, where, gather / index / index_put / scatter, cumsum of int64,
  searchsorted (Inductor's bucketize: the same left insertion point), integer adds, int64 -> int32 copies of values
  below 2^31. No float arithmetic is compiled (the draft's log-sum-exp merge and every GEMM stay as they are). The one
  scatter with duplicate targets writes the same value (FORK_BIG) at every duplicate (column cap of vs.lst_buf).
COMPOSITION WITH OUR CAPTURE (I4G's per-layer graphs): the section's first call per layer is I4G's WARM call (eager, on
  the static inputs); it compiles every helper (dynamo traces, Inductor generates and loads the kernels, the autotuner of a
  reduction runs) OUTSIDE any capture. The second call CAPTURES: the compiled wrapper launches its kernels into the
  stream capture (Inductor's wrapper takes the current stream; its temporaries come from the graph pool). Later passes
  replay. An eager probe pass calls the same compiled helpers on the real tensors (contiguous inputs: the same guards).
  COMPILE EVENTS are recorded per call (dynamo's unique_graphs counter before / after, host seconds, the joint pass, the
  capture flag): the cell's <arm>-COMPILE MUST gate needs every event before the first capture. Compile time is one-time
  setup (round 0, outside the steady set like the captures, L61) and is reported apart.
FALLBACK: a section with the test record, WS-G hooks, a lever's negative control (tneg, fneg, hneg, ops.neg) or
  unequal dtypes runs STK's eager section (counted in cc["fallback"]; the COMPILE gate needs none).
NEGATIVE CONTROLS (tests only; ccneg): "lost_lists" (the lists helper writes into a copy of vs.pslot: the fork reads
  stale arena slots; EQF must fail), "late" (inside a capture the pre helper runs behind another code object, the same
  ops: dynamo's first compile of it happens there; COMPILE must fail).
HOST SYNCS: none added (the compile events read a host counter only).
GRAPH BREAKS: none (fullgraph=True raises at one; every helper and variant compiled on the CPU backend in
  tests_cpu/test_i4cc.py). The Python was shaped for it before the first compile, not after a break: module-level
  functions (no state object, no counter dict, no host count inside), the layer's tensors passed in (no layer index),
  the sorts and the shipped kernel wrappers outside, the check-only and accounting branches outside (eager on the
  outputs), constants as Python ints / bools (cap, FORK_BIG, topk, fetch, stats: one specialisation each).
"""
import time

import torch

import r2_i4s_stack as SK

CCNEG = (None, "lost_lists", "late")
BACKENDS = ("inductor", "aot_eager", "eager")
HELPERS = ("pre", "write", "loc1", "loc2", "kphys", "lists")
ARMS_C = dict(L4PTFDGUDC="L4PTFDGUD", L4PTFDGUDKWC="L4PTFDGUDKW")
_INTV = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(t):
    """The integer view of a float tensor (the same bytes); an integer / bool tensor itself."""
    if t.dtype.is_floating_point:
        return t.view(_INTV[t.element_size()])
    return t


# ===================================================================================================================
# the compiled helpers (pure: no attribute of a state, no host read)
# ===================================================================================================================
def pre_fn(sel, qb3, qb4, qb1_4, minus1):
    """STK_S's hasq, s0, inj, ti (the residency's inputs) and sel_c, is_q, is_n."""
    hasq = (sel == qb4).any(-1)                                        # (H, B, R): the head selects block qb
    s0 = sel[:, :, 0]
    inj = (s0 > qb3) & ~hasq[:, :, 0:1]
    ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
    sel_c = torch.where(sel >= qb4, minus1, sel)
    is_q = sel == qb4
    is_n = sel == qb1_4
    return ti, inj, sel_c, is_q, is_n


def write_fn(kb, vb, bb, cis, bi, i_all, prow_all, act_all, k4, v4):
    """STK_S's _write_h for the R rows in row order, on integer views (module docstring): kb, vb (B, 64, H, D), bb
    (B, 64, H), cis (B, n, H), k4 / v4 (B, R, H, D); i_all / prow_all / act_all (R, B). In place."""
    B = kb.shape[0]
    for r in range(i_all.shape[0]):
        i = i_all[r]
        act = act_all[r]
        a3, a2 = act.view(B, 1, 1), act.view(B, 1)
        kb[bi, i] = torch.where(a3, k4[:, r], kb[bi, i])
        vb[bi, i] = torch.where(a3, v4[:, r], vb[bi, i])
        bb[bi, i] = torch.where(a2, cis[bi, prow_all[r]], bb[bi, i])


def loc1_fn(v, ws, wi, ps, pi, fullv, big, topk: int):
    """STK_S's _locate_h up to the host key: the window / pool searches (ws, wi / ps, pi: the eager sorts' outputs; ps
    None without the pool). Returns valid, in_win, wslot, in_pool, pslot, keyh (all (H, B, R x T))."""
    valid = v >= 0
    at_w = torch.searchsorted(ws, v).clamp(max=ws.shape[-1] - 1)
    in_win = (ws.gather(-1, at_w) == v) & valid
    wslot = wi.gather(-1, at_w)
    if ps is not None:
        at_p = torch.searchsorted(ps, v).clamp(max=ps.shape[-1] - 1)
        in_pool = (ps.gather(-1, at_p) == v) & valid & ~in_win
        pslot = pi.gather(-1, at_p) + topk
    else:
        in_pool = torch.zeros_like(valid)
        pslot = torch.zeros_like(v)
    host = valid & fullv & ~in_win & ~in_pool
    keyh = torch.where(host, v, big)
    return valid, in_win, wslot, in_pool, pslot, keyh


def loc2_fn(s, v, valid, in_win, wslot, in_pool, pslot, sel, is_q, is_n, eb, sb1, cb0_id, sb4, buf, host_x, minus1,
            big, cap_t, cap: int, fbig: int, fetch: bool, stats: bool):
    """STK_S's _locate_h from the sorted host key s on (the list into buf = vs.lst_buf, in place), then the section's
    present / phys. fetch (L4PTFDGUDC): returns (present, phys, the row lists' sort key) and writes vs.host_x[..., 1:];
    else (L4PTFDGUDKWC): (present, phys before is_q / is_n, in_lst) for the keep's fill. stats: also first, keep, in_lst
    (H, B, R x T) for the accounting."""
    first = torch.ones_like(s, dtype=torch.bool)
    first[..., 1:] = s[..., 1:] != s[..., :-1]
    first &= s < fbig
    rank = first.to(torch.int64).cumsum(-1) - 1
    keep = first & (rank < cap)
    buf.fill_(fbig)
    buf.scatter_(-1, torch.where(keep, rank, cap_t), torch.where(keep, s, big))
    lst = buf[..., :cap]
    at_l = torch.searchsorted(lst.contiguous(), v).clamp(max=cap - 1)
    in_lst = (lst.gather(-1, at_l) == v) & valid & ~in_win & ~in_pool
    present = in_win | in_pool | in_lst
    phys = torch.where(in_win, eb + wslot, torch.where(in_pool, eb + pslot, sb1 + at_l))
    phys = torch.where(present, phys, minus1)
    shp = is_q.shape
    present_f = present.view(shp) | is_q | is_n
    if fetch:
        phys_f = torch.where(is_q, cb0_id, torch.where(is_n, sb4, phys.view(shp)))
        host_x[..., 1:].copy_(torch.where(lst < fbig, lst, minus1))
        out = (present_f, phys_f, torch.where(present_f, sel, big))
    else:
        out = (present_f, phys.view(shp), in_lst.view(shp))
    if stats:
        out = out + (first, keep, in_lst)
    return out


def kphys_fn(is_q, is_n, cb0_id, sb4, lphys, present_f, sel, big):
    """STK_K's final phys from the keep's slots and the row lists' sort key."""
    phys_f = torch.where(is_q, cb0_id, torch.where(is_n, sb4, lphys))
    return phys_f, torch.where(present_f, sel, big)


def lists_fn(keys, order, phys_f, present_f, pkey, pslot, npg):
    """STK_S's _lists_h after the sort (keys, order = sort of the key, eager): in place into pkey / pslot / npg."""
    H, B, R, _ = keys.shape
    E = B * R
    slots = phys_f.gather(-1, order)
    n = present_f.sum(-1)
    pkey.copy_(keys.permute(1, 2, 0, 3).reshape(E, H, -1))
    pslot.copy_(slots.permute(1, 2, 0, 3).reshape(E, H, -1))
    npg.copy_(n.permute(1, 2, 0).reshape(E, H))


def _pre_late(sel, qb3, qb4, qb1_4, minus1):
    """The control "late" only: pre_fn behind another code object (dynamo compiles it at its first call)."""
    return pre_fn(sel, qb3, qb4, qb1_4, minus1)


FNS = dict(pre=pre_fn, write=write_fn, loc1=loc1_fn, loc2=loc2_fn, kphys=kphys_fn, lists=lists_fn, pre_late=_pre_late)
_COMPILED = {}


def compiled(name: str, backend: str):
    """The compiled helper (one per (name, backend) per process: dynamo's cache is per code object, so a later arm
    reuses the graphs of an earlier one with the same shapes)."""
    key = (name, backend)
    f = _COMPILED.get(key)
    if f is None:
        if backend not in BACKENDS:
            raise ValueError("compile backend %r (one of %s)" % (backend, BACKENDS))
        f = _COMPILED[key] = torch.compile(FNS[name], fullgraph=True, dynamic=False, backend=backend)
    return f


def configure() -> None:
    """Process-wide: a recompile past dynamo's cache limit RAISES (the default falls back to eager without an error,
    which every equality gate would pass)."""
    import torch._dynamo
    torch._dynamo.config.fail_on_cache_limit_hit = True


def dynamo_counts() -> dict:
    """Dynamo's process-wide counters (cumulative): compiled graphs, frames, graph breaks; Inductor's generated kernels."""
    from torch._dynamo.utils import counters as C
    out = dict(unique_graphs=int(C["stats"]["unique_graphs"]), frames_ok=int(C["frames"]["ok"]),
               graph_breaks=int(sum(C["graph_break"].values())))
    try:
        from torch._inductor import metrics
        out["inductor_kernels"] = int(metrics.generated_kernel_count)
    except Exception:                                                  # evidence only
        out["inductor_kernels"] = None
    return out


_CLASSES = {}


def compile_classes(STF) -> dict:
    """{arm: (state class, runner class)} for ARMS_C, built on the spec_tentative_fork module STF and STK's classes."""
    key = id(STF)
    if key not in _CLASSES:
        _CLASSES[key] = (STF, _make(STF))
    return _CLASSES[key][1]


def _make(STF):
    SpecError = STF.SpecError
    VR = STF.VR
    CLS = SK.stack_classes(STF)
    SG, RG = CLS["L4PTFDGUD"]
    SGK, RGK = CLS["L4PTFDGUDKW"]

    class _CCMix:
        """The compiled helpers' plumbing on a stacked state (a mixin before STK's state class in the MRO)."""

        def _cc_setup(self, backend: str, ccneg):
            if backend not in BACKENDS:
                raise SpecError("compile backend %r (one of %s)" % (backend, BACKENDS))
            if ccneg not in CCNEG:
                raise SpecError("unknown compile control %r (one of %s)" % (ccneg, CCNEG))
            self._cc_backend, self.ccneg = backend, ccneg
            self._cc_cap = False                                       # set by the runner's capture wrapper
            self._cc_wrap = None                                       # tests: the CPU capture stand-in's node maker
            self.cc = dict(calls={}, events=[], fallback={}, sections=0, first_capture_pass=None, compile_s=0.0)
            self._cc_c0 = dynamo_counts()
            sb = self.k[0], self.v[0], self.bias[0]
            tc = self.rs.st.twins[0].total_cis
            self._cc_ok = (sb[0].dtype == sb[1].dtype and sb[0].element_size() in _INTV
                           and sb[2].dtype == tc.dtype and sb[2].element_size() in _INTV)

        def _cc_why(self, plan):
            """None: the section runs through the compiled helpers; else the reason for STK's eager section."""
            ops = plan.vs.ops
            if ops.record or self.rec is not None:
                return "record"
            if self.hooks is not None and hasattr(self.hooks, "on_select"):
                return "hooks"
            if self.tneg is not None or self.fneg is not None or self.hneg is not None or ops.neg is not None:
                return "control"
            if not self._cc_ok:
                return "dtype"
            return None

        def _cc(self, name, l, *args):
            """One call of the compiled helper `name` (module docstring: COMPILE EVENTS)."""
            f = compiled(name, self._cc_backend)
            if self.ccneg == "late" and self._cc_cap and name == "pre":   # NEGATIVE CONTROL: a first compile in a capture
                f = compiled("pre_late", self._cc_backend)
            from torch._dynamo.utils import counters as C
            g0 = int(C["stats"]["unique_graphs"])
            t0 = time.perf_counter()
            out = self._cc_wrap(name, f, args) if self._cc_wrap is not None else f(*args)
            dt = time.perf_counter() - t0
            g1 = int(C["stats"]["unique_graphs"])
            self.cc["calls"][name] = self.cc["calls"].get(name, 0) + 1
            if g1 != g0:
                self.cc["events"].append(dict(name=name, graphs=g1 - g0, s=round(dt, 3), layer=int(l),
                                              joint_pass=int(self.calls["joint_passes"]), capture=bool(self._cc_cap)))
                self.cc["compile_s"] += dt
            return out

        def _cc_head(self, l, eng, pbs, plan):
            """The section's start (STK_S / STK_K): the hoist at layer 0, the layer count. Returns hp."""
            vs = plan.vs
            if l == 0:
                self._cur = self._hoist(eng, pbs, plan, vs)
            hp = self._cur
            if hp is None or hp.R != plan.R:
                raise SpecError("%s: layer %d of a joint pass without its hoist (R %d)" % (type(self).__name__, l,
                                                                                         plan.R))
            self.hoist["layers"] += 1
            self.cc["sections"] += 1
            return hp

        def _cc_write(self, l, twin, k4, v4, pbs, plan, hp, past):
            """Step 3: layer 0 eager (tpos, the counters: STK_S), layers >= 1 the compiled write."""
            R = plan.R
            if l == 0:
                for r in range(R):
                    self._write(l, twin, r, k4[:, r], v4[:, r], pbs[r], plan, past[:, r])
                return
            if k4.dtype != self.k[l].dtype or v4.dtype != self.v[l].dtype:     # the integer views need equal dtypes
                raise SpecError("%s: K / V rows %s / %s differ from the buffer %s / %s" % (
                    type(self).__name__, k4.dtype, v4.dtype, self.k[l].dtype, self.v[l].dtype))
            self._cc("write", l, bits(self.k[l]), bits(self.v[l]), bits(self.bias[l]), bits(twin.total_cis),
                     self.all_b, hp.i, hp.prow, plan.act.contiguous(), bits(k4.contiguous()), bits(v4.contiguous()))

        def _cc_locate(self, l, eng, sel, sel_c, is_q, is_n, hp, vs, fetch: bool):
            """Step 5 (STK_S's _locate_h and the section's present / phys): eager sorts, compiled loc1 / loc2; the
            accounting (ops.stats) eager on the compiled outputs, as _locate_h's. Returns loc2's outputs and lst."""
            rs = self.rs
            H, B, R, T = sel.shape
            st = vs.ops.stats
            v = sel_c.view(H, B, R * T)
            if vs.ops.check:                                           # CHECK-ONLY: the dropped test, recounted
                self.hoist_next += ((v == hp.qb1.view(1, B, 1)) & (v >= 0)).sum()
            ws, wi = torch.sort(eng._block_map, dim=-1)
            if rs.pool:
                ps, pi = torch.sort(eng._pool_map, dim=-1)
            else:
                ps = pi = None
            valid, in_win, wslot, in_pool, pslot, keyh = self._cc("loc1", l, v, ws, wi, ps, pi, hp.fullv, vs.big,
                                                                  int(vs.topk))
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

        def _cc_lists(self, l, vs, keys_in, phys_f, present_f):
            """Step 6's row lists: the eager sort, the compiled gather / counts / copies."""
            keys, order = torch.sort(keys_in, dim=-1)
            pslot = vs.pslot.clone() if self.ccneg == "lost_lists" else vs.pslot   # NEGATIVE CONTROL (tests only)
            self._cc("lists", l, keys, order, phys_f, present_f, vs.pkey, pslot, vs.npg)

        def _cc_tail(self, l, eng, sel, present, lst_v, hp, plan, s0, skip0, P, past, qf):
            """The section's end (STK_S / STK_K verbatim): the accounting, the checks, the "F" probe."""
            vs = plan.vs
            ops = vs.ops
            B, R, bs = self.B, plan.R, self.bs
            full, res, qb = hp.full, hp.res, hp.qb
            if ops.stats:                                              # ACCOUNTING (device counters), STF:187
                s = vs.stat
                roles = (full | res).t()
                s["rows_full"] += full.sum()
                s["rows_res"] += res.sum()
                s["win_sel"] += ((s0 >= 0) & plan.act[0].view(1, B, 1)).sum()
                s["win_host"] += ((eng._load_mask >= 0) & ~skip0).sum()
                s["past_edge"] += (past & roles).sum()
                s["f4_rows"] += ((~past) & (P % bs == bs - 1) & roles).sum()
            VR._mark(vs, l, "check")
            if ops.check:                                              # CHECK-ONLY, STF:212-240
                vs.chk_sel[l].copy_(sel)
                miss = ~present & (sel >= 0)
                f4 = full.t().reshape(1, B, R, 1)
                r4 = res.t().reshape(1, B, R, 1)
                d = miss & f4
                s_ = miss & r4
                s = vs.stat
                s["dropped"] += d.sum()
                s["skipped"] += s_.sum()
                vs.stat_row["dropped"] += d.sum((0, 1, 3))
                vs.stat_row["skipped"] += s_.sum((0, 1, 3))
                vis_cnt, bad, want, role, b0, tail = self._vb(vs, eng, sel, present, lst_v, full, res, P, qb, l)
                H = sel.shape[0]
                vbad = role & (vis_cnt != want)
                s["vis_bad"] += vbad.sum()
                s["vb_bad"] += bad.sum()
                s["vb_rows"] += role.sum()
                s["block0_bad"] += b0.sum()
                s["tail_reads"] += tail
                s["visible_blocks"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).sum()
                s["fork_loaded"] += vs.fork_len.to(torch.int64).sum()
                vs.stat_row["vis_bad"] += vbad.view(H, B, R).sum((0, 1))
                vs.stat_row["visible"] += torch.where(role, vis_cnt, torch.zeros_like(vis_cnt)).view(H, B, R).sum(
                    (0, 1))
                vs.stat_row["selected"] += (sel >= 0).sum(-1).sum((0, 1))
                cbad, crows, creads = self._chain_check(l, eng, sel, P, qb, past, full | res)
                s["chain_bad"] += cbad
                s["chain_rows"] += crows
                s["chain_reads"] += creads
            if vs.fa_check is not None and l == 0 and "F" not in vs.fa_check:
                VR._fork_probe(vs, qf)                                 # CHECK-ONLY: "F" and "RA"
            VR._mark(vs, l, "post")

        def _cc_fork(self, vs, q4, B, R):
            E = B * R
            Hq, Dh = q4.shape[2], q4.shape[3]
            qf = q4.reshape(E, Hq, Dh).contiguous()
            VR._fork_call(vs, qf, vs.fout, vs.fork_cu, vs.fork_seq, vs.fork_tab, vs.fork_ext, vs.fork_len, vs.lists,
                          vs.fork_qmax)
            vs.calls["attn"] += 1
            return qf, vs.fout.reshape(B, R, -1)

        def cc_report(self) -> dict:
            """The arm's compile record (host values; dynamo's counters as deltas over the arm)."""
            c1 = dynamo_counts()
            d = {k: (None if c1[k] is None or self._cc_c0.get(k) is None else c1[k] - self._cc_c0[k]) for k in c1}
            ev = list(self.cc["events"])
            return dict(backend=self._cc_backend, ccneg=self.ccneg, calls=dict(self.cc["calls"]), events=ev,
                        compile_s=round(self.cc["compile_s"], 3), sections=self.cc["sections"],
                        fallback=dict(self.cc["fallback"]), first_capture_pass=self.cc["first_capture_pass"],
                        dynamo=d, dtype_ok=bool(self._cc_ok))

    # ------------------------------------------------------------------ L4PTFDGUDC
    class TentStateFDGUDC(_CCMix, SG):
        """L4PTFDGUD whose hoisted joint section (STK_S) runs through the compiled helpers (module docstring)."""

        def _usection(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            why = self._cc_why(plan)
            if why is not None:
                self.cc["fallback"][why] = self.cc["fallback"].get(why, 0) + 1
                return super()._usection(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            vs = plan.vs
            rs, B, R, bs = self.rs, self.B, plan.R, self.bs
            kern = rs.ops.kernels
            hp = self._cc_head(l, eng, pbs, plan)
            qb3, P, past = hp.qb3, hp.P, hp.past
            ti, inj, sel_c, is_q, is_n = self._cc("pre", l, sel, qb3, hp.qb4, hp.qb1_4, vs.minus1)
            VR._mark(vs, l, "resid")
            # 2. THE RESIDENCY (eager; under a capture I4G's stub)
            s0 = sel[:, :, 0]
            self._in_union = True
            try:
                skip0 = self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)
            finally:
                self._in_union = False
            VR._mark(vs, l, "write")
            self._cc_write(l, twin, k4, v4, pbs, plan, hp, past)       # 3. the rows into the tentative buffer
            VR._mark(vs, l, "chain")
            self._chain_h(l, eng, hp)                                  # 4. the chain images (eager: arena views)
            VR._mark(vs, l, "union")
            (present, phys, keys_in), lst = self._cc_locate(l, eng, sel, sel_c, is_q, is_n, hp, vs, True)
            VR._mark(vs, l, "fetch")
            kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
            kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
            vs.calls["gather_host"] += 2
            vs.calls["fills"] += 1
            VR._mark(vs, l, "attn")
            self._cc_lists(l, vs, keys_in, phys, present)              # 6. the row lists (last_len: the hoist's)
            qf, out = self._cc_fork(vs, q4, B, R)
            self._cc_tail(l, eng, sel, present, lst, hp, plan, s0, skip0, P, past, qf)
            return out

    class TentRunnerFDGUDC(RG):
        """The arm L4PTFDGUDC. Keyword arguments: STK's (dgneg, gneg, hneg, sneg, graph_backend), cc_backend (BACKENDS;
        the cells: "inductor"), ccneg (CCNEG, tests only)."""

        def __init__(self, *args, cc_backend: str = "inductor", ccneg=None, **kw):
            if cc_backend not in BACKENDS or ccneg not in CCNEG:
                raise SpecError("compile backend %r / control %r (one of %s / %s)" % (cc_backend, ccneg, BACKENDS,
                                                                                     CCNEG))
            self._ccopt = (cc_backend, ccneg)
            super().__init__(*args, **kw)
            _wrap_capture(self.T)
            self.meta.update(compile=dict(arm=self._arm_name(), backend=cc_backend, ccneg=ccneg, helpers=HELPERS,
                                          torch=torch.__version__))

        def _arm_name(self):
            return "L4PTFDGUDC"

        def _make_state(self, rs, attn, tneg, poison):
            T = TentStateFDGUDC(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg)
            T._cc_setup(*self._ccopt)
            return T

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(compile=self.T.cc_report())
            return out

    # ------------------------------------------------------------------ L4PTFDGUDKWC
    class TentStateFDGUDKC(_CCMix, SGK):
        """L4PTFDGUDKW whose combined keep section (STK_K) runs through the compiled helpers (module docstring)."""

        def _section(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            why = self._cc_why(plan)
            if why is not None:
                self.cc["fallback"][why] = self.cc["fallback"].get(why, 0) + 1
                return super()._section(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            vs = plan.vs
            ops = vs.ops
            rs, B, R, bs = self.rs, self.B, plan.R, self.bs
            kern = rs.ops.kernels
            hp = self._cc_head(l, eng, pbs, plan)
            qb3, P, past = hp.qb3, hp.P, hp.past
            ti, inj, sel_c, is_q, is_n = self._cc("pre", l, sel, qb3, hp.qb4, hp.qb1_4, vs.minus1)
            VR._mark(vs, l, "resid")
            # 2. THE RESIDENCY (eager; under a capture I4G's stub): row 0's selection; kwin from the kept store
            s0 = sel[:, :, 0]
            self._in_union = True
            try:
                skip0 = self._resid_joint(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3, vs)
            finally:
                self._in_union = False
            VR._mark(vs, l, "write")
            self._cc_write(l, twin, k4, v4, pbs, plan, hp, past)
            VR._mark(vs, l, "chain")
            self._chain_h(l, eng, hp)
            VR._mark(vs, l, "union")
            (present, phys0, in_lst), lst = self._cc_locate(l, eng, sel, sel_c, is_q, is_n, hp, vs, False)
            VR._mark(vs, l, "fetch")
            lst_v = lst
            loc = dict(present=present, phys=phys0, lst=lst, in_lst=in_lst)
            lphys, valid = self._keep_fill(l, vs, eng, twin, lst, loc, kern)   # the keep (eager)
            phys, keys_in = self._cc("kphys", l, is_q, is_n, self.cb0_id, vs.store_b.view(1, B, 1, 1), lphys,
                                     present, sel, vs.big)
            if ops.check:                                              # the slot-ordered ids (check)
                lst_v = SK.KP.slot_ids(self.kid, 0 if self.kneg == "layer_blind" else l, vs.minus1)
            vs.calls["gather_host"] += 2
            vs.calls["fills"] += 1
            VR._mark(vs, l, "attn")
            self._cc_lists(l, vs, keys_in, phys, present)
            qf, out = self._cc_fork(vs, q4, B, R)
            self._cc_tail(l, eng, sel, present, lst_v, hp, plan, s0, skip0, P, past, qf)
            return out

    class TentRunnerFDGUDKC(RGK):
        """The arm L4PTFDGUDKWC (keep=True, kwin=True; kcheck in check cells). Keyword arguments: STK's, cc_backend,
        ccneg (tests only)."""

        def __init__(self, *args, cc_backend: str = "inductor", ccneg=None, **kw):
            if cc_backend not in BACKENDS or ccneg not in CCNEG:
                raise SpecError("compile backend %r / control %r (one of %s / %s)" % (cc_backend, ccneg, BACKENDS,
                                                                                     CCNEG))
            self._ccopt = (cc_backend, ccneg)
            super().__init__(*args, **kw)
            _wrap_capture(self.T)
            self.meta.update(compile=dict(arm=self._arm_name(), backend=cc_backend, ccneg=ccneg, helpers=HELPERS,
                                          torch=torch.__version__))

        def _arm_name(self):
            return "L4PTFDGUDKWC"

        def _make_state(self, rs, attn, tneg, poison):
            T = TentStateFDGUDKC(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                 gneg=self._gneg, dgneg=self._dgneg, hneg=self._hneg, vs=self.vs, sneg=self._sneg,
                                 cneg=self._cneg, **self._k)
            T._cc_setup(*self._ccopt)
            return T

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(compile=self.T.cc_report())
            return out

    return dict(L4PTFDGUDC=(TentStateFDGUDC, TentRunnerFDGUDC), L4PTFDGUDKWC=(TentStateFDGUDKC, TentRunnerFDGUDKC))


def _wrap_capture(T) -> None:
    """The capture flag (module docstring, COMPILE EVENTS): the joint capture of T.backend runs with T._cc_cap True and
    records the first capture pass. An instance attribute over the backend's capture()."""
    be = T.backend
    orig = be.capture

    def capture(fn, T_, vs):
        if T.cc["first_capture_pass"] is None:
            T.cc["first_capture_pass"] = int(T.calls["joint_passes"])
        T._cc_cap = True
        try:
            return orig(fn, T_, vs)
        finally:
            T._cc_cap = False
    be.capture = capture


def compile_gate(arm: str, rd, L: int, names) -> tuple:
    """<arm>-COMPILE (MUST): every section ran the compiled path (no fallback; sections = eager + warm + captures, the
    graph report's); every helper in `names` called; no graph break; no compile event at or after the first capture
    (and none inside one); at most one compile per helper in the arm (no guard churn)."""
    c = rd.get("compile") or {}
    g = rd.get("graph") or {}
    ev = c.get("events") or []
    fc = c.get("first_capture_pass")
    late = [e for e in ev if e.get("capture") or (fc is not None and e.get("joint_pass", -1) >= fc)]
    per = {}
    for e in ev:
        per[e["name"]] = per.get(e["name"], 0) + 1
    calls = c.get("calls") or {}
    want_sections = g.get("eager", 0) + g.get("warm", 0) + g.get("captures", 0)
    brk = (c.get("dynamo") or {}).get("graph_breaks")
    ok = (not c.get("fallback") and c.get("sections") == want_sections and want_sections > 0
          and all(calls.get(n, 0) > 0 for n in names) and brk == 0 and not late and all(v <= 1 for v in per.values())
          and c.get("dtype_ok") is True)
    return ("%s-COMPILE" % arm, True, bool(ok),
            "backend %s; sections %s (eager + warm + captures %s); fallback %s; calls %s; compile events %s (late %s; "
            "per helper %s); compile %s s; graph breaks %s; dynamo %s" % (
                c.get("backend"), c.get("sections"), want_sections, c.get("fallback"), calls,
                [(e["name"], e["joint_pass"], e["layer"], e["s"]) for e in ev], len(late), per, c.get("compile_s"), brk,
                c.get("dynamo")))
