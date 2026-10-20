"""R2 milestone 2: the per-request rounds RUN in the pilot (NOSI_SPEC_COMMIT=perreq). CPU tests of every piece:

  EP  the engine methods cache_engine.spec_draft_update_perreq / spec_round_update_perreq / spec_commit_perreq, executed
      from the source (AST) against a fake engine with the twins of the two Triton gathers, as the pilot's tests do:
      EP1 two full rounds (draft K = 3, restore, verify, ragged commit) on a ragged batch equal B independent B = 1
      runs of the LOCKSTEP methods, bit for bit, after every phase; EP2 an overflowing request is isolated (its store
      back, no fetch, a finite placeholder mask, ctx.round_bad) and the round goes on for the others; negative control:
      the plan alone leaves the overflowing stream claiming never-fetched blocks; EP3 (a finding of the LOCKSTEP pilot,
      documented): spec_round_update raises RoundOverflow AFTER plan_round marked unfetched blocks RESIDENT; EP4 the
      lockstep and per-request methods refuse each other's engines.
  VS  verify_step.build_varlen_args takes the per-request seqused_k (and refuses a wrong one); the lockstep path holds.
  MP  spec_perreq_model: MP1 the bridge equals the twin after the shipped prefill and decode steps (the twin suite's
      ShippedRef with the shipped F2 line), and refuses an F2-short table; MP2 score_position on a ragged batch equals
      B = 1 scoring (stand-ins for the two kernels: a torch stage 1, pool_reference); MP3 the per-request rollback probe
      passes after a real round and fails on a corrupted tail row and on a round without the masked restore.
  FX  the slot-map stride defect (fixed 2026-10-20: spec_loop._assign returned a narrowed view; the Triton gathers
      read their map from the data pointer as if contiguous): FX1 with gathers that ADDRESS THE MAP AS THE KERNELS DO,
      every draft prefetch lands in the slot the plan names and slot 63 is untouched (negative control: the pre-fix map
      misplaces blocks); FX2 the maps are contiguous, and the verify's fetch already was before the fix (the reach);
      FX3 both gather wrappers refuse a non-contiguous map. MP3b: replay_deterministic compares two replays.
  F4  findings F4 parity (added 2026-10-20 after run p2: every above-tolerance identity mismatch was at p % 64 == 63,
      where the shipped decode attends rows [0, 4032) only). The rows the shipped decode attends are written down from
      the shipped lines (_shipped_tail_rows / _shipped_reads). F4V1: for every (request, position, KV head) of a
      ragged round with F4 positions at u = 0, 2, 3 and of a U = 1 round, the per-request verify (default f4 =
      "shipped") reads exactly those positions; NEGATIVE CONTROL f4 = "fixed": the F4 positions read the tail block
      more, nothing else differs. F4D1: the same for the draft's bias (the tail block rows it leaves visible).
      F4R1: a bad mode is refused before anything changes; the F4 drop runs before the overflow isolation. F4P1:
      the pilot's NOSI_SPEC_F4 defaults to "shipped", is recorded in the cell and shown in the report.
      EP1 / EP5 compare against the LOCKSTEP bodies, which read the tail block at an F4 step: they run f4 = "fixed".
  PL  the pilot's host side: perreq_keep and PerReqLoop (ragged sequences, finished requests stop committing, the K = 0
      fallback, refusals), aggregate_rounds_perreq (the measured yield), the report's gates for a per-request cell.
  AN  benchmarks/Efficiency/r2_payload_analysis.py on a synthetic payload.
  ST / PIN  the step script's per-request cells (bash -n, text), its holder uuid file reader (ST2), the wiring pins.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        "benchmarks/Efficiency/tests_cpu/test_spec_perreq_pilot.py::<test>"

Modules of nosi/nosi are imported as members of a stand-in package (no nosi/nosi/__init__.py, which imports CUDA
modules); the pilot is loaded by path in report mode. Data: integer-valued float32 in [-8, 8] (exact in bf16).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PKG = "nosipkg_perreq_pilot_test"
if PKG not in sys.modules:
    _pkg = types.ModuleType(PKG)
    _pkg.__path__ = [str(ROOT / "nosi/nosi")]
    sys.modules[PKG] = _pkg
spm = importlib.import_module(PKG + ".spec_perreq_model")
sl = spm._sl                                    # ONE spec_loop instance for the engine bodies, the model module, the tests
tw = spm._tw
twin_mod = importlib.import_module(PKG + ".r2_positional_twin")
us = importlib.import_module(PKG + ".verify.union_store")
vs = importlib.import_module(PKG + ".verify.verify_step")
mp = importlib.import_module(PKG + ".max_pooling_per_request")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


os.environ.setdefault("NOSI_SPEC_OUT", tempfile.mkdtemp(prefix="nosi_spec_perreq_pilot_test_"))
os.environ["NOSI_SPEC_MODE"] = "report"
pilot = _load(ROOT / "benchmarks/Efficiency/spec_loop_pilot.py", "nosi_spec_loop_pilot_perreq_test")
SRC_ENGINE = (ROOT / "nosi/nosi/cache_engine.py").read_text()
SRC_LLAMA = (ROOT / "nosi/nosi/nosa_llama.py").read_text()
SRC_SPM = (ROOT / "nosi/nosi/spec_perreq_model.py").read_text()
STEP = ROOT / "benchmarks/Efficiency/spec_loop_r2_step.sh"

BS, TOPK, R, H, D = 64, 8, 5, 2, 4                # W = 15 slots
KSEL = 6
LAY = sl.store_layout(TOPK, R)
EDT = torch.bfloat16


def _g(*parts):
    s = 0
    for p in parts:
        s = (s * 1000003 + int(p)) % (2 ** 61 - 1)
    return torch.Generator().manual_seed(s)


def _ints(shape, g):
    return torch.randint(-8, 9, tuple(shape), generator=g).to(torch.float32)


# ===========================================================================
# the fake engine: the attributes the spec bodies read, the bodies from the source, the gathers' twins
# ===========================================================================
METHODS = ("spec_draft_update", "spec_round_update", "spec_rollback_tail",
           "spec_draft_update_perreq", "spec_round_update_perreq", "spec_commit_perreq")


def _method_src(name):
    tree = ast.parse(SRC_ENGINE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CacheEngine")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC_ENGINE, fn))


def _gather_k(gpu, cpu, ids, bs):
    """flash_h2d_from_mask's effect: slot m of (h, b) <- host block ids[h, b, m]."""
    for h, b, m in (ids >= 0).nonzero().tolist():
        blk = int(ids[h, b, m])
        gpu[b, m * bs:(m + 1) * bs, h] = cpu[b, blk * bs:(blk + 1) * bs, h]


def _gather_v(gpu, cpu, bias_buf, bias, ids, bs):
    """flash_h2d_from_mask_bias's effect: the values and the bias rows of the same blocks."""
    _gather_k(gpu, cpu, ids, bs)
    for h, b, m in (ids >= 0).nonzero().tolist():
        blk = int(ids[h, b, m])
        bias_buf[b, m * bs:(m + 1) * bs, h] = bias[b, blk * bs:(blk + 1) * bs, h].to(bias_buf.dtype)


NS = {"torch": torch, "_sl": sl, "_tw": tw, "_us": us, "_vs": vs, "_bias_rows": lambda x: x,
      "flash_h2d_from_mask": _gather_k, "flash_h2d_from_mask_bias": _gather_v}


def _flat_ids(ids):
    """What the Triton gathers read: entry (h, b, m) at h * B * M + b * M + m FROM THE DATA POINTER
    (flash_h2d_mask.py:34, flash_h2d_mask_bias.py:34), whatever the strides of the tensor they are handed."""
    Hn, Bn, Mn = ids.shape
    return torch.as_strided(ids, (Hn * Bn * Mn,), (1,), ids.storage_offset())


def _kernel_gather_k(gpu, cpu, ids, bs):
    """flash_h2d_from_mask as the KERNEL addresses its map (the twins above index the tensor, so they cannot see a
    stride defect)."""
    Hn, Bn, Mn = ids.shape
    flat = _flat_ids(ids)
    for h in range(Hn):
        for b in range(Bn):
            for m in range(Mn):
                blk = int(flat[h * Bn * Mn + b * Mn + m])
                if blk >= 0:
                    gpu[b, m * bs:(m + 1) * bs, h] = cpu[b, blk * bs:(blk + 1) * bs, h]


def _kernel_gather_v(gpu, cpu, bias_buf, bias, ids, bs):
    _kernel_gather_k(gpu, cpu, ids, bs)
    Hn, Bn, Mn = ids.shape
    flat = _flat_ids(ids)
    for h in range(Hn):
        for b in range(Bn):
            for m in range(Mn):
                blk = int(flat[h * Bn * Mn + b * Mn + m])
                if blk >= 0:
                    bias_buf[b, m * bs:(m + 1) * bs, h] = bias[b, blk * bs:(blk + 1) * bs, h].to(bias_buf.dtype)


NS_KERNEL = dict(NS, flash_h2d_from_mask=_kernel_gather_k, flash_h2d_from_mask_bias=_kernel_gather_v)


def _bind(eng, ns_base=None):
    for name in METHODS:
        ns = dict(ns_base or NS)
        exec(_method_src(name), ns)
        setattr(eng, name, types.MethodType(ns[name], eng))
    return eng


def _engine(pos, rids, seed, l, n_max, perreq, ns_base=None):
    """One layer's engine for requests ``rids`` at positions ``pos``; request rid's data from its own generator, so a
    B = 1 engine of request rid equals slice b of a batched one."""
    B = len(pos)
    eng = types.SimpleNamespace(block_size=BS, topk=TOPK, head_num=H, head_dim=D, _tail_block_idx_on_gpu=TOPK - 1,
                                verify_round_slots=R, pool_blocks=0, device=None)
    parts = {k: [] for k in ("k_gpu", "v_gpu", "b_gpu", "k_cpu", "v_cpu")}
    eng._block_map = torch.full((H, B, TOPK), -1, dtype=torch.int64)
    for b, (p, rid) in enumerate(zip(pos, rids)):
        g = _g(seed, rid, l, 1)
        parts["k_gpu"].append(_ints((LAY.W * BS, H, D), g))
        parts["v_gpu"].append(_ints((LAY.W * BS, H, D), g))
        parts["b_gpu"].append(_ints((LAY.W * BS, H), g))
        parts["k_cpu"].append(_ints((n_max, H, D), g))
        parts["v_cpu"].append(_ints((n_max, H, D), g))
        T = p // BS
        for h in range(H):
            eng._block_map[h, b, :TOPK - 1] = torch.randperm(T, generator=g)[:TOPK - 1]
        eng._block_map[:, b, TOPK - 1] = T
    eng._k_gpu = torch.stack(parts["k_gpu"]).to(EDT)
    eng._v_gpu = torch.stack(parts["v_gpu"]).to(EDT)
    eng._kv_bias_gpu = torch.stack(parts["b_gpu"]).to(EDT)
    eng._k_cpu = torch.stack(parts["k_cpu"]).to(EDT)
    eng._v_cpu = torch.stack(parts["v_cpu"]).to(EDT)
    pt = torch.tensor(pos, dtype=torch.int64)
    eng._cache_lens = ((TOPK - 1) * BS + pt % BS).to(torch.int32)
    if len(set(pos)) == 1:
        eng.seq_length, eng._tail_block_len_on_gpu = pos[0], pos[0] % BS
        if perreq:
            sl.to_perreq(eng)
    else:
        assert perreq, "a lockstep engine holds one seq_length"
        eng.seq_length = eng._tail_block_len_on_gpu = None
        eng._pos = pt.clone()
        sl.check_tail_state(eng, eng._pos)
    return _bind(eng, ns_base)


def _tables(pos, rids, seed, l, n_max):
    t = twin_mod.PositionalTwin(len(pos), H, D, lk_max=n_max // 16, n_max=n_max)
    for b, (S, rid) in enumerate(zip(pos, rids)):
        g = _g(seed, rid, l, 2)
        t.prefill(b, _ints((S, H, D), g), _ints((S, H), g))
    return t


class FakeEvent:
    def __init__(self):
        self.done = False
        self.stream = None

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        return self.done


class _Side:
    def __init__(self):
        self.waited = []

    def wait_event(self, e):
        self.waited.append(e)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Uni:
    """One batch of NL layers: engines with the bound spec bodies, twins, one store, a context; per-request or
    lockstep. ``f4``: the per-request context's F4 mode (None: the context's default, "shipped")."""

    def __init__(self, pos, rids, seed, n_max, perreq, NL=1, ns_base=None, f4=None):
        self.B, self.NL, self.perreq = len(pos), NL, perreq
        self.engines = [_engine(pos, rids, seed, l, n_max, perreq, ns_base) for l in range(NL)]
        self.twins = [_tables(pos, rids, seed, l, n_max) for l in range(NL)]
        self.store = sl.Store(NL, H, self.B, LAY)
        cls = sl.PerReqSpecContext if perreq else sl.SpecContext
        kw = dict(f4=f4) if (perreq and f4 is not None) else {}
        self.ctx = cls(self.store, side=_Side(), make_event=FakeEvent, current_stream=lambda: "main",
                       stream_ctx=lambda s: _Null(), num_layers=NL, block_size=BS, kernel_size=32, kernel_stride=16, **kw)
        tick = self.ctx.next_tick()
        for l, e in enumerate(self.engines):
            self.store.sync_window(l, e._block_map, tick)
        self.state = spm.PerReqState(self.twins, self.engines, 4, 6, n_max // BS + 2, torch.float32, "cpu")
        self.view = self.state.view

    def pos(self):
        e = self.engines[0]
        return e._pos.clone() if self.perreq else torch.full((self.B,), int(e.seq_length), dtype=torch.int64)


def _sel(g, maxid, k=KSEL):
    rows = []
    for _ in range(H):
        r = torch.randperm(maxid + 1, generator=g)[:k].clone()
        if int(torch.randint(0, 2, (1,), generator=g)) == 1:
            r[-1] = -1
        rows.append(r)
    return torch.stack(rows)


def _inputs(seed, rnd, rids, pos0, K, n_acc, NL=1):
    """Round inputs per request from its own generator, stacked; a selection at position p names blocks <= p // 64."""
    U = K + 1
    per = []
    for rid, p, na in zip(rids, pos0.tolist(), n_acc):
        g = _g(seed, rid, 1000 + rnd, 3)
        d = dict(dk=_ints((K, H, D), g), dv=_ints((K, H, D), g), dc=_ints((K, H), g),
                 vk=_ints((U, H, D), g), vv=_ints((U, H, D), g), vc=_ints((U, H), g),
                 sd=[[_sel(g, (p + j) // BS) for _ in range(NL)] for j in range(K)],
                 sv=[torch.stack([_sel(g, (p + u) // BS) for u in range(U)]) for _ in range(NL)])
        drafted = torch.randint(0, 1000, (K,), generator=g)
        va = torch.randint(0, 1000, (U,), generator=g)
        va[:K] = drafted
        if na < K:
            va[na] = (drafted[na] + 1) % 1000
        d["drafted"], d["va"] = drafted, va
        per.append(d)
    return types.SimpleNamespace(
        K=K, U=U,
        dk=[torch.stack([x["dk"][j] for x in per]) for j in range(K)],
        dv=[torch.stack([x["dv"][j] for x in per]) for j in range(K)],
        dc=[torch.stack([x["dc"][j] for x in per]) for j in range(K)],
        vk=torch.stack([x["vk"] for x in per]), vv=torch.stack([x["vv"] for x in per]), vc=torch.stack([x["vc"] for x in per]),
        sd=[[torch.stack([x["sd"][j][l] for x in per], 1) for l in range(NL)] for j in range(K)],
        sv=[torch.stack([x["sv"][l] for x in per], 2) for l in range(NL)],
        drafted=torch.stack([x["drafted"] for x in per]), va=torch.stack([x["va"] for x in per]))


# ===========================================================================
# the two engine pipelines (the order of the model's forwards: the tables step before each engine update)
# ===========================================================================
def _draft_and_restore_perreq(u, inp, rec):
    ctx, st = u.ctx, u.store
    ctx.begin_round(u.view)
    done = ctx.gen
    for j in range(inp.K):
        ctx.next_tick()
        st.mark_arrived(done)
        done = ctx.gen
        for l in range(u.NL):
            e, t = u.engines[l], u.twins[l]
            t.step(inp.dk[j], inp.dc[j])
            k_all, v_all, bias, lens = e.spec_draft_update_perreq(inp.dk[j].unsqueeze(1), inp.dv[j].unsqueeze(1), t.total_cis,
                                                                 inp.sd[j][l], ctx, l)
            assert k_all is e._k_gpu and v_all is e._v_gpu and lens.tolist() == [LAY.W * BS] * u.B
            rec["draft"].append(dict(plan=e._spec_last_draft, bias=bias.clone(), rolled=ctx.rolled[l].clone()))
    ctx.restore_before_verify(u.view)
    ctx.was_inflight = (st.state == sl.INFLIGHT).clone()
    st.mark_arrived(ctx.gen)
    ctx.next_tick()


def _verify_perreq(u, inp, rec):
    ctx = u.ctx
    rnds = []
    for l in range(u.NL):
        e, t = u.engines[l], u.twins[l]
        ctr = [t.counters()]
        for v in range(inp.U):
            t.step(inp.vk[:, v], inp.vc[:, v])
            ctr.append(t.counters())
        u.state.ctrs[l] = torch.stack(ctr)
        u.state.round_inputs[l] = (inp.vk, inp.vv, inp.vc)
        k_all, v_all, b_all, rnd = e.spec_round_update_perreq(inp.vk, inp.vv, t.total_cis, inp.sv[l], ctx, l)
        rnds.append(rnd)
        rec["round"].append(rnd.union)
        rec["tails"].append(dict(rolled_over=rnd.tail.rolled_over, rollover_at=rnd.tail.rollover_at, seqused_k=rnd.seqused_k))
    return rnds


def _round_perreq(u, inp, keep_fn=None):
    rec = dict(draft=[], round=[], tails=[], tr=[])
    _draft_and_restore_perreq(u, inp, rec)
    rec["rnds"] = _verify_perreq(u, inp, rec)
    acc = sl.accept_perreq(inp.va, inp.drafted)
    keep = acc.keep if keep_fn is None else keep_fn(acc.keep)
    keep = pilot.perreq_keep(keep, torch.zeros(u.B, dtype=torch.bool), u.ctx.round_bad)
    tr = spm.spec_rollback(u.view, u.ctx, u.state, keep)
    for l in range(u.NL):
        assert torch.equal(u.twins[l].pos, u.engines[l]._pos)
    rec["tr"].append(dict(undo=tr.undo_rollover, restored=tr.rows_restored))
    rec["keep"] = keep
    return rec


def _round_lock(u, inp):
    """B = 1 (or equal positions): the LOCKSTEP bodies spec_draft_update / spec_round_update / spec_rollback_tail."""
    ctx, st = u.ctx, u.store
    T = int(u.engines[0].seq_length) // BS
    tc = [sl.capture_tail_counters(e, T) for e in u.engines]
    t0 = [t.counters().unsqueeze(0) for t in u.twins]
    ctx.round_idx += 1
    ctx.tail_id, ctx.tail0, ctx.rolled, ctx.gen_round0 = T, tc[0], [False] * u.NL, ctx.gen + 1
    rec = dict(draft=[], round=[], tails=[], tr=[])
    done = ctx.gen
    for j in range(inp.K):
        ctx.next_tick()
        st.mark_arrived(done)
        done = ctx.gen
        for l in range(u.NL):
            e, t = u.engines[l], u.twins[l]
            t.step(inp.dk[j], inp.dc[j])
            _, _, bias, _ = e.spec_draft_update(inp.dk[j].unsqueeze(1), inp.dv[j].unsqueeze(1), t.total_cis, inp.sd[j][l], ctx, l)
            rec["draft"].append(dict(plan=e._spec_last_draft, bias=bias.clone(), rolled=torch.full((u.B,), ctx.rolled[l])))
    for l, (e, t) in enumerate(zip(u.engines, u.twins)):
        sl.restore_tail_counters(e, tc[l], ctx.rolled[l])
        t.commit(t0[l], torch.zeros(u.B, dtype=torch.int64))
    ctx.was_inflight = (st.state == sl.INFLIGHT).clone()
    st.mark_arrived(ctx.gen)
    ctx.next_tick()
    ctrs, rnds = [], []
    for l, (e, t) in enumerate(zip(u.engines, u.twins)):
        ctrs.append(t.advance(inp.vk, inp.vc, torch.full((u.B,), inp.U, dtype=torch.int64)))
        _, _, _, rnd = e.spec_round_update(inp.vk, inp.vv, t.total_cis, inp.sv[l], ctx, l)
        rnds.append(rnd)
        rec["round"].append(rnd.union)
        rec["tails"].append(dict(rolled_over=torch.full((u.B,), rnd.tail.rolled_over),
                                 rollover_at=torch.full((u.B,), rnd.tail.rollover_at, dtype=torch.int64), seqused_k=rnd.seqused_k))
    rec["rnds"] = rnds
    acc = sl.accept_commit(inp.va, inp.drafted)
    for l, (e, t) in enumerate(zip(u.engines, u.twins)):
        t.commit(ctrs[l], torch.full((u.B,), acc.c, dtype=torch.int64))
        tr = e.spec_rollback_tail(acc.c, LAY.mirror_lo)
    rec["tr"].append(dict(undo=torch.full((u.B,), tr.undo_rollover), restored=torch.full((u.B,), tr.rows_restored, dtype=torch.int64)))
    rec["keep"] = torch.full((u.B,), acc.c, dtype=torch.int64)
    return rec


TWIN_FIELDS = ("pos", "n_comp", "n_ccis", "prefilled", "comp_k", "ring", "total_cis")
PLAN_FIELDS = ("deny", "issue", "n_sel", "n_hit", "n_inflight", "n_issue", "n_unrequested", "n_forbidden", "n_evicted_unused")
ROUND_FIELDS = ("fetch", "overflow", "invalid", "n_new", "n_need", "n_hit_old", "n_timely", "n_late_wait", "n_late_fetch", "union_size")


def _state(u, b):
    d = {}
    for l, (e, t) in enumerate(zip(u.engines, u.twins)):
        pos = int(e._pos[b]) if u.perreq else int(e.seq_length)
        d["L%d.pos" % l] = torch.tensor(pos)
        d["L%d.tail_len" % l] = torch.tensor(pos % BS if u.perreq else int(e._tail_block_len_on_gpu))
        for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_cache_lens"):
            d["L%d.%s" % (l, name)] = getattr(e, name)[b]
        d["L%d._block_map" % l] = e._block_map[:, b]
        for name in TWIN_FIELDS:
            d["L%d.tables.%s" % (l, name)] = getattr(t, name)[b]
        d["L%d.tables.ccis" % l] = t.ccis[:, b]
    for name in sl.STORE_FIELDS + u.store.COUNTERS:
        d["store.%s" % name] = getattr(u.store, name)[:, :, b]
    return d


def _rec_state(rec, b, U):
    d = {}
    for i, x in enumerate(rec["draft"]):
        for f in PLAN_FIELDS:
            d["draft%d.%s" % (i, f)] = getattr(x["plan"], f)[:, b]
        d["draft%d.bias" % i] = x["bias"][b]
        d["draft%d.rolled" % i] = x["rolled"][b]
    for l, rp in enumerate(rec["round"]):
        d["round%d.mask" % l] = rp.mask[:, b * U:(b + 1) * U]
        for f in ROUND_FIELDS:
            d["round%d.%s" % (l, f)] = getattr(rp, f)[:, b]
    for l, x in enumerate(rec["tails"]):
        for k, v in x.items():
            d["tail%d.%s" % (l, k)] = v[b]
    for l, x in enumerate(rec["tr"]):
        for k, v in x.items():
            d["truncate%d.%s" % (l, k)] = v[b]
    d["keep"] = rec["keep"][b]
    return d


def _first_diff(da, db):
    if list(da) != list(db):
        return "keys differ: %s" % sorted(set(da) ^ set(db))
    for k in da:
        a, c = da[k], db[k]
        if a.dtype != c.dtype or a.shape != c.shape or not torch.equal(a, c):
            return "%s differs" % k
    return None


# ===========================================================================
# EP: the engine methods
# ===========================================================================
def test_EP1_two_ragged_rounds_through_the_engine_bodies_equal_B1_lockstep_runs():
    """f4 = "fixed": the LOCKSTEP bodies read the tail block at a findings-F4 step (p % 64 == 63; this batch has such
    steps), so the per-request run they equal bit for bit is the fixed one. F4V1 / F4D1 tie the default mode
    ("shipped") to the shipped decode and to this run."""
    pos = [9 * BS + 61, 10 * BS + 3, 9 * BS + 63, 11 * BS + 0]
    K, U, B, NL = 3, 4, 4, 2
    n_max = max(pos) + 4 * U + 2 * BS
    P = Uni(pos, list(range(B)), 21, n_max, perreq=True, NL=NL, f4="fixed")
    refs = [Uni([pos[b]], [b], 21, n_max, perreq=False, NL=NL) for b in range(B)]
    cov = Counter()
    for r, nacc in enumerate(([0, 3, 1, 2], [3, 0, 2, 1])):
        pos0 = P.pos()
        inp = _inputs(21, r, list(range(B)), pos0, K, nacc, NL)
        rp = _round_perreq(P, inp)
        for b in range(B):
            rb = _round_lock(refs[b], _inputs(21, r, [b], pos0[b:b + 1], K, [nacc[b]], NL))
            bad = _first_diff(_state(refs[b], 0), _state(P, b)) or _first_diff(_rec_state(rb, 0, U), _rec_state(rp, b, U))
            assert bad is None, "round %d request %d: %s" % (r, b, bad)
            # the verify call's arguments: request b's mask rows and length equal its B = 1 round's
            q = _ints((B * U, 4, D), _g(r, b)).to(EDT)
            a = vs.build_varlen_args(q, P.engines[0]._k_gpu, P.engines[0]._v_gpu, P.engines[0]._kv_bias_gpu, rp["rnds"][0])
            a1 = vs.build_varlen_args(q[b * U:(b + 1) * U].contiguous(), refs[b].engines[0]._k_gpu, refs[b].engines[0]._v_gpu,
                                      refs[b].engines[0]._kv_bias_gpu, rb["rnds"][0])
            assert torch.equal(a.topk_idx[:, b * U:(b + 1) * U], a1.topk_idx) and int(a.seqused_k[b]) == int(a1.seqused_k[0])
        assert len(set(rp["keep"].tolist())) > 1                                      # a ragged commit
        assert _no_slot_at_or_beyond_the_tail(P) == []
        cov["undo_restored"] += int((rp["tr"][0]["undo"] & (rp["tr"][0]["restored"] > 0)).sum())
        cov["undo_last"] += int((rp["tr"][0]["undo"] & (rp["tr"][0]["restored"] == 0)).sum())
        cov["kept_fill"] += int((rp["tails"][0]["rolled_over"] & ~rp["tr"][0]["undo"]).sum())
        cov["draft_rolled"] += sum(int(x["rolled"].sum()) for x in rp["draft"])
    for k in ("undo_restored", "kept_fill", "draft_rolled"):
        assert cov[k] > 0, (k, dict(cov))
    with pytest.raises(AssertionError):
        P.engines[0].spec_commit_perreq(P.twins[0], torch.zeros((U + 1, 4, B), dtype=torch.int64), torch.ones(B, dtype=torch.int64),
                                        LAY.mirror_lo)                                 # the round was consumed


def _no_slot_at_or_beyond_the_tail(u):
    """THE INVARIANT that makes a rollback across a block boundary safe without a resync: no slot of the store (window
    or round region, RESIDENT or INFLIGHT) names a block id >= the request's live tail id pos // 64. (plan_draft fetches
    only ids < T_b, the round-start tail; plan_round serves T_b and T_b + 1 from the mirrors and flags anything above
    as invalid; a commit never goes below the round-start position.) Returns the offending (layer, head, request,
    slot) list."""
    bad = []
    for l, e in enumerate(u.engines):
        tail = (e._pos // BS).view(1, -1, 1)
        live = u.store.state[l] != sl.FREE
        hit = live & (u.store.ids[l] >= tail) & (u.store.ids[l] >= 0)
        bad += [(l,) + tuple(x) for x in hit.nonzero().tolist()]
    return bad


def test_EP5_a_rollback_across_a_block_boundary_restores_slot_63_and_leaves_no_slot_at_or_beyond_the_tail():
    """Request 0 starts at row 62 of block 9: the draft fills block 9 at its 2nd step (prefetching into the store), the
    verify fills it at token 1 and writes tokens 2, 3 into block 10 (slot 63 rows 0, 1), and the commit keeps ONE
    token: pos 639, block 9 live again. Slot 63's rows [0, 62) are the round-start rows again (the masked restore),
    row 62 is the kept token, the map's tail column is 9, and no store slot names block 9 or 10. The next round runs
    from there and equals the B = 1 lockstep run. f4 = "fixed", as in EP1 (request 0's round has an F4 step)."""
    pos = [9 * BS + 62, 10 * BS + 5]
    K, U, B, NL = 3, 4, 2, 2
    n_max = max(pos) + 3 * U + 2 * BS
    P = Uni(pos, [0, 1], 23, n_max, perreq=True, NL=NL, f4="fixed")
    refs = [Uni([pos[b]], [b], 23, n_max, perreq=False, NL=NL) for b in range(B)]
    ts = LAY.tail_slot * BS
    rows0 = [e._k_gpu[0, ts:ts + 62].clone() for e in P.engines]
    for r, nacc in enumerate(([0, 2], [3, 1])):
        pos0 = P.pos()
        rp = _round_perreq(P, _inputs(23, r, [0, 1], pos0, K, nacc, NL))
        for b in range(B):
            rb = _round_lock(refs[b], _inputs(23, r, [b], pos0[b:b + 1], K, [nacc[b]], NL))
            bad = _first_diff(_state(refs[b], 0), _state(P, b)) or _first_diff(_rec_state(rb, 0, U), _rec_state(rp, b, U))
            assert bad is None, "round %d request %d: %s" % (r, b, bad)
        assert _no_slot_at_or_beyond_the_tail(P) == []
        if r == 0:
            assert bool(rp["tr"][0]["undo"][0]) and int(rp["tr"][0]["restored"][0]) == 2 and int(P.engines[0]._pos[0]) == 9 * BS + 63
            assert sum(int(x["rolled"][0]) for x in rp["draft"]) > 0                     # the draft had crossed too
            for l, e in enumerate(P.engines):
                assert torch.equal(e._k_gpu[0, ts:ts + 62], rows0[l]) and torch.equal(e._k_gpu[0, ts + 62], _inputs(23, 0, [0, 1], pos0, K, nacc, NL).vk[0, 0].to(EDT))
                assert int(e._block_map[0, 0, TOPK - 1]) == 9 and int(e._cache_lens[0]) == (TOPK - 1) * BS + 63


# ===========================================================================
# FX: the slot-map stride defect (fixed 2026-10-20): _assign returned ext[..., :W], a view with rows of W + 1, and the
# Triton gathers read their map at h * B * M + b * M + m from the data pointer
# ===========================================================================
def _old_assign(orig):
    """The pre-fix _assign: the same values as a narrowed (H, B, W + 1) view."""
    def f(want, ids, cand, key):
        issue, fits, n_cand = orig(want, ids, cand, key)
        Hn, Bn, Wn = issue.shape
        ext = torch.full((Hn, Bn, Wn + 1), -1, dtype=issue.dtype)
        ext[..., :Wn] = issue
        return ext[..., :Wn], fits, n_cand
    return f


def _draft_landing_violations(seed, steps=4):
    """Draft steps on a ragged batch with the KERNEL-faithful gathers. Returns the number of (head, request, slot)
    whose prefetch does not hold the host rows of the block the plan named, plus the number of requests whose slot-63
    rows below the round-start tail length changed (both 0 when the map is read as the plan wrote it)."""
    pos = [9 * BS + 50, 10 * BS + 3, 9 * BS + 40, 11 * BS + 0]
    B = len(pos)
    n_max = max(pos) + steps + 2 * BS
    P = Uni(pos, list(range(B)), seed, n_max, perreq=True, ns_base=NS_KERNEL)
    e, t, ctx, st = P.engines[0], P.twins[0], P.ctx, P.store
    ts = LAY.tail_slot * BS
    tl0 = [p % BS for p in pos]
    rows0 = [e._k_gpu[b, ts:ts + tl0[b]].clone() for b in range(B)]
    inp = _inputs(seed, 0, list(range(B)), P.pos(), steps, [0] * B)
    ctx.begin_round(P.view)
    bad = issued = 0
    for j in range(steps):
        ctx.next_tick()
        t.step(inp.dk[j], inp.dc[j])
        e.spec_draft_update_perreq(inp.dk[j].unsqueeze(1), inp.dv[j].unsqueeze(1), t.total_cis, inp.sd[j][0], ctx, 0)
        issue = e._spec_last_draft.issue
        for h, b, w in (issue >= 0).nonzero().tolist():
            issued += 1
            blk = int(issue[h, b, w])
            if not torch.equal(e._k_gpu[b, w * BS:(w + 1) * BS, h], e._k_cpu[b, blk * BS:(blk + 1) * BS, h]):
                bad += 1
    tail_hit = sum(1 for b in range(B) if not torch.equal(e._k_gpu[b, ts:ts + tl0[b]], rows0[b]))
    return bad, tail_hit, issued


def test_FX1_draft_prefetch_lands_in_the_slots_the_plan_names_with_the_kernels_addressing(monkeypatch):
    bad, tail_hit, issued = _draft_landing_violations(101)
    assert issued > 0 and bad == 0 and tail_hit == 0, (bad, tail_hit, issued)
    # NEGATIVE CONTROL: the pre-fix map (a narrowed view) puts blocks into slots the plan did not name
    monkeypatch.setattr(sl, "_assign", _old_assign(sl._assign))
    bad2, tail_hit2, _ = _draft_landing_violations(101)
    assert bad2 > 0, (bad2, tail_hit2)


def test_FX2_the_maps_handed_to_the_gathers_are_contiguous_and_the_verify_fetch_was_contiguous_before_the_fix(monkeypatch):
    """The fixed producer: plan_draft(_perreq).issue and plan_round(_perreq).fetch are contiguous. THE REACH of the
    defect: with the pre-fix _assign, plan_round's fetch was still contiguous (rebuilt by torch.where), so only the
    draft's prefetch (plan.issue) was misplaced; the verify's late fetch was not."""
    g = _g(102)
    st = sl.Store(1, H, 3, LAY)
    bm = torch.full((H, 3, TOPK), -1, dtype=torch.int64)
    for b in range(3):
        for h in range(H):
            bm[h, b, :TOPK - 1] = torch.randperm(10, generator=g)[:TOPK - 1]
        bm[:, b, TOPK - 1] = 10
    st.sync_window(0, bm, 1)
    sel = torch.stack([_sel(g, 10) for _ in range(3)], 1)                       # (H, 3, KSEL)
    T = torch.full((3,), 10, dtype=torch.int64)
    for old in (False, True):
        if old:
            monkeypatch.setattr(sl, "_assign", _old_assign(sl._assign))
        s1 = sl.Store(1, H, 3, LAY)
        for n in sl.STORE_FIELDS + st.COUNTERS:
            getattr(s1, n).copy_(getattr(st, n))
        d = sl.plan_draft(s1, 0, sel, 10, False, 1, 2, 10)
        dp = sl.plan_draft_perreq(s1, 0, sel, T, torch.zeros(3, dtype=torch.bool), 2, 3, T)
        s1.mark_arrived(10)
        su = torch.stack([sel, sel])                                             # (U = 2, H, 3, KSEL)
        r = sl.plan_round(s1, 0, su, 10, 2, 1, 4, 5, torch.zeros((H, 3, LAY.W), dtype=torch.bool))
        rp = sl.plan_round_perreq(s1, 0, su, T, torch.full((3,), 2), 1, 6, 7, torch.zeros((H, 3, LAY.W), dtype=torch.bool))
        assert r.fetch.is_contiguous() and rp.fetch.is_contiguous()             # the verify fetch, before and after the fix
        assert d.issue.is_contiguous() == (not old) and dp.issue.is_contiguous() == (not old)


def test_FX3_the_gather_wrappers_refuse_a_non_contiguous_map():
    fce = ROOT / "nosi/nosi/flash_cache_engine"
    for fname, fn in (("flash_h2d_mask.py", "flash_h2d_from_mask"), ("flash_h2d_mask_bias.py", "flash_h2d_from_mask_bias")):
        src = (fce / fname).read_text()
        tree = ast.parse(src)
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        first = f.body[0]
        assert isinstance(first, ast.Assert) and "load_ids.is_contiguous()" in ast.get_source_segment(src, first), fn
    m1 = _load(fce / "flash_h2d_mask.py", "nosi_flash_h2d_mask_guard_test")              # imports triton (no launch here)
    m2 = _load(fce / "flash_h2d_mask_bias.py", "nosi_flash_h2d_mask_bias_guard_test")
    ext = torch.full((H, 2, LAY.W + 1), -1, dtype=torch.int64)
    view = ext[..., :LAY.W]
    gpu, cpu = torch.zeros((2, LAY.W * BS, H, D)), torch.zeros((2, 4 * BS, H, D))
    with pytest.raises(AssertionError):
        m1.flash_h2d_from_mask(gpu, cpu, view, BS)
    with pytest.raises(AssertionError):
        m2.flash_h2d_from_mask_bias(gpu, cpu, torch.zeros((2, LAY.W * BS, H)), torch.zeros((2, 4 * BS, H)), view, BS)


def _overflow_setup(seed):
    """Request 0 at 9*64+10 with an ordinary selection; request 1 at 30*64+5 naming 32 distinct blocks over the round:
    more new blocks than the 12 evictable slots of its stream (overflow)."""
    pos = [9 * BS + 10, 30 * BS + 5]
    B, U = 2, 4
    n_max = max(pos) + 2 * BS
    P = Uni(pos, [0, 1], seed, n_max, perreq=True)
    g = _g(seed, 9)
    sel0 = torch.stack([_sel(g, 9) for _ in range(U)])                                       # (U, H, KSEL)
    big = torch.arange(32, dtype=torch.int64).view(U, 1, 8).expand(U, H, 8) % 30           # 30 distinct ids < T = 30
    sel = torch.full((U, H, B, 8), -1, dtype=torch.int64)
    sel[:, :, 0, :KSEL] = sel0
    sel[:, :, 1] = big
    k, v, c = _ints((B, U, H, D), g), _ints((B, U, H, D), g), _ints((B, U, H), g)
    return P, sel, k, v, c, pos


def test_EP2_an_overflowing_request_is_isolated_and_the_round_goes_on():
    P, sel, k, v, c, pos = _overflow_setup(31)
    B, U = 2, 4
    ctx, st, e, t = P.ctx, P.store, P.engines[0], P.twins[0]
    ctx.begin_round(P.view)
    ctx.was_inflight = torch.zeros_like(st.state, dtype=torch.bool)
    before = sl.store_layer_copy(st, 0)
    ref_store = sl.Store(1, H, 1, LAY)                     # request 0 alone, the lockstep plan
    for n in sl.STORE_FIELDS + st.COUNTERS:
        getattr(ref_store, n).copy_(getattr(st, n)[:, :, 0:1])
    ctr = [t.counters()]
    for w in range(U):
        t.step(k[:, w], c[:, w])
        ctr.append(t.counters())
    _, _, _, rnd = e.spec_round_update_perreq(k, v, t.total_cis, sel, ctx, 0)            # no RoundOverflow
    assert ctx.round_bad.tolist() == [False, True] and ctx.host_syncs == 1
    p = rnd.union
    assert bool(p.overflow[:, 1].all()) and not bool(p.overflow[:, 0].any())
    for n in sl.STORE_FIELDS + st.COUNTERS:                                                 # its store is back
        assert torch.equal(getattr(st, n)[0, :, 1], before[n][:, 1]), n
    assert bool((p.fetch[:, 1] == -1).all())
    m = p.mask.view(H, B, U, -1)
    assert bool((m[:, 1, :, 0] == LAY.mirror_lo).all()) and bool((m[:, 1, :, 1:] == -1).all())
    rb = sl.plan_round(ref_store, 0, sel[:, :, 0:1], int(pos[0]) // BS, int(rnd.tail.rollover_at[0]), ctx.gen_round0, ctx.gen,
                       ctx.tick, torch.zeros((H, 1, LAY.W), dtype=torch.bool))
    assert torch.equal(p.mask.view(H, B, U, -1)[:, 0], rb.mask.view(H, 1, U, -1)[:, 0]) and torch.equal(p.fetch[:, 0:1], rb.fetch)
    for n in sl.STORE_FIELDS + st.COUNTERS:                                                 # request 0's round is the lockstep one
        assert torch.equal(getattr(st, n)[:, :, 0:1], getattr(ref_store, n)), n
    vs.build_varlen_args(_ints((B * U, 4, D), _g(1)).to(EDT), e._k_gpu, e._v_gpu, e._kv_bias_gpu, rnd)   # a valid verify call
    # the commit: request 1 keeps nothing (the pilot's perreq_keep), request 0 two tokens
    keep = pilot.perreq_keep(torch.tensor([2, 3]), torch.zeros(2, dtype=torch.bool), ctx.round_bad)
    assert keep.tolist() == [2, 0]
    tr = e.spec_commit_perreq(t, torch.stack(ctr), keep, LAY.mirror_lo)
    assert tr.pos.tolist() == [pos[0] + 2, pos[1]] and t.pos.tolist() == tr.pos.tolist()
    # NEGATIVE CONTROL: the plan alone (no isolation) leaves request 1's stream claiming blocks nobody fetched
    st2 = sl.Store(1, H, B, LAY)
    for n in sl.STORE_FIELDS + st.COUNTERS:
        getattr(st2, n)[0].copy_(before[n])
    p2 = sl.plan_round_perreq(st2, 0, sel, rnd.tail.tail_id, rnd.tail.rollover_at, 1, 2, 3, torch.zeros((H, B, LAY.W), dtype=torch.bool))
    claimed = (st2.state[0, :, 1] == sl.RESIDENT) & (st2.ids[0, :, 1] != before["ids"][:, 1])
    assert bool(claimed.any()) and bool((p2.fetch[:, 1] == -1).all())


def test_EP3_finding_lockstep_round_overflow_leaves_unfetched_blocks_resident():
    """DOCUMENTS a defect of the LOCKSTEP pilot (unchanged here; its fallback path): plan_round marks the slots it
    assigned RESIDENT before spec_round_update raises RoundOverflow, and the caller's fallback restores only the
    counters and tables, so the U = 1 re-verify plans on a store that names blocks whose rows were never gathered."""
    pos = [30 * BS + 5]
    n_max = pos[0] + 2 * BS
    u = Uni(pos, [1], 41, n_max, perreq=False)
    e, st, ctx = u.engines[0], u.store, u.ctx
    T = pos[0] // BS
    ctx.tail_id, ctx.tail0, ctx.rolled, ctx.gen_round0 = T, sl.capture_tail_counters(e, T), [False], ctx.gen + 1
    ctx.was_inflight = torch.zeros_like(st.state, dtype=torch.bool)
    U = 4
    big = (torch.arange(32, dtype=torch.int64).view(U, 1, 1, 8) % 30).expand(U, H, 1, 8).contiguous()
    g = _g(41, 2)
    before_ids, before_state = st.ids.clone(), st.state.clone()
    with pytest.raises(vs.RoundOverflow):
        e.spec_round_update(_ints((1, U, H, D), g), _ints((1, U, H, D), g), u.twins[0].total_cis, big, ctx, 0)
    claimed = ((st.state == sl.RESIDENT) & (st.ids != before_ids))[0, :, 0]                 # (H, W)
    assert bool(claimed.any())
    h, w = [int(x) for x in claimed.nonzero()[0].tolist()]
    blk = int(st.ids[0, h, 0, w])
    assert not torch.equal(e._k_gpu[0, w * BS:(w + 1) * BS, h], e._k_cpu[0, blk * BS:(blk + 1) * BS, h])   # stale rows


def test_EP4_the_lockstep_and_per_request_bodies_refuse_each_others_engines():
    pos = [9 * BS + 10] * 2
    n_max = pos[0] + 2 * BS
    P = Uni(pos, [0, 1], 51, n_max, perreq=True)
    L = Uni(pos, [0, 1], 51, n_max, perreq=False)
    k, v = _ints((2, 1, H, D), _g(5)), _ints((2, 1, H, D), _g(6))
    sel = torch.full((H, 2, KSEL), -1, dtype=torch.int64)
    P.ctx.begin_round(P.view)
    lock_ctx = L.ctx
    lock_ctx.tail_id, lock_ctx.rolled, lock_ctx.gen_round0 = 9, [False], 1
    before = P.engines[0]._k_gpu.clone()
    with pytest.raises(TypeError):                         # the lockstep body: seq_length is None on a per-request engine
        P.engines[0].spec_draft_update(k, v, P.twins[0].total_cis, sel, lock_ctx, 0)
    assert torch.equal(P.engines[0]._k_gpu, before)
    with pytest.raises(AssertionError):                    # the per-request body on a lockstep engine
        L.engines[0].spec_draft_update_perreq(k, v, L.twins[0].total_cis, sel, P.ctx, 0)
    with pytest.raises(AssertionError):
        L.engines[0].spec_round_update_perreq(k, v, L.twins[0].total_cis, sel.unsqueeze(0), P.ctx, 0)
    with pytest.raises(AssertionError):                    # a commit without a round
        P.engines[0].spec_commit_perreq(P.twins[0], torch.zeros((2, 4, 2), dtype=torch.int64), torch.ones(2, dtype=torch.int64), LAY.mirror_lo)


# ===========================================================================
# F4: findings F4 parity (docs/R2_ENGINE_FINDINGS.md F4; run p2: every above-tolerance identity mismatch was at
# p % 64 == 63). The shipped decode's attended rows are written down from the shipped lines, and the per-request
# draft and verify are checked against them, position by position.
# ===========================================================================
MASKV = torch.tensor(sl.MASK_BIAS, dtype=EDT)


def _shipped_tail_rows(p):
    """Rows of the tail slot the SHIPPED decode attends at the step that writes position p: the cache_engine.py
    lines that set _cache_lens (pinned by tests/test_r2_engine_findings.py), mirrored."""
    tail_len = p % BS                                   # _tail_block_len_on_gpu before the step
    cache_lens = (TOPK - 1) * BS + tail_len             # :327 / the previous step's value
    cache_lens += 1                                     # :346
    tail_len += 1                                       # :357
    if tail_len == BS:                                  # :359 tail_full -> section D
        tail_len = 0                                    # :408
        cache_lens = (TOPK - 1) * BS + tail_len         # :409, returned at :412: findings F4
    return cache_lens - (TOPK - 1) * BS


def _shipped_reads(p, sel_row):
    """Absolute positions the SHIPPED decode attends at the step that writes p, for one (KV head, request): slots
    0 .. topk-2 hold the selection's blocks other than the tail block p // 64 (all rows: diff_offload maps them there),
    slot topk-1 holds the tail block, of which rows [0, _shipped_tail_rows(p)) are below cache_seqlens."""
    T = p // BS
    out = {x * BS + r for x in sel_row if x >= 0 and x != T for r in range(BS)}
    return out | {T * BS + r for r in range(_shipped_tail_rows(p))}


def _f4_sel(g, pos0, n):
    """(n, H, B, KSEL) selections for positions pos0_b + w, as the model's forced local window makes them: the live
    tail block p // 64, the round-start block T_b when p is already in T_b + 1, and 1 to 3 blocks of a per-(h, b) set of
    3 host blocks < T_b; -1 padded. The union of a round fits the store (no overflow)."""
    B = len(pos0)
    sel = torch.full((n, H, B, KSEL), -1, dtype=torch.int64)
    for b, p0 in enumerate(pos0):
        T = p0 // BS
        for h in range(H):
            base = torch.randperm(T, generator=g)[:3].tolist()
            for w in range(n):
                p = p0 + w
                ids = [p // BS] + ([T] if p // BS > T else []) + base[:1 + int(torch.randint(0, 3, (1,), generator=g))]
                sel[w, h, b, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
    return sel


def _f4_round(f4, pos, K, seed):
    """One per-request round (K draft steps, the restore, the verify of U = K + 1 positions; no commit) on one layer
    with the _f4_sel selections, in F4 mode ``f4``. Returns (universe, inputs, record, verify rounds)."""
    B = len(pos)
    n_max = max(pos) + 2 * (K + 1) + 2 * BS
    P = Uni(pos, list(range(B)), seed, n_max, perreq=True, f4=f4)
    assert P.ctx.f4 == f4
    inp = _inputs(seed, 0, list(range(B)), P.pos(), K, [K] * B)
    g = _g(seed, 77)
    sd = _f4_sel(g, pos, K)
    inp.sd = [[sd[j]] for j in range(K)]
    inp.sv = [_f4_sel(g, pos, K + 1)]
    rec = dict(draft=[], round=[], tails=[], tr=[])
    _draft_and_restore_perreq(P, inp, rec)
    rnds = _verify_perreq(P, inp, rec)
    return P, inp, rec, rnds


def _verify_reads(P, rnd, b, w, h):
    """Absolute positions the Path-1 call reads for request b, position w, KV head h: build_varlen_args' mask row
    b*U + w and seqused_k; the rows of every named slot below the causal limit seqused_k[b] - U + w + 1 (the kernel's
    rule, verify_step.py docstring); a store slot holds block store.ids (RESIDENT), mirror_lo T_b, mirror_hi T_b + 1."""
    e, U = P.engines[0], rnd.U
    a = vs.build_varlen_args(torch.zeros((P.B * U, 4, D), dtype=EDT), e._k_gpu, e._v_gpu, e._kv_bias_gpu, rnd)
    T = int(rnd.tail.tail_id[b])
    lim = int(a.seqused_k[b]) - U + w + 1
    out = set()
    for s in a.topk_idx[h, b * U + w].tolist():
        if s < 0:
            continue
        if s == LAY.mirror_lo:
            blk = T
        elif s == LAY.mirror_hi:
            blk = T + 1
        else:
            assert int(P.store.state[0, h, b, s]) == sl.RESIDENT
            blk = int(P.store.ids[0, h, b, s])
        out |= {blk * BS + r for r in range(BS) if s * BS + r < lim}
    return out


F4_CASES = (([9 * BS + 63, 10 * BS + 61, 9 * BS + 60, 11 * BS + 10, 10 * BS + 0], 3),   # F4 at u = 0 (then T + 1), 2, 3 (last); none; none
            ([9 * BS + 63, 10 * BS + 5], 0))                                            # the K = 0 fallback round (U = 1)


def test_F4V1_the_verify_reads_what_the_shipped_decode_reads_at_every_position_and_the_fixed_mode_does_not():
    """For every (request, position, KV head) of a round with F4 positions at u = 0, 2 and 3 (and of a U = 1 round),
    the positions the per-request verify reads equal the positions the shipped decode attends at that step
    (_shipped_reads, from the shipped lines: at p % 64 == 63 no row of the tail block). NEGATIVE CONTROL, f4 = "fixed":
    each F4 position reads exactly the tail block more (positions [64 T_b, p]), every other position reads the same;
    the two runs differ only in the mirror entries of the F4 rows of the mask (store, engine, twin, fetch, tail equal)."""
    for pos, K in F4_CASES:
        U = K + 1
        runs = {m: _f4_round(m, pos, K, 131) for m in sl.F4_MODES}
        f4_pos = [(b, w) for b in range(len(pos)) for w in range(U) if (pos[b] + w) % BS == BS - 1]
        assert len(f4_pos) == (3 if K == 3 else 1)
        for m, (P, inp, rec, rnds) in runs.items():
            for b, p0 in enumerate(pos):
                for w in range(U):
                    p = p0 + w
                    for h in range(H):
                        got = _verify_reads(P, rnds[0], b, w, h)
                        want = _shipped_reads(p, inp.sv[0][w, h, b].tolist())
                        if m == sl.F4_SHIPPED or (b, w) not in f4_pos:
                            assert got == want, (m, b, w, h, sorted(got ^ want)[:8])
                        else:
                            assert want <= got and got - want == set(range((p // BS) * BS, p + 1)), (m, b, w, h)
        (Ps, _, rs, _), (Pf, _, rf, _) = runs[sl.F4_SHIPPED], runs[sl.F4_FIXED]
        for b in range(len(pos)):
            assert _first_diff(_state(Pf, b), _state(Ps, b)) is None, b
        ms, mf = rs["round"][0].mask, rf["round"][0].mask
        for f in ROUND_FIELDS:
            assert torch.equal(getattr(rs["round"][0], f), getattr(rf["round"][0], f)), f
        want_mask = mf.clone()
        for b, w in f4_pos:
            row = want_mask[:, b * U + w]
            row[(row == LAY.mirror_lo) | (row == LAY.mirror_hi)] = -1
        assert torch.equal(ms, want_mask) and not torch.equal(ms, mf) and ms.is_contiguous() and ms.dtype == torch.int32


def test_F4D1_the_draft_masks_the_whole_tail_block_at_an_F4_step_and_the_fixed_mode_does_not():
    """The DRAFT steps of the same round (position pos0_b + j; F4 at request 0 step 0 and request 1 step 2). The tail
    rows the draft's bias leaves visible (slot topk-1: block T_b; mirror_hi: T_b + 1) equal the tail-block positions
    the shipped decode attends at that step: none at an F4 step with f4 = "shipped". NEGATIVE CONTROL, f4 = "fixed":
    at an F4 step the draft sees all 64 rows of T_b. The two runs' biases differ only there; every plan field is equal."""
    pos, K = F4_CASES[0]
    runs = {m: _f4_round(m, pos, K, 132) for m in sl.F4_MODES}
    f4_steps = [(b, j) for b in range(len(pos)) for j in range(K) if (pos[b] + j) % BS == BS - 1]
    assert f4_steps == [(0, 0), (1, 2)]
    ts, hs = LAY.tail_slot * BS, LAY.mirror_hi * BS
    tail_rows = torch.zeros(LAY.W * BS, dtype=torch.bool)
    tail_rows[ts:ts + BS] = True
    tail_rows[hs:hs + BS] = True
    for m, (P, inp, rec, _) in runs.items():
        for j in range(K):
            bias = rec["draft"][j]["bias"]
            for b, p0 in enumerate(pos):
                T0, p = p0 // BS, p0 + j
                for h in range(H):
                    hidden = bias[b, :, h] == MASKV
                    got = {T0 * BS + r for r in range(BS) if not bool(hidden[ts + r])}
                    got |= {(T0 + 1) * BS + r for r in range(BS) if not bool(hidden[hs + r])}
                    want = {x for x in _shipped_reads(p, inp.sd[j][0][h, b].tolist()) if x >= T0 * BS}
                    if m == sl.F4_SHIPPED or (b, j) not in f4_steps:
                        assert got == want, (m, b, j, h, sorted(got ^ want)[:8])
                    else:
                        assert want == set() and got == set(range(T0 * BS, (T0 + 1) * BS)), (m, b, j, h)
    rs, rf = runs[sl.F4_SHIPPED][2], runs[sl.F4_FIXED][2]
    for j in range(K):
        bs_, bf_ = rs["draft"][j]["bias"], rf["draft"][j]["bias"]
        assert torch.equal(bs_[:, ~tail_rows], bf_[:, ~tail_rows])
        for b in range(len(pos)):
            assert torch.equal(bs_[b], bf_[b]) == ((b, j) not in f4_steps), (b, j)
        for f in PLAN_FIELDS:
            assert torch.equal(getattr(rs["draft"][j]["plan"], f), getattr(rf["draft"][j]["plan"], f)), (j, f)


def test_F4R1_the_mode_is_checked_before_anything_changes_and_the_wiring_order():
    with pytest.raises(ValueError):
        sl.check_f4_mode("junk")
    with pytest.raises(ValueError):
        Uni([9 * BS + 10], [0], 141, 12 * BS, perreq=True, f4="junk")
    assert sl.F4_MODES == ("shipped", "fixed") and Uni([9 * BS + 10], [0], 141, 12 * BS, perreq=True).ctx.f4 == "shipped"
    pos = [9 * BS + 63, 10 * BS + 5]
    P = Uni(pos, [0, 1], 141, max(pos) + 4 * BS, perreq=True)
    e, t, ctx = P.engines[0], P.twins[0], P.ctx
    ctx.begin_round(P.view)
    ctx.was_inflight = torch.zeros_like(P.store.state, dtype=torch.bool)
    ctx.f4 = "junk"
    snap = {n: getattr(e, n).clone() for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_pos", "_block_map", "_cache_lens")}
    store0 = sl.store_layer_copy(P.store, 0)
    g = _g(141)
    with pytest.raises(ValueError):
        e.spec_draft_update_perreq(_ints((2, 1, H, D), g), _ints((2, 1, H, D), g), t.total_cis, _f4_sel(g, pos, 1)[0], ctx, 0)
    with pytest.raises(ValueError):
        e.spec_round_update_perreq(_ints((2, 2, H, D), g), _ints((2, 2, H, D), g), t.total_cis, _f4_sel(g, pos, 2), ctx, 0)
    for n, v in snap.items():
        assert torch.equal(getattr(e, n), v), n
    for n, v in store0.items():
        assert torch.equal(getattr(P.store, n)[0], v), n
    with pytest.raises(ValueError):                        # a mask and a flag table that do not agree
        sl.drop_tail_on_f4(torch.zeros((H, 6, KSEL), dtype=torch.int32), torch.zeros((2, 2), dtype=torch.bool), LAY)
    src_d, src_r = _method_src("spec_draft_update_perreq"), _method_src("spec_round_update_perreq")
    assert src_d.index("_sl.check_f4_mode(ctx.f4)") < src_d.index("_sl.draft_tail_write_perreq(") < src_d.index("f4=f4)")
    assert (src_r.index("_sl.check_f4_mode(ctx.f4)") < src_r.index("_sl.write_tail_perreq(") < src_r.index("_sl.plan_round_perreq(")
            < src_r.index("_sl.drop_tail_on_f4(") < src_r.index("_sl.isolate_bad_streams("))   # a bad stream keeps its placeholder


def test_F4P1_the_pilot_knob_defaults_to_shipped_parity_is_recorded_and_shown(tmp_path, monkeypatch):
    import json
    assert pilot.F4 == "shipped" and pilot.F4_MODES == sl.F4_MODES
    src = (ROOT / "benchmarks/Efficiency/spec_loop_pilot.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    arm = fns["run_spec_arm_perreq"]
    assert "f4=F4)" in arm and "commit=\"perreq\", f4=ctx.f4," in arm
    assert "f4=F4 if COMMIT == \"perreq\" else None" in fns["provenance"]
    assert "if F4 not in F4_MODES:" in src and "die(\"NOSI_SPEC_F4=%r" in src
    monkeypatch.setattr(pilot, "OUT", str(tmp_path))
    monkeypatch.setenv("NOSI_SPEC_TAU_CTRL", "0.4844")
    for f4, shown in (("shipped", "shipped"), (None, "-")):
        for f in tmp_path.iterdir():
            f.unlink()
        p, ship = _payload(torch.zeros((2, 6), dtype=torch.int64))
        p["cells"]["K2"]["seq"] = ship.clone()
        if f4 is not None:
            p["cells"]["K2"]["f4"] = f4
        torch.save(p, str(tmp_path / "spec_sperreqb2d2.pt"))
        assert pilot.report() == 0
        assert "| spec_sperreqb2d2/K2 | %s | 2 |" % shown in (tmp_path / "spec_loop.md").read_text()
        assert json.loads((tmp_path / "spec_loop.json").read_text())["perreq_cells"]["spec_sperreqb2d2/K2"]["f4"] == f4


# ===========================================================================
# VS: the verify call's per-request lengths
# ===========================================================================
def test_VS1_build_varlen_args_takes_per_request_lengths_and_refuses_a_wrong_one():
    pos = [9 * BS + 61, 10 * BS + 3]
    B, U = 2, 3
    n_max = max(pos) + 2 * BS
    P = Uni(pos, [0, 1], 61, n_max, perreq=True)
    e, t, ctx, st = P.engines[0], P.twins[0], P.ctx, P.store
    ctx.begin_round(P.view)
    ctx.was_inflight = torch.zeros_like(st.state, dtype=torch.bool)
    g = _g(61)
    k, v, c = _ints((B, U, H, D), g), _ints((B, U, H, D), g), _ints((B, U, H), g)
    for w in range(U):
        t.step(k[:, w], c[:, w])
    sel = torch.stack([torch.stack([_sel(g, (p + w) // BS) for p in pos], 1) for w in range(U)])
    _, _, _, rnd = e.spec_round_update_perreq(k, v, t.total_cis, sel, ctx, 0)
    want = [LAY.mirror_hi * BS + (pos[0] + U) % BS, LAY.mirror_lo * BS + (pos[1] + U) % BS]   # request 0 rolled over
    assert rnd.seqused_k.dtype == torch.int32 and rnd.seqused_k.tolist() == want
    q = _ints((B * U, 4, D), g).to(EDT)
    a = vs.build_varlen_args(q, e._k_gpu, e._v_gpu, e._kv_bias_gpu, rnd)
    assert a.seqused_k.tolist() == want
    with pytest.raises(ValueError):
        vs.build_varlen_args(q, e._k_gpu, e._v_gpu, e._kv_bias_gpu, rnd._replace(seqused_k=rnd.seqused_k + torch.tensor([0, 1], dtype=torch.int32)))
    bad_tail = rnd.tail._replace(seqused_k=torch.tensor([U - 1, want[1]]))
    with pytest.raises(ValueError):                        # a length below U, even when the call agrees with it
        vs.build_varlen_args(q, e._k_gpu, e._v_gpu, e._kv_bias_gpu, rnd._replace(tail=bad_tail, seqused_k=torch.tensor([U - 1, want[1]], dtype=torch.int32)))
    # the lockstep path: an int length, unchanged
    lock_tail = types.SimpleNamespace(seqused_k=want[1])
    ok = rnd._replace(tail=lock_tail, seqused_k=torch.full((B,), want[1], dtype=torch.int32))
    vs.build_varlen_args(q, e._k_gpu, e._v_gpu, e._kv_bias_gpu, ok)
    with pytest.raises(ValueError):
        vs.build_varlen_args(q, e._k_gpu, e._v_gpu, e._kv_bias_gpu, ok._replace(seqused_k=torch.tensor(want, dtype=torch.int32)))


# ===========================================================================
# MP: the model module on CPU
# ===========================================================================
def _shipped_ref_module():
    return _load(ROOT / "tests/test_r2_positional_twin.py", "nosi_r2_twin_suite_for_bridge")


def _fake_layer(ref):
    return types.SimpleNamespace(cached_compressed_max_seqlen=ref.ck_max, compress_k_cache_varlen=ref.ck,
                                 no_compress_k_len=ref.nck_len, no_compress_k_cache=ref.nck,
                                 compressed_cis=ref.compressed_cis, comp_cis_len=ref.comp_cis_len,
                                 total_cis=ref.total_cis, cis_len=ref.cis_len)


@pytest.mark.parametrize("S,steps", [(16 * 40, 1), (16 * 40 + 1, 1), (16 * 40 + 7, 1), (16 * 40, 3)])
def test_MP1_bridge_equals_the_twin_after_the_shipped_prefill_and_steps(S, steps):
    suite = _shipped_ref_module()
    B = 3
    g = _g(71, S, steps)
    keys, cis = _ints((B, S, H, D), g), _ints((B, S, H), g)
    ref = suite.ShippedRef(keys, cis, f2_as_shipped=True)                  # the shipped tables, the shipped F2 line
    want = twin_mod.PositionalTwin(B, H, D, lk_max=80, n_max=S + 64)
    for b in range(B):
        want.prefill(b, keys[b], cis[b])
    for s in range(steps):
        k1, c1 = _ints((B, H, D), g), _ints((B, H), g)
        ref.decode_step(k1, c1)
        want.step(k1, c1)
    got = twin_mod.PositionalTwin(B, H, D, lk_max=80, n_max=S + 64)
    spm.bridge_layer_at(got, _fake_layer(ref), S + steps)
    for b in range(B):
        suite._assert_request_equal(got, b, want, b, "bridge S=%d steps=%d request %d" % (S, steps, b))


def test_MP1b_bridge_refuses_an_F2_short_table_and_a_wrong_length_and_copies_nothing():
    suite = _shipped_ref_module()
    B, S = 2, 16 * 40 + 15                                   # the warm-up step completes a cis window: F2 drops it
    g = _g(72)
    keys, cis = _ints((B, S, H, D), g), _ints((B, S, H), g)
    ref = suite.ShippedRef(keys, cis, f2_as_shipped=True)
    ref.decode_step(_ints((B, H, D), g), _ints((B, H), g))
    got = twin_mod.PositionalTwin(B, H, D, lk_max=80, n_max=S + 64)
    with pytest.raises(spm.PerReqModelError):
        spm.bridge_layer_at(got, _fake_layer(ref), S + 1)
    with pytest.raises(spm.PerReqModelError):
        spm.bridge_layer_at(got, _fake_layer(ref), S + 2)
    assert int(got.pos.abs().sum()) == 0 and float(got.comp_k.abs().sum()) == 0.0 and not bool(got.prefilled.any())
    ok = suite.ShippedRef(keys, cis, f2_as_shipped=False)    # the intended table (the summary appended): accepted
    ok.decode_step(_ints((B, H, D), _g(73)), _ints((B, H), _g(73)))
    lay = _fake_layer(ok)
    lay.comp_cis_len = int(ok.compressed_cis.shape[-1])       # the counter of a table that grows with it
    spm.bridge_layer_at(got, lay, S + 1)
    assert got.n_ccis.tolist() == [spm.complete_windows(S + 1)] * B


def _stage1_stub(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None):
    """A torch stage 1 with the op's output contract: (Hkv, total_q, round128(max_seqlen_k)); request b's columns
    >= seqused_k[b] are zero (the op's zero fill)."""
    B = cu_seqlens_q.numel() - 1
    Hq, Hkv = q.shape[1], k.shape[1]
    grp = Hq // Hkv
    N = (max_seqlen_k + 127) // 128 * 128
    out = torch.zeros((Hkv, B, N), dtype=q.dtype)
    for b in range(B):
        kb = k[int(cu_seqlens_k[b]):int(cu_seqlens_k[b]) + max_seqlen_k]                 # (Lk, Hkv, D)
        for h in range(Hkv):
            s = (q[b, h * grp:(h + 1) * grp].unsqueeze(1) * kb[:, h].unsqueeze(0)).sum(-1).sum(0)   # (Lk,)
            n = int(seqused_k[b])
            out[h, b, :n] = s[:n]
    return out


def _pool_stub(q_score, cis_score, n_comp, M_b, q_out, c_out, stride, block_size, local_blocks, init_blocks):
    N = q_score.shape[-1]
    cis = torch.zeros_like(q_score)
    cis[..., :cis_score.shape[-1]] = cis_score
    a, b = mp.pool_reference(q_score, cis, n_comp, M_b, q_out.shape[-1], init_blocks=init_blocks, local_blocks=local_blocks)
    q_out.copy_(a)
    c_out.copy_(b)


def test_MP2_score_position_on_a_ragged_batch_equals_B1_scoring():
    pos = [9 * BS + 61, 10 * BS + 3, 9 * BS + 40]
    B, Hq = len(pos), 4
    n_max = max(pos) + 2 * BS
    twins = [_tables(pos, list(range(B)), 81, 0, n_max)]
    st = spm.PerReqState(twins, [None], 4, 6, n_max // BS + 2, torch.float32, "cpu")
    singles = [spm.PerReqState([_tables([pos[b]], [b], 81, 0, n_max)], [None], 4, 6, n_max // BS + 2, torch.float32, "cpu")
               for b in range(B)]
    g = _g(81)
    for step in range(3):
        q, k, c = _ints((B, Hq, D), g), _ints((B, H, D), g), _ints((B, H), g)
        before = twins[0].pos.clone()
        sel = spm.score_position(st, 0, q, k, c, 16, 32, 2, 1, stage1=_stage1_stub, pool=_pool_stub)
        assert torch.equal(twins[0].pos, before + 1)
        for b in range(B):
            s1 = spm.score_position(singles[b], 0, q[b:b + 1], k[b:b + 1], c[b:b + 1], 16, 32, 2, 1, stage1=_stage1_stub, pool=_pool_stub)
            assert torch.equal(sel[:, b].sort(-1).values, s1[:, 0].sort(-1).values), (step, b)
            T, Mb = int(before[b]) // BS, int(before[b]) // BS + 1
            ids = set(sel[:, b].flatten().tolist())
            assert T in ids and 0 in ids and max(ids) < Mb, (step, b, sorted(ids))   # the local window and block 0 forced
            suite_eq = [int(getattr(twins[0], n)[b]) == int(getattr(singles[b].twins[0], n)[0]) for n in ("pos", "n_comp", "n_ccis")]
            assert all(suite_eq)


def _probe_universe(seed):
    pos = [9 * BS + 61, 10 * BS + 3, 9 * BS + 63, 11 * BS + 0]
    B, K, NL = 4, 3, 2
    n_max = max(pos) + 4 * (K + 1) + 2 * BS
    return Uni(pos, list(range(B)), seed, n_max, perreq=True, NL=NL), pos, K


def test_MP3_the_per_request_rollback_probe_passes_after_a_round_and_its_controls_fail(monkeypatch):
    u, pos, K = _probe_universe(91)
    U = K + 1
    nacc = [0, 3, 1, 2]                         # keep 1 (undo, 1 row restored), 4, 2 (kept fill), 3
    snap = spm.ProbeSnapshot(u.view, u.state, LAY, U).take()
    rec = _round_perreq(u, _inputs(91, 0, list(range(4)), u.pos(), K, nacc, u.NL))
    keep = rec["keep"]
    assert keep.tolist() == [1, 4, 2, 3] and bool(rec["tr"][0]["undo"][0]) and int(rec["tr"][0]["restored"][0]) == 1
    pr = spm.probe(u.view, u.state, snap, keep, U, LAY)
    assert pr["passed"] and pr["same"] and pr["control_differs"] and pr["replay_deterministic"], pr
    assert pr["control_keep"][0] == 2
    # CONTROL 1: a corrupted committed tail row is caught
    u2, _, _ = _probe_universe(91)
    snap2 = spm.ProbeSnapshot(u2.view, u2.state, LAY, U).take()
    rec2 = _round_perreq(u2, _inputs(91, 0, list(range(4)), u2.pos(), K, nacc, u2.NL))
    u2.engines[0]._k_gpu[1, LAY.tail_slot * BS + 2] += 1                            # request 1 holds rows [0, 7) of its tail
    pr2 = spm.probe(u2.view, u2.state, snap2, rec2["keep"], U, LAY)
    assert not pr2["same"] and pr2["detail"] == "layer 0 request 1 k_tail differs", pr2
    # CONTROL 2: a round without the masked restore is caught at the request whose fill it undid
    u3, _, _ = _probe_universe(91)
    snap3 = spm.ProbeSnapshot(u3.view, u3.state, LAY, U).take()
    monkeypatch.setattr(sl, "_masked_tail_restore", lambda *a, **k: None)
    rec3 = _round_perreq(u3, _inputs(91, 0, list(range(4)), u3.pos(), K, nacc, u3.NL))
    monkeypatch.undo()
    pr3 = spm.probe(u3.view, u3.state, snap3, rec3["keep"], U, LAY)
    assert not pr3["same"] and pr3["detail"].startswith("layer 0 request 0 ") and "_tail" in pr3["detail"], pr3
    assert pr3["replay_deterministic"], pr3                 # the replay itself is repeatable: 'same' decides


def test_MP3b_replay_deterministic_compares_the_two_replays_and_catches_a_non_repeatable_replay(monkeypatch):
    """replay_deterministic is the FIRST replay(keep) against the THIRD (a repeat), not the post-round state against
    the third. Negative control: a replay whose tail write is perturbed only on its third pass fails it while 'same'
    (post-round state == first replay) still holds."""
    u, pos, K = _probe_universe(92)
    U = K + 1
    snap = spm.ProbeSnapshot(u.view, u.state, LAY, U).take()
    rec = _round_perreq(u, _inputs(92, 0, list(range(4)), u.pos(), K, [0, 3, 1, 2], u.NL))
    keep = rec["keep"]
    n_writes = u.NL * int((keep > 0).sum())                  # write_tail calls of one replay (the control's too: keep 2 at request 0)
    orig = spm._tw.write_tail
    calls = [0]

    def write_tail(engine, key_states, value_states, kv_bias, mirror_slot):
        calls[0] += 1
        if calls[0] > 2 * n_writes:                            # the third replay
            key_states = key_states + 1
        return orig(engine, key_states, value_states, kv_bias, mirror_slot)

    monkeypatch.setattr(spm._tw, "write_tail", write_tail)
    pr = spm.probe(u.view, u.state, snap, keep, U, LAY)
    assert calls[0] == 3 * n_writes
    assert pr["same"] and pr["control_differs"] and not pr["replay_deterministic"] and not pr["passed"], pr
    assert pr["deterministic_detail"] != "equal"


# ===========================================================================
# PL: the pilot's host side
# ===========================================================================
def test_PL1_perreq_keep_and_the_loop_bookkeeping():
    assert pilot.perreq_keep(torch.tensor([3, 1, 2]), torch.tensor([False, True, False]), torch.tensor([False, False, True])).tolist() == [3, 0, 0]
    t0, t1 = torch.tensor([[10], [20], [30]]), torch.tensor([[11], [21], [31]])
    lp = pilot.PerReqLoop(t0, t1, total=5, K=2)
    assert lp.next_K() == 2 and not lp.finished()
    com = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lp.after_round(2, torch.tensor([3, 1, 2]), torch.zeros(3, dtype=torch.bool), com)
    assert lp.n_comm.tolist() == [4, 2, 3] and lp.seqs[0] == [10, 11, 1, 2, 3] and lp.seqs[1] == [20, 21, 4]
    bad = torch.tensor([False, True, False])
    keep = pilot.perreq_keep(torch.tensor([3, 2, 1]), lp.done(), bad)
    lp.after_round(2, keep, bad, com)
    assert keep.tolist() == [3, 0, 1] and lp.n_comm.tolist() == [7, 2, 4] and lp.done().tolist() == [True, False, False]
    assert lp.next_K() == 0                                               # the fallback round after an overflow
    keep = pilot.perreq_keep(torch.tensor([1, 1, 1]), lp.done(), torch.zeros(3, dtype=torch.bool))
    lp.after_round(0, keep, torch.zeros(3, dtype=torch.bool), com[:, :1])
    assert keep.tolist() == [0, 1, 1] and lp.next_K() == 2 and lp.n_comm.tolist() == [7, 3, 5]
    keep = pilot.perreq_keep(torch.tensor([3, 3, 3]), lp.done(), torch.zeros(3, dtype=torch.bool))
    lp.after_round(2, keep, torch.zeros(3, dtype=torch.bool), com)
    assert keep.tolist() == [0, 3, 0] and lp.finished() and lp.lengths() == [8, 7, 6]
    assert lp.trimmed(6).tolist()[2] == [30, 31, 7, 8, 7, 7] and lp.padded().shape == (3, 8) and int(lp.padded()[2, 7]) == -1
    with pytest.raises(ValueError):
        lp.trimmed(7)
    # refusals
    lp2 = pilot.PerReqLoop(t0, t1, total=3, K=1, max_rounds=2)
    with pytest.raises(RuntimeError):                                     # an overflowing request that commits
        lp2.after_round(1, torch.tensor([1, 1, 1]), torch.tensor([True, False, False]), com)
    with pytest.raises(RuntimeError):                                     # a K = 0 round that overflows
        lp2.after_round(0, torch.tensor([0, 1, 1]), torch.tensor([True, False, False]), com[:, :1])
    with pytest.raises(ValueError):                                       # more tokens than the round has
        lp2.after_round(1, torch.tensor([3, 1, 1]), torch.zeros(3, dtype=torch.bool), com[:, :2])
    lp2.after_round(1, torch.tensor([2, 2, 2]), torch.zeros(3, dtype=torch.bool), com)
    with pytest.raises(RuntimeError):                                     # a finished request that commits
        lp2.after_round(1, torch.tensor([1, 0, 0]), torch.zeros(3, dtype=torch.bool), com)
    lp2.after_round(1, torch.tensor([0, 0, 0]), torch.zeros(3, dtype=torch.bool), com)
    with pytest.raises(RuntimeError):                                     # the progress guard
        lp2.next_K()


def _round(K, keep, nacc, bad, done, vms=10.0):
    return dict(K=K, keep=keep, n_acc=nacc, bad=bad, done_before=done, draft_ms=[1.0] * K, verify_ms=vms, rollback_ms=1.0,
                verify_split={"total_ms": vms}, draft_split=[{"total_ms": 1.0}] * K)


def test_PL2_aggregate_rounds_perreq_measures_the_yield():
    F, T = False, True
    rounds = [_round(2, [3, 1, 2], [2, 0, 1], [F, F, F], [F, F, F], vms=100.0),     # timing: excluded (first spec round)
              _round(2, [2, 0, 3], [1, 2, 2], [F, T, F], [F, F, F], vms=10.0),
              _round(0, [1, 1, 1], [0, 0, 0], [F, F, F], [F, F, F], vms=5.0),       # the fallback round
              _round(2, [0, 3, 1], [2, 2, 0], [F, F, F], [T, F, F], vms=20.0)]      # request 0 finished
    ag = pilot.aggregate_rounds_perreq(rounds)
    assert (ag["rounds"], ag["spec_rounds"], ag["fallback_rounds"], ag["overflow_rounds"], ag["overflow_request_rounds"]) == (4, 3, 1, 1, 1)
    assert ag["active_request_rounds"] == 11 and ag["committed_tokens"] == 3 + 1 + 2 + 2 + 0 + 3 + 1 + 1 + 1 + 3 + 1
    assert ag["yield_measured_per_round"] == pytest.approx(18 / 11)
    assert ag["yield_measured_spec_rounds"] == pytest.approx((6 + 5 + 4) / 8)
    assert ag["n_acc_mean_per_request"] == pytest.approx((2 + 0 + 1 + 1 + 2 + 2 + 0) / 7)
    assert ag["yield_lockstep_equivalent_per_round"] == pytest.approx(((1 + 0) + (1 + 1) + (1 + 0)) / 3)
    assert ag["verify_ms"] == pytest.approx(15.0) and ag["draft_step_ms"] == pytest.approx(1.0)


def _payload(seq_spec, overflow_rounds=0, probe_passed=True):
    B, n = seq_spec.shape
    ship = torch.arange(B * n).view(B, n) % 97
    margins = torch.full((B, n - 1), 5.0)
    margins[1, 2] = 0.1                                       # a near tie at request 1, position 3
    agg = dict(rounds=4, spec_rounds=3, fallback_rounds=overflow_rounds, overflow_rounds=overflow_rounds,
               overflow_request_rounds=overflow_rounds, yield_measured_per_round=1.8, yield_measured_spec_rounds=2.0,
               yield_lockstep_equivalent_per_round=1.2, n_acc_mean_per_request=1.0, verify_split={}, draft_split={})
    probe = dict(round=3, keep=[1, 2], U=3, same=probe_passed, detail="equal", control_keep=[2, 2], control_differs=True,
                 control_detail="layer 0 request 0 pos 1 vs 2", replay_deterministic=True, passed=probe_passed)
    cell = dict(commit="perreq", K=2, U=3, W=128, R=62, tokens=n - 1, tokens_per_request=[n - 1] * B, seq=seq_spec, probe=probe,
                agg=agg, counters={}, wire=dict(wire_bytes=10 ** 6, late_bytes=10 ** 5, prefetch_bytes=9 * 10 ** 5, wasted_bytes=0),
                timely_recall=0.5, host_syncs=7, committed_tokens_after_warmup=10, tok_s_after_warmup=100.0, wall_s_after_warmup=0.1,
                peak_gb=1.0, reserved_gb=1.0)
    arm = dict(seq=ship, margins=margins, tokens=n - 1, tok_s_after_warmup=50.0, peak_gb=1.0, reserved_gb=1.0)
    return dict(L=16128, batch=B, distinct_docs=2, docs=[1, 2], arms=dict(shipped=arm), cells=dict(K2=cell),
                engine_knobs=dict(NOSI_POOL_BLOCKS=0, NOSI_ATTN_SPLITS=4), host_pinned_gb=1.0, schedule="s",
                overlap_evidence={}, custom_kernels="none", commit="perreq"), ship


def test_PL3_the_report_gates_a_per_request_cell(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(pilot, "OUT", str(tmp_path))
    monkeypatch.setenv("NOSI_SPEC_TAU_CTRL", "0.4844")
    p, ship = _payload(torch.zeros((2, 6), dtype=torch.int64))
    p["cells"]["K2"]["seq"] = ship.clone()
    torch.save(p, str(tmp_path / "spec_sperreqb2d2.pt"))
    assert pilot.report() == 0
    js = json.loads((tmp_path / "spec_loop.json").read_text())
    g = js["gates"]
    assert g["identity[spec_sperreqb2d2/K2 vs spec_sperreqb2d2]"]["passed"] and g["rollback-probe[spec_sperreqb2d2/K2]"]["passed"]
    assert g["overflow[spec_sperreqb2d2/K2]"]["passed"] and "spec_sperreqb2d2/K2" in js["perreq_cells"] and js["cells"] == {}
    assert "per-request cells" in (tmp_path / "spec_loop.md").read_text()
    # a near-tie mismatch, an overflow, a failed probe: three failed gates, the near tie recorded as the expected failure
    for f in tmp_path.iterdir():
        f.unlink()
    p, ship = _payload(torch.zeros((2, 6), dtype=torch.int64), overflow_rounds=2, probe_passed=False)
    seq = ship.clone()
    seq[1, 3] += 1
    p["cells"]["K2"]["seq"] = seq
    torch.save(p, str(tmp_path / "spec_sperreqb2d2.pt"))
    assert pilot.report() == 3
    g = json.loads((tmp_path / "spec_loop.json").read_text())["gates"]
    idg = g["identity[spec_sperreqb2d2/K2 vs spec_sperreqb2d2]"]
    assert not idg["passed"] and "near-tie 1, above tol 0" in idg["detail"] and "expected until the bit-exact verifier lands" in idg["detail"]
    assert not g["overflow[spec_sperreqb2d2/K2]"]["passed"] and "2 K = 0 fallback rounds" in g["overflow[spec_sperreqb2d2/K2]"]["detail"]
    assert not g["rollback-probe[spec_sperreqb2d2/K2]"]["passed"] and "per request" in g["rollback-probe[spec_sperreqb2d2/K2]"]["detail"]
    # a mismatch above the tolerance is NOT the expected failure
    cl = pilot.classify_mismatches(dict(mismatches=[dict(ref_margin=9.0)]), 1.9)
    assert "NOT the expected" in pilot.perreq_identity_note(dict(passed=False), cl)


# ===========================================================================
# AN: the payload analysis (benchmarks/Efficiency/r2_payload_analysis.py) on a synthetic payload
# ===========================================================================
def _analysis_payload():
    """B = 4, documents [7, 8, 7, 8], L = 186, 10 tokens, one per-request cell (K = 2, three rounds of keep 3).
    Request 1 leaves the shipped sequence at j = 3 (margin 0.1: a near tie); request 2 at j = 5 (margin 3.0) while its
    partner 0, with the same prefix, keeps the shipped token; request 3 at j = 6 (margin 4.0) at position 191
    (191 % 64 == 63: an F4 step), after its group disagreed on n_acc in round 0."""
    L, B, n = 186, 4, 10
    ship = torch.zeros((B, n), dtype=torch.int64)
    ship[0] = ship[2] = torch.arange(1, n + 1)
    ship[1] = ship[3] = torch.arange(11, n + 11)
    margins = torch.full((B, n - 1), 5.0)
    margins[1, 2], margins[2, 4], margins[3, 5] = 0.1, 3.0, 4.0
    seq = ship.clone()
    seq[1, 3:] += 100
    seq[2, 5:] += 100
    seq[3, 6:] += 100
    F = False
    rounds = [dict(K=2, committed_before=[1, 1, 1, 1], keep=[3, 3, 3, 3], n_acc=[2, 1, 2, 2], bad=[F] * 4, done_before=[F] * 4),
              dict(K=2, committed_before=[4, 4, 4, 4], keep=[3, 3, 3, 3], n_acc=[2, 2, 1, 2], bad=[F] * 4, done_before=[F] * 4),
              dict(K=2, committed_before=[7, 7, 7, 7], keep=[3, 3, 3, 3], n_acc=[2, 2, 2, 2], bad=[F] * 4, done_before=[F] * 4)]
    probe = dict(passed=False, same=False, replay_deterministic=True, detail="layer 0 request 2 k_tail differs")
    cell = dict(commit="perreq", K=2, seq=seq, rounds=rounds, probe=probe)
    return dict(L=L, batch=B, distinct_docs=2, docs=[7, 8, 7, 8], arms=dict(shipped=dict(seq=ship, margins=margins)),
                cells=dict(K2=cell))


def test_AN1_the_payload_analysis_reports_first_mismatches_groups_and_the_above_tolerance_facts(tmp_path):
    ra = _load(ROOT / "benchmarks/Efficiency/r2_payload_analysis.py", "nosi_r2_payload_analysis_under_test")
    res = ra.analyse({"run/spec_x": _analysis_payload()}, 0.5)          # tol 2.0
    c = res["cells"][0]
    assert c["classes"] == {"near_tie": 1, "above_tol": 2} and c["mismatching_requests"] == 3 and c["f4_first_mismatches"] == 1
    assert c["spec_groups_with_differing_members"] == 2 and c["ref_groups_with_differing_members"] == 0
    assert c["nacc_disagreement_rounds"] == 2
    above = {r["request"]: r for r in c["above_tol"]}
    r2, r3 = above[2], above[3]
    assert (r2["j"], r2["position"], r2["f4_step"], r2["round"], r2["u"], r2["bonus"]) == (5, 190, False, 1, 0, False)
    assert r2["state_left_partners_before_j"] is False and r2["partners_same_prefix_and_shipped_token_at_j"] == [0]
    assert (r3["j"], r3["position"], r3["f4_step"], r3["round"], r3["u"]) == (6, 191, True, 1, 1)
    assert r3["state_left_partners_before_j"] is True and r3["first_partner_disagreement_round"] == 0
    assert r3["partners_same_prefix_and_shipped_token_at_j"] == []
    assert (r3["round_start_pos"], r3["tail_len0"], r3["rolled"], r3["u_after_fill"]) == (190, 62, True, False)
    # the CLI: payloads from a run directory, analysis.json / .md written, a fresh output directory required
    run = tmp_path / "r2m1_x_1"
    run.mkdir()
    torch.save(_analysis_payload(), str(run / "spec_x.pt"))
    out = tmp_path / "out"
    assert ra.main(["--run", str(run), "--tau", "0.5", "--out", str(out)]) == 0
    md = (out / "analysis.md").read_text()
    assert "ABOVE TOL r2m1_x_1/spec_x/K2 request 3: j=6 position 191 (F4 step True)" in md
    with pytest.raises(FileExistsError):
        ra.main(["--run", str(run), "--tau", "0.5", "--out", str(out)])


# ===========================================================================
# ST / PIN: the step script and the wiring
# ===========================================================================
def test_ST1_the_step_script_has_the_per_request_cells():
    src = STEP.read_text()
    assert subprocess.run(["bash", "-n", str(STEP)]).returncode == 0
    assert "spec_perreq) MODE=spec R=$R_SLOTS COMMIT=perreq STEP_MIN=120 TAG=sperreqb${B}d${D} ;;" in src
    assert "probe_perreq) MODE=spec R=$R_SLOTS COMMIT=perreq PROBE=1 KL=2 STEP_MIN=60 TAG=pperreqb${B}d${D} ;;" in src
    for old in ("ref) MODE=decode_ref R=0 STEP_MIN=40 TAG=refb${B}d${D} ;;", "spec) MODE=spec R=$R_SLOTS STEP_MIN=90 TAG=specb${B}d${D} ;;",
                "probe) MODE=spec R=$R_SLOTS PROBE=1 KL=2 STEP_MIN=60 TAG=probeb${B}d${D} ;;"):
        assert old in src, old                             # the milestone-1 cells keep their tags (TAG=${kind}b${B}d${D} before)
    assert "--env NOSI_SPEC_COMMIT=$COMMIT)" in src and "COMMIT=lockstep" in src
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-perreq " in src


def _r2_holder_uuid(d, job):
    src = STEP.read_text()
    fn = src[src.index("read_holder_uuid() {"):src.index("\nmain() {")]
    p = subprocess.run(["bash", "-c", fn + '\nread_holder_uuid "$1" "$2"', "t", str(d), job], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr


def test_ST2_the_step_script_reads_the_holder_uuid_file_and_stops_without_it(tmp_path):
    """The R2 step reads the holder's GPU uuid from HOLDERS/<job>.gpu_uuid (the eqmem v2 pattern); no uuid is pinned in
    the script; the child keeps its nvidia-smi check. NEGATIVE CONTROLS: missing, empty, GPU- prefixed, short and
    uppercase files STOP."""
    src = STEP.read_text()
    assert "HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders " in src and 'GPU_UUID=$(read_holder_uuid "$HOLDERS" "$JOB") || exit 1' in src
    assert "f31fc723" not in src and '[ "$seen" = "GPU-$U" ]' in src and '"$GPU_UUID" "$APPTAINER"' in src
    assert src.index('GPU_UUID=$(read_holder_uuid') < src.index('if [ "$DRY" = 1 ]')
    u = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "123.gpu_uuid").write_text(u + "\n")
    assert _r2_holder_uuid(tmp_path, "123") == (0, u, "")
    rc, out, err = _r2_holder_uuid(tmp_path, "999")
    assert rc == 1 and out == "" and "STOP: no" in err
    for job, text in (("125", ""), ("126", "GPU-" + u + "\n"), ("127", "f31fc723-9dfe\n"), ("128", u.upper() + "\n")):
        (tmp_path / ("%s.gpu_uuid" % job)).write_text(text)
        rc, out, err = _r2_holder_uuid(tmp_path, job)
        assert rc == 1 and out == "" and "STOP" in err, (job, rc, out, err)


def _bodies(src, cls_name):
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return {n.name: ast.get_source_segment(src, n) for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_PIN1_the_engine_model_and_module_wiring():
    e = SRC_ENGINE
    order = [e.index("    def spec_rollback_tail("), e.index("    def spec_draft_update_perreq("), e.index("    def spec_round_update_perreq("),
             e.index("    def spec_commit_perreq("), e.index("    def decode_update_has_kv_bias(self,")]
    assert order == sorted(order)
    for name, needs in (("spec_draft_update_perreq", ("_sl.draft_tail_write_perreq(", "_sl.plan_draft_perreq(", "_sl.apply_draft_bias_perreq(")),
                        ("spec_round_update_perreq", ("_sl.write_tail_perreq(", "_sl.plan_round_perreq(", "_sl.isolate_bad_streams(", "ctx.round_bad |= bad.any(0)")),
                        ("spec_commit_perreq", ("_sl.commit_layer_perreq(",))):
        src = _method_src(name)
        assert "diff.diff_offload" not in src and "raise _vs.RoundOverflow" not in src
        for s in needs:
            assert s in src, (name, s)
    lb = _bodies(SRC_LLAMA, "Llama")
    for name, call in (("draft_inference_perreq", "_spm.draft_inference("), ("spec_verify_inference_perreq", "_spm.spec_verify_inference("),
                       ("spec_rollback_perreq", "_spm.spec_rollback(")):
        assert "from . import spec_perreq_model as _spm" in lb[name] and call in lb[name], name
    rb = lb["spec_rollback"]                               # the lockstep rollback is the pinned one
    assert rb.count("spec_rollback_tail(keep, mirror_lo)") == 1 and "ctx.journals[l].replay(" in rb
    tree = ast.parse(SRC_SPM)
    fns = {n.name: ast.get_source_segment(SRC_SPM, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    d, v = fns["draft_layer"], fns["verify_layer"]
    assert d.count("spec_draft_update_perreq(") == 1 and d.count("flash_attn_nosa_with_kvcache(") == 1 and "num_splits=NL._ATTN_SPLITS" in d
    assert v.count("spec_round_update_perreq(") == 1 and v.count("verify_attention(") == 1 and "st.ctrs[l] = torch.stack(ctr)" in v
    for node in tree.body:                                 # the GPU modules are imported only inside the forwards
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("nosa_llama", "cache_engine"), node.module
    assert pilot.COMMIT == "lockstep"                      # the default


def test_PIN2_spec_loop_still_imports_only_torch_typing_and_collections():
    tree = ast.parse((ROOT / "nosi/nosi/spec_loop.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in ("torch", "typing", "collections"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in ("torch", "typing", "collections", "__future__"), node.module
