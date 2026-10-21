"""R2 step 7, PIECE 4: the FLATTENED JOINT TICK of the speculative round (docs/R2_STEP7_SPEC_ROUND.md sections 2.4,
4, 5.2, 6 and 7.2; the reference is the BI serial decode of r2_bi_decode.py, C's G0-BI path, dd5c4b9).

ONE TICK. The host plans the rows (TickRound.plan_tick = r2_spec_core.Round.plan_tick: per request at most one DRAFT
row and the verify rows of its oldest ready cohort, each verify position its own single-token decode row). Then ONE
flattened call per layer over ALL rows of the tick, with the ops of r2_bi_decode.layer_decode_bi in its order, each
row reading only its own state:
  1. rmsnorm; wqkv through ops.linear (the GEMM mode is a RUN PARAMETER of the ops object: EXACT = G-b
     PaddedLinear(m_fix = 4B), FAST = natural cuBLAS); nosa_linear; rope at each row's own position.
  2. the positional tables: ONE PositionalTwin of 2B banks per layer, the EXACT banks [0, B) (committed +
     verify-provisional) and the DRAFT banks [B, 2B) (the exact prefix + draft rows): every bank advances by its rows in
     position order (twin.advance; ops.mean = fixed_order_mean as the summaries' mean, as attach_bi sets it); a row
     reads the counters after its own position, ctr[u + 1].
  3. stage 1 with the alias: cu_seqlens_k[row] = bank * Lk_max, seqused_k[row] = the row's own n_comp.
  4. pooling per row (its bank's compressed cis, its n_comp, M_b = pos // 64 + 1, the run's M_alloc), then
     r2_spec_core.select_blocks on the pooled scores in float32 (the deterministic two-step rule).
  5. the rows' own K, V and bias (the cis row) into their bank's TAIL PAGE: a ring of two pages per bank, position p in
     page (p // 64) % 2, row p % 64 (note 5.2).
  6. ONE BI paged attention call (piece 3 v2, ops.attn.bi_paged_decode): per (row, KV head) the slots of its selected
     blocks in ASCENDING BLOCK ID: blocks <= T - 2 from the historical slots, T - 1 from its historical slot once it is
     written back (else from the row's bank's ring), T = p // 64 from the ring, cut at last_len = p % 64 + 1 (the
     row's own extent; the slot-63 naming rule of r2_bi_decode.page_table).
     [sharedkv] SELECTOR: when ops.attn has needs_page_key (bi_paged_attention_shared.SharedKV; default: piece 3
     itself, no change), the call also gets page_key (the block ids, backend.page_keys) and rows_max (the most rows
     of one request in this tick), and a request's rows read each page once (docs/R2_SHAREDKV_PLAN.md). The
     ResidentStore gives keys; [sharedkv2] the ResidencyBackend gives the plan's keys on S-C / B2 (plan.page_key,
     r2_residency_sc.page_keys_of) and refuses on v1 / S-B.
  7. wo, gate_up, silu_and_mul, down through ops.linear.
Final rmsnorm, lm_head through ops.linear, .float(); the argmax of every row goes to the DEVICE TOKEN LOG.

STORAGE (ResidentStore, this commit): every committed complete block of a request is RESIDENT on the GPU, in the
historical slot named by its block id; slots NB, NB + 1 are the exact ring, NB + 2, NB + 3 the draft ring. Residency
is content (note 5.1): the page order is the block-id order, so the slot numbers never reach a row's bits. The
store is filled from the R2 state of build_ragged (the host K / V rows of the prompt, the twin's cis rows), so the
tick and the BI serial decode start from the same bytes. The paper's residency mover (a bounded slot store, the
background and urgent gathers, note 5.3) is NOT built here: it is SG5's term dXFER. C's answer (Q1, 2026-10-21):
this store is valid for SG1-SG4 (exactness), NOT for SG5 or any memory / transfer claim.
DRAFT ROWS attend the RESIDENT blocks of their selection only (paper 3.1). With full residency every block is
resident, so a run parameter draft_drop marks a deterministic fraction of a draft row's MIDDLE blocks (not the init
block, not the local window) as not resident: a hash of (request, layer, head, block id, seed), never of the batch;
the kept pages stay in ascending block id, num_pages varies per row, the table width P = topk is a run constant.
Verify rows always read their whole selection. C's answer (Q2): a valid SG2 draft, but NO r / acceptance / R-PERF
number from this stand-in is representative (the gate labels it).

DECISIONS (TickRound). The draft tokens stay on the device (the token log); a row's input is a literal token or a log
slot. ONE host read per tick that finishes a cohort (with the flatten: every tick with verify rows), none in the
others: all log slots written since the last read. Then r2_spec_core.Round's decision (free running), or the
teacher-forced schedule of the gate SG1 (drafts and corrections forced to the teacher's tokens; the rejection index
from a schedule, so every index and the block boundaries are rejected).
ROLLBACK and PUBLICATION (TickEngine.apply_events, after the decision):
  exact bank      a rejection: ONE twin.commit per layer with this tick's ctr (r2_spec_core.rollback_bank), keep =
                  the counters after position keep - 1 (note 6); a publication: nothing (the bank is there already).
  tail pages      nothing: rows beyond the committed position are outside every later extent (note 5.2).
  write-back      a block whose 64 positions are COMMITTED is copied from its exact ring page to its historical slot.
  draft bank      a rejection: sync_draft_bank and the draft ring := the exact ring. A publication (K3X F5,
                  rebuild_from_own_rows): FIRST the draft bank's keys / cis of its still-unverified positions are read
                  from its own twin rows, THEN the bank is rebuilt (sync, then advance by them), and the draft ring
                  rows of the newly published positions take the exact rows.
HOST SYNCS. TickEngine.run_rows and apply_events contain no host read of their own (one pinned H2D copy of the row
table and the per-bank counts per tick); read_tokens is the one decision read. The twins advance through
PositionalTwin.advance_unchecked (C's 4fe5fc9, bitwise advance()'s): no host read; the capacity is checked ONCE per
tick on the host mirrors (row_table: positions < N_max, compressed rows < Lk_max); the twins' sticky overflow flag is
read WITH the decision read, and a set flag VOIDS the tick (C, 2026-10-21): every decision of it is refused (TickError,
nothing commits). The decision path itself (twin.commit, the draft rebuild) still uses the checked twin calls.

STORAGE BACKENDS (the SG5 seam, docs/R2_STEP7_TICK_SG5.md): the engine's arithmetic never changes; the backend says
where a row's pages are and moves committed history.
  ResidentStore (default; SG0-SG4)   the store above; pages built on the device; the draft drop stand-in.
  ResidencyBackend (SG5)             piece 5's Residency / ResidencySB (r2_residency*.py, C's 89f30ee) over a store of
                                     C + X historical slots per stream (NB = C + X) and the same tail rings: per layer
                                     svc.resolve(l, rows, sel) gives the page tables (historical slots in ascending
                                     block id, then the row's tail pages), read_done(l, plan, event) after the
                                     attention, end_tick() once per tick, poll() at its start; a committed block is
                                     copied to the pinned host rows and registered with history_written(uid, t, event);
                                     a publication calls decided(key), a rejection decided(key) then cancel_epoch(uid,
                                     old epoch); a finished request end_request(uid). Draft rows read the service's
                                     READY blocks (paper 3.1); verify rows their whole selection (urgent misses).
INSTRUMENTATION (SG5): engine.timer (a TickTimer) records CUDA event pairs around the GEMMs, the attention, the
selection, the twin advance, the store writes, the backend calls and the whole tick (run_round), and host time for the
row table, the backend and the decision read; read once at the end of a run. timer None: no event, no cost.

SERIAL MODE (run_serial): one exact row per request per call and an immediate commit: the BI serial decode in this
engine (the CPU tests' reference; the GPU gate's SG0 checks it bitwise against r2_bi_decode's).
"""
from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import r2_spec_core as SC
from .r2_decode_path import R2Error
from .r2_positional_twin import BLOCK, PositionalTwin

RING_EXACT, RING_DRAFT = 0, 2              # ring page offsets past the NB historical slots
TWIN_VARIANTS = ("torch", "fused")         # [twin-variant] TickEngine(twin=...): the twins' unchecked step
REBUILD_MODES = ("per_call", "batched")    # [batched-rebuild] TickEngine(rebuild=...): the publications' rebuild
FIELDS = ("pos", "b", "bank", "u", "ring_row", "prev_slot", "tail_slot", "draft", "lit", "slot")


class TickError(R2Error):
    """A refusal of the tick (an R2Error, as the BI serial decode refuses: C's Q4 answer, 2026-10-21)."""


class Ref:
    """A token that is still on the device: slot `slot` of the engine's token log."""
    __slots__ = ("slot",)

    def __init__(self, slot: int):
        self.slot = int(slot)

    def __repr__(self):
        return "Ref(%d)" % self.slot


def rebuild_from_own_rows(twin: PositionalTwin, b: int, B: int, p: int, dpos: int):
    """K3X F5 on the draft bank B + b after a publication up to position p (C's Q5 answer, 2026-10-21): FIRST capture
    the draft's own unverified rows, positions [p, dpos) (its ring rows q % 64 and its total_cis rows q), THEN
    r2_spec_core.rebuild_draft_bank (sync_draft_bank: the exact bank's tables and counters, then advance by the
    captured rows). Returns the captured (keys (n, H, D), cis (n, H))."""
    if not 0 <= dpos - p <= 64:
        raise TickError("draft bank %d: %d unverified rows (the ring holds 64)" % (B + b, dpos - p))
    idx = torch.arange(int(p), int(dpos), dtype=torch.int64, device=twin.pos.device)
    dk = twin.ring[B + b, idx % BLOCK].clone()
    dc = twin.total_cis[B + b, idx].clone()
    SC.rebuild_draft_bank(twin, b, B, dk, dc)
    return dk, dc


def rebuild_banks_unchecked(twin: PositionalTwin, items: Sequence[tuple], B: int, dev_items=None,   # [batched-rebuild]
                            swap: bool = False):                                                  # [batched-rebuild]
    """[batched-rebuild] (A2, 7f06273) rebuild_from_own_rows for EVERY publication of a tick at once, on one layer's
    twin, without a host read: items = [(b, p, dpos)], one per bank. The twin's banks are independent, so: FIRST every
    draft bank's own unverified rows [p, dpos) (ONE gather of the ring and ONE of total_cis over all items), THEN
    sync_draft_bank of every item (one index copy per table: r2_spec_core.sync_draft_bank batched), then ONE
    advance_unchecked over T = max(dpos - p) steps with n[B + b] = dpos - p (a bank is inactive past its own rows, as
    every other bank is in the per-call path). Bitwise rebuild_from_own_rows per bank (test P6a). dev_items: (3, P)
    int64 on the twin's device (b, p, n; the caller makes it once per tick for every layer). swap: NEGATIVE CONTROL
    (tests only), the first two items' rows exchanged. Returns {b: (keys (n, H, D), cis (n, H))}."""
    if not items:
        return {}
    dev = twin.pos.device
    bs = [int(b) for (b, _, _) in items]
    if len(set(bs)) != len(bs):
        raise TickError("rebuild_banks_unchecked: one publication per bank (got %s)" % bs)
    ns = [int(d) - int(p) for (_, p, d) in items]
    for b, n in zip(bs, ns):
        if not 0 <= n <= BLOCK:
            raise TickError("draft bank %d: %d unverified rows (the ring holds 64)" % (B + b, n))
    T, P = max(ns), len(items)
    if dev_items is None:
        dev_items = torch.tensor([bs, [int(p) for (_, p, _) in items], ns], dtype=torch.int64).to(dev)
    hb, hp, hn = dev_items[0], dev_items[1], dev_items[2]
    drafts = hb + B
    out = {}
    if T:
        j = torch.arange(T, dtype=torch.int64, device=dev).view(1, T)
        pos = hp.view(P, 1) + j                                         # (P, T): the absolute positions
        valid = j < hn.view(P, 1)
        rk = twin.ring[drafts.view(P, 1), pos % BLOCK]                  # (P, T, H, D): the draft rows, BEFORE the sync
        rc = twin.total_cis[drafts.view(P, 1), pos.clamp(max=twin.total_cis.shape[1] - 1)]   # (P, T, H)
        rk = torch.where(valid.view(P, T, 1, 1), rk, torch.zeros_like(rk))
        rc = torch.where(valid.view(P, T, 1), rc, torch.zeros_like(rc))
        for i, (b, n) in enumerate(zip(bs, ns)):
            out[b] = (rk[i, :n], rc[i, :n])
    else:
        for b in bs:
            out[b] = (twin.ring[B + b, :0].clone(), twin.total_cis[B + b, :0].clone())
    twin.comp_k.index_copy_(0, drafts, twin.comp_k.index_select(0, hb))      # sync_draft_bank, every item at once
    twin.ring.index_copy_(0, drafts, twin.ring.index_select(0, hb))
    twin.total_cis.index_copy_(0, drafts, twin.total_cis.index_select(0, hb))
    twin.ccis.index_copy_(1, drafts, twin.ccis.index_select(1, hb))
    for t in (twin.pos, twin.n_comp, twin.n_ccis, twin.prefilled):
        t.index_copy_(0, drafts, t.index_select(0, hb))
    twin.epoch.index_add_(0, drafts, torch.ones_like(drafts, dtype=twin.epoch.dtype))
    if T:
        keys = torch.zeros((twin.B, T, twin.H, twin.D), dtype=twin.ring.dtype, device=dev)
        cis = torch.zeros((twin.B, T, twin.H), dtype=twin.total_cis.dtype, device=dev)
        cnt = torch.zeros(twin.B, dtype=torch.int64, device=dev)
        order = drafts
        if swap and P >= 2:                                             # NEGATIVE CONTROL: two banks' rows exchanged
            order = drafts.clone()
            order[0], order[1] = drafts[1], drafts[0]
        keys.index_copy_(0, order, rk.to(keys.dtype))
        cis.index_copy_(0, order, rc.to(cis.dtype))
        cnt.index_copy_(0, drafts, hn)
        twin.advance_unchecked(keys, cis, cnt)
    return out


def default_nl():
    """nosa_llama (CUDA extensions), imported on first use."""
    from . import nosa_llama
    return nosa_llama


def default_kernels():
    from .max_pooling_per_request import nosa_pooling_per_request
    from .stage1_seqused import stage1_seqused
    return stage1_seqused, nosa_pooling_per_request


# =====================================================================================================================
# the round with device tokens
# =====================================================================================================================
class TickRound(SC.Round):
    """r2_spec_core.Round whose draft and verifier tokens stay on the device until the tick's decision.
    finish_tick(rows, slots, hi): slots[k] = the token-log slot of row k's argmax, hi = the log's length after the tick.
    read_tokens(lo, hi) -> list of ints: ONE host read of log slots [lo, hi) (counted in self.syncs).
    TEACHER-FORCED mode (the gate SG1): teacher(b, pos) gives the teacher's token AT absolute position pos; drafts and
    corrections are the teacher's; reject_at(b, cohort, E) -> None (accept all) or the index i in [a, b] where the
    cohort is rejected (positions >= E + i are rolled back, the correction is teacher(b, E + i))."""

    def __init__(self, first_tokens, start_pos, read_tokens: Callable, K: int = 3, LAMBDA: int = 7, max_new=None,
                 on_publish=None, on_rollback=None, ready=None, teacher: Callable = None, reject_at: Callable = None):
        super().__init__(first_tokens, start_pos, K=K, LAMBDA=LAMBDA, max_new=max_new, on_publish=on_publish,
                         on_rollback=on_rollback, ready=ready)
        if (teacher is None) != (reject_at is None):
            raise TickError("teacher and reject_at go together (the teacher-forced mode)")
        self.read_tokens = read_tokens
        self.teacher = teacher
        self.reject_at = reject_at
        self.read_upto = 0
        self.decisions = []                       # teacher-forced: (b, epoch, m, a, b_end, reject index or None, E)

    def finish_tick(self, rows, slots, hi: int) -> int:
        seen_d, seen_m = set(), {}
        for row in rows:
            if row.kind == "d":
                if row.b in seen_d:
                    raise RuntimeError("tick: request %d has two draft rows" % row.b)
                seen_d.add(row.b)
            elif seen_m.setdefault(row.b, row.m) != row.m:
                raise RuntimeError("tick: request %d has verify rows of two cohorts" % row.b)
        if len(slots) != len(rows):
            raise TickError("%d slots for %d rows" % (len(slots), len(rows)))
        self.ticks += 1
        done_any = False
        for row, slot in zip(rows, slots):
            r = self.reqs[row.b]
            if row.epoch != r.epoch:
                continue
            if row.kind == "d":
                self.stats["draft_rows"] += 1
                if row.i != r.s + 1:
                    raise RuntimeError("draft row out of order: i=%d s=%d" % (row.i, r.s))
                r.u.append(Ref(slot) if self.teacher is None else int(self.teacher(row.b, row.pos + 1)))
                r.s += 1
                m = SC.triggered_by(r.s, self.K)
                if m is not None:
                    r.cohorts.append(SC.Cohort(m, *SC.cohort_range(m, self.K)))
                    r.n_triggered = m
                if len(r.out) + self.lead(r) >= r.max_new and not r.flushed:      # C4: the final short cohort
                    a = SC.cohort_range(r.n_triggered + 1, self.K)[0]
                    if a <= r.s:
                        r.cohorts.append(SC.Cohort(r.n_triggered + 1, a, r.s))
                    r.flushed = True
            else:
                self.stats["verify_rows"] += 1
                c = next(c for c in r.cohorts if c.m == row.m)
                c.g[row.i] = Ref(slot)
                if len(c.g) == c.b - c.a + 1:
                    c.state = "DONE"
                    done_any = True
        if done_any:
            self._resolve(int(hi))
            self.syncs += 1
            for r in self.reqs:
                self._decide(r)
        return int(done_any)

    def _resolve(self, hi: int) -> None:
        """The ONE host read of the tick: log slots [read_upto, hi), substituted for every Ref still held."""
        lo = self.read_upto
        vals = self.read_tokens(lo, hi) if hi > lo else []
        if len(vals) != hi - lo:
            raise TickError("read_tokens(%d, %d) returned %d values" % (lo, hi, len(vals)))

        def v(x):
            if isinstance(x, Ref):
                if not lo <= x.slot < hi:
                    raise TickError("token slot %d outside the read [%d, %d)" % (x.slot, lo, hi))
                return int(vals[x.slot - lo])
            return x
        for r in self.reqs:
            r.u = [v(x) for x in r.u]
            for c in r.cohorts:
                c.g = {i: v(x) for i, x in c.g.items()}
        self.read_upto = hi

    def _decide(self, r) -> None:
        if self.reject_at is None:
            return super()._decide(r)                                          # r2_spec_core.Round's greedy decision
        while r.cohorts and r.cohorts[0].state == "DONE":                  # Round._decide with a forced rejection
            c = r.cohorts.pop(0)
            self.stats["cohorts"] += 1
            j = self.reject_at(r.uid, c, r.E)
            if j is not None and not c.a <= j <= c.b:
                raise TickError("reject_at gave %r outside [%d, %d]" % (j, c.a, c.b))
            self.decisions.append((r.uid, r.epoch, c.m, c.a, c.b, j, r.E))
            for i in range(c.a, c.b + 1):
                if j is None or i < j:
                    r.out.append(r.u[i])
                    r.committed_in_epoch += 1
                    self.stats["accepted"] += 1
                    continue
                self.stats["rejected"] += 1
                corr = int(self.teacher(r.uid, r.E + i))
                r.out.append(corr)
                old = r.epoch
                keep = r.E + i
                r.epoch += 1
                r.E, r.u, r.s, r.committed_in_epoch, r.n_triggered = keep, [corr], 0, 0, 0
                r.cohorts, r.flushed = [], False
                self.on_rollback(r.uid, old, keep)
                break
            else:
                self.on_publish(r.uid, r.epoch, r.E + c.b)
            if len(r.out) >= r.max_new:
                r.done = True
                del r.out[r.max_new:]
                r.cohorts = []
                return


# =====================================================================================================================
# the engine
# =====================================================================================================================
class TickEngine:
    """The flattened joint tick of B requests over all layers (module docstring).

    model: .layers (the LlamaLayer attributes the BI decode reads), .embed_tokens, .cos_sin_cache, .norm_weight,
    .norm_variance_epsilon, .lm_head. ops: .linear (the GEMM mode), .attn (piece 3: bi_paged_decode), .mean (the
    twins' summary mean). twins: one PositionalTwin(2B, ...) per layer, banks [0, B) exact, [B, 2B) draft, prefilled.
    store_k / store_v (B, (NB + 4) * 64, Hkv, D) and store_bias (B, (NB + 4) * 64, Hkv) per layer. S_list: the prompt
    lengths. M_alloc: the pooled width of the run (the BI serial decode's). nl, stage1, pool: the nosa_llama ops, the
    stage-1 call and the pooling (defaults: the CUDA ones; the CPU tests pass torch stand-ins). check: host checks of
    the selection's last two blocks (T - 1, T) every layer (tests and gates; one read per layer).
    twin ([twin-variant]): "torch" (the default, unchanged) or "fused": every twin's step_unchecked / advance_unchecked
    is bound on the instance to r2_twin_fused (ONE Triton kernel per layer, bitwise with ops.mean = fixed_order_mean);
    the call sites are unchanged.
    rebuild ([batched-rebuild]): "per_call" (the default, unchanged: one CHECKED rebuild_from_own_rows per publication
    and layer) or "batched": apply_events rebuilds every publication of the tick with ONE rebuild_banks_unchecked and
    ONE copy_rows_many per layer (no host read; with twin "fused" the advance is one fused kernel); bitwise the same."""

    LOG = 1 << 16

    def __init__(self, model, ops, twins: List[PositionalTwin], store_k, store_v, store_bias, S_list, M_alloc: int,
                 nl=None, stage1=None, pool=None, select=SC.select_blocks, draft_drop: float = 0.0, drop_seed: int = 0,
                 check: bool = False, twin: str = "torch",                                      # [twin-variant]
                 rebuild: str = "per_call"):                                                    # [batched-rebuild]
        if rebuild not in REBUILD_MODES:                                                        # [batched-rebuild]
            raise TickError("rebuild must be one of %s (got %r)" % (REBUILD_MODES, rebuild))    # [batched-rebuild]
        self.rebuild = rebuild                                                                  # [batched-rebuild]
        self.model, self.ops = model, ops
        self.nl = nl if nl is not None else default_nl()
        if stage1 is None or pool is None:
            s1, pl = default_kernels()
            stage1, pool = stage1 or s1, pool or pl
        self.stage1, self.pool, self.select = stage1, pool, select
        self.twins = list(twins)
        self.L = len(self.twins)
        if self.L != len(model.layers) or not (len(store_k) == len(store_v) == len(store_bias) == self.L):
            raise TickError("one twin and one store per layer")
        t0 = self.twins[0]
        if t0.B % 2:
            raise TickError("the twins need 2B banks")
        self.B = t0.B // 2
        self.H, self.D = t0.H, t0.D
        self.store_k, self.store_v, self.store_bias = list(store_k), list(store_v), list(store_bias)
        rows = int(self.store_k[0].shape[1])
        if rows % BLOCK or rows // BLOCK < 5 or tuple(self.store_k[0].shape) != (self.B, rows, self.H, self.D):
            raise TickError("store (B, (NB + 4) * 64, Hkv, D) expected, got %s" % (tuple(self.store_k[0].shape),))
        self.NB = rows // BLOCK - 4
        self.dev = t0.pos.device
        self.dtype = self.store_k[0].dtype
        self.M_alloc = int(M_alloc)
        self.S_list = [int(s) for s in S_list]
        if len(self.S_list) != self.B:
            raise TickError("S_list needs %d lengths" % self.B)
        for t in self.twins:
            t.mean_fn = ops.mean
        if twin not in TWIN_VARIANTS:                                                           # [twin-variant]
            raise TickError("twin must be one of %s (got %r)" % (TWIN_VARIANTS, twin))          # [twin-variant]
        self.twin_variant = twin                                                                # [twin-variant]
        if twin == "fused":                                                                     # [twin-variant]
            from . import r2_twin_fused                                                         # [twin-variant]
            for t in self.twins:                                                                # [twin-variant]
                try:                                                                            # [twin-variant]
                    r2_twin_fused.attach(t)                                                     # [twin-variant]
                except ValueError as e:                                                         # [twin-variant]
                    raise TickError("twin 'fused': %s" % e)                                     # [twin-variant]
        self.draft_drop = float(draft_drop)
        if not 0.0 <= self.draft_drop < 1.0:
            raise TickError("draft_drop in [0, 1)")
        self.drop_seed = int(drop_seed)
        self.check = bool(check)
        # host mirrors (no device reads)
        self.bank_pos = self.S_list + self.S_list           # the next position of every bank
        self.committed = list(self.S_list)                  # exact positions < committed[b] are committed
        self.wb = [s // BLOCK for s in self.S_list]         # blocks < wb[b] are in their historical slots
        self.synced = list(self.S_list)                     # draft bank b: positions < synced[b] are exact rows
        self.last_ctr: List[Optional[torch.Tensor]] = [None] * self.L
        self.last_pos0 = None
        self.last_count = None
        self.events = []
        self.tok_log = torch.zeros(self.LOG, dtype=torch.int64, device=self.dev)
        self.n_tok = 0
        self.stats = dict(ticks=0, rows=0, draft_rows=0, verify_rows=0, decision_reads=0, writebacks=0,
                          rollbacks=0, publications=0, ms_rows=0.0, void_ticks=0)
        self.cu_q_cache = {}
        self.backend = ResidentStore(self)
        self.timer = None
        self.void = False
        self.tie_acc = None          # count_ties(): per layer (pairs, ties at the QK_SELECT cut, ties at the top-k cut)
        self.plant = None            # plant_ties(): the gate SG4c's planted ties (tests and gates only)

    def count_ties(self) -> None:
        """Count, per layer, the (row, head) pairs whose pooled scores TIE across a cut of the two-step selection
        (device counters; read once with tie_report)."""
        self.tie_acc = torch.zeros((self.L, 5), dtype=torch.int64, device=self.dev)

    def tie_report(self) -> List[List[int]]:
        """TESTS AND GATES (one read): per layer [pairs, ties at the QK_SELECT cut, ties at the top-k cut, planted
        pairs, planted pairs that tie across the top-k cut]."""
        return self.tie_acc.tolist()

    def plant_ties(self, requests: Sequence[int], layers: Sequence[int]) -> None:
        """SG4c PLANTED (tests and gates only): at the given layers, the rows of the given requests get a TIE across
        the top-k cut of the second step (plant_tie), so a tie rule other than the stable one changes the selection."""
        self.plant = dict(layers=set(int(l) for l in layers),
                          b_dev=torch.tensor(sorted(int(b) for b in requests), dtype=torch.int64, device=self.dev))

    def attach_backend(self, backend) -> None:
        """Replace the storage backend before the first tick (module docstring, STORAGE BACKENDS)."""
        if self.stats["ticks"] or self.n_tok:
            raise TickError("attach_backend before the first tick")
        self.backend = backend

    def _span(self, cat: str):
        return self.timer.span(cat) if self.timer is not None else contextlib.nullcontext()

    def _lin(self, x, w):
        with self._span("gemm"):
            return self.ops.linear(x, w)

    # ------------------------------------------------------------------ construction from the R2 state
    @classmethod
    def from_r2(cls, model, cache, st, ops, hist_slots: Optional[int] = None, **kw):
        """A tick engine whose state is a COPY of a build_ragged state (body 'r2'; before its first decode step): the
        twins of every layer into both banks, the prompt's host K / V rows into the historical slots (complete
        blocks) and the exact ring (the partial last block), the twin's cis rows as the bias; the draft ring = the
        exact ring. The source is not changed.
        hist_slots None: the ResidentStore layout (a historical slot per block id). hist_slots = C + X: the Residency
        layout (piece 5, C's D4): C + X EMPTY historical slots per stream (the service fills them from the host rows),
        then the same tail rings; the caller attaches a ResidencyBackend."""
        from .r2_bi_decode import TWIN_FIELDS
        if st.body_mode != "r2" or st.steps != 0:
            raise TickError("from_r2 needs a fresh body-'r2' state (body %r, steps %d)" % (st.body_mode, st.steps))
        B = st.B
        S_list = [int(x) for x in st.pos_host.tolist()]
        resident = hist_slots is None
        NB = st.n_max // BLOCK + 1 if resident else int(hist_slots)
        if NB < 1:
            raise TickError("hist_slots must be >= 1")
        twins, sk, sv, sb = [], [], [], []
        for l, lay in enumerate(cache.layers):
            src = st.twins[l]
            t = PositionalTwin(2 * B, src.H, src.D, src.lk_max, src.n_max, dtype=src.ring.dtype, device=src.pos.device)
            for name in TWIN_FIELDS:
                dst = getattr(t, name)
                dst[:B].copy_(getattr(src, name))
                dst[B:].copy_(getattr(src, name))
            t.ccis[:, :B].copy_(src.ccis)
            t.ccis[:, B:].copy_(src.ccis)
            twins.append(t)
            eng = lay.cache_engine
            dev, dt = src.ring.device, src.ring.dtype                         # the twin's = the engine's (build_ragged)
            k = torch.zeros((B, (NB + 4) * BLOCK, src.H, src.D), dtype=dt, device=dev)
            v = torch.zeros_like(k)
            bias = torch.zeros((B, (NB + 4) * BLOCK, src.H), dtype=dt, device=dev)
            for b, S in enumerate(S_list):
                full = (S // BLOCK) * BLOCK
                if resident:
                    k[b, :full].copy_(eng._k_cpu[b, :full])
                    v[b, :full].copy_(eng._v_cpu[b, :full])
                    bias[b, :full].copy_(src.total_cis[b, :full])
                if S > full:
                    r0 = (NB + RING_EXACT + (S // BLOCK) % 2) * BLOCK
                    k[b, r0:r0 + S - full].copy_(eng._k_cpu[b, full:S])
                    v[b, r0:r0 + S - full].copy_(eng._v_cpu[b, full:S])
                    bias[b, r0:r0 + S - full].copy_(src.total_cis[b, full:S])
                e0, d0 = (NB + RING_EXACT) * BLOCK, (NB + RING_DRAFT) * BLOCK
                k[b, d0:d0 + 2 * BLOCK].copy_(k[b, e0:e0 + 2 * BLOCK])
                v[b, d0:d0 + 2 * BLOCK].copy_(v[b, e0:e0 + 2 * BLOCK])
                bias[b, d0:d0 + 2 * BLOCK].copy_(bias[b, e0:e0 + 2 * BLOCK])
            sk.append(k)
            sv.append(v)
            sb.append(bias)
        if torch.device(st.cu_q.device).type == "cuda":
            torch.cuda.synchronize()
        return cls(model, ops, twins, sk, sv, sb, S_list, st.M_alloc, **kw)

    def clone(self, ops=None, **kw):
        """An independent copy of the engine's state before its first tick (the gate's arms start from it). ops and
        the keyword arguments of the constructor (draft_drop, check, ...) may be replaced."""
        if self.stats["ticks"] or self.n_tok:
            raise TickError("clone() is for a fresh engine")
        twins = []
        for t in self.twins:
            c = PositionalTwin(t.B, t.H, t.D, t.lk_max, t.n_max, dtype=t.ring.dtype, device=t.pos.device)
            for name in ("comp_k", "ring", "total_cis", "pos", "n_comp", "n_ccis", "epoch", "prefilled", "ccis"):
                getattr(c, name).copy_(getattr(t, name))
            twins.append(c)
        args = dict(nl=self.nl, stage1=self.stage1, pool=self.pool, select=self.select, draft_drop=self.draft_drop,
                    drop_seed=self.drop_seed, check=self.check, twin=self.twin_variant)                 # [twin-variant]
        args.setdefault("rebuild", self.rebuild)                                                # [batched-rebuild]
        args.update(kw)
        return type(self)(self.model, ops if ops is not None else self.ops, twins, [x.clone() for x in self.store_k],
                          [x.clone() for x in self.store_v], [x.clone() for x in self.store_bias], self.S_list,
                          self.M_alloc, **args)

    # ------------------------------------------------------------------ the row table (host)
    def ring_slot(self, draft: bool, pos: int) -> int:
        return self.NB + (RING_DRAFT if draft else RING_EXACT) + (int(pos) // BLOCK) % 2

    def row_table(self, rows: Sequence, tokens: Sequence):
        """Host checks and the packed row table. rows: objects with b, kind ('v' exact / 'd' draft), pos; tokens: an
        int (a literal token) or a Ref per row. Returns (table (len(FIELDS), R) int64 CPU, per-bank counts, T)."""
        R = len(rows)
        if R == 0:
            raise TickError("a tick with no rows")
        topk = int(self.model.layers[0].topk_blocks)
        groups: Dict[int, List[int]] = {}
        tab = torch.empty((len(FIELDS), R), dtype=torch.int64)
        seen = set()
        for k, (row, tok) in enumerate(zip(rows, tokens)):
            b, kind, p = int(row.b), row.kind, int(row.pos)
            if not 0 <= b < self.B or kind not in ("v", "d"):
                raise TickError("row %d: request %d kind %r" % (k, b, kind))
            draft = kind == "d"
            bank = self.B + b if draft else b
            if (bank, p) in seen:
                raise TickError("row %d: bank %d position %d twice" % (k, bank, p))
            seen.add((bank, p))
            g = groups.setdefault(bank, [])
            if p != self.bank_pos[bank] + len(g):
                raise TickError("row %d: bank %d writes position %d, its next is %d" % (k, bank, p, self.bank_pos[bank] + len(g)))
            T = p // BLOCK
            if T + 1 < topk:                                     # the BI serial decode's limit: pos // 64 >= topk - 1
                raise TickError("row %d: position %d has %d blocks < topk %d (pos // 64 >= %d is needed: the selection "
                                "would name absent blocks)" % (k, p, T + 1, topk, topk - 1))
            t0 = self.twins[0]
            if T + 1 >= self.backend.max_blocks() or p >= t0.n_max or p // 16 + 1 >= t0.lk_max:
                raise TickError("row %d: position %d beyond the store's blocks (%d), N_max %d or Lk_max %d: the tick's ONE "
                                "capacity check (the unchecked twin step must never meet a full table)"
                                % (k, p, self.backend.max_blocks(), t0.n_max, t0.lk_max))
            if self.wb[b] < T - 1:
                raise TickError("row %d: block %d is not written back (wb %d); the lead bound is broken" % (k, T - 2, self.wb[b]))
            if p < self.committed[b]:
                raise TickError("row %d: position %d is committed already (%d)" % (k, p, self.committed[b]))
            tail = self.ring_slot(draft, p)
            prev = T - 1 if self.wb[b] >= T else self.ring_slot(draft, p - BLOCK)
            if isinstance(tok, Ref):
                lit, slot = -1, tok.slot
                if not 0 <= slot < self.n_tok:
                    raise TickError("row %d: token slot %d not written" % (k, slot))
            else:
                lit, slot = int(tok), 0
                if lit < 0:
                    raise TickError("row %d: token %d" % (k, lit))
            tab[:, k] = torch.tensor([p, b, bank, len(g), tail * BLOCK + p % BLOCK, prev, tail, int(draft), lit, slot])
            g.append(k)
        count = [0] * (2 * self.B)
        for bank, g in groups.items():
            count[bank] = len(g)
        return tab, count, max(count)

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        if self.dev.type == "cuda":
            return t.pin_memory().to(self.dev, non_blocking=True)
        return t.to(self.dev)

    # ------------------------------------------------------------------ one flattened call
    @torch.inference_mode()
    def run_rows(self, rows: Sequence, tokens: Sequence):
        """ONE flattened forward over the tick's rows (module docstring). Advances the banks' twins, writes the rows'
        K / V / bias into their ring pages, appends every row's argmax to the token log. Returns
        SimpleNamespace(logits (R, V) float, argmax (R,), slots [R ints], lo, hi)."""
        t_start = time.perf_counter()
        nl, ops, model = self.nl, self.ops, self.model
        self.backend.begin_tick()
        t_rows = time.perf_counter()
        tab, count, T = self.row_table(rows, tokens)
        svc_rows = self.backend.prepare(rows)
        R = len(rows)
        if self.timer is not None:
            self.timer.host("backend", t_rows - t_start)
            self.timer.host("rows", time.perf_counter() - t_rows)
        buf = self._to_device(torch.cat([tab.reshape(-1), torch.tensor(count, dtype=torch.int64)]))
        d = buf[:len(FIELDS) * R].view(len(FIELDS), R)
        n_dev = buf[len(FIELDS) * R:]                                       # (2B,) per-bank counts, on the device
        pos, rb, bank, u = d[0], d[1], d[2], d[3]
        ring_row, prev_slot, tail_slot, is_draft = d[4], d[5].to(torch.int32), d[6].to(torch.int32), d[7] != 0
        lit, slot = d[8], d[9]
        pos32 = pos.to(torch.int32).contiguous()
        rb32 = rb.to(torch.int32).contiguous()
        M_b = (pos // BLOCK + 1).to(torch.int32).contiguous()
        last = (pos % BLOCK + 1).to(torch.int32).contiguous()
        lk = self.twins[0].lk_max
        cu_k = torch.cat([bank * lk, torch.full((1,), 2 * self.B * lk, dtype=torch.int64, device=self.dev)]).to(torch.int32).contiguous()
        cu_q = self.cu_q_cache.get(R)
        if cu_q is None:
            cu_q = self.cu_q_cache[R] = torch.arange(R + 1, dtype=torch.int32, device=self.dev)
        ids = torch.where(lit >= 0, lit, self.tok_log[slot]).view(R, 1)
        any_draft = any(r.kind == "d" for r in rows)
        rows_max = None
        if getattr(ops.attn, "needs_page_key", False):                       # [sharedkv] host: rows per request
            per_req: Dict[int, int] = {}                                     # (the default path skips this)
            for r in rows:
                per_req[r.b] = per_req.get(r.b, 0) + 1
            rows_max = max(per_req.values())
        ctx = SimpleNamespace(pos=pos, rb=rb, rb32=rb32, prev_slot=prev_slot, tail_slot=tail_slot, is_draft=is_draft,
                              any_draft=any_draft, last=last, rows=svc_rows, rows_max=rows_max)
        pos0 = list(self.bank_pos)
        hidden = F.embedding(ids, model.embed_tokens)
        ctrs = []
        for l, layer in enumerate(model.layers):
            hidden, ctr = self._layer(l, layer, hidden, R, T, n_dev, pos32, bank, u, ring_row, M_b, cu_q, cu_k, ctx)
            ctrs.append(ctr)
        hidden = nl.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
        logits = self._lin(hidden, model.lm_head).float()[:, -1]
        t_be = time.perf_counter()
        self.backend.end_tick()
        if self.timer is not None:
            self.timer.host("backend", time.perf_counter() - t_be)
        arg = logits.argmax(-1)
        lo = self.n_tok
        if lo + R > self.LOG:
            raise TickError("token log full")
        self.tok_log[lo:lo + R].copy_(arg)
        self.n_tok = lo + R
        for bk, c in enumerate(count):
            self.bank_pos[bk] += c
        self.last_ctr, self.last_pos0, self.last_count = ctrs, pos0, count
        self.stats["ticks"] += 1
        self.stats["rows"] += R
        nd = sum(1 for r in rows if r.kind == "d")
        self.stats["draft_rows"] += nd
        self.stats["verify_rows"] += R - nd
        self.stats["ms_rows"] += 1e3 * (time.perf_counter() - t_start)
        return SimpleNamespace(logits=logits, argmax=arg, slots=list(range(lo, lo + R)), lo=lo, hi=lo + R)

    def _layer(self, l, layer, hidden, R, T, n_dev, pos32, bank, u, ring_row, M_b, cu_q, cu_k, ctx):
        nl, lin = self.nl, self._lin
        Hkv, D = layer.num_key_value_heads, layer.head_dim
        twin = self.twins[l]
        pos, rb = ctx.pos, ctx.rb
        residual = hidden
        h = nl.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = lin(h, layer.wqkv)
        q, k, v, cis = nl.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
        q = q.view(R, -1)
        k = k.view(R, -1)
        nl.apply_rope_with_cos_sin_cache_inplace(pos32, q, k, D, self.model.cos_sin_cache, True)
        q = q.reshape(R, layer.num_heads, D)
        k = k.reshape(R, Hkv, D)
        v = v.reshape(R, Hkv, D)
        cis = cis.reshape(R, Hkv)
        # 2. the positional tables: every bank advances by its rows in position order (no host read: C's 4fe5fc9)
        with self._span("twin"):
            keys = torch.zeros((twin.B, T, Hkv, D), dtype=twin.ring.dtype, device=self.dev)
            cis_b = torch.zeros((twin.B, T, Hkv), dtype=twin.total_cis.dtype, device=self.dev)
            keys[bank, u] = k.to(twin.ring.dtype)
            cis_b[bank, u] = cis.to(twin.total_cis.dtype)
            ctr = twin.advance_unchecked(keys, cis_b, n_dev)                          # (T + 1, 4, 2B)
            n_used = ctr[u + 1, 1, bank].to(torch.int32).contiguous()
        # 3. stage 1 with the alias; 4. pooling per row, the deterministic selection
        with self._span("sel"):
            ck = twin.comp_k_flat()
            score = self.stage1(q, ck, ck, cu_q, cu_k, 1, twin.lk_max, seqused_k=n_used)
            ccis = twin.ccis.index_select(1, bank)
            pq = torch.empty((Hkv, R, self.M_alloc), dtype=layer.wqkv.dtype, device=self.dev)
            pc = torch.empty_like(pq)
            self.pool(score, ccis, n_used, M_b, pq, pc, stride=layer.pooling_stride, block_size=layer.block_size,
                      local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
            planted = None
            if self.plant is not None and l in self.plant["layers"]:   # SG4c planted: a tie at the top-k boundary
                planted = plant_tie(pq, pc, torch.isin(rb, self.plant["b_dev"]), self.nl.QK_SELECT, layer.topk_blocks)
            if self.tie_acc is not None:                                  # SG4c: ties at the two cut boundaries
                self.tie_acc[l] += tie_stats(pq, pc, self.nl.QK_SELECT, layer.topk_blocks, planted)
            sel = self.select(pq.float(), pc.float(), self.nl.QK_SELECT, layer.topk_blocks)   # (H, R, topk) ascending
        if self.check:
            blk = (pos // BLOCK).view(1, R)
            if not (bool((sel[..., -1] == blk).all()) and bool((sel[..., -2] == blk - 1).all())):
                raise TickError("layer %d: a selection does not end with the row's blocks T - 1, T" % l)
        # 5. the rows' own K / V / bias into their ring pages
        with self._span("store"):
            self.store_k[l].index_put_((rb, ring_row), k.to(self.dtype))
            self.store_v[l].index_put_((rb, ring_row), v.to(self.dtype))
            self.store_bias[l].index_put_((rb, ring_row), cis.to(self.dtype))
        # 6. the page lists (the storage backend) and ONE BI attention call
        t_res = time.perf_counter()
        with self._span("res"):
            table, npg, last, seq_idx, plan = self.backend.pages(l, ctx, sel, layer)
        if self.timer is not None:
            self.timer.host("backend", time.perf_counter() - t_res)
        akw = {}
        if getattr(self.ops.attn, "needs_page_key", False):                # [sharedkv] the selector (default: piece 3)
            with self._span("res"):
                akw = dict(page_key=self.backend.page_keys(l, ctx, sel, layer, plan=plan), rows_max=ctx.rows_max)
        with self._span("att"):
            attn = self.ops.attn.bi_paged_decode(q, self.store_k[l], self.store_v[l], self.store_bias[l], seq_idx,
                                                 table, npg, last, **akw)
        t_res = time.perf_counter()
        with self._span("res"):
            self.backend.after_attention(l, plan)
        if self.timer is not None:
            self.timer.host("backend", time.perf_counter() - t_res)
        attn = attn.view(R, 1, -1)
        hidden = residual + lin(attn, layer.wo)
        residual = hidden
        h = nl.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = lin(h, layer.gate_up_proj)
        out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        nl.silu_and_mul(h, out)
        return residual + lin(out, layer.down_proj), ctr

    def page_table(self, l, sel, pos, rb, prev_slot, tail_slot, is_draft, any_draft, layer):
        """(table (R, H, topk) int32, num_pages (R, H) int32): the selection in ascending block id, its last two
        entries (T - 1, T) renamed to the row's slots; for draft rows the blocks marked not resident are left out
        (stable compaction: the order stays ascending)."""
        H, R, K = sel.shape
        table = sel.permute(1, 0, 2).to(torch.int32).contiguous()                       # historical slot = block id
        table[:, :, K - 1] = tail_slot.view(R, 1)
        table[:, :, K - 2] = prev_slot.view(R, 1)
        if not (any_draft and self.draft_drop > 0):
            return table, torch.full((R, H), K, dtype=torch.int32, device=self.dev)
        keep, dest = self._draft_keep(l, sel, pos, rb, is_draft, layer)
        out = torch.full((R, H, K + 1), -1, dtype=torch.int32, device=self.dev)
        out.scatter_(2, dest, table)
        return out[:, :, :K].contiguous(), keep.sum(-1).to(torch.int32).contiguous()

    def _draft_keep(self, l, sel, pos, rb, is_draft, layer):
        """The draft drop stand-in (module docstring): (keep (R, H, K) bool, dest (R, H, K) int64: the compacted
        position of each kept entry, K for a dropped one). Shared by page_table and page_keys (sharedkv), so the keys
        are compacted exactly like the table. Moved out of page_table unchanged (the same ops in the same order)."""
        H, R, K = sel.shape
        blk = sel.permute(1, 0, 2)                                                       # (R, H, K) int64
        T = (pos // BLOCK).view(R, 1, 1)
        hh = torch.arange(H, device=self.dev).view(1, H, 1)
        x = (blk * 40503 + rb.view(R, 1, 1) * 2654435 + l * 9973 + hh * 7919 + self.drop_seed * 104729) % 1000003
        middle = (blk >= layer.init_blocks) & (blk < T - layer.window_blocks)
        drop = is_draft.view(R, 1, 1) & middle & ((x % 1000) < int(round(self.draft_drop * 1000)))
        keep = ~drop
        dest = torch.where(keep, keep.to(torch.int64).cumsum(-1) - 1, torch.full_like(blk, K))
        return keep, dest

    def page_keys(self, l, sel, pos, rb, is_draft, any_draft, layer):
        """[sharedkv] (R, H, topk) int32: the BLOCK ID of every entry of page_table's lists (the selection itself, before
        the renaming of T - 1, T to slots), compacted exactly like the table. The shared-read attention orders a
        request's union pages by (block id, slot); a ring slot alone does not give the block order."""
        H, R, K = sel.shape
        blk = sel.permute(1, 0, 2).to(torch.int32).contiguous()
        if not (any_draft and self.draft_drop > 0):
            return blk
        _, dest = self._draft_keep(l, sel, pos, rb, is_draft, layer)
        out = torch.full((R, H, K + 1), -1, dtype=torch.int32, device=self.dev)
        out.scatter_(2, dest, blk)
        return out[:, :, :K].contiguous()

    # ------------------------------------------------------------------ decisions
    def overflow_flag(self) -> torch.Tensor:
        """The OR of every twin's sticky overflow flag (device, int64 0 / 1)."""
        return torch.stack([t.overflow for t in self.twins]).any().to(torch.int64).view(1)

    def read_tokens(self, lo: int, hi: int) -> List[int]:
        """THE decision read: log slots [lo, hi) and the twins' overflow flag, to the host in ONE read. A set flag VOIDS
        the tick (C, 2026-10-21): every decision of it is refused (TickError) and nothing commits."""
        self.stats["decision_reads"] += 1
        t0 = time.perf_counter()
        vals = torch.cat([self.tok_log[lo:hi], self.overflow_flag()]).tolist()
        if self.timer is not None:
            self.timer.host("read", time.perf_counter() - t0)
        if vals[-1]:
            self.void = True
            self.stats["void_ticks"] += 1
            raise TickError("VOID tick: a twin met a full table (the overflow flag); every decision is refused")
        return vals[:-1]

    def check_overflow(self) -> None:
        """The end of a run (tests and gates; one read): a set overflow flag voids the run's last ticks."""
        if int(self.overflow_flag().item()):
            self.void = True
            self.stats["void_ticks"] += 1
            raise TickError("VOID: a twin met a full table after the last decision read")

    def on_publish(self, b, epoch, upto):
        self.events.append(("pub", int(b), int(upto), int(epoch)))

    def on_rollback(self, b, old_epoch, keep):
        self.events.append(("rb", int(b), int(keep), int(old_epoch)))

    def end_request(self, b: int) -> None:
        self.backend.end_request(int(b))

    @torch.inference_mode()
    def apply_events(self) -> None:
        """The decisions of the tick (module docstring, ROLLBACK and PUBLICATION). Device ops and host bookkeeping."""
        if not self.events:
            return
        if self.void:
            raise TickError("a void tick commits nothing")
        rb, pub, rb_epoch = {}, {}, {}
        for kind, b, p, ep in self.events:
            (rb if kind == "rb" else pub)[b] = p
            if kind == "rb":
                pub.pop(b, None)
                rb_epoch[b] = ep
        self.events = []
        B = self.B
        if rb:                                                                   # 1. the exact banks: ONE commit
            keep = {}
            for b, p in rb.items():
                idx = p - self.last_pos0[b]
                if not 0 <= idx <= self.last_count[b] or p < self.committed[b]:
                    raise TickError("rollback of request %d to %d: outside this tick's rows (%d + %d) or below the "
                                    "committed %d" % (b, p, self.last_pos0[b], self.last_count[b], self.committed[b]))
                keep[b] = idx
            for l in range(self.L):
                SC.rollback_bank(self.twins[l], self.last_ctr[l], keep)
            for b, p in rb.items():
                self.bank_pos[b] = p
                self.committed[b] = p
                self.stats["rollbacks"] += 1
        for b, p in pub.items():
            if p != self.bank_pos[b] or p < self.committed[b]:
                raise TickError("publication of request %d to %d: its exact bank is at %d, committed %d"
                                % (b, p, self.bank_pos[b], self.committed[b]))
            self.committed[b] = p
            self.stats["publications"] += 1
        for b in set(rb) | set(pub):                                             # 2. write-back
            self.writeback(b)
        self.backend.decisions(pub, rb_epoch)                                   # the residency's references
        for b in rb:                                                             # 3. the draft banks
            for l in range(self.L):
                SC.sync_draft_bank(self.twins[l], b, B)
                self.copy_ring(l, b, exact_to_draft=True)
            self.bank_pos[B + b] = self.bank_pos[b]
            self.synced[b] = self.bank_pos[b]
        if self.rebuild == "batched":                                            # [batched-rebuild]
            self._publish_batched(pub)                                           # [batched-rebuild]
            return                                                               # [batched-rebuild]
        for b, p in pub.items():
            dpos = self.bank_pos[B + b]
            if dpos < p:
                raise TickError("request %d: the draft bank (%d) is behind the publication (%d)" % (b, dpos, p))
            for l in range(self.L):
                rebuild_from_own_rows(self.twins[l], b, B, p, dpos)
                self.copy_rows(l, b, range(self.synced[b], p))
            self.synced[b] = p

    def _publish_batched(self, pub, swap: bool = False) -> None:                 # [batched-rebuild]
        """[batched-rebuild] step 4 of apply_events for every publication at once (rebuild="batched"): the host checks
        of every item first, the items' device tensor and copy_rows' index lists ONCE per tick, then per layer ONE
        rebuild_banks_unchecked and ONE copy_rows_many; no host read. swap: NEGATIVE CONTROL (tests only)."""
        B = self.B
        items = []
        for b, p in pub.items():
            dpos = self.bank_pos[B + b]
            if dpos < p:
                raise TickError("request %d: the draft bank (%d) is behind the publication (%d)" % (b, dpos, p))
            items.append((b, p, dpos))
        if not items:
            return
        dev_items = self._to_device(torch.tensor([[b for (b, _, _) in items], [p for (_, p, _) in items],
                                                  [d - p for (_, p, d) in items]], dtype=torch.int64))
        cp = self.rows_to_copy([(b, range(self.synced[b], p)) for (b, p, _) in items])
        with self._span("rebuild"):        # D's profile names it t.rebuild; SG5's REST adds it back (p6's scoring)
            for l in range(self.L):
                rebuild_banks_unchecked(self.twins[l], items, B, dev_items=dev_items, swap=swap)
                if cp is not None:
                    self.copy_rows_many(l, cp)
        for (b, p, _) in items:
            self.synced[b] = p

    def rows_to_copy(self, per_bank):                                            # [batched-rebuild]
        """[batched-rebuild] copy_rows' index lists for every (bank b, positions) of a tick in ONE device tensor
        (3, Q): the bank, the source (exact ring) row, the destination (draft ring) row; None when there is no row."""
        bb, src, dst = [], [], []
        for b, positions in per_bank:
            for p in positions:
                bb.append(int(b))
                src.append(self.ring_slot(False, p) * BLOCK + p % BLOCK)
                dst.append(self.ring_slot(True, p) * BLOCK + p % BLOCK)
        if not bb:
            return None
        return self._to_device(torch.tensor([bb, src, dst], dtype=torch.int64))

    def copy_rows_many(self, l: int, cp) -> None:                                # [batched-rebuild]
        """[batched-rebuild] copy_rows for every (bank, position) of cp at once: the draft ring rows take the exact
        ring rows (distinct destinations; the sources are the exact ring, never a destination)."""
        for st in (self.store_k[l], self.store_v[l], self.store_bias[l]):
            st[cp[0], cp[2]] = st[cp[0], cp[1]]

    def writeback(self, b: int) -> None:
        """Every block whose 64 positions are committed goes from its exact ring page to the backend's history (the
        ResidentStore: its historical slot; the Residency: the pinned host rows, registered with history_written)."""
        while BLOCK * (self.wb[b] + 1) <= self.committed[b]:
            self.backend.commit_block(b, self.wb[b])
            self.wb[b] += 1
            self.stats["writebacks"] += 1

    def copy_ring(self, l: int, b: int, exact_to_draft: bool = True) -> None:
        e0, d0 = (self.NB + RING_EXACT) * BLOCK, (self.NB + RING_DRAFT) * BLOCK
        for st in (self.store_k[l], self.store_v[l], self.store_bias[l]):
            st[b, d0:d0 + 2 * BLOCK].copy_(st[b, e0:e0 + 2 * BLOCK])

    def copy_rows(self, l: int, b: int, positions) -> None:
        """The draft ring rows of `positions` take the exact ring rows (a publication)."""
        ps = list(positions)
        if not ps:
            return
        src = torch.tensor([self.ring_slot(False, p) * BLOCK + p % BLOCK for p in ps], dtype=torch.int64)
        dst = torch.tensor([self.ring_slot(True, p) * BLOCK + p % BLOCK for p in ps], dtype=torch.int64)
        src, dst = self._to_device(src), self._to_device(dst)
        for st in (self.store_k[l], self.store_v[l], self.store_bias[l]):
            st[b, dst] = st[b, src]

    def mirror_failures(self) -> list:
        """TESTS AND GATES (host reads): the (layer, bank, twin pos, host mirror) where a twin's position counter
        differs from the host mirror the row table uses (a bookkeeping check after a run)."""
        bad = []
        for l, t in enumerate(self.twins):
            for bank, p in enumerate(t.pos.tolist()):
                if p != self.bank_pos[bank]:
                    bad.append((l, bank, p, self.bank_pos[bank]))
        return bad

    # ------------------------------------------------------------------ serial mode (the BI serial decode)
    @torch.inference_mode()
    def run_serial(self, tokens) -> SimpleNamespace:
        """One exact row per request at its next position, committed at once (no draft, no rollback): the BI serial
        decode in this engine. tokens: B ints or Refs. The ResidentStore only (the reference of SG0)."""
        if not isinstance(self.backend, ResidentStore):
            raise TickError("the serial mode runs on the ResidentStore")
        rows = [SimpleNamespace(b=b, kind="v", pos=self.bank_pos[b]) for b in range(self.B)]
        out = self.run_rows(rows, tokens)
        for b in range(self.B):
            self.committed[b] = self.bank_pos[b]
            self.writeback(b)
            self.synced[b] = self.committed[b]
        return out


# =====================================================================================================================
# storage backends (the SG5 seam)
# =====================================================================================================================
class ResidentStore:
    """Every committed block resident in the historical slot named by its block id (SG0-SG4; module docstring)."""

    name = "resident"

    def __init__(self, eng: TickEngine):
        self.eng = eng

    def max_blocks(self) -> int:
        return self.eng.NB

    def begin_tick(self) -> None:
        pass

    def prepare(self, rows):
        return None

    def pages(self, l, ctx, sel, layer):
        table, npg = self.eng.page_table(l, sel, ctx.pos, ctx.rb, ctx.prev_slot, ctx.tail_slot, ctx.is_draft,
                                         ctx.any_draft, layer)
        return table, npg, ctx.last, ctx.rb32, None

    def page_keys(self, l, ctx, sel, layer, plan=None):
        """[sharedkv] the block id of every page_table entry (TickEngine.page_keys). plan: unused here ([sharedkv2])."""
        return self.eng.page_keys(l, sel, ctx.pos, ctx.rb, ctx.is_draft, ctx.any_draft, layer)

    def after_attention(self, l, plan) -> None:
        pass

    def end_tick(self) -> None:
        pass

    def commit_block(self, b: int, blk: int) -> None:
        eng = self.eng
        src = eng.ring_slot(False, blk * BLOCK) * BLOCK
        for l in range(eng.L):
            for st in (eng.store_k[l], eng.store_v[l], eng.store_bias[l]):
                st[b, blk * BLOCK:(blk + 1) * BLOCK].copy_(st[b, src:src + BLOCK])

    def decisions(self, pub, rb_epoch) -> None:
        pass

    def end_request(self, b: int) -> None:
        pass


def cohort_of(i: int, K: int) -> int:
    """The cohort m whose verify row i checks d_i (and writes position E + i - 1): r2_spec_core.cohort_range."""
    return 1 if i <= K - 1 else i // K + 1


class ResidencyBackend:
    """Piece 5's Residency (or ResidencySB) as the tick's storage (module docstring, STORAGE BACKENDS; C's answers
    Q-S1 .. Q-S5, 2026-10-21). svc: the service over THIS engine's store (svc.n_hist == eng.NB; its store_b is
    eng.store_bias, its cis tables the exact banks of the twins); host_k / host_v: the pinned host rows per layer
    (B, N_host, H, D); new_event(): an event recorded on the current stream (CUDA: torch.cuda.Event().record(); CPU
    tests: a completed event); K: the cadence. The uid of request b is b."""

    name = "residency"

    def __init__(self, eng: TickEngine, svc, host_k, host_v, new_event: Callable[[], Any], K: int):
        if svc.n_hist != eng.NB or svc.B != eng.B or svc.L != eng.L or svc.H != eng.H:
            raise TickError("the service's store (B %d, L %d, H %d, C + X %d) is not the engine's (B %d, L %d, H %d, NB %d)"
                            % (svc.B, svc.L, svc.H, svc.n_hist, eng.B, eng.L, eng.H, eng.NB))
        self.eng, self.svc, self.host_k, self.host_v = eng, svc, host_k, host_v
        self.new_event, self.K = new_event, int(K)
        from .r2_residency_sb import ResidencySB             # S-B and S-C (its subclass): the selection stays on the
        self.sb = isinstance(svc, ResidencySB)               # device (C's 41bc238 sets S-C's .mirrors to None)
        self.keys: Dict[int, tuple] = {}                     # b -> the verify key of the current tick
        self.ended = set()
        self.n_host = int(host_k[0].shape[1])
        for b, S in enumerate(eng.S_list):
            svc.begin_request(b, b, hist_blocks=S // BLOCK)
        if getattr(eng.ops.attn, "needs_page_key", False) and hasattr(svc, "want_keys"):   # [sharedkv2] S-C / B2
            svc.want_keys = True                             # resolve builds plan.page_key (page_keys)
        if callable(getattr(svc, "watch_flag", None)):       # S-C (C's 0d3a94a, Q-T5): the twins' overflow flags ride
            for l, t in enumerate(eng.twins):                 # in its ONE read per tick; a raised flag refuses that tick
                svc.watch_flag("twin_overflow_%d" % l, t.overflow)

    def max_blocks(self) -> int:
        return self.n_host // BLOCK

    def begin_tick(self) -> None:
        self.svc.poll()

    def prepare(self, rows):
        """The service's rows of the tick (host, once per tick): the cohort key; the history boundary (block T - 1 is
        served by the service once its write-back is registered, else from the row's ring); the tail pages."""
        from .r2_residency import Row as SvcRow
        eng, out = self.eng, []
        self.keys = {}
        for row in rows:
            b, p, draft = int(row.b), int(row.pos), row.kind == "d"
            T = p // BLOCK
            if draft:
                key = (b, int(row.epoch), cohort_of(int(row.i), self.K))
            else:
                key = (b, int(row.epoch), int(row.m))
                self.keys[b] = key
            if eng.wb[b] >= T:
                hb, tail = T, [(T, eng.ring_slot(draft, p))]
            else:
                hb, tail = T - 1, [(T - 1, eng.ring_slot(draft, p - BLOCK)), (T, eng.ring_slot(draft, p))]
            out.append(SvcRow(b, row.kind, key, p, hb, tail))
        return out

    def pages(self, l, ctx, sel, layer):
        plan = self.svc.resolve(l, ctx.rows, sel if self.sb else sel.cpu())
        dev = self.eng.dev
        return (plan.page_table.to(dev).contiguous(), plan.num_pages.to(dev).contiguous(),
                plan.last_len.to(dev).contiguous(), plan.seq_idx.to(dev).contiguous(), plan)

    def page_keys(self, l, ctx, sel, layer, plan=None):
        """[sharedkv2] the plan's own block-id tensor (plan.page_key, (R, H, P) int32 on the device, aligned with
        plan.page_table): S-C / B2 build it in resolve when want_keys is set (__init__ sets it when ops.attn has
        needs_page_key; r2_residency_sc.page_keys_of). v1 / S-B build none: refused, never guessed."""
        key = getattr(plan, "page_key", None)
        if key is None:
            raise TickError("sharedkv: this Residency service gave no page keys (S-C / B2 with want_keys only; use "
                            "sc / b2, the ResidentStore, or per_row)")
        return key.to(self.eng.dev).contiguous()

    def after_attention(self, l, plan) -> None:
        self.svc.read_done(l, plan, self.new_event())

    def end_tick(self) -> None:
        self.svc.end_tick()

    def commit_block(self, b: int, blk: int) -> None:
        """Q-S1: the exact ring page of block blk to the pinned host rows of every layer (on the current stream), ONE
        event after them, history_written(uid, blk, event). The tick never waits on the host."""
        eng = self.eng
        if (blk + 1) * BLOCK > self.n_host:
            raise TickError("block %d of request %d is past the host rows (%d)" % (blk, b, self.n_host))
        src = eng.ring_slot(False, blk * BLOCK) * BLOCK
        r0 = blk * BLOCK
        for l in range(eng.L):
            self.host_k[l][b, r0:r0 + BLOCK].copy_(eng.store_k[l][b, src:src + BLOCK], non_blocking=True)
            self.host_v[l][b, r0:r0 + BLOCK].copy_(eng.store_v[l][b, src:src + BLOCK], non_blocking=True)
        self.svc.history_written(b, blk, self.new_event())

    def decisions(self, pub, rb_epoch) -> None:
        """Q-S2 / Q-S3: a publication decided(key); a rejection decided(key), then cancel_epoch(uid, old epoch)."""
        for b in pub:
            self.svc.decided(self.keys[b])
        for b, ep in rb_epoch.items():
            self.svc.decided(self.keys[b])
            self.svc.cancel_epoch(b, ep)

    def end_request(self, b: int) -> None:
        if b not in self.ended:
            self.ended.add(b)
            self.svc.end_request(b)


class TickTimer:
    """SG5's instrumentation (module docstring): per tick, CUDA event pairs per category (or perf_counter on the CPU),
    and host seconds per category. Read once by report()."""

    def __init__(self, cuda: bool):
        self.cuda = bool(cuda)
        self.tick = -1
        self.spans: List[tuple] = []                         # (tick, cat, start, end)
        self.hosts: Dict[tuple, float] = {}                  # (tick, cat) -> seconds

    def next_tick(self) -> None:
        self.tick += 1

    def host(self, cat: str, sec: float) -> None:
        k = (self.tick, cat)
        self.hosts[k] = self.hosts.get(k, 0.0) + float(sec)

    @contextlib.contextmanager
    def span(self, cat: str):
        if self.cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            try:
                yield
            finally:
                e1.record()
                self.spans.append((self.tick, cat, e0, e1))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.spans.append((self.tick, cat, t0, time.perf_counter()))

    def report(self) -> Dict[str, Any]:
        """Per category: the per-tick device ms (sum of its spans) and host ms, as lists over the ticks (ONE sync)."""
        if self.cuda:
            torch.cuda.synchronize()
        n = self.tick + 1
        dev: Dict[str, List[float]] = {}
        for (t, cat, a, b) in self.spans:
            if t < 0:
                continue
            ms = a.elapsed_time(b) if self.cuda else 1e3 * (b - a)
            dev.setdefault(cat, [0.0] * n)[t] += ms
        host: Dict[str, List[float]] = {}
        for (t, cat), s in self.hosts.items():
            if t < 0:
                continue
            host.setdefault(cat, [0.0] * n)[t] += 1e3 * s
        return dict(ticks=n, device_ms=dev, host_ms=host)


# =====================================================================================================================
# the driver
# =====================================================================================================================
def run_round(engine: TickEngine, first_tokens: Sequence[int], max_new: Sequence[int], K: int = 3, LAMBDA: int = 7,
              teacher: Callable = None, reject_at: Callable = None, on_tick: Callable = None, max_ticks: int = 100000,
              ready: Callable = None):
    """The speculative round on the engine until every request has max_new committed tokens (the first token
    included, as r2_spec_core.Round counts them). on_tick(rows, tokens, out) sees every tick's rows, their input
    tokens (ints or Refs) and run_rows' result before the decision. ready(rnd, b, cohort) -> bool gates a TRIGGERED
    cohort (default: always; phase_ready staggers the requests: the gate SG3). A tick that plans no row (every live
    request waits for a gated cohort at its lead bound) runs nothing and counts as a tick. Returns the TickRound."""
    rnd = TickRound(first_tokens, engine.S_list, engine.read_tokens, K=K, LAMBDA=LAMBDA, max_new=max_new,
                    on_publish=engine.on_publish, on_rollback=engine.on_rollback, teacher=teacher, reject_at=reject_at)
    if ready is not None:
        rnd.ready = lambda b, c: bool(ready(rnd, b, c))
    empty = 0
    while not all(r.done for r in rnd.reqs):
        if rnd.ticks >= max_ticks:
            raise TickError("no progress after %d ticks" % max_ticks)
        rows = rnd.plan_tick()
        if not rows:
            empty += 1
            if empty > 64:
                raise TickError("64 ticks in a row plan no row (tick %d)" % rnd.ticks)
            rnd.finish_tick([], [], engine.n_tok)
            continue
        empty = 0
        tokens = [row.token for row in rows]
        if engine.timer is not None:
            engine.timer.next_tick()
        with engine._span("tick"):
            out = engine.run_rows(rows, tokens)
        if on_tick is not None:                                  # the gates' checks: outside the tick's spans
            on_tick(rows, tokens, out)
        with engine._span("tick"):
            rnd.finish_tick(rows, out.slots, out.hi)
            engine.apply_events()
        for b, r in enumerate(rnd.reqs):
            if r.done:
                engine.end_request(b)
    engine.check_overflow()
    return rnd


def phase_ready(period: int = 3):
    """The gate SG3's PHASE MIX: request b's TRIGGERED cohort may run only at ticks t with (t + b) % period == 0, so the
    requests verify at different ticks, a request drafts while others verify, and a request whose cohort waits reaches
    its lead bound and verifies without a draft row of its own."""
    if period < 1:
        raise TickError("period >= 1")
    return lambda rnd, b, c: (rnd.ticks + b) % period == 0


def compositions(rows) -> List[tuple]:
    """Per row, (kind, own draft row in the tick, another request's verify rows in the tick, another request's draft
    row in the tick, R): the composition classes of the gate SG3."""
    R = len(rows)
    dr = {r.b for r in rows if r.kind == "d"}
    vr = {r.b for r in rows if r.kind == "v"}
    return [(r.kind, r.b in dr, bool(vr - {r.b}), bool(dr - {r.b}), R) for r in rows]


# =====================================================================================================================
# the gate SG4's NEGATIVE CONTROLS (each must make a verify row differ from the BI serial decode, or the gate says
# that it could not see it)
# =====================================================================================================================
def topk_select(pq: torch.Tensor, pc: torch.Tensor, qk_select: int, k: int) -> torch.Tensor:
    """SG4 (c): the SHIPPED two-step rule with torch.topk(sorted=False) (nosa_llama decode graph): the set at a tie can
    depend on the row count and the algorithm. The result sorted ascending (the canonical page order), so only the
    tie rule differs from r2_spec_core.select_blocks."""
    iq = torch.topk(pq, qk_select, dim=-1, sorted=False).indices
    mask = torch.zeros_like(pc, dtype=torch.bool)
    mask.scatter_(2, iq, True)
    pc2 = pc.masked_fill(mask, float("inf"))
    return torch.sort(torch.topk(pc2, k, dim=-1, sorted=False).indices, dim=-1).values


def high_select(pq: torch.Tensor, pc: torch.Tensor, qk_select: int, k: int) -> torch.Tensor:
    """SG4 (d), a CONTROL of the planted ties: the two-step rule with ties broken by the HIGHER block id (a stable sort
    of the reversed rows). Under a tie across a cut it selects another block than select_blocks, by construction."""
    def top(x, n):
        M = x.shape[-1]
        return (M - 1) - torch.sort(x.flip(-1), dim=-1, descending=True, stable=True).indices[..., :n]
    iq = top(pq, qk_select)
    mask = torch.zeros_like(pc, dtype=torch.bool)
    mask.scatter_(2, iq, True)
    pc2 = pc.masked_fill(mask, float("inf"))
    return torch.sort(top(pc2, k), dim=-1).values


def tie_stats(pq: torch.Tensor, pc: torch.Tensor, qk_select: int, k: int, planted=None) -> torch.Tensor:
    """(5,) int64 on the device: the (head, row) pairs; those with EQUAL pooled q scores at ranks QK_SELECT - 1 /
    QK_SELECT (the first cut); those with equal pooled cis at ranks k - 1 / k after the first step's +inf (the top-k
    cut); the planted pairs (`planted`, (H, R) bool, from plant_tie); the planted pairs that tie across the top-k cut.
    A tie across a cut is where a tie rule decides the selection; finite values only."""
    q, c = pq.float(), pc.float()
    vq, iq = torch.sort(q, dim=-1, descending=True, stable=True)
    tq = (vq[..., qk_select - 1] == vq[..., qk_select]) & torch.isfinite(vq[..., qk_select - 1])
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask.scatter_(-1, iq[..., :qk_select], True)
    vc = torch.sort(c.masked_fill(mask, float("inf")), dim=-1, descending=True).values
    tc = (vc[..., k - 1] == vc[..., k]) & torch.isfinite(vc[..., k - 1])
    pl = torch.zeros_like(tc) if planted is None else planted
    n = torch.full((), tq.numel(), dtype=torch.int64, device=pq.device)
    return torch.stack([n, tq.sum().to(torch.int64), tc.sum().to(torch.int64), pl.sum().to(torch.int64),
                        (pl & tc).sum().to(torch.int64)])


def plant_tie(pq: torch.Tensor, pc: torch.Tensor, rows: torch.Tensor, qk_select: int, k: int,
              apply: bool = True) -> torch.Tensor:
    """SG4c PLANTED (in place on pc): for the rows marked in `rows` (R,) bool, every head, the block ranked k + 1 by
    the second step (after the first step's +inf) gets the k-th block's pooled value: two blocks tie across the top-k
    cut, so the stable rule keeps the lower block id and any other tie rule may keep the other one. Only where both
    are real blocks (finite values). Returns the planted pairs (H, R) bool. apply False (a NEGATIVE CONTROL of the
    precondition): the pairs are reported, pc is not changed."""
    q, c = pq.float(), pc.float()
    iq = torch.sort(q, dim=-1, descending=True, stable=True).indices[..., :qk_select]
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask.scatter_(-1, iq, True)
    vc, ic = torch.sort(c.masked_fill(mask, float("inf")), dim=-1, descending=True, stable=True)
    v = vc[..., k - 1:k]
    j = ic[..., k:k + 1]
    ok = rows.view(1, -1, 1) & torch.isfinite(v) & torch.isfinite(vc[..., k:k + 1])
    if apply:
        pc.scatter_(-1, j, torch.where(ok, v.to(pc.dtype), pc.gather(-1, j)))
    return ok[..., 0]


class BatchKeyedSplit:
    """SG4 (b): piece 3's attention with a split size chosen by the BATCH (8 pages per split for <= `rows` rows, else
    4), as a heuristic split-KV kernel does; the BI serial decode's calls have `rows` = B rows."""

    def __init__(self, attn, rows: int):
        self.attn, self.rows = attn, int(rows)
        self.build_page_table = attn.build_page_table

    def bi_paged_decode(self, q, *args, **kw):
        kw["pages_per_split"] = 8 if q.shape[0] <= self.rows else 4
        return self.attn.bi_paged_decode(q, *args, **kw)


def forced_schedule(K: int, block: int = BLOCK):
    """The gate SG1's rejection schedule (a function of positions and counts only, never of token values). Row i of a
    cohort wrote position E + i - 1; a rejection at index j keeps the positions < E + j (the rows a .. j) and rolls
    back the rows j + 1 .. b. Per decided cohort, in this order:
      cross_new / cross_old  the cohort's rows CROSS a block boundary X: rejected at the row that wrote X (the new
                             block's first row is kept, the rest rolled back) and at the row before it (every row of the
                             new block rolled back), in turn over all requests (one global counter);
      steer                  the next boundary X is 1 or 2 positions after the cohort's last row: rejected at
                             j = X - 1 - E, so the next epoch starts at X - 1 and its first cohort crosses X;
      cycle                  else the n-th such cohort of the request (size L) is rejected at offset n % (L + 1);
                             offset L = accepted whole.
    Every boundary a request passes is crossed by a cohort (a cohort cannot pass X without crossing it or ending at
    X - 1 or X - 2). Returns (reject_at, log): log lists (b, a, b_end, E, offset or None, kind)."""
    n_dec: Dict[int, int] = {}
    n_cross = [0]
    log = []

    def reject_at(b, c, E):
        L = c.b - c.a + 1
        p_first, p_last = E + c.a - 1, E + c.b - 1                     # positions the cohort's rows wrote
        X = (p_last // block + 1) * block                              # the next boundary after the last row
        if p_first // block != p_last // block:
            first_new = (p_last // block) * block
            i_new = first_new - E + 1                                  # the row that wrote the new block's first row
            kind = "cross_new" if n_cross[0] % 2 == 0 else "cross_old"
            n_cross[0] += 1
            j = i_new if kind == "cross_new" else i_new - 1
        elif X - p_last in (1, 2) and c.a <= X - 1 - E <= c.b:
            kind, j = "steer", X - 1 - E
        else:
            k = n_dec.get(b, 0)
            n_dec[b] = k + 1
            off = k % (L + 1)
            kind, j = "cycle", (None if off == L else c.a + off)
        log.append((b, c.a, c.b, E, None if j is None else j - c.a, kind))
        return j
    return reject_at, log


def schedule_coverage(log, K: int) -> dict:
    """What a forced_schedule log covered: the cycle rejection offsets, full acceptances, the two boundary kinds."""
    offs = sorted({o for (_, _, _, _, o, kind) in log if kind == "cycle" and o is not None})
    return dict(offsets=offs, full=sum(1 for e in log if e[4] is None),
                cross_new=sum(1 for e in log if e[5] == "cross_new"),
                cross_old=sum(1 for e in log if e[5] == "cross_old"),
                steer=sum(1 for e in log if e[5] == "steer"), decisions=len(log),
                complete=(set(range(K)) <= set(offs) and any(e[4] is None for e in log)
                          and any(e[5] == "cross_new" for e in log) and any(e[5] == "cross_old" for e in log)))


def simulate_schedule(S_list: Sequence[int], max_new: Sequence[int], K: int = 3, LAMBDA: int = 7, block: int = BLOCK):
    """The teacher-forced round's schedule WITHOUT a model (its decisions depend on positions and counts only): the
    gate checks its coverage before it loads the model, and compares it with the real run's log afterwards."""
    reject_at, log = forced_schedule(K, block)
    n = [0]

    def read(lo, hi):
        return [0] * (hi - lo)
    rnd = TickRound([0] * len(S_list), S_list, read, K=K, LAMBDA=LAMBDA, max_new=max_new,
                    teacher=lambda b, p: 0, reject_at=reject_at)
    while not all(r.done for r in rnd.reqs):
        rows = rnd.plan_tick()
        if not rows or rnd.ticks > 100000:
            raise TickError("the simulated round makes no progress")
        slots = list(range(n[0], n[0] + len(rows)))
        n[0] += len(rows)
        rnd.finish_tick(rows, slots, n[0])
    return log, rnd
