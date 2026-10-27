"""Phase 2 of KS/analysis/plan/PLAN_speckv_on_cached_nosa.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), rungs L3 and L4:
THE SPECULATIVE ROUND RUNNER on the per-request engine (paper sections 3 and 3.4). OPT-IN: only the cell
benchmarks/Efficiency/r2_spec_cell.py (--spec K) and the CPU tests import this module. No engine module imports it.
NO NEW KERNEL: the round composes the hooks of Phase 1 and the shipped kernels as they are.

THE PARTS
  ragged_engine     (Task 1.3) the per-request step. Every draft step and every serial verify step is
                    ragged_engine.decode_step, unchanged.
  ragged_rollback   (Task 1.5) the journal. A group holds the tentative steps of one round. Draft steps are always
                    undone; the verify keeps the accepted prefix and the correction (commit / truncate, per request).
  DraftResident     (Task 1.1 on this engine, this module) the RESIDENT-ONLY draft residency: a hook on the body's
                    decode_fixed. The step runs the shipped diff, victim pool and map commit. It does NOT run the
                    two host gathers. A slot that still needs host rows is SKIPPED: its map entry becomes -1 (the next
                    step loads it again) and a SCRATCH copy of the attended bias gets MASK_BIAS (-3e4) on its 64 rows;
                    attention reads the scratch, never the live bias. One exception keeps the draft close to the
                    target: the block that a step of the SAME group filled is in GPU memory only (the rollback's
                    staging block, DEFERRED WRITE-BACK): the draft loads it through the rollback's staging redirect
                    (the gathers restricted to that block; host flag `redirect`). Draft rows never reach the host
                    rows: the rollback defers every fill of a group and drops the fills of undone steps.
  verify_rows       (Task 1.4 route 1) L4's joint pass: K verify rows and the draft row in ONE forward pass. EXACT:
                    the serial residency per row (row K through DraftResident); FAST: one residency, the extras store
                    of the verify rows' union, the draft row resident-only (VerifyOps(draft_last=True)).
  KVMeter           (this module) the KV that the verify fetches from the host: blocks (device count) and, on probe
                    rounds, CUDA-event spans of the host gathers.

L3, ONE ROUND (serial verify; per request b at committed position p_b with its last committed token t_b, not fed yet)
  1. begin; K DRAFT steps: step j feeds x_j at p_b + j (x_0 = t_b, x_j = d_j) and gives d_{j+1} = argmax;
     reject_all (every draft step undone, the LRU clock back).
  2. begin; K VERIFY steps, the shipped ragged step: step j feeds x_j at p_b + j and gives g_{j+1} = argmax.
  3. ACCEPT (greedy, paper 3.4): n_b = the longest prefix with g_i == d_i (i = 1 .. K); request b commits
     m_b = min(n_b + 1, K) tokens: g_1 .. g_{m_b} (the accepted drafts and, on a reject, the target's correction
     g_{n_b + 1}). FULL ACCEPT gives K tokens and NO bonus token (no step fed d_K). commit(m_b): request b keeps its
     first m_b verify steps, which fed exactly its committed prefix (t_b, d_1 .. d_{m_b - 1}).
  The engine is then in the state of m_b shipped steps per request, so the next round is again the shipped decode.

L4, ONE ROUND (joint pass; the group holds the joint pass and the K - 1 draft steps after it, 2 K tentative steps)
  0. (first round only) K draft steps in their own group, reject_all: the drafts D_b = d_1 .. d_K.
  1. begin; ONE JOINT PASS of R = K + 1 rows: rows 0 .. K - 1 verify (they feed t_b, D_b[0 .. K - 2] at p_b ..
     p_b + K - 1), row K is the DRAFT ROW (it feeds D_b[K - 1] at p_b + K: the first draft of the next round).
  2. ACCEPT as L3 (n_b over the VALID drafts; m_b = min(n_b + 1, K)); commit the tokens g_1 .. g_{m_b}.
  3. truncate: a request with n_b = K (full accept) keeps all R rows (its draft row stays tentative); another request
     goes back to m_b rows now (the draft row and the rejected rows undone).
  4. K - 1 DRAFT steps: a full-accept request feeds its draft row's token e_0 at p_b + K + 1, then its drafts; another
     request feeds its correction at p_b + m_b, then its drafts. Next drafts: full accept [e_0, e_1 .. e_{K-1}] (K
     valid); else [d'_1 .. d'_{K-1}, none] (K - 1 valid: the last draft is missing, so that request cannot accept the
     K-th row and its next draft row is not used; it still commits up to K tokens per round).
  5. commit: a full-accept request keeps its K verify rows (the draft row and the drafts undone); another request keeps
     its m_b rows (its later steps undone; the truncated ones are marked and not undone twice).
  K >= 2 (with K = 1 a request that rejects once never drafts again).

EXACTNESS (the token proof; plan, HUMAN DECISION part 3). Every committed token is a target argmax of a step (L3) or a
verify row (L4) whose inputs are the committed prefix, on an engine state that the rollback restored bit for bit. With
the verify in the shipped bits (L3: ragged step, gemm natural, twin torch; L4: EXACT joint pass, gemm rowgroup) the
committed tokens are the shipped greedy decode, bitwise, for ANY draft tokens (a draft only changes how many tokens a
round commits). The LRU clock of the pool runs ahead of the shipped one after a partial commit (commit keeps it
monotone), so the absolute pool ages differ while their ORDER, the only thing pool_update compares, is the same.

THE RUN. One PLAIN step first (warm, default 1): until a step loads them, the window slots hold torch.empty bytes after
the prefill, and a draft reads its skipped slots through the masked bias (0 x NaN = NaN). Then the rounds.

HOST SYNCS: one per round, after the round (the committed counts and the twin positions: the termination and the host
position mirror that the draft's redirect flag reads), and one after the warm step. None inside a round.

A FINDING OF THE DESIGN (L4, as the task defines it): after its first reject a request has K - 1 valid drafts per round
(its draft row fed no valid token), so its row K - 1 always gives the target's own token and its draft row is never
used again. Both cases commit at most K tokens per round (the full-accept case K accepted drafts; the other case K - 1
accepted drafts and the target's token), so the draft row adds no committed token in this round structure: it only
replaces one draft step while a request has accepted every draft so far.

NEGATIVE CONTROLS (tests only; SpecOps.neg): commit_draft_at_reject (a reject commits the draft token, not the
correction), keep_rejected (the state keeps every verify step), bonus_on_full_accept (a full accept also counts the draft
row's token as committed), miscount (the statistics count accepted drafts, not committed tokens).
"""
from __future__ import annotations

import time

import torch

from . import ragged_engine as RE
from . import ragged_rollback as RR

MASK_BIAS = -3.0e4                     # avail_policy.py:113, verify_rows.MASK_BIAS: finite on purpose
LEVELS = ("L3", "L4")
L4_MODES = ("exact", "fast")
L4_GEMMS = ("natural", "rowgroup", "exact")
NEG_KINDS = (None, "commit_draft_at_reject", "keep_rejected", "bonus_on_full_accept", "miscount")
K_MAX = {"L3": 32, "L4": 31}           # L4: verify_rows.K_MAX; the rollback group holds 2 K <= 64 steps


class SpecError(RuntimeError):
    pass


def accept_prefix(g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(B, K) target argmaxes g and (B, K) drafted tokens d -> (B,) int64 n: the longest prefix with g[b, i] ==
    d[b, i]. A missing draft (-1) never matches. Device ops only (ragged_rollback.accept_lengths with U = K)."""
    if g.dim() != 2 or g.shape != d.shape:
        raise SpecError("accept_prefix: shapes %s and %s" % (tuple(g.shape), tuple(d.shape)))
    return torch.cumprod((g == d).to(torch.int64), dim=1).sum(1)


def _no_gather(*args, **kwargs) -> None:
    return None


# ------------------------------------------------------------------ the resident-only draft (Task 1.1, hook a)
class DraftResident:
    """The draft residency on the per-request engine (module docstring). It wraps rs.body.decode_fixed (attach it
    AFTER the rollback, detach it BEFORE). begin(mode, redirect) / end() bracket one step or one joint pass:
      mode "all"  every decode_fixed call of the bracket is a draft residency (a ragged draft step)
      mode r      only the r-th call of each layer (a joint pass in EXACT mode: row r)
    A draft call is refused outside an open rollback group (its fill would reach the host rows: the dr1 leak).
    Counters (device): skipped slots and redirected slots per layer; miss_ids[l] = the skipped block ids of the last
    draft call of layer l (H, B, topk; -1 elsewhere): the list that a background prefetch (rung L5) would fetch."""

    def __init__(self, rs, rb):
        if getattr(rs, "draft", None) is not None:
            raise SpecError("this RaggedState has a DraftResident already")
        if rb is None or getattr(rs, "rollback", None) is not rb:
            raise SpecError("DraftResident needs the RaggedState's Rollback (attached first)")
        self.rs, self.rb = rs, rb
        eng0 = rs.cache.layers[0].cache_engine
        self.L, self.B = int(rs.L), int(rs.B)
        self.H, self.topk, self.bs = int(eng0.head_num), int(eng0.topk), int(eng0.block_size)
        self.pool = bool(rs.pool)
        dev = eng0._k_gpu.device
        self.scratch = torch.empty_like(eng0._kv_bias_gpu)             # one bias scratch for every layer (one stream)
        self.scratch_att = self.scratch[:, :self.topk * self.bs] if self.pool else self.scratch
        self.minus1 = torch.tensor(-1, dtype=torch.int64, device=dev)
        self.miss_ids = [torch.full((self.H, self.B, self.topk), -1, dtype=torch.int64, device=dev)
                         for _ in range(self.L)]
        self.skipped = torch.zeros((self.L,), dtype=torch.int64, device=dev)
        self.redirected = torch.zeros((self.L,), dtype=torch.int64, device=dev)
        self.calls = 0                                                   # draft decode_fixed calls (host count)
        self.steps = 0                                                   # draft brackets (steps or rows)
        self.mode = None
        self.redirect = False
        self._calls = [0] * self.L
        self._lay, self._live = {}, {}
        for l, lay in enumerate(rs.cache.layers):
            e = lay.cache_engine
            self._lay[e._k_gpu.data_ptr()] = l
            if self.pool:
                self._live[l] = e._kv_bias_gpu_att                       # the live attended bias view (restored)
        self._orig = None
        self.attach()

    def attach(self) -> None:
        body = self.rs.body
        self._orig = dict(fn=body.decode_fixed, own="decode_fixed" in body.__dict__)
        body.decode_fixed = self._hook
        self.rs.draft = self

    def detach(self) -> None:
        if self.mode is not None:
            raise SpecError("detach inside a draft bracket: end() first")
        body = self.rs.body
        if body.__dict__.get("decode_fixed") != self._hook:
            raise SpecError("detach: another hook wraps decode_fixed after the draft hook")
        if self._orig["own"]:
            body.decode_fixed = self._orig["fn"]
        else:
            del body.decode_fixed
        self.rs.draft = None

    def begin(self, mode, redirect: bool = False) -> None:
        if self.mode is not None:
            raise SpecError("begin inside a draft bracket")
        if mode != "all" and not (isinstance(mode, int) and mode >= 0):
            raise SpecError("mode must be 'all' or a row index >= 0, got %r" % (mode,))
        if not self.rb.tentative:
            raise SpecError("a draft outside a rollback group: its rows could reach the host rows (the dr1 leak)")
        self.mode = mode
        self.redirect = bool(redirect)
        self._calls = [0] * self.L
        self.steps += 1

    def end(self) -> None:
        """Every layer's attended bias view back to the live one; the bracket is closed."""
        for l, view in self._live.items():
            self.rs.cache.layers[l].cache_engine._kv_bias_gpu_att = view
        self.mode = None

    def _hook(self, eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff, gather_k,
              gather_v_bias):
        l = self._lay[eng._k_gpu.data_ptr()]
        i = self._calls[l]
        self._calls[l] = i + 1
        mode = self.mode
        if mode is None or (mode != "all" and i != mode):
            return self._orig["fn"](eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff,
                                    gather_k, gather_v_bias)
        staged = self.rb.T[l]["id"].view(1, self.B, 1) if self.redirect else None
        if staged is None:
            gk, gv = _no_gather, _no_gather
        else:
            box = {}

            def gk(gpu, cpu, ids, bs):                                   # the staged block only (GPU staging rows)
                box["ids"] = torch.where((ids == staged) & (staged >= 0), ids, self.minus1)
                gather_k(gpu, cpu, box["ids"], bs)

            def gv(gpu_v, cpu_v, bias_buf, bias, ids, bs):
                gather_v_bias(gpu_v, cpu_v, bias_buf, bias, box["ids"], bs)
        out = self._orig["fn"](eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff, gk, gv)
        lm = eng._load_mask                                              # after pool_update: the host misses
        skip = lm >= 0
        if staged is not None:
            red = skip & (lm == staged) & (staged >= 0)
            skip = skip & ~red
            self.redirected[l] += red.sum()
        eng._block_map.masked_fill_(skip, -1)                           # the next step loads the skipped blocks
        self.miss_ids[l].copy_(torch.where(skip, lm, self.minus1))
        self.skipped[l] += skip.sum()
        rows = skip.permute(1, 2, 0).unsqueeze(2).expand(self.B, self.topk, self.bs, self.H)
        rows = rows.reshape(self.B, self.topk * self.bs, self.H)
        self.scratch_att.copy_(self._live[l] if self.pool else eng._kv_bias_gpu)
        self.scratch_att.masked_fill_(rows, MASK_BIAS)
        if self.pool:
            eng._kv_bias_gpu_att = self.scratch_att                      # attention reads the attended views
        self.calls += 1
        return out[0], out[1], self.scratch, out[3]                      # without a pool attention reads these


# ------------------------------------------------------------------ the KV that the verify fetches
class KVMeter:
    """Wraps the engine's two host gathers (rs.ops.kernels gather_k / gather_v_bias). Attach it BEFORE the rollback,
    so the rollback's own calls of the original gathers are counted too. While `on`: the (head, request, block) loads
    from the HOST rows (a gather whose source is an engine's _k_cpu / _v_cpu; the staging redirect reads GPU staging and
    is not counted) are summed on the device (`blocks`, the K gathers). This includes the host read of a block that a
    step of the same group filled (its host rows are stale until the commit; the redirect then overwrites the slot from
    the staging block): a real PCIe read, so it is counted. While `timing` (and a clock is set): a span of clock events
    around each host gather, K and V (`spans`)."""

    def __init__(self, rs, clock=None):
        self.rs = rs
        kern = rs.ops.kernels
        self.kern = kern
        self.orig = (kern["gather_k"], kern["gather_v_bias"])
        dev = rs.cache.layers[0].cache_engine._k_gpu.device
        self.host = set()
        for lay in rs.cache.layers:
            self.host.add(lay.cache_engine._k_cpu.data_ptr())
            self.host.add(lay.cache_engine._v_cpu.data_ptr())
        self.on = False
        self.timing = False
        self.clock = clock
        self.blocks = torch.zeros((), dtype=torch.int64, device=dev)
        self.spans = []
        kern["gather_k"] = self._gk
        kern["gather_v_bias"] = self._gv

    def detach(self) -> None:
        if self.kern["gather_k"] != self._gk or self.kern["gather_v_bias"] != self._gv:
            raise SpecError("detach the rollback before the meter")
        self.kern["gather_k"], self.kern["gather_v_bias"] = self.orig

    def _span(self, fn, *args):
        if not (self.timing and self.clock is not None):
            return fn(*args)
        e0, e1 = self.clock.event(), self.clock.event()
        self.clock.record(e0)
        fn(*args)
        self.clock.record(e1)
        self.spans.append((e0, e1))

    def _gk(self, gpu, cpu, ids, bs):
        if not (self.on and cpu.data_ptr() in self.host):
            return self.orig[0](gpu, cpu, ids, bs)
        self.blocks += (ids >= 0).sum()
        self._span(self.orig[0], gpu, cpu, ids, bs)

    def _gv(self, gpu_v, cpu_v, bias_buf, kv_bias, ids, bs):
        if not (self.on and cpu_v.data_ptr() in self.host):
            return self.orig[1](gpu_v, cpu_v, bias_buf, kv_bias, ids, bs)
        self._span(self.orig[1], gpu_v, cpu_v, bias_buf, kv_bias, ids, bs)


# ------------------------------------------------------------------ the run parameters
class SpecOps:
    """The run parameters of one speculative arm (recorded in SpecRunner.meta).
      level      "L3" (K drafts, then K serial verify steps) or "L4" (joint pass of K verify rows + the draft row, then
                 K - 1 drafts)
      K          drafts per round (L3: 1 .. 32; L4: 2 .. 31)
      mode       L4: "exact" (the proof form) or "fast" (the timing form)
      gemm       L4: the joint pass's GEMM ("rowgroup": the shipped call's shape, bitwise; "natural"; "exact")
      linear     L4: the GEMM callable (required for gemm "exact")
      extra      L4 FAST: X, the extras store slots per request
      attn       L4: the attention callable (None: nosa_llama's flash_attn_nosa_with_kvcache; the CPU tests' stand-in)
      journal    the rollback's saved slots per layer and step (a lost slot past it comes back from the host)
      redirect   True: a draft loads a block that a step of its group filled from the GPU staging (module docstring)
      draft_filter  tests only: f(runner, kind, j, tokens) -> tokens, applied to every draft token as it is made
                 ("draft": draft step j; "draft_row": the joint pass's draft row); the proof holds for any drafts
      prefetch   rung L5 (not wired to a transport here): an object with on_draft(miss_ids) and on_round(phase), told
                 the draft misses after each draft step; None in the L3 / L4 cells
      rb_neg     a negative control of the rollback (ragged_rollback.NEG_KINDS), tests only
      stream     the rollback's executor (None: ragged_rollback.ImmediateStream)
      neg        a negative control of this module (NEG_KINDS), tests only
      record     keep every committed row's logits (tests only)
      probe_every  a probe round (gather spans, phase events) every N rounds, round 0 excluded; 0: none
      after_round  tests only: f(runner) after each round's host sync (the state checks between rounds)"""

    def __init__(self, level="L3", K=3, mode="exact", gemm="rowgroup", linear=None, extra=16, attn=None, journal=256,
                 redirect=True, draft_filter=None, prefetch=None, rb_neg=None, stream=None, neg=None, record=False,
                 probe_every=16, after_round=None):
        if level not in LEVELS:
            raise SpecError("level %r: one of %s" % (level, LEVELS))
        lo = 2 if level == "L4" else 1
        if not lo <= int(K) <= K_MAX[level]:
            raise SpecError("%s: K in %d .. %d (got %r)" % (level, lo, K_MAX[level], K))
        if mode not in L4_MODES or gemm not in L4_GEMMS:
            raise SpecError("mode %r / gemm %r: one of %s / %s" % (mode, gemm, L4_MODES, L4_GEMMS))
        if gemm == "exact" and linear is None and level == "L4":
            raise SpecError("gemm 'exact' takes a batch-invariant linear")
        if neg not in NEG_KINDS:
            raise SpecError("unknown negative control %r" % (neg,))
        if neg == "bonus_on_full_accept" and level != "L4":
            raise SpecError("bonus_on_full_accept is an L4 control (L3 has no token past the K-th verify step)")
        if int(journal) < 1 or int(extra) < 2 or int(probe_every) < 0:
            raise SpecError("journal >= 1, extra >= 2, probe_every >= 0")
        self.level, self.K = level, int(K)
        self.mode, self.gemm, self.linear, self.extra, self.attn = mode, gemm, linear, int(extra), attn
        self.journal, self.redirect = int(journal), bool(redirect)
        self.draft_filter, self.prefetch = draft_filter, prefetch
        self.rb_neg, self.stream, self.neg, self.record = rb_neg, stream, neg, bool(record)
        self.probe_every = int(probe_every)
        self.after_round = after_round

    def describe(self) -> dict:
        return dict(level=self.level, K=self.K, mode=(self.mode if self.level == "L4" else None),
                    gemm=(self.gemm if self.level == "L4" else None), extra=self.extra, journal=self.journal,
                    redirect=self.redirect, prefetch=self.prefetch is not None, rb_neg=self.rb_neg, neg=self.neg,
                    filtered=self.draft_filter is not None, probe_every=self.probe_every)


# ------------------------------------------------------------------ the runner
class SpecRunner:
    """The speculative rounds of one RaggedState (module docstring). Build it after ragged_engine.attach / build: it
    attaches, in this order, the KVMeter, the Rollback (K or 2 K steps per group), the DraftResident and (L4) the
    joint-pass state. detach() undoes them in the reverse order. clock: an object with event() / record(e) /
    elapsed(a, b) in ms / sync() (r2_base_cell.CudaClock on the GPU, HostClock on the CPU)."""

    def __init__(self, model, rs, ops: SpecOps, clock, t0=None):
        self.model, self.rs, self.ops, self.clock = model, rs, ops, clock
        B, K = int(rs.B), ops.K
        self.B, self.K = B, K
        dev = rs.st.cu_q.device
        self.dev = dev
        RE.set_activity(rs, torch.ones(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool))
        self.meter = KVMeter(rs, clock)
        self.rb = RR.Rollback(rs, K if ops.level == "L3" else 2 * K, capacity=ops.journal, stream=ops.stream,
                              neg=ops.rb_neg)
        self.draft = DraftResident(rs, self.rb)
        self.vs = None
        if ops.level == "L4":
            from . import verify_rows as VR
            self.VR = VR
            # the joint pass's GEMM: the given one, else the ragged step's own (gemm natural: F.linear; exact: the
            # batch-invariant linear), so a rowgroup joint pass has the bits of the serial step it replaces
            base = ops.linear if ops.linear is not None else self.rb._orig["linear"]
            vops = VR.VerifyOps(K=K, mode=ops.mode, gemm=ops.gemm, linear=self._wait_linear(base), extra=ops.extra,
                                attn=ops.attn, draft_last=(ops.mode == "fast"))
            self.vs = VR.attach(model, rs, vops)
            self.vs.hooks = self.rb
        i64 = torch.int64
        self.t = torch.zeros(B, dtype=i64, device=dev) if t0 is None else t0.to(device=dev, dtype=i64).view(B).clone()
        self.D = torch.full((B, K), -1, dtype=i64, device=dev)          # the drafts of the next verify (-1: none)
        self.count = torch.zeros(B, dtype=i64, device=dev)              # committed tokens per request
        self.arK = torch.arange(K, dtype=i64, device=dev)
        self.arR = torch.arange(K + 1, dtype=i64, device=dev)
        self.ids1 = torch.zeros((B, 1), dtype=i64, device=dev)
        self.pos1 = torch.zeros((B, 1), dtype=torch.int32, device=dev)
        self.acc_hist = torch.zeros(K + 1, dtype=i64, device=dev)       # n per (request, round)
        self.valid_sum = torch.zeros((), dtype=i64, device=dev)         # valid drafts offered
        self.accepted_sum = torch.zeros((), dtype=i64, device=dev)
        self.committed_sum = torch.zeros((), dtype=i64, device=dev)
        self.full_sum = torch.zeros((), dtype=i64, device=dev)          # (request, round) with every valid draft taken
        self.round_committed, self.round_blocks, self.round_events, self.round_probe = [], [], [], []
        self.records = []                                               # tests (ops.record): per round host facts
        self.cap = 0
        self.out = self.fmax = self.fsum = self.lg_out = None
        self.round = -1
        self.next_col = None
        self.prelude_ms = None
        self.prelude_events = None
        self.wall_s = None
        self.warm_steps = 0
        self.rounds_run = 0
        self.pos_host = rs.st.pos_host.clone()
        self.meta = dict(ops=ops.describe(), B=B, rollback_K=self.rb.K, journal=ops.journal,
                         verify=(self.vs.meta if self.vs is not None else None))

    # -------------------------------------------------------------- helpers
    def _wait_linear(self, base):
        """The joint pass's GEMM with the rollback's release wait first (the joint pass reads twin.pos before its
        twin hook; the ragged step's first GEMM is the rollback's linear hook)."""
        rb = self.rb

        def linear(x, w):
            if rb._need_wait:
                rb._wait_release_all()
            return base(x, w)
        linear.__name__ = getattr(base, "__name__", type(base).__name__)
        linear.m_fix = getattr(base, "m_fix", None)
        return linear

    def detach(self) -> None:
        if self.rb.tentative:
            raise SpecError("detach inside a round")
        if self.vs is not None:
            self.vs.hooks = None
        self.draft.detach()
        self.rb.detach()
        self.meter.detach()

    def _filter(self, kind: str, j: int, tok, next_col=None):
        """Tests only (ops.draft_filter). next_col (B,): the committed column that this draft token would take if the
        verify accepts it (filters can read runner.next_col; None outside a filter call)."""
        f = self.ops.draft_filter
        if f is None:
            return tok
        self.next_col = next_col
        try:
            return f(self, kind, j, tok)
        finally:
            self.next_col = None

    def _edge_before(self, n: int) -> bool:
        """HOST: some request fills a block in the first n tentative positions of the group (from the position mirror
        taken at the round start): only then can a draft need the staging redirect."""
        if not self.ops.redirect or n <= 0:
            return False
        p = self.pos_host.view(-1, 1) + torch.arange(n, dtype=torch.int64).view(1, -1)
        return bool(((p % RE.BLOCK) == RE.BLOCK - 1).any())

    def _sync_release(self) -> None:
        """The rollback's release wait BEFORE this runner reads twin.pos (a commit / truncate on another executor may
        still move it back; the default executor is the same stream, so the wait is satisfied already)."""
        if self.rb._need_wait:
            self.rb._wait_release_all()

    def _step(self, x):
        """One ragged step of every request: x (B,) tokens at each request's twin position. Returns (B, V) logits."""
        self._sync_release()
        self.ids1.copy_(x.reshape(self.B, 1))
        self.pos1.copy_(self.rs.st.twins[0].pos.view(self.B, 1))
        return RE.decode_step(self.model, self.rs, self.ids1, self.pos1)[:, -1]

    def _draft_step(self, x, n_before: int):
        """A draft step (DraftResident 'all'); returns the argmax tokens. n_before: the tentative positions of the group
        before this step (the redirect flag)."""
        self.draft.begin("all", redirect=self._edge_before(n_before))
        try:
            lg = self._step(x)
        finally:
            self.draft.end()
        if self.ops.prefetch is not None:
            self.ops.prefetch.on_draft(self.draft.miss_ids)
        return lg.argmax(dim=-1)

    def _event(self, ev):
        e = self.clock.event()
        self.clock.record(e)
        ev.append(e)

    def _commit_tokens(self, G, m, FM, FS, LG) -> None:
        """Committed tokens G[b, :m[b]] (and their logits fingerprints) at columns count[b] .. ; count += m. Columns
        past the capacity go to the dummy column."""
        B, K = G.shape
        cols = self.count.view(B, 1) + self.arK[:K].view(1, K)
        keep = (self.arK[:K].view(1, K) < m.view(B, 1)) & (cols < self.cap)
        cols = torch.where(keep, cols, torch.full_like(cols, self.cap))
        self.out.scatter_(1, cols, G)
        self.fmax.scatter_(1, cols, FM)
        self.fsum.scatter_(1, cols, FS)
        if self.lg_out is not None:
            V = LG.shape[-1]
            self.lg_out.scatter_(1, cols.view(B, K, 1).expand(B, K, V), LG)
        self.count += m

    def _accept(self, G, D):
        """n (accepted drafts), v (valid drafts), m (committed tokens), full (every valid draft accepted and K valid)."""
        n = accept_prefix(G, D)
        v = (D >= 0).to(torch.int64).sum(1)
        m = (n + 1).clamp(max=self.K)
        full = n >= self.K
        return n, v, m, full

    def _stats(self, n, v, m, full) -> None:
        self.acc_hist.scatter_add_(0, n, torch.ones_like(n))
        self.valid_sum += v.sum()
        self.accepted_sum += n.sum()
        self.committed_sum += (n if self.ops.neg == "miscount" else m).sum()
        self.full_sum += (n >= v).to(torch.int64).sum()
        self.round_committed.append(m.sum())

    # -------------------------------------------------------------- the rounds
    def _prelude(self) -> None:
        """K draft steps from the committed position, in their own group, all undone: the drafts of the first verify
        (L4; L3 drafts inside every round)."""
        K = self.K
        self.rb.begin()
        x = self.t
        for j in range(K):
            d = self._filter("draft", j, self._draft_step(x, j), next_col=self.count + j)
            self.D[:, j].copy_(d)
            x = self.D[:, j]
        self.rb.reject_all()

    def _round_l3(self, probe: bool) -> None:
        K, B = self.K, self.B
        ev = []
        self._event(ev)
        self._prelude()                                                  # 1. K drafts, undone
        self._event(ev)
        D = self.D
        X = torch.cat((self.t.view(B, 1), D[:, :K - 1]), dim=1)
        self.rb.begin()                                                  # 2. K verify steps (the shipped step)
        G = torch.empty((B, K), dtype=torch.int64, device=self.dev)
        FM = torch.empty((B, K), dtype=torch.float32, device=self.dev)
        FS = torch.empty((B, K), dtype=torch.float64, device=self.dev)
        LG = [] if self.lg_out is not None else None
        b0 = self.meter.blocks.clone()
        self.meter.on, self.meter.timing = True, probe
        s0 = len(self.meter.spans)
        try:
            for j in range(K):
                lg = self._step(X[:, j].contiguous())
                G[:, j].copy_(lg.argmax(dim=-1))
                FM[:, j].copy_(lg.amax(dim=-1))
                FS[:, j].copy_(lg.sum(dim=-1, dtype=torch.float64))
                if LG is not None:
                    LG.append(lg.clone())
        finally:
            self.meter.on, self.meter.timing = False, False
        self._event(ev)
        n, v, m, full = self._accept(G, D)                               # 3. accept, correct, commit
        keep = m
        if self.ops.neg == "keep_rejected":
            keep = torch.full_like(m, K)
        Gc = self._plant_commit(G, D, n, m)
        self.rb.commit(keep)
        self._event(ev)
        self._commit_tokens(Gc, m, FM, FS, torch.stack(LG, 1) if LG is not None else None)
        self.t = Gc.gather(1, (m - 1).view(B, 1)).view(B)
        self._stats(n, v, m, full)
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if self.ops.record:
            self.records.append(dict(D=D.clone(), G=G.clone(), n=n.clone(), v=v.clone(), m=m.clone(),
                                     full=full.clone(), committed=Gc.clone(), pos=self.pos_host.clone()))

    def _round_l4(self, probe: bool) -> None:
        K, B = self.K, self.B
        R = K + 1
        ev = []
        self._event(ev)
        D = self.D
        ids = torch.cat((self.t.view(B, 1), D.clamp(min=0)), dim=1)     # rows 0 .. K - 1 verify, row K the draft row
        self._sync_release()
        pos = (self.rs.st.twins[0].pos.view(B, 1) + self.arR.view(1, R)).to(torch.int32)
        self.rb.begin()                                                  # 1. the joint pass
        exact = self.ops.mode == "exact"
        b0 = self.meter.blocks.clone()
        s0 = len(self.meter.spans)
        if exact:
            self.draft.begin(K, redirect=self._edge_before(K))
        self.meter.on, self.meter.timing = True, probe
        try:
            lg = self.VR.joint_pass(self.model, self.vs, ids, pos)
        finally:
            self.meter.on, self.meter.timing = False, False
            if exact:
                self.draft.end()
        self._event(ev)
        G = lg[:, :K].argmax(dim=-1)
        FM = lg[:, :K].amax(dim=-1)
        FS = lg[:, :K].sum(dim=-1, dtype=torch.float64)
        e0 = self._filter("draft_row", 0, lg[:, K].argmax(dim=-1), next_col=self.count + K)
        n, v, m, full = self._accept(G, D)                               # 2. accept
        Gc = self._plant_commit(G, D, n, m)
        if self.ops.neg == "keep_rejected":
            self.rb.truncate(torch.full_like(m, R))
        else:
            self.rb.truncate(torch.where(full, torch.full_like(m, R), m))   # 3. truncate the requests that rejected
        self._event(ev)
        self._commit_tokens(Gc, m, FM, FS, lg[:, :K].clone() if self.lg_out is not None else None)
        if self.ops.neg == "bonus_on_full_accept":                       # NEGATIVE CONTROL: e0 counted as committed
            self._commit_tokens(e0.view(B, 1).expand(B, K).contiguous(), full.to(torch.int64), FM, FS,
                                lg[:, :K].clone() if self.lg_out is not None else None)
        t_new = Gc.gather(1, (m - 1).view(B, 1)).view(B)
        Dn = torch.full((B, K), -1, dtype=torch.int64, device=self.dev)  # 4. K - 1 drafts
        Dn[:, 0].copy_(torch.where(full, e0, Dn[:, 0]))
        x = torch.where(full, e0, t_new)
        shift = full.to(torch.int64).view(B, 1)
        for j in range(K - 1):
            y = self._filter("draft", j, self._draft_step(x, R + j), next_col=self.count + j + shift.view(B))
            Dn.scatter_(1, shift + j, y.view(B, 1))
            x = y
        self._event(ev)
        nfin = torch.where(full, torch.full_like(m, K), torch.full_like(m, R))   # 5. commit (truncated steps marked)
        if self.ops.neg == "keep_rejected":
            nfin = torch.full_like(m, K)
        self.rb.commit(nfin)
        self._event(ev)
        self.t = t_new
        self.D = Dn
        self._stats(n, v, m, full)
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if self.ops.record:
            self.records.append(dict(D=D.clone(), G=G.clone(), n=n.clone(), v=v.clone(), m=m.clone(),
                                     full=full.clone(), committed=Gc.clone(), e0=e0.clone(), pos=self.pos_host.clone()))

    def _plant_commit(self, G, D, n, m):
        """The committed tokens of the round: G (the target argmaxes). NEGATIVE CONTROL commit_draft_at_reject: at a
        reject (a valid draft d_n != g_n) the draft token is committed in place of the correction."""
        if self.ops.neg != "commit_draft_at_reject":
            return G
        B, K = G.shape
        nc = n.clamp(max=K - 1).view(B, 1)
        dn = D.gather(1, nc)
        bad = (n < K).view(B, 1) & (dn >= 0) & (self.arK[:K].view(1, K) == nc)
        return torch.where(bad, dn.expand(B, K), G)

    def _warm(self, n: int) -> None:
        """n PLAIN steps (no group, no draft): greedy tokens, committed. The first step after the prefill must be one:
        until a step loads them, the window slots hold torch.empty bytes, and a draft reads the skipped slots through
        the masked bias (0 x NaN = NaN; Task 1.1's DraftBlockEdge refusal had the same reason)."""
        B = self.B
        one = torch.ones(B, dtype=torch.int64, device=self.dev)
        for _ in range(n):
            lg = self._step(self.t)
            g = lg.argmax(dim=-1)
            self._commit_tokens(g.view(B, 1), one, lg.amax(dim=-1).view(B, 1),
                                lg.sum(dim=-1, dtype=torch.float64).view(B, 1),
                                lg.view(B, 1, -1).clone() if self.lg_out is not None else None)
            self.t = g

    # -------------------------------------------------------------- the run
    @torch.inference_mode()
    def run(self, N: int, rounds: int = 0, max_rounds: int = 0, warm: int = 1) -> dict:
        """The speculative decode: `warm` plain steps (default 1, see _warm), then the rounds. rounds > 0: exactly that
        many rounds (the GPU cell: ceil((N - warm) / K), so no request passes N + K - 1 committed tokens). rounds 0:
        until every request has N committed tokens (at most max_rounds, default N + 8). Committed tokens, their logits
        fingerprints (and, with record, their logits) are kept for the first N + K columns. One host sync per round."""
        B, K = self.B, self.K
        N = int(N)
        if int(warm) < 1:
            raise SpecError("warm >= 1: the first step after the prefill loads the window (see _warm)")
        self.warm_steps = int(warm)
        self.cap = N + K
        dev = self.dev
        self.out = torch.full((B, self.cap + 1), -1, dtype=torch.int64, device=dev)
        self.fmax = torch.zeros((B, self.cap + 1), dtype=torch.float32, device=dev)
        self.fsum = torch.zeros((B, self.cap + 1), dtype=torch.float64, device=dev)
        if self.ops.record:
            V = int(self.model.lm_head.shape[0])
            self.lg_out = torch.zeros((B, self.cap + 1, V), dtype=torch.float32, device=dev)
        limit = int(rounds) if rounds > 0 else (int(max_rounds) if max_rounds > 0 else N + 8)
        pe = self.ops.probe_every
        self.clock.sync()
        t_wall = time.perf_counter()
        self._warm(self.warm_steps)
        self.pos_host = self.rs.st.twins[0].pos.cpu()                    # a host sync (once, before the rounds)
        self.rs.st.pos_host.copy_(self.pos_host)
        if self.ops.level == "L4":
            ev = []
            self._event(ev)
            self._prelude()
            self._event(ev)
            self.prelude_events = ev
        cnt = torch.zeros(B, dtype=torch.int64)
        g = 0
        while g < limit:
            self.round = g
            probe = pe > 0 and g > 0 and g % pe == 0
            if self.ops.prefetch is not None:
                self.ops.prefetch.on_round("begin")
            if self.ops.level == "L3":
                self._round_l3(probe)
            else:
                self._round_l4(probe)
            if self.ops.prefetch is not None:
                self.ops.prefetch.on_round("end")
            hc = torch.stack((self.count, self.rs.st.twins[0].pos)).cpu()          # THE host sync of the round
            cnt = hc[0]
            self.pos_host = hc[1].clone()
            self.rs.st.pos_host.copy_(self.pos_host)
            if self.ops.after_round is not None:
                self.ops.after_round(self)
            g += 1
            if rounds <= 0 and int(cnt.min()) >= N:
                break
        self.clock.sync()
        self.wall_s = time.perf_counter() - t_wall
        self.rounds_run = g
        return dict(rounds=g, count=cnt)

    @torch.inference_mode()
    def read(self) -> dict:
        """ONE host read after run(): the counters, the per-round sums and the event spans (ms)."""
        el = self.clock.elapsed
        out = dict(rounds=self.rounds_run, warm=self.warm_steps, wall_s=self.wall_s, count=self.count.cpu().tolist(),
                   acc_hist=self.acc_hist.cpu().tolist(), valid=int(self.valid_sum), accepted=int(self.accepted_sum),
                   committed=int(self.committed_sum), full=int(self.full_sum),
                   round_committed=[int(x) for x in torch.stack(self.round_committed).cpu()] if self.round_committed
                   else [], round_blocks=[int(x) for x in torch.stack(self.round_blocks).cpu()] if self.round_blocks
                   else [],
                   phases=[[el(ev[i], ev[i + 1]) for i in range(len(ev) - 1)] for ev in self.round_events],
                   round_ms=[el(ev[0], ev[-1]) for ev in self.round_events],
                   probe_wait_ms=[(sum(el(a, b) for a, b in sp) if sp is not None else None) for sp in self.round_probe],
                   draft_skipped=self.draft.skipped.cpu().tolist(), draft_redirected=self.draft.redirected.cpu().tolist(),
                   draft_calls=self.draft.calls, draft_steps=self.draft.steps, journal=self.rb.stats(),
                   truncations=self.rb.truncations, groups=self.rb.groups, errors=RE.errors(self.rs),
                   prelude_ms=(el(self.prelude_events[0], self.prelude_events[1])
                               if getattr(self, "prelude_events", None) else None),
                   verify=(self.VR.read_stats(self.vs) if self.vs is not None else None))
        return out


PHASES = {"L3": ("draft", "verify", "commit"), "L4": ("joint", "truncate", "draft", "commit")}
