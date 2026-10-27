"""Phase 2 of KS/analysis/plan/PLAN_speckv_on_cached_nosa.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), rungs L3, L4 and
L4P (the paper's schedule; tree nosi-r2-specp, docs/NOTES_specp.md):
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
                    of the verify rows' union, the draft row resident-only (VerifyOps(draft_last=True)). L4P's joint
                    pass: EXACT, K + 2 rows, DraftResident mode "rows" (the draft rows per request).
  KVMeter           (this module) the KV that the verify fetches from the host: blocks (device count) and, on probe
                    rounds, CUDA-event spans of the host gathers.
  RequestPin        (this module; L4P, OFF by default) the requested-KV pin of paper 3.3 with Task 1.2's soft pin.

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

L4P, ONE ROUND: THE PAPER'S SCHEDULE (section 3, Figure 3; section 3.4). Drafting is continuous. Request b has its
committed position q_b, its last committed token t_b (not fed yet) and u_b TENTATIVE drafts e_1 .. e_u (positions
q_b + 1 .. q_b + u_b). Every K draft steps ONE joint pass verifies every tentative draft BUT THE NEWEST (v_b = u_b - 1
verify rows) and drafts the next token. The newest draft e_u is DEFERRED to the next verification group (its KV
request is one draft step old; rung L5 fetches it during the next drafts).
  0. (first round only) K draft steps in their own group, reject_all: e_1 .. e_K (u = K).
  1. begin; ONE JOINT PASS of R = K + 2 rows at q_b .. q_b + K + 1; row r feeds x_r (x_0 = t_b, x_r = e_r):
       rows 0 .. v_b - 1  VERIFY rows: the target's full selection and the serial residency per row; row r gives the
                          target's token g_{r+1}, which checks e_{r+1}
       row v_b            the DEFERRED row: resident-only; it feeds e_v and makes the draft K / V of its position again
                          (the draft row reads them); its token is not used (e_u stays the draft of that position)
       row u_b = v_b + 1  the DRAFT row: resident-only; it feeds e_u and gives the next draft e_new
       rows > u_b         PAD rows: resident-only; truncate undoes them at once
     DraftResident mode "rows" makes rows >= v_b a draft residency PER REQUEST (one call serves verify and draft rows of
     different requests). K = 3, first round (Figure 3): rows feed t, d1 (verify d1, d2), d2 (the deferred row), d3 (the
     draft row: d4), one pad row. d3 is deferred: the next group verifies d3, d4, d5 and defers d6.
  2. ACCEPT (greedy, paper 3.4): n_b = the longest prefix of e_1 .. e_v that the target gives; full_b = (n_b = v_b);
     m_b = v_b after a full accept (NO bonus token: the deferred token stays tentative), else n_b + 1 (the accepted
     drafts and the target's correction g_{n+1}).
  3. truncate: a full accept keeps rows 0 .. u_b (its deferred row and draft row stay tentative); a reject keeps m_b
     rows (the joint pass's draft row and every row from the rejected position on are undone now). A fill of a kept
     tentative row stays staged (ragged_rollback: no host write in an open group); the commit drops it.
  4. K - 1 DRAFT steps: after a full accept from e_new at q_b + u_b + 1; after a reject from the correction at
     q_b + m_b. The next drafts: full accept [e_u, e_new, y_1 .. y_{K-1}] (u = K + 1); reject [y_1 .. y_{K-1}]
     (u = K - 1: the deferred token and e_new are discarded with the rejected rows).
  5. commit(m_b): every request keeps its m_b verify rows (the deferred row, the draft row and the drafts undone).
  The group holds R + K - 1 = 2 K + 1 tentative steps. A request commits at most v_b tokens per round: K - 1 in the
  first round, K after a full accept, K - 2 after a reject. K >= 3 (with K = 2 a request that rejects has no verify
  row in its next joint pass). EXACT only (gemm rowgroup or exact); a FAST L4P needs a per-row verify mask in
  verify_rows._extras (not built).
  THE DEFERRED ROW (a difference from Figure 3, which shows one draft row). The verify rows must start from the
  committed engine state (the serial residency, for the bitwise proof), so the round undoes every tentative step before
  the next joint pass. The draft row at q + u reads the K / V of position q + u - 1, and only a draft step made them.
  The deferred row makes them again in the same pass: one more resident-only row per request (GEMM rows and a draft
  residency; no host fetch).

REQUESTED-KV PIN (paper 3.3; SpecOps(pin=True), L4P only, OFF by default and in the cells). The draft calls record their
selection (the blocks that the draft requested: hits and misses; in the joint pass the draft row's). Before the joint
pass (outside a group) every pool slot that holds a requested block gets the SOFT PIN of Task 1.2 (pool_prefetch.py:
age PIN_BASE + n, so pool_update takes it as an LRU victim last; a pool hit gives the slot a real stamp: the pin ends).
After the round's commit every slot that is still pinned gets its age back. Nothing is fetched in this rung, so the pin
keeps the draft's pool HITS on the GPU until the verify has read them; rung L5 pins its fetched blocks with the same
rule. The tokens do not change: a window slot gets the same bytes from a pool slot or from the host rows.

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
L4P only: verify_newest (no deferral: the joint pass verifies the newest draft too), keep_joint_draft (a reject keeps
the joint pass's deferred and draft rows and drafts on from e_new), commit_draft_rows (the round's commit keeps the
deferred and draft rows of a full accept: their draft K / V become committed state), commit_deferred (a full accept also
commits the deferred row's token), pin_no_release (the pins stay after the round).
"""
from __future__ import annotations

import time

import torch

from . import ragged_engine as RE
from . import ragged_rollback as RR

MASK_BIAS = -3.0e4                     # avail_policy.py:113, verify_rows.MASK_BIAS: finite on purpose
LEVELS = ("L3", "L4", "L4P")
L4_MODES = ("exact", "fast")
L4_GEMMS = ("natural", "rowgroup", "exact")
NEG_KINDS = (None, "commit_draft_at_reject", "keep_rejected", "bonus_on_full_accept", "miscount")
NEG_L4P = ("verify_newest", "keep_joint_draft", "commit_draft_rows", "commit_deferred", "pin_no_release")
NEG_BY_LEVEL = {"L3": NEG_KINDS, "L4": NEG_KINDS,
                "L4P": (None, "commit_draft_at_reject", "miscount") + NEG_L4P}
# L4: verify_rows.K_MAX; the rollback group holds 2 K <= 64 steps. L4P: R = K + 2 rows <= verify_rows.K_MAX + 1 and a
# group of 2 K + 1 <= 64 steps; K >= 3 (module docstring)
K_MAX = {"L3": 32, "L4": 31, "L4P": 30}
K_MIN = {"L3": 1, "L4": 2, "L4P": 3}
ROWS_MAX = 32                          # DraftResident mode "rows": at most verify_rows.K_MAX + 1 rows per pass
PIN_BASE = 1 << 60                    # pool_prefetch.PIN_BASE (the same value; tests_cpu/test_spec_paper.py reads it)


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
      mode "rows" (L4P) PER REQUEST: the r-th call of each layer is a draft residency for the requests b with
                  rows[r, b] and the full (target) residency for the others (an EXACT joint pass whose verify rows and
                  draft rows differ by request). The host gathers get the load ids of the verify requests only (and,
                  with redirect, the staged block of the draft requests); the scratch bias masks the draft requests'
                  skipped slots only, so a verify request reads the live bias values (the same bits). rows: (R, B)
                  bool; record: (R, B) bool, the (row, request) pairs whose misses go to miss_ids and whose selection
                  goes to the pin (default: rows).
    A draft call is refused outside an open rollback group (its fill would reach the host rows: the dr1 leak).
    Counters (device): skipped slots and redirected slots per layer; miss_ids[l] = the skipped block ids of the last
    draft call of layer l (H, B, topk; -1 elsewhere): the list that a background prefetch (rung L5) would fetch. Mode
    "rows" also counts the skipped slots per row index (row_skipped) and the draft (request, call) pairs
    (rows_draft)."""

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
        self.rows = self.rec = None                                      # mode "rows" (L4P): (R, B) bool each
        self.row_skipped = torch.zeros((ROWS_MAX,), dtype=torch.int64, device=dev)
        self.rows_draft = torch.zeros((), dtype=torch.int64, device=dev)  # mode "rows": draft (request, call) pairs
        self.rows_calls = 0                                              # mode "rows": decode_fixed calls (host count)
        self.pin = None                                                  # L4P with SpecOps(pin=True): a RequestPin
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

    def begin(self, mode, redirect: bool = False, rows=None, record=None) -> None:
        if self.mode is not None:
            raise SpecError("begin inside a draft bracket")
        if mode == "rows":
            if rows is None or rows.dim() != 2 or rows.dtype != torch.bool or rows.shape[1] != self.B or \
                    not 1 <= rows.shape[0] <= ROWS_MAX:
                raise SpecError("mode 'rows' takes rows: an (R, %d) bool tensor, 1 <= R <= %d" % (self.B, ROWS_MAX))
            if record is not None and (record.dtype != torch.bool or tuple(record.shape) != tuple(rows.shape)):
                raise SpecError("record must be a bool tensor of the shape of rows")
        elif rows is not None or record is not None:
            raise SpecError("rows / record belong to mode 'rows'")
        elif mode != "all" and not (isinstance(mode, int) and mode >= 0):
            raise SpecError("mode must be 'all', 'rows' or a row index >= 0, got %r" % (mode,))
        if not self.rb.tentative:
            raise SpecError("a draft outside a rollback group: its rows could reach the host rows (the dr1 leak)")
        self.mode = mode
        self.redirect = bool(redirect)
        if mode == "rows":
            self.rows, self.rec = rows, (rows if record is None else record)
        self._calls = [0] * self.L
        self.steps += 1

    def end(self) -> None:
        """Every layer's attended bias view back to the live one; the bracket is closed."""
        for l, view in self._live.items():
            self.rs.cache.layers[l].cache_engine._kv_bias_gpu_att = view
        self.mode = None
        self.rows = self.rec = None

    def _hook(self, eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff, gather_k,
              gather_v_bias):
        l = self._lay[eng._k_gpu.data_ptr()]
        i = self._calls[l]
        self._calls[l] = i + 1
        mode = self.mode
        if mode == "rows":
            return self._rows_call(l, i, eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff,
                                   gather_k, gather_v_bias)
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
        if self.pin is not None:                                         # L4P, pin on: the requested blocks
            self.pin.record(l, topk_idx)
        rows = skip.permute(1, 2, 0).unsqueeze(2).expand(self.B, self.topk, self.bs, self.H)
        rows = rows.reshape(self.B, self.topk * self.bs, self.H)
        self.scratch_att.copy_(self._live[l] if self.pool else eng._kv_bias_gpu)
        self.scratch_att.masked_fill_(rows, MASK_BIAS)
        if self.pool:
            eng._kv_bias_gpu_att = self.scratch_att                      # attention reads the attended views
        self.calls += 1
        return out[0], out[1], self.scratch, out[3]                      # without a pool attention reads these

    def _rows_call(self, l: int, i: int, eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff,
                   gather_k, gather_v_bias):
        """Mode "rows": call i of layer l. The draft requests of this call (rows[i, b]) get the draft residency of
        _hook; the other requests get the full residency (their load ids go to the given gathers unchanged)."""
        if i >= self.rows.shape[0]:
            raise SpecError("layer %d: decode_fixed call %d, but rows has %d rows" % (l, i, self.rows.shape[0]))
        dm = self.rows[i].view(1, self.B, 1)                             # (1, B, 1): the draft requests of this call
        staged = self.rb.T[l]["id"].view(1, self.B, 1) if self.redirect else None
        box = {}

        def gk(gpu, cpu, ids, bs):                                       # verify requests: every load; draft
            keep = ~dm                                                   # requests: the staged block only (redirect)
            if staged is not None:
                keep = keep | ((ids == staged) & (staged >= 0))
            box["ids"] = torch.where(keep, ids, self.minus1)
            gather_k(gpu, cpu, box["ids"], bs)

        def gv(gpu_v, cpu_v, bias_buf, bias, ids, bs):
            gather_v_bias(gpu_v, cpu_v, bias_buf, bias, box["ids"], bs)
        out = self._orig["fn"](eng, key_states, value_states, kv_bias, topk_idx, pos, active, err, diff, gk, gv)
        lm = eng._load_mask                                              # after pool_update: the host misses
        skip = (lm >= 0) & dm
        if staged is not None:
            red = skip & (lm == staged) & (staged >= 0)
            skip = skip & ~red
            self.redirected[l] += red.sum()
        eng._block_map.masked_fill_(skip, -1)                           # the next step loads the skipped blocks
        rec = self.rec[i].view(1, self.B, 1)
        self.miss_ids[l].copy_(torch.where(rec, torch.where(skip, lm, self.minus1), self.miss_ids[l]))
        n = skip.sum()
        self.skipped[l] += n
        self.row_skipped[i] += n
        self.rows_draft += self.rows[i].sum()
        if self.pin is not None:
            self.pin.record(l, topk_idx, rec)
        rows = skip.permute(1, 2, 0).unsqueeze(2).expand(self.B, self.topk, self.bs, self.H)
        rows = rows.reshape(self.B, self.topk * self.bs, self.H)
        self.scratch_att.copy_(self._live[l] if self.pool else eng._kv_bias_gpu)
        self.scratch_att.masked_fill_(rows, MASK_BIAS)
        if self.pool:
            eng._kv_bias_gpu_att = self.scratch_att                      # attention reads the attended views
        self.calls += 1
        self.rows_calls += 1
        return out[0], out[1], self.scratch, out[3]


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


# ------------------------------------------------------------------ the requested-KV pin (paper 3.3)
class RequestPin:
    """Keep the requested KV until the verify has read it (module docstring, REQUESTED-KV PIN). L4P only, OFF unless
    SpecOps(pin=True). The SOFT PIN of Task 1.2 (pool_prefetch.py: _pool_age = PIN_BASE + n; pool_update takes a pinned
    slot as an LRU victim last; a pool hit rewrites the age to a real stamp, so the pin ends). Without a victim pool it
    does nothing (no slot to pin). Device ops only, no host read.
      record(l, sel, mask)  a draft call of layer l selected sel (H, B, topk); mask (1, B, 1): the requests that count
      apply()               OUTSIDE a group, before the joint pass: pin every pool slot that holds a recorded block; the
                            record starts empty again
      release()             OUTSIDE a group, after the round's commit: every slot that is still pinned (no kept verify
                            row hit it) gets its age from before apply() back
    Counters (device): recorded selections, pins applied, pins released without a hit (unread)."""

    def __init__(self, rs, neg=None):
        self.rs, self.neg = rs, neg
        eng0 = rs.cache.layers[0].cache_engine
        self.L, self.B, self.H = int(rs.L), int(rs.B), int(eng0.head_num)
        self.on = bool(rs.pool)
        dev = eng0._k_gpu.device
        self.NB = int(rs.st.M_alloc)                                   # block ids of a selection are < M_alloc
        z = lambda: torch.zeros((), dtype=torch.int64, device=dev)    # noqa: E731
        self.recorded, self.applied, self.unread = z(), z(), z()
        self.applies = self.releases = 0
        self.live = False
        self.req, self.saved, self.base = [], [], None
        if self.on:
            P = int(eng0.pool_blocks)
            self.req = [torch.zeros((self.H, self.B, self.NB + 1), dtype=torch.bool, device=dev) for _ in range(self.L)]
            self.saved = [torch.empty_like(lay.cache_engine._pool_age) for lay in rs.cache.layers]
            ar = torch.arange(self.H * self.B * P, dtype=torch.int64, device=dev)
            self.base = (PIN_BASE + ar).view(self.H, self.B, P)        # one pin age per slot

    def record(self, l: int, sel, mask=None) -> None:
        if not self.on:
            return
        ok = sel >= 0
        if mask is not None:
            ok = ok & mask
        idx = torch.where(ok, sel.clamp(max=self.NB - 1), torch.full_like(sel, self.NB))   # column NB: the dummy
        self.req[l].scatter_(2, idx, True)
        self.recorded += ok.sum()

    def apply(self) -> None:
        if not self.on:
            return
        for l, lay in enumerate(self.rs.cache.layers):
            e = lay.cache_engine
            pm = e._pool_map
            hit = (pm >= 0) & self.req[l].gather(2, pm.clamp(min=0, max=self.NB - 1))
            self.saved[l].copy_(e._pool_age)
            e._pool_age.copy_(torch.where(hit, self.base, e._pool_age))
            self.applied += hit.sum()
            self.req[l].zero_()
        self.applies += 1
        self.live = True

    def release(self) -> None:
        if not self.on:
            return
        self.live = False
        if self.neg == "pin_no_release":                               # NEGATIVE CONTROL (tests only)
            return
        for l, lay in enumerate(self.rs.cache.layers):
            e = lay.cache_engine
            pinned = e._pool_age >= PIN_BASE
            self.unread += pinned.sum()
            e._pool_age.copy_(torch.where(pinned, self.saved[l], e._pool_age))
        self.releases += 1


# ------------------------------------------------------------------ the run parameters
class SpecOps:
    """The run parameters of one speculative arm (recorded in SpecRunner.meta).
      level      "L3" (K drafts, then K serial verify steps), "L4" (joint pass of K verify rows + the draft row, then
                 K - 1 drafts) or "L4P" (the paper's schedule: a joint pass every K draft steps that verifies every
                 tentative draft but the newest and drafts the next token; module docstring)
      K          drafts per round (L3: 1 .. 32; L4: 2 .. 31; L4P: 3 .. 30)
      mode       L4: "exact" (the proof form) or "fast" (the timing form); L4P: "exact" only
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
      after_round  tests only: f(runner) after each round's host sync (the state checks between rounds)
      pin        L4P only: True = the REQUESTED-KV PIN (RequestPin, paper 3.3); default False (off in this rung)"""

    def __init__(self, level="L3", K=3, mode="exact", gemm="rowgroup", linear=None, extra=16, attn=None, journal=256,
                 redirect=True, draft_filter=None, prefetch=None, rb_neg=None, stream=None, neg=None, record=False,
                 probe_every=16, after_round=None, pin=False):
        if level not in LEVELS:
            raise SpecError("level %r: one of %s" % (level, LEVELS))
        lo = K_MIN[level]
        if not lo <= int(K) <= K_MAX[level]:
            raise SpecError("%s: K in %d .. %d (got %r)" % (level, lo, K_MAX[level], K))
        if mode not in L4_MODES or gemm not in L4_GEMMS:
            raise SpecError("mode %r / gemm %r: one of %s / %s" % (mode, gemm, L4_MODES, L4_GEMMS))
        if gemm == "exact" and linear is None and level in ("L4", "L4P"):
            raise SpecError("gemm 'exact' takes a batch-invariant linear")
        if level == "L4P" and mode != "exact":
            raise SpecError("L4P: mode 'exact' only (a FAST L4P needs a per-row verify mask in verify_rows._extras: "
                            "not built)")
        if neg not in NEG_BY_LEVEL[level]:
            raise SpecError("negative control %r: for %s one of %s" % (neg, level, NEG_BY_LEVEL[level]))
        if neg == "bonus_on_full_accept" and level != "L4":
            raise SpecError("bonus_on_full_accept is an L4 control (L3 has no token past the K-th verify step)")
        if bool(pin) and level != "L4P":
            raise SpecError("pin: the requested-KV pin is wired for L4P only")
        if neg == "pin_no_release" and not bool(pin):
            raise SpecError("pin_no_release needs pin=True")
        if int(journal) < 1 or int(extra) < 2 or int(probe_every) < 0:
            raise SpecError("journal >= 1, extra >= 2, probe_every >= 0")
        self.level, self.K = level, int(K)
        self.mode, self.gemm, self.linear, self.extra, self.attn = mode, gemm, linear, int(extra), attn
        self.journal, self.redirect = int(journal), bool(redirect)
        self.draft_filter, self.prefetch = draft_filter, prefetch
        self.rb_neg, self.stream, self.neg, self.record = rb_neg, stream, neg, bool(record)
        self.probe_every = int(probe_every)
        self.after_round = after_round
        self.pin = bool(pin)

    def describe(self) -> dict:
        joint = self.level in ("L4", "L4P")
        return dict(level=self.level, K=self.K, mode=(self.mode if joint else None),
                    gemm=(self.gemm if joint else None), extra=self.extra, journal=self.journal,
                    redirect=self.redirect, prefetch=self.prefetch is not None, rb_neg=self.rb_neg, neg=self.neg,
                    filtered=self.draft_filter is not None, probe_every=self.probe_every, pin=self.pin)


# ------------------------------------------------------------------ the runner
class SpecRunner:
    """The speculative rounds of one RaggedState (module docstring). Build it after ragged_engine.attach / build: it
    attaches, in this order, the KVMeter, the Rollback (K, 2 K or (L4P) 2 K + 1 steps per group), the DraftResident and
    (L4, L4P) the joint-pass state; (L4P, pin on) the RequestPin. detach() undoes them in the reverse order. clock: an
    object with event() / record(e) / elapsed(a, b) in ms / sync() (r2_base_cell.CudaClock on the GPU, HostClock on
    the CPU)."""

    def __init__(self, model, rs, ops: SpecOps, clock, t0=None):
        self.model, self.rs, self.ops, self.clock = model, rs, ops, clock
        B, K = int(rs.B), ops.K
        self.B, self.K = B, K
        dev = rs.st.cu_q.device
        self.dev = dev
        paper = ops.level == "L4P"
        RE.set_activity(rs, torch.ones(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool))
        self.meter = KVMeter(rs, clock)
        n_group = K if ops.level == "L3" else (2 * K if ops.level == "L4" else 2 * K + 1)
        self.rb = RR.Rollback(rs, n_group, capacity=ops.journal, stream=ops.stream, neg=ops.rb_neg)
        self.draft = DraftResident(rs, self.rb)
        self.vs = None
        if ops.level in ("L4", "L4P"):
            from . import verify_rows as VR
            self.VR = VR
            # the joint pass's GEMM: the given one, else the ragged step's own (gemm natural: F.linear; exact: the
            # batch-invariant linear), so a rowgroup joint pass has the bits of the serial step it replaces
            base = ops.linear if ops.linear is not None else self.rb._orig["linear"]
            if paper:                                  # R = K + 2 rows (verify_rows calls it K + 1 "verify" rows)
                vops = VR.VerifyOps(K=K + 1, mode="exact", gemm=ops.gemm, linear=self._wait_linear(base),
                                    extra=ops.extra, attn=ops.attn)
            else:
                vops = VR.VerifyOps(K=K, mode=ops.mode, gemm=ops.gemm, linear=self._wait_linear(base),
                                    extra=ops.extra, attn=ops.attn, draft_last=(ops.mode == "fast"))
            self.vs = VR.attach(model, rs, vops)
            self.vs.hooks = self.rb
        self.pin = RequestPin(rs, neg=ops.neg) if ops.pin else None
        self.draft.pin = self.pin
        i64 = torch.int64
        self.t = torch.zeros(B, dtype=i64, device=dev) if t0 is None else t0.to(device=dev, dtype=i64).view(B).clone()
        # the drafts of the next verify (-1: none); L4P: the tentative drafts e_1 .. e_u (at most K + 1)
        self.D = torch.full((B, K + 1 if paper else K), -1, dtype=i64, device=dev)
        self.u = torch.full((B,), K, dtype=i64, device=dev)              # L4P: tentative drafts per request
        self.count = torch.zeros(B, dtype=i64, device=dev)              # committed tokens per request
        self.arK = torch.arange(K, dtype=i64, device=dev)
        self.arR = torch.arange(K + 2 if paper else K + 1, dtype=i64, device=dev)   # the joint pass's rows
        self.v_hist = torch.zeros(K + 1, dtype=i64, device=dev)         # L4P: verify rows per (request, round)
        self.round_accepted = []                                        # L4P: accepted drafts per round (device)
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
        past the capacity go to the dummy column. G has at most len(arR) columns (K for L3 / L4; K + 1 for an L4P
        plant)."""
        B, K = G.shape
        ar = self.arR[:K]                                                # 0 .. K - 1 (the same values as arK[:K])
        cols = self.count.view(B, 1) + ar.view(1, K)
        keep = (ar.view(1, K) < m.view(B, 1)) & (cols < self.cap)
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

    def _round_l4p(self, probe: bool) -> None:
        """L4P, one round: the paper's schedule (module docstring, steps 1-5)."""
        K, B = self.K, self.B
        R = K + 2
        neg = self.ops.neg
        rec = self.ops.record
        ev = []
        self._event(ev)
        D, u = self.D, self.u
        defer = 0 if neg == "verify_newest" else 1                       # NEGATIVE CONTROL verify_newest: no deferral
        v = (u - defer).clamp(min=0)                                     # verify rows: every tentative draft but newest
        ids = torch.cat((self.t.view(B, 1), D.clamp(min=0)), dim=1)     # (B, R): row r feeds x_r (a pad row: any token)
        self._sync_release()
        pos = (self.rs.st.twins[0].pos.view(B, 1) + self.arR.view(1, R)).to(torch.int32)
        rr = self.arR.view(R, 1)
        drows = rr >= v.view(1, B)                                       # (R, B): the deferred row, the draft row, pads
        drow = rr == u.view(1, B)                                        # (R, B): the draft row (its misses, its pin)
        self.rb.begin()                                                  # 1. the joint pass
        b0 = self.meter.blocks.clone()
        s0 = len(self.meter.spans)
        self.draft.begin("rows", redirect=self._edge_before(R), rows=drows, record=drow)
        self.meter.on, self.meter.timing = True, probe
        try:
            lg = self.VR.joint_pass(self.model, self.vs, ids, pos)
        finally:
            self.meter.on, self.meter.timing = False, False
            self.draft.end()
        if self.ops.prefetch is not None:                                # rung L5 seam: the draft row's misses
            self.ops.prefetch.on_draft(self.draft.miss_ids)
        self._event(ev)
        G = lg.argmax(dim=-1)                                            # (B, R)
        Gv = G[:, :K]                                                    # row r < v_b gives g_{r+1}: it checks e_{r+1}
        Dv = torch.where(self.arK.view(1, K) < v.view(B, 1), D[:, :K], torch.full_like(D[:, :K], -1))
        n = accept_prefix(Gv, Dv)                                        # 2. accept (greedy, paper 3.4)
        full = n >= v
        m = torch.where(full, v, n + 1)                                  # no bonus token after a full accept
        e_new = self._filter("draft_row", 0, G.gather(1, u.view(B, 1)).view(B), next_col=self.count + u)
        Gc = self._plant_commit(Gv, Dv, n, m)
        W = K
        mc = m
        if neg == "commit_deferred":                                     # NEGATIVE CONTROL: the deferred row's token
            W, mc, Gc = K + 1, torch.where(full, v + 1, m), G[:, :K + 1]
        FM, FS = lg[:, :W].amax(dim=-1), lg[:, :W].sum(dim=-1, dtype=torch.float64)
        cont = torch.ones_like(full) if neg == "keep_joint_draft" else full    # NEGATIVE CONTROL keep_joint_draft
        self.rb.truncate(torch.where(cont, u + 1, m))                    # 3. truncate (a reject: m rows, now)
        pos_trunc = None
        if rec:
            self._sync_release()
            pos_trunc = self.rs.st.twins[0].pos.clone()
        self._event(ev)
        self._commit_tokens(Gc, mc, FM, FS, lg[:, :W].clone() if self.lg_out is not None else None)
        t_new = Gc.gather(1, (m - 1).view(B, 1)).view(B)
        Dn = torch.full((B, K + 1), -1, dtype=torch.int64, device=self.dev)    # 4. K - 1 drafts, the next drafts
        if defer:
            e_u = D.gather(1, (u - 1).clamp(min=0).view(B, 1)).view(B)   # the deferred draft (newest of this group)
            Dn[:, 0].copy_(torch.where(cont, e_u, Dn[:, 0]))
        Dn[:, defer].copy_(torch.where(cont, e_new, Dn[:, defer]))
        shift = torch.where(cont, torch.full_like(u, defer + 1), torch.zeros_like(u))
        x = torch.where(cont, e_new, t_new)
        x0 = x.clone() if rec else None
        for j in range(K - 1):
            y = self._filter("draft", j, self._draft_step(x, R + j), next_col=self.count + shift + j)
            Dn.scatter_(1, (shift + j).view(B, 1), y.view(B, 1))
            x = y
        self._event(ev)
        keep = torch.where(full, u + 1, m) if neg == "commit_draft_rows" else m    # NEGATIVE CONTROL commit_draft_rows
        self.rb.commit(keep)                                             # 5. commit: the m verify rows
        self._event(ev)
        self.t = t_new
        self.D = Dn
        self.u = torch.where(cont, torch.full_like(u, K + defer), torch.full_like(u, K - 1))
        self._stats(n, v, mc, full)
        self.v_hist.scatter_add_(0, v, torch.ones_like(v))
        self.round_accepted.append(n.sum())
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if rec:
            self.records.append(dict(D=D.clone(), u=u.clone(), v=v.clone(), Dv=Dv.clone(), G=Gv.clone(),
                                     Gall=G.clone(), n=n.clone(), m=mc.clone(), full=full.clone(),
                                     committed=Gc.clone(), e_new=e_new.clone(), pos=self.pos_host.clone(),
                                     pos_trunc=pos_trunc, x0=x0, Dn=Dn.clone(), un=self.u.clone(),
                                     drows=drows.clone()))

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
        # a host sync (once, before the rounds). clone(): on a CPU device .cpu() returns the SAME tensor, and the mirror
        # must not follow the live twin counter (CPU job sp_a6eb5dd_a: round 0's filters and record read a moving pos)
        self.pos_host = self.rs.st.twins[0].pos.cpu().clone()
        self.rs.st.pos_host.copy_(self.pos_host)
        if self.ops.level in ("L4", "L4P"):
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
            if self.pin is not None:                                     # L4P, pin on: outside the group
                self._sync_release()
                self.pin.apply()
            if self.ops.level == "L3":
                self._round_l3(probe)
            elif self.ops.level == "L4":
                self._round_l4(probe)
            else:
                self._round_l4p(probe)
            if self.pin is not None:
                self._sync_release()
                self.pin.release()
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
        if self.ops.level == "L4P":
            # verify_hist[v]: (request, round) pairs whose joint pass had v verify rows; round_accepted: accepted drafts
            # per round; row_skipped[r]: slots that row index r skipped as a draft row; rows_draft: draft (request,
            # call) pairs of the joint passes (rows_calls calls)
            pin = None
            if self.pin is not None:
                pin = dict(on=self.pin.on, applies=self.pin.applies, releases=self.pin.releases,
                           recorded=int(self.pin.recorded), applied=int(self.pin.applied), unread=int(self.pin.unread))
            out.update(verify_hist=self.v_hist.cpu().tolist(),
                       round_accepted=[int(x) for x in torch.stack(self.round_accepted).cpu()] if self.round_accepted
                       else [], row_skipped=self.draft.row_skipped[:self.K + 2].cpu().tolist(),
                       rows_calls=self.draft.rows_calls, rows_draft=int(self.draft.rows_draft), pin=pin)
        return out


PHASES = {"L3": ("draft", "verify", "commit"), "L4": ("joint", "truncate", "draft", "commit"),
          "L4P": ("joint", "truncate", "draft", "commit")}
