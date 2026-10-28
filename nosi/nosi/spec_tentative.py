"""WS-A of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE TENTATIVE BUFFERS of the
paper (KS/paper-notes/speckv/speckv.txt, sections 3.3 and 3.4) and THE SCHEDULE OF CONTRACT 4 (the plan's human
decisions of 2026-10-28). OPT-IN: `--spec-rollback tentative` of benchmarks/Efficiency/r2_wsa_cell.py. With the flag off
no code imports this module; no engine module imports it; spec_round.py is unchanged (its L4P is the schedule of sp3a /
sp3b). Docs: docs/NOTES_wsa.md. NO NEW KERNEL: the shipped kernels (diff_offload, pool_update, flash_pool_swap,
flash_h2d_from_mask(_bias), r2_writeback.masked_writeback, flash_attn_nosa_with_kvcache with return_softmax_lse) and
torch ops.

TWO RUNNERS, ONE SCHEDULE (so the proof and the timing compare like with like)
  TentRunner          FAST: the tentative buffers (no journal). Levels L4P (the schedule below) and L3 (reference).
  PaperJournalRunner  EXACT: spec_round's journal (Rollback, DraftResident, the EXACT joint pass), the same schedule.
                      Its committed tokens and logits rows are bitwise SH's (the proof of the schedule).

THE DEPARTURES THAT TentRunner REMOVES (the plan's table)
  D1  the journal saves and undoes the whole cache residency of every tentative step. HERE: tentative K / V / cis rows
      live in a per-request TENTATIVE BUFFER, never in the committed tail slot or the window (paper 3.3); the residency
      changes of every row on the COMMITTED history stay (paper 3.4: "historical KV for the committed prefix remains
      valid and can stay cached"); a reject drops buffer entries and restores the compressed summaries; nothing else.
  D3  every joint pass recomputes the deferred row. HERE the deferred draft stays valid in the buffer: the twin replays
      the deferred position from its buffer entry (no GEMM, no residency, no attention); a joint pass has K + 1 rows.

THE SCHEDULE (contract 4: Figure 3 with K = 3, the newest token deferred; the human decisions (a) and (b)). Request b
has its committed position q (t, the last committed token, at q, not fed), u pending drafts e_1 .. e_u (e_i the token
of position q + i), and N - count tokens left. Rounds are batch-wide: one joint slot J, then K - 1 draft slots D (the
first round is preceded by K draft slots P: the initial fill). The host plans each round from the state that the
round's one host sync read (count, q, u); per request and round the ROLE is:
  done    no token left: nothing.
  target  (ordinary decoding, after the final group) J: one target row at q (the shipped step's work), commit 1;
          each D slot: the same while tokens are left; else idle.
  final   (decision (b): N - count <= K) J: ONE final verification group of v = min(u, N - count) target rows that
          verifies e_1 .. e_v (no deferral, no draft row); commit the accepted prefix and, on a reject, the correction
          (never an unverified token; never past N); the D slots of this round idle; from the next round: target.
  fill    (u < K: decision (a), a new fill after a reject) J: one draft row feeds e_u at q + u and drafts e_{u+1}; the
          D slots of this round idle; the next round is a group with u = K, as the first group of Figure 3.
  group   (u >= K) J: v = u - 1 verify rows check e_1 .. e_v, e_u is DEFERRED, the draft row feeds e_u and drafts the
          next token. Greedy acceptance, no bonus token: a full accept commits v tokens and keeps [e_u, e_new]; a
          reject commits the accepted prefix and the correction and STARTS A NEW FILL from them (decision (a)): the D
          slots draft d1', d2' from the correction, the next J drafts d3' (role fill), the next group verifies d1', d2',
          defers d3', drafts d4' (Figure 3 from its first group). The old "draft K - 1, verify K - 2" restart is gone.
          D slots: draft.
So every request commits exactly N tokens. Per pass and request the runner keeps a POSITION TRACE (fed positions,
verified tokens' positions, the deferred token's position, the drafted token's position, the accepted count, the
committed count, and the positions a pass REMADE: the EXACT runner's deferred row and fill rows, FAST none); the GPU cell
prints the first TRACE_N passes and the last TRACE_N passes of every request.

THE STATE OF TentRunner, per request b (device tensors)
  qc_b      the committed position: the committed cache (window tail slot, host rows, twin rows below the committed
            counters) holds positions < qc_b.
  the chain the tentative positions qc_b .. (the frontier - 1), fed in position order. Their K, V and cis rows are in the
            BUFFER only.
  ctr_c     the committed twin counters (pos, n_comp, n_ccis): the same in every layer.
THE BUFFER, per layer: k, v (B, 64, H, D), bias (B, 64, H) (the cis row of the position, as the tail row holds it);
position x is row x % 64 (its tail row index); tpos (B, 64), shared by the layers: the position each row holds. A row at
position p reads the entries with qc <= tpos <= p: the chain feeds positions in order, so every position in [qc, p] was
written by the current chain before the row at p reads it; an entry of a dropped chain lies above the frontier. The
chain spans at most K + 2 positions past qc (< MAX_ROLLBACK = 32: the twin's ring stays exact).

ONE TENTATIVE ROW (draft row, verify row, target row), per layer, in the shipped order:
  1. the GEMMs, nosa_linear, rope (one row per request: the ragged step's GEMM; a joint pass: verify_rows' GEMM).
  2. the twin step; a group's draft row first REPLAYS the deferred position (twin.step_unchecked with the K and cis row
     that the buffer holds for it). Before the joint pass the twins of group / final / target requests are reset to
     ctr_c; a fill request keeps its twin at its frontier (its chain is all drafts).
  3. the selection (one row: ragged_engine's stage 1 and captured pooling / top-k graph; a joint pass:
     verify_rows._select_rows).
  4. the RESIDENCY of the selection on the COMMITTED window: diff_offload, pool_update, flash_pool_swap, the map, the
     LRU clock (the shipped order), an inactive row masked. KEPT, never undone. A draft row skips every miss (Task
     1.1's draft residency); every row skips the TENTATIVE BLOCK (an id above qc // 64: past an edge a row selects the
     block it writes, which has no host rows yet); a head of a row past the edge without the committed tail block
     qc // 64 gets that id in place of the tentative block (the diff keeps slot 63; the head masks slot 63). The
     shipped gathers load the other misses.
  5. its K, V and cis row into the BUFFER at row p % 64, and tpos (never into the committed tail slot).
  6. attention: call A (the committed window: a scratch bias that masks the skipped slots, cut at the committed rows of
     the tail block: base + min(qc % 64, n_tail(p)) in block qc // 64, base + qc % 64 past its edge; n_tail(p) = the
     shipped attention length of the step at p minus base, findings F4 kept) + call B (the buffer: entry x visible iff
     qc <= x <= p, x % 64 < n_tail(p) when x is in p's block, and, past the edge, x in block qc // 64 only for a head
     that selects qc // 64), merged by their log-sum-exp (verify_rows FAST's merge).
ACCEPT (commit): the first m[b] chain positions become committed: their buffer rows of block qc // 64 into the tail
slot (one where() per tensor: the buffer row and the tail row of a position have the same index); a request whose
committed block fills writes it to its host rows with the shipped device-driven write-back (only committed rows); its
rows of the next block into the tail slot; the tail id and the attention length; qc += m; ctr_c <- the counters after m.
REJECT / ROUND START (reset): the twins of the selected requests <- ctr_c (r2_positional_twin, ROLLBACK: a counter reset
is exact up to MAX_ROLLBACK positions). The rejected entries are dropped by qc and the frontier only (no data moves). No
slot byte, map entry, pool entry or LRU age moves.

WHY FAST AND NOT EXACT (what differs from SH; why it is not semantic; docs/NOTES_wsa.md section 3)
  - the window and pool CONTENTS and their slot ORDER (a cache: every named slot holds its block's host rows);
  - FLOAT ORDER of attention (slot order; the two calls merged in float32): K / V / logits from layer 1 on differ by
    rounding; layer 0 is bitwise SH's;
  - the DRAFTS (the deferred K / V are the draft step's, not a recomputation): acceptance and the trace differ from the
    journal runner's; the committed tokens are the target's.
  The CPU stand-ins model the arithmetic, not GPU dispatch, streams, ordering, timing or memory: the GPU cell must show
  those (docs/NOTES_wsa.md section 5).

CHECK-ONLY OPERATIONS (check rounds only: every round in the CPU tests, the probe rounds in the GPU cell, which the
steady median excludes): TentRunner's logits fingerprints, the hooks, the extended counters (forced / injected skips,
rows past an edge, F4 fill rows), the poison of dead entries (poison=True, tests only), the buffer-write record (rec,
tests only). REQUIRED (never check-only): every op of steps 1-6, the commit, the reset, the argmax and the acceptance,
tpos, the deferred-write counter (the D3 contract), the draft skip counter, the host plan and the trace (host only).
HOOKS for WS-G (TentRunner; called on check rounds): on_select(l, kind, r, pos (B,), verify (B,) bool, sel (H, B, topk)):
the TARGET selection of every row and layer, before the tail-id injection; on_logits(round, kind, pos (B, R), verify
(R, B) bool, logits (B, R, V)).
HOST SYNCS: one per round after the round, one after the warm step. None inside a round.
NEGATIVE CONTROLS (tests only; tneg): recompute_deferred (a group's joint rows remake the deferred row: EXACT's layout),
no_causal (call B ignores x <= p), no_lower (call B ignores qc <= x), undo_residency (reset() restores the window and
pool maps of the last round start), writeback_tentative (the commit writes the tail block to the host when the CHAIN
filled it), no_replay (the draft row skips the deferred replay).
"""
from __future__ import annotations

import collections
import time

import torch
import torch.nn.functional as F

from . import ragged_engine as RE
from . import spec_round as SR
from . import verify_rows as VR
from .r2_positional_twin import BLOCK, MAX_ROLLBACK

MASK_BIAS = SR.MASK_BIAS
POISON_V = 1.0e3                        # check-only: a dead entry's V; a masked read weighs it 0, a live read shows it
LEVELS_T = ("L3", "L4P")
NEG_T = (None, "recompute_deferred", "no_causal", "no_lower", "undo_residency", "writeback_tentative", "no_replay")
SR_NEG_T = (None, "commit_draft_at_reject", "miscount")
PHASES_T = {"L3": ("reset", "draft", "reset2", "verify", "commit"), "L4P": ("reset", "joint", "commit", "draft")}
PHASES_J = ("joint", "truncate", "draft", "commit")              # PaperJournalRunner: spec_round's L4P phases
ROLLBACK_PHASES_T = {"L3": ("reset", "reset2", "commit"), "L4P": ("reset", "commit")}
COUNTERS_T = ("rows", "draft_rows", "verify_rows", "replays", "deferred_writes", "appended", "crossed", "dropped",
              "span_max", "edge_rows", "fill_rows")
GROUP, FILL, FINAL, TARGET, DONE = 0, 1, 2, 3, 4
ROLE_NAMES = ("group", "fill", "final", "target", "done")
TRACE_N = 12
TOK_KEYS = ("proposed", "verified", "accepted", "corrected", "group_committed", "final_committed", "target_committed",
            "unverified_discarded", "groups", "fills", "finals", "targets", "remade_rows")


class _Plan:
    """Attributes only (a pass's rows, a round's roles)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------ the tentative buffers
class TentState:
    """The tentative buffers and the committed position of one RaggedState (module docstring). It installs NO hook on
    the engine: a tentative pass runs its own layer loop (run_pass); commit and reset are explicit calls."""

    def __init__(self, rs, attn, tneg=None, poison: bool = False):
        if tneg not in NEG_T:
            raise SR.SpecError("unknown negative control %r (one of %s)" % (tneg, NEG_T))
        eng0 = rs.cache.layers[0].cache_engine
        self.rs, self.attn, self.tneg, self.poison = rs, attn, tneg, bool(poison)
        self.L, self.B = int(rs.L), int(rs.B)
        self.H, self.D = int(eng0.head_num), int(eng0.head_dim)
        self.topk, self.bs = int(eng0.topk), int(eng0.block_size)
        if self.bs != BLOCK or int(eng0._tail_block_idx_on_gpu) != self.topk - 1:
            raise SR.SpecError("the engine must have %d-token blocks and the tail slot topk - 1" % BLOCK)
        self.tail = self.topk - 1
        self.base = self.tail * self.bs
        self.pool = bool(rs.pool)
        dev = eng0._k_gpu.device
        self.dev = dev
        for l, lay in enumerate(rs.cache.layers):
            e, tw = lay.cache_engine, rs.st.twins[l]
            if e._k_gpu.dtype != tw.ring.dtype or e._kv_bias_gpu.dtype != tw.total_cis.dtype:
                raise SR.SpecError("layer %d: the buffer must hold the twin's key / cis bits (the deferred replay)" % l)
        B, H, D, L = self.B, self.H, self.D, self.L
        i64 = torch.int64
        # zeros, not torch.empty: a masked entry still enters the kernel's sums with weight 0 (0 x NaN = NaN)
        self.k = [torch.zeros((B, BLOCK, H, D), dtype=eng0._k_gpu.dtype, device=dev) for _ in range(L)]
        self.v = [torch.zeros((B, BLOCK, H, D), dtype=eng0._v_gpu.dtype, device=dev) for _ in range(L)]
        self.bias = [torch.zeros((B, BLOCK, H), dtype=eng0._kv_bias_gpu.dtype, device=dev) for _ in range(L)]
        self.tpos = torch.full((B, BLOCK), -1, dtype=i64, device=dev)
        self.qc = torch.zeros((B,), dtype=i64, device=dev)
        self.ctr_c = torch.zeros((3, B), dtype=i64, device=dev)
        self.scratch = torch.empty_like(eng0._kv_bias_gpu)      # call A's bias: copied from the live view before use
        self.scratch_att = self.scratch[:, :self.topk * self.bs] if self.pool else self.scratch
        self.maskb = torch.full((), MASK_BIAS, dtype=eng0._kv_bias_gpu.dtype, device=dev)
        self.poisonv = torch.full((), POISON_V, dtype=eng0._v_gpu.dtype, device=dev)
        self.all_b = torch.arange(B, dtype=i64, device=dev)
        self.ar64 = torch.arange(BLOCK, dtype=i64, device=dev)
        self.minus1 = torch.tensor(-1, dtype=i64, device=dev)
        self.len_b = torch.full((B,), BLOCK, dtype=torch.int32, device=dev)
        self.skipped = torch.zeros((L,), dtype=i64, device=dev)     # draft rows' skipped (head, request) loads
        self.forced = torch.zeros((L,), dtype=i64, device=dev)      # check-only: tentative-block loads skipped
        self.injected = torch.zeros((L,), dtype=i64, device=dev)    # check-only: committed tail ids put back
        self.c = {n: torch.zeros((), dtype=i64, device=dev) for n in COUNTERS_T}
        self.calls = dict(reset=0, commit=0, passes=0, joint_passes=0, joint_rows=0, j1_passes=0, draft_steps=0,
                          verify_steps=0, mixed_steps=0, replay_calls=0, residency_undo=0)     # host counts
        self._ctr_dst = ([tw.pos for tw in rs.st.twins] + [tw.n_comp for tw in rs.st.twins]
                         + [tw.n_ccis for tw in rs.st.twins])
        self.checking = False
        self.hooks = None
        self.rec = None                                             # check-only (tests): a list -> layer-0 row writes
        self.round = -1
        self._maps0 = None                                          # NEGATIVE CONTROL undo_residency only

    # -------------------------------------------------------------- the committed position and the summaries
    def counters0(self) -> torch.Tensor:
        """(3, B): layer 0's twin counters (pos, n_comp, n_ccis), a new tensor."""
        tw = self.rs.st.twins[0]
        return torch.stack((tw.pos, tw.n_comp, tw.n_ccis))

    def init_committed(self) -> None:
        """After the warm steps (outside a round): qc and ctr_c from layer 0's twin; no entry is live."""
        self.qc.copy_(self.rs.st.twins[0].pos)
        self.ctr_c.copy_(self.counters0())
        self.tpos.fill_(-1)

    def set_twins(self, tgt) -> None:
        """Every layer's twin counters <- tgt (3, B): one fused copy (torch._foreach_copy_) when torch has it."""
        n = self.L
        src = [tgt[0]] * n + [tgt[1]] * n + [tgt[2]] * n
        if hasattr(torch, "_foreach_copy_"):
            torch._foreach_copy_(self._ctr_dst, src)
        else:
            for d, s in zip(self._ctr_dst, src):
                d.copy_(s)

    def begin_round(self) -> None:
        if self.tneg == "undo_residency":                          # NEGATIVE CONTROL (tests only)
            self._maps0 = [(lay.cache_engine._block_map.clone(),
                            lay.cache_engine._pool_map.clone() if self.pool else None) for lay in self.rs.cache.layers]

    def reset(self, mask=None) -> None:
        """REJECT / ROUND START: the twins of the requests in mask (B,) bool (all: None) back to the committed counters.
        Writes only the 3 L twin counters; no slot byte, map entry, pool entry or LRU age."""
        self.calls["reset"] += 1
        if self.tneg == "undo_residency" and self._maps0 is not None:     # NEGATIVE CONTROL: a residency undo
            self.calls["residency_undo"] += 1
            for lay, (bm, pm) in zip(self.rs.cache.layers, self._maps0):
                lay.cache_engine._block_map.copy_(bm)
                if pm is not None:
                    lay.cache_engine._pool_map.copy_(pm)
        if mask is None:
            self.set_twins(self.ctr_c)
        else:
            self.set_twins(torch.where(mask.view(1, -1), self.ctr_c, self.counters0()))

    def commit(self, m, ctr_m, frontier, dropped) -> None:
        """ACCEPT (module docstring): request b's chain positions [qc, qc + m[b]) become committed. ctr_m (3, B): the
        twin counters after them (a request with m = 0: ctr_c); frontier (B,): layer 0's twin position now (the
        writeback_tentative control only); dropped (B,): the chain entries this round discards (counted). No host
        read; the same ops for every m."""
        rs, B, bs = self.rs, self.B, self.bs
        self.calls["commit"] += 1
        q0 = self.qc
        q1 = q0 + m
        b0 = q0 // bs
        x0 = b0.view(B, 1) * bs + self.ar64.view(1, bs)                # (B, 64): the positions of block b0's rows
        m1 = (x0 >= q0.view(B, 1)) & (x0 < q1.view(B, 1))
        fill = (q1 // bs) > b0
        full = ((frontier // bs) > b0) if self.tneg == "writeback_tentative" else fill   # NEGATIVE CONTROL: the chain's
        m2 = fill.view(B, 1) & (x0 + bs < q1.view(B, 1))                # the next block's committed rows
        wpos = b0 * bs + (bs - 1)
        tid = (q1 // bs).view(1, B).expand(self.H, B)
        eng0 = rs.cache.layers[0].cache_engine
        cl = rs.body.cache_lens_value(eng0, q1 - 1).to(torch.int32)    # the shipped length after the step at q1 - 1
        m1k, m1b = m1.view(B, bs, 1, 1), m1.view(B, bs, 1)
        m2k, m2b = m2.view(B, bs, 1, 1), m2.view(B, bs, 1)
        t0, t1 = self.base, self.base + bs
        for l, lay in enumerate(rs.cache.layers):
            e = lay.cache_engine
            kt, vt, bt = e._k_gpu[:, t0:t1], e._v_gpu[:, t0:t1], e._kv_bias_gpu[:, t0:t1]
            torch.where(m1k, self.k[l], kt, out=kt)                    # in place (TensorIterator: full overlap)
            torch.where(m1k, self.v[l], vt, out=vt)
            torch.where(m1b, self.bias[l], bt, out=bt)
            rs.body.writeback_kernel(e._k_gpu, e._v_gpu, e._k_cpu, e._v_cpu, full, wpos, t0, bs)
            torch.where(m2k, self.k[l], kt, out=kt)
            torch.where(m2k, self.v[l], vt, out=vt)
            torch.where(m2b, self.bias[l], bt, out=bt)
            e._block_map[:, :, self.tail].copy_(tid)
            e._cache_lens.copy_(cl)
        c = self.c
        c["appended"] += m.sum()
        c["crossed"] += fill.sum()
        c["dropped"] += dropped.sum()
        self.qc.copy_(q1)
        self.ctr_c.copy_(ctr_m)

    def poison_dead(self, live_after) -> None:
        """CHECK-ONLY (tests, poison=True): V <- POISON_V for every entry outside [qc, qc + live_after) (published or
        dropped): a later read of such an entry shows in the logits."""
        if not self.poison:
            return
        B = self.B
        dead = (self.tpos < self.qc.view(B, 1)) | (self.tpos >= (self.qc + live_after).view(B, 1))
        for l in range(self.L):
            self.v[l].copy_(torch.where(dead.view(B, self.bs, 1, 1), self.poisonv, self.v[l]))

    # -------------------------------------------------------------- one tentative pass
    @torch.inference_mode()
    def run_pass(self, model, plan):
        """R rows per request (plan: R, kind, ids (B, R), pos (B, R) int32, act / verify / draft (R, B) bool, replay
        (R, B) bool or None, replay_rows (host tuple), deferred (B,) or None, vs (R > 1), lin, snap). Returns logits
        (B, R, V); with snap, plan.ctr[r] = layer 0's counters after row r."""
        rs = self.rs
        B, R = self.B, plan.R
        E = B * R
        plan.ctr = [None] * R
        hidden = F.embedding(plan.ids.reshape(E, 1), model.embed_tokens)
        pos_flat = plan.pos.reshape(E)
        for l, layer in enumerate(model.layers):
            hidden = self._layer(l, layer, hidden, pos_flat, model.cos_sin_cache, plan)
        hidden = rs.nl.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
        logits = plan.lin(hidden, model.lm_head).float()
        rs.st.steps += R
        self.calls["passes"] += 1
        return logits.reshape(B, R, -1)

    def _layer(self, l, layer, hidden, pos_flat, cos_sin_cache, plan):
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
            sel = VR._select_rows(plan.vs, layer, q, twin)
        q4 = q.view(B, R, q.shape[1], q.shape[2])
        outs = [self._row(l, eng, twin, r, sel[:, :, r], q4[:, r], k4[:, r], v4[:, r], pbs[r], plan)
                for r in range(R)]
        attn = torch.stack(outs, 1).reshape(E, 1, -1)
        hidden = residual + lin(attn, layer.wo)
        residual = hidden
        h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = lin(h, layer.gate_up_proj)
        out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        NL.silu_and_mul(h, out)
        return residual + lin(out, layer.down_proj)

    def _replay(self, l, twin, act, deferred) -> None:
        """The deferred position's twin step from its buffer entry (the draft step's K and cis row): no GEMM, no
        residency, no attention (D3)."""
        if l == 0:
            self.calls["replay_calls"] += 1
        if self.tneg == "no_replay":                                   # NEGATIVE CONTROL (tests only)
            return
        i = deferred % BLOCK
        kk = self.k[l][self.all_b, i]
        cc = self.bias[l][self.all_b, i]
        twin.step_unchecked(kk, cc, act)
        if l == 0:
            self.c["replays"] += act.sum()
            if self.rec is not None:                                   # CHECK-ONLY (tests)
                self.rec.append(dict(round=self.round, kind="replay", act=act.clone(), p=deferred.clone(),
                                     k=kk.clone(), c=cc.clone()))

    def _row(self, l, eng, twin, r, sel, q_r, k_r, v_r, p, plan):
        """Steps 4-6 of the module docstring for row r of every request. Returns (B, Hq * D)."""
        rs, B, bs = self.rs, self.B, self.bs
        kern = rs.ops.kernels
        act, drf = plan.act[r], plan.draft[r]
        qc = self.qc
        qb = qc // bs
        qb3 = qb.view(1, B, 1)
        bp = p // bs
        past = bp > qb                                                 # (B,): the row is past the committed block's edge
        hasq = (sel == qb3).any(-1, keepdim=True)                      # (H, B, 1)
        if self.hooks is not None and hasattr(self.hooks, "on_select"):    # CHECK-ONLY (WS-G)
            self.hooks.on_select(l, plan.kind, r, p, plan.verify[r], sel)
        inj = (sel > qb3) & ~hasq
        ti = torch.where(inj, qb3.expand_as(sel), sel).contiguous()
        # 4. the residency on the committed window (kept)
        kern["diff"](eng._block_map, ti, eng._new_block_map_buf, eng._load_mask)
        am = act.view(1, B, 1)
        eng._load_mask.masked_fill_(~am, -1)                           # an inactive row loads nothing, moves nothing
        if self.pool:
            kern["pool_update"](eng._block_map, eng._load_mask, eng._pool_map, eng._pool_age, eng._pool_target,
                                eng._pool_action, eng._pool_stamp_base, eng._tail_block_idx_on_gpu)
            kern["pool_swap"](eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, eng._pool_target, eng._pool_action, eng.topk,
                              eng.block_size)
            eng._pool_stamp_base += eng.topk
        eng._new_block_map_buf.copy_(torch.where(am, eng._new_block_map_buf, eng._block_map))
        eng._block_map.copy_(eng._new_block_map_buf)
        lm = eng._load_mask
        tent = (lm >= 0) & (lm > qb3)
        skip = (lm >= 0) & (drf.view(1, B, 1) | tent)
        ids = torch.where(skip, self.minus1, lm)
        kern["gather_k"](eng._k_gpu, eng._k_cpu, ids, bs)
        kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, ids, bs)
        eng._block_map.masked_fill_(skip, -1)                          # the next row loads a skipped block again
        self.skipped[l] += (skip & ~tent).sum()
        if self.checking:                                              # CHECK-ONLY counters
            self.forced[l] += tent.sum()
            self.injected[l] += inj.sum()
        # 5. the row's K, V and cis row into the buffer (never into the committed tail slot)
        bi = self.all_b
        i = p % bs
        kb, vb, bb = self.k[l], self.v[l], self.bias[l]
        a3, a2 = act.view(B, 1, 1), act.view(B, 1)
        kb[bi, i] = torch.where(a3, k_r.to(kb.dtype), kb[bi, i])
        vb[bi, i] = torch.where(a3, v_r.to(vb.dtype), vb[bi, i])
        prow = p.clamp(max=int(twin.total_cis.shape[1]) - 1)
        bb[bi, i] = torch.where(a2, twin.total_cis[bi, prow].to(bb.dtype), bb[bi, i])
        if l == 0:
            self.tpos[bi, i] = torch.where(act, p, self.tpos[bi, i])
            c = self.c
            c["rows"] += act.sum()
            c["draft_rows"] += (act & drf).sum()
            c["verify_rows"] += plan.verify[r].sum()
            if plan.deferred is not None:                              # the D3 contract: 0 in the tentative mode
                c["deferred_writes"] += (act & drf & (p == plan.deferred)).sum()
            c["span_max"].copy_(torch.maximum(c["span_max"], torch.where(act, p + 1 - qc, torch.zeros_like(p)).max()))
            if self.checking:
                c["edge_rows"] += (act & past).sum()
                c["fill_rows"] += (act & (p % bs == bs - 1)).sum()
            if self.rec is not None:                                   # CHECK-ONLY (tests)
                self.rec.append(dict(round=self.round, kind=plan.kind, r=r, act=act.clone(), draft=drf.clone(),
                                     p=p.clone(), rope=plan.pos[:, r].to(torch.int64).clone(), k=k_r.clone(),
                                     c=bb[bi, i].clone(), verify=plan.verify[r].clone(), qc=qc.clone()))
        # 6. attention: A (the committed window) + B (the buffer), merged by the log-sum-exp
        n_tail = rs.body.cache_lens_value(eng, p) - self.base           # (B,): rows of p's block the shipped step attends
        cq = qc % bs
        cut = (self.base + torch.where(past, cq, torch.minimum(cq, n_tail))).to(torch.int32)
        m63 = (~hasq) & past.view(1, B, 1)
        mslot = torch.cat((skip[:, :, :self.tail], skip[:, :, self.tail:] | m63), dim=2)
        rows = mslot.permute(1, 2, 0).unsqueeze(2).expand(B, self.topk, bs, self.H).reshape(B, self.topk * bs, self.H)
        if self.pool:
            live, katt, vatt = eng._kv_bias_gpu_att, eng._k_gpu_att, eng._v_gpu_att
        else:
            live, katt, vatt = eng._kv_bias_gpu, eng._k_gpu, eng._v_gpu
        self.scratch_att.copy_(live)
        self.scratch_att.masked_fill_(rows, MASK_BIAS)
        nsplit = rs.nl._ATTN_SPLITS
        qq = q_r.unsqueeze(1).contiguous()
        o_a, lse_a = self.attn(qq, katt, vatt, self.scratch_att, cache_seqlens=cut, num_splits=nsplit,
                               return_softmax_lse=True)
        tp = self.tpos
        lo = (tp >= 0) if self.tneg == "no_lower" else (tp >= qc.view(B, 1))       # NEGATIVE CONTROLS (tests only)
        hi = (tp >= 0) if self.tneg == "no_causal" else (tp <= p.view(B, 1))
        tb = tp // bs
        own = tb == bp.view(B, 1)
        ok = lo & hi & (~own | ((tp % bs) < n_tail.view(B, 1)))
        cmt = (tb == qb.view(B, 1)) & past.view(B, 1)
        vis = ok.view(B, bs, 1) & (~cmt.view(B, bs, 1) | hasq.permute(1, 2, 0))
        bias_b = torch.where(vis, bb, self.maskb)
        o_b, lse_b = self.attn(qq, kb, vb, bias_b, cache_seqlens=self.len_b, num_splits=nsplit,
                               return_softmax_lse=True)
        Hq = q_r.shape[1]
        la = lse_a.reshape(B, Hq, 1).float()
        lb = lse_b.reshape(B, Hq, 1).float()
        mx = torch.maximum(la, lb)
        wa, wb = torch.exp(la - mx), torch.exp(lb - mx)
        o = (o_a.reshape(B, Hq, -1).float() * wa + o_b.reshape(B, Hq, -1).float() * wb) / (wa + wb)
        return o.to(q_r.dtype).reshape(B, -1)

    # -------------------------------------------------------------- the report
    def report(self) -> dict:
        """ONE host sync (after a run): the counters."""
        names = list(COUNTERS_T)
        vals = torch.stack([self.c[n] for n in names]).cpu().tolist()
        out = {n: int(x) for n, x in zip(names, vals)}
        out.update(skipped=self.skipped.cpu().tolist(), forced=self.forced.cpu().tolist(),
                   injected=self.injected.cpu().tolist(), calls=dict(self.calls), tneg=self.tneg,
                   poison=self.poison)
        return out


# ------------------------------------------------------------------ the schedule of contract 4 (both runners)
class PaperSchedule:
    """THE SCHEDULE (module docstring) for an L4P runner: a host plan per round from the state that the round's one host
    sync read. The runner supplies _ps_begin(), _ps_prelude(), _ps_round(P, probe) -> (n, m, full) (device, (B,)),
    _ps_qpos() (the committed positions, device), _remade(role, q, u, v) and _set_check(g, probe)."""

    _fp_check_only = True

    def _commit_tokens_t(self, G, m, FM, FS, LG) -> None:
        """spec_round's _commit_tokens; the fingerprinted committed columns are marked (fp_ok)."""
        if self._check or not self._fp_check_only:
            B, K = G.shape
            ar = self.arR[:K]
            cols = self.count.view(B, 1) + ar.view(1, K)
            keep = (ar.view(1, K) < m.view(B, 1)) & (cols < self.cap)
            self.fp_ok.scatter_(1, torch.where(keep, cols, torch.full_like(cols, self.cap)), keep)
        self._commit_tokens(G, m, FM, FS, LG)

    def _fingerprints(self, lg):
        """(max float32, sum float64) of the rows lg (B, W, V); zeros off check rounds when they are check-only."""
        W = lg.shape[1]
        if self._fp_check_only and not self._check:
            return self.zfm[:, :W], self.zfs[:, :W]
        return lg.amax(dim=-1), lg.sum(dim=-1, dtype=torch.float64)

    def _ps_plan(self):
        """The roles of this round (module docstring, THE SCHEDULE) and their device masks (one host-to-device copy)."""
        B, K, N = self.B, self.K, self.N
        role, v, left = [], [], []
        for b in range(B):
            lf = N - self.h_count[b]
            u = self.h_u[b]
            if lf <= 0:
                r = DONE
            elif self.h_target[b]:
                r = TARGET
            elif lf <= K:
                r = FINAL
            elif u < K:
                r = FILL
            else:
                r = GROUP
            role.append(r)
            v.append(u - 1 if r == GROUP else min(u, lf) if r == FINAL else 1 if r == TARGET else 0)
            left.append(max(lf, 0))
        pin = self.dev.type == "cuda"                                  # pinned + non_blocking: no stream sync here
        d = torch.tensor([role, v, left], dtype=torch.int64, pin_memory=pin).to(self.dev, non_blocking=pin)
        P = _Plan(role=role, v_h=v, left=left, q=list(self.h_q), u=list(self.h_u), count=list(self.h_count))
        rl, P.v, lf = d[0], d[1], d[2]
        P.isS, P.isF, P.isX, P.isO, P.isDONE = rl == GROUP, rl == FILL, rl == FINAL, rl == TARGET, rl == DONE
        P.dO = [P.isO & (lf - 1 - j > 0) for j in range(K - 1)]
        P.h_dS = [r == GROUP for r in role]
        P.h_dO = [[role[b] == TARGET and left[b] - 1 - j > 0 for b in range(B)] for j in range(K - 1)]
        P.any_o = [any(x) for x in P.h_dO]
        P.any_d = [any(P.h_dS) or P.any_o[j] for j in range(K - 1)]
        P.rows = max([v[b] + 1 if role[b] == GROUP else v[b] if role[b] == FINAL else 1 if role[b] in (FILL, TARGET)
                      else 0 for b in range(B)] + [0])
        P.reset = P.isS | P.isX | P.isO
        return P

    def _te(self, slot, role, **kw):
        """A trace entry: q is the committed position at the start of the round (J entries; None otherwise)."""
        e = dict(round=self.round, slot=slot, role=role, q=None, fed_v=[], fed_d=[], verify=[], deferred=None,
                 drafted=None, n=None, committed=0, remade=[])
        e.update(kw)
        return e

    def _tr_add(self, b, entries) -> None:
        for e in entries:
            if self.trace_all is not None:
                self.trace_all[b].append(e)
            if len(self.trace_head[b]) < TRACE_N:
                self.trace_head[b].append(e)
            self.trace_tail[b].append(e)

    def _ps_after(self, P, hc) -> None:
        """After the round's sync: the trace of the round (host), the token accounting, the host state."""
        B, K = self.B, self.K
        cnt, qn, un, nn, mm, ff = hc.tolist()                          # host values of the round's one sync
        tt = self.tok_tot
        for b in range(B):
            r, q, u, v = P.role[b], P.q[b], P.u[b], P.v_h[b]
            n, m, full = int(nn[b]), int(mm[b]), bool(ff[b])
            es = []
            if r == GROUP:
                es.append(self._te("J", "group", q=q, fed_v=list(range(q, q + v)), fed_d=[q + u],
                                   verify=list(range(q + 1, q + v + 1)), deferred=q + u, drafted=q + u + 1, n=n,
                                   committed=m, remade=self._remade(GROUP, q, u, v)))
                base = q + u + 1 if full else q + m
                es += [self._te("D%d" % j, "draft", fed_d=[base + j], drafted=base + j + 1) for j in range(K - 1)]
                tt["proposed"] += K
                tt["verified"] += v
                tt["accepted"] += n
                tt["corrected"] += int(n < v)
                tt["group_committed"] += m
                tt["groups"] += 1
            elif r == FILL:
                es.append(self._te("J", "fill", q=q, fed_d=[q + u], drafted=q + u + 1,
                                   remade=self._remade(FILL, q, u, 0)))
                es += [self._te("D%d" % j, "idle") for j in range(K - 1)]
                tt["proposed"] += 1
                tt["fills"] += 1
            elif r == FINAL:
                es.append(self._te("J", "final", q=q, fed_v=list(range(q, q + v)),
                                   verify=list(range(q + 1, q + v + 1)), n=n, committed=m))
                es += [self._te("D%d" % j, "idle") for j in range(K - 1)]
                tt["verified"] += v
                tt["accepted"] += n
                tt["corrected"] += int(n < v)
                tt["final_committed"] += m
                tt["unverified_discarded"] += u - v
                tt["finals"] += 1
            elif r == TARGET:
                es.append(self._te("J", "target", q=q, fed_v=[q], committed=1))
                tt["target_committed"] += 1
                tt["targets"] += 1
                for j in range(K - 1):
                    if P.h_dO[j][b]:
                        es.append(self._te("D%d" % j, "target", fed_v=[q + 1 + j], committed=1))
                        tt["target_committed"] += 1
                        tt["targets"] += 1
                    else:
                        es.append(self._te("D%d" % j, "idle"))
            tt["remade_rows"] += sum(len(e["remade"]) for e in es)
            self._tr_add(b, es)
            self.h_count[b], self.h_q[b], self.h_u[b] = int(cnt[b]), int(qn[b]), int(un[b])
            if r in (FINAL, TARGET):
                self.h_target[b] = True

    @torch.inference_mode()
    def _run_paper(self, N: int, max_rounds: int = 0, warm: int = 1) -> dict:
        """THE RUN of the schedule: `warm` plain steps, the initial fill (K draft slots), then the rounds until every
        request has exactly N committed tokens (at most max_rounds; default 4 N + 16). One host sync per round."""
        B, K = self.B, self.K
        N = int(N)
        if int(warm) < 1 or N <= int(warm):
            raise SR.SpecError("warm >= 1 and N > warm (the first step after the prefill is a plain step)")
        self.warm_steps = int(warm)
        self.N = N
        self.cap = N + K
        dev = self.dev
        self.out = torch.full((B, self.cap + 1), -1, dtype=torch.int64, device=dev)
        self.fmax = torch.zeros((B, self.cap + 1), dtype=torch.float32, device=dev)
        self.fsum = torch.zeros((B, self.cap + 1), dtype=torch.float64, device=dev)
        self.fp_ok = torch.zeros((B, self.cap + 1), dtype=torch.bool, device=dev)
        if self.ops.record:
            V = int(self.model.lm_head.shape[0])
            self.lg_out = torch.zeros((B, self.cap + 1, V), dtype=torch.float32, device=dev)
        limit = int(max_rounds) if max_rounds > 0 else 4 * N + 16
        pe = self.ops.probe_every
        self.trace_all = [[] for _ in range(B)] if self.ops.record else None
        self.trace_head = [[] for _ in range(B)]
        self.trace_tail = [collections.deque(maxlen=TRACE_N) for _ in range(B)]
        self.tok_tot = {k: 0 for k in TOK_KEYS}
        self.clock.sync()
        t_wall = time.perf_counter()
        self._check = False
        self._warm(self.warm_steps)
        self.fp_ok[:, :self.warm_steps] = True                         # spec_round's _warm fingerprints every warm row
        self.pos_host = self.rs.st.twins[0].pos.cpu().clone()           # a host sync (once, before the rounds)
        self.rs.st.pos_host.copy_(self.pos_host)
        self.h_count = [self.warm_steps] * B
        self.h_q = self.pos_host.tolist()
        self.h_u = [K] * B
        self.h_target = [False] * B
        self._ps_begin()
        self.round = -1
        self._set_check(0, False)                                      # the fill is checked when every round is
        ev = []
        self._event(ev)
        self._ps_prelude()
        self._event(ev)
        self.prelude_events = ev
        for b in range(B):                                             # the initial fill's trace (host)
            q = self.h_q[b]
            self._tr_add(b, [self._te("P%d" % j, "draft", fed_d=[q + j], drafted=q + j + 1) for j in range(K)])
            self.tok_tot["proposed"] += K
        cnt = torch.tensor(self.h_count)
        g = 0
        while g < limit and min(self.h_count) < N:
            self.round = g
            probe = pe > 0 and g > 0 and g % pe == 0
            self._set_check(g, probe)
            P = self._ps_plan()
            n, m, full = self._ps_round(P, probe)
            hc = torch.stack((self.count, self._ps_qpos(), self.u, n, m, full.to(torch.int64))).cpu()   # THE sync
            cnt = hc[0]
            self.pos_host = hc[1].clone()
            self.rs.st.pos_host.copy_(self.pos_host)
            self._ps_after(P, hc)
            if self.ops.after_round is not None:
                self.ops.after_round(self)
            g += 1
        self._set_check(-1, False)
        self.clock.sync()
        self.wall_s = time.perf_counter() - t_wall
        self.rounds_run = g
        return dict(rounds=g, count=cnt)

    def _ps_after_drafts(self, P, j, x, y, Dn, shift, t_new, lg1, plan_ctr):
        """One D slot's results (both runners): a group request's draft token into the next drafts (filtered); a target
        request's token committed (returned for the commit). Returns (x, t_new, m_d)."""
        B = self.B
        ys = self._filter("draft", j, y, next_col=self.count + shift + j)
        ys = torch.where(P.isS, ys, y)
        idx = torch.where(P.isS, shift + j, torch.full_like(shift, self.K)).view(B, 1)
        Dn.scatter_(1, idx, torch.where(P.isS.view(B, 1), ys.view(B, 1), Dn.gather(1, idx)))
        m_d = P.dO[j].to(torch.int64)
        t_new = torch.where(P.dO[j], y, t_new)
        x = torch.where(P.isS, ys, torch.where(P.dO[j], y, x))
        return x, t_new, m_d

    def _ps_stats(self, P, n, v, m, full, committed_round) -> None:
        sv = P.isS | P.isX
        one = sv.to(torch.int64)
        self.acc_hist.scatter_add_(0, n.clamp(max=self.K), one)
        self.valid_sum += (v * one).sum()
        self.accepted_sum += (n * one).sum()
        mis = (m - n).sum() if self.ops.neg == "miscount" else 0      # NEGATIVE CONTROL: accepted, not committed
        self.committed_sum += committed_round - mis
        self.full_sum += ((n >= v) & sv).sum()
        self.round_committed.append(committed_round)
        self.v_hist.scatter_add_(0, v.clamp(max=self.K), one)
        self.round_accepted.append((n * one).sum())

    def _paper_read(self) -> dict:
        tt = dict(self.tok_tot)
        tt["committed"] = tt["group_committed"] + tt["final_committed"] + tt["target_committed"]
        tt["discarded"] = tt["proposed"] - tt["accepted"]
        return dict(schedule="contract4", tokens_acct=tt,
                    trace=dict(head=[list(x) for x in self.trace_head], tail=[list(x) for x in self.trace_tail]),
                    trace_all=self.trace_all)


# ------------------------------------------------------------------ FAST: the tentative runner
class TentRunner(PaperSchedule, SR.SpecRunner):
    """The speculative rounds with the TENTATIVE BUFFERS (module docstring): L4P with the schedule of contract 4, L3 as
    a reference (its rounds as spec_round's L3: K drafts, K verify steps; it runs a fixed number of rounds). It attaches
    the KVMeter and (L4P) a VerifyState for the joint pass's buffers; no journal, no draft hook. hooks: on_select /
    on_logits (WS-G; check rounds only). check_every: 0 never, 1 every round, k > 1 the rounds g > 0 with g % k == 0 (the
    GPU cell: the probe rounds). poison, tneg: tests only."""

    def __init__(self, model, rs, ops, clock, t0=None, hooks=None, check_every: int = 1, poison: bool = False,
                 tneg=None):
        if ops.level not in LEVELS_T:
            raise SR.SpecError("tentative mode: level L3 or L4P (got %r)" % (ops.level,))
        if ops.prefetch is not None or ops.pin or ops.rb_neg is not None or ops.stream is not None:
            raise SR.SpecError("tentative mode: no prefetch, no pin, no rollback control, no commit executor")
        if ops.neg not in SR_NEG_T:
            raise SR.SpecError("tentative mode: negative control %r (one of %s)" % (ops.neg, SR_NEG_T))
        if tneg not in NEG_T:                                          # every refusal before the meter attaches
            raise SR.SpecError("unknown negative control %r (one of %s)" % (tneg, NEG_T))
        if tneg == "recompute_deferred" and ops.level != "L4P":
            raise SR.SpecError("recompute_deferred is an L4P control")
        K = int(ops.K)
        span = K + 2 if ops.level == "L4P" else K
        if span >= MAX_ROLLBACK:
            raise SR.SpecError("tentative mode: the chain (%d positions) must stay below MAX_ROLLBACK %d" % (
                span, MAX_ROLLBACK))
        if int(check_every) < 0:
            raise SR.SpecError("check_every >= 0")
        self.model, self.rs, self.ops, self.clock = model, rs, ops, clock
        B = int(rs.B)
        self.B, self.K = B, K
        dev = rs.st.cu_q.device
        self.dev = dev
        paper = ops.level == "L4P"
        RE.set_activity(rs, torch.ones(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool))
        self.meter = SR.KVMeter(rs, clock)
        self.rb = self.draft = self.pin = None
        self.vs = None
        attn = ops.attn
        if paper:
            self.VR = VR
            base = ops.linear if ops.linear is not None else rs.ops.linear
            R = K + 2 if tneg == "recompute_deferred" else K + 1
            self.vs = VR.attach(model, rs, VR.VerifyOps(K=R - 1, mode="exact", gemm=ops.gemm, linear=base,
                                                       extra=ops.extra, attn=attn, stats=False))
            attn = self.vs.ops.attn
        if attn is None:
            attn = rs.nl.flash_attn_nosa_with_kvcache
        self.T = TentState(rs, attn, tneg=tneg, poison=poison)
        self.hooks, self.check_every = hooks, int(check_every)
        self._check = False
        i64 = torch.int64
        self.t = torch.zeros(B, dtype=i64, device=dev) if t0 is None else t0.to(device=dev, dtype=i64).view(B).clone()
        self.D = torch.full((B, K + 1 if paper else K), -1, dtype=i64, device=dev)
        self.u = torch.full((B,), K, dtype=i64, device=dev)
        self.count = torch.zeros(B, dtype=i64, device=dev)
        self.arK = torch.arange(K, dtype=i64, device=dev)
        self.arR = torch.arange(K + 2, dtype=i64, device=dev)
        self.v_hist = torch.zeros(K + 1, dtype=i64, device=dev)
        self.round_accepted = []
        self.ids1 = torch.zeros((B, 1), dtype=i64, device=dev)
        self.pos1 = torch.zeros((B, 1), dtype=torch.int32, device=dev)
        self.acc_hist = torch.zeros(K + 1, dtype=i64, device=dev)
        self.valid_sum = torch.zeros((), dtype=i64, device=dev)
        self.accepted_sum = torch.zeros((), dtype=i64, device=dev)
        self.committed_sum = torch.zeros((), dtype=i64, device=dev)
        self.full_sum = torch.zeros((), dtype=i64, device=dev)
        self.ones_rb = torch.ones((1, B), dtype=torch.bool, device=dev)
        self.zeros_rb = torch.zeros((1, B), dtype=torch.bool, device=dev)
        self.zfm = torch.zeros((B, K + 1), dtype=torch.float32, device=dev)
        self.zfs = torch.zeros((B, K + 1), dtype=torch.float64, device=dev)
        self.round_committed, self.round_blocks, self.round_events, self.round_probe = [], [], [], []
        self.records = []
        self.cap = 0
        self.out = self.fmax = self.fsum = self.lg_out = self.fp_ok = None
        self.round = -1
        self.next_col = None
        self.prelude_ms = None
        self.prelude_events = None
        self.wall_s = None
        self.warm_steps = 0
        self.rounds_run = 0
        self.N = None
        self.trace_all = self.trace_head = self.trace_tail = self.tok_tot = None
        self.pos_host = rs.st.pos_host.clone()
        self.meta = dict(ops=ops.describe(), B=B, rollback="tentative", fast=True, tneg=tneg, poison=bool(poison),
                         check_every=int(check_every), schedule=("contract4" if paper else "L3"),
                         joint_rows=(self.vs.R if self.vs is not None else None),
                         verify=(self.vs.meta if self.vs is not None else None))

    # -------------------------------------------------------------- helpers
    def _sync_release(self) -> None:
        """No journal, no release event (spec_round's hook point)."""
        return None

    def detach(self) -> None:
        self.meter.detach()

    def _set_check(self, g: int, probe: bool) -> None:
        """g >= 0: a round (or the fill before round 0); g < 0: checks off (after the run)."""
        ce = self.check_every
        self._check = g >= 0 and (ce == 1 or (ce > 1 and g > 0 and g % ce == 0))
        self.T.checking = self._check
        self.T.hooks = self.hooks if self._check else None
        self.T.round = g if self.round >= 0 else -1

    def _remade(self, role, q, u, v):
        return []                                                      # FAST remakes nothing (D3)

    def _plan_mixed(self, x, act, draft, verify, kind: str, snap: bool):
        """A one-row pass of every request at its twin position: draft / target roles per request (masks (B,))."""
        B = self.B
        self.ids1.copy_(x.reshape(B, 1))
        self.pos1.copy_(self.rs.st.twins[0].pos.view(B, 1))
        return _Plan(R=1, kind=kind, ids=self.ids1, pos=self.pos1, act=act.view(1, B), verify=verify.view(1, B),
                     draft=draft.view(1, B), replay=None, replay_rows=(), deferred=None, vs=None,
                     lin=self.rs.ops.linear, snap=snap)

    def _plan1(self, x, kind: str):
        ver = kind == "verify"
        return self._plan_mixed(x, self.ones_rb[0], self.zeros_rb[0] if ver else self.ones_rb[0],
                                self.ones_rb[0] if ver else self.zeros_rb[0], kind, ver)

    def _fast_draft(self, x):
        """One FAST draft step of every request (draft residency, tentative rows); returns the argmax tokens."""
        lg = self.T.run_pass(self.model, self._plan1(x, "draft"))[:, 0]
        self.T.calls["draft_steps"] += 1
        return lg.argmax(dim=-1)

    # -------------------------------------------------------------- the schedule's parts (L4P)
    def _ps_begin(self) -> None:
        self.T.init_committed()

    def _ps_qpos(self):
        return self.T.qc

    def _ps_prelude(self) -> None:
        """The initial fill: K FAST draft steps from t at qc (u = K); nothing is undone."""
        x = self.t
        for j in range(self.K):
            d = self._filter("draft", j, self._fast_draft(x), next_col=self.count + j)
            self.D[:, j].copy_(d)
            x = self.D[:, j]

    def _joint_plan(self, q, u, P):
        """The J slot's rows (module docstring, THE SCHEDULE). group: rows 0 .. v - 1 verify at q + r, row v the draft
        row at q + v + 1 (the deferred position q + v replayed first); fill: row 0 a draft row at q + u (its twin is at
        its frontier); final: rows 0 .. v - 1 verify; target: row 0 at q; done: no row."""
        K, B = self.K, self.B
        R = self.vs.R
        ar = self.arR[:R]
        rr = ar.view(R, 1)
        vb = P.v.view(1, B)
        S, Fm, X, O = P.isS.view(1, B), P.isF.view(1, B), P.isX.view(1, B), P.isO.view(1, B)
        verify = ((S | X) & (rr < vb)) | (O & (rr == 0))
        if self.T.tneg == "recompute_deferred":                        # NEGATIVE CONTROL: EXACT's layout for a group
            draft = (S & (rr >= vb) & (rr <= u.view(1, B))) | (Fm & (rr == 0))
            colS = ar.view(1, R).expand(B, R)
            replay, rows = None, ()
        else:
            draft = (S & (rr == vb)) | (Fm & (rr == 0))
            colS = ar.view(1, R) + (ar.view(1, R) >= P.v.view(B, 1)).to(torch.int64)
            rows = tuple(sorted({P.v_h[b] for b in range(B) if P.role[b] == GROUP and 0 < P.v_h[b] < R}))   # host
            replay = S & (rr == vb)
        col = torch.where(P.isF.view(B, 1), u.view(B, 1).expand(B, R), colS)
        ids0 = torch.cat((self.t.view(B, 1), self.D.clamp(min=0)), dim=1)    # (B, K + 2): x_r
        ids = ids0.gather(1, col.clamp(max=K + 1))
        pos = (q.view(B, 1) + col).to(torch.int32)
        return _Plan(R=R, kind="joint", ids=ids, pos=pos, act=verify | draft, verify=verify, draft=draft,
                     replay=replay, replay_rows=rows, deferred=q + P.v, vs=self.vs,
                     lin=lambda x, w: VR._lin(self.vs, x, w), snap=True)

    def _ps_round(self, P, probe: bool):
        """One round of the schedule, FAST. Returns the J slot's (n, m, full)."""
        K, B = self.K, self.B
        T = self.T
        rec = self.ops.record
        ev = []
        self._event(ev)
        T.reset(P.reset)                                               # group / final / target: back to qc
        T.begin_round()
        self._event(ev)
        D, u, v = self.D, self.u, P.v
        q = T.qc.clone()
        b0 = self.meter.blocks.clone()
        s0 = len(self.meter.spans)
        self.meter.on, self.meter.timing = True, probe
        try:
            if P.rows > 1:                                             # the joint pass: K + 1 rows
                plan = self._joint_plan(q, u, P)
                lg = T.run_pass(self.model, plan)
                T.calls["joint_passes"] += 1
                T.calls["joint_rows"] += plan.R
                drow = torch.where(P.isS, v, torch.zeros_like(v))
                if T.tneg == "recompute_deferred":
                    drow = torch.where(P.isS, u, drow)
            else:                                                      # every row of this J slot is one row: fill /
                x0 = torch.where(P.isF, D.gather(1, (u - 1).clamp(min=0).view(B, 1)).view(B), self.t)   # target / final
                plan = self._plan_mixed(x0, P.isF | P.isO | P.isX, P.isF, P.isO | P.isX, "joint", True)
                lg = T.run_pass(self.model, plan)
                T.calls["j1_passes"] += 1
                drow = torch.zeros_like(v)
        finally:
            self.meter.on, self.meter.timing = False, False
        if self.hooks is not None and self._check and hasattr(self.hooks, "on_logits"):    # CHECK-ONLY (WS-G)
            self.hooks.on_logits(self.round, "joint", plan.pos.clone(), plan.verify, lg)
        self._event(ev)
        Rj = lg.shape[1]
        G = lg.argmax(dim=-1)                                          # (B, Rj)
        Gv = G[:, :K] if Rj >= K else torch.cat((G, G.new_full((B, K - Rj), -1)), dim=1)
        Dv = torch.where((self.arK.view(1, K) < v.view(B, 1)) & ~P.isO.view(B, 1), D[:, :K],
                         torch.full_like(D[:, :K], -1))
        n = SR.accept_prefix(Gv, Dv)                                   # greedy; a target row has no draft (n = 0)
        full = n >= v
        m = torch.where(full, v, n + 1)                                # fill / done: v = 0 -> m = 0; target: m = 1
        e_new = self._filter("draft_row", 0, G.gather(1, drow.view(B, 1)).view(B), next_col=self.count + u)
        Gc = self._plant_commit(Gv, Dv, n, m)
        lgK = lg[:, :K] if Rj >= K else torch.cat((lg, lg.new_zeros((B, K - Rj, lg.shape[2]))), dim=1)
        FM, FS = self._fingerprints(lgK)
        C = torch.stack(plan.ctr)                                      # (Rj, 3, B)
        ctr_m = torch.where((m > 0).view(1, B), C.gather(0, (m - 1).clamp(min=0).view(1, 1, B).expand(1, 3, B))
                            .view(3, B), T.ctr_c)
        ctr_now = T.counters0()
        frontier = self.rs.st.twins[0].pos.clone()
        dropped = torch.where(P.isS, torch.where(full, torch.zeros_like(m), u + 1 - m),
                              torch.where(P.isX, u - m, torch.zeros_like(m)))
        T.commit(m, ctr_m, frontier, dropped)                          # ACCEPT: rows [q, q + m) into the committed tail
        keep_now = (P.isS & full) | P.isF | P.isDONE
        T.set_twins(torch.where(keep_now.view(1, B), ctr_now, ctr_m))  # REJECT: the twin back to q + m
        if T.poison:                                                   # CHECK-ONLY (tests)
            T.poison_dead(self.rs.st.twins[0].pos - T.qc)
        self._event(ev)
        self._commit_tokens_t(Gc, m, FM, FS, lgK.clone() if self.lg_out is not None else None)
        committed_round = m.sum()
        t_new = torch.where(m > 0, Gc.gather(1, (m - 1).clamp(min=0).view(B, 1)).view(B), self.t)
        # the next drafts: a full accept [e_u, e_new, y ..], a reject the new fill [y ..], a fill e_{u+1} appended
        e_u = D.gather(1, (u - 1).clamp(min=0).view(B, 1)).view(B)
        Dn = torch.full((B, K + 1), -1, dtype=torch.int64, device=self.dev)
        sf = P.isS & full
        Dn[:, 0].copy_(torch.where(sf, e_u, Dn[:, 0]))
        Dn[:, 1].copy_(torch.where(sf, e_new, Dn[:, 1]))
        Dn = torch.where(P.isF.view(B, 1), D, Dn)
        Dn.scatter_(1, u.clamp(max=K).view(B, 1), torch.where(P.isF.view(B, 1), e_new.view(B, 1),
                                                                Dn.gather(1, u.clamp(max=K).view(B, 1))))
        shift = torch.where(sf, torch.full_like(u, 2), torch.zeros_like(u))
        x = torch.where(sf, e_new, t_new)                              # group: the next draft input; target: t
        x0 = x.clone() if rec else None
        for j in range(K - 1):                                         # the D slots
            if not P.any_d[j]:
                continue
            act = P.isS | P.dO[j]
            plan1 = self._plan_mixed(x, act, P.isS, P.dO[j], "draft", P.any_o[j])
            self.meter.on = P.any_o[j]                                 # a target row's loads count as verify loads
            try:
                lg1 = T.run_pass(self.model, plan1)[:, 0]
            finally:
                self.meter.on = False
            T.calls["mixed_steps" if P.any_o[j] else "draft_steps"] += 1
            y = lg1.argmax(dim=-1)
            x, t_new, m_d = self._ps_after_drafts(P, j, x, y, Dn, shift, t_new, lg1, plan1.ctr)
            if P.any_o[j]:                                             # the drain: commit the target tokens
                ctr_d = torch.where(P.dO[j].view(1, B), plan1.ctr[0], T.ctr_c)
                T.commit(m_d, ctr_d, self.rs.st.twins[0].pos.clone(), torch.zeros_like(m_d))
                if T.poison:                                           # CHECK-ONLY (tests)
                    T.poison_dead(self.rs.st.twins[0].pos - T.qc)
                fm, fs = self._fingerprints(lg1.view(B, 1, -1))
                self._commit_tokens_t(y.view(B, 1), m_d, fm, fs, lg1.view(B, 1, -1).clone()
                                      if self.lg_out is not None else None)
                committed_round = committed_round + m_d.sum()
        self._event(ev)
        self.t = t_new
        self.D = torch.where((P.isX | P.isO | P.isDONE).view(B, 1), torch.full_like(Dn, -1), Dn)
        self.u = torch.where(sf, torch.full_like(u, K + 1),
                             torch.where(P.isS, torch.full_like(u, K - 1),
                                         torch.where(P.isF, u + 1, torch.zeros_like(u))))
        self._ps_stats(P, n, v, m, full, committed_round)
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if rec:
            self.records.append(dict(D=D.clone(), u=u.clone(), v=v.clone(), Dv=Dv.clone(), G=Gv.clone(),
                                     n=n.clone(), m=m.clone(), full=full.clone(), committed=Gc.clone(),
                                     e_new=e_new.clone(), pos=self.pos_host.clone(), q=q, x0=x0, Dn=self.D.clone(),
                                     un=self.u.clone(), frontier=frontier, role=torch.tensor(P.role),
                                     q_end=T.qc.clone(), rows=P.rows))
        return n, m, full

    # -------------------------------------------------------------- L3 (reference): spec_round's round, FAST
    def _round_l3_t(self, probe: bool) -> None:
        K, B = self.K, self.B
        T = self.T
        ev = []
        self._event(ev)
        T.reset()
        T.begin_round()
        self._event(ev)
        x = self.t
        for j in range(K):                                             # K FAST drafts (kept residency)
            d = self._filter("draft", j, self._fast_draft(x), next_col=self.count + j)
            self.D[:, j].copy_(d)
            x = self.D[:, j]
        self._event(ev)
        T.reset()                                                      # the drafts' summaries back to qc
        self._event(ev)
        D = self.D
        q = T.qc.clone()
        X = torch.cat((self.t.view(B, 1), D[:, :K - 1]), dim=1)
        G = torch.empty((B, K), dtype=torch.int64, device=self.dev)
        LG = [] if self.lg_out is not None else None
        lgs, ctrs = [], []
        b0 = self.meter.blocks.clone()
        s0 = len(self.meter.spans)
        self.meter.on, self.meter.timing = True, probe
        try:
            for j in range(K):                                         # K FAST verify steps (target residency)
                plan = self._plan1(X[:, j].contiguous(), "verify")
                lg = T.run_pass(self.model, plan)
                T.calls["verify_steps"] += 1
                if self.hooks is not None and self._check and hasattr(self.hooks, "on_logits"):    # CHECK-ONLY
                    self.hooks.on_logits(self.round, "verify", plan.pos.clone(), plan.verify, lg)
                lg = lg[:, 0]
                G[:, j].copy_(lg.argmax(dim=-1))
                if self._check:
                    lgs.append(lg)
                if LG is not None:
                    LG.append(lg.clone())
                ctrs.append(plan.ctr[0])
        finally:
            self.meter.on, self.meter.timing = False, False
        self._event(ev)
        n, v, m, full = self._accept(G, D)
        Gc = self._plant_commit(G, D, n, m)
        FM, FS = self._fingerprints(torch.stack(lgs, 1)) if lgs else (self.zfm[:, :K], self.zfs[:, :K])
        ctr_m = torch.stack(ctrs).gather(0, (m - 1).view(1, 1, B).expand(1, 3, B)).view(3, B)
        frontier = self.rs.st.twins[0].pos.clone()
        T.commit(m, ctr_m, frontier, frontier - q - m)
        if T.poison:                                                   # CHECK-ONLY (tests)
            T.poison_dead(torch.zeros_like(m))
        self._event(ev)
        self._commit_tokens_t(Gc, m, FM, FS, torch.stack(LG, 1) if LG is not None else None)
        self.t = Gc.gather(1, (m - 1).view(B, 1)).view(B)
        self._stats(n, v, m, full)
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if self.ops.record:
            self.records.append(dict(D=D.clone(), G=G.clone(), n=n.clone(), v=v.clone(), m=m.clone(),
                                     full=full.clone(), committed=Gc.clone(), pos=self.pos_host.clone(), q=q,
                                     frontier=frontier, q_end=T.qc.clone()))

    @torch.inference_mode()
    def _run_l3(self, N: int, rounds: int = 0, max_rounds: int = 0, warm: int = 1) -> dict:
        """spec_round.SpecRunner.run with the FAST L3 rounds (a fixed number of rounds or until every request has N)."""
        B, K = self.B, self.K
        N = int(N)
        if int(warm) < 1:
            raise SR.SpecError("warm >= 1: the first step after the prefill loads the window (spec_round._warm)")
        self.warm_steps = int(warm)
        self.cap = N + K
        dev = self.dev
        self.out = torch.full((B, self.cap + 1), -1, dtype=torch.int64, device=dev)
        self.fmax = torch.zeros((B, self.cap + 1), dtype=torch.float32, device=dev)
        self.fsum = torch.zeros((B, self.cap + 1), dtype=torch.float64, device=dev)
        self.fp_ok = torch.zeros((B, self.cap + 1), dtype=torch.bool, device=dev)
        if self.ops.record:
            V = int(self.model.lm_head.shape[0])
            self.lg_out = torch.zeros((B, self.cap + 1, V), dtype=torch.float32, device=dev)
        limit = int(rounds) if rounds > 0 else (int(max_rounds) if max_rounds > 0 else N + 8)
        pe = self.ops.probe_every
        self.clock.sync()
        t_wall = time.perf_counter()
        self._check = False
        self._warm(self.warm_steps)
        self.fp_ok[:, :self.warm_steps] = True
        self.pos_host = self.rs.st.twins[0].pos.cpu().clone()           # a host sync (once, before the rounds)
        self.rs.st.pos_host.copy_(self.pos_host)
        self.T.init_committed()
        cnt = torch.zeros(B, dtype=torch.int64)
        g = 0
        while g < limit:
            self.round = g
            probe = pe > 0 and g > 0 and g % pe == 0
            self._set_check(g, probe)
            self._round_l3_t(probe)
            hc = torch.stack((self.count, self.T.qc)).cpu()            # THE host sync of the round
            cnt = hc[0]
            self.pos_host = hc[1].clone()
            self.rs.st.pos_host.copy_(self.pos_host)
            if self.ops.after_round is not None:
                self.ops.after_round(self)
            g += 1
            if rounds <= 0 and int(cnt.min()) >= N:
                break
        self._set_check(-1, False)
        self.clock.sync()
        self.wall_s = time.perf_counter() - t_wall
        self.rounds_run = g
        return dict(rounds=g, count=cnt)

    def run(self, N: int, rounds: int = 0, max_rounds: int = 0, warm: int = 1) -> dict:
        """L4P: the schedule of contract 4 until every request has exactly N tokens (rounds is ignored); L3: the
        reference rounds."""
        if self.ops.level == "L4P":
            return self._run_paper(N, max_rounds=max_rounds, warm=warm)
        return self._run_l3(N, rounds=rounds, max_rounds=max_rounds, warm=warm)

    @torch.inference_mode()
    def read(self) -> dict:
        """ONE host read after run(): spec_round's keys (journal None: no journal), the tentative report and (L4P) the
        schedule's trace and token accounting."""
        el = self.clock.elapsed
        T = self.T
        tr = T.report()
        out = dict(rounds=self.rounds_run, warm=self.warm_steps, wall_s=self.wall_s, count=self.count.cpu().tolist(),
                   acc_hist=self.acc_hist.cpu().tolist(), valid=int(self.valid_sum), accepted=int(self.accepted_sum),
                   committed=int(self.committed_sum), full=int(self.full_sum),
                   round_committed=[int(x) for x in torch.stack(self.round_committed).cpu()] if self.round_committed
                   else [], round_blocks=[int(x) for x in torch.stack(self.round_blocks).cpu()] if self.round_blocks
                   else [],
                   phases=[[el(ev[i], ev[i + 1]) for i in range(len(ev) - 1)] for ev in self.round_events],
                   round_ms=[el(ev[0], ev[-1]) for ev in self.round_events],
                   probe_wait_ms=[(sum(el(a, b) for a, b in sp) if sp is not None else None) for sp in self.round_probe],
                   draft_skipped=tr["skipped"], draft_redirected=[0] * T.L, draft_calls=tr["draft_rows"],
                   draft_steps=T.calls["draft_steps"], journal=None, truncations=0, groups=0,
                   errors=RE.errors(self.rs),
                   prelude_ms=(el(self.prelude_events[0], self.prelude_events[1]) if self.prelude_events else None),
                   verify=None, tent=tr, rollback="tentative",
                   fp_ok=[int(x) for x in self.fp_ok[:, :self.cap].sum(1).cpu()] if self.fp_ok is not None else None)
        if self.ops.level == "L4P":
            out.update(verify_hist=self.v_hist.cpu().tolist(),
                       round_accepted=[int(x) for x in torch.stack(self.round_accepted).cpu()] if self.round_accepted
                       else [], row_skipped=None, rows_calls=T.calls["joint_passes"] * T.L,
                       rows_draft=tr["draft_rows"], pin=None)
            out.update(self._paper_read())
        return out


# ------------------------------------------------------------------ EXACT: the journal runner of the same schedule
class PaperJournalRunner(PaperSchedule, SR.SpecRunner):
    """The schedule of contract 4 with spec_round's JOURNAL (EXACT; the proof of the schedule): the Rollback, the
    DraftResident (mode "rows": per request draft or target rows in one call), the EXACT joint pass of K + 2 rows
    (gemm rowgroup). Every tentative step is undone at the round's commit except the committed rows, so a group remakes
    its deferred row and a fill remakes its fill positions (the trace's `remade`). Idle requests in a D slot are
    inactive (the host plan gives the activity, no extra sync). Committed tokens and logits rows: bitwise SH's."""

    _fp_check_only = False

    def __init__(self, model, rs, ops, clock, t0=None, hooks=None, check_every: int = 0):
        if ops.level != "L4P" or ops.mode != "exact":
            raise SR.SpecError("PaperJournalRunner: level L4P, mode exact")
        if ops.prefetch is not None or ops.pin:
            raise SR.SpecError("PaperJournalRunner: no prefetch, no pin")
        if ops.neg not in SR_NEG_T:
            raise SR.SpecError("PaperJournalRunner: negative control %r (one of %s)" % (ops.neg, SR_NEG_T))
        if int(check_every) < 0:
            raise SR.SpecError("check_every >= 0")
        SR.SpecRunner.__init__(self, model, rs, ops, clock, t0=t0)
        B, K = self.B, self.K
        self.hooks, self.check_every = hooks, int(check_every)
        self._check = False
        self.fp_ok = None
        self.N = None
        self.trace_all = self.trace_head = self.trace_tail = self.tok_tot = None
        self.zfm = torch.zeros((B, K + 1), dtype=torch.float32, device=self.dev)
        self.zfs = torch.zeros((B, K + 1), dtype=torch.float64, device=self.dev)
        self.ones_d = torch.ones(B, dtype=torch.bool, device=self.dev)
        self.ones_h = torch.ones(B, dtype=torch.bool)
        self.calls = dict(joint_passes=0, draft_steps=0, mixed_steps=0)
        self.meta.update(schedule="contract4", rollback="journal", fast=False)

    def _set_check(self, g: int, probe: bool) -> None:
        ce = self.check_every
        self._check = g >= 0 and (ce == 1 or (ce > 1 and g > 0 and g % ce == 0))

    def _remade(self, role, q, u, v):
        if role == GROUP:
            return [q + v]                                             # the deferred row (row v) remakes position q + v
        if role == FILL:
            return list(range(q, q + u))                               # the fill's earlier positions, as draft rows
        return []

    def _ps_begin(self) -> None:
        return None

    def _ps_qpos(self):
        self._sync_release()
        return self.rs.st.twins[0].pos

    def _ps_prelude(self) -> None:
        self._prelude()                                                # spec_round: K drafts in a group, reject_all

    def _ps_round(self, P, probe: bool):
        """One round of the schedule with the journal. Returns the J slot's (n, m, full)."""
        K, B = self.K, self.B
        R = K + 2
        rec = self.ops.record
        ev = []
        self._event(ev)
        D, u, v = self.D, self.u, P.v
        ids = torch.cat((self.t.view(B, 1), D.clamp(min=0)), dim=1)    # (B, R): row r feeds x_r at q + r
        self._sync_release()
        q = self.rs.st.twins[0].pos.clone()
        pos = (q.view(B, 1) + self.arR.view(1, R)).to(torch.int32)
        rr = self.arR.view(R, 1)
        SX, O = (P.isS | P.isX).view(1, B), P.isO.view(1, B)
        drows = (SX & (rr >= v.view(1, B))) | (O & (rr >= 1)) | (P.isF | P.isDONE).view(1, B)   # resident-only rows
        drow = (rr == u.view(1, B)) & (P.isS | P.isF).view(1, B)
        # the rows below the first draft row of every request go straight to the body (spec_round's verify_below)
        vbelow = min([P.v_h[b] if P.role[b] in (GROUP, FINAL) else 1 if P.role[b] == TARGET else 0
                      for b in range(B)] + [R])
        self.rb.begin()                                                # 1. the joint pass
        b0 = self.meter.blocks.clone()
        s0 = len(self.meter.spans)
        self.draft.begin("rows", redirect=self._edge_before(R), rows=drows, record=drow, verify_below=vbelow)
        self.meter.on, self.meter.timing = True, probe
        try:
            lg = self.VR.joint_pass(self.model, self.vs, ids, pos)
        finally:
            self.meter.on, self.meter.timing = False, False
            self.draft.end()
        self.calls["joint_passes"] += 1
        if self.hooks is not None and self._check and hasattr(self.hooks, "on_logits"):    # CHECK-ONLY
            self.hooks.on_logits(self.round, "joint", pos.clone(), ~drows, lg)
        self._event(ev)
        G = lg.argmax(dim=-1)
        Gv = G[:, :K]
        Dv = torch.where((self.arK.view(1, K) < v.view(B, 1)) & ~P.isO.view(B, 1), D[:, :K],
                         torch.full_like(D[:, :K], -1))
        n = SR.accept_prefix(Gv, Dv)                                   # 2. accept
        full = n >= v
        m = torch.where(full, v, n + 1)
        e_new = self._filter("draft_row", 0, G.gather(1, u.clamp(max=K + 1).view(B, 1)).view(B),
                             next_col=self.count + u)
        Gc = self._plant_commit(Gv, Dv, n, m)
        FM, FS = self._fingerprints(lg[:, :K])
        sf = P.isS & full
        keep_t = torch.where(P.isS, torch.where(full, u + 1, m),
                             torch.where(P.isX, m, torch.where(P.isO, torch.ones_like(m), torch.zeros_like(m))))
        self.rb.truncate(keep_t)                                       # 3. truncate
        self._event(ev)
        self._commit_tokens_t(Gc, m, FM, FS, lg[:, :K].clone() if self.lg_out is not None else None)
        committed_round = m.sum()
        t_new = torch.where(m > 0, Gc.gather(1, (m - 1).clamp(min=0).view(B, 1)).view(B), self.t)
        e_u = D.gather(1, (u - 1).clamp(min=0).view(B, 1)).view(B)
        Dn = torch.full((B, K + 1), -1, dtype=torch.int64, device=self.dev)
        Dn[:, 0].copy_(torch.where(sf, e_u, Dn[:, 0]))
        Dn[:, 1].copy_(torch.where(sf, e_new, Dn[:, 1]))
        Dn = torch.where(P.isF.view(B, 1), D, Dn)
        Dn.scatter_(1, u.clamp(max=K).view(B, 1), torch.where(P.isF.view(B, 1), e_new.view(B, 1),
                                                                Dn.gather(1, u.clamp(max=K).view(B, 1))))
        shift = torch.where(sf, torch.full_like(u, 2), torch.zeros_like(u))
        x = torch.where(sf, e_new, t_new)
        x0 = x.clone() if rec else None
        dpos = [None] * (K - 1) if rec else None
        partial = False
        for j in range(K - 1):                                         # 4. the D slots
            if not P.any_d[j]:
                continue
            act_h = torch.tensor([P.h_dS[b] or P.h_dO[j][b] for b in range(B)], dtype=torch.bool)
            RE.set_activity(self.rs, P.isS | P.dO[j], act_h)
            partial = True
            if rec:                                                    # CHECK-ONLY (tests): the slot's fed positions
                dpos[j] = self.rs.st.twins[0].pos.clone()
            self.draft.begin("rows", redirect=self._edge_before(R + j), rows=P.isS.view(1, B),
                             record=P.isS.view(1, B), verify_below=0)
            self.meter.on = P.any_o[j]
            try:
                lg1 = self._step(x)
            finally:
                self.draft.end()
                self.meter.on = False
            self.calls["mixed_steps" if P.any_o[j] else "draft_steps"] += 1
            y = lg1.argmax(dim=-1)
            x, t_new, m_d = self._ps_after_drafts(P, j, x, y, Dn, shift, t_new, lg1, None)
            if P.any_o[j]:
                fm, fs = self._fingerprints(lg1.view(B, 1, -1))
                self._commit_tokens_t(y.view(B, 1), m_d, fm, fs, lg1.view(B, 1, -1).clone()
                                      if self.lg_out is not None else None)
                committed_round = committed_round + m_d.sum()
        if partial:
            RE.set_activity(self.rs, self.ones_d, self.ones_h)
        self._event(ev)
        keep_c = torch.where(P.isS | P.isX, m, torch.where(P.isO, torch.full_like(m, R + K - 1), torch.zeros_like(m)))
        self.rb.commit(keep_c)                                         # 5. commit: the committed rows only
        self._event(ev)
        self.t = t_new
        self.D = torch.where((P.isX | P.isO | P.isDONE).view(B, 1), torch.full_like(Dn, -1), Dn)
        self.u = torch.where(sf, torch.full_like(u, K + 1),
                             torch.where(P.isS, torch.full_like(u, K - 1),
                                         torch.where(P.isF, u + 1, torch.zeros_like(u))))
        self._ps_stats(P, n, v, m, full, committed_round)
        self.round_blocks.append(self.meter.blocks - b0)
        self.round_events.append(ev)
        self.round_probe.append(self.meter.spans[s0:] if probe else None)
        if rec:
            self.records.append(dict(D=D.clone(), u=u.clone(), v=v.clone(), Dv=Dv.clone(), G=Gv.clone(),
                                     Gall=G.clone(), n=n.clone(), m=m.clone(), full=full.clone(), committed=Gc.clone(),
                                     e_new=e_new.clone(), pos=self.pos_host.clone(), q=q, x0=x0, Dn=self.D.clone(),
                                     un=self.u.clone(), drows=drows.clone(), role=torch.tensor(P.role), dpos=dpos))
        return n, m, full

    def run(self, N: int, rounds: int = 0, max_rounds: int = 0, warm: int = 1) -> dict:
        """The schedule of contract 4 until every request has exactly N tokens (rounds is ignored)."""
        return self._run_paper(N, max_rounds=max_rounds, warm=warm)

    @torch.inference_mode()
    def read(self) -> dict:
        out = SR.SpecRunner.read(self)
        out.update(self._paper_read(), rollback="journal", calls=dict(self.calls),
                   fp_ok=[int(x) for x in self.fp_ok[:, :self.cap].sum(1).cpu()] if self.fp_ok is not None else None)
        return out
