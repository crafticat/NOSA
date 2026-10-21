"""R2 step 7, piece 5, (B2): the FUSED RESIDENCY RESOLVE (docs/R2_FUSEDRES_PLAN.md). SOURCE ONLY, NOT RUN.

WHAT IT REPLACES. OwnResidency (r2_own2.py) inherits S-C's per-layer device path: DeviceMirror.layer (two segmented
sorts and scatters), device_verify / device_draft (stable sorts, rank_in_group with cummax, index, index_put_) and
C's sparse record (sparse_layer_record, _compact, _pack_bits, draft_bias_fill). On the GPU that is about 305 small
kernels per layer, 9,774 per draft tick at B 64 and 37.62 ms of kernel time (analysis/bound_4_2/other_res_r1,
"res:resolve G_d"). This module makes the SAME decisions with two Triton launches per row kind present in the layer:
FOUR with verify and draft rows (V1, V2, D1, D2), TWO with only draft rows (D1, D2) or only verify rows (V1, V2). No host read, no host-to-device copy in
the per-layer path. OwnResidency(resolve="fused") uses it; resolve="check" uses it and also runs the torch path on the
same inputs and compares every output (tests, the GPU equality gate); resolve="torch" (the default) is unchanged.

THE LAYOUT. One Triton program per stream s = b * H + h (grid S = B * H). A program reads its stream's NPOS = C + X
slots from the device mirror (dmirror.code / block / gen / cur / prot / stamp [l], the table B2 copies at each tick's
first resolve) into registers, and loops over the rows of its batch slot b in entry order (the per-tick CSR: for each
b the verify rows and the draft rows, ascending). A block -> slot lookup is a compare of the row's k selected blocks
against the stream's NPOS slot blocks (a (k, NPOS) register tile; K3: one resident slot per block). So the per-block
tables of DeviceMirror.layer (B, H, nblk) are never built.

THE ONE GLOBAL COUPLING is K5's shared extras: "the first (X - borrowed - reserve) extra-needing first misses of the
layer, in K3X entry order (row, head, ascending block)", and borrowed sums every stream's usage. Phase 1 (V1 / D1)
writes each (row, head)'s count of extra-needing first misses and each stream's usage; phase 2 (V2 / D2) reads the
whole count vector (R * H integers) and the usage vector (S integers) and computes its entries' global exclusive
prefix itself: base(row, head) + the in-row prefix. The sparse record's compaction uses the same scheme (the record
count per (row, head) comes from phase 1: a draft record is every valid entry, a verify record every valid entry that
is not READY, since pin_records is False for B2). Program 0 of phase 1 zeroes the counters that phase 2 adds into.

PER LAYER, IN S-C's ORDER:
  V1  per stream: the verify rows' classification (sorted unique selection, READY / IN FLIGHT / miss, the first miss
      per (stream, block) across the rows, K5's local rank); writes the extra-need count and the record count per
      (row, head), the stream's usage.
  V2  per stream: the same classification; K5 grants (global prefix), the q-th FREE slot, Policy E (the stream's
      candidates sorted by (class, stamp, slot) with tl.sort: the LRU candidates by recency, then the owner's
      prefetches in flight by issue stamp; neither protected nor selected by any row of the layer), the read slots,
      the page tables (plan, int32; the decision's, int64), the decision fields, the urgent load ids, the verify
      records, looks and read bitmap; and the stream's state after the verify rows (new blocks in flight, evicted
      blocks, FREE slots, usage) into a scratch buffer for the draft rows.
  (S-C's device wait and urgent gather run here, on the verify decision, as in ResidencySC.resolve.)
  D1  per stream: the draft rows' classification on the state after the verify rows.
  D2  per stream: K5 grants with the R_ver reserve, the take slots, the page tables, the decision fields, the draft
      records, the bias rows of the granted takes (S-B's draft_bias_fill), the counters; the selection bitmaps (sel,
      rowsel) of every row.

EXACTNESS. Every output equals the torch path's, field by field (test_r2_fusedres.py FR1, resolve="check"):
VerifyDecision.t (FIELDS_V), page, num_pages, load_ids, counters; DraftDecision.t (FIELDS), page, num_pages, counters;
the sparse record (vrec, vcnt, vlook, vread, drec, dcnt, sel, rowsel); the plan's page table and page counts; the
bias rows. The torch path's tie-breaks are reproduced: sorted selections (values only), "first occurrence in entry
order", the q-th FREE slot by slot index, and Policy E's order by key. ASSUMPTIONS (each one an invariant of the
ownership state; the err flag below is raised when the first two fail):
  A1  a block has at most one resident slot and at most one current-owner slot in flight per stream (K3). S-C's
      scatter picks one of several on the GPU (unspecified); this module picks the highest slot.
  A2  Policy E's candidate stamps are in [0, 2^47) and distinct within a stream (each stamp is a fresh clock value).
      torch.sort breaks ties in an unspecified way; this module breaks them by slot index.
  A3  a K5 grant always finds a FREE slot (grants <= NPOS - usage, by K5's arithmetic). S-C would index the next
      stream's slot 0 if it did not; this module writes nothing for such a take and raises the err flag.

THE BIAS ROWS are copied bit for bit in D2 when store_b[l] and cis[l] are contiguous (B, rows, H) tables of one 2- or
4-byte dtype; otherwise S-B's torch draft_bias_fill runs after D2 (counted in bias_fallbacks: torch ops, not the
design's launch count).

THE ERR FLAG (self.err, int32, watched by the service: a set flag voids the tick with CapacityRefusal at end_tick):
bit 1 A1 resident, bit 2 A1 in flight, bit 4 A3, bit 8 A2 (stamp range).

NEGATIVE CONTROLS (plant, tests only): 1 = Policy E takes the NEWEST candidate first (the order flipped); 2 = a draft
grant takes the (q + 1)-th FREE slot (an off-by-one slot). STATS counts, per run, where each plant CHANGES an output
(L34): plant 1 changes ev_slot only for an eviction whose flipped rank differs (stats "pe_flip_effective"); plant 2
changes every draft grant's take slot ("d_grant").

Imports torch (and triton when it is installed). No `from __future__ import annotations`: the kernels' constexpr
annotations must be objects.
"""
import torch

try:                                                   # triton is needed only to launch the kernels; the reference
    import triton                                      # path and the checks work without it
    import triton.language as tl
except Exception:                                      # pragma: no cover - not installed, or a broken install
    triton = tl = None

from .kstep.errors import OwnershipViolation
from .r2_residency import N_TAIL
from .r2_residency_sb import FIELDS, DraftDecision, device_draft
from .r2_residency_sc import (C_FREE, C_INFLIGHT, C_LRU, C_PO, C_RELEASING, C_VRES, FIELDS_V, HALF_STAMP,
                              ResidencySC, VerifyDecision, device_verify)
from .r2_residency_sparse import (DREC, DUP, EVICT, EVICT_INFL, FIRST, GRANT, HOLE, INFL, PIN, READY, REFUSED, VREC,
                                  ResidencySparse, draft_bias_fill, sparse_layer_record)

# the kernels use these values as literals (Triton's compiler sees only constexpr globals): pin them here
assert (C_FREE, C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING) == (0, 1, 2, 3, 4, 5)
assert (PIN, READY, INFL, DUP, GRANT, EVICT, EVICT_INFL, FIRST, HOLE, REFUSED) == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
assert HALF_STAMP == 2305843009213693952                     # 2^61
assert FIELDS_V == ("blk", "valid", "ready", "slot", "infl", "infl_slot", "infl_gen", "dup", "granted", "take_slot",
                    "rslot", "protect", "evict", "evict_blk", "evict_infl", "ev_gen")
assert FIELDS == ("blk", "valid", "ready", "slot", "infl", "granted", "refused", "take_slot", "protect")
assert VREC == ("o", "blk", "flags", "slot", "infl_slot", "infl_gen", "rslot", "take_slot", "evict_blk")
assert DREC == ("o", "blk", "flags", "slot", "take_slot")

RESOLVE_MODES = ("torch", "fused", "check")
PLANTS = {0: "none", 1: "Policy E takes the newest candidate first", 2: "a draft grant takes the (q + 1)-th FREE slot"}
STATS = ("v_evict", "pe_flip_effective", "v_grant", "d_grant", "d_refused", "v_over", "v_dup", "v_evict_infl")
ERR_BITS = {1: "A1: a block has two resident slots in a stream", 2: "A1: a block has two owner slots in flight",
            4: "A3: a K5 grant found no FREE slot", 8: "A2: a Policy E candidate stamp outside [0, 2^47)"}
MAX_NPOS = 512                                         # Policy E's sort key holds the slot index in 9 bits


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


if triton is not None:
    # =================================================================================================================
    # helpers (module level: Triton's GPU compiler resolves a kernel's names in its module globals, test F23 / L36)
    # =================================================================================================================
    @triton.jit
    def _row(order_ptr, ix_ptr, hb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rx, h, R, K, NBLK, KP: tl.constexpr):
        """Row i of the stream's list (entry order): active flag, entry row j, tick row tr, history bound, key id,
        the selection s (KP,) int32 (-1 padded) and its valid historical entries v0. The index is clamped, so an
        inactive row reads an in-bounds row (its results are masked)."""
        act = i < cnt
        pos = tl.where(act, start + i, start + cnt - 1)
        pos = tl.where(pos > Rx - 1, Rx - 1, pos)
        pos = tl.where(pos < 0, 0, pos)
        j = tl.load(order_ptr + pos)
        tr = tl.load(ix_ptr + j)
        hbr = tl.load(hb_ptr + j)
        kid = tl.load(kid_ptr + tr)
        kk = tl.arange(0, KP)
        mk = kk < K
        s = tl.load(sel_ptr + (h * R + tr) * K + kk, mask=mk, other=-1).to(tl.int32)
        v0 = mk & (s >= 0) & (s < hbr) & (s < NBLK)
        return act, j, tr, kid, s, v0

    @triton.jit
    def _sorted_unique(s, v0, K, NBLK, KP: tl.constexpr):
        """device_verify / device_draft's srt / valid / blk: the row's entries sorted ascending (invalid ones as
        NBLK + 1, padded lanes after them), valid = first of its value and < NBLK + 1, blk = valid ? srt : 0."""
        kk = tl.arange(0, KP)
        mk = kk < K
        key = tl.where(v0, s, tl.where(mk, NBLK + 1, NBLK + 2))
        cnt_lt = tl.sum((key[None, :] < key[:, None]).to(tl.int32), axis=1)          # [i, i']: key_i' < key_i
        srt = tl.max(tl.where(cnt_lt[None, :] <= kk[:, None], key[None, :], -1), axis=1)
        prv = tl.max(tl.where(cnt_lt[None, :] < kk[:, None], key[None, :], -1), axis=1)
        valid = mk & (srt < NBLK + 1) & (srt != prv)
        blk = tl.where(valid, srt, 0)
        return valid, blk

    @triton.jit
    def _lookup(blk, sblk, ok, jj):
        """Per entry: the slot of the stream whose block is blk among the slots `ok` (-1: none), and the count."""
        m = ok[None, :] & (sblk[None, :] == blk[:, None])
        slot = tl.max(tl.where(m, jj[None, :], -1), axis=1)
        n = tl.sum(m.to(tl.int32), axis=1)
        return slot, n

    @triton.jit
    def _at_slot(slot, jj, val, dflt):
        """Per entry: val[slot] (dflt where slot names no lane; dflt is below every value of val)."""
        m = jj[None, :] == slot[:, None]
        return tl.max(tl.where(m, val[None, :], dflt), axis=1)

    @triton.jit
    def _seen(order_ptr, ix_ptr, hb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rx, h, R, K, NBLK, kid, blk, valid,
              KP: tl.constexpr):
        """Per entry of row i: its block is a valid entry of an EARLIER row of the stream's list (seen), and of an
        earlier row with the same key id (seen_k). Rows are few per stream (one per draft-only tick)."""
        seen = valid & (blk < 0)
        seen_k = seen
        for i2 in range(0, i):
            a2, j2, tr2, kid2, s2, v2 = _row(order_ptr, ix_ptr, hb_ptr, kid_ptr, sel_ptr, start, cnt, i2, Rx, h, R,
                                             K, NBLK, KP)
            hit = tl.max((v2[None, :] & (s2[None, :] == blk[:, None])).to(tl.int32), axis=1) > 0
            seen = seen | hit
            seen_k = seen_k | (hit & (kid2 == kid))
        return seen, seen_k

    @triton.jit
    def _selbits(s, v0, ZP: tl.constexpr):
        z = tl.arange(0, ZP)
        return tl.max((v0[None, :] & (s[None, :] == z[:, None])).to(tl.int32), axis=1) > 0

    @triton.jit
    def _pack(bits, ZP: tl.constexpr):
        """(ZP,) bool -> (ZP // 32,) int64 words, bit i of word w = element 32 w + i (_pack_bits' format)."""
        z = tl.arange(0, ZP)
        w = bits.to(tl.int64) << (z % 32).to(tl.int64)
        return tl.sum(tl.reshape(w, [ZP // 32, 32]), axis=1)

    @triton.jit
    def _put1(ptr, val, m):
        """One value at ptr where the scalar m holds (a 2-lane vector store: the interpreter ignores a scalar mask)."""
        ln = tl.arange(0, 2)
        tl.store(ptr + ln * 0, tl.zeros([2], val.dtype) + val, mask=(ln == 0) & m)

    @triton.jit
    def _slots(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP: tl.constexpr):
        jj = tl.arange(0, NP)
        mj = jj < NPOS
        code = tl.load(code_ptr + base + jj, mask=mj, other=-1).to(tl.int32)
        sblk = tl.load(blk_ptr + base + jj, mask=mj, other=-1).to(tl.int32)
        gen = tl.load(gen_ptr + base + jj, mask=mj, other=0)
        cur = tl.load(cur_ptr + base + jj, mask=mj, other=0)
        prot = tl.load(prot_ptr + base + jj, mask=mj, other=0)
        stamp = tl.load(stamp_ptr + base + jj, mask=mj, other=0)
        okb = (sblk >= 0) & (sblk < NBLK)
        resid = (code == 2) | (code == 3) | (code == 4)                              # LRU / PO / VRES
        res = okb & resid
        fly = okb & (code == 1) & (cur != 0)                                       # the current owner's, in flight
        usage = tl.sum((mj & (code != 0)).to(tl.int32), axis=0)
        return jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage

    @triton.jit
    def _d_state(blk, sblk, res, fly, jj, newblk, evold, evinfl, HAS_V: tl.constexpr):
        """The draft rows' mirror lookups after the verify rows (device_verify's mirror update): a slot of a block
        that Policy E evicted is gone; a block the verify rows took is in flight; a superseded prefetch's block is
        not in flight."""
        slot, nres = _lookup(blk, sblk, res, jj)
        islot, nfly = _lookup(blk, sblk, fly, jj)
        infl = islot >= 0
        if HAS_V:
            ge = evold[None, :] == blk[:, None]
            gone = tl.max(ge.to(tl.int32), axis=1) > 0
            gonef = tl.max((ge & (evinfl[None, :] != 0)).to(tl.int32), axis=1) > 0
            took = tl.max((newblk[None, :] == blk[:, None]).to(tl.int32), axis=1) > 0
            slot = tl.where(gone, -1, slot)
            infl = (infl | took) & (gonef == 0)
        return slot, infl, nres, nfly

    # =================================================================================================================
    # V1: the verify rows' classification (counts for the global K5 prefix and the record compaction)
    # =================================================================================================================
    @triton.jit
    def _fr_v1(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               ne_ptr, rec_ptr, usg_ptr, err_ptr, cntv_ptr, look_ptr,
               H, R, K, NBLK, NPOS, C, Rv, MAXRV, NLOOK,
               KP: tl.constexpr, NP: tl.constexpr, LP: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        free_local = tl.maximum(C - usage, 0)
        _put1(usg_ptr + pid, usage, pid >= 0)
        lk = tl.arange(0, LP)                                                     # program 0 zeroes V2's sums
        tl.store(look_ptr + lk, tl.zeros([LP], tl.int64), mask=(lk < NLOOK) & (pid == 0))
        _put1(cntv_ptr, tl.sum(tl.zeros([2], tl.int64), axis=0), pid == 0)
        start = tl.load(vstart_ptr + b)
        cnt = tl.load(vcnt_ptr + b)
        carry = tl.sum(tl.zeros([2], tl.int32), axis=0)
        errb = tl.sum(tl.zeros([2], tl.int32), axis=0)
        for i in range(0, MAXRV):
            act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rv, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, nres = _lookup(blk, sblk, res, jj)
            islot, nfly = _lookup(blk, sblk, fly, jj)
            ready = valid & (slot >= 0)
            infl = valid & (ready == 0) & (islot >= 0)
            miss = valid & (ready == 0) & (infl == 0)
            seen, seen_k = _seen(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rv, h, R, K, NBLK,
                                 kid, blk, valid, KP)
            fmiss = miss & (seen == 0)
            fm = fmiss.to(tl.int32)
            rank = carry + tl.cumsum(fm, axis=0) - fm
            ne = fmiss & (rank >= free_local)
            idx = j * H + h
            _put1(ne_ptr + idx, tl.sum(ne.to(tl.int32), axis=0), act)
            _put1(rec_ptr + idx, tl.sum((valid & (ready == 0)).to(tl.int32), axis=0), act)
            carry = carry + tl.where(act, tl.sum(fm, axis=0), 0)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            errb = errb | e1 | e2
        tl.atomic_or(err_ptr, errb)

    # =================================================================================================================
    # V2: the verify decisions
    # =================================================================================================================
    @triton.jit
    def _fr_v2(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vtl_ptr, vnt_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               ne_ptr, rec_ptr, usg_ptr,
               tv_ptr, vpage_ptr, vnpg_ptr, ppage_ptr, pnpg_ptr, ids_ptr, vrec_ptr, vrcnt_ptr, cntv_ptr, look_ptr,
               vread_ptr, post_ptr, usg2_ptr, selw_ptr, rowsel_ptr, err_ptr, stats_ptr,
               H, B, R, K, NBLK, NPOS, C, X, Rv, Rd, MAXRV, MAXRD, NV, CAPV, RH, S, WB, WP, T, P,
               KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr, RHP: tl.constexpr,
               SP: tl.constexpr, ZP: tl.constexpr, PE_INFL: tl.constexpr, LAST: tl.constexpr, PLANT: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        okb = (sblk >= 0) & (sblk < NBLK)
        free = mj & (code == 0)
        fi = free.to(tl.int32)
        fr = tl.cumsum(fi, axis=0) - fi                                          # rank among the FREE slots
        free_local = tl.maximum(C - usage, 0)
        # K5's shared extras for the verifier: X - borrowed (the reserve R_ver included)
        ss = tl.arange(0, SP)
        usgv = tl.load(usg_ptr + ss, mask=ss < S, other=0)
        budget = tl.maximum(X - tl.sum(tl.maximum(usgv - C, 0), axis=0), 0)
        rr = tl.arange(0, RHP)
        nev = tl.load(ne_ptr + rr, mask=rr < RH, other=0)
        recv = tl.load(rec_ptr + rr, mask=rr < RH, other=0)
        totrec = tl.sum(recv, axis=0)
        vstart = tl.load(vstart_ptr + b)
        vcnt = tl.load(vcnt_ptr + b)
        dstart = tl.load(dstart_ptr + b)
        dcnt = tl.load(dcnt_ptr + b)
        # Policy E: the blocks that some row of the layer selects (verify and draft rows of b), per slot
        insel = mj & (code < -1)
        for i in range(0, MAXRV):
            act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv, h, R,
                                          K, NBLK, KP)
            hit = tl.max((v0[None, :] & (s[None, :] == sblk[:, None])).to(tl.int32), axis=1) > 0
            insel = insel | (hit & act)
        for i in range(0, MAXRD):
            act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, dstart, dcnt, i, Rd, h, R,
                                          K, NBLK, KP)
            hit = tl.max((v0[None, :] & (s[None, :] == sblk[:, None])).to(tl.int32), axis=1) > 0
            insel = insel | (hit & act)
        insel = insel & okb
        prot_r = resid & (prot != 0)
        cand = (code == 2) & (prot_r == 0) & (insel == 0)
        if PE_INFL:
            candf = fly & (prot == 0) & (insel == 0)
        else:
            candf = cand & (cand == 0)
        anyc = cand | candf
        # one key per slot: class (LRU candidate 0, in-flight candidate 1, none 2) << 56 | stamp << 9 | slot
        cls = tl.where(cand, 0, tl.where(candf, 1, 2)).to(tl.int64)
        kst = tl.where(cand, stamp, tl.where(candf, tl.minimum(stamp, 2305843009213693952), 0))
        comp = cls * 72057594037927936 + kst * 512 + jj.to(tl.int64)
        srtc = tl.sort(comp)
        ncand = tl.sum(anyc.to(tl.int32), axis=0)
        errb = tl.where(tl.max((anyc & ((stamp < 0) | (stamp >= 140737488355328))).to(tl.int32), axis=0) > 0, 8, 0)
        # the stream's state after its verify rows (for the draft rows)
        newblk = tl.full([NP], -1, tl.int32)
        evold = tl.full([NP], -1, tl.int32)
        evinfl = tl.zeros([NP], tl.int32)
        free2 = free
        usage2 = usage
        rbits = mj & (code < -1)
        selu = _selbits(tl.full([KP], -1, tl.int32), tl.full([KP], -1, tl.int32) >= 0, ZP)
        carry_f = tl.sum(tl.zeros([2], tl.int32), axis=0)
        carry_g = carry_f
        carry_r = carry_f
        kk = tl.arange(0, KP)
        mk = kk < K
        cc = tl.arange(0, PP)
        tt = tl.arange(0, TP)
        rrn = tl.arange(0, NP)
        q4 = tl.arange(0, 4)
        st8 = tl.arange(0, 8)
        zw = tl.arange(0, ZP // 32)
        for i in range(0, MAXRV):
            act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv, h, R,
                                          K, NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, nres = _lookup(blk, sblk, res, jj)
            islot, nfly = _lookup(blk, sblk, fly, jj)
            igen = _at_slot(islot, jj, gen, 0)
            ready = valid & (slot >= 0)
            infl = valid & (ready == 0) & (islot >= 0)
            miss = valid & (ready == 0) & (infl == 0)
            seen, seen_k = _seen(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv, h, R, K, NBLK,
                                 kid, blk, valid, KP)
            fmiss = miss & (seen == 0)
            dup = miss & seen
            first = valid & (seen_k == 0)
            fm = fmiss.to(tl.int32)
            rank = carry_f + tl.cumsum(fm, axis=0) - fm
            ne = fmiss & (rank >= free_local)
            nei = ne.to(tl.int32)
            idx = j * H + h
            g = tl.sum(tl.where(rr < idx, nev, 0), axis=0) + tl.cumsum(nei, axis=0) - nei
            granted = fmiss & ((ne == 0) | (g < budget))
            refused = fmiss & (granted == 0)
            gi = granted.to(tl.int32)
            q = carry_g + tl.cumsum(gi, axis=0) - gi
            qc = tl.minimum(q, NPOS - 1)
            fs = tl.max(tl.where(free[None, :] & (fr[None, :] == qc[:, None]), jj[None, :], -1), axis=1)
            fsl = tl.where(fs >= 0, fs, NPOS)
            ri = refused.to(tl.int32)
            rev = carry_r + tl.cumsum(ri, axis=0) - ri
            evicted = refused & (rev < ncand)
            if PLANT == 1:
                ruse = tl.where(evicted, ncand - 1 - rev, rev)
            else:
                ruse = rev
            evs = (tl.sum(tl.where(rrn[None, :] == ruse[:, None], srtc[None, :], 0), axis=1) & 511).to(tl.int32)
            evblk_raw = _at_slot(evs, jj, sblk, -1073741824)
            evfly = _at_slot(evs, jj, fly.to(tl.int32), 0)
            evgen_raw = _at_slot(evs, jj, gen, 0)
            ev_blk = tl.where(evicted, evblk_raw, -1)
            ev_infl = evicted & (evfly > 0)
            ev_gen = tl.where(ev_infl, evgen_raw, 0)
            over = refused & (evicted == 0)
            take = tl.where(granted, fsl, tl.where(evicted, evs, -1))
            taken = granted | evicted
            grp = tl.max(tl.where((newblk[None, :] >= 0) & (newblk[None, :] == blk[:, None]), jj[None, :], -1),
                         axis=1)
            rslot = tl.where(ready, slot, tl.where(infl, islot, tl.where(fmiss, take, tl.where(dup, grp, -1))))
            hole = valid & (rslot < 0)
            rpage = tl.where(hole, 0, rslot)
            # the decision's fields (K3X entry order: o = (j H + h) k + p)
            o = idx * K + kk
            me = mk & act
            tl.store(tv_ptr + 0 * NV + o, blk.to(tl.int64), mask=me)
            tl.store(tv_ptr + 1 * NV + o, valid.to(tl.int64), mask=me)
            tl.store(tv_ptr + 2 * NV + o, ready.to(tl.int64), mask=me)
            tl.store(tv_ptr + 3 * NV + o, slot.to(tl.int64), mask=me)
            tl.store(tv_ptr + 4 * NV + o, infl.to(tl.int64), mask=me)
            tl.store(tv_ptr + 5 * NV + o, islot.to(tl.int64), mask=me)
            tl.store(tv_ptr + 6 * NV + o, igen.to(tl.int64), mask=me)
            tl.store(tv_ptr + 7 * NV + o, dup.to(tl.int64), mask=me)
            tl.store(tv_ptr + 8 * NV + o, granted.to(tl.int64), mask=me)
            tl.store(tv_ptr + 9 * NV + o, take.to(tl.int64), mask=me)
            tl.store(tv_ptr + 10 * NV + o, rslot.to(tl.int64), mask=me)
            tl.store(tv_ptr + 11 * NV + o, valid.to(tl.int64), mask=me)
            tl.store(tv_ptr + 12 * NV + o, evicted.to(tl.int64), mask=me)
            tl.store(tv_ptr + 13 * NV + o, ev_blk.to(tl.int64), mask=me)
            tl.store(tv_ptr + 14 * NV + o, ev_infl.to(tl.int64), mask=me)
            tl.store(tv_ptr + 15 * NV + o, ev_gen.to(tl.int64), mask=me)
            # the page table: every valid block in ascending block id (its read slot), then the row's tail slots
            vi = valid.to(tl.int32)
            vp = tl.cumsum(vi, axis=0) - vi
            nvalid = tl.sum(vi, axis=0)
            hist = tl.max(tl.where(valid[None, :] & (vp[None, :] == cc[:, None]), rpage[None, :], -1), axis=1)
            tv = tl.load(vtl_ptr + j * T + tt, mask=tt < T, other=-1)
            ntr = tl.load(vnt_ptr + j)
            tcol = cc - nvalid
            tailv = tl.max(tl.where((tcol[:, None] == tt[None, :]) & (tt[None, :] < ntr), tv[None, :], -1), axis=1)
            page = tl.where(cc < nvalid, hist.to(tl.int64), tl.where((tcol >= 0) & (tcol < ntr), tailv, -1))
            mp = (cc < P) & act
            tl.store(vpage_ptr + idx * P + cc, page, mask=mp)
            tl.store(ppage_ptr + (tr * H + h) * P + cc, page.to(tl.int32), mask=mp)
            npg = nvalid + ntr
            _put1(vnpg_ptr + idx, npg.to(tl.int64), act)
            _put1(pnpg_ptr + tr * H + h, npg.to(tl.int32), act)
            # the verify record (pin_records False): every valid entry that is not READY, compacted in entry order
            rm = valid & (ready == 0)
            rmi = rm.to(tl.int32)
            rpos = tl.sum(tl.where(rr < idx, recv, 0), axis=0) + tl.cumsum(rmi, axis=0) - rmi
            wm = rm & act & (rpos < CAPV)
            flags = (ready.to(tl.int64) * 2 + infl.to(tl.int64) * 4 + dup.to(tl.int64) * 8 + granted.to(tl.int64) * 16
                     + evicted.to(tl.int64) * 32 + ev_infl.to(tl.int64) * 64 + first.to(tl.int64) * 128
                     + over.to(tl.int64) * 256)            # the record's HOLE: valid & ~(ready|infl|dup|gr|ev)
            tl.store(vrec_ptr + 0 * CAPV + rpos, o.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 1 * CAPV + rpos, blk.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 2 * CAPV + rpos, flags, mask=wm)
            tl.store(vrec_ptr + 3 * CAPV + rpos, slot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 4 * CAPV + rpos, islot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 5 * CAPV + rpos, igen.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 6 * CAPV + rpos, rslot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 7 * CAPV + rpos, take.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 8 * CAPV + rpos, ev_blk.to(tl.int64), mask=wm)
            pm = me & (o >= totrec) & (o < CAPV)                                  # the -1 padding (no record there)
            neg = tl.full([KP], -1, tl.int64)
            for f in tl.static_range(9):
                tl.store(vrec_ptr + f * CAPV + o, neg, mask=pm)
            # the looks: READY first looks of (key, head, block), by the slot's state class
            rf = ready & first
            cat = _at_slot(tl.maximum(slot, 0), jj, code, -1)
            c0 = tl.sum(rf.to(tl.int64), axis=0)
            c1 = tl.sum((rf & (cat == 2)).to(tl.int64), axis=0)
            c2 = tl.sum((rf & (cat == 3)).to(tl.int64), axis=0)
            c3 = tl.sum((rf & (cat == 4)).to(tl.int64), axis=0)
            lv = tl.where(q4 == 0, c0, tl.where(q4 == 1, c1, tl.where(q4 == 2, c2, c3)))
            tl.atomic_add(look_ptr + j * 4 + q4, lv, mask=(q4 < 4) & act)
            # the verify read bitmap: every slot a verify row reads
            rs = tl.where(ready, slot, tl.where(infl, islot, tl.where(dup, rslot, tl.where(taken, take, -1))))
            rd = valid & (rs >= 0) & act
            rbits = rbits | (tl.max((rd[:, None] & (rs[:, None] == jj[None, :])).to(tl.int32), axis=0) > 0)
            # the stream's state after this row (S-C's mirror update, per slot)
            upd = (taken & act)[:, None] & (take[:, None] == jj[None, :])
            hasu = tl.max(upd.to(tl.int32), axis=0) > 0
            newblk = tl.where(hasu, tl.max(tl.where(upd, blk[:, None], -1), axis=0), newblk)
            upe = (evicted & act)[:, None] & (evs[:, None] == jj[None, :])
            hase = tl.max(upe.to(tl.int32), axis=0) > 0
            evold = tl.where(hase, tl.max(tl.where(upe, tl.maximum(evblk_raw, 0)[:, None], -1), axis=0), evold)
            evinfl = tl.where(hase, tl.max(tl.where(upe, ev_infl.to(tl.int32)[:, None], 0), axis=0), evinfl)
            upg = (granted & act)[:, None] & (take[:, None] == jj[None, :])
            free2 = free2 & (tl.max(upg.to(tl.int32), axis=0) == 0)
            usage2 = usage2 + tl.where(act, tl.sum(gi, axis=0), 0)
            carry_f = carry_f + tl.where(act, tl.sum(fm, axis=0), 0)
            carry_g = carry_g + tl.where(act, tl.sum(gi, axis=0), 0)
            carry_r = carry_r + tl.where(act, tl.sum(ri, axis=0), 0)
            tl.atomic_add(cntv_ptr, tl.where(act, tl.sum(over.to(tl.int64), axis=0), 0))
            # REPORT / L34 preconditions: per kind, the entries of this row
            n_ev = tl.sum(evicted.to(tl.int64), axis=0)
            n_fl = tl.sum((evicted & (ncand - 1 - rev != rev)).to(tl.int64), axis=0)   # plant 1 changes these
            sv = tl.where(st8 == 0, n_ev, tl.where(st8 == 1, n_fl, tl.where(
                st8 == 2, tl.sum(gi.to(tl.int64), axis=0), tl.where(
                    st8 == 5, tl.sum(over.to(tl.int64), axis=0), tl.where(
                        st8 == 6, tl.sum(dup.to(tl.int64), axis=0), tl.where(
                            st8 == 7, tl.sum(ev_infl.to(tl.int64), axis=0), 0))))))
            tl.atomic_add(stats_ptr + st8, sv, mask=(st8 < 8) & act)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            e4 = tl.where(act & (tl.max((granted & (fs < 0)).to(tl.int32), axis=0) > 0), 4, 0)
            errb = errb | e1 | e2 | e4
            # this row's selection bitmap (rowsel), and the stream's union (sel) when no draft row follows
            rb_ = _selbits(s, v0, ZP)
            tl.store(rowsel_ptr + (tr * H + h) * WB + zw, _pack(rb_, ZP), mask=(zw < WB) & act)
            if LAST:
                selu = selu | (rb_ & act)
        tl.store(ids_ptr + (h * B + b) * NPOS + jj, newblk.to(tl.int64), mask=mj)
        pw = tl.arange(0, NP // 32)
        tl.store(vread_ptr + pid * WP + pw, _pack(rbits, NP), mask=pw < WP)
        tl.store(post_ptr + (0 * S + pid) * NP + jj, newblk)
        tl.store(post_ptr + (1 * S + pid) * NP + jj, evold)
        tl.store(post_ptr + (2 * S + pid) * NP + jj, evinfl)
        tl.store(post_ptr + (3 * S + pid) * NP + jj, free2.to(tl.int32))
        _put1(usg2_ptr + pid, usage2, pid >= 0)
        _put1(vrcnt_ptr, totrec.to(tl.int64), pid == 0)
        if LAST:
            tl.store(selw_ptr + pid * WB + zw, _pack(selu, ZP), mask=zw < WB)
        tl.atomic_or(err_ptr, errb)

    # =================================================================================================================
    # D1: the draft rows' classification on the state after the verify rows
    # =================================================================================================================
    @triton.jit
    def _fr_d1(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               post_ptr, usg2_ptr, ne_ptr, rec_ptr, usgd_ptr, err_ptr, cntd_ptr,
               H, R, K, NBLK, NPOS, C, Rd, MAXRD, S, NPAIRS,
               KP: tl.constexpr, NP: tl.constexpr, HAS_V: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        if HAS_V:
            newblk = tl.load(post_ptr + (0 * S + pid) * NP + jj)
            evold = tl.load(post_ptr + (1 * S + pid) * NP + jj)
            evinfl = tl.load(post_ptr + (2 * S + pid) * NP + jj)
            usage = tl.load(usg2_ptr + pid)
        else:
            newblk = tl.full([NP], -1, tl.int32)
            evold = newblk
            evinfl = tl.zeros([NP], tl.int32)
        free_local = tl.maximum(C - usage, 0)
        _put1(usgd_ptr + pid, usage, pid >= 0)
        q4 = tl.arange(0, 4)                                                       # program 0: (masked, refused, pairs)
        tl.store(cntd_ptr + q4, tl.where(q4 == 2, NPAIRS, 0).to(tl.int64), mask=(q4 < 3) & (pid == 0))
        start = tl.load(dstart_ptr + b)
        cnt = tl.load(dcnt_ptr + b)
        carry = tl.sum(tl.zeros([2], tl.int32), axis=0)
        errb = tl.sum(tl.zeros([2], tl.int32), axis=0)
        for i in range(0, MAXRD):
            act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, inflight, nres, nfly = _d_state(blk, sblk, res, fly, jj, newblk, evold, evinfl, HAS_V)
            ready = valid & (slot >= 0)
            infl0 = valid & (ready == 0) & inflight
            miss = valid & (ready == 0) & (infl0 == 0)
            seen, seen_k = _seen(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K, NBLK,
                                 kid, blk, valid, KP)
            fmiss = miss & (seen == 0)
            fm = fmiss.to(tl.int32)
            rank = carry + tl.cumsum(fm, axis=0) - fm
            ne = fmiss & (rank >= free_local)
            idx = j * H + h
            _put1(ne_ptr + idx, tl.sum(ne.to(tl.int32), axis=0), act)
            _put1(rec_ptr + idx, tl.sum(valid.to(tl.int32), axis=0), act)
            carry = carry + tl.where(act, tl.sum(fm, axis=0), 0)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            errb = errb | e1 | e2
        tl.atomic_or(err_ptr, errb)

    # =================================================================================================================
    # D2: the draft decisions, the draft record, the bias rows, the selection bitmaps
    # =================================================================================================================
    @triton.jit
    def _fr_d2(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               dix_ptr, dhb_ptr, dtl_ptr, dnt_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               post_ptr, ne_ptr, rec_ptr, usgd_ptr,
               td_ptr, dpage_ptr, dnpg_ptr, ppage_ptr, pnpg_ptr, drec_ptr, dcnt_out_ptr, cntd_ptr,
               selw_ptr, rowsel_ptr, err_ptr, stats_ptr, sb_ptr, cis_ptr,
               H, B, R, K, NBLK, NPOS, C, X, RVER, Rv, Rd, MAXRV, MAXRD, ND, CAPD, RH, S, WB, T, P,
               BS, ROWS_S, ROWS_C,
               KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr, RHP: tl.constexpr,
               SP: tl.constexpr, ZP: tl.constexpr, BSP: tl.constexpr, HAS_V: tl.constexpr, BIAS: tl.constexpr,
               PLANT: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        if HAS_V:
            newblk = tl.load(post_ptr + (0 * S + pid) * NP + jj)
            evold = tl.load(post_ptr + (1 * S + pid) * NP + jj)
            evinfl = tl.load(post_ptr + (2 * S + pid) * NP + jj)
            free = (tl.load(post_ptr + (3 * S + pid) * NP + jj) != 0) & mj
        else:
            newblk = tl.full([NP], -1, tl.int32)
            evold = newblk
            evinfl = tl.zeros([NP], tl.int32)
            free = mj & (code == 0)
        usage = tl.load(usgd_ptr + pid)
        fi = free.to(tl.int32)
        fr = tl.cumsum(fi, axis=0) - fi
        free_local = tl.maximum(C - usage, 0)
        ss = tl.arange(0, SP)
        usgv = tl.load(usgd_ptr + ss, mask=ss < S, other=0)
        budget = tl.maximum(X - tl.sum(tl.maximum(usgv - C, 0), axis=0) - RVER, 0)
        rr = tl.arange(0, RHP)
        nev = tl.load(ne_ptr + rr, mask=rr < RH, other=0)
        recv = tl.load(rec_ptr + rr, mask=rr < RH, other=0)
        totrec = tl.sum(recv, axis=0)
        start = tl.load(dstart_ptr + b)
        cnt = tl.load(dcnt_ptr + b)
        carry_f = tl.sum(tl.zeros([2], tl.int32), axis=0)
        carry_g = carry_f
        errb = carry_f
        kk = tl.arange(0, KP)
        mk = kk < K
        cc = tl.arange(0, PP)
        tt = tl.arange(0, TP)
        st8 = tl.arange(0, 8)
        zw = tl.arange(0, ZP // 32)
        bb = tl.arange(0, BSP)
        selu = _selbits(tl.full([KP], -1, tl.int32), tl.full([KP], -1, tl.int32) >= 0, ZP)
        for i in range(0, MAXRD):
            act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, inflight, nres, nfly = _d_state(blk, sblk, res, fly, jj, newblk, evold, evinfl, HAS_V)
            ready = valid & (slot >= 0)
            infl0 = valid & (ready == 0) & inflight
            miss = valid & (ready == 0) & (infl0 == 0)
            seen, seen_k = _seen(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K, NBLK,
                                 kid, blk, valid, KP)
            fmiss = miss & (seen == 0)
            dup = miss & seen
            fm = fmiss.to(tl.int32)
            rank = carry_f + tl.cumsum(fm, axis=0) - fm
            ne = fmiss & (rank >= free_local)
            nei = ne.to(tl.int32)
            idx = j * H + h
            g = tl.sum(tl.where(rr < idx, nev, 0), axis=0) + tl.cumsum(nei, axis=0) - nei
            granted = fmiss & ((ne == 0) | (g < budget))
            refused = fmiss & (granted == 0)
            gi = granted.to(tl.int32)
            q = carry_g + tl.cumsum(gi, axis=0) - gi
            if PLANT == 2:
                qc = tl.minimum(q + 1, NPOS - 1)
            else:
                qc = tl.minimum(q, NPOS - 1)
            fs = tl.max(tl.where(free[None, :] & (fr[None, :] == qc[:, None]), jj[None, :], -1), axis=1)
            take = tl.where(granted, tl.where(fs >= 0, fs, NPOS), -1)
            infl = infl0 | dup
            o = idx * K + kk
            me = mk & act
            tl.store(td_ptr + 0 * ND + o, blk.to(tl.int64), mask=me)
            tl.store(td_ptr + 1 * ND + o, valid.to(tl.int64), mask=me)
            tl.store(td_ptr + 2 * ND + o, ready.to(tl.int64), mask=me)
            tl.store(td_ptr + 3 * ND + o, slot.to(tl.int64), mask=me)
            tl.store(td_ptr + 4 * ND + o, infl.to(tl.int64), mask=me)
            tl.store(td_ptr + 5 * ND + o, granted.to(tl.int64), mask=me)
            tl.store(td_ptr + 6 * ND + o, refused.to(tl.int64), mask=me)
            tl.store(td_ptr + 7 * ND + o, take.to(tl.int64), mask=me)
            tl.store(td_ptr + 8 * ND + o, valid.to(tl.int64), mask=me)
            # the page table: the READY blocks in ascending block id (their slots), then the row's tail slots
            ri_ = ready.to(tl.int32)
            rp = tl.cumsum(ri_, axis=0) - ri_
            nready = tl.sum(ri_, axis=0)
            nvalid = tl.sum(valid.to(tl.int32), axis=0)
            hist = tl.max(tl.where(ready[None, :] & (rp[None, :] == cc[:, None]), slot[None, :], -1), axis=1)
            tv = tl.load(dtl_ptr + j * T + tt, mask=tt < T, other=-1)
            ntr = tl.load(dnt_ptr + j)
            tcol = cc - nready
            tailv = tl.max(tl.where((tcol[:, None] == tt[None, :]) & (tt[None, :] < ntr), tv[None, :], -1), axis=1)
            page = tl.where(cc < nready, hist.to(tl.int64), tl.where((tcol >= 0) & (tcol < ntr), tailv, -1))
            mp = (cc < P) & act
            tl.store(dpage_ptr + idx * P + cc, page, mask=mp)
            tl.store(ppage_ptr + (tr * H + h) * P + cc, page.to(tl.int32), mask=mp)
            npg = nready + ntr
            _put1(dnpg_ptr + idx, npg.to(tl.int64), act)
            _put1(pnpg_ptr + tr * H + h, npg.to(tl.int32), act)
            # the draft record: every valid entry, compacted in entry order
            vi = valid.to(tl.int32)
            rpos = tl.sum(tl.where(rr < idx, recv, 0), axis=0) + tl.cumsum(vi, axis=0) - vi
            wm = valid & act & (rpos < CAPD)
            hole = valid & ((ready | infl | granted | refused) == 0)
            flags = (ready.to(tl.int64) * 2 + infl.to(tl.int64) * 4 + granted.to(tl.int64) * 16
                     + refused.to(tl.int64) * 512 + hole.to(tl.int64) * 256)
            tl.store(drec_ptr + 0 * CAPD + rpos, o.to(tl.int64), mask=wm)
            tl.store(drec_ptr + 1 * CAPD + rpos, blk.to(tl.int64), mask=wm)
            tl.store(drec_ptr + 2 * CAPD + rpos, flags, mask=wm)
            tl.store(drec_ptr + 3 * CAPD + rpos, slot.to(tl.int64), mask=wm)
            tl.store(drec_ptr + 4 * CAPD + rpos, take.to(tl.int64), mask=wm)
            pm = me & (o >= totrec) & (o < CAPD)
            neg = tl.full([KP], -1, tl.int64)
            for f in tl.static_range(5):
                tl.store(drec_ptr + f * CAPD + o, neg, mask=pm)
            # the counters (masked pairs, refusals) and the REPORT / L34 preconditions
            nref = tl.sum(refused.to(tl.int64), axis=0)
            q4 = tl.arange(0, 4)
            cv = tl.where(q4 == 0, (nready < nvalid).to(tl.int64), tl.where(q4 == 1, nref, 0))
            tl.atomic_add(cntd_ptr + q4, cv, mask=(q4 < 2) & act)
            sv = tl.where(st8 == 3, tl.sum(gi.to(tl.int64), axis=0), tl.where(st8 == 4, nref, 0))
            tl.atomic_add(stats_ptr + st8, sv, mask=((st8 == 3) | (st8 == 4)) & act)
            # the bias rows of the granted takes (S-B's draft_bias_fill), bit-exact copies
            if BIAS:
                gm = (granted & act & (take >= 0) & (take < NPOS))[:, None] & (bb[None, :] < BS)
                b64 = b.to(tl.int64)
                src = cis_ptr + ((b64 * ROWS_C + blk[:, None].to(tl.int64) * BS + bb[None, :]) * H + h)
                dst = sb_ptr + ((b64 * ROWS_S + take[:, None].to(tl.int64) * BS + bb[None, :]) * H + h)
                tl.store(dst, tl.load(src, mask=gm, other=0), mask=gm)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            e4 = tl.where(act & (tl.max((granted & (fs < 0)).to(tl.int32), axis=0) > 0), 4, 0)
            errb = errb | e1 | e2 | e4
            rb_ = _selbits(s, v0, ZP)
            tl.store(rowsel_ptr + (tr * H + h) * WB + zw, _pack(rb_, ZP), mask=(zw < WB) & act)
            selu = selu | (rb_ & act)
            carry_f = carry_f + tl.where(act, tl.sum(fm, axis=0), 0)
            carry_g = carry_g + tl.where(act, tl.sum(gi, axis=0), 0)
        if HAS_V:                                                                  # the verify rows' blocks too
            vstart = tl.load(vstart_ptr + b)
            vcnt = tl.load(vcnt_ptr + b)
            for i in range(0, MAXRV):
                act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv, h,
                                              R, K, NBLK, KP)
                selu = selu | (_selbits(s, v0, ZP) & act)
        tl.store(selw_ptr + pid * WB + zw, _pack(selu, ZP), mask=zw < WB)
        _put1(dcnt_out_ptr, totrec.to(tl.int64), pid == 0)
        tl.atomic_or(err_ptr, errb)


def _launch(kernel, grid, *args, **kw):
    """Every launch of this module goes through here (tests count them)."""
    kernel[grid](*args, **kw)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A float tensor's raw bits (the bias copy is bit-exact for any 2- or 4-byte float dtype)."""
    if t.element_size() == 2:
        return t.view(torch.int16)
    if t.element_size() == 4:
        return t.view(torch.int32)
    raise ValueError("bias rows of %s: 2- or 4-byte floats only" % t.dtype)


class FusedResolver:
    """The fused resolve of one OwnResidency (module docstring). check: also run the torch path on the same inputs at
    every layer and record every differing output in self.diffs (tests, the GPU equality gate; it reads the device).
    plant: a NEGATIVE CONTROL (PLANTS), tests only."""

    def __init__(self, svc, check: bool = False, plant: int = 0, num_warps: int = 8):
        if triton is None:
            raise RuntimeError("resolve='fused' needs triton (not installed)")
        if getattr(svc, "xpool", None) is not None:
            raise ValueError("resolve='fused' does not bind the shared X pool's lanes yet (docs/R2_FUSEDRES_PLAN.md, "
                             "section 5: the merge with nosi-r2-xpool)")
        if plant not in PLANTS:
            raise ValueError("fused plant %r (one of %s)" % (plant, sorted(PLANTS)))
        if getattr(svc, "record_cap", None) is not None:
            raise ValueError("resolve='fused' writes every record (record_cap must be None)")
        if getattr(svc, "pin_records", True):
            raise ValueError("resolve='fused' is B2's path: pin_records must be False")
        for name in ("sc_mut", "sb_mut", "sparse_mut"):
            if getattr(svc, name, None):
                raise ValueError("resolve='fused' implements no %s negative control (got %s)"
                                 % (name, getattr(svc, name)))
        if type(svc)._layer_record is not ResidencySparse._layer_record:
            raise ValueError("resolve='fused' does not call _layer_record: %s overrides it" % type(svc).__name__)
        if svc.n_hist > MAX_NPOS:
            raise ValueError("resolve='fused': C + X = %d > %d" % (svc.n_hist, MAX_NPOS))
        self.svc = svc
        self.check = bool(check)
        self.plant = int(plant)
        self.num_warps = int(num_warps)
        self.pe_inflight = bool(svc.policy_e_inflight)
        dev = svc.device
        self.err = torch.zeros((), dtype=torch.int32, device=dev)
        self.stats = torch.zeros((len(STATS),), dtype=torch.int64, device=dev)
        svc.watch_flag("fused_res_err", self.err)
        self.launches = 0                                  # Triton launches (all layers, all ticks)
        self.layers = 0                                    # fused layers resolved
        self.bias_fallbacks = 0                            # layers whose bias rows took S-B's torch fill (dtypes)
        self.compared = 0                                  # check: layers compared with the torch path
        self.diffs = []                                    # check: (tick, layer, output) that differ
        self.trace = []                                    # check: (tick, layer, stats after the layer)

    # ----------------------------------------------------------------------------------------------- the service hook
    def resolve(self, l, rows, sel):
        """ResidencySC.resolve with the fused per-layer path (the prologue and the per-tick copies unchanged)."""
        svc = self.svc
        if len(rows) == 0:
            return ResidencySC.resolve(svc, l, rows, sel)
        if any(st["l"] == l for st in svc._tick):
            raise ValueError("S-C: layer %d resolved twice in one tick (its decisions are applied at end_tick)" % l)
        if sel.device.type != svc.device.type:
            raise ValueError("S-C: the selection must be on the service's device %s (got %s)" % (svc.device, sel.device))
        svc._check_rows(l, rows, sel)
        if not svc._tick:                                             # the tick's first layer: the per-tick copies
            svc.sync_mirror()
            svc._rows = svc._tick_rows(rows)
        elif svc._dirty:
            raise OwnershipViolation("S-C: the host state changed in the middle of a tick (the device mirror is stale)")
        elif svc._row_sig(rows) != svc._rows["sig"]:
            raise ValueError("S-C: every layer of a tick resolves the same rows (layer %d differs)" % l)
        tk = svc._rows
        plan = svc._new_plan(rows)
        fz = tk.get("fz")
        if fz is None:
            fz = tk["fz"] = self._tick_setup(rows, tk)
        step, page, npg = self._layer(l, rows, sel, tk, fz)
        svc._tick.append(step)
        plan.page_table = page
        plan.num_pages = npg
        plan.last_len = tk["last_len"]
        plan.seq_idx = tk["seq_idx"]
        return plan

    # ----------------------------------------------------------------------------------------------- per tick
    def _tick_setup(self, rows, tk):
        """The tick's row lists per batch slot (ONE host-to-device copy) and the scratch buffers (reused by every
        layer of the tick: the layers run in stream order)."""
        svc = self.svc
        B, H, dev = svc.B, svc.H, svc.device
        S = B * H
        NP = max(_pow2(svc.n_hist), 64)                     # >= 64: two words at least (no size-1 tile dimension)

        def group(idx):
            bs = [rows[i].b for i in idx]
            cnt = [0] * B
            for x in bs:
                cnt[x] += 1
            start, acc = [], 0
            for c in cnt:
                start.append(acc)
                acc += c
            order = sorted(range(len(bs)), key=lambda j: (bs[j], j))
            return start, cnt, order, (max(cnt) if bs else 0)
        self.err.zero_()                                    # per tick: the previous end_tick has read it (one op)
        vs, vc, vo, mv = group(tk["vrows"])
        ds, dc, do, md = group(tk["drows"])
        flat = vs + vc + (vo or [0]) + ds + dc + (do or [0])
        d = svc._h2d(torch.tensor(flat, dtype=torch.int64))
        parts, at = [], 0
        for n in (B, B, max(len(vo), 1), B, B, max(len(do), 1)):
            parts.append(d[at:at + n])
            at += n
        Rv, Rd = len(tk["vrows"]), len(tk["drows"])
        one = torch.zeros(1, dtype=torch.int64, device=dev)
        one_t = torch.full((1, N_TAIL), -1, dtype=torch.int64, device=dev)
        v = tk["v"] if tk["v"] is not None else (one, one, one_t, one, one)
        dd = tk["d"] if tk["d"] is not None else (one, one, one_t, one, one)
        i32 = dict(dtype=torch.int32, device=dev)
        return dict(vstart=parts[0], vcnt=parts[1], vorder=parts[2], dstart=parts[3], dcnt=parts[4],
                    dorder=parts[5], maxrv=mv, maxrd=md, Rv=Rv, Rd=Rd, v=v, d=dd, kid=tk["extra"][:len(rows)],
                    ne_v=torch.empty(max(Rv * H, 1), **i32), rec_v=torch.empty(max(Rv * H, 1), **i32),
                    ne_d=torch.empty(max(Rd * H, 1), **i32), rec_d=torch.empty(max(Rd * H, 1), **i32),
                    usg=torch.empty(S, **i32), usg2=torch.empty(S, **i32), usgd=torch.empty(S, **i32),
                    post=torch.empty((4, S, NP), **i32), NP=NP, bh={})

    def _bh(self, fz, k):
        """(b, h) of every verify entry (K3X order; VerifyDecision.bh, read by S-C's device wait), per selection width
        k (device ops, once per tick and width)."""
        out = fz["bh"].get(k)
        if out is None:
            svc = self.svc
            H, Rv = svc.H, fz["Rv"]
            bb = fz["v"][0].view(Rv, 1, 1).expand(Rv, H, k).reshape(-1)
            hh = torch.arange(H, device=svc.device).view(1, H, 1).expand(Rv, H, k).reshape(-1)
            out = fz["bh"][k] = (bb, hh)
        return out

    def _bhd(self, fz, k):
        """(b, h) of every draft entry (the bias fill's fallback only)."""
        key = ("d", k)
        out = fz["bh"].get(key)
        if out is None:
            svc = self.svc
            H, Rd = svc.H, fz["Rd"]
            bb = fz["d"][0].view(Rd, 1, 1).expand(Rd, H, k).reshape(-1)
            hh = torch.arange(H, device=svc.device).view(1, H, 1).expand(Rd, H, k).reshape(-1)
            out = fz["bh"][key] = (bb, hh)
        return out

    # ----------------------------------------------------------------------------------------------- one layer
    def _layer(self, l, rows, sel, tk, fz):
        """Layer l: allocations (no kernel), the launches, the decision objects. No host read, no host-to-device copy
        (test FR4 checks the source)."""
        svc = self.svc
        B, H, dev = svc.B, svc.H, svc.device
        S = B * H
        R = len(rows)
        NPOS, NBLK = svc.n_hist, svc.nblk
        k = sel.shape[2]
        P = k + N_TAIL
        if sel.dtype != torch.int64 or not sel.is_contiguous():       # L28: the kernels compute flat addresses
            sel = sel.to(torch.int64).contiguous()
        Rv, Rd = fz["Rv"], fz["Rd"]
        NP = fz["NP"]
        KP = max(_pow2(k), 2)
        PP = max(_pow2(P), 2)
        TP = max(_pow2(N_TAIL), 2)
        ZP = max(_pow2(NBLK), 64)
        SP = max(_pow2(S), 2)
        WB, WP = (NBLK + 31) // 32, (NPOS + 31) // 32
        dm = svc.dmirror
        tabs = (dm.code[l], dm.block[l], dm.gen[l], dm.cur[l], dm.prot[l], dm.stamp[l])
        kid = fz["kid"]
        i64 = dict(dtype=torch.int64, device=dev)
        page = torch.empty((R, H, P), dtype=torch.int32, device=dev)
        npg = torch.empty((R, H), dtype=torch.int32, device=dev)
        selw = torch.empty((B, H, WB), **i64)
        rowsel = torch.empty((R, H, WB), **i64)
        rec = {}
        vrb, vhb, vtl, vnt, vix = fz["v"]
        drb, dhb, dtl, dnt, dix = fz["d"]
        vdec = ddec = None
        nv, nd = Rv * H * k, Rd * H * k
        caps = (max(nv, 1), max(nd, 1))
        nw = self.num_warps
        ref = torch_reference(svc, l, tk, sel) if self.check else None     # check mode: before any launch
        where = (svc.tick_no, l)
        if Rv:
            RHP = max(_pow2(Rv * H), 2)
            tv = torch.empty((len(FIELDS_V), nv), **i64)
            vpage = torch.empty((Rv, H, P), **i64)
            vnpg = torch.empty((Rv, H), **i64)
            ids = torch.empty((H, B, NPOS), **i64)
            vrec = torch.empty((len(VREC), caps[0]), **i64)
            vcnt = torch.empty((1,), **i64)
            cntv = torch.empty((1,), **i64)
            vlook = torch.empty((Rv, 4), **i64)
            vread = torch.empty((B, H, WP), **i64)
            _launch(_fr_v1, (S,), *tabs, sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"],
                    fz["ne_v"], fz["rec_v"], fz["usg"], self.err, cntv, vlook,
                    H, R, k, NBLK, NPOS, svc.C, Rv, fz["maxrv"], Rv * 4,
                    KP=KP, NP=NP, LP=max(_pow2(Rv * 4), 2), num_warps=nw)
            _launch(_fr_v2, (S,), *tabs, sel, kid, vix, vhb, vtl, vnt, fz["vstart"], fz["vcnt"], fz["vorder"],
                    dix, dhb, fz["dstart"], fz["dcnt"], fz["dorder"],
                    fz["ne_v"], fz["rec_v"], fz["usg"],
                    tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, fz["post"], fz["usg2"], selw,
                    rowsel, self.err, self.stats,
                    H, B, R, k, NBLK, NPOS, svc.C, svc.X, Rv, max(Rd, 1), fz["maxrv"], fz["maxrd"], nv, caps[0],
                    Rv * H, S, WB, WP, N_TAIL, P,
                    KP=KP, NP=NP, PP=PP, TP=TP, RHP=RHP, SP=SP, ZP=ZP, PE_INFL=self.pe_inflight,
                    LAST=Rd == 0, PLANT=self.plant, num_warps=nw)
            self.launches += 2
            vdec = VerifyDecision(Rv, H, k, {f: tv[i] for i, f in enumerate(FIELDS_V)}, vpage, vnpg, ids,
                                  self._bh(fz, k), cntv)
            rec.update(vrec=vrec, vcnt=vcnt, vlook=vlook, vread=vread)
            if ref is None:
                svc.dev.wait(l, vdec)                                 # S-C's order: wait, gather, then the drafts
                svc.dev.gather(l, vdec)
            else:                                                     # check: compare first (a wrong decision may
                self.diffs += self._cmp_dec(where, "v", vdec, ref["vdec"])   # make the CPU stand-in raise)
                self._wait_gather_traced(l, vdec, where)
        sb_before = None
        if Rd:
            RHP = max(_pow2(Rd * H), 2)
            td = torch.empty((len(FIELDS), nd), **i64)
            dpage = torch.empty((Rd, H, P), **i64)
            dnpg = torch.empty((Rd, H), **i64)
            drec = torch.empty((len(DREC), caps[1]), **i64)
            dcnt = torch.empty((1,), **i64)
            cntd = torch.empty((3,), **i64)
            bias = svc.store_b is not None and svc.cis is not None
            sb = cis = sel
            rows_s = rows_c = 0
            fallback = False
            if bias:
                sb_t, cis_t = svc.store_b[l], svc.cis[l]
                if self.check:
                    sb_before = sb_t.clone()
                if (sb_t.dtype != cis_t.dtype or sb_t.element_size() not in (2, 4) or not sb_t.is_contiguous()
                        or not cis_t.is_contiguous() or sb_t.dim() != 3 or cis_t.dim() != 3):
                    fallback = True                                   # S-B's torch fill after D2 (counted)
                else:
                    sb, cis = _bits(sb_t), _bits(cis_t)
                    rows_s, rows_c = sb_t.shape[1], cis_t.shape[1]
            _launch(_fr_d1, (S,), *tabs, sel, kid, dix, dhb, fz["dstart"], fz["dcnt"], fz["dorder"],
                    fz["post"], fz["usg2"], fz["ne_d"], fz["rec_d"], fz["usgd"], self.err, cntd,
                    H, R, k, NBLK, NPOS, svc.C, Rd, fz["maxrd"], S, Rd * H,
                    KP=KP, NP=NP, HAS_V=Rv > 0, num_warps=nw)
            _launch(_fr_d2, (S,), *tabs, sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"],
                    dix, dhb, dtl, dnt, fz["dstart"], fz["dcnt"], fz["dorder"],
                    fz["post"], fz["ne_d"], fz["rec_d"], fz["usgd"],
                    td, dpage, dnpg, page, npg, drec, dcnt, cntd, selw, rowsel, self.err, self.stats, sb, cis,
                    H, B, R, k, NBLK, NPOS, svc.C, svc.X, svc.R_ver, max(Rv, 1), Rd, fz["maxrv"], fz["maxrd"], nd,
                    caps[1], Rd * H, S, WB, N_TAIL, P, svc.bs, rows_s, rows_c,
                    KP=KP, NP=NP, PP=PP, TP=TP, RHP=RHP, SP=SP, ZP=ZP, BSP=max(_pow2(svc.bs), 2), HAS_V=Rv > 0,
                    BIAS=bias and not fallback, PLANT=self.plant, num_warps=nw)
            self.launches += 2
            ddec = DraftDecision(Rd, H, k, {f: td[i] for i, f in enumerate(FIELDS)}, dpage, dnpg, cntd)
            if fallback:                                              # torch ops: reported, not the timed design
                self.bias_fallbacks += 1
                bbd, hhd = self._bhd(fz, k)
                draft_bias_fill(svc.store_b[l], svc.cis[l], ddec, bbd, hhd, svc.bs)
            rec.update(drec=drec, dcnt=dcnt)
        rec.update(sel=selw, rowsel=rowsel)
        self.layers += 1
        step = dict(l=l, vrows=[rows[i] for i in tk["vrows"]], vidx=tk["vrows"], vdec=vdec,
                    drows=[rows[i] for i in tk["drows"]], ddec=ddec, rec=rec, caps=caps)
        if ref is not None:
            self._compare(where, l, step, page, npg, sb_before, ref)
        return step, page, npg

    # ----------------------------------------------------------------------------------------------- check mode
    @staticmethod
    def _cmp(out, where, name, a, b):
        if a is None and b is None:
            return
        if a is None or b is None or tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
            out.append(where + (name,))

    def _cmp_dec(self, where, kind, d_f, d_r):
        """One decision object (verify or draft) against the torch path's: every field, the page table, the page
        counts, the counters (verify: the load ids and the entries' (b, h) too)."""
        out = []
        if (d_f is None) != (d_r is None):
            return [where + (kind + ":presence",)]
        if d_f is None:
            return out
        for f in (FIELDS_V if kind == "v" else FIELDS):
            self._cmp(out, where, "%s.t.%s" % (kind, f), d_f.t[f], d_r.t[f])
        self._cmp(out, where, kind + ".page", d_f.page, d_r.page)
        self._cmp(out, where, kind + ".num_pages", d_f.num_pages, d_r.num_pages)
        self._cmp(out, where, kind + ".counters", d_f.counters, d_r.counters)
        if kind == "v":
            self._cmp(out, where, "v.load_ids", d_f.load_ids, d_r.load_ids)
            self._cmp(out, where, "v.bh.b", d_f.bh[0], d_r.bh[0])
            self._cmp(out, where, "v.bh.h", d_f.bh[1], d_r.bh[1])
        return out

    def _trace(self, where):
        """check: the stats and the err flag after a layer (the L34 preconditions per layer)."""
        self.trace.append((int(where[0]), int(where[1]), self.stats.to("cpu").tolist(), int(self.err)))

    def _wait_gather_traced(self, l, vdec, where):
        """check: S-C's device wait and gather; a raise (a wrong decision the CPU stand-in refuses) still leaves the
        layer's trace entry, after its verify differences."""
        try:
            self.svc.dev.wait(l, vdec)
            self.svc.dev.gather(l, vdec)
        except Exception:
            self._trace(where)
            raise

    def _compare(self, where, l, step, page, npg, sb_before, ref):
        """The rest of the torch path on the same inputs (the draft decisions, C's sparse record, the plan, S-B's bias
        fill on a copy), every output compared exactly; differences go to self.diffs as (tick, layer, output). The
        verify decisions were compared before the device wait (_cmp_dec). Reads the device."""
        svc = self.svc
        out = self._cmp_dec(where, "d", step["ddec"], ref["ddec"])

        def cmp(name, a, b):
            self._cmp(out, where, name, a, b)
        rf, rr = step["rec"], ref["rec"]
        if sorted(rf) != sorted(rr):
            out.append(where + ("rec:keys %s / %s" % (sorted(rf), sorted(rr)),))
        for name in sorted(set(rf) & set(rr)):
            cmp("rec." + name, rf[name], rr[name])
        cmp("plan.page_table", page, ref["page"])
        cmp("plan.num_pages", npg, ref["npg"])
        cmp("caps", torch.tensor(step["caps"]), torch.tensor(ref["caps"]))
        if sb_before is not None and ref["ddec"] is not None:
            args = ref["args"]
            draft_bias_fill(sb_before, svc.cis[l], ref["ddec"], args[7], args[8], svc.bs)
            cmp("store_b", svc.store_b[l], sb_before)
        self.diffs += out
        self.compared += 1
        self._trace(where)

    def report(self):
        """REPORT (one device read; after a run): the mode, the counts, the stats and the err flag."""
        return dict(mode="check" if self.check else "fused", plant=self.plant, launches=self.launches,
                    layers=self.layers, launches_per_layer=self.launches / max(1, self.layers),
                    bias_fallbacks=self.bias_fallbacks, compared=self.compared, diffs=len(self.diffs),
                    first_diffs=[list(d) for d in self.diffs[:5]],
                    stats=dict(zip(STATS, self.stats.to("cpu").tolist())), err=int(self.err))


def torch_reference(svc, l, tk, sel):
    """The torch path's outputs for layer l on the tick's inputs (ResidencySC.resolve's decision part and
    ResidencySparse._layer_record's record), with NO side effect: no device wait or gather, no bias fill (the caller
    applies it to a copy), the B2 record (pin_records False: the pin table is not touched)."""
    m = svc.dmirror.layer(l)
    R = len(tk["vrows"]) + len(tk["drows"])
    H, dev = svc.H, svc.device
    k = sel.shape[2]
    page = torch.full((R, H, k + N_TAIL), -1, dtype=torch.int64, device=dev)
    npg = torch.zeros((R, H), dtype=torch.int64, device=dev)
    vdec = ddec = None
    if tk["v"] is not None:
        rb, hb, tl_, nt, ix = tk["v"]
        others = []
        if tk["d"] is not None:
            rbd, hbd, _, _, ixd = tk["d"]
            others.append((rbd, hbd, sel[:, ixd]))
        vdec = device_verify(m, rb, hb, tl_, nt, sel[:, ix], svc.C, svc.X, svc.sc_mut, others=others,
                             inflight=svc.policy_e_inflight)
        page[ix] = vdec.page
        npg[ix] = vdec.num_pages
    if tk["d"] is not None:
        rb, hb, tl_, nt, ix = tk["d"]
        ddec = device_draft(m, rb, hb, tl_, nt, sel[:, ix], svc.C, svc.X, svc.R_ver, svc.sb_mut)
        page[ix] = ddec.page
        npg[ix] = ddec.num_pages
    step = dict(l=l, vdec=vdec, ddec=ddec)
    args, kw = svc._record_inputs(l, step, tk)
    rec = sparse_layer_record(*args, **kw)
    return dict(vdec=vdec, ddec=ddec, rec=rec, page=page.to(torch.int32), npg=npg.to(torch.int32), args=args,
                caps=(args[11], args[12]))
