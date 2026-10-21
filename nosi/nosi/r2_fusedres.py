"""R2 step 7, piece 5, (B2): the FUSED RESIDENCY RESOLVE (docs/R2_FUSEDRES_PLAN.md). SOURCE ONLY, NOT RUN.

WHAT IT REPLACES. OwnResidency (r2_own2.py) inherits S-C's per-layer device path: DeviceMirror.layer (two segmented
sorts and scatters), device_verify / device_draft (stable sorts, rank_in_group with cummax, index, index_put_) and
C's sparse record (sparse_layer_record, _compact, _pack_bits, draft_bias_fill). On the GPU that is about 305 small
kernels per layer, 9,774 per draft tick at B 64 and 37.62 ms of kernel time (analysis/bound_4_2/other_res_r1,
"res:resolve G_d"). This module makes the SAME decisions with THREE Triton launches per row kind present in the
layer (fusedres2): SIX with verify and draft rows (V1, SCAN, V2, D1, SCAN, D2), THREE with only draft rows (D1, SCAN,
D2) or only verify rows. No host read, no host-to-device copy in the per-layer path. OwnResidency(resolve="fused") uses it; resolve="check" uses it and also runs the torch path on the
same inputs and compares every output (tests, the GPU equality gate); resolve="torch" (the default) is unchanged.

THE LAYOUT. One Triton program per stream s = b * H + h (grid S = B * H). A program reads its stream's NPOS = C + X
slots from the device mirror (dmirror.code / block / gen / cur / prot / stamp [l], the table B2 copies at each tick's
first resolve) into registers, and loops over the rows of its batch slot b in entry order (the per-tick CSR: for each
b the verify rows and the draft rows, ascending). A block -> slot lookup is a compare of the row's k selected blocks
against the stream's NPOS slot blocks (a (k, NPOS) register tile; K3: one resident slot per block). So the per-block
tables of DeviceMirror.layer (B, H, nblk) are never built.

THE ONE GLOBAL COUPLING is K5's shared extras: "the first (X - borrowed - reserve) extra-needing first misses of the
layer, in K3X entry order (row, head, ascending block)", and borrowed sums every stream's usage. Phase 1 (V1 / D1)
writes each (row, head)'s count of extra-needing first misses and each stream's usage; SCAN (one program) writes the
exclusive prefix per (row, head), the layer's budget and the record total; phase 2 (V2 / D2) loads ONE base per
(row, head) and adds the in-row prefix (fusedres2, review M4: 0e28cf6 had every program read the whole count vectors,
an O(B^2) term at B 287). Phase 2 re-checks its own counts against phase 1's (err bit 16, review m5). The sparse
record's compaction uses the same scheme (the record
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
      torch.sort breaks ties in an unspecified way; this module breaks them by slot index. Ties are NOT flagged (only
      the range is, bit 8); both picks are legal; a tie shows as a check-mode difference.
  A3  a K5 grant always finds a FREE slot (grants <= NPOS - usage, by K5's arithmetic). S-C would index the next
      stream's slot 0 (and the tail bias rows) if it did not; this module writes nothing for such a take, may set
      read bit NPOS / lane NPOS in its scratch, and raises the err flag: the outputs differ only on a voided tick.

THE BIAS ROWS: D2 copies them bit for bit when store_b[l] and cis[l] have one 2- or 4-byte dtype (BIAS 1), else casts
cis to store_b's dtype in the kernel (BIAS 2, counted in bias_casts; at p5b2 both are the tick dtype: r2_tick.py
from_r2 makes the twins and store_bias with src.ring.dtype). Non-contiguous or non-(B, rows, H) tables are refused.

THE ERR FLAG (self.err, int32, watched by the service: a set flag voids the tick with CapacityRefusal at end_tick):
bit 1 A1 resident, bit 2 A1 in flight, bit 4 A3, bit 8 A2 (stamp range), bit 16 the phase-1 / phase-2 self-check.
self.err is zeroed at each tick's first layer (after the previous end_tick read it); self.err_any keeps every tick's
bits (report()["err"]).

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
STATS = ("v_evict", "pe_flip_effective", "v_grant", "d_grant", "d_refused", "v_over", "v_dup", "v_evict_infl",
         "v_dup_over", "d_dup", "sel_dup")      # v_dup_over: a dup behind an overflowed first miss; sel_dup: a row
#                                                 selecting one block twice (entries beyond the first), both kinds
GATE_STATS = ("v_evict", "pe_flip_effective", "v_grant", "d_grant", "d_refused")   # FR-g0's coverage (M1)
ERR_BITS = {1: "A1: a block has two resident slots in a stream", 2: "A1: a block has two owner slots in flight",
            4: "A3: a K5 grant found no FREE slot", 8: "A2: a Policy E candidate stamp outside [0, 2^47)",
            16: "self-check: phase 2's classification count differs from phase 1's (the global prefix is wrong)"}
SCAN_BLOCK = 1024
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
        s64 = tl.load(sel_ptr + (h * R + tr) * K + kk, mask=mk, other=-1)
        v0 = mk & (s64 >= 0) & (s64 < hbr) & (s64 < NBLK)                       # validated in int64 (m8)
        s = tl.where(v0, s64, -1).to(tl.int32)                                  # only valid entries are compared
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
    # SCAN (one program, M4 of review-fusedres-0e28cf6): the exclusive prefixes of the per-(row, head) counts of a
    # phase, the layer's K5 budget and the record total. Phase 2 then loads ONE base per (row, head): no O(B^2) reads.
    # =================================================================================================================
    @triton.jit(do_not_specialize=["RH", "S", "C", "X", "RES"])
    def _fr_scan(ne_ptr, rec_ptr, usg_ptr, bne_ptr, brec_ptr, misc_ptr, RH, S, C, X, RES, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        cn = tl.sum(tl.zeros([2], tl.int32), axis=0)
        cr = cn
        for c0 in range(0, RH, BLOCK):
            m = c0 + offs < RH
            n = tl.load(ne_ptr + c0 + offs, mask=m, other=0)
            r = tl.load(rec_ptr + c0 + offs, mask=m, other=0)
            tl.store(bne_ptr + c0 + offs, cn + tl.cumsum(n, axis=0) - n, mask=m)
            tl.store(brec_ptr + c0 + offs, cr + tl.cumsum(r, axis=0) - r, mask=m)
            cn = cn + tl.sum(n, axis=0)
            cr = cr + tl.sum(r, axis=0)
        bor = cn * 0
        for c0 in range(0, S, BLOCK):
            m = c0 + offs < S
            u = tl.load(usg_ptr + c0 + offs, mask=m, other=0)
            bor = bor + tl.sum(tl.where(m, tl.maximum(u - C, 0), 0), axis=0)
        budget = tl.maximum(X - bor - RES, 0)
        q4 = tl.arange(0, 4)
        tl.store(misc_ptr + q4, tl.where(q4 == 0, budget, tl.where(q4 == 1, cr, cn)), mask=q4 < 3)

    # =================================================================================================================
    # V1:the verify rows' classification (counts for the global K5 prefix and the record compaction)
    # =================================================================================================================
    @triton.jit(do_not_specialize=["H", "R", "K", "NBLK", "NPOS", "C", "Rv", "MAXRV", "NLOOK"])   # m4: no re-compile
    def _fr_v1(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               ne_ptr, rec_ptr, usg_ptr, err_ptr, cntv_ptr, look_ptr,
               H, R, K, NBLK, NPOS, C, Rv, MAXRV, NLOOK,
               KP: tl.constexpr, NP: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        free_local = tl.maximum(C - usage, 0)
        _put1(usg_ptr + pid, usage, pid >= 0)
        lk = tl.arange(0, 256)                                                    # program 0 zeroes V2's sums
        for c0 in range(0, NLOOK, 256):
            tl.store(look_ptr + c0 + lk, tl.zeros([256], tl.int64), mask=(c0 + lk < NLOOK) & (pid == 0))
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
    @triton.jit(do_not_specialize=["H", "B", "R", "K", "NBLK", "NPOS", "C", "X", "Rv", "Rd", "MAXRV", "MAXRD", "NV",
                                   "CAPV", "S", "WB", "WP", "T", "P"])
    def _fr_v2(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vtl_ptr, vnt_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               ne_ptr, rec_ptr, bne_ptr, brec_ptr, misc_ptr,
               tv_ptr, vpage_ptr, vnpg_ptr, ppage_ptr, pnpg_ptr, ids_ptr, vrec_ptr, vrcnt_ptr, cntv_ptr, look_ptr,
               vread_ptr, post_ptr, usg2_ptr, selw_ptr, rowsel_ptr, err_ptr, stats_ptr,
               H, B, R, K, NBLK, NPOS, C, X, Rv, Rd, MAXRV, MAXRD, NV, CAPV, S, WB, WP, T, P,
               KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr,
               ZP: tl.constexpr, PE_INFL: tl.constexpr, LAST: tl.constexpr, PLANT: tl.constexpr,
               NSTAT: tl.constexpr):
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
        # K5's shared extras for the verifier (X - borrowed, the reserve R_ver included) and the record total: SCAN's
        budget = tl.load(misc_ptr)
        totrec = tl.load(misc_ptr + 1)
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
        st16 = tl.arange(0, 16)
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
            g = tl.load(bne_ptr + idx) + tl.cumsum(nei, axis=0) - nei
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
            rpos = tl.load(brec_ptr + idx) + tl.cumsum(rmi, axis=0) - rmi
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
            n_dov = tl.sum((dup & (grp < 0)).to(tl.int64), axis=0)                   # a dup behind an overflow
            n_sdup = (tl.sum(v0.to(tl.int32), axis=0) - tl.sum(valid.to(tl.int32), axis=0)).to(tl.int64)
            sv = tl.where(st16 == 0, n_ev, tl.where(st16 == 1, n_fl, tl.where(
                st16 == 2, tl.sum(gi.to(tl.int64), axis=0), tl.where(
                    st16 == 5, tl.sum(over.to(tl.int64), axis=0), tl.where(
                        st16 == 6, tl.sum(dup.to(tl.int64), axis=0), tl.where(
                            st16 == 7, tl.sum(ev_infl.to(tl.int64), axis=0), tl.where(
                                st16 == 8, n_dov, tl.where(st16 == 10, n_sdup, 0))))))))
            tl.atomic_add(stats_ptr + st16, sv, mask=(st16 < NSTAT) & act)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            e4 = tl.where(act & (tl.max((granted & (fs < 0)).to(tl.int32), axis=0) > 0), 4, 0)
            # m5: this copy of the classification must agree with V1's counts (else the global prefix is wrong)
            e16 = tl.where(act & ((tl.sum(nei, axis=0) != tl.load(ne_ptr + idx))
                                  | (tl.sum(rmi, axis=0) != tl.load(rec_ptr + idx))), 16, 0)
            errb = errb | e1 | e2 | e4 | e16
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
    @triton.jit(do_not_specialize=["H", "R", "K", "NBLK", "NPOS", "C", "Rd", "MAXRD", "S", "NPAIRS"])
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
    @triton.jit(do_not_specialize=["H", "B", "R", "K", "NBLK", "NPOS", "C", "Rv", "Rd", "MAXRV", "MAXRD", "ND",
                                   "CAPD", "S", "WB", "T", "P", "BS", "ROWS_S", "ROWS_C"])
    def _fr_d2(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               dix_ptr, dhb_ptr, dtl_ptr, dnt_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               post_ptr, ne_ptr, rec_ptr, bne_ptr, brec_ptr, misc_ptr, usgd_ptr,
               td_ptr, dpage_ptr, dnpg_ptr, ppage_ptr, pnpg_ptr, drec_ptr, dcnt_out_ptr, cntd_ptr,
               selw_ptr, rowsel_ptr, err_ptr, stats_ptr, sb_ptr, cis_ptr,
               H, B, R, K, NBLK, NPOS, C, Rv, Rd, MAXRV, MAXRD, ND, CAPD, S, WB, T, P,
               BS, ROWS_S, ROWS_C,
               KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr,
               ZP: tl.constexpr, BSP: tl.constexpr, HAS_V: tl.constexpr, BIAS: tl.constexpr,
               PLANT: tl.constexpr, NSTAT: tl.constexpr):
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
        budget = tl.load(misc_ptr)                                               # SCAN: X - borrowed - R_ver
        totrec = tl.load(misc_ptr + 1)
        start = tl.load(dstart_ptr + b)
        cnt = tl.load(dcnt_ptr + b)
        carry_f = tl.sum(tl.zeros([2], tl.int32), axis=0)
        carry_g = carry_f
        errb = carry_f
        kk = tl.arange(0, KP)
        mk = kk < K
        cc = tl.arange(0, PP)
        tt = tl.arange(0, TP)
        st16 = tl.arange(0, 16)
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
            g = tl.load(bne_ptr + idx) + tl.cumsum(nei, axis=0) - nei
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
            rpos = tl.load(brec_ptr + idx) + tl.cumsum(vi, axis=0) - vi
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
            n_sdup = (tl.sum(v0.to(tl.int32), axis=0) - nvalid).to(tl.int64)
            sv = tl.where(st16 == 3, tl.sum(gi.to(tl.int64), axis=0), tl.where(st16 == 4, nref, tl.where(
                st16 == 9, tl.sum(dup.to(tl.int64), axis=0), tl.where(st16 == 10, n_sdup, 0))))
            tl.atomic_add(stats_ptr + st16, sv, mask=(st16 < NSTAT) & act)
            # the bias rows of the granted takes (S-B's draft_bias_fill): BIAS 1 bit copies (one dtype), 2 a cast
            # (cis to store_b's dtype: torch's .to on the GPU rounds to nearest even, as Triton's cast does)
            if BIAS != 0:
                gm = (granted & act & (take >= 0) & (take < NPOS))[:, None] & (bb[None, :] < BS)
                b64 = b.to(tl.int64)
                src = cis_ptr + ((b64 * ROWS_C + blk[:, None].to(tl.int64) * BS + bb[None, :]) * H + h)
                dst = sb_ptr + ((b64 * ROWS_S + take[:, None].to(tl.int64) * BS + bb[None, :]) * H + h)
                x = tl.load(src, mask=gm, other=0)
                if BIAS == 2:
                    x = x.to(sb_ptr.dtype.element_ty)
                tl.store(dst, x, mask=gm)
            e1 = tl.where(act & (tl.max(nres, axis=0) > 1), 1, 0)
            e2 = tl.where(act & (tl.max(nfly, axis=0) > 1), 2, 0)
            e4 = tl.where(act & (tl.max((granted & (fs < 0)).to(tl.int32), axis=0) > 0), 4, 0)
            e16 = tl.where(act & ((tl.sum(nei, axis=0) != tl.load(ne_ptr + idx))
                                  | (nvalid != tl.load(rec_ptr + idx))), 16, 0)            # m5: D1 == D2
            errb = errb | e1 | e2 | e4 | e16
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
        self.err = torch.zeros((), dtype=torch.int32, device=dev)          # this tick's (zeroed at each tick's start)
        self.err_any = torch.zeros((), dtype=torch.int32, device=dev)      # m1: every tick's, OR-ed (sticky REPORT)
        self.stats = torch.zeros((len(STATS),), dtype=torch.int64, device=dev)
        svc.watch_flag("fused_res_err", self.err)
        self.launches = 0                                  # Triton launches (all layers, all ticks)
        self.layers = 0                                    # fused layers resolved
        self.bias_casts = 0                                # layers whose bias rows needed a dtype cast (BIAS 2)
        self.sel_conversions = 0                           # m3: layers whose selection was copied (dtype / layout)
        self.max_rows_per_b = [0, 0]                       # (verify, draft) rows of one batch slot, the run's max
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
        self.err_any.bitwise_or_(self.err)                  # m1: keep every tick's bits (the REPORT's err)
        self.err.zero_()                                    # per tick: the previous end_tick has read it
        vs, vc, vo, mv = group(tk["vrows"])
        ds, dc, do, md = group(tk["drows"])
        self.max_rows_per_b = [max(self.max_rows_per_b[0], mv), max(self.max_rows_per_b[1], md)]
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
        kid = tk["extra"][:len(rows)]
        # m2 (L28): every tensor the kernels address by flat offsets is a contiguous int64 tensor, or refuse
        dm = svc.dmirror
        named = [("dmirror." + n, getattr(dm, n)) for n in ("code", "block", "gen", "cur", "prot", "stamp")]
        named += [("kid", kid)] + [("%s[%d]" % (kind, i), t) for kind, tup in (("v", v), ("d", dd))
                                   for i, t in enumerate(tup)]
        named += [("rows." + n, t) for n, t in zip(("vstart", "vcnt", "vorder", "dstart", "dcnt", "dorder"), parts)]
        bad = [n for n, t in named if t.dtype != torch.int64 or not t.is_contiguous() or t.device.type != dev.type]
        if bad:
            raise ValueError("resolve='fused': not a contiguous int64 tensor on %s: %s" % (dev, bad))
        i32 = dict(dtype=torch.int32, device=dev)

        def scan_bufs(n):
            return dict(ne=torch.empty(n, **i32), rec=torch.empty(n, **i32), bne=torch.empty(n, **i32),
                        brec=torch.empty(n, **i32), misc=torch.empty(4, **i32))
        return dict(vstart=parts[0], vcnt=parts[1], vorder=parts[2], dstart=parts[3], dcnt=parts[4],
                    dorder=parts[5], maxrv=mv, maxrd=md, Rv=Rv, Rd=Rd, v=v, d=dd, kid=kid,
                    sv=scan_bufs(max(Rv * H, 1)), sd=scan_bufs(max(Rd * H, 1)),
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
            sel = sel.to(torch.int64).contiguous()                    # m3: a torch copy, counted (FR-g2 gates 0)
            self.sel_conversions += 1
        Rv, Rd = fz["Rv"], fz["Rd"]
        NP = fz["NP"]
        KP = max(_pow2(k), 2)
        PP = max(_pow2(P), 2)
        TP = max(_pow2(N_TAIL), 2)
        ZP = max(_pow2(NBLK), 64)
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
        sv, sd = fz["sv"], fz["sd"]
        if Rv:
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
                    sv["ne"], sv["rec"], fz["usg"], self.err, cntv, vlook,
                    H, R, k, NBLK, NPOS, svc.C, Rv, fz["maxrv"], Rv * 4,
                    KP=KP, NP=NP, num_warps=nw)
            _launch(_fr_scan, (1,), sv["ne"], sv["rec"], fz["usg"], sv["bne"], sv["brec"], sv["misc"],
                    Rv * H, S, svc.C, svc.X, 0, BLOCK=SCAN_BLOCK, num_warps=4)
            _launch(_fr_v2, (S,), *tabs, sel, kid, vix, vhb, vtl, vnt, fz["vstart"], fz["vcnt"], fz["vorder"],
                    dix, dhb, fz["dstart"], fz["dcnt"], fz["dorder"],
                    sv["ne"], sv["rec"], sv["bne"], sv["brec"], sv["misc"],
                    tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, fz["post"], fz["usg2"], selw,
                    rowsel, self.err, self.stats,
                    H, B, R, k, NBLK, NPOS, svc.C, svc.X, Rv, max(Rd, 1), fz["maxrv"], fz["maxrd"], nv, caps[0],
                    S, WB, WP, N_TAIL, P,
                    KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, PE_INFL=self.pe_inflight,
                    LAST=Rd == 0, PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
            self.launches += 3
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
            td = torch.empty((len(FIELDS), nd), **i64)
            dpage = torch.empty((Rd, H, P), **i64)
            dnpg = torch.empty((Rd, H), **i64)
            drec = torch.empty((len(DREC), caps[1]), **i64)
            dcnt = torch.empty((1,), **i64)
            cntd = torch.empty((3,), **i64)
            bias = 0
            sb = cis = sel
            rows_s = rows_c = 0
            if svc.store_b is not None and svc.cis is not None:
                sb_t, cis_t = svc.store_b[l], svc.cis[l]
                if (not sb_t.is_contiguous() or not cis_t.is_contiguous() or sb_t.dim() != 3 or cis_t.dim() != 3
                        or not sb_t.dtype.is_floating_point or not cis_t.dtype.is_floating_point):
                    raise ValueError("resolve='fused': the bias rows need contiguous (B, rows, H) float store_b and "
                                     "cis tables (got %s %s / %s %s)" % (sb_t.dtype, tuple(sb_t.shape), cis_t.dtype,
                                                                         tuple(cis_t.shape)))
                if self.check:
                    sb_before = sb_t.clone()
                if sb_t.dtype == cis_t.dtype and sb_t.element_size() in (2, 4):
                    bias, sb, cis = 1, _bits(sb_t), _bits(cis_t)      # bit copies
                else:
                    bias, sb, cis = 2, sb_t, cis_t                    # m9: the cast in the kernel (counted)
                    self.bias_casts += 1
                rows_s, rows_c = sb_t.shape[1], cis_t.shape[1]
            _launch(_fr_d1, (S,), *tabs, sel, kid, dix, dhb, fz["dstart"], fz["dcnt"], fz["dorder"],
                    fz["post"], fz["usg2"], sd["ne"], sd["rec"], fz["usgd"], self.err, cntd,
                    H, R, k, NBLK, NPOS, svc.C, Rd, fz["maxrd"], S, Rd * H,
                    KP=KP, NP=NP, HAS_V=Rv > 0, num_warps=nw)
            _launch(_fr_scan, (1,), sd["ne"], sd["rec"], fz["usgd"], sd["bne"], sd["brec"], sd["misc"],
                    Rd * H, S, svc.C, svc.X, svc.R_ver, BLOCK=SCAN_BLOCK, num_warps=4)
            _launch(_fr_d2, (S,), *tabs, sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"],
                    dix, dhb, dtl, dnt, fz["dstart"], fz["dcnt"], fz["dorder"],
                    fz["post"], sd["ne"], sd["rec"], sd["bne"], sd["brec"], sd["misc"], fz["usgd"],
                    td, dpage, dnpg, page, npg, drec, dcnt, cntd, selw, rowsel, self.err, self.stats, sb, cis,
                    H, B, R, k, NBLK, NPOS, svc.C, max(Rv, 1), Rd, fz["maxrv"], fz["maxrd"], nd,
                    caps[1], S, WB, N_TAIL, P, svc.bs, rows_s, rows_c,
                    KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, BSP=max(_pow2(svc.bs), 2), HAS_V=Rv > 0,
                    BIAS=bias, PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
            self.launches += 3
            ddec = DraftDecision(Rd, H, k, {f: td[i] for i, f in enumerate(FIELDS)}, dpage, dnpg, cntd)
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
        """REPORT (device reads; after a run): the mode, the counts, the stats, the err bits of every tick (err_any:
        m1), the coverage gate of FR-g0 (M1: a GATE_STATS counter at 0 is "NOT GATED", never a pass), and in check
        mode the first difference and, per stat, the first (tick, layer) where it became > 0 (FR-g0p, M2)."""
        stats = dict(zip(STATS, self.stats.to("cpu").tolist()))
        err_any = int(self.err_any) | int(self.err)
        first_eff = {}
        for (t, l, st, _) in self.trace:
            for name, v in zip(STATS, st):
                if v > 0 and name not in first_eff:
                    first_eff[name] = [t, l]
        return dict(mode="check" if self.check else "fused", plant=self.plant, launches=self.launches,
                    layers=self.layers, launches_per_layer=self.launches / max(1, self.layers),
                    bias_casts=self.bias_casts, sel_conversions=self.sel_conversions,
                    max_rows_per_b=list(self.max_rows_per_b), compared=self.compared, diffs=len(self.diffs),
                    first_diffs=[list(d) for d in self.diffs[:5]],
                    first_diff_at=list(self.diffs[0][:2]) if self.diffs else None, first_effective=first_eff,
                    stats=stats, gate={s: ("gated" if stats[s] > 0 else "NOT GATED") for s in GATE_STATS},
                    err=err_any, err_bits=[m for b, m in sorted(ERR_BITS.items()) if err_any & b],
                    compiled=compiled_variants())


def compiled_variants():
    """REPORT (m4): the compiled variants per kernel of this module (Triton's JIT cache per device; None where the
    installed Triton keeps no such cache, e.g. the interpreter). A variant compiled inside a timed span inflates it."""
    out = {}
    if triton is None:
        return out
    for name in ("_fr_v1", "_fr_v2", "_fr_d1", "_fr_d2", "_fr_scan"):
        try:
            cache = globals()[name].cache
            out[name] = sum(len(v) for v in cache.values())
        except Exception:                                  # noqa: BLE001 - a REPORT; the cache layout is version-bound
            out[name] = None
    return out


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
