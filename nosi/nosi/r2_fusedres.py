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

V3 ([fr3], FusedResolver(kernel="v3"); the default "v2" is unchanged). The v2 kernels hold NP-wide slot vectors and
(KP x NP) compare tiles per program, and _fr_v2 sorts NP keys with tl.sort. At NP 2048 and 8 warps that spills (about
3.0 GB of driver local memory, analysis/evalfast/SCORING_b192.md section 6). V3 makes the SAME decisions with bounded
tiles. No v3 kernel has an NP-wide vector, a (KP x NP) tile or tl.sort (test FR3-S checks the source):
  PREP  (new, one launch per layer, grid S) per stream: the BLOCK TABLES (ZP entries each: the highest resident slot
        and its count, the highest current-owner slot in flight and its count; built from (ZC x TC) compare tiles, so a
        table entry equals v2's _lookup for every block), the FREE LIST (the FREE slots in slot order: the q-th FREE
        slot is one load), the usage, and with verify rows Policy E's key per slot (v2's insel / class / stamp / slot
        key, per (CH x KP) chunk) and the candidate count; it initializes V2's per-slot scratch (post) and the tables
        that V2 fills (took / gone / gone in flight).
  V1 / D1 / D2  as v2, with table loads instead of compare tiles (D1 also builds the FREE list after the verify rows).
  V2    as v2, with table and FREE-list loads. Policy E, EXACT: a row's victims are the consecutive ranks carry_r ..
        carry_r + nev - 1, so V2 takes them in key order by one bounded pass over the stream's keys each ("the next key
        above the last one taken": keys are distinct, the slot is in the low bits). The cost is O(evictions x NP), not
        a sort. The per-row state (v2's newblk / evold / evinfl / free2 vectors) becomes scatters to distinct slots and
        blocks, with a CTA barrier (tl.debug_barrier) before a later row or the end pass reads them.
  Launches per layer: 4 draft-only or verify-only (PREP + 3), 7 with both (PREP, V1, SCAN, V2, D1, SCAN, D2).
  Determinism: no decision uses an atomic (the scatters write distinct addresses or one equal value), so v3 has no
  FAST / EXACT split. Equality with v2 (and so with the torch path) holds on every tick whose err flag is 0; on a voided
  tick (A3, err bit 4: a grant without a FREE slot) the scratch lane NPOS may differ from v2's (both are voided).
  Plant 3 (v3 only): a verify dup ignores the slot an earlier row took (the took table read); effective where
  v_dup - v_dup_over grows.

Imports torch (and triton when it is installed). No `from __future__ import annotations`: the kernels' constexpr
annotations must be objects.
"""
import sys

import torch

try:                                                   # triton is needed only to launch the kernels; the reference
    import triton                                      # path and the checks work without it
    import triton.language as tl
except Exception:                                      # pragma: no cover - not installed, or a broken install
    triton = tl = None

from .kstep.errors import OwnershipViolation
from . import r2_xpool as XP                                 # [integ] the shared X pool's binder and flat bias fill
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
SCAN_BLOCK = 1024                      # the lanes of every SCAN chunk ([stack2] always: no per-tick variant)
NSTAT_LANES = 16                       # the stats atomics' lanes; the stats buffer holds all of them (fusedres2b)
# fusedres2b (job 2191246: glibc "corrupted size vs. prev_size" in the interpreter child): every buffer a kernel
# writes is allocated with LANE_PAD spare elements after its logical end, so a masked lane past the end (SCAN's chunk,
# the vlook zeroing chunk, KP / PP / NP / ZP tile lanes) addresses allocated memory in the interpreter too. With
# guard > 0 (tests) the spare elements hold GUARD_SENTINEL and are checked after EVERY launch (a direct OOB detector
# that names the kernel and the buffer).
LANE_PAD = 2048
GUARD_SENTINEL = -95443717
# [stack2] Policy E's sort key is class << (47 + SB) | stamp << SB | slot, SB = slot_bits(NP) (V2's constexpr). SB is 9
# for NP <= 512: the key and the mask are the 23b8a38 ones, bit for bit. SB grows with NP above 512 (10 at NP 1024, 11
# at 2048, 12 at 4096). The largest legal key is 2 << (47 + SB) (class 2, stamp 0), < 2^63 for SB <= 14; MAX_NPOS 4096
# keeps SB <= 12 (key < 2^61). A candidate stamp outside [0, 2^47) sets err bit 8 (the tick is voided); its key can
# overflow int64, at every SB, as before. stack1a128 (job 386, C 96 + X 896 = 992) failed on the old 9-bit limit.
MAX_NPOS = 4096
STAMP_BITS = 47                                        # A2: a candidate stamp is in [0, 2^47)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# [stack2] NO DATA-DEPENDENT COMPILE (job 386 stack1a: 17 variants of these kernels were first compiled INSIDE the
# stack arm's timed round, Triton cache mtimes 20:22:59 .. 20:24:35 local; one _fr_v2 compile is about 31 s, the
# 42.4 s tick). Triton keys a compiled variant on every constexpr, on every int argument (== 1, % 16) and on every
# pointer's 16-byte alignment, unless the argument is in do_not_specialize. 23b8a38 listed the ints only, so these
# per-tick facts made new variants: the row-list slices' alignment (dix / dstart / dcnt / dorder follow len(vrows);
# kid follows len(rows)), LZ = pow2(Rv * 4) and SCAN's BLOCK = pow2(max(Rv H, S)). Now: the row-list pointers and the
# selection are not specialized (they are gathered, never vector-loaded), LZ and BLOCK are constants (masked lanes:
# the same values), and the variants left (V2's LAST, D1 / D2's HAS_V) are compiled by warm_up before the first
# decision. DNS_OLD keeps 23b8a38's lists (the tests' negative control). No output changes.
_INTS = dict(
    _fr_scan=["RH", "S", "C", "X", "RES"],
    _fr_v1=["H", "R", "K", "NBLK", "NPOS", "C", "Rv", "MAXRV", "NLOOK"],
    _fr_v2=["H", "B", "R", "K", "NBLK", "NPOS", "C", "X", "Rv", "Rd", "MAXRV", "MAXRD", "NV", "CAPV", "S", "WB", "WP",
            "T", "P"],
    _fr_d1=["H", "R", "K", "NBLK", "NPOS", "C", "Rd", "MAXRD", "S", "NPAIRS"],
    _fr_d2=["H", "B", "R", "K", "NBLK", "NPOS", "C", "Rv", "Rd", "MAXRV", "MAXRD", "ND", "CAPD", "S", "WB", "T", "P",
            "BS", "ROWS_S", "ROWS_C"])
_ROWS_V = ["vix_ptr", "vhb_ptr", "vtl_ptr", "vnt_ptr", "vstart_ptr", "vcnt_ptr", "vorder_ptr"]
_ROWS_D = ["dix_ptr", "dhb_ptr", "dtl_ptr", "dnt_ptr", "dstart_ptr", "dcnt_ptr", "dorder_ptr"]
_SEL = ["sel_ptr", "kid_ptr"]
DNS_OLD = {k: list(v) for k, v in _INTS.items()}
DNS = dict(_fr_scan=_INTS["_fr_scan"],
           _fr_v1=_INTS["_fr_v1"] + _SEL + [p for p in _ROWS_V if p not in ("vtl_ptr", "vnt_ptr")],
           _fr_v2=_INTS["_fr_v2"] + _SEL + _ROWS_V + [p for p in _ROWS_D if p not in ("dtl_ptr", "dnt_ptr")],
           _fr_d1=_INTS["_fr_d1"] + _SEL + [p for p in _ROWS_D if p not in ("dtl_ptr", "dnt_ptr")],
           _fr_d2=_INTS["_fr_d2"] + _SEL + [p for p in _ROWS_V if p not in ("vtl_ptr", "vnt_ptr")] + _ROWS_D
           + ["sb_ptr", "cis_ptr"])
# [fr3] the v3 kernels: every int, the row lists and the selection are not specialized ([stack2]'s rule)
_INTS3 = dict(
    _fr3_prep=["H", "R", "K", "NBLK", "NPOS", "Rv", "Rd", "MAXRV", "MAXRD", "S"],
    _fr3_v1=["H", "R", "K", "NBLK", "C", "Rv", "MAXRV", "NLOOK"],
    _fr3_v2=["H", "B", "R", "K", "NBLK", "NPOS", "C", "Rv", "MAXRV", "NV", "CAPV", "S", "WB", "WP", "T", "P"],
    _fr3_d1=["H", "R", "K", "NBLK", "NPOS", "C", "Rd", "MAXRD", "S", "NPAIRS"],
    _fr3_d2=["H", "B", "R", "K", "NBLK", "NPOS", "C", "Rv", "Rd", "MAXRV", "MAXRD", "ND", "CAPD", "S", "WB", "T", "P",
             "BS", "ROWS_S", "ROWS_C"])
_RV5 = [p for p in _ROWS_V if p not in ("vtl_ptr", "vnt_ptr")]
_RD5 = [p for p in _ROWS_D if p not in ("dtl_ptr", "dnt_ptr")]
DNS_OLD.update({k: list(v) for k, v in _INTS3.items()})
DNS.update(_fr3_prep=_INTS3["_fr3_prep"] + _SEL + _RV5 + _RD5,
           _fr3_v1=_INTS3["_fr3_v1"] + _SEL + _RV5,
           _fr3_v2=_INTS3["_fr3_v2"] + _SEL + _ROWS_V,
           _fr3_d1=_INTS3["_fr3_d1"] + _SEL + _RD5,
           _fr3_d2=_INTS3["_fr3_d2"] + _SEL + _RV5 + _ROWS_D + ["sb_ptr", "cis_ptr"])
LOOK_LANES = 256                                      # V1's zeroing chunk (23b8a38: min(256, pow2(Rv * 4)))
WARM_MODES = ("auto", "on", "off")                     # auto: on for a CUDA service (the interpreter compiles nothing)
_warm_depth = [0]                                      # > 0 while warm_up launches (tests tell its launches apart)
# [oomfix] the first launch of _fr_v2 (NP 2048) grows the driver's local-memory pool (register spills, for every thread
# the device can hold). The driver cannot take the blocks that torch's caching allocator holds, and torch does not free
# them on a DRIVER failure (analysis/evalfast/DIAG_fused_check_oom.md). One empty_cache per process, before that launch.
_LMEM_PREP = [False]
# [fr3] the kernel variants (module docstring, V3). FusedResolver(kernel=None) reads DEFAULT_KERNEL at construction:
# the CPU tests switch it per child process (test_r2_fusedres_v3.py); production passes kernel explicitly.
KERNELS = ("v2", "v3")
DEFAULT_KERNEL = "v2"
PLANTS3 = {3: "v3: a verify dup ignores the slot an earlier row took for its block (the took table read)"}
FR3_ZC = 64                                            # v3: the block chunk (rowsel / sel words), a multiple of 32, >= 64
FR3_TC = 64                                            # v3: the slot chunk of the tables' build: (ZC x TC) tiles
FR3_TILE = 4096                                        # v3: the largest (CH x KP) tile


def fr3_chunk(KP: int, NP: int) -> int:
    """[fr3] v3's slot chunk CH: a power of 2, >= 64 (two 32-bit words, no size-1 tile dimension), <= NP, and
    CH x KP <= FR3_TILE where KP <= 64."""
    return min(int(NP), max(64, FR3_TILE // int(KP)))


def _jit_dns(name: str):
    """[stack2] triton.jit with DNS[name] as do_not_specialize (and as do_not_specialize_on_alignment where the
    installed Triton has that argument: Triton 3.2 splits the pointer alignment off; older versions key the alignment
    of a pointer under do_not_specialize)."""
    kw = dict(do_not_specialize=DNS[name])
    try:
        import inspect
        if "do_not_specialize_on_alignment" in inspect.signature(triton.jit).parameters:
            kw["do_not_specialize_on_alignment"] = DNS[name]
    except (TypeError, ValueError):                    # pragma: no cover - a signature that inspect cannot read
        pass
    return triton.jit(**kw)


def slot_bits(np_: int) -> int:
    """The slot field's width in Policy E's key for NP lanes (a power of 2): 9 up to NP 512 (23b8a38's layout), else
    log2(NP). Tests force 9 at NP > 512 to rebuild the old layout (the negative control)."""
    return max(9, (int(np_) - 1).bit_length())


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
        ty = ptr.dtype.element_ty                           # fusedres2c: the destination's own type, cast explicitly
        tl.store(ptr + ln * 0, (tl.zeros([2], tl.int64) + val.to(tl.int64)).to(ty), mask=(ln == 0) & m)

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
    @_jit_dns("_fr_scan")
    def _fr_scan(ne_ptr, rec_ptr, usg_ptr, bne_ptr, brec_ptr, misc_ptr, RH, S, C, X, RES, BLOCK: tl.constexpr):
        # fusedres2c: every buffer here is int64 and every stored value is cast to int64 explicitly. With int32
        # buffers (7727e34) the interpreter wrote past d.bne / d.brec at every layer (FR10, job 2191259): a scan / sum
        # of int32 values is held as 8-byte numpy data under its declared int32 type, so an int32 store wrote 8 bytes
        # per element and the last lane spilled 4 bytes past the end (values stayed right, little-endian).
        offs = tl.arange(0, BLOCK)
        cn = tl.sum(tl.zeros([2], tl.int64), axis=0)
        cr = cn
        for c0 in range(0, RH, BLOCK):
            m = c0 + offs < RH
            n = tl.load(ne_ptr + c0 + offs, mask=m, other=0).to(tl.int64)
            r = tl.load(rec_ptr + c0 + offs, mask=m, other=0).to(tl.int64)
            tl.store(bne_ptr + c0 + offs, (cn + tl.cumsum(n, axis=0) - n).to(tl.int64), mask=m)
            tl.store(brec_ptr + c0 + offs, (cr + tl.cumsum(r, axis=0) - r).to(tl.int64), mask=m)
            cn = cn + tl.sum(n, axis=0).to(tl.int64)
            cr = cr + tl.sum(r, axis=0).to(tl.int64)
        bor = cn * 0
        for c0 in range(0, S, BLOCK):
            m = c0 + offs < S
            u = tl.load(usg_ptr + c0 + offs, mask=m, other=0).to(tl.int64)
            bor = bor + tl.sum(tl.where(m, tl.maximum(u - C, 0), 0), axis=0).to(tl.int64)
        budget = tl.maximum(X - bor - RES, 0).to(tl.int64)
        q4 = tl.arange(0, 4)
        tl.store(misc_ptr + q4, tl.where(q4 == 0, budget, tl.where(q4 == 1, cr, cn)).to(tl.int64), mask=q4 < 3)

    # =================================================================================================================
    # V1:the verify rows' classification (counts for the global K5 prefix and the record compaction)
    # =================================================================================================================
    @_jit_dns("_fr_v1")   # m4: no re-compile; [stack2] rows and sel too
    def _fr_v1(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               ne_ptr, rec_ptr, usg_ptr, err_ptr, cntv_ptr, look_ptr,
               H, R, K, NBLK, NPOS, C, Rv, MAXRV, NLOOK,
               KP: tl.constexpr, NP: tl.constexpr, LZ: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        base = pid.to(tl.int64) * NPOS
        jj, mj, code, sblk, gen, prot, stamp, resid, res, fly, usage = _slots(
            code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, base, NPOS, NBLK, NP)
        free_local = tl.maximum(C - usage, 0)
        _put1(usg_ptr + pid, usage, pid >= 0)
        lk = tl.arange(0, LZ)                                                     # program 0 zeroes V2's sums
        for c0 in range(0, NLOOK, LZ):
            tl.store(look_ptr + c0 + lk, tl.zeros([LZ], tl.int64), mask=(c0 + lk < NLOOK) & (pid == 0))
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
        tl.atomic_or(err_ptr, errb.to(tl.int64))                                  # err is int64 (fusedres2c)

    # =================================================================================================================
    # V2: the verify decisions
    # =================================================================================================================
    @_jit_dns("_fr_v2")                         # [stack2] ints, rows, sel
    def _fr_v2(code_ptr, blk_ptr, gen_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
               vix_ptr, vhb_ptr, vtl_ptr, vnt_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
               dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
               ne_ptr, rec_ptr, bne_ptr, brec_ptr, misc_ptr,
               tv_ptr, vpage_ptr, vnpg_ptr, ppage_ptr, pnpg_ptr, ids_ptr, vrec_ptr, vrcnt_ptr, cntv_ptr, look_ptr,
               vread_ptr, post_ptr, usg2_ptr, selw_ptr, rowsel_ptr, err_ptr, stats_ptr,
               H, B, R, K, NBLK, NPOS, C, X, Rv, Rd, MAXRV, MAXRD, NV, CAPV, S, WB, WP, T, P,
               KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr,
               ZP: tl.constexpr, PE_INFL: tl.constexpr, LAST: tl.constexpr, PLANT: tl.constexpr,
               NSTAT: tl.constexpr, SMUL: tl.constexpr, CMUL: tl.constexpr, SMASK: tl.constexpr):
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
        # one key per slot: class (LRU candidate 0, in-flight candidate 1, none 2) << (47 + SB) | stamp << SB | slot
        # [stack2] SMUL = 2^SB, CMUL = 2^(47 + SB), SMASK = 2^SB - 1 (host: slot_bits(NP)); SB 9 = 23b8a38's literals
        cls = tl.where(cand, 0, tl.where(candf, 1, 2)).to(tl.int64)
        kst = tl.where(cand, stamp, tl.where(candf, tl.minimum(stamp, 2305843009213693952), 0))
        comp = cls * CMUL + kst * SMUL + jj.to(tl.int64)
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
            evs = (tl.sum(tl.where(rrn[None, :] == ruse[:, None], srtc[None, :], 0), axis=1) & SMASK).to(tl.int32)
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
        tl.store(post_ptr + (0 * S + pid) * NP + jj, newblk.to(tl.int64))           # post is int64 (fusedres2c)
        tl.store(post_ptr + (1 * S + pid) * NP + jj, evold.to(tl.int64))
        tl.store(post_ptr + (2 * S + pid) * NP + jj, evinfl.to(tl.int64))
        tl.store(post_ptr + (3 * S + pid) * NP + jj, free2.to(tl.int64))
        _put1(usg2_ptr + pid, usage2, pid >= 0)
        _put1(vrcnt_ptr, totrec.to(tl.int64), pid == 0)
        if LAST:
            tl.store(selw_ptr + pid * WB + zw, _pack(selu, ZP), mask=zw < WB)
        tl.atomic_or(err_ptr, errb.to(tl.int64))                                  # err is int64 (fusedres2c)

    # =================================================================================================================
    # D1: the draft rows' classification on the state after the verify rows
    # =================================================================================================================
    @_jit_dns("_fr_d1")                         # [stack2] ints, rows, sel
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
        tl.atomic_or(err_ptr, errb.to(tl.int64))                                  # err is int64 (fusedres2c)

    # =================================================================================================================
    # D2: the draft decisions, the draft record, the bias rows, the selection bitmaps
    # =================================================================================================================
    @_jit_dns("_fr_d2")                         # [stack2] ints, rows, sel, bias
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
                dst = sb_ptr + ((b64 * ROWS_S + tl.maximum(take, 0)[:, None].to(tl.int64) * BS + bb[None, :]) * H
                                + h)                          # a masked lane (take -1) still addresses the table
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
        tl.atomic_or(err_ptr, errb.to(tl.int64))                                  # err is int64 (fusedres2c)

    # =================================================================================================================
    # [fr3] V3: the same decisions with bounded register tiles (module docstring, V3). The largest tiles per program:
    # (KP x KP) (_sorted_unique, _seen, as v2), (PP x KP) and (PP x TP) (the page table, as v2), (ZC x KP) (rowsel /
    # sel), (CH x KP) (PREP's insel), (ZC x TC) (PREP's tables). No NP-wide vector, no (KP x NP) tile, no tl.sort.
    # The per-stream block table tab (S, 7, ZP) int64: 0 the highest resident slot (-1), 1 their count, 2 the highest
    # current-owner slot in flight (-1), 3 their count, 4 the slot a verify row took for the block (-1), 5 a verify row
    # evicted the block (0 / 1), 6 it evicted it while in flight (0 / 1). post (4, S, NP) int64 (v3's use): 0 newblk
    # per slot (-1), 1 the verify read bit per slot, 2 Policy E's key per slot, 3 FREE after the verify rows.
    # flist (S, 2, NP): the FREE slots in slot order (0 before the verify rows: PREP; 1 after them: D1). pm (S, 4):
    # 0 Policy E's candidate count, 1 / 2 the FREE counts of flist 0 / 1.
    # =================================================================================================================
    @triton.jit
    def _tab3(tb, blk, f: tl.constexpr, ZP: tl.constexpr):
        """[fr3] Field f of the stream's block table at the blocks blk (all in [0, NBLK) <= ZP: an invalid entry's blk
        is 0), as int32."""
        return tl.load(tb + f * ZP + blk).to(tl.int32)

    @triton.jit
    def _d_state3(tb, blk, ZP: tl.constexpr, HAS_V: tl.constexpr):
        """[fr3] _d_state by table loads: the original resident / in-flight slot and counts (PREP), then (HAS_V) the
        verify rows' changes per block (V2): an evicted block's slot is gone, a block a verify row took is in flight,
        a superseded prefetch's block is not in flight."""
        slot = _tab3(tb, blk, 0, ZP)
        nres = _tab3(tb, blk, 1, ZP)
        islot = _tab3(tb, blk, 2, ZP)
        nfly = _tab3(tb, blk, 3, ZP)
        infl = islot >= 0
        if HAS_V:
            took = _tab3(tb, blk, 4, ZP) >= 0
            gone = _tab3(tb, blk, 5, ZP) != 0
            gonef = _tab3(tb, blk, 6, ZP) != 0
            slot = tl.where(gone, -1, slot)
            infl = (infl | took) & (gonef == 0)
        return slot, infl, nres, nfly

    @triton.jit
    def _selbits_at(s, v0, z0, ZC: tl.constexpr):
        """[fr3] _selbits on the blocks z0 .. z0 + ZC - 1 only (a (ZC x KP) tile)."""
        z = z0 + tl.arange(0, ZC)
        return tl.max((v0[None, :] & (s[None, :] == z[:, None])).to(tl.int32), axis=1) > 0

    # PREP: the block tables, the FREE list, the usage, Policy E's keys, V2's scratch
    @_jit_dns("_fr3_prep")
    def _fr3_prep(code_ptr, blk_ptr, cur_ptr, prot_ptr, stamp_ptr, sel_ptr, kid_ptr,
                  vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
                  dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
                  tab_ptr, flist_ptr, pm_ptr, post_ptr, usg_ptr, err_ptr,
                  H, R, K, NBLK, NPOS, Rv, Rd, MAXRV, MAXRD, S,
                  KP: tl.constexpr, NP: tl.constexpr, ZP: tl.constexpr, CH: tl.constexpr, ZC: tl.constexpr,
                  TC: tl.constexpr, HAS_V: tl.constexpr, PE_INFL: tl.constexpr, SMUL: tl.constexpr,
                  CMUL: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        pid64 = pid.to(tl.int64)
        base = pid64 * NPOS
        tb = tab_ptr + pid64 * (7 * ZP)
        # (1) the block tables. Per block: the highest slot of the resident slots (LRU / PO / VRES) and of the current
        # owner's slots in flight (v2's _lookup: the highest of several, A1), and how many there are (err bits 1 / 2)
        zl = tl.arange(0, ZC)
        tj = tl.arange(0, TC)
        for z0 in range(0, ZP, ZC):
            zz = z0 + zl
            rmax = tl.full([ZC], -1, tl.int32)
            rn = tl.zeros([ZC], tl.int32)
            fmax = tl.full([ZC], -1, tl.int32)
            fn = tl.zeros([ZC], tl.int32)
            for c0 in range(0, NP, TC):
                jc = c0 + tj
                mj = jc < NPOS
                code = tl.load(code_ptr + base + jc, mask=mj, other=-1).to(tl.int32)
                sblk = tl.load(blk_ptr + base + jc, mask=mj, other=-1).to(tl.int32)
                cur = tl.load(cur_ptr + base + jc, mask=mj, other=0)
                okb = (sblk >= 0) & (sblk < NBLK)
                res = okb & ((code == 2) | (code == 3) | (code == 4))
                fly = okb & (code == 1) & (cur != 0)
                hit = sblk[None, :] == zz[:, None]
                mr = res[None, :] & hit
                mf = fly[None, :] & hit
                rmax = tl.maximum(rmax, tl.max(tl.where(mr, jc[None, :], -1), axis=1))
                rn = rn + tl.sum(mr.to(tl.int32), axis=1)
                fmax = tl.maximum(fmax, tl.max(tl.where(mf, jc[None, :], -1), axis=1))
                fn = fn + tl.sum(mf.to(tl.int32), axis=1)
            tl.store(tb + 0 * ZP + zz, rmax.to(tl.int64))
            tl.store(tb + 1 * ZP + zz, rn.to(tl.int64))
            tl.store(tb + 2 * ZP + zz, fmax.to(tl.int64))
            tl.store(tb + 3 * ZP + zz, fn.to(tl.int64))
            tl.store(tb + 4 * ZP + zz, tl.full([ZC], -1, tl.int64))           # V2 fills 4-6 (the verify rows)
            tl.store(tb + 5 * ZP + zz, tl.zeros([ZC], tl.int64))
            tl.store(tb + 6 * ZP + zz, tl.zeros([ZC], tl.int64))
        # (2) the usage and the FREE slots in slot order (v2's fr rank: the q-th FREE slot is flist[q]); V2's scratch
        cl = tl.arange(0, CH)
        fl = flist_ptr + pid64 * (2 * NP)
        nfree = tl.sum(tl.zeros([2], tl.int32), axis=0)
        usage = nfree
        for c0 in range(0, NP, CH):
            jc = c0 + cl
            mj = jc < NPOS
            code = tl.load(code_ptr + base + jc, mask=mj, other=-1).to(tl.int32)
            free = mj & (code == 0)
            fi = free.to(tl.int32)
            tl.store(fl + nfree + tl.cumsum(fi, axis=0) - fi, jc.to(tl.int64), mask=free)
            nfree = nfree + tl.sum(fi, axis=0)
            usage = usage + tl.sum((mj & (code != 0)).to(tl.int32), axis=0)
            if HAS_V:
                tl.store(post_ptr + (0 * S + pid64) * NP + jc, tl.full([CH], -1, tl.int64))       # newblk
                tl.store(post_ptr + (1 * S + pid64) * NP + jc, tl.zeros([CH], tl.int64))          # read bits
                tl.store(post_ptr + (3 * S + pid64) * NP + jc, free.to(tl.int64))                 # FREE after V
        _put1(usg_ptr + pid, usage, pid >= 0)
        _put1(pm_ptr + pid64 * 4 + 1, nfree, pid >= 0)
        # (3) Policy E's key per slot (v2's V2: insel over the verify AND draft rows of b, cand, candf, the key)
        if HAS_V:
            vstart = tl.load(vstart_ptr + b)
            vcnt = tl.load(vcnt_ptr + b)
            dstart = tl.load(dstart_ptr + b)
            dcnt = tl.load(dcnt_ptr + b)
            ncand = tl.sum(tl.zeros([2], tl.int32), axis=0)
            errb = ncand
            for c0 in range(0, NP, CH):
                jc = c0 + cl
                mj = jc < NPOS
                code = tl.load(code_ptr + base + jc, mask=mj, other=-1).to(tl.int32)
                sblk = tl.load(blk_ptr + base + jc, mask=mj, other=-1).to(tl.int32)
                cur = tl.load(cur_ptr + base + jc, mask=mj, other=0)
                prot = tl.load(prot_ptr + base + jc, mask=mj, other=0)
                stamp = tl.load(stamp_ptr + base + jc, mask=mj, other=0)
                okb = (sblk >= 0) & (sblk < NBLK)
                resid = (code == 2) | (code == 3) | (code == 4)
                fly = okb & (code == 1) & (cur != 0)
                insel = mj & (code < -1)
                for i in range(0, MAXRV):
                    act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv,
                                                  h, R, K, NBLK, KP)
                    hit = tl.max((v0[None, :] & (s[None, :] == sblk[:, None])).to(tl.int32), axis=1) > 0
                    insel = insel | (hit & act)
                for i in range(0, MAXRD):
                    act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, dstart, dcnt, i, Rd,
                                                  h, R, K, NBLK, KP)
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
                cls = tl.where(cand, 0, tl.where(candf, 1, 2)).to(tl.int64)
                kst = tl.where(cand, stamp, tl.where(candf, tl.minimum(stamp, 2305843009213693952), 0))
                tl.store(post_ptr + (2 * S + pid64) * NP + jc, cls * CMUL + kst * SMUL + jc.to(tl.int64))
                ncand = ncand + tl.sum(anyc.to(tl.int32), axis=0)
                bad = tl.max((anyc & ((stamp < 0) | (stamp >= 140737488355328))).to(tl.int32), axis=0)
                errb = errb | tl.where(bad > 0, 8, 0)
            _put1(pm_ptr + pid64 * 4, ncand, pid >= 0)
            tl.atomic_or(err_ptr, errb.to(tl.int64))

    # V1 (v3): v2's V1 with table loads
    @_jit_dns("_fr3_v1")
    def _fr3_v1(sel_ptr, kid_ptr, vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
                tab_ptr, ne_ptr, rec_ptr, usg_ptr, err_ptr, cntv_ptr, look_ptr,
                H, R, K, NBLK, C, Rv, MAXRV, NLOOK,
                KP: tl.constexpr, ZP: tl.constexpr, LZ: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        tb = tab_ptr + pid.to(tl.int64) * (7 * ZP)
        usage = tl.load(usg_ptr + pid).to(tl.int32)                              # PREP's (v2: V1's own _slots)
        free_local = tl.maximum(C - usage, 0)
        lk = tl.arange(0, LZ)                                                     # program 0 zeroes V2's sums
        for c0 in range(0, NLOOK, LZ):
            tl.store(look_ptr + c0 + lk, tl.zeros([LZ], tl.int64), mask=(c0 + lk < NLOOK) & (pid == 0))
        _put1(cntv_ptr, tl.sum(tl.zeros([2], tl.int64), axis=0), pid == 0)
        start = tl.load(vstart_ptr + b)
        cnt = tl.load(vcnt_ptr + b)
        carry = tl.sum(tl.zeros([2], tl.int32), axis=0)
        errb = tl.sum(tl.zeros([2], tl.int32), axis=0)
        for i in range(0, MAXRV):
            act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rv, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot = _tab3(tb, blk, 0, ZP)
            nres = _tab3(tb, blk, 1, ZP)
            islot = _tab3(tb, blk, 2, ZP)
            nfly = _tab3(tb, blk, 3, ZP)
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
        tl.atomic_or(err_ptr, errb.to(tl.int64))

    # V2 (v3): v2's V2 with table / FREE-list loads, Policy E by bounded passes, the state by scatters
    @_jit_dns("_fr3_v2")
    def _fr3_v2(code_ptr, blk_ptr, gen_ptr, cur_ptr, sel_ptr, kid_ptr,
                vix_ptr, vhb_ptr, vtl_ptr, vnt_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
                tab_ptr, flist_ptr, pm_ptr, ne_ptr, rec_ptr, bne_ptr, brec_ptr, misc_ptr, usg_ptr,
                tv_ptr, vpage_ptr, vnpg_ptr, ppage_ptr, pnpg_ptr, ids_ptr, vrec_ptr, vrcnt_ptr, cntv_ptr, look_ptr,
                vread_ptr, post_ptr, usg2_ptr, selw_ptr, rowsel_ptr, err_ptr, stats_ptr,
                H, B, R, K, NBLK, NPOS, C, Rv, MAXRV, NV, CAPV, S, WB, WP, T, P,
                KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr, ZP: tl.constexpr,
                CH: tl.constexpr, ZC: tl.constexpr, LAST: tl.constexpr, PLANT: tl.constexpr,
                NSTAT: tl.constexpr, CMUL: tl.constexpr, SMASK: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        pid64 = pid.to(tl.int64)
        base = pid64 * NPOS
        tb = tab_ptr + pid64 * (7 * ZP)
        fl = flist_ptr + pid64 * (2 * NP)                                         # PREP: FREE slots in slot order
        pnb = post_ptr + (0 * S + pid64) * NP                                     # newblk per slot (PREP: -1)
        prd = post_ptr + (1 * S + pid64) * NP                                     # read bits per slot (PREP: 0)
        pk = post_ptr + (2 * S + pid64) * NP                                      # Policy E's keys (PREP)
        pfr = post_ptr + (3 * S + pid64) * NP                                     # FREE after V (PREP: FREE)
        usage = tl.load(usg_ptr + pid).to(tl.int32)
        free_local = tl.maximum(C - usage, 0)
        ncand = tl.load(pm_ptr + pid64 * 4).to(tl.int32)
        nfree = tl.load(pm_ptr + pid64 * 4 + 1).to(tl.int32)
        budget = tl.load(misc_ptr)
        totrec = tl.load(misc_ptr + 1)
        vstart = tl.load(vstart_ptr + b)
        vcnt = tl.load(vcnt_ptr + b)
        usage2 = usage
        errb = tl.sum(tl.zeros([2], tl.int32), axis=0)
        carry_f = errb
        carry_g = errb
        carry_r = errb
        # Policy E: the last key taken. EXACT: the next victim is the smallest key above it (keys are distinct: the
        # slot is in the low bits), so the e-th one taken is v2's srtc[e]. PLANT 1 takes the largest candidate key
        # below it (srtc[ncand - 1 - e]; a candidate key is < 2 CMUL when its stamp is in range, A2)
        if PLANT == 1:
            lastk = tl.sum(tl.zeros([2], tl.int64), axis=0) + 2 * CMUL
        else:
            lastk = tl.sum(tl.zeros([2], tl.int64), axis=0) - 1
        kk = tl.arange(0, KP)
        mk = kk < K
        cc = tl.arange(0, PP)
        tt = tl.arange(0, TP)
        q4 = tl.arange(0, 4)
        st16 = tl.arange(0, 16)
        cl = tl.arange(0, CH)
        zl = tl.arange(0, ZC // 32)
        for i in range(0, MAXRV):
            act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv, h, R,
                                          K, NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot = _tab3(tb, blk, 0, ZP)
            nres = _tab3(tb, blk, 1, ZP)
            islot = _tab3(tb, blk, 2, ZP)
            nfly = _tab3(tb, blk, 3, ZP)
            igen = tl.where(islot >= 0, tl.maximum(tl.load(gen_ptr + base + tl.maximum(islot, 0)), 0), 0)
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
            fs = tl.where(qc < nfree, tl.load(fl + tl.minimum(tl.maximum(qc, 0), NP - 1)).to(tl.int32), -1)
            fsl = tl.where(fs >= 0, fs, NPOS)
            ri = refused.to(tl.int32)
            rev = carry_r + tl.cumsum(ri, axis=0) - ri
            evicted = refused & (rev < ncand)
            # this row's victims: the ranks carry_r .. carry_r + nev - 1 (refused entries rank consecutively), one
            # bounded pass over the stream's keys each
            nev = tl.sum(evicted.to(tl.int32), axis=0)
            evs = tl.zeros([KP], tl.int32)
            for e in range(0, nev):
                if PLANT == 1:
                    best = tl.sum(tl.zeros([2], tl.int64), axis=0) - 1
                    for c0 in range(0, NP, CH):
                        kv = tl.load(pk + c0 + cl)
                        best = tl.maximum(best, tl.max(tl.where((kv < lastk) & (kv < 2 * CMUL), kv, -1), axis=0))
                else:
                    best = tl.sum(tl.zeros([2], tl.int64), axis=0) + 9223372036854775807
                    for c0 in range(0, NP, CH):
                        kv = tl.load(pk + c0 + cl)
                        best = tl.minimum(best, tl.min(tl.where(kv > lastk, kv, 9223372036854775807), axis=0))
                lastk = best
                evs = tl.where(evicted & (rev == carry_r + e), (best & SMASK).to(tl.int32), evs)
            # the victim slot's block, in-flight state and gen (v2's _at_slot on evs): loads at evs (< NPOS)
            evblk_raw = tl.load(blk_ptr + base + evs, mask=evicted, other=-1073741824).to(tl.int32)
            evcode = tl.load(code_ptr + base + evs, mask=evicted, other=-1).to(tl.int32)
            evcur = tl.load(cur_ptr + base + evs, mask=evicted, other=0)
            evfly = ((evblk_raw >= 0) & (evblk_raw < NBLK) & (evcode == 1) & (evcur != 0)).to(tl.int32)
            evgen_raw = tl.maximum(tl.load(gen_ptr + base + evs, mask=evicted, other=0), 0)
            ev_blk = tl.where(evicted, evblk_raw, -1)
            ev_infl = evicted & (evfly > 0)
            ev_gen = tl.where(ev_infl, evgen_raw, 0)
            over = refused & (evicted == 0)
            take = tl.where(granted, fsl, tl.where(evicted, evs, -1))
            taken = granted | evicted
            grp = tl.load(tb + 4 * ZP + blk, mask=dup, other=-1).to(tl.int32)    # an EARLIER row's take (barrier)
            if PLANT == 3:
                grp_use = tl.where(dup, -1, grp)
            else:
                grp_use = grp
            rslot = tl.where(ready, slot, tl.where(infl, islot, tl.where(fmiss, take, tl.where(dup, grp_use, -1))))
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
                     + over.to(tl.int64) * 256)
            tl.store(vrec_ptr + 0 * CAPV + rpos, o.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 1 * CAPV + rpos, blk.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 2 * CAPV + rpos, flags, mask=wm)
            tl.store(vrec_ptr + 3 * CAPV + rpos, slot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 4 * CAPV + rpos, islot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 5 * CAPV + rpos, igen.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 6 * CAPV + rpos, rslot.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 7 * CAPV + rpos, take.to(tl.int64), mask=wm)
            tl.store(vrec_ptr + 8 * CAPV + rpos, ev_blk.to(tl.int64), mask=wm)
            pm_ = me & (o >= totrec) & (o < CAPV)                                 # the -1 padding (no record there)
            neg = tl.full([KP], -1, tl.int64)
            for f in tl.static_range(9):
                tl.store(vrec_ptr + f * CAPV + o, neg, mask=pm_)
            # the looks: READY first looks of (key, head, block), by the slot's state class
            rf = ready & first
            cat = tl.load(code_ptr + base + tl.maximum(slot, 0)).to(tl.int32)
            c0_ = tl.sum(rf.to(tl.int64), axis=0)
            c1 = tl.sum((rf & (cat == 2)).to(tl.int64), axis=0)
            c2 = tl.sum((rf & (cat == 3)).to(tl.int64), axis=0)
            c3 = tl.sum((rf & (cat == 4)).to(tl.int64), axis=0)
            lv = tl.where(q4 == 0, c0_, tl.where(q4 == 1, c1, tl.where(q4 == 2, c2, c3)))
            tl.atomic_add(look_ptr + j * 4 + q4, lv, mask=(q4 < 4) & act)
            # the verify read bits (v2's rbits): one equal value per slot, any order
            rs = tl.where(ready, slot, tl.where(infl, islot, tl.where(dup, rslot, tl.where(taken, take, -1))))
            rd = valid & (rs >= 0) & (rs < NP) & act
            tl.store(prd + tl.minimum(tl.maximum(rs, 0), NP - 1), tl.full([KP], 1, tl.int64), mask=rd)
            # the stream's state after this row (v2's newblk / evold / evinfl / free2): the takes are distinct slots
            # and distinct blocks (a block's first miss is taken once per stream), the victims distinct slots
            tka = tl.minimum(tl.maximum(take, 0), NP - 1)
            tk = taken & act & (take >= 0) & (take < NP)
            tl.store(pnb + tka, blk.to(tl.int64), mask=tk)
            tl.store(tb + 4 * ZP + blk, take.to(tl.int64), mask=tk)
            evo = tl.maximum(evblk_raw, 0)
            evok = evicted & act & (evo < NBLK)
            evoa = tl.minimum(evo, ZP - 1)
            tl.store(tb + 5 * ZP + evoa, tl.full([KP], 1, tl.int64), mask=evok)
            tl.store(tb + 6 * ZP + evoa, tl.full([KP], 1, tl.int64), mask=evok & ev_infl)
            tl.store(pfr + tka, tl.zeros([KP], tl.int64), mask=granted & act & (take >= 0) & (take < NP))
            usage2 = usage2 + tl.where(act, tl.sum(gi, axis=0), 0)
            carry_f = carry_f + tl.where(act, tl.sum(fm, axis=0), 0)
            carry_g = carry_g + tl.where(act, tl.sum(gi, axis=0), 0)
            carry_r = carry_r + tl.where(act, tl.sum(ri, axis=0), 0)
            tl.atomic_add(cntv_ptr, tl.where(act, tl.sum(over.to(tl.int64), axis=0), 0))
            # REPORT / L34 preconditions: per kind, the entries of this row (n_dov from the TRUE grp: plant 3 is
            # effective at dup & grp >= 0, i.e. where v_dup - v_dup_over grows)
            n_ev = tl.sum(evicted.to(tl.int64), axis=0)
            n_fl = tl.sum((evicted & (ncand - 1 - rev != rev)).to(tl.int64), axis=0)
            n_dov = tl.sum((dup & (grp < 0)).to(tl.int64), axis=0)
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
            e16 = tl.where(act & ((tl.sum(nei, axis=0) != tl.load(ne_ptr + idx))
                                  | (tl.sum(rmi, axis=0) != tl.load(rec_ptr + idx))), 16, 0)
            errb = errb | e1 | e2 | e4 | e16
            # this row's selection bitmap (rowsel), per block chunk
            for z0 in range(0, ZP, ZC):
                zw = z0 // 32 + zl
                tl.store(rowsel_ptr + (tr * H + h) * WB + zw, _pack(_selbits_at(s, v0, z0, ZC), ZC),
                         mask=(zw < WB) & act)
            tl.debug_barrier()                    # the took table and the per-slot scratch: later rows read them
        tl.debug_barrier()
        # the end pass: the load ids (newblk per slot) and the verify read bitmap, per slot chunk
        for c0 in range(0, NP, CH):
            jc = c0 + cl
            tl.store(ids_ptr + (h * B + b) * NPOS + jc, tl.load(pnb + jc), mask=jc < NPOS)
            pw = c0 // 32 + tl.arange(0, CH // 32)
            tl.store(vread_ptr + pid * WP + pw, _pack(tl.load(prd + jc) != 0, CH), mask=pw < WP)
        _put1(usg2_ptr + pid, usage2, pid >= 0)
        _put1(vrcnt_ptr, totrec.to(tl.int64), pid == 0)
        if LAST:                                  # the stream's union (sel) when no draft row follows, per chunk
            for z0 in range(0, ZP, ZC):
                selu = tl.zeros([ZC], tl.int32) != 0
                for i in range(0, MAXRV):
                    act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i, Rv,
                                                  h, R, K, NBLK, KP)
                    selu = selu | (_selbits_at(s, v0, z0, ZC) & act)
                zw = z0 // 32 + zl
                tl.store(selw_ptr + pid * WB + zw, _pack(selu, ZC), mask=zw < WB)
        tl.atomic_or(err_ptr, errb.to(tl.int64))

    # D1 (v3): v2's D1 with table loads; it also writes D2's FREE list (the FREE slots after the verify rows)
    @_jit_dns("_fr3_d1")
    def _fr3_d1(code_ptr, sel_ptr, kid_ptr, dix_ptr, dhb_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
                tab_ptr, flist_ptr, pm_ptr, post_ptr, usg_ptr, usg2_ptr, ne_ptr, rec_ptr, usgd_ptr, err_ptr, cntd_ptr,
                H, R, K, NBLK, NPOS, C, Rd, MAXRD, S, NPAIRS,
                KP: tl.constexpr, NP: tl.constexpr, ZP: tl.constexpr, CH: tl.constexpr, HAS_V: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        pid64 = pid.to(tl.int64)
        base = pid64 * NPOS
        tb = tab_ptr + pid64 * (7 * ZP)
        if HAS_V:
            usage = tl.load(usg2_ptr + pid).to(tl.int32)                          # V2's (after the verify rows)
        else:
            usage = tl.load(usg_ptr + pid).to(tl.int32)                           # PREP's
        free_local = tl.maximum(C - usage, 0)
        _put1(usgd_ptr + pid, usage, pid >= 0)
        q4 = tl.arange(0, 4)                                                       # program 0: (masked, refused, pairs)
        tl.store(cntd_ptr + q4, tl.where(q4 == 2, NPAIRS, 0).to(tl.int64), mask=(q4 < 3) & (pid == 0))
        cl = tl.arange(0, CH)
        fl = flist_ptr + pid64 * (2 * NP) + NP
        nfree = tl.sum(tl.zeros([2], tl.int32), axis=0)
        for c0 in range(0, NP, CH):
            jc = c0 + cl
            mj = jc < NPOS
            if HAS_V:
                free = (tl.load(post_ptr + (3 * S + pid64) * NP + jc) != 0) & mj
            else:
                free = mj & (tl.load(code_ptr + base + jc, mask=mj, other=-1).to(tl.int32) == 0)
            fi = free.to(tl.int32)
            tl.store(fl + nfree + tl.cumsum(fi, axis=0) - fi, jc.to(tl.int64), mask=free)
            nfree = nfree + tl.sum(fi, axis=0)
        _put1(pm_ptr + pid64 * 4 + 2, nfree, pid >= 0)
        start = tl.load(dstart_ptr + b)
        cnt = tl.load(dcnt_ptr + b)
        carry = tl.sum(tl.zeros([2], tl.int32), axis=0)
        errb = tl.sum(tl.zeros([2], tl.int32), axis=0)
        for i in range(0, MAXRD):
            act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, inflight, nres, nfly = _d_state3(tb, blk, ZP, HAS_V)
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
        tl.atomic_or(err_ptr, errb.to(tl.int64))

    # D2 (v3): v2's D2 with table / FREE-list loads, rowsel and sel per block chunk
    @_jit_dns("_fr3_d2")
    def _fr3_d2(sel_ptr, kid_ptr, vix_ptr, vhb_ptr, vstart_ptr, vcnt_ptr, vorder_ptr,
                dix_ptr, dhb_ptr, dtl_ptr, dnt_ptr, dstart_ptr, dcnt_ptr, dorder_ptr,
                tab_ptr, flist_ptr, pm_ptr, ne_ptr, rec_ptr, bne_ptr, brec_ptr, misc_ptr, usgd_ptr,
                td_ptr, dpage_ptr, dnpg_ptr, ppage_ptr, pnpg_ptr, drec_ptr, dcnt_out_ptr, cntd_ptr,
                selw_ptr, rowsel_ptr, err_ptr, stats_ptr, sb_ptr, cis_ptr,
                H, B, R, K, NBLK, NPOS, C, Rv, Rd, MAXRV, MAXRD, ND, CAPD, S, WB, T, P,
                BS, ROWS_S, ROWS_C,
                KP: tl.constexpr, NP: tl.constexpr, PP: tl.constexpr, TP: tl.constexpr, ZP: tl.constexpr,
                ZC: tl.constexpr, BSP: tl.constexpr, HAS_V: tl.constexpr, BIAS: tl.constexpr,
                PLANT: tl.constexpr, NSTAT: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        pid64 = pid.to(tl.int64)
        tb = tab_ptr + pid64 * (7 * ZP)
        fl = flist_ptr + pid64 * (2 * NP) + NP                                    # D1: FREE slots after V
        nfree = tl.load(pm_ptr + pid64 * 4 + 2).to(tl.int32)
        usage = tl.load(usgd_ptr + pid)
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
        zl = tl.arange(0, ZC // 32)
        bb = tl.arange(0, BSP)
        for i in range(0, MAXRD):
            act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h, R, K,
                                          NBLK, KP)
            valid, blk = _sorted_unique(s, v0, K, NBLK, KP)
            slot, inflight, nres, nfly = _d_state3(tb, blk, ZP, HAS_V)
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
            fs = tl.where(qc < nfree, tl.load(fl + tl.minimum(tl.maximum(qc, 0), NP - 1)).to(tl.int32), -1)
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
            pm_ = me & (o >= totrec) & (o < CAPD)
            neg = tl.full([KP], -1, tl.int64)
            for f in tl.static_range(5):
                tl.store(drec_ptr + f * CAPD + o, neg, mask=pm_)
            # the counters (masked pairs, refusals) and the REPORT / L34 preconditions
            nref = tl.sum(refused.to(tl.int64), axis=0)
            q4 = tl.arange(0, 4)
            cv = tl.where(q4 == 0, (nready < nvalid).to(tl.int64), tl.where(q4 == 1, nref, 0))
            tl.atomic_add(cntd_ptr + q4, cv, mask=(q4 < 2) & act)
            n_sdup = (tl.sum(v0.to(tl.int32), axis=0) - nvalid).to(tl.int64)
            sv = tl.where(st16 == 3, tl.sum(gi.to(tl.int64), axis=0), tl.where(st16 == 4, nref, tl.where(
                st16 == 9, tl.sum(dup.to(tl.int64), axis=0), tl.where(st16 == 10, n_sdup, 0))))
            tl.atomic_add(stats_ptr + st16, sv, mask=(st16 < NSTAT) & act)
            # the bias rows of the granted takes (as v2's D2)
            if BIAS != 0:
                gm = (granted & act & (take >= 0) & (take < NPOS))[:, None] & (bb[None, :] < BS)
                b64 = b.to(tl.int64)
                src = cis_ptr + ((b64 * ROWS_C + blk[:, None].to(tl.int64) * BS + bb[None, :]) * H + h)
                dst = sb_ptr + ((b64 * ROWS_S + tl.maximum(take, 0)[:, None].to(tl.int64) * BS + bb[None, :]) * H
                                + h)
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
            for z0 in range(0, ZP, ZC):
                zw = z0 // 32 + zl
                tl.store(rowsel_ptr + (tr * H + h) * WB + zw, _pack(_selbits_at(s, v0, z0, ZC), ZC),
                         mask=(zw < WB) & act)
            carry_f = carry_f + tl.where(act, tl.sum(fm, axis=0), 0)
            carry_g = carry_g + tl.where(act, tl.sum(gi, axis=0), 0)
        # the stream's selection union (sel): the draft rows and, with verify rows, theirs too; per block chunk
        if HAS_V:
            vstart = tl.load(vstart_ptr + b)
            vcnt = tl.load(vcnt_ptr + b)
        for z0 in range(0, ZP, ZC):
            selu = tl.zeros([ZC], tl.int32) != 0
            for i in range(0, MAXRD):
                act, j, tr, kid, s, v0 = _row(dorder_ptr, dix_ptr, dhb_ptr, kid_ptr, sel_ptr, start, cnt, i, Rd, h,
                                              R, K, NBLK, KP)
                selu = selu | (_selbits_at(s, v0, z0, ZC) & act)
            if HAS_V:
                for i in range(0, MAXRV):
                    act, j, tr, kid, s, v0 = _row(vorder_ptr, vix_ptr, vhb_ptr, kid_ptr, sel_ptr, vstart, vcnt, i,
                                                  Rv, h, R, K, NBLK, KP)
                    selu = selu | (_selbits_at(s, v0, z0, ZC) & act)
            zw = z0 // 32 + zl
            tl.store(selw_ptr + pid * WB + zw, _pack(selu, ZC), mask=zw < WB)
        _put1(dcnt_out_ptr, totrec.to(tl.int64), pid == 0)
        tl.atomic_or(err_ptr, errb.to(tl.int64))


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

    def __init__(self, svc, check: bool = False, plant: int = 0, num_warps: int = 8, guard: bool = False,
                 warm: str = "auto", kernel: str = None):
        if triton is None:
            raise RuntimeError("resolve='fused' needs triton (not installed)")
        if warm not in WARM_MODES:
            raise ValueError("fused warm %r (one of %s)" % (warm, WARM_MODES))
        kernel = DEFAULT_KERNEL if kernel is None else kernel          # [fr3] the module's default, read now
        if kernel not in KERNELS:
            raise ValueError("fused kernel %r (one of %s)" % (kernel, KERNELS))
        self.kernel = kernel
        # [stack2] warm: compile every variant of a selection width before its first decision (warm_up). auto = on
        # for a CUDA service; the interpreter (CPU) compiles nothing, so auto is off there and the launch counts of
        # the CPU tests stay as they were (FR13 runs warm 'on' in the interpreter)
        self.warm = warm == "on" or (warm == "auto" and torch.device(svc.device).type == "cuda")
        self._warmed = set()                               # the variant keys warm_up launched
        self.warm_launches = 0                             # warm_up's launches (not in self.launches)
        self.warm_mem_gb = None                            # [oomfix] free GB before / after the process's first warm_up
        # [integ] the shared X pool (docs/R2_FUSEDRES_PLAN.md section 5, implemented): the kernels decide on LOGICAL
        # slots (I1-I5); the lanes are bound after V2 and after D2 (XPool.bind, torch ops), the draft bias rows go
        # through r2_xpool.draft_bias_fill_flat (D2 runs with BIAS 0), and the FREE slots' lanes are released after the
        # tick's sync_mirror, exactly where ResidencySC.resolve does it
        self.xp = getattr(svc, "xpool", None)
        if self.xp is not None and getattr(self.xp, "mode", None) != "pool":
            raise ValueError("resolve='fused': an attached xpool must be in mode 'pool' (got %r)"
                             % (getattr(self.xp, "mode", None),))
        # [integ3] review-integ2 M5: check mode keeps a SHADOW pool bound by S-C's rule from the torch path's decisions
        # (verify takes, then draft takes, per layer; the same release per tick); _compare compares the lane maps and
        # fills the reference bias rows on the SHADOW lanes, so a fused-only bind fault shows as a difference
        self.xp_ref = self.xp.clone() if (self.xp is not None and check) else None
        plants = {**PLANTS, **PLANTS3} if kernel == "v3" else PLANTS   # [fr3] plant 3 exists in v3 only
        if plant not in plants:
            raise ValueError("fused plant %r (one of %s)" % (plant, sorted(plants)))
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
        if max(_pow2(svc.n_hist), 64) - svc.n_hist > LANE_PAD:      # [stack2] a masked NP lane stays in the pad
            raise ValueError("resolve='fused': NP - (C + X) = %d > LANE_PAD %d"
                             % (max(_pow2(svc.n_hist), 64) - svc.n_hist, LANE_PAD))
        self.svc = svc
        self.check = bool(check)
        self.plant = int(plant)
        self.num_warps = int(num_warps)
        self.pe_inflight = bool(svc.policy_e_inflight)
        dev = svc.device
        self.guard = bool(guard)                           # tests: sentinel guards checked after every launch
        self.guard_hits = []                               # (tick, layer, kernel, buffer) whose guard changed
        self.guard_checks = 0                              # guard checks run (the detector's precondition)
        self._g_persist, self._g_tick, self._g_layer = [], [], []
        self.err = self._buf((), torch.int64, "err", self._g_persist).zero_()   # this tick's (zeroed per tick)
        self.err_any = self._buf((), torch.int64, "err_any", self._g_persist).zero_()   # m1: every tick's bits
        self.stats = self._buf((NSTAT_LANES,), torch.int64, "stats", self._g_persist).zero_()   # all 16 lanes
        svc.watch_flag("fused_res_err", self.err)
        self.launches = 0                                  # Triton launches (all layers, all ticks)
        self.layers = 0                                    # fused layers resolved
        self.bias_casts = 0                                # layers whose bias rows needed a dtype cast (BIAS 2)
        self.sel_conversions = 0                           # m3: layers whose selection was copied (dtype / layout)
        self.max_rows_per_b = [0, 0]                       # (verify, draft) rows of one batch slot, the run's max
        self.compared = 0                                  # check: layers compared with the torch path
        self.diffs = []                                    # check: (tick, layer, output) that differ
        self.trace = []                                    # check: (tick, layer, stats after the layer)

    # ----------------------------------------------------------------------------------------------- buffers
    def _buf(self, shape, dtype, name, group):
        """A kernel-written buffer: the logical tensor is a view of an allocation with LANE_PAD spare elements after
        it (fusedres2b). guard: the spare elements hold GUARD_SENTINEL and the buffer joins `group` for the checks."""
        n = 1
        for s in shape:
            n *= s
        full = torch.empty((n + LANE_PAD,), dtype=dtype, device=self.svc.device)
        if self.guard:
            full[n:].fill_(GUARD_SENTINEL)
            group.append((name, full, n))
        return full[:n].view(shape)

    def _check_guards(self, where, kernel):
        """guard: every guarded buffer's spare elements still hold the sentinel after `kernel` (host reads; tests).
        A changed guard is recorded with the kernel and the buffer, then restored."""
        self.guard_checks += 1
        for (name, full, n) in self._g_persist + self._g_tick + self._g_layer:
            tail = full[n:]
            if not bool((tail == GUARD_SENTINEL).all()):
                self.guard_hits.append((int(where[0]), int(where[1]), kernel, name))
                tail.fill_(GUARD_SENTINEL)

    def _go(self, where, name, kernel, grid, *args, **kw):
        """One launch (through _launch: tests count them), then the guard check in guard mode."""
        _launch(kernel, grid, *args, **kw)
        if self.guard:
            self._check_guards(where, name)

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
        lean = self._lean_on()                                        # [flean] --res-lean: the lean host checks
        same = False
        if lean:                                                      # ResidencySC._resolve_lean's order: a later
            same = bool(svc._tick) and svc._lean_same_rows(rows)      # layer with the tick's rows runs the per-layer
            if not same or not svc._lean_rows_ok(l, rows, sel):       # guards; else (and at the first layer) the
                svc._check_rows(l, rows, sel)                         # reference checks (the same exception)
        else:
            svc._check_rows(l, rows, sel)
        if svc.want_keys and not svc._tick:                           # [integ] S-C's [sharedkv2] hook, kept (host
            svc._check_tail_keys(rows)                                # rows only, once per tick)
        if not svc._tick:                                             # the tick's first layer: the per-tick copies
            svc.sync_mirror()
            svc._rows = svc._tick_rows(rows)
            if lean:                                                  # [flean] the tick's lean constants and the rows'
                svc._lean_c.new_tick()                                # snapshot (_lean_same_rows / _lean_rows_ok)
                svc._lean_snap(rows)
            if self.xp is not None:                                   # [integ][xpool] S-C's unbind, same place and
                if not svc.DECISION_READ_BLOCKING:                    # the same blocking-read precondition
                    raise OwnershipViolation("xpool: lanes are reused across requests only behind a blocking decision "
                                             "read (r2_xpool.py, the lane-level order); this class's is not")
                self.xp.release_free(svc.dmirror.code)
                if self.xp_ref is not None:                           # [integ3] M5: the shadow, the same release
                    self.xp_ref.release_free(svc.dmirror.code)
        elif svc._dirty:
            raise OwnershipViolation("S-C: the host state changed in the middle of a tick (the device mirror is stale)")
        elif not same and svc._row_sig(rows) != svc._rows["sig"]:     # [flean] same: the sig is equal (_lean_same_rows)
            raise ValueError("S-C: every layer of a tick resolves the same rows (layer %d differs)" % l)
        tk = svc._rows
        plan = svc._lean_plan(rows) if lean else svc._new_plan(rows)  # [flean] _new_plan's value, C-level work
        fz = tk.get("fz")
        if fz is None:
            fz = tk["fz"] = self._tick_setup(rows, tk)
        step, page, npg = self._layer(l, rows, sel, tk, fz)
        svc._tick.append(step)
        plan.page_table = page
        plan.num_pages = npg
        plan.last_len = tk["last_len"]
        plan.seq_idx = tk["seq_idx"]
        if svc.want_keys:                                             # [integ] S-C's [sharedkv2] hook, kept: the keys
            plan.page_key = svc._page_keys(len(rows), sel.shape[2] + N_TAIL, step, tk, plan.num_pages)   # from the
        return plan                                                   # FUSED decisions' (blk, valid / ready) columns

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
        self._g_tick = []                                   # this tick's guarded scratch (guard mode)

        def b32(shape, name):                                # int64 since fusedres2c (the name is kept)
            return self._buf(shape, torch.int64, name, self._g_tick)

        def scan_bufs(n, p):
            return dict(ne=b32((n,), p + "ne"), rec=b32((n,), p + "rec"), bne=b32((n,), p + "bne"),
                        brec=b32((n,), p + "brec"), misc=b32((4,), p + "misc"))
        out = dict(vstart=parts[0], vcnt=parts[1], vorder=parts[2], dstart=parts[3], dcnt=parts[4],
                   dorder=parts[5], maxrv=mv, maxrd=md, Rv=Rv, Rd=Rd, v=v, d=dd, kid=kid,
                   sv=scan_bufs(max(Rv * H, 1), "v."), sd=scan_bufs(max(Rd * H, 1), "d."),
                   usg=b32((S,), "usg"), usg2=b32((S,), "usg2"), usgd=b32((S,), "usgd"),
                   post=b32((4, S, NP), "post"), NP=NP, bh={})
        if self.kernel == "v3":                             # [fr3] the per-stream block tables, FREE lists, counts
            ZP = max(_pow2(svc.nblk), 64)
            out.update(tab=b32((S, 7, ZP), "fr3.tab"), flist=b32((S, 2, NP), "fr3.flist"), pm=b32((S, 4), "fr3.pm"))
        return out

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
        """[integ][xpool] (b, h) of every draft entry (K3X order), as ResidencySparse._record_inputs builds bbd / hhd
        for draft_bias_fill (device ops, once per tick and width)."""
        key = ("d", k)
        out = fz["bh"].get(key)
        if out is None:
            svc = self.svc
            H, Rd = svc.H, fz["Rd"]
            bb = fz["d"][0].view(Rd, 1, 1).expand(Rd, H, k).reshape(-1)
            hh = torch.arange(H, device=svc.device).view(1, H, 1).expand(Rd, H, k).reshape(-1)
            out = fz["bh"][key] = (bb, hh)
        return out

    # ----------------------------------------------------------------------------------------------- [flean]
    def _lean_on(self):
        """[flean] True iff the service runs --res-lean (OwnResidency(res_lean=True) -> enable_res_lean, after this
        resolver is built: so it is read at each call). Then resolve uses the lean host checks (_lean_same_rows,
        _lean_rows_ok, _lean_plan, once-per-tick LeanConsts), and _layer binds with XPool.bind_lean and fills the bias
        rows with draft_bias_fill_flat_lean (with --res-dev, LeanConsts.dev selects r2_resdev's ops inside them). The
        kernels, the decisions and the check mode's reference (torch_reference, the shadow pool's XPool.bind and
        draft_bias_fill_flat) do not change. False (the default): every call as before."""
        return bool(getattr(self.svc, "res_lean", False))

    def _xbase(self, fz, kind, k):
        """[flean] bind_lean's base (bb * H + hh) * n_hist of the verify ("v") or draft ("d") entries (K3X order), per
        selection width k (device ops, once per tick and width; the value bind_lean computes itself without it)."""
        key = ("xb", kind, k)
        out = fz["bh"].get(key)
        if out is None:
            bb, hh = self._bh(fz, k) if kind == "v" else self._bhd(fz, k)
            out = fz["bh"][key] = (bb * self.svc.H + hh) * self.xp.n_hist
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
        SB = slot_bits(NP)                                  # [stack2] Policy E's slot field (module-level: tests patch it)
        if 2 + STAMP_BITS + SB > 63:
            raise ValueError("resolve='fused': Policy E's key needs %d bits at NP %d" % (2 + STAMP_BITS + SB, NP))
        KP = max(_pow2(k), 2)
        PP = max(_pow2(P), 2)
        TP = max(_pow2(N_TAIL), 2)
        ZP = max(_pow2(NBLK), 64)
        WB, WP = (NBLK + 31) // 32, (NPOS + 31) // 32
        dm = svc.dmirror
        tabs = (dm.code[l], dm.block[l], dm.gen[l], dm.cur[l], dm.prot[l], dm.stamp[l])
        kid = fz["kid"]
        self._g_layer = []                                  # this layer's guarded outputs (guard mode)

        def b64(shape, name):                                # every kernel-written output: LANE_PAD spare elements
            return self._buf(shape, torch.int64, name, self._g_layer)
        page = self._buf((R, H, P), torch.int32, "plan.page", self._g_layer)
        npg = self._buf((R, H), torch.int32, "plan.npg", self._g_layer)
        selw = b64((B, H, WB), "sel")
        rowsel = b64((R, H, WB), "rowsel")
        rec = {}
        vrb, vhb, vtl, vnt, vix = fz["v"]
        drb, dhb, dtl, dnt, dix = fz["d"]
        vdec = ddec = None
        nv, nd = Rv * H * k, Rd * H * k
        caps = (max(nv, 1), max(nd, 1))
        nw = self.num_warps
        if self.warm:                                                 # [stack2] every variant of this width, once
            self.warm_up(l, sel, NP, k)                               # (scratch outputs; no state is read back)
        ref = torch_reference(svc, l, tk, sel) if self.check else None     # check mode: before any launch
        where = (svc.tick_no, l)
        sv, sd = fz["sv"], fz["sd"]
        v3 = self.kernel == "v3"                                      # [fr3] module docstring, V3
        CH = fr3_chunk(KP, NP)
        if v3:                                                        # PREP: tables, FREE list, keys, V2's scratch
            self._go(where, "_fr3_prep", _fr3_prep, (S,), dm.code[l], dm.block[l], dm.cur[l], dm.prot[l],
                     dm.stamp[l], sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"], dix, dhb, fz["dstart"],
                     fz["dcnt"], fz["dorder"], fz["tab"], fz["flist"], fz["pm"], fz["post"], fz["usg"], self.err,
                     H, R, k, NBLK, NPOS, max(Rv, 1), max(Rd, 1), fz["maxrv"], fz["maxrd"], S,
                     KP=KP, NP=NP, ZP=ZP, CH=CH, ZC=FR3_ZC, TC=FR3_TC, HAS_V=Rv > 0, PE_INFL=self.pe_inflight,
                     SMUL=1 << SB, CMUL=1 << (STAMP_BITS + SB), num_warps=nw)
            self.launches += 1
        if Rv:
            tv = b64((len(FIELDS_V), nv), "v.t")
            vpage = b64((Rv, H, P), "v.page")
            vnpg = b64((Rv, H), "v.num_pages")
            ids = b64((H, B, NPOS), "v.load_ids")
            vrec = b64((len(VREC), caps[0]), "vrec")
            vcnt = b64((1,), "vcnt")
            cntv = b64((1,), "v.counters")
            vlook = b64((Rv, 4), "vlook")
            vread = b64((B, H, WP), "vread")
            if v3:
                self._go(where, "_fr3_v1", _fr3_v1, (S,), sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"],
                         fz["tab"], sv["ne"], sv["rec"], fz["usg"], self.err, cntv, vlook,
                         H, R, k, NBLK, svc.C, Rv, fz["maxrv"], Rv * 4, KP=KP, ZP=ZP, LZ=LOOK_LANES, num_warps=nw)
            else:
                self._go(where, "_fr_v1", _fr_v1, (S,), *tabs, sel, kid, vix, vhb, fz["vstart"], fz["vcnt"],
                         fz["vorder"], sv["ne"], sv["rec"], fz["usg"], self.err, cntv, vlook,
                         H, R, k, NBLK, NPOS, svc.C, Rv, fz["maxrv"], Rv * 4,
                         KP=KP, NP=NP, LZ=LOOK_LANES, num_warps=nw)          # [stack2] a constant
            self._go(where, "_fr_scan.v", _fr_scan, (1,), sv["ne"], sv["rec"], fz["usg"], sv["bne"], sv["brec"],
                     sv["misc"], Rv * H, S, svc.C, svc.X, 0, BLOCK=SCAN_BLOCK,              # [stack2] a constant
                     num_warps=4)
            if v3:
                self._go(where, "_fr3_v2", _fr3_v2, (S,), dm.code[l], dm.block[l], dm.gen[l], dm.cur[l], sel, kid,
                         vix, vhb, vtl, vnt, fz["vstart"], fz["vcnt"], fz["vorder"],
                         fz["tab"], fz["flist"], fz["pm"], sv["ne"], sv["rec"], sv["bne"], sv["brec"], sv["misc"],
                         fz["usg"], tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, fz["post"],
                         fz["usg2"], selw, rowsel, self.err, self.stats,
                         H, B, R, k, NBLK, NPOS, svc.C, Rv, fz["maxrv"], nv, caps[0], S, WB, WP, N_TAIL, P,
                         KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, CH=CH, ZC=FR3_ZC, LAST=Rd == 0, PLANT=self.plant,
                         NSTAT=len(STATS), CMUL=1 << (STAMP_BITS + SB), SMASK=(1 << SB) - 1, num_warps=nw)
            else:
                self._go(where, "_fr_v2", _fr_v2, (S,), *tabs, sel, kid, vix, vhb, vtl, vnt, fz["vstart"], fz["vcnt"],
                         fz["vorder"], dix, dhb, fz["dstart"], fz["dcnt"], fz["dorder"],
                         sv["ne"], sv["rec"], sv["bne"], sv["brec"], sv["misc"],
                         tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, fz["post"], fz["usg2"], selw,
                         rowsel, self.err, self.stats,
                         H, B, R, k, NBLK, NPOS, svc.C, svc.X, Rv, max(Rd, 1), fz["maxrv"], fz["maxrd"], nv, caps[0],
                         S, WB, WP, N_TAIL, P,
                         KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, PE_INFL=self.pe_inflight,
                         LAST=Rd == 0, PLANT=self.plant, NSTAT=len(STATS), SMUL=1 << SB,
                         CMUL=1 << (STAMP_BITS + SB), SMASK=(1 << SB) - 1, num_warps=nw)
            self.launches += 3
            vdec = VerifyDecision(Rv, H, k, {f: tv[i] for i, f in enumerate(FIELDS_V)}, vpage, vnpg, ids,
                                  self._bh(fz, k), cntv)
            rec.update(vrec=vrec, vcnt=vcnt, vlook=vlook, vread=vread)
            if self.xp is not None and self._lean_on():               # [flean] the same binds, bind_lean (no hidden
                self.xp.bind_lean(l, vdec.bh[0], vdec.bh[1], vdec.t["take_slot"],   # sync; --res-dev: r2_resdev's
                                  vdec.t["granted"] | vdec.t["evict"], svc._lean_c,   # ops, cst.dev)
                                  base=self._xbase(fz, "v", k))
            elif self.xp is not None:                                 # [integ][xpool] recipe (2): the takes' lanes,
                self.xp.bind(l, vdec.bh[0], vdec.bh[1], vdec.t["take_slot"],   # before the wait and the gather
                             (vdec.t["granted"] | vdec.t["evict"]) != 0)
            if ref is None:
                svc.dev.wait(l, vdec)                                 # S-C's order: wait, gather, then the drafts
                svc.dev.gather(l, vdec)
            else:                                                     # check: compare first (a wrong decision may
                self.diffs += self._cmp_dec(where, "v", vdec, ref["vdec"])   # make the CPU stand-in raise)
                self._wait_gather_traced(l, vdec, where)
        sb_before = None
        if Rd:
            td = b64((len(FIELDS), nd), "d.t")
            dpage = b64((Rd, H, P), "d.page")
            dnpg = b64((Rd, H), "d.num_pages")
            drec = b64((len(DREC), caps[1]), "drec")
            dcnt = b64((1,), "dcnt")
            cntd4 = b64((4,), "d.counters")                   # the 4 lanes of the counter atomics; counters = [:3]
            bias, sb, cis, rows_s, rows_c, flat_bias = self._bias_args(l, sel)   # [stack2] shared with warm_up
            if self.check and (flat_bias or bias):                    # flat_bias: recipe (3), D2 with BIAS 0; the bias
                sb_before = svc.store_b[l].clone()                    # rows go through draft_bias_fill_flat
            if bias == 2:
                self.bias_casts += 1                                  # m9: the cast in the kernel (counted)
            if v3:
                self._go(where, "_fr3_d1", _fr3_d1, (S,), dm.code[l], sel, kid, dix, dhb, fz["dstart"], fz["dcnt"],
                         fz["dorder"], fz["tab"], fz["flist"], fz["pm"], fz["post"], fz["usg"], fz["usg2"], sd["ne"],
                         sd["rec"], fz["usgd"], self.err, cntd4, H, R, k, NBLK, NPOS, svc.C, Rd, fz["maxrd"], S,
                         Rd * H, KP=KP, NP=NP, ZP=ZP, CH=CH, HAS_V=Rv > 0, num_warps=nw)
            else:
                self._go(where, "_fr_d1", _fr_d1, (S,), *tabs, sel, kid, dix, dhb, fz["dstart"], fz["dcnt"],
                         fz["dorder"], fz["post"], fz["usg2"], sd["ne"], sd["rec"], fz["usgd"], self.err, cntd4,
                         H, R, k, NBLK, NPOS, svc.C, Rd, fz["maxrd"], S, Rd * H,
                         KP=KP, NP=NP, HAS_V=Rv > 0, num_warps=nw)
            self._go(where, "_fr_scan.d", _fr_scan, (1,), sd["ne"], sd["rec"], fz["usgd"], sd["bne"], sd["brec"],
                     sd["misc"], Rd * H, S, svc.C, svc.X, svc.R_ver,
                     BLOCK=SCAN_BLOCK, num_warps=4)                                    # [stack2] a constant
            if v3:
                self._go(where, "_fr3_d2", _fr3_d2, (S,), sel, kid, vix, vhb, fz["vstart"], fz["vcnt"], fz["vorder"],
                         dix, dhb, dtl, dnt, fz["dstart"], fz["dcnt"], fz["dorder"],
                         fz["tab"], fz["flist"], fz["pm"], sd["ne"], sd["rec"], sd["bne"], sd["brec"], sd["misc"],
                         fz["usgd"], td, dpage, dnpg, page, npg, drec, dcnt, cntd4, selw, rowsel, self.err,
                         self.stats, sb, cis,
                         H, B, R, k, NBLK, NPOS, svc.C, max(Rv, 1), Rd, fz["maxrv"], fz["maxrd"], nd,
                         caps[1], S, WB, N_TAIL, P, svc.bs, rows_s, rows_c,
                         KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, ZC=FR3_ZC, BSP=max(_pow2(svc.bs), 2), HAS_V=Rv > 0,
                         BIAS=bias, PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
            else:
                self._go(where, "_fr_d2", _fr_d2, (S,), *tabs, sel, kid, vix, vhb, fz["vstart"], fz["vcnt"],
                         fz["vorder"], dix, dhb, dtl, dnt, fz["dstart"], fz["dcnt"], fz["dorder"],
                         fz["post"], sd["ne"], sd["rec"], sd["bne"], sd["brec"], sd["misc"], fz["usgd"],
                         td, dpage, dnpg, page, npg, drec, dcnt, cntd4, selw, rowsel, self.err, self.stats, sb, cis,
                         H, B, R, k, NBLK, NPOS, svc.C, max(Rv, 1), Rd, fz["maxrv"], fz["maxrd"], nd,
                         caps[1], S, WB, N_TAIL, P, svc.bs, rows_s, rows_c,
                         KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, BSP=max(_pow2(svc.bs), 2), HAS_V=Rv > 0,
                         BIAS=bias, PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
            self.launches += 3
            ddec = DraftDecision(Rd, H, k, {f: td[i] for i, f in enumerate(FIELDS)}, dpage, dnpg, cntd4[:3])
            rec.update(drec=drec, dcnt=dcnt)
            if self.xp is not None and self._lean_on():               # [flean] the same binds and bias rows, lean
                bbd, hhd = self._bhd(fz, k)
                self.xp.bind_lean(l, bbd, hhd, ddec.t["take_slot"], ddec.t["granted"], svc._lean_c,
                                  base=self._xbase(fz, "d", k))
                if flat_bias:
                    XP.draft_bias_fill_flat_lean(self.xp, l, svc.store_b[l], svc.cis[l], ddec, bbd, hhd, svc.bs,
                                                 svc._lean_c)
            elif self.xp is not None:                                 # [integ][xpool] recipe (3): the draft takes'
                bbd, hhd = self._bhd(fz, k)                           # lanes, then their bias rows on the lanes
                self.xp.bind(l, bbd, hhd, ddec.t["take_slot"], ddec.t["granted"] != 0)
                if flat_bias:
                    XP.draft_bias_fill_flat(self.xp, l, svc.store_b[l], svc.cis[l], ddec, bbd, hhd, svc.bs)
        rec.update(sel=selw, rowsel=rowsel)
        self.layers += 1
        step = dict(l=l, vrows=[rows[i] for i in tk["vrows"]], vidx=tk["vrows"], vdec=vdec,
                    drows=[rows[i] for i in tk["drows"]], ddec=ddec, rec=rec, caps=caps)
        if ref is not None:
            self._compare(where, l, step, page, npg, sb_before, ref)
        return step, page, npg

    def _bias_args(self, l, sel):
        """D2's bias arguments for layer l (BIAS, sb, cis, rows_s, rows_c, flat_bias); no side effect. flat_bias: the
        xpool's flat fill does the bias rows (D2 BIAS 0, as without tables). BIAS 1: one 2- or 4-byte dtype (bit
        copies); BIAS 2: a cast in the kernel. Non-contiguous or non-(B, rows, H) tables are refused."""
        svc = self.svc
        flat_bias = self.xp is not None and svc.store_b is not None and svc.cis is not None   # [integ][xpool]
        if flat_bias or svc.store_b is None or svc.cis is None:
            return 0, sel, sel, 0, 0, flat_bias
        sb_t, cis_t = svc.store_b[l], svc.cis[l]
        if (not sb_t.is_contiguous() or not cis_t.is_contiguous() or sb_t.dim() != 3 or cis_t.dim() != 3
                or not sb_t.dtype.is_floating_point or not cis_t.dtype.is_floating_point):
            raise ValueError("resolve='fused': the bias rows need contiguous (B, rows, H) float store_b and "
                             "cis tables (got %s %s / %s %s)" % (sb_t.dtype, tuple(sb_t.shape), cis_t.dtype,
                                                                 tuple(cis_t.shape)))
        if sb_t.dtype == cis_t.dtype and sb_t.element_size() in (2, 4):
            return 1, _bits(sb_t), _bits(cis_t), sb_t.shape[1], cis_t.shape[1], False   # bit copies
        return 2, sb_t, cis_t, sb_t.shape[1], cis_t.shape[1], False                     # m9: the cast in the kernel

    # ----------------------------------------------------------------------------------------------- warm-up
    def warm_up(self, l, sel, NP, k):
        """[stack2] Compiles every variant that a layer of selection width k can launch, before its first decision:
        V1, SCAN, V2 with LAST True and False, D1 and D2 with HAS_V True and False (BIAS, PLANT, NP, KP, PP, ZP and the
        slot field are fixed per run and width). The launches have NO active row (MAXRV = MAXRD = 0) and write only
        scratch tensors made here (err and stats included); they read layer l's mirror tables and sel. The compile
        key of each launch equals a real launch's: the same constexprs and num_warps, the same dtypes, every int in
        do_not_specialize, and every pointer either in do_not_specialize or a fresh (16-byte aligned) allocation, as
        the real outputs are. Once per key (self._warmed); warm_launches counts them (not self.launches)."""
        svc = self.svc
        B, H, dev = svc.B, svc.H, svc.device
        S = B * H
        NPOS, NBLK = svc.n_hist, svc.nblk
        SB = slot_bits(NP)
        P = k + N_TAIL
        KP, PP, TP, ZP = max(_pow2(k), 2), max(_pow2(P), 2), max(_pow2(N_TAIL), 2), max(_pow2(NBLK), 64)
        WB, WP = (NBLK + 31) // 32, (NPOS + 31) // 32
        bias, sb, cis, rows_s, rows_c, _ = self._bias_args(l, sel)
        key = (k, NP, bias, sb.dtype, cis.dtype)
        if key in self._warmed:
            return
        mem0 = None
        if not _LMEM_PREP[0] and torch.device(dev).type == "cuda":   # [oomfix] once per process (the preflight:
            torch.cuda.synchronize()                                 # untimed); later engines skip it
            torch.cuda.empty_cache()
            mem0 = torch.cuda.mem_get_info()[0]
            _LMEM_PREP[0] = True
        dm = svc.dmirror
        tabs = (dm.code[l], dm.block[l], dm.gen[l], dm.cur[l], dm.prot[l], dm.stamp[l])
        R = int(sel.shape[1])
        nw = self.num_warps

        def z(*shape, dtype=torch.int64):                             # LANE_PAD spare elements, as _buf (fusedres2b)
            n = 1
            for s in shape:
                n *= s
            return torch.zeros((n + LANE_PAD,), dtype=dtype, device=dev)[:n].view(shape)
        rowv = (z(1), z(1), z(1, N_TAIL), z(1), z(1))                 # (rb, hb, tails, nt, ix): never read (no row)
        cnts = (z(B), z(B), z(1))                                     # (start, count, order) per batch slot: 0 rows
        kid = z(max(R, 1))
        err, stats = z(), z(NSTAT_LANES)
        ne, rec, bne, brec, misc = z(1), z(1), z(1), z(1), z(4)
        usg, usg2, usgd, post = z(S), z(S), z(S), z(4, S, NP)
        tv, vpage, vnpg = z(len(FIELDS_V), 1), z(1, H, P), z(1, H)
        page, npg = z(max(R, 1), H, P, dtype=torch.int32), z(max(R, 1), H, dtype=torch.int32)
        ids, vrec, vcnt, cntv, vlook, vread = z(H, B, NPOS), z(len(VREC), 1), z(1), z(1), z(1, 4), z(B, H, WP)
        selw, rowsel = z(B, H, WB), z(max(R, 1), H, WB)
        td, dpage, dnpg, drec, dcnt, cntd4 = z(len(FIELDS), 1), z(1, H, P), z(1, H), z(len(DREC), 1), z(1), z(4)
        _warm_depth[0] += 1
        try:
            if self.kernel == "v3":                                   # [fr3] the v3 variants: PREP (HAS_V both), V1,
                CH = fr3_chunk(KP, NP)                                # SCAN, V2 (LAST both), D1 / D2 (HAS_V both)
                tab, flist, pm = z(S, 7, ZP), z(S, 2, NP), z(S, 4)
                for has_v in (True, False):
                    _launch(_fr3_prep, (S,), dm.code[l], dm.block[l], dm.cur[l], dm.prot[l], dm.stamp[l], sel, kid,
                            rowv[4], rowv[1], *cnts, rowv[4], rowv[1], *cnts, tab, flist, pm, post, usg, err,
                            H, R, k, NBLK, NPOS, 1, 1, 0, 0, S,
                            KP=KP, NP=NP, ZP=ZP, CH=CH, ZC=FR3_ZC, TC=FR3_TC, HAS_V=has_v, PE_INFL=self.pe_inflight,
                            SMUL=1 << SB, CMUL=1 << (STAMP_BITS + SB), num_warps=nw)
                    self.warm_launches += 1
                _launch(_fr3_v1, (S,), sel, kid, rowv[4], rowv[1], *cnts, tab, ne, rec, usg, err, cntv, vlook,
                        H, R, k, NBLK, svc.C, 1, 0, 0, KP=KP, ZP=ZP, LZ=LOOK_LANES, num_warps=nw)
                _launch(_fr_scan, (1,), ne, rec, usg, bne, brec, misc, 0, S, svc.C, svc.X, 0, BLOCK=SCAN_BLOCK,
                        num_warps=4)
                self.warm_launches += 2
                for last in (True, False):
                    _launch(_fr3_v2, (S,), dm.code[l], dm.block[l], dm.gen[l], dm.cur[l], sel, kid,
                            rowv[4], rowv[1], rowv[2], rowv[3], *cnts, tab, flist, pm, ne, rec, bne, brec, misc, usg,
                            tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, post, usg2, selw,
                            rowsel, err, stats,
                            H, B, R, k, NBLK, NPOS, svc.C, 1, 0, 1, 1, S, WB, WP, N_TAIL, P,
                            KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, CH=CH, ZC=FR3_ZC, LAST=last, PLANT=self.plant,
                            NSTAT=len(STATS), CMUL=1 << (STAMP_BITS + SB), SMASK=(1 << SB) - 1, num_warps=nw)
                    self.warm_launches += 1
                for has_v in (True, False):
                    _launch(_fr3_d1, (S,), dm.code[l], sel, kid, rowv[4], rowv[1], *cnts, tab, flist, pm, post, usg,
                            usg2, ne, rec, usgd, err, cntd4, H, R, k, NBLK, NPOS, svc.C, 1, 0, S, 0,
                            KP=KP, NP=NP, ZP=ZP, CH=CH, HAS_V=has_v, num_warps=nw)
                    _launch(_fr3_d2, (S,), sel, kid, rowv[4], rowv[1], *cnts, rowv[4], rowv[1], rowv[2], rowv[3],
                            *cnts, tab, flist, pm, ne, rec, bne, brec, misc, usgd,
                            td, dpage, dnpg, page, npg, drec, dcnt, cntd4, selw, rowsel, err, stats, sb, cis,
                            H, B, R, k, NBLK, NPOS, svc.C, 1, 1, 0, 0, 1, 1, S, WB, N_TAIL, P, svc.bs, rows_s,
                            rows_c, KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, ZC=FR3_ZC, BSP=max(_pow2(svc.bs), 2),
                            HAS_V=has_v, BIAS=bias, PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
                    self.warm_launches += 2
            else:
                _launch(_fr_v1, (S,), *tabs, sel, kid, rowv[4], rowv[1], *cnts, ne, rec, usg, err, cntv, vlook,
                        H, R, k, NBLK, NPOS, svc.C, 1, 0, 0, KP=KP, NP=NP, LZ=LOOK_LANES, num_warps=nw)
                _launch(_fr_scan, (1,), ne, rec, usg, bne, brec, misc, 0, S, svc.C, svc.X, 0, BLOCK=SCAN_BLOCK,
                        num_warps=4)
                self.warm_launches += 2
                for last in (True, False):
                    _launch(_fr_v2, (S,), *tabs, sel, kid, rowv[4], rowv[1], rowv[2], rowv[3], *cnts,
                            rowv[4], rowv[1], *cnts, ne, rec, bne, brec, misc,
                            tv, vpage, vnpg, page, npg, ids, vrec, vcnt, cntv, vlook, vread, post, usg2, selw,
                            rowsel, err, stats,
                            H, B, R, k, NBLK, NPOS, svc.C, svc.X, 1, 1, 0, 0, 1, 1, S, WB, WP, N_TAIL, P,
                            KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, PE_INFL=self.pe_inflight, LAST=last, PLANT=self.plant,
                            NSTAT=len(STATS), SMUL=1 << SB, CMUL=1 << (STAMP_BITS + SB), SMASK=(1 << SB) - 1,
                            num_warps=nw)
                    self.warm_launches += 1
                for has_v in (True, False):
                    _launch(_fr_d1, (S,), *tabs, sel, kid, rowv[4], rowv[1], *cnts, post, usg2, ne, rec, usgd, err,
                            cntd4, H, R, k, NBLK, NPOS, svc.C, 1, 0, S, 0, KP=KP, NP=NP, HAS_V=has_v, num_warps=nw)
                    _launch(_fr_d2, (S,), *tabs, sel, kid, rowv[4], rowv[1], *cnts, rowv[4], rowv[1], rowv[2],
                            rowv[3], *cnts, post, ne, rec, bne, brec, misc, usgd,
                            td, dpage, dnpg, page, npg, drec, dcnt, cntd4, selw, rowsel, err, stats, sb, cis,
                            H, B, R, k, NBLK, NPOS, svc.C, 1, 1, 0, 0, 1, 1, S, WB, N_TAIL, P, svc.bs, rows_s, rows_c,
                            KP=KP, NP=NP, PP=PP, TP=TP, ZP=ZP, BSP=max(_pow2(svc.bs), 2), HAS_V=has_v, BIAS=bias,
                            PLANT=self.plant, NSTAT=len(STATS), num_warps=nw)
                    self.warm_launches += 2
        finally:
            _warm_depth[0] -= 1
        if mem0 is not None:                                         # [oomfix] free before - free after = the warm
            torch.cuda.synchronize()                                 # launches' local memory + modules
            self.warm_mem_gb = [round(mem0 / 1e9, 3), round(torch.cuda.mem_get_info()[0] / 1e9, 3)]
            print("[oomfix] fused warm_up free GB before/after", self.warm_mem_gb, file=sys.stderr, flush=True)
        self._warmed.add(key)

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
        self.trace.append((int(where[0]), int(where[1]), self.stats[:len(STATS)].to("cpu").tolist(), int(self.err)))

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
        args = ref["args"]
        if self.xp_ref is not None:                                   # [integ3] M5: S-C's binds on the shadow pool
            xr, rv, rd = self.xp_ref, ref["vdec"], ref["ddec"]        # (ResidencySC.resolve's order and masks)
            if rv is not None:
                xr.bind(l, rv.bh[0], rv.bh[1], rv.t["take_slot"], (rv.t["granted"] | rv.t["evict"]) != 0)
            if rd is not None:
                xr.bind(l, args[7], args[8], rd.t["take_slot"], rd.t["granted"] != 0)
            cmp("xpool.pos_of", self.xp.pos_of[l], xr.pos_of[l])
            cmp("xpool.occ", self.xp.occ[l], xr.occ[l])
        if sb_before is not None and ref["ddec"] is not None:
            if self.xp is not None:                                   # [integ3] the torch path's flat fill on the
                XP.draft_bias_fill_flat(self.xp_ref, l, sb_before, svc.cis[l], ref["ddec"], args[7], args[8],  # S-C
                                        svc.bs)                       # lanes (the shadow), not on the fused ones
            else:
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
        return dict(mode="check" if self.check else "fused", kernel=self.kernel, plant=self.plant,   # [fr3] kernel
                    launches=self.launches,
                    layers=self.layers, launches_per_layer=self.launches / max(1, self.layers),
                    bias_casts=self.bias_casts, sel_conversions=self.sel_conversions,
                    max_rows_per_b=list(self.max_rows_per_b), compared=self.compared, diffs=len(self.diffs),
                    first_diffs=[list(d) for d in self.diffs[:5]],
                    first_diff_at=list(self.diffs[0][:2]) if self.diffs else None, first_effective=first_eff,
                    stats=stats, gate={s: ("gated" if stats[s] > 0 else "NOT GATED") for s in GATE_STATS},
                    err=err_any, err_bits=[m for b, m in sorted(ERR_BITS.items()) if err_any & b],
                    guard=self.guard, guard_checks=self.guard_checks, warm=self.warm, warmed=len(self._warmed),
                    warm_launches=self.warm_launches, guard_hits=[list(x) for x in self.guard_hits],
                    warm_mem_gb=self.warm_mem_gb, compiled=compiled_variants())


def compiled_variants():
    """REPORT (m4): the compiled variants per kernel of this module (Triton's JIT cache per device; None where the
    installed Triton keeps no such cache, e.g. the interpreter). A variant compiled inside a timed span inflates it."""
    out = {}
    if triton is None:
        return out
    for name in ("_fr_v1", "_fr_v2", "_fr_d1", "_fr_d2", "_fr_scan",
                 "_fr3_prep", "_fr3_v1", "_fr3_v2", "_fr3_d1", "_fr3_d2"):                  # [fr3]
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
