"""R2 step 7, piece 5, (B1): S-C with the SPARSE DECISION RECORD (docs/R2_STEP7_FETCH_RETENTION.md, note 12l).

S-C (r2_residency_sc.py) reads EVERY decision field of every (row, head, selected block) entry of every layer at
end_tick and applies them in Python: at B 64 about 0.8 M verify and 0.26 M draft entries per tick, seconds of host
time (SG5 p3: S-B's draft-only ticks 7.06 s of backend host time). This class keeps S-C's device decisions and its
ownership objects (the vendored K3X streams), and changes only WHAT the device hands the host and HOW the host
applies it.

THE CUT. Per layer, on the device (_layer_record, device ops only), after device_verify / device_draft:
  - VERIFY RECORDS: the entries whose host action is not "a ready read of a resident block": an urgent take (K5
    grant or Policy E eviction, both kinds), an in-flight merge (infl / dup), a NEW protection pin, a hole (an entry
    with no slot that the device did not flag: refused at apply). Ready entries are not records.
  - DRAFT RECORDS: every ready entry (its LRU touch / acquisition and its read are host actions, in K3X order O1),
    an in-flight merge, a take, a refusal, a new pin, a hole.
  - NEW PINS ONLY. A protection (block, cohort key, kind) is a host action once: the device keeps a PIN TABLE
    (L, B, H, nblk + 1, 2 kinds) of (key id << 32 | epoch) and records a pin only when the table does not already
    hold it. The host bumps an EPOCH per (layer, batch slot) whenever it drops references there (_drop, end_request)
    or voids a tick; a table entry is valid only for the current epoch. So a pin the host may have lost is always
    recorded again (a redundant record is an idempotent protect; a missing one is impossible).
  - The VERIFY READ BITMAP (B, H, C + X) packed in 32-bit words: every slot a verify row reads (ready, in flight, an
    urgent take): the bulk first / last reader update. The draft reads come with the draft records.
  - The VERIFY LOOK COUNTS per verify row: the ready FIRST looks of each (key, layer) call (dedup by (key, head,
    block) in entry order) and their 4.6 classes (LRU / PO / VRES); the in_flight / new first looks are records
    (flag FIRST).
  - The LAYER SELECTION BITMAP (B, H, nblk) packed: every block a row of the layer selects; the host decodes it only
    for a stream with a Policy E eviction (its candidate check).
  - ROWSEL (R, H, ceil(nblk / 32)) packed, per TICK row (B2's request): every valid historical block of that row's
    selection at each head. It stays ON THE DEVICE in step["rec"]["rowsel"] for a device consumer (B2); the host
    consumer below does not read it, and it is not in the one device-to-host read.
  - pin_records=False (B2 as the consumer): no pin is recorded (no PIN flag, no pin table update), so an entry that
    would be a record only for its new pin is no record. This class's consumer protects only from pins, so it refuses
    to run on such a record. pin_only_records counts the PIN-only records the default consumer applies (the saving).
  - The BIAS ROWS of the granted draft takes: written on the device here (one batched write per layer), so the host
    take does not copy them (Q-C4).
  Each record list is COMPACTED with a prefix sum (stable: the entry order (row, head, ascending block) is K3X's)
  into a fixed-capacity buffer (record_cap; None = every entry, which cannot overflow) and a count.

THE HOST CONSUMER (apply_tick, the same signature as S-C's: the capture seam's one call): ONE device-to-host read
of the flags, the counters and the records; the void-tick refusals (a set flag, a verifier overflow, a record
over its capacity) before any change of the host state; then per layer, in v1's order, the verify records (pins,
merges, takes with Policy E's host check), the verify looks (device counts + records), the verify read bitmap, the
draft records (pins, ready reads, merges, takes, refusals), the draft reads, the draft recency (touches applied at
C speed: OrderedDict.move_to_end, with S-C's journal effects (counts, stamps, dirty marks) in bulk; acquisitions and
K1's evict_to_c through the vendored code). Per-record INVARIANT CHECKS replace the dense replay's checks of every
entry: a ready draft record names the host's resident slot; a merge names the owner's transfer in flight (and its
slot / generation); a take passes v1's _take checks (K5, first_free, the evicted slot FREE); a Policy E eviction is
the host's next candidate; a refusal is one K5 refuses; a hole raises. A ready VERIFY entry is not checked on the
host (the CPU lockstep R5a proves sparse == dense == v1).

STATE EQUALITY with the dense S-C: the ownership state (every stream's LRU order, PO, VRES, pending map, protections
with their kinds, slots), the in-flight transfers, the counters (4.3, 4.6, refusals, masked, transfers) and the
device mirror are equal after every tick (R5a). Differences by design: Stream.log does not receive the draft touch
events (their effects are applied directly); the lifetimes' verify read counts (4.8 rv) count a slot once per layer
instead of once per entry (4.8 reads them as > 0 only).

NEGATIVE CONTROLS (sparse_mut, tests only): drop=<kind> (the first state-changing record of that kind is dropped:
v_pin, v_merge, v_take, v_read, d_pin, d_ready, d_merge, d_take, d_refused), swap_takes (two draft takes of one
stream exchange their slots), no_cap_check (a record over its capacity is truncated, not refused), rowsel_by_entry
(rowsel indexed by the entry row of the verify / draft decision instead of the tick row).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F_nn

from .kstep.errors import CapacityRefusal, OwnershipViolation
from .kstep.executor import DRAFT, VERIFY
from . import r2_xpool as XP
from .r2_residency import Plan, Row
from .r2_residency_sc import C_LRU, C_PO, C_VRES, ResidencySC
from .r2_residency_sb import _nz                                     # [resfix] --res-lean
from . import r2_resdev as RD                                        # [resdev] --res-dev

# the record fields, in their buffer order
VREC = ("o", "blk", "flags", "slot", "infl_slot", "infl_gen", "rslot", "take_slot", "evict_blk")
DREC = ("o", "blk", "flags", "slot", "take_slot")
PIN, READY, INFL, DUP, GRANT, EVICT, EVICT_INFL, FIRST, HOLE, REFUSED = (1 << i for i in range(10))
TWO32 = 1 << 32
BIGKEY = 1 << 62


def _first_in_group(key: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the first entry of each group of equal keys, in entry order (a stable sort)."""
    ks, perm = torch.sort(key, stable=True)
    first_sorted = torch.cat([torch.ones_like(ks[:1], dtype=torch.bool), ks[1:] != ks[:-1]])
    out = torch.empty_like(first_sorted)
    out[perm] = first_sorted
    return out


def _compact(mask: torch.Tensor, fields: List[torch.Tensor], cap: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The masked entries' fields, in entry order (a prefix sum: stable), into a fixed (F, cap) buffer (-1 padded),
    and their count (1,) (which may exceed cap: the consumer refuses that). Device ops only, fixed shapes."""
    m64 = mask.to(torch.int64)
    pos = torch.cumsum(m64, 0) - 1
    cnt = m64.sum().reshape(1)
    dest = torch.where(mask & (pos < cap), pos, torch.full_like(pos, cap))
    src = torch.stack(fields)                                            # (F, N)
    out = torch.full((len(fields), cap + 1), -1, dtype=torch.int64, device=mask.device)
    out.scatter_(1, dest.view(1, -1).expand(len(fields), -1), src)      # the unmasked go to the dummy column cap
    return out[:, :cap], cnt


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """(..., n) bool -> (..., ceil(n / 32)) int64 words, bit i of word w = element 32 w + i."""
    n = bits.shape[-1]
    W = (n + 31) // 32
    b = F_nn.pad(bits.to(torch.int64), (0, W * 32 - n)).view(*bits.shape[:-1], W, 32)
    w = torch.bitwise_left_shift(torch.ones(32, dtype=torch.int64, device=bits.device),
                                 torch.arange(32, dtype=torch.int64, device=bits.device))
    return (b * w).sum(-1)


def unpack_bits(words: torch.Tensor, n: int) -> torch.Tensor:
    """The host side of _pack_bits: (..., W) int64 -> (..., n) bool (CPU tensor ops)."""
    sh = torch.arange(32, dtype=torch.int64)
    bits = torch.bitwise_and(torch.bitwise_right_shift(words.unsqueeze(-1), sh), 1) != 0
    return bits.reshape(*words.shape[:-1], -1)[..., :n]


def sparse_layer_record(pins_l, epoch_l, code_l, vdec, kid_v, ddec, kid_d, bbd, hhd, NBLK: int, NPOS: int,
                        cap_v: int, cap_d: int, ixv=None, ixd=None, R: int = 0, pins: bool = True,
                        rowsel_by_entry: bool = False) -> Dict[str, torch.Tensor]:
    """One layer's sparse record on the device (module docstring). pins_l (B, H, NBLK + 1, 2) the layer's pin table
    (updated in place), epoch_l (B,), code_l (B, H, NPOS) the mirror's state codes; vdec / ddec the layer's decisions
    (or None), kid_v / kid_d their rows' key ids, bbd / hhd the draft entries' batch slot / head. Device ops only.
    R > 0: also ROWSEL (R, H, ceil(NBLK / 32)) int64 words in _pack_bits' format: bit z of row i, head h = block z is
    a valid historical selection of TICK row i at head h (0 <= z < hist_bound, z < NBLK); ixv / ixd (Rv,) / (Rd,) map
    the verify / draft entry rows to tick rows. It stays on the device (the host consumer does not read it).
    pins False (B2 as the consumer, pin_records=False): no pin is recorded and the pin table is not touched; an entry
    that would be a record only for its new pin is no record. rowsel_by_entry: NEGATIVE CONTROL (tests only), rowsel
    indexed by the entry row instead of the tick row."""
    out: Dict[str, torch.Tensor] = {}
    dev = pins_l.device
    B, H = pins_l.shape[0], pins_l.shape[1]
    selm = torch.zeros((B, H, NBLK + 1), dtype=torch.bool, device=dev)
    rsel = torch.zeros((R, H, NBLK + 1), dtype=torch.bool, device=dev) if R > 0 else None
    if vdec is not None:
        t = vdec.t
        Rv, k = vdec.Rv, vdec.k
        N = Rv * H * k
        o = torch.arange(N, device=dev)
        j = torch.div(o, H * k, rounding_mode="floor")
        bb, hh = vdec.bh
        valid, ready, infl, dup = t["valid"] != 0, t["ready"] != 0, t["infl"] != 0, t["dup"] != 0
        gr, ev, evf, prot = t["granted"] != 0, t["evict"] != 0, t["evict_infl"] != 0, t["protect"] != 0
        blk = t["blk"]
        kid = kid_v[j]
        pinning = valid & prot
        blk_p = torch.where(pinning, blk, torch.full_like(blk, NBLK))         # the dummy row takes the others
        one = torch.ones_like(blk)
        if pins:
            pk = kid * TWO32 + epoch_l[bb]
            newpin = pinning & (pins_l[bb, hh, blk_p, one] != pk)
            pins_l.index_put_((bb, hh, blk_p, one), pk)
        else:
            newpin = torch.zeros_like(pinning)
        if rsel is not None:                                                  # the tick row of each entry
            rsel[j if rowsel_by_entry else ixv[j], hh, torch.where(valid, blk, torch.full_like(blk, NBLK))] = True
        fkey = torch.where(valid, (kid * H + hh) * (NBLK + 1) + blk, torch.full_like(blk, BIGKEY))
        first = _first_in_group(fkey) & valid
        hole = valid & ~(ready | infl | dup | gr | ev)
        mask = valid & (newpin | infl | dup | gr | ev | hole)
        flags = (newpin * PIN + ready * READY + infl * INFL + dup * DUP + gr * GRANT + ev * EVICT + evf * EVICT_INFL
                 + first * FIRST + hole * HOLE).to(torch.int64)
        out["vrec"], out["vcnt"] = _compact(mask, [o, blk, flags, t["slot"], t["infl_slot"], t["infl_gen"],
                                                   t["rslot"], t["take_slot"], t["evict_blk"]], cap_v)
        c = code_l[bb, hh, t["slot"].clamp(min=0)]
        rf = ready & first
        vals = torch.stack([rf, rf & (c == C_LRU), rf & (c == C_PO), rf & (c == C_VRES)], -1).to(torch.int64)
        out["vlook"] = torch.zeros((Rv, 4), dtype=torch.int64, device=dev).index_add_(0, j, vals)
        rs = torch.where(ready, t["slot"], torch.where(infl, t["infl_slot"], torch.where(
            dup, t["rslot"], torch.where(gr | ev, t["take_slot"], torch.full_like(blk, -1)))))
        rd = valid & (rs >= 0)
        rbits = torch.zeros((B, H, NPOS + 1), dtype=torch.bool, device=dev)
        rbits[bb, hh, torch.where(rd, rs, torch.full_like(rs, NPOS))] = True
        out["vread"] = _pack_bits(rbits[..., :NPOS])
        selm[bb, hh, torch.where(valid, blk, torch.full_like(blk, NBLK))] = True
    if ddec is not None:
        t = ddec.t
        Rd, k = ddec.Rd, ddec.k
        N = Rd * H * k
        o = torch.arange(N, device=dev)
        j = torch.div(o, H * k, rounding_mode="floor")
        valid, ready, infl = t["valid"] != 0, t["ready"] != 0, t["infl"] != 0
        gr, refused, prot = t["granted"] != 0, t["refused"] != 0, t["protect"] != 0
        blk = t["blk"]
        kid = kid_d[j]
        pinning = valid & prot
        blk_p = torch.where(pinning, blk, torch.full_like(blk, NBLK))
        zero = torch.zeros_like(blk)
        if pins:
            pk = kid * TWO32 + epoch_l[bbd]
            newpin = pinning & (pins_l[bbd, hhd, blk_p, zero] != pk)
            pins_l.index_put_((bbd, hhd, blk_p, zero), pk)
        else:
            newpin = torch.zeros_like(pinning)
        if rsel is not None:
            rsel[j if rowsel_by_entry else ixd[j], hhd, torch.where(valid, blk, torch.full_like(blk, NBLK))] = True
        hole = valid & ~(ready | infl | gr | refused)
        mask = valid & (newpin | ready | infl | gr | refused | hole)
        flags = (newpin * PIN + ready * READY + infl * INFL + gr * GRANT + refused * REFUSED
                 + hole * HOLE).to(torch.int64)
        out["drec"], out["dcnt"] = _compact(mask, [o, blk, flags, t["slot"], t["take_slot"]], cap_d)
        selm[bbd, hhd, torch.where(valid, blk, torch.full_like(blk, NBLK))] = True
    out["sel"] = _pack_bits(selm[..., :NBLK])
    if rsel is not None:
        out["rowsel"] = _pack_bits(rsel[..., :NBLK])
    return out


def draft_bias_fill(store_b_l, cis_l, ddec, bbd, hhd, bs: int) -> None:
    """The bias rows of one layer's granted draft takes, written on the device in ONE batched write (Q-C4): slot
    rows take_slot * bs .. + bs of (b, h) get the cis rows block * bs .. + bs. Entries that are not granted write a
    scratch row only (no race with a granted one). Device ops only."""
    t = ddec.t
    g = t["granted"] != 0
    N = g.shape[0]
    rows = store_b_l.shape[1]
    r = torch.arange(bs, device=g.device).view(1, bs)
    dst = torch.where(g.view(N, 1), t["take_slot"].clamp(min=0).view(N, 1) * bs + r, torch.full_like(r, rows))
    src = t["blk"].clamp(min=0).view(N, 1) * bs + r
    b2, h2 = bbd.view(N, 1).expand(N, bs), hhd.view(N, 1).expand(N, bs)
    buf = torch.zeros((store_b_l.shape[0], rows + 1, store_b_l.shape[2]), dtype=store_b_l.dtype, device=g.device)
    hit = torch.zeros((store_b_l.shape[0], rows + 1, store_b_l.shape[2]), dtype=torch.bool, device=g.device)
    buf[b2, dst, h2] = cis_l[b2, src, h2].to(store_b_l.dtype)
    hit[b2, dst, h2] = True
    store_b_l[...] = torch.where(hit[:, :rows], buf[:, :rows], store_b_l)


# ====================================================================================================================
# [resfix] --res-lean: the record's lean copies (the functions above stay the reference, unchanged)
# ====================================================================================================================
def _first_in_group_lean(key: torch.Tensor, cst) -> torch.Tensor:
    """[resfix] _first_in_group with the constant True from cst."""
    ks, perm = torch.sort(key, stable=True)
    first_sorted = torch.cat([cst.full_like(ks[:1], True, dtype=torch.bool), ks[1:] != ks[:-1]])
    out = torch.empty_like(first_sorted)
    out[perm] = first_sorted
    return out


def _compact_lean(mask: torch.Tensor, fields: List[torch.Tensor], cap: int, cst) -> Tuple[torch.Tensor, torch.Tensor]:
    """[resfix] _compact with the dummy-column fill from cst (the output buffer stays fresh)."""
    m64 = mask.to(torch.int64)
    pos = torch.cumsum(m64, 0) - 1
    cnt = m64.sum().reshape(1)
    src = torch.stack(fields)                                            # (F, N)
    if cst.dev:                                 # [resdev] an entry that is not recorded goes to its OWN dummy column
        n = pos.shape[0]                        # cap + i (the lean op sends them all to column cap: one element)
        dest = RD.spread(pos, mask & (pos < cap), cap, cst.ar(n))
        out = torch.full((len(fields), cap + n), -1, dtype=torch.int64, device=mask.device)
        out.scatter_(1, dest.view(1, -1).expand(len(fields), -1), src)
        return out[:, :cap], cnt
    dest = torch.where(mask & (pos < cap), pos, cst.full_like(pos, cap))
    out = torch.full((len(fields), cap + 1), -1, dtype=torch.int64, device=mask.device)
    out.scatter_(1, dest.view(1, -1).expand(len(fields), -1), src)      # the unmasked go to the dummy column cap
    return out[:, :cap], cnt


def _pack_bits_lean(bits: torch.Tensor, cst) -> torch.Tensor:
    """[resfix] _pack_bits with its bit weights from cst (cst.w32 = 1 << arange(32))."""
    n = bits.shape[-1]
    W = (n + 31) // 32
    b = F_nn.pad(bits.to(torch.int64), (0, W * 32 - n)).view(*bits.shape[:-1], W, 32)
    return (b * cst.w32).sum(-1)


def sparse_layer_record_lean(code_l, vdec, ddec, B: int, H: int, NBLK: int, NPOS: int, cap_v: int, cap_d: int,
                             R: int, cst, rcv=None, rcd=None) -> Dict[str, torch.Tensor]:
    """[resfix] sparse_layer_record for --res-lean with pins=False (B2: pin_records=False; ResidencySparse refuses the
    lean record with pins) and rowsel by the tick row, the same values:
      - no hidden sync: the four `x[...] = True` scatters (rsel, rbits, selm x 2; DIAG_torch_res.md: sites :160, :177,
        :179, :200, :206) write cst.true, a 0-dim device tensor, with index_put_;
      - the tick constants of each part (rcv / rcd, ResidencySparse._lean_rec_c): o = arange(N), j = o // (H k), the
        tick rows rowt = ix[j], fbase = (kid[j] * H + h) * (NBLK + 1) (fkey's left part), and the draft entries' (b, h);
      - where(valid, blk, NBLK) is computed once per part (the record computes it for rsel and for selm);
      - a bool decision field is used as it is (_nz: x != 0 equals x).
    pins=False: no pin is recorded, newpin is all False (as in the reference), the pin table is not touched."""
    out: Dict[str, torch.Tensor] = {}
    dev = code_l.device
    fl = cst.full_like
    selm = torch.zeros((B, H, NBLK + 1), dtype=torch.bool, device=dev)
    rsel = torch.zeros((R, H, NBLK + 1), dtype=torch.bool, device=dev) if R > 0 else None
    if vdec is not None:
        t = vdec.t
        Rv = vdec.Rv
        o, j = rcv["o"], rcv["j"]
        bb, hh = vdec.bh
        valid, ready, infl, dup = _nz(t["valid"]), _nz(t["ready"]), _nz(t["infl"]), _nz(t["dup"])
        gr, ev, evf = _nz(t["granted"]), _nz(t["evict"]), _nz(t["evict_infl"])
        blk = t["blk"]
        newpin = fl(valid, False)                                             # pins False: no pin is recorded
        vblk = torch.where(valid, blk, fl(blk, NBLK))
        if rsel is not None:                                                  # the tick row of each entry
            rsel.index_put_((rcv["rowt"], hh, vblk), cst.true)
        top = rcv.get("fk_bound", -1) + 1 if cst.dev else -1          # [resdev] a valid key <= fk_bound (host bound)
        if 0 < top < RD.INT32_MAX:                                    # the invalid ones at top, not BIGKEY: the same
            fkey = torch.where(valid, rcv["fbase"] + blk, fl(blk, top))   # order, so the same stable permutation, with
            ks, perm = RD.sort_key32(fkey, top)                       # int32 radix keys
            fs = torch.cat([fl(ks[:1], True, dtype=torch.bool), ks[1:] != ks[:-1]])
            first = torch.empty_like(fs)
            first[perm] = fs
            first = first & valid
        else:
            fkey = torch.where(valid, rcv["fbase"] + blk, fl(blk, BIGKEY))
            first = _first_in_group_lean(fkey, cst) & valid
        hole = valid & ~(ready | infl | dup | gr | ev)
        mask = valid & (newpin | infl | dup | gr | ev | hole)
        flags = (newpin * PIN + ready * READY + infl * INFL + dup * DUP + gr * GRANT + ev * EVICT + evf * EVICT_INFL
                 + first * FIRST + hole * HOLE).to(torch.int64)
        out["vrec"], out["vcnt"] = _compact_lean(mask, [o, blk, flags, t["slot"], t["infl_slot"], t["infl_gen"],
                                                        t["rslot"], t["take_slot"], t["evict_blk"]], cap_v, cst)
        c = code_l[bb, hh, t["slot"].clamp(min=0)]
        rf = ready & first
        vals = torch.stack([rf, rf & (c == C_LRU), rf & (c == C_PO), rf & (c == C_VRES)], -1).to(torch.int64)
        out["vlook"] = torch.zeros((Rv, 4), dtype=torch.int64, device=dev).index_add_(0, j, vals)
        rs = torch.where(ready, t["slot"], torch.where(infl, t["infl_slot"], torch.where(
            dup, t["rslot"], torch.where(gr | ev, t["take_slot"], fl(blk, -1)))))
        rd = valid & (rs >= 0)
        rbits = torch.zeros((B, H, NPOS + 1), dtype=torch.bool, device=dev)
        rbits.index_put_((bb, hh, torch.where(rd, rs, fl(rs, NPOS))), cst.true)
        out["vread"] = _pack_bits_lean(rbits[..., :NPOS], cst)
        selm.index_put_((bb, hh, vblk), cst.true)
    if ddec is not None:
        t = ddec.t
        o = rcd["o"]
        bbd, hhd = rcd["bb_f"], rcd["hh_f"]
        valid, ready, infl = _nz(t["valid"]), _nz(t["ready"]), _nz(t["infl"])
        gr, refused = _nz(t["granted"]), _nz(t["refused"])
        blk = t["blk"]
        newpin = fl(valid, False)
        vblk = torch.where(valid, blk, fl(blk, NBLK))
        if rsel is not None:
            rsel.index_put_((rcd["rowt"], hhd, vblk), cst.true)
        hole = valid & ~(ready | infl | gr | refused)
        mask = valid & (newpin | ready | infl | gr | refused | hole)
        flags = (newpin * PIN + ready * READY + infl * INFL + gr * GRANT + refused * REFUSED
                 + hole * HOLE).to(torch.int64)
        out["drec"], out["dcnt"] = _compact_lean(mask, [o, blk, flags, t["slot"], t["take_slot"]], cap_d, cst)
        selm.index_put_((bbd, hhd, vblk), cst.true)
    out["sel"] = _pack_bits_lean(selm[..., :NBLK], cst)
    if rsel is not None:
        out["rowsel"] = _pack_bits_lean(rsel[..., :NBLK], cst)
    return out


def draft_bias_fill_lean(store_b_l, cis_l, ddec, bbd, hhd, bs: int, cst) -> None:
    """[resfix] draft_bias_fill (the per-stream store) for --res-lean, the same writes: the hit mask's True is
    cst.true (index_put_: no host copy, no sync; the reference's `hit[...] = True` is DIAG site :228)."""
    t = ddec.t
    g = _nz(t["granted"])
    N = g.shape[0]
    rows = store_b_l.shape[1]
    r = cst.ar(bs).view(1, bs)
    dst = torch.where(g.view(N, 1), t["take_slot"].clamp(min=0).view(N, 1) * bs + r, cst.full_like(r, rows))
    src = t["blk"].clamp(min=0).view(N, 1) * bs + r
    b2, h2 = bbd.view(N, 1).expand(N, bs), hhd.view(N, 1).expand(N, bs)
    buf = torch.zeros((store_b_l.shape[0], rows + 1, store_b_l.shape[2]), dtype=store_b_l.dtype, device=g.device)
    hit = torch.zeros((store_b_l.shape[0], rows + 1, store_b_l.shape[2]), dtype=torch.bool, device=g.device)
    buf[b2, dst, h2] = cis_l[b2, src, h2].to(store_b_l.dtype)
    hit.index_put_((b2, dst, h2), cst.true)
    store_b_l[...] = torch.where(hit[:, :rows], buf[:, :rows], store_b_l)


class ResidencySparse(ResidencySC):
    """S-C with the sparse decision record (module docstring). Extra arguments: record_cap (cap_v, cap_d) the record
    buffers per layer (None: every entry, no overflow), sparse_mut (negative controls, tests only)."""

    def __init__(self, *args, record_cap: Optional[Tuple[int, int]] = None, sparse_mut=None, pin_records: bool = True,
                 **kw):
        super().__init__(*args, **kw)
        self.record_cap = record_cap
        # pin_records False (B2 as the consumer): the record carries no protection pins. This class's own consumer
        # (apply_tick) protects only from them, so it refuses such a record; a consumer that keeps the protection
        # itself (B2's table, from rowsel) replaces apply_tick
        self.pin_records = bool(pin_records)
        self.sparse_mut = dict(sparse_mut or {})
        self.pins = torch.zeros((self.L, self.B, self.H, self.nblk + 1, 2), dtype=torch.int64, device=self.device)
        self._kids: Dict[Any, int] = {}
        self._epoch = [[0] * self.B for _ in range(self.L)]
        self.records_applied = 0                          # records the host applied (reports, the microbench)
        self.pin_only_records = [0, 0]                    # (verify, draft) records applied only for a new pin
        self.last_counts = [0, 0]
        self.sparse_dropped_at: Optional[Tuple[int, str]] = None   # a negative control's (tick, kind)
        self._mut_done: Set[str] = set()

    # ------------------------------------------------------------------------------------------------ epochs and keys
    def _drop(self, key: Tuple, l: int) -> None:
        b = self.uid_b.get(key[0])
        if b is not None:
            self._epoch[l][b] += 1                        # its pins at (l, b) may be gone: record them again
        super()._drop(key, l)

    def end_request(self, uid) -> None:
        b = self.uid_b[uid]
        for l in range(self.L):
            self._epoch[l][b] += 1
        super().end_request(uid)

    def _void_pins(self) -> None:
        """A void tick: the device's pin table recorded pins that the host never applied."""
        for l in range(self.L):
            for b in range(self.B):
                self._epoch[l][b] += 1

    def _tick_extra(self, rows: Sequence[Row]) -> List[int]:
        kids = [self._kids.setdefault(r.key, len(self._kids) + 1) for r in rows]
        return kids + [e for per_l in self._epoch for e in per_l]

    # ------------------------------------------------------------------------------------------------ one layer
    def _caps(self, step: Dict) -> Tuple[int, int]:
        nv = step["vdec"].Rv * self.H * step["vdec"].k if step["vdec"] is not None else 0
        nd = step["ddec"].Rd * self.H * step["ddec"].k if step["ddec"] is not None else 0
        if self.record_cap is None:
            return max(nv, 1), max(nd, 1)
        return int(self.record_cap[0]), int(self.record_cap[1])

    def _record_inputs(self, l: int, step: Dict, tk: Dict):
        """sparse_layer_record's arguments for layer l (device tensors and host ints): (args, keywords)."""
        R = len(tk["vrows"]) + len(tk["drows"])
        kid_all = tk["extra"][:R]
        epoch_l = tk["extra"][R:].view(self.L, self.B)[l]
        kid_v = kid_all[tk["v"][4]] if tk["v"] is not None else None
        kid_d = kid_all[tk["d"][4]] if tk["d"] is not None else None
        bbd = hhd = None
        ddec = step["ddec"]
        if ddec is not None:
            Rd, k = ddec.Rd, ddec.k
            bbd = tk["d"][0].view(Rd, 1, 1).expand(Rd, self.H, k).reshape(-1)
            hhd = torch.arange(self.H, device=self.device).view(1, self.H, 1).expand(Rd, self.H, k).reshape(-1)
        cap_v, cap_d = self._caps(step)
        args = (self.pins[l], epoch_l, self.dmirror.code[l], step["vdec"], kid_v, ddec, kid_d, bbd, hhd, self.nblk,
                self.n_hist, cap_v, cap_d)
        kw = dict(ixv=tk["v"][4] if tk["v"] is not None else None, ixd=tk["d"][4] if tk["d"] is not None else None,
                  R=R, pins=self.pin_records, rowsel_by_entry=self.sparse_mut.get("rowsel_by_entry", False))
        return args, kw

    def _layer_record(self, l: int, step: Dict, tk: Dict) -> None:
        args, kw = self._record_inputs(l, step, tk)
        step["rec"] = sparse_layer_record(*args, **kw)
        step["caps"] = (args[11], args[12])
        ddec, bbd, hhd = args[5], args[7], args[8]
        if ddec is not None and self.store_b is not None and self.cis is not None:
            if self.xpool is not None:                                # [xpool] the takes' lanes (bound in resolve)
                XP.draft_bias_fill_flat(self.xpool, l, self.store_b[l], self.cis[l], ddec, bbd, hhd, self.bs)
            else:
                draft_bias_fill(self.store_b[l], self.cis[l], ddec, bbd, hhd, self.bs)

    def _layer_record_lean(self, l: int, step: Dict, tk: Dict, rcs: Dict) -> None:
        """[resfix] --res-lean: _layer_record with sparse_layer_record_lean and the lean bias fills, the same record and
        writes. The reference _layer_record runs instead for a class that overrides _layer_record (TrashWriterB2, a
        control) and with pin_records (the lean record has no pin path)."""
        if type(self)._layer_record is not ResidencySparse._layer_record or self.pin_records:
            return self._layer_record(l, step, tk)
        cst = self._lean_c
        vdec, ddec = step["vdec"], step["ddec"]
        R = len(tk["vrows"]) + len(tk["drows"])
        cap_v, cap_d = self._caps(step)
        rcv = self._lean_rec_c(tk, "v", R, rcs["v"]) if vdec is not None else None
        rcd = self._lean_rec_c(tk, "d", R, rcs["d"]) if ddec is not None else None
        step["rec"] = sparse_layer_record_lean(self.dmirror.code[l], vdec, ddec, self.B, self.H, self.nblk, self.n_hist,
                                               cap_v, cap_d, R, cst, rcv=rcv, rcd=rcd)
        step["caps"] = (cap_v, cap_d)
        if ddec is not None and self.store_b is not None and self.cis is not None:
            if self.xpool is not None:                                # [xpool] the takes' lanes (bound in resolve)
                XP.draft_bias_fill_flat_lean(self.xpool, l, self.store_b[l], self.cis[l], ddec, rcd["bb_f"],
                                             rcd["hh_f"], self.bs, cst)
            else:
                draft_bias_fill_lean(self.store_b[l], self.cis[l], ddec, rcd["bb_f"], rcd["hh_f"], self.bs, cst)

    def _lean_rec_c(self, tk: Dict, part: str, R: int, rc: Dict) -> Dict:
        """[resfix] The record's tick constants of one row part, added to its _lean_rows_c dict once per tick and width:
        o = arange(N), j = o // (H k) (the entry's row in the part), rowt = ix[j] (its tick row: rowsel), fbase =
        (kid[j] * H + h) * (NBLK + 1) (the left part of the record's FIRST key; kid = the key ids of the tick rows, the
        first R integers of the tick's extra copy, as _record_inputs reads them)."""
        if "rowt" not in rc:
            cst, H, k = self._lean_c, self.H, rc["k"]
            ix = tk[part][4]
            o = cst.ar(rc["sid"].shape[0])
            j = torch.div(o, H * k, rounding_mode="floor")
            kid = tk["extra"][:R][ix]
            rc.update(o=o, j=j, rowt=ix[j], fbase=(kid[j] * H + rc["hh_f"]) * (self.nblk + 1))
            # [resdev] a host bound of every valid FIRST key: kid <= len(self._kids) (_tick_extra numbers the keys 1, 2,
            # ...), h < H, blk < NBLK; the record sorts int32 keys when it is below 2**31 - 1 (--res-dev only)
            kids = getattr(self, "_kids", None)                       # (None: no bound, the int64 sort)
            rc["fk_bound"] = -1 if kids is None else (len(kids) * H + H - 1) * (self.nblk + 1) + self.nblk
        return rc

    # ------------------------------------------------------------------------------------------------ the consumer
    def apply_tick(self, steps: List[Dict]) -> None:
        """THE HOST CONSUMER (the S-C signature; module docstring): ONE device-to-host read, the void-tick refusals,
        then the records in v1's order."""
        if not self.pin_records:
            raise ValueError("pin_records=False: the sparse record carries no protection pins, and this consumer (B1) "
                             "protects only from them; a consumer that keeps the protection itself (B2) must replace "
                             "apply_tick")
        names = list(self._flags)
        parts = [self._flags[n].to(torch.int64).reshape(1) for n in names]
        W = (self.n_hist + 31) // 32
        Wb = (self.nblk + 31) // 32
        for st in steps:
            rec = st["rec"]
            if st["vdec"] is not None:
                parts += [rec["vrec"].reshape(-1), rec["vcnt"], st["vdec"].counters, rec["vlook"].reshape(-1),
                          rec["vread"].reshape(-1)]
            if st["ddec"] is not None:
                parts += [rec["drec"].reshape(-1), rec["dcnt"], st["ddec"].counters]
            parts.append(rec["sel"].reshape(-1))
        if not parts:
            return
        flat = self._d2h(torch.cat(parts))                            # ONE read per tick (a host tensor)
        self.flag_values = dict(zip(names, flat[:len(names)].tolist()))
        raised = sorted(n for n, v in self.flag_values.items() if v)
        if raised:
            self._void_pins()
            raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                  "of its decisions is applied)" % raised)
        at = len(names)
        parsed = []
        over, capped = [], []
        self.last_counts = [0, 0]                         # the tick's largest verify / draft record (reports, R5c)
        for st in steps:
            cap_v, cap_d = st["caps"]
            p = dict(st=st)
            if st["vdec"] is not None:
                dec = st["vdec"]
                F = len(VREC)
                buf = flat[at:at + F * cap_v].view(F, cap_v)
                at += F * cap_v
                n = int(flat[at])
                n_over = int(flat[at + 1])
                at += 2
                p["vlook"] = flat[at:at + dec.Rv * 4].view(dec.Rv, 4).tolist()
                at += dec.Rv * 4
                p["vread"] = flat[at:at + self.B * self.H * W].view(self.B, self.H, W)
                at += self.B * self.H * W
                self.last_counts[0] = max(self.last_counts[0], n)
                if n > cap_v:
                    capped.append((st["l"], "verify", n, cap_v))
                p["vrec"] = buf[:, :min(n, cap_v)].t().tolist()
                if n_over > 0:
                    over.append((st["l"], n_over))
            if st["ddec"] is not None:
                F = len(DREC)
                buf = flat[at:at + F * cap_d].view(F, cap_d)
                at += F * cap_d
                n = int(flat[at])
                p["dcounters"] = flat[at + 1:at + 4].tolist()
                at += 4
                self.last_counts[1] = max(self.last_counts[1], n)
                if n > cap_d:
                    capped.append((st["l"], "draft", n, cap_d))
                p["drec"] = buf[:, :min(n, cap_d)].t().tolist()
            p["sel"] = flat[at:at + self.B * self.H * Wb].view(self.B, self.H, Wb)
            at += self.B * self.H * Wb
            parsed.append(p)
        if over:
            self._void_pins()
            raise self._refusal(over, self._sparse_overflow_report(parsed, capped))
        if capped and not self.sparse_mut.get("no_cap_check"):
            self._void_pins()
            raise CapacityRefusal("S-C sparse record: a layer's record holds more entries than its capacity (layer, "
                                  "kind, entries, capacity: %s): the tick is void (none of its decisions is applied)"
                                  % capped)
        self._mutate(parsed)
        for p in parsed:
            st = p["st"]
            if st["vdec"] is not None:
                self._apply_verify_sparse(st["l"], st["vrows"], st["vidx"], st["vdec"], p)
            if st["ddec"] is not None:
                self._apply_draft_sparse(st["l"], st["drows"], st["ddec"], p)

    def _sparse_overflow_report(self, parsed: List[Dict], capped) -> str:
        """S-C's overflow report (_k5_report) from the verify records: the first misses are the GRANT, EVICT and HOLE
        records (a ready entry, a merge and a later occurrence of a miss are none of these), in entry order (K3X). A
        capped verify record holds only its first entries: the report says so."""
        out = []
        for p in parsed:
            st = p["st"]
            dec = st["vdec"]
            if dec is None:
                continue
            firsts = []
            for rec in p["vrec"]:
                o, fl = rec[0], rec[2]
                if not fl & (GRANT | EVICT | HOLE):
                    continue
                bh = (st["vrows"][o // (dec.H * dec.k)].b, (o // dec.k) % dec.H)
                firsts.append((bh, "grant" if fl & GRANT else ("evict" if fl & EVICT else "none")))
            n_none = sum(1 for _, what in firsts if what == "none")
            if not n_none:
                continue
            part = any(c[0] == st["l"] and c[1] == "verify" for c in capped)
            out.append(self._k5_report(st["l"], firsts, n_none,
                                       note=" (the verify record is capped: these counts are partial)" if part else ""))
        return " | ".join(out)

    # ------------------------------------------------------------------------------------------------ controls
    def _mutate(self, parsed: List[Dict]) -> None:
        """NEGATIVE CONTROLS (tests only): drop the first state-changing record of one kind (once per run), swap two
        draft takes of one stream (once per run)."""
        kind = self.sparse_mut.get("drop")
        if kind and kind not in self._mut_done:
            for p in parsed:
                if self._drop_one(p, kind):
                    self._mut_done.add(kind)
                    self.sparse_dropped_at = (self.tick_no, kind)
                    break
        if self.sparse_mut.get("swap_takes") and "swap_takes" not in self._mut_done:
            for p in parsed:
                recs = p.get("drec") or []
                takes = [i for i, r in enumerate(recs) if r[2] & GRANT and not r[2] & HOLE]
                dec = p["st"]["ddec"]
                for a in range(len(takes)):
                    for c in range(a + 1, len(takes)):
                        ra, rc = recs[takes[a]], recs[takes[c]]
                        if (ra[0] // (dec.H * dec.k), (ra[0] // dec.k) % dec.H) == \
                                (rc[0] // (dec.H * dec.k), (rc[0] // dec.k) % dec.H) and ra[4] != rc[4]:
                            ra[4], rc[4] = rc[4], ra[4]
                            self._mut_done.add("swap_takes")
                            self.sparse_dropped_at = (self.tick_no, "swap_takes")
                            return

    def _drop_one(self, p: Dict, kind: str) -> bool:
        st = p["st"]
        l = st["l"]
        pool = self.pools[l]
        if kind == "v_read":                                    # a verify-read slot that no draft row of the layer
            words = p.get("vread")                              # reads (else the draft read sets the same reader)
            if words is None:
                return False
            dread = set()
            ddec = st["ddec"]
            for r in (p.get("drec") or []) if ddec is not None else []:
                if r[2] & READY:
                    o = r[0]
                    dread.add((st["drows"][o // (ddec.H * ddec.k)].b, (o // ddec.k) % ddec.H, r[3]))
            for (b, h, idx) in torch.nonzero(unpack_bits(words, self.n_hist)).tolist():
                if (b, h, idx) in dread:
                    continue
                w, bit = divmod(idx, 32)
                words[b, h, w] = int(words[b, h, w]) & ~(1 << bit)
                return True
            return False
        side = "vrec" if kind.startswith("v_") else "drec"
        recs = p.get(side)
        if not recs:
            return False
        dec = st["vdec"] if side == "vrec" else st["ddec"]
        rows = st["vrows"] if side == "vrec" else st["drows"]
        want = {"v_pin": PIN, "v_merge": INFL | DUP, "v_take": GRANT | EVICT, "d_pin": PIN, "d_ready": READY,
                "d_merge": INFL, "d_take": GRANT, "d_refused": REFUSED}[kind]
        pkind = VERIFY if side == "vrec" else DRAFT
        for i, r in enumerate(recs):
            if not r[2] & want:
                continue
            o, t = r[0], r[1]
            j, h = o // (dec.H * dec.k), (o // dec.k) % dec.H
            row = rows[j]
            s = pool.stream(row.b, h)
            if kind.endswith("_pin"):                           # only a pin the host does not hold yet changes state
                if row.key in s.prot.get(t, ()) and pkind in s.pin_kind.get(t, {}).get(row.key, ()):
                    continue
                r[2] &= ~PIN
                return True
            if kind.endswith("_merge"):                         # (this runs BEFORE the tick's applies)
                tr = self._in_flight(l, row.b, h, t)
                if tr is not None and pkind in tr.demand:
                    continue                                    # the demand is there already: no state change
                if tr is None and kind == "v_merge":
                    continue                                    # a dup of this layer's urgent take: verify already
                # d_merge with no transfer yet: the urgent take of this layer creates it at apply, demand {verify}
            if kind == "d_ready" and (t in s.lru and next(reversed(s.lru)) == t):
                continue                                        # a touch of the MRU entry changes nothing
            del recs[i]
            return True
        return False

    # ------------------------------------------------------------------------------------------------ verify
    def _apply_verify_sparse(self, l: int, vrow_objs: List[Row], vidx: List[int], dec, p: Dict) -> None:
        """S-C's _apply_verify on the records (module docstring): v1's order per (row, head) group, then the looks,
        the reads, the release queue."""
        H, k = dec.H, dec.k
        pool = self.pools[l]
        now = self.clock()
        recs = p["vrec"]
        self.records_applied += len(recs)
        cands = self._candidates_from_bits(l, vrow_objs, p["sel"]) if any(r[2] & EVICT for r in recs) else {}
        firsts: Dict[Tuple, List[Tuple[int, int, str]]] = {}         # key -> [(h, block, class)]
        use0, n_first = {}, {}                                        # the K5 REPORT (S-C's _extra_need)
        i, n = 0, len(recs)
        while i < n:
            g = recs[i][0] // k                                       # (row, head) group: o // k = j * H + h
            j, h = g // H, g % H
            r = vrow_objs[j]
            st = pool.stream(r.b, h)
            takes = []
            while i < n and recs[i][0] // k == g:
                o, t, fl, slot, islot, igen, rslot, tslot, evb = recs[i]
                i += 1
                if not fl & (INFL | DUP | GRANT | EVICT | HOLE):
                    self.pin_only_records[0] += 1                     # a record only for its new pin
                if fl & HOLE:
                    raise OwnershipViolation("S-C: block %d of verify row %d (layer %d b %d h %d) has no slot and the "
                                             "device did not flag an overflow" % (t, vidx[j], l, r.b, h))
                if fl & PIN:
                    st.protect(t, r.key, VERIFY)
                if fl & (INFL | DUP):
                    tr = self._in_flight(l, r.b, h, t)
                    want = islot if fl & INFL else rslot
                    if tr is None or tr.slot != want or (fl & INFL and tr.gen != igen):
                        raise OwnershipViolation("S-C: the device read block %d in flight at slot %d (layer %d b %d h "
                                                 "%d), the host has %s" % (t, want, l, r.b, h,
                                                                         None if tr is None else (tr.slot, tr.gen)))
                    if not self.mut.drop_demand_merge:
                        tr.demand.add(VERIFY)
                    if fl & FIRST:
                        firsts.setdefault(r.key, []).append((h, t, "in_flight"))
                elif fl & (GRANT | EVICT):
                    takes.append((t, tslot, evb if fl & EVICT else None, "inflight" if fl & EVICT_INFL else "lru",
                                  fl & FIRST))
            if takes:                                                 # the usage at the layer's start (no take of
                if (r.b, h) not in use0:                              # this layer in the stream yet)
                    use0[(r.b, h)] = st.usage()
                n_first[(r.b, h)] = n_first.get((r.b, h), 0) + len(takes)
            for (t, at, ev_blk, ev_kind, first) in takes:
                if ev_blk is not None:                                # Policy E: the host's next candidate, evicted
                    q = cands.get((r.b, h))
                    nxt = q.popleft() if q else None
                    if nxt is None or nxt[:3] != (ev_kind, ev_blk, at):
                        raise OwnershipViolation("S-C Policy E: the device evicted the %s block %d at slot %d (layer %d "
                                                 "b %d h %d); the host's next candidate is %s"
                                                 % (ev_kind, ev_blk, at, l, r.b, h, None if nxt is None else nxt[:3]))
                    if ev_kind == "lru":
                        self._evict_now(st, ev_blk, at)
                    else:
                        self._evict_inflight(st, l, ev_blk, at, nxt[3])
                tr = self._take(l, r.b, h, t, r.key, VERIFY, at=at, evicted=ev_blk is not None)
                tr.delivered = True                                   # the device's urgent gather landed it
                self._landed.add(tr.seq)
                if first:
                    firsts.setdefault(r.key, []).append((h, t, "new"))
        self._note_extra_need(l, self._extra_need(use0, n_first))
        # the looks (4.3 / 4.6): the ready first looks counted on the device, the others from the records
        keys = []
        for jj, r in enumerate(vrow_objs):
            if r.key not in keys:
                keys.append(r.key)
        ready = {}
        for jj, r in enumerate(vrow_objs):
            acc = ready.setdefault(r.key, [0, 0, 0, 0])
            for q in range(4):
                acc[q] += p["vlook"][jj][q]
        for key in keys:
            call = self.metrics.open_call(key, l, now)
            nr, nl, npo, nv = ready[key]
            call["ready"] += nr
            self.metrics.k46["lru"] += nl
            self.metrics.k46["po"] += npo
            self.metrics.k46["vres"] += nv
            self.metrics.k46["not_resident"] += nr - nl - npo - nv
            b = self.uid_b[key[0]]
            for cls in ("in_flight", "new"):
                for (h, t, c) in firsts.get(key, []):
                    if c == cls:
                        self.metrics.look_one(call, l, pool.stream(b, h), cls, t, now)
        # the verify reads: the read bitmap (one bulk reader update)
        ev = self._layer_ev.get(l)
        for (b, h, idx) in torch.nonzero(unpack_bits(p["vread"], self.n_hist)).tolist():
            s = pool.stream(b, h).slots[idx]
            if s.first_reader is None:
                s.first_reader = ev
            s.last_reader = ev
            self.metrics.read(l, b, h, idx, s.gen, True)
        for key in keys:
            call = self.metrics.open_calls.pop((key, l), None)
            if call is not None:
                self.metrics.k43["layer_calls"] += 1
                for f in ("ready", "in_flight", "new", "wait_ns"):
                    self.metrics.k43[f] += call[f]
                self.metrics.k43["layer_wait_ns"] += max(0, self.clock() - call["t"])
        if not self.mut.release_only_at_decision:
            for key in keys:
                self._queue(key, l)

    def _candidates_from_bits(self, l: int, vrow_objs: List[Row], selw: torch.Tensor) -> Dict:
        """Policy E's host candidates (S-C's _host_candidates) with the layer's selections from its bitmap."""
        pool = self.pools[l]
        out = {}
        for r in vrow_objs:
            for h in range(self.H):
                if (r.b, h) in out:
                    continue
                st = pool.stream(r.b, h)
                sset = set(torch.nonzero(unpack_bits(selw[r.b, h], self.nblk)).view(-1).tolist())
                out[(r.b, h)] = deque([("lru", blk, idx, None) for blk, idx in st.lru.items()
                                       if not st.prot.get(blk) and blk not in sset]
                                      + self._inflight_candidates(l, st, sset))
        return out

    # ------------------------------------------------------------------------------------------------ draft
    def _apply_draft_sparse(self, l: int, drow_objs: List[Row], dec, p: Dict) -> None:
        """S-B's _apply on the records: every ready entry, merge, take and refusal in K3X order; the reads; the
        recency (touches at C speed, acquisitions and evict_to_c vendored)."""
        H, k = dec.H, dec.k
        pool = self.pools[l]
        masked, refused, pairs = p["dcounters"]
        got: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        trs = []
        n_ref = 0
        recs = p["drec"]
        self.records_applied += len(recs)
        for (o, t, fl, slot, tslot) in recs:
            if not fl & (READY | INFL | GRANT | REFUSED | HOLE):
                self.pin_only_records[1] += 1                         # a record only for its new pin
            j, h = o // (H * k), (o // k) % H
            r = drow_objs[j]
            st = pool.stream(r.b, h)
            if fl & HOLE:
                raise OwnershipViolation("S-B: draft block %d (layer %d b %d h %d) has no decision" % (t, l, r.b, h))
            if fl & PIN:
                st.protect(t, r.key, DRAFT)
            if fl & READY:
                if st.resident(t) != slot:
                    raise OwnershipViolation("S-B: the device read block %d at slot %d, the host holds it at %s (layer %d "
                                             "b %d h %d)" % (t, slot, st.resident(t), l, r.b, h))
                got.setdefault((j, h), []).append((t, slot, st.slots[slot].gen))
            elif fl & INFL:
                tr = self._in_flight(l, r.b, h, t)
                if tr is None:
                    raise OwnershipViolation("S-B: the device merged block %d (layer %d b %d h %d), which is not in flight"
                                             % (t, l, r.b, h))
                if not self.mut.drop_demand_merge:
                    tr.demand.add(DRAFT)
            elif fl & GRANT:
                trs.append(self._take(l, r.b, h, t, r.key, DRAFT, at=tslot, bias=False))
            elif fl & REFUSED:
                if self._draft_can_take(pool, st):
                    raise OwnershipViolation("S-B: the device refused block %d (layer %d b %d h %d), K5 allows it"
                                             % (t, l, r.b, h))
                self.metrics.count("draft_refused")
                n_ref += 1
        if n_ref != refused:
            raise OwnershipViolation("S-B: %d refusals applied, the device counted %d" % (n_ref, refused))
        ev = self._layer_ev.get(l)
        for (j, h), lst in sorted(got.items()):
            r = drow_objs[j]
            for (t, idx, gen) in lst:
                s = pool.stream(r.b, h).slots[idx]
                if s.first_reader is None:
                    s.first_reader = ev
                s.last_reader = ev
                self.metrics.read(l, r.b, h, idx, gen, False)
        self._draft_recency_fast(l, drow_objs, got)
        if trs:
            self._deferred.append((l, trs))
        self.metrics.layer_plan(Plan(page_table=None, num_pages=None, last_len=None, seq_idx=None, blocks=[],
                                     masked=int(masked), draft_pairs=int(pairs), refused=int(refused)))

    def _draft_recency_fast(self, l: int, rows: Sequence[Row], got_all) -> None:
        """v1's _draft_recency (K3X O1) with the touches applied directly: OrderedDict.move_to_end and S-C's journal
        effects of a "touch" event (its count, the slot's new recency stamp, its dirty mark) in place of
        Stream.touch's log record. Acquisitions and evict_to_c run the vendored code."""
        pool = self.pools[l]
        stamp, dirty = self._stamp, self._dirty
        mark = "touch" not in self.mirror_drop
        n_touch = 0
        for i, r in enumerate(rows):
            if r.kind != "d":
                continue
            for h in range(self.H):
                st = pool.stream(r.b, h)
                lst = got_all.get((i, h))
                if lst:
                    lru = st.lru
                    key = (l, r.b, h)
                    for (t, idx, gen) in sorted(lst):
                        if t in lru:
                            lru.move_to_end(t)
                            self._clock += 1
                            sk = key + (lru[t],)
                            stamp[sk] = self._clock
                            if mark:
                                dirty.add(sk)
                            n_touch += 1
                        elif t in st.po or t in st.vres:
                            if not self.mut.skip_acquisition:
                                st.acquire(t, self.mut, evict=False)
                        else:
                            raise OwnershipViolation("the draft row %d read block %d (layer %d head %d), which is no "
                                                     "longer resident at the recency update" % (i, t, l, h))
                st.evict_to_c(self.mut)
        if n_touch:
            self.journal_counts["touch"] = self.journal_counts.get("touch", 0) + n_touch
