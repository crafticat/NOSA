"""R2 step 7, piece 5, (B2): OwnResidency, the residency service whose OWNERSHIP AUTHORITY is the array-backed table of
r2_own2_ops (docs/R2_OWN2_B2_DESIGN.md rev 2; the lead's GO of 2026-10-21). SOURCE ONLY.

WHAT B2 REPLACES. ResidencySparse (B1, r2_residency_sparse.py) applies C's sparse record to the vendored K3X objects
(kstep/ownership.py) one record at a time. OwnResidency keeps everything of B1 / S-C that is not per-slot ownership
state and replaces that state:
  - KEPT (inherited, unchanged): the rows (_check_rows: cancelled keys I11, the late-protection guard I12, the batch
    slot), _tick_rows (the tail and trash checks I15), resolve (the device decisions device_verify / device_draft on
    the device mirror, the page tables, the device wait and urgent gather, C's sparse record with pin_records=False
    and rowsel, the draft bias fill), read_done (the key readers; the layer event), watch_flag, decided / cancel /
    cancel_epoch / history_written (host maps per key, per request, per block), the transport, Residency.end_tick
    (the draft issue), the metrics object.
  - REPLACED (B2): the slots, the LRU, PO / VRES, the protections, the readers, the owners: OwnTable (self.own); the
    device mirror's delta (sync_mirror from the table); the host consumer apply_tick(steps) (C's signature, steps[i] =
    dict(l, vrows, vidx, vdec, drows, ddec, rec, caps)) with r2_own2_ops' vectorized ops; poll (arrivals, the queued
    drops, the releases, in v1's order); begin_request / end_request; _in_flight (the current owner from the table).
  - THE TRANSFER SIDE stays host objects in this class (one vendored Transfer per take, on the vendored transport), so
    the CPU lockstep runs on the same transports as v1 and S-C; the device path replaces it with E's issue_desc (the
    take list is O.issue_desc_args' desc). [b2-desc] issue="desc" (default "object") keeps the Transfer objects and
    replaces only the transport issue: the tick's draft takes go to tp.issue_desc as ONE job (ported from 47db0c0 by
    hunks; [f3ld] the job is queued by the host consumer and submitted in end_tick's issue phase, ResidencySC.end_tick's
    _tick_issue hook, where Residency.end_tick issues on the object path, so --host-profile's h_et_issue times it),
    every draft Transfer holds the job's completion object, T3 / T4 become E's reader / write-back marks (_rid,
    history_written); docs/R2_B2_DESC_PLAN.md of 47db0c0.
  - `pools` and `stream()` are a READ-ONLY PROJECTION: real vendored LayerPool / Stream / Slot objects built from the
    table (cached per table version), so the vendored snapshot helpers, the device stand-in (CpuDeviceSim) and the
    reports read B2's state through the vendored interface. Nothing writes them.

CHECKED / UNCHECKED (checked=True by default): checked runs every predicate of rev 2 section 3.3 in two host reads per
apply (pre-write: a set flag raises OwnershipViolation before any change; post-write: the undo log rolls back, then it
raises); and I1 / I2 after every poll. Unchecked trusts the device's decisions (the timed path).

THE JOURNAL: journal_counts counts every stream transition by kind as the vendored log does (take, release:<why>,
evict_po, acquire, admit, admit_vres, drop_unowned, drop_vres_unprotected, free, touch).

NEGATIVE CONTROLS (b2_mut, tests only; the B2 forms of C's fixture plants and of rev 2's planted kinds):
  no_dedup_check, mirror_ignores_owner, owner_blind, no_uid_check, free_on_cancel, one_batch, ignore_flags, no_t4,
  no_reader_wait, drop_rd_last, keep_rd_last_on_drop, poll_order_swapped, key_alias, trash_writer, and the record
  mutations rec_mut (drop / alter the first record of a kind, after the read and before the checks).

[inflight-array] THE IN-FLIGHT TABLE OF ISSUES (--inflight-array; OPT-IN, default False: every line of the old poll
runs; analysis/evalfast/DIAG_effective_gap.md section 4, fix 2). The old poll sorts every transfer in flight and asks
the arrival rule once per transfer (h_poll_arrivals 4.9 s at tick 2 of evalfast2rl: the whole first fill arrives
there). With the flag, _host_side also keeps the draft takes of a tick as ISSUE PARTS, one per layer: the transfer list
that end_tick gives to ONE tp.issue call, and the takes' (flat slot, generation, block) columns, selected on the table's
device from write_takes' columns. The poll then:
  - admits the urgent takes in flight (the _landed set), as before (no arrival rule);
  - admits every transfer in flight of every DUE part (issue tick + arrival_lag <= tick), and none of a part not due:
    the old rule, per part instead of per transfer. A transfer that left the map (a Policy E victim) is skipped.
  - asks the transport ONE done (and at most ONE force) per part when the transport says that every transfer of one
    issue completes on one event, set in the transfers' order (one_event_per_issue: gilfree, gather); else it asks per
    transfer, in the old (layer, b, h, seq) order. The forced and held counts are the old ones (per transfer).
  - checks I6 in ONE vectorized compare on the device (the old per-transfer check gives the old exception on a
    failure), and removes the admitted transfers from the host maps in C-level map calls.
The admitted set is the old set at every poll, so every decision, page table and logit is the same
(tests_cpu/test_r2_admitfill.py). A COVERAGE CHECK at each poll refuses a transfer in flight that no part and no urgent
take holds. It needs the fixed arrival rule (arrival_lag): with free timing the due set is per-transfer completion.
NEGATIVE CONTROLS (b2_mut, tests only; each acts at the site under test, the array poll): ia_skip_one (a due part
leaves out one transfer), ia_due_early (a part is due one tick early), ia_bad_gen (the check's expected generation of
the first arrival is one too high).
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from itertools import chain, compress, repeat
from operator import attrgetter, is_
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import r2_own2_ops as O
from .kstep.backend import ev_done
from .kstep.errors import CapacityRefusal, OwnershipViolation, StaleGeneration
from .kstep.executor import DRAFT, VERIFY
from .kstep.ownership import FREE, INFLIGHT, LRU, PO, RELEASING, VRES, LayerPool, Stream
from .kstep.transfer import Transfer
from .r2_residency import N_TAIL, Plan, Row
from .r2_residency_sc import C_FREE, C_INFLIGHT, C_LRU, C_PO, C_RELEASING, C_VRES, InflightByLayer
from .r2_residency_sparse import DREC, VREC, ResidencySparse

STATE_OF = {C_FREE: FREE, C_INFLIGHT: INFLIGHT, C_LRU: LRU, C_PO: PO, C_VRES: VRES, C_RELEASING: RELEASING}
CLS_OF = {0: None, O.CLS_AD: "ad", O.CLS_V: "v"}
_SH32 = torch.arange(32, dtype=torch.int64)
# no_read_harvest (SG5 b2): the 4.8 lifetimes get no read counts from the table (report()["k48"] differs from S-C's)
B2_MUTS = ("no_dedup_check", "mirror_ignores_owner", "owner_blind", "no_uid_check", "free_on_cancel", "one_batch",
           "ignore_flags", "no_t4", "no_reader_wait", "drop_rd_last", "keep_rd_last_on_drop", "poll_order_swapped",
           "key_alias", "rec_mut", "no_read_harvest",
           "ia_skip_one", "ia_due_early", "ia_bad_gen")              # [inflight-array] the array poll's controls
ISSUES = ("object", "desc")                              # [b2-desc] the draft issue path (OwnResidency(issue=...))
REC_MUTS = ("take_nonfree", "take_swap", "k5_over", "pe_not_first", "drop_refusal", "drop_v_merge", "drop_d_merge",
            "drop_ready", "drop_rowsel", "drop_vread", "ignore_overflow", "reverse_stamps", "evict_po_unpinned",
            "gen_plus_two")
# [inflight-array] the Transfer fields the array poll reads in C-level map calls; _ORDER is the old poll's order
_SEQ, _B, _H, _BLK, _REQ = (attrgetter(n) for n in ("seq", "b", "h", "block", "requester"))
_ORDER = attrgetter("layer", "b", "h", "seq")


class _IAPart:
    """[inflight-array] One draft issue of one tick (module docstring): the layer l, the transfer list trs that end_tick
    gives to ONE tp.issue call (the same list object), their seqs (made at the first poll), and cols: their (flat slot,
    generation, block) columns, (3, len(trs)) int64 on the table's device, in the order of trs."""
    __slots__ = ("l", "trs", "lo", "hi", "cols", "seqs")

    def __init__(self, l: int, trs: list, lo: int, hi: int):
        self.l, self.trs, self.lo, self.hi = int(l), trs, int(lo), int(hi)
        self.cols = None
        self.seqs = None


class OwnResidency(ResidencySparse):
    """The residency service with B2's table as the ownership authority (module docstring). Extra arguments: keys (live
    cohort key slots per request, KEYS <= 7), checked, b2_mut (negative controls, tests only)."""

    def __init__(self, *args, keys: int = 7, checked: bool = True, b2_mut=None, resolve: str = "torch",
                 fused_plant: int = 0, fused_guard: bool = False, fused_warm: str = "auto", lean: bool = False,
                 wait_index: bool = False, fewer_reads: bool = False, res_lean: bool = False, res_dev: bool = False,
                 fused_kernel: str = None, inflight_array: bool = False, issue: str = "object",
                 **kw):                                       # [fr3] [inflight-array] ([f3lia]) [b2-desc] ([f3ld])
        kw["pin_records"] = False
        self.__dict__["_own_ready"] = False
        self.issue = "object"                             # [b2-desc] before the parent's constructor (lesson L19)
        self._hm = None                                   # [b2-desc] (B, nblk) host: each block's write-back mark
        self._hmark = 0                                   # [b2-desc] the last write-back mark given (marks grow)
        self._hev: Dict[int, Tuple[Any, int]] = {}         # [b2-desc] id(write-back event) -> (event, its mark)
        self.desc_jobs = 0                                # [b2-desc] REPORT: descriptor jobs issued
        self.desc_takes = 0                               # [b2-desc] REPORT: draft takes issued through them
        self._desc_next = None                            # [f3ld] the tick's queued job (_issue_desc -> _tick_issue)
        # [hostfix] --lean-time (OPT-IN, default False: every line as before): the REPORT-only host work is skipped
        # in a time cell (LEAN below); every decision, the release queue, the transfers and every counter a MUST gate
        # reads stay. Set before the base constructor (S-C's end_tick reads it).
        self.lean = bool(lean)
        # [hostfix] --b2-fewer-reads (OPT-IN, default False): the consumer's host values that exist on the host already
        # are uploaded in ONE pinned non-blocking copy, and its device values are read in ONE read each place (FEWER
        # READS below); the same values, so the same decisions
        self.fewer_reads = bool(fewer_reads)
        if resolve not in ("torch", "fused", "check"):                # [fusedres] r2_fusedres.RESOLVE_MODES
            raise ValueError("resolve %r: one of torch, fused, check" % (resolve,))
        if fused_plant and resolve == "torch":
            raise ValueError("fused_plant needs resolve 'fused' or 'check'")
        # [flean] res_lean is accepted with every resolve: "torch" runs ResidencySC._resolve_lean; "fused" / "check" run
        # FusedResolver with the lean host checks, bind_lean and the lean bias fill (r2_fusedres.FusedResolver._lean_*;
        # the fused kernels do not change; check mode keeps the REFERENCE bind on its shadow pool)
        if res_dev and not res_lean:                                  # [resdev] the lean resolve's device ops only
            raise ValueError("res_dev needs res_lean (it changes the lean resolve's device ops)")
        if fused_kernel is not None and resolve == "torch":           # [fr3] the kernel variant of the fused resolve
            raise ValueError("fused_kernel needs resolve 'fused' or 'check'")
        super().__init__(*args, **kw)
        # [b2-desc] the draft issue: "object" (the default; p5b2 / p6: Residency.end_tick, tp.issue per layer) or "desc"
        # (E's tp.issue_desc: the tick's draft takes as ONE (6, n) job; docs/R2_B2_DESC_PLAN.md). Checked HERE, not at
        # the first issue (lesson L31): desc needs E's C++ worker with packer v3's DESC. [f3ld] and no X pool: E's
        # issue_desc names logical slots and refuses xpool lanes (r2_transport_gilfree.py issue_desc), so the shared-pool
        # layout (--layout pool, every eval cell) cannot take the descriptor path
        if issue not in ISSUES:
            raise ValueError("issue must be one of %s (got %r)" % (ISSUES, issue))
        if issue == "desc":
            tp = self.tp
            ok = (getattr(tp, "worker", None) == "cpp" and int(getattr(getattr(tp, "ext", None), "DESC", 0)) == 1
                  and all(callable(getattr(tp, n, None)) for n in ("issue_desc", "mark_reader", "mark_hist", "job_ev")))
            if not ok:
                raise ValueError("OwnResidency(issue='desc') needs GilfreeTransport(worker='cpp') over packer v3 with "
                                 "DESC (issue_desc, mark_reader, mark_hist, job_ev); got %s" % (type(tp).__name__,))
            if self.xpool is not None:
                raise ValueError("OwnResidency(issue='desc') with an X pool: E's issue_desc names logical slots and "
                                 "does not translate xpool lanes (GilfreeTransport.issue_desc refuses xpool)")
        self.issue = issue
        if issue == "desc":
            self._hm = torch.full((self.B, self.nblk), -1, dtype=torch.int64)
        if wait_index:                                                # [hostfix] --wait-index (OPT-IN): the same map
            self.inflight = InflightByLayer(self.inflight)            # with a per-layer index (_superset_wait)
        # [inflight-array] --inflight-array (OPT-IN, default False: the old poll): the in-flight table of issues
        # (module docstring). It needs the fixed arrival rule: the due set is then a function of the issue tick only
        self.inflight_array = bool(inflight_array)
        if self.inflight_array and self.arrival_lag is None:
            raise ValueError("inflight_array needs a fixed arrival_lag (an int >= 1): with free timing a transfer "
                             "arrives at its own completion, so the poll must ask the transport per transfer")
        self._ia_groups: Dict[int, List[_IAPart]] = {}                # issue tick -> its draft issue parts
        self.ia_mut_done = None                                       # (tick, kind) of an applied array control
        self.ia_stats = dict(polls=0, parts_due=0, group_checks=0, transfer_checks=0, admitted=0)
        self.checked = bool(checked)
        self.b2_mut = dict(b2_mut or {})
        for k in self.b2_mut:
            if k not in B2_MUTS:
                raise ValueError("b2_mut %r" % (k,))
        if self.b2_mut.get("rec_mut") and self.b2_mut["rec_mut"] not in REC_MUTS:
            raise ValueError("rec_mut %r" % (self.b2_mut["rec_mut"],))
        self.KEYS = int(keys)
        G = O.Geo(L=self.L, B=self.B, H=self.H, C=self.C, X=self.X, R_ver=self.R_ver, NBLK=self.nblk, KEYS=self.KEYS)
        self.own = O.Table(G, device=self.device)
        self.own.clock = 1
        self.win = O.ReaderWindow(cap=1 << 16, device=self.device)
        self._evobj: Dict[int, Any] = {}                  # reader id -> its event object (every id a slot may name)
        self._eid: Dict[int, int] = {}                    # id(event object) -> its reader id
        self._uid_int: Dict[Any, int] = {}
        self._uid_obj: Dict[int, Any] = {}
        self._owner_py: List[Any] = [None] * self.B       # the request owning each batch slot (the table's owner)
        self._kslot: Dict[Any, int] = {}                  # cohort key -> its key slot (I13)
        self._key_of: Dict[Tuple[int, int], Any] = {}     # (b, key slot) -> cohort key
        self._ver = 0
        self._view = None
        self._view_ver = -1
        self._hc = None                                   # the table's host columns of version _hc_ver (projections)
        self._hc_ver = -1
        self._sviews: Dict[Tuple[int, int, int], Stream] = {}   # (l, b, h) -> its lazy projection (stream())
        self.rec_mut_done = None                         # (tick, kind) of an applied record mutation
        self.b2_applied = 0                               # ticks applied by the B2 consumer
        # [fusedres] the per-layer resolve: "torch" (S-C's device path, inherited), "fused" (r2_fusedres: 2 Triton
        # launches per draft-only layer, 4 with verify rows), "check" (fused, and the torch path compared at every layer)
        self.resolve_mode = resolve
        self.fusedres = None
        if resolve != "torch":
            from . import r2_fusedres
            self.fusedres = r2_fusedres.FusedResolver(self, check=resolve == "check", plant=int(fused_plant),
                                                      guard=bool(fused_guard),   # [fusedres2b] tests: OOB guards
                                                      warm=fused_warm,          # [stack2] warm_up (auto: CUDA)
                                                      kernel=fused_kernel)      # [fr3] None: DEFAULT_KERNEL (v2)
        # [resfix] --res-lean (OPT-IN, default False: resolve as before): S-C's torch resolve with less host work per
        # layer (ResidencySC._resolve_lean: no hidden sync, the per-row Python once per tick, the tick's constants once);
        # the same decisions, page tables and records (tests_cpu/test_r2_resfix.py)
        if res_lean:
            self.enable_res_lean()
        # [resdev] --res-dev (OPT-IN, default False): the lean resolve's device hot spots with r2_resdev's equal ops
        # (the same decisions, page tables and records; tests_cpu/test_r2_resdev.py)
        if res_dev:
            self.enable_res_dev()
        self.__dict__["_own_ready"] = True

    def resolve(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> Plan:
        """[fusedres] S-C's resolve (resolve_mode "torch") or the fused one (r2_fusedres.FusedResolver.resolve)."""
        if self.fusedres is None:
            return super().resolve(l, rows, sel)
        return self.fusedres.resolve(l, rows, sel)

    # ================================================================================================================
    # the projection: read-only vendored views of the table
    # ================================================================================================================
    @property
    def pools(self):
        if not self.__dict__.get("_own_ready"):
            return self.__dict__.get("_vpools")
        return self._materialize()

    @pools.setter
    def pools(self, v):
        self.__dict__["_vpools"] = v

    def _changed(self) -> None:
        self._ver += 1

    def _materialize(self) -> List[LayerPool]:
        """The table as vendored LayerPool objects (cached per table version; read-only by contract)."""
        if self._view is not None and self._view_ver == self._ver:
            return self._view
        hc = self._host_cols()
        pools = []
        for l in range(self.L):
            pool = LayerPool(l, self.B, self.H, self.C, self.X, self.R_ver)
            for (b, h), st in pool.streams.items():
                self._fill_stream(st, l, b, h, hc)
            pools.append(pool)
        self._view, self._view_ver = pools, self._ver
        return pools

    def _host_cols(self) -> Dict[str, Any]:
        """The table's columns on the host, ONE copy per table version (the projections read them); the pin planes
        that hold a bit, per stream."""
        if self._hc is not None and self._hc_ver == self._ver:
            return self._hc
        T = self.own
        hc: Dict[str, Any] = {n: T.c[n].to("cpu") for n in ("code", "block", "gen", "uid", "cls", "stamp", "rd_first",
                                                             "rd_last", "seq")}
        pins = T.pin_w.to("cpu")
        planes: Dict[Tuple[int, int, int], List[int]] = {}
        for (p, l, b, h) in pins.any(-1).nonzero().tolist():         # plane-major: each stream's planes ascending
            planes.setdefault((l, b, h), []).append(p)
        hc["pins"], hc["planes"] = pins, planes
        self._hc, self._hc_ver = hc, self._ver
        self._sviews = {}
        return hc

    def _fill_stream(self, st: Stream, l: int, b: int, h: int, hc: Dict[str, Any]) -> None:
        """One vendored Stream (fresh: no slot taken) set to the table's stream (l, b, h)."""
        st.owner = self._owner_py[b]
        code = hc["code"][l, b, h].tolist()
        blk = hc["block"][l, b, h].tolist()
        gen = hc["gen"][l, b, h].tolist()
        uid = hc["uid"][l, b, h].tolist()
        cls = hc["cls"][l, b, h].tolist()
        stamp = hc["stamp"][l, b, h].tolist()
        rf = hc["rd_first"][l, b, h].tolist()
        rl = hc["rd_last"][l, b, h].tolist()
        seq = hc["seq"][l, b, h].tolist()
        lru = []
        for i, s in enumerate(st.slots):
            c = code[i]
            s.state = STATE_OF[c]
            s.block = blk[i]
            s.gen = gen[i]
            s.owner = self._uid_obj.get(uid[i]) if uid[i] >= 0 else None
            s.cls = CLS_OF[cls[i]]
            s.first_reader = self._evobj.get(rf[i]) if rf[i] >= 0 else None
            s.last_reader = self._evobj.get(rl[i]) if rl[i] >= 0 else None
            s.seq = seq[i] if c == C_INFLIGHT else None
            if c == C_LRU:
                lru.append((stamp[i], blk[i], i))
            elif c == C_PO:
                st.po[blk[i]] = i
            elif c == C_VRES:
                st.vres[blk[i]] = i
            elif c == C_INFLIGHT and st.owner is not None and s.owner == st.owner:
                st.pending[blk[i]] = i
        for (_, bk, i) in sorted(lru):
            st.lru[bk] = i
        for plane in hc["planes"].get((l, b, h), ()):
            words = hc["pins"][plane, l, b, h]
            key = self._key_of.get((b, plane % self.KEYS))
            kind = VERIFY if plane < self.KEYS else DRAFT
            bits = (torch.bitwise_and(torch.bitwise_right_shift(words.unsqueeze(-1), _SH32), 1) != 0).reshape(-1)
            for bk in bits[:self.nblk].nonzero().view(-1).tolist():
                st.prot.setdefault(bk, set()).add(key)
                st.pin_kind.setdefault(bk, {}).setdefault(key, set()).add(kind)

    def stream(self, l: int, b: int, h: int):
        """One stream's read-only projection. Lazy (SG5 b2): from the whole projection when it is current, else ONE
        stream built from the host columns of this table version (cached until the table changes). Metrics.report
        reads one stream per lifetime record; the whole projection at SG5's p5b2 point (L 32, B 64, H 8, C + X 160)
        is 2.6 M Slot objects."""
        if not self.__dict__.get("_own_ready"):
            return super().stream(l, b, h)
        if self._view is not None and self._view_ver == self._ver:
            return self._view[l].stream(b, h)
        hc = self._host_cols()
        st = self._sviews.get((l, b, h))
        if st is None:
            st = Stream(l, h, b, self.C + self.X, self.C)
            self._fill_stream(st, l, b, h, hc)
            self._sviews[(l, b, h)] = st
        return st

    def memory(self) -> Dict[str, int]:
        """Residency.memory (r2_residency.py:799-812) on the table: ONE host read of the state counts. Metrics samples
        it at every poll and end_tick; the vendored version would build the whole projection each time."""
        if not self.__dict__.get("_own_ready"):
            return super().memory()
        code = self.own.c["code"]
        n = torch.stack([(code == x).sum() for x in (C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING)]).tolist()
        inflight, lru, po, vres, rel = (int(x) for x in n)
        as_lru = bool(self.mut.vres_counted_as_lru)
        c = dict(inflight=inflight, lru=lru + (vres if as_lru else 0), po=po, vres=0 if as_lru else vres,
                 releasing=0 if self.mut.mem_no_releasing else rel)
        c["slots_used"] = sum(c.values())
        c["slots_physical"] = self.L * self.B * self.H * (self.n_hist + N_TAIL + self.n_trash)
        if self.xpool is not None:                                    # [xpool] the flat store's (position, head) lanes
            c["slots_physical"] = self.L * self.H * self.xpool.M
        c["retained"] = c["po"] + c["vres"]
        return c

    def _flush_reads(self) -> None:
        """The 4.8 read counts (Metrics.lifetimes rd / rv) of the lifetimes whose slot was NOT taken again: the table
        holds them (rd_n / rv_n; a take resets them, and _host_side moves the old ones into their lifetime first).
        One host copy of three columns. no_read_harvest (NEGATIVE CONTROL): none."""
        if self.b2_mut.get("no_read_harvest") or not self.metrics.lifetimes:
            return
        G = self.own.geo
        keys = list(self.metrics.lifetimes)
        kt = torch.tensor(keys, dtype=torch.int64)
        f = ((kt[:, 0] * G.B + kt[:, 1]) * G.H + kt[:, 2]) * G.NPOS + kt[:, 3]
        gen, rd, rv = (self.own.flat(n).to("cpu")[f] for n in ("gen", "rd_n", "rv_n"))
        cur = (gen.to(torch.int64) == kt[:, 4]).tolist()
        rd, rv = rd.tolist(), rv.tolist()
        for i, k in enumerate(keys):
            if cur[i]:
                rec = self.metrics.lifetimes[k]
                rec["rd"], rec["rv"] = int(rd[i]), int(rv[i])

    def report(self) -> Dict:
        """S-C's report (Residency's 4.3 / 4.6 / 4.8 and "k5") on the table (the lifetimes' read counts flushed from
        the table first), plus "b2": the mode, the ticks applied, the key slots, the reader ids, the table's size and
        the mirror's form (a device copy of the table per tick: no host delta, so mirror_slots_uploaded stays 0)."""
        self._flush_reads()
        out = super().report()
        out["b2"] = dict(checked=self.checked, applied_ticks=self.b2_applied, keys=self.KEYS, reader_ids=self.win.next,
                         reader_live=self.win.next - self.win.lo,
                         table_bytes=sum(t.numel() * t.element_size() for t in self.own.tensors().values()),
                         mirror="the table's columns copied on the device at each tick's first resolve (no host delta)",
                         issue=self.issue, desc_jobs=self.desc_jobs, desc_takes=self.desc_takes)   # [b2-desc] REPORT
        out["b2"]["resolve"] = self.resolve_mode                      # [fusedres]
        if self.inflight_array:                                       # [inflight-array] REPORT (off: no key)
            out["b2"]["inflight_array"] = dict(self.ia_stats)
        if self.fusedres is not None:
            out["b2"]["fused"] = self.fusedres.report()
        return out

    def _in_flight(self, l: int, b: int, h: int, t: int):
        """The CURRENT owner's transfer in flight for (l, b, h, block t) (res.py:509-517), the owner from the table."""
        tr = self.pending.get((l, b, h, t))
        if tr is not None and tr.uid != self._owner_py[b] and not self.b2_mut.get("owner_blind"):
            return None
        return tr

    # ================================================================================================================
    # ids: requests, readers, cohort keys
    # ================================================================================================================
    def _uid_id(self, uid) -> int:
        k = self._uid_int.get(uid)
        if k is None:
            k = len(self._uid_int) + 1
            self._uid_int[uid] = k
            self._uid_obj[k] = uid
        return k

    def _rid(self, ev) -> int:
        """The reader id of an event object (registered once; the map keeps the object alive)."""
        if ev is None:
            return -1
        e = self._eid.get(id(ev))
        if e is None:
            e = self.win.register(ev, -1)
            self._eid[id(ev)] = e
            self._evobj[e] = ev
            if self.issue == "desc":
                # [b2-desc] T3 as E's reader marks: the reader id IS the mark. Ids grow in registration order, and the
                # only ids are the layers' read events (apply, layer order, tick order), all recorded on the main
                # stream: they complete in id order, so the job's wait on the newest id of a layer covers the older.
                self.tp.mark_reader(e, ev)
        return e

    def _key_slot(self, key) -> int:
        """The key slot of a cohort key (I13): the lowest slot of its request that no live key holds; a key is live while
        a pin bit of its slot is set at any layer, a (key, l) release is queued, or a key reader is pending. None free:
        CapacityRefusal (C's Q-B4: refused, never aliased). key_alias (NEGATIVE CONTROL): slot 0 for every key."""
        k = self._kslot.get(key)
        if k is not None:
            return k
        b = self.uid_b[key[0]]
        if self.b2_mut.get("key_alias"):
            self._kslot[key] = 0
            self._key_of[(b, 0)] = key
            return 0
        used = {kk for (bb, kk) in self._key_of if bb == b}
        free = [kk for kk in range(self.KEYS) if kk not in used]
        if not free:
            for (bb, kk), kkey in list(self._key_of.items()):
                if bb == b and not self._key_live(b, kk, kkey):
                    del self._key_of[(bb, kk)]
                    self._kslot.pop(kkey, None)
            used = {kk for (bb, kk) in self._key_of if bb == b}
            free = [kk for kk in range(self.KEYS) if kk not in used]
        if not free:
            raise CapacityRefusal("B2: request %s holds %d live cohort keys (KEYS %d): no key slot for %s (I13)"
                                  % (key[0], self.KEYS, self.KEYS, key))
        k = free[0]
        self._kslot[key] = k
        self._key_of[(b, k)] = key
        self._changed()                                               # the projection names the keys of the slots
        return k

    def _key_live(self, b: int, k: int, key) -> bool:
        if bool(self.own.pin_w[[k, k + self.KEYS], :, b].any()):
            return True
        if any(kk == key for (kk, _) in self.queue):
            return True
        return any(not ev_done(self.key_reader.get((key, l))) for l in range(self.L) if (key, l) in self.key_reader)

    # ================================================================================================================
    # the device mirror from the table
    # ================================================================================================================
    def sync_mirror(self) -> bool:
        """The device mirror (S-C's DeviceMirror, read by device_verify / device_draft) from the table: the state code,
        the block, the generation, the current owner's flag (an INFLIGHT slot of the batch slot's owner, S-C's cur),
        the protected flag (a resident or INFLIGHT slot whose block has a pin bit), the recency stamp. One copy of the
        table's columns (no journal delta)."""
        T, G = self.own, self.own.geo
        code = T.c["code"]
        blk = T.c["block"].to(torch.int64)
        cur = code == C_INFLIGHT
        if not self.b2_mut.get("mirror_ignores_owner"):
            cur = cur & O.owner_of_slots(T)
        s = torch.arange(G.S, device=T.dev).view(G.S, 1).expand(G.S, G.NPOS)
        pinned = O.bit_test(O.any_pin_words(T), s, blk.view(G.S, G.NPOS).clamp(min=0)).view(code.shape) & (blk >= 0)
        live = (code == C_LRU) | (code == C_PO) | (code == C_VRES) | (code == C_INFLIGHT)
        dm = self.dmirror
        dm.code.copy_(code.to(torch.int64))
        dm.block.copy_(blk)
        dm.gen.copy_(T.c["gen"].to(torch.int64))
        dm.cur.copy_(cur.to(torch.int64))
        dm.prot.copy_((pinned & live).to(torch.int64))
        dm.stamp.copy_(T.c["stamp"])
        self._dirty = set()
        self.mirror_uploads += 1
        return True

    # ================================================================================================================
    # requests and keys
    # ================================================================================================================
    def begin_request(self, b: int, uid, hist_blocks: Optional[int] = None) -> None:
        """res.py:203-218 on the table: refuses an owned batch slot; the owner, the T4 bound."""
        if uid in self.uid_b or not 0 <= b < self.B:
            raise OwnershipViolation("begin_request: uid %s already holds a batch slot, or slot %d is out of range"
                                     % (uid, b))
        if self._owner_py[b] is not None:
            raise OwnershipViolation("begin_request: batch slot %d is owned by %s" % (b, self._owner_py[b]))
        self._owner_py[b] = uid
        self.own.owner[b] = self._uid_id(uid)
        self.uid_b[uid] = b
        if hist_blocks is not None:
            self.hist_from[b] = int(hist_blocks)
            self.own.hist_from[b] = int(hist_blocks)
        self.own.xfer[b] = 0
        self._changed()

    def history_written(self, uid, block: int, ev) -> None:
        super().history_written(uid, block, ev)                       # hist_ev (T4), the main stream's waits
        self.own.hist_id[self.uid_b[uid], int(block)] = 1
        if self.issue == "desc":
            # [b2-desc] T4 as E's write-back marks: one mark per write-back event, in call order (the tick's write-backs
            # are recorded on the main stream in that order, so they complete in mark order); the block keeps the mark
            # of its event, as hist_ev keeps the event (None: no wait, as hist_ev None on the object path)
            if ev is None:
                self._hm[self.uid_b[uid], int(block)] = -1
            else:
                hit = self._hev.get(id(ev))
                if hit is None or hit[0] is not ev:                   # a new event: the next mark (first-seen order)
                    self._hmark += 1
                    self.tp.mark_hist(self._hmark, ev)
                    hit = (ev, self._hmark)
                    self._hev[id(ev)] = hit
                    if len(self._hev) > 256:                          # keep the recent events only (marks only grow)
                        for k in [k for k, (_, m) in self._hev.items() if m < self._hmark - 128]:
                            del self._hev[k]
                self._hm[self.uid_b[uid], int(block)] = hit[1]
        self._changed()

    def end_request(self, uid) -> None:
        """res.py:220-234 + own.py:363-384 on the table: b's LRU / PO / VRES -> RELEASING ("end"), its pins cleared, the
        owner none (its transfers in flight are dropped at arrival); the T4 maps; the queued keys cancelled."""
        b = self.uid_b.pop(uid)
        n = O.end_owner(self.own, b)
        self._count("release:end", n)
        self._owner_py[b] = None
        for k in [k for k in self.hist_ev if k[0] == b]:
            del self.hist_ev[k]
        self.hist_from.pop(b, None)
        self.own.hist_id[b] = -1
        if self._hm is not None:                                      # [b2-desc] as hist_ev: b's write-backs dropped
            self._hm[b] = -1
        self.own.hist_from[b] = 1 << 40
        for (key, l) in list(self.queued):
            if key[0] == uid:
                self.cancelled.add(key)
        self.queue = [(k, l) for (k, l) in self.queue if k[0] != uid]
        for (bb, kk), key in list(self._key_of.items()):
            if bb == b:
                del self._key_of[(bb, kk)]
                self._kslot.pop(key, None)
        self._changed()

    def cancel(self, key: Tuple) -> None:
        """res.py:768-785: the key refused from now on, its references queued at every layer; its transfers in flight
        KEEP their slots. free_on_cancel (NEGATIVE CONTROL, F3's): their destinations are released at once."""
        self.cancelled.add(key)
        self.metrics.cancelled(key)
        if self.b2_mut.get("free_on_cancel"):
            G = self.own.geo
            for tr in list(self.inflight.values()):
                if tr.requester == key:
                    f = torch.tensor([((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot], device=self.device)
                    self.own.flat("code")[f] = C_RELEASING
                    self.own.flat("rd_last")[f] = -1
                    self._count("release:cancel", 1)
                    if self.pending.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
                        del self.pending[(tr.layer, tr.b, tr.h, tr.block)]
            self._changed()
        if not self.mut.leak_cancelled_protection:
            for l in range(self.L):
                self._queue(key, l)

    def watch_flag(self, name: str, flag: torch.Tensor) -> None:
        if self.b2_mut.get("ignore_flags"):
            return None                                               # NEGATIVE CONTROL (F16's)
        return super().watch_flag(name, flag)

    def _count(self, kind: str, n: int) -> None:
        if n:
            self.journal_counts[kind] = self.journal_counts.get(kind, 0) + int(n)

    # [hostfix] --host-profile (S-C's host_prof, a callable (category, ns); None: no clock is read, no record)
    def _hp0(self) -> int:
        return time.perf_counter_ns() if self.host_prof is not None else 0

    def _hp(self, cat: str, t0: int) -> int:
        """Records the host ns since t0 under cat (when profiling); returns the clock now (the next part's t0)."""
        if self.host_prof is None:
            return 0
        t = time.perf_counter_ns()
        self.host_prof(cat, t - t0)
        return t

    # ================================================================================================================
    # poll
    # ================================================================================================================
    def poll(self) -> None:
        """res.py:283-297 on the table, in v1's order: the arrivals, then the queued drops, then the releases (the poll
        order is a contract: poll_order_swapped is its NEGATIVE CONTROL). Checked: I1 / I2 after the arrivals."""
        t = self._hp0()                                               # [hostfix] --host-profile (0: off)
        if self.b2_mut.get("poll_order_swapped"):
            self._poll_drops()
            self._poll_arrivals()
        else:
            self._poll_arrivals()
            t = self._hp("h_poll_arrivals", t)
            self._poll_drops()
            t = self._hp("h_poll_drops", t)
        self._poll_releases()
        t = self._hp("h_poll_releases", t)
        if not self.lean:                                             # [hostfix] --lean-time: a REPORT sample only
            self.metrics.sample_memory()
        self._hp("h_poll_memory", t)

    def _poll_arrivals(self) -> int:
        """res.py:245-264: every transfer whose own completion is observed (or landed in place: an urgent gather) is
        admitted, by the arrival rule (_arrived, note 12k); the admission branches are O.poll_arrivals' (own.py:322-361),
        on the slots of the arrived transfers only."""
        if self.inflight_array:                                       # [inflight-array] (off: the old lines below)
            return self._poll_arrivals_array()
        self.tp.poll()
        order = sorted(self.inflight.values(), key=lambda tr: (tr.layer, tr.b, tr.h, tr.seq))
        arrived = []
        for tr0 in order:
            tr = self.inflight.get(tr0.seq)
            if tr is None:
                continue
            if tr.seq not in self._landed and not self._arrived(tr):
                continue
            arrived.append(tr)
        if not arrived:
            return 0
        T, G = self.own, self.own.geo
        f = torch.tensor([((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot for tr in arrived],
                         dtype=torch.int64, device=self.device)
        gen = T.flat("gen")[f].tolist()
        code = T.flat("code")[f].tolist()
        blk = T.flat("block")[f].tolist()
        for i, tr in enumerate(arrived):                              # I6 (own.py:328-333): the arrival names its slot
            if gen[i] != tr.gen:
                raise StaleGeneration("arrival (uid %s, block %d) names slot %d gen %d but the slot holds gen %d"
                                      % (tr.uid, tr.block, tr.slot, tr.gen, gen[i]))
            if code[i] != C_INFLIGHT or blk[i] != tr.block:
                raise OwnershipViolation("arrival (uid %s, block %d, slot %d gen %d): slot is %s/%d"
                                         % (tr.uid, tr.block, tr.slot, tr.gen, STATE_OF[code[i]], blk[i]))
        T.flat("landed")[f] = True                                    # the transport's arrivals (urgent: landed)
        mut = {}
        if self.b2_mut.get("keep_rd_last_on_drop"):
            mut["keep_rd_last_on_drop"] = True
        if self.b2_mut.get("rec_mut") == "evict_po_unpinned":         # the admissions' K1 eviction (own.py:347-351)
            mut["evict_po_unpinned"] = True
        if self.b2_mut.get("no_uid_check"):                           # NEGATIVE CONTROL (T3): every arrival owned
            T.flat("uid")[f] = T.owner[torch.div(torch.div(f, G.NPOS, rounding_mode="floor"), G.H,
                                                 rounding_mode="floor").remainder(G.B)]
        res = O.poll_arrivals(T, self.tick_no, -1, None, None, mut or None)
        if int(res.get("forced_po", 0)) > 0 and self.rec_mut_done is None:
            self.rec_mut_done = (self.tick_no, "evict_po_unpinned")   # the control had an effect at this poll
        self._count("admit", int(res["admit_lru"]))
        self._count("admit_vres", int(res["admit_vres"]))
        self._count("drop_unowned", int(res["drop_unowned"]))
        self._count("drop_vres_unprotected", int(res["drop_vres_unprotected"]))
        self._count("evict_po", int(res["evict_po"]))
        self._count("release:evict", int(res["evict_rel"]))
        if int(res["admitted"]) != len(arrived):
            raise OwnershipViolation("B2 poll: %d slots admitted for %d arrived transfers" % (int(res["admitted"]),
                                                                                            len(arrived)))
        for tr in arrived:
            self.inflight.pop(tr.seq, None)
            if self.pending.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
                del self.pending[(tr.layer, tr.b, tr.h, tr.block)]
            self._landed.discard(tr.seq)
            if not self.lean:                                         # [hostfix] --lean-time: 4.8 / 4.3 REPORT only
                self.metrics.arrived(tr, None, tr.requester in self.cancelled)
            elif tr.requester in self.cancelled:                      # (the cheap count of Metrics.arrived is kept)
                self.metrics.arrivals_after_cancel += 1
        self._changed()
        if self.checked:
            pc = O.post_checks(T, None, 0, which=("I1", "I2", "I3"))
            fails = [n for n, v in zip(("I1", "I2", "I3"), torch.stack([pc["I1"], pc["I2"], pc["I3"]]).tolist()) if v]
            if fails:
                raise OwnershipViolation("B2 poll: after the arrivals %s fails (I1: one slot per resident or in-flight "
                                         "block; I2: the LRU holds at most C; I3: a PO / VRES slot is pinned)" % fails)
        return len(arrived)

    # ---------------------------------------------------------------------------------------------- [inflight-array]
    def _ia_add(self, takes: Dict, by_layer: Dict[int, List[Transfer]], rows_of: Dict[int, List[int]]) -> None:
        """[inflight-array] The draft takes of this tick as issue parts, one per layer in end_tick's issue order
        (sorted layers, as _host_side appends them to _deferred): the layer, the SAME transfer list that end_tick
        issues, and the takes' (flat slot, generation, block) columns. rows_of[l]: the rows of write_takes' columns
        (takes) of layer l's transfers, in the list's order. ONE host-to-device copy of the row indices and ONE
        index_select on the device (no read)."""
        order: List[int] = []
        parts = []
        for l in sorted(by_layer):
            idx = rows_of[l]
            if len(idx) != len(by_layer[l]):
                raise OwnershipViolation("inflight-array: layer %d issues %d transfers but names %d take rows"
                                         % (l, len(by_layer[l]), len(idx)))
            parts.append(_IAPart(l, by_layer[l], len(order), len(order) + len(idx)))
            order.extend(idx)
        sel = self._h2d(torch.tensor(order, dtype=torch.int64))
        cols = torch.stack([takes[n].to(torch.int64) for n in ("f", "gen", "blk")]).index_select(1, sel)
        for p in parts:
            p.cols = cols[:, p.lo:p.hi]
        self._ia_groups.setdefault(self.tick_no, []).extend(parts)

    def _ia_force(self, tr) -> None:
        """[inflight-array] _arrived's force of a due transfer that has not landed (the same counts and clock)."""
        t0 = time.perf_counter_ns()
        self.tp.force(tr)
        self.arrival_forced += 1
        self.arrival_forced_ns += time.perf_counter_ns() - t0

    def _poll_arrivals_array(self) -> int:
        """[inflight-array] _poll_arrivals with the issue table (module docstring). The arrival rule is the old one: an
        urgent take (in _landed) is admitted at the next poll; a draft transfer at the first poll of tick issue_tick +
        arrival_lag or later, forced there if its issue has not landed; one landed earlier is counted as held at every
        poll before. The admitted set is the old one. After the transport's poll: (1) the urgent takes in flight; (2)
        per issue part, its transfers still in flight (one C-level membership map); a due part joins the arrivals, a
        part not due counts its held transfers; (3) the COVERAGE CHECK; (4) the forces; (5) ONE vectorized I6 check;
        (6) the table's admission (the lines of _poll_arrivals); (7) the host maps, in bulk."""
        self.tp.poll()
        tp, infl, T, G = self.tp, self.inflight, self.own, self.own.geo
        has = infl.__contains__
        lag, now = self.arrival_lag, self.tick_no
        ev1 = bool(getattr(tp, "one_event_per_issue", False))
        mu = self.b2_mut
        st = self.ia_stats
        st["polls"] += 1
        early = 1 if mu.get("ia_due_early") else 0                    # NEGATIVE CONTROL: due one tick early
        lan = [infl[s] for s in self._landed if has(s)]               # (1) the urgent takes in flight
        n_cover = len(lan)
        due, due_keys = [], []
        for k in sorted(self._ia_groups):                             # (2) the issue parts, oldest tick first
            is_due = k + lag - early <= now
            for p in self._ia_groups[k]:
                if p.seqs is None:
                    p.seqs = list(map(_SEQ, p.trs))
                alive = list(map(has, p.seqs))
                na = sum(alive)
                n_cover += na
                if not na:
                    continue
                if not self._landed.isdisjoint(p.seqs):
                    raise OwnershipViolation("inflight-array: a draft transfer of tick %d (layer %d) is in the urgent "
                                             "set: it would arrive by two rules" % (k, p.l))
                if is_due:
                    if early and k + lag > now and self.ia_mut_done is None:
                        self.ia_mut_done = (now, "ia_due_early")
                    due.append([p, alive, na])
                elif ev1:                                             # one event per issue: the last transfer's
                    st["group_checks"] += 1                           # done is the issue's (set in list order)
                    if tp.done(p.trs[-1]):
                        self.arrival_held += na
                else:
                    for tr in compress(p.trs, alive):
                        st["transfer_checks"] += 1
                        if tp.done(tr):
                            self.arrival_held += 1
            if is_due:
                due_keys.append(k)
        if n_cover != len(infl):                                      # (3) every transfer in flight is held here
            raise OwnershipViolation("inflight-array: %d transfers are in flight, but the issue table and the urgent "
                                     "set hold %d: a transfer entered or left the in-flight map outside the take path"
                                     % (len(infl), n_cover))
        if mu.get("ia_skip_one") and due and self.ia_mut_done is None:   # NEGATIVE CONTROL: one transfer left out
            p, alive, na = due[0]
            alive = list(alive)
            alive[alive.index(True)] = False
            due[0] = [p, alive, na - 1]
            self.ia_mut_done = (now, "ia_skip_one")
        fb = []                                                       # (4) the forces of the due transfers
        for p, alive, na in due:
            st["parts_due"] += 1
            if not na:
                continue
            if ev1:
                st["group_checks"] += 1
                if not tp.done(p.trs[-1]):
                    self._ia_force(p.trs[-1])
            else:
                fb.extend(compress(p.trs, alive))
        if fb:
            fb.sort(key=_ORDER)                                       # the old poll's order (its force order)
            for tr in fb:
                st["transfer_checks"] += 1
                if not tp.done(tr):
                    self._ia_force(tr)
        n_arr = len(lan) + sum(x[2] for x in due)
        if n_arr == 0:
            for k in due_keys:
                del self._ia_groups[k]
            return 0
        cols = []                                                     # (5) ONE I6 check on the device
        if lan:
            fl = [((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot for tr in lan]
            cols.append(self._h2d(torch.tensor([fl, [tr.gen for tr in lan], [tr.block for tr in lan]],
                                               dtype=torch.int64)))
        for p, alive, na in due:
            if na == len(p.trs):
                cols.append(p.cols)
            elif na:
                keep = list(compress(range(len(alive)), alive))
                cols.append(p.cols.index_select(1, self._h2d(torch.tensor(keep, dtype=torch.int64))))
        C3 = torch.cat(cols, 1) if len(cols) > 1 else cols[0]
        f, gexp, bexp = C3[0], C3[1], C3[2]
        if mu.get("ia_bad_gen") and self.ia_mut_done is None:         # NEGATIVE CONTROL: a wrong expected generation
            gexp = gexp.clone()
            gexp[0] += 1
            self.ia_mut_done = (now, "ia_bad_gen")
        bad = ((T.flat("gen")[f].to(torch.int64) != gexp) | (T.flat("code")[f].to(torch.int64) != C_INFLIGHT)
               | (T.flat("block")[f].to(torch.int64) != bexp))
        if bool(bad.any()):
            self._ia_raise(lan, due)
        # (6) the table's admission: the lines of _poll_arrivals, with n_arr arrivals
        T.flat("landed")[f] = True                                    # the transport's arrivals (urgent: landed)
        mut = {}
        if mu.get("keep_rd_last_on_drop"):
            mut["keep_rd_last_on_drop"] = True
        if mu.get("rec_mut") == "evict_po_unpinned":                  # the admissions' K1 eviction (own.py:347-351)
            mut["evict_po_unpinned"] = True
        if mu.get("no_uid_check"):                                    # NEGATIVE CONTROL (T3): every arrival owned
            T.flat("uid")[f] = T.owner[torch.div(torch.div(f, G.NPOS, rounding_mode="floor"), G.H,
                                                 rounding_mode="floor").remainder(G.B)]
        res = O.poll_arrivals(T, self.tick_no, -1, None, None, mut or None)
        if int(res.get("forced_po", 0)) > 0 and self.rec_mut_done is None:
            self.rec_mut_done = (self.tick_no, "evict_po_unpinned")   # the control had an effect at this poll
        self._count("admit", int(res["admit_lru"]))
        self._count("admit_vres", int(res["admit_vres"]))
        self._count("drop_unowned", int(res["drop_unowned"]))
        self._count("drop_vres_unprotected", int(res["drop_vres_unprotected"]))
        self._count("evict_po", int(res["evict_po"]))
        self._count("release:evict", int(res["evict_rel"]))
        if int(res["admitted"]) != n_arr:
            raise OwnershipViolation("B2 poll: %d slots admitted for %d arrived transfers" % (int(res["admitted"]),
                                                                                            n_arr))
        self._ia_forget(lan, due)                                     # (7) the host maps
        for k in due_keys:
            del self._ia_groups[k]
        st["admitted"] += n_arr
        self._changed()
        if self.checked:
            pc = O.post_checks(T, None, 0, which=("I1", "I2", "I3"))
            fails = [n for n, v in zip(("I1", "I2", "I3"), torch.stack([pc["I1"], pc["I2"], pc["I3"]]).tolist()) if v]
            if fails:
                raise OwnershipViolation("B2 poll: after the arrivals %s fails (I1: one slot per resident or in-flight "
                                         "block; I2: the LRU holds at most C; I3: a PO / VRES slot is pinned)" % fails)
        return n_arr

    def _ia_raise(self, lan, due) -> None:
        """[inflight-array] The vectorized I6 check failed: the old per-transfer check, in the old order, gives the old
        exception. If it passes, the issue table's columns disagree with its transfers: that is a defect too."""
        T, G = self.own, self.own.geo
        arrived = sorted(chain(lan, *[compress(p.trs, alive) for p, alive, _ in due]), key=_ORDER)
        f = torch.tensor([((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot for tr in arrived],
                         dtype=torch.int64, device=self.device)
        gen = T.flat("gen")[f].tolist()
        code = T.flat("code")[f].tolist()
        blk = T.flat("block")[f].tolist()
        for i, tr in enumerate(arrived):                              # I6 (own.py:328-333): the arrival names its slot
            if gen[i] != tr.gen:
                raise StaleGeneration("arrival (uid %s, block %d) names slot %d gen %d but the slot holds gen %d"
                                      % (tr.uid, tr.block, tr.slot, tr.gen, gen[i]))
            if code[i] != C_INFLIGHT or blk[i] != tr.block:
                raise OwnershipViolation("arrival (uid %s, block %d, slot %d gen %d): slot is %s/%d"
                                         % (tr.uid, tr.block, tr.slot, tr.gen, STATE_OF[code[i]], blk[i]))
        raise OwnershipViolation("inflight-array: the vectorized arrival check fails where the per-transfer check "
                                 "passes: the issue table's columns (flat slot, generation, block) do not match its "
                                 "transfers")

    def _ia_forget(self, lan, due) -> None:
        """[inflight-array] The admitted transfers leave the host maps (the old loop's effect): the urgent takes per
        transfer (few); each due part in bulk: the in-flight map (InflightByLayer.pop_layer_many, or dict.pop mapped),
        the pending map (only the keys that still name the transfer), the metrics (not lean: Metrics.arrived per
        transfer in the old order; lean: the count of arrivals after a cancel)."""
        infl, pend = self.inflight, self.pending
        pop_many = getattr(infl, "pop_layer_many", None)
        for tr in lan:
            infl.pop(tr.seq, None)
            if pend.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
                del pend[(tr.layer, tr.b, tr.h, tr.block)]
            self._landed.discard(tr.seq)
        gone = []
        for p, alive, na in due:
            if not na:
                continue
            if na == len(p.trs):
                trs, seqs = p.trs, p.seqs
            else:
                trs, seqs = list(compress(p.trs, alive)), list(compress(p.seqs, alive))
            if pop_many is not None:
                pop_many(p.l, seqs)
            else:
                deque(map(infl.pop, seqs), maxlen=0)
            keys = list(zip(repeat(p.l), map(_B, trs), map(_H, trs), map(_BLK, trs)))
            cur = list(map(pend.get, keys))
            if all(map(is_, cur, trs)):
                deque(map(pend.__delitem__, keys), maxlen=0)
            else:
                for key, c, tr in zip(keys, cur, trs):
                    if c is tr:
                        del pend[key]
            gone.append(trs)
        if not self.lean:                                             # [hostfix] --lean-time: 4.8 / 4.3 REPORT only
            for tr in sorted(chain(lan, *gone), key=_ORDER):
                self.metrics.arrived(tr, None, tr.requester in self.cancelled)
        elif self.cancelled:                                          # (the cheap count of Metrics.arrived is kept)
            self.metrics.arrivals_after_cancel += sum(map(self.cancelled.__contains__, map(_REQ, chain(lan, *gone))))

    def _poll_drops(self) -> None:
        """res.py:287-293: the queued (key, l) releases whose key reader completed; O.drop_keys on the table."""
        keep, due = [], []
        for (key, l) in self.queue:
            if self.mut.prot_release_no_reader_gate or ev_done(self.key_reader.get((key, l))):
                due.append((key, l))
            else:
                keep.append((key, l))
        self.queue = keep
        trip = []
        for (key, l) in due:
            b = self.uid_b.get(key[0])
            k = self._kslot.get(key)
            if b is None or k is None:
                continue                                              # the request ended / the key never pinned
            trip.append((l, b, k))
        trip = sorted(set(trip))
        if trip:
            d = self.device
            res = O.drop_keys(self.own, torch.tensor([t[0] for t in trip], device=d),
                              torch.tensor([t[1] for t in trip], device=d), torch.tensor([t[2] for t in trip], device=d))
            self._count("release:unprotect", res["unprotect_rel"])
            self._changed()

    def _poll_releases(self) -> None:
        """own.py:435-452: RELEASING -> FREE behind each slot's last reader (the reader window, refreshed)."""
        self.win.refresh(ev_done)
        if self.b2_mut.get("no_reader_wait") or self.mut.no_reader_wait:     # NEGATIVE CONTROL
            code = self.own.flat("code")
            fr = (code == C_RELEASING).nonzero().squeeze(1)
            for n, v in (("code", C_FREE), ("block", -1), ("uid", -1), ("cls", 0)):
                self.own.flat(n)[fr] = v
            n = int(fr.numel())
        else:
            n = O.poll_releases(self.own, self.win)
        self._count("free", n)
        if n:
            self._changed()

    # ================================================================================================================
    # the host consumer
    # ================================================================================================================
    def apply_tick(self, steps: List[Dict]) -> None:
        """THE HOST CONSUMER (C's signature, note 12l): ONE read of the flags and the record counts; the void-tick
        refusals (a watched flag, a verify overflow, a record over its capacity) before any change; then the B2 ops
        (module docstring) on all layers at once, in K3X order; then the host side (the transfers, the metrics, the
        journal, the release queue)."""
        if not steps:
            return
        t_hp = self._hp0()                                            # [hostfix] --host-profile (0: off)
        names = list(self._flags)
        parts = [self._flags[n].to(torch.int64).reshape(1) for n in names]
        for st in steps:
            rec = st["rec"]
            if st["vdec"] is not None:                                # (the looks too: no read per layer, SG5 b2)
                parts += [rec["vcnt"].reshape(1), st["vdec"].counters.reshape(-1)[:1], rec["vlook"].reshape(-1)]
            if st["ddec"] is not None:
                parts += [rec["dcnt"].reshape(1), st["ddec"].counters.reshape(-1)[:3]]
        flat = self._d2h(torch.cat(parts)).tolist()                   # ONE read (the flags, the counts, the looks)
        self.flag_values = dict(zip(names, flat[:len(names)]))
        raised = sorted(n for n, v in self.flag_values.items() if v)
        if raised and raised != ["xpool_err"]:                        # ([xpool]: xpool_err alone waits, see below)
            raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                  "of its decisions is applied)" % raised)
        at = len(names)
        info = []
        over, capped = [], []
        for st in steps:
            cap_v, cap_d = st["caps"]
            p = dict(st=st, nv=0, nd=0, over=0, dc=(0, 0, 0))
            if st["vdec"] is not None:
                p["nv"], p["over"] = int(flat[at]), int(flat[at + 1])
                at += 2
                Rv = int(st["vdec"].Rv)
                p["vlook"] = [flat[at + 4 * jj:at + 4 * jj + 4] for jj in range(Rv)]   # (Rv, 4): rec["vlook"]
                at += 4 * Rv
                if p["nv"] > cap_v:
                    capped.append((st["l"], "verify", p["nv"], cap_v))
                if p["over"] > 0:
                    over.append((st["l"], p["over"]))
            if st["ddec"] is not None:
                p["nd"] = int(flat[at])
                p["dc"] = tuple(int(x) for x in flat[at + 1:at + 4])
                at += 4
                if p["nd"] > cap_d:
                    capped.append((st["l"], "draft", p["nd"], cap_d))
            info.append(p)
        rm = self.b2_mut.get("rec_mut")
        if over and rm != "ignore_overflow":
            rep = [dict(st=p["st"], vrec=p["st"]["rec"]["vrec"][:, :p["nv"]].t().tolist()) for p in info
                   if p["st"]["vdec"] is not None]
            raise self._refusal(over, self._sparse_overflow_report(rep, capped))
        if capped:
            raise CapacityRefusal("S-C sparse record: a layer's record holds more entries than its capacity (layer, "
                                  "kind, entries, capacity: %s): the tick is void (none of its decisions is applied)"
                                  % capped)
        if raised:                                                    # [xpool] after the overflow (review m7)
            raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                  "of its decisions is applied)" % raised)
        self._hp("h_apply_read", t_hp)                                # the ONE read and its parse
        self._apply_b2(steps, info)
        self.b2_applied += 1

    # ---------------------------------------------------------------------------------------------- the B2 apply
    def _tick_rows_of(self, st: Dict) -> List[Row]:
        """The tick's rows in their order (verify rows at vidx, the draft rows in the other positions)."""
        R = len(st["vrows"]) + len(st["drows"])
        out: List[Any] = [None] * R
        for j, i in enumerate(st["vidx"]):
            out[i] = st["vrows"][j]
        it = iter(st["drows"])
        for i in range(R):
            if out[i] is None:
                out[i] = next(it)
        return out

    # ---------------------------------------------------------------------------------------------- [hostfix] FEWER READS
    def _fr_rows(self, vrows, drows, trows):
        """[hostfix] --b2-fewer-reads: the tick's host row values in ONE pinned non-blocking host-to-device copy
        (S-C's _h2d) instead of six torch.tensor(..., device=d) copies (each a stream synchronization on CUDA): vb,
        vk, db, dk (O.Rows), row_b, row_plane (pins_or), and pins_or's ROUNDS computed on the host: rank_in_group of
        plane * B + b with every mask bit set is the occurrence index of the row's (plane, b) among the earlier rows,
        so round r = the rows (ascending) whose occurrence index is r, and the round count is the largest group's
        size: the values (rnd == r).nonzero() and int(rnd.max()) + 1 give. Also recency's round count: the largest
        number of draft rows of one batch slot (rank_in_group of db). Returns ((vb, vk, db, dk, row_b, row_plane),
        [round index tensors], recency rounds)."""
        vbl = [r.b for r in vrows]
        vkl = [self._kslot[r.key] for r in vrows]
        dbl = [r.b for r in drows]
        dkl = [self._kslot[r.key] for r in drows]
        rbl = [r.b for r in trows]
        rpl = [self._kslot[r.key] + (0 if r.kind == "v" else self.KEYS) for r in trows]
        seen: Dict[int, int] = {}
        rank = []
        for p, b in zip(rpl, rbl):
            g = p * self.B + b
            rank.append(seen.get(g, 0))
            seen[g] = rank[-1] + 1
        n_pin = max(rank) + 1 if rank else 0
        rounds = [[i for i, x in enumerate(rank) if x == r] for r in range(n_pin)]
        cnt: Dict[int, int] = {}
        for b in dbl:
            cnt[b] = cnt.get(b, 0) + 1
        n_rec = max(cnt.values()) if cnt else 0
        parts = [vbl, vkl, dbl, dkl, rbl, rpl] + rounds
        dev = self._h2d(torch.tensor([x for p in parts for x in p], dtype=torch.int64))
        out, at = [], 0
        for p in parts:
            out.append(dev[at:at + len(p)])
            at += len(p)
        return tuple(out[:6]), out[6:], n_rec

    def _fr_read(self, cols):
        """[hostfix] --b2-fewer-reads: ONE host read of several 1-D integer tensors (each cast to int64: exact for the
        integer and flag columns read here), split back into lists in order. Not the counted decision read (_d2h)."""
        cols = [c.reshape(-1).to(torch.int64) for c in cols]
        if not cols:
            return []
        flat = torch.cat(cols).tolist()
        out, at = [], 0
        for c in cols:
            n = c.numel()
            out.append(flat[at:at + n])
            at += n
        return out

    @staticmethod
    def _fr_merge_sel(V, D):
        """[frfix] --b2-fewer-reads: the demand merges' record columns (l, b, h, blk) selected ON THE DEVICE: the verify
        records with INFL or DUP, then the draft records with INFL, each part in record order (the entries that
        V["l"][m] and D["l"][m] give). ONE nonzero over both parts' masks (one synchronization: the selection's size),
        then one index_select. Returns (sel (4, n) int64, nv (1,) int64: the verify part's count), both for the caller's
        ONE read; _fr_merge_split makes the host lists of that read. The full record never crosses."""
        cols, masks = [], []
        nv = None
        for X, m_of in ((V, lambda f: O.has(f, O.INFL) | O.has(f, O.DUP)), (D, lambda f: O.has(f, O.INFL))):
            if X is None:
                continue
            m = m_of(X["flags"])
            if nv is None:                                            # the first part present: V's count, or 0
                nv = m.sum().reshape(1) if X is V else torch.zeros(1, dtype=torch.int64, device=m.device)
            cols.append(torch.stack([X[n].to(torch.int64) for n in ("l", "b", "h", "blk")]))
            masks.append(m)
        if not cols:
            return None, None
        idx = torch.cat(masks).nonzero().squeeze(1)                   # ascending: V's entries, then D's
        return torch.cat(cols, 1).index_select(1, idx), nv.to(torch.int64)

    @staticmethod
    def _fr_merge_split(flat, nv):
        """[frfix] the host lists of _fr_merge_sel's read: flat = sel.reshape(-1).tolist() (l, b, h, blk rows of n
        entries each), nv the verify count. Returns (verify merges, draft merges), lists of (l, b, h, blk)."""
        n = len(flat) // 4
        m = list(zip(flat[:n], flat[n:2 * n], flat[2 * n:3 * n], flat[3 * n:]))
        return m[:nv], m[nv:]

    def _apply_b2(self, steps: List[Dict], info: List[Dict]) -> None:
        t_hp = self._hp0()                                            # [hostfix] --host-profile (0: off)
        T, G = self.own, self.own.geo
        d = self.device
        L, B, H = self.L, self.B, self.H
        st0 = steps[0]
        vrows, drows = st0["vrows"], st0["drows"]
        trows = self._tick_rows_of(st0)
        for st in steps:                                              # every layer resolved the same rows (S-C)
            if st["vrows"] != vrows or st["drows"] != drows:
                raise ValueError("B2: the layers of a tick resolved different rows")
        # the key slots of the tick's rows (I13: a refusal before any change)
        for r in trows:
            self._key_slot(r.key)
        k = (st0["vdec"] or st0["ddec"]).k
        fr = self.fewer_reads                                         # [hostfix] --b2-fewer-reads
        pin_rounds = rec_rounds = None
        if fr:
            (vb, vk, db, dk, row_b, row_plane), pin_rounds, rec_rounds = self._fr_rows(vrows, drows, trows)
        else:
            vb = torch.tensor([r.b for r in vrows], dtype=torch.int64, device=d)
            vk = torch.tensor([self._kslot[r.key] for r in vrows], dtype=torch.int64, device=d)
            db = torch.tensor([r.b for r in drows], dtype=torch.int64, device=d)
            dk = torch.tensor([self._kslot[r.key] for r in drows], dtype=torch.int64, device=d)
        rows = O.Rows(vb=vb, vk=vk, db=db, dk=dk, k=int(k))
        layers = []
        for p in info:
            st = p["st"]
            rec = st["rec"]
            layers.append(dict(l=st["l"], k=int((st["vdec"] if st["vdec"] is not None else st["ddec"]).k),
                               vrec=rec["vrec"][:, :p["nv"]].clone() if st["vdec"] is not None else None,
                               drec=rec["drec"][:, :p["nd"]].clone() if st["ddec"] is not None else None))
        Wb, Wp = G.WB, G.WP
        R = len(trows)
        rowsel = torch.zeros((L, R, H, Wb), dtype=torch.int64, device=d)
        sel = torch.zeros((L, B, H, Wb), dtype=torch.int64, device=d)
        vread = torch.zeros((L, B, H, Wp), dtype=torch.int64, device=d)
        vlook = {}
        for p in info:
            st = p["st"]
            rec = st["rec"]
            l = st["l"]
            rowsel[l] = rec["rowsel"]
            sel[l] = rec["sel"]
            if st["vdec"] is not None:
                vread[l] = rec["vread"]
                vlook[l] = p["vlook"]                                 # read with the counts (apply_tick)
        if not fr:                                                    # ([hostfix] fewer reads: built in _fr_rows)
            row_b = torch.tensor([r.b for r in trows], dtype=torch.int64, device=d)
            row_plane = torch.tensor([self._kslot[r.key] + (0 if r.kind == "v" else self.KEYS) for r in trows],
                                     dtype=torch.int64, device=d)
        self._rec_mutate(layers, rowsel, vread, rows, row_b, row_plane)
        P = O.parse(T, layers, rows)
        ev = O.k5_events(P, T)
        so = O.slot_of(T)
        checked = self.checked
        mut = {}
        if self.b2_mut.get("one_batch"):
            mut["one_batch"] = True
        rm = self.b2_mut.get("rec_mut")
        if rm in ("reverse_stamps", "evict_po_unpinned", "gen_plus_two"):
            mut[rm] = True
        # ---------------------------------------------------------------- the pre-write predicates (one read)
        pre: Dict[str, torch.Tensor] = {}

        def note(dd):
            for n, v in dd.items():
                pre[n] = v.sum() if v.dim() else v.to(torch.int64)
        for part in ("V", "D"):
            if P[part] is not None:
                pre["hole_" + part] = O.has(P[part]["flags"], O.HOLE).sum()
        mc, fv, fd = O.merge_check(T, P, so, ev, check=checked)
        if checked:
            note(O.k5_check(T, ev))
            tc = O.take_checks(T, ev, so)
            if self.b2_mut.get("no_dedup_check"):
                tc.pop("dedup")
                tc.pop("take_twice")
            if self.b2_mut.get("no_t4"):
                tc.pop("t4")
            note(tc)
            if P["D"] is not None:
                note(O.ready_check(T, P["D"]))
            note(mc)
            note(O.policy_e_check(T, ev, P, sel, O.any_pin_words(T), self.policy_e_inflight))
        else:
            tc = O.take_checks(T, ev, so)                             # T4 and the holes stay refusals unchecked
            note(dict(t4=tc["t4"]) if not self.b2_mut.get("no_t4") else {})
        # the K5 REPORT (S-C's _extra_need, report()["k5"]; SG5 b2): per stream its verify takes (GRANT / EVICT)
        # beyond its FREE base slots C - usage at the tick's start, summed per layer; read with the predicates
        need_l = None
        if P["V"] is not None:
            V0 = P["V"]
            tv = O.has(V0["flags"], O.GRANT) | O.has(V0["flags"], O.EVICT)
            s_tv = V0["s"][tv]
            nf = torch.zeros(G.S, dtype=torch.int64, device=d).index_add_(0, s_tv, torch.ones_like(s_tv))
            use0 = (T.c["code"] != C_FREE).view(G.S, G.NPOS).sum(1)
            need_l = (nf - (self.C - use0).clamp(min=0)).clamp(min=0).view(L, B * H).sum(1)
        names = list(pre)
        parts_pre = [torch.stack([pre[n] for n in names]).to(torch.int64)] if names else []
        if need_l is not None:
            parts_pre.append(need_l.to(torch.int64))
        evh = None
        if fr:
            # [hostfix] --b2-fewer-reads: the K5 events' columns and the table at every event's take slot ride in the
            # predicates' ONE read. Nothing is written between here and the old reads below (the victims' and the 4.8
            # counts: the writes start after them), so the values are the ones those reads give; a refusal event's
            # take slot is read too and never used
            f_all = ev["s"] * G.NPOS + ev["take_slot"].clamp(0, G.NPOS - 1)   # (a refusal's sentinel stays in
            #                                                         its stream; a take's slot is in range already)
            ext = [ev[n] for n in ("kind", "l", "b", "h", "part", "j", "src")]
            ext += [T.flat(n)[f_all] for n in ("code", "block", "rd_last")]
            if not self.lean:
                ext += [T.flat(n)[f_all] for n in ("gen", "rd_n", "rv_n")]
            got = self._fr_read(([torch.cat(parts_pre)] if parts_pre else []) + ext)
            vals = got[0] if parts_pre else []
            evh = dict(zip(("kind", "l", "b", "h", "part", "j", "src", "code", "block", "rd_last", "gen", "rd_n",
                            "rv_n"), got[1 if parts_pre else 0:]))
        else:
            vals = torch.cat(parts_pre).tolist() if parts_pre else []
        need = vals[len(names):]
        vals = vals[:len(names)]
        viol = {n: int(v) for n, v in zip(names, vals) if v}
        if viol:
            self._raise_pre(viol, P, ev, rows)
        if not fr or checked:                                         # ([hostfix] fewer reads: n_ref is read only
            n_ref = int(O.has(P["D"]["flags"], O.REFUSED).sum()) if P["D"] is not None else 0   # where it is used)
        dev_ref = sum(p["dc"][1] for p in info if p["st"]["ddec"] is not None)
        if checked and n_ref != dev_ref:
            raise OwnershipViolation("S-B: %d refusals applied, the device counted %d" % (n_ref, dev_ref))
        # ---------------------------------------------------------------- the victims of Policy E (before the writes)
        victims = []
        if fr:                                                        # [hostfix] from the ONE read above, same order
            victims = [(evh["l"][q], evh["b"][q], evh["h"][q], evh["block"][q], evh["code"][q], evh["rd_last"][q])
                       for q, kd in enumerate(evh["kind"]) if kd == 1]
        else:
            pe = (ev["kind"] == 1).nonzero().squeeze(1)
            if pe.numel():
                fpe = ev["s"][pe] * G.NPOS + ev["take_slot"][pe]
                vcode = T.flat("code")[fpe].tolist()
                vblk = T.flat("block")[fpe].tolist()
                vrd = T.flat("rd_last")[fpe].tolist()
                for i, q in enumerate(pe.tolist()):
                    victims.append((int(ev["l"][q]), int(ev["b"][q]), int(ev["h"][q]), vblk[i], vcode[i], vrd[i]))
        # the take slots' generation and read counts before the writes (the 4.8 lifetime they end; SG5 b2), in the
        # order of write_takes' takes
        old = None                                                    # [hostfix] --lean-time: no 4.8 harvest
        if fr and not self.lean:                                      # [hostfix] host lists (og, ordn, orvn)
            tq = [q for q, kd in enumerate(evh["kind"]) if kd != 2]
            old = tuple([evh[n][q] for q in tq] for n in ("gen", "rd_n", "rv_n"))
        elif not self.lean:
            tkq = (ev["kind"] != 2).nonzero().squeeze(1)
            f_old = ev["s"][tkq] * G.NPOS + ev["take_slot"][tkq]
            old = torch.stack([T.flat(n)[f_old].to(torch.int64) for n in ("gen", "rd_n", "rv_n")])
        # ---------------------------------------------------------------- the writes
        self._changed()                                               # (a raise below leaves no stale projection)
        gen_before = T.c["gen"].clone() if checked else None
        undo = O.Undo() if checked else None
        if undo is not None:
            clock0 = T.clock
            undo.call(lambda: setattr(T, "clock", clock0))
        O.pins_or(T, rowsel, row_b, row_plane, undo, rounds=pin_rounds)       # [hostfix] rounds: None = on the device
        win_mut = dict(mut)
        if self.b2_mut.get("drop_rd_last"):
            win_mut["drop_rd_last"] = True
        takes = O.write_takes(T, ev, self.tick_no, self.tick_no, self.win, undo, win_mut)
        O.write_merges(T, fv, fd, undo)
        rc = O.recency(T, P["D"], rows, undo, mut, n_rounds=rec_rounds)      # [hostfix] None = on the device
        if rm in ("reverse_stamps", "evict_po_unpinned", "gen_plus_two") and self.rec_mut_done is None:
            hit = dict(reverse_stamps=rc.get("max_stream_reads", 0) >= 2,           # two touches of one stream
                       evict_po_unpinned=int(rc.get("forced_po", 0)) > 0,           # an unpinned victim made PO
                       gen_plus_two=takes["f"].numel() > 0)[rm]
            if hit:
                self.rec_mut_done = (self.tick_no, rm)                # the op-level control had an effect here
        if checked:
            post = dict(O.post_checks(T, gen_before, int(takes["f"].numel())))
            post["recency_bad"] = rc["bad"]
            pn = list(post)
            pv = torch.stack([post[n].to(torch.int64) for n in pn]).tolist()
            bad = {n: int(v) for n, v in zip(pn, pv) if v}
            if bad:
                undo.rollback()
                raise OwnershipViolation("B2 post-write predicates %s fail: the tick is void (rolled back)" % sorted(bad))
        elif int(rc["bad"]):
            raise OwnershipViolation("B2: a draft row read a block that is no longer resident at the recency update")
        if need_l is not None:                                        # the K5 REPORT of the committed tick
            for p in info:
                if p["st"]["vdec"] is not None:
                    self._note_extra_need(p["st"]["l"], int(need[p["st"]["l"]]))
        # ---------------------------------------------------------------- the readers (the layers' events)
        if fr:                                                        # [hostfix] a pinned non-blocking copy
            e_l = self._h2d(torch.tensor([self._rid(self._layer_ev.get(l)) for l in range(L)], dtype=torch.int64))
        else:
            e_l = torch.tensor([self._rid(self._layer_ev.get(l)) for l in range(L)], dtype=torch.int64, device=d)
        if self.b2_mut.get("rec_mut") == "drop_vread":
            pass                                                      # (the bit was cleared in _rec_mutate)
        O.read_done(T, vread, P["D"], rows, e_l)
        self._changed()
        self._hp("h_b2_ops", t_hp)                                    # [hostfix] parse .. read_done (ops and syncs)
        # ---------------------------------------------------------------- the host side
        self._host_side(steps, info, P, ev, takes, victims, rc, rows, vlook, fv, fd, old, evh=evh)

    def _raise_pre(self, viol: Dict[str, int], P, ev, rows) -> None:
        """A pre-write predicate failed: the tick is void before any change. The message names the predicates (and the
        vendored check each one repeats)."""
        what = {"hole_V": "a verify block has no slot and the device did not flag an overflow",
                "hole_D": "a draft block has no decision",
                "k5": "K5 (LayerPool.can_take) refuses a take or grants a refused one",
                "first_free": "a take is not the stream's first FREE slot (K3X first_free)",
                "slot_range": "a record names a slot outside the pool (I15)",
                "dedup": "a take of a block resident or in flight for the owner (I1, res.py:453)",
                "take_twice": "two takes of one (stream, block) in the tick",
                "t4": "block has no registered host write-back (history_written): its host rows may not exist yet",
                "ready": "a READY draft record names a slot that is not the resident slot of its block",
                "v_merge": "a verify merge names no owner's transfer in flight (slot, generation)",
                "v_dup": "a DUP record names no take of this tick", "d_merge": "a draft merge names no transfer",
                "policy_e": "a Policy E eviction is not the host's next candidate (I7)"}
        msg = "; ".join("%s x%d: %s" % (n, v, what.get(n, n)) for n, v in sorted(viol.items()))
        raise OwnershipViolation("B2 pre-write predicates fail (the tick is void, nothing applied): " + msg)

    def _host_side(self, steps, info, P, ev, takes, victims, rc, rows, vlook, fv, fd, old, evh=None) -> None:
        """After the commit: the transfers of the takes (vendored Transfer objects, on the transport), Policy E's
        victims (verifier_evictions; the superseded prefetches), the demand merges, the refusals, the looks (4.3 / 4.6),
        the journal, the release queue, the layer plans (masked, draft pairs, refusals). old (3, takes): each take
        slot's generation and read counts before the tick: the 4.8 lifetime that the take ends gets them (SG5 b2)."""
        G = self.own.geo
        V, D = P["V"], P["D"]
        vrows, drows = steps[0]["vrows"], steps[0]["drows"]
        t_hp = self._hp0()                                            # [hostfix] --host-profile (0: off)
        # Policy E's victims (res.py:399-418, :363-388)
        for (l, b, h, oblk, ocode, ord_) in victims:
            self.metrics.count("verifier_evictions")
            if ocode == C_INFLIGHT:
                tr = self.pending.get((l, b, h, oblk))
                if tr is not None:
                    if not self.tp.done(tr):
                        self.tp.force(tr)
                    self.inflight.pop(tr.seq, None)
                    del self.pending[(l, b, h, oblk)]
                    tr.superseded = True
                self.metrics.count("verifier_inflight_evictions")
                self._count("release:verifier_evict_inflight", 1)
            else:
                ev_r = self._evobj.get(ord_) if ord_ >= 0 else None
                if ev_r is not None and not ev_done(ev_r):
                    self.host_wait(ev_r)
                    self.metrics.count("verifier_force_waits")
                self._count("release:verifier_evict", 1)
            self._count("free", 1)
        t_hp = self._hp("h_hs_victims", t_hp)
        # the takes, in K3X order (the vendored _take's Transfer, res.py:476-489)
        vcols = rcl = fm = nref_fr = None
        if evh is not None:                                           # [hostfix] --b2-fewer-reads: ONE read of the
            vn = ("l", "b", "h", "blk", "flags", "j")                 # takes, the merges selected on the device
            cols = [takes[n] for n in ("l", "b", "h", "blk", "slot", "gen", "rd_old")]     # ([frfix]), the looks'
            looks = V is not None and not self.lean                                         # record columns (not
            cols += [V[n] for n in vn] if looks else []                                     # lean: the looks read
            msel, mnv = self._fr_merge_sel(V, D)                                            # all of V anyway), the
            cols += [msel, mnv] if msel is not None else []                                 # refusal count and the
            if D is not None and not self.lean:                                             # recency counts (none
                cols.append(O.has(D["flags"], O.REFUSED).sum())                             # changes after its op:
            cols.append(torch.stack([torch.as_tensor(rc[n]).to(device=self.device, dtype=torch.int64).reshape(())
                                     for n in ("touch", "acquire", "evict_po", "evict_rel")]))   # same values)
            got = self._fr_read(cols)
            lt, bt, ht, blk, slot, gen, rdo = got[:7]
            at = 7
            if looks:
                vcols = dict(zip(vn, got[at:at + len(vn)]), src=evh["src"])
                at += len(vn)
            if msel is not None:
                fm = self._fr_merge_split(got[at], got[at + 1][0])
                at += 2
            if D is not None and not self.lean:
                nref_fr = got[at][0]
            rcl = got[-1]
            tk = [q for q, kd in enumerate(evh["kind"]) if kd != 2]   # = (kind != 2).nonzero(), ascending
            part, jrow = evh["part"], evh["j"]
            og, ordn, orvn = old if old is not None else (None, None, None)
        else:
            tk = (ev["kind"] != 2).nonzero().squeeze(1).tolist()
            part = ev["part"].tolist()
            jrow = ev["j"].tolist()                                   # the record's row (decoded with its layer's k)
            lt, bt, ht = takes["l"].tolist(), takes["b"].tolist(), takes["h"].tolist()
            blk, slot, gen = takes["blk"].tolist(), takes["slot"].tolist(), takes["gen"].tolist()
            rdo = takes["rd_old"].tolist()
            og, ordn, orvn = old.tolist() if old is not None else (None, None, None)     # [hostfix] lean: no harvest
        harvest = not self.b2_mut.get("no_read_harvest") and old is not None
        lifetimes = self.metrics.lifetimes
        by_layer: Dict[int, List[Transfer]] = {}
        new_trs = []
        ia = self.inflight_array                                      # [inflight-array] the issue parts' take rows
        ia_rows: Dict[int, List[int]] = {}
        for i, q in enumerate(tk):
            j = jrow[q]
            isv = part[q] == 0
            r = vrows[j] if isv else drows[j]
            if harvest:                                               # the ended lifetime's reads (4.8, SG5 b2)
                rec = lifetimes.get((lt[i], bt[i], ht[i], slot[i], og[i]))
                if rec is not None:
                    rec["rd"], rec["rv"] = ordn[i], orvn[i]
            self.seq += 1
            prev = self._evobj.get(rdo[i]) if rdo[i] >= 0 else None
            if self.b2_mut.get("drop_rd_last"):
                prev = None
            tr = Transfer(self.seq, self._owner_py[bt[i]], lt[i], bt[i], ht[i], blk[i], slot[i], gen[i],
                          VERIFY if isv else DRAFT, r.key, hist_ev=self.hist_ev.get((bt[i], blk[i])), reader_ev=prev)
            tr.issue_tick = self.tick_no
            self.inflight[tr.seq] = tr
            self.pending[(lt[i], bt[i], ht[i], blk[i])] = tr
            if not self.lean:                                         # [hostfix] --lean-time: 4.8 REPORT only
                self.metrics.issued(tr)
            if isv:
                tr.delivered = True                                   # the device's urgent gather landed it
                self._landed.add(tr.seq)
            else:
                by_layer.setdefault(lt[i], []).append(tr)
                if ia:                                                # [inflight-array] row i of write_takes' columns
                    ia_rows.setdefault(lt[i], []).append(i)
            new_trs.append((q, tr))
        self._count("take", len(tk))
        if self.issue == "object":                                    # Residency.end_tick issues them, per layer
            for l in sorted(by_layer):
                self._deferred.append((l, by_layer[l]))
        if ia and by_layer:                                           # [inflight-array] the same lists, as issue parts
            self._ia_add(takes, by_layer, ia_rows)
        t_hp = self._hp("h_hs_takes", t_hp)                           # the take columns' reads and the per-take loop
        # the demand merges on the transfers (res.py:605-611, :672-674)
        if V is not None:
            if fm is not None:                                        # [frfix] selected on the device, in record
                merges = fm[0]                                        # order (_fr_merge_sel), from the ONE read
            else:
                m = O.has(V["flags"], O.INFL) | O.has(V["flags"], O.DUP)
                merges = zip(V["l"][m].tolist(), V["b"][m].tolist(), V["h"][m].tolist(), V["blk"][m].tolist())
            for (l, b, h, t) in merges:
                tr = self._in_flight(l, b, h, t)
                if tr is not None and not self.mut.drop_demand_merge:
                    tr.demand.add(VERIFY)
        if D is not None:
            if fm is not None:                                        # [frfix] selected on the device
                merges = fm[1]
            else:
                m = O.has(D["flags"], O.INFL)
                merges = zip(D["l"][m].tolist(), D["b"][m].tolist(), D["h"][m].tolist(), D["blk"][m].tolist())
            for (l, b, h, t) in merges:
                tr = self._in_flight(l, b, h, t)
                if tr is not None and not self.mut.drop_demand_merge:
                    tr.demand.add(DRAFT)
            if not self.lean:                                         # [hostfix] --lean-time: a REPORT count (a read)
                if nref_fr is not None:                               # [frfix] counted on the device, ONE read
                    nref = nref_fr
                else:
                    nref = int(O.has(D["flags"], O.REFUSED).sum())
                if nref:
                    self.metrics.count("draft_refused", nref)
        # the journal of the recency (touch, acquire, evict_to_c)
        if rcl is not None:                                           # [hostfix] fewer reads: from the ONE read
            for kind, n in zip(("touch", "acquire", "evict_po", "release:evict"), rcl):
                self._count(kind, n)
        else:
            self._count("touch", int(rc["touch"]))
            self._count("acquire", int(rc["acquire"]))
            self._count("evict_po", int(rc["evict_po"]))
            self._count("release:evict", int(rc["evict_rel"]))
        t_hp = self._hp("h_hs_merges", t_hp)                          # demand merges, refusals, recency journal
        if self.lean:                                                 # [hostfix] --lean-time: no 4.3 / 4.6 looks;
            self._lean_queue(steps, vrows)                            # the release queue exactly as below
        else:
            self._looks(steps, V, ev, part, new_trs, vrows, vlook, vcols=vcols)
        t_hp = self._hp("h_hs_looks", t_hp)                           # the 4.3 / 4.6 looks and the release queue
        for p in info:
            if p["st"]["ddec"] is not None:
                masked, refused, pairs = p["dc"]
                self.metrics.layer_plan(Plan(page_table=None, num_pages=None, last_len=None, seq_idx=None, blocks=[],
                                             masked=int(masked), draft_pairs=int(pairs), refused=int(refused)))
        self._changed()
        if self.issue == "desc":                                      # LAST: queued for end_tick's issue phase
            self._queue_desc(takes, by_layer)
        self._hp("h_hs_plans", t_hp)

    # ---------------------------------------------------------------------------------------------- [b2-desc] the issue
    def _issue_desc(self, takes: Dict[str, torch.Tensor], by_layer: Dict[int, List[Transfer]]) -> None:
        """[b2-desc] The tick's draft takes as ONE descriptor job (E's tp.issue_desc; docs/R2_B2_DESC_PLAN.md). The
        Transfer objects of the object path stay (the arrival rule, the maps, the metrics, Policy E's forces read
        them); only the transport issue changes: no per-layer tp.issue, no per-transfer glue in the transport. The
        desc is the take list itself (l, b, h, block, slot, gen of every draft take, the same values the Transfers
        carry), built on the table's device and copied to the host ONCE, contiguous (lesson L28). T3: per layer the
        newest reader id among the takes' replaced last readers (rd_old; -1 none); T4: the newest write-back mark of
        the takes' (b, block) (-1 none). Every Transfer of the job gets the job's completion object (tp.job_ev), as
        the C++ worker's issue() gives it. Checked mode: the desc equals the Transfers' (l, b, h, block, slot, gen)
        multiset (one host loop; the unchecked timed path skips it). [f3ld] Called by _tick_issue (end_tick's issue
        phase), not at the end of _host_side as in 47db0c0 (_queue_desc)."""
        trs = [tr for l in sorted(by_layer) for tr in by_layer[l]]
        if not trs:
            return
        dm = ~takes["isv"]
        X = torch.stack([takes[n][dm].to(torch.int64) for n in ("l", "b", "h", "blk", "slot", "gen", "rd_old")]).cpu()
        desc = X[:6].contiguous()
        if int(desc.shape[1]) != len(trs):
            raise OwnershipViolation("B2 desc: %d draft takes in the desc, %d draft Transfers" % (desc.shape[1], len(trs)))
        if self.checked:
            want = sorted((tr.layer, tr.b, tr.h, tr.block, tr.slot, tr.gen) for tr in trs)
            if sorted(map(tuple, desc.t().tolist())) != want:
                raise OwnershipViolation("B2 desc: the descriptor differs from the draft Transfers")
        rd = X[6]
        if self.b2_mut.get("drop_rd_last"):                           # as the object path's reader_ev = None
            rd = torch.full_like(rd, -1)
        rm = torch.full((self.L,), -1, dtype=torch.int64).scatter_reduce_(0, desc[0], rd, reduce="amax",
                                                                          include_self=True)
        hm = int(self._hm[desc[1], desc[3]].max())                    # (no_t4 drops a CHECK, not the T4 wait: as hist_ev)
        seq = self.tp.issue_desc(desc, rm, hm)
        ev = self.tp.job_ev(seq)
        for tr in trs:
            tr.ev = ev
            tr.payload = None
        self.desc_jobs += 1
        self.desc_takes += len(trs)

    def _queue_desc(self, takes: Dict[str, torch.Tensor], by_layer: Dict[int, List[Transfer]]) -> None:
        """[f3ld] The end of the host consumer: the tick's draft takes wait for end_tick's issue phase (_tick_issue).
        A job still queued from an earlier tick is a defect (end_tick submits each tick's job)."""
        if self._desc_next is not None:
            raise OwnershipViolation("B2 desc: the queued draft takes of an earlier tick were never issued")
        self._desc_next = (takes, by_layer)

    def _tick_issue(self) -> None:
        """[f3ld] end_tick's issue phase: ResidencySC.end_tick calls it before Residency.end_tick, inside the h_et_issue
        clock, so the descriptor job is issued where the object path issues its per-layer transfers (nothing runs
        between apply_tick and this point: end_tick's finally only clears _layer_ev; a void tick raises in apply_tick,
        so it queues nothing). Nothing queued (the object path, a tick without takes): nothing to do."""
        job, self._desc_next = self._desc_next, None
        if job is not None:
            self._issue_desc(*job)

    def _lean_queue(self, steps, vrows) -> None:
        """[hostfix] --lean-time: the release queue of _looks without the looks. The keys are the verify rows' keys in
        first-occurrence order (dict.fromkeys: the same list as _looks' dedup, linear instead of quadratic), queued at
        every layer with verify rows, in the layer order of steps: the same _queue calls in the same order."""
        if self.mut.release_only_at_decision:
            return
        keys = list(dict.fromkeys(r.key for r in vrows))
        for st in steps:
            if st["vdec"] is not None:
                for key in keys:
                    self._queue(key, st["l"])

    def _looks(self, steps, V, ev, part, new_trs, vrows, vlook, vcols=None) -> None:
        """The looks (4.3 / 4.6), as C's sparse consumer (r2_residency_sparse.py _apply_verify_sparse), and the
        release queue of the verify keys at every layer with verify rows ([hostfix]: moved here unchanged from
        _host_side, so --lean-time can skip the looks and keep the queue: _lean_queue). vcols ([hostfix]
        --b2-fewer-reads): the verify record's columns and ev["src"] as host lists from _host_side's ONE read."""
        now = self.clock()
        trs_of = {q: tr for (q, tr) in new_trs}
        firsts: Dict[Tuple[int, Any], List[Tuple[int, int, str, Any]]] = {}
        if V is not None:
            if vcols is not None:                                     # [hostfix] the same values, read already
                vfl, vl, vj, vh, vblk = vcols["flags"], vcols["l"], vcols["j"], vcols["h"], vcols["blk"]
            else:
                vfl = V["flags"].tolist()
                vl, vj, vh, vblk = V["l"].tolist(), V["j"].tolist(), V["h"].tolist(), V["blk"].tolist()
            src_q = {}
            if ev["o"].numel():
                for q, (pq, sq) in enumerate(zip(part, ev["src"].tolist() if vcols is None else vcols["src"])):
                    if pq == 0:
                        src_q[sq] = q
            for i, fl in enumerate(vfl):
                if not fl & O.FIRST:
                    continue
                r = vrows[vj[i]]
                if fl & O.INFL:
                    firsts.setdefault((vl[i], r.key), []).append((vh[i], vblk[i], "in_flight", None))
                elif fl & (O.GRANT | O.EVICT):
                    firsts.setdefault((vl[i], r.key), []).append((vh[i], vblk[i], "new", trs_of.get(src_q.get(i))))
        for st in steps:
            l = st["l"]
            if st["vdec"] is None:
                continue
            keys = []
            for r in vrows:
                if r.key not in keys:
                    keys.append(r.key)
            ready: Dict[Any, List[int]] = {}
            for jj, r in enumerate(vrows):
                acc = ready.setdefault(r.key, [0, 0, 0, 0])
                for qq in range(4):
                    acc[qq] += vlook[l][jj][qq]
            for key in keys:
                call = self.metrics.open_call(key, l, now)
                nr, nl, npo, nv = ready[key]
                call["ready"] += nr
                self.metrics.k46["lru"] += nl
                self.metrics.k46["po"] += npo
                self.metrics.k46["vres"] += nv
                self.metrics.k46["not_resident"] += nr - nl - npo - nv
                for cls in ("in_flight", "new"):
                    for (h, t, c, tr) in firsts.get((l, key), []):
                        if c != cls:
                            continue
                        call[cls] += 1
                        if cls == "in_flight":
                            self.metrics.k46["inflight_waited"] += 1
                        else:
                            rec = self.metrics.lifetimes.get((l, tr.b, tr.h, tr.slot, tr.gen)) if tr else None
                            self.metrics.k46["fetched_refetch" if rec is not None and rec["refetch"]
                                             else "fetched_first"] += 1
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

    # ---------------------------------------------------------------------------------------------- record mutations
    def _rec_mutate(self, layers: List[Dict], rowsel: torch.Tensor, vread: torch.Tensor, rows, row_b: torch.Tensor,
                    row_plane: torch.Tensor) -> None:
        """NEGATIVE CONTROLS (tests only, b2_mut rec_mut): alter the first record of one kind of the tick (once per run),
        after the read and before the checks: the device's decision is wrong in that one way."""
        kind = self.b2_mut.get("rec_mut")
        if not kind or self.rec_mut_done is not None or kind in ("ignore_overflow", "reverse_stamps",
                                                                 "evict_po_unpinned", "gen_plus_two"):
            return                                                    # (the op-level kinds: _note_op_mut)
        G = self.own.geo
        code = self.own.c["code"].view(G.S, G.NPOS)
        for x in layers:
            l = x["l"]
            kx = x.get("k", rows.k)                                   # the layer's own k (its selection width)
            D, V = x.get("drec"), x.get("vrec")
            if kind == "take_nonfree" and D is not None:
                g = (torch.bitwise_and(D[2], O.GRANT) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(D[0, i]) // (G.H * kx), (int(D[0, i]) // kx) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    busy = (code[s] != C_FREE).nonzero().view(-1)
                    if busy.numel():
                        D[4, i] = int(busy[0])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "take_swap" and D is not None:
                g = (torch.bitwise_and(D[2], O.GRANT) != 0).nonzero().view(-1).tolist()
                for a in range(len(g)):
                    for c in range(a + 1, len(g)):
                        ia, ic = g[a], g[c]
                        if int(D[0, ia]) // kx == int(D[0, ic]) // kx and int(D[4, ia]) != int(D[4, ic]):
                            D[4, ia], D[4, ic] = D[4, ic].clone(), D[4, ia].clone()
                            self.rec_mut_done = (self.tick_no, kind)
                            return
            if kind == "k5_over" and D is not None:
                g = (torch.bitwise_and(D[2], O.REFUSED) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(D[0, i]) // (G.H * kx), (int(D[0, i]) // kx) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    fr = (code[s] == C_FREE).nonzero().view(-1)
                    if fr.numel():
                        D[2, i] = O.GRANT
                        D[4, i] = int(fr[0])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "pe_not_first" and V is not None:
                g = (torch.bitwise_and(V[2], O.EVICT) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(V[0, i]) // (G.H * kx), (int(V[0, i]) // kx) % G.H
                    s = (l * G.B + int(rows.vb[j])) * G.H + h
                    lru = (code[s] == C_LRU).nonzero().view(-1).tolist()
                    other = [x2 for x2 in lru if x2 != int(V[7, i])]
                    if other:
                        V[7, i] = other[-1]
                        V[8, i] = int(self.own.c["block"].view(G.S, G.NPOS)[s, other[-1]])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            # the DROPS take the first record whose effect is not repeated by another one (else nothing changes):
            dem = self.own.flat("demand")
            if kind == "drop_refusal" and D is not None:
                g = (torch.bitwise_and(D[2], O.REFUSED) != 0).nonzero().view(-1).tolist()
                if g:
                    x["drec"] = _without(D, g[0])
                    self.rec_mut_done = (self.tick_no, kind)
                    return
            if kind == "drop_v_merge" and V is not None:              # a merge whose transfer lacks the verify bit
                for i in (torch.bitwise_and(V[2], O.INFL) != 0).nonzero().view(-1).tolist():
                    j, h = int(V[0, i]) // (G.H * kx), (int(V[0, i]) // kx) % G.H
                    s = (l * G.B + int(rows.vb[j])) * G.H + h
                    if int(dem[s * G.NPOS + int(V[4, i])]) & O.DEM_V == 0:
                        x["vrec"] = _without(V, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_d_merge" and D is not None:              # a merge whose transfer lacks the draft bit
                # C's predicate (r2_residency_sparse.py:491-495): the transfer in flight lacks the draft demand, OR it
                # does not exist yet: the layer's OWN urgent verify take of this tick (demand {verify}), created at
                # apply; the tick-start slot_of cannot name it, the layer's verify GRANT / EVICT records do
                so = O.slot_of(self.own)
                vtake = set()
                if V is not None:
                    for iv in (torch.bitwise_and(V[2], O.GRANT | O.EVICT) != 0).nonzero().view(-1).tolist():
                        jv, hv = int(V[0, iv]) // (G.H * kx), (int(V[0, iv]) // kx) % G.H
                        vtake.add(((l * G.B + int(rows.vb[jv])) * G.H + hv, int(V[1, iv])))
                for i in (torch.bitwise_and(D[2], O.INFL) != 0).nonzero().view(-1).tolist():
                    j, h = int(D[0, i]) // (G.H * kx), (int(D[0, i]) // kx) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    sl = int(so[s, int(D[1, i])])
                    if (sl < 0 and (s, int(D[1, i])) in vtake) or \
                            (sl >= 0 and int(dem[s * G.NPOS + sl]) & O.DEM_D == 0):
                        x["drec"] = _without(D, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_ready" and D is not None:                # a draft read that no verify row repeats
                vbits = O.unpack_words(vread[l], G.NPOS)
                for i in (torch.bitwise_and(D[2], O.READY) != 0).nonzero().view(-1).tolist():
                    j, h = int(D[0, i]) // (G.H * kx), (int(D[0, i]) // kx) % G.H
                    if not bool(vbits[int(rows.db[j]), h, int(D[3, i])]):
                        x["drec"] = _without(D, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_rowsel":                                 # a NEW pin bit that no other row sets
                pw = self.own.pin_w
                R = rowsel.shape[1]
                for r in range(R):
                    p, b = int(row_plane[r]), int(row_b[r])
                    others = torch.zeros_like(rowsel[l, r])
                    for r2 in range(R):
                        if r2 != r and int(row_plane[r2]) == p and int(row_b[r2]) == b:
                            others |= rowsel[l, r2]
                    new = rowsel[l, r] & ~pw[p, l, b] & ~others
                    nz = new.nonzero()
                    if nz.numel():
                        h, w = nz[0].tolist()
                        word = int(new[h, w])
                        rowsel[l, r, h, w] = int(rowsel[l, r, h, w]) & ~(word & -word)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_vread" and bool(vread[l].any()):         # a verify read that no draft row repeats
                dset = set()
                if D is not None:
                    for i in (torch.bitwise_and(D[2], O.READY) != 0).nonzero().view(-1).tolist():
                        j, h = int(D[0, i]) // (G.H * kx), (int(D[0, i]) // kx) % G.H
                        dset.add((int(rows.db[j]), h, int(D[3, i])))
                for (b, h, sl) in O.unpack_words(vread[l], G.NPOS).nonzero().tolist():
                    if (b, h, sl) not in dset:
                        w, bit = divmod(sl, 32)
                        vread[l, b, h, w] = int(vread[l, b, h, w]) & ~(1 << bit)
                        self.rec_mut_done = (self.tick_no, kind)
                        return


def _without(R: torch.Tensor, i: int) -> torch.Tensor:
    keep = torch.ones(R.shape[1], dtype=torch.bool, device=R.device)
    keep[i] = False
    return R[:, keep]


# the trash writer (F15's control, the B2 form): the draft bias fill also writes the trash slot
class TrashWriterB2(OwnResidency):
    def _layer_record(self, l, step, tk):
        super()._layer_record(l, step, tk)
        if self.trash_slot is not None and self.store_b is not None:
            s0 = self.trash_slot * self.bs
            self.store_b[l][:, s0:s0 + self.bs] = 0.0
