"""The PAPER METRICS of one dispatch (milestone 11, Track B): Dispatcher.paper_metrics() for the evaluation sections of
the paper draft (4.3, 4.5, 4.6, 4.8). Opt-in: Config.paper_metrics (default False) or Dispatcher.enable_paper_metrics()
before the first joint call. Off, nothing below runs, and paper_metrics() returns the existing counters and the IDEAL
sharing only. On, the bookkeeping is observation only: it reads the dispatcher's state and decides nothing (no
reference, slot, event, counter of Dispatcher.stats or report() entry changes). Its cost: dictionary updates per
verifier look, slot read, transfer issue and arrival, and ONE host clock reading (time.perf_counter_ns) per joint call
that looks at or executes a verifier and per host poll that observes a waited-for arrival.

DEFINITIONS (exact; DEFINITIONS below holds the same lines and paper_metrics() returns them):
  unit           one selected historical block = (request, layer, KV head, block id); bytes = blocks x block_bytes,
                 block_bytes = Geometry.unit()["slot"] (K + V (+ bias) of bs positions; the K6 convention)
  layer_call     one verifier cohort at one layer: from its FIRST LOOK (the first joint call in which it is eligible
                 at that layer, D3a passed, its exact selection known) to the call that reads its KV; counted when
                 that layer is published (exec_log); its blocks are the union over the cohort's rows of the selected
                 blocks below the cohort's history boundary (the blocks that its reads use, vread_log)
  ready          (4.3) resident at the first look (arrival observed: LRU, PO or VRES), not fetched by this cohort
  in_flight      (4.3) not resident at the first look and not fetched by this cohort: another requester's transfer
                 served it (in flight at the look, or issued after it: in_flight_missing_at_look)
  new            (4.3) fetched by this cohort: a successful verify issue of this cohort at this layer (vfetch ids)
  wait_ns        (4.3) per block: host clock at the first observed arrival of the block after the first look
                 (Dispatcher._poll) minus host clock at the first look; ready blocks 0; no arrival observed after
                 the look: the read call's clock (counted in unstamped)
  layer_wait_ns  (4.3) per layer_call: host clock at the read call minus host clock at the first look
  abandoned      (4.3) layer_calls looked at but never executed (cancelled or rejected cohorts), by first-look state
  served         (4.6) per block of a layer_call, the slot that served the verifier read: lru (the ordinary draft
                 LRU), po / vres (outside the LRU, kept by a reference of a live speculative block: RETAINED),
                 fetched_first / fetched_refetch (this cohort's own transfer; refetch when the same (request,
                 layer, head, block) was transferred before in the run), not_resident (only under a mutation)
  peak_retained  (4.6) the maximum over the K6 samples (one per joint call) of the PO + VRES slots, x block_bytes
  prefetch_use   (4.8) every transfer = one destination slot lifetime (from its take), classified by the reads of
                 that lifetime (draft rows / verifier rows) as draft_only / verify_only / both / neither when the
                 slot left the lifetime (released, re-taken, or its request ended); still_live = the lifetime holds
                 the slot at the report (INFLIGHT, LRU, PO or VRES, same generation and owner); never_arrived =
                 released before its arrival was observed; by issuer (the kind that issued it: draft = prefetch)
  repeated       (4.8) transfers minus distinct (request, layer, head, block) transferred
  sharing_IDEAL  (4.5) Dispatcher.sharing(): per joint call and (batch slot, head), SUM vs UNION of the rows'
                 selected blocks (draft and verifier rows together); the IDEAL from selections, not a measurement
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .executor import DRAFT, VERIFY
from .ownership import INFLIGHT, LRU, PO, RELEASING, VRES

RECORDS_CAP = 100000                     # per-layer_call records kept (records_dropped counts the rest)
CATS = ("ready", "in_flight", "new")    # 4.3
SERVED = ("lru", "po", "vres", "fetched_first", "fetched_refetch", "not_resident")   # 4.6
USES = ("draft_only", "verify_only", "both", "neither")                              # 4.8
LOOK_STATES = ("ready", "pending", "missing")                                          # the state at the first look
COUNTERS_43 = ("cohort_waits", "cohort_waits_d3a", "fetch_verify_issued", "fetch_verify_noslot")
COUNTERS_46 = ("arrivals_lru", "arrivals_vres", "arrivals_releasing", "arrivals_verify_joined_by_draft",
               "verify_refetch_attempts", "draft_acquisitions_po", "draft_acquisitions_vres", "evict_protected_only",
               "evict_released", "verifier_evictions")
COUNTERS_48 = ("fetch_draft_issued", "fetch_verify_issued", "fetch_draft_refused", "fetch_verify_noslot",
               "demand_merged_draft", "demand_merged_verify", "arrivals_admitted", "arrivals_dropped_unowned",
               "late_draft_arrivals")
DEFINITIONS = dict(
    unit="one selected historical block = (request, layer, KV head, block id); bytes = blocks x block_bytes "
         "(Geometry.unit()['slot']: K + V (+ bias) of bs positions, the K6 convention)",
    layer_call="one verifier cohort at one layer, from its first look (first joint call in which it is eligible there "
               "with its exact selection) to the call that reads its KV; counted when the layer is published",
    ready="resident at the first look and not fetched by this cohort",
    in_flight="not resident at the first look and not fetched by this cohort (another requester's transfer served it)",
    new="fetched by this cohort: a successful verify issue of this cohort at this layer (N3 P2 vfetch ids)",
    wait_ns="per block: host clock at its first observed arrival after the first look minus host clock at the first "
            "look; ready 0; no arrival observed: the read call's clock (unstamped)",
    layer_wait_ns="per layer_call: host clock at the read call minus host clock at the first look",
    abandoned="layer_calls looked at but never executed (cancelled or rejected cohorts), by first-look state",
    served="per block of a layer_call, the slot that served the verifier read: lru, po / vres (retained outside the "
           "LRU by a reference), fetched_first / fetched_refetch (this cohort's own transfer; refetch = the same "
           "(request, layer, head, block) was transferred before)",
    peak_retained="max over the K6 samples (one per joint call) of the PO + VRES slots, x block_bytes",
    prefetch_use="every transfer (one destination slot lifetime) by the reads of that lifetime: draft_only / "
                 "verify_only / both / neither at its end; still_live if it holds its slot at the report; by issuer",
    repeated="transfers minus distinct (request, layer, head, block) transferred",
    sharing_IDEAL="per joint call and (batch slot, head): SUM vs UNION of the rows' selected blocks (draft and "
                  "verifier rows together); IDEAL from actual selections, not a measurement",
)

_SEQ, _UID, _GEN, _KIND, _REPEAT, _ARRIVED, _RD, _RV = range(8)    # one lifetime record (a list)


def use_of(rd: int, rv: int) -> str:
    """4.8: the use class of one lifetime from its draft and verifier read counts."""
    if rd and rv:
        return "both"
    if rd:
        return "draft_only"
    if rv:
        return "verify_only"
    return "neither"


def existing_counters(d: Any) -> Dict[str, Dict[str, int]]:
    """The existing Dispatcher.stats counters that the sections use, read with .get (stats is a defaultdict: a read
    never creates a key)."""
    s = d.stats
    return {sec: {k: int(s.get(k, 0)) for k in keys}
            for sec, keys in (("4.3", COUNTERS_43), ("4.6", COUNTERS_46), ("4.8", COUNTERS_48))}


class PaperMetrics:
    """The bookkeeping behind Dispatcher.paper_metrics() (module docstring). The dispatcher calls: look (every
    verifier need evaluation), exec_reads (the call that reads a cohort's KV), executed (its publication),
    drop_cohort (decision or cancellation), issued (every transfer), arrived (every observed arrival), read (every
    slot read of a row), retained_sample (every K6 sample). It holds ints, strings, tuples and cohort keys only (no
    event, no tensor)."""

    def __init__(self, d: Any):
        self.pools = d.pools
        self.L, self.H = d.L, d.H
        self.block_bytes = int(d.acct.u["slot"])
        self._stamp: Tuple[Any, int] = (None, 0)
        self.looks: Dict[Tuple, Dict[str, Any]] = {}            # (cohort key, layer) -> the look record
        self.waiters: Dict[Tuple, set] = {}                      # (layer, b, h, block) -> {(cohort key, layer)}
        self.stash: Dict[Tuple, Dict[str, Any]] = {}             # (cohort key, layer) -> the record of its read call
        self.records: List[Dict[str, Any]] = []
        self.records_dropped = 0
        self.tot = {c: dict(blocks=0, wait_ns=0, wait_max_ns=0, unstamped=0) for c in CATS}
        self.layer_calls = 0
        self.layer_wait_ns = 0
        self.layer_wait_max_ns = 0
        self.looks_total = 0
        self.in_flight_missing_at_look = 0
        self.exec_without_look = 0
        self.unlooked_blocks = 0
        self.publish_without_reads = 0
        self.abandoned = dict(layer_calls=0, looks=0, ready=0, pending=0, missing=0, own_issued=0)
        self.served = {k: 0 for k in SERVED}
        self.peak = dict(po_vres=0, po=0, vres=0, samples=0, last_po_vres=0)
        self.life: Dict[Tuple[int, int, int, int], List[Any]] = {}   # (layer, b, h, slot) -> the open lifetime
        self.transferred: Dict[Tuple, int] = {}                  # (uid, layer, h, block) -> transfers so far
        self.transfers = {DRAFT: 0, VERIFY: 0}
        self.fin = {k: {u: 0 for u in USES} for k in (DRAFT, VERIFY)}
        self.fin_never_arrived = {DRAFT: 0, VERIFY: 0}
        self.repeats = {DRAFT: 0, VERIFY: 0}                     # transfers of an already transferred block

    # ---------------------------------------------------------------- the clock (one reading per joint call)
    def _now(self, cid: int) -> int:
        if self._stamp[0] != cid:
            self._stamp = (cid, time.perf_counter_ns())
        return self._stamp[1]

    # ---------------------------------------------------------------- 4.3 / 4.6: verifier looks and reads
    def look(self, r: Any, c: Any, l: int, need: List[Tuple[List[int], List[int], List[int]]], cid: int) -> None:
        """A verifier need evaluation of cohort c at layer l (Dispatcher._need: per head (ready, in flight,
        missing)). The first one opens the look (host clock); a block keeps the state of the first look that saw it;
        a block not resident there waits for its arrival (waiters)."""
        k = (c.key, l)
        t = self._now(cid)
        lk = self.looks.get(k)
        if lk is None:
            lk = self.looks[k] = dict(b=r.b, uid=r.uid, epoch=c.epoch, m=c.m, rows=len(c.positions), first=cid, t0=t,
                                      looks=0, blocks={})
        lk["looks"] += 1
        self.looks_total += 1
        blocks = lk["blocks"]
        for h, parts in enumerate(need):
            for state, js in zip(LOOK_STATES, parts):
                for j in js:
                    if (h, j) in blocks:
                        continue
                    blocks[(h, j)] = [state, t, t if state == "ready" else None]
                    if state != "ready":
                        self.waiters.setdefault((l, r.b, h, j), set()).add(k)

    def exec_reads(self, d: Any, l: int, exec_cohorts: List[Tuple[Any, Any]], cid: int) -> None:
        """The joint call cid reads the KV of every cohort in exec_cohorts at layer l: its record (4.3 categories and
        waits, 4.6 served) is computed now, from the state at the read, and kept until the layer is published
        (executed); a later read call of the same (cohort, layer) replaces it."""
        t = self._now(cid)
        for r, c in exec_cohorts:
            k = (c.key, l)
            lk = self.looks.get(k)
            if lk is None:                                      # never expected: an executing cohort was looked at
                self.exec_without_look += 1
                lk = self.looks[k] = dict(b=r.b, uid=r.uid, epoch=c.epoch, m=c.m, rows=len(c.positions), first=cid,
                                          t0=t, looks=1, blocks={})
                self.looks_total += 1
            own = d.vfetch.get(k, ())
            hv = d._vhist(r, c, l)
            sel = c.ws[l]["sel"]
            n = {x: 0 for x in CATS}
            w = {x: 0 for x in CATS}
            wm = {x: 0 for x in CATS}
            uns = {x: 0 for x in CATS}
            served = {x: 0 for x in SERVED}
            miss0 = 0
            for h in range(self.H):
                st = self.pools[l].stream(r.b, h)
                for j in sorted({j for row in sel for j in row[h] if j < hv}):
                    blk = lk["blocks"].get((h, j))
                    if blk is None:                             # never expected: the look saw the same blocks
                        self.unlooked_blocks += 1
                        blk = ["ready" if st.resident(j) is not None else "missing", t, t]
                    if (r.b, h, j) in own:
                        cat = "new"
                    elif blk[0] == "ready":
                        cat = "ready"
                    else:
                        cat = "in_flight"
                        miss0 += blk[0] == "missing"
                    wait = 0
                    if cat != "ready":
                        tr = blk[2]
                        if tr is None or blk[0] == "ready":
                            tr = t
                            uns[cat] += 1
                        wait = max(0, tr - blk[1])
                    n[cat] += 1
                    w[cat] += wait
                    wm[cat] = max(wm[cat], wait)
                    idx = st.resident(j)
                    if cat == "new":
                        rec = None if idx is None else self.life.get((l, r.b, h, idx))
                        rep = rec is not None and rec[_GEN] == st.slots[idx].gen and rec[_REPEAT]
                        served["fetched_refetch" if rep else "fetched_first"] += 1
                    elif idx is None:
                        served["not_resident"] += 1
                    elif j in st.lru:
                        served["lru"] += 1
                    elif j in st.po:
                        served["po"] += 1
                    else:
                        served["vres"] += 1
            self.stash[k] = dict(call_first=lk["first"], call_exec=cid, uid=r.uid, b=r.b, epoch=c.epoch, m=c.m, layer=l,
                                 rows=len(c.positions), looks=lk["looks"], blocks=n, in_flight_missing_at_look=miss0,
                                 wait_ns=w, wait_max_ns=wm, unstamped=uns, layer_wait_ns=max(0, t - lk["t0"]),
                                 served=served)

    def executed(self, key: Any, l: int) -> None:
        """The layer-l outputs of cohort `key` were published (Dispatcher._call_publish, the cohort advanced): its
        record of the read call is committed."""
        k = (key, l)
        rec = self.stash.pop(k, None)
        lk = self.looks.pop(k, None)
        if lk is not None:
            self._unwait(k, l, lk)
        if rec is None:
            self.publish_without_reads += 1
            return
        for cat in CATS:
            tt = self.tot[cat]
            tt["blocks"] += rec["blocks"][cat]
            tt["wait_ns"] += rec["wait_ns"][cat]
            tt["wait_max_ns"] = max(tt["wait_max_ns"], rec["wait_max_ns"][cat])
            tt["unstamped"] += rec["unstamped"][cat]
        for x in SERVED:
            self.served[x] += rec["served"][x]
        self.layer_calls += 1
        self.layer_wait_ns += rec["layer_wait_ns"]
        self.layer_wait_max_ns = max(self.layer_wait_max_ns, rec["layer_wait_ns"])
        self.in_flight_missing_at_look += rec["in_flight_missing_at_look"]
        if len(self.records) < RECORDS_CAP:
            self.records.append(rec)
        else:
            self.records_dropped += 1

    def drop_cohort(self, d: Any, key: Any, why: str) -> None:
        """The cohort was decided or cancelled (Dispatcher._vfetch_drop_cohort, BEFORE its vfetch records are dropped):
        every layer it looked at and never published is abandoned."""
        for l in range(self.L):
            k = (key, l)
            self.stash.pop(k, None)
            lk = self.looks.pop(k, None)
            if lk is None:
                continue
            self._unwait(k, l, lk)
            ab = self.abandoned
            ab["layer_calls"] += 1
            ab["looks"] += lk["looks"]
            for blk in lk["blocks"].values():
                ab[blk[0]] += 1
            ab["own_issued"] += len(d.vfetch.get(k, ()))

    def _unwait(self, k: Tuple, l: int, lk: Dict[str, Any]) -> None:
        for (h, j), blk in lk["blocks"].items():
            if blk[2] is None:
                s = self.waiters.get((l, lk["b"], h, j))
                if s is not None:
                    s.discard(k)
                    if not s:
                        del self.waiters[(l, lk["b"], h, j)]

    # ---------------------------------------------------------------- transfers, arrivals, reads (4.3 stamps, 4.8)
    def issued(self, tr: Any) -> None:
        """Transfer tr took its destination slot: a new lifetime (the slot's previous one ended: it was FREE)."""
        key = (tr.layer, tr.b, tr.h, tr.slot)
        old = self.life.pop(key, None)
        if old is not None:
            self._final(old, self.fin, self.fin_never_arrived)
        tk = (tr.uid, tr.layer, tr.h, tr.block)
        n = self.transferred.get(tk, 0)
        self.transferred[tk] = n + 1
        self.transfers[tr.kind] += 1
        self.repeats[tr.kind] += n > 0
        self.life[key] = [tr.seq, tr.uid, tr.gen, tr.kind, n > 0, False, 0, 0]

    def arrived(self, tr: Any, res: Dict[str, Any]) -> None:
        """The host observed tr's arrival (Stream.admit result res). An admitted block (LRU or VRES) is READY: every
        look waiting for it gets this host clock as its readiness."""
        rec = self.life.get((tr.layer, tr.b, tr.h, tr.slot))
        if rec is not None and rec[_GEN] == tr.gen:
            rec[_ARRIVED] = True
        if res.get("dropped") or res.get("where") == RELEASING:
            return
        ks = self.waiters.pop((tr.layer, tr.b, tr.h, tr.block), None)
        if not ks:
            return
        t = time.perf_counter_ns()
        for k in ks:
            lk = self.looks.get(k)
            blk = None if lk is None else lk["blocks"].get((tr.h, tr.block))
            if blk is not None and blk[2] is None:
                blk[2] = t

    def read(self, l: int, b: int, h: int, idx: int, gen: int, verify: bool) -> None:
        """One row's read of slot idx (generation gen) of stream (l, b, h): a verifier row (verify) or a draft row."""
        rec = self.life.get((l, b, h, idx))
        if rec is not None and rec[_GEN] == gen:
            rec[_RV if verify else _RD] += 1

    def retained_sample(self, po: int, vres: int) -> None:
        """One K6 sample: the PO and VRES slots (outside the LRU, kept by a reference)."""
        p = self.peak
        p["samples"] += 1
        p["last_po_vres"] = po + vres
        p["po_vres"] = max(p["po_vres"], po + vres)
        p["po"] = max(p["po"], po)
        p["vres"] = max(p["vres"], vres)

    @staticmethod
    def _final(rec: List[Any], fin: Dict, never: Dict) -> None:
        fin[rec[_KIND]][use_of(rec[_RD], rec[_RV])] += 1
        never[rec[_KIND]] += not rec[_ARRIVED]

    def _live(self, key: Tuple[int, int, int, int], rec: List[Any]) -> bool:
        l, b, h, idx = key
        st = self.pools[l].stream(b, h)
        s = st.slots[idx]
        return s.gen == rec[_GEN] and s.state in (INFLIGHT, LRU, PO, VRES) and st.owner == rec[_UID]

    # ---------------------------------------------------------------- the report (READ-ONLY)
    def report(self, d: Any) -> Dict[str, Any]:
        bb = self.block_bytes

        def cb(n: int) -> Dict[str, int]:
            return dict(blocks=int(n), bytes=int(n) * bb)
        fin = {k: dict(v) for k, v in self.fin.items()}
        never = dict(self.fin_never_arrived)
        live = {k: {u: 0 for u in USES} for k in (DRAFT, VERIFY)}
        for key, rec in self.life.items():
            if self._live(key, rec):
                live[rec[_KIND]][use_of(rec[_RD], rec[_RV])] += 1
            else:
                self._final(rec, fin, never)
        n_tr = sum(self.transfers.values())
        cnt = existing_counters(d)
        ab = dict(self.abandoned)
        ab["bytes"] = (ab["ready"] + ab["pending"] + ab["missing"]) * bb
        ret = self.served["po"] + self.served["vres"]
        return dict(
            enabled=True, block_bytes=bb, definitions=dict(DEFINITIONS),
            config=dict(K=d.K, LAMBDA=d.LAMBDA, B=d.cfg.B, C=d.cfg.C, X=d.cfg.X, R_ver=d.cfg.R_ver,
                        capacity=d.cfg.capacity, physical=d.cfg.physical),
            verify_kv_split=dict(
                layer_calls=self.layer_calls, looks=self.looks_total,
                per_category={c: dict(cb(self.tot[c]["blocks"]), wait_ns=self.tot[c]["wait_ns"],
                                      wait_max_ns=self.tot[c]["wait_max_ns"], unstamped=self.tot[c]["unstamped"])
                              for c in CATS},
                in_flight_missing_at_look=self.in_flight_missing_at_look,
                layer_wait_ns=self.layer_wait_ns, layer_wait_max_ns=self.layer_wait_max_ns, abandoned=ab,
                open_layer_calls=len(self.looks), exec_without_look=self.exec_without_look,
                unlooked_blocks=self.unlooked_blocks, publish_without_reads=self.publish_without_reads,
                records=list(self.records), records_dropped=self.records_dropped, counters=cnt["4.3"]),
            retained_kv=dict(
                served={x: cb(self.served[x]) for x in SERVED}, retained=cb(ret),
                refetch=cb(self.served["fetched_refetch"]),
                peak_retained=dict(slots=self.peak["po_vres"], bytes=self.peak["po_vres"] * bb,
                                   po_slots=self.peak["po"], vres_slots=self.peak["vres"],
                                   samples=self.peak["samples"], last_slots=self.peak["last_po_vres"]),
                counters=cnt["4.6"]),
            prefetch_use=dict(
                released={k: {u: cb(fin[k][u]) for u in USES} for k in (DRAFT, VERIFY)},
                still_live={k: {u: cb(live[k][u]) for u in USES} for k in (DRAFT, VERIFY)},
                never_arrived={k: cb(never[k]) for k in (DRAFT, VERIFY)},
                repeated_by_issuer={k: cb(self.repeats[k]) for k in (DRAFT, VERIFY)},
                transfers=cb(n_tr), transfers_by_issuer={k: cb(self.transfers[k]) for k in (DRAFT, VERIFY)},
                distinct=cb(len(self.transferred)), repeated=cb(n_tr - len(self.transferred)),
                counters=cnt["4.8"]),
            sharing_IDEAL=dict(d.sharing(), label="IDEAL: per joint call and (batch slot, head), SUM vs UNION of the "
                                                  "rows' selected blocks; from selections, not a measurement"),
        )


def disabled_report(d: Any) -> Dict[str, Any]:
    """paper_metrics() with the bookkeeping off: the existing counters and the IDEAL sharing only."""
    return dict(enabled=False, definitions=dict(DEFINITIONS), counters=existing_counters(d),
                sharing_IDEAL=dict(d.sharing(), label="IDEAL: per joint call and (batch slot, head), SUM vs UNION of "
                                                      "the rows' selected blocks; from selections, not a measurement"))
