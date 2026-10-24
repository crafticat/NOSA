"""[f3ld] CPU tests of --no-transfer-mirror (OwnResidency(no_transfer_mirror=True); r2_own2.py module docstring "NO
TRANSFER MIRROR"; analysis/evalfast/PLAN_admission_fill.md sections 3.4, 4 rank 1b and 5.1) and of the f3ld cells of
r2_tick_sg5_step.sh. The REFERENCE is the mirror path with the descriptor issue (issue="desc", test_r2_b2_desc.py BD1:
desc == object), so every arm runs on E's C++ worker (packer v3 of this tree, DESC). The world is
test_r2_b2_desc.py's: the tiny model on the Residency layout, S-C's CPU device stand-in, the events as ONE in-order
stream, a per-tick record after a flush of every job.

THE COMPLETION VIEW (_slow, a deterministic stand-in of a slow copy; the force precondition of TO0): job s is seen as
landed only when every job <= s is (E's jobs land in submission order), `delay(s)` transport polls after its issue,
or once it or a later job was forced (synchronize). The real jobs still land (the record flushes), so the store is
untouched; done(), landed_upto() and the forces see the view. Both arms poll once per tick and issue the same jobs, so
the view is the same in both.

  TO0  PRECONDITIONS (L34), over the TO1 draws: the arms issue descriptor jobs; the table arm holds transfers (landed
       before due) AND forces them (due, not landed); Policy E evicts an in-flight slot (in the SCRIPTED draw only, see
       THE CADENCE BOUND; the lag <= 2 draws evict none, asserted); a verify merge and a draft merge write the demand
       of a slot in flight; a request ends with a draft prefetch in flight (in the SCRIPTED END draw, see THE END IN
       FLIGHT); the mirror arm keeps Transfers in flight and the table arm none.

THE END IN FLIGHT (why the eval draw ended no request with a take in flight, and the SCRIPTED END draw). A request ends
after the tick whose decision makes it done (r2_tick.py:1400-1402 -> :1290-1293 -> r2_own2.py:649-671): its LRU / PO /
VRES slots are released and its takes in flight STAY in flight, dropped at their arrival (r2_own2_ops.py:451-468;
poll_arrivals drop_unowned, :352 and :380). It is done only at a verify decision (r2_spec_core.py:223-226), and that
tick has no draft row of the request (a draft row is planned only while want < max_new, :141-145, and a decision adds at
most the lead to len(out)). A transport take is admitted at issue_tick + lag, held before (r2_own2_ops.py:368-373). So a
take is in flight at the end only if it was issued in the last lag - 1 ticks before the end: at lag 2, only by a draft
row of the tick before the end. The eval draw's last draft rows select resident blocks, so it has none. This is
reachable, not a bound, so the SCRIPTED END draw (_script_end_sel) builds it: C = 5, X = 0, arrival_lag 8, no verify
miss, and the draft row at the request's LAST draft position (S_b + STEPS - 2: the final epoch always drafts it, as the
end needs want >= max_new) prefetches a block in no slot while a FREE slot exists (four FREE slots after block A: an
epoch rejected after its last draft row uses one). The final cohorts are verified within lag - 1 ticks of that row.

THE CADENCE BOUND (why the random draws never evict a slot in flight). Policy E's in-flight candidate is an UNPINNED,
not-landed draft prefetch of the stream (r2_own2_ops.py:706-707 policy_e_check cf; r2_residency_sc.py:518-525
device_verify cand_f). A draft row pins every block it selects with its cohort key (r2_residency_sb.py:160 protect =
valid; r2_own2.py:1528 pins_or). That key is released only behind the verify read of its cohort (r2_own2.py:1903-1913
_lean_queue; r2_residency.py:796-798) or by a cancel (r2_residency.py:806-834), and the drop runs at a later poll,
AFTER the arrivals (r2_own2.py:717-729). A cohort is verified at least one tick after its last draft row
(r2_spec_core.py:136-154 plan_tick, :178-182 the trigger; r2_tick.py:1175-1177 cohort_of). So a prefetch issued at
tick t is unpinned at the poll of t + 2 at the earliest, where the arrival rule (lag <= 2) admits it first
(r2_own2_ops.py:368-374). The one earlier release, a cancel at tick t, starts a new epoch with no cohort for K + 1
ticks, so the request has no verify row while the prefetch is in flight. Hence: with arrival_lag <= 2 (every random
draw and the GPU cells' --arrival-lag 2) no in-flight victim exists, by the cadence. The SCRIPTED draw (_script_setup) uses
arrival_lag 8 and a scripted selection that builds the state by construction: C = 3, X = 0 (K5 refuses the verifier
when the stream is full, ownership.py:419-427), the drafts prefetch three blocks, a rejection or a verify releases
their keys, and at the next verify tick of ONE stream (layer 0, request 3, head 0) the verify rows select a block that
is neither resident nor in flight plus every unpinned LRU block (so no LRU candidate exists) and spare the oldest
prefetch: Policy E supersedes the second-oldest prefetch, whose job is not forced yet. The draft row of that tick
selects the same blocks, so it merges onto the verify take of this tick (demand {verify}: the TO2i site,
r2_own2_ops.py:669-678 merge_check this_tick; r2_own2.py:1153-1162).
  TO1  ROUND LEVEL, randomized streams: per draw (random.Random(seed): the configuration, the data seed, teacher-forced
       or free-running, the event lateness 0 / 2, arrival_lag 1 / 2, checked / unchecked, --b2-fewer-reads,
       --wait-index, --res-lean / --res-dev, the completion view, the mirror arm's poll: the array poll or the old one)
       plus two fixed draws (the eval flag set; checked + free-running): the table arm == the mirror arm at every tick
       and every LAYER: rows, tokens, logits, pages, the projection, Table.diff empty, counters, readers, memory(), the
       arrival counts (held, forced), the STORE, and every layer's device decisions and plan (test_r2_resfix.py
       _layer_rec); at the end the stores, the committed tokens, the journal, the report's comparable parts. In the
       mirror arm, at every tick, the table carries the mirror's facts: the Transfers in flight are exactly the table's
       INFLIGHT slots, with their block, generation, issue tick and demand set (the table is the authority).
       TO1f: the fused v3 resolve (--b2-resolve fused --fused-kernel v3 --res-lean --res-dev, TRITON_INTERPRET=1 in a
       fresh process) on the eval flag set: table arm == mirror arm.
  TO2  NEGATIVE CONTROLS at the site under test (b2_mut, the table arm; L34: each records that it acted; L49: the plant
       is at the table-path line, not on a module global; (i) and (iv) act in the SCRIPTED draw, by construction: THE
       CADENCE BOUND above): (i) nm_drop_dem_d: one draft demand bit is not written ->
       the first difference is that tick's TABLE (the demand column) and a later tick's projection (the admission went
       another way); (ii) nm_due_early: one job due a tick early -> the first difference is at that tick; (iii)
       nm_skip_force: a due job that has not landed is not forced -> O.poll_arrivals refuses (CapacityRefusal "force_seq
       first") at that tick; (iv) nm_victim_unforced (with sc_mut no_inflight_wait, so the device wait does not force
       the superseded prefetch first): the in-flight victim's job is not forced -> the first difference is that tick's
       VIEW (the forced watermark). The plan's form "the store differs at that slot" needs a copy that lands after the
       next write: the CPU record flushes every job, so the store cannot show it here.
  TO2b WITH THE FLAG OFF the mirror's demand is write-only: the mirror arm with Mutations(drop_demand_merge) (no
       tr.demand.add) == the clean mirror arm at every tick (precondition: the merges wrote demand bits).
  TO3  STATIC: the table path constructs no Transfer (_host_side_nm, _issue_desc_nm), sorts nothing and reads no host
       map (_poll_arrivals_nm); CudaDeviceOps' waits take the jobs before the transfer loops; ResidencySparse._drop_one's
       read of tr.demand (r2_residency_sparse.py:672) is only on ResidencySparse.apply_tick, which OwnResidency
       replaces; OwnResidency.end_request / begin_request / sync_mirror read no mirror.
  TO3b THE HOST OBJECTS PER BLOCK FALL (counted on a fake): a counting Transfer in place of r2_own2's: the mirror arm
       constructs one per take (draft + urgent), the table arm none; the table arm's in-flight map stays empty.
  TO4  REFUSALS AND WIRING: OwnResidency(no_transfer_mirror=True) needs issue="desc", a fixed arrival_lag and lean,
       refuses inflight_array and the mirror-only controls; the nm_* controls need the flag; r2_tick_sg5
       --no-transfer-mirror parses with --b2-issue desc --worker cpp --arrival-lag >= 1 --lean-time and refuses
       otherwise; make_engine passes it; the T21 static lists (one_point's gate helpers, main's gate calls) hold.
  TO5  THE CELLS (static): evalfast2f3d / evalfast2c3d / evalfast2f3t / evalfast2c3t, NOSI_EVAL = this tree, packer v3
       of --nosi (GF3, [f3ldx] DESC 2), the REGISTERED F3LD block unstamped with every PREDICTION "TBD by lead", the
       TBD bands refused outside DRYRUN, [f3ldx] no F3LD_BLOCKED; TO5d: the DRYRUN argv of every new cell, and the
       harness now PARSES it on --layout pool (the blocker is cleared; the TBD bands aside).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3ld.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


BD = _shared("test_r2_b2_desc_shared_f3ld", "test_r2_b2_desc.py")      # the desc world (_run, _engine, _first_diff)
RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # _layer_rec, _same (shared name with f3lia)
HF = RF.HF                                                            # the step helpers (_dry, _clean_head, STEP)
TT, TRM, OWN2, RSC, GF = BD.TT, BD.TRM, BD.OWN2, BD.RSC, BD.GF
RES = TT.RES
p3, ref, v3 = BD.p3, BD.ref, BD.v3                                    # the fixtures, by name
C_INFLIGHT = RSC.C_INFLIGHT
F3LD_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3ldx"                        # [f3ldx] (was nosi-r2-f3ld)
CELLS = ("evalfast2f3d", "evalfast2c3d", "evalfast2f3t", "evalfast2c3t")
FIELDS = BD.FIELDS + ("layers",)


# =====================================================================================================================
# the world
# =====================================================================================================================
def _state_tbl(svc):
    """test_r2_residency._state without its last two parts (the service's pending map of Transfers and the
    transport's pending list): the vendored projection of every stream, which both arms build from the table."""
    return BD._state(svc)[:-2]


def _same_field(f, x, y):
    if f == "layers":
        return RF._same(x, y, "layers") is None
    if f == "view":
        return x == y
    return BD._same_field(f, x, y)


def _first_diff(a, b, fields=FIELDS):
    n = min(len(a.ticks), len(b.ticks))
    for t in range(n):
        for f in fields:
            if not _same_field(f, a.ticks[t][f], b.ticks[t][f]):
                return (t, f)
    if len(a.ticks) != len(b.ticks) or (a.err is None) != (b.err is None):
        return (n, "raise: %s / %s" % (a.err, b.err))
    return None


def _slow(tp, delay_of):
    """THE COMPLETION VIEW (module docstring) over E's transport. Returns its state (polls, issued: seq -> polls at
    issue, forced: the forced watermark)."""
    st = dict(polls=0, issued={}, forced=0)
    o_issue, o_poll, o_land, o_ev = tp.issue_desc, tp.poll, tp.landed_upto, tp.job_ev

    def vis_upto():
        v = 0
        for s in sorted(st["issued"]):
            if s <= st["forced"] or st["polls"] - st["issued"][s] >= delay_of(s):
                v = s
            else:
                break
        return v

    def issue_desc(desc, reader_marks=None, hist_mark=-1):
        seq = o_issue(desc, reader_marks, hist_mark)
        if seq:
            st["issued"][int(seq)] = st["polls"]
        return seq

    def poll():
        st["polls"] += 1
        return o_poll()

    def landed_upto():
        return min(int(o_land()), vis_upto())

    class _VEv:
        __slots__ = ("inner", "seq")

        def __init__(self, inner, seq):
            self.inner, self.seq = inner, int(seq)

        def query(self):
            return self.seq <= vis_upto() and bool(self.inner.query())

        def synchronize(self):
            self.inner.synchronize()
            st["forced"] = max(st["forced"], self.seq)

        def wait_submitted(self):
            self.inner.wait_submitted()

        def wait(self, stream=None):
            self.inner.wait(stream)
            st["forced"] = max(st["forced"], self.seq)

    tp.issue_desc, tp.poll, tp.landed_upto = issue_desc, poll, landed_upto
    tp.job_ev = lambda seq: _VEv(o_ev(seq), seq)
    tp.slow_view = st
    return st


SLOW = {None: None, "mix": lambda s: 1 + s % 3, "late": lambda s: 3, "early": lambda s: 1,
        "never": lambda s: 1 << 30}                                   # [f3ldx] seen landed only once forced (SCRIPT)


def _arm_kw(d, table):
    kw = dict(lean=True, fewer_reads=d["fewer_reads"], wait_index=d["wait_index"], res_lean=d["res_lean"],
              res_dev=d["res_dev"])
    kw.update(d.get("extra") or {})
    if table:
        kw["no_transfer_mirror"] = True
    else:
        kw["inflight_array"] = d["ia"]
    kw.update(d.get("table_extra" if table else "mirror_extra") or {})
    return kw


def _mirror_facts(svc):
    """The mirror arm's consistency with the table (TO1): None, or the first mismatch. The Transfers in flight are the
    table's INFLIGHT slots; each has the slot's block, generation, issue tick and demand set."""
    T = svc.own
    G = T.geo
    infl = (T.c["code"] == C_INFLIGHT).nonzero().tolist()
    trs = list(dict.values(svc.inflight))
    if sorted((tr.layer, tr.b, tr.h, tr.slot) for tr in trs) != sorted(tuple(x) for x in infl):
        return ("slots", len(trs), len(infl))
    for tr in trs:
        l, b, h, s = tr.layer, tr.b, tr.h, tr.slot
        dem = int(T.c["demand"][l, b, h, s])
        want = {k for k, bit in ((OWN2.DRAFT, OWN2.O.DEM_D), (OWN2.VERIFY, OWN2.O.DEM_V)) if dem & bit}
        got = (int(T.c["block"][l, b, h, s]), int(T.c["gen"][l, b, h, s]), int(T.c["issue_tick"][l, b, h, s]), want)
        if got != (tr.block, tr.gen, tr.issue_tick, set(tr.demand)):
            return ("transfer", (l, b, h, s), got, (tr.block, tr.gen, tr.issue_tick, set(tr.demand)))
    return None


def _round(p3, v3, toks, d, table, plant_kw=None, facts=False, count=None, setup=None):
    """One round of draw d on the mirror arm (table False) or the table arm; plant_kw: more OwnResidency arguments
    (a control); facts: record _mirror_facts per tick; count: a dict that gets the per-tick in-flight map sizes (also
    kept in r.box["inflight"]); setup(eng, svc, ticks): before the round. Every tick record gets "view": the
    completion view's forced watermark after the tick (None without a view). [f3ldx] d["setup"] (the SCRIPTED draw's
    selection): called before setup."""
    ds = d.get("setup")
    if ds is not None:
        s0 = setup

        def setup(eng, svc, ticks, _a=ds, _b=s0):
            _a(eng, svc, ticks)
            if _b is not None:
                _b(eng, svc, ticks)
    kw = _arm_kw(d, table)
    kw.update(plant_kw or {})
    box = dict(inflight=[], view_ticks=[])

    def plant(tp, svc, ticks, world, probes):
        if SLOW[d["slow"]] is not None:
            box["view"] = _slow(tp, SLOW[d["slow"]])

        def probe(t, eng):                                            # (called before the tick's record is appended)
            if facts:
                box.setdefault("facts", []).append(_mirror_facts(svc))
            box["inflight"].append(len(svc.inflight))
            v = box.get("view")
            box["view_ticks"].append(None if v is None else v["forced"])
        probes.append(probe)
    r = BD._run(p3, v3, toks, "desc", cfg=d["cfg"], checked=d["checked"], late=d["late"], plant=plant, seed=d["seed"],
                lag=d["lag"], svc_kw=kw, forced=d["forced"], state=_state_tbl, layer_rec=RF._layer_rec, setup=setup)
    for i, t in enumerate(r.ticks):
        t["view"] = box["view_ticks"][i]
    if count is not None:
        count["inflight"] = list(box["inflight"])
    r.box = box
    return r


def _draws(n=8, seed0=5100):
    """TO1's randomized streams (random.Random(seed0 + i)); the completion view forces the array poll in the mirror arm
    (the old poll counts its held transfers after forces of earlier layers: test_r2_admitfill.py IA's rule)."""
    out = []
    for i in range(n):
        r = random.Random(seed0 + i)
        rl = r.random() < 0.5
        d = dict(name="rand%d" % i, cfg=r.choice(("default", "pressure")), seed=r.choice((1, 2, 3)),
                 forced=r.random() < 0.6, late=r.choice((0, 2)), lag=r.choice((1, 2)), checked=r.random() < 0.3,
                 fewer_reads=r.random() < 0.5, wait_index=r.random() < 0.5, res_lean=rl,
                 res_dev=rl and r.random() < 0.5, slow=r.choice((None, "mix", "late")))
        d["ia"] = True if d["slow"] else r.random() < 0.5
        out.append(d)
    return out


FIXED = [dict(name="eval", cfg="pressure", seed=1, forced=True, late=2, lag=2, checked=False, fewer_reads=True,
              wait_index=True, res_lean=True, res_dev=True, slow="mix", ia=True),
         dict(name="checked-free", cfg="default", seed=2, forced=False, late=2, lag=2, checked=True, fewer_reads=False,
              wait_index=False, res_lean=False, res_dev=False, slow="late", ia=True)]

# ---------------------------------------------------------------------------------------------------------------------
# [f3ldx] THE SCRIPTED DRAW (module docstring, THE CADENCE BOUND): a selection that reads the table at each resolve
# ---------------------------------------------------------------------------------------------------------------------
BD.CFG.setdefault("script", dict(C=3, X=0, R_ver=0))                 # NPOS = C: a full stream gets no K5 slot
U_SCRIPT = tuple(range(6))           # history blocks of every row: T = pos // 64 >= 7, so hist_bound >= T - 1 >= 6
SCRIPT_TARGET = (0, 3, 0)            # (layer, request, head): the ONE stream where Policy E acts (TO2iv: one victim)


def _stream_view(T, l, b, h):
    """The table's state of stream (l, b, h) at this resolve (the decisions of the tick apply at end_tick, so every
    layer of a tick sees the state after the tick's poll): resident blocks, unpinned LRU blocks, the current owner's
    blocks in flight (take stamp, block) and the unpinned not-landed draft prefetches among them (Policy E's in-flight
    candidates before the layer's selection, r2_own2_ops.py:706-707), the FREE count, the blocks of U_SCRIPT in no
    slot."""
    code = T.c["code"][l, b, h].tolist()
    blk = T.c["block"][l, b, h].tolist()
    uid = T.c["uid"][l, b, h].tolist()
    iss = T.c["issuer"][l, b, h].tolist()
    landed = T.c["landed"][l, b, h].tolist()
    stamp = T.c["stamp"][l, b, h].tolist()
    acc = None
    for row in T.pin_w[:, l, b, h].tolist():                          # any key plane (any_pin_words)
        acc = list(row) if acc is None else [x | y for x, y in zip(acc, row)]

    def pinned(x):
        return (acc[x // 32] >> (x % 32)) & 1 == 1
    own = int(T.owner[b])
    sl = list(zip(code, blk, uid, iss, landed, stamp))
    res = sorted(x for (c, x, _, _, _, _) in sl if c in (RSC.C_LRU, RSC.C_PO, RSC.C_VRES))
    lru_up = sorted(x for (c, x, _, _, _, _) in sl if c == RSC.C_LRU and not pinned(x))
    fly = sorted((s, x) for (c, x, u, _, _, s) in sl if c == C_INFLIGHT and u == own)
    fly_up = sorted((s, x) for (c, x, u, i, ld, s) in sl
                    if c == C_INFLIGHT and u == own and i == OWN2.O.ISS_D and not ld and not pinned(x))
    used = {x for (c, x, _, _, _, _) in sl if c != RSC.C_FREE}
    return dict(res=res, lru_up=lru_up, fly=fly, fly_up=fly_up, nfree=sum(1 for c in code if c == RSC.C_FREE),
                fresh=[x for x in U_SCRIPT if x not in used])


def _script_sel(svc, l, rows, sel, target):
    """The scripted selection of one layer (the rows' last two entries, blocks T - 1 and T, are kept; the first k - 2
    entries name the history blocks, padded with repeats: the device keeps unique blocks, r2_residency_sc.py:465-468).
    Per stream (l, b, h):
      TARGET and a verify row and the stream full (no FREE slot; X = 0: K5 refuses) and an unpinned prefetch in flight:
          verify AND draft rows select {a block in no slot} + every unpinned LRU block + (two or more candidates) the
          oldest prefetch. So K5 refuses the miss, no LRU candidate exists, and Policy E supersedes the second-oldest
          prefetch (r2_residency_sc.py:516-538), whose job no read forced; the draft row merges onto the verify take.
      else (preparation, no Policy E): verify rows read the lowest resident block, else the oldest block in flight
          (the device waits for it: the oldest job is the one forced), else a block in no slot (a FREE slot exists:
          nothing is resident or in flight); draft rows select the lowest block in no slot (a prefetch while a slot
          is FREE, else a refusal)."""
    T = svc.own
    H, R, k = sel.shape
    new = sel.clone()

    def pad(s):
        s = sorted(set(s))
        return torch.tensor(s + [s[-1]] * (k - 2 - len(s)), dtype=sel.dtype)
    for b in sorted({r.b for r in rows}):
        vi = [i for i, r in enumerate(rows) if r.b == b and r.kind == "v"]
        di = [i for i, r in enumerate(rows) if r.b == b and r.kind == "d"]
        for i in vi + di:
            assert rows[i].hist_bound >= len(U_SCRIPT) and k - 2 >= len(U_SCRIPT), ("SCRIPT: U_SCRIPT is not history",
                                                                                     rows[i], k)
        for h in range(H):
            v = _stream_view(T, l, b, h)
            assert v["fresh"], ("SCRIPT: every block of U_SCRIPT holds a slot", l, b, h, v)
            if (l, b, h) == target and vi and v["nfree"] == 0 and v["fly_up"]:
                spare = [v["fly_up"][0][1]] if len(v["fly_up"]) >= 2 else []
                sv = sd = [v["fresh"][0]] + v["lru_up"] + spare
            else:
                sv = v["res"][:1] or [x for (_, x) in v["fly"][:1]] or v["fresh"][:1]
                sd = v["fresh"][:1]
            for i in vi:
                new[h, i, :k - 2] = pad(sv)
            for i in di:
                new[h, i, :k - 2] = pad(sd)
    return new


def _script_setup(target=SCRIPT_TARGET):
    """setup(eng, svc, ticks) of the SCRIPTED draw: svc.resolve (already _run's recorder) gets the scripted
    selection; the engine's attention reads the plan's pages, so the tokens stay teacher-forced (forced=True)."""
    def setup(eng, svc, ticks):
        inner = svc.resolve

        def resolve(l, rows, sel):
            return inner(l, rows, _script_sel(svc, l, rows, sel, target))
        svc.resolve = resolve
    return setup


SCRIPT = dict(name="script-evict", cfg="script", seed=1, forced=True, late=0, lag=8, checked=False, fewer_reads=False,
              wait_index=False, res_lean=False, res_dev=False, slow="never", ia=True, setup=_script_setup())

# ---------------------------------------------------------------------------------------------------------------------
# [f3ldx] THE SCRIPTED END (module docstring, THE END IN FLIGHT): a request ends with a draft prefetch in flight
# ---------------------------------------------------------------------------------------------------------------------
BD.CFG.setdefault("script_end", dict(C=5, X=0, R_ver=0))             # NPOS = C = 5 < len(U_SCRIPT): a block is fresh


def _script_end_sel(svc, S_list, l, rows, sel):
    """The scripted selection of the END draw, per stream (l, b, h) (entries as _script_sel):
      verify rows: the lowest resident block, else the oldest block in flight (the device waits for it), else a
          block in no slot (never at the end: block A is resident from tick 8 on, lag 8);
      draft rows: at the request's LAST draft position (pos >= S_b + STEPS - 2, module docstring) and a FREE slot:
          a block in no slot (a draft prefetch, a GRANT: usage < C); else the lowest resident block, else the oldest
          block in flight (a merge, no take), else a block in no slot (the first prefetch, block A, at tick 0).
    So no verify misses (no urgent take, no Policy E) and every request's final epoch prefetches one block at its
    last draft row while the stream has a FREE slot (1 + 4 takes fit C = 5)."""
    T = svc.own
    H, R, k = sel.shape
    new = sel.clone()

    def pad(s):
        s = sorted(set(s))
        return torch.tensor(s + [s[-1]] * (k - 2 - len(s)), dtype=sel.dtype)
    for i, r in enumerate(rows):
        assert r.hist_bound >= len(U_SCRIPT) and k - 2 >= len(U_SCRIPT), ("SCRIPT END: U_SCRIPT is not history", r, k)
        last = r.kind == "d" and r.pos >= S_list[r.b] + TT.STEPS - 2
        for h in range(H):
            v = _stream_view(T, l, r.b, h)
            assert v["fresh"], ("SCRIPT END: every block of U_SCRIPT holds a slot", l, r.b, h, v)
            if last and v["nfree"] > 0:
                s = v["fresh"][:1]
            else:
                s = v["res"][:1] or [x for (_, x) in v["fly"][:1]] or v["fresh"][:1]
            new[h, i, :k - 2] = pad(s)
    return new


def _script_end_setup():
    def setup(eng, svc, ticks):
        inner, S_list = svc.resolve, [int(x) for x in eng.S_list]

        def resolve(l, rows, sel):
            return inner(l, rows, _script_end_sel(svc, S_list, l, rows, sel))
        svc.resolve = resolve
    return setup


SCRIPT_END = dict(SCRIPT, name="script-end", cfg="script_end", setup=_script_end_setup())
DRAWS = FIXED + _draws() + [SCRIPT, SCRIPT_END]
_PAIRS = {}


def _pair(p3, v3, toks, d):
    if d["name"] not in _PAIRS:
        _PAIRS[d["name"]] = (_round(p3, v3, toks, d, False, facts=True), _round(p3, v3, toks, d, True))
    return _PAIRS[d["name"]]


# =====================================================================================================================
# TO1 the table arm equals the mirror arm on randomized streams
# =====================================================================================================================
@pytest.mark.parametrize("d", DRAWS, ids=[d["name"] for d in DRAWS])
def test_TO1_the_table_arm_equals_the_mirror_arm_at_every_tick_and_layer(p3, ref, v3, d):
    toks, _ = ref
    mi, tb = _pair(p3, v3, toks, d)
    assert mi.err is None and len(mi.ticks) > 10, ("precondition: the mirror arm ran", d, mi.err)
    diff = _first_diff(mi, tb)
    assert diff is None, ("the table arm differs from the mirror arm", d, diff, tb.err)
    for i, (x, y) in enumerate(zip(mi.final, tb.final)):
        assert torch.equal(x, y), ("final store tensor", i)
    assert [r.out for r in tb.rnd.reqs] == [r.out for r in mi.rnd.reqs]
    assert TRM._journal_kinds(tb.svc) == TRM._journal_kinds(mi.svc)
    ro, rt = BD._rep(mi.svc), BD._rep(tb.svc)
    for k in ro:
        assert rt[k] == ro[k], (k, ro[k], rt[k])
    assert (mi.svc.arrival_forced, mi.svc.arrival_held) == (tb.svc.arrival_forced, tb.svc.arrival_held)
    assert mi.svc.desc_jobs == tb.svc.desc_jobs > 0 and mi.svc.desc_takes == tb.svc.desc_takes
    assert mi.svc.dev.waited == tb.svc.dev.waited, (mi.svc.dev.waited, tb.svc.dev.waited)
    bad = [(t, f) for t, f in enumerate(mi.box.get("facts", [])) if f is not None]
    assert not bad, ("the table does not carry the mirror's facts", bad[:3])
    assert tb.svc.no_transfer_mirror and not mi.svc.no_transfer_mirror
    assert tb.svc.nm_stats["polls"] > 0 and tb.svc.nm_stats["admitted"] > 0, tb.svc.nm_stats
    assert len(tb.svc.inflight) == 0 and len(tb.svc.pending) == 0 and not tb.svc._landed
    print("TO1 %s: ticks %d, jobs %d, takes %d, forced %d, held %d, table-arm stats %s"
          % (d["name"], len(tb.ticks), tb.svc.desc_jobs, tb.svc.desc_takes, tb.svc.arrival_forced, tb.svc.arrival_held,
             tb.svc.nm_stats))


# =====================================================================================================================
# TO0 the preconditions over the draws
# =====================================================================================================================
def test_TO0_preconditions_the_draws_issue_hold_force_evict_merge_and_end_in_flight(p3, ref, v3, monkeypatch):
    toks, _ = ref
    pairs = [_pair(p3, v3, toks, d) for d in DRAWS]                    # cached by TO1
    assert all(t.svc.desc_jobs > 0 for _, t in pairs), "precondition: an arm issued no descriptor job"
    assert sum(t.svc.arrival_held for _, t in pairs) > 0, "precondition: no take landed before it was due"
    assert sum(t.svc.arrival_forced for _, t in pairs) > 0, "precondition: no due take was forced"
    assert sum(t.svc.nm_stats["forced_jobs"] for _, t in pairs) > 0
    ev_by = {d["name"]: t.svc.journal_counts.get("release:verifier_evict_inflight", 0)
             for d, (_, t) in zip(DRAWS, pairs)}
    low = {d["name"]: ev_by[d["name"]] for d in DRAWS if d["lag"] <= 2}
    assert low and sum(low.values()) == 0, ("THE CADENCE BOUND (module docstring): with arrival_lag <= 2 no in-flight "
                                            "Policy E victim exists, yet a draw evicted one", low)
    ev_in = sum(ev_by.values())
    assert ev_by[SCRIPT["name"]] > 0, ("precondition: Policy E never evicted a slot in flight in the SCRIPTED draw",
                                       ev_by, pairs[DRAWS.index(SCRIPT)][1].err)
    assert all(max(m.box["inflight"]) > 0 for m, _ in pairs), "precondition: a mirror arm held no Transfer in flight"
    assert all(max(t.box["inflight"]) == 0 for _, t in pairs), "a table arm held a Transfer in flight"
    # the merges and the ends in flight: one more table-arm round of the eval draw with counters on the shared ops
    seen = dict(fv=0, fd=0, end_in_flight=0, ends=0)
    real_wm = OWN2.O.write_merges

    def wm(T, fv, fd, undo=None):
        seen["fv"] += int(fv.numel())
        seen["fd"] += int(fd.numel())
        return real_wm(T, fv, fd, undo)
    monkeypatch.setattr(OWN2.O, "write_merges", wm)
    d = FIXED[0]
    kw = _arm_kw(d, True)

    def plant(tp, svc, ticks, world, probes):
        _slow(tp, SLOW[d["slow"]])
        end0 = svc.end_request

        def end_request(uid):
            b = svc.uid_b[uid]
            seen["ends"] += 1
            if bool((svc.own.c["code"][:, b] == C_INFLIGHT).any()):
                seen["end_in_flight"] += 1
            return end0(uid)
        svc.end_request = end_request
    r = BD._run(p3, v3, toks, "desc", cfg=d["cfg"], late=d["late"], plant=plant, seed=d["seed"], lag=d["lag"],
                svc_kw=kw, forced=d["forced"], state=_state_tbl)
    assert r.err is None, r.err
    assert seen["fv"] > 0, ("precondition: no verify merge wrote a demand bit of a slot in flight", seen)
    assert seen["fd"] > 0, ("precondition: no draft merge wrote a demand bit of a slot in flight", seen)
    # [f3ldx] the ends in flight: the table arm of the SCRIPTED END draw (module docstring, THE END IN FLIGHT)
    ends = dict(ends=0, end_in_flight=0, end_transport=0)

    def count_ends(eng, svc, ticks):
        end0 = svc.end_request

        def end_request(uid):
            b = svc.uid_b[uid]
            code = svc.own.c["code"][:, b]
            infl = code == C_INFLIGHT
            ends["ends"] += 1
            ends["end_in_flight"] += int(bool(infl.any()))
            ends["end_transport"] += int(bool((infl & ~svc.own.c["landed"][:, b]
                                               & (svc.own.c["issuer"][:, b] == OWN2.O.ISS_D)).any()))
            return end0(uid)
        svc.end_request = end_request
    re_ = _round(p3, v3, toks, SCRIPT_END, True, setup=count_ends)
    assert re_.err is None, re_.err
    assert ends["ends"] > 0 and ends["end_in_flight"] > 0 and ends["end_transport"] > 0, (
        "precondition: no request ended with a draft prefetch in flight (SCRIPTED END draw)", ends, seen)
    print("TO0: Policy E in-flight evictions %d; merges %s; scripted ends %s" % (ev_in, seen, ends))


# =====================================================================================================================
# TO1f the fused v3 resolve (child, TRITON_INTERPRET=1)
# =====================================================================================================================
F3D_CHILD = r'''
import importlib.util, json, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "f3ld child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules
import torch
HERE, P3PATH, BUILD = sys.argv[1], sys.argv[2], sys.argv[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T = load("test_r2_f3ld_child", os.path.join(HERE, "test_r2_f3ld.py"))
FR = T.TT._load("r2_fusedres")
assert FR.triton is not None, "f3ld child: triton did not import"
gp = load("gilfree_pack_v3_f3ld_child", str(T.BD.EFF / "gilfree_pack.py"))
ext = gp.load(BUILD)
P3 = load("piece3_v2_f3ld_child", P3PATH)
toks, _ = T.TT.serial_ref(T.TT.make_engine(T.TT.make_ops(P3)))
d = dict(T.FIXED[0], name="fused-v3", extra=dict(resolve="fused", fused_kernel="v3"))
mi = T._round(P3, ext, toks, d, False, facts=True)
tb = T._round(P3, ext, toks, d, True)
diff = T._first_diff(mi, tb)
rep = tb.svc.fusedres.report()
print("F3D_JSON " + json.dumps(dict(
    diff=None if diff is None else [diff[0], str(diff[1])], err=[mi.err, tb.err], ticks=[len(mi.ticks), len(tb.ticks)],
    out_same=[r.out for r in mi.rnd.reqs] == [r.out for r in tb.rnd.reqs] if mi.rnd and tb.rnd else False,
    forced=[mi.svc.arrival_forced, tb.svc.arrival_forced], held=[mi.svc.arrival_held, tb.svc.arrival_held],
    kernel=rep["kernel"], layers=rep["layers"], lean_on=tb.svc.fusedres._lean_on(), jobs=tb.svc.desc_jobs,
    facts_bad=sum(1 for f in mi.box.get("facts", []) if f is not None), nm=tb.svc.no_transfer_mirror)))
'''


def test_TO1f_the_fused_v3_path_table_arm_equals_the_mirror_arm(p3, tmp_path_factory):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    build = tmp_path_factory.mktemp("gilfree3_f3ld_child_build")
    r = subprocess.run([sys.executable, "-c", F3D_CHILD, str(HERE), p3.__file__, str(build)], env=env,
                       cwd=str(HERE.parents[2]), capture_output=True, text=True, timeout=5400)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("F3D_JSON ")]
    assert r.returncode == 0 and last, ("child exit %d\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-6000:]))
    x = json.loads(last[-1][len("F3D_JSON "):])
    assert x["err"] == [None, None] and min(x["ticks"]) > 10, x
    assert x["kernel"] == "v3" and x["layers"] > 0 and x["lean_on"] and x["nm"], ("precondition: the fused v3 lean "
                                                                                  "path ran on the table arm", x)
    assert x["diff"] is None and x["out_same"] and x["forced"][0] == x["forced"][1] and x["held"][0] == x["held"][1], x
    assert x["facts_bad"] == 0, x
    assert sum(x["forced"]) > 0, ("precondition: the completion view forced due jobs", x)


# =====================================================================================================================
# TO2 negative controls at the sites under test
# =====================================================================================================================
def _clean_tbl(p3, v3, toks, d):
    key = ("clean", d["name"])
    if key not in _PAIRS:
        _PAIRS[key] = _round(p3, v3, toks, d, True)
    return _PAIRS[key]


def test_TO2i_a_dropped_draft_demand_bit_is_seen_in_that_ticks_table(p3, ref, v3):
    toks, _ = ref
    tried = []
    for d in [SCRIPT] + [x for x in DRAWS if x is not SCRIPT]:        # [f3ldx] the SCRIPTED draw acts by construction
        bad = _round(p3, v3, toks, d, True, plant_kw=dict(b2_mut={"nm_drop_dem_d": True}))
        done = bad.svc.nm_mut_done
        tried.append((d["name"], done))
        if done is None:
            continue                                                  # L34: no merge onto a {verify} take here
        clean = _clean_tbl(p3, v3, toks, d)
        t = done[0]
        diff = _first_diff(clean, bad)
        assert diff == (t, "table"), ("the dropped bit is first seen in its tick's table", diff, done)
        n = min(len(clean.ticks), len(bad.ticks))
        later = [k for k in range(t + 1, n) if clean.ticks[k]["state"] != bad.ticks[k]["state"]
                 or clean.ticks[k]["counters"] != bad.ticks[k]["counters"]]
        if not later and bad.err is not None:                         # a raise after the plant is a change too
            later = [len(bad.ticks)]
        assert later, ("the admission of the slot did not change (the demand bit is not read)", done)
        print("TO2i %s: plant %s, first diff %s, state / counters first differ at tick %d" % (d["name"], done, diff,
                                                                                            later[0]))
        return
    pytest.fail("precondition (L34): nm_drop_dem_d was a no-op in every draw (no draft merge onto a take whose "
                "demand is {verify}): %s" % tried)


def test_TO2ii_a_job_due_a_tick_early_is_seen_at_that_tick(p3, ref, v3):
    toks, _ = ref
    d = dict(FIXED[0], name="due-early", slow=None)
    clean = _round(p3, v3, toks, d, True)
    bad = _round(p3, v3, toks, d, True, plant_kw=dict(b2_mut={"nm_due_early": True}))
    done = bad.svc.nm_mut_done
    assert done is not None and done[1] == "nm_due_early", ("precondition (L34): the plant acted", done)
    diff = _first_diff(clean, bad)
    assert diff is not None and diff[0] == done[0], ("the early job is seen at its poll's tick", diff, done)


def test_TO2iii_a_due_job_not_forced_is_refused_by_the_poll(p3, ref, v3):
    toks, _ = ref
    d = dict(FIXED[0], name="skip-force", slow="late")
    clean = _round(p3, v3, toks, d, True)
    assert clean.err is None and clean.svc.arrival_forced > 0, ("precondition: the clean round forces", clean.err)
    bad = _round(p3, v3, toks, d, True, plant_kw=dict(b2_mut={"nm_skip_force": True}))
    done = bad.svc.nm_mut_done
    assert done is not None and done[1] == "nm_skip_force", ("precondition (L34): a due job had not landed", done)
    assert bad.err is not None and "force_seq first" in bad.err, bad.err
    diff = _first_diff(clean, bad)
    assert diff is not None and diff[0] == done[0] and str(diff[1]).startswith("raise"), (diff, done)


def test_TO2iv_a_victim_job_not_forced_is_seen_in_that_ticks_view(p3, ref, v3):
    """On the CPU stand-in the device wait forces a superseded prefetch first (CpuDeviceSim.wait, Policy E 12j) and its
    gather refuses a write before it landed (check_t3): so the host-side force of the victim's job acts only with
    both off (sc_mut no_inflight_wait and dev.check_t3 False, in the clean and the planted round alike).
    [f3ldx] The draw is the SCRIPTED one (module docstring, THE CADENCE BOUND: the earlier lag-2 candidates "pressure"
    1-3 / "default" 1 can have no in-flight victim). Its view lands a job only when forced (slow "never"), and Policy E
    acts on ONE stream, so the victim's job is forced by nothing else in that tick."""
    toks, _ = ref

    def no_t3(eng, svc, ticks):
        svc.dev.check_t3 = False
    tried = []
    for d in (dict(SCRIPT, name="victim-script", extra=dict(sc_mut={"no_inflight_wait": True})),):
        bad = _round(p3, v3, toks, d, True, plant_kw=dict(b2_mut={"nm_victim_unforced": True}), setup=no_t3)
        done = bad.svc.nm_mut_done
        tried.append((d["name"], done, bad.err))
        if done is None:
            continue                                                  # L34: no in-flight victim was pending here
        clean = _round(p3, v3, toks, d, True, setup=no_t3)
        assert clean.svc.nm_stats["victims_forced"] > 0, ("precondition: the clean round forces a victim's job",
                                                          clean.svc.nm_stats)
        vb, vc = bad.box["view_ticks"], clean.box["view_ticks"]
        t = done[0]
        assert t < min(len(vb), len(vc)), ("the plant's tick was not recorded", t, len(vb), len(vc), bad.err)
        assert vc[t] != vb[t], ("the unforced victim job is not seen in the view at its tick", t, vc[t], vb[t])
        assert all(vc[k] == vb[k] for k in range(t)), "the view differs before the plant"
        print("TO2iv %s: plant %s; forced watermark at its tick clean %s / planted %s" % (d["name"], done, vc[t], vb[t]))
        return
    pytest.fail("precondition (L34): nm_victim_unforced was a no-op in every draw: %s" % tried)


def test_TO2b_with_the_flag_off_the_mirror_demand_is_write_only(p3, ref, v3, monkeypatch):
    toks, _ = ref
    d = FIXED[0]
    seen = dict(fv=0, fd=0)
    real_wm = OWN2.O.write_merges

    def wm(T, fv, fd, undo=None):
        seen["fv"] += int(fv.numel())
        seen["fd"] += int(fd.numel())
        return real_wm(T, fv, fd, undo)
    monkeypatch.setattr(OWN2.O, "write_merges", wm)
    clean = _round(p3, v3, toks, d, False)
    dd = dict(d, name="drop-demand-merge", mirror_extra=dict(mut=RES.Mutations(drop_demand_merge=True)))
    bad = _round(p3, v3, toks, dd, False)
    assert seen["fv"] > 0 and seen["fd"] > 0, ("precondition: the merges wrote demand bits", seen)
    assert bad.svc.mut.drop_demand_merge and not clean.svc.mut.drop_demand_merge
    assert _first_diff(clean, bad) is None, "the mirror's tr.demand changed a decision"


# =====================================================================================================================
# TO3 static
# =====================================================================================================================
def _body(cls, name):
    return inspect.getsource(getattr(cls, name)).split('"""', 2)[2]


def test_TO3_the_table_path_holds_no_transfer_no_sort_no_host_map():
    O2 = OWN2.OwnResidency
    hs, ids, pa = _body(O2, "_host_side_nm"), _body(O2, "_issue_desc_nm"), _body(O2, "_poll_arrivals_nm")
    for src, name in ((hs, "_host_side_nm"), (ids, "_issue_desc_nm"), (pa, "_poll_arrivals_nm")):
        for w in ("Transfer(", "self.inflight", "self.pending", "self._landed", "_ia_", "self.metrics.issued",
                  "self.metrics.arrived"):
            assert w not in src, (name, w)
    assert "sorted(" not in pa and ".sort(" not in pa
    assert "O.force_seq(T, now, lu0, lag)" in pa and "O.poll_arrivals(T, now, lu, eff, None, mut or None)" in pa
    # the dispatch: the table path first, before the mirror lines
    for name, call in (("_poll_arrivals", "return self._poll_arrivals_nm()"), ("_host_side", "return self._host_side_nm("),
                       ("_in_flight", "return self._nm_view(l, b, h, t)")):
        src = inspect.getsource(getattr(O2, name))
        assert "if self.no_transfer_mirror:" in src and call in src, name
        assert src.index(call) < src.index("self.pending") if "self.pending" in src else True, name
    assert "self._issue_desc_nm(job[0])" in inspect.getsource(O2._tick_issue)
    # CudaDeviceOps: the jobs before the transfer loops
    for name in ("_submitted", "_superset_wait"):
        src = inspect.getsource(getattr(RSC.CudaDeviceOps, name))
        assert "jobs = self._nm_jobs(l)" in src and src.index("jobs = self._nm_jobs(l)") < src.index(
            "self.svc.inflight"), name
    # consumer 2: sparse.py:672 reads tr.demand only in ResidencySparse._drop_one, reached only from its apply_tick
    RSP = TT._load("r2_residency_sparse")
    assert "pkind in tr.demand" in inspect.getsource(RSP.ResidencySparse._drop_one)
    assert "_mutate(" in inspect.getsource(RSP.ResidencySparse.apply_tick)
    assert O2.apply_tick is not RSP.ResidencySparse.apply_tick
    own_src = (TT.NOSI / "r2_own2.py").read_text()
    assert "_drop_one(" not in own_src and "super().apply_tick(" not in own_src   # B2 never reaches _drop_one
    assert "self._mutate(" not in own_src
    # consumers 6 and 7 do not exist in B2: these read no mirror
    for name in ("end_request", "begin_request", "sync_mirror"):
        src = inspect.getsource(getattr(O2, name))
        for w in ("self.inflight", "self.pending", "_in_flight("):
            assert w not in src, (name, w)


def test_TO3b_the_host_objects_per_block_fall(p3, ref, v3, monkeypatch):
    toks, _ = ref
    made = dict(n=0)
    Real = OWN2.Transfer

    class Counting(Real):                                             # the fake: counts every construction
        def __init__(self, *a, **k):
            made["n"] += 1
            super().__init__(*a, **k)
    monkeypatch.setattr(OWN2, "Transfer", Counting)
    d = FIXED[0]
    cm, ct = {}, {}
    mi = _round(p3, v3, toks, d, False, count=cm)
    n_mirror = made["n"]
    made["n"] = 0
    tb = _round(p3, v3, toks, d, True, count=ct)
    n_table = made["n"]
    takes = mi.svc.journal_counts.get("take", 0)
    assert mi.err is None and tb.err is None, (mi.err, tb.err)
    assert takes > 0 and n_mirror == takes, ("the mirror arm: one Transfer per take", n_mirror, takes)
    assert n_table == 0, ("the table arm built Transfers", n_table)
    assert max(cm["inflight"]) > 0 and max(ct["inflight"]) == 0, (max(cm["inflight"]), max(ct["inflight"]))
    print("TO3b: Transfers per round mirror %d (takes %d), table %d; in-flight map peak mirror %d, table %d"
          % (n_mirror, takes, n_table, max(cm["inflight"]), max(ct["inflight"])))


# =====================================================================================================================
# TO4 refusals and wiring
# =====================================================================================================================
SG5_BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
NM = SG5_BASE + ["--backends", "b2", "--X", "64", "--arrival-lag", "2", "--lean-time", "--worker", "cpp",
                 "--b2-issue", "desc", "--no-transfer-mirror"]


def test_TO4_refusals_and_the_harness_wiring(p3, v3):
    def eng(issue="desc", **kw):
        e, _ = BD._engine(p3, v3, issue, svc_kw=kw)
        e.res_world.tp.close()
        return e.res_world.svc
    s = eng(lean=True, no_transfer_mirror=True)                       # positive
    assert s.no_transfer_mirror and s.nm_jobs_at(0) == [] and s.report()["b2"]["transfer_mirror"] is False
    assert eng(lean=True).nm_jobs_at(0) is None                       # the mirror path: no jobs list
    for kw, issue, msg in ((dict(lean=True, no_transfer_mirror=True), "object", "issue='desc'"),
                           (dict(no_transfer_mirror=True), "desc", "lean=True"),
                           (dict(lean=True, no_transfer_mirror=True, inflight_array=True), "desc", "no inflight_array"),
                           (dict(lean=True, no_transfer_mirror=True, b2_mut={"free_on_cancel": True}), "desc",
                            "act on the Transfer mirror only"),
                           (dict(lean=True, no_transfer_mirror=True, b2_mut={"owner_blind": True}), "desc",
                            "act on the Transfer mirror only"),
                           (dict(lean=True, b2_mut={"nm_skip_force": True}), "desc", "need no_transfer_mirror")):
        with pytest.raises(ValueError, match=re.escape(msg)):
            eng(issue, **kw)
    with pytest.raises(ValueError, match="a fixed arrival_lag"):
        e, _ = BD._engine(p3, v3, "desc", lag=None, svc_kw=dict(lean=True, no_transfer_mirror=True))
    G = HF._sg5()
    a = G.parse_args(NM)
    assert a.no_transfer_mirror and a.b2_issue == "desc" and a.worker == "cpp" and a.lean_time and not a.inflight_array
    assert G.parse_args(SG5_BASE + ["--backends", "b2"]).no_transfer_mirror is False      # the default: the mirror
    for bad in ([x for x in NM if x != "--lean-time"],
                NM[:NM.index("--worker")] + ["--no-transfer-mirror"],                     # no --b2-issue desc
                [x if x != "2" else "0" for x in NM],                                      # --arrival-lag 0
                NM + ["--inflight-array"]):
        with pytest.raises(SystemExit):
            G.parse_args(bad)
    a = G.parse_args(NM + ["--layout", "pool"])                                            # [f3ldx] the lane row
    assert a.no_transfer_mirror and a.layout == "pool"
    me = inspect.getsource(G.make_engine)
    assert 'no_transfer_mirror=bool(getattr(a, "no_transfer_mirror", False))' in me
    assert 'issue=getattr(a, "b2_issue", "object")' in me and 'worker=getattr(a, "worker", "python")' in me
    main = inspect.getsource(G.main)
    assert "af.no_transfer_mirror = False" in main and "meta.update(no_transfer_mirror=a.no_transfer_mirror)" in main
    ps = inspect.signature(OWN2.OwnResidency.__init__).parameters
    assert ps["no_transfer_mirror"].default is False and ps["issue"].default == "object"
    # T21 style: the gate helpers and lists are unchanged by f3ld
    op = inspect.getsource(G.one_point)
    assert sorted(set(re.findall(r"gates\.extend\((\w+)\(", op))) == ["fused_gates", "stack_acted_gates"]
    assert sorted(set(re.findall(r'GB\.gate\("%s_([A-Za-z0-9_]+)"', op))) == ["SKV_ran", "bookkeeping", "completed",
                                                                              "gilfree_team"]
    for s_ in ("eval_gates(a, meta, res, gates)", "acc_gates(a, meta, res, gates, acc["):
        assert s_ in main, s_
    for name in ("_host_side_nm", "_poll_arrivals_nm", "_issue_desc_nm", "_nm_view", "nm_jobs_at", "_nm_landed_tick",
                 "_nm_arrived", "_nm_after_cancel", "_nm_drop_dem_d", "_queue_desc", "_tick_issue", "_issue_desc"):
        assert callable(getattr(OWN2.OwnResidency, name, None)), name
    for k in OWN2.NM_MUTS:
        assert k in OWN2.B2_MUTS, k
    assert set(OWN2.NM_REFUSED_MUTS) <= set(OWN2.B2_MUTS)
    assert callable(getattr(RSC.CudaDeviceOps, "_nm_jobs", None))


# =====================================================================================================================
# TO5 the cells
# =====================================================================================================================
def _reg(text):
    reg = text[text.index("# REGISTERED for evalfast2f3d / evalfast2c3d / evalfast2f3t / evalfast2c3t"):]
    return reg[:reg.index("\n#\n")]


def _cell(text, head):
    i = text.index(head)
    return text[i:text.index(" ;;\n", text.index("EVALARG=(", i)) + 3]


def test_TO5_the_cells_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + F3LD_ROOT
    assert F3LD_ROOT == HF.HF_ROOT
    assert "|evalfast2f3lia|evalfast2c3lia|evalacc216f3lia|evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t" \
           "|evalfast2f3c|evalfast2c3c] [gilfree|gather]" in text                     # [f3ldx] f3c / c3c appended
    assert "        evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    f3 = _cell(text, "    evalfast2f3d|evalfast2f3t) MODES=fast")
    for w in ("BATCH=216", "PX=1512 ; PRV=1464", "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL", "GF3=1",
              "F3LD_BANDS=", "F3LT_BANDS=", "F3FLAGS=(--inflight-array --worker cpp --b2-issue desc)",
              "F3FLAGS=(--worker cpp --b2-issue desc --no-transfer-mirror)", '[ "${DRYRUN:-0}" = 1 ]',
              "--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped",
              "--eval-role fast-time --mem-cap-gb 84", '--eval-bands "$F3BANDS" --b2-resolve fused --fused-kernel v3',
              '--wait-index --b2-fewer-reads --res-lean --res-dev "${F3FLAGS[@]}"'):
        assert w in f3, w
    assert "STEP_MIN=" not in f3 and "STEPS=" not in f3
    c3 = _cell(text, "    evalfast2c3d|evalfast2c3t) MODES=fast")
    for w in ("STEPS=8 ; DTICKS=4", "BATCH=216", "GF3=1", "C3FLAGS=(--inflight-array --worker cpp --b2-issue desc)",
              "C3FLAGS=(--lean-time --worker cpp --b2-issue desc --no-transfer-mirror)",
              '--b2-resolve check --fused-kernel v3 --res-lean --res-dev "${C3FLAGS[@]}"'):
        assert w in c3, w
    for w in ("--bare", "--eval-role", "--accuracy-record", "--eval-bands"):
        assert w not in c3, w
    # the control evalfast2f3lia keeps its arguments (test_r2_f3lia.py F3G)
    assert "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array) ;;" in text
    # the packer: v3 of --nosi for the GF3 cells (r2_gfw_step.sh's rule), DESC 2 checked ([f3ldx]); copied from PK_*
    for w in ("GF3=                ", 'PK_PY_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$GF_PY")',
              "grep -q 'mod.attr(\"DESC\") = 2;'", 'git -C "$NOSI" show "$PK_PY_BLOB" > "$GF_DIR/gilfree_pack.py"',
              'git -C "$NOSI" show "$PK_CPP_BLOB" > "$GF_DIR/gilfree_pack.cpp"'):
        assert w.strip() in text, w
    assert 'git -C "$NOSI" show "$GF_PY_BLOB" >' not in text
    assert "F3LD_BLOCKED" not in text.replace("so F3LD_BLOCKED is gone", "")       # [f3ldx] the blocker is gone
    assert "grep -q 'mod.attr(\"DESC\") = 1;'" not in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_TO5_the_registration_is_left_to_the_lead():
    text = HF.STEP.read_text()
    reg = _reg(text)
    for tag in ("F3LD-BLOCK", "C3D / C3T", "F3D-MED", "F3D-MEAN", "F3D-EFF", "F3D-ISSUE", "F3D-SAME", "F3D-FALS",
                "F3T-MED", "F3T-MEAN", "F3T-EFF", "F3T-T04", "F3T-SAME", "F3T-FALS", "F3LD-MEM", "nosi-r2-f3ld",
                "89c9169", "47db0c0", "evalfast2f3lia", "0 diffs"):
        assert tag in reg, tag
    preds = [ln for ln in reg.splitlines() if "PREDICTION:" in ln]
    assert len(preds) >= 15, preds
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state: nothing predicted
        assert all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
        for v in ("F3LD_BANDS", "F3LT_BANDS"):
            assert re.search(r"^\s+%s=TBD\s" % v, text, re.M), v
    else:                                                             # the lead filled and stamped it
        assert "TBD" not in reg, "a stamped block with a TBD left"


@pytest.mark.parametrize("cell", CELLS)
def test_TO5_the_step_stops_outside_dryrun(tmp_path, cell):
    """Outside DRYRUN: the timing cells stop on the TBD bands (while unstamped); [f3ldx] the CHECK cells are not
    blocked: with the test's holder directory they stop at the SG5_HOLDERS rule (DRYRUN only), before any launch."""
    text = HF.STEP.read_text()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", cell], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-300:])
    assert "r2_tick_sg5.py" not in r.stdout
    timing = cell in ("evalfast2f3d", "evalfast2f3t")
    assert "BLOCKED" not in r.stderr, r.stderr[-300:]
    if timing and re.search(r"^\s+F3L[DT]_BANDS=TBD\s", text, re.M):
        assert "cell %s: --eval-bands TBD" % cell in r.stderr, r.stderr[-300:]
    else:
        assert "SG5_HOLDERS is for DRYRUN" in r.stderr, r.stderr[-300:]


@pytest.mark.parametrize("cell", CELLS)
def test_TO5d_the_cells_dryrun(tmp_path, cell, capsys):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(F3LD_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the f3ld ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    print("TO5d DRYRUN %s: %s" % (cell, got))
    timing = cell in ("evalfast2f3d", "evalfast2f3t")
    nm = cell.endswith("t")
    for s in ("--backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2",
              "--repeat-docs --log-slots -1 --skip-shipped", "--variants fused/batched", "--mem=240G", "--time=120",
              "--fused-kernel v3", "--res-lean --res-dev", "--worker cpp --b2-issue desc",
              "PYTHONPATH=%s/nosi" % F3LD_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell,
              "--b2-resolve %s" % ("fused" if timing else "check")):
        assert s in got or s in line, (cell, s)
    assert ("--no-transfer-mirror" in got) == nm and ("--inflight-array" in got) == (not nm), cell
    assert "[f3ld] packer copied: %s (packer v3 of --nosi, DESC 2)" % HF._clean_head() in r.stdout
    assert "blocked:" not in r.stdout                                 # [f3ldx] F3LD-BLOCK cleared
    argv = got.split()
    if timing:
        bands = "TBD" if "--eval-bands TBD" in got else None
        assert bands == "TBD" or re.search(r"--eval-bands [0-9.]+(:[0-9.]+){5}", got), got
        assert "--bare" in got and "--eval-role fast-time" in got and "--lean-time" in got and "--steps 48" in got
    else:
        assert "--steps 8" in got and "--draft-ticks 4" in got and "--bare" not in got and "--eval-role" not in got
        assert ("--lean-time" in got) == nm
    G = HF._sg5()
    capsys.readouterr()
    if timing and "--eval-bands TBD" in got:                          # the placeholder bands: refused by the harness
        with pytest.raises(SystemExit):
            G.parse_args(argv)
        assert "--eval-bands" in capsys.readouterr().err
        alt = list(argv)
        alt[alt.index("--eval-bands") + 1] = "1:2:1:2:1:2"
        argv = alt
    pa = G.parse_args(argv)                                           # [f3ldx] on --layout pool: no blocker left
    assert pa.layout == "pool" and pa.X_pool == 1512
    assert pa.b2_issue == "desc" and pa.worker == "cpp" and pa.no_transfer_mirror == nm
    assert pa.inflight_array == (not nm) and pa.b2_resolve == ("fused" if timing else "check")
    assert pa.fused_kernel == "v3" and pa.res_lean and pa.res_dev and pa.arrival_lag == 2 and pa.batch == 216
