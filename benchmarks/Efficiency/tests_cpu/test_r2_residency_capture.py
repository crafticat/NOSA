"""CPU tests of W5: S-C at FIXED CAPACITY split for the captured tick (nosi/nosi/r2_residency_capture.py CaptureSC; C's
Q-T3 answers), against S-C's own resolve (r2_residency_sc.ResidencySC, C's 41078e8). The worlds, the schedule and the
state comparison are C's (test_r2_residency.py: world, _sched, _state, _counters).

  S1  LOCKSTEP on F6's 24-tick schedule (2 requests, 2 layers, draft rows every tick, verify rows every third tick,
      seeded delays): world A runs ResidencySC.resolve per layer (read_done after each) and end_tick; world B runs
      CaptureSC: host_part before the "replay", device_part per layer on the FIXED buffers (R_cap 6 tick rows, Rv_cap 3
      verify and Rd_cap 3 draft entries, so padded entries exist at every tick; the padded tick rows carry REAL
      selections, a copy of row 0's), after_replay with ONE reader event. Every tick and layer: the page lists, the
      num_pages, last_len and seq_idx of every live row equal; after every tick the ownership state, the counters, the
      device mirror, the stores and the copy counts (mirror, rows, device reads) equal. Preconditions: the verify rows
      met ready, in-flight and new blocks; the device waited and gathered.
  S2  the same with C's EXACT device wait (ExactWaitSim, a publishing transport).
  S3  NEGATIVE CONTROLS: (a) the inert row made live (another request's stream, every block valid): the lockstep
      breaks; (b) host_part without sync_mirror: the lockstep breaks; (c) a watched flag set before after_replay:
      end_tick refuses the tick (CapacityRefusal).
  S4  the lint: device_part is clean; C's per-layer device functions (device_verify, device_draft, rank_in_group,
      wait_list, DeviceMirror.layer) show only transient allocations, in-place mirror updates and the static
      negative-control switches.
  S5  FixedTickEngine.watch_flags_with(svc): S-C watches tick.err and every twin's overflow; a set twin overflow makes
      the next end_tick refuse its tick; the decision read then reads the tokens only. Negative control: unwatched, the
      engine's own read raises instead.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_residency_capture.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


RT = _load_by_path("test_r2_residency", HERE / "test_r2_residency.py")   # C's worlds and schedules
LINT = _load_by_path("r2_capture_lint", HERE / "r2_capture_lint.py")
CAP = RT._load("r2_residency_capture")
SC, RES, ER = RT.SC, RT.RES, RT.ER
Row, L, B, H, NB = RT.Row, RT.L, RT.B, RT.H, RT.NB
K_CAP, R_CAP, RV_CAP, RD_CAP = 10, 6, 3, 3


def pad_sel(sel, R_cap=None):
    """sel (H, R, k) -> (H, R_cap or R, K_CAP): -1 past the row's blocks; padded rows copy row 0's selection (in the
    tick a padded row's selection is real: the dummy row's)."""
    Hs, R, k0 = sel.shape
    out = torch.full((Hs, R_cap or R, K_CAP), -1, dtype=torch.int64)
    out[:, :R, :k0] = sel
    if R_cap:
        out[:, R:] = out[:, :1]
    return out


class LeakyPad(CAP.CaptureSC):
    """NEGATIVE CONTROL (a): the inert row made live (request 1's streams, every block below nblk valid)."""

    def host_part(self, rows):
        super().host_part(rows)
        self.rb[self.R_cap] = 1
        self.hb[self.R_cap] = self.svc.nblk


class NoSync(CAP.CaptureSC):
    """NEGATIVE CONTROL (b): host_part without the mirror delta (the device mirror goes stale)."""

    def host_part(self, rows):
        self.svc.sync_mirror = lambda: False
        try:
            super().host_part(rows)
        finally:
            del self.svc.sync_mirror


def lockstep(sched, C=8, X=48, R_ver=8, seed=1, exact=False, cap_cls=None, hb=15):
    common = dict(C=C, X=X, R_ver=R_ver, seed=seed, mode="delay", max_delay=4)
    kw = dict(nblk=NB, wait_mode="exact" if exact else "probe")
    wa = RT.world(**common, cls=SC.ResidencySC, publish=True if exact else None, svc_kw=dict(kw))
    wb = RT.world(**common, cls=SC.ResidencySC, publish=True if exact else None, svc_kw=dict(kw))
    cap = (cap_cls or CAP.CaptureSC)(wb.svc, R_CAP, RV_CAP, RD_CAP, K_CAP)
    uids = ("u0", "u1")
    for w in (wa, wb):
        for b, u in enumerate(uids):
            w.svc.begin_request(b, u)
    tail = wa.svc.tail0
    readers, bad = [], []
    for (t, m, ends, layers) in sched:
        for w in (wa, wb):
            w.svc.poll()
        rows = [Row(b, kind, key, 1000 + t, hb, [(hb, tail)]) for (b, kind, key, _) in layers[0]]
        R = len(rows)
        ev_a, ev_b = RT.Ev(), RT.Ev()
        pa = []
        for l, spec in enumerate(layers):                                 # A: S-C's own resolve
            p = wa.svc.resolve(l, rows, pad_sel(RT.S(*[s for (_, _, _, s) in spec])))
            wa.svc.read_done(l, p, ev_a)
            pa.append(p)
        wa.svc.end_tick()
        cap.host_part(rows)                                               # B: the split at fixed capacity
        pb = [cap.device_part(l, pad_sel(RT.S(*[s for (_, _, _, s) in spec]), R_CAP)) for l, spec in enumerate(layers)]
        try:
            cap.after_replay(ev_b)
        except Exception as e:                                         # noqa: BLE001 (a negative control may raise)
            bad.append(("raised", t, type(e).__name__, str(e)[:120]))
            break
        readers.append((t, ev_a, ev_b))
        for l in range(L):
            page, npg, last, seq = pb[l]
            for i in range(R):
                if not (torch.equal(pa[l].page_table[i], page[i]) and torch.equal(pa[l].num_pages[i], npg[i])
                        and int(pa[l].last_len[i]) == int(last[i]) and int(pa[l].seq_idx[i]) == int(seq[i])):
                    bad.append(("pages", t, l, i))
        for (tt, x, y) in readers:
            if tt <= t - 2:
                x.done = y.done = True
        if ends:
            for w in (wa, wb):
                for u in uids:
                    w.svc.decided((u, 1, m))
        if RT._state(wa) != RT._state(wb):
            bad.append(("state", t))
        if RT._counters(wa.svc) != RT._counters(wb.svc):
            bad.append(("counters", t))
        dm_a, dm_b = wa.svc.dmirror, wb.svc.dmirror
        if not all(torch.equal(getattr(dm_a, f), getattr(dm_b, f)) for f in ("code", "block", "gen", "cur")):
            bad.append(("device mirror", t))
        for l in range(L):
            for name in ("store_k", "store_v", "store_b"):
                if not torch.equal(getattr(wa, name)[l], getattr(wb, name)[l]):
                    bad.append(("store", t, l, name))
        if (wa.svc.mirror_uploads, wa.svc.row_uploads, wa.svc.d2h_reads) != \
                (wb.svc.mirror_uploads, wb.svc.row_uploads, wb.svc.d2h_reads):
            bad.append(("copies", t))
        if bad:
            break
    return bad, wa, wb, cap


def test_S1_fixed_capacity_split_equals_resolve():
    bad, wa, wb, cap = lockstep(RT._sched(101, 24, True))
    assert bad == [], bad[:4]
    assert cap.stats["ticks"] == 24 and wb.svc.d2h_reads == 24
    k43 = wb.svc.metrics.k43
    assert k43["ready"] > 0 and k43["in_flight"] > 0 and k43["new"] > 0, k43
    assert wb.svc.dev.waited > 0 and wb.svc.dev.gathered > 0


def test_S2_with_the_exact_device_wait():
    bad, wa, wb, cap = lockstep(RT._sched(101, 24, True), exact=True)
    assert bad == [], bad[:4]
    assert wb.svc.dev.waited > 0 and not bool(wb.svc.dev.err)


@pytest.mark.parametrize("cls", [LeakyPad, NoSync])
def test_S3_negative_controls_break_the_lockstep(cls):
    bad, _, _, _ = lockstep(RT._sched(101, 24, True), cap_cls=cls)
    assert bad, cls.__name__


def test_S3c_a_watched_flag_refuses_the_tick():
    sched = RT._sched(101, 6, True)
    common = dict(C=8, X=48, R_ver=8, seed=1, mode="delay", max_delay=4)
    w = RT.world(**common, cls=SC.ResidencySC, svc_kw=dict(nblk=NB))
    flag = torch.zeros((), dtype=torch.bool)
    w.svc.watch_flag("probe", flag)
    cap = CAP.CaptureSC(w.svc, R_CAP, RV_CAP, RD_CAP, K_CAP)
    for b, u in enumerate(("u0", "u1")):
        w.svc.begin_request(b, u)
    tail = w.svc.tail0
    refused = None
    for (t, m, ends, layers) in sched:
        w.svc.poll()
        rows = [Row(b, kind, key, 1000 + t, 15, [(15, tail)]) for (b, kind, key, _) in layers[0]]
        cap.host_part(rows)
        for l, spec in enumerate(layers):
            cap.device_part(l, pad_sel(RT.S(*[s for (_, _, _, s) in spec]), R_CAP))
        if t == 3:
            flag.fill_(True)
            before = RT._state(w)
        ev = RT.Ev(True)
        try:
            cap.after_replay(ev)
        except ER.CapacityRefusal:
            refused = t
            break
    assert refused == 3 and RT._state(w) == before and w.svc.flag_values.get("probe") == 1


# =====================================================================================================================
# S4: the lint
# =====================================================================================================================
def test_S4_device_part_is_clean_and_C_device_functions_show_only_documented_findings():
    root = HERE.parents[2] / "nosi/nosi"
    src = (root / "r2_residency_capture.py").read_text()
    assert LINT.lint_function(LINT.fn_from_source(src, "CaptureSC.device_part")) == []
    documented = [("ALLOC", r"."), ("HOSTSTATE", r"^m\.\w+ (\|=|\+=|-=|&=)"), ("BRANCH", r"mut\.get\(")]
    for mod, qual in (("r2_residency_sc.py", "device_verify"), ("r2_residency_sb.py", "device_draft"),
                      ("r2_residency_sb.py", "rank_in_group"), ("r2_residency_sc.py", "wait_list"),
                      ("r2_residency_sc.py", "DeviceMirror.layer")):
        found = LINT.lint_function(LINT.fn_from_source((root / mod).read_text(), qual))
        assert LINT.unexplained(found, documented) == [], (qual, LINT.unexplained(found, documented))
    planted = LINT.fn_from_source(src, "CaptureSC.device_part")
    import ast
    planted.body.insert(1, ast.parse("x = int(self.nd)").body[0])
    assert "SYNC" in {c for c, _, _ in LINT.lint_function(planted)}


# =====================================================================================================================
# S5: the twin overflow and the selection check through S-C's watch_flag
# =====================================================================================================================
TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")
p3 = TT.p3


def test_S5_watch_flags_with(p3):
    FX = TT._load("r2_tick_fixed")
    eng = FX.FixedTickEngine.from_engine(TT.make_engine(TT.make_ops(p3)), v_cap=3)
    w = RT.world(C=8, X=48, R_ver=8, seed=1, cls=SC.ResidencySC, svc_kw=dict(nblk=NB))
    eng.watch_flags_with(w.svc)
    assert sorted(w.svc._flags) == sorted(["tick.err"] + ["twin%d.overflow" % l for l in range(eng.L)])
    assert eng.flag_watcher is w.svc
    eng.twins[1].overflow.fill_(True)
    assert eng.read_tokens(0, 0) == []                                 # the decision read: the tokens only
    with pytest.raises(ER.CapacityRefusal):
        w.svc.end_tick()                                               # S-C's read refuses the tick
    assert w.svc.flag_values["twin1.overflow"] == 1
    own = FX.FixedTickEngine.from_engine(TT.make_engine(TT.make_ops(p3)), v_cap=3)   # NEGATIVE CONTROL: unwatched
    own.twins[1].overflow.fill_(True)
    with pytest.raises(TT.TK.TickError):
        own.read_tokens(0, 0)
