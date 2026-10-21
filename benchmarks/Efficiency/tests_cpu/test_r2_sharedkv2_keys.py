"""CPU tests of the Residency plan's PAGE KEYS ([sharedkv2]; docs/R2_SHAREDKV2_KEYS_S3.md): S-C / B2 build
plan.page_key (the block id of every page_table entry, on the device, no host read) so that the shared-read attention
(nosi/nosi/bi_paged_attention_shared.py) can run on the SG5 path. Oracle: the ResidentStore's own keys
(TickEngine.page_keys, the production function) on the same teacher-forced schedule, and the host ownership state at
the tick start (slot -> block). SOURCE ONLY; the lead runs them on a compute node:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_sharedkv2_keys.py
"""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIECE3_BLOB = "39ed5e1894ca1662f92e384a10e01a4eaab99042"
sys.path.insert(0, str(HERE))
import test_r2_tick as RT  # noqa: E402  (its tiny model, engines, serial reference and tick check)

TK = RT.TK
RSC = RT.RSC
RES = RT.RES

# the S-C lint's token lists (test_r2_residency.py FORBIDDEN / H2D, copied: that module is not imported here)
FORBIDDEN = (".item(", ".tolist(", ".nonzero(", ".cpu(", ".numpy(", "bool(", "int(", "float(", ".any()", ".all()")
H2D = ("torch.tensor(", "as_tensor(", "from_numpy(", ".copy_(", "non_blocking", "pin_memory", ".cuda(", ".to(dv",
       ".to(dev", ".to(self.device", ".refresh(", "_h2d(", "_d2h(")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sk = _load(ROOT / "nosi/nosi/bi_paged_attention_shared.py", "sharedkv2_keys_under_test")


@pytest.fixture(scope="module")
def p3(tmp_path_factory):
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3_BLOB], capture_output=True)
    assert body.returncode == 0 and body.stdout
    path = tmp_path_factory.mktemp("p3k") / "piece3_v2.py"
    path.write_bytes(body.stdout)
    return _load(path, "piece3_v2_sharedkv2_keys")


@pytest.fixture(scope="module")
def ref(p3):
    return RT.serial_ref(RT.make_engine(RT.make_ops(p3)))


def ops_with(attn):
    return SimpleNamespace(linear=RT.exact_linear, attn=attn, mean=RT.BD.fixed_order_mean)


def key_only(p3):
    """An attention object that asks for keys (needs_page_key) and runs piece 3's twin, ignoring them: the keys flow
    through the production tick, the outputs stay piece 3's (a wrong key cannot raise inside the attention)."""
    def call(*args, page_key=None, rows_max=None, group_token=None):       # [sharedkv3] the tick's per-tick token
        assert page_key is not None and page_key.dtype == torch.int32
        return p3.bi_paged_decode_reference(*args)
    return SimpleNamespace(needs_page_key=True, bi_paged_decode=call, build_page_table=p3.build_page_table)


class KeyCheck(TK.ResidencyBackend):
    """The production ResidencyBackend plus the ORACLE, per (layer call, row, head):
      verify rows: plan.page_key[:num_pages] == the ResidentStore's keys (TickEngine.page_keys on the same sel: the
                   whole selection, ascending, T - 1 and T included);
      draft rows:  == the ResidentStore's keys filtered to the blocks READY at the tick start (host ownership state
                   after the tick's poll: Stream.resident) plus the tail blocks (>= hist_bound);
      every row:   each historical entry whose block is resident at the tick start is read from THAT slot
                   (page_table and page_key aligned), and sk.validate_keys accepts the plan.
    Mismatches are recorded, never raised (the negative control counts them)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.snap = {}
        self.ticks = 0
        self.n = dict(v=0, d=0, d_masked=0, d_hist=0, calls=0, hist_slot_checked=0)
        self.bad = []
        self.bad_slot = []
        self.invalid = []

    def begin_tick(self):
        super().begin_tick()                                     # the service's poll: the tick-start state
        self.ticks += 1
        self.snap = {}
        for l in range(self.eng.L):
            for (b, h), st in self.svc.pools[l].streams.items():
                for blk, idx in list(st.lru.items()) + list(st.po.items()) + list(st.vres.items()):
                    self.snap[(l, b, h, blk)] = idx

    def page_keys(self, l, ctx, sel, layer, plan=None):
        key = super().page_keys(l, ctx, sel, layer, plan=plan)
        rs = self.eng.page_keys(l, sel, ctx.pos, ctx.rb, ctx.is_draft, ctx.any_draft, layer)
        K, RS = key.tolist(), rs.tolist()
        npg, pt = plan.num_pages.tolist(), plan.page_table.tolist()
        self.n["calls"] += 1
        try:
            sk.validate_keys(plan.page_table, plan.num_pages, key)
        except ValueError as e:
            self.invalid.append((self.ticks, l, str(e)[:120]))
        for i, r in enumerate(ctx.rows):
            for h in range(self.eng.H):
                n = npg[i][h]
                got, slots = K[i][h][:n], pt[i][h][:n]
                listed = list(RS[i][h])
                if r.kind == "v":
                    want = listed
                    self.n["v"] += 1
                else:
                    want = [t for t in listed if t >= r.hist_bound or (l, r.b, h, t) in self.snap]
                    self.n["d"] += 1
                    self.n["d_masked"] += len(want) < len(listed)
                    self.n["d_hist"] += any(t < r.hist_bound for t in want)
                if got != want:
                    self.bad.append((self.ticks, l, i, h, r.kind, got, want))
                for t, s in zip(got, slots):
                    if t < r.hist_bound and (l, r.b, h, t) in self.snap:
                        self.n["hist_slot_checked"] += 1
                        if s != self.snap[(l, r.b, h, t)]:
                            self.bad_slot.append((self.ticks, l, i, h, t, s, self.snap[(l, r.b, h, t)]))
        return key


@pytest.mark.parametrize("kind", ["sc", "b2"])
def test_K1_residency_keys_equal_the_resident_store(p3, ref, kind):
    """K1 (equality): on the teacher-forced schedule (SG1 on the CPU tiny model, K3X's adversarial transport), every
    keyed layer call of the Residency tick (S-C / B2) gives plan.page_key equal to the oracle of KeyCheck, with no
    slot misalignment and validate_keys passing; the tick still equals the serial rows (T15). Preconditions (else
    blind): keyed calls happened, verify and draft lists were compared, some draft list was MASKED (a selected block
    not ready: its keys are shorter than the ResidentStore's) and some draft list read historical blocks, and the
    slot alignment was checked on historical entries."""
    toks, logits = ref
    eng = RT.make_res_engine(ops_with(key_only(p3)), kind=kind, backend_cls=KeyCheck)
    svc, be = eng.res_world.svc, eng.backend
    assert svc.want_keys is True
    res = RT.tick_check(eng, toks, logits, forced=True)
    assert res["compared"] > 50 and not res["bad"] and all(res["committed_ok"]) and not res["mirror_bad"]
    n = be.n
    assert n["calls"] > 0 and n["v"] > 0 and n["d"] > 0, n
    assert n["d_masked"] > 0 and n["d_hist"] > 0 and n["hist_slot_checked"] > 0, n
    assert not be.bad, be.bad[:3]
    assert not be.bad_slot, be.bad_slot[:3]
    assert not be.invalid, be.invalid[:3]


@pytest.mark.parametrize("kind", ["sc", "b2"])
def test_K2_wrong_key_is_caught_with_its_precondition(p3, ref, kind):
    """K2 (negative control, L34): key_plant = (head 0, position 0) adds 1 to the first verify row's first key at
    every layer call. PRECONDITION on the run's own plans: the planted entry lay inside num_pages at least once
    (key_plant_live > 0: + 1 then changes a listed key; a plant outside num_pages would be a no-op and is reported as
    such). Detection: KeyCheck records mismatches, and every mismatch is at a verify row, head 0 (local)."""
    toks, logits = ref
    eng = RT.make_res_engine(ops_with(key_only(p3)), kind=kind, backend_cls=KeyCheck)
    svc, be = eng.res_world.svc, eng.backend
    svc.key_plant = (0, 0)
    RT.tick_check(eng, toks, logits, forced=True)
    live = int(svc.key_plant_live)
    assert live > 0, "PRECONDITION: the plant never hit a listed key (a NO-OP plant, not a detection)"
    assert len(be.bad) > 0, "the wrong key was NOT caught (BLIND)"
    assert all(kd == "v" and h == 0 for (_, _, _, h, kd, _, _) in be.bad), be.bad[:3]
    assert len(be.bad) <= live, (len(be.bad), live)


@pytest.mark.parametrize("kind", ["sc", "b2"])
@pytest.mark.parametrize("mode", ["split", "whole"])
def test_K3_shared_attention_on_the_residency_plan(p3, ref, kind, mode):
    """K3 (the keys in use): the shared twin on the Residency plan's keys. (i) the PROBE (piece 3's output drives the
    tick; the shared twin on every keyed call): 0 differing rows, shared calls happened; (ii) the shared twin's
    output drives the tick: every verify row torch.equal to the serial row; (iii) the probe's plant (one union page
    dropped) is DETECTED on these plans (diff rows > 0; 0 = a no-op plant, L34)."""
    toks, logits = ref
    probe = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference")
    res = RT.tick_check(RT.make_res_engine(ops_with(probe), kind=kind), toks, logits, forced=True)
    rep = probe.report()
    assert probe.stats["shared"] > 0 and rep["rows_compared"] > 0, probe.stats
    assert rep["diff_rows"] == 0, rep["diff_rows"]
    assert res["compared"] > 50 and not res["bad"] and all(res["committed_ok"])
    used = sk.SharedKV(p3, mode=mode, impl="reference")
    res2 = RT.tick_check(RT.make_res_engine(ops_with(used), kind=kind), toks, logits, forced=True)
    assert used.stats["shared"] > 0, used.stats
    assert res2["compared"] > 50 and not res2["bad"] and all(res2["committed_ok"])
    plant = sk.SharedKVProbe(p3, mode=mode, timing=False, impl="reference", plant=True)
    RT.tick_check(RT.make_res_engine(ops_with(plant), kind=kind), toks, logits, forced=True)
    assert plant.report()["diff_rows"] > 0, "the plant was a NO-OP on the Residency plans (BLIND)"


def test_K4_refusals_and_the_default_path(p3, ref):
    """K4: v1 / S-B give no keys: a keyed tick refuses (TickError) at its first keyed layer, never guesses. Without
    needs_page_key the S-C plan carries no keys (want_keys False: the default path unchanged). A row whose tail does
    not name hist_bound, hist_bound + 1, ... is refused when keys are wanted."""
    toks, logits = ref
    for kind in ("v1", "sb"):
        eng = RT.make_res_engine(ops_with(key_only(p3)), kind=kind)
        with pytest.raises(TK.TickError, match="no page keys"):
            RT.tick_check(eng, toks, logits, forced=True)
    seen = []

    class Plain(TK.ResidencyBackend):
        def after_attention(self, l, plan):
            seen.append(plan.page_key)
            super().after_attention(l, plan)
    eng = RT.make_res_engine(RT.make_ops(p3), kind="sc", backend_cls=Plain)
    assert eng.res_world.svc.want_keys is False
    RT.tick_check(eng, toks, logits, forced=True)
    assert seen and all(k is None for k in seen)
    svc = RT.make_res_engine(ops_with(key_only(p3)), kind="sc").res_world.svc
    good = RES.Row(0, "v", (0, 0, 1), 9 * 64 + 5, 8, [(8, svc.tail0), (9, svc.tail0 + 1)])
    svc._check_tail_keys([good])
    for tail in ([(9, svc.tail0)], [(8, svc.tail0), (10, svc.tail0 + 1)]):
        with pytest.raises(ValueError, match="tail blocks"):
            svc._check_tail_keys([RES.Row(0, "v", (0, 0, 1), 9 * 64 + 5, 8, tail)])


def test_K5_the_key_path_has_no_host_transfer():
    """K5 (lint, as F12d / R5e): S-C's resolve, page_keys_of and _page_keys hold no host read of a device value and
    no host-to-device copy (_check_tail_keys reads host Row objects only and is called once per tick). Negative
    control: the scan finds a planted .item() / torch.tensor( ."""
    for fn in (RSC.ResidencySC.resolve, RSC.page_keys_of, RSC.ResidencySC._page_keys):
        src = inspect.getsource(fn)
        assert [tok for tok in FORBIDDEN + H2D if tok in src] == [], fn.__qualname__
    assert [tok for tok in FORBIDDEN + H2D if tok in "n = x.sum().item(); y = torch.tensor([n])"] != []
